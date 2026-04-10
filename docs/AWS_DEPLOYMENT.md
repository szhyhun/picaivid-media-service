# AWS Deployment: MASt3R Staging

This guide is the operational runbook for deploying the current MASt3R media-service staging setup on AWS.

Scope:

- `picaivid-rails` API/orchestration
- `picaivid-react` UI
- `picaivid-media-service` API/worker

## 1) Target Architecture

Staging defaults:

- Region: `us-west-2`
- S3 bucket: `picaivid-staging-media`
- SQS queue: `picaivid-staging-jobs`
- App instance: `t3a.small`, 40 GB `gp3`
- GPU worker: `g6.2xlarge` Spot, 100-200 GB `gp3`
- GPU fallback: `g5.xlarge` Spot
- Avoid: `g6f.*`, `g4ad.*`, CPU-only workers
- Deployment model: EC2 + systemd + SSM
- Initial GPU scale: one worker instance
- Max staging GPU scale: four separate worker instances after the first worker is stable

The app host runs Rails + React. The GPU host runs `picaivid-media-worker`; the media API can run on the app host only if pair-debug must be available while the GPU worker is stopped.

## 2) Instance Sizing

### App Host

Use `t3a.small` for staging by default.

Use `t3a.medium` only if Rails + React + optional staging Postgres are memory constrained.

### GPU Worker

Use `g6.2xlarge` Spot first.

Reasoning:

- `g6.xlarge` and `g6.2xlarge` both provide one NVIDIA L4 with 24 GB GPU memory.
- `g6.2xlarge` has better CPU/RAM headroom for image loading, MASt3R retrieval graph construction, sparse global alignment, temp files, DB writes, and Python overhead.
- Staging cost should be controlled by stopping GPU workers when idle, not by under-sizing the MASt3R worker.

Use `g5.xlarge` Spot only as a fallback when `g6.2xlarge` is unavailable.

Check availability:

```bash
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g6.2xlarge,g5.xlarge \
  --region "$AWS_REGION" \
  --output table
```

## 3) Credentials and IAM

Local provisioning credentials:

```bash
aws configure sso
aws sts get-caller-identity --profile YOUR_PROFILE
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=us-west-2
```

AWS CLI stores these locally in:

- `~/.aws/config`
- `~/.aws/credentials`

Do not copy those values into the repo.

Runtime credentials:

- Use EC2 IAM instance profiles.
- Do not set `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` in `/etc/picaivid/*.env`.
- The media-service boto clients use the instance role when static credentials are absent.
- `S3_ENDPOINT` and `SQS_ENDPOINT` are local-only for MinIO/LocalStack and should be omitted on AWS.

Suggested runtime roles:

- App role: SSM, CloudWatch Logs, S3 read/write for media bucket as needed.
- Media role: SSM, CloudWatch Logs, S3 artifact/media read/write, SQS consume/delete/change-visibility/send as needed.

## 4) Core Resources

Create the staging bucket and queue if missing:

```bash
aws s3 mb s3://picaivid-staging-media --region "$AWS_REGION"

aws sqs create-queue \
  --queue-name picaivid-staging-jobs \
  --attributes VisibilityTimeout=3600 \
  --region "$AWS_REGION"
```

List resources:

```bash
aws s3 ls
aws sqs list-queues --region "$AWS_REGION"
aws rds describe-db-instances --region "$AWS_REGION" --output table
aws ec2 describe-instances --filters "Name=tag:Project,Values=picaivid" --region "$AWS_REGION" --output table
aws iam list-instance-profiles --query 'InstanceProfiles[].InstanceProfileName' --output table
aws logs describe-log-groups --log-group-name-prefix /picaivid --region "$AWS_REGION" --output table
```

Use RDS PostgreSQL if available. For early staging only, a DB on the app host is acceptable temporarily.

## 5) Artifact Storage

MASt3R is the phase-1 matching/reconstruction engine. Runtime must load local or S3-hydrated artifacts; live model downloads are not a deployment path.

S3 artifact prefixes:

- `s3://picaivid-staging-media/artifacts/mast3r`
- `s3://picaivid-staging-media/artifacts/dinov3-vitb16-pretrain-lvd1689m` temporarily, until DINO analyzer dependencies are fully removed
- `s3://picaivid-staging-media/artifacts/sam2` only if SAM2 remains enabled

Upload artifacts:

```bash
aws s3 sync /Users/serhiizhyhun/Desktop/projects/picaivid/third_party/mast3r \
  s3://picaivid-staging-media/artifacts/mast3r

aws s3 sync /Users/serhiizhyhun/Desktop/projects/picaivid/third_party/dinov3-vitb16-pretrain-lvd1689m \
  s3://picaivid-staging-media/artifacts/dinov3-vitb16-pretrain-lvd1689m
```

Hydrate artifacts on the GPU instance:

```bash
sudo mkdir -p /srv/picaivid/third_party
aws s3 sync s3://picaivid-staging-media/artifacts/mast3r /srv/picaivid/third_party/mast3r
aws s3 sync s3://picaivid-staging-media/artifacts/dinov3-vitb16-pretrain-lvd1689m /srv/picaivid/third_party/dinov3-vitb16-pretrain-lvd1689m
```

Required MASt3R files:

- `/srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- `/srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`
- `/srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl`

## 6) Launch Instances

Console path:

- EC2 -> Launch instance
- App instance type: `t3a.small`
- GPU instance type: `g6.2xlarge` Spot
- GPU fallback type: `g5.xlarge` Spot
- App storage: 40 GB `gp3`
- GPU storage: 100-200 GB `gp3`
- IAM instance profile: app role or media role
- Tags: `Project=picaivid`, `Env=staging`, `Role=app` or `Role=media-gpu`

Prefer SSM Session Manager over SSH. Use an NVIDIA/CUDA-ready AMI for the GPU host where possible. Otherwise bootstrap CUDA-enabled PyTorch explicitly and validate CUDA before starting the worker.

CLI launch commands should be added after AMI, subnet, security group, IAM instance profile, and SSM/SSH preference are chosen.

## 7) Instance Env Files

Real env files live only on instances:

- `/etc/picaivid/rails.env`
- `/etc/picaivid/react.env`
- `/etc/picaivid/media-api.env`
- `/etc/picaivid/media-worker.env`

Repo files remain examples only:

- `deploy/env/media-api.env.example`
- `deploy/env/media-worker.env.example`

Important media worker values:

```bash
WORKER_TYPE=gpu
ANALYSIS_MATCH_ENGINE=mast3r_graph
AWS_REGION=us-west-2
S3_BUCKET=picaivid-staging-media
SQS_QUEUE_URL=https://sqs.us-west-2.amazonaws.com/ACCOUNT_ID/picaivid-staging-jobs
MODEL_CACHE_DIR=/srv/picaivid/model-cache
MAST3R_REPO_DIR=/srv/picaivid/third_party/mast3r
MAST3R_MODEL_CHECKPOINT=/srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
MAST3R_RETRIEVAL_CHECKPOINT=/srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
MAST3R_RETRIEVAL_CODEBOOK=/srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
```

Do not set on AWS:

```bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
S3_ENDPOINT
SQS_ENDPOINT
```

## 8) Systemd Services

Use these systemd units:

- `picaivid-rails.service`
- `picaivid-react.service`
- `picaivid-media-api.service` optional
- `picaivid-media-worker.service` on the GPU host

Install media worker unit:

```bash
sudo mkdir -p /etc/picaivid
sudo cp /srv/picaivid/picaivid-media-service/deploy/systemd/picaivid-media-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picaivid-media-worker
sudo systemctl start picaivid-media-worker
```

Check worker:

```bash
sudo systemctl status picaivid-media-worker
journalctl -u picaivid-media-worker -f
```

## 9) GPU Runtime Validation

Install CUDA-enabled Torch on the GPU host if the AMI does not already provide a working CUDA PyTorch environment:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

Expected service signals:

- `Loaded MASt3R model checkpoint=... device=cuda`
- `Loaded MASt3R retriever checkpoint=...`
- `match_engine=mast3r_graph`
- no LoFTR/DINO production matcher path

## 10) Start/Stop GPU to Save Cost

Use the existing scripts from `scripts/aws/`:

```bash
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=us-west-2
export GPU_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx
./scripts/aws/gpu-status.sh
./scripts/aws/gpu-start.sh
./scripts/aws/gpu-stop.sh
```

The start script starts the EC2 instance and best-effort starts `picaivid-media-worker` via SSM. The stop script best-effort stops the worker, then stops the instance.

Use SSM shell access:

```bash
aws ssm start-session --target "$GPU_INSTANCE_ID" --region "$AWS_REGION"
```

## 11) Monitoring

Queue depth:

```bash
aws sqs get-queue-attributes \
  --queue-url "$SQS_QUEUE_URL" \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible ApproximateAgeOfOldestMessage \
  --region "$AWS_REGION"
```

Worker health:

```bash
sudo systemctl status picaivid-media-worker
journalctl -u picaivid-media-worker -f
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

CloudWatch:

```bash
aws logs describe-log-groups --log-group-name-prefix /picaivid --region "$AWS_REGION" --output table
```

Cost controls:

- Create an AWS Budget for staging.
- Stop GPU instances when idle.
- Start with one GPU instance only.
- Scale to 2-4 GPU workers only after the first worker completes staging jobs reliably.

## 12) Validation Checklist

- `aws sts get-caller-identity` succeeds locally.
- Staging S3 bucket exists.
- Staging SQS queue URL resolves.
- App and media EC2 instances use IAM instance profiles.
- No static AWS keys are present in `/etc/picaivid/*.env`.
- SSM session works.
- `nvidia-smi` works on the GPU host.
- CUDA Torch check returns `True`.
- MASt3R artifacts are present on local disk after S3 hydration.
- One small staging job completes.
- `photo_similarities.match_engine = mast3r_graph`.
- `photo_pose_alignments` rows are written.
- Final clusters are max 2 photos.
- Same-component photos appear only as debug suggestions.

## 13) Deferred Cleanup

After the MASt3R staging path is verified, remove legacy deployment/config surface for:

- DINO analyzer dependencies that are no longer used
- LoFTR
- MatchFormer
- RoMa
- stale CLIP/SAM acceptance logic if it remains in active deployment config

Until that cleanup is complete, DINO may remain as a temporary artifact entry only; it is not a production fallback matcher.
