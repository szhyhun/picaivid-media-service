# AWS Setup

This guide covers the current AWS setup for the MASt3R cutover and the app-first deployment flow.

Current live-first shape:

- one `picaivid` AWS environment
- Rails + React on one EC2 app host
- PostgreSQL on RDS
- one shared SQS queue
- one shared S3 media bucket
- GPU worker added only after the app deployment is stable

If you later split this into separate staging and production environments, keep the same pattern and change the resource names.

## Target Shape

- Region: us-west-2
- S3 bucket: picaivid-prod-media or a dedicated staging bucket
- SQS queue: picaivid-jobs or a dedicated staging queue
- App host: t3a.small, 40 GB gp3
- GPU worker: g6.2xlarge Spot, 100-200 GB gp3
- GPU fallback: g5.xlarge Spot
- Avoid: g6f.*, g4ad.*, CPU-only workers
- Runtime management: EC2 + systemd + SSM

Use g6.2xlarge for staging because it keeps the same 24 GB NVIDIA L4 GPU memory as g6.xlarge, but adds CPU/RAM headroom for MASt3R image loading, retrieval graph construction, sparse global alignment, temp files, DB writes, and Python overhead.

## Services Needed

- S3 for uploaded photos, rendered videos, and model artifacts
- SQS for job dispatch
- PostgreSQL via RDS or a temporary staging DB on the app host
- EC2 app instance for Rails + React
- EC2 GPU worker instance for picaivid-media-worker
- IAM instance profiles for runtime AWS access
- SSM for shell access and start/stop commands without SSH
- CloudWatch Logs for operational visibility
- AWS Budgets for staging cost alerts

## Credentials

Local provisioning credentials live only on your workstation.

Preferred local auth:

    aws login --profile YOUR_PROFILE
    aws sts get-caller-identity --profile YOUR_PROFILE
    export AWS_PROFILE=YOUR_PROFILE
    export AWS_REGION=us-west-2

If your account uses IAM Identity Center instead of aws login:

    aws configure sso
    aws sso login --profile YOUR_PROFILE
    aws sts get-caller-identity --profile YOUR_PROFILE

The AWS CLI stores local credentials in:

- ~/.aws/config
- ~/.aws/credentials

Do not copy those files or values into the repo.

Runtime credentials must come from EC2 IAM instance profiles. Do not set AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in:

- /etc/picaivid/media-worker.env
- /etc/picaivid/media-api.env
- /etc/picaivid/rails.env
- /etc/picaivid/react.env

The repo contains only example env files. Real env files stay on the instance under /etc/picaivid/.

## SSM Access

Prefer AWS Systems Manager Session Manager over SSH.

Local requirement:

    brew install --cask session-manager-plugin

Start a shell on an instance:

    aws ssm start-session --target INSTANCE_ID_VALUE --region AWS_REGION_VALUE

If the plugin is unavailable, use non-interactive commands:

    aws ssm send-command \
      --instance-ids INSTANCE_ID_VALUE \
      --document-name AWS-RunShellScript \
      --parameters commands='["uname -a","whoami","pwd"]' \
      --region AWS_REGION_VALUE

Use SSM for:

- host inspection
- bootstrap commands
- service restarts
- deploy automation from GitHub Actions later

## Git Ignore and Secrets

Already ignored by repo gitignore rules:

- .env
- .env.local
- *.pt
- *.pth
- *.safetensors
- ml_models/

Never commit:

- /etc/picaivid/*.env
- ~/.aws/*
- real API keys or database passwords
- MASt3R checkpoints
- DINO/SAM2 model weights

## Artifact Layout

Use S3 as artifact storage and hydrate to local disk before service startup.

    s3://picaivid-staging-media/
      artifacts/
        mast3r/
        dinov3-vitb16-pretrain-lvd1689m/   # temporary until DINO/analyzer cleanup is complete
        sam2/                              # only if SAM2 remains enabled

Upload artifacts from the workstation:

    aws s3 sync /Users/serhiizhyhun/Desktop/projects/picaivid/third_party/mast3r s3://picaivid-staging-media/artifacts/mast3r
    aws s3 sync /Users/serhiizhyhun/Desktop/projects/picaivid/third_party/dinov3-vitb16-pretrain-lvd1689m s3://picaivid-staging-media/artifacts/dinov3-vitb16-pretrain-lvd1689m

Hydrate artifacts on the GPU instance:

    sudo mkdir -p /srv/picaivid/third_party
    aws s3 sync s3://picaivid-staging-media/artifacts/mast3r /srv/picaivid/third_party/mast3r
    aws s3 sync s3://picaivid-staging-media/artifacts/dinov3-vitb16-pretrain-lvd1689m /srv/picaivid/third_party/dinov3-vitb16-pretrain-lvd1689m

Required MASt3R files after hydration:

    /srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    /srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
    /srv/picaivid/third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

Remote model download is disabled. If an artifact is missing, fix hydration instead of falling back to the internet.

## Resource CLI Checks

List services/resources:

    aws s3 ls
    aws sqs list-queues --region AWS_REGION_VALUE
    aws rds describe-db-instances --region AWS_REGION_VALUE --output table
    aws ec2 describe-instances --filters Name=tag:Project,Values=picaivid --region AWS_REGION_VALUE --output table
    aws iam list-instance-profiles --query 'InstanceProfiles[].InstanceProfileName' --output table
    aws logs describe-log-groups --log-group-name-prefix /picaivid --region AWS_REGION_VALUE --output table

Create bucket and queue if missing:

    aws s3 mb s3://BUCKET_NAME_VALUE --region AWS_REGION_VALUE
    aws sqs create-queue --queue-name QUEUE_NAME_VALUE --attributes VisibilityTimeout=3600 --region AWS_REGION_VALUE

Check GPU instance type availability:

    aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=g6.2xlarge,g5.xlarge --region AWS_REGION_VALUE --output table

## Instance Launch Console Path

App host:

- EC2 -> Launch instance
- Instance type: t3a.small
- Storage: 40 GB gp3
- IAM instance profile: app role with SSM access
- Tags: Project=picaivid, Env=staging, Role=app

GPU worker:

- EC2 -> Launch instance
- Instance type: g6.2xlarge Spot
- Fallback type: g5.xlarge Spot
- Storage: 100-200 GB gp3
- IAM instance profile: media role with S3, SQS, SSM, and CloudWatch access
- Tags: Project=picaivid, Env=staging, Role=media-gpu, AutoStop=true

Use an NVIDIA/CUDA-ready AMI where possible. Otherwise run the repo bootstrap and explicitly install CUDA-enabled PyTorch.

## Instance Management

Use SSM:

    aws ssm start-session --target GPU_INSTANCE_ID_VALUE --region AWS_REGION_VALUE

Use the existing scripts:

    export AWS_PROFILE=YOUR_PROFILE
    export AWS_REGION=us-west-2
    export GPU_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx
    ./scripts/aws/gpu-status.sh
    ./scripts/aws/gpu-start.sh
    ./scripts/aws/gpu-stop.sh

The start script starts the EC2 instance and best-effort starts picaivid-media-worker through SSM. The stop script best-effort stops the worker, then stops the instance.

## App-First Deployment Order

Deploy the app host before launching the GPU worker.

1. Start the app EC2 instance.
2. Attach an Elastic IP.
3. Open an SSM shell.
4. Clone or update `picaivid-rails` and `picaivid-react` under `/srv/picaivid/`.
5. Run each repo bootstrap script.
6. Create and fill:
   - `/etc/picaivid/rails.env`
   - `/etc/picaivid/react.env`
7. Point Rails at:
   - `DATABASE_URL=postgresql://...@picaivid-db.../picaivid`
   - `SQS_QUEUE_URL=https://sqs.us-west-2.amazonaws.com/ACCOUNT_ID/picaivid-jobs`
   - `AWS_REGION=us-west-2`
   - `AWS_S3_BUCKET=picaivid-prod-media` or your chosen bucket
8. Run Rails migrations.
9. Start `picaivid-rails` and `picaivid-react`.
10. Install nginx and proxy:
    - `/` to React
    - `/api` to Rails
11. Point the domain to the app Elastic IP.
12. Add HTTPS with Let's Encrypt.

Do not launch the GPU worker until the app host is healthy and reachable through the domain.

## GitHub Actions Follow-Up

After the first manual deploy works, add CI/CD:

- trigger on push to `master`
- authenticate to AWS with GitHub OIDC
- use SSM `send-command` to deploy to the app host
- run migrations and restart services
- extend the same pattern to the GPU worker later

Do not store long-lived AWS keys in GitHub secrets for deployment.

## Worker Env

Use /etc/picaivid/media-worker.env from deploy/env/media-worker.env.example.

Important AWS staging values:

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

Do not set in AWS:

    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    S3_ENDPOINT
    SQS_ENDPOINT

## Monitoring

Queue depth:

    aws sqs get-queue-attributes --queue-url SQS_QUEUE_URL_VALUE --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible ApproximateAgeOfOldestMessage --region AWS_REGION_VALUE

Worker logs and CUDA:

    sudo systemctl status picaivid-media-worker
    journalctl -u picaivid-media-worker -f
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Expected media-service logs:

- Loaded MASt3R model checkpoint=... device=cuda
- Loaded MASt3R retriever checkpoint=...
- match_engine=mast3r_graph

## Validation Checklist

- aws sts get-caller-identity works locally
- staging S3 bucket exists
- staging SQS queue URL resolves
- EC2 instance role can read S3 and consume SQS without static keys
- SSM session works
- nvidia-smi works
- CUDA Torch check returns True
- one small staging job completes
- photo_similarities.match_engine = mast3r_graph
- photo_pose_alignments rows are written
- final clusters are max 2 photos
- same-component photos appear only as debug suggestions
