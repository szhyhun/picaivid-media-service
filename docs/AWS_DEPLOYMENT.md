# AWS Deployment (Cost-First, Simple CD)

This guide is the practical path for running all 3 apps on AWS with the lowest operational complexity and controlled cost.

Scope:
- `picaivid-rails` (API/orchestration)
- `picaivid-react` (UI)
- `picaivid-media-service` (GPU-heavy matching/processing)

## 1) Recommended Architecture

- Region: pick one and keep everything there. Examples below use the active production region.
- S3: one bucket for photos/renders/artifacts.
- SQS: one main queue for jobs (DLQ can be added next).
- EC2 app instance (small, always on):
  - Runs Rails API + React app (+ optional Postgres if minimizing cost).
- EC2 GPU instance (Spot, started only when needed):
  - Runs media-service worker.
  - Optionally runs media-service API for pair-debug sessions.

This is simpler and cheaper than ECS/Kubernetes for your current stage.

## 2) Cost Targets and Instance Choice

### Rails + React

Use x86 burstable first for compatibility:
- `t3a.small` or `t3a.medium`.
- Verify current price in your chosen region before launch.

If you want Graviton later, migrate to `t4g.*` after confirming gem/node native deps.

### Media GPU

Use NVIDIA CUDA-capable family:
- `g4dn.xlarge` Spot (primary recommendation).

Important:
- CUDA-compatible NVIDIA is the requirement for your LoFTR workload.
- `g4ad` is cheaper on-demand in AWS tables, but AMD GPU (no CUDA), so not suitable.
- To stay under `$0.50/hr`, run GPU as Spot and stop when idle.

Check live Spot price before launch:

```bash
aws ec2 describe-spot-price-history \
  --instance-types g4dn.xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --region "${AWS_REGION:-us-west-2}" \
  --max-items 20 \
  --query 'SpotPriceHistory[].{AZ:AvailabilityZone,Price:SpotPrice,Time:Timestamp}' \
  --output table
```

## 3) Account Setup (Day 0)

1. Create AWS account.
2. Secure root user immediately:
   - enable MFA
   - do not use root for daily work
3. Create IAM admin user/role for yourself.
4. Configure AWS Budgets alerts:
   - monthly total budget (example `$150`)
   - alert at 50%, 80%, 100%
5. Choose one region and use it consistently.

## 4) Core AWS Resources

## 4.1 S3

Create bucket, example:
- `picaivid-prod-media`

Recommended:
- keep versioning off unless you explicitly need rollback/delete protection
- add lifecycle policies once object growth becomes material

## 4.2 SQS

Create queue:
- `picaivid-jobs`

For now:
- standard queue
- visibility timeout aligned with job duration

Next step (recommended soon):
- attach DLQ with `maxReceiveCount=2` (initial + one retry)

## 4.3 EC2 Instances

Create:
- `picaivid-app` (`t3a.small` or `t3a.medium`, on-demand)
- `picaivid-media-gpu` (`g4dn.xlarge`, Spot)

Tags:
- `Project=picaivid`
- `Env=prod`
- `Role=app` / `Role=media-gpu`
- `AutoStop=true` (if using scheduler)

## 5) Networking and Transfer Cost Rules

To keep transfer cost low:
- Keep EC2 + S3 + SQS in the same region.
- Upload once to S3 and process from S3 inside AWS.
- Avoid cross-region copies.
- Serve final assets via CloudFront only when needed.

Cost facts from AWS docs:
- S3 -> AWS service in same region is not charged as internet transfer.
- First 100 GB/month internet egress is free aggregate across eligible services.

## 6) Media-Service GPU Runtime (Critical)

Do not rely on CPU Torch wheels on GPU hosts.

Install CUDA-enabled torch explicitly on the GPU instance/container:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected runtime logs:
- `Loaded LoFTR matcher (indoor) on cuda`
- `model_device=cuda`

If logs show CPU, performance will be dramatically worse.

## 7) Deployment Model (Simple CD)

Use GitHub Actions + AWS OIDC + SSM Run Command.

Why this path:
- no static AWS keys in GitHub
- no inbound SSH needed
- simple restart flow per push

Flow:
1. GitHub Action assumes AWS IAM role via OIDC.
2. Action triggers SSM command on target EC2.
3. Instance runs:
   - `git pull`
   - dependency sync
   - service restart (`systemd`)

Keep separate workflows per repo:
- rails deploy -> app instance
- react deploy -> app instance
- media-service deploy -> gpu instance (or both app+gpu if API also there)

Included workflow files:
- `picaivid-rails/.github/workflows/deploy.yml`
- `picaivid-react/.github/workflows/deploy.yml`
- `picaivid-media-service/.github/workflows/deploy.yml`

### 7.1 GitHub OIDC + IAM (one-time)

Create one IAM role trusted by GitHub OIDC provider for each repo environment (or one shared role if you prefer).

Trust policy must allow:
- provider: `token.actions.githubusercontent.com`
- audience: `sts.amazonaws.com`
- subject restricted to your repo (example `repo:ORG/REPO:*`)

Minimum IAM permissions for the deploy role:
- `ssm:SendCommand`
- `ssm:GetCommandInvocation`
- `ssm:ListCommandInvocations`
- `ec2:DescribeInstances`

The EC2 instances themselves must have instance profile permission:
- `AmazonSSMManagedInstanceCore`

### 7.2 Repository Secrets/Variables

Each repo needs:

- Secret:
  - `AWS_DEPLOY_ROLE_ARN`

- Variables:
  - `AWS_REGION`

Rails repo (`picaivid-rails`) variables:
- `EC2_APP_INSTANCE_ID`
- `RAILS_DEPLOY_PATH` (example `/srv/picaivid/picaivid-rails`)

React repo (`picaivid-react`) variables:
- `EC2_APP_INSTANCE_ID`
- `REACT_DEPLOY_PATH` (example `/srv/picaivid/picaivid-react`)

Media repo (`picaivid-media-service`) variables:
- `EC2_MEDIA_INSTANCE_ID`
- `MEDIA_DEPLOY_PATH` (example `/srv/picaivid/picaivid-media-service`)
- `MEDIA_RESTART_API` (`0` or `1`)

## 8) Service Management on Instances

Use `systemd` units:
- `picaivid-rails.service`
- `picaivid-react.service`
- `picaivid-media-api.service` (optional on app host)
- `picaivid-media-worker.service` (on GPU host)

For GPU host, keep worker disabled by default and start only when required.

Included `systemd` templates:
- Rails: `picaivid-rails/deploy/systemd/picaivid-rails.service`
- React: `picaivid-react/deploy/systemd/picaivid-react.service`
- Media API: `deploy/systemd/picaivid-media-api.service`
- Media worker: `deploy/systemd/picaivid-media-worker.service`

### 8.1 Install `systemd` units (one-time per host)

App host (Rails + React):

```bash
sudo mkdir -p /etc/picaivid
sudo cp /srv/picaivid/picaivid-rails/deploy/systemd/picaivid-rails.service /etc/systemd/system/
sudo cp /srv/picaivid/picaivid-react/deploy/systemd/picaivid-react.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picaivid-rails picaivid-react
sudo systemctl start picaivid-rails picaivid-react
```

GPU host (worker, optional API):

```bash
sudo mkdir -p /etc/picaivid
sudo cp /srv/picaivid/picaivid-media-service/deploy/systemd/picaivid-media-worker.service /etc/systemd/system/
sudo cp /srv/picaivid/picaivid-media-service/deploy/systemd/picaivid-media-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picaivid-media-worker
sudo systemctl start picaivid-media-worker
```

Create env files referenced by units:
- `/etc/picaivid/rails.env`
- `/etc/picaivid/react.env`
- `/etc/picaivid/media-api.env`
- `/etc/picaivid/media-worker.env`

Templates provided:
- `picaivid-rails/deploy/env/rails.env.example`
- `picaivid-react/deploy/env/react.env.example`
- `deploy/env/media-api.env.example`
- `deploy/env/media-worker.env.example`

## 8.2 One Bootstrap Script Per Repo

Use repo-local bootstrap scripts instead of manual package/service steps:

App host:

```bash
cd /srv/picaivid/picaivid-rails
chmod +x scripts/aws/bootstrap-ec2.sh
./scripts/aws/bootstrap-ec2.sh --repo-dir /srv/picaivid/picaivid-rails --run-migrate 1

cd /srv/picaivid/picaivid-react
chmod +x scripts/aws/bootstrap-ec2.sh
./scripts/aws/bootstrap-ec2.sh --repo-dir /srv/picaivid/picaivid-react
```

GPU host:

```bash
cd /srv/picaivid/picaivid-media-service
chmod +x scripts/aws/bootstrap-ec2.sh
./scripts/aws/bootstrap-ec2.sh --repo-dir /srv/picaivid/picaivid-media-service --enable-api 0
```

Scripts:
- `picaivid-rails/scripts/aws/bootstrap-ec2.sh`
- `picaivid-react/scripts/aws/bootstrap-ec2.sh`
- `scripts/aws/bootstrap-ec2.sh`

## 9) Start/Stop GPU to Save Cost

Use provided scripts in `scripts/aws/`:
- `scripts/aws/gpu-start.sh`
- `scripts/aws/gpu-stop.sh`
- `scripts/aws/gpu-status.sh`

Usage:

```bash
export AWS_REGION=us-west-2
export GPU_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx
./scripts/aws/gpu-start.sh
./scripts/aws/gpu-status.sh
./scripts/aws/gpu-stop.sh
```

Optional automation:
- Systems Manager Quick Setup can schedule stop/start windows.

## 10) Minimal Rollout Plan

1. Create account + IAM + budgets.
2. Create S3 + SQS.
3. Launch app EC2 and deploy Rails/React.
4. Launch GPU Spot EC2 and deploy media worker.
5. Verify media logs show CUDA backend.
6. Add GitHub Actions OIDC + SSM deploy jobs.
7. Add GPU stop/start operational routine (manual first, then scheduler).
8. Add DLQ + one retry policy in SQS consumer rollout.

## 11) What to Watch (Cost + Reliability)

- Cost Explorer:
  - EC2-Instances
  - DataTransfer
  - S3-Requests and S3-Storage
- CloudWatch:
  - queue depth
  - worker failures
  - spot interruption events
- Spot interruptions:
  - handle graceful shutdown and idempotent jobs

---

## Sources

- AWS root user security best practices:
  - https://docs.aws.amazon.com/IAM/latest/UserGuide/root-user-best-practices.html
- AWS Budgets:
  - https://docs.aws.amazon.com/cost-management/latest/userguide/create-cost-budget.html
- EC2 on-demand/data transfer pricing:
  - https://aws.amazon.com/ec2/pricing/on-demand/
- EC2 T3 instance pricing table (reference):
  - https://aws.amazon.com/ec2/instance-types/t3/
- EC2 G4 family overview (CUDA-capable G4dn, AMD-based G4ad table):
  - https://aws.amazon.com/ec2/instance-types/g4/
- S3 pricing and transfer notes:
  - https://aws.amazon.com/s3/pricing/
- Spot interruption notices:
  - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html
- Spot best practices:
  - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html
- SQS DLQ:
  - https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html
- EC2 start/stop scheduling with Systems Manager Quick Setup:
  - https://docs.aws.amazon.com/systems-manager/latest/userguide/quick-setup-scheduler.html
- GitHub OIDC to AWS:
  - https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
  - https://github.com/aws-actions/configure-aws-credentials
