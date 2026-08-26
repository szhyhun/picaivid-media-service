# AWS Setup

This repo now deploys a **VGGT-first** media worker. Finish the code migration first, then resume AWS work from this shape.

VGGT is the core direction. A possible future `tour_3d` or splat-delivery product is secondary and should not change the current priority: get the VGGT-powered cinematic pipeline working first.

## Current target

- region: `us-west-2`
- app host: `picaivid-app`
- db: `picaivid-db`
- queue: `picaivid-jobs`
- media bucket: `picaivid-prod-media`
- app domain: `picaivid.com`
- GPU worker: one `g6.xlarge` Spot for first validation, then scale up if needed

## Current state

Already done:

- app host booted and reachable
- Rails service running on `3000`
- React service running on `3001`
- nginx installed and configured
- DNS points to the Elastic IP
- HTTPS is active for `picaivid.com` and `www.picaivid.com`

Still pending:

- GPU worker launch
- first real VGGT CUDA validation
- first CUDA-backed worker validation
- GitHub Actions deploy automation

## Auth

Local:

```bash
aws login --profile picaivid-staging
aws sts get-caller-identity --profile picaivid-staging
export AWS_PROFILE=picaivid-staging
export AWS_REGION=us-west-2
```

Runtime:

- EC2 instance profiles only
- no static AWS keys in `/etc/picaivid/*.env`

## Session Manager

Install locally:

```bash
brew install --cask session-manager-plugin
```

Connect:

```bash
aws ssm start-session --target INSTANCE_ID --region "$AWS_REGION"
```

Fallback:

```bash
aws ssm send-command \
  --instance-ids INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters commands='["uname -a","whoami","pwd"]' \
  --region "$AWS_REGION"
```

## GPU lifecycle control

The worker is intended to be started and stopped on demand. Existing helper scripts:

- [scripts/aws/gpu.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu.sh)
- [scripts/aws/gpu-start.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu-start.sh)
- [scripts/aws/gpu-stop.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu-stop.sh)
- [scripts/aws/gpu-status.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu-status.sh)
- [docs/GPU_OPERATIONS.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/GPU_OPERATIONS.md)

They assume:

- the instance already exists
- `GPU_INSTANCE_ID` is set
- `AWS_PROFILE` and `AWS_REGION` are set if needed

Example:

```bash
export AWS_PROFILE=picaivid-admin
export AWS_REGION=us-west-2
export GPU_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx

./scripts/aws/gpu.sh start
./scripts/aws/gpu.sh status
./scripts/aws/gpu.sh stop
```

## Required media env values

- `DATABASE_URL`
- `AWS_REGION`
- `S3_BUCKET`
- `SQS_QUEUE_URL`
- `RAILS_WEBHOOK_URL`
- `WORKER_TYPE=gpu`
- `ANALYSIS_MATCH_ENGINE=vggt_scene_graph`
- `VGGT_REPO_DIR`
- `VGGT_MODEL_CHECKPOINT`

Optional S3 hydration:

- `VGGT_REPO_ARCHIVE_S3_URI`
- `VGGT_MODEL_CHECKPOINT_S3_URI`

## Artifact layout

Use S3 for hydrated runtime assets:

```text
s3://picaivid-prod-media/artifacts/
  vggt/
    repo/
    checkpoints/
```

Recommended instance paths:

```text
/srv/picaivid/third_party/vggt
/srv/picaivid/third_party/vggt/checkpoints/vggt_1B_commercial.pt
```

Checkpoint source:

- request access at [facebook/VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial)
- after approval, download with a Hugging Face read token or approved local login
- upload the approved checkpoint into `s3://picaivid-prod-media/artifacts/vggt/checkpoints/`
- hydrate it onto the GPU host before starting `picaivid-media-worker`

Reference only for a later splat-side project:

- [playcanvas/supersplat](https://github.com/playcanvas/supersplat)
- [playcanvas/model-viewer](https://github.com/playcanvas/model-viewer)

## App-first order

1. app EC2 up
2. Rails + React deployed
3. nginx + DNS + HTTPS working
4. media-service docs/env switched to VGGT
5. GPU worker launched
6. VGGT checkpoint hydrated
7. one small end-to-end job passes

At the time of writing, steps 1 through 4 and step 6 are complete.
