# AWS Setup

This repo now deploys a **VGGT-first** media worker. Finish the code migration first, then resume AWS work from this shape.

## Current target

- region: `us-west-2`
- app host: `picaivid-app`
- db: `picaivid-db`
- queue: `picaivid-jobs`
- media bucket: `picaivid-prod-media`
- app domain: `picaivid.com`
- GPU worker: one `g6.2xlarge` Spot by default

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
/srv/picaivid/third_party/vggt/checkpoints/VGGT-1B-Commercial.safetensors
```

## App-first order

1. app EC2 up
2. Rails + React deployed
3. nginx + DNS + HTTPS working
4. media-service docs/env switched to VGGT
5. GPU worker launched
6. VGGT checkpoint hydrated
7. one small end-to-end job passes
