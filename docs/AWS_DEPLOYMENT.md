# AWS Deployment

Operational order for the workspace:

1. finish the VGGT code migration locally
2. validate migrations and debug APIs locally
3. finish app-host deployment (`picaivid-app`, Rails, React, nginx, TLS)
4. launch the GPU worker
5. hydrate VGGT repo/checkpoint to local disk
6. start `picaivid-media-worker`
7. run one small reconstruction job
8. add GitHub Actions OIDC + SSM deploy automation

## Current status

Completed:

- VGGT migration is merged
- local Alembic upgrade succeeded through `3f4e5d6c7b8a`
- `picaivid.com` and `www.picaivid.com` point to the app EC2 Elastic IP
- Rails and React are running on `picaivid-app`
- nginx is installed and proxies:
  - `/` -> React on `127.0.0.1:3001`
  - `/api` -> Rails on `127.0.0.1:3000`
- Let's Encrypt TLS is live for:
  - `https://picaivid.com`
  - `https://www.picaivid.com`

Pending:

- launch the GPU worker
- run the first real CUDA-backed end-to-end job
- add GitHub Actions OIDC + SSM deploy automation

## App host

The app host serves Rails + React and can optionally run the media API for scene/relation debug. Media-to-Rails traffic should stay inside the VPC once the GPU worker exists.

Current public entrypoints:

- `https://picaivid.com`
- `https://www.picaivid.com`

## GPU worker

Use:

- `g6.xlarge` Spot first for quota-compatible validation
- `g6.2xlarge` only after quota allows it or if validation proves `g6.xlarge` too small
- avoid `g4dn.xlarge` unless forced; the T4 is a weaker fit for VGGT than the L4

The worker needs:

- CUDA
- the VGGT commercial checkpoint on disk
- repo bootstrap complete
- `/etc/picaivid/media-worker.env`

The checkpoint and repo archive are already staged in S3. The remaining work is instance launch, hydration onto disk, and first real validation.

Use the existing lifecycle helpers after the instance exists:

- [gpu.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu.sh)
- [gpu-start.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu-start.sh)
- [gpu-stop.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu-stop.sh)
- [gpu-status.sh](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/scripts/aws/gpu-status.sh)
- [GPU_OPERATIONS.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/GPU_OPERATIONS.md)

## Bootstrap checklist

On the GPU host:

```bash
cd /srv/picaivid/picaivid-media-service
./scripts/aws/bootstrap-ec2.sh --repo-dir /srv/picaivid/picaivid-media-service --enable-api 0
```

Hydrate assets:

```bash
sudo mkdir -p /srv/picaivid/third_party/vggt/checkpoints
aws s3 sync s3://picaivid-prod-media/artifacts/vggt/repo /srv/picaivid/third_party/vggt
aws s3 sync s3://picaivid-prod-media/artifacts/vggt/checkpoints /srv/picaivid/third_party/vggt/checkpoints
```

Start services:

```bash
sudo systemctl restart picaivid-media-worker
sudo systemctl status picaivid-media-worker
journalctl -u picaivid-media-worker -f
```

Validate:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## GitHub Actions follow-up

After the manual deploy is stable, CD should do:

1. assume AWS role via OIDC
2. package a repo artifact in GitHub Actions
3. upload it to S3
4. deploy on-host via SSM:
   - download artifact from S3
   - unpack into a fresh release dir
   - install/build on the host
   - swap the working directory
   - restart only the affected service

This is intentionally minimal:

- no host-side GitHub credentials
- no Rails specs in Actions for now
- no Docker-in-Actions
- one OIDC role
- one S3 artifact bucket
- one SSM deploy path per repo

### Required GitHub configuration

Secret in each repo:

- `AWS_DEPLOY_ROLE_ARN`

Repository variables:

- `AWS_REGION`
- `DEPLOY_ARTIFACT_BUCKET`
- `DEPLOY_ARTIFACT_PREFIX`

React repo variables:

- `EC2_APP_INSTANCE_ID`
- `REACT_DEPLOY_PATH`

Rails repo variables:

- `EC2_APP_INSTANCE_ID`
- `RAILS_DEPLOY_PATH`

Media-service repo variables:

- `EC2_MEDIA_INSTANCE_ID`
- `MEDIA_DEPLOY_PATH`
- `MEDIA_RESTART_API` (`0` or `1`)
- `MEDIA_INSTALL_CUDA_TORCH` (`0` or `1`)

Recommended values today:

- `DEPLOY_ARTIFACT_BUCKET=picaivid-prod-media`
- `DEPLOY_ARTIFACT_PREFIX=deploy-artifacts`
- `REACT_DEPLOY_PATH=/srv/picaivid/picaivid-react`
- `RAILS_DEPLOY_PATH=/srv/picaivid/picaivid-rails`
- `MEDIA_DEPLOY_PATH=/srv/picaivid/picaivid-media-service`

Keep the first automated workflow simple. It should deploy only the repo that changed.
