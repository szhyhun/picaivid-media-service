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
- hydrate the VGGT commercial checkpoint after Hugging Face approval
- run the first real CUDA-backed end-to-end job
- add GitHub Actions OIDC + SSM deploy automation

## App host

The app host serves Rails + React and can optionally run the media API for scene/relation debug. Media-to-Rails traffic should stay inside the VPC once the GPU worker exists.

Current public entrypoints:

- `https://picaivid.com`
- `https://www.picaivid.com`

## GPU worker

Use:

- `g6.2xlarge` Spot first
- `g5.xlarge` Spot fallback

The worker needs:

- CUDA
- the VGGT commercial checkpoint on disk
- repo bootstrap complete
- `/etc/picaivid/media-worker.env`
- Hugging Face access approval for `facebook/VGGT-1B-Commercial`

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
2. deploy app host via SSM:
   - pull code
   - install deps
   - `bundle install`
   - `bundle exec rails db:migrate`
   - `npm ci`
   - `npm run build`
   - restart Rails/React services
3. deploy media worker via SSM:
   - pull code
   - install Python deps
   - validate VGGT assets
   - restart worker

Keep the first automated workflow simple. It should restart only services whose repo changed.
