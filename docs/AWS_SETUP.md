# AWS Setup (Minimal)

This is the minimum setup to run media-service workloads in AWS.

## 1) Core Services

- S3 bucket for media artifacts
- SQS queue for job messages
- Postgres (RDS or managed equivalent)
- ECS service(s) for API and worker

## 2) IAM

Create a least-privilege role for media-service with access to:
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on the media bucket
- `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:ChangeMessageVisibility`, `sqs:GetQueueAttributes` on queue
- logging permissions for CloudWatch logs

Avoid broad `*FullAccess` policies in production.

## 3) Environment Variables

At minimum configure:
- `DATABASE_URL`
- `AWS_REGION`
- `S3_BUCKET`
- `SQS_QUEUE_URL`
- `SQS_ENDPOINT` (only for localstack/local dev; omit in AWS)
- model/matcher settings required by the deployed pipeline

Use `.env.example` as the source for all required keys.

## 4) Deploy Shape

- API task (FastAPI)
- Worker task (`python -m app.worker`)
- Both point to same DB and queue

## 5) GPU Runtime Requirements (Critical)

LoFTR geometry is only fast enough in cloud when PyTorch runs on CUDA.

- Run worker on GPU-backed instances (for clustering/analysis jobs)
- If using pair-debug in production/staging, run API task on GPU too
- Build container with CUDA-enabled PyTorch (not CPU-only wheel)
- `WORKER_TYPE` does not select Torch backend; device is chosen by Torch runtime (`cuda -> mps -> cpu`)

Expected log signals after deploy:

- `Loaded LoFTR matcher (indoor) on cuda`
- `pair_debug_timing ... model_device=cuda tensor_device=cuda cuda_available=True preferred_device=cuda`

If logs show `model_device=cpu`, geometry inference will be much slower (seconds per pair).

## 6) Operational Checks

Before production rollout:
- run a full job in staging
- validate pair-debug endpoint for a known pair
- check `pair_debug_timing` logs and confirm matcher backend is CUDA
- compare clustering output against baseline (`scripts/baselines/README.md`)
- confirm alarms/logging for failed jobs and worker crashes

## 7) Future Consideration: SQS Retry + DLQ Policy (Not Implemented Yet)

Target behavior for robustness:
- If processing fails for any reason (photo download issue, transient service error, worker/server drop, unexpected exception), do not acknowledge the message.
- Allow exactly one retry.
- If the retry also fails, route the message to a Dead Letter Queue (DLQ).

Suggested SQS setup when this is implemented:
- Configure source queue redrive policy with `maxReceiveCount = 2` (initial attempt + one retry).
- Attach a dedicated DLQ for failed job messages.
- Add monitoring/alerts on DLQ depth and message age.

Note: this section is documentation-only for future implementation planning; no runtime behavior changes are included yet.
