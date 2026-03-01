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

## 5) Operational Checks

Before production rollout:
- run a full job in staging
- validate pair-debug endpoint for a known pair
- compare clustering output against baseline (`scripts/baselines/README.md`)
- confirm alarms/logging for failed jobs and worker crashes
