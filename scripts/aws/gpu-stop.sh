#!/usr/bin/env bash
set -euo pipefail

: "${GPU_INSTANCE_ID:?GPU_INSTANCE_ID is required}"
AWS_REGION="${AWS_REGION:-us-east-1}"

profile_args=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  profile_args=(--profile "${AWS_PROFILE}")
fi

echo "Stopping media worker service via SSM (best effort)..."
aws "${profile_args[@]}" ssm send-command \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}" \
  --document-name "AWS-RunShellScript" \
  --comment "Stop picaivid media worker" \
  --parameters 'commands=["sudo systemctl stop picaivid-media-worker"]' \
  >/dev/null || true

echo "Stopping GPU instance ${GPU_INSTANCE_ID} in ${AWS_REGION}..."
aws "${profile_args[@]}" ec2 stop-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}" \
  >/dev/null

aws "${profile_args[@]}" ec2 wait instance-stopped \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}"

echo "Instance is stopped."
