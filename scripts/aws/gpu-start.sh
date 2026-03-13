#!/usr/bin/env bash
set -euo pipefail

: "${GPU_INSTANCE_ID:?GPU_INSTANCE_ID is required}"
AWS_REGION="${AWS_REGION:-us-west-2}"

profile_args=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  profile_args=(--profile "${AWS_PROFILE}")
fi

echo "Starting GPU instance ${GPU_INSTANCE_ID} in ${AWS_REGION}..."
aws "${profile_args[@]}" ec2 start-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}" \
  >/dev/null

aws "${profile_args[@]}" ec2 wait instance-running \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}"

echo "Instance is running."
echo "Starting media worker service via SSM (if SSM agent/role are configured)..."
aws "${profile_args[@]}" ssm send-command \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}" \
  --document-name "AWS-RunShellScript" \
  --comment "Start picaivid media worker" \
  --parameters 'commands=["sudo systemctl start picaivid-media-worker"]' \
  >/dev/null || true

echo "Done."
