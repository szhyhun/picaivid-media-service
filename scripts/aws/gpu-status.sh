#!/usr/bin/env bash
set -euo pipefail

: "${GPU_INSTANCE_ID:?GPU_INSTANCE_ID is required}"
AWS_REGION="${AWS_REGION:-us-east-1}"

profile_args=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  profile_args=(--profile "${AWS_PROFILE}")
fi

aws "${profile_args[@]}" ec2 describe-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${GPU_INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].{State:State.Name,Type:InstanceType,AZ:Placement.AvailabilityZone,LaunchTime:LaunchTime,PublicIP:PublicIpAddress,PrivateIP:PrivateIpAddress}' \
  --output table
