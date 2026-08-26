#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/aws/gpu.sh start
  ./scripts/aws/gpu.sh stop
  ./scripts/aws/gpu.sh status

Required environment:
  GPU_INSTANCE_ID   EC2 instance id of the GPU worker

Optional environment:
  AWS_PROFILE       AWS CLI profile to use
  AWS_REGION        AWS region (default: us-west-2)

Examples:
  export AWS_PROFILE=picaivid-admin
  export AWS_REGION=us-west-2
  export GPU_INSTANCE_ID=i-0123456789abcdef0

  ./scripts/aws/gpu.sh start
  ./scripts/aws/gpu.sh status
  ./scripts/aws/gpu.sh stop
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  start)
    exec "${SCRIPT_DIR}/gpu-start.sh"
    ;;
  stop)
    exec "${SCRIPT_DIR}/gpu-stop.sh"
    ;;
  status)
    exec "${SCRIPT_DIR}/gpu-status.sh"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
