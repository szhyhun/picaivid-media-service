#!/bin/bash
# Stop GPU worker container gracefully
#
# Gives the container 2 minutes to finish current job before force stopping

set -e

CONTAINER_NAME="picaivid-gpu-worker"
TIMEOUT=120  # 2 minutes grace period

echo "=== Stopping GPU Worker ==="

if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Container $CONTAINER_NAME is not running"
    exit 0
fi

echo "Stopping container with ${TIMEOUT}s grace period..."
docker stop -t $TIMEOUT $CONTAINER_NAME

echo "Removing container..."
docker rm $CONTAINER_NAME

echo ""
echo "=== Worker Stopped ==="
