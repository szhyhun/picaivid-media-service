#!/bin/bash
# Setup script for GPU instance
# Run this after SSH-ing into a fresh GPU instance

set -e

echo "=== Picaivid GPU Instance Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y docker.io
    sudo usermod -aG docker $USER
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker for NVIDIA runtime
echo "Configuring Docker for GPU..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
echo "Verifying GPU access..."
nvidia-smi

# Test Docker GPU access
echo "Testing Docker GPU access..."
sudo docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Set environment variables (DATABASE_URL, S3_BUCKET, etc.)"
echo "2. Login to ECR: aws ecr get-login-password | docker login ..."
echo "3. Pull image: docker pull YOUR_ECR_URI/picaivid-media:gpu-latest"
echo "4. Run: ./scripts/aws/start_gpu_worker.sh"
