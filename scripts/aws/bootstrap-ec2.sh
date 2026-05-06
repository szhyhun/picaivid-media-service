#!/usr/bin/env bash
set -euo pipefail

# Bootstrap this repo on an EC2 GPU host.
#
# Usage:
#   ./scripts/aws/bootstrap-ec2.sh \
#     --repo-dir /srv/picaivid/picaivid-media-service \
#     --enable-api 0

REPO_DIR="/srv/picaivid/picaivid-media-service"
ENABLE_API=0
INSTALL_CUDA_TORCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --enable-api) ENABLE_API="$2"; shift 2 ;;
    --install-cuda-torch) INSTALL_CUDA_TORCH="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bootstrap-ec2.sh [--repo-dir PATH] [--enable-api 0|1] [--install-cuda-torch 0|1]
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
fi

log() { echo; echo "==> $*"; }

log "Installing OS dependencies"
${SUDO} apt-get update
${SUDO} apt-get install -y git curl ca-certificates awscli python3 python3-venv python3-pip

log "Installing Python dependencies"
cd "${REPO_DIR}"
python3 -m venv venv || true
source venv/bin/activate
pip install -U pip
pip install -r requirements.lock.txt

if [[ "${INSTALL_CUDA_TORCH}" == "1" ]]; then
  log "Installing CUDA-enabled PyTorch"
  pip uninstall -y torch torchvision torchaudio || true
  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
fi

log "Checking torch backend"
python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

log "Installing systemd units and env templates"
${SUDO} mkdir -p /etc/picaivid
${SUDO} cp "${REPO_DIR}/deploy/systemd/picaivid-media-worker.service" /etc/systemd/system/
if [[ "${ENABLE_API}" == "1" ]]; then
  ${SUDO} cp "${REPO_DIR}/deploy/systemd/picaivid-media-api.service" /etc/systemd/system/
fi

if [[ ! -f /etc/picaivid/media-worker.env ]]; then
  ${SUDO} cp "${REPO_DIR}/deploy/env/media-worker.env.example" /etc/picaivid/media-worker.env
  echo "Created /etc/picaivid/media-worker.env (edit required)"
fi
if [[ "${ENABLE_API}" == "1" && ! -f /etc/picaivid/media-api.env ]]; then
  ${SUDO} cp "${REPO_DIR}/deploy/env/media-api.env.example" /etc/picaivid/media-api.env
  echo "Created /etc/picaivid/media-api.env (edit required)"
fi

${SUDO} systemctl daemon-reload
${SUDO} systemctl enable picaivid-media-worker
if [[ "${ENABLE_API}" == "1" ]]; then
  ${SUDO} systemctl enable picaivid-media-api
fi

log "Done. Start when env is ready:"
echo "sudo systemctl restart picaivid-media-worker"
if [[ "${ENABLE_API}" == "1" ]]; then
  echo "sudo systemctl restart picaivid-media-api"
fi
