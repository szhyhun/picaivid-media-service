#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=""
DEPLOY_DIR="/srv/picaivid/picaivid-media-service"
ENV_FILE="/etc/picaivid/media-worker.env"
WORKER_SERVICE="picaivid-media-worker"
API_SERVICE="picaivid-media-api"
RESTART_API="${RESTART_API:-0}"
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-0}"
CUDA_WHL_INDEX="${CUDA_WHL_INDEX:-https://download.pytorch.org/whl/cu124}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --worker-service) WORKER_SERVICE="$2"; shift 2 ;;
    --api-service) API_SERVICE="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${SOURCE_DIR}" ]]; then
  echo "--source-dir is required"
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}"
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

PREV_DIR="${DEPLOY_DIR}.prev"

cd "${SOURCE_DIR}"
python3 -m venv venv || true
source venv/bin/activate
pip install -U pip
pip install -r requirements.lock.txt
if [[ "${INSTALL_CUDA_TORCH}" == "1" ]]; then
  pip uninstall -y torch torchvision torchaudio || true
  pip install --index-url "${CUDA_WHL_INDEX}" torch torchvision torchaudio
fi
python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

rm -rf "${PREV_DIR}"
if [[ -d "${DEPLOY_DIR}" ]]; then
  mv "${DEPLOY_DIR}" "${PREV_DIR}"
fi
mv "${SOURCE_DIR}" "${DEPLOY_DIR}"

sudo systemctl restart "${WORKER_SERVICE}"
if [[ "${RESTART_API}" == "1" ]]; then
  sudo systemctl restart "${API_SERVICE}"
fi
sudo systemctl --no-pager --full status "${WORKER_SERVICE}" | head -n 40

