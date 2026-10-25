#!/usr/bin/env bash
# Run a command inside the project container with the GPU, the repo and the data mounted
# (plus ~/.kaggle, read-only, when it exists). Builds the image first if it is missing.
#   RA_GPUS=device=1   GPU(s) to use (default: all; the code runs on the first one it sees)
#   RA_OFFLINE=1       no network, data mounted read-only (scripts/infer.sh sets this)
#   scripts/docker_run.sh python -m ra.train ...
#   scripts/docker_run.sh            # interactive shell
set -euo pipefail

IMAGE="${RA_IMAGE:-sadbodh-retail-action:latest}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${RA_DATA:-$REPO/data}"
KAGGLE_DIR="${KAGGLE_DIR:-$HOME/.kaggle}"
mkdir -p "$DATA_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -q -t "$IMAGE" "$REPO" >/dev/null

TTY_FLAGS="-i"
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS="-it"
# Rootless Docker maps container root to the calling user; a non-root --user there cannot write the repo.
USER_ARGS=(--user "$(id -u):$(id -g)")
case "$(docker info -f '{{.SecurityOptions}}' 2>/dev/null || true)" in *rootless*) USER_ARGS=() ;; esac
DATA_MOUNT="$DATA_DIR":/data
NET_ARGS=()
if [ "${RA_OFFLINE:-0}" = 1 ]; then DATA_MOUNT="$DATA_MOUNT:ro"; NET_ARGS=(--network none); fi
KAGGLE_MOUNT=()
[ -d "$KAGGLE_DIR" ] && KAGGLE_MOUNT=(-e KAGGLE_CONFIG_DIR=/tmp/home/.kaggle -v "$KAGGLE_DIR":/tmp/home/.kaggle:ro)

if [ $# -eq 0 ]; then set -- bash; fi

exec docker run --rm $TTY_FLAGS \
    --gpus "${RA_GPUS:-all}" \
    --ipc=host \
    ${USER_ARGS[@]+"${USER_ARGS[@]}"} \
    ${NET_ARGS[@]+"${NET_ARGS[@]}"} \
    -e HOME=/tmp/home \
    -e PYTHONPATH=/workspace \
    -e TORCH_HOME=/data/torch_cache \
    -e HF_HOME=/data/hf \
    -e HF_HUB_DISABLE_XET=1 \
    -e RA_PRED_BS \
    -v "$REPO":/workspace \
    -v "$DATA_MOUNT" \
    ${KAGGLE_MOUNT[@]+"${KAGGLE_MOUNT[@]}"} \
    -w /workspace \
    "$IMAGE" "$@"
