#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# Boot the DeepStream Eval & Fine-tune Web UI INSIDE the DeepStream container —
# the same `docker run` the validated run.sh uses, plus a published port. Run from
# the working root (where models/, runs/, reports/, build/.venv_train live):
#
#     bash .claude/skills/deepstream-eval-and-finetune/app/launch.sh
#
# Then open http://localhost:8078. The UI runs the SAME scripts as the CLI; the
# CLI / examples/*/run.sh flow remains canonical. Override PORT / IMAGE via env.
#
# To run the read-only viewer on the HOST (no GPU, against existing artifacts):
#     DS_EVAL_ROOT="$PWD" PYTHONDONTWRITEBYTECODE=1 build/.venv_train/bin/python -m uvicorn server:app \
#       --app-dir "$PWD/.claude/skills/deepstream-eval-and-finetune/app" \
#       --host 127.0.0.1 --port 8078
#
# The skill dir stays pristine: PYTHONDONTWRITEBYTECODE keeps __pycache__ out of it, and all
# durable run outputs (engines, metrics, reports, run history, charts, logs) are written under
# the working root. On Windows/WSL, high-frequency data/Arrow inputs use a persistent Docker volume.
set -euo pipefail

SK=.claude/skills/deepstream-eval-and-finetune
PORT=${PORT:-8078}
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}

# Windows/WSL performance: keep high-frequency dataset + Arrow reads in a persistent
# Linux-backed Docker volume. Native Linux stays unchanged unless explicitly enabled.
CACHE_ARGS=()
CACHE_MODE=${DS_DATA_CACHE:-auto}
IS_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
if [ "$CACHE_MODE" != "off" ] && { [ "$CACHE_MODE" = "on" ] || [ "$IS_WSL" = "1" ]; }; then
  if [ -n "${DS_DATA_VOLUME:-}" ]; then
    DATA_VOLUME=$DS_DATA_VOLUME
  else
    PROJECT_KEY=$(pwd -P)
    if [[ "$PROJECT_KEY" =~ ^/mnt/([a-zA-Z])/(.*)$ ]]; then
      DRIVE=${BASH_REMATCH[1],,}
      REST=${BASH_REMATCH[2]//\//\\}
      PROJECT_KEY="${DRIVE}:\\${REST}"
    fi
    PROJECT_KEY=${PROJECT_KEY,,}
    PROJECT_HASH=$(printf '%s' "$PROJECT_KEY" | sha256sum | cut -c1-12)
    DATA_VOLUME="ds-eval-data-${PROJECT_HASH}"
  fi
  docker volume create \
    --label com.nvidia.deepstream.eval-cache=true \
    --label "com.nvidia.deepstream.project=$(pwd -P)" \
    "$DATA_VOLUME" >/dev/null
  CACHE_ARGS=(-v "${DATA_VOLUME}:/work/data"
              -e "DS_DATA_CACHE_ROOT=/work/data"
              -e "DS_DATA_VOLUME=${DATA_VOLUME}")
  case "${DS_DATA_CACHE_REFRESH:-}" in
    1|true|TRUE|yes|YES|on|ON) CACHE_ARGS+=(-e "DS_DATA_CACHE_REFRESH=1") ;;
  esac
  echo "[launch] Windows/WSL dataset cache: $DATA_VOLUME -> /work/data"
fi

# Datasets that live OUTSIDE the project must be bind-mounted, or the container (which only sees the
# project at /work) can't read them — ingest then fails with "local dataset path not found inside this
# container". Mount read-only at the SAME absolute path so the dataset path you type in the UI (or an
# in-project symlink pointing at it) resolves unchanged inside the container. Two knobs:
#   DATASETS_ROOT  one or more roots (semicolon-separated when paths contain spaces), each mounted
#                  at the same path. The robust default
#                  for "bring any dataset under this root", e.g.:
#                    DATASETS_ROOT=/mnt/smb_share bash .../app/launch.sh
#                    DATASETS_ROOT="/mnt/smb_share;/data/team sets" bash .../app/launch.sh
#   DATASET_MOUNT  a single explicit mount, src or src:dst (kept for backward compatibility):
#                    DATASET_MOUNT=/data/voc bash .../app/launch.sh
#                    DATASET_MOUNT=/host/src:/container/dst bash .../app/launch.sh
MOUNT_ARGS=()
ROOTS=()
if [[ "${DATASETS_ROOT:-}" == *";"* ]]; then
  IFS=';' read -r -a ROOTS <<< "$DATASETS_ROOT"
elif [ -n "${DATASETS_ROOT:-}" ]; then
  read -r -a ROOTS <<< "$DATASETS_ROOT"
fi
for root in "${ROOTS[@]}"; do
  if [ -d "$root" ]; then MOUNT_ARGS+=(-v "${root}:${root}:ro"); else
    echo "[launch] WARNING: DATASETS_ROOT entry '$root' is not a directory on the host — skipping" >&2
  fi
done
if [ -n "${DATASET_MOUNT:-}" ]; then
  case "$DATASET_MOUNT" in
    *:*) MOUNT_ARGS+=(-v "${DATASET_MOUNT}:ro") ;;        # explicit src:dst
    *)   MOUNT_ARGS+=(-v "${DATASET_MOUNT}:${DATASET_MOUNT}:ro") ;;  # same path in + out
  esac
fi

# in-container command: ensure UI deps, then start uvicorn bound to all interfaces.
# --app-dir points at the app/ folder so the module is the bare `server:app`
# (server.py + pipeline.py are siblings there); DS_EVAL_ROOT stays the working root.
INNER="set -e
# bootstrap the full environment on first run (venv + all deps + ds_image_eval); also covers
# fresh machines. If torch is missing the env was never set up; run the full setup.
if ! build/.venv_train/bin/python -c 'import torch' 2>/dev/null; then
  bash $SK/setup.sh
fi
# UI deps live with the UI, not in the core manifest — install them on demand.
if ! build/.venv_train/bin/python -c 'import fastapi,uvicorn' 2>/dev/null; then
  bash $SK/app/setup.sh
fi
echo '== DeepStream Eval & Fine-tune UI -> http://localhost:${PORT} =='
DS_EVAL_ROOT=/work build/.venv_train/bin/python -m uvicorn server:app \
  --app-dir /work/$SK/app --host 0.0.0.0 --port ${PORT}"

exec docker run --rm -it --gpus all --shm-size=16g \
  -v "$PWD":/work -w /work \
  "${CACHE_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -e HF_HOME=/work/build/hf_cache \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint bash "$IMAGE" -c "$INNER"
