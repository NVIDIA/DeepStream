#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Preflight for deepstream-eval-and-finetune: verify the environment BEFORE running
# the loop, so failures surface up front (not mid-engine-build). Checks:
#   1. docker daemon reachable
#   2. the DeepStream container image is pulled
#   3. GPU is visible inside the container (docker --gpus all)
#   4. the shared venv has every Python package the skill needs
#
# Usage: bash scripts/preflight.sh [IMAGE] [VENV]
#   IMAGE default: nvcr.io/nvidia/deepstream:9.1-triton-multiarch  (latest)
#   VENV  default: build/.venv_train   (relative to the working root / $PWD)
set -u
IMG="${1:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
VENV="${2:-build/.venv_train}"
PKGS="torch transformers datasets torchmetrics pycocotools reportlab matplotlib albumentations onnx onnxscript huggingface_hub PIL yaml numpy"
fail=0
ok(){  echo "  [OK]      $1"; }
warn(){ echo "  [WARN]    $1"; }
bad(){  echo "  [MISSING] $1"; fail=1; }

# ---------------------------------------------------------------------------------------------
# Container-mode: when this script is itself run INSIDE the container (e.g. on Windows/macOS,
# where there's no host bash, you invoke it as
#   docker run --rm --gpus all -v <pwd>:/work -w /work <image> bash scripts/preflight.sh
# ), the host docker-daemon / image-pulled checks are moot (you're already running the image),
# and there's no nested `docker` CLI. So verify what still matters DIRECTLY: GPU + venv packages.
# On a Linux host (no /.dockerenv) we fall through to the original host-orchestration checks.
if [ -f /.dockerenv ]; then
  echo "[preflight] container-mode (running inside the container — verifying GPU + venv directly)"
  echo "[preflight] 1/2 GPU (nvidia-smi)"
  nvidia-smi -L >/dev/null 2>&1 && ok "GPU visible in container" \
    || bad "no GPU in container — run with --gpus all (Windows: Docker Desktop WSL2 backend + NVIDIA driver)"
  echo "[preflight] 2/2 python packages in $VENV"
  if [ -x "$VENV/bin/python" ]; then
    miss=$("$VENV/bin/python" -c "import importlib.util as u;print(' '.join(p for p in '$PKGS'.split() if u.find_spec(p) is None))" 2>/dev/null)
    if [ -z "${miss// /}" ]; then ok "all packages present"
    else bad "missing in venv: $miss  (run setup.sh to (re)build the venv)"; fi
  else
    bad "venv not found: $VENV/bin/python — run setup.sh first (creates build/.venv_train)"
  fi
  echo "[preflight] RESULT: $([ $fail -eq 0 ] && echo 'PASS — environment ready' || echo 'FAIL — fix [MISSING] items before running')"
  exit $fail
fi

echo "[preflight] 1/4 docker daemon"
docker info >/dev/null 2>&1 && ok "docker reachable" || bad "docker not installed or daemon not running"

echo "[preflight] 2/4 DeepStream image: $IMG"
have_img=0
if docker image inspect "$IMG" >/dev/null 2>&1; then ok "image present"; have_img=1
else warn "image not pulled — run: docker pull $IMG"; fi

if [ "$have_img" = 1 ]; then
  echo "[preflight] 3/4 GPU via docker --gpus all"
  # --entrypoint bypasses the container's startup banner so it can't pollute output.
  docker run --rm --gpus all --entrypoint nvidia-smi "$IMG" -L >/dev/null 2>&1 \
    && ok "GPU visible in container" \
    || bad "no GPU through --gpus all (check driver + nvidia-container-toolkit)"

  echo "[preflight] 4/4 python packages in $VENV"
  if [ -x "$VENV/bin/python" ]; then
    miss=$(docker run --rm --entrypoint /work/"$VENV"/bin/python -v "$PWD":/work "$IMG" \
      -c "import importlib.util as u;print(' '.join(p for p in '$PKGS'.split() if u.find_spec(p) is None))" 2>/dev/null)
    if [ -z "${miss// /}" ]; then ok "all packages present"
    else bad "missing in venv: $miss  (pip install them into $VENV)"; fi
  else
    bad "venv not found: $VENV/bin/python — create with 'virtualenv' (NOT python -m venv; container python lacks ensurepip)"
  fi
else
  echo "[preflight] 3/4 GPU            — SKIPPED (image not pulled)"
  echo "[preflight] 4/4 python packages — SKIPPED (image not pulled)"
fi

echo "[preflight] RESULT: $([ $fail -eq 0 ] && echo 'PASS — environment ready' || echo 'FAIL — fix [MISSING] items before running')"
exit $fail
