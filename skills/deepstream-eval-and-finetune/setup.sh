#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# One-command environment bootstrap for deepstream-eval-and-finetune on a FRESH machine.
# Creates the shared venv (build/.venv_train), installs all Python deps, and builds the
# ds_image_eval C app. Idempotent — safe to re-run.
#
# Prerequisites: Docker + an NVIDIA GPU + the DeepStream image pulled
#   docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch
#
# Run it INSIDE the DeepStream container, from the working root (where models/, runs/,
# reports/, build/ will live) — so torch/CUDA and the C app match the runtime:
#
#   docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
#     --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
#     .claude/skills/deepstream-eval-and-finetune/setup.sh
#
# Then run a demo (examples/rtdetr-aerial-sheep/run.sh) or launch the UI (app/launch.sh).
set -euo pipefail

SK=".claude/skills/deepstream-eval-and-finetune"
VENV="build/.venv_train"
PY="$VENV/bin/python"

if [ ! -d "$SK" ]; then
  echo "[setup] ERROR: run from the working root (the dir that contains $SK)." >&2
  exit 1
fi

# 1) virtualenv (the container python lacks ensurepip, so bootstrap virtualenv via pip)
if [ ! -x "$PY" ]; then
  echo "[setup] creating venv at $VENV"
  python3 -m pip install --quiet --user virtualenv 2>/dev/null || python3 -m pip install --quiet virtualenv
  python3 -m virtualenv "$VENV"
else
  echo "[setup] venv exists: $VENV"
fi

# 2) Python dependencies
echo "[setup] installing Python deps from $SK/scripts/requirements.txt (this can take several minutes)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install -r "$SK/scripts/requirements.txt"

# 3) build the DeepStream eval C app (rebuilt per machine/DeepStream version; not shipped)
if [ -f "$SK/scripts/ds_image_eval" ]; then
  echo "[setup] ds_image_eval already built"
else
  echo "[setup] building ds_image_eval (make -C $SK/scripts)"
  make -C "$SK/scripts"
fi

# 4) verify
echo "[setup] verifying imports…"
"$PY" - <<'PYV'
import importlib.util, sys
mods = ["torch","torchvision","transformers","datasets","accelerate","timm","torchmetrics",
        "pycocotools","onnx","onnxscript","albumentations","cv2","PIL","yaml","numpy",
        "reportlab","matplotlib","huggingface_hub"]   # UI deps: see app/setup.sh
missing = [m for m in mods if importlib.util.find_spec(m) is None]
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if missing: print("  MISSING:", missing); sys.exit(1)
print("  all required packages present")
PYV

echo
echo "[setup] DONE. Next:"
echo "  • run a validated demo:  bash $SK/examples/rtdetr-aerial-sheep/run.sh   (inside this container)"
echo "  • or launch the web UI:  bash $SK/app/launch.sh                          (from the host)"
