#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the DeepStream Eval & Fine-tune Web UI (read-only viewer, no GPU needed)
set -euo pipefail
SK=.claude/skills/deepstream-eval-and-finetune
echo "== DeepStream Eval & Fine-tune UI -> http://localhost:8078 =="
DS_EVAL_ROOT=/work \
  PYTHONDONTWRITEBYTECODE=1 \
  build/.venv_train/bin/python -m uvicorn server:app \
  --app-dir /work/$SK/app \
  --host 127.0.0.1 --port 8078
