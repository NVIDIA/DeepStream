#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
#
# Install the Web UI's extra deps into the shared fine-tune venv (build/.venv_train),
# so the server's subprocesses use the SAME interpreter as run.sh. Idempotent.
# Run from the working root (where build/.venv_train lives). Best run inside the
# DeepStream container so the wheels match the container python, but pip-only and
# host-safe too.
set -euo pipefail
PY=${PY:-build/.venv_train/bin/python}

if [ ! -x "$PY" ]; then
  echo "[setup] ERROR: $PY not found. Run from the working root (where build/.venv_train lives)." >&2
  exit 1
fi

echo "[setup] installing FastAPI + Uvicorn into $PY"
"$PY" -m pip install --quiet "fastapi>=0.111" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9"
"$PY" -c "import fastapi, uvicorn; print('[setup] UI deps OK — fastapi', fastapi.__version__, '/ uvicorn', uvicorn.__version__)"
