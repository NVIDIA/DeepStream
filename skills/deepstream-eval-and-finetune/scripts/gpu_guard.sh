#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  Licensed under Apache-2.0.
#
# Sequential-run guard: wait until the GPU is reasonably free before starting a
# heavy stage (engine build / fine-tune / deployed eval). Concurrent --gpus all
# jobs caused intermittent nvinfer build failures, so the skill runs its GPU
# stages ONE AT A TIME and calls this before each.
#
# Usage: gpu_guard.sh [max_used_mib] [timeout_s] [poll_s]
#   max_used_mib : proceed once GPU used memory is below this (default 6000;
#                  tolerates a small interactive container, blocks on a big job)
#   timeout_s    : give up waiting and proceed anyway (default 2400)
set -u
MAX=${1:-6000}; TIMEOUT=${2:-2400}; POLL=${3:-15}; t=0
q() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1; }
while :; do
  used=$(q); used=${used:-0}
  if [ "$used" -lt "$MAX" ]; then
    echo "[gpu_guard] GPU free enough (${used} MiB used < ${MAX}) — proceeding"; exit 0
  fi
  if [ "$t" -ge "$TIMEOUT" ]; then
    echo "[gpu_guard] timeout after ${t}s (GPU still ${used} MiB) — proceeding anyway"; exit 0
  fi
  echo "[gpu_guard] GPU busy (${used} MiB used); waiting ${POLL}s... (${t}/${TIMEOUT}s)"; sleep "$POLL"; t=$((t+POLL))
done
