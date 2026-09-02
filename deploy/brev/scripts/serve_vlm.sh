#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Start a local OpenAI-compatible VLM server (vLLM) for the rtvi demos.
#
# The rtvi_vlm_core_app / rtvi_vlm_openapi_spec prompts need a multimodal VLM
# reachable at an OpenAI-compatible endpoint. This serves one with vLLM in a
# container on http://127.0.0.1:8000/v1 using a small open model that fits an
# A40 (46 GB). It is idempotent: re-running reuses an already-running server.
#
# Gated by SERVE_VLM=1 so it never downloads a model unintentionally. The first
# run downloads the model from Hugging Face into a cache volume (several GB).
#
# Usage:
#   SERVE_VLM=1 bash deploy/brev/scripts/serve_vlm.sh
#   SERVE_VLM=1 VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct HF_TOKEN=hf_xxx \
#     bash deploy/brev/scripts/serve_vlm.sh
#
set -euo pipefail

SERVE_VLM=${SERVE_VLM:-0}
# Qwen2.5-VL (model type qwen2_5_vl) needs the Transformers shipped in vLLM >= v0.7.3;
# the older v0.6.6 image raised `KeyError: 'qwen2_5_vl'`. ~7GB download; fits an A40 (46GB).
VLM_MODEL=${VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
VLM_PORT=${VLM_PORT:-8000}
VLM_IMAGE=${VLM_IMAGE:-vllm/vllm-openai:v0.7.3}
VLM_CONTAINER=${VLM_CONTAINER:-ds-agent-vllm}
HF_TOKEN=${HF_TOKEN:-}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
# A40 has 46 GB; cap the KV-cache so a 3B-class VLM fits comfortably.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

VLM_ENDPOINT="http://127.0.0.1:${VLM_PORT}/v1"

if [[ "${SERVE_VLM}" != "1" ]]; then
  cat <<MSG
Local VLM serve is disabled (SERVE_VLM != 1).
To serve ${VLM_MODEL} on ${VLM_ENDPOINT} for the rtvi demos:
  1. Ensure the model fits your GPU. Default is a 3B-class VL model (fits an A40, 46 GB).
  2. For gated models, export HF_TOKEN=hf_xxx.
  3. Re-run: SERVE_VLM=1 bash deploy/brev/scripts/serve_vlm.sh
The first run downloads the model from Hugging Face (several GB).
MSG
  exit 0
fi

# Already healthy AND served by our own container? Reuse it. Guard against a different
# service occupying the port so we don't falsely report "already serving".
if docker ps --format '{{.Names}}' | grep -Fxq "${VLM_CONTAINER}" \
  && curl -fsS "${VLM_ENDPOINT}/models" >/dev/null 2>&1; then
  echo "VLM already serving at ${VLM_ENDPOINT}"
  curl -fsS "${VLM_ENDPOINT}/models" || true
  echo
  exit 0
fi

# Remove any stale container with the same name before starting fresh.
docker rm -f "${VLM_CONTAINER}" >/dev/null 2>&1 || true

mkdir -p "${HF_CACHE}"

echo "Pulling ${VLM_IMAGE} (skip if cached)"
docker pull "${VLM_IMAGE}"

echo "Starting vLLM (${VLM_MODEL}) on ${VLM_ENDPOINT}"
HF_ENV=()
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV+=(-e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi

docker run -d --name "${VLM_CONTAINER}" \
  --gpus all \
  --network host \
  --ipc host \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${HF_ENV[@]}" \
  "${VLM_IMAGE}" \
  --model "${VLM_MODEL}" \
  --port "${VLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code

echo "Waiting for ${VLM_ENDPOINT} to become healthy (model download may take minutes)..."
for i in $(seq 1 120); do
  if curl -fsS "${VLM_ENDPOINT}/models" >/dev/null 2>&1; then
    echo "VLM is up at ${VLM_ENDPOINT}"
    curl -fsS "${VLM_ENDPOINT}/models" || true
    echo
    exit 0
  fi
  # Heartbeat + the latest vLLM log line so the wait isn't a silent freeze (the
  # download/load progress is visible, and the user knows it's still working).
  echo "  ...starting ${i}/120 ($((i * 10))s elapsed); latest vLLM log:"
  docker logs --tail 1 "${VLM_CONTAINER}" 2>&1 | sed 's/^/    | /' || true
  sleep 10
done

echo "VLM did not become healthy in time. Recent logs:" >&2
docker logs --tail 60 "${VLM_CONTAINER}" || true
exit 1
