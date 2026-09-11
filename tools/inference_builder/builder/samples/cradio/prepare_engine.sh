#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare C-RADIOv3-H TensorRT engine: export ONNX and build engine.
# Usage: prepare_engine.sh <docker_image> <model_repo>
#
# Requires: model files at <model_repo>/cradio_v3_h (from HuggingFace)
# Produces: <model_repo>/cradio_v3_h/model.plan (TensorRT FP16 engine)

set -e

IMAGE="$1"
MODEL_REPO="${2:-/tmp/model-repo}"
MODEL_DIR="$MODEL_REPO/cradio_v3_h"
RESOLUTION=256

if [ -z "$IMAGE" ]; then
    echo "Usage: $0 <docker_image> [model_repo]"
    exit 1
fi

# Skip if engine already exists
if [ -f "$MODEL_DIR/model.plan" ]; then
    echo "TensorRT engine already exists at $MODEL_DIR/model.plan, skipping"
    exit 0
fi

# Check that model files exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "=== Step 1: Export ONNX ==="
if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    docker run --rm --gpus all \
        -v "$MODEL_REPO:/models" \
        -v "$MODEL_DIR:/root/.cache/huggingface/modules/transformers_modules/cradio_v3_h" \
        --entrypoint python3 \
        "$IMAGE" /workspace/export_onnx.py \
            --output-dir /models/cradio_v3_h \
            --resolution "$RESOLUTION" \
            --model-path /models/cradio_v3_h
    echo "ONNX export complete: $MODEL_DIR/model.onnx"
else
    echo "ONNX model already exists at $MODEL_DIR/model.onnx, skipping export"
fi

echo "=== Step 2: Build TensorRT Engine ==="
docker run --rm --gpus all \
    -v "$MODEL_REPO:/models" \
    --entrypoint polygraphy \
    "$IMAGE" convert /models/cradio_v3_h/model.onnx \
        --fp16 \
        --trt-min-shapes "pixel_values:[1,3,${RESOLUTION},${RESOLUTION}]" \
        --trt-opt-shapes "pixel_values:[8,3,${RESOLUTION},${RESOLUTION}]" \
        --trt-max-shapes "pixel_values:[16,3,${RESOLUTION},${RESOLUTION}]" \
        -o /models/cradio_v3_h/model.plan

echo "TensorRT engine built: $MODEL_DIR/model.plan"
