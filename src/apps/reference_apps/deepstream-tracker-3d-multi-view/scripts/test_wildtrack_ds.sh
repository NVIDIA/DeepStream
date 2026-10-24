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

set -euo pipefail

DS_DIR="${DS_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
MV3DT_DIR="$DS_DIR/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
cd "$MV3DT_DIR"

REPLAY="${REPLAY:-1}"
DATASET_DIR="$(realpath -e "${DATASET_DIR:-$MV3DT_DIR/datasets/wildtrack}")"
MODEL_REPO="$(realpath -e "${MODEL_REPO:-$MV3DT_DIR/models}")"
EXPERIMENT_DIR="$(realpath -m "${EXPERIMENT_DIR:-$MV3DT_DIR/experiments/deepstream/wildtrack_run1}")"
if [[ -e "$EXPERIMENT_DIR" ]]; then
    echo "ERROR: Choose a new EXPERIMENT_DIR: $EXPERIMENT_DIR already exists" >&2
    exit 1
fi

REPLAY_ARGS=()
if [[ "$REPLAY" = 1 ]]; then
    REPLAY_ARGS=(--replay "$DATASET_DIR/detections/PeopleNetTransformer")
fi

echo "WILDTRACK: detector=PeopleNetTransformer replay=$REPLAY output=$EXPERIMENT_DIR"
source mv3dt_venv/bin/activate
python -m utils.deepstream_auto_configurator \
    --profile wildtrack \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$EXPERIMENT_DIR" \
    --enable-file-output \
    "${REPLAY_ARGS[@]}"

# A terminal prevents the app's keyboard polling loop from repeatedly reading EOF.
docker run -t --rm --gpus all --network bridge \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v "$MODEL_REPO:/workspace/models" \
    -v "$DATASET_DIR:/workspace/inputs:ro" \
    -v "$EXPERIMENT_DIR:/workspace/experiments" \
    -w /workspace/experiments \
    "${DEEPSTREAM_IMAGE:-ghcr.io/nvidia/deepstream:9.2-triton-multiarch}" \
    deepstream-test5-app -c config_deepstream.txt \
    2>&1 | tee "$EXPERIMENT_DIR/run_deepstream_app.log"
grep -q 'App run successful' "$EXPERIMENT_DIR/run_deepstream_app.log"
