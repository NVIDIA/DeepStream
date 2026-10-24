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
REID_SERVICE="${REID_SERVICE:-0}"
if [[ "$REID_SERVICE" != 0 && "$REID_SERVICE" != 1 ]]; then
    echo "ERROR: REID_SERVICE must be 0 or 1" >&2
    exit 1
fi
DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"
case "$DETECTOR_MODEL" in
    PeopleNetTransformer) PROFILE=scout_pnt ;;
    PeopleNet2.6.3) PROFILE=scout_pn2 ;;
    *) echo "ERROR: DETECTOR_MODEL must be PeopleNetTransformer or PeopleNet2.6.3" >&2; exit 1 ;;
esac

DATASET_DIR="$(realpath -e "${DATASET_DIR:-$MV3DT_DIR/datasets/scout}")"
MODEL_REPO="$(realpath -e "${MODEL_REPO:-$MV3DT_DIR/models}")"
REID_ARGS=()
DOCKER_NETWORK=bridge
REID_SUFFIX=
if [[ "$REID_SERVICE" = 1 ]]; then
    REID_ARGS=(--enable-reid-service)
    DOCKER_NETWORK=host
    REID_SUFFIX=_reid
fi
EXPERIMENT_DIR="$(realpath -m "${EXPERIMENT_DIR:-$MV3DT_DIR/experiments/deepstream/${PROFILE}${REID_SUFFIX}_run1}")"
if [[ -e "$EXPERIMENT_DIR" ]]; then
    echo "ERROR: Choose a new EXPERIMENT_DIR: $EXPERIMENT_DIR already exists" >&2
    exit 1
fi

if [[ "$REID_SERVICE" = 1 ]]; then
    curl --fail --silent --show-error http://127.0.0.1:8000/health/ready > /dev/null
fi

REPLAY_ARGS=()
if [[ "$REPLAY" = 1 ]]; then
    REPLAY_ARGS=(--replay "$DATASET_DIR/detections/$DETECTOR_MODEL")
fi

echo "SCOUT: detector=$DETECTOR_MODEL replay=$REPLAY reid-service=$REID_SERVICE output=$EXPERIMENT_DIR"
source mv3dt_venv/bin/activate
python -m utils.deepstream_auto_configurator \
    --profile "$PROFILE" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$EXPERIMENT_DIR" \
    --enable-file-output \
    "${REID_ARGS[@]}" \
    "${REPLAY_ARGS[@]}"

# A terminal prevents the app's keyboard polling loop from repeatedly reading EOF.
docker run -t --rm --gpus all --network "$DOCKER_NETWORK" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v "$MODEL_REPO:/workspace/models" \
    -v "$DATASET_DIR:/workspace/inputs:ro" \
    -v "$EXPERIMENT_DIR:/workspace/experiments" \
    -w /workspace/experiments \
    "${DEEPSTREAM_IMAGE:-ghcr.io/nvidia/deepstream:9.2-triton-multiarch}" \
    deepstream-test5-app -c config_deepstream.txt \
    2>&1 | tee "$EXPERIMENT_DIR/run_deepstream_app.log"
grep -q 'App run successful' "$EXPERIMENT_DIR/run_deepstream_app.log"
