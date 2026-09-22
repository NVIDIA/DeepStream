#!/bin/bash
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
# Run MV3DT on a CUSTOM multi-camera dataset with off-the-shelf detectors.
#
# Usage (from the MV3DT app root):
#   DATASET_DIR=/abs/path/to/my_dataset ./scripts/custom_4cam_ds.sh
#
# Expected dataset layout:
#   my_dataset/
#   |-- videos/   cam_00.mp4, cam_01.mp4, ...   (synchronized, same resolution)
#   |-- camInfo/  cam_00.yml,  cam_01.yml,  ... (projectionMatrix_3x4_w2p + modelInfo)
#   |-- map.png        (optional, BEV only)
#   `-- transforms.yml (optional, BEV only)
#
# Tunables (env vars):
#   DATASET_DIR        required, dataset root
#   EXPERIMENT_DIR     default experiments/deepstream/<dataset name>
#   MODEL_REPO         default <app root>/models
#   DETECTOR_MODEL     PeopleNetTransformer (default) | RTDETR | PeopleNet2.6.3
#   TRACKER_CONFIG     default config_tracker.yml (use config_tracker_2d.yml to convert a 2D pipeline)
#   CONFIG_OVERRIDES   default override_tracker_4cam.yml (empty string disables)
#   RUN_MODE           display (default when $DISPLAY is set) | headless
#   ENABLE_BEV         1 (default) when map.png + transforms.yml exist
#   DEEPSTREAM_IMAGE   default nvcr.io/nvidia/deepstream:9.1-triton-multiarch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to your custom dataset root (must contain videos/ and camInfo/)}"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO_ROOT/experiments/deepstream/$(basename "$DATASET_DIR")}"
export MODEL_REPO="${MODEL_REPO:-$REPO_ROOT/models}"

# ---------------------------------------------------------------- validation
[[ -d "$DATASET_DIR/videos"  ]] || { echo "ERROR: $DATASET_DIR/videos missing"  >&2; exit 1; }
[[ -d "$DATASET_DIR/camInfo" ]] || { echo "ERROR: $DATASET_DIR/camInfo missing" >&2; exit 1; }

NUM_VIDEOS=$(find "$DATASET_DIR/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)
NUM_CALIB=$(find "$DATASET_DIR/camInfo" -maxdepth 1 -type f -name '*.yml' | wc -l)
NUM_CALIB_YAML=$(find "$DATASET_DIR/camInfo" -maxdepth 1 -type f -name '*.yaml' | wc -l)

(( NUM_VIDEOS > 0 )) || { echo "ERROR: no .mp4 files in $DATASET_DIR/videos" >&2; exit 1; }

# The auto-configurator only scans .yml; make non-destructive copies of .yaml exports (e.g. from AutoMagicCalib).
if (( NUM_CALIB == 0 )) && (( NUM_CALIB_YAML > 0 )); then
    echo "Copying camInfo/*.yaml -> *.yml (existing .yml files are never overwritten)"
    find "$DATASET_DIR/camInfo" -maxdepth 1 -type f -name '*.yaml' -print0 \
        | while IFS= read -r -d '' f; do cp -n "$f" "${f%.yaml}.yml"; done
    NUM_CALIB=$(find "$DATASET_DIR/camInfo" -maxdepth 1 -type f -name '*.yml' | wc -l)
fi

(( NUM_CALIB == NUM_VIDEOS )) || {
    echo "ERROR: $NUM_VIDEOS videos but $NUM_CALIB camInfo/*.yml files - one calibration per camera is required" >&2
    exit 1
}

# Videos and calibrations are paired by basename; sorted order is the fallback and is easy to get wrong.
for v in "$DATASET_DIR"/videos/*.mp4; do
    b="$(basename "$v" .mp4)"
    [[ -f "$DATASET_DIR/camInfo/$b.yml" ]] || echo "WARN: no camInfo/$b.yml for videos/$b.mp4 - pairing falls back to sorted order"
done

echo "Dataset: $DATASET_DIR ($NUM_VIDEOS cameras)"

# ------------------------------------------------------------ detector/tracker
export DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"
case "$DETECTOR_MODEL" in
    PeopleNetTransformer) DETECTOR_CONFIG="config_pgie.txt" ;;
    RTDETR)               DETECTOR_CONFIG="config_pgie_rt_detr.txt" ;;
    PeopleNet2.6.3)       DETECTOR_CONFIG="config_pgie_peoplenet.txt" ;;
    *) echo "ERROR: unsupported DETECTOR_MODEL=$DETECTOR_MODEL" >&2; exit 1 ;;
esac
TRACKER_CONFIG="${TRACKER_CONFIG:-config_tracker.yml}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES-override_tracker_4cam.yml}"

echo "Using detector model: $DETECTOR_MODEL (detector=$DETECTOR_CONFIG, tracker=$TRACKER_CONFIG)"

# ------------------------------------------------------------------- run mode
RUN_MODE="${RUN_MODE:-$([[ -n "${DISPLAY:-}" ]] && echo display || echo headless)}"
CONFIG_FLAGS=(--enable-msg-broker)
if [[ "$RUN_MODE" == "headless" ]]; then
    CONFIG_FLAGS+=(--enable-file-output)
else
    CONFIG_FLAGS+=(--enable-osd --enable-file-output)
fi

ENABLE_BEV="${ENABLE_BEV:-1}"
if [[ "$ENABLE_BEV" == "1" && ( ! -f "$DATASET_DIR/map.png" || ! -f "$DATASET_DIR/transforms.yml" ) ]]; then
    echo "WARN: map.png and/or transforms.yml missing - BEV visualization disabled"
    ENABLE_BEV=0
fi

# ----------------------------------------------------------------- GPU runtime
if docker info 2>/dev/null | grep -q 'Runtimes.*nvidia'; then
    GPU_FLAG="--runtime=nvidia"
elif docker run --help | grep -q -- "--gpus"; then
    GPU_FLAG="--gpus all"
else
    echo "No GPU support found in Docker." >&2
    exit 1
fi

mkdir -p "$EXPERIMENT_DIR"/{infer-kitti-dump,tracker-kitti-dump,outVideos,bev_outputs}

# ------------------------------------------------------------ generate configs
source mv3dt_venv/bin/activate

CONFIG_OVERRIDE_ARGS=()
[[ -n "$CONFIG_OVERRIDES" ]] && CONFIG_OVERRIDE_ARGS+=(--config-overrides="$CONFIG_OVERRIDES")

python utils/deepstream_auto_configurator.py \
    --dataset-dir="$DATASET_DIR" \
    "${CONFIG_FLAGS[@]}" \
    --detector-config="$DETECTOR_CONFIG" \
    --tracker-config="$TRACKER_CONFIG" \
    "${CONFIG_OVERRIDE_ARGS[@]}" \
    --output-dir="$EXPERIMENT_DIR"

# ---------------------------------------------------------------------- BEV
BEV_PID=""
if [[ "$ENABLE_BEV" == "1" && "$RUN_MODE" != "headless" ]]; then
    python utils/kafka_bev_visualizer.py \
        --dataset-path="$DATASET_DIR" \
        --msgconv-config="$EXPERIMENT_DIR/config_msgconv.txt" \
        --average-multi-cam \
        --show-ids &
    BEV_PID=$!
fi
trap '[[ -n "$BEV_PID" ]] && kill "$BEV_PID" 2>/dev/null || true' EXIT

# ------------------------------------------------------------------- pipeline
DOCKER_DISPLAY_ARGS=()
[[ "$RUN_MODE" != "headless" ]] && DOCKER_DISPLAY_ARGS=(-v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY="$DISPLAY")

docker run -t --privileged --rm --net=host $GPU_FLAG \
    -v "$MODEL_REPO:/workspace/models" \
    -v "$DATASET_DIR:/workspace/inputs" \
    -v "$EXPERIMENT_DIR:/workspace/experiments" \
    "${DOCKER_DISPLAY_ARGS[@]}" \
    -w /workspace/experiments \
    "${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}" \
    deepstream-test5-app -c config_deepstream.txt

echo "Done. Artifacts under $EXPERIMENT_DIR (outVideos/tiled_display_raw.mp4, tracker-kitti-dump/)"
