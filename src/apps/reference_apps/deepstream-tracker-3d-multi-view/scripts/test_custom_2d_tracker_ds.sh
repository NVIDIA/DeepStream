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

# Set dataset, model, and experiment directories
DS_DIR="${DS_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
MV3DT_DIR="$DS_DIR/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
cd "$MV3DT_DIR" || exit 1
DATASET_DIR="${DATASET_DIR:-$MV3DT_DIR/datasets/mtmc_6cam}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$MV3DT_DIR/experiments/deepstream/6cam}"
MODEL_REPO="${MODEL_REPO:-$MV3DT_DIR/models}"

# Set correct GPU flag considering diffent platforms
if docker info | grep -q 'Runtimes.*nvidia'; then
    GPU_FLAG="--runtime=nvidia"
elif docker run --help | grep -q -- "--gpus"; then
    GPU_FLAG="--gpus all"
else
    echo "No GPU support found in Docker."
    exit 1
fi

# Create directories for output
mkdir -p "$EXPERIMENT_DIR/infer-kitti-dump"
mkdir -p "$EXPERIMENT_DIR/tracker-kitti-dump"
mkdir -p "$EXPERIMENT_DIR/outVideos"

# Auto-generate DeepStream configuration files with custom 2D tracker config
source mv3dt_venv/bin/activate
python -m utils.deepstream_auto_configurator \
    --profile=mtmc_12cam_pnt \
    --dataset-dir="$DATASET_DIR" \
    --enable-msg-broker \
    --enable-osd \
    --tracker-config=config_tracker_2d.yml \
    --output-dir="$EXPERIMENT_DIR"

# Launch real-time BEV visualization
python utils/kafka_bev_visualizer.py \
    --dataset-path="$DATASET_DIR" \
    --msgconv-config="$EXPERIMENT_DIR/config_msgconv.txt" \
    --average-multi-cam \
    --show-ids &

# Launch MV3DT pipeline with custom 2D tracker config
docker run -t --privileged --rm --net=host $GPU_FLAG \
    -v "$MODEL_REPO:/workspace/models" \
    -v "$DATASET_DIR:/workspace/inputs" \
    -v "$EXPERIMENT_DIR:/workspace/experiments" \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /workspace/experiments \
    ${DEEPSTREAM_IMAGE:-ghcr.io/nvidia/deepstream:9.2-triton-multiarch} \
    deepstream-test5-app -c config_deepstream.txt
