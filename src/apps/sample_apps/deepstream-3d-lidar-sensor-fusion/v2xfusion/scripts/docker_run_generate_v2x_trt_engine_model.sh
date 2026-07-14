#!/bin/bash
################################################################################
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
################################################################################

# usage v2xfusion/scripts/docker_run_generate_v2x_trt_engine_model.sh

set -e

# debug
# set -x

function cleanup() {
  # delete temporary container
  docker rm ${CONTAINER_NAME}
}

trap cleanup EXIT

NVDS_VERSION=${NVDS_VERSION:-9.1}
DEFAULT_IMAGE="nvcr.io/nvidia/deepstream:${NVDS_VERSION}-triton-multiarch"
CONTAINER_NAME="v2x-trt-model-builder"
TARGET_WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/sample_apps/deepstream-3d-lidar-sensor-fusion
TARGET_MODEL_ROOT=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
TARGET_DEVICE=$(uname -m)

DOCKER_GPU_ARG="--gpus all"
if [[ "${TARGET_DEVICE}" = "x86_64" ]]; then
    DOCKER_GPU_ARG="--gpus all"
elif [[ "${TARGET_DEVICE}" = "aarch64" ]]; then
    DOCKER_GPU_ARG="--runtime nvidia"
else
    echo "Unsupported platform ${TARGET_DEVICE}"
    exit -1
fi

# delete old model
#rm -rf ../models/v2xfusion/1

# generate TRT engine files
docker run ${DOCKER_GPU_ARG} \
          --name ${CONTAINER_NAME} \
          --net=host \
          --privileged \
          -v $(pwd):${TARGET_WORKSPACE}/v2xfusion/scripts \
		  -v $(pwd)/../models:${TARGET_WORKSPACE}/v2xfusion/models \
		  -v ../../../../../utils/ds3d/inference_custom_lib/ds3d_v2x_infer_custom_preprocess/:${TARGET_WORKSPACE}/lib \
          -w ${TARGET_WORKSPACE}/v2xfusion/scripts \
          ${DEFAULT_IMAGE} \
          ./prepare.sh engine

# copy to host
docker cp -a ${CONTAINER_NAME}:${TARGET_MODEL_ROOT}/v2xfusion ./

