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

# usage bevfusion/docker_run_generate_trt_engine_models.sh {MODEL_ROOT}

set -e

# debug
# set -x

NVDS_VERSION=${NVDS_VERSION:-9.1}
DEFAULT_TARGET_IMAGE="deepstream-triton-bevfusion:${NVDS_VERSION}"

function usage() {
    echo "usage: bevfusion/docker_run_generate_trt_engine_models.sh {HOST_MODEL_ROOT} {DOCKER_IMAGE}"
    echo "It requires mounting local folder {HOST_MODEL_ROOT} into docker container"
    echo "to download and build TensorRT engine files before running triton server"
    echo ""
    echo "HOST_MODEL_ROOT, required, a local folder to store generated models"
    echo "DOCKER_IMAGE: default is ${DEFAULT_TARGET_IMAGE}, it's built by"
    echo "bevfusion/docker_build_bevfusion_image.sh"
}

TARGET_WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
TARGET_MODEL_ROOT=${TARGET_WORKSPACE}/bevfusion/model_root
TARGET_IMAGE=${DEFAULT_TARGET_IMAGE}

# set host model_root
HOST_MODEL_ROOT=bevfusion/model_root
if [[ $# -ge 1 ]]; then
    HOST_MODEL_ROOT=$1
else
    usage
    exit 1
fi

if [[ $# -ge 2 ]]; then
    TARGET_IMAGE=$2
fi

mkdir -p ${HOST_MODEL_ROOT} || (echo "failed to create mount folder: ${HOST_MODEL_ROOT}"; exit 1)
ABS_HOST_MODEL_ROOT=$(realpath ${HOST_MODEL_ROOT})

# docker run volumn mount options
MOUNT_OPTIONS="-v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "

# Mount whole deepstream-3d-lidar-sensor-fusion directory to target directory
# Note: enable if users want to validate local code/config changes
#MOUNT_OPTIONS+=" -v ./:${TARGET_WORKSPACE}"

# mount local dir model_root into container
MOUNT_OPTIONS+=" -v ${ABS_HOST_MODEL_ROOT}:${TARGET_MODEL_ROOT}"

# generate TRT engine files into local model_root directory
docker run \
    --gpus all --rm --net=host \
  ${MOUNT_OPTIONS} \
  --sig-proxy --privileged \
  ${TARGET_IMAGE} \
  bevfusion/build_bevfusion_model.sh ${TARGET_MODEL_ROOT} \
  || (echo "failed to build models into local folder: ${HOST_MODEL_ROOT}"; exit 1)

# list TRT models and ONNX models
echo "list the models and engine files:"
find ${HOST_MODEL_ROOT} \( -name "*.plan" -o -name "*.onnx" \)
