#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# usage bevfusion/docker_run_triton_server_bevfusion.sh {MODEL_ROOT}

set -e

# debug
# set -x

version=$(basename $(realpath /opt/nvidia/deepstream/deepstream) | awk -F'-' '{print $2}')
DEFAULT_TARGET_IMAGE="deepstream-triton-bevfusion:${version}"
DEFAULT_MODEL_ROOT=bevfusion/model_root

function usage() {
    echo "usage: bevfusion/docker_run_triton_server_bevfusion.sh {HOST_MODEL_ROOT}"
    echo "DEFAULT_MODEL_ROOT, optional, default is ${DEFAULT_MODEL_ROOT}"
    echo "DOCKER_IMAGE: ${DEFAULT_TARGET_IMAGE}, it's built by"
    echo "bevfusion/docker_build_bevfusion_image.sh"
}

TARGET_WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
TARGET_MODEL_ROOT=${TARGET_WORKSPACE}/bevfusion/model_root
TARGET_IMAGE=${DEFAULT_TARGET_IMAGE}

# host model root
HOST_MODEL_ROOT=${DEFAULT_MODEL_ROOT}
if [[ $# -ge 1 ]]; then
    HOST_MODEL_ROOT=$1
else
    usage
    exit 1
fi

# docker image
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
# Note: enable if users want to validate local code/config changes (globally or only python)
#MOUNT_OPTIONS+=" -v ./:${TARGET_WORKSPACE}"
MOUNT_OPTIONS+=" -v ./python:${TARGET_WORKSPACE}/python"

# mount local pre-built model_root into container
MOUNT_OPTIONS+=" -v ${ABS_HOST_MODEL_ROOT}:${TARGET_MODEL_ROOT}"

# starting tritonserver for bevfusion model inference
# the default grpc address, localhost:8001
echo "Starting tritonserver for bevfusion."
docker run \
    -it --gpus all --rm --net=host --ipc=host \
    ${MOUNT_OPTIONS} \
    -w ${TARGET_WORKSPACE}/python \
    -e PYTHONPATH=${TARGET_WORKSPACE}/python:${TARGET_WORKSPACE}/bevfusion/Lidar_AI_Solution/CUDA-BEVFusion/build: \
    -e BEVFUSION_MODEL=${TARGET_MODEL_ROOT}/model/resnet50int8 \
    -e BEVFUSION_PRECISION=int8 \
    --sig-proxy \
    --privileged \
    ${TARGET_IMAGE} \
    tritonserver --model-repository=${TARGET_WORKSPACE}/python/triton_lmm/triton_model_repo \
  || (echo "failed to start triton server : ${HOST_MODEL_ROOT}"; exit 1)
