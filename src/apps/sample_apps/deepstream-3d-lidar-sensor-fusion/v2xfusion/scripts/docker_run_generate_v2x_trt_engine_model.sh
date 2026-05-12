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

# usage v2xfusion/scripts/docker_run_generate_v2x_trt_engine_model.sh

set -e

# debug
# set -x

function cleanup() {
  # delete temporary container
  docker rm ${CONTAINER_NAME}
}

trap cleanup EXIT

version=$(basename $(realpath /opt/nvidia/deepstream/deepstream) | awk -F'-' '{print $2}')
DEFAULT_IMAGE="nvcr.io/nvidia/deepstream:${version}-triton-multiarch"
CONTAINER_NAME="v2x-trt-model-builder"
TARGET_WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/v2xfusion/scripts
TARGET_MODEL_ROOT=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
TARGET_DEVICE=$(uname -m)

DOCKER_GPU_ARG="--gpus all"
if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    DOCKER_GPU_ARG="--gpus all"
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    DOCKER_GPU_ARG="--runtime nvidia"
else
    echo "Unsupported platform ${TARGET_DEVICE}"
    exit -1
fi

# delete old model
rm -rf ${TARGET_MODEL_ROOT}/v2xfusion

# generate TRT engine files
docker run ${DOCKER_GPU_ARG} \
          --name ${CONTAINER_NAME} \
          --net=host \
          --privileged \
          -w ${TARGET_WORKSPACE} \
          ${DEFAULT_IMAGE} \
          ./prepare.sh engine

# copy to host
docker cp -a ${CONTAINER_NAME}:${TARGET_MODEL_ROOT}/v2xfusion ${TARGET_MODEL_ROOT}
