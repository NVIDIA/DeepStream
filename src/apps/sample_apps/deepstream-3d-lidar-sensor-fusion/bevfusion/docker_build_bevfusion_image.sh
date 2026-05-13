#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# This script should be run inside deepstream-triton(7.0+) container which it is lanuched with
#  docker socket bindings, host network bindings, ipc host binding, gpu runtime
# An example to start deepstream-triton docker container
# $ xhost +
# $ docker run --gpus all -it --rm --net=host --privileged \
#  -v /var/run/docker.sock:/var/run/docker.sock-v /tmp/.X11-unix:/tmp/.X11-unix \
#  -e DISPLAY=$DISPLAY nvcr.io/nvidia/deepstream:{xx.xx.xx}-triton-multiarch
#
# Usage:
#       bevfusion/docker_build_bevfusion_image.sh nvcr.io/nvidia/deepstream:{xx.xx.xx}-triton-multiarch [TARGET_IMAGE]
#       TARGET_IMAGE: user defined the target image name, default: deepstream-triton-bevfusion

set -e

# debug
# set -x

# update BASE_IMAGE to deepstream triton release version, minumum version: 7.0.0
# BASE_IMAGE=nvcr.io/nvidia/deepstream:{xx.xx.xx}-triton-multiarch
BASE_IMAGE="nvcr.io/nvidia/deepstream:7.1-triton-multiarch"
version=$(basename $(realpath /opt/nvidia/deepstream/deepstream) | awk -F'-' '{print $2}')
TARGET_IMAGE="deepstream-triton-bevfusion:${version}"
WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
COMMIT_ID=840392ade92dc75ae513ef829173d5e27ce33b70
SPCONV_CUDA_VERSION=13.0

if [[ $# -ge 1 ]]; then
    BASE_IMAGE=$1
fi

if [[ $# -ge 2 ]]; then
    TARGET_IMAGE=$2
fi

if [[ $# -ge 3 ]]; then
    COMMIT_ID=$3
fi

if [[ $# -ge 4 ]]; then
    SPCONV_CUDA_VERSION=$4
fi

echo "with BASE_IMAGE: ${BASE_IMAGE}"
echo "build bevfusion TARGET_IMAGE: ${TARGET_IMAGE}"
echo "with COMMIT_ID: ${COMMIT_ID}"
echo "with SPCONV_CUDA_VERSION: ${SPCONV_CUDA_VERSION}"

# install docker command if it does not exist
if ! command -v docker &> /dev/null; then
    # Install Docker CLI
    echo "Installing docker.cli to build docker inside container"
    apt update && apt-get install -y docker.io || { echo "installing docker.io failed"; exit 1; }
fi

# docker pull base image if it does not exist
if docker image inspect "$BASE_IMAGE" &> /dev/null; then
    echo "BASE_IMAGE: $BASE_IMAGE exist"
else
    echo "BASE_IMAGE: $BASE_IMAGE is not found, trying to pull"
    docker pull "$BASE_IMAGE"
fi

# get GPU 0 compute capbility, e.g. RTX4090, CUDASM=89
CUDASM=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader,nounits | sed 's/\.//')

# build bevfusion docker image inside deepstream triton container
(cd bevfusion && \
    docker build \
    --network="host" \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg WORKSPACE=${WORKSPACE} \
    --build-arg CUDASM=${CUDASM} \
    --build-arg COMMIT_ID=${COMMIT_ID} \
    --build-arg SPCONV_CUDA_VERSION=${SPCONV_CUDA_VERSION} \
    -f bevfusion.Dockerfile \
    -t ${TARGET_IMAGE} ./ || \
    (echo "build TARGET_IMAGE: $TARGET_IMAGE failed"; exit 1))
echo "built TARGET_IMAGE: ${TARGET_IMAGE} successfully!"
