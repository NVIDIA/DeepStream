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

# usage v2xfusion/scripts/docker_run_ds3d_sensor_fusion_v2x_pipeline.sh {CONFIGFILE}

set -e

# debug
# set -x

version=$(basename $(realpath /opt/nvidia/deepstream/deepstream) | awk -F'-' '{print $2}')
DEFAULT_CONFIG_FILE="ds3d_lidar_video_sensor_v2x_fusion.yaml"
TARGET_IMAGE="nvcr.io/nvidia/deepstream:${version}-triton-multiarch"
TARGET_DEVICE=$(uname -m)
TARGET_WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
TARGET_MODEL_ROOT=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo/v2xfusion

function usage() {
    echo "usage: v2xfusion/scripts/docker_run_ds3d_sensor_fusion_v2x_pipeline.sh {CONFIGFILE}"
    echo "CONFIGFILE, optional, default: ${DEFAULT_CONFIG_FILE}"
}

function echoRed() {
    echo -e "\033[31m $1 \033[0m"
}

CONFIG_FILE="ds3d_lidar_video_sensor_v2x_fusion.yaml"

if [[ $# -ge 1 ]]; then
    CONFIG_FILE=$1
else
    usage
    exit 1
fi

if [[ ! -d ${TARGET_MODEL_ROOT} ]];then
    echoRed "Please run sudo v2xfusion/scripts/docker_run_generate_v2x_trt_engine_model.sh to build model first"
    exit 1
fi

if [[ ! -d ${TARGET_WORKSPACE}/v2xfusion/example-data ]];then
    echoRed "Please download V2X-Seq-SPD-Example.zip from https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file#dataset"
    echoRed "Put it to v2xfusion/scripts"
    echoRed "Then cd v2xfusion/scripts;./prepare.sh dataset"
    exit 1
fi

APP="/opt/nvidia/deepstream/deepstream/bin/deepstream-3d-lidar-sensor-fusion"
APP_CMD="-c ${CONFIG_FILE}"

[ -z "$DISPLAY" ] && (echo "Please export correct DISPLAY before running the pipeline."; exit -1)
xhost +

MOUNT_OPTIONS="-v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "

# mount local sample data for test
MOUNT_OPTIONS+=" -v ${TARGET_WORKSPACE}/v2xfusion/example-data:${TARGET_WORKSPACE}/v2xfusion/example-data"
MOUNT_OPTIONS+=" -v ${TARGET_MODEL_ROOT}:${TARGET_MODEL_ROOT}"

DOCKER_GPU_ARG="--gpus all"
if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    DOCKER_GPU_ARG="--gpus all"
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    DOCKER_GPU_ARG="--runtime nvidia"
else
    echo "Unsupported platform ${TARGET_DEVICE}"
    exit -1
fi

echo "Starting deepstream-3d-lidar-sensor-fusion pipeline for v2xfusion."
docker run \
    ${DOCKER_GPU_ARG} --rm --net=host \
    ${MOUNT_OPTIONS} \
    -w ${TARGET_WORKSPACE} \
    -e DISPLAY=$DISPLAY \
    --sig-proxy \
    --privileged \
    --entrypoint ${APP} \
    ${TARGET_IMAGE} \
    ${APP_CMD}
