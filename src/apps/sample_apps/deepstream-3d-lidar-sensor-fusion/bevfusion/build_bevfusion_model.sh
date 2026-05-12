#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# usage bevfusion/build_bevfusion_model.sh {MODEL_ROOT} {MODEL_ID}

set -e

# debug
# set -x

function usage {
    echo "usage: bevfusion/build_bevfusion_model.sh MODEL_ROOT"
    echo "  MODEL_ROOT: default value, bevfusion/model_root"
}

export WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
MODEL_ROOT=${WORKSPACE}/bevfusion/model_root
if [[ $# -ge 1 ]]; then
    MODEL_ROOT=$1
fi

BEVFUSION_MODEL=${MODEL_ROOT}/model/resnet50int8
BEVFUSION_PRECISION=int8

TensorRT_Bin=/usr/src/tensorrt/bin/trtexec
if [ ! -f "${TensorRT_Bin}" ]; then
    echo "Can not find ${TensorRT_Bin}, there may be a mistake in the directory you configured."
    exit 1
fi

pip3 install onnx --break-system-packages

# download bevfusion models
(cd ${MODEL_ROOT} && \
    (wget https://nvidia.box.com/shared/static/t1t67uytge97tn4q7xteehtauutaqxz7 -O model.zip || true))

cd ${MODEL_ROOT}
if ! unzip -o model.zip; then
    echo "Can not find bevfusion/model_root/model.zip, meet error while downloading models."
    exit 1
fi

model_dir=${BEVFUSION_MODEL}
precision=${BEVFUSION_PRECISION}

function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    result_save_directory=$model_dir/build
    onnx=$model_dir/$name.onnx

    if [ -f "${result_save_directory}/$name.plan" ]; then
        echo Model ${result_save_directory}/$name.plan is already there, skipping the build.
        return 0
    fi

    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    # echo $number_of_input  $number_of_output

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    input_flags=${input_flags%?}
    output_flags=${output_flags%?}

    cmd="--onnx=$model_dir/$name.onnx ${precision_flags} ${input_flags} ${output_flags} \
        --saveEngine=${result_save_directory}/$name.plan \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"

    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment~.
    ${TensorRT_Bin} $cmd > ${result_save_directory}/$name.log 2>&1
    if [ $? != 0 ]; then
        echo Failed to build model ${result_save_directory}/$name.plan.
        echo You can check the error message by ${result_save_directory}/$name.log
        exit 1
    fi
    return 0
}

trtexec_dynamic_flags="--fp16 --int8"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

trtexec_fp16_flags="--fp16"

cd ${MODEL_ROOT}
compile_trt_model "camera.backbone" "$trtexec_dynamic_flags" 2 2 || exit 1
compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1 || exit 1

# fp16 only
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1 || exit 1
compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6 || exit 1
