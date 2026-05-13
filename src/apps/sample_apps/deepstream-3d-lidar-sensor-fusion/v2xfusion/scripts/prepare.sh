#!/usr/bin/env bash
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

set -e

max_batch=4
plugins=/opt/nvidia/deepstream/deepstream/lib/libnvds_3d_v2x_infer_custom_preprocess.so

MODEL_NAME=v2xfusionmodel-int8-sparsity-seq
DATASET=v2x-seq.4scenes.10Hz.200frame
MODEL_DOWNLOAD_URL=https://nvidia.box.com/shared/static/xqj7ob2sa3betojf1084juyrlr1eek1a

function echoRed() {
    echo -e "\033[31m $1 \033[0m"
}

function echoGreen() {
    echo -e "\033[32m $1 \033[0m"
}

function prepareEngine() {
    if [ ! -f ../models/v2xfusion/${MODEL_NAME}.onnx ];then
        echoGreen "Download ${MODEL_NAME}.onnx"
        wget ${MODEL_DOWNLOAD_URL} -O ../models/v2xfusion/${MODEL_NAME}.onnx
    fi

    if [ ! -d ../models/v2xfusion/1/ ];then
        mkdir -p ../models/v2xfusion/1/
    fi

    echoGreen "Download success. Build ${MODEL_NAME}.engine. Please wait a moment..."
    /usr/src/tensorrt/bin/trtexec --onnx=../models/v2xfusion/${MODEL_NAME}.onnx --plugins=${plugins} --skipInference \
        --fp16 --sparsity=enable --dumpLayerInfo --exportLayerInfo=../models/v2xfusion/${MODEL_NAME}.layer.json \
        --saveEngine=../models/v2xfusion/1/${MODEL_NAME}.engine \
        --minShapes=images:1x3x864x1536,feats:1x8000x10x9,coords:1x8000x4,N:1x1,intervals:1x10499x3,geometry:1x1086935,num_intervals:1x1 \
        --optShapes=images:${max_batch}x3x864x1536,feats:${max_batch}x8000x10x9,coords:${max_batch}x8000x4,N:${max_batch}x1,intervals:${max_batch}x10499x3,geometry:${max_batch}x1086935,num_intervals:${max_batch}x1 \
        --maxShapes=images:${max_batch}x3x864x1536,feats:${max_batch}x8000x10x9,coords:${max_batch}x8000x4,N:${max_batch}x1,intervals:${max_batch}x10499x3,geometry:${max_batch}x1086935,num_intervals:${max_batch}x1 \
        --inputIOFormats=fp16:chw,fp16:chw,int32:chw,int32:chw,int32:chw,int32:chw,int32:chw --verbose > ../models/v2xfusion/${MODEL_NAME}.log 2>&1 || \
        (echoRed "Build ${MODEL_NAME}.engine failed, Please check ../models/v2xfusion/${MODEL_NAME}.log"; exit 1)

    echoGreen "Moving the generated model engine file and configs to triton_model_repo"
    rm -rf /opt/nvidia/deepstream/deepstream/samples/triton_model_repo/v2xfusion
    mkdir -p /opt/nvidia/deepstream/deepstream/samples/triton_model_repo/v2xfusion/1
    mv ../models/v2xfusion/1/*.engine /opt/nvidia/deepstream/deepstream/samples/triton_model_repo/v2xfusion/1
    cp ../models/v2xfusion/config.pbtxt /opt/nvidia/deepstream/deepstream/samples/triton_model_repo/v2xfusion/
}

function prepareDataSet() {
    if [ ! -f V2X-Seq-SPD-Example.zip ];then
        # https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file#dataset
        echoRed "Please download V2X-Seq-SPD-Example.zip from https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file#dataset"
        # gdown 1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY
        exit 1
    fi

    echoGreen "Unzip V2X-Seq-SPD-Example.zip"
    unzip -oq V2X-Seq-SPD-Example.zip

    if [ ! -d ../example-data/${DATASET} ];then
        mkdir -p ../example-data/${DATASET}
    fi

    echoGreen "Precompute tensor"
    python3 v2x_data_prepare.py ./V2X-Seq-SPD-Example ../example-data/${DATASET} || \
        (echoRed "Precompute tensor failed"; exit 1)
    rm -rf V2X-Seq-SPD-Example
}

case "$1" in
    "")
        prepareDataSet
        prepareEngine
    ;;
    "engine")
        prepareEngine
    ;;
    "dataset")
        prepareDataSet
    ;;
    *)
        echo "Usage:"
        echo "  ./prepare.sh [engine|dataset]"
    ;;
esac
