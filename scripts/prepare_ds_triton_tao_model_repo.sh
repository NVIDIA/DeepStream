#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script downloads NVIDIA TAO Toolkit pre-trained models available on NGC
# and converts them to TRT engine files for adding to Triton Inference Server
# model repository.
# If required the script downloads the tao-converter tool and installs it in
# $HOME/bin/tao-converter directory.
# While generating the models, batch size of 16 is used for dGPU and 1 for Jetson
# platforms.

set -e

TRITON_REPO_DIR="/opt/nvidia/deepstream/deepstream/samples/triton_tao_model_repo"
DOWNLOAD_DIR="/tmp/tao_models"
export PATH=$PATH:/usr/src/tensorrt/bin
MODEL_REPO_DIR=$TRITON_REPO_DIR
WITH_INT8=false

if [[ $(uname -m) == "aarch64" ]]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=16
fi

if [[ -z "${TRTEXEC_BIN}" ]]; then
    TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
fi

if [[ ! -f "${TRTEXEC_BIN}" ]]; then
    echo "trtexec binary not found. Set TRTEXEC_BIN"
fi

function updateModelConfig {
    ModelName="$1"
    ConfigFile="$2"
    sed -i -e "s|default_model_filename.*|default_model_filename:\"$ModelName\"|" ${ConfigFile}
}

function buildEngineFromOnnx {
    ModelDir="$1"
    MaxBatch="$2"
    Dimensions="$3"

    if [[ $# -ne 3 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    CalibFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/cal_trt.bin)
    OnnxModelFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/*.onnx)
    ModelFileName=$(basename "${OnnxModelFile}")

    mkdir -p "${MODEL_REPO_DIR}/${ModelDir}/1/"

    if [[ "${WITH_INT8}" = true ]]; then
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_int8.engine"
    else
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp16.engine"
    fi

    TRTEXEC_CMD="${TRTEXEC_BIN} \
        --onnx=${OnnxModelFile} \
        --calib=${CalibFile} \
        --saveEngine=${EngineFile} \
        --minShapes="input_1:0":1x${Dimensions} \
        --optShapes="input_1:0":${MaxBatch}x${Dimensions} \
        --maxShapes="input_1:0":${MaxBatch}x${Dimensions} \
        --skipInference"

    if [[ "${WITH_INT8}" = true ]]; then
        TRTEXEC_CMD="${TRTEXEC_CMD} --int8"
    else
        TRTEXEC_CMD="${TRTEXEC_CMD} --fp16"
    fi

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
    if eval "${TRTEXEC_CMD}" >> "${LOG_FILE}" 2>&1 ; then
        echo "Finished building Model ${ModelDir}"
        rm "${LOG_FILE}"
    else
        echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information."
        return 1;
    fi

    if [[ "${WITH_INT8}" != true ]]; then
        EngineFileName=$(basename "${EngineFile}")
        updateModelConfig ${EngineFileName} ${MODEL_REPO_DIR}/${ModelDir}/config.pbtxt
    fi
}

if buildEngineFromOnnx "Primary_Detector" 1 3x544x960 > /dev/null; then
    echo "Platform supports INT8. Generating engine files using INT8 mode."
else
    echo "Platform does not support INT8. Generating engine files using FP16 mode."
    WITH_INT8=false
fi

buildEngineFromOnnx "Primary_Detector" 30 3x544x960 || exit 1

echo "Downloading PeopleNet Transformer model from NGC TAO repository"

MODEL_REPO_DIR=$TRITON_REPO_DIR/peoplenet_transformer
MODEL_DOWNLOAD_DIR=$DOWNLOAD_DIR/peoplenet_transformer

mkdir -p $MODEL_DOWNLOAD_DIR

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer/deployable_v1.1/files?redirect=true&path=resnet50_peoplenet_transformer_op17.onnx' \
 -O $MODEL_DOWNLOAD_DIR/resnet50_peoplenet_transformer_op17.onnx
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O $MODEL_REPO_DIR/labels.txt

echo "Creating TensorRT engine file for PeopleNet Transformer"
echo "Setting batch size to $BATCH_SIZE"
CONFIG_FILE=$MODEL_REPO_DIR/config.pbtxt
sed -i -e "s|max_batch_size.*|max_batch_size: $BATCH_SIZE|" $CONFIG_FILE

mkdir -p $MODEL_REPO_DIR/1

trtexec --onnx=$MODEL_DOWNLOAD_DIR/resnet50_peoplenet_transformer_op17.onnx --fp16 \
 --saveEngine=$MODEL_REPO_DIR/1/model.plan \
 --minShapes="inputs":1x3x544x960 --optShapes="inputs":${BATCH_SIZE}x3x544x960 --maxShapes="inputs":${BATCH_SIZE}x3x544x960
echo "Downloading PeopleSemSegNet Shuffle model from NGC TAO repository"

MODEL_REPO_DIR=$TRITON_REPO_DIR/peoplesemsegnet_shuffle
MODEL_DOWNLOAD_DIR=$DOWNLOAD_DIR/peoplesemsegnet_shuffle

mkdir -p $MODEL_DOWNLOAD_DIR

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0.1/files?redirect=true&path=labels.txt' -O $MODEL_REPO_DIR/labels.txt && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0.1/files?redirect=true&path=peoplesemsegnet_shuffleseg.onnx' -O $MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg.onnx && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0.1/files?redirect=true&path=peoplesemsegnet_shuffleseg_int8.txt' -O $MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg_cache.txt

echo "Setting batch size to $BATCH_SIZE"
CONFIG_FILE=$MODEL_REPO_DIR/config.pbtxt
sed -i -e "s|max_batch_size.*|max_batch_size: $BATCH_SIZE|" $CONFIG_FILE

mkdir -p $MODEL_REPO_DIR/1

trtexec --onnx=$MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg.onnx --int8 \
 --calib=$MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg_cache.txt --saveEngine=$MODEL_REPO_DIR/1/model.plan \
 --minShapes="input_2:0":1x3x544x960 --optShapes="input_2:0":${BATCH_SIZE}x3x544x960 --maxShapes="input_2:0":${BATCH_SIZE}x3x544x960

echo "Deleting the downloaded model files in $DOWNLOAD_DIR."
rm -rf $DOWNLOAD_DIR

echo "Model repository prepared successfully."
