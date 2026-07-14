#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

MODEL_REPO_DIR="/opt/nvidia/deepstream/deepstream/samples/triton_model_repo"
WITH_INT8=false

if [[ -z "${TRTEXEC_BIN}" ]]; then
    TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
fi

if [[ ! -f "${TRTEXEC_BIN}" ]]; then
    echo "trtexec binary not found. Set TRTEXEC_BIN"
fi

if [[ $(uname -m) == "aarch64" ]]; then
  export TRT_LIB_PATH=/usr/lib/aarch64-linux-gnu
  export TRT_INC_PATH=/usr/include/aarch64-linux-gnu
else
  export TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
  export TRT_INC_PATH="/usr/include/x86_64-linux-gnu"
fi

function updateModelConfig {
    ModelName="$1"
    ConfigFile="$2"
    sed -i -e "s|default_model_filename.*|default_model_filename:\"$ModelName\"|" ${ConfigFile}
}

function buildEngineFromCaffe {
    ModelDir="$1"
    MaxBatch="$2"
    OutputLayers="$3"

    if [[ $# -ne 3 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    CalibFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/cal_trt.bin)
    ProtoFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/*.prototxt)
    ModelFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/*.caffemodel)
    ModelFileName=$(basename "${ModelFile}")

    if [[ "${WITH_INT8}" = true ]]; then
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_int8.engine"
    else
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp16.engine"
    fi

    mkdir -p $(dirname ${EngineFile})

    TRTEXEC_CMD="${TRTEXEC_BIN} \
        --calib=${CalibFile} \
        --deploy=${ProtoFile} \
        --model=${ModelFile} \
        --maxBatch=${MaxBatch} \
        --saveEngine=${EngineFile} \
        --buildOnly"

    for layer in ${OutputLayers}; do
        TRTEXEC_CMD="${TRTEXEC_CMD} --output=${layer}"
    done

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

function build_engine_from_etlt {
    ModelDir="$1"
    MaxBatch="$2"
    TltModelKey="$3"
    OutputLayers="$4"
    Dimensions="$5"

    if [[ $# -ne 5 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    CalibFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/cal_trt.bin)
    TltEncodedModel=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/*.etlt)
    ModelFileName=$(basename "${TltEncodedModel}")

    mkdir -p "${MODEL_REPO_DIR}/${ModelDir}/1/"

    if [[ "${WITH_INT8}" = true ]]; then
        ModelFormat="int8"
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_int8.engine"
    else
        ModelFormat="fp16"
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp16.engine"
    fi

    TAO_CONVERTER_CMD="${TAO_CONVERTER_BIN} \
                ${TltEncodedModel} \
                -c ${CalibFile} \
                -k ${TltModelKey} \
                -b ${MaxBatch} \
                -m ${MaxBatch} \
                -e ${EngineFile}\
                -o ${OutputLayers} \
                -d ${Dimensions} \
                -t ${ModelFormat}"

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
    if eval "${TAO_CONVERTER_CMD}" >> "${LOG_FILE}" 2>&1 ; then
        echo "Finished building Model ${ModelDir}"
        rm "${LOG_FILE}"
    else
        echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information." >&2
        return 1;
    fi

    if [[ "${WITH_INT8}" != true ]]; then
        EngineFileName=$(basename "${EngineFile}")
        updateModelConfig ${EngineFileName} ${MODEL_REPO_DIR}/${ModelDir}/config.pbtxt
    fi

}

function buildEngineFromUff {
    ModelDir="$1"
    MaxBatch="$2"
    InputLayer="$3"
    OutputLayers="$4"
    ModelServerDir="$5"

    if [[ $# -ne 5 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    UffFile=$(ls "/opt/nvidia/deepstream/deepstream/samples/models/${ModelDir}"/*.uff)
    ModelFileName=$(basename "${UffFile}")
    EngineFile="${MODEL_REPO_DIR}/${ModelServerDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp32.engine"

    mkdir -p $(dirname ${EngineFile})

    TRTEXEC_CMD="${TRTEXEC_BIN} \
        --uff=${UffFile} \
        --maxBatch=${MaxBatch} \
        --saveEngine=${EngineFile} \
        --uffInput=${InputLayer} \
        --buildOnly"

    for layer in ${OutputLayers}; do
        TRTEXEC_CMD="${TRTEXEC_CMD} --output=${layer}"
    done

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
}

function build_engine_from_onnx {
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
        echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information." >&2
        return 1;
    fi

    if [[ "${WITH_INT8}" != true ]]; then
        EngineFileName=$(basename "${EngineFile}")
        updateModelConfig ${EngineFileName} ${MODEL_REPO_DIR}/${ModelDir}/config.pbtxt
    fi
}

echo "Generating Engine files for ONNX provided with the SDK"

echo "Checking for INT8 support..."
if build_engine_from_onnx "Primary_Detector" 1 3x544x960 > /dev/null; then
    echo "Platform supports INT8. Generating engine files using INT8 mode."
else
    echo "Platform does not support INT8. Generating engine files using FP16 mode."
    WITH_INT8=false
fi

build_engine_from_onnx "Primary_Detector" 30 3x544x960 || exit 1
build_engine_from_onnx "Secondary_VehicleMake" 16 3x224x224 || exit 1
build_engine_from_onnx "Secondary_VehicleTypes" 16 3x224x224 || exit 1

# echo "Generating Engine files for segmentation UFF models provided with the SDK"

# WAR: Remove the segmentation repo
rm -rf ${MODEL_REPO_DIR}/Segmentation* # Needs to be removed once ONNX model is available
# Disabling the segmentation Uff model
# buildEngineFromUff "Segmentation/semantic" 1 "data,3,512,512" "final_conv/BiasAdd" "Segmentation_Semantic" || exit 1
# buildEngineFromUff "Segmentation/industrial" 1 "input_1,1,512,512" "conv2d_19/Sigmoid" "Segmentation_Industrial" || exit 1

echo "Finished generating engine files."

echo "Downloading models from the TensorRT inference server samples"

# Refer to https://github.com/NVIDIA/tensorrt-inference-server/blob/r20.01/docs/examples/fetch_models.sh
# for the source of the models.

Gpu0Instance=$(cat <<-'EOL'

instance_group {
  count: 1
  gpus: 0
  kind: KIND_GPU
}
EOL
)

echo "Adding read permission for downloaded models"
find ${MODEL_REPO_DIR} -name '*.pb' -o -name 'model.*' -o -name '*.engine' | xargs -r chmod a+r

echo "Model repository prepared successfully."
