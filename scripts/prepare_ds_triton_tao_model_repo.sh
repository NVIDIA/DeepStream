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

set -e

SAMPLES_DIR="${DS_SAMPLES_ROOT:-/opt/nvidia/deepstream/deepstream/samples}"
TRITON_REPO_DIR="${SAMPLES_DIR}/triton_tao_model_repo"
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

TRT_MAJOR=$(python3 - <<'PY'
import ctypes
print(ctypes.CDLL("libnvinfer.so").getInferLibVersion() // 10000)
PY
)
if (( TRT_MAJOR >= 11 )); then
    TRTEXEC_BIN="${NVDS_ENGINE_BUILDER:-/opt/nvidia/deepstream/deepstream/bin/nvdsinfer-engine-builder}"
    command -v "$TRTEXEC_BIN" >/dev/null || { echo "Missing SDK engine builder: $TRTEXEC_BIN" >&2; exit 1; }
    # This model is not one of the three sample models handled by the SDK's
    # post-install helper. Prepare it with real application inputs (see
    # samples/modelopt/README.md), or obtain a qualified typed export.
    # Do not download an FP32 graph and assume --fp16 changes its types in TRT11.
    : "${PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH:?TensorRT11 requires a separately prepared mixed-FP16 PeopleNet Transformer ONNX; set PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH and preserve its original FP32 external I/O}"
    : "${SHUFFLESEG_TRT11_QDQ_MODEL_PATH:?TensorRT11 INT8 requires a genuine Q/DQ ShuffleSeg export; legacy calibration caches cannot be converted automatically}"
    if [[ ! -f "$PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH" || ! -r "$PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH" ]]; then
        echo "Prepared PeopleNet Transformer model is not readable: $PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH" >&2
        exit 1
    fi
    # This script removes its download scratch directory on success. A caller's
    # prepared input must never be placed under that disposable tree, including
    # through a symlink or relative-path alias.
    PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH=$(realpath -- "$PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH")
    DOWNLOAD_REAL_PATH=$(realpath -m -- "$DOWNLOAD_DIR")
    if [[ "$PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH" == "$DOWNLOAD_REAL_PATH" || "$PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH" == "$DOWNLOAD_REAL_PATH/"* ]]; then
        echo "Prepared PeopleNet Transformer input must be outside the disposable download directory: $DOWNLOAD_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$SHUFFLESEG_TRT11_QDQ_MODEL_PATH" || ! -r "$SHUFFLESEG_TRT11_QDQ_MODEL_PATH" ]]; then
        echo "Prepared ShuffleSeg model is not readable: $SHUFFLESEG_TRT11_QDQ_MODEL_PATH" >&2
        exit 1
    fi
    SHUFFLESEG_TRT11_QDQ_MODEL_PATH=$(realpath -- "$SHUFFLESEG_TRT11_QDQ_MODEL_PATH")
    if [[ "$SHUFFLESEG_TRT11_QDQ_MODEL_PATH" == "$DOWNLOAD_REAL_PATH" || "$SHUFFLESEG_TRT11_QDQ_MODEL_PATH" == "$DOWNLOAD_REAL_PATH/"* ]]; then
        echo "Prepared ShuffleSeg input must be outside the disposable download directory: $DOWNLOAD_DIR" >&2
        exit 1
    fi
    # Do not recursively remove the shared legacy scratch tree: an external
    # model outside it may still reference caller-owned weights inside it.
    # Each TRT11 invocation downloads and cleans only its newly owned directory.
    DOWNLOAD_DIR=$(mktemp -d "${DOWNLOAD_DIR}.XXXXXX")
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

    CalibFile=""
    if (( TRT_MAJOR < 11 )); then CalibFile=$(ls "${SAMPLES_DIR}/models/${ModelDir}"/cal_trt.bin); fi
    # Select the canonical name, not the fp32_ backup or a legacy derived file.
    # Post-install ModelOpt setup replaces this file without changing its name.
    OnnxModelFile="${SAMPLES_DIR}/models/${ModelDir}/resnet18_trafficcamnet_pruned.onnx"
    if [[ ! -f "$OnnxModelFile" || ! -r "$OnnxModelFile" ]]; then
        echo "Missing canonical model: $OnnxModelFile. Install the matching SDK sample models." >&2
        echo "For TRT11, run samples/modelopt/prepare_samples.py first; model filenames remain unchanged." >&2
        return 1
    fi
    ModelFileName=$(basename "${OnnxModelFile}")

    mkdir -p "${MODEL_REPO_DIR}/${ModelDir}/1/"

    if [[ "${WITH_INT8}" = true ]]; then
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_int8.engine"
    else
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp16.engine"
    fi

    local -a TRTEXEC_CMD=("${TRTEXEC_BIN}"
        "--onnx=${OnnxModelFile}" "--saveEngine=${EngineFile}"
        "--minShapes=input_1:0:1x${Dimensions}"
        "--optShapes=input_1:0:${MaxBatch}x${Dimensions}"
        "--maxShapes=input_1:0:${MaxBatch}x${Dimensions}" --skipInference)
    if (( TRT_MAJOR < 11 )); then TRTEXEC_CMD+=("--calib=${CalibFile}"); fi

    if [[ "${WITH_INT8}" = true ]]; then
        TRTEXEC_CMD+=(--int8)
    else
        TRTEXEC_CMD+=(--fp16)
    fi

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
    if "${TRTEXEC_CMD[@]}" >> "${LOG_FILE}" 2>&1 ; then
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

if (( TRT_MAJOR >= 11 )); then
    echo "TensorRT11: building engines from typed FP16/mixed models at canonical sample filenames; no conversion or legacy calibration runs here."
elif buildEngineFromOnnx "Primary_Detector" 1 3x544x960 > /dev/null; then
    echo "Platform supports INT8. Generating engine files using INT8 mode."
else
    echo "Platform does not support INT8. Generating engine files using FP16 mode."
    WITH_INT8=false
fi

buildEngineFromOnnx "Primary_Detector" 30 3x544x960 || exit 1

echo "Preparing PeopleNet Transformer model"

MODEL_REPO_DIR=$TRITON_REPO_DIR/peoplenet_transformer
MODEL_DOWNLOAD_DIR=$DOWNLOAD_DIR/peoplenet_transformer

mkdir -p $MODEL_DOWNLOAD_DIR

if (( TRT_MAJOR >= 11 )); then
    PEOPLENET_TRANSFORMER_MODEL="$PEOPLENET_TRANSFORMER_TRT11_ONNX_MODEL_PATH"
else
    PEOPLENET_TRANSFORMER_MODEL="$MODEL_DOWNLOAD_DIR/resnet50_peoplenet_transformer_op17.onnx"
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer/deployable_v1.1/files?redirect=true&path=resnet50_peoplenet_transformer_op17.onnx' \
     -O "$PEOPLENET_TRANSFORMER_MODEL"
fi
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O $MODEL_REPO_DIR/labels.txt

echo "Creating TensorRT engine file for PeopleNet Transformer"
echo "Setting batch size to $BATCH_SIZE"
CONFIG_FILE=$MODEL_REPO_DIR/config.pbtxt
sed -i -e "s|max_batch_size.*|max_batch_size: $BATCH_SIZE|" $CONFIG_FILE

mkdir -p $MODEL_REPO_DIR/1

"$TRTEXEC_BIN" "--onnx=$PEOPLENET_TRANSFORMER_MODEL" --fp16 \
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

if (( TRT_MAJOR >= 11 )); then
  SEGMENTATION_FLAGS=("--onnx=$SHUFFLESEG_TRT11_QDQ_MODEL_PATH" --int8)
else
  SEGMENTATION_FLAGS=("--onnx=$MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg.onnx" --int8 "--calib=$MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg_cache.txt")
fi
"$TRTEXEC_BIN" "${SEGMENTATION_FLAGS[@]}" --saveEngine=$MODEL_REPO_DIR/1/model.plan \
 --minShapes="input_2:0":1x3x544x960 --optShapes="input_2:0":${BATCH_SIZE}x3x544x960 --maxShapes="input_2:0":${BATCH_SIZE}x3x544x960

echo "Deleting the downloaded model files in $DOWNLOAD_DIR."
if (( TRT_MAJOR >= 11 )); then
    rm -rf -- "$DOWNLOAD_DIR"
else
    rm -rf $DOWNLOAD_DIR
fi

echo "Model repository prepared successfully."
