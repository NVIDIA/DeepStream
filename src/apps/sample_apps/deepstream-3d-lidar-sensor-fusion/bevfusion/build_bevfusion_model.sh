#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e -o pipefail

# debug
# set -x

function usage {
    echo "usage: bevfusion/build_bevfusion_model.sh MODEL_ROOT"
    echo "  MODEL_ROOT: default value, bevfusion/model_root"
    echo "  BEVFUSION_MODEL: optional directory containing separately qualified ONNX model chunks"
}

export WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
MODEL_ROOT=${WORKSPACE}/bevfusion/model_root
if [[ $# -ge 1 ]]; then
    MODEL_ROOT=$1
fi

# TRT11 users can supply separately prepared model chunks without overwriting
# the original downloaded models. camera.backbone/fuser must retain genuine INT8
# Q/DQ; camera.vtransform/head.bbox must already contain FP16 computation. Their
# public floating I/O must remain HALF as required by CUDA-BEVFusion below. The
# CPU AutoCast tool is not a quantizer or an automatic custom-op conversion path.
BEVFUSION_MODEL=${BEVFUSION_MODEL:-${MODEL_ROOT}/model/resnet50int8}
BEVFUSION_PRECISION=int8

TensorRT_Bin=/usr/src/tensorrt/bin/trtexec
TRT_MAJOR=$(python3 - <<'PY'
import ctypes
print(ctypes.CDLL("libnvinfer.so").getInferLibVersion() // 10000)
PY
)
if (( TRT_MAJOR >= 11 )); then
    TensorRT_Bin="${NVDS_ENGINE_BUILDER:-/opt/nvidia/deepstream/deepstream/bin/nvdsinfer-engine-builder}"
fi
if [[ ! -x "${TensorRT_Bin}" ]]; then
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

function model_dependency_identity(){
    # Hash the protobuf and every external TensorProto payload without loading
    # the tensors. Nested graphs, sparse tensors and local functions count too.
    python3 - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import onnx

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def tensors(message):
    if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            # Protobuf 4.x/5.x expose label, not is_repeated. Keep the fallback
            # lazy so newer runtimes need not provide the legacy label API.
            repeated = (field.is_repeated if hasattr(field, 'is_repeated')
                        else field.label == field.LABEL_REPEATED)
            for child in (value if repeated else [value]):
                yield from tensors(child)

try:
    source = Path(sys.argv[1]).resolve(strict=True)
    root = source.parent
    proto = onnx.load(str(source), load_external_data=False)
    identities = {'onnx': digest(source), 'external': {}}
    external_digests = {}
    for tensor in tensors(proto):
        if not tensor.external_data:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError('External tensor is missing its location')
            continue
        entries = {item.key: item.value for item in tensor.external_data}
        if len(entries) != len(tensor.external_data):
            raise ValueError('Duplicate external-data metadata')
        location = entries.get('location', '')
        relative = Path(location)
        if not location or relative.is_absolute() or '..' in relative.parts or '\\' in location:
            raise ValueError('Unsafe external-data location: ' + repr(location))
        target = (root / relative).resolve(strict=True)
        target.relative_to(root)
        if not target.is_file() or target == source:
            raise ValueError('Invalid external tensor payload: ' + location)
        size = target.stat().st_size
        offset = int(entries.get('offset', '0'))
        length = int(entries.get('length', str(size - offset)))
        if offset < 0 or length < 0 or offset + length > size:
            raise ValueError('External tensor payload exceeds file: ' + location)
        # Many tensors can share one large file. Validate each tensor's range,
        # but read each resolved payload only once in this identity check.
        if target not in external_digests:
            external_digests[target] = digest(target)
        identities['external'][relative.as_posix()] = {'sha256': external_digests[target], 'bytes': size}
    print(hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest())
except (OSError, ValueError) as error:
    print('[ERROR] Cannot verify ONNX dependency identity: ' + str(error), file=sys.stderr)
    raise SystemExit(1)
PY
}

function resolve_cuda_device_identity(){
    # The builder below uses CUDA logical ordinal 0, not physical NVML index 0.
    # UUIDv2 also distinguishes MIG instances; do not reuse if it is unavailable.
    python3 - <<'PY'
import ctypes
import sys
import uuid
try:
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    get_uuid = driver.cuDeviceGetUuid_v2
    get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    device = ctypes.c_int()
    value = (ctypes.c_ubyte * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0):
        raise RuntimeError('CUDA device 0 is unavailable')
    if get_uuid(ctypes.byref(value), device) or not any(value):
        raise RuntimeError('CUDA device UUID is unavailable')
    print('cuda-device-uuid:' + str(uuid.UUID(bytes=bytes(value))))
except (AttributeError, OSError, RuntimeError, ValueError) as error:
    print('[WARN] Engine cache reuse disabled: ' + str(error), file=sys.stderr)
    raise SystemExit(1)
PY
}

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

    if (( TRT_MAJOR < 11 )) && [[ -f "${result_save_directory}/$name.plan" ]]; then
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

    local identity="" identity_file="${result_save_directory}/$name.plan.mixed.identity"
    local device_identity="" model_identity="" runtime_identity="" engine_hash="" profile_flags="--dumpProfile --separateProfileRun"
    if (( TRT_MAJOR >= 11 )); then
        profile_flags="" # The SDK entry point builds engines; it is not a benchmark.
        # Use one canonical pathname for both external-data hashing and parsing.
        # Resolving only in the identity helper can select different sidecars
        # when an individual ONNX file is a symlink into another directory.
        onnx=$(realpath -e -- "$onnx") || return 1
        device_identity=$(resolve_cuda_device_identity) || device_identity=""
        model_identity=$(model_dependency_identity "$onnx") || return 1
        runtime_identity=$("$TensorRT_Bin" --version) || return 1
        identity=$({
            printf '%s\0' 'bevfusion-model-typed-v3-closure-cuda-uuid' "$device_identity" \
                "${CUDA_VISIBLE_DEVICES-unset}" "${CUDA_DEVICE_ORDER-unset}" "${NVIDIA_VISIBLE_DEVICES-unset}" \
                "$model_identity" "$runtime_identity" "$precision_flags" "$input_flags" "$output_flags"
        } | sha256sum | cut -d' ' -f1) || return 1
        if [[ -n "$device_identity" && -s "${result_save_directory}/$name.plan" && -f "$identity_file" ]]; then
            engine_hash=$(sha256sum -- "${result_save_directory}/$name.plan" | cut -d' ' -f1) || return 1
            if [[ "$(<"$identity_file")" == "$(printf '%s\n%s' "$identity" "$engine_hash")" ]]; then
                echo "Reusing TensorRT11 engine with matching model/device/precision identity: $name"
                return 0
            fi
        fi
    fi

    local -a precision_args profile_args cmd
    local build_plan="${result_save_directory}/$name.plan"
    local build_info="${result_save_directory}/$name.json"
    local build_directory=""
    mkdir -p "$result_save_directory" || return 1
    if (( TRT_MAJOR >= 11 )); then
        # Hash this invocation's private plan, never another writer's final plan.
        build_directory=$(mktemp -d "${result_save_directory}/$name.build.XXXXXX") || return 1
        build_plan="${build_directory}/model.plan"
        build_info="${build_directory}/model.json"
    fi
    read -r -a precision_args <<< "$precision_flags"
    read -r -a profile_args <<< "$profile_flags"
    cmd=("--onnx=$onnx" "${precision_args[@]}" "$input_flags" "$output_flags"
        "--saveEngine=$build_plan"
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo
        "${profile_args[@]}"
        --profilingVerbosity=detailed "--exportLayerInfo=$build_info")

    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment~.
    if ! "$TensorRT_Bin" "${cmd[@]}" > "${result_save_directory}/$name.log" 2>&1; then
        if (( TRT_MAJOR >= 11 )); then
            rm -f -- "$build_plan" "$build_info"
            rmdir -- "$build_directory"
        fi
        echo Failed to build model ${result_save_directory}/$name.plan.
        echo You can check the error message by ${result_save_directory}/$name.log
        exit 1
    fi
    if (( TRT_MAJOR >= 11 )); then
        # Do not publish a plan bound to an input identity that changed while
        # the builder was reading the protobuf or its external tensor files.
        local current_model_identity=""
        if ! current_model_identity=$(model_dependency_identity "$onnx") || [[ "$current_model_identity" != "$model_identity" ]]; then
            echo "Model dependencies changed during engine build; keeping the previous engine: $onnx" >&2
            rm -f -- "$build_plan" "$build_info"
            rmdir -- "$build_directory"
            return 1
        fi
        engine_hash=$(sha256sum -- "$build_plan" | cut -d' ' -f1) || return 1
        if [[ -f "${result_save_directory}/$name.plan" ]]; then
            chmod --reference="${result_save_directory}/$name.plan" -- "$build_plan" || return 1
        fi
        if [[ -f "${result_save_directory}/$name.json" ]]; then
            chmod --reference="${result_save_directory}/$name.json" -- "$build_info" || return 1
        fi
        mv -f -- "$build_plan" "${result_save_directory}/$name.plan" || return 1
        mv -f -- "$build_info" "${result_save_directory}/$name.json" || return 1
        rmdir -- "$build_directory" || return 1
        printf '%s\n%s\n' "$identity" "$engine_hash" > "${identity_file}.$$" || return 1
        mv -f -- "${identity_file}.$$" "$identity_file" || return 1
    fi
    return 0
}

trtexec_dynamic_flags="--fp16 --int8"
if [[ "$precision" == "int8" ]]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

trtexec_fp16_flags="--fp16"
if (( TRT_MAJOR >= 11 )); then
    # The INT8 branch must use a genuine Q/DQ export; the native builder checks
    # it instead of fabricating calibration or silently using FP32.
    trtexec_dynamic_flags="--int8"
fi

cd ${MODEL_ROOT}
compile_trt_model "camera.backbone" "$trtexec_dynamic_flags" 2 2 || exit 1
compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1 || exit 1

# fp16 only
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1 || exit 1
compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6 || exit 1
