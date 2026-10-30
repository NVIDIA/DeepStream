#!/usr/bin/env bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

# test_setup.sh
# Preflight for nvmodelmux tests:
#   1) plugin must be installed (hard requirement)
#   2) required engines must exist; missing ones are generated

DS_ROOT="/opt/nvidia/deepstream/deepstream"
MODEL_PATH="${DS_ROOT}/samples/models/Primary_Detector"
ONNX_NAME="resnet18_trafficcamnet_pruned.onnx"
ONNX_PATH="${MODEL_PATH}/${ONNX_NAME}"
TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"
PLUGIN_SO="/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_modelmux.so"

build_engine() {
    # Supported profiles:
    # - default -> resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine
    # - b15     -> resnet18_trafficcamnet_pruned.onnx_b15_gpu0_fp16.engine
    # - b30     -> resnet18_trafficcamnet_pruned.onnx_b30_gpu0_fp16.engine
    # - best    -> resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp32_best.engine
    # - worst   -> resnet18_trafficcamnet_pruned.onnx_b4_gpu0_int8_nocalib_worst.engine
    local profile="${1:-default}"
    local precision engine_name prec_arg extra_arg batch_size

    case "${profile}" in
        best)
            precision="fp32"
            engine_name="${ONNX_NAME}_b4_gpu0_fp32_best.engine"
            prec_arg=""
            extra_arg=""
            batch_size=4
            ;;
        worst)
            precision="int8"
            engine_name="${ONNX_NAME}_b4_gpu0_int8_nocalib_worst.engine"
            prec_arg="--int8"
            # INT8 calibration is intentionally disabled ("nocalib")
            extra_arg="--skipInference"
            batch_size=4
            ;;
        b15)
            precision="fp16"
            engine_name="${ONNX_NAME}_b15_gpu0_fp16.engine"
            prec_arg="--fp16"
            extra_arg=""
            batch_size=15
            ;;
        b30)
            precision="fp16"
            engine_name="${ONNX_NAME}_b30_gpu0_fp16.engine"
            prec_arg="--fp16"
            extra_arg=""
            batch_size=30
            ;;
        default|*)
            precision="fp16"
            engine_name="${ONNX_NAME}_b4_gpu0_fp16.engine"
            prec_arg="--fp16"
            extra_arg=""
            batch_size=4
            ;;
    esac

    local trtexec="${TRTEXEC_BIN}"
    if [ ! -x "${trtexec}" ]; then
        trtexec="$(command -v trtexec 2>/dev/null || true)"
    fi
    if [ -z "${trtexec}" ] || [ ! -x "${trtexec}" ]; then
        echo "[setup] ERROR: trtexec not executable (TRTEXEC_BIN='${TRTEXEC_BIN}')"
        echo "[setup]        set TRTEXEC_BIN or add trtexec to PATH"
        return 1
    fi
    if [ ! -f "${ONNX_PATH}" ]; then
        echo "[setup] ERROR: ONNX model not found: ${ONNX_PATH}"
        return 1
    fi

    echo "[setup] building ${profile} engine (${precision}) -> ${engine_name}"
    local trt_args=(
        --onnx="${ONNX_PATH}"
        --minShapes=input_1:0:1x3x544x960
        --optShapes=input_1:0:${batch_size}x3x544x960
        --maxShapes=input_1:0:${batch_size}x3x544x960
        --saveEngine="${MODEL_PATH}/${engine_name}"
    )
    [ -n "${prec_arg}" ]  && trt_args+=("${prec_arg}")
    [ -n "${extra_arg}" ] && trt_args+=("${extra_arg}")
    "${trtexec}" "${trt_args[@]}"
}

ensure_engine() {
    local profile="$1"
    local engine_name engine bak

    case "${profile}" in
        best)    engine_name="${ONNX_NAME}_b4_gpu0_fp32_best.engine" ;;
        worst)   engine_name="${ONNX_NAME}_b4_gpu0_int8_nocalib_worst.engine" ;;
        b15)     engine_name="${ONNX_NAME}_b15_gpu0_fp16.engine" ;;
        b30)     engine_name="${ONNX_NAME}_b30_gpu0_fp16.engine" ;;
        default) engine_name="${ONNX_NAME}_b4_gpu0_fp16.engine" ;;
        *) echo "[setup] ERROR: unknown engine profile '${profile}'"; return 1 ;;
    esac
    engine="${MODEL_PATH}/${engine_name}"
    bak="${engine}.bak"

    # Some cases (test_dummy/test_1/test_2) intentionally DELETE the default engine
    # to exercise cold engine generation / a long WARMING window. Keep a .bak copy
    # so the next run RESTORES it here instead of paying a full trtexec rebuild
    # (~35s) for an engine we already built on this machine.
    if [ -f "${engine}" ]; then
        [ -f "${bak}" ] || cp -f "${engine}" "${bak}"
        echo "[setup] ${profile} engine exists: ${engine_name}"
        return 0
    fi
    if [ -f "${bak}" ]; then
        cp -f "${bak}" "${engine}"
        echo "[setup] ${profile} engine missing (a test deletes it intentionally) -> restored from ${engine_name}.bak"
        return 0
    fi

    echo "[setup] ${profile} engine missing -> generating"
    build_engine "${profile}" && cp -f "${engine}" "${bak}"
}

modelmux_plugin_check() {
    if [ ! -f "${PLUGIN_SO}" ]; then
        echo "[setup] ERROR: nvmodelmux plugin not found: ${PLUGIN_SO}"
        echo "[setup]        install plugin first (e.g. sudo make install) and rerun tests"
        return 1
    fi
    echo "[setup] nvmodelmux plugin present: ${PLUGIN_SO}"
}

main() {
    echo "[setup] validating nvmodelmux test prerequisites..."

    modelmux_plugin_check || return 1

    ensure_engine "default" || return 1
    ensure_engine "b15" || return 1
    ensure_engine "b30" || return 1
    ensure_engine "best" || return 1
    ensure_engine "worst" || return 1

    echo "[setup] prerequisites OK"
}

main "$@"