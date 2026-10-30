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
###############################################################################
# stress_setup.sh -- preflight for the nvmodelmux STRESS tests (run_stress.sh).
# Prebuilds the batch/precision engine variants the stress cases request, so the
# per-version [model-<name>-<version>] engine= overrides resolve (the plugin only
# uses a per-version engine that already EXISTS on disk) and so no big TensorRT
# build lands inside a measured window.
#
# Builds (missing ones only), for the Trafficcamnet ONNX:
#   b30 fp16                 -> stress_2 (30-stream soak)
#   b15 fp16                 -> stress_4 (two-primary swap; Trafficcamnet + Car share it)
#   b15 int8 nocalib worst   -> stress_3 primary  (v1, "worst")
#   b15 fp32 best            -> stress_3 shadow   (v2, "best")
#
# Batches are configurable so they track the run_stress knobs:
#   NUM_STREAMS (30)  AB_STREAMS (15)  SWAP_STREAMS (30, uses half=15)
###############################################################################

DS_ROOT="${DS_ROOT:-/opt/nvidia/deepstream/deepstream}"
MODEL_PATH="${MODEL_PATH:-${DS_ROOT}/samples/models/Primary_Detector}"
ONNX_NAME="${ONNX_NAME:-resnet18_trafficcamnet_pruned.onnx}"
ONNX_PATH="${MODEL_PATH}/${ONNX_NAME}"
TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"
PLUGIN_SO="${PLUGIN_SO:-${DS_ROOT}/lib/gst-plugins/libnvdsgst_modelmux.so}"
INPUT_DIMS="input_1:0:%sx3x544x960"           # %s = batch
# a real Trafficcamnet engine is >1 MB; anything smaller = truncated/corrupt build
MIN_ENGINE_BYTES="${MIN_ENGINE_BYTES:-100000}"

# engine filename for (batch, profile) -- MUST match what run_stress / the configs expect
_engine_name() { # <batch> <profile>
  case "$2" in
    best)  echo "${ONNX_NAME}_b$1_gpu0_fp32_best.engine" ;;
    worst) echo "${ONNX_NAME}_b$1_gpu0_int8_nocalib_worst.engine" ;;
    *)     echo "${ONNX_NAME}_b$1_gpu0_fp16.engine" ;;
  esac
}

# build_engine <batch> <profile:fp16|best|worst>
build_engine() {
  local batch="$1" profile="${2:-fp16}" precision prec_arg extra_arg name
  name="$(_engine_name "${batch}" "${profile}")"
  case "${profile}" in
    best)  precision="fp32"; prec_arg="";       extra_arg="" ;;
    worst) precision="int8"; prec_arg="--int8"; extra_arg="--skipInference" ;;  # nocalib (intentionally poor)
    *)     precision="fp16"; prec_arg="--fp16"; extra_arg="" ;;
  esac

  local trtexec="${TRTEXEC_BIN}"
  [ -x "${trtexec}" ] || trtexec="$(command -v trtexec 2>/dev/null || true)"
  if [ -z "${trtexec}" ] || [ ! -x "${trtexec}" ]; then
    echo "[setup] ERROR: trtexec not executable (TRTEXEC_BIN='${TRTEXEC_BIN}') -- set TRTEXEC_BIN or add to PATH"; return 1
  fi
  [ -f "${ONNX_PATH}" ] || { echo "[setup] ERROR: ONNX not found: ${ONNX_PATH}"; return 1; }

  echo "[setup] building b${batch} ${profile} (${precision}) -> ${name}"
  local dims; dims="$(printf "${INPUT_DIMS}" "${batch}")"
  local mindims; mindims="$(printf "${INPUT_DIMS}" 1)"
  local out="${MODEL_PATH}/${name}" tmp="${MODEL_PATH}/.${name}.building"
  rm -f "${tmp}"
  local args=( --onnx="${ONNX_PATH}"
    --minShapes="${mindims}" --optShapes="${dims}" --maxShapes="${dims}"
    --saveEngine="${tmp}" )
  [ -n "${prec_arg}" ]  && args+=("${prec_arg}")
  [ -n "${extra_arg}" ] && args+=("${extra_arg}")
  # Build into a TEMP file; publish the final engine only if trtexec succeeded AND the
  # output is non-trivial. Stops an interrupted/failed build (e.g. Ctrl-C) from leaving a
  # 0-byte / truncated .engine that deserializes as NULL at runtime -> nvinfer silently
  # rebuilds from ONNX -> the model sits in "Warming" for the whole test.
  local sz=0
  if "${trtexec}" "${args[@]}"; then sz="$(stat -c%s "${tmp}" 2>/dev/null || echo 0)"; fi
  if [ "${sz}" -ge "${MIN_ENGINE_BYTES}" ]; then
    mv -f "${tmp}" "${out}"
    echo "[setup]   -> saved ${name} (${sz} bytes)"
  else
    rm -f "${tmp}"
    echo "[setup] ERROR: build failed or produced an invalid engine for ${name} (${sz} bytes)"; return 1
  fi
}

# ensure_engine <batch> <profile> : (re)build unless a VALID engine already exists.
# Treats a present-but-too-small file as corrupt (from an interrupted build) and rebuilds it.
ensure_engine() {
  local name; name="$(_engine_name "$1" "$2")"
  local f="${MODEL_PATH}/${name}" sz=0
  [ -f "${f}" ] && sz="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
  if [ "${sz}" -ge "${MIN_ENGINE_BYTES}" ]; then
    echo "[setup] b$1 $2 engine exists: ${name} (${sz} bytes)"; return 0
  fi
  if [ -f "${f}" ]; then
    echo "[setup] b$1 $2 engine present but INVALID (${sz} bytes) -> rebuilding: ${name}"; rm -f "${f}"
  else
    echo "[setup] b$1 $2 engine missing -> generating: ${name}"
  fi
  build_engine "$1" "$2"
}

plugin_check() {
  [ -f "${PLUGIN_SO}" ] || { echo "[setup] ERROR: nvmodelmux plugin not found: ${PLUGIN_SO} (install it first)"; return 1; }
  echo "[setup] nvmodelmux plugin present: ${PLUGIN_SO}"
}

main() {
  echo "[setup] validating nvmodelmux STRESS prerequisites ..."
  plugin_check || return 1

  local n="${NUM_STREAMS:-30}" ab="${AB_STREAMS:-15}" swap_half=$(( ${SWAP_STREAMS:-30} / 2 ))
  ensure_engine "${n}"        fp16  || return 1   # stress_2
  ensure_engine "${ab}"       worst || return 1   # stress_3 primary (v1)
  ensure_engine "${ab}"       best  || return 1   # stress_3 shadow  (v2)
  ensure_engine "${swap_half}" fp16 || return 1   # stress_4 (Trafficcamnet + Car)
  echo "[setup] stress prerequisites OK (engines under ${MODEL_PATH})"
}

main "$@"
