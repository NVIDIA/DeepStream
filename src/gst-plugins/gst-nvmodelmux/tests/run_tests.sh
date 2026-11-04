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
# run_tests.sh - functional test cases for the nvmodelmux PLUGIN, driven
# through deepstream-test5 + nvmultiurisrcbin's NATIVE DeepStream REST API.
# test5 selects the plugin via [primary-gie] plugin-type=4; the plugin parses its own
# [multimodel] config and handles the stream/model REST APIs internally.
#
# TWO configs (copies are mutated, originals never touched):
#   - MAIN config (test5):       configs/test5_config_multimodel.txt  -> .run_main.txt
#   - plugin [multimodel] config: configs/config_modelmux.txt   -> .run_mm.txt
# Edit them in a test with:  cfg_set <grp> <key> <val>  (auto-routes/translates),
#   or explicitly:           cfg_main <grp> <key> <val> / cfg_mm <grp> <key> <val>.
#
# AUTO mode (default, AUTO_APP=1): runs test_setup precheck (plugin + engines),
# starts test5, waits ready, runs the case(s), stops it:
#   ./run_tests.sh 5
#   ./run_tests.sh all
#
# MANUAL mode (AUTO_APP=0): test5 is managed in another terminal:
#   # terminal 1:  deepstream-test5-app -c configs/test5_config_multimodel.txt
#   # terminal 2:  AUTO_APP=0 ./run_tests.sh 5
#
# Override:  HOST=localhost PORT=9000 DELAY=5 PREFIX=cam- \
#            APP_BIN=... APP_START_TIMEOUT=90 APP_WARMUP=20 ./run_tests.sh
###############################################################################
ENABLE_DISPLAY="${ENABLE_DISPLAY:-1}"
if [ "${ENABLE_DISPLAY}" = "1" ]; then
  export DISPLAY="${DISPLAY:-:1}"   # honor an inherited DISPLAY; default :1 for headless/CI
  SINK_TYPE=2
else
  SINK_TYPE=1
fi
HOST="${HOST:-localhost}"
PORT="${PORT:-9000}"
DELAY="${DELAY:-5}"
PREFIX="${PREFIX:-cam-}"
ADD_URL="http://${HOST}:${PORT}/api/v1/stream/add"
REMOVE_URL="http://${HOST}:${PORT}/api/v1/stream/remove"
LOAD_URL="http://${HOST}:${PORT}/api/v1/model/load"
UNLOAD_URL="http://${HOST}:${PORT}/api/v1/model/unload"
ROUTE_URL="http://${HOST}:${PORT}/api/v1/model/routing"
MODEL_UPDATE_URL="http://${HOST}:${PORT}/api/v1/model/update"   # v1 in-place checkpoint transition (test_40)
STATUS_URL="http://${HOST}:${PORT}/api/v1/model/status"

# Dirs + binaries. This harness drives the nvmodelmux PLUGIN through
# deepstream-test5 ([primary-gie] plugin-type=4 -> nvmodelmux).
APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"     # this tests/ dir
PLUGIN_DIR="$(cd "${APP_DIR}/.." && pwd)"                               # plugin repo root
CFG_DIR="${PLUGIN_DIR}/configs"
DS_ROOT="/opt/nvidia/deepstream/deepstream"
STREAM="${STREAM:-file://${DS_ROOT}/samples/streams/sample_1080p_h264.mp4}"
# Prefer PATH resolution; allow override with an explicit absolute path via APP_BIN.
APP_BIN="${APP_BIN:-deepstream-test5-app}"
TEST_SETUP_SCRIPT="${TEST_SETUP_SCRIPT:-${APP_DIR}/test_setup.sh}"
RUN_TEST_SETUP="${RUN_TEST_SETUP:-1}"   # set 0 to skip plugin/engine precheck

# TWO configs are in play and we NEVER mutate the originals (same discipline as the app):
#   BASE_MAIN -> the deepstream-test5 main config ([source-list], [streammux], [primary-gie]
#                plugin-type=4 + config-file, [tiled-display], [sink0]).
#   BASE_MM   -> the plugin's own [multimodel] config (defaults/behaviour + [stream-model-*]).
# Each case copies BOTH bases into throwaway copies, mutates ONLY the copies (via cfg_set /
# cfg_main / cfg_mm), and launches deepstream-test5 with the MAIN copy (whose [primary-gie]
# config-file is repointed at the MM copy). The originals are left untouched.
BASE_MAIN="${BASE_MAIN:-${CFG_DIR}/test5_config_multimodel.txt}"
BASE_MM="${BASE_MM:-${CFG_DIR}/config_modelmux.txt}"
# Per-run artifacts (throwaway config copies + per-test logs) live under tests/.logs/.
# test_<n>_logs.txt is the SINGLE per-test log: it holds EVERYTHING for that case -- the script
# narration (config echoes, >>> REST API prints, [PASS]/[FAIL]) AND the app's stdout/stderr.
LOG_DIR="${LOG_DIR:-${APP_DIR}/.logs}"
# Persistent results ledger: a single human-readable file, one block per test keyed by
# test number, holding that test's latest PASS/FAIL outcome. Re-running a test updates
# only its block (see record_test_result). Per-test fragments live under .logs/results/
# and are concatenated in numeric order to regenerate RESULTS_FILE.
RESULTS_FILE="${RESULTS_FILE:-${LOG_DIR}/test_results.txt}"
RESULTS_DIR="${RESULTS_DIR:-${LOG_DIR}/results}"
# Working copies live in LOG_DIR but KEEP THEIR ORIGINAL NAMES (not renamed to .run_*), so
# the run dir mirrors the configs repo 1:1 -- the base configs + every per-model config are
# copied here (cfg_reset) and the app runs with CWD=LOG_DIR, so a bare/relative model-config
# reference resolves locally. Override via env if a different working name/location is wanted.
TEST_MAIN="${TEST_MAIN:-${LOG_DIR}/test5_config_multimodel.txt}"
TEST_MM="${TEST_MM:-${LOG_DIR}/config_modelmux.txt}"
GST_REG_FILE="${LOG_DIR}/.gstreg.bin"
mkdir -p "${LOG_DIR}"

# Reusable ZERO-FRAME-LOSS sanity gate (assert_no_frame_drops). Scans a test's
# saved log for any frameacct-reported drop; called by default from stop_app.
source "${APP_DIR}/frameacct_check.sh"

# Name of the configured default primary model (matches default_model in the base config).
# Each model: NAME, VERSION (Triton-style positive integer; one source of truth used by
# load_model / add_stream / model_routing AND the asserts), and base CONFIG (catalog default).
DEFAULT_MODEL="${DEFAULT_MODEL:-Trafficcamnet}"
DEFAULT_MODEL_VERSION="${DEFAULT_MODEL_VERSION:-1}"
DEFAULT_SHADOW_MODEL_VERSION="${DEFAULT_SHADOW_MODEL_VERSION:-2}"
PERSON_MODEL="${PERSON_MODEL:-Person}"
PERSON_MODEL_VERSION="${PERSON_MODEL_VERSION:-1}"
PERSON_MODEL_VERSION_NEW="${PERSON_MODEL_VERSION_NEW:-2}"
CAR_MODEL="${CAR_MODEL:-Car}"
CAR_MODEL_VERSION="${CAR_MODEL_VERSION:-1}"
CAR_MODEL_VERSION_NEW="${CAR_MODEL_VERSION_NEW:-2}"

# Created by test_setup.sh
DEFAULT_ENGINE="${DEFAULT_ENGINE:-/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine}"
BEST_ENGINE="${BEST_ENGINE:-/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp32_best.engine}"
WORST_ENGINE="${WORST_ENGINE:-/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_int8_nocalib_worst.engine}"
B15_ENGINE="${B15_ENGINE:-/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b15_gpu0_fp16.engine}"
B30_ENGINE="${B30_ENGINE:-/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b30_gpu0_fp16.engine}"
# Secondary stage (test 59): the classifier the [secondary-gie0] nvmodelmux serves.
SGIE_MODEL="${SGIE_MODEL:-VehicleTypes}"
SGIE_ENGINE="${SGIE_ENGINE:-/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet_pruned.onnx_b16_gpu0_fp16.engine}"
# Its nvinfer config, resolved relative to the app's working directory (the same path
# the [model-name-N] catalog entry uses). Loading a version with THIS as its artifact --
# rather than an engine path -- is what makes the load rebuild from ONNX once the cached
# engine is moved aside, which is how test 59 opens a real warming window.
SGIE_INFER_CONFIG="${SGIE_INFER_CONFIG:-config_infer_secondary_vehicletypes.txt}"

# Auto-orchestration: when AUTO_APP=1 (default), `./run_test_cases.sh <case>` starts
# the app, waits for it to be ready, runs the case, then stops the app. Set
# AUTO_APP=0 to use an already-running app (classic two-terminal workflow).
AUTO_APP="${AUTO_APP:-1}"
APP_START_TIMEOUT="${APP_START_TIMEOUT:-30}"  # max s to wait for REST to come up
APP_WARMUP="${APP_WARMUP:-5}"                # extra s after REST up for engine gen
APP_STOP_DELAY="${APP_STOP_DELAY:-5}"          # s to keep app alive after the case
APP_PID=""                                  # pid of the app started by run_multimodel
APP_LOG="${APP_LOG:-${LOG_DIR}/run_logs.txt}"  # default; set per-test to tests/.logs/test_<name>_logs.txt

# RACE_REPS: GLOBAL repetition count for the race/concurrency-sensitive cases
# (test_3/test_4 drain-refill cycles, test_31 swap ping-pong, test_32 teardown-
# under-reconfig cycles, test_34 remove+add id-recycle bursts, test_36 churn+
# teardown cycles). Races are timing-dependent: ONE clean burst can pass by luck;
# N clean bursts are evidence. Each case checks every rep SILENTLY and reports ONE
# aggregated verdict that passes ONLY if ALL reps passed.
# Override for a quick look: RACE_REPS=2 ./run_tests.sh <n>
RACE_REPS="${RACE_REPS:-5}"

# CORE DUMPS: raise the core-file soft limit to the hard maximum so ANY crash of the
# app leaves a core next to the logs (kernel core_pattern is 'core' -> written to the
# crashing process CWD, which start_app sets to LOG_DIR). A mid-run SIGSEGV without a
# core is unattributable (stress_1 2026-07-09: one-off crash, no stack, no verdict).
# Best-effort: never fail the harness over a restricted limit.
_core_hard="$(ulimit -Hc 2>/dev/null)"
ulimit -c "${_core_hard:-unlimited}" 2>/dev/null || ulimit -c unlimited 2>/dev/null || true
echo "[run] core-dump limit: soft=$(ulimit -Sc 2>/dev/null) hard=${_core_hard:-?} (core_pattern: $(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unknown))"

########################################################################################################################
############################################### Auto-orchestration #####################################################
########################################################################################################################

# Preflight setup for AUTO mode:
#   - verify nvmodelmux plugin is installed
#   - ensure required engines exist (build only if missing)
run_setup_check() {
  [ "${RUN_TEST_SETUP}" = "1" ] || return 0
  if [ ! -f "${TEST_SETUP_SCRIPT}" ]; then
    echo "[setup] ERROR: setup script not found: ${TEST_SETUP_SCRIPT}" >&2
    return 1
  fi
  echo "[setup] running ${TEST_SETUP_SCRIPT##*/} (plugin + engine checks)"
  bash "${TEST_SETUP_SCRIPT}"
}

# kill_existing_app_instances:
# Before launching AUTO mode app, clean up stale deepstream-test5-app processes
# that might still be running from an earlier aborted run.
kill_existing_app_instances() {
  local app_name pids pid i alive
  app_name="$(basename "${APP_BIN}")"
  pids="$(
    ps -eo pid=,args= | awk -v app="${app_name}" '
      {
        pid = $1
        $1 = ""
        cmd = $0
        sub(/^[[:space:]]+/, "", cmd)
        n = split(cmd, parts, /[[:space:]]+/)
        exe = parts[1]
        sub(/^.*\//, "", exe)
        if (exe == app) print pid
      }
    '
  )"

  [ -z "${pids}" ] && return 0

  echo "[run] found existing ${app_name} instance(s): $(echo "${pids}" | tr '\n' ' ')"
  echo "[run] sending SIGTERM to stale ${app_name} instance(s) ..."
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    kill "${pid}" 2>/dev/null || true
  done

  i=0
  while [ "${i}" -lt 5 ]; do
    alive=""
    for pid in ${pids}; do
      [ "${pid}" = "$$" ] && continue
      if kill -0 "${pid}" 2>/dev/null; then
        alive="${alive} ${pid}"
      fi
    done
    [ -z "${alive}" ] && break
    sleep 1
    i=$((i+1))
  done

  if [ -n "${alive}" ]; then
    echo "[run] sending SIGKILL to stale ${app_name} instance(s):${alive}"
    for pid in ${alive}; do
      kill -9 "${pid}" 2>/dev/null || true
    done
  fi
}

# cfg_reset: start each case from pristine copies of BOTH base configs, and point the
# main copy's [primary-gie] config-file at the MM copy (basename -- deepstream-app resolves
# it relative to the main config's dir, and both copies live in configs/).
cfg_reset() {
  cp -f "${BASE_MAIN}" "${TEST_MAIN}"
  cp -f "${BASE_MM}"   "${TEST_MM}"
  # Co-locate ALL repo configs (the per-model nvinfer/nvinferserver configs + any aux files
  # like labels) with the working copies in LOG_DIR, keeping their ORIGINAL names. The app
  # is launched with CWD=LOG_DIR (see run_multimodel), so a model config referenced by a
  # bare/relative name in [multimodel] (e.g. primary-model=...;config_infer_primary.txt)
  # resolves right here -- no hardcoded paths, and a user-supplied FULL path still wins as-is.
  cp -f "${CFG_DIR}"/* "${LOG_DIR}/" 2>/dev/null || true
  _cfg_raw primary-gie config-file "$(basename "${TEST_MM}")" "${TEST_MAIN}"
  # Display on/off is governed HERE (single point) so it applies to EVERY case -- functional
  # tests AND the stress workloads (run_stress.sh also calls cfg_reset). ENABLE_DISPLAY=1 ->
  # SINK_TYPE=2 (EGL on-screen display); =0 -> SINK_TYPE=1 (fakesink, headless/CI). Per-test
  # cfg_set sink0 type calls (if any) still override this afterwards.
  _cfg_raw sink0 type "${SINK_TYPE}" "${TEST_MAIN}"
  # NOTE on stream lifetime: the sample clip is ~46s and does NOT loop. deepstream-test5's
  # [tests] file-loop only loops the app's STATIC source list (empty here); there is no
  # config key that loops nvmultiurisrcbin's REST-added sources (the per-source "file-loop"
  # key is unwired in this app build), so a REST stream lives exactly one clip (~46s) then
  # EOSes and is auto-removed. Keep each case's post-add work comfortably under that window
  # (don't pile up long per-step sleeps), or the later REST ops will target a stream that
  # already ended. Every case removes its streams explicitly, so this only bounds duration.
  echo "[config] reset ${TEST_MAIN##*/} <- ${BASE_MAIN##*/} ; ${TEST_MM##*/} <- ${BASE_MM##*/}"
}

# _cfg_raw <group> <key> <value> <file>
# Low-level INI writer. In the target [group], prefer updating an ACTIVE key in place.
# If no active key exists, reuse a COMMENTED key (uncomment/update). If the key is absent,
# insert it at the end of the group; if the group is absent, create it at EOF.
_cfg_raw() {
  local group="$1" key="$2" val="$3" file="${4:-${TEST_MAIN}}"
  [ -f "${file}" ] || return 1
  group="${group#[}"; group="${group%]}"
  awk -v hdr="[${group}]" -v key="${key}" -v val="${val}" '
    function trim(s){ sub(/^[ \t]+/,"",s); sub(/[ \t]+$/,"",s); return s }
    function flush_target(    i,t,c,eq,active_idx,comment_idx) {
      if (!in_grp) return
      active_idx = 0
      comment_idx = 0
      for (i = 1; i <= n; i++) {
        t = trim(buf[i])
        if (t ~ /^#/) {
          c = t
          sub(/^#[ \t]*/, "", c)
          eq = index(c, "=")
          if (eq > 0 && trim(substr(c, 1, eq-1)) == key && comment_idx == 0) comment_idx = i
        } else {
          eq = index(t, "=")
          if (eq > 0 && trim(substr(t, 1, eq-1)) == key && active_idx == 0) active_idx = i
        }
      }

      if (active_idx > 0) {
        buf[active_idx] = key "=" val
      } else if (comment_idx > 0) {
        buf[comment_idx] = key "=" val
      } else {
        buf[++n] = key "=" val
      }

      for (i = 1; i <= n; i++) print buf[i]
      n = 0
      done = 1
    }
    BEGIN { in_grp=0; seen_grp=0; done=0; n=0 }
    {
      t = trim($0)
      if (t ~ /^\[.*\]$/) {                          # a group header line
        if (in_grp) { flush_target(); in_grp = 0 }
        in_grp = (!done && t == hdr) ? 1 : 0
        if (in_grp) seen_grp = 1
        print; next
      }
      if (in_grp) { buf[++n] = $0; next }
      print
    }
    END {
      if (in_grp) flush_target()                             # target group was last in the file
      if (!seen_grp) { print ""; print hdr; print key"="val } # group absent -> create at EOF
    }
  ' "${file}" > "${file}.tmp" && mv "${file}.tmp" "${file}"
}

# _cfg_get <group> <key> <file> : print the current value (active or commented), else "".
_cfg_get() {
  awk -v hdr="[$1]" -v key="$2" '
    function trim(s){ sub(/^[ \t]+/,"",s); sub(/[ \t]+$/,"",s); return s }
    { t=trim($0)
      if (t ~ /^\[.*\]$/){ ing=(t==hdr); next }
      if (ing){ c=t; sub(/^#[ \t]*/,"",c); eq=index(c,"=")
        if (eq>0 && trim(substr(c,1,eq-1))==key){ print trim(substr(c,eq+1)); exit } } }
  ' "$3" 2>/dev/null
}

# cfg_set <group> <key> <value>
# Router: edits the right config COPY and writes the property directly (key=value). Routing:
#   multimodel / stream-model-* / model-name-* -> plugin config (MM)
#   everything else (source-list / streammux / tiled-display / osd / sink* / ...) -> MAIN
# Model refs are plain values now: primary-model / shadow-model = "name;version"; the model's
# config comes from its [model-name-*] catalog entry, so there's nothing to decompose.
cfg_set() {
  local group="$1" key="$2" val="$3"
  group="${group#[}"; group="${group%]}"
  case "${group}" in
    multimodel|stream-model-*|model-*)
      # plugin config: keys are plain key=value now (primary-model / shadow-model carry
      # "name;version"; the catalog [model-name-*] and per-version [model-<name>-<version>]
      # blocks supply config/engine). Set the property directly.
      _cfg_raw "${group}" "${key}" "${val}" "${TEST_MM}"
      echo "[config] [${group}] ${key}=${val}  ->  ${TEST_MM##*/}" ;;
    *)
      _cfg_raw "${group}" "${key}" "${val}" "${TEST_MAIN}"
      # keep the streammux + plugin batch in lock-step with the source batch.
      if [ "${group}" = "source-list" ] && [ "${key}" = "max-batch-size" ]; then
        _cfg_raw streammux  batch-size "${val}" "${TEST_MAIN}"
        _cfg_raw multimodel batch-size "${val}" "${TEST_MM}"
      fi
      # keep the tiled-display canvas in lock-step with the grid: the Tegra VIC compositor
      # rejects per-tile downscales beyond 16x of the 1920x1080 streammux output (x86
      # composites on GPU, no such limit -- fatal only on Thor/Spark/Jetson). Size tiles at
      # 128x72 (a 15x factor, inside the ceiling with headroom) and never shrink below the
      # config's 1280x720 default. This follows the tiler's own guidance: 'Please use a
      # larger tiled window resolution or reduce number of streams'.
      if [ "${group}" = "tiled-display" ] && { [ "${key}" = "rows" ] || [ "${key}" = "columns" ]; }; then
        local _t_rows _t_cols _t_w _t_h
        _t_rows="$(_cfg_get tiled-display rows    "${TEST_MAIN}")"
        _t_cols="$(_cfg_get tiled-display columns "${TEST_MAIN}")"
        _t_w=$(( ${_t_cols:-1} * 128 )); _t_h=$(( ${_t_rows:-1} * 72 ))
        [ "${_t_w}" -lt 1280 ] && _t_w=1280
        [ "${_t_h}" -lt 720 ]  && _t_h=720
        _cfg_raw tiled-display width  "${_t_w}" "${TEST_MAIN}"
        _cfg_raw tiled-display height "${_t_h}" "${TEST_MAIN}"
        echo "[config] [tiled-display] canvas ${_t_w}x${_t_h} (grid ${_t_rows:-1}x${_t_cols:-1}, VIC 16x-safe tiles)  ->  ${TEST_MAIN##*/}"
      fi
      echo "[config] [${group}] ${key}=${val}  ->  ${TEST_MAIN##*/}" ;;
  esac
}

# Explicit per-file utilities (update either config copy directly, no translation):
#   cfg_main <group> <key> <value>  -> the deepstream-test5 MAIN config copy
#   cfg_mm   <group> <key> <value>  -> the plugin [multimodel] config copy
cfg_main() { _cfg_raw "$1" "$2" "$3" "${TEST_MAIN}"; echo "[config] [$1] $2=$3  ->  ${TEST_MAIN##*/}"; }
cfg_mm()   { _cfg_raw "$1" "$2" "$3" "${TEST_MM}";   echo "[config] [$1] $2=$3  ->  ${TEST_MM##*/}"; }

# comment_cfg <group> <key> [file]
# Comment out (disable) a single key inside an INI [group] -- the group-aware sibling of
# cfg_set, used to OMIT an optional property so the plugin sees it as unset (e.g. drop the
# default primary-model to test the no-default passthrough path). Only the ACTIVE line in the
# matching group is touched; lines in other groups, already-commented lines, and absent keys
# are left as-is (no-op). Defaults to the plugin [multimodel] copy (TEST_MM).
comment_cfg() {
  local group="$1" key="$2" file="${3:-${TEST_MM}}"
  group="${group#[}"; group="${group%]}"
  [ -f "${file}" ] || return 1
  awk -v hdr="[${group}]" -v key="${key}" '
    function trim(s){ sub(/^[ \t]+/,"",s); sub(/[ \t]+$/,"",s); return s }
    BEGIN { in_grp=0 }
    {
      t = trim($0)
      if (t ~ /^\[.*\]$/) { in_grp = (t == hdr) ? 1 : 0; print; next }
      if (in_grp && t !~ /^#/) {
        eq = index(t, "=")
        if (eq > 0 && trim(substr(t, 1, eq-1)) == key) { print "# " $0; next }
      }
      print
    }
  ' "${file}" > "${file}.tmp" && mv -f "${file}.tmp" "${file}"
  echo "[config] commented [${group}] ${key}  ->  ${file##*/}"
}

# Launch deepstream-test5 with the MAIN config copy. GST_DEBUG=nvmodelmux:4 routes
# the plugin's control/bin logs into APP_LOG so assert_log can match them.
run_multimodel() {
  echo "--------------------------------"
  echo "### main config: ${TEST_MAIN##*/}"; cat "${TEST_MAIN}"
  echo "### multimodel config: ${TEST_MM##*/}"; cat "${TEST_MM}"
  echo "--------------------------------"
  LOG_MARK=0                                  # reset assert scope (run_one truncated the per-case log)
  # DEBUG MODE IS ON BY DEFAULT: the plugin's category is APPENDED to GST_DEBUG so it
  # always wins (even when the environment already exports GST_DEBUG), giving you the
  # plugin's INFO traces (STREAM ADD / WIRED / model load / promote-swap ...).
  #   MM_LOG_LEVEL : plugin category level (default 4=INFO; 5=DEBUG for max detail)
  #   MM_DEBUG=1   : the bin's model-table / stream-mapping / frame-drop tables (default ON,
  #                  like the app); set MM_DEBUG=0 to quiet them.
  #   MM_OVERLAY=1 : per-tile on-screen debug label
  #                  "<stream>:<frame>:<model>-<version>:<role>" (Primary=green, Shadow=red),
  #                  drawn by the plugin bin from its own provenance meta. Default ON,
  #                  like the app's run_test_cases.sh; set MM_OVERLAY=0 to disable.
  local lvl="${MM_LOG_LEVEL:-4}"
  local mmd="${MM_DEBUG:-1}"
  local ovl="${MM_OVERLAY:-1}"
  # MM_FRAME_TRACE (default ON): carry each frame's ORIGINAL frame_num through the
  # internal muxes so the exact per-source contiguity gap check runs. No-op unless the
  # traced plugin build is loaded (older/system .so simply ignores the env), so this is
  # safe to leave on by default. Set MM_FRAME_TRACE=0 to disable.
  local ftr="${MM_FRAME_TRACE:-1}"
  local dbg="${GST_DEBUG:+${GST_DEBUG},}nvmodelmux:${lvl}"
  echo "[run] DEBUG mode ON -> GST_DEBUG='${dbg}'  MM_DEBUG=${mmd}  MM_OVERLAY=${ovl}  MM_FRAME_TRACE=${ftr}  (log: ${APP_LOG})"
  # stdin: deepstream-app's keyboard handler does g_print("\n") on every char read from
  # stdin (kbhit() = select()), so backgrounded on a terminal it reads the terminal's
  # focus/escape bytes and floods the log with blank lines. /dev/null is WORSE (select
  # reports EOF as readable -> infinite "\n"). Feed an open, data-less, never-closing pipe
  # so select() reports stdin NOT-readable -> kbhit() stays false -> no spurious newlines.
  # Runtime uses installed DeepStream plugin/library paths from the system.
  # No build-tree .so injection is used by this harness.
  # Launch with CWD = LOG_DIR so the plugin's per-model config-files -- referenced by a bare/
  # relative name in [multimodel] and resolved by nvinfer relative to CWD -- are found among
  # the configs cfg_reset co-located here. All other paths (TEST_MAIN, APP_LOG, GST_REGISTRY)
  # are absolute, so the cd is safe. A full/absolute model-config path is
  # unaffected (used as-is). The backgrounded job inherits this CWD; we restore it after.
  local _cwd; _cwd="$(pwd)"
  cd "${LOG_DIR}" || return 1
  local _env=(
    "GST_REGISTRY=${GST_REG_FILE}"
    "GST_DEBUG=${dbg}"
    "MM_DEBUG=${mmd}"
    "MM_OVERLAY=${ovl}"
    "MM_FRAME_TRACE=${ftr}"
  )
  env "${_env[@]}" stdbuf -oL -eL \
    "${APP_BIN}" -c "${TEST_MAIN}" -t < <(tail -f /dev/null) 2>&1 &
  APP_PID=$!                                 # remember it so stop_multimodel can stop it
  cd "${_cwd}" || true                       # restore CWD (the backgrounded app kept LOG_DIR)
  echo "[run] started deepstream-test5-app (pid ${APP_PID}, config ${TEST_MAIN##*/}, log ${APP_LOG})"
}

# Gracefully stop the app started by run_multimodel: SIGINT first (the app's
# handler sends EOS + clean shutdown), then SIGKILL if it doesn't exit in time.
stop_multimodel() {
  [ -z "${APP_PID}" ] && return 0
  kill -0 "${APP_PID}" 2>/dev/null || { APP_PID=""; return 0; }
  echo "[run] stopping app (pid ${APP_PID}) ..."
  kill -INT "${APP_PID}" 2>/dev/null
  local i=0
  while [ "${i}" -lt 10 ] && kill -0 "${APP_PID}" 2>/dev/null; do
    sleep 1; i=$((i+1))
  done
  if kill -0 "${APP_PID}" 2>/dev/null; then
    echo "[run] app did not exit on SIGINT; sending SIGKILL"
    kill -9 "${APP_PID}" 2>/dev/null
  fi
  wait "${APP_PID}" 2>/dev/null
  echo "[run] app stopped"
  APP_PID=""
}

# Wait until the app's REST server is up, then a short warm-up for the default
# model's engine to reach READY (stream-add is rejected before that).
wait_app_ready() {
  echo "[run] waiting for REST on ${HOST}:${PORT} (up to ${APP_START_TIMEOUT}s) ..."
  local i=0
  while [ "${i}" -lt "${APP_START_TIMEOUT}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[run] ERROR: app exited during startup"; return 1
    fi
    if curl -s -o /dev/null --connect-timeout 2 --max-time 5 \
         "http://${HOST}:${PORT}/api/v1/ready"; then
      echo "[run] REST up; warming up ${APP_WARMUP}s for default-model engine ..."
      sleep "${APP_WARMUP}"
      return 0
    fi
    sleep 1; i=$((i+1))
  done
  echo "[run] WARN: REST not detected in ${APP_START_TIMEOUT}s; proceeding anyway"
  return 1
}

########################################################################################################################
############################################### Assertions #############################################################
########################################################################################################################
# The REST layer is async: model/route control ops return HTTP 202 "...ACCEPTED" (stream add/remove stay 200) as soon as
# the request is forwarded; the real accept/reject happens in the app and is only
# visible in its log. So semantic pass/fail is asserted by grepping the app log
# (captured to APP_LOG in AUTO mode) for the app's own markers. In AUTO_APP=0 there
# is no captured log, so asserts SKIP (still informative, never false-fail).

PASS_CNT=0
FAIL_CNT=0
LOG_MARK=0
# Accumulate each check's description so print_summary can ENUMERATE every
# pass/fail by name -- the "N passed" count alone is opaque when checks print at
# different points in the run (mid-test assert_log vs end-of-test sanity gates).
PASS_LIST=()
FAIL_LIST=()

_pass() { echo "  [PASS] $*"; PASS_CNT=$((PASS_CNT+1)); PASS_LIST+=("$*"); }
_fail() { echo "  [FAIL] $*"; FAIL_CNT=$((FAIL_CNT+1)); FAIL_LIST+=("$*"); }

# Mark the current end of the app log; subsequent assert_log scans only NEW lines
# (so a recurring marker from an earlier step can't false-positive).
log_mark() { LOG_MARK=$(wc -l < "${APP_LOG}" 2>/dev/null || echo 0); }

# assert_log <fixed-substring> <description> [timeout-s]
#   PASS if the app logs <substring> (in lines added since the last log_mark) within
#   the timeout; FAIL otherwise. SKIP when AUTO_APP=0 (no captured log).
assert_log() {
  local pat="$1" desc="$2" to="${3:-${DELAY}}" i=0
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF -- "${pat}" \
      && { _pass "${desc}"; return 0; }
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (missing app-log marker: '${pat}')"
  return 1
}

# assert_restore_consistent <camera_name> <description> [timeout-s]
#   PASS if a SOURCE-ID restore line for <camera_name> appears and the provenance
#   source id equals the frame's restored source_id and pad_index. Do not assume
#   Cam-N has numeric source id N: nvmultiurisrcbin assigns/recycles source ids
#   by stream lifetime, especially under out-of-order add and remove/re-add races.
assert_restore_consistent() {
  local name="$1" desc="$2" to="${3:-${DELAY}}" i=0 line vals sid fsid fpad
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    line="$(tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null \
      | sed -E 's/\x1b\[[0-9;]*m//g' \
      | awk -v needle="name=${name} " '
          /SOURCE-ID restore/ && index($0, needle) { print; exit }
        ')"
    if [ -n "${line}" ]; then
      vals="$(printf '%s\n' "${line}" \
        | sed -nE 's/.*SOURCE-ID restore stream=([0-9]+) name=[^ ]+ role=[^ ]+ frame=\(source_id=([0-9]+),pad_index=([0-9]+)\).*/\1 \2 \3/p')"
      if [ -n "${vals}" ]; then
        set -- ${vals}
        sid="$1"; fsid="$2"; fpad="$3"
        if [ "${sid}" = "${fsid}" ] && [ "${sid}" = "${fpad}" ]; then
          _pass "${desc}  (${name} source_id=${sid}, pad_index=${fpad})"
          return 0
        fi
        _fail "${desc}  (inconsistent restore line: ${line})"
        return 1
      fi
      _fail "${desc}  (unparseable restore line: ${line})"
      return 1
    fi
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (missing SOURCE-ID restore line for ${name})"
  return 1
}

# assert_log_absent <fixed-substring> <description> [settle-s]
#   PASS if <substring> is NOT in the log (lines since the last log_mark) -- used to prove a
#   side effect did NOT happen. Waits <settle-s> first so a marker that WOULD appear has had
#   time to (the caller should have already waited for the operation's completion markers).
assert_log_absent() {
  local pat="$1" desc="$2" settle="${3:-3}"
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  sleep "${settle}"
  if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF -- "${pat}"; then
    _fail "${desc}  (unexpected app-log marker present: '${pat}')"
    return 1
  fi
  _pass "${desc}"
  return 0
}

# assert_count_min <pattern> <min-count> <desc> [timeout]
# Like assert_log, but requires the marker to appear at LEAST <min-count> times in
# the app log (since LOG_MARK). This distinguishes a stream that genuinely FLOWS
# (provenance is logged repeatedly as frames stream) from one that pushed a single
# frame and then froze (exactly one provenance line). A single-occurrence assert
# false-passes a wire-then-stall; this one does not.
assert_count_min() {
  local pat="$1" min="$2" desc="$3" to="${4:-${DELAY}}" i=0 n=0
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    n="$(tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -cF -- "${pat}")"
    [ "${n}" -ge "${min}" ] && { _pass "${desc}  (${n} occurrences >= ${min})"; return 0; }
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (saw ${n} of '${pat}', need >= ${min} -- likely wire-then-stall)"
  return 1
}

print_summary() {
  local d
  echo "======================================================================="
  echo "RESULT: ${PASS_CNT} passed, ${FAIL_CNT} failed"
  # list EVERY check by name so the count is self-explanatory (no scrollback hunt)
  if [ "${PASS_CNT}" -gt 0 ]; then
    echo "  passed (${PASS_CNT}):"
    for d in "${PASS_LIST[@]}"; do echo "    [PASS] ${d}"; done
  fi
  if [ "${FAIL_CNT}" -gt 0 ]; then
    echo "  failed (${FAIL_CNT}):"
    for d in "${FAIL_LIST[@]}"; do echo "    [FAIL] ${d}"; done
  fi
  echo "======================================================================="
  [ -f "${RESULTS_FILE}" ] && echo "  (per-test results ledger: ${RESULTS_FILE})"
  [ "${FAIL_CNT}" -eq 0 ]
}

# _results_rebuild: regenerate the single combined ledger from the per-test fragments,
# ordered by numeric test id. Called after each fragment is (re)written.
_results_rebuild() {
  local f id
  {
    echo "#######################################################################"
    echo "# nvmodelmux -- per-test results ledger (latest run of each test)"
    echo "# regenerated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# one block per test, keyed by test number; re-running a test updates only its block."
    echo "#######################################################################"
    echo
    for id in $(ls "${RESULTS_DIR}"/test_*.frag 2>/dev/null \
                  | sed -E 's#.*/test_(-?[0-9]+)\.frag$#\1#' | sort -n); do
      cat "${RESULTS_DIR}/test_${id}.frag"
    done
  } > "${RESULTS_FILE}"
}

# record_test_result <test-id> <pass-start-idx> <fail-start-idx>
# Captures THIS test's slice of PASS_LIST/FAIL_LIST (the entries added since the snapshot
# indices taken before the case ran), writes it as a per-test fragment, and regenerates
# the combined ledger. Idempotent per test: a later run overwrites the same fragment.
record_test_result() {
  local id="$1" ps="$2" fs="$3" i
  mkdir -p "${RESULTS_DIR}" 2>/dev/null || return 0
  local np=$(( ${#PASS_LIST[@]} - ps )) nf=$(( ${#FAIL_LIST[@]} - fs ))
  {
    printf '### TEST %s ###  [%s]  RESULT: %d passed, %d failed\n' \
        "${id}" "$(date '+%Y-%m-%d %H:%M:%S')" "${np}" "${nf}"
    for (( i=ps; i<${#PASS_LIST[@]}; i++ )); do printf '    [PASS] %s\n' "${PASS_LIST[$i]}"; done
    for (( i=fs; i<${#FAIL_LIST[@]}; i++ )); do printf '    [FAIL] %s\n' "${FAIL_LIST[$i]}"; done
    echo
  } > "${RESULTS_DIR}/test_${id}.frag"
  _results_rebuild
}

########################################################################################################################
############################################### API Functions ##########################################################
########################################################################################################################

# _rest_post <url> <json> <label>
# POST to a native REST endpoint, capturing AND printing the HTTP status code and
# the JSON {status,reason} body so a run is verifiable (not just fire-and-forget).
# Returns 0 on HTTP 2xx, 1 otherwise. NOTE: negative tests (e.g. unload-while-
# serving) EXPECT a non-2xx -- the caller decides pass/fail from context.
_rest_post() {
  local url="$1" json="$2" label="$3" body code
  # Echo the EXACT curl command (URL + headers + full JSON body) for reference,
  # so every API call is reproducible from the log. Clean, copy-pasteable block.
  echo "[${label}] >>> REST API: POST ${url}"
  echo "    curl -s -X POST '${url}' \\"
  echo "         -H 'Content-Type: application/json' \\"
  echo "         -d '${json}'"
  local cerr; cerr="$(mktemp 2>/dev/null || echo /tmp/.modelmux_curl_err)"
  # BOUNDED. Without --max-time a server that accepts the connection and never
  # answers blocks curl forever, which blocks the caller forever. In the stress
  # harness that silently kills the workload: the async churn worker removes its
  # batch, hangs on one REST call, and never reaches the re-add half of the loop,
  # so the pipeline drains to zero sources and the run continues for the rest of
  # STRESS_DURATION reporting 0 fps -- which reads like a plugin failure but is
  # really the harness sitting idle. A timeout turns that into HTTP 000, which the
  # check below already surfaces loudly and returns non-zero for.
  # Generous on purpose (these POSTs are synchronous admission and normally answer
  # in ms); override with MM_REST_TIMEOUT if a slow path ever needs longer.
  # %{content_type} rides along so the DECLARED media type is assertable. The
  # harness previously captured only the status and the body, so a response that
  # announced JSON as text/plain was invisible to every test.
  body="$(curl -s -w $'\n%{http_code}\n%{content_type}' -X POST "${url}" \
            --connect-timeout "${MM_REST_CONNECT_TIMEOUT:-5}" \
            --max-time "${MM_REST_TIMEOUT:-30}" \
            -H "Content-Type: application/json" -d "${json}" 2>"${cerr}")"
  local cx=$?
  body="${body%$'\n'*}"
  code="${body##*$'\n'}"
  body="${body%$'\n'*}"
  # keep the LAST response body queryable (assert_rest_body_contains) -- the async
  # POSTs echo correlation handles (e.g. model/update's "operation_id") only here.
  MM_REST_POST_BODY="${body}"
  # keep the EXACT status too. Returning 0/1 on 2xx-vs-not is enough to know a
  # request was refused, but not WHICH refusal: 400, 404 and 409 are one bucket
  # to every caller below. The unified contract distinguishes them, so a case
  # that means to pin a status needs the number, not the verdict.
  MM_REST_POST_CODE="${code}"
  echo "[${label}] <<< HTTP ${code}  ${body}"
  # HTTP 000 means curl got no response. Surface curl's exit code so a true server crash
  # (exit 52 'empty reply' then exit 7 'connection refused' once the app dies) is not
  # mistaken for a benign rejection. A common cause: stale/mismatched installed plugin
  # artifacts -- check APP_BIN and DS_ROOT.
  if [ "${code}" = "000" ]; then
    echo "[${label}] !!! curl could not get a response (exit=${cx}: $(cat "${cerr}" 2>/dev/null)) -- REST server unreachable/crashed"
  fi
  rm -f "${cerr}" 2>/dev/null
  case "${code}" in 2*) return 0 ;; *) return 1 ;; esac
}

# add_stream <camera_id> <camera_name> <uri> \
#            [p_name] [p_ver] [s_name] [s_ver]
#
# The model args are OPTIONAL and ride inside metadata as v1 NESTED refs (API_DESIGN.md 5.6):
#   "metadata": { ..., "model": {"name": ..., "version": ...}, "shadow": {"name", "version"} }
# A ref REQUIRES an explicit version (v1 never defaults to 1 server-side); when a caller
# passes p_name/s_name without a version, this helper pins version "1" explicitly to
# preserve the old default-to-1 call-site semantics. Config is NOT passed here --
# stream/add never loads a model; it routes to an already-loaded one (defined in the
# catalog / loaded via model/load), so the config comes from there. Omit the model args
# entirely to use the runtime defaults. Per-role graceful fallback applies: an
# unavailable/not-ready (name,version) falls back to that role's default model.
#   add_stream <cam> <name> <uri> [p_name] [p_ver] [s_name] [s_ver]
# Shadow opt-out: pass s_name="none" (no version needed) to run this stream PRIMARY ONLY --
# v1 spells that as "shadow": null. It does NOT attach the configured default shadow to
# this stream. This only affects routing for THIS stream; the default shadow model stays
# loaded/serving other streams (no unload).
add_stream() {
  local camera_id="$1" camera_name="$2" uri="$3"
  local p_name="$4" p_ver="$5" s_name="$6" s_ver="$7"

  # build the optional model-selection refs only for the args that were given.
  # version is REQUIRED inside a v1 ref -> pin "1" explicitly when omitted by the caller.
  local model_meta=""
  [ -n "${p_name}" ] && model_meta="${model_meta}, \"model\": {\"name\": \"${p_name}\", \"version\": \"${p_ver:-1}\"}"
  if [ -n "${s_name}" ]; then
    if [ "${s_name}" = "none" ]; then
      model_meta="${model_meta}, \"shadow\": null"
    else
      model_meta="${model_meta}, \"shadow\": {\"name\": \"${s_name}\", \"version\": \"${s_ver:-1}\"}"
    fi
  fi

  echo "[stream_add] adding camera_id='${camera_id}' name='${camera_name}'"
  echo "[stream_add]   uri=${uri}"
  [ -n "${p_name}" ] && echo "[stream_add]   primary=${p_name} v=${p_ver:-1}"
  [ -n "${s_name}" ] && echo "[stream_add]   shadow=${s_name} v=${s_ver:-1}"
  _rest_post "${ADD_URL}" "{
      \"key\": \"sensor\",
      \"value\": {
        \"camera_id\": \"${camera_id}\",
        \"camera_name\": \"${camera_name}\",
        \"camera_url\": \"${uri}\",
        \"change\": \"camera_add\",
        \"metadata\": { \"resolution\": \"1920 x1080\", \"codec\": \"h264\", \"framerate\": 30${model_meta} }
      }
    }" "stream_add"
}

remove_stream() {
  local camera_id="$1" uri="$2"
  echo "[stream_remove] removing camera_id='${camera_id}'"
  _rest_post "${REMOVE_URL}" "{
      \"key\": \"sensor\",
      \"value\": {
        \"camera_id\": \"${camera_id}\",
        \"camera_url\": \"${uri}\",
        \"change\": \"camera_remove\"
      }
    }" "stream_remove"
}

# load_model <name> <version> [engine_file] [config_file] [def_primary] [def_shadow] [batch]
# TYPE-LESS load (v1 model plane): model/load REQUIRES name + version (positive integer);
# config_file is required only on the FIRST load of a name (a known catalog name reuses its
# base config). It is the SOLE engine ingest: pass an optional prebuilt engine to load a
# specific checkpoint ("" => the config's engine). The model warms into limbo; which streams
# use it -- and in which role -- is decided exclusively by the ROUTING plane (model/routing /
# stream-add metadata).
#
# def_primary / def_shadow (truthy: True/true/1/yes): v1 REMOVED set_default_primary/
# set_default_shadow from model/load (a load can never move traffic -- strict plane
# separation). To preserve the old call-site semantics, a truthy flag makes this helper
# (a) POST v1 model/load, (b) wait_for_model until the version is loaded (the default route
# requires a LOADED target and the load is async), then (c) POST /api/v1/model/routing with
# {"default": {"model"|"shadow": {"name", "version"}}} -- the v1 way to make a loaded model
# the runtime default for that role.
load_model() {
  local name="$1" version="$2" engine="$3" model_cfg="$4"
  local def_primary="$5" def_shadow="$6" batch="$7"
  # name is mandatory -> start the body with it (no leading comma); append only the
  # optional fields that were given (config_file omitted => reuse the catalog base).
  local body="\"name\": \"${name}\""
  [ -n "${version}" ]   && body="${body}, \"version\": \"${version}\""
  [ -n "${engine}" ]    && body="${body}, \"engine_file\": \"${engine}\""
  [ -n "${model_cfg}" ] && body="${body}, \"config_file\": \"${model_cfg}\""
  # OPT per-version batch/stream cap (JSON INTEGER -- the admission rejects strings)
  [ -n "${batch}" ]     && body="${body}, \"batch_size\": ${batch}"
  echo "[model_load] name='${name}' version='${version:-<none>}' engine='${engine:-<from config>}' config='${model_cfg:-<catalog>}' default_primary='${def_primary:-no}' default_shadow='${def_shadow:-no}'"
  _rest_post "${LOAD_URL}" "{
      \"key\": \"model\",
      \"value\": { ${body} }
    }" "model_load"
  local rc=$?
  # v1 default designation = a separate stream-plane call (route default). The route target
  # must be LOADED, and the load above is async -> poll model/status first (wait_for_model).
  # NOTE: v1 refs REQUIRE a version; "1" is pinned when the caller omitted it.
  case "${def_primary}" in [Tt]rue|1|[Yy]es)
    wait_for_model "${name}" "${version:-1}"
    model_routing_default "model" "${name}" "${version:-1}" ;;
  esac
  case "${def_shadow}" in [Tt]rue|1|[Yy]es)
    wait_for_model "${name}" "${version:-1}"
    model_routing_default "shadow" "${name}" "${version:-1}" ;;
  esac
  return ${rc}
}

# unload_model <name> [version]   (TYPE-LESS: frees the model wherever it lives;
# version omitted => v1 unloads every non-serving version of the name)
unload_model() {
  local name="$1" version="$2"
  local body="\"name\": \"${name}\""
  [ -n "${version}" ] && body="${body}, \"version\": \"${version}\""
  echo "[model_unload] name='${name}' version='${version:-<all>}'"
  _rest_post "${UNLOAD_URL}" "{
      \"key\": \"model\",
      \"value\": { ${body} }
    }" "model_unload"
}

# model_routing <stream_csv|""> [p_name] [p_ver] [s_name] [s_ver]
# The v1 ROUTING plane: POST /api/v1/model/routing (replaces the retired routing-flavored
# model/update). Re-routes the in-scope streams to different ALREADY-LOADED (READY/SERVING)
# models -- NO engine file (engine ingest is model/load only) and NO promote flag: a promote
# or swap is DERIVED from the desired state. Scope ('streams' is REQUIRED per route):
#   stream_csv non-empty -> "streams": ["cam", ...] (explicit camera_id list)
#   stream_csv == ""      -> "streams": "all" (the literal; v1 never treats an omitted
#                            list as "everything")
# Only the NAMED roles change: an omitted role field leaves that role unchanged; a route
# that names NEITHER role is a v1 admission error. shadow == "none" clears the shadow
# (v1: "shadow": null). A swap is naming BOTH (model=<old shadow>, shadow=<old primary>);
# the app detects the flip and does the zero-drop swap. Refs REQUIRE a version; "1" is
# pinned when a caller passes a name without one.
model_routing() {
  local streams="$1" p_name="$2" p_ver="$3" s_name="$4" s_ver="$5"
  local scope roles="" sl="" cams c
  if [ -n "${streams}" ]; then
    IFS=',' read -ra cams <<< "${streams}"
    for c in "${cams[@]}"; do sl="${sl:+${sl}, }\"${c}\""; done
    scope="[${sl}]"
  else
    scope="\"all\""
  fi
  [ -n "${p_name}" ] && roles="\"model\": {\"name\": \"${p_name}\", \"version\": \"${p_ver:-1}\"}"
  if [ -n "${s_name}" ]; then
    if [ "${s_name}" = "none" ]; then
      roles="${roles:+${roles}, }\"shadow\": null"
    else
      roles="${roles:+${roles}, }\"shadow\": {\"name\": \"${s_name}\", \"version\": \"${s_ver:-1}\"}"
    fi
  fi
  echo "[model_routing] streams='${streams:-ALL}' model='${p_name:-<keep>}${p_ver:+@${p_ver}}' shadow='${s_name:-<keep>}${s_ver:+@${s_ver}}'"
  _rest_post "${ROUTE_URL}" "{
      \"key\": \"routing\",
      \"value\": { \"routes\": [ { \"streams\": ${scope}${roles:+, ${roles}} } ] }
    }" "model_routing"
}

# model_routing_default <model|shadow> <name> <version>
# Move a role's RUNTIME DEFAULT to an already-loaded (name, version) via the v1 routing
# plane: {"default": {"model"|"shadow": {"name", "version"}}}. Followers of the old
# default reroute zero-drop.
model_routing_default() {
  local role="$1" name="$2" version="$3"
  echo "[model_routing] default ${role} -> '${name}@${version}'"
  _rest_post "${ROUTE_URL}" "{
      \"key\": \"routing\",
      \"value\": { \"default\": { \"${role}\": {\"name\": \"${name}\", \"version\": \"${version}\"} } }
    }" "model_routing_default"
}

# model_routing_default_both <p_name> <p_ver> <s_name|none|-> [s_ver]
# Write BOTH default roles in ONE model/routing request. The single-role helper above
# cannot express the operations that depend on the two roles moving together -- a SWAP
# (the roles exchange) and a PROMOTE (the shadow becomes the primary and the A/B ends) --
# because sending them as two requests passes through an intermediate state the
# both-roles rule forbids. <s_name>:
#   "none"    -> "shadow": null   (clear the default shadow)
#   "-" or "" -> omit "shadow"    (leave the current default shadow untouched)
#   anything else -> "shadow": {"name": ..., "version": ...}
model_routing_default_both() {
  local p_name="$1" p_ver="$2" s_name="$3" s_ver="$4"
  local body=""
  [ -n "${p_name}" ] && body="\"model\": {\"name\": \"${p_name}\", \"version\": \"${p_ver:-1}\"}"
  case "${s_name}" in
    none)  body="${body:+${body}, }\"shadow\": null" ;;
    ""|-)  ;;
    *)     body="${body:+${body}, }\"shadow\": {\"name\": \"${s_name}\", \"version\": \"${s_ver:-1}\"}" ;;
  esac
  echo "[model_routing] default { model='${p_name:-<keep>}${p_ver:+@${p_ver}}' shadow='${s_name:-<keep>}' }"
  _rest_post "${ROUTE_URL}" "{
      \"key\": \"routing\",
      \"value\": { \"default\": { ${body} } }
    }" "model_routing_default_both"
}

# model_update <name> <from_version> <version> <engine_file>
# The v1 model/update endpoint: POST /api/v1/model/update -- an in-place checkpoint
# transition. Same network, new weights: the prebuilt engine is hot-swapped on the LIVE
# (name, from_version) instance across all its shards (no new bin, no reroute, gie kept;
# the streams never move) and the version identity moves from_version -> version.
# from_version is a REQUIRED CAS guard naming the currently-live version; engine_file is
# REQUIRED (model/update never auto-builds). The accepted response echoes an
# "operation_id":"update-NNNNNN" correlation handle (see assert_rest_body_contains /
# MM_REST_POST_BODY). NOTE: the retired routing-flavored model_update alias is gone --
# a reroute is model_routing (run_stress.sh now calls it directly).
model_update() {
  local name="$1" from_version="$2" version="$3" engine="$4" batch="$5"
  local extra=""
  [ -n "${batch}" ] && extra=", \"batch_size\": ${batch}"   # JSON INTEGER
  echo "[model_update] name='${name}' from_version='${from_version}' -> version='${version}' engine='${engine}'${batch:+ batch=${batch}}"
  _rest_post "${MODEL_UPDATE_URL}" "{
      \"key\": \"model\",
      \"value\": {
        \"name\": \"${name}\",
        \"from_version\": \"${from_version}\",
        \"version\": \"${version}\",
        \"engine_file\": \"${engine}\"${extra}
      }
    }" "model_update"
}

# assert_rest_body_contains <fixed-substring> <description>
#   PASS if the LAST _rest_post response body contains the substring. Whitespace-
#   insensitive (jsoncpp pretty-prints "k" : "v"; pass patterns as "k":"v") -- the
#   POST-response counterpart of assert_status_contains. Works in AUTO and manual
#   mode (reads the HTTP response, not the app log).
assert_rest_body_contains() {
  local body="${MM_REST_POST_BODY//[$' \t\n\r']/}"   # strip all whitespace
  case "${body}" in
    *"$1"*) _pass "$2" ;;
    *)      _fail "$2  (REST response body lacks: '$1')" ;;
  esac
}

# assert_rest_body_absent <fixed-substring> <description>
#   PASS if the LAST _rest_post response body does NOT contain the substring.
#   Negative counterpart of assert_rest_body_contains (same whitespace stripping).
#
#   Why a NEGATIVE body check earns its place: a status/code pair says the request
#   was accepted or refused, but not WHICH rule decided it. When two rules can both
#   reject an op, only pinning the absence of the wrong one's wording proves the
#   right rule ran -- see test_54, where a version-scoped unload must never be
#   refused by the NAME-level configured-default guard.
assert_rest_body_absent() {
  local body="${MM_REST_POST_BODY//[$' \t\n\r']/}"
  case "${body}" in
    *"$1"*) _fail "$2  (REST response body unexpectedly contains: '$1')" ;;
    *)      _pass "$2" ;;
  esac
}

# assert_http_status <expected-code> <description>
#   PASS if the LAST _rest_post returned EXACTLY this HTTP status.
#
#   Why this is not the same as "the request failed": every existing negative
#   assertion here is satisfied by any non-2xx, so 400, 404 and 409 are
#   interchangeable to the suite. A remap that returned the wrong one -- or that
#   never took effect at all -- passes every one of those checks. Pinning the
#   number is the only way a status contract can be verified rather than assumed.
assert_http_status() {
  if [ "${MM_REST_POST_CODE}" = "$1" ]; then
    _pass "$2"
  else
    _fail "$2  (expected HTTP $1, got ${MM_REST_POST_CODE:-<none>})"
  fi
}

# assert_error_code <expected-code> <description>
#   PASS if the LAST _rest_post body carries error.code == <expected-code>.
#   Pairs with assert_http_status: the status says which CLASS of failure, the
#   code says WHICH failure. A client is required to branch on the code, so a
#   reject that carries the right status with a generic code is still wrong.
assert_error_code() {
  local body="${MM_REST_POST_BODY//[$' \t\n\r']/}"
  case "${body}" in
    *"\"code\":\"$1\""*) _pass "$2" ;;
    *) _fail "$2  (REST body lacks error.code '$1'; got: ${MM_REST_POST_BODY})" ;;
  esac
}

# get_model_status: GET /api/v1/model/status. Captures the JSON body into the global
# MODEL_STATUS_BODY (and prints it). Unlike the async POST ops, this returns the real
# state synchronously, so it CAN be asserted directly on the response body.
MODEL_STATUS_BODY=""
# get_model_status [query-string]
#   GET /api/v1/model/status, optionally filtered. Pass the raw query-string (no '?')
#   using the v1 param names (model / version / stream / source_id), e.g.
#         get_model_status "model=Person"
#         get_model_status "stream=Cam-1"
#         get_model_status "source_id=1"
#         get_model_status "model=Person&version=1&stream=Cam-1"
#   Omit the arg for the full (unfiltered) dump. Result lands in MODEL_STATUS_BODY.
#   NOTE v1 filters JOIN (never orphan a view): ?model=X also returns the streams routed
#   to X; ?stream=Y also returns the model versions Y uses.
get_model_status() {
  local q="$1" code url
  url="${STATUS_URL}${q:+?${q}}"
  # Echo the EXACT curl command (full URL incl. any query params) for reference.
  echo "[model_status] >>> REST API: GET ${url}"
  echo "    curl -s '${url}'"
  # bounded: this GET blocks on a synchronous downstream query, so an element that
  # never answers would hang the test forever without --max-time (HTTP 000 instead).
  MODEL_STATUS_BODY="$(curl -s -w $'\n%{http_code}' \
      --connect-timeout "${MM_REST_CONNECT_TIMEOUT:-5}" --max-time "${MM_REST_TIMEOUT:-30}" "${url}")"
  code="${MODEL_STATUS_BODY##*$'\n'}"
  MM_REST_POST_CODE="${code}"   # so assert_http_status after a GET reads THIS call
  MODEL_STATUS_BODY="${MODEL_STATUS_BODY%$'\n'*}"
  echo "[model_status] <<< HTTP ${code}  ${MODEL_STATUS_BODY}"
  case "${code}" in 2*) return 0 ;; *) return 1 ;; esac
}

# wait_for_model <model_name>
# Simple poll loop: every 2s, GET /api/v1/model/status?model=<name> and check if the
# (name, version) is loaded (state ready/serving); return as soon as it is. Use right
# after load_model so the stream added next doesn't race the warm-up and fall back to
# the default primary. (model/status reports a warmed-but-unassigned limbo model too.)
wait_for_model() {
  # wait_for_model <name> [version]  -- waits for the SPECIFIC (name, version),
  # parsed as JSON (order-independent; never assumes serializer key order).
  local model="$1" version="${2:-${MM_MODEL_VERSION_DEFAULT:-1}}"
  echo "[wait_for_model] waiting for '${model}@${version}' to load (poll model/status every 2s) ..."
  while true; do
    get_model_status "model=${model}" >/dev/null 2>&1        # fills MODEL_STATUS_BODY
    # ORDER-INDEPENDENT check (never assume serializer key order): walk the v1
    # status JSON and pass when THIS (name, version) entry is ready/serving.
    if printf '%s' "${MODEL_STATUS_BODY}" | python3 -c '
import json, sys
name, ver = sys.argv[1], sys.argv[2]
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ms = doc.get("model-status", doc)   # tolerate enveloped or bare payloads
for m in (ms.get("models") or []):
    if m.get("name") != name:
        continue
    for v in (m.get("versions") or []):
        if v.get("version") == ver and v.get("state") in ("ready", "serving"):
            sys.exit(0)
sys.exit(1)' "${model}" "${version}"; then
      _pass "wait_for_model: '${model}@${version}' loaded (ready)"
      return 0
    fi
    sleep 2
  done
}

# assert_status_contains <fixed-substring> <description>: PASS if the last
# get_model_status response body contains the substring. Whitespace-insensitive
# (jsoncpp pretty-prints "k" : "v" with spaces; pass patterns as "k":"v"). Works in
# AUTO and manual mode (reads the HTTP response, not the app log).
assert_status_contains() {
  local body="${MODEL_STATUS_BODY//[$' \t\n\r']/}"   # strip all whitespace
  case "${body}" in
    *"$1"*) _pass "$2" ;;
    *)      _fail "$2  (model/status body lacks: '$1')" ;;
  esac
}

# assert_status_model_listed <model-name> <description>
#   PASS if the LAST model/status body has an entry in models[] whose name matches.
#
#   Deliberately NOT a substring check on the whole body: the same name appears
#   inside streams[].pending.model, so assert_status_contains would pass on the
#   stream's own reference even when models[] omits the model entirely -- exactly
#   the orphan this asserts against. Only models[] counts.
assert_status_model_listed() {
  if printf '%s' "${MODEL_STATUS_BODY}" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ms = doc.get("model-status", doc)
sys.exit(0 if any(m.get("name") == sys.argv[1]
                  for m in (ms.get("models") or [])) else 1)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (models[] has no entry named '$1')"
  fi
}

# assert_status_not_contains <fixed-substring> <description>
#   PASS if the LAST model/status body does NOT contain the (whitespace-stripped) substring.
#   Used to prove a filter EXCLUDED something (e.g. a model-only query has no stream entries).
assert_status_not_contains() {
  local body="${MODEL_STATUS_BODY//[$' \t\n\r']/}"   # strip all whitespace
  case "${body}" in
    *"$1"*) _fail "$2  (model/status body unexpectedly contains: '$1')" ;;
    *)      _pass "$2" ;;
  esac
}

# assert_routing_contains <fixed-substring> <description>
#   The model/routing twin of assert_status_contains: the routing plane otherwise has
#   only structural (python) assertions, so a plain "is this key published at all"
#   check had no helper. Reads MODEL_ROUTING_BODY -- assert_status_contains reads the
#   STATUS body, and using it after get_model_routing silently tests a stale response.
assert_routing_contains() {
  local body="${MODEL_ROUTING_BODY//[$' \t\n\r']/}"   # strip all whitespace
  case "${body}" in
    *"$1"*) _pass "$2" ;;
    *)      _fail "$2  (model/routing body lacks: '$1')" ;;
  esac
}

# routing_revision <- MODEL_ROUTING_BODY
#   Echoes the routing_revision scalar of the last get_model_routing (empty if the
#   key is absent). It is the if_revision CAS token, and there is exactly ONE of it
#   per pipeline however many nvmodelmux stages answered the query -- every stage
#   shares the counter through the routing GstContext, so what this prints is the
#   pipeline's revision, not the primary's.
routing_revision() {
  # the REST server wraps the plugin body: {"reason":..., "routing":{...}} -- look
  # inside the envelope first, then at the top level, so the helper survives either.
  printf '%s' "${MODEL_ROUTING_BODY}" | \
      python3 -c 'import json,sys
try:
  d = json.load(sys.stdin)
  d = d.get("routing", d) if isinstance(d, dict) else {}
  print(d.get("routing_revision", ""))
except Exception: print("")' 2>/dev/null
}

# assert_revision_advanced <before> <description>
#   The revision must be strictly GREATER than <before>. Used after an operation that
#   changed routing state ANYWHERE in the pipeline: a controller still holding <before>
#   has to lose its next compare-and-swap. Numeric, not substring -- "moved forward" is
#   the property, and any particular value would just encode the test's own history.
assert_revision_advanced() {
  local before="$1" desc="$2" now
  now="$(routing_revision)"
  if [ -n "${now}" ] && [ -n "${before}" ] && [ "${now}" -gt "${before}" ] 2>/dev/null; then
    _pass "${desc}  (routing_revision ${before} -> ${now})"
  else
    _fail "${desc}  (routing_revision did not advance: ${before} -> ${now:-<absent>})"
  fi
}

# ---------------------------------------------------------------------------
# GET /api/v1/model/routing -- the routing plane as a read-only snapshot.
#
# Same uri as the POST, different verb. The response reuses the POST's vocabulary
# (routes[] + default{} + routing_revision) but is NOT a POST body: it carries
# read-only fields (state / origin / pending) the parser would reject, and it can
# express states the POST refuses (a stream with no model at all).
#
# Entries are GROUPED: routes[].streams lists every stream whose routing state is
# identical. The grouping key is the whole entry minus `streams`, so the response
# is asserted STRUCTURALLY (python3, like wait_for_model) rather than by substring
# -- "these two cameras share one entry" is not a substring property.
MODEL_ROUTING_BODY=""
# get_model_routing [query-string]
#   GET /api/v1/model/routing, optionally filtered. v1 accepts one filter, ?stream=
#   (the point lookup); grouping already collapses a uniform fleet to one entry, so
#   the others buy little. Result lands in MODEL_ROUTING_BODY.
get_model_routing() {
  local q="$1" code url
  url="${ROUTE_URL}${q:+?${q}}"
  echo "[model_routing_get] >>> REST API: GET ${url}"
  echo "    curl -s '${url}'"
  # bounded for the same reason as model/status: the routing snapshot is answered by a
  # synchronous peer_query into nvmodelmux.
  MODEL_ROUTING_BODY="$(curl -s -w $'\n%{http_code}' \
      --connect-timeout "${MM_REST_CONNECT_TIMEOUT:-5}" --max-time "${MM_REST_TIMEOUT:-30}" "${url}")"
  code="${MODEL_ROUTING_BODY##*$'\n'}"
  MM_REST_POST_CODE="${code}"   # so assert_http_status after a GET reads THIS call
  MODEL_ROUTING_BODY="${MODEL_ROUTING_BODY%$'\n'*}"
  echo "[model_routing_get] <<< HTTP ${code}  ${MODEL_ROUTING_BODY}"
  case "${code}" in 2*) return 0 ;; *) return 1 ;; esac
}

# assert_route_entry <csv-cameras> <expect-key=value,...> <description>
#   PASS if the last model/routing body has an entry whose `streams` set is EXACTLY
#   the given cameras AND whose scalar expectations all hold. Expectations use dotted
#   paths against the entry: state, origin, model.name, model.version, shadow.name,
#   pending.model.name, ...
#
#   Asserting the stream SET (not "contains") is the point: it proves grouping merged
#   exactly these cameras and split everything else away.
assert_route_entry() {
  local cams="$1" expect="$2" desc="$3"
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
want = sorted(x for x in sys.argv[1].split(",") if x)
pairs = [p for p in sys.argv[2].split(",") if p]
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sr = doc.get("routing", doc)
for e in (sr.get("routes") or []):
    if sorted(e.get("streams") or []) != want:
        continue
    for p in pairs:                       # every expectation must hold on THIS entry
        k, _, v = p.partition("=")
        cur = e
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if str(cur) != v:
            break
    else:
        sys.exit(0)
sys.exit(1)' "${cams}" "${expect}"; then
    _pass "${desc}"
  else
    _fail "${desc}  (no routes[] entry with streams=={${cams}} and ${expect})"
  fi
}

# assert_route_count <n> <description>: PASS if routes[] has exactly n entries.
#   Entry count is the health signal -- one entry means the fleet is uniform, and
#   each extra entry is a real divergence (different model, device, shadow, origin
#   or a pending promote).
assert_route_count() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sr = doc.get("routing", doc)
sys.exit(0 if len(sr.get("routes") or []) == int(sys.argv[1]) else 1)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (routes[] entry count != $1)"
  fi
}

# assert_route_no_leak <description>
#   PASS if no ref anywhere in the response carries a model-status-only field.
#   Refs are built by WHITELIST ({name, version, origin} + gpu|placement); a leak of the
#   perf pair would be silent but fatal -- fps/latency_ms change every measurement window,
#   so no two streams would ever compare equal and grouping would never merge.
#   NOTE `origin` is legitimately PER-REF (the primary and shadow lanes can disagree),
#   so it is not banned here.
assert_route_no_leak() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
BANNED = {"fps", "latency_ms", "infer_latency_ms", "instance", "backend",
          "gie_id", "engine"}
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sr = doc.get("routing", doc)
def refs(e):
    for k in ("model", "shadow"):
        if isinstance(e.get(k), dict):
            yield e[k]
    p = e.get("pending")
    if isinstance(p, dict):
        # BOTH pending lanes. pending.shadow is built by the same s_route_ref as
        # every other ref, so a whitelist regression there is just as fatal to
        # grouping -- and it was the one ref this assertion never looked at, so a
        # leak into it passed silently. isinstance guards the null case: a
        # pending shadow-CLEAR is spelled `null`, which is not a ref to check.
        for k in ("model", "shadow"):
            if isinstance(p.get(k), dict):
                yield p[k]
for e in (sr.get("routes") or []):
    for r in refs(e):
        if BANNED & set(r.keys()):
            sys.exit(1)
for k in ("model", "shadow"):                 # default{} refs too
    d = (sr.get("default") or {}).get(k)
    if isinstance(d, dict) and BANNED & set(d.keys()):
        sys.exit(1)
sys.exit(0)'; then
    _pass "$1"
  else
    _fail "$1  (a ref carries a model-status-only field: one of fps/latency_ms/infer_latency_ms/instance/backend/gie_id/engine -- NOT origin, which the routing plane requires)"
  fi
}

# assert_route_no_refs <csv-cameras> <description>
#   PASS if the entry holding exactly these cameras carries NEITHER `model` NOR `pending`
#   (nor `shadow`). That combination is the "nothing is coming" state -- an overflow past
#   the per-model cap with model-sharding=0, or a route whose target never warmed. It is
#   what distinguishes ACT from WAIT, so asserting `state` alone is not enough: a bug that
#   emitted a stale `model` beside state="passthrough" would still pass a state check.
assert_route_no_refs() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
want = sorted(x for x in sys.argv[1].split(",") if x)
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for e in ((doc.get("routing", doc) or {}).get("routes") or []):
    if sorted(e.get("streams") or []) != want:
        continue
    sys.exit(1 if ("model" in e or "shadow" in e or "pending" in e) else 0)
sys.exit(1)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (entry {$1} is missing, or still carries model/shadow/pending)"
  fi
}

# assert_route_deterministic <description>
#   GET twice back to back and PASS only if the two bodies are byte-identical.
#   The projection buckets streams out of a HASH TABLE, so without the explicit sort
#   (streams within an entry, entries by first stream) two consecutive reads could
#   differ with nothing changed -- churn that would defeat any diff-based reconciler.
assert_route_deterministic() {
  local a b
  get_model_routing >/dev/null 2>&1; a="${MODEL_ROUTING_BODY}"
  get_model_routing >/dev/null 2>&1; b="${MODEL_ROUTING_BODY}"
  if [ "${a}" = "${b}" ]; then
    _pass "$1"
  else
    _fail "$1  (two consecutive GETs differ with no intervening change)"
  fi
}

# assert_route_default <expect-key=value,...> <description>
#   PASS if the snapshot's top-level default{} matches. Paths are dotted against the
#   default object: model.name, model.version, shadow.name, ...
#   The default block is the designation itself, so this is how a test proves a
#   default{} MOVE actually took (as opposed to individual lanes following it).
assert_route_default() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
pairs = [p for p in sys.argv[1].split(",") if p]
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
d = (doc.get("routing", doc) or {}).get("default")
if not isinstance(d, dict):
    sys.exit(1)
for p in pairs:
    k, _, v = p.partition("=")
    cur = d
    for part in k.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    if str(cur) != v:
        sys.exit(1)
sys.exit(0)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (default{} does not match ${1})"
  fi
}

# assert_route_sweep_count <n> <description>
#   PASS if exactly n streams report model.origin == "default" -- i.e. the BLAST RADIUS
#   of the next default{} move. This is the number an operator wants BEFORE issuing it;
#   the whole point of exposing `origin` is that it is countable in advance.
assert_route_sweep_count() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sr = doc.get("routing", doc)
n = 0
for e in (sr.get("routes") or []):
    m = e.get("model")
    if isinstance(m, dict) and m.get("origin") == "default":
        n += len(e.get("streams") or [])
sys.exit(0 if n == int(sys.argv[1]) else 1)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (default-origin stream count != $1)"
  fi
}

# assert_route_has_key / assert_route_lacks_key <dotted-path> <description>
#   Presence checks on the top-level snapshot, e.g. "default" (absent under ?stream=)
#   or "routing_revision" (always present, filtered or not -- it is the CAS token).
assert_route_has_key() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
cur = doc.get("routing", doc)
for part in sys.argv[1].split("."):
    if not isinstance(cur, dict) or part not in cur:
        sys.exit(1)
    cur = cur[part]
sys.exit(0)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (snapshot is missing '$1')"
  fi
}

# assert_route_entry_lacks <csv-cameras> <dotted-key> <description>
#   PASS if the entry whose `streams` set is EXACTLY those cameras does NOT carry the
#   dotted key. The per-ENTRY counterpart of assert_route_lacks_key (which walks from
#   the document root). Absence is the assertion that matters for `pending`: a stream
#   with no queued route must carry no pending key at all, not an empty object -- an
#   empty one would read as "something is coming" to a reconciler.
assert_route_entry_lacks() {
  local cams="$1" key="$2" desc="$3"
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
want = sorted(x for x in sys.argv[1].split(",") if x)
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sr = doc.get("routing", doc)
for e in (sr.get("routes") or []):
    if sorted(e.get("streams") or []) != want:
        continue
    cur = e
    for part in sys.argv[2].split("."):
        if not isinstance(cur, dict) or part not in cur:
            sys.exit(0)            # absent on the matching entry -> PASS
        cur = cur[part]
    sys.exit(1)                    # present -> FAIL
sys.exit(1)                        # no such entry at all -> FAIL
' "${cams}" "${key}"; then
    _pass "${desc}"
  else
    _fail "${desc}  (entry streams=={${cams}} is missing, or carries '${key}')"
  fi
}

# assert_route_entry_null <csv-cameras> <dotted-key> <description>
#   PASS only if the entry whose `streams` set is EXACTLY those cameras carries the
#   dotted key AND its value is JSON null.
#
#   Needed because a dotted-path walk cannot tell null from absent -- both come
#   back as Python None -- yet that is exactly the distinction under test for a
#   pending shadow: null means "the apply will CLEAR it", absent means "leave it
#   alone". Asserting "==None" would pass on either and prove nothing.
assert_route_entry_null() {
  local cams="$1" key="$2" desc="$3"
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
want = sorted(x for x in sys.argv[1].split(",") if x)
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sr = doc.get("routing", doc)
parts = sys.argv[2].split(".")
for e in (sr.get("routes") or []):
    if sorted(e.get("streams") or []) != want:
        continue
    cur = e
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            sys.exit(1)
        cur = cur[part]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        sys.exit(1)                    # ABSENT -> not the same as null -> FAIL
    sys.exit(0 if cur[parts[-1]] is None else 1)
sys.exit(1)
' "${cams}" "${key}"; then
    _pass "${desc}"
  else
    _fail "${desc}  (entry streams=={${cams}} lacks '${key}', or it is not null)"
  fi
}

assert_route_lacks_key() {
  if printf '%s' "${MODEL_ROUTING_BODY}" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
cur = doc.get("routing", doc)
for part in sys.argv[1].split("."):
    if not isinstance(cur, dict) or part not in cur:
        sys.exit(0)
    cur = cur[part]
sys.exit(1)' "$1"; then
    _pass "$2"
  else
    _fail "$2  (snapshot unexpectedly carries '$1')"
  fi
}

# assert_status_eventually <query> <fixed-substring> <description> [timeout-secs]
#   Re-query model/status until the body CONTAINS the substring, or timeout.
#
#   A live model/routing reroute is an ASYNCHRONOUS zero-drop hitless swap: the 'model/update
#   applied' log marker (still the plugin's wording for an applied scoped reroute) fires when
#   the swap STARTS, but the new model becomes the live
#   (queryable) routing only when the swap COMPLETES a few hundred ms later. A one-shot
#   status query right after the accept marker therefore races the swap and reads the OLD
#   model. Polling both (a) validates the swap's EFFECT and (b) gates the next operation
#   until the stream has settled -- so a following update is not issued mid-swap and
#   rejected as "no streams updated". Use this instead of get_model_status +
#   assert_status_contains whenever the query follows a live update.
assert_status_eventually() {
  local q="$1" want="$2" desc="$3" to="${4:-12}" i=0 body
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    get_model_status "${q}" >/dev/null 2>&1
    body="${MODEL_STATUS_BODY//[$' \t\n\r']/}"
    case "${body}" in *"${want}"*) _pass "${desc}  (settled after ${i}s)"; return 0 ;; esac
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (model/status never contained '${want}' within ${to}s)"
  return 1
}

# assert_status_eventually_absent <query> <fixed-substring> <description> [timeout-secs]
#   Re-query model/status until the body NO LONGER contains the substring, or timeout.
#   Async counterpart of assert_status_not_contains (e.g. wait for a shadow to clear).
assert_status_eventually_absent() {
  local q="$1" gone="$2" desc="$3" to="${4:-12}" i=0 body
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    get_model_status "${q}" >/dev/null 2>&1
    body="${MODEL_STATUS_BODY//[$' \t\n\r']/}"
    case "${body}" in *"${gone}"*) ;; *) _pass "${desc}  (settled after ${i}s)"; return 0 ;; esac
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (model/status still contained '${gone}' after ${to}s)"
  return 1
}

# _status_has_ref <role: model|shadow> <name> <version>
#   Returns 0 iff SOME streams[] entry in the LAST model/status body carries the given
#   <role> ref with that (name, version). Parsed as JSON (order-independent, envelope-
#   tolerant) and deliberately INSTANCE-AGNOSTIC: v1 model/status truthfully reports the
#   ACTUAL serving shard, and streams can land on an auto-grown shard instance 1+ when
#   shard 0 is full -- "instance":N is scheduler-dependent and must not be pinned.
#   ("gpu" is likewise not pinned; role/name/version stay strict.)
_status_has_ref() {
  printf '%s' "${MODEL_STATUS_BODY}" | python3 -c '
import json, sys
role, name, ver = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ms = doc.get("model-status", doc)   # tolerate enveloped or bare payloads
for s in (ms.get("streams") or []):
    ref = s.get(role) or {}
    if ref.get("name") == name and ref.get("version") == ver:
        sys.exit(0)
sys.exit(1)' "$1" "$2" "$3"
}

# assert_status_ref <role: model|shadow> <name> <version> <description>
#   PASS if the LAST model/status body has SOME streams[] entry whose <role> ref is
#   (name, version). Instance-agnostic replacement for pinning the whole
#   "model":{"gpu":0,"instance":0,...} substring with assert_status_contains.
assert_status_ref() {
  local role="$1" name="$2" version="$3" desc="$4"
  if _status_has_ref "${role}" "${name}" "${version}"; then
    _pass "${desc}"
  else
    _fail "${desc}  (model/status body lacks: a stream with ${role} ref '${name}@${version}')"
  fi
}

# assert_status_ref_absent <role: model|shadow> <name> <version> <description>
#   PASS if NO streams[] entry in the LAST model/status body carries that <role> ref.
#   Instance-agnostic counterpart of assert_status_not_contains for ref patterns.
assert_status_ref_absent() {
  local role="$1" name="$2" version="$3" desc="$4"
  if _status_has_ref "${role}" "${name}" "${version}"; then
    _fail "${desc}  (model/status body unexpectedly contains: a stream with ${role} ref '${name}@${version}')"
  else
    _pass "${desc}"
  fi
}

# assert_status_ref_eventually <query> <role> <name> <version> <description> [timeout-secs]
#   Re-query model/status until SOME streams[] entry carries the <role> ref, or timeout.
#   Instance-agnostic counterpart of assert_status_eventually (same async-reroute
#   rationale and retry loop; see assert_status_eventually above).
assert_status_ref_eventually() {
  local q="$1" role="$2" name="$3" version="$4" desc="$5" to="${6:-12}" i=0
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    get_model_status "${q}" >/dev/null 2>&1
    if _status_has_ref "${role}" "${name}" "${version}"; then _pass "${desc}  (settled after ${i}s)"; return 0; fi
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (model/status never had a stream with ${role} ref '${name}@${version}' within ${to}s)"
  return 1
}

# assert_status_ref_eventually_absent <query> <role> <name> <version> <description> [timeout-secs]
#   Re-query model/status until NO streams[] entry carries the <role> ref, or timeout.
#   Instance-agnostic counterpart of assert_status_eventually_absent.
assert_status_ref_eventually_absent() {
  local q="$1" role="$2" name="$3" version="$4" desc="$5" to="${6:-12}" i=0
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    get_model_status "${q}" >/dev/null 2>&1
    if ! _status_has_ref "${role}" "${name}" "${version}"; then _pass "${desc}  (settled after ${i}s)"; return 0; fi
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (model/status still had a stream with ${role} ref '${name}@${version}' after ${to}s)"
  return 1
}

# _status_has_version <name> <version>
#   Returns 0 iff the LAST model/status body has a models[].versions[] entry for
#   <name> with that version in state ready|serving. Parsed as JSON (order-
#   independent -- jsoncpp sorts object keys alphabetically, so "state" precedes
#   "version" in the body and a '"version":"N","state":...' substring can NEVER
#   match; never rely on serializer key order). Version-entry counterpart of
#   _status_has_ref.
_status_has_version() {
  printf '%s' "${MODEL_STATUS_BODY}" | python3 -c '
import json, sys
name, ver = sys.argv[1], sys.argv[2]
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ms = doc.get("model-status", doc)   # tolerate enveloped or bare payloads
for m in (ms.get("models") or []):
    if m.get("name") != name:
        continue
    for v in (m.get("versions") or []):
        if v.get("version") == ver and v.get("state") in ("ready", "serving"):
            sys.exit(0)
sys.exit(1)' "$1" "$2"
}

# assert_status_version_eventually <query> <name> <version> <description> [timeout-secs]
#   Re-query model/status until the models[].versions[] entry for (name, version)
#   exists in state ready|serving, or timeout. JSON-parsed counterpart of
#   assert_status_eventually for VERSION entries (same async rationale).
assert_status_version_eventually() {
  local q="$1" name="$2" version="$3" desc="$4" to="${5:-12}" i=0
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    get_model_status "${q}" >/dev/null 2>&1
    if _status_has_version "${name}" "${version}"; then _pass "${desc}  (settled after ${i}s)"; return 0; fi
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (model/status never had a ready/serving version entry '${name}@${version}' within ${to}s)"
  return 1
}

# assert_status_version_eventually_absent <query> <name> <version> <description> [timeout-secs]
#   Re-query model/status until NO ready/serving version entry for (name, version)
#   remains, or timeout. Absent counterpart of assert_status_version_eventually.
assert_status_version_eventually_absent() {
  local q="$1" name="$2" version="$3" desc="$4" to="${5:-12}" i=0
  if [ "${AUTO_APP}" != "1" ]; then echo "  [SKIP] ${desc} (AUTO_APP=0)"; return 0; fi
  while [ "${i}" -lt "${to}" ]; do
    get_model_status "${q}" >/dev/null 2>&1
    if ! _status_has_version "${name}" "${version}"; then _pass "${desc}  (settled after ${i}s)"; return 0; fi
    sleep 1; i=$((i+1))
  done
  _fail "${desc}  (model/status still had a ready/serving version entry '${name}@${version}' after ${to}s)"
  return 1
}

# _status_routing_revision: print the routing_revision from the LAST model/status body.
# Parsed as JSON (order-independent, envelope-tolerant) like the ref helpers; prints
# nothing on a parse failure so a numeric comparison at the call site fails loudly.
_status_routing_revision() {
  printf '%s' "${MODEL_STATUS_BODY}" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ms = doc.get("model-status", doc)   # tolerate enveloped or bare payloads
print(ms.get("routing_revision", ""))'
}

# _status_model_gies <name>: print the sorted, comma-joined gie_id set of EVERY live
# instance of <name> (all versions, all shards) from the LAST model/status body. An
# in-place model/update must keep this set IDENTICAL (gie kept, no new bin) even as
# the version identity moves -- compare snapshots taken before/after the update.
_status_model_gies() {
  printf '%s' "${MODEL_STATUS_BODY}" | python3 -c '
import json, sys
name = sys.argv[1]
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ms = doc.get("model-status", doc)   # tolerate enveloped or bare payloads
gies = set()
for m in (ms.get("models") or []):
    if m.get("name") != name:
        continue
    for v in (m.get("versions") or []):
        for inst in (v.get("instances") or []):
            if "gie_id" in inst:
                gies.add(inst["gie_id"])
print(",".join(str(g) for g in sorted(gies)))' "$1"
}

########################################################################################################################
############################################### Test Cases #############################################################
########################################################################################################################

# Test Dummy: placeholder for ad-hoc local experimentation.
# Intentionally empty.
test_dummy() {
 echo "-----------------------------------------------------------------------"
  echo "[test_1] starting test 1 (basic: add 4 streams one-by-one on default, then remove)"

  # 1a) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  # multimodel config updates
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4
  # BOTH defaults are established at RUNTIME via model/load + a model/routing default switch
  # (the load_model def_primary/def_shadow flags; v1 removed set_default_* from model/load) --
  # comment out the config defaults for both roles so the API is the sole source of the defaults.
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 1b) remove the engine file to test the engine file generation
  rm "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"

  # 3) load the default primary & default shadow models (warm to READY)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  echo "[test_1] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  sleep 20

  # 3) add the streams
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_1] added Cam-0 (Custom Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_1] added Cam-1 (Custom Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_1] added Cam-2 (Custom Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_1] added Cam-3 (Custom Primary Model)"

  # 4) remove the streams
  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}3" "${STREAM}"
  echo "[test_1] removed Cam-0, Cam-1, Cam-2, Cam-3"

  # 5) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 1: PGIE-disabled dynamic add/remove smoke check.
#
# Flow:
#   1. Disable [primary-gie] (no nvmodelmux/nvinfer path), keep 4-wide batch + 1x4 tiler.
#   2. Start the app and wait for REST readiness.
#   3. Add Cam-0..Cam-3 one-by-one.
#   4. Remove Cam-0..Cam-3, then add Cam-4 (post-remove add sanity).
#   5. Stop the app.
#
# What to validate:
#   - dynamic stream add/remove works with inference disabled;
#   - post-remove add still works with PGIE disabled;
#   - clean shutdown with no crash / stall (covered by default sanity checks).
test_1() {
  echo "-----------------------------------------------------------------------"
  echo "[test_1] starting test 1 (PGIE-disabled dynamic add/remove smoke check)"

  # 1) config for this case (edits the per-case config copies)
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set primary-gie          enable                0
  cfg_set source-list          max-batch-size        4
  cfg_set nvstreammux          max-size-buffers      4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4

  # 2) start the app with the current config copies
  start_app || return

  # 3) add the streams
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_1] added Cam-0"
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_1] added Cam-1"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_1] added Cam-2"
  sleep "${DELAY}"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
  echo "[test_1] added Cam-3"
  sleep "${DELAY}"

  # 3b) With inference disabled there is NO nvmodelmux downstream, so both
  #     model-plane reads must FAIL LOUDLY -- on the wire, not only in the body.
  #     This is the one configuration in the suite that reaches that path.
  get_model_status
  assert_http_status 500 "test_1: model/status fails on the WIRE when no model element is present (not 200)"
  assert_status_contains '"code":"MODEL_OPERATION_FAILED"' "test_1: ...carrying the contract's internal-failure code"
  get_model_routing
  assert_http_status 500 "test_1: model/routing likewise fails on the wire (siblings now agree)"
  # 4) remove the streams
  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}3" "${STREAM}"
  echo "[test_1] removed Cam-0, Cam-1, Cam-2, Cam-3"

  # 5) add Cam-4 again to prove post-remove add works
  add_stream "${PREFIX}4" "Cam-4" "${STREAM}"
  echo "[test_1] added Cam-4"
  sleep "${DELAY}"

  # 6) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 2: CUSTOM primary loaded via model/load, requested by streams WHILE it is still WARMING.
#
# Flow:
#   1. NO config default (both primary-model/shadow-model commented out) -- the API is the only
#      source of models.
#   2. model/load '${CAR_MODEL}@${CAR_MODEL_VERSION}' -> it warms in LIMBO (role Unknown). The
#      cached engine file is removed first so the load must REBUILD it (~30s warm window).
#   3. Add Cam-0, Cam-1, Cam-2 (and Cam-3) requesting that model via metadata -- the FIRST adds
#      land WHILE it is still warming.
#   4. Remove all streams.
#
# What happens (the behaviour under test -- the regression this guards):
#   A metadata-requested model that is loaded-but-still-WARMING must NOT be treated as "no model".
#   Each stream attaches PASSTHROUGH now (frames flow, no inference) tagged with its mapped model so
#   the routing table shows "${CAR_MODEL}@${CAR_MODEL_VERSION} (passthru)" (NOT a bare "passthrough"),
#   and is registered for a DEFERRED PROMOTE. When the engine WARMS, EVERY waiting stream -- the
#   ones added FIRST included, not just streams added after the warm -- is promoted from passthrough
#   to inference (limbo model promoted into the Primary pool), zero-drop.
#   The old bug: mmc_resolve_role discarded a WARMING limbo model to the (NULL) default, so these
#   streams took a plain passthrough lane with no pending tag and no promote registration -- the
#   routing table showed bare "(passthrough)" and only streams added AFTER the warm ever inferred;
#   Cam-0/1/2 stayed stuck on passthrough forever.
#
# What to validate (asserts localise the regression if it returns):
#   - while warming: streams are deferred to passthrough + registered to auto-promote;
#   - while warming: the routing table tags them "<model> (passthru)" (carries the model name);
#   - the custom engine warms and the limbo model is PROMOTED into the Primary pool;
#   - EVERY waiting stream (incl. Cam-0, added first) is promoted passthrough -> inference;
#   - the promoted streams actually infer end-to-end (provenance carries model@ver, role=Primary);
#   - after the warm, NO stream is left stuck in passthrough (routing clean);
#   - clean teardown + no crash / no GStreamer-CRITICAL (default sanity gates).
test_2() {
  echo "-----------------------------------------------------------------------"
  echo "[test_2] starting test 2 (custom primary requested while WARMING: passthru-with-tag, then promote ALL waiting streams)"


  # 1a) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  # multimodel config updates
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4
  # BOTH defaults are established at RUNTIME via model/load + a model/routing default switch
  # (the load_model def_primary/def_shadow flags; v1 removed set_default_* from model/load) --
  # comment out the config defaults for both roles so the API is the sole source of the defaults.
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 1b) remove the cached engine so the load must REBUILD it -- this is what guarantees a long
  #     WARMING window that overlaps the stream adds (the whole point of this case). -f so a
  #     re-run with the engine already gone does not error.
  rm -f "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"

  # 3) load the CUSTOM primary into limbo (warms in the background: step 1b deleted the engine,
  #    so this is a FULL TRT rebuild -- >= several seconds on every device class)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"

  # 3) add the streams -- Cam-0 IMMEDIATELY after the load is accepted: DETERMINISTIC deferral.
  #    NO sleep between load and first add. The old 'sleep 20' assumed a ~30s rebuild (x86) and
  #    false-failed on fast devices (Spark rebuilds in ~9s -> the sleep outlived the warm and
  #    every stream direct-attached, so the passthru-with-tag machinery under test never ran).
  #    Adding at load+~0s ALWAYS lands inside the rebuild window, on every device.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_2] added Cam-0 (Custom Primary Model '${CAR_MODEL}')"
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_2] added Cam-1 (Custom Primary Model '${CAR_MODEL}')"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_2] added Cam-2 (Custom Primary Model '${CAR_MODEL}')"
  sleep "${DELAY}"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_2] added Cam-3 (Custom Primary Model '${CAR_MODEL}')"

  # ---- ASSERTS: the warming-then-promote contract (catch the limbo-warm passthrough regression) ----
  # (a) WHILE WARMING: requested-but-warming model is deferred to passthrough + queued to auto-promote
  #     (the old bug discarded it to the default and never registered a promote). Cam-0 was added
  #     at load+~0s against a full TRT rebuild, so ITS deferral is guaranteed by construction --
  #     assert it per-stream, no timing assumption. (Cam-1..3 are DELAY-spaced and may legally
  #     defer OR direct-attach depending on device speed; both paths are covered below.)
  assert_log "stream 0: primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' still warming" "test_2: Cam-0 (added at load+0s) deferred to PASSTHROUGH while '${CAR_MODEL}' was still warming" 15
  # (b) WHILE WARMING: routing table TAGS the stream with the mapped model (NOT a bare passthrough).
  assert_log "${CAR_MODEL}@${CAR_MODEL_VERSION} (passthru)" "test_2: routing table shows '${CAR_MODEL}@${CAR_MODEL_VERSION} (passthru)' while warming (carries the model name, not bare passthrough)" 15
  # (c) the custom engine warms and the limbo model is promoted INTO the Primary pool.
  assert_log "engine warmed" "test_2: custom model '${CAR_MODEL}' engine reached WARMED/READY" 60
  assert_log "PROMOTED '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_2: '${CAR_MODEL}@${CAR_MODEL_VERSION}' promoted from limbo -> Primary once warmed" 60
  # (d) THE REGRESSION GUARD: EVERY waiting stream is promoted, INCLUDING Cam-0 (added FIRST).
  #     Old bug: only streams added AFTER the warm inferred; Cam-0/1/2 stayed on passthrough.
  assert_log "stream 0 PROMOTED passthrough -> model '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_2: Cam-0 (added FIRST, before the warm) auto-promoted passthrough -> '${CAR_MODEL}' (the exact regression this guards)" 60
  # Count the WIRED marker, NOT 'PROMOTED passthrough': a stream reaches inference on the model by
  # EITHER path -- deferred-promote (added while warming: emits PROMOTED passthrough + WIRED) or
  # direct-attach (added after the engine warmed: emits WIRED only). How many take each path is
  # pure timing (a cached/fast engine warm shrinks the passthrough window), so counting only the
  # promote line false-fails when some adds land post-warm even though every stream infers on the
  # model. WIRED fires exactly once per stream on Car regardless of path -> robust "promote OR
  # direct" count. Cam-0's promote (the actual regression) is still asserted explicitly above.
  assert_count_min "WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}" 3 "test_2: all streams requesting '${CAR_MODEL}@${CAR_MODEL_VERSION}' reached inference (promote OR direct-attach) (>=3), not just later ones" 60
  # (e) promoted streams actually INFER end-to-end (provenance at the combined-mux src).
  #     The generic model-level assertion can be satisfied by one healthy sibling stream; the
  #     per-camera checks below catch the intermittent dynamic-lane wedge where e.g. Cam-0
  #     entered/buffered forever while Cam-3 on the same model inferred normally.
  assert_log "model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" "test_2: promoted streams inferring end-to-end on '${CAR_MODEL}' (provenance carries model@ver, role=Primary)" 30
  assert_count_min "name=Cam-0  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" 2 \
    "test_2: Cam-0 (first waiting stream) infers sustained after limbo promote -- no per-lane stall" 45
  assert_count_min "name=Cam-1  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" 2 \
    "test_2: Cam-1 infers sustained after limbo promote -- no per-lane stall" 45
  assert_count_min "name=Cam-2  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" 2 \
    "test_2: Cam-2 infers sustained after limbo promote -- no per-lane stall" 45
  assert_count_min "name=Cam-3  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" 2 \
    "test_2: Cam-3 infers sustained after limbo promote -- no per-lane stall" 45
  # (f) after the warm, NO stream is left stuck in passthrough. Re-scope the log AFTER promotion so
  # this only inspects routing dumps that follow the switchover (the warming-phase tags are
  # expected and must not trip this).
  log_mark
  assert_log_absent "(passthru)" "test_2: no stream left stuck in passthrough after the warm (all routed to '${CAR_MODEL}')" 8
  sleep "${DELAY}"

  # 4) remove the streams
  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}3" "${STREAM}"
  echo "[test_2] removed Cam-0, Cam-1, Cam-2, Cam-3"

  # 5) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 3: drain-to-zero REFILL -- single stream, repeated.  (recovery path)
#
# Flow:
#   1. Add Cam-0 (default primary); let it serve, then remove it -> bin DRAINS TO ZERO.
#   2. Add Cam-1 (REFILL after drain); let it serve, then remove it (drain again).
#   3. Repeat once more with Cam-2 -- a second refill cycle proves repeatability.
#
# What happens (the behaviour under test):
#   When the last stream leaves, the bin keeps its persistent elements warm across the gap
#   (per-model nvstreammux/nvinfer/nvstreamdemux + the combined display mux). The NEXT add
#   is a REFILL: those drained elements must RESUME pumping frames. The legacy-nvstreammux
#   hazard is that after a full drain it sits in a stale state and the refilled stream WIRES
#   but never FLOWS ("data flow before segment" spin). The plugin recovers by resetting
#   state on the DRAIN side (each idled model bin + out_mux cycle PLAYING->READY->PLAYING)
#   and, on a GENUINE refill only (ever_attached), issuing a one-shot FLUSH down the new
#   branch. The flush is NOT sent on a first-ever attach -- there it would only race
#   nvstreamdemux's sticky-event delivery and crash (see modelmux_multimodel_bin_attach_stream).
#
# What to validate (asserts localise WHERE a refill breaks, if it does):
#   - the refill recovery path is actually taken (refill after a drain-to-zero);
#   - the refilled stream WIRES (attach succeeded);
#   - the refilled stream actually FLOWS end-to-end -- its PROVENANCE appears at the
#     combined-mux src. WIRED-but-no-PROVENANCE pinpoints the classic wire-then-stall;
#   - recovery is REPEATABLE (a second drain->refill cycle also flows);
#   - clean drain each cycle + no crash / no GStreamer-CRITICAL (default sanity gates).
test_3() {
  echo "-----------------------------------------------------------------------"
  echo "[test_3] starting test 3 (drain-to-zero refill -- STRESS: add/drain/refill xN)"
  #
  # WHAT THIS STRESSES
  #   Drain-to-zero is the path that historically (a) crashed -- nvinfer's WARM output loop
  #   racing the inner mux/demux rebuild -> 'gst_pad_push: GST_IS_PAD' + nvinfer 'Internal data
  #   stream error' -- and (b) wedged the NEXT refill (the next stream wired but never flowed).
  #   A single add/remove pair samples that race only once. This case hammers it: prime ONE
  #   stream, then repeat  add -> wait-for-flow -> remove -> drain  for N cycles (default 100,
  #   override with RACE_REPS). Every cycle empties the bin (triggering the recreate) and
  #   refills it, so any reintroduced teardown/refill race shows up as either:
  #     - a missing per-cycle flow marker (counted + reported here), or
  #     - a GStreamer-CRITICAL / assertion / frame-loss / frame-gap caught by the end-of-case
  #       sanity gates (run_one_auto), which scan the WHOLE N-cycle log -> a crash in ANY cycle
  #       is caught.
  #   Exactly ONE stream is active at a time (remove precedes the next add), so the freed
  #   source slot is reused and we never exceed max-batch-size.

  # 1) config: 2-slot bin, non-unified A/B, single stream at a time.
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  local reps="${RACE_REPS}"        # drain-to-zero refill cycles (global RACE_REPS)
  local flow_to=15                  # seconds to wait for a cycle's first provenance frame
  local fails=0 first_fail="" i j ok

  # 3) PRIME (Cam-0, first-ever attach): must FLOW SUSTAINED before the first drain. A single
  #    provenance line would green-light a STARVED first stream (the demux first-pad race:
  #    'segment'/'caps' before 'stream-start' on src_0), so require >=2 -- proof of real,
  #    sustained flow, not a one-frame blip. Scoped to this stream via log_mark.
  log_mark
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_3] added Cam-0 (prime, first-ever attach)"
  sleep "${DELAY}"
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 \
    "test_3: Cam-0 prime (FIRST stream) FLOWS sustained before the first drain -- not starved" \
    "${flow_to}"
  remove_stream "${PREFIX}0" "${STREAM}"
  echo "[test_3] removed Cam-0 (drained to zero)"
  sleep "${DELAY}"

  # 4) STRESS LOOP: repeat drain-to-zero refill for ${reps} cycles. Each cycle the bin drains
  #    to zero (inner mux/demux recreate) and a FRESH stream must WIRE and FLOW end-to-end
  #    (its provenance 'name=Cam-${i}' must reach the combined-mux log). Each cycle is checked
  #    SILENTLY and scoped via log_mark, then AGGREGATED into one pass/fail below -- so the
  #    results ledger gets one line, not ${reps}; per-cycle detail stays in the app log.
  #    (Skipped in MANUAL mode (AUTO_APP=0), like the assert_* helpers.)
  if [ "${AUTO_APP}" = "1" ]; then
    echo "[test_3] STRESS: ${reps} drain-to-zero refill cycles ..."
    for (( i=1; i<=reps; i++ )); do
      log_mark                                     # scope the flow check to THIS refill only
      add_stream "${PREFIX}${i}" "Cam-${i}" "${STREAM}"
      sleep 1                                      # brief settle; the poll below does the wait

      # poll up to ${flow_to}s for this cycle's provenance: refill WIRED and a frame FLOWED.
      ok=0
      for (( j=0; j<flow_to; j++ )); do
        if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null \
             | grep -qF -- "name=Cam-${i}  model=${DEFAULT_MODEL}"; then ok=1; break; fi
        sleep 1
      done
      if [ "${ok}" -ne 1 ]; then
        fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${i}"
        echo "[test_3] WARN cycle ${i}: Cam-${i} refill did NOT flow within ${flow_to}s"
      fi

      remove_stream "${PREFIX}${i}" "${STREAM}"
      sleep 2                                      # let the bin drain to zero before next add
    done

    # 5) one aggregate verdict for the whole stress loop.
    if [ "${fails}" -eq 0 ]; then
      _pass "test_3: all ${reps} drain-to-zero refill cycles WIRED and FLOWED (stress, repeatable recovery)"
    else
      _fail "test_3: ${fails}/${reps} drain-to-zero refill cycles failed to flow (first: Cam-${first_fail}) -- drain-to-zero refill regression"
    fi
  else
    echo "  [SKIP] test_3: stress refill loop (AUTO_APP=0)"
  fi

  # 6) stop the app. The default sanity gates (no GStreamer-CRITICAL/assert/crash, ZERO frame
  #    loss, no frame gaps) run AFTER this in run_one_auto and scan the full ${reps}-cycle log,
  #    so any crash/loss in ANY cycle is caught there -- this is the primary stress signal.
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 4: drain-to-zero REFILL -- FULL batch.  (recovery path under load)
#
# Flow:
#   1. Add Cam-0..Cam-3 (fill the whole max_streams=4 batch); all serve.
#   2. Remove ALL four -> bin DRAINS TO ZERO from a saturated state.
#   3. Add Cam-4 (REFILL after the full drain); let it serve, then remove it.
#
# What happens (the behaviour under test):
#   Same recovery path as test_3, but stressed: the drain releases ALL batch slots at once
#   (not a 1-stream churn), then a single stream refills. NOTE source_id is RECYCLED by
#   nvmultiurisrcbin -- after Cam-0 is removed its id 0 is reused, so Cam-4 comes back as
#   stream 0 (see the camera_name, not the id, to track it). The refilled stream must wire
#   AND resume flowing; the recovery relies on the drain-side resets + the gated one-shot
#   refill flush (only on a genuine refill, never on a first-ever attach).
#
# What to validate (asserts localise WHERE a refill breaks, if it does):
#   - all 4 prime streams wire (saturated baseline before the drain);
#   - the refill recovery path is taken (refill after the full drain-to-zero);
#   - the refilled Cam-4 WIRES and actually FLOWS end-to-end (PROVENANCE by camera_name);
#     WIRED-but-no-flow pinpoints a wire-then-stall after a saturated drain;
#   - clean teardown + no crash / no GStreamer-CRITICAL (default sanity gates).
test_4() {
  echo "-----------------------------------------------------------------------"
  echo "[test_4] starting test 4 (SATURATED drain-to-zero refill -- STRESS: 4-add/4-remove xN)"
  #
  # WHAT THIS STRESSES
  #   The HARDEST drain-to-zero variant: a FULL batch (4 streams) is added to saturate the bin,
  #   then ALL four are removed at once so the bin drains to zero from a saturated state (every
  #   batch slot released together -> the inner mux/demux recreate + combined-mux rebuild), then
  #   it is refilled with a fresh saturated batch. test_3 cycles ONE stream; this cycles FOUR,
  #   exercising the multi-slot teardown/refill race. Repeated ${RACE_REPS} times (default 10).
  #   A reintroduced teardown/refill race shows up as either a missing per-stream flow marker
  #   (counted + reported here) or a GStreamer-CRITICAL/assert/frame-loss/gap caught by the
  #   end-of-case sanity gates (which scan the WHOLE run -> any cycle's crash is caught).
  #   All streams of a batch are removed before the next batch is added, so source slots recycle
  #   and we never exceed max-batch-size; camera ids/names are unique across the whole run
  #   (Cam-0..Cam-${N}) so each cycle's provenance is unambiguous (further scoped by log_mark).

  # 1) config: 4-slot bin, non-unified A/B.
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4

  # 2) start the app with the current config copies
  start_app || return

  local reps="${RACE_REPS}"        # saturate->drain->refill cycles (global RACE_REPS)
  local flow_to=15                  # seconds to wait for a stream's first provenance frame
  local fails=0 first_fail="" k s j ok cam batch_first
  local cam_idx=0                   # running camera index -> unique id/name across the run

  echo "[test_4] STRESS: ${reps} saturate(4)->drain-to-zero->refill cycles ..."
  for (( k=1; k<=reps; k++ )); do
    log_mark                                       # scope checks to THIS saturate/drain cycle
    batch_first="${cam_idx}"

    # --- add 4 streams: saturate the batch ---
    for (( s=0; s<4; s++ )); do
      add_stream "${PREFIX}${cam_idx}" "Cam-${cam_idx}" "${STREAM}"
      cam_idx=$((cam_idx+1))
    done
    echo "[test_4] cycle ${k}: added Cam-${batch_first}..Cam-$((cam_idx-1)) (saturated)"
    sleep "${DELAY}"

    # BASELINE (first cycle only): full detail -- all 4 WIRE; the first stream FLOWS sustained
    # (not starved by the input-demux first-pad sticky-event race that blanks src_0 while the
    # others flow); its branch RECEIVES sustained input. (assert_* self-skip in manual mode.)
    if [ "${k}" -eq 1 ]; then
      assert_log "stream 3 WIRED" \
        "test_4: all 4 prime streams wired (saturated baseline)" 10
      assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 \
        "test_4: Cam-0 (FIRST stream) FLOWS sustained on the saturated baseline -- not starved" "${flow_to}"
      assert_log "TAP[branch-in] stream 0: 100 buffer" \
        "test_4: Cam-0 (FIRST stream) branch RECEIVED sustained input (>=100 buffers)" "${flow_to}"
    fi

    # PER-CYCLE (silent, aggregated): a refill cycle (k>=2) must take the refill path, and
    # EVERY one of the 4 streams in this batch must FLOW end-to-end (provenance reaches the
    # combined mux). Failures are counted into one verdict so the ledger stays one line.
    if [ "${AUTO_APP}" = "1" ]; then
      if [ "${k}" -ge 2 ]; then
        ok=0
        for (( j=0; j<flow_to; j++ )); do
          tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null \
            | grep -qF -- "refill (first stream after a drain-to-zero)" && { ok=1; break; }
          sleep 1
        done
        if [ "${ok}" -ne 1 ]; then
          fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="cycle ${k} (no refill marker)"
          echo "[test_4] WARN cycle ${k}: refill path not taken after the full drain"
        fi
      fi
      for (( s=0; s<4; s++ )); do
        cam=$((batch_first+s)); ok=0
        for (( j=0; j<flow_to; j++ )); do
          tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null \
            | grep -qF -- "name=Cam-${cam}  model=${DEFAULT_MODEL}" && { ok=1; break; }
          sleep 1
        done
        if [ "${ok}" -ne 1 ]; then
          fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="Cam-${cam}"
          echo "[test_4] WARN cycle ${k}: Cam-${cam} did not flow within ${flow_to}s"
        fi
      done
    fi

    # --- remove ALL four: drain to zero from a saturated state ---
    for (( s=0; s<4; s++ )); do
      remove_stream "${PREFIX}$((batch_first+s))" "${STREAM}"
    done
    echo "[test_4] cycle ${k}: removed 4 streams (drained to zero)"
    sleep "${DELAY}"
  done

  # one aggregate verdict for the whole stress loop.
  if [ "${AUTO_APP}" = "1" ]; then
    if [ "${fails}" -eq 0 ]; then
      _pass "test_4: all ${reps} saturated drain-to-zero refill cycles WIRED and FLOWED (stress)"
    else
      _fail "test_4: ${fails} stream/cycle flow failures over ${reps} saturated drain cycles (first: ${first_fail}) -- saturated drain-to-zero refill regression"
    fi
  else
    echo "  [SKIP] test_4: saturated stress loop (AUTO_APP=0)"
  fi

  # stop the app -- the default sanity gates (no crash / ZERO frame loss / no gaps) run after
  # this in run_one_auto and scan the full ${reps}-cycle log (primary stress signal).
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 5: per-model capacity -> automatic ModelBin sharding (grow) AND auto-compaction (shrink).
#
# Flow:
#   1. Start with max-batch-size=4 but per-model-batch-size=2 (one bin batches <=2 streams).
#   2. Add Cam-0..Cam-3 on the default primary -> shard#0=[0,1], shard#1=[2,3] (GREW shard #1).
#   3. Remove Cam-0 (off shard#0) AND Cam-2 (off shard#1) -> 2 streams left, fragmented
#      one-per-shard (shard#0=[1], shard#1=[3]). Pause -> expect AUTO-COMPACTION.
#   4. Remove the remaining two -> clean drain to zero.
#
# What happens (the behaviour under test):
#   GROW: the 3rd stream exceeds the per-bin cap, so the app grows shard #1 of the SAME model
#   (same name, same gie-id, same engine) -- no stall, no half-attach.
#   COMPACT: once two removes leave the 2 survivors spanning two half-full shards, the debounced
#   compaction pass migrates the tail shard's stream into shard #0's free slot (make-before-break:
#   same model, same gie, same display slot -> zero-drop, invisible downstream) and frees the now
#   redundant shard. Symmetric elastic sharding: grow on overflow, compact on underflow.
#
# What to validate:
#   - log shows "GREW shard #1" when the 3rd stream is added.
#   - after removing one stream per shard: "HITLESS shard-compact" + "compact: empty shard" (reclaimed).
#   - after the partial remove, the surviving output frames restore downstream
#     NvDsFrameMeta.source_id/pad_index to Cam-1 and Cam-3, not display-slot ids.
#   - shards share one gie-id (downstream identity stable); zero frame loss / no frame gaps
#     across the live migration (the default sanity gates); all detach cleanly.
test_5() {
  echo "-----------------------------------------------------------------------"
  echo "[test_5] starting test 5 (Modelbin sharding test)"

  # 1) Configure this case in the config copies before app startup.
  #    Keep the source batch at 4 while capping each model instance to 2 streams.
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  # Single-row layout makes it easier to visually confirm all 4 streams are active.
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  # multimodel config updates
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2

  # 2) Start app using this case-specific config.
  start_app || return

  # 3) Add 4 streams sequentially.
  #    With per-model-batch-size=2, streams 3/4 should not break routing.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_5] added Cam-0 (Default Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_5] added Cam-1 (Default Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_5] added Cam-2 (Default Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
  echo "[test_5] added Cam-3 (Default Primary Model)"
  assert_log "GREW shard #1" "test_5: per-model cap exceeded -> auto-sharded (overflow shard grown)" 8
  assert_log "stream 3 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}" "test_5: 4th stream still wires despite the 2-stream-per-bin cap (no half-attach)" 8
  sleep 10

  # 4) Remove ONE stream from EACH shard -> 2 survivors span 2 half-full shards
  #    (shard#0=[1], shard#1=[3]). This is the fragmentation that triggers compaction.
  log_mark
  remove_stream "${PREFIX}0" "${STREAM}"          # off shard #0
  remove_stream "${PREFIX}2" "${STREAM}"          # off shard #1
  echo "[test_5] removed Cam-0 & Cam-2 -- 2 streams now span 2 half-full shards"
  sleep "${DELAY}"
  assert_log "SOURCE-ID restore stream=1 name=Cam-1 role=Primary frame=(source_id=1,pad_index=1)" \
    "test_5: survivor Cam-1 leaves nvmodelmux with downstream source_id/pad_index restored after partial remove" 15
  assert_log "SOURCE-ID restore stream=3 name=Cam-3 role=Primary frame=(source_id=3,pad_index=3)" \
    "test_5: survivor Cam-3 leaves nvmodelmux with downstream source_id/pad_index restored after partial remove" 15

  # 5) AUTO-COMPACTION: the 2 survivors fit ONE shard, so the debounced pass migrates the
  #    tail shard's stream into shard #0 (zero-drop) and frees the redundant shard. Wait
  #    past the ~1s debounce + the make-before-break migration.
  assert_log "HITLESS shard-compact" "test_5: under-full shards auto-compacted (hitless migrate, zero-drop)" 10
  assert_log "compact: empty shard" "test_5: redundant shard reclaimed after compaction" 10
  sleep 20

  # 6) Stop app so the next test starts from a clean process state.
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 6: metadata-driven PRIMARY routing alongside default traffic.
#
# Flow:
#   1. Add Cam-0 on the Default Primary Model.
#   2. model/load a custom primary (Car) -- warmed, no stream yet.
#   3. Add Cam-1 and Cam-2 selecting Car via stream/add metadata.
#   4. Add Cam-3 back on the default primary.
#
# What happens (the behaviour under test):
#   A newly loaded model can be selected per-stream at add time via metadata, while other
#   streams stay on the default -- both routing paths coexist with no drain-to-zero on the
#   default bin.
#
# What to validate:
#   - log shows Car "LOADED into limbo" after model/load (and "PROMOTED" into Primary on first use).
#   - Cam-1/Cam-2 route to Car; Cam-0/Cam-3 stay on the default (per-stream provenance).
#   - all 4 run together, then detach cleanly.
test_6() {
  echo "-----------------------------------------------------------------------"
  echo "[test_6] starting test 6 (add stream on Default Primary Model; load primary; add streams using primary)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  # multimodel config updates
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  # 3) load a custom primary model and BLOCK until it is actually serveable. This
  #    replaces a fixed sleep + log-marker assert: wait_for_model polls model/status
  #    until ready, so the stream we add next cannot race the warm-up and silently
  #    fall back to the default primary.
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_6] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"

  # 4) add Cam-0 on the DEFAULT primary (it was added before the custom model finished loading)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_6] added Cam-0 (Default Primary Model)"
  sleep "${DELAY}"

  # 5) add a stream that selects the loaded custom primary via metadata
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_6] added Cam-1 (Custom Primary Model '${CAR_MODEL}')"
  sleep "${DELAY}"

  # 6) add a stream that selects the loaded custom primary via metadata
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_6] added Cam-2 (Custom Primary Model '${CAR_MODEL}')"
  sleep "${DELAY}"

  # 7) add a stream that selects the default primary model
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
  echo "[test_6] added Cam-3 (Default Primary Model)"
  # validate EVERY stream's routing matches the intent (Cam-1 & Cam-2 on the custom
  # primary, Cam-0 & Cam-3 on the default -- both paths coexist), plus the type-less
  # limbo model was PROMOTED into the Primary role on first use.
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}" "test_6: Cam-0 stayed on the default primary (added before '${CAR_MODEL}' was loaded)" 10
  assert_log "PROMOTED '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_6: '${CAR_MODEL}' promoted from limbo -> Primary on first use (type-less load)" 10
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}" "test_6: Cam-1 routed to the loaded custom primary '${CAR_MODEL}'" 10
  assert_log "stream 2 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}" "test_6: Cam-2 also routed to the custom primary '${CAR_MODEL}'" 10
  assert_log "stream 3 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}" "test_6: Cam-3 stayed on the default primary (both paths coexist)" 10
  sleep "${DELAY}"

  # 8) remove the stream
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}3" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_6] removed Cam-1, Cam-0, Cam-3, Cam-2"
  
  # 9) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 7: model load/unload lifecycle + serving guard.
#
# Flow:
#   1. Add Cam-0 (default), model/load Car, add Cam-1 selecting Car.
#   2. model/unload Car WHILE Cam-1 is using it  (expect: REJECTED).
#   3. Remove Cam-1, then model/unload Car again  (expect: SUCCEEDS).
#
# What happens (the behaviour under test):
#   A model that is actively serving a stream cannot be unloaded -- the unload is REJECTED so
#   live inference is never pulled out from under a stream. Once the last stream leaves and
#   the bin is idle, unload SUCCEEDS and frees the engine.
#
# What to validate:
#   - while serving: log shows "cannot unload 'Car@1' -- shard #0 status=Serving".
#   - after Cam-1 removed: log shows "unload idle model 'Car@1'" (success).
test_7() {
  echo "-----------------------------------------------------------------------"
  echo "[test_7] starting test 7 (model load/unload lifecycle)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the model and let its engine warm to READY
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_7] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"

  # 4) add Cam-0 on the DEFAULT primary (baseline traffic alongside the custom model)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_7] added Cam-0 (Default Primary Model)"
  sleep "${DELAY}"

  # 5) add a stream that uses it (now READY -> routes to it, no fallback)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_7] added Cam-1 (Custom Primary Model '${CAR_MODEL}')"
  # precondition for the serving guard: Cam-1 must ACTUALLY be running on Car@1
  # (not silently fall back to default) -- otherwise the unload-reject below is vacuous.
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}" "test_7: Cam-1 actually routed to '${CAR_MODEL}' (serving-guard precondition)" 10
  sleep "${DELAY}"

  # 6) unload WHILE serving -> must be rejected. The element now decides admission
  #    SYNCHRONOUSLY (control-admission query), so the rejection is returned in the
  #    REST response itself (MODEL_IN_USE) -- not just logged.
  log_mark
  unload_model "${CAR_MODEL}"
  echo "[test_7] Requested unloading of (Custom Primary Model '${CAR_MODEL}') while serving (expected: rejected)"
  assert_rest_body_contains "MODEL_IN_USE" "test_7: unload-while-serving REJECTED in the REST response (MODEL_IN_USE)"
  assert_http_status 409 "test_7: ...as a 409 conflict, not a 400"
  assert_log "model unload refused: '${CAR_MODEL}':" "test_7: unloading (Custom Primary Model '${CAR_MODEL}') while serving is REJECTED (log)"
  sleep "${DELAY}"

  # 7) remove the stream, then unload -> now READY/idle -> succeeds
  remove_stream "${PREFIX}1" "${STREAM}"
  echo "[test_7] removed Cam-1"
  sleep "${DELAY}"

  # 8) unload the model -> now READY/idle -> succeeds
  log_mark
  unload_model "${CAR_MODEL}"
  echo "[test_7] Requested unloading of (Custom Primary Model '${CAR_MODEL}') (expected: success)"
  assert_rest_body_contains "unloaded" "test_7: idle unload ACCEPTED in the REST response (202)"
  assert_log "unload idle model '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_7: unloading (Custom Primary Model '${CAR_MODEL}') SUCCEEDS"
  sleep "${DELAY}"

  # 8b) verify the engine was actually FREED: the model is gone from the live pool
  #     (model/status no longer lists it). Confirms the unload's stated effect, not
  #     just that we logged it.
  get_model_status "model=${CAR_MODEL}"
  assert_status_not_contains "\"name\":\"${CAR_MODEL}\"" "test_7: '${CAR_MODEL}' removed from the pool after unload (engine freed)"

  # 9) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 8: live scoped PRIMARY reroute (zero-drop) on running streams.
#
# Flow:
#   1. Add Cam-0 (default), model/load Car, add Cam-1 on Car, add Cam-2 (default).
#   2. model/routing model=Car@1 scoped to Cam-0 AND Cam-2 only.
#
# What happens (the behaviour under test):
#   Only the two named streams switch their primary to Car, live, via make-before-break
#   (the new bin is warm before the switch) so no frame is dropped. Cam-1 is untouched, and
#   no shadow is involved.
#
# What to validate:
#   - log shows "model/routing applied to 2 stream(s)" (exactly the two named streams).
#   - Cam-0 and Cam-2 now report primary=Car; Cam-1 unchanged.
#   - per-stream frame table: dropped = 0 across the reroute.
test_8() {
  echo "-----------------------------------------------------------------------"
  echo "[test_8] starting test 8 (Primary Model update on a live stream)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        3
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               3
  # multimodel config updates
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the model and let its engine warm to READY (the default primary comes from the
  #    config primary-model=DEFAULT;1 -- preloaded at startup, no API default needed here).
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_8] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"

  # 4) add Cam-0 on the DEFAULT primary (later rerouted to the custom model by the scoped update)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_8] added Cam-0 (Default Primary Model)"
  sleep "${DELAY}"

  # 5) add a stream that uses it (now READY -> routes to it, no fallback)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_8] added Cam-1 (Custom Primary Model '${CAR_MODEL}')"
  # precondition: Cam-1 starts on Car (it is OUTSIDE the reroute scope below, so this
  # is what proves the scoped update leaves it untouched rather than never-touched-it).
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}" "test_8: Cam-1 starts on '${CAR_MODEL}' (outside the reroute scope)" 10
  sleep "${DELAY}"

  # 6) add a stream on the Default Primary Model
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_8] added Cam-2 (Default Primary Model)"
  # precondition: Cam-0 & Cam-2 start on the DEFAULT primary (so the reroute is a real switch).
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}" "test_8: Cam-0 starts on the default primary (pre-reroute)" 10
  assert_log "stream 2 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}" "test_8: Cam-2 starts on the default primary (pre-reroute)" 10
  # GUARD (regression): the FIRST stream must actually FLOW, not merely WIRE. An intermittent
  # nvstreamdemux sticky-event misordering ('segment' before 'stream-start') on the first
  # requested src pad (src_0) can leave Cam-0 wired-but-starved -> 0 fps / blank tile while the
  # later streams flow fine. Sustained provenance (>=2, logged repeatedly as frames stream)
  # proves Cam-0 really flows; a 0/1 count means it was starved. Checked BEFORE the reroute,
  # while Cam-0 is still on the default primary.
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 \
             "test_8: Cam-0 (first stream) FLOWS sustained on default -- not starved by demux src_0 race" 15
  sleep "${DELAY}"

  # 7) re-route the stream to the Custom Primary Model
  log_mark
  model_routing "${PREFIX}0,${PREFIX}2" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_8] re-routed Cam-0 & Cam-2 to (Custom Primary Model '${CAR_MODEL}')"
  assert_log "model/routing applied to 2 stream" "test_8: scoped reroute hit exactly the 2 named streams (Cam-1 untouched)" 10
  sleep "${DELAY}"          # let the scoped reroute settle before querying live state

  # verify the ACTUAL post-reroute routing via model/status (the API truth -- the
  # scoped-reroute path does not emit a per-stream 'route stream N' log line). The two
  # named streams now report primary=Car; Cam-1 (out of scope) is unchanged.
  get_model_status "stream=Cam-0"
  assert_status_ref model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_8: Cam-0 primary actually switched to '${CAR_MODEL}' (model/status)"
  get_model_status "stream=Cam-2"
  assert_status_ref model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_8: Cam-2 primary actually switched to '${CAR_MODEL}' (model/status)"
  get_model_status "stream=Cam-1"
  assert_status_ref model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_8: Cam-1 still on '${CAR_MODEL}' (untouched by the scoped update)"
  sleep 20

  # 8) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 9: default primary + default SHADOW baseline, plus custom-primary and shadow=none overrides.
#
# The configured defaults are a cross-version pair of the SAME model: default primary
# Trafficcamnet@1 + default shadow Trafficcamnet@2. Every stream gets the default shadow UNLESS it
# opts out with shadow="none"; the primary can be overridden per stream via metadata. The four
# streams exercise all four primary x shadow combinations (the custom primary is Car):
#   Cam-0 = default primary + default shadow
#   Cam-1 = default primary + shadow=none      (opts out of the default shadow)
#   Cam-2 = custom primary (Car) + default shadow
#   Cam-3 = custom primary (Car) + shadow=none
#
# Flow:
#   1. config: default primary = Trafficcamnet@1, default shadow = Trafficcamnet@2.
#   2. model/load the custom primary Car.
#   3. Add Cam-0..Cam-3 with the four combinations above.
#
# What to validate:
#   - log shows the custom primary Car "LOADED into limbo".
#   - the four WIRED lines match the combos: Cam-0 def+def, Cam-1 def+none, Cam-2 Car+def, Cam-3 Car+none.
#   - Cam-0 (first stream) FLOWS sustained (demux src_0 starvation guard).
#   - tiler: primaries on the top row, shadows on the bottom row; all detach cleanly.
test_9() {
  echo "-----------------------------------------------------------------------"
  echo "[test_9] starting test 9 (default primary + default shadow baseline, then custom primary via metadata)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list                                               max-batch-size        4
  cfg_set tiled-display                                             rows                  2
  cfg_set tiled-display                                             columns               4
  # multimodel config updates
  cfg_set multimodel                                                unified-batch         1
  cfg_set multimodel                                                per-model-batch-size  2
  cfg_set multimodel                                                primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set multimodel                                                shadow-model          "${DEFAULT_MODEL};${DEFAULT_SHADOW_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}           engine                "${WORST_ENGINE}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_SHADOW_MODEL_VERSION}    engine                "${BEST_ENGINE}"

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the custom primary model and default shadow model (warm to READY)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_9] Requested loading of (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_9: custom primary '${CAR_MODEL}' loaded and warmed" 12

  # 4) add a stream on the default primary model and default shadow model
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_9] Added Cam-0 (Default Primary Model & Default Shadow Model)"
  sleep "${DELAY}"

  # 5) add a stream that selects the default primary & without default shadow via metadata
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "" "" "none"
  echo "[test_9] Added Cam-1 (Default Primary Model & without Default Shadow Model)"

  # 6) add a stream that selects the loaded custom primary & default shadow via metadata
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_9] Added Cam-2 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  sleep "${DELAY}"

  # 7) add a stream that selects the custom primary model & without default shadow via metadata
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "none"
  echo "[test_9] Added Cam-3 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & without Default Shadow Model)"
  sleep "${DELAY}"

  # 8) Assert all four role combinations
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_9: Cam-0 baseline = default primary + default shadow" 10
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=-)" "test_9: Cam-1 = default primary + without default shadow" 10
  assert_log "stream 2 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_9: Cam-2 = custom primary + default shadow" 10
  assert_log "stream 3 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=-)" "test_9: Cam-3 = custom primary + without default shadow" 10
  # GUARD (regression): the FIRST stream must actually FLOW, not merely WIRE -- catches the
  # intermittent nvstreamdemux sticky-event misordering on the first requested src pad (src_0)
  # that leaves Cam-0 wired-but-starved (0 fps / blank tile) while Cam-1..3 flow. Sustained
  # primary provenance (>=2) proves Cam-0 flows; 0/1 means starved.
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 "test_9: Cam-0 (first stream) FLOWS sustained -- not starved by demux src_0 race" 15
  sleep "${DELAY}"

  # 9) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 10: default primary & default shadow + custom primary & custom shadow, mixed per stream.
#
# Defaults are established at RUNTIME via the v1 API (no config primary-model/shadow-model):
#   - model/load <DEFAULT;v1> + model/routing {"default":{"model":...}}  -> runtime default model
#     (the load_model def_primary flag drives both calls; v1 model/load never sets defaults).
#   - model/load <DEFAULT;v2> + model/routing {"default":{"shadow":...}} -> runtime default shadow.
# The two CUSTOM models are loaded WITHOUT flags, so they warm into limbo (role decided on first
# use). This exercises the plain load and the load+route-default flows side by side.
#
# Flow:
#   1. model/load the default primary (def_primary flag) and default shadow (def_shadow flag).
#   2. model/load the custom primary & custom shadow (no flags -> limbo).
#   3. Add Cam-0 on the (runtime) default primary & default shadow.
#   4. Add Cam-1 = custom primary & custom shadow (via stream/add metadata).
#   5. Add Cam-2 = custom primary & default shadow.
#   6. Add Cam-3 = default primary & custom shadow.
#
# What happens (the behaviour under test):
#   Primary and shadow are chosen independently per stream from the loaded models, so the
#   same models serve different role combinations across streams at the same time. A stream that
#   selects no model resolves to the runtime defaults set via the load flags.
#
# What to validate:
#   - log shows the route-default markers "default model ->" / "default shadow ->", and
#     the custom primary & custom shadow both "LOADED into limbo".
#   - per-stream provenance: Cam-0 (default primary & default shadow), Cam-1 (custom primary &
#     custom shadow), Cam-2 (custom primary & default shadow), Cam-3 (default primary & custom shadow).
#   - all four run together, then detach cleanly.
test_10() {
  echo "-----------------------------------------------------------------------"
  echo "[test_10] starting test 10 (default primary & default shadow + custom primary & custom shadow, mixed per stream)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set        source-list          max-batch-size        4
  cfg_set        tiled-display        rows                  2
  cfg_set        tiled-display        columns               4
  # multimodel config updates
  cfg_set        multimodel           unified-batch         1
  cfg_set        multimodel           per-model-batch-size  2
  # BOTH defaults are established at RUNTIME via model/load + a model/routing default switch
  # (the load_model def_primary/def_shadow flags; v1 removed set_default_* from model/load) --
  # comment out the config defaults for both roles so the API is the sole source of the defaults.
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the default primary & default shadow models (warm to READY)
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True" 
  echo "[test_10] Requested loading of (Default Primary Model '${DEFAULT_MODEL}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  # def_primary -> loaded (limbo), then model/routing {"default":{"model":...}} makes it the runtime default model.
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_10: default primary '${DEFAULT_MODEL}' loaded into pool + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_10] Requested loading of (Default Shadow Model '${DEFAULT_MODEL}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  # def_shadow -> loaded (limbo), then model/routing {"default":{"shadow":...}} makes it the runtime default shadow.
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_10: default shadow '${DEFAULT_MODEL}' loaded into pool + designated default" 12

  # 4) load the custom primary & custom shadow models (warm to READY)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_10] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_10: custom primary '${CAR_MODEL}' loaded and warmed" 12
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "${BEST_ENGINE}"
  echo "[test_10] Requested loading of (Custom Shadow Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' LOADED into limbo" "test_10: custom shadow '${CAR_MODEL}' loaded and warmed" 12

  # 5) add a stream on the default primary & default shadow model
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_10] Added Cam-0 (Default Primary Model & Default Shadow Model)"
  sleep "${DELAY}"

  # 6) add a stream that selects the loaded custom primary & custom shadow via metadata
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_10] Added Cam-1 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Custom Shadow Model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  sleep "${DELAY}"

  # 7) add a stream that selects the loaded custom primary & without default shadow via metadata
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_10] Added Cam-2 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  sleep "${DELAY}"

  # 8) add a stream that selects the loaded default primary & custom shadow via metadata
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "" "" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_10] Added Cam-3 (Default Primary Model & Custom Shadow Model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  
  # 9) Assert all four role combinations
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_10: Cam-0 baseline = default primary + default shadow" 10
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_10: Cam-1 = custom primary + custom shadow" 10
  assert_log "stream 2 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_10: Cam-2 = custom primary + default shadow" 10
  assert_log "stream 3 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_10: Cam-3 = default primary + custom shadow (same models, different role combos)" 10
  sleep "${DELAY}"

  # 10) stop the app
  # stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 11: per-sensor routing via CONFIG ([stream-model-*]) -- test_10's outcome, config-driven.
#
# Flow:
#   1. Declare per-sensor bindings IN THE CONFIG (no REST model/load, no stream metadata):
#        [stream-model-cam-1]  primary=Car;1 + shadow=Car;2
#        [stream-model-cam-2]  primary=Car;1 (no shadow)
#        [stream-model-cam-3]  shadow=Car;2 (primary falls back to the default)
#      Per-version engine artifacts ([model-<name>-<version>]) pin each version's checkpoint.
#   2. Add Cam-0..Cam-3 over REST with PLAIN add_stream (no metadata). Each stream's models
#      come from its config binding matched by camera_id; an unbound sensor uses the default.
#
# What happens (the behaviour under test):
#   LOADING POLICY: only the [multimodel] DEFAULT primary/shadow are eager (preloaded + warmed at
#   startup). A per-sensor-bound model that is NOT a default is LAZY -- warmed on the first
#   stream-add that maps to it (passthrough while warming, then auto-promote, zero-drop).
#   on_stream_added resolves each REST add by camera_id to its binding (stream/add metadata would
#   override; an absent role -> default). The END STATE is IDENTICAL to test_10 (same outcome via
#   REST load_model + stream/add metadata) -- config and metadata are two front-ends to the same
#   per-stream (primary,shadow) resolution + attach.
#
# What to validate:
#   - at STARTUP only the DEFAULTS warm (Trafficcamnet@1 primary + @2 shadow); the bound Car models
#     do NOT preload -- they lazy-load on the first stream-add that maps to them.
#   - Cam-0 = default primary + default shadow (no binding); Cam-1 = Car;1 + Car;2; Cam-2 = Car;1 +
#     default shadow; Cam-3 = default primary + Car;2 -- all from config + plain add_stream. Same
#     four role combos as test_10.
test_11() {
  echo "-----------------------------------------------------------------------"
  echo "[test_11] starting test 11 (per-sensor routing via CONFIG [stream-model-*]; no metadata)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it.
  #    The [stream-model-<id>] groups bind each sensor to its models (cfg_set creates the groups;
  #    referenced models are validated against the catalog but LAZY-loaded on first matching add --
  #    only the [multimodel] defaults are eager-preloaded at startup).
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list                                               max-batch-size        4
  cfg_set tiled-display                                             rows                  2
  cfg_set tiled-display                                             columns               4
  # multimodel config updates
  cfg_set multimodel                                                unified-batch         1
  cfg_set multimodel                                                per-model-batch-size  2
  cfg_set multimodel                                                primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set multimodel                                                shadow-model          "${DEFAULT_MODEL};${DEFAULT_SHADOW_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}           engine                "${WORST_ENGINE}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_SHADOW_MODEL_VERSION}    engine                "${BEST_ENGINE}"
  cfg_set model-${CAR_MODEL}-${CAR_MODEL_VERSION}                   engine                "${WORST_ENGINE}"
  cfg_set model-${CAR_MODEL}-${CAR_MODEL_VERSION_NEW}               engine                "${BEST_ENGINE}"
  # per-sensor bindings: each role is a single "name;version" ref into the catalog
  # (config comes from the catalog, not here).
  cfg_set stream-model-cam-1   primary-model                    "${CAR_MODEL};${CAR_MODEL_VERSION}"
  cfg_set stream-model-cam-1   shadow-model                     "${CAR_MODEL};${CAR_MODEL_VERSION_NEW}"
  cfg_set stream-model-cam-2   primary-model                    "${CAR_MODEL};${CAR_MODEL_VERSION}"
  cfg_set stream-model-cam-3   shadow-model                     "${CAR_MODEL};${CAR_MODEL_VERSION_NEW}"

  # 2) start the app -- ONLY the defaults are eager-preloaded; the per-sensor-bound Car models are
  #    LAZY (warmed on the first stream-add that maps to them, not at boot).
  start_app || return
  assert_log "preload default PRIMARY model '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_11: default primary '${DEFAULT_MODEL}' eager-preloaded at startup" 12
  assert_log "preload default SHADOW model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_11: default shadow '${DEFAULT_MODEL}' eager-preloaded at startup" 12
  sleep "${DELAY}"

  # 3) Cam-0: no binding -> default primary, no shadow
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_11] added Cam-0 (no binding -> default primary)"
  sleep "${DELAY}"

  # 4) Cam-1: bound to primary=Car;1 + shadow=Car;2 (via config, no metadata). These bound models
  #    are NOT preloaded -> they LAZY-LOAD here on the first add that maps to them.
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_11] added Cam-1 (config binding: primary='${CAR_MODEL}@${CAR_MODEL_VERSION}' shadow='${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  assert_log "lazy-load primary model '${CAR_MODEL}@${CAR_MODEL_VERSION}' on stream-add" "test_11: bound primary '${CAR_MODEL}' lazy-loaded on Cam-1 add (not at startup)" 12
  assert_log "lazy-load shadow model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' on stream-add" "test_11: bound shadow '${CAR_MODEL}' lazy-loaded on Cam-1 add (not at startup)" 12
  sleep "${DELAY}"

  # 5) Cam-2: bound to primary=Car;1 (shadow falls back to the default)
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_11] added Cam-2 (config binding: primary='${CAR_MODEL}@${CAR_MODEL_VERSION}' shadow='${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  sleep "${DELAY}"

  # 6) Cam-3: bound to shadow=Car only -> primary falls back to the default
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
  echo "[test_11] added Cam-3 (config binding: default primary + shadow='${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  sleep "${DELAY}"

  # 7) Assert all four role combinations
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_11: Cam-0 baseline = default primary + default shadow" 10
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_11: Cam-1 = Car;1 primary + Car;2 shadow" 10
  assert_log "stream 2 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_11: Cam-2 = Car;1 primary + default shadow" 10
  assert_log "stream 3 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_11: Cam-3 = default primary + Car;2 shadow (same models, different role combos)" 10
  sleep "${DELAY}"

  # 8) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 12: model/status GET -- full dump + schema-stable filtering (?model / ?stream).
#
# Flow:
#   1. Load Car;1 (custom primary) + Car;2 (custom shadow); bring up 4 streams with a
#      MIX of routings so the tables have variety:
#        Cam-0 -> default primary (no shadow)
#        Cam-1 -> Car;1 primary + Car;2 shadow
#        Cam-2 -> Car;1 primary (no shadow)
#        Cam-3 -> default primary + Car;2 shadow
#   2. GET model/status four ways: no filter, ?model[&version], ?stream, and both.
#
# What happens (the behaviour under test):
#   model/status returns the live pool + routing under "model-status" as
#   {"routing_revision":N,"models":[{"name","versions":[...]}],"streams":[...]} -- a
#   SCHEMA-STABLE shape. The optional query-string filters JOIN (v1: never orphan a view):
#   ?model=X[&version=N] -> that model AND the streams routed to it; ?stream=Y/?source_id
#   -> that stream AND the model versions it uses; both -> the intersection; none ->
#   everything.
#
# What to validate:
#   - identity is reported as SEPARATE ref fields ({"name": ..., "version": ...} refs,
#     nested versions under one model entry), never a combined "name@version".
#   - no filter: models[] lists Car@1 + Car@2 (+ the default); streams[] lists every camera.
#   - ?model=Car&version=1: models[] = [Car@1] ONLY (version refines name; Car@2 and the
#     default excluded); streams[] = ONLY the streams routed to Car@1 (the JOIN -- Cam-1 and
#     Cam-2, not Cam-0/Cam-3). (params are SEPARATE, never "Car;1".)
#   - ?stream=Cam-1: streams[] = [Cam-1] (model=Car@1, shadow=Car@2); models[] = ONLY the
#     versions Cam-1 uses (the JOIN -- the unused default model is excluded).
#   - ?model=Car&version=1&stream=Cam-1: one model version + one stream; others excluded.
#   - the response shape is ALWAYS {"models":[...],"streams":[...]} (parsers never branch).
test_12() {
  echo "-----------------------------------------------------------------------"
  echo "[test_12] starting test 12 (model/status GET: full + filtered by model/stream)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               4
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the custom primary & custom shadow models (no flags -> type-less limbo; warm to READY)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_12] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_12: custom primary '${CAR_MODEL}' loaded and warmed" 12
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "${BEST_ENGINE}"
  echo "[test_12] Requested loading of (Custom Shadow Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' LOADED into limbo" "test_12: custom shadow '${CAR_MODEL}' loaded and warmed" 12

  # 4) add a stream on the Default Primary Model
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_12] Added Cam-0 (Default Primary Model '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  sleep "${DELAY}"

  # 6) add a stream that selects the loaded custom primary & custom shadow via metadata
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_12] Added Cam-1 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Custom Shadow Model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  sleep "${DELAY}"

  # 7) add a stream that selects the loaded custom primary via metadata
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_12] Added Cam-2 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}')"
  sleep "${DELAY}"

  # 8) add a stream that selects the loaded default primary & custom shadow via metadata
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "" "" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_12] Added Cam-3 (Default Primary Model & Custom Shadow Model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  sleep "${DELAY}"

  # 9) FULL status (no filter): both arrays populated. Identity is reported as SEPARATE
  #    name + version fields (a model entry is {"name": ..., "versions":[{"version": ...}]};
  #    stream refs are {"name", "version"}), never a combined "name@version" -- match each
  #    field independently. (jsoncpp re-serializes the body with keys sorted alphabetically;
  #    the asserts strip whitespace and rely only on alphabetical-adjacent pairs.)
  #    A model entry's keys are name < process_mode < versions, so the name->versions pair
  #    is NOT adjacent: process_mode sits between them and has to be spelled out. It tags
  #    WHICH STAGE hosts the model, which is what lets a client tell the stages apart once
  #    a multi-stage pipeline's per-stage answers are merged -- so asserting the 1 here is
  #    the primary-side complement of test_59's "process_mode":2 at the secondary.
  get_model_status
  assert_status_contains "\"name\":\"${CAR_MODEL}\",\"process_mode\":1,\"versions\":["  "test_12: status(no filter) lists model '${CAR_MODEL}'"
  assert_status_contains "\"version\":\"${CAR_MODEL_VERSION}\""     "test_12: status(no filter) lists ${CAR_MODEL}@${CAR_MODEL_VERSION}"
  assert_status_contains "\"version\":\"${CAR_MODEL_VERSION_NEW}\"" "test_12: status(no filter) lists ${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}"
  assert_status_contains "\"camera_name\":\"Cam-1\""             "test_12: status(no filter) lists stream Cam-1"
  assert_status_contains "\"camera_name\":\"Cam-3\""             "test_12: status(no filter) lists all streams"

  # 9b) THE TWO IDENTITY FIELDS ARE NOT INTERCHANGEABLE. These streams were added with a
  #     camera_id ("${PREFIX}1") that differs from the display name ("Cam-1") -- the normal
  #     case -- so reporting one where the other belongs is detectable. camera_id must be
  #     the ROUTABLE token: the same string model/routing lists in routes[].streams and the
  #     only one a per-stream POST accepts. Emit the display name there instead and every
  #     value a client lifts from this response is answered with STREAM_UNKNOWN.
  assert_status_contains     "\"camera_id\":\"${PREFIX}1\"" \
      "test_12: camera_id carries the ROUTABLE id, the token model/routing reports"
  assert_status_not_contains "\"camera_id\":\"Cam-1\"" \
      "test_12: ...and never the display name (that belongs in camera_name)"

  # 9c) the round trip that matters: an id taken from the ROUTING plane must resolve on the
  #     STATUS plane. These two endpoints describing the same stream by different names is
  #     what makes an id useless -- it has to be one identity, usable in both.
  get_model_routing "stream=${PREFIX}1"
  assert_route_entry "${PREFIX}1" "state=serving" \
      "test_12: model/routing resolves the stream by its camera_id"
  get_model_status "stream=${PREFIX}1"
  assert_status_contains     "\"camera_id\":\"${PREFIX}1\"" \
      "test_12: ...and model/status resolves that SAME id (cross-plane round trip)"
  assert_status_not_contains "\"camera_name\":\"Cam-2\"" \
      "test_12: ...narrowing to that one stream"
  get_model_status
  sleep "${DELAY}"

  # 10) MODEL filter (SEPARATE params): ?model=Car&version=1 -> models[] = [Car@1] ONLY
  #     (the version refines the name), and streams[] = the JOIN: only the streams ROUTED
  #     to Car@1 (Cam-1, Cam-2) -- a model filter never returns an orphaned/empty streams
  #     view of a model that is actively serving. (model + version params are SEPARATE,
  #     never combined as "Car;1".) NOTE: Cam-1's own entry still shows its Car@2 shadow
  #     REF ("name":"Car","version":"2"), so "version 2 excluded" is asserted on the
  #     MODELS array via the version-entry-only "state","version" adjacency.
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION}"
  assert_status_contains     "\"name\":\"${CAR_MODEL}\",\"process_mode\":1,\"versions\":["  "test_12: status(model=${CAR_MODEL};${CAR_MODEL_VERSION}) returns that model"
  assert_status_contains     "\"version\":\"${CAR_MODEL_VERSION}\"" "test_12: status(model filter) is version ${CAR_MODEL_VERSION}"
  assert_status_contains     "\"camera_name\":\"Cam-1\""            "test_12: status(model filter) JOINS the streams routed to ${CAR_MODEL}@${CAR_MODEL_VERSION} (Cam-1)"
  assert_status_contains     "\"camera_name\":\"Cam-2\""            "test_12: status(model filter) JOINS the streams routed to ${CAR_MODEL}@${CAR_MODEL_VERSION} (Cam-2)"
  assert_status_not_contains "\"camera_name\":\"Cam-0\""            "test_12: status(model filter) excludes streams NOT routed to the model (Cam-0 on the default)"
  assert_status_not_contains "\"state\":\"serving\",\"version\":\"${CAR_MODEL_VERSION_NEW}\"" "test_12: status(model+version filter) models[] excludes the OTHER version (${CAR_MODEL}@${CAR_MODEL_VERSION_NEW} serving entry)"
  assert_status_not_contains "\"state\":\"ready\",\"version\":\"${CAR_MODEL_VERSION_NEW}\"" "test_12: status(model+version filter) models[] excludes the OTHER version (${CAR_MODEL}@${CAR_MODEL_VERSION_NEW} idle entry)"
  assert_status_not_contains "\"name\":\"${DEFAULT_MODEL}\""       "test_12: status(model filter) excludes the default model"
  sleep "${DELAY}"

  # 11) STREAM filter: ?stream=Cam-1 -> streams[] = [Cam-1] only, and models[] = the JOIN:
  #     only the versions Cam-1 uses (Car@1 + Car@2; the unused default model is excluded).
  get_model_status "stream=Cam-1"
  assert_status_contains     "\"name\":\"${CAR_MODEL}\",\"process_mode\":1,\"versions\":["  "test_12: status(stream filter) JOINS the model versions Cam-1 uses"
  assert_status_not_contains "\"name\":\"${DEFAULT_MODEL}\""      "test_12: status(stream filter) models[] excludes the UNUSED default model"
  assert_status_contains     "\"camera_name\":\"Cam-1\""             "test_12: status(stream=Cam-1) returns that stream"
  assert_status_ref     model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"              "test_12: Cam-1 serving model is '${CAR_MODEL}'"
  assert_status_ref     model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"     "test_12: Cam-1 serving-model version is ${CAR_MODEL_VERSION}"
  assert_status_ref     shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"               "test_12: Cam-1 shadow model is '${CAR_MODEL}'"
  assert_status_ref     shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_12: Cam-1 shadow version is ${CAR_MODEL_VERSION_NEW}"
  assert_status_not_contains "\"camera_name\":\"Cam-2\""             "test_12: status(stream filter) excludes other streams"
  sleep "${DELAY}"

  # 11b) SHADOW-LANE JOIN. ${CAR_MODEL}@${CAR_MODEL_VERSION_NEW} is nobody's primary -- it is
  #      ONLY ever a shadow (Cam-1 and Cam-3). A model filter must therefore match on the
  #      SHADOW lane too, or "which streams use this model" silently answers "none" for every
  #      A/B model in the fleet. Cam-2 is the control: it runs ${CAR_MODEL}@${CAR_MODEL_VERSION}
  #      only, so a filter that matched on NAME instead of name+version would wrongly pull it in.
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION_NEW}"
  assert_status_contains     "\"camera_name\":\"Cam-1\"" \
      "test_12: status(model=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}) JOINS a stream via its SHADOW lane (Cam-1)"
  assert_status_contains     "\"camera_name\":\"Cam-3\"" \
      "test_12: ...and every such stream, not just the first (Cam-3)"
  assert_status_not_contains "\"camera_name\":\"Cam-2\"" \
      "test_12: ...but NOT a stream on a different version of the same name (Cam-2 is on @${CAR_MODEL_VERSION})"
  assert_status_not_contains "\"camera_name\":\"Cam-0\"" \
      "test_12: ...nor one that never touches the model (Cam-0)"
  assert_status_not_contains "\"state\":\"serving\",\"version\":\"${CAR_MODEL_VERSION}\"" \
      "test_12: ...and models[] holds the filtered version alone"
  # a shadow-only version reports the shadow lane ONLY: an explicit "streams":0 next to
  # "state":"serving" reads as a contradiction, so the lane it does not hold is omitted.
  assert_status_contains     "\"serving\":{\"shadow_streams\":2}" \
      "test_12: ...and its serving{} counts the SHADOW lane only (no misleading \"streams\":0)"

  # 11c) A filtered view carries no FLEET-WIDE state: default{} names a model that is usually
  #      outside the filter, and the fleet routing summary belongs to model/routing. Asserted
  #      against the KEY, not a model name, so it cannot pass just because the default happens
  #      to be the filtered model.
  assert_status_not_contains "\"default\":{" \
      "test_12: status(filtered) omits the fleet default{} block"
  assert_status_not_contains "\"routing\":{" \
      "test_12: status(filtered) carries no fleet routing summary"
  get_model_status
  assert_status_contains     "\"default\":{" \
      "test_12: status(UNFILTERED) DOES report default{} -- the checks above are not vacuous"
  assert_status_not_contains "\"routing\":{" \
      "test_12: status(unfiltered) still carries no routing summary (that plane is model/routing's)"

  # 11d) STREAM filter joins BOTH lanes: Cam-1 runs @${CAR_MODEL_VERSION} as primary and
  #      @${CAR_MODEL_VERSION_NEW} as shadow, so models[] must hold BOTH version entries.
  #      Anchored on the version-entry-only "state","version" adjacency so a stream REF
  #      carrying the same version cannot satisfy it.
  get_model_status "stream=Cam-1"
  assert_status_contains "\"state\":\"serving\",\"version\":\"${CAR_MODEL_VERSION}\"" \
      "test_12: status(stream=Cam-1) models[] holds its PRIMARY version"
  assert_status_contains "\"state\":\"serving\",\"version\":\"${CAR_MODEL_VERSION_NEW}\"" \
      "test_12: ...and its SHADOW version -- an A/B stream joins two versions, not one"

  # 11e) ?source_id= is the numeric spelling of ?stream=; it must narrow identically.
  get_model_status "source_id=1"
  assert_status_contains     "\"camera_name\":\"Cam-1\"" "test_12: status(source_id=1) resolves the same stream as ?stream=Cam-1"
  assert_status_not_contains "\"camera_name\":\"Cam-0\"" "test_12: ...and narrows streams[] like the named filter does"
  assert_status_not_contains "\"default\":{"           "test_12: ...and is a filtered view (no fleet default{})"

  # 11f) An unknown filter value is an EMPTY result, never the unfiltered fleet -- the failure
  #      mode that makes a filter dangerous is silently ignoring it.
  get_model_status "model=NoSuchModel"
  assert_status_contains     "\"models\":[]"       "test_12: status(model=NoSuchModel) returns an empty models[], not an error"
  assert_status_not_contains "\"camera_name\":\"Cam-1\"" "test_12: ...and no streams leak through an unmatched filter"
  get_model_status "stream=NoSuchCam"
  assert_status_not_contains "\"camera_name\":\"Cam-1\"" "test_12: status(stream=NoSuchCam) matches nothing -- must NOT fall back to every stream"
  assert_status_not_contains "\"camera_name\":\"Cam-0\"" "test_12: ...for any stream"

  # 12) BOTH filters: ?model=Car&version=1&stream=Cam-1 -> one model version + one stream.
  #     (Cam-1's entry still carries its Car@2 shadow REF; the "other version excluded"
  #     check is anchored to the version-entry-only "state","version" adjacency.)
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION}&stream=Cam-1"
  assert_status_contains     "\"name\":\"${CAR_MODEL}\",\"process_mode\":1,\"versions\":["  "test_12: status(model+stream) returns the model entry"
  assert_status_contains     "\"version\":\"${CAR_MODEL_VERSION}\"" "test_12: status(model+stream) model entry is version ${CAR_MODEL_VERSION}"
  assert_status_contains     "\"camera_name\":\"Cam-1\""             "test_12: status(model+stream) returns the stream entry"
  assert_status_not_contains "\"camera_name\":\"Cam-2\""             "test_12: status(model+stream) excludes other streams"
  assert_status_not_contains "\"state\":\"serving\",\"version\":\"${CAR_MODEL_VERSION_NEW}\"" "test_12: status(model+stream) models[] holds only ${CAR_MODEL}@${CAR_MODEL_VERSION} (no serving @${CAR_MODEL_VERSION_NEW} entry)"
  assert_status_not_contains "\"state\":\"ready\",\"version\":\"${CAR_MODEL_VERSION_NEW}\"" "test_12: status(model+stream) models[] holds only ${CAR_MODEL}@${CAR_MODEL_VERSION} (no idle @${CAR_MODEL_VERSION_NEW} entry)"
  sleep "${DELAY}"

  # 13) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 13: live per-stream model/routing -- reroute, reject, shadow-clear, rollback, shadow-only.
#
# Flow:
#   1. Add Cam-0 (default). model/load Car@1 + Car@2. Add Cam-1 = Car@1 + Car@2.
#   2. model/routing Cam-0 -> model=Car@1, shadow=Car@2          (live reroute).
#   3. model/routing Cam-0 -> model=Dummy_Primary@1             (NOT loaded -> reject).
#   4. model/routing Cam-0 -> model=Car@1, shadow=none              (clear the shadow).
#   5. model/routing Cam-0 -> model=DEFAULT_MODEL             (roll back primary to default).
#   6. model/routing Cam-0 -> model="" (OMITTED), shadow=Car@2    (SHADOW-ONLY; primary untouched).
#
# What happens (the behaviour under test):
#   A scoped live update reroutes only Cam-0. Targeting a model that was never loaded is
#   REJECTED and the stream keeps its current models (no auto-load, no fallback). shadow=none
#   drops just the shadow. The primary rollback exercises safely reusing a default bin that had
#   been drained earlier. The final step is a SHADOW-ONLY update with the PRIMARY OMITTED
#   entirely -- the primary is NOT mandatory, so only the shadow changes and the
#   primary is left exactly as it was (an omitted role is never touched).
#
# What to validate:
#   - reject: log shows "update rejected: primary 'Dummy_Primary@1'"; Cam-0 keeps Car@1+Car@2.
#   - reroute / clear / rollback / shadow-only each log "model/routing applied to 1 stream(s)".
#   - shadow-only: model/status shows primary UNCHANGED (still the default) + shadow now Car@2.
#   - per-stream frame table: dropped = 0 across every step (no stall on the rollback to default).
test_13() {
  echo "-----------------------------------------------------------------------"
  echo "[test_13] starting test 13 (live model/routing with reject, shadow-clear, rollback)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the custom primary & custom shadow models (no flags -> type-less limbo; warm to READY)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_13] Requested loading of (Custom Primary Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_13: custom primary '${CAR_MODEL}' loaded and warmed" 12
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "${BEST_ENGINE}"
  echo "[test_13] Requested loading of (Custom Shadow Model '${CAR_MODEL}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' LOADED into limbo" "test_13: custom shadow '${CAR_MODEL}' loaded and warmed" 12

  # 4) add Cam-0 with the default primary model without shadow
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_13] added Cam-0 (Default Primary Model)"

  # 5) add a stream that selects the loaded custom primary & custom shadow via metadata
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_13] Added Cam-1 (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Custom Shadow Model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  sleep "${DELAY}"

  # 6) re-route LIVE Cam-0 to the custom primary + custom shadow pair
  log_mark
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_13] re-routed Cam-0 -> Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Custom Shadow Model '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}'"
  assert_log "model/routing applied to 1 stream" "test_13: scoped reroute applied to Cam-0 only (Cam-1 untouched)" 10
  # Verify the reroute actually TOOK EFFECT (not just that the API was accepted). The swap
  # is async (zero-drop hitless), so POLL the live routing until it reflects Car@1+Car@2
  # -- this also gates the next update until the swap has settled (see assert_status_eventually).
  assert_status_ref_eventually "stream=Cam-0" model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_13: Cam-0 primary actually switched to '${CAR_MODEL}@${CAR_MODEL_VERSION}' (reroute took effect)"
  assert_status_ref_eventually "stream=Cam-0" shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_13: Cam-0 shadow actually switched to '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}'"
  sleep "${DELAY}"

  # 7) attempt re-route to a NOT-loaded model for Cam-0 -> must be rejected,
  #    stream keeps its current models (no auto-load, no fallback)
  log_mark
  model_routing "${PREFIX}0" "Dummy_Primary" "1"
  echo "[test_13] attempted re-route to unloaded model (expected: rejected)"
  # route admission is synchronous now -> the reject reason is in the REST response
  assert_rest_body_contains "MODEL_NOT_LOADED" "test_13: re-route to NOT-loaded model is REJECTED in the REST response (MODEL_NOT_LOADED)"
  # a rejected update must leave Cam-0's routing exactly as it was -- no auto-load, no
  # fallback. Confirm it still runs Car@1+Car@2.
  assert_status_ref_eventually "stream=Cam-0" model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_13: Cam-0 KEPT '${CAR_MODEL}@${CAR_MODEL_VERSION}' after the rejected update (no fallback)"
  assert_status_ref_eventually "stream=Cam-0" shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_13: Cam-0 KEPT shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' after the rejected update"
  sleep "${DELAY}"

  # 8) clear the custom shadow model and keep custom primary for Cam-0
  log_mark
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "none"
  echo "[test_13] cleared Custom Shadow Model on Cam-0 and kept the Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}'"
  assert_log "stream 0 Shadow role CLEARED" "test_13: shadow dropped on Cam-0 (zero-drop), custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' kept" 10
  # confirm the clear's EFFECT: Cam-0 keeps the custom primary but no longer carries the
  # Car shadow.
  assert_status_ref_eventually        "stream=Cam-0" model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_13: Cam-0 kept primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' after shadow clear"
  assert_status_ref_eventually_absent "stream=Cam-0" shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_13: Cam-0 shadow actually cleared (no longer '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  sleep "${DELAY}"

  # 9) move Cam-0 primary back to the default primary model
  log_mark
  model_routing "${PREFIX}0" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  echo "[test_13] moved Cam-0 back to Default Primary Model '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' from Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}'"
  assert_log "model/routing applied to 1 stream" "test_13: rolled Cam-0 back to default primary (reuses drained default bin, no stall)" 12
  # confirm the rollback landed on the DEFAULT primary (not stuck on the custom Car@1).
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_13: Cam-0 primary rolled back to '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'"
  sleep "${DELAY}"

  # 10) shadow-only update: primary OMITTED -> set shadow only, primary untouched
  log_mark
  model_routing "${PREFIX}0" "" "" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_13] shadow-only update on Cam-0 -> shadow='${CAR_MODEL}' (primary omitted, untouched)"
  assert_log "model/routing applied to 1 stream" "test_13: shadow-only update accepted (primary omitted)" 12
  # THE point of this step: an omitted primary must be left EXACTLY as it was. Verify the
  # primary is still the default (untouched) AND the shadow is now Car. This is the
  # check the case description promised but never actually made.
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_13: shadow-only update LEFT primary untouched (still '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  assert_status_ref_eventually "stream=Cam-0" shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_13: shadow-only update set shadow to '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}'"
  sleep "${DELAY}"

  # 11) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 14: PROMOTE family overview + the both-default cell. "Promotion makes the promoted model
#          THE primary -- so it becomes the new default_primary."
#
# PROMOTE RULE (in-place, scope-ALL): the shadow S is promoted to primary on every stream and the
# shadow role is cleared on the streams. The promoted model S ALWAYS becomes the new default_primary
# (it is now the primary baseline for everyone); default_shadow is cleared ONLY IF the promoted model
# WAS the default_shadow (the crowned challenger) -- an UNINVOLVED sideline default_shadow is left
# untouched. The displaced old primary -- plus ANY prior default_primary it supersedes -- is
# reclaimed/revoked. So all four promote cases converge on default_primary = the promoted model, and
# differ only in SETUP (and in whether default_shadow is cleared or kept):
#   test_14 = def_p / def_s    (both were defaults; default_shadow crowned -> cleared)
#   test_15 = def_p / custom   (only the primary was a default; default_shadow was none)
#   test_16 = custom / custom  (sideline default_shadow KEPT untouched; orphaned default_primary reclaimed)
#   test_17 = custom / def_s   (the default_shadow is the one promoted -> cleared; prior default_primary reclaimed)
#   test_18 = group-local PROMOTE (independent A/B groups; the promote done per-group, no model load)
# then the SWAP family. SWAP RULE (default status follows the SLOT, not the model -- whoever lands in
# a slot that WAS a default's slot acquires that status; the model leaving it gives it up; nothing is
# reclaimed since both models keep serving). All four default cases + scoped + group-local:
#   test_19 = def_p / def_s    (both defaults -> clean EXCHANGE: dp<->ds)
#   test_20 = def_p / custom   (custom ACQUIRES default_primary; default_shadow unchanged)
#   test_21 = custom / custom  (NEITHER a default -> configured defaults UNTOUCHED)
#   test_22 = custom / def_s   (custom ACQUIRES default_shadow; default_primary unchanged)
#   test_23 = scoped swap      (subset of streams -> create+warm fallback; defaults UNTOUCHED)
#   test_24 = group-local SWAP (independent A/B groups; the swap done per-group, no model load)
#
# This case (both-default): default primary and default shadow are both Trafficcamnet, different
# versions (v1 primary, v2 shadow), each pinned to its own engine, established AT RUNTIME via
# the def_primary / def_shadow load flags (load + route default; config defaults commented out).
#
# Flow:
#   1. model/load Trafficcamnet@1 (def_primary flag) + Trafficcamnet@2 (def_shadow flag).
#   2. Bring up Cam-0/Cam-1: primary=Trafficcamnet@1 + shadow=Trafficcamnet@2 (the defaults).
#   3. model/routing model=Trafficcamnet;2 shadow=none (scope = every stream) -- promote the shadow.
#
# What happens (behaviour under test):
#   The warm @2 bin is RE-LABELED shadow->primary IN PLACE (no new bin, no reload, gie kept). The
#   promoted @2 becomes the new default_primary, default_shadow is cleared, and the old
#   default_primary @1 -- no longer a default -- is RECLAIMED (auto-unloaded when idle).
#
# What to validate:
#   - "PROMOTE (IN-PLACE)" + "reused existing bins, gie kept, NO reload".
#   - "DEFAULT promote: promoted 'Trafficcamnet@2' is now default_primary" (default_shadow cleared).
#   - the displaced old primary @1 is RECLAIMED ("displaced by reroute and now idle -> auto-unload").
#   - model/status: @2 shows "default":true; streams run primary=@2, no shadow; dropped = 0.
test_14() {
  echo "-----------------------------------------------------------------------"
  echo "[test_14] starting test 14 (PROMOTE both-default pair | default rotates to promoted model, old default reclaimed)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # BOTH defaults are established at RUNTIME via model/load + a model/routing default switch
  # (the load_model def_primary/def_shadow flags; v1 removed set_default_* from model/load) --
  # comment out the config defaults for both roles so the API is the sole source of the defaults.
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the default primary & default shadow models (warm to READY)
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_14] Requested loading of (Default Primary Model '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  # def_primary -> loaded (limbo), then model/routing {"default":{"model":...}} makes it the runtime default model.
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_14: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded into pool + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_14] Requested loading of (Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  # def_shadow -> loaded (limbo), then model/routing {"default":{"shadow":...}} makes it the runtime default shadow.
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_14: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded into pool + designated default" 12

  # 4) add streams on the DEFAULT primary + DEFAULT shadow (a cross-version A/B of Trafficcamnet:
  #    v1 on the top row, v2 on the bottom row), both selected via the runtime defaults.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_14] Added Cam-0 (Default Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + Default Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_14: Cam-0 = default primary + default shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_14] Added Cam-1 (Default Primary Model + Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_14: Cam-1 = default primary + default shadow" 10
  sleep "${DELAY}"

  # 5) PROMOTE: make the default shadow model the primary on ALL streams and drop the shadow.
  #    Scope covers every stream, so the warm @2 bin is RE-LABELED shadow->primary IN PLACE
  #    (no new bin, no reload, gie kept). Both involved models are the configured defaults, so
  #    the defaults ROTATE: default_primary -> @2, default_shadow cleared, old default @1 revoked.
  log_mark
  model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "none"
  echo "[test_14] PROMOTE (ALL): primary -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}', shadow cleared (both-default pair -> default rotates)"
  assert_log "PROMOTE (IN-PLACE)" "test_14: promote done IN-PLACE (reused bin, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_14: bin REUSED -- gie-id preserved, no new bin / no engine reload" 10
  assert_log "Primary display moved to top row" "test_14: promoted model moved to the primary/top tiler row" 10
  # DEFAULT-ROTATION (the point of this case), proven in three parts:
  #  (a) the defaults followed the op: the promoted default-shadow @2 becomes the new
  #      default_primary and the default_shadow is emptied.
  assert_log "DEFAULT promote: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now default_primary" "test_14: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' became the new default_primary; default_shadow cleared" 10
  #  (b) the in-place drop path names the OLD primary -- the former default_primary @1 (the
  #      promoted new primary is the former shadow @2), so the drop-path log names @1.
  assert_log "IN-PLACE promote: dropping old primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_14: displaced old primary was the former default '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" 10
  #  (c) and because it is no longer a default, the old default @1 IS RECLAIMED -- it hits the
  #      "displaced by reroute and now idle -> auto-unload" path (NOT kept/protected anymore).
  assert_log "'${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' displaced by reroute and now idle" "test_14: old default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' REVOKED + reclaimed (no longer protected)" 10
  sleep "${DELAY}"

  # 6) end state: every stream runs the promoted primary @2 (no shadow), and @2 is now the
  #    configured default_primary ("default":true via model/status).
  assert_status_ref_eventually        "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_14: Cam-0 now runs promoted primary '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'"
  assert_status_ref_eventually_absent "stream=Cam-0" shadow "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_14: Cam-0 shadow cleared by the promote"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"model\"]" "test_14: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now the default_primary (default_for [\"model\"])"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 15: PROMOTE a non-default shadow over the DEFAULT primary -- the PROMOTED model becomes the
#          new default_primary; the old default primary is REVOKED + reclaimed.
#
# The def_p / custom cell of the promote matrix. Primary and shadow are the SAME model, DIFFERENT
# versions: default primary = Trafficcamnet@1 (def_primary flag), shadow = Trafficcamnet@2
# (CUSTOM -- loaded WITHOUT a def_* flag). PROMOTE RULE: the promoted model is now the primary
# for every stream, so it ALWAYS becomes the new default_primary (even a custom), default_shadow is
# cleared, and the displaced old primary (the former default_primary @1) is reclaimed.
#
# Flow:
#   1. model/load Trafficcamnet@1 (def_primary flag) + Trafficcamnet@2 (NO flag, custom).
#   2. Bring up Cam-0/Cam-1: primary OMITTED (-> default @1) + shadow=@2 (custom).
#   3. model/routing model=Trafficcamnet;2 shadow=none (scope=all) -- promote the shadow.
#
# What happens (behaviour under test):
#   The warm @2 bin is RE-LABELED shadow->primary IN PLACE (no new bin, no reload, gie kept). The
#   promoted @2 becomes the new default_primary; default_shadow is cleared; the displaced old
#   default_primary @1 is REVOKED (no longer a default) and reclaimed (auto-unloaded). Contrast
#   test_14 (both default) which is now identical in outcome -- it differs only in that @2 was the
#   default_shadow there (cleared) vs a plain custom shadow here. (See test_14 for the full promote
#   matrix overview.)
#
# What to validate:
#   - "PROMOTE (IN-PLACE)" + "reused existing bins, gie kept, NO reload".
#   - "DEFAULT promote: promoted '@2' is now default_primary" -- the promoted model is the new default.
#   - the old default primary @1 is RECLAIMED ("displaced by reroute and now idle -> auto-unload").
#   - model/status: @2 shows "default":true; streams run primary=@2; per-stream dropped = 0.
test_15() {
  echo "-----------------------------------------------------------------------"
  echo "[test_15] starting test 15 (PROMOTE non-default shadow over DEFAULT primary | promoted becomes default, old default revoked)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # default primary is established at RUNTIME (def_primary flag); comment the config defaults so
  # the API is the sole source. The shadow is loaded WITHOUT a flag -> stays CUSTOM (not default).
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the DEFAULT primary (def_primary flag) + a CUSTOM shadow -- SAME model, different
  #    version. The shadow gets NO def_* flag, so it is a plain custom (NOT default_shadow).
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_15] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_15: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded into pool + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}"
  echo "[test_15] Requested loading of (CUSTOM Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' -- NOT a default)"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "'${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' LOADED into limbo" "test_15: custom shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded (type-less, not a default)" 12

  # 4) streams: primary OMITTED (-> resolves to default @1) + custom shadow @2 via metadata.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "" "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "[test_15] Added Cam-0 (DEFAULT primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + CUSTOM shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_15: Cam-0 = default primary + custom shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "" "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "[test_15] Added Cam-1 (DEFAULT primary + CUSTOM shadow)"
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_15: Cam-1 = default primary + custom shadow" 10
  sleep "${DELAY}"

  # 5) PROMOTE the shadow @2 to primary on ALL streams, clear the shadow. By the promote rule the
  #    promoted model BECOMES the new default_primary (even though @2 here was a CUSTOM shadow, not
  #    a configured default), default_shadow is cleared, and the OLD default_primary @1 (the
  #    displaced primary) is REVOKED + reclaimed (auto-unloaded).
  log_mark
  model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "none"
  echo "[test_15] PROMOTE (ALL): primary -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}', shadow cleared; promoted model becomes default_primary, old default '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' revoked"
  assert_log "PROMOTE (IN-PLACE)" "test_15: promote done IN-PLACE (reused bin, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_15: bin REUSED -- gie-id preserved, no new bin / no engine reload" 10
  assert_log "Primary display moved to top row" "test_15: promoted model moved to the primary/top tiler row" 10
  # the promoted model becomes the new default_primary (a custom promoted to primary everywhere
  # still becomes the baseline), and default_shadow is cleared.
  assert_log "DEFAULT promote: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now default_primary" "test_15: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' became the new default_primary" 10
  # the old default_primary @1 -- the displaced old primary, now no longer a default -- is RECLAIMED.
  assert_log "'${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' displaced by reroute and now idle" "test_15: old default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' REVOKED + reclaimed after promote" 10
  sleep "${DELAY}"

  # 6) end state: streams run the promoted primary @2, which is now the default ("default":true);
  #    the old default @1 has been reclaimed.
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_15: Cam-0 now runs promoted primary '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"model\"]" "test_15: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now the default_primary (default_for [\"model\"])"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 16: PROMOTE a CUSTOM shadow over a CUSTOM primary -- displaced custom auto-unloaded; the
#          promoted model becomes default_primary, SUPERSEDING the previously-configured defaults.
#
# The custom/custom cell of the promote matrix. Neither model in the A/B pair is a configured
# default; a separate pair of configured defaults (TC@1 primary / TC@2 shadow) is loaded but idle
# (no stream uses them). PROMOTE RULE: the promoted model is now the primary for every stream, so
# it becomes the new default_primary and default_shadow is cleared -- which SUPERSEDES the prior
# configured defaults (they are no longer the defaults, though they stay loaded/idle). The
# displaced custom primary is auto-unloaded.
#
# Flow:
#   1. model/load TC@1 (def_primary flag) + TC@2 (def_shadow flag) -> runtime defaults
#      (eager, idle -- no stream uses them here). model/load Car@1 + Car@2 (custom, limbo).
#   2. Bring up Cam-0/Cam-1 with primary=Car;1 + shadow=Car;2 (both CUSTOM, via metadata).
#   3. model/routing model=Car;2 shadow=none (scope=all) -- promote the custom shadow.
#
# What to validate:
#   - PROMOTE (IN-PLACE) + reused bins/gie kept; displaced custom primary Car@1 AUTO-UNLOADED.
#   - "DEFAULT promote: promoted 'Car@2' is now default_primary"; Car@2 shows "default":true.
#   - the SIDELINE default_shadow TC@2 is UNTOUCHED (the promote didn't involve it) -- still
#     "default":true, kept warm.
#   - the SUPERSEDED prior default_primary TC@1 (now orphaned + idle) is RECLAIMED (freed; gone).
test_16() {
  echo "-----------------------------------------------------------------------"
  echo "[test_16] starting test 16 (PROMOTE custom shadow over CUSTOM primary | displaced custom unloaded; promoted->default_primary, sideline default_shadow kept, orphaned default_primary reclaimed)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # BOTH defaults are established at RUNTIME (def_primary + def_shadow flags -> load + route default); comment the
  # config defaults so the API is the sole source. The live A/B pair is a SEPARATE custom pair
  # (Car@1/Car@2) loaded WITHOUT flags -> stays custom (neither is a default).
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3a) configured DEFAULTS (eager, sit idle -- no stream uses them in this case).
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_16] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_16: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_16] Requested loading of (DEFAULT Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_16: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded + designated default" 12

  # 3b) the CUSTOM A/B pair (no def_* flag -> type-less limbo).
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_16] Requested loading of (CUSTOM Primary '${CAR_MODEL}@${CAR_MODEL_VERSION}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_16: custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' loaded (limbo)" 12
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "${BEST_ENGINE}"
  echo "[test_16] Requested loading of (CUSTOM Shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' LOADED into limbo" "test_16: custom shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' loaded (limbo)" 12

  # 4) streams on the CUSTOM A/B pair (defaults stay idle/unused).
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_16] Added Cam-0 (CUSTOM primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' + CUSTOM shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  assert_log "stream 0 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_16: Cam-0 = custom primary + custom shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_16] Added Cam-1 (CUSTOM primary + CUSTOM shadow)"
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_16: Cam-1 = custom primary + custom shadow" 10
  sleep "${DELAY}"

  # 5) PROMOTE the CUSTOM shadow @2 -> primary on ALL streams. Neither model is a default, so the
  #    displaced CUSTOM primary @1 is AUTO-UNLOADED and no default machinery runs.
  log_mark
  model_routing "" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "none"
  echo "[test_16] PROMOTE (ALL): primary -> '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' (custom), shadow cleared"
  assert_log "PROMOTE (IN-PLACE)" "test_16: promote done IN-PLACE (reused bin, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_16: bin REUSED -- gie-id preserved, no reload" 10
  # the gie-id names the BIN, so it is the identity proof: the promoted model must serve
  # from the very bin it was already inferring on as the shadow, merely retagged Primary.
  # A rebuilt bin would renumber (fresh bins start at 2 / 101, as test_23 pins).
  assert_log "name=Cam-0  model=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}  role=Primary  gie=1003" \
      "test_16: Cam-0 promoted on '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}'s ORIGINAL bin (gie 1003 kept)" 12
  assert_log "name=Cam-1  model=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}  role=Primary  gie=1003" \
      "test_16: Cam-1 too -- every stream moved onto that same bin, not a copy" 12
  assert_log "Primary display moved to top row" "test_16: promoted model moved to the primary/top tiler row" 10
  # the promoted model becomes the new default_primary. The promoted model was NOT the
  # default_shadow (the streams' shadow was custom Car@2), so the SIDELINE default_shadow TC@2 is
  # left UNTOUCHED -- only the involved defaults change.
  assert_log "DEFAULT promote: promoted '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' is now default_primary" "test_16: promoted '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' became the new default_primary" 10
  # the displaced CUSTOM primary is auto-unloaded.
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' displaced by reroute and now idle" "test_16: displaced CUSTOM primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' AUTO-UNLOADED" 10
  # the SUPERSEDED prior default_primary TC@1 (now orphaned + idle) is RECLAIMED (freed), not kept warm.
  assert_log "superseded default_primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' now orphaned + idle -> reclaimed" "test_16: orphaned prior default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' reclaimed (freed)" 10
  sleep "${DELAY}"

  # 6) end state: Car@2 is the new default_primary ("default":true). The SIDELINE default_shadow
  #    TC@2 is UNTOUCHED -- still the default shadow ("default":true), kept warm. The superseded
  #    prior default_primary TC@1 is GONE (reclaimed).
  assert_status_ref_eventually "stream=Cam-0" model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_16: Cam-0 now runs promoted primary '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}'"
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION_NEW}"
  assert_status_contains "\"default_for\":[\"model\"]" "test_16: promoted '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' is now the default_primary (default_for [\"model\"])"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"shadow\"]" "test_16: sideline default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' UNTOUCHED (still default_for [\"shadow\"])"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_MODEL_VERSION}"
  assert_status_contains "\"models\":[]" "test_16: superseded prior default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' RECLAIMED (no longer loaded)"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 17: PROMOTE the DEFAULT shadow over a CUSTOM primary -- the promoted (default-shadow) model
#          becomes the new default_primary; the prior default_primary is reclaimed.
#
# The custom/def_s cell of the promote matrix. The stream's primary is a CUSTOM model (Car@1) and
# its shadow is the configured default_shadow (TC@2); a separate default_primary (TC@1) is loaded
# idle. Promoting the shadow makes TC@2 the primary for every stream -> by the promote rule TC@2
# becomes the new default_primary, default_shadow is cleared (its model was the one promoted), and
# the PRIOR default_primary TC@1 (a sideline idle model, not even on these streams) is REVOKED and
# RECLAIMED (freed). The displaced CUSTOM primary Car@1 is AUTO-UNLOADED.
# (See test_14 for the full promote-matrix overview.)
test_17() {
  echo "-----------------------------------------------------------------------"
  echo "[test_17] starting test 17 (PROMOTE default shadow over CUSTOM primary | default_shadow cleared)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it 
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # default primary (TC@1) and default shadow (TC@2) are both established at RUNTIME (def_* flags);
  # comment the config defaults so the API is the sole source. The CUSTOM model here is the PRIMARY
  # (Car@1), loaded WITHOUT a flag -> stays custom; the SHADOW is the default (TC@2).
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) default primary (TC@1, idle) + default shadow (TC@2) + a CUSTOM primary (Car@1).
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_17] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_17: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_17] Requested loading of (DEFAULT Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_17: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded + designated default" 12
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_17] Requested loading of (CUSTOM Primary '${CAR_MODEL}@${CAR_MODEL_VERSION}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_17: custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' loaded (limbo)" 12

  # 4) streams: primary=Car;1 (CUSTOM via metadata) + shadow OMITTED -> default_shadow TC@2.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_17] Added Cam-0 (CUSTOM primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' + DEFAULT shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 0 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_17: Cam-0 = custom primary + default shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_17] Added Cam-1 (CUSTOM primary + DEFAULT shadow)"
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_17: Cam-1 = custom primary + default shadow" 10
  sleep "${DELAY}"

  # 5) PROMOTE the DEFAULT shadow @2 -> primary on ALL streams, clear shadow. By the promote rule
  #    the promoted model BECOMES default_primary (so @2, the old default_shadow, is now the default
  #    PRIMARY), default_shadow is cleared, and the PRIOR default_primary @1 (a separate idle model,
  #    not even on these streams) is REVOKED. The displaced CUSTOM primary Car@1 is AUTO-UNLOADED.
  log_mark
  model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "none"
  echo "[test_17] PROMOTE (ALL): primary -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' (was default shadow) becomes default_primary; prior default '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' revoked"
  assert_log "PROMOTE (IN-PLACE)" "test_17: promote done IN-PLACE (reused bin, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_17: bin REUSED -- gie-id preserved, no reload" 10
  # identity proof: the promoted default shadow keeps its OWN bin (gie 1001) and merely
  # wears the Primary label -- promoting must not instantiate a second copy of it.
  assert_log "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Primary  gie=1001" \
      "test_17: Cam-0 promoted on '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'s ORIGINAL bin (gie 1001 kept)" 12
  assert_log "name=Cam-1  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Primary  gie=1001" \
      "test_17: Cam-1 too -- every stream moved onto that same bin, not a copy" 12
  assert_log "Primary display moved to top row" "test_17: promoted model moved to the primary/top tiler row" 10
  # the promoted model becomes the new default_primary (default_shadow cleared).
  assert_log "DEFAULT promote: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now default_primary" "test_17: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' became the new default_primary; default_shadow cleared" 10
  # the displaced CUSTOM primary Car@1 is auto-unloaded.
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' displaced by reroute and now idle" "test_17: displaced CUSTOM primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' AUTO-UNLOADED" 10
  # the SUPERSEDED prior default_primary TC@1 (a sideline idle model) is RECLAIMED (freed).
  assert_log "superseded default_primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' now orphaned + idle -> reclaimed" "test_17: orphaned prior default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' reclaimed (freed)" 10
  sleep "${DELAY}"

  # 6) end state: streams run primary=TC@2 (no shadow). @2 is now the default ("default":true);
  #    the prior default primary @1 has been RECLAIMED (no longer loaded).
  assert_status_ref_eventually        "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_17: Cam-0 now runs promoted primary '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'"
  assert_status_ref_eventually_absent "stream=Cam-0" shadow "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_17: Cam-0 shadow cleared by the promote"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"model\"]" "test_17: promoted '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now the default_primary (default_for [\"model\"])"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_MODEL_VERSION}"
  assert_status_contains "\"models\":[]" "test_17: superseded prior default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' RECLAIMED (no longer loaded)"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 18: independent A/B GROUPS -> group-local in-place PROMOTE (no model load).
#
# Two INDEPENDENT champion/challenger experiments running side by side, each a cross-version A/B of
# ONE model -- the shadow is a DIFFERENT VERSION of the SAME model as the primary (exactly the
# test_14-16 pattern, just two of them at once on disjoint stream subsets):
#   group-A = Cam-0, Cam-2  -> primary=Person@1, shadow=Person@2   (the Person v1-vs-v2 experiment)
#   group-B = Cam-1, Cam-3  -> primary=Car@1,    shadow=Car@2      (the Car v1-vs-v2 experiment)
#
# Flow:
#   1. Load the two A/B pairs (4 models: Person@1/@2 + Car@1/@2, type-less -> warm into limbo).
#   2. Bring up the two DISJOINT stream groups, each on its OWN same-model version pair.
#   3. PROMOTE each group's challenger (the @2 shadow) to primary, scoped to just that group:
#        model/routing Cam-0,Cam-2  model=Person;2  shadow=none
#        model/routing Cam-1,Cam-3  model=Car;2     shadow=none
#
# What happens (the behaviour under test):
#   Each pair serves ONLY its own group's streams, so a promote scoped to that group FULLY covers
#   the pair -> it is done IN PLACE: the warm @2 shadow bin is RE-LABELED shadow->primary (same gie,
#   no new bin, no engine reload, no reroute), the displaced old primary @1 goes idle and is
#   AUTO-UNLOADED, and the promoted model moves to the top display row. The two experiments are
#   completely independent -- promoting group-A never touches group-B (and vice-versa). This is the
#   multi-experiment case: many parallel cross-version A/B tests on separate stream subsets, each
#   promoted on its own streams without loading or rebuilding any model.
#
# What to validate:
#   - each promote logs "PROMOTE (IN-PLACE)" + "reused existing bins, gie kept, NO reload".
#   - group-A's old primary 'Person@1' and group-B's 'Car@1' each auto-unload.
#   - the OTHER experiment keeps running untouched while one is promoted.
#   - per-stream frame table: dropped = 0 across both promotes.
test_18() {
  echo "-----------------------------------------------------------------------"
  echo "[test_18] starting test 18 (independent A/B groups | group-local IN-PLACE promote | no model load)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               4
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  VERSION_1=1
  VERSION_2=2

  # 3) load TWO A/B pairs (4 custom models, type-less -> warm into limbo until first use)
  load_model "${PERSON_MODEL}" "${VERSION_1}" "${WORST_ENGINE}"
  wait_for_model "${PERSON_MODEL}" "${VERSION_1}"
  load_model "${CAR_MODEL}" "${VERSION_1}" "${WORST_ENGINE}"
  wait_for_model "${CAR_MODEL}" "${VERSION_1}"
  load_model "${PERSON_MODEL}" "${VERSION_2}" "${BEST_ENGINE}"
  wait_for_model "${PERSON_MODEL}" "${VERSION_2}"
  load_model "${CAR_MODEL}" "${VERSION_2}" "${BEST_ENGINE}"
  wait_for_model "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_18] loaded the Person A/B ('${PERSON_MODEL}@${VERSION_1}' vs '${PERSON_MODEL}@${VERSION_2}') and the Car A/B ('${CAR_MODEL}@${VERSION_1}' vs '${CAR_MODEL}@${VERSION_2}')"

  # 4) group-A = the Person v1-vs-v2 experiment: Cam-0, Cam-2  (primary=Person@1, shadow=Person@2)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${PERSON_MODEL}" "${VERSION_1}" "${PERSON_MODEL}" "${VERSION_2}"
  echo "[test_18] added Cam-0 (group-A: primary='${PERSON_MODEL}@${VERSION_1}' shadow='${PERSON_MODEL}@${VERSION_2}')"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${PERSON_MODEL}" "${VERSION_1}" "${PERSON_MODEL}" "${VERSION_2}"
  echo "[test_18] added Cam-2 (group-A: primary='${PERSON_MODEL}@${VERSION_1}' shadow='${PERSON_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 5) group-B = the Car v1-vs-v2 experiment: Cam-1, Cam-3  (primary=Car@1, shadow=Car@2)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${VERSION_1}" "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_18] added Cam-1 (group-B: primary='${CAR_MODEL}@${VERSION_1}' shadow='${CAR_MODEL}@${VERSION_2}')"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "${CAR_MODEL}" "${VERSION_1}" "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_18] added Cam-3 (group-B: primary='${CAR_MODEL}@${VERSION_1}' shadow='${CAR_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 6) PROMOTE group-A in place: the Person@2 challenger shadow->primary on Cam-0,Cam-2; drop shadow.
  #    Scope == group-A's exact stream-set -> in-place (gie kept); the old primary Person@1 auto-unloads.
  log_mark
  model_routing "${PREFIX}0,${PREFIX}2" "${PERSON_MODEL}" "${VERSION_2}" "none"
  echo "[test_18] PROMOTE group-A: '${PERSON_MODEL}@${VERSION_2}' -> primary on Cam-0,Cam-2 (shadow dropped)"
  assert_log "PROMOTE (IN-PLACE)" "test_18: group-A promote done IN-PLACE (no new bin, no model load)" 12
  assert_log "reused existing bins, gie kept, NO reload" "test_18: group-A bin REUSED -- gie preserved, no reload" 10
  assert_log "'${PERSON_MODEL}@${VERSION_1}' displaced by reroute and now idle" "test_18: group-A old primary '${PERSON_MODEL}@${VERSION_1}' auto-unloaded" 10
  # OUTCOME + INDEPENDENCE (whole groups, with gie): BOTH members of group-A (Cam-0 AND Cam-2)
  # promote to the Person@2 challenger with the gie PRESERVED (in-place re-label: Person@2 was
  # shadow gie=1002 -> now primary gie=1002), while BOTH members of group-B (Cam-1, Cam-3) stay
  # untouched on their own Car pair (still primary Car@1 gie=1001) -- the core "independent A/B
  # experiments" claim.
  assert_log "name=Cam-0  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_18: group-A Cam-0 promoted to '${PERSON_MODEL}@${VERSION_2}' (gie 1002 kept, in-place)" 12
  assert_log "name=Cam-2  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_18: group-A Cam-2 ALSO promoted (whole group moved, gie 1002 kept)" 12
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_1}  role=Primary  gie=1001"  "test_18: group-B UNTOUCHED by group-A promote (Cam-1 still '${CAR_MODEL}@${VERSION_1}' gie 1001)" 12
  assert_log "name=Cam-3  model=${CAR_MODEL}@${VERSION_1}  role=Primary  gie=1001"  "test_18: group-B UNTOUCHED by group-A promote (Cam-3 still '${CAR_MODEL}@${VERSION_1}' gie 1001)" 12

  # 7) PROMOTE group-B in place: the Car@2 challenger shadow->primary on Cam-1,Cam-3; drop shadow.
  #    group-A is untouched -- the two A/B experiments are fully independent.
  log_mark
  model_routing "${PREFIX}1,${PREFIX}3" "${CAR_MODEL}" "${VERSION_2}" "none"
  echo "[test_18] PROMOTE group-B: '${CAR_MODEL}@${VERSION_2}' -> primary on Cam-1,Cam-3 (shadow dropped)"
  assert_log "PROMOTE (IN-PLACE)" "test_18: group-B promote done IN-PLACE (group-A untouched)" 12
  assert_log "reused existing bins, gie kept, NO reload" "test_18: group-B bin REUSED -- gie preserved, no reload" 10
  assert_log "'${CAR_MODEL}@${VERSION_1}' displaced by reroute and now idle" "test_18: group-B old primary '${CAR_MODEL}@${VERSION_1}' auto-unloaded" 10
  # OUTCOME (whole group + gie): BOTH members of group-B (Cam-1, Cam-3) move to the Car@2 challenger
  # primary (gie 1003 kept, in-place); group-A (Cam-0, Cam-2) stays on its already-promoted Person@2
  # (gie 1002) -- proving the two promotes are fully independent.
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_2}  role=Primary  gie=1003" "test_18: group-B Cam-1 promoted to '${CAR_MODEL}@${VERSION_2}' (gie 1003 kept, in-place)" 12
  assert_log "name=Cam-3  model=${CAR_MODEL}@${VERSION_2}  role=Primary  gie=1003" "test_18: group-B Cam-3 ALSO promoted (whole group moved, gie 1003 kept)" 12
  assert_log "name=Cam-0  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_18: group-A still on '${PERSON_MODEL}@${VERSION_2}' (gie 1002) after group-B promote" 12
  assert_log "name=Cam-2  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_18: group-A Cam-2 still on '${PERSON_MODEL}@${VERSION_2}' (gie 1002) after group-B promote" 12
  sleep 10

  # 8) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 19: live SWAP of BOTH DEFAULT roles (primary <-> shadow) on ALL streams -- the configured
#          defaults EXCHANGE roles (the swap sibling of test_14's both-default promote).
#
# Setup mirrors test_14: a cross-version A/B of the SAME model -- default primary = Trafficcamnet@1,
# default shadow = Trafficcamnet@2, both established AT RUNTIME via the def_* load flags (load + route default). A SWAP names BOTH
# models and covers every stream, so the two warm bins simply EXCHANGE roles IN PLACE (no new bin,
# no reload, both gie-ids preserved, both keep scoring -- no coverage gap). Because BOTH involved
# models are the configured defaults and the scope is ALL, the DEFAULTS exchange too:
# default_primary <-> default_shadow (so a new stream also follows the swapped baseline).
#
# Flow:
#   1. model/load Trafficcamnet@1 (def_primary flag) + Trafficcamnet@2 (def_shadow flag).
#   2. Bring up Cam-0/Cam-1: primary=Trafficcamnet@1 + shadow=Trafficcamnet@2 (the defaults).
#   3. model/routing model=Trafficcamnet;2 shadow=Trafficcamnet;1 (scope=all) -- flip both roles.
#
# What to validate:
#   - SWAP (IN-PLACE) + reused bins, gie-ids unchanged, no reload (both models keep serving).
#   - DEFAULT swap log: default_primary <-> default_shadow exchanged (@2 primary, @1 shadow).
#   - end state via model/status: streams run primary=@2 + shadow=@1; @2 is the Primary-pool
#     default, @1 the Shadow-pool default (roles exchanged). per-stream dropped = 0.
test_19() {
  echo "-----------------------------------------------------------------------"
  echo "[test_19] starting test 19 (MODEL SWAP | PRIMARY <-> SHADOW | LIVE | ALL streams)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # BOTH defaults are established at RUNTIME via model/load + a model/routing default switch
  # (the load_model def_primary/def_shadow flags; v1 removed set_default_* from model/load) --
  # comment out the config defaults for both roles so the API is the sole source of the defaults.
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the default primary & default shadow models (warm to READY)
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_19] Requested loading of (Default Primary Model '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  # def_primary -> loaded (limbo), then model/routing {"default":{"model":...}} makes it the runtime default model.
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_19: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded into pool + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_19] Requested loading of (Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  # def_shadow -> loaded (limbo), then model/routing {"default":{"shadow":...}} makes it the runtime default shadow.
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_19: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded into pool + designated default" 12

  # 4) add streams on the DEFAULT primary + DEFAULT shadow (a cross-version A/B of Trafficcamnet:
  #    v1 on the top row, v2 on the bottom row), both selected via the runtime defaults.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_19] Added Cam-0 (Default Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + Default Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_19: Cam-0 = default primary + default shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_19] Added Cam-1 (Default Primary Model + Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_19: Cam-1 = default primary + default shadow" 10
  sleep "${DELAY}"

  # 5) SWAP BOTH roles live on ALL streams: primary<->shadow. primary := old shadow (@2),
  #    shadow := old primary (@1). In-place role exchange (no new bin, gie kept, rows crossed).
  #    Both are the configured defaults, so the defaults EXCHANGE too (default_primary<->shadow).
  log_mark
  model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  echo "[test_19] SWAP (ALL): primary='${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' shadow='${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' (roles flipped)"
  assert_log "SWAP (IN-PLACE)" "test_19: swap done IN-PLACE (roles exchanged, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_19: both bins REUSED -- gie-ids preserved, no engine reload" 10
  # identity proof for a swap: BOTH models stay on their own bin and simply trade labels.
  # Rebuilding either one would renumber it and silently land both roles on one artifact.
  assert_log "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Primary  gie=1001" \
      "test_19: Cam-0 primary is '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' on ITS bin (gie 1001, retagged Primary)" 12
  assert_log "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  role=Shadow  gie=1000" \
      "test_19: Cam-0 shadow is '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' on ITS bin (gie 1000, retagged Shadow)" 12
  assert_log "name=Cam-1  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Primary  gie=1001" \
      "test_19: Cam-1 too -- the whole fleet flipped on the same two bins" 12
  # both involved models were defaults -> BOTH slots change hands: the new primary acquires
  # default_primary AND the new shadow acquires default_shadow == a clean EXCHANGE.
  assert_log "DEFAULT swap: default_primary acquired by new primary; default_shadow acquired by new shadow" "test_19: defaults EXCHANGED (both slots changed hands -> default_primary <-> default_shadow)" 10
  sleep "${DELAY}"

  # 6) end state: streams run primary=@2 + shadow=@1; defaults exchanged (status: @2->Primary, @1->Shadow).
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_19: Cam-0 primary swapped to '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'"
  assert_status_ref_eventually "stream=Cam-0" shadow "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_19: Cam-0 shadow swapped to '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"model\"]" "test_19: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is now the default serving model (defaults exchanged)"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"shadow\"]" "test_19: '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' is now the default shadow (defaults exchanged)"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 20: live SWAP of a DEFAULT primary <-> a CUSTOM shadow on ALL streams -- the CUSTOM model
#          ACQUIRES default_primary; the default_shadow is left unchanged.
#
# The def_p / custom cell of the SWAP matrix (the swap sibling of test_15). A cross-version A/B of
# the SAME model -- NO second model (no Car) is involved: default primary = Trafficcamnet@1
# (def_primary flag), shadow = Trafficcamnet@2 (CUSTOM -- loaded WITHOUT a def_* flag, so
# default_shadow stays NONE). A SWAP names both and covers every stream, so the two warm bins simply
# EXCHANGE roles IN PLACE (no new bin, no reload, both gie-ids preserved). SWAP RULE (status follows
# the slot): the model leaving the primary role (the default_primary @1) hands the primary slot --
# and its default status -- to the model now in primary (@2). So:
#   - default_primary := Trafficcamnet@2 (the custom ACQUIRES it -- it now serves primary on everyone).
#   - default_shadow  := UNCHANGED (@2 was not the default_shadow; nothing vacated that slot).
#   - the demoted old default_primary @1 keeps SERVING (now as shadow) but is NO LONGER a default
#     (it gives up its status + idle-reclaim exemption); nothing is reclaimed (it still serves).
#
# Flow:
#   1. model/load Trafficcamnet@1 (def_primary flag) + Trafficcamnet@2 (NO flag, custom).
#   2. Bring up Cam-0/Cam-1: primary OMITTED (-> default @1) + shadow=@2 (custom).
#   3. model/routing model=Trafficcamnet;2 shadow=Trafficcamnet;1 (scope=all) -- flip both roles.
#
# What to validate:
#   - SWAP (IN-PLACE) + reused bins, gie-ids unchanged, no reload.
#   - DEFAULT swap log: "default_primary acquired by new primary" (and NOT the shadow half).
#   - end state via model/status: @2 is now the Primary-pool default ("default":true); the demoted
#     @1 now serves Shadow but is non-default ("default":false). per-stream dropped = 0.
test_20() {
  echo "-----------------------------------------------------------------------"
  echo "[test_20] starting test 20 (SWAP default primary <-> CUSTOM shadow [same model, diff version] | custom ACQUIRES default_primary, default_shadow unchanged)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # default primary is established at RUNTIME (def_primary flag); comment the config defaults so
  # the API is the sole source. The shadow is loaded WITHOUT a flag -> stays CUSTOM (not default).
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the DEFAULT primary (def_primary flag) + a CUSTOM shadow == the SAME model, a DIFFERENT
  #    version (no def_* flag -> limbo; default_shadow stays NONE). No second model is needed.
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_20] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_20: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}"
  echo "[test_20] Requested loading of (CUSTOM Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' -- same model, diff version; NOT a default)"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "'${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' LOADED into limbo" "test_20: custom shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded (limbo, not a default)" 12

  # 4) streams: primary OMITTED (-> resolves to default @1) + custom shadow @2 via metadata.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "" "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "[test_20] Added Cam-0 (DEFAULT primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + CUSTOM shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_20: Cam-0 = default primary + custom shadow (same model, diff version)" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "" "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "[test_20] Added Cam-1 (DEFAULT primary + CUSTOM shadow)"
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_20: Cam-1 = default primary + custom shadow" 10
  sleep "${DELAY}"

  # 5) SWAP both roles live on ALL streams: primary := @2 (old shadow), shadow := @1 (old primary).
  #    In-place exchange. The model leaving primary (@1) was the default_primary -> the new primary
  #    occupant (@2) ACQUIRES default_primary; the shadow slot was not a default's slot ->
  #    default_shadow stays unchanged. @1 demoted to non-default shadow.
  log_mark
  model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  echo "[test_20] SWAP (ALL): primary='${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' shadow='${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' (roles flipped)"
  assert_log "SWAP (IN-PLACE)" "test_20: swap done IN-PLACE (roles exchanged, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_20: both bins REUSED -- gie-ids preserved, no engine reload" 10
  # identity proof: the default/custom mix does not change the mechanics -- both models
  # keep their own bin across the swap; only the role label (and the default flag) moves.
  assert_log "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Primary  gie=1001" \
      "test_20: Cam-0 primary is the CUSTOM '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' on ITS bin (gie 1001, retagged Primary)" 12
  assert_log "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  role=Shadow  gie=1000" \
      "test_20: Cam-0 shadow is the demoted default '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' on ITS bin (gie 1000, retagged Shadow)" 12
  assert_log "name=Cam-1  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Primary  gie=1001" \
      "test_20: Cam-1 too -- the whole fleet flipped on the same two bins" 12
  assert_log "DEFAULT swap: default_primary acquired by new primary" "test_20: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' ACQUIRED default_primary; default_shadow unchanged" 10
  sleep "${DELAY}"

  # 6) end state: streams run primary=@2 + shadow=@1; @2 is the new default_primary ("default":true),
  #    the demoted @1 serves Shadow but is non-default ("default":false).
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_20: Cam-0 primary swapped to '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'"
  assert_status_ref_eventually "stream=Cam-0" shadow "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_20: Cam-0 shadow swapped to '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"serving\":{\"streams\":2}" "test_20: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' now SERVES both streams as the model (swapped in)"
  assert_status_contains "\"default_for\":[\"model\"]" "test_20: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' ACQUIRED the default-model status (default_for [\"model\"])"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_MODEL_VERSION}"
  assert_status_contains "\"serving\":{\"shadow_streams\":2}" "test_20: demoted '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' now serves as Shadow only (still loaded, kept serving)"
  assert_status_not_contains "\"default_for\"" "test_20: demoted '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' GAVE UP default status (no default_for)"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 21: live SWAP of a CUSTOM primary <-> a CUSTOM shadow on ALL streams -- NEITHER model is a
#          default, so the swap touches NO default slot: the configured defaults are left UNTOUCHED.
#
# The custom / custom cell of the SWAP matrix (the swap sibling of test_16's custom/custom promote).
# A separate pair of configured defaults (TC@1 primary / TC@2 shadow) is loaded but idle (no stream
# uses them); the live A/B pair is two CUSTOM models Car@1 / Car@2. A SWAP names both customs and
# covers every stream -> IN-PLACE role exchange (no new bin, no reload, both gie-ids preserved).
# SWAP RULE (status follows the slot): neither swapped model occupied a default's slot, so neither
# default slot changes hands -- default_primary (TC@1) and default_shadow (TC@2) stay EXACTLY as they
# were. Nothing is reclaimed; both customs keep serving in their swapped roles. (Contrast test_16,
# where the custom/custom PROMOTE made the promoted custom the new default_primary + reclaimed the
# superseded sideline default; a SWAP commits to nothing, so it leaves the defaults alone.)
#
# Flow:
#   1. model/load TC@1 (def_primary flag) + TC@2 (def_shadow flag) -> runtime defaults
#      (eager, idle -- no stream uses them). model/load Car@1 + Car@2 (custom, limbo).
#   2. Bring up Cam-0/Cam-1 with primary=Car;1 + shadow=Car;2 (both CUSTOM, via metadata).
#   3. model/routing model=Car;2 shadow=Car;1 (scope=all) -- flip both custom roles.
#
# What to validate:
#   - SWAP (IN-PLACE) + reused bins, gie-ids unchanged, no reload.
#   - DEFAULT swap log: "neither swapped model was a default -- defaults untouched".
#   - both SIDELINE defaults are UNTOUCHED: TC@1 still default_primary ("default":true), TC@2 still
#     default_shadow ("default":true), both still loaded (NOT reclaimed).
#   - end state via model/status: streams run primary=Car@2 + shadow=Car@1; both customs are
#     non-default ("default":false). per-stream dropped = 0.
test_21() {
  echo "-----------------------------------------------------------------------"
  echo "[test_21] starting test 21 (SWAP custom primary <-> custom shadow | NEITHER a default -> configured defaults UNTOUCHED)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # defaults are established at RUNTIME via the def_* load flags (load + route default); comment the config defaults so the API is
  # the sole source. The live A/B pair is loaded WITHOUT a flag -> stays CUSTOM (neither a default).
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3a) configured DEFAULTS (eager, sit idle -- no stream uses them in this case).
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_21] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_21: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_21] Requested loading of (DEFAULT Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_21: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded + designated default" 12

  # 3b) the CUSTOM A/B pair (no def_* flag -> type-less limbo).
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_21] Requested loading of (CUSTOM Primary '${CAR_MODEL}@${CAR_MODEL_VERSION}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_21: custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' loaded (limbo)" 12
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "${BEST_ENGINE}"
  echo "[test_21] Requested loading of (CUSTOM Shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' LOADED into limbo" "test_21: custom shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' loaded (limbo)" 12

  # 4) streams on the CUSTOM A/B pair (the sideline defaults stay idle/unused).
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_21] Added Cam-0 (CUSTOM primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' + CUSTOM shadow '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}')"
  assert_log "stream 0 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_21: Cam-0 = custom primary + custom shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}"
  echo "[test_21] Added Cam-1 (CUSTOM primary + CUSTOM shadow)"
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@${CAR_MODEL_VERSION}  shadow=${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}" "test_21: Cam-1 = custom primary + custom shadow" 10
  sleep "${DELAY}"

  # 5) SWAP both roles live on ALL streams: primary := Car@2 (old shadow), shadow := Car@1 (old
  #    primary). In-place exchange. NEITHER model is a default, so NO default slot changes hands --
  #    the configured sideline defaults TC@1/TC@2 are left untouched, and nothing is reclaimed.
  log_mark
  model_routing "" "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_21] SWAP (ALL): primary='${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' shadow='${CAR_MODEL}@${CAR_MODEL_VERSION}' (custom roles flipped)"
  assert_log "SWAP (IN-PLACE)" "test_21: swap done IN-PLACE (roles exchanged, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_21: both bins REUSED -- gie-ids preserved, no engine reload" 10
  # neither swapped model occupied a default slot -> defaults untouched (the "neither" log branch).
  assert_log "DEFAULT swap: neither swapped model was a default -- defaults untouched" "test_21: swap of two customs left the configured defaults UNTOUCHED" 10
  sleep "${DELAY}"

  # 6) end state: streams run primary=Car@2 + shadow=Car@1 (both non-default); the SIDELINE defaults
  #    TC@1 (default_primary) and TC@2 (default_shadow) are UNTOUCHED -- still default + still loaded.
  assert_status_ref_eventually "stream=Cam-0" model "${CAR_MODEL}" "${CAR_MODEL_VERSION_NEW}" "test_21: Cam-0 primary swapped to '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}'"
  assert_status_ref_eventually "stream=Cam-0" shadow "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "test_21: Cam-0 shadow swapped to '${CAR_MODEL}@${CAR_MODEL_VERSION}'"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"model\"]" "test_21: sideline default_primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' UNTOUCHED (still default_for [\"model\"])"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"default_for\":[\"shadow\"]" "test_21: sideline default_shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' UNTOUCHED (still default_for [\"shadow\"])"
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION_NEW}"
  assert_status_not_contains "\"default_for\"" "test_21: live custom '${CAR_MODEL}@${CAR_MODEL_VERSION_NEW}' is NON-default (no default_for)"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 22: live SWAP of a CUSTOM primary <-> a DEFAULT shadow on ALL streams -- the CUSTOM model
#          ACQUIRES default_shadow; the default_primary is left unchanged.
#
# The custom / def_s cell of the SWAP matrix (the mirror of test_20; the swap sibling of test_17). A
# cross-version A/B of the SAME model -- NO second model (no Car) is involved: default shadow =
# Trafficcamnet@2 (def_shadow flag); primary = Trafficcamnet@1 (CUSTOM -- loaded WITHOUT a
# def_* flag, and default_primary is left NONE, to show it is NOT disturbed by the swap). A SWAP
# names both and covers every stream -> IN-PLACE role exchange. SWAP RULE (status follows the slot):
# the model leaving the shadow role (the default_shadow @2) hands the shadow slot -- and its default
# status -- to the model now in shadow (@1). So:
#   - default_shadow  := Trafficcamnet@1 (the custom ACQUIRES it).
#   - default_primary := UNCHANGED (@1 was not the default_primary; here it stays NONE). The model
#     now in primary (@2) serves but is NON-default -- it did not occupy a default-primary slot.
#   - nothing is reclaimed; both models keep serving in their new roles.
#
# Flow:
#   1. model/load Trafficcamnet@2 (def_shadow flag) + Trafficcamnet@1 (NO flag, custom).
#   2. Bring up Cam-0/Cam-1: primary=@1 (custom) + shadow OMITTED (-> default @2).
#   3. model/routing model=Trafficcamnet;2 shadow=Trafficcamnet;1 (scope=all) -- flip both roles.
#
# What to validate:
#   - SWAP (IN-PLACE) + reused bins, gie-ids unchanged, no reload.
#   - DEFAULT swap log: "default_shadow acquired by new shadow" (and NOT the primary half).
#   - end state via model/status: @1 is now the Shadow-pool default ("default":true); @2 now serves
#     Primary but is non-default ("default":false). per-stream dropped = 0.
test_22() {
  echo "-----------------------------------------------------------------------"
  echo "[test_22] starting test 22 (SWAP CUSTOM primary <-> default shadow [same model, diff version] | custom ACQUIRES default_shadow, default_primary unchanged)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # default shadow is established at RUNTIME (def_shadow flag); comment the config defaults so the
  # API is the sole source. NO default primary is set here (stays NONE -> shows it is not disturbed).
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the DEFAULT shadow (def_shadow flag) + a CUSTOM primary == the SAME model, a DIFFERENT
  #    version (no def_* flag -> limbo; default_primary stays NONE). No second model is needed.
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_22] Requested loading of (DEFAULT Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_22: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}"
  echo "[test_22] Requested loading of (CUSTOM Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' -- same model, diff version; NOT a default)"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "'${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' LOADED into limbo" "test_22: custom primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded (limbo, not a default)" 12

  # 4) streams: custom primary @1 via metadata + shadow OMITTED (-> resolves to default @2).
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "" ""
  echo "[test_22] Added Cam-0 (CUSTOM primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + DEFAULT shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_22: Cam-0 = custom primary + default shadow (same model, diff version)" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "" ""
  echo "[test_22] Added Cam-1 (CUSTOM primary + DEFAULT shadow)"
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_22: Cam-1 = custom primary + default shadow" 10
  sleep "${DELAY}"

  # 5) SWAP both roles live on ALL streams: primary := @2 (old shadow), shadow := @1 (old primary).
  #    In-place exchange. The model leaving shadow (@2) was the default_shadow -> the new shadow
  #    occupant (@1) ACQUIRES default_shadow; the primary slot was NOT a default's slot
  #    (default_primary is NONE) -> default_primary stays unchanged. @2 now serves primary, non-default.
  log_mark
  model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  echo "[test_22] SWAP (ALL): primary='${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' shadow='${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' (roles flipped)"
  assert_log "SWAP (IN-PLACE)" "test_22: swap done IN-PLACE (roles exchanged, no new bin)" 15
  assert_log "reused existing bins, gie kept, NO reload" "test_22: both bins REUSED -- gie-ids preserved, no engine reload" 10
  assert_log "DEFAULT swap: default_shadow acquired by new shadow" "test_22: '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' ACQUIRED default_shadow; default_primary unchanged" 10
  sleep "${DELAY}"

  # 6) end state: streams run primary=@2 + shadow=@1; @1 is the new default_shadow ("default":true),
  #    @2 serves Primary but is non-default ("default":false).
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "test_22: Cam-0 primary swapped to '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'"
  assert_status_ref_eventually "stream=Cam-0" shadow "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_22: Cam-0 shadow swapped to '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_MODEL_VERSION}"
  assert_status_contains "\"serving\":{\"shadow_streams\":2}" "test_22: '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' now serves as Shadow only (swapped in)"
  assert_status_contains "\"default_for\":[\"shadow\"]" "test_22: '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' ACQUIRED the default-shadow status (default_for [\"shadow\"])"
  get_model_status "model=${DEFAULT_MODEL}&version=${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_status_contains "\"serving\":{\"streams\":2}" "test_22: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' now SERVES both streams as the model (swapped in)"
  assert_status_not_contains "\"default_for\"" "test_22: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' is NON-default while serving (default model untouched, no default_for)"
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 23: scoped SWAP of BOTH roles on ONE stream only (the create+warm fallback).
#
# A cross-version A/B of the SAME model (champion/challenger across versions, like test_19): both
# streams run primary=Car@1 + shadow=Car@2. We flip BOTH roles on Cam-0 ONLY.
#
# Flow:
#   1. Load the Car A/B pair (Car@1 + Car@2).
#   2. Bring up Cam-0 AND Cam-1, both running primary=Car@1 + shadow=Car@2.
#   3. Flip BOTH roles on Cam-0 ONLY, leaving Cam-1 untouched:
#         model/routing  Cam-0  model=Car;2  shadow=Car;1   (scope = Cam-0).
#
# What happens (the behaviour under test):
#   Car@1 and Car@2 still serve Cam-1, so the requested scope does NOT fully cover those
#   bins -- an in-place re-label would corrupt Cam-1. The app therefore takes the safe fallback
#   (this is the key difference from test_19): it builds FRESH warm bins for Cam-0 (Car@2 as
#   primary, Car@1 as shadow) make-before-break, switches Cam-0 over with zero frame drop, and
#   leaves Cam-1 exactly as it was. The same model+version now serves DIFFERENT roles on different
#   streams -- Car@1 = primary on Cam-1 but shadow on Cam-0; Car@2 = shadow on Cam-1 but
#   primary on Cam-0.
#
# What to validate:
#   - log shows a scoped reroute that CREATES+WARMS new bins for Cam-0 (NOT an in-place swap).
#   - Cam-1's models/rows are unchanged; only Cam-0 flips.
#   - per-stream frame table: dropped = 0 on both streams across the switch.
#   - the new Cam-0 bins get their own gie-ids (attribution is by model_name@version + role).
#   - a scoped swap NEVER touches the global defaults.
test_23() {
  echo "-----------------------------------------------------------------------"
  echo "[test_23] starting test 23 (scoped SWAP of BOTH roles on ONE stream | same model, diff version)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  VERSION_1=1
  VERSION_2=2

  # 3) load the Car A/B pair (same model, two versions; type-less -> warm into limbo)
  load_model "${CAR_MODEL}" "${VERSION_1}" "${WORST_ENGINE}"
  echo "[test_23] Requested loading of (Car A/B primary '${CAR_MODEL}@${VERSION_1}')"
  wait_for_model "${CAR_MODEL}" "${VERSION_1}"
  load_model "${CAR_MODEL}" "${VERSION_2}" "${BEST_ENGINE}"
  echo "[test_23] Requested loading of (Car A/B shadow '${CAR_MODEL}@${VERSION_2}')"
  wait_for_model "${CAR_MODEL}" "${VERSION_2}"

  # 4) add Cam-0 on the Car A/B: primary=Car@1, shadow=Car@2 (via metadata)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "${VERSION_1}" "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_23] Added Cam-0 (primary='${CAR_MODEL}@${VERSION_1}' shadow='${CAR_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 5) add Cam-1 on the same Car A/B (shares the warm limbo bins)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${VERSION_1}" "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_23] Added Cam-1 (primary='${CAR_MODEL}@${VERSION_1}' shadow='${CAR_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 6) swap BOTH roles live on Cam-0: primary<->shadow (primary:=Car@2, shadow:=Car@1).
  #    (scoped swap; only Cam-0 flips, Cam-1 stays. Because the bins still serve Cam-1, the scope
  #     does NOT fully cover them -> in-place is rejected and the safe create+warm fallback runs.)
  log_mark
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${VERSION_2}" "${CAR_MODEL}" "${VERSION_1}"
  echo "[test_23] swapped BOTH roles live on Cam-0: primary='${CAR_MODEL}@${VERSION_2}' shadow='${CAR_MODEL}@${VERSION_1}'"
  assert_log "IN-PLACE not eligible" "test_23: scope doesn't fully cover the bins -> in-place REJECTED, create+warm fallback used" 12
  assert_log "model/routing applied to 1 stream" "test_23: scoped swap applied to Cam-0 only (Cam-1 untouched)" 12
  # POSITIVE proof of the fallback (not just that in-place was rejected): NEW role-specific bins were
  # actually BUILT for Cam-0. The original Car@1/Car@2 bins were warmed as 'Unknown' (gie
  # 1000/1001) and still serve Cam-1; the scoped swap creates a fresh Car@2-as-PRIMARY bin and a
  # fresh Car@1-as-SHADOW bin for Cam-0 (role disambiguates the two same-name bins).
  assert_log "nvmodelbin created  [Primary/${CAR_MODEL}]  uid=2"   "test_23: create+warm built a NEW Car@2-as-PRIMARY bin for Cam-0 (not in-place)" 12
  assert_log "nvmodelbin created  [Shadow/${CAR_MODEL}]  uid=102"  "test_23: create+warm built a NEW Car@1-as-SHADOW bin for Cam-0" 12
  # Verify the OUTCOME on BOTH streams, with gie-ids -- the same model+version now serves DIFFERENT
  # streams under DIFFERENT gie-ids, the data-level proof that Cam-0 got a fresh bin while Cam-1's
  # bins were untouched (contrast test_19, where the in-place swap PRESERVES gie):
  #   - Cam-0 (flipped, NEW bins): Car@2 primary gie=2, Car@1 shadow gie=102
  #   - Cam-1 (untouched, ORIGINAL limbo bins): Car@1 primary gie=1000, Car@2 shadow gie=1001
  #
  # WHERE EACH NUMBER COMES FROM (two different allocators are in play here, which is
  # the whole point of the assertion -- read them off the CONFIG, not off run order):
  #   * A bin created into a POOL takes the gie-unique-id its nvinfer config DECLARES,
  #     so a downstream secondary can name it in operate-on-gie-id straight from config.
  #     configs/config_infer_primary_car.txt declares 2  ->  Car as PRIMARY = 2, and the
  #     SHADOW lane adds MM_SHADOW_GIE_OFFSET (100)      ->  Car as SHADOW  = 102.
  #     (config_infer_primary.txt declares 1 and config_infer_primary_person.txt 3, so
  #     Trafficcamnet=1/101 and Person=3/103. The three are >0 and not exactly 100 apart,
  #     so the stage keeps configured ids rather than falling back to the allocator.)
  #   * A bin created by model/load lands in LIMBO first, which has its own 1000+ range
  #     (next_limbo_gie) and keeps that id when it is later promoted into a pool. Cam-1's
  #     two bins were loaded that way, hence 1000/1001 -- load order, not config.
  # So: change a config's gie-unique-id and the first pair moves; change the ORDER of the
  # model/load calls above and the second pair moves. A mismatch fails loudly with the gie
  # in the message.
  assert_log "name=Cam-0  model=${CAR_MODEL}@${VERSION_2}  role=Primary  gie=2"   "test_23: Cam-0 primary flipped to '${CAR_MODEL}@${VERSION_2}' on a NEW bin (gie=2, its config's declared id, distinct from Cam-1's gie=1001)" 12
  assert_log "name=Cam-0  model=${CAR_MODEL}@${VERSION_1}  role=Shadow  gie=102"  "test_23: Cam-0 shadow flipped to '${CAR_MODEL}@${VERSION_1}' on a NEW bin (gie=102, declared 2 + shadow offset, distinct from Cam-1's gie=1000)" 12
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_1}  role=Primary  gie=1000" "test_23: Cam-1 UNTOUCHED (still primary '${CAR_MODEL}@${VERSION_1}', original gie=1000)" 12
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_2}  role=Shadow  gie=1001" "test_23: Cam-1 UNTOUCHED (still shadow '${CAR_MODEL}@${VERSION_2}', original gie=1001)" 12
  # a SCOPED swap is the create+warm fallback and NEVER touches the global defaults -- the default
  # re-assignment lives only on the scope-ALL in-place path. Prove it: no "DEFAULT swap" log fires.
  assert_log_absent "DEFAULT swap" "test_23: scoped swap leaves the GLOBAL defaults untouched (no default re-assignment)" 3
  sleep 10

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 24: independent A/B GROUPS -> group-local in-place SWAP (no model load).
#
# Identical setup to test_18, but here we SWAP instead of PROMOTE. Two INDEPENDENT cross-version A/B
# experiments, each on ONE model (shadow = a DIFFERENT VERSION of the same model as the primary):
#   group-A = Cam-0, Cam-2  -> primary=Person@1, shadow=Person@2   (the Person v1-vs-v2 experiment)
#   group-B = Cam-1, Cam-3  -> primary=Car@1,    shadow=Car@2      (the Car v1-vs-v2 experiment)
#
# Flow:
#   1. Load the two A/B pairs (4 models: Person@1/@2 + Car@1/@2, type-less -> warm into limbo).
#   2. Bring up the two DISJOINT stream groups, each on its OWN same-model version pair.
#   3. SWAP BOTH roles within each group, scoped to just that group's streams:
#        model/routing Cam-0,Cam-2  model=Person;2  shadow=Person;1
#        model/routing Cam-1,Cam-3  model=Car;2     shadow=Car;1
#
# What happens (the behaviour under test):
#   Unlike test_18's promote, a SWAP keeps BOTH versions (primary<->shadow exchanged) rather than
#   dropping one. Because each pair serves ONLY its own group's streams, the scope FULLY covers the
#   pair, so the swap is done IN PLACE: the two warm bins simply EXCHANGE roles (both gie-ids
#   preserved, no new bin, no reload, no reroute), and the display rows are crossed so each new
#   primary moves to the top row. Nothing is dropped -- both versions keep scoring every frame. The
#   two experiments are fully independent: swapping group-A never touches group-B. (test_18 = group
#   promote; this = group swap -- the difference is shadow=<old primary> here vs shadow=none there.)
#
# What to validate:
#   - each swap logs "SWAP (IN-PLACE)" + "reused existing bins, gie kept, NO reload".
#   - provenance: Cam-0/Cam-2 primary -> Person@2, shadow -> Person@1;
#                 Cam-1/Cam-3 primary -> Car@2,    shadow -> Car@1.
#   - both versions in each pair stay loaded (nothing auto-unloads -- this is a swap, not a promote).
#   - per-stream frame table: dropped = 0 across both swaps.
test_24() {
  echo "-----------------------------------------------------------------------"
  echo "[test_24] starting test 24 (independent A/B groups | group-local IN-PLACE swap | no model load)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2

  # 2) start the app with the current config copies
  start_app || return

  VERSION_1=1
  VERSION_2=2

  # 3) load the two A/B pairs (4 models, type-less -> warm into limbo until first use)
  load_model "${PERSON_MODEL}" "${VERSION_1}" "${WORST_ENGINE}"
  wait_for_model "${PERSON_MODEL}" "${VERSION_1}"
  load_model "${CAR_MODEL}" "${VERSION_1}" "${WORST_ENGINE}"
  wait_for_model "${CAR_MODEL}" "${VERSION_1}"
  load_model "${PERSON_MODEL}" "${VERSION_2}" "${BEST_ENGINE}"
  wait_for_model "${PERSON_MODEL}" "${VERSION_2}"
  load_model "${CAR_MODEL}" "${VERSION_2}" "${BEST_ENGINE}"
  wait_for_model "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_24] loaded the Person A/B ('${PERSON_MODEL}@${VERSION_1}' vs '${PERSON_MODEL}@${VERSION_2}') and the Car A/B ('${CAR_MODEL}@${VERSION_1}' vs '${CAR_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 4) group-A = the Person v1-vs-v2 experiment: Cam-0, Cam-2  (primary=Person@1, shadow=Person@2)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${PERSON_MODEL}" "${VERSION_1}" "${PERSON_MODEL}" "${VERSION_2}"
  echo "[test_24] added Cam-0 (group-A: primary='${PERSON_MODEL}@${VERSION_1}' shadow='${PERSON_MODEL}@${VERSION_2}')"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${PERSON_MODEL}" "${VERSION_1}" "${PERSON_MODEL}" "${VERSION_2}"
  echo "[test_24] added Cam-2 (group-A: primary='${PERSON_MODEL}@${VERSION_1}' shadow='${PERSON_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 5) group-B = the Car v1-vs-v2 experiment: Cam-1, Cam-3  (primary=Car@1, shadow=Car@2)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${VERSION_1}" "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_24] added Cam-1 (group-B: primary='${CAR_MODEL}@${VERSION_1}' shadow='${CAR_MODEL}@${VERSION_2}')"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "${CAR_MODEL}" "${VERSION_1}" "${CAR_MODEL}" "${VERSION_2}"
  echo "[test_24] added Cam-3 (group-B: primary='${CAR_MODEL}@${VERSION_1}' shadow='${CAR_MODEL}@${VERSION_2}')"
  sleep "${DELAY}"

  # 6) SWAP group-A in place: primary<->shadow on Cam-0,Cam-2 (both versions kept).
  #    Scope == group-A's exact stream-set -> in-place exchange (both gies kept).
  log_mark
  model_routing "${PREFIX}0,${PREFIX}2" "${PERSON_MODEL}" "${VERSION_2}" "${PERSON_MODEL}" "${VERSION_1}"
  echo "[test_24] SWAP group-A: primary='${PERSON_MODEL}@${VERSION_2}' shadow='${PERSON_MODEL}@${VERSION_1}' on Cam-0,Cam-2"
  assert_log "SWAP (IN-PLACE)" "test_24: group-A swap done IN-PLACE (roles exchanged, no new bin, no model load)" 12
  assert_log "reused existing bins, gie kept, NO reload" "test_24: group-A both bins REUSED -- gie-ids preserved, no reload" 10
  # OUTCOME (whole group + gie + both roles): BOTH members of group-A (Cam-0, Cam-2) exchange roles,
  # with BOTH gie-ids PRESERVED (in-place: Person@2 stays gie=1002, Person@1 stays gie=1000 -- they
  # just cross roles). group-B (Cam-1) untouched (still primary Car@1 gie=1001).
  assert_log "name=Cam-0  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_24: group-A Cam-0 primary swapped to '${PERSON_MODEL}@${VERSION_2}' (gie 1002 kept)" 12
  assert_log "name=Cam-0  model=${PERSON_MODEL}@${VERSION_1}  role=Shadow  gie=1000"  "test_24: group-A Cam-0 shadow swapped to '${PERSON_MODEL}@${VERSION_1}' (gie 1000 kept)" 12
  assert_log "name=Cam-2  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_24: group-A Cam-2 ALSO swapped (whole group), primary '${PERSON_MODEL}@${VERSION_2}' gie 1002" 12
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_1}  role=Primary  gie=1001"  "test_24: group-B UNTOUCHED by group-A swap (Cam-1 still '${CAR_MODEL}@${VERSION_1}' gie 1001)" 12
  # USE-CASE distinction vs test_18 (promote): a SWAP keeps BOTH versions -- group-A's old primary
  # '${PERSON_MODEL}@${VERSION_1}' becomes the shadow, so it is NOT auto-unloaded.
  assert_log_absent "'${PERSON_MODEL}@${VERSION_1}' displaced by reroute and now idle" "test_24: group-A swap KEEPS both versions ('${PERSON_MODEL}@${VERSION_1}' NOT auto-unloaded, unlike a promote)" 3

  # 7) SWAP group-B in place: primary<->shadow on Cam-1,Cam-3 (group-A untouched).
  log_mark
  model_routing "${PREFIX}1,${PREFIX}3" "${CAR_MODEL}" "${VERSION_2}" "${CAR_MODEL}" "${VERSION_1}"
  echo "[test_24] SWAP group-B: primary='${CAR_MODEL}@${VERSION_2}' shadow='${CAR_MODEL}@${VERSION_1}' on Cam-1,Cam-3"
  assert_log "SWAP (IN-PLACE)" "test_24: group-B swap done IN-PLACE (group-A untouched)" 12
  assert_log "reused existing bins, gie kept, NO reload" "test_24: group-B both bins REUSED -- gie-ids preserved, no reload" 10
  # OUTCOME (whole group + gie + both roles): BOTH members of group-B (Cam-1, Cam-3) exchange roles
  # with both gie-ids preserved (Car@2 gie=1003, Car@1 gie=1001); group-A stays on its already-swapped
  # Person@2 (gie 1002) -- the two swaps are independent. As a swap (not promote) BOTH versions stay
  # loaded -- group-B's old primary '${CAR_MODEL}@${VERSION_1}' becomes the shadow, NOT auto-unloaded.
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_2}  role=Primary  gie=1003" "test_24: group-B Cam-1 primary swapped to '${CAR_MODEL}@${VERSION_2}' (gie 1003 kept)" 12
  assert_log "name=Cam-1  model=${CAR_MODEL}@${VERSION_1}  role=Shadow  gie=1001"   "test_24: group-B Cam-1 shadow swapped to '${CAR_MODEL}@${VERSION_1}' (gie 1001 kept)" 12
  assert_log "name=Cam-3  model=${CAR_MODEL}@${VERSION_2}  role=Primary  gie=1003" "test_24: group-B Cam-3 ALSO swapped (whole group), primary '${CAR_MODEL}@${VERSION_2}' gie 1003" 12
  assert_log "name=Cam-0  model=${PERSON_MODEL}@${VERSION_2}  role=Primary  gie=1002" "test_24: group-A UNTOUCHED by group-B swap (Cam-0 still primary '${PERSON_MODEL}@${VERSION_2}' gie 1002)" 12
  assert_log_absent "'${CAR_MODEL}@${VERSION_1}' displaced by reroute and now idle" "test_24: group-B swap KEEPS both versions ('${CAR_MODEL}@${VERSION_1}' NOT auto-unloaded)" 3
  sleep 10

  # 8) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 25: dedup-on-idle DEMOTE-TO-LIMBO policy (config compact-idle-models-across-pools=1; default ON).
#
# Validates the default-ON compaction policy: when a custom model bin drains to ZERO streams and is
# the ONLY warm copy of that model@version, it is DEMOTED to limbo (role-AGNOSTIC -- shown as role
# "unknown" in model/status) instead of being kept role-tagged in its pool. From limbo the SAME warm
# engine is re-PROMOTED into EITHER role on the next use -- NO new bin, NO engine re-warm. That is the
# memory/latency win: one warm copy serves primary OR shadow, reused across drains AND across roles.
#
# A persistent DEFAULT-primary stream (Cam-0) is kept up throughout as a baseline, so the pipeline
# never drains fully to zero while we cycle the custom model on the other streams. The lifecycle is
# exercised on ONE custom model (Car@1) -- there is no A/B pair here; the point is one model's
# idle-bin reuse, and especially that the limbo copy can come back in a DIFFERENT role:
#   primary-use -> drain -> DEMOTE-to-limbo (role "unknown") -> re-PROMOTE-from-limbo (SAME role)
#   -> drain -> demote again -> re-PROMOTE-from-limbo into the OTHER role (cross-role, as SHADOW).
# Zero-drop reuse relies on the ModelBin mux/demux recreate fix.
#
# Flow:
#   1. compact-idle-models-across-pools=1; load custom Car@1 (limbo).
#   2. Cam-0 = DEFAULT primary -> baseline stream (stays up the whole test).
#   3. Cam-1 primary=Car@1 -> flows. Remove (drain to zero) -> Car@1 DEMOTED to limbo (role "unknown").
#   4. Re-add Cam-2 primary=Car@1 -> re-PROMOTED from limbo (SAME role, reused), flows; remove
#      -> DEMOTED to limbo again (the cycle is repeatable).
#   5. Cross-role reuse: Cam-3 = default primary + shadow=Car@1 -> the SAME limbo copy is PROMOTED
#      into the SHADOW role with NO new shadow bin built (the cross-role win the flag enables).
#
# Validate: DEMOTED-to-limbo (role "unknown") on each drain; PROMOTED-from-limbo on each reuse; NO new
#   shadow bin on the cross-role reuse; sustained provenance flow each time; 0 frame loss; no crash.
# NOTE: provenance lines are versioned -- "name=Cam-X  model=<name>@<version>  role=<R>"; model/status
#   entries split the key as "model":"<name>","version":"<v>" (the limbo entry has "role":"unknown").
test_25() {
  echo "-----------------------------------------------------------------------"
  echo "[test_25] starting test 25 (dedup DEMOTE-TO-LIMBO | drain->demote->re-promote + cross-role reuse)"

  # 1) config + ENABLE the feature
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2     # 2 rows: primary top, shadow bottom
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  cfg_set multimodel           compact-idle-models-across-pools 1   # <-- feature under test (config-file key; also a GObject property; default ON)

  # 2) start the app
  start_app || return

  # 3) load the custom model into limbo
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_25] loaded custom '${CAR_MODEL}@${CAR_MODEL_VERSION}' (limbo)"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_25: custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' loaded (limbo)" 12

  # 4) Cam-0 = DEFAULT primary -> baseline stream (kept up the whole test; keeps the pipeline alive)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_25] added Cam-0 (DEFAULT primary baseline)"
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 "test_25: Cam-0 flows on the DEFAULT primary (baseline stream -- keeps the pipeline alive across the demote/re-promote cycles)" 15
  sleep "${DELAY}"

  # 5) custom PRIMARY (Car@1) flows on Cam-1
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_25] added Cam-1 (primary='${CAR_MODEL}@${CAR_MODEL_VERSION}')"
  assert_count_min "name=Cam-1  model=${CAR_MODEL}@${CAR_MODEL_VERSION}" 2 "test_25: Cam-1 flows on custom primary '${CAR_MODEL}'" 15
  sleep "${DELAY}"

  # 6) drain Cam-1 to zero -> the lone idle Car@1 bin is DEMOTED to limbo (role "unknown")
  log_mark
  remove_stream "${PREFIX}1" "${STREAM}"
  echo "[test_25] removed Cam-1 (drain to zero -> expect demote)"
  assert_log "DEMOTED to limbo" "test_25: lone-idle '${CAR_MODEL}@${CAR_MODEL_VERSION}' DEMOTED to limbo on drain (not freed, not kept role-tagged)" 12
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION}"
  assert_status_contains "\"state\":\"ready\",\"version\":\"${CAR_MODEL_VERSION}\"" "test_25: '${CAR_MODEL}@${CAR_MODEL_VERSION}' kept warm in LIMBO (still loaded + ready in model/status, not freed; limbo demote proven by the DEMOTED-to-limbo log)"
  sleep "${DELAY}"

  # 7) re-promote from limbo (SAME role), flows; then drain again (repeatable)
  log_mark
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_25] re-added Cam-2 (primary='${CAR_MODEL}@${CAR_MODEL_VERSION}') -> expect promote-from-limbo"
  assert_log "PROMOTED '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_25: '${CAR_MODEL}@${CAR_MODEL_VERSION}' re-PROMOTED from limbo (reused, no rebuild)" 12
  assert_count_min "name=Cam-2  model=${CAR_MODEL}@${CAR_MODEL_VERSION}" 2 "test_25: Cam-2 flows on reused-from-limbo '${CAR_MODEL}@${CAR_MODEL_VERSION}'" 15
  sleep "${DELAY}"

  # 8) drain Cam-2 to zero -> Car@1 DEMOTED to limbo again (repeatable)
  log_mark
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_25] removed Cam-2 (drain -> expect demote again)"
  assert_log "DEMOTED to limbo" "test_25: '${CAR_MODEL}@${CAR_MODEL_VERSION}' demoted to limbo again (repeatable cycle)" 12
  sleep "${DELAY}"

  # 9) CROSS-ROLE reuse: default primary + shadow=Car@1 -> the SAME limbo Car@1 is promoted from
  #    limbo as SHADOW, with NO new shadow bin built (the win vs keeping it role-tagged in primary).
  log_mark
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "" "" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_25] added Cam-3 (default primary + shadow='${CAR_MODEL}@${CAR_MODEL_VERSION}') -> cross-role reuse"
  assert_log "PROMOTED '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_25: '${CAR_MODEL}@${CAR_MODEL_VERSION}' promoted from limbo for the SHADOW role (cross-role reuse, no rebuild)" 12
  assert_log_absent "nvmodelbin created  [Shadow/${CAR_MODEL}]" "test_25: cross-role reuse did NOT build a new shadow bin (reused the limbo copy)" 3
  assert_count_min "name=Cam-3  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Shadow" 2 "test_25: Cam-3 shadow='${CAR_MODEL}@${CAR_MODEL_VERSION}' flows (reused from limbo)" 15
  sleep "${DELAY}"

  # 10) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 26: dedup-on-idle policy OFF (compact-idle-models-across-pools=0) -- the INVERSE of test_25.
#
# The compaction feature is ON by default (test_25 covers the ON path). This test DISABLES it to pin
# the OFF behaviour and prove the flag actually gates the policy: when a custom model bin drains to
# ZERO streams it is KEPT ROLE-TAGGED in its pool (NOT demoted to limbo). It stays warm and reusable,
# but ONLY by the SAME role -- a cross-role request cannot see it and must build a fresh bin. This is
# exactly the cost the default-ON feature removes.
#
# Same SHAPE as test_25 (a persistent DEFAULT-primary Cam-0 baseline keeps the pipeline alive while we
# cycle one custom model, Car@1, on the other streams), but with INVERTED expectations at every step:
#   1. compact-idle-models-across-pools=0; load custom Car@1 (limbo).
#   2. Cam-0 = DEFAULT primary -> baseline stream (stays up the whole test).
#   3. Cam-1 primary=Car@1 flows; remove (drain to zero) -> Car@1 KEPT role-tagged as Primary
#      (NO demote-to-limbo; model/status still shows "role":"Primary", NOT "unknown").
#   4. Re-add Cam-2 primary=Car@1 -> reused from the SAME Primary pool (NOT promoted from limbo); flows.
#   5. Cross-role: Cam-3 = default primary + shadow=Car@1 -> the idle copy is stranded role-tagged in
#      the PRIMARY pool (never returned to limbo), so it is NOT role-agnostic and the SHADOW role
#      CANNOT reuse it (resolve_role reports "NOT loaded" -> shadow falls back to none). With the
#      feature ON (test_25) that same warm copy IS reused cross-role from limbo.
#
# Validate: NO "DEMOTED to limbo" on any drain; the idle copy stays "role":"Primary" (not "unknown");
#   same-role reuse still flows (NOT from limbo); cross-role reuse does NOT happen; 0 frame loss; no crash.
# NOTE: provenance lines are versioned -- "name=Cam-X  model=<name>@<version>  role=<R>"; model/status
#   entries split the key as "model":"<name>","version":"<v>".
test_26() {
  echo "-----------------------------------------------------------------------"
  echo "[test_26] starting test 26 (dedup policy OFF | keep-role-tagged, no cross-role reuse)"

  # 1) same shape as test_25 but with the feature DISABLED
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         1     # allow a shadow role (needed by the cross-role step)
  cfg_set multimodel           per-model-batch-size  2
  cfg_set multimodel           compact-idle-models-across-pools 0   # <-- feature DISABLED (prove OFF behaviour)

  # 2) start the app
  start_app || return

  # 3) load the custom model into limbo
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_26] loaded custom '${CAR_MODEL}@${CAR_MODEL_VERSION}' (limbo)"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_26: custom '${CAR_MODEL}@${CAR_MODEL_VERSION}' loaded (limbo)" 12

  # 4) Cam-0 = DEFAULT primary -> baseline stream (kept up the whole test; keeps the pipeline alive)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_26] added Cam-0 (DEFAULT primary baseline)"
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 "test_26: Cam-0 flows on the DEFAULT primary (baseline stream)" 15
  sleep "${DELAY}"

  # 5) custom PRIMARY (Car@1) flows on Cam-1, then drain-to-zero -> KEEP role-tagged (NO demote)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_26] added Cam-1 (primary='${CAR_MODEL}@${CAR_MODEL_VERSION}')"
  assert_count_min "name=Cam-1  model=${CAR_MODEL}@${CAR_MODEL_VERSION}" 2 "test_26: Cam-1 flows on custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}'" 15
  log_mark
  remove_stream "${PREFIX}1" "${STREAM}"
  echo "[test_26] removed Cam-1 (drain to zero -> expect KEEP role-tagged, NO demote-to-limbo)"
  assert_log_absent "DEMOTED to limbo" "test_26: lone-idle '${CAR_MODEL}@${CAR_MODEL_VERSION}' KEPT role-tagged on drain (feature off -> NOT demoted to limbo)" 3
  get_model_status "model=${CAR_MODEL}&version=${CAR_MODEL_VERSION}"
  # INVERSE of test_25: the idle copy stays role-tagged as Primary (NOT demoted to role "unknown").
  assert_status_contains "\"state\":\"ready\",\"version\":\"${CAR_MODEL_VERSION}\"" "test_26: '${CAR_MODEL}@${CAR_MODEL_VERSION}' kept warm + loaded (v1 status carries no role field; the NOT-demoted-to-limbo proof is the absent DEMOTED log above)"
  sleep "${DELAY}"

  # 6) re-add SAME role on Cam-2 -> reused from the PRIMARY pool directly (NOT promoted from limbo)
  log_mark
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_26] re-added Cam-2 (primary='${CAR_MODEL}@${CAR_MODEL_VERSION}') -> expect same-pool reuse (no limbo)"
  assert_log_absent "PROMOTED '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_26: same-role reuse came from the Primary pool, NOT limbo (no promote-from-limbo)" 3
  assert_count_min "name=Cam-2  model=${CAR_MODEL}@${CAR_MODEL_VERSION}" 2 "test_26: Cam-2 flows on same-pool-reused '${CAR_MODEL}@${CAR_MODEL_VERSION}'" 15
  log_mark
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_26] removed Cam-2 (drain again -> still NO demote)"
  assert_log_absent "DEMOTED to limbo" "test_26: '${CAR_MODEL}@${CAR_MODEL_VERSION}' still kept role-tagged on second drain (no demote)" 3

  # 7) CROSS-ROLE: Cam-3 = default primary + shadow=Car@1. With the feature OFF, Car@1 is stranded
  #    role-tagged-idle in the PRIMARY pool (never returned to limbo), so it is NOT role-agnostic and
  #    the SHADOW role CANNOT reuse it: resolve_role reports "NOT loaded" and shadow falls back to
  #    none. This is EXACTLY the cost the default-ON feature removes (test_25 reuses the very same warm
  #    copy cross-role from limbo). We assert the OFF cost directly:
  #      - the cross-role reuse does NOT happen (no promote-from-limbo for Car@1), and
  #      - the warm primary copy is invisible to the shadow request (resolve_role "NOT loaded").
  log_mark
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}" "" "" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_26] added Cam-3 (default primary + shadow='${CAR_MODEL}@${CAR_MODEL_VERSION}') -> expect NO cross-pool reuse"
  assert_log "shadow model '${CAR_MODEL}@${CAR_MODEL_VERSION}' is NOT loaded" "test_26: OFF -> idle role-tagged primary copy is NOT reusable as shadow (no cross-pool reuse; the cost default-ON avoids)" 12
  assert_log_absent "PROMOTED '${CAR_MODEL}@${CAR_MODEL_VERSION}'" "test_26: OFF -> NO promote-from-limbo for the shadow role (cross-role reuse happens only with the feature ON, cf. test_25)" 3
  sleep 5

  # 8) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 27: model-sharding=0 -- streams beyond the per-model batch cap PASS THROUGH (no infer).
#
# Flow:
#   1. Start with max-batch-size=4 and per-model-batch-size=2 AND model-sharding=0.
#   2. Add Cam-0 and Cam-1 on the default primary -> fills the one bin (2 streams, cap=2).
#   3. Add Cam-2 on the same primary -> cap full + sharding disabled -> Cam-2 is routed to the
#      combined display mux WITHOUT inference (passthrough): frames keep flowing downstream,
#      just with no detection metadata -- exactly like an over-batch stream in a stock nvinfer
#      graph. NOT dropped, NOT sharded.
#
# What to validate:
#   - Cam-0 and Cam-1 both wire and flow WITH inference (provenance shows the default model).
#   - Cam-2 takes the passthrough path: the "passthrough" decision log + "PASSTHROUGH (no inference)"
#     marker are logged, Cam-2 IS wired, and frames flow through the bypass (TAP[pass-out]).
#   - Cam-2 is NOT inferred: output provenance is tagged model=passthrough, never the default model,
#     and the downstream source_id/pad_index restore uses Cam-2's real id.
#   - Cam-0 and Cam-1 keep flowing after Cam-2 joins (no regression on the inferred streams).
#   - No frame loss / gaps / criticals on the inferred streams (passthrough is excluded from
#     frame-accounting since it carries no inference, so it must not trip a false loss alarm).
test_27() {
  echo "-----------------------------------------------------------------------"
  echo "[test_27] starting test 27 (model-sharding=0: over-cap streams PASS THROUGH, no inference)"

  # 1) config: 4-wide total batch but per-model cap = 2; sharding DISABLED
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2
  cfg_set multimodel           model-sharding        0    # <-- key: disable sharding

  # 2) start the app
  start_app || return

  # 3) Add Cam-0 (fills slot 0 of the default primary bin)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_27] added Cam-0"
  sleep "${DELAY}"
  assert_log "stream 0 WIRED" "test_27: Cam-0 wired on default primary (slot 0)" 10
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 "test_27: Cam-0 FLOWS sustained (sharding=0 baseline)" 15

  # 4) Add Cam-1 (fills slot 1 -- per-model cap now FULL)
  log_mark
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_27] added Cam-1 (fills per-model cap=2)"
  sleep "${DELAY}"
  assert_log "stream 1 WIRED" "test_27: Cam-1 wired on default primary (slot 1, cap now full)" 10
  assert_count_min "name=Cam-1  model=${DEFAULT_MODEL}" 2 "test_27: Cam-1 FLOWS sustained (cap full, still sharding=0)" 15

  # 5) Add Cam-2: per-model cap full + model-sharding=0 -> PASSTHROUGH (no inference)
  log_mark
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_27] added Cam-2 (cap full + sharding disabled -> expect PASSTHROUGH, no inference)"
  sleep "${DELAY}"
  # The plugin logs the over-cap passthrough decision + the passthrough wiring marker
  assert_log "routing stream to output WITHOUT inference (passthrough)" "test_27: Cam-2 over-cap -> passthrough decision logged (cap full, sharding=0)" 10
  assert_log "stream 2 PASSTHROUGH (no inference)" "test_27: Cam-2 wired via the no-infer passthrough lane (tee -> queue -> display mux)" 10
  # Cam-2 IS wired (it flows -- not dropped)
  assert_log "stream 2 WIRED" "test_27: Cam-2 IS wired (passthrough flows downstream, not dropped)" 10
  # Frames actually flow through the bypass lane to the combined display mux
  assert_log "TAP[pass-out ] stream 2: 100 buffer" "test_27: Cam-2 frames FLOW downstream through the passthrough lane (un-inferred)" 20
  # Cam-2 is NOT inferred -> provenance exists only to preserve downstream source identity.
  assert_log "name=Cam-2  model=passthrough@" "test_27: Cam-2 carries passthrough provenance for downstream source-id restore" 15
  assert_log "SOURCE-ID restore stream=2 name=Cam-2 role=Primary frame=(source_id=2,pad_index=2)" \
    "test_27: Cam-2 passthrough leaves nvmodelmux with downstream source_id/pad_index restored" 15
  assert_log_absent "name=Cam-2  model=${DEFAULT_MODEL}" "test_27: Cam-2 is NOT inferred by the default model (passthrough only)" 3
  # Cam-0 and Cam-1 keep flowing WITH inference after Cam-2 joins
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 "test_27: Cam-0 still FLOWS+infers after Cam-2 passthrough (no regression on inferred streams)" 10
  assert_count_min "name=Cam-1  model=${DEFAULT_MODEL}" 2 "test_27: Cam-1 still FLOWS+infers after Cam-2 passthrough (no regression on inferred streams)" 10
  sleep 10

  # 6) remove all streams
  remove_stream "${PREFIX}0" "${STREAM}"
  remove_stream "${PREFIX}1" "${STREAM}"
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_27] removed Cam-0, Cam-1, Cam-2"

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 28: OPTIONAL default model -- no default configured => no-model streams PASS THROUGH.
#
# The default primary model is OPTIONAL. With NO default configured: nothing is preloaded at
# init (zero VRAM), and any stream that resolves to "no model" is PASSED THROUGH the display
# mux WITHOUT inference (frames flow, no detections) -- the SAME modular passthrough lane used
# by the model-sharding=0 over-cap case (test_27). A user can then load a custom model on
# demand and attach streams to it; those infer while the no-model streams keep passing through.
#
# Edge cases captured:
#   1. The plugin BUILDS + runs with NO default primary (the old mandatory-default error is gone).
#   2. Stream with NO model requested + no default            -> PASSTHROUGH (no inference).
#   3. Stream requesting a NON-EXISTENT (name,version) + no default -> PASSTHROUGH (+ "NOT loaded" warn).
#      (A model is addressed by (name, version); a versionless request is a SEPARATE edge -- it is
#      rejected for lacking a version and likewise falls back to passthrough.)
#   4. MIXED: after model/load, a stream on the custom model INFERS while the no-model streams
#      keep passing through -- inferred + un-inferred coexist in the same combined out_mux.
#   5. No-model passthrough frames still carry output provenance so downstream source_id/pad_index
#      can be restored, but they are not attributed to any inference model.
#   6. No frame loss / gaps / criticals on the inferred stream; passthrough streams are excluded
#      from frame-accounting (no false loss alarm).
test_28() {
  echo "-----------------------------------------------------------------------"
  echo "[test_28] starting test 28 (OPTIONAL default: no-model streams pass through, mixed with infer)"

  # 1) config: NO default primary model at all (remove the seeded primary-model line).
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         0
  comment_cfg multimodel primary-model          # <-- the key: omit the default primary model

  # 2) start the app -- it MUST come up with NO default model (the old mandatory-default build hard-errored here)
  start_app || return

  # 3) Stream with NO model + no default -> PASSTHROUGH (no inference)
  log_mark
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_28] added Cam-0 (no model requested, no default -> expect PASSTHROUGH)"
  sleep "${DELAY}"
  assert_log "stream 0 PASSTHROUGH (no inference)" "test_28: Cam-0 (no model, no default) takes the passthrough lane" 10
  assert_log "stream 0 WIRED" "test_28: Cam-0 IS wired (passthrough flows, not rejected)" 10
  assert_log "TAP[pass-out ] stream 0: 100 buffer" "test_28: Cam-0 frames FLOW downstream through the passthrough lane (un-inferred)" 20
  assert_log "name=Cam-0  model=passthrough@" "test_28: Cam-0 carries passthrough provenance for downstream source-id restore" 15
  assert_log "SOURCE-ID restore stream=0 name=Cam-0 role=Primary frame=(source_id=0,pad_index=0)" \
    "test_28: Cam-0 passthrough leaves nvmodelmux with downstream source_id/pad_index restored" 15
  assert_log_absent "name=Cam-0  model=${DEFAULT_MODEL}" "test_28: Cam-0 is NOT inferred by the default model (passthrough only)" 3

  # 4) Stream requesting a NON-EXISTENT (name,version) + no default -> PASSTHROUGH (+ NOT-loaded warn)
  log_mark
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "Ghost" "1"
  echo "[test_28] added Cam-1 (requests non-existent 'Ghost@1', no default -> PASSTHROUGH)"
  sleep "${DELAY}"
  assert_log "model 'Ghost@1' is NOT loaded" "test_28: Cam-1's non-existent model 'Ghost@1' (name+version) is reported NOT loaded" 10
  assert_log "stream 1 PASSTHROUGH (no inference)" "test_28: Cam-1 (unresolvable model, no default) takes the passthrough lane" 10
  assert_log "TAP[pass-out ] stream 1: 100 buffer" "test_28: Cam-1 frames FLOW downstream through the passthrough lane (un-inferred)" 20
  assert_log "name=Cam-1  model=passthrough@" "test_28: Cam-1 carries passthrough provenance for downstream source-id restore" 15
  assert_log "SOURCE-ID restore stream=1 name=Cam-1 role=Primary frame=(source_id=1,pad_index=1)" \
    "test_28: Cam-1 passthrough leaves nvmodelmux with downstream source_id/pad_index restored" 15
  assert_log_absent "name=Cam-1  model=Ghost" "test_28: Cam-1 is NOT attributed to the unresolved Ghost model (passthrough only)" 3

  # 5) MIXED: load a custom model, attach Cam-2 to it -> INFERS, while Cam-0/Cam-1 pass through
  log_mark
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_28] loaded custom '${CAR_MODEL}'"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_28] added Cam-2 (primary='${CAR_MODEL}') -> expect INFERENCE"
  assert_log "stream 2 WIRED" "test_28: Cam-2 wired on the custom primary '${CAR_MODEL}'" 10
  assert_count_min "name=Cam-2  model=${CAR_MODEL}" 2 "test_28: Cam-2 FLOWS+INFERS on the loaded custom model (coexists with passthrough)" 15
  # the no-model streams keep passing through after the inferred stream joins
  assert_count_min "TAP[pass-out ] stream 0:" 1 "test_28: Cam-0 still passes through after a model+inferred stream were added (coexistence)" 20
  sleep 10

  # 6) remove all streams
  remove_stream "${PREFIX}0" "${STREAM}"
  remove_stream "${PREFIX}1" "${STREAM}"
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_28] removed Cam-0, Cam-1, Cam-2"

  # 7) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 29: PERF METRICS (attach-perf-metric) -- when enabled, model/status reports per-model
#          (aggregate) and per-stream(per-role) fps + inference latency. The default-OFF path
#          (no perf fields, zero overhead) is implicitly covered by every other test's model/status.
#
# Flow: enable attach-perf-metric=1 + a short perf-metric-interval-sec=2, bring up a both-default
# A/B (Trafficcamnet@1 primary + Trafficcamnet@2 shadow) on Cam-0/Cam-1, let inference run for >2
# measurement windows, then GET model/status and assert the perf fields are present AND non-zero for
# the serving stream (proves the measurement actually computes, not just emits 0.00).
#
# What to validate:
#   - models[].instances[] carry "fps" + "infer_latency_ms"; each stream's model/shadow REFS
#     carry per-role "fps" + "latency_ms", and the stream carries "throughput_fps".
#   - the serving stream's model-ref fps AND shadow-ref fps are > 0 (not the 0.0 sentinel;
#     jsoncpp re-serializes the plugin's 0.00 as 0.0).
#   - unified A/B output restores primary to source_id=N and shadow to source_id=N+max_streams
#     (and pad_index follows), so downstream PERF and tiler rows use stable camera-indexed lanes.
test_29() {
  echo "-----------------------------------------------------------------------"
  echo "[test_29] starting test 29 (perf metrics | attach-perf-metric -> fps + latency in model/status)"

  # 1) config: enable perf metrics with a short window; runtime defaults via the def_* load flags (load + route default) (like test_19)
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  cfg_set multimodel           attach-perf-metric    1
  cfg_set multimodel           perf-metric-interval-sec 2
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app
  start_app || return

  # 3) load the default primary + default shadow (a both-default A/B so primary AND shadow are measured)
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_29] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_29: default primary loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_29] Requested loading of (DEFAULT Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_29: default shadow loaded + designated default" 12

  # 4) bring up two streams on the default primary + default shadow
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_29] added Cam-0 (default primary + default shadow)"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" "test_29: Cam-0 = default primary + default shadow" 10
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_29] added Cam-1 (default primary + default shadow)"
  sleep "${DELAY}"

  # 5) confirm BOTH roles are actually inferring (so the perf numbers are real)
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  role=Primary" 2 "test_29: Cam-0 primary inferring" 15
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}  role=Shadow" 2 "test_29: Cam-0 shadow inferring" 15
  assert_log "SOURCE-ID restore stream=0 name=Cam-0 role=Primary frame=(source_id=0,pad_index=0)" \
    "test_29: unified primary lane restores downstream source_id/pad_index to the camera id" 15
  assert_log "SOURCE-ID restore stream=0 name=Cam-0 role=Shadow frame=(source_id=2,pad_index=2)" \
    "test_29: unified shadow lane restores downstream source_id/pad_index to camera id + max_streams" 15

  # 6) let inference run for >2 measurement windows (interval=2s) so a non-zero snapshot is computed
  sleep 10

  # 7) perf fields PRESENT in model/status (flag ON): models[] aggregate + streams[] per-role
  get_model_status
  assert_status_contains "\"fps\":"                "test_29: models[] carry aggregate \"fps\""
  assert_status_contains "\"infer_latency_ms\":"   "test_29: models[] instance rows carry \"infer_latency_ms\""
  assert_status_contains "\"model\":{\"fps\":"      "test_29: streams[] serving-model refs carry per-role \"fps\""
  assert_status_contains "\"latency_ms\":"         "test_29: streams[] role refs carry per-role \"latency_ms\""
  assert_status_contains "\"shadow\":{\"fps\":"     "test_29: streams[] shadow refs carry per-role \"fps\""
  assert_status_contains "\"throughput_fps\":"     "test_29: streams[] carry \"throughput_fps\" (perf-on only)"

  # 8) measurement actually computes -> the serving stream's fps is NON-ZERO (not the 0.00 sentinel)
  get_model_status "stream=Cam-0"
  assert_status_not_contains "\"model\":{\"fps\":0.0,"  "test_29: Cam-0 serving-model fps is measured > 0 (ref not \"fps\":0.0)"
  assert_status_not_contains "\"shadow\":{\"fps\":0.0," "test_29: Cam-0 shadow fps is measured > 0 (ref not \"fps\":0.0)"
  sleep "${DELAY}"

  # 9) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 30: BUFFER-COPY-MODE (buffer-copy-mode = 0 none / 1 shadow / 2 primary / 3 both) -- a copy
#          lane splices nvvideoconvert(disable-passthrough=1) BEFORE its queue so the frame is
#          deep-copied and the upstream decoder buffer is freed at fan-out; the passthrough role
#          stays zero-copy (no element). Default = 1 (shadow copies, primary passthrough).
#
# Flow: for each mode 0..3, run a FRESH app (run_multimodel truncates the log) with a both-default
# A/B (Trafficcamnet@1 primary + Trafficcamnet@2 shadow) on Cam-0, then assert (a) the per-lane
# install token BUFFER-COPY[<role>]=ON/OFF, and (b) on a copy lane the runtime BUFFER-LEVEL proof
# BUFFER-COPY-VERIFY[<role>]=COPIED -- MM_DEBUG probes compare the GstBuffer + NvBufSurface pointer
# entering vs leaving the converter, so a NEW buffer/memory proves a real copy, not just the flag.
#
# What to validate:
#   - install token: mode 0 -> neither lane copies; 1 -> shadow only; 2 -> primary only; 3 -> both.
#   - every copy lane ACTUALLY copies at runtime (BUFFER-COPY-VERIFY[...]=COPIED), not merely wired.
#   - no frame loss / gaps / criticals -- the copy lane does not disturb the data flow.
test_30() {
  echo "-----------------------------------------------------------------------"
  echo "[test_30] starting test 30 (buffer-copy-mode 0/1/2/3 -> per-lane nvvideoconvert copy)"

  # base config: same both-default A/B as test_29 (defaults loaded at runtime via the def_* load flags (load + route default))
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # one app cycle per mode; assert the expected copy state on BOTH lanes of Cam-0.
  # (each role logs exactly one BUFFER-COPY[role]=ON|OFF at attach, so presence is unambiguous.)
  _bcm_case() {   # <mode> <primary-expect ON|OFF> <shadow-expect ON|OFF>
    local mode="$1" pexp="$2" sexp="$3"
    cfg_set multimodel buffer-copy-mode "${mode}"
    start_app || return 1
    load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
    wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
    load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
    wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
    log_mark                                   # scope asserts to the attach that follows
    add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
    echo "[test_30] mode=${mode}: expect Primary=${pexp} Shadow=${sexp}"
    assert_log "BUFFER-COPY[Primary]=${pexp}" "test_30: mode=${mode} -> primary lane copy=${pexp}" 12
    assert_log "BUFFER-COPY[Shadow]=${sexp}"  "test_30: mode=${mode} -> shadow lane copy=${sexp}" 12
    # runtime BUFFER-LEVEL proof (MM_DEBUG verify probes, on by default): compare the buffer at the
    # lane head (tee src) vs the queue input -- a copy lane yields a NEW buffer (COPIED), a
    # passthrough lane forwards the SAME buffer (SAME). Asserts BOTH directions per role.
    if [ "${pexp}" = "ON" ]; then
      assert_log "BUFFER-COPY-VERIFY[Primary]=COPIED" "test_30: mode=${mode} -> primary lane COPIED (new buffer tee->queue)" 15
    else
      assert_log "BUFFER-COPY-VERIFY[Primary]=SAME"   "test_30: mode=${mode} -> primary lane SAME (zero-copy tee->queue)" 15
    fi
    if [ "${sexp}" = "ON" ]; then
      assert_log "BUFFER-COPY-VERIFY[Shadow]=COPIED"  "test_30: mode=${mode} -> shadow lane COPIED (new buffer tee->queue)" 15
    else
      assert_log "BUFFER-COPY-VERIFY[Shadow]=SAME"    "test_30: mode=${mode} -> shadow lane SAME (zero-copy tee->queue)" 15
    fi
    sleep "${DELAY}"
    stop_app
  }

  _bcm_case 0 OFF OFF     # none    -> neither lane copies
  _bcm_case 1 OFF ON      # shadow  -> default: only the shadow lane copies
  _bcm_case 2 ON  OFF     # primary -> only the primary lane copies
  _bcm_case 3 ON  ON      # both    -> both lanes copy
  echo "-----------------------------------------------------------------------"
}

# Test 31: A/B SWAP with ASYMMETRIC branch rates -- anti-stall guard for the swap-display
#          row-cross safety timeout.
#
# Why this exists:
#   An in-place A/B swap that has to CROSS the display rows blocks BOTH out_q src pads and
#   only runs the cross once BOTH have blocked. If one branch delivers much slower (or is
#   momentarily idle), the fast branch could stay blocked waiting for the slow one. The
#   swap-display safety timeout aborts the cross if both pads do not block within a bounded
#   window, so frames keep flowing instead of stalling forever. This test makes the two
#   branches deliberately ASYMMETRIC (slow INT8-nocalib primary vs fast FP32 shadow) to
#   stress that window, then PING-PONGS the swap RACE_REPS times asserting BOTH rows
#   KEEP DELIVERING after every flip (one swap can miss the race window; N cannot).
#
# What to validate:
#   - the swap is applied in place (no new bin, gie kept);
#   - AFTER the swap, both streams CONTINUE to flow (assert_count_min on the post-swap
#     frameacct lines) -- i.e. the row-cross did NOT wedge a branch;
#   - default gates: zero frame loss + no CRITICAL/crash.
test_31() {
  echo "-----------------------------------------------------------------------"
  echo "[test_31] starting test 31 (A/B SWAP with asymmetric branch rates -- anti-stall)"

  # 1) config: 2 streams, unified A/B (top=primary row, bottom=shadow row), defaults via API
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  start_app || return

  # 2) defaults: SLOW primary (int8-nocalib WORST) + FAST shadow (fp32 BEST) -> asymmetric rates
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_31: default primary loaded" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_31: default shadow loaded" 12

  # 3) two A/B streams (default primary + default shadow)
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  sleep "${DELAY}"

  # 4) SWAP PING-PONG LOOP: flip both roles on ALL streams, N times. Every flip forces
  #    the display-row CROSS, and whether the anti-stall window is stressed depends on
  #    the branches' momentary phase -- ONE swap can miss the race; N swaps make the
  #    verdict real. Each iteration is checked SILENTLY (both rows must deliver post-
  #    swap frames within the window), AGGREGATED into one pass/fail like test_3.
  local reps="${RACE_REPS}"
  local flow_to=20
  local fails=0 first_fail="" si sj ok pv sv
  echo "[test_31] STRESS: ${reps} asymmetric A/B swap (row-cross) ping-pong cycles ..."
  for (( si=1; si<=reps; si++ )); do
    # alternate direction: odd = fast-fp32 primary, even = back to slow-int8 primary
    if (( si % 2 )); then pv="${DEFAULT_SHADOW_MODEL_VERSION}"; sv="${DEFAULT_MODEL_VERSION}";
    else                  pv="${DEFAULT_MODEL_VERSION}";        sv="${DEFAULT_SHADOW_MODEL_VERSION}"; fi
    log_mark
    model_routing "" "${DEFAULT_MODEL}" "${pv}" "${DEFAULT_MODEL}" "${sv}"
    # swap applied in place + BOTH rows still delivering in THIS swap's window
    ok=0
    for (( sj=0; sj<flow_to; sj++ )); do
      if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF "SWAP (IN-PLACE)" \
         && [ "$(tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -cF -- "name=Cam-0  model=${DEFAULT_MODEL}")" -ge 2 ] \
         && [ "$(tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -cF -- "name=Cam-1  model=${DEFAULT_MODEL}")" -ge 2 ]; then
        ok=1; break
      fi
      sleep 1
    done
    if [ "${ok}" -ne 1 ]; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${si}"
      echo "[test_31] WARN swap ${si}: in-place swap + post-swap flow on BOTH rows not seen within ${flow_to}s (row-cross stall?)"
    fi
    sleep 1
  done
  if [ "${fails}" -eq 0 ]; then
    _pass "test_31: all ${reps} asymmetric swaps applied IN-PLACE and both rows kept flowing (row-cross anti-stall)"
  else
    _fail "test_31: ${fails}/${reps} swap cycles stalled or missed the in-place marker (first: swap ${first_fail}) -- row-cross anti-stall regression"
  fi
  sleep "${DELAY}"

  # 6) stop (default gates assert zero loss + no crash)
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 32: TEARDOWN-UNDER-RECONFIG -- stop the app WHILE dynamic reconfig is in flight.
#
# Why this exists:
#   The dynamic-routing machinery schedules work on the main loop / streaming thread
#   (hitless cutover idles + drain timeouts, swap-display block probes + their safety
#   timeout, promote idles). If the pipeline tears down (READY->NULL) while any of those
#   are still queued, a callback that derefs the freed bin is a use-after-free. This test
#   issues a reconfig BURST and then stops the app IMMEDIATELY (no settle), so teardown
#   races the in-flight async work.
#
# What to validate (mostly via the DEFAULT gates):
#   - no CRITICAL / assertion / "Segmentation fault" / "(core dumped)" in the log
#     (assert_no_gst_critical) -- a teardown UAF would surface here as a crash;
#   - the burst was actually issued (so the scenario really ran).
#
# NOTE: races are timing-dependent -- ONE clean stop can pass by luck, so the whole
#   start -> burst -> immediate-stop cycle repeats RACE_REPS times (silent per-cycle
#   checks, ONE aggregated verdict). For fully deterministic detection also run against
#   an AddressSanitizer/ThreadSanitizer build of the plugin.
test_32() {
  echo "-----------------------------------------------------------------------"
  echo "[test_32] starting test 32 (teardown WHILE reconfig in flight -- UAF tripwire)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  local reps="${RACE_REPS}"        # teardown-under-reconfig cycles (global RACE_REPS)
  local tc tj ok fails=0 first_fail=""
  echo "[test_32] STRESS: ${reps} start -> reconfig-burst -> immediate-teardown cycles ..."
  for (( tc=1; tc<=reps; tc++ )); do
    echo "[test_32] ===== cycle ${tc}/${reps} ====="
    log_mark
    start_app || return

    load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
    wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
    load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
    wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"

    add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
    sleep "${DELAY}"
    add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
    sleep "${DELAY}"

    # BURST: arm async reconfig work (swap -> cutover + drain timeout + swap-display block
    # probes/timeout), then keep churning with NO settle so callbacks are still queued.
    model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
    add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
    model_routing "" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
    remove_stream "${PREFIX}0" "${STREAM}"
    echo "[test_32] c${tc}: reconfig burst issued -- tearing down IMMEDIATELY (no settle)"

    # silent per-cycle check: the burst really ran in THIS cycle's window
    ok=0
    for (( tj=0; tj<8; tj++ )); do
      if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF "MODEL ROUTING scope="; then ok=1; break; fi
      sleep 1
    done
    if [ "${ok}" -ne 1 ]; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${tc}"
      echo "[test_32] WARN cycle ${tc}: reconfig burst not observed before teardown"
    fi

    # IMMEDIATE teardown (no settle) -> races the queued cutover/swap-display/promote work
    stop_app
    # silent per-cycle check: the app must exit on SIGINT (SIGKILL fallback = teardown hang)
    if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF "did not exit on SIGINT"; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${tc}"
      echo "[test_32] WARN cycle ${tc}: app did NOT exit on SIGINT (teardown hang)"
    fi
  done
  if [ "${fails}" -eq 0 ]; then
    _pass "test_32: all ${reps} teardown-under-reconfig cycles issued the burst and exited cleanly on SIGINT"
  else
    _fail "test_32: ${fails} failure(s) across ${reps} teardown-under-reconfig cycles (first: cycle ${first_fail}) -- teardown race regression"
  fi
  echo "-----------------------------------------------------------------------"
}

# Test 33: downstream source-id/pad-index contract after partial remove.
#
# Why this exists:
#   The combined display nvstreammux rewrites NvDsFrameMeta.source_id/pad_index to its
#   internal sink-pad index. Those display slots are allocated in attach order, not camera id.
#   Downstream DeepStream PERF, "Active sources", tiler placement, and msgbroker source ids
#   consume those fields. Internal nvmodelmux provenance can be correct while downstream is
#   wrong, which is exactly why the partial-remove 0-fps regression escaped.
#
# Flow:
#   1. Non-unified primary-only, 4 streams.
#   2. Add streams out of camera-name order: Cam-1, Cam-3, Cam-0, Cam-2. nvmultiurisrcbin
#      assigns numeric source ids by stream lifetime, so the test must not assume Cam-N == N.
#   3. Remove Cam-2 and Cam-3, leaving Cam-0/Cam-1 still flowing from mismatched slots.
#   4. Assert the frame metadata leaving nvmodelmux has source_id AND pad_index restored to
#      that frame's real upstream source id.
test_33() {
  echo "-----------------------------------------------------------------------"
  echo "[test_33] starting test 33 (downstream source-id restore after partial remove)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4

  start_app || return

  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_33] added Cam-1 first"
  sleep "${DELAY}"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
  echo "[test_33] added Cam-3 second"
  sleep "${DELAY}"
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_33] added Cam-0 third"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_33] added Cam-2 fourth"
  sleep "${DELAY}"
  assert_log "stream 2 WIRED" "test_33: all out-of-order streams wired before partial remove" 10

  log_mark
  remove_stream "${PREFIX}2" "${STREAM}"
  remove_stream "${PREFIX}3" "${STREAM}"
  echo "[test_33] removed Cam-2 and Cam-3; Cam-0/Cam-1 should keep flowing with restored ids"
  sleep "${DELAY}"

  assert_restore_consistent "Cam-0" \
    "test_33: Cam-0 survives partial remove and leaves with source_id/pad_index restored to its upstream id" 20
  assert_restore_consistent "Cam-1" \
    "test_33: Cam-1 survives partial remove and leaves with source_id/pad_index restored to its upstream id" 20
  assert_count_min "name=Cam-0" 2 "test_33: Cam-0 still flows after partial remove (not data loss)" 20
  assert_count_min "name=Cam-1" 2 "test_33: Cam-1 still flows after partial remove (not data loss)" 20

  remove_stream "${PREFIX}0" "${STREAM}"
  remove_stream "${PREFIX}1" "${STREAM}"
  echo "[test_33] removed remaining Cam-0 and Cam-1"

  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 34: control-plane race tripwire -- duplicate add/remove and add-cancel-readd.
#
# Why this exists:
#   Stream add/remove is marshalled from REST/nvmultiurisrcbin messages onto the main loop,
#   while the actual graph work touches request pads, display columns, per-model pad indexes,
#   and pending/cancelled stream tables. The fragile cases are not the normal sequential
#   add/remove paths; they are duplicate requests and remove-while-attach races that can leave
#   stale pending entries, leaked display/model slots, or half-published streams.
#
# Flow:
#   1. Fire duplicate add requests for Cam-0 plus a concurrent Cam-1 add. Both intended streams
#      must still flow; the duplicate may be rejected by either nvmultiurisrcbin or nvmodelmux.
#   2. Fire duplicate removes while those streams are live, then re-add them. Re-add flow proves
#      the duplicate remove did not leak pending state or slots.
#   3. Fire add+duplicate-remove for Cam-2 while attach may still be pending, then re-add Cam-2.
#      This specifically exercises streams_pending / streams_cancelled.
#   4. STRESS LOOP (RACE_REPS, default 10): concurrently remove two live streams while
#      adding a new one. nvmultiurisrcbin recycles the freed source ids, so the new
#      stream's BIND can overtake the old incarnation's REMOVE (BIND -> REMOVE -> ADD) --
#      the ordering that once wiped the un-consumed bind (nameless attach, lost per-stream
#      model metadata) and that races the deferred demux-pad release. The ordering is
#      timing-dependent, so ONE burst can pass by luck; N bursts make the verdict real.
#      Each cycle asserts the new stream flows UNDER ITS NAME (a nameless attach fails
#      this) and the survivor keeps flowing; one aggregated pass/fail, like test_3.
test_34() {
  echo "-----------------------------------------------------------------------"
  echo "[test_34] starting test 34 (control-plane race: duplicate add/remove + add-cancel-readd)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4

  start_app || return

  # 1) duplicate add while attaching. Exact rejection path is intentionally not asserted;
  # the invariant is that one Cam-0 and one Cam-1 survive and flow.
  log_mark
  ( add_stream "${PREFIX}0" "Cam-0" "${STREAM}" ) & local p0=$!
  ( add_stream "${PREFIX}0" "Cam-0" "${STREAM}" ) & local p0dup=$!
  ( add_stream "${PREFIX}1" "Cam-1" "${STREAM}" ) & local p1=$!
  wait "${p0}" "${p0dup}" "${p1}" 2>/dev/null || true
  echo "[test_34] duplicate Cam-0 add + concurrent Cam-1 add completed"
  sleep "${DELAY}"
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 \
    "test_34: Cam-0 flows after duplicate add race (no half-attach/stuck pending)" 20
  assert_count_min "name=Cam-1  model=${DEFAULT_MODEL}" 2 \
    "test_34: Cam-1 flows alongside duplicate add race" 20

  # 2) duplicate remove while live, then re-add. This catches leaked columns / stale pending flags.
  log_mark
  ( remove_stream "${PREFIX}0" "${STREAM}" ) & local r0=$!
  ( remove_stream "${PREFIX}0" "${STREAM}" ) & local r0dup=$!
  ( remove_stream "${PREFIX}1" "${STREAM}" ) & local r1=$!
  wait "${r0}" "${r0dup}" "${r1}" 2>/dev/null || true
  echo "[test_34] duplicate Cam-0 remove + Cam-1 remove completed"
  sleep "${DELAY}"

  log_mark
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_34] re-added Cam-0 and Cam-1 after duplicate remove race"
  sleep "${DELAY}"
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 \
    "test_34: Cam-0 re-add flows after duplicate remove race" 20
  assert_count_min "name=Cam-1  model=${DEFAULT_MODEL}" 2 \
    "test_34: Cam-1 re-add flows after duplicate remove race" 20
  assert_restore_consistent "Cam-0" \
    "test_34: Cam-0 re-add still restores downstream source_id/pad_index" 20
  assert_restore_consistent "Cam-1" \
    "test_34: Cam-1 re-add still restores downstream source_id/pad_index" 20

  # 3) remove-while-attach for the same camera, then re-add. The cleanup remove is deliberately
  # allowed to be a no-op if the racing removes already won.
  log_mark
  ( add_stream "${PREFIX}2" "Cam-2" "${STREAM}" ) & local a2=$!
  sleep 0.05
  ( remove_stream "${PREFIX}2" "${STREAM}" ) & local r2=$!
  ( remove_stream "${PREFIX}2" "${STREAM}" ) & local r2dup=$!
  wait "${a2}" "${r2}" "${r2dup}" 2>/dev/null || true
  echo "[test_34] Cam-2 add-cancel burst completed"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"        # idempotent cleanup if the add won the race
  sleep 1

  log_mark
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_34] re-added Cam-2 after add-cancel burst"
  sleep "${DELAY}"
  assert_count_min "name=Cam-2  model=${DEFAULT_MODEL}" 2 \
    "test_34: Cam-2 re-add flows after add/remove-pending cancellation race" 20
  assert_restore_consistent "Cam-2" \
    "test_34: Cam-2 re-add still restores downstream source_id/pad_index" 20

  # 4) STRESS LOOP: N x concurrent remove-two-while-adding-one bursts over recycled
  #    source ids. Cam-2 stays live as the survivor throughout; Cam-0/Cam-1 seed the
  #    first burst. Each cycle:
  #      burst:  ( remove A ) & ( remove B ) & ( add NEW ) &   -- max id-recycle jitter
  #      check:  NEW flows UNDER ITS NAME (nameless attach = wiped bind = FAIL),
  #              survivor Cam-2 still flows
  #      reseed: add two fresh streams, remove NEW -> next cycle races on their ids.
  #    Checked SILENTLY per cycle (scoped via log_mark), AGGREGATED into one verdict.
  local reps="${RACE_REPS}"        # race bursts (global RACE_REPS)
  local flow_to=20
  local fails=0 first_fail="" ci cj ok
  local seedA="${PREFIX}0" seedB="${PREFIX}1"
  echo "[test_34] STRESS: ${reps} concurrent remove+add (id-recycle) race bursts ..."
  for (( ci=1; ci<=reps; ci++ )); do
    log_mark                                   # scope this cycle's flow checks
    ( remove_stream "${seedA}" "${STREAM}" ) & local cr0=$!
    ( remove_stream "${seedB}" "${STREAM}" ) & local cr1=$!
    ( add_stream "${PREFIX}r${ci}" "Cam-R${ci}" "${STREAM}" ) & local ca3=$!
    wait "${cr0}" "${cr1}" "${ca3}" 2>/dev/null || true
    sleep 1

    # poll for the NEW stream flowing under its NAME: catches wire-then-stall AND the
    # bind-wipe race (a nameless 'camera=?' attach never logs its name= provenance).
    ok=0
    for (( cj=0; cj<flow_to; cj++ )); do
      if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null \
           | grep -qF -- "name=Cam-R${ci}  model=${DEFAULT_MODEL}"; then ok=1; break; fi
      sleep 1
    done
    if [ "${ok}" -ne 1 ]; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${ci}"
      echo "[test_34] WARN cycle ${ci}: Cam-R${ci} did NOT flow (named) within ${flow_to}s after the race burst"
    fi
    # survivor must still be flowing in THIS cycle's window
    if ! tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null \
         | grep -qF -- "name=Cam-2  model=${DEFAULT_MODEL}"; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${ci}"
      echo "[test_34] WARN cycle ${ci}: survivor Cam-2 stopped flowing during the race burst"
    fi

    # reseed for the next cycle: two fresh streams to race-remove, retire this NEW one.
    # Their adds re-use the source ids just freed -> every cycle races on recycled ids.
    if [ "${ci}" -lt "${reps}" ]; then
      add_stream "${PREFIX}a${ci}" "Cam-A${ci}" "${STREAM}"
      add_stream "${PREFIX}b${ci}" "Cam-B${ci}" "${STREAM}"
      sleep 1
      remove_stream "${PREFIX}r${ci}" "${STREAM}"
      seedA="${PREFIX}a${ci}"; seedB="${PREFIX}b${ci}"
      sleep 1
    fi
  done
  if [ "${fails}" -eq 0 ]; then
    _pass "test_34: all ${reps} concurrent remove+add race bursts flowed NAMED (id-recycle bind + pad-release races)"
  else
    _fail "test_34: ${fails} failure(s) across ${reps} race bursts (first: cycle ${first_fail}) -- id-recycle race regression (nameless attach / wire-then-stall)"
  fi

  remove_stream "${PREFIX}2" "${STREAM}"
  remove_stream "${PREFIX}r${reps}" "${STREAM}"
  echo "[test_34] cleanup removes issued"

  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 35: multi-model config-bound custom primary -- lazy-load/reuse anti-stall.
#
# Why this exists:
#   This is a pure nvmodelmux multi-model regression test, not a backend-heterogeneous
#   nvinfer+nvinferserver/GDINO test. It uses the local sample nvinfer configs because the
#   bug being guarded is in dynamic lane wiring/sticky-event seeding, which is independent of
#   infer backend.
#
#   Some streams stay on the configured default primary while other streams are bound by
#   [stream-model-*] to a custom primary that is NOT the default. The custom model is
#   lazy-loaded on first use, assigned a limbo/high gie-id path, promoted into the Primary pool,
#   and then each bound stream gets a dynamically wired model lane. A regression in sticky-event
#   seeding or per-lane attach can leave one custom-bound stream with entered/buffered growing
#   while a sibling stream still infers, so generic model-level checks are not enough.
#
# Flow:
#   1. Default primary = Trafficcamnet; Cam-1/Cam-2 are config-bound to Car;1.
#   2. Add Cam-0..Cam-3 back-to-back: Cam-0/Cam-3 use the default, Cam-1/Cam-2 use custom Car.
#   3. Assert EVERY stream produces sustained provenance on the expected model.
#   4. Remove all streams and re-add the same mix once more to cover reused/refill lanes.
#      RACE_REPS sets the cycle count (global race-repetition knob).
test_35() {
  echo "-----------------------------------------------------------------------"
  echo "[test_35] starting test 35 (multi-model lazy-bound custom primary anti-stall)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4
  cfg_set multimodel           primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  comment_cfg    multimodel   shadow-model

  # Make the custom-bound primary distinct from the default model instance while still using
  # the existing sample config. The stream-model bindings are the key part: Cam-1/Cam-2 select
  # Car from config, not from REST metadata.
  cfg_set model-${CAR_MODEL}-${CAR_MODEL_VERSION} engine "${BEST_ENGINE}"
  cfg_set stream-model-cam-1   primary-model       "${CAR_MODEL};${CAR_MODEL_VERSION}"
  cfg_set stream-model-cam-2   primary-model       "${CAR_MODEL};${CAR_MODEL_VERSION}"

  start_app || return

  local reps="${RACE_REPS}" k
  for (( k=1; k<=reps; k++ )); do
    log_mark
    echo "[test_35] cycle ${k}/${reps}: add default/custom/default multi-model batch"
    add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
    add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
    add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
    add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
    sleep "${DELAY}"

    if [ "${k}" -eq 1 ]; then
      assert_log "lazy-load primary model '${CAR_MODEL}@${CAR_MODEL_VERSION}' on stream-add" \
        "test_35: config-bound custom primary '${CAR_MODEL}@${CAR_MODEL_VERSION}' lazy-loaded only when Cam-1/Cam-2 arrive" 20
    fi

    assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  role=Primary" 2 \
      "test_35: Cam-0 default primary infers sustained in multi-model batch cycle ${k}" 30
    assert_count_min "name=Cam-1  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" 2 \
      "test_35: Cam-1 config-bound custom primary infers sustained cycle ${k} -- no custom-lane stall" 30
    assert_count_min "name=Cam-2  model=${CAR_MODEL}@${CAR_MODEL_VERSION}  role=Primary" 2 \
      "test_35: Cam-2 config-bound custom primary infers sustained cycle ${k} -- no sibling hidden stall" 30
    assert_count_min "name=Cam-3  model=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  role=Primary" 2 \
      "test_35: Cam-3 default primary infers sustained alongside custom lanes cycle ${k}" 30

    remove_stream "${PREFIX}0" "${STREAM}"
    remove_stream "${PREFIX}1" "${STREAM}"
    remove_stream "${PREFIX}2" "${STREAM}"
    remove_stream "${PREFIX}3" "${STREAM}"
    echo "[test_35] cycle ${k}/${reps}: removed all streams"
    sleep 2
  done

  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 36: LIFECYCLE-RACE tripwire -- rapid stream churn + REPEATED mid-churn teardown.
#
# Why this exists (audit follow-up):
#   The 2026-07 production-safety audit found several teardown/lifetime races that the
#   suite could not see: (a) modelmux_multimodel_bin_free destroying shared state before the
#   streaming threads were quiesced, (b) deferred-free / tracked-source idles racing a
#   cancel from the teardown thread (double free), (c) control idles + the pending
#   poller dispatching against a freed bin, and (d) a shutdown HANG when a model bin's
#   load thread never finished (unbounded gst_element_get_state). Existing cases stop
#   the app once, at a timing-benign moment (test_32 stops mid-reconfig exactly once);
#   these bugs need REPEATED teardown while add/remove work is genuinely in flight.
#
# What this does:
#   RACE_REPS full app cycles. Each cycle: bring up streams, churn add/remove (including a
#   parallel no-settle burst like test_34), then stop the app IMMEDIATELY with the
#   churn still in flight. Cycle 1 additionally re-adds a churned camera and asserts
#   its output provenance carries the RIGHT stream name (regression for the lock-free
#   stream-name mirror that replaced the racy per-frame GHashTable read).
#
# What to validate:
#   - every cycle's app exits on SIGINT within the grace window: the log must NOT
#     contain "did not exit on SIGINT" (regression net for the shutdown-hang class);
#   - provenance name correct after churn/slot-recycling (cycle 1);
#   - the DEFAULT sanity gates (no CRITICAL / assert / segfault) cover the crash class.
#
# NOTE: per-cycle checks are SILENT and aggregated into ONE verdict (all cycles must
#   pass). For fully deterministic detection also run against an ASan/TSan build of
#   libnvdsgst_modelmux.so (see the audit notes; the suite has no sanitizer mode yet).
test_36() {
  echo "-----------------------------------------------------------------------"
  echo "[test_36] starting test 36 (rapid churn + repeated mid-churn teardown -- lifecycle-race tripwire)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2

  local cycles="${RACE_REPS}"      # churn+teardown app cycles (global RACE_REPS)
  local cycle wj wok fails=0 first_fail=""
  for (( cycle=1; cycle<=cycles; cycle++ )); do
    echo "[test_36] ===== cycle ${cycle}/${cycles} ====="
    log_mark
    start_app || return

    add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
    sleep "${DELAY}"
    # silent per-cycle check: Cam-0 wired in THIS cycle's window
    wok=0
    for (( wj=0; wj<15; wj++ )); do
      if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF "stream 0 WIRED"; then wok=1; break; fi
      sleep 1
    done
    if [ "${wok}" -ne 1 ]; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${cycle}"
      echo "[test_36] WARN cycle ${cycle}: Cam-0 did not wire on the default primary"
    fi

    # churn one camera: add/remove cycles recycle its source slot + branch elements
    local i
    for i in 1 2 3; do
      add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
      sleep 2
      remove_stream "${PREFIX}1" "${STREAM}"
      sleep 1
    done
    echo "[test_36] c${cycle}: churned Cam-1 add/remove x3"

    if [ "${cycle}" = "1" ]; then
      # after churn, a re-added camera must carry the RIGHT provenance stream name
      # (the per-frame name lookup was moved to a lock-free mirror; a stale/corrupt
      # slot after recycling would show here as a wrong/missing name). One-time,
      # named asserts -- kept out of the aggregate.
      log_mark
      add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
      sleep "${DELAY}"
      assert_count_min "name=Cam-1  model=${DEFAULT_MODEL}" 2 \
        "test_36: Cam-1 provenance carries the correct stream name after churned re-add" 20
      assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 1 \
        "test_36: Cam-0 kept flowing through the churn" 10
      log_mark   # re-scope so the SIGINT check below sees this cycle's stop only
    fi

    # PARALLEL no-settle burst, then stop IMMEDIATELY: teardown must race the
    # in-flight attach/detach + their deferred main-loop work and still exit cleanly.
    ( add_stream "${PREFIX}2" "Cam-2" "${STREAM}" ) & local ba=$!
    ( remove_stream "${PREFIX}0" "${STREAM}" ) & local br=$!
    wait "${ba}" "${br}" 2>/dev/null || true
    echo "[test_36] c${cycle}: burst issued -- stopping app with work in flight"
    stop_app
    # silent per-cycle check: the app must have exited on SIGINT inside stop_multimodel's
    # grace window; a SIGKILL fallback means a teardown deadlock/hang (the bug class fixed).
    if tail -n "+$((LOG_MARK+1))" "${APP_LOG}" 2>/dev/null | grep -qF "did not exit on SIGINT"; then
      fails=$((fails+1)); [ -z "${first_fail}" ] && first_fail="${cycle}"
      echo "[test_36] WARN cycle ${cycle}: app did NOT exit on SIGINT (teardown hang)"
    fi
  done
  if [ "${fails}" -eq 0 ]; then
    _pass "test_36: all ${cycles} churn+mid-churn-teardown cycles wired and exited cleanly on SIGINT"
  else
    _fail "test_36: ${fails} failure(s) across ${cycles} churn+teardown cycles (first: cycle ${first_fail}) -- lifecycle race regression"
  fi
  echo "-----------------------------------------------------------------------"
}


# Test 37: MODEL-NAME VALIDATION -- reserved separators ('@' / ';') are rejected at register.
#
# Why this exists (audit follow-up):
#   A model name containing '@' (the canonical "name@version" instance-key separator)
#   used to register fine, but every later key split cut at the FIRST '@' -- so
#   "yolo@fp16" silently became name "yolo" / version "fp16@1": status lookups,
#   routing and unload all mismatched with no error. The registry now rejects names
#   containing '@' or ';' up front, loudly.
#
# What to validate:
#   - model/load with a reserved-separator name is REJECTED with the explicit
#     "reserved separators" warning (not silently registered);
#   - model/status does NOT list the bad name;
#   - the pipeline is unaffected: the default-model stream keeps flowing.
test_37() {
  echo "-----------------------------------------------------------------------"
  echo "[test_37] starting test 37 (reserved-separator model names rejected at register)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2

  start_app || return

  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  sleep "${DELAY}"
  assert_log "stream 0 WIRED" "test_37: Cam-0 wired on the default primary" 15

  # a name embedding the instance-key separator must be rejected at registration
  log_mark
  load_model "Bad@Model" "1" "" "${CFG_DIR}/config_infer_primary.txt"
  sleep "${DELAY}"
  assert_log "reserved separators" \
    "test_37: model name 'Bad@Model' rejected with the reserved-separator warning" 10
  assert_log "NAME_INVALID" \
    "test_37: the load is HARD-rejected at the REST gate with code NAME_INVALID (no phantom bin warmed)" 10
  get_model_status
  assert_status_not_contains "Bad@Model" \
    "test_37: rejected name does not appear in model/status"
  assert_status_not_contains "\"Bad\"" \
    "test_37: no phantom 'Bad' model instance appears in model/status either"

  # pipeline unaffected by the rejected load
  log_mark
  sleep "${DELAY}"
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 \
    "test_37: Cam-0 keeps flowing after the rejected model/load" 15

  remove_stream "${PREFIX}0" "${STREAM}"
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 38: PERF OBSERVATION -- one Primary ModelBin serving 30 streams.
#
# Purpose:
#   Bring up a single runtime-default primary model with a batch-30 engine and attach
#   30 streams to that one ModelBin. Perf metrics are enabled so model/status captures
#   aggregate model fps plus per-stream primary fps/latency for manual comparison
#   against test_39.
#
# Notes:
#   - The default STREAM is a finite sample clip (~46s). The default 150s dwell is
#     meant for manual perf runs with a live/looping source:
#       MM_PERF_DWELL_SEC=150 STREAM=<live-uri> ./run_tests.sh 38
test_38() {
  echo "-----------------------------------------------------------------------"
  echo "[test_38] starting test 38 (perf observation: one Primary ModelBin with 30 streams)"

  # 1) Configure one 30-stream primary-only graph.
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        30
  cfg_set tiled-display        rows                  5
  cfg_set tiled-display        columns               6
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  30
  cfg_set multimodel           attach-perf-metric    1
  cfg_set multimodel           perf-metric-interval-sec 1
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) Start the app, then establish the runtime default primary with a batch-30 engine.
  start_app || return

  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${B30_ENGINE}" "" "True"
  echo "[test_38] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_38: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12

  # 3) Attach 30 streams to the single runtime default primary bin.
  local i
  for i in {0..29}; do
    log_mark
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STREAM}"
    echo "[test_38] Added Cam-${i} (DEFAULT primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
    assert_log "stream ${i} WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=-" \
      "test_38: Cam-${i} wired on the default primary" 15
  done

  # 4) Let throughput stabilize, then print full model/status into the log for manual
  # fps comparison. Do not filter here: the full dump carries every per-stream ref
  # with its per-role fps.
  local dwell="${MM_PERF_DWELL_SEC:-120}"
  echo "[test_38] stabilizing and collecting perf metrics for ${dwell}s (override with MM_PERF_DWELL_SEC)"
  sleep "${dwell}"
  get_model_status
  assert_status_contains "\"fps\":"         "test_38: model/status includes aggregate fps"
  assert_status_contains "\"model\":{\"fps\":" "test_38: model/status includes per-stream serving-model fps (ref \"fps\")"

  # 5) Stop with streams still active; default sanity gates verify teardown stayed clean.
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 39: PERF OBSERVATION -- A/B pair, 15 streams through Primary and Shadow bins.
#
# Purpose:
#   Bring up a runtime-default primary and runtime-default shadow, each with a batch-15
#   engine, then attach 15 streams with no metadata so each stream runs both roles.
#   This gives one Primary ModelBin + one Shadow ModelBin at 15 streams each, with
#   perf metrics captured in model/status for comparison against test_38.
#
# Notes:
#   - The 2x15 tiler is intentional: row 1 shows primary outputs, row 2 shows shadow.
#   - The default STREAM is finite. Use a live/looping source for the default 150s dwell.
test_39() {
  echo "-----------------------------------------------------------------------"
  echo "[test_39] starting test 39 (perf observation: A/B pair with 15 streams per role)"

  # 1) Configure a 15-stream A/B graph: 15 primary outputs + 15 shadow outputs.
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        15
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               15
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  15
  cfg_set multimodel           attach-perf-metric    1
  cfg_set multimodel           perf-metric-interval-sec 1
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) Start the app, then establish runtime default primary and shadow.
  start_app || return

  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${B15_ENGINE}" "" "True"
  echo "[test_39] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_39: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${B15_ENGINE}" "" "" "True"
  echo "[test_39] Requested loading of (DEFAULT Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_39: default shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded + designated default" 12

  # 3) Attach 15 streams. With both runtime defaults set, each stream runs primary + shadow.
  local i
  for i in {0..14}; do
    log_mark
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STREAM}"
    echo "[test_39] Added Cam-${i} (DEFAULT primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + DEFAULT shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
    assert_log "stream ${i} WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}" \
      "test_39: Cam-${i} wired on the default primary + shadow" 15
  done

  # 4) Let throughput stabilize, then print full model/status into the log for manual
  # fps comparison. Do not filter here: the full dump carries every per-stream ref
  # with its per-role (model + shadow) fps.
  local dwell="${MM_PERF_DWELL_SEC:-120}"
  echo "[test_39] stabilizing and collecting perf metrics for ${dwell}s (override with MM_PERF_DWELL_SEC)"
  sleep "${dwell}"
  get_model_status
  assert_status_contains "\"fps\":"         "test_39: model/status includes aggregate fps"
  assert_status_contains "\"model\":{\"fps\":" "test_39: model/status includes per-stream serving-model fps (ref \"fps\")"
  assert_status_contains "\"shadow\":{\"fps\":"  "test_39: model/status includes per-stream shadow fps (shadow ref \"fps\")"

  # 5) Stop with streams still active; default sanity gates verify teardown stayed clean.
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 40: model/update -- IN-PLACE OTA checkpoint transition (same network, new weights),
#          the API_DESIGN.md example flow: v1 -> v2 -> v3 on the LIVE default primary.
#
# Flow:
#   1. model/load Trafficcamnet@1 with the DEFAULT engine (def_primary flag = load +
#      wait_for_model + model/routing default -> the runtime default primary).
#   2. Add Cam-0..Cam-3 with NO per-stream model (all route to the default primary).
#   3. Let them flow ~10s.
#   4. POST /api/v1/model/update name=Trafficcamnet from_version=1 -> version=2 with the
#      WORST engine (prebuilt b4 int8-nocalib checkpoint).
#   5. sleep 10, then model/update again from_version=2 -> version=3 with the BEST engine
#      (prebuilt b4 fp32 checkpoint).
#   6. Remove the streams, stop the app.
#
# What happens (the behaviour under test):
#   Same network, new weights: each update hot-swaps the engine on the LIVE instance
#   across its shard(s) -- no new bin, NO reroute, gie kept, the streams never move; only
#   the version identity moves 1 -> 2 -> 3. routing_revision still bumps (the version
#   identity is routing CONTENT) and the accepted response is a 202 admission receipt.
#
# What to validate:
#   - "MODEL UPDATE (in-place)" admitted + "OTA reload TRIGGERED" + "group COMMITTED" for
#     BOTH updates (the group transaction commits only after EVERY shard's nvinfer
#     confirm); "group ROLLED BACK" never appears.
#   - model/status: the models[].versions[] entry moves to "2" (then "3") state serving and
#     the superseded entry disappears (JSON-parsed -- jsoncpp sorts keys, never substring-
#     match key pairs). The streams' REFS move with it ('@1' -> '@2' -> '@3'): the live key
#     is renamed from_version -> version at every touch point (API_DESIGN.md 5.3).
#   - streams never move: no "model/routing applied to" during either update phase, and the
#     model's gie_id set (status instances) is IDENTICAL before/after both updates.
#   - each update response is a 202 with reason "...accepted; swapping..." (admission receipt).
#   - routing_revision increased across the two updates.
#   - dropped = 0 / no stall / no gaps / no CRITICAL / source_id restore (the default
#     sanity gates run automatically in run_one_auto after the case ends).
test_40() {
  echo "-----------------------------------------------------------------------"
  echo "[test_40] starting test 40 (model/update in-place OTA checkpoint transition v1 -> v2 -> v3)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  # multimodel config updates: no config defaults -- the runtime default primary is
  # established below via the v1 load + route-default flow (def_primary flag).
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load VERSION 1 on the DEFAULT engine and make it the runtime default primary
  #    (def_primary flag = load + wait_for_model + model/routing default).
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${DEFAULT_ENGINE}" "" "True"
  echo "[test_40] Requested loading of (DEFAULT Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_40: default primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default" 12

  # 4) add Cam-0..Cam-3 with NO per-stream model -> all route to the default primary
  local i
  for i in {0..3}; do
    log_mark
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STREAM}"
    echo "[test_40] Added Cam-${i} (DEFAULT primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
    assert_log "stream ${i} WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=-" \
      "test_40: Cam-${i} wired on the default primary" 15
  done

  # 5) let the 4 streams flow, then snapshot the pre-update baseline: routing_revision
  #    (must INCREASE across the updates) and the model's gie_id set (must NOT change --
  #    the in-place swap keeps the gie, no new bin).
  sleep 10
  get_model_status "model=${DEFAULT_MODEL}"
  local rev0 gie0
  rev0="$(_status_routing_revision)"
  gie0="$(_status_model_gies "${DEFAULT_MODEL}")"
  echo "[test_40] pre-update baseline: routing_revision=${rev0} gie_id(s)=[${gie0}]"

  # 6) UPDATE 1: in-place checkpoint transition 1 -> 2 with the WORST engine. The live
  #    bin hot-swaps its engine (streams never move); the version identity moves to 2.
  log_mark
  model_update "${DEFAULT_MODEL}" "1" "2" "${WORST_ENGINE}"
  echo "[test_40] model/update '${DEFAULT_MODEL}' 1 -> 2 (WORST engine, in-place hot-swap)"
  assert_rest_body_contains "accepted;swapping" "test_40: update-1 response acknowledges the in-place swap (202 accepted)"
  assert_log "MODEL UPDATE (in-place) name='${DEFAULT_MODEL}' 1 -> 2" "test_40: update-1 admitted as an IN-PLACE checkpoint transition" 10
  assert_log "OTA reload TRIGGERED" "test_40: update-1 engine hot-swap TRIGGERED on the live bin (gie kept)" 10
  assert_log "group COMMITTED" "test_40: update-1 group transaction committed after nvinfer confirm" 30
  assert_log_absent "group ROLLED BACK" "test_40: update-1 was never rolled back"
  # streams NEVER move on model/update: no scoped reroute may fire during the swap
  # (scoped to this update phase via the log_mark above).
  assert_log_absent "model/routing applied to" "test_40: update-1 moved NO stream (no reroute during the swap)"
  # the VERSION entry moved to "2" (serving) and the superseded "1" entry is gone
  # (the identity MOVED, it was not copied). JSON-parsed: jsoncpp sorts keys, so a
  # '"version":...,"state":...' substring can never match.
  assert_status_version_eventually "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "2" "test_40: version entry moved to '2' (serving) after update-1"
  assert_status_version_eventually_absent "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "1" "test_40: superseded version entry '1' gone after update-1"
  # the streams' REFS moved WITH the identity: the live key (hash tables, shard
  # keys, routing-table stream refs) is renamed from_version -> version at every
  # touch point (API_DESIGN.md 5.3), so status stream refs now report '@2'.
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "2" "test_40: stream refs moved to the '@2' key (identity rename, API_DESIGN.md 5.3)"
  sleep 10

  # 7) UPDATE 2: transition again 2 -> 3 with the BEST engine (from_version=2 is the CAS
  #    guard naming the NOW-live version).
  log_mark
  model_update "${DEFAULT_MODEL}" "2" "3" "${BEST_ENGINE}"
  echo "[test_40] model/update '${DEFAULT_MODEL}' 2 -> 3 (BEST engine, in-place hot-swap)"
  assert_rest_body_contains "accepted;swapping" "test_40: update-2 response acknowledges the in-place swap (202 accepted)"
  assert_log "MODEL UPDATE (in-place) name='${DEFAULT_MODEL}' 2 -> 3" "test_40: update-2 admitted as an IN-PLACE checkpoint transition" 10
  assert_log "OTA reload TRIGGERED" "test_40: update-2 engine hot-swap TRIGGERED on the live bin (gie kept)" 10
  assert_log "group COMMITTED" "test_40: update-2 group transaction committed after nvinfer confirm" 30
  assert_log_absent "group ROLLED BACK" "test_40: update-2 was never rolled back"
  assert_log_absent "model/routing applied to" "test_40: update-2 moved NO stream (no reroute during the swap)"
  assert_status_version_eventually "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "3" "test_40: version entry moved to '3' (serving) after update-2"
  assert_status_version_eventually_absent "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "2" "test_40: superseded version entry '2' gone after update-2"
  # the refs follow the identity again: '@2' -> '@3' (chained CAS on the live key).
  assert_status_ref_eventually "stream=Cam-0" model "${DEFAULT_MODEL}" "3" "test_40: stream refs moved to the '@3' key after update-2 (identity rename)"

  # 8) gie preserved + routing_revision moved: compare the post-update snapshot against
  #    the pre-update baseline (same JSON-parsed helpers as the ref asserts).
  get_model_status "model=${DEFAULT_MODEL}"
  local rev1 gie1
  rev1="$(_status_routing_revision)"
  gie1="$(_status_model_gies "${DEFAULT_MODEL}")"
  echo "[test_40] post-update: routing_revision=${rev1} gie_id(s)=[${gie1}]"
  if [ -n "${gie0}" ] && [ "${gie1}" = "${gie0}" ]; then
    _pass "test_40: gie_id preserved across both updates (still [${gie0}] -- same bin, no reroute)"
  else
    _fail "test_40: gie_id changed across the updates (before=[${gie0}] after=[${gie1}])"
  fi
  if [ -n "${rev0}" ] && [ -n "${rev1}" ] && [ "${rev1}" -gt "${rev0}" ] 2>/dev/null; then
    _pass "test_40: routing_revision increased across the updates (${rev0} -> ${rev1})"
  else
    _fail "test_40: routing_revision did not increase across the updates (${rev0} -> ${rev1})"
  fi

  # 9) teardown: remove the streams, then stop the app (the default sanity gates --
  #    ZERO frame loss / no stall / no gaps / no CRITICAL / source_id restore -- run
  #    automatically in run_one_auto after the case ends).
  for i in {0..3}; do
    remove_stream "${PREFIX}${i}" "${STREAM}"
    sleep "${DELAY}"
  done
  echo "[test_40] removed Cam-0, Cam-1, Cam-2, Cam-3"

  # 10) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

test_41() {
  echo "-----------------------------------------------------------------------"
  echo "[test_41] starting test 41 (default switch sweeps ONLY default-routed streams; explicitly pinned streams stay -- via_default/origin)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set        source-list          max-batch-size        3
  cfg_set        tiled-display        rows                  2
  cfg_set        tiled-display        columns               3
  # multimodel config updates
  cfg_set        multimodel           unified-batch         1
  cfg_set        multimodel           per-model-batch-size  2
  # BOTH defaults are established at RUNTIME via model/load + a model/routing default switch --
  # comment out the config defaults for both roles so the API is the sole source of the defaults.
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load THREE versions of the same model:
  #    @1 -> runtime default PRIMARY, @2 -> runtime default SHADOW, @3 -> loaded only
  #    (limbo; it becomes the NEW default primary later in the test).
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${WORST_ENGINE}" "" "True"
  echo "[test_41] Loaded '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' + designated default PRIMARY"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "default model -> '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'" "test_41: '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' loaded + designated default primary" 12
  load_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}" "${BEST_ENGINE}" "" "" "True"
  echo "[test_41] Loaded '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' + designated default SHADOW"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  assert_log "default shadow -> '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}'" "test_41: '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}' loaded + designated default shadow" 12
  load_model "${DEFAULT_MODEL}" "3" "${DEFAULT_ENGINE}"
  echo "[test_41] Loaded '${DEFAULT_MODEL}@3' (limbo only -- the future default primary)"
  wait_for_model "${DEFAULT_MODEL}" "3"

  # 4) Cam-0: NO metadata refs -> DEFAULT-routed (primary=@1 shadow=@2, origin=default).
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_41] Added Cam-0 (no refs -> resolves via the runtime defaults)"
  assert_log "stream 0 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_41: Cam-0 wired on default primary @${DEFAULT_MODEL_VERSION} + default shadow @${DEFAULT_SHADOW_MODEL_VERSION}" 15

  # 5) Cam-1: EXPLICIT metadata refs naming the SAME models the defaults point at --
  #    value-identical to Cam-0, but the explicit refs make BOTH lanes PINNED.
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "${DEFAULT_MODEL}" "${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "[test_41] Added Cam-1 (explicit refs to the SAME @${DEFAULT_MODEL_VERSION}/@${DEFAULT_SHADOW_MODEL_VERSION} -> pinned)"
  assert_log "stream 1 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_41: Cam-1 wired on the same models (explicitly requested)" 15

  # 5b) Cam-2: MIXED origins in one stream -- explicit CUSTOM primary @3 (the version
  #     that will BECOME the default later), shadow ref OMITTED -> default shadow @2.
  #     primary lane = pinned, shadow lane = default-routed.
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}" "${DEFAULT_MODEL}" "3"
  echo "[test_41] Added Cam-2 (explicit custom primary @3 + omitted shadow -> default shadow @${DEFAULT_SHADOW_MODEL_VERSION})"
  assert_log "stream 2 WIRED  (primary=${DEFAULT_MODEL}@3  shadow=${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION})" "test_41: Cam-2 wired on explicit @3 primary + default shadow @${DEFAULT_SHADOW_MODEL_VERSION}" 15
  sleep "${DELAY}"

  # 6) model/status exposes the ORIGIN per lane: Cam-0 default-routed, Cam-1 pinned --
  #    same (name, version) on both streams, different origin.
  get_model_status "stream=Cam-0"
  assert_status_contains "\"origin\":\"default\"" "test_41: Cam-0 lanes report origin=default (resolved via the fallback)"
  assert_status_not_contains "\"origin\":\"pinned\"" "test_41: Cam-0 has NO pinned lane"
  get_model_status "stream=Cam-1"
  assert_status_contains "\"origin\":\"pinned\"" "test_41: Cam-1 lanes report origin=pinned (explicit refs, even though value-identical to the defaults)"
  assert_status_not_contains "\"origin\":\"default\"" "test_41: Cam-1 has NO default-routed lane"
  get_model_status "stream=Cam-2"
  # NOTE: patterns follow the REST layer's jsoncpp serialization, which sorts keys
  # ALPHABETICALLY (gpu, instance, name, origin, version) -- origin precedes version.
  assert_status_contains "\"origin\":\"pinned\",\"version\":\"3\"" "test_41: Cam-2 primary @3 is PINNED (explicit ref)"
  assert_status_contains "\"origin\":\"default\",\"version\":\"${DEFAULT_SHADOW_MODEL_VERSION}\"" "test_41: Cam-2 shadow @${DEFAULT_SHADOW_MODEL_VERSION} is DEFAULT-routed (ref omitted) -- mixed origins in ONE stream"

  # 7) SWITCH the default primary @1 -> @3. The sweep must move ONLY the
  #    default-routed stream (Cam-0); the pinned Cam-1 must stay on @1, and
  #    Cam-2 (already pinned on @3 = the NEW default) must be untouched too --
  #    it is not on the OLD default, so the sweep never even considers it.
  model_routing_default "model" "${DEFAULT_MODEL}" "3"
  echo "[test_41] default primary switched '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' -> '${DEFAULT_MODEL}@3'"
  assert_log "default model -> '${DEFAULT_MODEL}@3'" "test_41: default primary re-designated to @3" 12
  assert_log "default primary switch: rerouted 1 stream(s) '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' -> '${DEFAULT_MODEL}@3'" "test_41: sweep moved exactly ONE stream (the default-routed Cam-0)" 15
  assert_log "1 stream(s) on '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' are PINNED" "test_41: the pinned Cam-1 was visibly SKIPPED by the sweep (not silently)" 15

  # 8) end state: Cam-0 followed the default to @3 (still origin=default -- it keeps
  #    following future switches); Cam-1 is untouched on @1 (origin=pinned).
  assert_status_eventually "stream=Cam-0" "\"origin\":\"default\",\"version\":\"3\"" "test_41: Cam-0 rerouted to the NEW default @3, still default-routed" 20
  assert_status_eventually "stream=Cam-1" "\"origin\":\"pinned\",\"version\":\"${DEFAULT_MODEL_VERSION}\"" "test_41: Cam-1 STAYED pinned on @${DEFAULT_MODEL_VERSION} across the default switch" 20
  # Cam-2: SAME model as the new default (@3) but still origin=pinned -- becoming
  #  the default does NOT convert an explicitly-pinned lane into a follower; its
  #  default-routed shadow @2 is untouched (only the PRIMARY default changed).
  assert_status_eventually "stream=Cam-2" "\"origin\":\"pinned\",\"version\":\"3\"" "test_41: Cam-2 primary stays PINNED on @3 (same model as the new default, origin unchanged)" 20
  assert_status_eventually "stream=Cam-2" "\"origin\":\"default\",\"version\":\"${DEFAULT_SHADOW_MODEL_VERSION}\"" "test_41: Cam-2 default-routed shadow @${DEFAULT_SHADOW_MODEL_VERSION} untouched by the PRIMARY switch" 20
  # the demoted @1 must NOT be reclaimed -- the pinned Cam-1 still serves on it.
  assert_status_version_eventually "" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" "test_41: demoted old default @${DEFAULT_MODEL_VERSION} kept alive (pinned stream serves on it)" 12

  # 9) BOOKKEEPING-ONLY pin: explicitly route Cam-0 to the model it ALREADY runs (@3).
  #    No lane op is launched (same model), but the request's origin MUST land on the
  #    lane: Cam-0 flips default->pinned, so the NEXT default switch leaves it alone.
  model_routing "${PREFIX}0" "${DEFAULT_MODEL}" "3"
  assert_log "routing BOOKKEEPING committed" "test_41: same-model explicit route committed as bookkeeping (origin recorded, no lane op)" 12
  assert_status_eventually "stream=Cam-0" "\"origin\":\"pinned\",\"version\":\"3\"" "test_41: Cam-0 flipped default->pinned by explicitly naming its CURRENT model" 15

  # 10) switch the default BACK to @${DEFAULT_MODEL_VERSION}: the sweep must move NOBODY --
  #     Cam-0 (just pinned) and Cam-2 (pinned since attach) both sit on the old default @3.
  model_routing_default "model" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_log "stream(s) on '${DEFAULT_MODEL}@3' are PINNED" "test_41: BOTH pinned lanes on @3 visibly skipped by the second switch" 15
  assert_status_eventually "stream=Cam-0" "\"origin\":\"pinned\",\"version\":\"3\"" "test_41: Cam-0 UNMOVED by the second default switch (the recorded pin survives)" 20
  sleep 10

  # 11) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

test_42() {
  echo "-----------------------------------------------------------------------"
  echo "[test_42] starting test 42 (per-model/per-version batch-size: sizing, auto-shard, immutability, clamp, preflight)"

  # 1) config: 4-stream element; GLOBAL per-model batch 2; Car's CATALOG (name-level)
  #    batch 1; a CONFIG per-version block gives Car@3 batch 3. Defaults commented --
  #    every stream routes explicitly (this test is about capacity, not routing).
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set        source-list          max-batch-size        4
  cfg_set        tiled-display        rows                  2
  cfg_set        tiled-display        columns               2
  cfg_set        multimodel           unified-batch         0
  cfg_set        multimodel           per-model-batch-size  2
  cfg_set        model-name-1         batch-size            1
  cfg_set        model-Car-3          batch-size            3
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) CATALOG name-level batch: load Car@1 with NO batch in the payload -> the
  #    [model-name-1] batch-size=1 must size the instance (capacity 1, not the global 2).
  load_model "${CAR_MODEL}" "1" "${WORST_ENGINE}"
  wait_for_model "${CAR_MODEL}" "1"
  get_model_status "model=${CAR_MODEL}&version=1"
  assert_status_contains "\"capacity\":1" "test_42: Car@1 instance sized by the CATALOG batch-size=1 (not the global 2)"

  # 4) capacity enforcement -> AUTO-SHARD: two streams on a batch-1 instance must
  #    yield TWO instances (id 0 + id 1), never co-batching past the cap.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}" "${CAR_MODEL}" "1"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}" "${CAR_MODEL}" "1"
  assert_log "stream 0 WIRED  (primary=${CAR_MODEL}@1" "test_42: Cam-0 wired on Car@1" 15
  assert_log "stream 1 WIRED  (primary=${CAR_MODEL}@1" "test_42: Cam-1 wired on Car@1" 15
  assert_status_eventually "model=${CAR_MODEL}&version=1" "\"id\":1,\"state\"" "test_42: second stream AUTO-SHARDED onto a second Car@1 instance (capacity 1 enforced)" 20
  # provenance carries the producing instance's batch (new meta field)
  assert_log "batch=1  engine=" "test_42: provenance meta reports the instance's batch (batch=1)" 20

  # 5) BATCH IMMUTABILITY: re-loading the LIVE Car@1 with a different batch is rejected.
  #    The element now decides admission synchronously, so the reject comes back in
  #    the REST response (409 BATCH_IMMUTABLE), not just the log.
  load_model "${CAR_MODEL}" "1" "${WORST_ENGINE}" "" "" "" 2
  # element decides admission synchronously -> the reject is in the REST response
  assert_rest_body_contains "BATCH_IMMUTABLE" "test_42: re-load with a different batch REJECTED in the REST response (BATCH_IMMUTABLE)"
  assert_http_status 409 "test_42: ...as a 409 conflict -- the request is well-formed, the LIVE STATE refuses it"

  # 6) API per-version batch beats the catalog: Car@2 loaded with batch_size=2.
  load_model "${CAR_MODEL}" "2" "${BEST_ENGINE}" "" "" "" 2
  wait_for_model "${CAR_MODEL}" "2"
  get_model_status "model=${CAR_MODEL}&version=2"
  assert_status_contains "\"capacity\":2" "test_42: Car@2 sized by the API batch_size=2 (per-version beats the catalog's 1)"

  # 7) CONFIG per-version block: Car@3 loaded with NO payload batch -> the
  #    [model-Car-3] batch-size=3 block must size it.
  load_model "${CAR_MODEL}" "3" "${DEFAULT_ENGINE}"
  wait_for_model "${CAR_MODEL}" "3"
  get_model_status "model=${CAR_MODEL}&version=3"
  assert_status_contains "\"capacity\":3" "test_42: Car@3 sized by the [model-Car-3] batch-size=3 config block"

  # 8) ADMISSION CLAMP: batch_size=8 exceeds the element's batch-size (4) -> clamped
  #    AT ADMISSION with a visible warn (the registry then records the capacity that
  #    actually deploys, so later updates repeating the value are never falsely
  #    rejected, and the engine preflight compares against the REAL capacity).
  load_model "${CAR_MODEL}" "4" "${DEFAULT_ENGINE}" "" "" "" 8
  assert_log "batch_size 8 exceeds the element's batch-size 4 -- CLAMPED to 4" "test_42: over-element batch VISIBLY clamped at admission (not silent)" 15
  wait_for_model "${CAR_MODEL}" "4"
  get_model_status "model=${CAR_MODEL}&version=4"
  assert_status_contains "\"capacity\":4" "test_42: Car@4 capacity is the clamped element batch-size (4)"

  # 9) UPDATE BATCH MISMATCH: an in-place model/update keeps the element -- declaring a
  #    DIFFERENT batch for the to-version cannot be honoured and must be rejected.
  model_update "${CAR_MODEL}" "3" "5" "${DEFAULT_ENGINE}" 1
  # synchronous admission -> the reject comes back in the REST response
  assert_rest_body_contains "UPDATE_BATCH_MISMATCH" "test_42: in-place update with a different batch REJECTED in the REST response (UPDATE_BATCH_MISMATCH)"
  assert_http_status 409 "test_42: ...as a 409 conflict (batch mismatch is a live-state conflict, not a bad request)"
  get_model_status "model=${CAR_MODEL}&version=3"
  assert_status_contains "\"capacity\":3" "test_42: rejected update left Car@3 untouched (still live, capacity 3)"

  # 10) drain
  remove_stream "${PREFIX}0" "${STREAM}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  stop_app
  echo "-----------------------------------------------------------------------"
}

test_43() {
  echo "-----------------------------------------------------------------------"
  echo "[test_43] starting test 43 (pipeline BLOCKS on the default engine build; defaults infer from frame 0, NO passthrough)"

  # The engine is deleted below to force a full TRT build on startup. The element holds
  # the PLAYING transition until the configured default is WARMED, so the default is READY
  # before any stream attaches -> zero passthrough. Guards that a stream can never be served
  # by a model that is not ready.
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  4

  # remove the engine so the default must do a FULL TRT build on startup -- the wait we test
  rm -f "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"

  # start the app. With the flag ON the pipeline BLOCKS reaching PLAYING until the default
  # engine is built, so REST-ready can lag by the whole build -> give startup a generous
  # window (the build is finite; the warm thread caps it at 30 min internally).
  APP_START_TIMEOUT=180 APP_WARMUP=0 start_app || return

  # (1) THE PROOF: load_defaults logged that the default warmed BEFORE streaming -- emitted
  #     after the engine build completes and BEFORE the pipeline is allowed to serve any stream.
  assert_log "default PRIMARY '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' WARMED (ready before streaming)" "test_43: pipeline WAITED for the default engine build before streaming" 180

  # scan only the post-startup stream phase for any passthrough marker
  log_mark

  # (2) add the streams -- the default is ALREADY warmed, so they must attach DIRECTLY.
  add_stream "${PREFIX}0" "Cam-0" "${STREAM}"
  echo "[test_43] added Cam-0 (Default Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  echo "[test_43] added Cam-1 (Default Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_43] added Cam-2 (Default Primary Model)"
  sleep "${DELAY}"
  add_stream "${PREFIX}3" "Cam-3" "${STREAM}"
  echo "[test_43] added Cam-3 (Default Primary Model)"

  # (3) all four WIRE on the default primary -- directly (short timeout: no warm/promote latency,
  #     the engine was built at startup).
  assert_log "stream 3 WIRED  (primary=${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}" "test_43: all 4 streams wired directly on the pre-warmed default primary" 20

  # (4) THE PROOF OF NO DEFERRAL: because the default was warmed before streaming, NO stream is
  #     ever deferred to passthrough (no 'still warming -> PASSTHROUGH', no '(passthru)' tag).
  assert_log_absent "still warming -> PASSTHROUGH" "test_43: NO passthrough deferral -- default was ready before any stream" 5
  assert_log_absent "(passthru)" "test_43: no stream ever tagged '(passthru)' -- direct attach only" 3

  # (5) Cam-0 (first stream) flows sustained on the default from the start (inference, not passthrough)
  assert_count_min "name=Cam-0  model=${DEFAULT_MODEL}" 2 "test_43: Cam-0 FLOWS sustained on the default from the first frame (no passthrough gap)" 15

  # remove the streams
  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}3" "${STREAM}"
  echo "[test_43] removed Cam-0..Cam-3"

  stop_app
  echo "-----------------------------------------------------------------------"
}

# test_44: GET /api/v1/model/routing -- the routing plane as a read-only snapshot.
#
# What this covers, and why each part would otherwise fail SILENTLY:
#
#   (a) GROUPING MERGES. routes[].streams lists every stream in an identical
#       routing state. Four streams on one model must come back as ONE entry, not
#       four. A regression here is invisible in a smoke test (the data is still
#       correct, just unmerged) but makes the endpoint useless at fleet scale.
#
#   (b) GROUPING SPLITS ON `origin`. This is the subtle one. Streams that reached
#       the SAME model by different routes -- resolved from the default vs named
#       explicitly -- must NOT merge, because a later default{} move sweeps the
#       first group and leaves the second. Merging them would hide a difference
#       that changes what the next write does.
#
#   (c) NO FIELD LEAKAGE. Refs are built by whitelist. model-status refs also carry
#       origin/instance/backend/fps/latency_ms, and the perf pair changes every
#       measurement window -- leaking it would mean no two streams ever compare
#       equal, so (a) would quietly stop working the moment perf is enabled.
#
#   (d) FILTERED SHAPE. ?stream= returns just that stream and DROPS default{}
#       (redundant when origin=default, irrelevant when origin=pinned), while
#       keeping routing_revision -- the CAS token for whatever write follows.
#
# NOT covered here (needs 2 GPUs, this rig has one): grouping splitting on `gpu`
# across a multi-GPU instance group.
test_44() {
  echo "-----------------------------------------------------------------------"
  echo "[test_44] starting test 44 (model/routing GET: grouped routing snapshot)"

  # 1) config: one batch big enough for all 4 streams, so they share an instance
  #    (same device) and are therefore genuinely mergeable.
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               4
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  4
  # attach-perf-metric=1 is REQUIRED for the whitelist assertion to mean anything:
  # with perf off, model-status refs carry no fps/latency_ms at all, so
  # assert_route_no_leak would pass vacuously. With it on, the refs the projection
  # reads DO carry live perf fields that change every interval -- exactly what must
  # not reach the response (leaking them would also stop grouping from ever merging).
  cfg_set multimodel           attach-perf-metric    1
  cfg_set multimodel           perf-metric-interval-sec 2
  cfg_set multimodel                                                primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set multimodel                                                shadow-model          "${DEFAULT_MODEL};${DEFAULT_SHADOW_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}           engine                "${WORST_ENGINE}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_SHADOW_MODEL_VERSION}    engine                "${BEST_ENGINE}"

  # 2) start the app with the current config copies
  start_app || return

  # 3) load the custom primary model and default shadow model (warm to READY)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_44] Requested loading of (Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}' & Default Shadow Model '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_log "'${CAR_MODEL}@${CAR_MODEL_VERSION}' LOADED into limbo" "test_44: custom primary '${CAR_MODEL}' loaded and warmed" 12

  # 4) four streams, all resolving to the DEFAULT primary (origin "default")
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}"
  add_stream "${PREFIX}2" "${PREFIX}2" "${STREAM}"
  add_stream "${PREFIX}3" "${PREFIX}3" "${STREAM}"
  echo "[test_44] Added ${PREFIX}0..${PREFIX}3 (all on Default Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}' and Shadow '${DEFAULT_MODEL}@${DEFAULT_SHADOW_MODEL_VERSION}')"
  sleep "${DELAY}"

  # 5) BASELINE READ -- a uniform fleet is ONE entry.
  #    All four streams resolve to the same primary AND the same shadow, so every
  #    field of the entry except `streams` is identical and they merge. Both refs
  #    report origin "default": nothing named them explicitly.
  get_model_routing
  assert_route_count 1 "test_44: uniform fleet collapses to a single routes[] entry"
  assert_route_entry "${PREFIX}0,${PREFIX}1,${PREFIX}2,${PREFIX}3" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION},model.origin=default" \
      "test_44: one entry lists exactly ${PREFIX}0..${PREFIX}3 on the default primary"
  assert_route_entry "${PREFIX}0,${PREFIX}1,${PREFIX}2,${PREFIX}3" \
      "shadow.name=${DEFAULT_MODEL},shadow.version=${DEFAULT_SHADOW_MODEL_VERSION},shadow.origin=default" \
      "test_44: the same entry reports the default SHADOW alongside the primary"
  assert_route_has_key "routing_revision" "test_44: snapshot carries routing_revision (the if_revision CAS token)"
  assert_route_has_key "default"          "test_44: unfiltered snapshot carries the fleet-wide default{}"
  assert_route_no_leak                    "test_44: refs carry no model-status-only fields (whitelist projection)"

  # 6) WRITE -- move half the fleet to the custom primary. The route names only
  #    `model`, so the SHADOW is left unchanged (omitted == "leave as-is").
  #    The routing mechanics themselves are covered by tests 8 / 13 / 41; here the
  #    POST is only the stimulus, and everything below asserts what the READ says.
  model_routing "${PREFIX}2,${PREFIX}3" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_44] Routed ${PREFIX}2,${PREFIX}3 to the Custom Primary Model '${CAR_MODEL}@${CAR_MODEL_VERSION}'"
  sleep "${DELAY}"

  # 7) THE READ REFLECTS THE WRITE -- the group splits in two.
  #
  #    ${PREFIX}2/${PREFIX}3 now differ from ${PREFIX}0/${PREFIX}1 in model name, version AND origin,
  #    so they cannot share an entry. Note what did NOT change: their shadow is
  #    still the default shadow, still origin "default". That is why `origin` is
  #    reported PER REF and not once per entry -- these two lanes disagree, and a
  #    single entry-level origin would have to lie about one of them (claiming
  #    "pinned" for a stream whose shadow a default-shadow move WOULD still sweep).
  get_model_routing
  assert_route_count 2 "test_44: the read splits into two entries after the route"
  assert_route_entry "${PREFIX}0,${PREFIX}1" \
      "state=serving,model.name=${DEFAULT_MODEL},model.origin=default" \
      "test_44: ${PREFIX}0,${PREFIX}1 untouched -- still following the default primary"
  assert_route_entry "${PREFIX}2,${PREFIX}3" \
      "state=serving,model.name=${CAR_MODEL},model.version=${CAR_MODEL_VERSION},model.origin=pinned" \
      "test_44: ${PREFIX}2,${PREFIX}3 report the custom primary, now origin 'pinned'"
  assert_route_entry "${PREFIX}2,${PREFIX}3" \
      "shadow.name=${DEFAULT_MODEL},shadow.version=${DEFAULT_SHADOW_MODEL_VERSION},shadow.origin=default" \
      "test_44: their SHADOW is unchanged and still origin 'default' (lanes disagree -- per-ref origin)"
  assert_route_no_leak "test_44: refs still clean after an explicit route"

  # 8) POINT LOOKUP -- one entry, that stream only, and no fleet-wide default{}
  get_model_routing "stream=${PREFIX}2"
  assert_route_count 1 "test_44: ?stream= returns exactly one entry"
  assert_route_entry "${PREFIX}2" "state=serving,model.name=${CAR_MODEL},model.origin=pinned" \
      "test_44: ?stream=${PREFIX}2 returns ${PREFIX}2 alone (split out of its group)"
  assert_route_has_key   "routing_revision" "test_44: filtered snapshot still carries routing_revision"
  assert_route_lacks_key "default"          "test_44: filtered snapshot omits default{} (redundant/irrelevant for one stream)"

  # 9) FILTER MUST FAIL CLOSED. Every one of these previously returned EVERY route
  #    (or the wrong stream); each is a regression guard for a specific bug:
  #      ?stream=<unknown>  -> unknown id must not fall through to "no filter"
  #      ?stream=           -> present-but-EMPTY value read as "no filter"
  #      ?stream            -> bare key with no '=' at all, same fail-open
  #      ?stream=%00        -> decoded NUL became an empty C string downstream,
  #                            which the element reads as "no filter"
  get_model_routing "stream=${PREFIX}does-not-exist"
  assert_route_count 0 "test_44: unknown ?stream= yields an empty routes[], not an error"
  get_model_routing "stream="
  assert_route_count 0 "test_44: ?stream= (empty value) matches nothing -- must NOT return every route"
  get_model_routing "stream"
  assert_route_count 0 "test_44: bare ?stream (no '=') matches nothing -- must NOT return every route"
  get_model_routing "stream=%00"
  assert_route_count 0 "test_44: ?stream=%00 (NUL) matches nothing -- must NOT return every route"
  # sanity: the SAME endpoint with no filter at all still returns the fleet, so the
  # assertions above are proving fail-closed rather than a globally broken endpoint
  get_model_routing
  assert_route_count 2 "test_44: no filter still returns the fleet (the checks above are not vacuous)"

  # 10) DETERMINISM. The projection buckets out of a hash table, so the explicit
  #     sort is the only thing keeping two consecutive reads identical.
  assert_route_deterministic "test_44: two consecutive GETs are byte-identical (sorted output)"

  # 11) camera_id != camera_name. THIS is the case that proves the response carries
  #     the id model/routing accepts rather than the element's display name: the model
  #     element stores only ONE name per stream (the display name) and model/status
  #     reports that under its "camera_id" key. A read that echoed it would hand back
  #     "Display-Nine", which the POST rejects with STREAM_UNKNOWN.
  #     FREE A SLOT FIRST: max-batch-size is 4 and Cam-0..Cam-3 already fill it, so a
  #     5th add is QUEUED rather than activated -- it would never reach the routing
  #     table and the read would (correctly) not report it.
  remove_stream "${PREFIX}3" "${STREAM}"
  sleep "${DELAY}"
  add_stream "${PREFIX}9" "Display-Nine" "${STREAM}"
  echo "[test_44] Added camera_id='${PREFIX}9' with a DIFFERENT camera_name='Display-Nine'"
  sleep "${DELAY}"
  #     EVERY assertion below sits with the GET that produced the body it reads --
  #     assert_* inspects the LAST response, so interleaving another GET between a
  #     query and its assertion silently validates the wrong body.
  get_model_routing "stream=${PREFIX}9"
  assert_route_count 1 "test_44: ?stream= accepts the camera_id even when it differs from camera_name"
  assert_route_entry "${PREFIX}9" "state=serving" \
      "test_44: the response reports the camera_id ('${PREFIX}9'), NOT the display name"

  #     NEGATIVE: the display name must NOT resolve. Before the strict-filter fix this
  #     returned the stream whose real id is ${PREFIX}9.
  get_model_routing "stream=Display-Nine"
  assert_route_count 0 "test_44: ?stream=<camera_name> matches nothing -- ids only, never display names"

  # and the id it returned must be routable -- the round trip that actually matters
  model_routing "${PREFIX}9" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep "${DELAY}"
  get_model_routing "stream=${PREFIX}9"
  assert_route_entry "${PREFIX}9" "model.name=${CAR_MODEL},model.origin=pinned" \
      "test_44: that id round-trips -- routing by it succeeds and the read confirms it"
  remove_stream "${PREFIX}9" "${STREAM}"
  sleep "${DELAY}"

  # 12) PERCENT-DECODING, proved POSITIVELY. Asserting that an encoded UNKNOWN id
  #     returns nothing would pass even if decoding were completely broken, so use a
  #     stream whose REAL camera_id contains reserved characters and require the
  #     encoded query to find it. stream/add accepts any non-empty id.
  local _sp="cam 9&a=b"
  add_stream "${_sp}" "${_sp}" "${STREAM}"
  echo "[test_44] Added a camera_id containing space, '&' and '=': '${_sp}'"
  sleep "${DELAY}"
  get_model_routing "stream=cam%209%26a%3Db"
  assert_route_count 1 "test_44: percent-encoded ?stream= DECODES and finds the stream"
  assert_route_entry "${_sp}" "state=serving" \
      "test_44: ...and the entry returned is that exact camera_id, verbatim"
  #     '+' must decode to a space too (the other half of form encoding)
  get_model_routing "stream=cam+9%26a%3Db"
  assert_route_count 1 "test_44: '+' in ?stream= decodes to a space as well"
  #     the RAW (undecoded) form must NOT match -- proves the match is on decoded input
  get_model_routing "stream=cam%25209%2526a%253Db"
  assert_route_count 0 "test_44: double-encoded form matches nothing (decoding happens exactly once)"
  remove_stream "${_sp}" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_44] removed all streams (${PREFIX}3 went earlier, to free a slot for ${PREFIX}9)"

  sleep 10
  stop_app
  echo "-----------------------------------------------------------------------"
}

# test_45: `origin` and the default{} sweep -- the rule most likely to be misread.
#
# THE CLAIM UNDER TEST: `origin` is decided by HOW a lane got its model, never by
# WHAT the model is. So a stream EXPLICITLY routed to the very model the default
# already designates is "pinned", and a later default{} move SKIPS it -- the exact
# opposite of what "it's running the default model" suggests.
#
# Three streams, deliberately all on the SAME model to start with:
#   ${PREFIX}0, ${PREFIX}1  added plainly            -> fell back to the default  -> origin "default"
#   ${PREFIX}2         explicitly routed to that SAME name@version           -> origin "pinned"
#
# Before the move the three model refs are byte-identical apart from `origin`, so the
# split into two entries is caused by `origin` ALONE -- that is what makes this a real
# test of the field rather than of model identity. Then the default moves to a custom
# model and the read must show ${PREFIX}0/${PREFIX}1 carried across while ${PREFIX}2 stays put.
#
# (test_41 covers the same sweep from the ELEMENT's side via via_default/origin in
# model/status; this one asserts the model/routing GET reports it correctly.)
#
# Three streams rather than two so the test also proves the two default-followers stay
# MERGED in one entry across the sweep -- a sweep must not fragment a uniform group.
test_45() {
  echo "-----------------------------------------------------------------------"
  echo "[test_45] starting test 45 (model/routing GET: origin + default{} sweep)"

  # 1) config: a plain default primary, batch big enough that all three share an instance
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        3
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               3
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  3
  cfg_set multimodel                                      primary-model  "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION} engine         "${WORST_ENGINE}"

  start_app || return

  # 2) load the model the default will later MOVE TO (must be loaded first --
  #    routing never creates models)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_45] Loaded '${CAR_MODEL}@${CAR_MODEL_VERSION}' (the future default)"

  # 3) three streams, all landing on the default primary
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}"
  add_stream "${PREFIX}2" "${PREFIX}2" "${STREAM}" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  echo "[test_45] Added ${PREFIX}0..${PREFIX}2 (all resolving to Default Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}')"
  sleep "${DELAY}"


  # 4) BEFORE -- two entries whose model refs differ ONLY in origin
  get_model_routing
  assert_route_count 2 "test_45: origin alone splits the group (identical model, two entries)"
  assert_route_entry "${PREFIX}0,${PREFIX}1" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION},model.origin=default" \
      "test_45: ${PREFIX}0,${PREFIX}1 follow the default"
  assert_route_entry "${PREFIX}2" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION},model.origin=pinned" \
      "test_45: ${PREFIX}2 runs the SAME name@version but is pinned"
  assert_route_default "model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION}" \
      "test_45: default{} designates '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'"
  # the blast radius, countable BEFORE the move: 2 of the 3, not 3
  assert_route_sweep_count 2 "test_45: exactly 2 streams would be swept by a default move (${PREFIX}2 is pinned)"

  # 6) MOVE THE DEFAULT to the custom model
  model_routing_default "model" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_45] Moved the runtime default to '${CAR_MODEL}@${CAR_MODEL_VERSION}'"
  sleep "${DELAY}"
  sleep "${DELAY}"

  # 7) AFTER -- the default-followers were carried across; the pinned lane was not
  get_model_routing
  assert_route_default "model.name=${CAR_MODEL},model.version=${CAR_MODEL_VERSION}" \
      "test_45: default{} now designates '${CAR_MODEL}@${CAR_MODEL_VERSION}'"
  assert_route_count 2 "test_45: still two entries after the sweep (the followers stayed merged)"
  assert_route_entry "${PREFIX}0,${PREFIX}1" \
      "state=serving,model.name=${CAR_MODEL},model.version=${CAR_MODEL_VERSION},model.origin=default" \
      "test_45: SWEPT -- ${PREFIX}0,${PREFIX}1 moved to the new default, still origin 'default'"
  assert_route_entry "${PREFIX}2" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION},model.origin=pinned" \
      "test_45: UNTOUCHED -- ${PREFIX}2 kept its explicitly-routed model though it WAS the old default"
  assert_route_sweep_count 2 "test_45: the swept lanes are still the default-origin ones"
  assert_route_no_leak "test_45: refs still clean after a default move"

  # 8) the pinned lane is reported identically under a point lookup
  get_model_routing "stream=${PREFIX}2"
  assert_route_count 1 "test_45: ?stream=${PREFIX}2 returns one entry"
  assert_route_entry "${PREFIX}2" "model.name=${DEFAULT_MODEL},model.origin=pinned" \
      "test_45: filtered read agrees -- ${PREFIX}2 is still pinned to the old default model"

  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_45] removed ${PREFIX}0..${PREFIX}2"

  stop_app
  echo "-----------------------------------------------------------------------"
}

# test_46: the two states the other route-GET tests never reach -- `pending` and
# `passthrough`. Both are documented as carrying distinct operator meaning (WAIT vs
# ACT), and both were previously asserted only in prose.
#
#   pending      a stream is bound to a model that is STILL WARMING. Frames flow with
#                NO inference. The read must show state="pending" + pending.model and
#                NO model, then flip to state="serving" once the promote lands.
#                A pending->passthrough transition instead would mean the route FAILED
#                (target never warmed / was unloaded) -- the only in-band failure signal.
#
#   passthrough  model-sharding=0 and the per-model cap is full, so an extra stream
#                bypasses inference permanently: NO model and NO pending. Nothing is
#                coming; the operator must act.
#
# The warming window is made deterministic the same way test_2 does it -- delete the
# cached engine so the load is a full TRT rebuild, then add the stream immediately.
test_46() {
  echo "-----------------------------------------------------------------------"
  echo "[test_46] starting test 46 (model/routing GET: pending + passthrough states)"

  # 1) per-model cap 2 with sharding DISABLED -> the 3rd stream on that model
  #    becomes a permanent passthrough (this is test_27's mechanism).
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         0
  cfg_set multimodel           per-model-batch-size  2
  cfg_set multimodel           model-sharding        0
  comment_cfg    multimodel           shadow-model

  start_app || return

  # 2) force a LONG warm: remove the cached engine for THIS test's batch size so the
  #    load must rebuild it. -f so a re-run with it already gone does not error.
  #
  #    It is the b4 engine, NOT b2: per-model-batch-size caps streams per instance, it
  #    does NOT change the engine filename -- the model deserializes
  #    ..._b4_gpu0_fp16.engine regardless (confirmed in the nvinfer log). Moving the b2
  #    file aside therefore left the real engine cached, the load hit the cache, and the
  #    warming window this test depends on never existed.
  #
  #    Because that IS the shipped artifact, it is MOVED ASIDE and put back at the end --
  #    the rebuild writes a fresh engine to the same path, and the restore overwrites it
  #    with the original. test_2 deletes the same file outright; this test does not.
  #
  #    MOVE it aside rather than delete, and restore it at the end.
  #
  #    NO signal trap here on purpose. The suite installs its own
  #    `trap 'stop_multimodel' EXIT INT TERM`; adding RETURN/INT/TERM traps in a test
  #    would OVERWRITE that handler (and clearing them afterwards would remove it
  #    outright), so an interrupt would skip application cleanup. Recovery is made
  #    idempotent instead: a backup left behind by an aborted run is restored by the
  #    NEXT run, below, before anything else touches the engine.
  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test46.bak"
  if [ -f "${_bak}" ]; then
    echo "[test_46] restoring engine backup left by a previous interrupted run"
    mv -f "${_bak}" "${_eng}"
  fi
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  # NO sleep here -- the add must land INSIDE the rebuild window on every device.
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_46] Added ${PREFIX}0 bound to '${CAR_MODEL}@${CAR_MODEL_VERSION}' while it is still WARMING"
  sleep "${DELAY}"

  # 3) PENDING -- warming, so no inference is running on this stream yet
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=pending,pending.model.name=${CAR_MODEL},pending.model.version=${CAR_MODEL_VERSION}" \
      "test_46: warming stream reports state=pending with its intended target"
  assert_route_no_leak "test_46: refs clean while pending"

  # 4) once warm, the SAME stream must flip pending -> serving (promote landed).
  #    pending -> passthrough instead would mean the route FAILED.
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep "${DELAY}"
  sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=serving,model.name=${CAR_MODEL},model.version=${CAR_MODEL_VERSION}" \
      "test_46: promote landed -- pending -> serving, now reporting the model"

  # 5) fill the model to its cap (2), then overflow it. With model-sharding=0 the
  #    third stream on that model gets NO model and NO pending: permanent passthrough.
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep "${DELAY}"
  add_stream "${PREFIX}2" "${PREFIX}2" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_46] Added ${PREFIX}1 (fills the cap) and ${PREFIX}2 (over cap -> passthrough)"
  sleep "${DELAY}"

  get_model_routing
  assert_route_entry "${PREFIX}2" "state=passthrough" \
      "test_46: over-cap stream reports state=passthrough (no model, none coming)"
  assert_route_no_refs "${PREFIX}2" \
      "test_46: the passthrough entry carries NEITHER model NOR pending -- operator must ACT"
  assert_route_deterministic "test_46: mixed serving/passthrough snapshot is still byte-stable"

  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}2" "${STREAM}"
  echo "[test_46] removed ${PREFIX}0..${PREFIX}2"

  # put the cached engine back (the rebuild produced a fresh one; keep the ORIGINAL)
  if [ -f "${_bak}" ]; then
    mv -f "${_bak}" "${_eng}"
    echo "[test_46] restored the original b4 engine"
  fi

  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 47: model/routing GET -- an ACCEPTED but still-DEFERRED route is VISIBLE.
#
# The gap this closes: a POST /api/v1/model/routing naming a model that is still WARMING
# is ACCEPTED (202) and parked on the control layer's pending queue; the stream keeps
# serving its OLD model until the target reaches READY. Nothing in the read plane used
# to change across that window -- model/status (and the GET projected from it) reported
# the old model and nothing else -- so a client polling the API could not distinguish an
# in-flight route from one that had been dropped on the floor.
#
# DISTINCT FROM TEST 46. That one reaches "pending" through stream/add on a warming
# model: a PASSTHROUGH stream with no lane yet, so state IS "pending" and there is no
# model to report. This one reaches it through the POST, on a stream that is already
# SERVING -- so the entry must carry `model` AND `pending` at the same time and the
# state must stay "serving". Collapsing the two would lose the difference between
# "switched" and "about to switch".
#
#   1. cam-0, cam-1 serve the default primary -> one merged entry, no pending anywhere.
#   2. Car's cached engine is moved aside so model/load must REBUILD (a long warm).
#   3. POST route cam-0 -> Car@1 lands inside that window -> accepted, DEFERRED.
#   4. GET: cam-0 = serving <default> + pending Car@1 (origin "pinned"); cam-1 untouched
#      and carrying NO pending; the group splits because pending is part of the key.
#   5. Once Car warms the deferred apply lands: cam-0 = serving Car@1, pending GONE.
test_47() {
  echo "-----------------------------------------------------------------------"
  echo "[test_47] starting test 47 (model/routing GET: accepted-but-deferred route)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  4
  cfg_set multimodel           primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  comment_cfg multimodel       shadow-model
  # Pin the DEFAULT to its OWN engine file. Car and the default otherwise both resolve
  # to ..._b4_gpu0_fp16.engine, and step 2 moves that file aside -- without this pin the
  # default would be starved too and the streams would never reach "serving" at all.
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}  engine  "${WORST_ENGINE}"

  start_app || return

  # 1) BASELINE -- two streams on the default primary, merged, nothing pending.
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}"
  echo "[test_47] Added ${PREFIX}0,${PREFIX}1 on the Default Primary '${DEFAULT_MODEL}@${DEFAULT_MODEL_VERSION}'"
  sleep "${DELAY}"

  get_model_routing
  assert_route_count 1 "test_47: baseline -- both streams merge into one entry"
  assert_route_entry "${PREFIX}0,${PREFIX}1" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION}" \
      "test_47: baseline entry serves the default primary"
  assert_route_entry_lacks "${PREFIX}0,${PREFIX}1" "pending" \
      "test_47: nothing queued yet -- the baseline entry carries no pending key"

  # 2) force a LONG warm for Car: move its cached engine aside so the load must
  #    rebuild. Same mechanism (and same idempotent recovery) as test_46 -- see the
  #    long note there for why this is MOVED and restored rather than deleted, and
  #    why no signal trap is installed here.
  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test47.bak"
  if [ -f "${_bak}" ]; then
    echo "[test_47] restoring engine backup left by a previous interrupted run"
    mv -f "${_bak}" "${_eng}"
  fi
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # 3) load Car (async -> rebuild) and route cam-0 onto it IMMEDIATELY. NO sleep
  #    between the two: the POST has to land while Car is still WARMING, which is what
  #    makes the control layer DEFER it instead of applying it on the spot.
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_47] Routed ${PREFIX}0 -> '${CAR_MODEL}@${CAR_MODEL_VERSION}' while it is still WARMING (expect DEFERRED)"
  assert_log "route accepted -> warming target" \
      "test_47: the POST was ACCEPTED and deferred (not applied, not rejected)" 12

  # 4) THE READ SHOWS THE INTENT -- while still serving the old model.
  sleep 2
  get_model_routing
  assert_route_count 2 "test_47: the queued route splits the group (pending is part of the key)"
  assert_route_entry "${PREFIX}0" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION}" \
      "test_47: ${PREFIX}0 is STILL SERVING the old model -- a deferred route drops no frames"
  assert_route_entry "${PREFIX}0" \
      "pending.model.name=${CAR_MODEL},pending.model.version=${CAR_MODEL_VERSION},pending.model.origin=pinned" \
      "test_47: ...and reports where it is going, as an explicit (pinned) route"
  assert_route_entry_lacks "${PREFIX}1" "pending" \
      "test_47: ${PREFIX}1 was never in scope -- no pending leaks onto it"
  assert_route_no_leak "test_47: refs still clean while a route is queued"
  assert_route_deterministic "test_47: a snapshot carrying pending is still byte-stable"

  # the point lookup must carry it too -- a reconciler polls ONE stream, not the fleet
  get_model_routing "stream=${PREFIX}0"
  assert_route_count 1 "test_47: ?stream= returns exactly the queried stream"
  assert_route_entry "${PREFIX}0" \
      "state=serving,pending.model.name=${CAR_MODEL}" \
      "test_47: the filtered read reports the queued route too"

  # 4b) the model plane must not ORPHAN the intent. streams[] now names Car as the
  #     pending target, so a ?stream= filtered model/status has to describe Car in
  #     models[] too -- otherwise the response references a version it never lists
  #     and a reconciler cannot resolve what it is waiting for.
  get_model_status "stream=${PREFIX}0"
  assert_status_model_listed "${CAR_MODEL}" \
      "test_47: ?stream= JOIN lists the PENDING target in models[] (no orphan ref)"

  # 4c) COMPLETION EVENT TIMING -- the route has not been applied, so the app must
  #     NOT have been told it was. The POST answered 202; `streams routed` is the
  #     "done" signal and firing it here would announce a switch that has not
  #     happened. deepstream-app renders the event with '-' turned into ' '.
  # NOTE the exact marker: deepstream-app turns "streams-routed" into "streams
  # routed" and only prefixes "new " for model-loaded, so "new streams routed"
  # would never appear -- an absence check on it would pass vacuously.
  # match the RENDERED event ("streams routed [streams=...]"), NOT the POST
  # reason string, which literally quotes "'streams routed' when done" and
  # would make this fire on the accepted response itself.
  assert_log_absent "streams routed [streams=" \
      "test_47: no completion event while the route is still DEFERRED"

  # 5) once Car warms, the deferred apply lands and the intent is CONSUMED.
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep "${DELAY}"
  sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=serving,model.name=${CAR_MODEL},model.version=${CAR_MODEL_VERSION},model.origin=pinned" \
      "test_47: deferred apply landed -- ${PREFIX}0 now SERVES the target it was queued for"
  assert_route_entry_lacks "${PREFIX}0" "pending" \
      "test_47: the pending key is GONE once applied (not a stale 'forever pending')"
  assert_route_entry "${PREFIX}1" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION}" \
      "test_47: ${PREFIX}1 still on the default primary throughout"
  # ...and NOW the completion event is true, so it must have been posted (by the
  # pending poller, not the POST handler).
  assert_log "streams routed [streams=" \
      "test_47: completion event arrives only AFTER the deferred apply lands" 15

  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  echo "[test_47] removed ${PREFIX}0,${PREFIX}1"

  # put the cached engine back (the rebuild produced a fresh one; keep the ORIGINAL)
  if [ -f "${_bak}" ]; then
    mv -f "${_bak}" "${_eng}"
    echo "[test_47] restored the original b4 engine"
  fi

  stop_app
  echo "-----------------------------------------------------------------------"
}


# Shared setup for 48/49: two streams serving the default, with Car AND Person both
# forced into a long warm. All three models resolve to the SAME cached engine
# (..._b4_gpu0_fp16.engine), so moving that one file aside starves both custom
# models at once while the default keeps running on its own pinned engine.
# Echoes nothing; leaves ${_eng}/${_bak} set by the caller.
_route_defer_setup () {
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        4
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  4
  cfg_set multimodel           primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  comment_cfg multimodel       shadow-model
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}  engine  "${WORST_ENGINE}"
}

# Test 48: model/routing GET -- a NEWER deferred route SUPERSEDES an older one.
#
# The bug this guards: the deferred queue was append-only, so two routes for the
# same stream both stayed live and WARM ORDER decided the winner. Route S to
# warming B, then to warming C; if C warmed first it applied, then B warmed and
# OVERWROTE it -- the stream landed on the model the operator had already
# replaced. The read compounded it: status reports the newest entry, so the GET
# promised C while the apply could deliver B.
#
# The assertion that matters is the LAST one: after both models finish warming,
# the stream must serve C. A pass here means the queue holds one entry per
# stream, so warm order cannot decide the outcome.
test_48() {
  echo "-----------------------------------------------------------------------"
  echo "[test_48] starting test 48 (model/routing: newer deferred route supersedes older)"

  _route_defer_setup
  start_app || return

  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}"
  echo "[test_48] Added ${PREFIX}0,${PREFIX}1 on the Default Primary"
  sleep "${DELAY}"

  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test48.bak"
  if [ -f "${_bak}" ]; then
    echo "[test_48] restoring engine backup left by a previous interrupted run"
    mv -f "${_bak}" "${_eng}"
  fi
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # 1) FIRST deferred route: cam-0 -> Car@1, still warming.
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_48] Queued ${PREFIX}0 -> '${CAR_MODEL}@1' (warming)"
  sleep 2
  get_model_routing
  assert_route_entry "${PREFIX}0" "pending.model.name=${CAR_MODEL}" \
      "test_48: first route queued -- pending reports ${CAR_MODEL}"

  # 2) SECOND deferred route for the SAME stream, before the first can apply.
  load_model "${PERSON_MODEL}" "1" "" "" "False"
  model_routing "${PREFIX}0" "${PERSON_MODEL}" "1"
  echo "[test_48] Queued ${PREFIX}0 -> '${PERSON_MODEL}@1' -- must SUPERSEDE the ${CAR_MODEL} route"
  assert_log "SUPERSEDED" \
      "test_48: the older queued route is explicitly superseded, not left to race" 12
  sleep 2

  # 3) the read must show ONLY the freshest intent -- one entry, one target.
  get_model_routing
  assert_route_entry "${PREFIX}0" "pending.model.name=${PERSON_MODEL}" \
      "test_48: pending now reports the NEWER target (${PERSON_MODEL})"
  assert_route_entry_lacks "${PREFIX}1" "pending" \
      "test_48: the untargeted stream still carries no pending"
  assert_route_deterministic "test_48: snapshot stable while two routes were issued"

  # 4) THE POINT. Both models finish warming -- in whatever order. The stream must
  #    end on the model the operator asked for LAST, never on the superseded one.
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  wait_for_model "${PERSON_MODEL}" "1"
  sleep "${DELAY}"
  sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=serving,model.name=${PERSON_MODEL},model.version=1,model.origin=pinned" \
      "test_48: FRESHEST intent won -- serving ${PERSON_MODEL}, NOT the superseded ${CAR_MODEL}"
  assert_route_entry_lacks "${PREFIX}0" "pending" \
      "test_48: no stale pending left behind by the superseded route"

  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  if [ -f "${_bak}" ]; then
    mv -f "${_bak}" "${_eng}"
    echo "[test_48] restored the original b4 engine"
  fi
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 49: model/routing GET -- a deferred SHADOW-CLEAR is distinguishable.
#
# `shadow` is TRI-state in the write: omitted = leave it alone, {name,version} =
# set it, null = clear it. While a route waits on a warming primary, the read has
# to preserve all three. Encoding "clear" as an absent key -- the obvious
# shortcut -- makes "reroute the primary" and "reroute it AND drop the A/B"
# produce byte-identical snapshots, so an operator cannot tell whether their
# shadow model survives the switch.
test_49() {
  echo "-----------------------------------------------------------------------"
  echo "[test_49] starting test 49 (model/routing GET: deferred shadow-clear is explicit)"

  _route_defer_setup
  # this one NEEDS a shadow to clear, so re-enable it (the shared setup drops it)
  cfg_set multimodel  shadow-model  "${DEFAULT_MODEL};${DEFAULT_SHADOW_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_SHADOW_MODEL_VERSION}  engine  "${BEST_ENGINE}"

  start_app || return

  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}"
  echo "[test_49] Added ${PREFIX}0,${PREFIX}1 with a default primary AND shadow"
  sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0,${PREFIX}1" \
      "shadow.name=${DEFAULT_MODEL},shadow.version=${DEFAULT_SHADOW_MODEL_VERSION}" \
      "test_49: baseline -- both streams carry the default shadow"

  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test49.bak"
  if [ -f "${_bak}" ]; then
    echo "[test_49] restoring engine backup left by a previous interrupted run"
    mv -f "${_bak}" "${_eng}"
  fi
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # cam-0: reroute the primary to a WARMING model AND clear the shadow.
  # cam-1: reroute the primary to the same warming model, shadow UNTOUCHED.
  # The two differ ONLY in the shadow's fate, so if pending cannot express a
  # clear, their entries merge and the assertions below cannot both hold.
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "none"
  model_routing "${PREFIX}1" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_49] Queued ${PREFIX}0 (shadow=none) and ${PREFIX}1 (shadow untouched), both -> warming '${CAR_MODEL}'"
  sleep 2

  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=serving,pending.model.name=${CAR_MODEL}" \
      "test_49: ${PREFIX}0 reports its pending primary while still serving"
  assert_route_entry_null "${PREFIX}0" "pending.shadow" \
      "test_49: a pending CLEAR is reported as an explicit null shadow (present, not omitted)"
  assert_route_entry_lacks "${PREFIX}1" "pending.shadow" \
      "test_49: a pending route that does NOT touch the shadow omits the key entirely"
  assert_route_no_leak "test_49: refs clean across a deferred shadow-clear"

  # the two streams must NOT have merged -- that is the whole point of the split
  assert_route_entry "${PREFIX}1" "pending.model.name=${CAR_MODEL}" \
      "test_49: ${PREFIX}1 is its OWN entry -- clear vs unchanged did not collapse together"

  remove_stream "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"
  remove_stream "${PREFIX}1" "${STREAM}"
  if [ -f "${_bak}" ]; then
    mv -f "${_bak}" "${_eng}"
    echo "[test_49] restored the original b4 engine"
  fi
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 50: a newer route that applies IMMEDIATELY must supersede an older pending one.
#
# The regression: superseding was scoped to the newer route's DEFERRED branch, so a
# newer route to an ALREADY-LOADED model -- which applies on the spot -- left the
# stale entry queued. When its own target warmed later, the poller dragged the
# stream back off the model that was already serving. This is the common case
# (routing to a loaded model), and test 48 misses it entirely: 48 queues two
# DEFERRED routes, so it only ever exercises the defer path.
test_50() {
  echo "-----------------------------------------------------------------------"
  echo "[test_50] starting test 50 (immediate route supersedes an older PENDING one)"
  _route_defer_setup
  start_app || return
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  add_stream "${PREFIX}1" "${PREFIX}1" "${STREAM}"
  sleep "${DELAY}"

  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test50.bak"
  [ -f "${_bak}" ] && mv -f "${_bak}" "${_eng}"
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # 1) queue a route onto a WARMING model
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep 2
  get_model_routing
  assert_route_entry "${PREFIX}0" "pending.model.name=${CAR_MODEL}" \
      "test_50: route to the warming model is queued"

  # 2) now route the SAME stream back onto the DEFAULT, which is already serving.
  #    This applies IMMEDIATELY -- no defer -- and must still cancel the queued one.
  model_routing "${PREFIX}0" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  echo "[test_50] Routed ${PREFIX}0 back to the READY default -- applies at once, must supersede"
  sleep 2
  get_model_routing
  assert_route_entry_lacks "${PREFIX}0" "pending" \
      "test_50: the immediate route CANCELLED the queued one (no pending left)"

  # 3) THE POINT: the superseded target finishes warming. Nothing may move.
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep "${DELAY}"; sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=serving,model.name=${DEFAULT_MODEL},model.version=${DEFAULT_MODEL_VERSION}" \
      "test_50: ${CAR_MODEL} warming later did NOT drag the stream off the applied route"

  remove_stream "${PREFIX}0" "${STREAM}"; sleep "${DELAY}"; remove_stream "${PREFIX}1" "${STREAM}"
  [ -f "${_bak}" ] && mv -f "${_bak}" "${_eng}"
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 51: superseding is per LANE, not per stream.
#
# A route is partial -- an omitted `model` means "leave the primary alone", an
# omitted `shadow` likewise. So a newer SHADOW-only route must not cancel an older
# pending PRIMARY change. Withdrawing the stream from the whole older entry (the
# first implementation) silently dropped the primary intent, and the operator's
# first request vanished with no error anywhere.
test_51() {
  echo "-----------------------------------------------------------------------"
  echo "[test_51] starting test 51 (lane-aware supersede: shadow-only route keeps pending primary)"
  _route_defer_setup
  start_app || return
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  sleep "${DELAY}"

  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test51.bak"
  [ -f "${_bak}" ] && mv -f "${_bak}" "${_eng}"
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # 1) pending PRIMARY change onto a warming model
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  model_routing "${PREFIX}0" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  sleep 2
  get_model_routing
  assert_route_entry "${PREFIX}0" "pending.model.name=${CAR_MODEL}" \
      "test_51: primary lane change is queued"

  # 2) a SHADOW-only route for the same stream (no "model" field at all)
  load_model "${PERSON_MODEL}" "1" "" "" "False"
  model_routing "${PREFIX}0" "" "" "${PERSON_MODEL}" "1"
  echo "[test_51] Queued a SHADOW-only route -- the pending PRIMARY must survive it"
  sleep 2

  # 3) both intents must be visible: the untouched lane was not collateral damage
  get_model_routing
  assert_route_entry "${PREFIX}0" "pending.model.name=${CAR_MODEL}" \
      "test_51: the pending PRIMARY survived a shadow-only supersede"
  assert_route_entry "${PREFIX}0" "pending.shadow.name=${PERSON_MODEL}" \
      "test_51: ...and the newer pending SHADOW is reported alongside it"

  remove_stream "${PREFIX}0" "${STREAM}"
  [ -f "${_bak}" ] && mv -f "${_bak}" "${_eng}"
  stop_app
  echo "-----------------------------------------------------------------------"
}


# Test 52: a PASSTHROUGH stream that receives a newer deferred route.
#
# Covers two fixes that share one setup:
#
#  (a) A stream can hold TWO intents at once -- an older promote tag written when
#      it attached as passthrough onto a warming model, and a newer deferred ROUTE
#      queued after it. model/status used to read the tag exclusively for
#      passthrough streams, so GET reported the target the operator had ALREADY
#      replaced. The mirror must outrank the tag.
#
#  (b) When that deferred route finally applies to a passthrough stream, the
#      reroute cannot switch a lane that does not exist -- it only tags the stream
#      and enqueues a promote, which lands a tick later. The completion event must
#      wait for the promote, not fire on the reroute.
#
# Car and Person both resolve to the same cached engine, so moving that one file
# aside keeps BOTH warming long enough for the window to exist.
test_52() {
  echo "-----------------------------------------------------------------------"
  echo "[test_52] starting test 52 (passthrough stream + newer deferred route)"
  _route_defer_setup
  start_app || return

  local _eng="/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine"
  local _bak="${_eng}.test52.bak"
  [ -f "${_bak}" ] && mv -f "${_bak}" "${_eng}"
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # 1) attach the stream BOUND to a warming model -> it runs PASSTHROUGH carrying a
  #    promote tag naming that model (this is test_46's mechanism).
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" "" "" "False"
  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}" "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  echo "[test_52] Added ${PREFIX}0 bound to warming '${CAR_MODEL}' -> passthrough + promote tag"
  sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=pending,pending.model.name=${CAR_MODEL}" \
      "test_52: baseline -- the passthrough stream reports its promote tag"

  # 2) now route the SAME stream somewhere else, also still warming. The promote
  #    tag still says ${CAR_MODEL}; the freshest intent is ${PERSON_MODEL}.
  load_model "${PERSON_MODEL}" "1" "" "" "False"
  model_routing "${PREFIX}0" "${PERSON_MODEL}" "1"
  echo "[test_52] Routed ${PREFIX}0 -> warming '${PERSON_MODEL}' while the promote tag still names '${CAR_MODEL}'"
  sleep 2

  # (a) THE READ MUST FOLLOW THE NEWER INTENT, not the stale tag.
  get_model_routing
  assert_route_entry "${PREFIX}0" "pending.model.name=${PERSON_MODEL}" \
      "test_52: the newer queued route OUTRANKS the older promote tag"
  assert_route_no_leak "test_52: refs clean with both intents live"

  # (b) nothing has attached inference yet, so no completion may have been claimed.
  assert_log_absent "streams routed [streams=" \
      "test_52: no completion event while the stream is still passthrough"

  # 3) once the target warms the reroute applies, and because the stream is
  #    passthrough that only ENQUEUES a promote -- the promote itself lands a tick
  #    later. Either way the stream must end up SERVING the newer model.
  wait_for_model "${PERSON_MODEL}" "1"
  sleep "${DELAY}"; sleep "${DELAY}"; sleep "${DELAY}"
  get_model_routing
  assert_route_entry "${PREFIX}0" \
      "state=serving,model.name=${PERSON_MODEL},model.version=1" \
      "test_52: promote landed -- serving the model the LAST route asked for"
  assert_route_entry_lacks "${PREFIX}0" "pending" \
      "test_52: both intents consumed -- no stale pending survives"

  remove_stream "${PREFIX}0" "${STREAM}"
  [ -f "${_bak}" ] && mv -f "${_bak}" "${_eng}"
  stop_app
  echo "-----------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Test 53: the UNIFIED STATUS CONTRACT -- every reject class, pinned exactly.
#
# Every other negative assertion in this suite is satisfied by ANY non-2xx, so
# 400, 404 and 409 are one bucket to them. That is exactly what the unified API
# distinguishes, which means the suite could not tell a correct status remap
# from no remap at all. This case pins the number AND the machine-readable code,
# so a regression in either fails instead of silently passing.
#
#   400 -- the request itself is bad (unknown field, invalid version)
#   404 -- the model is not loaded / does not exist
#   409 -- lifecycle or routing-state conflict (in use, stale revision)
# ---------------------------------------------------------------------------
test_53() {
  echo "-----------------------------------------------------------------------"
  echo "[test_53] starting test 53 (unified status contract: 400 / 404 / 409)"

  cfg_set sink0            type                  ${SINK_TYPE}
  cfg_set source-list      max-batch-size        4
  cfg_set multimodel       unified-batch         1
  cfg_set multimodel       per-model-batch-size  4
  cfg_set multimodel       primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  comment_cfg multimodel   shadow-model

  start_app || return

  add_stream "${PREFIX}0" "${PREFIX}0" "${STREAM}"
  wait_for_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}" 60

  # --- 404: model plane, for a name that was never loaded -------------------
  unload_model "NoSuchModel" ""
  assert_http_status 404 "test_53: unload of an unknown model is 404 (was 400)"
  assert_error_code  "MODEL_NOT_LOADED" "test_53: ...and carries MODEL_NOT_LOADED"

  # --- 409: unloading a model that is the live default / serving streams ----
  unload_model "${DEFAULT_MODEL}" ""
  assert_http_status 409 "test_53: unload of an in-use model is 409 (was 400)"
  assert_error_code  "MODEL_IN_USE" "test_53: ...and carries MODEL_IN_USE"

  # --- 404: routing plane, for a target that is not loaded ------------------
  model_routing "${PREFIX}0" "NoSuchModel" "1"
  assert_http_status 404 "test_53: routing to an unloaded model is 404"
  assert_error_code  "MODEL_NOT_LOADED" "test_53: ...routing reject carries the CAUSE, not the generic bucket"

  # --- 409: a stale compare-and-set guard ----------------------------------
  _rest_post "${ROUTE_URL}" "{
      \"key\": \"routing\",
      \"value\": { \"if_revision\": 999999,
                   \"routes\": [ { \"streams\": [\"${PREFIX}0\"],
                                   \"model\": { \"name\": \"${DEFAULT_MODEL}\",
                                                \"version\": \"${DEFAULT_MODEL_VERSION}\" } } ] }
    }" "route_stale_revision"
  assert_http_status 409 "test_53: a stale if_revision is 409 (was 400)"
  assert_error_code  "REVISION_CONFLICT" "test_53: ...and carries REVISION_CONFLICT"

  # --- 400: the request itself is malformed --------------------------------
  _rest_post "${LOAD_URL}" "{
      \"key\": \"model\",
      \"value\": { \"name\": \"X\", \"version\": \"1\", \"banana\": 1 }
    }" "load_unknown_field"
  assert_http_status 400 "test_53: an unknown request field stays 400"
  assert_error_code  "FIELD_UNKNOWN" "test_53: ...and carries FIELD_UNKNOWN"
  assert_rest_body_contains '"field":"value.banana"' "test_53: ...and names the offending field"

  # --- 400 FIELD_UNSUPPORTED: a real unified field this profile does not do --
  _rest_post "${ROUTE_URL}" "{
      \"key\": \"routing\",
      \"value\": { \"default\": { \"model\": {\"name\": \"${DEFAULT_MODEL}\", \"version\": \"${DEFAULT_MODEL_VERSION}\"},
                                 \"canary\": {\"name\": \"${DEFAULT_MODEL}\", \"version\": \"${DEFAULT_MODEL_VERSION}\"},
                                 \"canary_percent\": 10 } }
    }" "route_canary_unsupported"
  assert_http_status 400 "test_53: an aoi-only field is 400"
  assert_error_code  "FIELD_UNSUPPORTED" "test_53: ...and is FIELD_UNSUPPORTED, not FIELD_UNKNOWN (it is a real field, wrong profile)"

  # --- 400 FIELD_UNSUPPORTED: the envelope discriminator is enforced ---------
  _rest_post "${ROUTE_URL}" "{
      \"key\": \"stream\",
      \"value\": { \"routes\": [ { \"streams\": [\"${PREFIX}0\"],
                                   \"model\": { \"name\": \"${DEFAULT_MODEL}\",
                                                \"version\": \"${DEFAULT_MODEL_VERSION}\" } } ] }
    }" "route_retired_key"
  assert_http_status 400 "test_53: the retired key \"stream\" is now rejected"
  assert_error_code  "FIELD_UNSUPPORTED" "test_53: ...with FIELD_UNSUPPORTED"

  # --- the contract's model{} + state, on a SUCCESSFUL op ------------------
  #     (only on success: naming a model in a reject would read as though
  #      something had happened to it)
  load_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}"
  assert_http_status 202 "test_53: a fresh load is accepted"
  assert_rest_body_contains '"state":"accepted"' "test_53: ...and echoes the resulting state"
  assert_rest_body_contains '"name":"Car"' "test_53: ...and which model it acted on"

  # --- the two planes stay separated ---------------------------------------
  # model/status answers "what is loaded and where"; the FLEET routing policy is
  # model/routing's to report. A fleet-level routing summary here would be a third
  # copy of a fact the response already carries twice -- as default{}, and per
  # version as default_for -- and three copies cannot be kept in agreement.
  get_model_status
  assert_status_not_contains '"routing":{' \
      "test_53: model/status carries NO fleet routing summary (that plane is model/routing's)"
  assert_status_contains '"default":{' \
      "test_53: ...the default designation itself is still reported"

  # --- a successful routing write reports its state ------------------------
  model_routing "${PREFIX}0" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_http_status 202 "test_53: a valid route is accepted"
  assert_rest_body_contains '"state":"applying"' \
      "test_53: ...and reports state=applying (admitted here, applied on the main loop)"

  # --- 404: routing to a camera that does not exist ------------------------
  model_routing "no-such-cam" "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_http_status 404 "test_53: routing to an unknown stream is 404 (was 400)"
  assert_error_code  "STREAM_UNKNOWN" "test_53: ...and carries STREAM_UNKNOWN"

  # --- 409: re-loading a model that is already loaded ----------------------
  load_model "${DEFAULT_MODEL}" "${DEFAULT_MODEL_VERSION}"
  assert_http_status 409 "test_53: re-loading an already-loaded model is 409"
  assert_error_code  "MODEL_ALREADY_LOADED" "test_53: ...and carries MODEL_ALREADY_LOADED"

  # --- 404: updating a model absent from the catalog -----------------------
  #     engine_file is REQUIRED by the parser, so it must be supplied here or the
  #     request is rejected as malformed (400 GPU_ENGINE_MISSING) before it ever
  #     reaches the catalog lookup this case is about.
  _rest_post "${MODEL_UPDATE_URL}" "{
      \"key\": \"model\",
      \"value\": { \"name\": \"NoSuchModel\", \"from_version\": \"1\", \"version\": \"2\",
                   \"engine_file\": \"${WORST_ENGINE}\" }
    }" "update_absent_model"
  assert_http_status 404 "test_53: updating a model not in the catalog is 404"
  assert_error_code  "MODEL_NOT_IN_CATALOG" "test_53: ...and carries MODEL_NOT_IN_CATALOG"

  # --- 409: the update precondition names a version that is not live -------
  wait_for_model "${CAR_MODEL}" "${CAR_MODEL_VERSION}" 60
  _rest_post "${MODEL_UPDATE_URL}" "{
      \"key\": \"model\",
      \"value\": { \"name\": \"${CAR_MODEL}\", \"from_version\": \"99\", \"version\": \"100\",
                   \"engine_file\": \"${BEST_ENGINE}\" }
    }" "update_from_not_live"
  assert_http_status 409 "test_53: a stale from_version is 409 -- a failed compare-and-set, not an absence"
  assert_error_code  "UPDATE_FROM_MISMATCH" "test_53: ...and carries UPDATE_FROM_MISMATCH"

  # --- 409: a second op arriving while an update awaits nvinfer's confirm --
  #     Fired back-to-back with NO settle on purpose: the in-flight window is the
  #     engine reload, and the point of the case is to land inside it. If the
  #     confirm beats the second call the window is simply missed -- see the
  #     conditional below, which reports rather than failing on a lost race.
  _rest_post "${MODEL_UPDATE_URL}" "{
      \"key\": \"model\",
      \"value\": { \"name\": \"${CAR_MODEL}\", \"from_version\": \"${CAR_MODEL_VERSION}\",
                   \"version\": \"7\", \"engine_file\": \"${BEST_ENGINE}\" }
    }" "update_trigger_inflight"
  #   RETRY the race rather than skipping it. The earlier form reported "window
  #   missed" and moved on, which meant a REGRESSION -- 400, 202, or the wrong
  #   error code -- was indistinguishable from a lost race and passed silently.
  #   The window is an engine reload, so a handful of attempts is ample; only if
  #   every attempt misses do we report rather than fail, because a lost race is
  #   genuinely not a defect.
  #   The live identity moves with each accepted update, so a re-arm must name the
  #   version that is CURRENTLY live -- reusing the original from_version would be
  #   rejected with UPDATE_FROM_MISMATCH and would never reopen the window.
  _mm_inflight_seen=0
  _mm_from="${CAR_MODEL_VERSION}"
  [ "${MM_REST_POST_CODE}" = "202" ] && _mm_from="7"   # the trigger above moved it
  for _mm_try in 1 2 3 4 5; do
    load_model "${CAR_MODEL}" "$((8 + _mm_try))"
    if [ "${MM_REST_POST_CODE}" = "409" ]; then _mm_inflight_seen=1; break; fi
    # not the in-flight window -- re-arm from whatever is live right now
    _mm_to="$((20 + _mm_try))"
    _rest_post "${MODEL_UPDATE_URL}" "{
        \"key\": \"model\",
        \"value\": { \"name\": \"${CAR_MODEL}\", \"from_version\": \"${_mm_from}\",
                     \"version\": \"${_mm_to}\", \"engine_file\": \"${BEST_ENGINE}\" }
      }" "update_rearm_${_mm_try}"
    # advance only on an accepted re-arm; a rejected one leaves the identity put
    [ "${MM_REST_POST_CODE}" = "202" ] && _mm_from="${_mm_to}"
  done
  if [ "${_mm_inflight_seen}" = "1" ]; then
    assert_http_status 409 "test_53: an op racing an in-flight update is 409"
    assert_error_code  "UPDATE_IN_FLIGHT" "test_53: ...and carries UPDATE_IN_FLIGHT"
  else
    _fail "test_53: never observed the UPDATE_IN_FLIGHT window in 5 attempts -- either the race is unreachable here or the 409 classification regressed"
  fi

  # --- 400: a value that fails validation ----------------------------------
  _rest_post "${LOAD_URL}" "{
      \"key\": \"model\",
      \"value\": { \"name\": \"X\", \"version\": \"1.0.0\" }
    }" "load_bad_version"
  assert_http_status 400 "test_53: an invalid version stays 400"
  assert_error_code  "VERSION_INVALID" "test_53: ...and carries VERSION_INVALID"

  stop_app
}


# Test 54: model/unload admission is scoped to the model IDENTITY, not the model NAME.
#
# A model instance is identified by (name, version) -- the canonical key "name@version" --
# and every unload rule is decided on that identity. Designating ONE version as a role
# default therefore protects THAT version only: the other versions of the same name stay
# independently unloadable, so an idle A/B candidate can be reclaimed without disturbing
# the default. Two rules can refuse an unload (the target is SERVING, or the target is a
# configured default); each refusal must name the rule that actually applied, since a
# client is expected to branch on it.
#
# Flow:
#   1. load Trafficcamnet@1 and designate it the runtime default primary.
#   2. add Cam-1 + Cam-2 -> both serve on the default.
#   3. load Trafficcamnet@2 (a second version of the SAME name).
#   4. route shadow=Trafficcamnet@2 for all streams -> @2 is SERVING.
#   5. clear the shadow -> @2 drains to idle (0 streams), still not a default.
#   6. unload Trafficcamnet@2.
#
# What to validate:
#   - while @2 is SERVING the unload is refused BECAUSE it is serving, and the response
#     says so: a sibling version holding the default is not what decides this.
#   - once @2 is idle the unload SUCCEEDS (202), even though its sibling @1 is the
#     default. The response must not carry the configured-default wording -- a 202 alone
#     would not show WHICH rule ran.
#   - @2 is really gone from the pool (engine freed), and @1 is untouched: still loaded,
#     still the default, still serving both streams.
#   - @1, the actual default, stays protected, and a version-less unload of the name is
#     still refused by the name-level rule.
test_54() {
  echo "-----------------------------------------------------------------------"
  echo "[test_54] starting test 54 (unload admission scoped to model IDENTITY, not model NAME)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  # test5 config updates
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # multimodel config updates: unified A/B batch (primary row 0, shadow row 1). No config
  # defaults -- the runtime default primary is established below by load+route-default
  # over the REST control plane.
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load version 1 and make it the RUNTIME DEFAULT primary (def_primary=true waits for
  #    the warm, then routes default{model} -- the two REST calls that designation takes)
  load_model "${DEFAULT_MODEL}" "1" "${DEFAULT_ENGINE}" "" true
  echo "[test_54] loaded '${DEFAULT_MODEL}@1' and designated it the default primary"
  assert_status_version_eventually "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "1" \
      "test_54: '${DEFAULT_MODEL}@1' is loaded"

  # 4) add both streams -- they resolve to the default (@1)
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_54] added Cam-1 + Cam-2 (both on the default primary '${DEFAULT_MODEL}@1')"
  sleep "${DELAY}"
  get_model_status "model=${DEFAULT_MODEL}"
  assert_status_ref "model" "${DEFAULT_MODEL}" "1" \
      "test_54: streams are serving on the default '${DEFAULT_MODEL}@1' (precondition)"

  # 5) load a SECOND VERSION of the SAME name -- the model this case is about
  load_model "${DEFAULT_MODEL}" "2" "${BEST_ENGINE}"
  wait_for_model "${DEFAULT_MODEL}" "2"
  echo "[test_54] loaded '${DEFAULT_MODEL}@2' (same name, different version -- NOT the default)"

  # 6) route it in as the SHADOW for every stream -> @2 becomes SERVING
  model_routing "" "" "" "${DEFAULT_MODEL}" "2"
  assert_status_ref_eventually "model=${DEFAULT_MODEL}" "shadow" "${DEFAULT_MODEL}" "2" \
      "test_54: '${DEFAULT_MODEL}@2' is serving as the shadow for the streams"

  # 6b) while SERVING, the unload must be refused BECAUSE IT IS SERVING -- not because a
  #     sibling version happens to be the default. Pinning the reason (not just the 409)
  #     is the point: both rules answer 409, so only the wording distinguishes them.
  log_mark
  unload_model "${DEFAULT_MODEL}" "2"
  echo "[test_54] Requested unloading of '${DEFAULT_MODEL}@2' while serving as shadow (expected: rejected, SERVING)"
  assert_http_status 409 "test_54: unload of a SERVING version is refused with 409"
  assert_error_code  "MODEL_IN_USE" "test_54: ...carrying MODEL_IN_USE"
  assert_rest_body_contains "${DEFAULT_MODEL}@2'isserving" \
      "test_54: ...and the reason is SERVING (names the exact version)"
  assert_rest_body_absent   "isaconfigureddefault" \
      "test_54: ...NOT the name-level configured-default reason (a sibling version is the default)"

  # 7) clear the shadow -> @2 drains to idle. Still loaded, still NOT the default.
  model_routing "" "" "" "none"
  echo "[test_54] cleared the shadow -- '${DEFAULT_MODEL}@2' drains to idle (0 streams)"
  assert_status_ref_eventually_absent "model=${DEFAULT_MODEL}" "shadow" "${DEFAULT_MODEL}" "2" \
      "test_54: no stream carries the '${DEFAULT_MODEL}@2' shadow ref any more"
  sleep "${DELAY}"

  # 8) unload the idle NON-DEFAULT version. Its sibling @1 holds the default, but the
  #    default protects that IDENTITY only -- @2 is unloadable on its own.
  log_mark
  unload_model "${DEFAULT_MODEL}" "2"
  echo "[test_54] Requested unloading of idle NON-DEFAULT '${DEFAULT_MODEL}@2' (expected: SUCCESS)"
  assert_http_status 202 "test_54: unload of an idle NON-DEFAULT version is ACCEPTED (202)"
  assert_rest_body_contains "${DEFAULT_MODEL}@2'unloaded" \
      "test_54: ...and the response names the unloaded version"
  assert_rest_body_absent   "isaconfigureddefault" \
      "test_54: ...and is NOT refused by the configured-default rule (a sibling holds the default)"
  assert_log "unload idle model '${DEFAULT_MODEL}@2'" \
      "test_54: '${DEFAULT_MODEL}@2' actually torn down (log)"

  # 8b) the engine is really gone -- the stated effect, not just the response
  assert_status_version_eventually_absent "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "2" \
      "test_54: '${DEFAULT_MODEL}@2' removed from the pool after unload (engine freed)"

  # 9) the default sibling is UNTOUCHED: still loaded, still default, still serving both
  get_model_status "model=${DEFAULT_MODEL}"
  assert_status_version_eventually "model=${DEFAULT_MODEL}" "${DEFAULT_MODEL}" "1" \
      "test_54: the default '${DEFAULT_MODEL}@1' survived the sibling's unload"
  assert_status_ref "model" "${DEFAULT_MODEL}" "1" \
      "test_54: ...and both streams are still serving on it"

  # 10) the ACTUAL default is STILL protected -- the fix narrowed the guard to the
  #     identity, it must not have removed it.
  log_mark
  unload_model "${DEFAULT_MODEL}" "1"
  echo "[test_54] Requested unloading of the DEFAULT '${DEFAULT_MODEL}@1' (expected: rejected)"
  assert_http_status 409 "test_54: the configured default is STILL refused (409)"
  assert_error_code  "MODEL_IN_USE" "test_54: ...carrying MODEL_IN_USE"
  assert_rest_body_contains "${DEFAULT_MODEL}@1'isaconfigureddefault" \
      "test_54: ...naming the exact version that is the default"

  # 11) a version-LESS unload of the name is refused too (it would sweep the default's
  #     own version) -- behaviour deliberately unchanged by the fix.
  log_mark
  unload_model "${DEFAULT_MODEL}"
  echo "[test_54] Requested unloading by BARE NAME '${DEFAULT_MODEL}' (expected: rejected)"
  assert_http_status 409 "test_54: a version-less unload of a defaulted name is still refused"
  assert_rest_body_contains "${DEFAULT_MODEL}'isaconfigureddefault" \
      "test_54: ...by the NAME-level guard (unchanged)"

  # 12) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# Test 55: the three ways an A/B default pair can end -- SWAP, PROMOTE, and the
# both-roles CONFLICT.
#
# A model may not hold both default roles at once (a model cannot A/B against itself),
# and whether a request produces that state depends on the WHOLE request, not on the
# role it happens to name first. From one starting pair (primary=v1, shadow=v2) the
# three outcomes an operator can ask for are:
#
#   SWAP     default{model: v2, shadow: v1}     roles exchange, the A/B continues
#   PROMOTE  default{model: v2, shadow: null}   v2 takes over, the A/B ends
#   CONFLICT default{model: v2}                 refused: v2 is still the shadow
#
# The first two are legal because the same request also says what happens to the other
# role; the third is not, because it leaves v2 in both. All three are decided in the
# ADMISSION pass, so the verdict reaches the caller as a status code -- an apply-pass
# refusal would arrive after 202 and be invisible on the wire.
#
# What to validate:
#   - CONFLICT is refused with 409 + MODEL_IN_USE, names the offending model, and
#     changes nothing (asserted before and after).
#   - SWAP moves both designations, and the streams' lanes follow.
#   - PROMOTE moves the primary AND drops the default shadow; the streams lose their
#     A/B lane and keep serving on the promoted model.
#   - clearing an already-absent default shadow is a no-op that still succeeds
#     (idempotent, so a reconciler can post a desired end-state blind).
#   - no frames are dropped across any of it.
test_55() {
  echo "-----------------------------------------------------------------------"
  echo "[test_55] starting test 55 (default pair: swap / promote / both-roles conflict)"

  # 1) config for this case (edits the per-case config copies), THEN start the app with it
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  # unified A/B batch: primary row 0, shadow row 1. No config defaults -- the runtime
  # pair is established below over the REST control plane.
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  comment_cfg    multimodel           primary-model
  comment_cfg    multimodel           shadow-model

  # 2) start the app with the current config copies
  start_app || return

  # 3) load both versions of the SAME model -- the pair the A/B runs between
  load_model "${DEFAULT_MODEL}" "1" "${DEFAULT_ENGINE}"
  wait_for_model "${DEFAULT_MODEL}" "1"
  load_model "${DEFAULT_MODEL}" "2" "${BEST_ENGINE}"
  wait_for_model "${DEFAULT_MODEL}" "2"

  # 4) ORIGINAL STATE: primary=v1, shadow=v2, both streams following the default
  model_routing_default_both "${DEFAULT_MODEL}" "1" "${DEFAULT_MODEL}" "2"
  assert_http_status 202 "test_55: the starting A/B pair (primary=v1, shadow=v2) is accepted"
  add_stream "${PREFIX}1" "Cam-1" "${STREAM}"
  add_stream "${PREFIX}2" "Cam-2" "${STREAM}"
  echo "[test_55] added Cam-1 + Cam-2 on the default pair"
  sleep "${DELAY}"
  get_model_routing
  assert_route_default "model.version=1,shadow.version=2" \
      "test_55: ORIGINAL state -- default{model:v1, shadow:v2}"
  assert_status_ref_eventually "model=${DEFAULT_MODEL}" "shadow" "${DEFAULT_MODEL}" "2" \
      "test_55: ...and the streams are actually running the v2 A/B lane"

  # CASE0: v2 is the current default shadow; naming it the default primary and saying nothing
  # about the shadow would leave it in both roles.
  log_mark
  model_routing_default_both "${DEFAULT_MODEL}" "2" "-"
  echo "[test_55] CASE 3: promote v2 WITHOUT saying what happens to the shadow (expected: refused)"
  assert_http_status 409 "test_55: CASE 3 -- both-roles conflict is refused with 409"
  assert_error_code  "MODEL_IN_USE" "test_55: ...carrying MODEL_IN_USE"
  assert_rest_body_contains "${DEFAULT_MODEL}@2" \
      "test_55: ...and the reason names the offending model"
  # The message must TEACH the fix, not restate the rule: a client that only knows
  # "refused" cannot proceed, so both legal end-states have to be named, and the hint
  # must be the specific one rather than the plane's generic retry advice.
  assert_rest_body_contains "isalreadythedefaultshadow" \
      "test_55: ...says WHICH role it already holds"
  assert_rest_body_contains "PROMOTEit" \
      "test_55: ...offers the PROMOTE end-state"
  assert_rest_body_contains "SWAPtheroles" \
      "test_55: ...and the SWAP end-state"
  # the hint embeds a JSON snippet inside a JSON string, so the quotes arrive escaped
  assert_rest_body_contains '\"shadow\":null' \
      "test_55: ...spelling out the shadow-clear that promotes it"
  assert_rest_body_absent "re-checkthereferencedmodels" \
      "test_55: ...and NOT the generic routing retry hint"
  get_model_routing
  assert_route_default "model.version=1,shadow.version=2" \
      "test_55: ...and NOTHING changed -- still default{model:v1, shadow:v2}"

  # CASE 1: SWAP -- the roles exchange in one request
  log_mark
  model_routing_default_both "${DEFAULT_MODEL}" "2" "${DEFAULT_MODEL}" "1"
  echo "[test_55] CASE 1: SWAP -> default{model:v2, shadow:v1}"
  assert_http_status 202 "test_55: CASE 1 -- the swap is accepted (202)"
  # Both models are already warm, one per role pool, so exchanging their roles is a
  # RELABEL of the existing bins -- not a rebuild. A swap that instantiates a second
  # bin re-runs the engine build, loses the gie-id and the per-version config, and
  # silently lands both roles on the same artifact (the A/B stops being an A/B).
  assert_log "SWAP (IN-PLACE)" "test_55: ...by RELABELING the warm bins (gie kept, no reload)"
  assert_log_absent "nvmodelbin created" "test_55: ...creating NO new model bin"
  sleep "${DELAY}"
  get_model_routing
  assert_route_default "model.version=2,shadow.version=1" \
      "test_55: ...both designations moved -- default{model:v2, shadow:v1}"
  assert_status_ref_eventually "model=${DEFAULT_MODEL}" "model" "${DEFAULT_MODEL}" "2" \
      "test_55: ...and the streams' PRIMARY lane followed to v2"
  assert_status_ref_eventually "model=${DEFAULT_MODEL}" "shadow" "${DEFAULT_MODEL}" "1" \
      "test_55: ...and the streams' SHADOW lane followed to v1"
  # the gie-id names the BIN, so it is the identity proof: each version must still be
  # wired on the very bin it was loaded on, now merely wearing the other role's label.
  # A rebuild would renumber these (fresh bins start at 2 / 101).
  assert_log "name=Cam-1  model=${DEFAULT_MODEL}@2  role=Primary  gie=1001" \
      "test_55: ...on v2's ORIGINAL bin (gie 1001, retagged Primary)" 12
  assert_log "name=Cam-1  model=${DEFAULT_MODEL}@1  role=Shadow  gie=1000" \
      "test_55: ...and v1 on ITS original bin (gie 1000, retagged Shadow)" 12
  assert_log "name=Cam-2  model=${DEFAULT_MODEL}@2  role=Primary  gie=1001" \
      "test_55: ...for every stream, not just one" 12

  # swap BACK, so the promote below starts from the documented original state (and the
  # exchange is proven to work in both directions, not just one)
  model_routing_default_both "${DEFAULT_MODEL}" "1" "${DEFAULT_MODEL}" "2"
  assert_http_status 202 "test_55: the reverse swap is accepted too"
  sleep "${DELAY}"
  get_model_routing
  assert_route_default "model.version=1,shadow.version=2" \
      "test_55: ...back to default{model:v1, shadow:v2}"

  # CASE 2: PROMOTE -- v2 takes over and the A/B ends
  log_mark
  model_routing_default_both "${DEFAULT_MODEL}" "2" "none"
  echo "[test_55] CASE 2: PROMOTE -> default{model:v2, shadow:null}"
  assert_http_status 202 "test_55: CASE 2 -- the promote is accepted (202)"
  # a promote is the same relabel with the demoted lane torn down: the shadow bin is
  # retagged Primary and the displaced old primary is dropped -- no new bin, no rebuild.
  assert_log "PROMOTE (IN-PLACE)" "test_55: ...by RELABELING the warm shadow bin (gie kept, no reload)"
  assert_log "dropping old primary" "test_55: ...and dropping the demoted primary"
  assert_log_absent "nvmodelbin created" "test_55: ...creating NO new model bin"
  sleep "${DELAY}"
  get_model_routing
  assert_route_default "model.version=2,shadow=None" \
      "test_55: ...default{} is now model:v2 with NO shadow"
  assert_status_ref_eventually "model=${DEFAULT_MODEL}" "model" "${DEFAULT_MODEL}" "2" \
      "test_55: ...the streams serve on v2"
  # same identity proof for the promote: v2 serves from the bin it was ALREADY inferring
  # on as the shadow -- the promotion is a relabel of that bin, not a fresh instance.
  assert_log "name=Cam-1  model=${DEFAULT_MODEL}@2  role=Primary  gie=1001" \
      "test_55: ...from v2's ORIGINAL shadow bin (gie 1001, retagged Primary)" 12
  assert_log "name=Cam-2  model=${DEFAULT_MODEL}@2  role=Primary  gie=1001" \
      "test_55: ...for every stream, not just one" 12
  assert_status_ref_eventually_absent "model=${DEFAULT_MODEL}" "shadow" "${DEFAULT_MODEL}" "2" \
      "test_55: ...and no stream carries a v2 SHADOW ref any more (the A/B lane is gone)"
  assert_status_ref_eventually_absent "model=${DEFAULT_MODEL}" "shadow" "${DEFAULT_MODEL}" "1" \
      "test_55: ...nor a v1 one -- the A/B is over, not re-pointed"

  # CASE 3: idempotence: clearing an absent default shadow still succeeds
  model_routing_default_both "" "" "none"
  echo "[test_55] clearing an ALREADY-absent default shadow (expected: accepted no-op)"
  assert_http_status 202 "test_55: clearing an absent default shadow is an accepted no-op"
  get_model_routing
  assert_route_default "model.version=2,shadow=None" \
      "test_55: ...and the state is unchanged by it"

  # 5) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Test 56: STATIC (config-listed) streams -- identity must match the REST path
# ---------------------------------------------------------------------------
# Every other case in this suite creates its streams with POST /stream/add, which
# carries camera_id + camera_name in the request body. This case is the OTHER way a
# stream can exist: pre-listed in the app config with
#
#     [source-list]
#     list=<uri>;<uri>;
#     sensor-id-list=cam-0;cam-1;
#     sensor-name-list=Cam-0;Cam-1;
#
# nvmultiurisrcbin forwards those bindings downstream as GST_NVEVENT_STREAM_MODEL_BIND
# once the pipeline reaches PLAYING, so the plugin has the same two facts it gets from
# the REST body -- just delivered by a different route, and possibly AFTER the stream
# has already attached.
#
# THE CONTRACT: how a stream was created must not change what it IS. A statically
# listed camera has to end up with the same identity a dynamically added one gets --
# its camera_id resolves in both GET APIs, per-stream filters find it, and it can be
# named in a routing request. A stream whose id never lands is invisible to
# GET /model/routing (which skips ids it cannot resolve) and unreachable by any
# per-stream POST: fleet-wide "all" would be the only way to route it.
test_56() {
  echo "-----------------------------------------------------------------------"
  echo "[test_56] starting test 56 (STATIC config-listed streams | identity parity with stream/add)"

  # 1) config for this case
  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  # multimodel config updates
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # the streams come up at PLAYING, before any REST call could run, so the default
  # primary is declared in the CONFIG -- otherwise they would attach as passthrough
  # and the routing plane would have nothing to report about them.
  cfg_set multimodel           primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}  engine  "${WORST_ENGINE}"

  # 2) THE POINT OF THIS CASE: two sources pre-listed in the app config, with their
  #    ids and display names alongside. No stream/add is issued anywhere below.
  # ONE definition of the provisioned identity: the config lines below and every
  # assertion further down read the same variables, so a change to what this case
  # provisions can never leave the assertions checking the old values.
  # ROUTABLE ids and DISPLAY names are deliberately DIFFERENT. If both were the same
  # string this case could not tell which of the two an API reported -- and reporting
  # the display name under a key called camera_id is a real failure mode: the value
  # round-trips nowhere, because a POST naming it is answered with STREAM_UNKNOWN.
  local SID0="Cam-0" SID1="Cam-1"
  local SNM0="Display-0" SNM1="Display-1"
  cfg_set source-list          list                  "${STREAM};${STREAM};"
  cfg_set source-list          sensor-id-list        "${SID0};${SID1};"
  cfg_set source-list          sensor-name-list      "${SNM0};${SNM1};"

  # 3) start the app -- streams exist from startup, nothing is added over REST
  start_app || return
  sleep "${DELAY}"

  # 4) the bindings must actually reach the plugin. Without this line the two facts
  #    never arrived and everything below is failing for one upstream reason.
  assert_log "STREAM BIND source=0 camera='${SID0}'" \
      "test_56: the static binding for ${SID0} reaches the plugin" 15
  assert_log "STREAM BIND source=1 camera='${SID1}'" \
      "test_56: ...and the one for ${SID1}" 15

  # 5) ROUTING PLANE. routes[] is keyed by the ROUTABLE camera_id and skips any stream
  #    whose id it cannot resolve -- so a missing id shows up here as an empty snapshot
  #    even while both streams are inferring.
  get_model_routing
  assert_route_count 1 "test_56: both static streams appear in routes[] (merged: identical state)"
  assert_route_entry "${SID0},${SID1}" "state=serving,model.name=${DEFAULT_MODEL}" \
      "test_56: ...listed by their configured camera_ids, serving the config default"

  # 6) STATUS PLANE, same identity. "src" is the plugin's fallback when no name reached
  #    it: asserting its ABSENCE is what catches a silently-unlabelled stream.
  get_model_status
  assert_status_contains     "\"camera_id\":\"${SID0}\"" "test_56: model/status reports the configured id for stream 0"
  assert_status_contains     "\"camera_id\":\"${SID1}\"" "test_56: ...and for stream 1"
  assert_status_not_contains "\"camera_id\":\"src\""        "test_56: ...and no stream falls back to the unnamed 'src' placeholder"
  assert_status_not_contains "\"camera_id\":\"${SNM0}\"" \
      "test_56: ...and camera_id carries the ROUTABLE id, never the display name"

  # 7) the id must be USABLE, not merely present: both per-stream filters have to
  #    resolve it, exactly as they do for a stream created over REST.
  get_model_routing "stream=${SID0}"
  assert_route_count 1 "test_56: ?stream=<configured id> resolves on the routing plane"
  assert_route_entry "${SID0}" "state=serving" \
      "test_56: ...and narrows to that stream alone"
  get_model_status "stream=${SID0}"
  assert_status_contains     "\"camera_id\":\"${SID0}\"" "test_56: ?stream=<configured id> resolves on the status plane"
  assert_status_not_contains "\"camera_id\":\"${SID1}\"" "test_56: ...and excludes the other stream"

  sleep "${DELAY}"

  # 8) stop the app
  stop_app
  echo "-----------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Test 57: STATIC streams, NO config default -- the LATE-BIND ordering
# ---------------------------------------------------------------------------
# Same static provisioning as case 56, with the config-time default removed so the
# streams attach the instant their pads appear, as passthrough, instead of waiting on
# a model. That shifts the attach EARLIER relative to the one-shot bind emitted from
# nvmultiurisrcbin's PAUSED->PLAYING transition, which is the ordering where a stream
# is wired BEFORE its identity is known.
#
# Both orderings are legal and neither is guaranteed -- they land milliseconds apart --
# so the contract cannot be "the bind wins the race". It is that a stream ends up with
# the SAME identity either way: the bind that arrives late must still be applied, not
# merely cached for an attach that already happened. Before that was true, losing this
# race left the stream permanently unnamed and, worse, with no routable camera_id --
# invisible to GET /model/routing and unreachable by any per-stream request.
#
# Case 56 samples one ordering, this one samples the other. What both assert is the
# outcome, because that is the part that must not depend on timing.
test_57() {
  echo "-----------------------------------------------------------------------"
  echo "[test_57] starting test 57 (STATIC streams, no config default | identity survives a LATE bind)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  # A config default that must be WAITED for: the pipeline is held in PAUSED until the
  # engine is ready. The one-shot static bind is emitted from PAUSED->PLAYING, so delaying
  # that transition lets the stream attach FIRST -- the ordering where identity arrives
  # after the stream is already wired.
  cfg_set multimodel           primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}  engine  "${WORST_ENGINE}"

  local SID0="cam-0" SID1="cam-1"
  local SNM0="Display-0" SNM1="Display-1"
  cfg_set source-list          list                  "${STREAM};${STREAM};"
  cfg_set source-list          sensor-id-list        "${SID0};${SID1};"
  cfg_set source-list          sensor-name-list      "${SNM0};${SNM1};"

  start_app || return
  sleep "${DELAY}"

  # the bindings reach the plugin on either ordering -- this only proves they arrived
  assert_log "STREAM BIND source=0 camera='${SID0}'" \
      "test_57: the static binding for ${SID0} reaches the plugin" 15
  assert_log "STREAM BIND source=1 camera='${SID1}'" \
      "test_57: ...and the one for ${SID1}" 15

  # THE CONTRACT: identity is correct regardless of which side of the race this run hit.
  get_model_routing
  assert_route_count 1 "test_57: both streams appear in routes[] (merged: identical state)"
  assert_route_entry "${SID0},${SID1}" "state=serving" \
      "test_57: ...listed by their configured camera_ids"

  get_model_status
  assert_status_contains     "\"camera_id\":\"${SID0}\"" "test_57: model/status reports the routable id for stream 0"
  assert_status_contains     "\"camera_id\":\"${SID1}\"" "test_57: ...and for stream 1"
  assert_status_contains     "\"camera_name\":\"${SNM0}\"" "test_57: ...with the display name in camera_name"
  assert_status_not_contains "\"camera_id\":\"${SNM0}\"" \
      "test_57: ...and never the display name under camera_id"
  assert_status_not_contains "\"camera_id\":\"src\"" \
      "test_57: ...and no stream is left wearing the unnamed 'src' placeholder"

  # the id must still be USABLE on both planes -- an identity applied late is only
  # repaired if the lookup maps were updated, not just the log line
  get_model_routing "stream=${SID0}"
  assert_route_count 1 "test_57: ?stream=<id> resolves on the routing plane"
  get_model_status "stream=${SID0}"
  assert_status_contains     "\"camera_id\":\"${SID0}\"" "test_57: ...and on the status plane"
  assert_status_not_contains "\"camera_id\":\"${SID1}\"" "test_57: ...narrowing to that stream alone"

  sleep "${DELAY}"
  stop_app
  echo "-----------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Test 58: every version must bring its own model files (ARTIFACT_MISSING)
# ---------------------------------------------------------------------------
# The catalog's base config is a legitimate artifact source -- for exactly ONE version,
# the first to claim it. A SECOND version that arrives with no engine_file, no
# config_file and no [model-<name>-<version>] block would resolve to that same base:
# two version numbers, one model. Loaded that way it comes up `ready` and routable at
# the requested version number while serving the model every other version already
# runs, so a rollout ships nothing and an A/B scores a model against itself.
#
# It must therefore be refused AT ADMISSION -- a 202 followed by a silently wrong model
# is the failure this guards, so a late rejection would not help the caller.
#
# The four shapes, and why only one of them is an error:
#   version 1, nothing supplied            -> OK, the base config is its artifact
#   version 2, nothing supplied            -> 400 ARTIFACT_MISSING
#   version 2, engine_file supplied        -> OK, loads that engine
#   version 3, config_file supplied        -> OK, builds from that config
test_58() {
  echo "-----------------------------------------------------------------------"
  echo "[test_58] starting test 58 (model/load: every version needs its own artifact)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  1
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  comment_cfg multimodel       primary-model
  comment_cfg multimodel       shadow-model

  start_app || return

  # 1) FIRST version, nothing but name+version. This is the normal catalog flow -- the
  #    base config declared for the name IS this version's artifact, and nothing else
  #    has claimed it. Must be accepted, or every config-driven load breaks.
  load_model "${CAR_MODEL}" "1"
  assert_http_status 202 "test_58: the FIRST version of a name loads bare (the base config is its artifact)"
  wait_for_model "${CAR_MODEL}" "1"

  # 2) SECOND version, same empty request. Now the base is taken, so this version has
  #    nothing of its own -- refused, and refused HERE rather than accepted and warmed.
  load_model "${CAR_MODEL}" "2"
  assert_http_status 400 "test_58: a SECOND version with no artifact is REFUSED at admission"
  assert_error_code  "ARTIFACT_MISSING" "test_58: ...carrying ARTIFACT_MISSING"
  assert_rest_body_contains "nomodelfilesofitsown" \
      "test_58: ...and the reason says the version brings no files of its own"
  assert_rest_body_contains "engine_file" \
      "test_58: ...and the hint names the fields that would fix it"
  # the refusal must be TOTAL: a rejected load may not leave a half-registered version
  get_model_status "model=${CAR_MODEL}"
  assert_status_not_contains "\"state\":\"ready\",\"version\":\"2\"" \
      "test_58: ...and version 2 was NOT registered by the rejected request"

  # 3) the same version, now carrying an engine -- accepted, and it gets its OWN derived
  #    config rather than the shared base (that separation is the whole point).
  load_model "${CAR_MODEL}" "2" "${BEST_ENGINE}"
  assert_http_status 202 "test_58: the SAME version with engine_file is accepted"
  wait_for_model "${CAR_MODEL}" "2"
  get_model_status "model=${CAR_MODEL}&version=2"
  assert_status_contains "${CAR_MODEL}_2_config" \
      "test_58: ...and is backed by its OWN derived config, not the shared base"

  # 4) config_file is the other way to satisfy it: no engine, build from that config.
  load_model "${CAR_MODEL}" "3" "" "${CFG_DIR}/config_infer_primary_car.txt"
  assert_http_status 202 "test_58: a version supplying config_file instead of an engine is accepted"

  sleep "${DELAY}"
  stop_app
  echo "-----------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Test 59: the model plane at a SECONDARY stage (process-mode=2)
# ---------------------------------------------------------------------------
# A secondary nvmodelmux classifies the objects a primary detected, so it differs from
# a primary in the two ways that shape this whole endpoint: it has no per-stream
# binding (it sees the batch AFTER the primary re-muxed it, and applies one model to
# every stream that reaches it), and it has no shadow lane (no A/B pair).
#
# From those two facts the contract follows -- "exactly one model INFERS per sgie
# stage", not "exactly one model exists there" (sgie approach, constraint 3):
#
#   model/update  in-place OTA on the element that is serving. The cheap path: one
#                 engine, built aside, identity renamed on nvinfer's confirm. Bounded
#                 by the OTA gate -- same input resolution, output width, batch.
#   model/load    builds and warms a SECOND model alongside the serving one. It is
#                 AVAILABLE, not ACTIVE: it costs VRAM, gets its own gie-unique-id and
#                 changes no output until a route promotes it. This path exists
#                 precisely because the OTA gate refuses a differently-shaped engine,
#                 so a structurally different model can arrive no other way.
#   model/routing streams:"all" rebinds which resident model serves. With no lanes
#                 there is no promote/swap distinction -- at a primary those differ
#                 only in the shadow field, and there is no shadow field here. One
#                 call, both directions: name v3 and v3 serves, name v2 and it is
#                 undone, instantly, because the displaced model stays loaded and warm.
#
# What must be REFUSED, and why each is a different mistake with a different fix:
#   routes[].streams = [cam-0]     400 ROUTING_PER_STREAM_UNSUPPORTED  (no per-stream
#                                      plane exists -- bind at the PRIMARY instead)
#   routes[].shadow  = {..}        400 ROUTING_SHADOW_UNSUPPORTED      (no A/B lane;
#                                      an explicit null is fine, it is already true)
#   routing to a version not resident  404 MODEL_NOT_LOADED   (routing never ingests
#                                      an artifact, at either stage -- load it first)
#   loading a version already resident 409 MODEL_ALREADY_LOADED (two entries claiming
#                                      one identity: no route could tell them apart)
#   updating ONTO a resident version   409 MODEL_ALREADY_LOADED (the OTA renames the
#                                      serving model, and that name is taken)
#   unloading the SERVING model        409 MODEL_IN_USE  (the stage cannot be left with
#                                      nothing inferring -- reroute first, then unload)
#   unloading one still WARMING        409 MODEL_WARMING (its engine build owns those
#                                      elements on another thread)
#   version-less unload, 2 resident    400 VERSION_INVALID (guessing frees the wrong
#                                      engine)
#
# The run below walks one model through all of it: OTA 1->2, load @3 resident and idle,
# promote @3, reverse to @2, unload @3. Both versions stay resident across the promote
# and the reversal, and the default sanity checks assert those rebinds cost ZERO frames
# -- a reroute is a relink at a buffer boundary, not a restart.
#
# It then closes the one claim the rest of the run cannot reach. A route to a model that
# is still WARMING must be ACCEPTED and PARKED, with the current model serving every
# frame until the target can infer. @3 deserialized a cached engine in milliseconds, so
# that window never opened; the last section moves the cached engine aside to force a
# real TensorRT rebuild and asserts the hand-over lands only when the build finishes.
# ---------------------------------------------------------------------------
test_59() {
  echo "-----------------------------------------------------------------------"
  echo "[test_59] starting test 59 (secondary stage: model/update + load + routing)"

  cfg_set sink0                type                  ${SINK_TYPE}
  cfg_set source-list          max-batch-size        2
  cfg_set tiled-display        rows                  2
  cfg_set tiled-display        columns               2
  cfg_set multimodel           unified-batch         1
  cfg_set multimodel           per-model-batch-size  2
  cfg_set multimodel                                                primary-model         "${DEFAULT_MODEL};${DEFAULT_MODEL_VERSION}"
  cfg_set multimodel                                                shadow-model          "${DEFAULT_MODEL};${DEFAULT_SHADOW_MODEL_VERSION}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}           engine                "${WORST_ENGINE}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_MODEL_VERSION}           gie-unique-id           "1"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_SHADOW_MODEL_VERSION}    engine                "${BEST_ENGINE}"
  cfg_set model-${DEFAULT_MODEL}-${DEFAULT_SHADOW_MODEL_VERSION}    gie-unique-id           "2"
  # enable the secondary stage
  cfg_main secondary-gie0      enable                1
  cfg_main secondary-gie0      batch-size            2

  local SID0="cam-0" SID1="cam-1"
  start_app || return

  sleep "${DELAY}"
  add_stream "${SID0}" "Cam-0" "${STREAM}"
  add_stream "${SID1}" "Cam-1" "${STREAM}"
  sleep "${DELAY}"

  assert_log "built (secondary): model='${SGIE_MODEL}@1'" "test_59: SECONDARY stage built from its own config"

  # The cheap reshape: an in-place OTA on the element that is serving. Runs FIRST so
  # it operates on a stage holding exactly one model, as it did before load existed.
  model_update "${SGIE_MODEL}" "1" "2" "${SGIE_ENGINE}"
  assert_http_status 202 "test_59: model/update at a SECONDARY accepted"
  assert_log "sgie: OTA reload CONFIRMED for '${SGIE_MODEL}@2'" "test_59: secondary OTA committed (nvinfer confirmed)"
  # secondary_models[] must FOLLOW the OTA rename. The stage still runs one element
  # with one gie; only the identity moved, and the routing plane has to say so.
  get_model_routing
  assert_routing_contains "\"name\":\"${SGIE_MODEL}\",\"version\":\"2\"" \
      "test_59: model/routing follows the OTA rename (@1 -> @2)"
  # Snapshot the CAS token before the secondary reroutes. There is one counter for
  # the whole pipeline -- the stages share it through a GstContext -- so this is the
  # number a controller would hold, whichever stage it read it from.
  local rev_before_promote
  rev_before_promote="$(routing_revision)"

  # model/load at a secondary makes a second model AVAILABLE, not ACTIVE (sgie
  # approach, constraint 3): it is built and warmed alongside the serving one, gets
  # its own gie id, and infers nothing until an all-streams route promotes it. 202
  # (not 200) because the engine build finishes after the response.
  load_model "${SGIE_MODEL}" "3" "${SGIE_ENGINE}"
  assert_http_status 202 "test_59: model/load at a SECONDARY is accepted (resident, idle)"
  assert_log "sgie: model '${SGIE_MODEL}@3' LOADED" "test_59: ...and the second model warmed to 'loaded'"

  # Residency is per (name, version): the same version twice would be two entries
  # claiming one identity, which no route could tell apart.
  load_model "${SGIE_MODEL}" "3" "${SGIE_ENGINE}"
  assert_http_status 409 "test_59: re-loading a version already resident is refused"
  assert_error_code  "MODEL_ALREADY_LOADED" "test_59: ... and says WHY (not the pass-through 404)"

  # An OTA RENAMES the serving model to the target version -- so a version that is
  # already resident cannot be taken as the new identity. The caller wants the
  # resident one anyway; it is already warm.
  model_update "${SGIE_MODEL}" "2" "3" "${SGIE_ENGINE}"
  assert_http_status 409 "test_59: model/update onto a RESIDENT version is refused"
  assert_error_code  "MODEL_ALREADY_LOADED" "test_59: ... naming the collision, not a generic conflict"

  # ---- model/routing at a SECONDARY -------------------------------------------
  # PER-STREAM routing is the commonest wrong request here and must fail by saying
  # WHY: the secondary sees the batch after the primary re-muxed it, so it has no
  # per-stream binding to write. A blanket 409 would leave the caller guessing.
  get_model_routing
  local rev_before_reject
  rev_before_reject="$(routing_revision)"
  model_routing "${SID0}" "${SGIE_MODEL}" "3"
  assert_http_status 400 "test_59: per-stream routing at a SECONDARY is refused"
  assert_error_code  "ROUTING_PER_STREAM_UNSUPPORTED" "test_59: ... and names the reason (not a generic conflict)"
  assert_log "model/routing rejected: a secondary stage has no per-stream routing" \
      "test_59: ...and the reject is LOGGED in the primary's reject wording"
  # A REJECTED route changed nothing, so it must not burn anyone's if_revision --
  # the counter tracks applied state, not attempts.
  get_model_routing
  if [ "$(routing_revision)" = "${rev_before_reject}" ]; then
    _pass "test_59: ...and a REJECTED route leaves routing_revision untouched"
  else
    _fail "test_59: a REJECTED route moved routing_revision (${rev_before_reject} -> $(routing_revision))"
  fi

  # A named shadow is an A/B request, and a secondary has no shadow lane.
  model_routing "" "${SGIE_MODEL}" "3" "${SGIE_MODEL}" "2"
  assert_http_status 400 "test_59: a shadow ref at a SECONDARY is refused"
  assert_error_code  "ROUTING_SHADOW_UNSUPPORTED" "test_59: ... because there is no A/B lane here"

  # Routing never ingests an artifact: a version that was never loaded is a 404,
  # not an implicit load.
  model_routing "" "${SGIE_MODEL}" "9"
  assert_http_status 404 "test_59: routing to a NON-RESIDENT version is refused"
  assert_error_code  "MODEL_NOT_LOADED" "test_59: ... and says to model/load it first"

  # THE PROMOTION: @3 is loaded and warm but idle; this makes it serve every stream.
  # No lanes exist here, so this one call IS both promote and swap.
  model_routing "" "${SGIE_MODEL}" "3"
  assert_http_status 202 "test_59: promoting the LOADED model to serving is accepted"
  assert_log "sgie: SERVING '${SGIE_MODEL}@3'" "test_59: ...and the stage rebound to it"
  assert_log "displaced '${SGIE_MODEL}@2' stays loaded and warm" "test_59: ...displacing @2 without unloading it"
  # ...and the routing plane tracks the PROMOTE, not just the config-declared model.
  # Asserted HERE, while @3 serves -- the later GET runs after the reversal, when the
  # answer is @2 again and a field that never updated would look identical.
  get_model_routing
  assert_routing_contains "\"name\":\"${SGIE_MODEL}\",\"version\":\"3\"" \
      "test_59: model/routing tracks the promote (now serving @3)"
  # ...and the SHARED revision moved. This is the cross-stage assertion: the reroute
  # happened at the SECONDARY, and the merged body reports the UPSTREAM stage's
  # scalar -- so a number that advances here can only be one both stages point at.
  # With a per-stage counter the primary's would still read exactly what it did
  # before the promote, and a controller holding it would win a CAS it should lose.
  assert_revision_advanced "${rev_before_promote}" \
      "test_59: a route applied at the SECONDARY advances the shared routing_revision"

  # A promote body written out of habit ({model, shadow:null}) stays valid: the null
  # asserts "no shadow after this", which is permanently true at a secondary.
  model_routing "" "${SGIE_MODEL}" "2" "none"
  assert_http_status 202 "test_59: reversal (with an explicit null shadow) is accepted"
  assert_log "sgie: SERVING '${SGIE_MODEL}@2'" "test_59: ...and @2 serves again -- instant, it never unloaded"

  sleep "${DELAY}"
  # Both models are still resident afterwards: the reversal cost no rebuild.
  assert_log "Secondary │ ${SGIE_MODEL}" "test_59: both versions still listed at the stage"

  # ---- the GET plane reports the stage in the SAME v1 shape the primary uses ----
  # One models[] entry per name, versions[] inside it, each with a state -- so a client
  # reads both stages with one parser. The only secondary-specific fact is process_mode.
  get_model_status
  assert_status_contains "\"name\":\"${SGIE_MODEL}\""  "test_59: model/status lists the secondary's model"
  assert_status_contains "\"process_mode\":2"          "test_59: ...tagged with the stage that answered"
  assert_status_contains "\"version\":\"2\""           "test_59: ...the serving version"
  assert_status_contains "\"version\":\"3\""           "test_59: ...AND the resident idle one"
  assert_status_contains "\"state\":\"loaded\""        "test_59: ...which reports 'loaded', not 'serving'"
  # routing[] stays PRIMARY-only; the stage publishes what it serves under the same
  # {name, version} shape default.model uses, so no new object type enters the schema.
  get_model_routing
  assert_routing_contains "\"secondary_models\":"      "test_59: model/routing publishes the secondary's serving model"
  assert_routing_contains "\"name\":\"${SGIE_MODEL}\",\"version\":\"2\"" \
      "test_59: ...as the {name, version} shape default.model already uses"
  # ...carrying the DEVICE too, exactly as the primary's routes[].model does: a
  # secondary can be pinned to another GPU, so "which model" without "running where"
  # is only half the answer.
  assert_routing_contains "\"gpu\":0,\"name\":\"${SGIE_MODEL}\"" \
      "test_59: ...and the gpu id, aligned with the primary's model ref"
  # routes[] stays PRIMARY-only: two cameras on identical models collapse to ONE entry.
  # A secondary leaking an entry in here would show up as a second one, which is the
  # structural half of "no new object type enters the schema".
  assert_route_count 1 "test_59: routes[] stays primary-only (no secondary entry leaked)"

  # ---- model/unload at a SECONDARY --------------------------------------------
  # Refused for exactly one model: the one serving. A secondary cannot be left with
  # nothing inferring, and the fix is a reroute, not an unload.
  unload_model "${SGIE_MODEL}" "2"
  assert_http_status 409 "test_59: unload of the SERVING model is refused"
  assert_error_code  "MODEL_IN_USE" "test_59: ... because the stage cannot run without one"

  # The idle resident one IS unloadable -- that is the VRAM a settled rollout wants back.
  unload_model "${SGIE_MODEL}" "3"
  assert_http_status 202 "test_59: unload of the idle RESIDENT model is accepted"
  assert_log "sgie: model '${SGIE_MODEL}@3' UNLOADED" "test_59: ...and its engine is released"

  # Gone means gone: a second unload is a 404, not a silent success.
  unload_model "${SGIE_MODEL}" "3"
  assert_http_status 404 "test_59: unloading it twice is a 404"
  assert_error_code  "MODEL_NOT_LOADED" "test_59: ... naming the absence"

  # ...and routing to it now fails the same way, since residency is what routing needs.
  model_routing "" "${SGIE_MODEL}" "3"
  assert_http_status 404 "test_59: routing to the unloaded version is refused again"

  sleep "${DELAY}"
  get_model_status
  assert_status_not_contains "\"version\":\"3\"" "test_59: model/status no longer lists the unloaded version"
  assert_status_contains     "\"version\":\"2\"" "test_59: ...and the serving one is untouched"

  # ---- a route that lands while the target is still WARMING --------------------
  # The gapless claim: routing to a model whose engine is still building is ACCEPTED,
  # not refused, and the model currently serving keeps every frame until the target can
  # actually infer. Without a real warming window this path is unreachable -- @3 above
  # deserialized a cached engine in milliseconds -- so move the cached engine aside and
  # make the load REBUILD from ONNX. Same mechanism, and same idempotent recovery, as
  # test_47; see the long note there for why this is MOVED and restored, not deleted.
  local _eng="${SGIE_ENGINE}"
  local _bak="${_eng}.test59.bak"
  if [ -f "${_bak}" ]; then
    echo "[test_59] restoring engine backup left by a previous interrupted run"
    mv -f "${_bak}" "${_eng}"
  fi
  [ -f "${_eng}" ] && mv -f "${_eng}" "${_bak}"

  # config_file, NOT engine_file: nvmultiurisrcbin validates engine_file EXISTS before
  # it ever issues the query (400 GPU_ENGINE_MISSING), so naming the moved-aside engine
  # cannot reach a stage at all. The config names the ONNX, which is what the rebuild
  # consumes. A config_file load used to be claimed by the PRIMARY as a "self-describing
  # load"; it no longer is, because the config declares process-mode=2 -- see
  # modelmux_control_stage_owns.
  load_model "${SGIE_MODEL}" "4" "" "${SGIE_INFER_CONFIG}"
  assert_http_status 202 "test_59: load of a version needing a REBUILD is accepted"
  # Everything below depends on that load having been accepted AND claimed by the
  # secondary. If it was not, the remaining assert_log calls would each sit out their
  # full timeout waiting for markers that can never appear -- minutes of dead waiting
  # for an already-known failure. Bail to the restore step instead.
  if [ "${MM_REST_POST_CODE}" != "202" ]; then
    _fail "test_59: rebuild-load not accepted -- skipping the parked-route section"
  else
  # Mid-build the target owns its elements on the warm thread, so it cannot be torn
  # down. Asserted BEFORE the route below: once a route is parked on this model the
  # parked-route guard answers first (and more usefully), so this is the only window in
  # which the bare warming refusal is reachable.
  unload_model "${SGIE_MODEL}" "4"
  assert_http_status 409 "test_59: unload of a WARMING model is refused"
  assert_error_code  "MODEL_WARMING" "test_59: ... because its engine build is still running"

  # NO sleep: the route has to land inside the warming window, which is the whole point.
  model_routing "" "${SGIE_MODEL}" "4"
  assert_http_status 202 "test_59: a route to a still-WARMING model is accepted, not refused"
  assert_log "route to '${SGIE_MODEL}@4' PARKED" "test_59: ...and is PARKED until it is ready"

  # The stage must not go dark across that window: @2 is still the one inferring.
  get_model_status
  assert_status_contains "\"state\":\"serving\"" "test_59: the previous model still SERVES while the target warms"
  assert_status_contains "\"version\":\"2\""     "test_59: ...and it is still @2"

  # Now that a route is PARKED on it, the parked-route guard answers instead -- and its
  # hint is the one that actually unblocks the caller ("route back to the model
  # currently serving first"), where MODEL_WARMING would wrongly suggest just waiting.
  unload_model "${SGIE_MODEL}" "4"
  assert_http_status 409 "test_59: unload of a PARKED-ROUTE target is refused"
  assert_error_code  "MODEL_IN_USE" "test_59: ... naming the parked route, not the build"

  # NOT ASSERTED HERE: the version-less unload (400 VERSION_INVALID when a name has two
  # versions resident). In a two-stage pipeline the PRIMARY claims that request and
  # answers it from its own registry --
  #   <primary_gie> model unload refused: 'VehicleTypes': all 1 loaded version(s) still
  #                 serving stream(s) [?] (nothing unloaded)
  # -- even though it neither declares nor hosts VehicleTypes, so the secondary's guard
  # is unreachable from here. The version-ED unloads above DO reach the secondary, which
  # is what makes this worth a note rather than a silent omission. Left as an open
  # question against the primary's claim rule, not worked around in the test.

  # The hand-over: applied by the warm thread the moment the engine is ready. Generous
  # timeout -- this is a real TensorRT build, not a deserialize.
  assert_log "parked route for '${SGIE_MODEL}@4' now applying" \
      "test_59: the parked route applies once the rebuild finishes" 180
  assert_log "sgie: SERVING '${SGIE_MODEL}@4'" "test_59: ...and @4 takes over every stream" 30
  fi

  # put the cached engine back (the rebuild produced a fresh one; keep the ORIGINAL)
  if [ -f "${_bak}" ]; then
    mv -f "${_bak}" "${_eng}"
    echo "[test_59] restored the original b16 engine"
  fi

  sleep "${DELAY}"
  stop_app
  echo "-----------------------------------------------------------------------"
}

# --- runner ------------------------------------------------------------------
run_case() {
  # NOTE: 1:1 mapping -- selection N runs test_N. Keep this in lockstep with the test_<N>
  # function definitions and ALL_TESTS. (test_2 was inserted here; everything from the old
  # test_2 onward shifted +1.)
  case "$1" in
    -1)  test_dummy ;;
    1)   test_1 ;;           # isolation probe: dynamic add/remove, inference disabled
    2)   test_2 ;;           # custom primary requested WHILE warming -> passthru-with-tag, promote ALL waiting streams
    3)   test_3 ;;           # drain-to-zero refill, single stream, repeated (lifecycle)
    4)   test_4 ;;           # drain-to-zero refill, full 4-stream batch
    5)   test_5 ;;           # per-model capacity -> auto-shard (grow) + auto-compact
    6)   test_6 ;;           # metadata-driven primary routing alongside default
    7)   test_7 ;;           # model load/unload lifecycle + serving guard
    8)   test_8 ;;           # live scoped PRIMARY reroute (zero-drop)
    9)   test_9 ;;           # default primary + default SHADOW baseline (+ overrides)
    10)  test_10 ;;          # default p&s + custom p&s, mixed per stream
    11)  test_11 ;;          # per-sensor routing via CONFIG [stream-model-*]
    12)  test_12 ;;          # model/status GET (full + filtered)
    13)  test_13 ;;          # live per-stream model/routing (reroute/reject/clear/rollback)
    14)  test_14 ;;          # PROMOTE both-default (default rotates to promoted model)
    15)  test_15 ;;          # PROMOTE non-default shadow over DEFAULT primary
    16)  test_16 ;;          # PROMOTE custom shadow over CUSTOM primary (displaced custom unloaded)
    17)  test_17 ;;          # PROMOTE DEFAULT shadow over CUSTOM primary (default_shadow cleared)
    18)  test_18 ;;          # group-local PROMOTE (independent A/B groups)
    19)  test_19 ;;          # SWAP both-default: defaults EXCHANGE (dp<->ds)
    20)  test_20 ;;          # SWAP def_p/custom: custom ACQUIRES default_primary
    21)  test_21 ;;          # SWAP custom/custom: NEITHER a default -> configured defaults UNTOUCHED
    22)  test_22 ;;          # SWAP custom/def_s: custom ACQUIRES default_shadow
    23)  test_23 ;;          # scoped SWAP: create+warm fallback, defaults UNTOUCHED
    24)  test_24 ;;          # group-local SWAP (independent A/B groups)
    25)  test_25 ;;          # dedup compaction ON (default): demote-to-limbo + cross-role reuse
    26)  test_26 ;;          # dedup compaction OFF: keep role-tagged, no cross-role reuse
    27)  test_27 ;;          # model-sharding=0: over-cap streams pass through (no inference)
    28)  test_28 ;;          # optional default: no-model streams pass through, mixed with inference
    29)  test_29 ;;          # perf metrics: attach-perf-metric -> fps + latency in model/status
    30)  test_30 ;;          # buffer-copy-mode 0/1/2/3 -> per-lane nvvideoconvert copy
    31)  test_31 ;;          # A/B swap with asymmetric branch rates -- anti-stall (swap-display timeout)
    32)  test_32 ;;          # teardown WHILE reconfig in flight -- UAF tripwire
    33)  test_33 ;;          # downstream source-id/pad-index restore after partial remove
    34)  test_34 ;;          # duplicate add/remove + add-cancel-readd race tripwire
    35)  test_35 ;;          # multi-model config-bound custom primary lazy-load/reuse anti-stall
    36)  test_36 ;;          # rapid churn + repeated mid-churn teardown (lifecycle-race tripwire)
    37)  test_37 ;;          # reserved-separator model names rejected at register
    38)  test_38 ;;          # perf observation: one 30-stream primary ModelBin
    39)  test_39 ;;          # perf observation: 15-stream primary + 15-stream shadow A/B bins
    40)  test_40 ;;          # model/update in-place OTA checkpoint transition (v1 -> v2 -> v3)
    41)  test_41 ;;          # default switch sweeps ONLY default-routed streams; explicitly pinned streams stay -- via_default/origin
    42)  test_42 ;;          # per-model/per-version batch-size: sizing, auto-shard, immutability, clamp, preflight
    43)  test_43 ;;          # pipeline blocks on the default engine build; no passthrough
    44)  test_44 ;;          # model/routing GET: grouped routing snapshot (merge, origin split, filter)
    45)  test_45 ;;          # model/routing GET: origin + default{} sweep (pinned lane is NOT swept)
    46)  test_46 ;;          # model/routing GET: pending + passthrough states
    47)  test_47 ;;          # model/routing GET: accepted-but-deferred route (serving + pending)
    48)  test_48 ;;          # model/routing: newer deferred route SUPERSEDES older (warm order cannot decide)
    49)  test_49 ;;
    50)  test_50 ;;          # immediate route supersedes an older PENDING route
    51)  test_51 ;;
    52)  test_52 ;;
    53)  test_53 ;;          # unified status contract: 400/404/409 + error.code pinned exactly
    54)  test_54 ;;          # unload admission is per-IDENTITY: a non-default version unloads while a sibling holds the default
    55)  test_55 ;;          # default pair: swap / promote (shadow:null) / both-roles conflict
    56)  test_56 ;;          # STATIC config-listed streams: identity parity with stream/add
    57)  test_57 ;;          # STATIC streams, no default: identity survives a LATE bind
    58)  test_58 ;;          # model/load: every version needs its own artifact (ARTIFACT_MISSING)
    59)  test_59 ;;          # SECONDARY stage: in-place OTA, resident load, all-streams rebind
    all) test_1; echo; test_2; echo; test_3; echo; test_4; echo; test_5; echo; test_6; echo; test_7; echo; test_8; echo; test_9; echo; test_10; echo; test_11; echo; test_12; echo; test_13; echo; test_14; echo; test_15; echo; test_16; echo; test_17; echo; test_18; echo; test_19; echo; test_20; echo; test_21; echo; test_22; echo; test_23; echo; test_24; echo; test_25; echo; test_26; echo; test_27; echo; test_28; echo; test_29; echo; test_30; echo; test_31; echo; test_32; echo; test_33; echo; test_34; echo; test_35; echo; test_36; echo; test_37; echo; test_38; echo; test_39; echo; test_40; echo; test_41; echo; test_42; echo; test_43; echo; test_44; echo; test_45; echo; test_46; echo; test_47; echo; test_48; echo; test_49; echo; test_50; echo; test_51; echo; test_52; echo; test_53; echo; test_54; echo; test_55; echo; test_56; echo; test_57; echo; test_58; echo; test_59 ;;
    *)   echo "usage: $0 [-1|0|1..59|all]  (0 = inference-disabled isolation probe; AUTO_APP=0 to skip app management)" >&2
         exit 1 ;;
  esac
}

ALL_TESTS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59"

# start_app: launch the app with the CURRENT config copies and wait until REST-ready. A test
# calls this AFTER its cfg_set edits and BEFORE any REST op. Returns non-zero (and records
# a FAIL) if the app never came up, so the test can `start_app || return`. In AUTO_APP=0
# (external app) it is a no-op -- the app is already running and owns its own config.
start_app() {
  [ "${AUTO_APP}" = "1" ] || return 0
  kill_existing_app_instances
  run_multimodel
  if ! wait_app_ready; then
    echo "[run] ERROR: app not ready" >&2
    _fail "app failed to reach REST-ready (startup)"
    stop_multimodel
    return 1
  fi
  return 0
}

# stop_app: stop the app this case started (AUTO mode only; no-op for external app).
# OPTIONAL -- run_one_auto also stops the app after each case, and stop_multimodel is
# idempotent, so calling stop_app at the end of a test just stops it explicitly/earlier.
stop_app() {
  [ "${AUTO_APP}" = "1" ] || return 0
  stop_multimodel
  # NOTE: the default sanity gates run in run_one_auto (after the case fully ends),
  # NOT here -- so EVERY case is gated exactly once, including the ones that never
  # call stop_app (e.g. the drain-to-zero cases test_3/test_4).
}

# AUTO mode per case: pristine config copy, run the case (which edits config + calls
# start_app + does the work), then stop its app. A FRESH app per case is deliberate:
# every case ends by removing its streams (drain to zero), and re-adding in the next case
# would hit the OPEN drain-to-zero refill bug -- a per-case app keeps cases independent.
run_one_auto() {
  local sel="$1" hold="$2"
  local _rs_pass=${#PASS_LIST[@]} _rs_fail=${#FAIL_LIST[@]}   # snapshot for this case's slice
  APP_LOG="${LOG_DIR}/test_${sel}_logs.txt"   # per-test log under tests/.logs/
  cfg_reset                                  # fresh config copies; the case tweaks them then start_app
  run_case "${sel}"

  # The app quits ONLY when the test calls stop_app. stop_app clears APP_PID, so:
  #  - APP_PID empty  -> the test stopped it; we're done.
  #  - APP_PID alive  -> the test left it running:
  #       single-case (hold) -> HOLD alive until the user presses Ctrl-C
  #       `all` mode         -> stop it so the next case can start its own fresh app
  if [ -z "${APP_PID}" ] || ! kill -0 "${APP_PID}" 2>/dev/null; then
    APP_PID=""
    echo "[run] case '${sel}' done; app stopped by the test (stop_app)."
  elif [ -n "${hold}" ]; then
    echo "[run] case '${sel}' done; app left RUNNING (no stop_app in the test)."
    echo "[run] >>> press Ctrl-C to stop the app and run the sanity checks <<<"
    wait "${APP_PID}" 2>/dev/null
  else
    echo "[run] case '${sel}' done; no stop_app -> stopping app in ${APP_STOP_DELAY}s (all mode) ..."
    sleep "${APP_STOP_DELAY}"
    stop_multimodel
  fi

  # DEFAULT sanity gates -- run ONCE per case here (after the app is fully stopped),
  # so EVERY case is gated, including those that never call stop_app (test_3/18).
  # A short settle lets the app log's final dump flush before we scan.
  sleep 1
  local _tag; _tag="$(basename "${APP_LOG}" _logs.txt 2>/dev/null)"
  if [ "${sel}" = "34" ]; then
    # Test 34 intentionally sends duplicate stream-add REST requests. nvmultiurisrcbin may
    # reject the duplicate by logging "Padname vsrc_N is not unique" as a GStreamer-CRITICAL.
    # Keep the nvmodelmux crash/assert gate strict for every other critical/assert line.
    MM_ALLOW_SOURCEBIN_DUP_PAD_CRITICAL=1 run_default_sanity_checks "${APP_LOG}" "${_tag:-sanity}"
  else
    run_default_sanity_checks "${APP_LOG}" "${_tag:-sanity}"
  fi

  # persist this case's outcome to the per-test ledger (updates only this test's block)
  record_test_result "${sel}" "${_rs_pass}" "${_rs_fail}"
}

# Run a case and capture EVERYTHING into the SINGLE per-test log test_<sel>_logs.txt: the script
# narration (config echoes, >>> REST API prints, [PASS]/[FAIL]) AND the app's stdout/stderr. Process
# substitution keeps run_one_auto running in THIS shell, so the PASS/FAIL counter globals are NOT
# lost to a subshell. The log is truncated ONCE here (run_multimodel no longer truncates per app
# start) and the app feeds this same tee (it has no separate tee of its own).
run_one() {
  local sel="$1" hold="$2"
  APP_LOG="${LOG_DIR}/test_${sel}_logs.txt"   # single combined log for this case
  : > "${APP_LOG}"                            # fresh log per case
  run_one_auto "${sel}" "${hold}" > >(tee -a "${APP_LOG}") 2>&1
}

main() {
  local sel="${1:-all}"
  clear 2>/dev/null || true
  if [ "${AUTO_APP}" = "1" ]; then
    # SINGLE-INSTANCE guard: two concurrent suites share REST port ${PORT}, the
    # .logs directory and the mutable config copies -- they silently corrupt each
    # other's results (REST commands land on the OTHER run's app; per-test logs
    # are truncated mid-assertion; cases appear to "skip around"). Abort loudly
    # instead. The flock releases automatically when this shell exits, however it
    # exits (Ctrl-C, kill, crash) -- no stale-lock handling needed.
    exec 9>"${LOG_DIR}/.run_tests.lock"
    if ! flock -n 9; then
      echo "[run] ERROR: another run_tests.sh instance is already running" >&2
      echo "[run]        (lock: ${LOG_DIR}/.run_tests.lock -- wait for it to" >&2
      echo "[run]        finish, or kill it and its deepstream-test5-app, then retry)" >&2
      exit 1
    fi
    run_setup_check || { _fail "test setup failed (plugin/engine precheck)"; print_summary; return; }
    # ensure the app is always stopped, even on a failed case or Ctrl-C
    trap 'stop_multimodel' EXIT INT TERM
    if [ "${sel}" = "all" ]; then
      for t in ${ALL_TESTS}; do
        echo "===================== AUTO case ${t} (fresh app) ====================="
        run_one "${t}"
        echo
      done
    else
      run_one "${sel}" hold     # single case: if the test omits stop_app, hold app alive
    fi
    trap - EXIT INT TERM
  else
    local _rs_pass=${#PASS_LIST[@]} _rs_fail=${#FAIL_LIST[@]}
    run_case "${sel}"           # app is managed externally (two-terminal workflow)
    [ "${sel}" != "all" ] && record_test_result "${sel}" "${_rs_pass}" "${_rs_fail}"
  fi
  print_summary                 # exits non-zero if any assert_log FAILED
}

# Allow other harnesses (e.g. run_stress.sh) to source this file for its helpers
# (cfg_set, start_app/stop_app, add_stream, load_model, wait_for_model, assert_*, ...)
# without running the functional suite: set RUN_TESTS_LIB_ONLY=1 before sourcing.
if [ "${RUN_TESTS_LIB_ONLY:-0}" != "1" ]; then
  main "${1:-all}"
fi
