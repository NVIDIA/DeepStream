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
# run_stress.sh -- stress / perf / soak / memory-leak harness for the nvmodelmux
# PLUGIN. Companion to run_tests.sh (functional correctness). This one is about
# LONG RUNS and RESOURCE behaviour: it drives a workload for a duration while
# sampling CPU (RSS) + GPU memory over time, dumps the samples to CSV, and prints
# a leak verdict at the end.
#
# It REUSES run_tests.sh's helpers (sourced in library mode: cfg_reset, cfg_set,
# comment_cfg, start_app/stop_app, add_stream, load_model, wait_for_model, ...)
# and mem_monitor.sh's sampling/analysis. Only the workload + monitoring wrapper
# lives here.
#
# Usage:
#   ./run_stress.sh [test]        # no arg => ALL tests; or 1|2|3|4; "list" to list
# Knobs (env):
#   STRESS_DURATION  seconds to hold the workload under monitoring   (default 300)
#   SAMPLE_INTERVAL  seconds between resource samples                (default 3)
#   NUM_STREAMS      streams to run                                  (default 4)
#   STRESS_STREAM    source URI (defaults to run_tests.sh's STREAM)
#   FILE_LOOP        loop file sources so they sustain the soak      (default 1)
#   plus every run_tests.sh knob (APP_BIN, DS_ROOT, HOST, PORT, AUTO_APP, ...)
#   plus every mem_monitor.sh leak threshold (LEAK_RSS_MB_PER_MIN, ...)
###############################################################################

# NOTE: no 'set -u' -- we source run_tests.sh, whose reused helpers (add_stream,
# load_model, ...) intentionally reference optional positional args ($4..$7);
# nounset would make those throw "unbound variable". All our own vars use ${x:-default}.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Pull in the functional harness as a LIBRARY (no auto-run) + the monitor.
RUN_TESTS_LIB_ONLY=1 source "${HERE}/run_tests.sh"
source "${HERE}/mem_monitor.sh"

# 2) Stress-specific knobs (run_tests.sh already set HOST/PORT/STREAM/LOG_DIR/etc.)
STRESS_DURATION="${STRESS_DURATION:-3600}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-3}"
PERF_METRIC="${PERF_METRIC:-1}"                     # 1=enable model/role fps+latency in model/status
PERF_METRIC_INTERVAL="${PERF_METRIC_INTERVAL:-1}"   # perf measurement window (s)
NUM_STREAMS="${NUM_STREAMS:-30}"
STRESS_STREAM="${STRESS_STREAM:-${STREAM}}"
FILE_LOOP="${FILE_LOOP:-1}"
PGIE_ENABLE="${PGIE_ENABLE:-1}"                     # stress_0/1 control default: plain nvinfer (plugin-type=0).
                                                    # 1 keeps stress_0/1 the CLOSEST controls to stress_2/3 -- same
                                                    # inference load, only nvmodelmux removed -- so a difference
                                                    # between them isolates modelmux rather than "modelmux AND
                                                    # inference". Set 0 to strip inference too and get the bare
                                                    # decode/streammux/source-churn pipeline.
# async random-churn knobs (stress_1 / 3 / 6)
CHURN_BATCH="${CHURN_BATCH:-5}"            # streams removed+re-added per churn cycle
CHURN_DOWN="${CHURN_DOWN:-5}"              # seconds the picked streams stay removed
CHURN_GAP="${CHURN_GAP:-10}"               # seconds between churn cycles
PER_MODEL_BATCH="${PER_MODEL_BATCH:-}"     # per-model batch; empty => each test sizes it to its stream
                                           # count so ONE engine (b<N>) serves it -- no auto-sharding.
AB_STREAMS="${AB_STREAMS:-15}"             # stress_4 A/B stream count (primary+shadow => ~2x compute)
SWAP_STREAMS="${SWAP_STREAMS:-30}"         # stress_6/7 total streams (half on model A, half on model B)
CHURN_PID=""                               # bg async-churn/swap worker pid (stress_1/3/5)
DUMP_DIR="${DUMP_DIR:-${LOG_DIR}/stress}"
STRESS_RESULTS="${STRESS_RESULTS:-${LOG_DIR}/stress_results.txt}"   # at-a-glance summary across tests
LEAK_TRACER="${LEAK_TRACER:-1}"    # 1=GStreamer leaks tracer ON by default (duration-independent leak proof); set 0 to disable
_STRESS_FAIL0=0                                                    # FAIL_LIST snapshot before a test

# _tracer_summary <app_log> : DURATION-INDEPENDENT leak verdict from the GStreamer leaks
# tracer (LEAK_TRACER=1). Unlike the RSS median (which grows during warm-up/allocator
# retention regardless of leaks), the tracer counts objects NOT unref'd at shutdown -- a
# real leak shows the same non-zero count whether the soak was 10 min or 30 min; a clean
# run shows CLEAN either way. States: off (tracer not enabled) · NO-EXIT (app killed before
# gst_deinit, so no report) · CLEAN · LEAKED(n).
_tracer_summary() {
  local log="$1" n=""
  [ "${LEAK_TRACER:-0}" = "1" ] || { printf "tracer:off"; return 0; }
  [ -f "${log}" ] || { printf "tracer:n/a"; return 0; }
  grep -aq "GST_TRACER" "${log}" 2>/dev/null || { printf "tracer:NO-EXIT"; return 0; }   # no clean shutdown -> no report
  # Prefer the tracer's explicit summary count ("N objects still alive" / "N leaked"); the
  # zero case ("0 objects...") correctly reads CLEAN. Fall back to counting per-object leak
  # lines, excluding our own console echoes about the tracer.
  n=$(grep -aoiE '[0-9]+[[:space:]]+(leaked|live objects|objects[^.]*(leaked|alive))' "${log}" 2>/dev/null | grep -aoE '[0-9]+' | sort -rn | head -1)
  if [ -z "${n}" ]; then
    n=$(grep -ai 'leaked' "${log}" 2>/dev/null | grep -aviE 'grep the app log|tracer ENABLED|leaks tracer|GST_TRACERS' | wc -l | tr -d ' ')
  fi
  n=${n:-0}
  if [ "${n}" -gt 0 ]; then printf "tracer:LEAKED(%s)" "${n}"; else printf "tracer:CLEAN"; fi
}

# _stress_record <name> <csv> [app_log] : append one summary line (gate PASS/FAIL + RSS-median
# leak screen + duration-independent tracer verdict + key memory numbers) to STRESS_RESULTS.
_stress_record() {
  local nm="$1" csv="$2" log="${3:-${APP_LOG}}" nfail=$(( ${#FAIL_LIST[@]} - _STRESS_FAIL0 ))
  local gate="PASS"; [ "${nfail}" -gt 0 ] && gate="FAIL(${nfail})"
  local memline lk tracer
  memline="$(mem_leak_oneline "${csv}")"; lk=$?
  # exit 0 = settled, 1 = leak (sustained rise), 2 = plateauing (bounded fill,
  # rate decaying to <=half the earlier rate -- allocator/pool residency, not a leak)
  local leak="OK"
  [ "${lk}" -eq 1 ] && leak="LEAK"
  [ "${lk}" -eq 2 ] && leak="PLATEAU"
  tracer="$(_tracer_summary "${log}")"
  printf "%-9s | gates:%-9s | leak:%-4s | %-16s | %s\n" "${nm}" "${gate}" "${leak}" "${tracer}" "${memline}" >> "${STRESS_RESULTS}"
}
mkdir -p "${DUMP_DIR}" "${LOG_DIR}" 2>/dev/null || true

# --- optional instrumentation layers (industry-standard leak localisation) -----
# The soak + steady-state regression above is the SCREEN (is it growing?). These
# opt-in layers CONFIRM + localise a leak on the real pipeline:
#
#   LEAK_TRACER=1        GStreamer object-leak tracer -> the app prints any
#                        GstBuffer/GstEvent/pad/GstObject not unref'd at shutdown
#                        (ideal for the add/remove churn; grep app log for "leaked").
#   GPU_SANITIZER=1      run under `compute-sanitizer --leak-check full` (NVIDIA's
#                        GPU-memory leak checker; needs it on PATH; slows the run).
#
# GST_TRACERS is inherited by the app (run_multimodel doesn't override it), and
# run_multimodel PREPENDS our GST_DEBUG, so GST_TRACER:7 survives.
if [ "${LEAK_TRACER:-1}" = "1" ]; then
  export GST_TRACERS="leaks"
  export GST_DEBUG="${GST_DEBUG:+${GST_DEBUG},}GST_TRACER:7"
  echo "[stress] GStreamer leaks tracer ENABLED -> leak report prints at app shutdown (grep the app log for 'leaked')"
  # the tracer only prints on a CLEAN shutdown, so give the app time to quit before SIGKILL
  APP_STOP_TIMEOUT="${APP_STOP_TIMEOUT:-30}"
fi
if [ "${GPU_SANITIZER:-0}" = "1" ]; then
  if command -v compute-sanitizer >/dev/null 2>&1; then
    # run_multimodel invokes "${APP_BIN}" as one token, so point APP_BIN at a
    # generated wrapper that execs compute-sanitizer + the real app.
    _san_wrap="${LOG_DIR}/.compute_sanitizer_app.sh"
    _real_app="$(command -v "${APP_BIN:-deepstream-test5-app}")"
    { echo '#!/usr/bin/env bash'
      echo "exec compute-sanitizer --tool memcheck --leak-check full \"${_real_app}\" \"\$@\""
    } > "${_san_wrap}"; chmod +x "${_san_wrap}"
    APP_BIN="${_san_wrap}"
    echo "[stress] GPU leak checker ENABLED (compute-sanitizer --leak-check full) -> report in the app log"
  else
    echo "[stress] WARN: GPU_SANITIZER=1 but compute-sanitizer not on PATH -- skipping"
  fi
fi

# Always tear down the app + monitor, even on Ctrl-C / error.
_stress_cleanup() { [ -n "${CHURN_PID}" ] && kill "${CHURN_PID}" 2>/dev/null; mem_monitor_stop 2>/dev/null; stop_app 2>/dev/null; }
trap '_stress_cleanup' EXIT INT TERM

# _wait_serving <n_streams> [timeout_s]
# Block until <n_streams> streams are WIRED to a model (i.e. the model engine is
# built/loaded and inference actually started), so the leak baseline is taken at STEADY
# STATE. This excludes the ONE-TIME on-the-fly TensorRT engine build (can be ~100s the
# first time a b<N> engine is generated) and the passthrough ramp -- otherwise those
# inflate the RSS/GPU slope and false-flag a leak. On later runs the engine is cached so
# this returns in seconds.
#
# Count the 'WIRED (primary=...)' marker, NOT 'PROMOTED passthrough': the plugin emits
# the SAME 'WIRED' line on BOTH attach paths -- direct attach (model already warm at
# add time) and deferred promote (stream added while model still warming). When engines
# are pre-built (stress_setup.sh) and primaries pre-warmed via load_model (stress_4/5/6),
# every stream binds directly and NEVER traverses passthrough, so 'PROMOTED passthrough'
# is emitted 0 times -- gating on it would stall the full timeout and the churn/swap
# worker would never launch. 'WIRED' is the path-independent serving signal.
_wait_serving() {
  local want="$1" timeout="${2:-300}" i=0 c
  echo "[stress] waiting for ${want} streams to reach serving (engine build/load + wire) ..."
  while [ "${i}" -lt "${timeout}" ]; do
    [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null && return 0
    c=$(grep -c 'WIRED  (primary=' "${APP_LOG}" 2>/dev/null)
    if [ "${c:-0}" -ge "${want}" ]; then echo "[stress] ${c}/${want} streams serving after ${i}s -> steady state; start monitoring"; return 0; fi
    sleep 2; i=$((i+2))
  done
  echo "[stress] WARN: only $(grep -c 'WIRED  (primary=' "${APP_LOG}" 2>/dev/null)/${want} serving after ${timeout}s -- proceeding anyway"
}

# _stress_sanity <app_log> <tag>
# Functional health gate for a stress run -- the SAME regression checks run_tests.sh
# applies at the end of every case, reused here so a stress run also catches a mid-run
# crash / frame loss / wedge, not just leaks. We run the CHURN-ROBUST subset:
#   - assert_no_frame_drops     : ZERO real frame loss (frameacct dropped==0)   [needs MM_DEBUG=1]
#   - assert_no_inference_stalls: no wedged stream (buffered grows while inferred=0)
#   - assert_no_gst_critical    : no GStreamer-CRITICAL / assertion / Segmentation fault / (core dumped)
# The strict per-source frame-gap / source-id-restore contiguity gates are intentionally
# NOT run: stress_2 removes+re-adds the SAME camera id, which legitimately resets that
# source's frame numbering and would false-fail those two.
_stress_sanity() {
  local log="${1:-${APP_LOG}}" tag="${2:-stress}"
  sleep 1                                   # let the app's final frameacct dump flush
  echo "[${tag}] ===== stress sanity checks (log: ${log##*/}) ====="
  assert_no_frame_drops      "${log}" "${tag}: ZERO frame loss (frameacct dropped==0)"
  assert_no_inference_stalls "${log}" "${tag}: no inference stall (buffered grows with inferred=0)"
  assert_no_gst_critical     "${log}" "${tag}: no GStreamer-CRITICAL / assertion / crash / segfault"
}

# _set_cfg_engine_batch <per-model-config-basename> <batch>
# Rewrite model-engine-file in the LOG_DIR copy of a per-model nvinfer config so it
# names the b<batch> engine (e.g. ..._b4_gpu0_ -> ..._b30_gpu0_). nvinfer then loads
# that engine, or rebuilds it from onnx-file on the fly at max-batch=<batch>. Must be
# called AFTER cfg_reset (which copies the configs into LOG_DIR).
_set_cfg_engine_batch() {
  local file="${LOG_DIR}/$1" batch="$2"
  [ -f "${file}" ] || { echo "[stress] WARN: ${1} not in LOG_DIR -- cannot set engine batch"; return 0; }
  sed -i -E "s#(model-engine-file=.*_b)[0-9]+(_gpu0_)#\1${batch}\2#" "${file}"
  echo "[stress] ${1}: model-engine-file -> b${batch} ($(grep -oE '[^/]*_b[0-9]+_gpu0_[^ ]*\.engine' "${file}" | head -1))"
}

###############################################################################
# stress_1 -- CONTROL: identical soak+churn workload as stress_3, but with NO
#   nvmodelmux. The [primary-gie] is plain nvinfer (plugin-type=0) pointed straight
#   at the default detector config (config_infer_primary.txt). Same nvmultiurisrcbin
#   REST add/remove churn, same streams, same b<N> engine. Purpose: attribute the
#   stress_3 RSS growth -- if stress_1 (no modelmux) ALSO grows, the leak is in the
#   DeepStream core/decoder/nvinfer churn path, NOT nvmodelmux. If stress_1 is flat
#   while stress_3 climbs, the growth is modelmux's. Compare the leak: columns.
#
#   Knobs: same as stress_3 -- NUM_STREAMS(30), CHURN_BATCH(5), CHURN_DOWN, CHURN_GAP,
#          STRESS_DURATION. NVINFER_WARMUP(15s) replaces _wait_serving (nvinfer emits
#          no 'WIRED' marker).
#          PGIE_ENABLE(default 1): 1 -> plain nvinfer (plugin-type=0); 0 -> [primary-gie]
#          disabled (NO inference at all, pure decode/streammux/source-churn). Both
#          reproduced the same ~55 MB/min RSS growth -> that leak is DeepStream-core
#          source churn, not modelmux.
#
#          Default is 1 on purpose: it makes stress_1 differ from stress_3 in exactly
#          ONE way (nvmodelmux in/out) instead of two (modelmux AND inference), so the
#          comparison actually isolates modelmux. If something reproduces here, drop to
#          PGIE_ENABLE=0 to see whether inference was needed to trigger it at all.
###############################################################################
stress_1() {
  local name="stress_1"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"
  local n="${NUM_STREAMS}" cb="${CHURN_BATCH}" cdown="${CHURN_DOWN}" cgap="${CHURN_GAP}"
  local eb="${PER_MODEL_BATCH:-${NUM_STREAMS}}"
  local cols rows
  cols=$(awk -v x="${n}" 'BEGIN{c=int(sqrt(x)+0.999); if(c<1)c=1; print c}')
  rows=$(awk -v x="${n}" -v c="${cols}" 'BEGIN{print int((x+c-1)/c)}')
  echo "-----------------------------------------------------------------------"
  local pgie_mode
  [ "${PGIE_ENABLE}" = "0" ] && pgie_mode="[primary-gie] DISABLED -> NO inference" \
                               || pgie_mode="plain nvinfer plugin-type=0"
  echo "[${name}] CONTROL (NO modelmux, ${pgie_mode}): soak + async random churn: ${n} streams (tiler ${rows}x${cols}), churn ${cb}/cycle (down ${cdown}s, gap ${cgap}s), hold ${STRESS_DURATION}s"

  # 1) config sized for n streams; NO nvmodelmux. plugin-type is ALWAYS forced to 0
  #    here (the base config ships plugin-type=4 -> nvmodelmux), so this workload never
  #    loads modelmux. PGIE_ENABLE then only decides whether that nvinfer runs:
  #      PGIE_ENABLE=1 (default) -> plain nvinfer (plugin-type=0) on the default detector
  #                                 config -- same inference load as stress_3, minus modelmux
  #      PGIE_ENABLE=0           -> [primary-gie] disabled -> NO inference at all
  #                                 (pipeline just nvmultiurisrcbin -> streammux -> tiler -> sink)
  cfg_reset
  cfg_main primary-gie enable      "${PGIE_ENABLE}"
  cfg_main primary-gie plugin-type 0
  cfg_main primary-gie config-file config_infer_primary.txt
  cfg_set source-list   max-batch-size "${n}"
  cfg_set tiled-display rows            "${rows}"
  cfg_set tiled-display columns         "${cols}"
  [ "${FILE_LOOP}" = "1" ] && cfg_main tests file-loop 1
  _set_cfg_engine_batch config_infer_primary.txt "${eb}"     # use the b${eb} engine
  # nvinfer needs batch-size >= stream count to batch all sources
  sed -i -E "s/^batch-size=.*/batch-size=${n}/" "${LOG_DIR}/config_infer_primary.txt" 2>/dev/null || true

  # 2) start app
  start_app || return 1

  # 3) bring all n streams up one-by-one
  echo "[${name}] adding ${n} streams ..."
  local i
  for (( i=0; i<n; i++ )); do
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"
    sleep 1
  done
  echo "[${name}] all ${n} streams added"

  # 4) warm-up then begin sampling (plain nvinfer emits no 'WIRED' marker, so no _wait_serving)
  sleep "${NVINFER_WARMUP:-15}"
  echo "[${name}] warmed up (${NVINFER_WARMUP:-15}s) -> baseline; start monitoring"
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"

  # 5) async random churn for the whole hold (same worker as stress_3 -- model-agnostic REST)
  _stress2_churn "${n}" "${cb}" "${cdown}" "${cgap}" &
  CHURN_PID=$!
  echo "[${name}] async churn started (pid ${CHURN_PID})"

  # 6) hold under load for STRESS_DURATION (abort if app dies)
  local waited=0
  while [ "${waited}" -lt "${STRESS_DURATION}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited early at ${waited}s (check ${APP_LOG})"; break
    fi
    sleep 5; waited=$((waited+5))
  done

  # 7) stop churn + sampling + app, then sanity + leak report
  [ -n "${CHURN_PID}" ] && kill "${CHURN_PID}" 2>/dev/null; wait "${CHURN_PID}" 2>/dev/null; CHURN_PID=""
  mem_monitor_stop
  stop_app
  _stress_sanity "${APP_LOG}" "${name}"
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

###############################################################################
# stress_0 / stress_2 -- drain-to-zero add/remove churn + leak watch.
#   Repeats  add stream -> hold -> remove stream -> drain  for STRESS_CYCLES
#   cycles (default 20), exactly ONE stream active at a time (the freed slot is
#   reused, so we never exceed max-batch-size). This hammers the dynamic
#   add/remove + inner mux/demux recreate path over a long run while the monitor
#   samples CPU (VmRSS/slab) + GPU memory, so any per-cycle leak shows up as a
#   sustained upward slope in the end-of-run report.
#
#   Knobs: STRESS_CYCLES (default 20), ADD_HOLD (default 5s), REMOVE_HOLD (5s).
###############################################################################
_stress_drain_to_zero() {
  local name="$1" use_mm="$2"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"   # run_one_stress sets+truncates + tees the whole case
  local cycles="${STRESS_CYCLES:-50}"
  local add_hold="${ADD_HOLD:-5}" rm_hold="${REMOVE_HOLD:-5}"
  echo "-----------------------------------------------------------------------"
  local mm_mode
  [ "${use_mm}" = "1" ] && mm_mode="nvmodelmux" || mm_mode="NO modelmux (plain nvinfer plugin-type=0)"
  echo "[${name}] drain-to-zero add/remove churn + leak watch [${mm_mode}]: ${cycles} cycles (add+hold ${add_hold}s / remove+drain ${rm_hold}s)"

  # 1) config: single stream at a time (slot reused each cycle). The two variants
  #    differ ONLY in which element sits in [primary-gie] -- same cycle count, same
  #    holds, same batch, same source -- so a difference between them isolates
  #    modelmux rather than "modelmux and something else".
  cfg_reset
  if [ "${use_mm}" = "1" ]; then
    cfg_set multimodel attach-perf-metric      "${PERF_METRIC}"          # per-model/role fps+latency (model/status)
    cfg_set multimodel perf-metric-interval-sec "${PERF_METRIC_INTERVAL}" # measurement window (s)
    cfg_set multimodel    unified-batch         0
    cfg_set multimodel    per-model-batch-size  2
  else
    # base config ships plugin-type=4 -> force it to plain nvinfer
    cfg_main primary-gie enable      "${PGIE_ENABLE}"
    cfg_main primary-gie plugin-type 0
    cfg_main primary-gie config-file config_infer_primary.txt
    _set_cfg_engine_batch config_infer_primary.txt 2      # b2 engine (built on the fly if absent)
    sed -i -E "s/^batch-size=.*/batch-size=2/" "${LOG_DIR}/config_infer_primary.txt" 2>/dev/null || true
  fi
  cfg_set source-list   max-batch-size        2
  cfg_set tiled-display rows                  1
  cfg_set tiled-display columns               2

  # 2) start app (populates APP_PID, waits for REST-ready). Capture the app's
  #    stdout+stderr into APP_LOG via tee (also shown live) so FPS parsing and
  #    post-run log checks work -- run_multimodel does not redirect on its own.
  start_app || return 1

  # 3) PRIME once (add -> hold -> remove) BEFORE monitoring, so the ONE-TIME
  #    startup ramp (CUDA context, decoder, first model serving path) is EXCLUDED
  #    from the leak baseline. Without this the ramp inflates the RSS/GPU slope
  #    and false-flags a leak. Prime uses Cam-0; the churn below uses Cam-1..N.
  echo "[${name}] priming (1 add/remove) to warm CUDA/decoder before monitoring ..."
  add_stream "${PREFIX}0" "Cam-0" "${STRESS_STREAM}"; sleep "${add_hold}"
  remove_stream "${PREFIX}0" "${STRESS_STREAM}";      sleep "${rm_hold}"

  # 4) sample resources across the WHOLE churn (baseline now = post-ramp steady state)
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"

  # 5) churn: add -> hold -> remove -> drain, N cycles, one stream at a time
  local i
  for (( i=1; i<=cycles; i++ )); do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited at cycle ${i} (see ${APP_LOG})"; break
    fi
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"
    echo "[${name}] cycle ${i}/${cycles}: added Cam-${i} (hold ${add_hold}s)"
    sleep "${add_hold}"
    remove_stream "${PREFIX}${i}" "${STRESS_STREAM}"
    echo "[${name}] cycle ${i}/${cycles}: removed Cam-${i} (drain ${rm_hold}s)"
    sleep "${rm_hold}"
  done

  # 6) stop sampling + app, then functional gate + leak report over the full run.
  #    FULL gate set here: stress_0/2 use a UNIQUE cam id per cycle, so per-source
  #    frame contiguity holds and the strict frame-gap / source-id-restore checks
  #    are valid (unlike stress_3's same-id re-add churn).
  mem_monitor_stop
  stop_app
  sleep 1                                   # let the app's final frameacct dump flush
  if [ "${use_mm}" = "1" ]; then
    run_default_sanity_checks "${APP_LOG}" "${name}"   # frameacct/WIRED markers exist
  else
    _stress_sanity "${APP_LOG}" "${name}"              # no modelmux -> no frameacct rows
  fi
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

stress_2() { _stress_drain_to_zero "stress_2" 1; }   # WITH modelmux
stress_0() { _stress_drain_to_zero "stress_0" 0; }   # control: same cycle, no modelmux

# _stress2_churn <n> <batch> <down_s> <gap_s>
# Background worker: every cycle pick <batch> RANDOM streams from [0,n), remove
# them one-by-one, wait <down_s>, re-add the SAME ones one-by-one, wait <gap_s>.
# Runs until the app dies or it's killed at test end. curl noise is suppressed;
# only the [churn] summary lines are printed.
_stress2_churn() {
  local n="$1" batch="$2" down="$3" gap="$4"
  local picks id cyc=0
  while kill -0 "${APP_PID}" 2>/dev/null; do
    cyc=$((cyc+1))
    picks=$(seq 0 $((n-1)) | shuf 2>/dev/null | head -n "${batch}")
    [ -z "${picks}" ] && picks=$(seq 0 $((batch-1)))     # fallback if no shuf
    echo "[churn] cycle ${cyc}: remove+readd $(echo ${picks} | tr '\n' ' ')"
    for id in ${picks}; do
      kill -0 "${APP_PID}" 2>/dev/null || return
      remove_stream "${PREFIX}${id}" "${STRESS_STREAM}"
      sleep 0.3
    done
    sleep "${down}"
    for id in ${picks}; do
      kill -0 "${APP_PID}" 2>/dev/null || return
      add_stream "${PREFIX}${id}" "Cam-${id}" "${STRESS_STREAM}"
      sleep 0.3
    done
    sleep "${gap}"
  done
}

###############################################################################
# stress_3 -- steady N-stream soak + ASYNC random churn + leak watch.
#   Brings NUM_STREAMS (default 30) up one-by-one on the default primary, then
#   runs a background worker that repeatedly removes CHURN_BATCH (default 5)
#   RANDOM streams one-by-one, waits CHURN_DOWN, re-adds the same ones, waits
#   CHURN_GAP -- for the whole STRESS_DURATION. So there is always a near-full
#   set of streams under real inference load while the add/remove path is
#   exercised continuously, and the monitor watches for CPU/GPU/slab drift.
#   The churn is async: when the stress time is up we just stop (leftover
#   removed streams need no draining).
#
#   Knobs: NUM_STREAMS(30), CHURN_BATCH(5), CHURN_DOWN(5s), CHURN_GAP(10s),
#          PER_MODEL_BATCH(4 = engine max-batch; >N auto-shards), STRESS_DURATION.
###############################################################################
stress_3() {
  local name="stress_3"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"   # run_one_stress sets+truncates + tees the whole case
  local n="${NUM_STREAMS}" cb="${CHURN_BATCH}" cdown="${CHURN_DOWN}" cgap="${CHURN_GAP}"
  local eb="${PER_MODEL_BATCH:-${NUM_STREAMS}}"   # one engine (b<N>) serves all -> no sharding
  local cols rows
  cols=$(awk -v x="${n}" 'BEGIN{c=int(sqrt(x)+0.999); if(c<1)c=1; print c}')   # ~square tiler grid
  rows=$(awk -v x="${n}" -v c="${cols}" 'BEGIN{print int((x+c-1)/c)}')
  echo "-----------------------------------------------------------------------"
  echo "[${name}] soak + async random churn + leak watch: ${n} streams (tiler ${rows}x${cols}), churn ${cb}/cycle (down ${cdown}s, gap ${cgap}s), hold ${STRESS_DURATION}s"

  # 1) config sized for n streams; default primary; loop sources for the soak
  cfg_reset
  cfg_set multimodel attach-perf-metric      "${PERF_METRIC}"          # per-model/role fps+latency (model/status)
  cfg_set multimodel perf-metric-interval-sec "${PERF_METRIC_INTERVAL}" # measurement window (s)
  cfg_set source-list   max-batch-size        "${n}"
  cfg_set tiled-display rows                  "${rows}"
  cfg_set tiled-display columns               "${cols}"
  cfg_set multimodel    unified-batch         0
  cfg_set multimodel    per-model-batch-size  "${eb}"       # = stream count -> single b${eb} engine, no shard
  [ "${FILE_LOOP}" = "1" ] && cfg_main tests file-loop 1
  _set_cfg_engine_batch config_infer_primary.txt "${eb}"    # use the b${eb} engine (built on the fly)

  # 2) start app; capture app log via tee
  start_app || return 1

  # 3) bring all n streams up one-by-one
  echo "[${name}] adding ${n} streams ..."
  local i
  for (( i=0; i<n; i++ )); do
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"
    sleep 1
  done
  echo "[${name}] all ${n} streams added; steady state reached"

  # 4) begin sampling (baseline = all streams up + warmed)
  _wait_serving "${n}"     # take the baseline only once all streams are serving (model loaded)
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"

  # 5) launch async random churn for the whole hold
  _stress2_churn "${n}" "${cb}" "${cdown}" "${cgap}" &
  CHURN_PID=$!
  echo "[${name}] async churn started (pid ${CHURN_PID})"

  # 6) hold under load for STRESS_DURATION (abort if app dies)
  local waited=0
  while [ "${waited}" -lt "${STRESS_DURATION}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited early at ${waited}s (check ${APP_LOG})"; break
    fi
    sleep 5; waited=$((waited+5))
  done

  # 7) stop churn + sampling + app, then functional gate + leak report
  [ -n "${CHURN_PID}" ] && kill "${CHURN_PID}" 2>/dev/null; wait "${CHURN_PID}" 2>/dev/null; CHURN_PID=""
  mem_monitor_stop
  stop_app
  _stress_sanity "${APP_LOG}" "${name}"
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

###############################################################################
# stress_4 -- A/B two-version STEADY soak + leak watch (NO churn).
#   Same A/B config as stress_4 (primary=<default>;v1 WORST engine, shadow=<default>;v2
#   BEST engine, unified-batch=1, b<AB_STREAMS>), but brings AB_STREAMS (default 15) A/B
#   streams up ONCE and just HOLDS -- no add/remove at all. This is the STABLE-PIPELINE
#   baseline for the A/B case: any RSS/GPU drift here (with zero churn) is a steady-state
#   leak, not a churn-path artefact. Streams are never removed -> contiguity holds -> FULL gate set.
#
#   Knobs: AB_STREAMS(15), PER_MODEL_BATCH, STRESS_DURATION.
###############################################################################
stress_4() {
  local name="stress_4"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"   # run_one_stress sets+truncates + tees the whole case
  local n="${AB_STREAMS}"
  local eb="${PER_MODEL_BATCH:-${AB_STREAMS}}"    # b<N> per version -> no sharding
  local worst_eng="${WORST_ENGINE/_b[0-9]*_gpu0_/_b${eb}_gpu0_}"   # b<eb> int8 nocalib (stress_setup.sh)
  local best_eng="${BEST_ENGINE/_b[0-9]*_gpu0_/_b${eb}_gpu0_}"     # b<eb> fp32 best
  local v1="${DEFAULT_MODEL_VERSION}" v2="${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "-----------------------------------------------------------------------"
  echo "[${name}] A/B STEADY soak (v${v1} worst-int8 / v${v2} best-fp32, b${eb}), NO churn: ${n} streams x2 (~$((n*2)) compute), hold ${STRESS_DURATION}s"

  # 1) config: unified A/B; primary=v1(worst int8) + shadow=v2(best fp32) of the default model
  cfg_reset
  cfg_set multimodel attach-perf-metric      "${PERF_METRIC}"          # per-model/role fps+latency (model/status)
  cfg_set multimodel perf-metric-interval-sec "${PERF_METRIC_INTERVAL}" # measurement window (s)
  cfg_set source-list   max-batch-size        "${n}"
  cfg_set tiled-display rows                  2          # A/B view: row0 primary, row1 shadow
  cfg_set tiled-display columns               "${n}"
  cfg_set multimodel    unified-batch         1
  cfg_set multimodel    per-model-batch-size  "${eb}"       # b${eb} per version -> no shard
  cfg_set multimodel    primary-model         "${DEFAULT_MODEL};${v1}"
  cfg_set multimodel    shadow-model          "${DEFAULT_MODEL};${v2}"
  cfg_set model-${DEFAULT_MODEL}-${v1}        engine "${worst_eng}"    # v1 = worst (int8 nocalib)
  cfg_set model-${DEFAULT_MODEL}-${v2}        engine "${best_eng}"     # v2 = best (fp32)
  [ "${FILE_LOOP}" = "1" ] && cfg_main tests file-loop 1

  # 2) start app
  start_app || return 1

  # 3) bring all n A/B streams up ONCE (each runs primary v1 + shadow v2)
  echo "[${name}] adding ${n} A/B streams ..."
  local i
  for (( i=0; i<n; i++ )); do add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"; sleep 1; done
  echo "[${name}] all ${n} A/B streams added; steady state reached"

  # 4) begin sampling (baseline = all streams up + both versions warmed)
  _wait_serving "${n}"
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"

  # 5) HOLD steady (NO churn) for STRESS_DURATION (abort if app dies)
  local waited=0
  while [ "${waited}" -lt "${STRESS_DURATION}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited early at ${waited}s (check ${APP_LOG})"; break
    fi
    sleep 5; waited=$((waited+5))
  done

  # 6) stop sampling + app, then FULL functional gate + leak report (no removals -> contiguity holds)
  mem_monitor_stop
  stop_app
  sleep 1
  run_default_sanity_checks "${APP_LOG}" "${name}"
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

###############################################################################
# stress_5 -- A/B two-version soak (worst v1 primary / best v2 shadow) + churn.
#   primary = <default>;v1 with the WORST engine, shadow = <default>;v2 with the
#   BEST engine, unified-batch=1 so primary+shadow share the combined A/B batch.
#   With AB_STREAMS (default 15) sources, BOTH models run per stream -> ~2x
#   AB_STREAMS (30) inferences of compute. Then the same async random churn as
#   stress_2 runs for the whole hold. Uses the churn-robust gate subset (same-id
#   re-add resets per-source frame numbering, so contiguity gates are skipped).
#
#   Knobs: AB_STREAMS(15), CHURN_BATCH(5), CHURN_DOWN(5s), CHURN_GAP(10s),
#          PER_MODEL_BATCH(4 = engine max-batch), STRESS_DURATION.
###############################################################################
stress_5() {
  local name="stress_5"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"   # run_one_stress sets+truncates + tees the whole case
  local n="${AB_STREAMS}" cb="${CHURN_BATCH}" cdown="${CHURN_DOWN}" cgap="${CHURN_GAP}"
  local eb="${PER_MODEL_BATCH:-${AB_STREAMS}}"    # b<N> per version -> no sharding
  local worst_eng="${WORST_ENGINE/_b[0-9]*_gpu0_/_b${eb}_gpu0_}"   # b<eb> int8 nocalib (built by stress_setup.sh)
  local best_eng="${BEST_ENGINE/_b[0-9]*_gpu0_/_b${eb}_gpu0_}"     # b<eb> fp32 best
  local v1="${DEFAULT_MODEL_VERSION}" v2="${DEFAULT_SHADOW_MODEL_VERSION}"
  echo "-----------------------------------------------------------------------"
  echo "[${name}] A/B soak (v${v1} worst-int8 / v${v2} best-fp32, b${eb}) + async churn: ${n} streams x2 (~$((n*2)) compute), churn ${cb}/cycle (down ${cdown}s, gap ${cgap}s), hold ${STRESS_DURATION}s"

  # 1) config: unified A/B; primary=v1(worst int8) + shadow=v2(best fp32) of the default model
  cfg_reset
  cfg_set multimodel attach-perf-metric      "${PERF_METRIC}"          # per-model/role fps+latency (model/status)
  cfg_set multimodel perf-metric-interval-sec "${PERF_METRIC_INTERVAL}" # measurement window (s)
  cfg_set source-list   max-batch-size        "${n}"
  cfg_set tiled-display rows                  2          # A/B view: row0 primary, row1 shadow
  cfg_set tiled-display columns               "${n}"
  cfg_set multimodel    unified-batch         1
  cfg_set multimodel    per-model-batch-size  "${eb}"       # b${eb} per version -> no shard
  cfg_set multimodel    primary-model         "${DEFAULT_MODEL};${v1}"
  cfg_set multimodel    shadow-model          "${DEFAULT_MODEL};${v2}"
  # per-version engines -- stress_setup.sh prebuilds these b${eb} files so the plugin's
  # per-version engine= override resolves (it only uses an engine that already EXISTS).
  cfg_set model-${DEFAULT_MODEL}-${v1}        engine "${worst_eng}"    # v1 = worst (int8 nocalib)
  cfg_set model-${DEFAULT_MODEL}-${v2}        engine "${best_eng}"     # v2 = best (fp32)
  [ "${FILE_LOOP}" = "1" ] && cfg_main tests file-loop 1

  # 2) start app
  start_app || return 1

  # 3) bring all n A/B streams up (each runs primary v1 + shadow v2)
  echo "[${name}] adding ${n} A/B streams ..."
  local i
  for (( i=0; i<n; i++ )); do
    add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"
    sleep 1
  done
  echo "[${name}] all ${n} A/B streams added; steady state reached"

  # 4) begin sampling (baseline = all streams up + both versions warmed)
  _wait_serving "${n}"     # take the baseline only once all streams are serving (model loaded)
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"

  # 5) async random churn for the whole hold
  _stress2_churn "${n}" "${cb}" "${cdown}" "${cgap}" &
  CHURN_PID=$!
  echo "[${name}] async churn started (pid ${CHURN_PID})"

  # 6) hold under load for STRESS_DURATION (abort if app dies)
  local waited=0
  while [ "${waited}" -lt "${STRESS_DURATION}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited early at ${waited}s (check ${APP_LOG})"; break
    fi
    sleep 5; waited=$((waited+5))
  done

  # 7) stop churn + sampling + app, then functional gate + leak report
  [ -n "${CHURN_PID}" ] && kill "${CHURN_PID}" 2>/dev/null; wait "${CHURN_PID}" 2>/dev/null; CHURN_PID=""
  mem_monitor_stop
  stop_app
  _stress_sanity "${APP_LOG}" "${name}"
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

# _stress4_swap <n> <batch> <gap_s> <mA> <vA> <mB> <vB>
# Background worker: half the streams start on model A, half on B. Every <gap_s>
# it live-swaps streams' PRIMARY to the OTHER model (model/routing), tracking each
# stream's current model so a swap toggles A<->B. Streams stay up, only their model
# changes. Each cycle picks the SCOPE with a 50/50 coin, to mix both stress shapes:
#   - FULL   : ALL streams toggle at once -> 2 bulk model_routings (A-set->B, B-set->A)
#   - SUBSET : <batch> RANDOM streams toggle individually (scoped reroute)
_stress4_swap() {
  local n="$1" batch="$2" gap="$3" mA="$4" vA="$5" mB="$6" vB="$7"
  local half=$(( n/2 )) id cyc=0 picks tgtn tgtv al bl
  declare -A curn curv
  for (( id=0; id<n; id++ )); do
    if [ "${id}" -lt "${half}" ]; then curn[$id]="${mA}"; curv[$id]="${vA}"; else curn[$id]="${mB}"; curv[$id]="${vB}"; fi
  done
  while kill -0 "${APP_PID}" 2>/dev/null; do
    cyc=$((cyc+1))
    if (( RANDOM % 2 == 0 )); then
      # FULL swap: build the current A-set and B-set, flip each with one bulk call
      al=""; bl=""
      for (( id=0; id<n; id++ )); do
        if [ "${curn[$id]}" = "${mA}" ]; then al="${al:+${al},}${PREFIX}${id}"; else bl="${bl:+${bl},}${PREFIX}${id}"; fi
      done
      kill -0 "${APP_PID}" 2>/dev/null || return
      [ -n "${al}" ] && model_routing "${al}" "${mB}" "${vB}"   # A-set -> B
      [ -n "${bl}" ] && model_routing "${bl}" "${mA}" "${vA}"   # B-set -> A
      for (( id=0; id<n; id++ )); do
        if [ "${curn[$id]}" = "${mA}" ]; then curn[$id]="${mB}"; curv[$id]="${vB}"; else curn[$id]="${mA}"; curv[$id]="${vA}"; fi
      done
      echo "[swap] cycle ${cyc}: FULL swap (all ${n} streams toggled)"
    else
      # SUBSET swap: batch random streams, per-stream toggle
      picks=$(seq 0 $((n-1)) | shuf 2>/dev/null | head -n "${batch}")
      [ -z "${picks}" ] && picks=$(seq 0 $((batch-1)))
      for id in ${picks}; do
        kill -0 "${APP_PID}" 2>/dev/null || return
        if [ "${curn[$id]}" = "${mA}" ]; then tgtn="${mB}"; tgtv="${vB}"; else tgtn="${mA}"; tgtv="${vA}"; fi
        model_routing "${PREFIX}${id}" "${tgtn}" "${tgtv}"
        curn[$id]="${tgtn}"; curv[$id]="${tgtv}"
        sleep 0.3
      done
      echo "[swap] cycle ${cyc}: SUBSET swap ($(echo ${picks} | tr '\n' ' '))"
    fi
    sleep "${gap}"
  done
}

###############################################################################
# stress_6 -- two-primary STEADY soak + leak/health watch (NO swap).
#   Same setup as stress_6: half of SWAP_STREAMS on the DEFAULT model, half on the CAR
#   model (both PRIMARY, no shadow, both loaded via model/load). Brings all streams up
#   ONCE and just HOLDS -- no model swap, no removal. STABLE-PIPELINE baseline for the
#   two-model case: drift here = steady-state leak, not the live-reroute path. Streams are
#   never removed/swapped -> contiguity holds -> FULL gate set.
#
#   Knobs: SWAP_STREAMS(30 -> 15+15), PER_MODEL_BATCH, STRESS_DURATION.
###############################################################################
stress_6() {
  local name="stress_6"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"   # run_one_stress sets+truncates + tees the whole case
  local n="${SWAP_STREAMS}"
  local mA="${DEFAULT_MODEL}" vA="${DEFAULT_MODEL_VERSION}" mB="${CAR_MODEL}" vB="${CAR_MODEL_VERSION}"
  local half=$(( n/2 )) cols rows
  local eb="${PER_MODEL_BATCH:-$(( n/2 ))}"       # b<half> per model (balanced); imbalance may shard
  local eng="${WORST_ENGINE/_b[0-9]*_gpu0_int8_nocalib_worst.engine/_b${eb}_gpu0_fp16.engine}"  # b${eb} fp16 (stress_setup.sh)
  cols=$(awk -v x="${n}" 'BEGIN{c=int(sqrt(x)+0.999); if(c<1)c=1; print c}')
  rows=$(awk -v x="${n}" -v c="${cols}" 'BEGIN{print int((x+c-1)/c)}')
  echo "-----------------------------------------------------------------------"
  echo "[${name}] two-primary STEADY soak (NO swap): ${half}x ${mA}@${vA} + ${half}x ${mB}@${vB} (=${n} streams), hold ${STRESS_DURATION}s"

  # 1) config: two primaries, no shadow; model/load API is the SOLE source of the defaults
  cfg_reset
  cfg_set multimodel attach-perf-metric      "${PERF_METRIC}"          # per-model/role fps+latency (model/status)
  cfg_set multimodel perf-metric-interval-sec "${PERF_METRIC_INTERVAL}" # measurement window (s)
  cfg_set source-list   max-batch-size        "${n}"
  cfg_set tiled-display rows                  "${rows}"
  cfg_set tiled-display columns               "${cols}"
  cfg_set multimodel    unified-batch         0
  cfg_set multimodel    per-model-batch-size  "${eb}"       # b${eb} per model, no shard when balanced
  comment_cfg           multimodel            primary-model  # API-driven default primary (loaded below)
  comment_cfg           multimodel            shadow-model   # primary-only test -> no shadow
  [ "${FILE_LOOP}" = "1" ] && cfg_main tests file-loop 1

  # 2) start app (no config default primary yet -> streams pass through until models load)
  start_app || return 1

  # 3) load BOTH primaries via model/load (default ${mA} + custom ${mB}, primary-only)
  echo "[${name}] model/load default primary ${mA}@${vA} + custom primary ${mB}@${vB} ..."
  load_model "${mA}" "${vA}" "${eng}" "" "True"     # default primary (set_default_primary), b${eb} fp16
  wait_for_model "${mA}" "${vA}"
  load_model "${mB}" "${vB}" "${eng}"               # custom primary -> limbo, b${eb} fp16
  wait_for_model "${mB}" "${vB}"

  # 4) bring streams up ONCE: first half default primary (${mA}), second half ${mB}
  echo "[${name}] adding ${half} default(${mA}) + ${half} ${mB} streams ..."
  local i
  for (( i=0; i<half; i++ )); do add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"; sleep 1; done                  # no model -> default primary (${mA})
  for (( i=half; i<n; i++ )); do add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}" "${mB}" "${vB}"; sleep 1; done   # explicit ${mB}
  echo "[${name}] all ${n} streams added; steady state reached"

  # 5) begin sampling + HOLD steady (NO swap) for STRESS_DURATION
  _wait_serving "${n}"
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"
  local waited=0
  while [ "${waited}" -lt "${STRESS_DURATION}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited early at ${waited}s (check ${APP_LOG})"; break
    fi
    sleep 5; waited=$((waited+5))
  done

  # 6) stop sampling + app, then FULL functional gate + leak report
  mem_monitor_stop
  stop_app
  sleep 1
  run_default_sanity_checks "${APP_LOG}" "${name}"
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

###############################################################################
# stress_7 -- two-primary live model-swap soak + leak/health watch.
#   Half of SWAP_STREAMS (default 30 -> 15) run on the DEFAULT model, half on the
#   CAR model (both PRIMARY, no shadow). Every CHURN_GAP a background worker
#   live-swaps CHURN_BATCH random streams to the OTHER model (model/update primary
#   reroute, zero-drop). Streams are NEVER removed -- only their model changes --
#   so per-source frame contiguity holds and we run the FULL gate set (a dropped
#   frame during a swap is a real bug we want to catch). Exercises the live
#   reroute + per-model shard grow/shrink path continuously.
#
#   Knobs: SWAP_STREAMS(30), CHURN_BATCH(5 swaps/cycle), CHURN_GAP(10s),
#          PER_MODEL_BATCH(4), STRESS_DURATION.
###############################################################################
stress_7() {
  local name="stress_7"
  local dump="${DUMP_DIR}/${name}_$(date +%Y%m%d_%H%M%S).csv"
  APP_LOG="${APP_LOG:-${LOG_DIR}/${name}_logs.txt}"   # run_one_stress sets+truncates + tees the whole case
  local n="${SWAP_STREAMS}" cb="${CHURN_BATCH}" cgap="${CHURN_GAP}"
  local mA="${DEFAULT_MODEL}" vA="${DEFAULT_MODEL_VERSION}" mB="${CAR_MODEL}" vB="${CAR_MODEL_VERSION}"
  local half=$(( n/2 )) cols rows
  local eb="${PER_MODEL_BATCH:-$(( n/2 ))}"       # b<half> per model (balanced); imbalance may shard
  local eng="${WORST_ENGINE/_b[0-9]*_gpu0_int8_nocalib_worst.engine/_b${eb}_gpu0_fp16.engine}"  # b${eb} fp16 (stress_setup.sh)
  cols=$(awk -v x="${n}" 'BEGIN{c=int(sqrt(x)+0.999); if(c<1)c=1; print c}')
  rows=$(awk -v x="${n}" -v c="${cols}" 'BEGIN{print int((x+c-1)/c)}')
  echo "-----------------------------------------------------------------------"
  echo "[${name}] two-primary live model-swap soak: ${half}x ${mA}@${vA} + ${half}x ${mB}@${vB} (=${n} streams), swap every ${cgap}s (50% FULL / 50% SUBSET of ${cb}), hold ${STRESS_DURATION}s"

  # 1) config: two primaries, no shadow. Comment out the config default primary/shadow so the
  #    model/load API is the SOLE source of the defaults (BOTH primaries loaded at runtime).
  cfg_reset
  cfg_set multimodel attach-perf-metric      "${PERF_METRIC}"          # per-model/role fps+latency (model/status)
  cfg_set multimodel perf-metric-interval-sec "${PERF_METRIC_INTERVAL}" # measurement window (s)
  cfg_set source-list   max-batch-size        "${n}"
  cfg_set tiled-display rows                  "${rows}"
  cfg_set tiled-display columns               "${cols}"
  cfg_set multimodel    unified-batch         0
  cfg_set multimodel    per-model-batch-size  "${eb}"       # b${eb} per model, no shard when balanced
  comment_cfg           multimodel            primary-model  # API-driven default primary (loaded below)
  comment_cfg           multimodel            shadow-model   # primary-only test -> no shadow
  [ "${FILE_LOOP}" = "1" ] && cfg_main tests file-loop 1

  # 2) start app (no config default primary yet -> streams pass through until models load)
  start_app || return 1

  # 3) load BOTH primaries via model/load (primary-only, no shadow):
  #      - ${mA} as the runtime DEFAULT primary (set_default_primary=true) -> no-model streams use it
  #      - ${mB} into limbo -> promoted to primary on streams that explicitly request it
  #    A stream requesting a NOT-loaded primary falls back to the default, so both must be loaded.
  echo "[${name}] model/load default primary ${mA}@${vA} + custom primary ${mB}@${vB} ..."
  load_model "${mA}" "${vA}" "${eng}" "" "True"     # default primary (set_default_primary), b${eb} fp16
  wait_for_model "${mA}" "${vA}"
  load_model "${mB}" "${vB}" "${eng}"               # custom primary -> limbo, b${eb} fp16
  wait_for_model "${mB}" "${vB}"

  # 4) bring streams up: first half default primary (${mA}), second half ${mB}
  echo "[${name}] adding ${half} default(${mA}) + ${half} ${mB} streams ..."
  local i
  for (( i=0; i<half; i++ )); do add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}"; sleep 1; done                  # no model -> default primary (${mA})
  for (( i=half; i<n; i++ )); do add_stream "${PREFIX}${i}" "Cam-${i}" "${STRESS_STREAM}" "${mB}" "${vB}"; sleep 1; done   # explicit ${mB}
  echo "[${name}] all ${n} streams added; steady state reached"

  # 4) begin sampling (baseline = both models warmed, streams split)
  _wait_serving "${n}"     # take the baseline only once all streams are serving (model loaded)
  mem_monitor_start "${APP_PID}" "${dump}" "${SAMPLE_INTERVAL}" "${APP_LOG}"

  # 5) async live model-swap for the whole hold
  _stress4_swap "${n}" "${cb}" "${cgap}" "${mA}" "${vA}" "${mB}" "${vB}" &
  CHURN_PID=$!
  echo "[${name}] async model-swap started (pid ${CHURN_PID})"

  # 6) hold under load for STRESS_DURATION (abort if app dies)
  local waited=0
  while [ "${waited}" -lt "${STRESS_DURATION}" ]; do
    if [ -n "${APP_PID}" ] && ! kill -0 "${APP_PID}" 2>/dev/null; then
      echo "[${name}] WARN: app exited early at ${waited}s (check ${APP_LOG})"; break
    fi
    sleep 5; waited=$((waited+5))
  done

  # 7) stop swap + sampling + app, then FULL functional gate + leak report
  #    (streams never removed -> contiguity holds -> full gates are valid)
  [ -n "${CHURN_PID}" ] && kill "${CHURN_PID}" 2>/dev/null; wait "${CHURN_PID}" 2>/dev/null; CHURN_PID=""
  mem_monitor_stop
  stop_app
  sleep 1
  run_default_sanity_checks "${APP_LOG}" "${name}"
  echo "[${name}] resource dump: ${dump}  (+ ${dump%.csv}.log)"
  mem_leak_report "${dump}"
  _stress_record "${name}" "${dump}" "${APP_LOG}"
  echo "-----------------------------------------------------------------------"
}

###############################################################################
# dispatcher
###############################################################################
ALL_STRESS="0 1 2 3 4 5 6 7"

run_stress_case() {
  case "$1" in
    0)    stress_0 ;;
    1)    stress_1 ;;
    2)    stress_2 ;;
    3)    stress_3 ;;
    4)    stress_4 ;;
    5)    stress_5 ;;
    6)    stress_6 ;;
    7)    stress_7 ;;
    *)    echo "usage: $0 [0|1|2|3|4|5|6|7|all|list]" >&2; return 1 ;;
  esac
}

# run_one_stress <sel> : capture the ENTIRE case (app stdout/stderr + every REST call
# + narration + mem samples + gates) into stress_<sel>_logs.txt via tee (also live) --
# exactly like run_tests.sh's run_one. Snapshots the FAIL tally for the results row.
run_one_stress() {
  local sel="$1"
  APP_LOG="${LOG_DIR}/stress_${sel}_logs.txt"
  : > "${APP_LOG}"
  _STRESS_FAIL0="${#FAIL_LIST[@]}"
  run_stress_case "${sel}" > >(tee -a "${APP_LOG}") 2>&1
}

main() {
  local sel="${1:-all}"
  if [ "${sel}" = "list" ]; then
    cat <<'LISTEOF'
available stress workloads (run in this order by 'all'):

  --- 0 and 1: CONTROLS, NO nvmodelmux (plain nvinfer, plugin-type=0) ---
      Each pairs with the modelmux test directly below it and differs from it in
      exactly ONE way: modelmux in vs out. Same cycles, holds, batch and source.
      So if a control ALSO fails/leaks, it is not modelmux -- it is the core
      decode / streammux / source-churn path. PGIE_ENABLE=1 (default) keeps the
      inference load matched; set 0 to strip inference too.

  0   drain-to-zero add/remove churn, NO modelmux      <- control for 2
  1   30-stream soak + async random churn, NO modelmux <- control for 3

  --- 2-7: the model plane (nvmodelmux, plugin-type=4) ---

  2   drain-to-zero add/remove churn (pipeline emptied to ZERO streams and
      refilled, 50 cycles -- exercises the inner mux/demux recreate path that
      the soak tests never reach)
  3   30-stream soak + async random churn
  4   A/B two-version STEADY soak, no churn
  5   A/B two-version soak + churn
  6   two-primary STEADY soak, no swap
  7   two-primary live model-swap soak

usage: run_stress.sh [0|1|2|3|4|5|6|7|all|list]
LISTEOF
    return 0
  fi
  clear 2>/dev/null || true                 # fresh console at run start (single test or 'all')
  if [ "${AUTO_APP:-1}" = "1" ]; then
    run_setup_check || { echo "[stress] setup precheck failed (plugin/engine)"; exit 1; }
    bash "${HERE}/stress_setup.sh" || { echo "[stress] stress_setup.sh (engine prebuild) failed"; exit 1; }
  fi
  # fresh at-a-glance results ledger for this invocation
  { echo "# nvmodelmux stress results   (leak: = RSS-median screen · tracer: = GObject leak count, duration-independent)"; printf "%-9s | %-15s | %-9s | %-16s | %s
" "test" "gates" "leak" "tracer" "memory (post-warmup)"; } > "${STRESS_RESULTS}"
  if [ "${sel}" = "all" ]; then
    local t; for t in ${ALL_STRESS}; do run_one_stress "${t}"; echo; done
  else
    run_one_stress "${sel}"
  fi
  echo; echo "======================= STRESS RESULTS ======================="
  cat "${STRESS_RESULTS}"
  echo "=============================================================="
  print_summary
}

main "${1:-all}"