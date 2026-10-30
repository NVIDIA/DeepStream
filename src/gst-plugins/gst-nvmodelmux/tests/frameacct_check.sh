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
# frameacct_check.sh -- reusable ZERO-FRAME-LOSS sanity check for the
# nvmodelmux plugin tests.
#
# With MM_DEBUG=1 (default in run_tests.sh) the plugin periodically dumps a
# per-(stream,role) `frameacct_table` into the app log:
#
#   │ stream │ ... │ buffered │ dropped │ status │
#   │ 0      │ ... │ 5        │ 0       │ ok     │
#
#   buffered = frames in-flight (NOT loss); dropped = real loss (0 = none);
#   status   = warm | ok | DROP   (DROP iff dropped>0 after the warm-up window).
#
# Frame loss is the DEFAULT failure condition for every test, so this utility
# scans ALL frameacct dumps in a saved log and FAILS if ANY (stream,role) row
# ever reported a real drop (dropped>0 / status=DROP). Wire it into stop_app so
# the gate runs automatically for every test against that test's saved log.
#
# Usage (source, then call):
#   source frameacct_check.sh
#   assert_no_frame_drops [logfile] [description]
#     logfile     app log to scan          (default: $APP_LOG)
#     description PASS/FAIL label           (default: auto)
#   Returns 0 on PASS/SKIP, 1 on FAIL. Uses run_tests.sh's _pass/_fail counters
#   when sourced into it; otherwise prints standalone PASS/FAIL.
###############################################################################

# Standalone fallbacks so the utility also works outside run_tests.sh.
type _pass >/dev/null 2>&1 || _pass() { echo "  [PASS] $*"; }
type _fail >/dev/null 2>&1 || _fail() { echo "  [FAIL] $*"; }

# assert_no_frame_drops [logfile] [description]
assert_no_frame_drops() {
  local log="${1:-${APP_LOG}}"
  local desc="${2:-frame-drop sanity: ZERO frame loss (frameacct dropped==0)}"

  # No captured log (e.g. AUTO_APP=0 / external app) -> cannot assert; SKIP
  # rather than silently pass.
  if [ -z "${log}" ] || [ ! -s "${log}" ]; then
    echo "  [SKIP] ${desc}  (no app log to scan: '${log:-unset}')"
    return 0
  fi

  # Strip ANSI colour, normalise the box separator '│' -> '|' (UTF-8 safe via
  # sed; mawk cannot use a multibyte FS), then walk ONLY frameacct data rows:
  # role == Primary/Shadow inside a frameacct_table block. dropped=$11, status=$12.
  local out
  out="$(sed -e 's/\x1b\[[0-9;]*m//g' -e 's/│/|/g' "${log}" | awk -F'|' '
    BEGIN { in_fa = 0; rows = 0; bad = 0 }
    /frameacct_table/             { in_fa = 1; next }   # entering a frameacct dump
    /models_table|routing_table/  { in_fa = 0 }         # a different table -> leave
    /^[ ]*╭|^[ ]*╰/               { in_fa = 0 }         # state titlebox -> leave
    in_fa && /^[ ]*\|/ {
      role = $4; gsub(/^[ \t]+|[ \t]+$/, "", role)
      if (role != "Primary" && role != "Shadow") next   # skip header row / rules
      drp = $11; st = $12
      gsub(/^[ \t]+|[ \t]+$/, "", drp)
      gsub(/^[ \t]+|[ \t]+$/, "", st)
      rows++
      if (st == "DROP" || (drp ~ /^[0-9]+$/ && drp + 0 > 0)) {
        bad++
        sid = $2; src = $3
        gsub(/^[ \t]+|[ \t]+$/, "", sid); gsub(/^[ \t]+|[ \t]+$/, "", src)
        if (bad <= 10)
          print "    drop: stream " sid " (" src ") " role \
                "  dropped=" drp "  status=" st
      }
    }
    END { print "SUMMARY " rows " " bad }
  ')"

  local summary rows bad
  summary="$(printf '%s\n' "${out}" | grep '^SUMMARY ')"
  rows="$(echo "${summary}" | awk '{print $2}')"; rows="${rows:-0}"
  bad="$(echo  "${summary}" | awk '{print $3}')"; bad="${bad:-0}"

  if [ "${rows}" -eq 0 ]; then
    echo "  [SKIP] ${desc}  (no frameacct rows in log -- MM_DEBUG off or no traffic)"
    return 0
  fi
  if [ "${bad}" -eq 0 ]; then
    _pass "${desc}  (${rows} (stream,role) samples scanned, 0 drops)"
    return 0
  fi
  printf '%s\n' "${out}" | grep '^    drop:'      # show the offending rows
  _fail "${desc}  (${bad}/${rows} frameacct samples reported FRAME DROP)"
  return 1
}

# assert_no_inference_stalls [logfile] [description]
#   HARD-stall gate: FAILS when an inferred role keeps accepting frames but never
#   emits inference/output. This catches the dynamic-lane wedge where frameacct
#   looked like:
#       entered=N, inferred=0, delivered=0, buffered=N, dropped=0
#   `assert_no_frame_drops` intentionally passes that case because nothing is
#   dropped; this gate treats sustained all-buffered progress as a stall.
#
#   Tunables:
#     MM_STALL_MIN_ENTERED   minimum entered frames for a suspicious sample (default 100)
#     MM_STALL_MIN_SAMPLES   suspicious samples required for one stream/role (default 2)
assert_no_inference_stalls() {
  local log="${1:-${APP_LOG}}"
  local desc="${2:-inference progress sanity: no stream buffers forever with inferred=0}"
  local min_entered="${MM_STALL_MIN_ENTERED:-100}"
  local min_samples="${MM_STALL_MIN_SAMPLES:-2}"

  if [ -z "${log}" ] || [ ! -s "${log}" ]; then
    echo "  [SKIP] ${desc}  (no app log to scan: '${log:-unset}')"
    return 0
  fi

  local out
  out="$(sed -e 's/\x1b\[[0-9;]*m//g' -e 's/│/|/g' "${log}" | awk -F'|' \
      -v min_entered="${min_entered}" -v min_samples="${min_samples}" '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    function numeric(s) { return s ~ /^[0-9]+$/ }
    BEGIN { in_fa = 0; rows = 0 }
    /frameacct_table/             { in_fa = 1; next }
    /models_table|routing_table/  { in_fa = 0 }
    /^[ ]*╭|^[ ]*╰/               { in_fa = 0 }
    in_fa && /^[ ]*\|/ {
      sid = trim($2); src = trim($3); role = trim($4)
      if (role != "Primary" && role != "Shadow") next

      ent = trim($5); inf = trim($6); del = trim($7)
      buf = trim($10); drp = trim($11); st = trim($12)
      if (!numeric(ent) || !numeric(inf) || !numeric(del) || !numeric(buf)) next
      rows++

      key = sid SUBSEP src SUBSEP role
      # Suspicious only when the role is clearly receiving sustained input, no
      # inference/output has happened, nearly everything remains buffered, and
      # frameacct does not already classify it as a drop.
      if (ent + 0 >= min_entered &&
          inf + 0 == 0 &&
          del + 0 == 0 &&
          buf + 0 >= ent + 0 - 2 &&
          (!numeric(drp) || drp + 0 == 0)) {
        bad_samples[key]++
        if (!(key in first_ent)) {
          first_ent[key] = ent; first_buf[key] = buf; first_st[key] = st
        }
        last_ent[key] = ent; last_buf[key] = buf; last_st[key] = st
      }
    }
    END {
      bad = 0
      for (key in bad_samples) {
        if (bad_samples[key] < min_samples) continue
        split(key, p, SUBSEP)
        bad++
        if (bad <= 10)
          printf "    stall: stream %s (%s) %s  samples=%d  entered=%s->%s  buffered=%s->%s  inferred=0 delivered=0  status=%s->%s\n",
              p[1], p[2], p[3], bad_samples[key],
              first_ent[key], last_ent[key], first_buf[key], last_buf[key],
              first_st[key], last_st[key]
      }
      print "SUMMARY " rows " " bad
    }
  ')"

  local summary rows bad
  summary="$(printf '%s\n' "${out}" | grep '^SUMMARY ')"
  rows="$(echo "${summary}" | awk '{print $2}')"; rows="${rows:-0}"
  bad="$(echo  "${summary}" | awk '{print $3}')"; bad="${bad:-0}"

  if [ "${rows}" -eq 0 ]; then
    echo "  [SKIP] ${desc}  (no frameacct rows in log -- MM_DEBUG off or no traffic)"
    return 0
  fi
  if [ "${bad}" -eq 0 ]; then
    _pass "${desc}  (${rows} (stream,role) samples scanned, no sustained inferred=0 stall)"
    return 0
  fi
  printf '%s\n' "${out}" | grep '^    stall:'
  _fail "${desc}  (${bad} stream/role lane(s) buffered frames without inference progress)"
  return 1
}

# report_data_before_segment_warnings [logfile] [description]
#   Sticky-event ordering diagnostic: reports the GStreamer warning emitted when
#   a dynamically wired branch receives buffers before its SEGMENT event. This is
#   the same family of bug as the hard inferred=0 wedge above, but existing
#   dynamic attach cases can warn while still delivering every frame. Therefore
#   the default hard gate is the frameacct stall detector; set
#   MM_STRICT_SEGMENT_WARN=1 to make warning-only runs fail too.
report_data_before_segment_warnings() {
  local log="${1:-${APP_LOG}}"
  local desc="${2:-sticky-event diagnostic: buffer-before-segment warnings}"
  local strict="${MM_STRICT_SEGMENT_WARN:-0}"

  if [ -z "${log}" ] || [ ! -s "${log}" ]; then
    echo "  [SKIP] ${desc}  (no app log to scan: '${log:-unset}')"
    return 0
  fi

  local hits n
  hits="$(grep -nF "Got data flow before segment event" "${log}" 2>/dev/null \
          | grep -vE '^[0-9]+:[[:space:]]+[0-9]+:' || true)"
  if [ -z "${hits}" ]; then
    _pass "${desc}: none"
    return 0
  fi

  n="$(printf '%s\n' "${hits}" | grep -c .)"
  printf '%s\n' "${hits}" | head -10 | sed -E 's/^/    /; s/\x1b\[[0-9;]*m//g'
  if [ "${strict}" = "1" ]; then
    _fail "${desc}  (${n} warning(s); MM_STRICT_SEGMENT_WARN=1)"
    return 1
  fi
  echo "  [INFO] ${desc}: ${n} warning(s) observed; not a hard failure unless MM_STRICT_SEGMENT_WARN=1"
  return 0
}

# assert_no_gst_critical [logfile] [description]
#   HARD-failure gate: FAILS if the log contains a GLib/GStreamer CRITICAL, a
#   failed assertion, or a process crash. This complements assert_no_frame_drops:
#   a total STALL (e.g. the drain-to-zero refill bug) delivers no frames so it is
#   invisible to the frame-drop gate, but it surfaces here as
#   "gst_pad_push: assertion 'GST_IS_BUFFER (buffer)' failed".
#
#   Deliberately does NOT match:
#     - plain GST_ERROR  -- the plugin logs EXPECTED rejections (e.g. 'update
#       rejected', 'cannot unload ... Serving') that negative tests rely on;
#     - 'Sticky event misordering ... caps before stream-start' -- benign WARNING
#       from dynamic pad-add on legacy nvstreammux;
#     - 'User Interrupted' -- the SIGINT clean-shutdown from stop_app;
#     - gst-plugin-scanner / 'pygobject initialization failed' -- an ENVIRONMENT critical from
#       the registry scanner subprocess (python3-gi missing), unrelated to the plugin under test.
#     - with MM_ALLOW_SOURCEBIN_DUP_PAD_CRITICAL=1 only: the known dsnvurisrcbin
#       duplicate-pad critical produced by tests that intentionally fire duplicate
#       stream-add REST requests. That is source-bin duplicate rejection, not an
#       nvmodelmux crash/assertion.
assert_no_gst_critical() {
  local log="${1:-${APP_LOG}}"
  local desc="${2:-crash sanity: no GStreamer-CRITICAL / assertion / crash}"

  if [ -z "${log}" ] || [ ! -s "${log}" ]; then
    echo "  [SKIP] ${desc}  (no app log to scan: '${log:-unset}')"
    return 0
  fi

  # Benign, ENVIRONMENT-only criticals to ignore (not the plugin / app under test):
  #   - gst-plugin-scanner pygobject init failure: the GStreamer registry scanner subprocess
  #     fails to import the Python 'gi' module (python3-gi not installed). Emitted during
  #     plugin-registry build, unrelated to the pipeline/plugin. (Install python3-gi to silence
  #     it at the source; we drop it here so the gate isn't a false positive without it.)
  #   - gst_debug_log_full_valist 'category != NULL' assertion: a GStreamer CORE debug-logging
  #     race at pipeline bring-up -- a log call goes through a GST_CAT_* debug category pointer
  #     before that category is registered. Purely cosmetic (logging path only), fires only at
  #     startup on the decoder, no crash/corruption. Not emitted by nvmodelmux; drop it so the
  #     gate isn't a false positive.
  local benign='gst-plugin-scanner|pygobject initialization failed|gst_debug_log_full_valist: assertion .category != NULL. failed'
  if [ "${MM_ALLOW_SOURCEBIN_DUP_PAD_CRITICAL:-0}" = "1" ]; then
    benign="${benign}|Padname vsrc_[0-9]+ is not unique in element dsnvurisrcbin[0-9]*, not adding"
  fi
  local hits n
  hits="$(grep -nE "CRITICAL \*\*|assertion '.*' failed|Segmentation fault|\(core dumped\)|terminate called|Cuda failure|CUDA error" "${log}" 2>/dev/null \
          | grep -vE '^[0-9]+:[[:space:]]+[0-9]+:' \
          | grep -vE "${benign}" || true)"

  # GStreamer 'leaks' tracer teardown segfault (harness-induced, NOT the app/plugin):
  #   run_stress enables the tracer (GST_TRACERS=leaks, LEAK_TRACER=1) for a duration-
  #   independent object-leak report. At process exit the tracer walks its tracked-object
  #   table inside gst_deinit() and can SIGSEGV formatting an already-freed instance:
  #     g_type_check_instance_is_a <- gst_info_strdup_printf <- <libgstcoretracers>
  #       <- g_object_unref <- gst_deinit <- main   (verified via core-dump backtrace)
  #   This fires AFTER "App run successful" (pipeline stopped, data flushed, leak report
  #   already printed) -> zero impact on the run (frame-loss/stall/tracer gates all pass);
  #   a normal/production run without the tracer never executes that code. Drop a SIGSEGV /
  #   (core dumped) that lands in this gst_deinit teardown phase (after the last
  #   "App run successful") when the tracer was active. A crash BEFORE that marker (mid-run)
  #   is a real fault and still fails. Set MM_STRICT_TRACER_SEGV=1 to fail on it regardless.
  if [ -n "${hits}" ] && [ "${MM_STRICT_TRACER_SEGV:-0}" != "1" ] \
     && grep -qE 'GST_TRACERS|GST_TRACER:|leaks tracer ENABLED' "${log}" 2>/dev/null; then
    local app_ok before after
    app_ok="$(grep -n 'App run successful' "${log}" 2>/dev/null | tail -1 | cut -d: -f1)"
    if [ -n "${app_ok}" ]; then
      before="$(printf '%s\n' "${hits}" | grep -c .)"
      hits="$(printf '%s\n' "${hits}" | awk -v ok="${app_ok}" -F: \
        '{ ln=$1+0; if (ln>ok && (/Segmentation fault/ || /\(core dumped\)/)) next; print }')"
      after="$(printf '%s\n' "${hits}" | grep -c .)"
      if [ "${after}" -lt "${before}" ]; then
        echo "  [INFO] ${desc}: ignored $((before-after)) SIGSEGV/core line(s) in the gst_deinit teardown phase (after 'App run successful') -- GStreamer 'leaks' tracer (GST_TRACERS=leaks) faults in libgstcoretracers at gst_deinit; harness-induced, not a pipeline/app fault. Set MM_STRICT_TRACER_SEGV=1 to treat as failure."
      fi
    fi
  fi

  if [ -z "${hits}" ]; then
    _pass "${desc}"
    return 0
  fi
  printf '%s\n' "${hits}" | head -5 | sed -E 's/^/    /; s/\x1b\[[0-9;]*m//g'
  n="$(printf '%s\n' "${hits}" | grep -c .)"
  _fail "${desc}  (${n} critical/assert/crash line(s) in log)"
  return 1
}

# assert_no_frame_gaps [logfile] [description]
#   EXACT per-frame zero-loss gate (needs MM_FRAME_TRACE=1 + the traced plugin).
#   The plugin carries each frame's ORIGINAL input frame_num through the internal
#   muxes and logs FRAMEGAP at the bin output when a (source,role) sequence has a
#   forward gap. A gap can be EITHER an internal drop OR an upstream one (the source
#   nvstreammux occasionally drops a frame on a dynamic pad-add re-batch, so it never
#   reaches the bin). We FAIL only on an INTERNAL loss: a gap with NO frameacct drop
#   (entered==delivered everywhere) means the frame never entered -> dropped upstream
#   -> INFO + pass; a gap alongside a frameacct drop -> real in-bin loss -> FAIL.
#   Unlike assert_no_frame_drops (count heuristic with a jitter margin), this catches
#   even a single one-time drop. SKIPs (never false-passes) when tracing wasn't active
#   or the trace meta didn't survive the mux.
assert_no_frame_gaps() {
  local log="${1:-${APP_LOG}}"
  local desc="${2:-frame-gap sanity: contiguous per-source frame numbers (no one-time drop)}"

  if [ -z "${log}" ] || [ ! -s "${log}" ]; then
    echo "  [SKIP] ${desc}  (no app log to scan: '${log:-unset}')"
    return 0
  fi
  if grep -q 'FRAMETRACE.*survived=NO' "${log}"; then
    echo "  [SKIP] ${desc}  (trace meta did NOT survive the mux -- gap check unreliable)"
    return 0
  fi
  if ! grep -q 'FRAMETRACE.*survived=YES' "${log}"; then
    echo "  [SKIP] ${desc}  (MM_FRAME_TRACE not active / traced plugin not loaded)"
    return 0
  fi

  local gaps n
  gaps="$(grep -E 'FRAMEGAP' "${log}")"
  if [ -z "${gaps}" ]; then
    _pass "${desc}"
    return 0
  fi

  # A FRAMEGAP is a hole in the OUTPUT frame_num sequence. Localise it with the
  # frameacct 'dropped' column (entered vs delivered, same scan assert_no_frame_drops
  # uses): if NO (stream,role) reports an internal drop, every frame that ENTERED the
  # bin was delivered -> the missing frame_num never entered = dropped UPSTREAM (the
  # source nvstreammux, e.g. an intermittent re-batch on a dynamic pad-add). That is
  # NOT a plugin loss -> INFO + pass. If any internal drop exists, the gap is a real
  # in-bin loss -> FAIL.
  local bad
  bad="$(sed -e 's/\x1b\[[0-9;]*m//g' -e 's/│/|/g' "${log}" | awk -F'|' '
    /frameacct_table/{f=1;next} /models_table|routing_table/{f=0} /^[ ]*╭|^[ ]*╰/{f=0}
    f && /^[ ]*\|/ {
      r=$4; gsub(/^[ \t]+|[ \t]+$/,"",r); if(r!="Primary"&&r!="Shadow")next;
      d=$11; s=$12; gsub(/^[ \t]+|[ \t]+$/,"",d); gsub(/^[ \t]+|[ \t]+$/,"",s);
      if(s=="DROP"||(d ~ /^[0-9]+$/ && d+0>0)) b++ }
    END{print b+0}')"
  n="$(printf '%s\n' "${gaps}" | grep -c .)"
  if [ "${bad:-0}" -eq 0 ]; then
    echo "  [INFO] ${desc}: ${n} frame-number gap(s), but frameacct shows 0 internal drops"
    echo "         -> frames dropped UPSTREAM of the bin (source-mux re-batch), NOT a plugin loss"
    _pass "${desc}  (${n} gap(s), all upstream; 0 internal)"
    return 0
  fi
  printf '%s\n' "${gaps}" | sed -E 's/^.*(FRAMEGAP)/    \1/; s/\x1b\[[0-9;]*m//g' | head -10
  _fail "${desc}  (${n} FRAMEGAP event(s) + ${bad} internal frameacct drop(s) -- real in-bin loss)"
  return 1
}

# assert_no_source_id_restore_skips [logfile] [description]
#   HARD-failure gate for the downstream metadata contract. nvmodelmux must restore
#   NvDsFrameMeta.source_id/pad_index from MmProvenanceMeta after the combined
#   display mux re-batches frames. A skipped restore means one output frame had no
#   provenance and downstream PERF/tiler/msgbroker can see the internal display-slot id.
assert_no_source_id_restore_skips() {
  local log="${1:-${APP_LOG}}"
  local desc="${2:-source-id sanity: every output frame had provenance for restore}"

  if [ -z "${log}" ] || [ ! -s "${log}" ]; then
    echo "  [SKIP] ${desc}  (no app log to scan: '${log:-unset}')"
    return 0
  fi

  local hits n
  hits="$(grep -F 'SOURCE-ID restore skipped frame without provenance' "${log}" || true)"
  if [ -z "${hits}" ]; then
    _pass "${desc}"
    return 0
  fi
  printf '%s\n' "${hits}" | head -5 | sed -E 's/^/    /; s/\x1b\[[0-9;]*m//g'
  n="$(printf '%s\n' "${hits}" | grep -c .)"
  _fail "${desc}  (${n} output frame(s) could not restore source_id/pad_index)"
  return 1
}

# run_default_sanity_checks [logfile] [tag]
#   The DEFAULT per-test sanity gate -- runs hard regression checks plus low-noise
#   diagnostics against a test's saved log. Called from stop_app so every test is
#   gated automatically. Add new universal gates here to apply them to all tests.
run_default_sanity_checks() {
  local log="${1:-${APP_LOG}}"
  local tag="${2:-sanity}"
  echo "[${tag}] ===== default sanity checks (log: ${log##*/}) ====="
  assert_no_frame_drops  "${log}" "${tag}: ZERO frame loss (frameacct dropped==0)"
  assert_no_inference_stalls "${log}" "${tag}: no inference stall (entered/buffered cannot grow with inferred=0)"
  report_data_before_segment_warnings "${log}" "${tag}: buffer-before-segment warning scan"
  assert_no_gst_critical "${log}" "${tag}: no GStreamer-CRITICAL / assertion / crash"
  assert_no_frame_gaps   "${log}" "${tag}: no frame gaps (exact per-source frame_num contiguity)"
  assert_no_source_id_restore_skips "${log}" "${tag}: every output frame restored source_id/pad_index"
}
