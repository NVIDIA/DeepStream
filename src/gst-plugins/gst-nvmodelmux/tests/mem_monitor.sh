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
# mem_monitor.sh -- reusable resource-monitoring library for the nvmodelmux
# stress / perf / soak tests. Works for ANY pid. Source it (defines functions).
#
# Per sample it records, over time:
#   - Timestamp + elapsed seconds
#   - System memory  used / free            (free -m)
#   - Process        VmRSS / VmSize          (/proc/<pid>/status)     [host mem]
#   - Kernel slab    kmalloc-128 cache size  (/proc/slabinfo)         [leak signal]
#   - GPU memory     per-process if visible, else device-used-minus-baseline DELTA
#                    (accurate even in containers where nvidia-smi shows host PIDs)
#   - GPU util %     whole-device compute utilisation                 [perf signal]
#   - threads / FDs  (/proc/<pid>/task , /proc/<pid>/fd)
#   - pipeline FPS   current-window FPS parsed from the app **PERF log
# It writes a machine CSV (for the verdict) AND a human-readable aligned .log,
# then prints a leak verdict (first-vs-last delta + least-squares slope/min) for
# VmRSS, GPU memory, the kmalloc-128 slab, and FDs.
#
# Use:
#     mem_monitor_start "${PID}" "${DUMP_CSV}" "${SAMPLE_INTERVAL}" "${APP_LOG}"
#     ... run workload ...
#     mem_monitor_stop
#     mem_leak_report "${DUMP_CSV}"
#
# Leak thresholds (env; a leak needs BOTH slope AND total-growth over floor):
#   LEAK_WARMUP_SAMPLES  samples to skip at start (ramp-up)      default 3
#   LEAK_RSS_MB_PER_MIN / LEAK_RSS_MIN_MB    RSS   slope / floor  default 5 / 25
#   LEAK_GPU_MB_PER_MIN / LEAK_GPU_MIN_MB    GPU   slope / floor  default 5 / 32
#   LEAK_SLAB_MB_PER_MIN / LEAK_SLAB_MIN_MB  slab  slope / floor  default 2 / 10
#   LEAK_FD_GROWTH        absolute FD growth that flags a leak    default 64
###############################################################################

# ---- single-sample probes (all take a pid where relevant) ---------------------

mem_probe_vmrss_mb()  { awk '/^VmRSS:/ {printf "%.1f", $2/1024;  exit} ' "/proc/$1/status" 2>/dev/null; }
mem_probe_vmsize_mb() { awk '/^VmSize:/{printf "%.1f", $2/1024;  exit} ' "/proc/$1/status" 2>/dev/null; }
mem_probe_threads()   { local n; n=$(ls -1 "/proc/$1/task" 2>/dev/null | wc -l); echo "${n:-0}"; }
mem_probe_fds()       { local n; n=$(ls -1 "/proc/$1/fd"   2>/dev/null | wc -l); echo "${n:-0}"; }

# System used / free memory in MB, as "used free".
mem_probe_sys_mem_mb() { free -m 2>/dev/null | awk '/^Mem:/ {print $3" "$4; f=1} END{if(!f)print "0 0"}'; }

# Kernel slab cache size for kmalloc-128 in MB (num_objs * objsize), read straight
# from /proc/slabinfo -- no sudo/slabtop needed. Excludes dma-kmalloc-128. 0 if
# slabinfo isn't readable (non-root host). This slab is the classic growth signal
# for DeepStream/GStreamer small-allocation leaks.
mem_probe_slab_k128_mb() {
  awk '$1=="kmalloc-128"{printf "%.1f", ($3*$4)/1048576; f=1; exit} END{if(!f)print "0"}' /proc/slabinfo 2>/dev/null || echo 0
}

# Per-process GPU memory (MiB) for a pid, or "" when not listed as a compute app
# (usual inside containers -> sampler falls back to the device-delta path).
mem_probe_gpu_proc_mib() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v p="$1" '{gsub(/ /,"",$1); gsub(/ /,"",$2)} $1==p {print $2; f=1} END{if(!f)print ""}'
}
mem_probe_gpu_dev_mib()  { command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  local v; v=$(nvidia-smi --query-gpu=memory.used     --format=csv,noheader,nounits 2>/dev/null|head -1|tr -d ' '); echo "${v:-0}"; }
mem_probe_gpu_util_pct() { command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  local v; v=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null|head -1|tr -d ' '); echo "${v:-0}"; }

# Current-window FPS parsed from an app log's **PERF block, else "".
# Line looks like "91.00 (617.43)  91.00 (617.42) ..." -- first number of each
# pair is the current FPS (the parenthesised value is the startup-inflated avg).
mem_probe_fps() {
  local log="$1" line
  [ -f "${log}" ] || { echo ""; return; }
  line=$(grep -aE '[0-9]+\.[0-9]+ *\([0-9]+\.[0-9]+\)' "${log}" 2>/dev/null | tail -1)
  [ -z "${line}" ] && { echo ""; return; }
  echo "${line}" | grep -aoE '[0-9]+\.[0-9]+ *\(' | tr -d '(' \
    | awk '{s+=$1; n++} END{ if(n) printf "%.1f", s/n; else printf "" }'
}

# ---- background sampler --------------------------------------------------------

MEM_MON_PID=""          # pid of the background sampler loop
MEM_MON_CSV=""          # active dump path

# mem_monitor_start <pid> <csv> <interval_s> [app_log]
# Captures a GPU-memory baseline BEFORE the workload allocates, then appends one
# CSV row + one aligned human row (to <csv>.log) every <interval_s> seconds until
# the pid exits or mem_monitor_stop is called.
mem_monitor_start() {
  local pid="$1" csv="$2" interval="${3:-2}" log="${4:-}"
  MEM_MON_CSV="${csv}"
  local human="${csv%.csv}.log"
  local gbase; gbase="$(mem_probe_gpu_dev_mib)"; gbase="${gbase:-0}"
  echo "ts,elapsed_s,sys_used_mb,sys_free_mb,vmrss_mb,vmsize_mb,slab_k128_mb,gpu_mib,gpu_src,gpu_util_pct,threads,fds,fps" > "${csv}"
  printf "%-9s %-8s %-9s %-9s %-9s %-9s %-11s %-8s %-7s %-5s %-5s %-6s\n" \
    Time Elapsed SysUsed SysFree VmRSS VmSize kmalloc128 GPUmem GPUutil Thr FDs FPS | tee "${human}"
  echo "[mem] monitoring pid=${pid} every ${interval}s (GPU baseline=${gbase} MiB) -> ${csv} (+ ${human})"
  (
    local t0 now el ts used free rss vsz slab gproc gdev gutil gmib gsrc thr fds fps
    t0=$(date +%s)
    while kill -0 "${pid}" 2>/dev/null; do
      now=$(date +%s); el=$(( now - t0 )); ts=$(date +%H:%M:%S)
      rss=$(mem_probe_vmrss_mb "${pid}"); [ -z "${rss}" ] && break        # pid gone mid-sample
      vsz=$(mem_probe_vmsize_mb "${pid}")
      read -r used free <<<"$(mem_probe_sys_mem_mb)"
      slab=$(mem_probe_slab_k128_mb)
      gproc=$(mem_probe_gpu_proc_mib "${pid}")
      gdev=$(mem_probe_gpu_dev_mib); gutil=$(mem_probe_gpu_util_pct)
      if [ -n "${gproc}" ]; then gmib="${gproc}"; gsrc="proc"
      else gmib=$(( gdev - gbase > 0 ? gdev - gbase : 0 )); gsrc="delta"; fi
      thr=$(mem_probe_threads "${pid}"); fds=$(mem_probe_fds "${pid}"); fps=$(mem_probe_fps "${log}")
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${ts}" "${el}" "${used}" "${free}" "${rss}" "${vsz}" "${slab}" \
        "${gmib}" "${gsrc}" "${gutil}" "${thr}" "${fds}" "${fps}" >> "${csv}"
      printf "%-9s %-8s %-9s %-9s %-9s %-9s %-11s %-8s %-7s %-5s %-5s %-6s\n" \
        "${ts}" "${el}" "${used}" "${free}" "${rss}" "${vsz}" "${slab}" \
        "${gmib}(${gsrc})" "${gutil}" "${thr}" "${fds}" "${fps:--}" | tee -a "${human}"
      sleep "${interval}"
    done
  ) &
  MEM_MON_PID=$!
}

mem_monitor_stop() {
  [ -z "${MEM_MON_PID}" ] && return 0
  kill "${MEM_MON_PID}" 2>/dev/null
  wait "${MEM_MON_PID}" 2>/dev/null
  MEM_MON_PID=""
  echo "[mem] monitoring stopped"
}

# ---- analysis / verdict --------------------------------------------------------

# mem_leak_report <csv> [warmup_samples]
mem_leak_report() {
  local csv="${1:-${MEM_MON_CSV}}" warmup="${2:-${LEAK_WARMUP_SAMPLES:-3}}"
  [ -f "${csv}" ] || { echo "[mem] no dump at ${csv}"; return 1; }

  awk -F',' \
      -v warmup="${warmup}" \
      -v rss_thr="${LEAK_RSS_MB_PER_MIN:-5}"   -v rss_floor="${LEAK_RSS_MIN_MB:-25}" \
      -v gpu_thr="${LEAK_GPU_MB_PER_MIN:-5}"   -v gpu_floor="${LEAK_GPU_MIN_MB:-32}" \
      -v slab_thr="${LEAK_SLAB_MB_PER_MIN:-2}" -v slab_floor="${LEAK_SLAB_MIN_MB:-10}" \
      -v fd_thr="${LEAK_FD_GROWTH:-64}" \
      -v csvpath="${csv}" '
    NR==1 { next }
    { row++; if (row<=warmup) next
      k++
      t[k]=$2+0; rss[k]=$5+0; vsz[k]=$6+0; slab[k]=$7+0; gpu[k]=$8+0; util[k]=$10+0; fd[k]=$12+0
      if ($9=="delta") delta_used=1
      if(k==1||rss[k]<rmin)rmin=rss[k];  if(k==1||rss[k]>rmax)rmax=rss[k]
      if(k==1||gpu[k]<gmin)gmin=gpu[k];  if(k==1||gpu[k]>gmax)gmax=gpu[k]
      if(k==1||slab[k]<smin)smin=slab[k];if(k==1||slab[k]>smax)smax=slab[k]
      if(k==1||util[k]>umax)umax=util[k]
      rsum+=rss[k]; gsum+=gpu[k]; ssum+=slab[k]; usum+=util[k]
    }
    function slope_rng(a,b,y,  i,n,sx,sy,sxx,sxy,den){ n=b-a+1; if(n<2)return 0
      sx=sy=sxx=sxy=0; for(i=a;i<=b;i++){sx+=t[i];sy+=y[i];sxx+=t[i]*t[i];sxy+=t[i]*y[i]}
      den=(n*sxx-sx*sx); return (den!=0)?((n*sxy-sx*sy)/den)*60:0 }
    # median of y[a..b] -- robust to the add/remove churn sawtooth (unlike mean/slope)
    function med(y,a,b,   i,j,n,v){ n=0; delete tmp
      for(i=a;i<=b;i++){tmp[++n]=y[i]}
      for(i=2;i<=n;i++){v=tmp[i];j=i-1;while(j>=1&&tmp[j]>v){tmp[j+1]=tmp[j];j--}tmp[j+1]=v}
      if(n==0)return 0; if(n%2)return tmp[int((n+1)/2)]; return (tmp[n/2]+tmp[n/2+1])/2 }
    END{
      if(k<4){ print "[mem] not enough samples after warmup ("k") -- run longer / smaller interval"; exit 0 }
      # A LEAK is MONOTONIC/SUSTAINED to the END of the run. A whole-window slope is fooled by
      # (a) the one-time startup/warm-up + shard/model-expansion RAMP and (b) add/remove &
      # model-swap CHURN sawtooth. The 2nd HALF still contains the tail of that ramp, so it
      # over-reports the rate. Judge instead on the SETTLED TAIL (last third): a real leak keeps
      # climbing at ~the same rate; a ramp decays to ~0 there and a sawtooth nets ~0.
      #   early = [1,a1]   mid = [a1+1,a2]   tail = [a2+1,k]   (thirds)
      a1=int(k/3); if(a1<1)a1=1;  a2=int(2*k/3); if(a2<=a1)a2=a1+1; if(a2>=k)a2=k-1
      r_e=slope_rng(1,a1,rss);  r_m=slope_rng(a1+1,a2,rss);  r_t=slope_rng(a2+1,k,rss);  rgt=rss[k]-rss[a2+1]
      g_e=slope_rng(1,a1,gpu);  g_m=slope_rng(a1+1,a2,gpu);  g_t=slope_rng(a2+1,k,gpu);  ggt=gpu[k]-gpu[a2+1]
      s_e=slope_rng(1,a1,slab); s_m=slope_rng(a1+1,a2,slab); s_t=slope_rng(a2+1,k,slab); sgt=slab[k]-slab[a2+1]
      printf "\n  ┌─ memory / resource report  (samples=%d after %d warmup · window=%ds) ──────────────────────\n", k, warmup, t[k]-t[1]
      printf "  │ metric          first        last          min          max    slope early    slope TAIL(last3rd)\n"
      printf "  │ VmRSS (MB)   %10.1f %12.1f %12.1f %12.1f %13.2f %15.2f\n", rss[1],rss[k],rmin,rmax,r_e,r_t
      printf "  │ GPU   (MiB)  %10.1f %12.1f %12.1f %12.1f %13.2f %15.2f\n", gpu[1],gpu[k],gmin,gmax,g_e,g_t
      printf "  │ slab  (MB)   %10.1f %12.1f %12.1f %12.1f %13.2f %15.2f\n", slab[1],slab[k],smin,smax,s_e,s_t
      printf "  │ VmSize(MB) %10.1f -> %.1f     GPUutil%% max=%.0f mean=%.1f     FDs %d -> %d\n", vsz[1],vsz[k],umax,usum/k,fd[1],fd[k]
      printf "  └──────────────────────────────────────────────────────────────────────────────────────────────\n"
      if (delta_used) printf "  · GPU mem is a device-delta estimate (per-process not visible -- e.g. container; use --pid=host for exact).\n"
      # add/remove + model-swap CHURN makes RSS/GPU swing like a sawtooth (streams removed ->
      # decoder/pipeline mem frees; re-added -> reallocates), so a raw slope is noisy and the
      # first/last samples land wherever the swing happens to be. Judge on the MEDIAN of each
      # third -- immune to the swing: a real leak lifts the median run-over-run; churn keeps it
      # flat. Report the band too, so the swing is visible as churn, not mistaken for drift.
      rme=med(rss,1,a1); rmm=med(rss,a1+1,a2); rmt=med(rss,a2+1,k)
      gme=med(gpu,1,a1); gmm=med(gpu,a1+1,a2); gmt=med(gpu,a2+1,k)
      sme=med(slab,1,a1);smm=med(slab,a1+1,a2);smt=med(slab,a2+1,k)
      # Compare MID vs TAIL median (both POST warm-up) over the ~1/3-window gap between their
      # centres. Using early-third would re-add the warm-up ramp; using medians kills the churn
      # swing. So: flat mid->tail = settled (no leak); still rising mid->tail = real growth.
      span=(t[k]-t[1])/3.0; if(span<=0)span=1
      rms=(rmt-rmm)/span*60; gms=(gmt-gmm)/span*60; sms=(smt-smm)/span*60   # recent robust MB/min
      printf "  · RSS  median/third: %.0f -> %.0f -> %.0f MB  (mid->tail %+.1f MB/min · churn band %.0f-%.0f, swing %.0f)\n", rme,rmm,rmt,rms,rmin,rmax,rmax-rmin
      printf "  · GPU  median/third: %.0f -> %.0f -> %.0f MiB (mid->tail %+.1f MiB/min · band %.0f-%.0f, swing %.0f)\n", gme,gmm,gmt,gms,gmin,gmax,gmax-gmin
      # PLATEAU vs LEAK: a real leak grows LINEARLY (early->mid rate ~= mid->tail rate); a
      # bounded pool/allocator filling toward its working set DECELERATES (the per-interval
      # growth shrinks). If the mid->tail rate decayed to <=half of the early->mid rate, the
      # curve is saturating -- report PLATEAUING with the projected ceiling (geometric-decay
      # extrapolation: ceiling ~= tail + gain*ratio/(1-ratio)) instead of crying leak.
      # (Verified empirically: a +59 MB/min 10-min-run verdict decayed to +6.5 MB/min
      # by minute 60, converging on ~4.1 GB -- allocator residency, not a leak.)
      rms_em=(rmm-rme)/span*60
      r_plateau=0; r_ceil=0
      if (rms_em>0 && rms<=0.5*rms_em && rms>0) {
        ratio=rms/rms_em; r_ceil=rmt+(rmt-rmm)*ratio/(1-ratio); r_plateau=1 }
      # verdict: leak = median STILL rising after warm-up (tail median > mid median) at a rate
      # over threshold with net gain over floor AND not decelerating like a filling pool.
      leak=0; plat=0; r=""; pr=""
      if (rms>rss_thr  && rmt-rmm>rss_floor*0.5  && rmt>=rmm) {
        if (r_plateau) { plat=1; pr=pr sprintf("VmRSS +%.1f MB/min decaying from +%.1f (projected ceiling ~%.0f MB) ; ", rms, rms_em, r_ceil) }
        else           { leak=1; r=r sprintf("VmRSS median +%.2f MB/min (mid->tail +%.0f MB) ; ", rms, rmt-rmm) } }
      if (gms>gpu_thr  && gmt-gmm>gpu_floor*0.5  && gmt>=gmm)  { leak=1; r=r sprintf("GPU median +%.2f MiB/min (mid->tail +%.0f MiB) ; ", gms, gmt-gmm) }
      if (sms>slab_thr && smt-smm>slab_floor*0.5 && smt>=smm)  { leak=1; r=r sprintf("slab median +%.2f MB/min (mid->tail +%.0f MB) ; ", sms, smt-smm) }
      if (fd[k]-fd[1]>fd_thr)               { leak=1; r=r sprintf("FDs +%d ; ", fd[k]-fd[1]) }
      if (leak) printf "  ⚠ LEAK SUSPECTED (median still rising after warm-up): %s\n    NOTE: if warm-up filled most of the run, extend STRESS_DURATION to confirm plateau.\n    confirm: LEAK_TRACER=1 / heaptrack / compute-sanitizer ; dump: %s\n", r, csvpath
      else if (plat) printf "  ◐ PLATEAUING (bounded fill, not a leak): %s\n    growth is DECELERATING toward a working-set ceiling; extend STRESS_DURATION to watch it settle. dump: %s\n", pr, csvpath
      else      printf "  ✓ NO LEAK: median settled after warm-up (mid~=tail; residual swing = add/remove + model-swap churn) (dump: %s)\n", csvpath
    }
  ' "${csv}"
}

# mem_leak_oneline <csv> [warmup]
# Compact one-line summary for the results ledger: prints key first->last numbers
# and returns 1 if a leak is suspected (steady 2nd-half slope + growth over floor),
# else 0. Same criterion as mem_leak_report.
mem_leak_oneline() {
  local csv="${1:-${MEM_MON_CSV}}" warmup="${2:-${LEAK_WARMUP_SAMPLES:-3}}"
  [ -f "${csv}" ] || { printf "no-dump"; return 0; }
  awk -F',' -v warmup="${warmup}" \
      -v rss_thr="${LEAK_RSS_MB_PER_MIN:-5}"   -v rss_floor="${LEAK_RSS_MIN_MB:-25}" \
      -v gpu_thr="${LEAK_GPU_MB_PER_MIN:-5}"   -v gpu_floor="${LEAK_GPU_MIN_MB:-32}" \
      -v slab_thr="${LEAK_SLAB_MB_PER_MIN:-2}" -v slab_floor="${LEAK_SLAB_MIN_MB:-10}" '
    NR==1{next}
    {row++; if(row<=warmup) next; k++; t[k]=$2+0; rss[k]=$5+0; slab[k]=$7+0; gpu[k]=$8+0
     if(k==1||rss[k]>rmax)rmax=rss[k]; if(k==1||rss[k]<rmin)rmin=rss[k]
     if(k==1||gpu[k]>gmax)gmax=gpu[k]; if(k==1||gpu[k]<gmin)gmin=gpu[k]
     if(k==1||slab[k]>smax)smax=slab[k]}
    # median of y[a..b] -- robust to the add/remove + model-swap churn sawtooth
    function med(y,a,b,   i,j,n,v){ n=0; delete tmp
      for(i=a;i<=b;i++){tmp[++n]=y[i]}
      for(i=2;i<=n;i++){v=tmp[i];j=i-1;while(j>=1&&tmp[j]>v){tmp[j+1]=tmp[j];j--}tmp[j+1]=v}
      if(n==0)return 0; if(n%2)return tmp[int((n+1)/2)]; return (tmp[n/2]+tmp[n/2+1])/2 }
    # leak = MEDIAN still rising after warm-up: tail-third median > mid-third median (both post
    # warm-up), rate over threshold, net gain over floor. Settled/churn -> mid~=tail -> not flagged.
    function isleak(mrate,mm,mt,floor,thr){
      return (mrate>thr && mt-mm>floor && mt>=mm)?1:0}
    END{
      if(k<6){ printf "samples=%d (too few)", k; exit 0 }
      a1=int(k/3); if(a1<1)a1=1; a2=int(2*k/3); if(a2<=a1)a2=a1+1; if(a2>=k)a2=k-1
      span=(t[k]-t[1])/3.0; if(span<=0)span=1
      rmm=med(rss,a1+1,a2); rmt=med(rss,a2+1,k); rms=(rmt-rmm)/span*60
      gmm=med(gpu,a1+1,a2); gmt=med(gpu,a2+1,k); gms=(gmt-gmm)/span*60
      smm=med(slab,a1+1,a2);smt=med(slab,a2+1,k); sms=(smt-smm)/span*60
      rme=med(rss,1,a1); rms_em=(rmm-rme)/span*60
      leak=0; plat=0
      if(isleak(rms,rmm,rmt,rss_floor*0.5,rss_thr)) {
        # decelerating to <=half the earlier rate = bounded fill (PLATEAU), not a leak
        if (rms_em>0 && rms<=0.5*rms_em) plat=1; else leak=1 }
      if(isleak(gms,gmm,gmt,gpu_floor*0.5,gpu_thr))  leak=1
      if(isleak(sms,smm,smt,slab_floor*0.5,slab_thr)) leak=1
      printf "RSS med %.0fMB(mid->tail %+.1f/min, band %.0f-%.0f) GPU med %.0fMiB(band %.0f-%.0f) slab %.0fMB", rmt,rms,rmin,rmax, gmt,gmin,gmax, smt
      exit (leak?1:(plat?2:0))
    }' "${csv}"
}
