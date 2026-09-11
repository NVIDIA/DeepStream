/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>
#include "gstnvmodelmux_perf.h"

/* ---- ModelMuxPerfCounter ---- */

void
modelmux_perf_counter_clear (ModelMuxPerfCounter * c)
{
  if (c)
    memset (c, 0, sizeof (*c));
}

void
modelmux_perf_counter_tick (ModelMuxPerfCounter * c, gint64 now_us)
{
  if (!c)
    return;

  /* First tick just establishes the window baseline (no rate yet). */
  if (c->last_ts_us == 0) {
    c->last_ts_us = now_us;
    c->last_frames = c->frames;
    c->last_lat_sum = c->lat_sum_us;
    c->last_lat_n = c->lat_n;
    c->fps = 0.0;
    c->lat_ms = 0.0;
    return;
  }

  /* snapshot the monotonic totals once (written lock-free by the hot path). */
  guint64 frames = c->frames, lat_sum = c->lat_sum_us, lat_n = c->lat_n;
  gdouble elapsed_s = (gdouble) (now_us - c->last_ts_us) / 1.0e6;

  c->fps = (elapsed_s > 0.0 && frames >= c->last_frames)
      ? (gdouble) (frames - c->last_frames) / elapsed_s : 0.0;

  guint64 dn = (lat_n >= c->last_lat_n) ? (lat_n - c->last_lat_n) : 0;
  c->lat_ms = dn ? (gdouble) (lat_sum - c->last_lat_sum) / (gdouble) dn / 1000.0 : 0.0;

  c->last_frames = frames;
  c->last_lat_sum = lat_sum;
  c->last_lat_n = lat_n;
  c->last_ts_us = now_us;
}

/* ---- ModelMuxPerf ---- */

ModelMuxPerf *
modelmux_perf_new (void)
{
  return g_new0 (ModelMuxPerf, 1);   /* all counters zeroed; t_enter_us = 0 => first lat skipped */
}

void
modelmux_perf_free (ModelMuxPerf * p)
{
  g_free (p);                  /* fixed arrays inline; nothing else owned */
}

void
modelmux_perf_mark_infer_out (ModelMuxPerf * p, gint64 now_us, gint source_id)
{
  if (!p)
    return;
  /* batch latency: identical for every frame of this batch (computed from the single
   * t_enter stamped at the nvinfer sink). 0 until the first batch has entered. */
  guint64 lat = (p->t_enter_us > 0 && now_us > p->t_enter_us)
      ? (guint64) (now_us - p->t_enter_us) : 0;

  modelmux_perf_counter_record (&p->agg, lat);
  if (source_id >= 0 && source_id < MM_PERF_MAX_SOURCES)
    modelmux_perf_counter_record (&p->by_source[source_id], lat);
}

void
modelmux_perf_tick (ModelMuxPerf * p, gint64 now_us)
{
  if (!p)
    return;
  modelmux_perf_counter_tick (&p->agg, now_us);
  for (gint i = 0; i < MM_PERF_MAX_SOURCES; i++) {
    ModelMuxPerfCounter *c = &p->by_source[i];
    /* skip never-touched sources cheaply (no frames recorded and no baseline set). */
    if (c->frames == 0 && c->last_ts_us == 0)
      continue;
    modelmux_perf_counter_tick (c, now_us);
  }
}

void
modelmux_perf_get_model (const ModelMuxPerf * p, gdouble * fps, gdouble * lat_ms)
{
  if (fps)
    *fps = p ? p->agg.fps : 0.0;
  if (lat_ms)
    *lat_ms = p ? p->agg.lat_ms : 0.0;
}

void
modelmux_perf_get_source (const ModelMuxPerf * p, gint source_id, gdouble * fps, gdouble * lat_ms)
{
  const ModelMuxPerfCounter *c =
      (p && source_id >= 0 && source_id < MM_PERF_MAX_SOURCES) ? &p->by_source[source_id] : NULL;
  if (fps)
    *fps = c ? c->fps : 0.0;
  if (lat_ms)
    *lat_ms = c ? c->lat_ms : 0.0;
}

/* ------------------------------------------------------------------ */
/* Per-DEVICE stats via NVML (utilization + VRAM), used by the perf   */
/* tables, model/status "gpus" section and the auto-gpu-scale health  */
/* probe. Whole-device numbers: attribution to a model is by          */
/* CO-LOCATION (which instances run on that device), never per-kernel.*/
/* The gpu argument is a CUDA ordinal, translated to the NVML device  */
/* via its PCI BUS ID (cudaDeviceGetPCIBusId is context-free): CUDA   */
/* and NVML enumerate devices in DIFFERENT orders unless              */
/* CUDA_DEVICE_ORDER=PCI_BUS_ID, so an index-keyed lookup would read  */
/* the WRONG device's stats on multi-GPU boxes. No CUDA context is    */
/* created here (safe under ib->lock, no hidden VRAM cost).           */
/* NVML init is lazy + one-shot; every failure degrades to "no data". */
/*                                                                    */
/* NVML is resolved via dlopen("libnvidia-ml.so.1"), NEVER linked:    */
/*  - Tegra/L4T (aarch64) has no NVML at all -- a link-time -lnvidia-ml */
/*    would make the plugin unbuildable/unloadable there; with dlopen  */
/*    the probe just degrades to "no data" (callers already handle it).*/
/*  - x86 build sysroots (GVS/tmake) carry no driver libs either, so  */
/*    even there a link-time dependency needs stub-dir hacks. dlopen  */
/*    removes the build-time dependency on every arch identically.    */
/* Only the 4 entry points used here are declared, ABI-minimal        */
/* (nvmlMemory_t is the v1 layout matching the unversioned            */
/* nvmlDeviceGetMemoryInfo symbol), so no NVML SDK header is needed.  */
/* ------------------------------------------------------------------ */
#include <cuda_runtime_api.h>
#include <dlfcn.h>

typedef void *ModelMuxNvmlDevice;                       /* nvmlDevice_t */
typedef struct { guint gpu; guint memory; } ModelMuxNvmlUtilization;  /* nvmlUtilization_t */
typedef struct { unsigned long long total, free, used; } ModelMuxNvmlMemory; /* nvmlMemory_t v1 */
#define MM_NVML_SUCCESS 0                          /* NVML_SUCCESS */

typedef struct
{
  int (*init) (void);                                       /* nvmlInit_v2 */
  int (*dev_by_busid) (const char *, ModelMuxNvmlDevice *);       /* nvmlDeviceGetHandleByPciBusId_v2 */
  int (*util_rates) (ModelMuxNvmlDevice, ModelMuxNvmlUtilization *);    /* nvmlDeviceGetUtilizationRates */
  int (*mem_info) (ModelMuxNvmlDevice, ModelMuxNvmlMemory *);           /* nvmlDeviceGetMemoryInfo */
} ModelMuxNvml;

/* one-shot: dlopen + resolve + init; any failure -> NULL forever (no retry
 * churn per perf tick; a driver does not appear mid-process anyway) */
static gpointer
modelmux_nvml_init_once (gpointer unused)
{
  static ModelMuxNvml api;
  void *h;

  (void) unused;
  h = dlopen ("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h)
    return NULL;
  api.init         = dlsym (h, "nvmlInit_v2");
  api.dev_by_busid = dlsym (h, "nvmlDeviceGetHandleByPciBusId_v2");
  api.util_rates   = dlsym (h, "nvmlDeviceGetUtilizationRates");
  api.mem_info     = dlsym (h, "nvmlDeviceGetMemoryInfo");
  if (!api.init || !api.dev_by_busid || !api.util_rates || !api.mem_info ||
      api.init () != MM_NVML_SUCCESS) {
    dlclose (h);
    return NULL;
  }
  return &api;                                   /* handle stays open for process life */
}

/* Per-device TTL cache: several callers sample under modelmux_bin->lock (status build,
 * debug dump, health checks) and NVML/CUDA driver calls run 1-10ms each -- a 1 Hz
 * status poller on a multi-GPU box would serialize those milliseconds against every
 * attach/route needing the lock. One real sample per device per second; everyone
 * else reads the cache. (Failures are cached too -- a broken driver must not be
 * hammered from under the bin lock either.) */
#define MM_GPU_STATS_TTL_US  (1 * G_USEC_PER_SEC)
#define MM_GPU_STATS_CACHE_MAX 64

gboolean
modelmux_gpu_stats (gint gpu, guint * util_pct, guint * mem_used_mb,
    guint * mem_total_mb)
{
  static GOnce once = G_ONCE_INIT;
  static GMutex cache_lock;
  static struct
  {
    gint64 t_us;                /* 0 = never sampled */
    gboolean ok;
    guint util, used_mb, total_mb;
  } cache[MM_GPU_STATS_CACHE_MAX];
  const ModelMuxNvml *nvml;
  gchar busid[32];
  ModelMuxNvmlDevice dev;
  ModelMuxNvmlUtilization util;
  ModelMuxNvmlMemory mem;
  guint u = 0, um = 0, tm = 0;
  gboolean ok = FALSE;
  gint64 now = g_get_monotonic_time ();

  if (util_pct)   *util_pct = 0;
  if (mem_used_mb)  *mem_used_mb = 0;
  if (mem_total_mb) *mem_total_mb = 0;
  if (gpu < 0)
    return FALSE;

  if (gpu < MM_GPU_STATS_CACHE_MAX) {
    g_mutex_lock (&cache_lock);
    if (cache[gpu].t_us && now - cache[gpu].t_us < MM_GPU_STATS_TTL_US) {
      ok = cache[gpu].ok;
      if (util_pct)     *util_pct = cache[gpu].util;
      if (mem_used_mb)  *mem_used_mb = cache[gpu].used_mb;
      if (mem_total_mb) *mem_total_mb = cache[gpu].total_mb;
      g_mutex_unlock (&cache_lock);
      return ok;
    }
    g_mutex_unlock (&cache_lock);
  }

  /* sample OUTSIDE cache_lock (concurrent duplicate samples are cheaper than
   * serializing every caller behind one driver ioctl) */
  g_once (&once, modelmux_nvml_init_once, NULL);
  nvml = (const ModelMuxNvml *) once.retval;
  if (nvml &&
      cudaDeviceGetPCIBusId (busid, sizeof (busid), gpu) == cudaSuccess &&
      nvml->dev_by_busid (busid, &dev) == MM_NVML_SUCCESS) {
    ok = TRUE;
    if (nvml->util_rates (dev, &util) == MM_NVML_SUCCESS)
      u = util.gpu;
    if (nvml->mem_info (dev, &mem) == MM_NVML_SUCCESS) {
      um = (guint) (mem.used / (1024 * 1024));
      tm = (guint) (mem.total / (1024 * 1024));
    }
  }

  if (gpu < MM_GPU_STATS_CACHE_MAX) {
    g_mutex_lock (&cache_lock);
    cache[gpu].t_us = now;
    cache[gpu].ok = ok;
    cache[gpu].util = u;
    cache[gpu].used_mb = um;
    cache[gpu].total_mb = tm;
    g_mutex_unlock (&cache_lock);
  }
  if (util_pct)     *util_pct = u;
  if (mem_used_mb)  *mem_used_mb = um;
  if (mem_total_mb) *mem_total_mb = tm;
  return ok;
}
