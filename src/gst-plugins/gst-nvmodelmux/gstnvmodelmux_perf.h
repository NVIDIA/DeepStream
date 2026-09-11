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

/*
 * Per-model / per-sensor performance metrics (fps + inference latency).
 *
 * Self-contained module: depends only on glib + the small types it owns. Callers pass
 * source_id / latency / now-timestamp as parameters; it never reaches into plugin globals.
 * Surfaced (additively) through the model/status REST API and gated by the
 * `attach-perf-metric` config flag (the caller owns the gate; this module is pure mechanism).
 *
 * THREADING CONTRACT (important):
 *   - modelmux_perf_mark_infer_in()  : HOT PATH (nvinfer sink probe), lock-free. Single writer per ModelMuxPerf.
 *   - modelmux_perf_mark_infer_out() : HOT PATH (nvinfer src/provenance probe), lock-free. Single writer
 *                                per ModelMuxPerf and per source_id (one bin/one shard serves a source).
 *                                Only INCREMENTS monotonic 64-bit counters (atomic load/store on the
 *                                target arch) -- never mutates container structure.
 *   - modelmux_perf_tick()           : caller holds the bin lock (e.g. ib->lock); reads the monotonic
 *                                counters and recomputes the published fps/latency snapshot.
 *   - modelmux_perf_get_model() /
 *     modelmux_perf_get_source()     : caller holds the bin lock; read the snapshot only.
 *   No lock is taken inside this module; the single-writer-per-counter + atomic-64 invariant plus
 *   the caller's lock around tick/get is what makes it safe (mirrors the plugin's existing lock-free
 *   `acct` frame-accounting). Per-source storage is a FIXED array (no hot-path allocation), so there
 *   is no structural race between the lock-free writer and the locked reader.
 */

#ifndef GST_NVMODELMUX_PERF_H
#define GST_NVMODELMUX_PERF_H

#include <glib.h>

/* default + clamp for the measurement window (seconds). */
#define MM_PERF_DEFAULT_INTERVAL_SEC 5
#define MM_PERF_MIN_INTERVAL_SEC     1
/* max source_id tracked per model bin (flat, lock-free; matches the plugin's MM_ACCT_MAX). */
#define MM_PERF_MAX_SOURCES          256

/*
 * ModelMuxPerfCounter -- one sliding-window rate+latency counter. THE single source of truth for the
 * fps/latency math: the delta/elapsed/divide-by-zero logic lives ONLY in modelmux_perf_counter_tick().
 * Reused identically for the bin-aggregate counter AND every per-source counter.
 */
typedef struct
{
  /* monotonic totals -- bumped lock-free by modelmux_perf_counter_record() (hot path). */
  guint64 frames;       /**< frames counted                                      */
  guint64 lat_sum_us;   /**< sum of per-record latencies (us)                    */
  guint64 lat_n;        /**< number of latency samples                           */
  /* window bookkeeping + published snapshot -- written ONLY by tick() (under caller lock). */
  guint64 last_frames;
  guint64 last_lat_sum;
  guint64 last_lat_n;
  gint64  last_ts_us;   /**< monotonic time of the previous tick (0 => uninit)   */
  gdouble fps;          /**< published: frames/sec over the last window          */
  gdouble lat_ms;       /**< published: avg latency (ms) over the last window    */
} ModelMuxPerfCounter;

/*
 * ModelMuxPerf -- a model bin's performance state: one aggregate counter + a fixed per-source array,
 * plus the batch-arrival timestamp. Opaque to callers other than via the modelmux_perf_* API.
 */
typedef struct _MmPerf
{
  gint64        t_enter_us;                      /**< nvinfer-sink batch arrival (single writer) */
  ModelMuxPerfCounter agg;                             /**< bin aggregate (model throughput/latency)   */
  ModelMuxPerfCounter by_source[MM_PERF_MAX_SOURCES];  /**< per source_id                              */
} ModelMuxPerf;

/* ---- counter primitive (the only place the rate/latency formula lives) ---- */

/** Hot-path record: count one frame; if lat_us>0 also add a latency sample. Lock-free. */
static inline void
modelmux_perf_counter_record (ModelMuxPerfCounter * c, guint64 lat_us)
{
  c->frames++;
  if (lat_us) {
    c->lat_sum_us += lat_us;
    c->lat_n++;
  }
}

/** Recompute fps/lat_ms from the delta since the previous tick; advance the window.
 *  The FIRST tick (last_ts_us==0) only establishes the baseline (reports 0). Caller-locked. */
void modelmux_perf_counter_tick (ModelMuxPerfCounter * c, gint64 now_us);

/** Zero a counter (totals + snapshot). */
void modelmux_perf_counter_clear (ModelMuxPerfCounter * c);

/* ---- ModelMuxPerf object (per model bin) ---- */

ModelMuxPerf *modelmux_perf_new (void);
void    modelmux_perf_free (ModelMuxPerf * p);

/** HOT PATH: stamp the batch arrival time at the nvinfer sink. */
static inline void
modelmux_perf_mark_infer_in (ModelMuxPerf * p, gint64 now_us)
{
  if (p)
    p->t_enter_us = now_us;
}

/** HOT PATH: called once per output frame at the nvinfer src. Derives the batch latency
 *  (now - t_enter, identical for every frame of the batch) and records it into the aggregate
 *  and the frame's per-source counter. `now_us` should be sampled ONCE per batch by the caller
 *  and passed unchanged for every frame of that batch. */
void modelmux_perf_mark_infer_out (ModelMuxPerf * p, gint64 now_us, gint source_id);

/** Caller-locked: tick the aggregate + every active per-source counter. */
void modelmux_perf_tick (ModelMuxPerf * p, gint64 now_us);

/** Caller-locked: read the aggregate snapshot. */
void modelmux_perf_get_model (const ModelMuxPerf * p, gdouble * fps, gdouble * lat_ms);

/** Caller-locked: read a per-source snapshot (0/0 if unknown source or NULL). */
void modelmux_perf_get_source (const ModelMuxPerf * p, gint source_id, gdouble * fps, gdouble * lat_ms);

/** WHOLE-device stats via NVML (dlopen'd lazily; FALSE = NVML/device unavailable).
 *  Attribution to a model is by CO-LOCATION only (which instances run there). */
gboolean modelmux_gpu_stats (gint gpu, guint * util_pct, guint * mem_used_mb,
    guint * mem_total_mb);

#endif /* GST_NVMODELMUX_PERF_H */
