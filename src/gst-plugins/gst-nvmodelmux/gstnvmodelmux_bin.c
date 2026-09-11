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
 * Reusable inference bins:
 *   - ModelBin : one model NAME for one ROLE (batched nvstreammux->nvinfer->nvstreamdemux)
 *   - ModelPool  : all ModelBins for one role (Primary / Shadow), keyed by name
 *   - ModelMuxBin : per-stream tee -> roles, role outputs -> one combined mux
 *
 * Per-model batching is achieved by sharing a ModelBin across the streams routed
 * to that model; a recycled index pool keeps the mux/demux request-pad ids bounded.
 *
 * Keep this file focused on graph lifetime, stream add/remove, and routing. REST
 * model-status JSON/query handling lives in gstnvmodel_status.c, and
 * periodic performance accounting lives in gstnvmodelmux_perf.c.
 */

#include "gstnvmodelmux_priv.h"
#include <stdio.h>
#include <stdlib.h>   /* atoi */

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvbufsurface.h"   /* MM_BUFCOPY_VERIFY: read NvBufSurface dataPtr to prove a real copy */
#include <string.h>
#include <cuda_runtime_api.h>   /* auto-gpu-scale: device count/properties/free-VRAM */

/* ------------------------------------------------------------------ */
/* logging + small helpers                                             */
/* ------------------------------------------------------------------ */

/* All logging routes through the plugin's GStreamer debug category
 * (gst_modelmux_debug_cat, registered in plugin_init), exactly like every stock
 * DeepStream/GStreamer element:
 *   - INFO  -> GST_INFO  : guarded runtime trace (GST_DEBUG=nvmodelmux:4+)
 *   - WARN  -> GST_WARNING
 *   - ERROR -> GST_ERROR
 * Nothing prints unconditionally; verbosity is controlled by GST_DEBUG. */
GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat
#define MM_INFO(fmt, ...)  GST_INFO (fmt, ##__VA_ARGS__)
#define MM_WARN(fmt, ...)  GST_WARNING (fmt, ##__VA_ARGS__)
#define MM_ERR(fmt, ...)   GST_ERROR (fmt, ##__VA_ARGS__)

/* Runtime toggle for the per-tile debug overlay (env MM_OVERLAY=1).
 * Cached on first use; 0 (off) by default. Replaces the old compile-time flag. */
static gboolean
modelmux_overlay_enabled (void)
{
  static gint cached = -1;
  if (cached < 0) {
    const gchar *e = g_getenv ("MM_OVERLAY");
    cached = (e && *e && g_strcmp0 (e, "0") != 0) ? 1 : 0;
  }
  return cached;
}

/* Runtime toggle for FRAME-NUMBER TRACE (env MM_FRAME_TRACE=1). OPT-IN, OFF by
 * default: when unset NOTHING below runs, so the data path is byte-for-byte
 * unchanged. When set, the bin stamps each input frame with its ORIGINAL
 * (source_id, frame_num) so the value can be recovered AFTER the internal
 * nvstreammux re-batches (which re-number frames) -- the only way to map an
 * output frame back to the exact input frame / detect a one-time drop by gap.
 * Diagnostic only; gated like MM_OVERLAY. */
static gboolean
modelmux_frametrace_enabled (void)
{
  static gint cached = -1;
  if (cached < 0) {
    const gchar *e = g_getenv ("MM_FRAME_TRACE");
    cached = (e && *e && g_strcmp0 (e, "0") != 0) ? 1 : 0;
  }
  return cached;
}

/* Debug overlay (temporary; removed once stable). TWO blocks per tile, same 5-line
 * format, so the data can be visually CROSS-CHECKED:
 *   LEFT  -- drawn by the ModelBin probe straight from the bin (authoritative).
 *   RIGHT -- drawn by the combined-mux probe by PARSING the InferenceProvenanceMeta off the
 *            batched buffer (proves the custom user-meta is produced + parsed right).
 *   source-id: <source_id>
 *   Frame-ID: <frame_num>
 *   Stream-Name: <camera_id>
 *   <model-name>: <Primary|Shadow>
 *   <engine basename>
 * nvdsosd renders one NvOSD_TextParams per LINE, so we emit 5 labels stacked by
 * y_offset. source_id is the nvmultiurisrcbin global id preserved on the frame. */

/* Draw a single info line at (x_offset, top) on one frame, colour-coded by role:
 * Primary -> green, Shadow -> red, anything else -> white. No background box.
 * `text` is copied (freed when the display meta is released). */
static void
modelmux_draw_label (NvDsBatchMeta * bmeta, NvDsFrameMeta * fm, guint x_offset,
    const gchar * text, const gchar * role)
{
  NvDsDisplayMeta *dm = nvds_acquire_display_meta_from_pool (bmeta);
  NvOSD_TextParams *tp;
  gdouble r = 1.0, g = 1.0, b = 1.0;             /* default: white */

  if (!dm) {                                     /* display-meta pool can exhaust under load */
    MM_WARN ("overlay: display-meta pool exhausted -- skipping label");
    return;
  }
  tp = &dm->text_params[0];

  if (g_strcmp0 (role, "Primary") == 0) {        /* Primary -> green */
    r = 0.0; g = 1.0; b = 0.0;
  } else if (g_strcmp0 (role, "Shadow") == 0) {  /* Shadow  -> red   */
    r = 1.0; g = 0.0; b = 0.35;
  }

  dm->num_labels = 1;
  tp->display_text = g_strdup (text);
  tp->x_offset = x_offset;
  tp->y_offset = 12;
  tp->font_params.font_name = (gchar *) "Sans";
  tp->font_params.font_size = 1;
  tp->font_params.font_color.red = r;
  tp->font_params.font_color.green = g;
  tp->font_params.font_color.blue = b;
  tp->font_params.font_color.alpha = 1.0;
  tp->set_bg_clr = 1;                             /* white background @ 0.5 alpha */
  tp->text_bg_clr.red = 0.0;
  tp->text_bg_clr.green = 0.0;
  tp->text_bg_clr.blue = 0.0;
  tp->text_bg_clr.alpha = 1.0;
  nvds_add_display_meta_to_frame (fm, dm);
}

/* ================================================================== */
/* A/B provenance user-meta -- standalone; independent of the  */
/* debug overlay above (which is temporary). Stamps every frame with   */
/* {stream, model, role, gie, engine} so an external comparator can    */
/* attribute and compare each model's outputs (§9.4).        */
/* ================================================================== */

/* NvDsUserMeta copy/release for InferenceProvenanceMeta. The struct is flat (fixed arrays),
 * so copy = memdup, release = free. REQUIRED so the meta survives the combined
 * nvstreammux (it rebuilds batch meta and invokes copy_func). DS convention. */
static gpointer
modelmux_provenance_copy (gpointer data, gpointer user_data)
{
  NvDsUserMeta *um = (NvDsUserMeta *) data;
  (void) user_data;
  if (!um || !um->user_meta_data)
    return NULL;
  return g_memdup2 (um->user_meta_data, sizeof (InferenceProvenanceMeta));
}

static void
modelmux_provenance_release (gpointer data, gpointer user_data)
{
  NvDsUserMeta *um = (NvDsUserMeta *) data;
  (void) user_data;
  if (um && um->user_meta_data) {
    g_free (um->user_meta_data);
    um->user_meta_data = NULL;
  }
}

/* ------------------------------------------------------------------ */
/* SOURCE TRACE meta (always-on source_id, MM_FRAME_TRACE uses frame_num too).   */
/* Stamped on each frame at the bin INPUT with its ORIGINAL (source_id, frame_num)*/
/* -- before any internal nvstreammux re-numbers it -- so the original identity   */
/* can be recovered downstream. Flat struct + copy_func so it survives the mux    */
/* re-batch (same mechanism that keeps InferenceProvenanceMeta alive across out_mux).    */
/* ------------------------------------------------------------------ */
#define MM_SRCTRACE_META_NAME "NVIDIA.DEEPSTREAM.MM.SRCTRACE"
typedef struct
{
  guint source_id;     /* global nvmultiurisrcbin source_id at the bin input  */
  guint frame_num;     /* ORIGINAL input frame_num (pre internal re-mux)      */
} ModelMuxSrcTraceMeta;

static NvDsMetaType
modelmux_srctrace_meta_type (void)
{
  static NvDsMetaType t = 0;
  if (t == 0)
    t = nvds_get_user_meta_type ((gchar *) MM_SRCTRACE_META_NAME);
  return t;
}

static gpointer
modelmux_srctrace_copy (gpointer data, gpointer user_data)
{
  NvDsUserMeta *um = (NvDsUserMeta *) data;
  (void) user_data;
  if (!um || !um->user_meta_data)
    return NULL;
  return g_memdup2 (um->user_meta_data, sizeof (ModelMuxSrcTraceMeta));
}

static void
modelmux_srctrace_release (gpointer data, gpointer user_data)
{
  NvDsUserMeta *um = (NvDsUserMeta *) data;
  (void) user_data;
  if (um && um->user_meta_data) {
    g_free (um->user_meta_data);
    um->user_meta_data = NULL;
  }
}

static ModelMuxSrcTraceMeta *
modelmux_find_srctrace_meta (NvDsFrameMeta * fm)
{
  NvDsMetaType ttype = modelmux_srctrace_meta_type ();
  NvDsMetaList *ul;
  if (!fm)
    return NULL;
  for (ul = fm->frame_user_meta_list; ul; ul = ul->next) {
    NvDsUserMeta *um = (NvDsUserMeta *) ul->data;
    if (um->base_meta.meta_type == ttype && um->user_meta_data)
      return (ModelMuxSrcTraceMeta *) um->user_meta_data;
  }
  return NULL;
}

/* ------------------------------------------------------------------ */
/* Frame accounting (MM_DEBUG): per-(source,role) drop detect+localize. */
/* Counters are bumped LOCK-FREE on streaming threads; the epoch/base    */
/* setters run on the main loop (attach paths, under modelmux_bin->lock).          */
/* ------------------------------------------------------------------ */
/* ModelBin -> owning ModelMuxBin (via the pool back-ptr), or NULL. */
static inline ModelMuxBin *
modelmux_ib_of (ModelBin * model_bin)
{
  ModelPool *pool = model_bin ? (ModelPool *) model_bin->pool : NULL;
  return pool ? (ModelMuxBin *) pool->owner : NULL;
}

static gboolean
model_bin_resolve_source_id_fallback (ModelBin * model_bin, guint frame_source_id,
    guint * source_id)
{
  guint i;
  gint mapped;
  if (!source_id)
    return FALSE;
  *source_id = frame_source_id;
  if (!model_bin || !model_bin->slot_stream)
    return FALSE;

  /* Some DeepStream mux/demux paths preserve the global upstream source_id at
   * the nvinfer output; others can expose the per-model mux slot. In the
   * ambiguous case where a value is both a live stream id and another stream's
   * internal slot, prefer the live stream id. Blind slot remapping relabels
   * stream 1 as the stream attached at idx1, which is the Test 8/11/12 failure.
   * NOTE: this runs LOCK-FREE on the nvinfer output thread, so it reads the flat
   * atomic slot_stream mirror -- never idx_to_stream (a GHashTable being
   * inserted/removed under modelmux_bin->lock on the control path is unsafe to read). */
  for (i = 0; i < model_bin->max_streams; i++) {
    if (g_atomic_int_get (&model_bin->slot_stream[i]) == (gint) frame_source_id)
      return TRUE;
  }

  if (frame_source_id < model_bin->max_streams) {
    mapped = g_atomic_int_get (&model_bin->slot_stream[frame_source_id]);
    if (mapped >= 0) {
      *source_id = (guint) mapped;
      return TRUE;
    }
  }
  return FALSE;
}

/* ------------------------------------------------------------------ */
/* Lock-free per-source stream-name mirror. The per-frame provenance   */
/* probes must NOT read modelmux_bin->stream_names (GHashTable reads racing the  */
/* locked attach/detach insert/remove are undefined, and the looked-up */
/* value string is freed by the remove). Publish the name into a flat  */
/* atomically-swapped slot instead; superseded strings are RETIRED and */
/* freed only at bin teardown, after the graph is quiesced (the same   */
/* discipline as ModelBin.retired for OTA engine strings).           */
/* Writers run under modelmux_bin->lock (attach/detach); readers are lock-free.  */
/* ------------------------------------------------------------------ */
static void
modelmux_stream_name_publish (ModelMuxBin * modelmux_bin, guint sid, const gchar * name)
{
  gchar *fresh, *old;
  if (!modelmux_bin || sid >= MM_ACCT_MAX)
    return;
  fresh = name ? g_strdup (name) : NULL;
  old = (gchar *) g_atomic_pointer_get (&modelmux_bin->name_slot[sid]);
  g_atomic_pointer_set (&modelmux_bin->name_slot[sid], fresh);
  if (old) {
    if (!modelmux_bin->retired_names)
      modelmux_bin->retired_names = g_ptr_array_new_with_free_func (g_free);
    g_ptr_array_add (modelmux_bin->retired_names, old);
  }
}

static inline const gchar *
modelmux_stream_name_get (ModelMuxBin * modelmux_bin, guint sid)
{
  if (!modelmux_bin || sid >= MM_ACCT_MAX)
    return NULL;
  return (const gchar *) g_atomic_pointer_get (&modelmux_bin->name_slot[sid]);
}

/* Control-plane wrapper (NOT hot path): friendly source name for a source id,
 * e.g. the camera name a stream/route touched, for notification detail. */
G_GNUC_INTERNAL const gchar *
modelmux_bin_source_name (ModelMuxBin * modelmux_bin, guint sid)
{
  return modelmux_stream_name_get (modelmux_bin, sid);
}

/* zero a source's whole accounting (new stream life; recycled id is clean). */
static void
modelmux_acct_stream_reset (ModelMuxBin * modelmux_bin, guint sid)
{
  if (!modelmux_bin || sid >= MM_ACCT_MAX)
    return;
  memset (&modelmux_bin->acct[sid], 0, sizeof (modelmux_bin->acct[sid]));
}

/* snapshot the epoch for a role at attach, so a role added mid-stream (live shadow
 * enable) starts at gap 0 instead of inheriting the stream's earlier input count.
 * NOT called on a model SWAP (role stays active) -- so a switch's drops stay visible. */
static void
modelmux_acct_role_attach (ModelMuxBin * modelmux_bin, guint sid, gint r)
{
  ModelMuxAcct *a;
  if (!modelmux_bin || sid >= MM_ACCT_MAX || r < 0 || r >= MM_ROLE_SLOTS)
    return;
  a = &modelmux_bin->acct[sid];
  a->role[r].in_base    = g_atomic_int_get (&a->in_count);
  a->role[r].infer_base = g_atomic_int_get (&a->role[r].infer);
  a->role[r].out_base   = g_atomic_int_get (&a->role[r].out);
  a->role[r].warm       = 0;
  a->role[r].floor      = 0;
  a->role[r].over       = 0;
  a->role[r].last_src_fn = -1;          /* MM_FRAME_TRACE: no frame delivered yet */
}

/* Re-learn the steady-depth floor for a role after a model SWAP. A swap changes the
 * model behind the (persistent) queues, which legitimately changes the in-flight depth
 * (a deeper/shallower engine + its mux batching), so the OLD floor would mis-flag the
 * new steady level as "dropped". Reset warm/floor ONLY -- the in/infer/out bases stay
 * put so entered/delivered remain continuous across the switch (a REAL drop at the swap
 * still shows as buffered growing without bound, and is proven exactly by the EOS
 * reconciliation). */
static void
modelmux_acct_role_rewarm (ModelMuxBin * modelmux_bin, guint sid, gint r)
{
  if (!modelmux_bin || sid >= MM_ACCT_MAX || r < 0 || r >= MM_ROLE_SLOTS)
    return;
  modelmux_bin->acct[sid].role[r].warm  = 0;
  modelmux_bin->acct[sid].role[r].floor = 0;
  /* `over` is deliberately NOT reset: rewarm keeps the in/infer/out bases, so a
   * pending real-loss excursion must survive the swap for the final-sample
   * reconciliation to prove it (zeroing it here would hide the loss). */
}

/* perf metrics ONLY (installed only when attach-perf-metric is on): stamp the batch
 * arrival time at the nvinfer SINK, so the src provenance probe can derive the batch's
 * inference latency. Lock-free, single writer. */
static GstPadProbeReturn
modelmux_perf_infer_in_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelBin *model_bin = (ModelBin *) udata;
  (void) pad; (void) info;
  if (model_bin && model_bin->perf)
    modelmux_perf_mark_infer_in (model_bin->perf, g_get_monotonic_time ());
  return GST_PAD_PROBE_OK;
}

/* Provenance for a PASSTHROUGH lane. These frames bypass nvinfer, so they never
 * hit modelmux_inference_provenance_probe(), but they still pass through the combined display mux
 * and still need their original source_id restored after that mux rewrites it to
 * the display-slot index. Attach the minimal per-frame provenance here, before the
 * combined mux re-batches the frame. */
typedef struct { ModelMuxBin *modelmux_bin; guint sid; } ModelMuxPassthruProvCtx;

static GstPadProbeReturn
modelmux_passthru_provenance_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxPassthruProvCtx *c = (ModelMuxPassthruProvCtx *) udata;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  NvDsMetaList *l;
  guint sid;
  const gchar *name = NULL;
  (void) pad;

  if (!c || !bmeta)
    return GST_PAD_PROBE_OK;

  sid = c->sid;
  /* lock-free per-frame path: read the atomic name mirror, never the hash */
  name = modelmux_stream_name_get (c->modelmux_bin, sid);

  for (l = bmeta->frame_meta_list; l; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *) l->data;
    NvDsMetaList *ul;
    NvDsUserMeta *um;
    InferenceProvenanceMeta *pm;
    gboolean have = FALSE;

    for (ul = fm->frame_user_meta_list; ul; ul = ul->next) {
      NvDsUserMeta *existing = (NvDsUserMeta *) ul->data;
      if (existing->base_meta.meta_type == (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE && existing->user_meta_data) {
        have = TRUE;
        break;
      }
    }
    if (have)
      continue;

    um = nvds_acquire_user_meta_from_pool (bmeta);
    if (!um)
      continue;

    pm = g_new0 (InferenceProvenanceMeta, 1);
    pm->magic = NVDS_INFER_PROV_MAGIC;
    {
      ModelMuxSrcTraceMeta *tm = modelmux_find_srctrace_meta (fm);
      pm->source_id = tm ? tm->source_id : sid;
      pm->frame_num = tm ? tm->frame_num : (guint) fm->frame_num;   /* NvDsFrameMeta.frame_num is gint (external API); ours are guint */
    }
    pm->gie_id = 0;
    g_strlcpy (pm->role, "Primary", sizeof (pm->role));
    g_strlcpy (pm->model_name, "passthrough", sizeof (pm->model_name));
    /* ONE display string reaches the bin (sensor_name, falling back to the camera
     * id at attach) -- fill BOTH fields with it until id+name are plumbed apart;
     * a Kafka consumer keying on camera_id still gets a stable value. */
    g_strlcpy (pm->camera_name, name ? name : "src", sizeof (pm->camera_name));
    g_strlcpy (pm->camera_id, name ? name : "src", sizeof (pm->camera_id));
    pm->gpu = (c->modelmux_bin && c->modelmux_bin->config) ? c->modelmux_bin->config->gpu : 0;   /* no inference: pipeline device */
    pm->batch = 0;                 /* passthrough: no inference, no batch (explicit) */

    um->user_meta_data = pm;
    um->base_meta.meta_type = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;
    um->base_meta.copy_func = (NvDsMetaCopyFunc) modelmux_provenance_copy;
    um->base_meta.release_func = (NvDsMetaReleaseFunc) modelmux_provenance_release;
    nvds_add_user_meta_to_frame (fm, um);
  }

  return GST_PAD_PROBE_OK;
}

/* attach-perf-metric: THROUGHPUT record for a PASSTHROUGH lane (no nvinfer, so
 * no ModelMuxPerf bin). One buffer == one frame on the demuxed single-stream bypass
 * queue, so each buffer bumps that source's throughput counter (lock-free; the
 * one passthrough lane is the single writer). lat=0 => rate only, no latency
 * sample (there is no inference to time). The ctx is freed by the probe's
 * GDestroyNotify when the lane is torn down, so it never dangles; thru_perf[]
 * lives for the bin's lifetime. */
typedef struct { ModelMuxBin *modelmux_bin; guint sid; } ModelMuxThruProbeCtx;

static GstPadProbeReturn
modelmux_perf_passthru_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxThruProbeCtx *c = (ModelMuxThruProbeCtx *) udata;
  (void) pad; (void) info;
  if (c && c->modelmux_bin && c->sid < MM_ACCT_MAX)
    modelmux_perf_counter_record (&c->modelmux_bin->thru_perf[c->sid], 0);
  return GST_PAD_PROBE_OK;
}

/* Tick every perf-enabled bin (+ shard chain) in a pool. Caller holds modelmux_bin->lock. */
static void
modelmux_perf_tick_pool (ModelPool * pool, gint64 now)
{
  GHashTableIter it;
  gpointer k, v;
  if (!pool || !pool->models)
    return;
  g_hash_table_iter_init (&it, pool->models);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelBin *model_bin;
    for (model_bin = (ModelBin *) v; model_bin; model_bin = (ModelBin *) model_bin->next_shard)
      if (model_bin->perf)
        modelmux_perf_tick (model_bin->perf, now);
  }
}

/* Periodic snapshot: recompute fps/latency for every model bin's aggregate + per-source
 * counters. Under modelmux_bin->lock (serialises against status reads + pool mutations). Limbo models
 * (modelmux_bin->limbo_models) carry no streams, so there is nothing to report there. */
static gboolean
modelmux_perf_timer_cb (gpointer udata)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) udata;
  gint64 now = g_get_monotonic_time ();
  gint i;
  g_mutex_lock (&modelmux_bin->lock);
  /* re-check under the lock: a dispatch that started just before _free()'s
   * g_source_remove must not touch pools _free is about to destroy (the free
   * path takes/releases modelmux_bin->lock as a barrier after setting shutting_down).
   * Return CONTINUE (skip the work only): _free unconditionally removes
   * perf_timer_id right after, and self-removing here would turn that into a
   * stale-id g_source_remove -> GLib CRITICAL in the teardown window. */
  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    g_mutex_unlock (&modelmux_bin->lock);
    return G_SOURCE_CONTINUE;
  }
  modelmux_perf_tick_pool (modelmux_bin->primary_pool, now);
  modelmux_perf_tick_pool (modelmux_bin->shadow_pool, now);
  /* recompute per-source PASSTHROUGH throughput over the window (skip untouched
   * sources cheaply). Same window cadence as the model/source counters. */
  for (i = 0; i < MM_ACCT_MAX; i++) {
    ModelMuxPerfCounter *c = &modelmux_bin->thru_perf[i];
    if (c->frames == 0 && c->last_ts_us == 0)
      continue;
    modelmux_perf_counter_tick (c, now);
  }
  g_mutex_unlock (&modelmux_bin->lock);
  return G_SOURCE_CONTINUE;
}

/* Per-frame probe on a ModelBin's nvinfer src pad: attach one InferenceProvenanceMeta to
 * every frame so the (stream, model, role) that produced these detections is
 * unambiguous downstream. Role is slot-derived (modelmux_resolve_role). */
static GstPadProbeReturn
modelmux_inference_provenance_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelBin *model_bin = (ModelBin *) udata;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta;
  NvDsMetaList *l;
  NvDsMetaType mtype = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;
  ModelMuxBin *modelmux_bin = modelmux_ib_of (model_bin);          /* for MM_DEBUG frame accounting */
  gint racct = modelmux_role_idx (modelmux_role_get (model_bin));   /* atomic: role may flip live */
  /* perf metrics (opt-in): one cheap bool when off. Sample the batch latency ONCE here --
   * it is identical for every frame of this batch (computed from the single sink t_enter). */
  gboolean perf = model_bin->perf_on && model_bin->perf;
  gint64 perf_now = perf ? g_get_monotonic_time () : 0;

  (void) pad;
  if (!buffer)
    return GST_PAD_PROBE_OK;
  bmeta = gst_buffer_get_nvds_batch_meta (buffer);
  if (!bmeta)
    return GST_PAD_PROBE_OK;

  for (l = bmeta->frame_meta_list; l != NULL; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *) l->data;
    guint frame_source_id = fm->source_id;     /* value after per-model mux/nvinfer */
    guint source_id = frame_source_id;
    ModelMuxSrcTraceMeta *tm = modelmux_find_srctrace_meta (fm);

    /* The upstream source id is a per-frame fact, not a model-bin slot fact.
     * DeepStream does not give a stable contract here: depending on the mux/demux
     * path, fm->source_id at nvinfer src can be the original global source id or
     * an internal per-model slot. The input trace is stamped before any internal
     * re-batching, so it is the authoritative identity. Use the slot map only as
     * a no-trace fallback for old/in-flight buffers. */
    if (tm) {
      source_id = tm->source_id;
    } else if (!model_bin_resolve_source_id_fallback (model_bin, frame_source_id,
            &source_id)) {
      static gint missing_src_logged = 0;
      if (g_atomic_int_get (&missing_src_logged) < 8) {
        g_atomic_int_inc (&missing_src_logged);
        MM_WARN ("ModelBin %s/%s: no source trace or idx->stream mapping for "
            "nvinfer source_id=%u; using it as global source_id",
            modelmux_role_get (model_bin), model_bin->name, frame_source_id);
      }
    }

    /* INFERRED count: this frame passed this role's nvinfer (lock-free, MM_DEBUG). */
    if (modelmux_bin && modelmux_bin->log_enabled && source_id < MM_ACCT_MAX)
      g_atomic_int_inc (&modelmux_bin->acct[source_id].role[racct].infer);

    /* perf metrics (opt-in): record this frame into the bin aggregate + its per-source
     * counter (lock-free). `perf_now` is the single batch timestamp sampled above. */
    if (perf)
      modelmux_perf_mark_infer_out (model_bin->perf, perf_now, (gint) source_id);

    /* lock-free per-frame path: read the atomic name mirror, never the hash */
    const gchar *name = modelmux_stream_name_get (modelmux_bin, source_id);
    NvDsUserMeta *um = nvds_acquire_user_meta_from_pool (bmeta);
    InferenceProvenanceMeta *pm;

    if (!um)
      continue;
    pm = g_new0 (InferenceProvenanceMeta, 1);
    pm->magic = NVDS_INFER_PROV_MAGIC;
    pm->source_id = source_id;
    pm->gie_id = model_bin->unique_id;
    g_strlcpy (pm->role, modelmux_resolve_role (model_bin, source_id), sizeof (pm->role));
    g_strlcpy (pm->model_name, model_bin->name ? model_bin->name : "", sizeof (pm->model_name));
    g_strlcpy (pm->model_version, model_version_str (model_bin), sizeof (pm->model_version));
    /* ONE display string reaches the bin (sensor_name, falling back to the camera
     * id at attach) -- fill BOTH fields with it until id+name are plumbed apart;
     * a Kafka consumer keying on camera_id still gets a stable value. */
    g_strlcpy (pm->camera_name, name ? name : "src", sizeof (pm->camera_name));
    g_strlcpy (pm->camera_id, name ? name : "src", sizeof (pm->camera_id));
    {
      /* atomic: may be hot-swapped live by an in-place OTA reload */
      const gchar *eng = (const gchar *) g_atomic_pointer_get (&model_bin->engine);
      g_strlcpy (pm->engine, eng ? eng : "", sizeof (pm->engine));
    }
    pm->frame_num = tm ? tm->frame_num : (guint) fm->frame_num;   /* NvDsFrameMeta.frame_num is gint (external API); ours are guint */
    /* the device this frame was ACTUALLY inferred on (plain gint read of the
     * shard's placement -- written only pre-warm, stable on this hot path) */
    pm->gpu = (guint) (model_bin->conv_gpu >= 0 ? model_bin->conv_gpu : 0);
    pm->batch = model_bin->max_streams;   /* the shard's configured batch/capacity */

    um->user_meta_data = pm;
    um->base_meta.meta_type = mtype;
    um->base_meta.copy_func = (NvDsMetaCopyFunc) modelmux_provenance_copy;
    um->base_meta.release_func = (NvDsMetaReleaseFunc) modelmux_provenance_release;
    nvds_add_user_meta_to_frame (fm, um);

    /* FRAME-NUMBER TRACE verification (opt-in): try to recover the ORIGINAL
     * (source_id, frame_num) stamped at the bin input. This pad is AFTER the
     * per-model nvstreammux, so fm->frame_num/source_id here are already the
     * mux's re-numbered values -- comparing them to the recovered originals
     * proves (a) the trace meta survived the re-batch and (b) how the mux
     * re-numbers. Logs only the first few frames to avoid spam. No-op unless
     * MM_FRAME_TRACE=1. */
    if (modelmux_frametrace_enabled ()) {
      /* Carry the recovered ORIGINAL frame_num on the provenance meta so it rides
       * (via the proven provenance copy_func) through out_mux to the output, where
       * the per-source contiguity gap check reads it. Falls back to the (re-numbered)
       * nvinfer frame_num if the trace meta is somehow absent. */
      {
        static gint traced = 0;            /* survival proof: log only the first few */
        if (g_atomic_int_get (&traced) < 8) {
          g_atomic_int_inc (&traced);
          if (tm)
            MM_INFO ("FRAMETRACE [%s/%s] survived=YES  nvinfer(source_id=%u "
                "frame_num=%u)  ORIGINAL(source_id=%u frame_num=%u)",
                modelmux_role_get (model_bin), model_bin->name, frame_source_id, fm->frame_num,
                tm->source_id, tm->frame_num);
          else
            MM_WARN ("FRAMETRACE [%s/%s] survived=NO  nvinfer(source_id=%u "
                "frame_num=%u)  -- input trace meta did NOT survive the per-model "
                "nvstreammux re-batch (original frame_num unrecoverable here)",
                modelmux_role_get (model_bin), model_bin->name, frame_source_id, fm->frame_num);
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* drops EOS so dynamic source add/remove never latches the persistent chain */
static GstPadProbeReturn modelmux_drop_eos_probe (GstPad * pad,
    GstPadProbeInfo * info, gpointer udata);

/* shard auto-compaction: schedule a debounced pass that packs a model's under-full
 * shards back into fewer shards as streams drain (definition further below). */
static void modelmux_schedule_compact (ModelMuxBin * modelmux_bin);
static void modelmux_schedule_compact_idle (ModelMuxBin * modelmux_bin);

static GstElement *
create_gst_element (const gchar * factory, const gchar * name)
{
  GstElement *e = gst_element_factory_make (factory, name);
  if (!e)
    MM_ERR ("failed to create element '%s' (factory '%s')", name, factory);
  return e;
}

static guint
modelmux_branch_serial_next (void)
{
  static gint serial = 0;
  return (guint) g_atomic_int_add (&serial, 1) + 1;
}

/* Set an integer property ONLY if the element's class actually exposes it. nvstreammux
 * has TWO implementations selected at runtime by USE_NEW_NVSTREAMMUX: the legacy one takes
 * width/height/batched-push-timeout/live-source as element properties, while the new one
 * (recommended for dynamic add/remove -- it does NOT spin its src push-loop to a
 * GST_IS_BUFFER critical when a source set drains to zero) takes those via its config-file
 * and does NOT expose them as properties. Blindly g_object_set'ing a missing property emits
 * a GLib-GObject-CRITICAL. Guarding with find_property makes the plugin work cleanly under
 * BOTH muxers -- the standard defensive pattern when an element's implementation can vary. */
static void
set_prop_if_exists (GstElement * el, const gchar * prop, gint val)
{
  if (el && g_object_class_find_property (G_OBJECT_GET_CLASS (el), prop))
    g_object_set (G_OBJECT (el), prop, val, NULL);
}

/* Link two element pads given names; returns TRUE on success. */
static gboolean
modelmux_link_pads (GstPad * src, GstPad * sink)
{
  GstPadLinkReturn r;
  if (!src || !sink)
    return FALSE;
  r = gst_pad_link (src, sink);
  if (r != GST_PAD_LINK_OK) {
    MM_ERR ("pad link failed: %s -> %s (%d)",
        GST_PAD_NAME (src), GST_PAD_NAME (sink), r);
    return FALSE;
  }
  return TRUE;
}

/* NULL + remove an element from its container bin (no-op on NULL). Used to undo a partially
 * wired lane element on an error path, or to drop a per-lane element at teardown. */
static void
modelmux_drop_from_bin (GstElement * container, GstElement * el)
{
  if (!el)
    return;
  gst_element_set_state (el, GST_STATE_NULL);
  gst_bin_remove (GST_BIN (container), el);
}

/* ============================ per-lane BUFFER-COPY policy + wiring ============================
 * A "copy lane" splices an nvvideoconvert (disable-passthrough=1) BEFORE the lane's queue so the
 * queue holds the lane's OWN buffers and the upstream decoder buffer is released at fan-out --
 * this decouples a SLOW lane from the decoder when per-model fps differ. Which roles copy is the
 * [multimodel] buffer-copy-mode policy (none|primary|shadow|both). These three small helpers are
 * the single source of truth, so the lane-attach code stays "make converter (maybe) -> link head".
 */

/* Policy: does THIS role's lane deep-copy? Only an EXACT Primary/Shadow lane may copy --
 * passthrough/limbo never do. NULL config or unknown role => no copy. */
static gboolean
modelmux_lane_copy_wanted (const ModelMuxConfig * config, const gchar * role)
{
  if (!config || !role)
    return FALSE;
  if (g_strcmp0 (role, "Primary") == 0) return config->copy_primary_buffer;
  if (g_strcmp0 (role, "Shadow")  == 0) return config->copy_shadow_buffer;
  return FALSE;
}

/* ---- per-lane buffer verification (gated by MM_DEBUG, like the taps/tables) ------------------
 * When MM_DEBUG is on, EVERY lane gets two probes that keep a tiny per-lane RECORD: the buffer
 * (+ its NvBufSurface device pointer) at the LANE HEAD (tee src) and the buffer at the QUEUE INPUT
 * (in_q sink). Comparing them validates the LANE AS BUILT (tee -> [nvvideoconvert] -> queue), not
 * the converter in isolation: a COPY lane yields a NEW buffer/memory (COPIED -> the decoder buffer
 * is freed at fan-out); a passthrough lane forwards the SAME buffer (SAME -> zero-copy, shares the
 * decoder buffer). Logged ONCE per lane (greppable). Reuses MM_DEBUG -- no separate flag. Off =>
 * no probes installed, zero hot-path cost. */
/* device data pointer of a (batched) NvBufSurface buffer, copied out before unmap (NULL on fail) */
static gpointer
modelmux_buf_surface_dataptr (GstBuffer * b)
{
  GstMapInfo m;
  gpointer d = NULL;
  if (b && gst_buffer_map (b, &m, GST_MAP_READ)) {
    NvBufSurface *s = (NvBufSurface *) (gpointer) m.data;
    if (s && s->surfaceList && s->numFilled > 0)
      d = s->surfaceList[0].dataPtr;
    gst_buffer_unmap (b, &m);
  }
  return d;
}

/* one record per copy lane: original (pre-convert) buffer+memory vs post-convert buffer */
typedef struct {
  gchar    role[16];
  guint    stream_id;
  gpointer in_buf;       /* GstBuffer* at the converter SINK (the original)  */
  gpointer in_data;      /* its NvBufSurface dataPtr                          */
  gboolean have_in;
  gboolean logged;       /* emit the verdict once, then go idle (no per-frame spam) */
} ModelMuxCopyVerify;

static GstPadProbeReturn
modelmux_copy_verify_in (GstPad * pad, GstPadProbeInfo * info, gpointer u)
{
  ModelMuxCopyVerify *v = (ModelMuxCopyVerify *) u;
  (void) pad;
  if (!v->logged) {
    GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER (info);
    v->in_buf = b;
    v->in_data = modelmux_buf_surface_dataptr (b);
    v->have_in = TRUE;
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
modelmux_copy_verify_out (GstPad * pad, GstPadProbeInfo * info, gpointer u)
{
  ModelMuxCopyVerify *v = (ModelMuxCopyVerify *) u;
  (void) pad;
  if (!v->logged && v->have_in) {
    GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER (info);
    gpointer qin_data = modelmux_buf_surface_dataptr (b);
    /* a NEW GstBuffer, OR DIFFERENT underlying NvBufSurface memory between the lane head (tee src)
     * and the queue input => the lane deep-COPIED; identical => passthrough (SAME buffer). */
    gboolean copied = ((gpointer) b != v->in_buf) ||
                      (v->in_data && qin_data && qin_data != v->in_data);
    MM_INFO ("    - BUFFER-COPY-VERIFY[%s]=%s  stream %u  tee_buf=%p qin_buf=%p  tee_data=%p qin_data=%p",
        v->role, copied ? "COPIED" : "SAME", v->stream_id,
        v->in_buf, (gpointer) b, v->in_data, qin_data);
    v->logged = TRUE;
  }
  return GST_PAD_PROBE_OK;
}

static void
modelmux_pad_probe_ref_clear (GstPad ** pad, gulong * probe_id)
{
  if (pad && *pad && probe_id && *probe_id)
    gst_pad_remove_probe (*pad, *probe_id);
  if (probe_id)
    *probe_id = 0;
  if (pad && *pad) {
    gst_object_unref (*pad);
    *pad = NULL;
  }
}

/* attach the verify probes to a LANE: capture at the lane HEAD (tee src = in_pad), compare at the
 * QUEUE INPUT (in_q sink). Debug-only; no-op when disabled. The caller owns the returned probe
 * refs and context and removes them explicitly before releasing this dynamic branch. That avoids
 * relying on pad/element disposal to tear down probes while an add/remove race is still draining
 * the just-created source. */
static void
modelmux_lane_install_verify (GstPad * in_pad, GstElement * in_q, const gchar * role, guint stream_id,
    GstPad ** head_pad_out, gulong * head_probe_out,
    GstPad ** queue_pad_out, gulong * queue_probe_out,
    gpointer * ctx_out)
{
  GstPad *qsink;
  ModelMuxCopyVerify *v;

  if (head_pad_out) *head_pad_out = NULL;
  if (head_probe_out) *head_probe_out = 0;
  if (queue_pad_out) *queue_pad_out = NULL;
  if (queue_probe_out) *queue_probe_out = 0;
  if (ctx_out) *ctx_out = NULL;

  if (!in_pad || !in_q || !modelmux_debug_enabled ())
    return;
  qsink = gst_element_get_static_pad (in_q, "sink");
  if (qsink) {
    gulong head_probe, queue_probe;
    v = g_new0 (ModelMuxCopyVerify, 1);
    g_strlcpy (v->role, role ? role : "?", sizeof (v->role));
    v->stream_id = stream_id;
    head_probe = gst_pad_add_probe (in_pad, GST_PAD_PROBE_TYPE_BUFFER,
        modelmux_copy_verify_in, v, NULL);
    queue_probe = gst_pad_add_probe (qsink, GST_PAD_PROBE_TYPE_BUFFER,
        modelmux_copy_verify_out, v, NULL);
    if (head_probe && queue_probe) {
      if (head_pad_out) *head_pad_out = gst_object_ref (in_pad);
      if (head_probe_out) *head_probe_out = head_probe;
      if (queue_pad_out) *queue_pad_out = gst_object_ref (qsink);
      if (queue_probe_out) *queue_probe_out = queue_probe;
      if (ctx_out) *ctx_out = v;
    } else {
      if (head_probe)
        gst_pad_remove_probe (in_pad, head_probe);
      if (queue_probe)
        gst_pad_remove_probe (qsink, queue_probe);
      g_free (v);
    }
    gst_object_unref (qsink);
  }
}

/* Build the OPTIONAL per-lane copy converter and add it to `container` in the running state.
 * Returns NULL when no copy is wanted for `role`, OR (GRACEFUL) when the element can't be made --
 * the caller then wires a plain zero-copy lane, so the feature can never break a stream. The copy
 * pool depth is tunable via config->copy_buffer_pool_size (0 = nvvideoconvert default). */
static GstElement *
modelmux_lane_make_copy_conv (GstElement * container, const gchar * role,
    const gchar * model_name, guint stream_id, guint branch_serial,
    const ModelMuxConfig * config)
{
  GstElement *conv;
  gchar nm[96];

  if (!modelmux_lane_copy_wanted (config, role))
    return NULL;

  g_snprintf (nm, sizeof (nm), "%s-cvt-%s-%u-%u", role, model_name,
      stream_id, branch_serial);
  conv = gst_element_factory_make ("nvvideoconvert", nm);
  if (!conv) {
    MM_WARN ("ModelPool %s: 'nvvideoconvert' unavailable -- stream %u model '%s' runs ZERO-COPY",
        role, stream_id, model_name);
    return NULL;
  }
  set_prop_if_exists (conv, "disable-passthrough", 1);          /* force the deep copy */
  if (config && config->copy_buffer_pool_size > 0)                       /* tunable copy-pool depth */
    set_prop_if_exists (conv, "output-buffers", (gint) config->copy_buffer_pool_size);
  gst_bin_add (GST_BIN (container), conv);
  gst_element_sync_state_with_parent (conv);
  return conv;
}

/* Link the HEAD of a lane: in_pad -> [conv ->] q_in_sink. With a converter the chain is
 * in_pad -> conv.sink, conv.src -> q_in_sink; without one it's the plain in_pad -> q_in_sink. */
static gboolean
modelmux_lane_link_head (GstPad * in_pad, GstElement * conv, GstPad * q_in_sink)
{
  GstPad *cs, *cr;
  gboolean ok;
  if (!conv)
    return modelmux_link_pads (in_pad, q_in_sink);
  cs = gst_element_get_static_pad (conv, "sink");
  cr = gst_element_get_static_pad (conv, "src");
  ok = cs && cr && modelmux_link_pads (in_pad, cs) && modelmux_link_pads (cr, q_in_sink);
  if (cs) gst_object_unref (cs);
  if (cr) gst_object_unref (cr);
  return ok;
}

/* ================================================================== */
/* ModelBin                                                            */
/* ================================================================== */

/* nvinfer 'model-updated' signal: fired by nvinfer whenever it (re)loads an engine
 * context -- on the initial warm-up, on the idle-reset cycle when the last stream
 * leaves (engine kept warm), AND on an OTA engine hot-swap. The wording
 * is therefore NEUTRAL (not "OTA"): for a real OTA the preceding ">> MODEL UPDATE"
 * / "OTA update accepted" log already identifies it. status==0 (NVDSINFER_SUCCESS)
 * = new context live; non-zero = load failed and the PREVIOUS engine keeps serving
 * (e.g. an OTA topology mismatch). The bin's ModelStatus is unchanged. */
static void modelmux_bin_retire_str (ModelBin * model_bin, gchar * old);
/* fwd decls (defined below): the identity-rename rollback is marshalled from
 * nvinfer's confirm thread to the main loop via the tracked-source machinery. */
static gboolean modelmux_bin_rename_model_locked (ModelMuxBin * modelmux_bin,
    const gchar * old_key, const gchar * new_key);
static guint modelmux_lookup_model_all_locked (ModelMuxBin * modelmux_bin,
    const gchar * name, ModelBin * out[3]);
static guint modelmux_tracked_idle_add (ModelMuxBin * modelmux_bin, GSourceFunc func,
    gpointer data, GDestroyNotify destroy, guint * slot);
static guint modelmux_tracked_timeout_add (ModelMuxBin * modelmux_bin, guint interval_ms,
    GSourceFunc func, gpointer data, GDestroyNotify destroy, guint * slot);

/* After a failed update's identity rename was rolled back, any configured role
 * DEFAULT that the update had moved to the rejected checkpoint must move back
 * too, or every later stream-add would resolve to the now-dead key.
 * rejected_key = the failed update's key (rolled back FROM), restored_key = the
 * serving checkpoint key. Caller must NOT hold modelmux_bin->lock: set_default
 * self-acquires it. */
static void
modelmux_ota_restore_default_refs (ModelMuxBin * modelmux_bin, const gchar * rejected_key,
    const gchar * restored_key)
{
  gchar *name = NULL, *from_ver = NULL, *to_ver = NULL;
  const gchar *dv;
  if (!modelmux_bin->config)
    return;
  model_key_split (rejected_key, &name, &from_ver);
  model_key_split (restored_key, NULL, &to_ver);
  {
    /* SNAPSHOT the ref identities under the bin lock (the default writers mutate
     * these strings under it; reading unlocked here worked only by the main-loop
     * convention -- take the lock, it is cheap and removes the assumption). The
     * set_default calls self-lock, so they run on the COPIES, after the unlock. */
    gchar *dp_name, *dp_ver, *ds_name, *ds_ver;
    g_mutex_lock (&modelmux_bin->lock);
    dp_name = g_strdup (modelmux_bin->primary_pool->default_ref.name);
    dp_ver = g_strdup (modelmux_bin->primary_pool->default_ref.version);
    ds_name = g_strdup (modelmux_bin->shadow_pool->default_ref.name);
    ds_ver = g_strdup (modelmux_bin->shadow_pool->default_ref.version);
    g_mutex_unlock (&modelmux_bin->lock);
    dv = dp_ver ? dp_ver : MM_MODEL_VERSION_DEFAULT;
    if (dp_name && name && g_strcmp0 (dp_name, name) == 0 &&
        g_strcmp0 (dv, from_ver) == 0)
      modelmux_bin_set_default (modelmux_bin, FALSE, name, to_ver, -1);
    dv = ds_ver ? ds_ver : MM_MODEL_VERSION_DEFAULT;
    if (ds_name && name && g_strcmp0 (ds_name, name) == 0 &&
        g_strcmp0 (dv, from_ver) == 0)
      modelmux_bin_set_default (modelmux_bin, TRUE, name, to_ver, -1);
    g_free (dp_name);
    g_free (dp_ver);
    g_free (ds_name);
    g_free (ds_ver);
  }
  g_free (name);
  g_free (from_ver);
  g_free (to_ver);
}

/* Deferred (main-loop) rollback of a FAILED OTA's identity rename. The nvinfer
 * 'model-updated' confirm runs on nvinfer's own thread: taking modelmux_bin->lock there
 * could deadlock against the main-loop reload, which holds modelmux_bin->lock across the
 * g_object_set that nvinfer's update machinery serializes on (modelmux_bin->lock ->
 * nvinfer-internal vs nvinfer-internal -> modelmux_bin->lock inversion). So the rename
 * back is marshalled here, mirroring modelmux_schedule_model_bin_free. Idempotent:
 * once the key was already renamed back (multi-shard: every failing shard
 * schedules one) or the update never renamed, the lookup misses -> no-op. */
typedef struct
{
  ModelMuxBin *modelmux_bin;
  gchar *from_key;            /* the failed update's key ("name@TO")       */
  gchar *to_key;              /* the serving checkpoint key ("name@FROM")  */
} ModelMuxOtaRenameCtx;

static void
modelmux_ota_rename_ctx_free (gpointer data)
{
  ModelMuxOtaRenameCtx *ctx = (ModelMuxOtaRenameCtx *) data;
  g_free (ctx->from_key);
  g_free (ctx->to_key);
  g_free (ctx);
}

static gboolean
modelmux_ota_rename_rollback_main (gpointer data)
{
  ModelMuxOtaRenameCtx *ctx = (ModelMuxOtaRenameCtx *) data;
  ModelMuxBin *modelmux_bin = ctx->modelmux_bin;
  gboolean renamed;

  g_mutex_lock (&modelmux_bin->lock);
  renamed = modelmux_bin_rename_model_locked (modelmux_bin, ctx->from_key, ctx->to_key);
  if (renamed)
    MM_ERR ("MultiModelBin: identity '%s' renamed BACK to '%s' -- the rejected "
        "OTA engine never served (UPDATE_ENGINE_REJECTED)", ctx->from_key,
        ctx->to_key);
  g_mutex_unlock (&modelmux_bin->lock);
  /* the update also moved a configured role default that named the FROM
   * checkpoint -- move it back too, or every later stream-add would resolve
   * to the now-dead key. Outside modelmux_bin->lock: set_default self-acquires it. */
  if (renamed)
    modelmux_ota_restore_default_refs (modelmux_bin, ctx->from_key, ctx->to_key);
  /* control-plane hook: let the owner re-sync its key-addressed state now that
   * the identity moved back. NO locks held at the call moment (modelmux_bin->lock was
   * released above; set_default self-acquired + released it). from_key = the
   * failed update's key (rolled back FROM), to_key = the restored serving key. */
  if (renamed && modelmux_bin->ota_rollback_cb)
    modelmux_bin->ota_rollback_cb (modelmux_bin->ota_rollback_owner, ctx->from_key, ctx->to_key);
  modelmux_ota_rename_ctx_free (ctx);      /* dispatched -> the notify skips destroy */
  return G_SOURCE_REMOVE;
}

/* ------------------------------------------------------------------ *
 * model/update GROUP TRANSACTION.
 *
 * A multi-shard / multi-twin update triggers nvinfer's hot-swap on EVERY shard
 * and renames the identity name@FROM -> name@TO immediately (one coherent view
 * for CAS/status during the confirm window). The per-shard confirms then land
 * asynchronously on each nvinfer's own thread. Deciding per shard (the old
 * behaviour) let ONE shard keep serving the NEW engine while another shard's
 * failure renamed the identity back -- a silent split brain. So every grouped
 * update carries ONE ModelMuxOtaGroup: each shard records its outcome in it, and the
 * LAST confirm resolves the whole group atomically on the MAIN loop:
 *   - ALL succeeded -> COMMIT: the trigger-time rename is kept, every shard's
 *     rollback stash is dropped.
 *   - ANY failed    -> ROLLBACK: the identity is renamed back, version/engine
 *     provenance is restored on ALL shards, and every shard that had CONFIRMED
 *     SUCCESS gets a COMPENSATING reload of its stashed previous engine. A
 *     compensating reload is flagged (ota_compensating) so its confirm never
 *     re-enters the group logic -- it only logs. modelmux_bin->ota_rollback_cb fires ONCE.
 *
 * Locking: confirms touch only group->lock (+ the shard's own prov_lock) --
 * NEVER modelmux_bin->lock (nvinfer-thread inversion, see modelmux_ota_rename_rollback_main).
 * The resolution is marshalled to the main loop, where modelmux_bin->lock re-resolves the
 * shards BY KEY (no stored shard pointers -> a shard freed meanwhile is simply
 * not found).
 *
 * ACCEPTED RESIDUAL (matches the pre-group behaviour): a shard whose confirm
 * NEVER arrives leaves the group unresolved forever -- its shards stay
 * ota_await, so a later update gets UPDATE_IN_FLIGHT on retry, and the group
 * ctx is only reclaimed at bin teardown.
 * ------------------------------------------------------------------ */
typedef struct
{
  ModelMuxBin *modelmux_bin;
  gchar *live_key;            /* key the group is live under while awaiting
                               * confirms (the post-rename key when a rename
                               * rode the update)                              */
  gchar *from_key;            /* pre-update identity "name@FROM" (NULL = the
                               * update carried no rename)                     */
  gchar *to_key;              /* post-update identity "name@TO" (NULL = none)  */
  GMutex lock;                /* guards the counters below: confirms land
                               * concurrently on the shards' nvinfer threads   */
  guint expected;             /* shards actually TRIGGERED (final once sealed) */
  guint confirmed;            /* confirms landed (success + failure)           */
  guint failed;               /* failed confirms                               */
  gboolean sealed;            /* trigger loop done; `expected` is final. Gates
                               * resolution so a synchronous confirm during the
                               * trigger loop cannot resolve a half-built group */
} ModelMuxOtaGroup;

static ModelMuxOtaGroup *
modelmux_ota_group_new (ModelMuxBin * modelmux_bin)
{
  ModelMuxOtaGroup *group = g_new0 (ModelMuxOtaGroup, 1);
  group->modelmux_bin = modelmux_bin;
  g_mutex_init (&group->lock);
  return group;
}

static void
modelmux_ota_group_free (gpointer data)
{
  ModelMuxOtaGroup *group = (ModelMuxOtaGroup *) data;
  g_mutex_clear (&group->lock);
  g_free (group->live_key);
  g_free (group->from_key);
  g_free (group->to_key);
  g_free (group);
}

/* Resolve a fully-confirmed group: commit or rollback, on the MAIN loop (this
 * takes modelmux_bin->lock -- never callable on nvinfer's confirm thread). Scheduled
 * exactly once, by whichever of {last confirm, trigger-loop seal} sees the
 * count complete. Owns + frees the group. */
static gboolean
modelmux_ota_group_resolve_main (gpointer data)
{
  ModelMuxOtaGroup *group = (ModelMuxOtaGroup *) data;
  ModelMuxBin *modelmux_bin = group->modelmux_bin;
  ModelBin *bins[3], *s;
  guint nbins, b, nshards = 0, ncomp = 0;
  gboolean commit, renamed_back = FALSE;

  /* every confirm has landed (this idle is only scheduled on the LAST one), so
   * the counters are settled -- no group->lock needed from here on. */
  commit = (group->failed == 0);

  g_mutex_lock (&modelmux_bin->lock);
  /* re-resolve the group's shards BY KEY: any bin/shard freed while the group
   * was in flight is simply absent; any bin loaded meanwhile under the same key
   * has ota_group==NULL and is skipped by the per-shard guard below. */
  nbins = modelmux_lookup_model_all_locked (modelmux_bin, group->live_key, bins);
  for (b = 0; b < nbins; b++) {
    for (s = bins[b]; s; s = (ModelBin *) s->next_shard) {
      gchar *oldv, *olde, *oldc;
      gboolean ok;
      g_mutex_lock (&s->prov_lock);
      if (s->ota_group != (gpointer) group) {
        g_mutex_unlock (&s->prov_lock);
        continue;                       /* not a member of THIS update */
      }
      s->ota_group = NULL;
      ok = s->ota_ok;
      s->ota_ok = FALSE;
      oldv = s->ota_old_version;        /* steal the rollback stash */
      olde = s->ota_old_engine;
      oldc = s->ota_old_config;
      s->ota_old_version = NULL;
      s->ota_old_engine = NULL;
      s->ota_old_config = NULL;
      g_free (s->ota_old_key);          /* rename rollback is group-level now */
      g_free (s->ota_new_key);
      s->ota_old_key = NULL;
      s->ota_new_key = NULL;
      s->ota_ib = NULL;
      nshards++;
      if (commit) {
        /* COMMIT: the identity moved at trigger time and STAYS (status/CAS were
         * coherent throughout the window); just drop the rollback stash. */
        s->ota_await = FALSE;
        g_mutex_unlock (&s->prov_lock);
        g_free (oldv);
        g_free (olde);
        g_free (oldc);
        continue;
      }
      /* ROLLBACK. A shard that CONFIRMED SUCCESS is serving the rejected NEW
       * engine/config: re-trigger a COMPENSATING reload of the stashed previous
       * one. Flag it BEFORE the property set (nvinfer can confirm
       * synchronously) so its confirm only logs and never re-enters the group
       * logic; ota_await stays armed until that confirm lands, keeping the
       * UPDATE_IN_FLIGHT admission shut meanwhile. A shard that FAILED already
       * kept the previous engine serving -- only its provenance rolls back. */
      if (ok && (olde || oldc) && s->infer) {
        s->ota_compensating = TRUE;     /* ota_await stays TRUE */
      } else {
        s->ota_await = FALSE;
        ok = FALSE;                     /* nothing to compensate */
      }
      g_mutex_unlock (&s->prov_lock);
      /* a by-config OTA (oldc != NULL) rolls the recorded config + its cached
       * device pin back, whatever the shard's own outcome -- the scheduler and
       * effective-gpu must keep describing what actually serves. */
      if (oldc) {
        g_free (s->config_file);
        s->config_file = g_strdup (oldc);
        s->cfg_gpu = modelmux_config_file_gpu (oldc);
      }
      /* restore the provenance strings the lock-free probe reads: atomic-swap
       * the previous ones back in, RETIRE the rejected-update strings. */
      if (oldv) {
        gchar *bad = (gchar *) g_atomic_pointer_get (&s->version);
        g_atomic_pointer_set (&s->version, oldv);
        modelmux_bin_retire_str (s, bad);
        oldv = NULL;                    /* transferred */
      }
      if (olde) {
        gchar *bad = (gchar *) g_atomic_pointer_get (&s->engine);
        g_atomic_pointer_set (&s->engine, g_strdup (olde));
        modelmux_bin_retire_str (s, bad);
      }
      if (ok) {
        /* compensate with what the OTA actually changed: config-reload OTAs
         * re-set config-file-path (full reload of the previous model);
         * engine-only OTAs re-set model-engine-file. */
        if (oldc)
          g_object_set (G_OBJECT (s->infer), "config-file-path", oldc, NULL);
        else
          g_object_set (G_OBJECT (s->infer), "model-engine-file", olde, NULL);
        ncomp++;
        MM_INFO ("      - nvmodelbin [%s/%s]  group rollback: COMPENSATING "
            "reload of the previous %s '%s' TRIGGERED (shard #%u had "
            "confirmed the rejected update)", s->role, s->name,
            oldc ? "config" : "engine", oldc ? oldc : olde, s->inst);
      }
      g_free (olde);
      g_free (oldv);
      g_free (oldc);
    }
  }
  /* the identity rename rolls back ONCE for the whole group (same lock hold as
   * the shard restores -- status/routing never see a half-rolled-back group).
   * Idempotent via the rename guards: a same-key reload carries no rename. */
  if (!commit && group->from_key && group->to_key)
    renamed_back = modelmux_bin_rename_model_locked (modelmux_bin, group->to_key,
        group->from_key);
  g_mutex_unlock (&modelmux_bin->lock);

  if (commit) {
    MM_INFO ("  - MultiModelBin  model/update group COMMITTED (%u/%u shards) "
        "-- '%s' serving the new checkpoint on every shard",
        group->confirmed, group->expected, group->live_key);
    /* app-facing completion notice: the OTA actually committed now. Show the
     * full identity TRANSITION -> `new model updated [Trafficcamnet@1 -> Trafficcamnet@2]`
     * when the update moved the version (from_key/to_key set); the whole string
     * rides in the "name" field so the app renders it in one bracket. Fall back
     * to the live key for a same-version reload. */
    if (group->from_key && group->to_key) {
      gchar *transition = g_strdup_printf ("%s -> %s", group->from_key, group->to_key);
      modelmux_post_model_event (modelmux_bin->parent_pipeline, "model-updated",
          transition, "", -1, TRUE, NULL);
      g_free (transition);
    } else {
      modelmux_post_model_event (modelmux_bin->parent_pipeline, "model-updated",
          group->live_key, "", -1, TRUE, NULL);
    }
  } else {
    MM_ERR ("  - MultiModelBin  model/update group ROLLED BACK (%u/%u failed): "
        "compensating reload of previous engine on %u shard(s)%s "
        "(UPDATE_ENGINE_REJECTED; %u member shard(s) restored)",
        group->failed, group->expected, ncomp,
        renamed_back ? " -- identity renamed back" : "", nshards);
    modelmux_post_model_event (modelmux_bin->parent_pipeline, "model-updated",
        group->live_key, "", -1, FALSE, "rolled back (engine rejected)");
    if (renamed_back) {
      MM_ERR ("MultiModelBin: identity '%s' renamed BACK to '%s' -- the "
          "rejected OTA engine no longer serves once the compensating "
          "reload(s) confirm (UPDATE_ENGINE_REJECTED)", group->to_key,
          group->from_key);
      /* defaults + control-plane hook, both OUTSIDE modelmux_bin->lock (set_default
       * self-acquires; the cb contract is 'no bin locks held'). Fires ONCE
       * per group. */
      modelmux_ota_restore_default_refs (modelmux_bin, group->to_key, group->from_key);
      if (modelmux_bin->ota_rollback_cb)
        modelmux_bin->ota_rollback_cb (modelmux_bin->ota_rollback_owner, group->to_key,
            group->from_key);
    }
  }
  modelmux_ota_group_free (group);           /* dispatched -> the notify skips destroy */
  return G_SOURCE_REMOVE;
}

/* Marshal the group resolution to the main loop (modelmux_bin->lock must not be taken on
 * nvinfer's confirm thread). Not schedulable only on shutdown -- teardown then
 * frees the shards + stashes wholesale, so dropping the group there is safe. */
static void
modelmux_ota_group_schedule_resolve (ModelMuxOtaGroup * group)
{
  if (!modelmux_tracked_idle_add (group->modelmux_bin, modelmux_ota_group_resolve_main, group,
          modelmux_ota_group_free, NULL))
    modelmux_ota_group_free (group);
}

/* Record one shard's confirm outcome in its group; the LAST outstanding confirm
 * schedules the resolution. nvinfer-thread safe: group->lock only. */
static void
modelmux_ota_group_confirm (ModelMuxOtaGroup * group, gboolean ok)
{
  gboolean last;
  g_mutex_lock (&group->lock);
  group->confirmed++;
  if (!ok)
    group->failed++;
  last = group->sealed && (group->confirmed == group->expected);
  g_mutex_unlock (&group->lock);
  if (last)
    modelmux_ota_group_schedule_resolve (group);
}

/* nvinfer 'model-updated' signal handler: fired by nvinfer AFTER it finishes a
 * model (re)load -- i.e. the async confirm of the OTA we triggered in
 * model_bin_reload_engine (g_object_set model-engine-file / config-file-path).
 * Also fires on the initial warm-up load. `status` is nvinfer's result (0 = OK).
 * It finalizes the optimistic swap done at trigger time:
 *   success -> COMMIT (drop the stashed old version/engine/config copies)
 *   failure -> ROLL BACK (restore the previous engine/config/identity; the
 *              previous checkpoint keeps serving)
 * Three modes: (1) compensating reload of a group rollback -> log only;
 * (2) grouped update -> record this shard's outcome, let the group commit/roll
 * back atomically; (3) ungrouped -> commit/rollback inline here. Clearing
 * ota_await reopens the shard for the next update (UPDATE_IN_FLIGHT). Runs on
 * nvinfer's own thread -> never takes modelmux_bin->lock (rename-back is deferred
 * to the main loop). */
static void
infer_engine_update_cb (GstElement * infer, gint status, gchar * cfg_file,
    gpointer udata)
{
  ModelBin *model_bin = (ModelBin *) udata;
  ModelMuxOtaGroup *group;
  gchar *oldv, *olde, *oldc, *oldk, *newk;
  gpointer ota_ib;
  gboolean awaiting;
  (void) infer;

  /* COMMIT-ON-CONFIRM: an OTA reload swaps the visible identity optimistically
   * at trigger time (the lock-free probes must see one coherent view); this
   * confirm decides whether that swap STAYS (success -> drop the rollback
   * copies) or is ROLLED BACK (failure -> the previous engine keeps serving,
   * so status/routing must keep claiming the previous version). */
  g_mutex_lock (&model_bin->prov_lock);
  if (model_bin->ota_compensating) {
    /* group-rollback COMPENSATING reload of the PREVIOUS engine: by design it
     * never re-enters the group logic (the group already resolved; provenance
     * was restored at the rollback) -- this confirm only logs. */
    model_bin->ota_compensating = FALSE;
    model_bin->ota_await = FALSE;
    g_mutex_unlock (&model_bin->prov_lock);
    if (status == 0)
      MM_INFO ("      - nvmodelbin [%s/%s]  compensating reload of the "
          "previous engine confirmed OK -- group rollback complete on this "
          "shard", model_bin->role, model_bin->name);
    else
      MM_ERR ("ModelBin %s/%s: COMPENSATING reload of the previous engine "
          "FAILED (status=%d) -- this shard keeps serving the rolled-back "
          "update's engine while identity/status claim the previous "
          "checkpoint. Re-run model/update to converge it.",
          model_bin->role, model_bin->name, status);
    return;
  }
  group = (ModelMuxOtaGroup *) model_bin->ota_group;
  if (group && model_bin->ota_await) {
    /* GROUPED update: only RECORD this shard's outcome -- the rollback stash
     * and ota_await stay armed until the whole group resolves (commit or one
     * group rollback) on the main loop, so the UPDATE_IN_FLIGHT admission
     * keeps refusing overlapping updates throughout. */
    model_bin->ota_ok = (status == 0);
    g_mutex_unlock (&model_bin->prov_lock);
    if (status == 0)
      MM_INFO ("      - nvmodelbin [%s/%s]  nvinfer engine (re)loaded OK  (%s) "
          "-- shard confirm recorded in the update group", model_bin->role, model_bin->name,
          cfg_file ? cfg_file : "engine");
    else
      MM_ERR ("ModelBin %s/%s: nvinfer engine (re)load FAILED (status=%d) -- "
          "previous engine still serving; shard failure recorded in the "
          "update group (the WHOLE group will roll back)", model_bin->role, model_bin->name,
          status);
    modelmux_ota_group_confirm (group, status == 0);
    return;
  }
  /* ungrouped confirm (initial warm-up, idle-reset, or a legacy per-shard
   * reload trigger): the original per-shard commit/rollback below. */
  awaiting = model_bin->ota_await;
  oldv = model_bin->ota_old_version;
  olde = model_bin->ota_old_engine;
  oldc = model_bin->ota_old_config;
  oldk = model_bin->ota_old_key;
  newk = model_bin->ota_new_key;
  ota_ib = model_bin->ota_ib;
  model_bin->ota_old_version = NULL;
  model_bin->ota_old_engine = NULL;
  model_bin->ota_old_config = NULL;
  model_bin->ota_old_key = NULL;
  model_bin->ota_new_key = NULL;
  model_bin->ota_ib = NULL;
  /* a FAILED by-config reload rolls config_file + cfg_gpu back HERE, while
   * ota_await is still TRUE: clearing it first would reopen UPDATE_IN_FLIGHT
   * admission, and a racing reload's own config_file swap could interleave
   * with this one (stale pointer retired after the reload freed it -> double
   * free at teardown). The pair is published together so no modelmux_bin->lock reader
   * ever sees new-config/stale-gpu. modelmux_config_file_gpu is a small file read;
   * prov_lock is only contended by rare retire/OTA ops, never per-frame. */
  if (status != 0 && awaiting && oldc) {
    gchar *bad = model_bin->config_file;
    gint rg = modelmux_config_file_gpu (oldc);
    g_atomic_pointer_set (&model_bin->config_file, oldc);   /* ownership transferred */
    model_bin->cfg_gpu = rg;
    oldc = NULL;
    g_mutex_unlock (&model_bin->prov_lock);
    modelmux_bin_retire_str (model_bin, bad);   /* re-takes prov_lock; modelmux_bin->lock readers that
                                    * already fetched `bad` keep valid memory */
    g_mutex_lock (&model_bin->prov_lock);
  }
  model_bin->ota_await = FALSE;           /* admission reopens ONLY now */
  g_mutex_unlock (&model_bin->prov_lock);

  if (status == 0) {
    MM_INFO ("      - nvmodelbin [%s/%s]  nvinfer engine (re)loaded OK  (%s)%s",
        model_bin->role, model_bin->name, cfg_file ? cfg_file : "engine",
        awaiting ? " -- OTA COMMITTED" : "");
    g_free (oldv);
    g_free (olde);
    g_free (oldc);
    g_free (oldk);
    g_free (newk);
    return;
  }
  MM_ERR ("ModelBin %s/%s: nvinfer engine (re)load FAILED (status=%d) -- previous "
      "engine still serving (e.g. OTA topology mismatch)", model_bin->role, model_bin->name, status);
  if (awaiting) {
    /* config_file/cfg_gpu were already rolled back above, BEFORE ota_await
     * cleared (see the admission-race note there). */
    /* roll the identity back (UPDATE_ENGINE_REJECTED): atomic-swap the stashed
     * previous version/engine back in, retire the failed-update strings. */
    if (oldv) {
      gchar *bad = (gchar *) g_atomic_pointer_get (&model_bin->version);
      g_atomic_pointer_set (&model_bin->version, oldv);
      modelmux_bin_retire_str (model_bin, bad);
      oldv = NULL;                     /* transferred */
    }
    if (olde) {
      gchar *bad = (gchar *) g_atomic_pointer_get (&model_bin->engine);
      g_atomic_pointer_set (&model_bin->engine, olde);
      modelmux_bin_retire_str (model_bin, bad);
      olde = NULL;                     /* transferred */
    }
    /* the identity RENAME (hash keys + routing strings) rolls back on the
     * MAIN LOOP -- modelmux_bin->lock must not be taken on nvinfer's thread (see
     * modelmux_ota_rename_rollback_main). On shutdown the idle is not schedulable;
     * teardown frees every table wholesale, so the rename back is moot. */
    if (oldk && newk && ota_ib) {
      ModelMuxOtaRenameCtx *ctx = g_new0 (ModelMuxOtaRenameCtx, 1);
      ctx->modelmux_bin = (ModelMuxBin *) ota_ib;
      ctx->from_key = newk;            /* transfer */
      ctx->to_key = oldk;
      oldk = newk = NULL;
      if (!modelmux_tracked_idle_add (ctx->modelmux_bin, modelmux_ota_rename_rollback_main, ctx,
              modelmux_ota_rename_ctx_free, NULL))
        modelmux_ota_rename_ctx_free (ctx);
    }
    MM_ERR ("ModelBin %s/%s: OTA ROLLED BACK (UPDATE_ENGINE_REJECTED) -- identity "
        "and engine restored to the serving checkpoint", model_bin->role, model_bin->name);
  }
  g_free (oldv);
  g_free (olde);
  g_free (oldc);
  g_free (oldk);
  g_free (newk);
}

/* Retire a provenance string that is being superseded by an in-place OTA reload
 *. The lock-free per-frame probe may still hold the old pointer, so we MUST
 * NOT free it now; stash it and release it only at bin teardown. Bounded by #reloads. */
static void
modelmux_bin_retire_str (ModelBin * model_bin, gchar * old)
{
  if (!old)
    return;
  g_mutex_lock (&model_bin->prov_lock);
  if (!model_bin->retired)
    model_bin->retired = g_ptr_array_new_with_free_func (g_free);
  g_ptr_array_add (model_bin->retired, old);
  g_mutex_unlock (&model_bin->prov_lock);
}

/* In-place engine/config hot-swap on ONE bin (one nvinfer). Caller holds
 * modelmux_bin->lock. Pure property set: no pad/element teardown, gie-id untouched, so streams
 * already on this bin keep flowing. nvinfer queues the swap on its own thread and
 * confirms via infer_engine_update_cb (on failure the previous engine keeps serving).
 * old_key/new_key (both-or-neither; NULL = no rename rides this reload) stash the
 * identity-rename rollback: the caller renames name@FROM -> name@TO once every
 * shard triggered, and a failed confirm renames it BACK via these copies. They
 * must be stashed BEFORE the trigger -- nvinfer can confirm immediately.
 * `group` (non-NULL on every model/update path today) enrolls the shard in the
 * caller's GROUP TRANSACTION: the confirm then only RECORDS the outcome and the
 * whole group commits/rolls back together (see ModelMuxOtaGroup). The caller counts
 * the shard into group->expected iff this returns TRUE, and must SEAL the group
 * after the trigger loop. NULL = legacy per-shard commit/rollback. */
/* fwd-decl: defined in the inference-backend helper layer below (used by the OTA guard here) */
static gboolean modelmux_infer_supports_inplace_reload (ModelType type);

static gboolean
model_bin_reload_engine (ModelBin * model_bin, const gchar * new_cfg,
    const gchar * new_engine, const gchar * new_version,
    ModelMuxBin * modelmux_bin, const gchar * old_key, const gchar * new_key,
    ModelMuxOtaGroup * group)
{
  gboolean by_engine = (new_engine && *new_engine);
  gboolean by_cfg = (new_cfg && *new_cfg);
  gboolean awaiting;
  if (!model_bin || !model_bin->infer || (!by_engine && !by_cfg))
    return FALSE;

  /* UPDATE_IN_FLIGHT guard (defensive -- the reload admission in the callers
   * already refuses a whole update on this): there is exactly ONE rollback
   * stash per shard, so a second trigger while the previous one still awaits
   * nvinfer's confirm would REPLACE it and cross-wire the confirms: confirm #1
   * would consume update #2's stash and clear ota_await, so a failed confirm
   * #2 would roll back nothing (or #1's failure would roll back to the WRONG
   * identity). Only reload paths SET ota_await (serialized under modelmux_bin->lock);
   * the confirm callback only CLEARS it, so check-then-stash is race-free. */
  g_mutex_lock (&model_bin->prov_lock);
  awaiting = model_bin->ota_await;
  g_mutex_unlock (&model_bin->prov_lock);
  if (awaiting) {
    MM_WARN ("ModelBin %s/%s: OTA reload refused -- the previous update is "
        "still awaiting nvinfer's confirm (UPDATE_IN_FLIGHT; retry shortly)",
        model_bin->role, model_bin->name);
    return FALSE;
  }

  /* never reload a WARMING bin: its load thread runs LOCK-FREE and writes
   * model_bin->engine itself -- swapping/retiring here would race it (double free of the
   * retired string / leak of the freshly-warmed one). The caller retries once the
   * warm-up settles (WARMED/SERVING/FAILED). */
  if (g_atomic_int_get (&model_bin->status) == MODEL_WARMING) {
    MM_WARN ("ModelBin %s/%s: OTA reload requested while the engine is still "
        "WARMING -- skipped for this shard (retry when it is READY)",
        model_bin->role, model_bin->name);
    return FALSE;
  }

  /* In-place OTA hot-swap is an nvinfer-native feature (model-engine-file / config-file-path +
   * the 'model-updated' signal). nvinferserver has no such mechanism, so reject the in-place
   * reload with a clear warning and leave the current model serving (unload + load to change
   * an nvinferserver model). */
  if (!modelmux_infer_supports_inplace_reload (model_bin->type)) {
    MM_WARN ("ModelBin %s/%s: in-place OTA reload is not supported for nvinferserver -- "
        "current model keeps serving (use model/unload + model/load to change it)",
        model_bin->role, model_bin->name);
    return FALSE;
  }

  /* stash rollback COPIES first: nvinfer confirms asynchronously via
   * infer_engine_update_cb, which either commits (frees these) or swaps them
   * back if the reload fails. The UPDATE_IN_FLIGHT guard above ensures no
   * unconfirmed stash exists here (a confirmed one leaves NULLs), so the
   * frees below are belt-and-braces only. */
  if (by_cfg) {
    /* an in-place config reload cannot MOVE the shard between devices:
     * nvinfer's on-the-fly update keeps its device, and the bin's migration
     * pair (or its deliberate absence on a device-local bin) is fixed at
     * creation -- accepting a device-changing config would leave placement
     * metadata describing a device the shard does not run on. */
    gint ng = modelmux_config_file_gpu (new_cfg);
    if (ng != model_bin->cfg_gpu) {
      MM_ERR ("ModelBin %s/%s: in-place config reload REJECTED -- it would "
          "move the shard's device (config pin %d -> %d); unload + load the "
          "model on the target gpu instead (UPDATE_DEVICE_MOVE)",
          model_bin->role, model_bin->name, model_bin->cfg_gpu, ng);
      return FALSE;
    }
  }
  g_mutex_lock (&model_bin->prov_lock);
  g_free (model_bin->ota_old_version);
  g_free (model_bin->ota_old_engine);
  g_free (model_bin->ota_old_config);
  g_free (model_bin->ota_old_key);
  g_free (model_bin->ota_new_key);
  model_bin->ota_old_version = g_strdup ((const gchar *)
      g_atomic_pointer_get (&model_bin->version));
  model_bin->ota_old_engine = g_strdup ((const gchar *)
      g_atomic_pointer_get (&model_bin->engine));
  model_bin->ota_old_config = by_cfg ? g_strdup (model_bin->config_file) : NULL;
  model_bin->ota_old_key = g_strdup (old_key);    /* NULL = no rename rides this OTA */
  model_bin->ota_new_key = g_strdup (new_key);
  model_bin->ota_ib = modelmux_bin;
  model_bin->ota_group = group;     /* enroll in the caller's group transaction (if any) */
  model_bin->ota_ok = FALSE;
  model_bin->ota_await = TRUE;
  g_mutex_unlock (&model_bin->prov_lock);

  /* update the provenance strings the lock-free probe reads: atomic-swap the pointer,
   * retire (don't free) the old one. config_file is NOT read on the streaming thread,
   * so a plain swap is fine. */
  if (new_version && *new_version) {
    gchar *nv = g_strdup (new_version);
    gchar *old = (gchar *) g_atomic_pointer_get (&model_bin->version);
    g_atomic_pointer_set (&model_bin->version, nv);
    modelmux_bin_retire_str (model_bin, old);
  }
  if (by_engine) {
    gchar *ne = g_strdup (new_engine);
    gchar *old = (gchar *) g_atomic_pointer_get (&model_bin->engine);
    g_atomic_pointer_set (&model_bin->engine, ne);
    modelmux_bin_retire_str (model_bin, old);
  }
  if (by_cfg) {
    /* RETIRE (never free) the superseded config path: the failed-confirm
     * rollback on nvinfer's thread swaps this pointer under prov_lock only --
     * retiring on BOTH sides means neither writer can free a string the other
     * (or an modelmux_bin->lock reader) still holds. cfg_gpu is published together with
     * the path so no reader sees a mixed pair. */
    gchar *bad = model_bin->config_file;
    gint ng = modelmux_config_file_gpu (new_cfg);
    g_mutex_lock (&model_bin->prov_lock);
    g_atomic_pointer_set (&model_bin->config_file, g_strdup (new_cfg));
    model_bin->cfg_gpu = ng;
    g_mutex_unlock (&model_bin->prov_lock);
    modelmux_bin_retire_str (model_bin, bad);
  }

  /* ==== ACTUAL OTA: invoke nvinfer's native on-the-fly model-update API ====
   * This g_object_set IS the hot-swap -- nvinfer reloads in place (no pad/element
   * teardown, gie-id kept, serving streams keep flowing) and confirms later via
   * the 'model-updated' signal -> infer_engine_update_cb (commit or rollback):
   *  - engine given -> set "model-engine-file"  => MODEL_LOAD_FROM_ENGINE
   *                    (engine-only swap; keeps networkMode/scale/format/offsets)
   *  - else config  -> set "config-file-path"   => MODEL_LOAD_FROM_CONFIG (full reload) */
  if (by_engine)
    g_object_set (G_OBJECT (model_bin->infer), "model-engine-file", new_engine, NULL);
  else
    g_object_set (G_OBJECT (model_bin->infer), "config-file-path", new_cfg, NULL);

  MM_INFO ("      - nvmodelbin [%s/%s]  OTA reload TRIGGERED (gie=%u kept): %s='%s' "
      "version='%s'", model_bin->role, model_bin->name, model_bin->unique_id,
      by_engine ? "engine" : "config", by_engine ? new_engine : new_cfg,
      new_version ? new_version : "(unset)");
  return TRUE;
}

/* ================================================================== *
 *  Inference-backend helper layer.
 *
 *  ALL nvinfer-vs-nvinferserver differences live here so the rest of the ModelBin stays
 *  backend-agnostic (same nvstreammux -> [backend] -> nvstreamdemux chain, routing, sharding,
 *  passthrough, provenance, drain-recovery). Differences handled:
 *    - element:    nvinfer            vs nvinferserver
 *    - properties: config-file-path/unique-id/batch-size on BOTH; model-engine-file nvinfer-only
 *    - signal:     'model-updated' is nvinfer-only (nvinferserver loads synchronously in start())
 *    - readiness:  same state-change path for both (nvinferserver's load is synchronous)
 *    - provenance: nvinfer reports the engine via model-engine-file; nvinferserver has none, so
 *                  we identify it by its config-file basename (Triton model repo)
 *    - OTA:        in-place hot-swap is nvinfer-only
 * ================================================================== */

/* Create the inference element for a backend type. */
static GstElement *
modelmux_infer_create (ModelType type, const gchar * elem_name)
{
  return create_gst_element (type == MODEL_INFERSERVER ? "nvinferserver" : "nvinfer", elem_name);
}

/* TRUE iff the backend supports nvinfer's native in-place engine/config hot-swap (OTA). */
static gboolean
modelmux_infer_supports_inplace_reload (ModelType type)
{
  return (type == MODEL_INFER);
}

/* Configure the inference element + wire its completion signal. config-file-path/unique-id/
 * batch-size exist on both backends; the 'model-updated' signal and a model-engine-file
 * override are nvinfer-only (nvinferserver has neither -- its engine/model is in its config). */
static void
modelmux_infer_configure (ModelBin * model_bin, const gchar * engine_file)
{
  g_object_set (G_OBJECT (model_bin->infer),
      "config-file-path", model_bin->config_file,
      "unique-id", model_bin->unique_id,
      "batch-size", model_bin->max_streams, NULL);

  if (model_bin->type == MODEL_INFER) {
    /* OTA completion signal (nvinfer fires "model-updated" on every (re)load;
     * signature (gint status, gchar* cfg) matches infer_engine_update_cb). The
     * name MUST be exactly what nvinfer registers -- a wrong name silently drops
     * the confirm (OTA never completes) and emits a GObject-CRITICAL at connect. */
    g_signal_connect (model_bin->infer, "model-updated",
        G_CALLBACK (infer_engine_update_cb), model_bin);
    /* optional prebuilt-engine override for A/B distinct checkpoints (nvinfer only) */
    if (engine_file && *engine_file)
      g_object_set (G_OBJECT (model_bin->infer), "model-engine-file", engine_file, NULL);
  } else if (engine_file && *engine_file) {
    MM_WARN ("ModelBin %s/%s: explicit engine ignored for nvinferserver (its engine/model is "
        "defined in the config-file; use distinct config-files for A/B)", model_bin->role, model_bin->name);
  }
}

/* Canonical model-instance key "name@version" (version defaults to 1 when unset).
 * The pool models map + the limbo map are keyed by this, so multiple versions of one
 * name coexist as distinct bins. */
gchar *
model_key (const gchar * name, const gchar * version)
{
  if (!name)
    return NULL;
  return g_strdup_printf ("%s%c%s", name, MM_MODEL_KEY_SEP,
      (version && *version) ? version : MM_MODEL_VERSION_DEFAULT);
}

/* Split "name@version" back into bare name + version. Either out-ptr may be NULL.
 * A key with no separator is treated as a bare name with the default version. */
void
model_key_split (const gchar * key, gchar ** name_out, gchar ** version_out)
{
  const gchar *sep;
  if (name_out)
    *name_out = NULL;
  if (version_out)
    *version_out = NULL;
  if (!key)
    return;
  sep = strchr (key, MM_MODEL_KEY_SEP);
  if (sep) {
    if (name_out)
      *name_out = g_strndup (key, (gsize) (sep - key));
    if (version_out)
      *version_out = g_strdup (sep + 1);
  } else {
    if (name_out)
      *name_out = g_strdup (key);
    if (version_out)
      *version_out = g_strdup (MM_MODEL_VERSION_DEFAULT);
  }
}

/* Sanitize a version string into a token safe for GStreamer element names
 * (keep [A-Za-z0-9._-], replace the rest with '_'). Caller frees. */
static gchar *
modelmux_version_token (const gchar * version)
{
  gchar *t = g_strdup ((version && *version) ? version : MM_MODEL_VERSION_DEFAULT);
  gchar *p;
  for (p = t; *p; p++)
    if (!g_ascii_isalnum (*p) && *p != '.' && *p != '_' && *p != '-')
      *p = '_';
  return t;
}

/**
 * Create a new model bin.
 * mux -> infer/nvinferserver -> demux
 */
ModelBin *
model_bin_new (const gchar * name, const gchar * role,
    const gchar * config_file, const gchar * engine_file, const gchar * version,
    guint unique_id, guint max_streams, const ModelMuxConfig * config,
    GstElement * parent_pipeline,
    GstElement * container, const GHashTable * stream_names, guint inst)
{
  ModelBin *model_bin = g_new0 (ModelBin, 1);
  g_mutex_init (&model_bin->prov_lock);
  gchar bname[192];
  gboolean creation_failed = FALSE;   /* cross-GPU preconditions (P2P/nvdsxfer) */
  gchar sfx[24];
  gchar *vtok;
  guint i;

  model_bin->name = g_strdup (name);
  modelmux_role_set (model_bin, role);   /* interned; atomic-swappable */
  model_bin->config_file = g_strdup (config_file);
  model_bin->cfg_gpu = modelmux_config_file_gpu (config_file);  /* config's own device pin (-1 = none) */
  /* select the inference backend from the config-file format (nvinfer vs nvinferserver).
   * Detected per bin so every shard of a model shares the same backend as its base.
   * All current callers pass a non-NULL config_file (the load paths check use_cfg/def->config_file);
   * guard defensively anyway -> default to nvinfer rather than deref a NULL config. */
  model_bin->type = config_file ? modelmux_detect_model_type (config_file) : MODEL_INFER;
  model_bin->version = (version && *version) ? g_strdup (version) : NULL;
  /* canonical instance key (name@version): the pool/limbo hash key. Distinct versions
   * of one name are distinct bins. version token folds into element names below so two
   * versions in one role pool never collide. */
  model_bin->key = model_key (name, version);
  vtok = modelmux_version_token (version);
  model_bin->unique_id = unique_id;
  model_bin->inst = inst;
  model_bin->gpu = -1;                /* placement recorded by the caller (-1 = config's device) */
  model_bin->max_streams = max_streams ? max_streams : 1;
  /* per-model nvstreammux properties (from [muxer]); captured so the drain-recovery recreate
   * reuses the same values. Shared width/height/live-source + the per-model flush timeout. */
  model_bin->mux_width        = config ? config->mux_width             : MM_MUX_WIDTH;
  model_bin->mux_height       = config ? config->mux_height            : MM_MUX_HEIGHT;
  model_bin->mux_push_timeout = config ? config->model_mux_push_timeout : 10000;
  model_bin->mux_live_source  = config ? config->mux_live_source       : TRUE;
  model_bin->mux_gpu          = config ? (gint) config->gpu             : 0;
  /* perf metrics: opt-in. `perf_on` is the cached hot-path gate; `perf` holds the counters. */
  model_bin->perf_on = (config && config->attach_perf_metric);
  model_bin->perf = model_bin->perf_on ? modelmux_perf_new () : NULL;
  model_bin->stream_names = stream_names;
  model_bin->free_idx = g_queue_new ();
  model_bin->idx_to_stream = g_hash_table_new (g_direct_hash, g_direct_equal);
  /* lock-free mirror for the per-frame probe (see resolve_source_id_fallback) */
  model_bin->slot_stream = g_new (gint, model_bin->max_streams);
  for (i = 0; i < model_bin->max_streams; i++) {
    g_queue_push_tail (model_bin->free_idx, GINT_TO_POINTER (i));
    model_bin->slot_stream[i] = -1;
  }

  /* Element names read as "<part>-<model>-<version>-g<gpu>" so a glance at the
   * pipeline tells you WHICH model+version is on WHICH device -- the three facts
   * that matter operationally. Role is deliberately omitted: it flips at runtime
   * (route Unknown->Primary/Shadow) and would leave a stale name behind. The
   * device is the config's own pin (cfg_gpu) when it has one, else the pipeline
   * device (mux_gpu); an explicit placement override applied after creation is a
   * display-only nicety and not reflected here.
   *
   * Uniqueness: name@version and gpu are DISPLAY context, NOT a unique key -- the
   * SAME name@version can be live as two distinct bins at once (e.g. one stream's
   * PRIMARY and another stream's SHADOW both on Car@2, or a scoped swap that spins
   * up a second Car@2 bin). The nvinfer gie-unique-id (`unique_id`) is the real
   * per-bin-group identity, so it goes into the name to guarantee no GstElement
   * name collision inside the container (a collision makes gst_bin_add fail and the
   * bin unusable). unique_id is SHARED across a model's shards; the "-i<inst>"
   * suffix then separates same-device overflow shards of one group. */
  gint name_gpu = (model_bin->cfg_gpu >= 0) ? model_bin->cfg_gpu : model_bin->mux_gpu;
  sfx[0] = '\0';
  if (inst)
    g_snprintf (sfx, sizeof (sfx), "-i%u", inst);

  g_snprintf (bname, sizeof (bname), "nvmodelbin-%s-%s-g%d-gie%u%s", name, vtok, name_gpu, model_bin->unique_id, sfx);
  model_bin->bin = gst_bin_new (bname);

  g_snprintf (bname, sizeof (bname), "mux-%s-%s-g%d-gie%u%s", name, vtok, name_gpu, model_bin->unique_id, sfx);
  model_bin->mux = create_gst_element ("nvstreammux", bname);
  g_snprintf (bname, sizeof (bname), "infer-%s-%s-g%d-gie%u%s", name, vtok, name_gpu, model_bin->unique_id, sfx);
  model_bin->infer = modelmux_infer_create (model_bin->type, bname);
  g_snprintf (bname, sizeof (bname), "demux-%s-%s-g%d-gie%u%s", name, vtok, name_gpu, model_bin->unique_id, sfx);
  model_bin->demux = create_gst_element ("nvstreamdemux", bname);
  /* MULTI-GPU device migration around the inference element. nvinfer HARD-REJECTS
   * input surfaces allocated on another GPU ("Memory Compatibility Error"), and the
   * decoders + shared demux live on the PIPELINE device ([multimodel] gpu-id, default
   * 0).
   *   same device  -> NO migration elements: the mux links straight to the
   *                   inference element (zero extra work; the common case).
   *   cross device -> xfer-in migrates the batch to THIS shard's device and
   *                   xfer-out brings it back, so demux -> combined display mux
   *                   always aggregate SAME-DEVICE buffers. nvdsxfer is the ONLY
   *                   supported transport (stock nvvideoconvert cannot move
   *                   batches across devices); it HARD-REQUIRES CUDA peer access
   *                   (NVLink / PCIe P2P) -- both that and the factory presence
   *                   are checked below, and a placement that cannot be served
   *                   FAILS CREATION deterministically (clean load/attach
   *                   rejection, never a per-buffer runtime failure). */
  {
    /* placement resolution at CREATION: the shard's OWN config pin first --
     * that is what nvinfer will actually run. The version REGISTRY records
     * only the BASE placement of a load, so for a grown clone / gpus[]
     * sibling / refresh whose derived config pins a DIFFERENT device, the
     * registry would lie (seed no/wrong migration pair, then set_gpu's
     * device-local refusal leaves a broken shard). Registry is the fallback
     * for configs that carry no pin of their own. */
    gint eff = model_bin->cfg_gpu;
    guint pipe_gpu = config ? config->gpu : 0;
    gboolean cross;
    if (eff < 0)
      eff = config ? modelmux_config_version_gpu (config, name, version) : -1;
    cross = (eff >= 0 && eff != (gint) pipe_gpu);
    model_bin->conv_gpu = eff >= 0 ? eff : (gint) pipe_gpu;   /* the pair's target device */
    if (!cross) {
      /* SAME device: NO migration elements at all -- the mux links straight to
       * the inference element (head/tail fallbacks below wire it). Placement is
       * resolved at creation (derived configs bake the gpu -> model_bin->cfg_gpu), so
       * a same-device bin never needs a device move; a later cross-device
       * re-pin is refused loudly in model_bin_set_gpu. */
      model_bin->conv_in = model_bin->conv_out = NULL;
    } else {
      /* CROSS device: nvdsxfer pair, the ONLY supported cross-GPU transport
       * (stock nvvideoconvert cannot move batches between devices, so there is
       * deliberately NO fallback). Both preconditions are checked HERE so a
       * cross-device placement fails DETERMINISTICALLY at admission -- a clear
       * load/attach rejection instead of a shard that warms then errors on
       * every buffer (or fails a bare READY->PAUSED transition):
       *   1. CUDA peer access (NVLink / PCIe P2P) must exist BOTH ways --
       *      nvdsxfer hard-requires it (its start() fails otherwise);
       *   2. the nvdsxfer factory must be installed. */
      int ab = 0, ba = 0;
      if (cudaDeviceCanAccessPeer (&ab, (int) pipe_gpu, eff) != cudaSuccess ||
          cudaDeviceCanAccessPeer (&ba, eff, (int) pipe_gpu) != cudaSuccess ||
          !ab || !ba) {
        MM_ERR ("ModelBin %s/%s: cross-GPU placement (pipeline gpu %u -> model "
            "gpu %d) REJECTED -- no CUDA peer access between the devices "
            "(nvdsxfer requires NVLink/PCIe P2P); place the model on the "
            "pipeline device or use a P2P-capable GPU pair (P2P_UNAVAILABLE)",
            role, name, pipe_gpu, eff);
        model_bin->conv_in = model_bin->conv_out = NULL;
        creation_failed = TRUE;
      } else {
        g_snprintf (bname, sizeof (bname), "xfer-in-%s-%s-%s%s", role, name, vtok, sfx);
        model_bin->conv_in = create_gst_element ("nvdsxfer", bname);
        g_snprintf (bname, sizeof (bname), "xfer-out-%s-%s-%s%s", role, name, vtok, sfx);
        model_bin->conv_out = create_gst_element ("nvdsxfer", bname);
        if (!model_bin->conv_in || !model_bin->conv_out) {
          MM_ERR ("ModelBin %s/%s: cross-GPU placement on gpu %d REJECTED -- "
              "the 'nvdsxfer' factory is unavailable (install "
              "libnvdsgst_xfer.so) (XFER_UNAVAILABLE)", role, name, eff);
          if (model_bin->conv_in)  { gst_object_unref (model_bin->conv_in);  model_bin->conv_in = NULL; }
          if (model_bin->conv_out) { gst_object_unref (model_bin->conv_out); model_bin->conv_out = NULL; }
          creation_failed = TRUE;
        }
      }
    }
    model_bin->conv_cross = cross;            /* remembers WHY the pair exists (or not) */
  }
  g_free (vtok);                       /* only needed for the element names above */

  if (creation_failed || !model_bin->bin || !model_bin->mux || !model_bin->infer || !model_bin->demux) {
    /* the children are NOT in model_bin->bin yet (gst_bin_add_many runs later), so
     * model_bin_free would only dispose the empty bin -- unref the floating
     * elements explicitly or they leak (e.g. missing nvinferserver factory). */
    if (model_bin->mux)   gst_object_unref (model_bin->mux);
    if (model_bin->infer) gst_object_unref (model_bin->infer);
    if (model_bin->demux) gst_object_unref (model_bin->demux);
    if (model_bin->conv_in)  gst_object_unref (model_bin->conv_in);
    if (model_bin->conv_out) gst_object_unref (model_bin->conv_out);
    model_bin->mux = model_bin->infer = model_bin->demux = NULL;
    model_bin->conv_in = model_bin->conv_out = NULL;
    model_bin_free (model_bin);
    return NULL;
  }
  if (model_bin->conv_in && model_bin->conv_out) {
    /* (a cross-device bin with a missing pair cannot reach here: creation
     * already failed deterministically above) */
    /* seed the nvdsxfer pair with the shard's EFFECTIVE device (model_bin->conv_gpu),
     * so placement holds from the very first buffer. (model_bin_set_gpu
     * overrides for explicit placements, still pre-first-buffer.)
     * nvdsxfer property semantics: gpu-id = OUTPUT/DESTINATION device (its
     * output pool is allocated there -- reference: its own README pipeline
     * `mux gpu-id=0 ! nvdsxfer gpu-id=1 p2p-gpu-id=0 ! nvinfer gpu-id=1`);
     * p2p-gpu-id = the PEER (upstream/source) device it enables peer access
     * with; batch-size sizes its output pool. */
    guint pipe_gpu = config ? config->gpu : 0;
    set_prop_if_exists (model_bin->conv_in, "gpu-id", model_bin->conv_gpu);
    set_prop_if_exists (model_bin->conv_in, "p2p-gpu-id", (gint) pipe_gpu);
    set_prop_if_exists (model_bin->conv_in, "batch-size", model_bin->max_streams);
    set_prop_if_exists (model_bin->conv_out, "gpu-id", (gint) pipe_gpu);
    set_prop_if_exists (model_bin->conv_out, "p2p-gpu-id", model_bin->conv_gpu);
    set_prop_if_exists (model_bin->conv_out, "batch-size", model_bin->max_streams);
  }

  /* nvstreammux configured to batch this model's streams. batch-size exists on both the
   * legacy and the new muxer; width/height/timeout/live-source are legacy-only (the new
   * muxer takes them from its config-file) -> set defensively so neither muxer warns. */
  set_prop_if_exists (model_bin->mux, "batch-size", model_bin->max_streams);
  set_prop_if_exists (model_bin->mux, "width", model_bin->mux_width);
  set_prop_if_exists (model_bin->mux, "height", model_bin->mux_height);
  set_prop_if_exists (model_bin->mux, "batched-push-timeout", model_bin->mux_push_timeout);
  set_prop_if_exists (model_bin->mux, "live-source", model_bin->mux_live_source);
  /* the model mux batches PIPELINE-device buffers (decoders/tee) -- its own surface
   * allocations must land on that device, not on default 0, or a non-zero pipeline
   * gpu gets a cross-device mux. (Shard placement is handled by conv-in, NOT here.) */
  set_prop_if_exists (model_bin->mux, "gpu-id", model_bin->mux_gpu);

  /* configure the inference element + wire its completion signal (backend-aware: the
   * 'model-updated' signal + model-engine-file override are nvinfer-only). */
  modelmux_infer_configure (model_bin, engine_file);

  gst_bin_add_many (GST_BIN (model_bin->bin), model_bin->mux, model_bin->infer, model_bin->demux, NULL);
  if (model_bin->conv_in)
    gst_bin_add (GST_BIN (model_bin->bin), model_bin->conv_in);
  if (model_bin->conv_out)
    gst_bin_add (GST_BIN (model_bin->bin), model_bin->conv_out);

  /* drop EOS at the model mux output: when this model's last stream is removed
   * (all sink pads released) the legacy nvstreammux would otherwise emit EOS
   * downstream and latch nvinfer/demux, so a re-added stream would never flow. */
  {
    GstPad *msrc = gst_element_get_static_pad (model_bin->mux, "src");
    if (msrc) {
      gst_pad_add_probe (msrc, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
          modelmux_drop_eos_probe, NULL, NULL);
      gst_object_unref (msrc);
    }
  }

  /* mux:src -> [conv-in] -> infer (static sink) -> [conv-out] -> demux (static sink) */
  {
    GstElement *head = model_bin->conv_in ? model_bin->conv_in : model_bin->infer;
    GstElement *tail = model_bin->conv_out ? model_bin->conv_out : model_bin->infer;
    gboolean ok = gst_element_link (model_bin->mux, head);
    if (ok && model_bin->conv_in)
      ok = gst_element_link (model_bin->conv_in, model_bin->infer);
    if (ok && model_bin->conv_out)
      ok = gst_element_link (model_bin->infer, model_bin->conv_out);
    ok = ok && gst_element_link (tail, model_bin->demux);
    if (!ok) {
      MM_ERR ("ModelBin %s/%s: static link mux->[conv]->infer->[conv]->demux failed",
          role, name);
      model_bin_free (model_bin);
      return NULL;
    }
  }

  gst_bin_add (GST_BIN (container), model_bin->bin);
  /* NOT synced to PLAYING here: the engine is warmed by model_bin_load()
   * (background) so creation never blocks. Status starts NOT_READY. */
  model_bin->status = MODEL_WARMING;

  /* Per-frame provenance user-meta on the nvinfer src pad -- ALWAYS attached (no flag,
   * regardless of unified-batch). It is the fundamental src->model attribution: when
   * different streams run different models (multiple primaries / per-sensor models),
   * this is the only way downstream knows which model produced which detections. Cheap
   * (one pooled meta + a small struct copy per frame). The debug overlay also reads it
   * at the combined mux; no display probe here. */
  {
    GstPad *isrc = gst_element_get_static_pad (model_bin->infer, "src");
    if (isrc) {
      gst_pad_add_probe (isrc, GST_PAD_PROBE_TYPE_BUFFER, modelmux_inference_provenance_probe, model_bin, NULL);
      gst_object_unref (isrc);
    }
  }

  /* perf metrics ONLY: stamp the batch arrival time at the nvinfer sink, so the src
   * provenance probe can derive per-batch inference latency. Installed only when enabled. */
  if (model_bin->perf_on) {
    GstPad *isnk = gst_element_get_static_pad (model_bin->infer, "sink");
    if (isnk) {
      gst_pad_add_probe (isnk, GST_PAD_PROBE_TYPE_BUFFER, modelmux_perf_infer_in_probe, model_bin, NULL);
      gst_object_unref (isnk);
    }
  }

  MM_INFO ("      - nvmodelbin created  [%s/%s]  uid=%u  max=%u  status=NOT_READY",
      role, name, unique_id, model_bin->max_streams);
  MM_INFO ("           config=%s", config_file);
  return model_bin;
}

ModelStatus
model_bin_status (ModelBin * model_bin)
{
  return model_bin ? (ModelStatus) g_atomic_int_get (&model_bin->status) : MODEL_WARMING;
}

/* Record a shard's device placement AND point its ingress device-migration element
 * at that GPU (the egress element stays targeted at the pipeline device). gpu=-1
 * keeps the config's device: the pair is left as seeded at create (per-version
 * registry placement, falling back to the config-file pin / pipeline device).
 * nvdsxfer pair: xfer-in OUTPUT/DESTINATION (gpu-id) = gpu;
 *                xfer-out PEER (p2p-gpu-id, its source side) = gpu.
 * Must run pre-first-buffer (both elements latch pools/sessions on first use). */
static void
model_bin_set_gpu (ModelBin * model_bin, gint gpu)
{
  if (!model_bin)
    return;
  if (gpu >= 0 && !model_bin->conv_in && gpu != model_bin->mux_gpu) {
    /* the bin was built DEVICE-LOCAL (no migration elements -- same-device
     * placement resolved at creation): a cross-device re-pin cannot be
     * honoured, and recording it anyway would make the scheduler route
     * gpu-addressed streams onto a shard that physically infers elsewhere.
     * Refuse loudly; the correct path is reloading the model with the gpu
     * (which derives the config and builds the bin WITH its xfer pair). */
    MM_ERR ("ModelBin %s/%s: cross-device re-pin to gpu %d REFUSED -- the bin "
        "was built device-local on gpu %d with no migration elements; reload "
        "the model with the target gpu instead", model_bin->role, model_bin->name, gpu,
        model_bin->mux_gpu);
    return;
  }
  model_bin->gpu = gpu;
  if (gpu < 0)
    return;
  model_bin->conv_gpu = gpu;
  /* re-point the nvdsxfer pair (the only migration transport): xfer-in's
   * DESTINATION and xfer-out's SOURCE PEER are the shard's device. */
  if (model_bin->conv_in)
    set_prop_if_exists (model_bin->conv_in, "gpu-id", gpu);      /* destination */
  if (model_bin->conv_out)
    set_prop_if_exists (model_bin->conv_out, "p2p-gpu-id", gpu); /* source peer */
}

/* The ACTUAL inference device for this bin: read the backend element's gpu-id
 * (nvinfer and nvinferserver both expose it), so a notification honours the
 * config-file / derived placement (request -> registry -> config gpu-id) rather
 * than a hardcoded default. Falls back to the mux device if the prop is absent. */
static gint
model_bin_infer_gpu (ModelBin * model_bin)
{
  guint g = 0;
  if (model_bin && model_bin->infer &&
      g_object_class_find_property (G_OBJECT_GET_CLASS (model_bin->infer), "gpu-id")) {
    g_object_get (model_bin->infer, "gpu-id", &g, NULL);
    return (gint) g;
  }
  return model_bin ? model_bin->mux_gpu : -1;
}

/* background warm-up: bring the bin to the pipeline state so nvinfer builds /
 * deserializes the TRT engine, then flip the status to READY. Role-agnostic. */
static gpointer
model_warmup_thread (gpointer data)
{
  ModelBin *model_bin = (ModelBin *) data;
  GstState st = GST_STATE_NULL;

  /* START WARMING: bring this model bin up to the (already-PLAYING) pipeline's
   * state. The NULL->READY->PAUSED transition is what makes nvinfer build/
   * deserialize the TRT engine -- i.e. the engine begins warming right here. */
  gst_element_sync_state_with_parent (model_bin->bin);

  /* BOUNDED wait (was GST_CLOCK_TIME_NONE): generous enough for a cold TensorRT
   * engine build, but finite -- an infinitely wedged element load would otherwise
   * pin this bin WARMING forever (unload refused) and hang every teardown path
   * that joins this thread (pipeline NULL / process shutdown). */
  gst_element_get_state (model_bin->bin, &st, NULL, 30 * 60 * GST_SECOND);

  if (st >= GST_STATE_PAUSED) {
    gchar *eng = NULL, *old;
    /* Cache the engine path for the per-frame provenance probe (no per-frame
     * g_object_get). Published BEFORE the WARMED flip so a consumer never sees
     * READY with a stale engine pointer. */
    eng = get_infer_engine_str (model_bin);   /* backend-aware (nvinferserver has no engine prop) */
    /* RETIRE (never free) the old string: an in-flight frame may still hold it
     * in the lock-free provenance probe; deferred-free avoids a use-after-free. */
    old = (gchar *) g_atomic_pointer_get (&model_bin->engine);
    g_atomic_pointer_set (&model_bin->engine, eng);
    /* must precede the READY status flip, consistent with the OTA reload path */
    modelmux_bin_retire_str (model_bin, old);
    g_atomic_int_set (&model_bin->status, MODEL_WARMED);
    MM_INFO ("      - nvmodelbin [%s/%s]  engine warmed  ->  READY", model_bin->role, model_bin->name);
    /* app-facing completion notice: the engine is ACTUALLY warmed + ready now */
    modelmux_post_model_event (model_bin->bin, "model-loaded", model_bin->name,
        model_bin->version, model_bin_infer_gpu (model_bin), TRUE, NULL);
  } else {
    /* engine build/deserialize failed -> FAILED (not silently NOT_READY): the
     * bin is now reloadable (model_bin_load) and unloadable so a
     * bad load is recoverable without restarting the app. */
    g_atomic_int_set (&model_bin->status, MODEL_FAILED);
    MM_ERR ("ModelBin %s/%s: engine load FAILED (state=%d) -> status FAILED "
        "(reload via model/load or remove via model/unload)",
        model_bin->role, model_bin->name, st);
    modelmux_post_model_event (model_bin->bin, "model-loaded", model_bin->name,
        model_bin->version, model_bin_infer_gpu (model_bin), FALSE,
        "engine build/deserialize failed");
  }
  return NULL;
}

/* Kick off the model's engine warm-up in the BACKGROUND. Does not build any
 * elements (model_bin_new already did) and does NOT block: it spawns a worker
 * thread that brings the bin to PAUSED so nvinfer builds/deserializes the TRT
 * engine, then flips the status to WARMED/FAILED. Idempotent -- a warm already
 * in flight or already done is left alone; only a FAILED bin is retried. */
void
model_bin_load (ModelBin * model_bin)
{
  if (!model_bin)
    return;
  if (model_bin->load_thread) {
    /* A prior load that FAILED can be retried: join the finished thread and
     * relaunch. A load still in flight (NOT_READY) or already done (READY/
     * SERVING) is left alone -- idempotent. */
    if (g_atomic_int_get (&model_bin->status) == MODEL_FAILED) {
      g_thread_join (model_bin->load_thread);     /* already exited (FAILED set at end) */
      model_bin->load_thread = NULL;
    } else {
      return;
    }
  }
  g_atomic_int_set (&model_bin->status, MODEL_WARMING);
  /* spawn the background warm-up worker (model_warmup_thread): syncs the bin's
   * state with the pipeline so nvinfer builds/deserializes the engine off the
   * caller's thread, so creation/load never stalls the control plane. */
  model_bin->load_thread = g_thread_new ("model-warmup", model_warmup_thread, model_bin);
}

gboolean
model_bin_has_capacity (ModelBin * model_bin)
{
  return model_bin && !g_queue_is_empty (model_bin->free_idx);
}

guint
model_bin_num_streams (ModelBin * model_bin)
{
  return model_bin ? model_bin->num_streams : 0;
}

static GstPadProbeReturn
modelmux_zero_stream_drop_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  (void) pad;
  (void) info;
  (void) udata;
  return GST_PAD_PROBE_DROP;
}

static void
model_bin_install_zero_stream_drop (ModelBin * model_bin)
{
  GstPad *src;

  if (!model_bin || !model_bin->infer || model_bin->zero_stream_drop_probe)
    return;
  src = gst_element_get_static_pad (model_bin->infer, "src");
  if (!src)
    return;
  model_bin->zero_stream_drop_probe = gst_pad_add_probe (src,
      GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_BUFFER_LIST |
      GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
      modelmux_zero_stream_drop_probe, NULL, NULL);
  if (model_bin->zero_stream_drop_probe) {
    model_bin->zero_stream_drop_pad = src;       /* keep the pad alive until probe removal */
    MM_INFO ("      - nvmodelbin [%s/%s]  zero-stream fence armed on nvinfer src",
        model_bin->role, model_bin->name);
  } else {
    gst_object_unref (src);
  }
}

static void
model_bin_remove_zero_stream_drop (ModelBin * model_bin)
{
  if (!model_bin || !model_bin->zero_stream_drop_probe)
    return;
  if (model_bin->zero_stream_drop_pad)
    gst_pad_remove_probe (model_bin->zero_stream_drop_pad, model_bin->zero_stream_drop_probe);
  model_bin->zero_stream_drop_probe = 0;
  if (model_bin->zero_stream_drop_pad) {
    gst_object_unref (model_bin->zero_stream_drop_pad);
    model_bin->zero_stream_drop_pad = NULL;
  }
  MM_INFO ("      - nvmodelbin [%s/%s]  zero-stream fence disarmed",
      model_bin->role, model_bin->name);
}

/* Attach a stream: request a mux sink + demux src for a recycled index,
 * ghost both on the bin, and return the per-stream demux src ghost. */
GstPad *
model_bin_attach (ModelBin * model_bin, guint stream_id)
{
  gint idx;
  gchar pname[64];
  GstPad *mux_sink, *demux_src, *ghost_sink, *ghost_src;

  if (!model_bin_has_capacity (model_bin)) {
    MM_WARN ("ModelBin %s/%s full (max=%u)", model_bin->role, model_bin->name, model_bin->max_streams);
    return NULL;
  }
  /* a drain-recovery recreate failure can leave mux/demux NULL (status FAILED). Reject
   * up front rather than passing NULL to gst_element_request_pad_simple (g_critical). */
  if (!model_bin->mux || !model_bin->demux) {
    MM_ERR ("ModelBin %s/%s: mux/demux unavailable (recreate failed) -- cannot attach",
        model_bin->role, model_bin->name);
    return NULL;
  }
  idx = GPOINTER_TO_INT (g_queue_pop_head (model_bin->free_idx));

  g_snprintf (pname, sizeof (pname), "sink_%u", idx);
  mux_sink = gst_element_request_pad_simple (model_bin->mux, pname);
  g_snprintf (pname, sizeof (pname), "src_%u", idx);
  demux_src = gst_element_request_pad_simple (model_bin->demux, pname);
  if (!mux_sink || !demux_src) {
    MM_ERR ("ModelBin %s/%s: failed to request pads for idx %d",
        model_bin->role, model_bin->name, idx);
    if (mux_sink) {
      gst_element_release_request_pad (model_bin->mux, mux_sink);
      gst_object_unref (mux_sink);     /* release does not consume the request ref */
    }
    if (demux_src) {
      gst_element_release_request_pad (model_bin->demux, demux_src);
      gst_object_unref (demux_src);
    }
    g_queue_push_head (model_bin->free_idx, GINT_TO_POINTER (idx));
    return NULL;
  }

  /* ghost the mux sink and the demux src onto the bin boundary */
  g_snprintf (pname, sizeof (pname), "sink_%u", idx);
  ghost_sink = gst_ghost_pad_new (pname, mux_sink);
  g_snprintf (pname, sizeof (pname), "src_%u", idx);
  ghost_src = gst_ghost_pad_new (pname, demux_src);
  if (!ghost_sink || !ghost_src) {
    MM_ERR ("ModelBin %s/%s: failed to create ghost pads for idx %d",
        model_bin->role, model_bin->name, idx);
    if (ghost_sink) gst_object_unref (ghost_sink);
    if (ghost_src)  gst_object_unref (ghost_src);
    gst_element_release_request_pad (model_bin->mux, mux_sink);
    gst_object_unref (mux_sink);
    gst_element_release_request_pad (model_bin->demux, demux_src);
    gst_object_unref (demux_src);
    g_queue_push_head (model_bin->free_idx, GINT_TO_POINTER (idx));
    return NULL;
  }
  gst_pad_set_active (ghost_sink, TRUE);
  gst_pad_set_active (ghost_src, TRUE);
  gst_element_add_pad (model_bin->bin, ghost_sink);
  gst_element_add_pad (model_bin->bin, ghost_src);

  gst_object_unref (mux_sink);
  gst_object_unref (demux_src);

  g_hash_table_insert (model_bin->idx_to_stream, GINT_TO_POINTER (idx),
      GINT_TO_POINTER ((gint) stream_id));
  if (model_bin->slot_stream && idx >= 0 && (guint) idx < model_bin->max_streams)
    g_atomic_int_set (&model_bin->slot_stream[idx], (gint) stream_id);
  model_bin->num_streams++;
  model_bin->served = TRUE;                                   /* mark: has carried traffic */
  g_atomic_int_set (&model_bin->status, MODEL_SERVING);     /* now serving stream(s) */

  MM_INFO ("      - nvmodelbin [%s/%s]  attach stream %u @idx%d  (n=%u)  ->  SERVING",
      model_bin->role, model_bin->name, stream_id, idx, model_bin->num_streams);
  return ghost_src;            /* per-stream output of this model */
}

/* DRAIN-TO-ZERO RECOVERY for a ModelBin's inner mux+demux.
 *
 * When a ModelBin's last stream leaves we keep the bin warm (engine stays loaded in
 * nvinfer), but the legacy nvstreammux/nvstreamdemux retain stale drained state. A
 * PLAYING->READY->PLAYING cycle does NOT clear it: the next stream's first buffer
 * passes and then the mux stops pulling -- verified by the per-stream taps (a refill
 * shows branch-in=1, prim-out=1, out-mux-src=1, then nothing, and the upstream source
   * push thread wedges on the backpressure). Testing ruled out the live refill flush as
   * the required recovery mechanism, confirming the stale muxer state is the
 * culprit. Rebuilding the two muxer elements from scratch gives them zero stale state;
 * nvinfer is left untouched so the (warm) engine is NOT re-deserialized.
 *
 * Rebuilt link:  mux.src -> infer.sink ,  infer.src -> demux.sink. At zero streams all
 * per-stream request pads (mux sink_N / demux src_N) are released, but nvinfer can
 * still have a late output in its internal output thread. The detach path arms a
 * temporary infer.src drop fence before releasing the last pads, and the next attach
 * removes it only after the fresh branch is fully linked. nvinfer's provenance probe
 * lives on its OWN src pad and survives; the per-stream prim-out tap is re-added on
 * the next attach. Reuses the exact element names so downstream identity / debug
 * labels are unchanged. */
static gboolean
modelmux_modelbin_recreate_mux_demux (ModelBin * model_bin)
{
  GstElement *fresh_mux, *fresh_demux;
  gchar *mux_name, *demux_name;
  GstPad *msrc;

  if (!model_bin->mux || !model_bin->infer || !model_bin->demux)
    return FALSE;

  mux_name = gst_element_get_name (model_bin->mux);
  demux_name = gst_element_get_name (model_bin->demux);
  fresh_mux = create_gst_element ("nvstreammux", mux_name);
  fresh_demux = create_gst_element ("nvstreamdemux", demux_name);
  g_free (mux_name);
  g_free (demux_name);
  if (!fresh_mux || !fresh_demux) {
    if (fresh_mux)   gst_object_unref (fresh_mux);
    if (fresh_demux) gst_object_unref (fresh_demux);
    MM_ERR ("      - nvmodelbin [%s/%s] drain-recovery: could not create fresh "
        "mux/demux -- keeping the old ones", model_bin->role, model_bin->name);
    return FALSE;
  }
  set_prop_if_exists (fresh_mux, "batch-size", model_bin->max_streams);
  set_prop_if_exists (fresh_mux, "width", model_bin->mux_width);
  set_prop_if_exists (fresh_mux, "height", model_bin->mux_height);
  set_prop_if_exists (fresh_mux, "batched-push-timeout", model_bin->mux_push_timeout);
  set_prop_if_exists (fresh_mux, "live-source", model_bin->mux_live_source);
  set_prop_if_exists (fresh_mux, "gpu-id", model_bin->mux_gpu);

  /* drop the old elements: NULL, unlink from the (kept) inner chain, remove (unref).
   * The conv-in/conv-out device-migration converters are KEPT with nvinfer (they hold
   * no per-stream state; only the legacy mux/demux wedge after a drain-to-zero). */
  {
    GstElement *head = model_bin->conv_in ? model_bin->conv_in : model_bin->infer;
    GstElement *tail = model_bin->conv_out ? model_bin->conv_out : model_bin->infer;
    gst_element_set_state (model_bin->mux, GST_STATE_NULL);
    gst_element_set_state (model_bin->demux, GST_STATE_NULL);
    gst_element_unlink (model_bin->mux, head);
    gst_element_unlink (tail, model_bin->demux);
    gst_bin_remove_many (GST_BIN (model_bin->bin), model_bin->mux, model_bin->demux, NULL);
    model_bin->mux = fresh_mux;
    model_bin->demux = fresh_demux;
    gst_bin_add_many (GST_BIN (model_bin->bin), model_bin->mux, model_bin->demux, NULL);

    /* reinstall the mux-src EOS-drop probe (see model_bin_new) */
    msrc = gst_element_get_static_pad (model_bin->mux, "src");
    if (msrc) {
      gst_pad_add_probe (msrc, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
          modelmux_drop_eos_probe, NULL, NULL);
      gst_object_unref (msrc);
    }
    /* relink mux.src -> [conv-in] -> infer -> [conv-out] -> demux.sink
     * (inner chain untouched, engine warm) */
    if (!gst_element_link (model_bin->mux, head) ||
        !gst_element_link (tail, model_bin->demux)) {
    MM_ERR ("      - nvmodelbin [%s/%s] drain-recovery: relink mux->infer->demux "
        "failed -- bin is BROKEN and will not accept new streams",
        model_bin->role, model_bin->name);
    /* remove the unlinked fresh elements; mark bin unusable so callers don't
     * try to request pads on a broken mux/demux chain */
    gst_element_set_state (model_bin->mux,   GST_STATE_NULL);
    gst_element_set_state (model_bin->demux, GST_STATE_NULL);
    gst_bin_remove_many (GST_BIN (model_bin->bin), model_bin->mux, model_bin->demux, NULL);
    model_bin->mux = NULL;
    model_bin->demux = NULL;
    g_atomic_int_set (&model_bin->status, MODEL_FAILED);
    return FALSE;
    }
  }
  gst_element_sync_state_with_parent (model_bin->mux);
  gst_element_sync_state_with_parent (model_bin->demux);
  gst_element_get_state (model_bin->mux, NULL, NULL, GST_SECOND);
  gst_element_get_state (model_bin->demux, NULL, NULL, GST_SECOND);
  MM_INFO ("      - nvmodelbin [%s/%s] drain-to-zero: rebuilt inner mux+demux "
      "(nvinfer kept warm) so the next stream flows continuously", model_bin->role, model_bin->name);
  return TRUE;
}

/* reverse lookup idx for stream_id */
static gint
model_idx_of (ModelBin * model_bin, guint stream_id)
{
  GHashTableIter it;
  gpointer k, v;
  g_hash_table_iter_init (&it, model_bin->idx_to_stream);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    if (GPOINTER_TO_INT (v) == (gint) stream_id)
      return GPOINTER_TO_INT (k);
  }
  return -1;
}

gboolean
model_bin_detach (ModelBin * model_bin, guint stream_id)
{
  gint idx = model_idx_of (model_bin, stream_id);
  gboolean draining_to_zero;
  gchar pname[64];
  GstPad *gp, *target;

  if (idx < 0)
    return FALSE;
  draining_to_zero = (model_bin->num_streams == 1);

  if (draining_to_zero)
    model_bin_install_zero_stream_drop (model_bin);

  g_snprintf (pname, sizeof (pname), "sink_%u", idx);
  gp = gst_element_get_static_pad (model_bin->bin, pname);
  if (gp) {
    target = gst_ghost_pad_get_target (GST_GHOST_PAD (gp));
    gst_pad_set_active (gp, FALSE);
    gst_element_remove_pad (model_bin->bin, gp);
    if (target) {
      gst_element_release_request_pad (model_bin->mux, target);
      gst_object_unref (target);
    }
    gst_object_unref (gp);
  }

  g_snprintf (pname, sizeof (pname), "src_%u", idx);
  gp = gst_element_get_static_pad (model_bin->bin, pname);
  if (gp) {
    target = gst_ghost_pad_get_target (GST_GHOST_PAD (gp));
    gst_pad_set_active (gp, FALSE);
    gst_element_remove_pad (model_bin->bin, gp);
    if (target) {
      gst_element_release_request_pad (model_bin->demux, target);
      gst_object_unref (target);
    }
    gst_object_unref (gp);
  }

  g_hash_table_remove (model_bin->idx_to_stream, GINT_TO_POINTER (idx));
  if (model_bin->slot_stream && idx >= 0 && (guint) idx < model_bin->max_streams)
    g_atomic_int_set (&model_bin->slot_stream[idx], -1);
  g_queue_push_tail (model_bin->free_idx, GINT_TO_POINTER (idx));   /* recycle */
  if (model_bin->num_streams)
    model_bin->num_streams--;
  /* engine stays warm; drop SERVING -> READY only when the last stream leaves */
  g_atomic_int_set (&model_bin->status,
      model_bin->num_streams ? MODEL_SERVING : MODEL_WARMED);

  MM_INFO ("      - nvmodelbin [%s/%s]  detach stream %u @idx%d  (n=%u)  ->  %s",
      model_bin->role, model_bin->name, stream_id, idx, model_bin->num_streams,
      model_bin->num_streams ? "SERVING" : "READY (kept warm)");

  /* DRAIN-TO-ZERO RECOVERY: when the last stream leaves, the legacy nvstreammux/
   * nvstreamdemux keep stale drained streaming state. A PLAYING->READY->PLAYING
   * cycle is NOT enough -- the next stream pushes exactly one buffer and then the
   * mux stops pulling. Rebuild the two muxer elements fresh instead; nvinfer is
   * kept warm (engine NOT re-deserialized). See modelmux_modelbin_recreate_mux_demux.
   * A late nvinfer output can still arrive from its output thread after the last
   * request pad is released, so the zero-stream fence above stays armed until the
   * next attach is fully linked.
   * SKIPPED while the owning bin is shutting down (modelmux_bin_free sets
   * shutting_down BEFORE detaching the remaining streams): the recovery rebuild
   * creates/links/syncs fresh elements inside a graph that is being torn down --
   * pointless work racing teardown. A limbo bin (pool==NULL) never has streams,
   * so modelmux_ib_of() covers every bin that can reach this drain-to-zero path. */
  if (model_bin->num_streams == 0 && model_bin->mux && model_bin->demux) {
    ModelMuxBin *modelmux_bin = modelmux_ib_of (model_bin);
    if (!modelmux_bin || !g_atomic_int_get (&modelmux_bin->shutting_down))
      modelmux_modelbin_recreate_mux_demux (model_bin);
  }
  return TRUE;
}

void
model_bin_free (ModelBin * model_bin)
{
  if (!model_bin)
    return;
  if (model_bin->load_thread) {        /* wait for any in-flight engine warm-up */
    g_thread_join (model_bin->load_thread);
    model_bin->load_thread = NULL;
  }
  model_bin_remove_zero_stream_drop (model_bin);
  /* the bin (with its nvinfer + the perf probes) goes to NULL below, so the lock-free
   * perf probes are detached before we free the perf state (no use-after-free). */
  if (model_bin->bin) {
    gst_element_set_state (model_bin->bin, GST_STATE_NULL);
    if (GST_OBJECT_PARENT (model_bin->bin))
      gst_bin_remove (GST_BIN (GST_OBJECT_PARENT (model_bin->bin)), model_bin->bin);
    else
      gst_object_unref (model_bin->bin);
  }
  if (model_bin->free_idx)
    g_queue_free (model_bin->free_idx);
  if (model_bin->idx_to_stream)
    g_hash_table_destroy (model_bin->idx_to_stream);
  g_free (model_bin->slot_stream);   /* probes quiesced by the NULL transition above */
  g_free (model_bin->name);
  /* model_bin->role is an interned static string (modelmux_role_intern) -- never freed */
  g_free (model_bin->config_file);
  g_free (model_bin->engine);                 /* the CURRENT (live) provenance strings */
  g_free (model_bin->version);
  g_free (model_bin->key);
  if (model_bin->retired)                     /* OTA-superseded strings */
    g_ptr_array_free (model_bin->retired, TRUE);
  g_free (model_bin->ota_old_version);        /* unconfirmed-OTA rollback stash */
  g_free (model_bin->ota_old_engine);
  g_free (model_bin->ota_old_config);
  g_free (model_bin->ota_old_key);
  g_free (model_bin->ota_new_key);
  g_mutex_clear (&model_bin->prov_lock);
  if (model_bin->perf)                        /* perf counters (probes already detached above) */
    modelmux_perf_free (model_bin->perf);
  g_free (model_bin);
}

/* forward decl: deferred (main-loop) ModelBin destruction -- defined with the
 * tracked-source machinery below. Use instead of model_bin_free on any path
 * that may run on a streaming thread or hold modelmux_bin->lock. */
static void modelmux_schedule_model_bin_free (ModelMuxBin * modelmux_bin, ModelBin * model_bin);

/* ================================================================== */
/* ModelPool                                                             */
/* ================================================================== */

ModelPool *
model_pool_new (const gchar * role, guint base_uid, guint max_streams,
    GstElement * parent_pipeline, GstElement * container, const ModelMuxConfig * config,
    const GHashTable * stream_names)
{
  ModelPool *rb = g_new0 (ModelPool, 1);
  rb->role = g_strdup (role);
  rb->base_uid = base_uid;
  rb->next_uid = base_uid;       /* monotonic: ++next_uid -> base+1, base+2, ... */
  rb->max_streams = max_streams;
  rb->parent_pipeline = parent_pipeline;
  rb->container = container;
  rb->config = config;
  rb->stream_names = stream_names;
  /* no default designated yet; gpu must NOT stay 0 from g_new0 (0 is a real device) */
  rb->default_ref.gpu = MM_GPU_ANY;
  rb->models = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  return rb;
}

/* Effective per-instance batch/stream cap for a (bare, ver) instance:
 *   explicit request (model/load batch_size)          -- strongest
 *   > version registry ([model-<name>-<ver>] batch-size= / an earlier load)
 *   > catalog name-level ([model-name-*] batch-size=; register_model already
 *     defaulted it to the global per-model-batch-size when absent)
 *   > global per-model-batch-size
 * ...clamped to `pool_cap` (the element's batch-size) with a VISIBLE warn --
 * an instance can never batch more streams than the element accepts. */
static guint
modelmux_effective_batch (const ModelMuxConfig * config, const ModelCatalogEntry * def,
    const gchar * bare, const gchar * ver, guint req_batch, guint pool_cap)
{
  guint mx = req_batch;
  if (!mx)
    mx = modelmux_config_version_batch (config, bare, ver ? ver : MM_MODEL_VERSION_DEFAULT);
  if (!mx && def)
    mx = def->max_streams;              /* embeds the global default (register_model) */
  if (!mx)
    mx = config->per_model_max_streams;
  if (mx > pool_cap) {
    MM_WARN ("model '%s@%s': batch %u exceeds the element's batch-size %u -- clamped "
        "(raise 'batch-size' to honour it)", bare ? bare : "?",
        ver ? ver : MM_MODEL_VERSION_DEFAULT, mx, pool_cap);
    mx = pool_cap;
  }
  return mx;
}

/* find or lazily create a ModelBin for this role */
static ModelBin *
model_pool_get_or_create (ModelPool * model_pool, const gchar * model_name,
    const gchar * cfg_override, guint req_batch)
{
  /* NOTE: req_batch applies only when this call CREATES the bin -- an existing
   * or limbo-promoted bin keeps its construction-time batch (a live version's
   * batch is immutable; the control plane guards mismatched requests). */
  ModelBin *model_bin = g_hash_table_lookup (model_pool->models, model_name);
  const ModelCatalogEntry *def;
  guint uid, mx;

  if (model_bin)
    return model_bin;

  /* This is the SINGLE choke point every attach/swap/warm flows through. The one
   * "reuse instead of build" rule that is SAFE lives here: promoting a type-less limbo
   * model (never served -> clean) into a role. We deliberately do NOT reuse a bin that
   * already SERVED then drained to 0 streams in the other pool -- DeepStream's legacy
   * nvstreammux does not reliably resume after a mid-pipeline drain-to-zero, so such
   * reuse stalls. Redundant warm copies are reclaimed by dedup instead. */
  {
    ModelMuxBin *modelmux_bin = (ModelMuxBin *) model_pool->owner;

    /* PROMOTE a type-less limbo model into THIS pool on first use. The bin is
     * already warmed in the shared container; promotion is pure bookkeeping (move the
     * map entry + set role/pool), so no element/pad changes and ZERO frame loss. */
    ModelBin *lb = (modelmux_bin && modelmux_bin->limbo_models) ?
        g_hash_table_lookup (modelmux_bin->limbo_models, model_name) : NULL;
    if (lb) {
      ModelBin *s;
      g_hash_table_remove (modelmux_bin->limbo_models, model_name);   /* frees key only; bin lives */
      /* the WHOLE shard chain rides with the base map entry -- role/pool are
       * per shard (an gpus[] group pre-creates limbo siblings). */
      for (s = lb; s; s = (ModelBin *) s->next_shard) {
        modelmux_role_set (s, model_pool->role);                     /* Unknown -> Primary/Shadow (interned) */
        s->pool = model_pool;
      }
      g_hash_table_insert (model_pool->models, g_strdup (lb->key), lb);  /* canonical key */
      MM_INFO ("    - ModelPool[%s]  PROMOTED '%s' (gie=%u) from limbo -> %s "
          "(type decided by first use)", model_pool->role, model_name, lb->unique_id,
          model_pool->role);
      return lb;
    }

  }

  /* model_name is the canonical "name@version" key: split it -- the CATALOG is keyed by
   * bare name (its config), the VERSION pins this instance. */
  {
    gchar *bare = NULL, *ver = NULL;
    model_key_split (model_name, &bare, &ver);
    def = modelmux_config_find_model (model_pool->config, bare);
    if (!def) {
      MM_ERR ("ModelPool %s: unknown model '%s' (not in catalog)", model_pool->role, bare);
      g_free (bare); g_free (ver);
      return NULL;
    }
    /* stable per-role unique-id: a MONOTONIC counter, never reused while the
     * process lives. Using the registry size would reuse a uid after an
     * unload+reload and collide with another live model's gie-unique-id. */
    uid = ++model_pool->next_uid;
    mx = modelmux_effective_batch (model_pool->config, def, bare, ver, req_batch,
        model_pool->max_streams);

    /* cfg_override (a per-version config derived with an engine swapped in -- e.g. a
     * config-default ref "name;version;engine") WINS over the catalog base; the engine is
     * baked into that derived config so engine_file is left NULL (mirrors model/load). */
    model_bin = model_bin_new (bare, model_pool->role,
        cfg_override ? cfg_override : def->config_file,
        cfg_override ? NULL : def->engine_file,
        ver, uid, mx, model_pool->config, model_pool->parent_pipeline, model_pool->container, model_pool->stream_names,
        0 /*base shard*/);
    g_free (bare); g_free (ver);
  }
  if (!model_bin)
    return NULL;
  model_bin->pool = model_pool;                       /* back-ptr: owning pool */
  g_hash_table_insert (model_pool->models, g_strdup (model_bin->key), model_bin);  /* canonical "name@version" */
  model_bin_load (model_bin);          /* warm the engine in the background -> READY */
  return model_bin;
}

/* ---- auto-sharding: a model is a CHAIN of ModelBins (base in rb->models, extras
 * via model_bin->next_shard). All shards share name/config/engine/gie-id; each batches up
 * to per-model-max-streams. New shards are created on overflow and torn down when an
 * EXTRA shard empties. Generic -- works for ANY pool (primary, shadow, ...).      */

/* block until the bin warms to READY/SERVING (engine is cached -> fast). Returns
 * FALSE on FAILED or timeout. The warm thread never takes modelmux_bin->lock, so this is
 * safe to call under it; only THIS new (not-yet-linked) shard's wiring waits --
 * existing streams keep flowing on their own threads (zero-drop). */
static gboolean
model_bin_wait_ready (ModelBin * model_bin, guint timeout_ms)
{
  guint waited = 0;
  for (;;) {
    ModelStatus st = model_bin_status (model_bin);
    if (st == MODEL_WARMED || st == MODEL_SERVING)
      return TRUE;
    if (st == MODEL_FAILED || waited >= timeout_ms)
      return FALSE;
    g_usleep (20 * 1000);                /* 20 ms poll */
    waited += 20;
  }
}

static gint modelmux_shard_gpu_effective (ModelMuxBin * modelmux_bin, ModelBin * model_bin);

/* Pick a warm shard with a free slot, restricted to the requested device when
 * gpu != MM_GPU_ANY; NULL if none qualifies. For unconstrained attaches the
 * [multimodel] placement-policy decides how a multi-GPU instance group fills:
 *   PACK (default): FIRST FIT in chain order (= gpus[] order) -- saturate
 *     gpu[0]'s shard to max-streams before gpu[1] sees traffic. Maximizes batch
 *     efficiency, keeps later GPUs free, and matches the overflow chains' own
 *     packing (which lets compaction reclaim drained shards).
 *   SPREAD: LEAST-LOADED eligible shard, PINNED members preferred over overflow
 *     clones (a spill shard must drain and be reclaimed, not be refilled
 *     forever). Latency-first distribution across the placed devices.
 * Plain overflow chains (no pinned member) always pack, whatever the policy.
 * Caller holds modelmux_bin->lock (effective-gpu reads the version registry). */
static ModelBin *
modelmux_pool_pick_shard (ModelPool * rb, ModelBin * base, gint gpu)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) rb->owner;
  ModelBin *model_bin, *best = NULL;
  gboolean spread = FALSE;
  guint best_n = 0;
  if (rb->config && rb->config->placement_policy == SPREAD_PLACEMENT) {
    for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard)
      if (model_bin->pinned) {
        spread = TRUE;                   /* multi-GPU group + spread policy */
        break;
      }
  }
  for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
    ModelStatus st = model_bin_status (model_bin);
    guint n;
    if ((st != MODEL_WARMED && st != MODEL_SERVING) ||
        !model_bin_has_capacity (model_bin) ||
        (gpu != MM_GPU_ANY && modelmux_shard_gpu_effective (modelmux_bin, model_bin) != gpu))
      continue;
    if (!spread)
      return model_bin;                         /* PACK: first fit in chain order */
    n = model_bin_num_streams (model_bin);
    if (!best || (model_bin->pinned && !best->pinned) ||
        (model_bin->pinned == best->pinned && n < best_n)) {
      best = model_bin;
      best_n = n;
    }
  }
  return best;
}

/* process-wide shard element-name allocator: every non-base shard (overflow OR
 * gpus[] pre-create) draws a unique inst so GStreamer element names never
 * collide across pools/limbo. */
static guint
modelmux_shard_next_inst (void)
{
  static volatile gint s_inst = 0;
  return (guint) g_atomic_int_add (&s_inst, 1) + 1;
}

/* AUTO-GPU-SCALE device picker: the healthiest OTHER device that is IDENTICAL to
 * the template's (same compute capability AND device name -- the condition under
 * which the existing TensorRT engine legally deserializes there; a mismatched
 * device would silently trigger a minutes-long rebuild). Healthiest = most free
 * VRAM. Returns -1 when there is no such device (single GPU / heterogeneous /
 * CUDA query failure) -- the caller then grows on the template's own device. */
static gint
modelmux_pool_pick_scale_gpu (ModelMuxBin * modelmux_bin, ModelBin * tmpl)
{
  struct cudaDeviceProp tprop, prop;
  int count = 0, dev;
  gint tmpl_gpu = modelmux_shard_gpu_effective (modelmux_bin, tmpl);
  gint pipe_gpu = modelmux_bin->config ? (gint) modelmux_bin->config->gpu : 0;
  gint best = -1;
  guint best_free = 0;

  if (cudaGetDeviceCount (&count) != cudaSuccess || count < 2)
    return -1;
  if (cudaGetDeviceProperties (&tprop, tmpl_gpu) != cudaSuccess)
    return -1;
  for (dev = 0; dev < count; dev++) {
    guint used = 0, total = 0;
    int ab = 0, ba = 0;
    if (dev == tmpl_gpu)
      continue;
    if (cudaGetDeviceProperties (&prop, dev) != cudaSuccess)
      continue;
    if (prop.major != tprop.major || prop.minor != tprop.minor ||
        strncmp (prop.name, tprop.name, sizeof (prop.name)) != 0)
      continue;                          /* engine NOT portable -> never auto-place */
    /* the candidate's shard needs an nvdsxfer pair to the PIPELINE device --
     * pre-check bidirectional peer access here so we never pick a device the
     * bin-creation admission would then (noisily) reject. A candidate that IS
     * the pipeline device needs no migration at all -> no P2P requirement. */
    if (dev != pipe_gpu &&
        (cudaDeviceCanAccessPeer (&ab, pipe_gpu, dev) != cudaSuccess ||
         cudaDeviceCanAccessPeer (&ba, dev, pipe_gpu) != cudaSuccess ||
         !ab || !ba))
      continue;
    /* health via NVML (modelmux_gpu_stats, PCI-bus-id matched): unlike
     * cudaMemGetInfo this creates NO CUDA context on the candidate (which
     * would block ~100ms-1s under modelmux_bin->lock, pin hundreds of MB of VRAM
     * per device forever, and skew the very number being measured) and
     * never touches this thread's current device. */
    if (!modelmux_gpu_stats (dev, NULL, &used, &total) || total == 0)
      continue;
    if (best < 0 || (total - used) > best_free) {
      best = dev;
      best_free = total - used;
    }
  }
  return best;
}

/* create + warm a new shard of base's model, append to the chain, return it (READY).
 * Shares base's gie-unique-id (downstream identity); unique element-name inst.
 * cfg_override/gpu (auto-gpu-scale): a per-device DERIVED config + placement for
 * an elastic cross-device overflow shard -- NULL/-1 = clone the template as-is. */
static ModelBin *
modelmux_pool_grow_shard (ModelPool * rb, ModelBin * base,
    const gchar * cfg_override, gint gpu)
{
  const ModelCatalogEntry *def = modelmux_config_find_model (rb->config, base->name);
  ModelBin *model_bin, *tail;
  guint inst = modelmux_shard_next_inst ();
  const gchar *use_cfg = cfg_override ? cfg_override : base->config_file;
  /* engine for the clone: ONLY the catalog engine, and ONLY when the clone
   * still runs the catalog BASE config. A derived/custom config has its
   * engine baked in -- an explicit model-engine-file would OVERRIDE it
   * with the catalog artifact (wrong checkpoint, or a TRT engine built for
   * another device that fails/rebuilds on this one). */
  const gchar *eng = (def && def->config_file &&
      g_strcmp0 (use_cfg, def->config_file) == 0)
      ? def->engine_file : NULL;

  model_bin = model_bin_new (base->name, rb->role, use_cfg,
      eng, base->version,
      base->unique_id /*SHARED gie-id*/, base->max_streams,
      rb->config, rb->parent_pipeline, rb->container, rb->stream_names, inst);
  if (!model_bin)
    return NULL;
  model_bin->pool = rb;
  /* overflow shard runs its config's device: the template's, or -- elastic
   * scale-out -- the auto-picked device (its derived config already pins it;
   * set_gpu records placement for the scheduler + migration pair). UNPINNED
   * either way: compaction reclaims it when it drains. */
  model_bin_set_gpu (model_bin, gpu >= 0 ? gpu : base->gpu);
  model_bin_load (model_bin);                 /* async warm (cached engine -> ~fast) */
  if (!model_bin_wait_ready (model_bin, 10000)) {
    MM_ERR ("ModelPool[%s]: new shard of '%s' failed to warm -- dropping it",
        rb->role, base->name);
    /* deferred: freeing inline would g_thread_join a load thread that may still be
     * building the engine, for minutes, while the caller holds modelmux_bin->lock. */
    modelmux_schedule_model_bin_free ((ModelMuxBin *) rb->owner, model_bin);
    return NULL;
  }
  for (tail = base; tail->next_shard; tail = (ModelBin *) tail->next_shard)
    ;
  tail->next_shard = model_bin;
  MM_INFO ("    - ModelPool[%s]  model '%s' GREW shard #%u (gie=%u, max=%u/shard) "
      "-- per-shard cap reached, scaling out", rb->role, base->name, inst,
      base->unique_id, base->max_streams);
  return model_bin;
}

/* SCHEDULER choke point: choose a shard for a new stream under an optional placement
 * constraint. gpu == MM_GPU_ANY -> any warm shard with room, else grow a new one;
 * gpu >= 0 -> only shards whose effective device matches, and scale-out clones the
 * chain member already ON that device (so an overflow shard inherits the right
 * per-GPU derived config, engine and placement).
 * Returns NULL if the model can't be created/warmed, has NO instance on the requested
 * device, OR (model-sharding=0) the per-model batch cap is full -- in that overflow
 * case *cap_overflow is set TRUE so the caller can fall back to a no-infer passthrough
 * instead of treating it as a hard failure.
 * @cap_overflow may be NULL (callers without passthrough support, e.g. swap/reroute). */
static ModelBin *
modelmux_pool_shard_for_attach (ModelPool * rb, const gchar * model_name,
    gboolean * cap_overflow, gint gpu)
{
  ModelBin *base = model_pool_get_or_create (rb, model_name, NULL, 0);
  ModelBin *shard;
  if (cap_overflow)
    *cap_overflow = FALSE;
  if (!base)
    return NULL;
  /* pick FIRST: any warm chain member with room can serve, even while the base
   * shard itself is still warming/failed (an gpus[] sibling on another
   * device may already be READY -- refusing on the base alone would strand it). */
  shard = modelmux_pool_pick_shard (rb, base, gpu);
  if (!shard) {
    ModelBin *tmpl = base;                 /* scale-out template: supplies config/engine/device */
    /* GROWING (below) does need a healthy template: cloning a warming/failed
     * config would replicate an unproven artifact. Gate only the scale-out. */
    ModelStatus bst = model_bin_status (base);
    if (bst != MODEL_WARMED && bst != MODEL_SERVING) {
      MM_WARN ("ModelPool[%s]: model '%s' is %s and no other shard has capacity; "
          "cannot attach yet", rb->role, model_name, model_status_str (bst));
      return NULL;
    }
    if (rb->config && !rb->config->model_sharding) {
      /* model-sharding disabled: nvinfer-compatible over-batch behaviour. The per-model
       * batch cap is full and we must NOT grow a new shard. Signal cap_overflow so the
       * caller routes this stream to the display mux WITHOUT inference (passthrough);
       * if the caller can't passthrough (NULL), it stays a plain refusal. */
      MM_WARN ("ModelPool[%s]: model '%s' at per-model batch cap and model-sharding=false "
          "-- routing stream to output WITHOUT inference (passthrough)", rb->role, model_name);
      if (cap_overflow)
        *cap_overflow = TRUE;
      return NULL;
    }
    if (gpu != MM_GPU_ANY) {
      /* constrained scale-out: clone the chain member already placed on the requested
       * device. A constraint with NO instance on that device is a refusal, not a
       * growth trigger -- the control plane validates gpu refs against the live
       * placement, so hitting this is a race/defensive error, never silent misplacement. */
      ModelMuxBin *modelmux_bin = (ModelMuxBin *) rb->owner;
      for (tmpl = base; tmpl; tmpl = (ModelBin *) tmpl->next_shard)
        if (modelmux_shard_gpu_effective (modelmux_bin, tmpl) == gpu)
          break;
      if (!tmpl) {
        MM_ERR ("ModelPool[%s]: model '%s' has NO instance on requested gpu %d -- "
            "refusing attach (load an instance on that device first)",
            rb->role, model_name, gpu);
        return NULL;
      }
      {
        /* the on-device template must itself be HEALTHY: cloning a WARMING or
         * FAILED shard's config would replicate an unproven/broken artifact
         * (and duplicate VRAM on a device whose instance never came up). */
        ModelStatus tst = model_bin_status (tmpl);
        if (tst != MODEL_WARMED && tst != MODEL_SERVING) {
          MM_WARN ("ModelPool[%s]: model '%s' instance on gpu %d is %s; "
              "cannot attach yet", rb->role, model_name, gpu,
              model_status_str (tst));
          return NULL;
        }
      }
    } else if (rb->config && rb->config->auto_gpu_scale) {
      /* ELASTIC AUTO-GPU-SCALE: every shard is full and the attach carries no
       * placement -- grow the overflow shard on the healthiest OTHER device,
       * but ONLY if that device is IDENTICAL to the template's (the picker
       * enforces compute-capability + name equality), so the CURRENT engine is
       * reused verbatim (fast deserialize, never a rebuild). The shard's config
       * is derived per-device (gpu-id/gpu_ids + model-engine-file baked in) and
       * it stays UNPINNED: compaction migrates its streams back and reclaims it
       * when load drops (scale-in). Any failure falls back to the same-device
       * grow below -- behaviour is then identical to auto-gpu-scale=0. */
      ModelMuxBin *modelmux_bin = (ModelMuxBin *) rb->owner;
      gint dev = modelmux_pool_pick_scale_gpu (modelmux_bin, base);
      if (dev >= 0) {
        const gchar *teng = (const gchar *)
            g_atomic_pointer_get (&base->engine);
        gchar *vtag = g_strdup_printf ("%s.g%d",
            model_version_str (base), dev);
        gchar *dcfg = modelmux_config_derive (base->config_file, teng, dev,
            base->name, vtag);
        g_free (vtag);
        if (dcfg) {
          MM_INFO ("    - ModelPool[%s]  AUTO-GPU-SCALE: '%s' at capacity on its "
              "device -> growing shard on IDENTICAL gpu %d (engine reused: %s)",
              rb->role, model_name, dev, teng ? teng : "from config");
          shard = modelmux_pool_grow_shard (rb, base, dcfg, dev);
          g_free (dcfg);
          if (shard)
            return shard;
          MM_WARN ("ModelPool[%s]: auto-gpu-scale grow of '%s' on gpu %d failed "
              "-- falling back to a same-device shard", rb->role, model_name, dev);
        }
      }
    }
    shard = modelmux_pool_grow_shard (rb, tmpl, NULL, -1);   /* all eligible shards full -> scale out */
  }
  return shard;
}

/* TRUE if `key` (a canonical "name@version") is a configured default (primary or shadow) -- such a
 * model must never be auto-freed or demoted: it backs the routing fallback and must stay live in
 * its pool. VERSION-AWARE: the configured default is (name, version), so a non-default VERSION that
 * merely shares a default's name is NOT protected (mirrors modelmux_pool_unload_if_idle's composite-key
 * compare; the prior bare-name check wrongly shielded e.g. Trafficcamnet@3 when @1 was the default). */
/* ============== runtime DEFAULT designation (pool-embedded DefaultModelRef) ==============
 * Each ModelPool carries a DefaultModelRef: "this pool holds these models, and THIS one
 * is the default". Identity strings (name/version/key) are authoritative; ref->bin is a
 * DERIVED CACHE into the inventory. All writers mutate under modelmux_bin->lock (the
 * assign/clear helpers are lock-free -- callers hold the lock or run pre-publication,
 * i.e. during bin construction before any other thread can see the pools). */

/* (Re)assign one role's default IDENTITY. gpu semantics: >= 0 persists the pin;
 * < 0 keeps the pin when the NAME is unchanged (OTA version rename) and resets
 * it to MM_GPU_ANY on an identity change (a pin belongs to its identity). */
static void
mm_default_ref_assign (DefaultModelRef * ref, const gchar * name,
    const gchar * version, gint gpu, ModelBin * bin)
{
  const gchar *ver = version ? version : MM_MODEL_VERSION_DEFAULT;
  /* SELF-ASSIGNMENT hardening: dup the inputs BEFORE freeing the old fields -- a
   * caller passing the ref's OWN name/version (e.g. "re-pin the current default")
   * must not have them freed out from under the strdup. */
  gchar *nname = g_strdup (name);
  gchar *nver = name ? g_strdup (ver) : NULL;
  gchar *nkey = name ? model_key (name, ver) : NULL;
  if (gpu >= 0)
    ref->gpu = gpu;
  else if (g_strcmp0 (ref->name, name) != 0)
    ref->gpu = MM_GPU_ANY;
  g_free (ref->name);    ref->name = nname;
  g_free (ref->version); ref->version = nver;
  g_free (ref->key);     ref->key = nkey;
  ref->bin = bin;
}

static void
mm_default_ref_clear (DefaultModelRef * ref)
{
  g_free (ref->name);    ref->name = NULL;
  g_free (ref->version); ref->version = NULL;
  g_free (ref->key);     ref->key = NULL;
  ref->gpu = MM_GPU_ANY;
  ref->bin = NULL;
}

/* Base shard for a default identity: role pool first, then the other pool (a
 * warm copy can serve cross-role after compaction), then limbo. Caller holds
 * modelmux_bin->lock. NULL when not loaded (the identity stands alone). */
static ModelBin *
mm_default_bin_lookup (ModelMuxBin * modelmux_bin, gboolean shadow, const gchar * key)
{
  ModelBin *model_bin = NULL;
  ModelPool *first = shadow ? modelmux_bin->shadow_pool : modelmux_bin->primary_pool;
  ModelPool *other = shadow ? modelmux_bin->primary_pool : modelmux_bin->shadow_pool;
  if (!key)
    return NULL;
  if (first && first->models)
    model_bin = g_hash_table_lookup (first->models, key);
  if (!model_bin && other && other->models)
    model_bin = g_hash_table_lookup (other->models, key);
  if (!model_bin && modelmux_bin->limbo_models)
    model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, key);
  return model_bin;
}

/* Destroy choke: a ModelBin about to be freed must not linger in either pool's
 * default-ref bin cache. Only the INSTANCE cache dies -- the identity strings
 * stay (a default whose bin is reclaimed is still the default; the next load
 * of that identity re-points the cache). NULL-safe on both arguments. */
static void
modelmux_bin_forget_default_bin (ModelMuxBin * modelmux_bin, ModelBin * model_bin)
{
  if (!modelmux_bin || !model_bin)
    return;
  if (modelmux_bin->primary_pool && modelmux_bin->primary_pool->default_ref.bin == model_bin)
    modelmux_bin->primary_pool->default_ref.bin = NULL;
  if (modelmux_bin->shadow_pool && modelmux_bin->shadow_pool->default_ref.bin == model_bin)
    modelmux_bin->shadow_pool->default_ref.bin = NULL;
}

const DefaultModelRef *
modelmux_bin_get_default (ModelMuxBin * ib, gboolean shadow)
{
  if (!ib)
    return NULL;
  return shadow ? &ib->shadow_pool->default_ref : &ib->primary_pool->default_ref;
}

static gboolean
modelmux_key_is_configured_default (ModelMuxBin * modelmux_bin, const gchar * key)
{
  if (!modelmux_bin || !key)
    return FALSE;
  return (modelmux_bin->primary_pool && modelmux_bin->primary_pool->default_ref.key &&
          g_strcmp0 (key, modelmux_bin->primary_pool->default_ref.key) == 0) ||
         (modelmux_bin->shadow_pool && modelmux_bin->shadow_pool->default_ref.key &&
          g_strcmp0 (key, modelmux_bin->shadow_pool->default_ref.key) == 0);
}

/* DEDUP-ON-IDLE: a base bin that just went idle (READY, 0 streams, single
 * shard) is REDUNDANT if a warm copy of the same model already lives elsewhere -- the
 * other pool or limbo. Two warm copies of one model is pure VRAM waste, so reclaim the
 * one that just idled and keep the survivor (still instantly reusable via rule (a)/(b)
 * of model_pool_get_or_create). Best-effort: it fires at idle transitions; it never
 * frees the LAST warm copy.
 *
 * Last-warm-copy policy (the lone idle bin, no twin):
 *   - config->compact_idle_models_across_pools == FALSE: keep it role-tagged in its pool
 *     (only the SAME role reuses it without a rebuild).
 *   - config->compact_idle_models_across_pools == TRUE (default): DEMOTE it to limbo (role-agnostic) so EITHER
 *     role's next use reuses it with no rebuild/re-warm. Safe because model_bin_detach
 *     already recreated this bin's inner mux+demux at 0 streams (nvinfer kept warm), so a
 *     later re-promotion flows zero-drop. Guarded: never demote a configured default,
 *     a non-READY (e.g. FAILED) bin, or a name already parked in limbo.
 *
 * Returns TRUE if `base` was reclaimed from `rb` (freed OR demoted to limbo) -- in both
 * cases the caller must not touch the rb->models entry afterwards. */
static gboolean
modelmux_pool_reclaim_if_redundant (ModelPool * rb, ModelBin * base)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) rb->owner;
  ModelPool *other;
  ModelBin *twin = NULL;

  if (!modelmux_bin || base->next_shard || model_bin_num_streams (base) != 0)
    return FALSE;                          /* not a lone, fully-idle base bin */

  /* a warm copy in the OTHER pool? */
  other = (rb == modelmux_bin->primary_pool) ? modelmux_bin->shadow_pool : (rb == modelmux_bin->shadow_pool) ? modelmux_bin->primary_pool : NULL;
  if (other)
    twin = g_hash_table_lookup (other->models, base->key);
  /* else a warm copy parked in limbo (loaded, unpromoted)? */
  if (!twin && modelmux_bin->limbo_models)
    twin = g_hash_table_lookup (modelmux_bin->limbo_models, base->key);

  if (!twin) {
    /* This is the only warm copy. Optionally demote it to limbo (role-agnostic) instead of
     * keeping it role-tagged -- see policy note above. All guards must hold; otherwise fall
     * back to the default (keep role-tagged). */
    if (modelmux_bin->config && modelmux_bin->config->compact_idle_models_across_pools && modelmux_bin->limbo_models &&
        model_bin_status (base) == MODEL_WARMED &&
        !modelmux_key_is_configured_default (modelmux_bin, base->key) &&
        !g_hash_table_contains (modelmux_bin->limbo_models, base->key)) {
      g_hash_table_remove (rb->models, base->key);    /* leave the role pool (frees key only) */
      modelmux_role_set (base, "Unknown");                  /* role-agnostic again (atomic store) */
      base->pool = NULL;                              /* limbo: not in any pool */
      g_hash_table_insert (modelmux_bin->limbo_models, g_strdup (base->key), base);
      MM_INFO ("    - ModelPool[%s]  model '%s' (gie=%u) lone-idle -> DEMOTED to limbo "
          "(role-agnostic, reusable by either role; compact-idle-models-across-pools)", rb->role,
          base->name, base->unique_id);
      return TRUE;                                    /* base relocated (NOT freed) */
    }
    return FALSE;                          /* keep it role-tagged (default behavior) */
  }
  {
    ModelStatus tst = model_bin_status (twin);
    if (tst != MODEL_WARMED && tst != MODEL_SERVING)
      return FALSE;                        /* twin not actually warm -> keep ours */
  }

  MM_INFO ("    - ModelPool[%s]  model '%s' (gie=%u) idle AND a warm copy already "
      "exists (gie=%u) -> RECLAIM this redundant bin (free VRAM)", rb->role,
      base->name, base->unique_id, twin->unique_id);
  g_hash_table_remove (rb->models, base->key);    /* frees key; bin freed below */
  /* deferred: reachable from the CLEAR probe (streaming thread, modelmux_bin->lock held) --
   * an inline free would tear down TensorRT on the streaming thread under the lock */
  modelmux_schedule_model_bin_free (modelmux_bin, base);
  return TRUE;
}

/* detach a stream from EVERY shard that holds it; tear down an EXTRA shard that
 * empties (immediate -- the base shard #0 is kept warm, unless DEDUP reclaims it).
 * Returns TRUE if found on at least one shard.
 * WHOLE-CHAIN WALK (not first-match): during a hitless cutover's drain window the
 * stream is legitimately attached to BOTH the source and the target shard -- a
 * teardown that detached only the first match would leave a PERMANENT phantom
 * attach on the other shard (num_streams never reaches 0 -> the shard is never
 * compacted, unload_model refuses forever = VRAM pin, and the stale slot can be
 * aliased by a recycled source id). */
static gboolean
modelmux_pool_detach_from_chain (ModelPool * rb, ModelBin * base, guint stream_id)
{
  ModelBin *model_bin, *prev = NULL, *next;
  gboolean found = FALSE;
  for (model_bin = base; model_bin; model_bin = next) {
    next = (ModelBin *) model_bin->next_shard;
    if (!model_bin_detach (model_bin, stream_id)) {
      prev = model_bin;
      continue;                            /* stream not on this shard */
    }
    found = TRUE;
    if (model_bin != base && !model_bin->pinned && model_bin_num_streams (model_bin) == 0) {
      /* extra OVERFLOW shard drained -> unlink + destroy immediately (frees VRAM);
       * it is recreated on demand. base shard #0 is never auto-destroyed, and a
       * PINNED gpus[] shard is desired state -- it stays warm on its GPU. */
      if (prev)
        prev->next_shard = model_bin->next_shard;
      MM_INFO ("    - ModelPool[%s]  model '%s' shard #%u empty -> UNLOADED "
          "(scaling in)", rb->role, model_bin->name, model_bin->inst);
      /* deferred: reachable from streaming-thread probes with modelmux_bin->lock held.
       * prev stays: model_bin is unlinked from the chain. */
      modelmux_schedule_model_bin_free ((ModelMuxBin *) rb->owner, model_bin);
    } else if (model_bin == base && model_bin_num_streams (model_bin) == 0) {
      /* base just went idle -> reclaim it if a warm twin exists elsewhere. If it WAS
       * reclaimed the whole chain is gone (a lone idle base has no extra shards) --
       * stop the walk. */
      if (modelmux_pool_reclaim_if_redundant (rb, base))
        return TRUE;
      prev = model_bin;
    } else {
      prev = model_bin;
    }
  }
  return found;
}

/* Pre-create + warm a model so it is READY before streams arrive (any role). */
gboolean
model_pool_load_model (ModelPool * rb, const gchar * model_name,
    const gchar * cfg_override, guint batch)
{
  ModelBin *model_bin = model_pool_get_or_create (rb, model_name, cfg_override, batch);
  if (!model_bin)
    return FALSE;
  /* a previously-FAILED bin is retried here: get-or-create returns the existing
   * bin without re-warming, so re-issue the load (recovery). For a fresh
   * bin get_or_create already kicked off the warm-up; model_bin_load is a
   * no-op while loading/loaded. */
  if (model_bin_status (model_bin) == MODEL_FAILED)
    model_bin_load (model_bin);
  return TRUE;
}

/* Attach: in_pad(tee src) -> in_q -> ModelBin -> out_q ; return out_q src.
 * On success the per-stream queues are handed back via in_q_out/out_q_out so
 * the caller can track and tear them down. */
GstPad *
model_pool_attach (ModelPool * rb, guint stream_id, const gchar * model_name,
    gint gpu, GstPad * in_pad, guint branch_serial,
    GstElement ** in_q_out, GstElement ** out_q_out,
    GstElement ** conv_out, gboolean * cap_overflow,
    ModelBin ** attached_bin_out)
{
  ModelBin *model_bin;
  GstElement *in_q, *out_q, *conv = NULL;
  GstPad *model_src, *q_in_sink, *q_in_src, *mb_sink, *q_out_sink, *q_out_src;
  gboolean linked;
  gchar nm[96];

  if (in_q_out)  *in_q_out = NULL;
  if (out_q_out) *out_q_out = NULL;
  if (conv_out)  *conv_out = NULL;
  if (attached_bin_out) *attached_bin_out = NULL;
  /* pick a warm shard with a free slot, or auto-create one on overflow (zero-drop:
   * existing streams untouched; the new shard warms its cached engine, then wires).
   * With model-sharding=0 an over-cap attach returns NULL + *cap_overflow=TRUE so the
   * caller can passthrough this stream to the display mux without inference. */
  model_bin = modelmux_pool_shard_for_attach (rb, model_name, cap_overflow, gpu);
  if (!model_bin)
    return NULL;

  g_snprintf (nm, sizeof (nm), "%s-inq-%s-%u-%u", rb->role, model_name,
      stream_id, branch_serial);
  in_q = create_gst_element ("queue", nm);
  g_snprintf (nm, sizeof (nm), "%s-outq-%s-%u-%u", rb->role, model_name,
      stream_id, branch_serial);
  out_q = create_gst_element ("queue", nm);
  if (!in_q || !out_q) {
    /* not yet added to a bin -> drop the floating ref of whichever one was made */
    if (in_q) gst_object_unref (in_q);
    if (out_q) gst_object_unref (out_q);
    return NULL;
  }
  /* Make BOTH per-lane queues LEAKY (leaky=downstream(2): drop the OLDEST queued buffer when
   * full, instead of blocking) so a momentarily-stalled ModelBin (engine busy / re-batch on
   * add-remove) or a briefly back-pressured combined display mux cannot propagate back-pressure
   * into the shared per-stream tee + input nvstreamdemux and stall the OTHER streams:
   *   in_q  (tee -> [nvvideoconvert] -> in_q -> ModelBin) : protects the input side
   *   out_q (ModelBin.src -> out_q -> combined display mux): protects the output side
   * Each lane just drops its stale frames and keeps the newest. CONFIGURABLE via
   * [multimodel] lane-leaky (default 1 = leaky, behaviour unchanged): lane-leaky=0 keeps
   * the queues non-leaky for COMPLETENESS (every frame reaches every routed model), at the
   * cost of backpressure stalls when per-model fps mismatch (see ModelMuxConfig.lane_leaky). */
  if (!rb->config || rb->config->lane_leaky) {
    g_object_set (G_OBJECT (in_q),  "leaky", 2 /* GST_QUEUE_LEAK_DOWNSTREAM */, NULL);
    g_object_set (G_OBJECT (out_q), "leaky", 2 /* GST_QUEUE_LEAK_DOWNSTREAM */, NULL);
  }
  gst_bin_add_many (GST_BIN (rb->container), in_q, out_q, NULL);
  gst_element_sync_state_with_parent (in_q);
  gst_element_sync_state_with_parent (out_q);

  /* OPTIONAL per-(stream,role) copy converter, spliced BEFORE in_q (buffer-copy-mode policy).
   * Returns NULL for a zero-copy role OR a graceful fallback -- only built when a lane is
   * actually created, so a non-existent role (e.g. no shadow) yields no converter and no queue. */
  conv = modelmux_lane_make_copy_conv (rb->container, rb->role, model_name, stream_id,
      branch_serial, rb->config);

  model_src = model_bin_attach (model_bin, stream_id);
  if (!model_src) {
    modelmux_drop_from_bin (rb->container, conv);
    gst_element_set_state (in_q, GST_STATE_NULL);
    gst_element_set_state (out_q, GST_STATE_NULL);
    gst_bin_remove_many (GST_BIN (rb->container), in_q, out_q, NULL);
    /* the scheduler may have GROWN a shard for this attach -- now empty and
     * otherwise unreclaimed until an unrelated stream-remove: compact it. */
    modelmux_schedule_compact_idle ((ModelMuxBin *) rb->owner);
    return NULL;
  }

  /* in_pad -> [conv ->] in_q -> ModelBin.sink_<idx> ; ModelBin.src_<idx> -> out_q */
  q_in_sink = gst_element_get_static_pad (in_q, "sink");
  q_in_src = gst_element_get_static_pad (in_q, "src");
  /* the ModelBin ghost sink mirrors the demux src index of model_src */
  {
    const gchar *sname = GST_PAD_NAME (model_src);     /* "src_<idx>" */
    gchar sinkname[64];
    g_snprintf (sinkname, sizeof (sinkname), "sink_%s", sname + 4);  /* skip "src_" */
    mb_sink = gst_element_get_static_pad (model_bin->bin, sinkname);
  }
  q_out_sink = gst_element_get_static_pad (out_q, "sink");
  q_out_src = gst_element_get_static_pad (out_q, "src");

  /* head: in_pad -> [conv ->] in_q ; then in_q -> ModelBin.sink ; ModelBin.src -> out_q */
  linked = modelmux_lane_link_head (in_pad, conv, q_in_sink) &&
      modelmux_link_pads (q_in_src, mb_sink) &&
      modelmux_link_pads (model_src, q_out_sink);

  if (!linked) {
    MM_ERR ("ModelPool %s: link chain failed for stream %u model %s",
        rb->role, stream_id, model_name);
    if (q_in_sink) gst_object_unref (q_in_sink);
    if (q_in_src) gst_object_unref (q_in_src);
    if (mb_sink) gst_object_unref (mb_sink);
    if (q_out_sink) gst_object_unref (q_out_sink);
    if (q_out_src) gst_object_unref (q_out_src);
    model_bin_detach (model_bin, stream_id);
    modelmux_drop_from_bin (rb->container, conv);
    gst_element_set_state (in_q, GST_STATE_NULL);
    gst_element_set_state (out_q, GST_STATE_NULL);
    gst_bin_remove_many (GST_BIN (rb->container), in_q, out_q, NULL);
    modelmux_schedule_compact_idle ((ModelMuxBin *) rb->owner);   /* see above */
    return NULL;
  }

  gst_object_unref (q_in_sink);
  gst_object_unref (q_in_src);
  gst_object_unref (mb_sink);
  gst_object_unref (q_out_sink);

  if (in_q_out)  *in_q_out = in_q;
  if (out_q_out) *out_q_out = out_q;
  if (conv_out)  *conv_out = conv;
  if (attached_bin_out) *attached_bin_out = model_bin;

  /* greppable for tests: BUFFER-COPY[<role>]=ON => nvvideoconvert(dp=1) deep-copy lane; OFF => zero-copy. */
  MM_INFO ("    - ModelPool[%s]  route stream %u  ->  model '%s'  (model n=%u)  BUFFER-COPY[%s]=%s",
      rb->role, stream_id, model_name, model_bin_num_streams (model_bin), rb->role, conv ? "ON" : "OFF");
  return q_out_src;            /* caller links this to the combined display mux */
}

gboolean
model_pool_detach (ModelPool * rb, guint stream_id, const gchar * model_name)
{
  ModelBin *base = g_hash_table_lookup (rb->models, model_name);
  if (!base)
    return FALSE;
  /* Find the shard holding this stream and release its pads (SERVING->READY when a
   * shard's last stream leaves). The BASE shard stays warm for reuse; an EXTRA
   * shard that empties is unloaded immediately (scale-in). Destruction of the base
   * is only via model_pool_unload_model(). */
  return modelmux_pool_detach_from_chain (rb, base, stream_id);
}

/* Explicit teardown of an idle model. This is permitted ONLY when
 * the bin is READY (engine warm, 0 streams): it MUST refuse a SERVING bin (>=1
 * stream) and a NOT_READY bin (engine still loading -- unloading mid-load would
 * race the background load thread). This is the ONLY path that destroys a
 * ModelBin; it will back the planned unload REST API. */
gboolean
model_pool_unload_model (ModelPool * rb, const gchar * model_name)
{
  ModelBin *base = g_hash_table_lookup (rb->models, model_name);
  ModelBin *model_bin, *next;
  guint nshards = 0;
  if (!base)
    return FALSE;
  /* EVERY shard must be idle (READY/FAILED, 0 streams). Unload is refused if any
   * shard is SERVING (>=1 stream) or NOT_READY (mid-load -- would race the warm
   * thread). Check the whole chain before destroying anything. */
  for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
    gint st = model_bin_status (model_bin);
    nshards++;
    if ((st != MODEL_WARMED && st != MODEL_FAILED) ||
        model_bin_num_streams (model_bin) > 0) {
      MM_WARN ("ModelPool %s: cannot unload '%s' -- shard #%u status=%s streams=%u "
          "(unload allowed only when ALL shards are idle: READY/FAILED, 0 streams)",
          rb->role, model_name, model_bin->inst, model_status_str (st),
          model_bin_num_streams (model_bin));
      return FALSE;
    }
  }
  MM_INFO ("    - ModelPool[%s]  unload idle model '%s'  (%u shard(s) -> destroyed)",
      rb->role, model_name, nshards);
  /* app-facing completion notice: this (name,version) is being torn down now */
  modelmux_post_model_event (base->bin, "model-unloaded", base->name, base->version,
      model_bin_infer_gpu (base), TRUE, NULL);
  g_hash_table_remove (rb->models, model_name);   /* frees the key only */
  for (model_bin = base; model_bin; model_bin = next) {                 /* free the whole shard chain */
    next = (ModelBin *) model_bin->next_shard;
    /* deferred: every caller holds modelmux_bin->lock (one is the CLEAR probe's streaming
     * thread via modelmux_pool_unload_if_idle) -- never join/NULL a bin under it */
    modelmux_schedule_model_bin_free ((ModelMuxBin *) rb->owner, model_bin);
  }
  return TRUE;
}

void
model_pool_free (ModelPool * rb)
{
  GHashTableIter it;
  gpointer k, v;
  if (!rb)
    return;
  g_hash_table_iter_init (&it, rb->models);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelBin *model_bin = (ModelBin *) v, *next;
    for (; model_bin; model_bin = next) {               /* free the base + all its shards */
      next = (ModelBin *) model_bin->next_shard;
      model_bin_free (model_bin);
    }
  }
  g_hash_table_destroy (rb->models);
  mm_default_ref_clear (&rb->default_ref);
  g_free (rb->role);
  g_free (rb);
}

/* ================================================================== */
/* MultiModelBin                                                        */
/* ================================================================== */

/* combined-mux src EOS probe (MM_DEBUG): when EOS reaches here the pipeline has
 * fully DRAINED -- every in-flight frame has been delivered -- so `in` MUST equal
 * `delivered` per stream. Dump the final frameacct as the definitive zero-loss
 * proof (inflight should be 0). Requires the EOS to actually propagate, i.e. run
 * with file-loop=0 AND drop-pipeline-eos=0. */
static GstPadProbeReturn
modelmux_outmux_eos_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) udata;
  GstEvent *event= GST_PAD_PROBE_INFO_EVENT (info);
  (void) pad;
  if (modelmux_bin && modelmux_bin->log_enabled && event&& GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
    MM_INFO ("FRAMEACCT  EOS at combined-mux src -- pipeline drained; final "
        "reconciliation below: buffered==0 and entered==delivered ==> ZERO FRAME LOSS");
    modelmux_schedule_dump (modelmux_bin, 0);
  }
  return GST_PAD_PROBE_OK;
}

/* combined-mux src buffer probe: count batches; every 100, dump on main loop.
 * Also (MM_DEBUG) logs the first buffer + every 100 so we can see whether the
 * combined display mux is actually PUSHING batches downstream after a drain. */
static GstPadProbeReturn
modelmux_state_log_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) udata;
  guint debug_interval = modelmux_bin->debug_interval ? modelmux_bin->debug_interval : 100;
  (void) pad;
  /* Atomic increment: frame_count is written here (streaming thread) and reset to 0
   * in modelmux_stream_teardown (main loop) when the bin drains to zero streams.  Plain ++ /
   * = would be a data race under TSAN even though the consequence is at most a slightly
   * off dump cadence.  Cast through gint* is safe: the value never exceeds INT_MAX in
   * any realistic session and GLib's atomic API requires a signed pointer. */
  guint fc = (guint) g_atomic_int_add (&modelmux_bin->frame_count, 1) + 1;
  /* One per-buffer pass (MM_DEBUG only) that BOTH accounts every delivered frame
   * (lock-free) AND, on the dump cadence, logs the out-mux tap + provenance. */
  if (modelmux_bin->log_enabled) {
    gboolean do_log = (fc == 1 || (fc % debug_interval) == 0);
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
    NvDsBatchMeta *batch_meta = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
    NvDsMetaType mtype = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;
    NvDsMetaList *fl;
    if (do_log)
      MM_INFO ("TAP[out-mux-src] combined mux pushed %u batch(es) downstream", fc);
    for (fl = batch_meta ? batch_meta->frame_meta_list : NULL; fl; fl = fl->next) {
      NvDsFrameMeta *fm = (NvDsFrameMeta *) fl->data;
      NvDsMetaList *ul;
      for (ul = fm->frame_user_meta_list; ul; ul = ul->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) ul->data;
        InferenceProvenanceMeta *pm;
        if (um->base_meta.meta_type != mtype || !um->user_meta_data)
          continue;
        pm = (InferenceProvenanceMeta *) um->user_meta_data;
        /* DELIVERED count: this frame left the combined mux for this role. */
        if (pm->source_id < MM_ACCT_MAX) {
          gint r = modelmux_role_idx (pm->role);
          g_atomic_int_inc (&modelmux_bin->acct[pm->source_id].role[r].out);
          g_atomic_int_set (&modelmux_bin->acct[pm->source_id].role[r].last_out_fn,
              (gint) fm->frame_num);
          /* CONTIGUITY GAP CHECK (MM_FRAME_TRACE): at this point -- the bin's output,
           * exactly what downstream consumes -- the delivered ORIGINAL frame numbers
           * for a (source,role) MUST be contiguous. A forward jump means frames were
           * dropped somewhere inside the bin. This is the EXACT per-frame proof (no
           * jitter margin) that complements the count heuristic and catches even a
           * single one-time drop. Single-writer per source on this src thread, so the
           * last_src_fn read/update needs no lock. A wrap/reset (cur <= last, e.g.
           * file-loop) is NOT a gap and is ignored -- no false positives. */
          if (modelmux_frametrace_enabled ()) {
            ModelMuxAcctRole *arr = &modelmux_bin->acct[pm->source_id].role[r];
            gint last = arr->last_src_fn;
            gint cur = (gint) pm->frame_num;
            if (last >= 0 && cur > last + 1)     /* forward jump in delivered originals */
              MM_WARN ("FRAMEGAP source=%u role=%s expected_orig_frame=%d got=%d "
                  "(%d frame(s) MISSING -> downstream WILL see a drop)",
                  pm->source_id, pm->role, last + 1, cur, cur - last - 1);
            if (cur > last)                      /* track MAX (ignore wrap/reorder) */
              arr->last_src_fn = cur;
          }
        }
        /* verify (on the dump cadence): provenance survived the mux copy intact. */
        if (do_log)
          MM_INFO ("PROVENANCE  stream=%u name=%s  model=%s@%s  role=%s  gie=%u  gpu=%u  batch=%u  engine=%s",
              pm->source_id, pm->camera_name, pm->model_name, pm->model_version,
              pm->role, pm->gie_id, pm->gpu, pm->batch,
              pm->engine[0] ? pm->engine : "-");
      }
    }
  }
  if ((fc % debug_interval) == 0)
    modelmux_schedule_dump (modelmux_bin, 0);
  return GST_PAD_PROBE_OK;
}

/* Overlay: at the COMBINED mux src (post-mux batched buffer), draw the info block
 * PARSED from the InferenceProvenanceMeta user-meta -- the single source of truth (verified
 * to match the old per-ModelBin overlay, which is now removed). One left-side block
 * per frame per provenance meta. Drawn only when MM_OVERLAY=1 AND provenance is
 * enabled (no meta to parse otherwise). */
static GstPadProbeReturn
modelmux_combined_overlay_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta;
  NvDsMetaType mtype = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;
  NvDsMetaList *l;
  (void) pad;
  (void) udata;

  if (!buffer)
    return GST_PAD_PROBE_OK;
  bmeta = gst_buffer_get_nvds_batch_meta (buffer);
  if (!bmeta)
    return GST_PAD_PROBE_OK;

  for (l = bmeta->frame_meta_list; l != NULL; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *) l->data;
    NvDsMetaList *ul;
    for (ul = fm->frame_user_meta_list; ul != NULL; ul = ul->next) {
      NvDsUserMeta *um = (NvDsUserMeta *) ul->data;
      if (um->base_meta.meta_type == mtype && um->user_meta_data) {
        InferenceProvenanceMeta *pm = (InferenceProvenanceMeta *) um->user_meta_data;
        /* single left-side line: <stream>:<frame-id>:<model>-<version>:<role>
         * colour-coded by role (Primary=green, Shadow=red). */
        gchar *txt = g_strdup_printf ("%s:%u:%s-%s:%s",
            pm->camera_name, fm->frame_num, pm->model_name, pm->model_version, pm->role);
        modelmux_draw_label (bmeta, fm, 12, txt, pm->role);
        g_free (txt);
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* Drop EOS at the InferenceBin entry. Source add/remove (and even all-sources-
 * removed) must NEVER latch the persistent inference/display chain into EOS;
 * otherwise re-adding a stream re-links pads but no data flows. The pipeline is
 * torn down explicitly (state=NULL on SIGINT), so we never rely on EOS here. */
static GstPadProbeReturn
modelmux_drop_eos_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  GstEvent *event= GST_PAD_PROBE_INFO_EVENT (info);
  (void) pad;
  (void) udata;
  if (event&& GST_EVENT_TYPE (event) == GST_EVENT_EOS)
    return GST_PAD_PROBE_DROP;        /* swallow EOS from source churn */
  return GST_PAD_PROBE_OK;
}

/* SOURCE-ID RESTORE at the combined-mux output (always on, both unified and per-role modes).
 * The combined display nvstreammux stamps NvDsFrameMeta.source_id with its OWN sink-pad
 * (display-slot) index. That slot is assigned in ATTACH order and is a purely internal id --
 * it is NOT the camera's upstream source_id. Every downstream per-source consumer keys on this
 * id: PERF / "Active sources" and msgbroker key on source_id, nvmultistreamtiler maps tiles on
 * pad_index (which stock nvstreammux keeps == source_id). Left as-is they all misattribute
 * frames; the breakage is invisible
 * while every stream flows at the same rate, but after a stream is removed the slot<->source_id
 * mapping shifts and the survivors read 0 fps / wrong tiles even though frames still flow.
 * The TRUE upstream id is carried per frame in InferenceProvenanceMeta (attached at the per-model
 * nvinfer src probe, or on the passthrough output path before any re-batching), so we restore
 * it here -- per frame, so it is correct regardless of how the combined mux batched it:
 *   - Primary / Unknown -> fm->source_id = pm->source_id              (the camera id)
 *   - Shadow (unified A/B 2-row display) -> pm->source_id + max_streams: a stable, per-camera
 *     "shadow lane" id, so the bottom display row stays distinct yet remains camera-indexed
 *     (mirrors the slot layout: primary=column, shadow=max_streams+column) instead of being
 *     attach-order. In non-unified / primary-only flows no shadow frames reach this batch, so
 *     the offset branch is simply never taken. */
static GstPadProbeReturn
modelmux_restore_srcid_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) udata;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch_meta = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  NvDsMetaType mtype = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;
  NvDsMetaList *fl;
  (void) pad;
  if (!batch_meta)
    return GST_PAD_PROBE_OK;
  for (fl = batch_meta->frame_meta_list; fl; fl = fl->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *) fl->data;
    NvDsMetaList *ul;
    gboolean restored = FALSE;
    for (ul = fm->frame_user_meta_list; ul; ul = ul->next) {
      NvDsUserMeta *um = (NvDsUserMeta *) ul->data;
      InferenceProvenanceMeta *pm;
      guint old_source_id;
      guint old_pad_index;
      if (um->base_meta.meta_type != mtype || !um->user_meta_data)
        continue;
      pm = (InferenceProvenanceMeta *) um->user_meta_data;
      old_source_id = fm->source_id;
      old_pad_index = fm->pad_index;
      if (modelmux_bin->unified && g_strcmp0 (pm->role, "Shadow") == 0)
        fm->source_id = pm->source_id + modelmux_bin->max_streams;
      else
        fm->source_id = pm->source_id;
      /* keep pad_index == source_id, the stock-nvstreammux invariant. nvmultistreamtiler
       * maps tiles by pad_index (NvTiler.cpp), NOT source_id, so restoring source_id alone
       * would fix PERF/msgbroker yet leave the tile grid on the internal slot. Setting both
       * fixes the tile layout too and restores the invariant downstream code assumes. */
      fm->pad_index = fm->source_id;
      if (modelmux_bin && modelmux_bin->log_enabled) {
        guint debug_interval = modelmux_bin->debug_interval ? modelmux_bin->debug_interval : 100;
        guint fc = (guint) g_atomic_int_get (&modelmux_bin->frame_count) + 1;
        if (fc == 1 || (fc % debug_interval) == 0)
          MM_INFO ("SOURCE-ID restore stream=%u name=%s role=%s "
              "frame=(source_id=%u,pad_index=%u) from slot=(source_id=%u,pad_index=%u)",
              pm->source_id, pm->camera_name, pm->role,
              fm->source_id, fm->pad_index, old_source_id, old_pad_index);
      }
      restored = TRUE;
      break;                              /* exactly one provenance meta per frame */
    }
    if (!restored && modelmux_bin && modelmux_bin->log_enabled) {
      static gint missing_logged = 0;
      if (g_atomic_int_get (&missing_logged) < 8) {
        g_atomic_int_inc (&missing_logged);
        MM_WARN ("SOURCE-ID restore skipped frame without provenance "
            "(slot source_id=%u pad_index=%u) -- downstream may see display-slot ids",
            fm->source_id, fm->pad_index);
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* DRAIN GUARD at the bin entry. When the bin has ZERO attached streams, the upstream
 * legacy nvstreammux inside nvmultiurisrcbin (src_bin_muxer) -- having no input sources --
 * pushes a periodic "dummy clear buffer" downstream while it waits for sources, and stops
 * only once that push returns GST_FLOW_OK (see gst-nvmultistream/gstnvstreammux.cpp,
 * gst_nvstreammux_src_push_loop: the `while (iter == NULL)` / `dummy_clear_buffer_sent`
 * loop). At zero streams our input nvstreamdemux has no src pads and returns NOT_LINKED, so
 * the mux never marks the dummy as delivered and RE-PUSHES the SAME (already-consumed)
 * buffer once a second -> `gst_pad_push: GST_IS_BUFFER` criticals + a refill that then
 * cannot start. A plain downstream (e.g. nvtiler in the no-plugin path) returns FLOW_OK and
 * the mux waits cleanly -- so we make the bin do the same: while idle, DROP the buffer here
 * (a probe DROP returns GST_FLOW_OK to the pusher). The dropped buffer is the mux's empty
 * keepalive batch (num_filled==0), never a real frame, so this loses ZERO frames. Once a
 * stream attaches, active_streams>0 and real batches pass through untouched.
 * NOTE: this is handled entirely in our plugin; the mux's re-push of a consumed buffer on a
 * non-OK return is a latent muxer fragility, but it only surfaces with a downstream that can
 * go not-linked, so the fix belongs here. */
static GstPadProbeReturn
modelmux_indemux_drain_guard_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) udata;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch_meta;
  (void) pad;
  /* Two triggers, both meaning "carries no real frames" -> swallow so the upstream mux's
   * push returns GST_FLOW_OK and it waits cleanly instead of re-pushing a consumed buffer:
   *   (a) no streams attached; and
   *   (b) an EMPTY batch (num_frames_in_batch == 0) -- the mux's idle keepalive buffer.
   * (b) is timing-INDEPENDENT, so it catches the dummy even in the brief window where the
   * mux has already lost its sources but our teardown has not yet zeroed active_streams --
   * exactly when (a) alone would miss it and the spin would start. Real batches always carry
   * >=1 frame, so neither trigger ever drops a real frame (zero frame loss). */
  if (g_atomic_int_get (&modelmux_bin->active_streams) == 0)
    return GST_PAD_PROBE_DROP;
  if (buffer && (batch_meta = gst_buffer_get_nvds_batch_meta (buffer)) != NULL &&
      batch_meta->num_frames_in_batch == 0)
    return GST_PAD_PROBE_DROP;
  return GST_PAD_PROBE_PASS;          /* a real, non-empty batch -> let it through */
}

/* DIAGNOSTIC: count batched buffers nvmultiurisrcbin delivers to our input demux,
 * and (for the FIRST few buffers after each batch-composition change) dump the
 * per-frame source_id + pad_index so we can see exactly which demux src_%u pad
 * each source maps to. The demux routes by frame_meta->pad_index, so if a
 * re-added source reports a NEW source_id but lands in a REUSED pad_index slot,
 * our src_<source_id> request pad would be wrong -- this proves/disproves it. */
static GstPadProbeReturn
modelmux_indemux_count_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  static guint n = 0;
  static guint last_sig = G_MAXUINT;
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) udata;
  guint debug_interval = (modelmux_bin && modelmux_bin->debug_interval) ? modelmux_bin->debug_interval : 100;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch_meta;
  (void) pad;

  (void) n; (void) last_sig; (void) debug_interval;        /* batch-composition DIAG removed */

  if (!buffer || !(batch_meta = gst_buffer_get_nvds_batch_meta (buffer)))
    return GST_PAD_PROBE_OK;
  {
    NvDsMetaList *l;
    NvDsMetaType ttype = modelmux_srctrace_meta_type ();
    for (l = batch_meta->frame_meta_list; l; l = l->next) {
      NvDsFrameMeta *fm = (NvDsFrameMeta *) l->data;
      /* IN count: this frame entered the bin (lock-free, MM_DEBUG). The input
       * reference every role's delivered/inferred count is compared against. */
      if (modelmux_bin && modelmux_bin->log_enabled && fm->source_id < MM_ACCT_MAX) {
        g_atomic_int_inc (&modelmux_bin->acct[fm->source_id].in_count);
        g_atomic_int_set (&modelmux_bin->acct[fm->source_id].last_in_fn, (gint) fm->frame_num);
      }
      /* Source trace: stamp the ORIGINAL (source_id, frame_num) here -- the
       * earliest point, before any internal nvstreammux re-numbers the frame.
       * Provenance and output source-id restoration depend on source_id; the
       * frame_num half is used by MM_FRAME_TRACE's contiguity checks. */
      if (!modelmux_find_srctrace_meta (fm)) {
        NvDsUserMeta *um = nvds_acquire_user_meta_from_pool (batch_meta);
        if (um) {
          ModelMuxSrcTraceMeta *tm = g_new0 (ModelMuxSrcTraceMeta, 1);
          tm->source_id = fm->source_id;
          tm->frame_num = fm->frame_num;
          um->user_meta_data = tm;
          um->base_meta.meta_type = ttype;
          um->base_meta.copy_func = (NvDsMetaCopyFunc) modelmux_srctrace_copy;
          um->base_meta.release_func = (NvDsMetaReleaseFunc) modelmux_srctrace_release;
          nvds_add_user_meta_to_frame (fm, um);
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* ------------------------------------------------------------------ */
/* DEBUG taps: per-stream buffer counters at chosen points along a branch.
 * Gated by MM_DEBUG; each tap is freed automatically when its pad/probe goes
 * away (GDestroyNotify), so they add/remove cleanly with dynamic streams.    */
/* ------------------------------------------------------------------ */
typedef struct
{
  guint stream_id;
  gchar tag[24];
  guint n;
} ModelMuxTap;

static GstPadProbeReturn
modelmux_tap_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u)
{
  ModelMuxTap *t = (ModelMuxTap *) u;
  (void) pad;
  (void) info;
  t->n++;
  if (t->n == 1 || (t->n % 100) == 0)
    MM_INFO ("TAP[%-9s] stream %u: %u buffer(s)", t->tag, t->stream_id, t->n);
  return GST_PAD_PROBE_OK;
}

/* attach a named per-stream buffer tap on pad (no-op unless MM_DEBUG). */
static void
modelmux_add_tap_tracked (ModelMuxBin * modelmux_bin, GstPad * pad, guint stream_id, const gchar * tag,
    GstPad ** pad_out, gulong * probe_out)
{
  ModelMuxTap *t;
  gulong probe_id;
  if (pad_out)
    *pad_out = NULL;
  if (probe_out)
    *probe_out = 0;
  if (!modelmux_bin->log_enabled || !pad)
    return;
  t = g_new0 (ModelMuxTap, 1);
  t->stream_id = stream_id;
  g_strlcpy (t->tag, tag, sizeof (t->tag));
  probe_id = gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, modelmux_tap_probe, t, g_free);
  if (probe_id) {
    if (pad_out)
      *pad_out = gst_object_ref (pad);
    if (probe_out)
      *probe_out = probe_id;
  }
}

/**
 * 
 */
ModelMuxBin *
modelmux_bin_new (GstElement * pipeline, const gchar * name,
    guint max_streams, const ModelMuxConfig * config)
{
  /* Create a new ModelMuxBin instance. */
  ModelMuxBin *modelmux_bin = g_new0 (ModelMuxBin, 1);
  guint i, out_slots;
  GstPad *mux_src, *demux_sink;

  modelmux_bin->bin = gst_bin_new (name ? name : "nvmultimodelbin");
  modelmux_bin->parent_pipeline = pipeline;

  modelmux_bin->config = config;
  modelmux_bin->max_streams = max_streams;
  modelmux_bin->unified = config->unified_batch;

  modelmux_bin->stream_names = g_hash_table_new_full (g_direct_hash, g_direct_equal,
    NULL, g_free);
  modelmux_bin->stream_cam_ids = g_hash_table_new_full (g_direct_hash, g_direct_equal,
    NULL, g_free);
  modelmux_bin->stream_wiring = g_hash_table_new (g_direct_hash, g_direct_equal);
  modelmux_bin->streams_pending = g_hash_table_new (g_direct_hash, g_direct_equal);
  modelmux_bin->streams_cancelled = g_hash_table_new (g_direct_hash, g_direct_equal);

  /* Initialize the mutexes. */
  g_mutex_init (&modelmux_bin->lock);
  g_mutex_init (&modelmux_bin->free_lock);

  /* Configure the debug logging. */
  modelmux_debug_configure (modelmux_bin);

  /* input demux: decouples the batched src (from nvmultiurisrcbin) -> per-stream */
  modelmux_bin->in_demux = create_gst_element ("nvstreamdemux", "infer-in-demux");

  /* output mux: combines the per-stream outputs into a single stream. */
  modelmux_bin->out_mux = create_gst_element ("nvstreammux", "combined-display-mux");
  if (!modelmux_bin->bin || !modelmux_bin->in_demux || !modelmux_bin->out_mux) {
    modelmux_bin_free (modelmux_bin);
    return NULL;
  }
  gst_bin_add (GST_BIN (modelmux_bin->bin), modelmux_bin->in_demux);

  /* the bin's sink GhostPad is its entry: batched src -> input demux */
  demux_sink = gst_element_get_static_pad (modelmux_bin->in_demux, "sink");
  /* swallow EOS here so dynamic source add/remove never latches the chain */
  gst_pad_add_probe (demux_sink, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
      modelmux_drop_eos_probe, NULL, NULL);
  /* DRAIN GUARD (always on): while 0 streams are attached, swallow the upstream mux's idle
   * keepalive buffer so it sees FLOW_OK and waits cleanly instead of re-pushing a consumed
   * buffer (GST_IS_BUFFER spin). Zero real frames are dropped (idle batches only). */
  gst_pad_add_probe (demux_sink, GST_PAD_PROBE_TYPE_BUFFER,
      modelmux_indemux_drain_guard_probe, modelmux_bin, NULL);

  /* MM_DEBUG: trace batches arriving from nvmultiurisrcbin */
  if (modelmux_bin->log_enabled)
    gst_pad_add_probe (demux_sink, GST_PAD_PROBE_TYPE_BUFFER,
        modelmux_indemux_count_probe, modelmux_bin, NULL);

  /* Link the demux sink to the bin sink GhostPad. */
  modelmux_bin->sink_ghost = gst_ghost_pad_new ("sink", demux_sink);
  gst_pad_set_active (modelmux_bin->sink_ghost, TRUE);
  gst_element_add_pad (modelmux_bin->bin, modelmux_bin->sink_ghost);

  /* answer the REST /api/v1/model/status downstream query at the bin boundary */
  gst_pad_add_probe (modelmux_bin->sink_ghost, GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM,
      modelmux_status_query_probe, modelmux_bin, NULL);
  /* answer the SYNCHRONOUS control-admission query (model/unload) here too, so the
   * REST caller gets our real verdict + reason instead of a blind 202 */
  gst_pad_add_probe (modelmux_bin->sink_ghost, GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM,
      modelmux_control_query_probe, modelmux_bin, NULL);
  gst_object_unref (demux_sink);

  /* output mux: set properties defensively (see set_prop_if_exists): the
   * new nvstreammux does not expose width/height/timeout/live-source as properties. */
  out_slots = modelmux_bin->unified ? max_streams * 2 : max_streams;
  set_prop_if_exists (modelmux_bin->out_mux, "batch-size", out_slots);
  set_prop_if_exists (modelmux_bin->out_mux, "width",
      modelmux_bin->config ? modelmux_bin->config->mux_width : MM_MUX_WIDTH);
  set_prop_if_exists (modelmux_bin->out_mux, "height",
      modelmux_bin->config ? modelmux_bin->config->mux_height : MM_MUX_HEIGHT);
  set_prop_if_exists (modelmux_bin->out_mux, "batched-push-timeout",
      modelmux_bin->config ? modelmux_bin->config->combined_mux_push_timeout : MM_MUX_PUSH_TIMEOUT);
  set_prop_if_exists (modelmux_bin->out_mux, "live-source",
      modelmux_bin->config ? modelmux_bin->config->mux_live_source : 1);
  set_prop_if_exists (modelmux_bin->out_mux, "gpu-id",
      modelmux_bin->config ? (gint) modelmux_bin->config->gpu : 0);
  gst_bin_add (GST_BIN (modelmux_bin->bin), modelmux_bin->out_mux);

  /* recycled DISPLAY columns (one per stream). The out_mux slot for a (stream,role)
   * is DERIVED from the column so the tiler lays primary on the top row and shadow
   * directly below it: primary -> slot=column, shadow -> slot=max_streams+column
   * (legacy nvstreammux sets frame source_id/pad_index = sink-pad index, and the
   * tiler tiles by pad_index row-major, so slot == tile). */
  modelmux_bin->col_free_idx = g_queue_new ();
  for (i = 0; i < max_streams; i++)
    g_queue_push_tail (modelmux_bin->col_free_idx, GINT_TO_POINTER (i));

  /* role bins: Primary uid base 1, Shadow uid base 100 */
  modelmux_bin->primary_pool = model_pool_new ("Primary", 1, max_streams,
      pipeline, modelmux_bin->bin, config, modelmux_bin->stream_names);
  modelmux_bin->shadow_pool = model_pool_new ("Shadow", 100, max_streams,
      pipeline, modelmux_bin->bin, config, modelmux_bin->stream_names);
  modelmux_bin->primary_pool->owner = modelmux_bin;
  modelmux_bin->shadow_pool->owner = modelmux_bin;

  /* SEED the runtime default designations from the parsed config -- the ONLY read of
   * the config's default_* fields after parse. From here on the pool refs are the
   * single source of truth and the config is read-only. Pre-publication (the bin is
   * still under construction, no other thread can see it), so no lock is needed;
   * every later writer mutates under modelmux_bin->lock. bin caches stay NULL until
   * load_defaults / model-load materializes the identities. */
  if (config && config->default_primary)
    mm_default_ref_assign (&modelmux_bin->primary_pool->default_ref,
        config->default_primary, config->default_primary_version,
        config->default_primary_gpu, NULL);
  if (config && config->default_shadow)
    mm_default_ref_assign (&modelmux_bin->shadow_pool->default_ref,
        config->default_shadow, config->default_shadow_version,
        config->default_shadow_gpu, NULL);

  /* perf metrics: opt-in. When attach-perf-metric is on, start ONE periodic snapshot timer that
   * ticks every model bin's fps/latency counters every perf-metric-interval-sec. Off by default
   * -> no timer, zero overhead, and model/status emits no perf fields. */
  if (config && config->attach_perf_metric)
    modelmux_bin->perf_timer_id = g_timeout_add_seconds (config->perf_metric_interval_sec,
        modelmux_perf_timer_cb, modelmux_bin);

  /* type-less model/load limbo: warm bins with role Unknown, promoted on first use.
   * gie ids 1000+ stay clear of the Primary (1..) and Shadow (100..) ranges and are
   * kept across promotion, so downstream identity never changes. */
  modelmux_bin->limbo_models = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  modelmux_bin->next_limbo_gie = 1000;

  /* ghost the combined mux src out of the MultiModelBin */
  mux_src = gst_element_get_static_pad (modelmux_bin->out_mux, "src");
  /* swallow EOS at the display-batch output: when the last stream leaves, the
   * combined nvstreammux would emit EOS and latch the loose downstream
   * (queue/tiler/osd/sink), so re-added streams would never render. */
  gst_pad_add_probe (mux_src, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
      modelmux_drop_eos_probe, NULL, NULL);
  /* always-on: restore the real upstream source_id the display mux overwrote with its slot */
  gst_pad_add_probe (mux_src, GST_PAD_PROBE_TYPE_BUFFER,
      modelmux_restore_srcid_probe, modelmux_bin, NULL);
  modelmux_bin->src_ghost = gst_ghost_pad_new ("src", mux_src);
  gst_pad_set_active (modelmux_bin->src_ghost, TRUE);
  gst_element_add_pad (modelmux_bin->bin, modelmux_bin->src_ghost);
  gst_object_unref (mux_src);

  gst_bin_add (GST_BIN (pipeline), modelmux_bin->bin);

  /* MM_DEBUG: periodic per-stream model table (every 100 combined-mux batches) +
   * an EOS reconciliation dump (definitive zero-loss proof once the pipeline drains). */
  if (modelmux_bin->log_enabled) {
    GstPad *osrc = gst_element_get_static_pad (modelmux_bin->out_mux, "src");
    if (osrc) {
      gst_pad_add_probe (osrc, GST_PAD_PROBE_TYPE_BUFFER, modelmux_state_log_probe,
          modelmux_bin, NULL);
      gst_pad_add_probe (osrc, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
          modelmux_outmux_eos_probe, modelmux_bin, NULL);
      gst_object_unref (osrc);
    }
    MM_INFO ("MM_DEBUG on: models_table + routing_table + frameacct_table dump every "
        "%u frames (set MM_DEBUG_INTERVAL to change)", modelmux_bin->debug_interval);
  }

  /* Overlay at the combined mux src: parse the provenance user-meta (always present)
   * and draw one left-side block per frame (MM_OVERLAY=1; debug only). This is
   * the single overlay -- the per-ModelBin display probe was removed. */
  if (modelmux_overlay_enabled ()) {
    GstPad *osrc = gst_element_get_static_pad (modelmux_bin->out_mux, "src");
    if (osrc) {
      gst_pad_add_probe (osrc, GST_PAD_PROBE_TYPE_BUFFER, modelmux_combined_overlay_probe,
          modelmux_bin, NULL);
      gst_object_unref (osrc);
    }
  }

  MM_INFO ("- MultiModelBin created  (max_streams=%u, unified=%d, display-slots=%u)",
      max_streams, modelmux_bin->unified, out_slots);

  return modelmux_bin;
}

GstPad *
modelmux_bin_get_src (ModelMuxBin * modelmux_bin)
{
  return modelmux_bin ? modelmux_bin->src_ghost : NULL;
}

/* TYPE-LESS model/load: warm the model into the limbo map (role Unknown). Its type
 * is decided on first use (stream-add/model-update promotes it into a pool). The bin
 * is created in the SHARED container with a limbo gie-id; promotion later keeps that
 * gie-id so downstream identity is stable. Idempotent (already-loaded -> no-op). */
ModelStatus
modelmux_bin_load_model (ModelMuxBin * modelmux_bin, const gchar * key,
    const gchar * cfg_override, gint gpu, guint batch)
{
  ModelBin *model_bin;
  const ModelCatalogEntry *def;
  guint uid, mx;
  if (!modelmux_bin || !key || !*key)
    return MODEL_WARMING;
  g_mutex_lock (&modelmux_bin->lock);
  /* this exact (name,version) instance already loaded (limbo OR a pool)? -> idempotent,
   * retry if FAILED. Different versions of the same name have different keys, so they do
   * NOT collide here -- they coexist as separate bins. */
  if ((model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, key)) ||
      (model_bin = g_hash_table_lookup (modelmux_bin->primary_pool->models, key)) ||
      (model_bin = g_hash_table_lookup (modelmux_bin->shadow_pool->models, key))) {
    if (model_bin_status (model_bin) == MODEL_FAILED)
      model_bin_load (model_bin);
    mx = model_bin_status (model_bin);
    g_mutex_unlock (&modelmux_bin->lock);
    return (ModelStatus) mx;
  }
  /* split the canonical key -- catalog config is by bare name; `cfg_override` (a per-version
   * custom OR engine-derived config from model/load) wins when supplied. Engine is baked into
   * the (derived) config, so engine_file is NULL here. */
  {
    gchar *bare = NULL, *ver = NULL;
    const gchar *use_cfg, *use_eng;
    model_key_split (key, &bare, &ver);
    def = modelmux_config_find_model (modelmux_bin->config, bare);
    use_cfg = cfg_override ? cfg_override : (def ? def->config_file : NULL);
    use_eng = cfg_override ? NULL : (def ? def->engine_file : NULL);
    if (!use_cfg) {
      g_free (bare); g_free (ver);
      g_mutex_unlock (&modelmux_bin->lock);
      MM_ERR ("MultiModelBin: load '%s' -- no config (model not in catalog and none supplied)",
          key);
      return MODEL_FAILED;   /* a rejected load must never read as "warming":
                                 * the control plane would register the version
                                 * and report the load accepted (phantom model) */
    }
    uid = modelmux_bin->next_limbo_gie++;                     /* limbo gie range; kept on promotion */
    mx = modelmux_effective_batch (modelmux_bin->config, def, bare, ver, batch,
        modelmux_bin->max_streams);
    /* build the per-model bin with ALL its elements (nvstreammux -> nvinfer/
     * nvinferserver -> nvstreamdemux, plus slot tables and any cross-GPU migration
     * elements) and add it to the shared container. Role "Unknown" = created into
     * limbo; not warmed or linked into the combined path yet (that happens below at
     * model_bin_load and later on stream attach/promotion). */
    model_bin = model_bin_new (bare, "Unknown", use_cfg, use_eng,
        ver, uid, mx, modelmux_bin->config, modelmux_bin->parent_pipeline, modelmux_bin->bin, modelmux_bin->stream_names, 0);
    g_free (bare); g_free (ver);
  }
  if (!model_bin) {
    g_mutex_unlock (&modelmux_bin->lock);
    /* bin creation fails DETERMINISTICALLY (P2P_UNAVAILABLE / XFER_UNAVAILABLE /
     * missing element factory -- already logged): report FAILED so the control
     * plane rejects the load instead of registering a phantom version. */
    return MODEL_FAILED;
  }
  model_bin->pool = NULL;                          /* limbo: not in any pool yet */
  if (gpu >= 0) {
    /* pin placement BEFORE the warm starts: the warm-up thread pushes the first
     * buffer through conv-in, which latches its device then -- a late set_gpu
     * would leave the lane migrating to the WRONG gpu until a recreate. */
    model_bin_set_gpu (model_bin, gpu);
    model_bin->pinned = TRUE;
  }
  g_hash_table_insert (modelmux_bin->limbo_models, g_strdup (model_bin->key), model_bin);  /* canonical "name@version" */
  /* background warm -> READY */
  model_bin_load (model_bin);
  g_mutex_unlock (&modelmux_bin->lock);
  MM_INFO ("  - MultiModelBin  model '%s' LOADED into limbo (gie=%u, role Unknown) "
      "-- primary/shadow decided on first use", model_bin->key, uid);
  return MODEL_WARMING;                /* warming */
}

/* model/load with gpus[]: one shard per listed GPU. The BASE shard is created by
 * the exact single-instance path above (cfgs[0] = the gpu[0]-derived config); the n-1
 * siblings are then appended to its chain, each PINNED with its own per-GPU derived
 * config, sharing the base gie-id, warming in the background exactly like the base
 * (nothing attaches to a shard before it is WARMED, so no synchronous wait is needed
 * -- mirrors the pool's overflow-shard creation minus the attach-pending wait). */
ModelStatus
modelmux_bin_load_model_instances (ModelMuxBin * modelmux_bin, const gchar * key,
    const gchar * const cfgs[], const gint gpus[], guint n, guint batch)
{
  ModelStatus st;
  ModelBin *model_bin, *tail;
  guint i;

  if (n <= 1)                              /* single instance = the legacy path, verbatim */
    return modelmux_bin_load_model (modelmux_bin, key, (n && cfgs) ? cfgs[0] : NULL,
        (n && gpus) ? gpus[0] : -1, batch);
  if (!modelmux_bin || !key || !*key || !cfgs || !gpus)
    return MODEL_WARMING;

  st = modelmux_bin_load_model (modelmux_bin, key, cfgs[0], gpus[0], batch);
  g_mutex_lock (&modelmux_bin->lock);
  /* the base just landed in limbo (the control plane rejects an already-loaded key
   * before calling, and load ops are serialized on the main loop under api_control_lock) --
   * a pre-existing / promoted / failed-to-create base gets NO siblings appended. */
  model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, key);
  if (!model_bin || model_bin->next_shard) {
    g_mutex_unlock (&modelmux_bin->lock);
    return st;
  }
  tail = model_bin;                               /* base already placed+pinned by the load above */
  for (i = 1; i < n; i++) {
    ModelBin *shard = model_bin_new (model_bin->name, "Unknown",
        cfgs[i] ? cfgs[i] : model_bin->config_file, NULL /* engine baked into the config */,
        model_bin->version, model_bin->unique_id /*SHARED gie-id*/, model_bin->max_streams,
        modelmux_bin->config, modelmux_bin->parent_pipeline, modelmux_bin->bin, modelmux_bin->stream_names,
        modelmux_shard_next_inst ());
    if (!shard) {
      /* ALL-OR-NOTHING: a shard that fails to materialize must not leave a
       * PARTIAL group silently serving fewer GPUs than requested. Roll back
       * every instance THIS call created -- the base included: it was created
       * by the load above and still lives unpublished in limbo (pool==NULL,
       * no streams can attach before WARMED, load ops are serialized under
       * api_control_lock), so stealing it out of the limbo map under modelmux_bin->lock makes
       * the whole chain unreachable; the bins are then freed OUTSIDE the lock
       * (model_bin_free joins the warm-up thread -- potentially seconds --
       * and must not stall every modelmux_bin->lock waiter). */
      ModelBin *victim, *next;
      modelmux_bin_forget_default_bin (modelmux_bin, model_bin);
      g_hash_table_remove (modelmux_bin->limbo_models, key);   /* frees the key copy only */
      g_mutex_unlock (&modelmux_bin->lock);
      for (victim = model_bin; victim; victim = next) {
        next = (ModelBin *) victim->next_shard;
        model_bin_free (victim);
      }
      MM_ERR ("MultiModelBin: instance of '%s' on gpu %d could NOT be created -- "
          "model/load FAILED and the %u sibling instance(s) already created were "
          "rolled back (all-or-nothing: no partial group is deployed)", key,
          gpus[i], i);
      return MODEL_FAILED;
    }
    shard->pool = NULL;                    /* limbo, like the base */
    model_bin_set_gpu (shard, gpus[i]);
    shard->pinned = TRUE;
    tail->next_shard = shard;
    tail = shard;
    model_bin_load (shard);             /* background warm -> READY */
    MM_INFO ("  - MultiModelBin  model '%s' instance shard #%u placed on gpu %d "
        "(gie=%u shared) -> warming", key, shard->inst, gpus[i], model_bin->unique_id);
  }
  g_mutex_unlock (&modelmux_bin->lock);
  return st;
}

/* look up a live bin for `name` in any table (limbo / primary / shadow). The returned
 * pointer is only valid while modelmux_bin->lock is HELD: a concurrent unload can free the bin once
 * the lock is dropped, so callers must not retain/deref it after unlocking. */
static ModelBin *
modelmux_lookup_model_locked (ModelMuxBin * modelmux_bin, const gchar * name)
{
  ModelBin *model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, name);
  if (!model_bin)
    model_bin = g_hash_table_lookup (modelmux_bin->primary_pool->models, name);
  if (!model_bin)
    model_bin = g_hash_table_lookup (modelmux_bin->shadow_pool->models, name);
  return model_bin;
}

/* Collect EVERY live bin registered under `name` (canonical key). The SAME key
 * can coexist as INDEPENDENT bins in the limbo map AND either role pool
 * (model_pool_get_or_create builds a per-pool copy), so operations that
 * transition the identity -- OTA reload, identity rename -- must cover ALL of
 * them: acting on the first match only leaves a TWIN silently serving the
 * stale checkpoint under the still-live old key. Fills out[3] (limbo ->
 * primary -> shadow order, matching modelmux_lookup_model_locked); returns the
 * count. Caller holds modelmux_bin->lock. */
static guint
modelmux_lookup_model_all_locked (ModelMuxBin * modelmux_bin, const gchar * name,
    ModelBin * out[3])
{
  GHashTable *tables[3];
  ModelBin *model_bin;
  guint i, n = 0;
  tables[0] = modelmux_bin->limbo_models;
  tables[1] = modelmux_bin->primary_pool ? modelmux_bin->primary_pool->models : NULL;
  tables[2] = modelmux_bin->shadow_pool ? modelmux_bin->shadow_pool->models : NULL;
  for (i = 0; i < 3; i++)
    if (tables[i] && (model_bin = g_hash_table_lookup (tables[i], name)))
      out[n++] = model_bin;
  return n;
}

/* TRUE while ANY live bin of bare model `name` has an OTA update awaiting
 * nvinfer's confirm. model/load admission gates on this: the update frees its
 * OLD key at trigger time, and a load occupying that key during the confirm
 * window would block a failed confirm's rename-back (the rollback refuses an
 * occupied destination). Lock order: modelmux_bin->lock -> prov_lock (matches the
 * reload admission scan). */
gboolean
modelmux_bin_update_in_flight (ModelMuxBin * modelmux_bin, const gchar * name)
{
  GHashTable *tables[3];
  GHashTableIter it;
  gpointer k, v;
  gboolean busy = FALSE;
  guint i;

  if (!modelmux_bin || !name || !*name)
    return FALSE;
  g_mutex_lock (&modelmux_bin->lock);
  tables[0] = modelmux_bin->limbo_models;
  tables[1] = modelmux_bin->primary_pool ? modelmux_bin->primary_pool->models : NULL;
  tables[2] = modelmux_bin->shadow_pool ? modelmux_bin->shadow_pool->models : NULL;
  for (i = 0; i < 3 && !busy; i++) {
    if (!tables[i])
      continue;
    g_hash_table_iter_init (&it, tables[i]);
    while (!busy && g_hash_table_iter_next (&it, &k, &v)) {
      ModelBin *model_bin = (ModelBin *) v, *shard;
      if (g_strcmp0 (model_bin->name, name) != 0)
        continue;
      for (shard = model_bin; shard && !busy;
          shard = (ModelBin *) shard->next_shard) {
        g_mutex_lock (&shard->prov_lock);
        busy = shard->ota_await;
        g_mutex_unlock (&shard->prov_lock);
      }
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);
  return busy;
}

gboolean
modelmux_bin_model_loaded (ModelMuxBin * modelmux_bin, const gchar * name)
{
  gboolean found;
  if (!modelmux_bin || !name || !*name)
    return FALSE;
  g_mutex_lock (&modelmux_bin->lock);
  found = (modelmux_lookup_model_locked (modelmux_bin, name) != NULL);
  g_mutex_unlock (&modelmux_bin->lock);
  return found;
}

/* IDENTITY RENAME (API_DESIGN.md 5.3): move a live model's canonical key
 * old_key -> new_key at EVERY bin-layer touch point, atomically under one
 * modelmux_bin->lock hold: the owning hash table entries (limbo / primary / shadow --
 * steal + reinsert in EVERY table holding the key, since the same key can be
 * live as independent twins in more than one; see
 * modelmux_lookup_model_all_locked), model_bin->key on every shard of every such chain,
 * and every routing-table role attach (ra->model / ra->pending_model)
 * referencing it. The key strings are only read under modelmux_bin->lock (status, dump,
 * reroute commit, compaction -- the lock-free per-frame probes read
 * model_bin->name/model_bin->version, never the key), so a plain free+strdup is safe.
 * Caller holds modelmux_bin->lock. Returns FALSE when old_key is not live or new_key
 * already names a DIFFERENT live instance (never corrupt the key space). */
static gboolean
modelmux_bin_rename_model_locked (ModelMuxBin * modelmux_bin, const gchar * old_key,
    const gchar * new_key)
{
  GHashTable *tables[3];
  ModelBin *s;
  gchar *old;
  guint i, nbins = 0, nshards = 0, nrefs = 0;

  if (!modelmux_bin || !old_key || !new_key || g_strcmp0 (old_key, new_key) == 0)
    return FALSE;
  /* callers may pass model_bin->key itself as old_key, which the chain loop below
   * frees -- work from an owned copy so every later compare/log stays valid. */
  old = g_strdup (old_key);
  old_key = old;
  if (!modelmux_lookup_model_locked (modelmux_bin, old_key)) {
    g_free (old);
    return FALSE;                          /* nothing live under the old key */
  }
  if (modelmux_lookup_model_locked (modelmux_bin, new_key)) {
    MM_ERR ("MultiModelBin: identity rename '%s' -> '%s' refused: the "
        "destination already names a live instance -- key kept", old_key,
        new_key);
    g_free (old);
    return FALSE;
  }
  tables[0] = modelmux_bin->limbo_models;
  tables[1] = modelmux_bin->primary_pool ? modelmux_bin->primary_pool->models : NULL;
  tables[2] = modelmux_bin->shadow_pool ? modelmux_bin->shadow_pool->models : NULL;
  /* re-key EVERY table holding the key (limbo twin + pool copies move
   * together -- a missed twin would keep serving the stale checkpoint under
   * the still-live old key). Steal + reinsert: each table owns its key copy
   * (g_free key destroy). */
  for (i = 0; i < 3; i++) {
    gpointer sk = NULL, sv = NULL;
    if (!tables[i] ||
        !g_hash_table_steal_extended (tables[i], old_key, &sk, &sv))
      continue;
    g_free (sk);
    g_hash_table_insert (tables[i], g_strdup (new_key), sv);
    nbins++;
    /* a shard chain shares ONE identity: every shard's cached key moves */
    for (s = (ModelBin *) sv; s; s = (ModelBin *) s->next_shard) {
      g_free (s->key);
      s->key = g_strdup (new_key);
      nshards++;
    }
  }
  /* routing table: every stream ref (serving or pending promotion) follows the
   * identity -- these strings feed the status stream refs, the reroute
   * comparisons and the default-switch stream collection. */
  if (modelmux_bin->stream_wiring) {
    GHashTableIter it;
    gpointer k, v;
    g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelMuxStreamEntry *e = (ModelMuxStreamEntry *) v;
      ModelMuxRoleAttach *ras[2] = { &e->prim, &e->shad };
      guint r;
      for (r = 0; r < 2; r++) {
        if (ras[r]->model && g_strcmp0 (ras[r]->model, old_key) == 0) {
          g_free (ras[r]->model);
          ras[r]->model = g_strdup (new_key);
          nrefs++;
        }
        if (ras[r]->pending_model &&
            g_strcmp0 (ras[r]->pending_model, old_key) == 0) {
          g_free (ras[r]->pending_model);
          ras[r]->pending_model = g_strdup (new_key);
          nrefs++;
        }
      }
    }
  }
  MM_INFO ("  - MultiModelBin  identity RENAMED '%s' -> '%s' (%u live bin(s), "
      "%u shard key(s), %u stream ref(s)) -- the checkpoint transition MOVES "
      "the live identity (API_DESIGN.md 5.3)", old_key, new_key, nbins,
      nshards, nrefs);
  g_free (old);
  return TRUE;
}

/* Effective placement of one shard: its own recorded gpu, falling back to the
 * version registry, then the device the shard's CONFIG FILE pins, then the
 * default device 0 (the same fallback the control plane's gpu-ref checks and
 * conv-in seeding use). Caller holds modelmux_bin->lock. */
static gint
modelmux_shard_gpu_effective (ModelMuxBin * modelmux_bin, ModelBin * model_bin)
{
  gint gpu = model_bin->gpu;
  if (gpu < 0)
    gpu = modelmux_config_version_gpu (modelmux_bin->config, model_bin->name, model_version_str (model_bin));
  if (gpu < 0)
    gpu = model_bin->cfg_gpu;
  return gpu >= 0 ? gpu : 0;
}

/* Shared OTA reload flow of modelmux_bin_reload_model (single engine)
 * and modelmux_bin_reload_model_multi (per-device engines): twin lookup,
 * ALL-OR-NOTHING admission, per-shard trigger, identity rename + group seal
 * -- ONE implementation so the two entry points cannot drift. The engine
 * PICKER is the only functional difference: gpus==NULL selects the
 * single-engine form (config/engine applied to every shard); otherwise each
 * shard swaps to the engine matching ITS effective gpu ({gpus[i] ->
 * engines[i]}, n_eng pairs) and admission additionally requires every
 * shard's device to be covered. The `multi` message variants keep every log
 * string byte-identical to the original per-flavour functions (tests assert
 * the exact phrasings). */
/* ---- idle-model OTA commit fallback ----------------------------------------
 * nvinfer commits an on-the-fly engine swap (and fires "model-updated") ONLY
 * when a buffer flows: ensureReplaceNextContext() runs solely from
 * gst_nvinfer_submit_input_buffer(). A model with ZERO streams thus stages the
 * new engine on nvinfer's load thread but never swaps, so its update would
 * never confirm and never announce `model updated`. This fallback -- armed
 * after the trigger for an idle model -- drives the group to COMMIT once
 * nvinfer's load thread has had a grace window to surface a deserialize FAILURE
 * (which nvinfer reports WITHOUT a buffer -> the normal grouped rollback path
 * handles it and clears ota_group, leaving this a no-op).
 *
 * Lifetime/lock safety: modelmux_ota_group_resolve_main() is the ONLY thing
 * that frees a group, and it does so under modelmux_bin->lock while clearing
 * each shard's ota_group. This handler runs under the SAME lock and rediscovers
 * the live group FROM the shards (never holds a pre-fetched group pointer), so
 * it can neither touch a freed group nor race a real confirm. It fills only the
 * confirm SHORTFALL, so exactly one path reaches confirmed==expected and
 * schedules the single resolve. */
/* Grace before an idle-model OTA is force-committed. It must exceed the worst-case
 * time nvinfer's load thread needs to deserialize/build the new engine AND surface
 * a FAILURE (nvinfer reports load failures on that thread, without a buffer -- the
 * normal grouped rollback then wins and this fallback backs off). A prebuilt
 * .engine deserializes in tens of ms; 10 s is a wide margin. A model that builds a
 * TRT engine from ONNX at update time may need this raised. Only cost of a larger
 * value is a later 'model updated' line for a NON-serving model (no traffic waits
 * on it). Failures WITHIN the window roll back correctly; the residual is a failure
 * that lands AFTER it -- unreachable for prebuilt engines. */
#define MM_OTA_IDLE_COMMIT_MS 10000

typedef struct { ModelMuxBin *modelmux_bin; gchar *key; } ModelMuxOtaIdleCtx;

static void
modelmux_ota_idle_ctx_free (gpointer data)
{
  ModelMuxOtaIdleCtx *c = (ModelMuxOtaIdleCtx *) data;
  g_free (c->key);
  g_free (c);
}

static gboolean
modelmux_ota_idle_commit_main (gpointer data)
{
  ModelMuxOtaIdleCtx *c = (ModelMuxOtaIdleCtx *) data;
  ModelMuxBin *modelmux_bin = c->modelmux_bin;
  ModelBin *bins[3], *s;
  ModelMuxOtaGroup *group = NULL;
  gboolean idle = TRUE, schedule = FALSE;
  guint nbins, b;

  g_mutex_lock (&modelmux_bin->lock);
  nbins = modelmux_lookup_model_all_locked (modelmux_bin, c->key, bins);
  for (b = 0; b < nbins; b++) {
    for (s = bins[b]; s; s = (ModelBin *) s->next_shard) {
      if (model_bin_num_streams (s) > 0)
        idle = FALSE;                              /* a stream arrived -> buffer path owns it */
      g_mutex_lock (&s->prov_lock);
      if (s->ota_await && s->ota_group)
        group = (ModelMuxOtaGroup *) s->ota_group; /* the in-flight group; alive under modelmux_bin->lock */
      g_mutex_unlock (&s->prov_lock);
    }
  }
  /* commit only a still-IDLE model whose OTA nvinfer never confirmed (no buffer);
   * if it already resolved, `group` stayed NULL above. */
  if (group && idle) {
    g_mutex_lock (&group->lock);
    if (group->sealed && group->confirmed < group->expected) {
      group->confirmed = group->expected;          /* fill the shortfall (idle: nvinfer sent none) */
      schedule = TRUE;                              /* we completed the count -> we schedule the resolve */
    }
    g_mutex_unlock (&group->lock);
    if (schedule)
      MM_INFO ("  - MultiModelBin  idle-model OTA '%s': no stream to trigger "
          "nvinfer's buffer-gated swap -> committing the staged engine "
          "(grace elapsed, nvinfer reported no load failure)", c->key);
  }
  g_mutex_unlock (&modelmux_bin->lock);

  if (schedule)                                     /* sole scheduler for an idle group -> stays alive until it runs */
    modelmux_ota_group_schedule_resolve (group);
  return G_SOURCE_REMOVE;
}

/* Arm the idle-commit fallback for `key` after the grace window. Tracked, so a
 * pipeline teardown cancels it. */
static void
modelmux_ota_idle_commit_arm (ModelMuxBin * modelmux_bin, const gchar * key)
{
  ModelMuxOtaIdleCtx *c = g_new0 (ModelMuxOtaIdleCtx, 1);
  c->modelmux_bin = modelmux_bin;
  c->key = g_strdup (key);
  if (!modelmux_tracked_timeout_add (modelmux_bin, MM_OTA_IDLE_COMMIT_MS,
          modelmux_ota_idle_commit_main, c, modelmux_ota_idle_ctx_free, NULL))
    modelmux_ota_idle_ctx_free (c);
}

static gboolean
modelmux_reload_model_common (ModelMuxBin * modelmux_bin, const gchar * name,
    const gchar * config, const gchar * engine, const gint gpus[],
    const gchar * const engines[], guint n_eng, const gchar * version)
{
  ModelBin *bins[3], *s;
  ModelMuxOtaGroup *group;
  gchar *new_key = NULL, *idle_key = NULL;
  gboolean multi = (gpus != NULL);
  gboolean rename, resolve_now = FALSE;
  guint n = 0, total = 0, nbins, b, i;

  g_mutex_lock (&modelmux_bin->lock);
  /* the SAME key can be live as independent twins in limbo AND either pool --
   * the update transitions EVERY one of them, or a missed twin keeps serving
   * the stale checkpoint under the still-live old key. */
  nbins = modelmux_lookup_model_all_locked (modelmux_bin, name, bins);
  if (!nbins) {                             /* not present -> caller reports "not found" */
    g_mutex_unlock (&modelmux_bin->lock);
    return FALSE;
  }
  /* IDENTITY MOVE (API_DESIGN.md 5.3): a NEW version renames the live key
   * name@FROM -> name@TO once every shard accepts the swap (the key is
   * chain-wide, so the rename happens ONCE for the whole group); the same
   * version is an engine-only refresh, no rename. */
  if (version && *version)
    new_key = model_key (bins[0]->name, version);
  rename = (new_key && g_strcmp0 (new_key, bins[0]->key) != 0);
  /* ALL-OR-NOTHING admission (one lock hold): 4 checks below must ALL pass
   * before a single shard is swapped -- each reason is documented at its check. */
  /* ---- ADMISSION CHECK 1/4: destination version key must be FREE ----
   * A rename (name@from -> name@to) cannot land on a key that already names a
   * live instance -> that would collide two live models. */
  if (rename && modelmux_lookup_model_locked (modelmux_bin, new_key)) {
    g_mutex_unlock (&modelmux_bin->lock);
    if (multi)
      MM_ERR ("MultiModelBin: multi-GPU reload of '%s' rejected: '%s' already "
          "names a live instance -- NOTHING was swapped (unload it first, or "
          "update FROM it)", name, new_key);
    else
      MM_ERR ("MultiModelBin: OTA reload of '%s' rejected: '%s' already names a "
          "live instance -- NOTHING was swapped (unload it first, or update FROM "
          "it)", name, new_key);
    g_free (new_key);
    return FALSE;
  }
  for (b = 0; b < nbins; b++) {
    for (s = bins[b]; s; s = (ModelBin *) s->next_shard) {
      gint gpu = modelmux_shard_gpu_effective (modelmux_bin, s);
      ModelStatus st = model_bin_status (s);
      gboolean covered = FALSE, awaiting;
      total++;
      /* ---- ADMISSION CHECK 2/4: every shard must be READY/SERVING ----
       * a WARMING/FAILED shard would be skipped by the swap, splitting the
       * chain's identity (some shards new engine/version, some stale). */
      if (st != MODEL_WARMED && st != MODEL_SERVING) {
        g_mutex_unlock (&modelmux_bin->lock);
        if (multi)
          MM_ERR ("MultiModelBin: multi-GPU reload of '%s' rejected: shard #%u "
              "(gpu %d) is %s -- NOTHING was swapped (every instance must be "
              "ready/serving; the group transitions together or not at all)",
              name, s->inst, gpu, model_status_str (st));
        else
          MM_ERR ("MultiModelBin: OTA reload of '%s' rejected: shard #%u is %s -- "
              "NOTHING was swapped (every shard must be ready/serving; the chain "
              "transitions together or not at all)", name, s->inst,
              model_status_str (st));
        g_free (new_key);
        return FALSE;
      }
      /* ---- ADMISSION CHECK 3/4: no shard mid-confirm of a PREVIOUS update ----
       * each shard holds ONE rollback stash; a second trigger would cross-wire
       * the two confirms (confirm #1 resolving update #2). UPDATE_IN_FLIGHT. */
      g_mutex_lock (&s->prov_lock);
      awaiting = s->ota_await;
      g_mutex_unlock (&s->prov_lock);
      if (awaiting) {
        g_mutex_unlock (&modelmux_bin->lock);
        if (multi)
          MM_ERR ("MultiModelBin: multi-GPU reload of '%s' rejected: shard #%u "
              "is still awaiting nvinfer's confirm of the previous update -- "
              "NOTHING was swapped (UPDATE_IN_FLIGHT; retry shortly)", name,
              s->inst);
        else
          MM_ERR ("MultiModelBin: OTA reload of '%s' rejected: shard #%u is "
              "still awaiting nvinfer's confirm of the previous update -- "
              "NOTHING was swapped (UPDATE_IN_FLIGHT; retry shortly)", name,
              s->inst);
        g_free (new_key);
        return FALSE;
      }
      /* ---- ADMISSION CHECK 4/4 (multi only): this shard's device is covered ----
       * every shard's effective gpu must have a matching engine in the
       * {gpus[i] -> engines[i]} map (TRT engines are device-built). */
      if (multi) {
        for (i = 0; i < n_eng && !covered; i++)
          covered = (gpus[i] == gpu);
        if (!covered) {
          g_mutex_unlock (&modelmux_bin->lock);
          MM_ERR ("MultiModelBin: multi-GPU reload of '%s' rejected: no engine for "
              "shard #%u's gpu %d -- NOTHING was swapped (the group transitions "
              "together or not at all)", name, s->inst, gpu);
          g_free (new_key);
          return FALSE;
        }
      }
    }
  }
  /* apply the hot-swap to EVERY shard of EVERY live bin under this key (each
   * shard is its own nvinfer sharing the gie-id) with the engine the PICKER
   * selects for it -- pure property set, no pad/element teardown -> serving
   * streams keep flowing, gie-id preserved. Every triggered shard is enrolled
   * in ONE GROUP TRANSACTION: the update commits or rolls back as a unit once
   * the LAST confirm lands (see ModelMuxOtaGroup) -- a lone shard can never keep
   * serving a rejected engine. */
  group = modelmux_ota_group_new (modelmux_bin);
  for (b = 0; b < nbins; b++) {
    for (s = bins[b]; s; s = (ModelBin *) s->next_shard) {
      /* ENGINE PICKER -- the ONLY difference between the two entry points:
       *   SINGLE (gpus==NULL): the same config/engine is applied to every shard.
       *   MULTI  (gpus!=NULL): pick THIS shard's engine by its effective gpu
       *                        from the {gpus[i] -> engines[i]} map. */
      const gchar *use_cfg = config, *use_eng = engine;
      if (multi) {
        gint gpu = modelmux_shard_gpu_effective (modelmux_bin, s);
        use_cfg = NULL;
        use_eng = NULL;
        for (i = 0; i < n_eng && !use_eng; i++)
          if (gpus[i] == gpu)
            use_eng = engines[i];
        if (!use_eng)
          continue;               /* unreachable: admission verified coverage */
      }
      /* ACTUAL OTA: trigger nvinfer's in-place engine hot-swap on this shard
       * (pure property set, gie-id kept) and enroll it in the group transaction. */
      if (model_bin_reload_engine (s, use_cfg, use_eng, version, modelmux_bin,
              rename ? name : NULL, rename ? new_key : NULL, group))
        n++;
    }
  }
  /* the swap is TRIGGERED on every shard -> move the identity NOW, in the same
   * lock hold (status/routing never see a half-renamed chain); the group's
   * resolution renames it back if ANY shard fails (modelmux_ota_group_resolve_main). */
  if (rename && n && n == total)
    modelmux_bin_rename_model_locked (modelmux_bin, name, new_key);
  /* SEAL the group: `expected` is final only now, so a synchronous confirm
   * during the trigger loop could not resolve a half-built group. If every
   * confirm already landed, THIS path schedules the resolution. */
  group->live_key = g_strdup ((rename && n == total) ? new_key : name);
  if (rename && n && n == total) {
    group->from_key = g_strdup (name);
    group->to_key = g_strdup (new_key);
  }
  g_mutex_lock (&group->lock);
  group->expected = n;
  group->sealed = TRUE;
  resolve_now = (n > 0 && group->confirmed == n);
  g_mutex_unlock (&group->lock);
  if (n == 0) {                 /* nothing triggered -> no shard holds the group */
    modelmux_ota_group_free (group);
    group = NULL;
  }
  /* IDLE MODEL: with no stream, nvinfer never gets a buffer to commit the swap
   * (see modelmux_ota_idle_commit_main). Capture the live key while we hold the
   * lock so the fallback (armed after unlock) can drive the commit. */
  if (group && !resolve_now) {
    guint nstr = 0;
    for (b = 0; b < nbins; b++)
      for (s = bins[b]; s; s = (ModelBin *) s->next_shard)
        nstr += model_bin_num_streams (s);
    if (nstr == 0)
      idle_key = g_strdup (group->live_key);
  }
  g_mutex_unlock (&modelmux_bin->lock);
  if (group && resolve_now)
    modelmux_ota_group_schedule_resolve (group);
  if (idle_key) {                                 /* buffer-gated commit won't fire -> drive it after a grace */
    modelmux_ota_idle_commit_arm (modelmux_bin, idle_key);
    g_free (idle_key);
  }
  if (multi)
    MM_INFO ("  - MultiModelBin  OTA multi-GPU reload '%s' applied IN-PLACE to %u "
        "shard(s) across %u live bin(s) [per-device engines, same gie, no "
        "reroute; group commit awaits the per-shard confirms]", name, n, nbins);
  else
    MM_INFO ("  - MultiModelBin  OTA reload '%s' applied IN-PLACE to %u shard(s) "
        "across %u live bin(s) [same gie, no new bin, no reroute; group commit "
        "awaits the per-shard confirms]", name, n, nbins);
  g_free (new_key);
  return n > 0;
}

gboolean
modelmux_bin_reload_model (ModelMuxBin * modelmux_bin, const gchar * name,
    const gchar * config, const gchar * engine, const gchar * version)
{
  if (!modelmux_bin || !name || !*name)
    return FALSE;
  return modelmux_reload_model_common (modelmux_bin, name, config, engine, NULL, NULL, 0,
      version);
}

gboolean
modelmux_bin_reload_model_multi (ModelMuxBin * modelmux_bin, const gchar * name,
    const gint gpus[], const gchar * const engines[], guint n,
    const gchar * version)
{
  if (!modelmux_bin || !name || !*name || !gpus || !engines || !n)
    return FALSE;
  return modelmux_reload_model_common (modelmux_bin, name, NULL, NULL, gpus, engines, n,
      version);
}

guint
modelmux_bin_model_batch (ModelMuxBin * modelmux_bin, const gchar * key)
{
  ModelBin *model_bin;
  guint b = 0;
  if (!modelmux_bin || !key || !*key)
    return 0;
  g_mutex_lock (&modelmux_bin->lock);
  if ((model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, key)) ||
      (model_bin = g_hash_table_lookup (modelmux_bin->primary_pool->models, key)) ||
      (model_bin = g_hash_table_lookup (modelmux_bin->shadow_pool->models, key)))
    b = model_bin->max_streams;
  g_mutex_unlock (&modelmux_bin->lock);
  return b;
}

guint
modelmux_bin_model_gpus (ModelMuxBin * modelmux_bin, const gchar * key,
    gint * out, guint cap)
{
  ModelBin *model_bin, *s;
  guint n = 0, i;
  if (!modelmux_bin || !key || !*key || !out || !cap)
    return 0;
  g_mutex_lock (&modelmux_bin->lock);
  model_bin = modelmux_lookup_model_locked (modelmux_bin, key);
  for (s = model_bin; s; s = (ModelBin *) s->next_shard) {
    gint gpu = modelmux_shard_gpu_effective (modelmux_bin, s);
    gboolean seen = FALSE;
    for (i = 0; i < n && !seen; i++)
      seen = (out[i] == gpu);
    if (!seen && n < cap)
      out[n++] = gpu;
  }
  g_mutex_unlock (&modelmux_bin->lock);
  return n;
}

gboolean
modelmux_bin_model_on_gpu (ModelMuxBin * modelmux_bin, const gchar * key, gint gpu)
{
  ModelBin *model_bin, *s;
  gboolean on = FALSE;
  if (!modelmux_bin || !key || !*key || gpu < 0)
    return FALSE;
  g_mutex_lock (&modelmux_bin->lock);
  model_bin = modelmux_lookup_model_locked (modelmux_bin, key);
  for (s = model_bin; s && !on; s = (ModelBin *) s->next_shard)
    on = (modelmux_shard_gpu_effective (modelmux_bin, s) == gpu);
  g_mutex_unlock (&modelmux_bin->lock);
  return on;
}

ModelStatus
modelmux_bin_prepare_model (ModelMuxBin * modelmux_bin, gboolean shadow,
    const gchar * name)
{
  ModelPool *rb;
  ModelBin *model_bin;
  ModelStatus st;
  if (!modelmux_bin || !name || !*name)
    return MODEL_WARMING;
  rb = shadow ? modelmux_bin->shadow_pool : modelmux_bin->primary_pool;
  /* lock the models-hash mutation against the swap-probe lookups */
  g_mutex_lock (&modelmux_bin->lock);
  /* REFRESH a served-then-drained bin before a reroute reuses it. A per-model nvstreammux
   * that drained to 0 streams does NOT reliably resume when re-pinned mid-pipeline (it
   * wedges on "data flow before stream-start"). So if the target model's bin is idle
   * (READY/FAILED, 0 streams, single shard) AND has carried traffic before, destroy it
   * here (safe: main-loop/bus thread, 0 streams) so load_model below builds a FRESH,
   * never-drained bin. The reroute's make-before-break swap then attaches the fresh bin
   * (the old model keeps serving during the cached re-warm -> zero drop). */
  {
    gchar *keep_cfg = NULL;              /* the retired bin's ARTIFACTS survive the refresh */
    gint keep_gpu = MM_GPU_ANY;
    gboolean keep_pinned = FALSE;
    ModelBin *old = g_hash_table_lookup (rb->models, name);
    if (old && old->served && !old->next_shard &&
        model_bin_num_streams (old) == 0) {
      ModelStatus ost = model_bin_status (old);
      if (ost == MODEL_WARMED || ost == MODEL_FAILED) {
        MM_INFO ("    - ModelPool[%s]  '%s' was served then drained -> refresh to a "
            "FRESH bin before reuse (avoids drained-mux stall)", rb->role, name);
        /* the old bin may run a DERIVED per-version config (engine/gpu baked in
         * by model/load) and carry an explicit placement pin -- rebuilding from
         * the catalog base would silently swap it for the wrong artifact/device
         * while the registry still reports the requested one. Carry both over. */
        keep_cfg = g_strdup (old->config_file);
        keep_gpu = old->gpu;
        keep_pinned = old->pinned;
        g_hash_table_remove (rb->models, name);   /* frees key; bin freed next */
        modelmux_bin_forget_default_bin ((ModelMuxBin *) rb->owner, old);
        model_bin_free (old);
      }
    }
    /* idempotent: get-or-create PROMOTES a limbo model into this role, else creates +
     * warms a fresh bin (from the retired bin's config when this was a refresh).
     * Either way the model ends up in rb after this call. */
    if (!model_pool_load_model (rb, name, keep_cfg, 0)) {
      g_free (keep_cfg);
      g_mutex_unlock (&modelmux_bin->lock);
      /* creation failed (or the model is genuinely unknown): FAILED lets the
       * callers (route deferral, set_default admission) REJECT instead of
       * waiting forever on a "warming" that can never complete. */
      return MODEL_FAILED;
    }
    model_bin = g_hash_table_lookup (rb->models, name);
    if (model_bin && keep_gpu >= 0 && model_bin->gpu < 0) {
      /* re-pin BEFORE the fresh bin's first buffer (no streams are attached yet) */
      model_bin_set_gpu (model_bin, keep_gpu);
      model_bin->pinned = keep_pinned;
    }
    g_free (keep_cfg);
  }
  st = model_bin ? model_bin_status (model_bin) : MODEL_WARMING;
  g_mutex_unlock (&modelmux_bin->lock);
  return st;
}

gint
modelmux_bin_model_status (ModelMuxBin * modelmux_bin, gboolean shadow,
    const gchar * name)
{
  ModelPool *rb;
  ModelBin *model_bin;
  gint ret;
  if (!modelmux_bin || !name || !*name)
    return -1;
  g_mutex_lock (&modelmux_bin->lock);
  /* a type-less limbo model is role-agnostic -- report it for EITHER role query so
   * stream-add routing uses it (it is promoted into the role at attach). */
  if (modelmux_bin->limbo_models && (model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, name))) {
    ret = (gint) model_bin_status (model_bin);
    g_mutex_unlock (&modelmux_bin->lock);
    return ret;
  }
  rb = shadow ? modelmux_bin->shadow_pool : modelmux_bin->primary_pool;
  model_bin = g_hash_table_lookup (rb->models, name);   /* lookup only -- never create */
  ret = model_bin ? (gint) model_bin_status (model_bin) : -1;   /* -1 = not loaded */
  g_mutex_unlock (&modelmux_bin->lock);
  return ret;
}

/* Append the sensor names of every stream on ONE shard `s` to `out` (comma-sep,
 * appending to whatever is there). `names` = source_id -> sensor-name map.
 * Caller holds the lock (slot_stream is also atomic, so reads are safe). */
static void
append_shard_streams (GString * out, ModelBin * s, GHashTable * names)
{
  guint slot;
  if (!s || !s->slot_stream)
    return;
  for (slot = 0; slot < s->max_streams; slot++) {
    gint sid = g_atomic_int_get (&s->slot_stream[slot]);
    const gchar *nm;
    if (sid < 0)
      continue;
    nm = names ? (const gchar *) g_hash_table_lookup (names, GINT_TO_POINTER (sid)) : NULL;
    if (out->len)
      g_string_append_c (out, ',');
    if (nm && *nm)
      g_string_append (out, nm);
    else
      g_string_append_printf (out, "src-%d", sid);   /* fallback: raw source id */
  }
}

/* ...and for the whole shard chain of `base`. */
static void
append_serving_streams (GString * out, ModelBin * base, GHashTable * names)
{
  ModelBin *s;
  for (s = base; s; s = (ModelBin *) s->next_shard)
    append_shard_streams (out, s, names);
}

gboolean
modelmux_bin_key_serving (ModelMuxBin * modelmux_bin, const gchar * key,
    guint * streams, gchar ** streams_csv)
{
  GHashTable *tabs[3];
  ModelBin *base = NULL, *s;
  guint i, total = 0;
  GString *csv = NULL;
  if (streams)
    *streams = 0;
  if (streams_csv)
    *streams_csv = NULL;
  if (!modelmux_bin || !key || !*key)
    return FALSE;
  g_mutex_lock (&modelmux_bin->lock);
  tabs[0] = modelmux_bin->limbo_models;
  tabs[1] = modelmux_bin->primary_pool ? modelmux_bin->primary_pool->models : NULL;
  tabs[2] = modelmux_bin->shadow_pool ? modelmux_bin->shadow_pool->models : NULL;
  for (i = 0; i < 3 && !base; i++)                 /* a key lives in exactly one */
    if (tabs[i])
      base = (ModelBin *) g_hash_table_lookup (tabs[i], key);
  if (base) {
    for (s = base; s; s = (ModelBin *) s->next_shard)
      total += model_bin_num_streams (s);
    if (streams_csv && total) {
      csv = g_string_new (NULL);
      append_serving_streams (csv, base, modelmux_bin->stream_names);
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);
  if (!base)
    return FALSE;
  if (streams)
    *streams = total;
  if (streams_csv)
    *streams_csv = csv ? g_string_free (csv, FALSE) : NULL;
  return TRUE;
}

/* Bare `name`: the comma-separated sensor names of every stream serving ANY
 * loaded version of the name (empty string if none). Owned; g_free it. */
gchar *
modelmux_bin_name_serving_streams (ModelMuxBin * modelmux_bin, const gchar * name)
{
  GHashTable *maps[3];
  GString *csv;
  GHashTableIter it;
  gpointer k, v;
  gsize nlen;
  guint i;
  if (!modelmux_bin || !name || !*name)
    return g_strdup ("");
  nlen = strlen (name);
  csv = g_string_new (NULL);
  g_mutex_lock (&modelmux_bin->lock);
  maps[0] = modelmux_bin->limbo_models;
  maps[1] = modelmux_bin->primary_pool ? modelmux_bin->primary_pool->models : NULL;
  maps[2] = modelmux_bin->shadow_pool ? modelmux_bin->shadow_pool->models : NULL;
  for (i = 0; i < 3; i++) {
    if (!maps[i])
      continue;
    g_hash_table_iter_init (&it, maps[i]);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      const gchar *key = (const gchar *) k;
      if (strncmp (key, name, nlen) == 0 && key[nlen] == MM_MODEL_KEY_SEP)
        append_serving_streams (csv, (ModelBin *) v, modelmux_bin->stream_names);
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);
  return g_string_free (csv, FALSE);
}

gboolean
modelmux_bin_unload_model (ModelMuxBin * modelmux_bin, const gchar * name)
{
  gboolean ret = FALSE;
  if (!modelmux_bin || !name || !*name)
    return FALSE;
  g_mutex_lock (&modelmux_bin->lock);
  /* TYPE-LESS: free the model wherever it lives. Limbo first (still unassigned),
   * else whichever pool promoted it. Idle-only enforced. */
  if (modelmux_bin->limbo_models) {
    ModelBin *model_bin = g_hash_table_lookup (modelmux_bin->limbo_models, name);
    if (model_bin) {
      /* EVERY shard must be idle (a limbo gpus[] group is a chain too) */
      ModelBin *s, *next;
      gint st = model_bin_status (model_bin);
      gboolean idle = TRUE;
      for (s = model_bin; s && idle; s = (ModelBin *) s->next_shard) {
        gint sst = model_bin_status (s);
        idle = ((sst == MODEL_WARMED || sst == MODEL_FAILED) &&
            model_bin_num_streams (s) == 0);
      }
      if (idle) {
        MM_INFO ("  - MultiModelBin  unload idle model '%s' (limbo copy -> destroyed)",
            name);
        /* app-facing completion notice: mirror the pool path (model_pool_unload_model).
         * A REST-loaded model that was never routed is torn down from HERE, so the
         * event must be posted on this branch too -- else `model unloaded` never
         * reaches the app. Post BEFORE the frees, from the bin (still parented -> has
         * the pipeline bus), reading name/version/gpu while the base bin is alive. */
        modelmux_post_model_event (model_bin->bin, "model-unloaded", model_bin->name,
            model_bin->version, model_bin_infer_gpu (model_bin), TRUE, NULL);
        /* detach this bin from either pool's default-instance cache before freeing
         * it (clears default_ref.bin only; the default identity/key stays) so no
         * dangling pointer to the freed bin survives. */
        modelmux_bin_forget_default_bin (modelmux_bin, model_bin);
        g_hash_table_remove (modelmux_bin->limbo_models, name);   /* frees key only */
        for (s = model_bin; s; s = next) {               /* free the whole shard chain --
                                                   * DEFERRED: TRT teardown takes 100s of
                                                   * ms..s and we hold modelmux_bin->lock here */
          next = (ModelBin *) s->next_shard;
          modelmux_schedule_model_bin_free (modelmux_bin, s);
        }
        ret = TRUE;
      } else {
        MM_WARN ("MultiModelBin: cannot unload limbo '%s' -- status=%s (allowed only "
            "when READY/FAILED, not mid-load)", name, model_status_str (st));
      }
      g_mutex_unlock (&modelmux_bin->lock);
      return ret;
    }
  }
  /* not in limbo -> it was promoted into a pool. A key lives in at most ONE pool,
   * so try primary then shadow ('||' short-circuits on the first hit; a miss
   * returns FALSE and falls through). Each call is idle-only (refuses SERVING /
   * mid-load), removes the name@version entry from that pool's models table and
   * deferred-frees the whole shard chain. ret = TRUE if either pool unloaded it. */
  ret = model_pool_unload_model (modelmux_bin->primary_pool, name) ||
        model_pool_unload_model (modelmux_bin->shadow_pool, name);
  g_mutex_unlock (&modelmux_bin->lock);
  return ret;
}

/* is `gpu` in gpus[0..n)? */
static gboolean
gpu_in_set (gint gpu, const gint * gpus, guint n)
{
  guint i;
  for (i = 0; i < n; i++)
    if (gpus[i] == gpu)
      return TRUE;
  return FALSE;
}

/* GPU-scoped teardown of the shard chain stored under `name` in table `ht`
 * (limbo OR a pool's models table). Caller holds modelmux_bin->lock. Returns:
 *    >=0 : # per-gpu instances removed (partial -- others kept serving)
 *    -1  : `name` is not in THIS table   (caller tries the next container)
 *    -2  : loaded here, but no instance sits on any requested gpu
 *    -3  : a targeted instance is serving/mid-load -> refused, nothing changed
 *          (sgpu / sstr out-params report the first offender)
 *    -4  : the set covered EVERY instance -> whole entry removed (full unload) */
static gint
chain_unload_gpus (ModelMuxBin * mb, GHashTable * ht, const gchar * role,
    const gchar * name, const gint * gpus, guint n_gpus,
    gint * sgpu, guint * sstr, gchar ** scsv)
{
  ModelBin *base = ht ? (ModelBin *) g_hash_table_lookup (ht, name) : NULL;
  ModelBin *s, *prev, *next, *newbase;
  guint total = 0, match = 0, removed = 0;
  if (!base)
    return -1;                                  /* not in this container */

  /* Phase 1 -- validate: every instance on a requested gpu must be idle
   * (READY/FAILED, 0 streams). A SERVING or still-warming target refuses the
   * WHOLE op (report the first offender for a clear "still inferring" error). */
  for (s = base; s; s = (ModelBin *) s->next_shard) {
    total++;
    if (gpu_in_set (s->mux_gpu, gpus, n_gpus)) {
      gint st = model_bin_status (s);
      guint ns = model_bin_num_streams (s);
      match++;
      if ((st != MODEL_WARMED && st != MODEL_FAILED) || ns > 0) {
        if (sgpu) *sgpu = s->mux_gpu;
        if (sstr) *sstr = ns;
        if (scsv && ns) {                       /* names of just THIS shard's streams */
          GString *c = g_string_new (NULL);
          append_shard_streams (c, s, mb ? mb->stream_names : NULL);
          *scsv = g_string_free (c, FALSE);
        }
        MM_WARN ("ModelPool[%s]: refusing gpu-scoped unload of '%s' on gpu %d -- "
            "that instance is %s with %u stream(s) (allowed only when the "
            "targeted instance is idle: READY/FAILED, 0 streams; route/drain "
            "its streams away first)", role, name, s->mux_gpu,
            model_status_str (st), ns);
        return -3;
      }
    }
  }
  if (match == 0)
    return -2;                                  /* nothing on the requested gpu(s) */

  if (match == total) {
    /* the set drains every instance -> full unload of the whole chain */
    MM_INFO ("    - ModelPool[%s]  gpu-scoped unload of '%s' covers all %u "
        "instance(s) -> whole version removed", role, name, total);
    for (s = base; s; s = (ModelBin *) s->next_shard)
      modelmux_bin_forget_default_bin (mb, s);
    g_hash_table_remove (ht, name);             /* frees the key only */
    for (s = base; s; s = next) {               /* deferred free of the whole chain */
      next = (ModelBin *) s->next_shard;
      modelmux_schedule_model_bin_free (mb, s);
    }
    return -4;
  }

  /* Phase 2 -- partial: unlink & free ONLY the matching instances; the rest keep
   * serving. The chain head (base) may itself be removed -> track a new head. */
  newbase = base;
  prev = NULL;
  for (s = base; s; s = next) {
    next = (ModelBin *) s->next_shard;
    if (gpu_in_set (s->mux_gpu, gpus, n_gpus)) {
      if (prev)
        prev->next_shard = s->next_shard;       /* unlink interior/tail instance */
      else
        newbase = (ModelBin *) s->next_shard;   /* removing the current head      */
      MM_INFO ("    - ModelPool[%s]  gpu-scoped unload: instance of '%s' on "
          "gpu %d -> freed (%u instance(s) kept)", role, name, s->mux_gpu,
          total - match);
      modelmux_bin_forget_default_bin (mb, s);
      modelmux_schedule_model_bin_free (mb, s); /* deferred: TRT teardown under lock */
      removed++;                                /* prev unchanged: s is gone */
    } else {
      prev = s;
    }
  }
  if (newbase != base)                          /* head changed -> repoint the entry.
                                                 * value-destroy is NULL, so the old
                                                 * base bin is NOT freed here (it was
                                                 * already deferred-freed above); the
                                                 * temp key is freed by key-destroy. */
    g_hash_table_insert (ht, g_strdup (name), newbase);
  return (gint) removed;
}

ModelMuxUnloadGpuResult
modelmux_bin_unload_model_gpus (ModelMuxBin * modelmux_bin, const gchar * name,
    const gint * gpus, guint n_gpus, gint * serving_gpu, guint * serving_streams,
    gchar ** serving_csv)
{
  gint r;
  if (serving_gpu)
    *serving_gpu = -1;
  if (serving_streams)
    *serving_streams = 0;
  if (serving_csv)
    *serving_csv = NULL;
  if (!modelmux_bin || !name || !*name || !gpus || n_gpus == 0)
    return MM_UNLOAD_GPU_ABSENT;

  g_mutex_lock (&modelmux_bin->lock);
  /* a key lives in exactly ONE container: limbo (still unassigned) or whichever
   * pool promoted it. -1 = "not here" -> fall through to the next. */
  r = chain_unload_gpus (modelmux_bin, modelmux_bin->limbo_models, "limbo",
      name, gpus, n_gpus, serving_gpu, serving_streams, serving_csv);
  if (r == -1 && modelmux_bin->primary_pool)
    r = chain_unload_gpus (modelmux_bin, modelmux_bin->primary_pool->models,
        "primary", name, gpus, n_gpus, serving_gpu, serving_streams, serving_csv);
  if (r == -1 && modelmux_bin->shadow_pool)
    r = chain_unload_gpus (modelmux_bin, modelmux_bin->shadow_pool->models,
        "shadow", name, gpus, n_gpus, serving_gpu, serving_streams, serving_csv);
  g_mutex_unlock (&modelmux_bin->lock);

  switch (r) {
    case -1:
      return MM_UNLOAD_GPU_ABSENT;
    case -2:
      return MM_UNLOAD_GPU_NO_MATCH;
    case -3:
      return MM_UNLOAD_GPU_SERVING;
    case -4:
      return MM_UNLOAD_GPU_ALL;
    default:
      return MM_UNLOAD_GPU_REMOVED;             /* r >= 0 */
  }
}

guint
modelmux_bin_unload_versions (ModelMuxBin * modelmux_bin, const gchar * name,
    guint * present)
{
  GHashTable *maps[3];
  GPtrArray *keys;
  GHashTableIter it;
  gpointer k, v;
  guint i, count = 0;
  gsize nlen;

  if (present)
    *present = 0;
  if (!modelmux_bin || !name || !*name)
    return 0;
  nlen = strlen (name);
  keys = g_ptr_array_new_with_free_func (g_free);
  /* collect every "name@*" key under the lock, then unload each (unload re-locks). */
  g_mutex_lock (&modelmux_bin->lock);
  maps[0] = modelmux_bin->limbo_models;
  maps[1] = modelmux_bin->primary_pool ? modelmux_bin->primary_pool->models : NULL;
  maps[2] = modelmux_bin->shadow_pool ? modelmux_bin->shadow_pool->models : NULL;
  for (i = 0; i < 3; i++) {
    if (!maps[i])
      continue;
    g_hash_table_iter_init (&it, maps[i]);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      const gchar *key = (const gchar *) k;
      /* "<name>@<version>" prefix match: relies on a model name never containing the key
       * separator '@' (MM_MODEL_KEY_SEP) -- guaranteed because catalog names are validated. */
      if (strncmp (key, name, nlen) == 0 && key[nlen] == MM_MODEL_KEY_SEP)
        g_ptr_array_add (keys, g_strdup (key));
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);
  if (present)
    *present = keys->len;                 /* how many "name@*" versions exist */
  for (i = 0; i < keys->len; i++)
    if (modelmux_bin_unload_model (modelmux_bin, (const gchar *) g_ptr_array_index (keys, i)))
      count++;                            /* serving versions are skipped (return FALSE) */
  g_ptr_array_free (keys, TRUE);
  return count;
}

guint
modelmux_bin_num_streams (ModelMuxBin * modelmux_bin)
{
  guint n;
  if (!modelmux_bin || !modelmux_bin->stream_wiring)
    return 0;
  g_mutex_lock (&modelmux_bin->lock);
  n = g_hash_table_size (modelmux_bin->stream_wiring);
  g_mutex_unlock (&modelmux_bin->lock);
  return n;
}

/* Pre-create + warm ONLY the configured default primary (+ shadow if set) in their pools, so
 * both are READY before any stream is attached. A version's checkpoint/config is resolved ONCE
 * from its [model-<name>-<version>] block (modelmux_config_resolve_version_cfg) -- NULL => catalog base.
 *
 * Per-sensor [stream-model-*] bound models are NOT preloaded here: they are LAZY -- warmed on
 * demand when their mapped stream is added (stream/add), with passthrough while warming and a
 * promote-when-ready (same deferred-promote path as a runtime model switch). Only the defaults
 * are eager (and unload-protected). */
gboolean
modelmux_bin_load_defaults (ModelMuxBin * modelmux_bin, gboolean wait)
{
  /* MM_DEFAULT_WARM_WAIT_MS: cap the blocking wait ABOVE the warm thread's own
   * 30-min build timeout, so the thread always resolves to WARMED/FAILED first
   * and this loop can never wedge (it observes that terminal status). */
  const guint MM_DEFAULT_WARM_WAIT_MS = 35u * 60u * 1000u;
  gboolean ok = TRUE;
  if (!modelmux_bin)
    return FALSE;

  {
    DefaultModelRef *default_primary = &modelmux_bin->primary_pool->default_ref;
    DefaultModelRef *default_shadow = &modelmux_bin->shadow_pool->default_ref;
    if (default_primary->name) {
      gboolean dfail = FALSE;
      gchar *derived = modelmux_config_resolve_version_cfg (modelmux_bin->config, default_primary->name,
          default_primary->version, &dfail);
      if (dfail) {
        /* explicit per-version engine/gpu that could not be derived: preloading the
         * BASE config would silently run the wrong artifact/device -- refuse. */
        MM_ERR ("MultiModelBin: default PRIMARY '%s' NOT preloaded -- its per-version "
            "engine/gpu override failed to derive (DERIVE_FAILED; see the error above)", default_primary->key);
        ok = FALSE;
      } else {
        MM_INFO ("  - MultiModelBin  preload default PRIMARY model '%s'%s", default_primary->key,
            derived ? " (per-version artifact)" : "");
        model_pool_load_model (modelmux_bin->primary_pool, default_primary->key, derived, 0);  /* derived NULL => catalog base */
        /* refresh the INSTANCE cache now the identity is materialized (under the
         * bin lock: streaming-thread readers may look at the ref concurrently) */
        g_mutex_lock (&modelmux_bin->lock);
        default_primary->bin = mm_default_bin_lookup (modelmux_bin, FALSE, default_primary->key);
        g_mutex_unlock (&modelmux_bin->lock);
        /* wait-for-default-models: block until this default's engine is WARMED so
         * it infers from the first frame; a FAILED warm makes the caller fail the
         * PLAYING transition (fail loud on a bad default). */
        if (wait) {
          if (!default_primary->bin ||
              !model_bin_wait_ready (default_primary->bin, MM_DEFAULT_WARM_WAIT_MS)) {
            MM_ERR ("MultiModelBin: default PRIMARY '%s' did NOT warm up (engine "
                "build/load failed) -- wait-for-default-models", default_primary->key);
            ok = FALSE;
          } else {
            MM_INFO ("  - MultiModelBin  default PRIMARY '%s' WARMED (ready before "
                "streaming)", default_primary->key);
          }
        }
      }
      g_free (derived);
    }
    if (default_shadow->name) {
      gboolean dfail = FALSE;
      gchar *derived = modelmux_config_resolve_version_cfg (modelmux_bin->config, default_shadow->name,
          default_shadow->version, &dfail);
      if (dfail) {
        MM_ERR ("MultiModelBin: default SHADOW '%s' NOT preloaded -- its per-version "
            "engine/gpu override failed to derive (DERIVE_FAILED; see the error above)", default_shadow->key);
        ok = FALSE;
      } else {
        MM_INFO ("  - MultiModelBin  preload default SHADOW model '%s'%s", default_shadow->key,
            derived ? " (per-version artifact)" : "");
        model_pool_load_model (modelmux_bin->shadow_pool, default_shadow->key, derived, 0);   /* derived NULL => catalog base */
        g_mutex_lock (&modelmux_bin->lock);
        default_shadow->bin = mm_default_bin_lookup (modelmux_bin, TRUE, default_shadow->key);
        g_mutex_unlock (&modelmux_bin->lock);
        if (wait) {
          if (!default_shadow->bin ||
              !model_bin_wait_ready (default_shadow->bin, MM_DEFAULT_WARM_WAIT_MS)) {
            MM_ERR ("MultiModelBin: default SHADOW '%s' did NOT warm up (engine "
                "build/load failed) -- wait-for-default-models", default_shadow->key);
            ok = FALSE;
          } else {
            MM_INFO ("  - MultiModelBin  default SHADOW '%s' WARMED (ready before "
                "streaming)", default_shadow->key);
          }
        }
      }
      g_free (derived);
    }
  }
  return ok;
}

/* Create + warm a model DIRECTLY into a role pool (primary/shadow), bypassing limbo --
 * the runtime equivalent of a config-time default preload. Used by model/load when a
 * set_default_primary/shadow flag designates the role up front. `cfg_override` is the
 * per-version/engine-derived config (NULL => catalog base). Returns the model status. */
ModelStatus
modelmux_bin_load_into_pool (ModelMuxBin * modelmux_bin, gboolean shadow,
    const gchar * key, const gchar * cfg_override)
{
  ModelPool *rb;
  ModelBin *model_bin;
  ModelStatus st;
  if (!modelmux_bin || !key || !*key)
    return MODEL_WARMING;
  rb = shadow ? modelmux_bin->shadow_pool : modelmux_bin->primary_pool;
  g_mutex_lock (&modelmux_bin->lock);
  if (!model_pool_load_model (rb, key, cfg_override, 0)) {
    g_mutex_unlock (&modelmux_bin->lock);
    MM_ERR ("  - MultiModelBin  load-into-%s '%s' FAILED (not in catalog / no config)",
        rb->role, key);
    return MODEL_WARMING;
  }
  model_bin = g_hash_table_lookup (rb->models, key);
  st = model_bin ? model_bin_status (model_bin) : MODEL_WARMING;
  g_mutex_unlock (&modelmux_bin->lock);
  MM_INFO ("  - MultiModelBin  model '%s' loaded straight into %s pool (runtime default)",
      key, rb->role);
  return st;
}

/* Re-designate the RUNTIME default for a role to (name, version). "Default" is derived
 * everywhere from these config pointers (routing fallback, unload protection, models_table,
 * dedup guard), so swapping them here makes every subsystem treat the new model as default
 * on its next read -- and the OLD default auto-demotes (loses protection -> idle-unloadable).
 * Done under modelmux_bin->lock because the pointers are read on streaming threads (dump/unload).
 * CONCURRENCY: this self-acquires modelmux_bin->lock -- callers MUST NOT already hold it (today the only
 * caller is the lock-free control plane; a future bin.c caller holding modelmux_bin->lock would deadlock). */
void
modelmux_bin_set_default (ModelMuxBin * modelmux_bin, gboolean shadow,
    const gchar * name, const gchar * version, gint gpu)
{
  DefaultModelRef *ref;
  gchar *key;
  if (!modelmux_bin || !modelmux_bin->primary_pool || !modelmux_bin->shadow_pool ||
      !name || !*name)
    return;
  key = model_key (name, version ? version : MM_MODEL_VERSION_DEFAULT);
  g_mutex_lock (&modelmux_bin->lock);
  ref = shadow ? &modelmux_bin->shadow_pool->default_ref
               : &modelmux_bin->primary_pool->default_ref;
  /* gpu >= 0 persists the request's pin; gpu < 0 keeps the pin on a same-identity
   * re-designation (OTA version rename) and resets it on an identity change (the
   * pin belonged to the old default) -- handled inside the assign helper. The bin
   * cache is re-derived in the same critical section so identity and instance can
   * never be observed disagreeing. */
  mm_default_ref_assign (ref, name, version, gpu,
      mm_default_bin_lookup (modelmux_bin, shadow, key));
  g_mutex_unlock (&modelmux_bin->lock);
  g_free (key);
  MM_INFO ("  - MultiModelBin  runtime default %s is now '%s@%s'",
      shadow ? "SHADOW" : "PRIMARY", name, version ? version : MM_MODEL_VERSION_DEFAULT);
}

/* Collect the source-ids of streams currently routed to `key` in the given role (primary or
 * shadow), including a passthrough stream still awaiting promotion to it (pending_model). The
 * caller (a runtime default switch) reroutes these off the demoted old default onto the new one.
 * Returns the count; fills `out` (caller-provided, capacity `cap`).
 * CONCURRENCY: this self-acquires modelmux_bin->lock -- callers MUST NOT already hold it (see set_default). */
guint
modelmux_bin_streams_on_model (ModelMuxBin * modelmux_bin, gboolean shadow,
    const gchar * key, guint * out, guint cap, gboolean default_only)
{
  GHashTableIter it;
  gpointer k, v;
  guint n = 0, matched = 0, pinned = 0;
  if (!modelmux_bin || !modelmux_bin->stream_wiring || !key || !*key || !out || !cap)
    return 0;
  g_mutex_lock (&modelmux_bin->lock);
  g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
  /* count ALL matches (don't break at cap) so an overflow is visible; fill `out` up to cap. */
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelMuxStreamEntry *se = (ModelMuxStreamEntry *) v;
    const ModelMuxRoleAttach *ra = shadow ? &se->shad : &se->prim;
    if ((ra->model && g_strcmp0 (ra->model, key) == 0) ||
        (ra->pending_model && g_strcmp0 (ra->pending_model, key) == 0)) {
      /* default_only: skip PINNED lanes (an explicit ref/route/binding named the
       * model, even though it equals the default) -- a default switch must not
       * move them. Counted so the skip is VISIBLE, never silent. */
      if (default_only && !ra->via_default) {
        pinned++;
        continue;
      }
      matched++;
      if (n < cap)
        out[n++] = se->stream_id;
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);
  if (pinned)
    MM_INFO ("  - MultiModelBin  %u stream(s) on '%s' are PINNED to it (explicit "
        "ref/route/binding) -> left in place by the default switch (route them "
        "explicitly to move them)", pinned, key);
  if (matched > cap)
    MM_WARN ("streams_on_model: %u streams on '%s' exceed the caller cap %u -- %u stream(s) will "
        "NOT be rerouted by this default switch", matched, key, cap, matched - cap);
  return n;
}

GstPad *
modelmux_bin_get_sink (ModelMuxBin * modelmux_bin)
{
  return modelmux_bin ? modelmux_bin->sink_ghost : NULL;     /* single batched entry pad */
}

/* Tracked one-shot main-loop sources for dynamic graph operations. This mirrors the
 * pending_frees teardown guard, but is generic enough for operation idles and their
 * safety timeouts. free_lock protects only the list/source ids; callbacks and cancel
 * destructors run after releasing it, preserving the modelmux_bin->lock -> free_lock hierarchy. */
/* Lifecycle state of a tracked source. The wrapper struct is OWNED BY GLIB (freed
 * exactly once by the source's GDestroyNotify, which never runs while the dispatch
 * callback is executing) -- so a canceller racing an already-dispatching callback
 * can no longer free the wrapper under it. `state` decides, in the notify, whether
 * the data destroy runs: it does NOT run when the func consumed the data (DISPATCHED)
 * or when an explicit cancel keeps the data alive elsewhere (CANCELLED). */
enum {
  MM_TS_PENDING = 0,          /* never dispatched, never cancelled -> destroy runs */
  MM_TS_DISPATCHED,           /* func ran (or will run) and consumed data          */
  MM_TS_CANCELLED,            /* cancelled, data owned elsewhere -> no destroy     */
  MM_TS_CANCELLED_DESTROY     /* cancelled at teardown -> destroy runs             */
};

typedef struct {
  ModelMuxBin *modelmux_bin;
  guint       source_id;
  guint      *slot;           /* optional owner field to clear, e.g. co->timeout_id */
  GSourceFunc func;
  gpointer    data;
  GDestroyNotify destroy;     /* minimal cancel cleanup; must not take modelmux_bin->lock */
  gint        state;          /* MM_TS_*; written under modelmux_bin->free_lock            */
} ModelMuxTrackedSource;

/* GDestroyNotify of the underlying GSource: the single owner of the wrapper.
 * Runs the data destroy for never-dispatched / teardown-cancelled sources. Must
 * not touch ts->modelmux_bin (it may already be freed on the teardown path); de-listing is
 * the dispatcher's/canceller's job, done before the source is removed. */
static void
modelmux_tracked_source_notify (gpointer udata)
{
  ModelMuxTrackedSource *ts = (ModelMuxTrackedSource *) udata;
  if ((ts->state == MM_TS_PENDING || ts->state == MM_TS_CANCELLED_DESTROY) &&
      ts->destroy)
    ts->destroy (ts->data);
  g_free (ts);
}

static gboolean
modelmux_tracked_source_dispatch (gpointer udata)
{
  ModelMuxTrackedSource *ts = (ModelMuxTrackedSource *) udata;
  ModelMuxBin *modelmux_bin = ts->modelmux_bin;
  gboolean run;
  gboolean ret = G_SOURCE_REMOVE;

  g_mutex_lock (&modelmux_bin->free_lock);
  modelmux_bin->pending_sources = g_list_remove (modelmux_bin->pending_sources, ts);
  if (ts->slot && *ts->slot == ts->source_id)
    *ts->slot = 0;
  /* claim the source: if a canceller got in first (state != PENDING) the func is
   * skipped and the notify honours the canceller's destroy decision. On teardown
   * (shutting_down) hand the data to the destroy instead of running the func. */
  run = (ts->state == MM_TS_PENDING);
  if (run) {
    if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
      ts->state = MM_TS_CANCELLED_DESTROY;
      run = FALSE;
    } else {
      ts->state = MM_TS_DISPATCHED;
    }
  }
  g_mutex_unlock (&modelmux_bin->free_lock);

  if (run && ts->func)
    ret = ts->func (ts->data);
  return ret;      /* wrapper freed by modelmux_tracked_source_notify, never here */
}

static guint
modelmux_tracked_source_add (ModelMuxBin * modelmux_bin, gboolean timeout, guint interval_ms,
    GSourceFunc func, gpointer data, GDestroyNotify destroy, guint * slot)
{
  ModelMuxTrackedSource *ts;

  if (!modelmux_bin || g_atomic_int_get (&modelmux_bin->shutting_down))
    return 0;

  ts = g_new0 (ModelMuxTrackedSource, 1);
  ts->modelmux_bin = modelmux_bin;
  ts->slot = slot;
  ts->func = func;
  ts->data = data;
  ts->destroy = destroy;

  g_mutex_lock (&modelmux_bin->free_lock);
  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    g_mutex_unlock (&modelmux_bin->free_lock);
    g_free (ts);
    return 0;
  }
  ts->source_id = timeout ?
      g_timeout_add_full (G_PRIORITY_DEFAULT, interval_ms,
          modelmux_tracked_source_dispatch, ts, modelmux_tracked_source_notify) :
      g_idle_add_full (G_PRIORITY_DEFAULT_IDLE,
          modelmux_tracked_source_dispatch, ts, modelmux_tracked_source_notify);
  if (slot)
    *slot = ts->source_id;
  modelmux_bin->pending_sources = g_list_prepend (modelmux_bin->pending_sources, ts);
  g_mutex_unlock (&modelmux_bin->free_lock);
  return ts->source_id;
}

static guint
modelmux_tracked_idle_add (ModelMuxBin * modelmux_bin, GSourceFunc func, gpointer data,
    GDestroyNotify destroy, guint * slot)
{
  return modelmux_tracked_source_add (modelmux_bin, FALSE, 0, func, data, destroy, slot);
}

static guint
modelmux_tracked_timeout_add (ModelMuxBin * modelmux_bin, guint interval_ms,
    GSourceFunc func, gpointer data, GDestroyNotify destroy, guint * slot)
{
  return modelmux_tracked_source_add (modelmux_bin, TRUE, interval_ms, func, data, destroy, slot);
}

/* Deferred (main-loop) ModelBin destruction. model_bin_free BLOCKS: it joins the
 * warm-up thread and sets the whole nvinfer bin to NULL (TensorRT context teardown,
 * task joins -- hundreds of ms to seconds). Several reclaim paths reach it on a
 * STREAMING thread inside a probe while holding modelmux_bin->lock (e.g. the CLEAR probe ->
 * modelmux_pool_unload_if_idle), which stalls that lane AND every modelmux_bin->lock waiter for the
 * whole teardown -- the exact set_state-under-lock trap the deferred branch-free
 * machinery exists to avoid (see modelmux_schedule_branch_free). The bin is already
 * unhooked from every map/shard-chain by the caller, so destruction is safely
 * deferrable. On shutdown (idle not schedulable / cancelled) the destroy runs
 * synchronously on the teardown thread -- the historical behavior. */
static gboolean
model_bin_free_main (gpointer data)
{
  model_bin_free ((ModelBin *) data);
  return G_SOURCE_REMOVE;
}

/* Tear down ONE ModelBin safely. Splits the work so the caller (which holds
 * modelmux_bin->lock, sometimes on a streaming thread) never blocks:
 *   NOW (cheap, inline)   -- drop the default-ref instance cache, then UNPARENT
 *                            the bin from its container so its deterministic
 *                            element name is freed immediately (a reload of the
 *                            same name@version can reuse it) and the graph is
 *                            detached before the slow teardown.
 *   DEFERRED (main loop)  -- model_bin_free (joins the warm-up thread + TRT/
 *                            element teardown, 100s of ms..s) runs off-lock.
 * Falls back to a synchronous free if the idle source can't be scheduled (e.g.
 * during shutdown), so the bin is never leaked. Call this per shard. */
static void
modelmux_schedule_model_bin_free (ModelMuxBin * modelmux_bin, ModelBin * model_bin)
{
  if (!model_bin)
    return;
  /* destroy choke: drop any default-ref INSTANCE cache pointing at this bin
   * (identity designation stays; a later load of the same key re-points it) */
  modelmux_bin_forget_default_bin (modelmux_bin, model_bin);
  /* Unparent the zombie NOW (cheap, non-blocking), keeping our own ref for the
   * deferred destruction. Two reasons:
   *  - element names are deterministic ("nvmodelbin-<role>-<name>-<ver>"): while
   *    the zombie stayed in the container, an immediate reload of the same
   *    (name,version) failed gst_bin_add on the duplicate name -> FAILED model;
   *  - the deferred free must not deref/mutate the container (gst_bin_remove on
   *    GST_OBJECT_PARENT) concurrently with a teardown already dismantling it.
   * After this the bin is fully detached from the graph; model_bin_free's
   * parentless branch releases our ref. */
  if (model_bin->bin && GST_OBJECT_PARENT (model_bin->bin)) {
    gst_object_ref (model_bin->bin);
    gst_bin_remove (GST_BIN (GST_OBJECT_PARENT (model_bin->bin)), model_bin->bin);
  }
  if (!modelmux_bin || !modelmux_tracked_idle_add (modelmux_bin, model_bin_free_main, model_bin,
          (GDestroyNotify) model_bin_free, NULL))
    model_bin_free (model_bin);
}

static gboolean
modelmux_tracked_source_cancel (ModelMuxBin * modelmux_bin, guint * slot, gboolean run_destroy)
{
  GList *l;
  ModelMuxTrackedSource *ts = NULL;
  guint source_id;

  if (!modelmux_bin || !slot || !*slot)
    return FALSE;

  g_mutex_lock (&modelmux_bin->free_lock);
  source_id = *slot;
  for (l = modelmux_bin->pending_sources; l; l = l->next) {
    ModelMuxTrackedSource *cur = (ModelMuxTrackedSource *) l->data;
    if (cur->source_id == source_id) {
      ts = cur;
      modelmux_bin->pending_sources = g_list_delete_link (modelmux_bin->pending_sources, l);
      break;
    }
  }
  *slot = 0;
  if (ts) {
    if (ts->slot && *ts->slot == source_id)
      *ts->slot = 0;                 /* before the remove: destroy fns gate their
                                      * tracked-cancel on the slot, so a synchronous
                                      * notify never re-takes free_lock */
    /* record the destroy decision for the notify; a dispatch already past its
     * claim keeps MM_TS_DISPATCHED and this store is skipped (list miss above
     * means we never get here with a claimed ts). */
    ts->state = run_destroy ? MM_TS_CANCELLED_DESTROY : MM_TS_CANCELLED;
    /* remove UNDER free_lock: ts is un-claimed here (claimed ones de-listed
     * themselves), so the source cannot auto-destroy under us -- a parked
     * dispatch is blocked on this very lock before its claim. Removing after
     * unlock would let it resume, skip the func, return REMOVE and destroy the
     * source first -> our remove then hits a stale id (GLib CRITICAL). */
    g_source_remove (source_id);
  }
  g_mutex_unlock (&modelmux_bin->free_lock);

  return ts != NULL;
}

static void
modelmux_tracked_sources_cancel_all (ModelMuxBin * modelmux_bin)
{
  GList *pending, *l;

  if (!modelmux_bin)
    return;

  g_mutex_lock (&modelmux_bin->free_lock);
  pending = modelmux_bin->pending_sources;
  modelmux_bin->pending_sources = NULL;
  for (l = pending; l; l = l->next) {
    ModelMuxTrackedSource *ts = (ModelMuxTrackedSource *) l->data;
    if (ts->slot && *ts->slot == ts->source_id)
      *ts->slot = 0;                       /* MUST precede the remove: destroy fns
                                            * gate their tracked-cancel on the slot,
                                            * so a synchronous notify never re-takes
                                            * free_lock (self-deadlock) */
    ts->state = MM_TS_CANCELLED_DESTROY;   /* notify runs the data destroy */
    /* remove UNDER free_lock: every ts still in this list is un-claimed (a
     * dispatch de-lists itself under this lock before claiming), so either the
     * source is intact (remove + synchronous notify here) or its dispatch is
     * parked on free_lock and cannot auto-destroy the source under us. Removing
     * after unlocking instead would race a resuming dispatch's auto-REMOVE:
     * the notify frees ts and this loop would read freed memory / remove a
     * stale id (GLib CRITICAL). For a mid-dispatch source GLib defers the
     * notify until the dispatch returns -- never under a running callback. */
    g_source_remove (ts->source_id);
  }
  g_mutex_unlock (&modelmux_bin->free_lock);
  g_list_free (pending);
}

/* Deferred (main-loop) NULL+remove of detached per-stream queues. Done off the
 * caller's path so we NEVER set an element to NULL while holding modelmux_bin->lock: a
 * swap/clear probe may be parked on that queue's streaming thread waiting for
 * modelmux_bin->lock, and set_state(NULL) would then wait on that parked task forever. */
typedef struct {
  ModelMuxBin *modelmux_bin;
  GstElement *conv;            /* optional per-lane copy converter (upstream of in_q), or NULL */
  GstElement *in_q;
  GstElement *out_q;
  GstElement *req_el;          /* owner element of req_pad (ref held), or NULL */
  GstPad     *req_pad;         /* request pad to release+unref DEFERRED (ref held), or NULL.
                                * nvstreamdemux fans events out to ALL its src pads in one
                                * sink_event call with NO lock against release: freeing a
                                * src_<id> pad synchronously on the control thread races that
                                * fan-out (valgrind: invalid read in gst_nvstreamdemux_sink_event
                                * -> gst_pad_push_event on the freed pad; freed at
                                * modelmux_stream_teardown). Unlink stays synchronous (event push on
                                * an unlinked pad is harmless); only the release/unref -- the
                                * actual free -- moves here, after the in-flight dispatch. */
  guint       src_id;          /* g_idle source id, so _free() can cancel a pending one */
} ModelMuxFreeCtx;

/* GDestroyNotify of the deferred-free idle: the SINGLE owner of the ModelMuxFreeCtx.
 * GLib guarantees it runs exactly once and never while the dispatch callback is
 * executing, so a teardown cancelling the source can no longer double-free the
 * ctx under a mid-flight dispatch. Must not touch fc->modelmux_bin (teardown may have
 * freed it); de-listing is done by the dispatch / the canceller. */
static void
modelmux_free_ctx_notify (gpointer udata)
{
  g_free (udata);
}

static gboolean
modelmux_deferred_free_branch (gpointer udata)
{
  ModelMuxFreeCtx *fc = (ModelMuxFreeCtx *) udata;
  ModelMuxBin *modelmux_bin = fc->modelmux_bin;
  gboolean shutting_down;
  /* De-register first: we are running, so this fc is no longer "pending". Done under
   * free_lock (NOT modelmux_bin->lock -- the scheduler may hold modelmux_bin->lock) so it races cleanly
   * against modelmux_bin_free(), which cancels still-pending frees under the same
   * lock. If _free() got here first it removed our source, so we never run -> no UAF. */
  g_mutex_lock (&modelmux_bin->free_lock);
  modelmux_bin->pending_frees = g_list_remove (modelmux_bin->pending_frees, fc);
  shutting_down = g_atomic_int_get (&modelmux_bin->shutting_down);
  g_mutex_unlock (&modelmux_bin->free_lock);
  /* deferred request-pad release (see ModelMuxFreeCtx.req_pad): drop OUR refs even on
   * shutdown (they would leak; the pad/element objects themselves are disposed
   * with the bin), but skip the release call -- _free()'s own teardown owns the
   * element then. */
  if (fc->req_pad) {
    if (!shutting_down)
      gst_element_release_request_pad (fc->req_el, fc->req_pad);
    gst_object_unref (fc->req_pad);
    gst_object_unref (fc->req_el);
  }
  /* teardown in progress: leave the queues to be disposed with modelmux_bin->bin -- do not
   * race _free()'s own state teardown of the same elements. */
  if (shutting_down)
    return G_SOURCE_REMOVE;
  /* free the copy converter FIRST (it is upstream of in_q); already unlinked at both ends. */
  if (fc->conv) {
    gst_element_set_state (fc->conv, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (modelmux_bin->bin), fc->conv);
  }
  if (fc->in_q) {
    gst_element_set_state (fc->in_q, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (modelmux_bin->bin), fc->in_q);
  }
  if (fc->out_q) {
    gst_element_set_state (fc->out_q, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (modelmux_bin->bin), fc->out_q);
  }
  return G_SOURCE_REMOVE;   /* fc freed by modelmux_free_ctx_notify, never here */
}

/* Schedule a deferred branch teardown and register it so _free() can cancel it. Holding
 * free_lock across the g_idle_add + list insert closes the race where the main loop
 * dispatches (and frees) the fc before we record it: the callback blocks on free_lock
 * until the insert completes. Safe to call with or without modelmux_bin->lock held (free_lock is
 * a distinct lock, always taken AFTER modelmux_bin->lock where both are held). */
static void
modelmux_schedule_branch_free (ModelMuxBin * modelmux_bin, GstElement * conv, GstElement * in_q, GstElement * out_q)
{
  ModelMuxFreeCtx *fc = g_new0 (ModelMuxFreeCtx, 1);
  fc->modelmux_bin = modelmux_bin;
  fc->conv = conv;
  fc->in_q = in_q;
  fc->out_q = out_q;
  g_mutex_lock (&modelmux_bin->free_lock);
  fc->src_id = g_idle_add_full (G_PRIORITY_DEFAULT_IDLE,
      modelmux_deferred_free_branch, fc, modelmux_free_ctx_notify);
  modelmux_bin->pending_frees = g_list_prepend (modelmux_bin->pending_frees, fc);
  g_mutex_unlock (&modelmux_bin->free_lock);
}

/* Schedule the DEFERRED release of a stream's input-demux request pad (+ its tee,
 * which rides the generic NULL+remove slot). Takes ownership of `pad` (the caller's
 * ref) and `tee`; refs `el` itself. See ModelMuxFreeCtx.req_pad for why the release must
 * not run on the caller's thread: nvstreamdemux's event fan-out reads every src pad
 * with no lock against release -- freeing one synchronously is a UAF the moment an
 * upstream event is in flight (valgrind-confirmed on churn-remove). The caller has
 * already UNLINKED the pad, so no new data reaches the branch; the pad object just
 * stays alive until the main loop's next idle, past any in-flight dispatch. Safe to
 * call with modelmux_bin->lock held (free_lock is a distinct lock, always taken after it). */
static void
modelmux_schedule_pad_release (ModelMuxBin * modelmux_bin, GstElement * el, GstPad * pad,
    GstElement * tee)
{
  ModelMuxFreeCtx *fc = g_new0 (ModelMuxFreeCtx, 1);
  fc->modelmux_bin = modelmux_bin;
  fc->in_q = tee;
  if (pad) {
    fc->req_el = gst_object_ref (el);
    fc->req_pad = pad;
  }
  g_mutex_lock (&modelmux_bin->free_lock);
  /* G_PRIORITY_DEFAULT (not DEFAULT_IDLE): control events (stream/add) dispatch at
   * DEFAULT, so a lower-priority release could be overtaken by a re-add that then
   * finds the src_<id> pad name still requested. Same priority = FIFO with the
   * control plane: this release (queued during the remove) runs before any add
   * queued after it. */
  fc->src_id = g_idle_add_full (G_PRIORITY_DEFAULT,
      modelmux_deferred_free_branch, fc, modelmux_free_ctx_notify);
  modelmux_bin->pending_frees = g_list_prepend (modelmux_bin->pending_frees, fc);
  g_mutex_unlock (&modelmux_bin->free_lock);
}

/* Execute a PENDING pad release for `padname` inline, if one is queued. Called by
 * attach (main loop) right before it requests src_<id>: a re-add of a recycled
 * source id can be QUEUED before the remove that frees the id is even dispatched,
 * so the deferred release -- scheduled during that remove -- would run after the
 * add and the request would hit "element already has a pad named src_<id>"
 * (undefined behaviour). Releasing here is as safe as the deferred idle running:
 * attach and the remove dispatch are serialized on the same main loop, so the
 * remove that scheduled this release has fully completed and the dead source's
 * event fan-out is past -- the only unsafe instant was DURING that dispatch.
 * The ctx's tee rides along (same NULL+remove the idle would do). */
static void
modelmux_flush_pending_pad_release (ModelMuxBin * modelmux_bin, const gchar * padname)
{
  ModelMuxFreeCtx flushed = { 0 };
  gboolean found = FALSE;
  GList *l;

  g_mutex_lock (&modelmux_bin->free_lock);
  for (l = modelmux_bin->pending_frees; l; l = l->next) {
    ModelMuxFreeCtx *fc = (ModelMuxFreeCtx *) l->data;
    if (fc->req_pad && strcmp (GST_PAD_NAME (fc->req_pad), padname) == 0) {
      flushed = *fc;                       /* copy BEFORE g_source_remove: the
                                            * source's notify g_free()s fc */
      found = TRUE;
      modelmux_bin->pending_frees = g_list_remove (modelmux_bin->pending_frees, fc);
      g_source_remove (fc->src_id);
      break;
    }
  }
  g_mutex_unlock (&modelmux_bin->free_lock);
  if (!found)
    return;
  MM_INFO ("MultiModelBin: flushing pending release of demux pad %s inline "
      "(re-add of a recycled source id overtook the deferred release)", padname);
  gst_element_release_request_pad (flushed.req_el, flushed.req_pad);
  gst_object_unref (flushed.req_pad);
  gst_object_unref (flushed.req_el);
  if (flushed.in_q) {                      /* the retired stream's tee */
    gst_element_set_state (flushed.in_q, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (modelmux_bin->bin), flushed.in_q);
  }
}

/* ================================================================== *
 *  RACE-FREE DYNAMIC LANE WIRING  (industry-standard block -> relink -> seed -> unblock).
 *
 *  When a stream's branch is (re)wired to a freshly-requested per-model nvstreammux sink (or the
 *  combined display mux), the new sink MUST receive stream-start/caps/segment BEFORE its first
 *  buffer. The tee/queue normally REPLAYS those sticky events, but the src pad is created/relinked
 *  before the sink below it exists, so the replay can miss the late-linked mux -> the mux holds
 *  that stream's buffers waiting for a SEGMENT it never saw -> the stream wedges (frames buffered,
 *  inferred=0, 0 fps) while siblings on the same model run.
 *
 *  The deterministic, leak-free, deadlock-free recipe (used at every dynamic-wire site below):
 *    1. BLOCK the upstream src pad FIRST (before any link), so no buffer can traverse the lane.
 *    2. Fully wire the lane (queue -> model mux -> demux -> queue -> combined mux).
 *    3. SEED: re-send stream-start -> caps -> segment, IN THAT ORDER, into the lane's first sink
 *       (fetched from the authoritative upstream holder), while still blocked.
 *    4. UNBLOCK last -> any held buffer now flows strictly AFTER the events.
 *  Idempotent: a sticky event the peer already holds is de-duplicated by GstPad, so re-seeding
 *  never double-segments. Flush-free: it can never wipe events or crash. The block callback takes
 *  no lock, so it cannot deadlock against modelmux_bin->lock held by the wiring thread.
 * ================================================================== */

/* Stay-blocked callback for a downstream BLOCK probe: OK keeps the pad blocked until removed. */
static GstPadProbeReturn
modelmux_lane_block_hold (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  (void) pad;
  (void) info;
  (void) udata;
  return GST_PAD_PROBE_OK;
}

/* (1) BLOCK a src pad before its lane is linked. Returns the probe id (0 on failure). */
static gulong
modelmux_lane_block (GstPad * src)
{
  gulong id = src ? gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
      modelmux_lane_block_hold, NULL, NULL) : 0;
  if (src && !id)
    MM_WARN ("MultiModelBin: failed to block dynamic lane pad %s before wiring -- "
        "sticky-event seeding will continue, but first-buffer ordering is not guaranteed",
        GST_PAD_NAME (src));
  return id;
}

/* (4) UNBLOCK (no-op if never blocked). MUST be called on every wiring exit path so a failed
 * attach can never leave the branch blocked. */
static void
modelmux_lane_unblock (GstPad * src, gulong block_id)
{
  if (src && block_id)
    gst_pad_remove_probe (src, block_id);
}

/* (3) Re-send the three lane-critical sticky events (stream-start -> caps -> segment, IN ORDER)
 * from a pad that already holds them (@src_with_events) into a freshly-linked downstream sink
 * (@dst_sink). Caller must keep the lane BLOCKED upstream so these land ahead of any buffer.
 * Logs a MISSING SEGMENT (upstream not negotiated yet -> replay cannot fix this lane) and a
 * REFUSED SEGMENT (the exact condition that wedges inference). Idempotent + flush-free. */
static void
modelmux_lane_seed_events (GstPad * src_with_events, GstPad * dst_sink,
    guint stream_id, const gchar * model_name)
{
  static const GstEventType order[3] =
      { GST_EVENT_STREAM_START, GST_EVENT_CAPS, GST_EVENT_SEGMENT };
  guint i;
  if (!src_with_events || !dst_sink)
    return;
  for (i = 0; i < G_N_ELEMENTS (order); i++) {
    GstEvent *event= gst_pad_get_sticky_event (src_with_events, order[i], 0);
    if (!event) {
      if (order[i] == GST_EVENT_SEGMENT)
        MM_WARN ("MultiModelBin: stream %u model '%s' -- no SEGMENT on %s to seed the new "
            "lane; deterministic start not guaranteed (upstream not negotiated yet)",
            stream_id, model_name ? model_name : "-", GST_PAD_NAME (src_with_events));
      continue;
    }
    /* gst_pad_send_event() consumes the ref returned by gst_pad_get_sticky_event(). */
    if (!gst_pad_send_event (dst_sink, event) && order[i] == GST_EVENT_SEGMENT)
      MM_WARN ("MultiModelBin: stream %u model '%s' -- SEGMENT not accepted by %s; "
          "stream may stall", stream_id, model_name ? model_name : "-",
          GST_PAD_NAME (dst_sink));
  }
}

/* Convenience for the tee-fed lanes (attach/passthrough): seed the lane below @tee_src from the
 * tee SINK (authoritative holder of this stream's sticky events), then UNBLOCK. */
static void
modelmux_lane_seed_and_unblock (GstElement * tee, GstPad * tee_src, gulong block_id,
    guint stream_id, const gchar * model_name)
{
  GstPad *tee_sink = tee ? gst_element_get_static_pad (tee, "sink") : NULL;
  GstPad *lane_sink = tee_src ? gst_pad_get_peer (tee_src) : NULL;  /* conv/in_q sink, below block */
  modelmux_lane_seed_events (tee_sink, lane_sink, stream_id, model_name);
  if (tee_sink) gst_object_unref (tee_sink);
  if (lane_sink) gst_object_unref (lane_sink);
  modelmux_lane_unblock (tee_src, block_id);                 /* unblock LAST */
}

/* PASSTHROUGH attach (model-sharding=0 over-cap): route the stream straight to the combined
 * display mux WITHOUT inference -- tee -> queue -> out_mux[column]. The frames flow downstream
 * un-inferred (no detection meta), exactly like an over-batch stream in a stock nvinfer graph.
 * Consumes @tee_src (already requested on e->tee by the caller). The branch carries NO model:
 * ra->role / ra->model stay NULL, so modelmux_detach_role tears it down generically (it skips the
 * model-pool detach). Always uses the PRIMARY top-row display slot (column). Returns TRUE on
 * success; on failure releases tee_src and returns FALSE. Caller holds modelmux_bin->lock. */
static gboolean
modelmux_attach_passthrough (ModelMuxBin * modelmux_bin, ModelMuxStreamEntry * e, ModelMuxRoleAttach * ra,
    GstPad * tee_src, gulong block_id)
{
  GstElement *in_q;
  GstPad *q_sink, *q_src, *mux_sink;
  gchar nm[96], pname[64];
  gint slot = e->column;                 /* passthrough shows in the primary (top) row */

  g_snprintf (nm, sizeof (nm), "passthru-q-%u-%u", e->stream_id,
      e->branch_serial);
  in_q = create_gst_element ("queue", nm);
  if (!in_q) {
    modelmux_lane_unblock (tee_src, block_id);     /* never leave the tee branch blocked on failure */
    gst_element_release_request_pad (e->tee, tee_src);
    gst_object_unref (tee_src);
    return FALSE;
  }
  gst_bin_add (GST_BIN (modelmux_bin->bin), in_q);
  gst_element_sync_state_with_parent (in_q);

  g_snprintf (pname, sizeof (pname), "sink_%u", slot);
  mux_sink = gst_element_request_pad_simple (modelmux_bin->out_mux, pname);
  q_sink = gst_element_get_static_pad (in_q, "sink");
  q_src = gst_element_get_static_pad (in_q, "src");
  if (!mux_sink || !modelmux_link_pads (tee_src, q_sink) || !modelmux_link_pads (q_src, mux_sink)) {
    MM_ERR ("MultiModelBin: passthrough wiring failed for stream %u", e->stream_id);
    if (q_sink) gst_object_unref (q_sink);
    if (q_src)  gst_object_unref (q_src);
    if (mux_sink) {
      gst_element_release_request_pad (modelmux_bin->out_mux, mux_sink);
      gst_object_unref (mux_sink);   /* release caller's ref from request_pad */
    }
    gst_element_set_state (in_q, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (modelmux_bin->bin), in_q);
    modelmux_lane_unblock (tee_src, block_id);     /* never leave the tee branch blocked on failure */
    gst_element_release_request_pad (e->tee, tee_src);
    gst_object_unref (tee_src);
    return FALSE;
  }
  /* DEBUG tap: count frames flowing through the bypass (proves un-inferred flow). */
  modelmux_add_tap_tracked (modelmux_bin, q_src, e->stream_id, "pass-out",
      &ra->tap_pad, &ra->tap_probe_id);
  {
    ModelMuxPassthruProvCtx *pc = g_new0 (ModelMuxPassthruProvCtx, 1);
    pc->modelmux_bin = modelmux_bin;
    pc->sid = e->stream_id;
    gst_pad_add_probe (q_src, GST_PAD_PROBE_TYPE_BUFFER,
        modelmux_passthru_provenance_probe, pc, g_free);
  }
  /* perf (attach-perf-metric): tap THROUGHPUT on the bypass so a passthrough
   * stream still reports a live frames/sec (it runs no nvinfer, so it has no
   * model-bin perf counter). Cleared first => each passthrough session starts a
   * fresh window; the ctx is g_free'd by the probe's destroy-notify on teardown. */
  if (modelmux_bin->config && modelmux_bin->config->attach_perf_metric && e->stream_id < MM_ACCT_MAX) {
    ModelMuxThruProbeCtx *pc = g_new0 (ModelMuxThruProbeCtx, 1);
    pc->modelmux_bin = modelmux_bin;
    pc->sid = e->stream_id;
    modelmux_perf_counter_clear (&modelmux_bin->thru_perf[e->stream_id]);
    gst_pad_add_probe (q_src, GST_PAD_PROBE_TYPE_BUFFER, modelmux_perf_passthru_probe,
        pc, g_free);
  }
  gst_object_unref (q_sink);
  gst_object_unref (q_src);

  ra->active = TRUE;
  ra->passthru = TRUE;
  ra->role = NULL;                       /* no model pool */
  ra->model = NULL;                      /* model field EMPTY -- not bound to any model */
  ra->conv = NULL;                       /* passthrough lane never copies (no converter) */
  ra->in_q = in_q;
  ra->out_q = NULL;
  ra->tee_src = tee_src;
  ra->out_mux_sink = mux_sink;
  ra->out_idx = slot;
  /* lane fully wired -> seed stream-start/caps/segment into the passthru queue (-> combined
   * display mux) while still blocked, then UNBLOCK, so the mux never sees "data flow before
   * segment" on a just-attached stream. (model_name NULL -> passthrough, logged as '-'.) */
  modelmux_lane_seed_and_unblock (e->tee, tee_src, block_id, e->stream_id, NULL);
  MM_INFO ("    - MultiModelBin  stream %u PASSTHROUGH (no inference) -> display slot %u "
      "(model-sharding=0, per-model cap full)", e->stream_id, slot);
  return TRUE;
}

/* Attach one role for a stream. The PRIMARY is the mandatory output path and falls back to
 * the no-infer PASSTHROUGH lane whenever there is no usable model -- this is the SINGLE place
 * that decides infer-vs-passthrough, covering ALL three triggers uniformly:
 *   (1) no model resolved        (model_name NULL/empty: nothing requested + no default, or a
 *                                 requested model that does not exist + no default),
 *   (2) model at its sharding cap (model-sharding=0 over-cap, signalled by cap_overflow), and
 *   (3) -- both of the above route through the one wiring helper modelmux_attach_passthrough().
 * A SHADOW never passes through (it has no standalone display row): no model => no shadow.
 *
 * park_out (normally NULL) requests the PROMOTE make-before-break variant: the retiring
 * passthrough lane still OWNS this stream's display slot (sink_<slot> is name-derived from
 * the column), so instead of requesting/linking the combined-mux sink the lane tail (out_q
 * src) is left unlinked and PARKED behind a held BLOCK probe -- on the lane's OWN queue
 * thread only, never a shared demux/tee thread -- and handed back via *park_out /
 * *park_block_id (pad ref transferred). The caller links it to the slot once the passthru
 * residue has drained and freed it. In this mode the passthrough fallbacks are DISABLED
 * (they would collide on the very slot being handed over): the caller keeps the original
 * passthru lane on failure. */
static gboolean
modelmux_attach_role (ModelMuxBin * modelmux_bin, ModelMuxStreamEntry *e, ModelMuxRoleAttach *ra,
    ModelPool * rb, const gchar * model_name, gint gpu, gboolean via_default,
    GstPad ** park_out, gulong * park_block_id)
{
  GstPad *tee_src, *role_out, *mux_sink;
  GstElement *in_q = NULL, *out_q = NULL, *conv = NULL;
  ModelBin *attached_mb = NULL;
  gboolean cap_overflow = FALSE;
  gboolean is_primary = (rb == modelmux_bin->primary_pool);
  gchar pname[64];
  gint slot;
  gulong block_id;

  /* (1) No usable model. PRIMARY -> passthrough (frames flow, no inference); SHADOW -> no
   * branch at all (callers gate empty-shadow, so this is a defensive refusal). Parked
   * (promote) mode always carries a model -- refuse rather than collide on the slot. */
  if (!model_name || !*model_name) {
    if (is_primary && !park_out) {
      tee_src = gst_element_request_pad_simple (e->tee, "src_%u");
      if (!tee_src)
        return FALSE;
      /* BLOCK before the passthru lane is linked; modelmux_attach_passthrough seeds + unblocks. */
      return modelmux_attach_passthrough (modelmux_bin, e, ra, tee_src, modelmux_lane_block (tee_src));
    }
    return FALSE;
  }

  /* A shadow needs the unified A/B display (its slot is the bottom row). With
   * unified-batch=0 there is no shadow row -- refuse rather than collide with the
   * primary's slot. Covers the default shadow AND a live model/update shadow-enable. */
  if (rb == modelmux_bin->shadow_pool && !modelmux_bin->unified) {
    MM_WARN ("MultiModelBin: refusing Shadow attach for stream %u -- unified-batch=0 "
        "(no display row for a shadow); running primary only", e->stream_id);
    return FALSE;
  }

  tee_src = gst_element_request_pad_simple (e->tee, "src_%u");
  if (!tee_src)
    return FALSE;
  /* (1) BLOCK FIRST -- before the lane below tee_src is linked -- so no buffer can reach the
   * not-yet-seeded per-model nvstreammux. Removed only after the lane is wired + seeded, or on
   * any failure path below (so a failed attach never leaves the tee branch blocked). */
  block_id = modelmux_lane_block (tee_src);

  role_out = model_pool_attach (rb, e->stream_id, model_name, gpu, tee_src,
      e->branch_serial, &in_q, &out_q, &conv, &cap_overflow, &attached_mb);
  if (!role_out) {
    /* (2) model-sharding=0 over-cap on the PRIMARY role -> passthrough (consumes tee_src +
     * the block, which it seeds/unblocks). Shadow over-cap just fails, as does a parked
     * (promote) attach -- the stream already HAS a passthru lane on this slot. */
    if (cap_overflow && is_primary && !park_out)
      return modelmux_attach_passthrough (modelmux_bin, e, ra, tee_src, block_id);
    modelmux_lane_unblock (tee_src, block_id);
    gst_element_release_request_pad (e->tee, tee_src);
    gst_object_unref (tee_src);
    return FALSE;
  }

  /* DERIVE the combined-mux slot from this stream's DISPLAY column + role, so the
   * tiler lays primary on the top row and shadow directly below it (slot == tile).
   * The slot is stable across model reroute/swap (the column never moves), so a tile
   * keeps its position while its CONTENT (model/version) changes dynamically. */
  {
    gboolean is_shadow = (rb == modelmux_bin->shadow_pool);
    slot = (is_shadow && modelmux_bin->unified)
        ? (gint) modelmux_bin->max_streams + e->column   /* shadow -> bottom row */
        : e->column;                           /* primary -> top row   */
  }
  if (park_out) {
    /* PROMOTE make-before-break: the display slot is still owned by the retiring
     * passthru lane. PARK the lane tail behind a held BLOCK probe on the lane's
     * own out_q src (stalls only that queue's thread; inferred output simply
     * accumulates in out_q for the bounded drain window) and hand the pad to the
     * caller for the deferred slot link. The park is always resolved: the finish
     * links + unblocks, and every failure/abandon end tears the lane down (the
     * branch-free's pad deactivation releases the block) -- so even a non-leaky
     * out_q (lane-leaky=0) can never fill up and backpressure the shared
     * tee/demux. Sticky events are stored on the pad regardless of the block, so
     * the late link replays them ahead of the first buffer as usual. */
    *park_block_id = modelmux_lane_block (role_out);
    *park_out = role_out;              /* pad ref transferred to the caller */
    mux_sink = NULL;                   /* linked by the caller once the slot frees */
  } else {
    g_snprintf (pname, sizeof (pname), "sink_%u", slot);
    mux_sink = gst_element_request_pad_simple (modelmux_bin->out_mux, pname);
    if (!mux_sink || !modelmux_link_pads (role_out, mux_sink)) {
      MM_ERR ("MultiModelBin: failed to link %s role of stream %u to display mux",
          rb->role, e->stream_id);
      gst_object_unref (role_out);                 /* drop the pad ref we own */
      if (mux_sink) {
        gst_element_release_request_pad (modelmux_bin->out_mux, mux_sink);
        gst_object_unref (mux_sink);   /* release does not consume the request ref */
      }
      /* tear down the per-stream queues + copy converter that model_pool_attach created
       * (detach only releases the ModelBin pads, not these) -- else they leak into the container. */
      if (out_q) { gst_element_set_state (out_q, GST_STATE_NULL);
                   gst_bin_remove (GST_BIN (rb->container), out_q); }
      if (in_q)  { gst_element_set_state (in_q,  GST_STATE_NULL);
                   gst_bin_remove (GST_BIN (rb->container), in_q); }
      if (conv)  { gst_element_set_state (conv, GST_STATE_NULL);
                   gst_bin_remove (GST_BIN (rb->container), conv); }
      model_pool_detach (rb, e->stream_id, model_name);
      modelmux_lane_unblock (tee_src, block_id);
      gst_element_release_request_pad (e->tee, tee_src);
      gst_object_unref (tee_src);
      return FALSE;
    }
  }
  model_bin_remove_zero_stream_drop (attached_mb);

  /* DEBUG tap: count buffers leaving this model (per-stream demux src) before
   * they enter the combined display mux. Brackets the ModelBin's output. */
  modelmux_add_tap_tracked (modelmux_bin, role_out, e->stream_id,
      g_strcmp0 (rb->role, "Shadow") == 0 ? "shad-out" : "prim-out",
      &ra->tap_pad, &ra->tap_probe_id);
  /* debug-only: prove this lane (tee -> [conv] -> queue) deep-COPIES or shares the buffer, by
   * comparing the buffer at the lane head (tee src) vs the queue input. Track the probes on the
   * role so a fast add/remove can remove them before the dynamic pads/elements are released. */
  modelmux_lane_install_verify (tee_src, in_q, rb->role, e->stream_id,
      &ra->verify_head_pad, &ra->verify_head_probe_id,
      &ra->verify_queue_pad, &ra->verify_queue_probe_id,
      &ra->verify_ctx);

  if (!park_out)
    gst_object_unref (role_out); /* linked + tapped -> drop the pad ref we own (pad lives in
                                  * out_q). In parked mode the ref was handed to the caller. */

  ra->active = TRUE;
  ra->model = g_strdup (model_name);
  ra->req_gpu = gpu;           /* remembered so internal re-attaches keep the device */
  ra->via_default = via_default; /* origin: default fallback vs explicit pin */
  ra->role = rb;
  ra->conv = conv;             /* copy converter (NULL if zero-copy lane); freed at teardown */
  ra->in_q = in_q;             /* track queues for teardown */
  ra->out_q = out_q;
  ra->tee_src = tee_src;       /* keep ref for teardown */
  ra->out_mux_sink = mux_sink;
  ra->out_idx = slot;

  /* (3)+(4) lane fully wired -> seed stream-start/caps/segment into the per-model nvstreammux
   * (still blocked), then UNBLOCK last. See "RACE-FREE DYNAMIC LANE WIRING" above. */
  modelmux_lane_seed_and_unblock (e->tee, tee_src, block_id, e->stream_id, model_name);
  return TRUE;
}

static void
modelmux_attach_reservation_clear (ModelMuxBin * modelmux_bin, guint stream_id,
    gint column, gboolean recycle_column)
{
  gpointer key = GINT_TO_POINTER ((gint) stream_id);

  g_mutex_lock (&modelmux_bin->lock);
  if (modelmux_bin->streams_pending)
    g_hash_table_remove (modelmux_bin->streams_pending, key);
  if (modelmux_bin->streams_cancelled)
    g_hash_table_remove (modelmux_bin->streams_cancelled, key);
  if (recycle_column && column >= 0)
    g_queue_push_tail (modelmux_bin->col_free_idx, GINT_TO_POINTER (column));
  g_mutex_unlock (&modelmux_bin->lock);
}

static void
modelmux_stream_unbuilt_free (ModelMuxBin * modelmux_bin, ModelMuxStreamEntry * e)
{
  if (!e)
    return;
  modelmux_pad_probe_ref_clear (&e->branch_tap_pad, &e->branch_tap_probe_id);
  if (e->demux_src) {
    GstPad *peer = gst_pad_get_peer (e->demux_src);
    if (peer) {
      gst_pad_unlink (e->demux_src, peer);
      gst_object_unref (peer);
    }
  }
  /* release DEFERRED, same as modelmux_stream_teardown: even a just-requested pad is
   * already part of the demux's event fan-out set (see ModelMuxFreeCtx.req_pad). */
  if (e->demux_src || e->tee)
    modelmux_schedule_pad_release (modelmux_bin, modelmux_bin->in_demux, e->demux_src, e->tee);
  g_free (e);
}

gboolean
modelmux_bin_attach_stream (ModelMuxBin * modelmux_bin, guint stream_id,
    const gchar * name, const gchar * cam_id, const gchar * primary,
    const gchar * shadow, gint p_gpu, gint s_gpu, gboolean p_via_default,
    gboolean s_via_default)
{
  ModelMuxStreamEntry *e;
  GstPad *tee_sink;
  gboolean was_empty;
  gint column;
  gchar nm[64];
  gpointer key = GINT_TO_POINTER ((gint) stream_id);

  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return FALSE;
  /* The input nvstreamdemux exposes exactly max_streams src pads, and stream_id is used
   * directly as the demux pad index AND as the flat frame-accounting index. A source_id
   * at/above max_streams (or the accounting cap) has no demux src_%u pad -- reject early
   * with a clear message instead of letting the request_pad below fail opaquely or, worse,
   * indexing acct[] out of bounds. */
  if (stream_id >= modelmux_bin->max_streams || stream_id >= MM_ACCT_MAX) {
    MM_ERR ("MultiModelBin: stream %u out of range (max_streams=%u, acct cap=%u) -- "
        "no input demux src_%u pad; reject", stream_id, modelmux_bin->max_streams,
        (guint) MM_ACCT_MAX, stream_id);
    return FALSE;
  }
  /* NOTE: a NULL/empty primary is NOT an error -- it means "no resolvable model", which the
   * primary attach below handles by routing the stream through the no-infer PASSTHROUGH lane
   * (frames flow to the display mux without inference). So we do not reject here. */
  /* reserve this stream's stable DISPLAY column up-front (capacity = max_streams).
   * Done before building any pad so a capacity reject needs no cleanup. Recycled
   * on teardown; the column fixes the stream's tiler position (primary top row =
   * slot column, shadow bottom row = slot max_streams+column). */
  g_mutex_lock (&modelmux_bin->lock);
  if (g_hash_table_lookup (modelmux_bin->stream_wiring, key) ||
      g_hash_table_lookup (modelmux_bin->streams_pending, key)) {
    g_mutex_unlock (&modelmux_bin->lock);
    MM_WARN ("MultiModelBin: stream %u already attached/attaching", stream_id);
    return FALSE;
  }
  /* attaching the FIRST stream after the bin drained to zero? if so we will
   * run the refill recovery path (see below). Safe = no active streams. */
  was_empty = (g_hash_table_size (modelmux_bin->stream_wiring) == 0);
  if (g_queue_is_empty (modelmux_bin->col_free_idx)) {
    g_mutex_unlock (&modelmux_bin->lock);
    MM_ERR ("MultiModelBin: capacity reached (%u streams) -- reject stream %u",
        modelmux_bin->max_streams, stream_id);
    return FALSE;
  }
  column = GPOINTER_TO_INT (g_queue_pop_head (modelmux_bin->col_free_idx));
  g_hash_table_insert (modelmux_bin->streams_pending, key, GINT_TO_POINTER (1));
  g_hash_table_remove (modelmux_bin->streams_cancelled, key);
  g_mutex_unlock (&modelmux_bin->lock);

  e = g_new0 (ModelMuxStreamEntry, 1);
  e->stream_id = stream_id;
  e->column = column;
  e->branch_serial = modelmux_branch_serial_next ();
  e->prim.req_gpu = MM_GPU_ANY;   /* g_new0 would read as gpu 0 -- explicit "any" */
  e->shad.req_gpu = MM_GPU_ANY;   /* (via_default: g_new0's FALSE is the right init) */

  /* request this source's branch off the input demux (src_<source_id>) */
  g_snprintf (nm, sizeof (nm), "src_%u", stream_id);
  /* a recycled source id may still have its OLD pad awaiting deferred release
   * (see modelmux_flush_pending_pad_release) -- flush it now or the request below
   * hits a duplicate pad name */
  modelmux_flush_pending_pad_release (modelmux_bin, nm);
  e->demux_src = gst_element_request_pad_simple (modelmux_bin->in_demux, nm);
  if (!e->demux_src) {
    MM_ERR ("MultiModelBin: failed to request input demux %s", nm);
    modelmux_attach_reservation_clear (modelmux_bin, stream_id, e->column, TRUE);
    g_free (e);
    return FALSE;
  }

  /* the per-stream tee fans out to the Primary/Shadow roles. Pacing of the
   * batch is handled upstream by nvmultiurisrcbin's internal nvstreammux. */
  g_snprintf (nm, sizeof (nm), "tee-%u-%u", stream_id, e->branch_serial);
  e->tee = create_gst_element ("tee", nm);
  if (!e->tee) {
    gst_element_release_request_pad (modelmux_bin->in_demux, e->demux_src);
    gst_object_unref (e->demux_src);
    modelmux_attach_reservation_clear (modelmux_bin, stream_id, e->column, TRUE);
    g_free (e);
    return FALSE;
  }

  gst_bin_add (GST_BIN (modelmux_bin->bin), e->tee);
  gst_element_sync_state_with_parent (e->tee);

  /* input demux src_<id> -> tee (both internal to the bin) */
  tee_sink = gst_element_get_static_pad (e->tee, "sink");
  if (gst_pad_link (e->demux_src, tee_sink) != GST_PAD_LINK_OK) {
    MM_ERR ("MultiModelBin: failed to link input demux src_%u -> tee", stream_id);
    gst_object_unref (tee_sink);
    gst_element_set_state (e->tee, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (modelmux_bin->bin), e->tee);
    gst_element_release_request_pad (modelmux_bin->in_demux, e->demux_src);
    gst_object_unref (e->demux_src);
    modelmux_attach_reservation_clear (modelmux_bin, stream_id, e->column, TRUE);
    g_free (e);
    return FALSE;
  }
  gst_object_unref (tee_sink);

  /* DEBUG tap: count buffers this source delivers into its branch (input demux
   * src_<id> -> tee). Tells us if nvmultiurisrcbin is feeding THIS stream. */
  modelmux_add_tap_tracked (modelmux_bin, e->demux_src, stream_id, "branch-in",
      &e->branch_tap_pad, &e->branch_tap_probe_id);

  /* Serialize stream-table + per-model pad-index pools (free_idx / col_free_idx)
   * against the streaming-thread swap/clear probes. The lock spans the
   * attach so a concurrent swap to the SAME model can't pop the same pad index.
   * Released before any teardown (which re-locks). No element is set to a running
   * state that would wait on a lock-parked thread (queues are fresh). */
  g_mutex_lock (&modelmux_bin->lock);
  if (g_atomic_int_get (&modelmux_bin->shutting_down) ||
      g_hash_table_lookup (modelmux_bin->streams_cancelled, key)) {
    g_hash_table_remove (modelmux_bin->streams_pending, key);
    g_hash_table_remove (modelmux_bin->streams_cancelled, key);
    g_queue_push_tail (modelmux_bin->col_free_idx, GINT_TO_POINTER (e->column));
    g_mutex_unlock (&modelmux_bin->lock);
    MM_INFO ("MultiModelBin: stream %u attach cancelled before publication", stream_id);
    modelmux_stream_unbuilt_free (modelmux_bin, e);
    return FALSE;
  }
  g_hash_table_remove (modelmux_bin->streams_pending, key);
  g_hash_table_insert (modelmux_bin->stream_wiring, key, e);
  g_atomic_int_set (&modelmux_bin->active_streams, (gint) g_hash_table_size (modelmux_bin->stream_wiring));
  modelmux_acct_stream_reset (modelmux_bin, stream_id);   /* fresh frame-accounting for this life */
  /* record the source name (overlay/provenance/status) only now that the entry is
   * in the table -- so the early-failure returns above never leave a stale name, and
   * a primary-attach failure below is cleaned by modelmux_stream_teardown (removes both). */
  g_hash_table_insert (modelmux_bin->stream_names, GINT_TO_POINTER ((gint) stream_id),
      g_strdup (name ? name : "src"));
  /* The ROUTABLE id, recorded alongside the display name and under the same
   * lock/lifetime rules. No "src" fallback here: an absent camera_id must stay
   * absent, because a synthesised one would not resolve in POST stream/route and
   * the read would emit an identifier its own write rejects -- precisely the
   * mislabel this map exists to avoid. */
  if (cam_id && *cam_id)
    g_hash_table_insert (modelmux_bin->stream_cam_ids,
        GINT_TO_POINTER ((gint) stream_id), g_strdup (cam_id));
  modelmux_stream_name_publish (modelmux_bin, stream_id, name ? name : "src");

  /* Primary: the mandatory output path. modelmux_attach_role routes it to its model, or -- when
   * there is no usable model (NULL/empty name) or the model is at its sharding cap -- to the
   * no-infer PASSTHROUGH lane (one modular decision, inside modelmux_attach_role). A successful
   * return means the stream reaches output either way. */
  if (!modelmux_attach_role (modelmux_bin, e, &e->prim, modelmux_bin->primary_pool, primary, p_gpu,
          p_via_default, NULL, NULL)) {
    g_mutex_unlock (&modelmux_bin->lock);
    MM_ERR ("MultiModelBin: primary attach failed for stream %u", stream_id);
    modelmux_bin_detach_stream (modelmux_bin, stream_id);
    return FALSE;
  }
  /* Frame-accounting tracks INFERENCE integrity (in==infer==out per role). A passthrough
   * branch carries no inference, so its frames are intentionally NOT accounted --
   * registering it would make the sanity check misread infer==0 as loss. */
  if (!e->prim.passthru)
    modelmux_acct_role_attach (modelmux_bin, stream_id, MM_ROLE_PRIMARY);
  /* Shadow (optional) -- skipped entirely for a passthrough primary (no model, no A/B). */
  if (!e->prim.passthru && shadow && *shadow) {
    if (!modelmux_attach_role (modelmux_bin, e, &e->shad, modelmux_bin->shadow_pool, shadow, s_gpu,
            s_via_default, NULL, NULL))
      MM_WARN ("MultiModelBin: shadow attach failed for stream %u (continuing)",
          stream_id);
    else
      modelmux_acct_role_attach (modelmux_bin, stream_id, MM_ROLE_SHADOW);
  }
  g_mutex_unlock (&modelmux_bin->lock);

  /* REFILL (first stream after a drain-to-zero): do NOT flush the demux_src here. The
   * per-stream tee REPLAYS its sticky events (stream-start/caps/segment) onto the freshly-
   * requested src pad, so the rewired branch negotiates on its own, and persistent-element
   * stale state is already cleared on the DRAIN side -- each model bin cycles its mux+demux
   * PLAYING->READY->PLAYING at 0 streams, and stream teardown recreates the combined out_mux.
   *
   * A live FLUSH_START + FLUSH_STOP(reset) down this new branch instead RACES the demux/tee
   * sticky-event delivery on the streaming thread: if the flush wins, those events are wiped
   * (the demux does not re-send them), the next buffer reaches the model-bin nvstreammux with
   * no caps, and it crashes in gst_video_info_to_caps (finfo NULL) -- observed intermittently
   * (~6/10 refill cycles) as "data flow before segment/stream-start" cascading down to the mux.
   * This mirrors the PROMOTION path, which documents the identical hazard and likewise never
   * flushes. */
  if (was_empty && modelmux_bin->ever_attached && e->demux_src) {
    MM_INFO ("  - MultiModelBin  refill (first stream after a drain-to-zero)");
    /* capture element states a few seconds after a refill, so if the chain
     * stalls we see exactly which element is stuck in which state */
    if (modelmux_bin->log_enabled)
      modelmux_schedule_dump (modelmux_bin, 3);
  }
  modelmux_bin->ever_attached = TRUE;    /* subsequent empty->first-stream transitions are refills */

  /* log the canonical model id (name@version); primary/shadow are already the keys. */
  MM_INFO ("  - MultiModelBin  stream %u WIRED  (primary=%s  shadow=%s)",
      stream_id, e->prim.passthru ? "(passthrough/no-infer)" : (primary ? primary : "-"),
      (!e->prim.passthru && shadow) ? shadow : "-");
  return TRUE;
}

/* Tear down one role branch. The caller MUST have blocked the stream's input
 * (so the branch is quiescent): out_mux slot -> ModelBin -> tee pad -> queues. */
static void
modelmux_role_clear_debug_probes (ModelMuxRoleAttach * ra)
{
  if (!ra)
    return;
  modelmux_pad_probe_ref_clear (&ra->tap_pad, &ra->tap_probe_id);
  modelmux_pad_probe_ref_clear (&ra->verify_head_pad, &ra->verify_head_probe_id);
  modelmux_pad_probe_ref_clear (&ra->verify_queue_pad, &ra->verify_queue_probe_id);
  g_free (ra->verify_ctx);
  ra->verify_ctx = NULL;
}

static void
modelmux_detach_role (ModelMuxBin * modelmux_bin, ModelMuxStreamEntry * e, ModelMuxRoleAttach * ra)
{
  if (!ra->active)
    return;
  modelmux_role_clear_debug_probes (ra);
  /* 1. unlink + release this role's combined-display-mux sink slot */
  if (ra->out_mux_sink) {
    GstPad *peer = gst_pad_get_peer (ra->out_mux_sink);
    if (peer) {
      gst_pad_unlink (peer, ra->out_mux_sink);
      gst_object_unref (peer);
    }
    gst_element_release_request_pad (modelmux_bin->out_mux, ra->out_mux_sink);
    gst_object_unref (ra->out_mux_sink);     /* release caller's ref from request_pad */
    /* the out_mux slot is DERIVED from the stream's column (not pooled) -- the column
     * is recycled once at stream teardown, not per role. */
    ra->out_mux_sink = NULL;
  }
  /* 2. detach from the shared ModelBin (releases its mux sink + demux src;
   *    GCs the ModelBin if this was its last stream) */
  if (ra->role && ra->model)
    model_pool_detach (ra->role, e->stream_id, ra->model);
  /* 3. release this role's tee branch */
  if (ra->tee_src) {
    gst_element_release_request_pad (e->tee, ra->tee_src);
    gst_object_unref (ra->tee_src);
    ra->tee_src = NULL;
  }
  /* 4. drop the per-stream queues -- DEFERRED to the main loop. We may be holding
   *    modelmux_bin->lock here (teardown runs under it); setting a queue to NULL while a
   *    swap/clear probe is parked on its streaming thread waiting for modelmux_bin->lock
   *    would deadlock. Hand them off; they are already unlinked at both ends. */
  if (ra->in_q || ra->out_q || ra->conv) {
    modelmux_schedule_branch_free (modelmux_bin, ra->conv, ra->in_q, ra->out_q);
    ra->conv = NULL;
    ra->in_q = NULL;
    ra->out_q = NULL;
  }
  g_free (ra->model);
  ra->model = NULL;
  g_free (ra->pending_model);
  ra->pending_model = NULL;
  ra->active = FALSE;
  ra->passthru = FALSE;        /* clean slate: a re-attach of this slot (model or passthru)
                                * must not inherit a stale passthru flag */
  ra->req_gpu = MM_GPU_ANY;    /* placement constraint dies with the branch */
  ra->via_default = FALSE;     /* origin dies with it */
  ra->role = NULL;
}

/* ================================================================== *
 *  PASSTHROUGH -> MODEL promotion (deferred attach).
 *
 *  A stream whose primary model was still WARMING was attached as a no-infer passthrough so
 *  its frames keep flowing. Once that model reaches WARMED, the control-plane poller calls
 *  modelmux_bin_promote_passthru() to swap the passthrough branch for a real model
 *  branch on that stream. This is SELF-CONTAINED -- it reuses modelmux_attach_role and touches
 *  NONE of the CLEAR/SWAP/cutover machinery, so existing paths are unaffected.
 *
 *  MAKE-BEFORE-BREAK, RELINK-FIRST (no shared-thread hold): the input nvstreamdemux pushes
 *  ALL streams' frames from ONE thread, so holding a BLOCK on the stream's demux src pad
 *  for a drain window would stall EVERY stream for that window (waves overlap under churn).
 *  Instead the promote never blocks a shared pad:
 *
 *    RELINK  (main loop, one modelmux_bin->lock hold) build + wire the model lane on a NEW tee src
 *            pad via modelmux_attach_role -- with its TAIL parked: the retiring passthru lane
 *            still owns this stream's display slot (sink_<slot> is column-derived), so the
 *            model lane's out_q src stays unlinked behind a held BLOCK probe on the lane's
 *            OWN queue thread (inferred output accumulates in the leaky out_q; nothing
 *            shared stalls). Then CUT the passthru's tee src pad. This is the moment the
 *            stream switches: new frames flow only into the model lane. Zero loss by
 *            construction -- every frame either reached the passthru queue (residual,
 *            drains below) or the model lane; a frame landing in the microscopic overlap
 *            between the model lane going live and the cut may reach BOTH (a one-frame
 *            make-before-break dup, which the frame accounting already tolerates).
 *    DRAIN   the passthru lane gets NO new input after the cut, so its residue flushes to
 *            the display mux through its still-linked slot on its own queue thread. A
 *            main-loop poll watches the queue's 'current-level-buffers' down to 0, capped
 *            by the MM_PROMOTE_DRAIN_MS hard deadline (the poll ALWAYS terminates).
 *    RETIRE  once empty + quiescent (IDLE probe on the drained queue's src -- the same
 *            proven context the CLEAR probe uses), release the passthru's display slot and
 *            hand the empty queue to the deferred branch-free.
 *    LINK    (main loop) request the freed sink_<slot> for the parked model lane, link it,
 *            and UNBLOCK LAST -- the parked (already inferred) output flows out strictly
 *            after the passthru residue, so per-slot ordering is preserved.
 *
 *  The promote ctx is bound to the stream INSTANCE via branch_serial (captured at schedule
 *  time): a remove + re-add reusing the same source_id during the drain window is detected
 *  at every re-resolution and the promote is ABANDONED instead of touching the fresh
 *  stream's branch. All ownership ends funnel through modelmux_promote_ctx_free, so no lane is
 *  ever left blocked and no retired piece leaks.
 * ================================================================== */

/* Hard deadline (ms) for the passthru residual drain. The lane runs no inference;
 * its depth is the queue occupancy plus ~one display-mux batch period, so it
 * empties well within this cap. A residue still queued at the deadline means the
 * display mux is wedged -- the retire proceeds anyway (bounded, never stalls). */
#define MM_PROMOTE_DRAIN_MS 150
/* Cadence (ms) of the queue-emptiness poll during the drain window. */
#define MM_PROMOTE_POLL_MS  10
typedef struct
{
  ModelMuxBin *modelmux_bin;
  guint    stream_id;
  guint    branch_serial;      /* stream INSTANCE guard: the per-lifetime serial of the
                                * entry this promote was scheduled for. A remove + re-add
                                * reusing the same source_id mid-drain gets a NEW serial,
                                * so every re-resolution can tell "same id, different
                                * stream" and abandon instead of touching the fresh lane */
  gchar   *model;              /* promoted-to primary model (for the completion logs) */
  gint     consumed;           /* ownership handed to the next stage: the IDLE probe's
                                * GDestroyNotify must NOT free it (single owner)     */
  gint64   deadline;           /* monotonic us; drain hard deadline (entry + DRAIN_MS) */
  /* RETIRED passthru lane pieces (moved out of e->prim at the relink; owned here).
   * Its tee src pad was already released at the relink -- these are the tail. */
  GstElement *r_in_q;          /* passthru queue, draining via the still-linked slot */
  GstPad     *r_mux_sink;      /* its combined-mux sink slot (request-pad ref)       */
  /* PARKED model lane tail: out_q src, blocked + unlinked until the slot frees. */
  GstPad  *park_out;           /* model lane out_q src (pad ref owned by the ctx) */
  gulong   park_block_id;      /* held block on park_out (0 = block failed)       */
  gint     out_idx;            /* the display slot both lanes hand over            */
} ModelMuxPromoteCtx;

/* Release whatever is left of the RETIRED passthru lane: unlink + give back its
 * display-mux slot (via the pad's own parent -- the combined mux may have been
 * recreated by a drain-to-zero if the stream vanished mid-promote, in which case
 * the pad is already parentless and only our ref is dropped) and hand the queue
 * to the deferred branch-free. Pad/element-scoped only. CONCURRENCY: while the
 * bound stream entry is LIVE this must run under modelmux_bin->lock (teardown and the
 * drain-poll/retire chain both reach the r_* fields; the poll re-reads + refs
 * r_in_q under the same lock hold that validates the entry). The ctx-free
 * funnel below calls it WITHOUT the lock, which is safe because every
 * ownership end that frees the ctx with the entry still live first drops the
 * entry's back-pointer under modelmux_bin->lock (modelmux_promote_ctx_unbind / the idle
 * probe), so no teardown can race it by then. */
static void
modelmux_promote_release_retired (ModelMuxPromoteCtx * pc)
{
  if (pc->r_mux_sink) {
    GstElement *mux = gst_pad_get_parent_element (pc->r_mux_sink);
    GstPad *peer = gst_pad_get_peer (pc->r_mux_sink);
    if (peer) {
      gst_pad_unlink (peer, pc->r_mux_sink);
      gst_object_unref (peer);
    }
    if (mux) {
      gst_element_release_request_pad (mux, pc->r_mux_sink);
      gst_object_unref (mux);
    }
    gst_object_unref (pc->r_mux_sink);
    pc->r_mux_sink = NULL;
  }
  if (pc->r_in_q) {
    /* on shutdown the queue is disposed with modelmux_bin->bin (mirrors the deferred
     * branch-free's own teardown guard) */
    if (!g_atomic_int_get (&pc->modelmux_bin->shutting_down))
      modelmux_schedule_branch_free (pc->modelmux_bin, NULL, pc->r_in_q, NULL);
    pc->r_in_q = NULL;
  }
}

static void
modelmux_promote_ctx_free (gpointer data)
{
  ModelMuxPromoteCtx *pc = (ModelMuxPromoteCtx *) data;
  if (!pc)
    return;
  /* EVERY ownership end (slot handover done, stream replaced, shutdown, armed-
   * but-never-fired) funnels here, so the retired passthru pieces can never
   * leak. Must be callable WITHOUT modelmux_bin->lock (tracked-source destroys run on
   * cancel paths that may hold it) -- only pad/element-scoped operations.
   * A still-armed park block is deliberately NOT removed: every funnel end
   * that reaches here with the model tail still parked (unlinked) is a dying/
   * abandoned lane -- unblocking would push into an unlinked pad (NOT_LINKED
   * into the shared upstream). Every such lane has (or will have) its
   * teardown scheduled -- stream teardown's modelmux_detach_role or the finish's
   * slot-failure detach -- so the deferred branch-free ALWAYS deactivates the
   * pad and wakes the parked thread, regardless of lane leakiness (a
   * non-leaky out_q must never be left to fill and backpressure the shared
   * tee/demux). The success path links + unblocks explicitly BEFORE freeing
   * the ctx, so no serving stream is ever left blocked. */
  if (pc->park_out)
    gst_object_unref (pc->park_out);
  modelmux_promote_release_retired (pc);
  g_free (pc->model);
  g_free (pc);
}

/* Drop the stream entry's weak promote-ctx back-pointer (set at schedule time
 * so modelmux_stream_teardown can release the ctx-owned retired display slot
 * synchronously). MUST be called before any ctx free that can happen while the
 * bound stream instance is still live, or the entry would hold a dangling
 * pointer. Takes modelmux_bin->lock -- callers must NOT hold it. */
static void
modelmux_promote_ctx_unbind (ModelMuxPromoteCtx * pc)
{
  ModelMuxBin *modelmux_bin = pc->modelmux_bin;
  ModelMuxStreamEntry *e;

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) pc->stream_id));
  if (e && e->promote_ctx == pc)
    e->promote_ctx = NULL;
  g_mutex_unlock (&modelmux_bin->lock);
}

/* GDestroyNotify of the promote RETIRE IDLE probe: frees the ctx UNLESS the callback
 * handed it to the main-loop finish. Covers the armed-but-never-fired case (branch torn
 * down / shutdown before the pad went idle), which would otherwise leak the ctx. Runs
 * after the callback returns (GStreamer hook semantics), so reading `consumed` is ordered. */
static void
modelmux_promote_ctx_notify (gpointer data)
{
  ModelMuxPromoteCtx *pc = (ModelMuxPromoteCtx *) data;
  if (pc && !pc->consumed)
    modelmux_promote_ctx_free (pc);
}

/* MAIN loop (retired lane released): hand the freed display slot to the parked model
 * lane -- request sink_<slot>, link the parked out_q src, UNBLOCK LAST. Only now is the
 * promote observably complete, so the PROMOTED/WIRED markers fire here: the end-state is
 * identical whether the model was warm at attach (direct WIRED) or promoted later. */
static gboolean
modelmux_promote_finish_main (gpointer data)
{
  ModelMuxPromoteCtx *pc = (ModelMuxPromoteCtx *) data;
  ModelMuxBin *modelmux_bin = pc->modelmux_bin;
  ModelMuxStreamEntry *e;
  GstPad *mux_sink = NULL;
  gchar pname[64];
  gboolean linked = FALSE;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_promote_ctx_free (pc);
    return G_SOURCE_REMOVE;
  }

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) pc->stream_id));
  if (e && e->branch_serial == pc->branch_serial && e->prim.active &&
      !e->prim.passthru && pc->park_out) {
    g_snprintf (pname, sizeof (pname), "sink_%u", (guint) pc->out_idx);
    mux_sink = gst_element_request_pad_simple (modelmux_bin->out_mux, pname);
    if (mux_sink && modelmux_link_pads (pc->park_out, mux_sink)) {
      e->prim.out_mux_sink = mux_sink;         /* request-pad ref transferred */
      linked = TRUE;
      MM_INFO ("  - MultiModelBin  stream %u PROMOTED passthrough -> model '%s' (now warmed; "
          "inference starts)", pc->stream_id, pc->model);
      /* emit the SAME wired marker as a direct attach so the end-state is observably
       * identical whether the model was warm at attach or promoted later (tooling/tests
       * watch for this). */
      MM_INFO ("  - MultiModelBin  stream %u WIRED  (primary=%s  shadow=%s)",
          pc->stream_id, pc->model,
          (e->shad.active && e->shad.model) ? e->shad.model : "-");
    } else {
      if (mux_sink) {
        gst_element_release_request_pad (modelmux_bin->out_mux, mux_sink);
        gst_object_unref (mux_sink);
      }
      /* NEVER leave the lane parked: a permanently parked model lane wastes
       * its shard even when the out_q is leaky, and with lane-leaky=0 the
       * non-leaky out_q would fill and backpressure the shared tee/demux --
       * stalling EVERY stream. Tear the parked tail down properly instead:
       * modelmux_detach_role unlinks the lane from the tee and hands its queues to
       * the deferred branch-free, whose pad deactivation removes the held
       * block and wakes the parked thread (unblocking by hand would push
       * NOT_LINKED up through the tee into the shared input demux). The
       * ctx's park_out pad ref is dropped by modelmux_promote_ctx_free below. */
      pc->park_block_id = 0;                   /* dies with the pad's deactivation */
      modelmux_detach_role (modelmux_bin, e, &e->prim);
      /* FALLBACK: never leave the stream with NO output lane. Re-attach the
       * no-infer PASSTHROUGH branch (the same wiring an initial no-model
       * attach uses: modelmux_attach_role with no model -> modelmux_attach_passthrough)
       * so frames keep flowing downstream while the operator retries; the
       * detach above reset e->prim to a clean slate, and the passthrough
       * re-requests sink_<column> fresh. Only if THAT fails too is the
       * stream left detached (the pre-existing behaviour). */
      if (modelmux_attach_role (modelmux_bin, e, &e->prim, modelmux_bin->primary_pool, NULL, MM_GPU_ANY,
              FALSE, NULL, NULL)) {
        MM_WARN ("MultiModelBin: stream %u promotion failed -> stream back on "
            "PASSTHROUGH (no inference; model '%s' could not take display "
            "slot %u)", pc->stream_id, pc->model, (guint) pc->out_idx);
      } else {
        MM_ERR ("MultiModelBin: stream %u promote to '%s' could not take display slot "
            "%u -- stream has NO primary output (parked lane torn down)",
            pc->stream_id, pc->model, (guint) pc->out_idx);
      }
    }
    e->prim.busy = FALSE;                      /* slot handover resolved either way */
  } else if (e && e->branch_serial == pc->branch_serial) {
    e->prim.busy = FALSE;                      /* defensive: lane changed under us */
  } else {
    MM_INFO ("  - MultiModelBin  stream %u was replaced during the drain -- "
        "promote abandoned", pc->stream_id);
  }
  g_mutex_unlock (&modelmux_bin->lock);

  if (linked) {
    /* UNBLOCK LAST, outside the lock (the woken queue thread pushes into the
     * combined mux immediately): the parked -- already inferred -- output now
     * flows strictly AFTER the drained passthru residue, preserving per-slot
     * ordering. Sticky stream-start/caps/segment replay ahead of the first
     * buffer, exactly like any late link. */
    modelmux_lane_unblock (pc->park_out, pc->park_block_id);
    pc->park_block_id = 0;
    gst_object_unref (pc->park_out);
    pc->park_out = NULL;
  }
  modelmux_promote_ctx_free (pc);
  return G_SOURCE_REMOVE;
}

/* STREAMING thread (RETIRE IDLE probe on the drained passthru queue's src -- empty AND
 * quiescent, the same proven context the CLEAR probe uses): release the retired lane's
 * display slot + hand its empty queue to the deferred branch-free, then hop the slot
 * handover to the main loop. The queue got no input since the relink cut its tee pad
 * and level==0 was observed, so nothing can arrive between the check and this teardown
 * -- the destroyed branch holds no frames. */
static GstPadProbeReturn
modelmux_promote_idle_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxPromoteCtx *pc = (ModelMuxPromoteCtx *) udata;
  ModelMuxBin *modelmux_bin = pc->modelmux_bin;
  (void) pad;
  (void) info;

  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return GST_PAD_PROBE_REMOVE;               /* probe notify frees pc */

  /* under modelmux_bin->lock: the freed sink_<slot> name must not race the finish's
   * re-request or a concurrent attach/teardown mutating out_mux slots. */
  g_mutex_lock (&modelmux_bin->lock);
  modelmux_promote_release_retired (pc);
  /* the retired pieces are gone -- the entry's teardown back-pointer has
   * served its purpose; drop it NOW so any later ctx free (idle-add failure,
   * finish, notify) can never leave the live entry dangling. */
  {
    ModelMuxStreamEntry *e = g_hash_table_lookup (modelmux_bin->stream_wiring,
        GINT_TO_POINTER ((gint) pc->stream_id));
    if (e && e->promote_ctx == pc)
      e->promote_ctx = NULL;
  }
  g_mutex_unlock (&modelmux_bin->lock);

  if (modelmux_tracked_idle_add (modelmux_bin, modelmux_promote_finish_main, pc,
          modelmux_promote_ctx_free, NULL))
    pc->consumed = 1;                          /* main-loop finish owns pc now */
  return GST_PAD_PROBE_REMOVE;                 /* one-shot; notify frees if !consumed */
}

/* MAIN loop drain poll: watch the retired passthru queue empty out (it gets no new
 * input since the relink cut its tee pad; nothing shared is blocked meanwhile), capped
 * by the MM_PROMOTE_DRAIN_MS hard deadline -- the poll ALWAYS terminates. Once empty
 * (or at the deadline: display mux wedged, retire anyway) arm the RETIRE IDLE probe on
 * the drained queue's src; it fires immediately on the quiescent pad. */
static gboolean
modelmux_promote_drain_poll (gpointer data)
{
  ModelMuxPromoteCtx *pc = (ModelMuxPromoteCtx *) data;
  ModelMuxBin *modelmux_bin = pc->modelmux_bin;
  ModelMuxStreamEntry *e;
  GstElement *r_in_q = NULL;
  GstPad *inq_src;
  guint level = 0;
  gboolean live;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_promote_ctx_free (pc);
    return G_SOURCE_REMOVE;
  }
  /* stream-INSTANCE guard (branch_serial): a remove + re-add reusing this
   * source_id mid-drain is a DIFFERENT stream -- abandon instead of applying
   * the stale promote to its fresh branch.
   *
   * The ctx-owned retired queue can be released FROM UNDER US between ticks:
   * modelmux_stream_teardown (which can run on the APP thread via the synchronous
   * detach-stream action signal) calls modelmux_promote_release_retired under
   * modelmux_bin->lock, NULLing pc->r_in_q and handing the queue to the deferred
   * branch-free. So the pointer must be re-read AND ref'd inside the SAME
   * lock hold that validated the stream instance; the ref keeps the element
   * safe to query after the unlock even if a teardown races in. */
  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) pc->stream_id));
  live = (e && e->branch_serial == pc->branch_serial);
  if (live && pc->r_in_q)
    r_in_q = gst_object_ref (pc->r_in_q);
  g_mutex_unlock (&modelmux_bin->lock);
  if (!live) {
    MM_INFO ("  - MultiModelBin  stream %u was replaced during the drain -- "
        "promote abandoned", pc->stream_id);
    modelmux_promote_ctx_free (pc);                  /* releases the retired + parked pieces */
    return G_SOURCE_REMOVE;
  }
  if (!r_in_q) {
    /* a teardown already released the retired lane mid-drain (stream detach
     * in flight) -- nothing left to drain or retire; drop the ctx. */
    modelmux_promote_ctx_unbind (pc);     /* the entry is live: drop its back-ptr first */
    modelmux_promote_ctx_free (pc);
    return G_SOURCE_REMOVE;
  }
  g_object_get (r_in_q, "current-level-buffers", &level, NULL);
  if (level > 0 && g_get_monotonic_time () < pc->deadline) {
    gst_object_unref (r_in_q);
    /* tracked sources are one-shot (the dispatcher claims them) -> re-arm for
     * the next tick; the dispatched source's notify skips the destroy. */
    if (!modelmux_tracked_timeout_add (modelmux_bin, MM_PROMOTE_POLL_MS, modelmux_promote_drain_poll,
            pc, modelmux_promote_ctx_free, NULL)) {
      modelmux_promote_ctx_unbind (pc);   /* the entry is live: drop its back-ptr first */
      modelmux_promote_ctx_free (pc);
    }
    return G_SOURCE_REMOVE;
  }
  if (level > 0)
    MM_WARN ("MultiModelBin: stream %u passthru residue (%u frame(s)) still "
        "queued at the %dms drain deadline -- retiring anyway (display mux not "
        "consuming?)", pc->stream_id, level, MM_PROMOTE_DRAIN_MS);
  inq_src = gst_element_get_static_pad (r_in_q, "src");
  gst_object_unref (r_in_q);
  if (!inq_src) {
    modelmux_promote_ctx_unbind (pc);     /* the entry is live: drop its back-ptr first */
    modelmux_promote_ctx_free (pc);
    return G_SOURCE_REMOVE;
  }
  /* retire probe on the drained lane; the notify frees pc (retired + parked
   * pieces) if the probe never fires. pc must NOT be touched after this call: a
   * pad that is already idle dispatches the probe synchronously, which may hand
   * pc to the main loop (or free it) before gst_pad_add_probe() returns. */
  gst_pad_add_probe (inq_src, GST_PAD_PROBE_TYPE_IDLE, modelmux_promote_idle_probe, pc,
      modelmux_promote_ctx_notify);
  gst_object_unref (inq_src);
  return G_SOURCE_REMOVE;
}

/* Promote a stream's passthrough primary to a now-warmed model. Called from the
 * control-plane poller once the model reaches WARMED. No-op if the stream vanished or
 * isn't passthrough. RELINK-FIRST: the stream switches to the model lane synchronously
 * HERE (make-before-break on a fresh tee pad; see the section header) -- only the
 * display-slot handover trails behind the bounded residual drain. No shared pad is
 * ever held blocked, so the other streams never stall. */
void
modelmux_bin_promote_passthru (ModelMuxBin * modelmux_bin, guint stream_id,
    const gchar * model, const gchar * shadow, gint p_gpu, gint s_gpu,
    gboolean p_via_default, gboolean s_via_default)
{
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach retired;
  ModelMuxPromoteCtx *pc;
  GstPad *park_out = NULL;
  gulong park_block_id = 0;

  if (!modelmux_bin || !model || !*model)
    return;
  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
  if (!e || !e->prim.passthru || !e->prim.in_q) {
    g_mutex_unlock (&modelmux_bin->lock);
    return;                                    /* stream gone / not passthrough anymore */
  }
  /* SNAPSHOT the passthru lane records: on success modelmux_attach_role overwrites
   * e->prim with the model lane and the snapshot becomes the RETIRED lane; on
   * failure e->prim is untouched (attach mutates it only on success) and the
   * stream simply stays on its intact passthrough -- nothing to undo. */
  retired = e->prim;                           /* struct copy */
  /* placement: the LANE's recorded constraint is the freshest intent --
   * set_pending_model rewrites it (INCLUDING back to ANY) on every re-route
   * while the promote sat queued, so whenever the lane was tagged with a
   * pending model its req_gpu supersedes the enqueue-time snapshot; the
   * snapshot only fills in for a lane that was never tagged. */
  if (retired.pending_model || retired.req_gpu != MM_GPU_ANY) {
    p_gpu = retired.req_gpu;
    /* the lane tag is the freshest intent for the ORIGIN too: set_pending_model
     * rewrites gpu+via_default together on every re-route while the promote
     * sat queued (same supersede rule as req_gpu above). */
    if (retired.pending_model)
      p_via_default = retired.via_default;
  }
  if (!modelmux_attach_role (modelmux_bin, e, &e->prim, modelmux_bin->primary_pool, model, p_gpu,
          p_via_default, &park_out, &park_block_id)) {
    g_mutex_unlock (&modelmux_bin->lock);
    MM_ERR ("MultiModelBin: stream %u promote to '%s' failed -- staying on "
        "passthrough (nothing was torn down)", stream_id, model);
    return;
  }
  /* the stream SWITCHED: e->prim is the model lane now (tail parked behind the
   * still-owned display slot). Scrub the passthru-only state it inherited. */
  e->prim.passthru = FALSE;
  g_free (e->prim.pending_model);              /* same string the snapshot holds */
  e->prim.pending_model = NULL;
  retired.pending_model = NULL;                /* freed above -- never touch again */
  e->prim.busy = TRUE;                         /* slot handover in flight: no swap/clear
                                                * may stack on this role until it lands */
  modelmux_acct_role_attach (modelmux_bin, stream_id, MM_ROLE_PRIMARY);
  /* CUT the passthru lane's input: drop its debug probes and release its tee
   * branch. New frames now flow only into the model lane; the residue already
   * queued keeps draining to the display mux through the still-linked slot. (A
   * frame in the microscopic overlap since the attach unblocked the new tee pad
   * may have entered BOTH lanes -- a one-frame make-before-break dup, which the
   * accounting tolerates; nothing is ever lost.) */
  modelmux_role_clear_debug_probes (&retired);
  if (retired.tee_src) {
    gst_element_release_request_pad (e->tee, retired.tee_src);
    gst_object_unref (retired.tee_src);
    retired.tee_src = NULL;
  }
  /* optional shadow alongside (its bottom-row slot is free -- no handover needed).
   * NOTE: do NOT flush the demux_src anywhere here. The attach helpers block each
   * freshly-requested tee lane, explicitly seed stream-start/caps/segment into the
   * new branch, then unblock. A FLUSH_STOP(reset) would instead wipe sticky events
   * that the demux does not re-send, reaching the model nvstreammux with no
   * caps/segment. */
  /* !e->shad.active: the scoped-reroute path that queued this promote ALSO ran
   * update_routing, whose ENABLE may have attached the shadow already -- a second
   * attach would double-book the stream on the model and collide on the shadow's
   * display slot (tearing down the LIVE lane in its failure path). */
  if (shadow && *shadow && modelmux_bin->unified && !e->shad.active &&
      modelmux_attach_role (modelmux_bin, e, &e->shad, modelmux_bin->shadow_pool, shadow, s_gpu,
          s_via_default, NULL, NULL))
    modelmux_acct_role_attach (modelmux_bin, stream_id, MM_ROLE_SHADOW);

  pc = g_new0 (ModelMuxPromoteCtx, 1);
  pc->modelmux_bin = modelmux_bin;
  pc->stream_id = stream_id;
  pc->branch_serial = e->branch_serial;        /* bind to THIS stream instance */
  pc->model = g_strdup (model);
  pc->deadline = g_get_monotonic_time () +
      (gint64) MM_PROMOTE_DRAIN_MS * G_TIME_SPAN_MILLISECOND;
  pc->r_in_q = retired.in_q;                   /* retired passthru tail (ctx-owned) */
  pc->r_mux_sink = retired.out_mux_sink;
  pc->park_out = park_out;                     /* parked model lane tail (ctx-owned) */
  pc->park_block_id = park_block_id;
  pc->out_idx = e->prim.out_idx;
  /* weak back-pointer for a mid-drain stream remove: teardown releases the
   * ctx-owned retired display slot synchronously through it (see
   * ModelMuxStreamEntry.promote_ctx); dropped by every ctx ownership end before the
   * ctx can be freed with the entry still live. */
  e->promote_ctx = pc;
  g_mutex_unlock (&modelmux_bin->lock);

  if (!modelmux_tracked_timeout_add (modelmux_bin, MM_PROMOTE_POLL_MS, modelmux_promote_drain_poll, pc,
          modelmux_promote_ctx_free, NULL)) {
    modelmux_promote_ctx_unbind (pc);
    modelmux_promote_ctx_free (pc);                  /* teardown in flight: retire + drop */
    return;
  }
  MM_INFO ("  - MultiModelBin  stream %u promotion scheduled: passthrough -> '%s' "
      "(stream already switched; residual drain + display-slot handover pending)",
      stream_id, model);
}

/* Tag a passthrough stream's primary with the model it is awaiting (deferred promotion), so
 * the routing table shows "<model> (passthru)" rather than "-". No-op unless the stream's
 * primary is currently a passthrough branch. `gpu` is the placement the eventual promote
 * must honour (MM_GPU_ANY = any) -- recorded on the lane so the promote inherits it. */
void
modelmux_bin_set_pending_model (ModelMuxBin * modelmux_bin, guint stream_id,
    const gchar * model, gint gpu, gboolean via_default)
{
  ModelMuxStreamEntry *stream_entry;
  if (!modelmux_bin || !model || !*model)
    return;
  g_mutex_lock (&modelmux_bin->lock);
  stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
  if (stream_entry && stream_entry->prim.passthru) {
    g_free (stream_entry->prim.pending_model);
    stream_entry->prim.pending_model = g_strdup (model);
    stream_entry->prim.req_gpu = gpu;
    stream_entry->prim.via_default = via_default;  /* origin rides with the pending intent */
  }
  g_mutex_unlock (&modelmux_bin->lock);
}

/* Clear a passthrough stream's pending-model tag (the deferred promote it was
 * waiting for has been ABANDONED -- target failed to warm / was unloaded). Without
 * this the tag lies forever: status shows "<model> (passthru)", streams_on_model
 * keeps matching the stream to the dead key (a default switch off that key would
 * sweep a stream that was never on it), and update_routing logs a misleading
 * "applies via deferred promotion". Resets the lane to a plain passthrough. */
void
modelmux_bin_clear_pending_model (ModelMuxBin * modelmux_bin, guint stream_id)
{
  ModelMuxStreamEntry *e;
  if (!modelmux_bin)
    return;
  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
  if (e && e->prim.passthru && e->prim.pending_model) {
    MM_INFO ("  - MultiModelBin  stream %u pending model '%s' ABANDONED -> plain "
        "passthrough (no intended model)", stream_id, e->prim.pending_model);
    g_free (e->prim.pending_model);
    e->prim.pending_model = NULL;
    e->prim.req_gpu = MM_GPU_ANY;      /* the intent (and its placement) is gone */
    e->prim.via_default = FALSE;
  }
  g_mutex_unlock (&modelmux_bin->lock);
}

/* DRAIN-TO-ZERO RECOVERY for the combined display mux.
 *
 * When the last stream leaves, every sink pad of the combined display nvstreammux
 * (out_mux) is released. The legacy nvstreammux keeps stale per-batch streaming
 * state in that condition; on the next attach it accepts exactly ONE buffer and
 * then stops pulling -- the backpressure propagates all the way up (out_mux full
 * -> ModelBin demux/nvinfer/mux full -> tee blocks -> input demux blocks ->
 * nvmultiurisrcbin's source mux delivers just one batch). Verified by the
 * per-stream taps on a refill: branch-in=1, prim-out=1, out-mux-src=1, then
 * nothing -- a single buffer flows end-to-end and then the chain wedges.
 *
 * We originally cycled out_mux PLAYING->READY->PLAYING to clear that state, but
 * that reset is NOT reliable for the legacy mux (the refill still wedges after
 * one batch). Recreating the element from scratch is: a brand-new nvstreammux has
 * zero stale streaming state and resumes continuous batching the moment the next
 * stream requests a sink pad. out_mux carries no engine (unlike a ModelBin), so a
 * rebuild is cheap, and it has no sink pads at zero streams, so there is nothing
 * in flight to drop.
 *
 * Self-contained (no muxer source change): out_mux is OUR element, its SRC is
 * ghosted out via modelmux_bin->src_ghost (-> downstream tiler/osd/sink) and we simply
 * retarget that ghost to the fresh element's src. The input side (modelmux_bin->in_demux /
 * modelmux_bin->bin) is never touched, so nothing flushes back into nvmultiurisrcbin. */
static gboolean
modelmux_outmux_recreate (ModelMuxBin * modelmux_bin)
{
  GstElement *fresh;
  GstPad *src;
  guint out_slots = modelmux_bin->unified ? modelmux_bin->max_streams * 2 : modelmux_bin->max_streams;

  fresh = create_gst_element ("nvstreammux", "combined-display-mux");
  if (!fresh) {
    MM_ERR ("MultiModelBin: drain-recovery failed to create a fresh combined "
        "display mux -- keeping the old one");
    return FALSE;
  }
  /* same defensive property set as the original (set_prop_if_exists is a no-op
   * on the new nvstreammux, which does not expose these as GObject props) */
  set_prop_if_exists (fresh, "batch-size", out_slots);   /* auto-derived; NOT user-settable */
  set_prop_if_exists (fresh, "width",
      modelmux_bin->config ? modelmux_bin->config->mux_width : MM_MUX_WIDTH);
  set_prop_if_exists (fresh, "height",
      modelmux_bin->config ? modelmux_bin->config->mux_height : MM_MUX_HEIGHT);
  set_prop_if_exists (fresh, "batched-push-timeout",
      modelmux_bin->config ? modelmux_bin->config->combined_mux_push_timeout : MM_MUX_PUSH_TIMEOUT);
  set_prop_if_exists (fresh, "live-source",
      modelmux_bin->config ? modelmux_bin->config->mux_live_source : 1);
  set_prop_if_exists (fresh, "gpu-id", modelmux_bin->config ? (gint) modelmux_bin->config->gpu : 0);

  /* tear the old element down to NULL and drop it (its probes die with it) */
  gst_element_set_state (modelmux_bin->out_mux, GST_STATE_NULL);
  gst_bin_remove (GST_BIN (modelmux_bin->bin), modelmux_bin->out_mux);      /* unref's the old mux */
  modelmux_bin->out_mux = fresh;
  gst_bin_add (GST_BIN (modelmux_bin->bin), modelmux_bin->out_mux);

  /* reinstall the SRC-pad probes the original carried (see modelmux_bin_new) */
  src = gst_element_get_static_pad (modelmux_bin->out_mux, "src");
  gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
      modelmux_drop_eos_probe, NULL, NULL);
  gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER,
      modelmux_restore_srcid_probe, modelmux_bin, NULL);
  if (modelmux_bin->log_enabled) {
    gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER, modelmux_state_log_probe,
        modelmux_bin, NULL);
    gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        modelmux_outmux_eos_probe, modelmux_bin, NULL);
  }
  if (modelmux_overlay_enabled ())
    gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER, modelmux_combined_overlay_probe,
        modelmux_bin, NULL);

  /* repoint the bin's src ghost at the fresh element (keeps the downstream peer) */
  gst_ghost_pad_set_target (GST_GHOST_PAD (modelmux_bin->src_ghost), src);
  gst_object_unref (src);

  gst_element_sync_state_with_parent (modelmux_bin->out_mux);
  if (gst_element_get_state (modelmux_bin->out_mux, NULL, NULL, GST_SECOND)
      == GST_STATE_CHANGE_FAILURE)
    MM_ERR ("MultiModelBin: drain-recovery combined mux failed to reach state "
        "-- downstream output may stall");
  return TRUE;
}

/* Dismantle a stream's whole branch. nvmultiurisrcbin has already stopped/flushed
 * the removed source before posting the remove message; see the direct-teardown
 * rationale in modelmux_bin_detach_stream. */
static void
modelmux_stream_teardown (ModelMuxBin * modelmux_bin, ModelMuxStreamEntry * e)
{
  guint sid = e->stream_id;
  gboolean now_empty = FALSE;

  /* hold the lock so a concurrent MM_DEBUG dump never reads a freed entry /
   * ModelBin (the dump iterates streams + pools under the same lock) */
  g_mutex_lock (&modelmux_bin->lock);

  /* FIRST cut this stream off at the input demux: unlink src_<id> so the demux
   * stops routing this source into the branch. Done before tearing down the
   * roles, so no buffer can enter half-dismantled pads (no block needed, and
   * the demux keeps running for the other streams). The release/unref of the
   * request pad is DEFERRED (below, with the tee): nvstreamdemux fans events to
   * ALL src pads in one sink_event call with no lock against release, so
   * freeing the pad here races an in-flight fan-out -- valgrind-confirmed UAF
   * (invalid read in gst_nvstreamdemux_sink_event on the freed pad). Unlinked
   * is enough to stop the branch; freed must wait for the dispatch to pass. */
  modelmux_pad_probe_ref_clear (&e->branch_tap_pad, &e->branch_tap_probe_id);
  if (e->demux_src) {
    GstPad *peer = gst_pad_get_peer (e->demux_src);   /* tee sink */
    if (peer) {
      gst_pad_unlink (e->demux_src, peer);
      gst_object_unref (peer);
    }
  }

  /* A promote may be MID-DRAIN for this stream: release the ctx-owned retired
   * display slot NOW (modelmux_promote_release_retired NULLs the fields, so the
   * ctx's own funnel ends release exactly once) instead of waiting for the
   * next drain-poll tick -- a racing stream/add reusing the freed column would
   * otherwise find 'sink_<col>' still requested, get a NULL request pad and
   * permanently fail. The ctx itself stays owned by its deferred chain (the
   * next poll tick sees the entry gone and frees it); on shutdown the mux is
   * disposed with the bin and the tracked-source destroys may already have
   * freed the ctx, so the back-pointer is left alone (release_retired's own
   * shutdown guard covers the queue). */
  if (e->promote_ctx && !g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_promote_release_retired ((ModelMuxPromoteCtx *) e->promote_ctx);
    e->promote_ctx = NULL;
  }

  modelmux_detach_role (modelmux_bin, e, &e->shad);
  modelmux_detach_role (modelmux_bin, e, &e->prim);

  /* deferred release of the (already-unlinked) demux request pad + the tee: an
   * in-flight demux event fan-out (or a push already past the tee's sink) may
   * still be executing on a streaming thread -- see the unlink comment above. */
  if (e->demux_src || e->tee) {
    modelmux_schedule_pad_release (modelmux_bin, modelmux_bin->in_demux, e->demux_src, e->tee);
    e->demux_src = NULL;
    e->tee = NULL;
  }
  g_hash_table_remove (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) sid));
  g_atomic_int_set (&modelmux_bin->active_streams, (gint) g_hash_table_size (modelmux_bin->stream_wiring));
  g_hash_table_remove (modelmux_bin->stream_names, GINT_TO_POINTER ((gint) sid));
  /* in lockstep with stream_names -- a camera_id outliving its stream would let
   * the routing read name a source_id that has since been recycled */
  g_hash_table_remove (modelmux_bin->stream_cam_ids, GINT_TO_POINTER ((gint) sid));
  modelmux_stream_name_publish (modelmux_bin, sid, NULL);
  /* recycle this stream's display column (once, here -- not per role) */
  g_queue_push_tail (modelmux_bin->col_free_idx, GINT_TO_POINTER (e->column));
  now_empty = (g_hash_table_size (modelmux_bin->stream_wiring) == 0);

  /* DRAIN-TO-ZERO RECOVERY (combined display mux): rebuild it fresh so the next
   * stream flows continuously. See modelmux_outmux_recreate -- a PLAYING->READY->PLAYING
   * cycle is NOT enough for the legacy nvstreammux (refill wedges after one batch);
   * recreating the element clears the stale drained state for good.
   * MUST run under modelmux_bin->lock: every other reader/writer of modelmux_bin->out_mux holds the
   * lock, and a concurrent attach racing this teardown has to request its sink pad
   * on the FRESH mux -- wiring into the old drained one both wedges that stream and
   * dangles its branch when the old element is destroyed. Cheap here: zero streams
   * are attached, so nothing is flowing through the combined mux.
   * SKIPPED on shutdown (modelmux_bin_free sets shutting_down before it
   * detaches the remaining streams): no next stream is coming, and rebuilding
   * into a graph that is being torn down just races teardown. */
  if (now_empty && modelmux_bin->out_mux && !g_atomic_int_get (&modelmux_bin->shutting_down)) {
    MM_INFO ("  - MultiModelBin  drain-to-zero: rebuilding combined display mux "
        "so the next stream flows continuously");
    modelmux_outmux_recreate (modelmux_bin);
    g_atomic_int_set (&modelmux_bin->frame_count, 0); /* so the out-mux-src tap restarts at 1 */
  }
  g_mutex_unlock (&modelmux_bin->lock);
  MM_INFO ("  - MultiModelBin  stream %u TORN DOWN%s", sid,
      now_empty ? "  (last -- bin now empty)" : "");
  g_free (e);

  /* A removed stream may have left a model's shards under-full (e.g. 2 shards each
   * holding 1 stream when 1 shard would do). Schedule a debounced compaction pass to
   * pack them back into fewer shards (zero-drop) and free the redundant shard. */
  modelmux_schedule_compact (modelmux_bin);
}

gboolean
modelmux_bin_detach_stream (ModelMuxBin * modelmux_bin, guint stream_id)
{
  gpointer key = GINT_TO_POINTER ((gint) stream_id);
  ModelMuxStreamEntry *e;

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, key);
  if (!e) {
    if (g_hash_table_lookup (modelmux_bin->streams_pending, key)) {
      g_hash_table_insert (modelmux_bin->streams_cancelled, key, GINT_TO_POINTER (1));
      g_mutex_unlock (&modelmux_bin->lock);
      return TRUE;
    }
    g_mutex_unlock (&modelmux_bin->lock);
    return FALSE;
  }
  if (g_hash_table_lookup (modelmux_bin->streams_pending, key)) {
    g_mutex_unlock (&modelmux_bin->lock);
    return TRUE;
  }
  g_hash_table_insert (modelmux_bin->streams_pending, key, GINT_TO_POINTER (1));
  g_mutex_unlock (&modelmux_bin->lock);

  /* Direct teardown -- NO pad-block on the input-demux src.
   *
   * nvmultiurisrcbin stops + flushes the removed source BEFORE it posts the
   * stream-remove message we are handling, so this stream's input-demux src
   * pad is already idle: blocking it is unnecessary. Worse, an IDLE/BLOCK probe
   * on ONE nvstreamdemux src pad stalls the WHOLE demux (it pushes to every src
   * pad in a single chain call), which back-pressures nvmultiurisrcbin's mux and
   * corrupts its source teardown -- breaking the NEXT add (its decoder spins on
   * "data flow before segment"). The original not-linked crash was the file-EOS
   * issue, now fixed via file-loop, not a live-relink race. */
  modelmux_stream_teardown (modelmux_bin, e);
  g_mutex_lock (&modelmux_bin->lock);
  g_hash_table_remove (modelmux_bin->streams_pending, key);
  g_hash_table_remove (modelmux_bin->streams_cancelled, key);
  g_mutex_unlock (&modelmux_bin->lock);
  return TRUE;
}

/* ================================================================== *
 *  Zero-drop / zero-copy live model swap.
 *
 *  Switching a running stream's model MUST NOT drop or copy a frame, and MUST
 *  NOT disturb other streams. Two invariants make that hold:
 *
 *   (1) Block on the role's in_q SRC pad -- NEVER on the shared nvstreamdemux
 *       src. nvstreamdemux pushes to ALL its src pads in one chain call, so
 *       blocking one stalls the whole demux and back-pressures nvmultiurisrcbin's
 *       live mux -> drops on every stream. The per-role `queue` decouples: while
 *       its src is blocked its sink keeps accepting from the tee, so it ABSORBS
 *       the few in-flight buffers (default ~1 s depth) -- they are held, not
 *       dropped, then drained into the new model on unblock. Buffers remain
 *       zero-copy (NvBufSurface by reference) throughout.
 *
 *   (2) Make-before-break at the ModelBin level: request the NEW model's pads and
 *       relink in_q.src->new.sink + new.src->out_q.sink, THEN release the old
 *       model's pads. in_q, out_q, the out_mux slot and the tee pad are
 *       PERSISTENT across a swap, so NO element state change and NO bin add/remove
 *       happen on the streaming thread (which would deadlock or add latency).
 *
 *  Threading: decisions are taken under modelmux_bin->lock by the (single) bus thread;
 *  probes run later on the queue's streaming thread and re-resolve the stream
 *  under modelmux_bin->lock (so a concurrent stream-remove can't free it underfoot). The
 *  probe is installed WITHOUT holding modelmux_bin->lock, because an IDLE probe may fire
 *  synchronously in the caller and the callback re-takes modelmux_bin->lock.
 * ================================================================== */

/* Attach ONLY the ModelBin half of a role (queues persist across a swap).
 * Returns the new model's per-stream src ghost (BORROWED -- owned by the
 * ModelBin, do not unref) and, via @mb_sink_out, its sink ghost (NEW ref --
 * caller unrefs). NULL on failure (capacity / not-ready / pad request). */
static GstPad *
model_pool_attach_bin (ModelPool * rb, guint stream_id,
    const gchar * model_name, gint gpu, GstPad ** mb_sink_out, ModelBin ** mb_out)
{
  ModelBin *model_bin;
  GstPad *model_src;
  const gchar *sname;
  gchar sinkname[64];

  *mb_sink_out = NULL;
  if (mb_out) *mb_out = NULL;
  /* swap target: a warm shard with room, or grow a new one (same as add path). A reroute
   * does NOT passthrough on a sharding-cap overflow -- it just fails (NULL), leaving the
   * stream on its current model (prepare-then-commit means no cutover happens). */
  model_bin = modelmux_pool_shard_for_attach (rb, model_name, NULL, gpu);
  if (!model_bin)
    return NULL;
  model_src = model_bin_attach (model_bin, stream_id);     /* ghost src_<idx> (borrowed) */
  if (!model_src)
    return NULL;
  sname = GST_PAD_NAME (model_src);                    /* "src_<idx>" */
  g_snprintf (sinkname, sizeof (sinkname), "sink_%s", sname + 4);
  *mb_sink_out = gst_element_get_static_pad (model_bin->bin, sinkname);   /* NEW ref */
  if (!*mb_sink_out) {
    model_bin_detach (model_bin, stream_id);
    return NULL;
  }
  if (mb_out) *mb_out = model_bin;
  return model_src;
}

typedef struct
{
  ModelMuxBin *modelmux_bin;
  guint stream_id;
  gboolean shadow;             /* which role slot to act on */
  gchar *new_model;            /* swap target (NULL for clear) */
  gboolean unload_if_idle;     /* clear only: auto-unload the cleared model if it goes
                                * idle (used by an in-place PROMOTE to drop the old
                                * primary it just demoted to shadow) */
  gboolean promote_top_row;    /* clear only: after freeing this (shadow) display slot,
                                * move the SIBLING primary's output up to the top row so a
                                * promoted model is shown as a primary */
  /* deferred FOLLOW-UP (see ModelMuxFollowup): the OTHER role's op on this stream,
   * dispatched to the MAIN LOOP once this CLEAR resolves -- so a combined
   * primary-CLEAR + shadow-op never runs concurrently on the shared per-stream tee
   * (the same race the hitless cutover's follow-up carry closes for SWAP). */
  gboolean fu_valid;
  gboolean fu_shadow;
  gint     fu_act;             /* ModelMuxActKind (declared later in this file) */
  gchar   *fu_model;
  gint     fu_gpu;
  gboolean fu_via_default;
} ModelMuxRoleOpCtx;

/* GDestroyNotify of the CLEAR IDLE probe: single owner of the ctx. GStreamer runs it
 * exactly once per installed hook -- including a hook that never fires (pad torn down
 * first), which previously LEAKED the ctx (NULL notify + free-in-callback). The
 * callback must NOT free the ctx anymore; it only steals what it consumes. */
static void
modelmux_role_op_ctx_notify (gpointer data)
{
  ModelMuxRoleOpCtx *ctx = (ModelMuxRoleOpCtx *) data;
  if (!ctx)
    return;
  g_free (ctx->new_model);
  g_free (ctx->fu_model);
  g_free (ctx);
}

/* deferred display-slot move (zero-drop): relink a role's out_q -> a different combined-
 * mux sink slot, under an IDLE probe on the out_q src so no buffer is in flight. */
typedef struct
{
  ModelMuxBin *modelmux_bin;
  guint stream_id;
  gint slot;                   /* target combined-mux sink index */
  gboolean shadow;             /* which role's branch to move */
} ModelMuxSlotMoveCtx;

static GstPadProbeReturn
modelmux_move_slot_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSlotMoveCtx *c = (ModelMuxSlotMoveCtx *) udata;
  ModelMuxBin *modelmux_bin = c->modelmux_bin;
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach *ra;
  GstPad *peer = NULL, *mux_sink;
  gchar pname[64];
  (void) pad;
  (void) info;

  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return GST_PAD_PROBE_REMOVE;               /* c freed by the probe notify */

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) c->stream_id));
  if (!e)
    goto out;
  ra = c->shadow ? &e->shad : &e->prim;
  if (!ra->active || !ra->out_q || !ra->out_mux_sink || ra->out_idx == c->slot)
    goto out;
  peer = gst_pad_get_peer (ra->out_mux_sink);            /* out_q.src */
  if (peer)
    gst_pad_unlink (peer, ra->out_mux_sink);
  gst_element_release_request_pad (modelmux_bin->out_mux, ra->out_mux_sink);  /* free old slot */
  gst_object_unref (ra->out_mux_sink);         /* release caller's ref from request_pad */
  g_snprintf (pname, sizeof (pname), "sink_%u", c->slot);
  mux_sink = gst_element_request_pad_simple (modelmux_bin->out_mux, pname);
  if (!mux_sink)
    MM_ERR ("MultiModelBin: move-slot stream %u %s -- request_pad '%s' failed; "
        "stream is now disconnected from combined-mux",
        c->stream_id, c->shadow ? "Shadow" : "Primary", pname);
  if (mux_sink && peer)
    modelmux_link_pads (peer, mux_sink);
  ra->out_mux_sink = mux_sink;
  ra->out_idx = c->slot;
  MM_INFO ("  - MultiModelBin  stream %u %s display moved to top row (slot %d) "
      "(zero-drop)", c->stream_id, c->shadow ? "Shadow" : "Primary", c->slot);
out:
  g_mutex_unlock (&modelmux_bin->lock);
  if (peer)
    gst_object_unref (peer);
  return GST_PAD_PROBE_REMOVE;                 /* c freed by the probe notify */
}

/* Auto-unload a model that a reroute just displaced, IFF it is now unused (0 streams
 * across all shards). This fires on a full promote/reroute where every stream moved off
 * the old model (e.g. model_update ALL primary=B shadow=none -> the old primary frees);
 * a SCOPED reroute that leaves the model serving other streams keeps it (not idle).
 * A configured DEFAULT model is never auto-unloaded (it is the baseline, reloaded only at
 * startup). Safe in the swap-probe context (same as the dedup free path). */
static void
modelmux_pool_unload_if_idle (ModelPool * rb, const gchar * model_name)
{
  ModelBin *base;
  if (!rb || !model_name || !*model_name)
    return;
  /* never auto-unload a configured default. model_name is the canonical composite
   * key ("name@version"); the configured default is also (name, version), so compare
   * against the composite default key -- NOT the bare default_primary/default_shadow
   * name (which would never match "Default_Primary@1" and leak the default). */
  if (modelmux_key_is_configured_default ((ModelMuxBin *) rb->owner, model_name))
    return;                                      /* a configured default is never auto-unloaded */
  base = g_hash_table_lookup (rb->models, model_name);
  if (base && model_bin_num_streams (base) == 0 && !base->next_shard) {
    MM_INFO ("    - ModelPool[%s]  '%s' displaced by reroute and now idle (0 streams) "
        "-> auto-unload (free VRAM)", rb->role, model_name);
    model_pool_unload_model (rb, model_name);
    return;
  }
  /* Not in the role pool: the stream-drain that displaced it may have COMPACTED it
   * into limbo (compact-idle-models-across-pools). A FULL displacement means nobody
   * wants this model any more, so free the limbo copy too -- identical net behaviour
   * to the pre-compaction path (the displaced model is reclaimed, not silently kept
   * warm). Runs under modelmux_bin->lock (held by every unload_if_idle caller). */
  {
    ModelMuxBin *modelmux_bin = (ModelMuxBin *) rb->owner;
    ModelBin *lb = (modelmux_bin && modelmux_bin->limbo_models) ?
        g_hash_table_lookup (modelmux_bin->limbo_models, model_name) : NULL;
    if (lb && model_bin_num_streams (lb) == 0 && !lb->next_shard) {
      MM_INFO ("    - ModelPool[%s]  '%s' displaced by reroute and now idle (0 streams) "
          "-> auto-unload (free VRAM; from limbo)", rb->role, model_name);
      g_hash_table_remove (modelmux_bin->limbo_models, model_name);   /* frees key only */
      /* deferred: reachable from the CLEAR probe (streaming thread, modelmux_bin->lock held) */
      modelmux_schedule_model_bin_free (modelmux_bin, lb);
    }
  }
}

/* ================================================================== *
 *  SHARD AUTO-COMPACTION (eager + debounced; config->shard_compact).
 *
 *  Auto-grow created extra shards on overflow; this packs them back together as
 *  streams drain. When a model's live streams would fit in fewer shards than are
 *  allocated, migrate streams off the highest shard(s) into free slots on lower
 *  shards (make-before-break, ONE stream at a time), then free the emptied extra
 *  shard. Each migration is a same-model, same-gie shard re-point -- the out_q,
 *  out_mux slot, role, provenance and display tile are ALL persistent/unchanged,
 *  so it is invisible downstream and zero-drop (the in_q absorbs the brief block).
 *
 *  Flow: a stream-remove schedules a debounced pass (modelmux_schedule_compact). The pass
 *  (modelmux_pool_compact_main, main loop) reclaims empty extra shards, then installs ONE
 *  migration IDLE-probe; the probe commits the move and re-triggers the pass to
 *  reclaim the now-empty source shard and continue, until every model is optimal.
 *  One-at-a-time + main-loop frees keep it race-free and off the streaming thread.
 * ================================================================== */

/* debounce window (ms) before a compaction pass runs; coalesces a burst of removes.
 * Default 1000; override with MM_COMPACT_DEBOUNCE. */
static guint
modelmux_compact_debounce_ms (void)
{
  static gsize cached = 0;
  if (g_once_init_enter (&cached)) {
    const gchar *e = g_getenv ("MM_COMPACT_DEBOUNCE");
    gint v = e ? atoi (e) : 0;
    g_once_init_leave (&cached, (gsize)((v > 0) ? v : 1000));
  }
  return (guint) cached;
}

static gboolean modelmux_pool_compact_main (gpointer data);   /* fwd (re-trigger) */

/* ================================================================== *
 *  HITLESS CUTOVER -- zero-loss live re-point of a stream's persistent in_q/out_q
 *  from its current ModelBin to another.
 *
 *  "Hitless" is the telecom/networking term (protection switching, failover) for a
 *  live switchover with ZERO traffic loss. Plain make-before-break has a SEAM: it
 *  cuts old_src->out_q while frames are still in flight inside the old bin
 *  (mux->nvinfer->demux), orphaning them. Hitless cutover closes that seam by DRAINING
 *  the old bin before the cut:
 *
 *    BLOCK  in_q.src (IDLE)         -> new input parks in in_q (held, not dropped)
 *    MAKE   attach the new bin pads -> ready; old still flowing
 *    DRAIN  with the input blocked, the old bin gets no new frames for this stream and
 *           flushes whatever is in flight out to out_q on its own thread; we wait a
 *           bounded window for that to finish
 *    RELINK swing in_q/out_q to the new bin, detach old, unblock -> parked input flows
 *
 *  WHY A TIMED DRAIN (not an in-band marker): the first design injected a serialized
 *  custom-downstream "drain marker" event behind the old bin's last frame and cut when
 *  it re-emerged. But legacy nvstreammux forwards custom downstream events OUT-OF-BAND
 *  (ahead of its batched buffers), so the marker raced past the in-flight frame and
 *  signalled "drained" prematurely -- the cutover still lost ~1 frame. A bounded drain
 *  wait is race-free: the bin is provably idle for this stream once no input arrives for
 *  longer than its in-flight depth. The window is generous vs realistic depth; it can
 *  NEVER stall (the timer always fires) and never drops (cut only after drain). All real
 *  work runs on the MAIN LOOP; streaming threads only block/idle.
 *
 *  ONE primitive backs both re-point flavours; they differ ONLY at MAKE + COMMIT:
 *    - shard compaction (co->new_model==NULL): re-point to ANOTHER SHARD of the SAME
 *      model -- MAKE attaches a sibling shard (co->target), COMMIT detaches the emptied
 *      source shard (co->source). gie-id/out_q/tile all unchanged.
 *    - model swap/reroute (co->new_model!=NULL): re-point to a DIFFERENT model -- MAKE
 *      attaches that model via its pool, COMMIT adopts the new name and releases the OLD
 *      model by name (auto-unload if it falls idle).
 *  The block->drain->relink->unblock seam is byte-identical for both, so neither can lose
 *  a frame; only the endpoints differ.
 * ================================================================== */
#define MM_HITLESS_DRAIN_MS  300       /* bounded drain window before the cut (ms) */
#define MM_HITLESS_PREBLOCK_MS 5000    /* safety net: max wait for the block probe to
                                        * ever FIRE (block probes only trigger on data
                                        * flow -- a stalled/paused source or a torn-down
                                        * lane would otherwise leak the ctx + pad refs
                                        * and leave ra->busy stuck forever) */

/* a role-level action decided by modelmux_role_decide. Defined here (earlier than its primary
 * use) so a cutover can carry a deferred follow-up op of this kind. MIGRATE = SAME model,
 * DIFFERENT device: a compaction-style shard migration constrained to the requested gpu
 * (the swap commit path detaches the old model BY NAME, which is ambiguous when old and
 * new name are equal -- migration cuts shard-to-shard instead). */
typedef enum { MM_ACT_NONE, MM_ACT_SWAP, MM_ACT_CLEAR, MM_ACT_ENABLE,
  MM_ACT_MIGRATE } ModelMuxActKind;

/* A SECOND role op on the SAME stream, deferred to run only AFTER an in-flight cutover on
 * the other role completes. Running both role ops at once races on the stream's shared
 * per-stream tee (one branch is block-drained for the cutover while the sibling branch is
 * relinked/attached), which intermittently STALLS the stream (frames stop reaching nvinfer).
 * Serializing the two ops removes the race -- it degrades the combined update to two
 * sequential single-role swaps, which are reliable. */
typedef struct {
  gboolean  valid;
  gboolean  shadow;            /* role the follow-up acts on */
  ModelMuxActKind act;               /* SWAP / ENABLE / CLEAR */
  gchar    *model;             /* target model (NULL for clear); borrowed by cutover_begin */
  gint      gpu;               /* target placement constraint (MM_GPU_ANY = any) */
  gboolean  via_default;       /* origin of the follow-up's target (see ModelMuxRoleAttach) */
} ModelMuxFollowup;

static void modelmux_hitless_launch_followup (ModelMuxBin * modelmux_bin, guint stream_id,
    gboolean shadow, ModelMuxActKind act, gchar * model, gint gpu, gboolean via_default);  /* defined after modelmux_role_launch */

/* main-loop trampoline for a follow-up launched from a STREAMING-thread probe (the
 * CLEAR IDLE probe): role ops mutate the graph (attach/set_state/cutover) and must not
 * run on a streaming thread. Fired via modelmux_tracked_idle_add: on dispatch the
 * callback frees the wrapper; a never-dispatched source frees it via the notify. */
typedef struct
{
  ModelMuxBin *modelmux_bin;
  guint    stream_id;
  gboolean shadow;
  gint     act;                /* ModelMuxActKind */
  gchar   *model;
  gint     gpu;
  gboolean via_default;
} ModelMuxFuDispatch;

static void
modelmux_fu_dispatch_free (gpointer data)
{
  ModelMuxFuDispatch *fd = (ModelMuxFuDispatch *) data;
  g_free (fd->model);
  g_free (fd);
}

static gboolean
modelmux_fu_dispatch_main (gpointer data)
{
  ModelMuxFuDispatch *fd = (ModelMuxFuDispatch *) data;
  modelmux_hitless_launch_followup (fd->modelmux_bin, fd->stream_id, fd->shadow,
      (ModelMuxActKind) fd->act, fd->model, fd->gpu, fd->via_default);
  modelmux_fu_dispatch_free (fd);       /* dispatched -> the notify skips destroy */
  return G_SOURCE_REMOVE;
}

typedef struct
{
  ModelMuxBin *modelmux_bin;
  const gchar *op;             /* operation label for logs (e.g. "shard-compact",
                               *   "swap") -- this is a SHARED primitive, so the caller
                               *   names what drove the cutover. Static string. */
  guint    stream_id;
  guint    branch_serial;      /* stream INSTANCE guard: source ids are recycled, so a
                               *   remove + re-add during the drain window would make
                               *   the id-only re-resolution operate on the WRONG
                               *   (fresh) incarnation -- every re-resolution below is
                               *   gated on e->branch_serial matching (same discipline
                               *   as ModelMuxPromoteCtx). */
  gboolean shadow;             /* which role slot */
  gchar   *new_model;          /* NULL = compaction (migrate to another shard of the SAME
                               *   model); non-NULL = switch this role to THIS model
                               *   (swap/reroute) -- the only difference is MAKE (attach a
                               *   different model) + COMMIT (detach the OLD model, adopt
                               *   the new). The block->drain->relink->unblock seam is
                               *   identical, so one primitive serves both. */
  gint     new_gpu;            /* placement constraint for the swap target
                               *   (MM_GPU_ANY = any instance) -- adopted into
                               *   ra->req_gpu at COMMIT. */
  gboolean via_default;        /* origin of the swap target -- adopted into
                               *   ra->via_default at COMMIT (swap/migrate only;
                               *   compaction preserves the lane's flag). */
  gint     finished;           /* one-shot CAS guard (drain-done vs any future trigger) */
  gint     block_sched;        /* one-shot CAS guard (block probe -> on_blocked) */
  /* persistent queue pads (refs held for the cutover's lifetime) */
  GstPad  *inq_src;            /* blocked input pad */
  GstPad  *outq_sink;
  /* old/new bins + their per-stream ghost pads (resolved at block time) */
  ModelBin *source;
  ModelBin *target;
  GstPad  *old_sink, *old_src; /* refs */
  GstPad  *new_sink;           /* ref */
  GstPad  *new_src;            /* borrowed (owned by the new bin) */
  gulong   block_id;           /* IDLE probe on inq_src */
  guint    timeout_id;         /* safety g_timeout */
  gint     refs;               /* 1 owner ref + 1 held by the input BLOCK probe (its
                                * GDestroyNotify). Guarantees the ctx survives a probe
                                * callback that is mid-flight when the owner resolves,
                                * and that an armed-but-never-fired probe (idle lane
                                * torn down / shutdown) cannot leak the ctx. */
  /* deferred follow-up: the OTHER role's op on this stream, launched only after THIS
   * cutover ends (commit OR abort), so the two never run concurrently. See ModelMuxFollowup. */
  gboolean  fu_valid;
  gboolean  fu_shadow;
  ModelMuxActKind fu_act;
  gchar    *fu_model;
  gint      fu_gpu;            /* follow-up's placement constraint (MM_GPU_ANY = any) */
  gboolean  fu_via_default;    /* follow-up target's origin */
} ModelMuxHitlessCutover;

/* destructor: pad refs + strings (NOT the borrowed new_src). Reached only via
 * modelmux_hitless_unref when the last ref (owner or block-probe notify) drops. */
static void
modelmux_hitless_destroy (ModelMuxHitlessCutover * co)
{
  if (co->inq_src) gst_object_unref (co->inq_src);
  if (co->outq_sink) gst_object_unref (co->outq_sink);
  if (co->old_sink) gst_object_unref (co->old_sink);
  if (co->old_src) gst_object_unref (co->old_src);
  if (co->new_sink) gst_object_unref (co->new_sink);
  g_free (co->new_model);
  g_free (co->fu_model);
  g_free (co);
}

static void
modelmux_hitless_unref (gpointer data)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) data;
  if (co && g_atomic_int_dec_and_test (&co->refs))
    modelmux_hitless_destroy (co);
}


static void
modelmux_hitless_ctx_drop (gpointer data)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) data;
  if (!co)
    return;
  if (co->timeout_id)
    modelmux_tracked_source_cancel (co->modelmux_bin, &co->timeout_id, FALSE);
  if (co->block_id && co->inq_src) {
    gst_pad_remove_probe (co->inq_src, co->block_id);
    co->block_id = 0;
  }
  modelmux_hitless_unref (co);                     /* owner ref */
}

/* Undo the MAKE (new attach) when a cutover aborts/rolls back, for EITHER flavour:
 *   - compaction (new_model==NULL): the new pads came from a specific shard (co->target)
 *     -> detach that shard;
 *   - swap (new_model!=NULL): the new pads came from the model's pool, which may have
 *     picked or grown any shard -> release by model name. The role pool is stable for the
 *     bin's lifetime (modelmux_bin->primary_pool/modelmux_bin->shadow_pool), so this is safe even if the stream entry is
 *     already gone. No-op if MAKE never produced pads. */
static void
modelmux_hitless_undo_new (ModelMuxHitlessCutover * co)
{
  if (!co->new_sink && !co->new_src)
    return;                                    /* nothing was attached */
  if (co->new_model) {
    ModelPool *role = co->shadow ? co->modelmux_bin->shadow_pool : co->modelmux_bin->primary_pool;
    if (role)
      model_pool_detach (role, co->stream_id, co->new_model);
  } else if (co->target) {
    model_bin_detach (co->target, co->stream_id);
  }
}

/* RELINK + COMMIT + cleanup. Runs ONCE on the main loop (the caller won the CAS). */
static gboolean
modelmux_hitless_finish (gpointer data)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) data;
  ModelMuxBin *modelmux_bin = co->modelmux_bin;
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach *ra = NULL;
  gboolean committed = FALSE;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_hitless_ctx_drop (co);
    return G_SOURCE_REMOVE;
  }

  /* cancel the safety timeout. The input block STAYS until AFTER the relink (removed
   * at the very end) -- unblocking before the cut would briefly feed parked frames into
   * the still-linked old bin that we then detach, losing them (the seam this fixes). */
  if (co->timeout_id)
    modelmux_tracked_source_cancel (modelmux_bin, &co->timeout_id, FALSE);

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) co->stream_id));
  if (e && e->branch_serial != co->branch_serial)
    e = NULL;                       /* id recycled by a re-add mid-drain: treat as
                                     * "stream gone" -- committing would relink RETIRED
                                     * queue pads into the fresh lane, steal its model
                                     * and detach its serving shard */
  if (e) {
    ra = co->shadow ? &e->shad : &e->prim;
    if (ra->active && ra->in_q && ra->out_q && co->new_sink && co->new_src) {
      /* old is drained (marker passed) -> the cut orphans nothing */
      GstPad *cur_old_sink = gst_pad_get_peer (co->inq_src);
      GstPad *cur_old_src = gst_pad_get_peer (co->outq_sink);
      if (cur_old_sink) gst_pad_unlink (co->inq_src, cur_old_sink);
      if (cur_old_src)  gst_pad_unlink (cur_old_src, co->outq_sink);
      if (gst_pad_link (co->inq_src, co->new_sink) == GST_PAD_LINK_OK &&
          gst_pad_link (co->new_src, co->outq_sink) == GST_PAD_LINK_OK) {
        /* SEED the freshly-linked NEW model nvstreammux sink with stream-start/caps/segment from
         * the live in_q src (authoritative holder), while inq_src is STILL blocked (the block is
         * removed at the very end -- "UNBLOCK LAST"). Without this the swapped-in mux can hold the
         * stream's buffers waiting for a SEGMENT it never saw -- the same wedge as a fresh attach.
         * Idempotent (GstPad de-dups) and ordered; logs a missing/refused SEGMENT. */
        modelmux_lane_seed_events (co->inq_src, co->new_sink, co->stream_id,
            co->new_model ? co->new_model : (ra ? ra->model : NULL));
        if (co->target)
          model_bin_remove_zero_stream_drop (co->target);
        if (co->new_model) {
          /* SWAP/REROUTE COMMIT: adopt the new model name and release the OLD model by
           * name (its pool detaches the right shard; auto-unloads if no stream is left on
           * it -> frees VRAM). The engine of the new model stays warm. */
          gchar *old_model = ra->model;                 /* steal */
          ra->model = g_strdup (co->new_model);
          ra->req_gpu = co->new_gpu;          /* the swap's placement is now this lane's */
          ra->via_default = co->via_default;  /* ...and so is its origin */
          if (old_model && ra->role) {
            model_pool_detach (ra->role, co->stream_id, old_model);
            modelmux_pool_unload_if_idle (ra->role, old_model);
          }
          MM_INFO ("  - MultiModelBin  stream %u %s HITLESS %s  '%s' -> '%s' (zero-drop)",
              co->stream_id, ra->role ? ra->role->role : "?", co->op ? co->op : "swap",
              old_model ? old_model : "-", co->new_model);
          g_free (old_model);
        } else {
          /* COMPACTION / GPU-MIGRATE COMMIT: same model, just release the source shard. */
          if (co->new_gpu != MM_GPU_ANY) {
            ra->req_gpu = co->new_gpu;   /* gpu-migrate: lane now pinned to the device */
            ra->via_default = co->via_default; /* an explicit gpu-addressed route PINS the
                                                * lane; pure compaction (gpu==ANY) preserves */
          }
          if (co->source)
            model_bin_detach (co->source, co->stream_id);
          MM_INFO ("  - MultiModelBin  stream %u %s HITLESS %s  shard#%u "
              "-> shard#%u (model '%s', gie kept, zero-drop)", co->stream_id,
              ra->role ? ra->role->role : "?", co->op ? co->op : "cutover",
              co->source ? co->source->inst : 0,
              co->target ? co->target->inst : 0, ra->model ? ra->model : "?");
        }
        modelmux_acct_role_rewarm (modelmux_bin, co->stream_id,
            co->shadow ? MM_ROLE_SHADOW : MM_ROLE_PRIMARY);
        committed = TRUE;
      } else {                              /* relink failed -> roll back to old */
        MM_ERR ("MultiModelBin: stream %u hitless relink failed -- rolling back",
            co->stream_id);
        gst_pad_unlink (co->inq_src, co->new_sink);
        gst_pad_unlink (co->new_src, co->outq_sink);
        modelmux_hitless_undo_new (co);
        /* verify the rollback relink actually restored the old path; a silent failure here
         * leaves the stream disconnected while the caller believes it recovered. */
        if (cur_old_sink && gst_pad_link (co->inq_src, cur_old_sink) != GST_PAD_LINK_OK)
          MM_ERR ("MultiModelBin: stream %u hitless ROLLBACK input relink FAILED -- "
              "stream may be disconnected", co->stream_id);
        if (cur_old_src && gst_pad_link (cur_old_src, co->outq_sink) != GST_PAD_LINK_OK)
          MM_ERR ("MultiModelBin: stream %u hitless ROLLBACK output relink FAILED -- "
              "stream may be disconnected", co->stream_id);
      }
      if (cur_old_sink) gst_object_unref (cur_old_sink);
      if (cur_old_src) gst_object_unref (cur_old_src);
    } else {                               /* stream/role changed -> drop new attach */
      modelmux_hitless_undo_new (co);
    }
    if (ra)
      ra->busy = FALSE;
  } else {                                  /* stream gone mid-cutover */
    modelmux_hitless_undo_new (co);
  }
  g_mutex_unlock (&modelmux_bin->lock);

  /* UNBLOCK LAST: in_q.src now points at the NEW bin, so releasing the block lets the
   * parked input flow to new -- never into the detached old. */
  if (co->block_id && co->inq_src)
    gst_pad_remove_probe (co->inq_src, co->block_id);

  /* steal the deferred follow-up (the OTHER role's op) before freeing the context, then
   * launch it now that this cutover is fully done -- serialized, never concurrent. */
  gboolean  fu_valid  = co->fu_valid;
  gboolean  fu_shadow = co->fu_shadow;
  ModelMuxActKind fu_act    = co->fu_act;
  gchar    *fu_model  = co->fu_model;  co->fu_model = NULL;   /* steal */
  gint      fu_gpu    = co->fu_gpu;
  gboolean  fu_via    = co->fu_via_default;
  guint     stream_id = co->stream_id;

  modelmux_hitless_unref (co);                     /* owner ref */
  if (committed)                            /* reclaim emptied shard + continue */
    modelmux_schedule_compact_idle (modelmux_bin);
  if (fu_valid)
    modelmux_hitless_launch_followup (modelmux_bin, stream_id, fu_shadow, fu_act, fu_model, fu_gpu,
        fu_via);
  g_free (fu_model);
  return G_SOURCE_REMOVE;
}

/* bounded drain window elapsed: the old bin has had no new input for this stream long
 * enough to flush its in-flight frames to out_q, so the cut now orphans nothing. */
static gboolean
modelmux_hitless_drain_done (gpointer data)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) data;
  co->timeout_id = 0;                       /* auto-removed on return */
  if (g_atomic_int_get (&co->modelmux_bin->shutting_down)) {
    modelmux_hitless_ctx_drop (co);
    return G_SOURCE_REMOVE;
  }
  if (g_atomic_int_compare_and_exchange (&co->finished, 0, 1)) {
    MM_INFO ("  - MultiModelBin  stream %u drained (%dms, input blocked) -> committing "
        "hitless cutover", co->stream_id, MM_HITLESS_DRAIN_MS);
    modelmux_hitless_finish (co);
  }
  return G_SOURCE_REMOVE;
}

/* main loop, in_q.src now blocked: MAKE the new pads, then arm the drain probe +
 * timeout and inject the marker. Aborts cleanly (unblock) if the target is gone. */
static gboolean
modelmux_hitless_on_blocked (gpointer data)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) data;
  ModelMuxBin *modelmux_bin = co->modelmux_bin;
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach *ra = NULL;
  ModelPool *rb;
  ModelBin *base, *model_bin;
  gboolean ok = FALSE;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_hitless_ctx_drop (co);
    return G_SOURCE_REMOVE;
  }

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) co->stream_id));
  if (!e || e->branch_serial != co->branch_serial)
    goto unlock;                    /* stream gone OR id recycled by a re-add: this
                                     * cutover belongs to a dead incarnation -- abort
                                     * without touching the fresh lane */
  ra = co->shadow ? &e->shad : &e->prim;
  if (!ra->active || !ra->in_q || !ra->out_q || !ra->model || !ra->role)
    goto unlock;
  rb = ra->role;
  if (co->new_model) {
    /* SWAP/REROUTE MAKE: bring up the pads of a DIFFERENT model. Its pool selects (or
     * grows) a shard with capacity; co->target records the NEW shard so any zero-stream
     * fence can be removed after COMMIT. The OLD model is released by NAME at COMMIT,
     * not by shard. new_src is borrowed (owned by the ModelBin); new_sink is an owned
     * ref freed in modelmux_hitless_destroy. */
    co->new_src = model_pool_attach_bin (rb, co->stream_id, co->new_model,
        co->new_gpu, &co->new_sink, &co->target);
    if (!co->new_src || !co->new_sink) {
      MM_ERR ("MultiModelBin: stream %u %s swap to '%s' failed (attach) -- "
          "keeping current model", co->stream_id, rb->role, co->new_model);
      goto unlock;
    }
  } else {
    /* COMPACTION / GPU-MIGRATE MAKE: migrate to ANOTHER SHARD of the SAME model. */
    gboolean is_migrate = (co->new_gpu != MM_GPU_ANY);
    base = g_hash_table_lookup (rb->models, ra->model);
    if (!base || (!base->next_shard && !is_migrate))
      goto unlock;                       /* plain compaction needs 2+ shards */
    for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard)
      if (model_idx_of (model_bin, co->stream_id) >= 0) { co->source = model_bin; break; }
    if (!co->source) {
      if (is_migrate)
        MM_WARN ("MultiModelBin: stream %u gpu-migrate to gpu %d ABORTED -- the "
            "stream's serving shard of '%s' vanished mid-cutover (stream keeps "
            "its current placement)", co->stream_id, co->new_gpu, ra->model);
      goto unlock;
    }
    /* device constraint: a gpu-MIGRATE cutover carries the REQUESTED device in
     * new_gpu; plain compaction inherits the lane's recorded constraint -- either
     * way a stream is never silently moved onto another device. */
    gint want_gpu = is_migrate ? co->new_gpu : ra->req_gpu;
    for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
      ModelStatus st = model_bin_status (model_bin);
      if (model_bin != co->source && (st == MODEL_WARMED || st == MODEL_SERVING) &&
          model_bin_has_capacity (model_bin) &&
          (want_gpu == MM_GPU_ANY ||
           modelmux_shard_gpu_effective (modelmux_bin, model_bin) == want_gpu)) { co->target = model_bin; break; }
    }
    if (!co->target && is_migrate) {
      /* every shard on the requested device is full (or none exists yet): scale
       * OUT on that device exactly like a constrained attach would, instead of
       * silently no-oping a route the control plane already acknowledged. */
      co->target = modelmux_pool_shard_for_attach (rb, ra->model, NULL, want_gpu);
      if (co->target == co->source)
        co->target = NULL;               /* defensive: never migrate onto itself */
    }
    if (!co->target) {
      if (is_migrate)
        MM_WARN ("MultiModelBin: stream %u gpu-migrate of '%s' to gpu %d ABORTED "
            "-- no instance/capacity on that device and scale-out failed; the "
            "stream stays on its current device", co->stream_id, ra->model,
            co->new_gpu);
      goto unlock;
    }
    co->new_src = model_bin_attach (co->target, co->stream_id);   /* MAKE */
    if (!co->new_src) { co->target = NULL; goto unlock; }
    {
      const gchar *sname = GST_PAD_NAME (co->new_src);
      gchar sn[64];
      g_snprintf (sn, sizeof (sn), "sink_%s", sname + 4);
      co->new_sink = gst_element_get_static_pad (co->target->bin, sn);
    }
  }
  co->outq_sink = gst_element_get_static_pad (ra->out_q, "sink");
  if (co->new_sink && co->outq_sink && co->inq_src) {
    co->old_sink = gst_pad_get_peer (co->inq_src);    /* old bin sink ghost */
    co->old_src = gst_pad_get_peer (co->outq_sink);   /* old bin src ghost  */
    ok = (co->old_sink != NULL && co->old_src != NULL);
  }
  if (!ok) {                                /* make partially failed -> roll back */
    modelmux_hitless_undo_new (co);
    co->target = NULL;
  }
unlock:
  g_mutex_unlock (&modelmux_bin->lock);

  if (!ok) {                                /* abort: unblock, release role, free */
    if (co->block_id && co->inq_src)
      gst_pad_remove_probe (co->inq_src, co->block_id);
    g_mutex_lock (&modelmux_bin->lock);
    if ((e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) co->stream_id))) &&
        e->branch_serial == co->branch_serial) {   /* never clear a FRESH incarnation's busy */
      ra = co->shadow ? &e->shad : &e->prim;
      ra->busy = FALSE;
    }
    g_mutex_unlock (&modelmux_bin->lock);
    /* still run the deferred follow-up so the OTHER role's op (and its reserved busy) is
     * not stranded just because THIS cutover could not make its target. */
    {
      gboolean  fu_valid  = co->fu_valid;
      gboolean  fu_shadow = co->fu_shadow;
      ModelMuxActKind fu_act    = co->fu_act;
      gchar    *fu_model  = co->fu_model;  co->fu_model = NULL;
      gint      fu_gpu    = co->fu_gpu;
      gboolean  fu_via    = co->fu_via_default;
      guint     stream_id = co->stream_id;
      modelmux_hitless_unref (co);                 /* owner ref */
      if (fu_valid)
        modelmux_hitless_launch_followup (modelmux_bin, stream_id, fu_shadow, fu_act, fu_model, fu_gpu,
            fu_via);
      g_free (fu_model);
    }
    return G_SOURCE_REMOVE;
  }

  /* the PRE-BLOCK safety net is now moot (the block engaged -- that is why we are
   * here); cancel it so the slot is free for the drain timer. A concurrent fire is
   * harmless: it loses the block_sched CAS and self-neutralizes. */
  if (co->timeout_id)
    modelmux_tracked_source_cancel (modelmux_bin, &co->timeout_id, FALSE);

  /* DRAIN: input is now blocked, so the old bin receives no new frames for this stream
   * and flushes its in-flight frames out to out_q on its own thread. Wait a bounded
   * window, then commit the cut (modelmux_hitless_drain_done -> modelmux_hitless_finish). */
  if (!modelmux_tracked_timeout_add (modelmux_bin, MM_HITLESS_DRAIN_MS, modelmux_hitless_drain_done,
          co, modelmux_hitless_ctx_drop, &co->timeout_id)) {
    modelmux_hitless_ctx_drop (co);
    return G_SOURCE_REMOVE;
  }
  return G_SOURCE_REMOVE;
}

/* BLOCK_DOWNSTREAM probe on in_q.src: fires ONCE when the block engages (holding the
 * in-transit buffer), schedules the cutover on the main loop, and STAYS blocking
 * (return OK holds the pad + the held buffer) until modelmux_hitless_finish removes the probe
 * AFTER relinking -- so the held buffer flows to the NEW bin, not into the void. */
static GstPadProbeReturn
modelmux_hitless_block_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) udata;
  (void) pad;
  (void) info;
  if (g_atomic_int_get (&co->modelmux_bin->shutting_down)) {
    if (g_atomic_int_compare_and_exchange (&co->block_sched, 0, 1)) {
      co->block_id = 0;                    /* this callback's return removes it */
      modelmux_hitless_ctx_drop (co);
    }
    return GST_PAD_PROBE_REMOVE;
  }
  if (g_atomic_int_compare_and_exchange (&co->block_sched, 0, 1) &&
      !modelmux_tracked_idle_add (co->modelmux_bin, modelmux_hitless_on_blocked, co,
          modelmux_hitless_ctx_drop, NULL)) {
    co->block_id = 0;                      /* this callback's return removes it */
    modelmux_hitless_ctx_drop (co);
    return GST_PAD_PROBE_REMOVE;
  }
  return GST_PAD_PROBE_OK;                   /* keep the pad blocked */
}

/* PRE-BLOCK safety net (main loop): the block probe fires only when DATA flows, so a
 * stalled/paused source (or a lane torn down before any buffer arrives) would leave the
 * cutover armed forever: ctx + pad refs leaked in a probe<->ctx ref cycle, and ra->busy
 * stuck TRUE so every later route on the role is refused. Bounded wait: if the block
 * never engages, steal the one-shot via block_sched (the same CAS the probe uses),
 * remove the probe, release the role and resolve the ctx. Losing the CAS means the
 * probe fired first -- on_blocked owns the ctx; do nothing (not even the slot). */
static gboolean
modelmux_hitless_preblock_timeout (gpointer data)
{
  ModelMuxHitlessCutover *co = (ModelMuxHitlessCutover *) data;
  ModelMuxBin *modelmux_bin = co->modelmux_bin;
  ModelMuxStreamEntry *e;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_hitless_ctx_drop (co);
    return G_SOURCE_REMOVE;
  }
  if (!g_atomic_int_compare_and_exchange (&co->block_sched, 0, 1))
    return G_SOURCE_REMOVE;         /* block engaged first: on_blocked owns the ctx (and
                                     * may have re-armed the slot as its DRAIN timer) */
  MM_WARN ("MultiModelBin: stream %u %s cutover ABORTED -- no data reached the lane "
      "within %dms (stalled/paused source, or the stream left before flowing); lane "
      "unblocked, role released -- re-issue the route once the source flows",
      co->stream_id, co->op ? co->op : "hitless", MM_HITLESS_PREBLOCK_MS);
  if (co->block_id && co->inq_src) {
    gst_pad_remove_probe (co->inq_src, co->block_id);   /* probe ref drops via notify */
    co->block_id = 0;
  }
  g_mutex_lock (&modelmux_bin->lock);
  if ((e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) co->stream_id))) &&
      e->branch_serial == co->branch_serial) {
    ModelMuxRoleAttach *ra = co->shadow ? &e->shad : &e->prim;
    ra->busy = FALSE;
  }
  g_mutex_unlock (&modelmux_bin->lock);
  /* launch the stranded follow-up (the OTHER role's op), then drop the owner ref */
  {
    gboolean  fu_valid  = co->fu_valid;
    gboolean  fu_shadow = co->fu_shadow;
    ModelMuxActKind fu_act = co->fu_act;
    gchar    *fu_model  = co->fu_model;  co->fu_model = NULL;
    gint      fu_gpu    = co->fu_gpu;
    gboolean  fu_via    = co->fu_via_default;
    guint     sid       = co->stream_id;
    modelmux_hitless_unref (co);                   /* owner ref */
    if (fu_valid)
      modelmux_hitless_launch_followup (modelmux_bin, sid, fu_shadow, fu_act, fu_model,
          fu_gpu, fu_via);
    g_free (fu_model);
  }
  return G_SOURCE_REMOVE;
}

/* Begin a hitless cutover of (stream_id, role). @op is a static label naming the
 * driving operation (e.g. "shard-compact", "swap") for the logs. Takes ownership of
 * the inq_src ref. Caller must have reserved ra->busy and captured the entry's
 * branch_serial under modelmux_bin->lock (stream-instance guard, see the ctx field).
 * Installs the input block; everything else is driven from the block probe. @fu (or
 * NULL) is a follow-up op on the OTHER role of this stream, launched once this
 * cutover ends -- so a combined two-role update runs sequentially, never concurrently. */
static gboolean
modelmux_hitless_cutover_begin (ModelMuxBin * modelmux_bin, const gchar * op, guint stream_id,
    guint branch_serial, gboolean shadow, GstPad * inq_src, const gchar * new_model,
    gint new_gpu, gboolean via_default, const ModelMuxFollowup * fu)
{
  ModelMuxHitlessCutover *co;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    /* teardown already past the flag: stream_wiring may be mid-destruction --
     * do not touch it (busy is irrelevant now); just release the pad ref. */
    gst_object_unref (inq_src);
    return FALSE;
  }
  co = g_new0 (ModelMuxHitlessCutover, 1);
  co->modelmux_bin = modelmux_bin;
  co->op = op;
  co->stream_id = stream_id;
  co->branch_serial = branch_serial;
  co->shadow = shadow;
  co->new_model = new_model ? g_strdup (new_model) : NULL;  /* NULL => compaction */
  co->new_gpu = new_gpu;
  co->via_default = via_default;
  if (fu && fu->valid) {
    co->fu_valid = TRUE;
    co->fu_shadow = fu->shadow;
    co->fu_act = fu->act;
    co->fu_model = fu->model ? g_strdup (fu->model) : NULL;
    co->fu_via_default = fu->via_default;
    co->fu_gpu = fu->gpu;
  } else {
    co->fu_gpu = MM_GPU_ANY;
  }
  co->inq_src = inq_src;                      /* takes the ref */
  co->refs = 1;                               /* owner ref (resolved by finish/abort/drop) */
  /* BLOCK_DOWNSTREAM (not IDLE): a real, HELD block. IDLE only calls back at each idle
   * instant and lets data keep flowing between calls, so it cannot hold the pad across
   * the async drain -- frames streamed to old and the in-flight one was orphaned at the
   * relink. A block probe holds the pad (and the in-transit buffer) until we remove it
   * in modelmux_hitless_finish, AFTER relinking to the new bin, so that held buffer flows to
   * the new shard instead of being lost. The probe holds its OWN ctx ref (dropped by its
   * GDestroyNotify when the hook is removed or the pad is finalized, deferred past an
   * IN-CALL callback) -- so a probe that never fires cannot leak the ctx and a resolving
   * owner cannot free it under a mid-flight callback. */
  g_atomic_int_inc (&co->refs);
  co->block_id = gst_pad_add_probe (inq_src, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
      modelmux_hitless_block_probe, co, modelmux_hitless_unref);
  if (!co->block_id) {
    ModelMuxStreamEntry *e;
    MM_ERR ("MultiModelBin: stream %u %s cutover failed to install input block",
        stream_id, op ? op : "hitless");
    g_mutex_lock (&modelmux_bin->lock);
    e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
    if (e) {
      ModelMuxRoleAttach *ra = shadow ? &e->shad : &e->prim;
      ra->busy = FALSE;
    }
    g_mutex_unlock (&modelmux_bin->lock);
    modelmux_hitless_unref (co);                    /* probe ref (never installed) */
    modelmux_hitless_unref (co);                    /* owner ref */
    return FALSE;
  }
  /* arm the PRE-BLOCK safety net (see modelmux_hitless_preblock_timeout). on_blocked
   * re-arms the same slot as the DRAIN timer once the block engages; if this timer is
   * still pending then, it loses the block_sched CAS and self-neutralizes. */
  if (!modelmux_tracked_timeout_add (modelmux_bin, MM_HITLESS_PREBLOCK_MS,
          modelmux_hitless_preblock_timeout, co, modelmux_hitless_ctx_drop,
          &co->timeout_id)) {
    /* shutting down: resolve like a probe-install failure */
    ModelMuxStreamEntry *e;
    gst_pad_remove_probe (co->inq_src, co->block_id);
    co->block_id = 0;
    g_mutex_lock (&modelmux_bin->lock);
    e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
    if (e && e->branch_serial == branch_serial) {
      ModelMuxRoleAttach *ra = shadow ? &e->shad : &e->prim;
      ra->busy = FALSE;
    }
    g_mutex_unlock (&modelmux_bin->lock);
    modelmux_hitless_unref (co);                    /* owner ref (probe ref drops via its notify) */
    return FALSE;
  }
  return TRUE;
}

/* One compaction pass (main loop): free empty extra shards, then -- if any model
 * still spans more shards than needed -- launch ONE migration. Returns G_SOURCE_REMOVE
 * so it doubles as a g_idle/g_timeout callback. */
static gboolean
modelmux_pool_compact_main (gpointer data)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) data;
  ModelPool *pools[2];
  GstPad *inq_src = NULL;
  gboolean selected = FALSE;       /* one migration chosen this pass */
  guint sel_sid = 0;
  guint sel_serial = 0;            /* stream-instance guard for the cutover */
  gboolean sel_shadow = FALSE;
  guint pi;

  if (!modelmux_bin)
    return G_SOURCE_REMOVE;
  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return G_SOURCE_REMOVE;
  g_mutex_lock (&modelmux_bin->lock);
  if (!modelmux_bin->config || !modelmux_bin->config->shard_compact) {
    g_mutex_unlock (&modelmux_bin->lock);
    return G_SOURCE_REMOVE;
  }
  pools[0] = modelmux_bin->primary_pool;
  pools[1] = modelmux_bin->shadow_pool;

  for (pi = 0; pi < 2 && !selected; pi++) {
    ModelPool *rb = pools[pi];
    GHashTableIter it;
    gpointer k, v;
    if (!rb)
      continue;
    g_hash_table_iter_init (&it, rb->models);
    while (!selected && g_hash_table_iter_next (&it, &k, &v)) {
      ModelBin *base = (ModelBin *) v;
      ModelBin *model_bin, *prev, *next, *tail = NULL, *target = NULL;
      guint total = 0, nsh = 0, cap, optimal;
      gint sid = -1;
      ModelMuxStreamEntry *e;
      ModelMuxRoleAttach *ra;

      /* instance groups are DELIBERATE placements (one pinned shard per GPU):
       * packing their streams onto fewer shards would defeat the multi-GPU
       * scale-out, so the whole chain is exempt from compaction. */
      for (model_bin = base; model_bin && !model_bin->pinned; model_bin = (ModelBin *) model_bin->next_shard)
        ;
      if (model_bin)
        continue;

      /* (1) reclaim empty EXTRA shards (base shard #0 is never freed) */
      prev = base;
      model_bin = (ModelBin *) base->next_shard;
      while (model_bin) {
        next = (ModelBin *) model_bin->next_shard;
        if (model_bin_num_streams (model_bin) == 0) {
          prev->next_shard = model_bin->next_shard;
          MM_INFO ("    - ModelPool[%s]  compact: empty shard #%u of '%s' -> freed",
              rb->role, model_bin->inst, model_bin->name);
          /* DEFERRED: model_bin_free joins the warm thread + drives TRT to NULL
           * (100s of ms..s); we HOLD modelmux_bin->lock here -- freeing inline would stall the
           * main loop AND every lock waiter (probes, attaches) for the teardown. */
          modelmux_schedule_model_bin_free (modelmux_bin, model_bin);
        } else {
          prev = model_bin;
        }
        model_bin = next;
      }

      /* (2) does this model still span more shards than needed? */
      if (!base->next_shard)
        continue;                            /* single shard -> done */
      for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
        guint n = model_bin_num_streams (model_bin);
        total += n;
        nsh++;
        if (n > 0)
          tail = model_bin;                         /* last non-empty shard */
      }
      cap = base->max_streams ? base->max_streams : 1;
      optimal = (total + cap - 1) / cap;
      if (nsh <= optimal || !tail)
        continue;

      /* target = lowest warm shard with a free slot, not the tail */
      for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
        ModelStatus st = model_bin_status (model_bin);
        if (model_bin != tail && (st == MODEL_WARMED || st == MODEL_SERVING) &&
            model_bin_has_capacity (model_bin)) { target = model_bin; break; }
      }
      if (!target)
        continue;

      /* pick one stream on the tail shard and reserve its role for migration */
      {
        GHashTableIter sit;
        gpointer sk, sv;
        g_hash_table_iter_init (&sit, tail->idx_to_stream);
        if (g_hash_table_iter_next (&sit, &sk, &sv))
          sid = GPOINTER_TO_INT (sv);
      }
      if (sid < 0)
        continue;
      e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER (sid));
      if (!e)
        continue;
      ra = (rb == modelmux_bin->shadow_pool) ? &e->shad : &e->prim;
      if (!ra->active || ra->busy || !ra->in_q ||
          g_strcmp0 (ra->model, base->key) != 0)   /* ra->model is the canonical key */
        continue;
      inq_src = gst_element_get_static_pad (ra->in_q, "src");
      if (!inq_src)
        continue;
      ra->busy = TRUE;                       /* hitless cutover clears it on finish */
      selected = TRUE;
      sel_sid = (guint) sid;
      sel_serial = e->branch_serial;         /* instance guard, captured under the lock */
      sel_shadow = (rb == modelmux_bin->shadow_pool);
      MM_INFO ("    - ModelPool[%s]  compact: '%s' uses %u shard(s), optimal %u -> "
          "hitless-migrate stream %d off shard #%u", rb->role, base->name, nsh,
          optimal, sid, tail->inst);
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);

  /* launch OUTSIDE the lock (the block probe may fire synchronously). begin() takes
   * the inq_src ref. */
  if (selected && inq_src)
    modelmux_hitless_cutover_begin (modelmux_bin, "shard-compact", sel_sid, sel_serial, sel_shadow,
        inq_src, NULL, MM_GPU_ANY, FALSE /* unused: compaction preserves the lane's flag */, NULL);
  else if (inq_src)
    gst_object_unref (inq_src);
  return G_SOURCE_REMOVE;
}

static gboolean
modelmux_compact_debounce_fire (gpointer data)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) data;
  /* under free_lock (like dump_source_id): the schedulers below run on whichever
   * thread drove a detach (a public action signal), not only the main loop */
  g_mutex_lock (&modelmux_bin->free_lock);
  modelmux_bin->compact_poll_id = 0;
  g_mutex_unlock (&modelmux_bin->free_lock);
  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return G_SOURCE_REMOVE;
  modelmux_pool_compact_main (modelmux_bin);
  return G_SOURCE_REMOVE;
}

static void
modelmux_schedule_compact_idle (ModelMuxBin * modelmux_bin)
{
  if (!modelmux_bin || !modelmux_bin->config || !modelmux_bin->config->shard_compact ||
      g_atomic_int_get (&modelmux_bin->shutting_down))
    return;
  g_mutex_lock (&modelmux_bin->free_lock);
  if (modelmux_bin->compact_poll_id)
    g_source_remove (modelmux_bin->compact_poll_id);
  modelmux_bin->compact_poll_id = g_idle_add (modelmux_compact_debounce_fire, modelmux_bin);
  g_mutex_unlock (&modelmux_bin->free_lock);
}

/* Schedule a debounced compaction pass (reset on each call so a burst of removes
 * collapses into one pass). No-op when disabled. Called on the main loop. */
static void
modelmux_schedule_compact (ModelMuxBin * modelmux_bin)
{
  if (!modelmux_bin || !modelmux_bin->config || !modelmux_bin->config->shard_compact ||
      g_atomic_int_get (&modelmux_bin->shutting_down))
    return;
  g_mutex_lock (&modelmux_bin->free_lock);
  if (modelmux_bin->compact_poll_id)
    g_source_remove (modelmux_bin->compact_poll_id);
  modelmux_bin->compact_poll_id =
      g_timeout_add (modelmux_compact_debounce_ms (), modelmux_compact_debounce_fire, modelmux_bin);
  g_mutex_unlock (&modelmux_bin->free_lock);
}

/* CLEAR probe: detach a role entirely (e.g. shadow -> "none"). Stops the tee
 * feeding THIS role (its primary sibling uses a separate tee pad and is
 * untouched), unlinks the branch + releases the out_mux slot, then hands the
 * queues to the main loop for a safe NULL+remove. No demux stall, no deadlock. */
static GstPadProbeReturn
modelmux_clear_idle_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxRoleOpCtx *ctx = (ModelMuxRoleOpCtx *) udata;
  ModelMuxBin *modelmux_bin = ctx->modelmux_bin;
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach *ra = NULL;
  GstPad *inq_src = NULL, *outq_sink = NULL, *outq_src = NULL;
  GstPad *old_sink = NULL, *old_src = NULL;
  GstElement *in_q = NULL, *out_q = NULL;
  GstPad *move_src = NULL;       /* sibling primary out_q.src to lift to the top row */
  gint move_slot = 0;            /* target (top) display slot for that lift            */

  (void) pad;
  (void) info;

  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return GST_PAD_PROBE_REMOVE;               /* ctx freed by the probe notify */

  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) ctx->stream_id));
  if (!e)
    goto out;
  ra = ctx->shadow ? &e->shad : &e->prim;
  if (!ra->active)
    goto out;
  modelmux_role_clear_debug_probes (ra);

  /* stop the tee feeding this role (sibling role keeps its own tee pad/flow) */
  if (ra->tee_src) {
    gst_element_release_request_pad (e->tee, ra->tee_src);
    gst_object_unref (ra->tee_src);
    ra->tee_src = NULL;
  }
  inq_src = ra->in_q ? gst_element_get_static_pad (ra->in_q, "src") : NULL;
  outq_sink = ra->out_q ? gst_element_get_static_pad (ra->out_q, "sink") : NULL;
  outq_src = ra->out_q ? gst_element_get_static_pad (ra->out_q, "src") : NULL;

  if (inq_src) {
    old_sink = gst_pad_get_peer (inq_src);            /* old ModelBin sink ghost */
    if (old_sink) gst_pad_unlink (inq_src, old_sink);
  }
  if (outq_sink) {
    old_src = gst_pad_get_peer (outq_sink);           /* old ModelBin src ghost  */
    if (old_src) gst_pad_unlink (old_src, outq_sink);
  }
  /* release the combined-display-mux slot (the slot is derived from the stream's
   * column; clearing the shadow frees its slot but NOT the column -- the primary
   * still holds the column; it is recycled only at stream teardown). */
  if (ra->out_mux_sink) {
    if (outq_src) gst_pad_unlink (outq_src, ra->out_mux_sink);
    gst_element_release_request_pad (modelmux_bin->out_mux, ra->out_mux_sink);
    gst_object_unref (ra->out_mux_sink);       /* release caller's ref from request_pad */
    ra->out_mux_sink = NULL;
  }
  /* detach the model bin (engine stays warm; -> READY if last stream) */
  if (ra->role && ra->model) {
    model_pool_detach (ra->role, ctx->stream_id, ra->model);
    /* in-place PROMOTE: the model we just demoted-to-shadow-then-cleared should be
     * dropped once it serves nobody (it was the old primary). */
    if (ctx->unload_if_idle)
      modelmux_pool_unload_if_idle (ra->role, ra->model);
  }
  g_free (ra->model);
  ra->model = NULL;
  ra->req_gpu = MM_GPU_ANY;    /* the placement constraint dies with the lane
                                * (mirrors modelmux_detach_role) */
  ra->via_default = FALSE;

  /* hand the queues to the main loop for a deadlock-free NULL+remove */
  in_q = ra->in_q;
  out_q = ra->out_q;
  ra->in_q = NULL;
  ra->out_q = NULL;
  ra->role = NULL;
  ra->out_idx = 0;
  ra->active = FALSE;
  {
    GstElement *conv = ra->conv;       /* tear down the copy converter with the lane */
    ra->conv = NULL;
    if (in_q || out_q || conv)
      modelmux_schedule_branch_free (modelmux_bin, conv, in_q, out_q);
  }
  MM_INFO ("  - MultiModelBin  stream %u %s role CLEARED  (zero-drop; sibling unaffected)",
      ctx->stream_id, ctx->shadow ? "Shadow" : "Primary");

  /* IN-PLACE PROMOTE display fix: clearing this shadow just freed the stream's
   * TOP (primary) display slot -> move the sibling primary's output up into it so the
   * promoted model is shown in the primary row. Zero-drop: the move runs on an IDLE probe
   * on the primary out_q src (queue holds buffers across the relink). */
  if (ctx->promote_top_row && e && e->prim.active && e->prim.out_q) {
    move_src = gst_element_get_static_pad (e->prim.out_q, "src");  /* ref; installed below */
    move_slot = e->column;                       /* primary top-row slot for this stream */
  }

out:
  if (ra)
    ra->busy = FALSE;
  g_mutex_unlock (&modelmux_bin->lock);
  /* install the display-lift probe AFTER releasing modelmux_bin->lock: an IDLE probe can fire
   * synchronously here, and modelmux_move_slot_probe re-takes modelmux_bin->lock -> would deadlock if we
   * were still holding it. */
  if (move_src) {
    ModelMuxSlotMoveCtx *mc = g_new0 (ModelMuxSlotMoveCtx, 1);
    mc->modelmux_bin = modelmux_bin;
    mc->stream_id = ctx->stream_id;
    mc->slot = move_slot;
    mc->shadow = FALSE;
    gst_pad_add_probe (move_src, GST_PAD_PROBE_TYPE_IDLE, modelmux_move_slot_probe, mc,
        g_free);   /* notify owns mc: a hook that never fires must not leak it */
    gst_object_unref (move_src);
  }
  if (inq_src) gst_object_unref (inq_src);
  if (outq_sink) gst_object_unref (outq_sink);
  if (outq_src) gst_object_unref (outq_src);
  if (old_sink) gst_object_unref (old_sink);
  if (old_src) gst_object_unref (old_src);
  /* deferred FOLLOW-UP: the CLEAR has resolved (branch released) -- dispatch the
   * OTHER role's op to the MAIN LOOP now (this probe runs on a streaming thread;
   * role ops mutate the graph). Steal the model so the ctx notify won't free it. */
  if (ctx->fu_valid) {
    ModelMuxFuDispatch *fd = g_new0 (ModelMuxFuDispatch, 1);
    fd->modelmux_bin = modelmux_bin;
    fd->stream_id = ctx->stream_id;
    fd->shadow = ctx->fu_shadow;
    fd->act = ctx->fu_act;
    fd->model = ctx->fu_model;  ctx->fu_model = NULL;   /* steal */
    fd->gpu = ctx->fu_gpu;
    fd->via_default = ctx->fu_via_default;
    ctx->fu_valid = FALSE;
    if (!modelmux_tracked_idle_add (modelmux_bin, modelmux_fu_dispatch_main, fd,
            modelmux_fu_dispatch_free, NULL))
      modelmux_fu_dispatch_free (fd);          /* shutting down: drop it */
  }
  return GST_PAD_PROBE_REMOVE;                 /* ctx freed by the probe notify */
}

/* Pure decision for one role (called under modelmux_bin->lock). NULL = leave unchanged;
 * "" = clear; "X" = route to X. */
static ModelMuxActKind
modelmux_role_decide (ModelMuxRoleAttach * ra, const gchar * new_model)
{
  const gchar *cur = ra->active ? ra->model : NULL;

  if (ra->passthru)
    return MM_ACT_NONE;                      /* a passthrough (no-infer, model-sharding=0
                                              * over-cap) branch is display-only and has no
                                              * model/in-q/out-q to swap or clear -- leave it
                                              * untouched (re-add the stream to re-evaluate
                                              * once the model frees capacity). */
  if (new_model == NULL)
    return MM_ACT_NONE;                      /* role unspecified */
  if (ra->busy)
    return MM_ACT_NONE;                      /* don't stack ops on a role */
  if (!*new_model)
    return ra->active ? MM_ACT_CLEAR : MM_ACT_NONE;
  if (!ra->active)
    return MM_ACT_ENABLE;
  if (cur && g_strcmp0 (cur, new_model) == 0)
    return MM_ACT_NONE;                      /* already on this model */
  return MM_ACT_SWAP;
}

/* Launch the chosen op for one role. ENABLE attaches inline (a make-only live
 * add, drop-free by construction); SWAP/CLEAR install an IDLE probe on the
 * role's in_q.src. MUST be called WITHOUT modelmux_bin->lock held (IDLE probes may fire
 * synchronously and re-take it). Returns TRUE if anything was launched.
 * RE-RESOLVES the stream entry by id under modelmux_bin->lock at every touch: the caller
 * decided lock-free, and a concurrent detach-stream (a public action signal on
 * any thread) can free the entry in between -- cached ModelMuxStreamEntry/ModelMuxRoleAttach
 * pointers must never be trusted across an unlocked window. */
static gboolean
modelmux_role_launch (ModelMuxBin * modelmux_bin, ModelPool * rb, guint stream_id,
    gboolean shadow, ModelMuxActKind act, const gchar * new_model, gint new_gpu,
    gboolean via_default, const ModelMuxFollowup * fu)
{
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach *ra;
  GstPad *inq_src = NULL;
  ModelMuxRoleOpCtx *ctx;

  switch (act) {
    case MM_ACT_ENABLE: {
      /* make-only live add; lock spans the pad-index pool mutation */
      gboolean ok = FALSE;
      g_mutex_lock (&modelmux_bin->lock);
      e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
      ra = e ? (shadow ? &e->shad : &e->prim) : NULL;
      if (ra) {
        ok = modelmux_attach_role (modelmux_bin, e, ra, rb, new_model, new_gpu, via_default, NULL, NULL);
        if (ok)                       /* fresh role epoch (live add starts at gap 0) */
          modelmux_acct_role_attach (modelmux_bin, stream_id, shadow ? MM_ROLE_SHADOW : MM_ROLE_PRIMARY);
      }
      g_mutex_unlock (&modelmux_bin->lock);
      if (ok) {
        MM_INFO ("  - MultiModelBin  stream %u %s role ENABLED -> '%s' (zero-drop)",
            stream_id, rb->role, new_model);
        return TRUE;
      }
      MM_ERR ("MultiModelBin: stream %u %s enable '%s' failed",
          stream_id, rb->role, new_model);
      return FALSE;
    }

    case MM_ACT_SWAP:
    case MM_ACT_MIGRATE:
      /* SWAP is a live RE-POINT to a different model -> route it through the hitless
       * cutover (block -> drain -> relink -> unblock), the same zero-loss primitive that
       * backs shard compaction. The old plain make-before-break (modelmux_swap_idle_probe) cut
       * the link while frames were still in flight inside the old bin, orphaning them; the
       * timed drain closes that seam. cutover_begin TAKES the inq_src ref (no unref here);
       * ra->busy was reserved by the caller and is released inside the cutover.
       * MIGRATE = same model, different DEVICE: the compaction-style shard cut (NULL
       * model) constrained to new_gpu -- the swap commit detaches the old model BY NAME,
       * which is ambiguous when both names are equal. */
    {
      guint cut_serial;
      g_mutex_lock (&modelmux_bin->lock);
      e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
      ra = e ? (shadow ? &e->shad : &e->prim) : NULL;
      if (ra && ra->in_q)
        inq_src = gst_element_get_static_pad (ra->in_q, "src");
      if (!inq_src) {
        if (ra)
          ra->busy = FALSE;
        g_mutex_unlock (&modelmux_bin->lock);
        return FALSE;
      }
      cut_serial = e->branch_serial;         /* instance guard, captured under the lock */
      g_mutex_unlock (&modelmux_bin->lock);
      if (act == MM_ACT_MIGRATE)
        return modelmux_hitless_cutover_begin (modelmux_bin, "gpu-migrate", stream_id, cut_serial,
            shadow, inq_src, NULL, new_gpu, via_default, fu);
      return modelmux_hitless_cutover_begin (modelmux_bin, "swap", stream_id, cut_serial,
          shadow, inq_src, new_model, new_gpu, via_default, fu);
    }

    case MM_ACT_CLEAR:
      /* CLEAR is a role TEARDOWN (no make-before-break target), so it keeps its own IDLE
       * probe: stop the tee, unlink, release the out_mux slot, free the queues. */
      g_mutex_lock (&modelmux_bin->lock);
      e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
      ra = e ? (shadow ? &e->shad : &e->prim) : NULL;
      if (ra && ra->in_q)
        inq_src = gst_element_get_static_pad (ra->in_q, "src");
      if (!inq_src) {
        if (ra)
          ra->busy = FALSE;
        g_mutex_unlock (&modelmux_bin->lock);
        return FALSE;
      }
      g_mutex_unlock (&modelmux_bin->lock);
      ctx = g_new0 (ModelMuxRoleOpCtx, 1);
      ctx->modelmux_bin = modelmux_bin;
      ctx->stream_id = stream_id;
      ctx->shadow = shadow;
      ctx->new_model = NULL;
      if (fu && fu->valid) {
        /* carry the OTHER role's op: launched (on the main loop) only after this
         * CLEAR resolves -- a concurrent async pair on the shared tee stalls the
         * stream, exactly like the SWAP case the cutover's follow-up closes. */
        ctx->fu_valid = TRUE;
        ctx->fu_shadow = fu->shadow;
        ctx->fu_act = (gint) fu->act;
        ctx->fu_model = fu->model ? g_strdup (fu->model) : NULL;
        ctx->fu_gpu = fu->gpu;
        ctx->fu_via_default = fu->via_default;
      }
      /* installed OUTSIDE modelmux_bin->lock: an IDLE probe may fire synchronously here and
       * modelmux_clear_idle_probe re-takes the lock (it re-resolves the stream itself).
       * The NOTIFY owns the ctx (a hook that never fires must not leak it). */
      gst_pad_add_probe (inq_src, GST_PAD_PROBE_TYPE_IDLE, modelmux_clear_idle_probe,
          ctx, modelmux_role_op_ctx_notify);
      gst_object_unref (inq_src);
      return TRUE;

    default:
      return FALSE;
  }
}

/* Launch a cutover's deferred follow-up (the OTHER role's op on this stream) once the
 * cutover has fully ended. Runs on the main loop (called from modelmux_hitless_finish / abort,
 * after modelmux_bin->lock is dropped). Re-resolves the stream under the lock -- if it was removed
 * meanwhile the op is moot (its reserved busy died with the entry). @model is borrowed. */
static void
modelmux_hitless_launch_followup (ModelMuxBin * modelmux_bin, guint stream_id,
    gboolean shadow, ModelMuxActKind act, gchar * model, gint gpu, gboolean via_default)
{
  ModelMuxStreamEntry *e;
  ModelMuxRoleAttach *ra;
  ModelPool *rb = shadow ? modelmux_bin->shadow_pool : modelmux_bin->primary_pool;

  if (act == MM_ACT_NONE)
    return;
  g_mutex_lock (&modelmux_bin->lock);
  e = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
  ra = e ? (shadow ? &e->shad : &e->prim) : NULL;
  g_mutex_unlock (&modelmux_bin->lock);
  if (!e || !ra)
    return;                                  /* stream gone -> nothing to serialize */
  /* modelmux_role_launch must run WITHOUT modelmux_bin->lock; the second op now runs ALONE (the first
   * cutover already unblocked + relinked), so it is a plain single-role op. No further
   * follow-up (NULL) -- a stream has at most two roles. (launch re-resolves the
   * stream by id under the lock; the lookup above is only the early-out.) */
  modelmux_role_launch (modelmux_bin, rb, stream_id, shadow, act, model, gpu, via_default, NULL);
}

/* Upgrade a NONE decision to MIGRATE when the route names the SAME model but a
 * DIFFERENT device than the shard currently serving this stream (gpu-addressed
 * reroute, e.g. drain gpu 0). Caller holds modelmux_bin->lock. */
static ModelMuxActKind
modelmux_role_gpu_upgrade (ModelMuxBin * modelmux_bin, guint stream_id, ModelMuxRoleAttach * ra,
    ModelMuxActKind act, const gchar * model, gint gpu)
{
  ModelBin *cur;
  if (act != MM_ACT_NONE || gpu == MM_GPU_ANY || !model || !*model ||
      !ra->active || ra->passthru || !ra->role)
    return act;
  if (ra->busy) {
    /* a cutover/promote handover is in flight on this lane -- the constraint
     * cannot be applied now and there is no retry queue: say so instead of
     * letting the route silently no-op. */
    MM_INFO ("MultiModelBin: stream %u gpu-addressed route (gpu %d) NOT applied "
        "-- lane busy (cutover in flight); re-issue the route once it settles",
        stream_id, gpu);
    return act;
  }
  cur = modelmux_perf_find_bin_for_source (ra->role, stream_id);
  if (cur && modelmux_shard_gpu_effective (modelmux_bin, cur) != gpu)
    return MM_ACT_MIGRATE;
  return act;
}

/* modelmux_bin_update_routing -- THE per-stream reroute (the real routing worker that
 * update_routing_scoped fans out). Relink ONE stream's primary and/or shadow role onto the
 * target model(s), zero-drop.
 *
 *   stream_id        : the stream to reroute.
 *   primary / shadow : target model key per role ("name@version"); ""  = CLEAR the role,
 *                      NULL = leave that role unchanged.
 *   p_gpu / s_gpu    : effective device per role (MM_GPU_ANY = no pin); same model on a
 *                      DIFFERENT device becomes a hitless gpu MIGRATE.
 *   *_via_default    : origin flag per lane (default-followed vs explicitly pinned).
 *   return           : TRUE if it launched a lane op OR committed a bookkeeping pin change
 *                      (both are routing-content changes -> the caller bumps if_revision).
 *
 * DECIDE-then-LAUNCH: under modelmux_bin->lock it looks up the stream, decides the action per
 * role (modelmux_role_decide -> NONE / ENABLE / SWAP / MIGRATE / CLEAR), RESERVES the role
 * (busy=TRUE) so a second update can't stack on it, then RELEASES the lock and runs the heavy
 * pad/lane work lock-free via modelmux_role_launch (make-before-break cutover: builds the target
 * bin and switches the lane, zero-drop -- it does not relabel across pools, so pool==slot and
 * the per-frame role lookup stays lock-free).
 *
 * Two special cases handled inline:
 *   - PASSTHROUGH role: has no lane to swap (decide returns NONE) -> NOT routed here; the
 *     control plane uses the deferred PROMOTE path instead (log INFO when a promote is armed).
 *   - SAME-MODEL request: launches no lane op, but still commits the ORIGIN / placement PIN
 *     (bookkeeping-only) so "route X to its current model" actually pins it and a later default
 *     switch won't sweep a lane the operator explicitly addressed.
 * When BOTH roles change AND the primary needs an async cutover, the shadow op is carried as a
 * FOLLOW-UP so the two never run concurrently (avoids the shared-tee two-role stall). */
gboolean
modelmux_bin_update_routing (ModelMuxBin * modelmux_bin, guint stream_id,
    const gchar * primary, const gchar * shadow, gint p_gpu, gint s_gpu,
    gboolean p_via_default, gboolean s_via_default)
{
  ModelMuxStreamEntry *stream_entry;
  ModelMuxActKind a_prim, a_shad;
  gchar *prim_model = NULL, *shad_model = NULL;
  gboolean bookkept = FALSE;    /* same-model origin/pin commit (no lane op) */

  /* Decide under the lock (and reserve the role with busy=TRUE for SWAP/CLEAR so
   * a second update can't stack on it), then launch lock-free. */
  g_mutex_lock (&modelmux_bin->lock);
  /* --- find the stream. If it was removed since the caller snapshotted the id set,
   * there is nothing to route -> FALSE (not counted as a change). --- */
  stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) stream_id));
  if (!stream_entry) {
    g_mutex_unlock (&modelmux_bin->lock);
    return FALSE;
  }

  /* NOTE: a primary<->shadow flip is handled by the normal make-before-break path
   * below (each role re-routes to the target in its own-role pool). We intentionally
   * do NOT do an in-place cross-pool record swap: that would make a bin's physical
   * pool differ from the stream's logical slot, which in turn would force the
   * per-frame role lookup (overlay/provenance) to consult shared state under a lock
   * on the streaming thread -- a deadlock risk. Keeping pool == slot lets the
   * per-frame role stay lock-free (modelmux_resolve_role). Make-before-break is still
   * fully zero-drop; it just builds the target bin instead of relabeling. */
  /* --- DECIDE the per-role action: NONE (already on that model) / ENABLE (role was off) /
   * SWAP (different model) / CLEAR (drop) -- see modelmux_role_decide. --- */
  a_prim = modelmux_role_decide (&stream_entry->prim, primary);
  a_shad = modelmux_role_decide (&stream_entry->shad, shadow);
  /* same model + an explicit gpu that differs from the serving shard's device -> upgrade the
   * action to a hitless gpu MIGRATE (otherwise a gpu-addressed reroute would no-op). */
  a_prim = modelmux_role_gpu_upgrade (modelmux_bin, stream_id, &stream_entry->prim, a_prim, primary, p_gpu);
  a_shad = modelmux_role_gpu_upgrade (modelmux_bin, stream_id, &stream_entry->shad, a_shad, shadow, s_gpu);
  /* A PASSTHROUGH primary has no model lane to swap/clear -- the role machinery
   * below can't act on it (modelmux_role_decide returns NONE). The control plane routes
   * passthrough streams through the deferred-PROMOTE path instead (see
   * modelmux_bin_passthru_streams / promote_passthru); when that is armed
   * (pending_model tagged) this is EXPECTED -- log INFO, not a warning. */
  if (stream_entry->prim.passthru && primary != NULL) {
    if (stream_entry->prim.pending_model)
      MM_INFO ("MultiModelBin: stream %u is passthrough -- route to '%s' applies "
          "via deferred promotion (pending '%s')", stream_id, primary,
          stream_entry->prim.pending_model);
    else
      MM_WARN ("MultiModelBin: model/update for stream %u primary='%s' has no effect "
          "-- stream is in passthrough (over-cap, model-sharding=false); "
          "detach and re-add the stream to re-evaluate once a model slot is free",
          stream_id, primary ? primary : "<clear>");
  }
  if (stream_entry->shad.passthru && shadow != NULL)
    MM_WARN ("MultiModelBin: model/update for stream %u shadow='%s' has no effect "
        "-- stream is in passthrough (over-cap, model-sharding=false); "
        "detach and re-add the stream to re-evaluate once a model slot is free",
        stream_id, shadow ? shadow : "<clear>");
  /* BOOKKEEPING-ONLY commit: a request naming the model this role ALREADY runs
   * launches no op (decide: NONE) -- but the request's ORIGIN (and an explicit
   * same-device pin) must still land on the lane. Otherwise "route stream X to its
   * current model" never PINS it, and a later default switch sweeps a lane the
   * operator explicitly addressed (see ModelMuxRoleAttach.via_default). Same-model
   * with a DIFFERENT device became MIGRATE above, so gpu here is same/any. Counts
   * as an update ONLY when a value actually changes (idempotent re-binds stay
   * silent and do not bump the routing revision). */
  if (a_prim == MM_ACT_NONE && primary && *primary && stream_entry->prim.active &&
      !stream_entry->prim.passthru && !stream_entry->prim.busy &&
      g_strcmp0 (stream_entry->prim.model, primary) == 0) {
    if (stream_entry->prim.via_default != p_via_default) {
      stream_entry->prim.via_default = p_via_default;
      bookkept = TRUE;
    }
    if (p_gpu != MM_GPU_ANY && stream_entry->prim.req_gpu != p_gpu) {
      stream_entry->prim.req_gpu = p_gpu;
      bookkept = TRUE;
    }
  }
  if (a_shad == MM_ACT_NONE && shadow && *shadow && stream_entry->shad.active &&
      !stream_entry->shad.passthru && !stream_entry->shad.busy &&
      g_strcmp0 (stream_entry->shad.model, shadow) == 0) {
    if (stream_entry->shad.via_default != s_via_default) {
      stream_entry->shad.via_default = s_via_default;
      bookkept = TRUE;
    }
    if (s_gpu != MM_GPU_ANY && stream_entry->shad.req_gpu != s_gpu) {
      stream_entry->shad.req_gpu = s_gpu;
      bookkept = TRUE;
    }
  }
  if (bookkept)
    MM_INFO ("  - MultiModelBin  stream %u routing BOOKKEEPING committed (same model; "
        "origin/placement pinned -- no lane op needed)", stream_id);
  /* --- RESERVE each async-op role (busy=TRUE) so a second update can't stack on it while its
   * cutover is in flight, and snapshot the target model names to use after we drop the lock. --- */
  if (a_prim == MM_ACT_SWAP || a_prim == MM_ACT_MIGRATE || a_prim == MM_ACT_CLEAR)
    stream_entry->prim.busy = TRUE;
  if (a_shad == MM_ACT_SWAP || a_shad == MM_ACT_MIGRATE || a_shad == MM_ACT_CLEAR)
    stream_entry->shad.busy = TRUE;
  if (a_prim == MM_ACT_SWAP || a_prim == MM_ACT_ENABLE)
    prim_model = g_strdup (primary);
  if (a_shad == MM_ACT_SWAP || a_shad == MM_ACT_ENABLE)
    shad_model = g_strdup (shadow);
  g_mutex_unlock (&modelmux_bin->lock);

  /* --- LAUNCH the lane op(s) LOCK-FREE: modelmux_role_launch runs the make-before-break cutover
   * (build the target bin, switch the lane, zero-drop) and returns TRUE if it started one. --- */
  gboolean launched = FALSE;
  if ((a_prim == MM_ACT_SWAP || a_prim == MM_ACT_MIGRATE || a_prim == MM_ACT_CLEAR) &&
      a_shad != MM_ACT_NONE) {
    /* BOTH roles change AND the primary needs an async cutover -> serialize (see below). */
    /* CLEAR included: a primary CLEAR is just as ASYNC (IDLE probe touching the
     * shared tee) as a SWAP -- running the shadow op concurrently is the same
     * two-role stall race the follow-up carry exists to close. */
    /* BOTH roles change and the primary needs an async cutover. Running the shadow op
     * concurrently races on the stream's shared tee and can stall it (the combined
     * two-role-update bug). Serialize: launch the primary cutover carrying the shadow op
     * as a FOLLOW-UP, which the cutover fires on completion -> two sequential single-role
     * ops, never concurrent. (Other combos are already safe: an inline ENABLE on the
     * primary completes before the shadow op is launched; a lone async op has no sibling
     * to race.) */
    ModelMuxFollowup fu = { TRUE, TRUE /*shadow*/, a_shad, shad_model, s_gpu, s_via_default };
    launched = modelmux_role_launch (modelmux_bin, modelmux_bin->primary_pool, stream_id, FALSE,
        a_prim, prim_model, p_gpu, p_via_default, &fu);
    /* shad_model is consumed via the follow-up (cutover_begin dup'd it) -- UNLESS the
     * primary cutover never actually started (launched==FALSE: no in_q pad). In that case
     * the follow-up will never fire, so run the shadow op directly to avoid stranding it
     * (and its reserved busy). */
    if (!launched)
      launched = modelmux_role_launch (modelmux_bin, modelmux_bin->shadow_pool, stream_id, TRUE,
          a_shad, shad_model, s_gpu, s_via_default, NULL);
  } else {
    /* simple case: no cross-role race to serialize -> launch each role independently. */
    launched |= modelmux_role_launch (modelmux_bin, modelmux_bin->primary_pool, stream_id, FALSE,
        a_prim, prim_model, p_gpu, p_via_default, NULL);
    launched |= modelmux_role_launch (modelmux_bin, modelmux_bin->shadow_pool, stream_id, TRUE,
        a_shad, shad_model, s_gpu, s_via_default, NULL);
  }

  g_free (prim_model);
  g_free (shad_model);
  return launched || bookkept;   /* a pin change IS a routing-content change (it
                                  * alters which streams the next default switch
                                  * moves) -> callers bump the revision for it */
}

/* modelmux_bin_update_routing_scoped -- FAN-OUT: apply the per-stream relink
 * (modelmux_bin_update_routing) to a SCOPE of streams and return how many actually changed.
 *
 *   scope :  src_ids[0..n)  -- the streams to relink;  n == 0  means ALL active streams.
 *   models:  route each in-scope stream onto `primary` (+ `shadow`), with the given gpus/origin.
 *   return:  `launched` -- the number of streams that ACTUALLY changed (a per-stream no-op
 *            returns FALSE and is not counted).
 *
 * This is the IN-PLACE, SYNCHRONOUS half of a reroute: it switches streams that ALREADY have a
 * model lane, right now. (Passthrough streams have no lane and are handled separately, via the
 * deferred PROMOTE path in the caller.) The real per-stream work is modelmux_bin_update_routing;
 * this function just iterates the scope and tallies the changes.
 *
 * CONCURRENCY: for n==0 it SNAPSHOTS the stream-id set under the lock, THEN acts lock-free --
 * so we never mutate the hash table while iterating it, and update_routing re-validates each
 * stream under the lock (a concurrent stream remove is safe). */
guint
modelmux_bin_update_routing_scoped (ModelMuxBin * modelmux_bin, const guint * src_ids,
    guint n, const gchar * primary, const gchar * shadow, gint p_gpu, gint s_gpu,
    gboolean p_via_default, gboolean s_via_default)
{
  guint launched = 0, i;

  if (!modelmux_bin)
    return 0;

  if (n == 0) {
    /* WHOLE-PIPELINE scope: snapshot EVERY stream-id under the lock first (so the per-stream
     * relink below can't mutate stream_wiring under our iterator), then act on the copy. */
    GArray *ids = g_array_new (FALSE, FALSE, sizeof (guint));
    GHashTableIter it;
    gpointer k, v;
    g_mutex_lock (&modelmux_bin->lock);
    g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      guint sid = (guint) GPOINTER_TO_INT (k);
      g_array_append_val (ids, sid);
    }
    g_mutex_unlock (&modelmux_bin->lock);
    /* relink each stream; count it only if update_routing actually changed something. */
    for (i = 0; i < ids->len; i++)
      if (modelmux_bin_update_routing (modelmux_bin, g_array_index (ids, guint, i),
              primary, shadow, p_gpu, s_gpu, p_via_default, s_via_default))
        launched++;
    g_array_free (ids, TRUE);
  } else {
    /* EXPLICIT scope: relink each requested source_id in turn (same count-if-changed rule). */
    for (i = 0; i < n; i++)
      if (modelmux_bin_update_routing (modelmux_bin, src_ids[i], primary, shadow,
              p_gpu, s_gpu, p_via_default, s_via_default))
        launched++;
  }
  return launched;                              /* number of streams actually relinked */
}

/* modelmux_bin_passthru_streams -- FILTER + COLLECT: return the source_ids, within the given
 * scope, of every stream whose PRIMARY lane is currently a no-infer PASSTHROUGH branch.
 *
 *   scope :  src_ids[0..n)  -- the streams to check;  n == 0  means the WHOLE pipeline.
 *   out   :  set to a freshly g_new'd array of the matching source_ids (caller g_free's);
 *            left NULL when none match.
 *   return:  the count of matches (== length of *out).
 *
 * WHY: a passthrough stream was attached display-only and has NO model lane, so the in-place
 * routing machinery cannot swap it. The control plane feeds this list to the deferred PROMOTE
 * path (promote_passthru) instead of the in-place reroute.
 *
 * passthru == NO inference right now (always). Two flavours, both returned here: 'stuck' (no
 * resolvable model at attach -- not loaded and no default) or 'pending' (a target WAS chosen but
 * is still WARMING, so it is tagged pending_model and shows "<model> (passthru)"). A reroute
 * PROMOTES either onto the new target -- the stuck one finally gets a model, the pending one is
 * re-tagged from its old target to the new one.
 *
 * CONCURRENCY: the scan runs under modelmux_bin->lock so the list is a coherent snapshot -- but
 * only a POINT-IN-TIME one. The caller re-validates each stream later (promote_passthru no-ops
 * if the stream has since left or was already promoted). *out is sized to the live scope so a
 * route-all can never truncate above a fixed caller-side capacity. */

/* Every wired stream's source_id (caller g_free's *out); returns the count.
 * Used to MATERIALISE a whole-pipeline scope into the concrete fleet it means
 * right now -- the scope-narrowing in the control layer cannot express "all
 * streams EXCEPT these", so an all-scope entry that has to give up one stream is
 * frozen to the live set instead. Hands back (NULL, 0) when nothing is wired. */
guint
modelmux_bin_active_stream_ids (ModelMuxBin * modelmux_bin, guint ** out)
{
  guint cnt = 0, cap;
  GHashTableIter it;
  gpointer k, v;

  if (!modelmux_bin || !out)
    return 0;
  *out = NULL;

  g_mutex_lock (&modelmux_bin->lock);
  cap = g_hash_table_size (modelmux_bin->stream_wiring);
  if (cap)
    *out = g_new (guint, cap);
  g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
  while (g_hash_table_iter_next (&it, &k, &v) && cnt < cap)
    (*out)[cnt++] = (guint) GPOINTER_TO_INT (k);       /* the hash key IS the source_id */
  g_mutex_unlock (&modelmux_bin->lock);

  if (!cnt)
    g_clear_pointer (out, g_free);
  return cnt;
}

guint
modelmux_bin_passthru_streams (ModelMuxBin * modelmux_bin, const guint * src_ids,
    guint n, guint ** out)
{
  guint cnt = 0, cap, i;
  ModelMuxStreamEntry *stream_entry;

  if (!modelmux_bin || !out)
    return 0;
  *out = NULL;

  g_mutex_lock (&modelmux_bin->lock);
  /* size the result to the scope: an explicit list caps at n; a whole-pipeline scan caps at
   * the live stream count -- so a route-all is never truncated. */
  cap = n ? n : g_hash_table_size (modelmux_bin->stream_wiring);
  if (cap)
    *out = g_new (guint, cap);

  if (n == 0) {
    /* WHOLE-PIPELINE scope: walk every wired stream, keep the passthrough ones. */
    GHashTableIter it;
    gpointer k, v;
    g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
    while (g_hash_table_iter_next (&it, &k, &v) && cnt < cap) {
      stream_entry = (ModelMuxStreamEntry *) v;
      if (stream_entry->prim.passthru)                 /* THE test: primary lane is no-infer passthrough */
        (*out)[cnt++] = (guint) GPOINTER_TO_INT (k);   /* the hash key IS the source_id */
    }
  } else {
    /* EXPLICIT scope: look up each requested source_id; keep it only if it exists AND is passthrough. */
    for (i = 0; i < n && cnt < cap; i++) {
      stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) src_ids[i]));
      if (stream_entry && stream_entry->prim.passthru)
        (*out)[cnt++] = src_ids[i];
    }
  }
  g_mutex_unlock (&modelmux_bin->lock);

  /* nothing matched -> hand back (NULL, 0), never an empty-but-allocated array. */
  if (!cnt)
    g_clear_pointer (out, g_free);
  return cnt;                                   /* == number of passthrough streams == length of *out */
}

/* ================================================================== *
 *  IN-PLACE MOVE: promote / swap by RE-LABELING bins, not rebuilding.
 *
 *  When a model/update's scope EXACTLY covers the stream-set of the bin(s) involved
 *  (group-local full coverage) and the A/B is uniform, the change is *total* -- every
 *  stream on those bins moves together -- so we change LABELS (model_bin->role, pool map, the
 *  per-stream prim/shad records) instead of WIRING. The surviving bins' pads, queues and
 *  nvinfer are NEVER touched, so nothing drains and NO frame is dropped; no new bin is
 *  built, no engine reloads, the gie-id is preserved.
 *
 *    swap   ("" B A)    = relabel-swap A<->B (exchange role + pool + records).
 *    promote("" B none) = relabel-swap A<->B, then CLEAR the (now-)shadow A + auto-unload.
 *
 *  The model_bin->role retag is a single atomic store of an interned string (modelmux_role_set), safe
 *  vs the lock-free per-frame reader. frameacct gets a fresh epoch on both role slots
 *  (the model behind each slot changed). If the scope does NOT fully cover the bins
 *  (partial / heterogeneous), this returns FALSE and the caller falls back to the generic
 *  create+warm reroute. NOTE: the display tile keeps following its bin (a promoted model
 *  stays in its old shadow row); attribute results by model_name+role (provenance), not
 *  by tile position or gie. ================================================================== */

/* relabel-swap the current Primary-pool bin `Aname` with the Shadow-pool bin `Bname`
 * for `streams`. Caller holds modelmux_bin->lock and has verified eligibility. Pure bookkeeping:
 * no pad/queue/element is touched, so it cannot drop a frame. Returns FALSE if either
 * bin is not where eligibility said it was (nothing was touched) -- the caller MUST
 * abort the in-place op, or it would reassign defaults and clear/unload lanes for a
 * relabel that never happened. */
static gboolean
modelmux_inplace_relabel_swap (ModelMuxBin * modelmux_bin, const gchar * Aname,
    const gchar * Bname, const guint * streams, guint nstreams)
{
  ModelBin *Abin = g_hash_table_lookup (modelmux_bin->primary_pool->models, Aname);
  ModelBin *Bbin = g_hash_table_lookup (modelmux_bin->shadow_pool->models, Bname);
  ModelBin *model_bin;
  guint i;
  if (!Abin || !Bbin) {
    MM_ERR ("MultiModelBin: IN-PLACE relabel-swap ABORTED -- '%s'/'%s' vanished from "
        "their pools between eligibility and relabel", Aname, Bname);
    return FALSE;
  }

  MM_INFO ("  - MultiModelBin  IN-PLACE relabel-swap: Primary '%s'(gie=%u) <-> Shadow "
      "'%s'(gie=%u)  [reuse bins, gie kept, no reload, no pad touch]", Aname,
      Abin->unique_id, Bname, Bbin->unique_id);

  /* retag whole shard chains (atomic role flip; safe vs the lock-free per-frame reader) */
  for (model_bin = Abin; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
    modelmux_role_set (model_bin, "Shadow"); model_bin->pool = modelmux_bin->shadow_pool;
  }
  for (model_bin = Bbin; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
    modelmux_role_set (model_bin, "Primary"); model_bin->pool = modelmux_bin->primary_pool;
  }

  /* move pool map entries (the table owns the key -> remove frees the old key) */
  g_hash_table_remove (modelmux_bin->primary_pool->models, Aname);
  g_hash_table_remove (modelmux_bin->shadow_pool->models, Bname);
  g_hash_table_insert (modelmux_bin->primary_pool->models, g_strdup (Bname), Bbin);
  g_hash_table_insert (modelmux_bin->shadow_pool->models, g_strdup (Aname), Abin);

  /* per stream: swap the prim<->shad branch records, fix the pool back-ptr, re-learn acct */
  for (i = 0; i < nstreams; i++) {
    ModelMuxStreamEntry *stream_entry =
        g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) streams[i]));
    ModelMuxRoleAttach tmp;
    if (!stream_entry)
      continue;
    tmp = stream_entry->prim; stream_entry->prim = stream_entry->shad; stream_entry->shad = tmp;   /* B branch -> primary slot */
    stream_entry->prim.role = modelmux_bin->primary_pool;
    stream_entry->shad.role = modelmux_bin->shadow_pool;
    /* Re-learn only the steady-depth floor (the model behind each slot changed). We do
     * NOT re-base the in/infer/out counters: those live on 3 different threads, so a
     * fresh epoch would straddle in-flight frames and read buffered<0 ("ovlp"). The
     * counters stay continuous (entered/delivered keep tracking 1:1) which is exactly
     * what zero-drop needs; only the floor is relearned to fit the new model. */
    modelmux_acct_role_rewarm (modelmux_bin, streams[i], MM_ROLE_PRIMARY);
    modelmux_acct_role_rewarm (modelmux_bin, streams[i], MM_ROLE_SHADOW);
    MM_INFO ("    - stream %u  IN-PLACE  primary='%s'  shadow='%s'", streams[i],
        stream_entry->prim.model ? stream_entry->prim.model : "-", stream_entry->shad.model ? stream_entry->shad.model : "-");
  }
  return TRUE;
}

/* Per-stream display-row CROSS for an in-place SWAP: primary's output goes to the TOP
 * mux slot, shadow's to the BOTTOM, even though both rows were occupied. Both out_q src
 * pads are held with BLOCK probes; once BOTH are blocked the cross runs on the main loop
 * (release both mux sinks, re-request them crossed, relink) then removes the probes to
 * unblock. Zero-drop: each out_q buffers across the brief block. */
/* Bounded wait for BOTH out_q src pads to block. If one branch is idle/EOS/stalled it never
 * produces a buffer to engage its BLOCK probe, so without this the other pad would stay
 * blocked forever (permanent stall + leaked ctx). Sized well above a slow shadow's frame
 * interval so a legitimately-active-but-slow branch is not aborted prematurely. */
#define MM_SWAP_DISP_TIMEOUT_MS  1000

typedef struct
{
  ModelMuxBin *modelmux_bin;
  guint stream_id;
  GstPad *prim_src, *shad_src;   /* out_q src pads (refs held) */
  gulong prim_pid, shad_pid;
  gint prim_blocked, shad_blocked;
  gint scheduled;                /* one-shot owner guard: whoever wins resolves (cross OR abort) */
  guint timeout_id;              /* safety g_timeout (0=none); cancelled by the cross */
  gint refs;                     /* 1 owner ref + 1 per installed BLOCK probe. The probe
                                  * refs are dropped by the probes' GDestroyNotify, which
                                  * GStreamer defers past an IN-CALL callback -- so the
                                  * resolving path (cross/timeout/drop) can never free the
                                  * ctx under a concurrently-executing block callback. */
} ModelMuxSwapDispCtx;

static void
modelmux_swap_disp_ctx_unref (gpointer data)
{
  ModelMuxSwapDispCtx *c = (ModelMuxSwapDispCtx *) data;
  if (c && g_atomic_int_dec_and_test (&c->refs))
    g_free (c);
}

/* Resolve-and-release (the OWNER ref): cancel the safety timeout, remove the
 * probes (their notifies drop the probe refs -- deferred if a callback is
 * mid-flight), drop the pad refs, then drop the owner ref. */
static void
modelmux_swap_disp_ctx_drop (gpointer data)
{
  ModelMuxSwapDispCtx *c = (ModelMuxSwapDispCtx *) data;
  if (!c)
    return;
  if (c->timeout_id)
    modelmux_tracked_source_cancel (c->modelmux_bin, &c->timeout_id, FALSE);
  if (c->prim_src) {
    if (c->prim_pid)
      gst_pad_remove_probe (c->prim_src, c->prim_pid);
    gst_object_unref (c->prim_src);
  }
  if (c->shad_src) {
    if (c->shad_pid)
      gst_pad_remove_probe (c->shad_src, c->shad_pid);
    gst_object_unref (c->shad_src);
  }
  modelmux_swap_disp_ctx_unref (c);
}

static gboolean
modelmux_swap_disp_cross (gpointer udata)
{
  ModelMuxSwapDispCtx *c = (ModelMuxSwapDispCtx *) udata;
  ModelMuxBin *modelmux_bin = c->modelmux_bin;
  ModelMuxStreamEntry *stream_entry;

  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    modelmux_swap_disp_ctx_drop (c);
    return G_SOURCE_REMOVE;
  }

  g_mutex_lock (&modelmux_bin->lock);
  stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) c->stream_id));
  if (stream_entry && stream_entry->prim.active && stream_entry->shad.active &&
      stream_entry->prim.out_mux_sink && stream_entry->shad.out_mux_sink) {
    GstPad *bp = gst_pad_get_peer (stream_entry->prim.out_mux_sink);   /* primary out_q.src */
    GstPad *ap = gst_pad_get_peer (stream_entry->shad.out_mux_sink);   /* shadow  out_q.src */
    gint top = stream_entry->column;
    gint bot = (gint) modelmux_bin->max_streams + stream_entry->column;
    GstPad *ms_top, *ms_bot;
    gchar pn[64];
    if (bp) gst_pad_unlink (bp, stream_entry->prim.out_mux_sink);
    if (ap) gst_pad_unlink (ap, stream_entry->shad.out_mux_sink);
    gst_element_release_request_pad (modelmux_bin->out_mux, stream_entry->prim.out_mux_sink);
    gst_object_unref (stream_entry->prim.out_mux_sink);   /* release caller's ref from request_pad */
    gst_element_release_request_pad (modelmux_bin->out_mux, stream_entry->shad.out_mux_sink);
    gst_object_unref (stream_entry->shad.out_mux_sink);   /* release caller's ref from request_pad */
    g_snprintf (pn, sizeof (pn), "sink_%u", top);
    ms_top = gst_element_request_pad_simple (modelmux_bin->out_mux, pn);
    g_snprintf (pn, sizeof (pn), "sink_%u", bot);
    ms_bot = gst_element_request_pad_simple (modelmux_bin->out_mux, pn);
    if (!ms_top || !ms_bot)
      MM_ERR ("MultiModelBin: swap-disp stream %u -- request_pad failed "
          "(ms_top=%p ms_bot=%p); stream display rows may be disconnected",
          c->stream_id, (void *) ms_top, (void *) ms_bot);
    if (bp && ms_top) modelmux_link_pads (bp, ms_top);            /* primary -> top row */
    if (ap && ms_bot) modelmux_link_pads (ap, ms_bot);            /* shadow  -> bottom  */
    stream_entry->prim.out_mux_sink = ms_top; stream_entry->prim.out_idx = top;
    stream_entry->shad.out_mux_sink = ms_bot; stream_entry->shad.out_idx = bot;
    if (bp) gst_object_unref (bp);
    if (ap) gst_object_unref (ap);
    MM_INFO ("  - MultiModelBin  stream %u SWAP display rows fixed: primary->top(%d), "
        "shadow->bottom(%d) (zero-drop)", c->stream_id, top, bot);
  }
  g_mutex_unlock (&modelmux_bin->lock);

  /* cancel the safety timeout (still pending iff it hasn't fired -- both run on the main loop,
   * so if it already fired it set timeout_id=0 and the tracked cancel is a no-op). */
  if (c->timeout_id)
    modelmux_tracked_source_cancel (modelmux_bin, &c->timeout_id, FALSE);
  if (c->prim_src) { gst_pad_remove_probe (c->prim_src, c->prim_pid); gst_object_unref (c->prim_src); }
  if (c->shad_src) { gst_pad_remove_probe (c->shad_src, c->shad_pid); gst_object_unref (c->shad_src); }
  modelmux_swap_disp_ctx_unref (c);            /* owner ref; probe refs dropped by their notifies */
  return G_SOURCE_REMOVE;
}

static GstPadProbeReturn
modelmux_swap_disp_block (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSwapDispCtx *c = (ModelMuxSwapDispCtx *) udata;
  gint *flag = (pad == c->prim_src) ? &c->prim_blocked : &c->shad_blocked;
  (void) info;
  if (g_atomic_int_get (&c->modelmux_bin->shutting_down))
    return GST_PAD_PROBE_OK;
  g_atomic_int_compare_and_exchange (flag, 0, 1);           /* mark this pad blocked once */
  if (g_atomic_int_get (&c->prim_blocked) && g_atomic_int_get (&c->shad_blocked) &&
      g_atomic_int_compare_and_exchange (&c->scheduled, 0, 1)) { /* exactly one scheduler */
    modelmux_tracked_source_cancel (c->modelmux_bin, &c->timeout_id, FALSE);
    if (!modelmux_tracked_idle_add (c->modelmux_bin, modelmux_swap_disp_cross, c,
            modelmux_swap_disp_ctx_drop, NULL)) {
      if (pad == c->prim_src)
        c->prim_pid = 0;                  /* this callback's return removes it */
      else
        c->shad_pid = 0;
      modelmux_swap_disp_ctx_drop (c);
      return GST_PAD_PROBE_REMOVE;
    }
  }
  return GST_PAD_PROBE_OK;                                  /* stay blocked until removed */
}

/* SAFETY NET (main loop): one branch never blocked within the window -> the cross can never be
 * scheduled and the OTHER pad would stay blocked forever. Take ownership via the same `scheduled`
 * CAS (so the both-blocked path can no longer schedule the cross), then ABORT: remove the probes
 * (unblocking) and free the ctx. The display-row cross is skipped -- purely cosmetic (which tile
 * row each shows on); frames keep flowing with NO drop and NO stall. If the both-blocked path
 * already won the CAS, the cross owns teardown, so we only clear our id and bow out. */
static gboolean
modelmux_swap_disp_timeout (gpointer udata)
{
  ModelMuxSwapDispCtx *c = (ModelMuxSwapDispCtx *) udata;
  c->timeout_id = 0;                                  /* this source auto-removed on return */
  if (g_atomic_int_get (&c->modelmux_bin->shutting_down)) {
    modelmux_swap_disp_ctx_drop (c);
    return G_SOURCE_REMOVE;
  }
  if (!g_atomic_int_compare_and_exchange (&c->scheduled, 0, 1))
    return G_SOURCE_REMOVE;                           /* cross already owns it -> it will free c */
  MM_WARN ("MultiModelBin: swap-disp stream %u timed out waiting for both rows to block "
      "-- aborting display cross (rows left as-is; no drop, no stall)", c->stream_id);
  if (c->prim_src) { gst_pad_remove_probe (c->prim_src, c->prim_pid); gst_object_unref (c->prim_src); }
  if (c->shad_src) { gst_pad_remove_probe (c->shad_src, c->shad_pid); gst_object_unref (c->shad_src); }
  modelmux_swap_disp_ctx_unref (c);            /* owner ref; a block callback that is mid-flight
                                          * right now still holds its probe ref -> no UAF */
  return G_SOURCE_REMOVE;
}

static void
modelmux_inplace_fix_swap_display (ModelMuxBin * modelmux_bin, const guint * streams, guint n)
{
  guint i;
  for (i = 0; i < n; i++) {
    GstPad *psrc = NULL, *ssrc = NULL;
    ModelMuxStreamEntry *stream_entry;
    ModelMuxSwapDispCtx *c;
    g_mutex_lock (&modelmux_bin->lock);
    stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) streams[i]));
    if (stream_entry && stream_entry->prim.active && stream_entry->shad.active && stream_entry->prim.out_q && stream_entry->shad.out_q &&
        stream_entry->prim.out_idx != stream_entry->column) {        /* primary not already on the top row */
      psrc = gst_element_get_static_pad (stream_entry->prim.out_q, "src");
      ssrc = gst_element_get_static_pad (stream_entry->shad.out_q, "src");
    }
    g_mutex_unlock (&modelmux_bin->lock);
    if (!psrc || !ssrc) {
      if (psrc) gst_object_unref (psrc);
      if (ssrc) gst_object_unref (ssrc);
      continue;
    }
    c = g_new0 (ModelMuxSwapDispCtx, 1);
    c->modelmux_bin = modelmux_bin;
    c->stream_id = streams[i];
    c->prim_src = psrc;                          /* refs transferred to ctx */
    c->shad_src = ssrc;
    c->refs = 1;                                 /* owner ref (resolved by cross/timeout/drop) */
    /* one ref per probe, dropped by the probe's GDestroyNotify -- which GStreamer
     * runs exactly once per installed hook (including the id==0 self-removed case),
     * and never while the callback is IN-CALL on another thread. */
    g_atomic_int_inc (&c->refs);
    c->prim_pid = gst_pad_add_probe (psrc, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
        modelmux_swap_disp_block, c, modelmux_swap_disp_ctx_unref);
    g_atomic_int_inc (&c->refs);
    c->shad_pid = gst_pad_add_probe (ssrc, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
        modelmux_swap_disp_block, c, modelmux_swap_disp_ctx_unref);
    /* arm the safety net AFTER both probes are installed (both this fn and the timeout/cross
     * run on the main loop, so the timeout cannot fire until we return). */
    if (!modelmux_tracked_timeout_add (modelmux_bin, MM_SWAP_DISP_TIMEOUT_MS, modelmux_swap_disp_timeout,
            c, modelmux_swap_disp_ctx_drop, &c->timeout_id))
      modelmux_swap_disp_ctx_drop (c);
  }
}

gboolean
modelmux_bin_try_inplace (ModelMuxBin * modelmux_bin, const guint * src_ids, guint n,
    const gchar * primary, const gchar * shadow)
{
  gboolean scope_all = (n == 0), is_promote, is_swap, eligible = TRUE;
  const gchar *P0 = NULL, *S0 = NULL;
  gchar *Aname = NULL, *Bname = NULL;
  GArray *streams;
  GHashTableIter it;
  gpointer k, v;
  guint i;

  if (!modelmux_bin || !primary || !*primary)
    return FALSE;
  is_promote = (shadow && g_strcmp0 (shadow, "none") == 0);
  is_swap = (shadow && *shadow && g_strcmp0 (shadow, "none") != 0);
  if (!is_promote && !is_swap)
    return FALSE;                                /* plain primary reroute -> generic path */

  g_mutex_lock (&modelmux_bin->lock);
  streams = g_array_new (FALSE, FALSE, sizeof (guint));

  /* pass 1: every SCOPED stream must be uniform (same primary P0 + shadow S0), both
   * roles active and not mid-op. */
  g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
  while (eligible && g_hash_table_iter_next (&it, &k, &v)) {
    ModelMuxStreamEntry *stream_entry = (ModelMuxStreamEntry *) v;
    guint sid = stream_entry->stream_id;
    gboolean in_scope = scope_all;
    if (!in_scope)
      for (i = 0; i < n; i++) if (src_ids[i] == sid) { in_scope = TRUE; break; }
    if (!in_scope)
      continue;
    if (!stream_entry->prim.active || !stream_entry->shad.active || stream_entry->prim.busy || stream_entry->shad.busy) {
      eligible = FALSE; break;
    }
    if (!P0) { P0 = stream_entry->prim.model; S0 = stream_entry->shad.model; }
    else if (g_strcmp0 (stream_entry->prim.model, P0) || g_strcmp0 (stream_entry->shad.model, S0)) {
      eligible = FALSE; break;
    }
    g_array_append_val (streams, sid);
  }
  if (eligible && (streams->len == 0 || !P0 || !S0))
    eligible = FALSE;

  /* the named op must be the exact in-place shape:
   *   promote -> primary == current shadow (S0)
   *   swap    -> primary == S0 AND shadow == current primary (P0) (a clean flip) */
  if (eligible) {
    if (is_promote && g_strcmp0 (primary, S0) != 0)
      eligible = FALSE;
    if (is_swap && (g_strcmp0 (primary, S0) != 0 || g_strcmp0 (shadow, P0) != 0))
      eligible = FALSE;
  }

  /* pass 2: group-local FULL COVERAGE -- no OUT-of-scope stream may use the P0 primary
   * bin or the S0 shadow bin, else relabeling the whole bin would strand it. */
  if (eligible) {
    g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelMuxStreamEntry *stream_entry = (ModelMuxStreamEntry *) v;
      guint sid = stream_entry->stream_id;
      gboolean in_scope = scope_all;
      if (!in_scope)
        for (i = 0; i < n; i++) if (src_ids[i] == sid) { in_scope = TRUE; break; }
      if (in_scope)
        continue;
      if ((stream_entry->prim.active && g_strcmp0 (stream_entry->prim.model, P0) == 0) ||
          (stream_entry->shad.active && g_strcmp0 (stream_entry->shad.model, S0) == 0)) {
        eligible = FALSE; break;
      }
    }
  }

  if (!eligible) {
    g_mutex_unlock (&modelmux_bin->lock);
    g_array_free (streams, TRUE);
    MM_INFO ("  - MultiModelBin  IN-PLACE not eligible (partial/heterogeneous scope or "
        "not a clean promote/swap) -> generic create+warm reroute");
    return FALSE;
  }

  Aname = g_strdup (P0);                          /* dup before records get swapped */
  Bname = g_strdup (S0);
  if (!modelmux_inplace_relabel_swap (modelmux_bin, Aname, Bname, (guint *) streams->data,
          streams->len)) {
    /* nothing was relabeled: fall back to the generic create+warm reroute -- do NOT
     * reassign defaults / reclaim / clear lanes for an op that never happened. */
    g_mutex_unlock (&modelmux_bin->lock);
    g_array_free (streams, TRUE);
    g_free (Aname);
    g_free (Bname);
    return FALSE;
  }

  /* DEFAULT re-assignment (scope-ALL, in-place only). Done here under modelmux_bin->lock (the config default
   * pointers are read on streaming threads).
   *   PROMOTE: the promoted model B is now THE primary for every stream -> it becomes the new
   *            default_primary (the intuitive baseline), and the shadow role is empty -> the
   *            default_shadow is cleared. This is UNCONDITIONAL: B always becomes the default
   *            primary, and ANY prior default_primary is revoked (even one not part of this op);
   *            the displaced old primary A, no longer a default, is reclaimed by the clear path
   *            below. (No parked defaults: the default always reflects what every stream runs.)
   *   SWAP:    the default status follows the ROLE/SLOT, not the model -- whoever lands in a slot
   *            that WAS a default's slot acquires that default status; the model moving out gives
   *            it up. Surgically: default_primary:=B iff A (old primary) was the default_primary
   *            (its model just left the primary role); default_shadow:=A iff B (old shadow) was the
   *            default_shadow (its model just left the shadow role); otherwise each is unchanged.
   *            This yields: both-default => clean exchange; def_p/custom => custom acquires
   *            default_primary, default_shadow untouched; custom/def_s => custom acquires
   *            default_shadow, default_primary untouched; custom/custom => defaults untouched.
   *            Nothing is reclaimed -- both swapped models keep serving; a demoted default simply
   *            loses its flag (and its idle-reclaim exemption) once it later drains to 0 streams. */
  if (scope_all && is_promote) {
    /* we HOLD modelmux_bin->lock here -- mutate the pool refs directly
     * (modelmux_bin_set_default self-locks and would deadlock). */
    DefaultModelRef *default_primary = &modelmux_bin->primary_pool->default_ref;
    DefaultModelRef *default_shadow = &modelmux_bin->shadow_pool->default_ref;
    /* capture the prior defaults (composite keys) before we overwrite them. */
    gchar *old_dp = g_strdup (default_primary->key);
    gchar *dsk = g_strdup (default_shadow->key);
    gboolean s_was_def_shadow = (dsk && g_strcmp0 (Bname, dsk) == 0);
    gchar *bn = NULL, *bv = NULL;

    /* (1) the promoted model B becomes the new default_primary (the new champion).
     *     The OLD default's placement pin dies with its identity: keeping it would
     *     hard-constrain every future default-resolved stream to a device the NEW
     *     default may have no instance on (attach refusals fleet-wide). */
    model_key_split (Bname, &bn, &bv);
    mm_default_ref_assign (default_primary, bn, bv, MM_GPU_ANY,
        mm_default_bin_lookup (modelmux_bin, FALSE, Bname));
    default_primary->gpu = MM_GPU_ANY;   /* UNCONDITIONAL reset -- even a same-name promote drops the pin */
    g_free (bn);            /* assign strdups; the split parts are ours to free */
    g_free (bv);

    /* (2) clear default_shadow ONLY if the promoted model WAS the default_shadow (the crowned
     *     challenger). An UNINVOLVED, sideline default_shadow is left untouched (kept warm). */
    if (s_was_def_shadow)
      mm_default_ref_clear (default_shadow);
    MM_INFO ("  - MultiModelBin  DEFAULT promote: promoted '%s' is now default_primary%s", Bname,
        s_was_def_shadow ? "; default_shadow cleared (it was the crowned challenger)"
                         : " (default_shadow unchanged)");

    /* every scoped stream now runs the NEW baseline: mark their primary lanes
     * via_default so a FUTURE default switch sweeps them (the promote's "the
     * default always reflects what every stream runs" semantics). Still under
     * modelmux_bin->lock. */
    for (i = 0; i < streams->len; i++) {
      ModelMuxStreamEntry *pe = g_hash_table_lookup (modelmux_bin->stream_wiring,
          GINT_TO_POINTER ((gint) g_array_index (streams, guint, i)));
      if (pe)
        pe->prim.via_default = TRUE;
    }

    /* (3) reclaim a SUPERSEDED, ORPHANED default_primary: the prior default_primary that is NOT
     *     the model just promoted AND NOT the displaced old primary A (the clear path below
     *     reclaims A). This frees a sideline default that nothing references anymore (e.g. a
     *     configured default that was idle while custom models served the streams). Idle-only.
     *     Also skip it when it is STILL the active default_shadow (dsk): a runtime set_default
     *     can point both roles at the same model (config forbids it, the API does not), and that
     *     model must not be unloaded out from under the default_shadow role. (g_strcmp0 is
     *     NULL-safe, so a NULL dsk -- no default_shadow -- never blocks the reclaim.) */
    if (old_dp && g_strcmp0 (old_dp, Bname) != 0 && g_strcmp0 (old_dp, Aname) != 0 &&
        g_strcmp0 (old_dp, dsk) != 0) {
      ModelPool *op = modelmux_bin->primary_pool;
      ModelBin *ob = g_hash_table_lookup (op->models, old_dp);
      if (!ob) { op = modelmux_bin->shadow_pool; ob = g_hash_table_lookup (op->models, old_dp); }
      if (ob && model_bin_num_streams (ob) == 0 && !ob->next_shard) {
        MM_INFO ("  - MultiModelBin  superseded default_primary '%s' now orphaned + idle -> "
            "reclaimed (free VRAM)", old_dp);
        model_pool_unload_model (op, old_dp);
      } else if (!ob && modelmux_bin->limbo_models) {
        /* a superseded default may instead be parked in LIMBO (role-agnostic, loaded but
         * unpromoted): reclaim it too when idle. We already hold modelmux_bin->lock here, so free inline
         * (mirrors the limbo branch of modelmux_bin_unload_model) -- calling that
         * self-locking helper from under the lock would deadlock. */
        ModelBin *lb = g_hash_table_lookup (modelmux_bin->limbo_models, old_dp);
        if (lb && model_bin_num_streams (lb) == 0 && !lb->next_shard) {
          MM_INFO ("  - MultiModelBin  superseded default_primary '%s' orphaned + idle in limbo -> "
              "reclaimed (free VRAM)", old_dp);
          modelmux_bin_forget_default_bin (modelmux_bin, lb);
          g_hash_table_remove (modelmux_bin->limbo_models, old_dp);   /* frees the key only */
          model_bin_free (lb);
        }
      }
    }
    g_free (old_dp);
    g_free (dsk);
  } else if (scope_all && is_swap) {
    /* we HOLD modelmux_bin->lock here -- mutate the pool refs directly. */
    DefaultModelRef *default_primary = &modelmux_bin->primary_pool->default_ref;
    DefaultModelRef *default_shadow = &modelmux_bin->shadow_pool->default_ref;
    /* A = old primary (now the new shadow), B = old shadow (now the new primary). */
    gboolean a_was_def_primary = (default_primary->key && g_strcmp0 (Aname, default_primary->key) == 0);
    gboolean b_was_def_shadow  = (default_shadow->key && g_strcmp0 (Bname, default_shadow->key) == 0);

    /* (1) default_primary follows the PRIMARY slot: if the model leaving primary (A) was the
     *     default_primary, the model now in primary (B) acquires that status. */
    if (a_was_def_primary) {
      gchar *bn = NULL, *bv = NULL;
      model_key_split (Bname, &bn, &bv);
      mm_default_ref_assign (default_primary, bn, bv, MM_GPU_ANY,
          mm_default_bin_lookup (modelmux_bin, FALSE, Bname));
      default_primary->gpu = MM_GPU_ANY;   /* pin belonged to the OLD identity */
      g_free (bn);
      g_free (bv);
    }
    /* (2) default_shadow follows the SHADOW slot: if the model leaving shadow (B) was the
     *     default_shadow, the model now in shadow (A) acquires that status. */
    if (b_was_def_shadow) {
      gchar *an = NULL, *av = NULL;
      model_key_split (Aname, &an, &av);
      mm_default_ref_assign (default_shadow, an, av, MM_GPU_ANY,
          mm_default_bin_lookup (modelmux_bin, TRUE, Aname));
      default_shadow->gpu = MM_GPU_ANY;   /* pin belonged to the OLD identity */
      g_free (an);
      g_free (av);
    }
    /* lane ORIGIN repair: relabel_swap exchanged the WHOLE RoleAttach records, so
     * each lane inherited the OTHER role's via_default. Status follows the slot
     * (above) -- so must the origin: a lane now running its slot's default FOLLOWS
     * it; one running a non-default is PINNED (it was explicitly swapped there). */
    for (i = 0; i < streams->len; i++) {
      ModelMuxStreamEntry *pe = g_hash_table_lookup (modelmux_bin->stream_wiring,
          GINT_TO_POINTER ((gint) g_array_index (streams, guint, i)));
      if (pe) {
        pe->prim.via_default = a_was_def_primary;
        pe->shad.via_default = b_was_def_shadow;
      }
    }
    if (a_was_def_primary || b_was_def_shadow)
      MM_INFO ("  - MultiModelBin  DEFAULT swap: %s%s%s -- status follows the slot",
          a_was_def_primary ? "default_primary acquired by new primary" : "",
          (a_was_def_primary && b_was_def_shadow) ? "; " : "",
          b_was_def_shadow ? "default_shadow acquired by new shadow" : "");
    else
      MM_INFO ("  - MultiModelBin  DEFAULT swap: neither swapped model was a default -- "
          "defaults untouched");
  } else if (!scope_all) {
    /* group-local (scoped) promote/swap: an EXPLICIT op named these models for these
     * streams, but relabel_swap left each lane with the SIBLING role's stale origin.
     * Both lanes are now explicitly placed => PINNED (defaults are untouched by
     * scoped ops; a promote's shadow lane is cleared right after -- harmless). */
    for (i = 0; i < streams->len; i++) {
      ModelMuxStreamEntry *pe = g_hash_table_lookup (modelmux_bin->stream_wiring,
          GINT_TO_POINTER ((gint) g_array_index (streams, guint, i)));
      if (pe) {
        pe->prim.via_default = FALSE;
        pe->shad.via_default = FALSE;
      }
    }
  }

  /* promote: after the swap the old primary A sits in the SHADOW slot -> reserve it busy
   * so we can clear+auto-unload it (it served only these streams). */
  if (is_promote)
    for (i = 0; i < streams->len; i++) {
      ModelMuxStreamEntry *stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring,
          GINT_TO_POINTER ((gint) g_array_index (streams, guint, i)));
      if (stream_entry) stream_entry->shad.busy = TRUE;
    }
  g_mutex_unlock (&modelmux_bin->lock);

  MM_INFO ("  model/update %s (IN-PLACE) on %u stream(s): primary -> '%s'%s "
      "-- reused existing bins, gie kept, NO reload, zero frame drop",
      is_promote ? "PROMOTE" : "SWAP", streams->len, Bname,
      is_promote ? ", shadow dropped" : "");

  /* promote only: clear the demoted old primary (now shadow) and auto-unload it */
  if (is_promote) {
    for (i = 0; i < streams->len; i++) {
      guint sid = g_array_index (streams, guint, i);
      ModelMuxStreamEntry *stream_entry;
      GstPad *inq_src;
      ModelMuxRoleOpCtx *ctx;
      /* re-resolve UNDER the lock: a concurrent detach-stream can free the entry
       * between the swap above and this loop (modelmux_bin->stream_wiring and entries must never
       * be read unlocked). Only the probe install runs outside (IDLE probes may
       * fire synchronously and re-take modelmux_bin->lock). */
      g_mutex_lock (&modelmux_bin->lock);
      stream_entry = g_hash_table_lookup (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) sid));
      inq_src = (stream_entry && stream_entry->shad.in_q) ?
          gst_element_get_static_pad (stream_entry->shad.in_q, "src") : NULL;
      if (!inq_src) {
        if (stream_entry) stream_entry->shad.busy = FALSE;
        g_mutex_unlock (&modelmux_bin->lock);
        continue;
      }
      g_mutex_unlock (&modelmux_bin->lock);
      ctx = g_new0 (ModelMuxRoleOpCtx, 1);
      ctx->modelmux_bin = modelmux_bin;
      ctx->stream_id = sid;
      ctx->shadow = TRUE;
      ctx->new_model = NULL;                      /* clear */
      ctx->unload_if_idle = TRUE;                 /* drop the demoted old primary */
      ctx->promote_top_row = TRUE;                /* move promoted primary to the top row */
      gst_pad_add_probe (inq_src, GST_PAD_PROBE_TYPE_IDLE, modelmux_clear_idle_probe, ctx,
          modelmux_role_op_ctx_notify);   /* notify owns the ctx */
      gst_object_unref (inq_src);
    }
    MM_INFO ("  - MultiModelBin  IN-PLACE promote: dropping old primary '%s' (clear "
        "shadow + auto-unload when idle)", Aname);
  } else {
    /* swap only: both bins stay, so both display rows are occupied -> cross the two
     * combined-mux slots per stream so primary stays on the TOP row, shadow on the
     * bottom. Two-pad coordinated block -> zero drop. */
    modelmux_inplace_fix_swap_display (modelmux_bin, (guint *) streams->data, streams->len);
  }

  g_free (Aname);
  g_free (Bname);
  g_array_free (streams, TRUE);
  return TRUE;
}

void
modelmux_bin_free (ModelMuxBin * modelmux_bin)
{
  if (!modelmux_bin)
    return;
  g_atomic_int_set (&modelmux_bin->shutting_down, 1);
  modelmux_tracked_sources_cancel_all (modelmux_bin);
  /* Tear down any streams still attached at finalize (e.g. abrupt shutdown without
   * draining first). The hash held only borrowed ModelMuxStreamEntry* with no value-destroy, so
   * destroying it would leak every entry + its model strings + per-role pad refs. Reuse
   * the real per-stream teardown (no logic divergence) over a SNAPSHOT of the ids, since
   * each call removes its own entry from the hash. */
  if (modelmux_bin->stream_wiring) {
    GList *ids = g_hash_table_get_keys (modelmux_bin->stream_wiring), *l;
    for (l = ids; l; l = l->next)
      modelmux_bin_detach_stream (modelmux_bin, (guint) GPOINTER_TO_INT (l->data));
    g_list_free (ids);
  }
  /* The detaches above may have re-armed a debounced compaction pass; cancel it (and any
   * pre-existing one) so no g_timeout fires against the freed bin. compact_poll_id is
   * guarded by free_lock (schedulers can run off the main loop via action signals). */
  g_mutex_lock (&modelmux_bin->free_lock);
  if (modelmux_bin->compact_poll_id) {                 /* cancel any pending compaction pass */
    g_source_remove (modelmux_bin->compact_poll_id);
    modelmux_bin->compact_poll_id = 0;
  }
  g_mutex_unlock (&modelmux_bin->free_lock);
  if (modelmux_bin->perf_timer_id) {                   /* stop the perf-metric snapshot timer */
    g_source_remove (modelmux_bin->perf_timer_id);
    modelmux_bin->perf_timer_id = 0;
  }
  /* Cancel any deferred branch-free idles still queued (teardown-race UAF guard): the
   * in_q/out_q they would have NULL+removed are children of modelmux_bin->bin and get disposed when
   * the bin is torn down below. The ModelMuxFreeCtx wrappers are owned by their idle source's
   * GDestroyNotify (freed exactly once, never under a mid-flight dispatch). Remove the
   * sources UNDER free_lock: every fc still listed is un-dispatched (the callback
   * de-lists itself under this lock first), so the source cannot auto-destroy -- and
   * thus free the fc -- under this loop; the synchronous notify is a bare g_free. */
  g_mutex_lock (&modelmux_bin->free_lock);
  if (modelmux_bin->dump_source_id) {                  /* cancel any pending MM_DEBUG state dump */
    g_source_remove (modelmux_bin->dump_source_id);
    modelmux_bin->dump_source_id = 0;
  }
  {
    GList *l;
    for (l = modelmux_bin->pending_frees; l; l = l->next) {
      ModelMuxFreeCtx *fc = (ModelMuxFreeCtx *) l->data;
      /* a cancelled pad-release ctx still owns its pad + element refs (the notify
       * is a bare g_free): drop them here -- the objects themselves are disposed
       * with the bin below, no release_request_pad needed. Safe: every fc still
       * listed is un-dispatched (see loop comment above), so the refs are intact. */
      if (fc->req_pad) {
        gst_object_unref (fc->req_pad);
        gst_object_unref (fc->req_el);
        fc->req_pad = NULL;
        fc->req_el = NULL;
      }
      g_source_remove (fc->src_id);
    }
    g_list_free (modelmux_bin->pending_frees);
    modelmux_bin->pending_frees = NULL;
  }
  g_mutex_unlock (&modelmux_bin->free_lock);
  /* QUIESCE the graph BEFORE destroying shared state: set_state(NULL) deactivates
   * every pad and joins every streaming task inside the bin, so no probe can still
   * be executing (or parked on modelmux_bin->lock) when the tables/pools below are destroyed.
   * (The bin is removed from its parent only at the end, after the pools have
   * released their children.) Then take/release modelmux_bin->lock once as a barrier for any
   * thread that had already entered a locked section before shutting_down was set. */
  if (modelmux_bin->bin)
    gst_element_set_state (modelmux_bin->bin, GST_STATE_NULL);
  /* Destroy the reader-visible stream tables UNDER the lock, and NULL each one,
   * so a reader that acquires after us sees NULL and returns an empty plane
   * instead of iterating freed storage.
   *
   * set_state(NULL) joins the STREAMING tasks, but the status/route queries
   * arrive on a civetweb server thread, which it does not join -- and a bare
   * lock/unlock barrier only drains threads already inside a locked section, not
   * one still on its way in. Leaving the pointers dangling was the other half of
   * that race: every reader tests them, so a non-NULL-but-freed table is worse
   * than no table at all.
   *
   * Safe to destroy while holding the lock: these five carry either no
   * value-destroy at all or a bare g_free, so nothing here can re-enter it.
   * (Same detach-then-free shape as gst_modelmux_control_detach.) */
  g_mutex_lock (&modelmux_bin->lock);
  if (modelmux_bin->stream_wiring) {
    g_hash_table_destroy (modelmux_bin->stream_wiring);
    modelmux_bin->stream_wiring = NULL;
  }
  if (modelmux_bin->stream_names) {
    g_hash_table_destroy (modelmux_bin->stream_names);
    modelmux_bin->stream_names = NULL;
  }
  if (modelmux_bin->stream_cam_ids) {
    g_hash_table_destroy (modelmux_bin->stream_cam_ids);
    modelmux_bin->stream_cam_ids = NULL;
  }
  if (modelmux_bin->streams_pending) {
    g_hash_table_destroy (modelmux_bin->streams_pending);
    modelmux_bin->streams_pending = NULL;
  }
  if (modelmux_bin->streams_cancelled) {
    g_hash_table_destroy (modelmux_bin->streams_cancelled);
    modelmux_bin->streams_cancelled = NULL;
  }
  g_mutex_unlock (&modelmux_bin->lock);
  if (modelmux_bin->pending_routes) {             /* mirrored deferred-route view */
    g_ptr_array_unref (modelmux_bin->pending_routes);
    modelmux_bin->pending_routes = NULL;
  }
  if (modelmux_bin->col_free_idx)
    g_queue_free (modelmux_bin->col_free_idx);
  if (modelmux_bin->limbo_models) {                          /* free any never-promoted limbo bins */
    GHashTableIter it;
    gpointer k, v;
    g_hash_table_iter_init (&it, modelmux_bin->limbo_models);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelBin *model_bin = (ModelBin *) v, *next;
      for (; model_bin; model_bin = next) {               /* free the base + all its shards */
        next = (ModelBin *) model_bin->next_shard;
        model_bin_free (model_bin);
      }
    }
    g_hash_table_destroy (modelmux_bin->limbo_models);
  }
  model_pool_free (modelmux_bin->primary_pool);
  model_pool_free (modelmux_bin->shadow_pool);
  if (modelmux_bin->bin) {
    /* already at GST_STATE_NULL (quiesced above, before the state teardown) */
    if (GST_OBJECT_PARENT (modelmux_bin->bin))
      gst_bin_remove (GST_BIN (GST_OBJECT_PARENT (modelmux_bin->bin)), modelmux_bin->bin);
    else
      gst_object_unref (modelmux_bin->bin);
  }
  {
    /* lock-free stream-name mirror: safe to free only now that every probe is
     * quiesced (set_state NULL above joined all streaming tasks). */
    guint i;
    for (i = 0; i < MM_ACCT_MAX; i++)
      g_free (modelmux_bin->name_slot[i]);
    if (modelmux_bin->retired_names)
      g_ptr_array_free (modelmux_bin->retired_names, TRUE);
  }
  g_mutex_clear (&modelmux_bin->lock);
  g_mutex_clear (&modelmux_bin->free_lock);
  g_free (modelmux_bin);
}
