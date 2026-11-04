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

/**
 * gstnvmodelmux_sgie.c
 * ====================
 * SECONDARY-stage (process-mode=2) data path: a LINEAR chain.
 *
 *   sink -> queue -> [xfer-in] -> infer -> [xfer-out] -> src
 *
 * A secondary stage runs ONE model over ALL streams on the already-batched buffer
 * it receives, and infers on the OBJECTS the primary produced. There is nothing to
 * split and nothing to re-batch, so the primary stage's demux -> per-stream tee ->
 * per-model nvstreammux -> nvstreamdemux -> combined mux machinery is all dead
 * weight here; this file builds the minimal chain instead.
 *
 * What IS shared with the primary path:
 *   - the model catalog + [model-name-N] template addressing (config.c),
 *   - the cross-GPU nvdsxfer pair and its P2P preconditions,
 *   - process-mode validation against the model's own infer config.
 *
 * The queue decouples this stage from the primary's push thread so the two run in
 * parallel, exactly as deepstream-app's own secondary bin does.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "gstnvmodelmux_priv.h"
#include "gstnvmodelmux_perf.h"

GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat

#define SG_INFO(fmt, ...)  GST_INFO (fmt, ##__VA_ARGS__)
#define SG_WARN(fmt, ...)  GST_WARNING (fmt, ##__VA_ARGS__)
#define SG_ERR(fmt, ...)   GST_ERROR (fmt, ##__VA_ARGS__)

/* gie-unique-id FALLBACK for a secondary stage, used only when the model's infer config
 * declares none. Clear of the primary path's Primary (1..), Shadow (100..) and limbo
 * (1000..) ranges so downstream meta attribution stays unambiguous when both stages are
 * in one pipeline. */
#define MM_SGIE_GIE_ID_BASE  2000

typedef struct _ModelMuxSgieModel ModelMuxSgieModel;

/* One model LOADED at this stage: its inference element and the identity it carries.
 * Exactly one entry is SERVING (linked into the chain and inferring); any other is
 * built, warmed and resident but not linked -- "loading a second model of the same
 * template makes it AVAILABLE, not ACTIVE" (sgie approach, constraint 3). */
struct _ModelMuxSgieModel
{
  ModelMuxSgieBin *owner;   /**< back-pointer: a signal callback gets the model    */
  GstElement *xfer_in;      /**< NULL unless the model is on another device        */
  GstElement *infer;        /**< nvinfer / nvinferserver / nvdsvisionencoder       */
  GstElement *xfer_out;     /**< NULL unless the model is on another device        */

  gchar      *name;
  gchar      *version;
  gchar      *config_file;  /**< EFFECTIVE path (version override or catalog base) */
  ModelType   type;
  guint       unique_id;
  gint        cfg_gpu;      /**< device the model config pins to (-1 = none)       */
  /** nvinfer `operate-on-gie-id` from this model's config: the gie whose objects it
   *  classifies. -1 = every GIE (nvinfer's default). Read once at configure time --
   *  it is the model's INPUT filter, so it is what decides which frames this model
   *  actually ran on, and therefore which tiles the MM_OVERLAY line belongs on. */
  gint        operate_on_gie;
  gint        gpu;          /**< effective device the infer element runs on        */
  /** the container bin owns the elements from here on; before it, they are floating
   *  and a build failure must unref them by hand. */
  gboolean    in_bin;
  /** a runtime model/load warms on its own thread (an engine build blocks for as long
   *  as it takes); set when that thread gives up, so status reports 'failed' rather
   *  than a 'warming' that never ends. Under @state_lock. */
  gboolean    warm_failed;
  /** TRUE from the moment a runtime load hands this model to its warm thread until that
   *  thread is done with it. AUTHORITATIVE, because the element's GstState is not: an
   *  nvinfer building a TensorRT engine does it inside the READY->PAUSED transition, and
   *  a zero-timeout gst_element_get_state() reports the CURRENT state with the change
   *  still pending -- so a model deep in a 4-minute build can already read as PAUSED.
   *  Inferring "warming" from that made model/routing take the immediate-rebind path on
   *  a model that could not yet infer. Atomic: set by the load before the thread starts,
   *  cleared by the thread, read by both. */
  gint        warming;
  /** the warm thread, held so teardown can JOIN it before freeing this model -- it
   *  touches these elements, and an engine build outliving the stage is a UAF. */
  GThread    *warm_thread;
  /** DROP probe on this model's tail src, present iff it is NOT serving. A resident
   *  model's tail has no peer (only the serving chain is ghosted out), so without this
   *  a late buffer -- the one still inside nvinfer when a rebind displaced it -- would
   *  push into nothing and surface as "Internal data stream error". 0 = serving. */
  gulong      park_probe;
};

struct _ModelMuxSgieBin
{
  GstElement *bin;          /**< container; a child of the nvmodelmux element   */
  GstElement *queue;        /**< decouples this stage from the primary's thread */
  GstPad     *sink_ghost;
  GstPad     *src_ghost;

  /** every model resident at this stage (ModelMuxSgieModel *), serving one included */
  GList      *models;
  /** the one that infers. Never NULL once built: a secondary stage always has a
   *  serving model (the config-declared one warms before PLAYING). */
  ModelMuxSgieModel *serving;

  guint       batch_size;   /**< max streams this stage serves                  */

  /** borrowed, read-only: the catalog + per-version artifacts model/update resolves against */
  const ModelMuxConfig *config;

  /** Batch meta + PTS as last seen ARRIVING at this model, so the output probe can
   *  tell whether the backend replaced them (see sgie_check_meta_passthrough).
   *  Written and read on the streaming thread only. */
  gpointer    in_bmeta;
  guint64     in_pts;
  /** one-shot latch for that check: 0 = not yet decided, 1 = done */
  gint        meta_checked;

  /** MM_OVERLAY line index for this stage (0 = first secondary). See
   *  modelmux_sgie_set_overlay_slot. */
  guint       overlay_slot;

  /** the PIPELINE-WIDE routing revision, shared with the primary (and any other
   *  secondary) through MODELMUX_CONTEXT_TYPE_ROUTING. Owns a ref; NULL until the
   *  element negotiates the context, in which case this stage reports 0 -- the same
   *  answer it gave before the counter existed. Self-locking, so it is read while
   *  @state_lock is held. */
  ModelMuxRevShared *rev_shared;

  /** perf metrics: opt-in. @perf_on is the cached hot-path gate, @perf the counters.
   *  Measured at the STAGE boundary (queue out -> infer src), so the number describes
   *  this stage and not the backend element inside it. */
  /** streams currently flowing through this stage: one frame per live stream in
   *  each batch, so num_frames_in_batch IS the count. Atomic, written by the tail
   *  probe, read by model/status + the MM_DEBUG table. */
  gint        live_streams;

  gboolean    perf_on;
  ModelMuxPerf *perf;
  guint       perf_timer_id;   /**< window tick (0 = not running)                */
  GMutex      perf_lock;       /**< serializes tick vs the status read           */

  /** ONE lock for everything mutable about the stage's models: the @models list, the
   *  @serving pointer, and each model's identity (an in-place OTA swaps version and
   *  config_file live). A single lock is deliberate -- the status reader walks the list
   *  AND reads identities, so two locks would only invent an ordering to get wrong. */
  GMutex      state_lock;
  /** in-place OTA state. @ota_await is set at trigger and cleared by nvinfer's
   *  'model-updated' confirm; the ota_old_* copies restore the visible identity when
   *  the reload FAILS (the previous engine keeps serving). */
  gboolean    ota_await;
  gchar      *ota_old_version;
  gchar      *ota_old_config;
  /** model/routing target while a rebind is in flight. @rebind_to is read by the block
   *  probe that performs the switch; @pending_serve is a route that arrived for a model
   *  still WARMING -- the current model keeps serving and the warm thread applies it on
   *  READY, so a reroute to a cold model costs no inference gap. Under @state_lock. */
  ModelMuxSgieModel *rebind_to;
  ModelMuxSgieModel *pending_serve;
};

/* nvinfer's OTA confirm (section 6); connected when the infer element is configured. */
static void sgie_model_updated_cb (GstElement * infer, gint status, gchar * cfg_file,
    gpointer udata);
/* Section 7 reads a model's lifecycle state; section 6c decides on it. Section 6c arms
 * the rebind, and 6b's warm thread applies a route that was parked waiting for it. */
static const gchar *sgie_state_str (ModelMuxSgieModel * sgie_model);
static void sgie_rebind_arm (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * to);
static void sgie_model_park (ModelMuxSgieModel * sgie_model, gboolean parked);

/* The two ends of one model's chain: queue -> HEAD -> ... -> TAIL -> src ghost. The
 * xfer pair only exists when the model sits on another device, so both ends collapse
 * onto the inference element in the common case. */
static GstElement *
sgie_model_head (ModelMuxSgieModel * sgie_model)
{
  /* OUTERMOST first: the rebind probe links queue->head, so head must be whatever
   * element actually starts this model's sub-chain. */
  if (sgie_model->xfer_in)
    return sgie_model->xfer_in;
  return sgie_model->infer;
}

static GstElement *
sgie_model_tail (ModelMuxSgieModel * sgie_model)
{
  if (sgie_model->xfer_out)
    return sgie_model->xfer_out;
  return sgie_model->infer;
}

/* Every GStreamer element this model owns, in chain order, NULL-terminated. ONE list,
 * because three separate places have to walk them -- warm (sync state), unload (set
 * NULL + remove from the bin) and free (unref while still floating) -- and a backend
 * that adds an element to the chain must not be able to be forgotten by one of them.
 * @out must hold at least SGIE_MODEL_MAX_ELEMENTS entries. */
#define SGIE_MODEL_MAX_ELEMENTS 3
static guint
sgie_model_elements (ModelMuxSgieModel * sgie_model, GstElement ** out)
{
  guint n = 0;

  if (sgie_model->xfer_in)  out[n++] = sgie_model->xfer_in;
  if (sgie_model->infer)    out[n++] = sgie_model->infer;
  if (sgie_model->xfer_out) out[n++] = sgie_model->xfer_out;
  out[n] = NULL;
  return n;
}

/* Identity only -- the elements are created separately, so a failed resolve costs
 * nothing to unwind. */
static ModelMuxSgieModel *
sgie_model_new (ModelMuxSgieBin * sgie_bin)
{
  ModelMuxSgieModel *sgie_model = g_new0 (ModelMuxSgieModel, 1);

  sgie_model->owner = sgie_bin;
  sgie_model->cfg_gpu = -1;
  sgie_model->gpu = sgie_bin->config ? (gint) sgie_bin->config->gpu : 0;
  return sgie_model;
}

/* Frees the identity. The elements belong to the container bin once assembled, so they
 * are NOT unreffed here -- @free_elements is for the build-failure path, where they are
 * still floating. */
static void
sgie_model_free (ModelMuxSgieModel * sgie_model, gboolean free_elements)
{
  if (!sgie_model)
    return;
  if (sgie_model->warm_thread) {
    g_thread_join (sgie_model->warm_thread);      /* it touches the elements below */
    sgie_model->warm_thread = NULL;
  }
  if (free_elements) {
    if (sgie_model->infer)    gst_object_unref (sgie_model->infer);
    if (sgie_model->xfer_in)  gst_object_unref (sgie_model->xfer_in);
    if (sgie_model->xfer_out) gst_object_unref (sgie_model->xfer_out);
  }
  g_free (sgie_model->name);
  g_free (sgie_model->version);
  g_free (sgie_model->config_file);
  g_free (sgie_model);
}

/* g_list_free_full adaptor: the container bin already owns the elements by then. */
static void
sgie_model_free_cb (gpointer sgie_model)
{
  sgie_model_free ((ModelMuxSgieModel *) sgie_model, FALSE);
}

/* NULL @version matches any version of @name. Caller holds @state_lock. */
static ModelMuxSgieModel *
sgie_find_model (ModelMuxSgieBin * sgie_bin, const gchar * name, const gchar * version)
{
  GList *l;

  for (l = sgie_bin->models; l; l = l->next) {
    ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) l->data;
    if (g_strcmp0 (sgie_model->name, name) == 0
        && (!version || g_strcmp0 (sgie_model->version, version) == 0))
      return sgie_model;
  }
  return NULL;
}

/* ================================================================== */
/* Small helpers (local copies: this file is self-contained)           */
/* ================================================================== */

static GstElement *
sgie_make (const gchar * factory, const gchar * name)
{
  GstElement *e = gst_element_factory_make (factory, name);
  if (!e)
    SG_ERR ("sgie: failed to create element '%s' (factory '%s')", name, factory);
  return e;
}

/* Set an integer property only when the element's class exposes it (see the primary
 * path's set_prop_if_exists: nvstreammux/nvdsxfer property sets vary by build). */
static void
sgie_set_int (GstElement * el, const gchar * prop, gint val)
{
  if (el && g_object_class_find_property (G_OBJECT_GET_CLASS (el), prop))
    g_object_set (G_OBJECT (el), prop, val, NULL);
}

/* ================================================================== */
/* 1. Resolve a model's identity                                       */
/* ================================================================== */

/* Fill @sgie_model from the catalog: the EFFECTIVE config path (through the per-version artifact
 * overrides, so a [model-<name>-<version>] block works here exactly as it does at a
 * primary), the backend type, the device the config pins and the gie id. */
static gboolean
sgie_resolve_identity (ModelMuxSgieModel * sgie_model, const ModelMuxConfig * config,
    const gchar * name, const gchar * version)
{
  const ModelCatalogEntry *entry;
  gboolean derive_failed = FALSE;
  gchar *derived;

  sgie_model->name = g_strdup (name);
  sgie_model->version = g_strdup (version ? version : MM_MODEL_VERSION_DEFAULT);

  entry = modelmux_config_find_model (config, sgie_model->name);
  if (!entry || !entry->config_file) {
    SG_ERR ("sgie: model '%s' has no [model-name-N] entry with a config -- declare it "
        "in the modelmux config", sgie_model->name);
    return FALSE;
  }

  derived = modelmux_config_resolve_version_cfg (config, sgie_model->name, sgie_model->version,
      &derive_failed);
  if (derive_failed) {
    SG_ERR ("sgie: model '%s@%s' declares an engine/gpu override that could not be "
        "derived -- refusing to run the base config instead", sgie_model->name, sgie_model->version);
    return FALSE;
  }
  sgie_model->config_file = derived ? derived : g_strdup (entry->config_file);
  sgie_model->type = entry->type;
  sgie_model->cfg_gpu = modelmux_config_file_gpu (sgie_model->config_file);

  /* gie-unique-id: PREFER the id the model's own infer config declares, same rule as the
   * primary stage -- that is what nvinfer stamps into the meta, so a further stage can
   * target this one straight from config. No shadow offset here: a secondary is
   * single-lane. Falls back to MM_SGIE_GIE_ID_BASE when the config declares none. */
  {
    gint declared = modelmux_config_file_gie_id (sgie_model->config_file);
    sgie_model->unique_id = (declared > 0) ? (guint) declared : MM_SGIE_GIE_ID_BASE;
    if (declared <= 0)
      SG_INFO ("sgie: '%s' declares no gie-unique-id -- using %u; declare one to make it "
          "targetable from config", sgie_model->config_file, sgie_model->unique_id);
  }
  return TRUE;
}

/* The model this stage serves at build time: the configured `primary-model`
 * (name;version). Compulsory in sgie mode -- a secondary stage always has a serving
 * model -- so a missing one is a build failure, not a passthrough. */
static ModelMuxSgieModel *
sgie_resolve_serving (ModelMuxSgieBin * sgie_bin, const ModelMuxConfig * config)
{
  ModelMuxSgieModel *sgie_model;

  /* config validation already guaranteed a catalog-declared primary-model (a secondary
   * cannot start without one); this is the defensive read, not a second policy. */
  if (!config->default_primary) {
    SG_ERR ("sgie: no serving model -- 'primary-model' must name a [model-name-N] entry");
    return NULL;
  }
  sgie_bin->batch_size = config->batch_size ? config->batch_size : MM_MAX_STREAMS_DEFAULT;

  sgie_model = sgie_model_new (sgie_bin);
  if (!sgie_resolve_identity (sgie_model, config, config->default_primary,
          config->default_primary_version)) {
    sgie_model_free (sgie_model, FALSE);
    return NULL;
  }
  return sgie_model;
}

/* ================================================================== */
/* 2. Create the elements                                              */
/* ================================================================== */

/* The inference backend, chosen from the model config's format (same auto-detection
 * as the primary path). Which element to build is a registry lookup, so a new backend
 * needs no change here. */
static GstElement *
sgie_create_infer (ModelMuxSgieModel * sgie_model, const gchar * elem_name)
{
  return sgie_make (modelmux_backend (sgie_model->type)->element, elem_name);
}

/* Queue at the head: the primary stage pushes from its own streaming thread, so this
 * hands the batch off and lets both stages run concurrently. Leaky=downstream would
 * silently drop objects, so it is deliberately NOT set -- a secondary must see every
 * frame the primary detected on. */
static gboolean
sgie_create_head (ModelMuxSgieBin * sgie_bin, const ModelMuxSgieModel * sgie_model)
{
  gchar nm[128];

  g_snprintf (nm, sizeof (nm), "sgie-queue-%s", sgie_model->name);
  sgie_bin->queue = sgie_make ("queue", nm);
  return sgie_bin->queue != NULL;
}

/* ================================================================== */
/* 3. Optional cross-GPU migration                                     */
/* ================================================================== */

/* When the model config pins a device other than the pipeline's, wrap the inference
 * element in an nvdsxfer pair: xfer-in migrates the batch to the model's device
 * (nvinfer hard-rejects foreign surfaces), xfer-out brings it back so downstream stays
 * on the pipeline device. Same preconditions as the primary path -- CUDA peer access
 * both ways plus the nvdsxfer factory -- and the same deterministic refusal when
 * either is missing. Returns FALSE to reject the placement. */
static gboolean
sgie_create_xfer (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * sgie_model,
    const ModelMuxConfig * config)
{
  guint pipe_gpu = config->gpu;
  gchar nm[128];
  int ab = 0, ba = 0;

  sgie_model->gpu = (sgie_model->cfg_gpu >= 0) ? sgie_model->cfg_gpu : (gint) pipe_gpu;
  if (sgie_model->gpu == (gint) pipe_gpu)
    return TRUE;                        /* same device: no migration elements at all */

  if (cudaDeviceCanAccessPeer (&ab, (int) pipe_gpu, sgie_model->gpu) != cudaSuccess ||
      cudaDeviceCanAccessPeer (&ba, sgie_model->gpu, (int) pipe_gpu) != cudaSuccess
      || !ab || !ba) {
    SG_ERR ("sgie: cross-GPU placement (pipeline gpu %u -> model gpu %d) REJECTED -- "
        "no CUDA peer access between the devices (nvdsxfer requires NVLink/PCIe P2P) "
        "(P2P_UNAVAILABLE)", pipe_gpu, sgie_model->gpu);
    return FALSE;
  }

  g_snprintf (nm, sizeof (nm), "sgie-xfer-in-%s-%s", sgie_model->name, sgie_model->version);
  sgie_model->xfer_in = sgie_make ("nvdsxfer", nm);
  g_snprintf (nm, sizeof (nm), "sgie-xfer-out-%s-%s", sgie_model->name, sgie_model->version);
  sgie_model->xfer_out = sgie_make ("nvdsxfer", nm);
  if (!sgie_model->xfer_in || !sgie_model->xfer_out) {
    SG_ERR ("sgie: cross-GPU placement on gpu %d REJECTED -- the 'nvdsxfer' factory is "
        "unavailable (install libnvdsgst_xfer.so) (XFER_UNAVAILABLE)", sgie_model->gpu);
    return FALSE;
  }

  /* nvdsxfer semantics: gpu-id = OUTPUT device, p2p-gpu-id = the peer it transfers from. */
  sgie_set_int (sgie_model->xfer_in, "gpu-id", sgie_model->gpu);
  sgie_set_int (sgie_model->xfer_in, "p2p-gpu-id", (gint) pipe_gpu);
  sgie_set_int (sgie_model->xfer_in, "batch-size", (gint) sgie_bin->batch_size);
  sgie_set_int (sgie_model->xfer_out, "gpu-id", (gint) pipe_gpu);
  sgie_set_int (sgie_model->xfer_out, "p2p-gpu-id", sgie_model->gpu);
  sgie_set_int (sgie_model->xfer_out, "batch-size", (gint) sgie_bin->batch_size);
  return TRUE;
}

/* ================================================================== */
/* 4. Configure the inference element                                  */
/* ================================================================== */

/* batch-size here is the OBJECT batch nvinfer forms per push; it is sized from the
 * stage's stream capacity like the primary path does. process-mode comes from the
 * model's own infer config -- warn (never override) when it disagrees, because a
 * secondary model running full-frame produces plausible-looking garbage silently. */
static void
sgie_configure_infer (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * sgie_model,
    const ModelMuxConfig * config)
{
  const ModelBackendInfo *be = modelmux_backend (sgie_model->type);

  if (be->config_group) {
    /* PROPERTY-CONFIGURED backend (no config-file-path): the model's config file is a
     * group of property names, applied one by one. batch-size comes from the FILE for
     * these -- it is the object/crop batch baked into the engine, not the stage's
     * stream capacity, and overriding it here would contradict the engine. */
    modelmux_config_apply_properties (sgie_model->infer, sgie_model->config_file,
        be->config_group);
    /* placement is the STAGE's decision, applied after the file so a model/load
     * carrying `gpu` wins over the config's own key. */
    sgie_set_int (sgie_model->infer, "gpu-id", sgie_model->gpu);
  } else {
    g_object_set (G_OBJECT (sgie_model->infer),
        be->config_prop, sgie_model->config_file,
        "batch-size", sgie_bin->batch_size, NULL);
  }

  /* A backend with no gie identity has neither property: nothing to stamp, and no
   * input filter to read. Its models are attributed by name/version alone. */
  if (be->gie_id) {
    g_object_set (G_OBJECT (sgie_model->infer), "unique-id", sgie_model->unique_id, NULL);
    /* Cache the model's input filter once (see ModelMuxSgieModel.operate_on_gie): read
     * here rather than per frame because the overlay consults it on the streaming thread. */
    sgie_model->operate_on_gie =
        modelmux_config_file_operate_on_gie (sgie_model->config_file);
  } else {
    sgie_model->operate_on_gie = -1;      /* every GIE: it cannot be scoped */
  }

  /* OTA completion signal, connected for EVERY backend that can reload in place --
   * nvinfer and nvdsvisionencoder both register "model-updated" with the same
   * signature. The name must be exactly right: a wrong one silently drops the confirm
   * and emits a GObject-CRITICAL at connect. The MODEL is the udata, because every
   * loaded model has its own element and fires its own confirm. */
  if (be->inplace_reload)
    g_signal_connect (sgie_model->infer, "model-updated",
        G_CALLBACK (sgie_model_updated_cb), sgie_model);

  SG_INFO ("sgie: '%s@%s' configured: backend=%s gie=%u operate-on-gie=%d batch=%s",
      sgie_model->name, sgie_model->version, be->element, sgie_model->unique_id,
      sgie_model->operate_on_gie,
      be->config_group ? "from config" : "from stage");

  /* Everything below is about a config file that DECLARES its position, which only a
   * config-file backend has. */
  if (sgie_model->type != MODEL_INFER)
    return;

  {
    gint declared = modelmux_config_file_process_mode (sgie_model->config_file);
    if (declared < 0)
      SG_WARN ("sgie: '%s' does not set process-mode; nvinfer defaults to 1 (full frame) "
          "but this stage is process-mode=%d -- set it in the model config",
          sgie_model->config_file, config->process_mode);
    else if (declared != config->process_mode)
      SG_WARN ("sgie: '%s' sets process-mode=%d but this stage is process-mode=%d -- the "
          "model config wins; fix it or the model runs at the wrong position",
          sgie_model->config_file, declared, config->process_mode);
  }
}

/* ================================================================== */
/* 4b. Perf at the stage boundary (opt-in)                             */
/*                                                                     */
/* Same contract as the primary's ModelBin: stamp the batch in at the  */
/* stage head, record every output frame at the tail. Because it is    */
/* measured at the BOUNDARY, the number means the same thing whichever */
/* backend runs inside -- nvinfer today, the vision encoder later.     */
/* ================================================================== */

static GstPadProbeReturn
sgie_perf_in_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSgieBin *sgie_bin = (ModelMuxSgieBin *) udata;
  (void) pad; (void) info;

  if (sgie_bin->perf)
    modelmux_perf_mark_bin_in (sgie_bin->perf, g_get_monotonic_time ());
  return GST_PAD_PROBE_OK;
}


/* Append @suffix to the label the PRIMARY drew for this (frame, role), so a tile shows
 * "<primary provenance>  |  <secondary>" on one line.
 *
 * Matched by PREFIX against the primary's own text, which is how a secondary finds the
 * line belonging to the lane it ran on. Returns TRUE when a label was found. */
static gboolean
sgie_overlay_append (NvDsFrameMeta * fm, const InferenceProvenanceMeta * pm,
    const gchar * suffix, const gchar * own)
{
  NvDsMetaList *dl;
  gchar *want = g_strdup_printf ("%s:%u:%s-%s:%s", pm->camera_name, fm->frame_num,
      pm->model_name, pm->model_version, pm->role);
  gboolean done = FALSE;
  /* CROSS-STAGE lock: sibling secondaries sit on separate tee branches and therefore
   * separate streaming threads, but they append to the SAME label. Unsynchronised, the
   * read-modify-write below runs twice concurrently -- one suffix overwrites the other
   * and both threads free the same pointer. */
  static GMutex overlay_lock;

  g_mutex_lock (&overlay_lock);
  for (dl = fm->display_meta_list; dl != NULL && !done; dl = dl->next) {
    NvDsDisplayMeta *dm = (NvDsDisplayMeta *) dl->data;
    guint i;
    for (i = 0; i < dm->num_labels; i++) {
      NvOSD_TextParams *tp = &dm->text_params[i];
      if (!tp->display_text || !g_str_has_prefix (tp->display_text, want))
        continue;
      /* idempotent PER MODEL: a shared marker would let the first secondary's label
       * block every later one. */
      if (!strstr (tp->display_text, own)) {
        gchar *merged = g_strconcat (tp->display_text, suffix, NULL);
        g_free (tp->display_text);
        tp->display_text = merged;
      }
      done = TRUE;
      break;
    }
  }
  g_mutex_unlock (&overlay_lock);
  g_free (want);
  return done;
}

/* Defined with the probe-installation code below; used by the output probe here. */
static void sgie_check_meta_passthrough (ModelMuxSgieBin * sgie_bin,
    ModelMuxSgieModel * sgie_model, GstBuffer * buffer, NvDsBatchMeta * out_bmeta);

/* One timestamp per BATCH, recorded against each frame's source. At a secondary the
 * batch is never re-muxed internally, so fm->source_id IS the upstream source id --
 * no slot translation is needed (the primary path needs one; this does not). */
static GstPadProbeReturn
sgie_out_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) udata;
  ModelMuxSgieBin *sgie_bin = sgie_model->owner;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta;
  NvDsMetaList *l;
  gint64 now;
  (void) pad;

  /* Installed on EVERY resident model, so a rebind needs no probe migration -- but only
   * the one actually inferring may report. An idle model is normally silent anyway
   * (nothing is linked into it); this also covers the buffer still inside nvinfer when
   * a rebind displaces it, which must not be counted as live traffic. */
  if (g_atomic_pointer_get (&sgie_bin->serving) != sgie_model)
    return GST_PAD_PROBE_OK;

  if (!buffer)
    return GST_PAD_PROBE_OK;
  bmeta = gst_buffer_get_nvds_batch_meta (buffer);
  if (!bmeta)
    return GST_PAD_PROBE_OK;

  /* ALWAYS: the live stream count, so model/status reports the same `streams` key
   * the primary does instead of a secondary-only shape. */
  g_atomic_int_set (&sgie_bin->live_streams, (gint) bmeta->num_frames_in_batch);

  /* MM_OVERLAY: put "sgie:<name>@<version> gie<id>" on every tile, so the tiled display
   * alone says which secondary model and version produced the classifications -- the
   * same question the primary's line answers for the detector. Appended to that line
   * drawn as this stage's OWN line (see overlay_slot), stacked under the primary's.
   *
   * The identity is read under @state_lock because an OTA swaps name/version live and
   * this runs on the streaming thread. That lock is only ever taken here when the
   * overlay is ON -- it is a debug path, opt-in, and off by default. */
  if (modelmux_overlay_enabled ()) {
    NvDsMetaType ptype = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;
    gint want_gie = sgie_model->operate_on_gie;
    NvDsMetaList *fl;
    gchar *suffix, *own;

    g_mutex_lock (&sgie_bin->state_lock);
    suffix = g_strdup_printf ("  |  %s@%s", sgie_model->name, sgie_model->version);
    g_mutex_unlock (&sgie_bin->state_lock);
    own = suffix + 5;                  /* same text without the "  |  " joiner */

    /* Label ONLY the frames this model actually ran on -- `operate-on-gie-id` is the
     * input filter, so a frame produced by another GIE was never seen by this model.
     * -1 means no filter (every GIE), which is also all a backend with no gie identity
     * can mean. */
    for (fl = bmeta->frame_meta_list; fl != NULL; fl = fl->next) {
      NvDsFrameMeta *fm = (NvDsFrameMeta *) fl->data;
      NvDsMetaList *ul;
      gboolean saw_prov = FALSE;

      for (ul = fm->frame_user_meta_list; ul != NULL; ul = ul->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) ul->data;
        InferenceProvenanceMeta *pm;
        if (um->base_meta.meta_type != ptype || !um->user_meta_data)
          continue;
        pm = (InferenceProvenanceMeta *) um->user_meta_data;
        saw_prov = TRUE;
        if (want_gie >= 0 && pm->gie_id != (guint) want_gie)
          continue;
        sgie_overlay_append (fm, pm, suffix, own);
      }
      if (!saw_prov && want_gie < 0)
        modelmux_draw_label (bmeta, fm, 12, 1000, own, "Secondary");
    }
    g_free (suffix);
  }

  if (!sgie_bin->perf)
    return GST_PAD_PROBE_OK;
  now = g_get_monotonic_time ();
  for (l = bmeta->frame_meta_list; l != NULL; l = l->next)
    modelmux_perf_mark_bin_out (sgie_bin->perf, now,
        (gint) ((NvDsFrameMeta *) l->data)->source_id);
  return GST_PAD_PROBE_OK;
}

/* Recompute the published fps/latency over the window. modelmux_perf_tick requires the
 * caller's lock; the status reader takes the same one. */
static gboolean
sgie_perf_timer_cb (gpointer udata)
{
  ModelMuxSgieBin *sgie_bin = (ModelMuxSgieBin *) udata;

  g_mutex_lock (&sgie_bin->perf_lock);
  modelmux_perf_tick (sgie_bin->perf, g_get_monotonic_time ());
  g_mutex_unlock (&sgie_bin->perf_lock);
  return G_SOURCE_CONTINUE;
}

/* Tail probe is ALWAYS installed (it maintains the live stream count that model/status
 * reports); the head probe and the window timer are perf-only. Head = the stage's
 * inference head (xfer-in when cross-GPU, else the inference element), tail = the
 * inference src, mirroring the primary's measurement window. */
/* Compare what LEAVES the stage against what entered it (see
 * sgie_check_meta_passthrough). Its own probe on the stage TAIL rather than a call
 * from the reporting probe on the inference src: the two must bracket the WHOLE
 * chain, or an element sitting beside the backend -- an nvdsxfer on a cross-GPU
 * model -- could replace the buffer between them and go unseen. With no xfer the
 * tail IS the inference element and this is the same pad as before. */
static GstPadProbeReturn
sgie_meta_check_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) udata;
  ModelMuxSgieBin *sgie_bin = sgie_model->owner;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta;
  (void) pad;

  if (g_atomic_int_get (&sgie_bin->meta_checked))
    return GST_PAD_PROBE_OK;            /* decided already */
  if (!buffer || !(bmeta = gst_buffer_get_nvds_batch_meta (buffer)))
    return GST_PAD_PROBE_OK;

  sgie_check_meta_passthrough (sgie_bin, sgie_model, buffer, bmeta);
  return GST_PAD_PROBE_OK;
}

/* Record the batch meta (and the PTS that identifies it) as it ARRIVES at the model,
 * for sgie_check_meta_passthrough below. */
static GstPadProbeReturn
sgie_in_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) udata;
  ModelMuxSgieBin *sgie_bin = sgie_model->owner;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta;
  (void) pad;

  if (g_atomic_int_get (&sgie_bin->meta_checked))
    return GST_PAD_PROBE_OK;            /* decided already -- cost nothing after that */
  if (!buffer || !(bmeta = gst_buffer_get_nvds_batch_meta (buffer)))
    return GST_PAD_PROBE_OK;

  sgie_bin->in_bmeta = bmeta;
  sgie_bin->in_pts = (guint64) GST_BUFFER_PTS (buffer);
  return GST_PAD_PROBE_OK;
}

/* Does this backend hand back the SAME NvDsBatchMeta it was given?
 *
 * A secondary stage sits on a branch whose buffer is shared, and it exists solely to
 * ANNOTATE the batch every other stage reads. An element that replaces the buffer --
 * a GstBaseTransform with passthrough=FALSE will, because it must obtain a writable
 * one and a shared buffer is not writable -- gets a DEEP COPY of the batch meta
 * (gst_nvds_meta_transform -> nvds_batch_meta_copy_func). It then annotates the copy,
 * the branch ends in a fakesink, and the copy is dropped. The stage runs perfectly,
 * reports serving, and produces nothing that anything downstream can see.
 *
 * That failure is invisible without this check, which is the only reason it is worth
 * paying for: ONE comparison on the first buffer, then a latched flag. Checked at
 * runtime rather than at build because whether an element replaces the buffer is not
 * knowable from its registration -- only from what it does with one. */
static void
sgie_check_meta_passthrough (ModelMuxSgieBin * sgie_bin,
    ModelMuxSgieModel * sgie_model, GstBuffer * buffer, NvDsBatchMeta * out_bmeta)
{
  const gchar *elem;

  if (!sgie_bin->in_bmeta
      || sgie_bin->in_pts != (guint64) GST_BUFFER_PTS (buffer))
    return;                             /* not the buffer we recorded -- try the next */
  if (!g_atomic_int_compare_and_exchange (&sgie_bin->meta_checked, 0, 1))
    return;                             /* another thread already decided */
  if (sgie_bin->in_bmeta == out_bmeta)
    return;                             /* in place, as a secondary must be */

  elem = modelmux_backend (sgie_model->type)->element;
  GST_ELEMENT_ERROR (sgie_bin->bin, STREAM, FAILED,
      ("'%s' replaces the batch metadata, so nothing it produces is visible "
          "downstream", elem),
      ("A secondary stage annotates a batch that other stages read, but this element "
       "handed back a DIFFERENT NvDsBatchMeta (%p in, %p out) -- it replaced the "
       "buffer, which deep-copies the metadata. Its output, including model '%s@%s's "
       "results, is written to that copy and dropped with it. An element used at a "
       "secondary position must consume the batch in place, as nvinfer does "
       "(gst_base_transform_set_passthrough TRUE); refusing rather than running and "
       "silently producing nothing", sgie_bin->in_bmeta, (void *) out_bmeta,
          sgie_model->name, sgie_model->version));
}

static void
sgie_install_probes (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * sgie_model)
{
  GstElement *head = sgie_model_head (sgie_model);
  GstPad *p;

  /* The in-place check brackets the STAGE: head sink -> tail src. head/tail already
   * account for the optional xfer pair, so this widens automatically for a cross-GPU
   * model and collapses onto the inference element when there is none. */
  p = gst_element_get_static_pad (sgie_model_head (sgie_model), "sink");
  if (p) {
    gst_pad_add_probe (p, GST_PAD_PROBE_TYPE_BUFFER, sgie_in_probe, sgie_model, NULL);
    gst_object_unref (p);
  }
  p = gst_element_get_static_pad (sgie_model_tail (sgie_model), "src");
  if (p) {
    gst_pad_add_probe (p, GST_PAD_PROBE_TYPE_BUFFER, sgie_meta_check_probe, sgie_model,
        NULL);
    gst_object_unref (p);
  }

  p = gst_element_get_static_pad (sgie_model->infer, "src");
  if (p) {
    gst_pad_add_probe (p, GST_PAD_PROBE_TYPE_BUFFER, sgie_out_probe, sgie_model, NULL);
    gst_object_unref (p);
  }

  if (!sgie_bin->perf_on)
    return;
  /* The counters and the window timer belong to the STAGE, not to a model: this runs
   * again for every model/load, and a second timer would tick the same window twice. */
  if (!sgie_bin->perf) {
    sgie_bin->perf = modelmux_perf_new ();
    g_mutex_init (&sgie_bin->perf_lock);
    sgie_bin->perf_timer_id = g_timeout_add_seconds (
        sgie_bin->config->perf_metric_interval_sec, sgie_perf_timer_cb, sgie_bin);
  }

  p = gst_element_get_static_pad (head, "sink");
  if (p) {
    gst_pad_add_probe (p, GST_PAD_PROBE_TYPE_BUFFER, sgie_perf_in_probe, sgie_bin, NULL);
    gst_object_unref (p);
  }
}

/* ================================================================== */
/* 5. Assemble: add, link, ghost                                       */
/* ================================================================== */

/* queue -> [xfer-in] -> [conv-in] -> infer -> [conv-out] -> [xfer-out], then ghost the
 * chain's ends out of the container bin. All links are static (single pads throughout),
 * so a failure here is a hard build error rather than something to recover from. */
static gboolean
sgie_assemble (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * sgie_model)
{
  /* the HELPER, never a local re-derivation: it is the one place that knows which
   * optional elements bracket the backend, and ghosting anything else targets a pad
   * the chain has already linked. */
  GstElement *tail = sgie_model_tail (sgie_model);
  GstPad *p;
  gboolean ok;

  gst_bin_add_many (GST_BIN (sgie_bin->bin), sgie_bin->queue, sgie_model->infer, NULL);
  if (sgie_model->xfer_in)
    gst_bin_add (GST_BIN (sgie_bin->bin), sgie_model->xfer_in);
  if (sgie_model->xfer_out)
    gst_bin_add (GST_BIN (sgie_bin->bin), sgie_model->xfer_out);
  sgie_model->in_bin = TRUE;    /* ownership transferred; a later failure must not unref */

  /* queue -> head, then the model's own inner chain. head/tail already account for
   * the optional xfer and converter elements, so this reads the same whatever the
   * backend asked for. */
  ok = gst_element_link (sgie_bin->queue, sgie_model_head (sgie_model));
  if (ok && sgie_model->xfer_in)
    ok = gst_element_link (sgie_model->xfer_in, sgie_model->infer);
  if (ok && sgie_model->xfer_out)
    ok = gst_element_link (sgie_model->infer, sgie_model->xfer_out);
  if (!ok) {
    SG_ERR ("sgie: static link queue->[xfer]->infer->[xfer] failed");
    return FALSE;
  }

  p = gst_element_get_static_pad (sgie_bin->queue, "sink");
  sgie_bin->sink_ghost = gst_ghost_pad_new ("sink", p);
  gst_object_unref (p);
  p = gst_element_get_static_pad (tail, "src");
  sgie_bin->src_ghost = gst_ghost_pad_new ("src", p);
  gst_object_unref (p);
  if (!sgie_bin->sink_ghost || !sgie_bin->src_ghost) {
    SG_ERR ("sgie: failed to ghost the chain boundary pads");
    return FALSE;
  }
  gst_pad_set_active (sgie_bin->sink_ghost, TRUE);
  gst_pad_set_active (sgie_bin->src_ghost, TRUE);
  gst_element_add_pad (sgie_bin->bin, sgie_bin->sink_ghost);
  gst_element_add_pad (sgie_bin->bin, sgie_bin->src_ghost);
  return TRUE;
}

/* ================================================================== */
/* 6. model/update: in-place OTA on the one inference element          */
/*                                                                     */
/* Same mechanism as the primary path (gstnvmodelmux_bin.c), minus     */
/* everything a secondary does not have: no shards, no group           */
/* transaction, no identity rename, no routing revision. Setting the   */
/* property IS the hot-swap -- nvinfer reloads in place (element kept, */
/* gie-id kept, frames keep flowing) and confirms asynchronously via   */
/* its 'model-updated' signal.                                         */
/* ================================================================== */

/* Fill the caller's verdict AND log it, in the same shape the primary's control plane
 * uses for its rejects (gstnvmodelmux_control.c: "<op> rejected: <why> (<CODE>) --
 * <hint>"). Every REST answer from this stage therefore leaves a matching log line, so
 * a 4xx the operator sees over HTTP can be found verbatim in the app log with the model
 * that caused it -- the secondary used to answer silently, which made a rejected load
 * indistinguishable from one that never reached the stage.
 *
 * Requires `sgie_op` (the endpoint name) in scope: the message is useless without it,
 * and making it a hard compile error is how that stays true. */
#define SGIE_VERDICT(h, code, why, what)                                          \
  G_STMT_START {                                                                  \
    *http = (h); *err_code = (code); *reason = (why); *hint = (what);             \
    if ((h) >= 500)                                                               \
      SG_ERR ("sgie: %s FAILED: %s (%s)%s%s", sgie_op, (why) ? (why) : "?",       \
          (code) ? (code) : "INTERNAL", (what) ? " -- " : "", (what) ? (what) : ""); \
    else if ((h) >= 400)                                                          \
      SG_WARN ("sgie: %s rejected: %s (%s)%s%s", sgie_op, (why) ? (why) : "?",    \
          (code) ? (code) : "?", (what) ? " -- " : "", (what) ? (what) : "");     \
    else                                                                          \
      SG_INFO ("sgie: %s accepted (%d): %s", sgie_op, (h), (why) ? (why) : "");   \
  } G_STMT_END

/* nvinfer's OTA confirm. status==0 commits the optimistic identity swap done at
 * trigger; anything else rolls it back -- the PREVIOUS engine is still the one
 * serving, so status/routing must go back to claiming the previous version. */
static void
sgie_model_updated_cb (GstElement * infer, gint status, gchar * cfg_file,
    gpointer udata)
{
  ModelMuxSgieModel * sgie_model = (ModelMuxSgieModel *) udata;
  ModelMuxSgieBin *sgie_bin = sgie_model->owner;
  (void) infer;

  g_mutex_lock (&sgie_bin->state_lock);
  /* Only the SERVING model is ever updated in place, and every model fires this on its
   * own initial load -- so a confirm from any other element has nothing to commit. */
  if (!sgie_bin->ota_await || sgie_model != sgie_bin->serving) {
    g_mutex_unlock (&sgie_bin->state_lock);
    return;
  }
  sgie_bin->ota_await = FALSE;

  if (status == 0) {
    g_clear_pointer (&sgie_bin->ota_old_version, g_free);
    g_clear_pointer (&sgie_bin->ota_old_config, g_free);
    g_mutex_unlock (&sgie_bin->state_lock);
    SG_INFO ("sgie: OTA reload CONFIRMED for '%s@%s' (cfg='%s')", sgie_model->name,
        sgie_model->version, cfg_file ? cfg_file : "(unchanged)");
    return;
  }

  if (sgie_bin->ota_old_version) {
    g_free (sgie_model->version);
    sgie_model->version = sgie_bin->ota_old_version;
    sgie_bin->ota_old_version = NULL;
  }
  if (sgie_bin->ota_old_config) {
    g_free (sgie_model->config_file);
    sgie_model->config_file = sgie_bin->ota_old_config;
    sgie_bin->ota_old_config = NULL;
    sgie_model->cfg_gpu = modelmux_config_file_gpu (sgie_model->config_file);
  }
  g_mutex_unlock (&sgie_bin->state_lock);
  SG_ERR ("sgie: OTA reload FAILED (status=%d) -- rolled back to '%s@%s'; the "
      "previous engine keeps serving", status, sgie_model->name, sgie_model->version);
}

/* Admission. Every rejection here leaves the stage exactly as it was, so the caller
 * gets a verdict it can act on instead of a swap that half-happened. Mirrors the
 * primary path's codes so one client handles both stages. */
static gboolean
sgie_update_admit (ModelMuxSgieBin * sgie_bin, const gchar * name,
    const gchar * from_version, const gchar * version, gint * http,
    const gchar ** err_code, const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/update";   /* SGIE_VERDICT logging */
  GstState st = GST_STATE_NULL;
  gboolean awaiting;

  if (!name || !*name || !from_version || !*from_version || !version || !*version) {
    SGIE_VERDICT (400, "FIELD_MISSING", "all three are required",
        "model/update requires 'name', 'from_version' and 'version'");
    return FALSE;
  }
  if (g_strcmp0 (from_version, version) == 0) {
    SGIE_VERDICT (400, "VERSION_INVALID",
        "model/update rejected: 'version' must differ from 'from_version'",
        "pick a new, distinct version (identity must move)");
    return FALSE;
  }
  if (g_strcmp0 (name, sgie_bin->serving->name) != 0) {
    SGIE_VERDICT (404, "MODEL_NOT_IN_CATALOG",
        "this secondary stage serves a different model",
        "GET /api/v1/model/status to see which model each stage serves");
    return FALSE;
  }
  /* from_version is a COMPARE-AND-SET guard, not a resource name -> 409, matching
   * the primary's UPDATE_FROM_MISMATCH. */
  if (g_strcmp0 (from_version, sgie_bin->serving->version) != 0) {
    SGIE_VERDICT (409, "UPDATE_FROM_MISMATCH",
        "from_version does not name the version this stage is serving",
        "re-read GET /api/v1/model/status and retry");
    return FALSE;
  }
  /* The OTA RENAMES the serving model to @version. If that version is already resident
   * the stage would hold two models claiming one identity -- and a route naming it
   * could not say which. The caller wants the resident one anyway: it is already warm. */
  g_mutex_lock (&sgie_bin->state_lock);
  if (sgie_find_model (sgie_bin, name, version)) {
    g_mutex_unlock (&sgie_bin->state_lock);
    SGIE_VERDICT (409, "MODEL_ALREADY_LOADED",
        "that version is already loaded at this stage, so an update cannot take its "
        "identity", "route all streams to it instead -- it is loaded and warm");
    return FALSE;
  }
  g_mutex_unlock (&sgie_bin->state_lock);

  if (!modelmux_backend (sgie_bin->serving->type)->inplace_reload) {
    /* the CAPABILITY, not the backend name: any backend whose engine property is
     * settable on a live element supports this, and one whose checkpoints live
     * elsewhere never will. */
    SGIE_VERDICT (400, "MODEL_UPDATE_UNSUPPORTED_BACKEND",
        "this backend has no in-place update; its checkpoints are not the "
        "plugin's to swap",
        "put the new checkpoint in the backend's own repository and restart the stage");
    return FALSE;
  }

  g_mutex_lock (&sgie_bin->state_lock);
  awaiting = sgie_bin->ota_await;
  g_mutex_unlock (&sgie_bin->state_lock);
  if (awaiting) {
    /* one rollback stash per stage: a second trigger would cross-wire the confirms. */
    SGIE_VERDICT (409, "UPDATE_IN_FLIGHT",
        "another model/update on this stage is awaiting nvinfer's confirm",
        "retry once it commits or rolls back");
    return FALSE;
  }

  /* never reload an engine that is still building: nvinfer owns its own state
   * until it reaches PAUSED. */
  gst_element_get_state (sgie_bin->serving->infer, &st, NULL, 0);
  if (st < GST_STATE_PAUSED) {
    SGIE_VERDICT (409, "MODEL_WARMING", "the engine is still being built",
        "retry once GET /api/v1/model/status reports state 'serving'");
    return FALSE;
  }
  return TRUE;
}

/* Resolve the artifact the to-version should run: an explicit engine/config from the
 * request wins, else the [model-<name>-<version>] block for that version. Caller frees
 * *out_config. Returns FALSE when the version names no artifact at all. */
static gboolean
sgie_update_resolve (ModelMuxSgieBin * sgie_bin, const gchar * version,
    const gchar * config_file, const gchar * engine_file, gchar ** out_config,
    gint * http, const gchar ** err_code, const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/update";   /* SGIE_VERDICT logging */
  gboolean derive_failed = FALSE;

  *out_config = NULL;
  if (engine_file && *engine_file)
    return TRUE;                        /* engine-only swap: no config change */
  if (config_file && *config_file) {
    *out_config = g_strdup (config_file);
    return TRUE;
  }

  *out_config = modelmux_config_resolve_version_cfg (sgie_bin->config,
      sgie_bin->serving->name, version, &derive_failed);
  if (derive_failed) {
    SGIE_VERDICT (500, "ARTIFACT_DERIVE_FAILED",
        "the version's engine/gpu override could not be derived",
        "check the [model-<name>-<version>] block and the artifact paths it names");
    return FALSE;
  }
  if (!*out_config) {
    SGIE_VERDICT (400, "ARTIFACT_MISSING",
        "the to-version names no artifact to swap in",
        "send 'engine_file' or 'config_file', or declare a "
        "[model-<name>-<version>] block for it");
    return FALSE;
  }
  return TRUE;
}

/* An in-place config reload keeps the element -- and therefore its device. A config
 * that pins elsewhere would leave the reported placement describing a device the
 * stage does not run on. */
static gboolean
sgie_update_check_device (ModelMuxSgieBin * sgie_bin, const gchar * new_cfg,
    gint * http, const gchar ** err_code, const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/update";   /* SGIE_VERDICT logging */
  gint ng = modelmux_config_file_gpu (new_cfg);

  if (ng == sgie_bin->serving->cfg_gpu)
    return TRUE;
  SGIE_VERDICT (409, "UPDATE_DEVICE_MOVE",
      "the new config would move this stage to another gpu; an in-place swap "
      "keeps the element and cannot move it",
      "pin the new config to the same gpu, or restart the pipeline to relocate it");
  return FALSE;
}

/* Trigger the swap and publish the new identity optimistically (readers must see one
 * coherent view); sgie_model_updated_cb either keeps it or rolls it back. */
gboolean
modelmux_sgie_update (ModelMuxSgieBin * sgie_bin, const gchar * name,
    const gchar * from_version, const gchar * version, const gchar * config_file,
    const gchar * engine_file, gint * http, const gchar ** err_code,
    const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/update";   /* SGIE_VERDICT logging */
  ModelMuxSgieModel *serving;
  gchar *new_cfg = NULL;
  gboolean by_engine = (engine_file && *engine_file);

  if (!sgie_bin || !sgie_bin->serving) {
    SGIE_VERDICT (500, "INTERNAL", "no secondary stage to update", NULL);
    return FALSE;
  }
  serving = sgie_bin->serving;
  if (!sgie_update_admit (sgie_bin, name, from_version, version, http, err_code,
          reason, hint))
    return FALSE;
  if (!sgie_update_resolve (sgie_bin, version, config_file, engine_file, &new_cfg,
          http, err_code, reason, hint))
    return FALSE;
  if (new_cfg && !sgie_update_check_device (sgie_bin, new_cfg, http, err_code,
          reason, hint)) {
    g_free (new_cfg);
    return FALSE;
  }

  g_mutex_lock (&sgie_bin->state_lock);
  sgie_bin->ota_old_version = g_strdup (serving->version);
  sgie_bin->ota_old_config = new_cfg ? g_strdup (serving->config_file) : NULL;
  sgie_bin->ota_await = TRUE;
  g_free (serving->version);
  serving->version = g_strdup (version);
  if (new_cfg) {
    g_free (serving->config_file);
    serving->config_file = g_strdup (new_cfg);
    serving->cfg_gpu = modelmux_config_file_gpu (new_cfg);
  }
  g_mutex_unlock (&sgie_bin->state_lock);

  /* ==== the hot-swap: the backend's native on-the-fly model update ====
   *   engine given -> engine_prop      (engine-only; keeps the config's other
   *                   settings -- nvinfer's networkMode/scale/format/offsets)
   *   else config  -> config_prop      (full reload)
   *
   * Both property names come from the registry, so a backend that spells its engine
   * "tensorrt-engine" and has no config-file property at all takes the same path.
   * For a property-configured backend a config-swap means re-applying the file's
   * keys; the engine key inside it is what actually triggers the reload. */
  {
    const ModelBackendInfo *be = modelmux_backend (serving->type);

    if (by_engine)
      g_object_set (G_OBJECT (serving->infer), be->engine_prop, engine_file, NULL);
    else if (be->config_prop)
      g_object_set (G_OBJECT (serving->infer), be->config_prop, new_cfg, NULL);
    else
      modelmux_config_apply_properties (serving->infer, new_cfg, be->config_group);

    SG_INFO ("sgie: OTA reload TRIGGERED for '%s' %s@%s -> @%s (gie=%u kept): %s='%s'",
        serving->name, serving->name, from_version, version, serving->unique_id,
        by_engine ? be->engine_prop : "config", by_engine ? engine_file : new_cfg);
  }
  g_free (new_cfg);

  /* The version identity just moved, so the ROUTING CONTENT changed even though no
   * stream did -- a controller watching if_revision must lose its stale token. The
   * primary bumps here for exactly this reason (see model/update in the control
   * plane); a secondary bumping the SAME shared counter is what keeps one revision
   * meaningful across the pipeline. Bumped at TRIGGER, like the primary: the
   * identity is already published, the engine swap merely follows. */
  modelmux_sgie_bump_routing_revision (sgie_bin);

  SGIE_VERDICT (202, NULL,
      "model/update accepted; swapping -> confirmed by the backend",
      "GET /api/v1/model/status to see when it commits");
  return TRUE;
}

/* ================================================================== */
/* 6b. model/load: a SECOND model, resident and idle                   */
/*                                                                     */
/* "Loading a second model of the same template makes it AVAILABLE,    */
/* not ACTIVE" (sgie approach, constraint 3). The model is built and   */
/* warmed alongside the serving one but is NOT linked into the chain,  */
/* so it costs VRAM and changes no output until an all-streams route   */
/* makes it serve. Keeping load inert is what lets the verb mean the   */
/* same thing here as at a primary.                                    */
/* ================================================================== */

/* A gie id no model at this stage is using. The version's own declared id wins (that is
 * what makes `operate-on-gie-id` knowable from config); otherwise take the first free
 * slot above the secondary base. Two resident models MUST NOT share an id -- nvinfer
 * stamps it into every object it classifies, so a duplicate makes the meta ambiguous
 * and un-targetable. Caller holds @state_lock. */
static guint
sgie_pick_gie_id (ModelMuxSgieBin * sgie_bin, const gchar * name, const gchar * version)
{
  gint declared = modelmux_config_version_gie (sgie_bin->config, name, version);
  guint candidate;
  GList *l;

  if (declared > 0) {
    for (l = sgie_bin->models; l; l = l->next)
      if (((ModelMuxSgieModel *) l->data)->unique_id == (guint) declared) {
        SG_WARN ("sgie: '%s@%s' declares gie-unique-id %d but it is already in use at "
            "this stage -- allocating a free one instead", name, version, declared);
        declared = -1;
        break;
      }
    if (declared > 0)
      return (guint) declared;
  }

  for (candidate = MM_SGIE_GIE_ID_BASE;; candidate++) {
    gboolean used = FALSE;
    for (l = sgie_bin->models; l && !used; l = l->next)
      used = (((ModelMuxSgieModel *) l->data)->unique_id == candidate);
    if (!used)
      return candidate;
  }
}

/* The config the loaded version runs. An explicit config_file or engine_file from the
 * request wins, else the [model-<name>-<version>] block. Whatever the source, the config
 * is DERIVED so the chosen gie id is baked in -- two models sharing the catalog base
 * would otherwise share its gie-unique-id too. Caller frees *out_config. */
static gboolean
sgie_load_resolve (ModelMuxSgieBin * sgie_bin, const gchar * name, const gchar * version,
    const gchar * config_file, const gchar * engine_file, guint gie, gchar ** out_config,
    gint * http, const gchar ** err_code, const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/load";   /* SGIE_VERDICT logging */
  const ModelCatalogEntry *entry = modelmux_config_find_model (sgie_bin->config, name);
  gboolean derive_failed = FALSE;
  const gchar *base;
  gchar *versioned = NULL;

  *out_config = NULL;
  if (!entry || !entry->config_file) {
    SGIE_VERDICT (404, "MODEL_NOT_IN_CATALOG",
        "no [model-name-N] entry at this stage declares that model",
        "declare it in the stage's modelmux config, or send the load to the stage "
        "whose catalog names it");
    return FALSE;
  }

  /* POSITION before artifact: a backend this stage cannot host is refused whatever
   * the payload carries. Checked here rather than at build time because a load
   * supplies its own config_file, so the backend is only known once it is sniffed --
   * and a stage that built the wrong element would fail deep inside nvinferserver
   * with an error naming neither the model nor the position. */
  if (!modelmux_backend_at_position (entry->type, MM_PROCESS_MODE_SGIE)) {
    SGIE_VERDICT (400, "MODEL_LOAD_UNSUPPORTED_BACKEND",
        "that backend cannot run at a secondary stage",
        "run this model at the PRIMARY stage, or give the secondary a model whose "
        "config declares a backend it supports");
    return FALSE;
  }

  if (!modelmux_backend (entry->type)->engine_file && engine_file && *engine_file) {
    SGIE_VERDICT (400, "MODEL_LOAD_UNSUPPORTED_BACKEND",
        "a raw .engine does not map to nvinferserver; Triton's model repository owns "
        "the checkpoints", "put the checkpoint in the repository and send 'config_file'");
    return FALSE;
  }

  base = (config_file && *config_file) ? config_file : NULL;
  if (!base && !(engine_file && *engine_file)) {
    /* Nothing explicit -> the version must carry its own artifact, or it would be the
     * serving model under a second version number. */
    versioned = modelmux_config_resolve_version_cfg (sgie_bin->config, name, version,
        &derive_failed);
    if (derive_failed) {
      SGIE_VERDICT (500, "ARTIFACT_DERIVE_FAILED",
          "the version's engine/gpu override could not be derived",
          "check the [model-<name>-<version>] block and the artifact paths it names");
      return FALSE;
    }
    if (!versioned) {
      SGIE_VERDICT (400, "ARTIFACT_MISSING",
          "this version names no artifact of its own, so it would load a second copy "
          "of the model already serving",
          "send 'engine_file' or 'config_file', or declare a "
          "[model-<name>-<version>] block for it");
      return FALSE;
    }
    base = versioned;
  }

  *out_config = modelmux_config_derive (base ? base : entry->config_file, engine_file,
      -1 /* keep the config's device */, (gint) gie, name, version);
  g_free (versioned);
  if (!*out_config) {
    SGIE_VERDICT (500, "ARTIFACT_DERIVE_FAILED",
        "the version's config could not be derived",
        "check the base config is readable and names a [property] section");
    return FALSE;
  }
  return TRUE;
}

/* Warm on our own thread: nvinfer builds/deserializes its engine inside the state
 * change, which takes as long as it takes -- the REST caller already has its 202 and
 * polls model/status for the transition to 'loaded'. */
static gpointer
sgie_warm_thread (gpointer udata)
{
  ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) udata;
  ModelMuxSgieBin *sgie_bin = sgie_model->owner;
  gboolean ok = TRUE;

  {
    /* EVERY element of the chain, from the shared list. A converter left in NULL
     * because it was not on a hand-written list stalls the branch the instant a
     * rebind links the queue to it -- silently, since the stage that is still
     * serving keeps the pipeline running. */
    GstElement *els[SGIE_MODEL_MAX_ELEMENTS + 1];
    guint n = sgie_model_elements (sgie_model, els), i;
    for (i = 0; i < n && ok; i++)
      ok = gst_element_sync_state_with_parent (els[i]);
  }

  g_mutex_lock (&sgie_bin->state_lock);
  sgie_model->warm_failed = !ok;
  /* Cleared INSIDE the lock and before the parked route is applied: a reader that sees
   * "not warming" must already be able to see the outcome that made it so. */
  g_atomic_int_set (&sgie_model->warming, 0);
  /* A route may have arrived while this was still building. Applying it HERE is what
   * makes "reroute to a cold model" gapless: the previous model served every frame up
   * to this point, and the switch happens the moment the target can actually infer. */
  if (ok && sgie_bin->pending_serve == sgie_model) {
    sgie_bin->pending_serve = NULL;
    SG_INFO ("sgie: parked route for '%s@%s' now applying -- it is ready",
        sgie_model->name, sgie_model->version);
    sgie_rebind_arm (sgie_bin, sgie_model);
  } else if (!ok && sgie_bin->pending_serve == sgie_model) {
    sgie_bin->pending_serve = NULL;
    SG_ERR ("sgie: parked route for '%s@%s' DROPPED -- it failed to warm; the current "
        "model keeps serving", sgie_model->name, sgie_model->version);
  }
  g_mutex_unlock (&sgie_bin->state_lock);

  if (ok)
    SG_INFO ("sgie: model '%s@%s' LOADED (gie=%u gpu=%d) -- resident and idle; route "
        "all streams to it to make it serve", sgie_model->name, sgie_model->version,
        sgie_model->unique_id, sgie_model->gpu);
  else
    SG_ERR ("sgie: model '%s@%s' FAILED to warm (gie=%u) -- it stays loaded but unusable; "
        "model/unload it and retry with a valid artifact", sgie_model->name,
        sgie_model->version, sgie_model->unique_id);
  return NULL;
}

/* Build + warm a model alongside the serving one. Synchronous admission (the caller gets
 * a real verdict), asynchronous warm (an engine build must not block the REST thread). */
gboolean
modelmux_sgie_load (ModelMuxSgieBin * sgie_bin, const gchar * name, const gchar * version,
    const gchar * config_file, const gchar * engine_file, gint * http,
    const gchar ** err_code, const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/load";   /* SGIE_VERDICT logging */
  const ModelCatalogEntry *entry;
  ModelMuxSgieModel *sgie_model;
  gchar *cfg = NULL;
  guint gie;
  gchar nm[192];

  if (!sgie_bin) {
    SGIE_VERDICT (500, "INTERNAL", "no secondary stage to load into", NULL);
    return FALSE;
  }
  if (!name || !*name || !version || !*version) {
    SGIE_VERDICT (400, "FIELD_MISSING", "'name' and 'version' are both required",
        "model/load at a secondary addresses a version of a declared model");
    return FALSE;
  }

  g_mutex_lock (&sgie_bin->state_lock);
  if (sgie_find_model (sgie_bin, name, version)) {
    g_mutex_unlock (&sgie_bin->state_lock);
    SGIE_VERDICT (409, "MODEL_ALREADY_LOADED", "this version is already loaded here",
        "GET /api/v1/model/status to see it, or load a different version");
    return FALSE;
  }
  gie = sgie_pick_gie_id (sgie_bin, name, version);
  g_mutex_unlock (&sgie_bin->state_lock);

  if (!sgie_load_resolve (sgie_bin, name, version, config_file, engine_file, gie, &cfg,
          http, err_code, reason, hint))
    return FALSE;

  entry = modelmux_config_find_model (sgie_bin->config, name);
  if (!entry) {                        /* sgie_load_resolve just proved otherwise */
    g_free (cfg);
    SGIE_VERDICT (500, "INTERNAL", "the catalog entry vanished mid-load", NULL);
    return FALSE;
  }

  /* Identity is filled HERE rather than through sgie_resolve_identity: that helper
   * re-derives the config from the catalog, and the derived path is keyed on
   * (name, version) alone -- so it would rewrite the very file sgie_load_resolve
   * just produced, dropping both the request's engine_file and the gie id picked
   * above. One derive per load, and this is its result. */
  sgie_model = sgie_model_new (sgie_bin);
  sgie_model->name = g_strdup (name);
  sgie_model->version = g_strdup (version);
  sgie_model->type = entry->type;
  sgie_model->config_file = cfg;
  sgie_model->cfg_gpu = modelmux_config_file_gpu (cfg);
  sgie_model->unique_id = gie;

  g_snprintf (nm, sizeof (nm), "sgie-infer-%s-%s-gie%u", name, version, gie);
  sgie_model->infer = sgie_create_infer (sgie_model, nm);
  if (!sgie_model->infer || !sgie_create_xfer (sgie_bin, sgie_model, sgie_bin->config)) {
    sgie_model_free (sgie_model, TRUE);
    SGIE_VERDICT (500, "MODEL_LOAD_FAILED", "the inference element could not be created",
        "check the backend plugin is installed and the config names a valid device");
    return FALSE;
  }
  sgie_configure_infer (sgie_bin, sgie_model, sgie_bin->config);

  /* Into the container UNLINKED: it is resident, not serving. */
  gst_bin_add (GST_BIN (sgie_bin->bin), sgie_model->infer);
  if (sgie_model->xfer_in)
    gst_bin_add (GST_BIN (sgie_bin->bin), sgie_model->xfer_in);
  if (sgie_model->xfer_out)
    gst_bin_add (GST_BIN (sgie_bin->bin), sgie_model->xfer_out);
  sgie_model->in_bin = TRUE;
  /* Its own xfer pair still needs linking to it -- only the link to the QUEUE is what a
   * route makes, so the model arrives as a complete chain with a free head. */
  if (sgie_model->xfer_in
      && !gst_element_link (sgie_model->xfer_in, sgie_model->infer))
    SG_ERR ("sgie: '%s@%s' xfer-in -> infer link failed", name, version);
  if (sgie_model->xfer_out
      && !gst_element_link (sgie_model->infer, sgie_model->xfer_out))
    SG_ERR ("sgie: '%s@%s' infer -> xfer-out link failed", name, version);
  /* Probes on every resident model, so a rebind never has to migrate them; the tail
   * probe reports only for whichever model is serving. */
  sgie_install_probes (sgie_bin, sgie_model);
  sgie_model_park (sgie_model, TRUE);       /* resident, not serving: emit nothing */

  g_mutex_lock (&sgie_bin->state_lock);
  sgie_bin->models = g_list_append (sgie_bin->models, sgie_model);
  g_mutex_unlock (&sgie_bin->state_lock);

  /* Set BEFORE the thread exists: a route arriving between g_thread_new and the
   * thread's first instruction must still see a model that cannot yet infer. */
  g_atomic_int_set (&sgie_model->warming, 1);
  /* kept (not unreffed) so teardown joins it: see ModelMuxSgieModel.warm_thread */
  g_snprintf (nm, sizeof (nm), "sgie-warm-%s-%s", name, version);
  sgie_model->warm_thread = g_thread_new (nm, sgie_warm_thread, sgie_model);

  SG_INFO ("sgie: model/load accepted for '%s@%s' (gie=%u cfg='%s') -- warming", name,
      version, gie, sgie_model->config_file);
  SGIE_VERDICT (202, NULL,
      "model/load accepted; the model warms alongside the serving one and stays idle",
      "GET /api/v1/model/status for the transition to 'loaded', then route all streams "
      "to it to make it serve");
  return TRUE;
}

/* ================================================================== */
/* 6c. model/routing: rebind WHICH resident model serves               */
/*                                                                     */
/* "One form covers both directions. Point it at v2 and v2 takes over; */
/* point it back at v1 and the change is undone" (sgie approach, §5).  */
/* There are no lanes here, so promote and swap are the SAME call --   */
/* both are defined by the shadow field at a primary, and a secondary  */
/* has no shadow. Reversal is instant because the displaced model      */
/* stays loaded and warm.                                              */
/* ================================================================== */

/* Anything a non-serving model emits goes nowhere -- only the serving chain is ghosted
 * out of the bin. Dropping at its tail keeps that a no-op instead of a NOT_LINKED push
 * (which nvinfer reports as an internal data stream error). */
static GstPadProbeReturn
sgie_park_drop_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  (void) pad; (void) info; (void) udata;
  return GST_PAD_PROBE_DROP;
}

/* @parked TRUE = this model is resident but not serving. Idempotent. */
static void
sgie_model_park (ModelMuxSgieModel * sgie_model, gboolean parked)
{
  GstPad *p = gst_element_get_static_pad (sgie_model_tail (sgie_model), "src");

  if (!p)
    return;
  if (parked && !sgie_model->park_probe)
    sgie_model->park_probe = gst_pad_add_probe (p,
        GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_BUFFER_LIST,
        sgie_park_drop_probe, NULL, NULL);
  else if (!parked && sgie_model->park_probe) {
    gst_pad_remove_probe (p, sgie_model->park_probe);
    sgie_model->park_probe = 0;
  }
  gst_object_unref (p);
}

/* The switch itself, from a BLOCK probe on the queue src: no buffer can enter either
 * chain while this runs. Order matters -- the displaced model is parked BEFORE the
 * ghost stops pointing at it, so the batch still inside its nvinfer is dropped rather
 * than pushed at a pad that no longer has a peer. */
static GstPadProbeReturn
sgie_rebind_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  ModelMuxSgieBin *sgie_bin = (ModelMuxSgieBin *) udata;
  ModelMuxSgieModel *from, *to;
  GstPad *tail_src;
  (void) pad; (void) info;

  g_mutex_lock (&sgie_bin->state_lock);
  to = sgie_bin->rebind_to;
  from = sgie_bin->serving;
  sgie_bin->rebind_to = NULL;
  if (!to || to == from) {                 /* superseded or already applied */
    g_mutex_unlock (&sgie_bin->state_lock);
    return GST_PAD_PROBE_REMOVE;
  }

  sgie_model_park (from, TRUE);            /* before the ghost moves -- see above */
  gst_element_unlink (sgie_bin->queue, sgie_model_head (from));

  if (!gst_element_link (sgie_bin->queue, sgie_model_head (to))) {
    /* Put it back: a stage with nothing linked would silently stop classifying. */
    gst_element_link (sgie_bin->queue, sgie_model_head (from));
    sgie_model_park (from, FALSE);
    g_mutex_unlock (&sgie_bin->state_lock);
    SG_ERR ("sgie: rebind to '%s@%s' FAILED to link -- '%s@%s' keeps serving",
        to->name, to->version, from->name, from->version);
    return GST_PAD_PROBE_REMOVE;
  }

  sgie_model_park (to, FALSE);
  tail_src = gst_element_get_static_pad (sgie_model_tail (to), "src");
  gst_ghost_pad_set_target (GST_GHOST_PAD (sgie_bin->src_ghost), tail_src);
  gst_object_unref (tail_src);

  /* atomic so sgie_out_probe, which runs on the streaming thread without this lock,
   * never sees a half-applied serving pointer */
  g_atomic_pointer_set (&sgie_bin->serving, to);
  g_atomic_int_set (&sgie_bin->live_streams, 0);   /* recounted from the next batch */
  g_mutex_unlock (&sgie_bin->state_lock);

  SG_INFO ("sgie: SERVING '%s@%s' (gie=%u) -- displaced '%s@%s' stays loaded and warm; "
      "the same call naming it restores it", to->name, to->version, to->unique_id,
      from->name, from->version);
  return GST_PAD_PROBE_REMOVE;
}

/* Arm the switch. Caller holds @state_lock. */
static void
sgie_rebind_arm (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * to)
{
  GstPad *p;

  sgie_bin->rebind_to = to;
  p = gst_element_get_static_pad (sgie_bin->queue, "src");
  if (!p) {
    sgie_bin->rebind_to = NULL;
    SG_ERR ("sgie: rebind to '%s@%s' FAILED -- no queue src pad", to->name, to->version);
    return;
  }
  gst_pad_add_probe (p, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM, sgie_rebind_probe,
      sgie_bin, NULL);
  gst_object_unref (p);
}

/* model/routing at a secondary: make an ALREADY-RESIDENT model the serving one.
 * The scope/shape checks (streams must be "all", no shadow, no camera list) are the
 * caller's -- they are payload shape, decided before the query reaches this stage. */
gboolean
modelmux_sgie_route (ModelMuxSgieBin * sgie_bin, const gchar * name,
    const gchar * version, gint * http, const gchar ** err_code,
    const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/routing";   /* SGIE_VERDICT logging */
  ModelMuxSgieModel *target;
  const gchar *state;

  if (!sgie_bin) {
    SGIE_VERDICT (500, "INTERNAL", "no secondary stage to route", NULL);
    return FALSE;
  }
  if (!name || !*name || !version || !*version) {
    SGIE_VERDICT (400, "FIELD_MISSING", "'model' must name both 'name' and 'version'",
        "a reroute at a secondary names the resident model that should serve");
    return FALSE;
  }

  g_mutex_lock (&sgie_bin->state_lock);
  target = sgie_find_model (sgie_bin, name, version);
  if (!target) {
    g_mutex_unlock (&sgie_bin->state_lock);
    /* Deliberately NOT a load: routing never ingests an artifact, at either stage. */
    SGIE_VERDICT (404, "MODEL_NOT_LOADED",
        "that version is not resident at this stage, and routing never loads",
        "POST /api/v1/model/load it first, then route all streams to it");
    return FALSE;
  }
  if (target == sgie_bin->serving) {
    g_mutex_unlock (&sgie_bin->state_lock);
    SGIE_VERDICT (200, NULL, "that model is already serving every stream at this stage",
        "GET /api/v1/model/status to confirm");
    return TRUE;
  }
  if (target->warm_failed) {
    g_mutex_unlock (&sgie_bin->state_lock);
    SGIE_VERDICT (409, "MODEL_NOT_READY",
        "that model failed to warm, so it cannot be made to serve",
        "model/unload it and load it again with a valid artifact");
    return FALSE;
  }

  /* Still building its engine: PARK the route rather than refuse it. The current model
   * keeps serving until the target is READY, so a reroute to a cold model costs no
   * inference gap -- the same deferral the primary applies to a warming target. */
  state = sgie_state_str (target);
  if (g_strcmp0 (state, "warming") == 0) {
    sgie_bin->pending_serve = target;
    g_mutex_unlock (&sgie_bin->state_lock);
    SG_INFO ("sgie: route to '%s@%s' PARKED -- still warming; '%s' keeps serving until "
        "it is ready", name, version, modelmux_sgie_model_name (sgie_bin));
    SGIE_VERDICT (202, NULL,
        "model/routing accepted; the target is still warming, so the current model "
        "keeps serving until it is ready",
        "GET /api/v1/model/status for the hand-over");
    return TRUE;
  }

  sgie_rebind_arm (sgie_bin, target);
  g_mutex_unlock (&sgie_bin->state_lock);

  SGIE_VERDICT (202, NULL,
      "model/routing accepted; rebinding every stream at this stage to that model",
      "GET /api/v1/model/status to see it serving");
  return TRUE;
}

/* ================================================================== */
/* 6d. model/unload: drop a RESIDENT model and reclaim its VRAM        */
/*                                                                     */
/* The mirror of load. A secondary always has a serving model, so the  */
/* one thing unload may never do is leave the stage empty -- but that  */
/* is a constraint on the SERVING entry, not on the stage: an idle     */
/* resident model is exactly the VRAM a rollout wants back once the    */
/* version it was hedging against is settled.                          */
/* ================================================================== */

gboolean
modelmux_sgie_unload (ModelMuxSgieBin * sgie_bin, const gchar * name,
    const gchar * version, gint * http, const gchar ** err_code,
    const gchar ** reason, const gchar ** hint)
{
  static const gchar *const sgie_op = "model/unload";   /* SGIE_VERDICT logging */
  ModelMuxSgieModel *target = NULL;
  gchar *gone_version;
  guint gone_gie;

  if (!sgie_bin) {
    SGIE_VERDICT (500, "INTERNAL", "no secondary stage to unload from", NULL);
    return FALSE;
  }
  if (!name || !*name) {
    SGIE_VERDICT (400, "FIELD_MISSING", "'name' is required",
        "model/unload addresses a model resident at this stage");
    return FALSE;
  }

  g_mutex_lock (&sgie_bin->state_lock);
  if (version && *version) {
    target = sgie_find_model (sgie_bin, name, version);
  } else {
    /* A version-less unload is only unambiguous while the name has ONE version here.
     * Guessing which of several to drop would silently free the wrong engine. */
    GList *l;
    guint n = 0;
    for (l = sgie_bin->models; l; l = l->next) {
      ModelMuxSgieModel *m = (ModelMuxSgieModel *) l->data;
      if (g_strcmp0 (m->name, name) == 0) {
        n++;
        target = m;
      }
    }
    if (n > 1) {
      g_mutex_unlock (&sgie_bin->state_lock);
      SGIE_VERDICT (400, "VERSION_INVALID",
          "more than one version of that model is resident here, so 'version' is "
          "required", "GET /api/v1/model/status to see which versions are loaded");
      return FALSE;
    }
  }

  if (!target) {
    g_mutex_unlock (&sgie_bin->state_lock);
    SGIE_VERDICT (404, "MODEL_NOT_LOADED", "that model is not resident at this stage",
        "GET /api/v1/model/status to see what this stage has loaded");
    return FALSE;
  }
  if (target == sgie_bin->serving) {
    g_mutex_unlock (&sgie_bin->state_lock);
    /* The ONE thing unload may not do here. Note the fix is a reroute, not an update:
     * the caller must first make another resident model serve. */
    SGIE_VERDICT (409, "MODEL_IN_USE",
        "that model is serving every stream at this stage, and a secondary cannot run "
        "without one", "model/load another version and route all streams to it, then "
        "unload this one");
    return FALSE;
  }
  if (sgie_bin->pending_serve == target) {
    g_mutex_unlock (&sgie_bin->state_lock);
    SGIE_VERDICT (409, "MODEL_IN_USE",
        "a parked route names that model: it is about to serve as soon as it is ready",
        "route back to the model currently serving first, then unload it");
    return FALSE;
  }
  if (!target->warm_failed
      && g_strcmp0 (sgie_state_str (target), "warming") == 0) {
    g_mutex_unlock (&sgie_bin->state_lock);
    /* Its engine build owns these elements on another thread; tearing them down under
     * it is a use-after-free, and blocking the REST thread on a build is not an answer. */
    SGIE_VERDICT (409, "MODEL_WARMING",
        "that model is still building its engine and cannot be torn down mid-build",
        "GET /api/v1/model/status until it reports 'loaded', then unload it");
    return FALSE;
  }

  /* Detach from the stage UNDER the lock -- after this no reader can reach it -- then
   * tear the elements down outside it (state changes must not hold a lock the
   * streaming thread's probes also take). */
  sgie_bin->models = g_list_remove (sgie_bin->models, target);
  gone_version = g_strdup (target->version);
  gone_gie = target->unique_id;
  g_mutex_unlock (&sgie_bin->state_lock);

  /* The warm thread has finished (refused above while it had not), but it is still
   * joinable and it touches these elements -- join before any of them changes state. */
  if (target->warm_thread) {
    g_thread_join (target->warm_thread);
    target->warm_thread = NULL;
  }

  {
    GstElement *els[SGIE_MODEL_MAX_ELEMENTS + 1];
    guint n = sgie_model_elements (target, els), i;
    for (i = 0; i < n; i++)
      gst_element_set_state (els[i], GST_STATE_NULL);
    if (target->in_bin)
      /* drops the container's ref -- gst_bin_remove unlinks the pads for us */
      for (i = 0; i < n; i++)
        gst_bin_remove (GST_BIN (sgie_bin->bin), els[i]);
  }
  sgie_model_free (target, FALSE);       /* the bin already dropped the elements */

  SG_INFO ("sgie: model '%s@%s' UNLOADED (gie=%u) -- its engine is released; '%s@%s' "
      "keeps serving", name, gone_version, gone_gie,
      modelmux_sgie_model_name (sgie_bin), modelmux_sgie_model_version (sgie_bin));
  g_free (gone_version);

  SGIE_VERDICT (202, NULL, "model/unload accepted; the model is released",
      "GET /api/v1/model/status to confirm it is gone");
  return TRUE;
}

/* ================================================================== */
/* 7. GET plane: model/status + model/routing                          */
/* ================================================================== */

/* Lifecycle of one model. The WARMING flag is checked first and is authoritative: it is
 * owned by the warm thread, whereas the element's GstState only reports where the state
 * machine has got to. nvinfer builds its engine inside READY->PAUSED, and a zero-timeout
 * gst_element_get_state() returns the current state with the change still PENDING, so a
 * model minutes into a TensorRT build can already read PAUSED. The GstState check stays
 * below it for the models that never had a warm thread -- the config-declared one, which
 * the pipeline brings up itself. */
static const gchar *
sgie_state_str (ModelMuxSgieModel * sgie_model)
{
  GstState st = GST_STATE_NULL;

  if (!sgie_model || !sgie_model->infer)
    return "unloaded";
  if (sgie_model->warm_failed)
    return "failed";
  if (g_atomic_int_get (&sgie_model->warming))
    return "warming";
  gst_element_get_state (sgie_model->infer, &st, NULL, 0);
  if (st < GST_STATE_PAUSED)
    return "warming";
  /* PAUSED+ means the engine is built. Whether it INFERS depends on whether it is the
   * one linked into the chain; a resident model is "loaded" until a route makes it
   * serve (sgie approach, constraint 3). */
  return (sgie_model->owner && sgie_model == sgie_model->owner->serving)
      ? "serving" : "loaded";
}

/* Minimal JSON string escape (the identifiers here are config-supplied names/paths). */
static gchar *
sgie_json_escape (const gchar * v)
{
  GString *o = g_string_new (NULL);
  const gchar *p;

  for (p = v ? v : ""; *p; p++) {
    if (*p == '"' || *p == '\\')
      g_string_append_printf (o, "\\%c", *p);
    else if ((guchar) * p < 0x20)
      g_string_append_printf (o, "\\u%04x", *p);
    else
      g_string_append_c (o, *p);
  }
  return g_string_free (o, FALSE);
}

/* model/status for a secondary, in the same v1 shape the primary emits so a client
 * parses both with one reader. Two stage-specific facts:
 *   process_mode:2          -- tags WHICH stage answered, so the host can merge the
 *                              per-stage responses without inventing a stage id.
 * streams[] is empty: a secondary has no per-stream routing, so there is nothing to
 * list there. `serving.streams` uses the SAME key the primary does -- the live count
 * comes from num_frames_in_batch (one frame per stream), so a client needs no
 * secondary-specific shape.
 * @f_model/@f_version filter to nothing when they name another model, keeping the
 * "a filtered view contains only the filter" contract. */
/* One versions[] entry. Caller holds @state_lock. */
static void
sgie_status_append_version (ModelMuxSgieBin * sgie_bin, ModelMuxSgieModel * sgie_model,
    GString * s)
{
  gboolean is_serving = (sgie_model == sgie_bin->serving);
  /* only the serving model sees frames, so only it reports a stream count */
  gint nstreams = is_serving ? g_atomic_int_get (&sgie_bin->live_streams) : 0;
  const gchar *state = sgie_state_str (sgie_model);
  gchar *engine = NULL;
  gchar *ever, *ecfg, *eeng;

  /* provenance: the engine the backend actually resolved, read from whichever
   * property carries one (NULL for a backend whose checkpoints live elsewhere) */
  if (sgie_model->infer && modelmux_backend (sgie_model->type)->engine_prop)
    g_object_get (G_OBJECT (sgie_model->infer),
        modelmux_backend (sgie_model->type)->engine_prop, &engine, NULL);
  ever = sgie_json_escape (sgie_model->version);
  ecfg = sgie_json_escape (sgie_model->config_file);
  eeng = sgie_json_escape (engine);

  g_string_append_printf (s,
      "{\"version\":\"%s\",\"state\":\"%s\",\"default_for\":[%s],"
      "\"serving\":{\"streams\":%d},\"config\":\"%s\",\"instances\":[{\"id\":0,",
      ever, state, is_serving ? "\"model\"" : "", nstreams, ecfg);
  if (modelmux_backend (sgie_model->type)->owns_placement)
    g_string_append_printf (s,
        "\"backend\":\"nvinferserver\",\"config\":\"%s\",\"gie_id\":%u,",
        ecfg, sgie_model->unique_id);
  else
    g_string_append_printf (s, "\"gpu\":%d,\"gie_id\":%u,\"engine\":\"%s\",",
        sgie_model->gpu, sgie_model->unique_id, eeng);
  g_string_append_printf (s, "\"state\":\"%s\",\"streams\":%d,\"capacity\":%u",
      state, nstreams, sgie_bin->batch_size);
  /* perf, when enabled: measured at the STAGE boundary, so it is comparable with the
   * primary's numbers and independent of which backend runs inside. The probes sit on
   * the SERVING chain, so an idle model has nothing to report.
   *
   * WHAT infer_latency_ms SPANS here: queue out -> inference src -- this STAGE's whole
   * contribution to pipeline latency (any [nvdsxfer] included), not the backend's pure
   * inference time. Same field name as the primary's and the same kind of measurement
   * (a boundary, not an element), so the two are comparable; the schema documents the
   * span, since the name alone is narrower than what is measured. */
  if (sgie_bin->perf && is_serving) {
    gdouble fps = 0, lat = 0;
    g_mutex_lock (&sgie_bin->perf_lock);
    modelmux_perf_get_model (sgie_bin->perf, &fps, &lat);
    g_mutex_unlock (&sgie_bin->perf_lock);
    g_string_append_printf (s, ",\"fps\":%.2f,\"infer_latency_ms\":%.2f", fps, lat);
  }
  g_string_append (s, "}]}");

  g_free (ever);
  g_free (ecfg);
  g_free (eeng);
  g_free (engine);
}

gchar *
modelmux_sgie_status_json (ModelMuxSgieBin * sgie_bin, const gchar * f_model,
    const gchar * f_version)
{
  GString *s;
  GList *l, *k;
  GList *done = NULL;            /* names already emitted (borrowed pointers) */
  gboolean any = FALSE;

  if (!sgie_bin || !sgie_bin->serving)
    return g_strdup ("{\"routing_revision\":0,\"models\":[],\"streams\":[]}");

  /* the PIPELINE's revision, not a per-stage one -- see the merge note in
   * gstnvmodelmux_model_status.c: the merged body keeps the upstream stage's
   * scalar, and that is only the right answer because it is the same number. */
  s = g_string_new (NULL);
  g_string_append_printf (s, "{\"routing_revision\":%" G_GUINT64_FORMAT
      ",\"models\":[", modelmux_rev_shared_get (sgie_bin->rev_shared));

  /* the whole walk under state_lock: the list, the serving pointer and every identity
   * it reads are all mutable (a load appends, an OTA swaps version/config live). */
  g_mutex_lock (&sgie_bin->state_lock);
  for (l = sgie_bin->models; l; l = l->next) {
    ModelMuxSgieModel *first = (ModelMuxSgieModel *) l->data;
    gboolean name_done = FALSE;
    gboolean opened = FALSE;
    gchar *enm;

    if (f_model && *f_model && g_strcmp0 (f_model, first->name) != 0)
      continue;
    for (k = done; k && !name_done; k = k->next)
      name_done = (g_strcmp0 ((const gchar *) k->data, first->name) == 0);
    if (name_done)
      continue;                  /* already emitted with its sibling versions */
    done = g_list_prepend (done, first->name);

    enm = sgie_json_escape (first->name);
    /* every version of this name, so a client reads one entry per model */
    for (k = sgie_bin->models; k; k = k->next) {
      ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) k->data;
      if (g_strcmp0 (sgie_model->name, first->name) != 0)
        continue;
      if (f_version && *f_version && g_strcmp0 (f_version, sgie_model->version) != 0)
        continue;
      if (!opened) {
        g_string_append_printf (s, "%s{\"name\":\"%s\",\"process_mode\":2,\"versions\":[",
            any ? "," : "", enm);
        opened = TRUE;
        any = TRUE;
      } else {
        g_string_append_c (s, ',');
      }
      sgie_status_append_version (sgie_bin, sgie_model, s);
    }
    if (opened)
      g_string_append (s, "]}");
    g_free (enm);
  }
  g_mutex_unlock (&sgie_bin->state_lock);
  g_list_free (done);

  g_string_append (s, "],\"streams\":[]}");
  return g_string_free (s, FALSE);
}

/* model/routing GET for a secondary. routes[] stays a PRIMARY-only plane -- routing is
 * per-stream and a secondary has none, which is why POST answers 409
 * ROUTING_NOT_APPLICABLE. Instead the stage lists the model it serves under
 * secondary_models[], reusing the {name, version} shape of default.model so no new
 * object type enters the schema. Purely additive: a client that ignores the key sees
 * the routing plane exactly as it was before secondary support. */
gchar *
modelmux_sgie_route_json (ModelMuxSgieBin * sgie_bin)
{
  gchar *enm, *ever, *out;
  gint gpu;

  if (!sgie_bin || !sgie_bin->serving)
    return g_strdup ("{\"routes\":[],\"routing_revision\":0}");

  g_mutex_lock (&sgie_bin->state_lock);        /* an OTA swaps name/version live */
  enm = sgie_json_escape (sgie_bin->serving->name);
  ever = sgie_json_escape (sgie_bin->serving->version);
  gpu = sgie_bin->serving->gpu;                /* fixed at creation; read with the rest */
  g_mutex_unlock (&sgie_bin->state_lock);

  /* {gpu, name, version} -- the same keys, in the same order, that the primary's
   * routes[].model carries, so a client reads a model ref the same way wherever it
   * appears. The device matters here for the same reason it does at a primary: a
   * secondary can be pinned to another GPU (see sgie_create_xfer), and "which model"
   * is only half an answer without "running where". */
  out = g_strdup_printf ("{\"routes\":[],\"secondary_models\":"
      "[{\"gpu\":%d,\"name\":\"%s\",\"version\":\"%s\"}],\"routing_revision\":%"
      G_GUINT64_FORMAT "}", gpu, enm, ever,
      modelmux_rev_shared_get (sgie_bin->rev_shared));
  g_free (enm);
  g_free (ever);
  return out;
}

void
modelmux_sgie_set_rev_shared (ModelMuxSgieBin * sgie_bin, ModelMuxRevShared * rev)
{
  ModelMuxRevShared *old;

  if (!sgie_bin)
    return;
  g_mutex_lock (&sgie_bin->state_lock);
  old = sgie_bin->rev_shared;
  sgie_bin->rev_shared = modelmux_rev_shared_ref (rev);
  g_mutex_unlock (&sgie_bin->state_lock);
  modelmux_rev_shared_unref (old);
}

void
modelmux_sgie_set_overlay_slot (ModelMuxSgieBin * sgie_bin, guint slot)
{
  if (sgie_bin)
    sgie_bin->overlay_slot = slot;
}

void
modelmux_sgie_bump_routing_revision (ModelMuxSgieBin * sgie_bin)
{
  if (!sgie_bin)
    return;
  /* the counter has its own lock, so the read of the pointer is all that needs
   * state_lock -- and it must not be held while another stage may be bumping. */
  g_mutex_lock (&sgie_bin->state_lock);
  modelmux_rev_shared_bump (sgie_bin->rev_shared);
  g_mutex_unlock (&sgie_bin->state_lock);
}

/* ================================================================== */
/* Public API                                                          */
/* ================================================================== */

void
modelmux_sgie_free (ModelMuxSgieBin * sgie_bin)
{
  GList *l;

  if (!sgie_bin)
    return;
  /* BEFORE the bin goes to NULL: a warm thread in flight is driving these elements'
   * state, so let it finish rather than race the teardown for them. */
  for (l = sgie_bin->models; l; l = l->next) {
    ModelMuxSgieModel *sgie_model = (ModelMuxSgieModel *) l->data;
    if (sgie_model->warm_thread) {
      g_thread_join (sgie_model->warm_thread);
      sgie_model->warm_thread = NULL;
    }
  }
  if (sgie_bin->bin) {
    GstObject *parent = gst_object_get_parent (GST_OBJECT (sgie_bin->bin));
    gst_element_set_state (sgie_bin->bin, GST_STATE_NULL);
    if (parent) {
      gst_bin_remove (GST_BIN (parent), sgie_bin->bin);   /* drops the container's ref */
      gst_object_unref (parent);
    } else {
      gst_object_unref (sgie_bin->bin);
    }
  }
  /* the elements went to NULL with the container above and are owned by it */
  g_list_free_full (sgie_bin->models, (GDestroyNotify) sgie_model_free_cb);
  sgie_bin->serving = NULL;
  if (sgie_bin->perf_timer_id)
    g_source_remove (sgie_bin->perf_timer_id);
  if (sgie_bin->perf) {
    modelmux_perf_free (sgie_bin->perf);
    g_mutex_clear (&sgie_bin->perf_lock);
  }
  g_free (sgie_bin->ota_old_version);
  g_free (sgie_bin->ota_old_config);
  /* the counter outlives this stage if the primary (or the cached context) still
   * references it -- we only drop OUR ref. */
  modelmux_rev_shared_unref (sgie_bin->rev_shared);
  g_mutex_clear (&sgie_bin->state_lock);
  g_free (sgie_bin);
}

ModelMuxSgieBin *
modelmux_sgie_new (GstElement * parent, const gchar * name,
    const ModelMuxConfig * config)
{
  ModelMuxSgieBin *sgie_bin = g_new0 (ModelMuxSgieBin, 1);
  ModelMuxSgieModel *serving = NULL;
  gchar nm[192];

  sgie_bin->config = config;      /* borrowed: outlives us (owned by the element) */
  sgie_bin->perf_on = config->attach_perf_metric;
  g_mutex_init (&sgie_bin->state_lock);

  serving = sgie_resolve_serving (sgie_bin, config);
  if (!serving)
    goto failed;

  g_snprintf (nm, sizeof (nm), "%s-sgie-%s", name ? name : "nvmodelmux", serving->name);
  sgie_bin->bin = gst_bin_new (nm);
  g_snprintf (nm, sizeof (nm), "sgie-infer-%s-%s-gie%u", serving->name, serving->version,
      serving->unique_id);
  serving->infer = sgie_create_infer (serving, nm);
  if (!sgie_bin->bin || !serving->infer || !sgie_create_head (sgie_bin, serving))
    goto failed;
  if (!sgie_create_xfer (sgie_bin, serving, config))
    goto failed;

  sgie_configure_infer (sgie_bin, serving, config);
  sgie_install_probes (sgie_bin, serving);
  if (!sgie_assemble (sgie_bin, serving))
    goto failed;

  /* published only once the chain is whole: every reader takes the serving model as
   * the stage's identity. */
  sgie_bin->models = g_list_append (sgie_bin->models, serving);
  sgie_bin->serving = serving;

  gst_bin_add (GST_BIN (parent), sgie_bin->bin);
  SG_INFO ("- SGIE chain created  (model='%s@%s' backend=%s batch=%u gpu=%d%s)",
      serving->name, serving->version,
      modelmux_backend (serving->type)->element,
      sgie_bin->batch_size, serving->gpu, serving->xfer_in ? " +nvdsxfer" : "");
  return sgie_bin;

failed:
  /* Until sgie_assemble hands them to the container bin the elements are floating, so
   * unref them here or they leak; AFTER it the bin owns them and unreffing would be one
   * too many (the bin drops its own refs when it is freed below). */
  if (serving && !serving->in_bin && sgie_bin->queue) {
    gst_object_unref (sgie_bin->queue);
    sgie_bin->queue = NULL;
  }
  sgie_model_free (serving, serving && !serving->in_bin);
  modelmux_sgie_free (sgie_bin);
  return NULL;
}

GstPad *
modelmux_sgie_get_sink (ModelMuxSgieBin * sgie_bin)
{
  return sgie_bin ? sgie_bin->sink_ghost : NULL;
}

GstPad *
modelmux_sgie_get_src (ModelMuxSgieBin * sgie_bin)
{
  return sgie_bin ? sgie_bin->src_ghost : NULL;
}

GstElement *
modelmux_sgie_get_infer (ModelMuxSgieBin * sgie_bin)
{
  return (sgie_bin && sgie_bin->serving) ? sgie_bin->serving->infer : NULL;
}

/* Display facts for the MM_DEBUG models_table, one resident model per @idx. One call so
 * the debug dump never reaches into the struct. *engine is newly-allocated (caller
 * frees); every other out-pointer borrows storage owned by the bin. FALSE means @idx is
 * past the last model, which is how the caller ends its walk. */
gboolean
modelmux_sgie_get_info (ModelMuxSgieBin * sgie_bin, guint idx, const gchar ** name,
    const gchar ** version, const gchar ** backend, guint * gie, gint * gpu,
    guint * streams, const gchar ** state, gboolean * serving, gchar ** engine)
{
  ModelMuxSgieModel *sgie_model;

  if (!sgie_bin)
    return FALSE;

  /* the list is mutable (a load appends) and an OTA swaps name/version live, so the
   * lookup AND every field read from it happen under the one lock. */
  g_mutex_lock (&sgie_bin->state_lock);
  sgie_model = (ModelMuxSgieModel *) g_list_nth_data (sgie_bin->models, idx);
  if (!sgie_model) {
    g_mutex_unlock (&sgie_bin->state_lock);
    return FALSE;
  }
  *name = sgie_model->name;
  *version = sgie_model->version;
  *backend = modelmux_backend (sgie_model->type)->element;
  *gie = sgie_model->unique_id;
  *gpu = sgie_model->gpu;
  *serving = (sgie_model == sgie_bin->serving);
  /* only the serving model is linked into the chain, so only it sees frames */
  *streams = *serving ? (guint) g_atomic_int_get (&sgie_bin->live_streams) : 0;
  *state = sgie_state_str (sgie_model);
  g_mutex_unlock (&sgie_bin->state_lock);

  /* deliberately outside the lock: `type` and `infer` are fixed at creation, and
   * g_object_get takes nvinfer's own lock -- nesting the two would order them. */
  *engine = NULL;
  if (sgie_model->infer && modelmux_backend (sgie_model->type)->engine_prop)
    g_object_get (G_OBJECT (sgie_model->infer),
        modelmux_backend (sgie_model->type)->engine_prop, engine, NULL);
  return TRUE;
}

const gchar *
modelmux_sgie_model_name (ModelMuxSgieBin * sgie_bin)
{
  return (sgie_bin && sgie_bin->serving) ? sgie_bin->serving->name : NULL;
}

const gchar *
modelmux_sgie_model_version (ModelMuxSgieBin * sgie_bin)
{
  return (sgie_bin && sgie_bin->serving) ? sgie_bin->serving->version : NULL;
}
