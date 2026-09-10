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
 * gstnvmodelmux.c
 * ========================
 * GObject element shell for the nvmodelmux plugin.
 *
 * This file is deliberately THIN: it owns no inference logic. It only
 *   (1) exposes configuration as GObject properties,
 *   (2) materialises the reusable ModelMuxBin (gstnvmodelmux_bin.c)
 *       lazily at the NULL->READY transition, wiring its ghost pads to ours, and
 *   (3) exposes runtime control as ACTION SIGNALS that forward to the bin's API.
 *
 * Keeping the shell separate from the bin keeps both reusable: the bin can be
 * embedded directly (as the sample app does) and the element can wrap any future
 * bin revision unchanged.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include "gstnvmodelmux.h"

/* The plugin's debug category. DEFINED here (one translation unit); bin.c and
 * config.c reference it via GST_DEBUG_CATEGORY_EXTERN, and plugin_init registers
 * it with GST_DEBUG_CATEGORY_INIT. All logging obeys GST_DEBUG. */
GST_DEBUG_CATEGORY (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat

/* Sentinel source-id for the "update-routing" signal meaning "all streams". */
#define MM_ALL_STREAMS  G_MAXUINT

/* ------------------------------------------------------------------ */
/* Pad templates                                                       */
/*                                                                     */
/* One batched NVMM input, one batched NVMM output -- the element is a  */
/* drop-in replacement for an nvinfer in a batched DeepStream graph.    */
/* ------------------------------------------------------------------ */
#define MM_PAD_CAPS "video/x-raw(memory:NVMM)"

static GstStaticPadTemplate gst_modelmux_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (MM_PAD_CAPS));

static GstStaticPadTemplate gst_modelmux_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (MM_PAD_CAPS));

#define DEFAULT_ENABLE                 TRUE
#define DEFAULT_BATCH_SIZE             1
#define DEFAULT_PER_MODEL_BATCH_SIZE   0      /* 0 => derive from batch-size */
#define DEFAULT_UNIFIED_BATCH          TRUE
#define DEFAULT_WAIT_FOR_DEFAULT_MODELS TRUE
/* on by default: when a model goes idle as a lone redundant warm copy, move it
 * into the shared limbo pool so either role reuses it without a reload. */
#define DEFAULT_COMPACT_IDLE_MODELS_ACROSS_POOLS  TRUE
#define DEFAULT_MODEL_SHARDING          TRUE

/* ------------------------------------------------------------------ */
/* Action signals (runtime control plane)                              */
/* ------------------------------------------------------------------ */
enum
{
  SIGNAL_LOAD_MODEL,
  SIGNAL_UNLOAD_MODEL,
  SIGNAL_RELOAD_MODEL,
  SIGNAL_ATTACH_STREAM,
  SIGNAL_DETACH_STREAM,
  SIGNAL_UPDATE_ROUTING,
  SIGNAL_GET_STATUS,
  SIGNAL_LAST
};
static guint gst_modelmux_signals[SIGNAL_LAST] = { 0 };

#define gst_modelmux_parent_class parent_class
G_DEFINE_TYPE (GstNvModelMux, gst_modelmux, GST_TYPE_BIN);

/* ================================================================== */
/* Small helpers                                                       */
/* ================================================================== */

/* Replace a heap string property: free old, dup new (NULL/"" -> NULL). */
static void
set_str (gchar ** slot, const gchar * v)
{
  g_free (*slot);
  *slot = (v && *v) ? g_strdup (v) : NULL;
}

/* Normalise an empty string to NULL (the bin/config API treat NULL as "unset"). */
static inline const gchar *
null_if_empty (const gchar * s)
{
  return (s && *s) ? s : NULL;
}

/* ================================================================== */
/* Build / teardown of the live sub-graph                             */
/* ================================================================== */

/* Assemble @self->config from the config-file (if any) overlaid with the
 * individual GObject properties (properties always win), then materialise the
 * sub-graph: either the full ModelMuxBin or, when disabled, a transparent
 * identity passthrough. Called once at NULL->READY. Returns FALSE on hard error
 * (already signalled via GST_ELEMENT_ERROR). */
static gboolean
gst_modelmux_build (GstNvModelMux * self)
{

  /* If the sub-graph is already built, return TRUE. */
  if (self->built)
    return TRUE;

  ModelMuxConfig *config = &self->config;

  /* Parse the config-file into self->config and reconcile with the GObject properties. */
  if (!gst_modelmux_parse_config_file (self, self->config_file_path))
    return FALSE;

  /* DISABLED: skip inference entirely and act as a transparent passthrough
   * (wire sink -> identity -> src). A mid-pipeline element must still forward buffers, so
   * a lightweight 'identity' provides the data path with correct caps/event/query handling.
   * Named "modelmux-passthrough" so it is identifiable in logs / gst-inspect graphs. */
  if (!self->enable) {
    GstPad *p;
    GST_INFO_OBJECT (self, "enable=0: bypassing inference (passthrough mode)");
    self->passthru = gst_element_factory_make ("identity", "modelmux-passthrough");
    if (!self->passthru) {
      GST_ELEMENT_ERROR (self, CORE, MISSING_PLUGIN,
          ("failed to create 'identity' for passthrough"), (NULL));
      return FALSE;
    }
    g_object_set (self->passthru, "silent", TRUE, NULL);
    gst_bin_add (GST_BIN (self), self->passthru);
    p = gst_element_get_static_pad (self->passthru, "sink");
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad), p);
    gst_object_unref (p);
    p = gst_element_get_static_pad (self->passthru, "src");
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad), p);
    gst_object_unref (p);
    self->built = TRUE;
    return TRUE;
  }

  /* Default primary model is OPTIONAL. When none is configured, no model is loaded at 
  init (zero VRAM) and any stream that resolves to "no model" -- because it requested
  nothing and there is no default, or it requested a model that does not exist and there
  is no default -- is PASSED THROUGH the display mux WITHOUT inference (frames flow, no
  detections). A user can then load a custom model and attach streams to it on demand. */
  /* pre-bin: the pool refs don't exist yet, so the PARSED seed value is the
   * only (and correct) source here. Post-bin, defaults are read from the pools. */
  if (!config->default_primary)
    GST_INFO_OBJECT (self, "no default primary model configured -- streams with no resolvable "
        "model will PASS THROUGH without inference (set 'primary-model' to preload a default)");

  /* Build the reusable inference bin (demux -> per-stream tees -> model pools -> combined mux).
   * Passing `self` as the parent makes modelmux_bin_new() add the bin's internal GstBin as a
   * CHILD of this element, so the element owns nothing internal -- it only bridges its ghost
   * pads to the bin's boundary pads below. Only the empty skeleton is built here; models are
   * instantiated later (defaults at PLAYING, per-stream lanes on attach). */
  ModelMuxBin *modelmux_bin = modelmux_bin_new (GST_ELEMENT (self),
      GST_ELEMENT_NAME (self), config->batch_size, config);
  if (!modelmux_bin) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("failed to create the multi-model inference bin"), (NULL));
    return FALSE;
  }

  /* Point our ghost pads at the bin's boundary pads. */
  gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad),
      modelmux_bin_get_sink (modelmux_bin));
  gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad),
      modelmux_bin_get_src (modelmux_bin));

  /* Subscribe to the pipeline bus + install the
   * data-plane auto-attach (the REST stream/model API is handled INTERNALLY). */
  gst_modelmux_control_attach (self);

  /* Wire callbacks, THEN publish the bin atomically -- both under api_control_lock, so no
   * other thread ever sees a half-wired bin (control handlers lock; probes read lock-free). */
  g_mutex_lock (&self->api_control_lock);
  gst_modelmux_control_wire_ota_rollback (self, modelmux_bin);
  g_atomic_pointer_set (&self->modelmux_bin, modelmux_bin);
  g_mutex_unlock (&self->api_control_lock);

  {
    const DefaultModelRef *bdp = modelmux_bin_get_default (modelmux_bin, FALSE);
    const DefaultModelRef *bds = modelmux_bin_get_default (modelmux_bin, TRUE);
    GST_INFO_OBJECT (self, "built: batch=%u unified=%d primary='%s' shadow='%s'",
        config->batch_size, config->unified_batch,
        (bdp && bdp->name) ? bdp->name : "(none)",
        (bds && bds->name) ? bds->name : "(none)");
  }

  /* Set the built flag to TRUE. */
  self->built = TRUE;

  return TRUE;
}

/* Tear the sub-graph down at READY->NULL: detach ghost targets, free the bin
 * (which removes its child GstBin from us) or the passthrough, and release the
 * assembled config so a subsequent NULL->READY rebuilds cleanly.
 * Takes api_control_lock for the mm swap + free, so no action-signal handler or queued
 * control dispatch can be mid-use of self->modelmux_bin when it is freed (the handlers
 * hold api_control_lock across their whole mm use). */
static void
gst_modelmux_teardown (GstNvModelMux * self)
{
  ModelMuxBin *mm;

  g_mutex_lock (&self->api_control_lock);
  mm = self->modelmux_bin;
  /* atomic (paired with the probes' lock-free advisory NULL-checks); queued
   * control idles must stop before free */
  g_atomic_pointer_set (&self->modelmux_bin, NULL);

  /* stop the REST control plane first (disconnect bus, drop pending reroutes) */
  gst_modelmux_control_detach (self);

  if (self->sinkpad)
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad), NULL);
  if (self->srcpad)
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad), NULL);

  if (mm)
    modelmux_bin_free (mm);  /* removes its child bin from us */
  if (self->passthru) {
    gst_element_set_state (self->passthru, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (self), self->passthru);
    self->passthru = NULL;
  }

  modelmux_config_clear (&self->config);
  memset (&self->config, 0, sizeof (self->config));
  self->built = FALSE;
  self->initial_default_load_started = FALSE;
  g_mutex_unlock (&self->api_control_lock);
}

/*
 * Element lifecycle hook: build/warm/tear down the internal inference bin in step
 * with the pipeline state.
 *   NULL->READY     : build the sub-graph (gst_modelmux_build); on failure, tear
 *                     down and fail the transition.
 *   PAUSED->PLAYING : kick off the one-time default-model warm-up.
 *   READY->NULL     : tear down the sub-graph and release the assembled config.
 * Build runs BEFORE chaining up to the parent GstBin (so the new children get the
 * state change); teardown runs AFTER (so children are already down first). A failed
 * parent chain-up on NULL->READY also tears down, so nothing leaks in NULL.
 */
static GstStateChangeReturn
gst_modelmux_change_state (GstElement * element, GstStateChange transition)
{
  GstNvModelMux *self = GST_NVMODELMUX (element);
  GstStateChangeReturn ret;

  /* Build BEFORE chaining up so GstBin propagates the upward state change into
   * the freshly-added children. */
  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    if (!gst_modelmux_build (self)) {
      /* release any partially-assembled config / children so the leaked ModelMuxConfig
       * (parsed file + registered model defs) is freed and a subsequent NULL->READY
       * re-parses from a clean slate instead of double-registering. teardown is
       * NULL-safe for every field that build may not have reached. */
      gst_modelmux_teardown (self);
      return GST_STATE_CHANGE_FAILURE;
    }
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    /* a NULL->READY that failed in the parent chain-up leaves the element in
     * NULL, so READY->NULL (and thus teardown) would never run: the built graph,
     * bus ref, control probes and timers would leak -- and finalize's defensive
     * teardown would later touch already-disposed children. Tear down now. */
    if (transition == GST_STATE_CHANGE_NULL_TO_READY && self->built)
      gst_modelmux_teardown (self);
    return ret;
  }

  /* Warm the default model(s) once the element (and its model bins) have reached
   * PLAYING -- sync_state_with_parent brings each model bin up and its nvinfer
   * deserializes/builds the engine, flipping it to READY.
   *   wait-for-default-models = TRUE (default): BLOCK here until the defaults are
   *     WARMED, so they infer on every frame (mirrors nvinfer's engine build at
   *     the state transition). A default that FAILS to warm fails the transition.
   *   FALSE: kick the warm async and return immediately -- streams added before it
   *     completes passthrough until ready (legacy behaviour).
   * Either way, RUNTIME model/load stays async; this gates only the startup default. */
  if (transition == GST_STATE_CHANGE_PAUSED_TO_PLAYING
      && self->modelmux_bin && !self->initial_default_load_started) {
    self->initial_default_load_started = TRUE;
    if (!modelmux_bin_load_defaults (self->modelmux_bin, self->wait_for_default_models)
        && self->wait_for_default_models) {
      GST_ELEMENT_ERROR (self, LIBRARY, INIT,
          ("a configured default model failed to warm up"),
          ("wait-for-default-models=true and the default engine could not be "
           "built/loaded -- see the nvinfer errors above; fix the model/engine or "
           "set wait-for-default-models=false to start in passthrough"));
      return GST_STATE_CHANGE_FAILURE;
    }
  }

  /* Tear down AFTER chaining up so children are already in NULL. */
  if (transition == GST_STATE_CHANGE_READY_TO_NULL)
    gst_modelmux_teardown (self);

  return ret;
}

/* ================================================================== */
/* Action-signal handlers (forward to the bin's control API)           */
/*                                                                     */
/* All are no-ops with a clear GST_WARNING when the bin is not live     */
/* (disabled / before READY), never a crash.                           */
/* ================================================================== */

/* "load-model": register (name, config-file[, engine][, version]) and warm it
 * into the limbo pool. Returns ModelStatus (gint). */
static gint
gst_modelmux_load_model (GstNvModelMux * self, const gchar * name,
    const gchar * config_file, const gchar * engine_file, const gchar * version)
{
  /* api_control_lock across the WHOLE handler: signals run on the emitter's thread, so an
   * unlocked check-then-use of self->modelmux_bin races READY->NULL teardown (UAF), and the
   * self->config catalog mutation below races the main-loop REST handlers. Same
   * pattern in every action-signal handler. */
  g_mutex_lock (&self->api_control_lock);
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "load-model: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return MODEL_FAILED;
  }
  if (!null_if_empty (name)) {
    GST_WARNING_OBJECT (self, "load-model: empty model name");
    g_mutex_unlock (&self->api_control_lock);
    return MODEL_FAILED;
  }
  if (null_if_empty (config_file)) {
    /* catalog entry is name -> base config (engine/version are per-instance now). */
    modelmux_config_register_model (&self->config, name, config_file, NULL, NULL, 0);
    if (!modelmux_config_find_model (&self->config, name)) {
      /* registration REFUSED (reserved '@'/';' in the name, or registry full):
       * do NOT load anyway -- the warmed bin would be invisible to the catalog
       * (lazy-warm, version resolution, unload guards all consult it). */
      GST_WARNING_OBJECT (self, "load-model '%s': could not register in the "
          "catalog (invalid name or registry full) -- rejected", name);
      g_mutex_unlock (&self->api_control_lock);
      return MODEL_FAILED;
    }
  } else if (!modelmux_config_find_model (&self->config, name)) {
    GST_WARNING_OBJECT (self,
        "load-model '%s': not in catalog and no config-file supplied", name);
    g_mutex_unlock (&self->api_control_lock);
    return MODEL_FAILED;
  }
  {
    const ModelCatalogEntry *def = modelmux_config_find_model (&self->config, name);
    const gchar *basecfg = null_if_empty (config_file) ? config_file : (def ? def->config_file : NULL);
    gchar *key = model_key (name, null_if_empty (version) ? version : MM_MODEL_VERSION_DEFAULT);
    gchar *derived = NULL;
    const gchar *eff = null_if_empty (config_file);   /* custom config, or NULL => catalog base */
    gint r;
    if (null_if_empty (engine_file) && basecfg) {     /* nvinfer: swap engine into a derived config */
      derived = modelmux_config_derive (basecfg, engine_file, -1, name,
          null_if_empty (version) ? version : MM_MODEL_VERSION_DEFAULT);   /* match the @1 instance key */
      if (!derived) {
        /* an EXPLICIT engine that failed to derive must FAIL the load: the base
         * config would run the wrong artifact while the caller believes the
         * override took (modelmux_config_derive already logged why). */
        GST_ERROR_OBJECT (self, "load-model '%s': engine override could not be "
            "derived from '%s' -- rejecting", name, basecfg);
        g_free (key);
        g_mutex_unlock (&self->api_control_lock);
        return MODEL_FAILED;
      }
      eff = derived;
    } else if (null_if_empty (engine_file) && !basecfg) {
      /* an engine override needs a base config to swap model-engine-file into -- without one
       * (model not in catalog and no config-file supplied) the engine is silently unusable. */
      GST_WARNING_OBJECT (self, "load-model '%s': engine-file given but no base config "
          "(not in catalog and no config-file supplied) -- ignoring the engine override", name);
    }
    r = (gint) modelmux_bin_load_model (self->modelmux_bin, key, eff, -1, 0);
    g_free (key);
    g_free (derived);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "unload-model": free an idle model by name (limbo or either pool). */
static gboolean
gst_modelmux_unload_model (GstNvModelMux * self, const gchar * name)
{
  gboolean r = FALSE;
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "unload-model: inference not active");
  } else if (null_if_empty (name)) {
    /* signal carries a bare name -> unload ALL non-serving versions of it. */
    r = modelmux_bin_unload_versions (self->modelmux_bin, name, NULL) > 0;
  }
  g_mutex_unlock (&self->api_control_lock);
  return r;
}

/* "reload-model": in-place OTA checkpoint swap on an already-loaded model.
 * engine -> nvinfer model-engine-file; else config-file -> full reload;
 * version updates provenance. */
static gboolean
gst_modelmux_reload_model (GstNvModelMux * self, const gchar * name,
    const gchar * config_file, const gchar * engine_file, const gchar * version)
{
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "reload-model: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  if (!null_if_empty (name)) {
    GST_WARNING_OBJECT (self, "reload-model: empty model name");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  /* OTA in-place reload is SUPERSEDED by versioned model/load (a new version = a new bin);
   * this signal is retained for compatibility and targets the 1 instance of the name. */
  {
    gchar *key = model_key (name, null_if_empty (version) ? version : MM_MODEL_VERSION_DEFAULT);
    gboolean r;
    modelmux_config_update_model (&self->config, name, null_if_empty (config_file),
        null_if_empty (engine_file), null_if_empty (version));
    r = modelmux_bin_reload_model (self->modelmux_bin, key, null_if_empty (config_file),
        null_if_empty (engine_file), null_if_empty (version));
    g_free (key);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "attach-stream": route one source_id to (primary[, shadow]); NULL roles fall
 * back to the configured / config-time-bound defaults. */
static gboolean
gst_modelmux_attach_stream (GstNvModelMux * self, guint source_id,
    const gchar * name, const gchar * primary, const gchar * shadow)
{
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "attach-stream: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  /* compose canonical keys (signal carries bare names -> version 1). */
  {
    gchar *pk = null_if_empty (primary) ? model_key (primary, MM_MODEL_VERSION_DEFAULT) : NULL;
    gchar *sk = null_if_empty (shadow) ? model_key (shadow, MM_MODEL_VERSION_DEFAULT) : NULL;
    /* The action-signal path carries ONE name. It is the display name, and this
     * entry point has no separate camera_id to offer -- pass it as the camera_id
     * too rather than NULL, since for a single-identifier host the two ARE the
     * same string, which is exactly what stream/add does when only one is given.
     * A stream attached this way stays routable by that name. */
    gboolean r = modelmux_bin_attach_stream (self->modelmux_bin, source_id, null_if_empty (name),
        null_if_empty (name), pk, sk, MM_GPU_ANY, MM_GPU_ANY,
        FALSE, FALSE /* signal names the models: pinned */);
    g_free (pk);
    g_free (sk);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "detach-stream": remove one source_id from the inference graph. */
static gboolean
gst_modelmux_detach_stream (GstNvModelMux * self, guint source_id)
{
  gboolean r = FALSE;
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin)
    GST_WARNING_OBJECT (self, "detach-stream: inference not active");
  else
    r = modelmux_bin_detach_stream (self->modelmux_bin, source_id);
  g_mutex_unlock (&self->api_control_lock);
  return r;
}

/* Parse a "name;version" ref (config-file convention) into a canonical
 * "name@version" pool key.  Accepts bare "name" (falls back to default
 * version) and "name;version".  Returns a newly-allocated string or NULL. */
static gchar *
routing_ref_to_key (const gchar * ref)
{
  if (!null_if_empty (ref))
    return NULL;
  gchar **parts = g_strsplit (ref, ";", 2);
  gchar  *key;
  if (!parts[0] || !*parts[0]) {        /* reject "name"-less refs like ";1" */
    g_strfreev (parts);
    return NULL;
  }
  key = model_key (parts[0],
      (parts[1] && *parts[1]) ? parts[1] : MM_MODEL_VERSION_DEFAULT);
  g_strfreev (parts);
  return key;
}

/* "update-routing": re-point a stream's primary/shadow at runtime (zero-drop).
 * source_id == G_MAXUINT applies to ALL active streams.
 * primary / shadow accept "name" or "name;version" (config-file convention). */
static gboolean
gst_modelmux_update_routing (GstNvModelMux * self, guint source_id,
    const gchar * primary, const gchar * shadow)
{
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "update-routing: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  {
    gchar *pk = routing_ref_to_key (primary);
    gchar *sk = routing_ref_to_key (shadow);
    gboolean r;
    if (source_id == MM_ALL_STREAMS)
      r = modelmux_bin_update_routing_scoped (self->modelmux_bin, NULL, 0, pk, sk,
          MM_GPU_ANY, MM_GPU_ANY, FALSE, FALSE /* explicit routing: pinned */) > 0;
    else
      r = modelmux_bin_update_routing (self->modelmux_bin, source_id, pk, sk,
          MM_GPU_ANY, MM_GPU_ANY, FALSE, FALSE);
    g_free (pk);
    g_free (sk);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "get-status": schema-stable JSON dump. Independent filters: model_filter
 * narrows models[]; stream_filter / source_id (>=0) narrows streams[]. Empty
 * filters / source_id<0 => no narrowing. Returns a newly-allocated string. */
static gchar *
gst_modelmux_get_status (GstNvModelMux * self, const gchar * model_filter,
    const gchar * stream_filter, gint source_id)
{
  gchar *r;
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin)
    r = g_strdup ("{\"models\":[],\"streams\":[]}");
  else
    r = modelmux_bin_status_json (self->modelmux_bin, null_if_empty (model_filter),
        NULL /* model_version filter: REST status-query path only */,
        null_if_empty (stream_filter), source_id);
  g_mutex_unlock (&self->api_control_lock);
  return r;
}

/* Write a property value by id. */
static void
gst_modelmux_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  /* Mark the property as being set through g_object_set (nvinfer convention), so
   * gst_modelmux_parse_config_file lets it override the config-file value. */
  if (prop_id < PROP_LAST)
    self->is_prop_set[prop_id] = TRUE;

  switch (prop_id) {
    case PROP_ENABLE:
      self->enable = g_value_get_boolean (value);
      break;
    case PROP_CONFIG_FILE_PATH:
      set_str (&self->config_file_path, g_value_get_string (value));
      break;
    case PROP_BATCH_SIZE:
      self->batch_size = g_value_get_uint (value);
      break;
    case PROP_PER_MODEL_BATCH_SIZE:
      self->per_model_batch_size = g_value_get_uint (value);
      break;
    case PROP_UNIFIED_BATCH:
      self->unified_batch = g_value_get_boolean (value);
      break;
    case PROP_WAIT_FOR_DEFAULT_MODELS:
      self->wait_for_default_models = g_value_get_boolean (value);
      break;
    case PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS:
      self->compact_idle_models_across_pools = g_value_get_boolean (value);
      break;
    case PROP_MODEL_SHARDING:
      self->model_sharding = g_value_get_boolean (value);
      break;
    case PROP_PRIMARY_MODEL:
      set_str (&self->primary_model, g_value_get_string (value));
      break;
    case PROP_SHADOW_MODEL:
      set_str (&self->shadow_model, g_value_get_string (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Read a property value by id. */
static void
gst_modelmux_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  switch (prop_id) {
    case PROP_ENABLE:
      g_value_set_boolean (value, self->enable);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, self->config_file_path);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, self->batch_size);
      break;
    case PROP_PER_MODEL_BATCH_SIZE:
      g_value_set_uint (value, self->per_model_batch_size);
      break;
    case PROP_UNIFIED_BATCH:
      g_value_set_boolean (value, self->unified_batch);
      break;
    case PROP_WAIT_FOR_DEFAULT_MODELS:
      g_value_set_boolean (value, self->wait_for_default_models);
      break;
    case PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS:
      g_value_set_boolean (value, self->compact_idle_models_across_pools);
      break;
    case PROP_MODEL_SHARDING:
      g_value_set_boolean (value, self->model_sharding);
      break;
    case PROP_PRIMARY_MODEL:
      g_value_set_string (value, self->primary_model);
      break;
    case PROP_SHADOW_MODEL:
      g_value_set_string (value, self->shadow_model);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_modelmux_dispose (GObject * object)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  /* defensive: normally torn down at READY->NULL. MUST run in dispose (idempotent
   * via self->built), BEFORE chaining up: the parent GstBin dispose removes and
   * unrefs our pads and children, so running teardown any later (finalize) would
   * operate on already-disposed objects (ghost pads, passthru, the mm child bin). */
  if (self->built)
    gst_modelmux_teardown (self);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_modelmux_finalize (GObject * object)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  g_free (self->config_file_path);
  g_free (self->primary_model);
  g_free (self->shadow_model);
  g_mutex_clear (&self->source_state_lock);
  g_mutex_clear (&self->api_control_lock);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * One-time class initialisation. Runs ONCE when the GType is first registered and
 * configures the class VTABLE that is SHARED by every nvmodelmux instance (it is
 * NOT per-instance -- per-instance setup lives in gst_modelmux_init). Here we:
 *   - override the base-class vmethods (GObject set/get_property, dispose, finalize;
 *     GstElement change_state);
 *   - install the GObject properties (enable, config-file-path, batch-size, ...);
 *   - register the runtime-control ACTION signals (load/unload/reload-model,
 *     attach/detach-stream, update-routing, get-status);
 *   - add the (always) sink/src pad templates and set the element metadata.
 * No pads are created and no sub-graph is linked here -- that is instance work done
 * lazily at NULL->READY (gst_modelmux_build); this function only defines the class.
 */
static void
gst_modelmux_class_init (GstNvModelMuxClass * klass)
{
  /* GObjectClass and GstElementClass */
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  /* Override base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_modelmux_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_modelmux_get_property);
  gobject_class->dispose = GST_DEBUG_FUNCPTR (gst_modelmux_dispose);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_modelmux_finalize);
  element_class->change_state = GST_DEBUG_FUNCPTR (gst_modelmux_change_state);

  /* properties that only take effect at (re)build are MUTABLE_READY */
  const GParamFlags rwc =
  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY;

  /* -------- properties -------- */
  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Run inference (TRUE) or act as a transparent passthrough (FALSE)",
          DEFAULT_ENABLE, rwc));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Config File Path",
          "Multimodel config file ([multimodel] defaults/behaviour + per-sensor "
          "[stream-model-<id>] bindings), parsed by the plugin. Individual "
          "properties override values from it.", NULL, rwc));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum number of concurrent streams (sizes every internal mux). "
          "0 = inherit from the config-file / default. Should match the upstream "
          "streammux batch.",
          0, MM_MAX_BATCH, 0, rwc));

  g_object_class_install_property (gobject_class, PROP_PER_MODEL_BATCH_SIZE,
      g_param_spec_uint ("per-model-batch-size", "Per-Model Batch Size",
          "nvinfer batch size for a single model instance (0 = use batch-size); "
          "streams beyond this auto-shard onto additional same-model instances",
          0, MM_MAX_BATCH, DEFAULT_PER_MODEL_BATCH_SIZE, rwc));

  g_object_class_install_property (gobject_class, PROP_UNIFIED_BATCH,
      g_param_spec_boolean ("unified-batch", "Unified Batch",
          "Combine primary + shadow outputs into ONE batched buffer (A/B view)",
          DEFAULT_UNIFIED_BATCH, rwc));

  g_object_class_install_property (gobject_class, PROP_WAIT_FOR_DEFAULT_MODELS,
      g_param_spec_boolean ("wait-for-default-models", "Wait For Default Models",
          "Block the pipeline's transition to PLAYING until the configured default "
          "model(s) are warmed, so the defaults run inference on every frame from the "
          "start (mirrors nvinfer). FALSE warms them asynchronously and streams "
          "passthrough until ready. Runtime model/load is always async regardless.",
          DEFAULT_WAIT_FOR_DEFAULT_MODELS, rwc));

  g_object_class_install_property (gobject_class, PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS,
      g_param_spec_boolean ("compact-idle-models-across-pools",
          "Compact Idle Models Across Pools",
          "When a model goes idle as a lone redundant warm copy, move it out of "
          "its role pool (primary/shadow) into the shared limbo pool so a later "
          "request from EITHER role reuses that one warm copy without a reload "
          "-- keeping fewer warm copies overall. Applies only when no twin "
          "already exists; the configured default model is never moved.",
          DEFAULT_COMPACT_IDLE_MODELS_ACROSS_POOLS, rwc));

  /* One property per role, combining the model's name, version and nvinfer
   * config-file as "name;version;config-file". Only the config-file is required;
   * name/version may be empty (e.g. ";;config.txt"), and a bare path with no ';'
   * is taken as the config-file alone. The config-file's existence is validated
   * when the element is built. */
  g_object_class_install_property (gobject_class, PROP_PRIMARY_MODEL,
      g_param_spec_string ("primary-model", "Primary Model",
          "Default PRIMARY model as \"name;version;config-file\" (config-file "
          "required; name defaults to its basename, version optional). Required "
          "to enable inference.", NULL, rwc));

  g_object_class_install_property (gobject_class, PROP_SHADOW_MODEL,
      g_param_spec_string ("shadow-model", "Shadow Model",
          "Optional default SHADOW model (A/B) as \"name;version;config-file\". "
          "Unset => no shadow.", NULL, rwc));

  g_object_class_install_property (gobject_class, PROP_MODEL_SHARDING,
      g_param_spec_boolean ("model-sharding", "Model Sharding",
          "Enable automatic sharding of streams across model instances. "
          "TRUE (default): overflow streams beyond the per-model batch cap are "
          "distributed onto new same-model shard instances (scale-out). "
          "FALSE: streams exceeding the cap are silently ignored (not processed), "
          "matching stock nvinfer behaviour.",
          DEFAULT_MODEL_SHARDING, rwc));

  /* -------- action signals (runtime control plane) --------
   * Generic (libffi) marshaller (NULL) handles the mixed signatures. Emit with
   * g_signal_emit_by_name(element, "<signal>", args..., &retval). */
  gst_modelmux_signals[SIGNAL_LOAD_MODEL] =
      g_signal_new_class_handler ("load-model", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_load_model),
      NULL, NULL, NULL, G_TYPE_INT, 4, G_TYPE_STRING, G_TYPE_STRING,
      G_TYPE_STRING, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_UNLOAD_MODEL] =
      g_signal_new_class_handler ("unload-model", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_unload_model),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 1, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_RELOAD_MODEL] =
      g_signal_new_class_handler ("reload-model", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_reload_model),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 4, G_TYPE_STRING, G_TYPE_STRING,
      G_TYPE_STRING, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_ATTACH_STREAM] =
      g_signal_new_class_handler ("attach-stream", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_attach_stream),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 4, G_TYPE_UINT, G_TYPE_STRING,
      G_TYPE_STRING, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_DETACH_STREAM] =
      g_signal_new_class_handler ("detach-stream", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_detach_stream),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 1, G_TYPE_UINT);

  gst_modelmux_signals[SIGNAL_UPDATE_ROUTING] =
      g_signal_new_class_handler ("update-routing", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_update_routing),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 3, G_TYPE_UINT, G_TYPE_STRING,
      G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_GET_STATUS] =
      g_signal_new_class_handler ("get-status", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_get_status),
      NULL, NULL, NULL, G_TYPE_STRING, 3, G_TYPE_STRING, G_TYPE_STRING,
      G_TYPE_INT);

  /* -------- pads + element metadata -------- */
  gst_element_class_add_static_pad_template (element_class,
      &gst_modelmux_sink_template);
  gst_element_class_add_static_pad_template (element_class,
      &gst_modelmux_src_template);

  gst_element_class_set_static_metadata (element_class,
      "NvModelMux",
      "Generic/Bin/Filter",
      "Per-stream / per-model DeepStream inference with optional A/B "
      "(primary + shadow) serving, live model load/unload/reload (OTA), "
      "promote/swap routing and per-(stream,model,role) provenance",
      "NVIDIA Corporation. DeepStream SDK");
}

/**
 * Per-INSTANCE initialisation. Runs ONCE for EACH nvmodelmux element created (unlike
 * gst_modelmux_class_init, which runs once for the whole class). @self is this one
 * element's own state; nothing here is shared between instances. It:
 *   - sets this instance's property defaults;
 *   - initialises this instance's mutexes (source_state_lock, api_control_lock);
 *   - creates the boundary ghost pads (no-target) so the element can be LINKED while
 *     still in NULL -- their targets, and the whole inference sub-graph, are wired
 *     later at NULL->READY (gst_modelmux_build).
 */
static void
gst_modelmux_init (GstNvModelMux * self)
{
  GstPadTemplate *tmpl;

  /* property defaults */
  self->enable = DEFAULT_ENABLE;
  self->batch_size = 0;                 /* 0 => inherit from config-file / default */
  self->per_model_batch_size = DEFAULT_PER_MODEL_BATCH_SIZE;
  self->unified_batch = DEFAULT_UNIFIED_BATCH;
  self->wait_for_default_models = DEFAULT_WAIT_FOR_DEFAULT_MODELS;
  self->compact_idle_models_across_pools = DEFAULT_COMPACT_IDLE_MODELS_ACROSS_POOLS;
  self->model_sharding = DEFAULT_MODEL_SHARDING;
  g_mutex_init (&self->source_state_lock);      /* guards the auto-attach source-id table */
  g_mutex_init (&self->api_control_lock);       /* serializes control entry points vs teardown */

  /* No-target ghost pads created up front so the element can be LINKED while
   * still in NULL; their targets are wired when the sub-graph is built. */
  tmpl = gst_static_pad_template_get (&gst_modelmux_sink_template);
  self->sinkpad = gst_ghost_pad_new_no_target_from_template ("sink", tmpl);
  gst_object_unref (tmpl);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  tmpl = gst_static_pad_template_get (&gst_modelmux_src_template);
  self->srcpad = gst_ghost_pad_new_no_target_from_template ("src", tmpl);
  gst_object_unref (tmpl);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);
}
