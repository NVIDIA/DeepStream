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
 * SECTION:element-nvmodelmux
 *
 * nvmodelmux
 * ===================
 * A self-contained DeepStream GStreamer bin element that runs PER-STREAM,
 * PER-MODEL inference with optional A/B (primary + shadow) serving.
 *
 * One batched input (e.g. from nvstreammux / nvmultiurisrcbin) enters the
 * element's sink pad; the element fans every source out to its assigned
 * model(s), batches each model independently, runs nvinfer, and re-batches all
 * outputs into a SINGLE batched buffer on the src pad. Each output frame carries
 * InferenceProvenanceMeta so a downstream consumer can attribute detections to the
 * exact (stream, model, role) that produced them.
 *
 *   sink (1 batched) --> [ demux -> per-stream tee -> {Primary, Shadow} model
 *                          pools (per-model nvstreammux->nvinfer->nvstreamdemux)
 *                          -> combined nvstreammux ] --> src (1 batched)
 *
 * ALL routing / pooling / sharding / A/B promote-swap / OTA reload / provenance
 * logic lives inside the bin (see gstnvmodelmux_bin.c); this element is
 * the thin GObject shell that exposes it as a reusable plugin:
 *
 *   - CONFIGURATION via GObject properties. (The HOST app -- e.g. deepstream-app
 *     / deepstream-test5 -- parses the DeepStream main config and sets these;
 *     per-sensor model assignment is driven through the attach-stream signal.)
 *   - DYNAMIC CONTROL via action signals (load/unload/reload model,
 *     attach/detach stream, update routing, query status) so any host app
 *     (custom app, deepstream-app, deepstream-test5, gst-launch test rig) can
 *     drive it without linking against the bin's private C API.
 *   - The existing model-status downstream GstQuery is still answered
 *     internally, so a native REST front-end (nvmultiurisrcbin) keeps working
 *     with zero glue.
 *
 * The element is a GstBin: when disabled (enable=FALSE) it degrades to a
 * transparent passthrough, exactly like other DeepStream components.
 */

#ifndef __GST_NVMODELMUX_H__
#define __GST_NVMODELMUX_H__

#include <gst/gst.h>
#include "gstnvmodelmux_priv.h"   /* ModelMuxConfig, ModelMuxBin, bin API */

G_BEGIN_DECLS

#define GST_TYPE_NVMODELMUX (gst_modelmux_get_type ())
#define GST_NVMODELMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVMODELMUX, GstNvModelMux))
#define GST_NVMODELMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_NVMODELMUX, GstNvModelMuxClass))
#define GST_IS_NVMODELMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVMODELMUX))
#define GST_IS_NVMODELMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_NVMODELMUX))

typedef struct _GstNvModelMux GstNvModelMux;
typedef struct _GstNvModelMuxClass GstNvModelMuxClass;

/* GObject property ids. Public (in the header, nvinfer convention) so the config
 * parser can consult is_prop_set[] -- an explicitly-set property OVERRIDES the
 * config-file value. Keep PROP_LAST last: it sizes is_prop_set[]. */
enum
{
  PROP_0,
  PROP_ENABLE,
  PROP_CONFIG_FILE_PATH,
  PROP_BATCH_SIZE,
  PROP_PER_MODEL_BATCH_SIZE,
  PROP_UNIFIED_BATCH,
  PROP_WAIT_FOR_DEFAULT_MODELS,
  PROP_PRIMARY_MODEL,
  PROP_SHADOW_MODEL,
  PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS,
  PROP_MODEL_SHARDING,
  PROP_LAST
};

/**
 * ModelMuxInstance struct. Property values are stored verbatim and only assembled into
 * @config + materialised into @mm at the NULL->READY state transition, so the
 * order in which properties are set does not matter. Everything that does real
 * work lives in @mm (the reusable ModelMuxBin); @passthru is used only when
 * the element is disabled.
 */
struct _GstNvModelMux
{
  GstBin parent;        /**< we are a bin                               */

  /** the inference bin (NULL until built, or in passthrough mode) */
  ModelMuxBin *modelmux_bin;

  /** identity element, used when enable=FALSE (transparent passthrough) */
  GstElement *passthru;

  /** ghost sink/src pad (no-target until the sub-graph is built) */
  GstPad *sinkpad;
  GstPad *srcpad;

  /** TRUE once the sub-graph has been materialised */
  gboolean built;

  /** default-model warm-up KICKED OFF at first PLAYING (once); reset on teardown. */
  gboolean initial_default_load_started;

  /** PAD_ADDED/PAD_DELETED/STREAM_EOS event probe on the sink */
  gulong sink_event_probe_id;
  /** buffer probe: drop a source's buffers until it is attached (safety net) */
  gulong sink_probe_id;

  /** serializes every control-plane ENTRY POINT (action signals, REST dispatch,
   *  pending-reroute poller) against each other AND against teardown: self->modelmux_bin
   *  may only be DEREFERENCED under it (published/cleared with g_atomic_pointer_set under
   *  the lock). Lock order: api_control_lock -> ib->lock -> free_lock; api_control_lock ->
   *  source_state_lock. NEVER taken from probes (they do a lock-free NULL-check only). */
  GMutex api_control_lock;

  /** guards source_attach_state + requested_binding_info */
  GMutex source_state_lock;
  /** source_id -> ModelMuxSourceAttachState (attach lifecycle; driven by stream/add ->
   * GST_NVEVENT_PAD_ADDED and stream/remove -> GST_NVEVENT_PAD_DELETED) */
  GHashTable *source_attach_state;
  /** source_id -> ModelMuxRequestedBindingInfo (per-stream model binding from the in-band
   * GST_NVEVENT_STREAM_MODEL_BIND event: sensor_id + metadata). Kept SEPARATE from
   * source_attach_state so it never affects the attach lifecycle/freshness logic. */
  GHashTable *requested_binding_info;
  /** batches seen on the sink (for staleness detection) */
  guint64 batch_idx;

  /** pipeline bus (ref held for the "sync-message" subscription) */
  GstBus *bus;
  /** "sync-message" handler id (0 = not connected) */
  gulong bus_sync_id;
  /** ModelMuxPendingReroute*: model/update flips deferred until the target warms up */
  GList *pending_reroutes;
  /** g_timeout id for the deferred pending-reroute poller (0 = not running) */
  guint pending_poll_id;


  /** assembled configuration (owned) */
  ModelMuxConfig config;
  /** properties the user set (by PROP_*); a set property overrides the config-file value. */
  gboolean  is_prop_set[PROP_LAST];

  /** flag to enable inference (TRUE) vs passthrough (FALSE) */
  gboolean enable;
  /** multimodel config (defaults, behaviour, [stream-model-<id>] bindings); parsed by the plugin. Individual properties override it. */
  gchar *config_file_path;
  /** max concurrent streams batch cap */
  guint batch_size;
  /** per-model nvstreammux batch cap (0=def) */
  guint per_model_batch_size;
  /** flag to enable unified A/B display batch */
  gboolean unified_batch;
  /** wait-for-default-models: block PLAYING until configured defaults warmed */
  gboolean wait_for_default_models;
  /** flag to compact idle models across pools on idle (dedup-on-idle) */
  gboolean compact_idle_models_across_pools;
  /** flag to enable model sharding on overflow */
  gboolean model_sharding;
  /** default primary model "name;version;config" */
  gchar *primary_model;
  /** default shadow model "name;version;config" */
  gchar *shadow_model;
};

struct _GstNvModelMuxClass
{
  GstBinClass parent_class;
};

GType gst_modelmux_get_type (void);

/* Parse the config-file into self->config, then reconcile with the GObject properties:
 * an EXPLICITLY-set property (self->is_prop_set[PROP_*]) overrides the file value; an
 * unset property is FILLED from the file (so a read-back reflects the effective config).
 * Returns FALSE if the path is missing/empty or the file cannot be parsed. Takes the
 * element (nvinfer convention) so it can update both self->config and the property
 * fields in one place. Defined in gstnvmodelmux_config.c. */
gboolean gst_modelmux_parse_config_file (GstNvModelMux * self, const gchar * cfg_file);

/* --- native-REST control plane (gstnvmodelmux_control.c) ---
 * Make the element self-contained: subscribe to the pipeline bus so the
 * nvmultiurisrcbin native-REST messages (stream/add|remove, model/load|unload|
 * update) are parsed and executed INTERNALLY -- the host app needs no glue.
 * Attached when the inference bin is built; detached on teardown. */
void gst_modelmux_control_attach (GstNvModelMux * self);
void gst_modelmux_control_detach (GstNvModelMux * self);
/* Wire the control plane's bin-invoked callbacks (OTA rollback follow-up)
 * into a freshly-built bin, before it is published to self->modelmux_bin. */
void gst_modelmux_control_wire_ota_rollback (GstNvModelMux * self, ModelMuxBin * mm);
/* Per-stream model-selection metadata parser (gstnvmodelmux_control_metadata.c;
 * called by the control plane). Declared here -- the one header BOTH files
 * include -- so the definition sees a prototype (tmake: -Werror=missing-prototypes). */
G_GNUC_INTERNAL void modelmux_control_parse_model_metadata (GstNvModelMux * self,
    const gchar * json, gchar ** primary_model_name, gchar ** primary_model_version,
    gint * primary_model_gpu, gchar ** shadow_model_name, gchar ** shadow_model_version,
    gint * shadow_model_gpu);

G_END_DECLS

#endif /* __GST_NVMODELMUX_H__ */
