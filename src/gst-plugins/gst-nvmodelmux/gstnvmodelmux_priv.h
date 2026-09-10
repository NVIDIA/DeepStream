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
 * deepstream-multimodel
 * =====================
 * Dynamic multi-stream / multi-model (primary + shadow) DeepStream app.
 *
 * Pipeline (high level):
 *
 *   nvmultiurisrcbin            (decode + NATIVE REST add/remove; internal nvstreammux)
 *        |  one BATCHED src, NvDsBatchMeta, source_id per camera
 *        v
 *   ModelMuxBin                <-- reusable; its GhostPad sink is the entry
 *        GhostPad sink (batched) -> nvstreamdemux -> per-source src_<id>
 *        per source_id -> tee -> { Primary role, Shadow role }
 *        each role: per-MODEL  nvstreammux -> nvinfer -> nvstreamdemux  (batched per model name)
 *        all role outputs -> ONE combined nvstreammux (display batch)
 *        v
 *   nvmultistreamtiler -> nvvideoconvert -> nvdsosd -> sink     (loose display chain)
 *
 * Stream add/remove is handled by the NATIVE DeepStream REST server embedded in
 * nvmultiurisrcbin (ip-address:port). This app does NOT run a custom HTTP server.
 * Per-stream model selection rides in the request `metadata` (stream/add); models are
 * managed at runtime via model/load, model/unload and model/update. Only the configured
 * default models preload.
 */

#ifndef __DEEPSTREAM_MULTIMODEL_APP_H__
#define __DEEPSTREAM_MULTIMODEL_APP_H__

#include <gst/gst.h>
#include <glib.h>

#include "nvdsmeta_schema.h"      /* InferenceProvenanceMeta (public schema struct). The
                                   * NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE meta-type enum
                                   * lives in nvdsmeta.h (pulled in via gstnvdsmeta.h) --
                                   * a consumer needs BOTH headers. */
#include "gstnvmodelmux_perf.h"   /* ModelMuxPerf / ModelMuxPerfCounter perf-metrics module */

G_BEGIN_DECLS

#define MM_MAX_STREAMS_DEFAULT   16
#define MM_MAX_BATCH             1024   /* hard cap on batch-size (sizes every internal mux) */
#define MM_MAX_MODELS            32
#define MM_DEFAULT_REST_PORT     9000
#define MM_DEFAULT_REST_IP       "0.0.0.0"

/* Model versions are POSITIVE INTEGERS ("1", "2", ...), matching Triton's model-repository
 * convention (numbered version dirs; highest = latest). The first version is "1"; model/load
 * auto-increments to max+1. A model is identified by (name, version), joined into a canonical
 * instance KEY "name@version" used as the pool / limbo-map hash key, so multiple versions of
 * one name coexist as distinct ModelBins. */
#define MM_MODEL_VERSION_DEFAULT "1"
#define MM_MODEL_KEY_SEP '@'

/* Canonical batched-mux geometry. Applied to the nvmultiurisrcbin input mux and
 * mirrored by the per-model + combined muxes, so all batches share resolution.
 * (Decided at runtime; not user config.) */
#define MM_MUX_WIDTH             1920
#define MM_MUX_HEIGHT            1080
#define MM_MUX_PUSH_TIMEOUT      33333

/** Inference backend element used inside a ModelBin. AUTO-DETECTED from the model's
 *  config-file: an nvinferserver protobuf config has a top-level `infer_config {` block;
 *  an nvinfer INI config has a `[property]` section. Default = nvinfer (back-compat). */
typedef enum
{
  MODEL_INFER = 0,     /** nvinfer       (TensorRT; default)         */
  MODEL_INFERSERVER    /** nvinferserver (Triton backend)            */
} ModelType;

/** A model definition from the [model-*] config groups. */
typedef struct
{
  gchar    *name;          /** logical model name, e.g. "Model-A"             */
  gchar    *config_file;   /** nvinfer/nvinferserver config-file-path         */
  gchar    *engine_file;   /** optional prebuilt engine override (NULL=config's) */
  gchar    *version;       /** optional DEFT checkpoint version (NULL=unset)  */
  guint     max_streams;   /** per-model batch / max streams (clamped)        */
  ModelType type;          /** nvinfer vs nvinferserver   (auto-detected)     */
} ModelCatalogEntry;

/** A per-version artifact override from a `[model-<name>-<version>]` config group: the
 *  engine/checkpoint and/or custom config to use when instantiating that specific
 *  (name, version). Declared ONCE; reused by every ref (default + per-sensor). Either
 *  field may be NULL -> falls back to the model's base catalog config/engine. */
typedef struct
{
  gchar    *name;          /** model name (must match a [model-name-*] catalog entry)  */
  gchar    *version;       /** positive-integer version this override applies to       */
  gchar    *engine;        /** OPT engine swapped into the effective config (NULL=base)*/
  gchar    *config;        /** OPT custom base config for this version (NULL=catalog)  */
  gint      gpu;           /** placement recorded at load/update (-1 = config's device)*/
  guint     batch;         /** OPT per-VERSION batch/stream cap (0 = unset -> falls back
                            *   to the catalog name-level batch, then the global
                            *   per-model-batch-size). A version's batch is IMMUTABLE
                            *   while that version is live (part of the loaded identity:
                            *   the mux, nvinfer property and slot tables are sized by
                            *   it at construction) -- guarded at load and model/update
                            *   admission. */
} ModelVersionArtifact;

/** A config-time per-sensor model binding, parsed from a `[stream-model-<sensor_id>]`
 *  group. Model NAMES reference entries in `ModelMuxConfig.models[]`; a NULL role falls back
 *  to the corresponding configured default. Keyed by sensor id (position-independent). */
typedef struct
{
  gchar    *sensor_id;        /** camera id this binding applies to (matches sensor-id-list) */
  gchar    *primary;          /** primary model NAME (NULL => default_primary)               */
  gchar    *primary_version;  /** primary model VERSION (NULL => default_primary_version)    */
  gchar    *shadow;           /** shadow  model NAME (NULL => default_shadow)                */
  gchar    *shadow_version;   /** shadow  model VERSION (NULL => default_shadow_version)     */
} StreamModelBinding;

typedef struct
{
  /** max concurrent streams. */
  guint     batch_size;

  /* defaults / behaviour */
  guint     per_model_max_streams; /** default cap for a model if unset       */
  gboolean  unified_batch;         /** one display batch (TRUE) vs per-role   */
  gboolean  wait_for_default_models; /** TRUE: block the pipeline's PLAYING
                                    *   transition until the configured default
                                    *   model(s) are WARMED, so defaults infer on
                                    *   every frame (mirrors nvinfer). FALSE:
                                    *   warm async, streams passthrough until
                                    *   ready. Runtime model/load is ALWAYS async
                                    *   regardless -- this gates only the
                                    *   startup default preload.               */
  gboolean  enable;                /** [multimodel] enable: 0 => bypass the inference
                                    *   bin (decode->display passthrough)             */
  gboolean  shard_compact;         /** auto-compact under-full shards of a model back
                                    *   into fewer shards when streams drain (default
                                    *   TRUE). Make-before-break, zero-drop. Set 0 to
                                    *   keep grown shards until they empty naturally.  */
  gboolean  compact_idle_models_across_pools; /** dedup-on-idle policy (default TRUE).
                                    *   When a model goes idle as a lone redundant warm
                                    *   copy (no twin), TRUE moves it out of its role pool
                                    *   (primary/shadow) into the shared limbo pool so a
                                    *   later request from EITHER role reuses that one warm
                                    *   copy without a reload -- fewer warm copies overall.
                                    *   FALSE keeps it role-tagged in its pool. The
                                    *   configured default model is never moved. Redundant
                                    *   twins are freed either way.                      */
  gboolean  model_sharding;         /** auto-shard streams when per-model cap exceeded.
                                    *   TRUE (default): overflow streams get a new shard
                                    *   of the same model (scale-out, existing behaviour).
                                    *   FALSE: streams beyond the per-model batch cap are
                                    *   silently ignored -- matching nvinfer behaviour when
                                    *   more streams than batch-size arrive.             */

  /* [muxer]: nvstreammux settings shared by BOTH the per-model inference muxers (one per model
   * bin, batching that model's streams before nvinfer) AND the single outer combined mux
   * (ib->out_mux, merging all model bins' outputs downstream). width/height/live-source are
   * common to all muxers; only the flush cadence differs (per-model regroups an already-batched
   * upstream -> 10ms; combined aggregates across parallel inference -> ~33ms). The per-model
   * BATCH size is per_model_max_streams above; the combined-mux batch is auto-derived
   * (out_slots = unified ? per-model batch x2 : x1) and is NOT settable. */
  guint     mux_width;               /** mux surface resolution, all muxers (default 1920) */
  guint     mux_height;              /** mux surface resolution, all muxers (default 1080) */
  gboolean  mux_live_source;         /**< 1=live/RTSP, 0=file -- a source property (def TRUE) */
  guint     model_mux_push_timeout;  /** per-model mux flush us (default 10000 = 10ms)      */
  guint     combined_mux_push_timeout; /** combined mux flush us (default 33333 ~= 30fps)   */

  /** [multimodel] perf metrics: when attach_perf_metric is on, the plugin measures per-model
   * (aggregate) and per-(stream,role) fps + inference latency and reports them additively in
   * model/status. OFF (default) => zero overhead, model/status byte-identical to before. */
  gboolean  attach_perf_metric;       /** enable fps/latency measurement (default FALSE)     */
  guint     perf_metric_interval_sec; /** measurement window seconds (default 5, clamp >=1)  */

  /** [multimodel] buffer-copy-mode: which per-(stream,role) lane(s) DEEP-COPY the decoded frame
   * via an nvvideoconvert (disable-passthrough=1) spliced BEFORE that lane's queue. A copy lane
   * releases the upstream decoder buffer at fan-out, so a SLOW lane can no longer backpressure the
   * decoder when per-model fps differ; the other (passthrough) role stays zero-copy (no element
   * added). Config string none|primary|shadow|both -> these two derived flags. Default: shadow.
   * "none" => byte-identical to before (no converter on any lane).                              */
  gboolean  copy_primary_buffer;      /** copy on the PRIMARY lane (nvvideoconvert dp=1; def 0) */
  gboolean  copy_shadow_buffer;       /** copy on the SHADOW  lane (nvvideoconvert dp=1; def 1) */
  guint     copy_buffer_pool_size;    /** copy-lane nvvideoconvert output-buffers (0 = element
                                       *   default). Size to the slow lane's depth to keep the
                                       *   copy from backpressuring the converter.            */

  /** [multimodel] lane-leaky: leak policy of the per-(stream,role) in_q/out_q lane queues.
   * TRUE (default, existing behaviour): leaky=downstream -- a full lane drops its OLDEST
   * buffer instead of blocking, so a stalled/slow model can never backpressure the shared
   * tee + input demux and stall the OTHER streams (freshness over completeness; live
   * sources keep moving when per-model fps mismatch). FALSE: non-leaky lanes guarantee
   * COMPLETENESS (every frame reaches every routed model) but a slower A/B twin then
   * backpressures the lane and can stall mismatched-fps siblings sharing the tee.       */
  gboolean  lane_leaky;

  /** [multimodel] gpu-id: the PIPELINE device -- where the decoders, the shared input
   * demux and the combined display mux live (default 0). A model bin ON this device
   * carries NO migration elements (mux links straight to nvinfer); a bin pinned to
   * ANOTHER device carries an nvdsxfer pair: xfer-in migrates the batched surfaces to
   * the shard's device (nvinfer rejects cross-GPU input memory), xfer-out brings the
   * batch back to THIS device so demux/display/OSD stay on the pipeline gpu.
   * Cross-device placement requires CUDA peer access (NVLink/PCIe P2P) + the nvdsxfer
   * factory; otherwise the placement is rejected at bin creation. */
  guint     gpu;

  /** [multimodel] placement-policy (pack|spread): how unconstrained streams are
   * placed across a multi-GPU instance group. See ModelPlacementPolicy. */
  gint      placement_policy;

  /** [multimodel] auto-gpu-scale (default 0): ELASTIC multi-GPU scale-out without
   * any per-model gpu configuration. When every shard of a model is at capacity
   * and a NEW shard must grow, pick the healthiest OTHER device instead of the
   * model's own -- but ONLY when that device is IDENTICAL to the model's (same
   * compute capability + device name), so the existing TensorRT engine is
   * legally reused (deserialize in seconds; a mismatched device would silently
   * trigger a minutes-long rebuild). The overflow shard is UNPINNED: compaction
   * migrates its streams back and reclaims it when load drops (scale-in).
   * Heterogeneous boxes / single-GPU: behaves exactly as auto-gpu-scale=0. */
  gboolean  auto_gpu_scale;

  /** model registry: built from [MultiModelInference] default primary (+ optional
   * shadow). No per-camera mapping -- every stream uses the default models. */
  /** MODEL CATALOG: model NAME -> base config (+ auto-detected backend type), parsed
   * from the [model-name-<idx>] blocks. A catalog entry is metadata only -- it is NOT
   * instantiated until a role ref (default_primary/shadow) or the model/load API loads
   * a concrete (name, version) instance from it. version/engine_file on a catalog entry
   * are unset (per-instance, runtime). */
  ModelCatalogEntry     models[MM_MAX_MODELS];
  guint          num_models;
  /** PER-VERSION artifacts: optional engine/config overrides per (name, version), parsed from
   * [model-<name>-<version>] blocks. Consulted when a (name,version) is instantiated (default,
   * per-sensor, or load). Absent => the model's base catalog config/engine is used. */
  ModelVersionArtifact versions[MM_MAX_MODELS * 4];
  guint          num_versions;
  /** guards versions[] / num_versions -- the ONLY ModelMuxConfig state mutable at
   * RUNTIME: modelmux_config_register_version (model/load|update trigger) and
   * modelmux_control_registry_unregister_version (OTA rollback) mutate on the main loop
   * under api_control_lock, while GET model/status resolves placements on the
   * CIVETWEB thread under ib->lock only. Readers may already hold api_control_lock
   * and/or ib->lock, so the LOCK ORDER is (api_control_lock | ib->lock) ->
   * versions_lock, and versions_lock NEVER wraps another lock: critical
   * sections copy out what they need and release -- never file I/O
   * (modelmux_config_derive runs strictly after unlock). Initialised wherever the
   * ModelMuxConfig is (re)zeroed (modelmux_config_parse / gst_modelmux_build defaults);
   * cleared in modelmux_config_clear. */
  GMutex         versions_lock;
  gchar         *default_primary;          /** default primary model NAME (catalog ref) */
  gchar         *default_primary_version;  /** default primary model VERSION (e.g. 1) */
  gchar         *default_shadow;           /** default shadow model NAME (NULL=>none)    */
  gchar         *default_shadow_version;   /** default shadow model VERSION (NULL=>none)  */
  gint           default_primary_gpu;      /** requested instance placement of the runtime
                                            *   default (stream/route default{}.gpu); -1 =
                                            *   any instance. Applied to every stream that
                                            *   RESOLVES to the default (add/bind/sweep).  */
  gint           default_shadow_gpu;

  /** config-time per-sensor model bindings ([stream-model-<sensor_id>] groups).
   * Empty => every stream uses the defaults (fully backward-compatible). */
  StreamModelBinding bindings[MM_MAX_STREAMS_DEFAULT];
  guint           num_bindings;
} ModelMuxConfig;

/* A/B provenance user-meta: InferenceProvenanceMeta is now PUBLIC, defined in
 * nvdsmeta_schema.h (tagged NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE), so downstream
 * DS components can parse it without this private header. It is attached as
 * NvDsUserMeta on EACH frame before the combined display mux (ModelBin nvinfer
 * src for inferred frames, passthrough queue src for no-infer frames), and lets
 * nvmodelmux restore source_id after the combined mux rewrites it to a display
 * slot index. Flat struct => the copy func is a single memcpy that survives the
 * combined nvstreammux. */

/** ModelBin lifecycle (applies to EVERY ModelBin, any role). Lifecycle reads
 *  WARMING -> WARMED -> SERVING (and any state -> FAILED on a load error). These are
 *  DOMAIN states, intentionally distinct from GstState (NULL/READY/PAUSED/PLAYING):
 *  a WARMED bin is already in GST_STATE_PLAYING with its engine in VRAM, just idle. */
typedef enum
{
  MODEL_WARMING = 0,     /** engine still building/loading; cannot attach yet           */
  MODEL_WARMED,          /** engine loaded, idle (0 streams); ready to take streams     */
  MODEL_SERVING,         /** engine loaded, >=1 stream attached                         */
  MODEL_FAILED           /** engine warm-up FAILED; not usable; unloadable + reloadable */
} ModelStatus;

/* --------------------------------------------------------------------------------------- */
/* ModelBin: one logical model NAME for one ROLE.                                          */
/*                                                                                         */
/*   ghost sink_<idx>  ->  nvstreammux  ->  nvinfer  ->  nvstreamdemux  -> ghost src_<idx> */
/*                                                                                         */
/* Multiple streams share a single ModelBin (true batched inference). A recycled           */
/* index pool keeps mux/demux pad ids bounded (sink_0..sink_{max-1}).                      */
/*                                                                                         */
/* Lifecycle (role-agnostic): created at NULL (NOT_READY) -> loaded async to               */
/* READY (engine warmed) -> SERVING when streams attach -> back to READY when the          */
/* last stream leaves. The bin is only destroyed by an explicit unload.                    */
/* --------------------------------------------------------------------------------------- */
typedef struct
{
  gchar       *name;  /** Modelbin name */
  gchar       *role;   /** Primary/Shadow */
  ModelType   type;     /** nvinfer vs nvinferserver */

  /** perf metrics (NULL unless [multimodel] attach-perf-metric is on). `perf_on` is the cached
   * hot-path gate so the provenance probe only does a single bool test when disabled. */
  gboolean     perf_on;
  ModelMuxPerf      *perf;

  gchar       *config_file;
  guint        unique_id;     /** nvinfer gie-unique-id (for provenance)       */
  guint        max_streams;

  /** per-model nvstreammux properties (from [muxer]; captured at create so the
   * drain-recovery recreate path reuses the same values). */
  guint        mux_width;
  guint        mux_height;
  guint        mux_push_timeout;
  gboolean     mux_live_source;
  gint         mux_gpu;       /** PIPELINE device ([multimodel] gpu-id): the model mux
                               *   batches decoder buffers, which live on that device --
                               *   its own allocations must land there too.            */

  GstElement  *bin;           /** the GstBin                                  */
  GstElement  *mux;           /** nvstreammux (legacy)                        */
  GstElement  *conv_in;       /** xfer-in (nvdsxfer) mux->infer: migrates the batch
                               *   to THIS shard's device -- nvinfer rejects input
                               *   surfaces on another GPU. Present ONLY on a
                               *   cross-device bin; NULL on a same-device bin
                               *   (mux links straight to infer, zero overhead).
                               *   Cross-device creation REJECTS when P2P or the
                               *   nvdsxfer factory is unavailable.              */
  GstElement  *infer;         /** nvinfer                                     */
  GstElement  *conv_out;      /** xfer-out (nvdsxfer) infer->demux: brings the batch
                               *   back to the PIPELINE device ([multimodel] gpu-id)
                               *   so demux/display aggregate same-device buffers.
                               *   Paired with conv_in; NULL on same-device bins. */
  GstElement  *demux;         /** nvstreamdemux                               */
  GstPad      *zero_stream_drop_pad; /** infer.src ref while drained to zero    */
  gulong       zero_stream_drop_probe; /** drops stale late nvinfer output       */

  GQueue      *free_idx;      /** available pad indices (GINT)                */
  GHashTable  *idx_to_stream; /** idx(GINT) -> stream_id(GINT)                */
  gint        *slot_stream;   /** lock-free mirror of idx_to_stream: idx ->
                               *   stream_id (-1 = free), max_streams entries.
                               *   Written with g_atomic under ib->lock; read
                               *   lock-free by the per-frame provenance probe
                               *   (GHashTable is NOT safe for unlocked reads
                               *   concurrent with insert/remove).            */
  guint        num_streams;
  gboolean     served;        /** TRUE once a stream has ever attached. A bin that
                               *   SERVED then drained to 0 streams must NOT be reused
                               *   in place (legacy nvstreammux does not resume after a
                               *   mid-pipeline drain-to-zero) -- the reroute refreshes
                               *   it to a fresh bin instead. */
  gint         status;        /** ModelStatus, accessed atomically          */
  GThread     *load_thread;   /** background engine warm-up                    */
  const GHashTable *stream_names; /**< borrowed: stream_id(GINT) -> name (overlay) */
  gpointer     pool;          /** owning ModelPool* (back-ptr; for the
                               *   slot-derived role lookup in the overlay)      */
  gchar       *engine;        /** cached engine path (provenance);
                               *   set when the bin warms to READY               */
  gchar       *version;       /** model checkpoint version (provenance/status);
                               *   NULL => fall back to the engine basename       */
  gchar       *key;           /** canonical instance id "name@version" -- the hash
                               *   key in the pool's models map + the limbo map, so
                               *   multiple versions of one name coexist. Distinct
                               *   from `name` (bare, for display/provenance).      */
  guint        inst;          /** shard instance index (0 = base; 1+ = overflow
                               *   shards auto-created when a shard fills). Only
                               *   for unique element names; shards SHARE gie-id.  */
  gint         gpu;           /** PER-SHARD placement recorded at creation (-1 =
                               *   the config's device). gpus[] groups place
                               *   one shard per listed GPU, each with its own
                               *   gpu-derived config.                             */
  gboolean     conv_cross;    /** placement was CROSS-device at creation: the bin
                               *   carries a migration pair. Same-device bins carry
                               *   NO migration elements (mux links directly to the
                               *   inference element) -- a later cross-device re-pin
                               *   is refused (model_bin_set_gpu).             */
  gint         conv_gpu;      /** the migration pair's current TARGET device     */
  gint         cfg_gpu;       /** device the model's own CONFIG FILE pins (nvinfer
                               *   gpu-id / inferserver gpu_ids), parsed ONCE at
                               *   creation; -1 = no key (backend default 0). Last
                               *   fallback of the effective-placement resolution. */
  gboolean     pinned;        /** deliberate gpus[] placement: this shard is
                               *   DESIRED STATE (not overflow) -- never auto
                               *   scaled-in or compacted away; freed only by an
                               *   explicit unload.                                */
  gpointer     next_shard;    /** next ModelBin* of the SAME model (shard chain;
                               *   NULL = last). Same name/engine/gie-id (an
                               *   instance-group shard carries its own per-GPU
                               *   derived config).                                */
  GMutex       prov_lock;     /** guards retired[] + the ota_* rollback stash: the
                               *   nvinfer 'model-updated' confirm callback runs on
                               *   nvinfer's own thread, concurrent with main-loop
                               *   reloads                                     */
  gchar       *ota_old_version; /** rollback COPIES while an OTA awaits nvinfer's
                               *   confirm; committed (freed) on success, swapped
                               *   back on a failed reload                     */
  gchar       *ota_old_engine;
  gchar       *ota_old_config; /** rollback COPY of config_file for a BY-CONFIG
                               *   OTA (NULL when the OTA was engine-only): a
                               *   failed/rejected config reload must restore
                               *   config_file + cfg_gpu, or later grows/effective-
                               *   gpu reads clone/report the rejected config.   */
  gchar       *ota_old_key;   /** identity-rename rollback COPIES (API_DESIGN.md
                               *   5.3): the pre-update ("name@FROM") and
                               *   post-update ("name@TO") keys while the OTA
                               *   awaits confirm -- a failed reload renames the
                               *   live identity BACK (marshalled to the main
                               *   loop; see modelmux_ota_rename_rollback_main). NULL
                               *   when no rename rides this reload.            */
  gchar       *ota_new_key;
  gpointer     ota_ib;        /** owning ModelMuxBin* for the deferred
                               *   rollback rename (a limbo bin has pool==NULL,
                               *   so the confirm callback can't derive it)     */
  gboolean     ota_await;     /** an OTA reload is in flight (set at trigger)  */
  gpointer     ota_group;     /** owning ModelMuxOtaGroup* (bin-internal) while a GROUPED
                               *   model/update awaits per-shard confirms: the whole
                               *   multi-shard/multi-twin update commits or rolls back
                               *   as ONE transaction once the LAST confirm lands.
                               *   Guarded by prov_lock (like the ota_* stash).      */
  gboolean     ota_ok;        /** this shard's confirm outcome, recorded for the
                               *   group resolution (prov_lock)                     */
  gboolean     ota_compensating; /** the in-flight reload is a group-rollback
                               *   COMPENSATING re-trigger of the PREVIOUS engine:
                               *   its confirm must NOT re-enter the group logic
                               *   (log-only; see modelmux_ota_group_resolve_main)        */
  GPtrArray   *retired;       /** OTA-retired provenance strings (old version/engine).
                               *   On an in-place reload the lock-free
                               *   per-frame probe may still hold the old pointer, so the
                               *   superseded string is RETIRED here (not freed) and
                               *   released only at bin teardown. Lazy-allocated; bounded
                               *   by the number of OTA reloads (tiny).             */
} ModelBin;

/**
 * DefaultModelRef: a role's RUNTIME default model DESIGNATION -- embedded in
 * that role's ModelPool ("this pool holds these models, and THIS one is the
 * default"). Designation, never ownership: the default's ModelBin lives in the
 * pool's `models` inventory (or limbo, pre-promotion) like any other model.
 *
 * SEEDED ONCE from the parsed ModelMuxConfig in modelmux_bin_new(); after that
 * the config's default_* fields are never read nor written again -- ALL runtime
 * re-designation (stream/route default{}, model/load default flags, OTA rename
 * commit/rollback, promote/swap default-follows-the-slot) mutates the pool
 * refs, under modelmux_bin->lock (same discipline the config fields had).
 *
 * IDENTITY (name/version/gpu/key) is authoritative: it may name a model whose
 * bin does not exist yet (config-file default before load_defaults; a load that
 * failed or was rejected). `bin` is a DERIVED CACHE only -- the base shard once
 * the identity is loaded, NULL until then. INVARIANT (all writers enforce it):
 *   bin == NULL  ||  bin's canonical key equals `key`.
 * Guards that decide "is X a default?" MUST compare `key` (identity), never
 * `bin` -- a designated-but-unloaded default is still the default.
 */
typedef struct
{
  gchar    *name;             /** model NAME; NULL => no default for this role  */
  gchar    *version;          /** model VERSION ("1", ...); NULL only with name */
  gint      gpu;              /** placement pin (stream/route default{}.gpu);
                               *   MM_GPU_ANY = any instance                    */
  gchar    *key;              /** cached canonical "name@version" (NULL <=> no
                               *   default) -- what every guard compares        */
  ModelBin *bin;              /** derived cache: base shard once loaded; NULL
                               *   until then. Re-pointed by the default
                               *   writers; cleared at the bin-destroy choke
                               *   point. NEVER authoritative.                  */
} DefaultModelRef;

/**
 * ModelPool: the live INVENTORY of one role's (Primary or Shadow) instantiated
 * models, plus the factory that builds them (model_pool_get_or_create: lookup ->
 * promote from limbo -> create + background warm).
 */
typedef struct
{
  gchar       *role;
  guint        base_uid;      /** unique-id base for this role's models       */
  guint        next_uid;      /** monotonic uid allocator (never reused)      */
  guint        max_streams;
  GstElement  *parent_pipeline;
  GstElement  *container;     /** ModelMuxBin GstBin that owns the elements  */
  GHashTable  *models;        /** canonical key "name@version" (gchar*, owned) ->
                               *   BASE-shard ModelBin* (no value destructor: bins
                               *   are torn down explicitly, never by a map remove).
                               *   Two versions of one name = two entries; a model's
                               *   extra shards are NOT in the map -- they chain off
                               *   the base bin via next_shard.                    */
  const ModelMuxConfig *config; /** borrowed, READ-ONLY reference data only:
                               *   model catalog, policy flags (placement/
                               *   sharding/leaky) and the versions registry
                               *   (own versions_lock). NEVER runtime default
                               *   state -- that lives in default_ref below.  */
  const GHashTable *stream_names; /** borrowed: stream_id -> name (overlay)    */
  DefaultModelRef default_ref; /** THIS role's runtime default designation
                               *   (see DefaultModelRef). Written by the
                               *   default writers under the bin lock; read by
                               *   resolution, the guards, status and the dump. */
  gpointer     owner;         /** owning ModelMuxBin* (back-ptr)    */
} ModelPool;

/** Stream placement policy across a multi-GPU instance group
 *  ([multimodel] placement-policy) -- applies only to unconstrained attaches
 *  (MM_GPU_ANY); an explicit route gpu always addresses its instance directly.
 *  PACK (default): saturate instances in gpus[] order -- a stream lands on
 *    the next GPU only when every earlier shard is at max-streams. Maximizes
 *    batch efficiency and leaves later GPUs free (ECS "binpack" / k8s
 *    MostAllocated consolidation semantics; matches the overflow chains' own
 *    first-fit packing).
 *  SPREAD: least-loaded eligible instance (pinned members preferred) --
 *    latency-first, smaller failure blast radius (ECS "spread"). */
typedef enum
{
  PACK_PLACEMENT = 0,
  SPREAD_PLACEMENT
} ModelPlacementPolicy;

/**
 * Placement constraint sentinel: "no specific GPU requested". Every attach /
 * route / promote / swap API below takes a gpu constraint with this contract:
 *   MM_GPU_ANY -> the scheduler (modelmux_pool_shard_for_attach) is free to pick any
 *                 warm shard with capacity (single-GPU behaviour, unchanged);
 *   >= 0       -> selection is restricted to shards whose EFFECTIVE device
 *                 (shard's own gpu, falling back to the version registry, then
 *                 device 0) matches -- scale-out then also stays on that device.
 */
#define MM_GPU_ANY (-1)

/* ------------------------------------------------------------------  */
/* Per-stream ROUTING TABLE                                            */
/*                                                                     */
/* ModelMuxStreamEntry + ModelMuxRoleAttach are the bin's record of    */
/* WHAT IT HAS WIRED UP for every attached stream, so a later detach / */
/* stream-route / promote / compaction can find and unplug exactly the */
/* pads and elements that belong to that stream -- and nothing else.   */
/*                                                                     */
/*   ModelMuxBin.stream_wiring (hash: source_id -> ModelMuxStreamEntry*) */
/*     |                                                               */
/*     +-> StreamEntry { column, demux_src, tee,                       */
/*           prim: RoleAttach  -- lane to the PRIMARY model            */
/*           shad: RoleAttach  -- lane to the SHADOW  model (or off) } */
/*                                                                     */
/* ONE StreamEntry per attached stream; each embeds TWO RoleAttach     */
/* records inline (not allocated), one per role. A RoleAttach is the   */
/* wiring of ONE physical lane:                                        */
/*                                                                     */
/*   tee --> [conv] --> in_q --> ModelBin --> out_q --> out_mux        */
/*                                                                     */
/* i.e. WHICH model this role is routed to plus every element/pad the  */
/* lane occupies (tee src pad, queues, optional copy converter, the    */
/* combined-mux sink pad). Rerouting a role swaps only that lane's     */
/* model hookup; the queues, the other role and all other streams keep */
/* running. model/status and the MM_DEBUG routing dump are walks over  */
/* this table.                                                         */
/*                                                                     */
/* Lifetime: created by modelmux_bin_attach_stream, freed by           */
/* modelmux_stream_teardown on detach. All mutation happens on the main loop */
/* under ib->lock.                                                     */
/* ------------------------------------------------------------------ */

/** One role branch (Primary or Shadow) attached for a stream: the model this
 *  role is currently routed to + every element/pad of its lane, recorded so
 *  the lane can later be swapped (stream/route) or torn down surgically. */
typedef struct
{
  gboolean     active;
  gboolean     busy;          /** a zero-drop swap/clear probe is in flight   */
  gchar       *model;         /** model name routed for this role            */
  ModelPool   *role;          /** owning role bin                            */
  GstElement  *conv;          /** OPTIONAL copy converter: tee -> nvvideoconvert -> in_q
                               *   (NULL = zero-copy passthrough lane). buffer-copy-mode.
                               *   Rides WITH this record on swap, like in_q/out_q.      */
  GstElement  *in_q;          /** queue: [conv ->] in_q -> ModelBin (persistent on swap) */
  GstElement  *out_q;         /** queue: ModelBin -> combined mux (persistent)*/
  GstPad      *tee_src;       /** requested src pad on the per-stream tee     */
  GstPad      *out_mux_sink;  /** requested sink pad on combined display mux  */
  gint         out_idx;       /** recycled combined-mux sink index           */
  GstPad      *tap_pad;       /** debug tap probe pad (held until teardown)  */
  gulong       tap_probe_id;  /** debug tap probe id                         */
  GstPad      *verify_head_pad;   /** debug copy-verify head probe pad        */
  gulong       verify_head_probe_id;
  GstPad      *verify_queue_pad;  /** debug copy-verify queue probe pad       */
  gulong       verify_queue_probe_id;
  gpointer     verify_ctx;    /** debug copy-verify context                  */
  gboolean     passthru;      /** model-sharding=0 overflow: this branch bypasses
                               *   inference entirely -- tee -> queue -> out_mux,
                               *   no model bin (role/model are NULL). Frames flow
                               *   downstream un-inferred (nvinfer-style over-batch). */
  gchar       *pending_model; /** when passthru AND awaiting deferred promotion: the model
                               *   this stream is MAPPED to but which is still warming. Shown
                               *   in the routing table as "<model> (passthru)". NULL for a
                               *   permanent/overflow passthrough (no intended model).        */
  gint         req_gpu;       /** requested instance gpu for this role's model (MM_GPU_ANY =
                               *   no constraint). Recorded on every EXPLICIT attach / route /
                               *   promote and REUSED by the internal re-attach paths (swap,
                               *   recreate, promote fallback), so a stream never silently
                               *   migrates off the device the operator asked for.           */
  gboolean     via_default;   /** TRUE: this role's model was resolved from the pool's
                               *   DefaultModelRef FALLBACK (nothing explicit named it) --
                               *   a runtime default switch may sweep this lane onto the
                               *   new default. FALSE: an explicit metadata ref, explicit
                               *   route or per-camera [stream-model-*] binding chose the
                               *   model (even if it HAPPENS to equal the default) -- the
                               *   lane is PINNED and default switches must not move it.
                               *   Sibling of req_gpu: written wherever the origin is
                               *   known (attach/route/promote commit), preserved by
                               *   internal re-attaches (compaction/recreate).            */
} ModelMuxRoleAttach;

/** Per-stream bookkeeping inside the ModelMuxBin: ONE record per attached
 *  stream (the routing-table row). Holds the stream's entry wiring (its input
 *  demux pad + fan-out tee), its stable display column, and the two embedded
 *  role lanes (prim/shad). Stored in ModelMuxBin.stream_wiring keyed by source_id. */
typedef struct
{
  guint        stream_id;     /** global source_id from nvmultiurisrcbin      */
  gint         column;        /** stable DISPLAY column [0,max_streams): primary
                              *   -> out_mux slot=column (tile row 0), shadow ->
                              *   slot=max_streams+column (tile row 1). Recycled.  */
  guint        branch_serial; /** unique lifetime suffix for deferred-free-safe
                               *   branch-local element names                  */
  GstPad      *demux_src;     /** requested src pad on the bin's input demux  */
  GstPad      *branch_tap_pad; /** debug branch-in tap pad (held until teardown) */
  gulong       branch_tap_probe_id;
  GstElement  *tee;           /** entry point + fan-out to roles              */
  ModelMuxRoleAttach prim;
  ModelMuxRoleAttach shad;
  gpointer     promote_ctx;   /**< live ModelMuxPromoteCtx* while a passthru->model
                               *   promote is mid-drain for THIS stream, else
                               *   NULL. Weak back-pointer (the promote's
                               *   deferred chain owns the ctx): lets a
                               *   mid-drain stream teardown release the
                               *   ctx-owned retired display slot SYNCHRONOUSLY
                               *   instead of waiting for the next drain-poll
                               *   tick (a racing re-add reusing the freed
                               *   column would find 'sink_<col>' still taken).
                               *   Set/cleared under ib->lock.               */
} ModelMuxStreamEntry;

/* --------------------------------------------------------------------  */
/* Frame accounting (MM_DEBUG only): per-(source,role) drop detection.   */
/*                                                                       */
/* Counts the SAME source frame at three points along its path so a      */
/* drop can be both DETECTED and LOCALIZED:                              */
/*   in       -- entered the bin              (in_demux probe)           */
/*   inferred -- passed the role's nvinfer    (provenance probe)         */
/*   delivered-- left the combined mux        (out-mux probe)            */
/* gap = in - delivered: steady = in-flight latency (OK); GROWING = a    */
/* real drop; <0 = make-before-break swap overlap. Counters are bumped   */
/* LOCK-FREE (g_atomic) on streaming threads; the *_base fields are set  */
/* on the main loop at role-attach / dump. Indexed by source_id (flat).  */
/* ------------------------------------------------------------------ */
#define MM_ACCT_MAX      256   /**< max source_id tracked (flat, lock-free)     */
#define MM_ROLE_PRIMARY  0
#define MM_ROLE_SHADOW   1
#define MM_ROLE_SLOTS    2

typedef struct
{
  gint  in_base;      /** source in_count at this role's epoch start           */
  gint  infer_base;   /** role infer count at epoch start                      */
  gint  out_base;     /** role delivered count at epoch start                  */
  gint  infer;        /** frames through this role's nvinfer (cumulative)      */
  gint  out;          /** frames delivered downstream for this role (cumul.)   */
  gint  last_out_fn;  /** last delivered frame_num                             */
  gint  warm;         /** dumps seen since attach (skip pipeline-fill warmup)  */
  gint  floor;        /** established steady in-flight depth (min gap post-warm)*/
  gint  over;         /** consecutive dump samples with buffered above floor+slack
                       *   (a 1-sample excursion is in-flight churn, not loss)   */
  gint  last_src_fn;  /** MM_FRAME_TRACE: last delivered ORIGINAL frame_num, for
                       *   per-source contiguity (gap = real one-time drop). -1 =
                       *   none delivered yet this epoch.                         */
} ModelMuxAcctRole;

typedef struct
{
  gint        in_count;          /** frames that entered the bin (in_demux)    */
  gint        last_in_fn;        /** last input frame_num                      */
  ModelMuxAcctRole  role[MM_ROLE_SLOTS];/** [0]=Primary, [1]=Shadow                  */
} ModelMuxAcct;

/** One ACCEPTED-but-not-yet-applied stream/route, mirrored down from the control
 *  layer so the status view can see it.
 *
 *  A POST that targets a still-warming model is accepted (202) and parked on the
 *  element's pending_reroutes list until the target is READY. The stream keeps
 *  serving its OLD model meanwhile, so without this mirror model/status shows no
 *  trace of the request at all -- and a reconciler polling it cannot tell an
 *  in-flight route from one that was dropped.
 *
 *  It is a MIRROR, not a reference: the control structs it copies from are freed
 *  the moment their reroute applies or is abandoned, and the status probe runs on
 *  a different thread under a different lock (ib->lock, not api_control_lock).
 *  Deep copies under ib->lock keep the two lifetimes independent. The whole array
 *  is REPLACED wholesale on every change rather than patched entry by entry, so
 *  there is no add/remove bookkeeping to drift out of sync.
 *
 *  Deferred PROMOTES are deliberately not mirrored: a passthrough stream awaiting
 *  one already reports its target via prim.pending_model, and mirroring it too
 *  would report the same intent twice. */
typedef struct
{
  guint   *src;               /** in-scope source ids; NULL/n==0 => ALL streams  */
  guint    n;
  gchar   *primary;           /** deferred primary target key, or NULL           */
  /** The shadow lane is TRI-state, and absence alone cannot carry it:
   *    shadow=NULL, clear_shadow=FALSE -> leave the shadow UNCHANGED
   *    shadow=<key>                    -> set it to that model
   *    shadow=NULL, clear_shadow=TRUE  -> CLEAR it ("shadow": null in the POST)
   *  Collapsing the first and last would make "route the primary" and "route the
   *  primary AND drop the shadow" read identically while deferred -- the reader
   *  could not predict whether an A/B pair survives the switch. */
  gchar   *shadow;            /** deferred shadow target key, or NULL            */
  gboolean clear_shadow;      /** TRUE: the apply DROPS the shadow lane          */
  gint     p_gpu;             /** requested placement per role (MM_GPU_ANY=any)  */
  gint     s_gpu;
  gboolean p_via_default;     /** origin the deferred apply will record           */
  gboolean s_via_default;
} ModelMuxPendingRouteNote;

/* --------------------------------------------------------------------------- */
/* ModelMuxBin: the reusable inference subgraph.                               */
/*                                                                             */
/*   GhostPad sink (batched) -> nvstreamdemux -> per-stream tee                */
/*       -> { Primary ModelPool, Shadow ModelPool }                            */
/*   role outputs -> combined nvstreammux -> GhostPad src (one batched output) */
/*                                                                             */
/* The bin accepts ONE batched src (from nvmultiurisrcbin) on its sink GhostPad*/
/* and decouples it internally, so the bin's pad is its first/entry element.   */
/*                                                                             */
/* --------------------------------------------------------------------------- */
typedef struct
{
  GstElement *bin;           /** ModelMuxBin GstBin */
  GstElement *parent_pipeline; /** parent pipeline */

  const ModelMuxConfig *config; /** multimodel config */
  guint max_streams; /** max concurrent streams batch cap */
  gboolean unified; /** flag to enable unified A/B display batch */

  gint         active_streams; /** lock-free mirror of g_hash_table_size(streams), kept in
                               *   sync under lock. Read by the input drain-guard probe on the
                               *   streaming thread to swallow the upstream mux's idle "clear"
                               *   buffer when 0 streams are attached (see drain-guard probe). */
  GHashTable  *stream_names;  /**< stream_id(GINT) -> gchar* name (for overlay) */
  /** stream_id(GINT) -> gchar* ROUTABLE camera_id (stream/add's `camera_id`).
   *
   *  Kept SEPARATE from stream_names on purpose. stream_names holds the DISPLAY
   *  name (stream/add's `camera_name`, falling back to camera_id when the host
   *  supplied only one), which is what the overlay and model/status show -- but
   *  a display name is NOT routable: POST /api/v1/stream/route resolves its
   *  `streams` scope by camera_id and rejects a display name with
   *  STREAM_UNKNOWN. Storing only the merged name is why the routing READ had to
   *  be assembled in nvmultiurisrcbin, joining this element's answer against the
   *  source manager's sensor list under a different lock.
   *
   *  Holding the real id here lets the routing snapshot be built where the
   *  routing state already lives. Populated and torn down in lockstep with
   *  stream_names; NULL entry when the host supplied no camera_id at all. */
  GHashTable  *stream_cam_ids;
  GHashTable  *stream_wiring; /**< stream_id(GINT) -> ModelMuxStreamEntry       */
  GPtrArray   *retired_names; /** retired name_slot strings (lazy-allocated)   */
  GHashTable  *streams_pending; /**< SET stream_id(GINT) -> 1: attach (or detach) in
                                *   flight; guards against double attach/detach       */
  GHashTable  *streams_cancelled; /**< SET stream_id(GINT) -> 1: a detach arrived while
                                  *   the attach was still pending -> the attach aborts
                                  *   on completion instead of going live              */
  GMutex       lock;          /**< guards streams/pools for the periodic dump   */
  GMutex       free_lock;     /**< guards pending_frees ONLY (separate from lock so a
                               *   deferred-free idle can deregister even when the
                               *   scheduler already holds lock)                 */

  gboolean     log_enabled;   /**< MM_DEBUG: single switch for ALL debug logs    */
  guint        debug_interval; /**< MM_DEBUG_INTERVAL: dump cadence in frames     */

  GstPad      *sink_ghost;    /** ModelMuxBin batched sink (entry)            */
  GstElement  *in_demux;      /** splits the batched input -> per-stream      */
  GstElement  *out_mux;       /** combined display nvstreammux                */
  GstPad      *src_ghost;     /**< ModelMuxBin batched src                    */

  GQueue      *col_free_idx;  /**< recycled DISPLAY columns [0,max_streams)     */

  ModelPool   *primary_pool;
  ModelPool   *shadow_pool;

  /* Type-less model/load: a model loaded via the API has NO primary/shadow type.
   * It is warmed into this "limbo" map (role Unknown) inside the shared container,
   * and PROMOTED into primary/shadow on first use (stream-add or model/update) --
   * pure bookkeeping, zero-drop. Stays here (READY) until explicit model/unload.   */
  GHashTable  *limbo_models;  /**< canonical key "name@version" -> ModelBin* (warm,
                               *   role Unknown; promoted into a role pool on first
                               *   use -- the map ENTRY moves, the bin lives on)   */
  guint        next_limbo_gie;      /**< gie-unique-id allocator for limbo models      */

  gchar       *name_slot[MM_ACCT_MAX]; /**< lock-free mirror of stream_names:
                               *   source_id -> name, atomically published under
                               *   ib->lock; read lock-free by the per-frame
                               *   provenance probes (the hash is for LOCKED
                               *   readers only). Superseded strings are RETIRED
                               *   (a probe may still hold the old pointer) and
                               *   freed only at bin teardown, after the graph is
                               *   quiesced. Bounded by attach/detach churn.    */

  GList       *pending_frees; /**< ModelMuxFreeCtx* deferred branch-teardowns awaiting g_idle;
                               *   cancelled in _free() so a queued idle never derefs a
                               *   freed bin (teardown-race UAF guard)            */
  GList       *pending_sources; /**< ModelMuxTrackedSource* deferred dynamic-op idles/timeouts;
                               *   guarded by free_lock and cancelled before graph teardown */
  guint        compact_poll_id; /**< debounce g_timeout id for shard compaction (0=none) */
  guint        dump_source_id; /**< pending debug dump source id, under free_lock */
  guint        perf_timer_id;   /**< perf-metric tick g_timeout id (0=none/off)          */
  guint64      routing_revision; /**< monotonic; bumps on every APPLIED stream/route,
                                 *   default switch or model/update (under ib->lock);
                                 *   surfaced by model/status for if_revision CAS      */
  GPtrArray   *pending_routes; /**< ModelMuxPendingRouteNote*: routes ACCEPTED but still
                               *   waiting on a warming target, mirrored from the control
                               *   layer under ib->lock (NULL = none queued). Read-only
                               *   here -- status surfaces it, nothing routes off it.  */
  /* OTA rename-rollback hook (control layer): invoked on the MAIN loop by
   * modelmux_ota_rename_rollback_main AFTER a failed update's identity rename was
   * rolled back, with NO bin locks held at the call moment -- from_key = the
   * rejected update's key (rolled back FROM), to_key = the restored serving
   * key. Lets the owner re-sync its own key-addressed state. */
  void (*ota_rollback_cb) (gpointer owner, const gchar * from_key,
      const gchar * to_key);
  gpointer     ota_rollback_owner;
  gint         shutting_down;  /**< set before free so async sources stop rescheduling */
  gboolean     ever_attached; /**< TRUE once any stream has attached. Distinguishes the
                               *   FIRST-EVER attach (fresh elements -> no flush needed)
                               *   from a REFILL after drain-to-zero (stale state -> flush).
                               *   Gating the flush-refill to refills only removes a
                               *   sticky-event race on the common first-attach path.    */
  gint         frame_count;   /**< batches seen on the combined-mux src; written
                               *   atomically by the streaming thread (probe) and reset
                               *   to 0 on the main loop (drain-to-zero teardown) --
                               *   use g_atomic_int_{add,set,get} at every access. Typed
                               *   gint (not guint) so g_atomic_int_* needs no aliasing cast */

  ModelMuxAcct       acct[MM_ACCT_MAX]; /**< MM_DEBUG per-(source,role) frame accounting*/
  /* attach-perf-metric: per-source THROUGHPUT (frames/sec leaving the bin) for
   * PASSTHROUGH streams -- which run no nvinfer, so they have no ModelMuxPerf bin
   * counter. Bin-lifetime (indexed by source_id, like acct[]) so the recording
   * probe never dangles. Single writer per source (one passthrough lane), so the
   * lock-free ModelMuxPerfCounter record is safe; ticked under ib->lock by the perf
   * timer; read by the perf table + model/status. No inference latency exists on
   * this path, so only the rate is meaningful. */
  ModelMuxPerfCounter thru_perf[MM_ACCT_MAX];
} ModelMuxBin;

/* ------------------------------------------------------------------ */
/* API                                                                */
/* ------------------------------------------------------------------ */

/* modelmux_config_parse takes the GObject element (GstNvModelMux) and is declared in
 * gstnvmodelmux.h -- it reconciles the parsed file with self->is_prop_set. */
void     modelmux_config_clear  (ModelMuxConfig * config);
/* Canonical model-instance key. model_key joins (name, version) into a freshly
 * allocated "name@version" (version defaults to MM_MODEL_VERSION_DEFAULT when NULL/empty);
 * used as the pool/limbo hash key so versions of one name coexist. model_key_split
 * splits a key back into bare name + version (both newly allocated; pass NULL to skip). */
gchar   *model_key (const gchar * name, const gchar * version);
void     model_key_split (const gchar * key, gchar ** name_out, gchar ** version_out);
/* TRUE iff `v` is a valid model version: a positive integer with no leading zero
 * ("1","2",...,"42"). Rejects NULL/empty, "0", "01", "1", "1.0", negatives, non-digits.
 * (Triton model-repository convention -- versions are numbered dirs.) */
gboolean modelmux_version_is_int (const gchar * v);
const ModelCatalogEntry *modelmux_config_find_model (const ModelMuxConfig * config, const gchar * name);
/* Auto-detect the inference backend from a model's config-file (nvinferserver protobuf
 * 'infer_config {' vs nvinfer INI '[property]'); defaults to nvinfer. modelmux_infer_plugin_str
 * returns the gst element name ("nvinfer" / "nvinferserver") for logs + element creation. */
ModelType  modelmux_detect_model_type (const gchar * config_file);
const gchar *modelmux_infer_plugin_str (ModelType type);
/* Derive a per-(name,version) config with this load's ARTIFACT + PLACEMENT applied:
 * `engine` (optional) swapped into the engine key and/or `gpu` (optional, -1 = keep the
 * config's device) written into the gpu key. Handles BOTH backends:
 *   nvinfer       : model-engine-file + gpu-id under [property] (INI) / property: (YAML)
 *   nvinferserver : gpu_ids: [N] inside infer_config (protobuf text or YAML). A raw
 *                   .engine does not map to Triton (its model repository owns the
 *                   checkpoints) -> warned + skipped; the gpu placement still applies.
 * Returns the derived config path (caller g_free), or NULL when nothing could be derived
 * (caller uses the base config as-is). The derived filename is tagged with BOTH
 * `model_name` and `version` (e.g. trafficcamnet_1_config_infer_primary.txt). */
gchar       *modelmux_config_derive (const gchar * base_cfg, const gchar * engine, gint gpu,
                               const gchar * model_name, const gchar * version);
/* Record/refresh the per-version artifact + placement of a runtime model/load or
 * model/update, so later lazy warms, in-place updates and gpu-addressed routing resolve
 * this version's engine/config/device. gpu=-1 keeps any previously recorded placement. */
/* batch: 0 = leave the version's batch as-is (unset on a fresh entry). */
void         modelmux_config_register_version (ModelMuxConfig * config, const gchar * name,
                                         const gchar * version, const gchar * engine,
                                         const gchar * config_file, gint gpu, guint batch);
/* Placement of a loaded (name, version): the gpu recorded at load/update time, or -1
 * when the version runs on its config's device (gpu 0 unless the config overrides). */
/* Per-version batch (0 = unset; resolve name-level/catalog then global). Locked scan. */
guint        modelmux_config_version_batch (const ModelMuxConfig * config, const gchar * name,
                                          const gchar * version);
gint         modelmux_config_version_gpu (const ModelMuxConfig * config, const gchar * name,
                                    const gchar * version);
/* Resolve the EFFECTIVE config path for a (name, version) from its per-version artifact
 * ([model-<name>-<version>] block): a custom config and/or an engine swapped into it.
 * Returns a newly-allocated config path (caller g_free) when an override applies, or NULL
 * when the version has no override (caller uses the catalog base config as-is).
 * *derive_failed (optional) is set TRUE when the version DOES carry an explicit engine/gpu
 * override but deriving it FAILED -- the caller must then refuse to instantiate rather than
 * fall back to the base config (which would run the wrong artifact/device). */
gchar       *modelmux_config_resolve_version_cfg (const ModelMuxConfig * config, const gchar * name,
                                            const gchar * version,
                                            gboolean * derive_failed);
/* Device a model CONFIG FILE pins its inference to: nvinfer keyfile "gpu-id=N" or
 * nvinferserver pbtxt "gpu_ids: [N]". -1 = no key present (backend defaults to 0). */
gint         modelmux_config_file_gpu (const gchar * config_file);
/* modelmux_gpu_stats (whole-device NVML stats) lives in gstnvmodelmux_perf.h -- the
 * module header its definition includes, so the prototype is visible there
 * (tmake builds with -Werror=missing-prototypes). Included above. */
/* Register a model at runtime (idempotent) so a stream can use a shadow/primary
 * that was not the configured default. Returns the (existing or new) entry. */
const ModelCatalogEntry *modelmux_config_register_model (ModelMuxConfig * config, const gchar * name,
                                            const gchar * config_file,
                                            const gchar * engine_file,
                                            const gchar * version,
                                            guint max_streams);
/* Update an ALREADY-registered model's artifacts in place (OTA): overwrites
 * config_file/engine_file/version when a non-NULL/non-empty value is supplied (others
 * left as-is). Returns the entry, or NULL if the name is not registered. */
const ModelCatalogEntry *modelmux_config_update_model (ModelMuxConfig * config, const gchar * name,
                                          const gchar * config_file,
                                          const gchar * engine_file,
                                          const gchar * version);
/* Resolve a stream's (primary, shadow) refs: config-time per-sensor binding
 * (matched by camera id) wins; any role it leaves unset falls back to that
 * role's RUNTIME default ref (the pool-embedded designation). Bindings are
 * static config; defaults are runtime state -- two sources, one resolution. */
void     modelmux_config_resolve_models (const ModelMuxConfig * config,
                                   const DefaultModelRef * def_primary,
                                   const DefaultModelRef * def_shadow,
                                   const gchar * camera_id,
                                   const gchar ** primary, const gchar ** shadow,
                                   const gchar ** primary_version,
                                   const gchar ** shadow_version,
                                   gboolean * primary_is_default,
                                   gboolean * shadow_is_default);

/* ModelBin */
ModelBin *model_bin_new (const gchar * name, const gchar * role,
                              const gchar * config_file, const gchar * engine_file,
                              const gchar * version, guint unique_id, guint max_streams,
                              const ModelMuxConfig * config,
                              GstElement * parent_pipeline, GstElement * container,
                              const GHashTable * stream_names, guint inst);
GstPad     *model_bin_attach (ModelBin * model_bin, guint stream_id);
gboolean    model_bin_detach (ModelBin * model_bin, guint stream_id);
gboolean    model_bin_has_capacity (ModelBin * model_bin);
guint       model_bin_num_streams (ModelBin * model_bin);
/* Warm the engine in the background so the bin reaches READY (idempotent). */
void        model_bin_load (ModelBin * model_bin);
ModelStatus model_bin_status (ModelBin * model_bin);
const gchar *model_status_str (ModelStatus s);
/* Borrowed model version identity used by provenance, diagnostics and status.
 * The returned pointer is owned by the model bin and remains valid for readers. */
G_GNUC_INTERNAL const gchar *model_version_str (ModelBin * model_bin);
/* Backend-aware engine/identity string captured from a warmed model bin.
 * Returns a newly-allocated string; caller frees. */
G_GNUC_INTERNAL gchar *get_infer_engine_str (ModelBin * model_bin);
void        model_bin_free (ModelBin * model_bin);

/* ModelPool */
ModelPool  *model_pool_new (const gchar * role, guint base_uid, guint max_streams,
                             GstElement * parent_pipeline, GstElement * container,
                             const ModelMuxConfig * config, const GHashTable * stream_names);
GstPad     *model_pool_attach (ModelPool * rb, guint stream_id,
             const gchar * model_name, gint gpu, GstPad * in_pad,
             guint branch_serial,
             GstElement ** in_q_out, GstElement ** out_q_out,
             GstElement ** conv_out, gboolean * cap_overflow,
             ModelBin ** attached_bin_out);
gboolean    model_pool_detach (ModelPool * rb, guint stream_id, const gchar * model_name);
/* Pre-create + warm a model in this pool so it is ready before streams arrive. */
/* batch: explicit per-instance cap (0 = resolve: version registry -> catalog -> global). */
gboolean    model_pool_load_model (ModelPool * rb, const gchar * model_name,
                                          const gchar * cfg_override, guint batch);
/* Explicit teardown of an IDLE (READY, 0-stream) model. Returns FALSE if it is
 * still serving streams. ModelBins are otherwise never destroyed on detach. */
gboolean    model_pool_unload_model (ModelPool * rb, const gchar * model_name);
void        model_pool_free (ModelPool * rb);
/* Internal read-only helper for status/perf reporting. Caller must already hold
 * the owning ModelMuxBin lock while traversing pool membership. */
G_GNUC_INTERNAL ModelBin *modelmux_perf_find_bin_for_source (ModelPool * pool,
                                                         guint source_id);
G_GNUC_INTERNAL void modelmux_role_set (ModelBin * model_bin, const gchar * role);
G_GNUC_INTERNAL const gchar *modelmux_role_get (ModelBin * model_bin);
G_GNUC_INTERNAL const gchar *modelmux_resolve_role (ModelBin * model_bin, guint source_id);
G_GNUC_INTERNAL gint modelmux_role_idx (const gchar * role);
G_GNUC_INTERNAL void modelmux_fmt_route_cell (gchar * buf, gsize n,
                                        const ModelMuxRoleAttach * ra);
G_GNUC_INTERNAL gboolean modelmux_debug_enabled (void);
G_GNUC_INTERNAL void modelmux_debug_configure (ModelMuxBin * ib);
G_GNUC_INTERNAL void modelmux_schedule_dump (ModelMuxBin * ib,
                                       guint delay_seconds);

/* ModelMuxBin */
ModelMuxBin *modelmux_bin_new (GstElement * pipeline, const gchar * name,
                                      guint max_streams, const ModelMuxConfig * config);
GstPad     *modelmux_bin_get_sink (ModelMuxBin * ib);
/* p_gpu/s_gpu: requested instance placement per role (MM_GPU_ANY = no constraint).
 * p_via_default/s_via_default: TRUE when that role's model came from the DEFAULT
 * fallback (see ModelMuxRoleAttach.via_default). */
/* name    = DISPLAY name (overlay, model/status) -- may be the camera_id when the
 *           host supplied only one identifier.
 * cam_id  = the ROUTABLE camera_id from stream/add, or NULL when none was given.
 *           Kept apart from `name` because only this one is accepted by
 *           POST /api/v1/stream/route (see ModelMuxBin.stream_cam_ids). */
gboolean    modelmux_bin_attach_stream (ModelMuxBin * ib, guint stream_id,
                                            const gchar * name, const gchar * cam_id,
                                            const gchar * primary, const gchar * shadow,
                                            gint p_gpu, gint s_gpu,
                                            gboolean p_via_default, gboolean s_via_default);
gboolean    modelmux_bin_detach_stream (ModelMuxBin * ib, guint stream_id);
/* Update the ROUTING TABLE for one stream: re-point its primary and/or shadow to
 * new model(s). Re-wires only the role(s) whose model changed, under a single
 * pad-block, leaving the unchanged role running. (Atomic per-stream model swap.)
 * p_gpu/s_gpu constrain WHICH instance serves each role (MM_GPU_ANY = any). */
gboolean    modelmux_bin_update_routing (ModelMuxBin * ib, guint stream_id,
                                             const gchar * primary, const gchar * shadow,
                                             gint p_gpu, gint s_gpu,
                                             gboolean p_via_default, gboolean s_via_default);
GstPad     *modelmux_bin_get_src (ModelMuxBin * ib);
/* number of streams currently attached (0 => the bin/sources have drained). */
guint       modelmux_bin_num_streams (ModelMuxBin * ib);
/* Pre-load the configured default primary (+ shadow if set) so they are READY
 * before any stream is added (zero attach-time downtime). */
/* Preload the configured default primary (+ shadow) so they are READY before any
 * stream attaches. wait=TRUE blocks until each default is WARMED and returns FALSE
 * if any FAILED to warm (caller fails the state change); wait=FALSE kicks the async
 * warm and returns TRUE immediately (legacy behaviour). */
gboolean    modelmux_bin_load_defaults (ModelMuxBin * ib, gboolean wait);
/* TYPE-LESS model/load: warm the named model into the limbo map (role Unknown) so
 * it is READY but unassigned. Its primary/shadow type is decided LATER, on first
 * use (stream-add or model/update), which PROMOTES it into the matching pool. The
 * model must already be in the config registry (name/config-file). Idempotent.
 * `gpu` >= 0 PINS the bin to that device BEFORE the warm starts (conv-in placed
 * on the final device from the first buffer); -1 = the config's device. */
ModelStatus modelmux_bin_load_model (ModelMuxBin * ib, const gchar * key,
                                            const gchar * cfg_override, gint gpu,
                                            guint batch);
/* model/load with gpus[]: materialize ONE shard per listed GPU for (name,version) --
 * the base shard from cfgs[0]/gpus[0] plus n-1 PINNED sibling shards, each with its own
 * per-GPU derived config (cfgs[i]; NULL => the base config), all sharing one gie-id and
 * warming in the background like the base. n<=1 degrades to modelmux_bin_load_model
 * (identical single-instance path). ALL-OR-NOTHING: if any sibling shard fails to
 * materialize, every instance this call created (base included) is rolled back and
 * MODEL_FAILED is returned -- a partial group is never silently deployed. Otherwise
 * returns the base shard's status. */
ModelStatus modelmux_bin_load_model_instances (ModelMuxBin * ib,
                                            const gchar * key,
                                            const gchar * const cfgs[],
                                            const gint gpus[], guint n,
                                            guint batch);
/* TRUE iff `key` has a LIVE shard whose effective placement is `gpu` (a shard's own
 * recorded gpu, falling back to the version registry, then device 0). FALSE when the
 * model is not live -- callers fall back to the config registry. */
gboolean    modelmux_bin_model_on_gpu (ModelMuxBin * ib, const gchar * key,
                                            gint gpu);
/* Collect the DISTINCT effective gpus of `key`'s live shard chain into `out` (capacity
 * `cap`); returns the count (0 = not live). Used for engine-coverage validation and
 * teaching error messages. */
guint       modelmux_bin_model_gpus (ModelMuxBin * ib, const gchar * key,
                                            gint * out, guint cap);
/* LIVE instance's batch/stream cap for `key` (base shard's max_streams; 0 = not live).
 * Load/update admission uses it: a loaded version's batch is immutable. */
guint       modelmux_bin_model_batch (ModelMuxBin * ib, const gchar * key);
/* model/load with set_default_primary/shadow: create + warm the model DIRECTLY into the
 * role's pool (bypassing limbo), like a config-time default preload. cfg_override = the
 * per-version/engine-derived config (NULL => catalog base). Returns the model status. */
ModelStatus modelmux_bin_load_into_pool (ModelMuxBin * ib, gboolean shadow,
                                            const gchar * key, const gchar * cfg_override);
/* Re-designate the RUNTIME default for a role to (name, version[, gpu]). "Default" is derived
 * everywhere from the role pool's DefaultModelRef (routing fallback, unload protection, dump,
 * dedup), so this
 * single swap makes every subsystem follow; the prior default auto-demotes. `gpu` >= 0 persists
 * a placement pin for default-resolved streams; gpu < 0 keeps the existing pin when the NAME is
 * unchanged (OTA version rename) and resets it to MM_GPU_ANY on an identity change (a pin
 * belongs to the identity it was set for). Locks internally (callers must NOT hold ib->lock). */
void        modelmux_bin_set_default (ModelMuxBin * ib, gboolean shadow,
                                            const gchar * name, const gchar * version,
                                            gint gpu);
/* Read-only access to a role's runtime default designation (the pool-embedded
 * ref) for the element-side control plane (route guards, resolution). Returned
 * pointer is owned by the pool; main-loop readers are same-thread with all
 * writers, streaming-thread readers must hold ib->lock. */
const DefaultModelRef *modelmux_bin_get_default (ModelMuxBin * ib, gboolean shadow);
/* Collect source-ids of streams currently routed to `key` in the given role (incl. a
 * passthrough stream awaiting promotion to it). Fills `out` (capacity `cap`), returns count.
 * Used to reroute streams off a demoted old default onto the new one. */
guint       modelmux_bin_streams_on_model (ModelMuxBin * ib, gboolean shadow,
                                            const gchar * key, guint * out, guint cap,
                                            gboolean default_only);
/* In-place OTA checkpoint reload: hot-swap the engine (and/or config) on the
 * already-loaded model `name` across ALL its shards -- same bin, same gie, no reroute.
 * `engine` set -> nvinfer "model-engine-file" (engine-only swap); else `config` -> full
 * "config-file-path" reload. `version` updates provenance/status. Returns TRUE if the
 * model was found and the reload was triggered; FALSE if `name` has no live bin. */
gboolean    modelmux_bin_reload_model (ModelMuxBin * ib, const gchar * name,
                                            const gchar * config, const gchar * engine,
                                            const gchar * version);
/* Multi-GPU in-place reload: like modelmux_bin_reload_model, but each shard is
 * swapped to the engine matching ITS effective gpu ({gpus[i] -> engines[i]}, n pairs).
 * ALL-OR-NOTHING admission under one lock hold: if any shard's gpu has no engine in
 * the map, NOTHING is swapped and FALSE is returned (caller reports GPU_ENGINE_MISSING). */
gboolean    modelmux_bin_reload_model_multi (ModelMuxBin * ib,
                                            const gchar * name, const gint gpus[],
                                            const gchar * const engines[], guint n,
                                            const gchar * version);
/* TRUE if `name` has a LIVE bin (limbo or either pool). Distinguishes a first-load
 * (create) from an OTA reload / update-on-missing. */
gboolean    modelmux_bin_model_loaded (ModelMuxBin * ib, const gchar * name);
/* TRUE while any live bin of bare model `name` has an OTA update awaiting nvinfer's
 * confirm (model/load admission gate: keeps the update's freed OLD key reservable
 * for a failed confirm's rename-back). */
gboolean    modelmux_bin_update_in_flight (ModelMuxBin * ib, const gchar * name);
/* Ensure the named model exists + is warming in the given role's pool (idempotent
 * get-or-create + background load). Promotes a limbo model into the role if needed.
 * Used by model/update; NOT called on stream-add. */
ModelStatus modelmux_bin_prepare_model (ModelMuxBin * ib,
                                               gboolean shadow, const gchar * name);
/* QUERY ONLY (no create, no load): current status of an ALREADY-loaded model in
 * the role's pool. Returns MODEL_WARMING/READY/SERVING, or -1 if the model
 * is absent (never loaded). Used by stream-add routing so a requested-but-not-
 * loaded model is NOT auto-loaded -- routing falls back to the default instead. */
gint        modelmux_bin_model_status (ModelMuxBin * ib,
                                            gboolean shadow, const gchar * name);
/* Explicit unload of a model in the given role's pool (model-unload API). Refuses
 * if the model is still SERVING; destroys the bin if READY. Returns TRUE
 * on success. */
/* JSON status dump; schema is always {"models":[...],"streams":[...]}. Filters are
 * INDEPENDENT -- each narrows its own array: f_model -> models[] = that model; f_stream/
 * f_sid (>=0) -> streams[] = that stream; both -> one model + one stream; none -> full.
 * An array whose filter is absent while the other's is present comes back []. g_free it. */
gchar      *modelmux_bin_status_json (ModelMuxBin * ib, const gchar * f_model,
                                           const gchar * f_version, const gchar * f_stream,
                                           gint f_sid);
/* Build the ROUTING-PLANE snapshot for GET /api/v1/stream/route: routes[] +
 * default{} + routing_revision, with streams GROUPED by identical routing state.
 * f_camera = optional strict point lookup on camera_id. NULL (and only NULL) is
 * the full snapshot; any non-NULL value -- INCLUDING "" -- is a PRESENT filter,
 * so an empty or unresolvable id yields an empty routes[] rather than the whole
 * fleet. Takes ib->lock itself.
 * Returns a newly allocated JSON string; caller frees with g_free. */
gchar      *modelmux_bin_route_json (ModelMuxBin * ib, const gchar * f_camera);
/* Bump the routing revision (under ib->lock): call after every APPLIED routing
 * change -- scoped reroute, default switch, in-place update trigger. */
void        modelmux_bin_bump_routing_revision (ModelMuxBin * ib);
/* REPLACE the mirrored list of accepted-but-deferred routes (under ib->lock),
 * taking ownership of `notes` and freeing whatever was there. notes==NULL clears
 * it. Wholesale replacement on purpose -- the control layer republishes its whole
 * queue after every change, so there is no per-entry sync to get wrong. */
void        modelmux_bin_set_pending_routes (ModelMuxBin * ib, GPtrArray * notes);
/* Free a ModelMuxPendingRouteNote (GDestroyNotify for the array above). */
void        modelmux_pending_route_note_free (gpointer note);
/* Sink-ghost query probe that answers /api/v1/model/status custom queries. */
G_GNUC_INTERNAL GstPadProbeReturn modelmux_status_query_probe (GstPad * pad,
                                                         GstPadProbeInfo * info,
                                                         gpointer udata);
/* Sink-ghost query probe that answers the SYNCHRONOUS control-admission query
 * (model/unload today): decides admission on the spot and writes back the
 * verdict + reason. udata is the ModelMuxBin (self = parent_pipeline). */
G_GNUC_INTERNAL GstPadProbeReturn modelmux_control_query_probe (GstPad * pad,
                                                         GstPadProbeInfo * info,
                                                         gpointer udata);
/* Post a model-plane COMPLETION event (model-loaded / -unloaded / -updated /
 * streams-routed) on the pipeline bus so apps react like they do for stream-add.
 * `src` is any element in the pipeline; the message routes up to the app bus. */
G_GNUC_INTERNAL void modelmux_post_model_event (GstElement * src,
                const gchar * event, const gchar * name, const gchar * version,
                gint gpu, gboolean ok, const gchar * detail);
/* Control-plane only: friendly source (camera) name for a source id, or NULL. */
G_GNUC_INTERNAL const gchar * modelmux_bin_source_name (ModelMuxBin * ib, guint sid);
/* Read-only: is (name@version)=`key` loaded (limbo/primary/shadow), and how many
 * streams is it serving across its shard chain? TRUE if present (*streams set).
 * *streams_csv (optional, owned) = the serving stream sensor names ("cam-1,cam-3").
 * Lets model/unload report a precise serving-vs-not-loaded reason up front. */
gboolean    modelmux_bin_key_serving (ModelMuxBin * ib, const gchar * key,
                guint * streams, gchar ** streams_csv);
/* Bare `name`: comma-separated sensor names of every stream serving ANY loaded
 * version of the name (owned; "" if none). For the name-only unload reason. */
gchar *     modelmux_bin_name_serving_streams (ModelMuxBin * ib, const gchar * name);
/* Scoped reroute -- n source ids, or n==0 => all active streams.
 * p_gpu/s_gpu: per-role placement constraint (MM_GPU_ANY = any). */
guint       modelmux_bin_update_routing_scoped (ModelMuxBin * ib, const guint * src_ids,
                                             guint n, const gchar * primary, const gchar * shadow,
                                             gint p_gpu, gint s_gpu,
                                             gboolean p_via_default, gboolean s_via_default);
/* Promote a stream's no-infer PASSTHROUGH primary to a now-WARMED model (deferred attach):
 * tears down the passthrough branch and attaches the model branch in its place. Called by
 * the control-plane poller once the model warms. No-op if the stream is gone / not passthru.
 * p_gpu/s_gpu: per-role placement constraint (MM_GPU_ANY = any). */
void        modelmux_bin_promote_passthru (ModelMuxBin * ib, guint stream_id,
                const gchar * model, const gchar * shadow, gint p_gpu, gint s_gpu,
                gboolean p_via_default, gboolean s_via_default);
/* Tag a passthrough stream with the model it is awaiting (deferred promotion), so the routing
 * table shows "<model> (passthru)" instead of "-". No-op if the stream isn't passthrough.
 * `gpu` records the requested placement the eventual promote must honour (MM_GPU_ANY = any). */
void        modelmux_bin_set_pending_model (ModelMuxBin * ib, guint stream_id,
                const gchar * model, gint gpu, gboolean via_default);
/* Clear an ABANDONED pending tag (deferred promote's target failed/unloaded): the
 * stream reverts to a plain passthrough (no intended model, no placement, no origin). */
void        modelmux_bin_clear_pending_model (ModelMuxBin * ib, guint stream_id);
/* List the streams (within the src_ids scope; n==0 => all active streams) whose PRIMARY is a
 * no-infer PASSTHROUGH branch. *out is allocated to the scope size (caller g_free's);
 * returns the count -- sized dynamically so a route-all can never silently truncate. Used by
 * stream/route: a passthrough stream can't be rerouted through the normal role machinery
 * (no model lane to swap) -- it must go through the deferred-PROMOTE path instead. */
guint       modelmux_bin_passthru_streams (ModelMuxBin * ib,
                const guint * src_ids, guint n, guint ** out);
/* Every wired stream's source_id (caller g_free's *out); returns the count.
 * Materialises a whole-pipeline scope into the fleet it means right now. */
guint       modelmux_bin_active_stream_ids (ModelMuxBin * ib, guint ** out);
/* IN-PLACE move: a promote (primary=<current shadow>, shadow="none") or a
 * clean swap (primary=<old shadow>, shadow=<old primary>) whose scope FULLY COVERS the
 * bins involved is done by re-labeling the existing bins (role+pool+records) instead of
 * building new ones -- no new bin, no engine reload, gie-id preserved, zero frame drop
 * (the bins' pads are never touched). Returns TRUE if handled in place; FALSE if not
 * eligible (caller falls back to the generic create+warm reroute). */
gboolean    modelmux_bin_try_inplace (ModelMuxBin * ib, const guint * src_ids,
                guint n, const gchar * primary, const gchar * shadow);
/* TYPE-LESS model/unload: free the model by name wherever it lives -- the limbo
 * map OR either pool -- when idle. No primary/shadow type needed. */
gboolean    modelmux_bin_unload_model (ModelMuxBin * ib, const gchar * key);
/* Result of a gpu-scoped model/unload (modelmux_bin_unload_model_gpus). */
typedef enum
{
  MM_UNLOAD_GPU_REMOVED = 0,  /* the requested per-gpu instance(s) were removed   */
  MM_UNLOAD_GPU_ALL,          /* the set covered every gpu -> whole version gone  */
  MM_UNLOAD_GPU_ABSENT,       /* (name@version) is not loaded                     */
  MM_UNLOAD_GPU_NO_MATCH,     /* loaded, but no instance sits on any listed gpu   */
  MM_UNLOAD_GPU_SERVING       /* a targeted instance is serving/mid-load: refused */
} ModelMuxUnloadGpuResult;
/* GPU-scoped model/unload: drain ONLY the instance(s) of `key` (name@version)
 * that sit on a gpu in gpus[0..n_gpus) -- the version's other per-gpu instances
 * keep serving. Idle-only per targeted instance (a SERVING/mid-load target
 * refuses the WHOLE op, changing nothing). If the set covers every gpu of the
 * version it degrades to a full unload (MM_UNLOAD_GPU_ALL). On SERVING refusal,
 * *serving_gpu / *serving_streams report the first offending instance so the
 * caller can emit a clear "still inferring" error. */
ModelMuxUnloadGpuResult modelmux_bin_unload_model_gpus (ModelMuxBin * ib,
                const gchar * key, const gint * gpus, guint n_gpus,
                gint * serving_gpu, guint * serving_streams, gchar ** serving_csv);
/* Unload EVERY non-serving version of `name` (bare): serving versions are left intact.
 * Returns the count unloaded; *present (optional) = how many "name@*" versions exist
 * (lets the caller tell "all serving" from "not loaded"). Used when no version given. */
guint       modelmux_bin_unload_versions (ModelMuxBin * ib, const gchar * name,
                guint * present);
void        modelmux_bin_free (ModelMuxBin * ib);

G_END_DECLS

#endif /* __DEEPSTREAM_MULTIMODEL_APP_H__ */
