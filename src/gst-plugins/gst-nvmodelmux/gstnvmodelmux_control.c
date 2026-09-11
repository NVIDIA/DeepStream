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
 * gstnvmodelmux_control.c
 * ================================
 * Native-REST control plane, INTERNAL to the element -- this is what makes the
 * plugin self-contained.
 *
 * nvmultiurisrcbin's embedded REST server turns the /api/v1/stream and
 * /api/v1/model REST calls into custom bus messages (gst_nvmessage_*). We
 * subscribe to the pipeline bus
 * (via the "sync-message" signal, which coexists with the host app's own bus
 * watch) and run the SAME parse + dispatch + operation logic the sample app
 * runs in its bus_call() -- only here it lives inside the plugin and operates on
 * the element's own ModelMuxBin. The host app therefore needs zero glue:
 * drop the element downstream of nvmultiurisrcbin and REST control just works.
 *
 * Threading: the messages are posted on the REST server thread. We deep-copy the
 * parsed info and marshal the actual operation onto the default main context
 * (g_idle_add), so the bin surgery runs on the application main loop -- exactly
 * where the sample app runs it. model/status is answered separately and directly
 * by the bin's downstream-query probe (no bus message involved).
 */

#include <string.h>
#include <stdlib.h>                    /* atoi */
#include "gstnvmodelmux.h"
#include "gst-nvdscommonconfig.h"      /* NvDsSensorInfo, NvDsModelInfo */
#include "gst-nvdscustommessage.h"     /* gst_nvmessage_is/parse_* */
#include "gstnvdsmeta.h"               /* gst_buffer_get_nvds_batch_meta */
#include "nvdsmeta.h"                  /* NvDsBatchMeta, NvDsFrameMeta */
#include "gst-nvevent.h"              /* GST_NVEVENT_PAD_ADDED/PAD_DELETED/STREAM_EOS */
#include "gst-nvdscustomevent.h"      /* GST_NVEVENT_MODEL_*/ /* + STREAM_ROUTE (in-band REST) */
#include <json-glib/json-glib.h>       /* v1 control payloads arrive as ONE JSON document */
#include <string.h>

GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat

/* A source_id absent from the batch for this many buffers is auto-detached. */
#define MM_AUTO_STALE 60

/* Per-source auto-attach state, driven by the source lifecycle events: a stream/add
 * (in-band GST_NVEVENT_PAD_ADDED, or the stream-add bus message) creates/attaches the
 * entry, a stream/remove (GST_NVEVENT_PAD_DELETED) removes it. The attach is asynchronous
 * (scheduled on the main loop), so until it completes the sink probe must DROP that
 * source's batched buffers -- otherwise the in-demux gets an un-routable source_id,
 * returns NOT_LINKED, and kills the upstream decoder. `attached` flips TRUE once the
 * main-loop attach has actually wired the stream. */
typedef struct
{
  gboolean attached;
  /* DETERMINISTIC reject (source_id >= modelmux batch-size): the attach can NEVER
   * succeed -- an upstream mux batching more sources than this element's batch-size
   * is a CONFIG mismatch. Retrying is futile; log the teaching error ONCE and stop
   * the backoff churn. (The buffer gate still drops batches containing the id --
   * forwarding it would NOT_LINK the in-demux and kill the upstream decoder; the
   * fix is config, and the log now says so explicitly.) */
  gboolean rejected;
  /* batch index last seen (for staleness detach) */
  guint64  last_seen;
  /* consecutive FAILED attach attempts (backoff shift) */
  guint    fails;
  /* batch index of the next re-attach retry (0 = none due: in flight or never failed).
   * A failed attach MUST be retried, else the entry gates (drops) every batch forever. */
  guint64  retry_at;
} ModelMuxSourceAttachState;

/* Cap the retry backoff at MM_AUTO_STALE << 4 batches (~32 s at 30 fps). */
#define MM_AUTO_FAIL_SHIFT_MAX 4

/* Per-stream model binding requested via stream/add. Carried in-band by the
 * GST_NVEVENT_STREAM_MODEL_BIND event (emitted by nvmultiurisrcbin, since the
 * muxer's PAD_ADDED carries only source_id) -> sensor_id + sensor_name + raw
 * metadata JSON. Stored in a SEPARATE hash (self->requested_binding_info) so it
 * never touches source_attach_state -- adding a source_attach_state entry would
 * suppress the attach (PAD_ADDED freshness + buffer-fallback both key off
 * source_attach_state membership). */
typedef struct
{
  gchar *sensor_id;
  gchar *sensor_name;
  gchar *sensor_metadata;
  /* TRUE once an ADD has used this request. A REMOVE clears only a CONSUMED one: a fast
   * source-id recycle can deliver the NEXT add's request before the old REMOVE, and wiping
   * an un-consumed request would lose the imminent add's sensor id + model metadata. */
  gboolean consumed;
} ModelMuxRequestedBindingInfo;

/* Free a stream binding structure. */
static void
requested_binding_info_free (gpointer p)
{
  ModelMuxRequestedBindingInfo *requested_binding_info = (ModelMuxRequestedBindingInfo *) p;
  if (!requested_binding_info)
    return;
  g_free (requested_binding_info->sensor_id);
  g_free (requested_binding_info->sensor_name);
  g_free (requested_binding_info->sensor_metadata);
  g_free (requested_binding_info);
}

/* --- thread-local capture of the FIRST MODELMUX_CONTROL_ERR message ---
 * Lets the synchronous control-admission query turn a handler's reject into a
 * REST reason WITHOUT instrumenting every reject site (stream/route has ~20).
 * Arm it on the calling (REST) thread, run the handler's admission pass, then
 * read the reason it logged. Keeps the FIRST message so a specific reject wins
 * over any generic "rejected whole" logged afterwards. Per-thread, so logging on
 * other threads is never captured or clobbered. */
static GPrivate mm_ctrl_err_capture = G_PRIVATE_INIT (g_free);   /* gchar*: armed if non-NULL */

static inline void
modelmux_control_err_capture (const gchar * fmt, ...)
{
  gchar *cur = (gchar *) g_private_get (&mm_ctrl_err_capture);
  va_list ap;
  if (!cur || *cur)                     /* not armed, or first message already kept */
    return;
  va_start (ap, fmt);
  g_private_replace (&mm_ctrl_err_capture, g_strdup_vprintf (fmt, ap));
  va_end (ap);
}

static inline void modelmux_control_err_capture_arm (void)
{ g_private_replace (&mm_ctrl_err_capture, g_strdup ("")); }   /* "" = armed, nothing yet */
static inline const gchar *modelmux_control_err_capture_get (void)
{ return (const gchar *) g_private_get (&mm_ctrl_err_capture); }
static inline void modelmux_control_err_capture_disarm (void)
{ g_private_replace (&mm_ctrl_err_capture, NULL); }

/* Element-scoped logging (guarded INFO; GST warning/error like any plugin).
 * ERR also feeds the admission-query capture above (no-op unless armed). */
#define MODELMUX_CONTROL_INFO(self, fmt, ...) GST_INFO_OBJECT (self, fmt, ##__VA_ARGS__)
#define MODELMUX_CONTROL_WARN(self, fmt, ...) GST_WARNING_OBJECT (self, fmt, ##__VA_ARGS__)
#define MODELMUX_CONTROL_ERR(self, fmt, ...)  do {          \
    GST_ERROR_OBJECT (self, fmt, ##__VA_ARGS__);            \
    modelmux_control_err_capture (fmt, ##__VA_ARGS__);      \
  } while (0)

/* ================================================================== */
/* Small helpers                                                       */
/* ================================================================== */

/* Resolve one role's model from a requested name. Stream-add MUST NOT load any model: a requested
 * model is used ONLY if already loaded -- READY/SERVING (attach now) or WARMING (attach passthrough
 * now + deferred-promote once warmed, see below); otherwise fall back to `def` (never auto-create
 * here). `def` may be NULL.
 * NOTE on lazy-warm asymmetry: a model requested via runtime stream/add METADATA that is NOT loaded
 * at all (status < 0) resolves to the DEFAULT (and only the resolved default is then lazy-warmed) --
 * so runtime metadata models must be pre-loaded via model/load. A pre-loaded metadata model that is
 * still WARMING is kept here (it WAS loaded) and rides the passthrough+promote path. Only CONFIG-TIME
 * [stream-model-*] bindings (which arrive here as `def`) are lazy-loaded on first stream-add. */
static const gchar *
modelmux_control_resolve_role (GstNvModelMux * self, gboolean shadow,
    const gchar * req_name, const gchar * def, const gchar * role_str)
{
  gint st;

  if (!req_name || !*req_name)
    return def;

  st = modelmux_bin_model_status (self->modelmux_bin, shadow, req_name);
  if (st == MODEL_WARMED || st == MODEL_SERVING)
    return req_name;

  /* Loaded (in limbo or a pool) but still WARMING: KEEP it as the resolved target instead of
   * discarding to `def`. The readiness gate in modelmux_control_handle_stream_add then attaches this stream as
   * PASSTHROUGH now -- tagged with pending_model so the routing table shows "<model> (passthru)" --
   * and enqueues a deferred promotion (modelmux_control_enqueue_promote) that switches it, AND every other
   * already-attached stream waiting on the SAME model, to inference once the engine WARMS (the
   * modelmux_control_pending_poll poller promotes each queued passthrough, staggered ~1 frame apart). Without
   * this, a model requested via stream/add metadata that was still warming got dropped to `def`
   * (plain passthrough, no pending tag, no promote), so ONLY streams added AFTER the warm ever
   * inferred and earlier ones stayed on passthrough forever. A still-warming SHADOW is likewise
   * kept so the gate's shadow-join / wait_shadow path attaches the A/B pair once it warms. */
  if (st == MODEL_WARMING)
    return req_name;

  /* NOT loaded (st < 0) or FAILED -> say EXACTLY what happens: fall back to the configured
   * default model (named), or -- if none -- passthrough (primary) / no-shadow (shadow). */
  {
    /* req_name / def are canonical "name@version" keys -> printed as-is. */
    const gchar *why = (st < 0) ? "is NOT loaded"
                     : (st == MODEL_FAILED) ? "FAILED to warm" : "is loaded but not READY";
    if (def)
      MODELMUX_CONTROL_WARN (self, "%s model '%s' %s -> falling back to the configured default %s model '%s'",
          role_str, req_name, why, role_str, def);
    else if (!shadow)
      MODELMUX_CONTROL_WARN (self, "primary model '%s' %s and no default primary is configured -> this stream "
          "will PASSTHROUGH (no inference) until a model is loaded/assigned", req_name, why);
    else
      MODELMUX_CONTROL_WARN (self, "shadow model '%s' %s and no default shadow is configured -> this stream "
          "runs PRIMARY ONLY (no shadow)", req_name, why);
  }
  return def;
}

/* ================================================================== */
/* v1 control payload decode (API_DESIGN.md)                           */
/*                                                                     */
/* Every REST control operation reaches the element as ONE verbatim    */
/* JSON document (the request's "value" object) -- on in-band custom   */
/* events and on bus messages alike. This block decodes those          */
/* documents ONCE at the boundary; all routing/lifecycle machinery     */
/* below is unchanged and payload-format agnostic.                     */
/* ================================================================== */

/* Upper bound on an untrusted control-plane JSON payload. */
#define MM_MAX_CTRL_JSON_LEN (256 * 1024)
/* Max per-model instance placements / per-GPU engine map entries. */
#define MM_MAX_MODEL_GPUS 16

static JsonParser *
modelmux_control_ctrl_json_load (GstNvModelMux * self, const gchar * json, const gchar * what)
{
  JsonParser *parser;
  JsonNode *root;
  GError *err = NULL;

  if (!json || !*json) {
    MODELMUX_CONTROL_ERR (self, "%s: empty control payload", what);
    return NULL;
  }
  /* bound untrusted input before handing it to json-glib (no built-in cap) */
  if (strlen (json) > MM_MAX_CTRL_JSON_LEN) {
    MODELMUX_CONTROL_ERR (self, "%s: control payload too large (%" G_GSIZE_FORMAT " bytes)",
        what, strlen (json));
    return NULL;
  }
  parser = json_parser_new ();
  if (!json_parser_load_from_data (parser, json, -1, &err)) {
    MODELMUX_CONTROL_ERR (self, "%s: malformed control JSON: %s", what,
        err ? err->message : "?");
    g_clear_error (&err);
    g_object_unref (parser);
    return NULL;
  }
  root = json_parser_get_root (parser);
  if (!root || !JSON_NODE_HOLDS_OBJECT (root)) {
    MODELMUX_CONTROL_ERR (self, "%s: control payload is not a JSON object", what);
    g_object_unref (parser);
    return NULL;
  }
  return parser;
}

/* Strict typed member reads: a present-but-wrong-type member reads as absent
 * (the REST parser already rejected it upstream with a teaching error; an
 * app-generated event must not g_critical here either). */
static gchar *
modelmux_control_ctrl_json_str (JsonObject * obj, const gchar * key)
{
  JsonNode *n;
  if (!obj || !json_object_has_member (obj, key))
    return NULL;
  n = json_object_get_member (obj, key);
  if (!n || !JSON_NODE_HOLDS_VALUE (n) ||
      json_node_get_value_type (n) != G_TYPE_STRING)
    return NULL;
  return g_strdup (json_node_get_string (n));
}

/* GPU ids are bounded 0..255 everywhere in this API (mirrors the REST
 * parser's GPU_ID_MAX): a huge 64-bit JSON value must never TRUNCATE into a
 * valid-looking device id at this element boundary either (events/messages
 * can be app-generated and bypass the REST gate). */
#define MODELMUX_CONTROL_GPU_ID_MAX 255

/* Read an OPTIONAL "gpu" member with the 0..MODELMUX_CONTROL_GPU_ID_MAX bound applied.
 * Absent / wrong-type reads as -1 ("no gpu given", the strict-typed-member
 * convention above); a PRESENT but out-of-range value reads as G_MAXINT -- an id no
 * device can ever match -- so the caller's admission (modelmux_control_route_gpu_ok /
 * the gpus[] check) REJECTS the route/payload instead of silently
 * truncating to a small valid id. */
static gint
modelmux_control_ctrl_json_gpu (JsonObject * obj, const gchar * key)
{
  JsonNode *n;
  gint64 g;
  if (!obj || !json_object_has_member (obj, key))
    return -1;
  n = json_object_get_member (obj, key);
  if (!n || !JSON_NODE_HOLDS_VALUE (n) ||
      json_node_get_value_type (n) != G_TYPE_INT64)
    return -1;
  g = json_node_get_int (n);
  if (g < 0 || g > MODELMUX_CONTROL_GPU_ID_MAX)
    return G_MAXINT;
  return (gint) g;
}

/* Read a {"name","version"[,"gpu"]} | null model-ref member. Returns TRUE when
 * @key is PRESENT; *is_null / *name / *version / *gpu tell which form it took.
 * *malformed (optional) reads TRUE when the member is PRESENT but neither null
 * nor an object (e.g. "model": "Car;2" or 123): name/version stay NULL, which
 * a caller would otherwise misread as "unchanged" -- callers must REJECT the
 * request instead of silently ignoring the stated (but undecodable) intent. */
static gboolean
modelmux_control_json_ref (JsonObject * obj, const gchar * key, gboolean * is_null,
    gchar ** name, gchar ** version, gint * gpu, gboolean * malformed)
{
  JsonNode *n;

  *is_null = FALSE;
  *name = *version = NULL;
  if (gpu)
    *gpu = -1;
  if (malformed)
    *malformed = FALSE;
  if (!obj || !json_object_has_member (obj, key))
    return FALSE;
  n = json_object_get_member (obj, key);
  if (!n || JSON_NODE_HOLDS_NULL (n)) {
    *is_null = TRUE;
    return TRUE;
  }
  if (JSON_NODE_HOLDS_OBJECT (n)) {
    JsonObject *ref = json_node_get_object (n);
    *name = modelmux_control_ctrl_json_str (ref, "name");
    *version = modelmux_control_ctrl_json_str (ref, "version");
    if (gpu)
      *gpu = modelmux_control_ctrl_json_gpu (ref, "gpu");   /* bounded 0..255; see helper */
  } else if (malformed) {
    *malformed = TRUE;                         /* present, not null, not object */
  }
  return TRUE;
}

/* Decoded model-plane payload (/api/v1/model/load|unload|update "value"). */
typedef struct
{
  gchar   *name;
  gchar   *version;
  gchar   *from_version;                 /* model/update CAS guard          */
  gchar   *config_file;
  gchar   *engine_file;                  /* scalar SEED / update engine     */
  struct { gint gpu; gchar *path; }
           engines[MM_MAX_MODEL_GPUS];   /* engine_files{gpu -> path} map   */
  guint    n_engines;
  gint     gpus[MM_MAX_MODEL_GPUS];      /* 'gpus'[]: load = desired placements,
                                          * unload = gpu subset to drain */
  guint    n_gpus;
  guint    batch_size;                   /* OPT per-version batch/stream cap
                                          * (0 = unset -> version registry ->
                                          * catalog -> per-model-batch-size) */
} ModelMuxModelPayload;

/* Synchronous outcome of a control op, carried back to the REST caller by the
 * control-admission query (model/unload today). A NULL result pointer means the
 * async event path (non-REST hosts): the op still runs, outcome logged only. */
typedef struct
{
  gint    http;        /* 202 accepted, 400 rejected                          */
  gchar  *err_code;    /* stable machine id (NULL on accept)                  */
  gchar  *reason;      /* human message                                       */
  gchar  *hint;        /* actionable next step (may be NULL)                  */
} ModelMuxControlResult;

static void
modelmux_control_result_set (ModelMuxControlResult * res, gint http, const gchar * err_code,
    const gchar * reason, const gchar * hint)
{
  if (!res)
    return;
  res->http = http;
  g_free (res->err_code);
  g_free (res->reason);
  g_free (res->hint);
  res->err_code = g_strdup (err_code);
  res->reason = g_strdup (reason);
  res->hint = g_strdup (hint);
}

static void
modelmux_control_result_clear (ModelMuxControlResult * res)
{
  if (!res)
    return;
  g_free (res->err_code);
  g_free (res->reason);
  g_free (res->hint);
  memset (res, 0, sizeof (*res));
}

/* Post a model-plane COMPLETION event on the pipeline bus, so an app can react
 * exactly like it does for `new stream added`. `src` is any element in the
 * pipeline (the message routes up to the app's bus). Called from the async
 * completion points -- an engine finished warming, a version was torn down, an
 * OTA committed, or a reroute settled. */
G_GNUC_INTERNAL void
modelmux_post_model_event (GstElement * src, const gchar * event,
    const gchar * name, const gchar * version, gint gpu, gboolean ok,
    const gchar * detail)
{
  NvDsModelEventInfo info;
  if (!src)
    return;
  info.event = event;
  info.name = name;
  info.version = version;
  info.gpu = gpu;
  info.ok = ok;
  info.detail = detail;
  gst_element_post_message (src,
      gst_nvmessage_new_model_event (GST_OBJECT (src), &info));
}

static void
modelmux_control_model_payload_clear (ModelMuxModelPayload * pl)
{
  guint i;
  g_free (pl->name);
  g_free (pl->version);
  g_free (pl->from_version);
  g_free (pl->config_file);
  g_free (pl->engine_file);
  for (i = 0; i < pl->n_engines; i++)
    g_free (pl->engines[i].path);
  memset (pl, 0, sizeof (*pl));
}

static gboolean
modelmux_control_decode_model_payload (GstNvModelMux * self, const gchar * what,
    const gchar * json, ModelMuxModelPayload * pl)
{
  JsonParser *parser;
  JsonObject *v;

  memset (pl, 0, sizeof (*pl));
  parser = modelmux_control_ctrl_json_load (self, json, what);
  if (!parser)
    return FALSE;
  v = json_node_get_object (json_parser_get_root (parser));

  pl->name = modelmux_control_ctrl_json_str (v, "name");
  pl->version = modelmux_control_ctrl_json_str (v, "version");
  pl->from_version = modelmux_control_ctrl_json_str (v, "from_version");
  pl->config_file = modelmux_control_ctrl_json_str (v, "config_file");
  pl->engine_file = modelmux_control_ctrl_json_str (v, "engine_file");
  /* an empty string means "not given" everywhere in this API -- normalize
   * ONCE so no handler ever feeds "" to a file test or key builder */
  {
    gchar **fields[] = { &pl->name, &pl->version, &pl->from_version,
      &pl->config_file, &pl->engine_file };
    guint fi;
    for (fi = 0; fi < G_N_ELEMENTS (fields); fi++)
      if (*fields[fi] && !**fields[fi]) {
        g_free (*fields[fi]);
        *fields[fi] = NULL;
      }
  }
  if (json_object_has_member (v, "batch_size")) {
    /* ATOMIC admission: a present-but-invalid batch_size rejects the WHOLE
     * payload (silently ignoring it would deploy a different capacity than the
     * caller stated while the request reads as accepted). */
    JsonNode *n = json_object_get_member (v, "batch_size");
    gint64 b = (n && JSON_NODE_HOLDS_VALUE (n) &&
        json_node_get_value_type (n) == G_TYPE_INT64) ? json_node_get_int (n) : -1;
    if (b < 1 || b > (gint64) MM_MAX_BATCH) {
      MODELMUX_CONTROL_ERR (self, "%s rejected: 'batch_size' must be an integer in "
          "1..%u (BATCH_SIZE_INVALID) -- it sizes this version's per-instance "
          "nvinfer/mux batch", what, (guint) MM_MAX_BATCH);
      goto payload_reject;
    }
    pl->batch_size = (guint) b;
  }
  /* engine_files{"<gpu>": "<path>"}: the per-device engine map. ATOMIC
   * admission: any present-but-invalid piece rejects the WHOLE payload --
   * silently dropping/truncating entries would deploy a smaller or different
   * set than the caller stated while the request reads as accepted. */
  if (json_object_has_member (v, "engine_files")) {
    JsonNode *n = json_object_get_member (v, "engine_files");
    JsonObject *ef;
    GList *members, *m;
    if (!n || !JSON_NODE_HOLDS_OBJECT (n)) {
      MODELMUX_CONTROL_ERR (self, "%s rejected: 'engine_files' must be an object of "
          "{\"<gpu>\": \"<path>\"} (FIELD_TYPE_INVALID)", what);
      goto payload_reject;
    }
    ef = json_node_get_object (n);
    members = json_object_get_members (ef);
    for (m = members; m; m = m->next) {
      const gchar *k = (const gchar *) m->data;
      gchar *path = modelmux_control_ctrl_json_str (ef, k);
      gchar *end = NULL;
      gint64 g = g_ascii_strtoll (k, &end, 10);
      gboolean key_ok = (end && !*end && g >= 0 && g <= MODELMUX_CONTROL_GPU_ID_MAX);
      if (!key_ok) {
        MODELMUX_CONTROL_ERR (self, "%s rejected: engine_files key '%s' is not a gpu id "
            "in 0..%d (FIELD_TYPE_INVALID)", what, k, MODELMUX_CONTROL_GPU_ID_MAX);
        g_free (path);
        g_list_free (members);
        goto payload_reject;
      }
      if (!path || !*path) {
        MODELMUX_CONTROL_ERR (self, "%s rejected: engine_files[\"%s\"] must be a non-empty "
            "string path (FIELD_TYPE_INVALID)", what, k);
        g_free (path);
        g_list_free (members);
        goto payload_reject;
      }
      if (pl->n_engines >= MM_MAX_MODEL_GPUS) {
        MODELMUX_CONTROL_ERR (self, "%s rejected: engine_files lists more than the "
            "supported %u devices (LIMIT_EXCEEDED)", what,
            (guint) MM_MAX_MODEL_GPUS);
        g_free (path);
        g_list_free (members);
        goto payload_reject;
      }
      pl->engines[pl->n_engines].gpu = (gint) g;
      pl->engines[pl->n_engines].path = path;
      pl->n_engines++;
    }
    g_list_free (members);
  }
  /* gpus:[N,...]: the explicit device set -- one instance per gpu id.
   *   load   -> the desired instance placement (one pinned instance per gpu).
   *   unload -> the subset of the version's per-gpu instances to drain
   *             (empty/absent -> the whole version).
   * Same ATOMIC admission contract as engine_files above: any bad element
   * rejects the WHOLE payload (never truncate/skip a device the caller named).
   * The REST gate (nvds_model_parse) already validated range/dedup; this is
   * defense-in-depth for any non-REST in-band event source. */
  if (json_object_has_member (v, "gpus")) {
    JsonNode *n = json_object_get_member (v, "gpus");
    JsonArray *a;
    guint i, len, j;
    if (!n || !JSON_NODE_HOLDS_ARRAY (n)) {
      MODELMUX_CONTROL_ERR (self, "%s rejected: 'gpus' must be an array of gpu ids "
          "(FIELD_TYPE_INVALID)", what);
      goto payload_reject;
    }
    a = json_node_get_array (n);
    len = json_array_get_length (a);
    for (i = 0; i < len; i++) {
      JsonNode *el = json_array_get_element (a, i);
      gint64 gv;
      gint g;
      if (!el || !JSON_NODE_HOLDS_VALUE (el) ||
          json_node_get_value_type (el) != G_TYPE_INT64) {
        MODELMUX_CONTROL_ERR (self, "%s rejected: gpus[%u] is not an integer gpu id "
            "(FIELD_TYPE_INVALID)", what, i);
        goto payload_reject;
      }
      gv = json_node_get_int (el);
      if (gv < 0 || gv > MODELMUX_CONTROL_GPU_ID_MAX) {
        MODELMUX_CONTROL_ERR (self, "%s rejected: gpus[%u]=%" G_GINT64_FORMAT
            " out of range 0..%d (FIELD_TYPE_INVALID)", what, i, gv,
            MODELMUX_CONTROL_GPU_ID_MAX);
        goto payload_reject;
      }
      g = (gint) gv;
      for (j = 0; j < pl->n_gpus; j++) {           /* dedup: one instance per device */
        if (pl->gpus[j] == g) {
          MODELMUX_CONTROL_ERR (self, "%s rejected: duplicate gpu %d in 'gpus' "
              "(FIELD_TYPE_INVALID)", what, g);
          goto payload_reject;
        }
      }
      if (pl->n_gpus >= MM_MAX_MODEL_GPUS) {
        MODELMUX_CONTROL_ERR (self, "%s rejected: 'gpus' lists more than the "
            "supported %u devices (LIMIT_EXCEEDED)", what,
            (guint) MM_MAX_MODEL_GPUS);
        goto payload_reject;
      }
      pl->gpus[pl->n_gpus++] = g;
    }
  }
  g_object_unref (parser);
  return TRUE;

payload_reject:
  g_object_unref (parser);
  modelmux_control_model_payload_clear (pl);
  return FALSE;
}

/* "0,1,2" list of the LIVE instance gpus of (name, ver) -- for teaching error
 * messages. Returns NULL when the model has no live shards. Caller g_free()s. */
static gchar *
modelmux_control_live_gpu_list (GstNvModelMux * self, const gchar * name, const gchar * ver)
{
  gint g[MM_MAX_MODEL_GPUS];
  gchar *key = model_key (name, ver);
  guint i, n = modelmux_bin_model_gpus (self->modelmux_bin, key, g,
      G_N_ELEMENTS (g));
  GString *s;
  g_free (key);
  if (!n)
    return NULL;
  s = g_string_new (NULL);
  for (i = 0; i < n; i++)
    g_string_append_printf (s, "%s%d", i ? "," : "", g[i]);
  return g_string_free (s, FALSE);
}

/* A gpu-addressed ref is valid if ANY live instance (shard) of the version runs on
 * that device (instance groups span GPUs); a model that is not live falls back to
 * the version registry's recorded placement. */
static gboolean
modelmux_control_ref_gpu_matches (GstNvModelMux * self, const gchar * name, const gchar * ver,
    gint want)
{
  gchar *key = model_key (name, ver);
  gboolean on = modelmux_bin_model_on_gpu (self->modelmux_bin, key, want);
  gboolean live = on || modelmux_bin_model_loaded (self->modelmux_bin, key);
  gint have;
  g_free (key);
  if (live)
    return on;
  have = modelmux_config_version_gpu (&self->config, name, ver);
  if (have < 0)
    have = 0;                   /* unrecorded placement = the default device */
  return have == want;
}

/* A gpu-addressed METADATA ref (stream/add) must match one of the version's
 * live instances (or its recorded placement when not live); a mismatch falls
 * back to the role default (warn, never a hard stream rejection -- the stream
 * itself is fine). */
static gboolean
modelmux_control_check_ref_gpu (GstNvModelMux * self, const gchar * name, const gchar * ver,
    gint want, guint source_id, const gchar * role)
{
  gchar *gpus;

  if (want < 0 || !name || !ver)
    return TRUE;
  if (want == G_MAXINT) {
    /* the decode's bound check flagged a PRESENT but out-of-range id (the
     * G_MAXINT sentinel; see modelmux_control_ctrl_json_gpu) -- print the teaching bound,
     * never the sentinel value itself */
    MODELMUX_CONTROL_WARN (self, "stream %u: %s '%s@%s' gpu id out of range (0..%d) "
        "(GPU_UNKNOWN) -- falling back to the default %s", source_id, role,
        name, ver, MODELMUX_CONTROL_GPU_ID_MAX, role);
    return FALSE;
  }
  if (modelmux_control_ref_gpu_matches (self, name, ver, want))
    return TRUE;
  gpus = modelmux_control_live_gpu_list (self, name, ver);
  MODELMUX_CONTROL_WARN (self, "stream %u: %s '%s@%s' requested on gpu %d but its instance(s) "
      "run on gpu(s) %s (GPU_UNKNOWN) -- falling back to the default %s",
      source_id, role, name, ver, want, gpus ? gpus : "0", role);
  g_free (gpus);
  return FALSE;
}

/* ================================================================== */
/* stream/add + stream/remove                                          */
/* ================================================================== */

/* fwd-decl: deferred-promotion / deferred-shadow-join enqueue (defined with the pending-reroute
 * machinery below). wait_shadow=TRUE makes the promote also gate on the shadow warming. */
static void modelmux_control_enqueue_promote (GstNvModelMux * self, guint source_id,
    const gchar * primary, const gchar * shadow, gboolean wait_shadow,
    gint p_gpu, gint s_gpu, gboolean p_via_default, gboolean s_via_default);
static void modelmux_control_enqueue_shadow_join (GstNvModelMux * self, guint source_id,
    const gchar * shadow, gint s_gpu, gboolean s_via_default);
/* fwd-decl: reroute streams off a demoted old default onto the new one (zero-drop, deferred
 * until the new default warms). Defined with the pending-reroute machinery below.
 * excl/n_excl/excl_all: source_ids already covered by an EXPLICIT SERVING-MODEL route
 * in the SAME stream/route request -- the sweep must skip them (an explicit route that
 * CHANGES THE SERVING MODEL always wins over the default sweep; shadow-only routes
 * remain sweepable). */
static void modelmux_control_switch_default_reroute (GstNvModelMux * self, gboolean shadow,
    const gchar * old_key, const gchar * new_key, gint gpu,
    const guint * excl, guint n_excl, gboolean excl_all);

/* LAZY warm: a per-sensor-bound (or default-resolved) model that is not loaded yet is kicked
 * into limbo NOW -- with its per-version artifact config -- so it starts warming; the readiness
 * gate then takes the existing passthrough+promote path. Only catalog-known names are warmed
 * (an unknown name stays NotLoaded -> genuine reject). Eager defaults are already loaded, so this
 * is a no-op for them. Returns the resulting status (-1 if it could not be warmed). */
static gint
modelmux_control_lazy_warm (GstNvModelMux * self, gboolean shadow, const gchar * key)
{
  gint st;
  gchar *name = NULL, *ver = NULL, *eff_cfg = NULL;
  gboolean dfail = FALSE;
  if (!key || !*key)
    return -1;
  st = modelmux_bin_model_status (self->modelmux_bin, shadow, key);
  if (st >= 0)
    return st;                            /* already loaded / warming / ready -> nothing to do */
  model_key_split (key, &name, &ver);
  if (!name || !modelmux_config_find_model (&self->config, name)) {
    g_free (name); g_free (ver);
    return st;                            /* not in the catalog -> cannot be lazily warmed */
  }
  eff_cfg = modelmux_config_resolve_version_cfg (&self->config, name, ver, &dfail);
  if (dfail) {
    /* the version's EXPLICIT engine/gpu override failed to derive: warming the
     * base config would run the wrong artifact/device -- genuine reject. */
    MODELMUX_CONTROL_ERR (self, "lazy-load %s model '%s' REJECTED: its per-version engine/gpu "
        "override failed to derive (DERIVE_FAILED; base config would run the "
        "wrong artifact/device)", shadow ? "shadow" : "primary", key);
    g_free (name); g_free (ver);
    return -1;
  }
  MODELMUX_CONTROL_INFO (self, "lazy-load %s model '%s' on stream-add%s -> warming into limbo",
      shadow ? "shadow" : "primary", key, eff_cfg ? " (per-version artifact)" : "");
  /* the version registry records the placement of the ORIGINAL load -- a lazy
   * warm must land on that same device (-1 when never gpu-addressed). */
  st = modelmux_bin_load_model (self->modelmux_bin, key, eff_cfg,
      modelmux_config_version_gpu (&self->config, name, ver),
      0 /* batch: version registry/catalog resolve inside */);
  g_free (eff_cfg); g_free (name); g_free (ver);
  return st;
}

/* Effective placement of a RESOLVED role key -- ONE rule for stream-add,
 * stream-bind and the deferred promote/join paths:
 *   1. an EXPLICIT gpu on the request's own ref always wins;
 *   2. otherwise, when the resolved model IS the GLOBAL runtime default, the
 *      persisted default pin (stream/route default{}.gpu) applies -- including
 *      when the request named the default explicitly WITHOUT a gpu (same model
 *      must not get different placement policies depending on whether the
 *      request spelled its name);
 *   3. anything else -- notably a PER-CAMERA bound model that merely resolved
 *      through the default slot -- gets NO constraint (the default pin was
 *      recorded for the default model, never for a camera binding).
 */
static gint
modelmux_control_eff_gpu (GstNvModelMux * self, gboolean shadow, const gchar * resolved,
    const gchar * req_key, gint req_gpu)
{
  /* the role's runtime default lives in its pool (safe to read here: every
   * default writer AND this reader run under api_control_lock). */
  const DefaultModelRef *dref =
      modelmux_bin_get_default (self->modelmux_bin, shadow);
  gint ret = MM_GPU_ANY;

  if (!resolved)
    return MM_GPU_ANY;
  if (req_key && g_strcmp0 (resolved, req_key) == 0 && req_gpu != MM_GPU_ANY)
    return req_gpu;                        /* explicit ref placement wins */
  if (dref && dref->key && dref->gpu != MM_GPU_ANY &&
      g_strcmp0 (resolved, dref->key) == 0)
    ret = dref->gpu;                       /* the GLOBAL default's pin */
  return ret;
}

/* Returns TRUE iff the stream was actually attached (so the auto-attach probe can
 * stop dropping that source's buffers). */
static gboolean
modelmux_control_handle_stream_add (GstNvModelMux * self, const NvDsSensorInfo * info)
{
  const gchar *camera_id = info->sensor_id;
  const gchar *primary, *shadow;
  const gchar *def_primary = NULL, *def_shadow = NULL, *def_pv = NULL, *def_sv = NULL;
  gchar *p_name = NULL, *s_name = NULL;
  gchar *p_ver = NULL, *s_ver = NULL;
  gint p_gpu = -1, s_gpu = -1;
  gint eff_pgpu = MM_GPU_ANY, eff_sgpu = MM_GPU_ANY;
  gchar *req_pkey = NULL, *req_skey = NULL, *def_pkey = NULL, *def_skey = NULL;
  gchar *defer_to = NULL;     /* owned copy of intended model KEY when deferring to passthrough */
  gint pst;
  gboolean attached = FALSE, shadow_cleared = FALSE;
  gboolean p_from_def = TRUE, s_from_def = TRUE;   /* resolver origin (default vs binding) */
  gboolean p_via = FALSE, s_via = FALSE;           /* final per-lane origin flags */

  modelmux_control_parse_model_metadata (self, info->sensor_metadata, &p_name, &p_ver,
      &p_gpu, &s_name, &s_ver, &s_gpu);

  /* "none" is the explicit shadow-clear token (same meaning as in model/update): this stream
   * opts OUT of the configured default shadow and runs PRIMARY ONLY. */
  shadow_cleared = (s_name && g_strcmp0 (s_name, "none") == 0);

  /* versions must be positive integers (Triton convention) -- coerce an invalid one to NULL
   * so it falls back to the default version (with the feedback warning below). */
  if (p_ver && !modelmux_version_is_int (p_ver)) {
    MODELMUX_CONTROL_WARN (self, "stream %u: primary_version '%s' is not a positive integer -- ignoring",
        info->source_id, p_ver);
    g_free (p_ver); p_ver = NULL;
  }
  if (s_ver && !modelmux_version_is_int (s_ver)) {
    MODELMUX_CONTROL_WARN (self, "stream %u: shadow_version '%s' is not a positive integer -- ignoring",
        info->source_id, s_ver);
    g_free (s_ver); s_ver = NULL;
  }

  /* per-sensor config binding (by camera_id) supplies this stream's DEFAULTS (name+version);
   * a stream/add metadata model still overrides. Unbound -> global defaults. */
  modelmux_config_resolve_models (&self->config,
      modelmux_bin_get_default (self->modelmux_bin, FALSE),
      modelmux_bin_get_default (self->modelmux_bin, TRUE),
      camera_id, &def_primary, &def_shadow, &def_pv, &def_sv,
      &p_from_def, &s_from_def);

  MODELMUX_CONTROL_INFO (self, "STREAM ADD source=%u camera='%s' (requested primary=%s shadow=%s)",
      info->source_id, camera_id ? camera_id : "?",
      p_name ? p_name : "(default)", s_name ? s_name : "(default)");

  /* compose canonical "name@version" keys. A requested model is addressed by (name, version):
   * STRICT -- a requested model WITHOUT a version is NOT resolvable and falls back to the role
   * default (which carries its OWN configured version). We do NOT silently assume version 1.
   * (The configured default below always pairs name+version.) */
  if (p_name && g_strcmp0 (p_name, "none") == 0) {
    /* 'none' is NOT valid for primary: a primary is required (it clears nothing). 'none' only
     * clears the SHADOW. Ignore it -> fall back to the default primary, or passthrough if no
     * default. (We do NOT auto-unload or touch any model here.) */
    MODELMUX_CONTROL_WARN (self, "stream %u: primary_model='none' is invalid -- primary is required ('none' "
        "only clears the shadow). Falling back to the default primary (passthrough if none).",
        info->source_id);
  } else if (p_name && p_ver) {
    req_pkey = model_key (p_name, p_ver);
  } else if (p_name) {
    MODELMUX_CONTROL_WARN (self, "stream %u: primary model '%s' requested WITHOUT a version -- a model is "
        "addressed by (name, version); falling back to the default primary model",
        info->source_id, p_name);
  }
  if (shadow_cleared) {
    MODELMUX_CONTROL_INFO (self, "stream %u: shadow_model='none' -> PRIMARY ONLY (opts out of the default "
        "shadow)", info->source_id);
  } else if (s_name && s_ver) {
    req_skey = model_key (s_name, s_ver);
  } else if (s_name) {
    MODELMUX_CONTROL_WARN (self, "stream %u: shadow model '%s' requested WITHOUT a version -- falling back "
        "to the default shadow model", info->source_id, s_name);
  }
  if (def_primary) def_pkey = model_key (def_primary, def_pv);
  if (def_shadow)  def_skey = model_key (def_shadow, def_sv);

  /* gpu-addressed metadata refs must match the version's recorded placement */
  if (req_pkey && !modelmux_control_check_ref_gpu (self, p_name, p_ver, p_gpu,
          info->source_id, "primary")) {
    g_free (req_pkey);
    req_pkey = NULL;
  }
  if (req_skey && !modelmux_control_check_ref_gpu (self, s_name, s_ver, s_gpu,
          info->source_id, "shadow")) {
    g_free (req_skey);
    req_skey = NULL;
  }

  primary = modelmux_control_resolve_role (self, FALSE, req_pkey, def_pkey, "primary");
  /* shadow_cleared ('none') forces PRIMARY ONLY -- skip the default-shadow fallback entirely. */
  shadow = shadow_cleared ? NULL
                          : modelmux_control_resolve_role (self, TRUE, req_skey, def_skey, "shadow");
  /* ORIGIN per lane (via_default): TRUE only when the role landed on the DEFAULT
   * fallback -- resolve_role picked def_pkey (POINTER identity: it returns one of
   * its arguments, so an explicit ref that VALUE-equals the default still reads
   * as pinned) AND that def came from the default ref, not a per-camera binding
   * (a binding is an explicit pin even when it names the default's model).
   * Pinned lanes survive future default switches; default lanes follow them. */
  p_via = (primary != NULL && primary == def_pkey && p_from_def);
  s_via = (shadow != NULL && shadow == def_skey && s_from_def);
  /* effective placement per role: explicit ref gpu when the ref won, the runtime
   * default's recorded gpu when the default won (captured NOW, before the gates
   * below may NULL the role for the deferred promote/join paths). */
  eff_pgpu = modelmux_control_eff_gpu (self, FALSE, primary, req_pkey, p_gpu);
  eff_sgpu = modelmux_control_eff_gpu (self, TRUE, shadow, req_skey, s_gpu);

  /* LAZY load: a resolved per-sensor-bound (or default-resolved) model that is not loaded yet is
   * warmed NOW (into limbo, with its per-version artifact). Eager defaults are already loaded so
   * this is a no-op for them; the gate below then drives the passthrough/promote/join handling. */
  if (primary) modelmux_control_lazy_warm (self, FALSE, primary);
  if (shadow)  modelmux_control_lazy_warm (self, TRUE,  shadow);

  if (!primary) {
    /* No resolvable primary: nothing requested + no default, OR a requested model that
     * does not exist + no default. The stream PASSES THROUGH without inference (primary=NULL
     * tells attach_stream to take the no-infer lane). It is NOT rejected -- frames still flow
     * downstream, just with no detections. A passthrough primary cannot host a shadow. */
    MODELMUX_CONTROL_INFO (self, "stream %u: no resolvable primary model (requested '%s') -> PASSTHROUGH "
        "(no inference)", info->source_id, p_name ? p_name : "-");
    shadow = NULL;
  } else {
    /* readiness gate for a RESOLVED primary:
     *   WARMED/SERVING -> attach to the model now (normal path); a still-warming shadow joins
     *                     later via a deferred shadow-add (primary infers immediately).
     *   WARMING        -> attach PASSTHROUGH now so frames flow, and enqueue a deferred
     *                     promotion that switches this stream to the model once it WARMS (also
     *                     waits on a warming shadow so a lazy A/B pair attaches together).
     *   FAILED / not-loaded -> genuine reject. */
    gint sst = shadow ? modelmux_bin_model_status (self->modelmux_bin, TRUE, shadow) : 1;
    pst = modelmux_bin_model_status (self->modelmux_bin, FALSE, primary);
    if (pst == MODEL_WARMED || pst == MODEL_SERVING) {
      /* primary ready */
      if (shadow && sst == MODEL_WARMING) {
        /* primary ready, shadow still warming (e.g. a lazy A/B shadow): attach primary NOW so
         * it infers immediately, and ADD the shadow via a deferred reroute once it WARMS
         * (zero-drop). */
        MODELMUX_CONTROL_INFO (self, "stream %u: primary '%s' ready, shadow '%s' still warming -> attach "
            "primary now, add shadow once WARMED", info->source_id, primary, shadow);
        modelmux_control_enqueue_shadow_join (self, info->source_id, shadow, eff_sgpu,
            s_via);
        shadow = NULL;                 /* attach primary-only now; reroute adds the shadow later */
      } else if (shadow && sst != MODEL_WARMED && sst != MODEL_SERVING) {
        MODELMUX_CONTROL_WARN (self, "shadow '%s' not ready (status=%s) -> attaching without shadow",
            shadow, sst < 0 ? "NotLoaded" : model_status_str (sst));
        shadow = NULL;
      }
      /* ready (+ ready/none shadow) -> fall through to the normal model attach */
    } else if (pst == MODEL_WARMING) {
      gboolean wait_shadow = (shadow && sst == MODEL_WARMING);
      MODELMUX_CONTROL_INFO (self, "stream %u: primary '%s' still warming -> PASSTHROUGH now, auto-promote "
          "to it once WARMED%s (frames flow meanwhile)", info->source_id, primary,
          wait_shadow ? " (with its warming shadow)" : "");
      modelmux_control_enqueue_promote (self, info->source_id, primary, shadow, wait_shadow,
          eff_pgpu, eff_sgpu, p_via, s_via);
      defer_to = g_strdup (primary);   /* remember the intended model KEY (routing-table tag) */
      primary = NULL;                  /* this attach goes through the no-infer passthrough lane */
      shadow = NULL;                   /* shadow (if any) is restored at promote time */
    } else {
      /* FAILED / not-loaded primary: attach the stream PASSTHROUGH instead of
       * rejecting it. A rejected stream would stay un-attached in source_attach_state, and
       * the sink probe gates (drops) WHOLE batches on any un-attached source --
       * one bad model would black out every healthy stream. Passthrough matches
       * the documented no-resolvable-primary behavior: frames flow, no inference. */
      MODELMUX_CONTROL_ERR (self, "stream %u: primary '%s' not usable (status=%s) -> attaching "
          "PASSTHROUGH (no inference)", info->source_id, primary,
          pst < 0 ? "NotLoaded" : model_status_str (pst));
      primary = NULL;
      shadow = NULL;
    }
  }

  /* BOTH identifiers go in: the display name for the overlay and model/status,
   * and the routable camera_id for the routing plane. They are the same string
   * only when the host supplied one identifier; when it supplied both and they
   * differ ("Cam-2" vs "cam-2"), collapsing them here is what forced the routing
   * read to be reassembled from the source manager's list. */
  if (modelmux_bin_attach_stream (self->modelmux_bin, info->source_id,
          info->sensor_name ? info->sensor_name : camera_id, camera_id,
          primary, shadow, eff_pgpu, eff_sgpu, p_via, s_via)) {
    attached = TRUE;
    /* deferred-promote passthrough: tag it with the intended model so the routing table
     * shows "<model> (passthru)" instead of "-" while that model is still warming.
     * eff_pgpu was captured before the warming gate NULLed `primary`. */
    if (defer_to)
      modelmux_bin_set_pending_model (self->modelmux_bin, info->source_id, defer_to,
          eff_pgpu, p_via);
  } else
    MODELMUX_CONTROL_ERR (self, "stream %u attach FAILED (camera '%s')",
        info->source_id, camera_id ? camera_id : "?");

  /* app-facing notice: announce what this stream/add actually BOUND to -- so the
   * app sees the routing outcome of every add, mirroring stream/route. Uses the
   * FINAL attach decision, not just the request, so all three cases are covered:
   *   - an explicit / per-camera-bound model  -> "model=<name@ver>"
   *   - the pool DEFAULT it fell back to       -> "model=<name@ver> (default)"
   *   - no resolvable/ready model              -> "model=passthrough" (no infer),
   *     or "(warming)" when it will auto-promote to a still-warming model.
   * Shadow shown only when one is actually attached (or "none" when opted out). */
  if (attached) {
    const gchar *sn = info->sensor_name ? info->sensor_name : camera_id;
    GString *b = g_string_new (NULL);
    g_string_append_printf (b, "stream=%s", sn ? sn : "?");
    if (primary)                                     /* attached to a live model  */
      g_string_append_printf (b, " model=%s%s", primary, p_via ? " (default)" : "");
    else if (defer_to)                               /* passthrough now, promotes later */
      g_string_append_printf (b, " model=%s (warming)", defer_to);
    else                                             /* pure passthrough, no inference  */
      g_string_append (b, " model=passthrough");
    if (shadow)
      g_string_append_printf (b, " shadow=%s%s", shadow, s_via ? " (default)" : "");
    else if (shadow_cleared)
      g_string_append (b, " shadow=none");
    modelmux_post_model_event (GST_ELEMENT (self), "stream-bound", b->str,
        "", -1, TRUE, NULL);
    g_string_free (b, TRUE);
  }

  g_free (p_name);
  g_free (s_name);
  g_free (p_ver);
  g_free (s_ver);
  g_free (req_pkey);
  g_free (req_skey);
  g_free (def_pkey);
  g_free (def_skey);
  g_free (defer_to);
  return attached;
}

static void modelmux_control_pending_drop_source (GstNvModelMux * self, guint source_id);

static void
modelmux_control_handle_stream_remove (GstNvModelMux * self,
    const NvDsSensorInfo * info)
{
  MODELMUX_CONTROL_INFO (self, "STREAM REMOVE source=%u camera='%s'", info->source_id,
      info->sensor_id ? info->sensor_id : "?");
  /* purge queued deferred ops FIRST: the id can be recycled the moment the pad frees */
  modelmux_control_pending_drop_source (self, info->source_id);
  modelmux_bin_detach_stream (self->modelmux_bin, info->source_id);
}

/* ================================================================== */
/* model/load + model/unload (type-less)                               */
/* ================================================================== */

static void
modelmux_control_handle_model_load (GstNvModelMux * self, const ModelMuxModelPayload * pl)
{
  const gchar *name = pl->name;
  const gchar *config = pl->config_file;
  const gchar *eng = pl->engine_file;
  const gchar *ver = pl->version;
  const ModelCatalogEntry *def;
  gchar *key = NULL, *derived = NULL;
  gchar *derived_multi[MM_MAX_MODEL_GPUS] = { NULL };
  const gchar *inst_cfgs[MM_MAX_MODEL_GPUS] = { NULL };
  const gchar *inst_engs[MM_MAX_MODEL_GPUS] = { NULL };
  const gchar *use_ver, *base_cfg, *eff_cfg;
  gboolean multi = (pl->n_gpus > 1);
  gint gpu = -1;
  ModelStatus st;
  guint i;

  MODELMUX_CONTROL_INFO (self, "MODEL LOAD name='%s' version=%s config=%s engine=%s gpus=%u",
      name ? name : "?", ver ? ver : "?", config ? config : "-",
      eng ? eng : (pl->n_engines ? "(per-gpu map)" : "(from config)"), pl->n_gpus);

  if (!name || !*name) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: 'name' required");
    return;
  }
  /* version is MANDATORY (no auto-increment): every load pins an explicit (name,version). */
  if (!ver || !*ver) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: 'version' required "
        "(provide an explicit positive integer: 1, 2, 3, ...; no auto-increment)");
    return;
  }
  /* version must be a positive integer (Triton model-repository convention). */
  if (!modelmux_version_is_int (ver)) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: version '%s' is not a positive integer "
        "(use 1, 2, 3, ... -- Triton version convention)", ver);
    return;
  }

  /* PLACEMENT: gpus[] names the desired per-GPU instance set. A single
   * placement (or none) takes the legacy single-instance path verbatim; two or
   * more materialize ONE instance (pinned shard) per listed GPU, each with its
   * own gpu-derived config + engine (API_DESIGN.md gpus[]). */
  if (pl->n_gpus)
    gpu = pl->gpus[0];
  for (i = 1; i < pl->n_gpus; i++) {
    guint j;
    for (j = 0; j < i; j++)
      if (pl->gpus[j] == pl->gpus[i]) {
        MODELMUX_CONTROL_ERR (self, "model load '%s@%s' rejected: gpus[] lists gpu %d "
            "twice (one instance per GPU today)", name, ver, pl->gpus[i]);
        return;
      }
  }

  if (!multi) {
    /* ENGINE for the placed device: the scalar engine_file, or the engine_files{}
     * entry matching the placement. EXACT coverage: a mapped load with no engine
     * for its device is an error, never a silent auto-build (TRT engines are
     * device-built; a foreign-device engine would rebuild or fail at warm). */
    if (!eng && pl->n_engines) {
      gint want = gpu >= 0 ? gpu : 0;    /* no placement = the default device 0 */
      for (i = 0; i < pl->n_engines; i++)
        if (pl->engines[i].gpu == want)
          eng = pl->engines[i].path;
      if (!eng) {
        MODELMUX_CONTROL_ERR (self, "model load '%s@%s' rejected: engine_files{} has no engine "
            "for gpu %d (GPU_ENGINE_MISSING)", name, ver, want);
        return;
      }
      for (i = 0; i < pl->n_engines; i++)
        if (pl->engines[i].gpu != want)
          MODELMUX_CONTROL_WARN (self, "model load '%s@%s': engine for gpu %d unused (single "
              "instance placed on gpu %d)", name, ver, pl->engines[i].gpu, want);
    }
  } else {
    /* ENGINES for a multi-GPU group: engine_files{} is the EXACT per-device
     * assignment; the scalar engine_file is a SEED offered to every instance
     * (each deserializes it and auto-builds for its own device on mismatch,
     * loudly, at warm time). EXACT coverage: a mapped load must name an engine
     * for every listed device, never a silent auto-build. */
    for (i = 0; i < pl->n_gpus; i++) {
      guint j;
      inst_engs[i] = eng;               /* scalar seed (NULL = config's engine) */
      for (j = 0; j < pl->n_engines; j++)
        if (pl->engines[j].gpu == pl->gpus[i])
          inst_engs[i] = pl->engines[j].path;
      if (!inst_engs[i] && pl->n_engines) {
        MODELMUX_CONTROL_ERR (self, "model load '%s@%s' rejected: engine_files{} has no "
            "engine for gpu %d (GPU_ENGINE_MISSING)", name, ver, pl->gpus[i]);
        return;
      }
    }
    for (i = 0; i < pl->n_engines; i++) {
      guint j;
      gboolean used = FALSE;
      for (j = 0; j < pl->n_gpus && !used; j++)
        used = (pl->engines[i].gpu == pl->gpus[j]);
      if (!used)
        MODELMUX_CONTROL_WARN (self, "model load '%s@%s': engine for gpu %d unused (that "
            "device is not in gpus[])", name, ver, pl->engines[i].gpu);
    }
  }

  /* artifact gate: named config/engine MUST exist before we touch anything */
  if (config && !g_file_test (config, G_FILE_TEST_IS_REGULAR)) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: config file not found: '%s'", config);
    return;
  }
  if (eng && !g_file_test (eng, G_FILE_TEST_IS_REGULAR)) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: engine file not found: '%s'", eng);
    return;
  }
  for (i = 0; multi && i < pl->n_gpus; i++)
    if (inst_engs[i] && !g_file_test (inst_engs[i], G_FILE_TEST_IS_REGULAR)) {
      MODELMUX_CONTROL_ERR (self, "model load rejected: engine file for gpu %d not found: "
          "'%s'", pl->gpus[i], inst_engs[i]);
      return;
    }

  def = modelmux_config_find_model (&self->config, name);   /* catalog entry (by bare name) */

  /* a brand-new model name DEFINES a network -> config-file REQUIRED. A known catalog name
   * may load a new version reusing its base config (config optional). */
  if (!def && !config) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: '%s' is not in the catalog -- provide "
        "config_file for a first load", name);
    return;
  }

  /* an unconfirmed model/update FREES its old key at trigger time; a load of
   * this name now could occupy the key a failed confirm must rename BACK to
   * (the rollback refuses an occupied destination -> inconsistent identity).
   * Refuse the load until the in-flight update resolves. */
  if (modelmux_bin_update_in_flight (self->modelmux_bin, name)) {
    MODELMUX_CONTROL_ERR (self, "model load '%s@%s' rejected: a model/update on '%s' is "
        "awaiting nvinfer's confirm (UPDATE_IN_FLIGHT) -- retry once it "
        "commits or rolls back", name, ver, name);
    return;
  }

  use_ver = ver;                          /* mandatory + integer-validated above */
  key = model_key (name, use_ver);

  /* CLAMP the requested batch to the element's cap AT ADMISSION: the registry and
   * every later re-warm must record the capacity that actually DEPLOYS -- an
   * unclamped record made a later model/update repeating the same accepted value
   * fail UPDATE_BATCH_MISMATCH, and the engine preflight warn against a phantom
   * capacity. (The bin's resolver still clamps as a backstop.) */
  if (pl->batch_size > self->config.batch_size) {
    MODELMUX_CONTROL_WARN (self, "model '%s': batch_size %u exceeds the element's "
        "batch-size %u -- CLAMPED to %u (raise 'batch-size' to honour it)",
        key, pl->batch_size, self->config.batch_size, self->config.batch_size);
    ((ModelMuxModelPayload *) pl)->batch_size = self->config.batch_size;
  }

  /* BATCH IMMUTABILITY -- checked BEFORE the generic already-loaded reject so the
   * operator gets the SPECIFIC teaching error: a loaded (name, version)'s batch is
   * part of its identity (mux + nvinfer property + slot tables are sized at
   * construction) and cannot change in place. */
  if (pl->batch_size) {
    guint live_b = modelmux_bin_model_batch (self->modelmux_bin, key);
    if (live_b && live_b != pl->batch_size) {
      MODELMUX_CONTROL_ERR (self, "model load rejected: '%s' is LIVE with batch %u; "
          "batch_size %u cannot apply in place (BATCH_IMMUTABLE) -- model/unload "
          "first, or load it as a new version", key, live_b, pl->batch_size);
      goto done;
    }
  }

  /* multi-version coexist: this EXACT (name,version) must not already be loaded. A new version
   * of an existing name is a NEW bin (an in-place swap is model/update's job). */
  if (modelmux_bin_model_loaded (self->modelmux_bin, key)) {
    MODELMUX_CONTROL_ERR (self, "model load rejected: '%s' already loaded -- use a different version "
        "(MODEL_ALREADY_LOADED)%s", key, pl->n_gpus ?
        "; changing a LIVE version's gpus[] set (scale up/down) is not "
        "supported yet -- unload the version when idle, or load the new "
        "placement as a new version" : "");
    goto done;
  }

  /* register a NEW catalog name so later versions reuse this base config. */
  if (!def && config) {
    modelmux_config_register_model (&self->config, name, config, NULL, NULL, 0);
    def = modelmux_config_find_model (&self->config, name);
    if (!def) {
      /* registration REFUSED (reserved '@'/';' in the name, or registry full):
       * bail HERE -- loading anyway would warm a bin the catalog knows nothing
       * about (invisible to lazy-warm / version resolution / unload guards). */
      MODELMUX_CONTROL_ERR (self, "model load rejected: could not register '%s' in the "
          "catalog (invalid name or registry full)", name);
      goto done;
    }
  }

  /* explicit engines never apply to nvinferserver -- Triton's model repository
   * owns the checkpoints. REJECT (never accept-and-ignore): an acknowledged
   * engine that isn't used would silently serve the wrong checkpoint. */
  if (def && def->type == MODEL_INFERSERVER && (eng || pl->n_engines)) {
    MODELMUX_CONTROL_ERR (self, "model load '%s' rejected: engine_file/engine_files do not "
        "apply to nvinferserver (ENGINE_UNSUPPORTED_BACKEND) -- add the "
        "checkpoint to the Triton model repository and load by config", key);
    goto done;
  }

  /* effective config for THIS version: a custom config wins over the catalog base; the
   * engine and/or gpu placement are swapped into a derived copy (both backends). */
  base_cfg = config ? config : (def ? def->config_file : NULL);
  eff_cfg = config;                          /* NULL => load_model uses the catalog base */

  /* a multi-GPU instance group is nvinfer-only: nvinferserver instance scaling
   * is delegated to Triton's instance_group (edit the model repository's
   * config.pbtxt and reload in Triton -- the plugin never scales Triton). */
  if (multi && base_cfg &&
      modelmux_detect_model_type (base_cfg) == MODEL_INFERSERVER) {
    MODELMUX_CONTROL_ERR (self, "model load '%s@%s' rejected: gpus[] with multiple GPUs "
        "is nvinfer-only -- scale an nvinferserver model via the Triton "
        "repository's instance_group (TRITON_INSTANCE_GROUP_MISMATCH)",
        name, use_ver);
    goto done;
  }

  if (!multi) {
    if ((eng || gpu >= 0) && base_cfg) {
      derived = modelmux_config_derive (base_cfg, eng, gpu, name, use_ver);
      /* an EXPLICIT engine/gpu that failed to derive must REJECT the load --
       * falling back to the base config would run the wrong artifact/device
       * while status and routing report the requested one (DERIVE_FAILED). */
      if (!derived) {
        MODELMUX_CONTROL_ERR (self, "model load '%s' rejected: could not derive '%s' with "
            "engine=%s gpu=%d (DERIVE_FAILED) -- the base config would run the "
            "wrong artifact/device; see the config-derive error above", key,
            base_cfg, eng ? eng : "(base)", gpu);
        goto done;
      }
      eff_cfg = derived;
    }
  } else if (base_cfg) {
    /* one gpu-derived config PER instance. The base instance keeps the plain
     * version token (same derived filename as a single-GPU load); siblings get
     * a ".g<gpu>" suffix so their derived configs never collide on one path. */
    for (i = 0; i < pl->n_gpus; i++) {
      gchar *vtag = (i == 0) ? g_strdup (use_ver)
          : g_strdup_printf ("%s.g%d", use_ver, pl->gpus[i]);
      derived_multi[i] = modelmux_config_derive (base_cfg, inst_engs[i], pl->gpus[i],
          name, vtag);
      g_free (vtag);
      if (!derived_multi[i]) {
        /* every gpus[] entry carries an EXPLICIT gpu -- a shard whose
         * placement failed to derive would silently warm on the wrong device */
        MODELMUX_CONTROL_ERR (self, "model load '%s' rejected: could not derive instance "
            "#%u config from '%s' for gpu %d (DERIVE_FAILED)", key, i,
            base_cfg, pl->gpus[i]);
        goto done;
      }
      inst_cfgs[i] = derived_multi[i];
    }
  }

  /* warm into limbo: roles/defaults/placement of STREAMS are the stream plane's
   * job (stream/route); model/load never designates a default. */
  /* engine/batch PRE-FLIGHT (best-effort, filename convention "_b<N>_"): a prebuilt
   * engine below the effective batch makes nvinfer reject it and rebuild (minutes)
   * or FAIL outright -- say so BEFORE the warm, with the cause. */
  {
    guint eff_b = pl->batch_size;
    const gchar *peng = multi ? inst_engs[0] : eng;
    if (!eff_b)
      eff_b = modelmux_config_version_batch (&self->config, name, use_ver);
    if (!eff_b && def)
      eff_b = def->max_streams;
    if (eff_b && peng) {
      GMatchInfo *mi = NULL;
      GRegex *re = g_regex_new ("_b([0-9]+)[._]", 0, 0, NULL);
      if (re && g_regex_match (re, peng, 0, &mi)) {
        gchar *bs = g_match_info_fetch (mi, 1);
        guint eng_b = bs ? (guint) g_ascii_strtoull (bs, NULL, 10) : 0;
        if (eng_b && eng_b < eff_b)
          MODELMUX_CONTROL_WARN (self, "model '%s': engine '%s' looks built for batch %u "
              "but the effective batch is %u -- nvinfer will REJECT it and rebuild from "
              "source (or FAIL if it can't); provide a _b%u_ engine or lower the batch",
              key, peng, eng_b, eff_b, eff_b);
        g_free (bs);
      }
      if (mi) g_match_info_free (mi);
      if (re) g_regex_unref (re);
    }
  }
  st = multi ?
      modelmux_bin_load_model_instances (self->modelmux_bin, key, inst_cfgs,
          pl->gpus, pl->n_gpus, pl->batch_size) :
      modelmux_bin_load_model (self->modelmux_bin, key, eff_cfg, gpu, pl->batch_size);

  /* record this version's artifact + placement so lazy warms, in-place updates
   * and gpu-addressed routing resolve it later (a live group's FULL gpu set is
   * answered by the bin query; the registry records the base placement).
   * Registered only AFTER an accepted load (both the single- and multi-instance
   * paths return through `st`): recording a load that failed outright would
   * leave a PHANTOM registry entry -- later gpu checks and lazy warms would
   * resolve a version that never deployed. */
  if (st != MODEL_FAILED)
    modelmux_config_register_version (&self->config, name, use_ver,
        multi ? inst_engs[0] : eng, config, gpu, pl->batch_size);
  if (st == MODEL_FAILED)
    /* deterministic creation/warm rejection (P2P_UNAVAILABLE, XFER_UNAVAILABLE,
     * missing config/factory -- specifics logged by the bin): the version was
     * NOT registered and the load must read as rejected, never accepted. */
    MODELMUX_CONTROL_ERR (self, "model '%s' load REJECTED (MODEL_LOAD_FAILED) -- see the "
        "bin errors above for the reason", key);
  else if (st == MODEL_WARMED || st == MODEL_SERVING)
    MODELMUX_CONTROL_INFO (self, "model '%s' loaded (already READY, status=%s)", key,
        model_status_str (st));
  else if (multi) {
    gchar *gl = modelmux_control_live_gpu_list (self, name, use_ver);
    MODELMUX_CONTROL_INFO (self, "model '%s' load accepted -> warming %u instance(s) on "
        "gpu(s) [%s]", key, pl->n_gpus, gl ? gl : "?");
    g_free (gl);
  } else
    MODELMUX_CONTROL_INFO (self, "model '%s' load accepted -> warming (engine=%s gpu=%d)",
        key, eng ? eng : "from config", gpu >= 0 ? gpu : 0);

done:
  g_free (key);
  g_free (derived);
  for (i = 0; i < pl->n_gpus; i++)
    g_free (derived_multi[i]);
}

static void
modelmux_control_handle_model_unload (GstNvModelMux * self,
    const ModelMuxModelPayload * pl, ModelMuxControlResult * res)
{
  const gchar *name = pl->name;
  const gchar *ver = pl->version;
  gchar *key = NULL, *msg = NULL;
  if (ver && !*ver) ver = NULL;

  MODELMUX_CONTROL_INFO (self, "MODEL UNLOAD name='%s' version=%s gpus=%u",
      name ? name : "?", ver ? ver : "(all versions)", pl->n_gpus);
  if (!name || !*name) {
    MODELMUX_CONTROL_ERR (self, "model unload rejected: model_name required");
    modelmux_control_result_set (res, 400, "NAME_INVALID", "model_name required",
        "set \"name\" to the model to unload");
    return;
  }
  /* never unload a configured default name -- it backs the routing fallback (any version) */
  {
  const DefaultModelRef *udp = modelmux_bin_get_default (self->modelmux_bin, FALSE);
  const DefaultModelRef *uds = modelmux_bin_get_default (self->modelmux_bin, TRUE);
  if ((udp && udp->name && g_strcmp0 (name, udp->name) == 0) ||
      (uds && uds->name && g_strcmp0 (name, uds->name) == 0)) {
    msg = g_strdup_printf ("'%s' is a configured default (backs the routing fallback)", name);
    MODELMUX_CONTROL_ERR (self, "model unload rejected: %s", msg);
    modelmux_control_result_set (res, 400, "MODEL_IS_DEFAULT", msg,
        "reassign the default via stream/route before unloading");
    g_free (msg);
    return;
  }
  }

  if (pl->n_gpus > 0) {
    /* GPU-SCOPED unload: drain only THIS version's instance(s) on the listed
     * gpus; the version's other per-gpu instances keep serving. Needs an exact
     * version (which version's per-gpu instance to drain). Each outcome gets a
     * distinct message so the caller knows exactly why it did/didn't happen. */
    gint sgpu = -1;
    guint sstr = 0, gi;
    gchar *scsv = NULL;
    GString *gs;
    ModelMuxUnloadGpuResult r;
    if (!ver) {
      MODELMUX_CONTROL_ERR (self, "model unload rejected: 'gpus' requires an explicit "
          "'version' (which version's per-gpu instance to drain)");
      modelmux_control_result_set (res, 400, "GPUS_INVALID",
          "'gpus' requires an explicit 'version' (which version's per-gpu instance to drain)",
          "add \"version\", or omit \"gpus\" to unload the whole name");
      return;
    }
    key = model_key (name, ver);
    gs = g_string_new (NULL);
    for (gi = 0; gi < pl->n_gpus; gi++)
      g_string_append_printf (gs, "%s%d", gi ? "," : "", pl->gpus[gi]);
    r = modelmux_bin_unload_model_gpus (self->modelmux_bin, key, pl->gpus,
        pl->n_gpus, &sgpu, &sstr, &scsv);
    switch (r) {
      case MM_UNLOAD_GPU_REMOVED: {
        gchar *det = g_strdup_printf ("gpu(s) [%s] removed (still serving on other gpu(s))",
            gs->str);
        msg = g_strdup_printf ("unloaded instance(s) of '%s' on gpu(s) [%s] "
            "(other instances kept serving)", key, gs->str);
        MODELMUX_CONTROL_INFO (self, "%s", msg);
        modelmux_control_result_set (res, 202, NULL, msg, NULL);
        /* app-facing notice: this teardown path frees bins directly (no bin.c post),
         * so announce the partial unload here (from the always-live element). */
        modelmux_post_model_event (GST_ELEMENT (self), "model-unloaded", key, "",
            -1, TRUE, det);
        g_free (det);
        break;
      }
      case MM_UNLOAD_GPU_ALL: {
        gchar *det = g_strdup_printf ("gpu(s) [%s] covered all instances", gs->str);
        msg = g_strdup_printf ("gpu(s) [%s] covered every instance of '%s' "
            "-> whole version unloaded", gs->str, key);
        MODELMUX_CONTROL_INFO (self, "%s", msg);
        modelmux_control_result_set (res, 202, NULL, msg, NULL);
        modelmux_post_model_event (GST_ELEMENT (self), "model-unloaded", key, "",
            -1, TRUE, det);
        g_free (det);
        break;
      }
      case MM_UNLOAD_GPU_SERVING:
        msg = g_strdup_printf ("'%s' instance on gpu %d is still inferring on "
            "%u stream(s) [%s]", key, sgpu, sstr, (scsv && *scsv) ? scsv : "?");
        MODELMUX_CONTROL_ERR (self, "model unload refused: %s", msg);
        modelmux_control_result_set (res, 400, "MODEL_SERVING", msg,
            "route/drain those streams via stream/route before unloading that gpu");
        break;
      case MM_UNLOAD_GPU_NO_MATCH:
        msg = g_strdup_printf ("'%s' has no instance on gpu(s) [%s]", key, gs->str);
        MODELMUX_CONTROL_ERR (self, "model unload: %s -- nothing to unload there", msg);
        modelmux_control_result_set (res, 400, "MODEL_NOT_LOADED", msg,
            "see GET model/status for the version's live gpus");
        break;
      case MM_UNLOAD_GPU_ABSENT:
      default:
        msg = g_strdup_printf ("'%s' is not loaded", key);
        MODELMUX_CONTROL_ERR (self, "model unload: %s", msg);
        modelmux_control_result_set (res, 400, "MODEL_NOT_LOADED", msg,
            "check the name/version, or GET model/status");
        break;
    }
    g_free (scsv);
    g_free (msg);
    g_string_free (gs, TRUE);
    g_free (key);
    return;
  }

  if (ver) {
    /* exact (name, version) whole-instance unload. Report a PRECISE reason
     * (serving vs not-loaded, with the offending streams) before attempting it. */
    guint sv = 0;
    gchar *sc = NULL;
    key = model_key (name, ver);
    if (!modelmux_bin_key_serving (self->modelmux_bin, key, &sv, &sc)) {
      msg = g_strdup_printf ("'%s' is not loaded", key);
      MODELMUX_CONTROL_ERR (self, "model unload: %s", msg);
      modelmux_control_result_set (res, 400, "MODEL_NOT_LOADED", msg,
          "check the name/version, or GET model/status");
    } else if (sv > 0) {
      msg = g_strdup_printf ("'%s' is serving %u stream(s) [%s]", key, sv,
          (sc && *sc) ? sc : "?");
      MODELMUX_CONTROL_ERR (self, "model unload refused: %s", msg);
      modelmux_control_result_set (res, 400, "MODEL_SERVING", msg,
          "route its streams elsewhere via stream/route, then unload");
    } else if (modelmux_bin_unload_model (self->modelmux_bin, key)) {
      msg = g_strdup_printf ("'%s' unloaded", key);
      MODELMUX_CONTROL_INFO (self, "model %s", msg);
      modelmux_control_result_set (res, 202, NULL, msg, NULL);
    } else {
      /* idle at the check, then raced mid-load: teardown re-checks and refused */
      msg = g_strdup_printf ("'%s' could not be unloaded (mid-load); retry shortly", key);
      MODELMUX_CONTROL_ERR (self, "model unload: %s", msg);
      modelmux_control_result_set (res, 400, "MODEL_UNLOAD_FAILED", msg,
          "retry after the model finishes warming");
    }
    g_free (sc);
    g_free (msg);
    g_free (key);
  } else {
    /* no version -> unload all NON-serving versions of the name (serving ones kept).
     * `present` distinguishes "every version is busy" (MODEL_SERVING) from
     * "the name isn't loaded at all" (MODEL_NOT_LOADED). */
    guint present = 0;
    guint n = modelmux_bin_unload_versions (self->modelmux_bin, name, &present);
    if (n) {
      msg = g_strdup_printf ("'%s': unloaded %u non-serving version(s)", name, n);
      MODELMUX_CONTROL_INFO (self, "model %s", msg);
      modelmux_control_result_set (res, 202, NULL, msg, NULL);
    } else if (present > 0) {
      gchar *sc = modelmux_bin_name_serving_streams (self->modelmux_bin, name);
      msg = g_strdup_printf ("'%s': all %u loaded version(s) still serving stream(s) "
          "[%s] (nothing unloaded)", name, present, (sc && *sc) ? sc : "?");
      MODELMUX_CONTROL_ERR (self, "model unload refused: %s", msg);
      modelmux_control_result_set (res, 400, "MODEL_SERVING", msg,
          "route their streams elsewhere via stream/route, then unload");
      g_free (sc);
    } else {
      msg = g_strdup_printf ("'%s' is not loaded", name);
      MODELMUX_CONTROL_ERR (self, "model unload: %s", msg);
      modelmux_control_result_set (res, 400, "MODEL_NOT_LOADED", msg,
          "check the name, or GET model/status");
    }
    g_free (msg);
  }
}

/* ================================================================== */
/* model/update (scoped A/B reroute / promote / swap)                  */
/* ================================================================== */

/* A reroute waiting on a still-warming target bin; applied (zero-drop) once the
 * warm target(s) reach READY. */
typedef struct
{
  guint   *src;
  guint    n;
  gchar   *eff_primary;
  gchar   *eff_shadow;
  gboolean clear_shadow;
  gchar   *warm_primary;
  gchar   *warm_shadow;
  gint     p_gpu;              /* requested instance placement per role, carried through
                                * the deferred apply/promote (MM_GPU_ANY = any) */
  gint     s_gpu;
  gboolean p_via_default;      /* origin of each role's target: default fallback vs
                                * explicit pin (rides the deferred op so the commit
                                * records the same origin an immediate apply would). */
  gboolean s_via_default;
  gboolean promote;            /* TRUE: this stream is on a no-infer PASSTHROUGH while its
                                * primary warms -> promote it to the model (not a reroute)
                                * once warm_primary reaches WARMED. */
  gboolean emit_routed;        /* TRUE: this entry came from an explicit stream/route POST
                                * whose completion event was withheld at accept time, so the
                                * poller owes a `streams-routed` when it lands. FALSE for
                                * shadow joins, metadata rebinds and default sweeps -- none
                                * of those announced a route before, and firing one for them
                                * would invent an event the app never used to see. */
} ModelMuxPendingReroute;

static void
modelmux_control_pending_free (gpointer data)
{
  ModelMuxPendingReroute *deferred_reroute = (ModelMuxPendingReroute *) data;
  if (!deferred_reroute)
    return;
  g_free (deferred_reroute->src);
  g_free (deferred_reroute->eff_primary);
  g_free (deferred_reroute->eff_shadow);
  g_free (deferred_reroute->warm_primary);
  g_free (deferred_reroute->warm_shadow);
  g_free (deferred_reroute);
}

/* Republish the deferred-route queue into the bin's status mirror, so an ACCEPTED
 * route that is still waiting on a warming target is visible to model/status (and
 * through it to GET /api/v1/stream/route). Until it applies the stream keeps
 * serving its old model, so without this the 202 is the only evidence the request
 * ever existed -- a reconciler polling status cannot tell an in-flight route from
 * one that was silently dropped.
 *
 * WHOLESALE rebuild, not an incremental mirror: pending_reroutes is appended from
 * five places and pruned from four more (apply, abandon, supersede, source
 * removal), and a per-entry mirror would have to hook every one of them correctly
 * forever. A full rebuild cannot drift out of sync -- the worst a missed call does
 * is leave the view one poll tick stale.
 *
 * Deferred PROMOTES are skipped: a passthrough stream awaiting one already reports
 * its target through prim.pending_model, and mirroring it here would report the
 * same intent twice. Caller holds api_control_lock. */
static void
modelmux_control_publish_pending (GstNvModelMux * self)
{
  GPtrArray *notes = NULL;
  GList *l;

  if (!self->modelmux_bin)
    return;

  for (l = self->pending_reroutes; l; l = l->next) {
    ModelMuxPendingReroute *deferred_reroute = (ModelMuxPendingReroute *) l->data;
    ModelMuxPendingRouteNote *note;

    if (!deferred_reroute || deferred_reroute->promote)
      continue;
    /* nothing to report: no target on either lane AND no clear pending. The
     * clear_shadow term matters -- dropping the A/B IS an intent, and testing
     * only the two target pointers would silently skip a route that changes the
     * shadow to "none". (Such a route cannot reach the queue today: deferral
     * needs a warming target, and a pure clear has none. Kept in the condition
     * so this stays correct if that ever changes.) */
    if (!deferred_reroute->eff_primary && !deferred_reroute->eff_shadow &&
        !deferred_reroute->clear_shadow)
      continue;
    if (!notes)
      notes = g_ptr_array_new_with_free_func (modelmux_pending_route_note_free);

    note = g_new0 (ModelMuxPendingRouteNote, 1);
    if (deferred_reroute->n) {
      note->src = g_memdup2 (deferred_reroute->src,
          deferred_reroute->n * sizeof (guint));
      note->n = deferred_reroute->n;
    }
    note->primary = g_strdup (deferred_reroute->eff_primary);
    /* carry the shadow lane's THREE states through, not two: an absent shadow
     * means "unchanged", and a cleared one has to say so explicitly or the read
     * cannot tell "reroute the primary" from "reroute it and drop the A/B". */
    note->clear_shadow = deferred_reroute->clear_shadow;
    note->shadow = deferred_reroute->clear_shadow
        ? NULL : g_strdup (deferred_reroute->eff_shadow);
    note->p_gpu = deferred_reroute->p_gpu;
    note->s_gpu = deferred_reroute->s_gpu;
    note->p_via_default = deferred_reroute->p_via_default;
    note->s_via_default = deferred_reroute->s_via_default;
    g_ptr_array_add (notes, note);
  }
  /* NULL when nothing is queued -- CLEARS the mirror rather than leaving an empty
   * array behind, so "no pending routes" and "never had any" read the same. */
  modelmux_bin_set_pending_routes (self->modelmux_bin, notes);
}

/* Withdraw `src`/`n` (n==0 => ALL streams) from every EARLIER queued deferred
 * reroute, so at most one queued entry ever covers a given stream.
 *
 * WITHOUT this the queue is append-only and the outcome is decided by WARM ORDER,
 * not by intent: route S to warming B, then to warming C; if C warms first it
 * applies, then B warms and overwrites it -- the stream lands on the older
 * target. The status view reports the NEWEST entry as authoritative, so the read
 * would also promise C and deliver B. enqueue_promote already guards its own
 * queue this way for exactly this reason; the reroute path did not.
 *
 * Scope arithmetic, in the only three shapes that can occur:
 *   - new scope is ALL      -> the new entry covers everything the old ones did;
 *                              drop them whole.
 *   - old scope is explicit -> remove the withdrawn ids; an entry that empties
 *                              has nothing left to do and is dropped.
 *   - old scope is ALL and the new one is not -> "all EXCEPT these" has no
 *                              representation here, so the old entry is
 *                              MATERIALISED to the live fleet minus the new
 *                              scope. It stops covering streams attached after
 *                              this point, which is the honest reading anyway:
 *                              a route-all issued at T meant the fleet at T.
 *
 * PROMOTES are left alone -- they are queued per stream and superseded by
 * enqueue_promote itself, and a promote and a reroute never target the same
 * stream (a stream is either passthrough or lane-attached, not both).
 * Caller holds api_control_lock. */
static void
modelmux_control_pending_supersede (GstNvModelMux * self, const guint * src,
    guint n, gboolean new_primary, gboolean new_shadow, gboolean only_default)
{
  GList *l = self->pending_reroutes, *next, *spawned = NULL, *sp;

  while (l) {
    ModelMuxPendingReroute *old = (ModelMuxPendingReroute *) l->data;
    guint *affected = NULL;
    guint n_affected = 0;
    gboolean affected_all = FALSE, drop = FALSE;
    gboolean keep_p, keep_s;
    gboolean may_take_p, may_take_s;
    guint w = 0, i, j;
    next = l->next;

    /* only_default: the caller is a DEFAULT SWEEP, which may only reclaim lanes
     * that are still FOLLOWING the default. An explicitly pinned pending route is
     * an operator decision and outranks the sweep -- the same invariant the
     * immediate path already enforces ("an explicit route that CHANGES THE SERVING
     * MODEL always wins over the default sweep"). Without this the sweep would
     * silently cancel a pinned route that had already been accepted with 202.
     * A promote carries no via_default of its own on the shadow lane, so it is
     * judged by its primary's origin, which is the lane a promote acts on. */
    may_take_p = new_primary && (!only_default || old->p_via_default);
    may_take_s = new_shadow && (!only_default || old->s_via_default);
    if (only_default && !may_take_p && !may_take_s) {
      l = next;                 /* nothing here this sweep is allowed to claim */
      continue;
    }

    /* PROMOTES are in scope too. A passthrough stream awaiting promotion to B can
     * receive a newer route to C: the promote fires first (it only waits on B), the
     * stream attaches to B, and the reroute that follows does not stick -- the
     * stream ends on the model the operator already replaced. Only a route that
     * claims the PRIMARY supersedes it, since a promote is a primary-lane
     * operation; a shadow-only route leaves it untouched.
     *
     * A promote is never SPLIT -- there is no surviving lane to carry, and
     * spawning a plain reroute from one would lose the promote semantics. */
    if (old->promote && !may_take_p) {
      l = next;
      continue;
    }

    /* ---- 1. which of this entry's streams does the new route cover? ---- */
    if (n == 0) {
      affected_all = (old->n == 0);
      if (old->n) {
        affected = g_memdup2 (old->src, old->n * sizeof (guint));
        n_affected = old->n;
      }
      drop = TRUE;                              /* new scope ALL subsumes it */
    } else if (old->n == 0) {
      /* all-scope old entry: materialise to the live fleet, split off the ids
       * the new route claims. */
      guint *live = NULL;
      guint n_live = modelmux_bin_active_stream_ids (self->modelmux_bin, &live);
      affected = g_new0 (guint, n ? n : 1);
      for (i = 0; i < n_live; i++) {
        gboolean taken = FALSE;
        for (j = 0; j < n && !taken; j++)
          if (live[i] == src[j])
            taken = TRUE;
        if (taken)
          affected[n_affected++] = live[i];
        else
          live[w++] = live[i];
      }
      if (w == 0) {
        drop = TRUE;
        g_free (live);
      } else {
        g_free (old->src);
        old->src = live;
        old->n = w;
      }
    } else {
      affected = g_new0 (guint, old->n);
      for (i = 0; i < old->n; i++) {
        gboolean taken = FALSE;
        for (j = 0; j < n && !taken; j++)
          if (old->src[i] == src[j])
            taken = TRUE;
        if (taken)
          affected[n_affected++] = old->src[i];
        else
          old->src[w++] = old->src[i];
      }
      old->n = w;
      drop = (w == 0);
    }

    if (!n_affected && !affected_all) {         /* no overlap -- leave it alone */
      g_free (affected);
      l = next;
      continue;
    }

    /* ---- 2. LANE arithmetic. A route is partial: an omitted `model` means
     * "leave the primary alone", an omitted `shadow` likewise. So a newer
     * shadow-only route must NOT cancel an older pending PRIMARY change, and
     * vice versa. Only the lanes the new route actually sets are superseded;
     * an older intent on the untouched lane survives for those streams. ---- */
    keep_p = !old->promote && (old->eff_primary != NULL) && !may_take_p;
    keep_s = !old->promote && (old->eff_shadow != NULL || old->clear_shadow)
        && !may_take_s;

    if (keep_p || keep_s) {
      /* SPLIT: the withdrawn streams keep the surviving lane in an entry of
       * their own, so the rest of the old entry's scope is untouched. */
      ModelMuxPendingReroute *keep = g_new0 (ModelMuxPendingReroute, 1);
      if (!affected_all && n_affected) {
        keep->src = g_memdup2 (affected, n_affected * sizeof (guint));
        keep->n = n_affected;
      }
      if (keep_p) {
        keep->eff_primary = g_strdup (old->eff_primary);
        keep->warm_primary = g_strdup (old->warm_primary);
        keep->p_gpu = old->p_gpu;
        keep->p_via_default = old->p_via_default;
      } else {
        keep->p_gpu = MM_GPU_ANY;
      }
      if (keep_s) {
        keep->eff_shadow = g_strdup (old->eff_shadow);
        keep->warm_shadow = g_strdup (old->warm_shadow);
        keep->clear_shadow = old->clear_shadow;
        keep->s_gpu = old->s_gpu;
        keep->s_via_default = old->s_via_default;
      } else {
        keep->s_gpu = MM_GPU_ANY;
      }
      keep->emit_routed = old->emit_routed;
      spawned = g_list_append (spawned, keep);
      MODELMUX_CONTROL_INFO (self, "queued route to '%s': %s lane SUPERSEDED for %u "
          "stream(s); its %s lane survives", old->eff_primary ? old->eff_primary : "-",
          new_primary ? "primary" : "shadow", n_affected ? n_affected : 0,
          keep_p ? "primary" : "shadow");
    } else if (n_affected || affected_all) {
      MODELMUX_CONTROL_INFO (self, "queued route to '%s' SUPERSEDED for %u stream(s) "
          "by a fresher one", old->eff_primary ? old->eff_primary : "-",
          affected_all ? 0 : n_affected);
    }
    g_free (affected);

    if (drop) {
      /* This entry is being destroyed, and with it any completion it still owed.
       * A route that answered 202 must reach SOME terminal state or the caller
       * waits forever -- an accepted operation that simply stops existing is the
       * one outcome an async API may not have. Announce the cancellation before
       * freeing, using the same event name so one subscription sees both
       * outcomes; ok=FALSE is what distinguishes them. Entries that never owed a
       * completion (default sweeps, shadow joins, metadata rebinds) stay silent,
       * exactly as before.
       *
       * NOT when the entry was SPLIT: the spawned `keep` inherited emit_routed,
       * so the route lives on for its surviving lane and will announce itself
       * when that lands. Emitting here too would report one accepted route as
       * both cancelled and completed. */
      if (old->emit_routed && !(keep_p || keep_s)) {
        gchar *summary = g_strdup_printf ("streams=%s primary=%s shadow=%s",
            old->n ? "scoped" : "all",
            old->eff_primary ? old->eff_primary : "-",
            old->clear_shadow ? "none"
            : (old->eff_shadow ? old->eff_shadow : "-"));
        modelmux_post_model_event (GST_ELEMENT (self), "streams-routed", summary,
            "", -1, FALSE /* superseded, not applied */,
            "superseded by a newer route");
        g_free (summary);
      }
      self->pending_reroutes = g_list_delete_link (self->pending_reroutes, l);
      modelmux_control_pending_free (old);
    }
    l = next;
  }

  /* appended AFTER the walk so the loop never re-examines what it just split off */
  for (sp = spawned; sp; sp = sp->next)
    self->pending_reroutes = g_list_append (self->pending_reroutes, sp->data);
  g_list_free (spawned);

  /* The queue just changed -- entries were dropped, narrowed or split -- so the
   * mirror the status probe reads is now stale. Republish HERE rather than at
   * each call site: the two routes that supersede WITHOUT deferring (the
   * in-place path and the immediate-apply path) both return before reaching a
   * publish, so a GET issued between the POST and the next 200 ms poll tick
   * would report the new serving model alongside a pending target that has
   * already been deleted and will never apply. Publishing from the one function
   * that mutates the queue also means a future caller cannot forget it.
   * (modelmux_control_pending_drop_source republishes for the same reason.)
   * Cheap on the common path: with no queued routes this rebuilds nothing. */
  modelmux_control_publish_pending (self);
}

/* Drop a departing source_id from every queued deferred op (promote / reroute /
 * shadow-join / sweep). WITHOUT this, nvmultiurisrcbin's id recycling makes a stale
 * entry fire against the NEXT incarnation of the id: a re-added stream gets promoted/
 * rerouted to the PREVIOUS stream's target. Entries that empty are freed; a scope-all
 * deferred route (n==0) is left alone (it re-resolves live streams at apply time).
 * Caller holds api_control_lock (the poller and every enqueue run under it too). */
static void
modelmux_control_pending_drop_source (GstNvModelMux * self, guint source_id)
{
  GList *l = self->pending_reroutes, *next;
  while (l) {
    ModelMuxPendingReroute *deferred_reroute = (ModelMuxPendingReroute *) l->data;
    guint w = 0, i;
    next = l->next;
    for (i = 0; i < deferred_reroute->n; i++)
      if (deferred_reroute->src[i] != source_id)
        deferred_reroute->src[w++] = deferred_reroute->src[i];
    if (w != deferred_reroute->n) {
      MODELMUX_CONTROL_INFO (self, "stream %u removed -> dropped from a queued deferred "
          "%s (%u stream(s) remain on it)", source_id,
          deferred_reroute->promote ? "promote" : "reroute", w);
      deferred_reroute->n = w;
      if (w == 0) {
        self->pending_reroutes = g_list_delete_link (self->pending_reroutes, l);
        modelmux_control_pending_free (deferred_reroute);
      }
    }
    l = next;
  }
  /* scopes were narrowed in place (and empty entries freed) -- the mirror holds
   * COPIES of those scopes, so it has to be rebuilt or it keeps naming a stream
   * that is gone. */
  modelmux_control_publish_pending (self);
}


static void modelmux_control_enqueue_promote (GstNvModelMux * self, guint source_id,
    const gchar * primary, const gchar * shadow, gboolean wait_shadow,
    gint p_gpu, gint s_gpu, gboolean p_via_default, gboolean s_via_default);

static guint
modelmux_control_apply_scoped_reroute (GstNvModelMux * self, const guint * src,
    guint n, const gchar * eff_primary, const gchar * eff_shadow,
    gboolean clear_shadow, gint p_gpu, gint s_gpu,
    gboolean p_via_default, gboolean s_via_default, guint * out_promoted)
{
  guint launched, promoted = 0;

  /* PART 1 -- streams in the requested scope that are CURRENTLY in PASSTHROUGH.
   * The reroute request targets these streams too, but a passthrough stream was attached
   * display-only (no model lane), so Part 2's in-place routing update below cannot switch it.
   * We handle them HERE -- but NOTE: the two calls in the loop DO NOT route/attach anything
   * themselves; they only RECORD INTENT so the reroute happens LATER, asynchronously:
   *   set_pending_model -> MARK the stream's target (a flag on its wiring entry; status/routing
   *                        then shows "<model> (passthru)" while it waits),
   *   enqueue_promote   -> QUEUE the switch on the pending list.
   * The actual promote (move off passthrough -> attach to inference) is done by the pending
   * poller once the model is WARMED. Meanwhile frames keep flowing on passthrough (zero-drop).
   * The `if` is only a PRECONDITION (the reroute must name a primary to promote onto); the
   * actual "is this stream passthrough?" test is per-stream inside modelmux_bin_passthru_streams
   * (stream_entry->prim.passthru). If no in-scope stream is passthrough, npt==0 and the loop does nothing. */
  if (eff_primary && *eff_primary) {      /* precondition: the reroute names a primary target */
    guint *pt = NULL;
    /* collect the source_ids in scope that are currently in passthrough (per prim.passthru) */
    guint npt = modelmux_bin_passthru_streams (self->modelmux_bin, n ? src : NULL, n,
        &pt);
    guint i;
    for (i = 0; i < npt; i++) {
      /* MARK only (no attach here): record this stream's target so status/routing shows
       * "<model> (passthru)" while it waits for promotion. */
      modelmux_bin_set_pending_model (self->modelmux_bin, pt[i], eff_primary, p_gpu,
          p_via_default);
      /* QUEUE only (no attach here): add it to the pending list -- the poller performs the
       * actual promote (attach to inference) once the model is WARMED. */
      modelmux_control_enqueue_promote (self, pt[i], eff_primary,
          clear_shadow ? NULL : eff_shadow, FALSE, p_gpu, s_gpu,
          p_via_default, s_via_default);
    }
    g_free (pt);                          /* the id array is copied into each promote entry */
    promoted = npt;                       /* remember how many were queued (for revision/log below) */
    if (npt)
      MODELMUX_CONTROL_INFO (self, "stream/route: %u passthrough stream(s) queued for "
          "promotion to '%s' (inference attaches once the model is ready)",
          npt, eff_primary);
  }

  /* PART 2 -- streams in scope that ALREADY have a model lane: switch them IN PLACE now.
   * This is the actual live relink (returns how many streams it changed). Passthrough
   * streams handled in Part 1 are skipped here -- they have no lane to swap. */
  launched = modelmux_bin_update_routing_scoped (self->modelmux_bin,
      n ? src : NULL, n, eff_primary, clear_shadow ? "" : eff_shadow,
      p_gpu, s_gpu, p_via_default, s_via_default);
  /* routing_revision moves only when routing CONTENT actually changed --
   * a no-op apply must not invalidate other controllers' if_revision.
   * (Queued promotions count: the declarative intent is recorded and the
   * routing table already shows "<model> (passthru)".) */
  if (launched || promoted)
    modelmux_bin_bump_routing_revision (self->modelmux_bin);
  if (launched)
    MODELMUX_CONTROL_INFO (self, "stream/route applied to %u stream(s) (model=%s shadow=%s)",
        launched, eff_primary ? eff_primary : "(unchanged)",
        clear_shadow ? "(cleared)" : (eff_shadow ? eff_shadow : "(unchanged)"));
  else if (!promoted)
    MODELMUX_CONTROL_ERR (self, "stream/route: no streams updated (none in scope / not active)");
  /* `promoted` counts streams that were only QUEUED for a later promote -- the
   * reroute has NOT attached inference to them yet. The fused return value hides
   * that, so a caller that needs to know whether the switch is genuinely complete
   * (the pending poller, before it announces one) reads it here. */
  if (out_promoted)
    *out_promoted = promoted;
  return launched + promoted;
}

/* 1 = READY/SERVING, 0 = warming, -1 = FAILED or GONE. NULL target => ready. */
static gint
modelmux_control_role_ready (GstNvModelMux * self, gboolean shadow, const gchar * name)
{
  gint st;
  if (!name)
    return 1;
  st = modelmux_bin_model_status (self->modelmux_bin, shadow, name);
  if (st == MODEL_WARMED || st == MODEL_SERVING)
    return 1;
  if (st == MODEL_FAILED || st < 0)
    return -1;   /* st < 0 = no live bin (unloaded/reclaimed mid-wait): treating it
                  * as "warming" would wedge the pending entry FOREVER -- the poller
                  * would spin for the pipeline's lifetime and the stream would stay
                  * parked on passthrough/old model. Abandon like FAILED. */
  return 0;
}

static gboolean modelmux_control_pending_poll (gpointer data);

/* Arm (or re-arm) the pending-reroute poller. The source holds an ELEMENT ref so a
 * dispatch that is blocked on api_control_lock while the element tears down can never run
 * against (or clear a mutex of) a finalized object; the ref drops with the source. */
static void
modelmux_control_arm_pending_poll (GstNvModelMux * self, guint interval_ms)
{
  self->pending_poll_id = g_timeout_add_full (G_PRIORITY_DEFAULT, interval_ms,
      modelmux_control_pending_poll, gst_object_ref (self), (GDestroyNotify) gst_object_unref);
}

/* Low-frequency poller: apply each pending reroute once its target(s) are READY;
 * drop entries whose target FAILED. Stops itself when the queue drains. */
static gboolean
modelmux_control_pending_poll (gpointer data)
{
  GstNvModelMux *self = GST_NVMODELMUX (data);
  GList *l, *next;
  gboolean did_promote = FALSE;
  /* streams a deferred reroute could only QUEUE for a later promote (they were
   * in passthrough, so there was no lane to switch). Non-zero means the reroute
   * is not finished, whatever its return total says. */
  guint still_queued = 0;
  /* total streams the deferred apply touched (launched + promoted). Zero means
   * the scope went empty while the target warmed -- nothing switched. */
  guint applied = 0;

  /* api_control_lock: serializes against teardown (self->modelmux_bin free) and the other control
   * entry points; the !mm check is only meaningful under it. */
  g_mutex_lock (&self->api_control_lock);
  if (!self->modelmux_bin) {                       /* torn down -> stop */
    self->pending_poll_id = 0;
    g_mutex_unlock (&self->api_control_lock);
    return G_SOURCE_REMOVE;
  }

  for (l = self->pending_reroutes; l; l = next) {
    ModelMuxPendingReroute *deferred_reroute = (ModelMuxPendingReroute *) l->data;
    gint rp = modelmux_control_role_ready (self, FALSE, deferred_reroute->warm_primary);
    gint rs = modelmux_control_role_ready (self, TRUE, deferred_reroute->warm_shadow);
    next = l->next;

    if (rp < 0 || rs < 0) {
      MODELMUX_CONTROL_ERR (self, "model/update abandoned: target failed to warm "
          "(primary=%s shadow=%s)", deferred_reroute->warm_primary ? deferred_reroute->warm_primary : "-",
          deferred_reroute->warm_shadow ? deferred_reroute->warm_shadow : "-");
      if (deferred_reroute->promote && deferred_reroute->n) {
        guint pi;
        /* the promote is dead: clear the lane's pending tag too, or status shows
         * "<model> (passthru)" forever and streams_on_model keeps matching the
         * stream to the dead key (a later default switch would mis-sweep it). */
        for (pi = 0; pi < deferred_reroute->n; pi++)
          modelmux_bin_clear_pending_model (self->modelmux_bin, deferred_reroute->src[pi]);
      }
      self->pending_reroutes = g_list_delete_link (self->pending_reroutes, l);
      modelmux_control_pending_free (deferred_reroute);
      continue;
    }
    if (rp == 0 || rs == 0)
      continue;                          /* still warming */

    if (deferred_reroute->promote) {
      /* deferred ATTACH: the stream is flowing on a no-infer passthrough while its primary
       * warmed; now WARMED -> promote that stream's passthrough to the model (inference on). */
      MODELMUX_CONTROL_INFO (self, "deferred promote: model warmed -> switching stream %u from passthrough "
          "to '%s' (inference starts)", deferred_reroute->n ? deferred_reroute->src[0] : 0, deferred_reroute->eff_primary);
      if (deferred_reroute->n)
        modelmux_bin_promote_passthru (self->modelmux_bin, deferred_reroute->src[0], deferred_reroute->eff_primary,
            deferred_reroute->eff_shadow, deferred_reroute->p_gpu, deferred_reroute->s_gpu, deferred_reroute->p_via_default, deferred_reroute->s_via_default);
      /* This promote carries the completion owed by the stream/route POST that
       * spawned it (set when that route's deferred apply could only queue
       * promotes). It is the LAST of that route's promotes, so the switch is now
       * complete for the whole scope -- announce it here, once. Ordinary promotes
       * never set the flag and stay silent, exactly as before. */
      if (deferred_reroute->emit_routed) {
        gchar *summary = g_strdup_printf ("streams=%s primary=%s shadow=%s",
            deferred_reroute->n ? "scoped" : "all",
            deferred_reroute->eff_primary ? deferred_reroute->eff_primary : "-",
            deferred_reroute->clear_shadow ? "none"
            : (deferred_reroute->eff_shadow ? deferred_reroute->eff_shadow : "-"));
        modelmux_post_model_event (GST_ELEMENT (self), "streams-routed", summary,
            "", -1, TRUE, NULL);
        g_free (summary);
      }
      self->pending_reroutes = g_list_delete_link (self->pending_reroutes, l);
      modelmux_control_pending_free (deferred_reroute);
      /* STAGGER promotions: apply ONE per tick, then re-arm FAST (~one frame @30fps) below.
       * Streams waiting on the SAME model warm together, but the legacy per-model nvstreammux
       * loses a sink pad added in the narrow window as it transitions 0->active. Spacing the
       * promotions ~one frame apart reproduces the working "sequential add" timing so every
       * sink integrates -- without the long full-poll-interval wait. This only affects the
       * one-time warm->serve switchover; steady-state runs at full frame rate. */
      did_promote = TRUE;
      break;
    }
    MODELMUX_CONTROL_INFO (self, "MODEL UPDATE (deferred apply: warmed target now READY)");
    still_queued = 0;
    applied = modelmux_control_apply_scoped_reroute (self, deferred_reroute->src, deferred_reroute->n, deferred_reroute->eff_primary,
        deferred_reroute->eff_shadow, deferred_reroute->clear_shadow, deferred_reroute->p_gpu, deferred_reroute->s_gpu,
        deferred_reroute->p_via_default, deferred_reroute->s_via_default,
        &still_queued);
    /* Announce ONLY if the switch is genuinely done. The POST handler stayed
     * quiet for this route (it answered 202), so this is the only streams-routed
     * the app sees for it -- and it must not fire early.
     *
     * still_queued > 0 means some in-scope stream was in PASSTHROUGH, so the
     * reroute merely tagged it and enqueued a promote; inference attaches on a
     * later tick. Announcing here would repeat the original bug one layer down:
     * a completion event for a switch that has not happened. The promote path
     * owns the notice for those streams.
     *
     * applied == 0 is the OTHER way this route can finish without switching
     * anything: apply_scoped_reroute returns launched+promoted, and it returns
     * zero when the scope no longer matches a live, active stream (every
     * targeted stream was removed while the model warmed). It logs that case as
     * an error, but the entry is still consumed here -- so without this term the
     * route would announce a completed switch that moved no stream at all. */
    if (deferred_reroute->emit_routed && still_queued > 0) {
      /* The apply could not switch these streams outright -- they were in
       * PASSTHROUGH, so it tagged them and queued promotes instead. This entry is
       * about to be deleted, and with it the only record that a completion is
       * owed, which left an accepted route that answered 202 emitting NOTHING,
       * ever.
       *
       * Hand the debt to the LAST promote just queued. Promotes are applied in
       * list order, one per poll tick (see the STAGGER note above), and
       * apply_scoped_reroute appends them at the tail, so the final list element
       * is the last of this route's promotes to land -- exactly one event, fired
       * when the whole scope has actually switched. Emitting from each promote
       * instead would produce N byte-identical events (the summary is scope-level
       * and names no stream), which a consumer could not tell apart.
       *
       * A promote that is later abandoned because its target failed to warm fires
       * nothing, which is correct: the route never landed. */
      GList *tail = g_list_last (self->pending_reroutes);
      ModelMuxPendingReroute *last = tail ? (ModelMuxPendingReroute *) tail->data : NULL;
      if (last && last->promote)
        last->emit_routed = TRUE;
    }
    if (deferred_reroute->emit_routed && still_queued == 0 && applied > 0) {
      /* same shape route_summary produces (no outer brackets -- the app adds
       * them when it renders the event), so the deferred notice and the
       * immediate one read identically in the log. */
      gchar *summary = g_strdup_printf ("streams=%s primary=%s shadow=%s",
          deferred_reroute->n ? "scoped" : "all",
          deferred_reroute->eff_primary ? deferred_reroute->eff_primary : "-",
          deferred_reroute->clear_shadow ? "none"
          : (deferred_reroute->eff_shadow ? deferred_reroute->eff_shadow : "-"));
      modelmux_post_model_event (GST_ELEMENT (self), "streams-routed", summary,
          "", -1, TRUE, NULL);
      g_free (summary);
    }
    self->pending_reroutes = g_list_delete_link (self->pending_reroutes, l);
    modelmux_control_pending_free (deferred_reroute);
  }

  /* the loop above applied, abandoned and/or promoted entries -- republish once
   * for the whole tick rather than per entry. Runs on EVERY tick, so a mirror
   * left stale by any path self-heals within one poll interval. */
  modelmux_control_publish_pending (self);

  if (!self->pending_reroutes) {
    self->pending_poll_id = 0;
    g_mutex_unlock (&self->api_control_lock);
    return G_SOURCE_REMOVE;
  }
  if (did_promote) {
    /* a promote was just applied and more remain -> re-arm at ~one frame (33ms @30fps)
     * instead of the slow default tick, so the warm->serve switchover for N streams takes
     * ~N*33ms (not N*200ms). Replace the current timer with the fast one. */
    modelmux_control_arm_pending_poll (self, 33);
    g_mutex_unlock (&self->api_control_lock);
    return G_SOURCE_REMOVE;
  }
  g_mutex_unlock (&self->api_control_lock);
  return G_SOURCE_CONTINUE;
}

/* Enqueue a DEFERRED PROMOTION: the stream was just attached as a no-infer passthrough because
 * its primary model is still warming; the poller switches it to the model once WARMED. We always
 * gate on the primary (the mandatory path). wait_shadow=TRUE additionally gates on the shadow
 * (used when a lazy A/B shadow is warming alongside the primary, so the pair promotes together);
 * otherwise the shadow is attached best-effort at promote time. */
static void
modelmux_control_enqueue_promote (GstNvModelMux * self, guint source_id,
    const gchar * primary, const gchar * shadow, gboolean wait_shadow,
    gint p_gpu, gint s_gpu, gboolean p_via_default, gboolean s_via_default)
{
  ModelMuxPendingReroute *deferred_reroute;
  GList *l = self->pending_reroutes, *next;

  /* REPLACE any earlier queued promote for this stream: only the FRESHEST intent may
   * fire. Two live entries race on warm order -- if the STALE target warms first the
   * stream attaches to it (e.g. the demoted old default) and the fresh entry then
   * no-ops because the lane is no longer passthrough. */
  while (l) {
    ModelMuxPendingReroute *old = (ModelMuxPendingReroute *) l->data;
    next = l->next;
    if (old->promote && old->n == 1 && old->src[0] == source_id) {
      MODELMUX_CONTROL_INFO (self, "stream %u: queued promote to '%s' SUPERSEDED by a "
          "fresher route to '%s'", source_id,
          old->eff_primary ? old->eff_primary : "?", primary ? primary : "?");
      self->pending_reroutes = g_list_delete_link (self->pending_reroutes, l);
      modelmux_control_pending_free (old);
    }
    l = next;
  }

  deferred_reroute = g_new0 (ModelMuxPendingReroute, 1);
  deferred_reroute->src = g_memdup2 (&source_id, sizeof (guint));
  deferred_reroute->n = 1;
  deferred_reroute->promote = TRUE;
  deferred_reroute->eff_primary = g_strdup (primary);
  deferred_reroute->eff_shadow = (shadow && *shadow) ? g_strdup (shadow) : NULL;
  deferred_reroute->warm_primary = g_strdup (primary);     /* poll waits on this to reach WARMED */
  deferred_reroute->warm_shadow = (wait_shadow && shadow && *shadow) ? g_strdup (shadow) : NULL;
  deferred_reroute->p_gpu = p_gpu;                         /* placement the promote must honour */
  deferred_reroute->s_gpu = s_gpu;
  deferred_reroute->p_via_default = p_via_default;
  deferred_reroute->s_via_default = s_via_default;
  self->pending_reroutes = g_list_append (self->pending_reroutes, deferred_reroute);
  modelmux_control_publish_pending (self);   /* make the accepted route visible NOW */
  if (!self->pending_poll_id)
    modelmux_control_arm_pending_poll (self, 200);
}

/* Enqueue a DEFERRED SHADOW JOIN: the stream is already attached and inferring on its (ready)
 * primary, but its shadow is still warming (typically a lazily-loaded A/B shadow). The poller
 * waits for the shadow to WARM, then reroutes this one stream to (same primary + shadow) -- a
 * zero-drop add that leaves the primary untouched. */
static void
modelmux_control_enqueue_shadow_join (GstNvModelMux * self, guint source_id,
    const gchar * shadow, gint s_gpu, gboolean s_via_default)
{
  ModelMuxPendingReroute *deferred_reroute = g_new0 (ModelMuxPendingReroute, 1);
  deferred_reroute->src = g_memdup2 (&source_id, sizeof (guint));
  deferred_reroute->n = 1;
  deferred_reroute->promote = FALSE;                       /* reroute (add shadow), not a passthrough promote */
  deferred_reroute->eff_primary = NULL;                    /* PRIMARY UNTOUCHED (NULL = role unspecified all
                                              * the way down): re-asserting the enqueue-time
                                              * primary would UNDO a default switch that swept
                                              * this stream while its shadow warmed -- and PIN
                                              * it to the demoted model (unrecoverable by any
                                              * future sweep). The join only ADDS the shadow. */
  deferred_reroute->eff_shadow = g_strdup (shadow);
  deferred_reroute->warm_primary = NULL;                   /* primary is already ready */
  deferred_reroute->warm_shadow = g_strdup (shadow);       /* gate solely on the shadow warming */
  deferred_reroute->p_gpu = MM_GPU_ANY;                    /* primary untouched (same model -> no-op) */
  deferred_reroute->s_gpu = s_gpu;                         /* shadow joins on the REQUESTED instance */
  deferred_reroute->p_via_default = FALSE;                 /* primary unchanged: same-model no-op */
  deferred_reroute->s_via_default = s_via_default;
  self->pending_reroutes = g_list_append (self->pending_reroutes, deferred_reroute);
  modelmux_control_publish_pending (self);   /* make the accepted route visible NOW */
  if (!self->pending_poll_id)
    modelmux_control_arm_pending_poll (self, 200);
}

/* Runtime-default switch (model/load with set_default_*): move every stream still routed to the
 * demoted OLD default (in this role) onto the NEW default. Only the changed role is re-pointed
 * (the other role is left unchanged). If the new default is already READY the reroute applies
 * now; otherwise it defers (zero-drop) on the pending poller until the new default WARMS.
 *
 * INVARIANT: an explicit route in the same request always wins over the default
 * sweep. excl/n_excl name the source_ids covered by explicit routes[] in the SAME
 * stream/route request (excl_all = an "all"-scope route covered everything): a
 * stream whose explicit route DEFERRED (target still warming, parked on
 * pending_reroutes, physically still on the old default) must NOT be swept onto
 * the new default -- that would clobber the explicit intent. */
static void
modelmux_control_switch_default_reroute (GstNvModelMux * self, gboolean shadow,
    const gchar * old_key, const gchar * new_key, gint gpu,
    const guint * excl, guint n_excl, gboolean excl_all)
{
  /* sized to the LIVE stream count -- a fixed array would silently truncate the
   * sweep on large deployments */
  guint cap = modelmux_bin_num_streams (self->modelmux_bin);
  guint *src = g_new (guint, cap ? cap : 1);
  /* default_only=TRUE: sweep ONLY lanes that resolved to the old default via the
   * FALLBACK -- a lane explicitly pinned to the same model stays where it is. */
  guint n = modelmux_bin_streams_on_model (self->modelmux_bin, shadow, old_key, src,
      cap ? cap : 1, TRUE);
  gint st;

  if (excl_all) {
    /* an explicit "all" route already stated every stream's desired target --
     * the sweep has nothing it may legally move */
    MODELMUX_CONTROL_INFO (self, "default %s switch: sweep skipped, an explicit \"all\" "
        "route in this request covers every stream", shadow ? "shadow" : "primary");
    g_free (src);
    return;
  }
  if (n && n_excl) {
    guint w = 0, i, k;
    for (i = 0; i < n; i++) {
      gboolean skip = FALSE;
      for (k = 0; k < n_excl && !skip; k++)
        skip = (src[i] == excl[k]);
      if (!skip)
        src[w++] = src[i];
    }
    if (w < n)
      MODELMUX_CONTROL_INFO (self, "default %s switch: %u stream(s) skipped by the sweep -- "
          "explicitly routed in this request (explicit routes win)",
          shadow ? "shadow" : "primary", n - w);
    n = w;
  }
  if (!n) {
    MODELMUX_CONTROL_INFO (self, "default %s switch: no live streams on '%s' to reroute",
        shadow ? "shadow" : "primary", old_key);
    g_free (src);
    return;
  }
  st = modelmux_bin_model_status (self->modelmux_bin, shadow, new_key);
  if (st == MODEL_WARMED || st == MODEL_SERVING) {
    /* the swept streams inherit the new default's recorded placement (gpu-addressed
     * default{} -> that instance; MM_GPU_ANY -> balance across its instances) */
    modelmux_control_apply_scoped_reroute (self, src, n,
        shadow ? NULL : new_key, shadow ? new_key : NULL, FALSE,
        shadow ? MM_GPU_ANY : gpu, shadow ? gpu : MM_GPU_ANY,
        TRUE, TRUE /* swept lanes keep FOLLOWING the default */, NULL);
    MODELMUX_CONTROL_INFO (self, "default %s switch: rerouted %u stream(s) '%s' -> '%s'",
        shadow ? "shadow" : "primary", n, old_key, new_key);
  } else {
    /* new default still warming -> defer (zero-drop) until READY; streams keep running the
     * (now-demoted) old default meanwhile. Reuses the pending-reroute poller. */
    ModelMuxPendingReroute *deferred_reroute = g_new0 (ModelMuxPendingReroute, 1);
    /* Withdraw these streams from any OLDER queued sweep first, or two default
     * switches overlapping while their targets warm let WARM ORDER pick the
     * winner: switch to B then to D, D warms first and applies, B warms later and
     * drags the fleet back to the default the operator already replaced.
     *
     * only_default = TRUE: a sweep may reclaim ONLY lanes still following the
     * default. A pending route the operator PINNED outranks it and survives
     * untouched -- the same invariant the immediate path enforces when it collects
     * the streams an explicit route just claimed so the sweep skips them. */
    modelmux_control_pending_supersede (self, src, n, !shadow, shadow,
        TRUE /* default sweep: default-following lanes only */);
    deferred_reroute->src = g_memdup2 (src, n * sizeof (guint));
    deferred_reroute->n = n;
    deferred_reroute->eff_primary = shadow ? NULL : g_strdup (new_key);
    deferred_reroute->eff_shadow  = shadow ? g_strdup (new_key) : NULL;
    deferred_reroute->warm_primary = shadow ? NULL : g_strdup (new_key);
    deferred_reroute->warm_shadow  = shadow ? g_strdup (new_key) : NULL;
    deferred_reroute->p_gpu = shadow ? MM_GPU_ANY : gpu;
    deferred_reroute->s_gpu = shadow ? gpu : MM_GPU_ANY;
    deferred_reroute->p_via_default = TRUE;                /* swept lanes keep FOLLOWING the default */
    deferred_reroute->s_via_default = TRUE;
    self->pending_reroutes = g_list_append (self->pending_reroutes, deferred_reroute);
    modelmux_control_publish_pending (self);   /* make the accepted route visible NOW */
    if (!self->pending_poll_id)
      modelmux_control_arm_pending_poll (self, 200);
    MODELMUX_CONTROL_INFO (self, "default %s switch: %u stream(s) will move '%s' -> '%s' once WARMED",
        shadow ? "shadow" : "primary", n, old_key, new_key);
  }
  g_free (src);
}

/* ================================================================== */
/* /api/v1/stream/route -- the declarative routing plane               */
/* ================================================================== */

/* ROUTE EXECUTOR: apply one admitted route (an explicit source scope +
 * desired model/shadow "name@version" keys) through the existing reroute
 * machinery -- in-place relabel when clean, else prepare/warm the target(s)
 * and defer the switch until READY (zero-drop). n=0 means ALL streams. */
static void
modelmux_control_exec_route (GstNvModelMux * self, const guint * src, guint n,
    const gchar * p_key, const gchar * p_bare, gint p_gpu,
    const gchar * s_key, const gchar * s_bare, gint s_gpu,
    gboolean clear_shadow, gboolean * out_deferred)
{
  const gchar *eff_primary = NULL, *eff_shadow = NULL;
  const gchar *warm_primary = NULL, *warm_shadow = NULL;

  MODELMUX_CONTROL_INFO (self, "STREAM ROUTE scope=%s model=%s shadow=%s",
      n ? "explicit" : "ALL", p_key ? p_key : "-",
      clear_shadow ? "none" : (s_key ? s_key : "-"));

  /* IN-PLACE fast path: a clean promote/swap covering the scope -> relabel the
   * bins (no new bin, gie preserved, zero drop). Shadow-clear is signalled to
   * try_inplace by the literal "none" token. SKIPPED for gpu-addressed routes:
   * relabeling keeps every stream on its current shard, but an explicit gpu may
   * require moving streams onto the instance on that device -- the full role
   * machinery (with its MIGRATE upgrade) handles that. */
  if (p_key && p_gpu == MM_GPU_ANY && s_gpu == MM_GPU_ANY &&
      modelmux_bin_try_inplace (self->modelmux_bin, n ? src : NULL, n, p_key,
          clear_shadow ? "none" : s_key)) {
    MODELMUX_CONTROL_INFO (self, "stream/route handled IN-PLACE (reused bins, gie preserved, "
        "zero drop)");
    /* This route just LANDED, so any route still queued for these streams is
     * stale -- leaving it would let a warm completing later drag them back off
     * what was just applied. Supersede applies to every route that succeeds,
     * not only to ones that defer. */
    modelmux_control_pending_supersede (self, src, n, p_key != NULL,
        (s_key != NULL || clear_shadow), FALSE /* explicit route: claims any lane */);
    modelmux_bin_bump_routing_revision (self->modelmux_bin);
    return;
  }

  if (p_key) {
    if (!modelmux_config_find_model (&self->config, p_bare)) {
      MODELMUX_CONTROL_ERR (self, "route rejected: model '%s' not loaded (MODEL_NOT_LOADED) "
          "-- POST /api/v1/model/load first", p_key);
      return;
    }
    if (modelmux_bin_prepare_model (self->modelmux_bin, FALSE, p_key) != MODEL_SERVING &&
        modelmux_bin_model_status (self->modelmux_bin, FALSE, p_key) != MODEL_WARMED)
      warm_primary = p_key;
    eff_primary = p_key;
  }

  if (s_key) {
    if (!modelmux_config_find_model (&self->config, s_bare)) {
      MODELMUX_CONTROL_ERR (self, "route rejected: shadow '%s' not loaded (MODEL_NOT_LOADED) "
          "-- POST /api/v1/model/load first", s_key);
      return;
    }
    if (modelmux_bin_prepare_model (self->modelmux_bin, TRUE, s_key) != MODEL_SERVING &&
        modelmux_bin_model_status (self->modelmux_bin, TRUE, s_key) != MODEL_WARMED)
      warm_shadow = s_key;
    eff_shadow = s_key;
  }

  if (!eff_primary && !eff_shadow && !clear_shadow) {
    MODELMUX_CONTROL_WARN (self, "route no-op: nothing to change");
    return;
  }

  /* FRESHEST intent wins. Withdraw these streams from any route already queued
   * for them, HERE -- past every rejection and the no-op guard, so a request that
   * was refused cannot cancel a pending one, and before the defer/apply split, so
   * BOTH outcomes supersede.
   *
   * Scoping this to the defer branch (as it first was) left the common case
   * broken: routing to an ALREADY-LOADED model applies immediately, and the stale
   * entry survived to drag the stream back when its own target warmed later. What
   * matters is that a newer route SUCCEEDED, not how it was carried out. */
  modelmux_control_pending_supersede (self, src, n, eff_primary != NULL,
      (eff_shadow != NULL || clear_shadow), FALSE /* explicit route: claims any lane */);

  /* defer until a still-warming target is READY (keeps the switch zero-drop) */
  if (warm_primary || warm_shadow) {
    ModelMuxPendingReroute *deferred_reroute = g_new0 (ModelMuxPendingReroute, 1);
    if (n) {
      deferred_reroute->src = g_memdup2 (src, n * sizeof (guint));
      deferred_reroute->n = n;
    } else {
      /* FREEZE an all-scope route to the fleet it meant AT ACCEPT TIME. Left as
       * the n==0 sentinel it would be re-read when the target finally warms, so a
       * stream added during the warm -- or a recycled id belonging to a stream the
       * operator never routed -- would be swept up by an instruction issued before
       * it existed. Warm times are engine-build long, so that window is wide.
       *
       * This also makes the scope arithmetic in pending_supersede honest: its
       * "old scope is ALL" branch materialises against the LIVE fleet, which is
       * the fleet at supersede time, not at accept time. Freezing here means a
       * queued reroute never carries the sentinel, so the two readings cannot
       * disagree. (0 live streams -> nothing to route; leave it empty and the
       * apply is a no-op, which is the honest outcome.) */
      deferred_reroute->n =
          modelmux_bin_active_stream_ids (self->modelmux_bin, &deferred_reroute->src);
      if (deferred_reroute->n == 0) {
        /* Nothing wired right now, so this route has no work -- and it must NOT be
         * queued with n==0, because that is the sentinel for ALL STREAMS in both
         * ModelMuxPendingReroute and the mirrored note. Queuing it would mean an
         * all-route issued against an empty pipeline silently claiming every
         * stream that attaches later, which is the exact defect freezing the scope
         * is here to prevent. Drop it instead: no stream was routed, so -- as with
         * an apply that touches nothing -- no completion is announced. */
        MODELMUX_CONTROL_WARN (self, "stream/route: 'all' scope matched no wired "
            "stream at accept time -> nothing to defer");
        modelmux_control_pending_free (deferred_reroute);
        return;
      }
    }
    deferred_reroute->eff_primary = g_strdup (eff_primary);
    deferred_reroute->eff_shadow = g_strdup (eff_shadow);
    deferred_reroute->clear_shadow = clear_shadow;
    deferred_reroute->warm_primary = g_strdup (warm_primary);
    deferred_reroute->warm_shadow = g_strdup (warm_shadow);
    deferred_reroute->p_gpu = p_gpu;
    deferred_reroute->s_gpu = s_gpu;
    deferred_reroute->p_via_default = FALSE;               /* explicit route: the lanes become PINNED */
    deferred_reroute->s_via_default = FALSE;
    deferred_reroute->emit_routed = TRUE;                  /* this POST's completion is owed */
    self->pending_reroutes = g_list_append (self->pending_reroutes, deferred_reroute);
    modelmux_control_publish_pending (self);   /* make the accepted route visible NOW */
    /* An ACCEPTED route changes the routing document even before it applies: the
     * read now reports a pending target that was not there a moment ago. Bump the
     * revision so if_revision keeps protecting what it advertises -- otherwise two
     * controllers holding the same revision R both pass the CAS, and the second
     * silently supersedes the first's accepted route while its 202 said the work
     * was taken. The no-op rule still holds (a rejected or no-op request never
     * reaches here); what changed is that "applied" is no longer the only kind of
     * routing change an observer can see. */
    modelmux_bin_bump_routing_revision (self->modelmux_bin);
    if (!self->pending_poll_id)
      modelmux_control_arm_pending_poll (self, 200);
    if (out_deferred)
      *out_deferred = TRUE;
    MODELMUX_CONTROL_INFO (self, "route accepted -> warming target(s); reroute applies "
        "(zero-drop) once READY");
    return;
  }

  modelmux_control_apply_scoped_reroute (self, src, n, eff_primary, eff_shadow, clear_shadow,
      p_gpu, s_gpu, FALSE, FALSE /* explicit route: the lanes become PINNED */,
      NULL);
}

/* A gpu-addressed ROUTE ref must land on one of the version's LIVE instances
 * (an instance group spans GPUs; a not-live model falls back to its RECORDED
 * placement) -- "gpu" SELECTS (and validates), it never creates. Unrecorded
 * placement = the default device 0. */
static gboolean
modelmux_control_route_gpu_ok (GstNvModelMux * self, const gchar * role, const gchar * name,
    const gchar * ver, gint want, guint route_idx)
{
  gchar *gpus;

  if (want < 0 || !name || !ver)
    return TRUE;
  if (want == G_MAXINT) {
    /* out-of-range sentinel (see modelmux_control_ctrl_json_gpu): print the teaching
     * bound, never "addressed on gpu 2147483647" */
    MODELMUX_CONTROL_ERR (self, "route[%u] rejected: %s '%s@%s' gpu id out of range "
        "(0..%d) (GPU_UNKNOWN)", route_idx, role, name, ver, MODELMUX_CONTROL_GPU_ID_MAX);
    return FALSE;
  }
  if (modelmux_control_ref_gpu_matches (self, name, ver, want))
    return TRUE;
  gpus = modelmux_control_live_gpu_list (self, name, ver);
  MODELMUX_CONTROL_ERR (self, "route[%u] rejected: %s '%s@%s' addressed on gpu %d but its "
      "instance(s) run on gpu(s) %s (GPU_UNKNOWN)", route_idx, role, name, ver,
      want, gpus ? gpus : "0");
  g_free (gpus);
  return FALSE;
}

/* Move a role's runtime DEFAULT to (name, ver): the fallback every unbound
 * stream resolves to. The target must already be LOADED (stream/route never
 * creates models); the previous default's streams move over zero-drop.
 * `gpu` = the default{} ref's requested instance placement (validated upstream
 * by modelmux_control_route_gpu_ok; MM_GPU_ANY = any). PERSISTED on the role's
 * pool DefaultModelRef so every FUTURE default-resolved stream (add/bind) and
 * the sweep below honour it.
 * excl/n_excl/excl_all: source_ids covered by explicit SERVING-MODEL routes[]
 * in the SAME request, forwarded to the sweep (an explicit route that changes
 * the serving model always wins over the default sweep). */
static void
modelmux_control_route_set_default (GstNvModelMux * self, gboolean shadow,
    const gchar * name, const gchar * ver, gint gpu,
    const guint * excl, guint n_excl, gboolean excl_all)
{
  const gchar *role_str = shadow ? "shadow" : "model";
  gchar *key = model_key (name, ver);
  gchar *other_key = NULL, *old_key = NULL;
  ModelStatus st;

  /* identical-default guard: the new default must not equal the OTHER role's
   * current default (same name@version in both roles is a pointless A/B). */
  {
    const DefaultModelRef *oref =
        modelmux_bin_get_default (self->modelmux_bin, !shadow);
    other_key = (oref && oref->key) ? g_strdup (oref->key) : NULL;
  }
  if (other_key && g_strcmp0 (other_key, key) == 0) {
    MODELMUX_CONTROL_ERR (self, "default %s '%s' rejected: it is already the default %s -- "
        "a model can't be BOTH default roles (pointless A/B)", role_str, key,
        shadow ? "model" : "shadow");
    g_free (other_key);
    g_free (key);
    return;
  }
  g_free (other_key);

  if (!modelmux_bin_model_loaded (self->modelmux_bin, key)) {
    MODELMUX_CONTROL_ERR (self, "default %s '%s' rejected: not loaded (MODEL_NOT_LOADED) -- "
        "POST /api/v1/model/load first", role_str, key);
    g_free (key);
    return;
  }

  /* capture the OLD default for THIS role (before the switch) so its streams can move. */
  {
    const DefaultModelRef *cref =
        modelmux_bin_get_default (self->modelmux_bin, shadow);
    old_key = (cref && cref->key) ? g_strdup (cref->key) : NULL;
  }

  /* ensure the target is (or is becoming) a member of the role's pool, then
   * promote it to the runtime default. A FAILED model is NEVER promoted --
   * that would route every subsequent stream-add to a dead model AND demote
   * the working old default (a one-request fleet outage on a typo). */
  st = modelmux_bin_prepare_model (self->modelmux_bin, shadow, key);
  if (st == MODEL_FAILED) {
    MODELMUX_CONTROL_ERR (self, "default %s '%s' rejected: engine FAILED to warm -- previous "
        "default (and all routing) left unchanged", role_str, key);
    g_free (old_key);
    g_free (key);
    return;
  }
  /* identity + requested placement pin land in the pool ref in ONE locked
   * write: every FUTURE stream that RESOLVES to this default (stream-add/bind
   * fallback) inherits the pin, exactly like an explicitly gpu-addressed
   * route (MM_GPU_ANY = any instance). */
  modelmux_bin_set_default (self->modelmux_bin, shadow, name, ver, gpu);
  modelmux_bin_bump_routing_revision (self->modelmux_bin);
  MODELMUX_CONTROL_INFO (self, "default %s -> '%s' (status=%s)%s", role_str, key,
      model_status_str (st),
      (old_key && g_strcmp0 (old_key, key) != 0)
          ? " -- previous default demoted to normal (will unload when idle)" : "");

  /* baseline swap: reroute every stream on the OLD default onto the NEW one,
   * zero-drop (deferred until the new default WARMS). */
  if (old_key && g_strcmp0 (old_key, key) != 0)
    modelmux_control_switch_default_reroute (self, shadow, old_key, key, gpu, excl, n_excl,
        excl_all);
  g_free (old_key);
  g_free (key);
}

/* Current routing revision (under ib->lock; written by the bump helper). */
static guint64
modelmux_control_routing_revision (GstNvModelMux * self)
{
  guint64 rev;
  g_mutex_lock (&self->modelmux_bin->lock);
  rev = self->modelmux_bin->routing_revision;
  g_mutex_unlock (&self->modelmux_bin->lock);
  return rev;
}

/* Reverse-resolve a camera_id to its live source_id via ib->stream_names
 * (locked readers only). -1 when no live stream carries that id. */
static gint
modelmux_control_lookup_source_by_name (GstNvModelMux * self, const gchar * cam)
{
  GHashTableIter it;
  gpointer k, v;
  gint found = -1;

  g_mutex_lock (&self->modelmux_bin->lock);
  if (self->modelmux_bin->stream_names) {
    g_hash_table_iter_init (&it, self->modelmux_bin->stream_names);
    while (found < 0 && g_hash_table_iter_next (&it, &k, &v))
      if (g_strcmp0 ((const gchar *) v, cam) == 0)
        found = GPOINTER_TO_INT (k);
  }
  g_mutex_unlock (&self->modelmux_bin->lock);
  return found;
}

/* Element-boundary ref validation. The REST parser enforces this for its own
 * clients, but stream-route events/messages can also be APP-GENERATED -- the
 * same gate here keeps a malformed ref from ever warming a bin under a
 * garbage key ('@'/';' corrupt the canonical name@version key space; a
 * non-integer version could never be addressed by unload). */
static gboolean
modelmux_control_ref_valid (GstNvModelMux * self, guint idx, const gchar * what,
    const gchar * name, const gchar * ver)
{
  if (!name || !*name || strchr (name, '@') || strchr (name, ';')) {
    MODELMUX_CONTROL_ERR (self, "route[%u] rejected: %s ref needs a 'name' without "
        "'@' or ';' (NAME_INVALID)", idx, what);
    return FALSE;
  }
  if (!modelmux_version_is_int (ver)) {
    MODELMUX_CONTROL_ERR (self, "route[%u] rejected: %s '%s' needs a positive-integer "
        "'version' (VERSION_INVALID)", idx, what, name);
    return FALSE;
  }
  return TRUE;
}

/* One admitted route, decoded. */
typedef struct
{
  gchar   *p_name, *p_ver, *p_key;    /* serving target (NULL = unchanged) */
  gchar   *s_name, *s_ver, *s_key;    /* shadow target  (NULL = unchanged) */
  gint     p_gpu, s_gpu;              /* requested instance gpu (-1 = any) */
  gboolean clear_shadow;
  guint   *src;                       /* scope ids, sized to the request    */
  guint    n;                         /* 0 = ALL streams ("all" route)      */
  /* set by the apply pass: this route parked on a warming target instead of
   * switching now, so it must be left OUT of the completion summary -- the
   * pending poller announces it when it actually lands. Per ROUTE, not per
   * request: a request may mix routes that applied with routes that deferred,
   * and the ones that applied still deserve their notice. */
  gboolean deferred;
} ModelMuxDecodedRoute;

static void
modelmux_control_decoded_routes_free (ModelMuxDecodedRoute * dr, guint n)
{
  guint i;
  for (i = 0; i < n; i++) {
    g_free (dr[i].p_name);
    g_free (dr[i].p_ver);
    g_free (dr[i].p_key);
    g_free (dr[i].s_name);
    g_free (dr[i].s_ver);
    g_free (dr[i].s_key);
    g_free (dr[i].src);
  }
  g_free (dr);
}

/* One-line summary for a `streams-routed` notification -- the app prints it
 * inside the brackets, e.g.
 *   new streams routed [streams=Cam-3 primary=Trafficcamnet@2 shadow=Trafficcamnet@1]
 * Per applied route: WHICH streams it touched (camera names when known, else
 * "#<id>"; "all" for a whole-pipeline route) and the primary/shadow it set.
 * A route that also moves the POOL DEFAULT appends a "default ..." segment.
 * Returns a newly-allocated string (never NULL, may be empty); caller frees. */
static gchar *
modelmux_control_route_summary (GstNvModelMux * self,
    const ModelMuxDecodedRoute * dr, guint nr,
    gboolean def_primary, const gchar * dm_name, const gchar * dm_ver,
    gboolean def_shadow, const gchar * ds_name, const gchar * ds_ver)
{
  GString *s = g_string_new (NULL);
  gboolean first = TRUE;
  guint i, k;
  for (i = 0; i < nr; i++) {
    if (!first)
      g_string_append (s, "; ");
    first = FALSE;
    /* stream scope */
    if (dr[i].n == 0) {
      g_string_append (s, "streams=all");
    } else {
      g_string_append (s, "streams=");
      for (k = 0; k < dr[i].n; k++) {
        const gchar *nm = modelmux_bin_source_name (self->modelmux_bin, dr[i].src[k]);
        if (k)
          g_string_append_c (s, ',');
        if (nm)
          g_string_append (s, nm);
        else
          g_string_append_printf (s, "#%u", dr[i].src[k]);
      }
    }
    /* primary + shadow targets (NULL = left unchanged) */
    if (dr[i].p_name)
      g_string_append_printf (s, " primary=%s@%s", dr[i].p_name,
          dr[i].p_ver ? dr[i].p_ver : "?");
    if (dr[i].clear_shadow)
      g_string_append (s, " shadow=cleared");
    else if (dr[i].s_name)
      g_string_append_printf (s, " shadow=%s@%s", dr[i].s_name,
          dr[i].s_ver ? dr[i].s_ver : "?");
  }
  /* pool DEFAULT switch: a route can move the fallback model/shadow too */
  if ((def_primary && dm_name) || (def_shadow && ds_name)) {
    if (!first)
      g_string_append (s, "; ");
    g_string_append (s, "default");
    if (def_primary && dm_name)
      g_string_append_printf (s, " primary=%s@%s", dm_name, dm_ver ? dm_ver : "?");
    if (def_shadow && ds_name)
      g_string_append_printf (s, " shadow=%s@%s", ds_name, ds_ver ? ds_ver : "?");
  }
  return g_string_free (s, FALSE);
}

/* Resolve one route's stream scope. Preference order:
 *   1. "source_ids" -- the ids the upstream nvmultiurisrcbin resolved and
 *      attached (the REST path).
 *   2. "streams"    -- the request's own member (bus/app-generated path):
 *      the literal "all", or a camera_id list resolved against the live
 *      stream table HERE.
 * An omitted or unresolvable scope REJECTS the route -- it must never widen
 * to "everything". */
static gboolean
modelmux_control_route_scope (GstNvModelMux * self, JsonObject * r, guint idx,
    ModelMuxDecodedRoute * out)
{
  out->n = 0;
  if (json_object_has_member (r, "source_ids")) {
    JsonNode *sn = json_object_get_member (r, "source_ids");
    JsonArray *ids;
    guint k, kn;
    if (!sn || !JSON_NODE_HOLDS_ARRAY (sn)) {
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: 'source_ids' must be an id array", idx);
      return FALSE;
    }
    ids = json_node_get_array (sn);
    kn = json_array_get_length (ids);
    if (!kn) {
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: 'source_ids' scope is empty", idx);
      return FALSE;
    }
    out->src = g_new (guint, kn);      /* sized to the request, no fixed cap */
    for (k = 0; k < kn; k++) {
      /* STRICT per-element typing: json_array_get_int_element coerces a
       * non-integer element (e.g. the string "3") to 0, which would silently
       * apply the route to source 0 -- the WRONG camera. Reject instead
       * (admission is atomic, the whole request rejects). */
      JsonNode *e = json_array_get_element (ids, k);
      gint64 id;
      if (!e || !JSON_NODE_HOLDS_VALUE (e) ||
          json_node_get_value_type (e) != G_TYPE_INT64) {
        MODELMUX_CONTROL_ERR (self, "route[%u] rejected: 'source_ids' element %u is not an "
            "integer (FIELD_TYPE_INVALID) -- a coerced id would route the "
            "wrong stream", idx, k);
        return FALSE;
      }
      id = json_node_get_int (e);
      if (id < 0) {
        MODELMUX_CONTROL_ERR (self, "route[%u] rejected: 'source_ids' element %u is "
            "negative (FIELD_TYPE_INVALID)", idx, k);
        return FALSE;
      }
      out->src[out->n++] = (guint) id;
    }
    return TRUE;
  }
  if (json_object_has_member (r, "streams")) {
    JsonNode *st = json_object_get_member (r, "streams");
    if (st && JSON_NODE_HOLDS_VALUE (st) &&
        json_node_get_value_type (st) == G_TYPE_STRING &&
        g_strcmp0 (json_node_get_string (st), "all") == 0)
      return TRUE;                     /* n stays 0 = every stream */
    if (st && JSON_NODE_HOLDS_ARRAY (st)) {
      JsonArray *cams = json_node_get_array (st);
      guint k, kn = json_array_get_length (cams);
      if (!kn) {
        MODELMUX_CONTROL_ERR (self, "route[%u] rejected: 'streams' scope is empty", idx);
        return FALSE;
      }
      out->src = g_new (guint, kn);    /* sized to the request, no fixed cap */
      for (k = 0; k < kn; k++) {
        JsonNode *c = json_array_get_element (cams, k);
        const gchar *cam = (c && JSON_NODE_HOLDS_VALUE (c) &&
            json_node_get_value_type (c) == G_TYPE_STRING) ?
            json_node_get_string (c) : NULL;
        gint sid = cam ? modelmux_control_lookup_source_by_name (self, cam) : -1;
        if (sid < 0) {
          MODELMUX_CONTROL_ERR (self, "route[%u] rejected: unknown stream '%s' "
              "(STREAM_UNKNOWN) -- no live source carries that camera_id",
              idx, cam ? cam : "?");
          return FALSE;
        }
        out->src[out->n++] = (guint) sid;
      }
      return TRUE;
    }
  }
  MODELMUX_CONTROL_ERR (self, "route[%u] rejected: no resolvable scope -- give an explicit "
      "'streams' camera_id list or the literal \"all\" (an omitted scope "
      "never means \"everything\")", idx);
  return FALSE;
}

/* /api/v1/stream/route: ADMIT the whole routing request first, apply only if
 * EVERY part passes (atomic admission -- a partially-applied request is worse
 * than a rejected one), then execute route by route and move the defaults.
 * The REST parser and the upstream bin have already gated their own layers
 * for REST clients; everything is re-checked here because events/messages can
 * also be app-generated. */
/* @admit_only: run only the ADMISSION PASS (validate + reject with a reason) and
 * stop before applying -- used by the synchronous control-admission query, which
 * then schedules the real apply asynchronously. FALSE = validate AND apply (the
 * normal in-band event path). */
static void
modelmux_control_handle_stream_route (GstNvModelMux * self, const gchar * json,
    gboolean admit_only)
{
  JsonParser *parser = modelmux_control_ctrl_json_load (self, json, "stream/route");
  JsonObject *v, *d = NULL;
  JsonArray *routes = NULL;
  ModelMuxDecodedRoute *dr = NULL;
  guint nr = 0, i;
  gchar *dm_name = NULL, *dm_ver = NULL, *ds_name = NULL, *ds_ver = NULL;
  gint dm_gpu = -1, ds_gpu = -1;
  gboolean dm_null = FALSE, ds_null = FALSE, has_def_shadow = FALSE;

  if (!parser)
    return;
  v = json_node_get_object (json_parser_get_root (parser));

  /* if_revision: optimistic-concurrency guard. A stale revision rejects the
   * WHOLE request -- the caller re-reads model/status and retries. */
  if (json_object_has_member (v, "if_revision")) {
    JsonNode *n = json_object_get_member (v, "if_revision");
    if (n && JSON_NODE_HOLDS_VALUE (n) &&
        json_node_get_value_type (n) == G_TYPE_INT64) {
      gint64 want = json_node_get_int (n);
      guint64 cur = modelmux_control_routing_revision (self);
      if (want >= 0 && (guint64) want != cur) {
        MODELMUX_CONTROL_ERR (self, "stream/route rejected: if_revision %" G_GINT64_FORMAT
            " does not match routing_revision %" G_GUINT64_FORMAT
            " (REVISION_CONFLICT) -- re-read model/status and retry",
            want, cur);
        g_object_unref (parser);
        return;
      }
    }
  }

  /* ---------- ADMISSION PASS: decode + validate everything ---------- */
  if (json_object_has_member (v, "routes")) {
    JsonNode *rn = json_object_get_member (v, "routes");
    if (!rn || !JSON_NODE_HOLDS_ARRAY (rn)) {
      /* a present-but-non-array 'routes' must fail the WHOLE request --
       * decoding it as "no routes" would silently apply the default{} with
       * an empty exclusion set (partial intent, exactly what atomic
       * admission forbids) */
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: 'routes' must be an array "
          "(FIELD_TYPE_INVALID)");
      goto reject;
    }
    routes = json_node_get_array (rn);
    nr = json_array_get_length (routes);
  }
  if (nr)
    dr = g_new0 (ModelMuxDecodedRoute, nr);

  for (i = 0; i < nr; i++) {
    JsonNode *el = json_array_get_element (routes, i);
    JsonObject *r =
        (el && JSON_NODE_HOLDS_OBJECT (el)) ? json_node_get_object (el) : NULL;
    gint p_gpu = -1, s_gpu = -1;
    gboolean model_null = FALSE, shadow_null = FALSE, has_shadow;
    gboolean p_malformed = FALSE, s_malformed = FALSE;

    if (!r) {
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: each route must be an object", i);
      goto reject;
    }
    if (!modelmux_control_route_scope (self, r, i, &dr[i]))
      goto reject;
    modelmux_control_json_ref (r, "model", &model_null, &dr[i].p_name, &dr[i].p_ver, &p_gpu,
        &p_malformed);
    has_shadow = modelmux_control_json_ref (r, "shadow", &shadow_null, &dr[i].s_name,
        &dr[i].s_ver, &s_gpu, &s_malformed);
    dr[i].clear_shadow = (has_shadow && shadow_null);
    /* record each ref's requested placement (validated below by modelmux_control_route_gpu_ok)
     * so the executor can constrain shard selection / trigger a gpu migration */
    dr[i].p_gpu = p_gpu;
    dr[i].s_gpu = s_gpu;
    /* a gpu with no name+version addresses nothing: it would pass admission
     * (the name/ver validators are skipped) and then be silently inert --
     * an acknowledged-then-ignored intent. Reject whole. */
    if (!dr[i].p_name && !dr[i].p_ver && !model_null && p_gpu != -1) {
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: model 'gpu' given without "
          "name+version (FIELD_MISSING)", i);
      goto reject;
    }
    if (!shadow_null && !dr[i].s_name && !dr[i].s_ver && s_gpu != -1) {
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: shadow 'gpu' given without "
          "name+version (FIELD_MISSING)", i);
      goto reject;
    }

    if (p_malformed || s_malformed) {
      /* a wrong-typed ref (e.g. "model": "Car;2" or 123) decodes to NULL
       * name/version and would silently read as "unchanged" -- the request
       * stated an intent we cannot decode, so it must reject WHOLE */
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: '%s' must be a {\"name\",\"version\"} "
          "object or null (FIELD_TYPE_INVALID) -- a wrong-typed ref would "
          "silently read as 'unchanged'", i, p_malformed ? "model" : "shadow");
      goto reject;
    }

    if (model_null) {
      /* explicit PASSTHROUGH ("model": null). TODO(API_DESIGN.md): live
       * detach-to-passthrough. Until it exists this is a HARD failure -- a
       * request must never be acknowledged and then silently not applied. */
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: \"model\": null (passthrough) is "
          "not available yet (NOT_IMPLEMENTED)", i);
      goto reject;
    }
    if (dr[i].p_name || dr[i].p_ver) {
      if (!modelmux_control_ref_valid (self, i, "model", dr[i].p_name, dr[i].p_ver))
        goto reject;
      if (!modelmux_config_find_model (&self->config, dr[i].p_name)) {
        MODELMUX_CONTROL_ERR (self, "route rejected: model '%s@%s' not loaded "
            "(MODEL_NOT_LOADED) -- POST /api/v1/model/load first",
            dr[i].p_name, dr[i].p_ver);
        goto reject;
      }
      if (!modelmux_control_route_gpu_ok (self, "model", dr[i].p_name, dr[i].p_ver, p_gpu, i))
        goto reject;
      dr[i].p_key = model_key (dr[i].p_name, dr[i].p_ver);
      /* the catalog check above matches the NAME only -- the target VERSION
       * must ALSO be live (loaded/warming), exactly like the default{} targets
       * below: stream/route never creates models, so a never-loaded version
       * (Car@99) must reject here instead of letting modelmux_control_exec_route's
       * prepare_model silently create + warm it. A WARMING model has a live
       * bin, so deferred reroutes still admit. */
      if (!modelmux_bin_model_loaded (self->modelmux_bin, dr[i].p_key)) {
        MODELMUX_CONTROL_ERR (self, "route rejected: model '%s@%s' not loaded "
            "(MODEL_NOT_LOADED) -- POST /api/v1/model/load first",
            dr[i].p_name, dr[i].p_ver);
        goto reject;
      }
    }
    if (!shadow_null && (dr[i].s_name || dr[i].s_ver)) {
      if (!modelmux_control_ref_valid (self, i, "shadow", dr[i].s_name, dr[i].s_ver))
        goto reject;
      if (!modelmux_config_find_model (&self->config, dr[i].s_name)) {
        MODELMUX_CONTROL_ERR (self, "route rejected: shadow '%s@%s' not loaded "
            "(MODEL_NOT_LOADED) -- POST /api/v1/model/load first",
            dr[i].s_name, dr[i].s_ver);
        goto reject;
      }
      if (!modelmux_control_route_gpu_ok (self, "shadow", dr[i].s_name, dr[i].s_ver, s_gpu, i))
        goto reject;
      dr[i].s_key = model_key (dr[i].s_name, dr[i].s_ver);
      /* same version-liveness admission as the serving target above */
      if (!modelmux_bin_model_loaded (self->modelmux_bin, dr[i].s_key)) {
        MODELMUX_CONTROL_ERR (self, "route rejected: shadow '%s@%s' not loaded "
            "(MODEL_NOT_LOADED) -- POST /api/v1/model/load first",
            dr[i].s_name, dr[i].s_ver);
        goto reject;
      }
    }
    if (!dr[i].p_key && !dr[i].s_key && !dr[i].clear_shadow) {
      MODELMUX_CONTROL_ERR (self, "route[%u] rejected: changes neither 'model' nor "
          "'shadow' (EMPTY_REQUEST)", i);
      goto reject;
    }
  }

  if (json_object_has_member (v, "default")) {
    JsonNode *dn = json_object_get_member (v, "default");
    d = (dn && JSON_NODE_HOLDS_OBJECT (dn)) ? json_node_get_object (dn) : NULL;
    if (!d) {
      /* a present-but-wrong-typed 'default' must reject WHOLE, like a
       * wrong-typed routes[]: proceeding would apply the routes while
       * silently dropping the stated default intent (partial mutation). */
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: 'default' must be an object "
          "(FIELD_TYPE_INVALID)");
      goto reject;
    }
  }
  if (d) {
    gboolean dm_malformed = FALSE, ds_malformed = FALSE;
    modelmux_control_json_ref (d, "model", &dm_null, &dm_name, &dm_ver, &dm_gpu,
        &dm_malformed);
    has_def_shadow = modelmux_control_json_ref (d, "shadow", &ds_null, &ds_name, &ds_ver,
        &ds_gpu, &ds_malformed);
    if (dm_malformed || ds_malformed) {
      /* same wrong-typed-ref hole as routes[] above: reject, never a silent
       * "unchanged" */
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: default '%s' must be a "
          "{\"name\",\"version\"} object or null (FIELD_TYPE_INVALID)",
          dm_malformed ? "model" : "shadow");
      goto reject;
    }
    if (dm_null) {
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: the default model cannot be null "
          "(SHADOW_WITHOUT_MODEL)");
      goto reject;
    }
    /* a gpu with no name+version addresses nothing -- acknowledging it and
     * then ignoring it would break the atomic-admission contract */
    if (!dm_name && !dm_ver && dm_gpu != -1) {
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: default model 'gpu' given without "
          "name+version (FIELD_MISSING)");
      goto reject;
    }
    if (has_def_shadow && !ds_null && !ds_name && !ds_ver && ds_gpu != -1) {
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: default shadow 'gpu' given without "
          "name+version (FIELD_MISSING)");
      goto reject;
    }
    if (dm_name || dm_ver) {
      if (!modelmux_control_ref_valid (self, 0, "default model", dm_name, dm_ver) ||
          !modelmux_control_route_gpu_ok (self, "default model", dm_name, dm_ver, dm_gpu, 0))
        goto reject;
    }
    if (has_def_shadow && ds_null) {
      /* clearing the DEFAULT shadow needs a dedicated bin primitive
       * (TODO API_DESIGN.md). HARD failure until then -- never a silent keep. */
      MODELMUX_CONTROL_ERR (self, "stream/route rejected: default \"shadow\": null (clear "
          "the default shadow) is not available yet (NOT_IMPLEMENTED)");
      goto reject;
    }
    if (has_def_shadow && !ds_null && (ds_name || ds_ver)) {
      if (!modelmux_control_ref_valid (self, 0, "default shadow", ds_name, ds_ver) ||
          !modelmux_control_route_gpu_ok (self, "default shadow", ds_name, ds_ver, ds_gpu, 0))
        goto reject;
    }
    /* F4: default targets must be LOADED at admission -- otherwise routes[]
     * would apply and the default switch alone would fail (partial mutation). */
    if (dm_name && dm_ver) {
      gchar *dk = model_key (dm_name, dm_ver);
      gboolean ok = modelmux_bin_model_loaded (self->modelmux_bin, dk);
      if (!ok)
        MODELMUX_CONTROL_ERR (self, "stream/route rejected: default model '%s' not loaded "
            "(MODEL_NOT_LOADED) -- POST /api/v1/model/load first", dk);
      g_free (dk);
      if (!ok)
        goto reject;
    }
    if (has_def_shadow && ds_name && ds_ver) {
      gchar *dk = model_key (ds_name, ds_ver);
      gboolean ok = modelmux_bin_model_loaded (self->modelmux_bin, dk);
      if (!ok)
        MODELMUX_CONTROL_ERR (self, "stream/route rejected: default shadow '%s' not loaded "
            "(MODEL_NOT_LOADED) -- POST /api/v1/model/load first", dk);
      g_free (dk);
      if (!ok)
        goto reject;
    }
  }

  /* everything admitted. In admit_only mode the query caller schedules the real
   * apply asynchronously -- skip straight to cleanup (go to 'out', NOT 'reject':
   * that path logs a rejection the capture would mistake for a failure). */
  if (admit_only)
    goto out;

  /* ---------- APPLY PASS: everything admitted -- execute ---------- */
  for (i = 0; i < nr; i++)
    modelmux_control_exec_route (self, dr[i].src, dr[i].n, dr[i].p_key, dr[i].p_name,
        dr[i].p_gpu, dr[i].s_key, dr[i].s_name, dr[i].s_gpu,
        dr[i].clear_shadow, &dr[i].deferred);
  if (d) {
    /* INVARIANT: an explicit route that CHANGES THE SERVING MODEL always wins
     * over the default sweep. Collect the source_ids such routes[] above just
     * covered (applied OR deferred -- a deferred route's streams are
     * physically still on the old default) and EXCLUDE them from the default
     * switch's sweep, so it cannot clobber the explicit intent. A
     * serving-model route scoped "all" covers everything (excl_all): the
     * sweep becomes a no-op. SHADOW-ONLY routes (p_key == NULL) stay
     * SWEEPABLE: they never stated a serving-model intent, so excluding them
     * would strand their streams on the demoted old default. */
    guint *excl = NULL;
    guint n_excl = 0;
    gboolean excl_all = FALSE;
    if (nr) {
      guint total = 0, k;
      for (i = 0; i < nr; i++) {
        if (!dr[i].p_key)
          continue;                     /* shadow-only route: sweepable */
        if (dr[i].n == 0)
          excl_all = TRUE;              /* "all"-scope serving-model route */
        total += dr[i].n;
      }
      if (!excl_all && total) {
        excl = g_new (guint, total);
        for (i = 0; i < nr; i++) {
          if (!dr[i].p_key)
            continue;
          for (k = 0; k < dr[i].n; k++)
            excl[n_excl++] = dr[i].src[k];
        }
      }
    }
    if (dm_name && dm_ver)
      modelmux_control_route_set_default (self, FALSE, dm_name, dm_ver, dm_gpu, excl, n_excl,
          excl_all);
    if (has_def_shadow && ds_name && ds_ver)
      modelmux_control_route_set_default (self, TRUE, ds_name, ds_ver, ds_gpu, excl, n_excl,
          excl_all);
    g_free (excl);
  }
  /* app-facing completion notice: the reroute has actually been applied now. The
   * whole routing summary rides in the "name" field so the app renders it as one
   * bracketed line: `new streams routed [streams=... primary=... shadow=...]`
   * (streams/route has no single model identity -- name+version would be
   * ambiguous). Includes the pool-default switch when the route moved it.
   *
   * SKIPPED when any route deferred: firing here would announce a switch that
   * has not happened, and the POST already answered 202 to say so. The pending
   * poller posts it once the warmed target is actually applied. */
  {
    /* Summarise ONLY what actually happened. A route that parked on a warming
     * target is left out -- the poller announces it when it lands -- while the
     * routes that DID apply, and a default{} move (which has no event of its
     * own; set_default rides on this summary), still get their notice. A
     * request can mix the two, so this filters per route rather than
     * suppressing the whole notice when any one of them deferred. */
    ModelMuxDecodedRoute *applied = NULL;
    guint n_applied = 0;
    gchar *summary;

    if (nr) {
      applied = g_new0 (ModelMuxDecodedRoute, nr);
      for (i = 0; i < nr; i++)
        if (!dr[i].deferred)
          applied[n_applied++] = dr[i];   /* SHALLOW: read-only for the summary,
                                           * the strings stay owned by dr[] */
    }
    summary = modelmux_control_route_summary (self, applied, n_applied,
        (d && dm_name != NULL), dm_name, dm_ver,
        (d && has_def_shadow && ds_name != NULL), ds_name, ds_ver);
    if (summary && *summary)
      modelmux_post_model_event (GST_ELEMENT (self), "streams-routed", summary,
          "", -1, TRUE, NULL);
    g_free (summary);
    g_free (applied);                     /* shallow -- never free its members */
  }
  goto out;

reject:
  MODELMUX_CONTROL_ERR (self, "stream/route: request REJECTED WHOLE -- nothing was applied "
      "(admission is atomic)");
out:
  g_free (dm_name);
  g_free (dm_ver);
  g_free (ds_name);
  g_free (ds_ver);
  if (dr)
    modelmux_control_decoded_routes_free (dr, nr);
  g_object_unref (parser);
}

/* ================================================================== */
/* /api/v1/model/update -- IN-PLACE checkpoint transition               */
/* ================================================================== */

/* Walk pending_reroutes and move every effective/warm ref that names
 * @from_key onto @to_key. Shared by the model/update identity-follow and the
 * OTA-rollback follow (the same walk in the opposite direction). Caller holds
 * api_control_lock. Returns the number of refs moved. */
static guint
modelmux_control_rename_pending_refs (GstNvModelMux * self, const gchar * from_key,
    const gchar * to_key)
{
  GList *l;
  guint moved = 0;

  for (l = self->pending_reroutes; l; l = l->next) {
    ModelMuxPendingReroute *deferred_reroute = (ModelMuxPendingReroute *) l->data;
    gchar **f[4] = { &deferred_reroute->eff_primary, &deferred_reroute->eff_shadow,
      &deferred_reroute->warm_primary, &deferred_reroute->warm_shadow };
    guint i;
    for (i = 0; i < 4; i++)
      if (*f[i] && g_strcmp0 (*f[i], from_key) == 0) {
        g_free (*f[i]);
        *f[i] = g_strdup (to_key);
        moved++;
      }
  }
  return moved;
}

/* Drop the (name, version) record from the version registry -- the exact
 * inverse of the modelmux_config_register_version done at model/update trigger
 * time. Used by the OTA rollback: a REJECTED engine must not stay recorded
 * under the new version, or status / gpu-addressed routing / a later lazy
 * warm would resolve an artifact that never served. Order-insensitive
 * compaction (every registry reader scans linearly for an exact
 * (name, version) match). Caller holds api_control_lock. */
static void
modelmux_control_registry_unregister_version (GstNvModelMux * self, const gchar * name,
    const gchar * version)
{
  ModelMuxConfig *config = &self->config;
  guint i;

  if (!name || !version)
    return;
  /* compaction frees + moves entries the CIVETWEB status thread may be
   * scanning (modelmux_config_version_gpu) -- serialize on versions_lock. Lock
   * order: api_control_lock (held by the caller) -> versions_lock (see priv.h). */
  g_mutex_lock (&config->versions_lock);
  for (i = 0; i < config->num_versions; i++) {
    ModelVersionArtifact *v = &config->versions[i];
    if (g_strcmp0 (v->name, name) == 0 &&
        g_strcmp0 (v->version, version) == 0) {
      g_free (v->name);
      g_free (v->version);
      g_free (v->engine);
      g_free (v->config);
      config->num_versions--;
      if (i < config->num_versions)
        *v = config->versions[config->num_versions];
      memset (&config->versions[config->num_versions], 0, sizeof (ModelVersionArtifact));
      break;
    }
  }
  g_mutex_unlock (&config->versions_lock);
}

/* ModelMuxBin.ota_rollback_cb: a FAILED nvinfer 'model-updated' confirm
 * rolled the LIVE identity back new_key -> restored key inside the bin. The
 * bin invokes this on the MAIN LOOP after the rename-back, with no bin locks
 * held, so the control-plane follow-ups from trigger time can be undone here:
 *   - pending reroute / warm refs still naming the FAILED new key move back
 *     onto the restored key (mirror of modelmux_control_rename_follow_update), and
 *   - the version registry drops the rejected engine's record (registered
 *     optimistically when the update was triggered).
 * Same context as modelmux_control_event_dispatch (a main-loop callback that ENTERS the
 * control plane), and api_control_lock is never held across a bin call that could
 * re-enter here -- so a plain g_mutex_lock is correct. @from_key is the
 * failed NEW key, @to_key the RESTORED original key. */
static void
modelmux_control_ota_rollback_cb (gpointer owner, const gchar * from_key,
    const gchar * to_key)
{
  GstNvModelMux *self = GST_NVMODELMUX (owner);
  gchar *name = NULL, *ver = NULL;
  guint moved;

  g_mutex_lock (&self->api_control_lock);
  if (!self->modelmux_bin) {              /* torn down between the bin post and now */
    g_mutex_unlock (&self->api_control_lock);
    return;
  }
  moved = modelmux_control_rename_pending_refs (self, from_key, to_key);
  model_key_split (from_key, &name, &ver);
  modelmux_control_registry_unregister_version (self, name, ver);
  g_free (name);
  g_free (ver);
  MODELMUX_CONTROL_INFO (self, "OTA rollback: control-plane refs restored to '%s' "
      "(%u pending ref(s) moved off the rejected '%s'; its registry record "
      "dropped)", to_key, moved, from_key);
  g_mutex_unlock (&self->api_control_lock);
}

/* The in-place update renamed the LIVE identity name@FROM -> name@TO (bin
 * layer: hash key, shard keys, routing-table stream refs). Follow it in the
 * CONTROL plane, under api_control_lock (this runs on the main-loop update handler):
 * pending deferred reroutes still naming the old key, and the configured role
 * default when it IS this same checkpoint (the default is a (name, version)
 * ref -- the version must move with the identity, or every later stream-add
 * would resolve to a key that no longer exists). */
static void
modelmux_control_rename_follow_update (GstNvModelMux * self, const gchar * name,
    const gchar * from_ver, const gchar * to_ver)
{
  gchar *old_key = model_key (name, from_ver);
  gchar *new_key = model_key (name, to_ver);
  const gchar *dv;
  guint moved;

  moved = modelmux_control_rename_pending_refs (self, old_key, new_key);
  if (moved)
    MODELMUX_CONTROL_INFO (self, "model/update: %u pending reroute ref(s) follow the "
        "identity '%s' -> '%s'", moved, old_key, new_key);

  /* the runtime defaults follow the identity rename. gpu = -1: same NAME, so
   * the ref's placement pin is KEPT (mm_default_ref_assign's rename rule). */
  {
    const DefaultModelRef *rp = modelmux_bin_get_default (self->modelmux_bin, FALSE);
    const DefaultModelRef *rs = modelmux_bin_get_default (self->modelmux_bin, TRUE);
    dv = (rp && rp->version) ? rp->version : MM_MODEL_VERSION_DEFAULT;
    if (rp && rp->name && g_strcmp0 (rp->name, name) == 0 &&
        g_strcmp0 (dv, from_ver) == 0)
      modelmux_bin_set_default (self->modelmux_bin, FALSE, name, to_ver, -1);
    dv = (rs && rs->version) ? rs->version : MM_MODEL_VERSION_DEFAULT;
    if (rs && rs->name && g_strcmp0 (rs->name, name) == 0 &&
        g_strcmp0 (dv, from_ver) == 0)
      modelmux_bin_set_default (self->modelmux_bin, TRUE, name, to_ver, -1);
  }

  g_free (old_key);
  g_free (new_key);
}

/* Same network, new weights: hot-swap the engine on the LIVE (name,
 * from_version) instance across all its shards -- no new bin, no reroute, the
 * streams never move. The engine must match the instance's device (TRT
 * engines are device-built): scalar engine_file for a single-instance group, or
 * the engine_files{} map covering EVERY device of a multi-GPU instance group
 * (all-or-nothing: nothing swaps unless every shard's gpu has its engine).
 * The identity MOVES from_version -> version at every touch point (hash key,
 * shard keys, routing-table stream refs, pending reroutes, configured default
 * -- API_DESIGN.md 5.3), so the next update CASes on the NEW version. Each
 * shard confirms via nvinfer's 'model-updated'; a rejected engine rolls the
 * shard's provenance back AND renames the identity back
 * (UPDATE_ENGINE_REJECTED, see infer_engine_update_cb). */
static void
modelmux_control_handle_model_inplace_update (GstNvModelMux * self, const ModelMuxModelPayload * pl)
{
  const ModelCatalogEntry *def;
  const gchar *eng = pl->engine_file;
  gchar *key = NULL;
  gint grp[MM_MAX_MODEL_GPUS];
  guint n_grp = 0;
  gint gpu;
  guint i;
  guint live_batch = 0;   /* the live instance's batch (identity carries it) */

  MODELMUX_CONTROL_INFO (self, "MODEL UPDATE (in-place) name='%s' %s -> %s",
      pl->name ? pl->name : "?",
      pl->from_version ? pl->from_version : "?",
      pl->version ? pl->version : "?");

  if (!pl->name || !pl->from_version || !pl->version) {
    MODELMUX_CONTROL_ERR (self, "model/update rejected: 'name', 'from_version' and 'version' "
        "are required (UPDATE_FROM_REQUIRED)");
    return;
  }
  if (!modelmux_version_is_int (pl->from_version) || !modelmux_version_is_int (pl->version)) {
    MODELMUX_CONTROL_ERR (self, "model/update rejected: versions must be positive integers");
    return;
  }
  if (g_strcmp0 (pl->from_version, pl->version) == 0) {
    MODELMUX_CONTROL_ERR (self, "model/update rejected: 'version' must differ from "
        "'from_version' (identity must move)");
    return;
  }
  def = modelmux_config_find_model (&self->config, pl->name);
  if (!def) {
    MODELMUX_CONTROL_ERR (self, "model/update rejected: '%s' not in the catalog "
        "(MODEL_NOT_LOADED)", pl->name);
    return;
  }
  if (def->type == MODEL_INFERSERVER) {
    MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: nvinferserver checkpoints live "
        "in the Triton model repository (MODEL_UPDATE_UNSUPPORTED_BACKEND) -- "
        "add the new version there, then model/load + stream/route to it",
        pl->name);
    return;
  }
  key = model_key (pl->name, pl->from_version);
  if (!modelmux_bin_model_loaded (self->modelmux_bin, key)) {
    MODELMUX_CONTROL_ERR (self, "model/update rejected: '%s' is not loaded "
        "(UPDATE_FROM_MISMATCH) -- the precondition names a live version", key);
    goto done;
  }
  /* BATCH IMMUTABILITY across the in-place transition: the element (and its slot
   * tables/mux) is KEPT -- a to-version that declares a DIFFERENT batch (payload,
   * or a [model-<name>-<version>] block / earlier load of that version) cannot be
   * honoured in place. Reject with the recovery path instead of renaming the
   * identity onto an instance whose capacity silently disagrees with its registry. */
  live_batch = modelmux_bin_model_batch (self->modelmux_bin, key);
  {
    guint to_b = pl->batch_size ? pl->batch_size :
        modelmux_config_version_batch (&self->config, pl->name, pl->version);
    if (live_batch && to_b && to_b != live_batch) {
      MODELMUX_CONTROL_ERR (self, "model/update '%s' -> '%s' rejected: version batch %u "
          "differs from the live instance's %u (UPDATE_BATCH_MISMATCH) -- an in-place "
          "checkpoint swap keeps the element and cannot resize it; model/load '%s@%s' "
          "as a new version and stream/route to it instead",
          key, pl->version, to_b, live_batch, pl->name, pl->version);
      goto done;
    }
  }

  /* the LIVE group's device set: every shard of the (name, from_version) chain,
   * all GPUs. >1 device = a multi-GPU instance group -> per-device engines. */
  n_grp = modelmux_bin_model_gpus (self->modelmux_bin, key, grp, G_N_ELEMENTS (grp));
  gpu = modelmux_config_version_gpu (&self->config, pl->name, pl->from_version);

  /* ===== CASE 1: UPDATE A MULTI-SHARD (multi-GPU instance) MODEL =====
   * The live (name, from_version) runs one shard per GPU. Swap the engine on
   * EVERY shard together (all-or-nothing) using a per-device engine_files{} map.
   * This block ends in `goto done`, so the single-instance path below is reached
   * only when n_grp <= 1. */
  if (n_grp > 1) {
    /* MULTI-GPU GROUP: engine_files{} must cover the group EXACTLY -- every
     * device mapped (GPU_ENGINE_MISSING) and no others (GPU_UNKNOWN); a scalar
     * engine_file cannot address a group (TRT engines are device-built). The
     * whole request is admitted or rejected BEFORE anything swaps. */
    const gchar *grp_engs[MM_MAX_MODEL_GPUS];
    guint j;
    if (eng) {
      MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: scalar engine_file is valid "
          "only for a single-instance group -- this group spans %u gpus; use "
          "engine_files{} with one engine per device (GPU_ENGINE_MISSING)",
          key, n_grp);
      goto done;
    }
    if (!pl->n_engines) {
      MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: a prebuilt engine per device "
          "is required (engine_files{}) -- model/update never auto-builds "
          "(GPU_ENGINE_MISSING)", key);
      goto done;
    }
    for (i = 0; i < n_grp; i++) {
      grp_engs[i] = NULL;
      for (j = 0; j < pl->n_engines; j++)
        if (pl->engines[j].gpu == grp[i])
          grp_engs[i] = pl->engines[j].path;
      if (!grp_engs[i]) {
        MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: engine_files{} has no "
            "engine for the group's gpu %d (GPU_ENGINE_MISSING) -- nothing was "
            "swapped (the group transitions together or not at all)",
            key, grp[i]);
        goto done;
      }
    }
    for (j = 0; j < pl->n_engines; j++) {
      gboolean in_grp = FALSE;
      for (i = 0; i < n_grp && !in_grp; i++)
        in_grp = (pl->engines[j].gpu == grp[i]);
      if (!in_grp) {
        MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: engine_files{} names gpu %d "
            "but the group has no instance there (GPU_UNKNOWN) -- gpu ids are "
            "file assignments, never an instance selector", key,
            pl->engines[j].gpu);
        goto done;
      }
    }
    for (i = 0; i < n_grp; i++)
      if (!g_file_test (grp_engs[i], G_FILE_TEST_IS_REGULAR)) {
        MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: engine for gpu %d not "
            "found: '%s'", key, grp[i], grp_engs[i]);
        goto done;
      }
    if (modelmux_bin_reload_model_multi (self->modelmux_bin, key, grp, grp_engs,
            n_grp, pl->version)) {
      /* record the NEW version's artifacts on the SAME placements (the base
       * device's engine seeds the registry; the live set stays queryable). */
      /* the identity moved onto the SAME instance: its batch is unchanged */
      modelmux_config_register_version (&self->config, pl->name, pl->version,
          grp_engs[0], NULL, gpu >= 0 ? gpu : grp[0], live_batch);
      /* the live identity moved with the group (bin rename) -- follow it in
       * the control plane (pending reroutes, configured default). */
      modelmux_control_rename_follow_update (self, pl->name, pl->from_version, pl->version);
      modelmux_bin_bump_routing_revision (self->modelmux_bin);
      MODELMUX_CONTROL_INFO (self, "model/update '%s' -> version %s: per-device engine "
          "hot-swap in flight on %u instance(s) (zero-drop; streams never "
          "move)", key, pl->version, n_grp);
    } else {
      MODELMUX_CONTROL_ERR (self, "model/update '%s' failed: no live bin to swap", key);
    }
    goto done;
  }

  /* ===== CASE 2: UPDATE A SINGLE-INSTANCE MODEL (n_grp <= 1) =====
   * The live model runs on one device. Swap its engine in place with a scalar
   * engine_file (or the one engine_files{} entry matching its device). */
  /* ENGINE for the instance's recorded device -- never auto-built. */
  if (!eng && pl->n_engines) {
    gint want = gpu >= 0 ? gpu : 0;
    for (i = 0; i < pl->n_engines; i++)
      if (pl->engines[i].gpu == want)
        eng = pl->engines[i].path;
    if (!eng) {
      MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: engine_files{} has no engine "
          "for its device gpu %d (GPU_ENGINE_MISSING)", key, want);
      goto done;
    }
    for (i = 0; i < pl->n_engines; i++)
      if (pl->engines[i].gpu != want)
        MODELMUX_CONTROL_WARN (self, "model/update '%s': engine for gpu %d unused (the "
            "instance runs on gpu %d)", key, pl->engines[i].gpu, want);
  }
  if (!eng) {
    MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: a prebuilt engine is required "
        "(engine_file or engine_files{}) -- model/update never auto-builds", key);
    goto done;
  }
  if (!g_file_test (eng, G_FILE_TEST_IS_REGULAR)) {
    MODELMUX_CONTROL_ERR (self, "model/update '%s' rejected: engine not found: '%s'", key, eng);
    goto done;
  }

  if (modelmux_bin_reload_model (self->modelmux_bin, key, NULL, eng, pl->version)) {
    /* record the NEW version's artifact on the SAME placement so status,
     * later updates and gpu-addressed routing resolve it. */
    modelmux_config_register_version (&self->config, pl->name, pl->version, eng,
        NULL, gpu, live_batch);   /* same instance -> same batch */
    /* the live identity moved with the bin (rename) -- follow it in the
     * control plane (pending reroutes, configured default). */
    modelmux_control_rename_follow_update (self, pl->name, pl->from_version, pl->version);
    /* routing CONTENT changed (the version identity moved) even though no
     * stream moved -- controllers watching if_revision must see it. */
    modelmux_bin_bump_routing_revision (self->modelmux_bin);
    MODELMUX_CONTROL_INFO (self, "model/update '%s' -> version %s: engine hot-swap in "
        "flight (zero-drop; streams never move)", key, pl->version);
  } else {
    MODELMUX_CONTROL_ERR (self, "model/update '%s' failed: no live bin to swap", key);
  }
done:
  g_free (key);
}

/* In-band per-stream model binding (stream/add metadata, delivered as a
 * serialized downstream event so it works even when the host owns the bus).
 * Stores the binding in the DEDICATED requested_binding_info map (never source_attach_state, which
 * would suppress the attach) so the PAD_ADDED/buffer attach can apply it; if the
 * source is ALREADY attached (the attach raced ahead on the defaults), it
 * reconciles by rerouting that one stream to the bound model(s) -- zero-drop,
 * reusing the scoped-reroute primitive. */
static void
modelmux_control_handle_stream_model_bind (GstNvModelMux * self, const NvDsSensorInfo * sensor_info)
{
  gboolean attached = FALSE;
  gchar *primary_model_name = NULL, *shadow_model_name = NULL;
  gchar *primary_model_version = NULL, *shadow_model_version = NULL;
  gint primary_model_gpu = -1, shadow_model_gpu = -1;
  const gchar *def_primary = NULL, *def_shadow = NULL, *def_pv = NULL, *def_sv = NULL;
  const gchar *glob_primary = NULL, *glob_shadow = NULL, *glob_pv = NULL, *glob_sv = NULL;
  gchar *req_pkey = NULL, *req_skey = NULL, *def_pkey = NULL, *def_skey = NULL;
  gchar *glob_pkey = NULL, *glob_skey = NULL;
  const gchar *primary, *shadow;
  gboolean shadow_cleared = FALSE;
  gboolean p_from_def = TRUE, s_from_def = TRUE;   /* resolver origin (default vs binding) */
  gboolean p_via = FALSE, s_via = FALSE;           /* final per-lane origin flags */

  /** TWO MAPS, TWO JOBS -- both keyed by source_id, kept deliberately SEPARATE so caching
   *  the pick can never disturb the live wiring state:
   *    requested_binding_info = WHAT this stream WANTS (the model pick / intent). Written
   *                             HERE (block 1); read later by the PAD_ADDED attach path.
   *    source_attach_state    = WHETHER this stream is already WIRED (the live state).
   *                             Created + marked attached by the PAD_ADDED / attach path;
   *                             read HERE (block 2) only to learn if the attach happened yet.
   */
  g_mutex_lock (&self->source_state_lock);
  {
    /**  BLOCK 1 -- CACHE the pick: deep-copy {sensor_id, name, metadata} into
    *  requested_binding_info (REPLACING any prior entry for this source_id). */
    if (self->requested_binding_info) {
      ModelMuxRequestedBindingInfo *requested_binding_info = g_new0 (ModelMuxRequestedBindingInfo, 1);
      requested_binding_info->sensor_id = g_strdup (sensor_info->sensor_id);
      requested_binding_info->sensor_name = g_strdup (sensor_info->sensor_name);
      requested_binding_info->sensor_metadata = g_strdup (sensor_info->sensor_metadata);
      g_hash_table_insert (self->requested_binding_info,
          GUINT_TO_POINTER (sensor_info->source_id), requested_binding_info);
    }
    /** BLOCK 2 -- has the ATTACH already happened for this source? `attached` answers the
     *  TIMING question that decides everything below (the two events race):
     *    attached == FALSE  (BIND before ADD, the common case) -> just cache & return; the
     *                        upcoming attach reads the cache and brings the stream up.
     *    attached == TRUE   (BIND after ADD: the stream is ALREADY live -- e.g. wired on the
     *                        default because PAD_ADDED carries no metadata) -> this is a
     *                        RE-BIND; fall through to REROUTE the live stream, zero-drop. */
    if (self->source_attach_state) {
      ModelMuxSourceAttachState *source_attach_state = g_hash_table_lookup (self->source_attach_state,
          GUINT_TO_POINTER (sensor_info->source_id));
      attached = (source_attach_state && source_attach_state->attached);
    }
  }
  g_mutex_unlock (&self->source_state_lock);

  MODELMUX_CONTROL_INFO (self, "STREAM BIND source=%u camera='%s' (%s)", sensor_info->source_id,
      sensor_info->sensor_id ? sensor_info->sensor_id : "?",
      attached ? "already attached -> reconcile" : "cached for attach");

  /** NOT attached yet (common case: BIND arrived before the ADD). Nothing more to do --
   * the binding is already cached, and the upcoming attach will read it and bring the
   * stream up on the requested model. Only an ALREADY-attached stream falls through
   * below to be actively rerouted (relinked) to the bound model. */
  if (!attached)
    return;

  /* ==========================================================================
   * RE-BIND OF A LIVE STREAM  (attached == TRUE)  --  THE ACTUAL REROUTE PATH.
   * The stream is already wired (usually on the default) and this new BIND wants a
   * DIFFERENT model. Below we relink it LIVE and ZERO-DROP: it keeps inferring on its
   * current model until the new target is ready. Three outcomes at the end:
   * reroute NOW / DEFER until warm / SKIP.
   * ========================================================================== */

  /* parse the bind's metadata model override (name/version/gpu per role). */
  modelmux_control_parse_model_metadata (self, sensor_info->sensor_metadata, &primary_model_name, &primary_model_version,
      &primary_model_gpu, &shadow_model_name, &shadow_model_version, &shadow_model_gpu);

  /* resolve the TWO operands the reroute gate compares:
   *   def_*  = WHERE IT SHOULD BE  -- this camera's target (per-sensor binding,
   *            else runtime default; p_from_def/s_from_def = default vs pin).
   *   glob_* = WHERE IT IS NOW     -- the GLOBAL defaults it attached on
   *            (camera_id = NULL, no per-sensor binding). */
  modelmux_config_resolve_models (&self->config,
      modelmux_bin_get_default (self->modelmux_bin, FALSE),
      modelmux_bin_get_default (self->modelmux_bin, TRUE),
      sensor_info->sensor_id, &def_primary, &def_shadow, &def_pv, &def_sv,
      &p_from_def, &s_from_def);
  modelmux_config_resolve_models (&self->config,
      modelmux_bin_get_default (self->modelmux_bin, FALSE),
      modelmux_bin_get_default (self->modelmux_bin, TRUE),
      NULL, &glob_primary, &glob_shadow, &glob_pv, &glob_sv, NULL, NULL);

  /* "none" shadow-clear token: this stream runs PRIMARY ONLY (opts out of the default shadow). */
  shadow_cleared = (shadow_model_name && g_strcmp0 (shadow_model_name, "none") == 0);

  /* compose canonical "name@version" keys. STRICT: a requested model without a version is not
   * resolvable -> falls back to the role default (we do not assume version 1). */
  if (primary_model_name && primary_model_version) req_pkey = model_key (primary_model_name, primary_model_version);
  if (!shadow_cleared && shadow_model_name && shadow_model_version) req_skey = model_key (shadow_model_name, shadow_model_version);
  if (def_primary) def_pkey = model_key (def_primary, def_pv);
  if (def_shadow)  def_skey = model_key (def_shadow, def_sv);
  if (glob_primary) glob_pkey = model_key (glob_primary, glob_pv);
  if (glob_shadow)  glob_skey = model_key (glob_shadow, glob_sv);

  /* gpu-addressed metadata refs must match the version's recorded placement */
  if (req_pkey && !modelmux_control_check_ref_gpu (self, primary_model_name, primary_model_version, primary_model_gpu,
          sensor_info->source_id, "primary")) {
    g_free (req_pkey);
    req_pkey = NULL;
  }
  if (req_skey && !modelmux_control_check_ref_gpu (self, shadow_model_name, shadow_model_version, shadow_model_gpu,
          sensor_info->source_id, "shadow")) {
    g_free (req_skey);
    req_skey = NULL;
  }

  primary = modelmux_control_resolve_role (self, FALSE, req_pkey, def_pkey, "primary");
  /* shadow_cleared ('none') forces PRIMARY ONLY -- skip the default-shadow fallback. */
  shadow = shadow_cleared ? NULL : modelmux_control_resolve_role (self, TRUE, req_skey, def_skey, "shadow");

  /* ORIGIN per lane (via_default): did each lane land on the DEFAULT
   * (pointer identity + resolver origin) or was it EXPLICITLY pinned? Same rule as the
   * attach path -- see modelmux_control_handle_stream_add. */
  p_via = (primary != NULL && primary == def_pkey && p_from_def);
  s_via = (shadow != NULL && shadow == def_skey && s_from_def);

  /* GATE: is a reroute needed at all? YES when the resolved primary/shadow
   * DIFFERS from the current globals, OR the lane is EXPLICITLY pinned (!p_via / !s_via)
   * even if it is value-equal to the default. */
  if (primary && (g_strcmp0 (primary, glob_pkey) != 0 ||
          g_strcmp0 (shadow, glob_skey) != 0 ||
          !p_via || (shadow && !s_via))) {
    /* why the !p_via / (shadow && !s_via) clauses: a model VALUE-equal to the global
     * default but EXPLICITLY named (metadata ref / per-camera binding) makes the reroute a
     * same-model NO-OP -- but it must STILL run so the PIN lands on the lane
     * (update_routing's bookkeeping-only commit). Skipping would leave via_default=TRUE and
     * a later default switch would sweep a stream the operator explicitly bound. */
    guint src[1];
    gint pst, sst;
    /* effective GPU placement (same rule as the attach path,
     * modelmux_control_eff_gpu): explicit ref gpu, else the GLOBAL default's persisted pin,
     * else any instance. */
    gint eff_pgpu = modelmux_control_eff_gpu (self, FALSE, primary, req_pkey, primary_model_gpu);
    gint eff_sgpu = modelmux_control_eff_gpu (self, TRUE, shadow, req_skey, shadow_model_gpu);
    src[0] = sensor_info->source_id;

    /* LAZY WARM: warm a not-yet-loaded bound model before rerouting to it (into
     * limbo, per-version artifact). The stream keeps inferring on its current model
     * meanwhile -- nothing is torn down yet (zero-drop). */
    modelmux_control_lazy_warm (self, FALSE, primary);
    if (shadow) modelmux_control_lazy_warm (self, TRUE, shadow);
    pst = modelmux_bin_model_status (self->modelmux_bin, FALSE, primary);
    sst = shadow ? modelmux_bin_model_status (self->modelmux_bin, TRUE, shadow) : 1;
    /* drop a shadow that can neither serve nor warm (failed / not in the catalog)
     * -> primary only. */
    if (shadow && sst != MODEL_WARMED && sst != MODEL_SERVING && sst != MODEL_WARMING) {
      MODELMUX_CONTROL_WARN (self, "stream %u rebind: shadow '%s' not usable (status=%s) -> primary only",
          sensor_info->source_id, shadow, sst < 0 ? "NotLoaded" : model_status_str (sst));
      shadow = NULL;
      sst = 1;
    }

    MODELMUX_CONTROL_INFO (self, "stream %u rebind -> primary='%s' shadow='%s' (was default)",
        sensor_info->source_id, primary, shadow ? shadow : "-");

    /* THE REROUTE DECISION (three outcomes):
     *   (a) target(s) READY   -> relink NOW   (modelmux_control_apply_scoped_reroute)
     *   (b) target(s) WARMING -> DEFER        (queue ModelMuxPendingReroute + poller)
     *   (c) primary UNUSABLE  -> SKIP         (stay on the current model) */

    /* (a) BOTH READY -> RELINK NOW.  apply_scoped_reroute() is THE ACTUAL LIVE SWITCH: it
     *     moves this one stream off its current model onto the new primary (+shadow),
     *     zero-drop. "shadow":null clears the running shadow (PRIMARY ONLY), not "leave
     *     unchanged". */
    if ((pst == MODEL_WARMED || pst == MODEL_SERVING) &&
        (!shadow || sst == MODEL_WARMED || sst == MODEL_SERVING)) {
      modelmux_control_apply_scoped_reroute (self, src, 1, primary, shadow, shadow_cleared,
          eff_pgpu, eff_sgpu, p_via, s_via, NULL);
    /* (b) TARGET(S) STILL WARMING -> DEFER the reroute until READY (still zero-drop): the
     *     stream keeps running its current model; the pending-reroute poller applies the
     *     SAME scoped reroute (modelmux_control_apply_scoped_reroute) once the target warms. */
    } else if (pst == MODEL_WARMING || (shadow && sst == MODEL_WARMING)) {
      ModelMuxPendingReroute *deferred_reroute = g_new0 (ModelMuxPendingReroute, 1);
      /* FRESHEST intent wins here too. A model bind can land on an ALREADY-LIVE
       * stream (see the STREAM_MODEL_BIND dispatch comment), so this stream may
       * already have a deferred route queued from stream/route. Appending beside
       * it would let WARM ORDER pick the winner: whichever target warms last
       * overwrites the other, so a rebind issued after a route could still lose to
       * it. Withdraw this stream from any older queued entry first, per lane --
       * the same rule modelmux_control_exec_route applies for explicit routes.
       *
       * enqueue_shadow_join deliberately does NOT need this: it only runs while a
       * stream is being ADDED, and pending_drop_source has already withdrawn that
       * source_id from every queued op when the previous incarnation left, so
       * there is never a stale entry to supersede. */
      modelmux_control_pending_supersede (self, src, 1, primary != NULL,
          (shadow != NULL || shadow_cleared), FALSE /* explicit bind: claims any lane */);
      deferred_reroute->src = g_memdup2 (src, sizeof (guint));
      deferred_reroute->n = 1;
      deferred_reroute->eff_primary = g_strdup (primary);
      deferred_reroute->eff_shadow = (shadow && *shadow) ? g_strdup (shadow) : NULL;
      deferred_reroute->warm_primary = (pst == MODEL_WARMING) ? g_strdup (primary) : NULL;
      deferred_reroute->warm_shadow  = (shadow && sst == MODEL_WARMING) ? g_strdup (shadow) : NULL;
      deferred_reroute->p_gpu = eff_pgpu;
      deferred_reroute->s_gpu = eff_sgpu;
      deferred_reroute->clear_shadow = shadow_cleared;     /* "shadow":null rides the deferred apply too */
      deferred_reroute->p_via_default = p_via;
      deferred_reroute->s_via_default = s_via;
      self->pending_reroutes = g_list_append (self->pending_reroutes, deferred_reroute);
      modelmux_control_publish_pending (self);   /* make the accepted route visible NOW */
      if (!self->pending_poll_id)
        modelmux_control_arm_pending_poll (self, 200);
      MODELMUX_CONTROL_INFO (self, "stream %u rebind deferred -> applies once target(s) WARMED",
          sensor_info->source_id);
    } else {
      /* (c) PRIMARY UNUSABLE (failed / not-loaded) -> SKIP: keep the stream on its current
       *     model, never black out a live stream. */
      MODELMUX_CONTROL_WARN (self, "stream %u rebind skipped: primary '%s' not usable (status=%s) -> staying "
          "on the current default", sensor_info->source_id, primary,
          pst < 0 ? "NotLoaded" : model_status_str (pst));
    }
  }

  g_free (primary_model_name);
  g_free (shadow_model_name);
  g_free (primary_model_version);
  g_free (shadow_model_version);
  g_free (req_pkey);
  g_free (req_skey);
  g_free (def_pkey);
  g_free (def_skey);
  g_free (glob_pkey);
  g_free (glob_skey);
}

/* ================================================================== */
/* Bus delivery: parse on the post thread, execute on the main loop   */
/* ================================================================== */
typedef enum
{
  MODELMUX_CONTROL_STREAM_ADD,
  MODELMUX_CONTROL_STREAM_REMOVE,
  MODELMUX_CONTROL_STREAM_MODEL_BIND,
  MODELMUX_CONTROL_MODEL_LOAD,
  MODELMUX_CONTROL_MODEL_UNLOAD,
  MODELMUX_CONTROL_MODEL_UPDATE,               /* IN-PLACE checkpoint transition */
  MODELMUX_CONTROL_STREAM_ROUTE                /* declarative stream-routing request */
} ModelMuxControlOperation;

/* A deep-copied control event marshalled onto the main loop. Holds a ref on the
 * element so the closure stays valid even if teardown races the dispatch. */
typedef struct
{
  GstNvModelMux *self;
  ModelMuxControlOperation operation;
  guint    source_id;
  gchar   *sensor_id;
  gchar   *sensor_name;
  gchar   *sensor_metadata;
  gchar   *value_json;              /* model/route ops: the verbatim "value" document */
} ModelMuxControlEvent;

/* Free a control event structure. */
static void
modelmux_control_event_free (gpointer data)
{
  ModelMuxControlEvent *event = (ModelMuxControlEvent *) data;
  if (!event)
    return;
  g_free (event->sensor_id);
  g_free (event->sensor_name);
  g_free (event->sensor_metadata);
  g_free (event->value_json);
  if (event->self)
    gst_object_unref (event->self);
  g_free (event);
}

/* Record a SUCCESSFUL attach for a source: set attached=TRUE (which tells the sink
 * buffer probe to stop dropping this source's buffers), stamp last_seen for staleness
 * detection, and clear the fail/retry backoff.
 *
 * Creates the source_attach_state entry if it does not exist yet. This matters because
 * the same logical stream can be announced through more than one control path (bus
 * stream-add, in-band PAD_ADDED, and the sink buffer fallback): whichever path attaches
 * first seeds the entry here, so all paths converge on one attached lifecycle state
 * instead of a later path creating a stale attached=FALSE entry that would gate (drop)
 * every batch forever. */
static void
modelmux_control_mark_sensor_attached (GstNvModelMux * self, guint source_id)
{
  ModelMuxSourceAttachState *source_attach_state;
  gpointer key = GUINT_TO_POINTER (source_id);

  g_mutex_lock (&self->source_state_lock);
  {
    if (!self->source_attach_state) {
      g_mutex_unlock (&self->source_state_lock);
      return;
    }

    source_attach_state = g_hash_table_lookup (self->source_attach_state, key);
    if (!source_attach_state) {
      source_attach_state = g_new0 (ModelMuxSourceAttachState, 1);
      g_hash_table_insert (self->source_attach_state, key, source_attach_state);
    }
    source_attach_state->attached = TRUE;
    source_attach_state->last_seen = self->batch_idx;
    source_attach_state->fails = 0;
    source_attach_state->retry_at = 0;
  }
  g_mutex_unlock (&self->source_state_lock);
}

/* Record a FAILED attach for a source (bin rejected the stream: id out of range,
 * capacity, attach-in-progress collision, ...). Leaves attached=FALSE (so the sink
 * probe keeps dropping this source's buffers) and schedules a bounded-backoff RETRY
 * by setting retry_at: without it the entry would stay un-attached forever and the
 * probe would gate (drop) every batch for the pipeline's lifetime. Only an existing,
 * not-yet-attached entry is touched (a successful attach already cleared this state). */
static void
modelmux_control_mark_sensor_attach_failed (GstNvModelMux * self, guint source_id)
{
  ModelMuxSourceAttachState *source_attach_state;

  g_mutex_lock (&self->source_state_lock);
  {
    source_attach_state = self->source_attach_state ?
        g_hash_table_lookup (self->source_attach_state, GUINT_TO_POINTER (source_id)) : NULL;
    if (source_attach_state && !source_attach_state->attached) {
      if (source_attach_state->fails < G_MAXUINT)
        source_attach_state->fails++;
      if (source_id >= self->config.batch_size) {
        /* deterministic: no retry can ever succeed -- teach the config fix, once */
        if (!source_attach_state->rejected)
          MODELMUX_CONTROL_ERR (self, "stream %u attach PERMANENTLY REJECTED: source_id >= "
              "modelmux batch-size (%u) -- upstream is batching more sources than this "
              "element accepts. Batches containing it are DROPPED; raise 'batch-size' "
              "(or lower the upstream mux's max-batch-size) to recover. Retries stopped.",
              source_id, self->config.batch_size);
        source_attach_state->rejected = TRUE;
        source_attach_state->retry_at = 0;   /* 0 = no retry due (see the struct doc) */
      } else {
        source_attach_state->retry_at = self->batch_idx +
            ((guint64) MM_AUTO_STALE << MIN (source_attach_state->fails, MM_AUTO_FAIL_SHIFT_MAX));
        MODELMUX_CONTROL_ERR (self, "stream %u attach FAILED %u time(s) -- batches containing it "
            "are DROPPED until it attaches; next retry in ~%u batches",
            source_id, source_attach_state->fails,
            (guint) (MM_AUTO_STALE << MIN (source_attach_state->fails, MM_AUTO_FAIL_SHIFT_MAX)));
      }
    }
  }
  g_mutex_unlock (&self->source_state_lock);
}

/* CENTRAL control-plane dispatcher: the single entry point that handles and routes
 * every control API received from upstream (nvmultiurisrcbin bus / action signals)
 * to its matching handler. Covers the model plane -- model/load, model/unload,
 * model/update -- and the stream plane -- stream add, stream model-bind, stream
 * remove and stream/route. Runs on the application main loop (marshalled here via
 * g_idle_add_full by the posting thread), reconstructs the event's info from its
 * deep-copied strings, then calls the right modelmux_control_handle_* routine under
 * api_control_lock (so no handler ever touches self->modelmux_bin while teardown
 * frees it). One-shot: always returns G_SOURCE_REMOVE. */
static gboolean
modelmux_control_event_dispatch (gpointer data)
{
  ModelMuxControlEvent *event = (ModelMuxControlEvent *) data;
  GstNvModelMux *self = event->self;

  /** api_control_lock across the whole dispatch: the plain `!self->modelmux_bin` check alone is a
   * TOCTOU against gst_modelmux_teardown (any thread) freeing the bin while a handler
   * is mid-surgery on it. Teardown swaps/frees self->modelmux_bin under the same lock. */
  g_mutex_lock (&self->api_control_lock);
  if (!self->modelmux_bin) {                        /* torn down between post and dispatch */
    g_mutex_unlock (&self->api_control_lock);
    return G_SOURCE_REMOVE;
  }

  switch (event->operation) {
    /** STREAM_ADD: wire a newly-added source into the inference graph, applying
     * its per-stream model selection (from the event, else the cached bind, else
     * defaults), then open/retry its buffer gate based on the attach result. */
    case MODELMUX_CONTROL_STREAM_ADD:{
      gchar *cached_sensor_id = NULL, *cached_sensor_name = NULL, *cached_sensor_metadata = NULL;
      g_mutex_lock (&self->source_state_lock);
      {
        /** If a bind for this source is cached in requested_binding_info, copy its
         * sensor id/name/metadata into the cached_* locals for use at attach time. */
        if (self->requested_binding_info) {
          ModelMuxRequestedBindingInfo *req_info = g_hash_table_lookup (self->requested_binding_info,
              GUINT_TO_POINTER (event->source_id));
          if (req_info) {
            cached_sensor_id = g_strdup (req_info->sensor_id);
            cached_sensor_name = g_strdup (req_info->sensor_name);
            cached_sensor_metadata = g_strdup (req_info->sensor_metadata);
            req_info->consumed = TRUE;   /* this incarnation's remove may now clear it */
          }
        }
      }
      g_mutex_unlock (&self->source_state_lock);
      /** Prefer the event's own sensor fields (the REST bus-add path carries them);
       * fall back to the cached in-band bind (the PAD_ADDED path has only source_id);
       * NULL on both -> handle_stream_add attaches on the global defaults. */
      NvDsSensorInfo sensor_info = { event->source_id, NULL,
        event->sensor_id ? event->sensor_id : cached_sensor_id,
        event->sensor_name ? event->sensor_name : cached_sensor_name,
        event->sensor_metadata ? event->sensor_metadata : cached_sensor_metadata };
      /** Perform the dynamic link operation where we attach the sensor to the inference graph.
       * If the operation is successful, mark the sensor as attached.
       * If the operation is not successful, mark the sensor as failed. */
      gboolean dynamic_link_operation_successful = modelmux_control_handle_stream_add (self, &sensor_info);
      if (dynamic_link_operation_successful)
        modelmux_control_mark_sensor_attached (self, event->source_id);
      else
        modelmux_control_mark_sensor_attach_failed (self, event->source_id);
      /** Free the cached sensor info. */
      g_free (cached_sensor_id);
      g_free (cached_sensor_name);
      g_free (cached_sensor_metadata);
      break;
    }
    /** STREAM_BIND: record/update this source's requested model+metadata in
     * requested_binding_info (does NOT attach). If the source is already live on
     * defaults, handle_stream_bind reroutes it to the requested model. */
    case MODELMUX_CONTROL_STREAM_MODEL_BIND:{
      NvDsSensorInfo sensor_info = { event->source_id, NULL, event->sensor_id, event->sensor_name,
        event->sensor_metadata };
      modelmux_control_handle_stream_model_bind (self, &sensor_info);
      break;
    }
    case MODELMUX_CONTROL_STREAM_REMOVE:{
      NvDsSensorInfo sensor_info = { event->source_id, NULL, event->sensor_id, event->sensor_name, NULL };
      modelmux_control_handle_stream_remove (self, &sensor_info);
      g_mutex_lock (&self->source_state_lock);
      if (self->source_attach_state)
        g_hash_table_remove (self->source_attach_state, GUINT_TO_POINTER (event->source_id));
      if (self->requested_binding_info) {
        ModelMuxRequestedBindingInfo *req_info = g_hash_table_lookup (self->requested_binding_info,
            GUINT_TO_POINTER (event->source_id));
        /* clear only a CONSUMED bind (this incarnation's own). An un-consumed
         * entry belongs to the NEXT incarnation of a recycled source id whose
         * BIND overtook this remove -- it must survive for the imminent ADD
         * (see ModelMuxRequestedBindingInfo.consumed). */
        if (!req_info || req_info->consumed)
          g_hash_table_remove (self->requested_binding_info, GUINT_TO_POINTER (event->source_id));
      }
      g_mutex_unlock (&self->source_state_lock);
      break;
    }
    case MODELMUX_CONTROL_MODEL_LOAD:
    case MODELMUX_CONTROL_MODEL_UNLOAD:
    case MODELMUX_CONTROL_MODEL_UPDATE:{
      const gchar *what = event->operation == MODELMUX_CONTROL_MODEL_LOAD ? "model/load" :
          event->operation == MODELMUX_CONTROL_MODEL_UNLOAD ? "model/unload" : "model/update";
      ModelMuxModelPayload pl;
      if (modelmux_control_decode_model_payload (self, what, event->value_json, &pl)) {
        if (event->operation == MODELMUX_CONTROL_MODEL_LOAD)
          modelmux_control_handle_model_load (self, &pl);
        else if (event->operation == MODELMUX_CONTROL_MODEL_UNLOAD)
          modelmux_control_handle_model_unload (self, &pl, NULL /* async: log only */);
        else
          modelmux_control_handle_model_inplace_update (self, &pl);
        modelmux_control_model_payload_clear (&pl);
      }
      break;
    }
    case MODELMUX_CONTROL_STREAM_ROUTE:
      modelmux_control_handle_stream_route (self, event->value_json,
          FALSE /* async event: validate AND apply */);
      break;
  }
  g_mutex_unlock (&self->api_control_lock);
  return G_SOURCE_REMOVE;
}

/* Build a sensor control event (STREAM_ADD/REMOVE) from an nvmultiurisrcbin
 * sensor-info payload and marshal it to the main loop. Runs on the posting
 * thread, so the sensor strings are deep-copied to outlive the message. */
static void
modelmux_control_post_sensor_event (GstNvModelMux * self, ModelMuxControlOperation operation,
    const NvDsSensorInfo * sensor_info)
{
  ModelMuxControlEvent *event = g_new0 (ModelMuxControlEvent, 1);
  event->self = gst_object_ref (self);
  event->operation = operation;
  event->source_id = sensor_info->source_id;
  event->sensor_id = g_strdup (sensor_info->sensor_id);
  event->sensor_name = g_strdup (sensor_info->sensor_name);
  event->sensor_metadata = g_strdup (sensor_info->sensor_metadata);
  /** Schedule the event dispatch on the main loop. */
  g_idle_add_full (G_PRIORITY_DEFAULT, modelmux_control_event_dispatch, event, modelmux_control_event_free);
}

static void
modelmux_control_post_ctrl_json (GstNvModelMux * self, ModelMuxControlOperation operation, const gchar * value_json)
{
  ModelMuxControlEvent *e;

  if (!value_json || !*value_json) {
    MODELMUX_CONTROL_WARN (self, "control op %d dropped: empty JSON payload", (gint) operation);
    return;
  }
  e = g_new0 (ModelMuxControlEvent, 1);
  e->self = gst_object_ref (self);
  e->operation = operation;
  e->value_json = g_strdup (value_json);
  g_idle_add_full (G_PRIORITY_DEFAULT, modelmux_control_event_dispatch, e, modelmux_control_event_free);
}

/* "sync-message" handler (runs on the REST/posting thread). Identifies the
 * custom nvmultiurisrcbin messages, deep-copies the payload and marshals the
 * operation to the main loop. Non-matching messages are ignored. */
static void
modelmux_control_bus_sync_message (GstBus * bus, GstMessage * msg, gpointer user_data)
{
  GstNvModelMux *self = GST_NVMODELMUX (user_data);

  if (gst_nvmessage_is_stream_add (msg)) {
    NvDsSensorInfo i = { 0 };
    if (gst_nvmessage_parse_stream_add (msg, &i))
      modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_ADD, &i);
  } else if (gst_nvmessage_is_stream_remove (msg)) {
    NvDsSensorInfo i = { 0 };
    if (gst_nvmessage_parse_stream_remove (msg, &i))
      modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_REMOVE, &i);
  } else if (gst_nvmessage_is_model_load (msg)) {
    NvDsModelInfo m = { 0 };
    if (gst_nvmessage_parse_model_load (msg, &m))
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_MODEL_LOAD, m.value_json);
  } else if (gst_nvmessage_is_model_unload (msg)) {
    NvDsModelInfo m = { 0 };
    if (gst_nvmessage_parse_model_unload (msg, &m))
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_MODEL_UNLOAD, m.value_json);
  } else if (gst_nvmessage_is_model_update (msg)) {
    NvDsModelInfo m = { 0 };
    if (gst_nvmessage_parse_model_update (msg, &m))
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_MODEL_UPDATE, m.value_json);
  } else if (gst_nvmessage_is_stream_route (msg)) {
    NvDsRouteInfo r = { 0 };
    if (gst_nvmessage_parse_stream_route (msg, &r))
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_STREAM_ROUTE, r.value_json);
  }
}

/* ---- synchronous ADMISSION (read-only) for the query path ----
 * Decide accept/reject for model/load and model/update WITHOUT mutating,
 * reusing the SAME read-only helpers the async handlers use (no divergent
 * logic). On accept the caller schedules the real async warm/swap; on reject
 * nothing runs. The request-shape errors (bad name/version/paths, gpu coverage)
 * are already rejected synchronously upstream in the REST parser -- these cover
 * the STATE-dependent conflicts that only the live element can see. */

/* Reject helper for the admission functions: LOG the reason (with its machine
 * code, so operators and log-based tests see WHY) AND fill the query result that
 * the REST caller receives. Keeps the two in lockstep from one call site. */
#define ADMIT_REJECT(self, res, code, hint, fmt, ...)  G_STMT_START {      \
    gchar *_am = g_strdup_printf ((fmt), ##__VA_ARGS__);                   \
    MODELMUX_CONTROL_ERR ((self), "%s (%s)", _am, (code));                 \
    modelmux_control_result_set ((res), 400, (code), _am, (hint));         \
    g_free (_am);                                                         \
  } G_STMT_END

static void
modelmux_control_admit_model_load (GstNvModelMux * self, const ModelMuxModelPayload * pl,
    ModelMuxControlResult * res)
{
  const gchar *name = pl->name, *ver = pl->version;
  const ModelCatalogEntry *def;
  gchar *key, *ok;

  if (!name || !*name || !ver || !*ver) {
    ADMIT_REJECT (self, res, "FIELD_MISSING", "both are required",
        "model/load requires 'name' and 'version'");
    return;
  }
  if (modelmux_bin_update_in_flight (self->modelmux_bin, name)) {
    ADMIT_REJECT (self, res, "UPDATE_IN_FLIGHT",
        "retry once the update commits or rolls back",
        "model/load rejected: a model/update on '%s' is awaiting nvinfer's confirm", name);
    return;
  }
  key = model_key (name, ver);
  if (pl->batch_size) {
    guint live_b = modelmux_bin_model_batch (self->modelmux_bin, key);
    if (live_b && live_b != pl->batch_size) {
      ADMIT_REJECT (self, res, "BATCH_IMMUTABLE",
          "model/unload first, or load it as a new version",
          "model/load rejected: '%s' is LIVE with batch %u; batch_size %u cannot apply "
          "in place", key, live_b, pl->batch_size);
      g_free (key);
      return;
    }
  }
  if (modelmux_bin_model_loaded (self->modelmux_bin, key)) {
    ADMIT_REJECT (self, res, "MODEL_ALREADY_LOADED",
        "use a different version (an in-place swap is model/update's job)",
        "model/load rejected: '%s' is already loaded", key);
    g_free (key);
    return;
  }
  def = modelmux_config_find_model (&self->config, name);
  if (def && def->type == MODEL_INFERSERVER && (pl->engine_file || pl->n_engines)) {
    ADMIT_REJECT (self, res, "ENGINE_UNSUPPORTED_BACKEND",
        "add the checkpoint to the Triton model repository and load by config",
        "model/load rejected: engine_file/engine_files do not apply to nvinferserver "
        "model '%s'", key);
    g_free (key);
    return;
  }
  ok = g_strdup_printf ("'%s' accepted; warming -> 'model loaded [%s]' when ready", key, key);
  modelmux_control_result_set (res, 202, NULL, ok, "or GET /api/v1/model/status");
  g_free (ok); g_free (key);
}

static void
modelmux_control_admit_model_update (GstNvModelMux * self, const ModelMuxModelPayload * pl,
    ModelMuxControlResult * res)
{
  const gchar *name = pl->name, *from_ver = pl->from_version, *ver = pl->version;
  const ModelCatalogEntry *def;
  guint live_b, to_b;
  gchar *key, *ok;

  if (!name || !*name || !from_ver || !*from_ver || !ver || !*ver) {
    ADMIT_REJECT (self, res, "FIELD_MISSING", "all three are required",
        "model/update requires 'name', 'from_version' and 'version'");
    return;
  }
  if (g_strcmp0 (from_ver, ver) == 0) {
    ADMIT_REJECT (self, res, "VERSION_INVALID", "pick a new, distinct version",
        "model/update rejected: 'version' must differ from 'from_version' "
        "(identity must move)");
    return;
  }
  def = modelmux_config_find_model (&self->config, name);
  if (!def) {
    ADMIT_REJECT (self, res, "MODEL_NOT_IN_CATALOG",
        "model/load the model before updating it",
        "model/update rejected: '%s' is not in the catalog", name);
    return;
  }
  if (def->type == MODEL_INFERSERVER) {
    ADMIT_REJECT (self, res, "MODEL_UPDATE_UNSUPPORTED_BACKEND",
        "put the new checkpoint in the Triton repository, then model/load + stream/route",
        "model/update rejected: '%s' is nvinferserver; in-place update is not supported",
        name);
    return;
  }
  if (modelmux_bin_update_in_flight (self->modelmux_bin, name)) {
    ADMIT_REJECT (self, res, "UPDATE_IN_FLIGHT", "retry once it commits or rolls back",
        "model/update rejected: another model/update on '%s' is awaiting confirm", name);
    return;
  }
  key = model_key (name, from_ver);
  if (!modelmux_bin_model_loaded (self->modelmux_bin, key)) {
    ADMIT_REJECT (self, res, "UPDATE_FROM_MISMATCH",
        "the precondition must name a live version; load it first",
        "model/update rejected: from_version '%s' is not loaded", key);
    g_free (key);
    return;
  }
  /* BATCH IMMUTABILITY across an in-place swap: the element (slot tables/mux) is
   * KEPT, so a to-version declaring a different batch cannot be honoured. Same
   * check the async handler does -- surfaced here so the caller gets it as a 400. */
  live_b = modelmux_bin_model_batch (self->modelmux_bin, key);
  to_b = pl->batch_size ? pl->batch_size
      : modelmux_config_version_batch (&self->config, name, ver);
  if (live_b && to_b && to_b != live_b) {
    ADMIT_REJECT (self, res, "UPDATE_BATCH_MISMATCH",
        "model/load the to-version as a NEW version and stream/route to it",
        "model/update '%s' -> '%s' rejected: version batch %u differs from the live "
        "instance's %u (an in-place swap keeps the element and cannot resize it)",
        key, ver, to_b, live_b);
    g_free (key);
    return;
  }
  g_free (key);
  ok = g_strdup_printf ("'%s@%s' -> '%s@%s' accepted; swapping -> 'model updated' when committed",
      name, from_ver, name, ver);
  modelmux_control_result_set (res, 202, NULL, ok, "or GET /api/v1/model/status");
  g_free (ok);
}

/* ================================================================== */
/* Synchronous control-admission query (sink-ghost QUERY probe)         */
/*                                                                      */
/* nvmultiurisrcbin sends every model-plane op (load/unload/update) as a */
/* DOWNSTREAM custom query so WE decide admission on the spot and write   */
/* the verdict + reason back into the query -- the REST caller then       */
/* returns OUR message instead of a blind 202. Runs on the caller's       */
/* (REST) thread; takes ib->lock, like the model/status query probe.      */
/*   - unload EXECUTES here (fast: idle check + hash remove; TRT teardown  */
/*     is deferred).                                                       */
/*   - load/update ADMIT here (read-only) and, on accept, SCHEDULE the     */
/*     real warm/swap on the main loop (creation must not run on this      */
/*     thread). The verdict is still returned synchronously.               */
/* Non-REST hosts still use the async in-band event (fire-and-forget).     */
/* ================================================================== */
G_GNUC_INTERNAL GstPadProbeReturn
modelmux_control_query_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  GstQuery *q = GST_PAD_PROBE_INFO_QUERY (info);
  ModelMuxBin *ib = (ModelMuxBin *) udata;
  GstNvModelMux *self;
  const gchar *op = NULL, *json = NULL;
  ModelMuxModelPayload pl;
  ModelMuxControlResult res = { 0, NULL, NULL, NULL };
  gboolean is_load, is_unload, is_update, is_route;
  (void) pad;

  if (!q || !gst_nvquery_is_control (q))
    return GST_PAD_PROBE_OK;             /* not our query -> let it pass */

  self = ib ? GST_NVMODELMUX (ib->parent_pipeline) : NULL;
  gst_nvquery_control_parse_request (q, &op, &json);
  is_load = (op && g_strcmp0 (op, "model/load") == 0);
  is_unload = (op && g_strcmp0 (op, "model/unload") == 0);
  is_update = (op && g_strcmp0 (op, "model/update") == 0);
  is_route = (op && g_strcmp0 (op, "stream/route") == 0);

  if (!self || !(is_load || is_unload || is_update || is_route)) {
    gst_nvquery_control_set_response (q, 501, "NOT_IMPLEMENTED",
        "operation not handled synchronously by this element", NULL);
    return GST_PAD_PROBE_HANDLED;
  }

  /* stream/route: run the handler's ADMISSION PASS only (admit_only) and capture
   * the reason it logs on reject. Route has its own payload (not a model payload),
   * so it is handled before the model-plane decode. */
  if (is_route) {
    const gchar *captured;
    modelmux_control_err_capture_arm ();
    modelmux_control_handle_stream_route (self, json, TRUE /* admit_only */);
    captured = modelmux_control_err_capture_get ();
    if (captured && *captured) {                    /* rejected -- specific reason */
      gst_nvquery_control_set_response (q, 400, "STREAM_ROUTE_REJECTED", captured,
          "re-check the referenced models/streams and if_revision, then retry");
    } else {                                        /* admitted -> apply async */
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_STREAM_ROUTE, json);
      gst_nvquery_control_set_response (q, 202, NULL,
          "stream/route accepted; applying -> 'streams routed' when done",
          "or GET /api/v1/model/status");
    }
    modelmux_control_err_capture_disarm ();
    return GST_PAD_PROBE_HANDLED;
  }

  /* model plane (load / unload / update): one shared payload decode. */
  if (!modelmux_control_decode_model_payload (self, op, json, &pl)) {
    gst_nvquery_control_set_response (q, 400, "FIELD_TYPE_INVALID",
        "malformed model-plane payload", "see the request schema for this endpoint");
    return GST_PAD_PROBE_HANDLED;
  }

  if (is_unload) {
    modelmux_control_handle_model_unload (self, &pl, &res);   /* executes now */
  } else {
    if (is_load)
      modelmux_control_admit_model_load (self, &pl, &res);
    else
      modelmux_control_admit_model_update (self, &pl, &res);
    if (res.http == 202)                  /* admitted -> run the real op async */
      modelmux_control_post_ctrl_json (self,
          is_load ? MODELMUX_CONTROL_MODEL_LOAD : MODELMUX_CONTROL_MODEL_UPDATE, json);
  }
  gst_nvquery_control_set_response (q, res.http ? res.http : 202,
      res.err_code, res.reason, res.hint);
  modelmux_control_result_clear (&res);
  modelmux_control_model_payload_clear (&pl);
  return GST_PAD_PROBE_HANDLED;
}

/* ================================================================== */
/* Data-plane auto-attach (the reliable, app-independent stream path)   */
/*                                                                      */
/* The bus "sync-message" path below only works when the HOST app does  */
/* not own the bus sync handler -- but deepstream-app (and its video    */
/* sink's window-handle handler) DOES, so stream-add never reaches us   */
/* there. This sink-pad buffer probe instead reads the source_ids that  */
/* are actually present in each batched buffer's NvDsBatchMeta and       */
/* attaches the new ones / detaches the vanished ones -- works in every  */
/* host (deepstream-app, test5, gst-launch, custom). Runs on the         */
/* streaming thread, so it only SCHEDULES attach/detach on the main loop */
/* (never blocks a pad from within its own probe).                       */
/* ================================================================== */
/* PRIMARY dynamic link/unlink driver: the nvstreammux stream-lifecycle events.
 *   GST_NVEVENT_PAD_ADDED   -> a source appeared: attach (LINK) its branch NOW, before
 *                              its segment/buffers proceed, so nvstreamdemux routes it.
 *                              Done SYNCHRONOUSLY (on the serialized event thread, before
 *                              returning OK) -- the new branch carries no data yet, so
 *                              this never blocks on the streaming path.
 *   GST_NVEVENT_PAD_DELETED  -> a source left: detach (UNLINK) its branch.
 *   GST_NVEVENT_STREAM_EOS   -> per-stream EOS only; do not detach until PAD_DELETED.
 * These events are serialized + sticky, flow through the data path, and reach us in ANY
 * host app -- so the plugin links/unlinks bins dynamically with no app glue and no
 * dependence on the bus (which the host app owns). */
static GstPadProbeReturn
modelmux_control_sink_event_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  GstNvModelMux *self = GST_NVMODELMUX (udata);
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT (info);
  GstEventType event_type;
  guint sid = 0;
  (void) pad;

  /* streaming thread: NEVER take api_control_lock here (teardown holds it while joining
   * streaming threads). This is a lock-free ADVISORY check only -- the pointer is
   * never dereferenced on this path; real users re-validate under api_control_lock. */
  if (!event || !g_atomic_pointer_get (&self->modelmux_bin))
    return GST_PAD_PROBE_OK;
  event_type = GST_EVENT_TYPE (event);

  if (event_type == (GstEventType) GST_NVEVENT_PAD_ADDED) {
    gboolean is_new_source;
    gst_nvevent_parse_pad_added (event, &sid);
    g_mutex_lock (&self->source_state_lock);
    is_new_source = (self->source_attach_state
        && !g_hash_table_contains (self->source_attach_state, GUINT_TO_POINTER (sid)));
    if (is_new_source) {
      ModelMuxSourceAttachState *state = g_new0 (ModelMuxSourceAttachState, 1);
      state->last_seen = self->batch_idx;
      g_hash_table_insert (self->source_attach_state, GUINT_TO_POINTER (sid), state);
    }
    g_mutex_unlock (&self->source_state_lock);
    if (is_new_source) {
      /* DYNAMIC LINK, deadlock-free & stall-free:
       *  - GST_NVEVENT_PAD_ADDED is in-band and serialized AHEAD of the new source's
       *    first decoded frame, so it is the earliest safe moment to wire the stream.
       *  - The attach itself runs on the MAIN LOOP (modelmux_control_handle_stream_add does pad/state
       *    work that deadlocks if run inline on this streaming thread -- the app always
       *    ran it from the bus/main loop).
       *  - The buffer drop-probe (modelmux_control_autoattach_probe) is the safety net: it drops a
       *    source's buffers until this attach completes, so a frame that races ahead of
       *    the demux src pad is dropped rather than triggering NOT_LINKED. We must NOT
       *    block the sink here -- that back-pressures and stalls the upstream source. */
      NvDsSensorInfo si = { 0 };
      si.source_id = sid;
      modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_ADD, &si);
    }
  } else if (event_type == (GstEventType) GST_NVEVENT_PAD_DELETED) {
    /* DYNAMIC UNLINK: PAD_DELETED is emitted by nvstreammux when it RELEASES the
     * source's sink pad -- which nvmultiurisrcbin does only AFTER it has stopped +
     * flushed that source. So it is serialized after the source's last batched frame:
     * the in-demux src pad can now be released with nothing in flight (exactly the
     * precondition the app gets from the post-flush stream-remove bus message).
     *
     * STREAM_EOS is deliberately NOT a detach trigger: it can fire mid-removal (or on
     * a file boundary) BEFORE the flush completes, so tearing down then releases the
     * in-demux src pad while source frames are still in flight -> the demux returns
     * NOT_LINKED upstream, which propagates back through nvmultiurisrcbin's muxer and
     * kills the OTHER sources' decoders ("Internal data stream error"). */
    NvDsSensorInfo si = { 0 };
    gst_nvevent_parse_pad_deleted (event, &sid);
    si.source_id = sid;
    g_mutex_lock (&self->source_state_lock);
    if (self->source_attach_state)
      g_hash_table_remove (self->source_attach_state, GUINT_TO_POINTER (sid));
    g_mutex_unlock (&self->source_state_lock);
    /* detach asynchronously on the main loop (teardown does pad/state work that is
     * unsafe to run inline on the streaming thread). */
    modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_REMOVE, &si);
  } else if (event_type == (GstEventType) GST_NVEVENT_MODEL_LOAD ||
      event_type == (GstEventType) GST_NVEVENT_MODEL_UNLOAD ||
      event_type == (GstEventType) GST_NVEVENT_MODEL_UPDATE ||
      event_type == (GstEventType) GST_NVEVENT_STREAM_ROUTE) {
    /* v1 REST control delivered IN-BAND on the data plane (by the upstream
     * nvmultiurisrcbin) instead of on the bus -- so it reaches us in ANY host
     * app, even one that owns the bus sync handler (deepstream-app/test5).
     * Each event carries the request's "value" object as ONE verbatim JSON
     * string; marshal it to the main loop (deep-copied) and decode it there.
     * The event is consumed here (not meaningful downstream). */
    gchar *json = NULL;
    if (event_type == (GstEventType) GST_NVEVENT_MODEL_LOAD) {
      gst_nvevent_parse_model_load (event, &json);
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_MODEL_LOAD, json);
    } else if (event_type == (GstEventType) GST_NVEVENT_MODEL_UNLOAD) {
      gst_nvevent_parse_model_unload (event, &json);
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_MODEL_UNLOAD, json);
    } else if (event_type == (GstEventType) GST_NVEVENT_MODEL_UPDATE) {
      gst_nvevent_parse_model_update (event, &json);
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_MODEL_UPDATE, json);
    } else {
      gst_nvevent_parse_stream_route (event, &json);
      modelmux_control_post_ctrl_json (self, MODELMUX_CONTROL_STREAM_ROUTE, json);
    }
    g_free (json);
    return GST_PAD_PROBE_DROP;
  } else if (event_type == (GstEventType) GST_NVEVENT_STREAM_MODEL_BIND) {
    /* per-stream model binding from stream/add, delivered in-band. Cache it /
     * reconcile on the main loop (modelmux_control_post_sensor_event deep-copies the strings). */
    NvDsSensorInfo si = { 0 };
    guint bsid = 0;
    gchar *cam = NULL, *cnm = NULL, *meta = NULL;
    gst_nvevent_parse_stream_model_bind (event, &bsid, &cam, &cnm, &meta);
    si.source_id = bsid;
    si.sensor_id = cam;
    si.sensor_name = cnm;
    si.sensor_metadata = meta;
    modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_MODEL_BIND, &si);
    g_free (cam); g_free (cnm); g_free (meta);
    return GST_PAD_PROBE_DROP;
  }
  /* GST_NVEVENT_STREAM_EOS is intentionally NOT a detach trigger: it can fire
   * mid-removal / on a file boundary before the source is flushed (see the
   * PAD_DELETED rationale above), so detaching on it would release the in-demux
   * src pad while frames are in flight. */
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
modelmux_control_autoattach_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  GstNvModelMux *self = GST_NVMODELMUX (udata);
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *bmeta;
  NvDsMetaList *l;
  guint64 idx;
  (void) pad;

  /* streaming thread: lock-free ADVISORY check only (see the event probe) --
   * never dereferenced here; api_control_lock must not be taken from probes. */
  if (!buf || !g_atomic_pointer_get (&self->modelmux_bin))
    return GST_PAD_PROBE_OK;
  bmeta = gst_buffer_get_nvds_batch_meta (buf);
  if (!bmeta)
    return GST_PAD_PROBE_OK;

  gboolean all_attached = TRUE;

  g_mutex_lock (&self->source_state_lock);
  if (!self->source_attach_state) {               /* detach swapped the table out mid-flight */
    g_mutex_unlock (&self->source_state_lock);
    return GST_PAD_PROBE_OK;
  }
  idx = ++self->batch_idx;

  /* attach source_ids that just appeared; refresh last-seen for all present */
  for (l = bmeta->frame_meta_list; l; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *) l->data;
    gpointer key = GUINT_TO_POINTER (fm->source_id);
    ModelMuxSourceAttachState *ent = g_hash_table_lookup (self->source_attach_state, key);
    if (!ent) {
      NvDsSensorInfo si = { 0 };
      ent = g_new0 (ModelMuxSourceAttachState, 1);
      ent->attached = FALSE;
      ent->last_seen = idx;
      g_hash_table_insert (self->source_attach_state, key, ent);
      si.source_id = fm->source_id;   /* camera_id unknown here -> default/binding */
      modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_ADD, &si);   /* attach on the main loop */
    } else {
      ent->last_seen = idx;
      if (!ent->attached && ent->retry_at && idx >= ent->retry_at) {
        /* a previous attach FAILED: bounded-backoff re-attach. Push retry_at out
         * so we post at most one retry per backoff window (the failure handler
         * re-arms it with the grown backoff if this attempt fails too). */
        NvDsSensorInfo si = { 0 };
        si.source_id = fm->source_id;
        ent->retry_at = idx +
            ((guint64) MM_AUTO_STALE << MIN (ent->fails, MM_AUTO_FAIL_SHIFT_MAX));
        modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_ADD, &si);
      }
    }
    if (!ent->attached)
      all_attached = FALSE;           /* still wiring -> drop this batch */
  }

  /* detach source_ids absent from the batch for MM_AUTO_STALE buffers (removed) */
  {
    GHashTableIter it;
    gpointer k, v;
    GList *gone = NULL, *g;
    g_hash_table_iter_init (&it, self->source_attach_state);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      if (idx > ((ModelMuxSourceAttachState *) v)->last_seen + MM_AUTO_STALE)
        gone = g_list_prepend (gone, k);
    }
    for (g = gone; g; g = g->next) {
      NvDsSensorInfo si = { 0 };
      si.source_id = GPOINTER_TO_UINT (g->data);
      modelmux_control_post_sensor_event (self, MODELMUX_CONTROL_STREAM_REMOVE, &si);
      g_hash_table_remove (self->source_attach_state, g->data);   /* frees the ModelMuxSourceAttachState */
    }
    g_list_free (gone);
  }
  g_mutex_unlock (&self->source_state_lock);

  /* DROP the batch until every source in it is attached: an un-routable source_id
   * at the in-demux returns NOT_LINKED and kills the upstream decoder. Dropping
   * here returns GST_FLOW_OK upstream, so the decoder keeps running until the
   * (async) attach completes -- then buffers flow normally. */
  return all_attached ? GST_PAD_PROBE_OK : GST_PAD_PROBE_DROP;
}

/* ================================================================== */
/* Attach / detach (called from the element build / teardown)          */
/* ================================================================== */

/* Wire the control plane's callbacks INTO the freshly-built bin. Called from
 * the element build (gstnvmodelmux.c) after modelmux_bin_new() and
 * BEFORE the bin is published to self->modelmux_bin (under api_control_lock), so no other
 * thread can observe the fields half-set. The bin invokes ota_rollback_cb on
 * the main loop after a FAILED nvinfer confirm rolled the identity back. */
void
gst_modelmux_control_wire_ota_rollback (GstNvModelMux * self, ModelMuxBin * modelmux_bin)
{
  modelmux_bin->ota_rollback_cb = modelmux_control_ota_rollback_cb;
  modelmux_bin->ota_rollback_owner = self;
}

/*
 * Turn on the element's self-driving control plane (called once from build(), after the
 * bin is created, so the element needs no host-app glue). Sets up two things:
 *   1) DATA-PLANE auto-attach -- allocate the per-stream tracking tables (source_attach_state,
 *      requested_binding_info) and install two sink-pad probes: an EVENT probe that attaches/detaches
 *      streams on PAD_ADDED/PAD_DELETED, and a BUFFER probe that drops a source's frames until
 *      it is attached (safety net + fallback attach).
 *   2) BUS REST control -- subscribe to the pipeline bus "sync-message" so native-REST ops
 *      (stream add/remove, model load/unload/update) are executed INTERNALLY.
 * Idempotent (every step is guarded) and undone by gst_modelmux_control_detach at teardown.
 */
void
gst_modelmux_control_attach (GstNvModelMux * self)
{
  /* Initialize the stream attach state and stream bindings tables. */
  if (!self->source_attach_state)
    self->source_attach_state = g_hash_table_new_full (NULL, NULL, NULL, g_free);
  if (!self->requested_binding_info)
    self->requested_binding_info = g_hash_table_new_full (NULL, NULL, NULL, requested_binding_info_free);

  /* Initialize the batch index. */
  self->batch_idx = 0;

  /* Add the EVENT probe: PAD_ADDED/PAD_DELETED drive link/unlink; STREAM_EOS is observed only. */
  if (!self->sink_event_probe_id && self->sinkpad)
    self->sink_event_probe_id = gst_pad_add_probe (self->sinkpad,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, modelmux_control_sink_event_probe, self, NULL);

  /* Add the BUFFER probe: drop a source's buffers until it is attached (safety net so an
   * un-routable source can never NOT_LINK the decoder; also a fallback attach for
   * sources that arrive without a PAD_ADDED event). */
  if (!self->sink_probe_id && self->sinkpad)
    self->sink_probe_id = gst_pad_add_probe (self->sinkpad,
        GST_PAD_PROBE_TYPE_BUFFER, modelmux_control_autoattach_probe, self, NULL);

  /* Subscribe to the bus: works when the host app does NOT own the bus
   * sync handler (standalone / gst-launch), and carries the REST model/load|
   * unload|update ops + per-stream model metadata. Harmless if it never fires. */
  if (!self->bus_sync_id) {
    self->bus = gst_element_get_bus (GST_ELEMENT (self));
    if (self->bus) {
      gst_bus_enable_sync_message_emission (self->bus);
      self->bus_sync_id = g_signal_connect (self->bus, "sync-message",
          G_CALLBACK (modelmux_control_bus_sync_message), self);
    }
  }

  MODELMUX_CONTROL_INFO (self, "control attached: data-plane auto-attach%s",
      self->bus ? " + bus REST control" : " (bus unavailable)");
}

void
gst_modelmux_control_detach (GstNvModelMux * self)
{
  if (self->sink_event_probe_id && self->sinkpad) {
    gst_pad_remove_probe (self->sinkpad, self->sink_event_probe_id);
    self->sink_event_probe_id = 0;
  }
  if (self->sink_probe_id && self->sinkpad) {
    gst_pad_remove_probe (self->sinkpad, self->sink_probe_id);
    self->sink_probe_id = 0;
  }
  if (self->pending_poll_id) {
    g_source_remove (self->pending_poll_id);
    self->pending_poll_id = 0;
  }
  g_list_free_full (self->pending_reroutes, modelmux_control_pending_free);
  self->pending_reroutes = NULL;
  /* queue is gone -> clear the mirror too, so a status read racing teardown
   * cannot report routes that will now never apply. No-op once modelmux_bin is
   * NULL (its own free drops the array). */
  modelmux_control_publish_pending (self);

  if (self->bus) {
    if (self->bus_sync_id) {
      g_signal_handler_disconnect (self->bus, self->bus_sync_id);
      self->bus_sync_id = 0;
    }
    gst_bus_disable_sync_message_emission (self->bus);
    gst_object_unref (self->bus);
    self->bus = NULL;
  }
  /* Detach the tables UNDER source_state_lock: gst_pad_remove_probe does NOT wait for a
   * probe callback already executing on the streaming thread, and both probes
   * dereference source_attach_state/requested_binding_info under source_state_lock. Swap the pointers out
   * inside the lock (a mid-flight probe then sees NULL and bails), destroy the
   * tables after releasing it. */
  {
    GHashTable *seen, *bind;
    g_mutex_lock (&self->source_state_lock);
    seen = self->source_attach_state;
    self->source_attach_state = NULL;
    bind = self->requested_binding_info;
    self->requested_binding_info = NULL;
    g_mutex_unlock (&self->source_state_lock);
    if (seen)
      g_hash_table_destroy (seen);
    if (bind)
      g_hash_table_destroy (bind);
  }
}
