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
 * Model-status REST query support.
 *
 * The REST layer issues a custom downstream query for /api/v1/model/status.
 * Keep the JSON builder isolated here so the bin implementation can focus on
 * graph construction, stream add/remove, and routing. The builder only takes a
 * snapshot under ib->lock; it does not mutate the pipeline.
 */

#include "gstnvmodelmux_priv.h"
#include "gstnvmodelmux.h"   /* GstNvModelMux: the query probes answer per STAGE */
#include <json-glib/json-glib.h>   /* cross-stage merge of the GET plane */
#include "gst-nvdscustommessage.h"  /* gst_nvquery_model_*_parse_response */

GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat

/* model-status custom query (nvmultiurisrcbin's REST server asks it on
 * /api/v1/model/status). Declared locally to avoid pulling the helper header;
 * symbols live in libnvdsgst_customhelper. */
extern gboolean gst_nvquery_is_model_status (GstQuery * query);
/* routing-plane read query -- same transport, different question. */
extern gboolean gst_nvquery_is_model_routing (GstQuery * query);
extern void gst_nvquery_model_routing_set_response (GstQuery * query, const gchar * json);
extern gboolean gst_nvquery_model_routing_parse_request (GstQuery * query,
    const gchar ** camera_id, gboolean * filtered);
extern void gst_nvquery_model_status_set_response (GstQuery * query, const gchar * json);
extern gboolean gst_nvquery_model_status_parse_request (GstQuery * query,
    const gchar ** model_name, const gchar ** model_version,
    const gchar ** stream_name, gint * source_id);

/* Routing revision with ib->lock ALREADY held. Defined with the rest of the
 * revision plumbing at the bottom of this file; declared here because the two GET
 * builders above it are its only callers. */
static guint64 modelmux_bin_revision_locked (ModelMuxBin * ib);

/* Minimal JSON string escaper for hand-built status JSON. It handles the
 * characters that would otherwise let a user-supplied model or camera name
 * produce invalid JSON. Returns a newly allocated string; caller frees it. */
static gchar *
modelmux_json_escape (const gchar * in)
{
  GString *o = g_string_new (NULL);
  const gchar *p;
  for (p = in ? in : ""; *p; p++) {
    guchar c = (guchar) *p;
    switch (c) {
      case '"':  g_string_append (o, "\\\""); break;
      case '\\': g_string_append (o, "\\\\"); break;
      case '\n': g_string_append (o, "\\n");  break;
      case '\r': g_string_append (o, "\\r");  break;
      case '\t': g_string_append (o, "\\t");  break;
      default:
        if (c < 0x20)
          g_string_append_printf (o, "\\u%04x", c);
        else
          g_string_append_c (o, *p);
    }
  }
  return g_string_free (o, FALSE);
}

/* API state vocabulary (API_DESIGN.md): lowercase, and the internal WARMED
 * (engine ready, not yet serving) surfaces as "ready". */
static const gchar *
modelmux_status_state_str (ModelBin * model_bin)
{
  switch (model_bin_status (model_bin)) {
    case MODEL_WARMING: return "warming";
    case MODEL_WARMED:  return "ready";
    case MODEL_SERVING: return "serving";
    case MODEL_FAILED:  return "failed";
    default:               return "warming";
  }
}

/* One logical (name, version) as seen across the role pools + limbo. The
 * SAME key can exist as a primary-pool bin, a shadow-pool bin and/or a limbo
 * bin; status reports it as ONE version entry with per-role instance rows. */
typedef struct
{
  ModelBin *prim;           /* bin in the primary pool (or NULL) */
  ModelBin *shad;           /* bin in the shadow pool (or NULL)  */
  ModelBin *limbo;          /* warmed-but-unassigned bin (or NULL) */
} ModelMuxStatusRow;

static void
modelmux_status_collect_pool (GHashTable * rows, ModelPool * rb, guint which)
{
  GHashTableIter it;
  gpointer k, v;

  if (!rb || !rb->models)
    return;
  g_hash_table_iter_init (&it, rb->models);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelBin *model_bin = (ModelBin *) v;
    ModelMuxStatusRow *row = g_hash_table_lookup (rows, model_bin->key);
    if (!row) {
      row = g_new0 (ModelMuxStatusRow, 1);
      g_hash_table_insert (rows, g_strdup (model_bin->key), row);
    }
    if (which == 0)
      row->prim = model_bin;
    else if (which == 1)
      row->shad = model_bin;
    else
      row->limbo = model_bin;
  }
}

/* Aggregate a shard chain: total streams, shard count, summed fps + base
 * latency (shards of a model share identity; they are ONE logical instance). */
static void
modelmux_status_chain_totals (ModelBin * base, gboolean perf_on, guint * streams,
    guint * shards, gdouble * fps, gdouble * lat)
{
  ModelBin *model_bin;

  *streams = *shards = 0;
  *fps = *lat = 0;
  for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
    *streams += model_bin->num_streams;
    (*shards)++;
    if (perf_on && model_bin->perf) {
      gdouble f = 0, l = 0;
      modelmux_perf_get_model (model_bin->perf, &f, &l);
      *fps += f;
      if (model_bin == base)
        *lat = l;
    }
  }
}

/* Append one instance row PER SHARD of the chain: every shard is one
 * materialized instance (id = the shard's stable element instance index;
 * instances[] groups place one shard per listed GPU). nvinfer rows carry the
 * shard's actual placement (its recorded gpu, falling back to the version
 * registry); for nvinferserver Triton schedules internally, so placement
 * fields are replaced by the backend's real identity (never a fake GPU). */
static void
modelmux_status_append_instances (GString * s, ModelMuxBin * ib, ModelBin * base,
    gboolean perf_on, gboolean * first)
{
  ModelBin *model_bin;
  gchar *ecfg, *eeng;

  for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
    g_string_append_printf (s, "%s{\"id\":%u,", *first ? "" : ",", model_bin->inst);
    *first = FALSE;
    if (modelmux_backend (model_bin->type)->owns_placement) {
      /* placement parity with nvinfer WHEN THE PLUGIN PLACED IT (model/load
       * gpu -> derived single-value gpu_ids, recorded pin/registry/config).
       * A multi-gpu gpu_ids list (or no pin) is Triton's internal scheduling
       * -- report that honestly instead of a fake single device. */
      gint gpu = model_bin->gpu;
      gchar *enm = modelmux_json_escape (model_bin->name);
      gchar *ever = modelmux_json_escape (model_version_str (model_bin));
      if (gpu < 0)
        gpu = modelmux_config_version_gpu (ib->config, model_bin->name,
            model_version_str (model_bin));
      if (gpu < 0)
        gpu = model_bin->cfg_gpu;               /* single-value gpu_ids; a list = -1 */
      ecfg = modelmux_json_escape (model_bin->config_file);
      g_string_append_printf (s,
          "\"backend\":\"nvinferserver\",\"config\":\"%s\","
          "\"triton_model\":\"%s\",\"triton_version\":\"%s\",\"gie_id\":%u,",
          ecfg, enm, ever, model_bin->unique_id);
      if (gpu >= 0)
        g_string_append_printf (s, "\"gpu\":%d,", gpu);
      else
        g_string_append (s, "\"placement\":\"triton_internal\",");
      g_free (ecfg);
      g_free (enm);
      g_free (ever);
    } else {
      /* same resolution the SCHEDULER uses (own gpu -> registry -> config-file
       * pin -> 0), so status never reports a device the shard doesn't run on */
      gint gpu = model_bin->gpu;
      if (gpu < 0)
        gpu = modelmux_config_version_gpu (ib->config, model_bin->name,
            model_version_str (model_bin));
      if (gpu < 0)
        gpu = model_bin->cfg_gpu;
      eeng = modelmux_json_escape (model_bin->engine);
      g_string_append_printf (s, "\"gpu\":%d,\"gie_id\":%u,\"engine\":\"%s\",",
          gpu >= 0 ? gpu : 0, model_bin->unique_id, eeng);
      g_free (eeng);
    }
    g_string_append_printf (s, "\"state\":\"%s\",\"streams\":%u,\"capacity\":%u",
        modelmux_status_state_str (model_bin), model_bin->num_streams, model_bin->max_streams);
    /* perf is measured at the BIN boundary (see modelmux_perf_bin_in/out_probe), so it
     * is a property of the bin, not of whatever backend runs inside it -- report it for
     * every backend.
     *
     * WHAT infer_latency_ms SPANS: inference HEAD -> inference src, i.e. the [nvdsxfer]
     * device migration plus the inference call, NOT the pure GPU time of the backend.
     * The name is narrower than the measurement and is kept only because it is a
     * published v1 field; the schema says the same thing, so the number is never
     * quietly something other than its name. Measuring the bin rather than the element
     * is deliberate -- it is what makes the figure comparable across backends. */
    if (perf_on && model_bin->perf) {
      gdouble fps = 0, lat = 0;
      modelmux_perf_get_model (model_bin->perf, &fps, &lat);
      g_string_append_printf (s, ",\"fps\":%.2f,\"infer_latency_ms\":%.2f",
          fps, lat);
    }
    g_string_append_c (s, '}');
  }
}

/* Append one version entry ({"version", "state", "default_for"?, "serving",
 * "config"?, "instances":[...]}). */
static void
modelmux_status_append_version (GString * s, ModelMuxBin * ib, ModelMuxStatusRow * row,
    gboolean perf_on, gboolean * first)
{
  ModelBin *rep = row->prim ? row->prim : (row->shad ? row->shad : row->limbo);
  const gchar *ver = model_version_str (rep);
  gboolean def_model = FALSE, def_shadow = FALSE;
  guint p_streams = 0, s_streams = 0, dummy;
  gdouble d1, d2;
  gboolean ifirst = TRUE;
  gchar *ever = modelmux_json_escape (ver);

  {
    /* runtime defaults live in the pool refs (config only seeded them); the
     * status walk runs under ib->lock, same lock the default writers take. */
    const DefaultModelRef *dp = ib->primary_pool ? &ib->primary_pool->default_ref : NULL;
    const DefaultModelRef *ds = ib->shadow_pool ? &ib->shadow_pool->default_ref : NULL;
    gchar *vkey = model_key (rep->name, ver);
    def_model  = (dp && dp->key && g_strcmp0 (vkey, dp->key) == 0);
    def_shadow = (ds && ds->key && g_strcmp0 (vkey, ds->key) == 0);
    g_free (vkey);
  }
  if (row->prim)
    modelmux_status_chain_totals (row->prim, FALSE, &p_streams, &dummy, &d1, &d2);
  if (row->shad)
    modelmux_status_chain_totals (row->shad, FALSE, &s_streams, &dummy, &d1, &d2);

  g_string_append_printf (s, "%s{\"version\":\"%s\",\"state\":\"%s\"",
      *first ? "" : ",", ever, modelmux_status_state_str (rep));
  *first = FALSE;
  g_free (ever);

  /* A/B vocabulary is omitted entirely when unused (progressive disclosure) */
  if (def_model || def_shadow) {
    g_string_append (s, ",\"default_for\":[");
    if (def_model)
      g_string_append (s, "\"model\"");
    if (def_shadow)
      g_string_append_printf (s, "%s\"shadow\"", def_model ? "," : "");
    g_string_append_c (s, ']');
  }
  /* serving{} counts this version's load PER ROLE: `streams` = streams it serves as
   * the PRIMARY, `shadow_streams` = streams it scores as the SHADOW. They are
   * separate because a version can hold both roles at once (different streams), and
   * only the primary count says whose detections are actually being used.
   *
   * Each is omitted when zero -- same progressive-disclosure rule `shadow_streams`
   * already followed. A shadow-only version used to report `"streams":0` beside
   * `"state":"serving"` and an instance carrying 2 streams, which reads as a
   * contradiction: the model IS inferring, just not as anyone's primary. Saying
   * nothing about a role it does not hold is the honest answer; a client that wants
   * a number reads absent as zero, exactly as it already must for `shadow_streams`.
   *
   * The one case that still emits an explicit zero is a version holding NEITHER role
   * (loaded but idle): `"serving":{"streams":0}`. That zero is the whole point there
   * -- it is the difference between warm-and-unused and not loaded -- and it keeps
   * the object from degenerating to `{}`. */
  g_string_append (s, ",\"serving\":{");
  if (p_streams || !s_streams)
    g_string_append_printf (s, "\"streams\":%u", p_streams);
  if (s_streams)
    g_string_append_printf (s, "%s\"shadow_streams\":%u",
        (p_streams || !s_streams) ? "," : "", s_streams);
  g_string_append_c (s, '}');
  if (rep->config_file && *rep->config_file) {
    gchar *ecfg = modelmux_json_escape (rep->config_file);
    g_string_append_printf (s, ",\"config\":\"%s\"", ecfg);
    g_free (ecfg);
  }
  g_string_append (s, ",\"instances\":[");
  if (row->prim)
    modelmux_status_append_instances (s, ib, row->prim, perf_on, &ifirst);
  if (row->shad)
    modelmux_status_append_instances (s, ib, row->shad, perf_on, &ifirst);
  if (!row->prim && !row->shad && row->limbo)
    modelmux_status_append_instances (s, ib, row->limbo, perf_on, &ifirst);
  g_string_append (s, "]}");
}

/* Resolve one ATTACHED lane to the facts both readers need: its split key, the
 * shard actually serving this stream, and that shard's effective device.
 *
 * Extracted so the MODEL view (modelmux_status_append_ref) and the ROUTING view
 * (modelmux_route_append_ref) cannot disagree. The shard walk is the subtle part
 * -- an instance group spans shards, so the serving shard is found by slot, not
 * by looking at the base ModelBin -- and a second hand-written copy of it was the
 * obvious way for two views of the same stream to start reporting different
 * devices. Device resolution order is unchanged: the serving shard's recorded
 * gpu, then the version registry, then the config-file pin.
 *
 * *triton_internal is TRUE only for an UNPLACED inferserver deployment, which has
 * no single device to report; every other case yields a gpu >= 0. */
static void
modelmux_status_resolve_lane (ModelMuxBin * ib, ModelMuxRoleAttach * ra,
    guint stream_id, gchar ** bare, gchar ** ver, ModelBin ** serving,
    gint * gpu, gboolean * triton_internal)
{
  ModelBin *model_bin = NULL, *cur = NULL, *sh;
  gint dev;

  *bare = *ver = NULL;
  *serving = NULL;
  *gpu = -1;
  *triton_internal = FALSE;

  model_key_split (ra->model, bare, ver);
  if (ra->role && ra->role->models)
    model_bin = g_hash_table_lookup (ra->role->models, ra->model);

  if (model_bin && modelmux_backend (model_bin->type)->owns_placement) {
    /* a PLUGIN-PLACED inferserver instance (model/load gpu -> single-value
     * gpu_ids) reports its device like nvinfer; only an unplaced / multi-gpu
     * gpu_ids Triton deployment keeps the honest "triton_internal". */
    dev = model_bin->gpu;
    if (dev < 0 && *bare)
      dev = modelmux_config_version_gpu (ib->config, *bare, *ver);
    if (dev < 0)
      dev = model_bin->cfg_gpu;
    *serving = model_bin;
    *gpu = dev;
    *triton_internal = (dev < 0);
    return;
  }

  /* the shard ACTUALLY serving this stream (instance groups span shards) --
   * its recorded gpu wins; unrecorded falls back to the version registry. */
  for (sh = model_bin; sh && !cur; sh = (ModelBin *) sh->next_shard)
    {
      guint slot;
      for (slot = 0; slot < sh->max_streams && !cur; slot++)
        if (sh->slot_stream[slot] == (gint) stream_id)
          cur = sh;
    }
  if (!cur)
    cur = model_bin;
  dev = (cur && cur->gpu >= 0) ? cur->gpu
      : (*bare ? modelmux_config_version_gpu (ib->config, *bare, *ver) : -1);
  if (dev < 0 && cur)
    dev = cur->cfg_gpu;                  /* config-file pin: scheduler parity */
  *serving = cur;
  *gpu = dev;
}

/* Append one stream's model/shadow REF ({"name","version",placement...}) --
 * mirrors the model/routing request shape so a controller can reconcile
 * status against what it PUT. */
static void
modelmux_status_append_ref (GString * s, ModelMuxBin * ib, const gchar * label,
    ModelMuxRoleAttach * ra, guint stream_id, gboolean perf_on)
{
  gchar *bare = NULL, *ver = NULL, *ebare, *ever;
  ModelBin *serving = NULL;
  gboolean triton_internal = FALSE;
  gint gpu = -1;

  if (!ra->active || !ra->model)
    return;
  modelmux_status_resolve_lane (ib, ra, stream_id, &bare, &ver, &serving,
      &gpu, &triton_internal);
  ebare = modelmux_json_escape (bare ? bare : "");
  ever = modelmux_json_escape (ver ? ver : "");
  /* origin: "default" = resolved via the pool's default fallback (a default
   * switch sweeps this lane); "pinned" = an explicit ref/route/binding chose
   * the model (default switches leave it alone). */
  g_string_append_printf (s, ",\"%s\":{\"name\":\"%s\",\"version\":\"%s\",\"origin\":\"%s\"",
      label, ebare, ever, ra->via_default ? "default" : "pinned");
  if (serving && modelmux_backend (serving->type)->owns_placement) {
    g_string_append (s, ",\"backend\":\"nvinferserver\"");
    if (!triton_internal)
      g_string_append_printf (s, ",\"gpu\":%d", gpu);
    else
      g_string_append (s, ",\"placement\":\"triton_internal\"");
  } else {
    g_string_append_printf (s, ",\"gpu\":%d,\"instance\":%u",
        gpu >= 0 ? gpu : 0, serving ? serving->inst : 0);
  }
  /* per-stream perf, for EVERY backend: the numbers come from the bin boundary, so
   * they mean the same thing whichever element serves this lane. */
  if (perf_on && serving && serving->perf) {
    gdouble f = 0, l = 0;
    modelmux_perf_get_source (serving->perf, (gint) stream_id, &f, &l);
    g_string_append_printf (s, ",\"fps\":%.2f,\"latency_ms\":%.2f", f, l);
  }
  g_string_append_c (s, '}');
  g_free (ebare);
  g_free (ever);
  g_free (bare);
  g_free (ver);
}

/* Append the top-level "default" object: the RUNTIME default designation for
 * each role, read from the pools' DefaultModelRef.
 *
 * DefaultModelRef -- not ModelMuxConfig.default_* -- is authoritative: the
 * config fields are seeded once in modelmux_bin_new() and never read again,
 * while every runtime re-designation (model/routing default{}, model/load
 * default flags, OTA rename commit/rollback, promote/swap) mutates the pool
 * ref under modelmux_bin->lock, which the caller already holds. Its IDENTITY
 * is valid even when `bin` is NULL -- a default may be designated and never
 * loaded -- which is exactly the case a derivation from models[].default_for
 * would miss (an unloaded default has no models[] row to carry the flag).
 *
 * A role with no default contributes nothing. `gpu` is emitted only when the
 * default was explicitly PINNED: a designation is not a placement, and an
 * unpinned default spreads its streams across the group's instances, so there
 * is no single device to report. */
static void
modelmux_status_append_default_ref (GString * s, const gchar * label,
    DefaultModelRef * dr, gboolean * first)
{
  gchar *en, *ev;

  if (!dr || !dr->name)
    return;
  en = modelmux_json_escape (dr->name);
  ev = modelmux_json_escape (dr->version ? dr->version : "");
  g_string_append_printf (s, "%s\"%s\":{\"name\":\"%s\",\"version\":\"%s\"",
      *first ? "" : ",", label, en, ev);
  if (dr->gpu != MM_GPU_ANY)
    g_string_append_printf (s, ",\"gpu\":%d", dr->gpu);
  g_string_append_c (s, '}');
  *first = FALSE;
  g_free (en);
  g_free (ev);
}

/* One ref inside a stream's "pending" object:
 *   "model":{"name":"X","version":"Y","origin":"pinned"[,"gpu":N]}
 *
 * Shared by the TWO ways a stream can have a pending target -- a passthrough
 * awaiting promotion (prim.pending_model) and a serving stream whose accepted
 * route is parked on a warming model (pending_routes) -- so the two cannot drift
 * into describing the same thing differently.
 *
 * `origin` says whether a later default{} move would re-point this pending route;
 * MM_GPU_ANY (<0) omits `gpu`, because no device is decided yet -- the scheduler
 * picks at apply time by capacity, so emitting one now would be a guess. */
static void
modelmux_status_append_pending_ref (GString * s, const gchar * label,
    const gchar * key, gint gpu, gboolean via_default, gboolean * first)
{
  gchar *bare = NULL, *ver = NULL, *eb, *ev;

  if (!key)
    return;
  model_key_split (key, &bare, &ver);
  eb = modelmux_json_escape (bare ? bare : "");
  ev = modelmux_json_escape (ver ? ver : "");
  g_string_append_printf (s,
      "%s\"%s\":{\"name\":\"%s\",\"version\":\"%s\",\"origin\":\"%s\"",
      *first ? "" : ",", label, eb, ev, via_default ? "default" : "pinned");
  if (gpu >= 0)
    g_string_append_printf (s, ",\"gpu\":%d", gpu);
  g_string_append_c (s, '}');
  *first = FALSE;
  g_free (eb);
  g_free (ev);
  g_free (bare);
  g_free (ver);
}

/* The mirrored deferred intent for `stream_id`, resolved PER LANE.
 *
 * A stream can be covered by MORE THAN ONE note: superseding is lane-aware, so a
 * newer shadow-only route splits the older entry's surviving primary into its own
 * note rather than cancelling it. Returning a single "last match" would then
 * report whichever note came last and silently drop the other lane's intent.
 *
 * Each lane independently takes the NEWEST note that sets it -- the control layer
 * appends in arrival order, so later wins per lane. Caller holds ib->lock. */
static void
modelmux_status_pending_lanes (ModelMuxBin * ib, guint stream_id,
    const ModelMuxPendingRouteNote ** p_out,
    const ModelMuxPendingRouteNote ** s_out)
{
  guint i, j;

  *p_out = NULL;
  *s_out = NULL;
  if (!ib->pending_routes)
    return;
  for (i = 0; i < ib->pending_routes->len; i++) {
    const ModelMuxPendingRouteNote *note = (const ModelMuxPendingRouteNote *)
        g_ptr_array_index (ib->pending_routes, i);
    gboolean covers;
    if (!note)
      continue;
    covers = (note->n == 0);
    for (j = 0; !covers && j < note->n; j++)
      if (note->src[j] == stream_id)
        covers = TRUE;
    if (!covers)
      continue;
    if (note->primary)
      *p_out = note;
    if (note->shadow || note->clear_shadow)
      *s_out = note;
  }
}

static void
modelmux_status_append_default (GString * s, ModelMuxBin * ib)
{
  gboolean first = TRUE;
  GString *d;

  if (!ib)
    return;
  d = g_string_new (NULL);
  if (ib->primary_pool)
    modelmux_status_append_default_ref (d, "model",
        &ib->primary_pool->default_ref, &first);
  if (ib->shadow_pool)
    modelmux_status_append_default_ref (d, "shadow",
        &ib->shadow_pool->default_ref, &first);
  /* omit the key entirely when neither role has a default, rather than
   * emitting an empty object a client would have to special-case. */
  if (d->len)
    g_string_append_printf (s, ",\"default\":{%s}", d->str);
  g_string_free (d, TRUE);
}

/* ==========================================================================
 * ROUTING PLANE READ -- GET /api/v1/model/routing
 *
 * Built HERE, next to modelmux_bin_status_json, for the same reason that one
 * lives here: this element owns the state being reported. It also owns the
 * identifiers the report is keyed by (stream_cam_ids), so the answer needs no
 * join afterwards -- the REST host sends the query and forwards this body
 * untouched, exactly as it does for model/status.
 *
 * The MODEL view and this ROUTING view share their lane resolution
 * (modelmux_status_resolve_lane) and their pending resolution
 * (modelmux_status_pending_lanes) so the two cannot report different facts
 * about the same stream. What differs is the SHAPE:
 *
 *   model/status  -- one object per stream, every field, per-window perf.
 *   model/routing  -- streams GROUPED by identical routing state, so "streams"
 *                    means what it means in the POST (a set of cameras sharing
 *                    one instruction), and a WHITELIST of fields, so the
 *                    per-window perf pair cannot leak in. Those numbers change
 *                    every measurement window; carried through, no two streams
 *                    would ever compare equal, grouping would never merge, and
 *                    two consecutive GETs would differ with nothing changed.
 *
 * Key order is ALPHABETICAL throughout, matching what the previous jsoncpp-based
 * projection emitted (Json::Value orders its members), so the response bytes are
 * unchanged by the move. Within an entry that is model, pending, shadow, state,
 * streams -- and `streams` sorting last is convenient, since grouping can only
 * append it once the members are known.
 * ========================================================================== */

/* One ref in the routing view: the WHITELIST shape, and nothing else.
 * {"gpu":N,}"name","origin","version"  -- or "placement" in gpu's stead for an
 * unplaced Triton deployment, which has no single device to report. */
static void
modelmux_route_append_ref (GString * s, const gchar * label, const gchar * bare,
    const gchar * ver, gboolean via_default, gint gpu, gboolean triton_internal)
{
  gchar *eb = modelmux_json_escape (bare ? bare : "");
  gchar *ev = modelmux_json_escape (ver ? ver : "");

  g_string_append_printf (s, "\"%s\":{", label);
  if (!triton_internal && gpu >= 0)
    g_string_append_printf (s, "\"gpu\":%d,", gpu);
  g_string_append_printf (s, "\"name\":\"%s\",\"origin\":\"%s\",", eb,
      via_default ? "default" : "pinned");
  if (triton_internal)
    g_string_append (s, "\"placement\":\"triton_internal\",");
  g_string_append_printf (s, "\"version\":\"%s\"}", ev);
  g_free (eb);
  g_free (ev);
}

/* An ATTACHED lane, resolved then emitted in the routing shape. */
static void
modelmux_route_append_lane (GString * s, ModelMuxBin * ib, const gchar * label,
    ModelMuxRoleAttach * ra, guint stream_id)
{
  gchar *bare = NULL, *ver = NULL;
  ModelBin *serving = NULL;
  gboolean triton = FALSE;
  gint gpu = -1;

  modelmux_status_resolve_lane (ib, ra, stream_id, &bare, &ver, &serving, &gpu,
      &triton);
  (void) serving;
  modelmux_route_append_ref (s, label, bare, ver, ra->via_default, gpu, triton);
  g_free (bare);
  g_free (ver);
}

/* A DEFERRED target (never yet attached), emitted in the same shape. No device
 * is decided until the apply lands, so MM_GPU_ANY omits `gpu` rather than
 * guessing one. */
static void
modelmux_route_append_pending (GString * s, const gchar * label,
    const gchar * key, gint gpu, gboolean via_default)
{
  gchar *bare = NULL, *ver = NULL;

  model_key_split (key, &bare, &ver);
  modelmux_route_append_ref (s, label, bare, ver, via_default, gpu, FALSE);
  g_free (bare);
  g_free (ver);
}

/* Grouping bucket: the entry body (everything but "streams") is the KEY, so two
 * streams merge exactly when their whole routing state matches. Members are the
 * camera_ids sharing it. */
typedef struct
{
  GPtrArray *cams;              /* gchar* (owned) */
} ModelMuxRouteGroup;

static void
modelmux_route_group_free (gpointer p)
{
  ModelMuxRouteGroup *g = (ModelMuxRouteGroup *) p;
  if (!g)
    return;
  g_ptr_array_unref (g->cams);
  g_free (g);
}

static gint
modelmux_route_str_cmp (gconstpointer a, gconstpointer b)
{
  return g_strcmp0 (*(const gchar * const *) a, *(const gchar * const *) b);
}

/* Sort entries by their FIRST camera_id. The source is a hash table, so without
 * a total order two consecutive GETs would diff as churn with nothing changed. */
static gint
modelmux_route_entry_cmp (gconstpointer a, gconstpointer b, gpointer udata)
{
  GHashTable *groups = (GHashTable *) udata;
  ModelMuxRouteGroup *ga = g_hash_table_lookup (groups, *(const gchar * const *) a);
  ModelMuxRouteGroup *gb = g_hash_table_lookup (groups, *(const gchar * const *) b);

  if (!ga || !gb || !ga->cams->len || !gb->cams->len)
    return 0;
  return g_strcmp0 ((const gchar *) g_ptr_array_index (ga->cams, 0),
      (const gchar *) g_ptr_array_index (gb->cams, 0));
}

gchar *
modelmux_bin_route_json (ModelMuxBin * ib, const gchar * f_camera)
{
  GString *s;
  GHashTable *groups;           /* entry-body(gchar*) -> ModelMuxRouteGroup* */
  GHashTableIter it;
  gpointer k, v;
  /* PRESENT vs non-empty: a non-NULL f_camera is a filter, even when empty --
   * an empty one matches nothing rather than everything, so a malformed point
   * lookup fails closed instead of dumping the fleet. */
  gboolean filtered = (f_camera != NULL);
  guint i;

  if (!ib)
    return g_strdup ("{\"routes\":[],\"routing_revision\":0}");

  groups = g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
      modelmux_route_group_free);

  g_mutex_lock (&ib->lock);

  /* Tables not up yet (a GET can land before attach builds them): answer the
   * SHAPE a client expects -- an empty routing plane -- rather than NULL, which
   * the caller would report as "application did not answer" and turn into a 500
   * for what is really an idle pipeline.
   *
   * Tested UNDER ib->lock, not before it, exactly as modelmux_bin_status_json
   * does: these are heap pointers the teardown path destroys, so reading them
   * outside the lock is a race whose window the lock/unlock barrier in
   * modelmux_bin_free does NOT close -- that barrier only drains threads already
   * inside a locked section, not one that passed an unlocked check and has yet
   * to acquire. */
  if (!ib->stream_wiring || !ib->stream_cam_ids) {
    guint64 rev = modelmux_bin_revision_locked (ib);
    g_mutex_unlock (&ib->lock);
    g_hash_table_destroy (groups);
    /* the REVISION is still real even with no routes to show -- it is the CAS
     * token, and answering 0 here would hand a controller a number that loses
     * the next compare-and-swap for no reason. */
    return g_strdup_printf ("{\"routes\":[],\"routing_revision\":%"
        G_GUINT64_FORMAT "}", rev);
  }

  g_hash_table_iter_init (&it, ib->stream_wiring);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelMuxStreamEntry *e = (ModelMuxStreamEntry *) v;
    /* ONLY the routable camera_id -- never the display name in stream_names.
     * A stream the host gave no camera_id is skipped rather than labelled with
     * an identifier this API's own POST would reject with STREAM_UNKNOWN. */
    const gchar *cam = (const gchar *) g_hash_table_lookup (ib->stream_cam_ids, k);
    gboolean is_pass = (e->prim.active && e->prim.passthru);
    gboolean has_model = (e->prim.active && e->prim.model && !is_pass);
    gboolean has_shadow = (e->shad.active && e->shad.model);
    const ModelMuxPendingRouteNote *pn = NULL, *sn = NULL;
    gboolean has_pending;
    GString *body;
    ModelMuxRouteGroup *grp;

    if (!cam || !*cam)
      continue;
    /* strict point lookup: an id that does not resolve matches nothing */
    if (filtered && g_strcmp0 (cam, f_camera) != 0)
      continue;

    modelmux_status_pending_lanes (ib, e->stream_id, &pn, &sn);
    has_pending = (pn && pn->primary) || (sn && (sn->shadow || sn->clear_shadow))
        || (is_pass && e->prim.pending_model);

    body = g_string_new (NULL);
    /* --- alphabetical: model, pending, shadow, state --- */
    if (has_model) {
      modelmux_route_append_lane (body, ib, "model", &e->prim, e->stream_id);
      g_string_append_c (body, ',');
    }
    if (has_pending) {
      gboolean pfirst = TRUE;
      g_string_append (body, "\"pending\":{");
      if (pn && pn->primary) {
        modelmux_route_append_pending (body, "model", pn->primary, pn->p_gpu,
            pn->p_via_default);
        pfirst = FALSE;
      } else if (is_pass && e->prim.pending_model) {
        /* a passthrough awaiting promotion, with no newer route queued over it */
        modelmux_route_append_pending (body, "model", e->prim.pending_model,
            e->prim.req_gpu, e->prim.via_default);
        pfirst = FALSE;
      }
      if (sn && sn->shadow) {
        if (!pfirst)
          g_string_append_c (body, ',');
        modelmux_route_append_pending (body, "shadow", sn->shadow, sn->s_gpu,
            sn->s_via_default);
      } else if (sn && sn->clear_shadow) {
        /* null is MEANINGFUL: it is how the POST spells "drop the shadow", and
         * the read reuses that token. An omitted key means "leave it alone", so
         * collapsing the two would make the pending views identical. */
        if (!pfirst)
          g_string_append_c (body, ',');
        g_string_append (body, "\"shadow\":null");
      }
      g_string_append (body, "},");
    }
    if (has_shadow) {
      modelmux_route_append_lane (body, ib, "shadow", &e->shad, e->stream_id);
      g_string_append_c (body, ',');
    }
    /* `serving` keys off EITHER lane: a live shadow with a momentarily detached
     * primary is still inferring, and reporting it as passthrough (documented as
     * "no inference") would be the opposite of the truth. */
    g_string_append_printf (body, "\"state\":\"%s\"",
        (has_model || has_shadow) ? "serving"
        : (has_pending ? "pending" : "passthrough"));

    grp = g_hash_table_lookup (groups, body->str);
    if (!grp) {
      grp = g_new0 (ModelMuxRouteGroup, 1);
      grp->cams = g_ptr_array_new_with_free_func (g_free);
      g_hash_table_insert (groups, g_strdup (body->str), grp);
    }
    g_ptr_array_add (grp->cams, g_strdup (cam));
    g_string_free (body, TRUE);
  }

  s = g_string_new (NULL);
  g_string_append_c (s, '{');
  /* UNFILTERED ONLY: under ?stream= the fleet default is either redundant
   * (origin "default" -> the stream's model IS the default) or irrelevant
   * (origin "pinned" -> a default move cannot touch it). */
  if (!filtered) {
    GString *d = g_string_new (NULL);
    gboolean dfirst = TRUE;
    if (ib->primary_pool)
      modelmux_status_append_default_ref (d, "model",
          &ib->primary_pool->default_ref, &dfirst);
    if (ib->shadow_pool)
      modelmux_status_append_default_ref (d, "shadow",
          &ib->shadow_pool->default_ref, &dfirst);
    if (d->len)
      g_string_append_printf (s, "\"default\":{%s},", d->str);
    g_string_free (d, TRUE);
  }

  /* entries sorted by first camera_id; members sorted within each entry */
  {
    GPtrArray *keys = g_ptr_array_new ();
    g_hash_table_iter_init (&it, groups);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      g_ptr_array_sort (((ModelMuxRouteGroup *) v)->cams, modelmux_route_str_cmp);
      g_ptr_array_add (keys, k);
    }
    g_ptr_array_sort_with_data (keys, modelmux_route_entry_cmp, groups);

    gboolean rfirst = TRUE;
    g_string_append (s, "\"routes\":[");
    for (i = 0; i < keys->len; i++) {
      const gchar *body = (const gchar *) g_ptr_array_index (keys, i);
      ModelMuxRouteGroup *grp = g_hash_table_lookup (groups, body);
      guint c;
      if (!grp || !grp->cams->len)
        continue;               /* defensive: a bucket always has >= 1 member */
      /* separator tracked by a FLAG, not by the loop index: a skipped bucket
       * would leave index 0 unemitted and make the next entry lead with a comma
       * -- "routes":[,{...}] -- which is not valid JSON. */
      /* No process_mode here: routes[] is a PRIMARY-only plane (a secondary has no
       * per-stream routing and is listed under secondary_models[] instead), so every
       * entry has one shape and the schema is unchanged from before sgie support. */
      g_string_append_printf (s, "%s{%s,\"streams\":[", rfirst ? "" : ",", body);
      rfirst = FALSE;
      for (c = 0; c < grp->cams->len; c++) {
        gchar *ec = modelmux_json_escape ((const gchar *)
            g_ptr_array_index (grp->cams, c));
        g_string_append_printf (s, "%s\"%s\"", c ? "," : "", ec);
        g_free (ec);
      }
      g_string_append (s, "]}");
    }
    g_string_append_c (s, ']');
    g_ptr_array_free (keys, TRUE);
  }

  g_string_append_printf (s, ",\"routing_revision\":%" G_GUINT64_FORMAT "}",
      modelmux_bin_revision_locked (ib));

  g_mutex_unlock (&ib->lock);

  g_hash_table_destroy (groups);
  return g_string_free (s, FALSE);
}

/* Returns a newly allocated JSON string; caller frees with g_free.
 *
 * v1 shape (API_DESIGN.md 5.5): { "routing_revision": N,
 *   "default": {"model"?,"shadow"?},
 *   "models":  [ {"name", "versions":[{version,state,default_for?,serving,
 *                 config?,instances[]}]} ],
 *   "streams": [ {"camera_id","source_id","state","model"?,"shadow"?,
 *                 "pending"?,"throughput_fps"?} ] }
 * (the REST layer wraps this under the "model-status" response key).
 *
 * Filters JOIN (never orphan a view):
 *   - ?model=X[&version=N]  -> that model AND the streams routed to it
 *   - ?stream=S / ?source_id=N -> that stream AND the versions it uses
 * TODO(API_DESIGN.md): pending_routes[] exposure (deferred reroutes live at
 * the element control layer) and ?include=/?fields= section selection. */
gchar *
modelmux_bin_status_json (ModelMuxBin * ib, const gchar * f_model,
    const gchar * f_version, const gchar * f_stream, gint f_sid)
{
  GString *s, *streams_s;
  GHashTable *rows;             /* key "name@version" -> ModelMuxStatusRow* */
  GHashTable *used_keys = NULL; /* stream filter: keys the stream uses */
  GHashTableIter it;
  gpointer k, v;
  GPtrArray *names;
  gboolean first;
  gboolean perf_on;
  gboolean model_filter = (f_model && *f_model);
  gboolean version_filter = (f_version && *f_version);
  gboolean stream_filter = ((f_stream && *f_stream) || f_sid >= 0);
  guint i;

  if (!ib)
    return g_strdup ("{\"routing_revision\":0,\"models\":[],\"streams\":[]}");

  if (version_filter && !model_filter) {
    GST_WARNING ("model/status: version='%s' given without model -- ignoring "
        "the version (a version filter needs a model name)", f_version);
    version_filter = FALSE;
    f_version = NULL;
  }

  g_mutex_lock (&ib->lock);

  /* Tables gone (a status GET racing teardown, or landing before attach builds
   * them): answer the empty SHAPE rather than iterating a NULL table. Teardown
   * NULLs these under this same lock, so reaching here with NULL is the normal
   * shutdown ordering, not a bug. */
  if (!ib->stream_wiring || !ib->stream_names) {
    guint64 rev = modelmux_bin_revision_locked (ib);
    g_mutex_unlock (&ib->lock);
    /* real revision even in the empty shape -- see modelmux_bin_route_json */
    return g_strdup_printf ("{\"routing_revision\":%" G_GUINT64_FORMAT
        ",\"models\":[],\"streams\":[]}", rev);
  }

  perf_on = (ib->config && ib->config->attach_perf_metric);

  /* ---- streams[] first: a stream filter feeds the model join ---- */
  streams_s = g_string_new (NULL);
  if (stream_filter)
    used_keys = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  first = TRUE;
  g_hash_table_iter_init (&it, ib->stream_wiring);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelMuxStreamEntry *e = (ModelMuxStreamEntry *) v;
    const gchar *nm = (const gchar *) g_hash_table_lookup (ib->stream_names, k);
    const gchar *cid = (const gchar *) g_hash_table_lookup (ib->stream_cam_ids, k);
    gboolean is_pass = (e->prim.active && e->prim.passthru);
    gchar *enm;

    if (stream_filter) {
      gboolean match = (f_sid >= 0 && e->stream_id == (guint) f_sid);
      /* ?stream= accepts the ROUTABLE camera_id -- the same token model/routing
       * reports and the POST plane requires -- so an id read out of one endpoint
       * resolves in the other. The display name is still accepted because it is
       * what this filter has always taken and callers pass it, but the id is
       * tried FIRST: it is the identifier that round-trips. */
      if (!match && f_stream && *f_stream && g_strcmp0 (cid, f_stream) == 0)
        match = TRUE;
      if (!match && f_stream && *f_stream && g_strcmp0 (nm, f_stream) == 0)
        match = TRUE;
      if (!match)
        continue;
      if (e->prim.active && e->prim.model)
        g_hash_table_add (used_keys, g_strdup (e->prim.model));
      if (e->shad.active && e->shad.model)
        g_hash_table_add (used_keys, g_strdup (e->shad.model));
      /* PENDING targets join too. streams[] reports them, so a models[] that
       * omits them hands back a response referencing a version it never
       * describes -- the filtered view would orphan its own pending ref. Both
       * sources of intent count: a passthrough awaiting promotion, and a
       * serving stream whose accepted route is still queued behind a warm. */
      if (e->prim.pending_model)
        g_hash_table_add (used_keys, g_strdup (e->prim.pending_model));
      {
        const ModelMuxPendingRouteNote *pn, *sn;
        modelmux_status_pending_lanes (ib, e->stream_id, &pn, &sn);
        if (pn && pn->primary)
          g_hash_table_add (used_keys, g_strdup (pn->primary));
        if (sn && sn->shadow)
          g_hash_table_add (used_keys, g_strdup (sn->shadow));
      }
    }
    if (model_filter && !stream_filter) {
      /* model-filter JOIN: only the streams routed to that model */
      gboolean routed = FALSE;
      gchar *b = NULL, *ver2 = NULL;
      if (e->prim.active && e->prim.model) {
        model_key_split (e->prim.model, &b, &ver2);
        routed = (g_strcmp0 (b, f_model) == 0 &&
            (!version_filter || g_strcmp0 (ver2, f_version) == 0));
        g_free (b); g_free (ver2); b = ver2 = NULL;
      }
      if (!routed && e->shad.active && e->shad.model) {
        model_key_split (e->shad.model, &b, &ver2);
        routed = (g_strcmp0 (b, f_model) == 0 &&
            (!version_filter || g_strcmp0 (ver2, f_version) == 0));
        g_free (b); g_free (ver2); b = ver2 = NULL;
      }
      if (!routed && e->prim.pending_model) {
        /* a passthrough stream PENDING promotion to X is routed to X in intent --
         * the unfiltered view surfaces it as "pending"; the ?model=X JOIN must not
         * hide it (a reconciler would falsely conclude the route never applied). */
        model_key_split (e->prim.pending_model, &b, &ver2);
        routed = (g_strcmp0 (b, f_model) == 0 &&
            (!version_filter || g_strcmp0 (ver2, f_version) == 0));
        g_free (b); g_free (ver2); b = ver2 = NULL;
      }
      if (!routed) {
        /* Same argument one step later in the lifecycle: a SERVING stream with an
         * accepted-but-deferred route to X is routed to X in intent, and the
         * unfiltered view reports it as pending -- so ?model=X must not hide it
         * either. Either lane's deferred target counts, matching how the active
         * lanes above are both consulted. */
        const ModelMuxPendingRouteNote *pn, *sn;
        guint r;
        modelmux_status_pending_lanes (ib, e->stream_id, &pn, &sn);
        for (r = 0; !routed && r < 2; r++) {
          const gchar *key = r ? (sn ? sn->shadow : NULL)
              : (pn ? pn->primary : NULL);
          if (!key)
            continue;
          model_key_split (key, &b, &ver2);
          routed = (g_strcmp0 (b, f_model) == 0 &&
              (!version_filter || g_strcmp0 (ver2, f_version) == 0));
          g_free (b); g_free (ver2); b = ver2 = NULL;
        }
      }
      if (!routed)
        continue;
    }

    /* IDENTITY, split by what each field is FOR:
     *   camera_id   = the ROUTABLE id -- the token model/routing reports in
     *                 routes[].streams and the one a POST must name. Read from
     *                 stream_cam_ids so the two planes cannot disagree about who
     *                 a stream is; a client that takes this value and routes with
     *                 it gets the stream it asked for.
     *   camera_name = the display label, for humans and overlays only.
     *
     * They were ONE field carrying the display name, which made `camera_id` a
     * value its own POST plane rejects with STREAM_UNKNOWN whenever a deployment
     * gives a camera a name different from its id -- the common case.
     *
     * camera_id is OMITTED when the stream has no routable id, exactly as
     * model/routing omits such streams: absent is honest, whereas substituting
     * the display name is the mislabel this split exists to end. */
    if (cid && *cid) {
      enm = modelmux_json_escape (cid);
      g_string_append_printf (streams_s, "%s{\"camera_id\":\"%s\",", first ? "" : ",", enm);
      g_free (enm);
    } else {
      g_string_append_printf (streams_s, "%s{", first ? "" : ",");
    }
    enm = modelmux_json_escape (nm);
    g_string_append_printf (streams_s,
        "\"camera_name\":\"%s\",\"source_id\":%u,\"state\":\"%s\"",
        enm, e->stream_id, is_pass ? "passthrough" : "serving");
    g_free (enm);
    first = FALSE;

    if (!is_pass) {
      modelmux_status_append_ref (streams_s, ib, "model", &e->prim, e->stream_id,
          perf_on);
      modelmux_status_append_ref (streams_s, ib, "shadow", &e->shad, e->stream_id,
          perf_on);
    }
    /* ---- PENDING: shared by BOTH states, and the mirror OUTRANKS the tag ----
     *
     * A stream can hold two intents at once: an older promote tag (prim.pending_model,
     * set when it attached as passthrough onto a warming model) and a newer deferred
     * ROUTE queued after it. Reading the tag first -- which the passthrough branch used
     * to do exclusively -- reports the target the operator has already replaced.
     * Whichever intent is NEWER wins, and the mirror is only ever populated by a route
     * accepted after the tag was written.
     *
     * The tag remains the fallback: a passthrough stream with no queued route still
     * reports where it is headed, byte-identically to before. */
    {
      const ModelMuxPendingRouteNote *pn, *sn;
      modelmux_status_pending_lanes (ib, e->stream_id, &pn, &sn);
      /* A SERVING stream can ALSO have an accepted route parked on a warming
       * target. It keeps serving the model above until the switch lands, so the
       * intent rides ALONGSIDE "model" rather than replacing it -- the state is
       * still "serving", just with a known next step. Without this the POST
       * answered 202 and nothing observable ever changed, leaving a reconciler
       * unable to tell an in-flight route from one that was dropped. */
      if (pn || sn) {
        gboolean pfirst = TRUE;
        g_string_append (streams_s, ",\"pending\":{");
        if (pn)
          modelmux_status_append_pending_ref (streams_s, "model", pn->primary,
              pn->p_gpu, pn->p_via_default, &pfirst);
        if (sn)
          modelmux_status_append_pending_ref (streams_s, "shadow", sn->shadow,
              sn->s_gpu, sn->s_via_default, &pfirst);
        /* explicit null, NOT an omitted key: omitting it is how "leave the
         * shadow alone" is spelled, so a pending clear has to say null or the
         * two requests read identically while they wait. Same token the POST
         * uses ("shadow": null), so the read speaks the write's vocabulary. */
        if (sn && sn->clear_shadow) {
          g_string_append_printf (streams_s, "%s\"shadow\":null",
              pfirst ? "" : ",");
          pfirst = FALSE;
        }
        g_string_append_c (streams_s, '}');
      } else if (is_pass && e->prim.pending_model) {
        /* passthrough awaiting promotion, with no newer route queued over it:
         * surface the intended target, plus the intent that rides with it --
         * req_gpu (the placement the eventual promote must honour) and
         * via_default (whether a default switch would re-point this pending
         * route). Both are recorded by modelmux_bin_set_pending_model; without
         * them a reader cannot tell a pinned deferred route from an unpinned
         * one, nor predict a sweep. */
        gboolean pfirst = TRUE;
        g_string_append (streams_s, ",\"pending\":{");
        modelmux_status_append_pending_ref (streams_s, "model",
            e->prim.pending_model, e->prim.req_gpu, e->prim.via_default, &pfirst);
        g_string_append_c (streams_s, '}');
      }
    }
    if (perf_on) {
      gdouble tput = 0, l = 0;
      if (is_pass) {
        if (e->stream_id < MM_ACCT_MAX)
          tput = ib->thru_perf[e->stream_id].fps;
      } else {
        ModelBin *pbin = modelmux_perf_find_bin_for_source (ib->primary_pool,
            e->stream_id);
        if (pbin && pbin->perf)
          modelmux_perf_get_source (pbin->perf, (gint) e->stream_id, &tput, &l);
      }
      g_string_append_printf (streams_s, ",\"throughput_fps\":%.2f", tput);
    }
    g_string_append_c (streams_s, '}');
  }

  /* ---- models[]: merge pools + limbo into (name, version) rows ---- */
  rows = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);
  modelmux_status_collect_pool (rows, ib->primary_pool, 0);
  modelmux_status_collect_pool (rows, ib->shadow_pool, 1);
  if (ib->limbo_models) {
    g_hash_table_iter_init (&it, ib->limbo_models);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelBin *model_bin = (ModelBin *) v;
      ModelMuxStatusRow *row = g_hash_table_lookup (rows, model_bin->key);
      if (!row) {
        row = g_new0 (ModelMuxStatusRow, 1);
        g_hash_table_insert (rows, g_strdup (model_bin->key), row);
      }
      row->limbo = model_bin;
    }
  }

  /* distinct names, honoring the filters/joins */
  names = g_ptr_array_new ();
  g_hash_table_iter_init (&it, rows);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelMuxStatusRow *row = (ModelMuxStatusRow *) v;
    ModelBin *rep = row->prim ? row->prim : (row->shad ? row->shad : row->limbo);
    gboolean seen = FALSE;
    if (model_filter && g_strcmp0 (rep->name, f_model) != 0)
      continue;
    if (version_filter &&
        g_strcmp0 (model_version_str (rep), f_version) != 0)
      continue;
    if (used_keys && !g_hash_table_contains (used_keys, (const gchar *) k))
      continue;                 /* stream-filter join: only versions it uses */
    for (i = 0; i < names->len && !seen; i++)
      seen = (g_strcmp0 ((const gchar *) g_ptr_array_index (names, i),
              rep->name) == 0);
    if (!seen)
      g_ptr_array_add (names, rep->name);
  }

  s = g_string_new (NULL);
  g_string_append_printf (s, "{\"routing_revision\":%" G_GUINT64_FORMAT,
      modelmux_bin_revision_locked (ib));
  /* UNFILTERED SNAPSHOTS ONLY. A filtered view promises to contain nothing
   * outside the filter -- the existing contract that ?model=X returns only X
   * and the streams routed to it. The runtime default is fleet-wide state and
   * is almost never the filtered model, so emitting it unconditionally would
   * leak an unrelated model name into every narrowed response (and does break
   * test_12's "filter excludes the default model" assertions).
   * model/routing's GET follows the same rule: it drops default{} under
   * ?stream= for the same reason (the default is redundant or irrelevant for a
   * single stream), so the two views stay consistent. */
  if (!model_filter && !version_filter && !stream_filter)
    modelmux_status_append_default (s, ib);
  g_string_append (s, ",\"models\":[");
  for (i = 0; i < names->len; i++) {
    const gchar *name = (const gchar *) g_ptr_array_index (names, i);
    gchar *enm = modelmux_json_escape (name);
    gboolean vfirst = TRUE;
    /* process_mode tags WHICH STAGE hosts this model. It lives on the model entry (not
     * at the top level) because models[] is FLAT across stages: a merged response from a
     * primary + a secondary is one array, and each entry says where it runs. */
    g_string_append_printf (s, "%s{\"name\":\"%s\",\"process_mode\":%d,\"versions\":[",
        i ? "," : "", enm,
        ib->config ? ib->config->process_mode : MM_PROCESS_MODE_PGIE);
    g_free (enm);
    g_hash_table_iter_init (&it, rows);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelMuxStatusRow *row = (ModelMuxStatusRow *) v;
      ModelBin *rep = row->prim ? row->prim
          : (row->shad ? row->shad : row->limbo);
      if (g_strcmp0 (rep->name, name) != 0)
        continue;
      if (version_filter &&
          g_strcmp0 (model_version_str (rep), f_version) != 0)
        continue;
      if (used_keys && !g_hash_table_contains (used_keys, (const gchar *) k))
        continue;
      modelmux_status_append_version (s, ib, row, perf_on, &vfirst);
    }
    g_string_append (s, "]}");
  }
  g_string_append (s, "],\"streams\":[");
  g_string_append_len (s, streams_s->str, streams_s->len);
  g_string_append (s, "]");

  /* ---- gpus[] (attach-perf-metric=1 only): WHOLE-device stats via NVML for
   * every device hosting a live instance, plus the instances resident there --
   * attribution by CO-LOCATION (device load + who runs on it), never
   * per-kernel accounting. Degrades to util/vram -1 when NVML is absent. */
  if (perf_on) {
    gint devs[16];
    guint ndev = 0, di;
    ModelPool *pools[2] = { ib->primary_pool, ib->shadow_pool };
    guint pi;
    g_string_append (s, ",\"gpus\":[");
    for (pi = 0; pi < 2; pi++) {
      if (!pools[pi] || !pools[pi]->models)
        continue;
      g_hash_table_iter_init (&it, pools[pi]->models);
      while (g_hash_table_iter_next (&it, &k, &v)) {
        ModelBin *model_bin;
        for (model_bin = (ModelBin *) v; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
          gint g = model_bin->gpu;
          gboolean dup = FALSE;
          if (g < 0)
            g = modelmux_config_version_gpu (ib->config, model_bin->name,
                model_version_str (model_bin));
          if (g < 0)
            g = model_bin->cfg_gpu;
          if (g < 0)
            g = 0;
          for (di = 0; di < ndev && !dup; di++)
            dup = (devs[di] == g);
          if (!dup) {
            if (ndev < G_N_ELEMENTS (devs))
              devs[ndev++] = g;
            else
              GST_WARNING ("model/status gpus[]: more than %u distinct devices -- "
                  "device %d omitted from the report", (guint) G_N_ELEMENTS (devs), g);
          }
        }
      }
    }
    for (di = 0; di < ndev; di++) {
      guint util = 0, used = 0, total = 0;
      gboolean have = modelmux_gpu_stats (devs[di], &util, &used, &total);
      gboolean mfirst = TRUE;
      g_string_append_printf (s, "%s{\"id\":%d", di ? "," : "", devs[di]);
      if (have)
        g_string_append_printf (s,
            ",\"util_pct\":%u,\"vram_used_mb\":%u,\"vram_total_mb\":%u",
            util, used, total);
      else
        g_string_append (s,
            ",\"util_pct\":-1,\"vram_used_mb\":-1,\"vram_total_mb\":-1");
      g_string_append (s, ",\"instances\":[");
      for (pi = 0; pi < 2; pi++) {
        if (!pools[pi] || !pools[pi]->models)
          continue;
        g_hash_table_iter_init (&it, pools[pi]->models);
        while (g_hash_table_iter_next (&it, &k, &v)) {
          ModelBin *model_bin;
          for (model_bin = (ModelBin *) v; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
            gint g = model_bin->gpu;
            if (g < 0)
              g = modelmux_config_version_gpu (ib->config, model_bin->name,
                  model_version_str (model_bin));
            if (g < 0)
              g = model_bin->cfg_gpu;
            if (g < 0)
              g = 0;
            if (g != devs[di])
              continue;
            {
              gchar *en = modelmux_json_escape (model_bin->name);
              gchar *ev = modelmux_json_escape (model_version_str (model_bin));
              g_string_append_printf (s,
                  "%s{\"name\":\"%s\",\"version\":\"%s\",\"role\":\"%s\","
                  "\"instance\":%u,\"streams\":%u}",
                  mfirst ? "" : ",", en, ev, pools[pi]->role, model_bin->inst,
                  model_bin->num_streams);
              g_free (en);
              g_free (ev);
              mfirst = FALSE;
            }
          }
        }
      }
      g_string_append (s, "]}");
    }
    g_string_append (s, "]");
  }
  g_string_append (s, "}");

  g_mutex_unlock (&ib->lock);
  g_ptr_array_free (names, TRUE);
  g_hash_table_destroy (rows);
  if (used_keys)
    g_hash_table_destroy (used_keys);
  g_string_free (streams_s, TRUE);
  return g_string_free (s, FALSE);
}

/* ================================================================== */
/* Pipeline-wide routing revision (see MODELMUX_CONTEXT_TYPE_ROUTING)   */
/* ================================================================== */

struct _ModelMuxRevShared
{
  gint     refcount;            /**< atomic; one ref per stage holding it  */
  GMutex   lock;                /**< guards @rev: GLib has no 64-bit atomic */
  guint64  rev;
};

/* Boxed rather than GObject: it only has to survive a trip through a GstStructure,
 * and boxed copy/free map straight onto ref/unref, so the context holds a real
 * reference and the counter outlives whichever stage created it. */
G_DEFINE_BOXED_TYPE (ModelMuxRevShared, modelmux_rev_shared,
    modelmux_rev_shared_ref, modelmux_rev_shared_unref);

ModelMuxRevShared *
modelmux_rev_shared_new (void)
{
  ModelMuxRevShared *rev = g_new0 (ModelMuxRevShared, 1);

  rev->refcount = 1;
  g_mutex_init (&rev->lock);
  return rev;
}

ModelMuxRevShared *
modelmux_rev_shared_ref (ModelMuxRevShared * rev)
{
  if (rev)
    g_atomic_int_inc (&rev->refcount);
  return rev;
}

void
modelmux_rev_shared_unref (ModelMuxRevShared * rev)
{
  if (!rev || !g_atomic_int_dec_and_test (&rev->refcount))
    return;
  g_mutex_clear (&rev->lock);
  g_free (rev);
}

guint64
modelmux_rev_shared_get (ModelMuxRevShared * rev)
{
  guint64 v;

  if (!rev)
    return 0;
  g_mutex_lock (&rev->lock);
  v = rev->rev;
  g_mutex_unlock (&rev->lock);
  return v;
}

guint64
modelmux_rev_shared_bump (ModelMuxRevShared * rev)
{
  guint64 v;

  if (!rev)
    return 0;
  g_mutex_lock (&rev->lock);
  v = ++rev->rev;
  g_mutex_unlock (&rev->lock);
  return v;
}

/* Read the revision with ib->lock ALREADY held (the JSON builders hold it for the
 * whole snapshot, and taking it again would deadlock). */
static guint64
modelmux_bin_revision_locked (ModelMuxBin * ib)
{
  /* the shared counter is authoritative once negotiated; ib->routing_revision is
   * only the pre-context fallback, and the two are never both written. */
  return ib->rev_shared ? modelmux_rev_shared_get (ib->rev_shared)
      : ib->routing_revision;
}

void
modelmux_bin_set_rev_shared (ModelMuxBin * ib, ModelMuxRevShared * rev)
{
  ModelMuxRevShared *old;

  if (!ib)
    return;
  g_mutex_lock (&ib->lock);
  old = ib->rev_shared;
  ib->rev_shared = modelmux_rev_shared_ref (rev);
  /* carry any bumps this bin took BEFORE the context arrived onto the shared
   * counter, so adopting it can only move the revision forward -- a controller
   * must never see it go backwards. */
  if (rev && ib->routing_revision) {
    guint64 carry = ib->routing_revision;
    ib->routing_revision = 0;
    while (carry--)
      modelmux_rev_shared_bump (rev);
  }
  g_mutex_unlock (&ib->lock);
  modelmux_rev_shared_unref (old);
}

guint64
modelmux_bin_routing_revision (ModelMuxBin * ib)
{
  guint64 v;

  if (!ib)
    return 0;
  g_mutex_lock (&ib->lock);
  v = modelmux_bin_revision_locked (ib);
  g_mutex_unlock (&ib->lock);
  return v;
}

void
modelmux_bin_bump_routing_revision (ModelMuxBin * ib)
{
  if (!ib)
    return;
  g_mutex_lock (&ib->lock);
  if (ib->rev_shared)
    modelmux_rev_shared_bump (ib->rev_shared);
  else
    ib->routing_revision++;
  g_mutex_unlock (&ib->lock);
}

void
modelmux_pending_route_note_free (gpointer note)
{
  ModelMuxPendingRouteNote *n = (ModelMuxPendingRouteNote *) note;

  if (!n)
    return;
  g_free (n->src);
  g_free (n->primary);
  g_free (n->shadow);
  g_free (n);
}

void
modelmux_bin_set_pending_routes (ModelMuxBin * ib, GPtrArray * notes)
{
  GPtrArray *old;

  if (!ib) {
    /* takes ownership either way, so a publish racing teardown frees rather
     * than leaks the notes it was handed. */
    if (notes)
      g_ptr_array_unref (notes);
    return;
  }
  g_mutex_lock (&ib->lock);
  old = ib->pending_routes;
  ib->pending_routes = notes;
  g_mutex_unlock (&ib->lock);
  /* freed OUTSIDE the lock: once unlinked the old array is private to this call,
   * and freeing it under the lock would stall the status path for nothing. */
  if (old)
    g_ptr_array_unref (old);
}

/* ================================================================== */
/* Cross-stage merge                                                   */
/*                                                                     */
/* A pipeline can hold several nvmodelmux stages in series (a primary   */
/* plus one or more secondaries). The REST host sends ONE downstream    */
/* query, and the first stage to answer it would otherwise end the      */
/* propagation -- so each stage FORWARDS the query on before writing    */
/* its own body, then folds the downstream answer into it. The reply    */
/* the host reads is therefore the whole pipeline, assembled upstream-  */
/* first, with no host-side fan-out and no shared state between stages. */
/*                                                                     */
/* Merge rule: models[]/streams[]/routes[] are FLAT arrays that simply  */
/* concatenate (each entry carries its own process_mode). Scalars --    */
/* routing_revision, default{} -- stay THIS stage's: only a primary     */
/* has a routing plane, and it is the upstream stage.                   */
/*                                                                      */
/* routing_revision needs no merge RULE because it needs no merge: every */
/* stage reads the ONE counter they share through the routing            */
/* GstContext, so all the bodies already carry the same value and        */
/* keeping this stage's is keeping the pipeline's. That is exactly why   */
/* it is shared -- with a per-stage counter, "keep the upstream one"     */
/* would report a revision that a route applied at a SECONDARY never     */
/* moved, and the client's next if_revision would guard nothing.         */
/* ================================================================== */

/* Append downstream's @arr entries into @mine's array of the same name. Both are
 * parsed JSON objects; a missing or non-array member on either side is a no-op. */
static void
modelmux_status_concat_array (JsonObject * mine, JsonObject * theirs,
    const gchar * member)
{
  JsonArray *dst, *src;
  guint i, n;

  if (!json_object_has_member (theirs, member)
      || !JSON_NODE_HOLDS_ARRAY (json_object_get_member (theirs, member)))
    return;
  /* CREATE the member when only the downstream stage has it (secondary_models[] is
   * contributed by secondaries alone) -- otherwise the merge would silently drop a
   * key just because the upstream stage has nothing to add to it. */
  if (!json_object_has_member (mine, member)
      || !JSON_NODE_HOLDS_ARRAY (json_object_get_member (mine, member)))
    json_object_set_array_member (mine, member, json_array_new ());
  dst = json_object_get_array_member (mine, member);
  src = json_object_get_array_member (theirs, member);
  n = json_array_get_length (src);
  for (i = 0; i < n; i++)
    json_array_add_element (dst, json_node_copy (json_array_get_element (src, i)));
}

/* Fold @theirs (a downstream stage's body) into @mine. Returns a newly-allocated
 * merged body, or a copy of @mine when @theirs is absent/unparseable -- a broken
 * downstream answer must not cost us our own. */
static gchar *
modelmux_status_merge (const gchar * mine, const gchar * theirs)
{
  JsonParser *pm, *pt;
  JsonObject *om, *ot;
  JsonGenerator *gen;
  gchar *out;

  if (!theirs || !*theirs)
    return g_strdup (mine);

  pm = json_parser_new ();
  pt = json_parser_new ();
  if (!json_parser_load_from_data (pm, mine, -1, NULL)
      || !json_parser_load_from_data (pt, theirs, -1, NULL)
      || !JSON_NODE_HOLDS_OBJECT (json_parser_get_root (pm))
      || !JSON_NODE_HOLDS_OBJECT (json_parser_get_root (pt))) {
    GST_WARNING ("model/status: a stage answered with a body that is not a JSON "
        "object -- reporting this stage only");
    g_object_unref (pm);
    g_object_unref (pt);
    return g_strdup (mine);
  }
  om = json_node_get_object (json_parser_get_root (pm));
  ot = json_node_get_object (json_parser_get_root (pt));

  modelmux_status_concat_array (om, ot, "models");
  modelmux_status_concat_array (om, ot, "streams");
  modelmux_status_concat_array (om, ot, "routes");
  modelmux_status_concat_array (om, ot, "secondary_models");
  modelmux_status_concat_array (om, ot, "gpus");

  gen = json_generator_new ();
  json_generator_set_root (gen, json_parser_get_root (pm));
  out = json_generator_to_data (gen, NULL);
  g_object_unref (gen);
  g_object_unref (pm);
  g_object_unref (pt);
  return out;
}

typedef struct
{
  GstQuery *q;                  /**< the original request, filters and all */
  gboolean  is_routing;
  gchar    *acc;                /**< merged bodies of every peer visited so far */
} ModelMuxGetCtx;

/* Visitor: fold ONE peer stage's body into the accumulator.
 *
 * Unlike the control plane this does NOT stop at the first answer -- a GET is a
 * census, so every stage contributes. Merging pairwise as we go reuses
 * modelmux_status_merge unchanged and keeps only one intermediate body alive. */
static gboolean
modelmux_status_get_visit (GstNvModelMux * self, GstNvModelMux * peer, gpointer udata)
{
  ModelMuxGetCtx *ctx = (ModelMuxGetCtx *) udata;
  const gchar *json = NULL;
  GstQuery *ans = NULL;

  (void) self;
  if (!modelmux_stage_ask_peer (peer, ctx->q, &ans))
    return TRUE;

  if (ctx->is_routing)
    gst_nvquery_model_routing_parse_response (ans, &json);
  else
    gst_nvquery_model_status_parse_response (ans, &json);

  if (json && *json) {
    if (!ctx->acc) {
      ctx->acc = g_strdup (json);
    } else {
      gchar *merged = modelmux_status_merge (ctx->acc, json);
      g_free (ctx->acc);
      ctx->acc = merged;
    }
  }
  gst_query_unref (ans);
  return TRUE;                          /* every stage answers a GET */
}

/* Collect the bodies of every OTHER stage in the pipeline, merged into one (caller
 * frees; NULL when no peer answered). The query carries the SAME filters, so each
 * stage applies them to its own models -- a filtered view stays filtered across the
 * whole pipeline.
 *
 * The pipeline walk rather than the src pad, for the reason spelled out in
 * modelmux_control_ask_downstream: sibling secondaries hang off a tee, and a
 * forwarded query stops at the first branch that answers, so pad-following silently
 * reports a subset of the stages -- the worst possible failure for a census. */
static gchar *
modelmux_status_ask_downstream (GstNvModelMux * self, GstQuery * q,
    gboolean is_routing)
{
  ModelMuxGetCtx ctx = { q, is_routing, NULL };

  /* already someone's fan-out: report this stage only (the caller merges). */
  if (modelmux_stage_query_is_delegated (q))
    return NULL;

  modelmux_foreach_peer_stage (self, modelmux_status_get_visit, &ctx);
  return ctx.acc;
}

/* Stage-agnostic reader of the GET plane. The udata is the ELEMENT (not a bin), so
 * one probe answers from whichever bin this stage built: the ModelMuxBin at a primary,
 * the ModelMuxSgieBin at a secondary. A stage answers only for its OWN models -- the
 * host merges the per-stage responses. */
G_GNUC_INTERNAL GstPadProbeReturn
modelmux_status_query_probe (GstPad * pad, GstPadProbeInfo * info, gpointer udata)
{
  GstQuery *q = GST_PAD_PROBE_INFO_QUERY (info);
  GstNvModelMux *self = GST_NVMODELMUX (udata);
  ModelMuxBin *ib = self ? self->modelmux_bin : NULL;
  ModelMuxSgieBin *sg = self ? self->modelmux_sgie_bin : NULL;
  gchar *json, *down, *merged;
  (void) pad;

  /* model/routing: the ROUTING plane, answered whole by this element -- the REST
   * host forwards the body untouched, exactly as it does for model/status. */
  if (q && gst_nvquery_is_model_routing (q)) {
    const gchar *f_camera = NULL;
    gboolean f_on = FALSE;
    gchar *rjson;
    gst_nvquery_model_routing_parse_request (q, &f_camera, &f_on);
    rjson = sg ? modelmux_sgie_route_json (sg)
        : modelmux_bin_route_json (ib, f_on ? (f_camera ? f_camera : "") : NULL);
    down = modelmux_status_ask_downstream (self, q, TRUE);
    merged = modelmux_status_merge (rjson, down);
    gst_nvquery_model_routing_set_response (q, merged);
    g_free (merged);
    g_free (down);
    g_free (rjson);
    return GST_PAD_PROBE_HANDLED;
  }

  if (!q || !gst_nvquery_is_model_status (q))
    return GST_PAD_PROBE_OK;

  {
    const gchar *f_model = NULL, *f_version = NULL, *f_stream = NULL;
    gint f_sid = -1;
    gst_nvquery_model_status_parse_request (q, &f_model, &f_version, &f_stream, &f_sid);
    json = sg ? modelmux_sgie_status_json (sg, f_model, f_version)
        : modelmux_bin_status_json (ib, f_model, f_version, f_stream, f_sid);
  }
  down = modelmux_status_ask_downstream (self, q, FALSE);
  merged = modelmux_status_merge (json, down);
  gst_nvquery_model_status_set_response (q, merged);
  g_free (merged);
  g_free (down);
  g_free (json);
  return GST_PAD_PROBE_HANDLED;
}
