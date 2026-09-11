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
 * nvds_eval_engine.cpp
 * ====================
 * C++ evaluation engine — per-model-pair context design.
 *
 * Each unique model_name seen in process_frame() owns an isolated
 * ModelPairContext that holds:
 *
 *   - EvalAccumulator : (role, model_name) -> per-class stats for this epoch
 *   - GTIndex         : loaded from the model's effective gt-file
 *   - epoch_frames    : Primary-frame counter (Shadow frames do NOT count)
 *   - sources         : source_id -> camera_id of the streams carrying this
 *                       model (from per-frame sensor info; targeted routing)
 *   - ModelConfig cfg : effective per-model thresholds / margins
 *
 * Epoch boundaries fire independently per model_name.  Promotion compares
 * Primary vs Shadow within the same context — there is no cross-model
 * comparison and no assumption about how many models are running.
 */

#include "nvds_eval_engine.h"
#include "nvds_eval_config.h"
#include "nvds_eval_gt_loader.h"
#include "nvds_eval_accumulator.h"
#include "nvds_eval_report.h"

#include "nvdsmeta.h"        /* NvDsObjectMeta */
#include "nvll_osd_struct.h" /* NvOSD_RectParams */

#include <gst/gst.h>
#include <glib.h>

GST_DEBUG_CATEGORY_EXTERN (gst_nv_infer_eval_debug);
#define GST_CAT_DEFAULT gst_nv_infer_eval_debug

#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>

/* ------------------------------------------------------------------ */
/* ModelPairContext                                                    */
/* ------------------------------------------------------------------ */

struct ModelPairContext
{
  ModelConfig      cfg;            /* effective config for this model_name   */
  GTIndex          gt_index;       /* (camera_id, frame_num) -> GTBoxes      */
  bool             gt_loaded = false;

  EvalAccumulator  accum;          /* (role, model_name) -> per-class stats  */
  uint32_t         epoch_frames = 0; /* incremented only by Primary frames   */

  /* Streams that carry this model pair: source_id -> {camera_id, liveness},
   * populated lazily on each frame from the per-frame sensor info the chain
   * function already reads (frame_meta->sensorInfo_meta.sensor_id -- the SAME
   * stable stream/add id nvmodelmux publishes as "camera_id" in model/status
   * and accepts in stream/route). Used for targeted stream/route at the epoch
   * boundary WITHOUT any status round-trip. A source whose sensor_id was
   * never seen maps to "" and is skipped (warned) at route time.
   *
   * LIVENESS: every frame stamps its entry with the pair's monotonic Primary
   * frame counter (frames_total), and the route scope includes ONLY entries
   * seen within the epoch being scored (last_seen >= epoch_start). A removed
   * stream's stale camera_id would otherwise ride every future route POST and
   * make nvmodelmux's ATOMIC admission reject the WHOLE request
   * (STREAM_UNKNOWN) each epoch -- permanently blocking promotion. Entries
   * not seen for one full epoch are pruned at the rollover, so dead ids can
   * never accumulate toward MAX_TRACKED_SOURCES either; a camera re-added
   * under a new source_id simply claims a fresh entry. Bounded: one entry per
   * distinct source_id (ids are recycled by the source bin);
   * MAX_TRACKED_SOURCES is a hard cap against a pathological id space. */
  struct SourceTrack
  {
    std::string cam;            /* camera_id ("" = sensor_id never seen)     */
    uint64_t    last_seen = 0;  /* frames_total stamp of this source's latest
                                 * frame (liveness window marker)            */
  };
  std::map<uint32_t, SourceTrack> sources;
  uint64_t         frames_total = 0; /* monotonic Primary frame counter; never
                                      * reset (the liveness clock)           */
  uint64_t         epoch_start  = 0; /* frames_total at the current epoch's
                                      * start (liveness window floor)        */
  static constexpr size_t MAX_TRACKED_SOURCES = 4096;
};

/* ------------------------------------------------------------------ */
/* NvdsEvalEngine                                                      */
/* ------------------------------------------------------------------ */

struct NvdsEvalEngine
{
  /* --- raw property values set via GObject setters (backward compat) --- */
  std::string prop_gt_file;
  guint       prop_eval_interval  = 1000;
  gfloat      prop_iou_threshold  = 0.5f;
  gfloat      prop_min_ap_margin  = 0.02f;
  gfloat      prop_kpi_ap50       = 0.0f;
  gfloat      prop_kpi_precision  = 0.0f;
  gfloat      prop_kpi_recall     = 0.0f;
  guint       prop_promotion_mode      = 0;
  gboolean    prop_show_gt_overlay = FALSE;
  gboolean    prop_show_ap_table   = FALSE;
  gboolean    prop_cumulative          = TRUE;
  gboolean    prop_exclude_undetected  = TRUE;
  std::string prop_exclude_classes; /* comma-separated, set via GObject property */
  std::string prop_report_file;
  std::string prop_config_file;

  /* --- resolved config, built at prepare() --- */
  EvalConfig cfg;

  /* --- per-model-name pair contexts, created lazily on first frame --- */
  std::map<std::string, ModelPairContext> pairs;

  /* --- upstream nvmodelmux element (borrowed ref, Mode B only) --- */
  GstElement *mm_element = nullptr;

  /* --- helpers --- */
  ModelPairContext &get_or_create_context (const std::string &model_name,
                                           GstElement        *self);
  void fire_epoch (const std::string    &model_name,
                   ModelPairContext     &ctx,
                   uint32_t             epoch_frames_snap,
                   GstElement           *self);
};

/* ------------------------------------------------------------------ */
/* get_or_create_context                                              */
/* ------------------------------------------------------------------ */

ModelPairContext &
NvdsEvalEngine::get_or_create_context (const std::string &model_name,
                                        GstElement        *self)
{
  auto it = pairs.find (model_name);
  if (it != pairs.end ())
    return it->second;

  /* First time this model_name is seen — create and initialise context */
  ModelPairContext &ctx = pairs[model_name];
  ctx.cfg = cfg.effective_model_config (model_name);

  if (ctx.cfg.gt_file.empty ()) {
    GST_INFO_OBJECT (self,
        "nvinfereval: [%s] no gt-file — running relative comparison only",
        model_name.c_str ());
    ctx.gt_loaded = false;
  } else {
    bool ok = gt_load (ctx.cfg.gt_file, ctx.gt_index);
    if (!ok) {
      GST_WARNING_OBJECT (self,
          "nvinfereval: [%s] failed to load GT from '%s' — "
          "evaluation will proceed without ground truth",
          model_name.c_str (), ctx.cfg.gt_file.c_str ());
      ctx.gt_loaded = false;
    } else {
      GST_INFO_OBJECT (self,
          "nvinfereval: [%s] loaded %zu GT (camera,frame) entries from '%s'",
          model_name.c_str (), ctx.gt_index.size (),
          ctx.cfg.gt_file.c_str ());
      ctx.gt_loaded = true;
    }
  }

  return ctx;
}

/* ------------------------------------------------------------------ */
/* mm_rest_stream_route                                               */
/* ------------------------------------------------------------------ */
/* Promote / reroute by POSTing to the nvmodelmux v1 ROUTING plane.
 * This replaces the in-process "update-routing" GObject signal, so promotion
 * no longer depends on discovering the upstream element handle (mm_element)
 * — it works regardless of pipeline topology, as long as the REST server URL
 * is configured.
 *
 *   POST <base_url>/api/v1/stream/route
 *   {"key":"stream","value":{"routes":[{"streams":"all",
 *     "model":{"name":"<name>","version":"<ver>"},
 *     "shadow":{"name":...,"version":...} | null}]}}
 *
 * model/shadow are structured {"name","version"} refs; "shadow": null clears
 * the shadow role (a shadow_model of "none" maps to it). The route SCOPE is
 * the eval's own streams: the v1 'streams' member takes camera_id strings,
 * and the eval resolves them LOCALLY from the per-frame sensor info it
 * tracked all epoch (ModelPairContext::sources) -- no GET model/status
 * round-trip, so the streaming thread never blocks on a status fetch and the
 * mapping can never miss because of a serializer formatting difference.
 * Routing "all" would promote streams belonging to UNRELATED A/B pairs, so
 * the route always carries the explicit camera_id list. Returns true on
 * HTTP 2xx. The one remaining `curl` POST is synchronous with --max-time
 * (bounded), and fires at most once per epoch win. */
/* Minimal JSON string escaping for values placed into the stream/route body.
 * model_name / model_version originate from InferenceProvenanceMeta and could in
 * principle contain characters ('"', '\\', control chars) that would otherwise
 * produce a malformed request body. */
static std::string
json_escape_value (const std::string &s)
{
  std::string o;
  o.reserve (s.size ());
  for (unsigned char c : s) {
    if      (c == '"')  o += "\\\"";
    else if (c == '\\') o += "\\\\";
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20)  { char b[8]; snprintf (b, sizeof (b), "\\u%04x", c); o += b; }
    else                o += (char) c;
  }
  return o;
}

/* Sanitize a rest-url config value into "<scheme>://host:port" (no trailing
 * '/', whitespace/inline-'#'-comment tolerant). Empty result = invalid. */
static std::string
mm_rest_clean_base_url (const std::string &base_url)
{
  std::string url = base_url;
  size_t b = url.find_first_not_of (" \t");
  if (b == std::string::npos) return std::string ();
  size_t e = url.find_first_of (" \t#", b);
  url = url.substr (b, (e == std::string::npos) ? std::string::npos : e - b);
  if (!url.empty () && url.back () == '/') url.pop_back ();
  return url;
}

static bool
mm_rest_stream_route (const std::string &base_url,
                      const std::map<uint32_t, std::string> &sources,
                      const std::string &primary_model,
                      const std::string &primary_version,
                      const std::string &shadow_model,
                      const std::string &shadow_version,
                      GstElement        *self)
{
  if (base_url.empty ())
    return false;

  /* Build the stream/route body to the v1 schema nvds_rest_server expects
   * (nvds_model_parse.cpp, nvds_rest_stream_route_parse): the envelope
   * { "key":"stream", "value": {"routes":[...]} } with ONE route covering
   * exactly the eval's streams. Refs carry SEPARATE name + version strings;
   * a shadow of "none" becomes the explicit clear ("shadow": null); an empty
   * shadow leaves the role unchanged (member omitted). The serving model is
   * mandatory here: the routing plane cannot clear it ("model": null is
   * passthrough, NOT_IMPLEMENTED) and the eval never asks for that. */
  if (primary_model.empty () || primary_model == "none") {
    GST_WARNING_OBJECT (self,
        "nvinfereval: stream/route needs a serving model ref — got '%s'",
        primary_model.c_str ());
    return false;
  }

  /* ---- route scope: THIS eval's streams, never someone else's ----
   * `sources` is the source_id -> camera_id map this pair accumulated from
   * per-frame sensor info (no status fetch). Route the explicit camera_id
   * list; with multiple A/B pairs in one pipeline, "streams":"all" would
   * clobber the routes of unrelated pairs. An empty/unresolvable scope must
   * NEVER widen: refuse instead (retried next epoch). */
  std::string streams_json;
  {
    std::string list;
    size_t resolved = 0;
    if (sources.empty ()) {
      GST_ERROR_OBJECT (self,
          "nvinfereval: no source-ids tracked for this pair — refusing to "
          "route (an unscoped route would hit every stream)");
      return false;
    }
    for (const auto &kv : sources) {
      if (kv.second.empty ()) {
        /* the frames of this source carried no sensor_id -- nvmodelmux knows
         * it under a name the eval never saw, so it cannot be routed */
        GST_WARNING_OBJECT (self,
            "nvinfereval: source-id %u carried no per-frame sensor_id "
            "(camera_id unknown) — leaving it out of the route", kv.first);
        continue;
      }
      if (!list.empty ()) list += ",";
      list += "\"" + json_escape_value (kv.second) + "\"";
      resolved++;
    }
    if (resolved == 0) {
      GST_ERROR_OBJECT (self,
          "nvinfereval: none of the %zu tracked source-ids has a known "
          "camera_id — routing SKIPPED (will not widen to \"all\")",
          sources.size ());
      return false;
    }
    streams_json = "[" + list + "]";
  }

  auto ref = [](const std::string &name, const std::string &ver) {
    return "{\"name\":\"" + json_escape_value (name) +
           "\",\"version\":\"" + json_escape_value (ver) + "\"}";
  };
  std::string route = "{\"streams\":" + streams_json + ",\"model\":" +
      ref (primary_model, primary_version);
  if (shadow_model == "none")
    route += ",\"shadow\":null";
  else if (!shadow_model.empty ())
    route += ",\"shadow\":" + ref (shadow_model, shadow_version);
  route += "}";

  std::string body = "{\"key\":\"stream\",\"value\":{\"routes\":[" + route + "]}}";

  /* Be forgiving of leading/trailing whitespace or an accidental inline
   * "# comment" in the config value (GKeyFile keeps inline comments as part of
   * the value). A URL has no spaces, so cut at the first whitespace or '#'. */
  std::string url = mm_rest_clean_base_url (base_url);
  if (url.empty ()) return false;
  url += "/api/v1/stream/route";

  gchar *q_url  = g_shell_quote (url.c_str ());
  gchar *q_body = g_shell_quote (body.c_str ());
  gchar *cmd = g_strdup_printf (
      "curl -sS --max-time 5 -o /dev/null -w '%%{http_code}' "
      "-X POST %s -H 'Content-Type: application/json' -d %s",
      q_url, q_body);

  /* Minimal: just echo the curl command being issued. */
  printf ("        curl -X POST %s -d %s\n", url.c_str (), body.c_str ());
  fflush (stdout);

  gchar  *out = nullptr, *err = nullptr;
  gint    status = 0;
  GError *gerr = nullptr;
  bool    ok = false;

  if (g_spawn_command_line_sync (cmd, &out, &err, &status, &gerr)) {
    const long http = out ? (long) g_ascii_strtoll (out, nullptr, 10) : 0;
    ok = (http >= 200 && http < 300);
    if (ok)
      GST_INFO_OBJECT (self, "nvinfereval: stream/route accepted (http=%ld)", http);
    else
      GST_WARNING_OBJECT (self,
          "nvinfereval: stream/route POST failed (http=%ld, curl-status=%d) %s",
          http, status, (err && *err) ? err : "");
  } else {
    GST_WARNING_OBJECT (self, "nvinfereval: could not run curl (%s) — is it installed?",
        gerr ? gerr->message : "unknown");
  }

  g_clear_error (&gerr);
  g_free (out); g_free (err); g_free (cmd); g_free (q_url); g_free (q_body);
  return ok;
}

/* ------------------------------------------------------------------ */
/* fire_epoch                                                         */
/* ------------------------------------------------------------------ */

void
NvdsEvalEngine::fire_epoch (const std::string &model_name,
                             ModelPairContext   &ctx,
                             uint32_t           epoch_frames_snap,
                             GstElement         *self)
{
  std::vector<ModelMetrics> metrics =
      compute_metrics (ctx.accum, cfg.global.iou_threshold,
                       cfg.global.exclude_classes, cfg.global.exclude_undetected);

  /* Warn about models with GT mismatch */
  for (const auto &mm : metrics) {
    if (ctx.gt_loaded && !mm.gt_available) {
      GST_WARNING_OBJECT (self,
          "nvinfereval: [%s] role=%s model=%s@%s saw %u frames but found "
          "NO ground-truth matches — check gt-file camera_id/frame_num alignment",
          model_name.c_str (), mm.role.c_str (),
          mm.model_name.c_str (), mm.model_version.c_str (),
          mm.frame_count);
    }
  }

  bool        promoted       = false;
  std::string promoted_model;

  if (cfg.global.promotion_mode != 2) {  /* 2 = report-only, skip all promotion logic */
    const ModelMetrics *primary = nullptr;
    const ModelMetrics *shadow  = nullptr;

    for (const auto &mm : metrics) {
      if (mm.role == "Primary") primary = &mm;
      if (mm.role == "Shadow")  shadow  = &mm;
    }

    if (primary && shadow) {
      /* Skip promotion if GT is loaded but one side has no GT coverage */
      bool gt_ok = !ctx.gt_loaded ||
                   (primary->gt_available && shadow->gt_available);

      if (!gt_ok) {
        GST_WARNING_OBJECT (self,
            "nvinfereval: [%s] skipping promotion — GT not available for "
            "one or both models this epoch", model_name.c_str ());
      } else {
        bool ap_wins  = shadow->ap50 > primary->ap50 + ctx.cfg.min_ap_margin;
        bool kpi_ap50 = (ctx.cfg.kpi_ap50 <= 0.0f ||
                         shadow->ap50 >= ctx.cfg.kpi_ap50);
        bool kpi_prec = (ctx.cfg.kpi_precision <= 0.0f ||
                         shadow->mean_precision >= ctx.cfg.kpi_precision);
        bool kpi_rec  = (ctx.cfg.kpi_recall <= 0.0f ||
                         shadow->mean_recall >= ctx.cfg.kpi_recall);

        /* Dynamic A/B presence check (re-evaluated every epoch): a real
         * comparison needs BOTH roles to have a substantial, comparable sample
         * this window. After a promote/clear (or swap) the other role is gone,
         * or only a transitional sliver straddling the reroute boundary remains
         * — in window mode that sliver has very few frames. Reject when either
         * role is absent or the smaller sample is < half the larger, so a tiny
         * transitional window cannot drive a (flip-flopping) decision. */
        const uint32_t pf = primary->frame_count;
        const uint32_t sf = shadow->frame_count;
        const uint32_t lo = (pf < sf) ? pf : sf;
        const uint32_t hi = (pf > sf) ? pf : sf;
        const bool both_present = (lo > 0 && 2u * lo >= hi);

        if (ap_wins && kpi_ap50 && kpi_prec && kpi_rec && both_present) {
          /* Common action flag: "swap" keeps A/B running with roles flipped
           * (old primary becomes the new shadow); anything else => "promote"
           * (default: winner becomes primary, shadow role cleared). */
          const bool do_swap = (cfg.global.promotion_mode == 1);  /* 0=promote, 1=swap, 2=report-only */
          const char *action = do_swap ? "SWAP" : "PROMOTE";

          if (cfg.global.rest_url.empty ()) {
            GST_WARNING_OBJECT (self,
                "nvinfereval: [%s] shadow '%s@%s' won (AP50=%.3f) but "
                "rest-url is not set — no %s performed. Add "
                "rest-url=http://<host>:<port> to [eval].",
                model_name.c_str (), shadow->model_name.c_str (),
                shadow->model_version.c_str (), shadow->ap50, action);
          } else {
            /* One stream/route REST POST per pair; the route is scoped to
             * THIS pair's streams via the locally tracked source_id ->
             * camera_id map (built from per-frame sensor info -- no status
             * round-trip on the streaming thread). For both actions
             * the winner (shadow) becomes the serving model; swap also keeps
             * the old primary as the new shadow, promote clears the shadow
             * role ("none" -> "shadow": null). */
            const std::string new_shadow_model = do_swap ? primary->model_name
                                                         : std::string ("none");
            const std::string new_shadow_ver   = do_swap ? primary->model_version
                                                         : std::string ();

            /* Route scope: only sources SEEN during the epoch just scored
             * (last_seen >= epoch_start). A stale entry from a removed
             * stream would carry a dead camera_id and fail the route's
             * atomic admission WHOLE (STREAM_UNKNOWN) -- every epoch. */
            std::map<uint32_t, std::string> live_sources;
            for (const auto &kv : ctx.sources)
              if (kv.second.last_seen >= ctx.epoch_start)
                live_sources[kv.first] = kv.second.cam;

            printf ("\n    >>> %s: %s@%s -> primary\n", action,
                    shadow->model_name.c_str (), shadow->model_version.c_str ());
            fflush (stdout);

            bool ok = mm_rest_stream_route (cfg.global.rest_url, live_sources,
                shadow->model_name, shadow->model_version,  /* -> new primary  */
                new_shadow_model,   new_shadow_ver,          /* swap: old prim; */
                self);                                       /* promote: none   */
            if (ok) {
              promoted       = true;
              promoted_model = shadow->model_name + "@" + shadow->model_version;
              GST_INFO_OBJECT (self,
                  "nvinfereval: [%s] %s applied: '%s@%s' (AP50=%.3f) is now "
                  "primary over '%s@%s' (AP50=%.3f) on %zu stream(s) via REST",
                  model_name.c_str (), action,
                  shadow->model_name.c_str (), shadow->model_version.c_str (),
                  shadow->ap50,
                  primary->model_name.c_str (), primary->model_version.c_str (),
                  primary->ap50,
                  live_sources.size ());
            } else {
              GST_WARNING_OBJECT (self,
                  "nvinfereval: [%s] %s REST POST failed — "
                  "roles left unchanged this epoch", model_name.c_str (), action);
            }
          }
        } else if (!both_present) {
          GST_DEBUG_OBJECT (self,
              "nvinfereval: [%s] no valid A/B this epoch — primary/shadow not "
              "both substantially present (primary=%u frames, shadow=%u frames); "
              "likely post-promote/swap transition — skipping",
              model_name.c_str (), pf, sf);
        } else {
          GST_DEBUG_OBJECT (self,
              "nvinfereval: [%s] shadow did not win epoch "
              "(shadow AP50=%.3f primary AP50=%.3f margin=%.3f "
              "ap_wins=%d kpi_ap50=%d kpi_prec=%d kpi_rec=%d)",
              model_name.c_str (),
              shadow->ap50, primary->ap50, ctx.cfg.min_ap_margin,
              (int)ap_wins, (int)kpi_ap50, (int)kpi_prec, (int)kpi_rec);
        }
      }
    } else if (!shadow) {
      GST_DEBUG_OBJECT (self,
          "nvinfereval: [%s] no Shadow role observed this epoch — "
          "single-model evaluation, no promotion", model_name.c_str ());
    }
  }

  post_eval_result (self, model_name, metrics, epoch_frames_snap,
                    promoted, promoted_model, cfg.global.report_file,
                    cfg.global.cumulative, cfg.global.show_ap_table);
}

/* ================================================================== */
/* C API                                                              */
/* ================================================================== */

extern "C" {

NvdsEvalEngine *
nvds_eval_engine_new (void)
{
  return new NvdsEvalEngine ();
}

void
nvds_eval_engine_free (NvdsEvalEngine *eng)
{
  delete eng;
}

/* Property setters — store raw values; applied at prepare() */
void nvds_eval_engine_set_gt_file       (NvdsEvalEngine *e, const gchar *v) { e->prop_gt_file        = v ? v : ""; }
void nvds_eval_engine_set_eval_interval (NvdsEvalEngine *e, guint v)        { e->prop_eval_interval   = v; }
void nvds_eval_engine_set_iou_threshold (NvdsEvalEngine *e, gfloat v)       { e->prop_iou_threshold   = v; }
void nvds_eval_engine_set_min_ap_margin (NvdsEvalEngine *e, gfloat v)       { e->prop_min_ap_margin   = v; }
void nvds_eval_engine_set_kpi_ap50      (NvdsEvalEngine *e, gfloat v)       { e->prop_kpi_ap50        = v; }
void nvds_eval_engine_set_kpi_precision (NvdsEvalEngine *e, gfloat v)       { e->prop_kpi_precision   = v; }
void nvds_eval_engine_set_kpi_recall    (NvdsEvalEngine *e, gfloat v)       { e->prop_kpi_recall      = v; }
void     nvds_eval_engine_set_promotion_mode   (NvdsEvalEngine *e, guint v)    { e->prop_promotion_mode      = v; }
guint    nvds_eval_engine_get_promotion_mode   (NvdsEvalEngine *e)             { return e->cfg.global.promotion_mode; }
void nvds_eval_engine_set_show_gt_overlay  (NvdsEvalEngine *e, gboolean v) { e->prop_show_gt_overlay = v; }
gboolean nvds_eval_engine_get_show_gt_overlay (NvdsEvalEngine *e)          { return e->cfg.global.show_gt_overlay; }
void nvds_eval_engine_set_show_ap_table    (NvdsEvalEngine *e, gboolean v) { e->prop_show_ap_table   = v; }
gboolean nvds_eval_engine_get_show_ap_table   (NvdsEvalEngine *e)          { return e->cfg.global.show_ap_table; }
void nvds_eval_engine_set_cumulative          (NvdsEvalEngine *e, gboolean v) { e->prop_cumulative = v; }
gboolean nvds_eval_engine_get_cumulative      (NvdsEvalEngine *e)             { return e->cfg.global.cumulative; }
void nvds_eval_engine_set_exclude_undetected  (NvdsEvalEngine *e, gboolean v) { e->prop_exclude_undetected = v; }
gboolean nvds_eval_engine_get_exclude_undetected (NvdsEvalEngine *e)          { return e->cfg.global.exclude_undetected; }
void nvds_eval_engine_set_exclude_classes     (NvdsEvalEngine *e, const gchar *v) { e->prop_exclude_classes = v ? v : ""; }
void nvds_eval_engine_set_report_file   (NvdsEvalEngine *e, const gchar *v) { e->prop_report_file     = v ? v : ""; }
void nvds_eval_engine_set_config_file   (NvdsEvalEngine *e, const gchar *v) { e->prop_config_file     = v ? v : ""; }

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_prepare                                           */
/* ------------------------------------------------------------------ */

gboolean
nvds_eval_engine_prepare (NvdsEvalEngine *eng,
                           GstElement     *self,
                           GstElement     *mm_element)
{
  eng->mm_element = mm_element;
  eng->pairs.clear ();

  /* Step 1: seed EvalConfig from GObject property values */
  eng->cfg = EvalConfig ();
  eng->cfg.global.eval_interval    = eng->prop_eval_interval;
  eng->cfg.global.iou_threshold    = eng->prop_iou_threshold;
  eng->cfg.global.promotion_mode          = eng->prop_promotion_mode;
  eng->cfg.global.show_gt_overlay     = eng->prop_show_gt_overlay;
  eng->cfg.global.show_ap_table       = eng->prop_show_ap_table;
  eng->cfg.global.cumulative          = eng->prop_cumulative;
  eng->cfg.global.exclude_undetected  = eng->prop_exclude_undetected;
  /* Seed exclude_classes from property (config file will add more below) */
  if (!eng->prop_exclude_classes.empty ()) {
    gchar **toks = g_strsplit (eng->prop_exclude_classes.c_str (), ",", -1);
    for (gint t = 0; toks && toks[t]; ++t) {
      gchar *tr = g_strstrip (toks[t]);
      if (tr && tr[0]) eng->cfg.global.exclude_classes.insert (tr);
    }
    g_strfreev (toks);
  }
  eng->cfg.global.report_file      = eng->prop_report_file;
  eng->cfg.global.default_gt_file  = eng->prop_gt_file;

  /* Truncate the report file at the start of each new run so stale entries
   * from previous runs are not mixed with the current run's epoch results. */
  if (!eng->cfg.global.report_file.empty ()) {
    std::ofstream trunc (eng->cfg.global.report_file, std::ios::trunc);
    if (!trunc.is_open ())
      GST_WARNING_OBJECT (self, "nvinfereval: cannot truncate report file '%s'",
                          eng->cfg.global.report_file.c_str ());
  }

  /* Seed a global ModelConfig fallback from the single-model properties */
  ModelConfig global_model_defaults;
  global_model_defaults.min_ap_margin = eng->prop_min_ap_margin;
  global_model_defaults.kpi_ap50      = eng->prop_kpi_ap50;
  global_model_defaults.kpi_precision = eng->prop_kpi_precision;
  global_model_defaults.kpi_recall    = eng->prop_kpi_recall;
  /* Register as the catch-all fallback under the empty model_name key
   * so effective_model_config() finds it for any model without a section. */
  eng->cfg.models[""] = global_model_defaults;

  /* Step 2: overlay config file values (config file wins for keys it sets) */
  if (!eng->prop_config_file.empty ()) {
    if (!eval_config_load (eng->prop_config_file, eng->cfg)) {
      GST_WARNING_OBJECT (self,
          "nvinfereval: config-file '%s' could not be loaded — "
          "falling back to GObject property values",
          eng->prop_config_file.c_str ());
    } else {
      GST_INFO_OBJECT (self,
          "nvinfereval: loaded config from '%s'",
          eng->prop_config_file.c_str ());
    }
  }

  GST_INFO_OBJECT (self,
      "nvinfereval: prepared — eval_interval=%u iou_threshold=%.2f "
      "promotion-mode=%u(%s) rest-url='%s' accumulation=%s "
      "default_gt='%s' mode=%s",
      eng->cfg.global.eval_interval,
      eng->cfg.global.iou_threshold,
      eng->cfg.global.promotion_mode,
      eng->cfg.global.promotion_mode == 2 ? "off" :
          eng->cfg.global.promotion_mode == 1 ? "swap" : "promote",
      eng->cfg.global.rest_url.c_str (),
      eng->cfg.global.cumulative ? "cumulative" : "window",
      eng->cfg.global.default_gt_file.c_str (),
      mm_element ? "B (multi-model A/B)" : "A (single model)");

  return TRUE;
}

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_process_frame                                     */
/* ------------------------------------------------------------------ */

void
nvds_eval_engine_process_frame (NvdsEvalEngine *eng,
                                 GstElement     *self,
                                 guint           source_id,
                                 guint           frame_num,
                                 const gchar    *role,
                                 const gchar    *model_name,
                                 const gchar    *model_version,
                                 const gchar    *stream_name,
                                 GList          *obj_meta_list)
{
  const std::string role_str  = role         ? role          : "Primary";
  const std::string mname     = model_name   ? model_name    : "";
  const std::string mver      = model_version ? model_version : "";

  /* Lazily create context for this model_name on first encounter */
  ModelPairContext &ctx = eng->get_or_create_context (mname, self);

  /* Track source_id -> camera_id for the epoch-boundary stream/route scope.
   * stream_name is the per-frame sensor_id (frame_meta->sensorInfo_meta) --
   * the SAME stable stream/add id nvmodelmux routes by; an absent sensor_id
   * is recorded as "" (kept for the stream count, skipped at route time).
   * Every frame refreshes the entry's liveness stamp (frames_total), so the
   * route scope / rollover prune can tell live streams from removed ones.
   * Runs on the single streaming thread only, like all engine state. */
  if (ctx.sources.size () < ModelPairContext::MAX_TRACKED_SOURCES ||
      ctx.sources.count (source_id)) {
    ModelPairContext::SourceTrack &tr = ctx.sources[source_id];
    if (stream_name && stream_name[0] != '\0')
      tr.cam = stream_name;              /* refresh: re-added ids keep current */
    tr.last_seen = ctx.frames_total;     /* seen within the current epoch */
  }

  /* camera_id for GT lookup: prefer stream_name, fall back to source_id */
  const std::string cam_id =
      (stream_name && stream_name[0] != '\0')
      ? std::string (stream_name)
      : std::to_string (source_id);

  /* GT lookup — empty when GT unavailable or frame not in index */
  static const std::vector<GTBox> empty_gt;
  GTKey key {cam_id, frame_num};
  const std::vector<GTBox> &gt_boxes =
      (ctx.gt_loaded && ctx.gt_index.count (key))
      ? ctx.gt_index.at (key)
      : empty_gt;

  /* Extract predictions from NvDsObjectMeta (no heap allocation when empty) */
  std::vector<std::string> labels;
  std::vector<float>       confs, x1s, y1s, x2s, y2s;

  /* Reserve up front to avoid repeated reallocation as detections are pushed. */
  const guint n_obj = g_list_length (obj_meta_list);
  if (n_obj) {
    labels.reserve (n_obj);
    confs.reserve (n_obj);
    x1s.reserve (n_obj);
    y1s.reserve (n_obj);
    x2s.reserve (n_obj);
    y2s.reserve (n_obj);
  }

  for (GList *l = obj_meta_list; l; l = l->next) {
    NvDsObjectMeta *om = static_cast<NvDsObjectMeta *> (l->data);
    if (!om) continue;

    const float left   = om->rect_params.left;
    const float top    = om->rect_params.top;
    const float width  = om->rect_params.width;
    const float height = om->rect_params.height;

    labels.emplace_back (om->obj_label[0] != '\0'
        ? std::string (om->obj_label) : std::to_string (om->class_id));
    confs.push_back  ((om->confidence >= 0.0f) ? om->confidence : 0.0f);
    x1s  .push_back  (left);
    y1s  .push_back  (top);
    x2s  .push_back  (left + width);
    y2s  .push_back  (top  + height);
  }

  ctx.accum.add_frame (role_str, mname, mver,
                        labels, confs, x1s, y1s, x2s, y2s, gt_boxes);

  /* Epoch counter incremented only on Primary frames.
   * This ensures the epoch always spans exactly eval_interval Primary
   * frames regardless of whether a Shadow is present or how many models
   * share this pipeline. */
  if (role_str == "Primary") {
    ++ctx.epoch_frames;
    ++ctx.frames_total;                  /* liveness clock: monotonic, never reset */

    if (ctx.epoch_frames >= eng->cfg.global.eval_interval) {
      const uint32_t epoch_snap = ctx.epoch_frames;
      ctx.epoch_frames = 0;

      eng->fire_epoch (mname, ctx, epoch_snap, self);

      /* EPOCH ROLLOVER: prune sources not seen during the epoch just scored
       * (one full epoch of silence == the stream is gone; its id is dead or
       * will be recycled), then start the next liveness window. Keeps dead
       * ids from ever accumulating toward MAX_TRACKED_SOURCES. */
      for (auto it = ctx.sources.begin (); it != ctx.sources.end ();) {
        if (it->second.last_seen < ctx.epoch_start)
          it = ctx.sources.erase (it);
        else
          ++it;
      }
      ctx.epoch_start = ctx.frames_total;

      /* In window mode reset after each epoch; in cumulative mode keep
       * accumulating so each report covers all frames since pipeline start. */
      if (!eng->cfg.global.cumulative)
        ctx.accum.reset ();
    }
  }
}

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_flush                                             */
/* ------------------------------------------------------------------ */
/* Force a final ("forced epoch") report for every model context that still
 * holds frames since its last eval_interval boundary. Meant to be called on
 * EOS so the trailing partial epoch is not lost: in window mode it would
 * otherwise be discarded at the next reset; in cumulative mode the definitive
 * end-of-stream totals would never be emitted/saved. Each accumulated frame
 * was already evaluated via add_frame() in process_frame() — this only emits
 * the report covering them. Contexts with no pending frames are skipped (their
 * last boundary report already covered everything), so it is safe to call more
 * than once. */
void
nvds_eval_engine_flush (NvdsEvalEngine *eng, GstElement *self)
{
  if (!eng)
    return;
  for (auto &kv : eng->pairs) {
    ModelPairContext &ctx = kv.second;
    if (ctx.epoch_frames == 0)
      continue;                       /* nothing new since the last epoch */
    const uint32_t epoch_snap = ctx.epoch_frames;
    ctx.epoch_frames = 0;
    eng->fire_epoch (kv.first, ctx, epoch_snap, self);
    if (!eng->cfg.global.cumulative)
      ctx.accum.reset ();
  }
}

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_attach_gt_overlay                                 */
/* ------------------------------------------------------------------ */

void
nvds_eval_engine_attach_gt_overlay (NvdsEvalEngine *eng,
                                     GstElement     *self,
                                     NvDsBatchMeta  *batch_meta,
                                     NvDsFrameMeta  *frame_meta,
                                     const gchar    *stream_name,
                                     guint           frame_num,
                                     const gchar    *model_name)
{
  if (!batch_meta || !frame_meta)
    return;

  const std::string mname = model_name ? model_name : "";
  auto it = eng->pairs.find (mname);
  if (it == eng->pairs.end () || !it->second.gt_loaded)
    return;

  const std::string cam_id =
      (stream_name && stream_name[0] != '\0')
      ? std::string (stream_name)
      : std::to_string (frame_meta->source_id);

  GTKey key { cam_id, (uint32_t) frame_num };
  const GTIndex &idx = it->second.gt_index;
  auto git = idx.find (key);
  if (git == idx.end () || git->second.empty ())
    return;

  const std::vector<GTBox> &boxes = git->second;

  /* Fill display metas, MAX_ELEMENTS_IN_DISPLAY_META rects per slot */
  NvDsDisplayMeta *dmeta = nullptr;
  guint            slot  = MAX_ELEMENTS_IN_DISPLAY_META; /* force new alloc */

  for (const GTBox &b : boxes) {
    if (slot >= MAX_ELEMENTS_IN_DISPLAY_META) {
      dmeta = nvds_acquire_display_meta_from_pool (batch_meta);
      if (!dmeta) break;
      dmeta->num_rects  = 0;
      dmeta->num_labels = 0;
      nvds_add_display_meta_to_frame (frame_meta, dmeta);
      slot = 0;
    }

    /* Green rectangle (solid border, no fill) */
    NvOSD_RectParams &rp = dmeta->rect_params[slot];
    rp.left        = (guint) b.x1;
    rp.top         = (guint) b.y1;
    rp.width       = (guint) (b.x2 - b.x1);
    rp.height      = (guint) (b.y2 - b.y1);
    rp.border_width = 2;
    rp.border_color = { 0.0, 1.0, 0.0, 0.7 };   /* RGBA green, 70% opacity */
    rp.has_bg_color = 0;

    /* Small label above the box: "GT:<class>" */
    NvOSD_TextParams &tp  = dmeta->text_params[slot];
    tp.display_text       = g_strdup_printf ("GT:%s", b.class_label.c_str ());
    tp.x_offset           = (guint) b.x1 + 4;
    tp.y_offset           = (guint) ((b.y1 + b.y2) / 2.0f);
    tp.font_params.font_name  = (gchar *) "Serif";
    tp.font_params.font_size  = 5;
    tp.font_params.font_color = { 0.0, 1.0, 0.0, 1.0 };
    tp.set_bg_clr         = 0;

    dmeta->num_rects++;
    dmeta->num_labels++;
    ++slot;
  }

  (void) self;  /* available for future GST_DEBUG calls */
}

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_reset                                             */
/* ------------------------------------------------------------------ */

void
nvds_eval_engine_reset (NvdsEvalEngine *eng)
{
  eng->pairs.clear ();
  eng->mm_element = nullptr;
}

} /* extern "C" */
