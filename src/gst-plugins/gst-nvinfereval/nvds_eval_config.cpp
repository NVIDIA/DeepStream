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
 * nvds_eval_config.cpp
 * ====================
 * GKeyFile-based INI loader for nvinfereval plugin configuration.
 *
 * Loading strategy:
 *   - Caller pre-fills EvalConfig with GObject property defaults.
 *   - eval_config_load() overlays only the keys present in the file.
 *   - Per-model sections add per-model overrides; absent keys keep defaults.
 */

#include "nvds_eval_config.h"

#include <glib.h>
#include <string>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Section / key names                                                 */
/* ------------------------------------------------------------------ */

static constexpr const gchar *SECT_EVAL          = "eval";
static constexpr const gchar *KEY_EXCL_CLASSES   = "exclude-classes";
static constexpr const gchar *MODEL_PREFIX       = "model-";
static constexpr gsize        MODEL_PREFIX_LEN   = 6; /* strlen("model-") */

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/* Read a string key if present; otherwise leave @out unchanged. */
static void
read_string (GKeyFile *kf, const gchar *group, const gchar *key,
             std::string &out)
{
  GError *err = nullptr;
  gchar  *val = g_key_file_get_string (kf, group, key, &err);
  if (err) {
    if (err->code != G_KEY_FILE_ERROR_KEY_NOT_FOUND &&
        err->code != G_KEY_FILE_ERROR_GROUP_NOT_FOUND) {
      g_warning ("nvinfereval config: [%s] %s: %s", group, key, err->message);
    }
    g_error_free (err);
    return;
  }
  out = val ? val : "";
  g_free (val);
}

/* Read a boolean key if present.
 * Accepts true/false/yes/no (GKeyFile native) AND 1/0 integers. */
static void
read_bool (GKeyFile *kf, const gchar *group, const gchar *key,
           gboolean &out)
{
  if (!g_key_file_has_key (kf, group, key, nullptr)) return;

  GError *err = nullptr;
  gboolean val = g_key_file_get_boolean (kf, group, key, &err);
  if (!err) { out = val; return; }
  g_error_free (err); err = nullptr;

  /* GLib boolean failed — try integer (1 = true, 0 = false) */
  gint ival = g_key_file_get_integer (kf, group, key, &err);
  if (!err) { out = (ival != 0); return; }

  g_warning ("nvinfereval config: [%s] %s: cannot parse as boolean (use true/false/1/0)",
             group, key);
  g_error_free (err);
}

/* Read a uint key if present. */
static void
read_uint (GKeyFile *kf, const gchar *group, const gchar *key, guint &out)
{
  GError *err = nullptr;
  if (!g_key_file_has_key (kf, group, key, nullptr)) return;
  gint val = g_key_file_get_integer (kf, group, key, &err);
  if (err) {
    g_warning ("nvinfereval config: [%s] %s: %s", group, key, err->message);
    g_error_free (err);
    return;
  }
  if (val < 1) {
    g_warning ("nvinfereval config: [%s] %s must be >= 1 (got %d), ignored",
               group, key, val);
    return;
  }
  out = (guint)val;
}

/* Read a float key if present. */
static void
read_float (GKeyFile *kf, const gchar *group, const gchar *key, gfloat &out)
{
  GError *err = nullptr;
  if (!g_key_file_has_key (kf, group, key, nullptr)) return;
  gdouble val = g_key_file_get_double (kf, group, key, &err);
  if (err) {
    g_warning ("nvinfereval config: [%s] %s: %s", group, key, err->message);
    g_error_free (err);
    return;
  }
  out = (gfloat)val;
}

/* ------------------------------------------------------------------ */
/* EvalConfig::effective_model_config                                  */
/* ------------------------------------------------------------------ */

ModelConfig
EvalConfig::effective_model_config (const std::string &model_name) const
{
  /*
   * Three-level priority (lowest → highest):
   *   1. Hardcoded struct defaults
   *   2. Catch-all entry (key="") — populated from GObject properties at
   *      prepare() time so that single-model property values propagate to
   *      all named models seen at runtime.
   *   3. Per-model entry (key=model_name) — from a [model-<name>] section
   *      in the config file; fully overrides the catch-all for every field
   *      in that section.
   */

  /* Level 1: hardcoded defaults */
  ModelConfig eff;
  eff.gt_file       = global.default_gt_file;
  eff.min_ap_margin = 0.02f;
  eff.kpi_ap50      = 0.0f;
  eff.kpi_precision = 0.0f;
  eff.kpi_recall    = 0.0f;

  /* Level 2: catch-all from GObject properties */
  auto base_it = models.find ("");
  if (base_it != models.end ()) {
    const ModelConfig &base = base_it->second;
    eff.min_ap_margin = base.min_ap_margin;
    eff.kpi_ap50      = base.kpi_ap50;
    eff.kpi_precision = base.kpi_precision;
    eff.kpi_recall    = base.kpi_recall;
    if (!base.gt_file.empty ())
      eff.gt_file = base.gt_file;
  }

  /* Level 3: per-model section from config file */
  if (!model_name.empty ()) {
    auto it = models.find (model_name);
    if (it != models.end ()) {
      const ModelConfig &mc = it->second;
      if (!mc.gt_file.empty ()) eff.gt_file       = mc.gt_file;
      eff.min_ap_margin = mc.min_ap_margin;
      eff.kpi_ap50      = mc.kpi_ap50;
      eff.kpi_precision = mc.kpi_precision;
      eff.kpi_recall    = mc.kpi_recall;
    }
  }

  return eff;
}

/* ------------------------------------------------------------------ */
/* eval_config_load                                                    */
/* ------------------------------------------------------------------ */

bool
eval_config_load (const std::string &path, EvalConfig &cfg)
{
  GKeyFile *kf  = g_key_file_new ();
  GError   *err = nullptr;

  if (!g_key_file_load_from_file (kf, path.c_str (),
          G_KEY_FILE_NONE, &err)) {
    g_warning ("nvinfereval: cannot load config '%s': %s",
               path.c_str (), err ? err->message : "unknown error");
    if (err) g_error_free (err);
    g_key_file_free (kf);
    return false;
  }

  /* ---- [eval] section ---- */
  if (g_key_file_has_group (kf, SECT_EVAL)) {
    read_uint   (kf, SECT_EVAL, "eval-interval",    cfg.global.eval_interval);
    read_float  (kf, SECT_EVAL, "iou-threshold",    cfg.global.iou_threshold);
    read_bool   (kf, SECT_EVAL, "show-gt-overlay",  cfg.global.show_gt_overlay);
    read_bool   (kf, SECT_EVAL, "show-ap-table",    cfg.global.show_ap_table);
    read_bool   (kf, SECT_EVAL, "cumulative",           cfg.global.cumulative);
    read_bool   (kf, SECT_EVAL, "exclude-undetected",  cfg.global.exclude_undetected);

    /* exclude-classes: comma-separated list of class labels to skip in mean */
    std::string excl_str;
    read_string (kf, SECT_EVAL, KEY_EXCL_CLASSES, excl_str);
    if (!excl_str.empty ()) {
      gchar **tokens = g_strsplit (excl_str.c_str (), ",", -1);
      for (gint t = 0; tokens && tokens[t]; ++t) {
        gchar *trimmed = g_strstrip (tokens[t]);
        if (trimmed && trimmed[0])
          cfg.global.exclude_classes.insert (trimmed);
      }
      g_strfreev (tokens);
    }
    read_string (kf, SECT_EVAL, "report-file",      cfg.global.report_file);
    read_string (kf, SECT_EVAL, "default-gt-file",  cfg.global.default_gt_file);
    read_string (kf, SECT_EVAL, "rest-url",       cfg.global.rest_url);
    read_uint   (kf, SECT_EVAL, "promotion-mode",  cfg.global.promotion_mode);
  }

  /* ---- [model-<name>] sections ---- */
  gsize  n_groups = 0;
  gchar **groups  = g_key_file_get_groups (kf, &n_groups);

  for (gsize i = 0; i < n_groups; ++i) {
    const gchar *grp = groups[i];

    if (g_strcmp0 (grp, SECT_EVAL) == 0) continue;

    if (strncmp (grp, MODEL_PREFIX, MODEL_PREFIX_LEN) != 0) {
      g_warning ("nvinfereval config: unknown group [%s] — ignored", grp);
      continue;
    }

    std::string model_name (grp + MODEL_PREFIX_LEN);
    if (model_name.empty ()) {
      g_warning ("nvinfereval config: group [%s] has empty model name — ignored", grp);
      continue;
    }

    ModelConfig &mc = cfg.models[model_name]; /* default-construct if new */
    read_string (kf, grp, "gt-file",       mc.gt_file);
    read_float  (kf, grp, "min-ap-margin", mc.min_ap_margin);
    read_float  (kf, grp, "kpi-ap50",      mc.kpi_ap50);
    read_float  (kf, grp, "kpi-precision", mc.kpi_precision);
    read_float  (kf, grp, "kpi-recall",    mc.kpi_recall);
  }

  g_strfreev (groups);
  g_key_file_free (kf);
  return true;
}
