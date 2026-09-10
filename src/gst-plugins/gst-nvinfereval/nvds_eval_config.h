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
 * nvds_eval_config.h
 * ==================
 * Configuration structures and GKeyFile-based INI loader for nvinfereval.
 *
 * Config file format (GKeyFile / INI):
 *
 *   [eval]
 *   eval-interval    = 1000
 *   iou-threshold    = 0.50
 *   promotion-mode   = 0
 *   report-file      = /tmp/eval.json
 *   default-gt-file  = /data/gt/common.json
 *
 *   [model-yolo_v8]
 *   gt-file       = /data/gt/yolo.json
 *   min-ap-margin = 0.02
 *   kpi-ap50      = 0.75
 *   kpi-precision = 0.0
 *   kpi-recall    = 0.0
 *
 *   [model-resnet50]
 *   gt-file       = /data/gt/resnet.json
 *   min-ap-margin = 0.03
 *   kpi-ap50      = 0.80
 *
 * Per-model sections ([model-<name>]) define per-model overrides.
 * Fields absent from a model section fall back to global values.
 * Fields absent from the [eval] section fall back to GObject property values
 * (set before prepare() is called).
 */

#pragma once

#include <string>
#include <map>
#include <set>
#include <glib.h>

/* ------------------------------------------------------------------ */
/* GlobalConfig                                                        */
/* ------------------------------------------------------------------ */

struct GlobalConfig
{
  guint    eval_interval   = 1000;
  gfloat   iou_threshold   = 0.5f;
  gboolean show_gt_overlay     = FALSE;  /* draw GT boxes as green rects via nvosd */
  gboolean show_ap_table       = FALSE;  /* print per-class AP table to console */
  gboolean cumulative          = TRUE;   /* don't reset accumulator between epochs */
  gboolean exclude_undetected  = TRUE;   /* auto-exclude classes with 0 TP and 0 FP */
  std::set<std::string> exclude_classes; /* classes skipped in mean AP/P/R */
  std::string report_file;
  /* Fallback GT file used for any model that has no per-model gt-file. */
  std::string default_gt_file;
  /* Base URL of the nvmodelmux REST server (e.g. http://localhost:9000).
   * When a shadow wins an epoch, the win-action below is applied by POSTing to
   * <rest_url>/api/v1/stream/route. Empty (default) => no action taken. */
  std::string rest_url;
  /* Action when the shadow wins an epoch:
   *   0 = promote (default): winner -> primary, shadow cleared (A/B ends)
   *   1 = swap             : winner -> primary, loser -> shadow (A/B continues)
   *   2 = report-only      : no REST call, pure monitoring mode */
  guint promotion_mode = 0;
};

/* ------------------------------------------------------------------ */
/* ModelConfig                                                         */
/* ------------------------------------------------------------------ */

struct ModelConfig
{
  /* Empty → use GlobalConfig::default_gt_file */
  std::string gt_file;

  gfloat  min_ap_margin  = 0.02f;
  gfloat  kpi_ap50       = 0.0f;   /* 0 = disabled */
  gfloat  kpi_precision  = 0.0f;   /* 0 = disabled */
  gfloat  kpi_recall     = 0.0f;   /* 0 = disabled */
};

/* ------------------------------------------------------------------ */
/* EvalConfig                                                          */
/* ------------------------------------------------------------------ */

struct EvalConfig
{
  GlobalConfig                        global;
  std::map<std::string, ModelConfig>  models; /* model_name → per-model config */

  /**
   * effective_model_config:
   * Return the ModelConfig for model_name, filling absent fields from global
   * defaults. If the model has no section in the file the global fallbacks
   * are used for every field.
   */
  ModelConfig effective_model_config (const std::string &model_name) const;
};

/* ------------------------------------------------------------------ */
/* Loader                                                              */
/* ------------------------------------------------------------------ */

/**
 * eval_config_load:
 * Parse the GKeyFile INI at @path and overlay found values into @cfg.
 * Only keys that are present in the file are written; absent keys leave
 * the corresponding @cfg fields unchanged (caller pre-populates defaults
 * from GObject properties before calling this).
 *
 * Returns true on success (file opened and parsed). Returns false if the
 * file could not be opened or has a parse error; @cfg is left partially
 * updated in that case and the caller should treat it as invalid.
 */
bool eval_config_load (const std::string &path, EvalConfig &cfg);
