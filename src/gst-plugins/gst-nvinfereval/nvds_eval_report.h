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

#pragma once

#include "nvds_eval_accumulator.h"

#include <gst/gst.h>

#include <string>
#include <vector>
#include <set>
#include <cstdint>

/* ------------------------------------------------------------------ */
/* ClassMetrics                                                        */
/* ------------------------------------------------------------------ */

struct ClassMetrics
{
  std::string class_label;
  float       ap50       = 0.0f;
  float       ap50_95    = 0.0f;
  float       precision  = 0.0f;
  float       recall     = 0.0f;
  float       f1         = 0.0f;
  uint32_t    tp         = 0;
  uint32_t    fp         = 0;
  uint32_t    total_gt   = 0;
  bool        excluded        = false;  /* excluded from mean (manual or auto) */
  bool        auto_excluded   = false;  /* true when excluded because 0 TP+FP */
};

/* ------------------------------------------------------------------ */
/* ModelMetrics                                                        */
/* ------------------------------------------------------------------ */

struct ModelMetrics
{
  std::string role;
  std::string model_name;
  std::string model_version;
  float       ap50           = 0.0f;   /* mean AP@0.5 over all classes */
  float       ap50_95        = 0.0f;   /* mean AP@0.5:0.95             */
  float       mean_precision = 0.0f;
  float       mean_recall    = 0.0f;
  uint32_t    frame_count    = 0;
  uint32_t    frames_with_gt = 0;      /* 0 = GT not available         */
  bool        gt_available   = false;
  std::vector<ClassMetrics> per_class;
};

/* ------------------------------------------------------------------ */
/* compute_metrics
 *
 * Compute ModelMetrics for every (role, model_name) entry in @accum.
 * @iou_thresh is used for TP/FP classification (precision/recall/F1);
 * AP@0.5 and AP@0.5:0.95 always use their fixed standard thresholds.
 * ------------------------------------------------------------------ */
std::vector<ModelMetrics> compute_metrics (EvalAccumulator             &accum,
                                            float                        iou_thresh,
                                            const std::set<std::string> &exclude_classes    = {},
                                            bool                         exclude_undetected  = false);

/* ------------------------------------------------------------------ */
/* build_json_report
 *
 * Serialise one epoch's results to a JSON string.
 *
 * @model_pair_name  logical model_name that this epoch belongs to
 *                   (identifies which ModelPairContext fired)
 * @epoch_frames     Primary frame count that triggered this epoch
 * ------------------------------------------------------------------ */
std::string build_json_report (const std::string               &model_pair_name,
                                const std::vector<ModelMetrics> &metrics,
                                uint32_t                         epoch_frames,
                                bool                             promoted,
                                const std::string               &promoted_model);

/* ------------------------------------------------------------------ */
/* post_eval_result
 *
 * Post a GST_MESSAGE_APPLICATION("model-eval-result") on the element's
 * bus and optionally append the JSON to @report_file.
 *
 * @model_pair_name  forwarded to build_json_report and included in the
 *                   console summary line for easy grepping
 * ------------------------------------------------------------------ */
void post_eval_result (GstElement                      *self,
                       const std::string               &model_pair_name,
                       const std::vector<ModelMetrics> &metrics,
                       uint32_t                         epoch_frames,
                       bool                             promoted,
                       const std::string               &promoted_model,
                       const std::string               &report_file,
                       bool                             cumulative   = false,
                       bool                             show_ap_table = false);
