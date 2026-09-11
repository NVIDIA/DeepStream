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

#include "nvds_eval_accumulator.h"
#include "nvds_eval_ap.h"   /* match_preds_to_gt */

void
EvalAccumulator::add_frame (const std::string              &role,
                            const std::string              &model_name,
                            const std::string              &model_version,
                            const std::vector<std::string> &preds_label,
                            const std::vector<float>       &preds_conf,
                            const std::vector<float>       &preds_x1,
                            const std::vector<float>       &preds_y1,
                            const std::vector<float>       &preds_x2,
                            const std::vector<float>       &preds_y2,
                            const std::vector<GTBox>       &gt_boxes)
{
  ModelKey     key {role, model_name};
  ModelAccum  &ma = models_[key];
  ma.model_version = model_version;
  ++ma.frame_count;
  ++total_frames_;

  if (!gt_boxes.empty ())
    ++ma.frames_with_gt;

  /* Tally GT box counts per class for this frame */
  for (const auto &gb : gt_boxes)
    ma.per_class[gb.class_label].total_gt++;

  size_t np = preds_label.size ();
  if (np == 0)
    return;

  /* Run greedy IoU matching */
  std::vector<float> best_iou;
  uint32_t           fn_count_unused = 0;

  match_preds_to_gt (preds_label, preds_conf,
                     preds_x1, preds_y1, preds_x2, preds_y2,
                     gt_boxes, best_iou, fn_count_unused);
  (void) fn_count_unused;

  /* Accumulate (confidence, best_iou) per class */
  for (size_t i = 0; i < np; ++i) {
    float iou = (i < best_iou.size ()) ? best_iou[i] : 0.0f;
    ma.per_class[preds_label[i]].preds.emplace_back (preds_conf[i], iou);
  }
}

void
EvalAccumulator::reset ()
{
  models_.clear ();
  total_frames_ = 0;
}
