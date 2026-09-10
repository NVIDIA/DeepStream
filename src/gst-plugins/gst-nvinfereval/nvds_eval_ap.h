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

#include "nvds_eval_gt_loader.h"

#include <string>
#include <vector>
#include <utility>
#include <cstdint>

/**
 * compute_iou:
 * Standard intersection-over-union for axis-aligned boxes.
 */
float compute_iou (float ax1, float ay1, float ax2, float ay2,
                   float bx1, float by1, float bx2, float by2);

/**
 * match_preds_to_gt:
 * Full greedy IoU matching with explicit bbox coordinate arrays.
 * Called by the accumulator for every frame.
 *
 * Predictions are sorted by confidence descending; each is matched to the
 * best unmatched GT box of the same class with highest IoU.
 * out_best_iou[i] receives the best IoU for prediction i (0.0 if unmatched).
 * fn_count_out receives the count of unmatched GT boxes.
 */
void match_preds_to_gt (const std::vector<std::string> &preds_label,
                        const std::vector<float>       &preds_conf,
                        const std::vector<float>       &preds_x1,
                        const std::vector<float>       &preds_y1,
                        const std::vector<float>       &preds_x2,
                        const std::vector<float>       &preds_y2,
                        const std::vector<GTBox>       &gt_boxes,
                        std::vector<float>             &out_best_iou,
                        uint32_t                       &fn_count_out);

/**
 * compute_ap:
 * 101-point COCO-style interpolated AP at a single IoU threshold.
 *
 * @preds_scored  vector of (confidence, best_iou) accumulated over the epoch
 * @total_gt      total GT box count for this class over the epoch
 * @iou_thresh    IoU threshold for TP/FP classification
 */
float compute_ap (const std::vector<std::pair<float,float>> &preds_scored,
                  uint32_t total_gt, float iou_thresh);

/** AP@0.5 shorthand */
float compute_ap50   (const std::vector<std::pair<float,float>> &preds_scored,
                      uint32_t total_gt);

/** AP@0.5:0.95 (mean over 10 thresholds, COCO standard) */
float compute_ap50_95 (const std::vector<std::pair<float,float>> &preds_scored,
                       uint32_t total_gt);
