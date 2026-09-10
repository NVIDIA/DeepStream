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
 * nvds_eval_ap.cpp
 * ================
 * IoU computation, greedy TP/FP matching, and COCO-style 101-point AP.
 *
 * Ported from evaluation/ap_calculator.py (100% Python/numpy) into plain
 * C++17 using std::sort and std::vector — semantically identical algorithm.
 */

#include "nvds_eval_ap.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

/* ------------------------------------------------------------------ */
/* IoU                                                                */
/* ------------------------------------------------------------------ */

float
compute_iou (float ax1, float ay1, float ax2, float ay2,
             float bx1, float by1, float bx2, float by2)
{
  float ix1 = std::max (ax1, bx1);
  float iy1 = std::max (ay1, by1);
  float ix2 = std::min (ax2, bx2);
  float iy2 = std::min (ay2, by2);

  float iw = std::max (0.0f, ix2 - ix1);
  float ih = std::max (0.0f, iy2 - iy1);
  float inter = iw * ih;
  if (inter <= 0.0f) return 0.0f;

  float area_a = std::max (0.0f, ax2 - ax1) * std::max (0.0f, ay2 - ay1);
  float area_b = std::max (0.0f, bx2 - bx1) * std::max (0.0f, by2 - by1);
  float uni    = area_a + area_b - inter;
  return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

/* ------------------------------------------------------------------ */
/* Greedy matching (bbox-aware)                                       */
/* ------------------------------------------------------------------ */
void
match_preds_to_gt (const std::vector<std::string> &preds_label,
                   const std::vector<float>       &preds_conf,
                   const std::vector<float>       &preds_x1,
                   const std::vector<float>       &preds_y1,
                   const std::vector<float>       &preds_x2,
                   const std::vector<float>       &preds_y2,
                   const std::vector<GTBox>       &gt_boxes,
                   std::vector<float>             &out_best_iou,
                   uint32_t                       &fn_count_out)
{
  size_t np = preds_label.size ();
  out_best_iou.assign (np, 0.0f);

  /* Sort by confidence descending */
  std::vector<size_t> order (np);
  std::iota (order.begin (), order.end (), 0);
  std::sort (order.begin (), order.end (),
      [&](size_t a, size_t b){ return preds_conf[a] > preds_conf[b]; });

  std::vector<bool> gt_matched (gt_boxes.size (), false);

  for (size_t pi : order) {
    float best_iou = 0.0f;
    int   best_gi  = -1;

    for (size_t gi = 0; gi < gt_boxes.size (); ++gi) {
      if (gt_matched[gi]) continue;
      if (gt_boxes[gi].class_label != preds_label[pi]) continue;

      float iou = compute_iou (preds_x1[pi], preds_y1[pi],
                               preds_x2[pi], preds_y2[pi],
                               gt_boxes[gi].x1, gt_boxes[gi].y1,
                               gt_boxes[gi].x2, gt_boxes[gi].y2);
      if (iou > best_iou) {
        best_iou = iou;
        best_gi  = (int)gi;
      }
    }

    out_best_iou[pi] = best_iou;
    if (best_gi >= 0 && best_iou > 0.0f)
      gt_matched[best_gi] = true;
  }

  fn_count_out = 0;
  for (bool m : gt_matched)
    if (!m) ++fn_count_out;
}

/* ------------------------------------------------------------------ */
/* AP computation                                                     */
/* ------------------------------------------------------------------ */

/*
 * 101-point interpolation (COCO):
 *   1. Sort all predictions by confidence descending.
 *   2. Classify each as TP (best_iou >= thresh) or FP.
 *   3. Compute cumulative precision/recall arrays.
 *   4. Interpolate precision at 101 recall points [0, 0.01, ..., 1.0].
 *   5. AP = mean of those 101 values.
 */
/*
 * ap_from_sorted: 101-point interpolated AP from predictions already sorted by
 * confidence descending. Numerically identical to the previous O(101*n) form
 * but computed in O(n): cumulative recall is non-decreasing in this order, so
 * { i : rec[i] >= recall_thr } is always a suffix [lo, n). The maximum
 * precision over that suffix is read from a precomputed suffix-max array, and
 * lo advances monotonically as the recall threshold increases.
 *
 * Caller must guarantee total_gt > 0 and a non-empty input.
 */
static float
ap_from_sorted (const std::vector<std::pair<float,float>> &sorted,
                uint32_t total_gt, float iou_thresh)
{
  size_t n = sorted.size ();
  std::vector<float> prec (n), rec (n);
  uint32_t tp_cum = 0, fp_cum = 0;

  for (size_t i = 0; i < n; ++i) {
    bool is_tp = (sorted[i].second >= iou_thresh);
    if (is_tp) ++tp_cum; else ++fp_cum;

    prec[i] = (float)tp_cum / (float)(tp_cum + fp_cum);
    rec[i]  = (float)tp_cum / (float)total_gt;
  }

  /* suffix_max[i] = max(prec[i..n-1]); suffix_max[n] = 0 (empty suffix). */
  std::vector<float> suffix_max (n + 1, 0.0f);
  for (size_t i = n; i-- > 0; )
    suffix_max[i] = std::max (prec[i], suffix_max[i + 1]);

  float  ap = 0.0f;
  size_t lo = 0;
  for (int r = 0; r <= 100; ++r) {
    float recall_thr = (float)r / 100.0f;
    while (lo < n && rec[lo] < recall_thr) ++lo;
    ap += (lo < n) ? suffix_max[lo] : 0.0f;
  }
  return ap / 101.0f;
}

float
compute_ap (const std::vector<std::pair<float,float>> &preds_scored,
            uint32_t total_gt, float iou_thresh)
{
  if (total_gt == 0 || preds_scored.empty ())
    return 0.0f;

  /* Sort by confidence descending */
  std::vector<std::pair<float,float>> sorted = preds_scored;
  std::sort (sorted.begin (), sorted.end (),
      [](const std::pair<float,float> &a, const std::pair<float,float> &b){
        return a.first > b.first;
      });

  return ap_from_sorted (sorted, total_gt, iou_thresh);
}

float
compute_ap50 (const std::vector<std::pair<float,float>> &preds_scored,
              uint32_t total_gt)
{
  return compute_ap (preds_scored, total_gt, 0.5f);
}

float
compute_ap50_95 (const std::vector<std::pair<float,float>> &preds_scored,
                 uint32_t total_gt)
{
  if (total_gt == 0 || preds_scored.empty ())
    return 0.0f;

  /* The confidence sort order is independent of the IoU threshold, so sort
   * once and reuse it for all 10 thresholds instead of re-sorting per call. */
  std::vector<std::pair<float,float>> sorted = preds_scored;
  std::sort (sorted.begin (), sorted.end (),
      [](const std::pair<float,float> &a, const std::pair<float,float> &b){
        return a.first > b.first;
      });

  float sum = 0.0f;
  int   cnt = 0;
  for (int t = 50; t <= 95; t += 5) {
    sum += ap_from_sorted (sorted, total_gt, (float)t / 100.0f);
    ++cnt;
  }
  return (cnt > 0) ? (sum / (float)cnt) : 0.0f;
}
