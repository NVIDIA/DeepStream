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
#include "nvds_eval_ap.h"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <cstdint>

/**
 * ClassAccum: per-(role, model_name, class_label) accumulator over one epoch.
 */
struct ClassAccum
{
  std::vector<std::pair<float,float>> preds;     /* (conf, best_iou) per prediction */
  uint32_t total_gt  = 0;   /* GT boxes of this class seen this epoch               */
};

/**
 * ModelAccum: accumulator for one (role, model_name) pair over an epoch.
 *
 * Keyed by (role, model_name) so that multiple primary models across
 * different streams are tracked independently, not merged.
 */
struct ModelAccum
{
  std::string model_version;
  std::map<std::string, ClassAccum> per_class;   /* class_label -> ClassAccum */
  uint32_t frame_count    = 0;   /* frames processed for this (role, model)      */
  uint32_t frames_with_gt = 0;   /* frames where GT was non-empty (GT available) */
};

/**
 * ModelKey: compound key for the accumulator map.
 * Ordered so std::map works without a custom comparator.
 */
struct ModelKey
{
  std::string role;         /* "Primary" | "Shadow"      */
  std::string model_name;   /* logical model name        */

  bool operator< (const ModelKey &o) const {
    if (role != o.role) return role < o.role;
    return model_name < o.model_name;
  }
  bool operator== (const ModelKey &o) const {
    return role == o.role && model_name == o.model_name;
  }
};

/**
 * EvalAccumulator: per-epoch accumulator across all (role, model) pairs.
 *
 * add_frame() is called from the GStreamer streaming thread. The engine
 * serialises calls to this (one chain function per pad), so no locking needed.
 */
class EvalAccumulator
{
public:
  /**
   * add_frame: record one frame's predictions vs GT.
   *
   * role / model_name: from InferenceProvenanceMeta (Mode B) or hardcoded (Mode A).
   * preds_*: parallel prediction arrays from NvDsObjectMeta.
   * gt_boxes: GT for (camera_id, frame_num) — empty when GT unavailable.
   */
  void add_frame (const std::string              &role,
                  const std::string              &model_name,
                  const std::string              &model_version,
                  const std::vector<std::string> &preds_label,
                  const std::vector<float>       &preds_conf,
                  const std::vector<float>       &preds_x1,
                  const std::vector<float>       &preds_y1,
                  const std::vector<float>       &preds_x2,
                  const std::vector<float>       &preds_y2,
                  const std::vector<GTBox>       &gt_boxes);

  void reset ();

  const std::map<ModelKey, ModelAccum> &models () const { return models_; }
  uint32_t total_frames () const { return total_frames_; }

private:
  std::map<ModelKey, ModelAccum> models_;    /* (role, model_name) -> accum  */
  uint32_t                       total_frames_ = 0;
};
