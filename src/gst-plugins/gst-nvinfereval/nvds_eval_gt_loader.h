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

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <cstdint>

/**
 * GTBox: one ground-truth bounding box in absolute pixel coordinates.
 */
struct GTBox
{
  std::string class_label;
  float       x1, y1, x2, y2;
};

/**
 * GTKey: (camera_id, frame_num).
 *
 * camera_id is a string — it matches:
 *   Mode B: InferenceProvenanceMeta.camera_name
 *   Mode A: std::to_string(source_id)
 *   COCO:   "0" (single-camera convention)
 *   KITTI:  sub-directory name or "0" for a flat directory
 */
using GTKey   = std::pair<std::string, uint32_t>;
using GTIndex = std::map<GTKey, std::vector<GTBox>>;

enum class GtFormat {
  KITTI,
  NVSCHEMA,
  COCO,
  TAO,
  UNKNOWN
};

/**
 * Detect the format of the GT file or directory at `path`.
 */
GtFormat gt_detect_format (const std::string &path);

/**
 * Load ground truth from `path` into `index`.
 * Format is auto-detected if fmt == UNKNOWN.
 * Returns true on success.
 */
bool gt_load (const std::string &path, GTIndex &index,
              GtFormat fmt = GtFormat::UNKNOWN);
