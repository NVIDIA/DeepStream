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
#ifndef LIDAR_POST_PROCESS_H_
#define LIDAR_POST_PROCESS_H_
#include <vector>
#include "ds3d/common/ds3d_analysis_datatype.h"

using namespace ds3d;

int ParseCustomBatchedNMS(std::vector<Lidar3DBbox> bndboxes, const float nms_thresh,
              std::vector<Lidar3DBbox> &nms_pred, const int pre_nms_top_n);

#endif