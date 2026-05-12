/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef LIDAR_POST_PROCESS_H_
#define LIDAR_POST_PROCESS_H_
#include <vector>
#include "ds3d/common/ds3d_analysis_datatype.h"

using namespace ds3d;

int ParseCustomBatchedNMS(std::vector<Lidar3DBbox> bndboxes, const float nms_thresh,
              std::vector<Lidar3DBbox> &nms_pred, const int pre_nms_top_n);

#endif