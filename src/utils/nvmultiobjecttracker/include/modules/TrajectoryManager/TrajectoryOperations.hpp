/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMULTIOBJECTTRACKERLIB_TRAJECTORYOPERATIONS_HPP_
#define NVMULTIOBJECTTRACKERLIB_TRAJECTORYOPERATIONS_HPP_

#include <map>
#include <deque>
#include <utility>

#include "modules/utility/cv_functions.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"

namespace TrajOps {

/**
 * @brief Combine same-frame target records from overlapping tracklets.
 * @param[in] prevTarget Target record from the earlier tracklet.
 * @param[in] nextTarget Target record from the later tracklet.
 */
TargetInfo fuseTargets(const TargetInfo prevTarget, const TargetInfo nextTarget);

/**
 * @brief Interpolate target records between two observations.
 * @param[in] prevTarget Earlier target observation.
 * @param[in] newxtTarget Later target observation.
 */
std::deque<TargetInfo> generateInterpolatedTargets(const TargetInfo prevTarget, const TargetInfo newxtTarget);

/**
 * @brief Generate a per-frame tracklet over an optional frame range.
 * @param[in] traj Sparse trajectory to interpolate.
 * @param[in] startFrameNum First frame to include, or zero for the complete trajectory.
 * @param[in] endFrameNum Last frame to include, or zero for the complete trajectory.
 */
std::list<TargetInfo> generateFullTracklet(
	const std::list<TargetInfo> &traj, const int startFrameNum = 0, const int endFrameNum = 0
);

}

#endif // NVMULTIOBJECTTRACKERLIB_TRAJECTORYOPERATIONS_HPP_
