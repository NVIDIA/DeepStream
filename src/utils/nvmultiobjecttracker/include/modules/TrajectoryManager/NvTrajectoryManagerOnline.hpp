/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMULTIOBJECTTRACKERLIB_NVTRAJECTORYMANAGERONLINE_HPP_
#define NVMULTIOBJECTTRACKERLIB_NVTRAJECTORYMANAGERONLINE_HPP_

#include <list>
#include <memory>
#include <iostream>
#include <map>
#include <fstream>

#include "modules/TrajectoryManager/NvTrajectoryManager.hpp"

#ifdef VIZ_DEBUG
/** @brief Forward declaration. */
class TrackerVizOutput;
#endif

/** @brief Manage trajectories for online tracking and ReID extraction. */
class NvTrajectoryManagerOnline : public NvTrajectoryManager
{
public:
	/**
	 * @brief Construct an online trajectory manager from configuration.
	 * @param[in] configFilePath Path to the configuration file.
	 */
	NvTrajectoryManagerOnline(const std::string &configFilePath);

	/** @brief Destroy the online trajectory manager. */
	virtual ~NvTrajectoryManagerOnline();

	/**
	 * @brief Generate target detections that require ReID extraction for one stream.
	 * @param[in] streamId Source stream identifier.
	 * @param[in] targetList Tracked targets to inspect.
	 */
	std::deque<DetectedObjInfo> generateTrackedObjects4ReidPerStream(
		const NvMOTStreamId streamId, const std::map<TrackerIndex, std::shared_ptr<NvTrackedObject>> &targetList
	) override;
};

#endif // NVMULTIOBJECTTRACKERLIB_NVTRAJECTORYMANAGERONLINE_HPP_
