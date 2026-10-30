/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef IREIDSERVICE_HPP_
#define IREIDSERVICE_HPP_

#include <deque>
#include <map>
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Identifies the available ReID-service implementations. */
enum REID_SERVICE_TYPES : int
{
	DUMMY_REID_SERVICE = 0, //!< No-op implementation that performs no remote association.
	REID_SERVICE = 1 //!< Implementation that associates targets through a remote ReID service.
};

/** @brief Defines the interface for remote ReID association services. */
class IReIDService
{
public:
	/** @brief Release resources owned by the IReIDService instance. */
	virtual ~IReIDService() {}

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image, such as size and color.
	 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
	 * @param[in] targetManager Target manager of the tracked objects.
	 */
	virtual void initialize(
		const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
		const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) = 0;

	/**
	 * @brief Populate the capabilities reported to the tracker.
	 * @param[out] paramsQuery The output query parameters.
	 */
	virtual void queryParams(NvMOTQuery &paramsQuery) = 0;

	/**
	 * @brief Send query to reid service for all unmatched active objects.
	 * @param[in] inputMeta Input frames keyed by stream ID.
	 */
	virtual void query(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta) = 0;

	/** @brief Update target ID using remoteTargetID from reid service. */
	virtual void updateTargetIDs() = 0;

	/** @brief Get the configured ReID-service implementation type. */
	virtual const REID_SERVICE_TYPES getReIDServiceType() = 0;
};

#endif // IREIDSERVICE_HPP_
