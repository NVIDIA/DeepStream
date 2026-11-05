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

#ifndef DATAASSOCIATORLIB_IDATAASSOCIATOR_HPP_
#define DATAASSOCIATORLIB_IDATAASSOCIATOR_HPP_

#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

enum DATA_ASSOCIATOR_TYPES : int
{
	DEFAULT_ASSOCIATOR = 0 //!< Selects the default data associator.
};

/** @brief Define the interface for associating detections with tracked targets. */
class IDataAssociator
{
public:
	/** @brief Destroy the data-associator interface. */
	virtual ~IDataAssociator() {}

	/**
	 * @brief Initialize the data associator.
	 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
	 * @param[in] maxTargetsPerStream The max number of targets to track per stream.
	 * @param[in] useVisualTracker Whether visual-tracker similarity is used.
	 * @param[in] useReid Whether ReID similarity is used.
	 */
	virtual void initialize(
		const SourceFrameInfo &frameInfo, const uint32_t maxTargetsPerStream,
		const bool useVisualTracker = false, const bool useReid = false
	) = 0;

	/**
	 * @brief Associate detections with existing targets for one stream.
	 * @param[in] in Newly detected object bounding boxes and existing targets.
	 * @return (1) matched pairs, (2) unmatched detector bboxes, and (3) unmatched targets.
	 */
	virtual assocDataOut_t associatePerStream(const assocDataIn_t in) = 0;

	/** @brief Return whether the data associator has been initialized. */
	virtual bool isInitialized() = 0;

	/** @brief Return the configured data-associator type. */
	virtual const DATA_ASSOCIATOR_TYPES getDataAssociatorType() = 0;

	/**
	 * @brief Update dynamic parameters during runtime.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	virtual void updateParams(const YAML::Node &fn) = 0;

};

#endif // DATAASSOCIATORLIB_IDATAASSOCIATOR_HPP_
