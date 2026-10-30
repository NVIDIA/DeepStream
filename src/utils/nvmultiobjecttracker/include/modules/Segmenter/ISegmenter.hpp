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

#ifndef SEGMENTERLIB_ISEGMENTER_HPP_
#define SEGMENTERLIB_ISEGMENTER_HPP_

#include <deque>
#include <map>
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

enum SEGMENTER_TYPES : int
{
	DUMMY_SEGMENTER = 0, //!< Segmenter that performs no segmentation.
	SAM2 = 1, //!< SAM2 segmenter.
	NanoSAM = 2, //!< NanoSAM segmenter.
	EfficientVitSAM = 3 //!< EfficientViT-SAM segmenter.
};

/** @brief Defines the interface for target segmentation modules. */
class ISegmenter
{
public:
	/** @brief Release resources owned by the ISegmenter instance. */
	virtual ~ISegmenter() {}

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
	 * @brief Localize and segment target in a new frame.
	 * @param[in] inputMeta A batch of input frames.
	 */
	virtual void localize(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta) = 0;

	/**
	 * @brief Update Segmenter after data association.
	 * @param[in] inputMeta A batch of input frames.
	 */
	virtual void update(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta) = 0;

	/**
	 * @brief Output segmentation mask of a target.
	 * @param[in] target The target.
	 * @param[out] mask The output mask.
	 */
	virtual void outputMaskObj(const std::shared_ptr<NvTrackedObject> target, NvMOTMask& mask) = 0;

	/**
	 * @brief Remove a stream's Segmenter features.
	 * @param[in] videoStreamId Video stream id.
	 */
	virtual void removeStream(const NvMOTStreamId videoStreamId) = 0;

	/** @brief Report whether the module is initialized. */
	virtual bool isInitialized() = 0;

	/** @brief Get the configured segmenter type. */
	virtual const SEGMENTER_TYPES getSegmenterType() = 0;
};

#endif // SEGMENTERLIB_ISEGMENTER_HPP_
