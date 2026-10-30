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
#ifndef NVMULTIOBJECTTRACKERLIB_NVTLOCALIZER_H
#define NVMULTIOBJECTTRACKERLIB_NVTLOCALIZER_H

#include <map>
#include <set>
#include <memory>

#include "modules/utility/dataTypes.hpp"

#include "nvdstracker.h"
#include "nvds_tracker_meta.h"

/** @brief Defines batched multi-stream tracker operations. */
class IBatchLocalizer
{
public:
	/** @brief Initialize the batch-localizer interface. */
	IBatchLocalizer() {};

	/** @brief Release the batch-localizer interface. */
	virtual ~IBatchLocalizer() {};

	/** @brief Report whether the batch localizer is initialized. */
	virtual bool isInitialized() = 0;

	/**
	 * @brief Initialize the batch localizer for the tracker input surface.
	 * @param[in] srcImgSize Source image size.
	 * @param[in] srcImgPitch Source image row pitch in bytes.
	 */
	virtual void initialize(const NvMotDataTypes::Size srcImgSize, const size_t srcImgPitch) = 0;

	/**
	 * @brief Populate tracker capability and memory requirement fields.
	 * @param[out] paramsQuery Query structure populated with tracker requirements.
	 */
	virtual void queryParams(NvMOTQuery &paramsQuery) = 0;

	/**
	 * @brief Perform tracking.
	 * @param[in] nvFramesInBatch Source frames and detector bboxes keyed by video stream ID.
	 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch filled with tracked objects for this frame batch.
	 */
	virtual void update(
		const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch, NvMOTTrackedObjBatch *&pTrackedObjectsBatch
	) = 0;

	/**
	 * @brief Output past-frame tracking data and clear the corresponding buffers.
	 * @param[in] videoStreamIdList Stream IDs whose past-frame tracking data is reported.
	 * @param[in,out] pPastFrameObjBatch Caller-allocated batch that receives the past-frame objects.
	 */
	virtual void outputPastFrameObjs(
		const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pPastFrameObjBatch
	) = 0;

	/**
	 * @brief Output complete frame history for terminated tracks and clear the buffer.
	 * @param[in] videoStreamIdList Stream IDs whose terminated tracks are reported.
	 * @param[in,out] pPastFrameObjBatch Caller-allocated batch that receives the terminated-track history.
	 */
	virtual void outputTerminatedObjs(
		const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pPastFrameObjBatch
	) = 0;

	/**
	 * @brief Output the most recent frame for current shadow tracks and clear the buffer.
	 * @param[in] videoStreamIdList Stream IDs whose shadow tracks are reported.
	 * @param[in,out] pShadowTracksBatch Caller-allocated batch that receives the latest shadow-track frame.
	 */
	virtual void outputShadowObj(
		const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pShadowTracksBatch
	) = 0;

	/**
	 * @brief Delete trackers and per-stream resources when a stream is removed.
	 * @param[in] videoStreamID Stream ID whose trackers are removed.
	 */
	virtual void deleteRemovedStreamTrackers(const NvMOTStreamId videoStreamID) = 0;

	/**
	 * @brief Apply supported runtime tracker configuration updates.
	 * @param[in] configStr Stringified YAML configuration parameters to update.
	 */
	virtual void updateParams(const std::string &configStr) = 0;

protected:
};

#endif // NVMULTIOBJECTTRACKERLIB_NVTLOCALIZER_H
