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

#ifndef VISUALTRACKERLIB_IVISUALTRACKER_HPP_
#define VISUALTRACKERLIB_IVISUALTRACKER_HPP_

#include <memory>
#include <map>
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Visual tracker implementations that can be selected by configuration. */
enum VISUAL_TRACKER_TYPES : int
{
    DUMMY_VOT = 0, //!< Selects the dummy visual tracker.
    NvDCF_VOT, //!< Selects the NvDCF visual tracker.
    NvDCF_VPI_VOT //!< Selects the VPI-based NvDCF visual tracker.
};

/** @brief Defines the interface implemented by visual trackers. */
class IVisualTracker
{
public:
	/** @brief Destroy the visual tracker interface. */
	virtual ~IVisualTracker() {}

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
	 * @param[in] maxTargetsToTrack The maximum number of targets to track.
	 * @param[in] maxTargetsPerStream The maximum number of targets to track per stream.
	 * @param[in] targetManager The target manager through which all the targets can be accessed.
	 */
	virtual void initialize(
		const SourceFrameInfo &frameInfo, const uint maxTargetsToTrack,
		const uint maxTargetsPerStream, const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) = 0;

	/**
	 * @brief Populate the visual tracker's capability query.
	 * @param[out] paramsQuery The output query parameters.
	 */
	virtual void queryParams(NvMOTQuery &paramsQuery) = 0;

    /**
     * @brief Localize tracked targets and evaluate nearby detections across an input batch.
     * @param[in] inputMeta A batch of input frames.
     * @param[in] detectedObjects The list of newly-detected objects.
     */
    virtual void localize(
        const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta,
        const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
    ) = 0;

    /**
     * @brief Update each target's visual model from an input batch.
     * @param[in] inputMeta A batch of input frames.
     */
    virtual void update(const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta) = 0;

    /** @brief Report whether tracker initialization completed. */
    virtual bool isInitialized() = 0;

    /** @brief Return the configured visual tracker type. */
    virtual const VISUAL_TRACKER_TYPES getVisualTrackerType() = 0;

    /**
     * @brief Update runtime-configurable visual tracker parameters.
     * @param[in] fn Parsed YAML structure of config to update.
     */
    virtual void updateParams(const YAML::Node &fn) = 0;
};

#endif // VISUALTRACKERLIB_IVISUALTRACKER_HPP_
