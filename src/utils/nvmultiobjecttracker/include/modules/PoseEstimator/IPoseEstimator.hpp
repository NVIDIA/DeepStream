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

#ifndef POSEESTIMATORLIB_IPOSEESTIMATOR_HPP_
#define POSEESTIMATORLIB_IPOSEESTIMATOR_HPP_

#include <deque>
#include <map>
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Enumerate the supported pose-estimator implementation types. */
enum POSEESTIMATOR_TYPES : int
{
	DUMMY_POSE = 0, //!< Disables pose estimation.
	POSE_INFERENCE = 1, //!< Selects TensorRT pose inference.
	POSE_TRACKER = 2 //!< Selects pose tracking.
};

/** @brief Defines the interface for per-target pose estimation. */
class IPoseEstimator
{
public:
	/** @brief Release resources owned by the IPoseEstimator instance. */
	virtual ~IPoseEstimator() {}

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image, such as size and color.
	 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
	 * @param[in] targetManager Target manager of the tracked objects.
	 */
	virtual void initializeRuntime(
		const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
		const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) = 0;

	/** @brief Create TensorRT inference engine; if engine does not exist, serialize from model file. */
	virtual void initializeModel() = 0;

	/**
	 * @brief Populate the capabilities reported to the tracker.
	 * @param[out] paramsQuery The output query parameters.
	 */
	virtual void queryParams(NvMOTQuery &paramsQuery) = 0;

	/**
	 * @brief Run pose inference on cropped targets from the input frames.
	 * @param[in] inputMeta A batch of input frames.
	 * @param[in,out] detectedObjects Newly detected objects in each stream.
	 * @param[in] isForReAssoc Whether this inference is for reassociation.
	 */
	virtual void doInference(
		const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
		std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
	) = 0;

	/** @brief Update the pose estimator after data association. */
	virtual void update() = 0;

	/**
	 * @brief Remove a stream from the pose estimator.
	 * @param[in] videoStreamId Video stream id.
	 */
	virtual void removeStream(const NvMOTStreamId videoStreamId) = 0;

	/** @brief Get the configured pose-estimator type. */
	virtual const POSEESTIMATOR_TYPES getPoseEstimatorType() = 0;
};

#endif // POSEESTIMATORLIB_IPOSEESTIMATOR_HPP_
