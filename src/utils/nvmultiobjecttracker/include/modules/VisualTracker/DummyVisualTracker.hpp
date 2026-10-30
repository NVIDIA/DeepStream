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

#ifndef VISUALTRACKERLIB_DUMMYVISUALTRACKER_HPP_
#define VISUALTRACKERLIB_DUMMYVISUALTRACKER_HPP_

#include "modules/VisualTracker/IVisualTracker.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Configuration parameters for the dummy visual tracker. */
class DummyVisualTrackerConfigParams : public NvTrackerParams
{
public:
	/** @brief Construct the default configuration. */
	DummyVisualTrackerConfigParams();

	/**
	 * @brief Construct configuration parameters from a named module in a file.
	 * @param[in] configFilePath The path to the configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	DummyVisualTrackerConfigParams(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Construct configuration parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure containing the configuration.
	 */
	DummyVisualTrackerConfigParams(const YAML::Node &fn);

	/**
	 * @brief Read visual tracker parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of configuration parameters.
	 */
	void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Debug message verbose mode level.

	// [Configuration parameters]
	VISUAL_TRACKER_TYPES m_visualTrackerType; //!< The type of visual tracker.
};

/** @brief Provides a no-op visual tracker implementation. */
class DummyVisualTracker : public IVisualTracker
{
public:

	/**
	 * @brief Construct from a configuration file path.
	 * @param[in] configFilePath The path to the configuration file.
	 */
	DummyVisualTracker(const std::string &configFilePath);

	/** @brief Destroy the dummy visual tracker. */
	virtual ~DummyVisualTracker();

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
	 * @param[in] maxTargetsToTrack The maximum number of targets to track.
	 * @param[in] maxDetectorBboxesPerFrame The maximum number of detector bounding boxes per frame.
	 * @param[in] targetManager The target manager through which all the targets can be accessed.
	 */
	void initialize(
		const SourceFrameInfo &frameInfo, const uint maxTargetsToTrack,
		const uint maxDetectorBboxesPerFrame, const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) override;

	/**
	 * @brief Populate the visual tracker's capability query.
	 * @param[out] paramsQuery The output query parameters.
	 */
	void queryParams(NvMOTQuery &paramsQuery);

	/**
	 * @brief Leave target locations unchanged for an input batch.
	 * @param[in] inputMeta A batch of input frames.
	 * @param[in] detectedObjects The list of newly-detected objects.
	 */
	void localize(
		const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta,
		const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
	) override;

	/**
	 * @brief Leave target models unchanged for an input batch.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void update(const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta) override;

	/** @brief Report whether tracker initialization completed. */
	bool isInitialized() override { return m_isInit; }

	/** @brief Return the configured visual tracker type. */
	const VISUAL_TRACKER_TYPES getVisualTrackerType() { return m_params.m_visualTrackerType; }

	/**
	 * @brief Update any Dynamic Params as provided during runtime by the user.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void updateParams(const YAML::Node &fn) { m_params.read(fn); }

protected:

	bool m_isInit; //!< Indicates whether tracker initialization completed.
	DummyVisualTrackerConfigParams m_params; //!< Config params for visual tracker module.
	std::shared_ptr<NvTrackedObjectManager> m_targetManager; //!< Target manager.
};

#endif // VISUALTRACKERLIB_DUMMYVISUALTRACKER_HPP_
