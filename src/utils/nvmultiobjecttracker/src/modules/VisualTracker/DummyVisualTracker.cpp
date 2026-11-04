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

#include "modules/VisualTracker/DummyVisualTracker.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct from a configuration file path.
 * @param[in] configFilePath The path to the configuration file.
 */
DummyVisualTracker::DummyVisualTracker(const std::string &configFilePath)
{
	m_isInit = false;

	/// Parse the visual tracker configuration.
	m_params = DummyVisualTrackerConfigParams(configFilePath, "VisualTracker");
}

/** @brief Destroy the dummy visual tracker. */
DummyVisualTracker::~DummyVisualTracker()
{
}

/**
 * @brief Initialize variables and allocate memories.
 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
 * @param[in] maxTargetsToTrack The maximum number of targets to track.
 * @param[in] maxDetectorBboxesPerFrame The maximum number of detector bounding boxes per frame.
 * @param[in] targetManager The target manager through which all the targets can be accessed.
 */
void DummyVisualTracker::initialize(
	const SourceFrameInfo &frameInfo, const uint maxTargetsToTrack,
	const uint maxDetectorBboxesPerFrame, const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
	m_isInit = true;

	m_targetManager = targetManager;

	if(m_targetManager == nullptr)
	{
		throw std::runtime_error("[DummyVisualTracker::initialize()] m_targetManager == nullptr");
	}
}

/**
 * @brief Populate the visual tracker's capability query.
 * @param[out] paramsQuery The output query parameters.
 */
void DummyVisualTracker::queryParams(NvMOTQuery &paramsQuery)
{
}

/**
 * @brief Leave target locations unchanged for an input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in] detectedObjects The list of newly-detected objects.
 */
void DummyVisualTracker::localize(
	const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta,
	const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
}

/**
 * @brief Leave target models unchanged for an input batch.
 * @param[in] inputMeta A batch of input frames.
 */
void DummyVisualTracker::update(const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta)
{
}

/** @brief Construct the default configuration. */
DummyVisualTrackerConfigParams::DummyVisualTrackerConfigParams()
{
	m_debugVerboseLevel = 0;

	m_visualTrackerType = VISUAL_TRACKER_TYPES::DUMMY_VOT;
}

/**
 * @brief Construct configuration parameters from a named module in a file.
 * @param[in] configFilePath The path to the configuration file.
 * @param[in] moduleName The configuration module name.
 */
DummyVisualTrackerConfigParams::DummyVisualTrackerConfigParams(
	const std::string &configFilePath, const std::string moduleName
)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct configuration parameters from a YAML node.
 * @param[in] fn Parsed YAML structure containing the configuration.
 */
DummyVisualTrackerConfigParams::DummyVisualTrackerConfigParams(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Read visual tracker parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of configuration parameters.
 */
void DummyVisualTrackerConfigParams::read(const YAML::Node &fn)
{
	/// Set the default values.
	*this = DummyVisualTrackerConfigParams();

	if(!fn.size())
	{
		return;
	}

	for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
	{
		std::string paramKey = fit->first.as<std::string>();

		if (paramKey == "debugVerboseLevel")
		{
			m_debugVerboseLevel = (int)fit->second.as<float>();
		}
		else if (paramKey == "visualTrackerType")
		{
			m_visualTrackerType = (VISUAL_TRACKER_TYPES)((int)fit->second.as<float>());
		}
	}

	if(m_debugVerboseLevel > 1)
	{
		cout << "[DummyVisualTracker] visualTrackerType = " << m_visualTrackerType << endl;
	}
}
