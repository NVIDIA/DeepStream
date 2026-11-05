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

#include "modules/Segmenter/DummySegmenter.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct a DummySegmenter instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
DummySegmenter::DummySegmenter(const std::string &configFilePath)
{
	m_isInit = false;

	/// Parse the module config.
	m_params = DummySegmenterConfigParams(configFilePath, "Segmenter");
}

/** @brief Release resources owned by the DummySegmenter instance. */
DummySegmenter::~DummySegmenter()
{
}

/**
 * @brief Mark the dummy segmenter as initialized.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void DummySegmenter::initialize(
	const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
	const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
	m_isInit = true;
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void DummySegmenter::queryParams(NvMOTQuery &paramsQuery)
{
}

/**
 * @brief Run segmentation to localize targets in the current frames.
 * @param[in] inputMeta A batch of input frames.
 */
void DummySegmenter::localize(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta)
{
}

/**
 * @brief Update retained state after data association.
 * @param[in] inputMeta A batch of input frames.
 */
void DummySegmenter::update(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta)
{
}

/**
 * @brief Populate the output mask for a tracked target.
 * @param[in] target The target.
 * @param[out] mask The output mask.
 */
void DummySegmenter::outputMaskObj(const std::shared_ptr<NvTrackedObject> target, NvMOTMask& mask)
{
}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void DummySegmenter::removeStream(const NvMOTStreamId videoStreamId)
{
}

/** @brief Construct a DummySegmenterConfigParams instance. */
DummySegmenterConfigParams::DummySegmenterConfigParams()
{
	m_debugVerboseLevel = 0;

	m_segmenterType = SEGMENTER_TYPES::DUMMY_SEGMENTER;
}

/**
 * @brief Construct a DummySegmenterConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 */
DummySegmenterConfigParams::DummySegmenterConfigParams(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct a DummySegmenterConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
DummySegmenterConfigParams::DummySegmenterConfigParams(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void DummySegmenterConfigParams::read(const YAML::Node &fn)
{
	/// Set the default values.
	*this = DummySegmenterConfigParams();

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
		else if (paramKey == "SegmenterType")
		{
			m_segmenterType = (SEGMENTER_TYPES)((int)fit->second.as<float>());
		}
	}

	/// Print the parsed segmenter type for debugging.
	if(m_debugVerboseLevel > 1)
	{
		cout << "[DummySegmenter] SegmenterType = " << m_segmenterType << endl;
	}
}
