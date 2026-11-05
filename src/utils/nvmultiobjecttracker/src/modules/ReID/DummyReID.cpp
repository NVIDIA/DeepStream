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

#include "modules/ReID/DummyReID.hpp"

/**
 * @brief Construct a DummyReID instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
DummyReID::DummyReID(const std::string &configFilePath)
{
    m_isModelInit = false;
    m_isRuntimeInit = false;

    m_params = DummyReIDConfigParams(configFilePath, "ReID");
}

/** @brief Release resources owned by the DummyReID instance. */
DummyReID::~DummyReID()
{
}

/**
 * @brief Initialize runtime resources for the configured source.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void DummyReID::initializeRuntime(
    const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
    const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
	m_isRuntimeInit = true;
}

/** @brief Create or load the configured inference model. */
void DummyReID::initializeModel()
{
	m_isModelInit = true;
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void DummyReID::queryParams(NvMOTQuery &paramsQuery)
{
	paramsQuery.outputReidTensor = false;
	paramsQuery.reidFeatureSize = 0;
}

/**
 * @brief Run inference for the targets in the input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in,out] detectedObjects Newly detected objects in each stream.
 * @param[in] isForReAssoc Whether this inference is for reassociation.
 */
void DummyReID::doInference(
    const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
)
{
}

/** @brief Update retained state after data association. */
void DummyReID::update()
{
}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void DummyReID::removeStream(const NvMOTStreamId videoStreamId)
{
}

/**
 * @brief Expose the current batch of ReID features.
 * @param[out] reidTensorBatch Batch reid tensor output to be filled.
 */
void DummyReID::outputReIDTensor(NvDsReidTensorBatch& reidTensorBatch)
{
}

/**
 * @brief Get a target feature index in the batch tensor.
 * @param[in] streamId Stream index.
 * @param[in] targetInd The target index.
 * @return The target's ReID vector index in the batch tensor, -1 means no reid feature.
 */
int DummyReID::outputTargetReIDInd(NvMOTStreamId streamId, uint targetInd)
{
    return -1;
}

/**
 * @brief Populate ReID metadata for a tracked target.
 * @param[in] streamId Stream index.
 * @param[in] targetInd The target index.
 * @param[in] reidTensorBatch Batch reid tensor to get the memory pointers.
 * @param[out] reidObj The target's ReID vector memory pointers and feature size.
 */
void DummyReID::outputTargetReIDObj(
    NvMOTStreamId streamId, uint targetInd, NvDsReidTensorBatch& reidTensorBatch, NvDsObjReid& reidObj
)
{
}

/** @brief Construct a DummyReIDConfigParams instance. */
DummyReIDConfigParams::DummyReIDConfigParams()
{
    m_debugVerboseLevel = 0;
    m_reidType = REID_TYPES::DUMMY_REID;
}

/**
 * @brief Construct a DummyReIDConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 */
DummyReIDConfigParams::DummyReIDConfigParams(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct a DummyReIDConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
DummyReIDConfigParams::DummyReIDConfigParams(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void DummyReIDConfigParams::read(const YAML::Node &fn)
{
	/// Set the default values.
	*this = DummyReIDConfigParams();

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
    }

    /// Print the parsed configuration for debugging.
    if(m_debugVerboseLevel > 1)
    {
		std::cout << "[DummyReID] reIDType = " << m_reidType << std::endl;
    }
}
