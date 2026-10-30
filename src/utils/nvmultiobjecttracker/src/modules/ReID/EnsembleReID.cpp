/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "modules/ReID/EnsembleReID.hpp"

/**
 * @brief Construct a EnsembleReID instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
EnsembleReID::EnsembleReID(const std::string &configFilePath)
{
    m_isModelInit = false;
    m_isRuntimeInit = false;

    m_params = EnsembleReIDConfigParams(configFilePath, "ReID");

    /// Init individual reid models.
    if (m_params.m_modelNameList.empty())
    {
        /// Single model.
        if(m_params.m_debugVerboseLevel > 1)
        {
			std::cout << "[EnsembleReID] Ensemble method is disabled. Only use one ReID model. " << std::endl;
        }
        m_reidList.emplace_back(configFilePath);
    }
    else
    {
        /// Multiple models.
        if(m_params.m_debugVerboseLevel > 1)
        {
			std::cout << "[EnsembleReID] Ensemble method is enabled. " << std::endl;
        }

        if (m_params.m_enableModelId >= 0)
        {
            std::string modelName = ReIDModelNameParams::generateName(m_params.m_enableModelId);
            if (m_params.m_modelNameList.find(modelName) == m_params.m_modelNameList.end())
            {
                throw std::runtime_error(
                    "ReID model ID not found. ReID model names should have format modelx, where x is a non-negative int specified by enableModelId"
                );
            }
            m_reidList.emplace_back(configFilePath, modelName, true);
        }
        else
        {
            for (auto it = m_params.m_modelNameList.begin(); it != m_params.m_modelNameList.end(); it++)
            {
                /// Only the first model needs to compute Mahalanobis distance.
                bool calDisMaha = (it == m_params.m_modelNameList.begin());
                m_reidList.emplace_back(configFilePath, *it, calDisMaha);
            }
        }
    }
}

/** @brief Release resources owned by the EnsembleReID instance. */
EnsembleReID::~EnsembleReID()
{
}

/**
 * @brief Initialize runtime resources for the configured source.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void EnsembleReID::initializeRuntime(
    const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
    const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
    for (auto it = m_reidList.begin(); it != m_reidList.end(); it++)
    {
        it->initializeRuntime(frameInfo, maxTargetsPerStream, targetManager);
    }

    m_maxTargetsPerStream = maxTargetsPerStream;
    m_isRuntimeInit = true;
}

/** @brief Create or load the configured inference model. */
void EnsembleReID::initializeModel()
{
    for (auto it = m_reidList.begin(); it != m_reidList.end(); it++)
    {
        it->initializeModel();
    }
    m_isModelInit = true;
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void EnsembleReID::queryParams(NvMOTQuery &paramsQuery)
{
	paramsQuery.outputReidTensor = false;
	paramsQuery.reidFeatureSize = 0;
	paramsQuery.numTransforms = 1;

	for (auto it = m_reidList.begin(); it != m_reidList.end(); it++)
	{
		it->queryParams(paramsQuery);
	}
}

/**
 * @brief Run inference for the targets in the input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in,out] detectedObjects Newly detected objects in each stream.
 * @param[in] isForReAssoc Whether this inference is for reassociation.
 */
void EnsembleReID::doInference(
    const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
)
{
    for (auto it = m_reidList.begin(); it != m_reidList.end(); it++)
    {
        it->doInference(inputMeta, detectedObjects, isForReAssoc);
    }
}

/** @brief Update retained state after data association. */
void EnsembleReID::update()
{
    for (auto it = m_reidList.begin(); it != m_reidList.end(); it++)
    {
        it->update();
    }
}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void EnsembleReID::removeStream(const NvMOTStreamId videoStreamId)
{
    for (auto it = m_reidList.begin(); it != m_reidList.end(); it++)
    {
        it->removeStream(videoStreamId);
    }
}

/**
 * @brief Expose the current batch of ReID features.
 * @param[out] reidTensorBatch Batch reid tensor output to be filled.
 */
void EnsembleReID::outputReIDTensor(NvDsReidTensorBatch& reidTensorBatch)
{
    if (m_reidList.size() > 0)
    {
        m_reidList.at(0).outputReIDTensor(reidTensorBatch);
    }
}

/**
 * @brief Get a target feature index in the batch tensor.
 * @param[in] streamId Stream index.
 * @param[in] targetInd The target index.
 * @return The target's ReID vector index in the batch tensor, -1 means no reid feature.
 */
int EnsembleReID::outputTargetReIDInd(NvMOTStreamId streamId, uint targetInd)
{
    if (m_reidList.size() > 0)
    {
        return m_reidList.at(0).outputTargetReIDInd(streamId, targetInd);
    }
    else
    {
        return -1;
    }
}

/**
 * @brief Populate ReID metadata for a tracked target.
 * @param[in] streamId Stream index.
 * @param[in] targetInd The target index.
 * @param[in] reidTensorBatch Batch reid tensor to get the memory pointers.
 * @param[out] reidObj The target's ReID vector memory pointers and feature size.
 */
void EnsembleReID::outputTargetReIDObj(
    NvMOTStreamId streamId, uint targetInd, NvDsReidTensorBatch& reidTensorBatch, NvDsObjReid& reidObj
)
{
    if (m_reidList.size() > 0)
    {
        m_reidList.at(0).outputTargetReIDObj(streamId, targetInd, reidTensorBatch, reidObj);
    }
}

/**
 * @brief Combine similarity scores produced by the ensemble models.
 * @param[in] inputMeta Input frames keyed by stream ID.
 * @param[in,out] detectedObjects Detected objects keyed by stream ID.
 */
void EnsembleReID::fuseReIDSimilarity(
    const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
    for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
        /// Generate ReID distance scores for targets in this stream.
        const NvTrackedObjectList &targetList = m_targetManager->getTrackedObjectsInUseList(it->first);

        uint numDets = 0;
        if (detectedObjects.find(it->first) != detectedObjects.end())
        {
            numDets = detectedObjects.at(it->first).size();
        }
        else
        {
            /// Skip current frame because of no detection input.
            continue;
        }

        for(auto tIt = targetList.begin(); tIt != targetList.end(); ++tIt)
        {
            const std::shared_ptr<NvTrackedObject> target = tIt->second;
            std::vector<std::vector<float>> &confidenceListReidDets = target->getConfidenceListReidDets();
            if (confidenceListReidDets.size() == 1)
            {
                target->setConfidenceReidDets(std::move(confidenceListReidDets.at(0)));
            }
            else if (confidenceListReidDets.size() > 1)
            {
                target->setConfidenceReidDets(reidSimilarityFunc(confidenceListReidDets, numDets));
            }
        }
    }
}

/**
 * @brief Combine per-model detection similarities into one score vector.
 * @param[in] confidenceListReidDets Per-model ReID similarity values for detections.
 * @param[in] numDets Number of detections.
 * @return The highest similarity across the models for each detection.
 */
std::vector<float> EnsembleReID::reidSimilarityFunc(
    const std::vector<std::vector<float>> & confidenceListReidDets, const uint numDets
)
{
    /// Calculate the final reid score from all the reid models.
    std::vector<float> scoreDets(m_maxTargetsPerStream, 0);
    for (uint dBboxInd = 0; dBboxInd < numDets; dBboxInd++)
    {
        float score = 0;
        for (uint reidInd = 0; reidInd < confidenceListReidDets.size(); reidInd++)
        {
            score = std::max(score, confidenceListReidDets.at(reidInd).at(dBboxInd));
        }
        scoreDets[dBboxInd] = score;
    }
    return scoreDets;
}

/** @brief Construct a EnsembleReIDConfigParams instance. */
EnsembleReIDConfigParams::EnsembleReIDConfigParams()
{
    m_debugVerboseLevel = 0;
    m_enableModelId = 0;
    m_reidType = REID_TYPES::NvDEEPSORT;
}

/**
 * @brief Construct a EnsembleReIDConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 */
EnsembleReIDConfigParams::EnsembleReIDConfigParams(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct a EnsembleReIDConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
EnsembleReIDConfigParams::EnsembleReIDConfigParams(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void EnsembleReIDConfigParams::read(const YAML::Node &fn)
{
	/// Set the default values.
	*this = EnsembleReIDConfigParams();

    if(!fn.size())
    {
		return;
    }

    /// Ensemble required params.
    for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
		std::string paramKey = fit->first.as<std::string>();

		if (paramKey == "debugVerboseLevel")
		{
			m_debugVerboseLevel = (int)fit->second.as<float>();
		}
        else if (paramKey == "reidType")
        {
            m_reidType = (REID_TYPES) fit->second.as<float>();
        }
        else if (paramKey == "enableModelId")
        {
            m_enableModelId = (int)fit->second.as<float>();
        }
        else if (paramKey.rfind("model", 0) == 0)
        {
            if (ReIDModelNameParams::isValidName(paramKey))
            {
                m_modelNameList.insert(paramKey);
            }
        }
    }

    if(m_debugVerboseLevel > 1)
    {
		std::cout << "[EnsembleReID] ReIDType = " << m_reidType << std::endl;
    }
}
