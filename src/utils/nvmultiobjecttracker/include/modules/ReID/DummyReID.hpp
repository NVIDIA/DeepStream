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

#ifndef REIDLIB_DUMMYREID_HPP_
#define REIDLIB_DUMMYREID_HPP_

#include "modules/ReID/IReID.hpp"
#include "modules/VisualTracker/IVisualTracker.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Stores configuration for a ReID implementation. */
class DummyReIDConfigParams : public NvTrackerParams
{
public:
	/** @brief Construct a DummyReIDConfigParams instance. */
	DummyReIDConfigParams();

	/**
	 * @brief Construct a DummyReIDConfigParams instance.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Configuration module name.
	 */
	DummyReIDConfigParams(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Construct a DummyReIDConfigParams instance.
	 * @param[in] fn Parsed YAML node.
	 */
	DummyReIDConfigParams(const YAML::Node &fn);

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Debug message verbose mode level.
	REID_TYPES m_reidType; //!< The type of reid.
};

/** @brief Provides a no-op ReID implementation. */
class DummyReID : public IReID
{
public:

	/**
	 * @brief Construct a DummyReID instance from configuration.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 */
	DummyReID(const std::string &configFilePath);

	/** @brief Release resources owned by the DummyReID instance. */
	virtual ~DummyReID();

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image, such as size and color.
	 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
	 * @param[in] targetManager Target manager of the tracked objects.
	 */
	void initializeRuntime(
		const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
		const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) override;

    /** @brief Create TensorRT inference engine; if engine does not exist, serialize from model file. */
    void initializeModel() override;

	/**
	 * @brief Populate the capabilities reported to the tracker.
	 * @param[out] paramsQuery The output query parameters.
	 */
	void queryParams(NvMOTQuery &paramsQuery);

    /**
     * @brief Crop and rescale detected objects to create input batch to reid model.
     * @param[in] inputMeta A batch of input frames.
     * @param[in,out] detectedObjects Newly detected objects in each stream.
     * @param[in] isForReAssoc Whether this inference is for reassociation.
     */
    void doInference(
        const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
    ) override;

    /** @brief Update ReID feature gallery after data association. */
    void update() override;

    /**
     * @brief Remove a stream's reid features.
     * @param[in] videoStreamId Video stream id.
     */
    void removeStream(const NvMOTStreamId videoStreamId) override;

    /**
     * @brief Output the ReID tensor of a batch.
     * @param[out] reidTensorBatch Batch reid tensor output to be filled.
     */
    void outputReIDTensor(NvDsReidTensorBatch& reidTensorBatch) override;

    /**
     * @brief Output a target's ReID vector index in the batch tensor.
     * @param[in] streamId Stream index.
     * @param[in] targetInd The target index.
     * @return The target's ReID vector index in the batch tensor, -1 means no reid feature.
     */
    int outputTargetReIDInd(NvMOTStreamId streamId, uint targetInd) override;

    /**
     * @brief Output a target's ReID vector information.
     * @param[in] streamId Stream index.
     * @param[in] targetInd The target index.
     * @param[in] reidTensorBatch Batch reid tensor to get the memory pointers.
     * @param[out] reidObj The target's ReID vector memory pointers and feature size.
     */
    void outputTargetReIDObj(
        NvMOTStreamId streamId, uint targetInd, NvDsReidTensorBatch& reidTensorBatch, NvDsObjReid& reidObj
    ) override;

    /** @brief Report whether the inference model is initialized. */
    bool isModelInitialized() override { return m_isModelInit; }

    /** @brief Report whether runtime resources are initialized. */
    bool isRuntimeInitialized() override { return m_isRuntimeInit; }

    /** @brief Get the configured ReID implementation type. */
    const REID_TYPES getReIDType() override { return m_params.m_reidType; }

protected:

	bool m_isModelInit; //!< Reid status if reid model initialized.
    bool m_isRuntimeInit; //!< Reid status if runtime variables initialized.
	DummyReIDConfigParams m_params; //!< Config params for reid module.
};

#endif // REIDLIB_DUMMYREID_HPP_
