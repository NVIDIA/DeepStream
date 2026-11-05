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

#ifndef REIDLIB_IREID_HPP_
#define REIDLIB_IREID_HPP_

#include <deque>
#include <map>
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Types of ReID usage. */
enum REID_TYPES : int
{
    DUMMY_REID = 0, //!< No ReID feature extraction.
    NvDEEPSORT = 1 << 0, //!< ReID features used for the NvDeepSORT association.
    REASSOC = 1 << 1, //!< ReID features used for re-association.
    BOTH = (1 << 0) + (1 << 1) //!< ReID features used for both association and re-association.
};

/** @brief Defines the interface for extracting and managing ReID features. */
class IReID
{
public:
	/** @brief Release resources owned by the IReID instance. */
	virtual ~IReID() {}

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
     * @brief Crop and rescale detected objects to create input batch to reid model.
     * @param[in] inputMeta A batch of input frames.
     * @param[in,out] detectedObjects Newly detected objects in each stream.
     * @param[in] isForReAssoc Whether this inference is for reassociation.
     */
    virtual void doInference(
        const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
    ) = 0;

    /** @brief Update ReID feature gallery after data association. */
    virtual void update() = 0;

    /**
     * @brief Remove a stream's reid features.
     * @param[in] videoStreamId Video stream id.
     */
    virtual void removeStream(const NvMOTStreamId videoStreamId) = 0;

    /**
     * @brief Output the ReID tensor of a batch.
     * @param[out] reidTensorBatch Batch reid tensor output to be filled.
     */
    virtual void outputReIDTensor(NvDsReidTensorBatch& reidTensorBatch) = 0;

    /**
     * @brief Output a target's ReID vector index in the batch tensor.
     * @param[in] streamId Stream index.
     * @param[in] targetInd The target index.
     * @return The target's ReID vector index in the batch tensor, -1 means no reid feature.
     */
    virtual int outputTargetReIDInd(NvMOTStreamId streamId, uint targetInd) = 0;

    /**
     * @brief Output a target's ReID vector information.
     * @param[in] streamId Stream index.
     * @param[in] targetInd The target index.
     * @param[in] reidTensorBatch Batch reid tensor to get the memory pointers.
     * @param[out] reidObj The target's ReID vector memory pointers and feature size.
     */
    virtual void outputTargetReIDObj(
        NvMOTStreamId streamId, uint targetInd, NvDsReidTensorBatch& reidTensorBatch, NvDsObjReid& reidObj
    ) = 0;

    /** @brief Report whether the inference model is initialized. */
    virtual bool isModelInitialized() = 0;

    /** @brief Report whether runtime resources are initialized. */
    virtual bool isRuntimeInitialized() = 0;

    /** @brief Get the configured ReID implementation type. */
    virtual const REID_TYPES getReIDType() = 0;
};

#endif // REIDLIB_IREID_HPP_
