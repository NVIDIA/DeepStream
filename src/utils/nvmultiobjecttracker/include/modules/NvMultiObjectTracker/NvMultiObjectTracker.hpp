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

#ifndef NVMULTIOBJECTTRACKERBASE_HPP_
#define NVMULTIOBJECTTRACKERBASE_HPP_

#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <deque>
#include <random>
#include <limits>
#ifndef _WIN32
#include <sys/syscall.h>
#endif

#include "nvdstracker.h"
#include "nvds_tracker_meta.h"

#include "modules/utility/debug_common.hpp"
#include "modules/DataAssociator/DataAssociator_factory.hpp"
#include "modules/VisualTracker/VisualTracker_factory.hpp"
#include "modules/ReID/ReID_factory.hpp"
#include "modules/Segmenter/Segmenter_factory.hpp"
#include "modules/ReIDService/ReIDService_factory.hpp"
#include "modules/PoseEstimator/PoseEstimator_factory.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/localizer.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Stores configuration parameters used by the base tracker. */
class NvTrackerParamsBase : public NvTrackerParams
{
public:
	/** @brief Initialize the base tracker parameters with default values. */
	NvTrackerParamsBase();

	/**
	 * @brief Load the base tracker parameters from a configuration module.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Name of the configuration module to load.
	 */
	NvTrackerParamsBase(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Load the base tracker parameters from a YAML node.
	 * @param[in] fn YAML node containing the base tracker configuration.
	 */
	NvTrackerParamsBase(const YAML::Node &fn);

	/**
	 * @brief Read base tracker parameters from a YAML node.
	 * @param[in] fn YAML node containing the base tracker configuration.
	 */
	void read(const YAML::Node &fn) override;

	// [General]
	int m_debugVerboseLevel; //!< Debug message verbose mode level.
	bool m_isInit = false; //!< Indicates whether the parameters have been initialized.
	//!< Prevents changing static params in dynamic updates.

	// [Class match]
	bool m_checkClassMatch; //!< Restricts association to objects of the same class when enabled.

	// [Target candidacy]
	float m_minDetectorConfidence; //!< Minimum detector confidence required for tracking.

	// [Frame ID]
	bool m_useBatchNumForFrameId; //!< Selects the batch counter instead of the per-stream frame number for cross-stream synchronization.
};

/** @brief Coordinates tracker submodules to process batched multi-stream input. */
class NvMultiObjectTrackerBase : public IBatchLocalizer
{
public:
	/**
	 * @brief Constructor.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 */
	NvMultiObjectTrackerBase(const std::string &configFilePath);

	/** @brief Release the tracker submodules. */
	~NvMultiObjectTrackerBase();

	/** @brief Report whether the tracker is initialized. */
	bool isInitialized() { return m_isInit; }

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] srcImgSize Source image size.
	 * @param[in] srcImgPitch Source image row pitch in bytes.
	 */
	virtual void initialize(const NvMotDataTypes::Size srcImgSize, const size_t srcImgPitch);

	/**
	 * @brief Perform multi-object tracking for a batch of frames from multiple streams, with at most one frame per stream.
	 * @param[in] nvFramesInBatch Source frames and detector bboxes keyed by video stream ID.
	 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch filled with tracked objects for this frame batch.
	 */
	virtual void update(
		const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch, NvMOTTrackedObjBatch *&pTrackedObjectsBatch
	);

	/**
	 * @brief Populate tracker capability and memory requirement fields.
	 * @param[out] paramsQuery Query structure populated with tracker requirements.
	 */
	void queryParams(NvMOTQuery &paramsQuery);

	/**
	 * @brief Output past-frame tracking data to tracker plugin and clear the buffer.
	 * @param[in] videoStreamIdList Stream IDs whose past-frame tracking data is reported.
	 * @param[in,out] pPastFrameObjBatch Caller-allocated batch that receives the past-frame objects.
	 */
	void outputPastFrameObjs(
		const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pPastFrameObjBatch
	);

	/**
	 * @brief Output complete frame history for all terminated tracks to the plugin.
	 * @param[in] videoStreamIdList Stream IDs whose terminated tracks are reported.
	 * @param[in,out] pPastFrameObjBatch Caller-allocated batch that receives the terminated-track history.
	 */
	void outputTerminatedObjs(
		const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pPastFrameObjBatch
	);

	/**
	 * @brief Output most recent frame history for current shadow tracks to the plugin.
	 * @param[in] videoStreamIdList Stream IDs whose shadow tracks are reported.
	 * @param[in,out] pShadowTracksBatch Caller-allocated batch that receives the latest shadow-track frame.
	 */
	void outputShadowObj(const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pShadowTracksBatch);

	/**
	 * @brief Delete trackers when a stream is removed.
	 * @param[in] videoStreamID Stream ID whose trackers are removed.
	 */
	void deleteRemovedStreamTrackers(const NvMOTStreamId videoStreamID);

	/**
	 * @brief Apply supported runtime tracker configuration updates.
	 * @param[in] configStr Stringified YAML configuration parameters to update.
	 */
	void updateParams(const std::string &configStr);

	/** @brief Return the configured input surface color format. */
	NvBufSurfaceColorFormat getColorFormatType() const { return m_frameInfo.colorFormat; }

	/**
	 * @brief Set the input surface color format.
	 * @param[in] colorFormat Input surface color format.
	 */
	void setColorFormatType(const NvBufSurfaceColorFormat colorFormat) { m_frameInfo.colorFormat = colorFormat; }

	/** @brief Return the maximum number of supported streams. */
	uint32_t getMaxStreams() const { return m_frameInfo.maxStreams; }

	/**
	 * @brief Set the maximum number of supported streams.
	 * @param[in] maxStreams Maximum number of supported streams.
	 */
	void setMaxStreams(const uint32_t maxStreams) { m_frameInfo.maxStreams = maxStreams; }

	/**
	 * @brief Set the maximum surface-buffer address pool size.
	 * @param[in] maxBufSurfAddrSize Maximum number of surface-buffer addresses.
	 */
	void setMaxBufSurfAddrSize(const uint32_t maxBufSurfAddrSize)
	{
		m_frameInfo.maxBufSurfAddrSize = maxBufSurfAddrSize;
	}

protected:

	// [Config params set by parsing the config file or by deriving from it]
	NvTrackerParamsBase m_paramsBase; //!< Configuration parameters used by the base tracker.
	const std::string m_configFilePath; //!< Path to the tracker configuration file.

	// [Info about the source frame batch]
	SourceFrameInfo m_frameInfo; //!< Source video frame information.
	std::map<NvMOTStreamId, NvMOTFrame *> m_inputMeta; //!< Input frame metadata keyed by stream ID.
	std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> m_detectedObjects; //!< Valid detected objects obtained from the input metadata.

	// [Target management module (including the state estimator for each target)]
	std::shared_ptr<NvTrackedObjectManager> m_targets; //!< Targets in use and in the object pool.

	// [Data association module]
	std::shared_ptr<IDataAssociator> m_dataAssociator; //!< Data association module.

	// [Visual tracker module]
	std::shared_ptr<IVisualTracker> m_visualTracker; //!< Visual tracker module for target localization.

	// [ReID module]
	std::shared_ptr<IReID> m_reid; //!< ReID module.

	// [Segmenter module]
	std::shared_ptr<ISegmenter> m_segmenter; //!< Object segmentation module.

	// [Pose estimator module]
	std::shared_ptr<IPoseEstimator> m_poseEstimator; //!< Pose estimation module.

	// [Re-ID service module]
	std::shared_ptr<IReIDService> m_reidService; //!< Re-ID Service module.

	// [Camera model module]
	std::map<const NvMOTStreamId, std::shared_ptr<ObjectModelProjector>> m_objModelProjectors; //!< Refine bbox based on 3D object model.

	// [Multi-threaded producer-consumer for stream specific tasks]
	std::shared_ptr<NvTrackerMultiThreadWorker<assocDataIn_t>> m_threadWorker; //!< Multi-threaded producer-consumer for stream specific tasks.
	bool m_isInit; //!< Indicates whether the base tracker is initialized.
	bool m_isInitSubModules; //!< Indicates whether all tracker submodules are initialized.
	REID4REASSOC_STATE m_reid4ReAssocState; //!< Indicates whether per-stream input must be resubmitted for ReID-based reassociation.

	/** @brief Initialize the sub-modules. */
	void initializeSubModules();

	/** @brief Add per stream submodules runtime whenever there is a new stream coming in. */
	void addStreamSubModulesRuntime();

	/**
	 * @brief Check the input and output for the update function.
	 * @param[in] nvFramesInBatch Input frames and detector objects keyed by stream ID.
	 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch whose filled counts are cleared before tracking.
	 * @return False when there are no input frames and no existing trackers; true otherwise.
	 */
	bool checkInputOutput(
		const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch, NvMOTTrackedObjBatch *&pTrackedObjectsBatch
	);

	/** @brief Parse and filter the input metadata and store only valid detected objects. */
	void parseInputMetadata();

	/** @brief Recover the full bbox if 3D camera params are provided. */
	void recoverFullBbox();

	/** @brief Submit all streams' objects from main thread to each working thread to perform data association, update and re-association. */
	void performDataAssociationAndUpdate();

	/**
	 * @brief Function called in each thread to perform each stream's data association and update.
	 * @param[in] in Association input for one stream.
	 */
	void performDataAssociationAndUpdatePerStream(assocDataIn_t in);

	/**
	 * @brief Fill out the output metadata.
	 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch filled with tracked objects for this frame batch.
	 */
	void setOutputData(NvMOTTrackedObjBatch *&pTrackedObjectsBatch);

	/** @brief Advance batch state and emit optional debug output. */
	void wrapUpProcessingFrameBatch();

	friend class NvTrackedObject;
};

#endif // NVMULTIOBJECTTRACKERBASE_HPP_
