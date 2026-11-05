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

#include <chrono>
#include "modules/utility/cv_functions.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker.hpp"

using std::cout;
using std::endl;

/**
 * @brief Initialize tracker state from a configuration file.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
NvMultiObjectTrackerBase::NvMultiObjectTrackerBase(const std::string &configFilePath)
: m_configFilePath(configFilePath)
{
	m_isInit = false;
	m_isInitSubModules = false;
	m_reid4ReAssocState = NOT_NEEDED;

	/// Parse config file for base tracker module.
	m_paramsBase = NvTrackerParamsBase(m_configFilePath, "BaseConfig");
}

/** @brief Release the tracker submodules. */
NvMultiObjectTrackerBase::~NvMultiObjectTrackerBase()
{
	m_dataAssociator.reset();
	m_visualTracker.reset();
	m_targets.reset();
	m_reid.reset();
	m_threadWorker.reset();

	cout << "[NvMultiObjectTracker] De-initialized" << endl;
}

/**
 * @brief Initialize variables and allocate memories.
 * @param[in] imgSize Source image size.
 * @param[in] srcImagePitch Source image row pitch in bytes.
 */
void NvMultiObjectTrackerBase::initialize(const NvMotDataTypes::Size imgSize, const size_t srcImagePitch)
{
	if(m_isInit)
		return;

	if(m_frameInfo.maxStreams == 0)
	{
		throw std::invalid_argument("m_maxStreams == 0 !! Make sure m_maxStreams > 0");
	}

	m_frameInfo.srcImageSize = imgSize;
	m_frameInfo.srcImagePitch = srcImagePitch;

	/// Pre-allocate all the memories for targets at once.
	m_targets = std::make_shared<NvTrackedObjectManager>(m_configFilePath);

	/// Instantiate a visual tracker based on the config.
	m_visualTracker = VisualTrackerFactory::getInstance().makeVisualTracker(m_configFilePath);

	/// Instantiate a data association module and set the association functor to that.
	m_dataAssociator = DataAssociatorFactory::getInstance().makeDataAssociator(m_configFilePath);

	/// Initialize ReID module and create ReID model engine.
	m_reid = ReIDFactory::getInstance().makeReID(m_configFilePath, m_visualTracker->getVisualTrackerType());
	m_reid->initializeModel();

	m_segmenter = SegmenterFactory::getInstance().makeSegmenter(m_configFilePath);
	m_poseEstimator = PoseEstimatorFactory::getInstance().makePoseEstimator(m_configFilePath);
	m_reidService = ReIDServiceFactory::getInstance().makeReIDService(m_configFilePath);

	m_threadWorker = std::make_shared<NvTrackerMultiThreadWorker<assocDataIn_t>>();

	if(m_paramsBase.m_debugVerboseLevel > 1)
	{
		cout << "[NvMultiObjectTracker] MaxStreams = " << m_frameInfo.maxStreams << endl;
		cout << "[NvMultiObjectTracker] MaxTargetsToTrack = " << m_frameInfo.maxStreams*m_targets->getMaxTargetsPerStream() << endl;
	}

	m_frameInfo.frameBatchNo = 0;

	m_isInit = true;

	cout << "[NvMultiObjectTracker] Initialized" << endl;
}

/**
 * @brief Populate tracker capability and memory requirement fields.
 * @param[out] paramsQuery Query structure populated with tracker requirements.
 */
void NvMultiObjectTrackerBase::queryParams(NvMOTQuery &paramsQuery)
{
	if (m_targets)
	{
		m_targets->queryParams(paramsQuery);
	}
	else
	{
		throw std::runtime_error("[NvMultiObjectTracker] Failed to query from TargetManager");
	}
	if (m_visualTracker)
	{
		m_visualTracker->queryParams(paramsQuery);
	}
	else
	{
		throw std::runtime_error("[NvMultiObjectTracker] Failed to query from VisualTracker");
	}
	if (m_segmenter)
	{
		m_segmenter->queryParams(paramsQuery);
	}
	else
	{
		throw std::runtime_error("[NvMultiObjectTracker] Failed to query from Segmenter");
	}
	if (m_reid)
	{
		m_reid->queryParams(paramsQuery);
	}
	else
	{
		throw std::runtime_error("[NvMultiObjectTracker] Failed to query from ReID");
	}
	if (m_poseEstimator)
	{
		m_poseEstimator->queryParams(paramsQuery);
	}
	if (m_reidService)
	{
		m_reidService->queryParams(paramsQuery);
	}

	ObjectModelProjector objModelProjector(m_configFilePath);
	objModelProjector.queryParams(paramsQuery);
}

/**
 * @brief Validate batch input and reset output storage for a tracker update.
 * @param[in] inputMeta Input frames and detector objects keyed by stream ID.
 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch whose filled counts are cleared before tracking.
 * @return False when there are no input frames and no existing trackers; true otherwise.
 */
bool NvMultiObjectTrackerBase::checkInputOutput(
	const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta, NvMOTTrackedObjBatch *&pTrackedObjectsBatch
)
{
	/// Clean previous batch data.
	m_targets->cleanPreviousBatchData();

	/// Reset the output data.
	pTrackedObjectsBatch->numFilled = 0;
	for(uint32_t i=0; i<pTrackedObjectsBatch->numAllocated; i++)
	{
		pTrackedObjectsBatch->list[i].numFilled = 0;
		pTrackedObjectsBatch->list[i].valid = false;
	}

	/// {videoStreamID, {MotFrame w/ detector bboxes} }.
	m_inputMeta = inputMeta;

	m_frameInfo.isDetectionDonePerStream.clear();

	for (auto it = m_inputMeta.begin(); it != m_inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		const NvMOTFrame *srcMotFrame = it->second;

		/// Check if there are multiple frames from the same source.
		if(m_frameInfo.isDetectionDonePerStream.find(videoStreamId) != m_frameInfo.isDetectionDonePerStream.end())
		{
			throw std::runtime_error(
				"[NvMultiObjectTrackerBase::checkInputOutput] There must be at most a single frame from the same stream, but there are multiple frames from the same stream in this frame batch!!"
			);
		}

		m_frameInfo.isDetectionDonePerStream[videoStreamId] = std::make_pair(
			srcMotFrame->frameNum, srcMotFrame->objectsIn.detectionDone
		);
	}

	if(0 && m_paramsBase.m_debugVerboseLevel > 1)
	{
		if(m_frameInfo.isDetectionDonePerStream.size() > 0)
		{
			printf(
				"[NvMultiObjectTrackerBase::checkInputOutput] FrameBatch[%d] Detection done.\n",
				(int)m_frameInfo.frameBatchNo
			);
			for (auto it = m_inputMeta.begin(); it != m_inputMeta.end(); ++it)
			{
				const NvMOTStreamId &videoStreamId = it->first;
				printf(
					"  - Stream[%d]: Frame[%d] isDetectionDone = %d\n",
					(int)(videoStreamId>>32), m_frameInfo.isDetectionDonePerStream[videoStreamId].first,
					m_frameInfo.isDetectionDonePerStream[videoStreamId].second
				);
			}
		}
	}

	if (m_frameInfo.isDetectionDonePerStream.size() == 0 && m_targets->getLastTargetInd() < 0)
	{
		/// Finish the batch when there is no detector input or existing tracker.
		if(m_paramsBase.m_debugVerboseLevel > 4)
			CLOG(
				std::string( "[Info] NvMultiObjectTrackerBase::checkInputOutput() at FrameBatch[ " + std::to_string(m_frameInfo.frameBatchNo) + "]: No detector input and no existing trackers" ).c_str()
			);

		m_frameInfo.frameBatchNo++;
		return false;
	}

	return true;
}

/** @brief Parse and filter the input metadata and store only valid detected objects. */
void NvMultiObjectTrackerBase::parseInputMetadata()
{
	m_detectedObjects.clear();

	for(auto it = m_inputMeta.begin(); it != m_inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		NvMOTFrame *motFrame = it->second;
		const NvMOTObjToTrackList &detBboxes_perStream = motFrame->objectsIn;

		/// Frame id for this batch: either global batch counter or per-stream source index (written in-place on input).
		const NvMOTFrameNum resolvedFrameId = m_paramsBase.m_useBatchNumForFrameId
			? static_cast<NvMOTFrameNum>(m_frameInfo.frameBatchNo)
			: motFrame->frameNum;
		motFrame->frameNum = resolvedFrameId;
		motFrame->timeMeta.frameNum = resolvedFrameId;

		/// The original source video frame size, assuming they are the same in all the frame batch.
		m_frameInfo.originalSrcImageSize = NvMotDataTypes::Size(motFrame->srcFrameWidth, motFrame->srcFrameHeight);

		/// Mark this stream as valid even when it has no detected objects.
		m_detectedObjects[videoStreamId] = std::deque<DetectedObjInfo>();

		if(detBboxes_perStream.numFilled > 0 && m_frameInfo.isDetectionDonePerStream[videoStreamId].second == true)
		{
			/// Visit each detector bounding box in the video stream.
			for(uint i=0; i < detBboxes_perStream.numFilled; i++)
			{
				const NvMOTObjToTrack *detObj = &detBboxes_perStream.list[i];

				/// Filter out un-qualified detector bboxes.
				if(detObj->doTracking == false)
				{
					continue;
				}

				DetectedObjInfo detObjInfo(detObj, i);

				/// Replace an unavailable negative detector confidence with one.
				detObjInfo.confidence = detObj->confidence < 0 ? 1 : detObj->confidence;

				/// Store valid detected objects.
				m_detectedObjects.at(videoStreamId).push_back(detObjInfo);
			}
		}
	}
}

/** @brief Submit each stream to worker threads for association, update, and reassociation. */
void NvMultiObjectTrackerBase::performDataAssociationAndUpdate()
{
	/// When reid based reassoc is enabled, batched reid launched in main thread after target update.
	m_reid4ReAssocState = m_reid->getReIDType() == REID_TYPES::REASSOC ? TO_DO : NOT_NEEDED;

	/// Update stream id in target manager.
	m_targets->updateInputInfo(m_inputMeta);

	/// Submit per-stream association work, deferring trajectory updates until after ReID when reassociation is enabled.
	for(auto it = m_inputMeta.begin(); it != m_inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		const NvMOTFrameNum frameNum = it->second->frameNum;
		NvMOTTimeMeta timeMeta = it->second->timeMeta;

		/// Create a pointer list to the source input data; if no inference, det obj is empty.
		std::deque<const DetectedObjInfo *> detObjPtrs;
		bool isDetDone = m_frameInfo.isDetectionDonePerStream.at(videoStreamId).second;
		if (isDetDone && m_detectedObjects.find(videoStreamId) != m_detectedObjects.end())
		{
			const std::deque<DetectedObjInfo>& detObjsPerStream = m_detectedObjects.at(videoStreamId);
			for(auto detIt = detObjsPerStream.begin(); detIt != detObjsPerStream.end(); ++detIt)
			{
				const DetectedObjInfo *detObjPtr = &(*detIt);

				/// Consider only qualified detector bboxes for data association.
				if(detObjPtr->confidence >= m_paramsBase.m_minDetectorConfidence)
				{
					detObjPtrs.push_back(detObjPtr);
				}
			}
		}
		else if (isDetDone)
		{
			throw std::runtime_error("Detection is done but cannot find detection results");
		}

		/// Compose the input to the data association module.
		assocDataIn_t inData = {
			videoStreamId,
			frameNum,
			timeMeta,
			isDetDone,
			detObjPtrs,
			m_targets->getTrackedObjectsInUseList(videoStreamId),
			{},
			m_targets->getMultiViewAssociators(videoStreamId)
		};

		/// Submit the association for each stream.
		m_threadWorker->submitInput(std::move(inData));
	}
	/// Sync all worker threads.
	m_threadWorker->waitToFinish();

	/// Update targets on the main thread after worker completion when stream order is preserved.
	m_targets->updateTargets();

	/// Run pose estimation.
	if (m_poseEstimator)
	{
		m_poseEstimator->doInference(m_inputMeta, m_detectedObjects, false);
	}

	/// For reid based reassoc, run reid inference and call multi-thread API a second time for multi-thread traj update Skipped if reid based re-assoc is disabled (no re-assoc or motion re-assoc only).
	if (m_reid4ReAssocState == TO_DO)
	{
		/// Extract ReID features for reassociation.
		std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> objects4Reid = m_targets->generateTrackedObjects4Reid(
			m_inputMeta
		);
		m_reid->doInference(m_inputMeta, objects4Reid, true);

		m_reid4ReAssocState = DONE;

		/// Submit trajectory update task for each stream.
		for(auto it = m_inputMeta.begin(); it != m_inputMeta.end(); ++it)
		{
			const NvMOTStreamId &videoStreamId = it->first;
			const NvMOTFrameNum frameNum = it->second->frameNum;
			NvMOTTimeMeta timeMeta = it->second->timeMeta;

			bool isDetDone = m_frameInfo.isDetectionDonePerStream.at(videoStreamId).second;

			/// Create association input containing only the stream and frame identifiers.
			assocDataIn_t inData = {
				videoStreamId,
				frameNum,
				timeMeta,
				isDetDone,
				{},
				{},
				{}
			};

			/// Submit trajectory update and re-assoc task for each stream.
			m_threadWorker->submitInput(std::move(inData));
		}
		/// Sync all worker threads.
		m_threadWorker->waitToFinish();
	}
	/// Update trajectories on the main thread after target or ReID work when stream order is preserved.
	m_targets->updateTrajectoryManager();
}

/**
 * @brief Perform data association and target updates for one stream.
 * @param[in] in Association input for one stream.
 */
void NvMultiObjectTrackerBase::performDataAssociationAndUpdatePerStream(assocDataIn_t in)
{
	/// Init assoc out with only stream and frame id.
	assocDataOut_t outData = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		{},
		{},
		{}
	};

	/// M_reid4ReAssocState NOT_NEEDED or TO_DO: perform data association, because this is the first call for multi-thread API m_reid4ReAssocState DONE: skip data association, because is this is the second call for multi-thread API, only perform re-assoc in each thread.
	if (m_reid4ReAssocState != DONE)
	{
		outData = m_dataAssociator->associatePerStream(in);
	}

	m_targets->updatePerStream(outData, m_reid4ReAssocState);
}

/** @brief Initialize tracker submodules that are not yet ready. */
void NvMultiObjectTrackerBase::initializeSubModules()
{
	if(!m_targets->isInitialized())
	{
		/// Initialize the object model projector.
		m_targets->initialize(m_frameInfo);
	}

	if(!m_dataAssociator->isInitialized())
	{
		/// Initialize the default object model.
		m_dataAssociator->initialize(
			m_frameInfo, m_targets->getMaxTargetsPerStream(),
			(m_visualTracker->getVisualTrackerType() == VISUAL_TRACKER_TYPES::NvDCF_VOT || m_visualTracker->getVisualTrackerType() == VISUAL_TRACKER_TYPES::NvDCF_VPI_VOT),
			(m_reid->getReIDType() == REID_TYPES::NvDEEPSORT || m_reid->getReIDType() == REID_TYPES::BOTH)
		);
	}

	if(!m_visualTracker->isInitialized())
	{
		/// Initialize the visual tracker to allocate memories.
		m_visualTracker->initialize(
			m_frameInfo, m_frameInfo.maxStreams*m_targets->getMaxTargetsPerStream(),
			m_targets->getMaxTargetsPerStream(), m_targets
		);
	}

	if(!m_reid->isRuntimeInitialized())
	{
		/// Load model at run time.
		m_reid->initializeRuntime(m_frameInfo, m_targets->getMaxTargetsPerStream(), m_targets);
	}

	if (!m_segmenter->isInitialized())
	{
		m_segmenter->initialize(m_frameInfo, m_targets->getMaxTargetsPerStream(), m_targets);
	}

	if (m_poseEstimator)
	{
		m_poseEstimator->initializeRuntime(m_frameInfo, m_targets->getMaxTargetsPerStream(), m_targets);
	}

	if (m_reidService)
	{
		m_reidService->initialize(m_frameInfo, m_targets->getMaxTargetsPerStream(), m_targets);
	}

	if(!m_threadWorker->isInitialized())
	{
		/// Initialize multi-thread worker to perform data association and target update.
		m_threadWorker->initialize(
			m_frameInfo.maxStreams, [this](assocDataIn_t in) { this->performDataAssociationAndUpdatePerStream(in); }
		);
	}

	if (m_paramsBase.m_debugVerboseLevel > 2)
	{
		cout << "[NvMultiObjectTracker] SubModules Initialized" << endl;
	}

	m_isInitSubModules = true;
}

/** @brief Add per-stream submodules for newly observed streams. */
void NvMultiObjectTrackerBase::addStreamSubModulesRuntime()
{
	m_targets->addStreamSubModules(m_inputMeta, m_frameInfo.srcImageSize);
}

/** @brief Recover full object models and bounding boxes when camera models are available. */
void NvMultiObjectTrackerBase::recoverFullBbox()
{
	for(auto streamIt = m_detectedObjects.begin(); streamIt != m_detectedObjects.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamId = streamIt->first;

		for(auto objIt = streamIt->second.begin(); objIt != streamIt->second.end(); ++objIt)
		{
			DetectedObjInfo &detObjInfo = *objIt;

			/// Estimate a full-body box for partial or duplicate detections with the projected 3D human model.
			m_targets->recoverDefaultModelFromBbox(videoStreamId, m_inputMeta.at(videoStreamId)->frameNum, detObjInfo);
		}

		/// Compare all the detected bboxes with each other to get the unoccluded bbox and visibility.
		for (auto objIt = streamIt->second.begin(); objIt != streamIt->second.end(); ++objIt)
		{
			DetectedObjInfo &detObjInfo = *objIt;
			NvMotDataTypes::Rect2f unOccludedBbox = detObjInfo.origBbox;
			for (auto objIt2 = streamIt->second.begin(); objIt2 != streamIt->second.end(); ++objIt2)
			{
				if (objIt2 == objIt)
				{
					continue;
				}
				DetectedObjInfo &detObjInfo2 = *objIt2;
				/// If the distance of detObjInfo from camera is less than the distance of detObjInfo2 from camera then visisbility is not affected since the object is in front, hence continue.
				if (detObjInfo.distFromCam < detObjInfo2.distFromCam)
				{
					continue;
				}
				/// Compute the area that is NOT in detObjInfo2 by subtracting the intersection.
				if (unOccludedBbox.intersects(detObjInfo2.origBbox))
				{
					NvMotDataTypes::Rect2f intersection = unOccludedBbox.intersection(detObjInfo2.origBbox);
					Eigen::Vector2f unOccMin = unOccludedBbox.min();
					Eigen::Vector2f unOccMax = unOccludedBbox.max();
					Eigen::Vector2f intersectMin = intersection.min();
					Eigen::Vector2f intersectMax = intersection.max();
					float leftArea = 0.0f, rightArea = 0.0f, topArea = 0.0f, bottomArea = 0.0f;
					if (intersectMin(0) > unOccMin(0))
					{
						leftArea = (intersectMin(0) - unOccMin(0)) * (unOccMax(1) - unOccMin(1));
					}
					if (intersectMax(0) < unOccMax(0))
					{
						rightArea = (unOccMax(0) - intersectMax(0)) * (unOccMax(1) - unOccMin(1));
					}
					if (intersectMin(1) > unOccMin(1))
					{
						topArea = (unOccMax(0) - unOccMin(0)) * (intersectMin(1) - unOccMin(1));
					}
					if (intersectMax(1) < unOccMax(1))
					{
						bottomArea = (unOccMax(0) - unOccMin(0)) * (unOccMax(1) - intersectMax(1));
					}
					float maxArea = std::max({leftArea, rightArea, topArea, bottomArea});
					if (maxArea > 0.0f)
					{
						if (maxArea == leftArea)
						{
							/// Keep left portion.
							unOccludedBbox = CV_FUNCS::constructRect2f(
								unOccMin(0), unOccMin(1), intersectMin(0) - unOccMin(0), unOccMax(1) - unOccMin(1)
							);
						}
						else if (maxArea == rightArea)
						{
							/// Keep right portion.
							unOccludedBbox = CV_FUNCS::constructRect2f(
								intersectMax(0), unOccMin(1), unOccMax(0) - intersectMax(0), unOccMax(1) - unOccMin(1)
							);
						}
						else if (maxArea == topArea)
						{
							/// Keep top portion.
							unOccludedBbox = CV_FUNCS::constructRect2f(
								unOccMin(0), unOccMin(1), unOccMax(0) - unOccMin(0), intersectMin(1) - unOccMin(1)
							);
						}
						else
						{
							/// Keep bottom portion.
							unOccludedBbox = CV_FUNCS::constructRect2f(
								unOccMin(0), intersectMax(1), unOccMax(0) - unOccMin(0), unOccMax(1) - intersectMax(1)
							);
						}
					}
					else
					{
						/// Fully occluded, set to empty or minimal bbox.
						unOccludedBbox = CV_FUNCS::constructRect2f(unOccMin(0), unOccMin(1), 0.0f, 0.0f);
						break;
					}
				}
			}
			/// Compute the recovered bounding box's visibility relative to its unoccluded extent.
			float visibility = CV_FUNCS::computeVisibility(detObjInfo.bbox,unOccludedBbox);
			if (visibility <= detObjInfo.visibility)
			{
				if (m_paramsBase.m_debugVerboseLevel > 4)
				{
					cout << "StreamID: " << videoStreamId << ", bboxInd: " << detObjInfo.bboxInd << ", Updating visibility from " << detObjInfo.visibility << " to " << visibility << endl;
				}
				detObjInfo.visibility = visibility;
			}
			else
			{
				throw std::runtime_error(
					"Visibility of recovered bbox w.r.t. unoccluded bbox is greater than the visibility of recovered bbox w.r.t. detected bbox"
				);
			}
		}
	}
}

/**
 * @brief Perform multi-object tracking for a batch containing at most one frame per stream.
 * @param[in] inputMeta Source frames and detector bboxes keyed by video stream ID.
 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch filled with tracked objects for this frame batch.
 */
void NvMultiObjectTrackerBase::update(
	const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta, NvMOTTrackedObjBatch *&pTrackedObjectsBatch
)
{
	/// Check the input and output for the update function.
	if(!checkInputOutput(inputMeta, pTrackedObjectsBatch))
		return;

	/// Parse and filter the input metadata and store only valid detected objects.
	parseInputMetadata();

	/// Initialize sub-modules if not done.
	if(!m_isInitSubModules)
	{
		initializeSubModules();
	}

	/// When there are new streams, initialize per stream modules.
	addStreamSubModulesRuntime();

	/// Recover the full bbox if 3D camera params are provided.
	MEAS_TIME( recoverFullBbox(), m_paramsBase.m_debugVerboseLevel > 3, "recoverFullBbox" );

	/// Perform reid service update to update target IDs based on response from reid service.
	MEAS_TIME(
		m_reidService->updateTargetIDs(), m_paramsBase.m_debugVerboseLevel > 3, "m_reidService->updateTargetIDs"
	);

	/// Perform visual detection/localization in existing trackers in case the visual tracker module is enabled.
	MEAS_TIME(
		m_visualTracker->localize(m_inputMeta, m_detectedObjects),
		m_paramsBase.m_debugVerboseLevel > 3, "m_visualTracker->localize"
	);

	/// Perform localization by segmentation.
	MEAS_TIME( m_segmenter->localize(m_inputMeta), m_paramsBase.m_debugVerboseLevel > 3, "m_segmenter->localize");

	/// Run network inference unless ReID is used only for reassociation.
	MEAS_TIME(
		m_reid->doInference(m_inputMeta, m_detectedObjects, false),
		m_paramsBase.m_debugVerboseLevel > 3, "m_reid->doInference"
	);

	/// Associate and update targets, including interval-based ReID inference and measurement-free non-inference frames.
	MEAS_TIME(
		performDataAssociationAndUpdate(), m_paramsBase.m_debugVerboseLevel > 3, "performDataAssociationAndUpdate"
	);

	/// Perform reid service query for all unmatched active objects.
	MEAS_TIME( m_reidService->query(m_inputMeta), m_paramsBase.m_debugVerboseLevel > 3, "m_reidService->query" );

	/// Perform segmenter update.
	MEAS_TIME( m_segmenter->update(m_inputMeta), m_paramsBase.m_debugVerboseLevel > 3, "m_segmenter->update");

	/// Perform visual tracker model update in case the visual tracker module is enabled.
	MEAS_TIME( m_visualTracker->update(m_inputMeta), m_paramsBase.m_debugVerboseLevel > 3, "m_visualTracker->update" );

	/// Update detected reid features to trackers' reid feature gallery.
	MEAS_TIME( m_reid->update(), m_paramsBase.m_debugVerboseLevel > 3, "m_reid->update" );

	/// Fill out the output data.
	setOutputData(pTrackedObjectsBatch);

	/// Complete processing for the frame batch.
	wrapUpProcessingFrameBatch();
}

/** @brief Advance batch state and emit optional debug output. */
void NvMultiObjectTrackerBase::wrapUpProcessingFrameBatch()
{
	m_frameInfo.frameBatchNo++;

	if(m_paramsBase.m_debugVerboseLevel > 1)
	{
		if(m_frameInfo.frameBatchNo%100 == 0)
		{
			cout << std::dec << "FrameBatch[" << m_frameInfo.frameBatchNo
			<< "] lastTrackerInd = " << m_targets->getLastTargetInd()
			<< ", when maxTargetsToTrack = " << m_frameInfo.maxStreams*m_targets->getMaxTargetsPerStream()
			<< " --> (Target capacity utilization = " << m_targets->getTargetCapacityUtilization() << " [%])"
			<< endl;
		}
	}
}

/**
 * @brief Fill tracked-object output metadata for the current frame batch.
 * @param[in,out] pTrackedObjectsBatch Caller-allocated batch filled with tracked objects for this frame batch.
 */
void NvMultiObjectTrackerBase::setOutputData(NvMOTTrackedObjBatch *&pTrackedObjectsBatch)
{
	/// Reset the number of filled stream lists.
	pTrackedObjectsBatch->numFilled = 0;

	/// Map from video stream id to that stream's object list.
	std::map<NvMOTStreamId, NvMOTTrackedObjList *> outputObjListMapper;

	/// Generate outputs only for streams that have a source frame in this batch.
	for(auto streamIt = m_inputMeta.begin(); streamIt != m_inputMeta.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamID = streamIt->first;
		NvMOTTrackedObjList *objList = nullptr;

		const NvTrackedObjectList trackerList = m_targets->getTrackedObjectsInUseList(videoStreamID);

		if(trackerList.size() == 0)
			continue;

		/// Retrieve the per-stream object list.
		if(outputObjListMapper.find(videoStreamID) != outputObjListMapper.end())
		{
			objList = outputObjListMapper[videoStreamID];
		}
		else if(pTrackedObjectsBatch->numFilled < pTrackedObjectsBatch->numAllocated)
		{
			outputObjListMapper[videoStreamID] = &pTrackedObjectsBatch->list[pTrackedObjectsBatch->numFilled++];
			objList = outputObjListMapper[videoStreamID];
			objList->streamID = videoStreamID;
			objList->numFilled = 0;
			objList->frameNum = streamIt->second->frameNum;
			objList->valid = true;
		}
		else
		{
			continue;
		}

		for(auto tIt : trackerList)
		{
			const std::shared_ptr<NvTrackedObject> target = tIt.second;
			const TargetInfo tInfo = target->getTargetInfo();
			const TRACKER_STATE trackerState = target->getTrackerState();

			if (tInfo.visibility < m_targets->getMinVisibility4TrajDump())
			{
				continue;
			}

			if(trackerState == TRACKER_STATE::ACTIVE)
			{
				NvMOTTrackedObj *tObj = &objList->list[objList->numFilled++];

				tObj->trackingId = target->getTargetID();

				tObj->associatedObjectIn = tInfo.getMatchedDetObjIn();

				tObj->classId = tInfo.classId;

				/// Provide the tracker bbox to the downstream.
				if(!m_targets->getDumpDetectorBboxes() || tObj->associatedObjectIn == nullptr)
				{
					tObj->bbox.x = tInfo.tBbox.min()(0);
					tObj->bbox.y = tInfo.tBbox.min()(1);
					tObj->bbox.width = tInfo.tBbox.sizes()(0);
					tObj->bbox.height = tInfo.tBbox.sizes()(1);
				}
				else
				{
					/// TODO: Apply detector-box selection consistently to current, past-frame, and trajectory output.
					const NvMOTObjToTrack *objIn = tObj->associatedObjectIn;

					tObj->bbox.x = objIn->bbox.x;
					tObj->bbox.y = objIn->bbox.y;
					tObj->bbox.width = objIn->bbox.width;
					tObj->bbox.height = objIn->bbox.height;
				}

				tObj->confidence = tInfo.confidence;
				tObj->age = tInfo.age;

				/// Output segmentation mask and ReID features.
				m_segmenter->outputMaskObj(target, tObj->mask);
				m_reid->outputTargetReIDObj(
					videoStreamID, target->getTargetInd(), *pTrackedObjectsBatch->pReidTensorBatch, tObj->reid
				);

				tObj->visibility = std::min(tInfo.visibility, tInfo.imgBorderCroppingVisibility);
				/// Generated by object model projector.
				if(target->isObjModelProjectorInitialized())
				{
					tObj->ptImgFeet[0] = tInfo.pt_img_feet(0);
					tObj->ptImgFeet[1] = tInfo.pt_img_feet(1);
					tObj->ptWorldFeet[0] = tInfo.pt_world_feet(0);
					tObj->ptWorldFeet[1] = tInfo.pt_world_feet(1);
					tObj->convexHull.numPoints = 0;
					for (uint ci=0; ci<tObj->convexHull.numPointsAllocated && ci<tInfo.convexHull.size(); ci++)
					{
						tObj->convexHull.list[2*ci] = (int) (tInfo.convexHull.at(ci)(0) + 0.5);
						tObj->convexHull.list[2*ci + 1] = (int) (tInfo.convexHull.at(ci)(1) + 0.5);
						tObj->convexHull.numPoints++;
					}

					target->setBbox3D(tInfo.pt_world_feet.cast<double>(), tObj->bbox3DWorld, tObj->bbox3DImg);
				}
			}
			else if(trackerState == TRACKER_STATE::INACTIVE || trackerState == TRACKER_STATE::TENTATIVE)
			{
				target->addPastFrameData();
			}
		}
	}

	m_reid->outputReIDTensor(*pTrackedObjectsBatch->pReidTensorBatch);
}

/**
 * @brief Output past-frame tracking data and clear the corresponding buffers.
 * @param[in] videoStreamIdList Stream IDs whose past-frame tracking data is reported.
 * @param[in,out] pPastFrameObjBatch Caller-allocated batch that receives the past-frame objects.
 */
void NvMultiObjectTrackerBase::outputPastFrameObjs(
	const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pPastFrameObjBatch
)
{
	/// Process each stream.
	for(auto it = videoStreamIdList.begin(); it != videoStreamIdList.end(); ++it)
	{
		const NvMOTStreamId surfaceStreamID = *it;
		const NvTrackedObjectList &trackerList = m_targets->getTrackedObjectsInUseList(surfaceStreamID);

		if(pPastFrameObjBatch->numFilled < pPastFrameObjBatch->numAllocated)
		{
			NvDsTargetMiscDataStream &pastFrameObjStream = pPastFrameObjBatch->list[pPastFrameObjBatch->numFilled++];

			pastFrameObjStream.numFilled = 0;
			pastFrameObjStream.surfaceStreamID = uint64_t(surfaceStreamID);

			/// If fail to allocate data for pastFrameObjStream.list, throw error.
			if(pastFrameObjStream.list == nullptr)
			{
				throw std::runtime_error(
					"[NvMultiObjectTrackerBase::outputPastFrameObjs] Make sure pastFrameObjStream.list != NULL"
				);
			}

			if(trackerList.size() == 0)
				continue;

			/// For each tracker in a particular stream.
			for(auto jt = trackerList.begin(); jt != trackerList.end(); ++jt)
			{
				std::shared_ptr<NvTrackedObject> tracker = jt->second;
				const TRACKER_STATE trackerState = tracker->getTrackerState();
				const std::deque<TargetInfo> &targetBuffer = tracker->getPastFrameData();

				if(trackerState == TRACKER_STATE::ACTIVE && targetBuffer.size() > 0)
				{
					if(pastFrameObjStream.numFilled < pastFrameObjStream.numAllocated)
					{
						NvDsTargetMiscDataObject &pastFrameObjList = pastFrameObjStream.list[pastFrameObjStream.numFilled++];

						pastFrameObjList.numObj = int(targetBuffer.size());
						pastFrameObjList.classId = targetBuffer[0].classId;
						pastFrameObjList.uniqueId = tracker->getTargetID();

						if(pastFrameObjList.list == nullptr)
						{
							throw std::runtime_error(
								"[NvMultiObjectTrackerBase::outputPastFrameObjs] Make sure pastFrameObjStream.list != NULL"
							);
						}

						for (uint bi=0; bi<pastFrameObjList.numObj; bi++)
						{
							NvDsTargetMiscDataFrame &bufferedObj = pastFrameObjList.list[bi];
							bufferedObj.frameNum = targetBuffer[bi].frameNum;

							bufferedObj.tBbox.height = targetBuffer[bi].tBbox.sizes()(1);
							bufferedObj.tBbox.width = targetBuffer[bi].tBbox.sizes()(0);
							bufferedObj.tBbox.left = targetBuffer[bi].tBbox.min()(0);
							bufferedObj.tBbox.top = targetBuffer[bi].tBbox.min()(1);
							bufferedObj.confidence = targetBuffer[bi].confidence;
							bufferedObj.age = targetBuffer[bi].age;
							bufferedObj.trackerState = targetBuffer[bi].trackerState;
							bufferedObj.visibility = targetBuffer[bi].visibility;
							bufferedObj.ptWorldFeet[0] = targetBuffer[bi].pt_world_feet(0);
							bufferedObj.ptWorldFeet[1] = targetBuffer[bi].pt_world_feet(1);
						}
					}
					else
					{
						if(m_paramsBase.m_debugVerboseLevel > 1)
							CLOG(std::string( "!!![WARNING] numFilledLists >= numAllocatedLists" ).c_str());
						break;
					}

					tracker->resetPastFrameData();
				}
			}
		}
	}
}

/**
 * @brief Output complete frame history for all terminated tracks to the plugin.
 * @param[in] videoStreamIdList Stream IDs whose terminated tracks are reported.
 * @param[in,out] pTerminatedTrackoutput Caller-allocated batch that receives the terminated-track history.
 */
void NvMultiObjectTrackerBase::outputTerminatedObjs(
	const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pTerminatedTrackoutput
)
{
	float widthScale = float(m_frameInfo.originalSrcImageSize.width) / m_frameInfo.srcImageSize.width;
	float heightScale = float(m_frameInfo.originalSrcImageSize.height) / m_frameInfo.srcImageSize.height;

	/// Process each stream.
	for(auto it = videoStreamIdList.begin(); it != videoStreamIdList.end(); ++it)
	{
		const NvMOTStreamId surfaceStreamID = *it;

		if(pTerminatedTrackoutput->numFilled < pTerminatedTrackoutput->numAllocated)
		{
			NvDsTargetMiscDataStream &miscDataStream = pTerminatedTrackoutput->list[pTerminatedTrackoutput->numFilled++];

			miscDataStream.numFilled = 0;
			miscDataStream.surfaceStreamID = uint64_t(surfaceStreamID);

			/// If fail to allocate data for miscDataStream.list, throw error.
			if(miscDataStream.list == nullptr)
			{
				throw std::runtime_error(
					"[NvMultiObjectTrackerBase::outputTerminatedObjs] Make sure miscDataStream.list != NULL"
				);
			}

			std::list<TerminatedTrack> * pTermTrackData;
			/// Get list of terminated objects from manager.
			if (! m_targets->getOutputTerminatedTracks(surfaceStreamID, pTermTrackData))
			{
				/// No data for this stream.
				continue;
			}

			/// For each tracker in a particular stream.
			for(auto curTrack = pTermTrackData->begin(); curTrack != pTermTrackData->end(); ++curTrack)
			{
				/// Check if this track is part of the current stream.
				if(curTrack->videoStreamId != miscDataStream.surfaceStreamID)
				{
					continue;
				}

				if(miscDataStream.numFilled < miscDataStream.numAllocated)
				{
					NvDsTargetMiscDataObject &terminatedObjList = miscDataStream.list[miscDataStream.numFilled++];

					terminatedObjList.classId = curTrack->classID;
					terminatedObjList.uniqueId = curTrack->targetID;
					terminatedObjList.numObj = curTrack->targetHistory.size();

					/// Reject frame history that exceeds the allocation derived from the maximum trajectory length.
					if(curTrack->targetHistory.size() > terminatedObjList.numAllocated )
					{
						throw std::runtime_error(
							"[NvMultiObjectTrackerBase::outputTerminatedObjs] pastFrameHist > numAllocated"
						);
					}

					if(terminatedObjList.list == nullptr)
					{
						throw std::runtime_error(
							"[NvMultiObjectTrackerBase::outputTerminatedObjs] Make sure miscDataStream.list != NULL"
						);
					}

					for (uint bi=0; bi<terminatedObjList.numObj; bi++)
					{
						NvDsTargetMiscDataFrame &bufferedObj = terminatedObjList.list[bi];
						bufferedObj.frameNum = curTrack->targetHistory[bi].frameNum;
						bufferedObj.tBbox.height = curTrack->targetHistory[bi].tBbox.sizes()(1) * heightScale;
						bufferedObj.tBbox.width = curTrack->targetHistory[bi].tBbox.sizes()(0) * widthScale;
						bufferedObj.tBbox.left = curTrack->targetHistory[bi].tBbox.min()(0) * widthScale;
						bufferedObj.tBbox.top = curTrack->targetHistory[bi].tBbox.min()(1) * heightScale;
						bufferedObj.confidence = curTrack->targetHistory[bi].confidence;
						bufferedObj.age = curTrack->targetHistory[bi].age;
						bufferedObj.trackerState = curTrack->state;
						bufferedObj.visibility = curTrack->targetHistory[bi].visibility;
						bufferedObj.ptWorldFeet[0] = curTrack->targetHistory[bi].pt_world_feet(0);
						bufferedObj.ptWorldFeet[1] = curTrack->targetHistory[bi].pt_world_feet(1);
					}
				}
				else
				{
					if(m_paramsBase.m_debugVerboseLevel > 1)
						CLOG(std::string( "!!![WARNING] Target numFilledLists >= numAllocatedLists" ).c_str());
					break;
				}
			}
		}
	}

	m_targets->dumpTerminatedTracks();
}
/**
 * @brief Output the most recent frame for current shadow tracks to the plugin.
 * @param[in] videoStreamIdList Stream IDs whose shadow tracks are reported.
 * @param[in,out] pShadowTracksBatch Caller-allocated batch that receives the latest shadow-track frame.
 */
void NvMultiObjectTrackerBase::outputShadowObj(
	const std::set<NvMOTStreamId> videoStreamIdList, NvDsTargetMiscDataBatch *&pShadowTracksBatch
)
{
	float widthScale = float(m_frameInfo.originalSrcImageSize.width) / m_frameInfo.srcImageSize.width;
	float heightScale = float(m_frameInfo.originalSrcImageSize.height) / m_frameInfo.srcImageSize.height;

	/// Process each stream.
	for(auto it = videoStreamIdList.begin(); it != videoStreamIdList.end(); ++it)
	{
		const NvMOTStreamId surfaceStreamID = *it;

		if(pShadowTracksBatch->numFilled < pShadowTracksBatch->numAllocated)
		{
			NvDsTargetMiscDataStream &miscDataStream = pShadowTracksBatch->list[pShadowTracksBatch->numFilled++];

			miscDataStream.numFilled = 0;
			miscDataStream.surfaceStreamID = uint64_t(surfaceStreamID);

			/// If fail to allocate data for miscDataStream.list, throw error.
			if(miscDataStream.list == nullptr)
			{
				throw std::runtime_error(
					"[NvMultiObjectTrackerBase::outputShadowObj] Make sure miscDataStream.list != NULL"
				);
			}

			/// Get list of shadow objects from manager.
			const std::set<std::shared_ptr<NvTrackedObject>> & shadowTracks = m_targets->getShadowTracks(
				surfaceStreamID
			);

			/// For each tracker in a particular stream.
			for( auto curTrackIter = shadowTracks.begin(); curTrackIter != shadowTracks.end(); ++curTrackIter)
			{
				std::shared_ptr<NvTrackedObject> curTrack = *curTrackIter;
				/// Check if this track is part of the current stream.
				if( (curTrack->getVideoStreamId() != miscDataStream.surfaceStreamID) || (curTrack->getTrackerState() != TRACKER_STATE::INACTIVE) )
				{
					continue;
				}

				if(miscDataStream.numFilled < miscDataStream.numAllocated)
				{
					NvDsTargetMiscDataObject &shadowObjList = miscDataStream.list[miscDataStream.numFilled++];

					shadowObjList.numObj = 1;
					shadowObjList.classId = curTrack->getClassId();
					shadowObjList.uniqueId = curTrack->getTargetID();

					if(shadowObjList.list == nullptr)
					{
						throw std::runtime_error(
							"[NvMultiObjectTrackerBase::outputShadowObj] Make sure miscDataStream.list != NULL"
						);
					}

					/// Only output most recent frame.
					NvDsTargetMiscDataFrame &bufferedObj = shadowObjList.list[0];
					const TargetInfo & lastFrameInfo = curTrack->getTargetInfo();
					bufferedObj.frameNum = lastFrameInfo.frameNum;
					bufferedObj.tBbox.height = lastFrameInfo.tBbox.sizes()(1) * heightScale;
					bufferedObj.tBbox.width = lastFrameInfo.tBbox.sizes()(0) * widthScale;
					bufferedObj.tBbox.left = lastFrameInfo.tBbox.min()(0) * widthScale;
					bufferedObj.tBbox.top = lastFrameInfo.tBbox.min()(1) * heightScale;
					bufferedObj.confidence = lastFrameInfo.confidence;
					bufferedObj.age = lastFrameInfo.age;
					bufferedObj.trackerState = curTrack->getTrackerState();
					bufferedObj.visibility = lastFrameInfo.visibility;
					bufferedObj.ptWorldFeet[0] = lastFrameInfo.pt_world_feet(0);
					bufferedObj.ptWorldFeet[1] = lastFrameInfo.pt_world_feet(1);
				}
				else
				{
					if(m_paramsBase.m_debugVerboseLevel > 1)
						CLOG(std::string( "!!![WARNING] numFilledLists >= numAllocatedLists" ).c_str());
					break;
				}
			}
		}
	}
}

/**
 * @brief Delete trackers and per-stream resources when a stream is removed.
 * @param[in] videoStreamID Stream ID whose trackers are removed.
 */
void NvMultiObjectTrackerBase::deleteRemovedStreamTrackers(const NvMOTStreamId videoStreamID)
{
	m_targets->removeStream(videoStreamID);
	m_reid->removeStream(videoStreamID);
}

/**
 * @brief Apply supported runtime tracker configuration updates.
 * @param[in] configStr Stringified YAML configuration parameters to update.
 */
void NvMultiObjectTrackerBase::updateParams(const std::string &configStr )
{
	/// Loads the stringified version of the config into a Node structure.
	YAML::Node fn = YAML::Load(configStr);

	/// Control Section unique to the update section.
	if (fn["Control"] && fn["Control"].IsMap())
	{
		auto controlSection = fn["Control"];
		for (YAML::const_iterator fit = controlSection.begin(); fit != controlSection.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();

			if (paramKey == "tracker-reset")
			{
				int resetType = (int)fit->second.as<float>();
				if( resetType != 0)
				{
					/// Remove all active tracks from every stream.
					for(auto it = m_targets->getTrackedObjectsInUseList().begin(); it != m_targets->getTrackedObjectsInUseList().end(); ++it)
					{
						NvMOTStreamId streamId = it->first;
						m_targets->resetStream(streamId);
						m_reid->removeStream(streamId);
					}
				}
			}
		}
	}

	/// Apply BaseConfig updates.
	if (fn["BaseConfig"] && fn["BaseConfig"].IsMap())
	{
		m_paramsBase.read(fn["BaseConfig"]);
	}

	// [Target and trajectory management]
	if ((fn["TargetManagement"] && fn["TargetManagement"].IsMap()) ||
		(fn["TrajectoryManagement"] && fn["TrajectoryManagement"].IsMap())
	   )
	{
		m_targets->updateParams(fn);
	}

	// [Data associator]
	if (fn["DataAssociator"] && fn["DataAssociator"].IsMap())
	{
		m_dataAssociator->updateParams(fn["DataAssociator"]);
	}

	// [Visual tracker]
	if (fn["VisualTracker"] && fn["VisualTracker"].IsMap())
	{
		m_visualTracker->updateParams(fn["VisualTracker"]);
	}
}

/** @brief Initialize the base tracker parameters with default values. */
NvTrackerParamsBase::NvTrackerParamsBase()
{
	/// Default values for config params.
	m_debugVerboseLevel = 0;
	m_checkClassMatch = true;
	m_minDetectorConfidence = -1.f;
	m_useBatchNumForFrameId = false;
}

/**
 * @brief Load the base tracker parameters from a configuration module.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Name of the configuration module to load.
 */
NvTrackerParamsBase::NvTrackerParamsBase(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath, true);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Load the base tracker parameters from a YAML node.
 * @param[in] fn YAML node containing the base tracker configuration.
 */
NvTrackerParamsBase::NvTrackerParamsBase(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Read base tracker parameters from a YAML node.
 * @param[in] fn YAML node containing the base tracker configuration.
 */
void NvTrackerParamsBase::read(const YAML::Node &fn)
{
	if(!m_isInit)
	{
		/// Set the default values.
		*this = NvTrackerParamsBase();
	}

	if(!fn.size())
	{
		m_isInit = true;
		return;
	}

	for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
	{
		std::string paramKey = fit->first.as<std::string>();

		if (paramKey == "debugVerboseLevel")
		{
			m_debugVerboseLevel = fit->second.as<float>();
		}
		else if (paramKey == "checkClassMatch")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_checkClassMatch = (bool)fit->second.as<int>();
			else
				CLOG(std::string("!!![WARNING] BaseConfig:checkClassMatch is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "minDetectorConfidence")
		{
			m_minDetectorConfidence = fit->second.as<float>();
		}
		else if (paramKey == "useBatchNumForFrameId")
		{
			m_useBatchNumForFrameId = (bool)fit->second.as<int>();
		}
		else
		{
			cout << "!! [WARNING][NvTrackerParamsBase] Unknown param found: " << paramKey << endl;
		}
	}

	/// Set init flag after initialization to prevent updates from changing.
	m_isInit = true;
}
