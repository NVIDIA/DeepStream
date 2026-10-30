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

#ifndef NVTRACKEDOBJECT_HPP_
#define NVTRACKEDOBJECT_HPP_

#include <cstdint>
#include <deque>
#include <list>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <iomanip>

#include <atomic>
#include <limits>
#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
#endif
#include <random>

#include "modules/utility/dataTypes.hpp"
#include "modules/utility/cv_functions.hpp"
#include "modules/Estimatorv2/estimator_factory2.hpp"
#include "modules/NvMultiObjectTracker/localizer.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/TrajectoryManager/NvTrajectoryManager.hpp"
#include "modules/ObjectModelProjector/ObjectModelProjector.hpp"
#include "modules/MultiViewAssociator/MultiViewAssociator_factory.hpp"

/** @brief The types of target termination mode. */
enum TRACKER_DELETE_MODE : uint {
	REGULAR_TERMINATION = 1, //!< Target reached a regular termination condition.
	EARLY_TERMINATION, //!< TENTATIVE target reached an early termination condition.
	INSUFFICIENT_VISIBILITY, //!< Target visibility fell below the required threshold.
	DUPLICATE_TRACKER, //!< Target duplicated another tracker.
	CORRUPTED_TRACKER, //!< Target state became invalid.
	TOO_SMALL_OBJECT, //!< Target bounding box was too small.
	TOO_LARGE_OBJECT, //!< Target bounding box was too large.
	REMOVED_STREAM_TRACKER, //!< Target belonged to a removed stream.
	FUSED_BY_REASSOCIATION //!< Target was fused into another track through reassociation.
};

/** @brief Identifies the ReID processing stage for reassociation. */
enum REID4REASSOC_STATE : int {
	NOT_NEEDED, //!< ReID is not needed for reassociation.
	TO_DO, //!< ReID is needed for reassociation but has not completed.
	DONE //!< ReID has completed for reassociation.
};

/** @brief Stores a terminated target and its frame history. */
struct TerminatedTrack
{
	/** @brief Initialize an empty terminated-track record. */
	TerminatedTrack () : targetID(0), classID(0), videoStreamId(0), state(TRACKER_STATE::EMPTY) {}
	TargetId targetID; //!< Unique target ID of the terminated track.
	uint16_t classID; //!< Class ID of the terminated track.
	NvMOTStreamId videoStreamId; //!< Video stream ID associated with the track.
	TRACKER_STATE state; //!< Tracker state of the track.
	std::vector<TargetInfo> targetHistory; //!< Target information for each frame of the track.
};

/** @brief Describes batched segmentation-mask storage. */
struct SegmentationInBatch
{
	float *segMaskInBatch_d; //!< CUDA array containing segmentation results.
	NvMotDataTypes::Size size; //!< Size of each segmentation mask.
};

/** @brief Input for the data association module. */
struct assocDataIn_t
{
	const NvMOTStreamId videoStreamId; //!< Video stream ID.
	const NvMOTFrameNum frameNum; //!< Frame number of the video stream.
	const NvMOTTimeMeta timeMeta; //!< Frame, stream, and local timestamps for this association pass.
	bool isDetDone; //!< Indicates whether detection is complete for this video stream.
	std::deque<const DetectedObjInfo *> detObjs; //!< Detected objects to associate with targets.
	std::map<TrackerIndex, std::shared_ptr<NvTrackedObject>> targets; //!< Targets to associate, keyed by tracker index.
	std::map<uint16_t, float> sumsCorrRespAtDetectedPt; //!< Sum of tracker correlation responses at each detected object for cascade stage zero.
	std::shared_ptr<IMultiViewAssociator> multiViewAssociator; //!< Multi-view associator.
};

/** @brief Output from the data association module. */
struct assocDataOut_t
{
	NvMOTStreamId videoStreamId; //!< Video stream ID.
	NvMOTFrameNum frameNum; //!< Frame number of the video stream.
	NvMOTTimeMeta timeMeta; //!< Frame, stream, and local timestamps received with the association input.
	bool isAssocDone; //!< Indicates whether association is complete for this video stream.
	std::map<const TrackerIndex, const DetectedObjInfo*> unmatchedDetObjs; //!< Detected objects without a matched target.
	std::deque<std::pair<const DetectedObjInfo *, TrackerIndex>> matchedPairs; //!< Matched pairs of detected object and target index.
	std::set<TrackerIndex> unmatchedTargets; //!< Target indexes without a matched detected object.
};

/** @brief Output from ReID service association module. */
struct reidServiceAssocInfo
{
	TargetId targetID; //!< Best matched target ID from reid service.
	int groupCount = 0; //!< Number of vectors from the cluster that are present in group search results.
	float confidence = 0.0; //!< Match confidence computed as one divided by one plus distance.
	float expConfidence = 0.0; //!< Match confidence computed as an exponential of distance.
	float distance; //!< L2 squared average distance of the search vector to the "groupCount" vectors in the cluster.
	float distanceToCentroid; //!< L2 squared distance to centroid of the cluster.
	float clusterVariance; //!< Variance of the cluster.
};

#ifdef VIZ_DEBUG
/** @brief Provides visualization access to tracked-object internals. */
class TrackerVizOutput;
#endif

/** @brief Manages target trajectories and reassociation. */
class NvTrajectoryManager;

/** @brief Stores target-manager configuration parameters. */
class NvTrackerParamsTargetManager : public NvTrackerParams
{
public:
	/** @brief Initialize target-manager parameters with default values. */
	NvTrackerParamsTargetManager();

	/**
	 * @brief Load target-manager parameters from a configuration module.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Name of the configuration module to load.
	 */
	NvTrackerParamsTargetManager(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Load target-manager parameters from a YAML node.
	 * @param[in] fn YAML node containing the target-manager configuration.
	 */
	NvTrackerParamsTargetManager(const YAML::Node &fn);

	/**
	 * @brief Read target-manager parameters from a YAML node.
	 * @param[in] fn YAML node containing the target-manager configuration.
	 */
	void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Debug message verbose mode level.
	bool m_preserveStreamUpdateOrder; //!< Preserves input stream order when assigning target IDs for deterministic results.
	bool m_enableBboxUnClipping; //!< Enables recovery of bounding boxes clipped by the image border.
	bool m_outputVisibility; //!< Output visibility flag.
	bool m_saveTerminatedTracks; //!< Enables saving terminated tracks for user reporting.
	bool m_useVOTBasedTargetUpdate; //!< Enables VOT-based target updates.
	std::string m_terminatedTrackFilename; //!< Terminated tracker file name.
	std::string m_intermediateInfoFileName; //!< Base filename for intermediate detector and target dumps.
	int m_maxTargetsPerStream; //!< The max # of targets to track per stream.

	// [Class match]
	bool m_checkClassMatch; //!< If true, only objects of the same class are associated (sourced from BaseConfig:checkClassMatch).

	// [Creation & Termination Policy]
	float m_minDetectorConfidence; //!< If the confidence of a detected object is lower than this, it will not be considered as a new target.
	float m_minTrackerConfidence; //!< Localization confidence threshold for the peak correlation response.
	int m_probationAge; //!< If the target's age exceeds this, the target will be considered to be valid.
	int m_maxShadowTrackingAge; //!< Max length of shadow tracking. If the shadowTrackingAge exceeds this limit, the tracker will be terminated.
	int m_maxActiveShadowTrackingAge; //!< Override the shadow cap for tracks that were ever ACTIVE.
	//!< When greater than m_maxShadowTrackingAge, ACTIVE-lineage tracks survive longer.
	int m_earlyTerminationAge; //!< If the shadowTrackingAge reaches this threshold while in TENTATIVE period, the target will be terminated prematurely.
	//!< If this is zero/negative or larger than m_probationAge, then no early termination would be performed.
	float m_minIouDiff4NewTarget; //!< If the IOU between the newly detected object and any of the existing targets is higher than this threshold, this newly detected object will be discarded.
	float m_searchRegionPaddingScale; //!< Search-region padding scale relative to the target bounding box.
	//!< A value of three produces a region four times the target bounding-box size in each dimension.

	// [Trajectory Management]
	int m_maxTrajectoryBufferLength; //!< Maximum retained trajectory length before subsampling.
	bool m_saveShadowTracks; //!< Output shadow track objects.

	// [Debug parameters]
	int m_debugParamInt0;
	int m_debugParamInt1;
	int m_debugParamInt2;
	float m_debugParamFloat0;
	float m_debugParamFloat1;
	float m_debugParamFloat2;

	// [Runtime parameters]
	uint m_maxTargetsToTrack; //!< Maximum number of targets across all streams.
	NvMotDataTypes::Rect2f originalSrcImageBox; //!< Original video source image box.
	NvMotDataTypes::Rect2f srcImageBox; //!< Tracker source image box.
	bool m_isInit = false; //!< Indicates whether the parameters have been initialized.
	//!< Prevents changing static params in dynamic updates.
};

/** @brief Stores and updates the state of one tracked target. */
class NvTrackedObject
{
public:
	/**
	 * @brief Initialize a tracked target and its estimator.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] params Parsed target-manager parameters.
	 * @param[in] trackerInd Index of this target in the allocated memory layout.
	 */
	NvTrackedObject(
		const std::string &configFilePath, const NvTrackerParamsTargetManager *params, const TrackerIndex trackerInd
	);

	/** @brief Release resources owned by the tracked target. */
	virtual ~NvTrackedObject();

	/** @brief Reset the target states for later use. */
	void reset();

	/**
	 * @brief Update the states with the new measurement.
	 * @param[in] frameNum Current frame number.
	 * @param[in] debObj Detected object from the input metadata.
	 * @param[in] state New tracker state.
	 */
	void updateTargetStates(const NvMOTFrameNum frameNum, const DetectedObjInfo *debObj, TRACKER_STATE state);

	/**
	 * @brief Update the states with its own prediction because of no-inference frame.
	 * @param[in] frameNum Frame number of the prediction update.
	 */
	void updateTargetStates(const NvMOTFrameNum frameNum);

	/**
	 * @brief Initialize a target tracked by a peer camera in quasi-active state.
	 * @param[in] frameNum Current frame number.
	 * @param[in] peerTarget Peer-camera target used to initialize target information.
	 */
	void initializeQuasiActiveStates(const NvMOTFrameNum frameNum, const PeerTarget *peerTarget);

	/**
	 * @brief Update localization states from the current tracker result.
	 * @param[in] frameNum Current frame number.
	 * @param[in] trackingConfidence Current tracking confidence.
	 * @param[in] localizationBbox Localized bounding box.
	 */
	void setLocalizationResults(
		const NvMOTFrameNum frameNum, const float trackingConfidence, const NvMotDataTypes::Rect2f& localizationBbox
	);

	/**
	 * @brief Increment the number of frames spent in shadow tracking.
	 * @param[in] increment Number of frames to add to the shadow-tracking age.
	 */
	void incrementShadowTrackingAge(const NvMOTFrameNum increment = 1) { m_targetInfo.shadowTrackingAge += increment; }

	/** @brief Reset the shadow-tracking age to zero. */
	void resetShadowTrackingAge() { m_targetInfo.shadowTrackingAge = 0; }

	/** @brief Add the target info for the current frame into a buffer for later reporting. */
	void addPastFrameData();

	/** @brief Clear the past-frame data buffer. */
	void resetPastFrameData() { return m_pastFrameDataBuffer.clear(); }

	/**
	 * @brief Expand a target bounding box into its search region.
	 * @param[in] bbox Target bounding box to expand.
	 */
	const NvMotDataTypes::Rect2f defineSearchRegion(const NvMotDataTypes::Rect2f bbox) const;

	/** @brief Add the current target info to the trajectory buffer. */
	void updateTrajectoryBuffer();

	/**
	 * @brief Purge the trajectory buffer until the frame num specified.
	 * @param[in] frameNum Newest frame number that is kept; later points are removed.
	 */
	void purgeTrajectoryBufferUntilSpecified(const NvMOTFrameNum frameNum);

	/**
	 * @brief In case of re-assoc with a newer tracklet, adopt and fuse it to the existing one.
	 * @param[in] tracklet Newer tracklet fused into this target's trajectory.
	 */
	void adoptTracklet(const std::list<TargetInfo> &tracklet);

	/**
	 * @brief Fuse the peer's tracklet into the target's.
	 * @param[in] tracklet This target's tracklet.
	 * @param[in] peerTracklet Peer tracklet fused with the target tracklet.
	 * @return Fused tracklet, or the target tracklet when either input is empty.
	 */
	std::list<TargetInfo> fuseOverlappingTracklets(
		const std::list<TargetInfo> &tracklet, const std::list<TargetInfo> &peerTracklet
	);

	/**
	 * @brief In case of re-assoc with a newer tracklet, fuse its past frame object into the existing one.
	 * @param[in] pastFrameBuffer Past-frame objects appended to this target's buffer.
	 */
	void adoptPastFrameData(const std::deque<TargetInfo> &pastFrameBuffer);

	/**
	 * @brief Package this target for ReID extraction when eligible.
	 * @param[in] inputMeta Input frames keyed by video stream ID.
	 */
	std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> generateTrackedObjects4Reid(
		const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta
	);

	/**
	 * @brief Set the video stream associated with this target.
	 * @param[in] videoStreamId Video stream ID associated with the target.
	 */
	void setVideoStreamId(const NvMOTStreamId videoStreamId) { m_targetInfo.videoStreamId = videoStreamId; }

	/** @brief Return the video stream associated with this target. */
	const NvMOTStreamId getVideoStreamId() { return m_targetInfo.videoStreamId; }

	/** @brief Return the target index in the allocated memory layout. */
	const TrackerIndex &getTargetInd() { return m_targetInd; }

	/** @brief Return the current target information. */
	const TargetInfo &getTargetInfo() { return m_targetInfo; }

	/**
	 * @brief Replace the current target information.
	 * @param[in] tInfo Target information to store.
	 */
	void setTargetInfo(const TargetInfo &tInfo) { m_targetInfo = tInfo; }

	/** @brief Return the current target bounding box. */
	const NvMotDataTypes::Rect2f getTargetBbox() const { return m_targetInfo.tBbox; }

	/** @brief Get the search region for the current frame. */
	const NvMotDataTypes::Rect2f getSearchRegion() const { return m_searchRegion; }

	/** @brief Get the predicted search region for the next frame. */
	const NvMotDataTypes::Rect2f getPredictedSearchRegion() const { return m_predictedSearchRegion; }

	/** @brief Return the progressed bounding box generated by the segmenter. */
	const NvMotDataTypes::Rect2f getProgBbox() const { return m_progBbox; }

	/**
	 * @brief Store the progressed bounding box generated by the segmenter.
	 * @param[in] bbox Progressed target bounding box.
	 */
	void setProgBbox(const NvMotDataTypes::Rect2f& bbox) { m_progBbox = bbox; }

	/** @brief Return the target age in frames. */
	const NvMOTFrameNum getAge() { return m_targetInfo.age; }

	/** @brief Return the visible fraction of the target. */
	const float getVisibility() { return m_targetInfo.visibility; }

	/**
	 * @brief Set the target age in frames.
	 * @param[in] newAge New target age in frames.
	 */
	void setAge(const NvMOTFrameNum newAge) { m_targetInfo.age = newAge; }

	/** @brief Return the current tracker state. */
	const TRACKER_STATE getTrackerState() { return m_targetInfo.trackerState; }

	/**
	 * @brief Set the current tracker state.
	 * @param[in] state New tracker state.
	 */
	void setTrackerState(const TRACKER_STATE state) { m_targetInfo.trackerState = state; }

	/** @brief Return the number of frames spent in shadow tracking. */
	const NvMOTFrameNum getShadowTrackingAge() { return m_targetInfo.shadowTrackingAge; }

	/** @brief Return the unique target ID. */
	const TargetId getTargetID() { return m_targetID; }

	/**
	 * @brief Set the unique target ID.
	 * @param[in] newTargetID New unique target ID.
	 */
	void setTargetID(const TargetId newTargetID) { m_targetID = newTargetID; }

	/** @brief Return the target class ID. */
	const uint16_t getClassId() { return m_targetInfo.classId; }

	#ifdef VIZ_DEBUG
		/** @brief Return the visualization color assigned to the target. */
		const cv::Scalar getColor() { return m_color; }
	#endif

	/** @brief Return buffered past-frame target information. */
	const std::deque<TargetInfo> &getPastFrameData() { return m_pastFrameDataBuffer; }

	/** @brief Return the target trajectory buffer. */
	std::list<TargetInfo> &getTrajectoryBuffer() { return m_trajectoryBuffer; }

	/**
	 * @brief Replace the target trajectory buffer.
	 * @param[in] traj Trajectory points to store.
	 */
	void setTrajectoryBuffer(const std::list<TargetInfo> &traj) { m_trajectoryBuffer = traj; }

	/** @brief Return the bounding box predicted by the state estimator. */
	const NvMotDataTypes::Rect2f getPredictedTargetBbox()
	{
		return CV_FUNCS::convertXToBbox(m_stateEstimator->getPredictedStates(), m_stateEstimator->useAspectRatio());
	}

	/** @brief Return the timestamp at which the target ID was created. */
	const TimeStamp getTargetIDTimeStamp() { return m_targetIDTimeStamp; }

	/**
	 * @brief Set the timestamp at which the target ID was created.
	 * @param[in] time Target ID creation timestamp.
	 */
	void setTargetIDTimeStamp(const TimeStamp time) { m_targetIDTimeStamp = time; }

	/** @brief Report whether the target became ACTIVE in the current frame. */
	const bool becameActive() { return m_becameActive; }

	/**
	 * @brief Set whether the target became ACTIVE in the current frame.
	 * @param[in] flag New became-ACTIVE flag value.
	 */
	void setBecameActive(const bool flag) { m_becameActive = flag; }

	/** @brief Report whether the target acquired an ID for the first time. */
	const bool acquiredID() { return m_acquiredID; }

	/**
	 * @brief Set whether the target acquired an ID for the first time.
	 * @param[in] flag New acquired-ID flag value.
	 */
	void setAcquiredID(const bool flag) { m_acquiredID = flag; }

	/** @brief Return peer-camera targets associated with this target. */
	const std::vector<PeerTarget> getPeerTargets() { return m_peerTargets; }

	/**
	 * @brief Replace peer-camera targets associated with this target.
	 * @param[in] peerTargets Peer-camera targets to store.
	 */
	void setPeerTargets(const std::vector<PeerTarget> peerTargets) { m_peerTargets = peerTargets; }

	/** @brief Mark the target as ready for target ID assignment. */
	void setNeedTargetIDAssigned() { m_needToSetTargetID = true; }

	/** @brief Return the target distance from the camera. */
	float getDistFromCam() { return m_targetInfo.distFromCam; }

	/** @brief Return the current estimator state and covariance. */
	std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getStatesWithCov()
	{
		return std::move(m_stateEstimator->getStatesWithCov());
	}

	/**
	 * @brief Return the predicted estimator state and covariance.
	 * @param[in] dT Prediction time step.
	 */
	std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedStatesWithCov(const float dT = 1.f)
	{
		return std::move(m_stateEstimator->getPredictedStatesWithCov(dT));
	}

	/** @brief Return predicted measurements and their covariance. */
	std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedMeasurementsWithCov()
	{
		return std::move(m_stateEstimator->getPredictedMeasurementsWithCov());
	}

	/** @brief Return the current tracker confidence. */
	const float getTrackerConfidence() { return m_targetInfo.confidence; }

	/** @brief Return the matched detection index or negative one when unmatched. */
	const int getMatchedDetObjInd()
	{
		return m_targetInfo.matchedDetObjInfo ? int(m_targetInfo.matchedDetObjInfo->bboxInd) : -1;
	}

	/** @brief Return the detected object matched to the target. */
	const DetectedObjInfo* getMatchedDetObj() { return m_targetInfo.matchedDetObjInfo; }

	/**
	 * @brief Store confidence-weighted correlation responses at detected points.
	 * @param[in] ptr Pointer to the response values.
	 * @param[in] length Number of response values.
	 */
	void setConfidenceCorrRespAtDetectedPts(const float *ptr, const size_t length)
	{
		m_confidenceCorrRespAtDetectedPts = std::vector<float>(ptr, ptr+length);
	}

	/** @brief Return confidence-weighted correlation responses at detected points. */
	const std::vector<float> &getConfidenceCorrRespAtDetectedPts() { return m_confidenceCorrRespAtDetectedPts; }

	/**
	 * @brief Store correlation responses at detected points.
	 * @param[in] ptr Pointer to the response values.
	 * @param[in] length Number of response values.
	 */
	void setCorrRespAtDetectedPts(const float *ptr, const size_t length)
	{
		m_corrRespAtDetectedPts = std::vector<float>(ptr, ptr+length);
	}

	/** @brief Return correlation responses at detected points. */
	const std::vector<float> &getCorrRespAtDetectedPts() { return m_corrRespAtDetectedPts; }

	/**
	 * @brief Store ReID confidence values for detections.
	 * @param[in] ptr Pointer to the confidence values.
	 * @param[in] length Number of confidence values.
	 */
	void setConfidenceReidDets(const float *ptr, const size_t length)
	{
		m_confidenceReidDets = std::vector<float>(ptr, ptr+length);
	}

	/**
	 * @brief Move ReID confidence values into the target.
	 * @param[in] vec ReID confidence values to store.
	 */
	void setConfidenceReidDets(std::vector<float> vec) { m_confidenceReidDets = std::move(vec); }

	/** @brief Return ReID confidence values for detections. */
	const std::vector<float> &getConfidenceReidDets() { return m_confidenceReidDets; }

	/**
	 * @brief Append a model's ReID confidence values.
	 * @param[in] ptr Pointer to the confidence values.
	 * @param[in] length Number of confidence values.
	 */
	void addConfidenceListReidDets(const float *ptr, const size_t length)
	{
		m_confidenceListReidDets.emplace_back(ptr, ptr+length);
	}

	/** @brief Return ReID confidence values from all models. */
	std::vector<std::vector<float>> &getConfidenceListReidDets() { return m_confidenceListReidDets; }

	/**
	 * @brief Store frame numbers associated with ReID confidence values.
	 * @param[in] input Frame numbers to store.
	 */
	void setConfidenceReidDetsFrameNum(const std::vector<int> input) { m_confidenceReidDetsFrameNum = input; }

	/** @brief Return frame numbers associated with ReID confidence values. */
	const std::vector<int> &getConfidenceReidDetsFrameNum() { return m_confidenceReidDetsFrameNum; }

	/**
	 * @brief Store Mahalanobis distances from the target to detections.
	 * @param[in] disVec Mahalanobis distances to store.
	 */
	void setDisMahalanobisDets(const std::vector<float> disVec) { m_disMahalanobisDets = disVec; }

	/** @brief Return Mahalanobis distances from the target to detections. */
	const std::vector<float> &getDisMahalanobisDets() { return m_disMahalanobisDets; }

	/**
	 * @brief Store the target segmentation mask.
	 * @param[in] mask Segmentation mask in original streammux coordinates.
	 */
	void setSegmentationMask(const NvMOTMask mask) { m_maskOriginal = mask; }

	/** @brief Return the target segmentation mask. */
	const NvMOTMask &getSegmentationMask() { return m_maskOriginal; }

	/** @brief Reset the target state estimator. */
	void resetStateEstimator() { m_stateEstimator->reset(); }

	/** @brief Return the last frame number in the trajectory buffer. */
	const NvMOTFrameNum getlastFrameinTraj() { return m_trajectoryBuffer.rbegin()->frameNum; }

	/** @brief Return the target 2.5D pose keypoints. */
	const std::vector<float> &getPose() { return m_targetInfo.pose25d; }

	/**
	 * @brief Store target 2.5D pose keypoints.
	 * @param[in] ptr Pointer to the pose values.
	 * @param[in] length Number of pose values.
	 */
	void setPose25d(const float *ptr, const size_t length)
	{
		m_targetInfo.pose25d = std::vector<float>(ptr, ptr+length);
	}

	/**
	 * @brief Store target 3D pose keypoints.
	 * @param[in] ptr Pointer to the pose values.
	 * @param[in] length Number of pose values.
	 */
	void setPose3d(const float *ptr, const size_t length) { m_targetInfo.pose3d = std::vector<float>(ptr, ptr+length); }

	/**
	 * @brief Store a deep copy of the object model projector.
	 * @param[in] objModelProjector Object model projector to copy.
	 */
	void setObjModelProjector(std::shared_ptr<ObjectModelProjector> objModelProjector);

	/** @brief Return the target object model projector. */
	std::shared_ptr<ObjectModelProjector> getObjModelProjector() { return m_objModelProjector; }

	/**
	 * @brief Check whether the object model projector is initialized.
	 * @return True when the object model projector is initialized.
	 */
	bool isObjModelProjectorInitialized();

	/**
	 * @brief Set 3D bounding box information.
	 * @param[in] pt_world_feet World coordinates of the feet point.
	 * @param[out] bbox3DWorld 3D bounding box in world coordinates.
	 * @param[out] bbox3DImg 2D projection of the 3D bounding box corners, eight corners with x and y coordinates.
	 */
	void setBbox3D(const Eigen::Vector2d& pt_world_feet, NvDsObj3DBbox& bbox3DWorld, float (*bbox3DImg)[2]);

	/** @brief Report whether the target is ready for target ID assignment. */
	const bool needTargetIDAssigned() { return m_needToSetTargetID; }

	/** @brief Clear the target ID assignment request. */
	void resetNeedTargetIDAssigned() { m_needToSetTargetID = false; }

	/** @brief Report whether the target is ready for reassociation. */
	const bool ready4ReAssoc() { return m_ready4ReAssoc; }

	/** @brief Clear the reassociation-ready flag. */
	void resetReady4ReAssoc() { m_ready4ReAssoc = false; }

	/** @brief Report whether the target must leave the pending reassociation database. */
	const bool needToRemoveFromPendingReAssocDB() { return m_needToRemoveFromPendingReAssocDB; }

	/** @brief Clear the pending reassociation database removal flag. */
	void resetNeedToRemoveFromPendingReAssocDB() { m_needToRemoveFromPendingReAssocDB = false; }

	/** @brief Report whether the state estimator uses aspect ratio. */
	const bool getStateUseAspc() { return m_stateEstimator->useAspectRatio(); }

	/**
	 * @brief Return the detection ReID feature pointer and size.
	 * @param[out] ptr Pointer updated with the ReID feature address.
	 * @param[out] reidFeatureSize Number of values in the ReID feature.
	 */
	void getReidFeaturesDet_h(const float **ptr, int &reidFeatureSize)
	{
		*ptr = m_reidFeaturesDet_h;
		reidFeatureSize = m_reidFeatureSize;
	}

	/**
	 * @brief Store detection ReID feature metadata and persist non-empty features.
	 * @param[in] ptr Pointer to the detection ReID feature.
	 * @param[in] reidFeatureSize Number of values in the ReID feature.
	 */
	void setReidFeaturesDet_h(const float *ptr, int reidFeatureSize)
	{
		m_reidFeaturesDet_h = ptr; m_reidFeatureSize = reidFeatureSize;
		/// Persist a copy of every non-null embedding so getPersistentReidFeature() is always valid.
		if (ptr && reidFeatureSize > 0) {
			m_persistentReidFeature.assign(ptr, ptr + reidFeatureSize);
		}
	}

	/** @brief Return the last persisted non-empty ReID feature. */
	const std::vector<float>& getPersistentReidFeature() const { return m_persistentReidFeature; }

	/** @brief Return the estimated target height or the default human height. */
	float getTargetHeight() const
	{
		return (m_objModelProjector && m_objModelProjector->isInitialized())
			? m_objModelProjector->getTargetHeight() : 1.6f;
	}

	/** @brief Return the confidence of the estimated target height. */
	float getHeightConfidence() const
	{
		return (m_objModelProjector && m_objModelProjector->isInitialized())
			? m_objModelProjector->getHeightConfidence() : 0.f;
	}

	/**
	 * @brief Update the estimated target height when an object model projector is available.
	 * @param[in] height Target height in world coordinates.
	 * @param[in] confidence Confidence of the target-height estimate.
	 * @param[in] force Forces the object model projector to accept the update when true.
	 */
	void setTargetHeight(float height, float confidence, bool force = false)
	{
		if (m_objModelProjector && m_objModelProjector->isInitialized()) {
			m_objModelProjector->setTargetHeight(height, confidence, force);
		}
	}

	/** @brief Return the minimum pose confidence or its default value. */
	float getMinPoseConfidence() const
	{
		return (m_objModelProjector && m_objModelProjector->isInitialized())
			? m_objModelProjector->getMinPoseConfidence() : 0.5f;
	}

	/** @brief Return the location covariance from the state estimator. */
	Eigen::Matrix2d getLocCovariance() const
	{
		return m_stateEstimator->getMeasurementNoiseCovariance().block<2,2>(0,0);
	}

	/** @brief Report whether the target uses a projective state estimator. */
	bool usesProjectiveEstimator() const
	{
		return m_stateEstimator->getStateEstimatorType() == STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_PROJECTIVE;
	}

	/** @brief Report whether the target has ever reached ACTIVE state. */
	bool hasBeenActive() const { return m_hasBeenActive; }

	/** @brief Report whether the target requires ReID-service reassociation. */
	const bool needToReAssocFromReidService() { return m_needToReAssocFromReidService; }

	/**
	 * @brief Set whether the target requires ReID-service reassociation.
	 * @param[in] flag New ReID-service reassociation flag value.
	 */
	void setNeedToReAssocFromReidService(const bool flag) { m_needToReAssocFromReidService = flag; }

	/** @brief Clear the ReID-service reassociation flag. */
	void resetNeedToReAssocFromReidService() { m_needToReAssocFromReidService = false; }

	/** @brief Report whether the target adopted an ID from the ReID service. */
	const bool adoptedReidServiceID() { return m_adoptedReidServiceID; }

	/**
	 * @brief Set whether the target adopted an ID from the ReID service.
	 * @param[in] flag New ReID-service ID adoption flag value.
	 */
	void setAdoptedReidServiceID(const bool flag) { m_adoptedReidServiceID = flag; }

	/** @brief Return the target ID held before ReID-service adoption. */
	const TargetId getTargetIDBeforeAdoption() { return m_targetIDBeforeAdoption; }

	/**
	 * @brief Store the target ID held before ReID-service adoption.
	 * @param[in] targetID Target ID to preserve before adoption.
	 */
	void setTargetIDBeforeAdoption(const TargetId targetID) { m_targetIDBeforeAdoption = targetID; }

	/** @brief Return the target ID assigned by the ReID service. */
	const TargetId getTargetIDFromReidService() { return m_targetIDFromReidService; }

	/**
	 * @brief Store the target ID assigned by the ReID service.
	 * @param[in] targetID Target ID assigned by the ReID service.
	 */
	void setTargetIDFromReidService(const TargetId targetID) { m_targetIDFromReidService = targetID; }

	/**
	 * @brief Store association responses returned by the ReID service.
	 * @param[in] reidServiceAssocResponseMap ReID-service responses keyed by target ID.
	 */
	void setReidServiceAssocResponse(const std::map<TargetId, reidServiceAssocInfo> reidServiceAssocResponseMap)
	{
		m_reidServiceAssocResponseMap = reidServiceAssocResponseMap;
	}

	/** @brief Return association responses from the ReID service. */
	const std::map<TargetId, reidServiceAssocInfo> &getReidServiceAssocResponse()
	{
		return m_reidServiceAssocResponseMap;
	}

	/**
	 * @brief Return the ReID-service match confidence for a target ID.
	 * @param[in] targetID Target ID whose match confidence is requested.
	 */
	float getReidServiceMatchConfidence(const TargetId targetID)
	{
		return (
			m_reidServiceAssocResponseMap.find(targetID) != m_reidServiceAssocResponseMap.end()
				? m_reidServiceAssocResponseMap.at(targetID).expConfidence : 0.0
		);
	}
protected:

	// [Configuration]
	const NvTrackerParamsTargetManager *m_params; //!< Configuration parameters for target management.
	std::shared_ptr<ObjectModelProjector> m_objModelProjector; //!< Refine bbox based on 3D object model.
	const TrackerIndex m_targetInd; //!< Index of this target in the allocated memory layout.
	TargetInfo m_targetInfo; //!< Tracked target information.

	// [Multi-view association and fusion]
	std::atomic<TargetId> m_targetID; //!< Unique target ID.
	std::atomic<TimeStamp> m_targetIDTimeStamp; //!< Time at which the target ID was created.
	std::atomic<bool> m_becameActive; //!< Indicates whether the target became ACTIVE in the current frame.
	bool m_hasBeenActive; //!< Indicates whether the target has ever transitioned to ACTIVE.
	std::atomic<bool> m_acquiredID; //!< Indicates whether the target acquired an ID for the first time.
	std::vector<PeerTarget> m_peerTargets; //!< Peer targets from other cameras.
	MultiViewAssociatorParams m_multiViewParams; //!< Parameters for the multi-view associator.
	NvMotDataTypes::Rect2f m_searchRegion; //!< Given a target bbox, expand the bbox to some extent to define a search region.
	NvMotDataTypes::Rect2f m_predictedSearchRegion; //!< Given a search region, predict the search region for the next frame.
	NvMotDataTypes::Rect2f m_progBbox; //!< Progressed bounding box generated by segmenter.
	NvMOTMask m_maskOriginal; //!< Object segmentation mask in original streammux coordinates.
	bool m_needToSetTargetID; //!< Indicates whether this target is ready for target ID assignment.
	bool m_ready4ReAssoc; //!< Indicates whether this target is ready for reassociation.
	bool m_needToRemoveFromPendingReAssocDB; //!< Indicates whether the target must leave the pending reassociation database after becoming ACTIVE.

	// [Reassociation from ReID service]
	bool m_needToReAssocFromReidService; //!< Indicates whether this target requires ReID-service reassociation.
	TargetId m_targetIDFromReidService; //!< Target ID assigned by the ReID service.
	std::map<TargetId, reidServiceAssocInfo> m_reidServiceAssocResponseMap; //!< Association information returned by the ReID service.
	bool m_adoptedReidServiceID; //!< Indicates whether an ID was adopted from the ReID service in the current frame.
	TargetId m_targetIDBeforeAdoption; //!< Target ID held before ReID-service adoption in the current frame.

	#ifdef VIZ_DEBUG
		cv::Scalar m_color; //!< Unique visualization color.
	#endif

	float m_peakCorrResp_long; //!< Peak correlation response over a long interval.
	float m_peakCorrResp_short; //!< Exponentially averaged peak correlation response over a short interval.
	//!< Used to judge whether the current correlation response is higher or lower than average.
	std::deque<TargetInfo> m_pastFrameDataBuffer; //!< Buffer storing past-frame data.
	std::list<TargetInfo> m_trajectoryBuffer; //!< Record the whole trajectory.
	int m_trajectorySkipInterval; //!< Frame interval between stored trajectory points.
	std::shared_ptr<IEstimator2> m_stateEstimator; //!< State estimator such as a Kalman or moving-average filter.
	std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool>> m_measList; //!< Measurements, covariances, and detector-source flags for the state estimator.
	std::vector<float> m_confidenceCorrRespAtDetectedPts; //!< Visual-tracker confidence scores at detected points.
	std::vector<float> m_corrRespAtDetectedPts; //!< Visual-tracker correlation responses at detected points.
	std::vector<float> m_confidenceReidDets; //!< ReID feature-similarity confidence for each detection.
	std::vector<std::vector<float>> m_confidenceListReidDets; //!< Detection confidence values from multiple ReID models.
	std::vector<int> m_confidenceReidDetsFrameNum; //!< Frame number associated with each extracted ReID feature.
	std::vector<float> m_disMahalanobisDets; //!< Mahalanobis distance between the target and each detection.
	const float *m_reidFeaturesDet_h; //!< Target ReID feature data.
	int m_reidFeatureSize = 0; //!< Number of values in each ReID feature.
	std::vector<float> m_persistentReidFeature; //!< Persisted copy of the last non-null embedding; survives non-extraction frames.

	bool m_isOffline = false; //!< Indicates whether the tracked object is part of online tracking or offline metadata fusion.

	#ifdef VIZ_DEBUG
		friend class TrackerVizOutput;
	#endif
};

typedef std::map< std::shared_ptr<NvTrackedObject>, std::pair<std::shared_ptr<NvTrackedObject>, int> > FusedObjectList;

/** @brief Target management module. */
class NvTrackedObjectManager
{
public:
	/**
	 * @brief Initialize the target manager from a tracker configuration.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 */
	NvTrackedObjectManager(const std::string &configFilePath);

	/** @brief Release all managed targets and stream submodules. */
	~NvTrackedObjectManager();

	/**
	 * @brief Allocate the target pool and initialize stream-independent submodules.
	 * @param[in] frameInfo Source frame sizes and stream limits used to allocate the target pool.
	 */
	void initialize(const SourceFrameInfo &frameInfo);

	/**
	 * @brief Populate target-manager memory requirements.
	 * @param[out] paramsQuery Query structure populated with target-manager requirements.
	 */
	void queryParams(NvMOTQuery &paramsQuery);

	/**
	 * @brief Update per-stream input information before target-manager processing.
	 * @param[in] inputMeta Batch of input frames keyed by stream ID.
	 */
	void updateInputInfo(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta);

	/** @brief Clean previous batch data. */
	void cleanPreviousBatchData();

	/**
	 * @brief Update targets and trajectories per stream in each thread.
	 * @param[in] assocOut Association output for one stream.
	 * @param[in] reid4reAssocState Re-identification state that selects target update, trajectory update, or both.
	 */
	void updatePerStream(const assocDataOut_t& assocOut, const REID4REASSOC_STATE reid4reAssocState=NOT_NEEDED);

	/** @brief Update targets for all streams on the main thread. */
	void updateTargets();

	/** @brief Update trajectories for all streams on the main thread. */
	void updateTrajectoryManager();

	/**
	 * @brief Update obj model projectors when streams are added or deleted.
	 * @param[in] inputMeta Input frames keyed by stream ID.
	 * @param[in] srcImageSize Source image size.
	 * @return True when at least one new stream is added.
	 */
	bool addStreamSubModules(
		const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta, const NvMotDataTypes::Size& srcImageSize
	);

	/**
	 * @brief Recover object model from detection bounding box using default model projector.
	 * @param[in] videoStreamId Video stream ID.
	 * @param[in] frameNum Current frame number.
	 * @param[in,out] detObjInfo Detected-object information updated with the recovered model.
	 */
	void recoverDefaultModelFromBbox(
		const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum, DetectedObjInfo &detObjInfo
	);

	/**
	 * @brief Update any Dynamic Params as provided during runtime by the user.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void updateParams(const YAML::Node &fn);

	/**
	 * @brief Remove all trackers for a stream WITHOUT deleting stream.
	 * @param[in] videoStreamId Stream whose trackers are removed while the stream is kept.
	 */
	void resetStream(const NvMOTStreamId videoStreamId);

	/** @brief Resets all trackers for all streams. */
	void resetAllStreams();

	/**
	 * @brief Remove all trackers for a stream and deletes stream.
	 * @param[in] videoStreamId Stream whose trackers and per-stream modules are removed.
	 */
	void removeStream(const NvMOTStreamId videoStreamId);

	/** @brief Remove all trackers for all streams. */
	void removeAllStreams();

	/**
	 * @brief Select targets for ReID feature extraction during reassociation.
	 * @param[in] inputMeta Input frames keyed by video stream ID.
	 */
	std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> generateTrackedObjects4Reid(
		const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta
	);

	/** @brief Report whether the target manager is initialized. */
	bool isInitialized() { return m_isInit; }

	/** @brief Return in-use targets for all streams. */
	const std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>> &getTrackedObjectsInUseList()
	{
		return m_TrackedObjectsInUse;
	}

	/**
	 * @brief Return in-use targets for one stream.
	 * @param[in] streamId Video stream ID whose targets are requested.
	 */
	NvTrackedObjectList getTrackedObjectsInUseList(const NvMOTStreamId streamId);

	/** @brief Return targets terminated in the current batch. */
	const std::map<NvMOTStreamId, std::vector<std::shared_ptr<NvTrackedObject>>> & \
		getTrackedObjectsTerminated() { return m_trackedObjectsTerminated; }

	/** @brief Return the index of the last allocated tracker. */
	const int getLastTargetInd() { return m_lastTrackerInd; }

	/** @brief Return the fraction of total target capacity currently in use. */
	const float getTargetCapacityUtilization() { return m_targetCapacityUtilization; }

	/** @brief Return the maximum number of targets allowed per stream. */
	const uint getMaxTargetsPerStream() { return m_params.m_maxTargetsPerStream; }

	/** @brief Return targets fused during reassociation. */
	const std::map<NvMOTStreamId, FusedObjectList> &getTrackedObjectsFused() { return m_trackedObjectsFused; }

	/** @brief Return default object model projectors keyed by stream ID. */
	std::map<const NvMOTStreamId, std::shared_ptr<ObjectModelProjector>>& getObjModelProjectors()
	{
		return m_defaultModelProjectors;
	}

	/** @brief Return the minimum visibility required for trajectory dumping. */
	const float getMinVisibility4TrajDump();

	/** @brief Report whether detector bounding boxes are included in debug dumps. */
	const bool getDumpDetectorBboxes();

	/**
	 * @brief Return peer targets visible to a stream.
	 * @param[in] streamId Video stream ID whose peer targets are requested.
	 */
	const PeerTargetDB getPeerTargets(const NvMOTStreamId streamId);

	/**
	 * @brief Return the multi-view associator for a stream.
	 * @param[in] streamId Video stream ID whose associator is requested.
	 */
	const std::shared_ptr<IMultiViewAssociator> getMultiViewAssociators(const NvMOTStreamId streamId)
	{
		if (m_multiViewAssociators.find(streamId) != m_multiViewAssociators.end())
		{
			return m_multiViewAssociators.at(streamId);
		}
		else
		{
			return nullptr;
		}
	};

	/**
	 * @brief Return the terminated-track output list for a stream when available.
	 * @param[in] streamId Video stream ID whose terminated tracks are requested.
	 * @param[out] pTerminatedList Pointer updated with the stream's terminated-track list.
	 * @return True when a terminated-track list exists for the stream.
	 */
	bool getOutputTerminatedTracks(const NvMOTStreamId streamId, std::list<TerminatedTrack>* & pTerminatedList)
	{
		if(m_terminatedTrackOutputMap_.find(streamId) != m_terminatedTrackOutputMap_.end())
		{
			pTerminatedList = &m_terminatedTrackOutputMap_[streamId];
			return true;
		}
		return false;
	}

	/** @brief Clear buffered terminated-track output when saving is enabled. */
	void clearOutputTerminatedTracks()
	{
		if(!m_params.m_saveTerminatedTracks)
		{
			return;
		}

		for( auto curStreamIter = m_terminatedTrackOutputMap_.begin(); curStreamIter != m_terminatedTrackOutputMap_.end(); curStreamIter++ )
		{
			std::list<TerminatedTrack> & curtermTrackList = curStreamIter->second;
			curtermTrackList.clear();
		}
	}

	/**
	 * @brief Terminate a target through the public manager interface.
	 * @param[in] target Target to terminate.
	 * @param[in] deleteMode Reason the target is terminated.
	 * @param[in] peer Peer target that absorbed this target, or null.
	 */
	void terminateTrackerPublic(
		const std::shared_ptr<NvTrackedObject> target, const TRACKER_DELETE_MODE deleteMode,
		const std::shared_ptr<NvTrackedObject> peer=nullptr
	)
	{
		terminateTracker(target, deleteMode, peer);
	}

	/** @brief Write all buffered terminated tracks to the configured debug output. */
	void dumpTerminatedTracks();

	/**
	 * @brief Write intermediate detector and target information for one stream.
	 * @param[in] assocOut Association output for the stream being dumped.
	 */
	void dumpIntermediateTargetInfo(const assocDataOut_t& assocOut);

	/**
	 * @brief Return shadow tracks collected for a stream in the current batch.
	 * @param[in] streamId Video stream ID whose shadow tracks are requested.
	 */
	std::set<std::shared_ptr<NvTrackedObject>> & getShadowTracks(const NvMOTStreamId streamId)
	{
		return m_shadowTracks[streamId];
	}

	/** @brief Clear shadow tracks collected for the current batch. */
	void clearShadowTracks() { m_shadowTracks.clear(); }

	/**
	 * @brief Store batched segmentation-mask information.
	 * @param[in] segInfo Batched segmentation-mask storage to use.
	 */
	void setSegmentationBatchInfo(SegmentationInBatch segInfo) { m_segInfo = segInfo; }

	/** @brief Return batched segmentation-mask information. */
	const SegmentationInBatch &getSegmentationBatchInfo() { return m_segInfo; }

	NvTrackerParamsTargetManager m_params; //!< Configuration parameters for target management.
	MultiViewAssociatorParams m_multiViewParams; //!< Configuration parameters for multi-view association.

	/** @brief ACTIVE track termination counters (per NvTrackedObjectManager instance). */
	struct TM_Stats
	{
		std::atomic<uint64_t> term_reg_total{0}; //!< Total REGULAR_TERMINATION events.
		std::atomic<uint64_t> term_reg_was_active{0}; //!< Targets that had reached ACTIVE before regular termination.
		std::atomic<uint64_t> term_reg_shadow_1{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age one.
		std::atomic<uint64_t> term_reg_shadow_2{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age two.
		std::atomic<uint64_t> term_reg_shadow_3{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age three.
		std::atomic<uint64_t> term_reg_shadow_4{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age four.
		std::atomic<uint64_t> term_reg_shadow_5{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age five.
		std::atomic<uint64_t> term_reg_shadow_6{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age six.
		std::atomic<uint64_t> term_reg_shadow_7{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age seven.
		std::atomic<uint64_t> term_reg_shadow_8{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age eight.
		std::atomic<uint64_t> term_reg_shadow_9plus{0}; //!< ACTIVE-lineage targets terminated regularly at shadow age nine or later.
		std::atomic<uint64_t> term_reg_young{0}; //!< ACTIVE-lineage targets terminated before age 30.
		std::atomic<uint64_t> term_reg_mature{0}; //!< ACTIVE-lineage targets terminated at age 30 or later.

		// [Shadow rescues]
		std::atomic<uint64_t> shadow_rescue_6{0}; //!< INACTIVE-to-ACTIVE matches at shadow age six.
		std::atomic<uint64_t> shadow_rescue_7{0}; //!< INACTIVE-to-ACTIVE matches at shadow age seven.
		std::atomic<uint64_t> shadow_rescue_8plus{0}; //!< INACTIVE to ACTIVE re-match at shadowAge of 8 or more.

		// [TENTATIVE terminations]
		std::atomic<uint64_t> term_tent_total{0}; //!< Total EARLY_TERMINATION events for TENTATIVE targets.
		std::atomic<uint64_t> term_tent_age_le2{0}; //!< TENTATIVE death at age ≤ 2 (very short-lived).
		std::atomic<uint64_t> term_tent_age_3_4{0}; //!< TENTATIVE death at age 3-4.
		std::atomic<uint64_t> term_tent_age_5_7{0}; //!< TENTATIVE death at age 5-7.
		std::atomic<uint64_t> term_tent_age_8_10{0}; //!< TENTATIVE death at age 8-10 (near probationAge=10).
		std::atomic<uint64_t> term_tent_age_11plus{0}; //!< TENTATIVE death at age > 10 (past probation).
	} m_tm_stats;

protected:

	bool m_isInit; //!< Indicates whether the target manager is initialized.

	// [Configuration]
	const std::string m_configFilePath; //!< Path to the tracker configuration file.

	// [Worker-thread shared state]
	std::mutex m_trackedObjectsLock; //!< Lock associated with tracked objects for all streams.
	NvTrackedObjectList m_TrackedObjectsInPool; //!< Pool of all preallocated tracked objects.
	std::map<NvMOTStreamId, std::vector<std::shared_ptr<NvTrackedObject>>> m_trackedObjectsTerminated; //!< Stores terminated tracker in the current batch.
	std::map<NvMOTStreamId, FusedObjectList> m_trackedObjectsFused; //!< Targets fused during reassociation, keyed by stream.
	std::map<NvMOTStreamId, assocDataOut_t> m_assocOuts; //!< Store assoc output for each stream. Only used when preserving update order.
	int m_lastTrackerInd; //!< The ID of the last active tracker along the device memory space.
	std::map<NvMOTStreamId, bool> m_isFileTerminatedDumpCreated; //!< Indicates whether each stream's terminated-track dump file was created.
	std::map<NvMOTStreamId, std::ofstream> m_fileTerminatedDump; //!< Terminated-track dump stream for each video stream.
	std::map<NvMOTStreamId, std::ofstream> m_fileIntermediateDetDump; //!< Intermediate detection dump stream for each video stream.
	std::map<NvMOTStreamId, std::ofstream> m_fileIntermediateTrackDump; //!< Intermediate target dump stream for each video stream.
	std::map<NvMOTStreamId, std::list<TerminatedTrack>> m_terminatedTrackOutputMap_; //!< Terminated tracks keyed by video stream ID.
	std::map<NvMOTStreamId, std::set<std::shared_ptr<NvTrackedObject>>> m_shadowTracks; //!< Stores shadow tracks in the current batch.
	std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>> m_TrackedObjectsInUse; //!< Tracked object info should only be modified in main thread and is read only in worker threads.

	float m_targetCapacityUtilization; //!< The utilization of the total allocated target capacity. [Unit: %].
	std::shared_ptr<NvTrajectoryManager> m_trajectoryManager; //!< Manages the trajectory generated by each target.
	std::map<const NvMOTStreamId, std::shared_ptr<ObjectModelProjector>> m_defaultModelProjectors; //!< Default object model projectors for each stream.
	SegmentationInBatch m_segInfo; //!< Segmentation results for all the targets.
	std::map<NvMOTStreamId, std::shared_ptr<IMultiViewAssociator>> m_multiViewAssociators; //!< Multi-view associators.

	/**
	 * @brief Actually perform target update per stream.
	 * @param[in] assocOut Association results used to update targets on one stream.
	 */
	void updateTargetsPerStream(const assocDataOut_t& assocOut);

	/**
	 * @brief Update the targets that have matched detector objects.
	 * @param[in] videoStreamId Video stream ID.
	 * @param[in] frameNum Current frame number.
	 * @param[in] matchedPairs Matched pairs as detected object and target index.
	 */
	void updateMatched(
		const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum,
		const std::deque<std::pair<const DetectedObjInfo *, TrackerIndex >> &matchedPairs
	);

	/**
	 * @brief Update the targets that do not have any matched detector objects.
	 * @param[in] videoStreamId Stream that owns the unmatched targets.
	 * @param[in] frameNum Current frame number.
	 * @param[in] unmatchedTargets Target indexes with no matched detection.
	 */
	void updateUnMatched(
		const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum, const std::set<TrackerIndex> &unmatchedTargets
	);

	/**
	 * @brief Add new targets for the new detected objects.
	 * @param[in] videoStreamId Stream that owns the new detections.
	 * @param[in] frameNum Current frame number.
	 * @param[in] unmatchedDetObjs Detections that were not matched to an existing target.
	 */
	void addTargetsBatch(
		const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum,
		const std::map<const TrackerIndex, const DetectedObjInfo*> &unmatchedDetObjs
	);

	/**
	 * @brief Add see-through targets from peer cameras.
	 * @param[in] videoStreamId Video stream ID.
	 * @param[in] frameNum Current frame number.
	 */
	void addSeeThroughTargets(const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum);

	/**
	 * @brief Add a new target for a new object.
	 * @param[in] videoStreamId Stream that receives the new target.
	 * @param[in] frameNum Current frame number.
	 * @param[in] detObj Detected object for the new target, or null when a peer target is supplied.
	 * @param[in] peerTarget Peer-camera target used to initialize target information.
	 * @return False when the new target overlaps an existing target or no tracker slot is available.
	 */
	bool addTarget(
		const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum,
		const DetectedObjInfo *detObj, const PeerTarget *peerTarget=nullptr
	);

	/**
	 * @brief Check the termination criteria for a target and terminate if met.
	 * @param[in] target Target whose shadow age is checked against the termination limits.
	 */
	void checkToTerminateTracker(const std::shared_ptr<NvTrackedObject> target);

	/**
	 * @brief Actually terminate tracker.
	 * @param[in] target Target to terminate.
	 * @param[in] deleteMode Reason the tracker is terminated.
	 * @param[in] peer Peer that absorbed this track on re-association, or null.
	 */
	void terminateTracker(
		const std::shared_ptr<NvTrackedObject> target, const TRACKER_DELETE_MODE deleteMode,
		const std::shared_ptr<NvTrackedObject> peer=nullptr
	);

	/**
	 * @brief Update traj manager to assign ids, perform re-association and terminate trackers for each stream.
	 * @param[in] assocOut Association results for the stream whose trajectories are updated.
	 */
	virtual void updateTrajectoryManagerPerStream(const assocDataOut_t& assocOut);

	/**
	 * @brief Add the current target info to the trajectory buffer.
	 * @param[in] videoStreamId Stream whose in-use targets append the current frame.
	 */
	void updateTrajectoryBuffers(const NvMOTStreamId videoStreamId);

	/** @brief Update target capacity utilization. */
	void updateTargetCapacityUtilization();

	/** @brief Output shadow track objects in current batch. */
	void saveShadowTrackInfo();

	/**
	 * @brief Save the current target in shadow track objects.
	 * @param[in] target Inactive target recorded when its shadow-tracking age is positive.
	 */
	void addTargetInShadowTrack(std::shared_ptr<NvTrackedObject> target);

	/** @brief Dump all the target states in a structured text for debug. */
	const std::string dumpAllTargetStates();

	/**
	 * @brief Save and output the terminated tracks.
	 * @param[in] terminatedTarget Target whose identity and trajectory are copied.
	 * @return Terminated-track record for the target, or an empty record when terminated tracks are not saved.
	 */
	TerminatedTrack createTerminatedTrack(const std::shared_ptr<NvTrackedObject> terminatedTarget);

	/**
	 * @brief Remove submodules when a stream is removed.
	 * @param[in] videoStreamId Stream whose model projector and multi-view associator are removed.
	 */
	void removeStreamSubModules(const NvMOTStreamId videoStreamId);

	/** @brief Create multiview associators by replacing the existing instances after stream addition or deletion. */
	void createMultiViewAssociators();

	/**
	 * @brief Append a terminated track to its stream output buffer.
	 * @param[in] newTermTrack Terminated track appended to that stream's output buffer.
	 */
	void addTerminatedTrackToBufferPerStream(TerminatedTrack& newTermTrack);

	/** @brief Return the number of tracker slots in the object pool. */
	uint getSizeTrackedObjectsInPool() const { return m_TrackedObjectsInPool.size(); }

	#ifdef VIZ_DEBUG
		friend class TrackerVizOutput;
	#endif
};

#endif // NVTRACKEDOBJECT_HPP_
