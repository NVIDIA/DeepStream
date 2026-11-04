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

#ifndef NVMULTIOBJECTTRACKERLIB_NVMULTIOBJECTTRACKER_COMMON_HPP_
#define NVMULTIOBJECTTRACKERLIB_NVMULTIOBJECTTRACKER_COMMON_HPP_

#include "nvdstracker.h"
#include "nvds_tracker_meta.h"

#include <vector>
#include <map>
#include <deque>
#include <queue>
#include <utility>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef _WIN32
using uint = unsigned int;
#endif

#include "modules/utility/cv_functions.hpp"
#include "modules/NvMultiObjectTracker/localizer.hpp"

typedef uint64_t TargetId;
typedef uint32_t TrackerIndex;
typedef NvMOTStreamId StreamId;

/** @brief Camera IDs are represented as strings to maintain consistency across different communicator types. */
typedef std::string CamId;

/** @brief Information about the input source frames for the input frame batch. */
struct SourceFrameInfo
{
	/** @brief Initialize per-batch source-frame state. */
	SourceFrameInfo()
	{
		frameBatchNo = 0;
		isDetectionDonePerStream.clear();
	}

	// [Per-frame source parameters]
	uint64_t frameBatchNo; //!< Frame batch count from the beginning.
	std::map<NvMOTStreamId, std::pair<uint32_t, bool>> isDetectionDonePerStream; //!< Indicates whether inference is complete for each stream.

	// [Plugin configuration]
	NvBufSurfaceColorFormat colorFormat; //!< Color format of the input surfaces.
	uint32_t maxStreams; //!< Max # of video streams to be supported.
	uint32_t maxBufSurfAddrSize; //!< Max size of surface buffer pool.
	NvMotDataTypes::Size originalSrcImageSize; //!< Size of the original video source image.
	NvMotDataTypes::Size srcImageSize; //!< Size of the source image.
	size_t srcImagePitch; //!< Row pitch of the source-frame memory.
};

/** @brief Detected object info obtained from input metadata. */
struct DetectedObjInfo
{
	/** @brief Move-construct detected-object information. */
	DetectedObjInfo(DetectedObjInfo&&) noexcept = default;

	/** @brief Move-assign detected-object information. */
	DetectedObjInfo& operator=(DetectedObjInfo&&) noexcept = default;

	/** @brief Copy-construct detected-object information. */
	DetectedObjInfo(const DetectedObjInfo&) = default;

	/** @brief Copy-assign detected-object information. */
	DetectedObjInfo& operator=(const DetectedObjInfo&) = default;

	/** @brief Initialize detected-object information with default values. */
	DetectedObjInfo()
	{
		objIn = nullptr;
		visibility = 1.f;
		pt_img_feet << -1, -1;
		pt_world_feet << -1, -1;
		convexHull.clear();
		distFromCam = std::numeric_limits<float>::max();
	}

	/**
	 * @brief Initialize detected-object information from explicit metadata and geometry.
	 * @param[in] objIn_ Pointer to the input object metadata.
	 * @param[in] bboxInd_ Object bounding-box index in the input metadata.
	 * @param[in] classId_ Object class identifier.
	 * @param[in] confidence_ Detector confidence score.
	 * @param[in] bbox_ Object bounding box.
	 * @param[in] loc_ Object location in image coordinates.
	 */
	DetectedObjInfo(
		const NvMOTObjToTrack *objIn_, const uint16_t bboxInd_, NvMOTClassId classId_,
		float confidence_, NvMotDataTypes::Rect2f bbox_, Eigen::Vector2f loc_
	)
	: objIn(objIn_),
	  bboxInd(bboxInd_),
	  classId(classId_),
	  confidence(confidence_),
	  bbox(bbox_),
	  origBbox(bbox_),
	  loc(loc_)
	{
		visibility = 1.f;
		pt_img_feet << -1, -1;
		pt_world_feet << -1, -1;
		convexHull.clear();
	}

	/**
	 * @brief Initialize detected-object information from input object metadata.
	 * @param[in] objIn_ Pointer to the input object metadata.
	 * @param[in] bboxInd_ Object bounding-box index in the input metadata.
	 */
	DetectedObjInfo(const NvMOTObjToTrack *objIn_, const uint16_t bboxInd_)
	: objIn(objIn_), bboxInd(bboxInd_)
	{
		classId = objIn_->classId;
		confidence = objIn_->confidence;
		visibility = 1.f;
		distFromCam = std::numeric_limits<float>::max();
		bbox = CV_FUNCS::constructRect2f(objIn_->bbox.x, objIn_->bbox.y, objIn_->bbox.width, objIn_->bbox.height);
		origBbox = bbox;
		loc = CV_FUNCS::getBboxCenterPt(bbox);
		pt_img_feet << -1, -1;
		pt_world_feet << -1, -1;
		convexHull.clear();
	}

	const NvMOTObjToTrack *objIn; //!< Holds a pointer to the associated input object in the metadata.
	uint16_t bboxInd; //!< Object bbox ind in the metadata.
	NvMOTClassId classId; //!< Object class ID.
	float confidence; //!< Detector confidence.
	float visibility; //!< Visible fraction of the object after occlusion and image-border cropping.
	float distFromCam; //!< Distance from the camera center to the model waist in world coordinates.
	NvMotDataTypes::Rect2f bbox; //!< Object bbox.
	NvMotDataTypes::Rect2f origBbox; //!< Original bounding box from the detector.
	Eigen::Vector2f loc; //!< Bounding-box centroid in image coordinates.
	Eigen::Vector2f pt_img_feet; //!< Person foot location in image coordinates.
	Eigen::Vector2f pt_world_feet; //!< Person foot location on the floor in world coordinates.
	std::vector<Eigen::Vector2f> convexHull; //!< Convex hull of the projected 3D object model centered on the bounding box.
};

/** @brief Tracked target info. */
struct TargetInfo
{
	/** @brief Move-construct target information. */
	TargetInfo(TargetInfo&&) noexcept = default;

	/** @brief Move-assign target information. */
	TargetInfo& operator=(TargetInfo&&) noexcept = default;

	/** @brief Copy-construct target information. */
	TargetInfo(const TargetInfo&) = default;

	/** @brief Copy-assign target information. */
	TargetInfo& operator=(const TargetInfo&) = default;

	/**
	 * @brief Initialize a target state from optional detector information.
	 * @param[in] detObjIn_ Optional detected object matched to the target.
	 * @param[in] videoStreamId_ Video stream identifier associated with the target.
	 * @param[in] frameNum_ Frame number associated with the target state.
	 * @param[in] classId_ Object class identifier.
	 * @param[in] bbox_ Target bounding box.
	 * @param[in] age_ Target age in frames.
	 * @param[in] shadowAge_ Number of frames spent in shadow tracking.
	 * @param[in] state Initial tracker state.
	 */
	TargetInfo(
		const DetectedObjInfo* detObjIn_ = nullptr, const NvMOTStreamId videoStreamId_ = 0,
		const NvMOTFrameNum frameNum_ = 0, const uint16_t classId_ = 0,
		const NvMotDataTypes::Rect2f bbox_ = NvMotDataTypes::Rect2f(), const NvMOTFrameNum age_ = 0,
		const NvMOTFrameNum shadowAge_ = 0, const TRACKER_STATE state = TRACKER_STATE::EMPTY
	)
	{
		matchedDetObjInfo = (DetectedObjInfo *) detObjIn_;

		videoStreamId = videoStreamId_;
		frameNum = frameNum_;
		classId = classId_;
		tBbox = bbox_;
		loc << float(tBbox.min()(0) + tBbox.sizes()(0)/2.f), float(tBbox.min()(1) + tBbox.sizes()(1)/2.f);
		age = age_;

		shadowTrackingAge = shadowAge_;
		trackerState = state;

		if(detObjIn_ != nullptr)
		{
			distFromCam = detObjIn_->distFromCam;
			origBbox = detObjIn_->origBbox;
			visibility = detObjIn_->visibility;
			convexHull = detObjIn_->convexHull;
			pt_img_feet = detObjIn_->pt_img_feet;
			pt_world_feet = detObjIn_->pt_world_feet;
		}
		else
		{
			distFromCam = std::numeric_limits<float>::max();
			origBbox = bbox_;
			visibility = 1.f;
			convexHull.clear();
			pt_img_feet << -1, -1;
			pt_world_feet << -1, -1;
		}
		pt_world_feet_cov = Eigen::Matrix2f::Identity();
		imgBorderCroppingVisibility = 1.f;
		/// Tracking confidence until a visual tracker overwrites it.
		confidence = 0.5f;
		/// No pose-refined height until OMP updates it.
		height = 0.0f;
	}

	/**
	 * @brief Update target state from a matched detection.
	 * @param[in] frameNum_ Frame number associated with the update.
	 * @param[in] detObjIn_ Detected object matched to the target.
	 * @param[in] bbox_ Updated target bounding box.
	 * @param[in] imgBorderCroppingVisibility_ Target visibility after image-border cropping.
	 */
	void update(
		const NvMOTFrameNum frameNum_, const DetectedObjInfo *detObjIn_,
		const NvMotDataTypes::Rect2f bbox_, const float imgBorderCroppingVisibility_
	)
	{
		matchedDetObjInfo = (DetectedObjInfo *) detObjIn_;

		frameNum = frameNum_;
		classId = detObjIn_->classId;
		tBbox = bbox_;
		loc << float(tBbox.min()(0) + tBbox.sizes()(0)/2.f), float(tBbox.min()(1) + tBbox.sizes()(1)/2.f);
		distFromCam = detObjIn_->distFromCam;
		origBbox = detObjIn_->origBbox;
		visibility = detObjIn_->visibility;
		convexHull = detObjIn_->convexHull;
		imgBorderCroppingVisibility = imgBorderCroppingVisibility_;
		age++;

		pt_img_feet = detObjIn_->pt_img_feet;
		pt_world_feet = detObjIn_->pt_world_feet;
	}

	/**
	 * @brief Update target state without a matched detection.
	 * @param[in] frameNum_ Frame number associated with the update.
	 * @param[in] bbox_ Updated target bounding box.
	 * @param[in] imgBorderCroppingVisibility_ Target visibility after image-border cropping.
	 */
	void update(
		const NvMOTFrameNum frameNum_, const NvMotDataTypes::Rect2f bbox_, const float imgBorderCroppingVisibility_
	)
	{
		matchedDetObjInfo = nullptr;

		frameNum = frameNum_;
		tBbox = bbox_;
		loc << float(tBbox.min()(0) + tBbox.sizes()(0)/2.f), float(tBbox.min()(1) + tBbox.sizes()(1)/2.f);
		distFromCam = std::numeric_limits<float>::max();
		origBbox = bbox_;
		visibility = 0.f;
		imgBorderCroppingVisibility = imgBorderCroppingVisibility_;
		age++;
	}

	/**
	 * @brief Update the target bounding box and centroid.
	 * @param[in] bbox_ Updated target bounding box.
	 */
	void setLocation(const NvMotDataTypes::Rect2f bbox_)
	{
		tBbox = bbox_;
		loc << float(tBbox.min()(0) + tBbox.sizes()(0)/2.f), float(tBbox.min()(1) + tBbox.sizes()(1)/2.f);
	}

	/** @brief Return the input object metadata matched to the target. */
	NvMOTObjToTrack* getMatchedDetObjIn() const
	{
		return (NvMOTObjToTrack *) (matchedDetObjInfo ? matchedDetObjInfo->objIn : nullptr);
	}

	DetectedObjInfo *matchedDetObjInfo; //!< Holds a pointer to the associated input object used in low level tracker.
	NvMOTStreamId videoStreamId; //!< Video stream ID set during initialization.
	NvMOTFrameNum frameNum; //!< Frame number.
	NvMOTClassId classId; //!< Object class ID.
	NvMotDataTypes::Rect2f tBbox; //!< Object bbox.
	NvMotDataTypes::Rect2f origBbox; //!< Original bounding box from the detector.
	Eigen::Vector2f loc; //!< Bounding-box centroid.
	float confidence; //!< Tracker confidence.
	float distFromCam; //!< Distance from the camera.
	float height; //!< OMP-estimated person height in world coordinates (m).
	//!< Zero until a bodypose3dnet update exceeds minPoseConfidence.
	float imgBorderCroppingVisibility; //!< Bounding-box visibility with respect to the image border.
	float visibility; //!< Visible fraction of the object after occlusion and image-border cropping.
	NvMOTFrameNum age; //!< Number of frames for which the target has been tracked.
	Eigen::Vector2f pt_img_feet; //!< Person foot location in image coordinates.
	Eigen::Vector2f pt_world_feet; //!< Person foot location on the floor in world coordinates.
	Eigen::Matrix2f pt_world_feet_cov; //!< Covariance of the foot location.
	std::vector<Eigen::Vector2f> convexHull; //!< Convex hull of the projected 3D object model centered on the bounding box.
	std::vector<float> pose25d; //!< Per-object 2.5D pose keypoints.
	std::vector<float> pose3d; //!< Per-object 3D pose keypoints.
	uint32_t shadowTrackingAge; //!< Number of frames since the tracker entered INACTIVE state.
	TRACKER_STATE trackerState; //!< Current tracker state such as ACTIVE or INACTIVE.
};

/** @brief Represents a tracked object managed by the tracker. */
class NvTrackedObject;
typedef std::map<TrackerIndex, std::shared_ptr<NvTrackedObject>> NvTrackedObjectList;

/** @brief Stores a terminated target trajectory for archival output. */
struct Trajectory4Archive
{
	uint32_t frameNumTerminated; //!< Frame number at which the target was terminated.
	uint32_t lastFrameMatched; //!< Last frame number on which the target matched a detector bounding box.
	std::list<TargetInfo> tracklet; //!< Target trajectory.
	bool isReported; //!< Indicates whether the tracklet has been reported externally.
	std::vector<float> reidEmbedding; //!< Archived L2-normalized ReID embedding captured at termination.

	#ifdef VIZ_DEBUG
		bool isVisualized; //!< Indicates whether the tracklet has been visualized.
	#endif
};

/** @brief Input data structure to compute Mahalanobis distance. */
typedef std::pair<const std::map<NvMOTStreamId, NvMOTFrame *> *, \
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> *> DisMahalanobisIn;

/** @brief Multi-threaded producer-consumer queue to perform some task. */
template <typename T> class NvTrackerMultiThreadWorker
{
public:

	// [Worker thread operations]
	bool m_isInit; //!< Indicates whether the worker has been initialized.
	std::mutex m_inputLock; //!< Lock associated with input queue.
	std::mutex m_outputLock; //!< Lock associated with output.
	std::condition_variable m_cond4In; //!< Condition variable associated with the input queue.
	std::condition_variable m_cond4Out; //!< Condition variable associated with the output.
	std::queue<T> m_inputQueue; //!< Queue for input.
	std::vector<std::thread> m_threadList; //!< Persistent threads over batches.
	bool m_persistThreadRunning; //!< Whether associate persistent threads should keep running; set false if receive deinit signal.
	uint m_submittedJobCnt; //!< Keep track of how many tasks are submitted.
	uint m_finishedJobCnt; //!< Keep track of how many tasks are finished.
	std::function<void(T)> m_funcPerThread; //!< Data processing function in each worker thread.

	/** @brief Initialize an empty worker queue. */
	NvTrackerMultiThreadWorker();

	/** @brief Stop and join all persistent worker threads. */
	~NvTrackerMultiThreadWorker();

	/**
	 * @brief Launch all worker threads and worker function.
	 * @param[in] numThreads Number of worker threads.
	 * @param[in] funcPerThread Function to execute in each thread.
	 */
	void initialize(const uint numThreads, std::function<void(T)> funcPerThread);

	/**
	 * @brief Submit association input data to queue in main thread.
	 * @param[in] inData Job data queued for a worker thread.
	 */
	void submitInput(T inData);

	/** @brief Wait all worker threads to finish in main thread. */
	void waitToFinish();

	/** @brief Process queued jobs until the worker receives a deinitialization signal. */
	void threadLoopFunc();

	/** @brief Report whether the worker queue is initialized. */
	bool isInitialized() { return m_isInit; }
};

#endif // NVMULTIOBJECTTRACKERLIB_NVMULTIOBJECTTRACKER_COMMON_HPP_
