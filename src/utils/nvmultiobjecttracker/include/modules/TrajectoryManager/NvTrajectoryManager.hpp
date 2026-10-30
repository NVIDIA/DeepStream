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

#ifndef NVMULTIOBJECTTRACKERLIB_NVTRAJECTORYMANAGER_HPP_
#define NVMULTIOBJECTTRACKERLIB_NVTRAJECTORYMANAGER_HPP_

#include <atomic>
#include <list>
#include <memory>
#include <iostream>
#include <map>
#include <set>
#include <fstream>
#ifndef _WIN32
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "modules/utility/debug_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/MultiViewAssociator/IMultiViewAssociator.hpp"

#ifdef VIZ_DEBUG
/** @brief Forward declaration. */
class TrackerVizOutput;
#endif

/** @brief Defines the parameters that are specific to offline trajectory management. */
typedef struct
{
	int m_maxProjTrajectoryLength4Matching; //!< Maximum length of the projected trajectory that will be considered for matching.
	int m_minOverlappedTrajectoryLength4Matching; //!< Minimum length of overlapped trajectory that is required for matching.
	//!< If the overlapped trajectory length is lower than this, then projected trajectory is included in matching.
	float m_iouAvgScaleFactor; //!< Scales the average IOU proportional to the distance between the two tracklets being associated.
} TrajectoryManagerOfflineParams;

/** @brief Store trajectory-manager configuration parameters. */
class NvTrackerParamsTrajectoryManager : public NvTrackerParams
{
public:
	/** @brief Initialize trajectory-manager parameters with default values. */
	NvTrackerParamsTrajectoryManager();

	/**
	 * @brief Initialize trajectory-manager parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	NvTrackerParamsTrajectoryManager(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Initialize trajectory-manager parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	NvTrackerParamsTrajectoryManager(const YAML::Node &fn);

	/**
	 * @brief Update trajectory-manager parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Controls debug message verbosity.
	bool m_useUniqueID; //!< Enables unique tracker IDs.
	int m_targetIDOffset; //!< Offset value for Target ID.
	bool m_enableReAssoc; //!< Enable Re-Assoc.
	bool m_checkClassMatch; //!< Check class match for re-assoc (sourced from BaseConfig:checkClassMatch).
	bool m_enable3DReAssoc; //!< Archive-only world-distance re-association, independent of m_enableReAssoc.
	//!< The YAML key enableArchiveReAssoc is accepted as an alias.
	int m_maxFrameGap4WorldDistReAssoc; //!< Max frame gap for archive world-dist re-assoc (0=unlimited).
	float m_maxNmCascadeWorldDist; //!< Max world-dist for nm cascade activation. 99.0 = no limit.
	float m_nmCascadeMinCosSim; //!< Near-miss cascade cosine similarity at the cross-stream gate.
	float m_nmCascadeMaxCosSim; //!< Near-miss cascade cosine-similarity plateau.
	float m_nmCascadeDecayRate; //!< Near-miss cascade decay rate in 1/m.
	float m_maxWsExt25WorldDist; //!< Within-stream extended-rescue upper limit.
	float m_wsExt25MinCosSim; //!< Minimum cosine similarity for within-stream extended rescue between the world-distance gate and 3.0 m.
	float m_wsExt25StrictCosSim; //!< Stricter cosine similarity for within-stream extended rescue from 3.0 m out to m_maxWsExt25WorldDist.

	// [Re-association metric: thresholds for valid candidates]
	float m_minMatchingScore4Overall; //!< Min matching score for overall.
	float m_minTrackletMatchingScore; //!< Min tracklet similarity score for matching.
	float m_minMatchingScore4ReidSimilarity; //!< Min reid similarity score for re-assoc.

	// [Re-association metric: weights]
	float m_matchingScoreWeight4TrackletSimilarity; //!< Weight for tracklet similarity score.
	float m_matchingScoreWeight4ReidSimilarity; //!< Weight for reid similarity score.

	// [Re-association: motion-based]
	int m_maxAge4TrackletArchiving; //!< The max age (i.e., frames) of the tracklets that are kept in the archive.
	int m_minTrajectoryLength4Projection; //!< Min trajectory length required to make projected trajectory.
	int m_trajectoryProjectionLength; //!< The length of the projected trajectory.
	int m_prepLength4TrajectoryProjection; //!< The length of the trajectory during which the state estimator is updated to make projections.

	// [Re-association: trajectory projector parameters]
	float m_trajectoryProjectionProcessNoiseScale; //!< Trajectory projector's process noise scale w.r.t. state estimator.
	float m_trajectoryProjectionMeasurementNoiseScale; //!< Trajectory projector's measurement noise scale w.r.t. state estimator.

	// [Re-association: trajectory similarity]
	float m_maxAngle4TrackletMatching; //!< Max angle difference for tracklet matching [degree].
	float m_minSpeedSimilarity4TrackletMatching; //!< Min speed similarity for tracklet matching.
	float m_minBboxSizeSimilarity4TrackletMatching; //!< Min bbox size similarity for tracklet matching.
	float m_trackletSpacialSearchRegionScale; //!< The search region scale for peer tracklet.
	int m_maxTrackletMatchingTimeSearchRange; //!< The search space in time for max tracklet similarity.
	float m_minSpeed4AngleChecking; //!< Min. speed (pixels/frame) to check the angle difference using "m_maxAngle4TrackletMatching".
	float m_maxIou4AngleChecking; //!< Max. IOU to compare angles for tracklet matching. If IOU is higher, angle checking is not performed.

	// [Re-association: ReID similarity]
	int m_reidExtractionInterval; //!< Frame interval to extract reid features per target.

	// [Post-processing: tracklet validity]
	float m_minMatchRatio4ValidTrackletDetermination; //!< Tracklet will be considered to be valid if the ratio of valid matching is greater than or equal to this threshold.

	// [Post-processing: duplicate tracklet removal]
	float m_minIou4DuplicateTrackletRemoval; //!< Min iou between bboxes from two tracklets for duplication tracklet detection.
	float m_minInclusionRatio4DuplicateTrackletRemoval; //!< Min inclusion ratio for Duplicate Tracklet Removal. 0.8 means 80%.

	// [Post-processing: vanishing tracklet reconstruction]
	bool m_enableVanishingTrackletReconstruction; //!< Enable Vanishing Tracklet reconstruction.
	float m_visibilityThreshold4VanishingTrackletDetection; //!< If the visibility of the last projected tracklet gets lower than this, this tracklet will be considered as a Vanishing Tracklet.
	float m_minVisibility4VanishingTrackletReconstruction; //!< The projected object bboxes that have higher visibility than this would be appended to the Vanishing Tracklet.

	// [Trajectory dump]
	bool m_dumpTrajectoryIntoFile; //!< Dump the archived trajectories into a file for accuracy evaluation.
	bool m_dumpDetectorBboxes; //!< Dump the detector bounding boxes instead of recovered bounding boxes.
	bool m_dumpTrajectoryInactive; //!< Dump the inactive points in the trajectories.
	int m_maxInactiveDumpFrames; //!< When dumpTrajectoryInactive is true, cap consecutive inactive frames dumped per gap (0 = no cap / disabled).
	bool m_dumpTrajectoryTentative; //!< Dump the tentative points in the trajectories.
	bool m_dumpTrajectoryQuasiActive; //!< Dump the quasi-active points in the trajectories.
	float m_minVisibility4TrajDump; //!< Minimum visibility to dump a trajectory.
	std::string m_trajectoryDumpFileName; //!< Prefix to the filename where the archived trajectories are dumped to.

	// [Runtime parameters: frame info]
	NvMotDataTypes::Rect2f m_originalSrcImageBox; //!< The original video source image box.
	NvMotDataTypes::Rect2f m_srcImageBox; //!< The tracker source image box.
	Eigen::Vector2f m_scale2OriginalSourceCoordinates; //!< The scale factor that converts to the width/height of the original source frame.
	//!< This is a runtime parameter and is not read from the config.

	// [Offline trajectory management]
	TrajectoryManagerOfflineParams m_offlineParams; //!< Offline trajectory management.

	// [Archive re-association world-distance fallback]
	float m_maxWorldDist4ArchiveReAssoc; //!< Max world-plane distance (meters) for archive re-assoc fallback when IoU=0.
	//!< Meters. 0.0=disabled.
	float m_maxWorldDist4ArchiveReAssocCS; //!< Meters. 0.0=disabled (use m_maxWorldDist4ArchiveReAssoc).
	float m_minBboxSizeSimilarity4ReAssocCS; //!< CS bboxSim gate: rejects cross-camera archive candidates with bboxSim < threshold.
	//!< CS bboxSim gate (0.0=disabled).
	float m_maxWorldDist4ArchiveReAssocWS; //!< Wider within-stream world-distance gate in meters; 0 uses m_maxWorldDist4ArchiveReAssoc.
	//!< Meters. 0.0=disabled (use m_maxWorldDist4ArchiveReAssoc).
	bool m_isInit = false; //!< Status if Params have been initialized.
	//!< Prevents changing static params in dynamic updates.
};

/** @brief Class to generate unique IDs. */
class UIDGenerator
{
public:
	/** @brief Initialize the generator with a random upper-ID prefix. */
	UIDGenerator()
	{
		reset();
	};

	/** @brief Return the next ID without consuming it. */
	uint64_t peekNextId() const
	{
		return prevId_;
	}

	/** @brief Generate and consume the next unique ID. */
	uint64_t generateId()
	{
		return prevId_++;
	}

	/** @brief Reset the generator with a new random upper-ID prefix. */
	void reset()
	{
		std::uint_least32_t seed;

#ifndef _WIN32
		uint32_t bytes = syscall(SYS_getrandom, &seed, sizeof(seed), 0);
#else
		uint32_t bytes = 0;
#endif

		/// Falls back to a Mersenne Twister seed when the system generator does not fill the seed.
		if (bytes != sizeof(seed))
		{
			/// Get a random seed from the OS entropy device.
			std::random_device rd;
			seed = rd();
		}

		/// Use the 64-bit Mersenne Twister 19937 generator.
		std::mt19937_64 eng(seed);

		/// Define the distribution, by default it goes from 0 to MAX(unsigned long long).
		std::uniform_int_distribution<uint64_t> distr;

		/// Puts a random value in the upper 32 bits and leaves the lower 32 bits to increment.
		prevId_ = distr(eng) << 32;
	}

private:
	std::atomic<uint64_t> prevId_; //!< Stores the next unique ID.
};

typedef std::map<uint64_t, Trajectory4Archive> Trajectory4ArchiveList;
typedef std::map<std::shared_ptr<NvTrackedObject>, std::pair<std::list<TargetInfo>, float>> ProjectedTrajectoryList;
typedef std::map<std::shared_ptr<NvTrackedObject>, uint> TrackedObjectsReidIndList;

/** @brief Manage target IDs, archived trajectories, and trajectory reassociation. */
class NvTrajectoryManager
{
public:
	/**
	 * @brief Construct a trajectory manager from configuration.
	 * @param[in] configFilePath Path to the configuration file.
	 */
	NvTrajectoryManager(const std::string &configFilePath);

	/** @brief Report statistics and destroy the trajectory manager. */
	virtual ~NvTrajectoryManager();

	/**
	 * @brief Store the trajectories of the terminated objects up to a certain time.
	 * @param[in] target Terminated object whose trajectory is stored.
	 */
	void archiveTrajectory(const std::shared_ptr<NvTrackedObject> target);

	/**
	 * @brief Assign target IDs and set targetIDTimeStamp.
	 * @param[in] targetList Tracked objects that may receive target IDs.
	 * @param[out] toBeTerminated Map from a terminated target to its peer.
	 * @param[in] timeMeta Timing metadata from association.
	 */
	void assignTargetIDs(
		const NvTrackedObjectList &targetList,
		std::map<std::shared_ptr<NvTrackedObject>, std::shared_ptr<NvTrackedObject> > &toBeTerminated,
		const NvMOTTimeMeta &timeMeta
	);

	/**
	 * @brief Remove expired archived trajectories for all input streams.
	 * @param[in] inputMeta Current input metadata by stream.
	 */
	void cleanUpArchivedTrajectories(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta);

	/**
	 * @brief Remove expired archived trajectories for one stream.
	 * @param[in] streamId Source stream identifier.
	 * @param[in] currFrameNum Current frame number.
	 * @param[in] doPurge Whether to remove every archived trajectory.
	 */
	void cleanUpArchivedTrajectories_perStream(
		const NvMOTStreamId streamId, const uint32_t currFrameNum, const bool doPurge = false
	);

	/** @brief Reset the unique-ID generator with a new random prefix. */
	void resetIDs() { m_uidGenerator.reset(); }

	/** @brief Purge all archived trajectories. */
	void purgeArchivedTrajectories();

	/**
	 * @brief Generate target detections that require ReID extraction for one stream.
	 * @param[in] streamId Source stream identifier.
	 * @param[in] targetList Tracked targets to inspect.
	 */
	virtual std::deque<DetectedObjInfo> generateTrackedObjects4ReidPerStream(
		const NvMOTStreamId streamId, const NvTrackedObjectList &targetList
	);

	/**
	 * @brief Update per-stream trajectory state when streams are added or removed.
	 * @param[in] streamIdsToAdd Stream identifiers to add.
	 * @param[in] streamIdsToDel Stream identifiers to remove.
	 */
	void updateStreamIds(
		const std::vector<NvMOTStreamId>& streamIdsToAdd={}, const std::vector<NvMOTStreamId>& streamIdsToDel={}
	);

	/**
	 * @brief Update dynamic parameters during runtime.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void updateParams(const YAML::Node &fn);

	/**
	 * @brief Set source-frame geometry used for trajectory output.
	 * @param[in] originalSrcImageBox Original source image bounds.
	 * @param[in] srcImageBox Tracker source image bounds.
	 */
	void setSrcFrameInfo(const NvMotDataTypes::Rect2f originalSrcImageBox, const NvMotDataTypes::Rect2f srcImageBox);

	/**
	 * @brief Set the multi-view associators.
	 * @param[in] multiViewAssociators Per-stream multi-view associators.
	 */
	void setMultiViewAssociators(std::map<NvMOTStreamId, std::shared_ptr<IMultiViewAssociator>> multiViewAssociators);

	/**
	 * @brief Return an archived trajectory by stream and target ID.
	 * @param[in] videoStreamId Source stream identifier.
	 * @param[in] targetId Target ID.
	 */
	const Trajectory4Archive &getArchivedTrajectory(const NvMOTStreamId videoStreamId, const uint64_t targetId)
	{
		return m_trajectoryArchive.at(videoStreamId)->at(targetId);
	}

	/** @brief Return a copy of trajectory-manager parameters. */
	const NvTrackerParamsTrajectoryManager getParams() { return m_params; }

	/** @brief Return the minimum visibility required for trajectory dumping. */
	const float getMinVisibility4TrajDump() { return m_params.m_minVisibility4TrajDump; }

	/** @brief Return whether detector bounding boxes are dumped. */
	const bool getDumpDetectorBboxes() { return m_params.m_dumpDetectorBboxes; }
protected:

	NvTrackerParamsTrajectoryManager m_params; //!< Config params for target management.
	const std::string m_configFilePath; //!< Config file path.

	// [Trajectory info (written in the main thread, read in worker threads)]
	std::map<NvMOTStreamId, std::shared_ptr<Trajectory4ArchiveList>> m_trajectoryArchive; //!< Archive of past trajectories up to a configured age, modified on the main thread and read by workers.
	std::map<NvMOTStreamId, std::shared_ptr<ProjectedTrajectoryList>> m_projectedTrajectories; //!< Temporary buffer of the projected trajectories to be re-associated.
	std::map<NvMOTStreamId, std::shared_ptr<TrackedObjectsReidIndList>> m_trackedObjectsReidInd; //!< Target's index in reid batch.
	std::map<NvMOTStreamId, bool> m_isFileTrajectoryDumpCreated; //!< Post-processing: dump trajectories into a file: only accessed in main thread.
	std::map<NvMOTStreamId, std::ofstream> m_fileTrajectoryDump; //!< Per-stream trajectory dump file, accessed only on the main thread.

	// [Re-association: per-stream projector]
	std::map<NvMOTStreamId, std::shared_ptr<IEstimator2>> m_trajProjector; //!< Per-stream state estimator used to project trajectories for re-association.
	std::map<NvMOTStreamId, std::shared_ptr<IMultiViewAssociator>> m_multiViewAssociators; //!< Multi-view associators.

	// [Peer-archive deduplication]
	std::map<uint64_t, NvMOTFrameNum> m_usedPeerTargetIds; //!< Claimed peer target id mapped to the frame of the claim.
	//!< The claim blocks reuse for m_maxAge4TrackletArchiving frames.

	// [ID generator]
	UIDGenerator m_uidGenerator; //!< ID generator.

	// [Offline processing]
	bool m_isOffline = false; //!< Boolean to indicate if the trajectory manager is used for offline processing.

	/** @brief Re-association pipeline counters: dumped to stderr at destructor time. */
	struct ReAssocStats
	{
		std::atomic<uint64_t> db_add_calls{0}; //!< Times add2ProjectedTrajectoryDB was called (shadow tracks entering DB).
		std::atomic<uint64_t> find_calls{0}; //!< Times findMatch4ReAssoc was called (tracks needing an ID).
		std::atomic<uint64_t> db_empty_at_find{0}; //!< Of those, how many had an empty projTrajList (no shadow tracks in DB).
		std::atomic<uint64_t> candidates_seen{0}; //!< Total peer candidates iterated across all findMatch4ReAssoc calls.
		std::atomic<uint64_t> temporal_fail{0}; //!< Candidates failing temporal containment check (peer window doesn't span new track).
		std::atomic<uint64_t> zero_tracklet_score{0}; //!< Candidates that passed temporal check but got trackletScore == 0 (size gate or DTW all-zero).
		std::atomic<uint64_t> below_thresh{0}; //!< Calls where best score > 0 but < minMatchingScore4Overall.
		std::atomic<uint64_t> reassoc_success{0}; //!< Successful trajectory re-associations (tryReAssoc returned true).
		std::atomic<uint64_t> archive_reassoc{0}; //!< Subset of reassoc_success from archive (terminated-track) search.
		std::atomic<uint64_t> peer_reassoc_success{0}; //!< Successful peer/MVA re-associations.
		std::atomic<uint64_t> new_id{0}; //!< New IDs assigned (no re-assoc found).
		std::atomic<float> best_failing_score{0.0f}; //!< Highest tracklet score that failed minMatchingScore4Overall threshold.
		std::atomic<uint64_t> archive_candidates{0}; //!< Archive entries attempted (non-reported, non-empty, gap > 0) in findArchiveMatch4ReAssoc.
		std::atomic<uint64_t> archive_zero_score{0}; //!< Archive candidates that passed temporal check but got trackletScore == 0.
		std::atomic<uint64_t> proj_near_zero_vel{0}; //!< Live shadow projections where |imgVx|<0.5 AND |imgVy|<0.5 at prep-window end.
		std::atomic<uint64_t> zs_disp_0_50{0}; //!< Zero_score candidates with min peer-center distance < 50px.
		std::atomic<uint64_t> zs_disp_50_100{0}; //!< Zero_score candidates with min peer-center distance in [50, 100)px.
		std::atomic<uint64_t> zs_disp_100_200{0}; //!< Zero_score candidates with min peer-center distance in [100, 200)px.
		std::atomic<uint64_t> zs_disp_200_plus{0}; //!< Zero_score candidates with min peer-center distance ≥ 200px.
		std::atomic<uint64_t> reid_ind_found{0}; //!< Pairs where the new target is in m_trackedObjectsReidInd.
		std::atomic<uint64_t> reid_score_valid{0}; //!< Pairs where reidMatchingScoreIsValid == true.
		std::atomic<uint64_t> reid_rescue_possible{0}; //!< Pairs where reidCanRescue == true.
		std::atomic<uint64_t> reid_zs_lt02{0}; //!< Zero-tracklet pairs: reidScore < 0.2.
		std::atomic<uint64_t> reid_zs_02_04{0}; //!< Zero-tracklet pairs: reidScore [0.2, 0.4).
		std::atomic<uint64_t> reid_zs_04_05{0}; //!< Zero-tracklet pairs: reidScore [0.4, 0.5).
		std::atomic<uint64_t> reid_zs_ge05{0}; //!< Zero-tracklet pairs: reidScore >= 0.5 (rescue threshold).
		std::atomic<uint64_t> zs_bsim_lt50{0}; //!< Zero-score live-DB candidates with bboxSim < 0.5.
		std::atomic<uint64_t> zs_bsim_50_70{0}; //!< Zero-score live-DB cands with bboxSim in [0.5, 0.7).
		std::atomic<uint64_t> zs_bsim_ge70{0}; //!< Zero-score live-DB cands with bboxSim >= 0.7.
		std::atomic<uint64_t> az_bsim_lt50{0}; //!< Zero-score archive candidates with bboxSim < 0.5.
		std::atomic<uint64_t> az_bsim_50_70{0}; //!< Zero-score archive cands with bboxSim in [0.5, 0.7).
		std::atomic<uint64_t> az_bsim_ge70{0}; //!< Zero-score archive cands with bboxSim >= 0.7.
		std::atomic<uint64_t> az_score_hist[10] = { //!< Non-zero archive scores, bucket i = floor(score * 10).
			{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
		};
		std::atomic<float> az_score_best{0.f}; //!< Running max archive candidate score across all findArchiveMatch4ReAssoc calls.
		std::atomic<uint64_t> archive_world_fallback{0}; //!< World-distance fallback fires in findArchiveMatch4ReAssoc.
		std::atomic<uint64_t> live_world_fallback{0}; //!< World-distance fallback fires in findMatch4ReAssoc.
		std::atomic<uint64_t> az_worlddist_lt1{0}; //!< Eligible archive candidates with world distance < 1.0 m.
		std::atomic<uint64_t> az_worlddist_1_2{0}; //!< Eligible arch cands with world dist in [1, 2)m.
		std::atomic<uint64_t> az_worlddist_2_3{0}; //!< Eligible arch cands with world dist in [2, 3)m.
		std::atomic<uint64_t> az_worlddist_3_4{0}; //!< Eligible arch cands with world dist in [3, 4)m.
		std::atomic<uint64_t> az_worlddist_4_5{0}; //!< Eligible arch cands with world dist in [4, 5)m.
		std::atomic<uint64_t> az_worlddist_ge5{0}; //!< Eligible arch cands with world dist >= 5.0m.
		std::atomic<uint64_t> lz_worlddist_lt1{0}; //!< Eligible live-DB candidates with world distance < 1.0 m.
		std::atomic<uint64_t> lz_worlddist_1_2{0}; //!< Eligible live-DB cands with world dist in [1, 2)m.
		std::atomic<uint64_t> lz_worlddist_2_3{0}; //!< Eligible live-DB cands with world dist in [2, 3)m.
		std::atomic<uint64_t> lz_worlddist_3_4{0}; //!< Eligible live-DB cands with world dist in [3, 4)m.
		std::atomic<uint64_t> lz_worlddist_4_5{0}; //!< Eligible live-DB cands with world dist in [4, 5)m.
		std::atomic<uint64_t> lz_worlddist_ge5{0}; //!< Eligible live-DB cands with world dist >= 5.0m.
		std::atomic<uint64_t> azbt_worlddist_lt1{0}; //!< Below-threshold archive candidates with world distance < 1.0 m.
		std::atomic<uint64_t> azbt_worlddist_1_2{0}; //!< Below-thresh archive cands with world dist in [1, 2)m.
		std::atomic<uint64_t> azbt_worlddist_2_3{0}; //!< Below-thresh archive cands with world dist in [2, 3)m.
		std::atomic<uint64_t> azbt_worlddist_3_4{0}; //!< Below-thresh archive cands with world dist in [3, 4)m.
		std::atomic<uint64_t> azbt_worlddist_4_5{0}; //!< Below-thresh archive cands with world dist in [4, 5)m.
		std::atomic<uint64_t> azbt_worlddist_ge5{0}; //!< Below-thresh archive cands with world dist >= 5.0m.
		std::atomic<uint64_t> azbt_worlddist_nopos{0}; //!< Below-thresh archive cands with no valid world position.
		std::atomic<uint64_t> az_calls_total{0}; //!< Total findArchiveMatch4ReAssoc invocations.
		std::atomic<uint64_t> az_calls_no_archive{0}; //!< Archive map missing or empty at entry.
		std::atomic<uint64_t> az_calls_loop_empty{0}; //!< Archive had entries but all pre-filtered.
		std::atomic<uint64_t> az_calls_cand_failed{0}; //!< Had candidates but best score < threshold.
		std::atomic<uint64_t> inact_dumped{0}; //!< Inactive frames actually dumped.
		std::atomic<uint64_t> inact_frames_21_30{0}; //!< Cap-skipped inactive frames at gap positions 21-30.
		std::atomic<uint64_t> inact_frames_31_40{0}; //!< Cap-skipped inactive frames at gap positions 31-40.
		std::atomic<uint64_t> inact_frames_41plus{0}; //!< Cap-skipped inactive frames at gap positions 41+.
		std::atomic<uint64_t> ws_gap_lt50{0}; //!< Within-stream: frameGap < 50.
		std::atomic<uint64_t> ws_gap_50_100{0}; //!< Within-stream: frameGap [50, 100).
		std::atomic<uint64_t> ws_gap_100_200{0}; //!< Within-stream: frameGap [100, 200).
		std::atomic<uint64_t> ws_gap_ge200{0}; //!< Within-stream: frameGap >= 200.
		std::atomic<uint64_t> ws_wd_15_2{0}; //!< Within-stream FAILED worldDist gate: [1.5, 2.0)m.
		std::atomic<uint64_t> ws_wd_2_3{0}; //!< Within-stream FAILED worldDist gate: [2.0, 3.0)m.
		std::atomic<uint64_t> ws_wd_3_5{0}; //!< Within-stream FAILED worldDist gate: [3.0, 5.0)m.
		std::atomic<uint64_t> ws_wd_ge5{0}; //!< Within-stream FAILED worldDist gate: >= 5.0m.
		std::atomic<uint64_t> ws_passed{0}; //!< Within-stream candidates PASSING distance gate.
		std::atomic<uint64_t> ws_bsim_lt10{0}; //!< Within-stream worldDist-pass: bboxSim < 0.10.
		std::atomic<uint64_t> ws_bsim_10_25{0}; //!< Within-stream worldDist-pass: 0.10 <= bboxSim < 0.25.
		std::atomic<uint64_t> ws_bsim_25_50{0}; //!< Within-stream worldDist-pass: 0.25 <= bboxSim < 0.50.
		std::atomic<uint64_t> ws_bsim_ge50{0}; //!< Within-stream worldDist-pass: bboxSim >= 0.50 (passes gate).
	} m_stats; //!< Stores trajectory-reassociation counters.

	/**
	 * @brief Try to perform the re-assoc with the targets in the temporary DB that stores projected trajectories.
	 * @param[in] target Target to reassociate.
	 * @param[out] toBeFused Map from a terminated target to its peer.
	 * @return True when reassociation succeeds.
	 */
	bool tryReAssoc(
		const std::shared_ptr<NvTrackedObject> target,
		std::map<std::shared_ptr<NvTrackedObject>, std::shared_ptr<NvTrackedObject>> &toBeFused
	);

	/**
	 * @brief Trim the tracklet by removing the last un-validated trajectory points.
	 * @param[in] targetId ID of the tracklet.
	 * @param[in] streamId Stream that owns the tracklet.
	 * @param[in,out] tracklet Tracklet trimmed in place by removing unvalidated tail points.
	 * @return True when the tracklet is valid to archive.
	 */
	bool processTrajectory(const uint64_t targetId, const NvMOTStreamId streamId, std::list<TargetInfo> &tracklet);

	/**
	 * @brief Trim the tracklet by removing the last un-validated trajectory points and return the projected trajectory points only.
	 * @param[in] targetId ID of the tracklet.
	 * @param[in] streamId Stream that owns the tracklet.
	 * @param[in] tracklet Tracklet to project.
	 * @param[in] overrideShadowAge Shadow age to use when non-negative; a negative value counts INACTIVE frames at the tail.
	 * @return Projected trajectory points.
	 */
	std::list<TargetInfo> createProjectedTrajectories(
		const uint64_t targetId, const NvMOTStreamId streamId,
		const std::list<TargetInfo> &tracklet, int overrideShadowAge = -1
	);

	/**
	 * @brief Add the projected trajectory to the temporary DB so that it can be re-associated.
	 * @param[in] target Shadow track whose projected trajectory is stored.
	 */
	void add2ProjectedTrajectoryDB(std::shared_ptr<NvTrackedObject> target);

	/**
	 * @brief Remove from the projected trajectory DB.
	 * @param[in] target Track to remove from the projected trajectory database.
	 */
	void removeFromProjectedTrajectoryDB(std::shared_ptr<NvTrackedObject> target);

	/**
	 * @brief Get the trimmed new trajectory for reassociation with peer trajectory.
	 * @param[in] newTraj Trajectory to be trimmed.
	 * @param[in] peerTraj Peer trajectory to reassociate against.
	 * @param[in] lastTrackedFrameNum_peer Last tracked frame number of the peer.
	 * @return The trimmed new trajectory.
	 */
	virtual std::list<TargetInfo> getTrimmedTraj4ReAssoc(
		const std::list<TargetInfo> &newTraj, const std::list<TargetInfo> &peerTraj,
		const NvMOTFrameNum lastTrackedFrameNum_peer
	);

	/**
	 * @brief Find the best live projected-trajectory match for a target.
	 * @param[in] target Target to reassociate.
	 * @param[out] bestMatchPeer Best matching peer target.
	 * @return True when a match meets the reassociation threshold.
	 */
	bool findMatch4ReAssoc(
		const std::shared_ptr<NvTrackedObject> target, std::shared_ptr<NvTrackedObject> &bestMatchPeer
	);

	/**
	 * @brief Fallback re-association against terminated tracks in m_trajectoryArchive.
	 * @param[in] newTarget Newly appearing track needing an ID.
	 * @return True if a match above minMatchingScore4Overall was found.
	 */
	bool findArchiveMatch4ReAssoc(const std::shared_ptr<NvTrackedObject> newTarget);

	/**
	 * @brief Perform archive-only world-distance and frame-gap re-association.
	 *
	 * Searches m_trajectoryArchive for terminated tracks within
	 * maxFrameGap4WorldDistReAssoc whose last valid world foot point is within
	 * maxWorldDist4ArchiveReAssoc of the new track and whose bbox similarity is
	 * at least 0.5.
	 *
	 * @param[in] newTarget Newly appearing track needing an ID.
	 * @return True if a match above minMatchingScore4Overall was found.
	 */
	bool try3DReAssoc(const std::shared_ptr<NvTrackedObject> newTarget);

	/**
	 * @brief Calculate bounding-box size similarity between two trajectories.
	 * @param[in] traj First trajectory.
	 * @param[in] peerTraj Peer trajectory.
	 * @return Average bounding-box size similarity.
	 */
	virtual float calculateSizeSimilarityScore(
		const std::list<TargetInfo> &traj, const std::list<TargetInfo> &peerTraj
	);

	/**
	 * @brief Calculate tracklet, direction, speed, and size similarity metrics.
	 * @param[in] traj First trajectory.
	 * @param[in] peerTraj Peer trajectory.
	 * @param[in] peerTrajCov Peer trajectory covariance.
	 * @return Tuple of tracklet score, frame difference, angle, speed similarity, speed unit magnitude, and size similarity.
	 */
	const std::tuple<float, int, float, float, float, float> calculateSimilarityBtwTracklets(
		const std::list<TargetInfo> &traj, const std::list<TargetInfo> &peerTraj, const float peerTrajCov
	);

	/**
	 * @brief Calculate quasi-DTW association metrics between two trajectories.
	 * @param[in] traj First trajectory.
	 * @param[in] peerTraj Peer trajectory.
	 * @param[in] peerTrajCov Peer trajectory covariance.
	 * @return Tuple of average IOU, frame difference, angle, speed similarity, and speed unit magnitude.
	 */
	const std::tuple<float, int, float, float, float> calculateAssocScore4QuasiDTW(
		const std::list<TargetInfo> &traj, const std::list<TargetInfo> &peerTraj, const float peerTrajCov
	);

	/**
	 * @brief Compute the angle between the two tracklets direction [degree].
	 * @param[in] vec1 Direction vector of the first tracklet.
	 * @param[in] vec2 Direction vector of the second tracklet.
	 * @return Angle between the two directions in degrees.
	 */
	const float computeAngleInDegree(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2);

	/**
	 * @brief Compute the speed similarity between the two tracklets.
	 * @param[in] vec1 Velocity vector of the first tracklet.
	 * @param[in] vec2 Velocity vector of the second tracklet.
	 * @return Speed similarity, where a higher value means a closer match.
	 */
	const float computeSpeedSimilarity(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2);

	/**
	 * @brief Detect if a tracklet is in motion, that is, if its speed is at least the threshold.
	 * @param[in] vec Velocity vector of the tracklet.
	 * @param[in] threshold Minimum speed treated as motion.
	 * @return True when the vector speed is at least the threshold.
	 */
	const bool detectMotion(const Eigen::Vector2f &vec, float threshold);

	/**
	 * @brief Fuse two tracklets.
	 * @param[in] targetToBeKept Track that is kept and receives the fused trajectory.
	 * @param[in] targetToBeFused Track whose trajectory is copied into the kept track.
	 */
	void fuseTracklets(
		std::shared_ptr<NvTrackedObject> targetToBeKept, std::shared_ptr<NvTrackedObject> targetToBeFused
	);

	/**
	 * @brief Check if the tracklet is a duplicate to another existing tracklet.
	 * @param[in] targetId ID of the tracklet to check.
	 * @param[in,out] tracklets Archived tracklets for the stream; a duplicate may be fused into its peer.
	 * @param[in,out] peersToBeDeleted Peer IDs already marked for deletion; a duplicate peer is inserted here.
	 * @return True when this tracklet is a duplicate of a peer.
	 */
	bool checkDuplicateTracklet(
		const uint64_t targetId, std::map<uint64_t, Trajectory4Archive> &tracklets, std::set<uint64_t> &peersToBeDeleted
	);

	/**
	 * @brief Calculate the number of duplicate bboxes between the two tracklets.
	 * @param[in] tracklet Tracklet being compared.
	 * @param[in] peerTracklet Peer tracklet being compared.
	 * @return Number of duplicate boxes, or -1 when a tracklet is empty or the tracklets do not overlap in time.
	 */
	int calculateTrackletInclusionScore(
		const std::list<TargetInfo> &tracklet, const std::list<TargetInfo> &peerTracklet
	);

	/**
	 * @brief Fuse the peer's tracklet into the target's.
	 * @param[in] targetId ID of the tracklet that is kept.
	 * @param[in] peerId ID of the peer tracklet that is merged in.
	 * @param[in,out] tracklets Archived tracklets; the kept tracklet is replaced by the fused result.
	 */
	void fuseArchivedTracklets(
		const uint64_t targetId, const uint64_t peerId, std::map<uint64_t, Trajectory4Archive> &tracklets
	);

	/**
	 * @brief Dump the trajectory data into a file in a specified format.
	 * @param[in] streamId Stream that owns the trajectory.
	 * @param[in] targetId ID of the trajectory to dump.
	 * @param[in] traj Archived trajectory to write.
	 */
	void dumpTrajectoryIntoFile(const NvMOTStreamId streamId, const TargetId targetId, const Trajectory4Archive &traj);

	/** @brief Generate a thread-safe target ID according to the configured ID mode. */
	const uint64_t generateNewTargetID();

	/** @brief Return the next target ID without consuming it. */
	const uint64_t peekNewTargetID() const;

	#ifdef VIZ_DEBUG
		friend class TrackerVizOutput;
		/**
		 * @brief Export trajectory state for visualization.
		 * @param[in] curTarget Current target to visualize.
		 * @param[in,out] toBeTerminated Target-termination map used by visualization.
		 */
		virtual void dumpVizTrajectories(
			const std::shared_ptr<NvTrackedObject> curTarget,
			std::map<std::shared_ptr<NvTrackedObject>, std::shared_ptr<NvTrackedObject> > &toBeTerminated
		)
		{
			;
		}
	#endif
};

#endif // NVMULTIOBJECTTRACKERLIB_NVTRAJECTORYMANAGER_HPP_
