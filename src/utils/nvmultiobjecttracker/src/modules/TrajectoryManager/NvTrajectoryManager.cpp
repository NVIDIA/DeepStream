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

#include "modules/utility/cv_functions.hpp"
#include "modules/TrajectoryManager/NvTrajectoryManager.hpp"
#include "modules/TrajectoryManager/TrajectoryOperations.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct a trajectory manager from configuration.
 * @param[in] configFilePath Path to the configuration file.
 */
NvTrajectoryManager::NvTrajectoryManager(const std::string &configFilePath) : m_configFilePath(configFilePath)
{
	m_params = NvTrackerParamsTrajectoryManager(configFilePath, "TrajectoryManagement");

	/// Reset the ID.
	m_uidGenerator.reset();

	if(m_params.m_debugVerboseLevel > 1)
		cout << "[NvTrajectoryManager] Instantiated" << endl;
}

/** @brief Report statistics and destroy the trajectory manager. */
NvTrajectoryManager::~NvTrajectoryManager()
{

	/// Check overlapped/duplicate tracklet, discard/fuse if there are, and delete the all archived trajectories.
	purgeArchivedTrajectories();

	/// Dump re-association pipeline counters to stdout (captured in deepstream_app.log).
	if (m_params.m_debugVerboseLevel >= 1) {
		printf(
			"[NvTrajectoryManager] db_add=%llu find_calls=%llu db_empty=%llu candidates=%llu" " temporal_fail=%llu zero_score=%llu below_thresh=%llu" " reassoc=%llu archive_reassoc=%llu peer_reassoc=%llu new_id=%llu best_fail_score=%.4f" " arch_cands=%llu arch_zero=%llu proj_static=%llu" " zs_disp_0_50=%llu zs_disp_50_100=%llu zs_disp_100_200=%llu zs_disp_200+=%llu" " zs_bsim_lt50=%llu zs_bsim_50_70=%llu zs_bsim_ge70=%llu" " az_bsim_lt50=%llu az_bsim_50_70=%llu az_bsim_ge70=%llu" " inact_dumped=%llu inact_frames_21_30=%llu inact_frames_31_40=%llu inact_frames_41plus=%llu" " az_score_hist=[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu] az_score_best=%.4f" " arch_world_fallback=%llu" " az_worlddist=[%llu,%llu,%llu,%llu,%llu,%llu]" " live_world_fallback=%llu" " lz_worlddist=[%llu,%llu,%llu,%llu,%llu,%llu]" " azbt_worlddist=[%llu,%llu,%llu,%llu,%llu,%llu] azbt_nopos=%llu" " az_calls_total=%llu az_calls_no_arch=%llu az_calls_loop_empty=%llu az_calls_cand_failed=%llu" " reid_ind_found=%llu reid_score_valid=%llu reid_rescue_possible=%llu reid_zs=[%llu,%llu,%llu,%llu]" " ws_gap=[%llu,%llu,%llu,%llu] ws_wd_fail=[%llu,%llu,%llu,%llu] ws_passed=%llu" " ws_bsim=[%llu,%llu,%llu,%llu]\n",
			(unsigned long long)m_stats.db_add_calls, (unsigned long long)m_stats.find_calls,
			(unsigned long long)m_stats.db_empty_at_find, (unsigned long long)m_stats.candidates_seen,
			(unsigned long long)m_stats.temporal_fail, (unsigned long long)m_stats.zero_tracklet_score,
			(unsigned long long)m_stats.below_thresh, (unsigned long long)m_stats.reassoc_success,
			(unsigned long long)m_stats.archive_reassoc, (unsigned long long)m_stats.peer_reassoc_success,
			(unsigned long long)m_stats.new_id, m_stats.best_failing_score.load(std::memory_order_relaxed),
			(unsigned long long)m_stats.archive_candidates, (unsigned long long)m_stats.archive_zero_score,
			(unsigned long long)m_stats.proj_near_zero_vel, (unsigned long long)m_stats.zs_disp_0_50,
			(unsigned long long)m_stats.zs_disp_50_100, (unsigned long long)m_stats.zs_disp_100_200,
			(unsigned long long)m_stats.zs_disp_200_plus, (unsigned long long)m_stats.zs_bsim_lt50,
			(unsigned long long)m_stats.zs_bsim_50_70, (unsigned long long)m_stats.zs_bsim_ge70,
			(unsigned long long)m_stats.az_bsim_lt50, (unsigned long long)m_stats.az_bsim_50_70,
			(unsigned long long)m_stats.az_bsim_ge70, (unsigned long long)m_stats.inact_dumped,
			(unsigned long long)m_stats.inact_frames_21_30, (unsigned long long)m_stats.inact_frames_31_40,
			(unsigned long long)m_stats.inact_frames_41plus, (unsigned long long)m_stats.az_score_hist[0],
			(unsigned long long)m_stats.az_score_hist[1], (unsigned long long)m_stats.az_score_hist[2],
			(unsigned long long)m_stats.az_score_hist[3], (unsigned long long)m_stats.az_score_hist[4],
			(unsigned long long)m_stats.az_score_hist[5], (unsigned long long)m_stats.az_score_hist[6],
			(unsigned long long)m_stats.az_score_hist[7], (unsigned long long)m_stats.az_score_hist[8],
			(unsigned long long)m_stats.az_score_hist[9], m_stats.az_score_best.load(std::memory_order_relaxed),
			(unsigned long long)m_stats.archive_world_fallback, (unsigned long long)m_stats.az_worlddist_lt1,
			(unsigned long long)m_stats.az_worlddist_1_2, (unsigned long long)m_stats.az_worlddist_2_3,
			(unsigned long long)m_stats.az_worlddist_3_4, (unsigned long long)m_stats.az_worlddist_4_5,
			(unsigned long long)m_stats.az_worlddist_ge5, (unsigned long long)m_stats.live_world_fallback,
			(unsigned long long)m_stats.lz_worlddist_lt1, (unsigned long long)m_stats.lz_worlddist_1_2,
			(unsigned long long)m_stats.lz_worlddist_2_3, (unsigned long long)m_stats.lz_worlddist_3_4,
			(unsigned long long)m_stats.lz_worlddist_4_5, (unsigned long long)m_stats.lz_worlddist_ge5,
			(unsigned long long)m_stats.azbt_worlddist_lt1, (unsigned long long)m_stats.azbt_worlddist_1_2,
			(unsigned long long)m_stats.azbt_worlddist_2_3, (unsigned long long)m_stats.azbt_worlddist_3_4,
			(unsigned long long)m_stats.azbt_worlddist_4_5, (unsigned long long)m_stats.azbt_worlddist_ge5,
			(unsigned long long)m_stats.azbt_worlddist_nopos, (unsigned long long)m_stats.az_calls_total,
			(unsigned long long)m_stats.az_calls_no_archive, (unsigned long long)m_stats.az_calls_loop_empty,
			(unsigned long long)m_stats.az_calls_cand_failed, (unsigned long long)m_stats.reid_ind_found,
			(unsigned long long)m_stats.reid_score_valid, (unsigned long long)m_stats.reid_rescue_possible,
			(unsigned long long)m_stats.reid_zs_lt02, (unsigned long long)m_stats.reid_zs_02_04,
			(unsigned long long)m_stats.reid_zs_04_05, (unsigned long long)m_stats.reid_zs_ge05,
			(unsigned long long)m_stats.ws_gap_lt50, (unsigned long long)m_stats.ws_gap_50_100,
			(unsigned long long)m_stats.ws_gap_100_200, (unsigned long long)m_stats.ws_gap_ge200,
			(unsigned long long)m_stats.ws_wd_15_2, (unsigned long long)m_stats.ws_wd_2_3,
			(unsigned long long)m_stats.ws_wd_3_5, (unsigned long long)m_stats.ws_wd_ge5,
			(unsigned long long)m_stats.ws_passed, (unsigned long long)m_stats.ws_bsim_lt10,
			(unsigned long long)m_stats.ws_bsim_10_25, (unsigned long long)m_stats.ws_bsim_25_50,
			(unsigned long long)m_stats.ws_bsim_ge50
		);
	}

	if(m_params.m_debugVerboseLevel > 1)
		cout << "[NvTrajectoryManager] De-initialized" << endl;
}

/**
 * @brief Assign target IDs and target-ID timestamps.
 * @param[in] targetList Tracked objects that may receive target IDs.
 * @param[out] toBeTerminated Map from a terminated target to its peer.
 * @param[in] timeMeta Timing metadata from association.
 */
void NvTrajectoryManager::assignTargetIDs(
	const NvTrackedObjectList &targetList,
	std::map<std::shared_ptr<NvTrackedObject>, std::shared_ptr<NvTrackedObject>> &toBeTerminated,
	const NvMOTTimeMeta &timeMeta
)
{
	if(targetList.size() > 0)
	{
		/// Pass 1 maintains the DB by fully populating and pruning m_projectedTrajectories before any ID-assignment search.
		if(m_params.m_enableReAssoc)
		{
			for(auto &kv : targetList)
			{
				std::shared_ptr<NvTrackedObject> target = kv.second;

				/// Create PROJECTED tracklet points if the shadow tracking age >= probation age.
				if(target->ready4ReAssoc())
				{
					/// Add to projected trajectory DB.
					add2ProjectedTrajectoryDB(target);

					target->resetReady4ReAssoc();
				}

				/// Clear the projected tracklet points of a target that was on ready-to-re-assoc mode and got activated again.
				if(target->getTrackerState() == TRACKER_STATE::ACTIVE || target->getTrackerState() == TRACKER_STATE::QUASIACTIVE)
				{
					/// Remove from the projected trajectory DB.
					removeFromProjectedTrajectoryDB(target);
				}

				/// If this target was in re-assoc pending DB, but now it gets activated and no need for re-assoc, then remove it from the pending DB.
				if(target->needToRemoveFromPendingReAssocDB())
				{
					removeFromProjectedTrajectoryDB(target);

					target->resetNeedToRemoveFromPendingReAssocDB();
				}
			}
		}

		/// Pass 2 assigns IDs now that all shadow tracks are in m_projectedTrajectories.
		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); )
		{
			auto targetIt_next = std::next(targetIt);

			std::shared_ptr<NvTrackedObject> target = targetIt->second;

			/// Assign the target ID.
			if(target->needTargetIDAssigned())
			{
				TimeStamp localTimestamp = Time().getUnixTimestampNanoSeconds();
				target->setTargetIDTimeStamp(localTimestamp);
				if (timeMeta.frameTimeValid) target->setTargetIDTimeStamp(timeMeta.frameTimeStamp + localTimestamp - timeMeta.localTimestamp);
				/// Check if there's any matching tracklet in the DB.
				if(m_params.m_enableReAssoc && tryReAssoc(target, toBeTerminated))
				{
					/// Found a re-assoc match: for live-DB matches, toBeTerminated[target] is the shadow peer (set inside tryReAssoc via fuseTracklets path).
					if(toBeTerminated.count(target) > 0 && toBeTerminated.at(target))
					{
						toBeTerminated[target]->setBecameActive(true);
						toBeTerminated[target]->setAcquiredID(true);
					}
					m_stats.reassoc_success++;
				}
				/// Enable3DReAssoc: archive-only world-distance + frame-gap re-association: independently gated by m_enable3DReAssoc (no dependency on !m_enableReAssoc).
				else if (m_params.m_enable3DReAssoc
					&& try3DReAssoc(target))
				{
					/// ID set inside try3DReAssoc: archive entry marked isReported.
					m_stats.reassoc_success++;
				}
				else if(m_multiViewAssociators.find(target->getVideoStreamId()) != m_multiViewAssociators.end() &&
					m_multiViewAssociators.at(target->getVideoStreamId())->tryPeerReAssoc(target))
				{
					/// PeerReAssoc succeeded, so the target ID is adopted in tryPeerReAssoc and nothing else is needed here.
					if(m_params.m_debugVerboseLevel > 3)
					{
						cout << "[NvTrajectoryManager::assignTargetIDs()] stream ID: [" << (target->getVideoStreamId() >> 32) << "] frameNum[" << target->getTargetInfo().frameNum
								<< "] PeerReAssoc succeeded: [" << target->getTargetID() << "]" << endl;
					}
					m_stats.peer_reassoc_success++;
				}
				else
				{
					/// Assign a new ID because no re-assoc found.
					target->setTargetID(generateNewTargetID());
					m_stats.new_id++;

					if(m_params.m_debugVerboseLevel > 3)
					{
						cout << "[NvTrajectoryManager::assignTargetIDs()] stream ID: [" << (target->getVideoStreamId() >> 32) << "] frameNum[" << target->getTargetInfo().frameNum
								<< "] A New UniqueID is assigned: [" << target->getTargetID() << "]" << endl;
					}
					if (timeMeta.frameTimeValid) target->setTargetIDTimeStamp(target->getTargetIDTimeStamp() + target->getTargetID());
				}
				target->setAcquiredID(true);
				target->resetNeedTargetIDAssigned();
				#if defined(VIZ_DEBUG)
					dumpVizTrajectories(target, toBeTerminated);
				#endif
			}

			targetIt = targetIt_next;
		}
	}
}

/**
 * @brief Try to reassociate a target using live and archived trajectories.
 * @param[in] newTarget Target to reassociate.
 * @param[out] toBeFused Map from a terminated target to its peer.
 * @return True when reassociation succeeds.
 */
bool NvTrajectoryManager::tryReAssoc(
	const std::shared_ptr<NvTrackedObject> newTarget,
	std::map<std::shared_ptr<NvTrackedObject>, std::shared_ptr<NvTrackedObject>> &toBeFused
)
{
	std::shared_ptr<NvTrackedObject> bestMatchPeer;

	/// Find a tracklet in the DB that can be re-assoced.
	if(findMatch4ReAssoc(newTarget, bestMatchPeer))
	{
		/// Fuse the trajectory and continue with the older one.
		fuseTracklets(bestMatchPeer, newTarget);

		/// Remove the fused one from the tracker list, record its peer.
		toBeFused[newTarget] = bestMatchPeer;

		/// Remove the matched projected tracklet from the DB.
		removeFromProjectedTrajectoryDB(bestMatchPeer);

		return true;
	}

	/// Archive fallback: when the live DB was empty, search terminated tracks in m_trajectoryArchive.
	if(findArchiveMatch4ReAssoc(newTarget))
	{
		return true;
	}

	return false;
}

/**
 * @brief Fuse two tracked-object trajectories.
 * @param[in] targetToBeKept Track that is kept and receives the fused trajectory.
 * @param[in] targetToBeFused Track whose trajectory is copied into the kept track.
 */
void NvTrajectoryManager::fuseTracklets(
	std::shared_ptr<NvTrackedObject> targetToBeKept, std::shared_ptr<NvTrackedObject> targetToBeFused
)
{
	/// Move the tracklet data and states from the new one to the old one.
	targetToBeKept->resetPastFrameData();
	targetToBeKept->adoptTracklet(targetToBeFused->getTrajectoryBuffer());

	/// Update the tracker state.
	targetToBeKept->setTrackerState(TRACKER_STATE::ACTIVE);
	targetToBeKept->resetShadowTrackingAge();
	targetToBeKept->resetReady4ReAssoc();

}

/**
 * @brief Trim a new trajectory to the temporal extent needed for peer reassociation.
 * @param[in] newTraj Trajectory to be trimmed.
 * @param[in] peerTraj Peer trajectory to reassociate against.
 * @param[in] lastTrackedFrameNum_peer Last tracked frame number of the peer.
 * @return The trimmed new trajectory.
 */
std::list<TargetInfo> NvTrajectoryManager::getTrimmedTraj4ReAssoc(
	const std::list<TargetInfo> &newTraj, const std::list<TargetInfo> &peerTraj,
	const NvMOTFrameNum lastTrackedFrameNum_peer
)
{
	/// Default implementation uses the complete input trajectory for reassociation without trimming.
	return std::move(newTraj);
}

/**
 * @brief Find the best live projected-trajectory match for a target.
 * @param[in] newTarget Target to reassociate.
 * @param[out] bestPeer Best matching peer target.
 * @return True when a match meets the reassociation threshold.
 */
bool NvTrajectoryManager::findMatch4ReAssoc(
	const std::shared_ptr<NvTrackedObject> newTarget, std::shared_ptr<NvTrackedObject> &bestPeer
)
{
	m_stats.find_calls++;

	const std::list<TargetInfo> newTraj = newTarget->getTrajectoryBuffer();
	const NvMOTStreamId videoStreamId = newTarget->getVideoStreamId();

	if(m_projectedTrajectories.find(videoStreamId) != m_projectedTrajectories.end() && m_projectedTrajectories.at(videoStreamId))
	{
		std::map<std::shared_ptr<NvTrackedObject>, std::pair<std::list<TargetInfo>, float>> &projTrajList =
			*m_projectedTrajectories.at(videoStreamId);

		/// Track empty DB at time of lookup.
		if(projTrajList.empty()) { m_stats.db_empty_at_find++; }

		std::stringstream msg;
		if(m_params.m_debugVerboseLevel > 3)
		{
			msg << "[NvTrajectoryManager::findMatch4ReAssoc] finding a Re-Assoc match for to-be-TargetID[" << peekNewTargetID() << "]: " << endl;
			msg << "         ";
			for(auto it = projTrajList.begin(); it != projTrajList.end(); ++it )
			{
				std::shared_ptr<NvTrackedObject> peer = it->first;
				msg << DEBUG_UTIL::to_string_fmt(" t[%3d]", peer->getTargetID());
			}
			msg << endl;
			msg << DEBUG_UTIL::to_string_fmt("ID[%3d]:", peekNewTargetID());
		}

		float bestMatchingScore = 0;

		float bestReidMatchingScore = 0;
		int bestReidFrameDiff = 0;

		float bestTrackletMatchingScore = 0;
		int bestTrackletFrameDiff = 0;
		float bestMatchAngle = 0;
		float bestMatchSpeedSimilarity = 0;
		float bestMatchSpeedUnitMagnitude = 0;
		float bestBboxSizeSimilarity = 0;
		std::shared_ptr<NvTrackedObject> bestMatchPeer = nullptr;

		for(auto it = projTrajList.begin(); it != projTrajList.end(); ++it )
		{
			std::shared_ptr<NvTrackedObject> peer = it->first;

			std::list<TargetInfo> &peerTraj = it->second.first;
			const NvMOTFrameNum lastFrameNum_peer = peerTraj.rbegin()->frameNum;
			const NvMOTFrameNum firstFrameNum_peer = peerTraj.begin()->frameNum;
			const NvMOTFrameNum lastTrackedFrameNum_peer = peer->getlastFrameinTraj();

			float peerTrajCov = it->second.second;

			int reidFrameDiff = 0;
			float reidMatchingScore = 0;

			float trackletMatchingScore = 0;
			int trackletFrameDiff = 0;
			float angle = 0;
			float speedSimilarity = 0;
			float speedUnitMagnitude = 0;
			float bboxSizeSimilarity = 0;

			/// Check if the class of the new target matches the class of the peer.
			if(m_params.m_checkClassMatch)
			{
				if(newTarget->getTargetInfo().classId != peer->getTargetInfo().classId)
				{
					if(m_params.m_debugVerboseLevel > 3)
					{
						cout << "[NvTrajectoryManager::findMatch4ReAssoc] Skipping Re-Assoc match since class mismatch: to-be-TargetID[" << peekNewTargetID() << "]: ";
						cout << "with TargetID["  << peer->getTargetID() << "] " << endl;
					}
					continue;
				}
			}

			/// Count candidates that pass class check.
			m_stats.candidates_seen++;

			if(m_params.m_debugVerboseLevel > 3)
			{
				cout << "[NvTrajectoryManager::findMatch4ReAssoc] finding a Re-Assoc match for to-be-TargetID[" << peekNewTargetID() << "]: ";
				cout << "with TargetID["  << peer->getTargetID() << "] " << endl;
			}

			std::list<TargetInfo> traj = getTrimmedTraj4ReAssoc(newTraj, peerTraj, lastTrackedFrameNum_peer);

			const NvMOTFrameNum lastFrameNum = traj.rbegin()->frameNum;
			const NvMOTFrameNum firstFrameNum = traj.begin()->frameNum;

			/// Consider only if the existing projected trajectory has all the points of the new target.
			if(lastFrameNum_peer >= lastFrameNum && firstFrameNum_peer <= firstFrameNum)
			{
				/// Get the matching score for trajectory based re-assoc.
				std::tie(
					trackletMatchingScore, trackletFrameDiff, angle,
					speedSimilarity, speedUnitMagnitude, bboxSizeSimilarity
				) = calculateSimilarityBtwTracklets(traj, peerTraj, peerTrajCov);
				/// If temporal check passed but score is still zero, the size gate or DTW killed it.
				if(trackletMatchingScore == 0.0f)
				{
					m_stats.zero_tracklet_score++;
					/// Displacement histogram: compute min center distance from any new-track frame to the peer's last projected position.
					const float pCx = (peerTraj.rbegin()->tBbox.min()(0) + peerTraj.rbegin()->tBbox.max()(0)) * 0.5f;
					const float pCy = (peerTraj.rbegin()->tBbox.min()(1) + peerTraj.rbegin()->tBbox.max()(1)) * 0.5f;
					float minDist = std::numeric_limits<float>::max();
					for(const auto &tp : traj)
					{
						const float nx = (tp.tBbox.min()(0) + tp.tBbox.max()(0)) * 0.5f;
						const float ny = (tp.tBbox.min()(1) + tp.tBbox.max()(1)) * 0.5f;
						const float d  = std::sqrt((nx - pCx)*(nx - pCx) + (ny - pCy)*(ny - pCy));
						if(d < minDist) minDist = d;
					}
					if     (minDist <  50.0f) m_stats.zs_disp_0_50++;
					else if(minDist < 100.0f) m_stats.zs_disp_50_100++;
					else if(minDist < 200.0f) m_stats.zs_disp_100_200++;
					else                      m_stats.zs_disp_200_plus++;
					/// BboxSizeSimilarity distribution for zero-score live-DB candidates: bboxSizeSimilarity is already computed above (calculateSimilarityBtwTracklets).
					if      (bboxSizeSimilarity < 0.5f) m_stats.zs_bsim_lt50++;
					else if (bboxSizeSimilarity < 0.7f) m_stats.zs_bsim_50_70++;
					else                                m_stats.zs_bsim_ge70++;
					if (m_params.m_maxWorldDist4ArchiveReAssoc > 0.0f) {
						/// The peer trajectory buffer holds the actual detection history, not the projected trajectory.
						const std::list<TargetInfo> peerActualTraj = peer->getTrajectoryBuffer();
						Eigen::Vector2f peerLastWorld(-1.0f, -1.0f);
						for (auto rit = peerActualTraj.rbegin(); rit != peerActualTraj.rend(); ++rit) {
							if (rit->pt_world_feet.x() >= 0.0f) { peerLastWorld = rit->pt_world_feet; break; }
						}
						Eigen::Vector2f newFirstWorld(-1.0f, -1.0f);
						for (const auto &tp : newTraj) {
							if (tp.pt_world_feet.x() >= 0.0f) { newFirstWorld = tp.pt_world_feet; break; }
						}
						if (peerLastWorld.x() >= 0.0f && newFirstWorld.x() >= 0.0f) {
							const float worldDist = (peerLastWorld - newFirstWorld).norm();
							/// Live-DB world-distance histogram covering all zero-score candidates.
							if      (worldDist <  1.0f) m_stats.lz_worlddist_lt1++;
							else if (worldDist <  2.0f) m_stats.lz_worlddist_1_2++;
							else if (worldDist <  3.0f) m_stats.lz_worlddist_2_3++;
							else if (worldDist <  4.0f) m_stats.lz_worlddist_3_4++;
							else if (worldDist <  5.0f) m_stats.lz_worlddist_4_5++;
							else                        m_stats.lz_worlddist_ge5++;
							/// Rescue when the bboxSizeSimilarity gate passes and worldDist is below m_maxWorldDist4ArchiveReAssoc.
							if (bboxSizeSimilarity >= 0.5f &&
								worldDist < m_params.m_maxWorldDist4ArchiveReAssoc) {
								/// Score decays linearly: 1.0 at dist=0, minMatchingScore4Overall at dist=maxDist.
								trackletMatchingScore = m_params.m_minMatchingScore4Overall +
									(1.0f - m_params.m_minMatchingScore4Overall) *
									(1.0f - worldDist / m_params.m_maxWorldDist4ArchiveReAssoc);
								m_stats.live_world_fallback++;
							}
						}
					}
				}
			}
			else
			{
				/// Temporal containment check failed.
				m_stats.temporal_fail++;

				if (m_params.m_debugVerboseLevel > 3)
				{
					cout << "[NvTrajectoryManager::findMatch4ReAssoc] Skipping Re-Assoc match since : (firstFrameNum, lastFrameNum) = ("
					<< firstFrameNum << "," << lastFrameNum << ")  whereas (firstFrameNum_peer, lastFrameNum_peer) = ("
					<< firstFrameNum_peer << "," << lastFrameNum_peer << ")" << endl;
				}
			}
			bool reidMatchingScoreIsValid = false;
			/// Get the matching score for reid based re-assoc, where each peer (shadow tracking objects in DB) has a list of reid features for selected current frame objects.
			if ( m_trackedObjectsReidInd.at(videoStreamId)->find(newTarget) != m_trackedObjectsReidInd.at(videoStreamId)->end())
			{
				m_stats.reid_ind_found++;
				int reidInd = m_trackedObjectsReidInd.at(videoStreamId)->at(newTarget);
				reidMatchingScore = (int(peer->getConfidenceReidDets().size()) > reidInd) ? \
						peer->getConfidenceReidDets().at(reidInd) : 0.f;
				reidFrameDiff = (int(peer->getConfidenceReidDetsFrameNum().size()) > reidInd && peer->getConfidenceReidDetsFrameNum().at(reidInd) >= 0) ? \
						peer->getTargetInfo().frameNum - peer->getConfidenceReidDetsFrameNum().at(reidInd) : -1;
				if(reidFrameDiff >= 0)
					reidMatchingScoreIsValid = true;
				if (m_params.m_debugVerboseLevel > 3)
				{
					cout << "(reidMatchingScore, reidFrameDiff) = (" << reidMatchingScore << "," <<
					reidFrameDiff << ")" << endl;
				}
			}

			if (reidMatchingScoreIsValid) m_stats.reid_score_valid++;
			/// ReID-only rescue for large-displacement re-association: reidCanRescue is always false when minMatchingScore4ReidSimilarity=0 (guard min > 0.0f), so YAML min=0 is a reliable kill-switch for the ReID rescue.
			const bool reidCanRescue = reidMatchingScoreIsValid &&
				(m_params.m_minMatchingScore4ReidSimilarity > 0.0f) &&
				(reidMatchingScore >= m_params.m_minMatchingScore4ReidSimilarity);
			if (reidCanRescue) m_stats.reid_rescue_possible++;
			/// For zero-tracklet-score pairs (rescue scenario), histogram of reidScore.
			if (reidMatchingScoreIsValid && trackletMatchingScore == 0.0f) {
				if      (reidMatchingScore < 0.2f) m_stats.reid_zs_lt02++;
				else if (reidMatchingScore < 0.4f) m_stats.reid_zs_02_04++;
				else if (reidMatchingScore < 0.5f) m_stats.reid_zs_04_05++;
				else                               m_stats.reid_zs_ge05++;
			}

			float matchingScore;
			if (reidCanRescue && trackletMatchingScore == 0.0f) {
				/// Large-displacement rescue path: spatial extrapolation failed entirely.
				matchingScore = reidMatchingScore;
			} else {
				/// Divides the weighted tracklet and ReID scores by the sum of their active weights.
				const float reidWeightActive = reidMatchingScoreIsValid ? m_params.m_matchingScoreWeight4ReidSimilarity : 0.0f;
				const float effectiveWeight = m_params.m_matchingScoreWeight4TrackletSimilarity + reidWeightActive;
				matchingScore = (effectiveWeight > 0.0f) ?
					(reidMatchingScore * reidWeightActive +
						trackletMatchingScore * m_params.m_matchingScoreWeight4TrackletSimilarity) / effectiveWeight :
					0.0f;
			}

			/// Gate: zero out if tracklet score below minimum AND no rescue path is active.
			if (trackletMatchingScore < m_params.m_minTrackletMatchingScore && !reidCanRescue)
			{
				matchingScore = 0;
			}

			if(bestMatchingScore < matchingScore)
			{
				bestMatchPeer = peer;

				bestMatchingScore = matchingScore;
				bestReidMatchingScore = reidMatchingScore;
				bestReidFrameDiff = reidFrameDiff;
				bestTrackletMatchingScore = trackletMatchingScore;
				bestTrackletFrameDiff = trackletFrameDiff;
				bestMatchAngle = angle;
				bestMatchSpeedSimilarity = speedSimilarity;
				bestMatchSpeedUnitMagnitude = speedUnitMagnitude;
				bestBboxSizeSimilarity = bboxSizeSimilarity;
			}

			if(m_params.m_debugVerboseLevel > 3)
			{
				if(matchingScore > 0)
					msg << DEBUG_UTIL::to_string_fmt("  %.3f", matchingScore);
				else
					msg << "      0";
			}
		}

		if(m_params.m_debugVerboseLevel > 3)
		{
			msg << endl;
			cout << msg.str();
		}

		/// Track sub-threshold scores (best score > 0 but below the gate).
		if(bestMatchingScore > 0.0f && bestMatchingScore < m_params.m_minMatchingScore4Overall)
		{
			m_stats.below_thresh++;
			if(bestMatchingScore > m_stats.best_failing_score.load(std::memory_order_relaxed))
				m_stats.best_failing_score.store(bestMatchingScore, std::memory_order_relaxed);
		}

		/// If the best matched tracklet has a high similarity, then use the older one.
		if(bestMatchingScore >= m_params.m_minMatchingScore4Overall && bestMatchPeer)
		{
			bestPeer = bestMatchPeer;

			if(m_params.m_debugVerboseLevel > 1)
			{
				cout << "Frame[" << newTarget->getTargetInfo().frameNum + 1 << "] New target ind[" << newTarget->getTargetInd() << "]---> Found a Re-Assoc match for TargetID[" << bestMatchPeer->getTargetID()
						<< "] w/ ((bestTrackletFrameDiff, bestReidFrameDiff), angle, speedSimilarity, speedUnitMagnitude, bboxSizeSimilarity, (trajMatching, reidMatching), fusedScore) = (("<< bestTrackletFrameDiff
						<< ", " << bestReidFrameDiff
						<< "), " << bestMatchAngle
						<< ", " << bestMatchSpeedSimilarity
						<< ", " << bestMatchSpeedUnitMagnitude
						<< ", " << bestBboxSizeSimilarity
						<< ", (" << bestTrackletMatchingScore
						<< ", " << bestReidMatchingScore
						<< "), " << bestMatchingScore << ")" << endl;
			}

			return true;
		}
	}

	return false;
}

/**
 * @brief Find a reassociation match among archived terminated tracks.
 * @param[in] newTarget Newly appearing track needing an ID.
 * @return True if a match above minMatchingScore4Overall was found.
 */
bool NvTrajectoryManager::findArchiveMatch4ReAssoc(const std::shared_ptr<NvTrackedObject> newTarget)
{
	const NvMOTStreamId videoStreamId = newTarget->getVideoStreamId();
	const NvMOTFrameNum currFrameNum  = newTarget->getTargetInfo().frameNum;
	/// Track whether any candidate reaches DTW scoring.
	bool az87_found_cand = false;

	if(m_trajectoryArchive.find(videoStreamId) == m_trajectoryArchive.end() || !m_trajectoryArchive.at(videoStreamId)) {
		m_stats.az_calls_total++; m_stats.az_calls_no_archive++;
		return false;
	}

	const Trajectory4ArchiveList &archiveMap = *m_trajectoryArchive.at(videoStreamId);
	if(archiveMap.empty()) {
		m_stats.az_calls_total++; m_stats.az_calls_no_archive++;
		return false;
	}

	const std::list<TargetInfo> newTraj = newTarget->getTrajectoryBuffer();

	float    bestMatchScore = 0.0f;
	uint64_t bestMatchId    = 0;
	/// Separate flag: avoids treating ID=0 as "no match".
	bool     foundMatch     = false;

	for(const auto &kv : archiveMap)
	{
		const uint64_t           archTargetId = kv.first;
		const Trajectory4Archive &archEntry   = kv.second;

		/// Already matched or output.
		if(archEntry.isReported)          continue;
		if(archEntry.tracklet.empty())    continue;

		/// ArchShadowAge: frames from last active detection to current frame.
		const int archShadowAge = (int)currFrameNum - (int)archEntry.lastFrameMatched;
		/// Archived track active after new track appeared.
		if(archShadowAge <= 0) continue;

		/// Count every archive entry the search attempt to match against.
		m_stats.archive_candidates++;
		/// At least one candidate reached DTW scoring.
		az87_found_cand = true;

		std::list<TargetInfo> projTracklet = createProjectedTrajectories(
			archTargetId, videoStreamId, archEntry.tracklet, archShadowAge
		);
		if(projTracklet.empty()) continue;

		const NvMOTFrameNum lastFrameNum_peer  = projTracklet.rbegin()->frameNum;
		const NvMOTFrameNum firstFrameNum_peer = projTracklet.begin()->frameNum;

		std::list<TargetInfo> traj = getTrimmedTraj4ReAssoc(newTraj, projTracklet, archEntry.lastFrameMatched);
		if(traj.empty()) continue;

		const NvMOTFrameNum lastFrameNum  = traj.rbegin()->frameNum;
		const NvMOTFrameNum firstFrameNum = traj.begin()->frameNum;

		/// Temporal containment: peer projection must span the new track's frame window.
		if(lastFrameNum_peer < lastFrameNum || firstFrameNum_peer > firstFrameNum)
		{
			m_stats.temporal_fail++;
			continue;
		}

		/// PeerTrajCov=0: uses trackletSpacialSearchRegionScale directly, so this value does not affect the IoU search region computation.
		const float peerTrajCov = 0.0f;
		float trackletScore = 0.0f;
		int   frameDiff     = 0;
		float angle = 0.0f, speedSim = 0.0f, speedMag = 0.0f, bboxSim = 0.0f;
		std::tie(trackletScore, frameDiff, angle, speedSim, speedMag, bboxSim) =
			calculateSimilarityBtwTracklets(traj, projTracklet, peerTrajCov);

		/// Count archive candidates that passed temporal check but scored zero.
		if(trackletScore == 0.0f)
		{
			m_stats.archive_zero_score++;
			const float pCx = (projTracklet.rbegin()->tBbox.min()(0) + projTracklet.rbegin()->tBbox.max()(0)) * 0.5f;
			const float pCy = (projTracklet.rbegin()->tBbox.min()(1) + projTracklet.rbegin()->tBbox.max()(1)) * 0.5f;
			float minDist = std::numeric_limits<float>::max();
			for(const auto &tp : traj)
			{
				const float nx = (tp.tBbox.min()(0) + tp.tBbox.max()(0)) * 0.5f;
				const float ny = (tp.tBbox.min()(1) + tp.tBbox.max()(1)) * 0.5f;
				const float d  = std::sqrt((nx - pCx)*(nx - pCx) + (ny - pCy)*(ny - pCy));
				if(d < minDist) minDist = d;
			}
			if     (minDist <  50.0f) m_stats.zs_disp_0_50++;
			else if(minDist < 100.0f) m_stats.zs_disp_50_100++;
			else if(minDist < 200.0f) m_stats.zs_disp_100_200++;
			else                      m_stats.zs_disp_200_plus++;
			/// BboxSizeSimilarity distribution for zero-score archive candidates: bboxSim is already computed above (calculateSimilarityBtwTracklets).
			if      (bboxSim < 0.5f) m_stats.az_bsim_lt50++;
			else if (bboxSim < 0.7f) m_stats.az_bsim_50_70++;
			else                     m_stats.az_bsim_ge70++;
		}

		if (trackletScore == 0.0f &&
			m_params.m_maxWorldDist4ArchiveReAssoc > 0.0f) {
			/// Find the last valid world-foot position in the archived track (scan backward shadow-tracking frames carry pt_world_feet = (-1,-1) sentinel).
			Eigen::Vector2f archLastWorld(-1.0f, -1.0f);
			for (auto rit = archEntry.tracklet.rbegin(); rit != archEntry.tracklet.rend(); ++rit) {
				if (rit->pt_world_feet.x() >= 0.0f) { archLastWorld = rit->pt_world_feet; break; }
			}
			/// Find the first valid world-foot position in the new track.
			Eigen::Vector2f newFirstWorld(-1.0f, -1.0f);
			for (const auto &tp : newTraj) {
				if (tp.pt_world_feet.x() >= 0.0f) { newFirstWorld = tp.pt_world_feet; break; }
			}
			if (archLastWorld.x() >= 0.0f && newFirstWorld.x() >= 0.0f) {
				const float worldDist = (archLastWorld - newFirstWorld).norm();
				/// World-distance histogram for all eligible candidates, regardless of whether they fall within the fallback gate.
				if     (worldDist <  1.0f) m_stats.az_worlddist_lt1++;
				else if(worldDist <  2.0f) m_stats.az_worlddist_1_2++;
				else if(worldDist <  3.0f) m_stats.az_worlddist_2_3++;
				else if(worldDist <  4.0f) m_stats.az_worlddist_3_4++;
				else if(worldDist <  5.0f) m_stats.az_worlddist_4_5++;
				else                       m_stats.az_worlddist_ge5++;
				/// Apply the bboxSim gate to the rescue only, not to the histogram.
				if (bboxSim >= 0.5f && worldDist < m_params.m_maxWorldDist4ArchiveReAssoc) {
					/// Score decays linearly: 1.0 at dist=0, minMatchingScore4Overall at dist=maxDist.
					trackletScore = m_params.m_minMatchingScore4Overall +
						(1.0f - m_params.m_minMatchingScore4Overall) *
						(1.0f - worldDist / m_params.m_maxWorldDist4ArchiveReAssoc);
					m_stats.archive_world_fallback++;
				}
			}
		}

		/// Bucket non-zero archive scores + track running best.
		if (trackletScore > 0.0f) {
			m_stats.az_score_hist[std::min(9, (int)(trackletScore * 10.f))]++;
			if (trackletScore > m_stats.az_score_best.load(std::memory_order_relaxed))
				m_stats.az_score_best.store(trackletScore, std::memory_order_relaxed);
		}

		/// World-distance of positive-score below-threshold archive candidates: fires when trackletScore in (0, minMatchingScore4Overall).
		if (trackletScore > 0.0f && trackletScore < m_params.m_minMatchingScore4Overall) {
			Eigen::Vector2f archLastWorld84(-1.0f, -1.0f);
			for (auto rit = archEntry.tracklet.rbegin(); rit != archEntry.tracklet.rend(); ++rit) {
				if (rit->pt_world_feet.x() >= 0.0f) { archLastWorld84 = rit->pt_world_feet; break; }
			}
			Eigen::Vector2f newFirstWorld84(-1.0f, -1.0f);
			for (const auto &tp : newTraj) {
				if (tp.pt_world_feet.x() >= 0.0f) { newFirstWorld84 = tp.pt_world_feet; break; }
			}
			if (archLastWorld84.x() >= 0.0f && newFirstWorld84.x() >= 0.0f) {
				const float wdBt = (archLastWorld84 - newFirstWorld84).norm();
				if     (wdBt <  1.0f) m_stats.azbt_worlddist_lt1++;
				else if(wdBt <  2.0f) m_stats.azbt_worlddist_1_2++;
				else if(wdBt <  3.0f) m_stats.azbt_worlddist_2_3++;
				else if(wdBt <  4.0f) m_stats.azbt_worlddist_3_4++;
				else if(wdBt <  5.0f) m_stats.azbt_worlddist_4_5++;
				else                   m_stats.azbt_worlddist_ge5++;
			} else {
				m_stats.azbt_worlddist_nopos++;
			}
		}

		if(trackletScore > bestMatchScore)
		{
			bestMatchScore = trackletScore;
			bestMatchId    = archTargetId;
			foundMatch     = true;
		}
	}

	if(bestMatchScore >= m_params.m_minMatchingScore4Overall && foundMatch)
	{
		newTarget->setTargetID(bestMatchId);

		/// Mark archive entry as reported so cleanUpArchivedTrajectories evicts it, preventing the same archived ID from being re-assigned to another new track.
		m_trajectoryArchive.at(videoStreamId)->at(bestMatchId).isReported = true;

		m_stats.archive_reassoc++;
		/// Success path counted in total.
		m_stats.az_calls_total++;

		if(m_params.m_debugVerboseLevel > 0)
		{
			cout << "[NvTrajectoryManager::findArchiveMatch4ReAssoc] Archive match found: "
				<< "newTarget inherits archived ID[" << bestMatchId << "] score=" << bestMatchScore << endl;
		}

		return true;
	}

	/// Failure path: count total and categorize.
	m_stats.az_calls_total++;
	if (!az87_found_cand) m_stats.az_calls_loop_empty++;
	else                  m_stats.az_calls_cand_failed++;
	return false;
}

/**
 * @brief Perform archive-only world-distance and frame-gap reassociation.
 * @param[in] newTarget Newly appearing track needing an ID.
 * @return True if a match above minMatchingScore4Overall was found.
 */
bool NvTrajectoryManager::try3DReAssoc(const std::shared_ptr<NvTrackedObject> newTarget)
{
	const NvMOTStreamId videoStreamId = newTarget->getVideoStreamId();

	if (m_params.m_maxWorldDist4ArchiveReAssoc <= 0.0f) return false;

	/// Archive loop is restricted to this stream's own archive.
	if (m_trajectoryArchive.empty()) return false;

	const std::list<TargetInfo> newTraj = newTarget->getTrajectoryBuffer();
	if (newTraj.empty()) return false;

	/// First valid world-foot position from new track.
	Eigen::Vector2f newFirstWorld(-1.0f, -1.0f);
	/// Capture the frame number where the first valid world position was found.
	int newFirstWorldFrame = (int)newTraj.begin()->frameNum;
	for (const auto &tp : newTraj) {
		if (tp.pt_world_feet.x() >= 0.0f) {
			newFirstWorld     = tp.pt_world_feet;
			newFirstWorldFrame = (int)tp.frameNum;
			break;
		}
	}
	if (newFirstWorld.x() < 0.0f) return false;

	const int newFirstFrame = (int)newTraj.begin()->frameNum;
	const TargetInfo &newLatest = *newTraj.rbegin();

	float       bestMatchScore    = 0.0f;
	uint64_t    bestMatchId       = 0;
	/// Separate flag: avoids treating ID=0 as "no match".
	bool        foundMatch        = false;
	/// Stream that provided the best archive match.
	NvMOTStreamId bestMatchStreamId = videoStreamId;

	/// Shared temporal gate used by both the archive and peer-archive loops below.
	const int effectiveMaxGap = 2 * m_params.m_maxFrameGap4WorldDistReAssoc;

	/// Best CosSim in [gateUsed, 3.0m) WS near-miss zone.
	float       wsExt25BestCosSim = -2.0f;
	/// Archive id of the best within-stream extended-rescue candidate.
	uint64_t    wsExt25BestId     = 0;

	/// Holds the best cross-stream near-miss cosine similarity under the exponential distance threshold.
	float    nmFuncBestCosSim = -2.0f;
	/// Peer ID of best functional-nm candidate.
	uint64_t nmFuncBestId     = 0;
	/// Stores the best direction-gate failure candidate for the separate near-miss rescue.
	float    nm2DirFailBestCosSim = -2.0f;  uint64_t nm2DirFailBestId = 0;

	/// Searches only this stream's trajectory archive for a same-camera re-association.
	const auto archStreamIt = m_trajectoryArchive.find(videoStreamId);
	if (archStreamIt != m_trajectoryArchive.end() && archStreamIt->second && !archStreamIt->second->empty()) {
	for (const auto &kv : *archStreamIt->second) {
		const uint64_t           archTargetId = kv.first;
		const Trajectory4Archive &archEntry   = kv.second;

		if (archEntry.isReported)       continue;
		if (archEntry.tracklet.empty()) continue;

		const int archLastFrame = (int)archEntry.lastFrameMatched;

		/// Temporal gate 1: archived track must have ended before new track appeared.
		if (archLastFrame >= newFirstFrame) continue;

		/// Collects up to ten valid world positions, including QUASIACTIVE frames, for the archive velocity fit.
		constexpr int kMaxVelPoints = 10;
		Eigen::Vector2f regPos[kMaxVelPoints];
		float           regT[kMaxVelPoints];
		int posCount = 0;
		for (auto rit = archEntry.tracklet.rbegin();
			rit != archEntry.tracklet.rend() && posCount < kMaxVelPoints; ++rit) {
			if (rit->pt_world_feet.x() >= 0.0f) {
				regPos[posCount] = rit->pt_world_feet;
				regT[posCount]   = static_cast<float>(rit->frameNum);
				posCount++;
			}
		}
		if (posCount == 0) continue;
		/// Most-recent valid world position.
		const Eigen::Vector2f archPos0 = regPos[0];

		/// OLS slope: v = Σ(t_i - t̄)(p_i - p̄) / Σ(t_i - t̄)².
		Eigen::Vector2f archVelocity(0.0f, 0.0f);
		if (posCount >= 2) {
			float tMean = 0.0f;
			Eigen::Vector2f pMean(0.0f, 0.0f);
			for (int i = 0; i < posCount; i++) { tMean += regT[i]; pMean += regPos[i]; }
			tMean /= static_cast<float>(posCount);
			pMean /= static_cast<float>(posCount);
			float tVar = 0.0f;
			Eigen::Vector2f tpCov(0.0f, 0.0f);
			for (int i = 0; i < posCount; i++) {
				const float dt = regT[i] - tMean;
				tVar  += dt * dt;
				tpCov += (regPos[i] - pMean) * dt;
			}
			if (tVar > 0.0f) {
				archVelocity = tpCov / tVar;
			}
		}

		/// Skips the archive candidate when the gap from its last frame to the new track's first valid world frame exceeds twice maxFrameGap4WorldDistReAssoc.
		const int frameGap = newFirstWorldFrame - archLastFrame;
		if (m_params.m_maxFrameGap4WorldDistReAssoc > 0 && frameGap > effectiveMaxGap) continue;

		/// Extrapolate archived position to newFirstWorldFrame using archLastFrame as reference.
		const Eigen::Vector2f archPredictedWorld =
			archPos0 + archVelocity * static_cast<float>(frameGap);

		const float worldDist = (archPredictedWorld - newFirstWorld).norm();

		/// Within-stream archive candidate counters.
		if      (frameGap < 50)  m_stats.ws_gap_lt50++;
		else if (frameGap < 100) m_stats.ws_gap_50_100++;
		else if (frameGap < 200) m_stats.ws_gap_100_200++;
		else                     m_stats.ws_gap_ge200++;
		if (worldDist >= m_params.m_maxWorldDist4ArchiveReAssoc) {
			if      (worldDist < 2.0f) m_stats.ws_wd_15_2++;
			else if (worldDist < 3.0f) m_stats.ws_wd_2_3++;
			else if (worldDist < 5.0f) m_stats.ws_wd_3_5++;
			else                       m_stats.ws_wd_ge5++;
		} else {
			m_stats.ws_passed++;
		}

		/// Bidirectional velocity prediction for within-stream re-association: if the track stops or reverses during the gap, forward extrapolation overshoots and the world-distance gate fails.
		float effectiveWorldDist = worldDist;
		{
			/// Measures the new track's velocity span from its first valid world frame to its latest frame.
			const int wsSpan = (int)newLatest.frameNum - newFirstWorldFrame;
			if (wsSpan > 0 && newLatest.pt_world_feet.x() >= 0.0f) {
				const Eigen::Vector2f wsNewVel =
					(newLatest.pt_world_feet - newFirstWorld) / static_cast<float>(wsSpan);
				const Eigen::Vector2f wsArchBackPred =
					newFirstWorld - wsNewVel * static_cast<float>(frameGap);
				const float wsWorldDistBack = (wsArchBackPred - archPos0).norm();
				effectiveWorldDist = std::min(worldDist, wsWorldDistBack);
			}
		}

		/// Within-stream archive gate when m_maxWorldDist4ArchiveReAssocWS is configured.
		const float gateUsed = (m_params.m_maxWorldDist4ArchiveReAssocWS > 0.0f)
			? m_params.m_maxWorldDist4ArchiveReAssocWS
			: m_params.m_maxWorldDist4ArchiveReAssoc;
		if (effectiveWorldDist >= gateUsed) {
			/// Cosine similarity for within-stream candidates in [gateUsed, m_maxWsExt25WorldDist).
			if (effectiveWorldDist < m_params.m_maxWsExt25WorldDist) {
				const std::vector<float>& extNewEmb = newTarget->getPersistentReidFeature();
				if (!extNewEmb.empty() && !archEntry.reidEmbedding.empty() &&
					extNewEmb.size() == archEntry.reidEmbedding.size()) {
					float extCs = 0.0f;
					for (size_t i = 0; i < extNewEmb.size(); ++i)
						extCs += extNewEmb[i] * archEntry.reidEmbedding[i];
					/// Base threshold for [gateUsed, 3.0m): stricter for [3.0m, maxWsExt25WorldDist) to guard FP.
					const float wsExtThresh = (effectiveWorldDist < 3.0f)
						? m_params.m_wsExt25MinCosSim
						: m_params.m_wsExt25StrictCosSim;
					if (extCs >= wsExtThresh) {
						if (wsExt25BestCosSim < extCs) {
							wsExt25BestCosSim = extCs;
							wsExt25BestId     = archTargetId;
						}
					}
				}
			}
			continue;
		}

		/// BboxSizeSimilarity gate.
		const TargetInfo &archLatest = *archEntry.tracklet.rbegin();
		/// Use the most recent ACTIVE frame for observed size fields, defaulting to the last QUASIACTIVE frame.
		const NvMotDataTypes::Rect2f *archBbox4Sim = &archLatest.tBbox;
		/// Last ACTIVE OMP-estimated height (m): 0=unknown.
		float archHeight4Sim = 0.0f;
		/// Preserve archLatest.distFromCam so the size gate remains unnormalized.
		for (auto rit = archEntry.tracklet.rbegin(); rit != archEntry.tracklet.rend(); ++rit) {
			if (rit->trackerState == TRACKER_STATE::ACTIVE) {
				/// Last ACTIVE frame's real OMP-measured bbox.
				archBbox4Sim   = &rit->tBbox;
				/// Last ACTIVE frame's OMP-estimated height.
				archHeight4Sim = rit->height;
				break;
			}
		}
		const float bboxSim = CV_FUNCS::computeSizeScore(newLatest.tBbox, *archBbox4Sim);
		if      (bboxSim < 0.10f) m_stats.ws_bsim_lt10++;
		else if (bboxSim < 0.25f) m_stats.ws_bsim_10_25++;
		else if (bboxSim < 0.50f) m_stats.ws_bsim_25_50++;
		else                      m_stats.ws_bsim_ge50++;
		/// Reject candidates whose bounding-box similarity is below 0.25.
		if (bboxSim < 0.25f) continue;
		/// Tighter bounding-box similarity floor for the extended zone.
		if (effectiveWorldDist >= gateUsed * 0.80f && bboxSim < 0.40f) continue;

		/// Rejects the candidate when both pose heights are known and their ratio is below 0.82.
		const float newHeight = newLatest.height;
		if (archHeight4Sim > 0.0f && newHeight > 0.0f) {
			const float heightRatio = std::min(archHeight4Sim, newHeight)
				/ std::max(archHeight4Sim, newHeight);
			/// A height difference above 18% indicates different people.
			if (heightRatio < 0.82f) continue;
		}

		/// Normalize by gateUsed so score ∈ [minThreshold, 1.0] regardless of which gate applied.
		const float worldDistScore = 1.0f - effectiveWorldDist / gateUsed;
		const float score = m_params.m_minMatchingScore4Overall +
			(1.0f - m_params.m_minMatchingScore4Overall) * worldDistScore;

		if (score > bestMatchScore) {
			bestMatchScore         = score;
			bestMatchId            = archTargetId;
			foundMatch             = true;
			bestMatchStreamId      = videoStreamId;
		}
	} // end archive-entry loop
	} // end same-stream archive block

	/// True while the current best came from the archive loop.
	bool matchFromArchive = foundMatch;
	/// Peer-archive: search stale PeerTargetDB entries from all local MVAs.
	for (const auto &mvaEntry : m_multiViewAssociators) {
		const PeerTargetDB peerDB = mvaEntry.second->getPeerTargetDB();
		for (const auto &camEntry : peerDB) {
			for (const auto &targetEntry : camEntry.second) {
				const PeerTarget &peer = targetEntry.second;

				/// Deduplication: mirrors archive's isReported + eviction cycle.
				{
					auto dedup_it = m_usedPeerTargetIds.find(peer.targetId);
					if (dedup_it != m_usedPeerTargetIds.end()) {
						const int claimAge = (int)newLatest.frameNum - (int)dedup_it->second;
						if (claimAge < m_params.m_maxAge4TrackletArchiving) continue;
						/// Claim expired: allow re-association.
						m_usedPeerTargetIds.erase(dedup_it);
					}
				}

				/// Use the tracklet's last ego-translated frameNum as the temporal reference.
				const int peerLastFrame = peer.tracklet.empty()
					? (int)peer.frameNum // Fallback when there is no tracklet.
					: (int)peer.tracklet.back().frameNum; // Prefer the ego-translated peer state time.
				const int pFrameGap = newFirstWorldFrame - peerLastFrame;
				/// Still live or concurrent: not an archive candidate.
				if (pFrameGap <= 0) continue;
				if (m_params.m_maxFrameGap4WorldDistReAssoc > 0
					&& pFrameGap > effectiveMaxGap) continue;

				/// OLS velocity from peer tracklet: mirrors archive loop above.
				constexpr int kMaxPVelPts = 10;
				Eigen::Vector2f pRegPos[kMaxPVelPts];
				float           pRegT[kMaxPVelPts];
				int pPosCount = 0;
				for (auto rit = peer.tracklet.rbegin();
					rit != peer.tracklet.rend() && pPosCount < kMaxPVelPts; ++rit) {
					pRegPos[pPosCount] = Eigen::Vector2f(rit->x, rit->y);
					pRegT[pPosCount]   = static_cast<float>(rit->frameNum);
					pPosCount++;
				}
				/// Fall back to nextState when tracklet is empty.
				Eigen::Vector2f pPos0(
					static_cast<float>(peer.nextState(0)), static_cast<float>(peer.nextState(1))
				);
				Eigen::Vector2f pVelocity(
					static_cast<float>(peer.nextState(2)), static_cast<float>(peer.nextState(3))
				);
				if (pPosCount > 0) {
					pPos0 = pRegPos[0];
					if (pPosCount >= 2) {
						float tMean = 0.0f; Eigen::Vector2f pMean(0.0f, 0.0f);
						for (int i = 0; i < pPosCount; i++) { tMean += pRegT[i]; pMean += pRegPos[i]; }
						tMean /= pPosCount; pMean /= pPosCount;
						float tVar = 0.0f; Eigen::Vector2f tpCov(0.0f, 0.0f);
						for (int i = 0; i < pPosCount; i++) {
							const float dt = pRegT[i] - tMean;
							tVar  += dt * dt;
							tpCov += (pRegPos[i] - pMean) * dt;
						}
						pVelocity = (tVar > 0.0f) ? tpCov / tVar : Eigen::Vector2f(0.0f, 0.0f);
					}
				}

				const Eigen::Vector2f pPredWorld = pPos0 + pVelocity * static_cast<float>(pFrameGap);
				const float pWorldDist = (pPredWorld - newFirstWorld).norm();

				/// Always treat peer entries as cross-stream: use CS gate.
				const float pGate = (m_params.m_maxWorldDist4ArchiveReAssocCS > 0.0f)
					? m_params.m_maxWorldDist4ArchiveReAssocCS
					: m_params.m_maxWorldDist4ArchiveReAssoc;
				if (pWorldDist >= pGate) {
					/// CS nm functional cascade: exponential threshold t(d) = max - (max-min)*exp(-decay*(d-pGate)).
					if (pWorldDist >= m_params.m_maxNmCascadeWorldDist) { continue; }
					const std::vector<float>& nmNewEmb = newTarget->getPersistentReidFeature();
					if (!peer.reidEmbedding.empty() && !nmNewEmb.empty() &&
						nmNewEmb.size() == peer.reidEmbedding.size()) {
						float nmCosSim = 0.0f;
						for (size_t i = 0; i < nmNewEmb.size(); ++i)
							nmCosSim += nmNewEmb[i] * peer.reidEmbedding[i];

						/// Exponential threshold: it rises steeply near pGate and plateaus at nmCascadeMaxCosSim.
						const float nmThreshold = m_params.m_nmCascadeMaxCosSim -
							(m_params.m_nmCascadeMaxCosSim - m_params.m_nmCascadeMinCosSim) *
							std::exp(-m_params.m_nmCascadeDecayRate * (pWorldDist - pGate));

						if (nmCosSim >= nmThreshold) {
							bool nmDirOk = true;
							if (pPosCount >= 2) {
								const float nmVelMag = pVelocity.norm();
								if (nmVelMag > 0.05f) {
									const Eigen::Vector2f nmDir = (newFirstWorld - pPos0).normalized();
									if (pVelocity.normalized().dot(nmDir) < -0.3f) nmDirOk = false;
								}
							}
							bool nmHgtOk = true;
							if (peer.height > 0.f && newLatest.height > 0.f) {
								if (std::min(peer.height, newLatest.height) /
									std::max(peer.height, newLatest.height) < 0.707f) nmHgtOk = false;
							}
							if (nmDirOk && nmHgtOk && nmCosSim > nmFuncBestCosSim) {
								nmFuncBestCosSim = nmCosSim;
								nmFuncBestId     = peer.targetId;
							}
						}
						/// Records the best height-gated candidate in the 2.5–3.0 m band whose cosine similarity is at least 0.85.
						if (pWorldDist >= 2.5f && pWorldDist < 3.0f && nmCosSim >= 0.85f) {
							bool nm2HgtOk = true;
							if (peer.height > 0.f && newLatest.height > 0.f) {
								if (std::min(peer.height, newLatest.height) /
									std::max(peer.height, newLatest.height) < 0.707f) nm2HgtOk = false;
							}
							if (nm2HgtOk && nmCosSim > nm2DirFailBestCosSim) {
								nm2DirFailBestCosSim = nmCosSim;
								nm2DirFailBestId = peer.targetId;
							}
						}
					}
					continue;
				}

				/// Bidirectional velocity (logic: stream-agnostic).
				float pEffWorldDist = pWorldDist;
				{
					const int wsSpan = (int)newLatest.frameNum - newFirstWorldFrame;
					if (wsSpan > 0 && newLatest.pt_world_feet.x() >= 0.0f) {
						const Eigen::Vector2f wsNewVel =
							(newLatest.pt_world_feet - newFirstWorld) / static_cast<float>(wsSpan);
						const Eigen::Vector2f wsBackPred =
							newFirstWorld - wsNewVel * static_cast<float>(pFrameGap);
						pEffWorldDist = std::min(pWorldDist, (wsBackPred - pPos0).norm());
					}
				}
				/// This check is dead code after Gate A and is kept for symmetry.
				if (pEffWorldDist >= pGate) continue;

				/// Direction gate: for a valid cross-camera re-association the new track must appear in the forward hemisphere of the peer's motion.
				if (pPosCount >= 2) {
					const float pVelMag = pVelocity.norm();
					if (pVelMag > 0.05f) {
						const Eigen::Vector2f dirToNew = (newFirstWorld - pPos0).normalized();
						const float cosAlign = pVelocity.normalized().dot(dirToNew);
						/// A peer moving away from the new track is a wrong match, at an angle of about 107 degrees or more.
						if (cosAlign < -0.3f) continue;
					}
				}

				/// Height-similarity gate: peer re-association is inherently cross-camera, so the size discriminator must be camera-independent: heights are world-space metres (focal-length / resolution invariant), whereas pixel-area bboxSim conflates the two cameras' viewing geometries and produces false negatives across views.
				const float pHPeer = peer.height;
				const float pHNew  = newLatest.height;
				if (pHPeer > 0.f && pHNew > 0.f) {
					const float pHRatio = std::min(pHPeer, pHNew) / std::max(pHPeer, pHNew);
					/// A height mismatch above about 29% indicates different people.
					if (pHRatio < 0.707f) continue;
				}

				const float pWorldDistScore = 1.0f - pEffWorldDist / pGate;
				const float pScore = m_params.m_minMatchingScore4Overall +
					(1.0f - m_params.m_minMatchingScore4Overall) * pWorldDistScore;

				/// CS bboxSim gate: rejects cross-camera candidates with dissimilar apparent sizes.
				if (m_params.m_minBboxSizeSimilarity4ReAssocCS > 0.0f && !peer.tBbox.isEmpty()) {
					const float csBboxSim = CV_FUNCS::computeSizeScore(newLatest.tBbox, peer.tBbox);
					if (csBboxSim < m_params.m_minBboxSizeSimilarity4ReAssocCS) continue;
				}

				if (pScore > bestMatchScore) {
					bestMatchScore       = pScore;
					bestMatchId          = peer.targetId;
					foundMatch           = true;
					matchFromArchive     = false;
				}
			}
		}
	} // end peer-archive loop

	/// CS nm functional cascade adoption: exponential threshold, single best candidate.
	if (!foundMatch && nmFuncBestCosSim >= 0.0f) {
		bestMatchScore = m_params.m_minMatchingScore4Overall +
			(1.0f - m_params.m_minMatchingScore4Overall) * nmFuncBestCosSim;
		bestMatchId      = nmFuncBestId;
		matchFromArchive = false;
		foundMatch       = true;
	} else if (!foundMatch && nm2DirFailBestCosSim >= 0.85f && m_params.m_maxNmCascadeWorldDist <= 14.5f) {
		/// Adopts the direction-gate failure candidate when no match exists, its cosine similarity is at least 0.85, and the cascade distance limit is at most 14.5 m.
		bestMatchScore = m_params.m_minMatchingScore4Overall +
			(1.0f - m_params.m_minMatchingScore4Overall) * nm2DirFailBestCosSim;
		bestMatchId      = nm2DirFailBestId;
		matchFromArchive = false;
		foundMatch       = true;
	}

	/// Adopts the best same-camera extended-rescue candidate when no regular match was found and its cosine similarity meets the configured minimum.
	if (!foundMatch && wsExt25BestCosSim >= m_params.m_wsExt25MinCosSim) {
		bestMatchScore   = m_params.m_minMatchingScore4Overall +
			(1.0f - m_params.m_minMatchingScore4Overall) * wsExt25BestCosSim;
		bestMatchId      = wsExt25BestId;
		bestMatchStreamId= videoStreamId;
		matchFromArchive = true;
		foundMatch       = true;
	}

	if (bestMatchScore >= m_params.m_minMatchingScore4Overall && foundMatch) {
		newTarget->setTargetID(bestMatchId);
		if (matchFromArchive) {
			m_trajectoryArchive.at(bestMatchStreamId)->at(bestMatchId).isReported = true;
		} else {
			/// Peer-archive deduplication: record claim frame to enforce the expiry window.
			m_usedPeerTargetIds[bestMatchId] = newLatest.frameNum;
		}
		m_stats.archive_reassoc++;
		if (m_params.m_debugVerboseLevel > 0) {
			fprintf(
				stderr,
				"[NvTrajectoryManager::try3DReAssoc] " "stream[%llu] frame[%d] -> archived ID[%lu] worldDist<%.1fm gap<%dframes score=%.3f\n",
				(unsigned long long)(videoStreamId >> 32), newFirstFrame, (unsigned long)bestMatchId,
				m_params.m_maxWorldDist4ArchiveReAssoc, m_params.m_maxFrameGap4WorldDistReAssoc, bestMatchScore
			);
		}
		return true;
	}
	return false;
}

/**
 * @brief Generate target detections that require ReID extraction for one stream.
 * @param[in] streamId Source stream identifier.
 * @param[in] targetList Tracked targets to inspect.
 */
std::deque<DetectedObjInfo> NvTrajectoryManager::generateTrackedObjects4ReidPerStream(
	const NvMOTStreamId streamId, const NvTrackedObjectList &targetList
)
{
	std::deque<DetectedObjInfo> targetReidList;
	return targetReidList;
}

/**
 * @brief Calculate bounding-box size similarity between two trajectories.
 * @param[in] traj First trajectory.
 * @param[in] peerTraj Peer trajectory.
 * @return Average bounding-box size similarity.
 */
float NvTrajectoryManager::calculateSizeSimilarityScore(
	const std::list<TargetInfo> &traj, const std::list<TargetInfo> &peerTraj
)
{
	float sizeSimilarityScore = CV_FUNCS::computeSizeScore(traj.rbegin()->tBbox, peerTraj.rbegin()->tBbox);
	return sizeSimilarityScore;
}

/**
 * @brief Calculate tracklet, direction, speed, and size similarity metrics.
 * @param[in] traj First trajectory.
 * @param[in] peerTraj Peer trajectory.
 * @param[in] peerTrajCov Peer trajectory covariance.
 * @return Tuple of tracklet score, frame difference, angle, speed similarity, speed unit magnitude, and size similarity.
 */
const std::tuple<float, int, float, float, float, float> NvTrajectoryManager::calculateSimilarityBtwTracklets(
	const std::list<TargetInfo> &traj, const std::list<TargetInfo> &peerTraj, const float peerTrajCov
)
{
	float totalScore = 0;
	float avgIouScore = 0;
	int bestFrameDiff = 0;
	float bestMatchAngle = 0;
	float bestMatchSpeedSimilarity = 0;
	float bestMatchSpeedUnitMagnitude = 0;

	/// Size similarity.
	float sizeSimilarityScore = calculateSizeSimilarityScore(traj, peerTraj);
	if(sizeSimilarityScore >= m_params.m_minBboxSizeSimilarity4TrackletMatching)
	{
		/// Compute the quasi-Dynamic Time Warping score.
		std::tie(
			avgIouScore, bestFrameDiff, bestMatchAngle, bestMatchSpeedSimilarity, bestMatchSpeedUnitMagnitude
		) = calculateAssocScore4QuasiDTW(traj, peerTraj, peerTrajCov);

		totalScore = avgIouScore;

		if(m_params.m_debugVerboseLevel > 2)
		{
			cout  << "[NvTrajectoryManager::calculateSimilarityBtwTracklets] sizeSimilarityScore = " << sizeSimilarityScore
					<< ", (avgIouScore, bestFrameDiff, bestMatchAngle, bestMatchSpeedSimilarity, bestMatchSpeedUnitMagnitude) = "
					<< "(" << avgIouScore << ", " << bestFrameDiff << ", " << bestMatchAngle << ", " << bestMatchSpeedSimilarity << ", " << bestMatchSpeedUnitMagnitude << ")" << endl;
		}
	}
	else
	{
		if(m_params.m_debugVerboseLevel > 3)
		{
			cout << "sizeSimilarityScore = " << sizeSimilarityScore << ", which should be larger than " << m_params.m_minBboxSizeSimilarity4TrackletMatching << endl;
		}
	}
	return {
		totalScore, bestFrameDiff, bestMatchAngle, bestMatchSpeedSimilarity,
		bestMatchSpeedUnitMagnitude, sizeSimilarityScore
	};
}

/**
 * @brief Compute the angle between two tracklet direction vectors.
 * @param[in] vec1 Direction vector of the first tracklet.
 * @param[in] vec2 Direction vector of the second tracklet.
 * @return Angle between the two directions in degrees.
 */
const float NvTrajectoryManager::computeAngleInDegree(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2)
{

	const float dot = vec1(0)*vec2(0) + vec1(1)*vec2(1);
	const float det = vec1(0)*vec2(1) - vec1(1)*vec2(0);
	const float angleInDeg = atan2(det, dot) * 180/M_PI;

	return angleInDeg;
}

/**
 * @brief Compute speed similarity between two tracklets.
 * @param[in] vec1 Velocity vector of the first tracklet.
 * @param[in] vec2 Velocity vector of the second tracklet.
 * @return Speed similarity, where a higher value means a closer match.
 */
const float NvTrajectoryManager::computeSpeedSimilarity(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2)
{
	const float magSq1 = std::pow(vec1(0),2)+std::pow(vec1(1),2);
	const float magSq2 = std::pow(vec2(0),2)+std::pow(vec2(1),2);

	/// Apply a cosine to (s1 - s2)^2 / (s1^2 + s2^2) so that a higher similarity is better.
	const float speedSimilarity = std::cos(
		std::pow(std::sqrt(magSq1) - std::sqrt(magSq2),2) / (magSq1 + magSq2) * M_PI/2
	);

	return speedSimilarity;
}

/**
 * @brief Determine whether a tracklet speed reaches a threshold.
 * @param[in] vec Velocity vector of the tracklet.
 * @param[in] threshold Minimum speed treated as motion.
 * @return True when the vector speed is at least the threshold.
 */
const bool NvTrajectoryManager::detectMotion(const Eigen::Vector2f &vec, float threshold)
{
	const float speed = std::sqrt(std::pow(vec(0),2)+std::pow(vec(1),2));

	bool isMoving = (speed >= threshold);

	return isMoving;
}

/**
 * @brief Calculate quasi-DTW association metrics between two trajectories.
 * @param[in] traj First trajectory.
 * @param[in] peerTraj_init Peer trajectory.
 * @param[in] peerTrajCov Peer trajectory covariance.
 * @return Tuple of average IOU, frame difference, angle, speed similarity, and speed unit magnitude.
 */
const std::tuple<float, int, float, float, float> NvTrajectoryManager::calculateAssocScore4QuasiDTW(
	const std::list<TargetInfo> &traj, const std::list<TargetInfo> &peerTraj_init, const float peerTrajCov
)
{
	float maxIouAvg = 0;
	int bestFrameDiff = 0;
	float bestMatchAngle = 0;
	float bestMatchSpeedSimilarity = 0;
	float bestMatchSpeedUnitMagnitude = 0;

	if (traj.size() <= 1 || peerTraj_init.size() <= 1)
	{
		CLOG("Trajectory and initial peer trajectory list should have length larger than 1");
		return {maxIouAvg, bestFrameDiff, bestMatchAngle, bestMatchSpeedSimilarity, bestMatchSpeedUnitMagnitude};
	}

	const uint32_t lastFrameNum = traj.rbegin()->frameNum;
	const uint32_t firstFrameNum = traj.begin()->frameNum;

	/// Middle point to avoid using some initial measurements.
	auto it_mid = traj.begin();
	std::advance(it_mid, 1);

	const Eigen::Vector2f &loc_1 = it_mid->loc;
	const Eigen::Vector2f &loc_2 = traj.rbegin()->loc;
	if(m_params.m_debugVerboseLevel > 3)
	{
		for (auto it = traj.begin(); it != traj.end(); ++it)
		{
			cout << " pt: frameNum = " << it->frameNum << ", loc (x,y) = (" << it->loc(0) << ", " << it->loc(1) << ")" << endl;
		}
	}

	std::list<TargetInfo> peerTraj = TrajOps::generateFullTracklet(
		peerTraj_init, firstFrameNum - m_params.m_maxTrackletMatchingTimeSearchRange,
		lastFrameNum + m_params.m_maxTrackletMatchingTimeSearchRange
	);

	if (peerTraj.empty())
	{
		CLOG("Peer trajectory is empty");
		return {maxIouAvg, bestFrameDiff, bestMatchAngle, bestMatchSpeedSimilarity, bestMatchSpeedUnitMagnitude};
	}

	for(int dT = -(int)m_params.m_maxTrackletMatchingTimeSearchRange;
		dT < (int)m_params.m_maxTrackletMatchingTimeSearchRange; dT += 1)
	{
		/// Iterate both tracklets on the same frame numbers.
		auto peerIt = peerTraj.begin();
		while(peerIt != peerTraj.end() && peerIt->frameNum < firstFrameNum + dT)
		{
			++peerIt;
		}

		if(peerIt->frameNum != firstFrameNum + dT)
		{
			continue;
		}

		const Eigen::Vector2f &peerLoc_1 = peerIt->loc;
		const uint32_t firstFrameNum_peer = peerIt->frameNum;

		/// Average IOU score.
		float iou_avg = 0;
		float iou_first = 0;
		bool isValid = true;
		uint32_t trajLength = 0;
		for(auto it = traj.begin(); it != traj.end() && peerIt != peerTraj.end(); ++it, ++peerIt, ++trajLength)
		{
			const TargetInfo &pt = *it;
			const TargetInfo &pt_peer = *peerIt;

			/// Print the debug information.
			if(m_params.m_debugVerboseLevel > 3)
			{
				cout << " - trajPtInd = " << trajLength << ", pt_peer: frameNum = " << peerIt->frameNum << ", loc (x,y) = (" << pt_peer.tBbox.min()(0) << ", " << pt_peer.tBbox.min()(1) << ")" << endl;
			}

			/// Bounded position tolerance: paddingScale = trackletSpacialSearchRegionScale directly (NOT multiplied by peerTrajCov, which grows unboundedly for long shadow tracks).
			NvMotDataTypes::Rect2f peerSeachRegion = CV_FUNCS::computeSearchRegion(
				pt_peer.tBbox, m_params.m_trackletSpacialSearchRegionScale
			);

			float iou = CV_FUNCS::computeVisibility(pt.tBbox, peerSeachRegion);
			if(it == traj.begin())
				iou_first = iou;

			iou_avg += iou;
		}

		if (trajLength > 0)
		{
			iou_avg /= (float)trajLength;
		}

		if(m_isOffline)
		{
			/// Modify iou_avg based on the distance between frame numbers in two tracklets.
			iou_avg = iou_avg * (1.0f - m_params.m_offlineParams.m_iouAvgScaleFactor * (float)abs(dT));
		}

		peerIt--;
		const Eigen::Vector2f peerLoc_2 = peerIt->loc;
		const uint32_t lastFrameNum_peer = peerIt->frameNum;

		/// Motion vectors.
		const Eigen::Vector2f unitVecSelf = (loc_2 - loc_1) / (float)(traj.size() - 1) * 100;
		const Eigen::Vector2f unitVecPeer = (peerLoc_2 - peerLoc_1) / (float)trajLength * 100;
		const float angle = std::fabs(computeAngleInDegree(unitVecSelf, unitVecPeer));
		const float speedSimilarity = computeSpeedSimilarity(unitVecSelf, unitVecPeer);
		const bool isObjectMoving = detectMotion(unitVecSelf, m_params.m_minSpeed4AngleChecking*100);

		if(m_params.m_debugVerboseLevel > 3)
		{

			cout << "[NvTrajectoryManager::calculateAssocScore4QuasiDTW] (isValid, iou_first, iou_avg, speedSimilarity, angle, isObjectMoving) = ("
					<< isValid << ", " << iou_first << ", " << iou_avg << ", " << speedSimilarity << ", " << angle << ", " << isObjectMoving << ")" << endl
					<< "           trajLength = " << trajLength
					<< ",  unitVectSelf(" << unitVecSelf(0) << ", " << unitVecSelf(1) << ") w/ frame (first, last) = (" << firstFrameNum << ", " << lastFrameNum << ")"
					<< ", unitVectPeer(" << unitVecPeer(0) << ", " << unitVecPeer(1) << ") w/ frame (first, last) = (" << firstFrameNum_peer << ", " << lastFrameNum_peer << ")" << endl;
		}

		if(isValid
			&& maxIouAvg < iou_avg
			&& speedSimilarity >= m_params.m_minSpeedSimilarity4TrackletMatching
			&& (!isObjectMoving || (iou_first >= m_params.m_maxIou4AngleChecking) || (angle <= m_params.m_maxAngle4TrackletMatching))
			)
		{
			bestMatchAngle = angle;
			bestMatchSpeedSimilarity = speedSimilarity;
			bestMatchSpeedUnitMagnitude =
				std::sqrt(unitVecPeer(0)*unitVecPeer(0) + unitVecPeer(1)*unitVecPeer(1)) / 100;
			maxIouAvg = iou_avg;
			bestFrameDiff = dT;
		}
	}

	return {maxIouAvg, bestFrameDiff, bestMatchAngle, bestMatchSpeedSimilarity, bestMatchSpeedUnitMagnitude};
}

/**
 * @brief Add a shadow target's projected trajectory to the reassociation database.
 * @param[in] target Shadow track whose projected trajectory is stored.
 */
void NvTrajectoryManager::add2ProjectedTrajectoryDB(std::shared_ptr<NvTrackedObject> target)
{
	/// Count shadow tracks entering the projected trajectory DB.
	m_stats.db_add_calls++;

	const uint64_t targetID = target->getTargetID();
	const NvMOTStreamId videoStreamId = target->getVideoStreamId();

	/// Generate projected trajectory.
	std::list<TargetInfo> tracklet = createProjectedTrajectories(
		targetID, videoStreamId, target->getTrajectoryBuffer()
	);

	/// Generate trajectory uncertainty based on KF.
	Eigen::VectorXd meas;
	Eigen::MatrixXd measCov;
	std::tie(meas, measCov) = target->getPredictedMeasurementsWithCov();
	Eigen::Diagonal<Eigen::MatrixXd> measCovDiag = measCov.diagonal();
	float trackletCov = std::sqrt(measCovDiag(0) * measCovDiag(1));

	if(tracklet.size() > 0)
	{
		/// Add to the temporary projected trajectory DB.
		if (m_projectedTrajectories.find(videoStreamId) != m_projectedTrajectories.end() && m_projectedTrajectories.at(videoStreamId))
		{
			(*m_projectedTrajectories.at(videoStreamId))[target] = {tracklet, trackletCov};
		}
		else
		{
			CLOG("Stream not found in projected trajectory data base");
		}

		if(m_params.m_debugVerboseLevel > 3)
		{
			cout << "[NvTrajectoryManager::add2ProjectedTrajectoryDB()] Stream[" << (videoStreamId>>32) << "] frameNum[" << target->getTargetInfo().frameNum
					<< "] A new projected trajectory is added: TargetID[" << target->getTargetID()
					<< "] m_projectedTrajectories.at(videoStreamId)->size() = " << m_projectedTrajectories.at(videoStreamId)->size() << endl;
		}
	}
}

/**
 * @brief Remove a target from the projected-trajectory database.
 * @param[in] target Track to remove from the projected trajectory database.
 */
void NvTrajectoryManager::removeFromProjectedTrajectoryDB(std::shared_ptr<NvTrackedObject> target)
{
	const NvMOTStreamId videoStreamId = target->getVideoStreamId();

	if(m_projectedTrajectories.find(videoStreamId) != m_projectedTrajectories.end() && m_projectedTrajectories.at(videoStreamId)
			&& m_projectedTrajectories.at(videoStreamId)->find(target) != m_projectedTrajectories.at(videoStreamId)->end())
	{
		m_projectedTrajectories.at(videoStreamId)->erase(target);
	}
}

/** @brief Generate a thread-safe target ID according to the configured ID mode. */
const TargetId NvTrajectoryManager::generateNewTargetID()
{
	/// Generate a new unique target ID.
	TargetId newtargetID = m_uidGenerator.generateId();

	/// Add the start value to the target ID.
	newtargetID += m_params.m_targetIDOffset;

	/// If unique ID is not enabled, then just take the lower 32-bit, which increment from 0.
	if(!m_params.m_useUniqueID)
	{
		newtargetID &= 0xFFFFFFFF;
	}

	return newtargetID;
}

/** @brief Return the next target ID without consuming it. */
const TargetId NvTrajectoryManager::peekNewTargetID() const
{
	/// Generate a new unique target ID.
	TargetId newtargetID = m_uidGenerator.peekNextId();

	/// Add the start value to the target ID.
	newtargetID += m_params.m_targetIDOffset;

	/// If unique ID is not enabled, then just take the lower 32-bit, which increment from 0.
	if(!m_params.m_useUniqueID)
	{
		newtargetID &= 0xFFFFFFFF;
	}

	return newtargetID;
}

/**
 * @brief Store a terminated target's trajectory in the archive.
 * @param[in] target Terminated object whose trajectory is stored.
 */
void NvTrajectoryManager::archiveTrajectory(const std::shared_ptr<NvTrackedObject> target)
{
	const NvMOTStreamId videoStreamId = target->getVideoStreamId();
	const TargetId targetId = target->getTargetID();
	std::list<TargetInfo> tracklet = target->getTrajectoryBuffer();

	/// Remove from the projected trajectory DB.
	removeFromProjectedTrajectoryDB(target);

	/// Perform any post-processing on the tracklet before actual archival if needed.
	if(processTrajectory(targetId, videoStreamId, tracklet))
	{
		const uint32_t lastFrameMatched =  tracklet.rbegin()->frameNum;

		Trajectory4Archive traj4archive = {
			/// When terminated.
			target->getTargetInfo().frameNum,
			/// When matched last time.
			lastFrameMatched,
			/// Tracklet data.
			std::move(tracklet),
			/// If reported.
			false,
			#ifdef VIZ_DEBUG
				false
			#endif
		};
		/// Copies the target's persistent ReID embedding into the archived trajectory.
		traj4archive.reidEmbedding = target->getPersistentReidFeature();

		/// Perform the actual archival, keeping multiple tracklets with the same ID to fuse or split later.
		if (m_trajectoryArchive.find(videoStreamId) != m_trajectoryArchive.end() && m_trajectoryArchive.at(videoStreamId))
		{
			if (m_trajectoryArchive.at(videoStreamId)->find(targetId) != m_trajectoryArchive.at(videoStreamId)->end())
			{
				m_trajectoryArchive.at(videoStreamId)->at(targetId).tracklet.insert(
					m_trajectoryArchive.at(videoStreamId)->at(targetId).tracklet.end(),
					tracklet.begin(),
					tracklet.end()
				);
			}
			else
			{
				(*m_trajectoryArchive.at(videoStreamId))[targetId] = std::move(traj4archive);
			}
			if(m_params.m_debugVerboseLevel > 3)
			{
				cout << "[NvTrajectoryManager::archiveTrajectory()] Stream[" << (videoStreamId>>32)
					<< "] Archiving the tracklet of Target ID[" << targetId << "]: lastFrameMatched = " << lastFrameMatched << endl;
			}
		}
		else
		{
			CLOG("Stream not found in archived trajectory data base");
		}
	}
}

/**
 * @brief Trim unvalidated tail points before archiving a trajectory.
 * @param[in] targetId ID of the tracklet.
 * @param[in] streamId Stream that owns the tracklet.
 * @param[in,out] tracklet Tracklet trimmed in place by removing unvalidated tail points.
 * @return True when the tracklet is valid to archive.
 */
bool NvTrajectoryManager::processTrajectory(
	const TargetId targetId, const NvMOTStreamId streamId, std::list<TargetInfo> &tracklet
)
{
	/// Search back to the last ACTIVE state point and remove the non-valid tracked points at the end of the tracklet.
	{
		auto it = tracklet.rbegin();
		while(it != tracklet.rend()
			&& it->trackerState != TRACKER_STATE::QUASIACTIVE
			&& it->trackerState != TRACKER_STATE::ACTIVE) // Archive only validated tracklets.
			{
			it = decltype(it)(tracklet.erase( std::next(it).base() ));
		}
	}

	if(tracklet.size() == 0)
		return false;

	/// Calculate the ratio of the valid match with detection.
	{
		int validMatchCnt = 0;
		for(auto it = tracklet.begin(); it != tracklet.end(); ++it)
		{
			if((*it).trackerState == TRACKER_STATE::ACTIVE
				|| (*it).trackerState == TRACKER_STATE::QUASIACTIVE
				|| (*it).trackerState == TRACKER_STATE::TENTATIVE)
			{
				validMatchCnt++;
			}
		}
		const float validMatchRatio = validMatchCnt/(float)tracklet.size();

		if(validMatchRatio < m_params.m_minMatchRatio4ValidTrackletDetermination)
		{
			if(m_params.m_debugVerboseLevel > 2)
			{
				cout << "[NvTrajectoryManager::processTrajectory] TargetID[" << targetId
						<< "] Tracklet will not be archived: tracklet validMatchRatio = " << validMatchRatio << endl;
			}

			return false;
		}
		else
		{
			if(m_params.m_debugVerboseLevel > 3)
			{
				cout << "[NvTrajectoryManager::processTrajectory] TargetID[" << targetId
						<< "] tracklet validMatchRatio = " << validMatchRatio << endl;
			}
		}
	}

	/// Get the projected trajectories and append it if it is a tail of a Vanishing Tracklet.
	if(m_params.m_enableVanishingTrackletReconstruction)
	{
		std::list<TargetInfo> projectedTracklet = createProjectedTrajectories(targetId, streamId, tracklet);

		if(m_params.m_debugVerboseLevel > 2)
		{
			cout << "[NvTrajectoryManager::processTrajectory] TargetID[" << targetId
					<< "] Check if Vanishing Tracklet: visibility change "
					<< projectedTracklet.begin()->visibility
					<< " -> " << projectedTracklet.rbegin()->visibility << endl;
		}

		if(projectedTracklet.begin()->visibility >= m_params.m_visibilityThreshold4VanishingTrackletDetection
				&& projectedTracklet.rbegin()->visibility < m_params.m_visibilityThreshold4VanishingTrackletDetection)
		{
			/// Get the tail tracklet.
			auto it = projectedTracklet.rbegin();
			while(it != projectedTracklet.rend()
					&& it->visibility < m_params.m_minVisibility4VanishingTrackletReconstruction)
			{
				it = decltype(it)(projectedTracklet.erase( std::next(it).base() ));
			}

				if(m_params.m_debugVerboseLevel > 2)
			{
				cout << "[NvTrajectoryManager::processTrajectory] TargetID[" << targetId
						<< "][Vanishing Tracklet] Add projected tracklets: frame "
						<< projectedTracklet.begin()->frameNum
						<< "->" << projectedTracklet.rbegin()->frameNum << endl;
			}

			/// Insert the projected tracklet at the end of the existing tracklet.
			tracklet.insert(tracklet.end(), projectedTracklet.begin(), projectedTracklet.end());
		}
	}

	return true;
}

/**
 * @brief Create projected trajectory points from a tracklet.
 * @param[in] targetId ID of the tracklet.
 * @param[in] streamId Stream that owns the tracklet.
 * @param[in] tracklet_ Tracklet to project.
 * @param[in] overrideShadowAge Shadow age to use when non-negative; a negative value counts INACTIVE frames at the tail.
 * @return Projected trajectory points.
 */
std::list<TargetInfo> NvTrajectoryManager::createProjectedTrajectories(
	const TargetId targetId, const NvMOTStreamId streamId, const std::list<TargetInfo> &tracklet_, int overrideShadowAge
)
{
	/// Deep copy.
	std::list<TargetInfo> tracklet = tracklet_;

	/// Count shadow frames stripped from the tail so the projection can extend to cover the new track's appearance frame.
	int shadowAge = 0;
	{
		auto it = tracklet.rbegin();
		while(it != tracklet.rend()
			&& it->trackerState != TRACKER_STATE::QUASIACTIVE
			&& it->trackerState != TRACKER_STATE::ACTIVE) // Archive only validated tracklets.
			{
			++shadowAge;
			it = decltype(it)(tracklet.erase( std::next(it).base() ));
		}
	}
	/// Override: for archive tracks the INACTIVE tail was stripped before archival, so the counted shadowAge would be 0.
	if(overrideShadowAge >= 0)
	{
		shadowAge = overrideShadowAge;
	}

	if((int)tracklet.size() > 1)
	{
		if(m_isOffline)
		{
			/// Recover to full tracklet.
			tracklet = TrajOps::generateFullTracklet(tracklet);
		}
		else
		{
			/// Recover to full tracklet for the portion.
			tracklet = TrajOps::generateFullTracklet(
				tracklet, tracklet.rbegin()->frameNum -  m_params.m_prepLength4TrajectoryProjection,
				tracklet.rbegin()->frameNum
			);
		}

		if (tracklet.empty())
		{
			return std::list<TargetInfo>();
		}

		/// Compute image-space velocity at prep-window end for live tracks only.
		if(overrideShadowAge < 0 && tracklet.size() >= 2)
		{
			const auto &last  = *tracklet.rbegin();
			const auto &prev  = *std::next(tracklet.rbegin());
			const int   dF    = (int)last.frameNum - (int)prev.frameNum;
			if(dF > 0)
			{
				const float cx1   = float(prev.tBbox.min()(0) + prev.tBbox.sizes()(0) * 0.5f);
				const float cy1   = float(prev.tBbox.min()(1) + prev.tBbox.sizes()(1) * 0.5f);
				const float cx2   = float(last.tBbox.min()(0) + last.tBbox.sizes()(0) * 0.5f);
				const float cy2   = float(last.tBbox.min()(1) + last.tBbox.sizes()(1) * 0.5f);
				const float imgVx = (cx2 - cx1) / (float)dF;
				const float imgVy = (cy2 - cy1) / (float)dF;
				if(imgVx > -0.5f && imgVx < 0.5f && imgVy > -0.5f && imgVy < 0.5f)
					m_stats.proj_near_zero_vel++;
			}
		}

		/// Print the debug information.
		if(m_params.m_debugVerboseLevel > 2)
		{
			cout << "[NvTrajectoryManager::createProjectedTrajectories] Recovered full tracklet (prior to make projection) = " << endl;
			for(auto it = tracklet.begin(); it != tracklet.end(); ++it)
			{
				cout << "- Frame: " << it->frameNum << ", (" << it->tBbox.min()(0) << ", " << it->tBbox.min()(1) << ")" << endl;
			}
		}

		/// Append the projected trajectory points based on a long-term Kalman Filter.
		{
			m_trajProjector.at(streamId)->reset();

			/// Prime the state estimator using some of the last trajectory points.
			auto it = tracklet.begin();
			if(m_isOffline)
			{
				/// In the offline case the tracklet contains the complete tracklet, so the state estimator uses only the last "m_prepLength4TrajectoryProjection" points.
				it = std::next(tracklet.begin(), (tracklet.size() - m_params.m_prepLength4TrajectoryProjection));
			}
			for(; it != tracklet.end(); ++it)
			{
				std::deque<TargetInfo> targets;

				auto it_next = std::next(it);
				if(it_next != tracklet.end())
				{
					/// Get all the interpolated target points.
					targets = TrajOps::generateInterpolatedTargets(*it, *it_next);
				}
				/// Add the current target point at the front.
				targets.push_front(*it);

				/// Update the state estimator with all the interpolated target points if there are.
				for(uint i=0; i<targets.size(); i++)
				{
					const std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool>> measurements{
						std::make_tuple(
							CV_FUNCS::convertBboxToZ(targets[i].tBbox),
							-Eigen::Matrix2d::Identity(), targets[i].matchedDetObjInfo != nullptr
						)
					};

					/// Update the state estimator for longer-term prediction.
					m_trajProjector.at(streamId)->update(measurements);

					if(m_params.m_debugVerboseLevel > 4)
					{
						cout << "[NvTrajectoryManager::createProjectedTrajectories] TargetID[" << targetId << "] re-run State Estimator for trajPt (frameNum, lastFrameNum) = (" << targets[i].frameNum << ", " << tracklet.rbegin()->frameNum << ")" << endl;
					}
				}

			}

			TargetInfo trajPt = *tracklet.rbegin();
			if(!m_isOffline)
			{
				/// Clears the tracklet so online matching keeps only the projected points.
				tracklet.clear();
			}

			/// Append projected points, extended by shadowAge so the trajectory covers the new track's appearance frame.
			const int projectionHorizon = m_params.m_trajectoryProjectionLength + shadowAge;
			for(int i=0; i<projectionHorizon; i++)
			{
				/// Generate projections/predictions with no measurement.
				m_trajProjector.at(streamId)->update();

				NvMotDataTypes::Rect2f projBbox = CV_FUNCS::convertXToBbox(m_trajProjector.at(streamId)->getStates());
				const float projW = float(projBbox.max()(0) - projBbox.min()(0));
				const float projH = float(projBbox.max()(1) - projBbox.min()(1));
				NvMotDataTypes::Rect2f newTargetBbox = (projW >= 1.0f && projH >= 1.0f) ? projBbox : trajPt.tBbox;
				const float visibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params.m_srcImageBox);

				/// Update the target states with the estimated state back.
				trajPt.update(trajPt.frameNum + 1, newTargetBbox, visibility);
				trajPt.matchedDetObjInfo = nullptr;
				trajPt.shadowTrackingAge = 0;
				trajPt.trackerState = TRACKER_STATE::PROJECTED;

				tracklet.push_back(trajPt);
			}

			return std::move(tracklet);
		}
	}

	return std::list<TargetInfo>();
}

/**
 * @brief Update per-stream trajectory state when streams are added or removed.
 * @param[in] streamIdsToAdd Stream identifiers to add.
 * @param[in] streamIdsToDel Stream identifiers to remove.
 */
void NvTrajectoryManager::updateStreamIds(
	const std::vector<NvMOTStreamId>& streamIdsToAdd, const std::vector<NvMOTStreamId>& streamIdsToDel
)
{
	/// Add new stream ids if not existed.
	for (auto streamId : streamIdsToAdd)
	{
		if (m_projectedTrajectories.find(streamId) == m_projectedTrajectories.end() || !(m_projectedTrajectories.at(streamId)))
		{
			m_projectedTrajectories[streamId] = std::make_shared<ProjectedTrajectoryList>();
		}
		if (m_trajectoryArchive.find(streamId) == m_trajectoryArchive.end() || (!m_trajectoryArchive.at(streamId)))
		{
			m_trajectoryArchive[streamId] = std::make_shared<Trajectory4ArchiveList>();
		}
		if (m_trajProjector.find(streamId) == m_trajProjector.end() || (!m_trajProjector.at(streamId)))
		{
			m_trajProjector[streamId] = std::make_shared<SimpleEstimator2>(m_configFilePath, 6, 4, true);
		}
		if (m_trackedObjectsReidInd.find(streamId) == m_trackedObjectsReidInd.end() || (!m_trackedObjectsReidInd.at(streamId)))
		{
			m_trackedObjectsReidInd[streamId] = std::make_shared<TrackedObjectsReidIndList>();
		}
	}

	/// Delete stream ids.
	for (auto streamId : streamIdsToDel)
	{
		if (m_projectedTrajectories.find(streamId) != m_projectedTrajectories.end())
		{
			m_projectedTrajectories.erase(streamId);
		}
		if (m_trajectoryArchive.find(streamId) != m_trajectoryArchive.end())
		{
			m_trajectoryArchive.erase(streamId);
		}
		if (m_trajProjector.find(streamId) != m_trajProjector.end())
		{
			m_trajProjector.erase(streamId);
		}
		if (m_trackedObjectsReidInd.find(streamId) != m_trackedObjectsReidInd.end())
		{
			m_trackedObjectsReidInd.erase(streamId);
		}
	}
}

/**
 * @brief Update dynamic parameters during runtime.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void NvTrajectoryManager::updateParams(const YAML::Node &fn)
{
	m_params.read(fn);
}

/**
 * @brief Set source-frame geometry used for trajectory output.
 * @param[in] originalSrcImageBox Original source image bounds.
 * @param[in] srcImageBox Tracker source image bounds.
 */
void NvTrajectoryManager::setSrcFrameInfo(
	const NvMotDataTypes::Rect2f originalSrcImageBox, const NvMotDataTypes::Rect2f srcImageBox
)
{
	m_params.m_originalSrcImageBox = originalSrcImageBox;
	m_params.m_srcImageBox = srcImageBox;
	m_params.m_scale2OriginalSourceCoordinates(0) = originalSrcImageBox.max()(0)/(float)srcImageBox.max()(0);
	m_params.m_scale2OriginalSourceCoordinates(1) = originalSrcImageBox.max()(1)/(float)srcImageBox.max()(1);
}

/**
 * @brief Set the per-stream multi-view associators.
 * @param[in] multiViewAssociators Per-stream multi-view associators.
 */
void NvTrajectoryManager::setMultiViewAssociators(
	std::map<NvMOTStreamId, std::shared_ptr<IMultiViewAssociator>> multiViewAssociators
)
{
	m_multiViewAssociators = multiViewAssociators;
}

/**
 * @brief Remove expired archived trajectories for all input streams.
 * @param[in] inputMeta Current input metadata by stream.
 */
void NvTrajectoryManager::cleanUpArchivedTrajectories(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta)
{
	for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
	{
		const NvMOTStreamId streamId = streamIt->first;
		const NvMOTFrame *frameMeta = streamIt->second;
		const uint32_t currFrameNum = frameMeta->frameNum;

		cleanUpArchivedTrajectories_perStream(streamId, currFrameNum);
	}
}

/** @brief Purge all archived trajectories. */
void NvTrajectoryManager::purgeArchivedTrajectories()
{
	for(auto streamIt = m_trajectoryArchive.begin(); streamIt != m_trajectoryArchive.end();)
	{
		const NvMOTStreamId streamId = streamIt->first;

		auto streamIt_next = std::next(streamIt);
		cleanUpArchivedTrajectories_perStream(streamId, 0, true);
		streamIt = streamIt_next;
	}

	if(m_trajectoryArchive.size() != 0)
	{
		CLOG("Fail to purge archived trajectories");
	}
}

/**
 * @brief Remove expired archived trajectories for one stream.
 * @param[in] streamId Source stream identifier.
 * @param[in] currFrameNum Current frame number.
 * @param[in] doPurge Whether to remove every archived trajectory.
 */
void NvTrajectoryManager::cleanUpArchivedTrajectories_perStream(
	const NvMOTStreamId streamId, const uint32_t currFrameNum, const bool doPurge
)
{
	if(m_trajectoryArchive.find(streamId) == m_trajectoryArchive.end() || m_trajectoryArchive.at(streamId)==nullptr)
		return;

	std::map<uint64_t, Trajectory4Archive> &tracklets = *m_trajectoryArchive.at(streamId);

	for(auto targetIt = tracklets.begin(); targetIt != tracklets.end();)
	{
		const uint64_t targetId = targetIt->first;
		const uint32_t frameNumTerminated = targetIt->second.frameNumTerminated;
		const bool isReported = targetIt->second.isReported;

		std::set<uint64_t> peersToBeDeleted;

		if(doPurge || isReported || currFrameNum > frameNumTerminated + m_params.m_maxAge4TrackletArchiving)
		{

			if(!checkDuplicateTracklet(targetId, tracklets, peersToBeDeleted))
			{
				/// Write into a file (or transmit elsewhere) if not merged into another tracklet.
				if(m_params.m_dumpTrajectoryIntoFile)
				{
					dumpTrajectoryIntoFile(streamId, targetId, targetIt->second);
				}

				if(m_params.m_debugVerboseLevel > 3)
				{
					const uint32_t lastFrameMatched = targetIt->second.lastFrameMatched;

					cout << "[NvTrajectoryManager::cleanUpArchivedTrajectories_perStream()] Stream[" << (streamId>>32) << "] frameNum[" << currFrameNum
							<< "] Deleted an archived tracklet that is too old, yet valid: Target ID[" << targetId
							<< "] last frame was " << lastFrameMatched << ", now # of archived tracklets = " << tracklets.size() << endl;

				}
			}
			else
			{

				if(m_params.m_debugVerboseLevel > 3)
				{
					const uint32_t lastFrameMatched = targetIt->second.lastFrameMatched;

					cout << "[NvTrajectoryManager::cleanUpArchivedTrajectories_perStream()] Stream[" << (streamId>>32) << "] frameNum[" << currFrameNum
							<< "] Deleted an archived tracklet too old, and duplicate: Target ID[" << targetId
							<< "] last frame was " << lastFrameMatched << ", now # of archived tracklets = " << tracklets.size() << endl;
				}
			}

			/// Delete duplicate peers.
			if(peersToBeDeleted.size() > 0)
			{
				for(auto it=peersToBeDeleted.begin(); it != peersToBeDeleted.end(); ++it)
				{
					tracklets.erase(*it);
				}
			}

			/// Actually delete from the list.
			targetIt = tracklets.erase(targetIt);
		}
		else
		{
			++targetIt;
		}
	}
}

/**
 * @brief Dump archived trajectory data to the configured output file.
 * @param[in] streamId Stream that owns the trajectory.
 * @param[in] targetId ID of the trajectory to dump.
 * @param[in] traj Archived trajectory to write.
 */
void NvTrajectoryManager::dumpTrajectoryIntoFile(
	const NvMOTStreamId streamId, const TargetId targetId, const Trajectory4Archive &traj
)
{
	/// < MOT Format >
	/// For detection/tracking result, the format is:
	///     <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
	/// For ground-truth result, the format is:
	///     <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <obj_class>, <visibility>
	///
	/// < KITTI Tracking Data Format >
	///
	/// #Values    Name      Description
	/// ----------------------------------------------------------------------------
	///    1    frame        Frame within the sequence where the object appearers
	///    1    track id     Unique tracking id of this object within this sequence
	///    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
	///                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
	///                      'Misc' or 'DontCare'
	///    1    truncated    Integer (0,1,2) indicating the level of truncation.
	///                      Note that this is in contrast to the object detection
	///                      benchmark where truncation is a float in [0,1].
	///    1    occluded     Integer (0,1,2,3) indicating occlusion state:
	///                      0 = fully visible, 1 = partly occluded
	///                      2 = largely occluded, 3 = unknown
	///    1    alpha        Observation angle of object, ranging [-pi..pi]
	///    4    bbox         2D bounding box of object in the image (0-based index):
	///                      contains left, top, right, bottom pixel coordinates
	///    3    dimensions   3D object dimensions: height, width, length (in meters)
	///    3    location     3D object location x,y,z in camera coordinates (in meters)
	///    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
	///    1    score        Only for results: Float, indicating confidence in
	///                      detection, needed for p/r curves, higher is better.

	/// Opens the dump file for append after the first create.
	if(m_params.m_trajectoryDumpFileName.size() > 0)
	{
		const int streamIntId = streamId >> 32;
		const std::string fileName =  m_params.m_trajectoryDumpFileName + std::to_string(streamIntId) + ".txt";

		if(m_isFileTrajectoryDumpCreated.find(streamId) == m_isFileTrajectoryDumpCreated.end() || !m_isFileTrajectoryDumpCreated[streamId])
		{
			m_fileTrajectoryDump[streamId].open(fileName, std::ofstream::out);
			m_isFileTrajectoryDumpCreated[streamId] = true;
		}
		else
		{
			m_fileTrajectoryDump[streamId].open(fileName, std::ofstream::app);
		}

		if(!m_fileTrajectoryDump[streamId])
		{
			throw std::runtime_error(
				"[NvTrajectoryManager::dumpTrajectoryIntoFile()] fileTrajectoryDump is not opened !!!"
			);
		}
	}
	else
	{
		throw std::runtime_error(
			"[NvTrajectoryManager::dumpTrajectoryIntoFile()] No trajectoryDumpFileName is provided !!!"
		);
	}

	/// Write the data into file.
	/// The counter holds the consecutive inactive frames dumped in the current gap.
	/// It resets when a non-inactive frame is output and bounds the inactive dump to m_maxInactiveDumpFrames per gap.
	/// A value of 0 disables the cap.
	int consecutiveInactiveDumped = 0;
	for(auto it = traj.tracklet.begin(); it != traj.tracklet.end(); ++it)
	{
		std::deque<TargetInfo> trajPts;

		/// Check the tracklets are sub-sampled, and if so, generate the interpolated track points.
		if(std::next(it) != traj.tracklet.end() && std::next(it)->frameNum - it->frameNum > 1)
		{
			/// Get all the interpolated target points.
			trajPts = TrajOps::generateInterpolatedTargets(*it, *std::next(it));
		}
		/// Add the current target point at the front.
		trajPts.push_front(*it);

		for(uint i=0; i<trajPts.size(); i++)
		{
			const TargetInfo &tInfo = trajPts[i];

			/// Reset gap counter on any non-inactive frame.
			const bool isInactive = (tInfo.trackerState == TRACKER_STATE::INACTIVE);
			if(!isInactive) consecutiveInactiveDumped = 0;

			/// Inactive is allowed when flag is set AND (no cap OR cap not yet reached).
			const bool inactiveAllowed = m_params.m_dumpTrajectoryInactive && isInactive &&
				(m_params.m_maxInactiveDumpFrames <= 0 || consecutiveInactiveDumped < m_params.m_maxInactiveDumpFrames);

			/// Make sure to not write the projected points.
			if (tInfo.visibility >= m_params.m_minVisibility4TrajDump && (tInfo.trackerState == TRACKER_STATE::ACTIVE ||
				(m_params.m_dumpTrajectoryTentative && tInfo.trackerState == TRACKER_STATE::TENTATIVE) ||
				(m_params.m_dumpTrajectoryQuasiActive && tInfo.trackerState == TRACKER_STATE::QUASIACTIVE) ||
				inactiveAllowed))
			{
				if(inactiveAllowed)
				{
					consecutiveInactiveDumped++;
					/// Count actually-dumped inactive frames (confirms TP).
					m_stats.inact_dumped++;
				}

				/// Select from tBbox or origBbox according to the config.
				auto bbox = tInfo.tBbox;
				if (m_params.m_dumpDetectorBboxes && !isInactive) bbox = tInfo.origBbox;
				m_fileTrajectoryDump[streamId] << tInfo.frameNum + 1 << ","
						<< (targetId & 0xffff) << ","
						<< (int)(bbox.min()(0)*m_params.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
						<< (int)(bbox.min()(1)*m_params.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
						<< (int)(bbox.sizes()(0)*m_params.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
						<< (int)(bbox.sizes()(1)*m_params.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
						<< std::setprecision(3) << std::fixed
						<< tInfo.confidence << ","
						<< tInfo.pt_world_feet(0) << ","
						<< tInfo.pt_world_feet(1) << ","
						<< "-1,"
						<< tInfo.classId << ","
						<< tInfo.visibility << ","
						<< int(tInfo.pt_img_feet(0)*m_params.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
						<< int(tInfo.pt_img_feet(1)*m_params.m_scale2OriginalSourceCoordinates(1) + 0.5f);

				/// Save convex hull.
				if (tInfo.convexHull.size() == 0)
				{
					m_fileTrajectoryDump[streamId] << ",-1";
				}
				else
				{
					for (uint i=0; i<tInfo.convexHull.size(); i++)
					{
						m_fileTrajectoryDump[streamId] << (i == 0 ? "," : "|")
							<< int(tInfo.convexHull.at(i)(0)*m_params.m_scale2OriginalSourceCoordinates(0) + 0.5f)
							<< "|"
							<< int(tInfo.convexHull.at(i)(1)*m_params.m_scale2OriginalSourceCoordinates(1) + 0.5f);
					}
				}

				/// Save 2.5D pose.
				if (tInfo.pose25d.size() == 0)
				{
					m_fileTrajectoryDump[streamId] << ",-1";
				}
				else
				{
					for (uint i=0; i<tInfo.pose25d.size()/4; i++)
					{
						m_fileTrajectoryDump[streamId] << (i == 0 ? "," : "|")
							<< int(tInfo.pose25d.at(i*4)*m_params.m_scale2OriginalSourceCoordinates(0) + 0.5f)
							<< "|"
							<< int(tInfo.pose25d.at(i*4+1)*m_params.m_scale2OriginalSourceCoordinates(1) + 0.5f)
							<< "|" << tInfo.pose25d.at(i*4+2) << "|" << tInfo.pose25d.at(i*4+3);
					}
				}

				/// Save 3D pose.
				if (tInfo.pose3d.size() == 0)
				{
					m_fileTrajectoryDump[streamId] << ",-1";
				}
				else
				{
					for (uint i=0; i<tInfo.pose3d.size()/3; i++)
					{
						m_fileTrajectoryDump[streamId] << (i == 0 ? "," : "|")
							<< tInfo.pose3d.at(i*3) << "|" << tInfo.pose3d.at(i*3+1) << "|" << tInfo.pose3d.at(i*3+2);
					}
				}

				m_fileTrajectoryDump[streamId] << endl;
			}
			else if (isInactive && m_params.m_dumpTrajectoryInactive &&
				m_params.m_maxInactiveDumpFrames > 0 &&
				consecutiveInactiveDumped >= m_params.m_maxInactiveDumpFrames)
			{
				/// Count cap-skipped inactive frames: those that would be dumped if maxInactiveDumpFrames were higher.
				if      (consecutiveInactiveDumped < 30) m_stats.inact_frames_21_30++;
				else if (consecutiveInactiveDumped < 40) m_stats.inact_frames_31_40++;
				else                                     m_stats.inact_frames_41plus++;
				consecutiveInactiveDumped++;
			}
		}
	}

	m_fileTrajectoryDump[streamId].close();
}

/**
 * @brief Check whether a tracklet duplicates another archived tracklet.
 * @param[in] targetId ID of the tracklet to check.
 * @param[in,out] tracklets Archived tracklets for the stream; a duplicate may be fused into its peer.
 * @param[in,out] peersToBeDeleted Peer IDs already marked for deletion; a duplicate peer is inserted here.
 * @return True when this tracklet is a duplicate of a peer.
 */
bool NvTrajectoryManager::checkDuplicateTracklet(
	const TargetId targetId, std::map<TargetId, Trajectory4Archive> &tracklets, std::set<TargetId> &peersToBeDeleted
)
{
	const std::list<TargetInfo> &tracklet = tracklets.at(targetId).tracklet;

	/// Calculate the tracklet duplicate scores.
	for(auto tIt = tracklets.begin(); tIt != tracklets.end(); ++tIt)
	{
		const uint64_t peerId = tIt->first;
		const std::list<TargetInfo> &peerTracklet = tIt->second.tracklet;

		if(peerId == targetId
				|| peersToBeDeleted.find(peerId) != peersToBeDeleted.end())
		{
			continue;
		}

		const int duplicateBboxCnt = calculateTrackletInclusionScore(tracklet, peerTracklet);

		float inclusionRatioSelf2Peer = duplicateBboxCnt/(float)tracklet.size();
		float inclusionRatioPeer2Self = duplicateBboxCnt/(float)peerTracklet.size();

		if(m_params.m_debugVerboseLevel > 4)
		{
			if(duplicateBboxCnt > 0)
			{
				cout << "[NvTrajectoryManager::checkDuplicateTracklet()] TargetID[" << targetId
						<< "] to Peer ID[" << peerId << "]"
						<< " inclusionRatio (Self2Peer, Peer2Self) = (" << inclusionRatioSelf2Peer << ", "
						<< inclusionRatioPeer2Self << ")" << endl;
			}
		}

		if(inclusionRatioSelf2Peer >= m_params.m_minInclusionRatio4DuplicateTrackletRemoval
				|| inclusionRatioPeer2Self >= m_params.m_minInclusionRatio4DuplicateTrackletRemoval)
		{
			if(inclusionRatioSelf2Peer > inclusionRatioPeer2Self)
			{
				/// This tracklet is duplicate to a part of a peer tracklet.
				fuseArchivedTracklets(peerId, targetId, tracklets);

				if(m_params.m_debugVerboseLevel > 2)
				{
					cout << "[NvTrajectoryManager::checkDuplicateTracklet()] TargetID[" << targetId
							<< "] Found this is a duplicate tracklet to Peer ID[" << peerId
							<< "] inclusion score = " << inclusionRatioSelf2Peer << endl;
				}

				return true;
			}
			else
			{
				fuseArchivedTracklets(targetId, peerId, tracklets);

				if(m_params.m_debugVerboseLevel > 2)
				{
					cout << "[NvTrajectoryManager::checkDuplicateTracklet()] TargetID[" << targetId
							<< "] Found a duplicate peer tracklet: Peer ID[" << peerId
							<< "] inclusion score = " << inclusionRatioPeer2Self << endl;
				}

				peersToBeDeleted.insert(peerId);
			}
		}
	}

	return false;
}

/**
 * @brief Fuse a peer tracklet into an archived target tracklet.
 * @param[in] targetId ID of the tracklet that is kept.
 * @param[in] peerId ID of the peer tracklet that is merged in.
 * @param[in,out] tracklets Archived tracklets; the kept tracklet is replaced by the fused result.
 */
void NvTrajectoryManager::fuseArchivedTracklets(
	const TargetId targetId, const TargetId peerId, std::map<TargetId, Trajectory4Archive> &tracklets
)
{
	uint32_t &lastFrameMatched = tracklets.at(targetId).lastFrameMatched;
	std::list<TargetInfo> &selfTracklet = tracklets.at(targetId).tracklet;
	std::list<TargetInfo> &peerTracklet = tracklets.at(peerId).tracklet;
	std::list<TargetInfo> newTracket;

	if (selfTracklet.size() <= 0 || peerTracklet.size() <= 0)
	{
		CLOG("Self tracklet and peer tracklet should not be empty");
		return;
	}

	const uint32_t firstFrameNum_self = selfTracklet.begin()->frameNum;
	const uint32_t lastFrameNum_self = selfTracklet.rbegin()->frameNum;
	const uint32_t firstFrameNum_peer = peerTracklet.begin()->frameNum;
	const uint32_t lastFrameNum_peer = peerTracklet.rbegin()->frameNum;

	if(m_params.m_debugVerboseLevel > 2)
	{
		cout << "[NvTrajectoryManager::fuseArchivedTracklets()] TargetID[" << targetId
				<< "] FrameNum (first, last) = (" << firstFrameNum_self << ", " << lastFrameNum_self
				<< "), Peer ID[" << peerId
				<< "] FrameNum (first, last) = (" << firstFrameNum_peer << ", " << lastFrameNum_peer
				<< ")" << endl;
	}

	/// Extended range.
	const uint32_t firstFrameNum = std::min(firstFrameNum_self, firstFrameNum_peer);
	const uint32_t lastFrameNum = std::max(lastFrameNum_self, lastFrameNum_peer);

	/// Start from the earliest frameNum and fuse frame-by-frame.
	auto it_self = selfTracklet.begin();
	auto it_peer = peerTracklet.begin();

	uint32_t currFrameNum = firstFrameNum;

	/// Get to the start position.
	while(it_self->frameNum < currFrameNum && std::next(it_self) != selfTracklet.end())
	{
		++it_self;
	}

	while(it_peer->frameNum < currFrameNum && std::next(it_peer) != peerTracklet.end())
	{
		++it_peer;
	}

	/// Iterate until both tracklets are all covered.
	while(it_self != selfTracklet.end() || it_peer != peerTracklet.end())
	{
		if(it_self != selfTracklet.end() && it_peer != peerTracklet.end())
		{
			currFrameNum = std::min(it_self->frameNum, it_peer->frameNum);
		}
		else if(it_self != selfTracklet.end())
		{
			currFrameNum = it_self->frameNum;
		}
		else
		{
			currFrameNum = it_peer->frameNum;
		}

		if(it_self != selfTracklet.end() && it_self->frameNum == currFrameNum
				&& it_peer != peerTracklet.end() && it_peer->frameNum == currFrameNum)
		{
			if(it_self->matchedDetObjInfo != nullptr || it_peer->matchedDetObjInfo == nullptr)
			{
				newTracket.push_back(*it_self);
			}
			else
			{
				newTracket.push_back(*it_peer);
			}

			it_self++;
			it_peer++;
		}
		else if(it_self != selfTracklet.end() && it_self->frameNum == currFrameNum)
		{
			newTracket.push_back(*it_self);

			it_self++;
		}
		else if(it_peer != peerTracklet.end() && it_peer->frameNum == currFrameNum)
		{
			newTracket.push_back(*it_peer);

			it_peer++;
		}
		else
		{
			std::stringstream msg;
			msg << "[NvTrajectoryManager::fuseArchivedTracklets][ERROR] Invalid case!!"
					<< "newTracket Age = " << currFrameNum - firstFrameNum
					<< ", tracketLength (self, peer) = (" << selfTracklet.size() << ", " << peerTracklet.size() << "), "
					<< "(first, last) frameNum = ("
					<< firstFrameNum << ", " << lastFrameNum << "), "
					<< "(frameNum, selfFrameNum, peerFrameNum) = ("
					<< currFrameNum << ", " << it_self->frameNum << ", " << it_peer->frameNum
					<< ")";

			CLOG(msg.str().c_str());
			continue;
		}

		newTracket.rbegin()->age = currFrameNum - firstFrameNum;

		if(newTracket.rbegin()->matchedDetObjInfo != nullptr)
		{
			lastFrameMatched = currFrameNum;
		}

	}

	if(m_params.m_debugVerboseLevel > 2)
	{
		cout << "           ---> TargetID[" << targetId
				<< "] FrameNum (first, last) = (" << newTracket.begin()->frameNum
				<< ", " << newTracket.rbegin()->frameNum
				<< ")" << endl;
	}

	selfTracklet = std::move(newTracket);
}

/**
 * @brief Count duplicate bounding boxes between two tracklets.
 * @param[in] tracklet Tracklet being compared.
 * @param[in] peerTracklet Peer tracklet being compared.
 * @return Number of duplicate boxes, or -1 when a tracklet is empty or the tracklets do not overlap in time.
 */
int NvTrajectoryManager::calculateTrackletInclusionScore(
	const std::list<TargetInfo> &tracklet, const std::list<TargetInfo> &peerTracklet
)
{
	if (tracklet.size() <= 0 || peerTracklet.size() <= 0)
	{
		CLOG("Tracklet and peer tracklet should not be empty");
		return -1;
	}

	const uint32_t firstFrameNum_self = tracklet.begin()->frameNum;
	const uint32_t lastFrameNum_self = tracklet.rbegin()->frameNum;
	const uint32_t firstFrameNum_peer = peerTracklet.begin()->frameNum;
	const uint32_t lastFrameNum_peer = peerTracklet.rbegin()->frameNum;

	/// If the tracklets are disjoint in time, then return -1.
	if(lastFrameNum_self < firstFrameNum_peer
			|| lastFrameNum_peer < firstFrameNum_self)
	{
		return -1;
	}
	else
	{
		/// Common range.
		const uint32_t firstFrameNum = std::max(firstFrameNum_self, firstFrameNum_peer);
		const uint32_t lastFrameNum = std::min(lastFrameNum_self, lastFrameNum_peer);

		auto it_self = tracklet.begin();
		auto it_peer = peerTracklet.begin();

		/// Move the iterators to the common starting point.
		{
			while(it_self != tracklet.end())
			{
				if(it_self->frameNum == firstFrameNum)
					break;
				else
					it_self++;
			}

			while(it_peer != peerTracklet.end())
			{
				if(it_peer->frameNum == firstFrameNum)
					break;
				else
					it_peer++;
			}
		}

		/// Counts duplicate boxes on frames both tracklets still contain.
		int duplicateBboxCnt = 0;
		do
		{
			float iou = CV_FUNCS::computeIOU(it_self->tBbox, it_peer->tBbox);

			if(iou >= m_params.m_minIou4DuplicateTrackletRemoval)
			{
				duplicateBboxCnt++;
			}

			++it_self;
			++it_peer;
		} while(it_self != tracklet.end() && it_peer != peerTracklet.end()
				&& it_self->frameNum <= lastFrameNum && it_peer->frameNum <= lastFrameNum);

		return duplicateBboxCnt;
	}
}

/** @brief Initialize trajectory-manager parameters with default values. */
NvTrackerParamsTrajectoryManager::NvTrackerParamsTrajectoryManager()
{
	m_debugVerboseLevel = 0;
	m_useUniqueID = false;
	m_targetIDOffset = 0;
	m_enableReAssoc = false;
	m_checkClassMatch = true;
	m_enable3DReAssoc = false;
	m_maxFrameGap4WorldDistReAssoc = 0;

	m_minMatchingScore4Overall = 0.4f;
	m_minTrackletMatchingScore = 0.4f;
	m_minMatchingScore4ReidSimilarity = 0.8f;

	m_matchingScoreWeight4TrackletSimilarity = 1.0f;
	m_matchingScoreWeight4ReidSimilarity = 0.f;

	m_maxAge4TrackletArchiving = 100;
	m_minTrajectoryLength4Projection = 20;
	m_prepLength4TrajectoryProjection = 10;
	m_trajectoryProjectionLength = 90;

	m_trajectoryProjectionProcessNoiseScale = 1.0f;
	m_trajectoryProjectionMeasurementNoiseScale = 1.0f;

	m_maxAngle4TrackletMatching = 40;
	m_minSpeedSimilarity4TrackletMatching = 0.3f;
	m_minBboxSizeSimilarity4TrackletMatching = 0.6f;
	m_maxTrackletMatchingTimeSearchRange = 20;
	m_trackletSpacialSearchRegionScale = 0.f;
	m_minSpeed4AngleChecking = 1.0f;
	m_maxIou4AngleChecking = 0.85f;

	m_reidExtractionInterval = 0;

	m_minMatchRatio4ValidTrackletDetermination = 0.2f;

	m_minIou4DuplicateTrackletRemoval = 1.0;
	m_minInclusionRatio4DuplicateTrackletRemoval = 1.0;

	m_enableVanishingTrackletReconstruction = false;
	m_visibilityThreshold4VanishingTrackletDetection = 0.5f;
	m_minVisibility4VanishingTrackletReconstruction = 0.1f;

	m_dumpTrajectoryIntoFile = false;
	m_dumpDetectorBboxes = true;
	m_dumpTrajectoryTentative = true;
	m_dumpTrajectoryQuasiActive = true;
	m_dumpTrajectoryInactive = true;
	m_maxInactiveDumpFrames = 0;
	m_minVisibility4TrajDump = 0.f;
	m_trajectoryDumpFileName = "trajDump_Stream_";
	m_scale2OriginalSourceCoordinates << 0, 0;
	m_maxWorldDist4ArchiveReAssoc = 0.0f;
	m_maxWorldDist4ArchiveReAssocCS = 0.0f;
	m_minBboxSizeSimilarity4ReAssocCS = 0.0f;
	m_maxNmCascadeWorldDist = 99.0f;
	m_nmCascadeMinCosSim  = 0.59f;
	m_nmCascadeMaxCosSim  = 0.85f;
	m_nmCascadeDecayRate  = 2.12f;
	m_maxWsExt25WorldDist  = 3.0f;
	m_wsExt25MinCosSim     = 0.90f;
	m_wsExt25StrictCosSim  = 0.93f;
	m_maxWorldDist4ArchiveReAssocWS = 0.0f;

	m_offlineParams.m_maxProjTrajectoryLength4Matching = 5;
	m_offlineParams.m_minOverlappedTrajectoryLength4Matching = 3;
	m_offlineParams.m_iouAvgScaleFactor = 0.01f;

}

/**
 * @brief Initialize trajectory-manager parameters from a configuration module.
 * @param[in] configFilePath The path to a configuration file.
 * @param[in] moduleName The configuration module name.
 */
NvTrackerParamsTrajectoryManager::NvTrackerParamsTrajectoryManager(
	const std::string &configFilePath, const std::string moduleName
)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
	const YAML::Node &baseNode = configRoot["BaseConfig"];
	if (baseNode && baseNode["checkClassMatch"])
		m_checkClassMatch = (bool)baseNode["checkClassMatch"].as<int>();
}

/**
 * @brief Initialize trajectory-manager parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
NvTrackerParamsTrajectoryManager::NvTrackerParamsTrajectoryManager(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Update trajectory-manager parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void NvTrackerParamsTrajectoryManager::read(const YAML::Node &fn)
{
	if(!m_isInit)
	{
		/// Set the default values.
		*this = NvTrackerParamsTrajectoryManager();
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
		else if (paramKey == "useUniqueID")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_useUniqueID = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_useUniqueID is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "targetIDOffset")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_targetIDOffset = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_targetIDOffset is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "enableReAssoc")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_enableReAssoc = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_enableReAssoc is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "enable3DReAssoc" || paramKey == "enableArchiveReAssoc")
		{
			if(!m_isInit) m_enable3DReAssoc = fit->second.as<int>();
		}
		else if (paramKey == "maxFrameGap4WorldDistReAssoc")
		{
			m_maxFrameGap4WorldDistReAssoc = fit->second.as<int>();
		}
		else if (paramKey == "checkClassMatchForReAssoc")
		{
			CLOG(
				std::string("!!![DEPRECATED] TrajectoryManagement:checkClassMatchForReAssoc is ignored. Use BaseConfig:checkClassMatch instead.").c_str()
			);
		}
		else if (paramKey == "minMatchingScore4Overall")
		{
			m_minMatchingScore4Overall = fit->second.as<float>();
		}
		else if (paramKey == "maxAge4TrackletArchiving")
		{
			m_maxAge4TrackletArchiving = fit->second.as<float>();
		}
		else if (paramKey == "minTrajectoryLength4Projection")
		{
			m_minTrajectoryLength4Projection = fit->second.as<float>();
		}
		else if (paramKey == "prepLength4TrajectoryProjection")
		{
			m_prepLength4TrajectoryProjection = fit->second.as<float>();
		}
		else if (paramKey == "trajectoryProjectionLength")
		{
			m_trajectoryProjectionLength = fit->second.as<float>();
		}
		else if (paramKey == "trajectoryProjectionProcessNoiseScale")
		{
			m_trajectoryProjectionProcessNoiseScale = fit->second.as<float>();
		}
		else if (paramKey == "trajectoryProjectionMeasurementNoiseScale")
		{
			m_trajectoryProjectionMeasurementNoiseScale = fit->second.as<float>();
		}
		else if (paramKey == "minTrackletMatchingScore")
		{
			m_minTrackletMatchingScore = fit->second.as<float>();
		}
		else if (paramKey == "matchingScoreWeight4TrackletSimilarity")
		{
			m_matchingScoreWeight4TrackletSimilarity = fit->second.as<float>();
		}
		else if (paramKey == "maxAngle4TrackletMatching")
		{
			m_maxAngle4TrackletMatching = fit->second.as<float>();
		}
		else if (paramKey == "minSpeedSimilarity4TrackletMatching")
		{
			m_minSpeedSimilarity4TrackletMatching = fit->second.as<float>();
		}
		else if (paramKey == "minBboxSizeSimilarity4TrackletMatching")
		{
			m_minBboxSizeSimilarity4TrackletMatching = fit->second.as<float>();
		}
		else if (paramKey == "minMatchRatio4ValidTrackletDetermination")
		{
			m_minMatchRatio4ValidTrackletDetermination = fit->second.as<float>();
		}
		else if (paramKey == "minIou4DuplicateTrackletRemoval")
		{
			m_minIou4DuplicateTrackletRemoval = fit->second.as<float>();
		}
		else if (paramKey == "minInclusionRatio4DuplicateTrackletRemoval")
		{
			m_minInclusionRatio4DuplicateTrackletRemoval = fit->second.as<float>();
		}
		else if (paramKey == "enableVanishingTrackletReconstruction")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_enableVanishingTrackletReconstruction = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_enableVanishingTrackletReconstruction is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "visibilityThreshold4VanishingTrackletDetection")
		{
			m_visibilityThreshold4VanishingTrackletDetection = fit->second.as<float>();
		}
		else if (paramKey == "minVisibility4VanishingTrackletReconstruction")
		{
			m_minVisibility4VanishingTrackletReconstruction = fit->second.as<float>();
		}
		else if (paramKey == "trackletSpacialSearchRegionScale")
		{
			m_trackletSpacialSearchRegionScale = fit->second.as<float>();
		}
		else if (paramKey == "maxTrackletMatchingTimeSearchRange")
		{
			m_maxTrackletMatchingTimeSearchRange = fit->second.as<float>();
		}
		else if (paramKey == "maxWorldDist4ArchiveReAssoc")
		{
			m_maxWorldDist4ArchiveReAssoc = fit->second.as<float>();
		}
		else if (paramKey == "maxWorldDist4ArchiveReAssocCS")
		{
			/// Cross-stream override.
			m_maxWorldDist4ArchiveReAssocCS = fit->second.as<float>();
		}
		else if (paramKey == "minBboxSizeSimilarity4ReAssocCS")
			m_minBboxSizeSimilarity4ReAssocCS = std::max(0.0f, fit->second.as<float>());
		else if (paramKey == "maxNmCascadeWorldDist")
			m_maxNmCascadeWorldDist = fit->second.as<float>();
		else if (paramKey == "nmCascadeMinCosSim")   m_nmCascadeMinCosSim  = fit->second.as<float>();
		else if (paramKey == "nmCascadeMaxCosSim")   m_nmCascadeMaxCosSim  = fit->second.as<float>();
		else if (paramKey == "nmCascadeDecayRate")   m_nmCascadeDecayRate  = fit->second.as<float>();
		else if (paramKey == "maxWsExt25WorldDist")   m_maxWsExt25WorldDist  = fit->second.as<float>();
		else if (paramKey == "wsExt25MinCosSim")      m_wsExt25MinCosSim     = fit->second.as<float>();
		else if (paramKey == "wsExt25StrictCosSim")   m_wsExt25StrictCosSim  = fit->second.as<float>();
		else if (paramKey == "maxWorldDist4ArchiveReAssocWS")
		{
			/// Within-stream override.
			m_maxWorldDist4ArchiveReAssocWS = fit->second.as<float>();
		}
		else if (paramKey == "minSpeed4AngleChecking")
		{
			m_minSpeed4AngleChecking = fit->second.as<float>();
		}
		else if (paramKey == "maxIou4AngleChecking")
		{
			m_maxIou4AngleChecking = fit->second.as<float>();
		}
		else if (paramKey == "reidExtractionInterval")
		{
			m_reidExtractionInterval = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4ReidSimilarity")
		{
			m_minMatchingScore4ReidSimilarity = fit->second.as<float>();
		}
		else if (paramKey == "matchingScoreWeight4ReidSimilarity")
		{
			m_matchingScoreWeight4ReidSimilarity = fit->second.as<float>();
		}
		else if (paramKey == "dumpTrajectoryIntoFile")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_dumpTrajectoryIntoFile = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_dumpTrajectoryIntoFile is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "dumpDetectorBboxes")
		{
			m_dumpDetectorBboxes = fit->second.as<float>();
		}
		else if (paramKey == "dumpTrajectoryTentative")
		{
			m_dumpTrajectoryTentative = fit->second.as<float>();
		}
		else if (paramKey == "dumpTrajectoryQuasiActive")
		{
			m_dumpTrajectoryQuasiActive = fit->second.as<float>();
		}
		else if (paramKey == "dumpTrajectoryInactive")
		{
			m_dumpTrajectoryInactive = fit->second.as<float>();
		}
		else if (paramKey == "maxInactiveDumpFrames")
		{
			m_maxInactiveDumpFrames = fit->second.as<int>();
		}
		else if (paramKey == "minVisibility4TrajDump")
		{
			m_minVisibility4TrajDump = fit->second.as<float>();
		}
		else if (paramKey == "trajectoryDumpFileName")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_trajectoryDumpFileName = fit->second.as<std::string>();
			else
				CLOG(std::string("!!![WARNING] m_trajectoryDumpFileName is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "maxProjTrajectoryLength4Matching")
		{
			m_offlineParams.m_maxProjTrajectoryLength4Matching = fit->second.as<float>();
		}
		else if (paramKey == "minOverlappedTrajectoryLength4Matching")
		{
			m_offlineParams.m_minOverlappedTrajectoryLength4Matching = fit->second.as<float>();
		}
		else if (paramKey == "iouAvgScaleFactor")
		{
			m_offlineParams.m_iouAvgScaleFactor = fit->second.as<float>();
		}
		else
		{
			cout << "!! [WARNING][NvTrackerParamsTrajectoryManager] Unknown param found: " << paramKey << endl;
		}
	}

	/// Set init flag after initialization to prevent updates from changing static parameters.
	m_isInit = true;
}
