/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "modules/NvMultiObjectTracker/NvMultiObjectTracker.hpp"

/** @brief Construct an uninitialized multi-view associator. */
MultiViewAssociator::MultiViewAssociator()
{
    m_params = MultiViewAssociatorParams(m_configFilePath, "MultiViewAssociator");
    m_communicator = CommunicatorFactory::getInstance().makeCommunicator(m_configFilePath);
    m_isInit = false;
}

/**
 * @brief Construct a multi-view associator from configuration.
 * @param[in] configFilePath Path to the configuration file.
 */
MultiViewAssociator::MultiViewAssociator(const std::string &configFilePath)
{
    m_configFilePath = configFilePath;
    m_params = MultiViewAssociatorParams(m_configFilePath, "MultiViewAssociator");
    m_communicator = CommunicatorFactory::getInstance().makeCommunicator(m_configFilePath);
    m_isInit = false;
}

/** @brief Report statistics and destroy the multi-view associator. */
MultiViewAssociator::~MultiViewAssociator()
{
    if (m_params.m_debugVerboseLevel >= 1) {
        std::cerr << "[MultiViewAssociator] cam=" << m_camId
            << " reAssoc_tries=" << m_mv_stats.reAssoc_tries
            << " ra_pairs=" << m_mv_stats.ra_pairs_evaluated
            << " ra_no_overlap=" << m_mv_stats.ra_no_overlap
            << " ra_partial=" << m_mv_stats.ra_partial_overlap
            << " ra_score_fail=" << m_mv_stats.ra_score_fail
            << " ra_adopted=" << m_mv_stats.ra_adopted
            << " ra_partial_adopted=" << m_mv_stats.ra_partial_adopted
            << " idc_pairs=" << m_mv_stats.idc_pairs_evaluated
            << " idc_no_overlap=" << m_mv_stats.idc_no_overlap
            << " idc_partial=" << m_mv_stats.idc_partial_overlap
            << " idc_score_fail=" << m_mv_stats.idc_score_fail
            << " cfg_minCommonFrames=" << m_params.m_minCommonFrames4MatchScore
            << " cfg_minScore=" << m_params.m_minPeerTrackletMatchScore
            << " cfg_trackletSize=" << m_params.m_maxPeerTrackletSize
            << " cfg_egoHistMult=" << m_params.m_egoTrackletHistoryMultiplier
            << " cfg_egoMaxSize=" << (m_params.m_maxPeerTrackletSize * m_params.m_egoTrackletHistoryMultiplier)
            << " ra_score_hist=[";
        for (int i = 0; i < 10; ++i)
            std::cerr << (i ? "," : "") << m_mv_stats.ra_score_hist[i];
        std::cerr << "]"
            << " ra_partial_hist=[";
        for (int i = 0; i < 10; ++i)
            std::cerr << (i ? "," : "") << m_mv_stats.ra_partial_hist[i];
        std::cerr << "]"
            << " ra_partial_count_hist=[";
        for (int i = 0; i < 5; ++i)
            std::cerr << (i ? "," : "") << m_mv_stats.ra_partial_count_hist[i];
        std::cerr << "]"
            << " noov_seq_lt2=" << m_mv_stats.noov_seq_lt2
            << " noov_seq_2_5=" << m_mv_stats.noov_seq_2_5
            << " noov_seq_5_15=" << m_mv_stats.noov_seq_5_15
            << " noov_seq_15plus=" << m_mv_stats.noov_seq_15plus
            << " noov_seq_bridge_ts_pass=" << m_mv_stats.noov_seq_bridge_ts_pass
            << " noov_seq_bridge_adopted=" << m_mv_stats.noov_seq_bridge_adopted
            << " bridge_seq_ts_fail=" << m_mv_stats.bridge_seq_ts_fail
            << " bridge_best_score_sum_at_fail=" << std::fixed << std::setprecision(3)
            << m_mv_stats.bridge_seq_best_score_sum_at_ts_fail
            << " peer_ts_blocked=" << m_mv_stats.peer_ts_blocked;
        std::cerr << " ra_adopted_hist=[";
        for (int i = 0; i < 10; ++i)
            std::cerr << (i ? "," : "") << m_mv_stats.regular_adopted_score_hist[i];
        std::cerr << "] ra_partial_adopted_hist=[";
        for (int i = 0; i < 10; ++i)
            std::cerr << (i ? "," : "") << m_mv_stats.partial_adopted_score_hist[i];
        std::cerr << "]";
        std::cerr << " height_ego_gt0=" << m_mv_stats.height_ego_gt0
            << " height_peer_gt0=" << m_mv_stats.height_peer_gt0
            << " height_both_valid=" << m_mv_stats.height_both_valid
            << " height_adopted_both_valid=" << m_mv_stats.height_adopted_both_valid
            << " height_adopted_delta_sum=" << std::fixed << std::setprecision(3)
            << m_mv_stats.height_adopted_delta_sum
            << " height_adopted_delta_hist=[";
        for (int i = 0; i < 5; ++i)
            std::cerr << (i ? "," : "") << m_mv_stats.height_adopted_delta_hist[i];
        std::cerr << "]\n";
    }
}

/** @brief Return whether the multi-view associator is initialized. */
bool MultiViewAssociator::isInitialized()
{
    return m_isInit;
}

/**
 * @brief Initialize peer-target state and the communicator.
 * @param[in] streamId Stream identifier for the ego camera.
 * @param[in] start Reference start time for timestamp conversion.
 * @param[in] localInstanceCommunicators Communicators for local streams.
 * @param[in] egoTargets Ego-camera targets managed by the tracker.
 * @param[in] objModelProjector Projector used to recover world-space target state.
 */
void MultiViewAssociator::initialize(
    const StreamId streamId, system_clock::time_point start,
    std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators,
    std::shared_ptr<NvTrackedObjectList> egoTargets, std::shared_ptr<ObjectModelProjector> objModelProjector
)
{
    m_frameNum = 0;
    m_streamId = streamId;
    m_egoTargets = egoTargets;
    m_objModelProjector = objModelProjector;

    /// Open the comm log file and set the start time stamp.
    if (m_params.m_debugVerboseLevel > 0) {
        m_logFile.open("outMVReAssoc/Stream" + std::to_string(m_streamId >> 32) + ".log");
        m_frameAlignLog.open("outMVReAssoc/Stream" + std::to_string(m_streamId >> 32) + "_frameAlign.log");
    }
    m_startTime = start;
    const std::string sensorName = objModelProjector ? objModelProjector->getSensorName() : "";
    m_communicator->initialize(m_streamId, start, localInstanceCommunicators, sensorName);
    if (m_communicator->getCommunicatorType() == COMMUNICATOR_TYPES::MQTT_COMMUNICATOR) {
        if (m_params.m_communicatorInitSleepTime > 0) {
            /// Wait for mqtt communicator to be initialized.
            std::this_thread::sleep_for(std::chrono::milliseconds(m_params.m_communicatorInitSleepTime));
        } else {
            static_cast<MqttCommunicator*>(m_communicator.get())->waitForConnectionCompletion();
        }
    }
    m_peerTargetDB.clear();
    m_peerLastMsgByCam.clear();
    m_camId = m_communicator->getEgoCamId();

    /// Initialize the peer-target database for each peer camera (ego camera not included).
    for (auto peerCamId : m_communicator->getPeerCamIds()) {
        m_peerTargetDB[peerCamId] = std::map<TargetId, PeerTarget>();
        m_receivedSyncFromCameras[peerCamId] = 0;
        if (m_params.m_debugVerboseLevel > 0) {
            m_logFile << "PeerTargetDB entry initialized for peerCamID " << peerCamId << "\n";
        }
    }

    m_isInit = true;
    if (m_params.m_debugVerboseLevel > 0) {
        std::cout << "MultiViewAssociator initialized for stream " << (m_streamId >> 32)
                  << " (cam " << m_camId << ")\n";
    }
}

/**
 * @brief Synchronize the peer-target database with communicator messages.
 * @param[in] timeMeta Current frame timing metadata.
 */
void MultiViewAssociator::update(const NvMOTTimeMeta &timeMeta)
{
    std::unique_lock<std::mutex> commLock(m_commMutex); commLock.unlock();
    std::unique_lock<std::mutex> peerLock(m_peerTargetDBMutex); peerLock.unlock();

    if (timeMeta.frameNum > 0) m_egoDeltaTime = timeMeta.frameTimeStamp - m_timeMeta.frameTimeStamp;

    m_timeMeta = timeMeta;
    m_frameNum = timeMeta.frameNum;

    /// Update m_egoTargetsByID.
    m_egoTargetsByID.clear();
    for (const auto &egoTargetPair : *m_egoTargets)
	{
        m_egoTargetsByID[egoTargetPair.second->getTargetID()] = egoTargetPair.second;
	} // end of for (auto egoTargetPair : m_egoTargets)
    m_egoTargetsByID.erase(-1);

    /// Update m_adoptedIdMap based on m_adoptedReidServiceID.
    for (const auto &egoTargetPair : *m_egoTargets) {
        if (egoTargetPair.second->adoptedReidServiceID()) {
            m_adoptedIdMap[egoTargetPair.second] = egoTargetPair.second->getTargetIDBeforeAdoption();
            egoTargetPair.second->setAdoptedReidServiceID(false);
        }
    }

    /// Process all the received messages from previous frames.
    if (timeMeta.frameNum > 0) processReceivedMessagesToCurrentFrame();

    /// Write a periodic peerTargetDB summary log every 100 frames.
    if (m_params.m_debugVerboseLevel > 0 && m_frameAlignLog.is_open() && m_frameNum % 100 == 0) {
        std::unique_lock<std::mutex> dbLogLock(m_peerTargetDBMutex);
        m_frameAlignLog << "frame: " << m_frameNum << ", [PeerTargetDB/Summary] egoFr=" << m_frameNum << "\n";
        for (const auto &camPair : m_peerTargetDB) {
            for (const auto &tgtPair : camPair.second) {
                const auto &pt = tgtPair.second;
                const int64_t ptAge = static_cast<int64_t>(m_frameNum) - static_cast<int64_t>(pt.frameNum);
                const NvMOTFrameNum tkFirst = pt.tracklet.empty() ? 0 : pt.tracklet.front().frameNum;
                const NvMOTFrameNum tkLast  = pt.tracklet.empty() ? 0 : pt.tracklet.back().frameNum;
                m_frameAlignLog << "  cam=" << camPair.first
                    << ", tgtID=" << int(tgtPair.first)
                    << ", ptFr=" << pt.frameNum << ", ptAge=" << ptAge
                    << ", tkSz=" << pt.tracklet.size()
                    << ", tk=[" << tkFirst << "," << tkLast << "]"
                    << ", pos=(" << (pt.nextState.size() >= 2 ? pt.nextState(0) : 0.f)
                    << "," << (pt.nextState.size() >= 2 ? pt.nextState(1) : 0.f) << ")"
                    << "\n";
            }
        }
        m_frameAlignLog << std::flush;
        dbLogLock.unlock();
    }

    /// Try late peer re-association.
    if (m_params.m_enableLatePeerReAssoc) tryLatePeerReAssoc();

    /// Try ID correction.
    if (m_params.m_enableIDCorrection) tryIDCorrection();

    /// Cleanup tracklets in peerTargetDB.
    cleanupPeerTargetDB();

    /// Update current egoTargets with corresponding peerTargets.
    peerLock.lock();
    for (auto &egoTargetPair : *m_egoTargets) {
        std::vector<PeerTarget> peers; peers.clear();
        if (egoTargetPair.second->needTargetIDAssigned()) continue;
        for (auto &peerCamPair : m_peerTargetDB) {
            for (auto peerTargetPair : peerCamPair.second) {
                if (egoTargetPair.second->getTargetID() == peerTargetPair.second.targetId) {
                    peers.push_back(peerTargetPair.second);
                }
            }
        }
        egoTargetPair.second->setPeerTargets(peers);
    }
    peerLock.unlock();

    /// Collect the tracklet, adoptedId, and states payloads.
    std::vector<Payload> trackletList, adoptedIdList, statesList, mergedList;
    for (auto egoTargetPair : *m_egoTargets)
	{
        auto egoTarget = egoTargetPair.second;
        if (egoTarget->getTrackerState() == TRACKER_STATE::ACTIVE)
		{
            if (egoTarget->becameActive() || egoTarget->acquiredID() || m_adoptedIdMap.find(egoTarget) != m_adoptedIdMap.end()) {
                trackletList.push_back(generateTrackletPayload(egoTarget));
                if (m_adoptedIdMap.find(egoTarget) != m_adoptedIdMap.end()) {
                    adoptedIdList.push_back(generateAdoptedIdPayload(egoTarget, m_adoptedIdMap[egoTarget]));
                    m_adoptedIdMap.erase(egoTarget);
                }
                egoTarget->setBecameActive(false);
                egoTarget->setAcquiredID(false);
            }
            else if (!egoTarget->needTargetIDAssigned()) {
                /// Exception for ID correction.
                statesList.push_back(generateStatesPayload(egoTarget));
            }
        }
	} // end of for (auto egoTargetPair : *m_egoTargets)

    /// Merge the payloads.
    for (auto p: adoptedIdList) mergedList.push_back(std::move(p));
    for (auto p: trackletList) mergedList.push_back(std::move(p));
    for (auto p: statesList) mergedList.push_back(std::move(p));
    mergedList.push_back(generateSyncPayload());

    /// Send the payloads to the communicator.
    m_communicator->requestToSend(mergedList);

    /// Process the debug logs.
    if (m_params.m_debugVerboseLevel > 1) {
        std::stringstream debugMsg;
        /// Debug message: egoTarget and associated peerTargets with last point.
        for (const auto &egoTargetPair : *m_egoTargets) {
            debugMsg << "systemTime: " << Time().getStrSystemTime() << ", ";
            debugMsg << "time: " << Time().getStrTimeRelativeTo(m_startTime) << ", ";
            debugMsg << "frame: " << m_frameNum << ", ";
            debugMsg << "[EgoTargetAssocPeers] (egoCam" << m_camId << ", ID"
                     << int(egoTargetPair.second->getTargetID()) << ", ";
            debugMsg << "state: " << state2string(egoTargetPair.second->getTrackerState()) << ", loc: ";
            debugMsg << egoTargetPair.second->getTargetInfo().pt_world_feet.transpose() << "), AssocPeers: {";
            for (auto peer : egoTargetPair.second->getPeerTargets()) {
                debugMsg << "(cam" << peer.cameraId << ", ID" << peer.targetId << ", ";
                if(peer.tracklet.size() > 0)
                    debugMsg << peer.tracklet.back().x << ", " << peer.tracklet.back().y << "),";
            }
            debugMsg << "}\n";
        }

        /// Print the debug message to the log file.
        if (m_params.m_debugVerboseLevel > 1) {
            commLock.lock();
            m_logFile << debugMsg.str() << std::flush;
            commLock.unlock();
        }
    }
}

/**
 * @brief Compute the matching score between two tracklets.
 * @param[in] tracklet1 First tracklet.
 * @param[in] tracklet2 Second tracklet.
 * @param[in] useMahalanobis Use Mahalanobis distance when true.
 * @param[out] commonFrames Common-frame count written when non-null.
 * @param[out] partialScore Partial-overlap score written when non-null.
 * @param[in] partialTimeRange Extended dT range for the partial-score pass.
 * @param[out] partialCount Frame count paired with the best partial score.
 * @return The best full-overlap matching score.
 */
float MultiViewAssociator::computeTrackletMatchScore(
    const Tracklet &tracklet1, const Tracklet &tracklet2, bool useMahalanobis,
    NvMOTFrameNum* commonFrames, float* partialScore, int partialTimeRange, int* partialCount
)
{
    if (commonFrames) *commonFrames = 0;
    if (partialScore) *partialScore = 0.f;
    if (partialCount) *partialCount = 0;
    if (tracklet1.empty() || tracklet2.empty()) return 0.f;

    /// Build hash map for tracklet2: frame number -> index (O(n)).
    std::unordered_map<NvMOTFrameNum, size_t> t2FrameToIdx;
    t2FrameToIdx.reserve(tracklet2.size());
    for (size_t i = 0; i < tracklet2.size(); ++i) {
        t2FrameToIdx[tracklet2[i].frameNum] = i;
    }

    int count = 0, maxCount = 0, maxPartialScoreCount = 0;
    float matchScore = 0.f, maxMatchScore = 0.f, maxPartialScore = 0.f, totalSquaredDistance = 0.f;

    for (int dT = -m_params.m_maxTrackletMatchingTimeSearchRange; dT <= m_params.m_maxTrackletMatchingTimeSearchRange; ++dT) {
        count = 0;
        totalSquaredDistance = 0.f;

        /// O(n) lookup instead of O(n²) nested loop.
        for (const auto &p1 : tracklet1) {
            auto it = t2FrameToIdx.find(p1.frameNum + dT);
            if (it == t2FrameToIdx.end()) continue;

            const auto &p2 = tracklet2[it->second];
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;

            if (useMahalanobis) {
                /// Assuming cam-to-cam independent covariance, so add them Mahalanobis squared: diff^T * (Σ1+Σ2)^(-1) * diff.
                float sxx = p1.sxx + p2.sxx;
                float sxy = p1.sxy + p2.sxy;
                float syy = p1.syy + p2.syy;
                float det = sxx * syy - sxy * sxy;
                if (det > 1e-6f) {
                    totalSquaredDistance += (syy * dx * dx - 2.f * sxy * dx * dy + sxx * dy * dy) / det;
                } else {
                    totalSquaredDistance += dx * dx + dy * dy;
                }
            } else {
                /// Euclidean squared (default covariance).
                totalSquaredDistance += dx * dx + dy * dy;
            }
            count++;
        }
        if (count > maxCount) maxCount = count;

        /// Compute score if enough common frame numbers were found.
        if (count >= m_params.m_minCommonFrames4MatchScore) {
            if (useMahalanobis) {
                /// Exp(-sumDistSq / (2*n)) = geometric mean of individual Gaussian likelihoods.
                matchScore = std::exp(-totalSquaredDistance / (2.f * count));
            } else {
                /// 1 / (1 + x) legacy scoring for Euclidean distance.
                matchScore = 1.f / (1.f + std::sqrt(totalSquaredDistance / count));
            }
            if (matchScore > maxMatchScore) maxMatchScore = matchScore;
        } else if (partialScore && count > 0) {
            /// Score partial overlap when the common-frame count is below minCommonFrames4MatchScore.
            float ps = useMahalanobis ?
                std::exp(-totalSquaredDistance / (2.f * count)) :
                1.f / (1.f + std::sqrt(totalSquaredDistance / count));
            if (ps > maxPartialScore) {
                maxPartialScore = ps;
                /// Remember the frame count paired with the best partial score.
                maxPartialScoreCount = count;
            }
        }
    }

    /// Second pass for an extended partial-score dT range.
    if (partialScore && maxCount == 0 && partialTimeRange > m_params.m_maxTrackletMatchingTimeSearchRange) {
        for (int dT = -partialTimeRange; dT <= partialTimeRange; ++dT) {
            /// Skip dT values already covered by the primary loop to avoid redundant work.
            if (std::abs(dT) <= m_params.m_maxTrackletMatchingTimeSearchRange) continue;
            count = 0;
            totalSquaredDistance = 0.f;
            for (const auto &p1 : tracklet1) {
                auto it = t2FrameToIdx.find(p1.frameNum + dT);
                if (it == t2FrameToIdx.end()) continue;
                const auto &p2 = tracklet2[it->second];
                float dx = p1.x - p2.x;
                float dy = p1.y - p2.y;
                if (useMahalanobis) {
                    float sxx = p1.sxx + p2.sxx;
                    float sxy = p1.sxy + p2.sxy;
                    float syy = p1.syy + p2.syy;
                    float det = sxx * syy - sxy * sxy;
                    if (det > 1e-6f) {
                        totalSquaredDistance += (syy * dx * dx - 2.f * sxy * dx * dy + sxx * dy * dy) / det;
                    } else {
                        totalSquaredDistance += dx * dx + dy * dy;
                    }
                } else {
                    totalSquaredDistance += dx * dx + dy * dy;
                }
                count++;
            }
            if (count > 0) {
                float ps = useMahalanobis ?
                    std::exp(-totalSquaredDistance / (2.f * count)) :
                    1.f / (1.f + std::sqrt(totalSquaredDistance / count));
                if (ps > maxPartialScore) {
                    maxPartialScore = ps;
                    /// Remember the frame count paired with the best partial score.
                    maxPartialScoreCount = count;
                }
                /// Include this overlap in the common-frame count.
                if (count > maxCount) maxCount = count;
            }
        }
    }

    if (commonFrames) *commonFrames = static_cast<NvMOTFrameNum>(maxCount);
    if (partialScore) *partialScore = maxPartialScore;
    /// Return the frame count at the offset that produced the best partial score.
    if (partialCount) *partialCount = maxPartialScoreCount;
    return maxMatchScore;
}

/**
 * @brief Populate a tracklet from a target buffer.
 * @param[in] target Tracked object to read.
 * @param[out] tracklet Tracklet to populate.
 * @param[in] maxSize Maximum number of frames to include.
 */
void MultiViewAssociator::populateTracklet(
    const std::shared_ptr<NvTrackedObject> target, Tracklet* tracklet, int maxSize
)
{
    if (maxSize <= 0) maxSize = m_params.m_maxPeerTrackletSize;
    tracklet->clear();
    for (auto tInfo = target->getTrajectoryBuffer().rbegin(); tInfo != target->getTrajectoryBuffer().rend(); ++tInfo) {
        tracklet->push_front(
            TrackletPoint(
                tInfo->frameNum, tInfo->pt_world_feet(0), tInfo->pt_world_feet(1),
                tInfo->pt_world_feet_cov(0,0), tInfo->pt_world_feet_cov(0,1), tInfo->pt_world_feet_cov(1,1)
            )
        );
        if (int(tracklet->size()) >= maxSize) break;
    }
}

/**
 * @brief Associate ego detections with recent peer targets.
 * @param[in] detObjs Ego detections to match against peer targets.
 * @param[in] minMatchingScore4PeerAssocIou Minimum IoU required to match a detection to a peer target.
 * @return Detections that match a recent peer target.
 */
std::deque<const DetectedObjInfo *> MultiViewAssociator::tryPeerDataAssoc(
    const std::deque<const DetectedObjInfo *> detObjs, const float minMatchingScore4PeerAssocIou
)
{
    std::set<const DetectedObjInfo *>  matchedDets;

    std::unique_lock<std::mutex> lock(m_peerTargetDBMutex);
    for (const auto &peerCamPair : m_peerTargetDB) {
		for (const auto &peerTargetPair : peerCamPair.second) {
			const PeerTarget& peerTarget = peerTargetPair.second;
            /// If the current peerTarget's ID is already stored in ego camera, detection already tried ego association in data association module, so skip here.
            if (m_egoTargetsByID.find(peerTarget.targetId) != m_egoTargetsByID.end())
                continue;

            /// Only match with recently updated peer targets.
            if (peerTarget.frameNum + m_params.m_maxPeerFrameDiff4NoDet >= m_frameNum)
            {
                Eigen::Vector2d peer_world_feet(peerTarget.nextState(0), peerTarget.nextState(1));
                /// TODO: Use the peer target class instead of the default class 0.
                NvMotDataTypes::Rect2f peerMotBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
                    Eigen::Vector2d(peer_world_feet(0), peer_world_feet(1)), 0, -1
                );

                for (auto &detObj : detObjs)
                {
                    if (CV_FUNCS::computeIOU(peerMotBbox, detObj->bbox) > minMatchingScore4PeerAssocIou)
                    {
                        matchedDets.insert(detObj);
                    }
                }
            }
        }
    }
    lock.unlock();

    return std::deque<const DetectedObjInfo *>(matchedDets.begin(), matchedDets.end());
}

/**
 * @brief Try to reassociate a target with peer tracklets.
 * @param[in] egoTarget Target to match.
 * @return True when peer reassociation succeeds.
 */
bool MultiViewAssociator::tryPeerReAssoc(std::shared_ptr<NvTrackedObject> egoTarget)
{
    m_mv_stats.reAssoc_tries++;
    TimeStamp bestMatchIdTimeStamp = egoTarget->getTargetIDTimeStamp();
    TargetId bestMatchId = egoTarget->getTargetID();
    float bestMatchScore = 0.0;
    /// ID timestamp captured before the peer loop.
    const auto egoInitialTs = bestMatchIdTimeStamp;
    const float egoHeight = egoTarget->getTargetHeight();
    const float egoHeightConf = egoTarget->getHeightConfidence();
    if (egoHeightConf > 0.f) m_mv_stats.height_ego_gt0++;
    float bestMatchPeerHeight = 1.6f;
    float bestMatchPeerHeightConf = 0.f;
    /// Winning pair used the partial-overlap path.
    bool bestMatchIsPartial = false;
    /// Winning pair used the sequential bridge.
    bool bestMatchIsBridge  = false;
    /// Build the ego tracklet with the extended history window.
    Tracklet tracklet;
    int egoMaxSize = m_params.m_maxPeerTrackletSize * m_params.m_egoTrackletHistoryMultiplier;
    populateTracklet(egoTarget, &tracklet, egoMaxSize);
    /// Compute match score between the ego-tracklet and the tracklets in the PeerReAssocDB.
    for (const auto &peerCamPair : m_peerTargetDB) {
        for (const auto &peerTargetPair : peerCamPair.second) {
            const auto &peerTarget = peerTargetPair.second;

            if (m_egoTargetsByID.find(peerTarget.targetId) != m_egoTargetsByID.end()) continue;
            /// Skip cross-class peers when class-match gating is enabled.
            if (m_params.m_checkClassMatch && egoTarget->getTargetInfo().classId != peerTarget.classId) continue;

            NvMOTFrameNum commonFrames = 0;
            /// Set when the bridge formula applies to this peer.
            bool isBridgePeer = false;
            /// Partial score for pairs whose common-frame count is below minCommonFrames4MatchScore.
            float partialScore = 0.f;
            /// Partial dT range: wider than the primary range only when the multiplier is greater than 1.
            int partialRange = m_params.m_maxTrackletMatchingTimeSearchRange * m_params.m_partialTimeSearchMultiplier;
            /// Frame count at the offset that produced partialScore.
            int partialCount = 0;
            float matchScore = computeTrackletMatchScore(
                tracklet, peerTarget.tracklet, egoTarget->usesProjectiveEstimator(),
                &commonFrames, &partialScore, partialRange, &partialCount
            );

            m_mv_stats.ra_pairs_evaluated++;
            if (commonFrames == 0) {
                m_mv_stats.ra_no_overlap++;
                if (!peerTarget.tracklet.empty() && !tracklet.empty() &&
                    peerTarget.tracklet.back().frameNum < tracklet.front().frameNum) {
                    float dx = tracklet.front().x - peerTarget.tracklet.back().x;
                    float dy = tracklet.front().y - peerTarget.tracklet.back().y;
                    float d = std::sqrt(dx*dx + dy*dy);
                    if      (d < 2.f)  m_mv_stats.noov_seq_lt2++;
                    else if (d < 5.f)  m_mv_stats.noov_seq_2_5++;
                    else if (d < 15.f) m_mv_stats.noov_seq_5_15++;
                    else               m_mv_stats.noov_seq_15plus++;
                    /// Sequential no-overlap bridge: score is D / (D + d) when the junction distance is below maxSeqBridgeDist.
                    if (m_params.m_maxSeqBridgeDist > 0.f && d < m_params.m_maxSeqBridgeDist) {
                        matchScore = m_params.m_maxSeqBridgeDist / (m_params.m_maxSeqBridgeDist + d);
                        /// This peer is a bridge candidate.
                        isBridgePeer = true;
                    }
                }
            } else if (int(commonFrames) < m_params.m_minCommonFrames4MatchScore) {
                m_mv_stats.ra_partial_overlap++;
                m_mv_stats.ra_partial_hist[std::min(9, (int)(partialScore * 10.f))]++;
                /// Use partialScore when it meets the partial-overlap score and frame-count gates.
                if (partialScore >= m_params.m_minMatchingScore4PartialOverlap &&
                    partialCount >= m_params.m_minCommonFrames4PartialOverlap) {
                    matchScore = partialScore;
                    int cb = (partialCount == 1) ? 0 : (partialCount <= 3) ? 1 : (partialCount <= 7) ? 2 :
                        (partialCount <= 11) ? 3 : 4;
                    m_mv_stats.ra_partial_count_hist[cb]++;
                }
            } else if (matchScore < m_params.m_minPeerTrackletMatchScore) {
                m_mv_stats.ra_score_fail++;
                m_mv_stats.ra_score_hist[std::min(9, (int)(matchScore * 10.f))]++;
            }

            if (m_params.m_debugVerboseLevel > 0 && m_frameAlignLog.is_open()) {
                const auto &egoInfo = egoTarget->getTargetInfo();
                m_frameAlignLog << "frame: " << m_frameNum
                    << ", [ReAssoc/Try] egoID=" << int(egoTarget->getTargetID())
                    << ", peerCam=" << peerTarget.cameraId
                    << ", peerID=" << int(peerTarget.targetId)
                    << ", ego_pos=(" << egoInfo.pt_world_feet(0) << "," << egoInfo.pt_world_feet(1) << ")"
                    << ", peer_pos=(" << peerTarget.nextState(0) << "," << peerTarget.nextState(1) << ")"
                    << ", egoTk=[";
                for (size_t i = 0; i < tracklet.size(); ++i)
                    m_frameAlignLog << (i ? "," : "") << tracklet[i].frameNum;
                m_frameAlignLog << "], peerTk=[";
                for (size_t i = 0; i < peerTarget.tracklet.size(); ++i)
                    m_frameAlignLog << (i ? "," : "") << peerTarget.tracklet[i].frameNum;
                m_frameAlignLog << "]"
                    << ", commonFrames=" << commonFrames
                    << ", matchScore=" << matchScore
                    << "\n" << std::flush;
            }
            if (m_params.m_debugVerboseLevel > 1) {
                messageProcessLog(peerTarget.cameraId, peerTarget.targetId, "(peerReAssoc), ");
                std::unique_lock<std::mutex> commLock(m_commMutex);
                m_logFile << "(PeerReAssoc from above) IDtime: " << std::setw(1) <<
                    peerTarget.targetIDTimeStamp - duration_cast<nanoseconds>(m_startTime.time_since_epoch()).count()
                    << ", ";
                m_logFile << "egoTargetID: " << std::setw(1) << int(egoTarget->getTargetID()) << ", ";
                m_logFile << "egoTrackletSize: " << std::setw(1) << tracklet.size() << ", ";
                m_logFile << "egoIDtime: " << std::setw(1) <<
                    egoTarget->getTargetIDTimeStamp() -
                    duration_cast<nanoseconds>(m_startTime.time_since_epoch()).count() << ", ";
                m_logFile << "matchScore: " << std::setw(1) << matchScore << "\n";
                commLock.unlock();
            }
            if (!isBridgePeer && matchScore > m_params.m_minPeerTrackletMatchScore &&
                peerTarget.targetIDTimeStamp >= egoInitialTs) {
                m_mv_stats.peer_ts_blocked++;
            }
            if (peerTarget.heightConfidence > 0.f) m_mv_stats.height_peer_gt0++;
            if (matchScore > m_params.m_minPeerTrackletMatchScore &&
                egoHeightConf >= 0.5f && peerTarget.heightConfidence >= 0.5f)
                m_mv_stats.height_both_valid++;
            /// MatchScore may be partialScore for a qualifying partial-overlap pair.
            if (isBridgePeer && matchScore > m_params.m_minPeerTrackletMatchScore &&
                peerTarget.targetIDTimeStamp < bestMatchIdTimeStamp) {
                m_mv_stats.noov_seq_bridge_ts_pass++;
            }
            if (isBridgePeer && matchScore > m_params.m_minPeerTrackletMatchScore &&
                !(peerTarget.targetIDTimeStamp < bestMatchIdTimeStamp)) {
                m_mv_stats.bridge_seq_ts_fail++;
                m_mv_stats.bridge_seq_best_score_sum_at_ts_fail += bestMatchScore;
            }
            if ((peerTarget.targetIDTimeStamp < bestMatchIdTimeStamp && matchScore > m_params.m_minPeerTrackletMatchScore) ||
                (peerTarget.targetIDTimeStamp == bestMatchIdTimeStamp && matchScore > m_params.m_minPeerTrackletMatchScore &&
                 peerTarget.targetId < bestMatchId)
                )
            {
                bestMatchIdTimeStamp = peerTarget.targetIDTimeStamp;
                bestMatchId = peerTarget.targetId;
                bestMatchScore = matchScore;
                /// The winning pair is partial when common frames are below minCommonFrames4MatchScore.
                bestMatchIsPartial = (int(commonFrames) < m_params.m_minCommonFrames4MatchScore);
                bestMatchIsBridge  = isBridgePeer;
                bestMatchPeerHeight = peerTarget.height;
                bestMatchPeerHeightConf = peerTarget.heightConfidence;
            }
        }
    }
    if (bestMatchScore > m_params.m_minPeerTrackletMatchScore) {
        /// Check if the Target was associated using reid service.
        if (!egoTarget->needToReAssocFromReidService()) {
            /// If so, check the confidence of bestMatchId from reid service response.
            if (egoTarget->getReidServiceMatchConfidence(bestMatchId) < m_params.m_minReidServiceMatchConfidence) {
                if (m_params.m_debugVerboseLevel > 1) {
                    cout << "Stream ID: [" << (egoTarget->getVideoStreamId() >> 32) << "] frameNum[" << egoTarget->getTargetInfo().frameNum << "] Discarding TargetID matched using peer re-association from Target ID : [" << egoTarget->getTargetID() << "] to [" << bestMatchId << "] due to low reid service match confidence" << endl;
                }
                return false;
            }
        }

        /// Extra score gate for regular full-overlap adoptions.
        if (!bestMatchIsPartial && !bestMatchIsBridge &&
            m_params.m_minMatchingScore4RegularReAssoc > 0.0f &&
            bestMatchScore < m_params.m_minMatchingScore4RegularReAssoc) {
            return false;
        }
        TargetId oldId = egoTarget->getTargetID();
        egoTarget->setTargetID(bestMatchId);
        egoTarget->setTargetIDTimeStamp(bestMatchIdTimeStamp);
        /// Update m_egoTargetsByID with the new ID.
        m_egoTargetsByID.erase(oldId);
        m_egoTargetsByID[bestMatchId] = egoTarget;
        m_mv_stats.ra_adopted++;
        if (bestMatchIsPartial) m_mv_stats.ra_partial_adopted++;
        if (bestMatchIsBridge)  m_mv_stats.noov_seq_bridge_adopted++;
        if (egoHeightConf >= 0.5f && bestMatchPeerHeightConf >= 0.5f) {
            m_mv_stats.height_adopted_both_valid++;
            float delta = std::abs(egoHeight - bestMatchPeerHeight);
            m_mv_stats.height_adopted_delta_sum += delta;
            int hb = (delta < 0.05f) ? 0 : (delta < 0.10f) ? 1 : (delta < 0.20f) ? 2 : (delta < 0.30f) ? 3 : 4;
            m_mv_stats.height_adopted_delta_hist[hb]++;
        }
        {
            int bucket = std::min(9, (int)(bestMatchScore * 10.0f));
            if (bestMatchIsPartial)
                m_mv_stats.partial_adopted_score_hist[bucket]++;
            else
                m_mv_stats.regular_adopted_score_hist[bucket]++;
        }
        return true;
    }
    return false;
}

/** @brief Try to reassociate recently active ego targets with peer tracklets. */
void MultiViewAssociator::tryLatePeerReAssoc()
{
    /// Iterate over ego targets.
    for (const auto& egoTargetPair : *m_egoTargets) {
        std::shared_ptr<NvTrackedObject> egoTarget = egoTargetPair.second;

        /// Only process recently active targets.
        if (int(egoTarget->getAge()) >= int(m_params.m_recentlyActiveAge)) continue;
        if (egoTarget->getTrackerState() != TRACKER_STATE::ACTIVE) continue;

        /// Try peer re-association for this ego target.
        TargetId originalId = egoTarget->getTargetID();
        if (tryPeerReAssoc(egoTarget)) {
            if (!egoTarget->acquiredID()) m_adoptedIdMap[egoTarget] = originalId;
        }
    }
}

/** @brief Correct ego and peer target IDs using tracklet matches. */
void MultiViewAssociator::tryIDCorrection()
{
    std::stringstream debugMsg;
    std::unique_lock<std::mutex> peerLock(m_peerTargetDBMutex);

    for (const auto &egoTargetPair : m_egoTargetsByID) {
        TargetId egoTargetId = egoTargetPair.first;
        std::shared_ptr<NvTrackedObject> egoTarget = egoTargetPair.second;
        /// Use the extended ego history window, as in tryPeerReAssoc.
        Tracklet egoTracklet;
        int egoMaxSize = m_params.m_maxPeerTrackletSize * m_params.m_egoTrackletHistoryMultiplier;
        populateTracklet(egoTarget, &egoTracklet, egoMaxSize);

        /// Skip if not active or just acquired an ID.
        if (egoTarget->acquiredID()) continue;
        if (egoTarget->becameActive()) continue;
        if (egoTarget->getTrackerState() != TRACKER_STATE::ACTIVE) continue;

        if (m_params.m_debugVerboseLevel > 1) {
            debugMsg << "systemTime: " << Time().getStrSystemTime() << ", ";
            debugMsg << "time: " << Time().getStrTimeRelativeTo(m_startTime) << ", ";
            debugMsg << "frame: " << m_frameNum << ", ";
            debugMsg << "[MatchStatus] egoCam" << m_camId << ", ";
            debugMsg << "ID" << egoTargetId << ", {";
        }

        for (const auto &peerCamPair : m_peerTargetDB) {
            for (const auto &peerTargetPair : peerCamPair.second) {
                const auto &peerTarget = peerTargetPair.second;

                /// Skip if different IDs or the peer target's tracklet is not recent.
                if (peerTarget.targetId != egoTargetId) continue;
                /// Skip cross-class peers when class-match gating is enabled.
                if (m_params.m_checkClassMatch && egoTarget->getTargetInfo().classId != peerTarget.classId) continue;
                const bool idcFresh = (peerTarget.frameNum + m_params.m_maxPeerFrameDiff4NoDet >= m_frameNum);
                if (m_params.m_debugVerboseLevel > 0 && m_frameAlignLog.is_open() && !idcFresh) {
                    m_frameAlignLog << "frame: " << m_frameNum
                        << ", [IDCorrection/Stale] egoID=" << int(egoTargetId)
                        << ", peerCam=" << peerCamPair.first
                        << ", peerFrame=" << peerTarget.frameNum
                        << ", age=" << (static_cast<int64_t>(m_frameNum) - static_cast<int64_t>(peerTarget.frameNum))
                        << "\n" << std::flush;
                }
                if (!idcFresh) continue;

                NvMOTFrameNum commonFrames = 0;
                float matchScore = computeTrackletMatchScore(
                    egoTracklet, peerTarget.tracklet, egoTarget->usesProjectiveEstimator(), &commonFrames
                );

                m_mv_stats.idc_pairs_evaluated++;
                if (commonFrames == 0) {
                    m_mv_stats.idc_no_overlap++;
                } else if (int(commonFrames) < m_params.m_minCommonFrames4MatchScore) {
                    m_mv_stats.idc_partial_overlap++;
                } else if (matchScore < m_params.m_minPeerTrackletMatchScore) {
                    m_mv_stats.idc_score_fail++;
                }

                if (m_params.m_debugVerboseLevel > 0 && m_frameAlignLog.is_open()) {
                    m_frameAlignLog << "frame: " << m_frameNum
                        << ", [IDCorrection/Score] egoID=" << int(egoTargetId)
                        << ", peerCam=" << peerCamPair.first
                        << ", ego_pos=(" << egoTarget->getTargetInfo().pt_world_feet(0)
                        << "," << egoTarget->getTargetInfo().pt_world_feet(1) << ")"
                        << ", peer_pos=(" << peerTarget.nextState(0) << "," << peerTarget.nextState(1) << ")"
                        << ", egoTk=[";
                    for (size_t i = 0; i < egoTracklet.size(); ++i)
                        m_frameAlignLog << (i ? "," : "") << egoTracklet[i].frameNum;
                    m_frameAlignLog << "], peerTk=[";
                    for (size_t i = 0; i < peerTarget.tracklet.size(); ++i)
                        m_frameAlignLog << (i ? "," : "") << peerTarget.tracklet[i].frameNum;
                    m_frameAlignLog << "]"
                        << ", commonFrames=" << commonFrames
                        << ", matchScore=" << matchScore
                        << "\n" << std::flush;
                }

                if (matchScore == 0.f) continue;
                if (m_params.m_debugVerboseLevel > 1) {
                    debugMsg << "(peerCam" << peerCamPair.first << ", ";
                    debugMsg << "matchScore: " << matchScore << ", ";
                }
                if (matchScore > m_params.m_minPeerTrackletMatchScore) {
                    if (m_params.m_debugVerboseLevel > 1) {
                        debugMsg << "match: SUCCESS)";
                    }
                }
                else {
                    if (m_params.m_debugVerboseLevel > 1) {
                        debugMsg << "match: FAILED, ";
                    }
                    /// Keep the ID when the target needs reid-service reassociation and is older than the peer target.
                    if ( egoTarget->needToReAssocFromReidService() && (egoTarget->getAge() > peerTarget.age + m_frameNum - peerTarget.frameNum)) {
                        if (m_params.m_debugVerboseLevel > 1) {
                            debugMsg << "action: keep ID)";
                        }
                    }
                    else {
                        if (m_params.m_debugVerboseLevel > 1) {
                            debugMsg << "action: drop ID)";
                        }
                        /// TODO: Define the behavior when peer re-association does not adopt an ID.
                        if (tryPeerReAssoc(egoTarget)) {
                            m_adoptedIdMap[egoTarget] = egoTargetId;
                        }
                        else {
                            egoTarget->setNeedTargetIDAssigned();
                        }
                    }
                }
            }
        }
        if (m_params.m_debugVerboseLevel > 1) {
            debugMsg << "}\n";
        }
    }
    peerLock.unlock();

    /// Print the debug message to the log file.
    if (m_params.m_debugVerboseLevel > 1) {
        std::unique_lock<std::mutex> commLock(m_commMutex);
        m_logFile << debugMsg.str() << std::flush;
        commLock.unlock();
    }
}

/**
 * @brief Convert a tracker state to a string.
 * @param[in] state Tracker state.
 * @return Name of the tracker state.
 */
std::string MultiViewAssociator::state2string(TRACKER_STATE state)
{
    switch (state)
    {
        case EMPTY: return "EMPTY";
        case ACTIVE: return "ACTIVE";
        case INACTIVE: return "INACTIVE";
        case TENTATIVE: return "TENTATIVE";
        case PROJECTED: return "PROJECTED";
        case QUASIACTIVE: return "QUASIACTIVE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Log a processed peer message.
 * @param[in] camId Camera ID.
 * @param[in] targetId Target ID.
 * @param[in] msg_type Message text for logging.
 */
void MultiViewAssociator::messageProcessLog(CamId camId, TargetId targetId, std::string msg_type)
{
    if (m_params.m_debugVerboseLevel < 1) return;
    int numTargets = 0, trackletSize = 0;
    std::string complement;
    if (m_peerTargetDB.find(camId) != m_peerTargetDB.end()) {
        numTargets = m_peerTargetDB[camId].size();
        if (m_peerTargetDB[camId].find(targetId) != m_peerTargetDB[camId].end())
        {
            trackletSize = m_peerTargetDB[camId][targetId].tracklet.size();
            msg_type = "SUCCESS: target, type: " + msg_type;
        }
    }
    std::unique_lock<std::mutex> lock(m_commMutex);
    m_logFile << "systemTime: " << Time().getStrSystemTime() << ", ";
    m_logFile << "time: " << Time().getStrTimeRelativeTo(m_startTime) << ", ";
    m_logFile << "frame: " << std::setw(1) << m_frameNum << ", " << msg_type;
    m_logFile << "camID: " << std::setw(1) << camId << ", ";
    m_logFile << "targetID: " << std::setw(1) << (int)targetId << ", ";
    if (trackletSize > 0) m_logFile << "frameNum: " << std::setw(1) << m_peerTargetDB[camId][targetId].frameNum << ", ";
    if (trackletSize > 0) m_logFile << "age: " << std::setw(1) << m_peerTargetDB[camId][targetId].age << ", ";
    m_logFile << "trackletSize: " << std::setw(1) << trackletSize << ", ";
    m_logFile << "numTargets = " << std::setw(1) << numTargets << ", ";
    if (trackletSize > 0) m_logFile << "visibility: " << m_peerTargetDB[camId][targetId].visibility << ", ";
    if (trackletSize > 0) m_logFile << "distFromCam: " << m_peerTargetDB[camId][targetId].distFromCam << ", ";
    m_logFile << "activeIDs: [ ";
    for (auto &target : m_peerTargetDB[camId]) m_logFile << target.first << " ";
    m_logFile << "]\n" << std::flush;
    lock.unlock();
}

/**
 * @brief Check whether a camera or target entry exists in the peer-target database.
 * @param[in] camId Camera ID.
 * @param[in] targetId Target ID.
 * @param[in] msg_type Message type name for logging.
 * @param[in] onlyCam If true, check only for the camera ID.
 * @return True if the entry is in the peer target database.
 */
bool MultiViewAssociator::checkEntryInPeerTargetDB(CamId camId, TargetId targetId, std::string msg_type, bool onlyCam)
{
    std::string supplement = "SUCCESS: target, type";
    bool available = true;
    if (m_peerTargetDB.find(camId) == m_peerTargetDB.end()) {
        supplement = "WARNING: MultiViewAssociator::checkEntryInPeerTargetDB()::line" + std::to_string(__LINE__) + ": ";
        supplement += "camera, type: ";
        available = false;
    }
    auto peerTargtList = m_peerTargetDB.at(camId);
    if (peerTargtList.find(targetId) == peerTargtList.end() && !onlyCam) {
        supplement = "WARNING: MultiViewAssociator::checkEntryInPeerTargetDB()::line" + std::to_string(__LINE__) + ": ";
        supplement += "target, type: ";
        available = false;
    }
    if (!available) messageProcessLog(camId, targetId, supplement + msg_type);
    return available;
}

/**
 * @brief Send a target tracklet using the communicator.
 * @param[in] target Target whose trajectory buffer is serialized.
 */
void MultiViewAssociator::sendTracklet(std::shared_ptr<NvTrackedObject> target)
{
    Payload trackletPayload = generateTrackletPayload(target);
    m_communicator->requestToSend(trackletPayload);
}

/**
 * @brief Send target states using the communicator.
 * @param[in] target Target whose states are serialized.
 */
void MultiViewAssociator::sendStates(std::shared_ptr<NvTrackedObject> target)
{
    Payload statesPayload = generateStatesPayload(target);
    m_communicator->requestToSend(statesPayload);
}

/**
 * @brief Send an adopted-ID message using the communicator.
 * @param[in] target Target whose adopted ID is sent.
 * @param[in] oldTargetId Previous target ID.
 */
void MultiViewAssociator::sendAdoptedId(std::shared_ptr<NvTrackedObject> target, TargetId oldTargetId)
{
    /// The input target pointer is assumed to have the new ID.
    Payload adoptedIdPayload = generateAdoptedIdPayload(target, oldTargetId);
    m_communicator->requestToSend(adoptedIdPayload);
}

/** @brief Send a synchronization message using the communicator. */
void MultiViewAssociator::sendSync()
{
    Payload syncPayload = generateSyncPayload();
    m_communicator->requestToSend(syncPayload);
}

/** @brief Return the payload message clock timestamp. */
TimeStamp MultiViewAssociator::currentTime() const
{
	TimeStamp localTimestamp = Time().getUnixTimestampNanoSeconds();
    if (m_timeMeta.frameTimeValid)
		return static_cast<TimeStamp>(m_timeMeta.frameTimeStamp) + localTimestamp - m_timeMeta.localTimestamp;
	return localTimestamp;
}

/**
 * @brief Build a tracklet message for a target.
 * @param[in] target Target whose trajectory buffer is serialized.
 */
Payload MultiViewAssociator::generateTrackletPayload(std::shared_ptr<NvTrackedObject> target)
{
    Payload trackletPayload = {};
    TargetInfo targetInfo = target->getTargetInfo();
    trackletPayload.type = TRACKLET;
    trackletPayload.frameNum = m_frameNum;
    trackletPayload.fromCamId = m_camId;
    trackletPayload.targetId = target->getTargetID();
    trackletPayload.classId = target->getClassId();
    trackletPayload.age = target->getAge();
    trackletPayload.state = std::get<0>(target->getStatesWithCov());
    trackletPayload.nextState = std::get<0>(target->getPredictedStatesWithCov());
    trackletPayload.feetCov = target->getLocCovariance();
    trackletPayload.visibility = std::min(targetInfo.visibility, targetInfo.imgBorderCroppingVisibility);
    trackletPayload.distFromCam = target->getDistFromCam();
    trackletPayload.targetIDTimeStamp = target->getTargetIDTimeStamp();
    trackletPayload.messageTimeStamp = currentTime();
    trackletPayload.height = target->getTargetHeight();
    trackletPayload.heightConfidence = target->getHeightConfidence();
    trackletPayload.tBbox = targetInfo.tBbox;
    populateTracklet(target, &trackletPayload.tracklet);
    /// Copy the ReID embedding only when enablePeerReidEmbedding is true.
    if (m_params.m_enablePeerReidEmbedding)
        trackletPayload.reidEmbedding = target->getPersistentReidFeature();
    return std::move(trackletPayload);
}

/**
 * @brief Build a state-update message for a target.
 * @param[in] target Target whose states are serialized.
 */
Payload MultiViewAssociator::generateStatesPayload(std::shared_ptr<NvTrackedObject> target)
{
    Payload statesPayload = {};
    TargetInfo targetInfo = target->getTargetInfo();
    statesPayload.type = STATE_UPDATE;
    statesPayload.frameNum = m_frameNum;
    statesPayload.fromCamId = m_camId;
    statesPayload.targetId = target->getTargetID();
    statesPayload.classId = target->getClassId();
    statesPayload.age = target->getAge();
    statesPayload.state = std::get<0>(target->getStatesWithCov());
    statesPayload.nextState = std::get<0>(target->getPredictedStatesWithCov());
    statesPayload.feetCov = target->getLocCovariance();
    statesPayload.visibility = std::min(targetInfo.visibility, targetInfo.imgBorderCroppingVisibility);
    statesPayload.distFromCam = target->getDistFromCam();
    statesPayload.messageTimeStamp = currentTime();
    statesPayload.height = target->getTargetHeight();
    statesPayload.heightConfidence = target->getHeightConfidence();
    statesPayload.tBbox = targetInfo.tBbox;
    statesPayload.tracklet.push_back(
        TrackletPoint(
            m_frameNum, targetInfo.pt_world_feet(0), targetInfo.pt_world_feet(1),
            targetInfo.pt_world_feet_cov(0,0), targetInfo.pt_world_feet_cov(0,1), targetInfo.pt_world_feet_cov(1,1)
        )
    );
    return std::move(statesPayload);
}

/**
 * @brief Build an adopted-ID message for a target.
 * @param[in] target Target whose adopted ID is sent.
 * @param[in] oldTargetId Previous target ID.
 */
Payload MultiViewAssociator::generateAdoptedIdPayload(std::shared_ptr<NvTrackedObject> target, TargetId oldTargetId)
{
    /// The input target pointer is assumed to have the new ID.
    Payload adoptedIdPayload = {};
    adoptedIdPayload.type = ADOPTED_ID;
    adoptedIdPayload.fromCamId = m_camId;
    adoptedIdPayload.frameNum = m_frameNum;
    adoptedIdPayload.targetId = oldTargetId;
    adoptedIdPayload.adoptedId = target->getTargetID();
    adoptedIdPayload.classId = target->getClassId();
    adoptedIdPayload.targetIDTimeStamp = target->getTargetIDTimeStamp();
    adoptedIdPayload.messageTimeStamp = currentTime();
    return std::move(adoptedIdPayload);
}

/** @brief Build a synchronization message for the current frame. */
Payload MultiViewAssociator::generateSyncPayload()
{
    Payload syncPayload = {};
    syncPayload.type = SYNC;
    syncPayload.fromCamId = m_camId;
    syncPayload.frameNum = m_frameNum;
    syncPayload.messageTimeStamp = currentTime();
    return std::move(syncPayload);
}

/** @brief Process received messages through the current frame. */
void MultiViewAssociator::processReceivedMessagesToCurrentFrame()
{
    /// For message processing.
    Payload trackletPayload = {}, statesPayload = {}, adoptedIdPayload = {}, syncPayload = {};
    std::unique_lock<std::mutex> peerLock(m_peerTargetDBMutex); peerLock.unlock();
    bool continueProcessing = true;
    PeerTarget peerTarget;
    TargetId newId, oldId;
    std::string msg;

    m_logFile << "systemTime: " << Time().getStrSystemTime() << ", ";
    m_logFile << "time: " << Time().getStrTimeRelativeTo(m_startTime);
    m_logFile << " - Running loop for cam " << m_camId << "\n" << std::flush;

    while (continueProcessing)
    {
        peerLock.lock();
        switch (m_communicator->available()) {
            case TRACKLET:
                m_communicator->get(trackletPayload);
                msg = "(Tracklet),    ";
                /// Check if the camId is in the PeerReAssocDB, and add the tracklet if so.
                if (!checkEntryInPeerTargetDB(trackletPayload.fromCamId, trackletPayload.targetId, msg, true)) break;
                peerTarget = m_peerTargetDB[trackletPayload.fromCamId][trackletPayload.targetId];
                peerTarget.cameraId = trackletPayload.fromCamId;
                peerTarget.targetId = trackletPayload.targetId;
                peerTarget.targetIDTimeStamp = trackletPayload.targetIDTimeStamp;
                peerTarget.classId = trackletPayload.classId;
                peerTarget.age = trackletPayload.age;
                peerTarget.state = trackletPayload.state;
                peerTarget.nextState = trackletPayload.nextState;
                peerTarget.feetCov = trackletPayload.feetCov;
                peerTarget.visibility = trackletPayload.visibility;
                peerTarget.distFromCam = trackletPayload.distFromCam;
                peerTarget.tracklet = trackletPayload.tracklet;
                peerTarget.height = trackletPayload.height;
                peerTarget.heightConfidence = trackletPayload.heightConfidence;
                /// Copy the received ReID embedding onto the peer target.
                peerTarget.reidEmbedding = trackletPayload.reidEmbedding;
                {
                    /// Map peer tracklet frame indices to ego frame space.
                    const int64_t peerMsgFrame = static_cast<int64_t>(trackletPayload.frameNum);
                    int64_t egoRefFrame;
                    if (m_egoDeltaTime > 0) {
                        const int64_t peerDeltaTime = static_cast<int64_t>(trackletPayload.messageTimeStamp) -
                            static_cast<int64_t>(m_timeMeta.frameTimeStamp);
                        const double framePhase = static_cast<double>(peerDeltaTime) / static_cast<double>(
                            m_egoDeltaTime
                        );
                        egoRefFrame = static_cast<int64_t>(m_frameNum) + static_cast<int64_t>(std::llround(framePhase));
                    } else {
                        egoRefFrame = static_cast<int64_t>(m_frameNum) - 1;
                    }
                    m_peerFrameOffsetByCam[trackletPayload.fromCamId] = peerMsgFrame - egoRefFrame;
                    if (m_params.m_debugVerboseLevel > 0 && m_frameAlignLog.is_open()) {
                        const int64_t storedOffset = peerMsgFrame - egoRefFrame;
                        m_frameAlignLog << "frame: " << m_frameNum
                            << ", [FrameAlign/TRACKLET] fromCam=" << trackletPayload.fromCamId
                            << ", tgtID=" << int(trackletPayload.targetId)
                            << ", peerMsgFrame=" << peerMsgFrame
                            << ", egoRefFrame=" << egoRefFrame
                            << ", offset(peer-ego)=" << storedOffset
                            << ", tracklet(" << peerTarget.tracklet.size() << "pts):";
                        if (!peerTarget.tracklet.empty()) {
                            const int64_t oldestAge = peerMsgFrame - static_cast<int64_t>(
                                peerTarget.tracklet.front().frameNum
                            );
                            const int64_t newestAge = peerMsgFrame - static_cast<int64_t>(
                                peerTarget.tracklet.back().frameNum
                            );
                            m_frameAlignLog << " rawOldest=" << peerTarget.tracklet.front().frameNum
                                << "->ego=" << std::max(int64_t(0), egoRefFrame - oldestAge)
                                << ", rawNewest=" << peerTarget.tracklet.back().frameNum
                                << "->ego=" << std::max(int64_t(0), egoRefFrame - newestAge);
                        }
                        m_frameAlignLog << "\n" << std::flush;
                    }
                    for (auto &tp : peerTarget.tracklet) {
                        const int64_t ageInPeerFrames = peerMsgFrame - static_cast<int64_t>(tp.frameNum);
                        const int64_t egoFrame = egoRefFrame - ageInPeerFrames;
                        tp.frameNum = static_cast<NvMOTFrameNum>(std::max(int64_t(0), egoFrame));
                    }
                }
                /// Messages are from the previous frame; reflect that in the freshness timestamp.
                peerTarget.frameNum = m_frameNum - 1;
                m_peerTargetDB[trackletPayload.fromCamId][trackletPayload.targetId] = peerTarget;
                m_peerLastMsgByCam[trackletPayload.fromCamId] = {
                    trackletPayload.messageTimeStamp, trackletPayload.frameNum
                };
                messageProcessLog(trackletPayload.fromCamId, trackletPayload.targetId, msg);
                break;
            case STATE_UPDATE:
                m_communicator->get(statesPayload);
                msg = "(StateUpdate), ";
                {
                    auto camIt = m_peerTargetDB.find(statesPayload.fromCamId);
                    if (camIt == m_peerTargetDB.end()) {
                        messageProcessLog(
                            statesPayload.fromCamId, statesPayload.targetId,
                            "WARNING: STATE_UPDATE from unregistered camera, skipping: " + msg
                        );
                        break;
                    }
                    if (camIt->second.find(statesPayload.targetId) == camIt->second.end()) {
                        /// TRACKLET was missed (e.g., peer started before us or message lost).
                        PeerTarget bootstrapped;
                        bootstrapped.cameraId = statesPayload.fromCamId;
                        bootstrapped.targetId = statesPayload.targetId;
                        bootstrapped.targetIDTimeStamp = 0;
                        camIt->second[statesPayload.targetId] = std::move(bootstrapped);
                        if (m_frameAlignLog.is_open()) {
                            m_frameAlignLog << "frame: " << m_frameNum
                                << ", [Bootstrap/STATE_UPDATE] fromCam=" << statesPayload.fromCamId
                                << ", tgtID=" << int(statesPayload.targetId) << "\n" << std::flush;
                        }
                    }
                }
                /// Update the target in the PeerReAssocDB with the state update.
                peerTarget = m_peerTargetDB[statesPayload.fromCamId][statesPayload.targetId];
                peerTarget.age = statesPayload.age;
                peerTarget.classId = statesPayload.classId;
                peerTarget.state = statesPayload.state;
                peerTarget.nextState = statesPayload.nextState;
                peerTarget.feetCov = statesPayload.feetCov;
                peerTarget.visibility = statesPayload.visibility;
                peerTarget.distFromCam = statesPayload.distFromCam;
                peerTarget.height = statesPayload.height;
                peerTarget.heightConfidence = statesPayload.heightConfidence;
                peerTarget.tBbox = statesPayload.tBbox;
                /// Update the tracklet with the new state.
                peerTarget.tracklet.push_back(statesPayload.tracklet.back());
                if (int(peerTarget.tracklet.size()) > m_params.m_maxPeerTrackletSize) peerTarget.tracklet.pop_front();
                {
                    /// Translate using the stored per-camera frame offset established at TRACKLET receipt.
                    int64_t egoFrame;
                    auto offsetIt = m_peerFrameOffsetByCam.find(statesPayload.fromCamId);
                    if (offsetIt != m_peerFrameOffsetByCam.end()) {
                        egoFrame = static_cast<int64_t>(statesPayload.frameNum) - offsetIt->second;
                    } else {
                        /// Assign the current ego frame as a best effort when no TRACKLET was received yet.
                        egoFrame = static_cast<int64_t>(m_frameNum);
                    }
                    peerTarget.tracklet.back().frameNum = static_cast<NvMOTFrameNum>(std::max(int64_t(0), egoFrame));
                    if (m_params.m_debugVerboseLevel > 0 && m_frameAlignLog.is_open()) {
                        m_frameAlignLog << "frame: " << m_frameNum
                            << ", [FrameAlign/STATE_UPDATE] fromCam=" << statesPayload.fromCamId
                            << ", tgtID=" << int(statesPayload.targetId)
                            << ", peerMsgFrame=" << statesPayload.frameNum
                            << ", offset=" << (offsetIt != m_peerFrameOffsetByCam.end() ? offsetIt->second : 0)
                            << ", egoFrame=" << peerTarget.tracklet.back().frameNum
                            << "\n" << std::flush;
                    }
                }
                peerTarget.frameNum = m_frameNum - 1;
                m_peerTargetDB[statesPayload.fromCamId][statesPayload.targetId] = peerTarget;
                m_peerLastMsgByCam[statesPayload.fromCamId] = {statesPayload.messageTimeStamp, statesPayload.frameNum};
                messageProcessLog(statesPayload.fromCamId, statesPayload.targetId, msg);
                break;
            case ADOPTED_ID:
                m_communicator->get(adoptedIdPayload);
                oldId = adoptedIdPayload.targetId;
                newId = adoptedIdPayload.adoptedId;
                msg = "(AdoptID) new: " + std::to_string(newId) + ", ";
                if (!checkEntryInPeerTargetDB(adoptedIdPayload.fromCamId, oldId, msg)) break;
                /// Update the target in the PeerReAssocDB with the new ID by replacement if needed.
                m_peerTargetDB.at(adoptedIdPayload.fromCamId)[newId] = std::move(
                    m_peerTargetDB.at(adoptedIdPayload.fromCamId)[oldId]
                );
                /// Propagate new ID and its creation timestamp.
                m_peerTargetDB.at(adoptedIdPayload.fromCamId)[newId].targetId = newId;
                m_peerTargetDB.at(adoptedIdPayload.fromCamId)[newId].targetIDTimeStamp =
                    adoptedIdPayload.targetIDTimeStamp;
                if (oldId != newId) m_peerTargetDB.at(adoptedIdPayload.fromCamId).erase(oldId);
                /// FIXME: Update ego targets that still use the old ID, and define behavior when the new ID is already in use.
                messageProcessLog(adoptedIdPayload.fromCamId, newId, msg);
                break;
            case SYNC:
                m_communicator->get(syncPayload);
                msg = "                       (Sync),        ";
                m_receivedSyncFromCameras[syncPayload.fromCamId] = syncPayload.frameNum;
                m_peerLastMsgByCam[syncPayload.fromCamId] = {syncPayload.messageTimeStamp, syncPayload.frameNum};
                messageProcessLog(syncPayload.fromCamId, -1, msg);
                continueProcessing = false;
                for (const auto &peerCamPair : m_receivedSyncFromCameras) {
                    if (peerCamPair.second + 1 < m_frameNum) {
                        continueProcessing = true;
                        break;
                    }
                }
                break;
            case NONE:
                if (!m_params.m_enableMsgSync) continueProcessing = false;
                /// The first frames still process messages so the initial timestamps are available.
                if (m_frameNum < 2) continueProcessing = true;
                break;
            default: break;
        }
        peerLock.unlock();
    }
}

/** @brief Return a copy of the peer-target database. */
const PeerTargetDB MultiViewAssociator::getPeerTargetDB()
{
    std::unique_lock<std::mutex> lock(m_peerTargetDBMutex);
    PeerTargetDB peerTargetDB = m_peerTargetDB;
    lock.unlock();

    return peerTargetDB;
}

/** @brief Remove expired tracklet data from the peer-target database. */
void MultiViewAssociator::cleanupPeerTargetDB()
{
    std::stringstream debugMsg;
    std::unique_lock<std::mutex> peerLock(m_peerTargetDBMutex);
    for (const auto &peerCamPair : m_peerTargetDB) {
        CamId cam = peerCamPair.first;
        for (auto peerTargetIt = m_peerTargetDB.at(cam).begin(); peerTargetIt != m_peerTargetDB.at(cam).end(); ) {
            auto &peerTarget = peerTargetIt->second;
            if (peerTarget.tracklet.size() > 0) {
                /// If the first point in the tracklet is older than m_maxPeerTrackletSize frames, remove it.
                if (peerTarget.tracklet.front().frameNum + m_params.m_maxPeerTrackletSize <= m_frameNum) {
                    peerTarget.tracklet.pop_front();
                }
                if (peerTarget.tracklet.empty()) {
                    peerTargetIt = m_peerTargetDB.at(cam).erase(peerTargetIt);
                    if (m_params.m_debugVerboseLevel > 1) {
                        debugMsg << "systemTime: " << Time().getStrSystemTime() << ", ";
                        debugMsg << "time: " << Time().getStrTimeRelativeTo(m_startTime)
                                 << ", frame: " << m_frameNum << ", ";
                        debugMsg << "[PeerTargetDB] (cam" << cam << ", ID" << (int)peerTarget.targetId << ") tracklet is empty, removed\n";
                    }
                }
                else {
                    ++peerTargetIt;
                }
            }
            else {
                peerTargetIt = m_peerTargetDB.at(cam).erase(peerTargetIt);
                if (m_params.m_debugVerboseLevel > 1) {
                    debugMsg << "systemTime: " << Time().getStrSystemTime() << ", ";
                    debugMsg << "time: " << Time().getStrTimeRelativeTo(m_startTime)
                             << ", frame: " << m_frameNum << ", ";
                    debugMsg << "[PeerTargetDB] (cam" << cam << ", ID" << (int)peerTarget.targetId << ") tracklet is empty, removed\n";
                }
            }
        }
    }
    peerLock.unlock();
    /// Print the debug message to the log file.
    if (m_params.m_debugVerboseLevel > 1) {
        std::unique_lock<std::mutex> commLock(m_commMutex);
        m_logFile << debugMsg.str() << std::flush;
        commLock.unlock();
    }
}
