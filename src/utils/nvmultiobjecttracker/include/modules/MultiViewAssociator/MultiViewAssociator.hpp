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

#ifndef NVMULTIOBJECTTRACKERLIB_MULTIVIEWASSOCIATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_MULTIVIEWASSOCIATOR_HPP_

#include "modules/MultiViewAssociator/Communicator_factory.hpp"
#include "modules/MultiViewAssociator/IMultiViewAssociator.hpp"

/** @brief Store multi-view association configuration parameters. */
class MultiViewAssociatorParams : public NvTrackerParams
{
public:
	/** @brief Initialize multi-view association parameters with default values. */
	MultiViewAssociatorParams()
	{
		m_debugVerboseLevel = 0;
        m_multiViewAssociatorType = DUMMY_MULTI_VIEW_ASSOCIATOR;
        m_checkClassMatch = true;
        m_enableLatePeerReAssoc = true;
        m_enableIDCorrection = true;
        m_enableSeeThrough = true;
        m_enableMsgSync = false;
        m_enablePeerReidEmbedding = false;
        m_maxPeerTrackletSize = 50;
        m_recentlyActiveAge = 200;
        m_minCommonFrames4MatchScore = 2;
        m_minPeerTrackletMatchScore = 0.5;
        m_maxPeerToPredDistance4Fusion = 1.0;
        m_minPeerVisibility4Fusion = 0.25;
        m_maxTrackletMatchingTimeSearchRange = 1;
        m_maxPeerFrameDiff4NoDet = 2;
        m_communicatorInitSleepTime = 0;
        m_minReidServiceMatchConfidence = 0.58;
        m_egoTrackletHistoryMultiplier = 1;
        m_minMatchingScore4PartialOverlap = 1.0f;
        m_partialTimeSearchMultiplier = 1;
        m_minCommonFrames4PartialOverlap = 1;
        m_maxSeqBridgeDist = 0.0f;
        m_minMatchingScore4RegularReAssoc = 0.0f;
	}

	/**
	 * @brief Initialize multi-view association parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	MultiViewAssociatorParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
		/// Read cross-cutting BaseConfig params (startup-only; overrides in-section default).
		const YAML::Node &baseNode = configRoot["BaseConfig"];
		if (baseNode && baseNode["checkClassMatch"])
			m_checkClassMatch = (bool)baseNode["checkClassMatch"].as<int>();
	}

	/**
	 * @brief Initialize multi-view association parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	MultiViewAssociatorParams(const YAML::Node &fn)
	{
		read(fn);
	}

	/**
	 * @brief Update multi-view association parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override
	{
		*this = MultiViewAssociatorParams();
		if(!fn.size()) { return;}

		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
            if (paramKey == "debugVerboseLevel") {
                m_debugVerboseLevel = (int)fit->second.as<float>();
            }
            else if (paramKey == "multiViewAssociatorType") {
                m_multiViewAssociatorType = (MULTI_VIEW_ASSOCIATOR_TYPES)((int)fit->second.as<float>());
            }
            else if (paramKey == "enableLatePeerReAssoc") {
                m_enableLatePeerReAssoc = (bool)fit->second.as<float>();
            }
            else if (paramKey == "enableIDCorrection") {
                m_enableIDCorrection = (bool)fit->second.as<float>();
            }
            else if (paramKey == "enableSeeThrough") {
                m_enableSeeThrough = (bool)fit->second.as<float>();
            }
            else if (paramKey == "enableMsgSync") {
                m_enableMsgSync = (bool)fit->second.as<float>();
            }
            else if (paramKey == "maxPeerTrackletSize") {
                m_maxPeerTrackletSize = (int)fit->second.as<float>();
            }
            else if (paramKey == "recentlyActiveAge") {
                m_recentlyActiveAge = (int)fit->second.as<float>();
            }
            else if (paramKey == "minCommonFrames4MatchScore") {
                m_minCommonFrames4MatchScore = std::max((int)fit->second.as<float>(), 2);
            }
            else if (paramKey == "maxPeerToPredDistance4Fusion") {
                m_maxPeerToPredDistance4Fusion = fit->second.as<float>();
            }
            else if (paramKey == "minPeerVisibility4Fusion") {
                m_minPeerVisibility4Fusion = fit->second.as<float>();
            }
            else if (paramKey == "minPeerTrackletMatchScore") {
                m_minPeerTrackletMatchScore = fit->second.as<float>();
            }
            else if (paramKey == "maxTrackletMatchingTimeSearchRange") {
                m_maxTrackletMatchingTimeSearchRange = (int)fit->second.as<float>();
            }
            else if (paramKey == "maxPeerFrameDiff4NoDet") {
                m_maxPeerFrameDiff4NoDet = std::max((int)fit->second.as<float>(), 2);
            }
            else if (paramKey == "communicatorInitSleepTime") {
                m_communicatorInitSleepTime = (int)fit->second.as<float>();
            }
            else if (paramKey == "minReidServiceMatchConfidence") {
                m_minReidServiceMatchConfidence = fit->second.as<float>();
            }
            else if (paramKey == "egoTrackletHistoryMultiplier") {
                m_egoTrackletHistoryMultiplier = std::max(1, (int)fit->second.as<float>());
            }
            else if (paramKey == "minMatchingScore4PartialOverlap") {
                m_minMatchingScore4PartialOverlap = fit->second.as<float>();
            }
            else if (paramKey == "partialTimeSearchMultiplier") {
                m_partialTimeSearchMultiplier = std::max(1, (int)fit->second.as<float>());
            }
            else if (paramKey == "minCommonFrames4PartialOverlap") {
                m_minCommonFrames4PartialOverlap = std::max(1, (int)fit->second.as<float>());
            }
            else if (paramKey == "maxSeqBridgeDist") {
                m_maxSeqBridgeDist = fit->second.as<float>();
            }
            else if (paramKey == "minMatchingScore4RegularReAssoc") {
                m_minMatchingScore4RegularReAssoc = fit->second.as<float>();
            }
            else if (paramKey == "enablePeerReidEmbedding") {
                m_enablePeerReidEmbedding = (bool)fit->second.as<float>();
            }
            else std::cerr << "Unknown parameter: " << paramKey << "\n";
		}
	}

    MULTI_VIEW_ASSOCIATOR_TYPES m_multiViewAssociatorType; //!< Selects the multi-view-associator implementation.
    int m_debugVerboseLevel; //!< Controls debug message verbosity.
    bool m_enableLatePeerReAssoc; //!< Enables late peer reassociation.
    bool m_enableIDCorrection; //!< Enables correction of target IDs from peer messages.
    bool m_enableSeeThrough; //!< Enables peer-state fusion for occluded targets.
    bool m_enableMsgSync; //!< Enables synchronization messages between cameras.
    bool m_checkClassMatch; //!< Requires peer targets to have the same class when enabled.
    int m_maxPeerTrackletSize; //!< Specifies the maximum peer tracklet size used for transmission and storage.
    int m_recentlyActiveAge; //!< Specifies the target age limit for late peer reassociation.
    int m_minCommonFrames4MatchScore; //!< Specifies the minimum common-frame count for tracklet similarity.
    int m_maxTrackletMatchingTimeSearchRange; //!< Specifies the maximum temporal offset searched during tracklet matching.
    float m_maxPeerToPredDistance4Fusion; //!< Specifies the maximum distance from an ego prediction to a peer measurement used for fusion.
    float m_minPeerVisibility4Fusion; //!< Specifies the minimum peer visibility for measurement fusion.
    float m_minPeerTrackletMatchScore; //!< Specifies the tracklet-similarity threshold for matching.
    int m_maxPeerFrameDiff4NoDet; //!< Specifies the maximum peer-frame difference when detections are unavailable.
    int m_communicatorInitSleepTime; //!< Specifies the delay after communicator initialization.
    float m_minReidServiceMatchConfidence; //!< Specifies the minimum ReID-service confidence for peer reassociation.
    int m_egoTrackletHistoryMultiplier; //!< Scales the ego comparison window relative to the maximum peer tracklet size.
    float m_minMatchingScore4PartialOverlap; //!< Specifies the minimum score for adopting a partial-overlap pair; 1.0 disables adoption.
    int m_partialTimeSearchMultiplier; //!< Scales the temporal search range for partial-overlap scoring.
    int m_minCommonFrames4PartialOverlap; //!< Specifies the minimum co-visible frames for partial-overlap adoption.
    float m_maxSeqBridgeDist; //!< Specifies the maximum junction distance in meters for sequential bridging; zero disables it.
    float m_minMatchingScore4RegularReAssoc; //!< Specifies the minimum score for regular reassociation; zero uses the shared score gate.
    bool m_enablePeerReidEmbedding; //!< Enables copying ReID embeddings into peer tracklet payloads.
};

/** @brief MultiViewAssociator class is responsible for associating the objects across multiple views. */
class MultiViewAssociator : public IMultiViewAssociator
{
public:
    /** @brief Construct an uninitialized multi-view associator. */
    MultiViewAssociator();

    /**
     * @brief Construct a multi-view associator from configuration.
     * @param[in] configFilePath Path to the configuration file.
     */
    MultiViewAssociator(const std::string &configFilePath);

    /** @brief Report statistics and destroy the multi-view associator. */
    ~MultiViewAssociator();

    /**
     * @brief Check if the MultiViewAssociator is initialized.
     * @return True if the multi-view associator is initialized.
     */
    bool isInitialized() override;

    /**
     * @brief Initialize peer-target state and the communicator.
     * @param[in] streamId Stream identifier for the ego camera.
     * @param[in] start Reference start time for timestamp conversion.
     * @param[in] localInstanceCommunicators Communicators for local streams.
     * @param[in] egoTargets Ego-camera targets managed by the tracker.
     * @param[in] objModelProjector Projector used to recover world-space target state.
     */
    void initialize(
        const StreamId streamId, system_clock::time_point start,
        std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators,
        std::shared_ptr<NvTrackedObjectList> egoTargets, std::shared_ptr<ObjectModelProjector> objModelProjector
    ) override;

    /**
     * @brief Synchronize the peer-target database with communicator messages.
     * @param[in] timeMeta Current frame timing metadata.
     */
    void update(const NvMOTTimeMeta &timeMeta) override;

    /**
     * @brief Try to reassociate a target with peer tracklets.
     * @param[in] target Target to match.
     * @return True when peer reassociation succeeds.
     */
    bool tryPeerReAssoc(std::shared_ptr<NvTrackedObject> target) override;

    /**
     * @brief Send the tracklet using the communicator.
     * @param[in] target Target whose trajectory buffer is serialized.
     */
    void sendTracklet(std::shared_ptr<NvTrackedObject> target) override;

    /**
     * @brief Send the states using the communicator.
     * @param[in] target Target whose states are serialized.
     */
    void sendStates(std::shared_ptr<NvTrackedObject> target) override;

    /**
     * @brief Send the AdoptedId message using the communicator.
     * @param[in] target Target whose adopted ID is sent.
     * @param[in] oldTargetId Previous target ID.
     */
    void sendAdoptedId(std::shared_ptr<NvTrackedObject> target, TargetId oldTargetId) override;

    /** @brief Send the synchronization message using the communicator. */
    void sendSync() override;

    /**
     * @brief Data association between ego detection and peer targets.
     * @param[in] detObjs Ego detections to match against peer targets.
     * @param[in] minMatchingScore4PeerAssocIou Minimum IoU required to match a detection to a peer target.
     * @return Detections that match a recent peer target.
     */
    std::deque<const DetectedObjInfo *> tryPeerDataAssoc(
        const std::deque<const DetectedObjInfo *> detObjs, const float minMatchingScore4PeerAssocIou
    ) override;

    /** @brief Return the peer-target database. */
    const PeerTargetDB getPeerTargetDB() override;

    /** @brief Return the communicator used for peer messages. */
    std::shared_ptr<ICommunicator> getCommunicator() { return m_communicator; }

private:
    /** @brief Process the received messages to the current frame (Sync to the previous frame). */
    void processReceivedMessagesToCurrentFrame();

    /**
     * @brief Build a tracklet message for a target.
     * @param[in] target Target whose trajectory buffer is serialized.
     */
    Payload generateTrackletPayload(std::shared_ptr<NvTrackedObject> target);

    /**
     * @brief Build a state-update message for a target.
     * @param[in] target Target whose states are serialized.
     */
    Payload generateStatesPayload(std::shared_ptr<NvTrackedObject> target);

    /**
     * @brief Build an adopted-ID message for a target.
     * @param[in] target Target whose adopted ID is sent.
     * @param[in] oldTargetId Previous target ID.
     */
    Payload generateAdoptedIdPayload(std::shared_ptr<NvTrackedObject> target, TargetId oldTargetId);

    /** @brief Build a synchronization message for the current frame. */
    Payload generateSyncPayload();

    /** @brief Try to assocate a received tracklet (PeerTarget) with the recently active targets. */
    void tryLatePeerReAssoc();

    /** @brief Try to re-assign the IDs of the targets in the PeerTargetDB. */
    void tryIDCorrection();

    /** @brief Cleanup tracklets in peerTargetDB. */
    void cleanupPeerTargetDB();

    /**
     * @brief Populate a tracklet from a target buffer.
     * @param[in] target Tracked object to read.
     * @param[out] tracklet Tracklet to populate.
     * @param[in] maxSize Maximum number of frames to include.
     */
    void populateTracklet(std::shared_ptr<NvTrackedObject> target, Tracklet* tracklet, int maxSize = -1);

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
    float computeTrackletMatchScore(
        const Tracklet &tracklet1, const Tracklet &tracklet2, bool useMahalanobis,
        NvMOTFrameNum* commonFrames = nullptr, float* partialScore = nullptr,
        int partialTimeRange = -1, int* partialCount = nullptr
    );

    /**
     * @brief Check if the entry (cam [and target]) is in the peerTargetDB.
     * @param[in] camId Camera ID.
     * @param[in] targetId Target ID.
     * @param[in] msg_type Message type name for logging.
     * @param[in] onlyCam If true, check only for the camera ID.
     * @return True if the entry is in the peer target database.
     */
    bool checkEntryInPeerTargetDB(CamId camId, TargetId targetId, std::string msg_type, bool onlyCam = false);

    /**
     * @brief Process the message and log the message.
     * @param[in] camId Camera ID.
     * @param[in] targetId Target ID.
     * @param[in] msg_type Message text for logging.
     */
    void messageProcessLog(CamId camId, TargetId targetId, std::string msg_type);

    /** @brief Get the payload message clock timestamp. */
    TimeStamp currentTime() const;

    /**
     * @brief Convert the tracker state to a string.
     * @param[in] state Tracker state.
     * @return Name of the tracker state.
     */
    std::string state2string(TRACKER_STATE state);

    PeerTargetDB m_peerTargetDB; //!< Stores targets received from peer cameras.
    std::ofstream m_logFile; //!< Debug log file and init timestamp.
    std::ofstream m_frameAlignLog; //!< Logs receipt-time frame translation for multi-instance verification.
    system_clock::time_point m_startTime; //!< Stores the associator reference start time.
    MultiViewAssociatorParams m_params; //!< Parameters for the MultiViewAssociator.
    std::mutex m_commMutex; //!< Mutex for the communicator.
    std::mutex m_peerTargetDBMutex; //!< Mutex for the peerTargetDB.
    std::map<TargetId, std::shared_ptr<NvTrackedObject>> m_egoTargetsByID; //!< Map of active targets.
    std::map<std::shared_ptr<NvTrackedObject>, TargetId> m_adoptedIdMap; //!< Map of targets that adopted new IDs and their old IDs.

    // [SYNC message tracking]
    std::map<CamId, NvMOTFrameNum> m_receivedSyncFromCameras; //!< Tracks which cameras sent SYNC for current frame.
    std::map<CamId, std::pair<TimeStamp, NvMOTFrameNum>> m_peerLastMsgByCam; //!< Per peer cam: last (messageTimeStamp, frameNum); frameNum may be 0.
    std::map<CamId, int64_t> m_peerFrameOffsetByCam; //!< (peerMsgFrame - egoRefFrame) at last TRACKLET; egoRefFrame from timestamps when m_egoDeltaTime>0.
    std::shared_ptr<NvTrackedObjectList> m_egoTargets; //!< Pointer to list of tracked objects in ego-cam.
    std::shared_ptr<ICommunicator> m_communicator; //!< Own communicator.
    std::shared_ptr<ObjectModelProjector> m_objModelProjector; //!< Object model projector.
    std::string m_configFilePath; //!< Path to the configuration file.
    StreamId m_streamId; //!< Stream ID.
    NvMOTFrameNum m_frameNum; //!< Frame number.
    NvMOTTimeMeta m_timeMeta; //!< Time metadata of the current frame.
    TimeStamp m_egoDeltaTime; //!< Time difference between the current frame and the previous frame.
    CamId m_camId; //!< Camera number.
    bool m_isInit; //!< Indicates whether the associator has been initialized.

    /** @brief Per-session multi-view association counters. */
    struct MV_Stats {
        uint64_t reAssoc_tries = 0; //!< Counts peer-reassociation attempts.
        uint64_t ra_pairs_evaluated = 0; //!< Counts target pairs evaluated for reassociation.
        uint64_t ra_no_overlap = 0; //!< Counts reassociation pairs without temporal overlap.
        uint64_t ra_partial_overlap = 0; //!< Counts reassociation pairs with partial temporal overlap.
        uint64_t ra_score_fail = 0; //!< Counts reassociation pairs rejected by score.
        uint64_t ra_adopted = 0; //!< Counts regularly adopted reassociation pairs.
        uint64_t ra_partial_adopted = 0; //!< Counts adopted partial-overlap reassociation pairs.
        uint64_t idc_pairs_evaluated = 0; //!< Counts target pairs evaluated for ID correction.
        uint64_t idc_no_overlap = 0; //!< Counts ID-correction pairs without temporal overlap.
        uint64_t idc_partial_overlap = 0; //!< Counts ID-correction pairs with partial temporal overlap.
        uint64_t idc_score_fail = 0; //!< Counts ID-correction pairs rejected by score.
        uint64_t ra_score_hist[10] = {}; //!< Stores the regular reassociation score histogram.
        uint64_t ra_partial_hist[10] = {}; //!< Stores the partial-overlap score histogram.
        uint64_t ra_partial_count_hist[5] = {}; //!< Stores the partial-overlap frame-count histogram.
        uint64_t noov_seq_lt2 = 0; //!< Counts sequential gaps shorter than two frames.
        uint64_t noov_seq_2_5 = 0; //!< Counts sequential gaps from two through five frames.
        uint64_t noov_seq_5_15 = 0; //!< Counts sequential gaps from five through fifteen frames.
        uint64_t noov_seq_15plus = 0; //!< Counts sequential gaps of at least fifteen frames.
        uint64_t noov_seq_bridge_ts_pass = 0; //!< Counts sequential bridges that pass timestamp checks.
        uint64_t noov_seq_bridge_adopted = 0; //!< Counts adopted sequential bridges.
        uint64_t bridge_seq_ts_fail = 0; //!< Counts sequential bridges that fail timestamp checks.
        double   bridge_seq_best_score_sum_at_ts_fail = 0.0; //!< Accumulates best scores for sequential bridges that fail timestamp checks.
        uint64_t peer_ts_blocked = 0; //!< Counts peer candidates blocked by timestamps.
        uint64_t regular_adopted_score_hist[10] = {}; //!< Stores scores for regular adopted matches.
        uint64_t partial_adopted_score_hist[10] = {}; //!< Stores scores for adopted partial-overlap matches.
        uint64_t height_ego_gt0 = 0; //!< Counts ego targets with positive height.
        uint64_t height_peer_gt0 = 0; //!< Counts peer targets with positive height.
        uint64_t height_both_valid = 0; //!< Counts pairs with valid ego and peer heights.
        uint64_t height_adopted_both_valid = 0; //!< Counts adopted pairs with valid ego and peer heights.
        double   height_adopted_delta_sum = 0.0; //!< Accumulates height differences for adopted pairs.
        uint64_t height_adopted_delta_hist[5] = {}; //!< Stores height differences for adopted pairs.
    } m_mv_stats; //!< Stores multi-view association counters.
};

#endif // NVMULTIOBJECTTRACKERLIB_MULTIVIEWASSOCIATOR_HPP_
