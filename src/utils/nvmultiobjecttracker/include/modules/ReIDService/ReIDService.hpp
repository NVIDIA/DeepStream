/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef REIDSERVICE_HPP_
#define REIDSERVICE_HPP_

#include <deque>
#include <map>
#include "modules/ReIDService/IReIDService.hpp"
#include <curl/curl.h>

/** @brief Stores configuration for the remote ReID service. */
class ReIDServiceConfigParams : public NvTrackerParams
{
public:
    /** @brief Construct a ReIDServiceConfigParams instance. */
    ReIDServiceConfigParams();

    /**
     * @brief Construct a ReIDServiceConfigParams instance.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] moduleName Configuration module name.
     */
    ReIDServiceConfigParams(const std::string &configFilePath, const std::string moduleName);

    /**
     * @brief Construct a ReIDServiceConfigParams instance.
     * @param[in] fn Parsed YAML node.
     */
    ReIDServiceConfigParams(const YAML::Node &fn);

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override;

    REID_SERVICE_TYPES m_reidServiceType; //!< Type of reid service.
    int m_debugVerboseLevel; //!< Debug message verbose mode level.
    bool m_enableBlocking; //!< Whether to enable blocking mode.
    float m_minTopClusterConfidence; //!< Minimum confidence for top cluster to be considered for reid service association.
    float m_minSecondClusterConfidence; //!< Minimum confidence for second cluster to be considered for ambiguity checking.
    float m_maxClusterVariance; //!< Maximum cluster variance to be considered for reid service association.
    int m_topK; //!< Maximum number of object ids to be returned by reid service association.
    int m_maxGroupCount; //!< Maximum number of vectors per object id to be considered for reid service association.
    float m_minVisibility4ReidServiceQuery; //!< Minimum visibility for reid service association query.
    std::string m_serviceAddress; //!< Address/hostname for the reid service.
    int m_servicePort; //!< Port for the reid service.
    std::vector<int> m_operateOnClassIds; //!< Class IDs used for ReID service association. An empty list enables association for all classes.
    bool m_enableIDCorrection; //!< Whether to enable ID correction.
    int m_minAssociationCount4IDCorrection; //!< Minimum number of successful associations for ID correction.
    float m_maxIou4IDCorrection; //!< Maximum IOU for ID correction.
    int m_maxInFlightRequests; //!< Maximum number of in-flight requests, requests beyond this are dropped.
    int m_curlConnectTimeout; //!< CURL connect timeout in seconds.
    int m_curlRequestTimeout; //!< CURL request timeout in seconds.
};

/** @brief Output of a successful reid service association. */
struct remoteTargetInfo
{
    NvMOTStreamId streamId; //!< Stream ID of the queried target.
    TrackerIndex targetIndex; //!< Tracker index of the queried target.
    TargetId originalTargetID; //!< Target ID at query time.
    TargetId remoteTargetID; //!< Target ID returned by the remote service.
    bool validRemoteTargetID; //!< Indicates whether the remote target ID can be adopted.
    std::map<TargetId, reidServiceAssocInfo> reidServiceAssocResponseMap; //!< Association details keyed by target ID.
};

/** @brief Holds CURL state and target metadata for one service request. */
struct CurlRequestTargetInfo {
    NvMOTStreamId streamId; //!< Stream ID of the queried target.
    TrackerIndex targetIndex; //!< Tracker index of the queried target.
    TargetId targetID; //!< Target ID at query time.
    std::string jsonPayload; //!< JSON request payload.
    std::string response; //!< Response text received from the service.
    struct curl_slist *headers = nullptr; //!< Per-request header list freed with curl_slist_free_all.
};

/** @brief Associates local targets with identities returned by a remote service. */
class ReIDService: public IReIDService
{
public:
    /**
     * @brief Construct a ReIDService instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     */
    ReIDService(const std::string &configFilePath);

    /** @brief Release resources owned by the ReIDService instance. */
    virtual ~ReIDService();

    /**
     * @brief Initialize variables and allocate memories.
     * @param[in] frameInfo Metadata about the source image, such as size and color.
     * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
     * @param[in] targetManager Target manager of the tracked objects.
     */
    void initialize(
        const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
        const std::shared_ptr<NvTrackedObjectManager> &targetManager
    ) override;

    /**
     * @brief Populate the capabilities reported to the tracker.
     * @param[out] paramsQuery The output query parameters.
     */
    void queryParams(NvMOTQuery &paramsQuery);

    /**
     * @brief Send query to reid service for all unmatched active objects.
     * @param[in] inputMeta Input frames keyed by stream ID.
     */
    void query(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta) override;

    /** @brief Update target ID using remoteTargetID from reid service. */
    void updateTargetIDs() override;

    /** @brief Get the configured ReID-service implementation type. */
    const REID_SERVICE_TYPES getReIDServiceType() override {return m_params.m_reidServiceType; }

protected:
    bool m_isInit; //!< Reid status if runtime variables initialized.
    ReIDServiceConfigParams m_params; //!< Config params for reid module.
    std::shared_ptr<NvTrackedObjectManager> m_targetManager; //!< Target manager of the tracked objects.
    std::map<CURL*, CurlRequestTargetInfo> m_curlRequestInfoMap; //!< Active CURL handles.
    //!< All CURL-related variables are used only by the worker thread.
    CURLM *m_curlMultiHandle; //!< CURL multi handle.
    std::string m_curlRequestUrl; //!< CURL request URL.

    // [Thread-related variables]
    std::thread m_workerThread; //!< Worker thread.
    std::atomic<bool> m_persistThreadRunning; //!< Indicates whether the worker thread should keep running.
    std::mutex m_inputLock; //!< Lock associated with input queue.
    std::condition_variable m_cond4In; //!< Wakes the worker when the input queue receives a request.
    std::queue<CurlRequestTargetInfo> m_inputQueue; //!< Queue for input.

    /** @brief Process queued remote ReID requests. */
    void workerThread();

    std::mutex m_outputLock; //!< Lock associated with output.
    std::condition_variable m_cond4Out; //!< Wakes the caller when the worker finishes a request.
    std::queue<remoteTargetInfo> m_outputQueue; //!< Queue for output.
    std::atomic<uint32_t> m_submittedReqCnt; //!< Number of submitted tasks.
    //!< Blocking mode uses this count to wait for the worker thread to finish.
    //!< These are accessed from both the query thread (producer) and the worker thread (consumer), so they must be atomic to avoid data races (UB).
    std::atomic<uint32_t> m_finishedReqCnt; //!< Keep track of how many tasks are finished.

    /** @brief Wait until all submitted service requests have completed. */
    void waitToFinish();

    // [ID correction]
    std::map<NvMOTStreamId, std::map<TargetId, std::pair<TargetId, int>>> m_idCorrectionMap; //!< Target-ID corrections keyed by stream ID.
    //!< Each nested key is the target ID to correct.
    //!< Value is a pair of successfully associated target ID and the number of times the target ID has been associated successfully.

    /**
     * @brief Apply a remote identity to a compatible local target.
     * @param[in] targetToBeCorrected Target whose identity may be corrected.
     * @param[in] existingTarget Existing target associated with the remote identity.
     * @param[in] associatedTargetID Target identifier returned by the service.
     * @return True when the correction is accepted and the existing target's ID is dropped; false otherwise.
     */
    bool tryIDCorrection(
        const std::shared_ptr<NvTrackedObject> targetToBeCorrected,
        const std::shared_ptr<NvTrackedObject> existingTarget, TargetId associatedTargetID
    );

    /**
     * @brief Serialize a ReID feature vector as JSON.
     * @param[in] reidFeatures ReID feature vector.
     * @param[in] reidFeatureSize Number of values in each ReID feature.
     */
    std::string convertReidFeaturesToJson(const float* reidFeatures, int reidFeatureSize);

    /**
     * @brief Send one synchronous HTTP POST request.
     * @param[in] jsonPayload JSON request payload.
     */
    std::string send_post_request(const std::string& jsonPayload);

    /**
     * @brief Parse a remote identity result from a service response.
     * @param[in] response Service response text.
     */
    remoteTargetInfo parseResponseForTargetID(const std::string& response);

    /**
     * @brief Add a request to the asynchronous CURL multi-handle.
     * @param[in] curlRequestInfoInput Request state to append to the active CURL set.
     * @return False when the CURL handle cannot be created; true otherwise, including when the request is dropped.
     */
    bool appendCurlRequest(const CurlRequestTargetInfo &curlRequestInfoInput);

    /**
     * @brief Collect the result of a completed CURL request.
     * @param[in] msg Completed CURL message.
     */
    void processCompletedCurlRequest(CURLMsg *msg);
};

#endif // REIDSERVICE_HPP_
