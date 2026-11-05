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

#include "modules/ReIDService/ReIDService.hpp"
#include "modules/utility/debug_common.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct a ReIDService instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
ReIDService::ReIDService(const std::string &configFilePath)
{
	m_params = ReIDServiceConfigParams(configFilePath, "ReIDService");
	m_curlMultiHandle = NULL;
	m_persistThreadRunning = false;
	m_isInit = false;
}

/**
 * @brief Initialize remote-service state and start the request worker.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void ReIDService::initialize(
	const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
	const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
	m_targetManager = targetManager;
	m_persistThreadRunning = true;
	m_submittedReqCnt = 0;
	m_finishedReqCnt = 0;
	/// Construct URL before starting worker thread to ensure it's available when needed.
	m_curlRequestUrl = "http://" + m_params.m_serviceAddress + ":" + std::to_string(m_params.m_servicePort) + "/search";
	m_workerThread = std::thread([this](){ workerThread(); });

	m_isInit = true;
}

/** @brief Release resources owned by the ReIDService instance. */
ReIDService::~ReIDService()
{
	if (m_isInit)
	{
		m_persistThreadRunning = false;
		/// Clear input queue and notify the waiting thread.
		std::unique_lock<std::mutex> inputLock(m_inputLock);
		m_inputQueue = std::queue<CurlRequestTargetInfo>();
		inputLock.unlock();
		m_cond4In.notify_one();
		m_workerThread.join();
	}
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void ReIDService::queryParams(NvMOTQuery &paramsQuery)
{
}

/** @brief Process queued remote ReID requests. */
void ReIDService::workerThread()
{
	curl_global_init(CURL_GLOBAL_DEFAULT);
	m_curlMultiHandle = curl_multi_init();
	if(!m_curlMultiHandle) {
		cout << "[ReIDService] Failed to initialize CURL" << endl;
		m_persistThreadRunning = false;
		m_cond4Out.notify_all();
		curl_global_cleanup();
		return;
	}

	int curlReqsInFlight = 0;
	/// Track the completed-transfer messages reported by libcurl.
	int msgsLeft = 0;
	CURLMsg *msg;
	while (m_persistThreadRunning)
	{
		/// Per-iteration try/catch: bare std::thread + an uncaught exception would call std::terminate() and abort the entire process.
		/// The throws below are intentional (loud diagnostics for unexpected states), but must be caught here so the issue is logged with full what() info and the worker keeps running rather than killing the pipeline.
		try
		{
			std::unique_lock<std::mutex> inputLock(m_inputLock);

			/// Wait for work or shutdown signal.
			m_cond4In.wait(
				inputLock,
				[this, &curlReqsInFlight, &msgsLeft]() { return !m_inputQueue.empty() || !m_persistThreadRunning || curlReqsInFlight || msgsLeft; }
			);

			/// Check for spurious wakeup!!
			if (!m_persistThreadRunning)
			{
				inputLock.unlock();
				break;
			}

			if(!m_inputQueue.empty())
			{
				/// Get an input.
				CurlRequestTargetInfo curlRequestInfo = std::move(m_inputQueue.front());
				m_inputQueue.pop();
				inputLock.unlock();
				/// Create a CURL request and add it to the m_curlMultiHandle.
				/// Cap-hit drops are handled inside appendCurlRequest (returns true).
				/// A false return never completes, so increment m_finishedReqCnt to keep waitToFinish() balanced and then throw.
				if(!appendCurlRequest(curlRequestInfo)) {
					m_finishedReqCnt++;
					m_cond4Out.notify_all();
					throw std::runtime_error(
						"[ReIDService] Failed to append CURL request for targetID: " +
						std::to_string(curlRequestInfo.targetID)
					);
				}
			}
			else
			{
				inputLock.unlock();
			}

			CURLMcode res = curl_multi_perform(m_curlMultiHandle, &curlReqsInFlight);
			if (res != CURLM_OK)
			{
				cout << "[ReIDService] Failed to perform CURL request: " << curl_multi_strerror(res) << endl;
				throw std::runtime_error(
					std::string("[ReIDService] Failed to perform CURL request: ") +
					curl_multi_strerror(res)
				);
			}
			while ((msg = curl_multi_info_read(m_curlMultiHandle, &msgsLeft)))
			{
				processCompletedCurlRequest(msg);
				m_finishedReqCnt++;
				/// Send signal to ReIDService::waitToFinish().
				m_cond4Out.notify_all();
			}
			/// Wait for activity with a timeout if there are any requests in flight & input queue is empty.
			inputLock.lock();
			bool waitForActivity = curlReqsInFlight && m_inputQueue.empty();
			inputLock.unlock();
			if (waitForActivity)
			{
				curl_multi_poll(m_curlMultiHandle, NULL, 0, 30, NULL);
			}
		}
		catch (const std::exception &e)
		{
			cout << "[ReIDService] WORKER EXCEPTION: " << e.what()
				 << " (worker continuing)" << endl;
		}
		catch (...)
		{
			cout << "[ReIDService] WORKER EXCEPTION: unknown (worker continuing)" << endl;
		}
	}

	/// Per libcurl, every easy handle still attached to the multi handle must be removed and curl_easy_cleanup'd (along with its slist headers) before curl_multi_cleanup; otherwise handles/headers/connection state are leaked across ReIDService destroy/recreate cycles.
	for (auto &kv : m_curlRequestInfoMap)
	{
		CURL *handle = kv.first;
		CurlRequestTargetInfo &info = kv.second;
		curl_multi_remove_handle(m_curlMultiHandle, handle);
		if (info.headers)
		{
			curl_slist_free_all(info.headers);
			info.headers = nullptr;
		}
		curl_easy_cleanup(handle);
	}
	m_curlRequestInfoMap.clear();

	/// Cleanup curl resources in THIS thread.
	if (m_curlMultiHandle)
	{
		curl_multi_cleanup(m_curlMultiHandle);
		m_curlMultiHandle = nullptr;
	}
	curl_global_cleanup();
}

/**
 * @brief Serialize a ReID feature vector as JSON.
 * @param[in] reidFeatures ReID feature vector.
 * @param[in] reidFeatureSize Number of values in each ReID feature.
 */
std::string ReIDService::convertReidFeaturesToJson(const float* reidFeatures, int reidFeatureSize)
{
	nlohmann::json jsonObj;

	/// Create JSON array directly from the pointer (if supported).
	jsonObj["vector"] = nlohmann::json::array_t(reidFeatures, reidFeatures + reidFeatureSize);
	jsonObj["top_k"] = m_params.m_topK;
	jsonObj["max_group_count"] = m_params.m_maxGroupCount;
	return jsonObj.dump();
}

/**
 * @brief Append received HTTP response bytes to a string.
 * @param[in] contents Received response bytes.
 * @param[in] size Size of each received element in bytes.
 * @param[in] nmemb Number of received elements.
 * @param[out] userp Response string to append to.
 */
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp)
{
	userp->append((char*)contents, size * nmemb);
	return size * nmemb;
}

#if 0
/**
 * @brief Send one synchronous HTTP POST request.
 * @param[in] jsonPayload JSON request payload.
 */
std::string ReIDService::send_post_request(const std::string& jsonPayload)
{
    /// Initialize curl.
    CURL* curl = curl_easy_init();
    if (!curl) {
        cout << "[NvTrajectoryManager] Failed to initialize CURL" << endl;
        return "";
    }

    /// Set up headers.
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    /// String to store the response.
    std::string response = "";

    /// Configure curl options.
    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/search");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    /// Set up response callback.
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        cout << "[NvTrajectoryManager] CURL request failed: " << curl_easy_strerror(res) << endl;
    } else {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        cout << "[NvTrajectoryManager] CURL request successful, response code: " << response_code << endl;

        /// Print the response.
        cout << "[NvTrajectoryManager] Response: " << response << endl;
        #if 0
            /// Print response with lower 32 bits of object_id.
            try {
                nlohmann::json jsonObj = nlohmann::json::parse(response);
                nlohmann::json modifiedJson = jsonObj;

                /// Modify each object to show lower 32 bits of object_id.
                for(auto& obj : modifiedJson) {
                    if(obj.contains("object_id")) {
                        TargetId originalId = obj["object_id"];
                        uint32_t lower32Bits = originalId & 0xFFFFFFFF;
                        obj["object_id"] = lower32Bits;
                    }
                }

                cout << "[NvTrajectoryManager] Response (lower32): " << modifiedJson.dump() << endl;
            } catch (const nlohmann::json::parse_error& e) {
                cout << "[NvTrajectoryManager] Failed to parse response for lower32 display: " << e.what() << endl;
            }
        #endif
    }

    /// Cleanup.
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
	return response;
}
#endif

/**
 * @brief Add a request to the asynchronous CURL multi-handle.
 * @param[in] curlRequestInfoInput Request state to append to the active CURL set.
 * @return False when the CURL handle cannot be created; true otherwise, including when the request is dropped.
 */
bool ReIDService::appendCurlRequest(const CurlRequestTargetInfo &curlRequestInfoInput)
{
	if (m_curlRequestInfoMap.size() >= (size_t)m_params.m_maxInFlightRequests) {
		if (m_params.m_debugVerboseLevel > 1) {
			cout << "[ReIDService] WARNING: Too many in-flight requests (" << m_curlRequestInfoMap.size() << "), dropping request" << endl;
		}
		m_finishedReqCnt++;
		m_cond4Out.notify_all();
		return true;
	}

	/// Initialize curl.
	CURL* curlHandle = curl_easy_init();
	if (!curlHandle) {
		cout << "[ReIDService] Failed to initialize CURL" << endl;
		return false;
	}

	m_curlRequestInfoMap[curlHandle] = curlRequestInfoInput;
	CurlRequestTargetInfo &curlRequestInfo = m_curlRequestInfoMap[curlHandle];
	if (m_params.m_debugVerboseLevel > 3) {
		cout << "[ReIDService] appendCurlRequest for targetID: " << curlRequestInfo.targetID << endl;
	}

	/// Set up headers.
	/// Libcurl does not own the header list, so retain it and free it with the request.
	curlRequestInfo.headers = curl_slist_append(nullptr, "Content-Type: application/json");

	/// Configure curl options using the copy stored in the map.
	curl_easy_setopt(curlHandle, CURLOPT_URL, m_curlRequestUrl.c_str());
	curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDS, curlRequestInfo.jsonPayload.c_str());
	curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, curlRequestInfo.headers);
	curl_easy_setopt(curlHandle, CURLOPT_TIMEOUT, m_params.m_curlRequestTimeout);
	curl_easy_setopt(curlHandle, CURLOPT_CONNECTTIMEOUT, m_params.m_curlConnectTimeout);
	/// Enables TCP keepalive, which helps the curl_multi handle reuse connections across different easy handles.
	/// This reduces overhead and avoids connection limit issues.
	curl_easy_setopt(curlHandle, CURLOPT_TCP_KEEPALIVE, 1L);

	/// Set up response callback.
	curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, WriteCallback);
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &(curlRequestInfo.response));

	curl_multi_add_handle(m_curlMultiHandle, curlHandle);
	return true;
}

/**
 * @brief Parse a remote identity result from a service response.
 * @param[in] response Service response text.
 */
remoteTargetInfo ReIDService::parseResponseForTargetID(const std::string& response)
{
	nlohmann::json jsonObj;
	remoteTargetInfo remoteTarget;
	remoteTarget.remoteTargetID = 0;
	remoteTarget.validRemoteTargetID = false;
	try {
		jsonObj = nlohmann::json::parse(response);
		/// Parse the response and return the object_id with the highest confidence.
		float maxConfidence = 0;
		std::map<TargetId, float> expConfidenceMap;
		std::map<TargetId, reidServiceAssocInfo> reidServiceAssocResponseMap;
		for(const auto& obj : jsonObj) {
			try {
				if(obj["group_count"] == -1) {
					/// Skip non-group search outputs because they are not used.
					continue;
				}
				reidServiceAssocInfo reidServiceAssocInfo;
				TargetId objectID = std::stoull(obj["object_id"].get<std::string>());
				reidServiceAssocInfo.targetID = objectID;
				reidServiceAssocInfo.groupCount = obj["group_count"];
				reidServiceAssocInfo.confidence = obj["confidence"];
				reidServiceAssocInfo.expConfidence = obj["exp_confidence"];
				reidServiceAssocInfo.distance = obj["distance"];
				reidServiceAssocInfo.distanceToCentroid = obj["distance_to_centroid"];
				reidServiceAssocInfo.clusterVariance = obj["cluster_variance"];
				reidServiceAssocResponseMap[objectID] = reidServiceAssocInfo;
				if (m_params.m_debugVerboseLevel > 3) {
					cout << std::fixed << std::setprecision(3);
					cout << "(" << reidServiceAssocInfo.targetID << "," << reidServiceAssocInfo.groupCount << ","
					<< reidServiceAssocInfo.confidence << ","  << reidServiceAssocInfo.expConfidence << ","
					<< reidServiceAssocInfo.distance << ","  << reidServiceAssocInfo.distanceToCentroid << ","
					<< std::defaultfloat << reidServiceAssocInfo.clusterVariance << ")";
					cout << endl;
				}
				if(obj["group_count"] != (int)m_params.m_maxGroupCount) {
					/// Ignore if group count is not equal to m_maxGroupCount.
					continue;
				}
				expConfidenceMap[objectID] = obj["exp_confidence"];
			} catch (const std::invalid_argument& e) {
				/// Handles invalid string format (e.g., "abc").
				std::cerr << "Invalid object_id format: " << e.what() << std::endl;
			} catch (const std::out_of_range& e) {
				/// Handles numbers too large for unsigned long long.
				std::cerr << "object_id out of range: " << e.what() << std::endl;
			}
		}

		for(const auto& entry : expConfidenceMap) {
			TargetId id = entry.first;
			if (m_params.m_debugVerboseLevel > 3) {
				cout << "Object ID: " << id << ", Average confidence: " << reidServiceAssocResponseMap[id].confidence	 << ", Average exp confidence: " << reidServiceAssocResponseMap[id].expConfidence << ", Average distance: " << reidServiceAssocResponseMap[id].distance << ", ";
				cout << "Distance to centroid: " << reidServiceAssocResponseMap[id].distanceToCentroid << ", cluster variance "  << reidServiceAssocResponseMap[id].clusterVariance << ", count: " << reidServiceAssocResponseMap[id].groupCount << endl;
			}
		}
		/// Sort the expConfidenceMap by expConfidence in descending order.
		std::vector<std::pair<TargetId, float>> sortedExpConfidenceMap(
			expConfidenceMap.begin(), expConfidenceMap.end()
		);
		std::sort(sortedExpConfidenceMap.begin(), sortedExpConfidenceMap.end(), [](const auto &a, const auto &b) {
			return a.second > b.second;
		});
		/// Check the topmost confidence value.
		if(sortedExpConfidenceMap.size() > 0) {
			float topmostConfidence = sortedExpConfidenceMap[0].second;
			TargetId topmostTargetID = sortedExpConfidenceMap[0].first;
			if((topmostConfidence >= m_params.m_minTopClusterConfidence) &&
			(reidServiceAssocResponseMap[topmostTargetID].clusterVariance <= m_params.m_maxClusterVariance)) {
				if(reidServiceAssocResponseMap[topmostTargetID].groupCount != m_params.m_maxGroupCount) {
					/// Ignore if group count is not equal to maxGroupCount.
					throw std::runtime_error("0 : Group count is not equal to maxGroupCount");
				}
				maxConfidence = topmostConfidence;
				remoteTarget.remoteTargetID = sortedExpConfidenceMap[0].first;
				remoteTarget.validRemoteTargetID = true;
				/// Check whether the second highest confidence value is also very high.
				if(sortedExpConfidenceMap.size() > 1) {
					float secondHighestConfidence = sortedExpConfidenceMap[1].second;
					TargetId secondHighestTargetID = sortedExpConfidenceMap[1].first;
					if((secondHighestConfidence >= m_params.m_minSecondClusterConfidence)) {
						if(reidServiceAssocResponseMap[secondHighestTargetID].groupCount != m_params.m_maxGroupCount) {
							/// Report error if group count is not equal to maxGroupCount.
							throw std::runtime_error("1 :Group count is not equal to maxGroupCount");
						}
						if (m_params.m_debugVerboseLevel > 2) {
							cout << "Two clusters with very high confidence values: " << topmostTargetID << " and " << secondHighestTargetID << endl;
							cout << "Can be one of the two object IDs" << endl;
						}

						remoteTarget.validRemoteTargetID = false;
					}
				}
			}
		}
		if(remoteTarget.validRemoteTargetID) {
			if (m_params.m_debugVerboseLevel > 3) {
				cout << "targetID: " << remoteTarget.remoteTargetID << "\t" << "lower32: " << (remoteTarget.remoteTargetID & 0xFFFFFFFF) << "\t" << "maxConfidence: " << maxConfidence << endl;
			}
			remoteTarget.reidServiceAssocResponseMap = reidServiceAssocResponseMap;
		}
		return remoteTarget;
	} catch (const nlohmann::json::parse_error& e) {
		cout << "[NvTrajectoryManager] JSON parse error: " << e.what() << endl;
		return remoteTarget;
	}
	return remoteTarget;
}

#if 0
/** @brief Submit queued targets to the remote ReID service. */
void ReIDService::query()
{
    const std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>> targets = m_targetManager->getTrackedObjectsInUseList();
	for(auto streamIt = targets.begin(); streamIt != targets.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamId = streamIt->first;
		const std::shared_ptr<NvTrackedObjectList> targetList = streamIt->second;
		for(auto targetIt = targetList->begin(); targetIt != targetList->end(); ++targetIt)
		{
			const std::shared_ptr<NvTrackedObject> target = targetIt->second;
			bool tryReidServiceReAssoc = false;
			int reidFeatureSize = 0;
			const float *reidFeatures = nullptr;
			target->getReidFeaturesDet_h(&reidFeatures, reidFeatureSize);
			/// Set to true if needToReAssocFromReidService is true & reid embeddings were extracted.
			tryReidServiceReAssoc = (target->needToReAssocFromReidService() && reidFeatures != nullptr);
			if(tryReidServiceReAssoc)
			{
				cout << "targetID: [" << target->getTargetID() << "] tryReidServiceReAssoc: true" << endl;
				remoteTargetInfo remoteTarget;
				remoteTarget.remoteTargetID = 0;
				remoteTarget.validRemoteTargetID = false;
				/// Convert reidFeatures vector to JSON string.
				std::string json = convertReidFeaturesToJson(reidFeatures, reidFeatureSize);
				cout << "json: " << json << endl;

				std::string response = send_post_request(json);
				if (response != "")
				{
					remoteTarget = parseResponseForTargetID(response);
					if(remoteTarget.validRemoteTargetID)
					{
						cout << "remoteTargetID: " << remoteTarget.remoteTargetID << "\t lower32: " << (remoteTarget.remoteTargetID & 0xFFFFFFFF) << "\t" << endl;
						/// Check if the remoteTargetID is already used by any of the active trajectories.
						/// This might happen when same target ID is assigned to multiple targets by tracker due to tracking inaccuracies.
						for(auto jt = targetList->begin(); jt != targetList->end(); ++jt)
						{
							if(jt->second->getTargetID() == remoteTarget.remoteTargetID)
							{
								cout << "Remote Target ID " << remoteTarget.remoteTargetID << " is already used by tracker Index " << jt->first << endl;
								cout << "Shadow Tracking Age: " << jt->second->getShadowTrackingAge() << endl;
								remoteTarget.validRemoteTargetID = false;
								break;
							}
						}
					}
				}
				if(remoteTarget.validRemoteTargetID){
					cout << "frameNum[" << target->getTargetInfo().frameNum << "] Replacing TargetID using remoteTargetID from reid service: from Target ID : [" << target->getTargetID() << "] to [" << remoteTarget.remoteTargetID <<  "]" << endl;
					target->setTargetID(remoteTarget.remoteTargetID);
					target->resetNeedToReAssocFromReidService();
				}
			}
		}
	}
}
#endif

/**
 * @brief Collect the result of a completed CURL request.
 * @param[in] msg Completed CURL message.
 */
void ReIDService::processCompletedCurlRequest(CURLMsg *msg)
{
	if (msg->msg == CURLMSG_DONE) {
		CURL *handle = msg->easy_handle;
		CURLcode return_code = msg->data.result;
		if(return_code!=CURLE_OK) {
			if (m_params.m_debugVerboseLevel > 1) {
				cout << "[ReIDService] WARNING: CURL request failed: " << curl_easy_strerror(return_code) << " (error code: " << return_code << ")" << endl;
			}
			auto failIt = m_curlRequestInfoMap.find(handle);
			if(failIt != m_curlRequestInfoMap.end()) {
				CurlRequestTargetInfo &info = failIt->second;
				if (m_params.m_debugVerboseLevel > 1) {
					cout << "[ReIDService] WARNING: Failed request details - targetID: " << info.targetID << ", streamId: " << info.streamId << endl;
				}

				/// Free it before destroying the easy handle to avoid a leak per request (see appendCurlRequest).
				if (info.headers) {
					curl_slist_free_all(info.headers);
					info.headers = nullptr;
				}
			}
			curl_multi_remove_handle(m_curlMultiHandle, handle);
			curl_easy_cleanup(handle);
			m_curlRequestInfoMap.erase(handle);
			return;
		}
		if(m_curlRequestInfoMap.find(handle) == m_curlRequestInfoMap.end()) {
			throw std::runtime_error("[ReIDService] CURL request handle not found in m_curlRequestInfoMap");
			return;
		}
		CurlRequestTargetInfo &curlRequestInfo = m_curlRequestInfoMap[handle];
		remoteTargetInfo remoteTarget = {};
		if (curlRequestInfo.response != "")
		{
			if (m_params.m_debugVerboseLevel > 3) {
				cout << "[ReIDService] parseResponse for targetID: " << curlRequestInfo.targetID << endl;
			}
			remoteTarget = parseResponseForTargetID(curlRequestInfo.response);
		}
		if(remoteTarget.validRemoteTargetID){
			remoteTarget.originalTargetID = curlRequestInfo.targetID;
			remoteTarget.streamId = curlRequestInfo.streamId;
			remoteTarget.targetIndex = curlRequestInfo.targetIndex;
			if (m_params.m_debugVerboseLevel > 2) {
				cout << "Adding output queue entry for replacing TargetID using remoteTargetID from reid service: from Target ID : [" << remoteTarget.originalTargetID << "] to [" << remoteTarget.remoteTargetID <<  "]" << endl;
			}
			std::unique_lock<std::mutex> outputLock(m_outputLock);
			m_outputQueue.push(std::move(remoteTarget));
			outputLock.unlock();
		}

		curl_multi_remove_handle(m_curlMultiHandle, handle);
		/// Same reason as above: free the per-request header slist before destroying the easy handle.
		if (curlRequestInfo.headers) {
			curl_slist_free_all(curlRequestInfo.headers);
			curlRequestInfo.headers = nullptr;
		}
		curl_easy_cleanup(handle);
		m_curlRequestInfoMap.erase(handle);
	}
	else {
		fprintf(stderr, "error: after curl_multi_info_read(), CURLMsg=%d\n", msg->msg);
		throw std::runtime_error("[ReIDService] Failed to process completed CURL request");
	}
}

/**
 * @brief Submit queued targets to the remote ReID service.
 * @param[in] inputMeta Input frames keyed by stream ID.
 */
void ReIDService::query(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta)
{
	/// Before sending the queries to reid service, check if there are any targets that were already associated using reid service but the target ID changed after that.
	/// This might happen due to late peer association and target ID correction in MultiViewAssociator.
	/// If so, set the flag to indicate this target needs to be re-associated from reid service again.
	for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamId = streamIt->first;
		const NvMOTFrame *srcMotFrame = streamIt->second;
		const uint32_t frameNum = srcMotFrame->frameNum;
		const NvTrackedObjectList &targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);
		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
			const std::shared_ptr<NvTrackedObject> target = targetIt->second;
			if(!target->needToReAssocFromReidService() && target->getTargetIDFromReidService() != target->getTargetID()) {
				if (m_params.m_debugVerboseLevel > 3) {
					cout << "[ReIDService] Target ID has changed after reid service association. targetID: [" << target->getTargetID() << "] targetIDFromReidService: [" << target->getTargetIDFromReidService() << "] " << endl;
				}
				target->setNeedToReAssocFromReidService(true);
			}
		}
	}

	/// Query the ReID service only for streams that contributed a source frame in this batch.
	for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamId = streamIt->first;
		const NvMOTFrame *srcMotFrame = streamIt->second;
		const uint32_t frameNum = srcMotFrame->frameNum;
		const NvTrackedObjectList &targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);
		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
			const std::shared_ptr<NvTrackedObject> target = targetIt->second;
			bool tryReidServiceReAssoc = false;
			int reidFeatureSize = 0;
			const float *reidFeatures = nullptr;
			target->getReidFeaturesDet_h(&reidFeatures, reidFeatureSize);
			/// Use only the targets with class IDs in m_operateOnClassIds.
			if(!m_params.m_operateOnClassIds.empty() && std::find(m_params.m_operateOnClassIds.begin(), m_params.m_operateOnClassIds.end(), target->getTargetInfo().classId) == m_params.m_operateOnClassIds.end()) {
				if (m_params.m_debugVerboseLevel > 3) {
					cout << "targetID: [" << target->getTargetID() << "] classId: [" << target->getTargetInfo().classId << "] is not in the operateOnClassIds list. Skipping reid service association" << endl;
				}
				continue;
			}
			/// Set to true if needToReAssocFromReidService is true & reid embeddings were extracted.
			tryReidServiceReAssoc = (target->needToReAssocFromReidService()
					&& reidFeatures != nullptr
					&& reidFeatureSize > 0
					&& target->getTargetInfo().visibility >= m_params.m_minVisibility4ReidServiceQuery
					&& target->getTargetInfo().imgBorderCroppingVisibility >= m_params.m_minVisibility4ReidServiceQuery);
			if(tryReidServiceReAssoc)
			{
				if (m_params.m_debugVerboseLevel > 2) {
					cout << "targetID: [" << target->getTargetID() << "] tryReidServiceReAssoc: true" << endl;
				}
				CurlRequestTargetInfo curlRequestInfo = {};
				curlRequestInfo.streamId = target->getVideoStreamId();
				curlRequestInfo.targetIndex = target->getTargetInd();
				curlRequestInfo.targetID = target->getTargetID();
				curlRequestInfo.response = "";
				/// Convert reidFeatures vector to JSON string.
				std::string json = convertReidFeaturesToJson(reidFeatures, reidFeatureSize);
				if (m_params.m_debugVerboseLevel > 4) {
					cout << "[ReIDService] json: " << json << endl;
				}
				curlRequestInfo.jsonPayload = json;
				/// Cap the input queue so a slow/unreachable ReID service cannot grow it without bound and OOM the process.
				/// Use m_maxInFlightRequests as the bound: the worker drains at that rate anyway, so allowing queue >> in-flight only buys latency, not throughput.
				/// Drop on overflow without touching the counters (no submit => no finish needed).
				std::unique_lock<std::mutex> inputLock(m_inputLock);
				if(m_inputQueue.size() >= (size_t)m_params.m_maxInFlightRequests)
				{
					inputLock.unlock();
					if (m_params.m_debugVerboseLevel > 1) {
						cerr << "[ReIDService] input queue full ("
							 << m_params.m_maxInFlightRequests
							 << "), dropping request for targetID: ["
							 << target->getTargetID() << "]" << endl;
					}
					continue;
				}
				/// Count the submitted request atomically.
				m_submittedReqCnt++;
				/// Push the request into the queue and wake the worker thread.
				m_inputQueue.push(std::move(curlRequestInfo));
				inputLock.unlock();
				m_cond4In.notify_one();
			}
		}
	}
}

/** @brief Wait until all submitted service requests have completed. */
void ReIDService::waitToFinish()
{
	if (m_submittedReqCnt > 0)
	{
		/// Wait for output.
		while (m_persistThreadRunning)
		{
			std::unique_lock<std::mutex> outputLock(m_outputLock);

			/// The wait predicate rechecks the unlocked completion counter to avoid missing a notification.
			/// Re-checking under the lock on every wakeup (and once before sleeping) closes that race.
			m_cond4Out.wait(outputLock, [this]{
				return !m_persistThreadRunning || m_finishedReqCnt >= m_submittedReqCnt;
			});

			/// Check for deinit signal or spurious wakeup or output size is not equal to input!!
			if(!m_persistThreadRunning)
			{
				outputLock.unlock();
				break;
			}
			m_finishedReqCnt = 0;
			outputLock.unlock();
			break;
		}
		m_submittedReqCnt = 0;
		m_finishedReqCnt = 0;
	}
	else
	{
		m_finishedReqCnt = 0;
	}
}

/**
 * @brief Apply a remote identity to a compatible local target.
 * @param[in] targetToBeCorrected Target whose identity may be corrected.
 * @param[in] existingTarget Existing target associated with the remote identity.
 * @param[in] associatedTargetID Target identifier returned by the service.
 * @return True when the correction is accepted and the existing target's ID is dropped; false otherwise.
 */
bool ReIDService::tryIDCorrection(
	const std::shared_ptr<NvTrackedObject> targetToBeCorrected,
	const std::shared_ptr<NvTrackedObject> existingTarget, TargetId associatedTargetID
)
{
	/// Error checking.
	if((targetToBeCorrected->getTargetID() == associatedTargetID) || (existingTarget->getTargetID() != associatedTargetID)) {
		throw std::runtime_error(
			"[ReIDService] Something is wrong. targetToBeCorrected->getTargetID(): [" + std::to_string(targetToBeCorrected->getTargetID()) + "] associatedTargetID: [" + std::to_string(associatedTargetID) + "] existingTarget->getTargetID(): [" + std::to_string(existingTarget->getTargetID()) + "]"
		);
	}
	if(targetToBeCorrected->getVideoStreamId() != existingTarget->getVideoStreamId()) {
		throw std::runtime_error(
			"[ReIDService] Something is wrong. targetToBeCorrected->getVideoStreamId(): [" + std::to_string(targetToBeCorrected->getVideoStreamId()) + "] existingTarget->getVideoStreamId(): [" + std::to_string(existingTarget->getVideoStreamId()) + "]"
		);
	}
	NvMOTStreamId streamId = targetToBeCorrected->getVideoStreamId();
	std::map<TargetId, std::pair<TargetId, int>> &idCorrectionMap = m_idCorrectionMap[streamId];
	/// If the target ID to be corrected is not in the map, add it.
	if(idCorrectionMap.find(targetToBeCorrected->getTargetID()) == idCorrectionMap.end()) {
		idCorrectionMap[targetToBeCorrected->getTargetID()] = std::make_pair(associatedTargetID, 1);
	}
	else {
		/// If the target ID to be corrected is in the map, first check if the entry in the map is same as associatedTargetID.
		if(idCorrectionMap[targetToBeCorrected->getTargetID()].first == associatedTargetID) {
			idCorrectionMap[targetToBeCorrected->getTargetID()].second++;
		}
		else {
			/// Replace the entry in the map with the new associated target ID and reset the count.
			idCorrectionMap[targetToBeCorrected->getTargetID()] = std::make_pair(associatedTargetID, 1);
		}
	}
	/// If the count is greater than the minimum association count for ID correction, return true.
	if(idCorrectionMap[targetToBeCorrected->getTargetID()].second >= m_params.m_minAssociationCount4IDCorrection) {
		{
			/// First check if the bounding boxes overlap by more than threshold.
			const float iou = CV_FUNCS::computeIOU(
				targetToBeCorrected->getTargetInfo().tBbox, existingTarget->getTargetInfo().tBbox
			);
			if(iou > m_params.m_maxIou4IDCorrection)
			{
				cout << "[ReIDService] Bounding boxes overlap by more than threshold for targetID: [" << targetToBeCorrected->getTargetID() << "] associatedTargetID: [" << associatedTargetID << "]" << endl;
				cout << "IOU: [" << iou << "]" << endl;
				cout << "Ignoring ID correction" << endl;
				return false;
			}
		}

		cout << "[ReIDService] ID correction successful for targetID: [" << targetToBeCorrected->getTargetID() << "] associatedTargetID: [" << associatedTargetID << "] count: [" << idCorrectionMap[targetToBeCorrected->getTargetID()].second << "]" << endl;
		/// Remove the entry from the map.
		idCorrectionMap.erase(targetToBeCorrected->getTargetID());
		/// Drop the ID from existingTarget i.e. set the flag to indicate this target needs a new ID assigned.
		existingTarget->setNeedTargetIDAssigned();
		if (m_params.m_debugVerboseLevel > 2) {
			cout << "Stream ID: [" << (existingTarget->getVideoStreamId() >> 32) << "] frameNum[" << existingTarget->getTargetInfo().frameNum << "] Dropping TargetID using ID correction: from Target ID : [" << existingTarget->getTargetID() << "]" << endl;
		}
		return true;
	}
	else {
		return false;
	}
}

/** @brief Apply completed remote associations to local targets. */
void ReIDService::updateTargetIDs()
{
	/// For debugging purposes, wait for the worker thread to finish its work.
	if(m_params.m_enableBlocking)
	{
		waitToFinish();
	}
	std::unique_lock<std::mutex> outputLock(m_outputLock);
	while(!m_outputQueue.empty())
	{
		remoteTargetInfo remoteTarget = m_outputQueue.front();
		m_outputQueue.pop();
		outputLock.unlock();
		const std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>> targets = m_targetManager->getTrackedObjectsInUseList();
		if(targets.find(remoteTarget.streamId) == targets.end()) {
			throw std::runtime_error("[ReIDService] Stream ID not found in targets");
			return;
		}
		const std::shared_ptr<NvTrackedObjectList> targetList = targets.at(remoteTarget.streamId);
		if(targetList->find(remoteTarget.targetIndex) == targetList->end()) {
			if (m_params.m_debugVerboseLevel > 2) {
				cout << "[ReIDService] Target [" << remoteTarget.originalTargetID << "] not found in active targets list" << endl;
				cout << "Must have got terminated by the tracker. Hence, ignoring the response" << endl;
			}
			goto next_target;
		}
		/// Open a new scope for the target.
		{
			const std::shared_ptr<NvTrackedObject> target = targetList->at(remoteTarget.targetIndex);
			if(target->getTargetID() != remoteTarget.originalTargetID) {
				if (m_params.m_debugVerboseLevel > 2) {
					cout << "[ReIDService] Target ID mismatch" << endl;
					cout << "remoteTarget.originalTargetID: [" << remoteTarget.originalTargetID << "] \t ";
					cout << "target->getTargetID(): [" << target->getTargetID() << "]" << endl;
					cout << "Hence, ignoring the response" << endl;
				}
			}
			else {
				if (m_params.m_debugVerboseLevel > 3) {
					cout << "remoteTargetID: " << remoteTarget.remoteTargetID << " for targetID: " << remoteTarget.originalTargetID << endl;
				}
				/// Check if the remoteTargetID is already used by any of the active trajectories.
				/// This might happen when same target ID is assigned to multiple targets by tracker due to tracking inaccuracies.
				for(auto jt = targetList->begin(); jt != targetList->end(); ++jt)
				{
					auto scannedTarget = jt->second;
					if(scannedTarget->getTargetID() == remoteTarget.remoteTargetID)
					{
						/// If it's a self match, ignore the response.
						if(scannedTarget->getTargetInd() == target->getTargetInd()) {
							if (m_params.m_debugVerboseLevel > 3) {
								cout << "It's a self match. Hence, ignoring the response" << endl;
							}
							remoteTarget.validRemoteTargetID = false;
							break;
						}
						if (m_params.m_debugVerboseLevel > 2) {
							cout << "Remote Target ID " << remoteTarget.remoteTargetID << " is already used by tracker Index " << jt->first << endl;
							cout << "Shadow Tracking Age: " << scannedTarget->getShadowTrackingAge() << endl;
						}
						/// Check if the tracker state is not ACTIVE and it's not the same target.
						/// Delete the scanned target and reuse its ID for the current target.
						if(scannedTarget->getTrackerState() != TRACKER_STATE::ACTIVE && scannedTarget->getTargetInd() != target->getTargetInd()) {
							if (m_params.m_debugVerboseLevel > 2) {
								cout << "Tracker state is not ACTIVE, terminating the target and using the ID for the current target" << endl;
							}
							m_targetManager->terminateTrackerPublic(scannedTarget, DUPLICATE_TRACKER);
						}
						else if(m_params.m_enableIDCorrection) {
							if(!tryIDCorrection(target, scannedTarget, remoteTarget.remoteTargetID))
								remoteTarget.validRemoteTargetID = false;
						}
						else
						{
							/// There is an Active target with the same ID.
							remoteTarget.validRemoteTargetID = false;
						}
						break;
					}
				}
				if(remoteTarget.validRemoteTargetID)
				{
					target->setTargetID(remoteTarget.remoteTargetID);
					target->setTargetIDFromReidService(remoteTarget.remoteTargetID);
					target->setReidServiceAssocResponse(remoteTarget.reidServiceAssocResponseMap);
					/// Set flags to indicate re-association from reid service is done.
					/// So this target will not be re-associated from reid service again.
					target->resetNeedToReAssocFromReidService();
					/// Following two settings are used to set m_adoptedIdMap in MultiViewAssociator.
					target->setAdoptedReidServiceID(true);
					target->setTargetIDBeforeAdoption(remoteTarget.originalTargetID);
					if (m_params.m_debugVerboseLevel > 1) {
						cout << "Stream ID: [" << (remoteTarget.streamId >> 32) << "] frameNum[" << target->getTargetInfo().frameNum << "] Replacing TargetID using remoteTargetID from reid service: from Target ID : [" << remoteTarget.originalTargetID << "] to [" << remoteTarget.remoteTargetID <<  "]" << endl;
					}
				}
			}
		}
next_target:
		outputLock.lock();
	}
	outputLock.unlock();
}

/** @brief Construct a ReIDServiceConfigParams instance. */
ReIDServiceConfigParams::ReIDServiceConfigParams()
{
	m_reidServiceType = REID_SERVICE_TYPES::REID_SERVICE;
	m_debugVerboseLevel = 0;
	m_enableBlocking = false;
	m_minTopClusterConfidence = 0.58;
	m_minSecondClusterConfidence = 0.56;
	m_maxClusterVariance = 0.0001;
	m_topK = 8;
	m_maxGroupCount = 8;
	m_minVisibility4ReidServiceQuery = 0.0f;
	m_serviceAddress = "localhost";
	m_servicePort = 8000;
	/// Empty vector means operate on all classes.
	m_operateOnClassIds = {};
	m_enableIDCorrection = false;
	m_minAssociationCount4IDCorrection = 2;
	m_maxIou4IDCorrection = 0.2f;
	m_maxInFlightRequests = 100;
	m_curlConnectTimeout = 1;
	m_curlRequestTimeout = 3;
}

/**
 * @brief Construct a ReIDServiceConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 */
ReIDServiceConfigParams::ReIDServiceConfigParams(const std::string &configFilePath, const std::string moduleName)
{
	/// Set the default values.
	*this = ReIDServiceConfigParams();

	YAML::Node configRoot = getConfigRoot(configFilePath);
	/// Read the parameters of the named module.
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct a ReIDServiceConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
ReIDServiceConfigParams::ReIDServiceConfigParams(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void ReIDServiceConfigParams::read(const YAML::Node &fn)
{
	if(!fn.size())
	{
		return;
	}

	for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
	{
		std::string paramKey = fit->first.as<std::string>();
		if (paramKey == "debugVerboseLevel")
		{
			m_debugVerboseLevel = (int)fit->second.as<float>();
		}
		else if (paramKey == "reidServiceType")
		{
			m_reidServiceType = (REID_SERVICE_TYPES)fit->second.as<int>();
		}
		else if (paramKey == "enableBlocking")
		{
			m_enableBlocking = (bool)fit->second.as<float>();
		}
		else if (paramKey == "minTopClusterConfidence")
		{
			m_minTopClusterConfidence = fit->second.as<float>();
		}
		else if (paramKey == "minSecondClusterConfidence")
		{
			m_minSecondClusterConfidence = fit->second.as<float>();
		}
		else if (paramKey == "maxClusterVariance")
		{
			m_maxClusterVariance = fit->second.as<float>();
		}
		else if (paramKey == "topK")
		{
			m_topK = (int)fit->second.as<float>();
		}
		else if (paramKey == "maxGroupCount")
		{
			m_maxGroupCount = (int)fit->second.as<float>();
		}
		else if (paramKey == "minVisibility4ReidServiceQuery")
		{
			m_minVisibility4ReidServiceQuery = fit->second.as<float>();
		}
		else if (paramKey == "serviceAddress")
		{
			m_serviceAddress = fit->second.as<std::string>();
		}
		else if (paramKey == "servicePort")
		{
			m_servicePort = (int)fit->second.as<float>();
		}
		else if (paramKey == "operateOnClassIds")
		{
			m_operateOnClassIds.clear();
			for (std::size_t i=0; i<fit->second.size(); i++)
			{
				m_operateOnClassIds.push_back((int)fit->second[i].as<float>());
			}
		}
		else if (paramKey == "enableIDCorrection")
		{
			m_enableIDCorrection = (bool)fit->second.as<float>();
		}
		else if (paramKey == "minAssociationCount4IDCorrection")
		{
			m_minAssociationCount4IDCorrection = (int)fit->second.as<float>();
		}
		else if (paramKey == "maxIou4IDCorrection")
		{
			m_maxIou4IDCorrection = fit->second.as<float>();
		}
		else if (paramKey == "maxInFlightRequests")
		{
			m_maxInFlightRequests = (int)fit->second.as<float>();
		}
		else if (paramKey == "curlConnectTimeout")
		{
			m_curlConnectTimeout = (int)fit->second.as<float>();
		}
		else if (paramKey == "curlRequestTimeout")
		{
			m_curlRequestTimeout = (int)fit->second.as<float>();
		}
		else
		{
			cout << "!! [WARNING][ReIDServiceConfigParams] Unknown param found: " << paramKey << endl;
		}
	}
}
