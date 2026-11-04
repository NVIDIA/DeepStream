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

#include "modules/NvMultiObjectTracker/NvMultiObjectTracker.hpp"

/**
 * @brief Split a delimited string into serialized payloads.
 * @param[in] s Delimited payload string.
 * @param[in] delim Payload delimiter.
 * @return The extracted serialized payloads.
 */
static std::vector<std::string> splitPayloadListStr(const std::string &s, char delim = '|')
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while(getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

/**
 * @brief Join serialized payloads using a delimiter.
 * @param[in] payloadStrList Serialized payloads to join.
 * @param[in] delim Payload delimiter.
 * @return The joined payload string.
 */
static std::string concatPayloadListStr(const std::vector<std::string>& payloadStrList, char delim = '|')
{
    if (payloadStrList.empty()) {
        printf("[MqttCommunicator] Warning: payloadStrList is empty\n");
        return "";
    }

    std::string outString = payloadStrList[0];
    for (size_t i = 1; i < payloadStrList.size(); ++i) {
        outString += delim;
        outString += payloadStrList[i];
    }
    return outString;
}

/**
 * @brief Extract the topic portion of a connection-and-topic string.
 * @param[in] connTopicStr Connection-and-topic string.
 */
static std::string getTopicStr(const std::string& connTopicStr)
{
    return connTopicStr.substr(connTopicStr.find_first_of(";") + 1);
}

/**
 * @brief Convert the connection portion to the message-API format.
 * @param[in] connTopicStr Connection-and-topic string.
 */
static std::string getConnStr(const std::string& connTopicStr)
{
    auto connStr = connTopicStr.substr(0, connTopicStr.find_first_of(";"));
    connStr.replace(connStr.find(':'), 1, ";");
    return connStr;
}

/**
 * @brief Process an MQTT connection-state callback.
 * @param[in] h_ptr Message-API connection handle.
 * @param[in] evt Connection event type.
 */
static void connect_cb(NvDsMsgApiHandle h_ptr, NvDsMsgApiEventType evt)
{
    NvDsMqttClientHandle* mh = static_cast<NvDsMqttClientHandle*>(h_ptr);
    void* user_ctx = mh->user_ctx;
    MqttCommunicator* communicator = static_cast<MqttCommunicator*>(user_ctx);

    switch (evt) {
        case NVDS_MSGAPI_EVT_SUCCESS:
            printf(
                "[MqttCommunicator] Msg broker connection established between stream %ld and %s\n",
                communicator->getStreamId() >> 32, mh->connection_str
            );
            communicator->m_connStatusMap.at(std::string(mh->connection_str)) = true;
            break;

        case NVDS_MSGAPI_EVT_DISCONNECT:
            printf(
                "[MqttCommunicator] Msg broker %s connection lost on stream %ld\n",
                mh->connection_str, communicator->getStreamId() >> 32
            );
            break;

        case NVDS_MSGAPI_EVT_SERVICE_DOWN:
            printf(
                "[MqttCommunicator] Msg broker %s service down on stream %ld\n",
                mh->connection_str, communicator->getStreamId() >> 32
            );
            break;

        default:
            std::cerr << "[MqttCommunicator] Msg broker connection unexpected event: " << evt
                      << " on stream " << (communicator->getStreamId() >> 32) << std::endl;
            break;
    }
}

/**
 * @brief Process a message received from the broker.
 * @param[in] flag Message-receive status.
 * @param[in] msg Received message buffer.
 * @param[in] msg_len Received message length.
 * @param[in] topic Subscribed topic that received the message.
 * @param[in] user_ptr MQTT communicator callback context.
 */
static void subscribe_cb(NvDsMsgApiErrorType flag, void *msg, int msg_len, char *topic, void *user_ptr)
{
    if(flag != NVDS_MSGAPI_ERR) {
        MqttCommunicator *communicator = static_cast<MqttCommunicator*>(user_ptr);
        std::string strBuf((char*)msg, (char*)msg + msg_len);
        std::string msgStr = strBuf;
        auto payloadStrList = splitPayloadListStr(msgStr);
        for (auto payloadStr: payloadStrList) {
            try {
                communicator->processMessage(Payload(payloadStr));
            } catch (const std::exception& e) {
                printf("[MqttCommunicator] Error processing received message: %s\n", e.what());
            }
        }
        if (communicator->m_params.m_debugVerboseLevel > 3) {
            printf(
                "[MqttCommunicator] Communicator %ld consuming %ld message on [%s].\n",
                communicator->getStreamId()>>32, payloadStrList.size(), topic
            );
        }
    }
}

/**
 * @brief Report an asynchronous MQTT send failure.
 * @param[in] user_ptr MQTT communicator callback context.
 * @param[in] completion_flag Message-send completion status.
 */
static void send_callback (void *user_ptr, NvDsMsgApiErrorType completion_flag)
{
    if(completion_flag != NVDS_MSGAPI_OK) {
        printf("[MqttCommunicator] Send failed from mqtt msg broker \n");
    }
}

/** @brief Initialize publish and subscribe topics from configuration. */
void MqttCommunicator::initializePubSubInfo()
{
    if (m_params.m_pubSubInfoConfigPath.empty()) {
        throw std::runtime_error("[MqttCommunicator] PubSubInfoConfigPath is not set");
    }

    YAML::Node config = YAML::LoadFile(m_params.m_pubSubInfoConfigPath);

    if (config["pubBrokerTopicStr"]) {
        const YAML::Node &node = config["pubBrokerTopicStr"];
        if (node.IsMap() && !m_sensorName.empty()) {
            m_pubBrokerTopicStr = node[m_sensorName].as<std::string>();
        } else {
            m_pubBrokerTopicStr = node[m_streamId >> 32].as<std::string>();
        }
    }

    if (config["subPeerBrokerTopicStrs"]) {
        const YAML::Node &node = config["subPeerBrokerTopicStrs"];
        if (node.IsMap() && !m_sensorName.empty()) {
            m_subPeerBrokerTopicStrs = node[m_sensorName].as<std::vector<std::string>>();
        } else {
            m_subPeerBrokerTopicStrs = node[m_streamId >> 32].as<std::vector<std::string>>();
        }
    }
}

/**
 * @brief Construct an MQTT communicator from configuration.
 * @param[in] configFilePath Path to the configuration file.
 */
MqttCommunicator::MqttCommunicator(const std::string &configFilePath)
{
    m_configFilePath = configFilePath;
    m_params = MqttCommunicatorParams(configFilePath, "Communicator");
    m_isRunning = false;

    m_isInit = false;

    m_connStatusMap = std::map<std::string, bool>();
}

/** @brief Stop and destroy the MQTT communicator. */
MqttCommunicator::~MqttCommunicator()
{
    for (auto const& handleItem: m_connHandleMap) {
        NvDsMqttClientHandle *mqttH = (NvDsMqttClientHandle *)handleItem.second;
        mosquitto_loop_stop(mqttH->mosq, true);
        nvds_msgapi_disconnect(handleItem.second);
    }
    m_isRunning = false;
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
    if (m_commLatencyLogFile.is_open()) {
        m_commLatencyLogFile.close();
    }
    int maxWaitTime = 1000, waitInterval = 100;
    for (int i = 0; i < maxWaitTime; i+=waitInterval) {
        if (m_inPayloadBuffer.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(waitInterval));
    }
}

/** @brief Return whether the communicator is initialized. */
bool MqttCommunicator::isInitialized()
{
    return m_isInit;
}

/** @brief Wait until all configured MQTT connections complete or time out. */
void MqttCommunicator::waitForConnectionCompletion()
{
    /// Use millisecond intervals.
    int &waitInterval = m_params.m_waitForConnInterval;
    int &connTimeout = m_params.m_connTimeout;
    for (auto connItem: m_connStatusMap) {
        std::string connStr = connItem.first;
        if (!m_connStatusMap[connStr]) {
            printf("Waiting for mqtt communicator to establish connection...\n");
            for (int j = 0; j < connTimeout / waitInterval; j++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(waitInterval));
                if (m_connStatusMap[connStr]) {
                    break;
                }
            }
        }
    }
}

/**
 * @brief Initialize the communicator and start its processing loop.
 * @param[in] streamId Stream identifier for this communicator.
 * @param[in] start Reference start time for timestamp conversion.
 * @param[in] localInstanceCommunicators Communicators for local streams.
 * @param[in] sensorName Sensor name used by external transports.
 */
void MqttCommunicator::initialize(
    const StreamId streamId, system_clock::time_point start,
    std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators, const std::string &sensorName
)
{
    m_streamId = streamId;
    m_sensorName = sensorName;
    try {
        initializePubSubInfo();
    } catch (const std::exception& e) {
        throw std::runtime_error("[MqttCommunicator] Error initializing pub/sub info: " + std::string(e.what()));
    }
    m_isRunning = true;

    std::set<std::string> connStrSet;
    connStrSet.insert(getConnStr(m_pubBrokerTopicStr));
    for (const auto& connStr : m_subPeerBrokerTopicStrs) {
        connStrSet.insert(getConnStr(connStr));
    }

    for (const auto& connStr : connStrSet) {
        m_connStatusMap[connStr] = false;
        m_connHandleMap[connStr] = nvds_msgapi_connect(
            (char*)connStr.c_str(), connect_cb, (char*)m_params.m_mqttProtoAdaptorConfigPath.c_str()
        );
        NvDsMqttClientHandle *mqttH = (NvDsMqttClientHandle*)(m_connHandleMap[connStr]);
        mqttH->user_ctx = this;
        int rc = mosquitto_loop_start(mqttH->mosq);
        if (rc > 0) {
            printf("[MqttCommunicator] Error starting mosquitto loop: %s\n", mosquitto_strerror(rc));
        }
        if (m_params.m_debugVerboseLevel > 1) {
            printf("[MqttCommunicator] stream %ld connected to %s\n", m_streamId >> 32, connStr.c_str());
        }
    }

    std::map<std::string, std::vector<std::string>> connStr2Topics;
    for (auto connTopicStr: m_subPeerBrokerTopicStrs) {
        std::string connStr = getConnStr(connTopicStr);
        std::string topicStr = getTopicStr(connTopicStr);
        connStr2Topics[connStr].push_back(topicStr);
    }
    for (const auto& connTopicPair: connStr2Topics) {
        const auto& topics = connTopicPair.second;
        std::vector<char*> topic_ptrs(topics.size());
        std::vector<std::vector<char>> topic_buffers(topics.size());

        for (size_t i = 0; i < topics.size(); i++) {
            /// +1 for null terminator.
            topic_buffers[i].resize(topics[i].size() + 1);
            std::copy(topics[i].begin(), topics[i].end(), topic_buffers[i].begin());
            /// Ensure null termination.
            topic_buffers[i].back() = '\0';
            topic_ptrs[i] = topic_buffers[i].data();
        }

        NvDsMsgApiErrorType rc = nvds_msgapi_subscribe(
            m_connHandleMap[connTopicPair.first], topic_ptrs.data(), topics.size(), subscribe_cb, this
        );

        if (rc != NVDS_MSGAPI_OK) {
            printf("[MqttCommunicator] Subscription Error for stream %ld\n", m_streamId >> 32);
        }
    }

    /// Open the comm log file.
    if (m_params.m_saveLog2File > 0) {
        m_logFile.open("outCommunicator/Stream" + std::to_string(m_streamId >> 32) + ".log");
        m_commLatencyLogFile.open("outCommunicator/Stream" + std::to_string(m_streamId >> 32) + "_commLatency.log");
    }
    m_startTime = start;
    m_isInit = true;
    printf("[MqttCommunicator] Mqtt communicator initialized for stream %ld\n", m_streamId >> 32);
}

/** @brief Return the type of the next buffered message, or NONE when the buffer is empty. */
MessageType MqttCommunicator::available()
{
    if (m_inPayloadBuffer.empty())
        return NONE;
    else
        return m_inPayloadBuffer.front().type;
}

/**
 * @brief Request to send a message.
 * @param[in] payload Message to send.
 */
void MqttCommunicator::requestToSend(Payload payload)
{
    broadcast(payload);
}

/**
 * @brief Request to send multiple messages.
 * @param[in] payloads Messages to send.
 */
void MqttCommunicator::requestToSend(const std::vector<Payload>& payloads)
{
    broadcast(payloads);
}

/**
 * @brief Get the next message from the input buffer.
 * @param[out] payload Message removed from the input buffer.
 */
void MqttCommunicator::get(Payload &payload)
{
    std::unique_lock<std::mutex> inPayloadBufferLock(m_inPayloadBufferMutex);
    payload = m_inPayloadBuffer.front();
    m_inPayloadBuffer.pop_front();
    inPayloadBufferLock.unlock();
}

/**
 * @brief Process an incoming message.
 * @param[in] payload Incoming message to place in the input buffers.
 */
void MqttCommunicator::processMessage(Payload payload)
{
    std::unique_lock<std::mutex> inPayloadBufferLock(m_inPayloadBufferMutex);
    m_inPayloadBuffer.push_back(payload);
    inPayloadBufferLock.unlock();
    if ((m_params.m_debugVerboseLevel > 2) && (m_params.m_saveLog2File > 0)) {
        commLog("RECV", payload);
        commLatencyLog(payload.messageTimeStamp, Time().getUnixTimestampNanoSeconds());
    }
}

/**
 * @brief Broadcast one message through local and MQTT transports.
 * @param[in] payload Message to broadcast.
 */
void MqttCommunicator::broadcast(Payload payload)
{
    broadcast({payload});
}

/**
 * @brief Broadcast multiple messages through local and MQTT transports.
 * @param[in] payloads Messages to broadcast.
 */
void MqttCommunicator::broadcast(std::vector<Payload> payloads)
{
    if (payloads.empty()) return;
    std::vector<std::string> payloadStrs;
    payloadStrs.reserve(payloads.size());
    for (auto payload : payloads) {
        payloadStrs.push_back(payload.toJsonStr());
    }
    std::string payloadStrConcat = concatPayloadListStr(payloadStrs);
    std::string topic = getTopicStr(m_pubBrokerTopicStr);
    if (m_params.m_debugVerboseLevel > 3) {
        printf(
            "[MqttCommunicator] Communicator %ld broadcasting %ld messages\n", m_streamId >> 32, payloadStrs.size()
        );
    }
    if ((m_params.m_debugVerboseLevel > 2) && (m_params.m_saveLog2File > 0)) {
        for (auto payload : payloads) {
            commLog("SEND", payload);
        }
    }
    NvDsMsgApiErrorType rc = nvds_msgapi_send_async(
        m_connHandleMap[getConnStr(m_pubBrokerTopicStr)], (char*)topic.c_str(),
        reinterpret_cast<const uint8_t*>(payloadStrConcat.c_str()), payloadStrConcat.size(), send_callback, this
    );
    if (rc != NVDS_MSGAPI_OK) {
        printf("[MqttCommunicator] Publish Error for stream %ld\n", m_streamId >> 32);
    }
}

/**
 * @brief Log communication latency.
 * @param[in] sendTime Timestamp when the message was sent.
 * @param[in] recvTime Timestamp when the message was received.
 */
void MqttCommunicator::commLatencyLog(TimeStamp sendTime, TimeStamp recvTime)
{
    std::unique_lock<std::mutex> lock(m_commLatencyMutex);
    m_commLatencyLogFile << "sendTime: " << sendTime << ", recvTime: " << recvTime << ", diff: " << recvTime - sendTime
                         << "ns";
    m_commLatencyLogFile << "\n" << std::flush;
    lock.unlock();
}

/**
 * @brief Log a communication event.
 * @param[in] event Event name.
 * @param[in] payload Payload to log.
 */
void MqttCommunicator::commLog(std::string event, Payload payload)
{
    std::string type;
    switch (payload.type) {
        case TRACKLET:     type = "  TRACKLET  "; break;
        case STATE_UPDATE: type = "STATE_UPDATE"; break;
        case ADOPTED_ID:   type = " ADOPTED_ID "; break;
        case SYNC:         type = "    SYNC    "; break;
        default:           type = "    NONE    "; break;
    }
    std::unique_lock<std::mutex> lock(m_commMutex);
    m_logFile << "systemTime: " << Time().getStrSystemTime() << ", ";
    m_logFile << "time: " << Time().getStrTimeRelativeTo(m_startTime) << ", ";
    m_logFile << "frame: " << payload.frameNum << ", ";
    m_logFile << "event: " << event << ", ";
    m_logFile << "type: " << type << ", ";
    m_logFile << "fromCamID: " << payload.fromCamId << ", ";
    m_logFile << "egoCamID: " << m_pubBrokerTopicStr << ", ";
    m_logFile << "targetID: " << payload.targetId;
    if (payload.type == ADOPTED_ID)
        m_logFile << ", " << "adoptedID: " << payload.adoptedId;
    m_logFile << "\n" << std::flush;
    lock.unlock();
}
