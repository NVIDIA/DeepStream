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

/** @brief Construct an uninitialized memory-access communicator. */
MemoryAccessCommunicator::MemoryAccessCommunicator()
{
    m_params = CommunicatorParams(m_configFilePath, "Communicator");
    m_isRunning = false;
    m_isFinished = false;
    m_isInit = false;
}

/**
 * @brief Construct a memory-access communicator from configuration.
 * @param[in] configFilePath Path to the configuration file.
 */
MemoryAccessCommunicator::MemoryAccessCommunicator(const std::string &configFilePath)
{
    m_configFilePath = configFilePath;
    m_params = CommunicatorParams(m_configFilePath, "Communicator");
    m_isRunning = false;
    m_isFinished = false;
    m_isInit = false;
}

/** @brief Stop and destroy the memory-access communicator. */
MemoryAccessCommunicator::~MemoryAccessCommunicator()
{
    m_isRunning = false;
    /// Hold until the thread finishes.
    while(!m_isFinished);
}

/** @brief Return whether the communicator is initialized. */
bool MemoryAccessCommunicator::isInitialized()
{
    return m_isInit;
}

/**
 * @brief Initialize peer communication configuration.
 * @param[in] localInstanceCommunicators Map of stream IDs to communicator instances.
 */
void MemoryAccessCommunicator::initializePubSubInfo(
    std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators
)
{
    if (m_params.m_pubSubInfoConfigPath.empty()) {
        throw std::runtime_error("[MemoryAccessCommunicator] PubSubInfoConfigPath is not set");
    }
    YAML::Node config = YAML::LoadFile(m_params.m_pubSubInfoConfigPath);
    if (config["camIds"]) {
        m_camId = config["camIds"][m_streamId >> 32].as<CamId>();
    }
    if (config["subToCamIds"]) {
        m_peerCamIds = config["subToCamIds"][m_streamId >> 32]
                                      .as<std::vector<CamId>>();
    }
    /// Only store the communicators that subscribe to the ego cam.
    for (auto &streamCommunicator : localInstanceCommunicators) {
        StreamId streamId = streamCommunicator.first;
        if (streamId == m_streamId) continue;
        auto peerStreamSubToCamIds = config["subToCamIds"][streamId >> 32].as<std::vector<CamId>>();
        if (find(peerStreamSubToCamIds.begin(), peerStreamSubToCamIds.end(), m_camId) != peerStreamSubToCamIds.end()) {
            m_localInstancePeerCommunicators[streamId] = streamCommunicator.second;
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
void MemoryAccessCommunicator::initialize(
    const StreamId streamId, system_clock::time_point start,
    std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators, const std::string &/*sensorName*/
)
{
    m_streamId = streamId;
    try {
        initializePubSubInfo(localInstanceCommunicators);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "WARNING: PubSubInfoConfig is not valid, using fully-connected subscription graph." << std::endl;
        m_camId = std::to_string(m_streamId >> 32);
        for (auto &streamCommunicator : localInstanceCommunicators) {
            StreamId streamId = streamCommunicator.first;
            if (streamId == m_streamId) continue;
            m_peerCamIds.push_back(std::to_string(streamId >> 32));
            m_localInstancePeerCommunicators[streamId] = streamCommunicator.second;
        }
    }

    m_startTime = start;
    if (m_params.m_saveLog2File) {
        /// Open the comm log file.
        m_logFile.open("outCommunicator/Stream" + std::to_string(m_streamId >> 32) + ".log");
        m_logFile << "systemTime: " << Time().getStrSystemTime() << ", ";
        m_logFile << "time: " << Time().getStrTimeRelativeTo(m_startTime) << ", ";
        m_logFile << "subscriber streams: ";
        for (auto &localPeerCommunicator : m_localInstancePeerCommunicators) m_logFile << (localPeerCommunicator.first >> 32) << " ";
        m_logFile << "\n";
        if (m_params.m_debugVerboseLevel > 1) {
            m_commLatencyLogFile.open("outCommunicator/Stream" + std::to_string(m_streamId >> 32) + "_commLatency.log");
        }
    }

    m_isRunning = true;
    std::thread mainLoopThread([this](){ mainLoop(); });
    mainLoopThread.detach();
    m_isInit = true;
    if (m_params.m_debugVerboseLevel > 1) {
        std::cout << "MemoryAccessCommunicator initialized for stream " << (m_streamId >> 32)
                  << " (cam " << m_camId << ")\n";
    }
}

/** @brief Return peer camera IDs from camera configuration. */
std::vector<CamId> MemoryAccessCommunicator::getPeerCamIds()
{
    return m_peerCamIds;
}

/** @brief Return the ego camera ID from camera configuration. */
CamId MemoryAccessCommunicator::getEgoCamId()
{
    return m_camId;
}

/** @brief Return the type of the next buffered message, or NONE when the buffer is empty. */
MessageType MemoryAccessCommunicator::available()
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
void MemoryAccessCommunicator::requestToSend(Payload payload)
{
    std::unique_lock<std::mutex> outPayloadBufferLock(m_outPayloadBufferMutex);
    m_outPayloadBuffer.push_back(payload);
    outPayloadBufferLock.unlock();
}

/**
 * @brief Request to send multiple messages.
 * @param[in] payloads Messages to send.
 */
void MemoryAccessCommunicator::requestToSend(const std::vector<Payload>& payloads)
{
    std::unique_lock<std::mutex> outPayloadBufferLock(m_outPayloadBufferMutex);
    m_outPayloadBuffer.insert(m_outPayloadBuffer.end(), payloads.begin(), payloads.end());
    outPayloadBufferLock.unlock();
}

/**
 * @brief Get the next message from the input buffer.
 * @param[out] payload Message removed from the input buffer.
 */
void MemoryAccessCommunicator::get(Payload &payload)
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
void MemoryAccessCommunicator::processMessage(Payload payload)
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
 * @brief Broadcast a message to subscribed local communicators.
 * @param[in] payload Message to broadcast.
 */
void MemoryAccessCommunicator::broadcast(Payload payload)
{
    for(auto &peerCommunicator : m_localInstancePeerCommunicators) {
        peerCommunicator.second->processMessage(payload);
    }
}

/** @brief Run the communicator loop that sends queued messages. */
void MemoryAccessCommunicator::mainLoop()
{
    m_logFile << "systemTime: " << Time().getStrSystemTime() << ", ";
    m_logFile << "time: " << Time().getStrTimeRelativeTo(m_startTime);
    m_logFile << " - Running Communicator for cam " << m_camId << "\n" << std::flush;

    while (m_isRunning) {
        /// Send outgoing messages.
        if (!m_outPayloadBuffer.empty()) {
            std::unique_lock<std::mutex> outPayloadBufferLock(m_outPayloadBufferMutex);
            auto payload = m_outPayloadBuffer.front();
            m_outPayloadBuffer.pop_front();
            outPayloadBufferLock.unlock();
            if ((m_params.m_debugVerboseLevel > 2) && (m_params.m_saveLog2File > 0)) {
                commLog("SEND", payload);
            }
            broadcast(payload);
        }
    }
    m_isFinished = true;
}

/**
 * @brief Log a communication event.
 * @param[in] event Event name.
 * @param[in] payload Payload to log.
 */
void MemoryAccessCommunicator::commLog(std::string event, Payload payload)
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
    m_logFile << "egoCamID: " << m_camId << ", ";
    m_logFile << "targetID: " << payload.targetId;
    if (payload.type == ADOPTED_ID)
        m_logFile << ", " << "adoptedID: " << payload.adoptedId;
    m_logFile << "\n" << std::flush;
    lock.unlock();
}

/**
 * @brief Log communication latency.
 * @param[in] sendTime Timestamp when the message was sent.
 * @param[in] recvTime Timestamp when the message was received.
 */
void MemoryAccessCommunicator::commLatencyLog(TimeStamp sendTime, TimeStamp recvTime)
{
    std::unique_lock<std::mutex> lock(m_commLatencyMutex);
    m_commLatencyLogFile << "sendTime: " << sendTime << ", recvTime: " << recvTime << ", diff: " << recvTime - sendTime
                         << "ns";
    m_commLatencyLogFile << "\n" << std::flush;
    lock.unlock();
}
