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

#ifndef NVMULTIOBJECTTRACKERLIB_DUMMYCOMMUNICATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_DUMMYCOMMUNICATOR_HPP_

#include "modules/MultiViewAssociator/ICommunicator.hpp"

/** @brief DummyCommunicator is a dummy implementation of the ICommunicator interface. */
class DummyCommunicator : public ICommunicator
{
public:
    /** @brief Construct an uninitialized dummy communicator. */
    DummyCommunicator() { m_isInit = false; }

    /**
     * @brief Construct an uninitialized dummy communicator from configuration.
     * @param[in] configFilePath Path to the configuration file.
     */
    DummyCommunicator(const std::string &configFilePath) { m_isInit = false; }

    /** @brief Stop and destroy the dummy communicator. */
    ~DummyCommunicator() { m_isRunning = false; }

    /**
     * @brief Check if the communicator is initialized.
     * @return True if the communicator is initialized.
     */
    bool isInitialized() override { return m_isInit; }

    /**
     * @brief Initialize the communicator and start its processing loop.
     * @param[in] streamId Stream identifier for this communicator.
     * @param[in] start Reference start time for timestamp conversion.
     * @param[in] localInstanceCommunicators Communicators for local streams.
     * @param[in] sensorName Sensor name used by external transports.
     */
    void initialize(
        const StreamId streamId, system_clock::time_point start,
        std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators,
        const std::string &sensorName = ""
    ) override
    {
        m_streamId = streamId;
        m_isRunning = true;

        m_isInit = true;
        std::cout << "DummyCommunicator initialized for stream " << (m_streamId >> 32) << "\n";
    }

    /** @brief Return the type of the next buffered message, or NONE when the buffer is empty. */
    MessageType available() override
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
    void requestToSend(Payload payload) override {m_outPayloadBuffer.push_back(payload);};

    /**
     * @brief Request to send a message.
     * @param[in] payloads Messages to send.
     */
    void requestToSend(const std::vector<Payload>& payloads) override
    {
        m_outPayloadBuffer.insert(m_outPayloadBuffer.end(), payloads.begin(), payloads.end());
    };

    /**
     * @brief Get the message from buffer.
     * @param[out] payload Message removed from the input buffer.
     */
    void get(Payload &payload) override {payload = m_inPayloadBuffer.front(); m_inPayloadBuffer.pop_front();};

    /**
     * @brief Process an incoming message.
     * @param[in] payload Incoming message to place in the input buffers.
     */
    void processMessage(Payload payload) override {m_inPayloadBuffer.push_back(payload);};

    /**
     * @brief Broadcast message.
     * @param[in] payload Message to broadcast.
     */
    void broadcast(Payload payload) override {m_outPayloadBuffer.pop_front();};

    /** @brief Return the communicator implementation type. */
    COMMUNICATOR_TYPES getCommunicatorType() override {return COMMUNICATOR_TYPES::DUMMY_COMMUNICATOR;};

private:
    /** @brief Runs the communicator loop that receives, processes, and sends messages. */
    void mainLoop()
    {
        /// Open a log that records activity while the dummy loop is running.
        std::ofstream commLogFile("outCommunicator/Stream" + std::to_string(m_streamId >> 32) + ".log");
        auto loopStart = system_clock::now();
        while (m_isRunning) {
            auto loopNow = system_clock::now();
            commLogFile << "Running Communicator for stream " << (m_streamId >> 32) << " at "
                        << duration_cast<minutes>(loopNow - loopStart).count() << ":"
                        << std::setw(2) << std::setfill('0') << duration_cast<seconds>(loopNow - loopStart).count()
                        << "." << std::setw(3) << std::setfill('0')
                        << duration_cast<milliseconds>(loopNow - loopStart).count() % 1000 << "\n" << std::flush;
            std::this_thread::sleep_for(milliseconds(1075));
        }
    }

    std::atomic<bool> m_isRunning; //!< Indicates whether the communicator loop is running.
    StreamId m_streamId; //!< Stream ID.
    bool m_isInit; //!< Indicates whether the communicator has been initialized.
    std::deque<Payload> m_inPayloadBuffer; //!< Incoming message buffer.
    std::deque<Payload> m_outPayloadBuffer; //!< Outgoing message buffer.
};

#endif // NVMULTIOBJECTTRACKERLIB_DUMMYCOMMUNICATOR_HPP_
