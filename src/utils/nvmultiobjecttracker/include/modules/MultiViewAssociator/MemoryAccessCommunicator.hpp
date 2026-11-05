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

#ifndef NVMULTIOBJECTTRACKERLIB_MEMORYACCESSCOMMUNICATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_MEMORYACCESSCOMMUNICATOR_HPP_

#include "modules/MultiViewAssociator/ICommunicator.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"

/** @brief Store communicator configuration parameters. */
class CommunicatorParams : public NvTrackerParams
{
public:
	/** @brief Initialize communicator parameters with default values. */
	CommunicatorParams()
	{
		m_debugVerboseLevel = 0;
        m_communicatorType = DUMMY_COMMUNICATOR;
        m_saveLog2File = 0;
        m_mqttProtoAdaptorConfigPath = "";
        m_pubSubInfoConfigPath = "";
        m_connTimeout = 200;
        m_waitForConnInterval = 50;
	}

	/**
	 * @brief Initialize communicator parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	CommunicatorParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
		m_pubSubInfoConfigPath =
		    NvTrackerParams::resolvePathRelativeToConfig(configFilePath, m_pubSubInfoConfigPath);
		m_mqttProtoAdaptorConfigPath =
		    NvTrackerParams::resolvePathRelativeToConfig(configFilePath, m_mqttProtoAdaptorConfigPath);
	}

	/**
	 * @brief Initialize communicator parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	CommunicatorParams(const YAML::Node &fn)
	{
		read(fn);
	}

	/**
	 * @brief Update communicator parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override
	{
		*this = CommunicatorParams();
		if(!fn.size()) { return;}

		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
            if (paramKey == "debugVerboseLevel") {
                m_debugVerboseLevel = (int)fit->second.as<float>();
            } else if (paramKey == "communicatorType") {
                m_communicatorType = (COMMUNICATOR_TYPES)((int)fit->second.as<float>());
            } else if (paramKey == "saveLog2File") {
                m_saveLog2File = (int)fit->second.as<float>();
            } else if (paramKey == "pubSubInfoConfigPath") {
                m_pubSubInfoConfigPath = fit->second.as<std::string>();
            } else if (paramKey == "mqttProtoAdaptorConfigPath"){
                m_mqttProtoAdaptorConfigPath = fit->second.as<std::string>();
            } else if (paramKey == "connTimeout") {
                m_connTimeout = (int)fit->second.as<float>();
            } else if (paramKey == "waitForConnInterval") {
                m_waitForConnInterval = (int)fit->second.as<float>();
            } else {std::cerr << "Unknown parameter: " << paramKey << "\n"; }
		}
	}

	COMMUNICATOR_TYPES m_communicatorType; //!< Selects the communicator implementation.
    int m_debugVerboseLevel; //!< Controls debug message verbosity.
    int m_saveLog2File; //!< Indicates whether communicator logs are saved to a file.
    std::string m_mqttProtoAdaptorConfigPath; //!< Stores the MQTT protocol-adaptor configuration path.
    std::string m_pubSubInfoConfigPath; //!< Path to publish/subscribe configuration file.
    int m_waitForConnInterval; //!< Specifies the interval between connection checks.
    int m_connTimeout; //!< Specifies the connection timeout.
};

/** @brief MemoryAccessCommunicator class is responsible for handling the communication between the MultiViewAssociator and the TargetManager (NvTrackedObjectManager). */
class MemoryAccessCommunicator : public ICommunicator
{
public:
    /** @brief Construct an uninitialized memory-access communicator. */
    MemoryAccessCommunicator();

    /**
     * @brief Construct a memory-access communicator from configuration.
     * @param[in] configFilePath Path to the configuration file.
     */
    MemoryAccessCommunicator(const std::string &configFilePath);

    /** @brief Stop and destroy the memory-access communicator. */
    ~MemoryAccessCommunicator() override;

    /** @brief Return whether the communicator is initialized. */
    bool isInitialized() override;

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
    ) override;

    /** @brief Return the type of the next buffered message, or NONE when the buffer is empty. */
    MessageType available() override;

    /**
     * @brief Request to send a message.
     * @param[in] payload Message to send.
     */
    void requestToSend(Payload payload) override;

    /**
     * @brief Request to send a message.
     * @param[in] payloads Messages to send.
     */
    void requestToSend(const std::vector<Payload>& payloads) override;

    /**
     * @brief Get the message from buffer.
     * @param[out] payload Message removed from the input buffer.
     */
    void get(Payload &payload) override;

    /**
     * @brief Process an incoming message.
     * @param[in] payload Incoming message to place in the input buffers.
     */
    void processMessage(Payload payload) override;

    /**
     * @brief Broadcast message.
     * @param[in] payload Message to broadcast.
     */
    void broadcast(Payload payload) override;

    /** @brief Return the communicator implementation type. */
    COMMUNICATOR_TYPES getCommunicatorType() override {return COMMUNICATOR_TYPES::MEMORY_ACCESS_COMMUNICATOR;};

    /** @brief Return peer camera IDs from camera configuration. */
    std::vector<CamId> getPeerCamIds() override;

    /** @brief Return the ego camera ID from camera configuration. */
    CamId getEgoCamId() override;

private:
    /** @brief Runs the communicator loop that receives, processes, and sends messages. */
    void mainLoop();

    /**
     * @brief Helper function to log the communication events.
     * @param[in] event Event name.
     * @param[in] payload Payload to log.
     */
    void commLog(std::string event, Payload payload);

    /**
     * @brief Helper function to log the communication latency.
     * @param[in] sendTime Timestamp when the message was sent.
     * @param[in] recvTime Timestamp when the message was received.
     */
    void commLatencyLog(TimeStamp sendTime, TimeStamp recvTime);

    /**
     * @brief Initialize peer communication configuration.
     *
     * Initializes the ego camera ID, subscribed peer camera IDs, and peer
     * communicators that subscribe to the ego camera.
     *
     * @param[in] localInstanceCommunicators Map of stream IDs to communicator instances.
     */
    void initializePubSubInfo(std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators);

    CommunicatorParams m_params; //!< Stores communicator configuration parameters.
    std::ofstream m_logFile; //!< Debug log file and init timestamp.
    std::ofstream m_commLatencyLogFile; //!< Stores communication-latency log output.
    system_clock::time_point m_startTime; //!< Stores the communicator reference start time.
    std::string m_configFilePath; //!< Stores the configuration-file path.
    std::map<StreamId, std::shared_ptr<ICommunicator>> m_localInstancePeerCommunicators; //!< Maps local peer streams to their communicators.
    std::atomic<bool> m_isRunning, m_isFinished; //!< Records communicator-loop lifecycle state.
    StreamId m_streamId; //!< Stream ID.
    CamId m_camId; //!< Camera ID.
    std::vector<CamId> m_peerCamIds; //!< Peer Camera IDs that the ego cam subscribes to.
    bool m_isInit; //!< Indicates whether the communicator has been initialized.
    std::deque<Payload> m_inPayloadBuffer, m_outPayloadBuffer; //!< Incoming message buffers.
    std::mutex m_inPayloadBufferMutex, m_outPayloadBufferMutex; //!< Protects simultaneous access to the message buffers.
    std::mutex m_commMutex; //!< Serializes communicator event logging.
    std::mutex m_commLatencyMutex; //!< Serializes communication-latency logging.
};

#endif // NVMULTIOBJECTTRACKERLIB_MEMORYACCESSCOMMUNICATOR_HPP_
