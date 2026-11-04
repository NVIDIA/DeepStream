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

#ifndef NVMULTIOBJECTTRACKERLIB_DUMMYMULTIVIEWASSOCIATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_DUMMYMULTIVIEWASSOCIATOR_HPP_

#ifndef _WIN32
#include "modules/MultiViewAssociator/Communicator_factory.hpp"
#endif
#include "modules/MultiViewAssociator/IMultiViewAssociator.hpp"

/** @brief Provide a no-op multi-view associator implementation. */
class DummyMultiViewAssociator : public IMultiViewAssociator
{
public:
    /** @brief Construct an uninitialized dummy multi-view associator. */
    DummyMultiViewAssociator()
    {
        /// The dummy multi-view associator does not need a communicator.
        m_isInit = false;
    }

    /**
     * @brief Construct an uninitialized dummy multi-view associator from configuration.
     * @param[in] configFilePath Path to the configuration file.
     */
    DummyMultiViewAssociator(const std::string &configFilePath)
    {
        m_configFilePath = configFilePath;
        m_isInit = false;
    }

    /** @brief Stop and destroy the dummy multi-view associator. */
    ~DummyMultiViewAssociator() override
    {
        m_isRunning = false;
    }

    /**
     * @brief Check if the MultiViewAssociator is initialized.
     * @return True if the multi-view associator is initialized.
     */
    bool isInitialized() override { return m_isInit; }

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
    ) override
    {
        m_streamId = streamId;
        m_camId = streamId >> 32;
        m_isRunning = true;
        m_isInit = true;
    }

    /**
     * @brief Synchronize the peer-target database with communicator messages.
     * @param[in] timeMeta Current frame timing metadata.
     */
    void update(const NvMOTTimeMeta &timeMeta) override {}

    /**
     * @brief Try to reassociate a target with peer tracklets.
     * @param[in] target Target to match.
     * @return True when peer reassociation succeeds.
     */
    bool tryPeerReAssoc(std::shared_ptr<NvTrackedObject> target) override { return false; }

    /**
     * @brief Send the tracklet using the communicator.
     * @param[in] target Target whose trajectory buffer is serialized.
     */
    void sendTracklet(std::shared_ptr<NvTrackedObject> target) override {}

    /**
     * @brief Send the states using the communicator.
     * @param[in] target Target whose states are serialized.
     */
    void sendStates(std::shared_ptr<NvTrackedObject> target) override {}

    /**
     * @brief Send the AdoptedId message using the communicator.
     * @param[in] target Target whose adopted ID is sent.
     * @param[in] oldTargetId Previous target ID.
     */
    void sendAdoptedId(std::shared_ptr<NvTrackedObject> target, TargetId oldTargetId) override {}

    /** @brief Send the synchronization message using the communicator. */
    void sendSync() override {}

    /** @brief Get the communicator. */
    std::shared_ptr<ICommunicator> getCommunicator() override { return m_communicator; }

    /**
     * @brief Data association between ego detection and peer targets.
     * @param[in] detObjs Ego detections to match against peer targets.
     * @param[in] minMatchingScore4PeerAssocIou Minimum IoU required to match a detection to a peer target.
     * @return Detections that match a recent peer target.
     */
    std::deque<const DetectedObjInfo *> tryPeerDataAssoc(
        const std::deque<const DetectedObjInfo *> detObjs, const float minMatchingScore4PeerAssocIou
    ) override { return {}; };

    /** @brief Return an empty peer-target database. */
    const PeerTargetDB getPeerTargetDB() override { return PeerTargetDB(); };

private:
    /** @brief Main loop for the LatePeerReAssoc and PeerTargetDB handling based on incoming messages. */
    void mainLoop()
    {
        /// Open a log that records activity while the dummy loop is running.
        std::ofstream m_logFile("outMVReAssoc/Stream" + std::to_string(m_streamId >> 32) + ".log");
        auto loopStart = system_clock::now();
        while (m_isRunning) {
            auto loopNow = system_clock::now();
            m_logFile << "Running DummyMultiViewAssociator for cam " << m_camId << " at "
                        << duration_cast<minutes>(loopNow - loopStart).count() << ":"
                        << std::setw(2) << std::setfill('0') << duration_cast<seconds>(loopNow - loopStart).count()
                        << "." << std::setw(3) << std::setfill('0')
                        << duration_cast<milliseconds>(loopNow - loopStart).count() % 1000 << "\n" << std::flush;
            std::this_thread::sleep_for(milliseconds(1001));
            /// Clear the communicator buffer to avoid memory leak.
            while (m_communicator->available() != NONE) {
                Payload payload;
                /// The full implementation attempts late peer reassociation for tracklet messages.
                m_communicator->get(payload);
            }
        }
    }

    std::shared_ptr<ICommunicator> m_communicator; //!< Stores the owned communicator.
    std::atomic<bool> m_isRunning; //!< Indicates whether the processing loop is running.
    std::string m_configFilePath; //!< Path to the configuration file.
    StreamId m_streamId; //!< Stream ID.
    CamId m_camId; //!< Camera number.
    bool m_isInit; //!< Indicates whether the associator has been initialized.
};

#endif // NVMULTIOBJECTTRACKERLIB_DUMMYMULTIVIEWASSOCIATOR_HPP_
