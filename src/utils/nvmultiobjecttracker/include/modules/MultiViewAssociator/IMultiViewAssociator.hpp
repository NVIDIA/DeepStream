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

#ifndef NVMULTIOBJECTTRACKERLIB_IMULTIVIEWASSOCIATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_IMULTIVIEWASSOCIATOR_HPP_

#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/ObjectModelProjector/ObjectModelProjector.hpp"
#include "modules/MultiViewAssociator/ICommunicator.hpp"

/** @brief MultiViewAssociator types. */
enum MULTI_VIEW_ASSOCIATOR_TYPES : int {
    DUMMY_MULTI_VIEW_ASSOCIATOR = 0, //!< Selects the dummy multi-view associator.
    MULTI_VIEW_ASSOCIATOR, //!< Selects the full multi-view associator.
};

/** @brief PeerTargetDB structure. */
typedef std::map<CamId, std::map<TargetId, PeerTarget>> PeerTargetDB;

/** @brief Define the interface for peer and late-peer reassociation. */
class IMultiViewAssociator
{
public:
    /** @brief Destroy the multi-view associator interface. */
    virtual ~IMultiViewAssociator() {};

    /**
     * @brief Check if the MultiViewAssociator is initialized.
     * @return True if the multi-view associator is initialized.
     */
    virtual bool isInitialized() = 0;

    /**
     * @brief Initialize peer-target state and the communicator.
     * @param[in] streamId Stream identifier for the ego camera.
     * @param[in] start Reference start time for timestamp conversion.
     * @param[in] localInstanceCommunicators Communicators for local streams.
     * @param[in] egoTargets Ego-camera targets managed by the tracker.
     * @param[in] objModelProjector Projector used to recover world-space target state.
     */
    virtual void initialize(
        const StreamId streamId, system_clock::time_point start,
        std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators,
        std::shared_ptr<NvTrackedObjectList> egoTargets, std::shared_ptr<ObjectModelProjector> objModelProjector
    ) = 0;

    /**
     * @brief Synchronize the peer-target database with communicator messages.
     * @param[in] timeMeta Current frame timing metadata.
     */
    virtual void update(const NvMOTTimeMeta &timeMeta) = 0;

    /**
     * @brief Try to reassociate a target with peer tracklets.
     * @param[in] target Target to match.
     * @return True when peer reassociation succeeds.
     */
    virtual bool tryPeerReAssoc(std::shared_ptr<NvTrackedObject> target) = 0;

    /**
     * @brief Send the tracklet using the communicator.
     * @param[in] target Target whose trajectory buffer is serialized.
     */
    virtual void sendTracklet(std::shared_ptr<NvTrackedObject> target) = 0;

    /**
     * @brief Send the states using the communicator.
     * @param[in] target Target whose states are serialized.
     */
    virtual void sendStates(std::shared_ptr<NvTrackedObject> target) = 0;

    /**
     * @brief Send the AdoptedId message using the communicator.
     * @param[in] target Target whose adopted ID is sent.
     * @param[in] oldTargetId Previous target ID.
     */
    virtual void sendAdoptedId(std::shared_ptr<NvTrackedObject> target, TargetId oldTargetId) = 0;

    /** @brief Send the synchronization message using the communicator. */
    virtual void sendSync() = 0;

    /** @brief Get the communicator. */
    virtual std::shared_ptr<ICommunicator> getCommunicator() = 0;

    /**
     * @brief Data association between ego detection and peer targets.
     * @param[in] detObjs Ego detections to match against peer targets.
     * @param[in] minMatchingScore4PeerAssocIou Minimum IoU required to match a detection to a peer target.
     * @return Detections that match a recent peer target.
     */
    virtual std::deque<const DetectedObjInfo *> tryPeerDataAssoc(
        const std::deque<const DetectedObjInfo *> detObjs, const float minMatchingScore4PeerAssocIou
    ) = 0;

    /** @brief Return the peer-target database. */
    virtual const PeerTargetDB getPeerTargetDB() = 0;
};

#endif // NVMULTIOBJECTTRACKERLIB_IMULTIVIEWASSOCIATOR_HPP_
