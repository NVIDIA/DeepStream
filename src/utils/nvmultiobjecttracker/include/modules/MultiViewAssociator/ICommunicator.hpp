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

#ifndef NVMULTIOBJECTTRACKERLIB_ICOMMUNICATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_ICOMMUNICATOR_HPP_

#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/utility/timeStamp.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using namespace std::chrono;

/** @brief Communicator implementations that can be selected by configuration. */
enum COMMUNICATOR_TYPES : int {
    DUMMY_COMMUNICATOR = 0, //!< Selects the dummy communicator.
    MEMORY_ACCESS_COMMUNICATOR, //!< Selects the in-process memory-access communicator.
    MQTT_COMMUNICATOR //!< Selects the MQTT communicator.
};

/** @brief Message kinds exchanged between multi-view tracker instances. */
enum MessageType : int {
    NONE = 0, //!< Indicates that no message is available.
    TRACKLET, //!< Identifies a tracklet message.
    STATE_UPDATE, //!< Identifies a target-state update.
    ADOPTED_ID, //!< Identifies an adopted-ID update.
    SYNC, //!< Identifies a synchronization message.
};

/** @brief Tracklet point: position + symmetric covariance (sxy = syx) Covariance values < 0 indicate unavailable (sentinel). */
struct TrackletPoint {
    NvMOTFrameNum frameNum; //!< Frame number.
    float x, y; //!< Position.
    float sxx, sxy, syy; //!< Symmetric covariance.

    /**
     * @brief Construct a timestamped tracklet point with symmetric covariance.
     * @param[in] frameNum_ Frame number for the point.
     * @param[in] x_ Horizontal position.
     * @param[in] y_ Vertical position.
     * @param[in] sxx_ Horizontal variance.
     * @param[in] sxy_ Cross covariance.
     * @param[in] syy_ Vertical variance.
     */
    TrackletPoint(NvMOTFrameNum frameNum_, float x_, float y_, float sxx_, float sxy_, float syy_)
        : frameNum(frameNum_),
          x(x_),
          y(y_),
          sxx(sxx_),
          sxy(sxy_),
          syy(syy_)
    {}
};
typedef std::deque<TrackletPoint> Tracklet;

/** @brief Store the latest state and trajectory received for a peer target. */
class PeerTarget {
  public:
    NvMOTFrameNum frameNum; //!< Frame number (of the last update).
    CamId cameraId; //!< Camera id.
    TargetId targetId; //!< Target id.
    TimeStamp targetIDTimeStamp; //!< Timestamp when target ID is created.
    NvMOTClassId classId; //!< Class id.
    NvMOTFrameNum age; //!< Age of the target.
    Eigen::VectorXd state; //!< State vector (in the last update).
    Eigen::VectorXd nextState; //!< Predicted feet location.
    Eigen::Matrix2d feetCov; //!< Feet location covariance.
    float visibility; //!< Visibility (in the last update).
    float distFromCam; //!< Distance from the camera (in the last update).
    float height; //!< Adaptive height of the target.
    float heightConfidence; //!< Confidence of height estimation (0 = no pose data).
    NvMotDataTypes::Rect2f tBbox; //!< Last active-frame OMP bbox (image-space pixels); for peer bboxSim gate.
    std::vector<float> reidEmbedding; //!< L2-normalized ReID embedding; empty if unavailable.
    Tracklet tracklet; //!< Stores recent world-space positions for the peer target.
};

/** @brief Represent a message exchanged between multi-view tracker instances. */
struct Payload: PeerTarget {
    MessageType type; //!< Identifies the message contents.
    CamId fromCamId = ""; //!< Identifies the sending camera.
    TargetId adoptedId = 0; //!< Stores the target ID adopted by a peer.
    TimeStamp messageTimeStamp = 0; //!< Stores the message creation timestamp.
    bool response = false; //!< Indicates whether this message is a response.

    /** @brief Serialize the payload as a JSON string. */
    std::string toJsonStr()
    {
        /// Serialize tracklet as flat arrays: [frameNum, x, y, sxx, sxy, syy].
        json trackletJson = json::array();
        for (const auto& p : tracklet) {
            trackletJson.push_back({p.frameNum, p.x, p.y, p.sxx, p.sxy, p.syy});
        }
        json payloadJson = {
            {"frameNum", frameNum},
            {"cameraId", cameraId},
            {"targetId", targetId},
            {"classId", classId},
            {"age", age},
            {"state", state},
            {"nextState", nextState},
            {"visibility", visibility},
            {"distFromCam", distFromCam},
            {"type", type},
            {"fromCamId", fromCamId},
            {"adoptedId", adoptedId},
            {"targetIDTimeStamp", targetIDTimeStamp},
            {"messageTimeStamp", messageTimeStamp},
            {"tracklet", trackletJson},
            {"height", height},
            {"heightConfidence", heightConfidence},
            {"feetCov", {feetCov(0,0), feetCov(0,1), feetCov(1,0), feetCov(1,1)}},
            /// TBbox: [x1, y1, x2, y2] using Eigen::AlignedBox min/max corners.
            {"tBbox", {tBbox.min()(0), tBbox.min()(1), tBbox.max()(0), tBbox.max()(1)}},
        };
        /// Serialize the ReID embedding only when it is non-empty.
        if (!reidEmbedding.empty())
            payloadJson["reid"] = reidEmbedding;
        return payloadJson.dump();
    }

    /** @brief Construct an empty payload. */
    Payload() = default;

    /**
     * @brief Deserialize a payload from JSON.
     * @param[in] jsonStr Serialized payload JSON string.
     */
    Payload(std::string jsonStr)
    {
        json jsonObj = json::parse(jsonStr);
        frameNum = jsonObj["frameNum"];
        cameraId = jsonObj["cameraId"];
        targetId = jsonObj["targetId"];
        classId = jsonObj["classId"];
        age = jsonObj["age"];
        visibility = jsonObj["visibility"];
        distFromCam = jsonObj["distFromCam"];
        height = jsonObj.value("height", 1.6f);
        heightConfidence = jsonObj.value("heightConfidence", 0.f);
        type = jsonObj["type"];
        fromCamId = jsonObj["fromCamId"];
        adoptedId = jsonObj["adoptedId"];
        targetIDTimeStamp = jsonObj["targetIDTimeStamp"];
        messageTimeStamp = jsonObj["messageTimeStamp"];

        state = Eigen::VectorXd::Zero(4);
        for (uint i = 0; i < jsonObj["state"].size(); ++i) {
            state(i) = jsonObj["state"][i];
        }
        nextState = Eigen::VectorXd::Zero(4);
        for (uint i = 0; i < jsonObj["nextState"].size(); ++i) {
            nextState(i) = jsonObj["nextState"][i];
        }
        auto trackletJson = jsonObj["tracklet"];
        for (auto el : trackletJson) {
            tracklet.push_back(TrackletPoint(el[0], el[1], el[2], el[3], el[4], el[5]));
        }
        auto covArr = jsonObj["feetCov"];
        feetCov << covArr[0].get<double>(), covArr[1].get<double>(), covArr[2].get<double>(), covArr[3].get<double>();
        /// TBbox: deserialize [x1, y1, x2, y2] into Eigen::AlignedBox min/max corners.
        if (jsonObj.contains("tBbox")) {
            auto bb = jsonObj["tBbox"];
            tBbox = NvMotDataTypes::Rect2f(
                Eigen::Vector2f(bb[0].get<float>(), bb[1].get<float>()),
                Eigen::Vector2f(bb[2].get<float>(), bb[3].get<float>())
            );
        }
        /// Read the ReID embedding when the payload contains one.
        if (jsonObj.contains("reid"))
            reidEmbedding = jsonObj["reid"].get<std::vector<float>>();
    }
};

/** @brief Interface for the communicator. */
class ICommunicator
{
public:
    /** @brief Destroy the communicator interface. */
    virtual ~ICommunicator() {};

    /**
     * @brief Check if the communicator is initialized.
     * @return True if the communicator is initialized.
     */
    virtual bool isInitialized() = 0;

    /**
     * @brief Initialize the communicator and start its processing loop.
     * @param[in] streamId Stream identifier for this communicator.
     * @param[in] start Reference start time for timestamp conversion.
     * @param[in] localInstanceCommunicators Communicators for local streams.
     * @param[in] sensorName Sensor name used by external transports.
     */
    virtual void initialize(
        const StreamId streamId, system_clock::time_point start,
        std::map<StreamId, std::shared_ptr<ICommunicator>> localInstanceCommunicators,
        const std::string &sensorName = ""
    ) = 0;

    /** @brief Return the type of the next buffered message, or NONE when the buffer is empty. */
    virtual MessageType available() = 0;

    /**
     * @brief Request to send a message.
     * @param[in] payload Message to send.
     */
    virtual void requestToSend(Payload payload) = 0;

    /**
     * @brief Request to send a message.
     * @param[in] payload Messages to send.
     */
    virtual void requestToSend(const std::vector<Payload>& payload) = 0;

    /**
     * @brief Get the message from buffer.
     * @param[out] payload Message removed from the input buffer.
     */
    virtual void get(Payload &payload) = 0;

    /**
     * @brief Process an incoming message.
     * @param[in] payload Incoming message to place in the input buffers.
     */
    virtual void processMessage(Payload payload) = 0;

    /**
     * @brief Broadcast message.
     * @param[in] payload Message to broadcast.
     */
    virtual void broadcast(Payload payload) = 0;

    /** @brief Return the communicator implementation type. */
    virtual COMMUNICATOR_TYPES getCommunicatorType() = 0;

    /** @brief Return peer camera IDs from broker topics or camera configuration. */
    virtual std::vector<CamId> getPeerCamIds()
    {
        return std::vector<CamId>();
    }

    /** @brief Return the ego camera ID from the broker topic or camera configuration. */
    virtual CamId getEgoCamId()
    {
        return std::string();
    }
};

#endif // NVMULTIOBJECTTRACKERLIB_ICOMMUNICATOR_HPP_
