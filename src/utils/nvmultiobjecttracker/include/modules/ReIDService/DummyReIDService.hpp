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

#ifndef DUMMYREIDSERVICE_HPP_
#define DUMMYREIDSERVICE_HPP_

#include <deque>
#include <map>
#include "modules/ReIDService/IReIDService.hpp"

/** @brief Provides a no-op remote ReID service implementation. */
class DummyReIDService: public IReIDService
{
public:
    /**
     * @brief Construct a DummyReIDService instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     */
    DummyReIDService(const std::string &configFilePath)
    {
        m_isInit = false;
    };

    /** @brief Release resources owned by the DummyReIDService instance. */
    virtual ~DummyReIDService(){};

    /**
     * @brief Initialize variables and allocate memories.
     * @param[in] frameInfo Metadata about the source image, such as size and color.
     * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
     * @param[in] targetManager Target manager of the tracked objects.
     */
    void initialize(
        const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
        const std::shared_ptr<NvTrackedObjectManager> &targetManager
    ) override
    {
        m_isInit = true;
    };

    /**
     * @brief Populate the capabilities reported to the tracker.
     * @param[out] paramsQuery The output query parameters.
     */
    void queryParams(NvMOTQuery &paramsQuery){};

    /**
     * @brief Send query to reid service for all unmatched active objects.
     * @param[in] inputMeta Input frames keyed by stream ID.
     */
    void query(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta){};

    /** @brief Update target ID using remoteTargetID from reid service. */
    void updateTargetIDs(){};

    /** @brief Get the configured ReID-service implementation type. */
    const REID_SERVICE_TYPES getReIDServiceType() override {return REID_SERVICE_TYPES::DUMMY_REID_SERVICE; }

protected:
    bool m_isInit; //!< Reid status if runtime variables initialized.
};

#endif // DUMMYREIDSERVICE_HPP_
