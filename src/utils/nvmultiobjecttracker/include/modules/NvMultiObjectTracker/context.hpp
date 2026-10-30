/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _NvMOTCONTEXT_H
#define _NvMOTCONTEXT_H

#include "modules/NvMultiObjectTracker/localizer.hpp"

/** @brief Context for input video streams. */
class NvMOTContext
{
public:
    /**
     * @brief Initialize a tracking context from the negotiated configuration.
     * @param[in] configIn Tracker configuration supplied by the client.
     * @param[out] configResponse Configuration response populated with initialization status.
     */
    NvMOTContext(const NvMOTConfig &configIn, NvMOTConfigResponse& configResponse);

    /** @brief Release the batch localizer owned by the context. */
    ~NvMOTContext();

    /**
     * @brief Output query parameters.
     * @param[out] paramsQuery Query structure populated with tracker requirements.
     */
    NvMOTStatus queryParams(NvMOTQuery &paramsQuery);

    /**
     * @brief Process a batch of frames.
     * @param[in] params Parameters for the frames to process.
     * @param[out] pTrackedObjectsBatch Object tracks produced for the batch.
     */
    NvMOTStatus processFrame(const NvMOTProcessParams *params, NvMOTTrackedObjBatch *pTrackedObjectsBatch);

    /**
     * @brief Output miscellaneous past-frame data when it is available.
     * @param[in] params Parameters for the frames to process.
     * @param[out] pTrackerMiscData Miscellaneous tracker output.
     */
    NvMOTStatus retrieveMiscData(const NvMOTProcessParams *params, NvMOTTrackerMiscData *pTrackerMiscData);

    /**
     * @brief Terminate trackers and release resources for a stream when the stream is removed.
     * @param[in] streamIdMask Stream ID mask selecting streams to remove.
     */
    NvMOTStatus removeStream(const NvMOTStreamId streamIdMask);

    /**
     * @brief Update the subset of tracker parameters that supports dynamic configuration.
     * @param[in] configStr Stringified YAML configuration parameters to update.
     */
    NvMOTStatus updateParams(const std::string &configStr);

protected:
    NvMOTConfig m_Config; //!< Tracker configuration retained by the context.
    std::shared_ptr<IBatchLocalizer> m_pLocalizer; //!< Batch localizer used by this context.
};

#endif // _NvMOTCONTEXT_H
