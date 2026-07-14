/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DS3D_COMMON_IMPL_BASE_DATA_MIXER_H
#define DS3D_COMMON_IMPL_BASE_DATA_MIXER_H

#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/impl/impl_dataprocess.h"

namespace ds3d { namespace impl {

/**
 * @brief Any custom datamixer must derive from BaseImplDataMixer,
 *
 *   For custom lib implementation, user need to implement the following
 *   virtual functions:
 *     startImpl(...), // user also need setCaps(port) in startImpl
 *     stopImpl(), // stop all resources for datamixer
 *     prepollImpl(data), // prepoll on 1st coming data
 *     renderImpl(data, dataConsumedCb), // rendering data. Once data is done, invoke
 *         dataConsumedCb(datamap) callback to notify application
 *     flushImpl(), // flush data in process
 */
class BaseImplDataMixer : public BaseImplDataProcessor<abiDataMixer> {
public:
    BaseImplDataMixer() {}
    ~BaseImplDataMixer() override = default;

    ErrCode process_i(
        const int portId,
        const abiRefDataMap* inputData,
        const abiOnDataCB* inputDataConsumedCb) final
    {
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState, "datamixer is not started");

        DS_ASSERT(inputData);
        GuardDataMap inData(*inputData);
        GuardCB<abiOnDataCB> guardConsumedCb(
            inputDataConsumedCb ? inputDataConsumedCb->refCopy() : nullptr);
        OnGuardDataCBImpl consumedCbImpl = [gConsumedCb = std::move(guardConsumedCb), this](
                                               ErrCode code, GuardDataMap data) {
            gConsumedCb(code, data.abiRef());
        };
        return processImpl(portId, std::move(inData), std::move(consumedCbImpl));
    }

    ErrCode setOutputCb_i(
        const abiOnDataCB* outputDataCb) final
    {
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState, "datamixer is not started");

        DS_ASSERT(outputDataCb);
        GuardCB<abiOnDataCB> guardOutputCb(outputDataCb ? outputDataCb->refCopy() : nullptr);
        OnGuardDataCBImpl outputCbImpl = [gOutputCb = std::move(guardOutputCb), this](
                                             ErrCode code, GuardDataMap data) {
            gOutputCb(code, data.abiRef());
        };

        return setOutputCbImpl(std::move(outputCbImpl));
    }

    ErrCode updateInput_i(int portId, MixerUpdate updateType) { return updateInputImpl(portId, updateType); }

protected:
    // cplusplus virtual interface, user need to derive from
    virtual ErrCode processImpl(
        const int portId,
        GuardDataMap datamap,
        OnGuardDataCBImpl inputConsumedCb) = 0;
    virtual ErrCode setOutputCbImpl(
        OnGuardDataCBImpl outputDataCb) = 0;
    virtual ErrCode updateInputImpl(int portId, MixerUpdate updateType) = 0;
};

}}  // namespace ds3d::impl

#endif  // DS3D_COMMON_IMPL_BASE_DATA_MIXER_H
