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


#ifndef DS3D_COMMON_IMPL_BASE_DATA_RENDER_H
#define DS3D_COMMON_IMPL_BASE_DATA_RENDER_H

#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/impl/impl_dataprocess.h"

namespace ds3d { namespace impl {

/**
 * @brief Any custom datarender must derive from BaseImplDataRender,
 *
 *   For custom lib implementation, user need to implement the following
 *   virtual functions:
 *     startImpl(...), // user also need setCaps(port) in startImpl
 *     stopImpl(), // stop all resources for datarender
 *     prepollImpl(data), // prepoll on 1st coming data
 *     renderImpl(data, dataConsumedCb), // rendering data. Once data is done, invoke
 *         dataConsumedCb(datamap) callback to notify application
 *     flushImpl(), // flush data in process
 */
class BaseImplDataRender : public BaseImplDataProcessor<abiDataRender> {
public:
    BaseImplDataRender() {}
    ~BaseImplDataRender() override = default;

    ErrCode preroll_i(const abiRefDataMap* inputData) final
    {
        DS_ASSERT(inputData);
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState, "datarender is not started");
        GuardDataMap data(*inputData);
        return prerollImpl(std::move(data));
    }

    const abiRefWindow* getWindow_i() const final {
        return _window.abiRef();
    }

    ErrCode render_i(const abiRefDataMap* inputData, const abiOnDataCB* dataDoneCb) final
    {
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState, "datarender is not started");

        DS_ASSERT(inputData);
        GuardDataMap inData(*inputData);
        GuardCB<abiOnDataCB> guardDoneCb(dataDoneCb ? dataDoneCb->refCopy() : nullptr);
        OnGuardDataCBImpl consumedCB = [guardCb = std::move(guardDoneCb), this](
                                           ErrCode code, GuardDataMap data) {
            guardCb(code, data.abiRef());
        };
        return renderImpl(std::move(inData), std::move(consumedCB));
    }

    ErrCode stop_i() final {
        _window.reset();
        return BaseImplDataProcessor<abiDataRender>::stop_i();
    }

protected:
    // cplusplus virtual interface, user need to derive from
    virtual ErrCode prerollImpl(GuardDataMap datamap) = 0;
    virtual ErrCode renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb) = 0;
    void setWindow(GuardWindow window) {
        _window = window;
    }
private:
    GuardWindow _window;
};

}}  // namespace ds3d::impl

#endif  // DS3D_COMMON_IMPL_BASE_DATA_RENDER_H
