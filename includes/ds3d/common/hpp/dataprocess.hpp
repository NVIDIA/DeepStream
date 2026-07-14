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

#ifndef DS3D_COMMON_HPP_DATA_PROCESS_HPP
#define DS3D_COMMON_HPP_DATA_PROCESS_HPP

#include <ds3d/common/abi_dataprocess.h>
#include <ds3d/common/abi_frame.h>
#include <ds3d/common/abi_obj.h>
#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/obj.hpp>

namespace ds3d {

template <class abiDataProcessorT, _EnableIfBaseOf<abiProcess, abiDataProcessorT> = true>
class GuardDataProcess : public GuardDataT<abiDataProcessorT> {
    using _Base = GuardDataT<abiDataProcessorT>;

protected:
    GuardDataProcess() = default;
    template <typename... Args /*, _EnableIfConstructible<_Base, Args...> = true*/>
    GuardDataProcess(Args&&... args) : _Base(std::forward<Args>(args)...)
    {
    }
    using GuardDataT<abiDataProcessorT>::ptr;

public:
    ~GuardDataProcess() = default;

    template <typename DelF>
    void setUserData(void* data, DelF delF)
    {
        DS_ASSERT(ptr());
        if (!data) {
            ptr()->setUserData_i(nullptr);
            return;
        }
        SharedRefObj<void> uData(data, std::move(delF));
        ptr()->setUserData_i(&uData);
    }

    virtual void* getUserData() const
    {
        DS_ASSERT(ptr());
        const abiRefAny* udata = ptr()->getUserData_i();
        return (udata ? udata->data() : nullptr);
    }

    void setErrorCallback(abiErrorCB::CppFunc errCb)
    {
        DS_ASSERT(ptr());
        GuardCB<abiErrorCB> guardErrCb;
        guardErrCb.setFn<ErrCode, const char*>(std::move(errCb));
        ptr()->setErrorCallback_i(*guardErrCb.abiRef());
    }

    ErrCode start(const std::string& content, const std::string& path = "")
    {
        DS_ASSERT(ptr());
        return ptr()->start_i(content.c_str(), content.size(), path.c_str());
    }

    ErrCode flush()
    {
        DS_ASSERT(ptr());
        return ptr()->flush_i();
    }

    ErrCode stop()
    {
        DS_ASSERT(ptr());
        return ptr()->stop_i();
    }

    State state() const
    {
        DS_ASSERT(ptr());
        return ptr()->state_i();
    }

    std::string getCaps(CapsPort port)
    {
        DS_ASSERT(ptr());
        return cppString(ptr()->getCaps_i(port));
    }

    std::string getOutputCaps() { return getCaps(CapsPort::kOutput); }
    std::string getInputCaps() { return getCaps(CapsPort::kInput); }
};

}  // namespace ds3d

#endif  // DS3D_COMMON_HPP_DATA_PROCESS_HPP