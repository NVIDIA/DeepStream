/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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