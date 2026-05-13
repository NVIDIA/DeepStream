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


#ifndef _DS3D_COMMON_IMPL_BASE_DATA_PROCESS_H
#define _DS3D_COMMON_IMPL_BASE_DATA_PROCESS_H

#include <ds3d/common/abi_dataprocess.h>
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/obj.hpp"

namespace ds3d { namespace impl {

/** BaseProcessIF could be abiDataLoader/abiDataRender/abiDataFilter or any others
 *    abi interface that derived from abiProcess
 */
template <class abiDataProcessorT, _EnableIfBaseOf<abiProcess, abiDataProcessorT> = true>
class BaseImplDataProcessor : public abiDataProcessorT {
public:
    using OnGuardDataCBImpl = std::function<void(ErrCode, GuardDataMap)>;
    using OnGuardBridgeDataCBImpl = std::function<void(ErrCode, const struct VideoBridge2dInput*)>;
    using ImplMutex = std::recursive_mutex;

    BaseImplDataProcessor() {}
    ~BaseImplDataProcessor() override
    {
        _errCB.reset();
        _userData.reset();
    }

    void setUserData_i(const abiRefAny* userdata) final
    {
        if (userdata) {
            _userData = GuardDataT<void>(*userdata);
        } else {
            _userData.reset();
        }
    }
    const abiRefAny* getUserData_i() const final { return _userData.abiRef(); }
    void setErrorCallback_i(const abiErrorCB& cb) final { _errCB = GuardCB<abiErrorCB>(cb); }
    ErrCode start_i(const char* configStr, uint32_t strLen, const char* path) override
    {
        if (getStateSafe() == State::kRunning) {
            return ErrCode::kGood;
        }
        setStateSafe(State::kStarting);
        ErrCode c = startImpl(cppString(configStr, strLen), cppString(path));
        if (isGood(c)) {
            setStateSafe(State::kRunning);
        }
        return c;
    }
    const char* getCaps_i(CapsPort p) const final
    {
        std::unique_lock<ImplMutex> lock(_mutex);
        auto i = _caps_txt.find(p);
        if (i != _caps_txt.cend()) {
            return i->second.c_str();
        } else {
            return "";
        }
    }
    ErrCode flush_i() override
    {
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState,
            "flush failed since dataloader was not started");
        return flushImpl();
    }

    ErrCode stop_i() override
    {
        State s = getStateSafe();
        if (s == State::kStopped || s == State::kNone) {
            return ErrCode::kGood;
        }
        ErrCode ret = stopImpl();
        _errCB.reset();
        setStateSafe(State::kStopped);
        return ret;
    }

    State state_i() const final { return getStateSafe(); }

protected:
    // cplusplus virtual interface, user need to derive from
    virtual ErrCode startImpl(const std::string& content, const std::string& path) = 0;
    virtual ErrCode stopImpl() = 0;
    virtual ErrCode flushImpl() = 0;

    // internal calls
    void setOutputCaps(const std::string& caps) { return setCaps(CapsPort::kOutput, caps); }

    void setInputCaps(const std::string& caps) { return setCaps(CapsPort::kInput, caps); }

    void emitError(ErrCode code, const std::string& msg)
    {
        if (_errCB) {
            _errCB(code, msg.c_str());
        }
    }

    State getStateSafe() const
    {
        std::unique_lock<ImplMutex> lock(_mutex);
        return _state;
    }

    void setStateSafe(State flag)
    {
        std::unique_lock<ImplMutex> lock(_mutex);
        _state = flag;
    }

protected:
    ImplMutex& mutex() const { return _mutex; }

private:
    void setCaps(CapsPort p, const std::string& caps)
    {
        std::unique_lock<ImplMutex> lock(_mutex);
        _caps_txt[p] = caps;
    }

    GuardCB<abiErrorCB> _errCB;
    std::unordered_map<CapsPort, std::string> _caps_txt{
        {CapsPort::kInput, ""}, {CapsPort::kOutput, ""}};
    GuardDataT<void> _userData;

    mutable ImplMutex _mutex;
    State _state = State::kNone;
};

}}  // namespace ds3d::impl

#endif  // _DS3D_COMMON_IMPL_BASE_DATA_PROCESS_H
