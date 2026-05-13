/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef _DS3D_COMMON_HPP_DATAMAP_HPP
#define _DS3D_COMMON_HPP_DATAMAP_HPP

#include <ds3d/common/func_utils.h>

#include <ds3d/common/hpp/obj.hpp>

namespace ds3d {

class GuardDataMap : public GuardDataT<abiDataMap> {
    using _Base = GuardDataT<abiDataMap>;

public:
    using KeyName = std::string;
    GuardDataMap() = default;

    template <typename... Args /*, _EnableIfConstructible<_Base, Args&&...> = true*/>
    GuardDataMap(Args&&... args) : _Base(std::forward<Args>(args)...)
    {
    }

    ~GuardDataMap() = default;

    bool hasData(const KeyName& name)
    {
        DS_ASSERT(ptr());
        return ptr()->has_i(name.c_str());
    }

    template <class T, _EnableIfValidIdType<T> = true>
    inline ErrCode setData(const KeyName& name, const T& value);  // copy T

    template <class T>
    inline ErrCode setGuardData(const KeyName& name, const GuardDataT<T>& value);

    template <class T>
    inline ErrCode setRefData(const KeyName& name, const abiRefT<T>& value);

    template <class T>
    inline ErrCode setPtrData(const KeyName& name, ShrdPtr<T> value);

    template <class T>
    inline ErrCode setPtrData(const KeyName& name, UniqPtr<T> value)
    {
        return this->setPtrData(ShrdPtr<T>(std::move(value)));
    }

    template <class T>
    inline ErrCode getPtrData(const KeyName& name, ShrdPtr<T>& value);

    template <class T>
    inline ErrCode getRefData(const KeyName& name, abiRefT<T>*& value);

    template <class T>
    inline ErrCode getGuardData(const KeyName& name, GuardDataT<T>& value);

    template <class T, _EnableIfValidIdType<T> = true>
    inline ErrCode getData(const KeyName& name, T& value);

    inline ErrCode removeData(const KeyName& name)
    {
        DS_ASSERT(ptr());
        return ptr()->removeBuf_i(name.c_str());
    }
    inline ErrCode clear()
    {
        DS_ASSERT(ptr());
        return ptr()->clear_i();
    }
    inline ErrCode copy(
        GuardDataMap input, DataMapPolicy policy = DataMapPolicy::kCopyPolicyNone, char* policyData = nullptr)
    {
        DS_ASSERT(ptr());
        DS_ASSERT(input.ptr());
        return ptr()->copy_i(input.ptr(), policy, policyData);
    }
    inline ErrCode copy(
        GuardDataMap input, const KeyName key, DataMapPolicy policy = DataMapPolicy::kCopyPolicyNone,
        char* policyData = nullptr)
    {
        DS_ASSERT(ptr());
        DS_ASSERT(input.ptr());
        return ptr()->copy_i(input.ptr(), key.empty() ? nullptr : key.c_str(), policy, policyData);
    }

    inline int32_t getSize()
    {
        DS_ASSERT(ptr());
        return ptr()->getSize_i();
    }

    void printDebug() const
    {
        DS_ASSERT(ptr());
        ptr()->printDebug_i();
    }
};

template <class T>
ErrCode
GuardDataMap::setGuardData(const GuardDataMap::KeyName& name, const GuardDataT<T>& value)
{
    if (!value.abiRef()) {
        return ErrCode::kNullPtr;
    }
    return this->setRefData(name, *value.abiRef());
}

template <class T>
ErrCode
GuardDataMap::setRefData(const GuardDataMap::KeyName& name, const abiRefT<T>& value)
{
    using t = std::remove_const_t<T>;
    TIdType tyid = TpId<t>::__typeid();
    abiRefAny* u = new abiRefCast<T, void>(value);
    DS_ASSERT(u && u->data());
    DS_ASSERT(ptr());
    ErrCode code = ptr()->setBuf_i(name.c_str(), tyid, u);
    if (!isGood(code)) {
        u->destroy();
    }
    return code;
}

template <class T, _EnableIfValidIdType<T>>
ErrCode
GuardDataMap::setData(const GuardDataMap::KeyName& name, const T& value)
{
    using t = std::remove_const_t<std::remove_reference_t<T>>;
    // TIdType tyid = TpId<t>::__typeid();
    ShrdPtr<t> data(new t(value));
    return this->setPtrData(name, std::move(data));
}

template <class T>
ErrCode
GuardDataMap::setPtrData(const GuardDataMap::KeyName& name, ShrdPtr<T> value)
{
    using t = std::remove_const_t<T>;
    TIdType tyid = TpId<t>::__typeid();
    abiRefAny* u = PtrToAbiRef<T, void>(std::move(value));
    DS_ASSERT(u && u->data());
    DS_ASSERT(ptr());
    ErrCode code = ptr()->setBuf_i(name.c_str(), tyid, u);
    if (!isGood(code)) {
        u->destroy();
    }
    return code;
}

template <class T>
ErrCode
GuardDataMap::getGuardData(const GuardDataMap::KeyName& name, GuardDataT<T>& guardData)
{
    abiRefT<T>* refData = nullptr;
    ErrCode code = getRefData(name, refData);
    guardData.reset(refData);
    return code;
}

template <class T>
ErrCode
GuardDataMap::getRefData(const GuardDataMap::KeyName& name, abiRefT<T>*& value)
{
    using t = std::remove_const_t<T>;
    TIdType tyid = TpId<t>::__typeid();
    const abiRefAny* ud = nullptr;
    DS_ASSERT(ptr());
    ErrCode code = ptr()->getBuf_i(name.c_str(), tyid, ud);
    if (!isGood(code)) {
        return code;
    }
    abiRefT<T>* u = new abiRefCast<void, T>(const_cast<abiRefAny*>(ud), false);
    DS_ASSERT(u && u->data());
    value = u;
    DS_ASSERT(value);
    return code;
}

template <class T>
ErrCode
GuardDataMap::getPtrData(const GuardDataMap::KeyName& name, ShrdPtr<T>& value)
{
    using t = std::remove_const_t<T>;
    TIdType tyid = TpId<t>::__typeid();
    const abiRefAny* ud = nullptr;
    DS_ASSERT(ptr());
    ErrCode code = ptr()->getBuf_i(name.c_str(), tyid, ud);
    if (!isGood(code)) {
        return code;
    }
    DS_ASSERT(ud && ud->data());
    value = AbiRefToPtr<void, T>(*ud);
    DS_ASSERT(value);
    return code;
}

template <class T, _EnableIfValidIdType<T>>
ErrCode
GuardDataMap::getData(const GuardDataMap::KeyName& name, T& value)
{
    using t = std::remove_const_t<T>;
    TIdType tyid = TpId<t>::__typeid();
    const abiRefAny* ud = nullptr;
    DS_ASSERT(ptr());
    ErrCode code = ptr()->getBuf_i(name.c_str(), tyid, ud);
    if (!isGood(code)) {
        return code;
    }
    DS_ASSERT(ud && ud->data());
    value = *(static_cast<T*>(ud->data()));
    return code;
}

}  // namespace ds3d

#endif  // _DS3D_COMMON_HPP_DATAMAP_HPP
