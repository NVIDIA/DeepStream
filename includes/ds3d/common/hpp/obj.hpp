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


#ifndef _DS3D_COMMON_HPP_OBJ_HPP
#define _DS3D_COMMON_HPP_OBJ_HPP

#include <ds3d/common/abi_obj.h>
#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#define DS3D_REF_COPY_DESTROY_IMPL(clss)  \
    void destroy() final { delete this; } \
    ~clss() override = default;           \
    abiRefObj* refCopy_i() const final { return new clss(*this); }

namespace ds3d {

template <class T>
using ShrdPtr = std::shared_ptr<T>;
template <class T>
using UniqPtr = std::unique_ptr<T, std::function<void(T*)>>;
template <class T>
using Ptr = ShrdPtr<T>;

template <class T>
void
DeleteTFunc(T* t)
{
    if (t) {
        delete t;
    }
}

template <typename T>
using _EnableIfValidIdType =
    typename std::enable_if_t<TpId<std::remove_cv_t<T>>::__typeid() >= 0, bool>;

template <typename T, typename... Args>
using _IsConstructible = typename std::is_constructible<T, Args...>::value;

template <typename T, typename... Args>
using _EnableIfConstructible =
    typename std::enable_if<std::is_constructible<T, Args...>::value, bool>::type;

template <class From, class To>
using _Convertible = std::enable_if_t<
    std::is_convertible<From, To>::value ||
        std::is_constructible<std::remove_reference<To>, std::remove_reference<From>>::value,
    bool>;
template <class Base, class Derived>
using _EnableIfBaseOf = std::enable_if_t<std::is_base_of<Base, Derived>::value, bool>;

template <class From, class To>
using _PtrConvertible = std::enable_if_t<
    std::is_convertible<From*, To*>::value ||
        std::is_constructible<std::remove_reference<To*>, std::remove_reference<From*>>::value ||
        std::is_base_of<From, To>::value,
    bool>;

template <class From, class To, class convertType>
To* pointerCast(From* a);

template <class From, class To, _Convertible<From*, To*> = true>
To*
pointerCast(From* a)
{
    return static_cast<To*>(a);
}

template <class From, class To, _EnableIfBaseOf<From, To> = true>
To*
pointerCast(From* a)
{
    return dynamic_cast<To*>(a);
}

template <class From, class To, std::enable_if_t<std::is_void<From>::value, bool> = true>
To*
pointerCast(From* a)
{
    return static_cast<To*>(a);
}

template <class From, class To>
class abiRefCast : public abiRefT<To> {
    abiRefT<From>* _from = nullptr;
    abiRefCast() = default;

public:
    abiRefCast(const abiRefT<From>& from) { _from = from.refCopy(); }
    abiRefCast(abiRefT<From>* from, bool takeOwner)
    {
        if (takeOwner) {
            _from = from;
        } else {
            _from = from->refCopy();
        }
    }
    void reset()
    {
        if (_from) {
            _from->destroy();
            _from = nullptr;
        }
    }
    ~abiRefCast() { reset(); }
    void destroy() final
    {
        reset();
        delete this;
    }
    To* data() const final { return pointerCast<From, To>(_from->data()); }
    abiRefObj* refCopy_i() const final
    {
        auto newptr = new abiRefCast();
        newptr->_from = _from->refCopy();
        return newptr;
    }
    DS3D_DISABLE_CLASS_COPY(abiRefCast);
};

template <class Tp>
class SharedRefObj : public abiRefT<Tp> {
    ShrdPtr<Tp> _ptr;

public:
    SharedRefObj(ShrdPtr<Tp>&& v) : _ptr(std::move(v)) { DS_ASSERT(_ptr); }
    template <class Ty, _PtrConvertible<Ty, Tp> = true>
    SharedRefObj(ShrdPtr<Ty>&& v)
    {
        DS_ASSERT(v);
        Tp* d = pointerCast<Ty, Tp>(v.get());
        _ptr = ShrdPtr<Tp>(std::move(v), d);
    }
    // template <class DelF>
    // SharedRefObj(Tp* v, DelF f) : _ptr(v, std::move(f))
    SharedRefObj(Tp* v, std::function<void(Tp*)> f) : _ptr(v, (f ? std::move(f) : [](Tp*) {})) {}

    Tp* data() const final { return _ptr.get(); }
    DS3D_REF_COPY_DESTROY_IMPL(SharedRefObj)
};

template <class Tp>
abiRefT<Tp>*
NewAbiRef(Tp* rawAbiObj)
{
    return new SharedRefObj<Tp>(rawAbiObj, &DeleteTFunc<Tp>);
}

using RefDataMapObj = SharedRefObj<abiRefDataMap>;

template <class From, class To = From>
inline SharedRefObj<To>*
PtrToAbiRef(ShrdPtr<From>&& p)
{
    return new SharedRefObj<To>(std::move(p));
}

template <class From, class To = From>
inline ShrdPtr<To>
AbiRefToPtr(const abiRefT<From>& p)
{
    if (!p.data())
        return nullptr;
    abiRefT<From>* copy = p.refCopy();
    DS_ASSERT(copy);
    return ShrdPtr<To>(pointerCast<From, To>(p.data()), [copy](To*) {
        if (copy) {
            copy->destroy();
        }
    });
}

/* Examples:
 * auto f = [](ErrCode c, const char* msg){
 *     printf("error: %s detected, msg: %s", ErrCodeStr(c), msg);
 * }
 * // decltype(f) should be same as CBObjT<ErrCode, const char*>::cbType
 * CBObjT<ErrCode, const char*> cbObj(std::move(f));
 */
template <typename... Args>
class CBObjT : public abiCallBackT<Args...> {
public:
    using cbType = std::function<void(Args...)>;
    bool isValid() const { return bool(_f); }
    CBObjT(cbType&& f) { _f = std::move(f); }
    void notify(Args... args) final
    {
        if (_f)
            _f(args...);
    }
    DS3D_REF_COPY_DESTROY_IMPL(CBObjT)
private:
    cbType _f;
};

/**
 * @brief Guard to wrapper all abiRefObj& data.
 *        It is safe to use the Guard to access any kind of abiRefObj data.
 *
 */
template <typename ref, _EnableIfBaseOf<abiRefObj, ref> = true>
class GuardRef {
    ref* _abiRef = nullptr;

public:
    GuardRef() = default;
    // make a safe copy on data reference abiref
    GuardRef(const ref& abiref)
    {
        DS_ASSERT((std::is_base_of<abiRefObj, ref>::value));
        _abiRef = static_cast<ref*>(abiref.refCopy_i());
    }
    // take owership of the data reference abiref pointer
    GuardRef(ref* abiref, bool takeowner)
    {
        DS_ASSERT(std::is_base_of<abiRefObj, ref>::value);
        if (takeowner) {
            _abiRef = abiref;
        } else {
            _abiRef = abiref->refCopy();
        }
    }
    // move constructor
    GuardRef(GuardRef&& o)
    {
        _abiRef = o._abiRef;
        o._abiRef = nullptr;
    }
    // copy from another GuardRef
    GuardRef(const GuardRef& o)
    {
        _abiRef = o._abiRef ? static_cast<ref*>(o._abiRef->refCopy_i()) : nullptr;
        //DS_ASSERT(_abiRef);
    }
    // copy from another GuardRef
    GuardRef& operator=(const GuardRef& o)
    {
        _abiRef = o._abiRef ? static_cast<ref*>(o._abiRef->refCopy_i()) : nullptr;
        return *this;
    }
    // release this abiRef. user need to take owership of the released abiRef and destroy after use.
    ref* release()
    {
        ref* res = _abiRef;
        _abiRef = nullptr;
        return res;
    }
    // reset and destroy the abiRef
    void reset(ref* abiref = nullptr)
    {
        if (_abiRef) {
            _abiRef->destroy();
        }
        _abiRef = abiref;
    }
    // destructor
    virtual ~GuardRef() { reset(nullptr); }
    ref* abiRef() const { return _abiRef; }
};

/*
 * examples to get GuardCB from abiCallBackT:
 *   using abiErrorCB = abiCallBackT<ErrCode, const char*>;
 *   const abiErrorCB& cb = ...;
 *   GuardCB<abiErrorCB> guardCB(cb);
 *   guardCB(ErrCode::kGood, "sample");
 * examples to get GuardCB from std::function
 *   auto errFn = [](ErrCode, const char*) {...};
 *   GuardCB<abiErrorCB> guardCB(errFn);
 *   guardCB(ErrCode::kGood, "sample");
 */

template <typename abiCB>
class GuardCB : public GuardRef<abiCB> {
    using _GuardCBBase = GuardRef<abiCB>;

public:
    GuardCB() = default;
    GuardCB(const abiCB& cb) : _GuardCBBase(cb) {}
    GuardCB(abiCB* cb, bool takeowner = true) : _GuardCBBase(cb, takeowner) {}
    GuardCB(nullptr_t) {}
    GuardCB(const GuardCB& o) : _GuardCBBase(o) {}
    ~GuardCB() override = default;

    template <typename... Args, typename F>
    void setFn(F f)
    {
        auto obj = std::make_unique<CBObjT<Args...>>(std::move(f));
        if (obj && obj->isValid()) {
            this->reset(obj.release());
        } else {
            this->reset();
        }
    }

    // disable operator ->, redirect it to callback abi object.
    // abiCB* operator->() const { return this->abiRef(); }
    // check whether it is intialized
    operator bool() { return this->abiRef(); }

    template <typename... Args>
    void operator()(Args&&... args) const
    {
        DS_ASSERT(this->abiRef());
        this->abiRef()->notify(std::forward<Args>(args)...);
    }
};

/*
 * examples:
 * const abiRefT<abiDataMap>& a_data_map_ref = ...;
 * GuardDataT<abiDataMap> guard(a_data_map_ref);
 * guard->bytes(); // -> point to abiDataMap interface.
 * guard.abiRef() // get abiDataMap ref. abiRefT<abiDataMap>
 * GuardDataT<abiDataMap> another_guard = guard; // copy into another guard.
 * Ptr<abiDataMap> data_ptr = guard; // get a abiDataMap safe pointer, which could be copied into
 * everywhere
 */
template <class Tp>
class GuardDataT : public GuardRef<abiRefT<Tp>> {
public:
    using abiRefType = abiRefT<Tp>;
    GuardDataT() = default;
    GuardDataT(nullptr_t) {}
    GuardDataT(const abiRefT<Tp>& rf) : GuardRef<abiRefT<Tp>>(rf) {}
    GuardDataT(abiRefT<Tp>* refPtr, bool takeOwner) : GuardRef<abiRefT<Tp>>(refPtr, takeOwner) {}
    GuardDataT(const GuardDataT& o):  GuardRef<abiRefT<Tp>>(o) {}
    ~GuardDataT() override = default;

    // convert from a derived reference
    template <class Ty, _PtrConvertible<Ty, Tp> = true>
    GuardDataT(const abiRefT<Ty>& derived)
    {
        this->reset(new abiRefCast<Ty, Tp>(derived));
    }

    // convert from a derived ref pointer
    template <class Ty, _PtrConvertible<Ty, Tp> = true>
    GuardDataT(abiRefT<Ty>* abiref, bool takeOwner)
    {
        this->reset(new abiRefCast<Ty, Tp>(abiref, takeOwner));
    }

    template<class GuardTy>
    GuardTy cast() {
        return GuardTy(this->abiRef(), false);
    }

    // return pointer of the actual data.
    Tp* ptr() const
    {
        if (this->abiRef())
            return this->abiRef()->data();
        return nullptr;
    }
    // override operator ->, redirect it to abiData object.
    Tp* operator->() const { return ptr(); }
    // check whether it is intialized
    operator bool() const { return ptr(); }
    // convert it into shared_ptr<Tp> for externer safe use
    operator ShrdPtr<Tp>()
    {
        if (this->abiRef()) {
            return AbiRefToPtr(*(this->abiRef()));
        }
        return nullptr;
    }
};

}  // namespace ds3d

#endif  // _DS3D_COMMON_HPP_OBJ_HPP