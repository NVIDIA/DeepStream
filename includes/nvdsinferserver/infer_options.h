/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSINFERSERVER_OPTIONS_H__
#define __NVDSINFERSERVER_OPTIONS_H__

#include <infer_datatypes.h>

#include <optional>
#include <string>
#include <unordered_map>

#ifdef FOR_PRIVATE
#include "infer_common.h"
#include "infer_utils.h"
#else
inline void
dsInferLogPrint__(NvDsInferLogLevel level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
#define safeStr(str) str.c_str()

#endif

namespace nvdsinferserver {

class BufOptions;
using SharedBufOptions = std::shared_ptr<BufOptions>;

class BufOptions : public IOptions {
private:
    struct D {
        struct BasicValue {
            union {
                int64_t vInt64;
                uint64_t vUint64;
                double vDouble;
                bool vBool;
                void* vPtr;
            } value;
            OptionType type = OptionType::oNone;
            std::string vStr;
            template <typename V>
            inline void setV(const V& v, OptionType t)
            {
                *((V*)(void*)&value) = v;
                this->type = t;
            }
        } vHead;
        std::vector<BasicValue> vArray;
    };

public:
    OptionType getType(const std::string& key) const override
    {
        const auto i = m_Fields.find(key);
        return (i == m_Fields.end() ? OptionType::oNone : i->second.vHead.type);
    }
    bool hasValue(const std::string& key) const override
    {
        const auto i = m_Fields.find(key);
        return (i == m_Fields.end() ? false : true);
    }
    uint32_t getCount() const final { return (uint32_t)m_Fields.size(); }
    std::string getKey(uint32_t idx) const final
    {
        assert(idx < m_Fields.size());
        auto i = m_Fields.cbegin();
        std::advance(i, idx);
        return i->first;
    }

private:
    NvDsInferStatus getValuePtr(const std::string& key, OptionType t, void*& ptr) const override
    {
        assert(t != OptionType::oNone && t != OptionType::oArray);
        auto d = getValueD(key, t);
        RETURN_IF_FAILED(
            d, NVDSINFER_INVALID_PARAMS, "failed to get pointer value:%s", safeStr(key));
        if (t == OptionType::oString) {
            ptr = (void*)&(d->vHead.vStr);
        } else {
            ptr = (void*)&(d->vHead.value);
        }
        return NVDSINFER_SUCCESS;
    }

    NvDsInferStatus getArraySize(const std::string& key, uint32_t& size) const override
    {
        auto d = getValueD(key, OptionType::oArray);
        RETURN_IF_FAILED(d, NVDSINFER_INVALID_PARAMS, "failed to get array value:%s", safeStr(key));
        size = d->vArray.size();
        return NVDSINFER_SUCCESS;
    }

    NvDsInferStatus getRawPtrArray(
        const std::string& key, OptionType ot, void** ptrBase, uint32_t size) const override
    {
        auto d = getValueD(key, OptionType::oArray);
        RETURN_IF_FAILED(
            d, NVDSINFER_INVALID_PARAMS, "failed to get pointer array value:%s", safeStr(key));
        assert(size <= d->vArray.size());
        for (uint32_t i = 0; i < size; ++i) {
            auto& each = d->vArray[i];
            assert(each.type != OptionType::oArray && each.type != OptionType::oNone);
            RETURN_IF_FAILED(
                each.type == ot, NVDSINFER_INVALID_PARAMS,
                "query value type:%d doesn't match exact type:%d in array.", (int)ot,
                (int)each.type);
            if (ot == OptionType::oString) {
                ptrBase[i] = (void*)&(each.vStr);
            } else {
                ptrBase[i] = (void*)&(each.value);
            }
        }
        return NVDSINFER_SUCCESS;
    }

    template <typename In>
    struct convertType {
    };

public:
    template <typename T>
    inline void setValue(const std::string& key, const T& v)
    {
        using t = typename convertType<std::remove_const_t<std::remove_reference_t<T>>>::t;
        auto& field = m_Fields[key];
        field.vHead.setV<t>(v, oType<t>::v);
    }

    template <typename T>
    inline void setValueArray(const std::string& key, const std::vector<T>& values)
    {
        if (values.empty()) {
            return;
        }
        using t = typename convertType<std::remove_const_t<std::remove_reference_t<T>>>::t;
        auto& field = m_Fields[key];
        field.vHead.type = OptionType::oArray;
        field.vArray = std::vector<D::BasicValue>(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            auto& data = field.vArray[i];
            data.setV<t>(t(values[i]), oType<t>::v);
        }
    }

private:
    const D* getValueD(const std::string& key, OptionType t) const
    {
        const auto i = m_Fields.find(key);
        if (i == m_Fields.end()) {
            InferError("BufOptions: No option:%s found.", safeStr(key));
            return nullptr;
        }
        if (i->second.vHead.type != t) {
            InferError(
                "BufOptions: get option:%s but type is not matched.",
                safeStr(key));
            return nullptr;
        }
        return &(i->second);
    }

    std::unordered_map<std::string, D> m_Fields;
};

template <>
inline void
BufOptions::D::BasicValue::setV<std::string>(const std::string& v, OptionType t)
{
    this->vStr = v;
    assert(t == OptionType::oString);
    this->type = t;
}

template <typename T>
struct BufOptions::convertType<T*> {
    typedef std::remove_const_t<T>* t;
};
template <>
struct BufOptions::convertType<int64_t> {
    typedef int64_t t;
};
template <>
struct BufOptions::convertType<int32_t> {
    typedef int64_t t;
};
template <>
struct BufOptions::convertType<int16_t> {
    typedef int64_t t;
};
template <>
struct BufOptions::convertType<int8_t> {
    typedef int64_t t;
};
template <>
struct BufOptions::convertType<uint64_t> {
    typedef uint64_t t;
};
template <>
struct BufOptions::convertType<uint32_t> {
    typedef uint64_t t;
};
template <>
struct BufOptions::convertType<uint16_t> {
    typedef uint64_t t;
};
template <>
struct BufOptions::convertType<uint8_t> {
    typedef uint64_t t;
};
template <>
struct BufOptions::convertType<double> {
    typedef double t;
};
template <>
struct BufOptions::convertType<float> {
    typedef double t;
};
template <>
struct BufOptions::convertType<bool> {
    typedef bool t;
};
template <>
struct BufOptions::convertType<std::string> {
    typedef std::string t;
};

template <typename T>  // not supported
struct BufOptions::convertType<std::vector<T>> {
};

template <typename T>  // not supported
struct BufOptions::convertType<std::vector<T*>> {
};

}  // namespace nvdsinferserver

#endif  //__NVDSINFERSERVER_OPTIONS_H__
