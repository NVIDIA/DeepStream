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


#ifndef _DS3D_COMMON_TYPE_TRAIT__H
#define _DS3D_COMMON_TYPE_TRAIT__H
#include <ds3d/common/common.h>
#include <ds3d/common/typeid.h>
#include <ds3d/common/idatatype.h>

#if ENABLE_HALF
#include <half.hpp>
#endif

namespace ds3d {

template <typename Tp>
struct TpId {
    static constexpr TIdType __typeid() { return Tp::__typeid(); }
};

template <TIdType v>
struct __TypeID {
    static constexpr TIdType __typeid() { return v; }
};

template <DataType v>
struct __DataTypeVal {
    static constexpr DataType _data_type_value() { return v; }
};

// define TypeId
template <>
struct TpId<uint32_t> : __TypeID<DS3D_TYPEID_UINT32_T>, __DataTypeVal<DataType::kUint32> {
};
template <>
struct TpId<int32_t> : __TypeID<DS3D_TYPEID_INT32_T>, __DataTypeVal<DataType::kInt32> {
};
template <>
struct TpId<float> : __TypeID<DS3D_TYPEID_FLOAT>, __DataTypeVal<DataType::kFp32> {
};
template <>
struct TpId<int8_t> : __TypeID<DS3D_TYPEID_INT8_T>, __DataTypeVal<DataType::kInt8> {
};
template <>
struct TpId<uint8_t> : __TypeID<DS3D_TYPEID_UINT8_T>, __DataTypeVal<DataType::kUint8> {
};
template <>
struct TpId<uint16_t> : __TypeID<DS3D_TYPEID_UINT16_T>, __DataTypeVal<DataType::kUint16> {
};
template <>
struct TpId<int16_t> : __TypeID<DS3D_TYPEID_INT16_T>, __DataTypeVal<DataType::kInt16> {
};

template <>
struct TpId<double> : __TypeID<DS3D_TYPEID_DOUBLE>, __DataTypeVal<DataType::kDouble> {
};

template <>
struct TpId<bool> : __TypeID<DS3D_TYPEID_BOOL>, __DataTypeVal<DataType::kDouble> {
};

template <>
struct TpId<int64_t> : __TypeID<DS3D_TYPEID_INT64_T>, __DataTypeVal<DataType::kInt64> {
};

template <typename TP>
struct __DataTypeTrait {
    static constexpr DataType _data_type() { return TpId<TP>::_data_type_value(); }
};

template <DataType t>
struct NativeData;

template <>
struct NativeData<DataType::kFp32> {
    using type = float;
};

template <>
struct NativeData<DataType::kFp16> {
#if ENABLE_HALF
    using type = half_float::half;
#else
    using type = int16_t;
#endif
};

template <>
struct NativeData<DataType::kInt8> {
    using type = int8_t;
};

template <>
struct NativeData<DataType::kInt32> {
    using type = int32_t;
};
template <>
struct NativeData<DataType::kInt16> {
    using type = int16_t;
};
template <>
struct NativeData<DataType::kUint8> {
    using type = uint8_t;
};
template <>
struct NativeData<DataType::kUint16> {
    using type = uint16_t;
};
template <>
struct NativeData<DataType::kUint32> {
    using type = uint32_t;
};
template <>
struct NativeData<DataType::kDouble> {
    using type = double;
};

template <>
struct NativeData<DataType::kInt64> {
    using type = int64_t;
};

}  // namespace ds3d

#endif  // _DS3D_COMMON_TYPE_TRAIT__H