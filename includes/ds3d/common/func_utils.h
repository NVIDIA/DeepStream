/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef _DS3D_COMMON_FUNC_UTILS__H
#define _DS3D_COMMON_FUNC_UTILS__H

#include <ds3d/common/abi_frame.h>
#include <ds3d/common/common.h>
#include <ds3d/common/idatatype.h>

#include <algorithm>

namespace ds3d {

using LockMutex = std::unique_lock<std::mutex>;

inline bool
isGood(ErrCode c)
{
    return c == ErrCode::kGood;
}

inline bool
isNotBad(ErrCode c)
{
    return c >= ErrCode::kGood;
}

inline bool
isCpuMem(MemType t)
{
    return t == MemType::kCpu || t == MemType::kCpuPinned;
}

inline bool
isGpuMem(MemType t)
{
    return t == MemType::kGpuCuda;
}

inline const char*
ErrCodeStr(ErrCode code)
{
    static const std::unordered_map<ErrCode, const char*> kCodeTable = {
#define __DS3D_ERR_STR_DEF(code) {ErrCode::code, #code}
        __DS3D_ERR_STR_DEF(kGood),         __DS3D_ERR_STR_DEF(kByPass),
        __DS3D_ERR_STR_DEF(kLoadLib),      __DS3D_ERR_STR_DEF(kMem),
        __DS3D_ERR_STR_DEF(kParam),        __DS3D_ERR_STR_DEF(kNotFound),
        __DS3D_ERR_STR_DEF(kTimeOut),      __DS3D_ERR_STR_DEF(kTypeId),
        __DS3D_ERR_STR_DEF(kNvDsMeta),     __DS3D_ERR_STR_DEF(kUnsupported),
        __DS3D_ERR_STR_DEF(kUnknown),      __DS3D_ERR_STR_DEF(kConfig),
        __DS3D_ERR_STR_DEF(kRealSense),    __DS3D_ERR_STR_DEF(kNullPtr),
        __DS3D_ERR_STR_DEF(kOutOfRange),   __DS3D_ERR_STR_DEF(kGst),
        __DS3D_ERR_STR_DEF(kState),        __DS3D_ERR_STR_DEF(kGL),
        __DS3D_ERR_STR_DEF(kLockWakeup),   __DS3D_ERR_STR_DEF(kCuda),
        __DS3D_ERR_STR_DEF(kIncompatible),
#undef __DS3D_ERR_STR_DEF
    };
    auto iter = kCodeTable.find(code);
    if (iter == kCodeTable.cend()) {
        return "undefined";
    }
    return iter->second;
};

inline void
throwError(ErrCode code, const std::string& msg)
{
    if (isGood(code)) {
        return;
    }
    throw Exception(code, msg);
}

template <class F, typename... Args>
inline ErrCode
CatchError(F f, Args... args)
{
    ErrCode code = ErrCode::kGood;
    DS3D_TRY { code = f(std::forward<Args>(args)...); }
    DS3D_CATCH_ERROR(Exception, e.code(), "Catch ds3d error")
    DS3D_CATCH_ERROR(std::exception, ErrCode::kUnknown, "Catch std exception")
    DS3D_CATCH_ANY(ErrCode::kUnknown, "Catch unknown error")
    return code;
}

inline ErrCode
CatchVoidCall(std::function<void()> f)
{
    DS3D_TRY { f(); }
    DS3D_CATCH_ERROR(Exception, e.code(), "Catch ds3d error")
    DS3D_CATCH_ERROR(std::exception, ErrCode::kUnknown, "Catch std exception")
    DS3D_CATCH_ANY(ErrCode::kUnknown, "Catch unknown error")
    return ErrCode::kGood;
}

inline std::string
cppString(const char* str, size_t len = 0)
{
    if (str) {
        return (len ? std::string(str, len) : std::string(str));
    }
    return std::string("");
}

template <typename T>
inline uint32_t
bytesPerPixel(FrameType f)
{
    switch (f) {
    case FrameType::kDepth:
        return sizeof(T);
    case FrameType::kColorRGBA:
        DS_ASSERT(sizeof(T) == 1);
        return sizeof(T) * 4;
    case FrameType::kColorRGB:
        DS_ASSERT(sizeof(T) == 1);
        return sizeof(T) * 3;
    case FrameType::kPointXYZ:
        return sizeof(T) * 3;
    case FrameType::kLidarXYZI:
        return sizeof(T) * 4;
    case FrameType::kPointCoordUV:
        return sizeof(T) * 3;
    default:
        LOG_ERROR("bytesPerPixel with unsupported frametype: %d", static_cast<int>(f));
    }
    return 0;
}

inline uint32_t
dataTypeBytes(DataType t)
{
    switch (t) {
#define __DS3D_DATATYPE_BYTES(t) \
    case DataType::t:            \
        return sizeof(NativeData<DataType::t>::type)
        __DS3D_DATATYPE_BYTES(kFp32);
        __DS3D_DATATYPE_BYTES(kFp16);
        __DS3D_DATATYPE_BYTES(kInt8);
        __DS3D_DATATYPE_BYTES(kInt32);
        __DS3D_DATATYPE_BYTES(kInt16);
        __DS3D_DATATYPE_BYTES(kUint8);
        __DS3D_DATATYPE_BYTES(kUint16);
        __DS3D_DATATYPE_BYTES(kUint32);
        __DS3D_DATATYPE_BYTES(kDouble);
        __DS3D_DATATYPE_BYTES(kInt64);
#undef __DS3D_DATATYPE_BYTES
    default:
        assert(t);
        return 0;
    }
}

inline size_t
ShapeSize(const Shape& shape)
{
    if (!shape.numDims) {
        return 0;
    }
    if (std::any_of(shape.d, shape.d + shape.numDims, [](int d) { return d < 0; })) {
        assert(false);
        return 0;
    }
    return (size_t)std::accumulate(
        shape.d, shape.d + shape.numDims, 1, [](int s, int i) { return s * i; });
}

inline bool
operator==(const Shape& a, const Shape& b)
{
    if (a.numDims != b.numDims) {
        return false;
    }
    for (uint32_t i = 0; i < a.numDims; ++i) {
        if (a.d[i] != b.d[i]) {
            return false;
        }
    }
    return true;
}

inline bool
operator!=(const Shape& a, const Shape& b)
{
    return !(a == b);
}

inline bool
readFile(const std::string& path, std::string& context)
{
    std::ifstream fileIn(path, std::ios::in | std::ios::binary);
    DS3D_FAILED_RETURN(fileIn, false, "open file %s failed", path.c_str());

    fileIn.seekg(0, std::ios::end);
    size_t fileSize = fileIn.tellg();
    context.resize(fileSize, 0);
    fileIn.seekg(0, std::ios::beg);
    fileIn.read(&context[0], fileSize);
    fileIn.close();
    return true;
}

inline bool NvDs3dEnableDebug()
{
    static bool debug = std::getenv("DS3D_ENABLE_DEBUG") ? true : false;
    return debug;
}

template <typename Data>
void
array2Vec3(Data* from, vec3<Data>& to)
{
    for (int i = 0; i < 3; ++i) {
        to.data[i] = from[i];
    }
}

inline bool
isNear(float a, float b)
{
    if (fabs(a - b) <= std::numeric_limits<float>::epsilon())
        return true;
    return false;
}

inline TransformMatrix
getIntrinsicMat(IntrinsicsParam& param)
{
    TransformMatrix mat = {{
        {{param.fx, 0.0f, 0.0f, 0.0f}},
        {{0.0f, param.fy, 0.0f, 0.0f}},
        {{param.centerX, param.centerY, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f, 1.0f}},
    }};
    return mat;
}

}  // namespace ds3d

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API ds3d::abiRefDataMap* NvDs3d_CreateDataHashMap();
DS3D_EXTERN_C_END

#endif  // _DS3D_COMMON_FUNC_UTILS__H