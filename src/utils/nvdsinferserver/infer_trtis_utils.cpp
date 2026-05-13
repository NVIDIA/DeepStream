/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "infer_trtis_utils.h"

namespace nvdsinferserver {

InferDataType
DataTypeFromTriton(TRITONSERVER_DataType type)
{
    const static std::unordered_map<TRITONSERVER_DataType, InferDataType>
        sFrTriton{
            {TRITONSERVER_TYPE_BOOL, InferDataType::kBool},
            {TRITONSERVER_TYPE_UINT8, InferDataType::kUint8},
            {TRITONSERVER_TYPE_UINT16, InferDataType::kUint16},
            {TRITONSERVER_TYPE_UINT32, InferDataType::kUint32},
            {TRITONSERVER_TYPE_UINT64, InferDataType::kUint64},
            {TRITONSERVER_TYPE_INT8, InferDataType::kInt8},
            {TRITONSERVER_TYPE_INT16, InferDataType::kInt16},
            {TRITONSERVER_TYPE_INT32, InferDataType::kInt32},
            {TRITONSERVER_TYPE_INT64, InferDataType::kInt64},
            {TRITONSERVER_TYPE_FP16, InferDataType::kFp16},
            {TRITONSERVER_TYPE_FP32, InferDataType::kFp32},
            {TRITONSERVER_TYPE_FP64, InferDataType::kFp64},
            {TRITONSERVER_TYPE_BYTES, InferDataType::kString},
            {TRITONSERVER_TYPE_INVALID, InferDataType::kNone},
        };
    auto const i = sFrTriton.find(type);
    if (i == sFrTriton.end()) {
        InferError("unsupported data type:%d from Triton", type);
        return InferDataType::kNone;
    }
    return i->second;
}

InferDataType
DataTypeFromTritonPb(ni::DataType type)
{
    const static std::unordered_map<ni::DataType, InferDataType> sFrTriton{
        {ni::TYPE_BOOL, InferDataType::kBool},
        {ni::TYPE_UINT8, InferDataType::kUint8},
        {ni::TYPE_UINT16, InferDataType::kUint16},
        {ni::TYPE_UINT32, InferDataType::kUint32},
        {ni::TYPE_UINT64, InferDataType::kUint64},
        {ni::TYPE_INT8, InferDataType::kInt8},
        {ni::TYPE_INT16, InferDataType::kInt16},
        {ni::TYPE_INT32, InferDataType::kInt32},
        {ni::TYPE_INT64, InferDataType::kInt64},
        {ni::TYPE_FP16, InferDataType::kFp16},
        {ni::TYPE_FP32, InferDataType::kFp32},
        {ni::TYPE_FP64, InferDataType::kFp64},
        {ni::TYPE_STRING, InferDataType::kString},
        {ni::TYPE_INVALID, InferDataType::kNone},
    };
    auto const i = sFrTriton.find(type);
    if (i == sFrTriton.end()) {
        InferError(
            "unsupported data type:%s from Triton proto",
            safeStr(ni::DataType_Name(type)));
        return InferDataType::kNone;
    }
    return i->second;
}


TRITONSERVER_DataType
DataTypeToTriton(InferDataType type)
{
    const static std::unordered_map<InferDataType, TRITONSERVER_DataType>
        sFrTriton{
            {InferDataType::kBool, TRITONSERVER_TYPE_BOOL},
            {InferDataType::kUint8, TRITONSERVER_TYPE_UINT8},
            {InferDataType::kUint16, TRITONSERVER_TYPE_UINT16},
            {InferDataType::kUint32, TRITONSERVER_TYPE_UINT32},
            {InferDataType::kUint64, TRITONSERVER_TYPE_UINT64},
            {InferDataType::kInt8, TRITONSERVER_TYPE_INT8},
            {InferDataType::kInt16, TRITONSERVER_TYPE_INT16},
            {InferDataType::kInt32, TRITONSERVER_TYPE_INT32},
            {InferDataType::kInt64, TRITONSERVER_TYPE_INT64},
            {InferDataType::kFp16, TRITONSERVER_TYPE_FP16},
            {InferDataType::kFp32, TRITONSERVER_TYPE_FP32},
            {InferDataType::kFp64, TRITONSERVER_TYPE_FP64},
            {InferDataType::kString, TRITONSERVER_TYPE_BYTES},
            {InferDataType::kNone, TRITONSERVER_TYPE_INVALID},
        };
    auto const i = sFrTriton.find(type);
    if (i == sFrTriton.end()) {
        InferError("unsupported data type:%d to Triton", type);
        return TRITONSERVER_TYPE_INVALID;
    }
    return i->second;
}

InferTensorOrder
TensorOrderFromTritonPb(ni::ModelInput::Format order)
{
    switch (order) {
    case ni::ModelInput::FORMAT_NCHW:
        return InferTensorOrder::kLinear;
    case ni::ModelInput::FORMAT_NHWC:
        return InferTensorOrder::kNHWC;
    case ni::ModelInput::FORMAT_NONE:
        return InferTensorOrder::kNone;
    default:
        InferError(
            "unsupported Triton tensor order:%s", safeStr(ni::ModelInput::Format_Name(order)));
        return InferTensorOrder::kNone;
    }
}

InferTensorOrder
TensorOrderFromTritonMeta(const std::string& format)
{
    const static std::unordered_map<std::string, InferTensorOrder> sFrTriton{
        {"FORMAT_NCHW", InferTensorOrder::kLinear},
        {"FORMAT_NHWC", InferTensorOrder::kNHWC},
        {"FORMAT_NONE", InferTensorOrder::kNone},
    };
    auto const i = sFrTriton.find(format);
    if (i == sFrTriton.end()) {
        InferError("unsupported Triton tensor order:%s", safeStr(format));
        return InferTensorOrder::kNone;
    }
    return i->second;
}

TRITONSERVER_MemoryType
MemTypeToTriton(InferMemType type)
{
    switch (type) {
    case InferMemType::kGpuCuda:
        return TRITONSERVER_MEMORY_GPU;
    case InferMemType::kCpu:
        return TRITONSERVER_MEMORY_CPU;
    case InferMemType::kCpuCuda:
        return TRITONSERVER_MEMORY_CPU_PINNED;
    default:
        assert(false);
        InferError("failed to convert infer-mem-type: %d to TRT-IS memory type",
            static_cast<int>(type));
        return (TRITONSERVER_MemoryType)-1;
    }
}

InferMemType
MemTypeFromTriton(TRITONSERVER_MemoryType type)
{
    switch (type) {
    case TRITONSERVER_MEMORY_GPU:
        return InferMemType::kGpuCuda;
    case TRITONSERVER_MEMORY_CPU:
        return InferMemType::kCpu;
    case TRITONSERVER_MEMORY_CPU_PINNED:
        return InferMemType::kCpuCuda;
    default:
        assert(false);
        InferError("failed to convert TRT-IS-mem-type: %d to ds mem-type",
            static_cast<int>(type));
        return InferMemType::kNone;
    }
}

const char* TritonControlModeToStr(int32_t mode)
{
    switch ((TRITONSERVER_ModelControlMode)mode) {
    case TRITONSERVER_MODEL_CONTROL_NONE:
        return "none";
    case TRITONSERVER_MODEL_CONTROL_EXPLICIT:
        return "explicit";
    case TRITONSERVER_MODEL_CONTROL_POLL:
        return "poll";
    default:
        InferError("unknow TRITONSERVER_ModelControlMode:%d", (int32_t)mode);
        break;
    }
    return "unknown";
}

} // namespace nvdsinferserver
