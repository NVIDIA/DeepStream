/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "nvdsinfer_func_utils.h"
#include "nvdsinfer_model_builder.h"

namespace nvdsinfer {

DlLibHandle::DlLibHandle(const std::string& path, int mode)
    : m_LibPath(path)
{
    assert(!path.empty());
    m_LibHandle = dlopen(path.c_str(), mode);
    if (!m_LibHandle)
    {
        dsInferError("Could not open lib: %s, error string: %s", path.c_str(),
                dlerror());
    }
}

DlLibHandle::~DlLibHandle()
{
    if (m_LibHandle)
    {
        dlclose(m_LibHandle);
    }
}

nvinfer1::Dims
ds2TrtDims(const NvDsInferDimsCHW& dims)
{
    return nvinfer1::Dims{3, {(int)dims.c, (int)dims.h, (int)dims.w}};
}

nvinfer1::Dims
ds2TrtDims(const NvDsInferDims& dims)
{
    nvinfer1::Dims ret;
    ret.nbDims = dims.numDims;
    std::copy(dims.d, dims.d + dims.numDims, ret.d);
    return ret;
}

NvDsInferDims
trt2DsDims(const nvinfer1::Dims& dims)
{
    NvDsInferDims ret;
    ret.numDims = dims.nbDims;
    int sum = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        ret.d[i] = dims.d[i];
        if (dims.d[i] < 0)
        {
            sum = 0;
        }
        else
        {
            sum *= dims.d[i];
        }
    }
    //Min num elements has to be 1 to support empty tensors
    ret.numElements = (dims.nbDims ? sum : 1);
    return ret;
}

nvinfer1::Dims
CombineDimsBatch(const NvDsInferDims& dims, int batch)
{
    nvinfer1::Dims ret;
    ret.nbDims = dims.numDims + 1;
    /* Set batch size as 0th dim and copy rest of the dims. */
    ret.d[0] = batch;
    std::copy(dims.d, dims.d + dims.numDims, &ret.d[1]);
    return ret;
}

void
SplitFullDims(const nvinfer1::Dims& fullDims, NvDsInferDims& dims, int& batch)
{
    if (!fullDims.nbDims)
    {
        dims.numDims = 0;
        dims.numElements = 0;
        batch = 0;
    }
    else
    {
        /* Use 0th dim as batch size and get rest of the dims. */
        batch = fullDims.d[0];
        if (fullDims.nbDims == 1) {
            /* scaler output*/
            dims.numDims = 1;
            dims.d[0] = 1;
        } else {
            dims.numDims = fullDims.nbDims - 1;
            std::copy(fullDims.d + 1, fullDims.d + fullDims.nbDims, dims.d);
        }

        normalizeDims(dims);
    }
}

void
normalizeDims(NvDsInferDims& dims)
{
    if (hasWildcard(dims) || !dims.numDims)
    {
        dims.numElements = 0;
    }
    else
    {
        dims.numElements = std::accumulate(dims.d, dims.d + dims.numDims, 1,
            [](int s, int i) { return s * i; });
    }
}

std::string
dims2Str(const nvinfer1::Dims& d)
{
    std::stringstream s;
    for (int i = 0; i < d.nbDims - 1; ++i)
    {
        s << d.d[i] << "x";
    }
    s << d.d[d.nbDims - 1];

    return s.str();
}

std::string
dims2Str(const NvDsInferDims& d)
{
    return dims2Str(ds2TrtDims(d));
}

std::string
batchDims2Str(const NvDsInferBatchDims& d)
{
    return dims2Str(CombineDimsBatch(d.dims, d.batchSize));
}

std::string
dataType2Str(const nvinfer1::DataType type)
{
    switch (type)
    {
        case nvinfer1::DataType::kFLOAT:
            return "kFLOAT";
        case nvinfer1::DataType::kHALF:
            return "kHALF";
        case nvinfer1::DataType::kINT8:
            return "kINT8";
        case nvinfer1::DataType::kUINT8:
            return "kUINT8";
        case nvinfer1::DataType::kINT32:
            return "kINT32";
        case nvinfer1::DataType::kINT64:
            return "kINT64";
        default:
            return "UNKNOWN";
    }
}

std::string
dataType2Str(const NvDsInferDataType type)
{
    switch (type)
    {
        case FLOAT:
            return "kFLOAT";
        case HALF:
            return "kHALF";
        case INT8:
            return "kINT8";
        case UINT8:
            return "kUINT8";
        case INT32:
            return "kINT32";
        case INT64:
            return "kINT64";
        default:
            return "UNKNOWN";
    }
}

std::string
networkMode2Str(const NvDsInferNetworkMode type)
{
    switch (type)
    {
        case NvDsInferNetworkMode_FP32:
            return "fp32";
        case NvDsInferNetworkMode_INT8:
            return "int8";
        case NvDsInferNetworkMode_FP16:
            return "fp16";
	case NvDsInferNetworkMode_BEST:
	    return "best";
        default:
            return "UNKNOWN";
    }
}

bool
hasWildcard(const nvinfer1::Dims& dims)
{
    return std::any_of(
        dims.d, dims.d + dims.nbDims, [](int d) { return d == -1; });
}

bool
hasWildcard(const NvDsInferDims& dims)
{
    return std::any_of(
        dims.d, dims.d + dims.numDims, [](int d) { return d == -1; });
}

bool
operator<=(const nvinfer1::Dims& a, const nvinfer1::Dims& b)
{
    assert(a.nbDims == b.nbDims);
    for (int i = 0; i < a.nbDims; ++i)
    {
        if (a.d[i] > b.d[i])
            return false;
    }
    return true;
}

bool
operator>(const nvinfer1::Dims& a, const nvinfer1::Dims& b)
{
    return !(a <= b);
}

bool
operator<=(const NvDsInferDims& a, const NvDsInferDims& b)
{
    assert(a.numDims == b.numDims);
    for (uint32_t i = 0; i < a.numDims; ++i)
    {
        if (a.d[i] > b.d[i])
            return false;
    }
    return true;
}

bool
operator==(const nvinfer1::Dims& a, const nvinfer1::Dims& b)
{
    if (a.nbDims != b.nbDims)
        return false;

    for (int i = 0; i < a.nbDims; ++i)
    {
        if (a.d[i] != b.d[i])
            return false;
    }
    return true;
}

bool
operator!=(const nvinfer1::Dims& a, const nvinfer1::Dims& b)
{
    return !(a == b);
}

bool
operator>(const NvDsInferDims& a, const NvDsInferDims& b)
{
    return !(a <= b);
}

bool
operator==(const NvDsInferDims& a, const NvDsInferDims& b)
{
    if (a.numDims != b.numDims)
        return false;

    for (uint32_t i = 0; i < a.numDims; ++i)
    {
        if (a.d[i] != b.d[i])
            return false;
    }
    return true;
}

bool
operator!=(const NvDsInferDims& a, const NvDsInferDims& b)
{
    return !(a == b);
}

bool isValidOutputFormat(const std::string& fmt)
{
    static std::unordered_set<std::string> ioFmt{"chw","chw2","chw4","hwc8","chw16","chw32"};
    return ioFmt.find(fmt) != ioFmt.end() ? true : false;
}

bool isValidOutputDataType(const std::string& dataType)
{
    static std::unordered_set<std::string> ioDataType{"fp32","fp16","int32","int8"};
    return ioDataType.find(dataType) != ioDataType.end() ? true : false;
}

nvinfer1::DataType str2DataType(const std::string& dataType)
{
    if(!dataType.compare("fp32"))
        return nvinfer1::DataType::kFLOAT;
    else if (!dataType.compare("fp16"))
        return nvinfer1::DataType::kHALF;
    else if (!dataType.compare("int32"))
        return nvinfer1::DataType::kINT32;
    else if(!dataType.compare("int8"))
        return nvinfer1::DataType::kINT8;
    else
        dsInferError("Invalid datatype string %s. Using default kFLOAT datatype", dataType.c_str());

    return nvinfer1::DataType::kFLOAT;
}

uint32_t str2TensorFormat(const std::string& fmt)
{
    if(!fmt.compare("chw"))
        return 1U << (uint32_t)nvinfer1::TensorFormat::kLINEAR;
    else if(!fmt.compare("chw2"))
        return 1U << (uint32_t)nvinfer1::TensorFormat::kCHW2;
    else if(!fmt.compare("chw4"))
        return 1U << (uint32_t)nvinfer1::TensorFormat::kCHW4;
    else if(!fmt.compare("hwc8"))
        return 1U << (uint32_t)nvinfer1::TensorFormat::kHWC8;
    else if(!fmt.compare("chw16"))
        return 1U << (uint32_t)nvinfer1::TensorFormat::kCHW16;
    else if(!fmt.compare("chw32"))
        return 1U << (uint32_t)nvinfer1::TensorFormat::kCHW32;
    else
        dsInferError("Invalid tensor format string %s. Using default kLINEAR", fmt.c_str());

    return 1U << (uint32_t)nvinfer1::TensorFormat::kLINEAR;
}

bool validateIOTensorNames(const BuildParams& params,
            const  nvinfer1::INetworkDefinition& network)
{
    for(auto fmt : params.outputFormats)
    {
        bool found = false;
        for(int i=0; !found && (i < network.getNbOutputs()); ++i)
        {
            auto output = network.getOutput(i);
            if(!fmt.first.compare(output->getName()))
                found = true;
        }
        if(!found)
        {
            dsInferError("Invalid output layer name specified %s", fmt.first.c_str());
            return false;
        }
    }
    return true;
}

bool isValidDeviceType(const std::string& dev)
{
  static std::unordered_set<std::string> deviceType{"gpu","dla"};
  return deviceType.find(dev) != deviceType.end() ? true : false;
}

bool isValidPrecisionType(const std::string& dataType)
{
  static std::unordered_set<std::string> precisionType{"fp32","fp16","int8"};
  return precisionType.find(dataType) != precisionType.end() ? true : false;
}

nvinfer1::DataType str2PrecisionType(const std::string& dataType)
{
  if(!dataType.compare("fp32"))
    return nvinfer1::DataType::kFLOAT;
  else if (!dataType.compare("fp16"))
    return nvinfer1::DataType::kHALF;
  else if(!dataType.compare("int8"))
    return nvinfer1::DataType::kINT8;
  else
    dsInferError("Invalid precisionType string %s. Using default kFLOAT(fp32) precisonType", dataType.c_str());

  return nvinfer1::DataType::kFLOAT;
}

nvinfer1::DeviceType str2DeviceType(const std::string& deviceType)
{
  if(!deviceType.compare("gpu"))
    return nvinfer1::DeviceType::kGPU;
  else if (!deviceType.compare("dla"))
    return nvinfer1::DeviceType::kDLA;
  else
    dsInferError("Invalid deviceType string %s. Using default kGPU deviceType", deviceType.c_str());

  return nvinfer1::DeviceType::kGPU;
}

} // namespace nvdsinfer

__attribute__ ((visibility ("default")))
const char*
NvDsInferStatus2Str(NvDsInferStatus status)
{
#define CHECK_AND_RETURN_STRING(status_iter) \
    if (status == status_iter)               \
    return #status_iter

    CHECK_AND_RETURN_STRING(NVDSINFER_SUCCESS);
    CHECK_AND_RETURN_STRING(NVDSINFER_CONFIG_FAILED);
    CHECK_AND_RETURN_STRING(NVDSINFER_CUSTOM_LIB_FAILED);
    CHECK_AND_RETURN_STRING(NVDSINFER_INVALID_PARAMS);
    CHECK_AND_RETURN_STRING(NVDSINFER_OUTPUT_PARSING_FAILED);
    CHECK_AND_RETURN_STRING(NVDSINFER_CUDA_ERROR);
    CHECK_AND_RETURN_STRING(NVDSINFER_TENSORRT_ERROR);
    CHECK_AND_RETURN_STRING(NVDSINFER_RESOURCE_ERROR);
    CHECK_AND_RETURN_STRING(NVDSINFER_UNKNOWN_ERROR);

    return "NVDSINFER_NULL";
#undef CHECK_AND_RETURN_STRING
}
