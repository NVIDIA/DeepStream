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

#include "infer_utils.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>

#include "infer_cuda_utils.h"
#include "infer_ibackend.h"
#include "infer_icontext.h"
#include "infer_post_datatypes.h"

namespace nvdsinferserver {

DlLibHandle::DlLibHandle(const std::string& path, int mode) : m_LibPath(path) {
    const char* libPath = (path.empty() ? nullptr : path.c_str());
    m_LibHandle = dlopen(libPath, mode);
    if (!m_LibHandle) {
        InferError(
            "Could not open lib: %s, error string: %s", safeStr(path),
            safeStr(dlerror()));
    }
    InferDebug("dlopen lib:%s", safeStr(m_LibPath));
}

DlLibHandle::~DlLibHandle() {
    if (m_LibHandle) {
        dlclose(m_LibHandle);
        InferDebug("dlopen lib:%s", safeStr(m_LibPath));
    }
}

bool fEqual(float a, float b) {
    if (fabs(a - b) <= std::numeric_limits<float>::epsilon())
        return true;
    return false;
}

std::string dims2Str(const InferDims& d) {
    if (d.numDims <= 0)
        return "";

    std::stringstream s;
    assert(d.numDims < NVDSINFER_MAX_DIMS);
    for (uint32_t i = 0; i < d.numDims - 1; ++i) {
        s << d.d[i] << "x";
    }
    s << d.d[d.numDims - 1];

    return s.str();
}

std::string batchDims2Str(const InferBatchDims& d) {
    return std::string("batch:") + std::to_string(d.batchSize) +
           ", dims:" + dims2Str(d.dims);
}

std::string dataType2Str(const InferDataType type) {
    const static std::unordered_map<InferDataType, std::string> sD2S{
        {InferDataType::kFp32, "kFp32"},
        {InferDataType::kFp16, "kFp16"},
        {InferDataType::kInt8, "kInt8"},
        {InferDataType::kInt32, "kInt32"},
        {InferDataType::kInt16, "kInt16"},
        {InferDataType::kUint8, "kUint8"},
        {InferDataType::kUint16, "kUint16"},
        {InferDataType::kUint32, "kUint32"},
        {InferDataType::kFp64, "kFp64"},
        {InferDataType::kInt64, "kInt64"},
        {InferDataType::kUint64, "kUint64"},
        {InferDataType::kString, "kString"},
        {InferDataType::kBool, "kBool"},
        {InferDataType::kNone, "kNone"},
    };

    auto const i = sD2S.find(type);
    if (i == sD2S.end()) {
        InferError("Unknown data-type:%d", static_cast<int>(type));
        return "InferDatatype::nullptr";
    }
    return i->second;
}

InferDataType grpcStr2DataType(const std::string &type) {
    const static std::unordered_map<std::string, InferDataType> sD2S{
        {"FP32", InferDataType::kFp32},
        {"FP16", InferDataType::kFp16 },
        {"INT8", InferDataType::kInt8 },
        {"INT32", InferDataType::kInt32 },
        {"INT16", InferDataType::kInt16 },
        {"UINT8", InferDataType::kUint8 },
        {"UINT16", InferDataType::kUint16 },
        {"UINT32", InferDataType::kUint32 },
        {"FP64", InferDataType::kFp64 },
        {"INT64", InferDataType::kInt64 },
        {"UINT64", InferDataType::kUint64 },
        {"BYTES", InferDataType::kString },
        {"BOOL", InferDataType::kBool },
    };

    auto const i = sD2S.find(type);
    if (i == sD2S.end()) {
        InferError("Unknown data-type:%d", type.c_str());
        return InferDataType::kNone;
    }
    return i->second;
}

std::string dataType2GrpcStr(const InferDataType type) {
    const static std::unordered_map<InferDataType, std::string> sD2S{
        {InferDataType::kFp32, "FP32"},
        {InferDataType::kFp16, "FP16"},
        {InferDataType::kInt8, "INT8"},
        {InferDataType::kInt32, "INT32"},
        {InferDataType::kInt16, "INT16"},
        {InferDataType::kUint8, "UINT8"},
        {InferDataType::kUint16, "UINT16"},
        {InferDataType::kUint32, "UINT32"},
        {InferDataType::kFp64, "FP64"},
        {InferDataType::kInt64, "INT64"},
        {InferDataType::kUint64, "UINT64"},
        {InferDataType::kString, "BYTES"},
        {InferDataType::kBool, "BOOL"},
    };

    auto const i = sD2S.find(type);
    if (i == sD2S.end()) {
        InferError("Unknown data-type:%d", static_cast<int>(type));
        return "";
    }
    return i->second;
}

bool operator<=(const InferDims& a, const InferDims& b) {
    assert(a.numDims == b.numDims);
    for (uint32_t i = 0; i < a.numDims; ++i) {
        if (a.d[i] > b.d[i])
            return false;
    }
    return true;
}

bool operator>(const InferDims& a, const InferDims& b) {
    return !(a <= b);
}

bool operator==(const InferDims& a, const InferDims& b) {
    if (a.numDims != b.numDims)
        return false;

    for (uint32_t i = 0; i < a.numDims; ++i) {
        if (a.d[i] != b.d[i])
            return false;
    }
    return true;
}

bool operator!=(const InferDims& a, const InferDims& b) {
    return !(a == b);
}

NvDsInferNetworkInfo dims2ImageInfo(
    const InferDims& dims, InferTensorOrder order) {
    if (InferTensorOrder::kNone == order) {
        InferWarning(
            "unsupported tensor order for dims to image-info, retry as "
            "kLinear");
        order = InferTensorOrder::kLinear;
    }

    assert(dims.numDims > 0);
    assert(!hasWildcard(dims));
    if (dims.numDims == 2) {
        return NvDsInferNetworkInfo{(uint32_t)dims.d[1], (uint32_t)dims.d[0], 1};
    }
    if (dims.numDims == 1) {
        return NvDsInferNetworkInfo{(uint32_t)dims.d[0], 1, 1};
    }
    int offset = std::max<int>(dims.numDims - 3, 0);
    if (InferTensorOrder::kLinear == order) {
        return NvDsInferNetworkInfo{(uint32_t)dims.d[offset + 2],
            (uint32_t)dims.d[offset + 1], (uint32_t)dims.d[offset + 0]};
    } else {
        return NvDsInferNetworkInfo{(uint32_t)dims.d[offset + 1],
            (uint32_t)dims.d[offset + 0], (uint32_t)dims.d[offset + 2]};
    }
}

std::string tensorOrder2Str(InferTensorOrder order) {
    switch (order) {
    case InferTensorOrder::kLinear:
        return "kLinear";
    case InferTensorOrder::kNHWC:
        return "kNHWC";
    case InferTensorOrder::kNone:
        return "kNone";
    default:
        return "UNKNOWN";
    }
}

static const char* strLogLevel(NvDsInferLogLevel l) {
    switch (l) {
    case NVDSINFER_LOG_ERROR:
        return "ERROR";
    case NVDSINFER_LOG_WARNING:
        return "WARNING";
    case NVDSINFER_LOG_INFO:
        return "INFO";
    case NVDSINFER_LOG_DEBUG:
        return "DEBUG";
    default:
        return "UNKNOWN";
    }
}

struct LogEnv {
    NvDsInferLogLevel levelLimit = NVDSINFER_LOG_INFO;
    std::mutex printMutex;
    LogEnv() {
        const char* cEnv = std::getenv("NVDSINFERSERVER_LOG_LEVEL");
        if (cEnv) {
            levelLimit = (NvDsInferLogLevel)std::stoi(cEnv);
        }
    }
};

static LogEnv gLogEnv;

void dsInferLogVPrint__(
    NvDsInferLogLevel level, const char* fmt, va_list args) {
    if (level > gLogEnv.levelLimit) {
        return;
    }
    constexpr int kMaxBufLen = 4096;

    std::array<char, kMaxBufLen> logMsgBuffer{{'\0'}};
    vsnprintf(logMsgBuffer.data(), kMaxBufLen - 1, fmt, args);

    FILE* f = (level <= NVDSINFER_LOG_ERROR) ? stderr : stdout;

    std::unique_lock<std::mutex> locker(gLogEnv.printMutex);
    fprintf(f, "%s: %s\n", strLogLevel(level), logMsgBuffer.data());
}

void dsInferLogPrint__(NvDsInferLogLevel level, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    dsInferLogVPrint__(level, fmt, args);
    va_end(args);
}

NvDsInferDims toCapi(const InferDims &dims)
{
    return reinterpret_cast<const NvDsInferDims&>(dims);
}

NvDsInferDataType
toCapiDataType(InferDataType dt)
{
    switch (dt) {
    case InferDataType::kFp32:
        return FLOAT;
    case InferDataType::kFp16:
        return HALF;
    case InferDataType::kInt8:
        return INT8;
    case InferDataType::kUint8:
        return UINT8;
    case InferDataType::kInt16:
    case InferDataType::kUint16: {
        InferWarning(
            "force convert inferserver datatype:%s to capi half",
            safeStr(dataType2Str(dt)));
        return HALF;
    }
    case InferDataType::kInt32:
    case InferDataType::kUint32:
        return INT32;
    case InferDataType::kInt64:
        return INT64;
    case InferDataType::kNone:
    default:
        InferError(
            "Unsupported data-type:%s to capi", safeStr(dataType2Str(dt)));
        return (NvDsInferDataType)-1;
    }
    return (NvDsInferDataType)-1;
}

NvDsInferLayerInfo toCapi(const LayerDescription& desc, void* bufPtr) {
    NvDsInferLayerInfo capiInfo{toCapiDataType(desc.dataType),
                                {toCapi(desc.inferDims)},
                                desc.bindingIndex,
                                desc.name.c_str(),
                                bufPtr,
                                desc.isInput};
    return capiInfo;
}

NvDsInferLayerInfo
toCapiLayerInfo(const InferBufferDescription& desc, void* buf)
{
    NvDsInferLayerInfo info{toCapiDataType(desc.dataType),
                            {toCapi(desc.dims)},
                            -1,
                            desc.name.c_str(),
                            buf,
                            desc.isInput};
    return info;
}

bool intersectDims(
    const InferDims& a, const InferDims& b, InferDims& c) {
    if (a.numDims != b.numDims) {
        return false;
    }
    c.numDims = a.numDims;
    for (uint32_t i = 0; i < c.numDims; ++i) {
        if (a.d[i] == b.d[i]) {
            c.d[i] = a.d[i];
        } else if (a.d[i] <= INFER_WILDCARD_DIM_VALUE) {
            c.d[i] = b.d[i];
        } else if (b.d[i] <= INFER_WILDCARD_DIM_VALUE) {
            c.d[i] = a.d[i];
        } else {
            return false;
        }
    }
    normalizeDims(c);
    return true;
}

bool isPrivateTensor(const std::string &tensorName) {
    return tensorName.find(INFER_SERVER_PRIVATE_BUF) == 0;
}

bool
isAbsolutePath(const std::string& path)
{
    return !path.empty() && (path[0] == '/');
}

std::string
dirName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string("/");
  }

  size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return std::string(".");
  }
  if (idx == 0) {
    return std::string("/");
  }

  return path.substr(0, idx);
}

std::string
joinPath(const std::string& a, const std::string& b)
{
  if (a.empty()) {
    return b;
  }
  if (b.empty()) {
    return a;
  }

  std::string joined = a;
  if (a[a.size() - 1] == '/') {
    return a + b;
  } else {
    return a + "/" + b;
  }
}

bool realPath(const std::string &inPath, std::string &absPath) {
    if (inPath.empty())
        return false;

    char realStr[PATH_MAX + 1] = {0};
    char *ret = realpath(inPath.c_str(), realStr);
    if (ret == nullptr && errno != ENOENT) {
        return false;
    }
    absPath = realStr;

    return true;
}

bool
isCpuMem(InferMemType type)
{
    if (type == InferMemType::kCpu || type == InferMemType::kCpuCuda)
        return true;
    return false;
}

std::string
memType2Str(InferMemType type)
{
    const static std::unordered_map<InferMemType, std::string> typeStrs{
#define MEMTYPE_2_STR(type) {InferMemType::type, #type}
        MEMTYPE_2_STR(kNone),    MEMTYPE_2_STR(kGpuCuda),   MEMTYPE_2_STR(kCpu),
        MEMTYPE_2_STR(kCpuCuda), MEMTYPE_2_STR(kNvSurface), MEMTYPE_2_STR(kNvSurfaceArray),
#undef MEMTYPE_2_STR
    };
    auto const i = typeStrs.find(type);
    if (i == typeStrs.end()) {
        return "UnknownMemType";
    }
    return i->second;
}

InferDims
fullDims(int batch, const InferDims& in)
{
    if (isNonBatch(batch)) {
        return in;
    }
    InferDims ret;
    ret.numDims = in.numDims + 1;
    assert(ret.numDims <= NVDSINFER_MAX_DIMS);
    ret.d[0] = batch;
    std::copy(in.d, in.d + in.numDims, &ret.d[1]);
    normalizeDims(ret);
    return ret;
}

bool
debatchFullDims(const InferDims& full, InferDims& debatched, uint32_t& batch)
{
    if (full.numDims < 1) {
        return false;
    }
    batch = (uint32_t)full.d[0];
    debatched.numDims = full.numDims - 1;
    std::copy(&full.d[1], &full.d[full.numDims], &debatched.d[0]);
    normalizeDims(debatched);
    return true;
}

bool
squeezeMatch(const InferDims& a, const InferDims& b)
{
    auto squeeze = [&](const InferDims& in) {
        InferDims out;
        for (uint32_t i = 0; i < in.numDims; ++i) {
            if (in.d[i] != 1) {
                out.d[out.numDims++] = in.d[i];
            }
        }
        return out;
    };
    InferDims a0 = squeeze(a);
    InferDims b0 = squeeze(b);
    if (a0 == b0) {
        return true;
    }
    return false;
}

SharedBatchBuf
ReshapeBuf(const SharedBatchBuf& in, uint32_t batch, const InferDims& reshape, bool reCalcBytes)
{
    assert(batch >= 0 && !hasWildcard(reshape));
    assert(in);
    InferBufferDescription desc = in->getBufDesc();
    if (!reCalcBytes && desc.dims == reshape && in->getBatchSize() == batch) {
        return in;
    }
    auto outFullDims = fullDims(batch, reshape);
    assert(dimsSize(outFullDims) <= dimsSize(fullDims(in->getBatchSize(), desc.dims)));
    desc.dims = reshape;
    uint64_t bytes = in->getTotalBytes();
    if (reCalcBytes) {
        RETURN_IF_FAILED(
            InferDataType::kString != desc.dataType, nullptr,
            "ReshapeBuf doesn't support kString to calculate bytes");
        bytes = getElementSize(desc.dataType) * dimsSize(outFullDims);
        assert(bytes > 0);
    }
    size_t bufOffset = in->getBufOffset(0);
    if (bufOffset == (size_t) -1) {
        InferError(
            "ReshapBuf failed, invalid buffer offset.");
        return nullptr;
    }
    SharedRefBatchBuf ret(
        new RefBatchBuffer(in->getBufPtr(0), bufOffset, bytes, desc, batch),
        [priv = in](RefBatchBuffer* ptr) mutable {
            priv.reset();
            delete ptr;
        });
    return ret;
}

SharedBatchBuf
reshapeToFullDimsBuf(const SharedBatchBuf& buf, bool reCalcBytes)
{
    if (buf->getBatchSize() == 0) {
        return buf;
    }
    InferDims newDims = fullDims(buf->getBatchSize(), buf->getBufDesc().dims);
    auto ret = ReshapeBuf(buf, 0, newDims, reCalcBytes);
    return ret;
}

NvDsInferStatus
tensorBufferCopy(const SharedBatchBuf& in, const SharedBatchBuf& out, const SharedCuStream& stream)
{
    assert(in && out);
    const auto& inDesc = in->getBufDesc();
    const auto& outDesc = out->getBufDesc();
    uint64_t byteSize = in->getTotalBytes();
    assert(in->getTotalBytes() == out->getTotalBytes());
    int gpuId = 0;
    enum cudaMemcpyKind kind = (cudaMemcpyKind)-1;

    static std::map<InferMemType, enum cudaMemcpyKind> GPUsrc2dst {
        {InferMemType::kGpuCuda, cudaMemcpyDeviceToDevice},
        {InferMemType::kCpu, cudaMemcpyDeviceToHost},
        {InferMemType::kCpuCuda, cudaMemcpyDeviceToHost},
    };
    static std::map<InferMemType, enum cudaMemcpyKind> CPUsrc2dst {
        {InferMemType::kGpuCuda, cudaMemcpyHostToDevice},
        {InferMemType::kCpu, cudaMemcpyHostToHost},
        {InferMemType::kCpuCuda, cudaMemcpyHostToHost},
    };
    switch (inDesc.memType) {
    case InferMemType::kGpuCuda: {
        auto i = GPUsrc2dst.find(outDesc.memType);
        if (i != GPUsrc2dst.end()) {
            kind = i->second;
        }
        gpuId = inDesc.devId;
        break;
    }
    case InferMemType::kCpuCuda:
    case InferMemType::kCpu: {
        auto i = CPUsrc2dst.find(outDesc.memType);
        if (i != CPUsrc2dst.end()) {
            kind = i->second;
        }
        gpuId = outDesc.devId;
        break;
    }
    default:
        break;
    }

    RETURN_IF_FAILED(kind != -1, NVDSINFER_CUDA_ERROR, "tensor copy kind not supported");

    if (kind != cudaMemcpyHostToHost) {
        cudaStream_t cuStream = 0;
        if (stream) {
            cuStream = *stream;
        }
        RETURN_CUDA_ERR(cudaSetDevice(gpuId), "cudaSetDevice failed to set dev-id:%d", gpuId);
        RETURN_CUDA_ERR(
            cudaMemcpyAsync(out->getBufPtr(0), in->getBufPtr(0), byteSize, kind, cuStream),
            "Failed to cudamemcopy from %s to %s", safeStr(inDesc.name), safeStr(outDesc.name));
    } else {
        memcpy(in->getBufPtr(0), out->getBufPtr(0), byteSize);
    }
    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver

using namespace nvdsinferserver;

extern "C" {

const char* NvDsInferStatus2Str(NvDsInferStatus status) {
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
    CHECK_AND_RETURN_STRING(NVDSINFER_TRITON_ERROR);
    CHECK_AND_RETURN_STRING(NVDSINFER_UNKNOWN_ERROR);

    return "NVDSINFER_NULL";
#undef CHECK_AND_RETURN_STRING
}

SharedIBatchBuffer
NvDsInferServerWrapBuf(
    void* buf, size_t bufBytes,
    const nvdsinferserver::InferBufferDescription& desc, uint32_t batchSize,
    std::function<void(void* buf)> freeFunc)
{
    if (hasWildcard(desc.dims)) {
        InferError(
            "NvDsInferServerWrapBuf failed since wildcard dims observed");
        return nullptr;
    }

    SharedRefBatchBuf ret(
        new RefBatchBuffer(buf, 0, bufBytes, desc, batchSize),
        [ff = std::move(freeFunc)](RefBatchBuffer* ptr) {
            assert(ptr);
            void* base = ptr->basePtr();
            delete ptr;
            if (ff) {
                ff(base);
            }
        });
    return ret;
}

SharedIBatchArray
NvDsInferServerCreateBatchArray()
{
    return std::make_shared<BaseBatchArray>();
}

SharedIBatchBuffer
NvDsInferServerCreateStrBuf(
    const std::vector<std::string>& strings, const InferDims& dims,
    uint32_t batchSize, const std::string& name, bool isInput)
{
    if (strings.empty()) {
        InferError(
            "NvDsInferServerCreateStrBuf failed since string vector is empty");
        return nullptr;
    }
    if (hasWildcard(dims)) {
        InferError(
            "NvDsInferServerCreateStrBuf failed since wildcard dims observed");
        return nullptr;
    }

    std::unique_ptr<std::string> inputStr = std::make_unique<std::string>();
    assert(inputStr);
    for (const auto& in : strings) {
        uint32_t len = in.size();
        inputStr->append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
        inputStr->append(in);
    }
    InferBufferDescription desc{
        memType : InferMemType::kCpu,
        devId : 0,
        dataType : InferDataType::kString,
        dims : dims,
        elementSize : 0,  // kString per element Size is 0
        name : name,
        isInput : isInput
    };
    void* data = &((*inputStr)[0]);
    uint32_t dataBytes = inputStr->size();
    SharedRefBatchBuf outBuf(
        new RefBatchBuffer((void*)data, 0, dataBytes, desc, batchSize),
        [strOwner = std::move(inputStr)](RefBatchBuffer* ref) mutable {
            delete ref;
            strOwner.reset();
        });
    return outBuf;
}
}
