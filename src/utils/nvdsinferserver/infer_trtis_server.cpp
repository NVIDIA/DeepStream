/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file infer_trtis_server.cpp
 *
 * @brief Source file of wrapper classes for Triton Inference Server
 * server instance, inference request, response.
 *
 * This file defines the wrapper classes used for inference processing
 * using the Triton Inference Server C-API mode.
 *
 */

#include "infer_trtis_server.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>

#include <half.hpp>

#include "infer_icontext.h"
#include "infer_options.h"
#include "infer_postproc_buf.h"
#include "infer_trtis_backend.h"

namespace nvdsinferserver {

TrtServerRequest::TrtServerRequest(TrtServerPtr server)
    : m_ReqPtr(nullptr, TRITONSERVER_InferenceRequestDelete),
      m_Server(std::move(server))
{
    assert(m_Server.lock());
}

TrtServerRequest::~TrtServerRequest()
{
    m_ReqPtr.reset();
}

NvDsInferStatus
TrtServerRequest::init(
    const std::string& model, int64_t version, SharedBatchArray& inputs,
    const std::vector<std::string>& outputs, uint64_t reqId,
    const std::vector<TritonClassParams>& clasList)
{
    auto server = m_Server.lock();
    if (!server || !server->serverPtr()) {
        InferError("failed to init trtserver-request because the server has been shutdown");
        return NVDSINFER_RESOURCE_ERROR;
    }
    TRITONSERVER_Server* serverPtr = server->serverPtr();
    assert(!model.empty());
    assert(inputs && inputs->getSize());
    const auto option = inputs->getOptions();
    m_Model = model;
    m_ReqId = reqId;
    for (const auto& c : clasList) {
        m_ClasList.emplace(c.tensorName, c);
    }

    TRITONSERVER_InferenceRequest* req = nullptr;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceRequestNew(
            &req, serverPtr, model.c_str(), version),
        "TrtServerRequest failed to create request");

    m_ReqPtr.reset(req);

    std::string reqIdStr = std::to_string(reqId);
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceRequestSetId(req, safeStr(reqIdStr)),
        "TrtServerRequest failed to set request id:%" PRIu64, reqId);
    if (option) {
        RETURN_NVINFER_ERROR(
            setOption(option),
            "TrtServerRequest set option failed, req_id:%" PRIu64, reqId);
    }

    for (size_t itO = 0; itO < outputs.size(); ++itO) {
        // if tensor is not cls, add as raw tensor
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceRequestAddRequestedOutput(
                req, safeStr(outputs[itO])),
            "TrtServerRequest failed to add output:%s", safeStr(outputs[itO]));
    }
    m_Outputs = outputs;

    RETURN_NVINFER_ERROR(
        setInputs(inputs), "TrtServerRequest failed to set inputs data");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtServerRequest::setOption(const IOptions* option)
{
    TRITONSERVER_InferenceRequest* req = ptr();
    assert(req);
    assert(option);

    uint64_t seqId = 0;
    if (option->hasValue(OPTION_SEQUENCE_ID)) {
        RETURN_NVINFER_ERROR(
            option->getUInt(OPTION_SEQUENCE_ID, seqId),
            "TrtServerRequest failed to get option " OPTION_SEQUENCE_ID);
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceRequestSetCorrelationId(req, seqId),
            "TrtServerRequest failed to set correlation id:%" PRIu64, seqId);
    }

    uint32_t flags = 0;
    if (option->hasValue(OPTION_SEQUENCE_START)) {
        bool f = 0;
        RETURN_NVINFER_ERROR(
            option->getBool(OPTION_SEQUENCE_START, f),
            "TrtServerRequest failed to get option" OPTION_SEQUENCE_START);
        flags |= (f ? TRITONSERVER_REQUEST_FLAG_SEQUENCE_START : 0);
    }

    if (option->hasValue(OPTION_SEQUENCE_END)) {
        bool f = 0;
        RETURN_NVINFER_ERROR(
            option->getBool(OPTION_SEQUENCE_END, f),
            "TrtServerRequest failed to get option" OPTION_SEQUENCE_END);
        flags |= (f ? TRITONSERVER_REQUEST_FLAG_SEQUENCE_END : 0);
    }
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceRequestSetFlags(req, flags),
        "TrtServerRequest failed to set flags id:%" PRIu32, flags);

    InferDebug(
        "TrtServerRequest setOption correlation id:%" PRIu64 "flags:%u\n",
        seqId, flags);

    if (option->hasValue(OPTION_PRIORITY)) {
        uint64_t priority = 0;
        RETURN_NVINFER_ERROR(
            option->getUInt(OPTION_PRIORITY, priority),
            "TrtServerRequest failed to get option " OPTION_PRIORITY);
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceRequestSetPriority(req, (uint32_t)priority),
            "TrtServerRequest failed to set priority id:%" PRIu64, priority);
    }

    if (option->hasValue(OPTION_TIMEOUT)) {
        uint64_t timeout = 0;
        RETURN_NVINFER_ERROR(
            option->getUInt(OPTION_TIMEOUT, timeout),
            "TrtServerRequest failed to get option " OPTION_TIMEOUT);
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
                req, (uint32_t)timeout),
            "TrtServerRequest failed to set timeout:%" PRIu64, timeout);
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtServerRequest::setInputs(SharedBatchArray& inputs)
{
    auto server = m_Server.lock();
    assert(server);
    TRITONSERVER_InferenceRequest* req = ptr();
    assert(req);
    assert(inputs && inputs->getSize());
    const std::vector<SharedBatchBuf>& inBufs = inputs->bufs();
    assert(inBufs[0]);

    for (const auto& inbuf : inBufs) {
        const InferBufferDescription& inDesc = inbuf->getBufDesc();
        assert(inDesc.isInput);
        assert(!hasWildcard(inDesc.dims));
        InferDims fullShape = fullDims(inbuf->getBatchSize(), inDesc.dims);
        std::vector<int64_t> inDims(
            fullShape.d, fullShape.d + fullShape.numDims);
        TRITONSERVER_DataType dt = DataTypeToTriton(inDesc.dataType);
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceRequestAddInput(
                req, safeStr(inDesc.name), dt, inDims.data(), inDims.size()),
            "TrtServerRequest failed to add input tensor:%s",
            safeStr(inDesc.name));

        size_t bytes = inbuf->getTotalBytes();
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceRequestAppendInputData(
                req, safeStr(inDesc.name), inbuf->getBufPtr(0), bytes,
                MemTypeToTriton(inDesc.memType), inDesc.devId),
            "failed to set input: %s data", safeStr(inDesc.name));
    }
    m_Inputs = inputs;
    m_BufId = inputs->bufId();
    return NVDSINFER_SUCCESS;
}

void
TrtServerRequest::RequestOnRelease(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    assert(request);
    std::unique_ptr<SharedRequest> pThis(
        reinterpret_cast<SharedRequest*>(userp));
    assert(pThis && *pThis);
    assert((*pThis)->m_ReqPtr.get() == request);

    assert(flags);
    if (TRITONSERVER_REQUEST_RELEASE_ALL & flags) {
        (*pThis)->m_ReqPtr.reset();
    }
    return;
}

NvDsInferStatus
TrtServerRequest::setRequestComplete(
    TRITONSERVER_InferenceRequestReleaseFn_t requestCompleteCb, void* userPtr)
{
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceRequestSetReleaseCallback(
            ptr(), requestCompleteCb, userPtr),
        "TrtServerRequest failed to set request release callback");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtServerRequest::setResponseComplete(
    ShrTritonAllocator& allocator,
    TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteCb,
    void* responseUserPtr)
{
    assert(ptr());
    assert(allocator);
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
            ptr(), allocator->ptr(), reinterpret_cast<void*>(allocator.get()),
            responseCompleteCb, reinterpret_cast<void*>(responseUserPtr)),
        "TrtServerRequest failed to set response callback");
    return NVDSINFER_SUCCESS;
}

TrtServerResponse::TrtServerResponse(
    TrtServerPtr server, UniqTritonT<TRITONSERVER_InferenceResponse> data,
    uint64_t id)
    : m_ResponseId(id), m_Data(std::move(data)), m_Server(std::move(server))
{
    assert(m_Server.lock());
}

NvDsInferStatus
TrtServerResponse::parse(const TrtServerRequest* req)
{
    m_Status = NVDSINFER_TRITON_ERROR;
    if (!m_Data) {
        return NVDSINFER_TRITON_ERROR;
    }
    assert(req);
    assert(m_Data);
    TRITONSERVER_InferenceResponse* response = m_Data.get();
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceResponseError(response),
        "TritonServer response error received.");
    const char *modelName = nullptr, *idStr = nullptr;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceResponseModel(
            response, &modelName, &m_ModelVersion),
        "TritonServer get response model info failed.");
    m_Model = safeStr(modelName);
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceResponseId(response, &idStr),
        "TritonServer get response id failed.");
    std::istringstream issId(safeStr(idStr));
    issId >> m_ResponseId;

    // Parse parameters
    RETURN_NVINFER_ERROR(
        parseParams(),
        "TritonServer model:%s response failed to parse parameters",
        safeStr(model()));

    RETURN_NVINFER_ERROR(
        parseOutputData(req),
        "TritonServer model:%s response failed to parse output data",
        safeStr(model()));

    m_Status = NVDSINFER_SUCCESS;
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtServerResponse::parseParams()
{
    assert(m_Data);
    TRITONSERVER_InferenceResponse* response = m_Data.get();
    // Get parameters
    uint32_t paramNum = 0;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceResponseParameterCount(response, &paramNum),
        "TritonServer get response parameters count failed. model:%s",
        safeStr(model()));
    if (!paramNum) {
        return NVDSINFER_SUCCESS;
    }
    std::shared_ptr<BufOptions> params = std::make_shared<BufOptions>();
    assert(params);

    for (uint32_t iP = 0; iP < paramNum; ++iP) {
        const void* vvalue = nullptr;
        TRITONSERVER_ParameterType type = (TRITONSERVER_ParameterType)0;
        const char* name = nullptr;
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceResponseParameter(
                response, iP, &name, &type, &vvalue),
            "TritonServer get response parameter:%d failed. model:%s", (int)iP,
            safeStr(model()));
        switch (type) {
        case TRITONSERVER_PARAMETER_BOOL:
            params->setValue(
                std::string(name), *(reinterpret_cast<const bool*>(vvalue)));
            break;
        case TRITONSERVER_PARAMETER_INT:
            params->setValue(
                std::string(name), *(reinterpret_cast<const int64_t*>(vvalue)));
            break;
        case TRITONSERVER_PARAMETER_STRING:
            params->setValue(
                std::string(name),
                std::string(reinterpret_cast<const char*>(vvalue)));
            break;
        default:
            // Continue processing but report unknown errors
            InferError(
                "TritonServer model:%s response parameter:%s unknown type:%d",
                safeStr(model()), safeStr(name), (int)type);
        }
    }
    if (params->getCount() > 0) {
        m_Options = std::move(params);
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtServerResponse::parseOutputData(const TrtServerRequest* req)
{
    assert(req);
    assert(m_Data);
    auto server = m_Server.lock();
    if (!server) {
        InferError(
            "TrtServerResponse internal error, model:%s checked server is null",
            safeStr(model()));
        return NVDSINFER_UNKNOWN_ERROR;
    }
    TRITONSERVER_InferenceResponse* response = m_Data.get();
    uint32_t outputNum = 0;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_InferenceResponseOutputCount(response, &outputNum),
        "TritonServer model:%s response get output count failed.",
        safeStr(model()));
    if (outputNum != (uint32_t)req->outputs().size()) {
        InferError(
            "TritonServer model:%s response output count mismatched"
            "with request output",
            safeStr(model()));
        return NVDSINFER_TRITON_ERROR;
    }

    uint32_t batchFlags = 0;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerModelBatchProperties(
            server->serverPtr(), safeStr(model()), m_ModelVersion, &batchFlags,
            nullptr /* voidp */),
        "TritonServer model:%s get batch properties failed", safeStr(model()));
    bool hasBatch = (batchFlags & TRITONSERVER_BATCH_FIRST_DIM);
    const auto& classMap = req->classParams();

    for (uint32_t iOut = 0; iOut < outputNum; ++iOut) {
        const char* tensorName{nullptr};
        TRITONSERVER_DataType dataType = (TRITONSERVER_DataType)0;
        const int64_t* shape{nullptr};
        uint64_t dimNum = 0;
        const void* base{nullptr};
        size_t bytes = 0;
        TRITONSERVER_MemoryType memoryType = (TRITONSERVER_MemoryType)0;
        int64_t memoryId = 0;
        void* userp{nullptr};
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceResponseOutput(
                response, iOut, &tensorName, &dataType, &shape, &dimNum, &base,
                &bytes, &memoryType, &memoryId, &userp),
            "TritonServer model:%s response get output:%d failed",
            safeStr(model()), (int)iOut);
        InferDataType dsDT = DataTypeFromTriton(dataType);
        assert(dsDT != InferDataType::kNone);
        InferDims dsDims{0, {0}};
        assert(dimNum);
        // assert(base); It's possible to get nullptr for Riva
        assert(tensorName);
        dsDims.numDims = (hasBatch && dimNum) ? dimNum - 1 : dimNum;
        uint32_t batchSize = (hasBatch && dimNum) ? shape[0] : 0;
        uint32_t iOffset = (hasBatch ? 1 : 0);
        assert(NVDSINFER_MAX_DIMS + iOffset >= dimNum);
        for (uint32_t iD = 0; iD < dsDims.numDims; ++iD) {
            dsDims.d[iD] = (int)shape[iOffset + iD];
        }
        assert(!hasWildcard(dsDims));
        normalizeDims(dsDims);
        InferBufferDescription bufDesc{
            memType : MemTypeFromTriton(memoryType),
            devId : memoryId,
            dataType : dsDT,
            dims : dsDims,
            elementSize : getElementSize(dsDT),  // need reset later
            name : safeStr(tensorName),
            isInput : false
        };

        auto iterC = classMap.find(tensorName);
        if (iterC != classMap.end()) {
            const TritonClassParams& classP = iterC->second;
            assert(
                bufDesc.memType == InferMemType::kCpu ||
                bufDesc.memType == InferMemType::kCpuCuda);
            RETURN_NVINFER_ERROR(
                addClass(classP, bufDesc, std::max(batchSize, 1U), iOut, base),
                "TritonServer response unsupported tensor output parsing. "
                "model:%s, tensor name:%s, datatype:%s, dims:%s",
                safeStr(model()), safeStr(bufDesc.name),
                safeStr(dataType2Str(bufDesc.dataType)),
                safeStr(dims2Str(bufDesc.dims)));
        } else {
            SharedRefBatchBuf refBuf(
                new RefBatchBuffer((void*)base, 0, bytes, bufDesc, batchSize),
                [priv = m_Data](RefBatchBuffer* buf) mutable {
                    priv.reset();
                    delete buf;
                });
            m_BufOutputs.emplace_back(std::move(refBuf));
        }
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtServerResponse::addClass(
    const TritonClassParams& classP, const InferBufferDescription& desc, uint32_t batchSize,
    uint32_t tensorIdx, const void* base)
{
    assert(m_Data);
    assert(m_Server.lock());

    auto classOut = std::make_shared<ClassificationOutput>(batchSize);
    assert(classOut);
    for (uint32_t i = 0; i < batchSize; ++i) {
        InferClassificationOutput& ret = classOut->mutableOutput(i);
        RETURN_NVINFER_ERROR(
            topKClass(ret, classP, desc, tensorIdx, base),
            "unsupported tensor parsing. model:%s, tensor name:%s, datatype:%s,"
            "dims:%s",
            safeStr(model()), safeStr(desc.name),
            safeStr(dataType2Str(desc.dataType)), safeStr(dims2Str(desc.dims)));
    }
    classOut->finalize();
    m_BufOutputs.emplace_back(std::move(classOut));

    return NVDSINFER_SUCCESS;
}

using TopKTuple = std::tuple<uint32_t, float>;  // idx, prob
template <typename T>
static void
topKSort(
    const void* ptr, uint32_t total, uint32_t topK, std::vector<TopKTuple>& out)
{
    const T* prob = reinterpret_cast<const T*>(ptr);
    topK = std::min(topK, total);
    if (!topK) {
        return;
    }

    out.clear();
    out.reserve(total);
    for (uint32_t i = 0; i < total; ++i) {
        out.emplace_back(i, float(prob[i]));
    }
    std::partial_sort(
        out.begin(), out.begin() + topK, out.end(),
        [](const TopKTuple& a, const TopKTuple& b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    out.erase(out.begin() + topK, out.end());
    assert(out.size() == topK);
}

NvDsInferStatus
TrtServerResponse::topKClass(
    InferClassificationOutput& ret, const TritonClassParams& classP,
    const InferBufferDescription& desc, uint32_t tensorIdx, const void* base)
{
    assert(m_Data);
    assert(m_Server.lock());

    std::vector<TopKTuple> idxScore;

    switch (desc.dataType) {
    case InferDataType::kFp32:
        topKSort<float>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kFp16:
        topKSort<half_float::half>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kInt8:
        topKSort<int8_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kUint8:
        topKSort<uint8_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kInt16:
        topKSort<int16_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kUint16:
        topKSort<uint16_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kInt32:
        topKSort<int32_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kUint32:
        topKSort<uint32_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kInt64:
        topKSort<int64_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    case InferDataType::kUint64:
        topKSort<uint64_t>(base, desc.dims.numElements, classP.topK, idxScore);
        break;
    default:
        InferError(
            "mode:%s, unsupported output classification type:%s",
            safeStr(model()), safeStr(dataType2Str(desc.dataType)));
        return NVDSINFER_INVALID_PARAMS;
    }

    for (const auto& c : idxScore) {
        uint32_t idx = 0;
        float score = 0;
        std::tie(idx, score) = c;
        if (score < classP.threshold)
            continue;

        const char* label{nullptr};
        RETURN_TRTIS_ERROR(
            TRITONSERVER_InferenceResponseOutputClassificationLabel(
                m_Data.get(), tensorIdx, idx, &label),
            "TritonServer response mode:%s tensor:%s, get class label failed.",
            safeStr(model()), safeStr(desc.name));

        InferAttribute attr;
        attr.attributeIndex = (uint32_t)tensorIdx;
        attr.attributeValue = idx;
        attr.attributeConfidence = score;
        attr.attributeLabel = nullptr;
        attr.safeAttributeLabel = safeStr(label);
        ret.attributes.emplace_back(attr);
        if (ret.label.empty()) {
            ret.label = attr.safeAttributeLabel;
        } else {
            ret.label += " " + attr.safeAttributeLabel;
        }
    }
    return NVDSINFER_SUCCESS;
}


TrtServerAllocator::TrtServerAllocator(AllocFn alloc, FreeFn release)
    : m_Allocator(nullptr, TRITONSERVER_ResponseAllocatorDelete),
      m_allocFn(alloc), m_releaseFn(release)
{
    TRITONSERVER_ResponseAllocator* allocator = nullptr;
    CONTINUE_TRTIS_ERROR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator, ResponseAlloc, ResponseRelease, nullptr /*start_fn*/),
        "TrtServerAllocator failed to create response allocator");
    m_Allocator.reset(allocator);
}

using ResponseBufUserData = std::tuple<WeakTritonAllocator, std::string, SharedSysMem>;

TRITONSERVER_Error*
TrtServerAllocator::ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensorName,
    size_t bytes, TRITONSERVER_MemoryType preferredMemType,
    int64_t preferredDevId, void* userP, void** buffer, void** bufferUserP,
    TRITONSERVER_MemoryType* actualMemType, int64_t* actualMemTypeId)
{
    TrtServerAllocator* pThis = reinterpret_cast<TrtServerAllocator*>(userP);
    assert(pThis);
    assert(tensorName);
    *bufferUserP = nullptr;
    *buffer = nullptr;
    *actualMemType = preferredMemType;
    *actualMemTypeId = preferredDevId;
    InferMemType dsPreferredMemType = MemTypeFromTriton(preferredMemType);
    assert(dsPreferredMemType != InferMemType::kNone);

    // triton has its own behavior but not guaranteed
    if (!pThis->m_allocFn) {
        InferWarning(
            "TrtServerAllocator allocFn is not set, fallback to "
            "Triton default behavior");
        return nullptr;
    }
    SharedSysMem sysMem =
        pThis->m_allocFn(safeStr(tensorName), bytes, dsPreferredMemType, preferredDevId);
    if (!sysMem) {
        InferError(
            "TrtServerAllocator failed to allocate tensor:%s buf, fallback to "
            "Triton",
            safeStr(tensorName));
        *buffer = nullptr;
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("failed to allocate output tensor: ") +
             safeStr(tensorName) + ", with bytes:" + std::to_string(bytes))
                .c_str());
    }
    assert(sysMem->bytes() >= bytes);
    // classification or zero dims callback with 0 bytes
    if (!bytes) {
        return nullptr;
    }

    *buffer = sysMem->ptr();
    *actualMemType = MemTypeToTriton(sysMem->type());
    *actualMemTypeId = sysMem->devId();
    ResponseBufUserData* ud =
        new ResponseBufUserData(pThis->shared_from_this(), safeStr(tensorName), std::move(sysMem));
    *bufferUserP = reinterpret_cast<void*>(ud);

    assert(
        TRITONSERVER_MEMORY_GPU == *actualMemType ||
        TRITONSERVER_MEMORY_CPU == *actualMemType ||
        TRITONSERVER_MEMORY_CPU_PINNED == *actualMemType);
    InferDebug("TrtServerAllocator allocate tensor:%s bytes:%zu",
        safeStr(tensorName), bytes);

    return nullptr;
}

TRITONSERVER_Error*
TrtServerAllocator::ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* bufferUserP,
    size_t bytes, TRITONSERVER_MemoryType memType, int64_t devId)
{
    std::unique_ptr<ResponseBufUserData> ud(
        reinterpret_cast<ResponseBufUserData*>(bufferUserP));

    // delete cpu buffer for classification
    if (!ud && memType == TRITONSERVER_MEMORY_CPU) {
        delete[](char*) buffer;
        return nullptr;
    }

    assert(ud);
    WeakTritonAllocator wkThis;
    std::string tensorName;
    SharedSysMem mem;
    std::tie(wkThis, tensorName, mem) = *ud;
    ud.reset();
    assert(mem && !tensorName.empty());
    ShrTritonAllocator pThis = wkThis.lock();

    InferDebug("TrtServerAllocator releasing tensor:%s bytes:%zu",
        safeStr(tensorName), bytes);

    if (!pThis) {
        InferWarning(
            "TrtServerAllocator released before tensor memory free",
            safeStr(tensorName));
        return nullptr;
    }
    if (pThis->m_releaseFn) {
        pThis->m_releaseFn(tensorName, std::move(mem));
    }
    return nullptr;
}

// Triton server crashed when deleted as static global variable.
// To keep the server safe, we'll delete it when reference count to zero, this is
// usually in InferContext deinit.
std::weak_ptr<TrtISServer> TrtISServer::sTrtServerInstance;
std::mutex TrtISServer::sTrtServerMutex;

bool
triton::RepoSettings::initFrom(
    const ic::TritonModelRepo& modelRepo, const std::vector<int>& devIds)
{
    // Update roots
    for (auto const& rootDir : modelRepo.root()) {
        if (!rootDir.empty()) {
            std::string absRoot;
            if (!realPath(rootDir, absRoot)) {
                InferError(
                    "RepoSettings ignored root:%s since path not existed",
                    safeStr(rootDir));
                continue;
            }
            roots.emplace(absRoot);
        }
    }
    if (roots.empty()) {
        InferError("RepoSettings roots is empty");
        return false;
    }

    logLevel = modelRepo.log_level();
    strictModelConfig = modelRepo.strict_model_config();
    tfAllowSoftPlacement = !modelRepo.tf_disable_soft_placement();
    tfGpuMemoryFraction = modelRepo.tf_gpu_memory_fraction();
    if (modelRepo.model_control_mode().empty()) {
        controlMode = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
    } else if (modelRepo.model_control_mode() == "none") {
        controlMode = TRITONSERVER_MODEL_CONTROL_NONE;
    } else if (modelRepo.model_control_mode() == "explicit") {
        controlMode = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
    } else {
        InferError(
            "RepoSettings detect unsupported model_control_mode:%s",
            safeStr(modelRepo.model_control_mode()));
        return false;
    }

    if (!modelRepo.backend_dir().empty()) {
        backendDirectory = modelRepo.backend_dir();
    }

    // Update minComputeCapacity
    double minCC = TRITON_DEFAULT_MINIMUM_COMPUTE_CAPABILITY;
    if (!fEqual(modelRepo.min_compute_capacity(), 0.0f)) {
        minCC = std::min<double>(modelRepo.min_compute_capacity(), minCC);
    }
#if 0  // Disabled since Tegra header version mismatched.
    if (!devIds.empty()) {
        for (int32_t gpuid : devIds) {
            cudaDeviceProp cuprops{{0}};
            if (cudaSuccess != cudaGetDeviceProperties(&cuprops, gpuid)) {
                InferWarning(
                    "RepoSettings failed to get gpu_device(%d) property",
                    gpuid);
                continue;
            }
            double cc = (double)cuprops.major + cuprops.minor * 0.1f;
            if (cc < minCC)
                minCC = cc;
        }
    }
#endif
    assert(minCC > 0);
    minComputeCapacity = minCC;
    for (const auto& iDevMem : modelRepo.cuda_device_memory()) {
        cudaDevMemMap.emplace(
            iDevMem.device(), iDevMem.memory_pool_byte_size());
    }

    pinnedMemBytes = TRITON_DEFAULT_PINNED_MEMORY_BYTES;
    if (modelRepo.pinned_mem_case() ==
        ic::TritonModelRepo::kPinnedMemoryPoolByteSize) {
        pinnedMemBytes = modelRepo.pinned_memory_pool_byte_size();
    }

    for (const auto& iBeConf : modelRepo.backend_configs()) {
        backendConfigs.emplace_back(BackendConfig{
            iBeConf.backend(), iBeConf.setting(), iBeConf.value()});
    }

    debugStr = modelRepo.DebugString();
    return true;
}

bool
triton::RepoSettings::operator==(const triton::RepoSettings& other) const
{
    if (roots.size() != other.roots.size()) {
        return false;
    }
    auto iB = other.roots.begin();
    for (auto iA = roots.begin(); iA != roots.end(); ++iA, ++iB) {
        if (*iA != *iB) {
            return false;
        }
    }

    if (logLevel != other.logLevel ||
        tfAllowSoftPlacement != other.tfAllowSoftPlacement ||
        !fEqual(tfGpuMemoryFraction, other.tfGpuMemoryFraction) ||
        strictModelConfig != other.strictModelConfig ||
        !fEqual(minComputeCapacity, other.minComputeCapacity) ||
        pinnedMemBytes != other.pinnedMemBytes ||
        backendDirectory != other.backendDirectory) {
        return false;
    }
    if (controlMode != other.controlMode) {
        InferWarning("the 2 repo's controlMode are different. first:%s, second:%s"
          " , program will continue but may cause unexpected behavior.",
          safeStr(TritonControlModeToStr(controlMode)),
          safeStr(TritonControlModeToStr(other.controlMode)));
    }
    return true;
}

TrtServerPtr
TrtISServer::getInstance(const triton::RepoSettings* repo)
{
    TrtServerPtr server;
#if 1
    {
        std::unique_lock<std::mutex> locker(sTrtServerMutex);
        server = sTrtServerInstance.lock();
        if (!repo && !server) {
            InferError(
                "failed to initialize triton server repo since settings is "
                "null");
            return nullptr;
        }
        if (!server) {
            server.reset(new TrtISServer(*repo));
            if (server->initialize() == NVDSINFER_SUCCESS) {
                sTrtServerInstance = server;
            } else {
                InferError(
                    "failed to initialize trtserver on repo dir: %s",
                    safeStr(repo->debugStr));
                return nullptr;
            }
        }
    }
#else
    std::call_once(sInstanceOnceflag, [&repoDir, &server = sTrtServerInstance]() mutable {
        assert(!repoDir.empty());
        TrtServerPtr instance(new TrtISServer(repoDir));
        assert(instance);
        if (instance->initialize() == NVDSINFER_SUCCESS) {
            server = std::move(instance);
        } else {
            InferError("failed to initialize trtserver instance.");
        }
    });
    server = sTrtServerInstance;
#endif
    assert(server);
    if (!repo) {
        return server;
    }
    const triton::RepoSettings& existing = server->getRepoSettings();
    if (*repo != existing) {
        InferError(
            "New request repo settings do not match exist server settings."
            "\n Request:\n%s, \nexisting:\n%s",
            safeStr(repo->debugStr), safeStr(existing.debugStr));
        return nullptr;
    }

    return server;
}

TrtISServer::TrtISServer(const triton::RepoSettings& repo)
    : m_Impl(nullptr, TRITONSERVER_ServerDelete), m_RepoSettings{repo}
{
    assert(!m_RepoSettings.roots.empty());
}

TrtISServer::~TrtISServer() {
    try {
        if (m_Impl) {
            CONTINUE_TRTIS_ERROR(
                TRITONSERVER_ServerStop(m_Impl.get()),
                "failed to stop repo server");
        }
        m_Impl.reset();
    } catch (const std::exception &e) {
        InferError("Catch exception when delete Triton server, msg:%s", e.what());
    } catch (...) {
        InferError("Catch unknown exception when delete Triton server");
    }
}

NvDsInferStatus TrtISServer::initialize() {
    TRITONSERVER_ServerOptions* rawOpt = nullptr;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsNew(&rawOpt), "failed to new server option");
    UniqTritonT<TRITONSERVER_ServerOptions> options(
        rawOpt, TRITONSERVER_ServerOptionsDelete);
    for (const auto& path : m_RepoSettings.roots) {
        RETURN_TRTIS_ERROR(
            TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                rawOpt, path.c_str()),
            "failed to set model repo path: %s", safeStr(path));
    }
    bool bError = true, bWarn = false, bInfo = false;
    int dVerb = 0;
    if (m_RepoSettings.logLevel >= NVDSINFER_LOG_WARNING) {
        bWarn = true;
    }
    if (m_RepoSettings.logLevel >= NVDSINFER_LOG_INFO) {
        bInfo = true;
    }
    if (m_RepoSettings.logLevel > NVDSINFER_LOG_INFO) {
        dVerb = m_RepoSettings.logLevel - NVDSINFER_LOG_INFO;
    }
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetLogError(rawOpt, bError),
        "failed to set log error");
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetLogWarn(rawOpt, bWarn),
        "failed to set log warning");

    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetLogInfo(rawOpt, bInfo),
        "failed to set log info");
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetLogVerbose(rawOpt, dVerb),
        "failed to set log verbose");

    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetModelControlMode(
            rawOpt, (TRITONSERVER_ModelControlMode)m_RepoSettings.controlMode),
        "failed to set model control");
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetStrictReadiness(rawOpt, true),
        "failed to set strict readiness");
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(
            rawOpt, m_RepoSettings.strictModelConfig),
        "failed to set  strict model configuration");

    // same as TRITONSERVER_ServerOptionsSetBackendConfig(
    //  rawOpt, "", "backend-directory",
    //  safeStr(m_RepoSettings.backendDirectory)));
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetBackendDirectory(
            rawOpt, safeStr(m_RepoSettings.backendDirectory)),
        "failed to set backend directory:%s",
        safeStr(m_RepoSettings.backendDirectory));
    std::string softPlace =
        (m_RepoSettings.tfAllowSoftPlacement ? "true" : "false");
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetBackendConfig(rawOpt,
        "tensorflow", "allow-soft-placement", safeStr(softPlace)),
        "failed to set tensorflow softplacement:%s", safeStr(softPlace));
    std::string memFracStr = std::to_string(m_RepoSettings.tfGpuMemoryFraction);
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetBackendConfig(rawOpt,
        "tensorflow", "gpu-memory-fraction", safeStr(memFracStr)),
        "failed to set tensorflow gpu-memory-fraction:%s", safeStr(memFracStr));
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
            rawOpt, m_RepoSettings.minComputeCapacity),
        "failed to set minimum compute capability");

    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
            rawOpt, m_RepoSettings.pinnedMemBytes),
        "failed to set pinned memory size");

    for (const auto& iDevMem : m_RepoSettings.cudaDevMemMap) {
        RETURN_TRTIS_ERROR(
            TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
                rawOpt, iDevMem.first, iDevMem.second),
            "failed to set device: %u, cuda_mem:%" PRIu64 "bytes",
            iDevMem.first, iDevMem.second);
    }

    for (const auto& iBeConf : m_RepoSettings.backendConfigs) {
        RETURN_TRTIS_ERROR(
            TRITONSERVER_ServerOptionsSetBackendConfig(
                rawOpt, safeStr(iBeConf.backend), safeStr(iBeConf.key),
                safeStr(iBeConf.value)),
            "failed to set backend_config, backend: %s, setting: %s, value: %s",
            safeStr(iBeConf.backend), safeStr(iBeConf.key),
            safeStr(iBeConf.value));
    }

    TRITONSERVER_Server* server = nullptr;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerNew(&server, rawOpt),
        "failed to create repo server");
    m_Impl.reset(server);
    bool isReady = false;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerIsReady(server, &isReady),
        "failed to check readiness");
    if (!isReady)
        return NVDSINFER_TRITON_ERROR;
    return NVDSINFER_SUCCESS;
}

bool TrtISServer::isServerLive() {
    assert(m_Impl);

    bool isLive = false;
    CHECK_TRTIS_ERR_W_ACTION(
        TRITONSERVER_ServerIsLive(m_Impl.get(), &isLive), return false,
        "failed to get server live");
    if (!isLive) {
        InferWarning("Triton repo server is not live");
        return false;
    }
    return true;
}

bool TrtISServer::isServerReady() {
    assert(m_Impl);

    bool isReady = false;
    CHECK_TRTIS_ERR_W_ACTION(
        TRITONSERVER_ServerIsReady(m_Impl.get(), &isReady), return false,
        "failed to get server ready status");
    if (!isReady) {
        InferWarning("Triton repo server is not ready");
        return false;
    }
    return true;
}

bool
TrtISServer::isModelReady(const std::string& model, int64_t version)
{
    assert(m_Impl);

    bool isReady = false;
    CHECK_TRTIS_ERR_W_ACTION(
        TRITONSERVER_ServerModelIsReady(
            m_Impl.get(), safeStr(model), version, &isReady),
        return false, "TritonServer failed to check model status");
    if (!isReady) {
        InferDebug(
            "Triton model:%s, v:%" PRId64 " is not ready", safeStr(model),
            version);
    }
    return isReady;
}

NvDsInferStatus TrtISServer::loadModel(const std::string& modelName) {
    assert(m_Impl);
    assert(!modelName.empty());

    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerLoadModel(m_Impl.get(), modelName.c_str()),
        "failed to load model %s", safeStr(modelName));
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus TrtISServer::unloadModel(const std::string& modelName) {
    assert(m_Impl);
    assert(!modelName.empty());

    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerUnloadModel(m_Impl.get(), modelName.c_str()),
        "failed to unload model %s", safeStr(modelName));
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus TrtISServer::getModelConfig(
    const std::string& model, int64_t version, ni::ModelConfig& config) {
    assert(m_Impl);
    assert(!model.empty());

    TRITONSERVER_Message* rawMsg = nullptr;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerModelConfig(
            m_Impl.get(), model.c_str(), version, 1 /* config_version */, &rawMsg),
        "Triton: get model:%s config failed", safeStr(model));
    assert(rawMsg);
    UniqTritonT<TRITONSERVER_Message> safeMsg(
        rawMsg, TRITONSERVER_MessageDelete);

    const char* buffer = nullptr;
    size_t byteSize = 0;
    RETURN_TRTIS_ERROR(
        TRITONSERVER_MessageSerializeToJson(rawMsg, &buffer, &byteSize),
        "failed to serialize model:%s config to json", safeStr(model));

    ::google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;

    std::string tmpStr{buffer, byteSize};
    ::google::protobuf::util::JsonStringToMessage(tmpStr, &config, options);

    return NVDSINFER_SUCCESS;
}

SharedRequest
TrtISServer::createRequest(
    const std::string& model, int64_t version, SharedBatchArray& inputs,
    const std::vector<std::string>& outputs, const std::vector<TritonClassParams>& clasList)
{
    SharedRequest request(new TrtServerRequest(shared_from_this()));
    assert(request);
    uint64_t id = m_LastRequestId++;
    if (request->init(model, version, inputs, outputs, id, clasList) !=
        NVDSINFER_SUCCESS) {
        InferError(
            "Triton failed to create request for model: %s version:%" PRId64, safeStr(model),
            version);
        return nullptr;
    }
    return request;
}

NvDsInferStatus
TrtISServer::inferAsync(
    SharedRequest request, WeakTritonAllocator allocator,
    TritonInferAsyncDone done)
{
    assert(m_Impl);
    assert(request && request->ptr());
    auto resAllocator = allocator.lock();
    assert(resAllocator);

    std::unique_ptr<SharedRequest> reqUserPtr =
        std::make_unique<SharedRequest>(request);

    RETURN_NVINFER_ERROR(
        request->setRequestComplete(
            TrtServerRequest::RequestOnRelease, reqUserPtr.release()),
        "Triton inferAsync failed to set request completeCB.");

    std::unique_ptr<InferUserData> userData(
        new InferUserData(request, done, this));

    RETURN_NVINFER_ERROR(
        request->setResponseComplete(
            resAllocator, TrtISServer::InferComplete, userData.release()),
        "Triton inferAsync failed to set response completeCB.");

    RETURN_TRTIS_ERROR(
        TRITONSERVER_ServerInferAsync(
            m_Impl.get(), request->ptr(), nullptr /*trace*/),
        "Triton inferAsync API call failed");

    return NVDSINFER_SUCCESS;
}

void
TrtISServer::InferComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
        // TODO Decoupled Streaming support later.
        assert(false);
        InferError("TritonServer received response without FINAL flags");
        return;
    }
    SharedRequest request;
    TritonInferAsyncDone done;
    TrtISServer* pThis = nullptr;
    std::unique_ptr<InferUserData> userData(reinterpret_cast<InferUserData*>(userp));
    tie(request, done, pThis) = *userData;
    assert(request && done && pThis);
    UniqTritonT<TRITONSERVER_InferenceResponse> responsePtr(
        response, TRITONSERVER_InferenceResponseDelete);
    UniqResponse newResponse =
        pThis->createResponse(std::move(responsePtr), request->id());
    userData.reset(nullptr);
    done(std::move(request), std::move(newResponse));
}

UniqResponse
TrtISServer::createResponse(
    UniqTritonT<TRITONSERVER_InferenceResponse>&& data, uint64_t id)
{
    if (!data) {
        InferError("Triton server get null response data on request:%" PRIu64, id);
        return nullptr;
    }

    UniqResponse response(
        new TrtServerResponse(shared_from_this(), std::move(data), id));
    assert(response);
    return response;
}

} // namespace nvdsinferserver


using namespace nvdsinferserver;
NvDsInferStatus
NvDsTritonServerInit(
    ITritonServerInstance** instance, const char* configStr,
    uint32_t configStrLen)
{
    assert(instance);
    assert(configStrLen > 0);
    ic::TritonModelRepo modelRepo;
    if (!google::protobuf::TextFormat::ParseFromString(
        {configStr, (size_t)configStrLen}, &modelRepo)) {
        InferError("NvDsTritonServerInit: failed to parse TritonModelRepo prototxt");
    }
    triton::RepoSettings repoSettings;
    if (!repoSettings.initFrom(modelRepo, {})) {
        InferError(
            "NvDsTritonServerInit extract repo settings failed. info:%s",
            safeStr(modelRepo.DebugString()));
        return NVDSINFER_CONFIG_FAILED;
    }
    if (repoSettings.controlMode != (int32_t)TRITONSERVER_MODEL_CONTROL_NONE) {
        InferWarning("NvDsTritonServerInit suggest to set model_control_mode:none."
          " otherwise may cause unknow issues.");
    }

    auto ptr = TrtISServer::getInstance(&repoSettings);
    RETURN_IF_FAILED(
        ptr, NVDSINFER_CONFIG_FAILED,
        "NvDsTritonServerInit failed to get global triton instance");
    TrtServerPtr* outPtr = new TrtServerPtr(ptr);
    *instance = reinterpret_cast<ITritonServerInstance*>(outPtr);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsTritonServerDeinit(ITritonServerInstance* instance)
{
    RETURN_IF_FAILED(
        instance, NVDSINFER_INVALID_PARAMS,
        "TritonDeinit failed with no instance");
    nvdsinferserver::TrtServerPtr* ptr =
        reinterpret_cast<nvdsinferserver::TrtServerPtr*>(instance);
    delete ptr;
    return NVDSINFER_SUCCESS;
}
