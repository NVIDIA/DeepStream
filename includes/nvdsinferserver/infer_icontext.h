/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_icontext.h
 *
 * @brief Inference context library interface header file.
 *
 * This file defines the low level library interface for inference using
 * Triton inference server.
 *
 */

#ifndef __NVDSINFERSERVER_ICONTEXT_H__
#define __NVDSINFERSERVER_ICONTEXT_H__

#ifdef __cplusplus

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include <infer_datatypes.h>

namespace nvdsinferserver {

/**
 * Inference output callback interface.
 */
using InferOutputCb = std::function<void(NvDsInferStatus, SharedIBatchArray)>;

/**
 * Inference logging function interface.
 */
using InferLoggingFunc =
    std::function<void(NvDsInferLogLevel, const char* msg)>;

/**
 * The DeepStream inference interface class.
 */
class IInferContext {
public:
    virtual ~IInferContext() = default;

    /**
     * Initialize InferConxt with config text.
     *
     * @param[in]  prototxt is the config string.
     * @param[in]  logFunc use for print logs. If value is nullptr, use default
     *             log printer.
     * @return NVDSINFER_SUCCESS if run good.
     */
    virtual NvDsInferStatus initialize(
        const std::string& prototxt, InferLoggingFunc logFunc) = 0;

    /**
     * Run inference relevant processing behind. expect the call is async_mode.
     * When all behind processing finished. \a done would be called.
     *
     * @param[in]  input holds all batch buffer array.
     * @param[in]  outputCb use for callback with final status and output array.
     * @return NVDSINFER_SUCCESS if run good.
     */
    virtual NvDsInferStatus run(
        SharedIBatchArray input, InferOutputCb outputCb) = 0;

    /**
     * Destroy InferConxt
     *
     * @return NVDSINFER_SUCCESS if run good.
     */
    virtual NvDsInferStatus deinit() = 0;

    /**
     * Get the network input information.
     *
     * @param[in,out] networkInfo Reference to a NvDsInferNetworkInfo structure.
     */
    virtual void getNetworkInputInfo(NvDsInferNetworkInfo &networkInfo) = 0;
};

/**
 * Triton Server global instance. When it is instantiated, all models would be
 * loaded prior to all InferContext. Class interfaces is coming soon.
 */
class ITritonServerInstance;

} // namespace nvdsinferserver

extern "C" {

/**
 * Creates a new instance of IInferContext initialized using the supplied
 * parameters. The Triton Inference server is accessed in C-API mode.
 *
 * @param[in]  configStr Parameters to use for initialization of the context.
 * @param[in]  configStrLen use for string length of \a configStr
 * @return new instance of IInferContext. If failed, return nullptr
 */
INFER_EXPORT_API nvdsinferserver::IInferContext* createInferTrtISContext(
    const char* configStr, uint32_t configStrLen);

/**
 * Creates a light weight Triton instance of IInferContext.
 *
 * @return new instance of IInferContext. If failed, return nullptr
 */
INFER_EXPORT_API nvdsinferserver::IInferContext*
createInferTritonSimpleContext();

/**
 * Creates a new instance of IInferContext initialized using the supplied
 * parameters. The Triton Inference server is accessed using gRPC.
 *
 * @param[in]  configStr Parameters to use for initialization of the context.
 * @param[in]  configStrLen use for string length of \a configStr
 * @return new instance of IInferContext. If failed, return nullptr
 */
INFER_EXPORT_API nvdsinferserver::IInferContext*
createInferTritonGrpcContext(const char* configStr, uint32_t configStrLen);

/**
 * Creates Triton Server instance as global singleton. Application need hold it
 * until no component need triton inference in process.
 *
 * @param[in]  configStr Parameters for Triton model repo settings.
 * @param[in]  configStrLen use for string length of \a configStr
 * @param[out] instance use for output.
 * @return status. If ok, return NVDSINFER_SUCCESS.
 */
INFER_EXPORT_API NvDsInferStatus NvDsTritonServerInit(
    nvdsinferserver::ITritonServerInstance** instance, const char* configStr,
    uint32_t configStrLen);

/**
 * Destroys Triton Server Instance. Application need call this function before
 * process exist.
 *
 * @param[in] instance use for instance to be destroyed.
 * @return status. If ok, return NVDSINFER_SUCCESS.
 */
INFER_EXPORT_API NvDsInferStatus
NvDsTritonServerDeinit(nvdsinferserver::ITritonServerInstance* instance);

/**
 * Wrap a user buffer into SharedIBatchBuffer for IInferContext to use.
 *
 * @param[in]  buf The raw content data pointer.
 * @param[in]  bufBytes Byte size of \a buf.
 * @param[in]  desc Buffer Description of \a buf
 * @param[in]  batchSize Batch size of \a buf. 0 indicates a full-dim buffer.
 * @param[in]  freeFunc A C++14 function indicates how to free \a buf.
 * @return Batched buffer in shared_ptr. If failed, return nullptr.
 */
INFER_EXPORT_API nvdsinferserver::SharedIBatchBuffer NvDsInferServerWrapBuf(
    void* buf, size_t bufBytes,
    const nvdsinferserver::InferBufferDescription& desc, uint32_t batchSize,
    std::function<void(void* data)> freeFunc);

/**
 * Create an empty BatchArray.
 *
 * @return An empty batched array in shared_ptr. If failed, return nullptr.
 */
INFER_EXPORT_API nvdsinferserver::SharedIBatchArray
NvDsInferServerCreateBatchArray();

/**
 * Create a SharedIBatchBuffer with a vector of strings stored inside.
 *
 * @param[in]  strings A bunch of strings.
 * @param[in]  dims The shapes for each batch. It could be a full-dim if
 *             \a batchSize is 0.
 * @param[in]  batchSize Batch size of \a strings. 0 indicates non-batching.
 * @param[in]  name Tensor name of this buffer.
 * @param[in]  isInput Indicates whether the buffer is for input. It should
 *             always be true for external users.
 * @return A batched buffer storing all strings with memtype InferMemType::kCpu.
 */
INFER_EXPORT_API nvdsinferserver::SharedIBatchBuffer
NvDsInferServerCreateStrBuf(
    const std::vector<std::string>& strings,
    const nvdsinferserver::InferDims& dims, uint32_t batchSize,
    const std::string& name, bool isInput);
}

#endif

#endif
