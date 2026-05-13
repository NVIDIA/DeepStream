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
 * @file infer_base_context.h
 *
 * @brief Header file of the base class for inference context.
 *
 * This file declares the base class for handling inference context for the
 * nvinferserver low level library.
 *
 */

#ifndef __INFER_BASE_CONTEXT_H__
#define __INFER_BASE_CONTEXT_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include "infer_base_backend.h"
#include "infer_common.h"
#include "infer_icontext.h"
#include "nvdsinferserver_config.pb.h"

namespace ic = nvdsinferserver::config;

namespace nvdsinferserver {

class DlLibHandle;

using InferCompleted = std::function<void(NvDsInferStatus, SharedBatchArray)>;

/**
 * @brief The base class for handling the inference context. It creates the
 * NN backend, pre-processors, post-processors and calls these for the
 * inference processing.
 */
class InferBaseContext : public IInferContext {
public:
    InferBaseContext();
    ~InferBaseContext() override;

    NvDsInferStatus initialize(
        const std::string& prototxt, InferLoggingFunc logFunc) final;
    NvDsInferStatus run(SharedIBatchArray input, InferOutputCb outputCb) final;
    NvDsInferStatus deinit() override;

private:
    virtual NvDsInferStatus createNNBackend(const ic::BackendParams& params,
        int maxBatchSize, UniqBackend& backend) = 0;
    virtual NvDsInferStatus fixateInferenceInfo(
        const ic::InferenceConfig& config, BaseBackend& backend) = 0;
    virtual NvDsInferStatus createPreprocessor(
        const ic::PreProcessParams& params,
        std::vector<UniqPreprocessor>& processors) = 0;
    virtual NvDsInferStatus createPostprocessor(
        const ic::PostProcessParams& params, UniqPostprocessor& processor) = 0;
    virtual NvDsInferStatus allocateResource(
        const ic::InferenceConfig& config) = 0;

    virtual NvDsInferStatus preInference(
        SharedBatchArray& inputs, const ic::InferenceConfig& config)
    {
        return NVDSINFER_SUCCESS;
    }

    virtual NvDsInferStatus extraOutputTensorCheck(
        SharedBatchArray& outputs, SharedOptions inOptions)
    {
        return NVDSINFER_SUCCESS;
    }
    virtual void notifyError(NvDsInferStatus status) = 0;

    void rawDataInferDone(
        NvDsInferStatus status, SharedBatchArray outputs, SharedOptions inOptions,
        InferCompleted done);

protected:
    virtual void backendConsumedInputs(SharedBatchArray inputs) {
        inputs.reset();
    }
    virtual SharedCuStream& mainStream() = 0;

    const ic::InferenceConfig& config() const { return m_Config; }
    int maxBatchSize() const { return m_MaxBatchSize; }
    int uniqueId() const { return m_UniqueID; }
    BaseBackend* backend() { return m_Backend.get(); }
    const SharedDllHandle& customLib() const { return m_CustomLib; }
    bool needCopyInputToHost() const;
    void print(NvDsInferLogLevel l, const char* msg);
    bool needPreprocess() const;

private:
    NvDsInferStatus buidNextPreprocMap();
    NvDsInferStatus forEachPreprocess(
        IPreprocessor* cur, SharedBatchArray input, InferCompleted done);
    NvDsInferStatus doInference(SharedBatchArray inputs, InferCompleted done);
    NvDsInferStatus doPostCudaProcess(
        SharedBatchArray inputs, InferCompleted done);
    NvDsInferStatus doPostHostProcess(
        SharedBatchArray inputs, InferCompleted done);

private:
    InferLoggingFunc m_LoggingFunc;
    uint32_t m_UniqueID = 0;
    /**
     * @brief Maximum batch size, 0 or 1 indicates non batch case.
     */
    uint32_t m_MaxBatchSize = 1;
    bool m_Initialized = false;

    /**
     * Custom unique pointers. These nvinferserver objects will get deleted
     * automatically when the NvDsInferContext object is deleted.
     */
    /**@{*/
    std::vector<UniqPreprocessor> m_Preprocessors;
    std::unordered_map<IPreprocessor*, IPreprocessor*> m_NextPreprocMap;
    UniqBackend m_Backend;
    UniqPostprocessor m_Postprocessor;
    SharedDllHandle m_CustomLib;
    /**@}*/

    ic::InferenceConfig m_Config;
};

} // namespace nvdsinferserver

#define _MAX_LOG_LENGTH 4096
#define printMsg(level, tag_str, fmt, ...)                                  \
    do {                                                                    \
        char* baseName = strrchr((char*)__FILE__, '/');                     \
        baseName = (baseName) ? (baseName + 1) : (char*)__FILE__;           \
        std::vector<char> logMsgBuffer(_MAX_LOG_LENGTH, 0);                 \
        snprintf(logMsgBuffer.data(), _MAX_LOG_LENGTH - 1,                  \
            tag_str " %s() <%s:%d> [UID = %d]: " fmt, \
            __func__, baseName, __LINE__, uniqueId(), ##__VA_ARGS__);       \
        this->print(level, logMsgBuffer.data());                            \
    } while (0)

#define printError(fmt, ...)                                           \
    do {                                                               \
        printMsg(NVDSINFER_LOG_ERROR, "Error in", fmt, ##__VA_ARGS__); \
    } while (0)

#define printWarning(fmt, ...)                                               \
    do {                                                                     \
        printMsg(NVDSINFER_LOG_WARNING, "Warning from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printInfo(fmt, ...)                                            \
    do {                                                               \
        printMsg(NVDSINFER_LOG_INFO, "Info from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printDebug(fmt, ...)                                        \
    do {                                                            \
        printMsg(NVDSINFER_LOG_DEBUG, "DEBUG", fmt, ##__VA_ARGS__); \
    } while (0)

#define CTX_RETURN_NVINFER_ERROR(err, fmt, ...) \
    CHECK_NVINFER_ERROR_PRINT(                  \
        err, return ifStatus, printError, fmt, ##__VA_ARGS__)

#define CTX_RETURN_CUDA_ERR(err, fmt, ...) \
    CHECK_CUDA_ERR_W_ACTION(               \
        err, return NVDSINFER_CUDA_ERROR, printError, fmt, ##__VA_ARGS__)

#endif /* __INFER_BASE_CONTEXT_H__ */
