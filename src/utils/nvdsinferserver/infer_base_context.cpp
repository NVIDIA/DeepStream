/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include <google/protobuf/text_format.h>

#include "infer_base_backend.h"
#include "infer_base_context.h"
#include "infer_cuda_utils.h"
#include "infer_iprocess.h"
#include "infer_proto_utils.h"
#include "infer_utils.h"

namespace nvdsinferserver {

const static int kMakBatchSize = 1024;

InferBaseContext::InferBaseContext() {}

InferBaseContext::~InferBaseContext() {
}

NvDsInferStatus
InferBaseContext::initialize(
    const std::string& prototxt, InferLoggingFunc logFunc)
{
    m_LoggingFunc = logFunc;
    if (!google::protobuf::TextFormat::ParseFromString(prototxt, &m_Config)) {
        printError("error: failed to parse inference config prototxt");
    }
    ic::InferenceConfig& config = m_Config;

    m_UniqueID = config.unique_id();
    m_MaxBatchSize = config.max_batch_size();

    if (m_UniqueID == 0) {
        printError("Unique ID not set");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (m_MaxBatchSize > kMakBatchSize) {
        printError("Batch-size (%d) more than maximum allowed batch-size (%d)",
            m_MaxBatchSize, kMakBatchSize);
        return NVDSINFER_CONFIG_FAILED;
    }

    /* Load the custom library if specified. */
    if (config.has_custom_lib() && !config.custom_lib().path().empty()) {
        std::unique_ptr<DlLibHandle> dlHandle = std::make_unique<DlLibHandle>(
            config.custom_lib().path(), RTLD_LAZY);
        if (!dlHandle->isValid()) {
            printError("Could not open custom lib: %s", dlerror());
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
        m_CustomLib = std::move(dlHandle);
    }

    if (!config.has_backend()) {
        printError("no backend configurated");
        return NVDSINFER_CONFIG_FAILED;
    }

    CTX_RETURN_NVINFER_ERROR(
        createNNBackend(config.backend(), maxBatchSize(), m_Backend),
        "create nn-backend failed, check config file settings");
    assert(m_Backend);

    CTX_RETURN_NVINFER_ERROR(
        fixateInferenceInfo(config, *m_Backend),
        "Infer context faied to initialize inference information");

    if (config.has_preprocess() && needPreprocess()) {
        CTX_RETURN_NVINFER_ERROR(
            createPreprocessor(config.preprocess(), m_Preprocessors),
            "Infer Context failed to create preprocessors.");
        assert(!m_Preprocessors.empty());
        CTX_RETURN_NVINFER_ERROR(
            buidNextPreprocMap(),
            "Infer Context failed to build next preprocessing map.");
    }

    if (config.has_postprocess()) {
        CTX_RETURN_NVINFER_ERROR(
            createPostprocessor(config.postprocess(), m_Postprocessor),
            "Infer Context failed to create postprocessor.");
        assert(m_Postprocessor);
    }

    /* Allocate binding buffers on the device and the corresponding host
     * buffers. */
    NvDsInferStatus status = allocateResource(config);
    if (status != NVDSINFER_SUCCESS) {
        printError("Failed to allocate buffers");
        return status;
    }

    m_Initialized = true;

    printDebug("InferContext initialized.");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferBaseContext::buidNextPreprocMap() {
    assert(m_NextPreprocMap.empty());
    assert(!m_Preprocessors.empty());
    for (auto i = m_Preprocessors.begin(); i != m_Preprocessors.end(); ++i) {
        assert(*i);
        IPreprocessor* next =
            (i + 1 == m_Preprocessors.end() ? nullptr : (i + 1)->get());
        m_NextPreprocMap[i->get()] = next;
    }
    assert(m_NextPreprocMap.size() == m_Preprocessors.size());
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferBaseContext::forEachPreprocess(
    IPreprocessor* cur, SharedBatchArray inputs, InferCompleted preprocessDone)
{
    assert(inputs && inputs->getSize() > 0);
    IPreprocessor* next = nullptr;
    if (!cur) {
        next =
            m_Preprocessors.empty() ? nullptr : m_Preprocessors.front().get();
    } else {
        assert(m_NextPreprocMap.count(cur));
        auto i = m_NextPreprocMap.find(cur);
        if (i == m_NextPreprocMap.end()) {
            // preprocessDone(NVDSINFER_UNKNOWN_ERROR, std::move(inputs));
            return NVDSINFER_UNKNOWN_ERROR;
        }
        next = i->second;
    }

    if (next) {
        return next->transform(
            std::move(inputs), mainStream(),
            [this, preprocessDone, next](NvDsInferStatus err, SharedBatchArray outs) mutable {
                if (err == NVDSINFER_SUCCESS) {
                    err = forEachPreprocess(next, outs, preprocessDone);
                }
                if (err != NVDSINFER_SUCCESS) {
                    preprocessDone(err, std::move(outs));
                }
            });
    } else {
        preprocessDone(NVDSINFER_SUCCESS, std::move(inputs));
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferBaseContext::run(SharedIBatchArray input, InferOutputCb outputCb)
{
    printDebug("InferContext run");

    SharedBatchArray inputs = std::dynamic_pointer_cast<BaseBatchArray>(input);
    if (!inputs) {
        printError(
            "InferBaseContext input buffer is invalid to cast to "
            "BaseBatchArray");
        return NVDSINFER_INVALID_PARAMS;
    }
    SharedOptions inOptions = inputs->getSafeOptions();

    auto inferDone = [this, done = std::move(outputCb)](NvDsInferStatus s, SharedBatchArray out) {
        if (s != NVDSINFER_SUCCESS) {
            notifyError(s);
        }
        done(s, std::move(out));
    };
    NvDsInferStatus status = NVDSINFER_SUCCESS;

    assert(inputs && inputs->getSize() > 0);
    if (!m_Preprocessors.empty()) {
        auto preprocessDone = [this, inferDone, inOptions](
                                  NvDsInferStatus err, SharedBatchArray out) {
            // update inference input options after preprocessing
            out->setOptions(inOptions);
            if (err == NVDSINFER_SUCCESS) {
                err = doInference(out, inferDone);
            }
            if (err != NVDSINFER_SUCCESS) {
                inferDone(err, std::move(out));
            }
        };
        assert(inputs->getSize() == 1);
        status = forEachPreprocess(nullptr, std::move(inputs), preprocessDone);
    } else {
        status = doInference(std::move(inputs), inferDone);
    }

    // if failed directly, no need callback
    if (status != NVDSINFER_SUCCESS) {
        notifyError(status);
    }
    return status;
}

NvDsInferStatus InferBaseContext::deinit() {
    printDebug("InferContext deinit");
    m_Preprocessors.clear();
    m_Backend.reset();
    m_Postprocessor.reset();
    m_NextPreprocMap.clear();
    m_Initialized = false;
    m_LoggingFunc = nullptr;
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferBaseContext::doPostCudaProcess(
    SharedBatchArray inputs, InferCompleted done) {
    assert(inputs && inputs->getSize() > 0);
    assert(m_Postprocessor);
    printDebug("InferContext doPostCudaProcess");
    return m_Postprocessor->postCudaProcess(std::move(inputs), mainStream(),
        [this, done](NvDsInferStatus err, SharedBatchArray outputs) {
            if (err == NVDSINFER_SUCCESS) {
                err = this->doPostHostProcess(outputs, done);
            }
            if (err != NVDSINFER_SUCCESS) {
                done(err, std::move(outputs));
            }
        });
}

NvDsInferStatus InferBaseContext::doPostHostProcess(
    SharedBatchArray inputs, InferCompleted done) {
    assert(inputs && inputs->getSize() > 0);
    assert(m_Postprocessor);
    printDebug("InferContext doPostHostProcess");
    return m_Postprocessor->postHostProcess(
        std::move(inputs), mainStream(), done);
}

void
InferBaseContext::rawDataInferDone(
    NvDsInferStatus status, SharedBatchArray outputs, SharedOptions inOptions, InferCompleted done)
{
    if (status == NVDSINFER_SUCCESS) {
        status = extraOutputTensorCheck(outputs, inOptions);
    }

    // No postprocessing needed, done directly.
    if (!m_Postprocessor) {
        done(status, std::move(outputs));
        return;
    }

    // need postprocesing
    if (status == NVDSINFER_SUCCESS) {
        status = this->doPostCudaProcess(outputs, done);
    }
    // failed then done
    if (status != NVDSINFER_SUCCESS) {
        done(status, std::move(outputs));
    }
}

NvDsInferStatus InferBaseContext::doInference(
    SharedBatchArray inputs, InferCompleted done) {
    assert(inputs && inputs->getSize() > 0);
    assert(m_Backend);
    printDebug("InferContext doInference");

    for (SharedBatchBuf& b : inputs->mutableBufs()) {
        b->mutableBufDesc().isInput = true;
        assert(!b->getBufDesc().name.empty());
    }

    RETURN_NVINFER_ERROR(preInference(inputs, config()), "pre-inference on input tensors failed.");

    if (inputs->getSize() < m_Backend->getInputLayerSize()) {
        InferError("input tensor number is less than backends size");
        return NVDSINFER_UNKNOWN_ERROR;
    }
    assert(inputs->getSize() == m_Backend->getInputLayerSize());

    SharedOptions inOptions = inputs->getSafeOptions();
    return m_Backend->enqueue(
        std::move(inputs), mainStream(),
        [this](SharedBatchArray in) mutable { this->backendConsumedInputs(std::move(in)); },
        [this, inOptions, done](NvDsInferStatus status, SharedBatchArray out) mutable {
            rawDataInferDone(status, std::move(out), std::move(inOptions), std::move(done));
        });
}

bool InferBaseContext::needCopyInputToHost() const {
    return config().has_extra() &&
        config().extra().copy_input_to_host_buffers();
}

bool
InferBaseContext::needPreprocess() const
{
    return config().has_preprocess();
}

void InferBaseContext::print(NvDsInferLogLevel l, const char* msg) {
    if (m_LoggingFunc) {
        m_LoggingFunc(l, msg);
    } else {
        dsInferLogPrint__(l, msg);
    }
}

} // namespace nvdsinferserver
