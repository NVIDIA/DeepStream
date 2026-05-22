/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file infer_extra_processor.h
 *
 * @brief Header file of class for processing extra inputs and custom post
 * processing.
 */

#ifndef __NVDSINFERSERVER_EXTRA_PROCESSOR_H__
#define __NVDSINFERSERVER_EXTRA_PROCESSOR_H__

#include "infer_base_backend.h"
#include "infer_common.h"
#include "infer_custom_process.h"
#include "infer_datatypes.h"
#include "infer_utils.h"

namespace nvdsinferserver {

using TensorMapPool = MapBufferPool<std::string, UniqCudaTensorBuf>;
using TensorMapPoolPtr = std::unique_ptr<TensorMapPool>;

/**
 * @brief: Extra processing pre/post inference.
 */
class InferExtraProcessor {
public:
    InferExtraProcessor();
    ~InferExtraProcessor();

    /*
     * @brief Load custom processor from custom library.
     */
    NvDsInferStatus initCustomProcessor(
        SharedDllHandle dlHandle, const std::string& funcName, const std::string& config);

    /*
     * @brief Allocate extra input resources including both CPU/GPU buffers.
     */
    NvDsInferStatus allocateExtraInputs(
        BaseBackend& backend, const std::set<std::string>& excludes, int32_t poolSize, int gpuId);

    /*
     * @brief Process extra input tensors per batched input.
     */
    NvDsInferStatus processExtraInputs(SharedBatchArray& inputs);

    /*
     * @brief Notify errors.
     */
    void notifyError(NvDsInferStatus status);

    /*
     * @brief Inference done callback outputs.
     */
    NvDsInferStatus checkInferOutputs(SharedBatchArray& outputs, SharedOptions inOptions);

    /*
     * @brief Destroy all resources including custom processors.
     */
    NvDsInferStatus destroy();

private:
    bool requireLoop() const { return m_RequireInferLoop; }

    /*
     * @brief Stores all input layers except primary input.
     */
    LayerDescriptionList m_ExtraInputLayers;
    /*
     * @brief Stores all input layers.
     */
    LayerDescriptionList m_FullInputLayers;
    /*
     * @brief Max batch size, 0 indicates no batching.
     */
    uint32_t m_maxBatch = 0;
    /*
     * @brief Flag indicating first dimension is dynamic size batch.
     * Only valid if m_maxBatch == 0
     */
    bool m_firstDimDynamicBatch = false;

    bool m_RequireInferLoop = false;
    UniqStreamManager m_StreamManager;
    InferCustomProcessorPtr m_CustomProcessor;
    TensorMapPoolPtr m_ExtInputHostPool;
    TensorMapPoolPtr m_ExtInputGpuPool;
    SharedCuStream m_Host2GpuStream;
};

}  // namespace nvdsinferserver

#endif
