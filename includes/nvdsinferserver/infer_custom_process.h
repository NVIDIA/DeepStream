/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSINFERSERVER_CUSTOM_PROCESSOR_H__
#define __NVDSINFERSERVER_CUSTOM_PROCESSOR_H__

#include <infer_datatypes.h>

#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <queue>

namespace nvdsinferserver {

/**
 * @brief Interface of Custom processor which is created and loaded at runtime
 *        through CreateCustomProcessorFunc.
 * Note: Full dimensions are used for all the inputs and output tensors. then
 *       IBatchBuffer::getBatchSize() usually return 0. This is matched with
 *       Triton-V2 shapes for public. to get buf_ptr, user always use
 *       IBatchBuffer::getBufPtr(idx=0).
 */
class IInferCustomProcessor {
public:
    /** @brief IInferCustomProcessor will be deleted by nvdsinferserver lib.
     */
    virtual ~IInferCustomProcessor() = default;

    /** @brief Query the memory type, extraInputProcess() implementation supports.
     *    Memory will be allocated based on the return type and passed to
     *    extraInputProcess().
     *
     * @param type, [output], must be chosen from InferMemType::kCpu or
     *              InferMemType::kGpuCuda,
     */
    virtual void supportInputMemType(InferMemType& type) { type = InferMemType::kCpu; }

    /**
     * @brief Indicate whether this custom processor requires inference loop,
     *    in which nvdsinferserver lib guarantees extraInputProcess() and
     *    InferenceDone() running in order per each stream id. User can process last
     *    frame's output tensor from inferenceDone() and feed into next frame's
     *    inference input tensor in extraInputProcess()
     * @return true if need loop(e.g. LSTM based processing); Else @return false.
     */
    virtual bool requireInferLoop() const { return false; }

    /**
     * @brief Custom processor for extra input data.
     *
     * @param primaryInputs, [input], the primary image input
     * @param extraInputs [input/output], custom processing to generate extra tensor
     *        input data. The memory is pre-allocated. memory type is same as
     *        supportInputMemType returned types.
     * @param options, [input]. Associated options along with the input buffers.
     *        It has most of the common Deepstream metadata along with primary data.
     *        e.g. NvDsBatchMeta, NvDsObjectMeta, NvDsFrameMeta, stream ids and so on.
     *        See infer_ioptions.h to get all the potential key name and structures
     *        in the key-value table.
     * @return NvDsInferStatus, if successful implementation must return NVDSINFER_SUCCESS
     *        or an error value in case of error.
     */
    virtual NvDsInferStatus extraInputProcess(
        const std::vector<IBatchBuffer*>& primaryInputs, std::vector<IBatchBuffer*>& extraInputs,
        const IOptions* options) = 0;

    /**
     * @brief Inference done callback for custom postpocessing.
     *
     * @param outputs, [input], the inference output tensors. the tensor
     *        memory type could be controled by
     *        infer_config{ backend{ output_mem_type: MEMORY_TYPE_DEFAULT } },
     *        The default output tensor memory type is decided by triton model.
     *        User can set other values from MEMORY_TYPE_CPU, MEMORY_TYPE_GPU.
     * @param inOptions, [input], corresponding options from input tensors. It is
     *        same as options in extraInputProcess().
     * @return NvDsInferStatus, if successful implementation must return NVDSINFER_SUCCESS
     *        or an error value in case of error.
     */
    virtual NvDsInferStatus inferenceDone(
        const IBatchArray* outputs, const IOptions* inOptions) = 0;

    /**
     * @brief Notification of an error to the interface implementation.
     *
     * @param status, [input], error code
     */
    virtual void notifyError(NvDsInferStatus status) = 0;
};

}  // namespace nvdsinferserver

extern "C" {

/**
 * Custom processor context is created and loaded in runtime.
 *
 * @param[in]  config Contents of prototxt configuration file serialized as a string.
 * @param[in]  configLen use for string length of \a config
 * @return new instance of IInferCustomProcessor. If failed, return nullptr
 */
typedef nvdsinferserver::IInferCustomProcessor* (*CreateCustomProcessorFunc)(
    const char* config, uint32_t configLen);
}

#endif
