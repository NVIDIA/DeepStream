/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_datatypes.h
 *
 * @brief Header file for the data types used in the inference processing.
 */

#ifndef __NVDSINFERSERVER_DATA_TYPES_H__
#define __NVDSINFERSERVER_DATA_TYPES_H__

#include <infer_defines.h>
#include <infer_ioptions.h>
#include <nvdsinfer.h>
#include <stdarg.h>

#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace nvdsinferserver {

/**
 * @brief The type of tensor order.
 */
enum class InferTensorOrder : int {
    kNone = 0,
    /**
     * @brief NCHW (batch-channels-height-width) tensor order.
     */
    kLinear = 1,
    /**
     * @brief NHWC (batch-height-width-channels) tensor order.
     */
    kNHWC = 2,
};

/**
 * @brief The memory types of inference buffers.
 */
enum class InferMemType : int {
    kNone = 0,
    /**
     * @brief GPU CUDA memory.
     */
    kGpuCuda = 1,
    /**
     * @brief Host (CPU) memory.
     */
    kCpu = 2,
    /**
     * @brief CUDA pinned memory.
     */
    kCpuCuda = 3,
    /**
     * @brief NVRM surface memory.
     */
    kNvSurface = 5,
    /**
     * @brief NVRM surface array memory.
     */
    kNvSurfaceArray = 6,
};

/**
 * @brief Datatype of the tensor buffer.
 */
enum class InferDataType : int {
    kFp32 = FLOAT,  // 0
    kFp16 = HALF,  // 1
    kInt8 = INT8,  // 2
    kInt32 = INT32,  // 3
    kInt16 = 7,
    kUint8,
    kUint16,
    kUint32,
    kFp64,
    kInt64,
    kUint64,
    kString,  // for text/bytes => str_len(4byte) + str('a\0')
    kBool,
    kNone = -1,
};

/**
 * @brief Inference post processing types.
 */
enum class InferPostprocessType : int {
    /**
     * @brief Post processing for object detection.
     */
    kDetector = 0,
    /**
     * @brief Post processing for object classification.
     */
    kClassifier = 1,
    /**
     * @brief Post processing for image segmentation.
     */
    kSegmentation = 2,
    /**
     * @brief Post processing using Triton Classifier.
     */
    kTrtIsClassifier = 3,
    /**
     * @brief Custom post processing.
     */
    kOther = 100,
};

/**
 * @brief Image formats.
 */
enum class InferMediaFormat : int {
    /** 24-bit interleaved R-G-B */
    kRGB = 0,
    /** 24-bit interleaved B-G-R */
    kBGR,
    /** 8-bit Luma */
    kGRAY,
    /** 32-bit interleaved R-G-B-A */
    kRGBA,
    /** 32-bit interleaved B-G-R-x */
    kBGRx,
    kUnknown = -1,
};

/**
 * @brief Holds the information about the dimensions of a neural network layer.
 */
struct InferDims
{
  /** Number of dimensions of the layer.*/
  unsigned int numDims = 0;
  /** Size of the layer in each dimension. */
  int d[NVDSINFER_MAX_DIMS] = {0};
  /** Number of elements in the layer including all dimensions.*/
  unsigned int numElements = 0;
};

/**
 * Holds full dimensions (including batch size) for a layer.
 */
struct InferBatchDims
{
    int batchSize = 0;
    InferDims dims;
};

/**
 * @brief Holds the information about a inference buffer.
 */
struct InferBufferDescription {
    /**
     * @brief Memory type of the buffer allocation.
     */
    InferMemType memType;
    /**
     * @brief Device (GPU) ID where the buffer is allocated.
     */
    long int devId;
    /**
     * @brief Datatype associated with the buffer.
     */
    InferDataType dataType;
    /**
     * @brief Dimensions of the tensor.
     */
    InferDims dims;
    /**
     * @brief Per element bytes, except kString (with elementSize is 0)
     */
    uint32_t elementSize;
    /**
     * @brief Name of the buffer.
     */
    std::string name;
    /**
     * @brief Boolean indicating input or output buffer.
     */
    bool isInput;
};

// Common buffer interface [external]
class IBatchBuffer;
class IBatchArray;
class IOptions;

using SharedIBatchBuffer = std::shared_ptr<IBatchBuffer>;
using SharedIBatchArray = std::shared_ptr<IBatchArray>;
using SharedIOptions = std::shared_ptr<IOptions>;

/**
 * @brief Interface class for a batch buffer.
 */
class IBatchBuffer {
public:
    IBatchBuffer() = default;
    virtual ~IBatchBuffer() = default;
    virtual const InferBufferDescription& getBufDesc() const = 0;
    virtual void* getBufPtr(uint32_t batchIdx) const = 0;
    virtual uint32_t getBatchSize() const = 0;
    virtual uint64_t getTotalBytes() const = 0;
    virtual size_t getBufOffset(uint32_t batchIdx) const = 0;

private:
    DISABLE_CLASS_COPY(IBatchBuffer);
};

/**
 * @brief Interface class for an array of batch buffers.
 */
class IBatchArray {
public:
    IBatchArray() = default;
    virtual ~IBatchArray() = default;
    virtual uint32_t getSize() const = 0;
    virtual const IBatchBuffer* getBuffer(uint32_t arrayIdx) const = 0;
    virtual const IOptions* getOptions() const = 0;

    virtual SharedIBatchBuffer getSafeBuf(uint32_t arrayIdx) const = 0;

    // add values
    virtual void appendIBatchBuf(SharedIBatchBuffer buf) = 0;
    virtual void setIOptions(SharedIOptions o) = 0;

private:
    DISABLE_CLASS_COPY(IBatchArray);
};

} // namespace nvdsinferserver

#endif
