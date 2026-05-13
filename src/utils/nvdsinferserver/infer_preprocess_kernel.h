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
 * @brief Header file for pre-processing CUDA kernels with normalization and
 * mean subtraction required by nvdsinferserver.
 */

#ifndef __NVDSINFER_PREPROCESS_KERNEL_H__
#define __NVDSINFER_PREPROCESS_KERNEL_H__

#include <cuda.h>
#include <cuda_fp16.h>

#define NVDSINFER_MAX_PREPROC_CHANNELS 4
#define NVDSINFER_INVALID_CONVERSION_KEY UINT32_MAX

struct BatchSurfaceInfo {
    unsigned int widthAlign;
    unsigned int heightAlign;
    unsigned int channelAlign;
    // NCHW:pitchPerRow >= widthAlign
    // NHWC:pitchPerRow >= widthAlign*channelAlign
    unsigned int pitchPerRow;
    unsigned int dataSizePerBatch;
};

bool NvDsInferConvert_CxToPx(void* outBuffer, int outDataType,
    BatchSurfaceInfo outBufferInfo, void* inBuffer, int inDatatype,
    BatchSurfaceInfo inBufferInfo, int validN, int cropC, int cropH,
    int cropW, // crop area
    float scaleFactor, float meanOffsets[NVDSINFER_MAX_PREPROC_CHANNELS],
    int fromChannelIndices[NVDSINFER_MAX_PREPROC_CHANNELS], int toNCHW,
    cudaStream_t stream);

bool NvDsInferConvert_CxToPxWithMeanBuffer(void* outBuffer, int outDataType,
    BatchSurfaceInfo outBufferInfo, void* inBuffer, int inDatatype,
    BatchSurfaceInfo inBufferInfo, int validN, int cropC, int cropH,
    int cropW, // crop area
    float scaleFactor, float* meanDataBuffer,
    int fromChannelIndices[NVDSINFER_MAX_PREPROC_CHANNELS], int toNCHW,
    cudaStream_t stream);

/**
 * Function pointer type to which any of the NvDsInferConvert functions can be
 * assigned.
 */
typedef void (*NvDsInferConvertFcn)(void* outBuffer, int outDataType,
    BatchSurfaceInfo outBufferInfo, void* inBuffer, int inDatatype,
    BatchSurfaceInfo inBufferInfo, int validN, int cropC, int cropH,
    int cropW, // crop area
    float scaleFactor, float* meanDataBuffer,
    int fromChannelIndices[NVDSINFER_MAX_PREPROC_CHANNELS], int toNCHW,
    cudaStream_t stream);

#endif /* __NVDSINFER_CONVERSION_H__ */
