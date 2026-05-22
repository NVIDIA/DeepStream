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
