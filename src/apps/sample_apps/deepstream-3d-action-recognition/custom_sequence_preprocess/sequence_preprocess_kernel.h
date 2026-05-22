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

#ifndef __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__
#define __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__

#include <cuda.h>
#include <cuda_runtime.h>

#define VEC4_SIZE 4
// float vector structure for multiple channels
typedef struct {
    float d[VEC4_SIZE];
} Float4Vec;

/**
 * NCDHW preprocess per ROI image
 *
 * @param outPtr output data pointer offset to current image position
 * @param inPtr input data pointer
 */
cudaError_t preprocessNCDHW(
    void* outPtr, unsigned int outC, unsigned int H, unsigned int W, unsigned int S,
    const void* inPtr, unsigned int inC, unsigned int inRowPitch, Float4Vec scales, Float4Vec means,
    bool swapRB, cudaStream_t stream);

#endif  // __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__