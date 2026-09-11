/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <ostream>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

void nv12_to_rgba(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, cudaStream_t stream);

void nv12_to_bgr_planar_batch (uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, bool bSwap, cudaStream_t stream, float scale_factor, float *mean_data);

void resize_rgba_float_planar_batch(float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
    float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream);

void nv12_to_bgra(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch,
					int nWidth, int nHeight, cudaStream_t stream);

void nv12_to_bgra_batch(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream, bool bSwap=0);

void nv12_to_rgba_batch(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream, bool bSwap=0);

void resize_nv12_batch(const uint8_t *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, uint8_t *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
