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

#ifndef __OBB_SCANLINE_H__
#define __OBB_SCANLINE_H__

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform scanline-based OBB rotation on GPU
 *
 * Extracts a rotated OBB from an axis-aligned AABB using GPU acceleration.
 * All buffers must be in GPU memory.
 *
 * @param aabb_pixels Input AABB pixels (RGBA, GPU memory)
 * @param obb_pixels Output OBB pixels (RGBA, GPU memory, must be pre-allocated)
 * @param aabb_width Width of input AABB
 * @param aabb_height Height of input AABB
 * @param aabb_pitch Pitch (stride) of AABB in bytes
 * @param obb_width Width of output OBB
 * @param obb_height Height of output OBB
 * @param obb_cx Center X of OBB in AABB space
 * @param obb_cy Center Y of OBB in AABB space
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution (use 0 for default stream)
 * @return cudaError_t CUDA error code (cudaSuccess on success)
 */
cudaError_t obb_scanline_rotate_gpu(
    const uint8_t *aabb_pixels,
    uint8_t *obb_pixels,
    int aabb_width,
    int aabb_height,
    int aabb_pitch,
    int obb_width,
    int obb_height,
    float obb_cx,
    float obb_cy,
    float angle_rad,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* __OBB_SCANLINE_H__ */

