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

#include "obb_scanline.h"
#include <cuda_runtime.h>

/**
 * @brief GPU kernel for scanline-based OBB rotation
 *
 * Extracts a rotated OBB from an axis-aligned AABB using inverse rotation.
 * Each thread processes one output pixel.
 *
 * @param aabb_pixels Input AABB pixels (RGBA, GPU memory)
 * @param obb_pixels Output OBB pixels (RGBA, GPU memory)
 * @param aabb_width Width of input AABB
 * @param aabb_height Height of input AABB
 * @param aabb_pitch Pitch (stride) of AABB in bytes
 * @param obb_width Width of output OBB
 * @param obb_height Height of output OBB
 * @param obb_cx Center X of OBB in AABB space
 * @param obb_cy Center Y of OBB in AABB space
 * @param angle_rad Rotation angle in radians
 */
__global__ void obb_scanline_rotate_kernel(
    const uint8_t *aabb_pixels,
    uint8_t *obb_pixels,
    int aabb_width,
    int aabb_height,
    int aabb_pitch,
    int obb_width,
    int obb_height,
    float obb_cx,
    float obb_cy,
    float angle_rad)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= obb_width || row >= obb_height)
        return;

    // Precompute trig functions
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);
    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;

    // Calculate position in OBB local space (centered at origin)
    float local_x = (col + 0.5f) - half_w;
    float local_y = (row + 0.5f) - half_h;

    // Rotate to AABB space (inverse rotation)
    // [x']   [cos θ  -sin θ] [local_x]   [cx]
    // [y'] = [sin θ   cos θ] [local_y] + [cy]
    float src_x = obb_cx + local_x * cos_a - local_y * sin_a;
    float src_y = obb_cy + local_x * sin_a + local_y * cos_a;

    // Round to nearest pixel
    int src_xi = (int)(src_x + 0.5f);
    int src_yi = (int)(src_y + 0.5f);

    // Output pixel address
    uint8_t *dst = &obb_pixels[(row * obb_width + col) * 4];

    // Sample pixel with bounds checking
    if (src_xi >= 0 && src_xi < aabb_width &&
        src_yi >= 0 && src_yi < aabb_height)
    {
        const uint8_t *src = &aabb_pixels[src_yi * aabb_pitch + src_xi * 4];
        dst[0] = src[0];  // R
        dst[1] = src[1];  // G
        dst[2] = src[2];  // B
        dst[3] = src[3];  // A
    }
    else
    {
        // Out of bounds: black with full alpha
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 255;
    }
}

/**
 * @brief Host wrapper for OBB scanline rotation on GPU
 *
 * @param aabb_pixels Input AABB pixels (GPU memory)
 * @param obb_pixels Output OBB pixels (GPU memory, must be pre-allocated)
 * @param aabb_width Width of input AABB
 * @param aabb_height Height of input AABB
 * @param aabb_pitch Pitch (stride) of AABB in bytes
 * @param obb_width Width of output OBB
 * @param obb_height Height of output OBB
 * @param obb_cx Center X of OBB in AABB space
 * @param obb_cy Center Y of OBB in AABB space
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution
 * @return cudaError_t CUDA error code
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
    cudaStream_t stream)
{
    // Configure kernel launch parameters
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((obb_width + block.x - 1) / block.x,
              (obb_height + block.y - 1) / block.y);

    // Launch kernel
    obb_scanline_rotate_kernel<<<grid, block, 0, stream>>>(
        aabb_pixels,
        obb_pixels,
        aabb_width,
        aabb_height,
        aabb_pitch,
        obb_width,
        obb_height,
        obb_cx,
        obb_cy,
        angle_rad);

    // Check for launch errors
    return cudaGetLastError();
}

