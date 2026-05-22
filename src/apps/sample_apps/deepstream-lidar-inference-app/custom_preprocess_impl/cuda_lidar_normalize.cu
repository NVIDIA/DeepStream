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

#include <assert.h>
#include <cuda_runtime.h>

extern "C" void ds3dCustomCudaLidarNormalize(
    float* in, float* out, int points, float offset, float scale, cudaStream_t stream);

static __global__ void
cuda_lidar_normalize_c4(float4* in, float4* out, int points, float offset, float scale)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= points)
        return;
    float4 p = in[idx];
    p.w = (p.w - offset) * scale;
    out[idx] = p;
}

extern "C" void
ds3dCustomCudaLidarNormalize(
    float* in, float* out, int points, float offset, float scale, cudaStream_t stream)
{
    size_t threads = 64;
    size_t blocks = (points + threads - 1) / threads;
    cuda_lidar_normalize_c4<<<blocks, threads, 0, stream>>>(
        (float4*)in, (float4*)out, points, offset, scale);
}