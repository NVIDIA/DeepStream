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
#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)

__global__ void
Convert_FtFTensorKernel(
        float *inBuffer,
        unsigned char *outBuffer,
        unsigned int width,
        unsigned int height)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        int v  = 255 * inBuffer[row * width + col];
        if(v < 0) {
          v = 0;
        } else if(v > 255) {
          v = 255;
        }
        outBuffer[row * width + col] = v;
    }
}

void
Convert_FtFTensor(
        float *inBuffer,
        unsigned char *outBuffer,
        unsigned int width,
        unsigned int height)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    Convert_FtFTensorKernel <<<blocks, threadsPerBlock, 0>>>
        (inBuffer, outBuffer, width, height);
}
