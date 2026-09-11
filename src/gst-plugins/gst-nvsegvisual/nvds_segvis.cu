/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <device_launch_parameters.h>
#include "nvds_segvis.h"

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)

__global__ void updateBufferWithMaskedColor(unsigned char* buffer, int* mask, unsigned char*
                class2BGR, int width, int height, int pitch, unsigned int class_id, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pix_id = row * pitch + col * 4;
    int pix_id_mask = row * width + col;

    if ((col < width && row < height) ) {
        unsigned char* color = class2BGR + (mask[pix_id_mask] + 3) * 3;
        unsigned char* buffer_R = buffer + pix_id + 0;
        unsigned char* buffer_G = buffer + pix_id + 1;
        unsigned char* buffer_B = buffer + pix_id + 2;
        *buffer_R = color[0];
        *buffer_G = color[1];
        *buffer_B = color[2];
    }
}


__global__ void updateBufferWithOriginalBackground(unsigned char* buffer, int* mask, unsigned char* 
                class2BGR, int width, int height, int pitch, unsigned int class_id, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pix_id = row * pitch + col * 4;
    int pix_id_mask = row * width + col;

    if ((col < width && row < height) ) {
      if(mask[pix_id_mask] != class_id) {
        unsigned char* color = class2BGR + (mask[pix_id_mask] + 3) * 3;
        unsigned char* buffer_R = buffer + pix_id + 0;
        unsigned char* buffer_G = buffer + pix_id + 1;
        unsigned char* buffer_B = buffer + pix_id + 2;
        *buffer_R = (unsigned char)((color[0] * alpha) + (*buffer_R * (1-alpha)));
        *buffer_G = (unsigned char)((color[1] * alpha) + (*buffer_G * (1-alpha)));
        *buffer_B = (unsigned char)((color[2] * alpha) + (*buffer_B * (1-alpha)));
      }
    }
}

void updatePixelBuffer(unsigned char* buffer, int* mask, unsigned char* class2BGR, int width,
     int height, bool original_background, unsigned int class_id, float alpha, int pitch, cudaStream_t stream) {

    // Launch the CUDA kernel to update the pixel buffer for the entire 2D grid
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if(!original_background) {
      updateBufferWithMaskedColor <<<blocks, threadsPerBlock, 0, stream>>>(buffer, mask, class2BGR, width, height, pitch, class_id, alpha);
    }
    else {
      updateBufferWithOriginalBackground <<<blocks, threadsPerBlock, 0, stream>>>(buffer, mask, class2BGR, width, height, pitch, class_id, alpha);
    }
}
