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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>

/* Forward declarations for extern "C" functions */
extern "C" int initSAD_CUDA(cudaStream_t* stream, 
                            unsigned int** d_partial, 
                            unsigned int** h_partial,
                            int max_width, int max_height);
extern "C" void cleanupSAD_CUDA(cudaStream_t stream,
                                unsigned int* d_partial,
                                unsigned int* h_partial);
extern "C" unsigned int runSAD_CUDA(const unsigned char* frameA,
                                    const unsigned char* frameB,
                                    int width, int height,
                                    cudaStream_t stream,
                                    unsigned int* d_partial,
                                    unsigned int* h_partial,
                                    int max_blocks);

__global__ void computeSADKernel(const unsigned char* frameA, 
                                 const unsigned char* frameB, 
                                 unsigned int* partialSums, 
                                 int totalPixels) {
    extern __shared__ unsigned int sdata[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local = threadIdx.x;
    unsigned int diff = 0;

    // Compute thread-local absolute difference
    if (tid < totalPixels) {
        diff = abs((int)frameA[tid] - (int)frameB[tid]);
    }

    sdata[local] = diff;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local < s)
            sdata[local] += sdata[local + s];
        __syncthreads();
    }

    // Write block result to global memory
    if (local == 0)
        partialSums[blockIdx.x] = sdata[0];
}

// Initialize CUDA resources for SAD computation
// Call this once during plugin initialization
extern "C" int initSAD_CUDA(cudaStream_t* stream, 
                            unsigned int** d_partial, 
                            unsigned int** h_partial,
                            int max_width, int max_height) {
    // Create CUDA stream
    if (cudaStreamCreate(stream) != cudaSuccess) {
        printf("Error: Failed to create CUDA stream\n");
        return -1;
    }

    // Calculate maximum blocks needed
    int totalPixels = max_width * max_height;
    int threads = 256;
    int max_blocks = (totalPixels + threads - 1) / threads;

    // Allocate device memory for partial sums
    if (cudaMalloc(d_partial, max_blocks * sizeof(unsigned int)) != cudaSuccess) {
        printf("Error: Failed to allocate GPU memory for partial sums\n");
        cudaStreamDestroy(*stream);
        return -1;
    }

    // Allocate pinned host memory for async transfers
    if (cudaMallocHost(h_partial, max_blocks * sizeof(unsigned int)) != cudaSuccess) {
        printf("Error: Failed to allocate pinned host memory for partial sums\n");
        cudaFree(*d_partial);
        cudaStreamDestroy(*stream);
        return -1;
    }

    return max_blocks;
}

// Cleanup CUDA resources
// Call this during plugin cleanup
extern "C" void cleanupSAD_CUDA(cudaStream_t stream,
                                unsigned int* d_partial,
                                unsigned int* h_partial) {
    if (d_partial) cudaFree(d_partial);
    if (h_partial) cudaFreeHost(h_partial);
    if (stream) cudaStreamDestroy(stream);
}

// Host-callable wrapper - safe for use from C
// Note: frameA and frameB are already device pointers from DeepStream pipeline
// Uses pre-allocated memory and stream for better performance
extern "C" unsigned int runSAD_CUDA(const unsigned char* frameA,
                                    const unsigned char* frameB,
                                    int width, int height,
                                    cudaStream_t stream,
                                    unsigned int* d_partial,
                                    unsigned int* h_partial,
                                    int max_blocks) {
    if (!frameA || !frameB || width <= 0 || height <= 0) {
        printf("Error: Invalid input parameters to runSAD\n");
        return 0;
    }
    
    if (!stream || !d_partial || !h_partial) {
        printf("Error: Invalid CUDA resources\n");
        return 0;
    }
    
    int totalPixels = width * height;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    
    if (blocks > max_blocks) {
        printf("Error: Required blocks (%d) exceeds max_blocks (%d)\n", blocks, max_blocks);
        return 0;
    }

    // Launch kernel with stream - frameA and frameB are already device pointers
    computeSADKernel<<<blocks, threads, threads * sizeof(unsigned int), stream>>>(
        frameA, frameB, d_partial, totalPixels);

    // Check for kernel launch errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaError));
        return 0;
    }

    // Async copy results back to host using the stream
    if (cudaMemcpyAsync(h_partial, d_partial, blocks * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        printf("Error: Failed to copy results from GPU\n");
        return 0;
    }

    // Synchronize only the stream (not the entire device)
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        printf("Error: CUDA stream synchronization failed\n");
        return 0;
    }

    // Sum partial results
    unsigned int sad = 0;
    for (int i = 0; i < blocks; i++) {
        sad += h_partial[i];
    }

    return sad;
}
