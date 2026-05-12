/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>

#include "sequence_preprocess_kernel.h"

#define THREADS_PER_BLOCK_W 16
#define THREADS_PER_BLOCK_H 8

// device functions
__device__ __constant__ unsigned int kSwapChannels[4] = {2, 1, 0, 3};

template <bool swap>
__device__ __forceinline__ unsigned int inChannel(unsigned int i);

template <>
__device__ __forceinline__ unsigned int
inChannel<false>(unsigned int i)
{
    return i;
}

template <>
__device__ __forceinline__ unsigned int
inChannel<true>(unsigned int i)
{
    return kSwapChannels[i];
}

/**
 * convert HWC to NCDHW preprocess with normalization
 */
template <typename OutT, typename InT, bool swapRB>
__global__ void
ImageHWCToCSHW(
    OutT* out, unsigned int C, unsigned int H, unsigned int W, unsigned int SHW, const InT* in,
    unsigned int inC, unsigned int inRowPitch, Float4Vec mult, Float4Vec plus)
{
    unsigned int c = threadIdx.x;
    unsigned int w = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int h = blockIdx.y * blockDim.z + threadIdx.z;

    if (w >= W || h >= H) {
        return;
    }
    unsigned int inIdx = h * inRowPitch + w * inC + inChannel<swapRB>(c);
    float inData = 0.0f;
    if (c < inC) {
        inData = (float)in[inIdx];
    }
    unsigned int outIdx = c * SHW + h * W + w;
    float val = inData * mult.d[c] + plus.d[c];
    out[outIdx] = val;
}

/**
 * NCDHW preprocess host function
 */
cudaError_t
preprocessNCDHW(
    void* outPtr, unsigned int outC, unsigned int H, unsigned int W, unsigned int S,
    const void* inPtr, unsigned int inC, unsigned int inRowPitch, Float4Vec scales, Float4Vec means,
    bool swapRB, cudaStream_t stream)
{
    unsigned int HW = H * W;
    unsigned int SHW = S * HW;

    Float4Vec mult = scales;
    Float4Vec plus;
    for (int i = 0; i < VEC4_SIZE; ++i) {
        plus.d[i] = -scales.d[i] * means.d[i];
    }

    dim3 blocks(outC, THREADS_PER_BLOCK_W, THREADS_PER_BLOCK_H);  // (C, W, H)
    // grids (W_block, H_block, 1)
    dim3 grids(
        (W + THREADS_PER_BLOCK_W - 1) / THREADS_PER_BLOCK_W,
        (H + THREADS_PER_BLOCK_H - 1) / THREADS_PER_BLOCK_H, 1);

    if (swapRB) {
        ImageHWCToCSHW<float, unsigned char, true><<<grids, blocks, 0, stream>>>(
            (float*)outPtr, outC, H, W, SHW, (const unsigned char*)inPtr, inC, inRowPitch, mult,
            plus);
    } else {
        ImageHWCToCSHW<float, unsigned char, false><<<grids, blocks, 0, stream>>>(
            (float*)outPtr, outC, H, W, SHW, (const unsigned char*)inPtr, inC, inRowPitch, mult,
            plus);
    }
    return cudaGetLastError();
}