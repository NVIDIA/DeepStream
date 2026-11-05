/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include "modules/utility/cuda_common.hpp"
#include "modules/Segmenter/cuSegmenter.h"

namespace cuSegmenter
{
#define CUDA_BLOCK_SIZE_1D 1024
#define EPS 1e-3

/**
 * @brief Rescale a segmentation mask into its destination region.
 * @param[in] src_mask_frame Source segmentation mask.
 * @param[in] src_frame_width Source mask width.
 * @param[in] src_frame_element_size Number of elements in the source mask.
 * @param[in] dst_width Destination mask width.
 * @param[in] dst_element_size Number of elements in the destination mask.
 * @param[in] src_roi Source region of interest.
 * @param[out] dst_mask_patch Destination mask patch to populate.
 */
static __global__ void cuda_rescaleSegmentationOutput(
    const float* src_mask_frame, const int src_frame_width, const int src_frame_element_size,
    const int dst_width, const int dst_element_size, const float4 src_roi, float *dst_mask_patch
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    /// Each thread writes one target's one dst mask pixel.
    if (i >= dst_element_size)
        return;

    /// X and y in current target's dst mask.
    const int dst_x = i % dst_width;
    const int dst_y = i / dst_width;
    const int dst_height = dst_element_size / dst_width;

    /// Location in src whole frame.
    float src_x = float(dst_x) / dst_width * src_roi.w + src_roi.x;
    float src_y = float(dst_y) / dst_height * src_roi.z + src_roi.y;

    src_x = min(max(0., src_x), float(src_frame_width - 1));
    src_y = min(max(0., src_y), float(src_frame_element_size / src_frame_width - 1));

    const int src_x0 = (int) src_x;
    const int src_x1 = min(src_x0 + 1, src_frame_width - 1);
    const int src_y0 = (int) src_y;
    const int src_y1 = min(src_y0 + 1, src_frame_element_size / src_frame_width - 1);

    /// Pixel index in src batch.
    const int src_ind_00 = src_y0 * src_frame_width + src_x0;
    const int src_ind_01 = src_y1 * src_frame_width + src_x0;
    const int src_ind_10 = src_y0 * src_frame_width + src_x1;
    const int src_ind_11 = src_y1 * src_frame_width + src_x1;

    float dst_val = src_mask_frame[src_ind_11] * (src_x - src_x0) * (src_y - src_y0) + \
        src_mask_frame[src_ind_10] * (src_x - src_x0) * (src_y1 - src_y) + \
        src_mask_frame[src_ind_01] * (src_x1 - src_x) * (src_y - src_y0) + \
        src_mask_frame[src_ind_00] * (src_x1 - src_x) * (src_y1 - src_y);

    dst_mask_patch[i] = (dst_val>=0);
}

/**
 * @brief Permute tensor dimensions on the GPU.
 * @param[in] src Source tensor.
 * @param[in] src_sizes Source tensor dimensions.
 * @param[in] dst_dim_order Destination dimension order.
 * @param[in] nbDims Number of tensor dimensions.
 * @param[out] dst Destination tensor to populate.
 */
static __global__ void cuda_permuteTensor(
    const float* src, const int* src_sizes, const int* dst_dim_order, const int nbDims, float *dst
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    int stride_element_size = 1;
    for (int di=0; di < nbDims; di++)
    {
        stride_element_size *= src_sizes[di];
    }
    /// The accumulated stride is the total tensor element count.
    if (i >= stride_element_size)
        return;

    int src_ind = i;
    int src_dim_inds[MAX_NBDIMS];

    for (int di=0; di < nbDims; di++)
    {
        /// Compute the remaining stride.
        stride_element_size /= src_sizes[di];
        src_dim_inds[di] = src_ind / stride_element_size;
        src_ind = src_ind % stride_element_size;
    }

    stride_element_size = 1;
    int dst_ind = 0, src_dim;
    for (int di=nbDims-1; di >= 0; di--)
    {
        src_dim = dst_dim_order[di];
        dst_ind += src_dim_inds[src_dim] * stride_element_size;
        stride_element_size *= src_sizes[src_dim];
    }

    dst[dst_ind] = src[i];
}

/**
 * @brief Broadcast-add a vector into a tensor on the GPU.
 * @param[in] src_large Larger input tensor.
 * @param[in] src_small Input vector broadcast across the larger tensor.
 * @param[in] large_size Number of elements in the larger tensor.
 * @param[in] small_size Number of elements in the broadcast vector.
 * @param[out] dst Destination tensor to populate.
 */
static __global__ void cuda_broadcastAdd(
    const float* src_large, const float* src_small, const int large_size, const int small_size, float *dst
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= large_size)
        return;
    dst[i] = src_large[i] + src_small[i%small_size];
}

/**
 * @brief Add temporal codes to positional embeddings on the GPU.
 * @param[in] temporal_code Temporal-code tensor.
 * @param[in] pos_embed Positional-embedding tensor.
 * @param[out] dst Destination tensor to populate.
 * @param[in] num_mask_mem Number of mask-memory entries.
 * @param[in] mask_mem_feat_size Number of values in each mask-memory feature.
 * @param[in] temporal_code_size Number of values in each temporal code.
 */
static __global__ void cuda_addTemporalCode2PosEmbed(
    float* temporal_code, float* pos_embed, float *dst, const int num_mask_mem,
    const int mask_mem_feat_size, const int temporal_code_size
)
{
    /// temporal_code [7 x 64] reverse + pos_embed [7 x 4096 x 64] -> dst [7 x 4096 x 64].
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_mask_mem * mask_mem_feat_size)
        return;

    int mask_mem_ind = i / mask_mem_feat_size;
    /// Reverse mask_mem_ind based on num_mask_mem.
    /// For example, for ind=0, take the temporal code ind=6, for ind=1, take the temporal code ind=5, etc.
    mask_mem_ind = num_mask_mem - 1 - mask_mem_ind;

    int temporal_code_ind = i % temporal_code_size;

    dst[i] = temporal_code[mask_mem_ind * temporal_code_size + temporal_code_ind] + pos_embed[i];
}

/**
 * @brief Launch segmentation-mask rescaling.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] src_mask_frame Source segmentation mask.
 * @param[in] src_frame_width Source mask width.
 * @param[in] src_frame_element_size Number of elements in the source mask.
 * @param[in] dst_width Destination mask width.
 * @param[in] dst_element_size Number of elements in the destination mask.
 * @param[in] src_roi Source region of interest.
 * @param[out] dst_mask_patch Destination mask patch to populate.
 */
void rescaleSegmentationOutput(
    const cudaStream_t &stream, const float *src_mask_frame,
    const int src_frame_width, const int src_frame_element_size, const int dst_width,
    const int dst_element_size, const float4 src_roi, float *dst_mask_patch
)
{
    /// Use the standard one-dimensional CUDA block size.
    const dim3 Threads(CUDA_BLOCK_SIZE_1D);
    /// Each thread writes one target's one mask pixel in all channels.
    const dim3 Blocks((dst_element_size + Threads.x - 1)/Threads.x);

    cuda_rescaleSegmentationOutput<<< Blocks, Threads, 0, stream >>>(
        src_mask_frame, src_frame_width, src_frame_element_size, dst_width, dst_element_size, src_roi, dst_mask_patch
    );
}

/**
 * @brief Launch tensor-dimension permutation.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] number_of_eles Number of tensor elements.
 * @param[in] src Source tensor.
 * @param[in] src_sizes Source tensor dimensions.
 * @param[in] dst_dim_order Destination dimension order.
 * @param[in] nbDims Number of tensor dimensions.
 * @param[out] dst Destination tensor to populate.
 */
void permuteTensor(
    const cudaStream_t &stream, const int number_of_eles, const float* src,
    const int* src_sizes, const int* dst_dim_order, const int nbDims, float *dst
)
{
    /// Use the standard one-dimensional CUDA block size.
    const dim3 Threads(CUDA_BLOCK_SIZE_1D);
    const dim3 Blocks((number_of_eles + Threads.x - 1)/Threads.x);
    cuda_permuteTensor<<< Blocks, Threads, 0, stream >>>(src, src_sizes, dst_dim_order, nbDims, dst);
}

/**
 * @brief Launch broadcast addition for a tensor and vector.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] src_large Larger input tensor.
 * @param[in] src_small Input vector broadcast across the larger tensor.
 * @param[in] large_size Number of elements in the larger tensor.
 * @param[in] small_size Number of elements in the broadcast vector.
 * @param[out] dst Destination tensor to populate.
 */
void broadcastAdd(
    const cudaStream_t &stream, const float* src_large, const float* src_small,
    const int large_size, const int small_size, float *dst
)
{
    const dim3 Threads(CUDA_BLOCK_SIZE_1D);
    const dim3 Blocks((large_size + Threads.x - 1)/Threads.x);
    cuda_broadcastAdd<<< Blocks, Threads, 0, stream >>>(src_large, src_small, large_size, small_size, dst);
}

/**
 * @brief Add temporal codes to positional embeddings.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] temporal_code Temporal-code tensor.
 * @param[in] pos_embed Positional-embedding tensor.
 * @param[out] dst Destination tensor to populate.
 * @param[in] num_mask_mem Number of mask-memory entries.
 * @param[in] mask_mem_feat_size Number of values in each mask-memory feature.
 * @param[in] temporal_code_size Number of values in each temporal code.
 */
void addTemporalCode2PosEmbed(
    const cudaStream_t &stream, float* temporal_code, float* pos_embed, float *dst,
    const int num_mask_mem, const int mask_mem_feat_size, const int temporal_code_size
)
{
    const dim3 Threads(CUDA_BLOCK_SIZE_1D);
    const dim3 Blocks((num_mask_mem * mask_mem_feat_size * temporal_code_size + Threads.x - 1)/Threads.x);
    cuda_addTemporalCode2PosEmbed<<< Blocks, Threads, 0, stream >>>(
        temporal_code, pos_embed, dst, num_mask_mem, mask_mem_feat_size, temporal_code_size
    );
}
} // end of cuSegmenter
