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

#ifndef CUSEGMENTERLIB_CUSEGMENTER_H_
#define CUSEGMENTERLIB_CUSEGMENTER_H_

#define MAX_NBDIMS 4
namespace cuSegmenter
{
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
);

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
);

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
);

/**
 * @brief Add temporal codes to positional embeddings.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] temporal_code Temporal-code tensor.
 * @param[in] pos_embed Positional-embedding tensor.
 * @param[out] dst Destination tensor to populate.
 * @param[in] num_mask_mem Number of mask-memory entries.
 * @param[in] broadcast_size Number of positions receiving each temporal code.
 * @param[in] temporal_code_size Number of values in each temporal code.
 */
void addTemporalCode2PosEmbed(
    const cudaStream_t &stream, float* temporal_code, float* pos_embed, float *dst,
    const int num_mask_mem, const int broadcast_size, const int temporal_code_size
);
} // end of cuSegmenter

#endif // CUSEGMENTERLIB_CUSEGMENTER_H_
