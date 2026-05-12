/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>NVIDIA DeepStream mask utils API</b>
 *
 * @b Description: This file specifies the APIs used to transform mask buffers
 */

#ifndef _NVDS_MAKS_UTILS_H_
#define _NVDS_MAKS_UTILS_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief  resize FP32 Tensor and apply threshold to create INT32 binary tensor
 *         Output INT32 tensor pixels are assumed ARGB32
 *         For resized pixels > threshold, transparency is set to 50%
 *         and other pixels are set to 100% transparent; RGB = 0 for all pixels
 * @param  src [IN/OUT] source FP32 tensor
 * @param  dst [IN/OUT] dst INT32 (ARGB32) tensor
 * @param  src_width [IN] source FP32 tensor width
 * @param  src_height [IN] source FP32 tensor height
 * @param  dst_width [IN] dst INT32 (ARGB32) tensor width
 * @param  dst_height [IN] dst INT32 (ARGB32) tensor height
 * @param  channel [IN]
 * @param  threshold [IN]
 * @param  argb_32 [IN] The pixel value in dst when src pixel > threshold
 * @param  interpolation [IN] The NPP interpolation method to use
 *         Enumeration copied below:
 *         NPPI_INTER_NN =1, Nearest neighbor filtering.
 *         NPPI_INTER_LINEAR Linear interpolation.
 *         NPPI_INTER_CUBIC Cubic interpolation.
 *         NPPI_INTER_CUBIC2P_BSPLINE Two-parameter cubic filter (B=1, C=0)
 *         NPPI_INTER_CUBIC2P_CATMULLROM Two-parameter cubic filter (B=0, C=1/2)
 *         NPPI_INTER_CUBIC2P_B05C03 Two-parameter cubic filter (B=1/2, C=3/10)
 *         NPPI_INTER_SUPER Super sampling.
 *         NPPI_INTER_LANCZOS Lanczos filtering.
 *         NPPI_INTER_LANCZOS3_ADVANCED Generic Lanczos filtering with order 3.
 *         NPPI_SMOOTH_EDGE Smooth edge filtering.
 * @param  stream [IN] The cuda-stream to use for scaling operation on GPU
 */
bool nvds_mask_utils_resize_to_binary_argb32(float *src, uint32_t* dst,
                uint32_t src_width, uint32_t src_height,
                uint32_t dst_width, uint32_t dst_height,
                uint32_t channel, float threshold,
                uint32_t argb32_px, uint32_t interpolation,
                cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
