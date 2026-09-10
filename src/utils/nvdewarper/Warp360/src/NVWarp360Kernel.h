/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/** @file Warp360Kernel.h
 *  CUDA implementation for the 360 Image and Coordinate Warp SDK.
 */

#ifndef __WARP360_KERNEL_H__
#define __WARP360_KERNEL_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "NVWarp360.h"


/****************************************************************************//**
 * Device Parameters for FishBroom
 ********************************************************************************/

struct Warp360KernelParams
{
    float       srcX0;          /**< Center of projection X */
    float       srcY0;          /**< Center of projection Y */
    float       srcFocLenX;     /**< Source horizontal focal length */
    float       srcFocLenY;     /**< Source  vertical  focal length */
    float       srcDist[5];     /**< Radial (and tangential) distortion in source */
    float       srcRot[9];      /**< Rotation of source */

    unsigned    dstWidth;       /**< Size of output surface in pixels */
    unsigned    dstHeight;      /**< Size of output surface in pixels */
    float       dstX0;          /**< Center of destination image X */
    float       dstY0;          /**< Center of destination image Y */
    float       dstInvFocLenX;  /**< Reciprocal of the destination horizontal focal length */
    float       dstInvFocLenY;  /**< Reciprocal of the destination  vertical   focal length */
    float       control;        /**< Control parameter */
};


/****************************************************************************//**
 * Warp to a surface.
 * @param[in]   dim_grid        The number of blocks in the grid.
 * @param[in]   dim_block       The size of each block in the grid.
 * @param[in]   stream          The stream on which the CUDA kernel should run.
 * @param[in]   params          The parameters describing the projection
 * @param[in]   stream          The CUDA stream to be used for the computation.
 * @param[in]   srcTex          The source texture.
 * @param[out]  dstSurface      Render onto this surface.
 * @return      cudaSuccess     If successful.
 ********************************************************************************/

template<nvwarpSurface_t Tsrc, nvwarpSurface_t Tdst> __host__ cudaError_t Warp(dim3 dim_grid, dim3 dim_block, cudaStream_t stream,
    const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);


/****************************************************************************//**
 * Warp to a pitched buffer.
 * @param[in]   dim_grid        The number of blocks in the grid.
 * @param[in]   dim_block       The size of each block in the grid.
 * @param[in]   stream          The stream on which the CUDA kernel should run.
 * @param[in]   params          The parameters describing the projection
 * @param[in]   stream          The CUDA stream to be used for the computation.
 * @param[in]   srcTex          The source texture.
 * @param[out]  dstAddr         The address of the destination buffer pixel (0,0) on the device.
 * @param[in]   dstRowBytes     The byte stride between pixels vertically.
 * @return      cudaSuccess     If successful.
 ********************************************************************************/

template<nvwarpSurface_t Tsrc, nvwarpSurface_t Tdst> __host__  cudaError_t Warp(dim3 dim_grid, dim3 dim_block, cudaStream_t stream,
    const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);




__host__ void NV12RGBABuffer(cudaStream_t stream, const nvwarpYUVRGBParams_t *params, const unsigned char *yuv, size_t yuvRowBytes, uchar4 *dst, size_t dstRowBytes);



#endif /* __WARP360_KERNEL_H__ */
