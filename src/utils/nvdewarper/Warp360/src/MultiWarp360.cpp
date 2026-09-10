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
/** @file MultiWarp360.cpp
 *  Sample code to illustrate multiple warps on the same image.
 *  This implements a serial-parallel topology, where the first stage is an optional, and
 *  converts a YUV 4:2:0 NV12 image to RGBA as an optional serial first stage.
 *  The subsequent stages all use the same RGBA image and performs different warps on it.
 *
 */

#include "NVWarp360.h"


/********************************************************************************
 * MultiWarp360 - with array of parameter blocks.
 ********************************************************************************/

nvwarpResult NVWARPAPI nvwarpMultiWarp360(cudaStream_t stream,
    const nvwarpYUVRGBParams_t *yuvParams, const void *yuvBuffer, size_t yuvRowBytes,
    void *rgbBuffer, size_t rgbRowBytes, cudaTextureObject_t rgbTex,
    uint32_t numWarps, const nvwarpParams_t *paramArray, void **dstBuffers, const size_t *dstRowBytes)
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    nvwarpHandle    warper  = nullptr;

    // In this implementation, we create one warp instance to process all warps of one frame.
    // A more efficient version would allocate several warp instances in the caller, i.e.
    // one instance for each warp in the array. The same warp instances would be used for all frames
    // in a video, and would set their parameters only once for the whole video.
    // The improvement in throughput may or may not  be measureable, though.
    err = nvwarpCreateInstance(&warper);
    if (NVWARP_SUCCESS != err)
        goto bail;

    if (yuvParams)
    {
        err = nvwarpConvertYUVNV12ToRGBA(stream, yuvParams, yuvBuffer, yuvRowBytes, rgbBuffer, rgbRowBytes);
        if (NVWARP_SUCCESS != err)
            goto bail;
    }

    for (; numWarps--; ++paramArray, ++dstBuffers, ++dstRowBytes)
    {
        err = nvwarpSetParams(warper, paramArray);
        if (NVWARP_SUCCESS != err)
            goto bail;
        err = nvwarpWarpBuffer(warper, stream, rgbTex, *dstBuffers, *dstRowBytes);
        if (NVWARP_SUCCESS != err)
            goto bail;
    }

bail:
    nvwarpDestroyInstance(warper);

    return err;
}
