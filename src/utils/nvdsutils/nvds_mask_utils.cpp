/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvds_mask_utils.h"

#include <cmath>
#include "npp.h"
#include "nppcore.h"
#include "nppdefs.h"
#include <stdio.h>
#define NVDS_MASK_UTILS_THRES_GPU_IMPL
#ifdef NVDS_MASK_UTILS_THRES_GPU_IMPL
#include "nppi_threshold_and_compare_operations.h"
#include "nppi_geometry_transforms.h"
#endif
#include <pthread.h>

#define CHECK(status)                                   \
{                                                       \
    if ((status) != 0)                                    \
    {                                                   \
        printf ("[%s:%d]failure: status=%d\n", __func__, __LINE__, (status)); \
        ret = false;           \
        goto done; \
    }                                                   \
}

#define CHECK_NPP(status)                                   \
{                                                       \
    if ((status) < NPP_SUCCESS)                                    \
    {                                                   \
        printf ("[%s:%d]failure: status=%d\n", __func__, __LINE__, (status)); \
        ret = false; \
        goto done;    \
    }                                                   \
}

/** Library wide mutex to
 * enforce thread-safety for nvds_mask_utils API
 * This is to avoid observed perf issues with context management
 */
static pthread_mutex_t g_mutex;

/* Library constructor */
void __attribute__((constructor)) libnvds_utils_init(void);
/* Library destructor */
void __attribute__((destructor)) libnvds_utils_deinit(void);

void __attribute__((constructor)) libnvds_utils_init(void)
{
    pthread_mutex_init(&g_mutex, NULL);
}

void __attribute__((destructor)) libnvds_utils_deinit(void)
{
    pthread_mutex_destroy(&g_mutex);
}

static bool mask_utils_create_stream_context(cudaStream_t stream, NppStreamContext* pNppStreamCtx)
{
    bool ret = true;

    // Initialize the stream context
    if (pNppStreamCtx != NULL)
        pNppStreamCtx->hStream = stream;

    // Get device properties
    CHECK(cudaGetDevice(&pNppStreamCtx->nCudaDeviceId));

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, pNppStreamCtx->nCudaDeviceId));

    pNppStreamCtx->nMultiProcessorCount = deviceProp.multiProcessorCount;
    pNppStreamCtx->nMaxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
    pNppStreamCtx->nMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    pNppStreamCtx->nSharedMemPerBlock = deviceProp.sharedMemPerBlock;
    pNppStreamCtx->nCudaDevAttrComputeCapabilityMajor = deviceProp.major;
    pNppStreamCtx->nCudaDevAttrComputeCapabilityMinor = deviceProp.minor;

    // Get stream flags
    unsigned int streamFlags;
    CHECK(cudaStreamGetFlags(stream, &streamFlags));
    pNppStreamCtx->nStreamFlags = streamFlags;

    pNppStreamCtx->nReserved0 = 0;

done:
    return ret;
}

bool nvds_mask_utils_resize_to_binary_argb32(float *src, uint32_t* dst,
                uint32_t src_width, uint32_t src_height,
                uint32_t dst_width, uint32_t dst_height,
                uint32_t channel, float threshold,
                uint32_t argb32_px, uint32_t interpolation,
                cudaStream_t stream)
{
    bool ret = true;
    NppStatus status = NPP_SUCCESS;
    NppStreamContext nppStreamCtx = {0};

    // Declare all variables at the beginning to avoid goto issues
    void* dst_f = nullptr;
    void* dst_u = nullptr;
    float* src_f = nullptr;

    pthread_mutex_lock (&g_mutex);
    ret = mask_utils_create_stream_context(stream, &nppStreamCtx);
    pthread_mutex_unlock (&g_mutex);

    if (!ret) {
        goto done;
    }

    CHECK (cudaMalloc (&dst_f, dst_width * dst_height * channel * sizeof(Npp32f)));

    CHECK (cudaMalloc (&dst_u, dst_width * dst_height * channel * sizeof(Npp32u)));

    CHECK (cudaMalloc (&src_f, src_width * src_height * channel * sizeof(Npp32f)));

    CHECK (cudaMemcpyAsync(src_f, src, src_width * src_height * channel * sizeof(Npp32f), cudaMemcpyDefault, stream));

    status = nppiResizeSqrPixel_32f_C1R_Ctx(
        static_cast<const Npp32f *>(src_f), {int(src_width), int(src_height)},
        src_width * sizeof(Npp32f), {0, 0, int(src_width), int(src_height)},
        static_cast<Npp32f *>(dst_f), dst_width * sizeof(Npp32f),
        {0, 0, int(dst_width), int(dst_height)}, double(dst_width) / double(src_width),
        double(dst_height) / double(src_height), 0.0, 0.0, interpolation,
        nppStreamCtx
        );

    CHECK_NPP (status);

#ifdef NVDS_MASK_UTILS_THRES_GPU_IMPL
    /** If the scaled F32 Tensor pixels > threshold,
     * Set Alpha (transparency=127 ~50%; 0:full-transparency ff:opaque) [mask]
     * and when <= threshold, transparency = 100% (0) [no-mask]
     * NOTE: The following loop is not offloaded to GPU for reasons:
     * a) dst memory pointer is system memory (as actual user of the API needs
     * it on system mem)
     * b) Threshold check induces warp/thread divergence
     * NOTE for nppiThreshold_LTValGTVal_32f_C1R_Ctx() :
     * sourcePixel is less than nThresholdLT is true, pixel is set to nValueLT
     * else if sourcePixel is greater than nThresholdGT pixel is set to nValueGT
     * otherwise it is set to sourcePixel.
     * 
     */
    status = nppiThreshold_LTValGTVal_32f_C1R_Ctx(
        static_cast<Npp32f *>(dst_f), dst_width * sizeof(Npp32f),
        reinterpret_cast<Npp32f *>(dst_u), dst_width * sizeof(Npp32u),
        {int(dst_width), int(dst_height)},
        threshold, /**< nThresholdLT */
        0,
        threshold, /**< nThresholdGT */
        /** nValue: though we pass float, the eventual dest is INT32 array */
        argb32_px,
        nppStreamCtx
        );

    CHECK_NPP (status);

    CHECK (cudaMemcpyAsync(dst, dst_u, dst_width * dst_height * channel * sizeof(Npp32u), cudaMemcpyDefault, stream));

#else
    /** CPU implementation; retained for PERF assessment */
    CHECK (cudaStreamSynchronize(stream) != cudaSuccess);
    uint32_t dst_idx = 0;
    for (uint32_t y = 0; y < dst_height; ++y)
    {
        for (uint32_t x = 0; x < dst_width; ++x)
        {
            for (uint32_t c = 0; c < channel; ++c)
            {
                // H, W, C ordering
                float lerp = ((float*)dst_f)[dst_idx];
                if (lerp > threshold)
                    dst[dst_idx++] = 127 << 24;
                else
                    dst[dst_idx++] = 0;
            }
        }
    }
#endif

done:
    if (dst_u) {
        cudaFree(dst_u);
    }
    if (dst_f) {
        cudaFree(dst_f);
    }
    if (src_f) {
        cudaFree(src_f);
    }

    return ret;
}

#if 0
bool nvds_mask_utils_resize_to_binary_uint8(float *src, uint8_t* dst,
                uint32_t src_width, uint32_t src_height,
                uint32_t dst_width, uint32_t dst_height,
                uint32_t channel, float threshold,
                uint32_t interpolation,
                cudaStream_t stream)
{
    bool ret = true;
    NppStatus status = NPP_SUCCESS;

    pthread_mutex_lock (&g_mutex);
    cudaStream_t prev_stream = nppGetStream();
    pthread_mutex_unlock (&g_mutex);

    void* dst_f = nullptr;
    float* src_f = nullptr;

    CHECK (cudaMalloc (&dst_f, dst_width * dst_height * channel * sizeof(Npp32f)));

    CHECK (cudaMalloc (&src_f, src_width * src_height * channel * sizeof(Npp32f)));

    CHECK (cudaMemcpyAsync(src_f, src, src_width * src_height * channel * sizeof(Npp32f), cudaMemcpyDefault, stream ));

    CHECK ( !mask_utils_set_stream(stream, prev_stream));

    status = nppiResizeSqrPixel_32f_C1R(
        static_cast<const Npp32f *>(src_f), {int(src_width), int(src_height)},
        src_width * sizeof(Npp32f), {0, 0, int(src_width), int(src_height)},
        static_cast<Npp32f *>(dst_f), dst_width * sizeof(Npp32f),
        {0, 0, int(dst_width), int(dst_height)}, double(dst_width) / double(src_width),
        double(dst_height) / double(src_height), 0.0, 0.0, interpolation
        );

    CHECK (cudaStreamSynchronize(stream) != cudaSuccess);

    CHECK ( !mask_utils_set_stream(prev_stream, stream));

    CHECK_NPP (status);

    /** If the scaled F32 Tensor pixels > threshold,
     * Set Alpha (transparency=127 ~50%; 0:full-transparency ff:opaque) [mask]
     * and when <= threshold, transparency = 100% (0) [no-mask]
     * NOTE: The following loop is not offloaded to GPU for reasons:
     * a) dst memory pointer is system memory (as actual user of the API needs
     * it on system mem)
     * b) Threshold check induces warp/thread divergence
     */
    uint32_t dst_idx = 0;
    for (uint32_t y = 0; y < dst_height; ++y)
    {
        for (uint32_t x = 0; x < dst_width; ++x)
        {
            for (uint32_t c = 0; c < channel; ++c)
            {
                // H, W, C ordering
                float lerp = ((float*)dst_f)[dst_idx];
                if (lerp > threshold)
                    dst[dst_idx++] = 1;
                else
                    dst[dst_idx++] = 0;
            }
        }
    }

done:
    if (dst_f) {
        cudaFree(dst_f);
    }
    if (src_f) {
        cudaFree(src_f);
    }

    return ret;
}
#endif
