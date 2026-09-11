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

#include "motion_detector.h"
#include <nvOpticalFlowCuda.h>
#include <nvOpticalFlowCommon.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <dlfcn.h>

// Function pointer type for NvOFAPICreateInstanceCuda
typedef NV_OF_STATUS (NVOFAPI *PfnNvOFAPICreateInstanceCuda)(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST *cudaOf);

// Dynamic library handle and function pointer
static void *g_nvof_module = NULL;
static PfnNvOFAPICreateInstanceCuda g_pfnNvOFAPICreateInstanceCuda = NULL;

// Load optical flow library dynamically (similar to gst-nvof plugin)
static int loadOpticalFlowLibrary(void) {
    if (g_pfnNvOFAPICreateInstanceCuda != NULL) {
        // Already loaded
        return 0;
    }

    // Try to load the optical flow library
    g_nvof_module = dlopen("libnvidia-opticalflow.so.1", RTLD_LAZY);
    if (g_nvof_module == NULL) {
        printf("Error: Failed to load libnvidia-opticalflow.so.1: %s\n", dlerror());
        return -1;
    }

    // Get the function pointer
    g_pfnNvOFAPICreateInstanceCuda = (PfnNvOFAPICreateInstanceCuda)dlsym(g_nvof_module, "NvOFAPICreateInstanceCuda");
    if (g_pfnNvOFAPICreateInstanceCuda == NULL) {
        printf("Error: Failed to get NvOFAPICreateInstanceCuda: %s\n", dlerror());
        dlclose(g_nvof_module);
        g_nvof_module = NULL;
        return -1;
    }

    return 0;
}

/**
 * Per-instance motion detector context
 * Holds all optical flow resources for thread-safe multi-instance operation
 */
struct MotionDetectorContext_s {
    // Optical flow API and handle
    NV_OF_CUDA_API_FUNCTION_LIST cudaOf;
    NvOFHandle hOF;
    
    // CUDA resources
    CUcontext cuCtx;
    CUstream cuStream;  // Dedicated CUDA stream for OF operations
    
    // Frame dimensions
    int width;
    int height;
    uint32_t outWidth;
    uint32_t outHeight;
    
    // Reusable GPU buffers
    NvOFGPUBufferHandle hIn0;
    NvOFGPUBufferHandle hIn1;
    NvOFGPUBufferHandle hFlow;
};

#define OF_CHECK(call, msg)                                                      \
    do {                                                                         \
        NV_OF_STATUS _s = call;                                                  \
        if (_s != NV_OF_SUCCESS) {                                               \
            printf("Optical Flow error %d at %s:%d - %s\n",                      \
                   _s, __FILE__, __LINE__, msg);                                 \
            return -1;                                                           \
        }                                                                        \
    } while (0)

#define CUDA_DRV_CHECK_SIMPLE(call)                                              \
    do {                                                                         \
        CUresult _status = call;                                                 \
        if (_status != CUDA_SUCCESS) {                                           \
            printf("CUDA error %d at %s:%d\n", _status, __FILE__, __LINE__);    \
            return -1;                                                           \
        }                                                                        \
    } while (0)

const char* motionLevelToString(MotionLevel level) {
    switch (level) {
        case MOTION_STATIC:    return "STATIC";
        case MOTION_VERY_LOW:  return "VERY_LOW";
        case MOTION_LOW:       return "LOW";
        case MOTION_MEDIUM:    return "MEDIUM";
        case MOTION_HIGH:      return "HIGH";
        case MOTION_VERY_HIGH: return "VERY_HIGH";
        case MOTION_EXTREME:   return "EXTREME";
        default:               return "UNKNOWN";
    }
}

// Sampling gap (in frames) the original displacement thresholds were tuned against.
// Thresholds are divided by this so classification operates on per-frame velocity,
// making the result independent of the configured optical_flow_interval while
// preserving the historical behavior when optical_flow_interval == 15.
#define MOTION_REFERENCE_INTERVAL 15.0f

// Classify per-frame motion velocity (pixels/frame), scale-adjusted for resolution.
// `magnitudePerFrame` is the optical-flow displacement normalized to per-SOURCE-frame
// velocity (see analyzeChunkMotion: timestamp-based normalization). Base thresholds are
// for 480p (640x480) reference; classifyMotion multiplies by `scale` (=diag/200) for the
// actual resolution.
//
// RECALIBRATED for true per-source-frame motion. The original ladder
// (0.3/1.0/4/10/20/40 over 15 frames) was tuned against magnitudes that were inflated by
// the input decimation factor (~2x for the 150-in-10s RTVI pool). After the timestamp
// fix, magnitudes are physically correct (~2x smaller), so the old ladder pushed almost
// everything to STATIC/VERY_LOW. These bases are re-scaled to the measured per-source-
// frame range (warehouse-cam content: avg ~0.03-0.35 px/frame @ scale 2.75, peaks ~3),
// so genuinely-active chunks reach LOW/MEDIUM/HIGH and pull frames, while truly-static
// chunks (e.g. 365 ~0.033) stay VERY_LOW and floor. Effective px/frame @ scale=2.75 shown.
static MotionLevel classifyMotion(double magnitudePerFrame, float scale) {
    // base/15*scale ; effective@2.75 in comments
    float staticThresh   = (0.15f / MOTION_REFERENCE_INTERVAL) * scale;  // ~0.0275 px/f
    float veryLowThresh  = (0.90f / MOTION_REFERENCE_INTERVAL) * scale;  // ~0.165  px/f
    float lowThresh      = (1.50f / MOTION_REFERENCE_INTERVAL) * scale;  // ~0.275  px/f
    float mediumThresh   = (3.50f / MOTION_REFERENCE_INTERVAL) * scale;  // ~0.642  px/f
    float highThresh     = (8.00f / MOTION_REFERENCE_INTERVAL) * scale;  // ~1.467  px/f
    float veryHighThresh = (16.0f / MOTION_REFERENCE_INTERVAL) * scale;  // ~2.933  px/f

    if (magnitudePerFrame < staticThresh)   return MOTION_STATIC;
    if (magnitudePerFrame < veryLowThresh)  return MOTION_VERY_LOW;
    if (magnitudePerFrame < lowThresh)      return MOTION_LOW;
    if (magnitudePerFrame < mediumThresh)   return MOTION_MEDIUM;
    if (magnitudePerFrame < highThresh)     return MOTION_HIGH;
    if (magnitudePerFrame < veryHighThresh) return MOTION_VERY_HIGH;
    return MOTION_EXTREME;
}

int initMotionDetector(CUcontext cuda_ctx, int width, int height, MotionDetectorHandle *handle) {
    
    if (!handle) {
        printf("Error: handle pointer is NULL\n");
        return -1;
    }
    
    // Allocate context structure
    MotionDetectorContext_s *ctx = (MotionDetectorContext_s*)malloc(sizeof(MotionDetectorContext_s));
    if (!ctx) {
        printf("Error: Failed to allocate motion detector context\n");
        return -1;
    }
    memset(ctx, 0, sizeof(MotionDetectorContext_s));
    
    ctx->width = width;
    ctx->height = height;
    
    // Use provided context or current context
    if (cuda_ctx) {
        ctx->cuCtx = cuda_ctx;
    } else {
        CUresult res = cuCtxGetCurrent(&ctx->cuCtx);
        if (res != CUDA_SUCCESS || !ctx->cuCtx) {
            printf("Error: No CUDA context available\n");
            free(ctx);
            return -1;
        }
    }
    
    // Create dedicated CUDA stream for optical flow operations
    CUresult res = cuStreamCreate(&ctx->cuStream, CU_STREAM_NON_BLOCKING);
    if (res != CUDA_SUCCESS) {
        printf("Error: Failed to create CUDA stream\n");
        free(ctx);
        return -1;
    }
    
    // Load optical flow library dynamically
    if (loadOpticalFlowLibrary() != 0) {
        printf("Error: Failed to load optical flow library\n");
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }

    // Get optical flow API via dynamically loaded function
    NV_OF_STATUS status = g_pfnNvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &ctx->cudaOf);
    if (status != NV_OF_SUCCESS) {
        printf("Failed to create Optical Flow API instance: %d\n", status);
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }
    
    // Create optical flow handle
    status = ctx->cudaOf.nvCreateOpticalFlowCuda(ctx->cuCtx, &ctx->hOF);
    if (status != NV_OF_SUCCESS) {
        printf("Error: Failed to create OF handle\n");
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }
    
    // Initialize parameters (optimized for static camera)
    NV_OF_INIT_PARAMS initParams = {};
    initParams.width              = width;
    initParams.height             = height;
    initParams.enableExternalHints = NV_OF_FALSE;
    initParams.enableOutputCost    = NV_OF_FALSE;
    initParams.outGridSize         = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    initParams.hintGridSize        = NV_OF_HINT_VECTOR_GRID_SIZE_4;
    initParams.mode                = NV_OF_MODE_OPTICALFLOW;
    initParams.perfLevel           = NV_OF_PERF_LEVEL_FAST;  // Fast for realtime
    initParams.enableRoi           = NV_OF_FALSE;
    initParams.inputBufferFormat   = NV_OF_BUFFER_FORMAT_NV12;
    initParams.enableGlobalFlow    = NV_OF_FALSE;
    initParams.predDirection       = NV_OF_PRED_DIRECTION_FORWARD;
    
    status = ctx->cudaOf.nvOFInit(ctx->hOF, &initParams);
    if (status != NV_OF_SUCCESS) {
        printf("Error: Failed to initialize OF\n");
        ctx->cudaOf.nvOFDestroy(ctx->hOF);
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }

    /* Bind OF input/output to our dedicated stream so nvOFExecute is ordered ON
     * ctx->cuStream. Without this, OF runs on its own (default) stream and the
     * post-execute cuStreamSynchronize(ctx->cuStream) does NOT wait for it (the
     * stream is CU_STREAM_NON_BLOCKING) — a data race that corrupts flow output
     * and makes motion classification nondeterministic. With the streams bound,
     * the existing async D2H download + cuStreamSynchronize(ctx->cuStream) is a
     * correct and sufficient barrier, with no device-wide cuCtxSynchronize. */
    if (ctx->cudaOf.nvOFSetIOCudaStreams) {
        status = ctx->cudaOf.nvOFSetIOCudaStreams(ctx->hOF, ctx->cuStream, ctx->cuStream);
        if (status != NV_OF_SUCCESS) {
            printf("Error: Failed to set OF IO CUDA streams\n");
            ctx->cudaOf.nvOFDestroy(ctx->hOF);
            cuStreamDestroy(ctx->cuStream);
            free(ctx);
            return -1;
        }
    } else {
        printf("Warning: nvOFSetIOCudaStreams unavailable; relying on NVDS_OF_FORCE_CTX_SYNC\n");
    }

    // Calculate output dimensions
    // Grid size 4 optimized for 480p (640x480 → 160x120 output)
    // Note: NVIDIA Optical Flow only supports grid sizes 1, 2, 4
    uint32_t outGridSize = 4;
    ctx->outWidth  = (width + outGridSize - 1) / outGridSize;
    ctx->outHeight = (height + outGridSize - 1) / outGridSize;
    
    // Create persistent GPU buffers
    NV_OF_BUFFER_DESCRIPTOR inBufDesc = {};
    inBufDesc.width = width;
    inBufDesc.height = height;
    inBufDesc.bufferFormat = NV_OF_BUFFER_FORMAT_NV12;
    inBufDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    
    NV_OF_BUFFER_DESCRIPTOR outBufDesc = {};
    outBufDesc.width = ctx->outWidth;
    outBufDesc.height = ctx->outHeight;
    outBufDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    outBufDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    
    status = ctx->cudaOf.nvOFCreateGPUBufferCuda(ctx->hOF, &inBufDesc, 
             NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &ctx->hIn0);
    if (status != NV_OF_SUCCESS) {
        printf("Error: Failed to create input buffer 0\n");
        ctx->cudaOf.nvOFDestroy(ctx->hOF);
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }
    
    status = ctx->cudaOf.nvOFCreateGPUBufferCuda(ctx->hOF, &inBufDesc,
             NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &ctx->hIn1);
    if (status != NV_OF_SUCCESS) {
        printf("Error: Failed to create input buffer 1\n");
        ctx->cudaOf.nvOFDestroyGPUBufferCuda(ctx->hIn0);
        ctx->cudaOf.nvOFDestroy(ctx->hOF);
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }
    
    status = ctx->cudaOf.nvOFCreateGPUBufferCuda(ctx->hOF, &outBufDesc,
             NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &ctx->hFlow);
    if (status != NV_OF_SUCCESS) {
        printf("Error: Failed to create output buffer\n");
        ctx->cudaOf.nvOFDestroyGPUBufferCuda(ctx->hIn0);
        ctx->cudaOf.nvOFDestroyGPUBufferCuda(ctx->hIn1);
        ctx->cudaOf.nvOFDestroy(ctx->hOF);
        cuStreamDestroy(ctx->cuStream);
        free(ctx);
        return -1;
    }
    
    *handle = ctx;
    return 0;
}

void cleanupMotionDetector(MotionDetectorHandle handle) {
    if (!handle) {
        return;
    }
    
    MotionDetectorContext_s *ctx = (MotionDetectorContext_s*)handle;
    
    if (ctx->hIn0) {
        ctx->cudaOf.nvOFDestroyGPUBufferCuda(ctx->hIn0);
        ctx->hIn0 = nullptr;
    }
    if (ctx->hIn1) {
        ctx->cudaOf.nvOFDestroyGPUBufferCuda(ctx->hIn1);
        ctx->hIn1 = nullptr;
    }
    if (ctx->hFlow) {
        ctx->cudaOf.nvOFDestroyGPUBufferCuda(ctx->hFlow);
        ctx->hFlow = nullptr;
    }
    if (ctx->hOF) {
        ctx->cudaOf.nvOFDestroy(ctx->hOF);
        ctx->hOF = nullptr;
    }
    if (ctx->cuStream) {
        cuStreamDestroy(ctx->cuStream);
        ctx->cuStream = nullptr;
    }
    
    free(ctx);
}

// Analyze single frame pair motion (device memory optimized)
/* An OF grid cell counts as "moving" (a real, coherent displacement rather than
 * compression noise) if its magnitude exceeds this (scale-adjusted, px over the
 * sampled gap). Counting such cells finds a small/distant moving subject that the
 * whole-frame mean dilutes below the noise floor. */
#define MOVING_CELL_MAG_THRESH 33.04f   /* per-cell "is moving" displacement threshold, in px @1080p
                                         * (referenced to the 1080p OF grid, consistent with the count
                                         * thresholds). Scaled by grid-diagonal/1080p-diagonal at runtime.
                                         * (Equivalent to the former 12.0 @480p-reference.) */
/* Min number of coherent moving cells in one frame-pair to call it real localized
 * motion (rescues a small/distant subject the whole-frame mean misses). */
#define MOVING_CELL_MIN_COUNT 2000

/* Calibration gap (seconds) at which MOVING_CELL_MAG_THRESH / extreme threshold were
 * tuned: 2 fps == 0.5 s between the two sampled frames. The "is a cell moving" test is
 * a DISPLACEMENT-over-a-gap, so it is only meaningful relative to that gap. To make it
 * decimation/framerate-invariant we scale the px thresholds by (actual_gap / this), i.e.
 * we hold a fixed physical SPEED bar (33.04 px / 0.5 s == 66 px/s @1080p). At the 0.5 s
 * calibration gap the factor is exactly 1.0, so 2 fps runs are unchanged. */
#define OF_CAL_GAP_SEC 0.5f

static int analyzeSingleMotion(MotionDetectorContext_s *ctx, NvBufSurface *frame0, NvBufSurface *frame1, double *meanMag, double *maxMag, bool *isSceneCut, int *out_moving_cells, double gap_factor) {
    if (!ctx || !frame0 || !frame1) {
        printf("Invalid params for motion analysis\n");
        return -1;
    }
    
    // Get device pointers from NvBufSurface (already on GPU!)
    CUdeviceptr src_frame0 = (CUdeviceptr)frame0->surfaceList[0].dataPtr;
    CUdeviceptr src_frame1 = (CUdeviceptr)frame1->surfaceList[0].dataPtr;
    
    // Get OF buffer pointers
    CUdeviceptr d_in0 = ctx->cudaOf.nvOFGPUBufferGetCUdeviceptr(ctx->hIn0);
    CUdeviceptr d_in1 = ctx->cudaOf.nvOFGPUBufferGetCUdeviceptr(ctx->hIn1);
    CUdeviceptr d_flow = ctx->cudaOf.nvOFGPUBufferGetCUdeviceptr(ctx->hFlow);
    
    // Get stride info
    NV_OF_CUDA_BUFFER_STRIDE_INFO strideInfo0 = {}, strideInfo1 = {};
    OF_CHECK(ctx->cudaOf.nvOFGPUBufferGetStrideInfo(ctx->hIn0, &strideInfo0), "get stride 0");
    OF_CHECK(ctx->cudaOf.nvOFGPUBufferGetStrideInfo(ctx->hIn1, &strideInfo1), "get stride 1");
    
    // Get source pitch from NvBufSurface
    uint32_t src_pitch = frame0->surfaceList[0].pitch;
    uint32_t src_height = frame0->surfaceList[0].height;
    
    // Copy frame 0 to OF buffer using dedicated stream (device-to-device, minimal overhead)
    CUDA_MEMCPY2D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    
    // Frame 0 Y plane
    copyParams.srcDevice = src_frame0;
    copyParams.srcPitch = src_pitch;
    copyParams.dstDevice = d_in0;
    copyParams.dstPitch = strideInfo0.strideInfo[0].strideXInBytes;
    copyParams.WidthInBytes = ctx->width;
    copyParams.Height = src_height;
    CUDA_DRV_CHECK_SIMPLE(cuMemcpy2DAsync(&copyParams, ctx->cuStream));
    
    // Frame 0 UV plane
    copyParams.srcDevice = src_frame0 + src_pitch * src_height;
    copyParams.dstDevice = d_in0 + strideInfo0.strideInfo[0].strideYInBytes;
    copyParams.Height = (src_height + 1) / 2;
    CUDA_DRV_CHECK_SIMPLE(cuMemcpy2DAsync(&copyParams, ctx->cuStream));
    
    // Frame 1 Y plane
    copyParams.srcDevice = src_frame1;
    copyParams.srcPitch = src_pitch;
    copyParams.dstDevice = d_in1;
    copyParams.dstPitch = strideInfo1.strideInfo[0].strideXInBytes;
    copyParams.WidthInBytes = ctx->width;
    copyParams.Height = src_height;
    CUDA_DRV_CHECK_SIMPLE(cuMemcpy2DAsync(&copyParams, ctx->cuStream));
    
    // Frame 1 UV plane
    copyParams.srcDevice = src_frame1 + src_pitch * src_height;
    copyParams.dstDevice = d_in1 + strideInfo1.strideInfo[0].strideYInBytes;
    copyParams.Height = (src_height + 1) / 2;
    CUDA_DRV_CHECK_SIMPLE(cuMemcpy2DAsync(&copyParams, ctx->cuStream));
    CUDA_DRV_CHECK_SIMPLE(cuStreamSynchronize(ctx->cuStream));
    
    // Execute optical flow
    NV_OF_EXECUTE_INPUT_PARAMS execIn = {};
    NV_OF_EXECUTE_OUTPUT_PARAMS execOut = {};
    
    execIn.inputFrame = ctx->hIn0;
    execIn.referenceFrame = ctx->hIn1;
    execIn.disableTemporalHints = NV_OF_TRUE;
    
    execOut.outputBuffer = ctx->hFlow;
    
    OF_CHECK(ctx->cudaOf.nvOFExecute(ctx->hOF, &execIn, &execOut), "execute OF");

    /* CORRECTNESS: nvOFExecute is bound to ctx->cuStream (nvOFSetIOCudaStreams at
     * init), so the flow computation is enqueued ON ctx->cuStream, ordered before
     * the async D2H download below which uses the same stream. The subsequent
     * cuStreamSynchronize(ctx->cuStream) therefore waits for BOTH the OF kernel and
     * the download — a correct and sufficient barrier with no device-wide stall.
     *
     * Previously this did an unconditional cuCtxSynchronize() — a DEVICE-WIDE
     * barrier draining EVERY CUDA stream (decoder, nvvideoconvert, SAD) once per OF
     * sample (~num_frames/interval ≈ 30x per chunk). Without the stream binding,
     * removing it caused a data race (OF on its own stream, download on a
     * NON_BLOCKING stream → cuStreamSynchronize did not wait for OF) that made
     * motion classification nondeterministic. The stream binding fixes the race
     * while keeping the perf win.
     *
     * Escape hatch: NVDS_OF_FORCE_CTX_SYNC=1 restores the device-wide sync (e.g.
     * if nvOFSetIOCudaStreams is unavailable on an older SDK). */
    static const bool of_force_ctx_sync =
        (getenv("NVDS_OF_FORCE_CTX_SYNC") != nullptr);
    if (of_force_ctx_sync) {
        CUDA_DRV_CHECK_SIMPLE(cuCtxSynchronize());
    }

    // Download flow results (small amount of data - output is downsampled)
    std::vector<NV_OF_FLOW_VECTOR> flow(ctx->outWidth * ctx->outHeight);

    NV_OF_CUDA_BUFFER_STRIDE_INFO strideInfoFlow = {};
    OF_CHECK(ctx->cudaOf.nvOFGPUBufferGetStrideInfo(ctx->hFlow, &strideInfoFlow), "get flow stride");

    CUDA_MEMCPY2D downloadParams = {};
    downloadParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    downloadParams.dstMemoryType = CU_MEMORYTYPE_HOST;
    downloadParams.srcDevice = d_flow;
    downloadParams.srcPitch = strideInfoFlow.strideInfo[0].strideXInBytes;
    downloadParams.dstHost = flow.data();
    downloadParams.dstPitch = ctx->outWidth * sizeof(NV_OF_FLOW_VECTOR);
    downloadParams.WidthInBytes = ctx->outWidth * sizeof(NV_OF_FLOW_VECTOR);
    downloadParams.Height = ctx->outHeight;
    /* Async D2H copy on dedicated stream — avoids null-stream serialization which
     * would stall all other CUDA streams (decoder, encoder, nvvideoconvert) in the
     * process. OF results are guaranteed ready by cuCtxSynchronize() above. */
    CUDA_DRV_CHECK_SIMPLE(cuMemcpy2DAsync(&downloadParams, ctx->cuStream));
    CUDA_DRV_CHECK_SIMPLE(cuStreamSynchronize(ctx->cuStream));
    
    // Analyze flow vectors
    size_t numPixels = ctx->outWidth * ctx->outHeight;
    double sumMag = 0.0;
    double maxMagnitude = 0.0;
    size_t extremeCount = 0;
    
    // Resolution scale factor, referenced to 1080p (consistent with the moving-cell COUNT
    // thresholds in the selector). 1080p (1920x1080) with grid=4 → 480x270 grid →
    // diagonal = sqrt(480^2 + 270^2) ≈ 550.73. scale == 1.0 at 1080p.
    float diagonal = std::sqrt(static_cast<float>(ctx->outWidth * ctx->outWidth + ctx->outHeight * ctx->outHeight));
    float refDiagonal = std::sqrt(480.0f * 480.0f + 270.0f * 270.0f);  // 1080p grid diagonal (~550.73)
    float scale = diagonal / refDiagonal;

    // Gap-normalize the displacement thresholds so "is a cell moving" is a fixed physical
    // SPEED, not a fixed displacement. gap_factor = actual_gap_sec / OF_CAL_GAP_SEC (==1.0
    // at the 2 fps / 0.5 s calibration point). This keeps the test consistent across input
    // framerates / decimation (2 fps .. 60 fps): a cell must move faster than 66 px/s @1080p
    // to count, regardless of how far apart the two sampled frames are in time.
    float gf = (gap_factor > 1e-6) ? (float)gap_factor : 1.0f;
    float extremeThresh = 137.68f * scale * gf;   // ~137 px @1080p at the 0.5 s calibration gap
    float movingCellThresh = MOVING_CELL_MAG_THRESH * scale * gf;
    size_t movingCells = 0;

    for (size_t i = 0; i < numPixels; ++i) {
        float fx = flow[i].flowx / 32.0f;  // S10.5 format
        float fy = flow[i].flowy / 32.0f;
        float mag = std::sqrt(fx * fx + fy * fy);

        sumMag += mag;
        if (mag > maxMagnitude) maxMagnitude = mag;
        if (mag > extremeThresh) extremeCount++;
        /* Moving cell (real subject), but not an extreme/scene-cut cell. */
        if (mag > movingCellThresh && mag <= extremeThresh) movingCells++;
    }

    *meanMag = sumMag / numPixels;
    *maxMag = maxMagnitude;

    /* Localized-motion signal reported to the selector. DEFAULT: the LARGEST connected-
     * component ("blob") of moving cells — a real subject is one big contiguous blob,
     * whereas repetitive-texture / compression noise is many tiny SCATTERED blobs, so
     * largest-blob rejects that noise. Set env NVDS_OF_BLOB=0 to revert to the raw
     * moving-cell count. NVDS_OF_COH_DEBUG=1 additionally prints per-pair blob stats. */
    int report_mc = (int) movingCells;
    static int blob_mode = -1, coh_debug = -1;
    if (blob_mode < 0) { const char *e = getenv("NVDS_OF_BLOB");      blob_mode = (e && e[0]=='0') ? 0 : 1; }  /* default ON; NVDS_OF_BLOB=0 disables */
    if (coh_debug < 0) { const char *e = getenv("NVDS_OF_COH_DEBUG"); coh_debug = (e && e[0]=='1') ? 1 : 0; }
    if ((blob_mode || coh_debug) && movingCells > 0) {
        int W = (int)ctx->outWidth, H = (int)ctx->outHeight;
        std::vector<char> mv(numPixels, 0);
        double sux = 0.0, suy = 0.0;
        for (size_t i = 0; i < numPixels; ++i) {
            float fx = flow[i].flowx / 32.0f, fy = flow[i].flowy / 32.0f;
            float mag = std::sqrt(fx*fx + fy*fy);
            if (mag > movingCellThresh && mag <= extremeThresh) { mv[i] = 1; sux += fx/mag; suy += fy/mag; }
        }
        std::vector<int> stk; stk.reserve(numPixels);
        int ncomp = 0, largest = 0;
        for (int idx = 0; idx < (int)numPixels; ++idx) {
            if (mv[idx] != 1) continue;
            ncomp++; int sz = 0; stk.clear(); stk.push_back(idx); mv[idx] = 2;
            while (!stk.empty()) {
                int c = stk.back(); stk.pop_back(); sz++;
                int x = c % W, y = c / W;
                int nb[4] = { (x>0?c-1:-1), (x<W-1?c+1:-1), (y>0?c-W:-1), (y<H-1?c+W:-1) };
                for (int k=0;k<4;k++){ int nc=nb[k]; if(nc>=0 && mv[nc]==1){ mv[nc]=2; stk.push_back(nc);} }
            }
            if (sz > largest) largest = sz;
        }
        if (blob_mode) report_mc = largest;   /* use largest-blob as the localized-motion signal */
        if (coh_debug) {
            double coh = movingCells ? std::sqrt(sux*sux + suy*suy) / (double)movingCells : 0.0;
            printf("[OF-COH] mc=%zu comps=%d largest=%d dir_coherence=%.2f\n",
                   movingCells, ncomp, largest, coh);
        }
    }
    if (out_moving_cells) *out_moving_cells = report_mc;

    double extremePercentage = 100.0 * extremeCount / numPixels;
    
    // Scene cut detection (scale-adjusted thresholds)
    float sceneCutThresh = 50.0f * scale;
    *isSceneCut = (*meanMag > sceneCutThresh || extremePercentage > 30.0);
    
    return 0;
}

int analyzeChunkMotion(
    MotionDetectorHandle handle,
    NvBufSurface **cached_frames,
    int num_cached,
    int sample_interval,
    const unsigned long long *timestamps_ns,
    double source_fps,
    ChunkMotionVerdict *verdict,
    MotionLevel *out_sample_levels,
    int *out_sample_moving_cells)
{
    if (!handle || !cached_frames || num_cached < 2 || sample_interval < 1 || !verdict) {
        printf("Invalid parameters for chunk motion analysis\n");
        return -1;
    }
    
    MotionDetectorContext_s *ctx = (MotionDetectorContext_s*)handle;
    
    if (!ctx->hOF) {
        printf("Motion detector not initialized\n");
        return -1;
    }
    
    auto _of_t0 = std::chrono::steady_clock::now();

    memset(verdict, 0, sizeof(ChunkMotionVerdict));
    verdict->minMagnitude = 1e9;
    
    #if 0
    printf("\n╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║          CHUNK MOTION ANALYSIS (Optical Flow)                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("Analyzing %d cached frames with interval %d...\n", num_cached, sample_interval);
    #endif
    // Calculate resolution scale for classification
    float diagonal = std::sqrt(static_cast<float>(ctx->outWidth * ctx->outWidth + ctx->outHeight * ctx->outHeight));
    float refDiagonal = 200.0f;  // Reference for 480p (640x480 with grid=4)
    float scale = diagonal / refDiagonal;
    
    int samplesAnalyzed = 0;
    int validSamplesForAvg = 0;  // Exclude scene cuts from average
    int sceneCutCount = 0;       // Track number of scene cuts
    double sumMagnitude = 0.0;
    
    // Analyze every Nth frame
    for (int i = 0; i < num_cached - sample_interval; i += sample_interval) {
        int idx0 = i;
        int idx1 = i + sample_interval;
        
        if (idx1 >= num_cached) break;
        
        double meanMag = 0.0, maxMag = 0.0;
        bool isSceneCut = false;
        int moving_cells = 0;

        // Real elapsed time between the two sampled frames (from PTS). Used both to
        // gap-normalize the moving-cell SPEED threshold (gap_factor, passed into
        // analyzeSingleMotion) and, below, to normalize the mean magnitude to
        // px-per-source-frame. Timestamp-only, so it is correct under VFR and any input
        // framerate (2..60 fps) without needing source_fps for the speed bar.
        double gap_seconds = 0.0;
        if (timestamps_ns) {
            unsigned long long t0 = timestamps_ns[idx0];
            unsigned long long t1 = timestamps_ns[idx1];
            if (t0 != (unsigned long long)-1 && t1 != (unsigned long long)-1 && t1 > t0)
                gap_seconds = (double)(t1 - t0) / 1e9;
        }
        // Scales the px "is-moving"/extreme thresholds to hold a fixed physical speed.
        // 1.0 == the 0.5 s calibration gap (2 fps) → thresholds unchanged there.
        double gap_factor = (gap_seconds > 1e-6) ? (gap_seconds / (double)OF_CAL_GAP_SEC) : 1.0;

        if (analyzeSingleMotion(ctx, cached_frames[idx0], cached_frames[idx1],
                                 &meanMag, &maxMag, &isSceneCut, &moving_cells, gap_factor) == 0) {
            // meanMag is the total flow displacement across the gap between the two
            // sampled frames. We normalize to px-per-SOURCE-frame so classification is
            // invariant to (a) the configured optical_flow_interval AND (b) any input
            // decimation upstream (e.g. RTVI pre-filters 300 stream frames -> 150, so
            // consecutive received frames are 2 source frames apart).
            //
            // Frame-count division (meanMag / sample_interval) is WRONG under decimation:
            // it yields px-per-received-frame = decimation * px-per-source-frame, inflating
            // the magnitude by the decimation factor. Instead, use the real elapsed time
            // between the two frames (from their PTS) and the true source fps:
            //   source_frames_in_gap = (pts[idx1] - pts[idx0]) seconds * source_fps
            //   perFrameMag          = meanMag / source_frames_in_gap
            // This is exact regardless of how the stream was decimated or what interval
            // was configured. Falls back to sample_interval if timestamps/fps unavailable.
            double gap_frames = (double)sample_interval;
            if (gap_seconds > 1e-6 && source_fps > 0.0) {
                double frames = gap_seconds * source_fps;
                if (frames > 1e-6) gap_frames = frames;
            }
            double perFrameMag = meanMag / gap_frames;

            // Only add to average if NOT a scene cut
            if (!isSceneCut) {
                sumMagnitude += perFrameMag;
                validSamplesForAvg++;
                // Track the peak per-sample velocity so callers can tell a localized
                // motion event (one sample spikes well above the chunk average) from
                // uniform low motion / flicker (all samples near the average).
                if (perFrameMag > verdict->maxSampleMagnitude)
                    verdict->maxSampleMagnitude = perFrameMag;
            }

            if (maxMag > verdict->maxMagnitude) verdict->maxMagnitude = maxMag;
            if (perFrameMag < verdict->minMagnitude) verdict->minMagnitude = perFrameMag;

            MotionLevel level = classifyMotion(perFrameMag, scale);

            /* Localized-subject rescue: a small/distant mover (person, forklift far
             * from a fixed camera) barely moves the whole-frame mean, but lights up a
             * coherent CLUSTER of moving OF cells. If enough cells moved, treat the
             * sample as at least LOW motion so the caller's blend flags it active,
             * even though perFrameMag classified it STATIC/VERY_LOW. */
            if (moving_cells > verdict->maxMovingCells)
                verdict->maxMovingCells = moving_cells;

            /* Per-pair OF level + moving-cell count for the caller's blend. Sample
             * starts at idx0=i => the transition into frame i+sample_interval. The
             * moving-cell activation is done WITHIN-CHUNK-RELATIVE in the caller
             * (robust to each stream's compression-noise floor), not with a fixed
             * absolute threshold here. Scene cuts count as high motion. */
            if (out_sample_levels)
                out_sample_levels[idx0] = isSceneCut ? MOTION_HIGH : level;
            if (out_sample_moving_cells)
                out_sample_moving_cells[idx0] = isSceneCut ? 0 : moving_cells;

#if 1
            if (getenv("NVDS_OF_PERFRAME_DUMP"))
            printf("  PERFRAME sample %d: frames [%d->%d] gap=%.2ff/%.3fs gfac=%.3f  thr=%.1fpx  meanMag=%.3f px  perFrameMag=%.3f px  maxMag=%.2f px  movingCells=%d [%s]%s\n",
                   samplesAnalyzed + 1, idx0, idx1, gap_frames, gap_seconds, gap_factor,
                   MOVING_CELL_MAG_THRESH * (gap_factor > 1e-6 ? gap_factor : 1.0),
                   meanMag, perFrameMag, maxMag,
                   moving_cells, motionLevelToString(level),
                   isSceneCut ? " SCENE CUT" : "");
#endif
            if (level >= MOTION_HIGH) {
                verdict->highMotionCount++;
            } else if (level <= MOTION_LOW) {
                verdict->lowMotionCount++;
            }
            
            if (isSceneCut) {
                verdict->hasSceneCuts = true;
                sceneCutCount++;
            }
            
            samplesAnalyzed++;
        }
    }
    
    verdict->numSamples = samplesAnalyzed;
    verdict->gridTotalCells = (int)(ctx->outWidth * ctx->outHeight);

    if (samplesAnalyzed == 0) {
        printf("No samples could be analyzed\n");
        return -1;
    }
    
    // Calculate aggregated results (excluding scene cuts)
    if (validSamplesForAvg > 0) {
        verdict->avgMagnitude = sumMagnitude / validSamplesForAvg;
        verdict->overallMotionLevel = classifyMotion(verdict->avgMagnitude, scale);
    } else {
        // All samples were scene cuts — this is high visual change, not stillness
        printf("All samples were scene cuts, classifying as HIGH motion\n");
        verdict->avgMagnitude = 0.0;
        verdict->overallMotionLevel = MOTION_HIGH;
    }
    
    // Final verdict: high motion if avg is high or >80% scene cuts
    double sceneCutRatio = (double)sceneCutCount / samplesAnalyzed;
    verdict->isHighMotionVideo = (verdict->overallMotionLevel >= MOTION_HIGH ||
                                   sceneCutRatio > 0.8);
    
#if 0
    // Print summary
    printf("\n┌───────────────────────────────────────────────────────────────────┐\n");
    printf("│ CHUNK MOTION VERDICT                                              │\n");
    printf("├───────────────────────────────────────────────────────────────────┤\n");
    printf("│  Samples analyzed:     %3d / %d frames                           │\n", 
           samplesAnalyzed, num_cached);
    printf("│  Valid for avg:        %3d (scene cuts excluded)                 │\n", 
           validSamplesForAvg);
    printf("│  Avg magnitude:        %.2f px                                    │\n", 
           verdict->avgMagnitude);
    printf("│  Max magnitude:        %.2f px                                    │\n", 
           verdict->maxMagnitude);
    printf("│  Min magnitude:        %.2f px                                    │\n", 
           verdict->minMagnitude);
    printf("│  Overall level:        %-12s                              │\n", 
           motionLevelToString(verdict->overallMotionLevel));
    printf("│  High motion samples:  %d / %d (%.1f%%)                          │\n", 
           verdict->highMotionCount, samplesAnalyzed,
           100.0 * verdict->highMotionCount / samplesAnalyzed);
    printf("│  Low motion samples:   %d / %d (%.1f%%)                          │\n",
           verdict->lowMotionCount, samplesAnalyzed,
           100.0 * verdict->lowMotionCount / samplesAnalyzed);
    printf("│  Scene cuts:           %d / %d (%.1f%%)                          │\n",
           sceneCutCount, samplesAnalyzed,
           100.0 * sceneCutRatio);
    printf("├───────────────────────────────────────────────────────────────────┤\n");
    printf("│  ▶ VERDICT: %-12s VIDEO                                  │\n",
           verdict->isHighMotionVideo ? "HIGH MOTION" : "LOW MOTION");
    if (verdict->isHighMotionVideo && sceneCutRatio > 0.8) {
        printf("│     (High motion due to >80%% scene cuts)                         │\n");
    }
#endif
    
    /* Set NVDS_FRAME_SELECTOR_OF_TIMING=1 to print per-chunk optical-flow timing. */
    if (getenv("NVDS_FRAME_SELECTOR_OF_TIMING")) {
        float elapsedTime = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - _of_t0).count();
        printf("[OF-TIMING] interval=%d samples=%d total=%.2fms per_sample=%.3fms\n",
               sample_interval, samplesAnalyzed, elapsedTime,
               samplesAnalyzed > 0 ? elapsedTime / samplesAnalyzed : 0.0f);
        fflush(stdout);
    }
    
    return 0;
}

