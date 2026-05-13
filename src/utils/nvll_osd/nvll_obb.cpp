/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief OBB (Oriented Bounding Box) dumping implementation
 *
 * This module handles extraction and JPEG encoding of rotated OBBs.
 * It uses:
 * - NvBufSurfTransform for GPU-accelerated AABB cropping
 * - obb_scanline_rotate_gpu for extracting rotated pixels
 * - nvds_obj_encode for JPEG encoding and file I/O
 */

#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdsmeta.h"
#include "nvds_obj_encode.h"
#include "cuda/obb_scanline.h"
#include "nvll_obb.h"

// Environment variable names
#define ENV_OBB_DUMP_ENABLED        "NVDS_OBB_DUMP_ENABLED"
#define ENV_OBB_DUMP_DIR            "NVDS_OBB_DUMP_DIR"
#define ENV_OBB_DUMP_QUALITY        "NVDS_OBB_DUMP_QUALITY"
#define ENV_OBB_DUMP_FRAME_INTERVAL "NVDS_OBB_DUMP_FRAME_INTERVAL"
#define ENV_OBB_DUMP_MIN_ANGLE      "NVDS_OBB_DUMP_MIN_ANGLE"

// Default values
#define DEFAULT_OBB_DUMP_DIR        "/tmp/obb_dumps"
#define DEFAULT_OBB_DUMP_QUALITY    90
#define DEFAULT_FRAME_INTERVAL      30
#define DEFAULT_MIN_ANGLE           1.0f

// Global singleton context for OBB dumping
static NvOBBDumpCtx *g_obb_dump_singleton = NULL;
static GMutex g_obb_dump_mutex;

void __attribute__((destructor)) nvobb_dump_auto_cleanup(void);

/**
 * @brief OBB dump context structure
 */
struct _NvOBBDumpCtx {
    NvDsObjEncCtxHandle enc_ctx;     // nvds_obj_encode context
    int gpu_id;                       // GPU ID
    cudaStream_t cuda_stream;         // CUDA stream

    // Configuration
    bool enabled;                     // Is OBB dumping enabled?
    char dump_dir[512];               // Output directory
    int quality;                      // JPEG quality (0-100)
    int frame_interval;               // Dump every Nth frame
    float min_angle;                  // Minimum angle threshold (degrees)

    // Frame tracking
    unsigned int frame_count;         // Total frames processed
    unsigned int dump_count;          // Total OBBs dumped

    // Scratch buffers for AABB cropping
    NvBufSurface *aabb_surface;       // RGBA surface for AABB crop
    int aabb_max_width;               // Max AABB width allocated
    int aabb_max_height;              // Max AABB height allocated

    // Scratch buffer for rotated OBB extraction
    uint8_t *d_obb_pixels;            // Device memory for OBB pixels (RGBA)
    int obb_max_width;                // Max OBB width allocated
    int obb_max_height;               // Max OBB height allocated
    size_t obb_buffer_size;           // Size of d_obb_pixels buffer
};

/**
 * @brief Create OBB dump context
 */
NvOBBDumpCtx* nvobb_dump_create_context(int gpu_id, cudaStream_t cuda_stream)
{
    // Check if OBB dumping is enabled
    const char *enabled_str = getenv(ENV_OBB_DUMP_ENABLED);
    if (!enabled_str || atoi(enabled_str) != 1) {
        g_debug("OBB_DUMP: OBB dumping disabled (set %s=1 to enable)",
                ENV_OBB_DUMP_ENABLED);
        return NULL;
    }

    NvOBBDumpCtx *ctx = (NvOBBDumpCtx*)calloc(1, sizeof(NvOBBDumpCtx));
    if (!ctx) {
        g_printerr("OBB_DUMP: Failed to allocate context\n");
        return NULL;
    }

    ctx->gpu_id = gpu_id;
    ctx->cuda_stream = cuda_stream;
    ctx->enabled = true;
    ctx->frame_count = 0;
    ctx->dump_count = 0;

    // Read configuration from environment variables
    const char *dump_dir = getenv(ENV_OBB_DUMP_DIR);
    snprintf(ctx->dump_dir, sizeof(ctx->dump_dir), "%s", 
             dump_dir ? dump_dir : DEFAULT_OBB_DUMP_DIR);

    const char *quality_str = getenv(ENV_OBB_DUMP_QUALITY);
    ctx->quality = quality_str ? atoi(quality_str) : DEFAULT_OBB_DUMP_QUALITY;
    ctx->quality = (ctx->quality < 1) ? 1 : (ctx->quality > 100) ? 100 : ctx->quality;

    const char *interval_str = getenv(ENV_OBB_DUMP_FRAME_INTERVAL);
    ctx->frame_interval = interval_str ? atoi(interval_str) : DEFAULT_FRAME_INTERVAL;
    ctx->frame_interval = (ctx->frame_interval < 1) ? 1 : ctx->frame_interval;

    const char *min_angle_str = getenv(ENV_OBB_DUMP_MIN_ANGLE);
    ctx->min_angle = min_angle_str ? atof(min_angle_str) : DEFAULT_MIN_ANGLE;

    // Validate that output directory exists
    if (!g_file_test(ctx->dump_dir, G_FILE_TEST_IS_DIR)) {
        g_printerr("OBB_DUMP: Error: Directory %s does not exist. Please create it manually.\n", 
                   ctx->dump_dir);
        g_printerr("OBB_DUMP: OBB dumping will be disabled.\n");
        free(ctx);
        return NULL;
    }

    // Create nvds_obj_encode context
    ctx->enc_ctx = nvds_obj_enc_create_context(gpu_id);
    if (!ctx->enc_ctx) {
        g_printerr("OBB_DUMP: Failed to create nvds_obj_encode context\n");
        free(ctx);
        return NULL;
    }

    // Initialize scratch buffers (will be allocated on first use)
    ctx->aabb_surface = NULL;
    ctx->aabb_max_width = 0;
    ctx->aabb_max_height = 0;
    ctx->d_obb_pixels = NULL;
    ctx->obb_max_width = 0;
    ctx->obb_max_height = 0;
    ctx->obb_buffer_size = 0;

    return ctx;
}

/**
 * @brief Destroy OBB dump context
 */
void nvobb_dump_destroy_context(NvOBBDumpCtx *ctx)
{
    if (!ctx) return;

    // Check if CUDA context is still valid before cleanup
    cudaError_t cuda_status = cudaSetDevice(ctx->gpu_id);
    bool cuda_valid = (cuda_status == cudaSuccess);

    if (!cuda_valid) {
        // CUDA context already destroyed, skip CUDA cleanup
        g_debug("OBB_DUMP: CUDA context already destroyed, skipping GPU resource cleanup");
        free(ctx);
        return;
    }

    // Synchronize device to ensure all operations are complete
    cudaDeviceSynchronize();

    // Destroy nvds_obj_encode context
    if (ctx->enc_ctx) {
        nvds_obj_enc_finish(ctx->enc_ctx);
        nvds_obj_enc_destroy_context(ctx->enc_ctx);
        ctx->enc_ctx = NULL;
    }

    if (ctx->aabb_surface) {
        NvBufSurfaceDestroy(ctx->aabb_surface);
        ctx->aabb_surface = NULL;
    }

    if (ctx->d_obb_pixels) {
        cudaError_t err = cudaFree(ctx->d_obb_pixels);
        if (err != cudaSuccess) {
            g_debug("OBB_DUMP: cudaFree failed: %s", cudaGetErrorString(err));
        }
        ctx->d_obb_pixels = NULL;
    }

    free(ctx);
}

/**
 * @brief Get or create singleton OBB dump context
 *
 * This function implements thread-safe lazy initialization of the global
 * OBB dump context. The context is created once and reused across all
 * OSD drawing calls.
 *
 * @param gpu_id GPU device ID
 * @param cuda_stream CUDA stream (0 for default stream)
 * @return Pointer to singleton context, or NULL if disabled/failed
 */
NvOBBDumpCtx* nvobb_dump_get_singleton(int gpu_id, cudaStream_t cuda_stream)
{
    g_mutex_lock(&g_obb_dump_mutex);

    // Create singleton if it doesn't exist
    if (!g_obb_dump_singleton) {
        g_obb_dump_singleton = nvobb_dump_create_context(gpu_id, cuda_stream);

        // Register cleanup handler on first creation
        if (g_obb_dump_singleton) {
            g_debug("OBB_DUMP: Singleton context created (GPU %d)", gpu_id);
        }
    }

    NvOBBDumpCtx *ctx = g_obb_dump_singleton;
    g_mutex_unlock(&g_obb_dump_mutex);

    return ctx;
}

/**
 * @brief Cleanup singleton OBB dump context
 *
 * This function should be called during application shutdown to properly
 * release all OBB dump resources. It's thread-safe and idempotent.
 * Safe to call even if CUDA context is already destroyed.
 */
void nvobb_dump_cleanup_singleton(void)
{
    // Try to lock, but if mutex is already destroyed, just return
    // This can happen during late-stage shutdown
    if (!g_mutex_trylock(&g_obb_dump_mutex)) {
        // If we can't get the lock immediately during shutdown, skip cleanup
        // The OS will reclaim the memory anyway
        return;
    }

    if (g_obb_dump_singleton) {
        g_debug("OBB_DUMP: Cleaning up singleton context (dumped %u OBBs from %u frames)",
                  g_obb_dump_singleton->dump_count,
                  g_obb_dump_singleton->frame_count);

        nvobb_dump_destroy_context(g_obb_dump_singleton);
        g_obb_dump_singleton = NULL;
    }

    g_mutex_unlock(&g_obb_dump_mutex);
}
/**
 * @brief Automatic cleanup function called when library is unloaded
 *
 * This destructor is automatically called when the shared library (libnvds_osd.so)
 * is unloaded from memory. It ensures the OBB dump singleton context is properly
 * cleaned up without requiring explicit calls from the application.
 *
 */
void __attribute__((destructor)) nvobb_dump_auto_cleanup(void)
{
    nvobb_dump_cleanup_singleton();
}

/**
 * @brief Allocate or resize AABB scratch surface
 */
static bool ensure_aabb_surface(NvOBBDumpCtx *ctx, int width, int height)
{
    if (ctx->aabb_surface &&
        ctx->aabb_max_width >= width &&
        ctx->aabb_max_height >= height) {
        return true;
    }

    if (ctx->aabb_surface) {
        NvBufSurfaceDestroy(ctx->aabb_surface);
        ctx->aabb_surface = NULL;
    }

    // Allocate new surface with headroom (round up to 128)
    ctx->aabb_max_width = ((width + 127) / 128) * 128;
    ctx->aabb_max_height = ((height + 127) / 128) * 128;

    NvBufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(NvBufSurfaceCreateParams));
    create_params.gpuId = ctx->gpu_id;
    create_params.width = ctx->aabb_max_width;
    create_params.height = ctx->aabb_max_height;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;

    // Follow nvds_obj_encode pattern: use CUDA_DEVICE for both platforms
    // VIC can work with CUDA_DEVICE memory when using default compute mode
    create_params.memType = NVBUF_MEM_CUDA_DEVICE;

    if (NvBufSurfaceCreate(&ctx->aabb_surface, 1, &create_params) != 0) {
        g_printerr("OBB_DUMP: Failed to create AABB surface %dx%d (memType=%d)\n",
                   ctx->aabb_max_width, ctx->aabb_max_height, create_params.memType);
        return false;
    }

    return true;
}

/**
 * @brief Allocate or resize OBB pixel buffer
 */
static bool ensure_obb_buffer(NvOBBDumpCtx *ctx, int width, int height)
{
    size_t required_size = width * height * 4;  // RGBA

    // Check if current buffer is large enough
    if (ctx->d_obb_pixels && ctx->obb_buffer_size >= required_size) {
        ctx->obb_max_width = width;
        ctx->obb_max_height = height;
        return true;
    }

    // Free old buffer if exists
    if (ctx->d_obb_pixels) {
        cudaFree(ctx->d_obb_pixels);
        ctx->d_obb_pixels = NULL;
    }

    // Allocate new buffer with some headroom
    ctx->obb_max_width = ((width + 127) / 128) * 128;
    ctx->obb_max_height = ((height + 127) / 128) * 128;
    ctx->obb_buffer_size = ctx->obb_max_width * ctx->obb_max_height * 4;

    cudaError_t err = cudaMalloc(&ctx->d_obb_pixels, ctx->obb_buffer_size);
    if (err != cudaSuccess) {
        g_printerr("OBB_DUMP: Failed to allocate OBB buffer %dx%d: %s\n",
                   ctx->obb_max_width, ctx->obb_max_height,
                   cudaGetErrorString(err));
        ctx->obb_buffer_size = 0;
        return false;
    }

    return true;
}

/**
 * @brief Calculate AABB (Axis-Aligned Bounding Box) for a rotated OBB
 */
static void calculate_aabb(
    float cx, float cy, float width, float height, float angle_rad,
    int *aabb_left, int *aabb_top, int *aabb_width, int *aabb_height,
    float *obb_cx_in_aabb, float *obb_cy_in_aabb)
{
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);
    float half_w = width * 0.5f;
    float half_h = height * 0.5f;

    // Calculate 4 corners of rotated OBB
    float corners_x[4], corners_y[4];

    // Corner 0: (-half_w, -half_h)
    corners_x[0] = cx + (-half_w * cos_a - (-half_h) * sin_a);
    corners_y[0] = cy + (-half_w * sin_a + (-half_h) * cos_a);

    // Corner 1: (+half_w, -half_h)
    corners_x[1] = cx + (half_w * cos_a - (-half_h) * sin_a);
    corners_y[1] = cy + (half_w * sin_a + (-half_h) * cos_a);

    // Corner 2: (+half_w, +half_h)
    corners_x[2] = cx + (half_w * cos_a - half_h * sin_a);
    corners_y[2] = cy + (half_w * sin_a + half_h * cos_a);

    // Corner 3: (-half_w, +half_h)
    corners_x[3] = cx + (-half_w * cos_a - half_h * sin_a);
    corners_y[3] = cy + (-half_w * sin_a + half_h * cos_a);

    // Find min/max to get AABB
    float min_x = corners_x[0], max_x = corners_x[0];
    float min_y = corners_y[0], max_y = corners_y[0];

    for (int i = 1; i < 4; i++) {
        if (corners_x[i] < min_x) min_x = corners_x[i];
        if (corners_x[i] > max_x) max_x = corners_x[i];
        if (corners_y[i] < min_y) min_y = corners_y[i];
        if (corners_y[i] > max_y) max_y = corners_y[i];
    }

    *aabb_left = (int)floorf(min_x);
    *aabb_top = (int)floorf(min_y);
    *aabb_width = (int)ceilf(max_x - min_x);
    *aabb_height = (int)ceilf(max_y - min_y);

    // Ensure minimum dimensions (avoid 0 or very small AABBs)
    if (*aabb_width < 2) *aabb_width = 2;
    if (*aabb_height < 2) *aabb_height = 2;

    // Calculate OBB center in AABB space
    *obb_cx_in_aabb = cx - *aabb_left;
    *obb_cy_in_aabb = cy - *aabb_top;
}

/**
 * @brief Dump a single OBB to JPEG file
 */
static bool dump_single_obb(
    NvOBBDumpCtx *ctx,
    NvBufSurface *input_surf,
    float cx, float cy, float width, float height, float angle_rad,
    int obj_idx, unsigned int frame_num)
{
    // Calculate AABB containing the rotated OBB
    int aabb_left, aabb_top, aabb_width, aabb_height;
    float obb_cx_in_aabb, obb_cy_in_aabb;

    calculate_aabb(cx, cy, width, height, angle_rad,
                   &aabb_left, &aabb_top, &aabb_width, &aabb_height,
                   &obb_cx_in_aabb, &obb_cy_in_aabb);

    // Clamp AABB to frame boundaries
    int frame_width = input_surf->surfaceList[0].width;
    int frame_height = input_surf->surfaceList[0].height;

    if (aabb_left < 0) {
        obb_cx_in_aabb += aabb_left;  // Adjust OBB center
        aabb_width += aabb_left;
        aabb_left = 0;
    }
    if (aabb_top < 0) {
        obb_cy_in_aabb += aabb_top;
        aabb_height += aabb_top;
        aabb_top = 0;
    }
    if (aabb_left + aabb_width > frame_width) {
        aabb_width = frame_width - aabb_left;
    }
    if (aabb_top + aabb_height > frame_height) {
        aabb_height = frame_height - aabb_top;
    }

    // Validate AABB
    if (aabb_width <= 0 || aabb_height <= 0) {
        g_printerr("OBB_DUMP: Invalid AABB dimensions: %dx%d\n",
                   aabb_width, aabb_height);
        return false;
    }

    // VIC hardware on Jetson has minimum crop size requirements (16x16)
    // Skip OBBs whose AABB is too small for VIC to handle
    // Also skip if OBB dimensions themselves are too small
    #ifdef __aarch64__
        const int VIC_MIN_CROP_SIZE = 16;
        if (aabb_width < VIC_MIN_CROP_SIZE || aabb_height < VIC_MIN_CROP_SIZE ||
            (int)width < VIC_MIN_CROP_SIZE || (int)height < VIC_MIN_CROP_SIZE) {
            return true;  // Not an error, just skip
        }
    #endif

    // Ensure AABB surface is large enough
    if (!ensure_aabb_surface(ctx, aabb_width, aabb_height)) {
        return false;
    }

    // Step 1: Crop AABB using NvBufSurfTransform
    NvBufSurfTransformConfigParams transform_config;
    memset(&transform_config, 0, sizeof(transform_config));

    // Follow nvds_obj_encode pattern for compute mode selection
    #ifdef __aarch64__
        // Detect if integrated GPU
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, ctx->gpu_id);
        int is_integrated_gpu = prop.integrated;

        if (is_integrated_gpu) {
            // On Jetson integrated GPU, use default mode (VIC hardware)
            transform_config.compute_mode = NvBufSurfTransformCompute_Default;
        } else {
            // On Jetson dGPU mode, use GPU (CUDA)
            transform_config.compute_mode = NvBufSurfTransformCompute_GPU;
        }
    #else
        // On x86_64, always use GPU (CUDA)
        transform_config.compute_mode = NvBufSurfTransformCompute_GPU;
    #endif

    transform_config.gpu_id = ctx->gpu_id;
    transform_config.cuda_stream = ctx->cuda_stream;

    if (NvBufSurfTransformSetSessionParams(&transform_config) != NvBufSurfTransformError_Success) {
        g_printerr("OBB_DUMP: NvBufSurfTransformSetSessionParams failed\n");
        return false;
    }

    NvBufSurfTransformRect src_rect = {
        .top = (uint32_t)aabb_top,
        .left = (uint32_t)aabb_left,
        .width = (uint32_t)aabb_width,
        .height = (uint32_t)aabb_height
    };

    NvBufSurfTransformRect dst_rect = {
        .top = 0,
        .left = 0,
        .width = (uint32_t)aabb_width,
        .height = (uint32_t)aabb_height
    };

    NvBufSurfTransformParams transform_params;
    memset(&transform_params, 0, sizeof(NvBufSurfTransformParams));
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER |
                                      NVBUFSURF_TRANSFORM_CROP_SRC |
                                      NVBUFSURF_TRANSFORM_CROP_DST;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;

    if (NvBufSurfTransform(input_surf, ctx->aabb_surface, &transform_params) != NvBufSurfTransformError_Success) {
        g_printerr("OBB_DUMP: NvBufSurfTransform failed\n");
        return false;
    }

    // Step 2: Extract rotated OBB pixels using scanline kernel
    int obb_width = (int)width;
    int obb_height = (int)height;

    // Ensure minimum OBB dimensions
    if (obb_width < 2) obb_width = 2;
    if (obb_height < 2) obb_height = 2;

    if (!ensure_obb_buffer(ctx, obb_width, obb_height)) {
        return false;
    }

    cudaError_t err = obb_scanline_rotate_gpu(
        (const uint8_t*)ctx->aabb_surface->surfaceList[0].dataPtr,  // AABB pixels
        ctx->d_obb_pixels,                                           // OBB output
        aabb_width,
        aabb_height,
        ctx->aabb_surface->surfaceList[0].pitch,
        obb_width,
        obb_height,
        obb_cx_in_aabb,
        obb_cy_in_aabb,
        angle_rad,
        ctx->cuda_stream
    );

    if (err != cudaSuccess) {
        g_printerr("OBB_DUMP: obb_scanline_rotate_gpu failed: %s\n",
                   cudaGetErrorString(err));
        return false;
    }

    // Synchronize to ensure extraction is complete
    cudaStreamSynchronize(ctx->cuda_stream);

    // Create temporary NvBufSurface for the extracted OBB
    // Follow nvds_obj_encode pattern: use CUDA_DEVICE for all platforms
    NvBufSurfaceCreateParams obb_create_params;
    memset(&obb_create_params, 0, sizeof(NvBufSurfaceCreateParams));
    obb_create_params.gpuId = ctx->gpu_id;
    obb_create_params.width = obb_width;
    obb_create_params.height = obb_height;
    obb_create_params.layout = NVBUF_LAYOUT_PITCH;
    obb_create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    obb_create_params.memType = NVBUF_MEM_CUDA_DEVICE;  // Works on both Jetson and x86_64

    NvBufSurface *obb_surface = NULL;
    if (NvBufSurfaceCreate(&obb_surface, 1, &obb_create_params) != 0) {
        g_printerr("OBB_DUMP: Failed to create OBB surface %dx%d\n",
                   obb_width, obb_height);
        return false;
    }

    // Copy extracted pixels to the surface
    // CUDA_DEVICE memory has valid dataPtr on both platforms
    cudaMemcpy2D(
        obb_surface->surfaceList[0].dataPtr,
        obb_surface->surfaceList[0].pitch,
        ctx->d_obb_pixels,
        obb_width * 4,  // Source pitch
        obb_width * 4,  // Width in bytes
        obb_height,
        cudaMemcpyDeviceToDevice
    );

    // Step 4: Use nvds_obj_encode to save as JPEG
    NvDsObjEncUsrArgs enc_args;
    memset(&enc_args, 0, sizeof(NvDsObjEncUsrArgs));
    enc_args.saveImg = true;
    enc_args.attachUsrMeta = false;
    enc_args.scaleImg = false;
    enc_args.quality = ctx->quality;
    enc_args.isFrame = true;  // Encode the whole extracted OBB surface
    enc_args.objNum = obj_idx;
    enc_args.calcEncodeTime = false;

    // Generate filename (convert radians to degrees for readability)
    float angle_deg = angle_rad * 180.0f / M_PI;
    snprintf(enc_args.fileNameImg, FILE_NAME_SIZE,
             "%s/obb_frame%04u_obj%d_angle%.1f_w%d_h%d.jpg",
             ctx->dump_dir, frame_num, obj_idx, angle_deg, obb_width, obb_height);

    // Create dummy frame metadata (nvds_obj_encode requires it)
    NvDsFrameMeta frame_meta_dummy;
    memset(&frame_meta_dummy, 0, sizeof(NvDsFrameMeta));
    frame_meta_dummy.frame_num = frame_num;
    frame_meta_dummy.batch_id = 0;

    bool success = nvds_obj_enc_process(
        ctx->enc_ctx,
        &enc_args,
        obb_surface,
        NULL,  // No object meta (we're encoding the whole surface)
        &frame_meta_dummy
    );

    if (!success) {
        g_printerr("OBB_DUMP: nvds_obj_enc_process failed\n");
    } else {
        ctx->dump_count++;
    }

    // Cleanup
    NvBufSurfaceDestroy(obb_surface);

    return success;
}

/**
 * @brief Process frame and dump OBBs
 * 
 * This is the main entry point called from nvll_osd_int.cpp
 */
bool nvobb_dump_frame(
    NvOBBDumpCtx *ctx,
    NvBufSurface *input_surf,
    void *rect_params_array,
    int num_rects,
    unsigned int frame_num)
{
    if (!ctx || !ctx->enabled) {
        return false;
    }

    ctx->frame_count++;

    // Check frame interval
    if ((ctx->frame_count % ctx->frame_interval) != 0) {
        return true;  // Skip this frame
    }

    // Cast rect_params to the correct type
    // NvOSD_RectParams and NvOSD_ColorParams are defined in nvll_obb.h
    NvOSD_RectParams *rect_params = (NvOSD_RectParams*)rect_params_array;

    int dumped_this_frame = 0;

    for (int i = 0; i < num_rects; i++) {
        NvOSD_RectParams *rect = &rect_params[i];

        // Check if rotation angle meets minimum threshold
        // Convert rotation_angle from radians to degrees for comparison
        float angle_deg = fabsf(rect->rotation_angle * 180.0f / M_PI);
        if (angle_deg < ctx->min_angle) {
            continue;  // Skip boxes below minimum angle threshold
        }

        // Skip very small OBBs (likely false detections) - reduced threshold
        if (rect->width < 2.0f || rect->height < 2.0f) {
            continue;  // Skip very small boxes
        }

        // Calculate center
        float cx = rect->left + rect->width / 2.0f;
        float cy = rect->top + rect->height / 2.0f;

        // Dump this OBB
        if (dump_single_obb(ctx, input_surf,
                           cx, cy, rect->width, rect->height, rect->rotation_angle,
                           i, frame_num)) {
            dumped_this_frame++;
        }
    }

    // Wait for all encodes to finish before returning
    if (dumped_this_frame > 0) {
        nvds_obj_enc_finish(ctx->enc_ctx);
    }

    return true;
}

