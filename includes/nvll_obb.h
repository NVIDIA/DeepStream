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

#ifndef __NVLL_OBB_H__
#define __NVLL_OBB_H__

#include "nvll_osd_api.h"
#include "nvbufsurface.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief OBB dump context handle
 *
 * Note: NvOSD_RectParams and NvOSD_ColorParams are defined in nvll_osd_struct.h
 * and are included via nvll_osd_api.h
 */
typedef struct _NvOBBDumpCtx NvOBBDumpCtx;

/**
 * @brief Create OBB dump context
 *
 * Initializes the OBB dumping system. Reads configuration from environment variables:
 * - NVDS_OBB_DUMP_ENABLED: Set to 1 to enable (default: disabled)
 * - NVDS_OBB_DUMP_DIR: Output directory (default: /tmp/obb_dumps)
 * - NVDS_OBB_DUMP_QUALITY: JPEG quality 1-100 (default: 90)
 * - NVDS_OBB_DUMP_FRAME_INTERVAL: Dump every Nth frame (default: 30)
 * - NVDS_OBB_DUMP_MIN_ANGLE: Minimum angle threshold in degrees (default: 1.0)
 *
 * @param gpu_id GPU device ID
 * @param cuda_stream CUDA stream to use for async operations
 * @return NvOBBDumpCtx* Context handle, or NULL if disabled or failed
 */
NvOBBDumpCtx* nvobb_dump_create_context(int gpu_id, cudaStream_t cuda_stream);

/**
 * @brief Destroy OBB dump context
 *
 * Waits for all pending encodes to finish and frees all resources.
 *
 * @param ctx Context handle
 */
void nvobb_dump_destroy_context(NvOBBDumpCtx *ctx);

/**
 * @brief Get or create singleton OBB dump context
 *
 * Thread-safe lazy initialization of global OBB dump context.
 * The context is created once and reused across all OSD drawing calls.
 *
 * @param gpu_id GPU device ID
 * @param cuda_stream CUDA stream (0 for default stream)
 * @return Pointer to singleton context, or NULL if disabled/failed
 */
NvOBBDumpCtx* nvobb_dump_get_singleton(int gpu_id, cudaStream_t cuda_stream);

/**
 * @brief Cleanup singleton OBB dump context
 *
 * Should be called during application shutdown to properly release
 * all OBB dump resources. Thread-safe and idempotent.
 */
void nvobb_dump_cleanup_singleton(void);

/**
 * @brief Process frame and dump OBBs
 *
 * Extracts and saves rotated OBBs from the frame. This function:
 * 1. Checks frame interval (only processes every Nth frame)
 * 2. For each OBB with angle >= min_angle:
 *    a. Calculates AABB containing the rotated OBB
 *    b. Crops AABB using NvBufSurfTransform (GPU)
 *    c. Extracts rotated pixels using obb_scanline_rotate_gpu
 *    d. Encodes to JPEG using nvds_obj_encode
 *
 * @param ctx Context handle
 * @param input_surf Input surface containing the frame
 * @param rect_params_array Array of rectangle parameters (NvOSD_RectParams*)
 * @param num_rects Number of rectangles in the array
 * @param frame_num Frame number (for filename generation)
 * @return bool true on success, false on failure
 */
bool nvobb_dump_frame(
    NvOBBDumpCtx *ctx,
    NvBufSurface *input_surf,
    void *rect_params_array,
    int num_rects,
    unsigned int frame_num);

/**
 * @brief Rotate OBB to rectangular RGBA on GPU (standalone)
 *
 * This function performs ONLY the rotation on GPU. No dumping, no threading,
 * no encoding. Use this when you want to process rotated OBBs yourself
 * (e.g., for tracking, feature extraction, etc.)
 *
 * @param aabb_pixels Input AABB pixels (GPU memory, RGBA format)
 * @param aabb_width Width of input AABB
 * @param aabb_height Height of input AABB
 * @param aabb_pitch Pitch (stride) of AABB in bytes
 * @param obb_width Width of output OBB
 * @param obb_height Height of output OBB
 * @param obb_cx Center X of OBB in AABB space
 * @param obb_cy Center Y of OBB in AABB space
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution (use 0 for default stream)
 * @param[out] d_output Pointer to receive allocated GPU buffer with rotated OBB
 *                      Caller must free with cudaFree()
 * @return cudaError_t cudaSuccess on success, error code on failure
 *
 * @note Memory management: Caller is responsible for freeing *d_output with cudaFree()
 * @note Output format: RGBA, tightly packed, row-major, size = obb_width × obb_height × 4 bytes
 */
cudaError_t nvobb_rotate_gpu(
    const uint8_t *aabb_pixels,
    int aabb_width,
    int aabb_height,
    int aabb_pitch,
    int obb_width,
    int obb_height,
    float obb_cx,
    float obb_cy,
    float angle_rad,
    cudaStream_t stream,
    uint8_t **d_output);

/**
 * @brief Rotate OBB to rectangular RGBA on CPU (standalone)
 *
 * This function performs ONLY the rotation on CPU. No dumping, no threading,
 * no encoding. Use this when you want to process rotated OBBs yourself
 * and don't need GPU acceleration.
 *
 * @param aabb_pixels Input AABB pixels (CPU memory, RGBA format)
 * @param aabb_width Width of input AABB
 * @param aabb_height Height of input AABB
 * @param aabb_pitch Pitch (stride) of AABB in bytes
 * @param obb_width Width of output OBB
 * @param obb_height Height of output OBB
 * @param obb_cx Center X of OBB in AABB space
 * @param obb_cy Center Y of OBB in AABB space
 * @param angle_rad Rotation angle in radians
 * @return uint8_t* Allocated CPU buffer with rotated OBB (RGBA), or NULL on failure
 *                  Caller must free with g_free() or free()
 *
 * @note Memory management: Caller is responsible for freeing returned buffer with g_free()
 * @note Output format: RGBA, tightly packed, row-major, size = obb_width × obb_height × 4 bytes
 */
uint8_t *nvobb_rotate_cpu(
    const uint8_t *aabb_pixels,
    int aabb_width,
    int aabb_height,
    int aabb_pitch,
    int obb_width,
    int obb_height,
    float obb_cx,
    float obb_cy,
    float angle_rad);

#ifdef __cplusplus
}
#endif

#endif /* __NVLL_OBB_H__ */

