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

/**
 * @file drawOBB.h
 * @brief Header for Oriented Bounding Box (OBB) rendering functions
 *
 * Provides GPU-accelerated rendering of rotated bounding boxes with support for:
 * - Border rendering (configurable thickness)
 * - Fill rendering
 * - Combined border + fill
 * - Opaque and alpha-blended variants
 */

#ifndef __DRAW_OBB_H__
#define __DRAW_OBB_H__

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Draw OBB border with unit alpha (opaque, fast path)
 *
 * Renders the border of a rotated bounding box without alpha blending.
 * This is the fastest variant as it doesn't require reading background pixels.
 *
 * @param image Pointer to RGBA image buffer (GPU memory)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param pitch Image pitch in bytes (stride)
 * @param center_x OBB center X coordinate
 * @param center_y OBB center Y coordinate
 * @param obb_width OBB width (before rotation)
 * @param obb_height OBB height (before rotation)
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @param border_width Border thickness in pixels
 *
 * @note Alpha is implicitly 1.0 (fully opaque)
 * @note Uses inverse rotation method for pixel-perfect rendering
 */
void drawOBB_border_cuda_unit_alpha(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b,
    int border_width);

/**
 * @brief Draw OBB border with alpha blending
 *
 * Renders the border of a rotated bounding box with alpha blending.
 * Slower than unit alpha variant due to read-modify-write operations.
 *
 * @param image Pointer to RGBA image buffer (GPU memory)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param pitch Image pitch in bytes (stride)
 * @param center_x OBB center X coordinate
 * @param center_y OBB center Y coordinate
 * @param obb_width OBB width (before rotation)
 * @param obb_height OBB height (before rotation)
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @param alpha Alpha value (0.0-1.0)
 * @param border_width Border thickness in pixels
 *
 * @note Alpha blending formula: C_out = C_fg * α + C_bg * (1 - α)
 */
void drawOBB_border_cuda(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b,
    float alpha,
    int border_width);

/**
 * @brief Fill OBB with solid color (unit alpha, fast path)
 *
 * Fills the interior of a rotated bounding box without alpha blending.
 *
 * @param image Pointer to RGBA image buffer (GPU memory)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param pitch Image pitch in bytes (stride)
 * @param center_x OBB center X coordinate
 * @param center_y OBB center Y coordinate
 * @param obb_width OBB width (before rotation)
 * @param obb_height OBB height (before rotation)
 * @param angle_rad Rotation angle in radians
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @param stream CUDA stream for async execution
 *
 * @note Alpha is implicitly 1.0 (fully opaque)
 */
void drawOBB_fill_cuda_unit_alpha(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    unsigned int r, unsigned int g, unsigned int b,
    cudaStream_t stream);

/**
 * @brief Fill OBB with alpha blending
 *
 * Fills the interior of a rotated bounding box with alpha blending.
 *
 * @param image Pointer to RGBA image buffer (GPU memory)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param pitch Image pitch in bytes (stride)
 * @param center_x OBB center X coordinate
 * @param center_y OBB center Y coordinate
 * @param obb_width OBB width (before rotation)
 * @param obb_height OBB height (before rotation)
 * @param angle_rad Rotation angle in radians
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @param alpha Alpha value (0.0-1.0)
 * @param stream CUDA stream for async execution
 */
void drawOBB_fill_cuda(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    unsigned int r, unsigned int g, unsigned int b,
    float alpha,
    cudaStream_t stream);

/**
 * @brief Draw OBB with both fill and border (unit alpha, fast path)
 *
 * Renders both fill and border in a single pass without alpha blending.
 * Most efficient for opaque OBBs.
 *
 * @param image Pointer to RGBA image buffer (GPU memory)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param pitch Image pitch in bytes (stride)
 * @param center_x OBB center X coordinate
 * @param center_y OBB center Y coordinate
 * @param obb_width OBB width (before rotation)
 * @param obb_height OBB height (before rotation)
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution
 * @param border_r Border red component (0-255)
 * @param border_g Border green component (0-255)
 * @param border_b Border blue component (0-255)
 * @param border_width Border thickness in pixels
 * @param fill_r Fill red component (0-255)
 * @param fill_g Fill green component (0-255)
 * @param fill_b Fill blue component (0-255)
 * @param is_filled Whether to render fill (true) or border only (false)
 *
 * @note Both fill and border alpha are implicitly 1.0 (fully opaque)
 * @note Single-pass rendering for maximum efficiency
 */
void drawOBB_combined_cuda_unit_alpha(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    cudaStream_t stream,
    unsigned int border_r, unsigned int border_g, unsigned int border_b,
    int border_width,
    unsigned int fill_r, unsigned int fill_g, unsigned int fill_b,
    bool is_filled);

/**
 * @brief Draw OBB with both fill and border (with alpha blending)
 *
 * Renders both fill and border with independent alpha values.
 * Most flexible but slowest variant.
 *
 * @param image Pointer to RGBA image buffer (GPU memory)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param pitch Image pitch in bytes (stride)
 * @param center_x OBB center X coordinate
 * @param center_y OBB center Y coordinate
 * @param obb_width OBB width (before rotation)
 * @param obb_height OBB height (before rotation)
 * @param angle_rad Rotation angle in radians
 * @param stream CUDA stream for async execution
 * @param border_r Border red component (0-255)
 * @param border_g Border green component (0-255)
 * @param border_b Border blue component (0-255)
 * @param border_alpha Border alpha value (0.0-1.0)
 * @param border_width Border thickness in pixels
 * @param fill_r Fill red component (0-255)
 * @param fill_g Fill green component (0-255)
 * @param fill_b Fill blue component (0-255)
 * @param fill_alpha Fill alpha value (0.0-1.0)
 * @param is_filled Whether to render fill (true) or border only (false)
 *
 * @note Supports independent alpha for fill and border
 * @note Single-pass rendering with conditional blending
 */
void drawOBB_combined_cuda(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    cudaStream_t stream,
    unsigned int border_r, unsigned int border_g, unsigned int border_b,
    float border_alpha,
    int border_width,
    unsigned int fill_r, unsigned int fill_g, unsigned int fill_b,
    float fill_alpha,
    bool is_filled);

#ifdef __cplusplus
}
#endif

#endif // __DRAW_OBB_H__

