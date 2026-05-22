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
 * @file drawOBB.cu
 * @brief CUDA kernels for rendering Oriented Bounding Boxes (OBBs)
 *
 * This file implements GPU-accelerated rendering of rotated bounding boxes
 * using inverse rotation method. Supports:
 * - Border rendering (with configurable thickness)
 * - Fill rendering
 * - Combined border + fill
 * - Alpha blending and opaque (unit alpha) variants
 */

#include "drawOBB.h"
#include <cuda_runtime.h>

/**
 * @brief Check if a point is inside an oriented bounding box
 *
 * Uses inverse rotation to transform pixel coordinates to OBB's local space,
 * then performs axis-aligned bounds check.
 *
 * @param px Pixel X coordinate
 * @param py Pixel Y coordinate
 * @param center_x OBB center X
 * @param center_y OBB center Y
 * @param half_width Half of OBB width
 * @param half_height Half of OBB height
 * @param cos_angle Cosine of rotation angle
 * @param sin_angle Sine of rotation angle
 * @return true if point is inside OBB
 */
__device__ static __forceinline__ bool isPointInOBB(
    float px, float py,
    float center_x, float center_y,
    float half_width, float half_height,
    float cos_angle, float sin_angle)
{
    // Translate to OBB center
    float dx = px - center_x;
    float dy = py - center_y;

    // Inverse rotation to OBB's local coordinate system
    float local_x = dx * cos_angle + dy * sin_angle;
    float local_y = -dx * sin_angle + dy * cos_angle;

    // Axis-aligned bounds check in local space
    return (local_x >= -half_width && local_x <= half_width &&
            local_y >= -half_height && local_y <= half_height);
}

/**
 * @brief Check if a point is on the border of an OBB
 *
 * @param px Pixel X coordinate
 * @param py Pixel Y coordinate
 * @param center_x OBB center X
 * @param center_y OBB center Y
 * @param half_width Half of OBB width
 * @param half_height Half of OBB height
 * @param cos_angle Cosine of rotation angle
 * @param sin_angle Sine of rotation angle
 * @param border_width Border thickness in pixels
 * @return true if point is on border
 */
__device__ static __forceinline__ bool isPointOnOBBBorder(
    float px, float py,
    float center_x, float center_y,
    float half_width, float half_height,
    float cos_angle, float sin_angle,
    int border_width)
{
    // Check if inside outer box
    if (!isPointInOBB(px, py, center_x, center_y, half_width, half_height, cos_angle, sin_angle))
        return false;

    // Check if outside inner box (shrunk by border_width)
    float inner_half_width = half_width - border_width;
    float inner_half_height = half_height - border_width;

    if (inner_half_width <= 0.0f || inner_half_height <= 0.0f)
        return true;  // Border fills entire box

    return !isPointInOBB(px, py, center_x, center_y, inner_half_width, inner_half_height, cos_angle, sin_angle);
}

/**
 * @brief Render OBB border with unit alpha (opaque, no blending)
 *
 * Fast path for fully opaque borders. Simply overwrites pixels without
 * reading background.
 */
__global__ void drawOBB_border_unit_alpha_kernel(
    uchar4 *pBGRA,
    int width, int height, int stride,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float cos_angle, float sin_angle,
    unsigned int r, unsigned int g, unsigned int b,
    int border_width)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;

    // Check if pixel is on border
    if (isPointOnOBBBorder(px + 0.5f, py + 0.5f, center_x, center_y, 
                           half_w, half_h, cos_angle, sin_angle, border_width))
    {
        // Opaque write (no blending)
        pBGRA[py * stride + px].x = r;
        pBGRA[py * stride + px].y = g;
        pBGRA[py * stride + px].z = b;
        pBGRA[py * stride + px].w = 255;
    }
}

/**
 * @brief Host wrapper for OBB border rendering (opaque)
 */
void drawOBB_border_cuda_unit_alpha(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b,
    int border_width)
{
    // Pre-compute trig functions on host
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawOBB_border_unit_alpha_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)image, width, height, pitch / sizeof(uchar4),
        center_x, center_y, obb_width, obb_height,
        cos_angle, sin_angle, r, g, b, border_width);
}

/**
 * @brief Render OBB border with alpha blending
 *
 * Slower path that reads background and blends with foreground color.
 */
__global__ void drawOBB_border_kernel(
    uchar4 *pBGRA,
    int width, int height, int stride,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float cos_angle, float sin_angle,
    unsigned int r, unsigned int g, unsigned int b,
    float alpha,
    int border_width)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;

    // Check if pixel is on border
    if (isPointOnOBBBorder(px + 0.5f, py + 0.5f, center_x, center_y,
                           half_w, half_h, cos_angle, sin_angle, border_width))
    {
        // Alpha blending: C_out = C_fg * α + C_bg * (1 - α)
        float alpha_inv = 1.0f - alpha;
        int idx = py * stride + px;

        pBGRA[idx].x = (unsigned char)(pBGRA[idx].x * alpha_inv + r * alpha);
        pBGRA[idx].y = (unsigned char)(pBGRA[idx].y * alpha_inv + g * alpha);
        pBGRA[idx].z = (unsigned char)(pBGRA[idx].z * alpha_inv + b * alpha);
    }
}

/**
 * @brief Host wrapper for OBB border rendering (with alpha)
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
    int border_width)
{
    // Pre-compute trig functions on host
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawOBB_border_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)image, width, height, pitch / sizeof(uchar4),
        center_x, center_y, obb_width, obb_height,
        cos_angle, sin_angle, r, g, b, alpha, border_width);
}

/**
 * @brief Fill OBB with solid color (opaque, no blending)
 * 
 * Fast path for fully opaque fills.
 */
__global__ void drawOBB_fill_unit_alpha_kernel(
    uchar4 *pBGRA,
    int width, int height, int stride,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float cos_angle, float sin_angle,
    unsigned int r, unsigned int g, unsigned int b)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;

    // Check if pixel is inside OBB
    if (isPointInOBB(px + 0.5f, py + 0.5f, center_x, center_y,
                     half_w, half_h, cos_angle, sin_angle))
    {
        // Opaque write
        pBGRA[py * stride + px].x = r;
        pBGRA[py * stride + px].y = g;
        pBGRA[py * stride + px].z = b;
        pBGRA[py * stride + px].w = 255;
    }
}

/**
 * @brief Host wrapper for OBB fill rendering (opaque)
 */
void drawOBB_fill_cuda_unit_alpha(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    unsigned int r, unsigned int g, unsigned int b,
    cudaStream_t stream)
{
    // Pre-compute trig functions on host
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawOBB_fill_unit_alpha_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)image, width, height, pitch / sizeof(uchar4),
        center_x, center_y, obb_width, obb_height,
        cos_angle, sin_angle, r, g, b);
}

/**
 * @brief Fill OBB with alpha blending
 */
__global__ void drawOBB_fill_kernel(
    uchar4 *pBGRA,
    int width, int height, int stride,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float cos_angle, float sin_angle,
    unsigned int r, unsigned int g, unsigned int b,
    float alpha)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;

    // Check if pixel is inside OBB
    if (isPointInOBB(px + 0.5f, py + 0.5f, center_x, center_y,
                     half_w, half_h, cos_angle, sin_angle))
    {
        // Alpha blending
        float alpha_inv = 1.0f - alpha;
        int idx = py * stride + px;

        pBGRA[idx].x = (unsigned char)(pBGRA[idx].x * alpha_inv + r * alpha);
        pBGRA[idx].y = (unsigned char)(pBGRA[idx].y * alpha_inv + g * alpha);
        pBGRA[idx].z = (unsigned char)(pBGRA[idx].z * alpha_inv + b * alpha);
    }
}

/**
 * @brief Host wrapper for OBB fill rendering (with alpha)
 */
void drawOBB_fill_cuda(
    uint8_t *image,
    int width, int height, int pitch,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float angle_rad,
    unsigned int r, unsigned int g, unsigned int b,
    float alpha,
    cudaStream_t stream)
{
    // Pre-compute trig functions on host
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawOBB_fill_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)image, width, height, pitch / sizeof(uchar4),
        center_x, center_y, obb_width, obb_height,
        cos_angle, sin_angle, r, g, b, alpha);
}

/**
 * @brief Render OBB with both fill and border (opaque)
 *
 * Optimized kernel that renders fill and border in single pass.
 */
__global__ void drawOBB_combined_unit_alpha_kernel(
    uchar4 *pBGRA,
    int width, int height, int stride,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float cos_angle, float sin_angle,
    unsigned int border_r, unsigned int border_g, unsigned int border_b,
    int border_width,
    unsigned int fill_r, unsigned int fill_g, unsigned int fill_b)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;
    float pixel_x = px + 0.5f;
    float pixel_y = py + 0.5f;

    // Check if inside OBB
    if (!isPointInOBB(pixel_x, pixel_y, center_x, center_y,
                      half_w, half_h, cos_angle, sin_angle))
        return;

    int idx = py * stride + px;

    // Check if on border
    float inner_half_w = half_w - border_width;
    float inner_half_h = half_h - border_width;

    if (inner_half_w > 0.0f && inner_half_h > 0.0f &&
        isPointInOBB(pixel_x, pixel_y, center_x, center_y,
                     inner_half_w, inner_half_h, cos_angle, sin_angle))
    {
        // Inside (fill color)
        pBGRA[idx].x = fill_r;
        pBGRA[idx].y = fill_g;
        pBGRA[idx].z = fill_b;
    }
    else
    {
        // On border
        pBGRA[idx].x = border_r;
        pBGRA[idx].y = border_g;
        pBGRA[idx].z = border_b;
    }

    pBGRA[idx].w = 255;
}

/**
 * @brief Host wrapper for combined OBB rendering (opaque)
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
    bool is_filled)
{
    // Pre-compute trig functions on host
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawOBB_combined_unit_alpha_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)image, width, height, pitch / sizeof(uchar4),
        center_x, center_y, obb_width, obb_height,
        cos_angle, sin_angle,
        border_r, border_g, border_b, border_width,
        fill_r, fill_g, fill_b);
}

/**
 * @brief Render OBB with both fill and border (with alpha blending)
 *
 * Full-featured kernel supporting separate alpha values for fill and border.
 */
__global__ void drawOBB_combined_kernel(
    uchar4 *pBGRA,
    int width, int height, int stride,
    float center_x, float center_y,
    float obb_width, float obb_height,
    float cos_angle, float sin_angle,
    unsigned int border_r, unsigned int border_g, unsigned int border_b,
    float border_alpha,
    int border_width,
    unsigned int fill_r, unsigned int fill_g, unsigned int fill_b,
    float fill_alpha)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float half_w = obb_width * 0.5f;
    float half_h = obb_height * 0.5f;
    float pixel_x = px + 0.5f;
    float pixel_y = py + 0.5f;

    // Check if inside OBB
    if (!isPointInOBB(pixel_x, pixel_y, center_x, center_y,
                      half_w, half_h, cos_angle, sin_angle))
        return;

    int idx = py * stride + px;

    // Check if on border
    float inner_half_w = half_w - border_width;
    float inner_half_h = half_h - border_width;

    bool on_border = true;
    if (inner_half_w > 0.0f && inner_half_h > 0.0f &&
        isPointInOBB(pixel_x, pixel_y, center_x, center_y,
                     inner_half_w, inner_half_h, cos_angle, sin_angle))
    {
        on_border = false;
    }

    if (on_border)
    {
        // Border with alpha blending
        float alpha_inv = 1.0f - border_alpha;
        pBGRA[idx].x = (unsigned char)(pBGRA[idx].x * alpha_inv + border_r * border_alpha);
        pBGRA[idx].y = (unsigned char)(pBGRA[idx].y * alpha_inv + border_g * border_alpha);
        pBGRA[idx].z = (unsigned char)(pBGRA[idx].z * alpha_inv + border_b * border_alpha);
    }
    else
    {
        // Fill with alpha blending
        float alpha_inv = 1.0f - fill_alpha;
        pBGRA[idx].x = (unsigned char)(pBGRA[idx].x * alpha_inv + fill_r * fill_alpha);
        pBGRA[idx].y = (unsigned char)(pBGRA[idx].y * alpha_inv + fill_g * fill_alpha);
        pBGRA[idx].z = (unsigned char)(pBGRA[idx].z * alpha_inv + fill_b * fill_alpha);
    }
}

/**
 * @brief Host wrapper for combined OBB rendering (with alpha)
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
    bool is_filled)
{
    // Pre-compute trig functions on host
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawOBB_combined_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)image, width, height, pitch / sizeof(uchar4),
        center_x, center_y, obb_width, obb_height,
        cos_angle, sin_angle,
        border_r, border_g, border_b, border_alpha, border_width,
        fill_r, fill_g, fill_b, fill_alpha);
}

