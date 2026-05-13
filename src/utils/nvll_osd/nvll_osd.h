/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVLL_OSD_H__
#define __NVLL_OSD_H__

#include "nvll_osd_api.h"

#include "nvbufsurface.h"
#include "cuosd.h"
#include <cairo.h>
#include <unordered_map>
#include "time.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _NvOSDFdMap
{
    void *mapping;
    cairo_surface_t *surface;
    cairo_t *cairo_context;
    int dmabuf_fd;
    int size;
    NvBufSurface *surf;
}NvOSDFdMap;

typedef struct _NvOSD_Ctx {
    int enable_clock;
    void * in_buf;
    int frame_width;
    int frame_height;
    NvBufSurfacePlaneParams *params;
    NvOSD_TextParams clk_params;
    NvOSD_RectParams *solid_rect_params_list;
    NvOSD_RectParams *alpha_rect_params_list;
    unsigned int map_cnt;
    std::unordered_map <int, void *> *map_list;
    struct timeval t1, t2, prev_t1;
    int frameNum;
    int is_integrated;
    double elapsedTime;
    double totalReadTime;
    double fps;
    double avg_fps;
    double display_fps;
    void * conv_buf;
    NvOSDFdMap *nvosd_map;
    int num_classes;
    float *mask_buf;
    unsigned int mask_buf_size;
    unsigned int mask_buf_offset;
    cudaStream_t stream { nullptr };
    cuOSDContext_t cuosd_context;
} NvOSD_Ctx;

int CopytoHostMem(void *nvosd_ctx, void *cuDevPtr);

int CopytoDeviceMem(void *nvosd_ctx, void *cuDevPtr);

/* Mask the region using VIC */
bool color_compare(NvOSD_RectParams lhs, NvOSD_RectParams rhs);

NvOSDFdMap *set_cairo_context(NvOSD_Ctx *ctx, NvBufSurfaceParams *buf_ptr, NvBufSurface *nvbuf_surf);

/* frame green, cyan, blue, yellow, red, black, white */
/* TODO: Get predetermined clr from app */
//int COLORS[8] = {0xff00ff, 0x00ff00, 0xff00ff, 0x0000ff, 0xffff000, 0xff0000, 0x000000, 0xffffff};

#ifdef __cplusplus
}
#endif

/** @} */
#endif
