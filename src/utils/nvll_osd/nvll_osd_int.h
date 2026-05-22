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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pango/pangocairo.h>
#include <math.h>
#include "nvll_osd.h"

inline bool CHECK_(int e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        //cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile;
        exit (-1);
        return false;
    }
    return true;
}

#define clip_rect(ctx, left, top, right, bottom) \
{ \
    if(left < 0) \
    { \
        left = 0; \
    } \
    if(top < 0) \
    { \
        top = 0; \
    } \
    if(right > ctx->frame_width) \
    { \
        right = ctx->frame_width; \
    } \
    if(bottom > ctx->frame_height) \
    { \
        bottom = ctx->frame_height; \
    } \
}

#define ck(call) CHECK_(call, __LINE__, __FILE__)

void nvll_osd_resize_segment_masks_bg(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map, NvOSD_RectParams *rect_params, NvOSD_MaskParams *mask_params, void** mask_argb32);
void nvll_osd_draw_segment_masks_bg(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map, NvOSD_RectParams *rect_params, NvOSD_MaskParams *mask_params, void* mask_argb32);
void nvll_osd_draw_bounding_rectangles(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map, NvOSD_RectParams *rect_params);
void nvll_osd_draw_mask_regions(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map, NvOSD_RectParams *rect_params);
void nvll_osd_rect_with_border_bg(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map, NvOSD_RectParams *rect_params);

void nvll_osd_construct_draw_arrows_cpu(NvOSD_Ctx * ctx, NvOSDFdMap *nvosd_map,
    NvOSD_ArrowParams *arrow_params_list);

int nvll_osd_draw_segment_masks_cpu(void *nvosd_ctx, NvOSD_FrameSegmentMaskParams *frame_mask_params);
int nvll_osd_draw_segment_masks_gpu(void *nvosd_ctx, NvOSD_FrameSegmentMaskParams *frame_mask_params);

int nvll_osd_blur_rectangles_gpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params);
int nvll_osd_draw_rectangles_cpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params);
int nvll_osd_draw_rectangles_gpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params);

int nvll_osd_draw_arrows_cpu(void *nvosd_ctx, NvOSD_FrameArrowParams *frame_arrow_params);
int nvll_osd_draw_arrows_gpu(void *nvosd_ctx, NvOSD_FrameArrowParams *frame_arrow_params);

int nvll_osd_draw_circles_cpu(void *nvosd_ctx, NvOSD_FrameCircleParams *frame_circle_params);
int nvll_osd_draw_circles_gpu(void *nvosd_ctx, NvOSD_FrameCircleParams *frame_circle_params);

int nvll_osd_draw_lines_cpu(void *nvosd_ctx, NvOSD_FrameLineParams *frame_line_params);
int nvll_osd_draw_lines_gpu(void *nvosd_ctx, NvOSD_FrameLineParams *frame_line_params);

int nvll_osd_put_text_cpu(void *nvosd_ctx, NvOSD_FrameTextParams *frame_text_params);
int nvll_osd_put_text_gpu(void *nvosd_ctx, NvOSD_FrameTextParams *frame_text_params);
int nvll_osd_gpu_apply(void *nvosd_ctx, NvBufSurfaceParams *buf_ptr, NvBufSurface *surf);
