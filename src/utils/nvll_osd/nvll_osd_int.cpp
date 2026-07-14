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

#include "nvll_osd_int.h"
#include "nvll_obb.h"
#include "nvds_mask_utils.h"
#include "cudaEGL.h"
#include <sys/time.h>
#include "cuda/drawOBB.h"
#include "cuda/drawBbox.h"

//#define CUDA_PERF
//#define DEBUG_PERF
#define CUOSD

#define NVBUF_ALIGN_VAL 256
#define NVBUF_ALIGN_PITCH(pitch, align_val) (((pitch) + (align_val) - 1) & ~((align_val) - 1))
#define NVBUF_PLATFORM_ALIGNED_PITCH(pitch) NVBUF_ALIGN_PITCH(pitch, NVBUF_ALIGN_VAL)
#define ARROW_ANGLE 15
#define DEFAULT_THICKNESS 2

void nvll_osd_resize_segment_masks_bg(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map,
        NvOSD_RectParams *rect_params, NvOSD_MaskParams *mask_params,
        void** mask_argb32)
{
    int x, y, width, height, stride;
    int border_width;

    border_width = rect_params->border_width;

    x = rect_params->left;
    y = rect_params->top;

    width = (x + rect_params->width > ctx->frame_width) ?
        ctx->frame_width - x - border_width : rect_params->width;
    height = (y + rect_params->height > ctx->frame_height) ?
        ctx->frame_height - y - border_width : rect_params->height;
    clip_rect(ctx, x, y, width, height);

    stride = cairo_format_stride_for_width (CAIRO_FORMAT_ARGB32, width);
    *mask_argb32 = (unsigned int *)malloc(stride * height * sizeof(uint32_t));
    memset (*mask_argb32, 0, stride * height * sizeof(uint32_t));

    bool ok = nvds_mask_utils_resize_to_binary_argb32((float*)mask_params->data,
                     (uint32_t*)(*mask_argb32),
                     (uint32_t)(mask_params->width),
                     (uint32_t)(mask_params->height),
                     (uint32_t)width, (uint32_t)height, (uint32_t)1,
                     mask_params->threshold,
                     (127 << 24), 2 /**< NPPI_INTER_LINEAR = 2 */,
                     ctx->stream);

    if(!ok)
    {
        NVOSD_PRINT_E("ERROR: [%s]\n", __func__);
        return;
    }

}


void nvll_osd_draw_segment_masks_bg(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map,
        NvOSD_RectParams *rect_params, NvOSD_MaskParams *mask_params,
        void* mask_argb32)
{
    int x, y, width, height, stride;
    double r, g, b ,a;
    int border_width;
    cairo_surface_t *cr_surface;
    cairo_t *cr = nvosd_map->cairo_context;

    border_width = rect_params->border_width;

    r = rect_params->border_color.red;
    g = rect_params->border_color.green;
    b = rect_params->border_color.blue;
    a = rect_params->border_color.alpha;

    /** Set the color to be used for all pixels in the mask;
     * NOTE: Transparency is set in nvds_mask_utils_resize_to_binary()
     */
    cairo_set_source_rgba(cr, b, g, r, a);

    x = rect_params->left;
    y = rect_params->top;

    width = (x + rect_params->width > ctx->frame_width) ?
        ctx->frame_width - x - border_width : rect_params->width;
    height = (y + rect_params->height > ctx->frame_height) ?
        ctx->frame_height - y - border_width : rect_params->height;
    clip_rect(ctx, x, y, width, height);

    stride = cairo_format_stride_for_width (CAIRO_FORMAT_ARGB32, width);
    cr_surface = cairo_image_surface_create_for_data (
                (unsigned char*)mask_argb32, CAIRO_FORMAT_ARGB32,
                width, height, stride);
    cairo_mask_surface (cr, cr_surface, x, y);
    free(mask_argb32);
}

/* sets and draws bounding rectangles */
void nvll_osd_draw_bounding_rectangles(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map,
        NvOSD_RectParams *rect_params)
{
    double r, g, b ,a;
    int x, y, width, height;
    int border_width;
    cairo_t *cr = nvosd_map->cairo_context;

    border_width = rect_params->border_width;

    r = rect_params->border_color.red;
    g = rect_params->border_color.green;
    b = rect_params->border_color.blue;
    a = rect_params->border_color.alpha;

    cairo_set_source_rgba(cr, b, g, r, a);
    cairo_set_line_width(cr, border_width);

    x = rect_params->left;
    y = rect_params->top;

    if (fabsf(rect_params->rotation_angle) > 0.001f)
    {
        width = rect_params->width;
        height = rect_params->height;
        // Calculate center point for rotation
        double center_x = x + width / 2.0;
        double center_y = y + height / 2.0;

        // Save current transformation matrix
        cairo_save(cr);

        // Move to center and rotate
        cairo_translate(cr, center_x, center_y);
        cairo_rotate(cr, rect_params->rotation_angle * G_PI / 180.0);
        cairo_translate(cr, -width / 2.0, -height / 2.0);

        // Draw rectangle centered at origin
        cairo_rectangle(cr, 0, 0, width, height);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_MITER);
        cairo_stroke(cr);

        // Restore transformation matrix
        cairo_restore(cr);
    }
    else
    {
        // Original axis-aligned rectangle drawing
    width = (x + rect_params->width > ctx->frame_width) ?
        ctx->frame_width - x - border_width : rect_params->width;
    height = (y + rect_params->height > ctx->frame_height) ?
        ctx->frame_height - y - border_width : rect_params->height;
    clip_rect(ctx, x, y, width, height);
    cairo_rectangle (cr, x, y, width, height);
    cairo_set_line_join (cr, CAIRO_LINE_JOIN_MITER);

    cairo_stroke(cr);
    }
}

void nvll_osd_rect_with_border_bg(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map,
        NvOSD_RectParams *rect_params)
{
    double r, g, b ,a;
    int x, y, width, height, x1, y1, x2, y2;
    int border_width;
    cairo_t *cr = nvosd_map->cairo_context;

    border_width = rect_params->border_width;

    r = rect_params->border_color.red;
    g = rect_params->border_color.green;
    b = rect_params->border_color.blue;
    a = rect_params->border_color.alpha;

    cairo_set_source_rgba(cr, b, g, r, a);
    cairo_set_line_width(cr, border_width);

    x = rect_params->left;
    y = rect_params->top;

    if (fabsf(rect_params->rotation_angle) > 0.001f)
    {
        width = rect_params->width;
        height = rect_params->height;
        double center_x = x + width / 2.0;
        double center_y = y + height / 2.0;

        // Save current transformation matrix
        cairo_save(cr);

        // Move to center and rotate
        cairo_translate(cr, center_x, center_y);
        cairo_rotate(cr, rect_params->rotation_angle * G_PI / 180.0);
        cairo_translate(cr, -width / 2.0, -height / 2.0);

        // Draw border
        cairo_rectangle(cr, 0, 0, width, height);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_MITER);
        cairo_stroke(cr);

        /* background fill */
        r = rect_params->bg_color.red;
        g = rect_params->bg_color.green;
        b = rect_params->bg_color.blue;
        a = rect_params->bg_color.alpha;

        cairo_set_source_rgba(cr, b, g, r, a);
        cairo_rectangle(cr, 0, 0, width, height);
        cairo_fill(cr);

        // Restore transformation matrix
        cairo_restore(cr);
    }
    else
    {
        // Original axis-aligned rectangle drawing
        width = (x + rect_params->width > ctx->frame_width) ?
            ctx->frame_width - x - border_width : rect_params->width;
        height = (y + rect_params->height > ctx->frame_height) ?
            ctx->frame_height - y - border_width : rect_params->height;
        clip_rect(ctx, x, y, width, height);
        cairo_rectangle (cr, x, y, width, height);
        cairo_set_line_join (cr, CAIRO_LINE_JOIN_MITER);

        cairo_stroke(cr);

        /* background fill */
        r = rect_params->bg_color.red;
        g = rect_params->bg_color.green;
        b = rect_params->bg_color.blue;
        a = rect_params->bg_color.alpha;

        x1= rect_params->left;
        y1= rect_params->top;
        x2 = rect_params->left + rect_params->width;
        y2 = rect_params->top + rect_params->height;
        width= rect_params->width;
        height= rect_params->height;

        clip_rect(ctx, x1, y1, x2, y2);
        cairo_set_source_rgba(cr, b, g, r, a);
        cairo_rectangle(cr, x1, y1, width, height);
        cairo_fill(cr);
    }
}

void nvll_osd_draw_mask_regions(NvOSD_Ctx *ctx, NvOSDFdMap *nvosd_map,
        NvOSD_RectParams *rect_params)
{
    cairo_t *cr = nvosd_map->cairo_context;
    double r, g, b, a;
    int x1, y1, x2, y2, width, height;

    r = rect_params->bg_color.red;
    g = rect_params->bg_color.green;
    b = rect_params->bg_color.blue;
    a = rect_params->bg_color.alpha;

    x1= rect_params->left;
    y1= rect_params->top;
    width= rect_params->width;
    height= rect_params->height;

    if (fabsf(rect_params->rotation_angle) > 0.001f)
    {
        double center_x = x1 + width / 2.0;
        double center_y = y1 + height / 2.0;

        // Save current transformation matrix
        cairo_save(cr);

        // Move to center and rotate
        cairo_translate(cr, center_x, center_y);
        cairo_rotate(cr, rect_params->rotation_angle * G_PI / 180.0);
        cairo_translate(cr, -width / 2.0, -height / 2.0);

        // Draw filled rectangle
        cairo_set_source_rgba(cr, b, g, r, a);
        cairo_rectangle(cr, 0, 0, width, height);
        cairo_fill(cr);

        // Restore transformation matrix
        cairo_restore(cr);
    }
    else
    {
        // Original axis-aligned rectangle drawing
        x2 = rect_params->left + rect_params->width;
        y2 = rect_params->top + rect_params->height;

    clip_rect(ctx, x1, y1, x2, y2);
    cairo_set_source_rgba(cr, b, g, r, a);
    cairo_rectangle(cr, x1, y1, width, height);
    cairo_fill(cr);
    }
}

static int check_supported_colorformat(NvBufSurfaceParams *buf_ptr)
{
    if(!((buf_ptr->colorFormat == NVBUF_COLOR_FORMAT_RGBA) ||
            (buf_ptr->colorFormat == NVBUF_COLOR_FORMAT_ABGR) ||
            (buf_ptr->colorFormat == NVBUF_COLOR_FORMAT_BGRx)))
    {
        return -1;
    }
    return 0;
}

int nvll_osd_draw_segment_masks_cpu(void *nvosd_ctx, NvOSD_FrameSegmentMaskParams *frame_mask_params)
{
    int i = 0;
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    NvOSDFdMap *nvosd_map = NULL;
    int ret = 0;
    /** Holds point to an array of CPU buffers to save
     * resized masks */
    void** resized_masks;

    if(!frame_mask_params->num_segments)
    {
      NVOSD_PRINT_E("ERROR: No segments to draw\n");
      return -1;
    }

    resized_masks = (void**)calloc(frame_mask_params->num_segments, sizeof(void*));
    if(!resized_masks)
    {
      NVOSD_PRINT_E("ERROR: %s alloc error\n", __func__);
      free(resized_masks);
      return -1;
    }
    if(ctx->is_integrated) {
        if (frame_mask_params->buf_ptr != NULL)
        {
            ret = check_supported_colorformat(frame_mask_params->buf_ptr);
        } else {
            ret = check_supported_colorformat(frame_mask_params->surf->surfaceList);
        }
        if(ret != 0)
        {
            NVOSD_PRINT_E("ERROR: Unsupported color format\n");
	    free(resized_masks);
            return -1;
        }
    }

    nvosd_map = set_cairo_context(ctx, frame_mask_params->buf_ptr, frame_mask_params->surf);
    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
	free(resized_masks);
        return -1;
    }

    /** perform resize and thresholding on GPU first */
    for(i = 0; i < frame_mask_params->num_segments; i++)
    {
        nvll_osd_resize_segment_masks_bg(ctx, nvosd_map, &frame_mask_params->rect_params_list[i], &frame_mask_params->mask_params_list[i], &resized_masks[i]);
    }

    /** sync the stream before performing drawing operations on resized_masks */
    ck (cudaStreamSynchronize(ctx->stream));

    for(i = 0; i < frame_mask_params->num_segments; i++)
    {
        nvll_osd_draw_segment_masks_bg(ctx, nvosd_map, &frame_mask_params->rect_params_list[i], &frame_mask_params->mask_params_list[i], resized_masks[i]);
    }

    if(ctx->is_integrated) {
        ret = NvBufSurfaceSyncForDevice (nvosd_map->surf, -1, -1);
        if(ret != 0)
        {
            free(resized_masks);
            return ret;
        }
    }

    free(resized_masks);
    return ret;
}

int nvll_osd_draw_segment_masks_gpu(void *nvosd_ctx, NvOSD_FrameSegmentMaskParams *frame_mask_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    unsigned char r, g, b, a, bg_r, bg_g, bg_b, bg_a;
    NvOSD_RectParams *rect_params;
    NvOSD_MaskParams *mask_params;
    int i = 0;
    int x1, x2, y1, y2;
    int ret = 0;

    for(i = 0; i < frame_mask_params->num_segments; i++)
    {
        rect_params = &frame_mask_params->rect_params_list[i];
        mask_params = &frame_mask_params->mask_params_list[i];

        x1 = rect_params->left;
        y1 = rect_params->top;
        x2 = rect_params->width + x1;
        y2 = rect_params->height + y1;

        r = g = b = a = 0;
        bg_r = (unsigned char)(rect_params->border_color.red * 255);
        bg_g = (unsigned char)(rect_params->border_color.green * 255);
        bg_b = (unsigned char)(rect_params->border_color.blue * 255);
        bg_a = (unsigned char)(rect_params->border_color.alpha * 255);

        if(ctx->mask_buf_size - ctx->mask_buf_offset < mask_params->size)
        {
            float* src_f = NULL;
            ck(cudaMallocAsync(&src_f, ctx->mask_buf_size +
                    ctx->frame_width * ctx->frame_height * sizeof(float), ctx->stream));
            if(ctx->mask_buf)
            {
                ck(cudaMemcpyAsync(src_f, ctx->mask_buf, ctx->mask_buf_offset,
                            cudaMemcpyDeviceToDevice, ctx->stream));
                ck(cudaFreeAsync(ctx->mask_buf, ctx->stream));
            }
            ctx->mask_buf = src_f;
            ctx->mask_buf_size += ctx->frame_width * ctx->frame_height *
                sizeof(float);
        }
        ck(cudaMemcpyAsync((unsigned char*)ctx->mask_buf + ctx->mask_buf_offset,
                    mask_params->data, mask_params->size,
                    cudaMemcpyHostToDevice, ctx->stream));

        //printf ("mask x1: %d y1: %d x2: %d y2: %d\n", x1, y1, x2, y2);
        //printf ("mask width: %d height: %d, threshold: %f\n",
        //mask_params->width, mask_params->height, mask_params->threshold);
        //printf("mask r g b a bg r g b a: %d %d %d %d %d %d %d %d\n",
        //r, g, b, a, bg_r, bg_g, bg_b, bg_a);
        cuosd_draw_segmentmask(ctx->cuosd_context, x1, y1, x2, y2, 0,
                             (float*)((unsigned char*)ctx->mask_buf + ctx->mask_buf_offset),
                             mask_params->width, mask_params->height,
                             mask_params->threshold, {r, g, b, a},
                             {bg_r, bg_g, bg_b, bg_a});
        ctx->mask_buf_offset += mask_params->size;
    }

    return ret;
}

int nvll_osd_draw_rectangles_cpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params)
{	
    int i = 0, has_bg_color = 0, border_width = 0;
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    NvOSDFdMap *nvosd_map = NULL;
    int ret = 0;

    if(ctx->is_integrated) {
        if (frame_rect_params->buf_ptr != NULL)
        {
            ret = check_supported_colorformat(frame_rect_params->buf_ptr);
        } else {
            ret = check_supported_colorformat(frame_rect_params->surf->surfaceList);
        }
        if(ret != 0)
        {
            NVOSD_PRINT_E("ERROR: Unsupported color format\n");
            return -1;
        }
    }

    nvosd_map = set_cairo_context(ctx, frame_rect_params->buf_ptr, frame_rect_params->surf);
    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    for(i = 0; i < frame_rect_params->num_rects; i++)
    {
        has_bg_color = frame_rect_params->rect_params_list[i].has_bg_color;
        border_width = frame_rect_params->rect_params_list[i].border_width;

        if((border_width < 0) || (border_width > MAX_BORDER_WIDTH))
        {
            NVOSD_PRINT_E("Unsupported border width\n");
            return -1;
        }

        if(!has_bg_color && border_width) /* rectangle with border.
                                             No BG= No fill */
        {
            nvll_osd_draw_bounding_rectangles(ctx, nvosd_map, &frame_rect_params->rect_params_list[i]);
        }
        else if(has_bg_color && (border_width == 0)) /* rectangle with no
                                                        border. Mask rect with
                                                        given color */
        {
            nvll_osd_draw_mask_regions(ctx, nvosd_map, &frame_rect_params->rect_params_list[i]);
        }
        else if(has_bg_color && (border_width)) /* rectangle with border and
                                                   BG color */
        {
            nvll_osd_rect_with_border_bg(ctx, nvosd_map, &frame_rect_params->rect_params_list[i]);
        }
    }
    if(ctx->is_integrated) {
        ret = NvBufSurfaceSyncForDevice (nvosd_map->surf, -1, -1);
        if(ret != 0)
        {
            return ret;
        }
    }
    return ret;
}

#ifdef CUOSD
int nvll_osd_blur_rectangles_gpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params)
{
  NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
  NvOSD_RectParams *rect_params;
  NvOSD_MaskParams *mask_params;
  int i = 0;
  int x1, x2, y1, y2;
  int ret = 0;

  float* d_seg = NULL;
  int seg_width = 0;
  int seg_height = 0;
  for(i = 0; i < frame_rect_params->num_rects; i++)
  {
    rect_params = &frame_rect_params->rect_params_list[i];
    x1 = rect_params->left;
    y1 = rect_params->top;
    x2 = rect_params->width + x1;
    y2 = rect_params->height + y1;
    mask_params = NULL;

    if(frame_rect_params->mask_params_list) {
        mask_params = &frame_rect_params->mask_params_list[i];
        if(mask_params) {
            if(ctx->mask_buf_size - ctx->mask_buf_offset < mask_params->size)
            {
                float* src_f = NULL;
                ck(cudaMallocAsync(&src_f, ctx->mask_buf_size +
                        ctx->frame_width * ctx->frame_height * sizeof(float), ctx->stream));
                if(ctx->mask_buf)
                {
                    ck(cudaMemcpyAsync(src_f, ctx->mask_buf, ctx->mask_buf_offset,
                                cudaMemcpyDeviceToDevice, ctx->stream));
                    ck(cudaFreeAsync(ctx->mask_buf, ctx->stream));
                }
                ctx->mask_buf = src_f;
                ctx->mask_buf_size += ctx->frame_width * ctx->frame_height *
                    sizeof(float);
            }
            ck(cudaMemcpyAsync((unsigned char*)ctx->mask_buf + ctx->mask_buf_offset,
                    mask_params->data, mask_params->size,
                    cudaMemcpyHostToDevice, ctx->stream));
            d_seg = (float*)((unsigned char*)ctx->mask_buf + ctx->mask_buf_offset);
            seg_width = mask_params->width;
            seg_height = mask_params->height;
        }
    }
    else {
        d_seg = NULL;
        seg_width = seg_height = 0;
    }

    cuosd_draw_boxblur(ctx->cuosd_context, x1, y1, x2, y2, d_seg, seg_width, seg_height);
    if(mask_params)
        ctx->mask_buf_offset += mask_params->size;
  }

  return ret;
}

int nvll_osd_draw_rectangles_gpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    // Get OBB dump singleton context (thread-safe lazy initialization)
    int gpu_id = frame_rect_params->surf ? frame_rect_params->surf->gpuId : 0;
    NvOBBDumpCtx *obb_dump_ctx = nvobb_dump_get_singleton(gpu_id, 0);

    unsigned char r, g, b, a, bg_r, bg_g, bg_b, bg_a;
    NvOSD_RectParams *rect_params;
    int i = 0, border_width = 0;
    int x1, x2, y1, y2;
    int ret = 0;

    for(i = 0; i < frame_rect_params->num_rects; i++)
    {
        rect_params = &frame_rect_params->rect_params_list[i];

        border_width = rect_params->border_width;
        float rotation_angle = rect_params->rotation_angle;

        if((border_width < 0) || (border_width > MAX_BORDER_WIDTH))
        {
            NVOSD_PRINT_E("Unsupported border width\n");
            return -1;
        }

        x1 = rect_params->left;
        y1 = rect_params->top;
        x2 = rect_params->width + x1;
        y2 = rect_params->height + y1;

        r = (unsigned char)(rect_params->border_color.red * 255);
        g = (unsigned char)(rect_params->border_color.green * 255);
        b = (unsigned char)(rect_params->border_color.blue * 255);
        a = (unsigned char)(rect_params->border_color.alpha * 255);
        bg_r = (unsigned char)(rect_params->bg_color.red * 255);
        bg_g = (unsigned char)(rect_params->bg_color.green * 255);
        bg_b = (unsigned char)(rect_params->bg_color.blue * 255);
        if(rect_params->has_bg_color)
            bg_a = (unsigned char)(rect_params->bg_color.alpha * 255);
        else
            bg_a = 0;

        if (fabsf(rotation_angle) > 0.001f)
        {
            // OBB path - use cuosd_draw_rotationbox
            float center_x = (x1 + x2) / 2.0f;
            float center_y = (y1 + y2) / 2.0f;
            int width = x2 - x1;
            int height = y2 - y1;
            float yaw_rad = rotation_angle * M_PI / 180.0f;

            cuosd_draw_rotationbox(ctx->cuosd_context,
                                   (int)center_x, (int)center_y,
                                   width, height, yaw_rad,
                                   border_width, {r, g, b, a}, false, {bg_r, bg_g, bg_b, bg_a});
        }
        else
        {
        cuosd_draw_rectangle(ctx->cuosd_context, x1, y1, x2, y2, border_width,
                             {r, g, b, a}, {bg_r, bg_g, bg_b, bg_a});
        }
    }

    // Dump OBBs if enabled
    if (obb_dump_ctx && frame_rect_params->surf)
    {
        nvobb_dump_frame(obb_dump_ctx, frame_rect_params->surf, frame_rect_params->rect_params_list,
                        frame_rect_params->num_rects, 0);
    }

    return ret;
}
#else
int nvll_osd_draw_rectangles_gpu(void *nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params)
{
    int i = 0, has_bg_color = 0, border_width = 0;
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    int ret = 0;

    // Get OBB dump singleton context (thread-safe lazy initialization)
    int gpu_id = frame_rect_params->surf ? frame_rect_params->surf->gpuId : 0;
    NvOBBDumpCtx *obb_dump_ctx = nvobb_dump_get_singleton(gpu_id, 0);

    int x1, x2, y1, y2;
    unsigned int r, g, b;
    float a;
    NvOSDFdMap *nvosd_map = NULL;

    nvosd_map = set_cairo_context(ctx, frame_rect_params->buf_ptr, frame_rect_params->surf);
    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    gboolean unmap_egl = FALSE;
    CUresult status;
    CUgraphicsResource buf_ptr = NULL;
    CUeglFrame eglFrame;
    NvBufSurface *surface = NULL;

    #if defined(__aarch64__)
    if(ctx->is_integrated) {
        ret = NvBufSurfaceFromFd (nvosd_map->dmabuf_fd, (void**)(&surface));
        if(ret != 0)
        {
        return ret;
        }

        if (surface->surfaceList[0].mappedAddr.eglImage == NULL)
        {
        if (NvBufSurfaceMapEglImage(surface, 0) != 0)
        {
            NVOSD_PRINT_E ("Unable to map EGL Image");
            return -1;
        }
        unmap_egl = TRUE;
        }

        EGLImageKHR eglimage_src = surface->surfaceList[0].mappedAddr.eglImage;

        status = cuGraphicsEGLRegisterImage(&buf_ptr, eglimage_src, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (status != CUDA_SUCCESS) {
        NVOSD_PRINT_E("cuGraphicsEGLRegisterImage failed : %d \n", status);
        return -1;
        }

        status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, buf_ptr, 0, 0);
        if (status != CUDA_SUCCESS)
        {
        NVOSD_PRINT_E("cuGraphicsSubResourceGetMappedArray failed\n");
        status = cuGraphicsUnregisterResource(buf_ptr);
        if (status != CUDA_SUCCESS) {
            printf ("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
        }
        return -1;
        }
    }
    #endif

    uint8_t *ptr = NULL;
    if(ctx->is_integrated) {
        ptr = (uint8_t *) eglFrame.frame.pPitch[0];
    }
    else {
        // buf_ptr is deprecated, use surf->surfaceList
        if (frame_rect_params->buf_ptr) {
            ptr = (uint8_t *) frame_rect_params->buf_ptr[0].dataPtr;
        } else if (frame_rect_params->surf) {
            ptr = (uint8_t *) frame_rect_params->surf->surfaceList[0].dataPtr;
        }
    }

    for(i = 0; i < frame_rect_params->num_rects; i++)
    {
        has_bg_color = frame_rect_params->rect_params_list[i].has_bg_color;
        border_width = frame_rect_params->rect_params_list[i].border_width;

        if((border_width < 0) || (border_width > MAX_BORDER_WIDTH))
        {
            NVOSD_PRINT_E("Unsupported border width\n");
            return -1;
        }

        x1 = frame_rect_params->rect_params_list[i].left;
        y1 = frame_rect_params->rect_params_list[i].top;
        x2 = frame_rect_params->rect_params_list[i].width + x1;
        y2 = frame_rect_params->rect_params_list[i].height + y1;

        float current_angle = frame_rect_params->rect_params_list[i].rotation_angle;

        if (fabsf(current_angle) > 0.001f)
        {
            // Convert rectangle coordinates to OBB parameters
            float center_x = (x1 + x2) / 2.0f;
            float center_y = (y1 + y2) / 2.0f;
            float width = x2 - x1;
            float height = y2 - y1;
            float angle_rad = frame_rect_params->rect_params_list[i].rotation_angle * M_PI / 180.0f;

            // Get colors
            unsigned int border_r = (int)(frame_rect_params->rect_params_list[i].border_color.red * 255);
            unsigned int border_g = (int)(frame_rect_params->rect_params_list[i].border_color.green * 255);
            unsigned int border_b = (int)(frame_rect_params->rect_params_list[i].border_color.blue * 255);
            float border_a = frame_rect_params->rect_params_list[i].border_color.alpha;

            unsigned int fill_r = (int)(frame_rect_params->rect_params_list[i].bg_color.red * 255);
            unsigned int fill_g = (int)(frame_rect_params->rect_params_list[i].bg_color.green * 255);
            unsigned int fill_b = (int)(frame_rect_params->rect_params_list[i].bg_color.blue * 255);
            float fill_a = frame_rect_params->rect_params_list[i].bg_color.alpha;

            if (has_bg_color && border_width > 0)
            {
                // Both fill and border
                if (border_a >= 0.99f && fill_a >= 0.99f)
                {
                    drawOBB_combined_cuda_unit_alpha(ptr, ctx->frame_width, ctx->frame_height,
                                                     NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4),
                                                     center_x, center_y, width, height, angle_rad,
                                                     ctx->stream, border_r, border_g, border_b, border_width,
                                                     fill_r, fill_g, fill_b, true);
                }
                else
                {
                    drawOBB_combined_cuda(ptr, ctx->frame_width, ctx->frame_height,
                                          NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4),
                                          center_x, center_y, width, height, angle_rad,
                                          ctx->stream, border_r, border_g, border_b, border_a, border_width,
                                          fill_r, fill_g, fill_b, fill_a, true);
                }
            }
            else if (has_bg_color)
            {
                // Fill only
                if (fill_a >= 0.99f)
                {
                    drawOBB_fill_cuda_unit_alpha(ptr, ctx->frame_width, ctx->frame_height,
                                                 NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4),
                                                 center_x, center_y, width, height, angle_rad,
                                                 fill_r, fill_g, fill_b, ctx->stream);
                }
                else
                {
                    drawOBB_fill_cuda(ptr, ctx->frame_width, ctx->frame_height,
                                      NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4),
                                      center_x, center_y, width, height, angle_rad,
                                      fill_r, fill_g, fill_b, fill_a, ctx->stream);
                }
            }
            else if (border_width > 0)
            {
                // Border only
                if (border_a >= 0.99f)
                {
                    drawOBB_border_cuda_unit_alpha(ptr, ctx->frame_width, ctx->frame_height,
                                                   NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4),
                                                   center_x, center_y, width, height, angle_rad,
                                                   ctx->stream, border_r, border_g, border_b, border_width);
                }
                else
                {
                    drawOBB_border_cuda(ptr, ctx->frame_width, ctx->frame_height,
                                        NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4),
                                        center_x, center_y, width, height, angle_rad,
                                        ctx->stream, border_r, border_g, border_b, border_a, border_width);
                }
            }
        }
        else
        {
            if(has_bg_color) /* rectangle with no
                                                            border. Mask rect with
                                                            given color */
            {
                r = (int)(frame_rect_params->rect_params_list[i].bg_color.red * 255);
                g = (int)(frame_rect_params->rect_params_list[i].bg_color.green * 255);
                b = (int)(frame_rect_params->rect_params_list[i].bg_color.blue * 255);
                a = frame_rect_params->rect_params_list[i].bg_color.alpha;
                bboxAlphaFill_cuda(ptr, ctx->frame_width, ctx->frame_height,
                    NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4), x1, y1, x2, y2, r, g, b, a, ctx->stream);
            }
            if(border_width) /* rectangle with border.
                                                No BG= No fill */
            {
                r = (int)(frame_rect_params->rect_params_list[i].border_color.red * 255);
                g = (int)(frame_rect_params->rect_params_list[i].border_color.green * 255);
                b = (int)(frame_rect_params->rect_params_list[i].border_color.blue * 255);
                a = frame_rect_params->rect_params_list[i].border_color.alpha;
                if(a==1){
                        drawBoundingBox_cuda_unit_alpha (ptr, ctx->frame_width, ctx->frame_height,
                            NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4), x1, y1, x2, y2, ctx->stream, r, g, b, border_width);
                }
                else{

                        drawBoundingBox_cuda (ptr, ctx->frame_width, ctx->frame_height,
                            NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4), x1, y1, x2, y2, ctx->stream, r, g, b, a, border_width);
                }
            }
        }
    }
    //printf ("Doing CUDA based OSD Drawing... \n");
    ck(cudaStreamSynchronize(ctx->stream));

    if(ctx->is_integrated) {
        status = cuGraphicsUnregisterResource(buf_ptr);
        if (status != CUDA_SUCCESS) {
        printf ("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
        }

        if (unmap_egl)
        {
        if (NvBufSurfaceUnMapEglImage(surface, 0) != 0)
        {
            NVOSD_PRINT_E ("Unable to unmap EGL Image");
            return -1;
        }
        }
    }

    // Dump OBBs if enabled
    if (obb_dump_ctx && frame_rect_params->surf)
    {
        nvobb_dump_frame(obb_dump_ctx, frame_rect_params->surf, frame_rect_params->rect_params_list,
                         frame_rect_params->num_rects, 0);
    }

    return ret;
}
#endif

int nvll_osd_put_text_cpu(void *nvosd_ctx, NvOSD_FrameTextParams *frame_text_params)
{
    int i = 0;
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;
    double r, g, b, a;
    int x1, y1;
    char *display_text = NULL;
    PangoLayout *layout = NULL;
    PangoFontDescription *desc = NULL;
    char display_time[256];
    char font_size_buffer[256];
    NvOSDFdMap *nvosd_map = NULL;
    cairo_t *cr = NULL;
    NvOSD_RectParams rect_params{};

    /* Get time */
    int hr = 0,  min = 0, sec = 0;
    int year = 0, month = 0, day = 0;
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);

    year = tm.tm_year;
    month = tm.tm_mon;
    day = tm.tm_mday;
    hr = tm.tm_hour;
    min = tm.tm_min;
    sec = tm.tm_sec;

#ifdef MEASURE_TIME
    if (ctx->frameNum == 1)
        gettimeofday(&ctx->t1, NULL);

    ctx->prev_t1 = ctx->t1;
    gettimeofday(&ctx->t1, NULL);
    ctx->fps = (ctx->t1.tv_sec - ctx->prev_t1.tv_sec) * 1000.0;
    ctx->fps += (ctx->t1.tv_usec - ctx->prev_t1.tv_usec) / 1000.0;
    ctx->avg_fps += (1000 / ctx->fps);
#endif

    if ( (ctx->frameNum % 30) == 0)
    {
        ctx->display_fps = ctx->avg_fps / 30;
      ctx->avg_fps = 0;
    }
    //snprintf(display_time, 256,"Frame=%d FPS=%02f %d-%d-%d  %02d:%02d:%02d",
    //    ctx->frameNum++, ctx->display_fps, year + 1900, month + 1, day, hr, min, sec);

    snprintf(display_time, 256,"%d-%d-%d  %02d:%02d:%02d",
            year + 1900, month + 1, day, hr, min, sec);

    nvosd_map = set_cairo_context(ctx, frame_text_params->buf_ptr, frame_text_params->surf);

    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    cr = nvosd_map->cairo_context;

    layout = pango_cairo_create_layout (cr);

    if(layout == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    for(i = 0; i < frame_text_params->num_strings; i++)
    {
        r = frame_text_params->text_params_list->font_params.font_color.red;
        g = frame_text_params->text_params_list->font_params.font_color.green;
        b = frame_text_params->text_params_list->font_params.font_color.blue;
        a = frame_text_params->text_params_list->font_params.font_color.alpha;

        x1= frame_text_params->text_params_list->x_offset;
        y1= frame_text_params->text_params_list->y_offset;

        if(x1 < 0)
        {
            x1 = 0;
        }

        if(y1 < 0)
        {
            y1 = 0;
        }

        display_text = frame_text_params->text_params_list->display_text;
        if(display_text == NULL)
        {
            frame_text_params->text_params_list++;
            continue;
        }

        snprintf(font_size_buffer, 256, "%s %d",
                frame_text_params->text_params_list->font_params.font_name,
                frame_text_params->text_params_list->font_params.font_size);

        desc = pango_font_description_from_string (font_size_buffer);

        pango_layout_set_font_description (layout, desc);
        pango_layout_set_width (layout, ctx->frame_width * PANGO_SCALE);
        pango_layout_set_wrap (layout, PANGO_WRAP_WORD);
        pango_layout_set_text (layout, display_text, -1);

        if(frame_text_params->text_params_list->set_bg_clr)
        {
            int w = 0, h = 0;

            pango_layout_get_pixel_size (layout, &w, &h);


            rect_params.bg_color.red = frame_text_params->text_params_list->text_bg_clr.red;
            rect_params.bg_color.green = frame_text_params->text_params_list->text_bg_clr.green;
            rect_params.bg_color.blue = frame_text_params->text_params_list->text_bg_clr.blue;
            rect_params.bg_color.alpha = frame_text_params->text_params_list->text_bg_clr.alpha;

            rect_params.left = x1;
            rect_params.top = y1;
            rect_params.width = w + frame_text_params->text_params_list->font_params.font_size;
            rect_params.height  = h + frame_text_params->text_params_list->font_params.font_size / 2;
            nvll_osd_draw_mask_regions(ctx, nvosd_map, &rect_params);
        }

        cairo_set_source_rgba (cr, b, g, r, a);
        pango_cairo_update_layout (cr, layout);
        cairo_move_to(cr, x1 + (frame_text_params->text_params_list->font_params.font_size/ 2),
                y1 + (frame_text_params->text_params_list->font_params.font_size/4));
        pango_cairo_show_layout (cr, layout);
        pango_font_description_free (desc);
        frame_text_params->text_params_list++;
    }

    if(ctx->enable_clock && layout)
    {
        pango_layout_set_text (layout, display_time, -1);
        snprintf(font_size_buffer, 256, "%s %d", ctx->clk_params.font_params.font_name, ctx->clk_params.font_params.font_size);
        desc = pango_font_description_from_string (font_size_buffer);

        pango_layout_set_font_description (layout, desc);

        r = ctx->clk_params.font_params.font_color.red;
        g = ctx->clk_params.font_params.font_color.green;
        b = ctx->clk_params.font_params.font_color.blue;
        a = ctx->clk_params.font_params.font_color.alpha;

        x1= ctx->clk_params.x_offset;
        y1= ctx->clk_params.y_offset;

        if(x1 < 0)
        {
            x1 = 0;
        }

        if(y1 < 0)
        {
            y1 = 0;
        }

        cairo_set_source_rgba (cr, b, g, r, a);
        pango_cairo_update_layout (cr, layout);
        cairo_move_to(cr, x1, y1);
        pango_cairo_show_layout (cr, layout);
        pango_font_description_free (desc);
    }

    if(layout)
    {
        g_object_unref(layout);
    }
    if(ctx->is_integrated) {
        ret = NvBufSurfaceSyncForDevice (nvosd_map->surf, -1, -1);
        if(ret != 0) {
            return ret;
        }
    }

#ifdef MEASURE_TIME
    gettimeofday(&ctx->t2, NULL);
    ctx->elapsedTime = (ctx->t2.tv_sec - ctx->t1.tv_sec) * 1000.0;
    ctx->elapsedTime += (ctx->t2.tv_usec - ctx->t1.tv_usec) / 1000.0;
    //printf("elapsedTime = %f\n",ctx->elapsedTime);
#endif

    return ret;
}

int nvll_osd_put_text_gpu(void *nvosd_ctx, NvOSD_FrameTextParams *frame_text_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    unsigned char r, g, b, a, bg_r, bg_g, bg_b, bg_a;
    NvOSD_TextParams *text_params;
    NvOSD_TextParams *clk_params;
    int i, x, y, font_size;
    const char *display_text;
    int ret = 0;
#ifdef DEBUG_PERF
    struct timeval start_time;
    struct timeval stop_time;
    int total_time_us = 0;

    gettimeofday (&start_time, NULL);
#endif

    for(i = 0; i < frame_text_params->num_strings; i++)
    {
        text_params = &frame_text_params->text_params_list[i];
        r = (unsigned char)(text_params->font_params.font_color.red * 255);
        g = (unsigned char)(text_params->font_params.font_color.green * 255);
        b = (unsigned char)(text_params->font_params.font_color.blue * 255);
        a = (unsigned char)(text_params->font_params.font_color.alpha * 255);
        bg_r = (unsigned char)(text_params->text_bg_clr.red * 255);
        bg_g = (unsigned char)(text_params->text_bg_clr.green * 255);
        bg_b = (unsigned char)(text_params->text_bg_clr.blue * 255);
        if(text_params->set_bg_clr)
            bg_a = (unsigned char)(text_params->text_bg_clr.alpha * 255);
        else
            bg_a = 0;

        x= text_params->x_offset;
        y= text_params->y_offset;

        x = (x < 0) ? 0 : x;
        y = (y < 0) ? 0 : y;

        display_text = text_params->display_text;
        font_size = text_params->font_params.font_size;
        //printf ("text x: %d y: %d font_size: %d display_text: %s\n", x, y,
        //font_size, display_text);
        //printf("text r g b a bg r g b a: %d %d %d %d %d %d %d %d\n",
        //r, g, b, a, bg_r, bg_g, bg_b, bg_a);
        cuosd_draw_text(ctx->cuosd_context, display_text, font_size,
                        text_params->font_params.font_name, x, y,
                        {r, g, b, a}, {bg_r, bg_g, bg_b, bg_a});
    }

#ifdef DEBUG_PERF
    gettimeofday (&stop_time, NULL);
    total_time_us = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
        (stop_time.tv_usec - start_time.tv_usec);
    printf("Add text takes: %d us\n", total_time_us);
#endif

    if(ctx->enable_clock)
    {
        clk_params = &ctx->clk_params;
        r = (unsigned char)(clk_params->font_params.font_color.red * 255);
        g = (unsigned char)(clk_params->font_params.font_color.green * 255);
        b = (unsigned char)(clk_params->font_params.font_color.blue * 255);
        a = (unsigned char)(clk_params->font_params.font_color.alpha * 255);
        bg_r = (unsigned char)(clk_params->text_bg_clr.red * 255);
        bg_g = (unsigned char)(clk_params->text_bg_clr.green * 255);
        bg_b = (unsigned char)(clk_params->text_bg_clr.blue * 255);
        if(clk_params->set_bg_clr)
            bg_a = (unsigned char)(clk_params->text_bg_clr.alpha * 255);
        else
            bg_a = 0;

        x= ctx->clk_params.x_offset;
        y= ctx->clk_params.y_offset;

        x = (x < 0) ? 0 : x;
        y = (y < 0) ? 0 : y;

        font_size = ctx->clk_params.font_params.font_size;
        //printf ("x: %d y: %d font_size: %d r: %d g: %d b: %d a: %d\n", x, y,
        //font_size, r, g, b, a);
        cuosd_draw_clock(ctx->cuosd_context, cuOSDClockFormat::YYMMDD_HHMMSS,
                         0, font_size, ctx->clk_params.font_params.font_name,
                         x, y, {r, g, b, a}, {bg_r, bg_g, bg_b, bg_a});
    }

    return ret;
}

int nvll_osd_draw_arrows_cpu(void *nvosd_ctx, NvOSD_FrameArrowParams *frame_arrow_params)
{
    int i = 0, arrow_width = 0;
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    NvOSDFdMap *nvosd_map = NULL;
    int ret = 0;

    if(ctx->is_integrated) {
        if (frame_arrow_params->buf_ptr != NULL)
        {
            ret = check_supported_colorformat(frame_arrow_params->buf_ptr);
        } else {
            ret = check_supported_colorformat(frame_arrow_params->surf->surfaceList);
        }
        if(ret != 0)
        {
            NVOSD_PRINT_E("ERROR: Unsupported color format\n");
            return -1;
        }
    }

    nvosd_map = set_cairo_context(ctx, frame_arrow_params->buf_ptr, frame_arrow_params->surf);
    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    for(i = 0; i < frame_arrow_params->num_arrows; i++)
    {
        arrow_width = frame_arrow_params->arrow_params_list[i].arrow_width;

        if((arrow_width < 0) || (arrow_width > MAX_BORDER_WIDTH))
        {
            NVOSD_PRINT_E("Unsupported border width\n");
            return -1;
        }

        /* sets and draws bounding rectangles */
        nvll_osd_construct_draw_arrows_cpu(ctx, nvosd_map, &frame_arrow_params->arrow_params_list[i]);
    }
    if(ctx->is_integrated) {
        ret = NvBufSurfaceSyncForDevice (nvosd_map->surf, -1, -1);
        if(ret != 0)
        {
            return ret;
        }
    }
    return ret;
}

int nvll_osd_draw_arrows_gpu(void *nvosd_ctx, NvOSD_FrameArrowParams *frame_arrow_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    NvOSD_ArrowParams *arrow_params;
    int i, arrow_width, arrow_size;
    int h_arrow, v_arrow, arrow_len;
    unsigned char r, g, b, a;
    int x1, x2, y1, y2;
    int ret = 0;

    for(i = 0; i < frame_arrow_params->num_arrows; i++)
    {
        arrow_params = &frame_arrow_params->arrow_params_list[i];
        arrow_width = arrow_params->arrow_width;

        if((arrow_width < 0) || (arrow_width > MAX_BORDER_WIDTH))
        {
            NVOSD_PRINT_E("Unsupported border width\n");
            return -1;
        }

        if(arrow_params->arrow_head == START_HEAD)
        {
            x2 = arrow_params->x1;
            y2 = arrow_params->y1;
            x1 = arrow_params->x2;
            y1 = arrow_params->y2;
        }
        else
        {
            x1 = arrow_params->x1;
            y1 = arrow_params->y1;
            x2 = arrow_params->x2;
            y2 = arrow_params->y2;
        }

        r = (unsigned char)(arrow_params->arrow_color.red * 255);
        g = (unsigned char)(arrow_params->arrow_color.green * 255);
        b = (unsigned char)(arrow_params->arrow_color.blue * 255);
        a = (unsigned char)(arrow_params->arrow_color.alpha * 255);

        h_arrow = (x2 - x1);
        v_arrow = (y2 - y1);

        arrow_len = sqrt((h_arrow * h_arrow) + (v_arrow * v_arrow));

        arrow_size = (arrow_len * 10) / 100;

        cuosd_draw_arrow(ctx->cuosd_context, x1, y1, x2, y2, arrow_size,
                         arrow_width, {r, g, b, a}, true);
    }

    return ret;
}

void nvll_osd_construct_draw_arrows_cpu(NvOSD_Ctx * ctx, NvOSDFdMap *nvosd_map,
    NvOSD_ArrowParams *arrow_params)
{
    double r, g, b ,a;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    int l_x, l_y, r_x, r_y;
    int h_arrow, v_arrow, arrow_len, sel_arrow_len;
    float theta = 0, rad = 0;

    int arrow_width = 0;
    cairo_t *cr = nvosd_map->cairo_context;

    arrow_width = arrow_params->arrow_width;

    r = arrow_params->arrow_color.red;
    g = arrow_params->arrow_color.green;
    b = arrow_params->arrow_color.blue;
    a = arrow_params->arrow_color.alpha;
    a = 1; /* Do not consider opacity */

    cairo_set_source_rgba(cr, b, g, r, a);
    cairo_set_line_width(cr, arrow_width);

    if(arrow_params->arrow_head == START_HEAD)
    {
      x2 = arrow_params->x1;
      y2 = arrow_params->y1;
      x1 = arrow_params->x2;
      y1 = arrow_params->y2;
    }
    else if(arrow_params->arrow_head == END_HEAD)
    {
      x1 = arrow_params->x1;
      y1 = arrow_params->y1;
      x2 = arrow_params->x2;
      y2 = arrow_params->y2;
    }

    /* calculate l_x, l_y, r_x, r_y */

    /* Find the length of arrow */
    h_arrow = (x2 - x1);
    v_arrow = (y2 - y1);

    theta = atan2(v_arrow, h_arrow); /* slope of the line*/

    //arrow_len = h_arrow / cos(theta);
    arrow_len = sqrt((h_arrow * h_arrow) + (v_arrow * v_arrow));

    sel_arrow_len = (arrow_len * 10) / 100;

    rad = (ARROW_ANGLE * M_PI) / 180;

    l_x =  x2 + sel_arrow_len * cos(theta - rad + M_PI);
    l_y =  y2 + sel_arrow_len * sin(theta - rad + M_PI);

    r_x =  x2 + sel_arrow_len * cos((theta + rad + M_PI));
    r_y =  y2 + sel_arrow_len * sin((theta + rad + M_PI));

    //printf("theta = %f\n", theta);
    //printf("l_x = %d l_y = %d r_x = %d r_y = %d\n", l_x, l_y, r_x, r_y);

    cairo_move_to(cr, x1, y1);
    cairo_line_to(cr, x2, y2);

    cairo_move_to(cr, x2, y2);
    cairo_line_to(cr, l_x, l_y);

    cairo_move_to(cr, x2, y2);
    cairo_line_to(cr, r_x, r_y);
    cairo_stroke(cr);
}

int nvll_osd_draw_circles_cpu(void *nvosd_ctx, NvOSD_FrameCircleParams *frame_circle_params)
{
    int i = 0, radius = 0, circle_width = 0, has_bg_color = 0;
    int xc = 0, yc = 0;
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    NvOSDFdMap *nvosd_map = NULL;
    int ret = 0;
    cairo_t *cr;

    if(ctx->is_integrated) {
        if (frame_circle_params->buf_ptr != NULL)
        {
            ret = check_supported_colorformat(frame_circle_params->buf_ptr);
        } else {
            ret = check_supported_colorformat(frame_circle_params->surf->surfaceList);
        }
        if(ret != 0)
        {
            NVOSD_PRINT_E("ERROR: Unsupported color format\n");
            return -1;
        }
    }

    nvosd_map = set_cairo_context(ctx, frame_circle_params->buf_ptr, frame_circle_params->surf);
    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    cr = nvosd_map->cairo_context;

    for(i = 0; i < frame_circle_params->num_circles; i++)
    {
        double b = frame_circle_params->circle_params_list[i].circle_color.blue;
        double g = frame_circle_params->circle_params_list[i].circle_color.green;
        double r = frame_circle_params->circle_params_list[i].circle_color.red;
        double a = frame_circle_params->circle_params_list[i].circle_color.alpha;

        radius = frame_circle_params->circle_params_list[i].radius;
        if (frame_circle_params->circle_params_list[i].circle_width)
            circle_width = frame_circle_params->circle_params_list[i].circle_width;
        else
            circle_width = DEFAULT_THICKNESS;

        xc = frame_circle_params->circle_params_list[i].xc;
        yc = frame_circle_params->circle_params_list[i].yc;
    	has_bg_color = frame_circle_params->circle_params_list[i].has_bg_color;

        if (radius < 0)
        {
            NVOSD_PRINT_E ("Negative Radius\n");
            return -1;
        }

        cairo_set_source_rgba(cr, b, g, r, a);
        /* sets and draws bounding rectangles */
        cairo_move_to(cr, xc + radius, yc);
        cairo_set_line_width (cr, circle_width);
        cairo_arc(cr, xc, yc, radius, 0.0, 2 * M_PI);
        cairo_stroke(cr);
	if(has_bg_color)
	{
	    double b_bg = frame_circle_params->circle_params_list[i].bg_color.blue;
	    double r_bg = frame_circle_params->circle_params_list[i].bg_color.red;
	    double g_bg = frame_circle_params->circle_params_list[i].bg_color.green;
	    double a_bg = frame_circle_params->circle_params_list[i].bg_color.alpha;
	    cairo_set_source_rgba(cr,b_bg,g_bg,r_bg,a_bg);
	    cairo_arc(cr,xc,yc,radius-1,0.0,2 * M_PI);
	    cairo_fill(cr);
	}
    }
    if(ctx->is_integrated) {
        ret = NvBufSurfaceSyncForDevice (nvosd_map->surf, -1, -1);
        if(ret != 0)
        {
            return ret;
        }
    }
    return ret;
}

int nvll_osd_draw_circles_gpu(void *nvosd_ctx, NvOSD_FrameCircleParams *frame_circle_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    unsigned char r, g, b, a, bg_r, bg_g, bg_b, bg_a;
    NvOSD_CircleParams *circle_params;
    int i, radius, circle_width, xc, yc;
    int ret = 0;

    for(i = 0; i < frame_circle_params->num_circles; i++)
    {
        circle_params = &frame_circle_params->circle_params_list[i];

        r = (unsigned char)(circle_params->circle_color.red * 255);
        g = (unsigned char)(circle_params->circle_color.green * 255);
        b = (unsigned char)(circle_params->circle_color.blue * 255);
        a = (unsigned char)(circle_params->circle_color.alpha * 255);
        bg_r = (unsigned char)(circle_params->bg_color.red * 255);
        bg_g = (unsigned char)(circle_params->bg_color.green * 255);
        bg_b = (unsigned char)(circle_params->bg_color.blue * 255);
        if(circle_params->has_bg_color)
            bg_a = (unsigned char)(circle_params->bg_color.alpha * 255);
        else
            bg_a = 0;

        radius = circle_params->radius;
        if (circle_params->circle_width)
            circle_width = circle_params->circle_width;
        else
            circle_width = DEFAULT_THICKNESS;

        xc = circle_params->xc;
        yc = circle_params->yc;

        //printf("circle r g b a bg r g b a: %d %d %d %d %d %d %d %d\n",
        //r, g, b, a, bg_r, bg_g, bg_b, bg_a);
        cuosd_draw_circle(ctx->cuosd_context, xc, yc, radius, circle_width,
                          {r, g, b, a}, {bg_r, bg_g, bg_b, bg_a});
    }

    return ret;
}

int nvll_osd_draw_lines_cpu(void *nvosd_ctx, NvOSD_FrameLineParams *frame_line_params)
{
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;
    int ret = 0;
    int x1, x2, y1, y2;
    int i = 0;
    NvOSDFdMap *nvosd_map;
    cairo_t *cr;

    nvosd_map = set_cairo_context(ctx, frame_line_params->buf_ptr, frame_line_params->surf);
    if(nvosd_map == NULL)
    {
        NVOSD_PRINT_E("Error in %s",__func__);
        return -1;
    }

    cr = nvosd_map->cairo_context;

    for(i = 0; i < frame_line_params->num_lines; i++)
    {
      x1 = frame_line_params->line_params_list[i].x1;
      y1 = frame_line_params->line_params_list[i].y1;
      x2 = frame_line_params->line_params_list[i].x2;
      y2 = frame_line_params->line_params_list[i].y2;

      double b = frame_line_params->line_params_list[i].line_color.blue;
      double g = frame_line_params->line_params_list[i].line_color.green;
      double r = frame_line_params->line_params_list[i].line_color.red;
      double a = frame_line_params->line_params_list[i].line_color.alpha;

      cairo_set_source_rgba(cr, b, g, r, a);
      cairo_set_line_width(cr, frame_line_params->line_params_list[i].line_width);

      cairo_move_to(cr, x1, y1);
      cairo_line_to(cr, x2, y2);
      cairo_stroke(cr);
    }
    if(ctx->is_integrated) {
      ret = NvBufSurfaceSyncForDevice (nvosd_map->surf, -1, -1);
      if (ret != 0) {
        return ret;
      }
    }
    return ret;
}

int nvll_osd_draw_lines_gpu(void *nvosd_ctx, NvOSD_FrameLineParams *frame_line_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    NvOSD_LineParams *line_params;
    unsigned char r, g, b, a;
    int x1, x2, y1, y2;
    int i, line_width;
    int ret = 0;

    for(i = 0; i < frame_line_params->num_lines; i++)
    {
        line_params = &frame_line_params->line_params_list[i];
        line_width = line_params->line_width;

        if ((line_width < 0) || (line_width > MAX_BORDER_WIDTH))
        {
            NVOSD_PRINT_E("Unsupported border width\n");
            return -1;
        }

        x1 = line_params->x1;
        y1 = line_params->y1;
        x2 = line_params->x2;
        y2 = line_params->y2;

        r = (unsigned char)(line_params->line_color.red * 255);
        g = (unsigned char)(line_params->line_color.green * 255);
        b = (unsigned char)(line_params->line_color.blue * 255);
        a = (unsigned char)(line_params->line_color.alpha * 255);

        //printf("line r g b a: %d %d %d %d\n", r, g, b, a);
        cuosd_draw_line(ctx->cuosd_context, x1, y1, x2, y2, line_width,
                        {r, g, b, a}, true);
    }

    return ret;
}

static bool is_nv12(NvBufSurfaceColorFormat format)
{
    return format == NVBUF_COLOR_FORMAT_NV12 \
        || format == NVBUF_COLOR_FORMAT_NV12_ER \
        || format == NVBUF_COLOR_FORMAT_NV12_709 \
        || format == NVBUF_COLOR_FORMAT_NV12_709_ER \
        || format == NVBUF_COLOR_FORMAT_NV12_2020;
}

int nvll_osd_gpu_apply(void *nvosd_ctx, NvBufSurfaceParams *buf_ptr, NvBufSurface *surface)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    cudaSurfaceObject_t YSurfObj;
    cudaSurfaceObject_t UVSurfObj;
    bool destroy_surfaceOjbect = false;
    bool is_nvmm_buf = false;
    int ret = 0;
#ifdef DEBUG_PERF
    struct timeval start_time;
    struct timeval stop_time;
    int total_time_us = 0;

    gettimeofday (&start_time, NULL);
#endif
    gboolean unmap_egl = FALSE;
    CUresult status;
    CUgraphicsResource cuda_buf_ptr = NULL;
    CUeglFrame eglFrame;

    if (ctx->is_integrated) {
        if (buf_ptr != NULL)
        {
            ret = NvBufSurfaceFromFd (buf_ptr->bufferDesc, (void**)(&surface));
            if(ret != 0)
            {
                return ret;
            }
        }
        else {
            buf_ptr = surface->surfaceList;
        }
        if(surface->memType == NVBUF_MEM_SURFACE_ARRAY)
        {
            is_nvmm_buf = true;
        }
    } else {
        if (buf_ptr == NULL)
        {
            buf_ptr = surface->surfaceList;
        }
    }

    #if defined(__aarch64__)
    if(ctx->is_integrated && is_nvmm_buf) {
        if (surface->surfaceList[0].mappedAddr.eglImage == NULL)
        {
            if (NvBufSurfaceMapEglImage(surface, 0) != 0)
            {
                NVOSD_PRINT_E ("Unable to map EGL Image");
                return -1;
            }
            unmap_egl = TRUE;
        }

        EGLImageKHR eglimage_src = surface->surfaceList[0].mappedAddr.eglImage;

        status = cuGraphicsEGLRegisterImage(&cuda_buf_ptr, eglimage_src,
                                            CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (status != CUDA_SUCCESS) {
            NVOSD_PRINT_E("cuGraphicsEGLRegisterImage failed : %d \n", status);
            return -1;
        }

        status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuda_buf_ptr, 0, 0);
        if (status != CUDA_SUCCESS)
        {
            NVOSD_PRINT_E("cuGraphicsSubResourceGetMappedArray failed\n");
            status = cuGraphicsUnregisterResource(cuda_buf_ptr);
            if (status != CUDA_SUCCESS) {
                NVOSD_PRINT_E("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
            }
            return -1;
        }
    }
    #endif

    uint8_t *ptr = NULL;
    uint8_t *ptr1 = NULL;

    if(ctx->is_integrated && is_nvmm_buf) {
        ptr = (uint8_t *) eglFrame.frame.pPitch[0];
        if (is_nv12 (buf_ptr[0].colorFormat)) {
            ptr1 = (uint8_t *) eglFrame.frame.pPitch[1];
        }
    }
    else {
        ptr = (uint8_t *) buf_ptr[0].dataPtr;
        if (is_nv12 (buf_ptr[0].colorFormat)) {
            ptr1 = (uint8_t *) ((unsigned char *)buf_ptr[0].dataPtr \
                    + buf_ptr[0].planeParams.offset[1]);
        }
    }

    int pitch = buf_ptr[0].planeParams.pitch[0];

    if (is_nv12 (buf_ptr[0].colorFormat)
            && buf_ptr[0].layout == NVBUF_LAYOUT_BLOCK_LINEAR) {
        // Create the surface objects for Y
        cudaResourceDesc resDesc;
        memset (&resDesc, 0, sizeof (cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = (cudaArray_t) ptr;
        ck(cudaCreateSurfaceObject (&YSurfObj, &resDesc));

        // Create the surface objects for UV
        resDesc.res.array.array = (cudaArray_t) ptr1;
        ck(cudaCreateSurfaceObject (&UVSurfObj, &resDesc));
        ptr = (uint8_t *) YSurfObj;
        ptr1 = (uint8_t *) UVSurfObj;
        destroy_surfaceOjbect = true;
    }

    cuOSDImageFormat format = cuOSDImageFormat::None;
    if (buf_ptr[0].colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
        format = cuOSDImageFormat::RGBA;
    } else if (is_nv12 (buf_ptr[0].colorFormat)
            && buf_ptr[0].layout == NVBUF_LAYOUT_PITCH) {
        format = cuOSDImageFormat::PitchLinearNV12;
    } else if (is_nv12 (buf_ptr[0].colorFormat)
            && buf_ptr[0].layout == NVBUF_LAYOUT_BLOCK_LINEAR) {
        format = cuOSDImageFormat::BlockLinearNV12;
    } else {
        printf ("wrong surface format: %d layout: %d\n", \
                buf_ptr[0].colorFormat, buf_ptr[0].layout);
        return -1;
    }
    //printf ("nvdsosd surface format: %d\n", (int)format);
#ifdef DEBUG_PERF
    gettimeofday (&stop_time, NULL);
    total_time_us = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
        (stop_time.tv_usec - start_time.tv_usec);
    printf("cuda map takes: %d us\n", total_time_us);
#endif

#ifdef CUDA_PERF
    cudaEvent_t start, end;
    ck(cudaEventCreate(&end));
    ck(cudaEventCreate(&start));

    ck(cudaStreamSynchronize(ctx->stream));
    ck(cudaEventRecord(start, ctx->stream));
#endif

    cuosd_apply(ctx->cuosd_context, ptr, ptr1, ctx->frame_width, pitch,
            ctx->frame_height, format, ctx->stream, true);

#ifdef CUDA_PERF
    float gpu_time;
    ck(cudaEventRecord(end, ctx->stream));
    ck(cudaEventSynchronize(end));
    ck(cudaEventElapsedTime(&gpu_time, start, end));
    printf("cuosd consumed time: %.2f ms\n", gpu_time);
    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(end));
#else
    ck(cudaStreamSynchronize(ctx->stream));
#endif
    ck(cudaGetLastError());

    ctx->mask_buf_offset = 0;

#ifdef DEBUG_PERF
    gettimeofday (&start_time, NULL);
#endif

    if(destroy_surfaceOjbect) {
        ck(cudaDestroySurfaceObject(YSurfObj));
        ck(cudaDestroySurfaceObject(UVSurfObj));
    }

    if(ctx->is_integrated && is_nvmm_buf) {
        status = cuGraphicsUnregisterResource(cuda_buf_ptr);
        if (status != CUDA_SUCCESS)
        {
            NVOSD_PRINT_E("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
        }

        if (unmap_egl)
        {
            if (NvBufSurfaceUnMapEglImage(surface, 0) != 0)
            {
                NVOSD_PRINT_E("Unable to unmap EGL Image");
                return -1;
            }
        }
    }

#ifdef DEBUG_PERF
    gettimeofday (&stop_time, NULL);
    total_time_us = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
        (stop_time.tv_usec - start_time.tv_usec);
    printf("cuda unmap takes: %d us\n", total_time_us);
#endif

    return ret;
}
