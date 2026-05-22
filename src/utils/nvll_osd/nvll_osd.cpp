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

#include "nvll_osd_api.h"
#include "nvll_osd_int.h"

#define NVOSD_MAX_NUM_RECTS 128
#define NVBUF_ALIGN_VAL (512)

#define NVBUF_ALIGN_PITCH(pitch, align_val) ((pitch%align_val == 0) ? pitch:((pitch/align_val + 1)*align_val))

#define NVBUF_PLATFORM_ALIGNED_PITCH(pitch) NVBUF_ALIGN_PITCH(pitch, NVBUF_ALIGN_VAL)

bool color_compare(NvOSD_RectParams lhs, NvOSD_RectParams rhs)
{
    return lhs.reserved < rhs.reserved;
}

int CopytoHostMem(void *nvosd_ctx, void *cuDevPtr)
{
  NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
  cudaError_t CUerr = cudaSuccess;

  CUerr = cudaMemcpy2D (ctx->conv_buf, (ctx->frame_width * 4), (void *)cuDevPtr,
      NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4), (ctx->frame_width * 4), (ctx->frame_height), cudaMemcpyDeviceToHost);

  if(CUerr != cudaSuccess)
  {
    return FALSE;
  }
  return TRUE;
}

int CopytoDeviceMem(void *nvosd_ctx, void *cuDevPtr)
{
  NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaMemcpy2D ((void *)cuDevPtr, NVBUF_PLATFORM_ALIGNED_PITCH(ctx->frame_width * 4), ctx->conv_buf,
      (ctx->frame_width * 4), (ctx->frame_width * 4), (ctx->frame_height), cudaMemcpyHostToDevice);

  if(CUerr != cudaSuccess)
  {
    return FALSE;
  }
  return TRUE;
}

NvOSDFdMap *set_cairo_context(NvOSD_Ctx *ctx, NvBufSurfaceParams *buf_ptr, NvBufSurface *nvbuf_surf)
{
    NvOSDFdMap *nvosd_map = NULL;
    if(ctx->is_integrated)
    {
        int err = 0;
        if (buf_ptr != NULL)
        {
            NvBufSurface *nvbuf_surf = NULL;
            err = NvBufSurfaceFromFd (buf_ptr->bufferDesc, (void**)(&nvbuf_surf));
            if(err != 0)
            {
                return NULL;
            }
        }
        auto map_entry = ctx->map_list->find (nvbuf_surf->surfaceList[0].bufferDesc);

        if (map_entry == ctx->map_list->end() || !nvbuf_surf->surfaceList[0].mappedAddr.addr[0])
        {
            err = NvBufSurfaceMap (nvbuf_surf, -1, -1, NVBUF_MAP_READ_WRITE);
            if (err != 0)
            {
                NVOSD_PRINT_E("Buffer mapping failed \n");
                return NULL;
            }

            if (map_entry != ctx->map_list->end())
            {
                nvosd_map = (NvOSDFdMap *) map_entry->second;
                cairo_destroy (nvosd_map->cairo_context);
                cairo_surface_destroy (nvosd_map->surface);
                free(nvosd_map);
                ctx->map_list->erase (map_entry);
            }

            nvosd_map = (NvOSDFdMap *)calloc(1, sizeof(NvOSDFdMap));

            nvosd_map->surface = cairo_image_surface_create_for_data
                ((unsigned char *)nvbuf_surf->surfaceList[0].mappedAddr.addr[0], CAIRO_FORMAT_ARGB32, nvbuf_surf->surfaceList[0].width,
                nvbuf_surf->surfaceList[0].height, nvbuf_surf->surfaceList[0].pitch);

            nvosd_map->dmabuf_fd = nvbuf_surf->surfaceList[0].bufferDesc;
            nvosd_map->size = nvbuf_surf->surfaceList[0].dataSize;
            nvosd_map->mapping = (void *)nvbuf_surf->surfaceList[0].mappedAddr.addr[0];
            nvosd_map->cairo_context = cairo_create (nvosd_map->surface);
            nvosd_map->surf = nvbuf_surf;

            ctx->map_list->insert (std::make_pair (nvbuf_surf->surfaceList[0].bufferDesc, nvosd_map));
        }
        else
        {
            nvosd_map = (NvOSDFdMap *) map_entry->second;
        }

        err = NvBufSurfaceSyncForCpu (nvbuf_surf, -1, -1);

        if(err != 0) {
            NVOSD_PRINT_E("Cache sync failed \n");
            return NULL;
        }
    }
    else if (!ctx->is_integrated)
    {
        if (!ctx->nvosd_map)
        {
            nvosd_map = (NvOSDFdMap *)calloc(1, sizeof(NvOSDFdMap));

            //stride = cairo_format_stride_for_width (CAIRO_FORMAT_ARGB32, pSurf[0].Width);

            nvosd_map->surface = cairo_image_surface_create_for_data
                ((unsigned char *)ctx->conv_buf, CAIRO_FORMAT_ARGB32, ctx->frame_width,
                ctx->frame_height, ctx->frame_width * 4);

            nvosd_map->mapping = (void *)ctx->conv_buf;
            nvosd_map->cairo_context = cairo_create (nvosd_map->surface);
            nvosd_map->surf = nvbuf_surf;
            ctx->nvosd_map  = nvosd_map;
        }
        else
        {
            nvosd_map = ctx->nvosd_map;
        }
    }

    return nvosd_map;
}

static int is_device_integrated()
{
    int is_nvgpu = 0;
    NvBufSurfaceDeviceInfo dev_info{};
    if (NvBufSurfaceGetDeviceInfo (&dev_info) == 0) {
        if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
            is_nvgpu = 1;
        }
    }
    return is_nvgpu;
}

/* nvll_osd.h APIs */

NvOSDCtxHandle nvll_osd_create_context()
{
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)calloc(1, sizeof(NvOSD_Ctx));

    if(!ctx)
    {
        NVOSD_PRINT_E("Can not NVOSD create ctx");
    }

    ctx->params = (NvBufSurfacePlaneParams *)calloc(1, sizeof(NvBufSurfacePlaneParams));
    ctx->is_integrated = is_device_integrated();

    if(ctx->is_integrated) {
        ctx->map_list = new std::unordered_map <int, void *>;

        ctx->solid_rect_params_list = (NvOSD_RectParams *)calloc(1,
                (sizeof(NvOSD_RectParams) * NVOSD_MAX_NUM_RECTS));

        ctx->alpha_rect_params_list = (NvOSD_RectParams *)calloc(1,
                (sizeof(NvOSD_RectParams) * NVOSD_MAX_NUM_RECTS));
    }

    ctx->frameNum = 1;
    ctx->elapsedTime = 0;
    ctx->fps = 0;
    ctx->avg_fps = 0;
    ctx->display_fps = 0;
    ctx->mask_buf = NULL;
    ctx->mask_buf_size = 0;
    ctx->mask_buf_offset = 0;

    ck(cudaStreamCreateWithFlags(&(ctx->stream), cudaStreamNonBlocking));

    ctx->cuosd_context = cuosd_context_create();
    if(!ctx->cuosd_context)
    {
        NVOSD_PRINT_E("Can not create cuosd context");
    }

    return ctx;
}

void nvll_osd_destroy_context(NvOSDCtxHandle ctx)
{
    NvOSD_Ctx *pctx = (NvOSD_Ctx *)ctx;
    NvOSDFdMap *nvosd_map = NULL;
    if(pctx->is_integrated) {
        if(pctx->solid_rect_params_list)
        {
            free(pctx->solid_rect_params_list);
            pctx->solid_rect_params_list = NULL;
        }

        if(pctx->alpha_rect_params_list)
        {
            free(pctx->alpha_rect_params_list);
            pctx->alpha_rect_params_list = NULL;
        }
    }
    else if(!pctx->is_integrated ) {
        cudaError_t CUerr = cudaSuccess;
        NvOSDFdMap *nvosd_map = pctx->nvosd_map;

        if (nvosd_map) {
            cairo_destroy (nvosd_map->cairo_context);
            cairo_surface_destroy (nvosd_map->surface);
            free (nvosd_map);
        }

        if(CUerr != cudaSuccess)
        {
            return;
        }
    }

    if(pctx->conv_buf)
    {
        cudaFreeHost(pctx->conv_buf);
        pctx->conv_buf = NULL;
    }

    if(pctx->params)
    {
        free(pctx->params);
        pctx->params = NULL;
    }

    if(pctx->cuosd_context)
    {
        cuosd_context_destroy(pctx->cuosd_context);
        pctx->cuosd_context = NULL;
    }

    if(pctx->mask_buf)
    {
        ck(cudaFree(pctx->mask_buf));
        pctx->mask_buf = NULL;
    }

    if(pctx->is_integrated) {
        int err = 0;
        NvBufSurface *nvbuf_surf = NULL;

        if (pctx->map_list->size())
        {
            for (auto map_entry = pctx->map_list->begin();
                    map_entry != pctx->map_list->end(); ++map_entry)
            {
                nvosd_map = (NvOSDFdMap *) map_entry->second;
                err = NvBufSurfaceFromFd (nvosd_map->dmabuf_fd, (void**)(&nvbuf_surf));
                if (err != 0)
                {
                    return;
                }

                err = NvBufSurfaceUnMap (nvbuf_surf, -1, -1);
                if (err != 0)
                {
                    return;
                }

                cairo_destroy (nvosd_map->cairo_context);
                cairo_surface_destroy (nvosd_map->surface);
                free(nvosd_map);
            }
            pctx->map_list->clear();
        }
        if (pctx->map_list)
        {
            delete pctx->map_list;
            pctx->map_list = NULL;
        }

        if(pctx)
        {
            ck(cudaStreamDestroy(pctx->stream));
            free(pctx);
            pctx = NULL;
        }
    }
    else if(!pctx->is_integrated) {
        ck(cudaStreamDestroy(pctx->stream));
        free(ctx);
    }
}

/* sets system date and time to be embedded */
void nvll_osd_set_clock_params(NvOSDCtxHandle nvosd_ctx, NvOSD_TextParams *clk_params)
{
    NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
    ctx->enable_clock = 0;

    if(clk_params)
    {
        ctx->enable_clock = 1;
        ctx->clk_params = *clk_params;
    }
}

int nvll_osd_put_text(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameTextParams *frame_text_params)
{
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;
    if(frame_text_params->mode == MODE_CPU)
    {
        if(!ctx->is_integrated) {
	    CopytoHostMem(nvosd_ctx, (void *)frame_text_params->buf_ptr ? frame_text_params->buf_ptr[0].dataPtr : frame_text_params->surf->surfaceList[0].dataPtr);
        }
        ret = nvll_osd_put_text_cpu(nvosd_ctx, frame_text_params);
        if(!ctx->is_integrated) {
            CopytoDeviceMem(nvosd_ctx, (void *)frame_text_params->buf_ptr ? frame_text_params->buf_ptr[0].dataPtr : frame_text_params->surf->surfaceList[0].dataPtr);
        }
    }
    else if(frame_text_params->mode == MODE_GPU)
    {
        ret = nvll_osd_put_text_gpu(nvosd_ctx, frame_text_params);
    }
    return ret;
}

int nvll_osd_draw_segment_masks(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameSegmentMaskParams *frame_mask_params)
{
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;
    if(frame_mask_params->mode == MODE_CPU)
    {
        if(!ctx->is_integrated) {
            CopytoHostMem(nvosd_ctx, (void *)frame_mask_params->buf_ptr ? frame_mask_params->buf_ptr[0].dataPtr : frame_mask_params->surf->surfaceList[0].dataPtr);
        }
        ret = nvll_osd_draw_segment_masks_cpu(nvosd_ctx, frame_mask_params);
        if(!ctx->is_integrated) {
            CopytoDeviceMem(nvosd_ctx, (void *)frame_mask_params->buf_ptr ? frame_mask_params->buf_ptr[0].dataPtr : frame_mask_params->surf->surfaceList[0].dataPtr);
        }
    }
    else if(frame_mask_params->mode == MODE_GPU)
    {
        ret = nvll_osd_draw_segment_masks_gpu(nvosd_ctx, frame_mask_params);
    }
    return ret;
}

int nvll_osd_blur_rectangles(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params)
{
    int ret = 0;
    if(frame_rect_params->mode == MODE_GPU)
    {
        ret = nvll_osd_blur_rectangles_gpu(nvosd_ctx, frame_rect_params);
    } else {
        NVOSD_PRINT_E("Blur is only supported on GPU mode, please set the process-mode=1!");
        ret = -1;
    }
    return ret;
}

int nvll_osd_draw_rectangles(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params)
{
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;
    if(frame_rect_params->mode == MODE_CPU)
    {
        if(!ctx->is_integrated) {
            CopytoHostMem(nvosd_ctx, (void *)frame_rect_params->buf_ptr ? frame_rect_params->buf_ptr[0].dataPtr : frame_rect_params->surf->surfaceList[0].dataPtr);
        }
        ret = nvll_osd_draw_rectangles_cpu(nvosd_ctx, frame_rect_params);
        if(!ctx->is_integrated) {
            CopytoDeviceMem(nvosd_ctx, (void *)frame_rect_params->buf_ptr ? frame_rect_params->buf_ptr[0].dataPtr : frame_rect_params->surf->surfaceList[0].dataPtr);
        }
    }
    else if(frame_rect_params->mode == MODE_GPU)
    {
        ret = nvll_osd_draw_rectangles_gpu(nvosd_ctx, frame_rect_params);
    }
    return ret;
}

int nvll_osd_draw_arrows(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameArrowParams *frame_arrow_params)
{
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;

    if(frame_arrow_params->mode == MODE_CPU) {
        if(!ctx->is_integrated) {
            CopytoHostMem(nvosd_ctx, (void *)frame_arrow_params->buf_ptr ? frame_arrow_params->buf_ptr[0].dataPtr : frame_arrow_params->surf->surfaceList[0].dataPtr);
        }
        ret = nvll_osd_draw_arrows_cpu(nvosd_ctx, frame_arrow_params);
        if(!ctx->is_integrated) {
            CopytoDeviceMem(nvosd_ctx, (void *)frame_arrow_params->buf_ptr ? frame_arrow_params->buf_ptr[0].dataPtr : frame_arrow_params->surf->surfaceList[0].dataPtr);
        }
    }
    else if(frame_arrow_params->mode == MODE_GPU)
        ret = nvll_osd_draw_arrows_gpu(nvosd_ctx, frame_arrow_params);
    return ret;
}

int nvll_osd_draw_circles(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameCircleParams *frame_circle_params)
{
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;

    if(frame_circle_params->mode == MODE_CPU) {
        if(!ctx->is_integrated) {
            CopytoHostMem(nvosd_ctx, (void *)frame_circle_params->buf_ptr ? frame_circle_params->buf_ptr[0].dataPtr : frame_circle_params->surf->surfaceList[0].dataPtr);
        }
        ret = nvll_osd_draw_circles_cpu(nvosd_ctx, frame_circle_params);
        if(!ctx->is_integrated) {
            CopytoDeviceMem(nvosd_ctx, (void *)frame_circle_params->buf_ptr ? frame_circle_params->buf_ptr[0].dataPtr : frame_circle_params->surf->surfaceList[0].dataPtr);
        }
    }
    else if(frame_circle_params->mode == MODE_GPU)
        ret = nvll_osd_draw_circles_gpu(nvosd_ctx, frame_circle_params);
    return ret;
}

int nvll_osd_draw_lines(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameLineParams *frame_line_params)
{
    int ret = 0;
    NvOSD_Ctx *ctx = (NvOSD_Ctx *)nvosd_ctx;

    if(frame_line_params->mode == MODE_CPU) {
        if(!ctx->is_integrated) {
            CopytoHostMem(nvosd_ctx, (void *)frame_line_params->buf_ptr ? frame_line_params->buf_ptr[0].dataPtr : frame_line_params->surf->surfaceList[0].dataPtr);
        }
        ret = nvll_osd_draw_lines_cpu(nvosd_ctx, frame_line_params);
        if(!ctx->is_integrated) {
            CopytoDeviceMem(nvosd_ctx, (void *)frame_line_params->buf_ptr ? frame_line_params->buf_ptr[0].dataPtr : frame_line_params->surf->surfaceList[0].dataPtr);
        }
    }
    else if(frame_line_params->mode == MODE_GPU)
        ret = nvll_osd_draw_lines_gpu(nvosd_ctx, frame_line_params);
    return ret;
}

void *nvll_osd_set_params (NvOSDCtxHandle nvosd_ctx, int width, int height)
{
  cudaError_t CUerr = cudaSuccess;
  NvOSD_Ctx * ctx = (NvOSD_Ctx *)nvosd_ctx;
  ctx->frame_width = width;
  ctx->frame_height = height;

  if(ctx->conv_buf)
  {
    cudaFreeHost(ctx->conv_buf);
    ctx->conv_buf = NULL;
  }

  if(CUerr != cudaSuccess)
  {
    return NULL;
  }

  CUerr = cudaMallocHost (&ctx->conv_buf, width * height * 4);

  if(CUerr != cudaSuccess)
  {
    return NULL;
  }
  return ctx->conv_buf;
}

int nvll_osd_apply(NvOSDCtxHandle nvosd_ctx, NvBufSurfaceParams *buf_ptr, NvBufSurface *surf)
{
    return nvll_osd_gpu_apply(nvosd_ctx, buf_ptr, surf);
}
