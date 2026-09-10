/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvTiler.h"
#include <gmodule.h>
#include "nvdsmeta_internal.h"
#include <string.h>

#include "nvtx_helper.h"
#include <functional>

#define ROUND_DOWN_2(n)  ((static_cast<uint32_t>(n))&(~1))

#define LOGW(...) printf(__VA_ARGS__)
#define LOGE(...) printf(__VA_ARGS__)

extern "C" {

typedef struct
{
    NvDsFrameMeta* frameMeta;
    int32_t sameSourceFrameMetaCount;
}FrameMetaFinder;

static gint FindIfAlreadyAcquired(gconstpointer node, gconstpointer customData);
}

NvTiler::NvTiler()
    : tilerConfig(),
      tiles(0),
      cacheCanvasMeta(nullptr),
      singleSourceModeON(false),
      singleSourceId(0),
      singleSourceNumSurfacesPerFrame(0),
      squareGridMode(0),
      tilerMap(nullptr),
      tilerMapLock(nullptr),
      cudaStream(0),
      mtxSingleSource(),
      MaxScaleFactorVIC(16.0f),
      MinScaleFactorVIC(1.0f / 16.0f)
{
}

bool NvTiler::Init(TilerConfig config)
{
    int currentDevice = -1;
    if(!config.columns
       || !config.rows
       || !config.width
       || !config.height)
    {
        return false;
    }

    /** If Re-initialization in progress */
    Deinit();

    /**
     * @default-config:
     * use same tileWidth X tileHeight for all tiles
     * if custom-tile-config is not passed with config
     * Also, (left, top) shall be based on sourceId as
     * mentioned below
     */
    {
        /** First, find the tile index;
         * Assumption: source_id starts from 0 and is a positive integer
         * increasing in steady ascending order for each additional source
         * (top, left) == (0, 0) is the first tile, (0, 1) the second tile in first row
         * (1, 0) the first tile in second row
         */
        uint32_t tileW = ROUND_DOWN_2(config.width / config.columns);
        uint32_t tileH = ROUND_DOWN_2(config.height / config.rows);
        if(!tileW || !tileH)
        {
            return false;
        }
        for(uint32_t i = 0; i < config.rows * config.columns; i++)
        {
            tiles.push_back(TileConfig((i % config.columns) * tileW,
                                       (i / config.columns) * tileH,
                                       tileW, tileH));
        }
    }

    if(config.customTileConfig.tiles)
    {
        /** Individual CustomTile configuration shall be as per
         * user's request
         * NOTE: If a sourceId is missing, flag WARN message
         */
        for(uint32_t i = 0; i < config.rows * config.columns; i++)
        {
            CustomTile* tileConfig = GetTileBySourceId(config.customTileConfig.tiles, config.customTileConfig.length, i);
            if(tileConfig)
            {
                /** Update default tileConfig to the user-configured values */
                tiles[i].left   = ROUND_DOWN_2(tileConfig->x * config.width);
                tiles[i].top    = ROUND_DOWN_2(tileConfig->y * config.height);
                tiles[i].width  = ROUND_DOWN_2(tileConfig->width * config.width);
                tiles[i].height = ROUND_DOWN_2(tileConfig->height * config.height);
            }
            else
            {
                /** Default tile position shall be used for sourceId=i;
                 * This might overlap with other user-configured tile positions
                 * So, better flag a WARNING */
                LOGW("[NvTiler::%s] WARN: Custom tile config unavailable for source%d"
                     " (if applicable); default-config shall be used\n", __func__, i);
            }
        }
    }

    tilerConfig = config;

    cacheCanvasMeta = nvds_create_batch_meta(tilerConfig.rows * tilerConfig.columns);
    if(!cacheCanvasMeta)
    {
        return false;
    }
    if (cudaGetDevice(&currentDevice) != cudaSuccess)
    {
        return false;
    }

    if (currentDevice != (int) config.gpuId){
      if (cudaSuccess != cudaSetDevice(config.gpuId)){
        return false;
      }
    }

    if(cudaSuccess != cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking))
    {
        return false;
    }

    if (currentDevice !=-1 && int(config.gpuId) != currentDevice){
      if (cudaSuccess != cudaSetDevice(currentDevice)){
        return false;
      }
    }

    return true;
}

bool NvTiler::ScaleAndPlaceAllMeta(NvDsBatchMeta* inputBufferMeta, NvBufSurfTransformRect* srcRects, NvBufSurfTransformRect* destRects)
{
    uint32_t frameIdx = 0;
    if(!inputBufferMeta || !srcRects || !destRects)
    {
        return false;
    }
    nvds_acquire_meta_lock(inputBufferMeta);
    for(GList* nodeFrame = inputBufferMeta->frame_meta_list; nodeFrame; nodeFrame = g_list_next(nodeFrame))
    {
        NvBufSurfTransformRect* srcRect  = &srcRects[frameIdx];
        NvBufSurfTransformRect* destRect = &destRects[frameIdx];
        float scaleX = static_cast<float>(destRect->width) / srcRect->width;
        float scaleY = static_cast<float>(destRect->height) / srcRect->height;
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(nodeFrame->data);
        if(GetSingleSourceMode() && (GetSingleSourceId() != frameMeta->pad_index))
        {
            /** Skip scaling for sources other than singleSourceId when ON */
            continue;
        }
        if(!scaleX || !scaleY || !frameMeta)
        {
            LOGE("[NvTiler::%s] ERROR: %d; scaleX=%f scaleY=%f frameMeta=%p\n",
                 __func__, __LINE__, scaleX, scaleY, frameMeta);
            nvds_release_meta_lock(inputBufferMeta);
            return false;
        }
        /** Scale all obj_meta_list in-place */
        for(GList* nodeObj = frameMeta->obj_meta_list; nodeObj; nodeObj = g_list_next(nodeObj))
        {
            NvDsObjectMeta* objMeta = static_cast<NvDsObjectMeta*>(nodeObj->data);
            if(objMeta)
            {
                ScaleRectParams(&objMeta->rect_params, destRect, scaleX, scaleY);
                ScaleTextParams(&objMeta->text_params, destRect, scaleX, scaleY);
            }
        }
        /** Scale all display_meta_list in-place */
        for(GList* nodeDisplay = frameMeta->display_meta_list; nodeDisplay; nodeDisplay = g_list_next(nodeDisplay))
        {
            NvDsDisplayMeta* displayMeta = static_cast<NvDsDisplayMeta*>(nodeDisplay->data);
            if(displayMeta)
            {
                for(uint32_t i = 0; i < displayMeta->num_rects; i++)
                {
                    ScaleRectParams(&displayMeta->rect_params[i], destRect, scaleX, scaleY);
                }
                for(uint32_t i = 0; i < displayMeta->num_labels; i++)
                {
                    ScaleTextParams(&displayMeta->text_params[i], destRect, scaleX, scaleY);
                }
                for(uint32_t i = 0; i < displayMeta->num_lines; i++)
                {
                    ScaleLineParams(&displayMeta->line_params[i], destRect, scaleX, scaleY);
                }
		for(uint32_t i = 0; i < displayMeta->num_circles; i++)
                {
                    ScaleCircleParams(&displayMeta->circle_params[i], destRect, scaleX, scaleY);
                }
		for(uint32_t i = 0; i < displayMeta->num_arrows; i++)
                {
                    ScaleArrowParams(&displayMeta->arrow_params[i], destRect, scaleX, scaleY);
                }
            }
        }
        frameIdx++;
    }
    nvds_release_meta_lock(inputBufferMeta);
    return true;
}


std::vector<NvBufSurfaceParams> NvTiler::GetSingleSourceSurfaces(NvBufSurface* inputBuffer, NvDsBatchMeta* inputBufferMeta)
{
    std::vector<NvBufSurfaceParams> singleSourceSurfaces;
    uint32_t i = 0;
    uint32_t j = 0;
    if(GetSingleSourceNumSurfacesPerFrame(inputBufferMeta, true) == 0)
    {
        /** This batch has no surfaces from singleSourceId */
        return singleSourceSurfaces;
    }
    nvds_acquire_meta_lock(inputBufferMeta);
    for(GList* node = inputBufferMeta->frame_meta_list; node; node = g_list_next(node))
    {
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(node->data);
        if(frameMeta->pad_index == GetSingleSourceId())
        {
            singleSourceSurfaces.push_back(inputBuffer->surfaceList[j]);
            i++;
            /** If we already got num_surfaces_per_frame surfaces; break!
             * Any following surfaces with pad_index matching required
             * sourceId can be safely ignored
             */
            if(i == GetSingleSourceNumSurfacesPerFrame(inputBufferMeta))
            {
                break;
            }
        }
        j++;
    }
    nvds_release_meta_lock(inputBufferMeta);

    return singleSourceSurfaces;
}

unsigned int frame_number = 0;

bool NvTiler::Composite(NvBufSurface*    inputBuffer,
               NvDsBatchMeta*   inputBufferMeta,
               NvBufSurface*    outputBuffer,
               NvBufSurface*    tilerScratchBuffer,
               uint32_t         gpuId,
               int32_t compute_hw,
               int32_t interpolation_method)
{
    NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
    bool ok;
    NvBufSurface inputBufferSingleSource;

    NvBufSurfTransformConfigParams transformSessionParams = {static_cast<NvBufSurfTransform_Compute>(compute_hw), static_cast<int32_t>(gpuId), cudaStream};
    memset (&compositeParams, 0, sizeof (compositeParams));
    if (NvBufSurfTransformSetSessionParams(&transformSessionParams) != NvBufSurfTransformError_Success)
    {
        LOGE("[NvTiler::%s] ERROR: %d; NvBufSurfTransformSetSessionParams failed\n", __func__, __LINE__);
        return false;
    }

    char context_name[100];

    std::scoped_lock lockNow(mtxSingleSource);
    std::vector<NvBufSurfaceParams> singleSourceSurfaces;

    /** inputBuffer holds the input batch buffers
     * which we shall composite to outputBuffer
     */
    /** flag to indicate composition/blending */
    compositeParams.composite_flag = NVBUFSURF_TRANSFORM_COMPOSITE;
    /** source rectangle coordinates of input buffers for composition. */
    /** and destination rectangle coordinates of input buffers for composition. */
    if(GetSingleSourceMode())
    {
        /** setup modified version of inputBuffer for single-source mode */
        memcpy(&inputBufferSingleSource, inputBuffer, sizeof(NvBufSurface));
        singleSourceSurfaces = GetSingleSourceSurfaces(inputBuffer, inputBufferMeta);
        inputBufferSingleSource.batchSize = inputBufferSingleSource.numFilled = singleSourceSurfaces.size();
        if(singleSourceSurfaces.size())
        {
            inputBufferSingleSource.surfaceList = &singleSourceSurfaces[0];
        }
        else
        {
            /** if singleSourceId unabailable, we can copy the older scratch */
            CopyNvBufSurface(tilerScratchBuffer, outputBuffer, interpolation_method);
            return true;
        }
        compositeParams.src_comp_rect = GetSingleSourceSrcCompositionRect(inputBuffer, inputBufferMeta);
        compositeParams.dst_comp_rect = GetSingleSourceDestCompositionRect(inputBufferMeta);
        inputBuffer = &inputBufferSingleSource;
    }
    else
    {
        compositeParams.src_comp_rect = GetSrcCompositionRects(inputBuffer);
        compositeParams.dst_comp_rect = GetDestCompositionRects(inputBufferMeta);
    }

#ifdef __aarch64__
    if(compositeParams.src_comp_rect && compositeParams.dst_comp_rect && ( compute_hw == 0 || compute_hw == 2 ))
    {
        ok = CheckVICMaxScaleFactorCompatibility(compositeParams.src_comp_rect, compositeParams.dst_comp_rect, inputBuffer->numFilled);
        if(!ok)
        {
            return false;
        }
    }
#endif

    if (inputBufferMeta->num_frames_in_batch > 0
        /** && We did have input frames from the sources that need
         * tiling; If we did not get any frames, src_comp_rect and dst_comp_rect
         * shall be nullptr; in which case, we shall copy tilerScratchBuffer to outputBuffer
         */
        && (compositeParams.src_comp_rect && compositeParams.dst_comp_rect)
        )
    {
        NvBufSurfTransformSyncObj_t sync_obj = NULL;
        ok = ScaleAndPlaceAllMeta(inputBufferMeta,
                    compositeParams.src_comp_rect, compositeParams.dst_comp_rect);
        if(!ok)
        {
            LOGE("[NvTiler::%s] ERROR: %d; ScaleAndPlaceAllMeta failed (%p;%p)\n",
                 __func__, __LINE__, compositeParams.src_comp_rect, compositeParams.dst_comp_rect);
            return false;
        }

        /** Add surfaces to tilerScratchBuffer->surfaceList to make sure underlying API works */
        compositeParams.composite_flag |=
            (NVBUFSURF_TRANSFORM_COMPOSITE_FILTER);
        compositeParams.composite_filter =
            static_cast<NvBufSurfTransform_Inter>(interpolation_method);

        nvtx_helper_push_pop (context_name);
        err = NvBufSurfTransformCompositeAsync(inputBuffer, tilerScratchBuffer, &compositeParams, &sync_obj);
        sync_objects.push_back(sync_obj);
        nvtx_helper_push_pop (NULL);
    }

    tilerScratchBuffer->numFilled = tilerScratchBuffer->batchSize = 1;

    if(err == NvBufSurfTransformError_Success)
    {
        /** Now, copy tilerScratchBuffer to outputBuffer */
        CopyNvBufSurface(tilerScratchBuffer, outputBuffer, interpolation_method);
    }
    else
    {
        LOGE("[NvTiler::%s] ERROR: %d; NvBufSurfTransformComposite failed(%d)\n", __func__, __LINE__, err);
        return false;
    }
    outputBuffer->numFilled = tilerScratchBuffer->numFilled;

    if (inputBufferMeta->num_frames_in_batch > 0)
    {
        /** Sync the inputBufferMeta to cacheCanvasMeta */
        return SyncBufferMeta(inputBufferMeta);
    }
    else
    {
        return true;
    }
}

bool NvTiler::CopyNvBufSurface(NvBufSurface* src, NvBufSurface* dest, int32_t interpolation_method)
{
    bool ok;
    NvBufSurfTransformParams transformParams;
    NvBufSurfTransformRect srcRect[1];
    NvBufSurfTransformRect destRect[1];
    NvBufSurfTransformSyncObj_t sync_obj = NULL;
    srcRect[0].top = srcRect[0].left = 0;
    destRect[0].top = destRect[0].left = 0;
    srcRect[0].width   = src->surfaceList[0].width;
    srcRect[0].height  = src->surfaceList[0].height;
    destRect[0].width  = dest->surfaceList[0].width;
    destRect[0].height = dest->surfaceList[0].height;
    transformParams.src_rect = &srcRect[0];
    transformParams.dst_rect = &destRect[0];
    transformParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transformParams.transform_flip = NvBufSurfTransform_None;
    transformParams.transform_filter = static_cast<NvBufSurfTransform_Inter>(interpolation_method);
    dest->numFilled = dest->batchSize = 1;
    ok = (NvBufSurfTransformError_Success == NvBufSurfTransformAsync(src, dest, &transformParams, &sync_obj));
    sync_objects.push_back(sync_obj);
    return ok;
}

bool NvTiler::AdjustOutputMeta(NvDsBatchMeta*   outputBatchMeta)
{
    /** now update the bufferMeta with saved latest unavailable source meta */
    return CacheCanvasCopyAllFrameMetaIntoOneNewFrame(outputBatchMeta);
}

uint32_t NvTiler::GetTileWidth(uint32_t sourceId)
{
    return tiles[sourceId].width;
}

uint32_t NvTiler::GetTileHeight(uint32_t sourceId)
{
    return tiles[sourceId].height;
}

void NvTiler::SyncObjWait()
{
    // Wait for the async calls to complete
    for(auto sync_object: sync_objects){
      NvBufSurfTransformSyncObjWait(sync_object, -1);
      NvBufSurfTransformSyncObjDestroy(&sync_object);
    }
    sync_objects.clear();

    if(compositeParams.src_comp_rect)
    {
        free(compositeParams.src_comp_rect);
    }

    if(compositeParams.dst_comp_rect)
    {
        free(compositeParams.dst_comp_rect);
    }

}

void NvTiler::Deinit()
{
    if(cacheCanvasMeta)
    {
        /** Re-initialization in progress */
        nvds_destroy_batch_meta(cacheCanvasMeta);
        cacheCanvasMeta = nullptr;
    }

    if(cudaStream)
    {
        cudaStreamDestroy(cudaStream);
    }

    /** Clear the individual tile configuration params */
    tiles.clear();
}

NvBufSurfTransformRect* NvTiler::GetSrcCompositionRects(NvBufSurface* inputBuffer)
{
    NvBufSurfTransformRect* transformRect = nullptr;

    if(!inputBuffer || (inputBuffer->numFilled == 0)
       || (inputBuffer->numFilled > inputBuffer->batchSize)
       || !(inputBuffer->surfaceList) )
    {
        return nullptr;
    }

    /** inputBuffer->numFilled shall represent the number of
     * surfaces in the batched buffer
     * There could be more than one surface per frame
     */
    transformRect = static_cast<NvBufSurfTransformRect*>(malloc(sizeof(NvBufSurfTransformRect) * inputBuffer->numFilled));
    if(!transformRect)
    {
        return nullptr;
    }

    for(uint32_t i = 0; i < inputBuffer->numFilled; i++)
    {
        transformRect[i].top = 0;
        transformRect[i].left = 0;
        transformRect[i].width = inputBuffer->surfaceList[i].width;
        transformRect[i].height = inputBuffer->surfaceList[i].height;
    }

    return transformRect;
}

NvBufSurfTransformRect* NvTiler::GetDestCompositionRects(NvDsBatchMeta* inputBufferMeta)
{
    NvBufSurfTransformRect* transformRect = nullptr;

    if(!inputBufferMeta || !inputBufferMeta->num_frames_in_batch)
    {
        return nullptr;
    }

    transformRect = static_cast<NvBufSurfTransformRect*>(malloc(sizeof(NvBufSurfTransformRect) * inputBufferMeta->num_frames_in_batch));
    if(!transformRect)
    {
        return nullptr;
    }

    nvds_acquire_meta_lock(inputBufferMeta);
    int32_t j = 0;

    for(uint32_t i = 0; i < inputBufferMeta->num_frames_in_batch; i++)
    {
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(g_list_nth_data(inputBufferMeta->frame_meta_list, i));
        #ifndef ENABLE_NVTILER_UNIT_TESTS
        /** TODO - remove this work-around code;
         * currently in to test dewarper with old muxer
         * This shall never happen
         * (num_surfaces_per_frame shall be >= 1)
         */
        if(frameMeta && frameMeta->num_surfaces_per_frame == 0)
        {
            frameMeta->num_surfaces_per_frame = 1;
        }
        #endif
        if(!frameMeta || (frameMeta->num_surfaces_per_frame == 0))
        {
            /** unavaliable or invalid frameMeta */
            nvds_release_meta_lock(inputBufferMeta);
            if(transformRect) {
                free(transformRect);
            }
            return nullptr;
        }
        uint32_t index = frameMeta->pad_index;
        if (squareGridMode) {
            g_mutex_lock(tilerMapLock);
            auto it = tilerMap->find(frameMeta->pad_index);
            if (it == tilerMap->end()) {
                printf ("Source ID Key not found in the tiler map.\n");
                g_mutex_unlock(tilerMapLock);
                nvds_release_meta_lock(inputBufferMeta);
                if(transformRect) {
                    free(transformRect);
                }
                return nullptr;
            }
            index = it->second;
            g_mutex_unlock(tilerMapLock);
        }
        transformRect[i].left   = (GetTileConfig(index).left);
        transformRect[i].top   = (GetTileConfig(index).top);

        /** NOTE:
         * Important Assumption - surfaces of the same frame shall be adjacent
         * in both NvDsBatchMeta->frame_meta_list[] and NvBufSurface->surfaceList[]
         * TODO - If above assumption is incorrect, we need
         * NvDsFrameMeta->surfaceIndex to know the index of this surface
         * in the frame
         */
        transformRect[i].top    += (ROUND_DOWN_2(GetTileHeight(index) / frameMeta->num_surfaces_per_frame) * (j));
        transformRect[i].width  =  GetTileWidth(index);
        transformRect[i].height =  ROUND_DOWN_2(GetTileHeight(index) / frameMeta->num_surfaces_per_frame);
        if(++j == frameMeta->num_surfaces_per_frame)
        {
            j = 0;
        }
    }
    nvds_release_meta_lock(inputBufferMeta);

    return transformRect;
}

void NvTiler::CacheCanvasFindAndDeleteFrameMetaBySource(uint32_t pad_index, bool deleteAll)
{
    nvds_acquire_meta_lock(cacheCanvasMeta);
    for(GList* node = cacheCanvasMeta->frame_meta_list; node; node = g_list_next(node))
    {
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(node->data);
        if(frameMeta && (deleteAll || (frameMeta->pad_index == pad_index)))
        {
            /** advance node iterator before removing the frameMeta element */
            node = g_list_next(node);
            nvds_release_meta_lock(cacheCanvasMeta);
            nvds_remove_frame_meta_from_batch(cacheCanvasMeta, frameMeta);
            nvds_acquire_meta_lock(cacheCanvasMeta);
        }
    }
    nvds_release_meta_lock(cacheCanvasMeta);
}

bool NvTiler::CacheCanvasCopyAllFrameMetaIntoOneNewFrame(NvDsBatchMeta* canvasBatchMeta)
{
    if(!canvasBatchMeta)
    {
        return false;
    }

    /** 1) Delete all frames */
    DeleteAllFrameMeta(canvasBatchMeta);

    /** 2) Create one NvDsFrameMeta */
    NvDsFrameMeta* canvasFrameMeta = static_cast<NvDsFrameMeta*>(nvds_acquire_frame_meta_from_pool(canvasBatchMeta));
    if(!canvasFrameMeta || (canvasBatchMeta->num_frames_in_batch != 0))
    {
        return false;
    }
    nvds_add_frame_meta_to_batch(canvasBatchMeta, canvasFrameMeta);
    if(canvasBatchMeta->num_frames_in_batch != 1)
    {
        return false;
    }

    /** Now cacheCanvasMeta has every frame (NvDsFrameMeta) and associated metadata
     * Move every NvDsFrameMeta-> (UPDATED - means copied into destination-meta canvasBatchMeta from cacheCanvasMeta):
     * a) source_id; - source_id of the frame in the batch shall be ambiguous and zero'd
     * b) pad_index; - Pad Index of the frame in the batch shall be zero'd
     * c) frame_num; - Frame number of the source - shall be zero'd
     * d) buf_pts;   - pts of the frame buffer ; zero'd (invalid post tiler)
     * e) ntp_timestamp - ntp timestamp ; zero'd (invalid post tiler)
     * f) num_obj_meta  - number of object meta attached to the current frame - UPDATED with actual
     * g) NvDsObjectMetaList *obj_meta_list; - A list of pointers of type “NvDsObjectMeta” - UPDATED
     * h) NvDisplayMetaList *display_meta_list; - A list of pointers of type “NvDsDisplayMeta” - UPDATED
     * i) NvDsUserMetaList *frame_user_meta_list; - user data to be set of type “NvDsUserMeta” - UPDATED
     * j) gint64 misc_frame_info[MAX_USER_FIELDS]; - For additional user frame info - zero'd
     * k) gint64 reserved[MAX_RESERVED_FIELDS]; - For internal purpose - zero'd
     * l) gint num_surfaces_per_frame; - UPDATED to 1
     */

    nvds_acquire_meta_lock(canvasBatchMeta);

    canvasFrameMeta->source_id              = 0;
    canvasFrameMeta->pad_index              = 0;
    canvasFrameMeta->buf_pts                = 0;
    canvasFrameMeta->ntp_timestamp          = 0;
    canvasFrameMeta->num_obj_meta           = 0;
    canvasFrameMeta->num_surfaces_per_frame = 1;
    memset(canvasFrameMeta->misc_frame_info, 0, sizeof(canvasFrameMeta->misc_frame_info));
    memset(canvasFrameMeta->reserved, 0, sizeof(canvasFrameMeta->reserved));

    nvds_acquire_meta_lock(cacheCanvasMeta);
    for(GList* node = cacheCanvasMeta->frame_meta_list; node; node = g_list_next(node))
    {
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(node->data);
        if(frameMeta)
        {
            /** if singleSourceModeON, skip every other frameMeta */
            if(GetSingleSourceMode() && frameMeta->pad_index != GetSingleSourceId())
            {
                continue;
            }

            /** UPDATE (f), (g); (h); (i) */
            nvds_release_meta_lock(canvasBatchMeta);
            nvds_release_meta_lock(cacheCanvasMeta);
            nvds_copy_obj_meta_list(frameMeta->obj_meta_list, canvasFrameMeta);
            nvds_copy_display_meta_list(frameMeta->display_meta_list, canvasFrameMeta);
            nvds_copy_frame_user_meta_list(frameMeta->frame_user_meta_list, canvasFrameMeta);
            nvds_acquire_meta_lock(canvasBatchMeta);
            nvds_acquire_meta_lock(cacheCanvasMeta);
        }
    }
    nvds_release_meta_lock(cacheCanvasMeta);

    nvds_release_meta_lock(canvasBatchMeta);

    return true;
}

void NvTiler::DeleteAllFrameMeta(NvDsBatchMeta* canvasBatchMeta)
{
    canvasBatchMeta->frame_meta_list = nvds_clear_meta_list(
        canvasBatchMeta,
        canvasBatchMeta->frame_meta_list,
        canvasBatchMeta->frame_meta_pool);
    nvds_acquire_meta_lock(canvasBatchMeta);
    canvasBatchMeta->num_frames_in_batch = 0;
    nvds_release_meta_lock(canvasBatchMeta);
}

bool NvTiler::SyncBufferMeta(NvDsBatchMeta* bufferMeta)
{
    if(!bufferMeta || !bufferMeta->num_frames_in_batch || !bufferMeta->max_frames_in_batch)
    {
        return false;
    }

    /** synchronize contents of bufferMeta->frame_meta_list
     *  with cacheCanvasMeta->frame_meta_list
     */
    /**
     * 1) for each incoming bufferMeta->frame_meta_list.get(i),
     * check if we have previous copy in cacheCanvasMeta->frame_meta_list
     * 2) If no - copy NvDsFrameMeta using the copy API and add into
     * cacheCanvasMeta (nvds_acquire_frame_meta_from_pool() + nvds_add_frame_meta_to_batch())
     * 3) If yes - copy NvDsFrameMeta content into the existing NvDsFrameMeta
     */

    nvds_acquire_meta_lock(bufferMeta);
    for(GList* node = bufferMeta->frame_meta_list; node; node = g_list_next(node))
    {
        if(!node->data)
        {
            nvds_release_meta_lock(bufferMeta);
            return false;
        }
        /** TODO: Here we delete all surface from a multi-surface source
         * Thus assume - each batch will have all surfaces from a source
         * if that source is present in the batch
         * Note: To avoid this assumption, we need to have a variable in NvDsFrameMeta
         * to identify which surface-of-a-frame and use that in conjunction with source_id
         * to delete corresponding meta in cache
         */
        /** Delete current meta */
        CacheCanvasFindAndDeleteFrameMetaBySource((static_cast<NvDsFrameMeta*>(node->data))->pad_index);
    }
    nvds_release_meta_lock(bufferMeta);

    /** Sync latest frameMeta's into cacheCanvasMeta
     * copy all the new frame-meta's to cacheCanvasMeta
     */
    CacheCanvasCopyFrameMetaWithoutSourceIdDuplication(bufferMeta->frame_meta_list);

    return true;
}

bool NvTiler::SetSingleSourceMode(bool on, int32_t sourceId)
{
    std::scoped_lock lockNow(mtxSingleSource);
    singleSourceId = static_cast<uint32_t>(sourceId);
    /** invalidate singleSourceNumSurfacesPerFrame as it needs
     * to be found out from the next available video batch buffer
     */
    singleSourceNumSurfacesPerFrame = 0;
    if(on && (GetSingleSourceId() >= tilerConfig.rows * tilerConfig.columns))
    {
        /** Cannot accomodate sourceId; it's invalid */
        LOGE("[NvTiler::%s] ERROR: Cannot accomodate sourceId=%d; it's invalid\n", __func__, GetSingleSourceId());
        return false;
    }
    singleSourceModeON = on;
    /** clear the cacheMeta for this source */
    CacheCanvasFindAndDeleteFrameMetaBySource(singleSourceId, true);
    return true;
}

bool NvTiler::GetSingleSourceMode()
{
    return singleSourceModeON;
}

uint32_t NvTiler::GetSingleSourceId()
{
    return singleSourceId;
}

uint32_t NvTiler::GetSingleSourceNumSurfacesPerFrame(NvDsBatchMeta* inputBufferMeta, bool forceCalc)
{
    if(singleSourceNumSurfacesPerFrame == 0 || forceCalc)
    {
        singleSourceNumSurfacesPerFrame = 0;
        /** Find the num_surfaces_per_frame for
         * selected source */
        for(GList* nodeFrame = inputBufferMeta->frame_meta_list; nodeFrame; nodeFrame = g_list_next(nodeFrame))
        {
            NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(nodeFrame->data);
            if(GetSingleSourceId() == frameMeta->pad_index)
            {
                singleSourceNumSurfacesPerFrame = frameMeta->num_surfaces_per_frame;
                return singleSourceNumSurfacesPerFrame;
            }
        }
        /** could still be 0 if inputBuffer has no surface for singleSourceId */
    }
    return singleSourceNumSurfacesPerFrame;
}

uint32_t NvTiler::GetSingleSourceSurfacesCount(NvDsBatchMeta* inputBufferMeta)
{
    uint32_t surfacesCount = 0;
    for(GList* nodeFrame = inputBufferMeta->frame_meta_list; nodeFrame; nodeFrame = g_list_next(nodeFrame))
    {
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(nodeFrame->data);
        if(GetSingleSourceId() == frameMeta->pad_index)
        {
            surfacesCount++;
        }
    }
    return surfacesCount;
}

NvBufSurfTransformRect* NvTiler::GetSingleSourceSrcCompositionRect(NvBufSurface* inputBuffer, NvDsBatchMeta* inputBufferMeta)
{
    NvBufSurfTransformRect* transformRect = nullptr;

    /** count the # of frames from single-source-id;
     * There shall be one frameMeta for each surface
     * So, if we have a frame with num_surfaces_per_frame=2,
     * then we shall have 2 X frameMeta's for one single frame.
     */
    uint32_t singleSourceSurfacesCount = 0;

    /* Split into separate checks to avoid side effects in right-hand operands
     * of logical || operators (cpp:S912 / MISRA C++ 5-14-1). */
    if(!inputBuffer)
    {
        return nullptr;
    }

    if((inputBuffer->numFilled == 0)
       || (inputBuffer->numFilled > inputBuffer->batchSize)
       || !(inputBuffer->surfaceList))
    {
        return nullptr;
    }

    if(!inputBufferMeta)
    {
        return nullptr;
    }

    if((inputBufferMeta->num_frames_in_batch != inputBuffer->numFilled)
       || !inputBufferMeta->frame_meta_list)
    {
        return nullptr;
    }

    singleSourceSurfacesCount = GetSingleSourceSurfacesCount(inputBufferMeta);

    if(singleSourceSurfacesCount == 0)
    {
        return nullptr;
    }

    /** inputBuffer->numFilled shall represent the number of
     * surfaces in the batched buffer
     * Allocating max space as there could be multiple frames from a single-source
     */
    transformRect = static_cast<NvBufSurfTransformRect*>(malloc(sizeof(NvBufSurfTransformRect) * singleSourceSurfacesCount));
    if(!transformRect)
    {
        return nullptr;
    }

    uint32_t j = 0;
    for(uint32_t i = 0; (i < inputBuffer->numFilled) && (j < singleSourceSurfacesCount); i++)
    {
        NvDsFrameMeta* frameMeta =  static_cast<NvDsFrameMeta*>(g_list_nth_data(inputBufferMeta->frame_meta_list, i));
        if(frameMeta->pad_index != GetSingleSourceId())
        {
            continue;
        }
        transformRect[j].top = 0;
        transformRect[j].left = 0;
        transformRect[j].width = inputBuffer->surfaceList[i].width;
        transformRect[j].height = inputBuffer->surfaceList[i].height;
        j++;
    }

    if(j > 0)
    {
        return transformRect;
    }

    free(transformRect);

    return nullptr;
}

NvBufSurfTransformRect* NvTiler::GetSingleSourceDestCompositionRect(NvDsBatchMeta* inputBufferMeta)
{
    NvBufSurfTransformRect* transformRect = nullptr;
    uint32_t singleSourceSurfacesCount = 0;

    if(!(singleSourceSurfacesCount = GetSingleSourceSurfacesCount(inputBufferMeta))
        || !GetSingleSourceNumSurfacesPerFrame(inputBufferMeta))
    {
        return nullptr;
    }

    transformRect = static_cast<NvBufSurfTransformRect*>(malloc(sizeof(NvBufSurfTransformRect) * singleSourceSurfacesCount));
    if(!transformRect)
    {
        return nullptr;
    }

    for(uint32_t j = 0; j < singleSourceSurfacesCount; j++)
    {
        /** There can be multiple frames (NvDsFrameMeta) from the same source whose num_surfaces_per_frame=1;
         * There can be multiple frames (NvDsFrameMeta) carrying the surfaces in one single camera-capture-frame
         */
        uint32_t k = j % GetSingleSourceNumSurfacesPerFrame(inputBufferMeta);
        transformRect[j].top = k * ROUND_DOWN_2(tilerConfig.height / GetSingleSourceNumSurfacesPerFrame(inputBufferMeta));
        transformRect[j].left = 0;
        transformRect[j].width = tilerConfig.width;
        transformRect[j].height = ROUND_DOWN_2(tilerConfig.height / GetSingleSourceNumSurfacesPerFrame(inputBufferMeta));
    }

    return transformRect;
}

void NvTiler::CacheCanvasCopyFrameMetaWithoutSourceIdDuplication(NvDsFrameMetaList* frameMetaList)
{
    /**
     * NOTE 1: If bufferMeta has multiple NvDsFrameMeta from same source,
     * we shall use the latest only
     * NOTE 2: Assumption - bufferMeta->frame_meta_list and buffer->surfaceList
     *         shall correspond to each other;
     *         Also that surfaceList[] and frame_meta_list[] are arranged chronologically
     */
    GList* acquiredSourceIds = nullptr;
    FrameMetaFinder frameMetaFinder;
    for(GList* node = g_list_last(frameMetaList); node; node = g_list_previous(node))
    {
        NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(node->data);
        frameMetaFinder.frameMeta = frameMeta;
        frameMetaFinder.sameSourceFrameMetaCount = 0;
        if(frameMeta)
        {
            if(GetSingleSourceMode() && (GetSingleSourceId() != frameMeta->pad_index))
            {
                /** skip sync'ing other streams when in singleSourceModeON */
                continue;
            }
            /** If this unique "frame"
             * already acquired; skip this one
             * NOTE: when frameMeta->num_surfaces_per_frame > 0, there can be
             * num_surfaces_per_frame X NvDsFrameMeta in frameMetaList
             * with same frameMeta->pad_index
             */
            if(g_list_find_custom(acquiredSourceIds, &frameMetaFinder, FindIfAlreadyAcquired))
            {
                continue; /**< skip frameMeta */
            }
            /** else add frameMeta to cacheCanvasMeta */
            NvDsFrameMeta* frameMetaCache = nvds_acquire_frame_meta_from_pool(cacheCanvasMeta);
            nvds_copy_frame_meta(frameMeta, frameMetaCache);
            nvds_add_frame_meta_to_batch(cacheCanvasMeta, frameMetaCache);
            /** Acquired frameMeta into cacheCanvasMeta! */
            acquiredSourceIds = g_list_prepend(acquiredSourceIds, GUINT_TO_POINTER(frameMeta->pad_index));
        }
    }
    g_list_free (acquiredSourceIds);
}

void NvTiler::ScaleRectParams(NvOSD_RectParams* rect_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY)
{
    /** scale width and height of rects */
    rect_params->width  *= scaleX;
    rect_params->height *= scaleY;
    /** scale and place position co-ordinates */
    rect_params->left = ((rect_params->left * scaleX)
                              + destRect->left);
    rect_params->top = ((rect_params->top * scaleY)
                              + destRect->top);
}

void NvTiler::ScaleTextParams(NvOSD_TextParams* text_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY)
{
    text_params->x_offset = ((text_params->x_offset * scaleX)
                              + destRect->left);
    text_params->y_offset = ((text_params->y_offset * scaleY)
                              + destRect->top);
}

void NvTiler::ScaleLineParams(NvOSD_LineParams* line_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY)
{
    line_params->x1 = ((line_params->x1 * scaleX)
                              + destRect->left);
    line_params->x2 = ((line_params->x2 * scaleX)
                              + destRect->left);
    line_params->y1 = ((line_params->y1 * scaleY)
                              + destRect->top);
    line_params->y2 = ((line_params->y2 * scaleY)
                              + destRect->top);
}

void NvTiler::ScaleCircleParams(NvOSD_CircleParams* circle_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY)
{
    float min_scale = (scaleX > scaleY) ? scaleY : scaleX;
    circle_params->xc = ((circle_params->xc * scaleX)
                                + destRect->left);
    circle_params->yc = ((circle_params->yc * scaleY)
                                + destRect->top);
    circle_params->radius = circle_params->radius * min_scale;
}

void NvTiler::ScaleArrowParams(NvOSD_ArrowParams* arrow_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY)
{
    arrow_params->x1 = ((arrow_params->x1 * scaleX)
                              + destRect->left);
    arrow_params->x2 = ((arrow_params->x2 * scaleX)
                              + destRect->left);
    arrow_params->y1 = ((arrow_params->y1 * scaleY)
                              + destRect->top);
    arrow_params->y2 = ((arrow_params->y2 * scaleY)
                              + destRect->top);
}

TileConfig NvTiler::GetTileConfig(uint32_t sourceId)
{
    return tiles[sourceId];
}

CustomTile* NvTiler::GetTileBySourceId(CustomTile* tiles, uint32_t tileConfigSize,
        uint32_t sourceId)
{
    if(!tiles)
    {
        return nullptr;
    }
    for(uint32_t i = 0; i < tileConfigSize; i++)
    {
        if(tiles[i].sourceId == sourceId)
        {
            /** Basic sanity on tileConfig */
            if(tiles[i].x + tiles[i].width > 1.0)
            {
                LOGE("[NvTiler::%s] ERROR: Source%d tile-config width exceed canvas (%f, %f)\n", __func__, i, tiles[i].x, tiles[i].height);
                return nullptr;
            }
            if(tiles[i].y + tiles[i].height > 1.0)
            {
                LOGE("[NvTiler::%s] ERROR: Source%d tile-config width exceed canvas (%f, %f)\n", __func__, i, tiles[i].x, tiles[i].height);
                return nullptr;
            }
            return &tiles[i];
        }
    }
    return nullptr;
}

void NvTiler::DeleteSource(int32_t sourceId)
{
    CacheCanvasFindAndDeleteFrameMetaBySource(sourceId);
}

bool NvTiler::SetSquareGridMode(bool squareGrid)
{
    squareGridMode = squareGrid;
    return true;
}

bool NvTiler::SetTilerMap(std::map<uint32_t, uint32_t> *tiler_map, GMutex *tiler_map_lock)
{
    if (tiler_map == nullptr) {
        printf("Tiler Map is NULL\n");
        return false;
    }
    tilerMap = tiler_map;
    tilerMapLock = tiler_map_lock;
    return true;
}

bool NvTiler::CheckVICMaxScaleFactorCompatibility(NvBufSurfTransformRect const * srcRects,
        NvBufSurfTransformRect const * dstRects, uint32_t const numFrames)
{
    /** Check if any of the required scaling operation
     * exceed the max scale factor supported by VIC
     */
    for(uint32_t i = 0; i < numFrames; i++)
    {
        if( !( (static_cast<float>(srcRects[i].width) * MinScaleFactorVIC) <=  static_cast<float>(dstRects[i].width) )
            || !( (static_cast<float>(srcRects[i].height) * MinScaleFactorVIC) <=  static_cast<float>(dstRects[i].height) )
            || !( (static_cast<float>(srcRects[i].width) * MaxScaleFactorVIC) >=  static_cast<float>(dstRects[i].width) )
            || !( (static_cast<float>(srcRects[i].height) * MaxScaleFactorVIC) >=  static_cast<float>(dstRects[i].height) )
          )
        {
            LOGE("[NvTiler::%s] ERROR: Scaling requested from [%d X %d] to [%d X %d] "
                 "exceeds max scaling factor (%f) supported by VIC\n",
                 __func__, srcRects[i].width, srcRects[i].height,
                dstRects[i].width, dstRects[i].height, MaxScaleFactorVIC);
            LOGE("[NvTiler::%s] Please use a larger tiled window resolution or reduce number of streams\n" , __func__);
            return false;
        }
    }
    return true;
}

TileConfig::TileConfig()
    : left(0),
      top(0),
      width(0),
      height(0)
{
}

TileConfig::TileConfig(uint32_t l, uint32_t t, uint32_t w, uint32_t h)
    : left(l),
      top(t),
      width(w),
      height(h)
{
}

gint FindIfAlreadyAcquired(gconstpointer node, gconstpointer customData)
{
    /** This API is used from CacheCanvasCopyFrameMetaWithoutSourceIdDuplication()
     * and is called for as-many entries in acquiredSourceIds
     * when g_list_find_custom() is invoked
     */
    uint32_t sourceId = GPOINTER_TO_UINT(node);
    FrameMetaFinder* frameMetaFinder = const_cast<FrameMetaFinder*>(reinterpret_cast<FrameMetaFinder const *>(customData));
    if(frameMetaFinder->frameMeta->pad_index == sourceId)
    {
        frameMetaFinder->sameSourceFrameMetaCount++;
        /** Count the sourceId entry in acquiredSourceIds;
         * This entry is already copied
         */
        if(frameMetaFinder->sameSourceFrameMetaCount
           >= frameMetaFinder->frameMeta->num_surfaces_per_frame)
        {
            /** this frame is part of another (camera) capture
             * we already acquired num_surfaces_per_frame X frameMeta's
             * So reject this!
             */
            return 0;
        }
    }
    return 1; /** this is a unique frame */
}

