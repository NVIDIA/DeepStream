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

#ifndef _NVTILER_H_
#define _NVTILER_H_

#include "INvTiler.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurftransform.h"
#include <mutex>

class TileConfig
{
    public:

    uint32_t left;
    uint32_t top;
    uint32_t width;
    uint32_t height;

    TileConfig();
    TileConfig(uint32_t l, uint32_t t, uint32_t w, uint32_t h);
};

class NvTiler : public INvTiler
{
    public:

    NvTiler();

    bool Init(TilerConfig config);

    /**
     * @brief  TilerComposite helps composite an input batch onto outputBuffer
     * @param  inputBuffer [IN]
     * @param  inputBufferMeta [IN]
     * @param  outputBuffer [IN/OUT] Output tiler composited buffer
     * @param  tilerScratchBuffer [IN] Used to keep a copy of the generated tile
     *         for future reference
     * @param  gpu_id [IN] GPU to be used for tiling
     * @param  stream [IN] cudaStream to be used for tiling
     */
    bool Composite(NvBufSurface*    inputBuffer,
                   NvDsBatchMeta*   inputBufferMeta,
                   NvBufSurface*    outputBuffer,
                   NvBufSurface*    tilerScratchBuffer,
                   uint32_t         gpuId,
                   int32_t compute_hw,
                   int32_t interpolation_method);
    /**
     * @brief  Scale every bounding box meta to match tile resolution
     * @param  outputBatchMeta [IN/OUT] Output tiler composited buffer meta
     *         with stream-display-specific meta-data.
     *         The function shall modify outputBatchMeta with
     *         meta-data like bounding boxes scaled appropriately.
     */
    bool AdjustOutputMeta(NvDsBatchMeta*   outputBatchMeta);

    uint32_t GetTileWidth(uint32_t sourceId);

    uint32_t GetTileHeight(uint32_t sourceId);

    void SyncObjWait();

    void Deinit();

    bool SetSingleSourceMode(bool on, int32_t sourceId=-1);

    void DeleteSource(int32_t sourceId);

    bool SetSquareGridMode (bool squareGrid);

    bool SetTilerMap (std::map<uint32_t, uint32_t> *tiler_map, GMutex *tiler_map_lock);

    private:

    /**
     * @brief  Get Source Composition rect array
     * @param  inputBuffer [IN] The input NvBufSurface which holds
     *         input batch to be composited to the tile-canvas
     * @return [transfer full] Array of source composition rects
     *         for each frame in the inputBuffer->surfaceList
     */
    NvBufSurfTransformRect* GetSrcCompositionRects(NvBufSurface* inputBuffer);

    /**
     * @brief  Get Destination Composition rect array
     *         Any bbox scaling shall be done and saved in @inputBufferMeta
     * @param  inputBuffer [IN] The input NvBufSurface which holds
     *         input batch to be composited to the tile-canvas
     * @param  inputBufferMeta [IN] Carrying specific information on source-ID
     *         that is crucial in deciding the target tile location
     * @return [transfer full] Array of source composition rects
     *         for each surface in the inputBuffer->surfaceList
     *         Note: There could be multiple surfaces which are part of a
     *         single source (frame). This fact shall be known by the user
     *         of this API
     */
    NvBufSurfTransformRect* GetDestCompositionRects(NvDsBatchMeta* inputBufferMeta);

    void ScaleRectParams(NvOSD_RectParams* rect_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY);
    void ScaleTextParams(NvOSD_TextParams* text_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY);
    void ScaleLineParams(NvOSD_LineParams* line_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY);
    void ScaleCircleParams(NvOSD_CircleParams* circle_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY);
    void ScaleArrowParams(NvOSD_ArrowParams* arrow_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY);

    bool ScaleAndPlaceAllMeta(NvDsBatchMeta* inputBufferMeta, NvBufSurfTransformRect* srcRects, NvBufSurfTransformRect* destRects);

    /**
     * @brief  Sync the bufferMeta into cacheCanvasMeta
     *         and add from cacheCanvasMeta unavailable source-surface
     *         meta back into in bufferMeta
     * @param  bufferMeta [IN] the current input buffer batch meta
     */
    bool SyncBufferMeta(NvDsBatchMeta* bufferMeta);

    void CacheCanvasFindAndDeleteFrameMetaBySource(uint32_t sourceId, bool deleteAll=false);

    void DeleteAllFrameMeta(NvDsBatchMeta* canvasBatchMeta);

    /**
     * @brief  Copy all cacheCanvasMeta->frame_meta_list contents to
     *         one NvDsFrameMeta in provided canvasBatchMeta
     *         Any frame_meta_list in canvasBatchMeta will be deleted
     * @param  canvasBatchMeta [IN] The destination batchMeta for
     *         copying all NvDsFrameMeta info into
     * @return true if successful, false otherwise
     */
    bool CacheCanvasCopyAllFrameMetaIntoOneNewFrame(NvDsBatchMeta* canvasBatchMeta);

    NvBufSurfTransformRect* GetSingleSourceSrcCompositionRect(NvBufSurface* inputBuffer, NvDsBatchMeta* inputBufferMeta);

    NvBufSurfTransformRect* GetSingleSourceDestCompositionRect(NvDsBatchMeta* inputBufferMeta);

    bool GetSingleSourceMode();
    uint32_t GetSingleSourceId();
    uint32_t GetSingleSourceNumSurfacesPerFrame(NvDsBatchMeta* inputBufferMeta, bool forceCalc=false);
    /** @return The number of surfaces from the single source singleSourceId */
    uint32_t GetSingleSourceSurfacesCount(NvDsBatchMeta* inputBufferMeta);

    void CacheCanvasCopyFrameMetaWithoutSourceIdDuplication(NvDsFrameMetaList* frameMetaList);

    /**
     * @return pointer to array of first num_surfaces_per_frame
     *         surfaces that are part of singleSourceId
     *         nullptr if no associated surfaces in inputBuffer
     */
    std::vector<NvBufSurfaceParams> GetSingleSourceSurfaces(NvBufSurface* inputBuffer, NvDsBatchMeta* inputBufferMeta);

    bool CopyNvBufSurface(NvBufSurface* src, NvBufSurface* dest, int32_t interpolation_method);

    TileConfig GetTileConfig(uint32_t sourceId);

    CustomTile* GetTileBySourceId(CustomTile* tiles, uint32_t tileConfigSize, uint32_t sourceId);

    /**
     * @brief  Check if any of the required scaling operation
     *         exceed the max scale factor supported by VIC
     *         NOTE: VIC (NvDdkVic API) is used for scaling on
     *         nvidia Jetson platforms
     * @param  srcRects [IN] Pointer to the array of source rects
     * @param  dstRects [IN] Pointer to the array of destination rects
     * @return false when src to dest scaling operation will exceed
     *         the MaxScaleFactorVIC supported by VIC;
     *         true otherwise
     */
    bool CheckVICMaxScaleFactorCompatibility(NvBufSurfTransformRect const * srcRects,
         NvBufSurfTransformRect const * dstRects, uint32_t const numFrames);

    TilerConfig tilerConfig;

    std::vector<TileConfig> tiles;

    /** The local frame_meta_list cache for outputBuffer NvDsBatchMeta
     * This cache will always have the most recent frame meta-data for
     * the tiler generated canvas (all individual sources' latest meta)
     */
    NvDsBatchMeta* cacheCanvasMeta;

    bool singleSourceModeON;

    uint32_t singleSourceId;

    uint32_t singleSourceNumSurfacesPerFrame;

    bool squareGridMode;

    std::map<uint32_t, uint32_t> *tilerMap;

    GMutex *tilerMapLock;

    cudaStream_t cudaStream;
    /** Vector of sync objects for async transformation of the batch */
    std::vector<NvBufSurfTransformSyncObj_t> sync_objects;
    NvBufSurfTransformCompositeParams compositeParams;

    std::mutex mtxSingleSource;

    /** Max Scale factor is the NvDDK VIC software enforced limit
     * This is currently hardcoded to 16.0
     * More info: graphics/2d/vic/vic4_common.h
     */
    float const MaxScaleFactorVIC;
    float const MinScaleFactorVIC;

#ifdef ENABLE_NVTILER_UNIT_TESTS
    /** Enabled & Compiled in - only for gst-nvtiler/tests
     * All Unit-Tests for NvTiler private functions: */
    friend class TestNvTilerGetSrcCompositionRects_InvalidParams1_Test;
    friend class TestNvTilerGetSrcCompositionRects_InvalidParams2_Test;
    friend class TestNvTilerGetSrcCompositionRects_InputTestBufferBatch1Filled1_Test;
    friend class TestNvTilerGetSrcCompositionRects_InputTestBufferBatch100Filled100_Test;
    friend class TestNvTilerGetSrcCompositionRects_InputTestBufferBatch100Filled50_Test;
    friend class TestNvTilerGetSrcCompositionRects_InvalidInputTestBufferBatch50Filled100_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch2Filled2_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch2Filled2_columns2_rows1_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch2Filled2_columns2_rows1_SameSourceId0_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch2Filled2_columns2_rows1_SameSourceId1_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch2Filled2_columns1_rows2_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch2Filled2_columns1_rows2_SameSourceId1_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch100Filled100_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch100Filled50_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch100Filled50_Alternative_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch100Filled50_Last50Sources_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch100Filled100_SurfacesPerAllFrame2_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch16Filled16_SurfacesPerAllFrame2_2_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch16Filled16_SurfacesPerFrameDifferent_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch16Filled16_SurfacesPerFrameDifferent_Tiler4x4_Test;
    friend class TestNvTilerGetDestCompositionRects_InvalidParams1_Test;
    friend class TestNvTilerGetDestCompositionRects_InvalidParams2_Batch16Filled17_Test;
    friend class TestNvTilerGetDestCompositionRects_InvalidParams3_Batch16Filled0_Test;
    friend class TestNvTilerGetDestCompositionRects_InvalidParams4_NullSurfaceList_Test;
    friend class TestNvTilerGetDestCompositionRects_InvalidParams5_FrameMetaSurfacesPerFrame0_Test;
    friend class TestNvTilerGetDestCompositionRects_InvalidParams6_NullFrameMeta_Test;
    friend class TestNvTilerScaleAndPlaceAllMeta_InputTestBufferBatch16Filled8_SurfacesPerFrameDifferent_ScaleTwice_8Plus8_Test;
    friend class TestNvTilerScaleAndPlaceAllMeta_InputTestBufferBatch100FilledX_SurfacesPerFrameDifferent_ScaleTwice_50Plus50_2_Test;
    friend class TestNvTilerScaleAndPlaceAllMeta_InputTestBufferBatch100FilledX_SurfacesPerFrameDifferent_ScaleTwice_50Plus40_3_Test;
    friend class TestNvTilerScaleAndPlaceAllMeta_InputTestBufferBatch16Filled8_SingleSourceModeON_Test;
    friend class TestNvTilerAdjustOutputMeta_InputTestBufferBatch100FilledX_SurfacesPerFrameDifferent_SyncTwice_50Plus40_Test;
    friend class TestNvTilerAdjustOutputMeta_InputTestBufferBatch100FilledX_SurfacesPerFrameDifferent_SyncTwice_50Plus40_MultipleFramesWithSameSourceIdInBatch_Test;
    friend class TestNvTilerAdjustOutputMeta_InputTestBufferBatch100FilledX_SurfacesPerFrameDifferent_SyncXTimes_2_Test;
    friend class TestNvTilerAdjustOutputMeta_InputTestBufferBatch100FilledX_SurfacesPerFrameDifferent_SyncXTimes_OverlappedBatches_Test;
    friend class TestNvTilerSyncBufferMeta_InputTestBufferBatch16Filled16_SingleSourceModeON_8S0_8S1_Test;
    friend class TestNvTilerSyncBufferMeta_InputTestBufferBatch16Filled16_SingleSourceModeOFF_8S0_8S1_Test;
    friend class TestNvTilerSyncBufferMeta_InputTestBufferBatch16Filled16_SingleSourceModeOFF_8S0_8S1_num_surfaces_per_frame_4S0_8S1_Test;
    friend class TestNvTilerSyncBufferMeta_InputTestBufferBatch16Filled16_SingleSourceModeON_8S0_8S1_num_surfaces_per_frame_4S0_2S1_Test;
    friend class TestNvTilerSyncBufferMeta_InputTestBufferBatch16Filled16_SingleSourceModeON_ForS0_8S0_8S1_num_surfaces_per_frame_4S0_2S1_Test;
    friend class TestNvTilerGetDestCompositionRects_InputTestBufferBatch16Filled16_columns4_rows4_Test;
    friend class TestNvTilerInit_CustomTileConfig_2x2_Test;
    friend class TestNvTilerInit_CustomTileConfig_2x2_TilesIncomplete_Test;
    friend class TestNvTilerInit_CustomTileConfig_2x2_WithBorders_Test;
    friend class TestNvTilerInit_CustomTileConfig_2x2_WithBorders_OneSource2SurfacesPerFrame_Test;
    friend class TestNvTilerGetSingleSourceSurfaces_InputTestBufferBatch16Filled8_SingleSourceModeON_Test;
    friend class TestNvTilerGetSingleSourceSurfaces_InputTestBufferBatch16Filled8_SingleSourceModeON_2_Test;
    friend class TestNvTilerGetSingleSourceSrcAndDestRect_InputTestBufferBatch16Filled8_SingleSourceModeON_NumSurfaces1_Test;
    friend class TestNvTilerGetSingleSourceSrcAndDestRect_InputTestBufferBatch16Filled8_SingleSourceModeON_NumSurfaces2_Test;
    friend class TestNvTilerGetSingleSourceSrcAndDestRect_InputTestBufferBatch16Filled8_SingleSourceModeON_NumSurfaces7_Test;
    friend class TestNvTilerGetSingleSourceSrcAndDestRect_InputTestBufferBatch16Filled8_SingleSourceModeON_NumSurfaces8_Test;
    friend class TestNvTilerGetSingleSourceSrcAndDestRect_InputTestBufferBatch16Filled16_SingleSourceModeON_8S0_8S1_Test;
    friend class TestNvTilerGetSingleSourceSrcAndDestRect_InputTestBufferBatch16Filled16_SingleSourceModeON_0S0_16S1_Test;
    friend class TestNvTilerCheckVICMaxScaleFactorCompatibility_InputAllGood_ScaleUp_AndDown_Test;
    friend class TestNvTilerCheckVICMaxScaleFactorCompatibility_Input_ScaleUpGood_ScaleDownWidthBad_Test;
    friend class TestNvTilerCheckVICMaxScaleFactorCompatibility_Input_ScaleUpGood_ScaleDownHeightBad_Test;
    friend class TestNvTilerCheckVICMaxScaleFactorCompatibility_Input_ScaleDownGood_ScaleUpWidthBad_Test;
    friend class TestNvTilerCheckVICMaxScaleFactorCompatibility_Input_ScaleDownGood_ScaleUpHeightBad_Test;
#endif /**< ENABLE_NVTILER_UNIT_TESTS */
#ifdef ENABLE_GST_NVTILER_UNIT_TESTS
    friend bool TestVerifyNvTilerCurrentConfig(NvTiler* tilerIface, uint32_t expectedTileWidth, uint32_t expectedTileHeight, uint32_t expectedColumns, uint32_t expectedRows);
#endif /**< ENABLE_GST_NVTILER_UNIT_TESTS */

};

#endif /**< _NVTILER_H_ */
