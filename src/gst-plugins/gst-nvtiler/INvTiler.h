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
/**
 * @file  TileImlementation.h
 * @brief The main Tiler API Interface file
 */

/**
 * @dev-note:
 * Tiler Composition Steps:
 * 1) Extract GstNvStreamMeta from the input muxed buffer
 * 2) loop for GstNvStreamMeta->num_filled (number of filled batch buffers)
 *    a) Each buffer according to its stream_id is assigned a
 *       blit position - (x,y) [width, height]
 * 3) NvBufSurfaceComposite() each stream buffer on the output buffer at
 *    the computed tile position
 * 4) Iterate through NvDsBatchMeta->NvDsFrameMeta[according_to_stream_id]
 *                    ->NvDsObjectMeta list
 *    and scale all bounding boxes to the computed tile rect
 */

#ifndef _INVTILER_H_
#define _INVTILER_H_

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <map>
#include <cstdint>
#include <cstdio>
#include "nvbufsurface.h"
#include "nvdsmeta.h"
#include "nvdstilerconfig.h"

class INvTiler
{
    public:

    /**
     * @brief  Initialize or re-initialize the Tiler
     * @param  config [IN][transfer-none]
     *         The new Tiler Configuration
     * @return true if (re-)Initialization succeeded;
     *         false otherwise
     */
    virtual bool Init(TilerConfig config) = 0;

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
    virtual bool Composite(NvBufSurface*    inputBuffer,
                           NvDsBatchMeta*   inputBufferMeta,
                           NvBufSurface*    outputBuffer,
                           NvBufSurface*    tilerScratchBuffer,
                           uint32_t         gpuId,
                           int32_t compute_hw,
                           int32_t interpolation_method) = 0;
    /**
     * @brief  Scale every bounding box meta to match tile resolution
     * @param  outputBatchMeta [IN/OUT] Output tiler composited buffer meta
     *         with stream-display-specific meta-data.
     *         The function shall modify outputBatchMeta with
     *         meta-data like bounding boxes scaled appropriately.
     */
    virtual bool AdjustOutputMeta(NvDsBatchMeta*   outputBatchMeta) = 0;

    virtual uint32_t GetTileWidth(uint32_t sourceId) = 0;

    virtual uint32_t GetTileHeight(uint32_t sourceId) = 0;

    virtual void SyncObjWait() = 0;

    /**
     * @brief Deinitialize the Tiler
     */
    virtual void Deinit() = 0;

    /**
     * @brief  Set ON/OFF the single source mode
     * @param  on [IN] If true, the single-source-mode is ON
     *         and OFF - if false
     * @param  sourceId [IN] The non-negative and valid sourceId
     *         which shall be the only source in canvas when
     *         single-source mode is enabled
     */
    virtual bool SetSingleSourceMode(bool on, int32_t sourceId=-1) = 0;

    /**
     * @brief  Delete a source for which a Composite() was done before
     *         This API ensure all cache'd metadata for this source is cleared
     * @param  sourceId [IN] The valid sourceId that needs to be
     *         deleted
     */
    virtual void DeleteSource(int32_t sourceId) = 0;

    /**
     * @brief Set Tiles based on square Grid.
     */
    virtual bool SetSquareGridMode(bool squareGrid) = 0;

    /**
     * @brief Pass ptr to the map of <source_id, tile_index> with its lock.
     */
    virtual bool SetTilerMap(std::map<uint32_t, uint32_t> *tiler_map, GMutex *tiler_map_lock) = 0;

    virtual ~INvTiler(){}
};

#endif /**< _INVTILER_H_ */
