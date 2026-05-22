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


#ifndef _NVDS3D_GST_NVDS3D_META__H
#define _NVDS3D_GST_NVDS3D_META__H

#include <ds3d/common/abi_frame.h>
#include <ds3d/common/abi_obj.h>
#include <ds3d/common/idatatype.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"

#define NVDS3D_MAGIC_ID(a, b, c, d) \
    ((uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(c) << 8) | (uint32_t(d)))

#define NVDS3D_BUF_MAGIC_ID NVDS3D_MAGIC_ID('D', 'S', '3', 'D')

#define NVDS_3D_DATAMAP_META_TYPE (NvDsMetaType)((uint32_t)NVDS_GST_CUSTOM_META + 128)

struct NvDs3DBuffer {
    uint32_t magicID;  // must be 'DS3D'
    ds3d::abiRefDataMap* datamap;
};

DS3D_EXTERN_C_BEGIN

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_CreateGstBuf(GstBuffer*& outBuf, ds3d::abiRefDataMap* datamap, bool takeOwner);

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_CreateEmptyBatchMeta(
    GstBuffer* buf, NvDsBatchMeta*& batchMeta, uint32_t maxBatch);

DS3D_EXPORT_API bool NvDs3D_IsDs3DBuf(GstBuffer* buf);

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_BatchMeta_SetDataMapAsMeta(
    NvDsBatchMeta* batchMeta, const ds3d::abiRefDataMap* datamap);

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_Find1stDataMap(GstBuffer* buf, const ds3d::abiRefDataMap*& datamap);

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_UpdateDataMap(GstBuffer* buf, const ds3d::abiRefDataMap* datamap);

DS3D_EXTERN_C_END

#endif  // _NVDS3D_COMMON_NVDS3D_META__H