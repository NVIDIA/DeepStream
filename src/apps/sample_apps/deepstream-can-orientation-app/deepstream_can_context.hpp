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

#ifndef DS_APP_DEEPSTREAM_CAN_CONTEXT_H
#define DS_APP_DEEPSTREAM_CAN_CONTEXT_H

#include "gstnvdsmeta.h"

#include <sstream>
#include <unistd.h>

#include <ds3d/common/hpp/profiling.hpp>
#include <ds3d/common/func_utils.h>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <ds3d/common/hpp/profiling.hpp>
// inlucde nvds3d Gst header files
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>
#include <gst/gst.h>

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

const uint32_t kNvDsCanResultMeta = (NVDS_GST_CUSTOM_META + 100);

using namespace ds3d;

enum class NvDsCanContextConfigType : int {
    kConfigPath = 0,
    kConfigContent = 1,
};

struct NvDsCanContextFrame {
    void* buf;
    size_t bufLen;
    uint32_t fourcc;  // GREY, RGBA
    uint32_t width, height;
};

struct NvDsCanContextResultMeta {
    float rotation;  // normalized result
};

enum class NvDsCanSrcType : int {
    kNone = 0,
    kUri,
    kFrame,
    kBaslerCamera,
};

struct NvDsCanFrameInfo {
    uint32_t fourcc = 0;
    uint32_t height = 0;
    uint32_t width = 0;
};

struct NvDsCanCameraInfo {
    std::string devSN;  // Basler Camera Device Serial Number
    std::string pfsPath;  // PFS location
    uint32_t height = 0;
    uint32_t width = 0;
    std::string format = "GRAY8";
};

struct NvDsCanContextConfig {
    NvDsCanSrcType srcType = NvDsCanSrcType::kNone;

    std::string srcUri;
    NvDsCanCameraInfo cameraInfo;
    NvDsCanFrameInfo srcFrameInfo;
    std::string srcFrameRate;

    uint32_t muxWidth = 1280, muxHeight = 720;
    int32_t muxBatchTimeout = 40000;

    std::string templateCustomLibName;
    std::string templateCustomLibProps;

    std::string inferConfig;
    uint32_t debugLevel = 0;
    bool printRes = true;
    bool enableEglSink = false;
    bool syncDisplay = true;
};

#define NVDS_MAKE_FOURCC(c0, c1, c2, c3) \
    ((uint32_t)(c0) | ((uint32_t)(c1)) << 8 | ((uint32_t)(c2)) << 16 | ((uint32_t)(c3)) << 24)


const uint32_t NvDsCanFormatRGBA = NVDS_MAKE_FOURCC('R', 'G', 'B', 'A');
const uint32_t NvDsCanFormatGREY = NVDS_MAKE_FOURCC('G', 'R', 'E', 'Y');

typedef void (*NvDsCanContextTaskQuitCb)(void* usrData);

class DsCanContext;

DS3D_EXTERN_C_BEGIN

DS3D_EXPORT_API ErrCode NvDs_CanContextParseConfig(
    NvDsCanContextConfig* config, const char* str, uint32_t len, NvDsCanContextConfigType type);
DS3D_EXPORT_API ErrCode
NvDs_CanContextCreate(DsCanContext** ctx, const NvDsCanContextConfig* config);
DS3D_EXPORT_API ErrCode
NvDs_CanContextStart(DsCanContext* ctx, NvDsCanContextTaskQuitCb cb, void* usrData);
DS3D_EXPORT_API ErrCode NvDs_CanContextProcessFrame(
    DsCanContext* ctx, const NvDsCanContextFrame* frame, NvDsCanContextResultMeta* res);
DS3D_EXPORT_API bool NvDs_CanContextIsRunning(DsCanContext* ctx);
DS3D_EXPORT_API ErrCode NvDs_CanContextStop(DsCanContext* ctx);
DS3D_EXPORT_API ErrCode NvDs_CanContextDestroy(DsCanContext* ctx);

DS3D_EXTERN_C_END

#endif  // DS_APP_DEEPSTREAM_CAN_CONTEXT_H
