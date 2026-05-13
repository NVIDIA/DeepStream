/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
