/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef _DS3D_COMMON_COMMON__H
#define _DS3D_COMMON_COMMON__H

#include <ds3d/common/defines.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace ds3d {

enum class ErrCode : int {
    kGood = 0,
    kByPass = 1,
    KEndOfStream = 2,

    kLoadLib = -1,
    kMem = -2,
    kParam = -3,
    kNotFound = -4,
    kTimeOut = -5,
    kTypeId = -6,
    kNvDsMeta = -7,
    kUnsupported = -8,
    kConfig = -9,
    kRealSense = -10,
    kNullPtr = -11,
    kOutOfRange = -12,
    kGst = -13,
    kState = -14,
    kGL = -15,
    kLockWakeup = -16,
    kCuda = -17,
    kIncompatible = -18,
    kUnknown = -255,
};

using TIdType = uint64_t;

// structure TimeStamp
static constexpr const char* kTimeStamp = DS3D_KEY_NAME("Timestamp");
// get from Frame2DGuard
static constexpr const char* kColorFrame = DS3D_KEY_NAME("ColorFrame");
// get from Frame2DGuard
static constexpr const char* kDepthFrame = DS3D_KEY_NAME("DepthFrame");
// nvdspreprocess output tensor
static constexpr const char* kVideoPreProcTensor = DS3D_KEY_NAME("VideoPreprocessTensor");
// structure DepthScale
static constexpr const char* kDepthScaleUnit = DS3D_KEY_NAME("DepthScaleUnit");
// structure IntrinsicsParam
static constexpr const char* kDepthIntrinsics = DS3D_KEY_NAME("DepthIntrinsics");
// structure IntrinsicsParam
static constexpr const char* kColorIntrinsics = DS3D_KEY_NAME("ColorIntrinsics");
// structure ExtrinsicsParam
static constexpr const char* kDepth2ColorExtrinsics = DS3D_KEY_NAME("Depth2ColorExtrinsics");
// structure bool
static constexpr const char* kColorDepthAligned = DS3D_KEY_NAME("ColorDepthAligned");

// get from FrameGuard, N x 3
static constexpr const char* kTextureVertexKey = DS3D_KEY_NAME("TextureVertexKey");
// get from FrameGuard, N x 2
static constexpr const char* kTextureCoordinateKey = DS3D_KEY_NAME("TextureCoordKey");

// structure bool
static constexpr const char* kEOS = DS3D_KEY_NAME("EndOfStream");
// get from FrameGuard
static constexpr const char* kPointXYZ = DS3D_KEY_NAME("PointXYZ");
// get from FrameGuard
static constexpr const char* kPointCoordUV = DS3D_KEY_NAME("PointColorCoord");
// get from FrameGuard
static constexpr const char* kLidarXYZI = DS3D_KEY_NAME("LidarXYZI");
// lidar preprocess output tenor
static constexpr const char* kLidarFeatureTensor = DS3D_KEY_NAME("LidarFeatureTensor");
static constexpr const char* kLidarCoordTensor = DS3D_KEY_NAME("LidarCoordTensor");
static constexpr const char* kLidarPointNumTensor = DS3D_KEY_NAME("LidarPointNumTensor");
// get from FrameGuard
// LidarInferenceParas is deprecated, use kDs3dInferenceParas instead
static constexpr const char* kLidarInferenceParas = DS3D_KEY_NAME("LidarInferenceParas");
// get from FrameGuard
static constexpr const char* kDs3dInferenceParas = DS3D_KEY_NAME("DS3DInferneceParasKey");
//get from FrameGuard
static constexpr const char* kLidarRefDataMap = DS3D_KEY_NAME("LidarRefDataMap");
/// get array of Lidar3DBbox from FrameGuard [N, sizeof(Lidar3DBbox)]
static constexpr const char* kLidar3DBboxRawData = DS3D_KEY_NAME("Lidar3DBboxRawData");

static constexpr const char* kLidarAlignedXYZI = DS3D_KEY_NAME("LidarAlignedXYZIKey");
static constexpr const char* kCamIntrinsicParm = DS3D_KEY_NAME("CameraIntrinsicParm");
static constexpr const char* kCamIntrinsicMat = DS3D_KEY_NAME("CameraIntrinsicMatrix");
static constexpr const char* kLidarToCamExtrinsicMat = DS3D_KEY_NAME("LidarToCameraExtrinsicMatrix");

//VideoBridge2dInput for video/x-raw(memory:NVMM)
static constexpr const char* kVideoBridge2dData = DS3D_KEY_NAME("VideoBridge2dInput");
//GstBuffer for video/x-raw(memory:NVMM)
static constexpr const char* kGstBuffer = DS3D_KEY_NAME("GstBuffer");
//NvBufSurface for video/x-raw(memory:NVMM)
static constexpr const char* kNvBufSurface = DS3D_KEY_NAME("NvBufSurface");
//NvDsBatchMeta for video/x-raw(memory:NVMM) buffer gst meta == batch meta
static constexpr const char* kNvDsBatchMeta = DS3D_KEY_NAME("NvDsBatchMeta");
//first frame, set by source
static constexpr const char* kFirstSourceFrame = DS3D_KEY_NAME("FirstSourceFrame");
//source id
static constexpr const char* kSourceId = DS3D_KEY_NAME("SourceId");

/// get array of Object2DBbox from FrameGuard [N, sizeof(Object2DBbox)]
static constexpr const char* kObject2DBboxKey = DS3D_KEY_NAME("Object2DBboxKey");

/// get array of FusedDetection from FrameGuard [N, sizeof(FusedDetection)]
static constexpr const char* kFusedDetectionKey = DS3D_KEY_NAME("FusedDetectionKey");

// default caps for input and ouptut
static constexpr const char* kDefaultDs3dCaps = "ds3d/datamap";

class Exception : public std::exception {
    std::string _msg;
    ErrCode _code = ErrCode::kGood;

public:
    Exception(ErrCode code, const std::string& s = "") : _msg(s), _code(code) {}
    ErrCode code() const { return _code; }
    const char* what() const noexcept override { return _msg.c_str(); }
};


}  // namespace ds3d

#endif  // _DS3D_COMMON_COMMON__H
