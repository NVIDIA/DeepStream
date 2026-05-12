/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "infer_proto_utils.h"

#include <google/protobuf/text_format.h>

#include "infer_utils.h"

#define INFER_DEFAULT_PREPPROCESS_SCALE 1.0f

namespace nvdsinferserver {

InferDataType dataTypeFromDsProto(ic::TensorDataType dt) {
    const static std::unordered_map<ic::TensorDataType, InferDataType> sFrProto{
        {ic::TENSOR_DT_FP32, InferDataType::kFp32},
        {ic::TENSOR_DT_FP16, InferDataType::kFp16},
        {ic::TENSOR_DT_INT8, InferDataType::kInt8},
        {ic::TENSOR_DT_UINT8, InferDataType::kUint8},
        {ic::TENSOR_DT_INT16, InferDataType::kInt16},
        {ic::TENSOR_DT_UINT16, InferDataType::kUint16},
        {ic::TENSOR_DT_INT32, InferDataType::kInt32},
        {ic::TENSOR_DT_UINT32, InferDataType::kUint32},
        {ic::TENSOR_DT_FP64, InferDataType::kFp64},
        {ic::TENSOR_DT_INT64, InferDataType::kInt64},
        {ic::TENSOR_DT_UINT64, InferDataType::kUint64},
        {ic::TENSOR_DT_STRING, InferDataType::kString},
        {ic::TENSOR_DT_NONE, InferDataType::kNone},
    };
    auto const i = sFrProto.find(dt);
    if (i == sFrProto.end()) {
        InferError(
            "unsupported proto TensorDataType: %s",
            safeStr(ic::TensorDataType_Name(dt)));
        return InferDataType::kNone;
    }
    return i->second;
}

InferTensorOrder tensorOrderFromDsProto(ic::TensorOrder o) {
    switch (o) {
    case ic::TENSOR_ORDER_NONE:
        return InferTensorOrder::kNone;
    case ic::TENSOR_ORDER_LINEAR:
        return InferTensorOrder::kLinear;
    case ic::TENSOR_ORDER_NHWC:
        return InferTensorOrder::kNHWC;
    default:
        InferError(
            "unsupported proto TensorOrder: %s",
            safeStr(ic::TensorOrder_Name(o)));
        return InferTensorOrder::kNone;
    }
}

InferMediaFormat mediaFormatFromDsProto(ic::MediaFormat f) {
    switch (f) {
    case ic::IMAGE_FORMAT_RGB:
        return InferMediaFormat::kRGB;
    case ic::IMAGE_FORMAT_BGR:
        return InferMediaFormat::kBGR;
    case ic::IMAGE_FORMAT_GRAY:
        return InferMediaFormat::kGRAY;
    case ic::MEDIA_FORMAT_NONE:
        return InferMediaFormat::kUnknown;
    default:
        InferError(
            "unsupported proto MediaFormat: %s",
            safeStr(ic::MediaFormat_Name(f)));
        return InferMediaFormat::kUnknown;
    }
}

InferMemType
memTypeFromDsProto(ic::MemoryType t)
{
    static const std::unordered_map<ic::MemoryType, InferMemType> sTypes{
        {ic::MEMORY_TYPE_DEFAULT, InferMemType::kNone},
        {ic::MEMORY_TYPE_CPU, InferMemType::kCpuCuda},
        {ic::MEMORY_TYPE_GPU, InferMemType::kGpuCuda},
    };
    auto const i = sTypes.find(t);
    if (i == sTypes.end()) {
        InferError(
            "unsupported proto MemoryType: %s, fallback to MEMORY_TYPE_DEFAULT",
            safeStr(ic::MemoryType_Name(t)));
        return InferMemType::kNone;
    }
    return i->second;
}

NvBufSurfTransform_Compute computeHWFromDsProto(ic::FrameScalingHW h) {
    switch (h) {
    case ic::FRAME_SCALING_HW_DEFAULT:
        return NvBufSurfTransformCompute_Default;
    case ic::FRAME_SCALING_HW_GPU:
        return NvBufSurfTransformCompute_GPU;
    case ic::FRAME_SCALING_HW_VIC:
        return NvBufSurfTransformCompute_VIC;
    default:
        InferError(
            "unsupported proto FrameScalingHW: %s",
            safeStr(ic::FrameScalingHW_Name(h)));
        return NvBufSurfTransformCompute_Default;
    }
}

NvBufSurfTransform_Inter scalingFilterFromDsProto(uint32_t filter) {
    if (filter > NvBufSurfTransformInter_Default) {
        InferError("unsupported proto frame_scaling_filter: %u", filter);
        return NvBufSurfTransformInter_Default;
    }
    return (NvBufSurfTransform_Inter)filter;
}

#define ENSURE_REAL_PATH(config, attr, configPath) \
    do { \
      if (!config.attr().empty() && !isAbsolutePath(config.attr())) { \
            const std::string path = joinPath(dirName(configPath), config.attr()); \
            std::string absPath; \
            if (!realPath(path, absPath)) { \
                InferError("config file:%s cannot get real path for " #attr, \
                    safeStr(configPath)); \
                return false; \
            } \
            config.set_##attr(absPath); \
        } \
    } while(0)

static bool validatePreprocess(
    ic::PreProcessParams& preprocess, const std::string& configPath) {
    assert(
        ic::MediaFormat_IsValid(static_cast<int>(preprocess.network_format())));
    if (preprocess.network_format() == ic::MEDIA_FORMAT_NONE) {
        preprocess.set_network_format(ic::IMAGE_FORMAT_RGB);
        InferWarning(
            "auto-update preprocess.network_format to IMAGE_FORMAT_RGB");
    }

    assert(
        ic::TensorOrder_IsValid(static_cast<int>(preprocess.tensor_order())));
    if (preprocess.preprocess_method_case() ==
        ic::PreProcessParams::PREPROCESS_METHOD_NOT_SET) {
        preprocess.mutable_normalize()->set_scale_factor(
            INFER_DEFAULT_PREPPROCESS_SCALE);
        InferWarning(
            "auto-update preprocess.normalize.scale_factor to %.4f",
            INFER_DEFAULT_PREPPROCESS_SCALE);
    }
    if (preprocess.has_normalize()) {
        auto& normalize = *preprocess.mutable_normalize();
        if (fEqual(normalize.scale_factor(), 0.0f)) {
            normalize.set_scale_factor(INFER_DEFAULT_PREPPROCESS_SCALE);
            InferWarning(
                "auto-update preprocess.normalize.scale_factor from 0"
                " to %.4f",
                INFER_DEFAULT_PREPPROCESS_SCALE);
        }
        ENSURE_REAL_PATH(normalize, mean_file, configPath);
    }
    return true;
}

static bool validatePostprocess(ic::PostProcessParams &postprocess, const std::string& configPath) {
#ifndef WITH_OPENCV
    if (postprocess.detection().has_group_rectangle()) {
        ic::DetectionParams::GroupRectangle gr = postprocess.mutable_detection()->group_rectangle();
        ic::DetectionParams::Nms * nms = postprocess.mutable_detection()->mutable_nms();
        nms->set_topk(20);
        nms->set_confidence_threshold(gr.confidence_threshold());
        nms->set_iou_threshold(0.5);
        assert(postprocess.detection().has_nms());
        InferWarning(
            "Warning, OpenCV has been deprecated. Using detection.nms for clustering instead of detection.group_rectangle. falling back to nms {%s}", safeStr(nms->DebugString()));
    }
#endif
    ENSURE_REAL_PATH(postprocess, labelfile_path, configPath);
    if (postprocess.has_trtis_classification()) {
        ic::TritonClassifyParams clsParam = postprocess.trtis_classification();
        postprocess.mutable_triton_classification()->CopyFrom(clsParam);
        InferWarning(
            "infer_config.postprocess.trtis_classification is DEPRECATED. "
            "Update it to infer_config.postprocess.triton_classification");
        assert(postprocess.has_triton_classification());
    }
    return true;
}

static bool validateBackend(ic::BackendParams &backend, const std::string& configPath) {
    if (backend.has_trt_is()) {
        ic::TritonParams tritonParam = backend.trt_is();
        backend.mutable_triton()->CopyFrom(tritonParam);
        InferWarning("backend.trt_is is deprecated. updated it to backend.triton");
        assert(backend.has_triton());
    }
    if (hasTriton(backend) && getTritonParam(backend).has_model_repo()) {
        auto& modelRepo = *mutableTriton(backend)->mutable_model_repo();
        for (int i = 0; i < modelRepo.root_size(); ++i) {
            std::string r = modelRepo.root(i);
            // network folder "gs://"
            if (!r.empty() && r.find("://") == std::string::npos && !isAbsolutePath(r)) {
                const std::string path = joinPath(dirName(configPath), r);
                std::string absPath;
                if (!realPath(path, absPath)) {
                    InferError(
                        "config file:%s cannot get real path for "
                        "model_repo.root", safeStr(configPath));
                    return false;
                }
                modelRepo.set_root(i, absPath);
            }
        }
    }

    return true;
}

static bool validateCustomLib(ic::CustomLib &c, const std::string& configPath) {
    if (c.path().empty() || c.path().find('/') == std::string::npos) {
        // use system lib path
        return true;
    }
    ENSURE_REAL_PATH(c, path, configPath);
    return true;
}

static bool
validateLSTM(ic::LstmParams& c, const std::string& configPath)
{
    if (!c.loops_size()) {
        InferError(
            "lstm does not have loops, check config: %s", safeStr(configPath));
        return false;
    }
    for (auto& perLoop : *c.mutable_loops()) {
        if (perLoop.input().empty() || perLoop.output().empty()) {
            InferError(
                "lstm each loop must have input/output setting, check config "
                "%s",
                safeStr(configPath));
            return false;
        }
        if (perLoop.init_state_case() ==
            ic::LstmParams::LstmLoop::INIT_STATE_NOT_SET) {
            perLoop.mutable_init_const()->set_value(0);
        }
    }
    return true;
}

bool validateProtoConfig(ic::InferenceConfig& c, const std::string& configPath) {
    if (!c.gpu_ids_size()) {
        c.add_gpu_ids(0);
    }
    if (c.gpu_ids_size() > 1) {
        InferError(
            "update gpu_ids to keep single gpu in config:%s",
            safeStr(configPath));
        return false;
    }
    if (c.max_batch_size() <= 0) {
        c.set_max_batch_size(1);
        InferWarning(
            "update max_bath_size to 1 in config:%s", safeStr(configPath));
    }

    if (c.has_backend() &&
        !validateBackend(*c.mutable_backend(), configPath)) {
        InferError(
            "failed to validate backend config file:%s.", safeStr(configPath));
        return false;
    }

    // preprocess is not required once input_tensor_from_meta is set.
    if (!c.has_input_tensor_from_meta()) {
        if (!c.has_preprocess()) {
            InferWarning(
                "preprocess is not configured in in file: %s, will use default "
                "settings.",
                safeStr(configPath));
        }

        if (!validatePreprocess(*c.mutable_preprocess(), configPath)) {
            InferError("failed to validate preprocess config file:%s.", safeStr(configPath));
            return false;
        }
    }

    if (c.has_postprocess() &&
        !validatePostprocess(*c.mutable_postprocess(), configPath)) {
        InferError(
            "failed to validate postprocess config file:%s.", safeStr(configPath));
        return false;
    }

    // validate custom lib
    if (c.has_custom_lib() &&
        !validateCustomLib(*c.mutable_custom_lib(), configPath)) {
        InferError(
            "failed to validate custom-lib in config file:%s.", safeStr(configPath));
        return false;
    }

    // validate custom lib
    if (c.has_lstm()) {
        if (!validateLSTM(*c.mutable_lstm(), configPath)) {
            InferError(
                "failed to validate lstm in config file:%s.",
                safeStr(configPath));
            return false;
        }
        if (c.max_batch_size() > 1) {
            InferError(
                "LSTM support batch-size 1 only, check config file:%s.",
                safeStr(configPath));
            return false;
        }
    }
    return true;
}

}  // namespace nvdsinferserver

using namespace nvdsinferserver;

bool
validateInferConfigStr(const std::string& configStr, const std::string& path, std::string& updated)
{
    ic::InferenceConfig config;
    if (!google::protobuf::TextFormat::ParseFromString(configStr, &config)) {
        InferError("error: failed to parse internal lowlevel infer_config: %s", safeStr(path));
        return false;
    }
    if (!validateProtoConfig(config, path)) {
        InferError("error: failed to validate lowlevel infer_config: %s", safeStr(path));
        return false;
    }
    updated = config.DebugString();
    return true;
}