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

/**
 * @file gstnvinferserver_impl.cpp
 *
 * @brief nvinferserver implementation source file.
 *
 * This file contains the definitions for the nvinferserver implementation
 * class: GstNvInferServerImpl.
 */

#include <cassert>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <gst/gst.h>
#include <google/protobuf/text_format.h>

#include "infer_surface_bufs.h"
#include "infer_post_datatypes.h"
#include "infer_options.h"
#include "infer_proto_utils.h"

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"

#include "gstnvinferserver.h"
#include "gstnvinferserver_impl.h"
#include "gstnvinferserver_meta_utils.h"

GST_DEBUG_CATEGORY_EXTERN(gst_nvinfer_server_debug);
#define GST_CAT_DEFAULT gst_nvinfer_server_debug

/** Tracked objects will be re-inferred only when their area in terms of pixels
 * increase by this ratio. */
#define REINFER_AREA_THRESHOLD 0.2

#define MAX_SECONDARY_REINFER_INTERVAL 90

/** Tracked objects in the infer history map will be removed if they have not
 * been accessed for at least this number of frames. The tracker would
 * definitely have dropped references to an unseen object by 150 frames. */
#define CLEANUP_ACCESS_CRITERIA 150

/** Object history map cleanup interval. 1800 frames is a minute with a 30fps
 * input */
#define MAP_CLEANUP_INTERVAL 1800

/** Warn about untracked objects in async mode every 5 minutes. */
#define UNTRACKED_OBJECT_WARN_INTERVAL (GST_SECOND * 60 * 5)

using namespace nvdsinferserver;

namespace gstnvinferserver {

/**
 * @brief Holds information about one frame in a batch for inferencing.
 */
struct InferFrame {
    /** Ratio by which the frame / object crop was scaled in the horizontal
     * direction. Required when scaling the detector boxes from the network
     * resolution to input resolution. Not required for classifiers. */
    double scaleRatioX = 0;
    /** Ratio by which the frame / object crop was scaled in the vertical
     * direction. Required when scaling the detector boxes from the network
     * resolution to input resolution. Not required for classifiers. */
    double scaleRatioY = 0;
    /** Offset incase Symmetric Padding was enabled. */
    uint32_t offsetLeft = 0;
    uint32_t offsetTop = 0;
    /** Left pixel coordinate of the ROI in the parent frame. */
    uint32_t roiLeft = 0;
    /** Top pixel coordinate of the ROI in the parent frame. */
    uint32_t roiTop = 0;
    /** Width of the ROI in the parent frame. */
    uint32_t roiWidth = 0;
    /** Height of the ROI in the parent frame. */
    uint32_t roiHeight = 0;
    /** Width of the frames in the input batch buffer. */
    uint32_t frameWidth = 0;
    /** Height of the frames in the input batch buffer. */
    uint32_t frameHeight = 0;
    /** NvDsObjectParams object metadata belonging to the parent object.
     *  Valid in case of inference on objects, null when processing on frames
     *  or input tensors. */
    NvDsObjectMeta* objMeta = nullptr;
    /** NvDsFrameMeta of the input frame. */
    NvDsFrameMeta* frameMeta = nullptr;
    /** NvDsRoiMeta incase of processing on input tensors. */
    NvDsRoiMeta* roiMeta = nullptr;
    /* Pointer to the NvBufSurfaceParams for the frame */
    NvBufSurfaceParams* surfParams = nullptr;
    /** Index of the frame in the batched input GstBuffer. Not required for
     * classifiers. */
    uint32_t batchIdx = 0;
    /** Frame number of the frame in the source. */
    uint32_t frameNum = 0;
    /** Pointer to object history if processing on objects. */
    std::weak_ptr<GstNvInferServerObjectHistory> history;
};

/**
 * @brief Holds information about a single inference request.
 */
struct RequestBuffer {
    /** The sequence number of the input batch. */
    uint64_t seqId = INT64_C(0);
    /** Pointer to the input GStreamer buffer. */
    void *gstBuf = nullptr;
    /** Batch metadata associated with this request. */
    NvDsBatchMeta* batchMeta = nullptr;
    /** Pointer to NvBufSurface structure of the input batch buffer. */
    NvBufSurface* inSurf = nullptr;
    /** Array of inference frames in the input batch for the inference request. */
    std::vector<InferFrame> frames;
    /** Status of the inference request, populated by the inference context run
     * and read in output thread. */
    NvDsInferStatus status = NVDSINFER_SUCCESS;
    /** Array of tensor outputs generated for the request */
    SharedIBatchArray outputs;
    /** Synchronization mechanism to indicate inference complete
     * in the inference thread */
    std::promise<void> promise;
    /** Synchronization mechanism to start inference output processing
     * in the output thread */
    std::future<void> future;
};

using SharedRequest = std::shared_ptr<RequestBuffer>;

GstNvInferServerSourceInfo* GstNvInferServerImpl::ObjTrackingData::findSource(
    int id) {
    auto iter = m_SourceInfo.find(id);
    if (iter == m_SourceInfo.end()) {
        return nullptr;
    }
    return &iter->second;
}

SharedObjHistory GstNvInferServerImpl::ObjTrackingData::findObjHistory(
    uint32_t sourceId, uint64_t objId) {
    GstNvInferServerSourceInfo *source = findSource(sourceId);
    if (!source)
        return nullptr;

    auto iter = source->object_history_map.find(objId);
    if (iter == source->object_history_map.end())
        return nullptr;
    return iter->second;
}

SharedObjHistory GstNvInferServerImpl::ObjTrackingData::newHistory(
    uint32_t sourceId, uint64_t objId) {
    GstNvInferServerSourceInfo *source = findSource(sourceId);
    if (!source) {
        assert(source);
        return nullptr;
    }
    SharedObjHistory& history = source->object_history_map[objId];
    history = std::make_shared<GstNvInferServerObjectHistory>();
    return history;
}

void GstNvInferServerImpl::ObjTrackingData::clearUpHistory(uint64_t seqId) {
    if (seqId - m_LastMapCleanupSeqId < MAP_CLEANUP_INTERVAL) {
        return;
    }

    /* Find the history map for each source whose frames are present in the
     * batch and trim the map. */
    for (auto& source_iter : m_SourceInfo) {
        GstNvInferServerSourceInfo& source_info = source_iter.second;
        if (source_info.last_seen_frame_num -
                source_info.last_cleanup_frame_num <
            MAP_CLEANUP_INTERVAL)
            continue;
        source_info.last_cleanup_frame_num = source_info.last_seen_frame_num;

        /* Remove entries for objects which have not been seen for
         * CLEANUP_ACCESS_CRITERIA */
        auto iterator = source_info.object_history_map.begin();
        while (iterator != source_info.object_history_map.end()) {
            auto history = iterator->second;
            if (!history->under_inference &&
                source_info.last_seen_frame_num -
                        history->last_accessed_frame_num >
                    CLEANUP_ACCESS_CRITERIA) {
                iterator = source_info.object_history_map.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    m_LastMapCleanupSeqId = seqId;
}

namespace {

bool readTxtFile(const std::string& path, std::string &context) {
    std::ifstream fileIn(path, std::ios::in | std::ios::binary);
    if (!fileIn) {
        InferError("Failed to read path :%s", safeStr(path));
        return false;
    }

    fileIn.seekg(0, std::ios::end);
    size_t fileSize = fileIn.tellg();
    context.resize(fileSize, 0);
    fileIn.seekg(0, std::ios::beg);
    fileIn.read(&context[0], fileSize);
    fileIn.close();
    return true;
}

bool
validateInputControl(ic::PluginControl::InputControl& input)
{
    if (input.process_mode() == ic::PluginControl::PROCESS_MODE_DEFAULT) {
        input.set_process_mode(ic::PluginControl::PROCESS_MODE_FULL_FRAME);
        InferWarning("Updated process_mode to FULL_FRAME in config");
    }
    return true;
}

bool
validateInferConfig(ic::InferenceConfig& c, const std::string& path)
{
    std::string updated;
    bool ret = validateInferConfigStr(c.DebugString(), path, updated);
    if (!ret) {
        InferError("Validation of infer_config in file: %s failed.",
                safeStr(path));
        return false;
    }
    c.Clear();
    if (!google::protobuf::TextFormat::ParseFromString(updated, &c)) {
        InferError("Error: Failed to parse back inference config: %s",
                safeStr(path));
        return false;
    }
    return true;
}

InferDataType
fromDSType(NvDsDataType t)
{
    switch (t) {
    case NvDsDataType_FP32:
        return InferDataType::kFp32;
    case NvDsDataType_UINT8:
        return InferDataType::kUint8;
    case NvDsDataType_INT8:
        return InferDataType::kInt8;
    case NvDsDataType_UINT32:
        return InferDataType::kUint32;
    case NvDsDataType_INT32:
        return InferDataType::kInt32;
    case NvDsDataType_FP16:
        return InferDataType::kFp16;
    default:
        InferError("Unknown NvDsDataType: %d to nvdsinferserver type.", (int32_t)t);
        return InferDataType::kNone;
    }
}

} // namespace

GstNvInferServerImpl::GstNvInferServerImpl(GstNvInferServer* gstPlugin)
    : m_GstPlugin(gstPlugin), m_NvtxDomain(nullptr, nvtxDomainDestroy) {}

GstNvInferServerImpl::~GstNvInferServerImpl() {}

uint32_t GstNvInferServerImpl::maxBatchSize() const {
    return config().infer_config().max_batch_size();
}

bool GstNvInferServerImpl::isClassify() const {
    return config().infer_config().has_postprocess() &&
           (config().infer_config().postprocess().has_classification() ||
            config().infer_config().postprocess().has_triton_classification());
}

bool GstNvInferServerImpl::isDetection() const {
    return config().infer_config().has_postprocess() &&
           config().infer_config().postprocess().has_detection();
}

bool GstNvInferServerImpl::isSegmentation() const {
    return config().infer_config().has_postprocess() &&
           config().infer_config().postprocess().has_segmentation();
}

bool GstNvInferServerImpl::isOtherNetowrk() const {
    return config().infer_config().has_postprocess() &&
           config().infer_config().postprocess().has_other();
}

bool GstNvInferServerImpl::isFullFrame() const {
    assert(config().has_input_control());
    return config().input_control().process_mode() ==
           ic::PluginControl::PROCESS_MODE_FULL_FRAME;
}

bool GstNvInferServerImpl::needOutputTensorMeta() const {
    return config().has_output_control() &&
           config().output_control().output_tensor_meta();
}

bool GstNvInferServerImpl::maintainAspectRatio() const {
    return config().has_infer_config() && config().infer_config().has_preprocess() &&
           config().infer_config().preprocess().maintain_aspect_ratio();
}

bool GstNvInferServerImpl::symmetricPadding() const {
    return config().has_infer_config() && config().infer_config().has_preprocess() &&
           config().infer_config().preprocess().symmetric_padding();
}

bool
GstNvInferServerImpl::hasCustomProcess() const
{
    return config().infer_config().has_extra() &&
           !config().infer_config().extra().custom_process_funcion().empty();
}

bool GstNvInferServerImpl::isAsyncMode() const {
    return config().input_control().async_mode();
}

bool GstNvInferServerImpl::canSupportGpu(int gpuId) const {
    auto const& gpus = config().infer_config().gpu_ids();
    return std::any_of(
        gpus.begin(), gpus.end(), [gpuId](int id) { return gpuId == id; });
}

bool GstNvInferServerImpl::validatePluginConfig(
    ic::PluginControl& config, const std::string& path,
    const GstNvInferServerProperties& update) {

    // validate infer_config
    if (!config.has_infer_config()) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file parsing failed"),
            ("Config file: %s doesn't have infer_config", safeStr(path)));
        return false;
    }

    //update data by properties
    uint32_t configBatch = config.infer_config().max_batch_size();
    if (update.maxBatchSize && update.maxBatchSize != configBatch) {
        config.mutable_infer_config()->set_max_batch_size(update.maxBatchSize);
        GST_ELEMENT_WARNING(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file batch-size reset to: %d", update.maxBatchSize),
            (nullptr));
    }
    if (update.uniqueId && update.uniqueId != config.infer_config().unique_id()) {
        config.mutable_infer_config()->set_unique_id(update.uniqueId);
        GST_ELEMENT_WARNING(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file unique-id reset to: %d", update.uniqueId),
            (nullptr));
    }
    // check input_control
    auto &inputControl = *config.mutable_input_control();
    if (update.processMode &&
        update.processMode != (uint32_t)inputControl.process_mode()) {
        if (!ic::PluginControl::ProcessMode_IsValid(update.processMode)) {
            GST_ELEMENT_ERROR(
                m_GstPlugin, LIBRARY, SETTINGS,
                ("Configuration file process_mode %d is invalid",
                 update.processMode),
                (nullptr));
            return false;
        }
        inputControl.set_process_mode(
            (ic::PluginControl::ProcessMode)update.processMode);
        GST_ELEMENT_WARNING(
            m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file process_mode reset to: %s",
             safeStr(ic::PluginControl::ProcessMode_Name(
                 inputControl.process_mode()))),
            (nullptr));
    }
    // update operate_on_gie_id
    if (update.inferOnGieId >= 0) {
        inputControl.set_operate_on_gie_id(update.inferOnGieId);
    }
    // update operate_on_gie_id
    if (update.inputTensorFromMeta && !config.infer_config().has_input_tensor_from_meta()) {
        config.mutable_infer_config()->mutable_input_tensor_from_meta()->set_is_first_dim_batch(
            true);
    }
    //update interval
    if (update.interval >= 0) {
        inputControl.set_interval((uint32_t)update.interval);
    }
    if (!update.operateOnClassIds.empty()) {
        inputControl.clear_operate_on_class_ids();
        for (int32_t id : update.operateOnClassIds) {
            inputControl.add_operate_on_class_ids(id);
        }
    }
    if (!config.infer_config().has_input_tensor_from_meta()) {
        if (!validateInputControl(inputControl)) {
            GST_ELEMENT_ERROR(
                m_GstPlugin, LIBRARY, SETTINGS,
                ("Configuration config input_control validation failed"),
                ("Config file path: %s", safeStr(path)));
            return false;
        }
    }

    // validate context config
    if (!validateInferConfig(*config.mutable_infer_config(), path)) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration config infer_config validatation failed"),
            ("Config file path: %s", safeStr(path)));
        return false;
    }

    // validate mixed settings
    bool isClass = config.infer_config().has_postprocess() &&
                   config.infer_config().postprocess().has_classification();

    if (inputControl.async_mode()) {
        if (!isClass || inputControl.process_mode() !=
                            ic::PluginControl::PROCESS_MODE_CLIP_OBJECTS) {
            GST_ELEMENT_WARNING(m_GstPlugin, LIBRARY, SETTINGS,
                ("NvInferServer asynchronous mode is applicable for secondary"
                 "classifiers only. Turning off asynchronous mode"),
                (nullptr));
            config.mutable_input_control()->set_async_mode(false);
        }
    }

    //update property back to m_GstProperties
    m_GstProperties.uniqueId = config.infer_config().unique_id();
    m_GstProperties.maxBatchSize = config.infer_config().max_batch_size();
    m_GstProperties.interval = (int32_t)config.input_control().interval();
    m_GstProperties.processMode = (int)inputControl.process_mode();
    m_GstProperties.inferOnGieId = inputControl.operate_on_gie_id();
    m_GstProperties.inputTensorFromMeta = config.infer_config().has_input_tensor_from_meta();
    if(m_GstProperties.classifierType.empty() && !config.output_control().classifier_type().empty()) {
        m_GstProperties.classifierType = config.output_control().classifier_type();
    }
    if (m_GstProperties.operateOnClassIds.empty()) {
        for (int32_t id : inputControl.operate_on_class_ids()) {
            m_GstProperties.operateOnClassIds.push_back(id);
        }
    }

    return true;
}

static void ProtoBufLogHandler(google::protobuf::LogLevel level, const char* filename, int line,
    const std::string& message)
{
    NvDsInferLogLevel dsLevel = NVDSINFER_LOG_ERROR;
    switch (level) {
    case google::protobuf::LOGLEVEL_INFO:
        dsLevel = NVDSINFER_LOG_INFO;
        break;
    case google::protobuf::LOGLEVEL_WARNING:
        dsLevel = NVDSINFER_LOG_WARNING;
        break;
    case google::protobuf::LOGLEVEL_ERROR:
        dsLevel = NVDSINFER_LOG_ERROR;
        break;
    case google::protobuf::LOGLEVEL_FATAL:
        dsLevel = NVDSINFER_LOG_ERROR;
        break;
    default:
        return;
    }
    std::string res = std::string("file: ") + filename +
        "line: " + std::to_string(line) + "msg: " + message;
    dsInferLogPrint__(dsLevel, res.c_str());
}

NvDsInferStatus GstNvInferServerImpl::start() {
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u starting", uniqueId());

    if (0) {
        static std::once_flag protoLoggingFlag;
        std::call_once(protoLoggingFlag,
            []() { google::protobuf::SetLogHandler(&ProtoBufLogHandler); });
    }
    {
        static std::once_flag nvtxInitFlag;
        std::call_once(nvtxInitFlag,
            []() { nvtxInitialize(nullptr); });
    }

    NvDsInferStatus status = NVDSINFER_SUCCESS;
    std::unique_lock<std::mutex> locker(m_ProcessMutex);

    std::string nvtx_str =
        "GstNvInferServer: UID=" + std::to_string(uniqueId());
    m_NvtxDomain.reset(nvtxDomainCreate(nvtx_str.c_str()));

    /* Providing a valid config file is mandatory. */
    if (m_GstProperties.configPath.empty()) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file not provided"), (nullptr));
        return NVDSINFER_CONFIG_FAILED;
    }

    std::string prototxt;
    if (!readTxtFile(m_GstProperties.configPath, prototxt)) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file read failed"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
        return NVDSINFER_CONFIG_FAILED;
    }
    if (!google::protobuf::TextFormat::ParseFromString(prototxt, &m_PluginConfig)) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration file parsing failed"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
        return NVDSINFER_CONFIG_FAILED;
    }

    if (!validatePluginConfig(m_PluginConfig, m_GstProperties.configPath,
        m_GstProperties)) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration config validatation failed"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
        return NVDSINFER_CONFIG_FAILED;
    }
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u config-file:%s validation passed",
        uniqueId(), safeStr(m_GstProperties.configPath));

    std::string contextStr = m_PluginConfig.infer_config().DebugString();
    m_IntervalCounter = 0;

    if (isOtherNetowrk() && !needOutputTensorMeta() && !hasCustomProcess()) {
        GST_ELEMENT_WARNING(
            m_GstPlugin, LIBRARY, SETTINGS,
            ("Network(uid: %u) is defined for other postprocessing but "
             "output_tensor_meta is disabled to attach. If needed, "
             "please update output_control.output_tensor_meta: true in "
             "config file: %s.",
             uniqueId(), safeStr(m_GstProperties.configPath)),
            (nullptr));
    }

    SharedInferContext context;
    if (m_PluginConfig.infer_config().has_backend()) {
        const ic::BackendParams &params = m_PluginConfig.infer_config().backend();
        if (params.has_triton()) {
            const ic::TritonParams& tritonParams = getTritonParam(params);
            if(tritonParams.has_grpc() && tritonParams.has_model_repo()) {
                GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
                ("Configuration either native or grpc should be provided"),
                ("Config file path: %s", safeStr(m_GstProperties.configPath)));
            } else if (tritonParams.has_grpc()) {
                context = std::shared_ptr<dsis::IInferContext>(
                createInferTritonGrpcContext(contextStr.c_str(), contextStr.size()));
            } else {
                context = std::shared_ptr<dsis::IInferContext>(
                createInferTrtISContext(contextStr.c_str(), contextStr.size()));
            }
        } else {
            GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration config triton missing"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
        }
    } else {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, SETTINGS,
            ("Configuration config backend missing"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
    }
    if (!context) {
        GST_ELEMENT_ERROR(m_GstPlugin, RESOURCE, FAILED,
            ("Failed to create InferContext"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
        return NVDSINFER_CONFIG_FAILED;
    }

    auto logger = [id = uniqueId(), plugin = m_GstPlugin](
                      NvDsInferLogLevel level, const char* msg) {
        gst_nvinfer_server_logger(id, level, msg, plugin);
    };
    status = context->initialize(contextStr, logger);
    if (status != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(m_GstPlugin, RESOURCE, FAILED,
            ("Failed to initialize InferTrtIsContext"),
            ("Config file path: %s", safeStr(m_GstProperties.configPath)));
        return status;
    }
    m_InferCtx = context;

    /* Initialize the object history map for source 0. */
    if (!m_ObjTrackingData.initSource(0)) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, FAILED,
            ("Failed to init object history tracking"), (nullptr));
        return NVDSINFER_CONFIG_FAILED;
    }
    m_OutputThread =
        std::make_unique<OutputThread>(
        std::bind(&GstNvInferServerImpl::outputLoop, this, std::placeholders::_1),
        "GstInferServImpl");
    assert(m_OutputThread);
    m_1stInferDone = false;

    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u started", uniqueId());
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus GstNvInferServerImpl::stop() {
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u stopping", uniqueId());

    sync();
    m_OutputThread->join();
    m_OutputThread.reset();

    m_InferCtx->deinit();
    std::unique_lock<std::mutex> locker(m_ProcessMutex);
    m_Stopped = true;
    m_ObjTrackingData.clear();
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u stopped", uniqueId());
    return NVDSINFER_SUCCESS;
}

bool GstNvInferServerImpl::addTrackingSource(uint32_t sourceId) {
    std::unique_lock<std::mutex> locker(m_ProcessMutex);
    return m_ObjTrackingData.initSource(sourceId);
}

void GstNvInferServerImpl::eraseTrackingSource(uint32_t sourceId) {
    std::unique_lock<std::mutex> locker(m_ProcessMutex);
    return m_ObjTrackingData.removeSource(sourceId);
}

void GstNvInferServerImpl::resetIntervalCounter() {
    std::unique_lock<std::mutex> locker(m_ProcessMutex);
    m_IntervalCounter = 0;
}

NvDsInferStatus GstNvInferServerImpl::processBatchMeta(
    NvDsBatchMeta* batchMeta, NvBufSurface* inSurf, uint64_t seqId, void *gstBuf) {
    if (inputTensorFromMeta()) {
        return processInputTensor(batchMeta, inSurf, seqId, gstBuf);
    } else if (isFullFrame()) {
        return processFullFrame(batchMeta, inSurf, seqId, gstBuf);
    } else {
        return processObjects(batchMeta, inSurf, seqId, gstBuf);
    }
}

static void ReturnScaleParams(BatchSurfaceBuffer* buf, SharedRequest req) {
    assert(req);
    assert(buf->getBatchSize() == req->frames.size());
    for (uint32_t i = 0; i < req->frames.size(); ++i) {
        InferFrame& frame = req->frames.at(i);
        buf->getScaleRatio(i, frame.scaleRatioX, frame.scaleRatioY);
        buf->getOffsets(i, frame.offsetLeft, frame.offsetTop);
    }
}

namespace {
// Set all options for input batchArray
NvDsInferStatus
addBatchOptions(SharedBatchArray& batchArray, SharedRequest reqBuf, int64_t uniqueId)
{
    assert(batchArray && reqBuf);
    std::unique_ptr<BufOptions> option = std::make_unique<BufOptions>();
    assert(option);
    option->setValue(OPTION_NVDS_UNIQUE_ID, uniqueId);
    option->setValue(OPTION_NVDS_BATCH_META, reqBuf->batchMeta);
    if (reqBuf->gstBuf) {
        option->setValue(OPTION_NVDS_GST_BUFFER, reqBuf->gstBuf);
        uint64_t timestamp = GST_BUFFER_DTS_OR_PTS(reqBuf->gstBuf);
        if (GST_CLOCK_TIME_IS_VALID(timestamp)) {
            option->setValue(OPTION_TIMESTAMP, timestamp);
        }
    }
    if (reqBuf->inSurf) {
        option->setValue(OPTION_NVDS_BUF_SURFACE, reqBuf->inSurf);
    }
    std::vector<NvDsFrameMeta*> frameMetaList;
    std::vector<uint64_t> streamIds;
    std::vector<NvDsObjectMeta*> objMetaList;
    std::vector<NvBufSurfaceParams*> surfParamsList;
    bool hasObjs = false;
    bool hasSurfParams = false;
    assert(!reqBuf->frames.empty());
    for (const auto& each : reqBuf->frames) {
        assert(each.frameMeta);
        frameMetaList.push_back(each.frameMeta);
        streamIds.push_back(each.frameMeta->pad_index);
        objMetaList.push_back(each.objMeta);
        surfParamsList.push_back(each.surfParams);
        if (each.objMeta) {
            hasObjs = true;
        }
        if (each.surfParams) {
            hasSurfParams = true;
        }
    }
    option->setValueArray(OPTION_NVDS_SREAM_IDS, streamIds);
    option->setValueArray(OPTION_NVDS_FRAME_META_LIST, frameMetaList);
    if (hasObjs) {
        option->setValueArray(OPTION_NVDS_OBJ_META_LIST, objMetaList);
    }
    if (hasSurfParams) {
        option->setValueArray(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST, surfParamsList);
    }
    batchArray->setOptions(std::move(option));
    return NVDSINFER_SUCCESS;
}
}

NvDsInferStatus
GstNvInferServerImpl::batchInference(SharedRequest reqBuf, std::vector<SharedBatchBuf> batchBuf)
{
    assert(m_InferCtx);
    NvDsInferStatus status = NVDSINFER_SUCCESS;
    SharedBatchArray batchArray = std::make_shared<BaseBatchArray>();
    batchArray->setBufId(reqBuf->seqId);
    for (auto& buf : batchBuf) {
        buf->setBufId(reqBuf->seqId);
    }
    batchArray->mutableBufs() = std::move(batchBuf);
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u batch-inference buf:%" PRIu64,
        uniqueId(), reqBuf->seqId);
    status = addBatchOptions(batchArray, reqBuf, uniqueId());
    if (status != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, FAILED,
            ("Failed to add options for inference "), (nullptr));
        return status;
    }

    reqBuf->future = reqBuf->promise.get_future();
    status = m_InferCtx->run(
        std::move(batchArray),
        [this, reqBuf](NvDsInferStatus s, SharedIBatchArray outputs) {
            reqBuf->status = s;
            reqBuf->outputs = std::move(outputs);
            reqBuf->promise.set_value();
        });
    if (status != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(
            m_GstPlugin, LIBRARY, FAILED, ("Failed to run inference "), (nullptr));
        return status;
    }
    status = queueOperation([this, reqBuf] {
        reqBuf->future.wait();
        this->InferenceDone(std::move(reqBuf));
    });
    if (status != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, FAILED,
            ("Failed to queue full frame inference operations"), (nullptr));
        return status;
    }
    return NVDSINFER_SUCCESS;
}

namespace {

SharedBatchSurface newBatchSurface(SharedRequest& reqBuf, int gpuId, uint32_t maxBatch,
    NvBufSurfaceMemType memType) {
    auto bufDeleter = [req = reqBuf](BatchSurfaceBuffer* buf) {
        if (buf && req) {
            ReturnScaleParams(buf, std::move(req));
        }
        delete buf;
    };
    SharedBatchSurface batchBuf(new BatchSurfaceBuffer(gpuId, maxBatch, memType), bufDeleter);
    return batchBuf;
}

} // namespace


NvDsInferStatus
GstNvInferServerImpl::processInputTensor(
    NvDsBatchMeta* batchMeta, NvBufSurface* inSurf, uint64_t seqId, void* gstBuf)
{
    assert(inputTensorFromMeta());
    bool isFirstDimBatch =
        m_PluginConfig.infer_config().input_tensor_from_meta().is_first_dim_batch();

    typedef struct{
        guint batchSize = 0;
        std::vector<SharedBatchBuf> roiTensors;
        std::vector<std::vector<InferFrame>> roiFrameList;
    } TensorInputBatch;
    std::unordered_map<guint, TensorInputBatch> tensorMetaMap;

    for (NvDsMetaList* l_user = batchMeta->batch_user_meta_list; l_user != NULL;
         l_user = l_user->next) {

        NvDsUserMeta* user_meta = (NvDsUserMeta*)(l_user->data);
        if (user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META) {
            continue;
        }
        GstNvDsPreProcessBatchMeta* preprocMeta =
            (GstNvDsPreProcessBatchMeta*)user_meta->user_meta_data;

        const auto& uids = preprocMeta->target_unique_ids;
        if (std::find(uids.begin(), uids.end(), (uint64_t)uniqueId()) == uids.end()) {
            continue;
        }

        NvDsPreProcessTensorMeta* tensorMeta = preprocMeta->tensor_meta;
        if (!tensorMeta || tensorMeta->tensor_shape.empty()) {
            continue;
        }

        guint metaId = preprocMeta->tensor_meta->meta_id;

        if (isFirstDimBatch) {
            if (!tensorMetaMap[metaId].batchSize) {
                tensorMetaMap[metaId].batchSize = tensorMeta->tensor_shape[0];
            }
            if ((int)tensorMetaMap[metaId].batchSize != tensorMeta->tensor_shape[0]) {
                GST_ELEMENT_ERROR(
                    m_GstPlugin, STREAM, FAILED,
                    ("Mismatch in input tensor batch sizes %d vs %d, "
                     "if input-tensors are non-batched, update config with\n"
                     "\tinput_tensor_from_meta { is_first_dim_batch: false }",
                     (int)tensorMetaMap[metaId].batchSize, tensorMeta->tensor_shape[0]),
                    (nullptr));
                return NVDSINFER_INVALID_PARAMS;
            }
        }

        std::vector<InferFrame> frames;

        // search each ROI
        uint32_t iBatch = 0;
        for (auto& roi : preprocMeta->roi_vector) {
            InferFrame frame;
            frame.frameWidth = inSurf->surfaceList[roi.frame_meta->batch_id].width;
            frame.frameHeight = inSurf->surfaceList[roi.frame_meta->batch_id].height;
            frame.scaleRatioX = roi.scale_ratio_x;
            frame.scaleRatioY = roi.scale_ratio_y;
            frame.offsetLeft = roi.offset_left;
            frame.offsetTop = roi.offset_top;
            frame.roiLeft = roi.roi.left;
            frame.roiTop = roi.roi.top;
            frame.roiWidth = roi.roi.width;
            frame.roiHeight = roi.roi.height;
            frame.objMeta = roi.object_meta;
            frame.surfParams = &inSurf->surfaceList[roi.frame_meta->batch_id];
            frame.roiMeta = &roi;
            frame.frameMeta = roi.frame_meta;
            frame.frameNum = roi.frame_meta->frame_num;
            frame.batchIdx = iBatch++;
            frames.emplace_back(frame);
        }
        tensorMetaMap[metaId].roiFrameList.emplace_back(frames);

        InferDataType dt = fromDSType(tensorMeta->data_type);
        InferDims tDims;
        uint32_t nDims = (uint32_t)tensorMeta->tensor_shape.size();
        tDims.numDims = isFirstDimBatch ? nDims - 1 : nDims;
        for (uint32_t iD = 0; iD < tDims.numDims; ++iD) {
            tDims.d[iD] = tensorMeta->tensor_shape[iD + (isFirstDimBatch ? 1 : 0)];
        }

        InferBufferDescription tensorDesc{
            memType : InferMemType::kGpuCuda,
            devId : tensorMeta->gpu_id,
            dataType : dt,
            dims : tDims,
            elementSize : getElementSize(dt),
            name : tensorMeta->tensor_name,
            isInput : true,
        };

        SharedRefBatchBuf tensor(new RefBatchBuffer(
            tensorMeta->raw_tensor_buffer, 0, tensorMeta->buffer_size,
            tensorDesc, tensorMetaMap[metaId].batchSize));
        tensorMetaMap[metaId].roiTensors.emplace_back(tensor);
    }

    NvDsInferStatus status = NVDSINFER_SUCCESS;
    //for each batch in tensorMetaMap
    for (auto & it : tensorMetaMap) {
        auto & tensorInputBatch = it.second;

        if (tensorInputBatch.roiFrameList.empty() || tensorInputBatch.roiTensors.empty()) {
            return NVDSINFER_SUCCESS;
        }

        /* Find first ROI frames as primary frames. */
        auto roiIter =
            std::find_if(tensorInputBatch.roiFrameList.begin(), tensorInputBatch.roiFrameList.end(), [](auto& f) { return !f.empty(); });

        if (roiIter == tensorInputBatch.roiFrameList.end()) {
            GST_ELEMENT_ERROR(
                m_GstPlugin, STREAM, FAILED,
                ("There is no ROI frame for input tensors to inference, check "
                "preprocMeta->roi_vector"),
                (nullptr));
            return NVDSINFER_INVALID_PARAMS;
        }

        if (isFirstDimBatch && maxBatchSize() > 0) {
            assert(roiIter->size() == tensorInputBatch.batchSize);
            for (uint32_t batchIdx = 0; batchIdx < tensorInputBatch.batchSize; batchIdx += maxBatchSize()) {
                uint32_t batches = std::min<uint32_t>(tensorInputBatch.batchSize - batchIdx, maxBatchSize());
                SharedRequest reqBuf(new RequestBuffer);
                reqBuf->seqId = seqId;
                reqBuf->gstBuf = gstBuf;
                reqBuf->batchMeta = batchMeta;
                reqBuf->inSurf = inSurf;
                reqBuf->frames.insert(
                    reqBuf->frames.begin(), roiIter->begin() + batchIdx,
                    roiIter->begin() + (batchIdx + batches));
                std::vector<SharedBatchBuf> tensors;
                for (auto& t : tensorInputBatch.roiTensors) {
                    const auto& desc = t->getBufDesc();
                    size_t batchBytes = dimsSize(desc.dims) * getElementSize(desc.dataType);
                    size_t bufOffset = t->getBufOffset(0);
                    if (bufOffset == (size_t) -1) {
                        return NVDSINFER_MEM_ERROR;
                    }
                    tensors.emplace_back(new RefBatchBuffer(
                        (void*)((uint8_t*)t->getBufPtr(0) + batchBytes * batchIdx),
                        bufOffset + batchBytes * batchIdx /* offset from start of allocation */,
                        batchBytes * batches, desc, batches));
                }
                status = batchInference(std::move(reqBuf), std::move(tensors));
                if (status != NVDSINFER_SUCCESS) {
                    break;
                }
            }
        } else {
            SharedRequest reqBuf(new RequestBuffer);
            reqBuf->seqId = seqId;
            reqBuf->gstBuf = gstBuf;
            reqBuf->batchMeta = batchMeta;
            reqBuf->inSurf = inSurf;
            reqBuf->frames = *roiIter;
            status = batchInference(std::move(reqBuf), std::move(tensorInputBatch.roiTensors));
        }
    }

    if (status != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(
            m_GstPlugin, LIBRARY, FAILED, ("Failed to batch inference for input meta tensors"),
            (nullptr));
    }
    return status;
}

NvDsInferStatus GstNvInferServerImpl::processFullFrame(
    NvDsBatchMeta* batchMeta, NvBufSurface* inSurf, uint64_t seqId, void *gstBuf) {
    /* Process batch only when interval_counter is 0. */
    bool skip = (m_IntervalCounter++ % (inferInterval() + 1) > 0);
    if (skip) {
        return NVDSINFER_SUCCESS;
    }
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u process full frame buf:%" PRIu64,
        uniqueId(), seqId);

    uint32_t numFilled = batchMeta->num_frames_in_batch;

    for (uint32_t i = 0; i < numFilled;) {
        SharedRequest reqBuf(new RequestBuffer);
        reqBuf->seqId = seqId;
        reqBuf->gstBuf = gstBuf;
        reqBuf->batchMeta = batchMeta;
        reqBuf->inSurf = inSurf;

        SharedBatchSurface batchBuf = newBatchSurface (reqBuf, inSurf->gpuId,
            maxBatchSize(), inSurf->memType);

        /* fill batchBuf and request buffer. */
        for (uint32_t batchIdx = 0; i < numFilled && batchIdx < maxBatchSize();
             ++i, ++batchIdx) {
            /* Adding a frame to the current batch. Set the frames members. */
            InferFrame frame;
            frame.frameWidth = inSurf->surfaceList[i].width;
            frame.frameHeight = inSurf->surfaceList[i].height;
            frame.objMeta = nullptr;
            frame.surfParams = &inSurf->surfaceList[i];
            frame.frameMeta =
                nvds_get_nth_frame_meta(batchMeta->frame_meta_list, i);
            frame.frameNum = frame.frameMeta->frame_num;
            frame.batchIdx = i;
            reqBuf->frames.emplace_back(frame);

            NvBufSurfTransformRect crop = {0, 0, inSurf->surfaceList[i].width,
                inSurf->surfaceList[i].height};
            batchBuf->append(inSurf->surfaceList[i], crop);
        }

        NvDsInferStatus status = batchInference(std::move(reqBuf), {batchBuf});
        if (status != NVDSINFER_SUCCESS) {
            GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, FAILED,
                ("Failed to batch inference"), (nullptr));
            return status;
        }
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus GstNvInferServerImpl::processObjects(
    NvDsBatchMeta* batchMeta, NvBufSurface* inSurf, uint64_t seqId, void *gstBuf) {
    SharedRequest reqBuf;
    SharedBatchSurface batchBuf;

    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u process objects buf:%" PRIu64,
        uniqueId(), seqId);

    bool warnUntrackedObj = false;
    for (NvDsMetaList* l_frame = batchMeta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta* frameMeta = (NvDsFrameMeta*)(l_frame->data);
        uint32_t sourceId = frameMeta->pad_index;

        /* Update last source seen */
        {
            std::unique_lock<std::mutex> locker(m_ProcessMutex);
            GstNvInferServerSourceInfo* source = m_ObjTrackingData.findSource(sourceId);

            /* Find the source info instance. */
            if (!source) {
                GST_WARNING_OBJECT(m_GstPlugin,
                    "Source info not found for source %d. Maybe the "
                    "GST_NVEVENT_PAD_ADDED"
                    " event was never generated for the source.",
                    frameMeta->pad_index);
                continue;
            }
            source->last_seen_frame_num = frameMeta->frame_num;
        }

        int surfaceIdx = frameMeta->batch_id;
        uint32_t frameWidth = inSurf->surfaceList[surfaceIdx].width;
        uint32_t frameHeight = inSurf->surfaceList[surfaceIdx].height;

        /* Iterate through all the objects. */
        for (NvDsMetaList* l_obj = frameMeta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta* objectMeta = (NvDsObjectMeta*)(l_obj->data);
            uint32_t frameNum = frameMeta->frame_num;

            /* Cannot infer on untracked objects in asynchronous mode. */
            if (isAsyncMode() && objectMeta->object_id == UNTRACKED_OBJECT_ID) {
                if (!warnUntrackedObj) {
                    /* Warn periodically about untracked objects in the
                     * metadata. */
                    if (m_UntrackedObjectWarnPts == GST_CLOCK_TIME_NONE ||
                        (frameMeta->buf_pts - m_UntrackedObjectWarnPts >
                            UNTRACKED_OBJECT_WARN_INTERVAL)) {
                        GST_WARNING_OBJECT(m_GstPlugin,
                            "Untracked objects in metadata. Cannot"
                            " infer on untracked objects in asynchronous "
                            "mode.");
                        m_UntrackedObjectWarnPts = frameMeta->buf_pts;
                    }
                }
                warnUntrackedObj = TRUE;
                continue;
            }

            SharedObjHistory objHistory;
            {
                // lock process
                std::unique_lock<std::mutex> locker(m_ProcessMutex);

                uint64_t objId = objectMeta->object_id;
                if (objId != UNTRACKED_OBJECT_ID) {
                    objHistory = m_ObjTrackingData.findObjHistory(sourceId, objId);
                }

                bool shouldReinfer =
                    shouldInferObject(objectMeta, frameNum, objHistory.get());

                if (objHistory && isClassify() &&
                    (!shouldReinfer || isAsyncMode())) {
                    if (!objHistory->cached_info.label.empty()) {
                        attachClassificationMetadata(
                            objectMeta, frameMeta, nullptr, objHistory->cached_info, uniqueId(),
                            classifierType(), frameWidth, frameHeight);
                    } else if (!shouldReinfer && !objHistory->under_inference) {
                        shouldReinfer = true;
                    }
                    objHistory->last_accessed_frame_num = frameMeta->frame_num;
                }

                if (!shouldReinfer) {
                    continue;
                }

                /* Object has a valid tracking id but does not have any history.
                 * Create an entry in the map for the object. */
                if (objId != UNTRACKED_OBJECT_ID && !objHistory) {
                    objHistory = m_ObjTrackingData.newHistory(sourceId, objId);
                    assert(objHistory);
                }

                /* Update the object history if it is found. */
                if (objHistory) {
                    objHistory->under_inference = TRUE;
                    objHistory->last_inferred_frame_num = frameNum;
                    objHistory->last_accessed_frame_num = frameNum;
                    objHistory->last_inferred_coords = objectMeta->rect_params;
                }
            }

            /* No existing GstNvInferServerBatch structure. Allocate a new
             * structure, acquire a buffer from our internal pool for
             * conversions. */
            if (!batchBuf) {
                assert(!reqBuf);
                reqBuf = std::make_shared<RequestBuffer>();
                reqBuf->seqId = seqId;
                reqBuf->gstBuf = gstBuf;
                reqBuf->batchMeta = batchMeta;
                reqBuf->inSurf = inSurf;
                batchBuf = newBatchSurface (reqBuf, inSurf->gpuId, maxBatchSize(),
                               inSurf->memType);
            }
            assert(batchBuf);
            /* Adding a frame to the current batch. Set the frames members. */
            InferFrame frame;
            frame.frameWidth = frameWidth;
            frame.frameHeight = frameHeight;
            frame.objMeta = objectMeta;
            frame.frameMeta = frameMeta;
            frame.surfParams = &inSurf->surfaceList[surfaceIdx];
            frame.frameNum = frameMeta->frame_num;
            frame.batchIdx = surfaceIdx;
            frame.history = objHistory;
            reqBuf->frames.emplace_back(frame);

            NvBufSurfTransformRect crop = {(uint32_t) objectMeta->rect_params.top,
                (uint32_t) objectMeta->rect_params.left,
                (uint32_t) objectMeta->rect_params.width,
                (uint32_t) objectMeta->rect_params.height};
            batchBuf->append(inSurf->surfaceList[surfaceIdx], crop);

            /* Submit batch if the batch size has reached max_batch_size. */
            if (reqBuf->frames.size() == maxBatchSize()) {
                NvDsInferStatus status = batchInference(reqBuf, {batchBuf});
                if (status != NVDSINFER_SUCCESS) {
                    GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, FAILED,
                        ("Failed to batch inference"), (nullptr));
                    return status;
                }
                reqBuf.reset();
                batchBuf.reset();
            }
        }
    }

    if (batchBuf) {
        NvDsInferStatus status = batchInference(reqBuf, {batchBuf});
        if (status != NVDSINFER_SUCCESS) {
            GST_ELEMENT_ERROR(m_GstPlugin, LIBRARY, FAILED,
                ("Failed to batch inference"), (nullptr));
            return status;
        }
    }

    {
        std::unique_lock<std::mutex> locker(m_ProcessMutex);
        m_ObjTrackingData.clearUpHistory(seqId);
    }
    return NVDSINFER_SUCCESS;
}

bool GstNvInferServerImpl::shouldInferObject(NvDsObjectMeta* objMeta,
    uint32_t frameNum, GstNvInferServerObjectHistory* history) {
    const ic::PluginControl::InputControl& inputConfig = config().input_control();
    uint32_t  secondary_reinfer_interval;
    if (inputConfig.operate_on_gie_id() > -1 &&
        objMeta->unique_component_id != inputConfig.operate_on_gie_id())
        return false;

    if (inputConfig.has_object_control()) {
        const ic::PluginControl::BBoxFilter& bboxFilter =
            inputConfig.object_control().bbox_filter();
        if (objMeta->rect_params.width < bboxFilter.min_width())
            return false;

        if (objMeta->rect_params.height < bboxFilter.min_height())
            return false;

        if (bboxFilter.max_width() > 0 &&
            objMeta->rect_params.width > bboxFilter.max_width())
            return false;

        if (bboxFilter.max_height() > 0 &&
            objMeta->rect_params.height > bboxFilter.max_height())
            return false;
    }

    /* Infer on object if the operate_on_class_ids list is empty or if
     * the flag at index  class_id is TRUE. */
    const auto& classIds = inputConfig.operate_on_class_ids();
    if (!classIds.empty () &&
        std::none_of(classIds.begin(), classIds.end(),
        [objId = objMeta->class_id](int32_t id){
        if (objId == id)
            return true;
        else
            return false;
         })){
        return false;
    }

    /* Assign the default value if the field is not set through config file*/
    if (inputConfig.has_secondary_reinfer_interval()) {
        secondary_reinfer_interval = inputConfig.secondary_reinfer_interval();
    }
    else {
        secondary_reinfer_interval = MAX_SECONDARY_REINFER_INTERVAL;
    }

    /* History is irrelevant for detectors. */
    if (history && isClassify()) {
        /* Do not infer if the object is already being inferred on maybe from a
         * previous frame. This check should be skipped when:
         * - async_mode is false (synchronous processing expected)
         * - secondary_reinfer_interval is 0 */
        if (isAsyncMode() && secondary_reinfer_interval > 0 && history->under_inference)
            return false;

        bool shouldReinfer = false;

        /* Do not reinfer if the object area has not grown by the reinference
         * area threshold and reinfer interval criteria is not met. */
        if ((history->last_inferred_coords.width *
                history->last_inferred_coords.height *
                (1 + REINFER_AREA_THRESHOLD)) <
            (objMeta->rect_params.width * objMeta->rect_params.height))
            shouldReinfer = true;

        if (frameNum - history->last_inferred_frame_num >
            secondary_reinfer_interval)
            shouldReinfer = true;

        return shouldReinfer;
    }

    if (history && isDetection()) {
        gboolean shouldReinfer = FALSE;

        if (frameNum - history->last_inferred_frame_num >
           secondary_reinfer_interval)
            shouldReinfer = TRUE;

        return shouldReinfer;
    }

    return true;
}

NvDsInferStatus GstNvInferServerImpl::sync() {
    GST_DEBUG_OBJECT(m_GstPlugin, "inferserver:%u sync event", uniqueId());
    {
        std::unique_lock<std::mutex> locker(m_ProcessMutex);
        if (m_Stopped)
            return NVDSINFER_SUCCESS;
    }

    std::promise<void> p;
    std::future<void> f = p.get_future();
    assert(m_OutputThread);
    if (!m_OutputThread->queueItem([&p]() { p.set_value(); })) {
        return NVDSINFER_UNKNOWN_ERROR;
    }
    f.wait();
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus GstNvInferServerImpl::queueOperation(FuncItem func) {
    if (!m_OutputThread->queueItem(std::move(func))) {
        return NVDSINFER_UNKNOWN_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

bool GstNvInferServerImpl::outputLoop(FuncItem func) {
    if (func) {
        func();
    }
    return true;
}

void GstNvInferServerImpl::updateLastError(NvDsInferStatus s) {
    std::unique_lock<std::mutex> locker(m_ProcessMutex);
    m_LastInferError = s;
}

NvDsInferStatus GstNvInferServerImpl::lastError() const {
    std::unique_lock<std::mutex> locker(m_ProcessMutex);
    return m_LastInferError;
}

void GstNvInferServerImpl::InferenceDone(SharedRequest req) {
    NvDsInferStatus status = req->status;

    GST_DEBUG_OBJECT(
        m_GstPlugin, "inferserver:%u inference done on buf:%" PRIu64
        " status:%s", uniqueId(), req->seqId, NvDsInferStatus2Str(status));

    if (status != NVDSINFER_SUCCESS) {
        updateLastError(status);
        return;
    }

    std::unordered_map<std::string, SharedIBatchBuffer> mapBufs;
    assert(req->outputs);
    const auto& bufArray = req->outputs;

    for (uint32_t iB = 0; iB < bufArray->getSize(); ++iB) {
        const auto& buf = bufArray->getSafeBuf(iB);
        const InferBufferDescription& desc = buf->getBufDesc();
        assert(!desc.name.empty());
        mapBufs[desc.name] = buf;
    }

    auto getBuf = [&mapBufs](const std::string& name) -> SharedIBatchBuffer {
        auto iter = mapBufs.find(name);
        if (iter == mapBufs.end())
            return nullptr;
        return iter->second;
    };

    SharedIBatchBuffer specBuf;
    std::function<NvDsInferStatus(SharedIBatchBuffer)> specFunc;
    if (isDetection()) {
        specBuf = getBuf(INFER_SERVER_DETECTION_BUF_NAME);
        specFunc = std::bind(
            &GstNvInferServerImpl::attachBatchDetection, this, req.get(),
            std::placeholders::_1);
    } else if (isClassify()) {
        specBuf = getBuf(INFER_SERVER_CLASSIFICATION_BUF_NAME);
        specFunc = std::bind(
            &GstNvInferServerImpl::attachBatchClassification, this, req.get(),
            std::placeholders::_1);
    } else if (isSegmentation()) {
        specBuf = getBuf(INFER_SERVER_SEGMENTATION_BUF_NAME);
        specFunc = std::bind(
            &GstNvInferServerImpl::attachBatchSegmentation, this, req.get(),
            std::placeholders::_1);
    }

    if (specFunc) {
        if (!specBuf) {
            GST_WARNING_OBJECT(
                m_GstPlugin,
                "inferserver:%u failed to find postprocessing "
                "result on buf:%" PRIu64,
                uniqueId(), req->seqId);
        } else {
            status = specFunc(std::move(specBuf));
        }
    }

    if (needOutputTensorMeta() || m_RawoutputCb) {
        status = handleOutputTensors(req.get());
    }
    m_1stInferDone = true;

    if (status != NVDSINFER_SUCCESS) {
        updateLastError(status);
    }
}

NvDsInferStatus
GstNvInferServerImpl::attachBatchDetection(
    RequestBuffer* req, SharedIBatchBuffer output)
{
    assert(output);
    assert(output->getBatchSize() == req->frames.size() || req->frames.size() == 1);
    assert(
        output->getBufDesc().elementSize == sizeof(NvDsInferDetectionOutput));

    GST_DEBUG_OBJECT(
        m_GstPlugin, "inferserver:%u attach batch detection on buf:%" PRIu64,
        uniqueId(), req->seqId);

    for (uint32_t i = 0; i < req->frames.size(); ++i) {
        const InferFrame& frame = req->frames[i];
        const NvDsInferDetectionOutput* detection = (const NvDsInferDetectionOutput*)(output->getBufPtr(i));
        assert(detection);
        attachDetectionMetadata(
            frame.frameMeta, frame.objMeta, *detection, frame.scaleRatioX, frame.scaleRatioY,
            frame.offsetLeft, frame.offsetTop, frame.roiLeft, frame.roiTop, frame.roiWidth,
            frame.roiHeight, frame.frameWidth, frame.frameHeight, uniqueId(), config());
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
GstNvInferServerImpl::attachBatchClassification(
    RequestBuffer* req, SharedIBatchBuffer output)
{
    assert(output);
    assert(output->getBatchSize() == req->frames.size() || req->frames.size() == 1);
    assert(
        output->getBufDesc().elementSize == sizeof(InferClassificationOutput));

    GST_DEBUG_OBJECT(
        m_GstPlugin, "inferserver:%u attach batch classification on buf:%" PRIu64,
        uniqueId(), req->seqId);

    for (uint32_t i = 0; i < req->frames.size(); ++i) {
        InferFrame& frame = req->frames[i];
        const InferClassificationOutput* classification =
            (const InferClassificationOutput*)output->getBufPtr(i);
        assert(classification);
        InferClassificationOutput newInfo = *classification;
        auto obj_history = frame.history.lock();

        /* If we have an object's history and the buffer PTS is same as last
         * inferred PTS mark the object as not being inferred. This check could
         * be useful if object is inferred multiple times before completion of
         * an existing inference. */
        if (obj_history) {
            std::unique_lock<std::mutex> locker(m_ProcessMutex);
            if (obj_history->last_inferred_frame_num == frame.frameNum)
                obj_history->under_inference = FALSE;
            /* Object history is available merge the old and new classification
             * results. */
            mergeClassificationOutput(obj_history->cached_info, newInfo);
            /* If cache is still empty after merge (inference returned null and
             * no previous valid result exists), reset last_inferred_frame_num
             * to force re-inference on the next frame. */
            if (obj_history->cached_info.label.empty()) {
                obj_history->last_inferred_frame_num = 0;
            }
            /* Use the merged classification results if available otherwise use
             * the new results. */
            newInfo = obj_history->cached_info;
        }

        if (!isAsyncMode() && !newInfo.label.empty()) {
            attachClassificationMetadata(
                frame.objMeta, frame.frameMeta, frame.roiMeta, newInfo, uniqueId(),
                classifierType(), frame.frameWidth, frame.frameHeight);
        }
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
GstNvInferServerImpl::attachBatchSegmentation(
    RequestBuffer* req, SharedIBatchBuffer output)
{
    assert(output);
    assert(output->getBatchSize() == req->frames.size() || req->frames.size() == 1);
    assert(output->getBufDesc().elementSize ==
           sizeof(NvDsInferSegmentationOutput));

   GST_DEBUG_OBJECT(
        m_GstPlugin, "inferserver:%u attach batch segmentation on buf:%" PRIu64,
        uniqueId(), req->seqId);

   for (uint32_t i = 0; i < req->frames.size(); ++i) {
       const InferFrame& frame = req->frames[i];
       const NvDsInferSegmentationOutput* segment =
           (const NvDsInferSegmentationOutput*)output->getBufPtr(i);
       attachSegmentationMetadata(frame.objMeta, frame.frameMeta, frame.roiMeta, *segment, output);
   }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus GstNvInferServerImpl::handleOutputTensors(RequestBuffer* req) {
    assert(req);
    assert(!req->frames.empty());
    uint32_t frameNum = req->frames.size();

    GST_DEBUG_OBJECT(
        m_GstPlugin, "inferserver:%u attach batch output tensors on buf:%" PRIu64,
        uniqueId(), req->seqId);

    auto& bufArray = req->outputs;
    std::vector<SharedIBatchBuffer> tensors;
    std::vector<SharedIBatchBuffer> fullTensors;
    for (uint32_t iB = 0; iB < bufArray->getSize(); ++iB) {
        const auto& buf = bufArray->getSafeBuf(iB);
        assert(buf);
        const InferBufferDescription& desc = buf->getBufDesc();
        assert(!desc.name.empty());
        if (isPrivateTensor(desc.name)) {
            continue;
        }
        fullTensors.emplace_back(buf);
        if (buf->getBatchSize() != frameNum && frameNum != 1 && !m_1stInferDone) {
            GST_WARNING_OBJECT(
                m_GstPlugin,
                "Output tensor: %s is mix-batched. Could not be attached. "
                "Please implement custom-lib derived from nvdsinferserver::IInferCustomProcessor "
                "and remove field \n\toutput_control {output_tensor_meta}\n in config file.",
                safeStr(desc.name));
            continue;
        }
        assert(isCpuMem(desc.memType));

        tensors.emplace_back(buf);
    }
    if (tensors.empty()) {
        return NVDSINFER_SUCCESS;
    }

    NvDsInferNetworkInfo inputInfo{0, 0, 0};
    m_InferCtx->getNetworkInputInfo(inputInfo);

    /* Raw output callback */
    if (m_RawoutputCb) {
        std::vector<NvDsInferLayerInfo> cLayersInfo(tensors.size());
        for (size_t iT = 0; iT < tensors.size(); ++iT) {
            const InferBufferDescription& desc = tensors[iT]->getBufDesc();
            NvDsInferLayerInfo &layerInfo = cLayersInfo[iT];
            layerInfo = toCapiLayerInfo(desc);
            layerInfo.bindingIndex = iT;
            layerInfo.buffer = tensors[iT]->getBufPtr(0);
        }
        m_RawoutputCb(req->gstBuf, inputInfo, cLayersInfo.data(), cLayersInfo.size(), frameNum);
    }

    /* Attach metadata */
    if (needOutputTensorMeta()) {
        for (uint32_t i = 0; i < frameNum; ++i) {
            attachTensorOutputMeta(
                req->frames[i].objMeta, req->frames[i].frameMeta, req->frames[i].roiMeta,
                uniqueId(), tensors, i, inputInfo, maintainAspectRatio(), symmetricPadding());
        }
        if (inputTensorFromMeta()) {
            attachFullTensorOutputMeta(req->batchMeta, uniqueId(), fullTensors, inputInfo);
        }
    }
    return NVDSINFER_SUCCESS;
}

} // namespace gstnvinferserver
