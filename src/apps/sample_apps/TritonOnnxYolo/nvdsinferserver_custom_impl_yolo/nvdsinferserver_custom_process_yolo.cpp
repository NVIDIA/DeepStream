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

#include <string.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include "infer_custom_process.h"
#include "nvbufsurface.h"
#include "nvdsmeta.h"

typedef struct _GstBuffer GstBuffer;

/** This is a example how DeepStream Triton plugin(gst-nvinferserver) do
 * custom extra input preprocess and custom postprocess on triton based models.
 */

// enable debug log
#define ENABLE_DEBUG 0

namespace dsis = nvdsinferserver;

#if ENABLE_DEBUG
#define LOG_DEBUG(fmt, ...) fprintf(stdout, "%s:%d" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

#define LOG_ERROR(fmt, ...) fprintf(stderr, "%s:%d" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#define CSTR(str) (str).empty() ? "" : (str).c_str()

// constant values definition
static const std::vector<std::string> kClassLabels = {
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush",
};

/** Define a function for custom processor for DeepStream Triton plugin(nvinferserver)
 * do custom extra input preprocess and custom postprocess on triton based models.
 * The sysmbol is loaded through
 *   infer_config {
 *     custom_lib {  path: "path/to/custom_impl_process.so" }
 *     extra {
 *       custom_process_funcion: "CreateInferServerCustomProcess"
 *     }}
 */
extern "C" dsis::IInferCustomProcessor* CreateInferServerCustomProcess(
    const char* config, uint32_t configLen);

namespace {
using namespace dsis;

std::string
memType2Str(InferMemType t)
{
    switch (t) {
    case InferMemType::kGpuCuda:
        return "kGpuCuda";
    case InferMemType::kCpu:
        return "kCpu";
    case InferMemType::kCpuCuda:
        return "kCpuPinned";
    default:
        return "Unknown";
    }
}

std::string
dataType2Str(dsis::InferDataType t)
{
    switch (t) {
    case InferDataType::kFp32:
        return "kFp32";
    case InferDataType::kFp16:
        return "kFp16";
    case InferDataType::kInt8:
        return "kInt8";
    case InferDataType::kInt32:
        return "kInt32";
    case InferDataType::kInt16:
        return "kInt16";
    case InferDataType::kUint8:
        return "kUint8";
    case InferDataType::kUint16:
        return "kUint16";
    case InferDataType::kUint32:
        return "kUint32";
    case InferDataType::kFp64:
        return "kFp64";
    case InferDataType::kInt64:
        return "kInt64";
    case InferDataType::kUint64:
        return "kUint64";
    case InferDataType::kString:
        return "kString";
    case InferDataType::kBool:
        return "kBool";
    default:
        return "Unknown";
    }
}

// return buffer description string
std::string
strOfBufDesc(const dsis::InferBufferDescription& desc)
{
    std::stringstream ss;
    ss << "*" << desc.name << "*, shape: ";
    for (uint32_t i = 0; i < desc.dims.numDims; ++i) {
        if (i != 0) {
            ss << "x";
        } else {
            ss << "[";
        }
        ss << desc.dims.d[i];
        if (i == desc.dims.numDims - 1) {
            ss << "]";
        }
    }
    ss << ", dataType:" << dataType2Str(desc.dataType);
    ss << ", memType:" << memType2Str(desc.memType);
    return ss.str();
}

}  // namespace

/** Example of a Custom process instance for deepstream-triton(gst-nvinferserver) plugin
 * It is derived from nvdsinferserver::IInferCustomProcessor
 * If should be loaded through
 * config_triton_inferserver_primary_fasterRCNN.txt:
 *   infer_config {
 *     custom_lib {  path: "path/to/custom_impl_process.so" }
 *     extra {
 *       custom_process_funcion: "CreateInferServerCustomProcess"
 *     }
 *   }
 */
class NvInferServerCustomProcess : public dsis::IInferCustomProcessor {
private:
    std::map<uint64_t, std::vector<float>> _streamFeedback;
    std::mutex _streamMutex;

public:
    ~NvInferServerCustomProcess() override = default;
    /** override function
     * Specifies supported extraInputs memtype in extraInputProcess()
     */
    void supportInputMemType(dsis::InferMemType& type) override { type = dsis::InferMemType::kCpu; }

    /** override function
     * check whether custom loop process needed.
     * If return True, extraInputProcess() and inferenceDone() runs in order per stream_ids
     * This is usually for LSTM loop purpose. FasterRCNN does not need it.
     * The code for requireInferLoop() conditions just sample when user has
     * a LSTM-like Loop model and requires loop custom processing.
     * */
    bool requireInferLoop() const override { return false; }

    /**
     * override function
     * Do custom processing on extra inputs.
     * @primaryInput is already preprocessed. DO NOT update it again.
     * @extraInputs, do custom processing and fill all data according the tensor shape
     * @options, it has most of the common Deepstream metadata along with primary data.
     *           e.g. NvDsBatchMeta, NvDsObjectMeta, NvDsFrameMeta, stream ids...
     *           see infer_ioptions.h to see all the potential key name and structures
     *           in the key-value table.
     */
    NvDsInferStatus extraInputProcess(
        const std::vector<dsis::IBatchBuffer*>&
            primaryInputs,  // primary tensor(image) has been processed
        std::vector<dsis::IBatchBuffer*>& extraInputs, const dsis::IOptions* options) override
    {
        INFER_ASSERT(primaryInputs.size() > 0);
        INFER_ASSERT(extraInputs.size() == 1);
        // primary input tensor: input_1 [batch, channel, height, width]
        dsis::InferBufferDescription input0Desc = primaryInputs[0]->getBufDesc();
        // extra input tensor: image_shape [batch, 2]
        dsis::InferBufferDescription extra1Desc = extraInputs[0]->getBufDesc();
        INFER_ASSERT(extra1Desc.dataType == dsis::InferDataType::kFp32);
        INFER_ASSERT(extra1Desc.elementSize == sizeof(float));  // bytes per element

        INFER_ASSERT(!strOfBufDesc(input0Desc).empty());
        LOG_DEBUG("extraInputProcess: primary input %s", strOfBufDesc(input0Desc).c_str());
        LOG_DEBUG("extraInputProcess: extra input %s", strOfBufDesc(extra1Desc).c_str());

        // batch size must be get from primary input tensor.
        // extra inputs 'image_shape' does not have a batch size in this specific model
        int batchSize = input0Desc.dims.d[0];
        INFER_ASSERT(extra1Desc.dims.numDims == 2 && extra1Desc.dims.d[0] == batchSize);
        INFER_ASSERT(batchSize >= 1);
        if (!options) {
            LOG_ERROR("custom process does not receive IOptions");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }

        NvDsBatchMeta* batchMeta = nullptr;
        std::vector<NvDsFrameMeta*> frameMetaList;
        NvBufSurface* bufSurf = nullptr;
        std::vector<NvBufSurfaceParams*> surfParamsList;
        std::vector<uint64_t> streamIds;
        int64_t unique_id = 0;
        INFER_ASSERT(options->getValueArray(OPTION_NVDS_SREAM_IDS, streamIds) == NVDSINFER_SUCCESS);
        INFER_ASSERT(streamIds.size() == (uint32_t)batchSize);

        // get NvBufSurface
        if (options->hasValue(OPTION_NVDS_BUF_SURFACE)) {
            INFER_ASSERT(options->getObj(OPTION_NVDS_BUF_SURFACE, bufSurf) == NVDSINFER_SUCCESS);
        }
        INFER_ASSERT(bufSurf);

        // get NvDsBatchMeta
        if (options->hasValue(OPTION_NVDS_BATCH_META)) {
            INFER_ASSERT(options->getObj(OPTION_NVDS_BATCH_META, batchMeta) == NVDSINFER_SUCCESS);
        }
        INFER_ASSERT(batchMeta);

        // get all frame meta list into vector<NvDsFrameMeta*>
        if (options->hasValue(OPTION_NVDS_FRAME_META_LIST)) {
            INFER_ASSERT(
                options->getValueArray(OPTION_NVDS_FRAME_META_LIST, frameMetaList) ==
                NVDSINFER_SUCCESS);
        }

        // get unique_id
        if (options->hasValue(OPTION_NVDS_UNIQUE_ID)) {
            INFER_ASSERT(options->getInt(OPTION_NVDS_UNIQUE_ID, unique_id) == NVDSINFER_SUCCESS);
        }

        // get all surface params list into vector<NvBufSurfaceParams*>
        if (options->hasValue(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST)) {
            INFER_ASSERT(
                options->getValueArray(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST, surfParamsList) ==
                NVDSINFER_SUCCESS);
        }

        // fill extra input tensor "image_shape[-1,2]"
        float* image_shape = (float*)extraInputs[0]->getBufPtr(0);

        for (int iBatch = 0; iBatch < batchSize; ++iBatch) {
            image_shape[iBatch * 2 + 1] = (float)surfParamsList[iBatch]->width;
            image_shape[iBatch * 2] = (float)surfParamsList[iBatch]->height;
        }
        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Custom processing for inferenced output tensors.
     * output memory types is controlled by gst-nvinferserver config file
     *     config_triton_inferserver_primary_fasterRCNN.txt:
     *       infer_config {
     *         backend {  output_mem_type: MEMORY_TYPE_CPU }
     *     }
     * User can even attach parsed metadata into GstBuffer from this function
     */
    NvDsInferStatus inferenceDone(
        const dsis::IBatchArray* outputs, const dsis::IOptions* inOptions) override
    {
        std::vector<uint64_t> streamIds;
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_SREAM_IDS, streamIds) == NVDSINFER_SUCCESS);
        INFER_ASSERT(!streamIds.empty());
        uint32_t batchSize = streamIds.size();
        std::vector<std::vector<NvDsInferObjectDetectionInfo>> parsedBboxes(batchSize);

        // add 3 output tensors into map
        std::unordered_map<std::string, const dsis::IBatchBuffer*> tensors;
        for (uint32_t iTensor = 0; iTensor < outputs->getSize(); ++iTensor) {
            const dsis::IBatchBuffer* outTensor = outputs->getBuffer(iTensor);
            INFER_ASSERT(outTensor);
            auto desc = outTensor->getBufDesc();
            LOG_DEBUG("out tensor: %s, desc: %s \n", CSTR(desc.name), strOfBufDesc(desc).c_str());
            tensors.emplace(desc.name, outTensor);
        }

        // indices dim format is [total, 3], each row represents [batch_index, class_index,
        // box_index]
        auto indices = tensors["yolonms_layer_1/concat_2:0"];
        // boxes dim format is [batch, n_candidates, 4]
        auto boxes = tensors["yolonms_layer_1/ExpandDims_1:0"];
        // scores dim format is [1, n_classes, n_candidates]
        auto scores = tensors["yolonms_layer_1/ExpandDims_3:0"];
        INFER_ASSERT(indices && boxes && scores);
        auto idxDesc = indices->getBufDesc();
        auto boxDesc = boxes->getBufDesc();
        auto scoreDesc = scores->getBufDesc();
        INFER_ASSERT(idxDesc.dims.numDims == 2 && idxDesc.dims.d[1] == 3);
        INFER_ASSERT(boxDesc.dims.numDims == 3 && boxDesc.dims.d[2] == 4);
        INFER_ASSERT(
            scoreDesc.dims.numDims == 3 && scoreDesc.dims.d[1] == (int)kClassLabels.size());
        int32_t* indicesPtr = (int32_t*)indices->getBufPtr(0);
        float* boxesPtr = (float*)boxes->getBufPtr(0);
        float* scoresPtr = (float*)scores->getBufPtr(0);
        for (int idx = 0; idx < idxDesc.dims.d[0]; ++idx) {
            NvDsInferObjectDetectionInfo obj{};
            uint32_t batchId = (uint32_t)indicesPtr[idx * 3];
            obj.classId = (uint32_t)indicesPtr[idx * 3 + 1];
            INFER_ASSERT(obj.classId < kClassLabels.size());
            uint32_t boxIdx = indicesPtr[idx * 3 + 2];
            float* bbox = &boxesPtr[4 * boxIdx];
            obj.top = bbox[0];
            obj.left = bbox[1];
            float bottom = bbox[2];
            float right = bbox[3];
            obj.height = bottom - obj.top;
            obj.width = right - obj.left;
            float score = scoresPtr
                [batchId * scoreDesc.dims.d[1] * scoreDesc.dims.d[2] +
                 obj.classId * scoreDesc.dims.d[2] + boxIdx];
            obj.detectionConfidence = score;
            LOG_DEBUG(
                "cid: %u, n: %u, obj [%.2f, %.2f, %.2f, %.2f], score:%.2f\n", obj.classId, batchId,
                obj.top, obj.left, bottom, right, score);
            parsedBboxes[batchId].emplace_back(obj);
        }

        for (uint32_t iBatch = 0; iBatch < batchSize; ++iBatch) {
            INFER_ASSERT(attachObjMeta(inOptions, parsedBboxes[iBatch], 0) == NVDSINFER_SUCCESS);
        }

        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Receiving errors if anything wrong inside lowlevel lib
     */
    void notifyError(NvDsInferStatus s) override
    {
        std::unique_lock<std::mutex> locker(_streamMutex);
        _streamFeedback.clear();
    }

private:
    /** function for loop processing only. not requried for fasterRCNN
     */
    NvDsInferStatus feedbackStreamInput(
        const dsis::IBatchArray* outputs, const dsis::IOptions* inOptions);

    /**
     * attach bounding boxes into NvDsBatchMeta and NvDsFrameMeta
     */
    NvDsInferStatus attachObjMeta(
        const dsis::IOptions* inOptions, const std::vector<NvDsInferObjectDetectionInfo>& objs,
        uint32_t batchIdx);
};

/** Implementation to Create a custom processor for DeepStream Triton
 * plugin(nvinferserver) to do custom extra input preprocess and custom
 * postprocess on triton based models.
 */
extern "C" {
dsis::IInferCustomProcessor*
CreateInferServerCustomProcess(const char* config, uint32_t configLen)
{
    return new NvInferServerCustomProcess();
}
}

/**
 * attach bounding boxes into NvDsBatchMeta and NvDsFrameMeta
 */
NvDsInferStatus
NvInferServerCustomProcess::attachObjMeta(
    const dsis::IOptions* inOptions, const std::vector<NvDsInferObjectDetectionInfo>& detectObjs,
    uint32_t batchIdx)
{
    INFER_ASSERT(inOptions);
    GstBuffer* gstBuf = nullptr;
    NvDsBatchMeta* batchMeta = nullptr;
    std::vector<NvDsFrameMeta*> frameMetaList;
    NvBufSurface* bufSurf = nullptr;
    std::vector<NvBufSurfaceParams*> surfParamsList;
    int64_t unique_id = 0;

    // get GstBuffer
    if (inOptions->hasValue(OPTION_NVDS_GST_BUFFER)) {
        INFER_ASSERT(inOptions->getObj(OPTION_NVDS_GST_BUFFER, gstBuf) == NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(gstBuf);

    // get NvBufSurface
    if (inOptions->hasValue(OPTION_NVDS_BUF_SURFACE)) {
        INFER_ASSERT(inOptions->getObj(OPTION_NVDS_BUF_SURFACE, bufSurf) == NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(bufSurf);

    // get NvDsBatchMeta
    if (inOptions->hasValue(OPTION_NVDS_BATCH_META)) {
        INFER_ASSERT(inOptions->getObj(OPTION_NVDS_BATCH_META, batchMeta) == NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchMeta);

    // get all frame meta list into vector<NvDsFrameMeta*>
    if (inOptions->hasValue(OPTION_NVDS_FRAME_META_LIST)) {
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_FRAME_META_LIST, frameMetaList) ==
            NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchIdx < frameMetaList.size());  // batchsize

    // get unique_id
    if (inOptions->hasValue(OPTION_NVDS_UNIQUE_ID)) {
        INFER_ASSERT(inOptions->getInt(OPTION_NVDS_UNIQUE_ID, unique_id) == NVDSINFER_SUCCESS);
    }

    // get all surface params list into vector<NvBufSurfaceParams*>
    if (inOptions->hasValue(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST)) {
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST, surfParamsList) ==
            NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchIdx < surfParamsList.size());  // batchsize

    // attach object's boundingbox
    for (const auto& obj : detectObjs) {
        NvDsObjectMeta* objMeta = nvds_acquire_obj_meta_from_pool(batchMeta);
        objMeta->unique_component_id = unique_id;
        objMeta->confidence = obj.detectionConfidence;

        /* This is an untracked object. Set tracking_id to -1. */
        objMeta->object_id = UNTRACKED_OBJECT_ID;
        objMeta->class_id = obj.classId;

        NvOSD_RectParams& rect_params = objMeta->rect_params;
        NvOSD_TextParams& text_params = objMeta->text_params;

        rect_params.left = obj.left;
        rect_params.top = obj.top;
        rect_params.width = obj.width;
        rect_params.height = obj.height;

        /* Border of width 3. */
        rect_params.border_width = 3;
        rect_params.has_bg_color = 0;
        rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};

        /* display_text requires heap allocated memory. */
        if (obj.classId < kClassLabels.size()) {
            text_params.display_text = g_strdup(kClassLabels[obj.classId].c_str());
            strncpy(objMeta->obj_label, kClassLabels[obj.classId].c_str(), MAX_LABEL_SIZE - 1);
            objMeta->obj_label[MAX_LABEL_SIZE - 1] = 0;
        }
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = (gchar*)"Serif";
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};

        nvds_acquire_meta_lock(batchMeta);
        nvds_add_obj_meta_to_frame(frameMetaList[batchIdx], objMeta, NULL);
        frameMetaList[batchIdx]->bInferDone = TRUE;
        nvds_release_meta_lock(batchMeta);
    }

    return NVDSINFER_SUCCESS;
}
