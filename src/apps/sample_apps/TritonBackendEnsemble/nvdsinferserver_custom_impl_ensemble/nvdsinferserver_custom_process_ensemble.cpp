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
#include "infer_post_datatypes.h"
#include "nvbufsurface.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"

typedef struct _GstBuffer GstBuffer;

/** This file implements the custom extra input preprocess and custom
 * postprocess for the Triton ensemble model example in DeepStream.
 */

namespace dsis = nvdsinferserver;

#if ENABLE_DEBUG
#define LOG_DEBUG(fmt, ...) fprintf(stdout, "%s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

#define LOG_ERROR(fmt, ...) fprintf(stderr, "%s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

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

static inline const char* safeStr(const char* str)
{
    return !str ? "" : str;
}

static inline const char* safeStr(const std::string& str)
{
    return str.c_str();
}

/** Labels for the VehicleMake classifier model. */
static const std::vector<std::string> kVehicleMakeLabels = {
     "acura",    "audi",    "bmw",      "chevrolet",
     "chrysler", "dodge",   "ford",     "gmc",
     "honda",    "hyundai", "infiniti", "jeep",
     "kia",      "lexus",   "mazda",    "mercedes",
     "nissan",   "subaru",  "toyota",   "volkswagen"
};

/** Labels for the VehicleTypes classifier model. */
static const std::vector<std::string> kVehicleTypeLabels = {
     "coupe", "largevehicle", "sedan", "suv", "truck", "van"
};

constexpr float kClassifierThreshold = 0.51f;

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

    /** Override function
     * check whether custom loop process needed.
     * If return True, extraInputProcess() and inferenceDone() runs in order per stream_ids
     * This is usually for LSTM loop purpose. FasterRCNN does not need it.
     * The code for requireInferLoop() conditions just sample when user has
     * a LSTM-like Loop model and requires loop custom processing.
     * */
    bool requireInferLoop() const override { return false; }

    /**
     * Override function
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
        // primary input tensor: INPUT [batch, channel, height, width]
        dsis::InferBufferDescription input0Desc = primaryInputs[0]->getBufDesc();
        // extra input tensor: STREAM_ID [batch, 1]
        dsis::InferBufferDescription extra1Desc = extraInputs[0]->getBufDesc();
        INFER_ASSERT(extra1Desc.dataType == dsis::InferDataType::kInt32);
        INFER_ASSERT(extra1Desc.elementSize == sizeof(int32_t));  // bytes per element

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

        // fill extra input tensor "STREAM_ID[-1,1]"
        int32_t* streamIdTensor = (int32_t*)extraInputs[0]->getBufPtr(0);

        for (int iBatch = 0; iBatch < batchSize; ++iBatch) {
            streamIdTensor[iBatch] = streamIds[iBatch];
        }
        return NVDSINFER_SUCCESS;
    }

    /** Override function
     * Custom processing for inferenced output tensors.
     * output memory types is controlled by gst-nvinferserver config file
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

        // add output tensors into map
        std::unordered_map<std::string, const dsis::IBatchBuffer*> tensors;

        for (uint32_t iTensor = 0; iTensor < outputs->getSize(); ++iTensor) {
            const dsis::IBatchBuffer* outTensor = outputs->getBuffer(iTensor);
            INFER_ASSERT(outTensor);
            auto desc = outTensor->getBufDesc();
            LOG_DEBUG("out tensor: %s, desc: %s \n", CSTR(desc.name), strOfBufDesc(desc).c_str());
            tensors.emplace(desc.name, outTensor);
        }

        auto outputBuf = tensors["INFERENCE_COUNT"];
        INFER_ASSERT(outputBuf);
        auto outDesc = outputBuf->getBufDesc();
        INFER_ASSERT(outDesc.dims.numDims == 2 && outDesc.dims.d[1] == 1);
        INFER_ASSERT(outDesc.dims.d[0] == (int32_t)batchSize);

        /* Read the inference count output and print using debug logs. */
        int32_t* outInferenceCounts = (int32_t*)outputBuf->getBufPtr(0);
        for (int idx = 0; idx < outDesc.dims.d[0]; ++idx) {
            LOG_DEBUG("Inference count = %d for batch index %d\n", outInferenceCounts[idx], idx);
            (void)outInferenceCounts; // avoid warning
        }

        /* Parse the classifier outputs. */
        std::vector<const dsis::IBatchBuffer*> classTensorOutput;
        std::vector<std::vector<std::string>> labels;

        classTensorOutput.push_back(tensors["VEHICLE_MAKE"]);
        classTensorOutput.push_back(tensors["VEHICLE_TYPE"]);

        labels.push_back(kVehicleMakeLabels);
        labels.push_back(kVehicleTypeLabels);

        /* Get the number of attributes supported by the classifier. */
        unsigned int numAttributes = classTensorOutput.size();
        /* Iterate through all the output coverage layers of the classifier.  */
        for (unsigned int l = 0; l < numAttributes; l++) {
            const dsis::IBatchBuffer* outputBuf = classTensorOutput[l];
            INFER_ASSERT(outputBuf);
            auto outDesc = outputBuf->getBufDesc();
            // Support both 2D [batch, classes] and 4D [batch, classes, 1, 1] tensor formats
            LOG_DEBUG("Classifier output %u: name=%s, numDims=%d, dims=[%d, %d, %d, %d]\n",
                     l, CSTR(outDesc.name), outDesc.dims.numDims,
                     outDesc.dims.d[0], outDesc.dims.d[1],
                     outDesc.dims.numDims > 2 ? outDesc.dims.d[2] : 0,
                     outDesc.dims.numDims > 3 ? outDesc.dims.d[3] : 0);

            INFER_ASSERT(outDesc.dims.numDims == 2 || outDesc.dims.numDims == 4);

            int numClasses = outDesc.dims.d[1];
            INFER_ASSERT(numClasses == (int)labels[l].size());
            LOG_DEBUG("Verified: numClasses=%d matches labels[%u].size()=%zu\n",
                     numClasses, l, labels[l].size());
            INFER_ASSERT(outDesc.dims.d[0] == (int32_t)batchSize);
        }

        for (unsigned int idx = 0; idx < batchSize; ++idx) {
            std::string attrString;
            std::vector<NvDsInferAttribute> attributes;

            for (unsigned int l = 0; l < numAttributes; l++) {
                const dsis::IBatchBuffer* outputBuf = classTensorOutput[l];
                auto outDesc = outputBuf->getBufDesc();

                // Calculate numClasses and pointer offset based on tensor dimensions
                unsigned int numClasses = outDesc.dims.d[1];
                int stridePerBatch = numClasses;

                // For 4D tensors [batch, classes, h, w], stride is classes * h * w
                if (outDesc.dims.numDims == 4) {
                    stridePerBatch = outDesc.dims.d[1] * outDesc.dims.d[2] * outDesc.dims.d[3];
                }

                float* outPtr = (float*)outputBuf->getBufPtr(0) + idx * stridePerBatch;
                float maxProbability = 0;
                bool attrFound = false;
                NvDsInferAttribute attr;

                /* Iterate through all the probabilities that the object belongs to
                 * each class. Find the maximum probability and the corresponding class
                 * which meets the minimum threshold. */
                for (unsigned int c = 0; c < numClasses; c++) {
                    float probability = outPtr[c];
                    if (probability > kClassifierThreshold &&
                            probability > maxProbability) {
                        maxProbability = probability;
                        attrFound = true;
                        attr.attributeIndex = l;
                        attr.attributeValue = c;
                        attr.attributeConfidence = probability;
                    }
                }
                if (attrFound) {
                    if (labels.size() > attr.attributeIndex &&
                            attr.attributeValue < labels[attr.attributeIndex].size())
                        attr.attributeLabel =
                            strdup(labels[attr.attributeIndex][attr.attributeValue].c_str());
                    else
                        attr.attributeLabel = nullptr;
                    attributes.push_back(attr);
                    if (attr.attributeLabel)
                        attrString.append(attr.attributeLabel).append(" ");
                }
            }

            InferClassificationOutput output;

            for (auto& attr : attributes) {
                InferAttribute safeAttr;
                static_cast<NvDsInferAttribute&>(safeAttr) = attr;
                safeAttr.safeAttributeLabel = safeStr(attr.attributeLabel);
                safeAttr.attributeLabel = (char *) safeAttr.safeAttributeLabel.c_str();
                output.attributes.emplace_back(safeAttr);
                free(attr.attributeLabel);
            }

            INFER_ASSERT(attachClassificationMeta(inOptions, output, idx)
                    == NVDSINFER_SUCCESS);

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
    NvDsInferStatus attachClassificationMeta(
        const dsis::IOptions* inOptions, InferClassificationOutput& objInfo,
        uint32_t batchIdx);
};

/** Implementation to create a custom processor for DeepStream Triton
 * plugin (nvinferserver) to do custom extra input preprocess and custom
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
 * Attach classification results into NvDsObjectMeta
 */
NvDsInferStatus
NvInferServerCustomProcess::attachClassificationMeta(
    const dsis::IOptions* inOptions, InferClassificationOutput& objInfo,
    uint32_t batchIdx)
{
    INFER_ASSERT(inOptions);
    GstBuffer* gstBuf = nullptr;
    NvDsBatchMeta* batchMeta = nullptr;
    std::vector<NvDsObjectMeta*> objectMetaList;
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

    // get all object meta list into vector<NvDsObjectMeta*> for secondary classifier
    if (inOptions->hasValue(OPTION_NVDS_OBJ_META_LIST)) {
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_OBJ_META_LIST, objectMetaList) ==
            NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchIdx < objectMetaList.size());  // batchsize

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

    auto objMeta = objectMetaList[batchIdx];
    std::string string_label = objInfo.label;

    /* Fill the attribute info structure for the object. */
    guint numAttrs = objInfo.attributes.size();

    nvds_acquire_meta_lock(batchMeta);

    NvDsClassifierMeta* classifier_meta =
        nvds_acquire_classifier_meta_from_pool(batchMeta);

    classifier_meta->unique_component_id = unique_id;

    for (unsigned int i = 0; i < numAttrs; i++) {
        NvDsLabelInfo* label_info =
            nvds_acquire_label_info_meta_from_pool(batchMeta);
        const InferAttribute& attr = objInfo.attributes[i];
        label_info->label_id = attr.attributeIndex;
        label_info->result_class_id = attr.attributeValue;
        label_info->result_prob = attr.attributeConfidence;
        if (!attr.safeAttributeLabel.empty()) {
            g_strlcpy(
                label_info->result_label, attr.safeAttributeLabel.c_str(),
                MAX_LABEL_SIZE);
            if (objInfo.label.length() == 0)
                string_label.append(attr.safeAttributeLabel).append(" ");
        }

        nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
    }

    if (string_label.length() > 0 && objMeta) {
        gchar* temp = objMeta->text_params.display_text;
        objMeta->text_params.display_text =
            g_strconcat(temp, " ", string_label.c_str(), nullptr);
        g_free(temp);
    }

    nvds_add_classifier_meta_to_object(objMeta, classifier_meta);
    nvds_release_meta_lock(batchMeta);

    return NVDSINFER_SUCCESS;
}
