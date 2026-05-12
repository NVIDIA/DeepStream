/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <fstream>
#include <cassert>

#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#ifdef WITH_OPENCV
#include <opencv2/objdetect/objdetect.hpp>
#endif
#pragma GCC diagnostic pop

#include "infer_batch_buffer.h"
#include "infer_datatypes.h"
#include "infer_post_datatypes.h"
#include "infer_postproc_buf.h"
#include "infer_postprocess.h"
#include "infer_utils.h"
#include "nvdsinfer_dbscan.h"

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

static const bool ATHR_ENABLED = true;
static const float ATHR_THRESHOLD = 60.0;

namespace nvdsinferserver {

/* Parse all object bounding boxes for the class `classIndex` in the frame
 * meeting the minimum threshold criteria.
 *
 * This parser function has been specifically written for the sample resnet10
 * model provided with the SDK. Other models will require this function to be
 * modified.
 */

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the
 * DeepStreamSDK documentation.
 */
NvDsInferStatus Postprocessor::parseLabelsFile(
    const std::string& labelsFilePath) {
    std::ifstream labels_file(labelsFilePath, std::ios_base::in);
    std::string delim{';'};
    if (!labels_file) {
        InferError("Could not open labels file:%s", safeStr(labelsFilePath));
        return NVDSINFER_CONFIG_FAILED;
    }
    while (labels_file.good() && !labels_file.eof()) {
        std::string line, word;
        std::vector<std::string> l;
        size_t pos = 0, oldpos = 0;

        std::getline(labels_file, line, '\n');
        if (line.empty())
            continue;

        while ((pos = line.find(delim, oldpos)) != std::string::npos) {
            word = line.substr(oldpos, pos - oldpos);
            l.push_back(word);
            oldpos = pos + delim.length();
        }
        l.push_back(line.substr(oldpos));
        m_Labels.push_back(l);
    }

    if (labels_file.bad()) {
        InferError("Failed to parse labels file:%s, iostate:%d",
            safeStr(labelsFilePath), (int)labels_file.rdstate());
        return NVDSINFER_CONFIG_FAILED;
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus Postprocessor::allocateResource(
    const std::vector<int>& devIds) {
    if (!m_LabelPath.empty()) {
        RETURN_NVINFER_ERROR(parseLabelsFile(m_LabelPath),
            "parse label file:%s failed", safeStr(m_LabelPath));
    }
    return NVDSINFER_SUCCESS;
}

SharedBatchArray Postprocessor::requestCudaOutBufs(
    const SharedBatchArray& inBufs) {
    assert(inBufs);
    SharedBatchArray outBufs = std::make_shared<BaseBatchArray>();

    bool needCuEvent = false;
    for (const SharedBatchBuf &in : inBufs->bufs()) {
        auto const & inDesc = in->getBufDesc();
        if (inDesc.isInput && !needInputCopy()) {
            continue;
        }
        if (isCpuMem(inDesc.memType)) {
            outBufs->mutableBufs().emplace_back(in);
            continue;
        }
        if (!m_CpuAllocator) {
            InferError("failed to request cuda out buffers, allocator is not set");
            return nullptr;
        }
        needCuEvent = true;
        size_t expectBytes = in->getTotalBytes();
        SharedSysMem outMem = m_CpuAllocator(inDesc.name, expectBytes);
        RETURN_IF_FAILED(
            outMem, nullptr, "Postprocess allocate CPU mem failed on tensor: %s",
            safeStr(inDesc.name));
        assert(isCpuMem(outMem->type()) && outMem->ptr());
        assert(outMem->bytes() >= expectBytes);
        InferBufferDescription outDesc = inDesc;
        outDesc.memType = outMem->type();
        outDesc.devId = outMem->devId();
        SharedBatchBuf outBuf(
            new RefBatchBuffer(outMem->ptr(), 0, expectBytes, outDesc, in->getBatchSize()),
            [outMem](RefBatchBuffer* ptr) {
                assert(ptr);
                delete ptr;
            });
        if (!outBuf) {
            InferError(
                "failed to allocate host buffer:%s for post-cuda-process", safeStr(outDesc.name));
            return nullptr;
        }
        outBuf->attach(in);
        outBufs->mutableBufs().emplace_back(std::move(outBuf));
    }
    assert(outBufs->getSize() > 0);

    if (needCuEvent) {
        assert(m_EventAllocator);
        SharedCuEvent event = m_EventAllocator();
        assert(event);
        if (!event) {
            InferError("failed to allocate cuda event but continue post-cuda-process");
            return nullptr;
        }
        outBufs->setCuEvent(std::move(event));
    }

    return outBufs;
}

SharedBatchArray Postprocessor::requestHostOutBufs(const SharedBatchArray& inBuf)
{
    return std::make_shared<BaseBatchArray>();
}

NvDsInferStatus Postprocessor::postCudaImpl(SharedBatchArray& inBufs,
    SharedBatchArray& outBufs, SharedCuStream& mainStream) {
    INFER_UNUSED(inBufs);
    assert(inBufs && inBufs->getSize() > 0);
    assert(outBufs && outBufs->getSize() > 0 && outBufs->getSize() <= inBufs->getSize());
    assert(mainStream);

    int gpuId = outBufs->findFirstGpuId();
    if (gpuId >= 0) {
        RETURN_CUDA_ERR(
            cudaSetDevice(gpuId), "post host failed to set cuda device(%d)",
            gpuId);
    }

    if (inBufs->cuEvent()) {
        RETURN_CUDA_ERR(
            cudaStreamWaitEvent(*mainStream, *inBufs->cuEvent(), 0),
            "Failed to make stream wait on event during post cuda process");
    }

    InferDebug("Postprocessors id:%d post cuda process", uniqueId());

    for (auto &buf : outBufs->mutableBufs()) {
        assert(buf);
        const InferBufferDescription& bufDesc = buf->getBufDesc();
        assert(isCpuMem(bufDesc.memType));

        if (!buf->hasAttachedBufs()) {
            continue;
        }
        assert(buf->attachedBufs().size() == 1);
        SharedBatchBuf inBuf = buf->attachedBufs().at(0);
        assert(inBuf);
        const InferBufferDescription& inDesc = inBuf->getBufDesc();
        assert(inDesc.dataType == bufDesc.dataType);
        assert(inDesc.dims.numElements == bufDesc.dims.numElements);
        assert(inDesc.memType == InferMemType::kGpuCuda);
        assert(getElementSize(bufDesc.dataType) == bufDesc.elementSize);
        uint32_t totalEleNum =
            dimsSize(fullDims(inBuf->getBatchSize(), bufDesc.dims));
        assert(totalEleNum > 0 && totalEleNum < INT32_MAX);
        if (inDesc.memType == InferMemType::kGpuCuda) {
            void *inBufPtr = inBuf->getBufPtr(0);
            void *outBufPtr = buf->getBufPtr(0);
            RETURN_CUDA_ERR(
                cudaMemcpyAsync(
                    outBufPtr, inBufPtr, bufDesc.elementSize * totalEleNum,
                    cudaMemcpyDeviceToHost, *mainStream),
                "failed to postprocessing cudaMemcpyAsync for in/out buffer: "
                "%s",
                safeStr(bufDesc.name));
        }
    }

    if (outBufs->cuEvent()) {
        RETURN_CUDA_ERR(cudaEventRecord(*outBufs->cuEvent(), *mainStream),
            "Failed to record batch cuda copy-complete-event");
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus Postprocessor::postHostImpl(SharedBatchArray& inBufs,
    SharedBatchArray& outBufs, SharedCuStream& mainStream) {

    assert(inBufs && inBufs->getSize() > 0);
    assert(outBufs && outBufs->getSize() == 0);
    int gpuId = inBufs->findFirstGpuId();

    InferDebug("Postprocessors id:%d post host process", uniqueId());

    if (gpuId >= 0) {
        RETURN_CUDA_ERR(
            cudaSetDevice(gpuId), "post host failed to set cuda device(%d)",
            gpuId);
    }

    if (inBufs->cuEvent()) {
        RETURN_CUDA_ERR(cudaEventSynchronize(*inBufs->cuEvent()),
            "Failed to synchronize on final post host event");
    }
    inBufs->setCuEvent(nullptr);

    auto const &inBufList = inBufs->bufs();
    std::vector<NvDsInferLayerInfo> cOutputLayers;
    std::vector<SharedBatchBuf> cOutputTensors;
    uint32_t batchSize = 1;
    for (const SharedBatchBuf &buf : inBufList) {
        const InferBufferDescription &bufDesc = buf->getBufDesc();
        if (isPrivateTensor(bufDesc.name)) {
            outBufs->addBuf(buf);
            continue;
        }
        // release backend's gpu buffers
        buf->detach();
        if (!bufDesc.isInput) {
            cOutputLayers.emplace_back(
                toCapiLayerInfo(bufDesc, buf->getBufPtr(0)));
            cOutputTensors.emplace_back(buf);
        }
        if (bufDesc.isInput && !needInputCopy()) {
            continue;
        }
        if (!buf->getBatchSize()) {  // All output have batchSize > 0
            buf->setBatchSize(1);
        } else {
            batchSize = std::max(buf->getBatchSize(), batchSize);
        }
        outBufs->addBuf(buf);
    }

    if (!cOutputLayers.empty()) {
        assert(batchSize > 0);
        RETURN_NVINFER_ERROR(
            batchParse(cOutputLayers, cOutputTensors, batchSize, outBufs),
            "Infer context initialize inference info failed");
    }

    return NVDSINFER_SUCCESS;
}

DetectPostprocessor::DetectPostprocessor(int uid, const ic::DetectionParams &params)
    : Postprocessor(InferPostprocessType::kDetector, uid)
{
    m_DetectConfig.Clear();
    m_DetectConfig.CopyFrom(params);
    m_NumDetectedClasses = params.num_detected_classes();
}

NvDsInferStatus DetectPostprocessor::allocateResource(
    const std::vector<int>& devIds) {
    if (m_DetectConfig.num_detected_classes() <= 0) {
        InferError("Config DetectionParams.num_detected_classes is not set");
        return NVDSINFER_CONFIG_FAILED;
    }

    NvDsInferDetectionParams defaultParams{0};
    if (m_DetectConfig.has_dbscan()) {
        const ic::DetectionParams::DbScan &dbParam = m_DetectConfig.dbscan();
        defaultParams.threshold = dbParam.pre_threshold();
        //defaultParams.postThreshold = dbParam.post_threshold();
        defaultParams.minBoxes = dbParam.min_boxes();
        defaultParams.minScore = dbParam.min_score();
        defaultParams.eps = dbParam.eps();
#ifdef WITH_OPENCV
    } else if (m_DetectConfig.has_group_rectangle()) {
        const ic::DetectionParams::GroupRectangle &gpParams = m_DetectConfig.group_rectangle();
        defaultParams.threshold = gpParams.confidence_threshold();
        defaultParams.groupThreshold = gpParams.group_threshold();
        defaultParams.eps = gpParams.eps();
#endif
    } else if (m_DetectConfig.has_simple_cluster()) {
        const ic::DetectionParams::SimpleCluster& simpleParam =
            m_DetectConfig.simple_cluster();
        defaultParams.threshold = simpleParam.threshold();
    } else if (m_DetectConfig.has_nms()) {
        const ic::DetectionParams::Nms& nmsParam = m_DetectConfig.nms();
        defaultParams.threshold = nmsParam.confidence_threshold();
        defaultParams.nmsIOUThreshold = nmsParam.iou_threshold();
    }

    m_PerClassDetectionParams.resize(m_NumDetectedClasses, defaultParams);
    for (const auto& perClass: m_DetectConfig.per_class_params()) {
        if (perClass.first < 0 || perClass.first >= (int)m_NumDetectedClasses) {
            InferError("Config DetectionParams.per_class_pre_threshold is larger than num_detected_classes");
            return NVDSINFER_CONFIG_FAILED;
        }
        auto &perC = m_PerClassDetectionParams[perClass.first];
        perC.threshold = perClass.second.pre_threshold();
    }

    /* Fill the class thresholds in the m_DetectionParams structure. This
     * will be required during parsing. */
    m_DetectionParams.numClassesConfigured = m_NumDetectedClasses;
    m_DetectionParams.perClassPreclusterThreshold.resize(m_NumDetectedClasses);
    for (uint32_t i = 0; i < m_NumDetectedClasses; i++) {
        m_DetectionParams.perClassPreclusterThreshold[i] =
            m_PerClassDetectionParams[i].threshold;
    }

    /* If custom parse function is specified get the function address from the
     * custom library. */
    if (!m_DetectConfig.custom_parse_bbox_func().empty()) {
        if(!m_CustomLibHandle) {
            InferError("detetor custom func:%s defined but there's no custom_lib specified",
                safeStr(m_DetectConfig.custom_parse_bbox_func()));
            return NVDSINFER_CONFIG_FAILED;
        }
        m_CustomBBoxParseFunc =
            m_CustomLibHandle->symbol<NvDsInferParseCustomFunc>(
                m_DetectConfig.custom_parse_bbox_func());
        if (!m_CustomBBoxParseFunc) {
            InferError(
                "Detect-postprocessor failed to init resource "
                "because dlsym failed to get func %s pointer",
                safeStr(m_DetectConfig.custom_parse_bbox_func()));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }

    if (m_DetectConfig.has_dbscan()) {
        m_DBScanHandle.reset(
            NvDsInferDBScanCreate(), &NvDsInferDBScanDestroy);
        if (!m_DBScanHandle) {
            InferError("Detect-postprocessor failed to create dbscan handle");
            return NVDSINFER_RESOURCE_ERROR;
        }
    }

    RETURN_NVINFER_ERROR(Postprocessor::allocateResource(devIds),
        "failed to allocate detection processing resource");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
DetectPostprocessor::batchParse(
    std::vector<NvDsInferLayerInfo>& outputLayers,
    const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
    SharedBatchArray& results)
{
    std::vector<std::vector<NvDsInferObject>> detectionOutput(batchSize);
    for (uint32_t i = 0; i < batchSize; ++i) {
        for (size_t kL = 0; kL < outputLayers.size(); ++kL) {
            outputLayers[kL].buffer = outputBufs[kL]->getBufPtr(i);
        }
        RETURN_NVINFER_ERROR(
            fillDetectionOutput(outputLayers, detectionOutput[i]),
            "detection parsing output tensor data failed, uid:%d", uniqueId());
    }
    auto output = std::make_shared<DetectionOutput>();
    output->swapObjects(detectionOutput);
    results->addBuf(std::move(output));
    return NVDSINFER_SUCCESS;
}

ClassifyPostprocessor::ClassifyPostprocessor(int uid,
    const ic::ClassificationParams &params)
    : Postprocessor(InferPostprocessType::kClassifier, uid)
{
    m_Config.CopyFrom(params);
    m_ClassifierThreshold = params.threshold();
}

NvDsInferStatus ClassifyPostprocessor::allocateResource(
    const std::vector<int>& devIds) {
    if (!m_Config.custom_parse_classifier_func().empty()) {
        if(!m_CustomLibHandle) {
            InferError("classifier custom func:%s defined but there's no custom_lib specified",
                safeStr(m_Config.custom_parse_classifier_func()));
            return NVDSINFER_CONFIG_FAILED;
        }
        m_CustomClassifierParseFunc =
            m_CustomLibHandle->symbol<NvDsInferClassiferParseCustomFunc>(
            m_Config.custom_parse_classifier_func());
        if (!m_CustomClassifierParseFunc) {
            InferError(
                "Failed to init classify-postprocessor"
                "because dlsym failed to get func %s pointer",
                safeStr(m_Config.custom_parse_classifier_func()));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }

    /* read labels path */
    RETURN_NVINFER_ERROR(Postprocessor::allocateResource(devIds),
        "failed to allocate classify processing resource");

    /* Merge all multi-layer's lables into single layer labels */
    if (m_OutputLayerCount == 1 && m_Labels.size() > 1) {
        std::vector<std::string> totalLables;
        totalLables.reserve(m_Labels.size());
        for (const auto& perLayerLabel : m_Labels) {
            totalLables.insert(
                totalLables.end(), perLayerLabel.begin(), perLayerLabel.end());
        }
        m_Labels.clear();
        m_Labels.emplace_back(std::move(totalLables));
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
ClassifyPostprocessor::batchParse(
    std::vector<NvDsInferLayerInfo>& outputLayers,
    const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
    SharedBatchArray& results)
{
    auto output = std::make_shared<ClassificationOutput>(batchSize);
    assert(output);
    for (uint32_t i = 0; i < batchSize; ++i) {
        for (size_t kL = 0; kL < outputLayers.size(); ++kL) {
            outputLayers[kL].buffer = outputBufs[kL]->getBufPtr(i);
        }
        InferClassificationOutput& ret = output->mutableOutput(i);
        RETURN_NVINFER_ERROR(
            fillClassificationOutput(outputLayers, ret),
            "classification parsing output tensor data failed, uid:%d", uniqueId());
    }
    output->finalize();
    results->addBuf(std::move(output));
    return NVDSINFER_SUCCESS;
}

SegmentPostprocessor::SegmentPostprocessor(int uid,
    const ic::SegmentationParams &params)
    : Postprocessor(InferPostprocessType::kSegmentation, uid)
{
    m_Config.CopyFrom(params);
    m_SegmentationThreshold = params.threshold();
    m_NumSegmentationClasses = params.num_segmentation_classes();
}

NvDsInferStatus SegmentPostprocessor::allocateResource(
    const std::vector<int>& devIds) {
    if (!m_Config.custom_parse_segmentation_func().empty()) {
        if(!m_CustomLibHandle) {
            InferError("Semantic segmentation custom func:%s defined but"
                       " no custom_lib specified",
                safeStr(m_Config.custom_parse_segmentation_func()));
            return NVDSINFER_CONFIG_FAILED;
        }
        m_CustomSemSegmentationParseFunc =
            m_CustomLibHandle->symbol<NvDsInferSemSegmentationParseCustomFunc>(
            m_Config.custom_parse_segmentation_func());
        if (!m_CustomSemSegmentationParseFunc) {
            InferError(
                "Failed to init sememantic segmentation postprocessor"
                "because dlsym failed to get func %s pointer",
                safeStr(m_Config.custom_parse_segmentation_func()));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }
    RETURN_NVINFER_ERROR(Postprocessor::allocateResource(devIds),
        "failed to allocate segmentation processing resource");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
SegmentPostprocessor::batchParse(
    std::vector<NvDsInferLayerInfo>& outputLayers,
    const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
    SharedBatchArray& results)
{
    auto output = std::make_shared<SegmentationOutput>(batchSize);
    assert(output);
    for (uint32_t i = 0; i < batchSize; ++i) {
        for (size_t kL = 0; kL < outputLayers.size(); ++kL) {
            outputLayers[kL].buffer = outputBufs[kL]->getBufPtr(i);
        }
        NvDsInferSegmentationOutput &ret = output->mutableOutput(i);
        RETURN_NVINFER_ERROR(
            fillSegmentationOutput(outputLayers, ret),
            "segmentation parsing output tensor data failed, uid:%d", uniqueId());
    }
    for (const SharedBatchBuf& buf : results->bufs()) {
        const InferBufferDescription & desc = buf->getBufDesc();
        if (!desc.isInput && isCpuMem(desc.memType)) {
            output->attach(buf);
        }
    }
    results->addBuf(std::move(output));
    return NVDSINFER_SUCCESS;
}

OtherPostprocessor::OtherPostprocessor(int uid,
    const ic::OtherNetworkParams &params)
    : Postprocessor(InferPostprocessType::kOther, uid)
{
    m_Config.CopyFrom(params);
}

TrtIsClassifier::TrtIsClassifier(int uid, const ic::TritonClassifyParams& params)
    : Postprocessor(InferPostprocessType::kTrtIsClassifier, uid)
{
    m_Config.CopyFrom(params);
}

NvDsInferStatus
TrtIsClassifier::postHostImpl(
    SharedBatchArray& inBuf, SharedBatchArray& outBuf,
    SharedCuStream& mainStream)
{
    RETURN_NVINFER_ERROR(
        Postprocessor::postHostImpl(inBuf, outBuf, mainStream),
        "TrtIsClassifier post host processing failed, uid:%d", uniqueId());
    assert(outBuf && outBuf->getSize());

    SharedBatchBuf classBuf;
    for (const SharedBatchBuf& buf : outBuf->bufs()) {
        assert(buf);
        const InferBufferDescription& desc = buf->getBufDesc();
        if (desc.name == INFER_SERVER_CLASSIFICATION_BUF_NAME) {
            classBuf = buf;
            assert(desc.elementSize == sizeof(InferClassificationOutput));
            break;
        }
    }
    if (!classBuf) {
        return NVDSINFER_SUCCESS;
    }
    // TODO, filter out more data according to ic::TritonClassifyParams

    return NVDSINFER_SUCCESS;
}

void
DetectPostprocessor::copyWithoutCluster(
    const std::vector<NvDsInferObjectDetectionInfo>& objectList,
    std::vector<NvDsInferObject>& outputs)
{
    outputs.resize(objectList.size());
    for (size_t i = 0; i < objectList.size(); ++i) {
        const NvDsInferObjectDetectionInfo& from = objectList[i];
        NvDsInferObject& object = outputs[i];
        object.left = from.left;
        object.top = from.top;
        object.width = from.width;
        object.height = from.height;
        object.classIndex = from.classId;
        object.confidence = from.detectionConfidence;
        object.label = nullptr;
        if (from.classId < m_Labels.size() &&
            m_Labels[from.classId].size() > 0) {
            object.label = strdup(m_Labels[from.classId][0].c_str());
        }
    }
}

bool DetectPostprocessor::parseBoundingBox(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
    int outputCoverageLayerIndex = -1;
    int outputBBoxLayerIndex = -1;

    for (auto const& layer : outputLayersInfo) {
        if (layer.dataType != FLOAT) {
            InferError(
                "Default bbox parsing function support datatype"
                "FP32 only but received output tensor: %s with datatype: %s",
                safeStr(layer.layerName),
                safeStr(
                    dataType2Str(static_cast<InferDataType>(layer.dataType))));
            return false;
        }
    }

    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strstr(outputLayersInfo[i].layerName, "bbox") != nullptr) {
            outputBBoxLayerIndex = i;
        }
        if (strstr(outputLayersInfo[i].layerName, "cov") != nullptr) {
            outputCoverageLayerIndex = i;
        }
    }

    if (outputCoverageLayerIndex == -1) {
        InferError("Could not find output coverage layer for parsing objects");
        return false;
    }
    if (outputBBoxLayerIndex == -1) {
        InferError("Could not find output bbox layer for parsing objects");
        return false;
    }

    float* outputCoverageBuffer =
        (float*)outputLayersInfo[outputCoverageLayerIndex].buffer;
    float* outputBboxBuffer =
        (float*)outputLayersInfo[outputBBoxLayerIndex].buffer;

    NvDsInferDimsCHW outputCoverageDims;
    NvDsInferDimsCHW outputBBoxDims;

    getDimsCHWFromDims(outputCoverageDims,
        outputLayersInfo[outputCoverageLayerIndex].inferDims);
    getDimsCHWFromDims(
        outputBBoxDims, outputLayersInfo[outputBBoxLayerIndex].inferDims);

    unsigned int targetShape[2] = {outputCoverageDims.w, outputCoverageDims.h};
    float bboxNorm[2] = {35.0, 35.0};
    float gcCenters0[targetShape[0]];
    float gcCenters1[targetShape[1]];
    int gridSize = outputCoverageDims.w * outputCoverageDims.h;
    int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, outputBBoxDims.w);
    int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, outputBBoxDims.h);

    for (unsigned int i = 0; i < targetShape[0]; i++) {
        gcCenters0[i] = (float)(i * strideX + 0.5);
        gcCenters0[i] /= (float)bboxNorm[0];
    }
    for (unsigned int i = 0; i < targetShape[1]; i++) {
        gcCenters1[i] = (float)(i * strideY + 0.5);
        gcCenters1[i] /= (float)bboxNorm[1];
    }

    unsigned int numClasses =
        MIN(outputCoverageDims.c, detectionParams.numClassesConfigured);
    for (unsigned int classIndex = 0; classIndex < numClasses; classIndex++) {
        /* Pointers to memory regions containing the (x1,y1) and (x2,y2)
         * coordinates of rectangles in the output bounding box layer. */
        float* outputX1 = outputBboxBuffer + classIndex * sizeof(float) *
                                                 outputBBoxDims.h *
                                                 outputBBoxDims.w;

        float* outputY1 = outputX1 + gridSize;
        float* outputX2 = outputY1 + gridSize;
        float* outputY2 = outputX2 + gridSize;

        /* Iterate through each point in the grid and check if the rectangle at
         * that point meets the minimum threshold criteria. */
        for (unsigned int h = 0; h < outputCoverageDims.h; h++) {
            for (unsigned int w = 0; w < outputCoverageDims.w; w++) {
                int i = w + h * outputCoverageDims.w;
                float confidence =
                    outputCoverageBuffer[classIndex * gridSize + i];

                if (confidence < detectionParams.perClassPreclusterThreshold[classIndex])
                    continue;

                float rectX1Float, rectY1Float, rectX2Float, rectY2Float;

                /* Centering and normalization of the rectangle. */
                rectX1Float =
                    outputX1[w + h * outputCoverageDims.w] - gcCenters0[w];
                rectY1Float =
                    outputY1[w + h * outputCoverageDims.w] - gcCenters1[h];
                rectX2Float =
                    outputX2[w + h * outputCoverageDims.w] + gcCenters0[w];
                rectY2Float =
                    outputY2[w + h * outputCoverageDims.w] + gcCenters1[h];

                rectX1Float *= -bboxNorm[0];
                rectY1Float *= -bboxNorm[1];
                rectX2Float *= bboxNorm[0];
                rectY2Float *= bboxNorm[1];

                /* Clip parsed rectangles to frame bounds. */
                if (rectX1Float >= (int)m_NetworkInfo.width)
                    rectX1Float = m_NetworkInfo.width - 1;
                if (rectX2Float >= (int)m_NetworkInfo.width)
                    rectX2Float = m_NetworkInfo.width - 1;
                if (rectY1Float >= (int)m_NetworkInfo.height)
                    rectY1Float = m_NetworkInfo.height - 1;
                if (rectY2Float >= (int)m_NetworkInfo.height)
                    rectY2Float = m_NetworkInfo.height - 1;

                if (rectX1Float < 0)
                    rectX1Float = 0;
                if (rectX2Float < 0)
                    rectX2Float = 0;
                if (rectY1Float < 0)
                    rectY1Float = 0;
                if (rectY2Float < 0)
                    rectY2Float = 0;

                //Prevent underflows
                if(((rectX2Float - rectX1Float) < 0) ||
                        ((rectY2Float - rectY1Float) < 0))
                    continue;

                objectList.push_back({classIndex, rectX1Float,
                    rectY1Float, (rectX2Float - rectX1Float),
                    (rectY2Float - rectY1Float), confidence});
            }
        }
    }
    return true;
}

#ifdef WITH_OPENCV
/**
 * Cluster objects using OpenCV groupRectangles and fill the output structure.
 */
void DetectPostprocessor::clusterAndFillDetectionOutputCV(
    const std::vector<NvDsInferObjectDetectionInfo>& objectList,
    std::vector<NvDsInferObject> &outputs) {
    size_t totalObjects = 0;

    std::vector<std::vector<cv::Rect>> perClassCvRectList(m_NumDetectedClasses);

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (auto& object : objectList) {
        perClassCvRectList[object.classId].emplace_back(
            object.left, object.top, object.width, object.height);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object. Refer
         * to opencv documentation of groupRectangles for more
         * information about the tuning parameters for grouping. */
        if (m_PerClassDetectionParams[c].groupThreshold > 0)
            cv::groupRectangles(perClassCvRectList[c],
                m_PerClassDetectionParams[c].groupThreshold,
                m_PerClassDetectionParams[c].eps);
        totalObjects += perClassCvRectList[c].size();
    }

    outputs.resize(totalObjects);
    int numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (auto& rect : perClassCvRectList[c]) {
            NvDsInferObject& object = outputs[numObjects];
            object.left = rect.x;
            object.top = rect.y;
            object.width = rect.width;
            object.height = rect.height;
            object.classIndex = c;
            object.label = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            object.confidence = -0.1;
            numObjects++;
        }
    }
}
#endif

/**
 * Cluster objects using DBSCAN and fill the output structure.
 */
void DetectPostprocessor::clusterAndFillDetectionOutputDBSCAN(
    const std::vector<NvDsInferObjectDetectionInfo>& objectList,
    std::vector<NvDsInferObject>& outputs) {
    size_t totalObjects = 0;
    NvDsInferDBScanClusteringParams clusteringParams;
    clusteringParams.enableATHRFilter = ATHR_ENABLED;
    clusteringParams.thresholdATHR = ATHR_THRESHOLD;
    assert(m_NumDetectedClasses);
    std::vector<size_t> numObjectsList(m_NumDetectedClasses);
    std::vector<std::vector<NvDsInferObjectDetectionInfo>> perClassObjectList(m_NumDetectedClasses);

    assert(m_DBScanHandle);

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (const auto& object : objectList) {
        perClassObjectList[object.classId].emplace_back(object);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        NvDsInferObjectDetectionInfo* objArray = perClassObjectList[c].data();
        size_t numObjects = perClassObjectList[c].size();

        clusteringParams.eps = m_PerClassDetectionParams[c].eps;
        clusteringParams.minBoxes = m_PerClassDetectionParams[c].minBoxes;
        clusteringParams.minScore = m_PerClassDetectionParams[c].minScore;

        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object using
         * DBSCAN. */
        if (m_PerClassDetectionParams[c].minBoxes > 0) {
            NvDsInferDBScanCluster(
                m_DBScanHandle.get(), &clusteringParams, objArray, &numObjects);
        }
        totalObjects += numObjects;
        numObjectsList[c] = numObjects;
    }

    outputs.resize(totalObjects);
    int numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (size_t i = 0; i < numObjectsList[c]; i++) {
            NvDsInferObject& object = outputs[numObjects];
            object.left = perClassObjectList[c][i].left;
            object.top = perClassObjectList[c][i].top;
            object.width = perClassObjectList[c][i].width;
            object.height = perClassObjectList[c][i].height;
            object.classIndex = c;
            object.label = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            object.confidence = perClassObjectList[c][i].detectionConfidence;
            numObjects++;
        }
    }
    outputs.resize(numObjects);
}

std::vector<int>
DetectPostprocessor::nonMaximumSuppression(
    const std::vector<std::pair<float, int>>& scoreIndex,
    const std::vector<NvDsInferParseObjectInfo>& bbox, const float nmsThreshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min,
                        float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](
                          const NvDsInferParseObjectInfo& bbox1,
                          const NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX = overlap1D(
            bbox1.left, bbox1.left + bbox1.width, bbox2.left,
            bbox2.left + bbox2.width);
        float overlapY = overlap1D(
            bbox1.top, bbox1.top + bbox1.height, bbox2.top,
            bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto& i : scoreIndex) {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k) {
            if (keep) {
                const int kept_idx = indices[k];
                float overlap = computeIoU(bbox.at(idx), bbox.at(kept_idx));
                keep = overlap <= nmsThreshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices.push_back(idx);
        }
    }
    return indices;
}

/** Cluster objects using Non Max Suppression */
void
DetectPostprocessor::clusterAndFillDetectionOutputNMS(
    const std::vector<NvDsInferObjectDetectionInfo>& objectList, uint32_t topk,
    std::vector<NvDsInferObject>& outputs)
{
    std::vector<NvDsInferObjectDetectionInfo> clusteredBboxes;

    std::vector<std::vector<NvDsInferObjectDetectionInfo>> perClassObjectList(
        m_NumDetectedClasses);

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (auto& object : objectList) {
        perClassObjectList[object.classId].emplace_back(object);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        if (!perClassObjectList.at(c).empty()) {
            const auto& classObjects = perClassObjectList.at(c);
            std::vector<std::pair<float, int>> scoreIndex;
            for (size_t r = 0; r < classObjects.size(); ++r) {
                scoreIndex.push_back(
                    std::make_pair(classObjects.at(r).detectionConfidence, r));
            }
            std::stable_sort(
                scoreIndex.begin(), scoreIndex.end(),
                [](const std::pair<float, int>& pair1,
                    const std::pair<float, int>& pair2) {
                    return pair1.first > pair2.first;
                });

            // Apply NMS algorithm
            const std::vector<int> indices = nonMaximumSuppression(
                scoreIndex, classObjects,
                m_PerClassDetectionParams.at(c).nmsIOUThreshold);
            for (auto idx : indices) {
                clusteredBboxes.push_back(classObjects.at(idx));
            }
        }
    }

    if (topk && topk < clusteredBboxes.size()) {
        std::nth_element(
            clusteredBboxes.begin(), clusteredBboxes.begin() + topk,
            clusteredBboxes.end(),
            [](const NvDsInferObjectDetectionInfo& a,
               const NvDsInferObjectDetectionInfo& b) {
                return a.detectionConfidence > b.detectionConfidence;
            });
        clusteredBboxes.erase(
            clusteredBboxes.begin() + topk, clusteredBboxes.end());
        assert(clusteredBboxes.size() == topk);
    }

    outputs.resize(clusteredBboxes.size());

    for (size_t i = 0; i < clusteredBboxes.size(); ++i) {
        NvDsInferObject& object = outputs.at(i);
        object.left = clusteredBboxes.at(i).left;
        object.top = clusteredBboxes.at(i).top;
        object.width = clusteredBboxes.at(i).width;
        object.height = clusteredBboxes.at(i).height;
        object.classIndex = clusteredBboxes.at(i).classId;
        object.label = nullptr;
        if (object.classIndex < static_cast<int>(m_Labels.size()) &&
            m_Labels[object.classIndex].size() > 0)
            object.label = strdup(m_Labels[object.classIndex][0].c_str());
        object.confidence = clusteredBboxes.at(i).detectionConfidence;
    }
}

bool ClassifyPostprocessor::parseAttributesFromSoftmaxLayers(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList, std::string& attrString) {
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = outputLayersInfo.size();

    for (auto const& layer : outputLayersInfo) {
        if (layer.dataType != FLOAT) {
            InferError(
                "Default classification parsing function support datatype"
                "FP32 only but received output tensor: %s with datatype: %s",
                safeStr(layer.layerName),
                safeStr(
                    dataType2Str(static_cast<InferDataType>(layer.dataType))));
            return false;
        }
    }

    /* Iterate through all the output coverage layers of the classifier.
     */
    for (unsigned int l = 0; l < numAttributes; l++) {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, outputLayersInfo[l].inferDims);
        unsigned int numClasses = dims.c;
        float* outputCoverageBuffer = (float*)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++) {
            float probability = outputCoverageBuffer[c];
            if (probability > m_ClassifierThreshold &&
                probability > maxProbability) {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }
        }
        if (attrFound) {
            if (m_Labels.size() > attr.attributeIndex &&
                attr.attributeValue < m_Labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    strdup(m_Labels[attr.attributeIndex][attr.attributeValue].c_str());
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                attrString.append(attr.attributeLabel).append(" ");
        }
    }

    return true;
}

/**
 * Filter out objects which have been specificed to be removed from the metadata
 */
void
DetectPostprocessor::filterDetectionOutput(
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    objectList.erase(
        std::remove_if(
            objectList.begin(), objectList.end(),
            [detectionParams](const NvDsInferObjectDetectionInfo& obj) {
                return (obj.classId >= detectionParams.numClassesConfigured) ||
                       (obj.detectionConfidence <
                                detectionParams.perClassPreclusterThreshold[obj.classId]
                            ? true
                            : false);
            }),
        objectList.end());
}

NvDsInferStatus DetectPostprocessor::fillDetectionOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    std::vector<NvDsInferObject> &output) {
    /* Clear the object lists. */
    std::vector<NvDsInferObjectDetectionInfo> objectList;

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomBBoxParseFunc) {
        if (!m_CustomBBoxParseFunc(
                outputLayers, m_NetworkInfo, m_DetectionParams, objectList)) {
            InferError("Failed to parse bboxes using custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        if (!parseBoundingBox(
                outputLayers, m_NetworkInfo, m_DetectionParams, objectList)) {
            InferError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    filterDetectionOutput(m_DetectionParams, objectList);

    if (m_DetectConfig.has_dbscan()) {
        clusterAndFillDetectionOutputDBSCAN(objectList, output);
#ifdef WITH_OPENCV
    } else if (m_DetectConfig.has_group_rectangle()) {
        clusterAndFillDetectionOutputCV(objectList, output);
#endif
    } else if (m_DetectConfig.has_nms()) {
        uint32_t topk = (uint32_t)(
            m_DetectConfig.nms().topk() > 0 ? m_DetectConfig.nms().topk() : 0);
        clusterAndFillDetectionOutputNMS(objectList, topk, output);
    } else {
        copyWithoutCluster(objectList, output);
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
ClassifyPostprocessor::fillClassificationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    InferClassificationOutput& output)
{
    std::string attrString;
    std::vector<NvDsInferAttribute> attributes;

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomClassifierParseFunc) {
        if (!m_CustomClassifierParseFunc(outputLayers, m_NetworkInfo,
                m_ClassifierThreshold, attributes, attrString)) {
            InferError(
                "Failed to parse classification attributes using "
                "custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        if (!parseAttributesFromSoftmaxLayers(outputLayers, m_NetworkInfo,
                m_ClassifierThreshold, attributes, attrString)) {
            InferError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }
    for (auto& attr : attributes) {
        InferAttribute safeAttr;
        static_cast<NvDsInferAttribute&>(safeAttr) = attr;
        safeAttr.safeAttributeLabel = safeStr(attr.attributeLabel);
        safeAttr.attributeLabel = (char *) safeAttr.safeAttributeLabel.c_str();
        output.attributes.emplace_back(safeAttr);
        free(attr.attributeLabel);
    }

    output.label = attrString;
    return NVDSINFER_SUCCESS;
}

bool SegmentPostprocessor::parseSemanticSegmentationOutput(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap) {
    assert(classificationMap);

    for (auto const& layer : outputLayersInfo) {
        if (layer.dataType != FLOAT) {
            InferError(
                "Default segment parsing function supports datatype"
                "FP32 only but received output tensor: %s with datatype: %s",
                safeStr(layer.layerName),
                safeStr(
                    dataType2Str(static_cast<InferDataType>(layer.dataType))));
            return false;
        }
    }

    NvDsInferDimsCHW outputDimsCHW;
    getDimsCHWFromDims(outputDimsCHW, outputLayersInfo[0].inferDims);

    if (numClasses != outputDimsCHW.c) {
        InferError(
                "Configured number for classes %u differs from"
                " number of channels in output %u",
                 numClasses, outputDimsCHW.c);
        return false;
    }

    classProbabilityMap = (float*)outputLayersInfo[0].buffer;

    for (unsigned int y = 0; y < networkInfo.height; y++) {
        for (unsigned int x = 0; x < networkInfo.width; x++) {
            float max_prob = -1;
            int& cls = classificationMap[y * networkInfo.width + x] = -1;
            for (unsigned int c = 0; c < numClasses; c++) {
                float prob = classProbabilityMap[c * networkInfo.width *
                                                      networkInfo.height +
                                                      y * networkInfo.width + x];
                if (prob > max_prob && prob > segmentationThreshold) {
                    cls = c;
                    max_prob = prob;
                }
            }
        }
    }
    return true;
}

NvDsInferStatus SegmentPostprocessor::fillSegmentationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferSegmentationOutput& output) {

    output.width = m_NetworkInfo.width;
    output.height = m_NetworkInfo.height;
    output.classes = m_NumSegmentationClasses;
    output.class_map = new int[output.width * output.height];
    output.class_probability_map = nullptr;


    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomSemSegmentationParseFunc) {
        if (!m_CustomSemSegmentationParseFunc(outputLayers, m_NetworkInfo,
                m_SegmentationThreshold, m_NumSegmentationClasses,
                output.class_map, output.class_probability_map)) {
            InferError(
                "Failed to parse semantic segmentation output using "
                "custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        if (!parseSemanticSegmentationOutput(outputLayers, m_NetworkInfo,
                m_SegmentationThreshold, m_NumSegmentationClasses,
                output.class_map, output.class_probability_map)) {
            InferError(
                "Failed to parse semantic segmentation output.");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver
