/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file infer_postprocess.h
 *
 * @brief Header file for the post processing on inference results.
 *
 * This file declares the classes that implement the various post processing
 * functionalities required after the inference.
 */

#ifndef __NVDSINFERSERVER_POST_PROCESS_H__
#define __NVDSINFERSERVER_POST_PROCESS_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include <cuda_runtime_api.h>

#include "infer_batch_buffer.h"
#include "infer_common.h"
#include "infer_cuda_utils.h"
#include "infer_datatypes.h"
#include "infer_ibackend.h"
#include "infer_iprocess.h"
#include "infer_post_datatypes.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsinferserver_config.pb.h"

namespace ic = nvdsinferserver::config;

struct NvDsInferDBScan;

namespace nvdsinferserver {

/**
 * @brief A generic post processor class.
 */
class Postprocessor : public BasePostprocessor {
public:
    using TensorAllocator = std::function<SharedSysMem(const std::string& name, size_t bytes)>;
    using EventAllocator = std::function<SharedCuEvent()>;

protected:
    Postprocessor(InferPostprocessType type, int id)
        : BasePostprocessor(type, id){}

public:
    virtual ~Postprocessor() = default;
    void setDllHandle(const SharedDllHandle& dlHandle) {
        m_CustomLibHandle = dlHandle;
    }
    void setLabelPath(const std::string& path) { m_LabelPath = path; }
    void setNetworkInfo(const NvDsInferNetworkInfo& info) {
        m_NetworkInfo = info;
    }
    void setOutputLayerCount(uint32_t num) { m_OutputLayerCount = num; }
    void setInputCopy(bool enable) { m_CopyInputToHostBuffers = enable; }
    const std::vector<std::vector<std::string>>& getLabels() const {
        return m_Labels;
    }
    void setAllocator(TensorAllocator cpuAlloc, EventAllocator event)
    {
        m_CpuAllocator = cpuAlloc;
        m_EventAllocator = event;
    }
    bool needInputCopy() const { return m_CopyInputToHostBuffers; }

    virtual NvDsInferStatus allocateResource(
        const std::vector<int>& devIds) override;

protected:
    NvDsInferStatus postCudaImpl(SharedBatchArray& inBuf,
        SharedBatchArray& outbuf, SharedCuStream& mainStream) override;

    NvDsInferStatus postHostImpl(SharedBatchArray& inBuf,
        SharedBatchArray& outbuf, SharedCuStream& mainStream) override;

    SharedBatchArray requestCudaOutBufs(const SharedBatchArray& inBuf) override;
    SharedBatchArray requestHostOutBufs(const SharedBatchArray& inBuf) override;

private:
    virtual NvDsInferStatus batchParse(
        std::vector<NvDsInferLayerInfo>& outputLayers,
        const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
        SharedBatchArray& results) = 0;

protected:
    NvDsInferStatus parseLabelsFile(const std::string& path);

private:
    DISABLE_CLASS_COPY(Postprocessor);

protected:
    /** Custom library implementation. */
    SharedDllHandle m_CustomLibHandle;
    bool m_CopyInputToHostBuffers = false;
    /** Network input information. */
    NvDsInferNetworkInfo m_NetworkInfo = {0};
    uint32_t m_OutputLayerCount = 0;
    std::string m_LabelPath;

    /** Holds the string labels for classes. */
    std::vector<std::vector<std::string>> m_Labels;

    TensorAllocator m_CpuAllocator;
    EventAllocator m_EventAllocator;
};

/**
 * @brief Post processor class for detection output.
 */
class DetectPostprocessor : public Postprocessor {
public:
    DetectPostprocessor(int uid, const ic::DetectionParams &params);
    ~DetectPostprocessor() override = default;

    NvDsInferStatus allocateResource(const std::vector<int>& devIds) override;

private:
    NvDsInferStatus batchParse(
        std::vector<NvDsInferLayerInfo>& outputLayers,
        const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
        SharedBatchArray& results) override;

    bool parseBoundingBox(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams,
        std::vector<NvDsInferObjectDetectionInfo>& objectList);

    void clusterAndFillDetectionOutputCV(
    const std::vector<NvDsInferObjectDetectionInfo>& objectList,
    std::vector<NvDsInferObject> &outputs);
    void clusterAndFillDetectionOutputDBSCAN(
        const std::vector<NvDsInferObjectDetectionInfo>& objectList,
        std::vector<NvDsInferObject>& outputs);
    void copyWithoutCluster(
        const std::vector<NvDsInferObjectDetectionInfo>& objectList,
        std::vector<NvDsInferObject>& outputs);
    NvDsInferStatus fillDetectionOutput(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        std::vector<NvDsInferObject> &output);
    void filterDetectionOutput(
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);
    void clusterAndFillDetectionOutputNMS(
        const std::vector<NvDsInferObjectDetectionInfo>& objectList,
        uint32_t topk, std::vector<NvDsInferObject>& outputs);
    std::vector<int> nonMaximumSuppression(
        const std::vector<std::pair<float, int>>& scoreIndex,
        const std::vector<NvDsInferParseObjectInfo>& bbox,
        const float nmsThreshold);

private:
    struct NvDsInferDetectionParams
    {
        /** Bounding box detection threshold. */
        float threshold;
        /** Epsilon to control merging of overlapping boxes. Refer to OpenCV
         * groupRectangles and DBSCAN documentation for more information on epsilon */
        float eps;
        /** Minimum boxes in a cluster to be considered an object during
         * grouping using DBSCAN. */
        int minBoxes;
        /** Minimum boxes in a cluster to be considered an object during
         * grouping using OpenCV groupRectangles. */
        int groupThreshold;
        /** Minimum score in a cluster for it to be considered as an object
         * during grouping, different clustering could algorithm use different
         * scores */
        float minScore;
        /** IOU threshold to be used with NMS mode of clustering. */
        float nmsIOUThreshold;
    };

private:
    std::shared_ptr<NvDsInferDBScan> m_DBScanHandle;
    /** Number of classes detected by the model. */
    uint32_t m_NumDetectedClasses = 0;

    /** Detection / grouping parameters. */
    std::vector<NvDsInferDetectionParams> m_PerClassDetectionParams;
    NvDsInferParseDetectionParams m_DetectionParams = {0, {}, {}};

    NvDsInferParseCustomFunc m_CustomBBoxParseFunc = nullptr;
    ic::DetectionParams m_DetectConfig;
};

/**
 * @brief Post processor class for classification output.
 */
class ClassifyPostprocessor : public Postprocessor {
public:
    ClassifyPostprocessor(int uid, const ic::ClassificationParams &params);
    NvDsInferStatus allocateResource(const std::vector<int>& devIds) override;

private:
    NvDsInferStatus batchParse(
        std::vector<NvDsInferLayerInfo>& outputLayers,
        const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
        SharedBatchArray& results) override;

    NvDsInferStatus fillClassificationOutput(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        InferClassificationOutput& output);

    bool parseAttributesFromSoftmaxLayers(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
        std::vector<NvDsInferAttribute>& attrList, std::string& attrString);

private:
    float m_ClassifierThreshold = 0;
    NvDsInferClassiferParseCustomFunc m_CustomClassifierParseFunc = nullptr;

    ic::ClassificationParams m_Config;
};

/**
 * @brief Post processor class for segmentation output.
 */
class SegmentPostprocessor : public Postprocessor {
public:
    SegmentPostprocessor(int uid, const ic::SegmentationParams &params);
    NvDsInferStatus allocateResource(const std::vector<int>& devIds) override;

private:
    NvDsInferStatus batchParse(
        std::vector<NvDsInferLayerInfo>& outputLayers,
        const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
        SharedBatchArray& results) override;

    NvDsInferStatus fillSegmentationOutput(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferSegmentationOutput& output);

    bool parseSemanticSegmentationOutput(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo,
        float segmentationThreshold, unsigned int numClasses,
        int* classificationMap, float*& classProbabilityMap);

private:
    float m_SegmentationThreshold = 0.0f;
    NvDsInferSemSegmentationParseCustomFunc
        m_CustomSemSegmentationParseFunc = nullptr;
    unsigned int m_NumSegmentationClasses = 0;
    ic::SegmentationParams m_Config;
};

/**
 * @brief Post processor class for tensor output for custom post processing.
 */
class OtherPostprocessor : public Postprocessor {
public:
    OtherPostprocessor(int uid, const ic::OtherNetworkParams &params);

private:
    NvDsInferStatus batchParse(
        std::vector<NvDsInferLayerInfo>& outputLayers,
        const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
        SharedBatchArray& results) override
    {
        return NVDSINFER_SUCCESS;
    }

private:
    ic::OtherNetworkParams m_Config;
};

/**
 * @brief Post processor class for Triton Classification option.
 */
class TrtIsClassifier : public Postprocessor {
public:
    TrtIsClassifier(int uid, const ic::TritonClassifyParams& params);

private:
    NvDsInferStatus postHostImpl(
        SharedBatchArray& inBuf, SharedBatchArray& outbuf,
        SharedCuStream& mainStream) override;
    NvDsInferStatus batchParse(
        std::vector<NvDsInferLayerInfo>& outputLayers,
        const std::vector<SharedBatchBuf> outputBufs, uint32_t batchSize,
        SharedBatchArray& results) override
    {
        /* should never reach */
        InferError(
            "TrtIsClassifer(uid:%d) should not reach here, check error",
            uniqueId());
        return NVDSINFER_UNKNOWN_ERROR;
    }

private:
    ic::TritonClassifyParams m_Config;
};

} // namespace nvdsinferserver

#endif
