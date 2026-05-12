/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSINFER_CONTEXT_IMPL_H__
#define __NVDSINFER_CONTEXT_IMPL_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <iostream>
#include <fstream>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#ifdef WITH_OPENCV
#include <opencv2/objdetect/objdetect.hpp>
#endif
#pragma GCC diagnostic pop

#include <nvdsinfer_context.h>
#include <nvdsinfer_custom_impl.h>
#include <nvdsinfer_utils.h>
#include <nvdsinfer_logger.h>

#include "nvdsinfer_backend.h"

namespace nvdsinfer {

using NvDsInferLoggingFunc =
    std::function<void(NvDsInferLogLevel, const char* msg)>;

/**
 * Holds information for one batch for processing.
 */
typedef struct
{
    std::vector<void*> m_DeviceBuffers;
    std::vector<std::unique_ptr<CudaHostBuffer>> m_HostBuffers;

    std::vector<std::unique_ptr<CudaDeviceBuffer>> m_OutputDeviceBuffers;

    unsigned int m_BatchSize = 0;
    std::unique_ptr<CudaEvent> m_OutputCopyDoneEvent = nullptr;
    bool m_BuffersWithContext = true;

} NvDsInferBatch;

/**
 * Provides pre-processing functionality like mean subtraction and normalization.
 */
class InferPreprocessor
{
public:
    InferPreprocessor(const NvDsInferNetworkInfo& info, NvDsInferFormat format,
        const NvDsInferBatchDimsLayerInfo& layerInfo, int id = 0);
    virtual ~InferPreprocessor() = default;

    void setLoggingFunc(const NvDsInferLoggingFunc& func)
    {
        m_LoggingFunc = func;
    }
    bool setScaleOffsets(float scale, const std::vector<float>& offsets = {});
    bool setMeanFile(const std::string& file);
    bool setInputOrder(const NvDsInferTensorOrder order);

    NvDsInferStatus allocateResource();
    NvDsInferStatus syncStream();

    NvDsInferStatus transform(NvDsInferContextBatchInput& batchInput,
        void* devBuf, CudaStream& mainStream, CudaEvent* waitingEvent);

private:
    NvDsInferStatus readMeanImageFile();
    DISABLE_CLASS_COPY(InferPreprocessor);

private:
    int m_UniqueID = 0;
    NvDsInferLoggingFunc m_LoggingFunc;

    NvDsInferNetworkInfo m_NetworkInfo = {0};
    /** Input format for the network. */
    NvDsInferFormat m_NetworkInputFormat = NvDsInferFormat_RGB;
    NvDsInferTensorOrder m_InputOrder = NvDsInferTensorOrder_kNCHW;
    NvDsInferBatchDimsLayerInfo m_NetworkInputLayer;
    float m_Scale = 1.0f;
    std::vector<float> m_ChannelMeans; // same as channels
    std::string m_MeanFile;

    std::unique_ptr<CudaStream> m_PreProcessStream;
    /* Cuda Event for synchronizing completion of pre-processing. */
    std::shared_ptr<CudaEvent> m_PreProcessCompleteEvent;
    std::unique_ptr<CudaDeviceBuffer> m_MeanDataBuffer;
};

/**
 * Base class for post-processing on inference output.
 */
class InferPostprocessor
{
protected:
    InferPostprocessor(NvDsInferNetworkType type, int id, int gpuId)
        : m_NetworkType(type), m_UniqueID(id), m_GpuID(gpuId) {}

public:
    virtual ~InferPostprocessor();
    void setDlHandle(const std::shared_ptr<DlLibHandle>& dlHandle)
    {
        m_CustomLibHandle = dlHandle;
    }
    void setNetworkInfo(const NvDsInferNetworkInfo& info)
    {
        m_NetworkInfo = info;
    }
    void setAllLayerInfo(std::vector<NvDsInferBatchDimsLayerInfo>& info)
    {
        m_AllLayerInfo.resize(info.size());
        std::copy(info.begin(), info.end(), m_AllLayerInfo.begin());
    }
    void setOutputLayerInfo(std::vector<NvDsInferBatchDimsLayerInfo>& info)
    {
        m_OutputLayerInfo.resize(info.size());
        std::copy(info.begin(), info.end(), m_OutputLayerInfo.begin());
    }
    void setLoggingFunc(const NvDsInferLoggingFunc& func)
    {
        m_LoggingFunc = func;
    }
    const std::vector<std::vector<std::string>>& getLabels() const
    {
        return m_Labels;
    }
    bool needInputCopy() const { return m_CopyInputToHostBuffers; }

    bool needOutputCopyB4Processing() const { return !m_disableOutputHostCopy; }

    virtual NvDsInferStatus initResource(
        const NvDsInferContextInitParams& initParams);

    /* Copy inference output from device to host memory. */
    virtual NvDsInferStatus copyBuffersToHostMemory(
        NvDsInferBatch& buffer, CudaStream& mainStream);

    virtual NvDsInferStatus postProcessHost(
        NvDsInferBatch& buffer, NvDsInferContextBatchOutput& output);

    void freeBatchOutput(NvDsInferContextBatchOutput& batchOutput);

private:
    /* Parse the output of each frame in batch. */
    virtual NvDsInferStatus parseEachBatch(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferFrameOutput& result) = 0;

protected:
    NvDsInferStatus parseLabelsFile(const std::string& path);
    NvDsInferStatus allocDeviceResource();
    void releaseFrameOutput(NvDsInferFrameOutput& frameOutput);

private:
    DISABLE_CLASS_COPY(InferPostprocessor);

protected:
    /* Processor type */
    NvDsInferNetworkType m_NetworkType = NvDsInferNetworkType_Other;

    int m_UniqueID = 0;
    uint32_t m_GpuID = 0;
    NvDsInferLoggingFunc m_LoggingFunc;

    /* Custom library implementation. */
    std::shared_ptr<DlLibHandle> m_CustomLibHandle;
    bool m_CopyInputToHostBuffers = false;
    bool m_disableOutputHostCopy = false;
    bool m_DumpOpTensor = false;
    std::vector<std::pair<std::string, std::string>> m_DumpOpTensorFiles;
    bool m_OverwriteOpTensor = false;
    std::vector<std::pair<std::string, int>> m_OverwriteOpTensorFilePairs;
    std::vector<std::ifstream *> m_OverwriteOpTensorFiles;
    /* Network input information. */
    NvDsInferNetworkInfo m_NetworkInfo = {0};
    std::vector<NvDsInferLayerInfo> m_AllLayerInfo;
    std::vector<NvDsInferLayerInfo> m_OutputLayerInfo;

    /* Holds the string labels for classes. */
    std::vector<std::vector<std::string>> m_Labels;
};

/** Implementation of post-processing class for object detection networks. */
class DetectPostprocessor : public InferPostprocessor
{
public:
    DetectPostprocessor(int id, int gpuId = 0)
        : InferPostprocessor(NvDsInferNetworkType_Detector, id, gpuId) {}
    ~DetectPostprocessor() override = default;

    NvDsInferStatus initResource(
        const NvDsInferContextInitParams& initParams) override;

private:
    NvDsInferStatus parseEachBatch(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferFrameOutput& result) override;

    bool parseBoundingBox(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams,
        std::vector<NvDsInferObjectDetectionInfo>& objectList);

    std::vector<int> nonMaximumSuppression
                     (std::vector<std::pair<float, int>>& scoreIndex,
                      std::vector<NvDsInferParseObjectInfo>& bbox,
                      const float nmsThreshold);
    void clusterAndFillDetectionOutputNMS(NvDsInferDetectionOutput &output);
    void clusterAndFillDetectionOutputCV(NvDsInferDetectionOutput& output);
    void clusterAndFillDetectionOutputDBSCAN(NvDsInferDetectionOutput& output);
    void clusterAndFillDetectionOutputHybrid(NvDsInferDetectionOutput& output);
    void fillUnclusteredOutput(NvDsInferDetectionOutput& output);
    NvDsInferStatus fillDetectionOutput(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferDetectionOutput& output);
    void preClusteringThreshold(NvDsInferParseDetectionParams const &detectionParams,
            std::vector<NvDsInferObjectDetectionInfo> &objectList);
    void filterTopKOutputs(const int topK,
                          std::vector<NvDsInferObjectDetectionInfo> &objectList);

private:
    _DS_DEPRECATED_("Use m_ClusterMode instead")
    bool m_UseDBScan = false;
    std::shared_ptr<NvDsInferDBScan> m_DBScanHandle;
    NvDsInferClusterMode m_ClusterMode;

    /* Number of classes detected by the model. */
    uint32_t m_NumDetectedClasses = 0;

    /* Detection / grouping parameters. */
    std::vector<NvDsInferDetectionParams> m_PerClassDetectionParams;
    NvDsInferParseDetectionParams m_DetectionParams = {0, {}, {}};

    /* Vector for all parsed objects. */
    std::vector<NvDsInferObjectDetectionInfo> m_ObjectList;
#ifdef WITH_OPENCV
    /* Vector of cv::Rect vectors for each class. */
    std::vector<std::vector<cv::Rect>> m_PerClassCvRectList;
#endif
    /* Vector of NvDsInferObjectDetectionInfo vectors for each class. */
    std::vector<std::vector<NvDsInferObjectDetectionInfo>> m_PerClassObjectList;

    NvDsInferParseCustomFunc m_CustomBBoxParseFunc = nullptr;
};

/** Implementation of post-processing class for instance segmentation networks. */
class InstanceSegmentPostprocessor : public InferPostprocessor
{
public:
    InstanceSegmentPostprocessor(int id, int gpuId = 0)
        : InferPostprocessor(NvDsInferNetworkType_InstanceSegmentation, id, gpuId) {}
    ~InstanceSegmentPostprocessor() override = default;

    NvDsInferStatus initResource(
        const NvDsInferContextInitParams& initParams) override;

private:
    NvDsInferStatus parseEachBatch(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferFrameOutput& result) override;

    void fillUnclusteredOutput(NvDsInferDetectionOutput& output);
    NvDsInferStatus fillDetectionOutput(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferDetectionOutput& output);
    void preClusteringThreshold(NvDsInferParseDetectionParams const &detectionParams,
            std::vector<NvDsInferInstanceMaskInfo> &objectList);
    void filterTopKOutputs(const int topK,
                          std::vector<NvDsInferInstanceMaskInfo> &objectList);

private:
    NvDsInferClusterMode m_ClusterMode;

    /* Number of classes detected by the model. */
    uint32_t m_NumDetectedClasses = 0;

    /* Detection / grouping parameters. */
    std::vector<NvDsInferDetectionParams> m_PerClassDetectionParams;
    NvDsInferParseDetectionParams m_DetectionParams = {0, {}, {}};

    /* Vector for all parsed instance masks. */
    std::vector<NvDsInferInstanceMaskInfo> m_InstanceMaskList;
    /* Vector of NvDsInferInstanceMaskInfo vectors for each class. */
    std::vector<std::vector<NvDsInferInstanceMaskInfo>> m_PerClassInstanceMaskList;

    NvDsInferInstanceMaskParseCustomFunc m_CustomParseFunc = nullptr;
};

/** Implementation of post-processing class for classification networks. */
class ClassifyPostprocessor : public InferPostprocessor
{
public:
    ClassifyPostprocessor(int id, int gpuId = 0)
        : InferPostprocessor(NvDsInferNetworkType_Classifier, id, gpuId) {}

    NvDsInferStatus initResource(
        const NvDsInferContextInitParams& initParams) override;

private:
    NvDsInferStatus parseEachBatch(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferFrameOutput& result) override;

    NvDsInferStatus fillClassificationOutput(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferClassificationOutput& output);

    bool parseAttributesFromSoftmaxLayers(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
        std::vector<NvDsInferAttribute>& attrList, std::string& attrString);

private:
    float m_ClassifierThreshold = 0.0f;
    NvDsInferClassiferParseCustomFunc m_CustomClassifierParseFunc = nullptr;
};

/** Implementation of post-processing class for segmentation networks. */
class SegmentPostprocessor : public InferPostprocessor
{
public:
    SegmentPostprocessor(int id, int gpuId = 0)
        : InferPostprocessor(NvDsInferNetworkType_Segmentation, id, gpuId) {}

    NvDsInferStatus initResource(
        const NvDsInferContextInitParams& initParams) override;

private:
    NvDsInferStatus parseEachBatch(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferFrameOutput& result) override;

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
    NvDsInferTensorOrder m_SegmentationOutputOrder = NvDsInferTensorOrder_kNCHW;
    NvDsInferSemSegmentationParseCustomFunc m_CustomSegmentationParseFunc = nullptr;
    uint32_t m_NumSegmentationClasses = 0;
};

class OtherPostprocessor : public InferPostprocessor
{
public:
    OtherPostprocessor(int id, int gpuId = 0)
        : InferPostprocessor(NvDsInferNetworkType_Other, id, gpuId) {}

    NvDsInferStatus initResource(
        const NvDsInferContextInitParams& initParams) override;

private:
    NvDsInferStatus parseEachBatch(
        const std::vector<NvDsInferLayerInfo>& outputLayers,
        NvDsInferFrameOutput& result) override {
        return NVDSINFER_SUCCESS;
    }
};

class BackendContext;

/**
 * Implementation of the INvDsInferContext interface.
 */
class NvDsInferContextImpl : public INvDsInferContext
{
public:
    /**
     * Default constructor.
     */
    NvDsInferContextImpl();

    /**
     * Initializes the Infer engine, allocates layer buffers and other required
     * initialization steps.
     */
    NvDsInferStatus initialize(NvDsInferContextInitParams &initParams,
            void *userCtx, NvDsInferContextLoggingFunc logFunc);

private:
    /**
     * Free up resouces and deinitialize the inference engine.
     */
    ~NvDsInferContextImpl() override;

    /* Implementation of the public methods of INvDsInferContext interface. */
    NvDsInferStatus queueInputBatch(NvDsInferContextBatchInput &batchInput) override;
    NvDsInferStatus queueInputBatchPreprocessed(NvDsInferContextBatchPreprocessedInput &batchInput) override;
    NvDsInferStatus dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput) override;
    void releaseBatchOutput(NvDsInferContextBatchOutput &batchOutput) override;
    void fillLayersInfo(std::vector<NvDsInferLayerInfo> &layersInfo) override;
    void getNetworkInfo(NvDsInferNetworkInfo &networkInfo) override;
    const std::vector<std::vector<std::string>>& getLabels() override;
    void destroy() override;

    /* Other private methods. */
    NvDsInferStatus initInferenceInfo(
        const NvDsInferContextInitParams& initParams, BackendContext& ctx);
    NvDsInferStatus preparePreprocess(
        const NvDsInferContextInitParams& initParams);
    NvDsInferStatus preparePostprocess(
        const NvDsInferContextInitParams& initParams);

    std::unique_ptr<BackendContext> generateBackendContext(
        NvDsInferContextInitParams& initParams);
    std::unique_ptr<BackendContext> buildModel(
        NvDsInferContextInitParams& initParams);
    bool deserializeEngineAndBackend(const std::string enginePath, int dla,
        std::shared_ptr<TrtEngine>& engine,
        std::unique_ptr<BackendContext>& backend);
    NvDsInferStatus checkBackendParams(
        BackendContext& ctx, const NvDsInferContextInitParams& initParams);

    NvDsInferStatus getBoundLayersInfo();
    NvDsInferStatus resizeOutputBufferpool(uint32_t numBuffers);
    NvDsInferStatus allocateBuffers();
    NvDsInferStatus initNonImageInputLayers();
    NvDsInferStatus warmupEngine();

    /* Input layer has a binding index of 0 */
    static const int INPUT_LAYER_INDEX = 0;

    /** Unique identifier for the instance. This can be used to identify the
     * instance generating log and error messages. */
    uint32_t m_UniqueID = 0;
    uint32_t m_GpuID = 0;

    /* Custom unique_ptrs. These TensorRT objects will get deleted automatically
     * when the NvDsInferContext object is deleted. */
    std::unique_ptr<BackendContext> m_BackendContext;
    std::shared_ptr<DlLibHandle> m_CustomLibHandle;

    std::unique_ptr<InferPreprocessor> m_Preprocessor;
    std::unique_ptr<InferPostprocessor> m_Postprocessor;

    uint32_t m_MaxBatchSize = 0;
    /* Network input information. */
    NvDsInferNetworkInfo m_NetworkInfo;

    /* Vectors for holding information about bound layers. */
    std::vector<NvDsInferBatchDimsLayerInfo> m_AllLayerInfo;
    std::vector<NvDsInferBatchDimsLayerInfo> m_OutputLayerInfo;
    NvDsInferBatchDimsLayerInfo m_InputImageLayerInfo;

    std::vector<void *> m_BindingBuffers;
    std::vector<std::unique_ptr<CudaDeviceBuffer>> m_InputDeviceBuffers;

    uint32_t m_OutputBufferPoolSize = NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE;
    std::vector<std::shared_ptr<NvDsInferBatch>> m_Batches;
    std::mutex m_BatchesMutex;

    /* Queues and synchronization members for processing multiple batches
     * in parallel.
     */
    GuardQueue<std::list<NvDsInferBatch*>> m_FreeBatchQueue;
    GuardQueue<std::list<NvDsInferBatch*>> m_ProcessBatchQueue;

    std::unique_ptr<CudaStream> m_InferStream;
    std::unique_ptr<CudaStream> m_PostprocessStream;

    /* Cuda Event for synchronizing input consumption by TensorRT CUDA engine. */
    std::shared_ptr<CudaEvent> m_InputConsumedEvent;

    /* Cuda Event for synchronizing infer completion by TensorRT CUDA engine. */
    std::shared_ptr<CudaEvent> m_InferCompleteEvent;

    NvDsInferLoggingFunc m_LoggingFunc;

    bool m_Initialized = false;
    uint32_t m_AutoIncMem = 1;
    double m_MaxGPUMem = 99;
    bool m_DumpIpTensor = false;
    std::string m_DumpIpTensorFilePath = " ";
    bool m_OverwriteIpTensor = false;
    std::string m_OverwriteIpTensorFilePath = " ";
    std::ifstream m_OverwriteIpTensorFile;
};

}

#define printMsg(level, tag_str, fmt, ...)                                  \
    do {                                                                    \
        char* baseName = strrchr((char*)__FILE__, '/');                     \
        baseName = (baseName) ? (baseName + 1) : (char*)__FILE__;           \
        char logMsgBuffer[5 * _MAX_STR_LENGTH + 1];                             \
        snprintf(logMsgBuffer, 5 * _MAX_STR_LENGTH,                             \
            tag_str " NvDsInferContextImpl::%s() <%s:%d> [UID = %d]: " fmt, \
            __func__, baseName, __LINE__, m_UniqueID, ##__VA_ARGS__);       \
        if (m_LoggingFunc) {                                                \
            m_LoggingFunc(level, logMsgBuffer);                             \
        } else {                                                            \
            fprintf(stderr, "%s\n", logMsgBuffer);                          \
        }                                                                   \
    } while (0)

#define printError(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_ERROR, "Error in", fmt, ##__VA_ARGS__); \
    } while (0)

#define printWarning(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_WARNING, "Warning from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printInfo(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_INFO, "Info from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printDebug(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_DEBUG, "DEBUG", fmt, ##__VA_ARGS__); \
    } while (0)

#endif
