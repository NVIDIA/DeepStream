/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file gstnvinferserver_impl.h
 *
 * @brief nvinferserver implementation header file.
 *
 * This file contains the declarations for the nvinferserver implementation
 * class: GstNvInferServerImpl.
 */

#ifndef __GSTNVINFERSERVER_IMPL_H__
#define __GSTNVINFERSERVER_IMPL_H__

#include <glib.h>
#include <gst/gst.h>
#include <string.h>
#include <sys/time.h>

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include "infer_icontext.h"
#include "infer_utils.h"

#include "nvdsmeta.h"
#include "nvtx3/nvToolsExt.h"

#include "gstnvinferserver_meta_utils.h"
#include "infer_datatypes.h"
#include "infer_post_datatypes.h"
#include "nvdsinferserver_config.pb.h"
#include "nvdsinferserver_plugin.pb.h"

/**
 * @brief Function to output the logs using the GStreamer's debugging
 * subsystem.
 */
void gst_nvinfer_server_logger(uint32_t unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx);

namespace dsis = nvdsinferserver;
namespace ic = nvdsinferserver::config;

using SharedInferContext = std::shared_ptr<dsis::IInferContext>;

namespace nvdsinferserver {
class BatchSurfaceBuffer;
}

namespace gstnvinferserver {

/**
 * @brief Holds the inference information/history for one object based on its
 * tracking ID.
 */
struct GstNvInferServerObjectHistory {
    /** Boolean indicating if the object is already being inferred on. */
    bool under_inference = FALSE;
    /** Bounding box co-ordinates of the object when it was last inferred on. */
    NvOSD_RectParams last_inferred_coords = {0, 0, 0, 0};
    /** Number of the frame in the stream when the object was last inferred on.
     */
    uint64_t last_inferred_frame_num = 0;
    /** Number of the frame in the stream when the object was last accessed.
     * This is useful for clearing stale entries in map of the object histories
     * and keeping the size of the map in check. */
    uint64_t last_accessed_frame_num = 0;
    /** Cached object information. */
    InferClassificationOutput cached_info;
};

using SharedObjHistory = std::shared_ptr<GstNvInferServerObjectHistory>;
using WeakObjHistory = std::weak_ptr<GstNvInferServerObjectHistory>;

/**
 * @brief Map for maintaining inference history for objects based on their tracking
 * ids: Object ID : Object History*/
typedef std::unordered_map<uint64_t, SharedObjHistory>
    GstNvInferServerObjectHistoryMap;

/**
 * @brief Holds information about the detected objects in the specific input source.
 */
typedef struct {
    /** Map of object tracking IDs and inference history of the object. */
    GstNvInferServerObjectHistoryMap object_history_map;
    /** Frame number of the buffer when the history map was last cleaned up. */
    gulong last_cleanup_frame_num = 0;
    /** Frame number of the frame which . */
    gulong last_seen_frame_num = 0;
} GstNvInferServerSourceInfo;

/**
 * @brief Holds the configuration information from the nvinferserver element properties.
 */
struct GstNvInferServerProperties {
    std::string configPath;
    uint32_t maxBatchSize = 0;
    int32_t interval = -1;
    uint32_t uniqueId = 0;
    uint32_t processMode = 0;
    int32_t inferOnGieId = -1;
    bool inputTensorFromMeta = false;
    std::vector<int32_t> operateOnClassIds;
    std::string classifierType;
};

/**
 * @brief Function for raw output callback .
 *
 * Function for raw output callback
 * (void *gstBuf, NvDsInferNetworkInfo *network_info,
 *   NvDsInferLayerInfo *layers_info,
 *   uint32_t num_layers, uint32_t batch_size)
 */
using RawOutputCallback = std::function<void(void *, NvDsInferNetworkInfo&,
    NvDsInferLayerInfo*, uint32_t, uint32_t)>;

struct RequestBuffer;

/**
 * @brief Class of the nvinferserver element implementation.
 */
class GstNvInferServerImpl {
private:
    /**
     * @brief A class for storing information about tracked objects.
     *
     * Maintains the history of objects detected in each input source.
     */
    class ObjTrackingData {
    public:
        /**
         * @brief Constructor, do nothing.
         */
        ObjTrackingData() {}

        /**
         * @brief Creates the object history structure for specified source.
         *
         * @param id Source ID.
         */
        bool initSource(uint32_t id) {
            m_SourceInfo.emplace(id, GstNvInferServerSourceInfo{});
            return true;
        }

        /**
         * @brief Deletes the object history structure for specified source.
         *
         * @param id Source ID.
         */
        void removeSource(uint32_t id) { m_SourceInfo.erase(id); }

        /**
         * @brief Retrieves the object history for the specified source.
         *
         * @param id Source ID.
         * @returns Pointer to object history structure for the requested
         * source ID.
         */
        GstNvInferServerSourceInfo* findSource(int id);

        /**
         * @brief Checks if the object history for the specified source is
         * available.
         *
         * @param id Source ID.
         * @returns Boolean indicating if the object history is present or not.
         */
        bool hasSource(uint32_t id) const {
            return m_SourceInfo.find(id) != m_SourceInfo.end();
        }

        /**
         * @brief Retrieves the history of the object in a source.
         *
         * @param sourceID Source ID.
         * @param sourceID Object ID.
         * @returns Shared pointer to the history structure of the object.
         * Null pointer if the object or source is not found.
         */
        SharedObjHistory findObjHistory(
            uint32_t sourceId, uint64_t objId);

        /**
         * @brief Create a new history for a object in the source.
         *
         * @param sourceID Source ID.
         * @param sourceID Object ID.
         * @returns Shared pointer to the history structure of the object.
         * Null pointer if the source is not found.
         */
        SharedObjHistory newHistory(
            uint32_t sourceId, uint64_t objId);

        /**
         * @brief Deletes the object history for all sources and removes
         * the sources from the map.
         */
        void clear() { m_SourceInfo.clear(); }

        /**
         * @brief Periodically cleanups the object history for sources present
         * in current batch.
         *
         * Removes objects not seen for CLEANUP_ACCESS_CRITERIA number of
         * frames. The cleanup is triggered after an interval of
         * MAP_CLEANUP_INTERVAL number of input batches.
         *
         * @param[in] seqId Sequence ID (batch number) of the input batch.
         */
        void clearUpHistory(uint64_t seqId);

    private:
        /** Map of history of detected objects in each source.
         * source_id : GstNvInferServerSourceInfo */
        std::unordered_map<uint32_t, GstNvInferServerSourceInfo> m_SourceInfo;

        /** Last input batch number when the object history was cleaned.
         */
        gulong m_LastMapCleanupSeqId = 0;
    };

    using FuncItem = std::function<void()>;

public:
    /**
     * @brief Constructor, registers the handle of the parent GStreamer
     * element.
     *
     * @param[in] infer Pointer to the nvinferserver GStreamer element.
     */
    GstNvInferServerImpl(GstNvInferServer* infer);

    /**
     * @brief Constructor, do nothing.
     */
    ~GstNvInferServerImpl();

    /**
     * @brief Saves the callback function pointer for the raw tensor output.
     *
     * @param[in] cb Pointer to the callback function.
     */
    void setRawoutputCb(RawOutputCallback cb) {
        m_RawoutputCb = cb;
    }

    /**
     * @brief Reads the configuration file and sets up processing context.
     *
     * This function reads the configuration file and validates the user
     * provided configuration. Configuration file settings are overridden
     * with those set by element properties.
     * It then creates the inference context, initializes it and starts the
     * output thread. The inference context is either InferGrpcContext
     * (Triton Inference Server in gRPC mode) or
     * InferTrtISContext (Triton Inference server C-API mode) depending
     * on the configuration setting.
     * Object history is initialized for source 0.
     */
    NvDsInferStatus start();

    /**
     * @brief Deletes the inference context.
     *
     * This function waits for the output thread to finish and then
     * de-initializes the inference context. The object history is
     * cleared.
     */
    NvDsInferStatus stop();

    /**
     * @brief Submits the input batch for inference.
     *
     * This function submits the input batch buffer for inferencing
     * as per the configured processing mode: full frame inference,
     * inference on detected objects or inference on attached input tensors.
     *
     * @param[inout] batchMeta NvDsBatchMeta associated with the input buffer.
     * @param[in]    inSurf    Input batch buffer.
     * @param[in]    seqId     The sequence number of the input batch.
     * @param[in]    gstBuf    Pointer to the input GStreamer buffer.
     */
    NvDsInferStatus processBatchMeta(NvDsBatchMeta* batchMeta,
        NvBufSurface* inSurf, uint64_t seqId, void *gstBuf);

    /**
     * @brief Waits for the output thread to finish processing queued
     * operations.
     */
    NvDsInferStatus sync();

    /**
     * @brief Queues the inference done operation for the request to
     * the output thread.
     */
    NvDsInferStatus queueOperation(FuncItem func);


    /**
     * @brief Add a new source to the object history structure
     *
     * Whenever a new source is added to the pipeline, corresponding source ID
     * is captured in the GStreamer event on the sink pad and the object history
     * for this source is initialized.
     *
     * @param[in] id Source ID.
     */
    bool addTrackingSource(uint32_t sourceId);

    /**
     * @brief Removes a source from the object history structure.
     *
     * Whenever a source is removed from the pipeline, corresponding source ID
     * is captured in the GStreamer event on the sink pad and the object history
     * for this source is deleted.
     *
     * @param[in] id Source ID.
     */
    void eraseTrackingSource(uint32_t sourceId);

    /**
     * @brief Resets the inference interval used in frame process mode to 0.
     */
    void resetIntervalCounter();

    // helper functions
    uint32_t uniqueId() const { return m_GstProperties.uniqueId; }
    const std::string& classifierType() const { return m_GstProperties.classifierType; }
    uint32_t maxBatchSize() const;
    nvtxDomainHandle_t nvtxDomain() {
        return m_NvtxDomain.get();
    }
    bool isAsyncMode() const;
    bool canSupportGpu(int gpuId) const;
    NvDsInferStatus lastError() const;
    const ic::PluginControl& config() const { return m_PluginConfig; }
    void updateInterval(guint interval)  {m_PluginConfig.mutable_input_control()->set_interval(interval);}
private:
    /**
     * @brief Runs inference on full frames.
     *
     * This function submits the input batched frames for inference.
     * The inference is skipped for number of batches specified by the
     * inference interval. Up to the configured max_batch_size number of frames
     * are grouped together in a single inference request.
     *
     * @param[inout] batchMeta NvDsBatchMeta associated with the input buffer.
     * @param[in]    inSurf    Input batch buffer.
     * @param[in]    seqId     The sequence number of the input batch.
     * @param[in]    gstBuf    Pointer to the input GStreamer buffer.
     */
    NvDsInferStatus processFullFrame(NvDsBatchMeta* batchMeta,
        NvBufSurface* inSurf, uint64_t seqId, void *gstBuf);

    /**
     * @brief Runs inference on objects specified in the metadata of the input
     * batch.
     *
     * This function iterates over the object metadata associated with each
     * frame within the input batch and submits them for inference based on
     * object history. For classification type of networks the inference is
     * skipped if the object is present in object history of the frame and it
     * was last inferred within MAX_SECONDARY_REINFER_INTERVAL.
     * Upto the configured max_batch_size number of objects are grouped
     * together in a single inference request.
     *
     * @param[inout] batchMeta NvDsBatchMeta associated with the input buffer.
     * @param[in]    inSurf    Input batch buffer.
     * @param[in]    seqId     The sequence number of the input batch.
     * @param[in]    gstBuf    Pointer to the input GStreamer buffer.
     */
    NvDsInferStatus processObjects(NvDsBatchMeta* batchMeta,
        NvBufSurface* inSurf, uint64_t seqId, void *gstBuf);

    /**
     * @brief Runs inference on tensors attached as preprocess metadata
     * by upstream nvdspreprocess element.
     *
     * This function iterates over the preprocess metadata attached to
     * the input batch buffer and submits them for inference.
     *
     * @param[inout] batchMeta NvDsBatchMeta associated with the input buffer.
     * @param[in]    inSurf    Input batch buffer.
     * @param[in]    seqId     The sequence number of the input batch.
     * @param[in]    gstBuf    Pointer to the input GStreamer buffer.
     */
    NvDsInferStatus processInputTensor(
        NvDsBatchMeta* batchMeta, NvBufSurface* inSurf, uint64_t seqId, void* gstBuf);

    /**
     * @brief Submits the inference request to the inference context and
     * add InferenceDone task to the output thread queue.
     *
     * This function forms an array of batch inputs, BaseBatchArray, and
     * triggers the inference with the inference context after updating
     * the options from the inference request. It queues the corresponding
     * InferenceDone task to the output thread.
     *
     * @param[inout] reqBuf Pointer to the inference request.
     * @param[in] batchBuf Array of input buffers for the inference batch.
     */
    NvDsInferStatus batchInference(
        std::shared_ptr<RequestBuffer> reqBuf, std::vector<dsis::SharedBatchBuf> batchBuf);

    /** @brief Calls the post processing functions after completion of
     * inference request.
     *
     * This function is queued to the output thread once a inference request
     * is triggered. On completion of the inference request, corresponding post
     * processing routine is executed, attachDetectionMetadata(),
     * attachClassificationMetadata() or attachSegmentationMetadata().
     * It also calls the handleOutputTensors() function.
     *
     * @param[inout] req Pointer to the inference request.
     */
    void InferenceDone(std::shared_ptr<RequestBuffer> req);

    /** @brief Attaches the detection output to the batch metadata.
     *
     * @param[inout] req Pointer to the inference request.
     * @param[in] output Pointer to the detection output batched buffer.
     */
    NvDsInferStatus attachBatchDetection(
        RequestBuffer* req, dsis::SharedIBatchBuffer output);

    /** @brief Attaches the classification output to the batch metadata.
     *
     * @param[inout] req Pointer to the inference request.
     * @param[in] output Pointer to the classification output batched buffer.
     */
    NvDsInferStatus attachBatchClassification(
        RequestBuffer* req, dsis::SharedIBatchBuffer output);

    /** @brief Attaches the segmentation output to the batch metadata.
     *
     * @param[inout] req Pointer to the inference request.
     * @param[in] output Pointer to the segmentation output batched buffer.
     */
    NvDsInferStatus attachBatchSegmentation(
        RequestBuffer* req, dsis::SharedIBatchBuffer output);

    /**
     * @brief Attaches the output tensors in the metadata and triggers raw
     * output tensor callback if configured.
     *
     * This functions attaches the output tensors generated for the inference
     * request to the batch metadata. It also executes the output tensor
     * callback function if set by user.
     *
     * @param[inout] req Pointer to the inference request.
     */
    NvDsInferStatus handleOutputTensors(RequestBuffer* req);

    void updateLastError(NvDsInferStatus s);

    /**
     * Helper functions to access the configuration settings.
     */
    /**@{*/
    ic::PluginControl& mutableConfig() { return m_PluginConfig; }
    bool isClassify() const;
    bool isDetection() const;
    bool isSegmentation() const;
    bool isOtherNetowrk() const;
    bool isFullFrame() const;
    bool maintainAspectRatio() const;
    bool symmetricPadding() const;
    bool needOutputTensorMeta() const;
    bool hasCustomProcess() const;
    uint32_t inferInterval() const { return m_PluginConfig.input_control().interval(); }
    bool inputTensorFromMeta() const
    {
        return m_PluginConfig.infer_config().has_input_tensor_from_meta();
    }
    /**@}*/

    /**
     * @brief Validates the configuration parameters specified in the
     * configuration file.
     *
     * This function validates the configuration settings received from
     * the configuration file. The configuration from GStreamer element
     * properties over-ride those from the configuration file.
     */
    bool validatePluginConfig(
        ic::PluginControl& config, const std::string& path,
        const GstNvInferServerProperties& update);

    /**
     * @brief Checks if inference should be done on the object.
     *
     * This function checks if the object present in the metadata satisfies
     * the configuration setting and should be passed to inference or not.
     * It checks for following conditions:
     * - Parent GIE ID matches the one configured by operate_on_gie_id.
     * - The bounding box height, width are with the bounds specified by
     *   BBoxFilter in InputObjectControl.
     * - Class IDs of parent GIE match those specified by operate_on_class_ids.
     * - In case of classification, if the object history is available it is
     *   not inferred if the last frame was within
     *   MAX_SECONDARY_REINFER_INTERVAL and object area has not increased more
     *   than REINFER_AREA_THRESHOLD.
     * - In case of classification and if the object history is available, if the
     *   object is already under inference, in previous frame(s), the
     *   inference is skipped.
     *
     * @param[in] obj_meta Metadata of the object.
     * @param[in] frameNum Frame number corresponding to the object.
     * @param[in] history History for the object.
     * @returns Boolean indicating if the obeject should be included in the
     * inference request.
     */
    bool shouldInferObject(NvDsObjectMeta* obj_meta, uint32_t frameNum,
        GstNvInferServerObjectHistory* history);

    /**
     * @brief Output processing wrapper function called in the output thread.
     *
     * This function executes supplied function object and is used in the
     * output thread to execute the queued tasks in a loop.
     *
     * @param[in] func Function to be executed in the output thread.
     * @returns[out] status (true always).
     */
    bool outputLoop(FuncItem func);

public:
    GstNvInferServerProperties m_GstProperties;

private:
    using OutputThread = dsis::QueueThread<std::list<FuncItem>>;

    /** NvDsInferContext to be used for inferencing. */
    SharedInferContext m_InferCtx;
    /** Pointer to the GStreamer element. */
    GstNvInferServer* m_GstPlugin = nullptr;
    /** Mutex to control access to object history and other data. */
    mutable std::mutex m_ProcessMutex;
    /** Configuration information in protobuf format. */
    ic::PluginControl m_PluginConfig;
    /** Object history for all the input sources. */
    ObjTrackingData m_ObjTrackingData;

    /** Thread for output processing tasks. */
    std::unique_ptr<OutputThread> m_OutputThread;

    /** NvTx domain for registering NvTx events. */
    std::unique_ptr<nvtxDomainRegistration, std::function<void(nvtxDomainRegistration*)>>
        m_NvtxDomain;
    /** Indicates that the context has been de-initialized and processing
     * stopped. */
    bool m_Stopped = false;
    /** Stores the error status of last inference, used when sending the input
     * buffer downstream. */
    NvDsInferStatus m_LastInferError = NVDSINFER_SUCCESS;

    /** Pointer to callback function for processing on raw output tensors. */
    RawOutputCallback m_RawoutputCb = nullptr;

    /** Interval of batches to be skipped for inference in process frame mode */
    uint32_t m_IntervalCounter = 0;
    /** Stores the frame PTS for untracked objects */
    uint64_t m_UntrackedObjectWarnPts = UINT64_C(-1);
    /** Flag indicating if it is first inference, used for warning prints if
     * any */
    bool m_1stInferDone = false;
};

} // namespace gstnvinferserver

#endif
