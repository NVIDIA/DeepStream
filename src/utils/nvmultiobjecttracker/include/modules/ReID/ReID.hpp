/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef REIDLIB_REID_HPP_
#define REIDLIB_REID_HPP_

#include <map>
#include <iterator>
#include <memory>

#ifdef VIZ_DEBUG_REID
#include <opencv2/core.hpp>
#endif

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "modules/ReID/IReID.hpp"
#include "modules/ReID/ReIDFeatureGallery.hpp"
#include "modules/ReID/ReIDFrameTransformTexture.hpp"
#include "modules/ReID/ReIDCropScaler.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/FrameTransform/trtModule.hpp"

/** @brief Stores configuration for a ReID inference module. */
class ReIDConfigParams : public NvTrackerParams
{
public:
    /** @brief Construct a ReIDConfigParams instance. */
    ReIDConfigParams();

    /**
     * @brief Construct a ReIDConfigParams instance.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] moduleName Configuration module name.
     * @param[in] submoduleName Configuration submodule name.
     */
    ReIDConfigParams(const std::string &configFilePath, const std::string moduleName, const std::string submoduleName);

    /**
     * @brief Construct a ReIDConfigParams instance.
     * @param[in] fn Parsed YAML node.
     */
    ReIDConfigParams(const YAML::Node &fn);

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override;

    bool m_keepAspc; //!< Keep original aspect ratio when rescaling.
    bool m_addFeatureNormalization; //!< Whether to perform additional feature normalization for network output features.
    bool m_outputReidTensor; //!< Whether to output reid tensor.
    int m_maxBatchSize; //!< Maximum batch size for reid network.
    int m_reidFeatureSize; //!< Size of each reid feature.
    int m_inputOrder; //!< Input dimension order: 0: NCHW, 1: NHWC.
    int m_width; //!< Input image patch width.
    int m_height; //!< Input image patch height.
    int m_channel; //!< Input image patch channel.
    int m_reidHistorySize; //!< Max number of reid features per object.
    int m_gpuId; //!< Gpu id.
    float m_netScaleFactor; //!< Input normalization factor.
    float m_minVisibility4GalleryUpdate; //!< Add ReID embedding to the gallery only if the visibility is not lower than this.
    float m_minVisibility4ReidOutput; //!< Minimum visibility to output the reid features.
    size_t m_workspaceSize; //!< Work space for reid network in MiB.
    std::vector<int> m_inferDims; //!< Reid input dimension.
    std::vector<float> m_offsets; //!< List of mean values of color components to be subtracted from input.
    NvBufSurfaceColorFormat m_colorFormat; //!< Color format of input to reid network.
    REID_TYPES m_reidType; //!< Type of reid.
    bool m_useVPICropScaler; //!< Use VPI crop scaler instead of in built crop scaler.
    bool m_useSegmentationMask4Input; //!< Multiply input patches with segmentation mask, so background is set as zero.
    int m_debugVerboseLevel; //!< Debug message verbose mode level.
    #ifdef VIZ_DEBUG_REID
        bool m_useExternalImages; //!< Read pre generated opencv images.
        std::string m_extImagePath; //!< External image folder.
    #endif
};

/** @brief ReID module to extract deep ReID features from input frames and detected bounding boxes and save them in GPU memory. */
class ReID : public IReID
{
public:

    /**
     * @brief Construct a ReID instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     */
    ReID(const std::string &configFilePath);

    /**
     * @brief Construct a ReID instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] modelName Configured model name.
     * @param[in] calDisMaha If Mahalanobis distance needs to be computed.
     */
    ReID(const std::string &configFilePath, const std::string &modelName, const bool calDisMaha=true);

    /** @brief Release resources owned by the ReID instance. */
    virtual ~ReID();

    /**
     * @brief Initialize variables and allocate memories.
     * @param[in] frameInfo Metadata about the source image, such as size and color.
     * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
     * @param[in] targetManager Target manager of the tracked objects.
     */
    void initializeRuntime(
        const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
        const std::shared_ptr<NvTrackedObjectManager> &targetManager
    ) override;

    /** @brief Create TensorRT inference engine; if engine does not exist, serialize from model file. */
    void initializeModel() override;

    /**
     * @brief Populate the capabilities reported to the tracker.
     * @param[out] paramsQuery The output query parameters.
     */
    void queryParams(NvMOTQuery &paramsQuery);

    /**
     * @brief Extract ReID features from input images and save them in m_reidFeaturesDet_d.
     * @param[in] inputMeta A batch of input frames.
     * @param[in,out] detectedObjects Newly detected objects in each stream.
     * @param[in] isForReAssoc Whether this inference is for reassociation.
     */
    void doInference(
        const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
    ) override;

    /** @brief Update ReID feature gallery after data association. */
    void update() override;

    /**
     * @brief Remove a stream's reid features.
     * @param[in] videoStreamId Video stream id.
     */
    void removeStream(const NvMOTStreamId videoStreamId) override;

    /**
     * @brief Output the ReID tensor of a batch.
     * @param[out] reidTensorBatch Batch reid tensor output to be filled.
     */
    void outputReIDTensor(NvDsReidTensorBatch& reidTensorBatch) override;

    /**
     * @brief Output a target's ReID vector index in the batch tensor.
     * @param[in] streamId Stream index.
     * @param[in] targetInd The target index.
     * @return The target's ReID vector index in the batch tensor, -1 means no reid feature.
     */
    int outputTargetReIDInd(NvMOTStreamId streamId, uint targetInd) override;

    /**
     * @brief Output a target's ReID vector information.
     * @param[in] streamId Stream index.
     * @param[in] targetInd The target index.
     * @param[in] reidTensorBatch Batch reid tensor to get the memory pointers.
     * @param[out] reidObj The target's ReID vector memory pointers and feature size.
     */
    void outputTargetReIDObj(
        NvMOTStreamId streamId, uint targetInd, NvDsReidTensorBatch& reidTensorBatch, NvDsObjReid& reidObj
    ) override;

    #ifdef VIZ_DEBUG_REID
        /**
         * @brief Visualize image input and reid features.
         * @param[in] subImgs Cropped target images keyed by stream and target.
         * @param[in] inputMeta Input frames keyed by stream ID.
         */
        void vizImagePatches(
            const std::map<NvMOTStreamId, std::map<uint, cv::Mat > >& subImgs,
            const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta
        );

        /** @brief Load externally supplied image patches for inference. */
        void loadExtImagePatchesBatch();
    #endif

    /** @brief Copy ReID features from device memory to host memory. */
    void getFeatures2Host();

    /** @brief Report whether the inference model is initialized. */
    bool isModelInitialized() override { return m_isModelInit; }

    /** @brief Report whether runtime resources are initialized. */
    bool isRuntimeInitialized() override { return m_isRuntimeInit; }

    /** @brief Get the configured ReID implementation type. */
    const REID_TYPES getReIDType() override { return m_params.m_reidType; }

protected:
    bool m_isModelInit; //!< Reid status if reid model initialized.
    bool m_isRuntimeInit; //!< Reid status if runtime variables initialized.
    ReIDConfigParams m_params; //!< Config params for reid module.
    TrtModule m_inference; //!< TensorRT module that runs the reid network.
    std::shared_ptr<ReIDFeatureGallery> m_reidFeatureGallery; //!< ReID feature gallery for target management.
    std::shared_ptr<NvTrackedObjectManager> m_targetManager; //!< Target manager providing target status to update feature gallery.
    // Debug timers.
    #ifdef VIZ_DEBUG_REID
        int m_extFrameNum; //!< External frame number.
        double m_timeTotal; //!< A timer to measure total time.
        int m_timeCnt; //!< How many times timer has called.
    #endif
    bool m_calDisMaha; //!< If the module needs to compute Mahalanobis distance.
    int m_maxTargetsToTrack; //!< Max number of targets to keep for all streams.
    int m_maxTargetsPerStream; //!< Max number of targets to keep per stream.
    int m_maxStreams; //!< Max number of streams.
    int m_numBboxes; //!< Number of detected boxes in the current batch tracker recieved.
    NvMotDataTypes::Size m_srcImageSize; //!< Tracker library frame size.
    std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f>>> m_inputRoisInBatch; //!< Reid input rois to crop.
    std::shared_ptr<ReIDFrameTransformTexture> m_subImageCropper2; //!< Image cropper, may share with visual tracker in the future.
    std::shared_ptr<ReIDCropScaler> m_subImageCropScaler; //!< Image crop scaler. Uses VPI crop scaler.
    std::map<NvMOTStreamId, std::pair<float*, std::map<int, int>>> m_detObjReidFeatures; //!< Mapping each detected object to its reid features : {streamId, <each stream's reid features beginning address, {detInd : pointer to each detection}> }.
    std::map<NvMOTStreamId, std::map<int, int>> m_featureIndFrameNumMap; //!< Frame number of each gallery reid feature, keyed by stream and global feature index.

    // [ReID network parameters]
    int m_inputInd; //!< The index of pointer to input tensor.
    int m_outputInd; //!< The index of pointer to output tensor.
    float *m_reidFeaturesDet_d, *m_reidFeaturesDet_h; //!< Output features in a batch from reid network.
    float *m_inputBuf_d, *m_inputBuf_h; //!< Input patches to reid network.
    std::vector<void*> m_bindings; //!< Pointers to TensorRT tensors.
    cudaStream_t m_cuStream; //!< Cuda stream to run all GPU tasks.

    // [ReID feature similarity computation]
    int m_maxReidProductSizePerStream; //!< Max reid CUDA mat size for each stream.
    float *m_reidProduct_d, *m_reidProduct_h; //!< Dot products between detection and gallery reid features of each stream.
    int *m_targetReidFeatureInds_d, *m_targetReidFeatureInds_h; //!< Tracked objects' list of re-id feature gallery inds, size: m_maxTargetsToTrack x m_reidHistorySize.
    int2 *m_reidProductSize_d, *m_reidProductSize_h; //!< Actual number of re-id products in each stream; x: NumOfFeaturesInGalleryThisStream, y: numOfDetThisStream.
    int *m_targetWork_1_d, *m_targetWork_1_h; //!< Work space storing each target's info, size: m_maxTargetsToTrack.
    int *m_targetWork_2_d, *m_targetWork_2_h; //!< Work space storing each target's info, size: m_maxTargetsToTrack.
    int *m_targetWork_3_d, *m_targetWork_3_h; //!< Work space storing each target's info, size: m_maxTargetsToTrack.
    float *m_confidenceReidDetTargets_d, *m_confidenceReidDetTargets_h; //!< Re-id similarity between each target and detection after taking nearest neighbors, size: m_maxTargetsToTrack x maxDetsPerStream.
    float4 *m_inputRoisInBatch_d, *m_inputRoisInBatch_h; //!< Reid input rois to crop stored in a continuous array.
    int *m_targetInds4Rois_d, *m_targetInds4Rois_h; //!< Target ind for each roi input.
    int *m_nearestIndReidDetTargets_d, *m_nearestIndReidDetTargets_h; //!< Targets' nearest neighbor index in reid features.
    int m_normFeatureCudaBlockSize; //!< Cuda block size for reid feature normalization.
    cublasHandle_t m_cublasHandle; //!< Cublas handle.

    // [Mahalanobis distance thread]
    std::shared_ptr<NvTrackerMultiThreadWorker<DisMahalanobisIn>> m_disMahaThreadWorker; //!< Mahalanobis distance thread.

    /**
     * @brief Crop input frames using detected bboxes and rescale them as ReID network input size.
     * @param[in] inputMeta Input frames keyed by stream ID.
     * @param[in] detectedObjects Newly detected objects in each stream.
     */
    void extractImagePatchesBatch(
        const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
    );

    /**
     * @brief Multiply segmentation mask with image patches for ReID input.
     * @param[in,out] detectedObjects Newly detected objects in each stream.
     */
    void generateImagePatchesWithSegmentationMask(
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
    );

    /**
     * @brief Removed a target's all reid features.
     * @param[in] videoStreamId Video stream id.
     * @param[in] target Tracker associated with the detection.
     */
    void removeTrackerReIDFeature(const NvMOTStreamId videoStreamId, const std::shared_ptr<NvTrackedObject> target);

    /**
     * @brief Compute reid feature distance between detection and trackers.
     * @param[in] inputMeta Input frames keyed by stream ID.
     */
    void computeReIDFeatureDist(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta);

    /**
     * @brief Compute dot products between detection and gallery ReID features.
     * @param[in] streamInd Index of this stream in the current input batch.
     * @param[in] videoStreamId Video stream identifier.
     */
    void multiplyReIDFeatures(const int streamInd, const NvMOTStreamId videoStreamId);

    /** @brief Normalize ReID network output to ensure l2 norm equals to 1. */
    void normalizaReIDFeatures();

    /**
     * @brief Compute Mahalanobis distance for NvDeepSORT tracker.
     * @param[in] in Input to compute Mahalanobis distance.
     */
    void computeMahalanobisDist(DisMahalanobisIn in);
};

#endif // REIDLIB_REID_HPP_
