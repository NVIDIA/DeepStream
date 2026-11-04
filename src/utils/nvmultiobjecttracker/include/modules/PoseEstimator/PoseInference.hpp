/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef POSEESTIMATORLIB_POSEINFERENCE_HPP_
#define POSEESTIMATORLIB_POSEINFERENCE_HPP_

#include <deque>
#include <map>

#ifdef VIZ_DEBUG_POSE
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "modules/PoseEstimator/IPoseEstimator.hpp"
#include "modules/FrameTransform/trtModule.hpp"
#include "modules/ReID/ReIDFrameTransformTexture.hpp"
#include "modules/ReID/ReIDCropScaler.hpp"

/** @brief Stores configuration for pose inference. */
class PoseInferenceConfigParams : public NvTrackerParams
{
public:
    /** @brief Construct a PoseInferenceConfigParams instance. */
    PoseInferenceConfigParams();

    /**
     * @brief Construct a PoseInferenceConfigParams instance.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] moduleName Configuration module name.
     */
    PoseInferenceConfigParams(const std::string &configFilePath, const std::string moduleName);

    /**
     * @brief Construct a PoseInferenceConfigParams instance.
     * @param[in] fn Parsed YAML node.
     */
    PoseInferenceConfigParams(const YAML::Node &fn);

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override;

    POSEESTIMATOR_TYPES m_poseEstimatorType; //!< Type of pose estimator.
    int m_hiddenDim; //!< Size of each reid feature.
    std::vector<int> m_inferDims; //!< Reid input dimension.
    int m_inputOrder; //!< Input dimension order: 0: NCHW, 1: NHWC.
    int m_encoderFeatureDim; //!< Size of each encoder feature.
    int m_width; //!< Input image patch width.
    int m_height; //!< Input image patch height.
    int m_channel; //!< Input image patch channel.
    int m_maxBatchSize; //!< Maximum batch size for reid network.
    int m_gpuId; //!< Gpu id.
    int m_poseInferenceInterval; //!< Pose inference interval.
    float m_netScaleFactor; //!< Input normalization factor.
    std::vector<float> m_offsets; //!< List of mean values of color components to be subtracted from input.
    NvBufSurfaceColorFormat m_colorFormat; //!< Color format of input to reid network.
    bool m_useVPICropScaler; //!< Use VPI crop scaler instead of in built crop scaler.
    std::string m_vizDir; //!< Dir to save visualization dumps.
    int m_debugVerboseLevel; //!< Debug message verbose mode level.
    bool m_saveFrameViz; //!< Save overlay masks for each target in each frame.
    std::vector<int> m_operateOnClassIds; //!< Class IDs selected for pose estimation.
    //!< An empty list enables all classes, while [0] selects only the person class.
};

/** @brief Runs pose inference for tracked target regions. */
class PoseInference: public IPoseEstimator
{
public:
    /**
     * @brief Construct a PoseInference instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     */
    PoseInference(const std::string &configFilePath);

    /**
     * @brief Construct a PoseInference instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] modelName Configured model name.
     */
    PoseInference(const std::string &configFilePath, const std::string &modelName);

    /** @brief Release resources owned by the PoseInference instance. */
    virtual ~PoseInference();

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
     * @brief Run pose inference on cropped targets from the input frames.
     * @param[in] inputMeta A batch of input frames.
     * @param[in,out] detectedObjects Newly detected objects in each stream.
     * @param[in] isForReAssoc Whether this inference is for reassociation.
     */
    void doInference(
        const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
        std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
    ) override;

    /** @brief Update the pose estimator after data association. */
    void update() override;

    /**
     * @brief Remove a stream from the pose estimator.
     * @param[in] videoStreamId Video stream id.
     */
    void removeStream(const NvMOTStreamId videoStreamId) override;

    /** @brief Get the configured pose-estimator type. */
    const POSEESTIMATOR_TYPES getPoseEstimatorType() override { return m_params.m_poseEstimatorType; }

protected:
    bool m_isRuntimeInit; //!< Reid status if runtime variables initialized.
    int m_maxPointNum; //!< Max number of points.
    int m_numBboxes; //!< Number of detected boxes in the current batch tracker recieved.
    float *m_inputBuf_d, *m_inputBuf_h; //!< Input patches to encoder.
    float *m_points_d; //!< Input a list of points (prompt) to decoder.
    float *m_pointLabels_d; //!< Input a list of point types (single point, bounding box corners) to decoder.
    float *m_inputMask_d; //!< Mask prompt, currently unused and set as 0.
    float *m_inputMaskLabel_d; //!< Mask label, currently unused and set as 0.
    float *m_targetMaskDecSize_d, *m_targetMaskDecSize_h; //!< Mask output for each target.
    float *m_batchMaskDecSize_d; //!< Mask output in a batch.
    float *m_targetMaskOriginalSrcSize_d; //!< Mask for each target resized to original streammux size.
    float *m_kInv_d, *m_tFormInv_d, *m_scaleNormalizedMeanLimbLengths_d, *m_meanLimbLengths_d; //!< Device inverse intrinsics, inverse transforms, and limb-length priors.
    float *m_pose2d_d, *m_pose2d_org_img_d, *m_pose25d_d, *m_pose3d_d; //!< Device 2D, original-image 2D, 2.5D, and 3D poses.
    float *m_pose25d_h, *m_pose3d_h; //!< Host 2.5D and 3D poses.
    float *m_outputIou_d; //!< Device output IoU.
    PoseInferenceConfigParams m_params; //!< Config params for reid module.
    NvMotDataTypes::Size m_srcImageSize; //!< Size of the source image.
    NvMotDataTypes::Size m_originalSrcImageSize; //!< Size of the original source image.
    std::vector<std::tuple<uint, NvMOTStreamId, NvMotDataTypes::Rect2f>> m_targetStreamIdList; //!< Target index, stream ID, and crop ROI for each box.

    cudaStream_t m_cuStream; //!< Cuda stream to run all GPU tasks.
    std::vector<void*> m_bindings; //!< Pointers to TensorRT tensors.
    float *m_featuresEncoder_d; //!< Encoder output features.
    std::shared_ptr<NvTrackedObjectManager> m_targetManager; //!< Target manager providing target status to update feature gallery.
    std::shared_ptr<ReIDFrameTransformTexture> m_subImageCropper2; //!< Image cropper, may share with visual tracker in the future.
    std::shared_ptr<ReIDCropScaler> m_subImageCropScaler; //!< Image crop scaler.
    //!< Uses VPI crop scaler.
    std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f>>> m_inputRoisInBatch; //!< Reid input rois to crop.
    TrtModule m_inference; //!< Inference module.
    int m_maxTargetsToTrack; //!< Max number of targets to keep for all streams.

    /**
     * @brief Rescale all tracker frames into encoder input.
     * @param[in] inputMeta Input frames keyed by stream ID.
     */
    void extractImagePatchesBatch(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta);

    /** @brief Upload fixed geometry tensors used by pose inference. */
    void setValues4PoseGeometryInput();
};

#endif // POSEESTIMATORLIB_POSEINFERENCE_HPP_
