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

// Runs TensorRT pose inference on tracked crops and stores 2.5D and 3D joint poses.
#include "modules/PoseEstimator/PoseInference.hpp"
#include "modules/utility/debug_common.hpp"
#include "modules/utility/cuda_common.hpp"

using std::cout;
using std::endl;

/**
 * @brief Configure dynamic input shapes for pose inference.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4Pose(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

    for (int ind = 0; ind < network->getNbInputs(); ind++)
    {
        nvinfer1::ITensor* input = network->getInput(ind);

        const char* inputName = input->getName();
        nvinfer1::Dims inputDim = input->getDimensions();
        inputDim.d[0] = 1;
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, inputDim);
        inputDim.d[0] = batchSize;
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, inputDim);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, inputDim);

        inputDim.d[0] = -1;
        input->setDimensions(inputDim);
    }

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for pose inference.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4Pose(TrtModule *trtModule, const std::vector<int>& sizes)
{
    for (int ind = 0; ind < trtModule->m_engine->getNbIOTensors(); ++ind)
    {
        const char* tensorName = trtModule->m_engine->getIOTensorName(ind);
        if (trtModule->m_engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
        {

            nvinfer1::Dims lastInputDims = trtModule->m_context->getTensorShape( tensorName );
            lastInputDims.d[0] = sizes.front();
            trtModule->m_context->setInputShape(tensorName, lastInputDims);
        }
    }
}

/**
 * @brief Construct a PoseInference instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
PoseInference::PoseInference(const std::string &configFilePath)
{
    m_params = PoseInferenceConfigParams(configFilePath, "PoseEstimator");

    m_inference.m_params = TRTConfigParams(configFilePath, "PoseEstimator", "");
    m_inference.initializeModel(&createOnnxProfile4Pose, &setRuntimeDim4Pose);

    m_isRuntimeInit = false;
}

/**
 * @brief Construct a PoseInference instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] modelName Configured model name.
 */
PoseInference::PoseInference(const std::string &configFilePath, const std::string &modelName)
{
    m_params = PoseInferenceConfigParams(configFilePath, "PoseEstimator");

    m_inference.m_params = TRTConfigParams(configFilePath, "PoseEstimator", "");
    m_inference.initializeModel(&createOnnxProfile4Pose, &setRuntimeDim4Pose);

    m_isRuntimeInit = false;
}

/**
 * @brief Initialize runtime resources for the configured source.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void PoseInference::initializeRuntime(
    const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
    const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{

    m_targetManager = targetManager;

    m_maxTargetsToTrack = frameInfo.maxStreams * maxTargetsPerStream;
    if(m_params.m_useVPICropScaler)
    {
        m_subImageCropScaler = std::make_shared<ReIDCropScaler>();
        m_subImageCropScaler->initialize(
            m_params.m_debugVerboseLevel, frameInfo.maxStreams, frameInfo.maxBufSurfAddrSize,
            m_maxTargetsToTrack, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
            NvMotDataTypes::Size(m_params.m_width, m_params.m_height), m_params.m_colorFormat, m_params.m_inputOrder
        );
        m_subImageCropper2 = nullptr;
    }
    else
    {
        m_subImageCropper2 = std::make_shared<ReIDFrameTransformTexture>();
        m_subImageCropper2->initialize(
            m_params.m_debugVerboseLevel, frameInfo.maxStreams, frameInfo.maxBufSurfAddrSize,
            m_maxTargetsToTrack, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
            NvMotDataTypes::Size(m_params.m_width, m_params.m_height), m_params.m_colorFormat, m_params.m_inputOrder
        );
        m_subImageCropScaler = nullptr;
    }

    CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );
    m_inference.setCudaStream(m_cuStream);

    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_kInv_d), m_params.m_maxBatchSize * 9 * sizeof(float))));
    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_tFormInv_d), m_params.m_maxBatchSize * 9  * sizeof(float))));
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_scaleNormalizedMeanLimbLengths_d),
            m_params.m_maxBatchSize * 36 * sizeof(float)
        ))
    );
    CUDA_CHECK(
        (cudaMalloc(reinterpret_cast<void **>( &m_meanLimbLengths_d), m_params.m_maxBatchSize * 36 * sizeof(float)))
    );

    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_pose2d_d), m_params.m_maxBatchSize * 34 * 3 * sizeof(float))));
    CUDA_CHECK(
        (cudaMalloc(reinterpret_cast<void **>( &m_pose2d_org_img_d), m_params.m_maxBatchSize * 34 * 3  * sizeof(float)))
    );
    CUDA_CHECK(
        (cudaMalloc(reinterpret_cast<void **>( &m_pose25d_d), m_params.m_maxBatchSize * 34 * 4 * sizeof(float)))
    );
    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_pose3d_d), m_params.m_maxBatchSize * 34 * 3 * sizeof(float))));

    CUDA_CHECK(
        (cudaMallocHost(reinterpret_cast<void **>( &m_pose25d_h), m_params.m_maxBatchSize * 34 * 4 * sizeof(float)))
    );
    CUDA_CHECK(
        (cudaMallocHost(reinterpret_cast<void **>( &m_pose3d_h), m_params.m_maxBatchSize * 34 * 3 * sizeof(float)))
    );

    setValues4PoseGeometryInput();

    m_isRuntimeInit = true;
}

/** @brief Create or load the configured inference model. */
void PoseInference::initializeModel()
{

}

/** @brief Upload fixed geometry tensors used by pose inference. */
void PoseInference::setValues4PoseGeometryInput()
{
    const std::vector<float> scale_normalized_mean_limb_lengths = {
        0.5000, 0.5000, 1.0000, 0.8175, 0.9889, 0.2610, 0.7942, 0.5724, 0.5078,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3433, 0.8171,
        0.9912, 0.2610, 0.8259, 0.5724, 0.5078, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.3422, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    };
    const std::vector<float> mean_limb_lengths = {
        246.3427, 246.3427, 492.6854, 402.4380, 487.0321, 128.6856, 391.6295,
        281.9928, 249.9478,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000, 169.1832, 402.2611, 488.1824, 128.6848, 407.5836,
        281.9897, 249.9489,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000, 168.6137,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000
    };

    /// The inverse calibration matrix changes with the camera parameters.
    const std::vector<float> k_inv = {
        0.00124876620338, 0, -0.119881555525,
        0, 0.00124876620338, -0.159842074033,
        0, 0, 1
    };

    const std::vector<float> t_form_inv = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    for (int oi=0; oi < m_params.m_maxBatchSize; oi++)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_scaleNormalizedMeanLimbLengths_d + oi * 36, scale_normalized_mean_limb_lengths.data(),
                36 * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_meanLimbLengths_d + oi * 36, mean_limb_lengths.data(),
                36 * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync( m_kInv_d + oi * 9, k_inv.data(), 9 * sizeof(float), cudaMemcpyHostToDevice, m_cuStream)
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_tFormInv_d + oi * 9, t_form_inv.data(), 9 * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
            )
        );
    }
}

/** @brief Release resources owned by the PoseInference instance. */
PoseInference::~PoseInference()
{
    if (m_isRuntimeInit)
    {
        CUDA_CHECK( cudaFree(m_kInv_d) );
        CUDA_CHECK( cudaFree(m_tFormInv_d) );
        CUDA_CHECK( cudaFree(m_scaleNormalizedMeanLimbLengths_d) );
        CUDA_CHECK( cudaFree(m_meanLimbLengths_d) );
        CUDA_CHECK( cudaFree(m_pose2d_d) );
        CUDA_CHECK( cudaFree(m_pose2d_org_img_d) );
        CUDA_CHECK( cudaFree(m_pose25d_d) );
        CUDA_CHECK( cudaFree(m_pose3d_d) );

        CUDA_CHECK( cudaFreeHost(m_pose25d_h) );
        CUDA_CHECK( cudaFreeHost(m_pose3d_h) );

        m_subImageCropper2.reset();
        m_subImageCropScaler.reset();
    }
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void PoseInference::queryParams(NvMOTQuery &paramsQuery)
{
    paramsQuery.numTransforms = 1;
}

/**
 * @brief Crop target regions from the input frame batch.
 * @param[in] inputMeta Input frames keyed by stream ID.
 */
void PoseInference::extractImagePatchesBatch(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta)
{
    m_inputRoisInBatch.clear();
    m_numBboxes = 0;
    m_targetStreamIdList.clear();
    /// Collect the crop regions of the targets in each stream.
    for (auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
    {
        const NvMOTStreamId videoStreamId = streamIt->first;
        const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList = \
            m_targetManager->getTrackedObjectsInUseList(videoStreamId);

        if (targetList.empty())
        {
            continue;
        }

        for (auto targetIt = targetList.begin(); targetIt != targetList.end(); targetIt++)
        {
            const std::shared_ptr<NvTrackedObject> target = targetIt->second;

            /// Skip classes not in operateOnClassIds list (optimization to avoid unnecessary inference).
            if (!m_params.m_operateOnClassIds.empty() &&
                std::find(m_params.m_operateOnClassIds.begin(), m_params.m_operateOnClassIds.end(),
                          target->getClassId()) == m_params.m_operateOnClassIds.end())
            {
                continue;
            }

            if (target->getTrackerState() != TRACKER_STATE::TENTATIVE && target->getTrackerState() != TRACKER_STATE::ACTIVE)
            {
                continue;
            }

            if (target->needTargetIDAssigned() || target->getTrackerState() == TRACKER_STATE::TENTATIVE ||
                (m_params.m_poseInferenceInterval >= 0 && target->getAge() % (m_params.m_poseInferenceInterval + 1) == 0))
            {
                NvMotDataTypes::Rect2f tBbox;
                if (target->getTargetInfo().matchedDetObjInfo != nullptr)
                {
                    const DetectedObjInfo* detObjInfo = target->getTargetInfo().matchedDetObjInfo;
                    tBbox = detObjInfo->origBbox;
                }
                else
                {
                    tBbox = target->getTargetInfo().tBbox;
                }

                float xCenter = tBbox.min()(0) + tBbox.sizes()(0) * 0.5;
                float yCenter = tBbox.min()(1) + tBbox.sizes()(1) * 0.5;
                float wRoi = 1.25 * std::max(tBbox.sizes()(0), tBbox.sizes()(1));
                float hRoi = wRoi / m_params.m_width * m_params.m_height;
                NvMotDataTypes::Rect2f roi = CV_FUNCS::constructRect2f(
                    xCenter - wRoi * 0.5, yCenter - hRoi * 0.5, wRoi, hRoi
                );

                m_targetStreamIdList.push_back(std::make_tuple(target->getTargetInd(), videoStreamId, roi));
                m_inputRoisInBatch[videoStreamId].push_back(std::make_pair(m_numBboxes++, roi));
            }
        }
    }

    if (m_numBboxes == 0)
    {
        return;
    }

    if(m_params.m_useVPICropScaler)
    {
        /// Crop and scale input frames using VPI crop scaler, perform format conversion.
        if (!m_subImageCropScaler->frameTransformInBatch(m_inputRoisInBatch,
                                                         m_numBboxes - 1, m_params.m_netScaleFactor, m_params.m_offsets, inputMeta))
        {
            throw std::runtime_error("[FAILED] m_subImageCropScaler->frameTransformInBatch()");
        }

        m_inputBuf_d = m_subImageCropScaler->getOutputImagePatchPtrs();
    }
    else
    {
        /// Crop and scale input frames, perform format conversion.
        /// Prepare input texture object.
        m_subImageCropper2->updateTextureObject(inputMeta);

        /// Crop, rescale and convert image format.
        /// May change dimension order, perform image normalization and convert to float.
        if (m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
        {
            if (!m_subImageCropper2->frameTransformInBatch_NV12(m_inputRoisInBatch,
                                                                m_numBboxes - 1, m_params.m_netScaleFactor, m_params.m_offsets))
            {
                throw std::runtime_error("[FAILED] m_subImageCropper2->formatConvert_batch_NV12()");
            }
        }
        else if (m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
        {
            if (!m_subImageCropper2->frameTransformInBatch(m_inputRoisInBatch,
                                                            m_numBboxes - 1, m_params.m_netScaleFactor, m_params.m_offsets))
            {
                throw std::runtime_error("[FAILED] m_subImageCropper2->formatConvert_batch()");
            }
        }
        else
        {
            throw std::runtime_error("[FAILED] Invalid color format");
        }

        /// Resized image size: [num_frames_current_batch, C, m_hiddenDim, m_hiddenDim] or num_frames_current_batch, m_hiddenDim, m_hiddenDim, C].
        m_inputBuf_d = m_subImageCropper2->getOutputImagePatchPtrs();
    }
}

/**
 * @brief Run inference for the targets in the input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in,out] detectedObjects Newly detected objects in each stream.
 * @param[in] isForReAssoc Whether this inference is for reassociation.
 */
void PoseInference::doInference(
    const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta,
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
)
{
    extractImagePatchesBatch(inputMeta);

    #ifdef VIZ_DEBUG_POSE
        std::map<NvMOTStreamId, std::map<uint, cv::Mat>> srcImages;
        if (m_params.m_saveFrameViz)
        {
            if(m_params.m_useVPICropScaler)
            {
                srcImages = m_subImageCropScaler->getImagePatches2Host_OCV_batch(
                    m_params.m_netScaleFactor, m_params.m_offsets
                );
            }
            else
            {
                srcImages = m_subImageCropper2->getImagePatches2Host_OCV_batch(
                    m_params.m_netScaleFactor, m_params.m_offsets
                );
            }
        }
        std::map<uint, std::vector<float>> poseVizMap;
    #endif

    m_bindings.resize(9);
    m_bindings[0] = nullptr;
    m_bindings[1] = m_kInv_d;
    m_bindings[2] = m_tFormInv_d;
    m_bindings[3] = m_scaleNormalizedMeanLimbLengths_d;
    m_bindings[4] = m_meanLimbLengths_d;
    m_bindings[5] = m_pose2d_d;
    m_bindings[6] = m_pose2d_org_img_d;
    m_bindings[7] = m_pose25d_d;
    m_bindings[8] = m_pose3d_d;

    for (int dBboxInd = 0; dBboxInd < m_numBboxes; dBboxInd += m_params.m_maxBatchSize)
    {

        /// Limit the last batch to the number of remaining boxes.
        int batchSize = std::min(m_params.m_maxBatchSize, m_numBboxes - dBboxInd);
        /// Binding input and output address to nvinfer engine.
        m_bindings[0] = (void*) (m_inputBuf_d + dBboxInd * m_params.m_channel *\
            m_params.m_height * m_params.m_width);

        m_inference.submitInput(m_bindings, {batchSize}, m_cuStream);

        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pose25d_h, m_pose25d_d, batchSize * 34 * 4 * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pose3d_h, m_pose3d_d, batchSize * 34 * 3 * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
            )
        );

        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );

        for (int bi = 0; bi < batchSize; bi++)
        {
            uint targetInd = std::get<0>(m_targetStreamIdList.at(dBboxInd + bi));
            NvMOTStreamId videoStreamId = std::get<1>(m_targetStreamIdList.at(dBboxInd + bi));
            const NvMotDataTypes::Rect2f& roi = std::get<2>(m_targetStreamIdList.at(dBboxInd + bi));
            const std::map<uint32_t, std::shared_ptr<NvTrackedObject>>& targetList = \
                m_targetManager->getTrackedObjectsInUseList(videoStreamId);
            const std::shared_ptr<NvTrackedObject> target = targetList.at(targetInd);
            m_inputRoisInBatch[videoStreamId].push_back(std::make_pair(target->getTargetInd(), roi));
            #ifdef VIZ_DEBUG_POSE
                poseVizMap[targetInd] = std::vector<float>(
                    m_pose25d_h + bi * 34 * 4, m_pose25d_h + bi * 34 * 4 + 34 * 4
                );
            #endif
            /// Rescale x, y to input frame coordinates.
            for (int ki=0; ki < 34; ki++)
            {
                m_pose25d_h[bi * 34 * 4 + ki * 4] = roi.min()(0) + \
                m_pose25d_h[bi * 34 * 4 + ki * 4] / m_params.m_width * roi.sizes()(0);
                m_pose25d_h[bi * 34 * 4 + ki * 4 + 1] = roi.min()(1) + \
                m_pose25d_h[bi * 34 * 4 + ki * 4 + 1] / m_params.m_height * roi.sizes()(1);
            }
            target->setPose25d(m_pose25d_h + bi * 34 * 4, 34 * 4);
            target->setPose3d(m_pose25d_h + bi * 34 * 3, 34 * 3);
        }
    }

    #ifdef VIZ_DEBUG_POSE
        if (m_params.m_saveFrameViz && m_numBboxes > 0)
        {
            std::vector<int> idxJointsUpper = {
                 0,  1,  2,  3,  6, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
            };
            std::vector<int> idxBonesUpper = {
                21,  6, 20,  6, 21, 23, 20, 22, 24, 22, 23, 25,
                27, 25, 31, 25, 33, 25, 29, 25, 24, 30, 24, 26,
                24, 32, 24, 28,  2, 21,  1, 20,  3,  6,  6, 15,
                15, 16, 15, 17, 19, 17, 18, 16,  0,  1,  0,  2,
                 0,  3
            };

            std::vector<int> idxJointsLower = {4, 5, 7, 8, 9, 10, 11, 12, 13, 14};
            std::vector<int> idxBonesLower = {
                 2,  5,  5,  8,  1,  4,  4,  7,  7, 13,
                 8, 14,  8, 10,  7,  9, 11,  9, 12, 10
            };

            int bboxInd = 0;
            for(auto it = srcImages.begin(); it != srcImages.end(); ++it)
            {
                NvMOTStreamId streamId = it->first;
                auto frameNum = inputMeta.at(streamId)->frameNum;
                for(auto jt=it->second.begin(); jt != it->second.end(); ++jt, bboxInd++)
                {
                    cv::Mat subImg = jt->second;

                    uint targetInd = std::get<0>(m_targetStreamIdList.at(bboxInd));
                    const std::map<uint32_t, std::shared_ptr<NvTrackedObject>>& targetList = \
                        m_targetManager->getTrackedObjectsInUseList(streamId);
                    const std::shared_ptr<NvTrackedObject> target = targetList.at(targetInd);
                    const std::vector<float> pose25d = poseVizMap.at(targetInd);

                    for (auto bi = 0; bi < idxBonesUpper.size(); bi += 2)
                    {
                        int ind0 = idxBonesUpper.at(bi), ind1 = idxBonesUpper.at(bi + 1);
                        cv::Point pt0 = cv::Point(pose25d.at(ind0 * 4), pose25d.at(ind0 * 4 + 1));
                        cv::Point pt1 = cv::Point(pose25d.at(ind1 * 4), pose25d.at(ind1 * 4 + 1));
                        cv::line(subImg, pt0, pt1, cv::Scalar(0, 255, 0), 2);
                    }

                    for (auto bi = 0; bi < idxBonesLower.size(); bi += 2)
                    {
                        int ind0 = idxBonesLower.at(bi), ind1 = idxBonesLower.at(bi + 1);
                        cv::Point pt0 = cv::Point(pose25d.at(ind0 * 4), pose25d.at(ind0 * 4 + 1));
                        cv::Point pt1 = cv::Point(pose25d.at(ind1 * 4), pose25d.at(ind1 * 4 + 1));
                        cv::line(subImg, pt0, pt1, cv::Scalar(0, 255, 0), 2);
                    }

                    for (auto ji = 0; ji < idxJointsUpper.size(); ji++)
                    {
                        int ind = idxJointsUpper.at(ji);
                        cv::Point pt = cv::Point(pose25d.at(ind * 4), pose25d.at(ind * 4 + 1));
                        float conf = pose25d.at(ind * 4 + 3);
                        conf = std::min(1.0f, std::max(0.0f, conf));
                        cv::circle(subImg, pt, 2, cv::Scalar(0, 0, int(conf*255)), 2);

                    }

                    for (auto ji = 0; ji < idxJointsLower.size(); ji++)
                    {
                        int ind = idxJointsLower.at(ji);
                        cv::Point pt = cv::Point(pose25d.at(ind * 4), pose25d.at(ind * 4 + 1));
                        float conf = pose25d.at(ind * 4 + 3);
                        conf = std::min(1.0f, std::max(0.0f, conf));
                        cv::circle(subImg, pt, 2, cv::Scalar(0, 0, int(conf*255)), 2);
                        if (ind == 9)
                        {
                            cv::putText(
                                subImg, std::to_string(conf).substr(0, 3).c_str(), pt,
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, int(conf*255)), 2
                            );
                        }
                    }

                    std::string outDir = m_params.m_vizDir;
                    std::string label = "s" + std::to_string(streamId) + \
                        "_" + std::to_string(frameNum) + "_" + \
                        std::to_string(bboxInd) + ".png";

                    cv::imwrite(outDir + label, subImg);
                }
            }
        }
    #endif
}

/** @brief Update retained state after data association. */
void PoseInference::update()
{

}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void PoseInference::removeStream(const NvMOTStreamId videoStreamId)
{

}

/** @brief Construct a PoseInferenceConfigParams instance. */
PoseInferenceConfigParams::PoseInferenceConfigParams()
{
    /// Set the default pose-network input size and color normalization.
    m_hiddenDim = 256;
    m_encoderFeatureDim = 64;
    m_inputOrder = 0;
    m_width = 192;
    m_height = 256;
    m_inferDims = {3, 256, 192};
    m_channel = 3;
    m_gpuId = 0;
    m_netScaleFactor = 0.00392156;
    /// Rgb.
    m_offsets = {0.0, 0.0, 0.0};
    m_useVPICropScaler = false;
    m_maxBatchSize = 1;
    m_poseInferenceInterval = -1;

    m_vizDir = "trackerViz/";
    m_saveFrameViz = false;
    m_colorFormat = NVBUF_COLOR_FORMAT_BGR;
    m_debugVerboseLevel = 0;
}

/**
 * @brief Construct a PoseInferenceConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 */
PoseInferenceConfigParams::PoseInferenceConfigParams(const std::string &configFilePath, const std::string moduleName)
{
    /// Set the default values.
    *this = PoseInferenceConfigParams();

    YAML::Node configRoot = getConfigRoot(configFilePath);
    /// Read the module parameters from the configuration file.
    this->read(configRoot[moduleName]);

    /// Interpret inferDims according to the input order.
    if (m_inputOrder == 0)
    {
        m_channel = m_inferDims[0], m_height = m_inferDims[1], m_width = m_inferDims[2];
    }
    else if(m_inputOrder == 1)
    {
        m_height = m_inferDims[0], m_width = m_inferDims[1], m_channel = m_inferDims[2];
    }
    if (m_channel != 3 || m_width <= 0 || m_height <= 0)
    {
        throw std::runtime_error("inferDims channel should be 3, widht and height should be positive int");
    }
}

/**
 * @brief Construct a PoseInferenceConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
PoseInferenceConfigParams::PoseInferenceConfigParams(const YAML::Node &fn)
{
    read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void PoseInferenceConfigParams::read(const YAML::Node &fn)
{
    if(!fn.size())
    {
        return;
    }

    for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
        std::string paramKey = fit->first.as<std::string>();
        if (paramKey == "debugVerboseLevel")
        {
            m_debugVerboseLevel = (int)fit->second.as<float>();
        }
        else if (paramKey == "hiddenDim")
        {
            m_hiddenDim = fit->second.as<float>();
        }
        else if (paramKey == "inferDims")
        {
            m_inferDims.clear();
            /// Interpret later because the input order is unknown at this time.
            for (std::size_t i=0; i<fit->second.size(); i++)
            {
                m_inferDims.push_back((int)fit->second[i].as<float>());
            }
            /// Check network dimension size.
            if (m_inferDims.size() != 3)
            {
                throw std::runtime_error("inferDims should be list of positive int with length 3");
            }
        }
        else if (paramKey == "inputOrder")
        {
            int inputOrder = (int)fit->second.as<float>();
            if (inputOrder == 0 || inputOrder == 1)
            {
                m_inputOrder = inputOrder;
            }
            else
            {
                throw std::runtime_error("inputOrder should be 0 or 1");
            }
        }
        else if (paramKey == "encoderFeatureDim")
        {
            m_encoderFeatureDim = fit->second.as<float>();
        }
        else if (paramKey == "netScaleFactor")
        {
            m_netScaleFactor = fit->second.as<float>();
        }
        else if (paramKey == "offsets")
        {
            m_offsets.clear();
            for (std::size_t i=0; i<fit->second.size(); i++)
            {
                m_offsets.push_back(fit->second[i].as<float>());
            }
            if (m_offsets.size() != 3)
            {
                throw std::runtime_error("offsets should have length 3");
            }
        }
        else if (paramKey == "colorFormat")
        {
            int colorFormat = (int)fit->second.as<float>();
            if (colorFormat == 0)
            {
                m_colorFormat = NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGB;
            }
            else if (colorFormat == 1)
            {
                m_colorFormat = NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_BGR;
            }
            else
            {
                throw std::runtime_error("colorFormat should be 0 or 1");
            }
        }
        else if (paramKey == "useVPICropScaler")
        {
            m_useVPICropScaler = (int)fit->second.as<float>();
        }
        else if (paramKey == "poseInferenceInterval")
        {
            m_poseInferenceInterval = fit->second.as<float>();
        }
        else if (paramKey == "vizDir")
        {
            m_vizDir = fit->second.as<std::string>();
        }
        else if (paramKey == "saveFrameViz")
        {
            m_saveFrameViz = fit->second.as<float>();
        }
        else if (paramKey == "operateOnClassIds")
        {
            m_operateOnClassIds.clear();
            for (std::size_t i=0; i<fit->second.size(); i++)
            {
                m_operateOnClassIds.push_back((int)fit->second[i].as<float>());
            }
        }
    }
}
