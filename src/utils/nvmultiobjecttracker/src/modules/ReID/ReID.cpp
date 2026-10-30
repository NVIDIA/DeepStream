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

#include <fstream>

#ifdef VIZ_DEBUG_REID
// Helper for viz debug.
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#endif

#include "modules/utility/debug_common.hpp"
#include "modules/utility/cuda_common.hpp"
#include "modules/ReID/ReID.hpp"
#include "modules/ReID/cuReID.h"
#include "modules/ReID/ReIDFeatureGallery.hpp"

using std::cout;
using std::endl;

/**
 * @brief Configure dynamic input shapes for ReID inference.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4Reid(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

    nvinfer1::ITensor* input = network->getInput(0);

    const char* inputName = input->getName();
    nvinfer1::Dims inputDim = input->getDimensions();
    inputDim.d[0] = 1;
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, inputDim);
    inputDim.d[0] = batchSize;
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, inputDim);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, inputDim);

    inputDim.d[0] = -1;
    input->setDimensions(inputDim);

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for ReID inference.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4Reid(TrtModule *trtModule, const std::vector<int>& sizes)
{
    int batchSize = sizes.front();
    for (int ind = 0; ind < trtModule->m_engine->getNbIOTensors(); ++ind)
    {
        if (trtModule->m_engine->getTensorIOMode(trtModule->m_engine->getIOTensorName(ind)) == nvinfer1::TensorIOMode::kINPUT)
        {

            nvinfer1::Dims lastInputDims = trtModule->m_context->getTensorShape(
                trtModule->m_engine->getIOTensorName(ind)
            );
            lastInputDims.d[0] = batchSize;
            trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(ind), lastInputDims);
        }
    }
}

/**
 * @brief Construct a ReID instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
ReID::ReID(const std::string &configFilePath)
{
    m_params = ReIDConfigParams(configFilePath, "ReID", "");
    m_calDisMaha = true;

    #ifdef VIZ_DEBUG_REID
        m_extFrameNum = 0;
    #endif

    m_inference.m_params = TRTConfigParams(configFilePath, "ReID", "");
    m_inference.initializeModel(&createOnnxProfile4Reid, &setRuntimeDim4Reid);

    m_bindings.resize(2, nullptr);
    m_inputInd = 0;
    m_outputInd = 1;

    m_isModelInit = false;
    m_isRuntimeInit = false;
}

/**
 * @brief Construct a ReID instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] modelName Configured model name.
 * @param[in] calDisMaha If Mahalanobis distance needs to be computed.
 */
ReID::ReID(const std::string &configFilePath, const std::string &modelName, const bool calDisMaha)
{
    m_params = ReIDConfigParams(configFilePath, "ReID", modelName);
    m_calDisMaha = calDisMaha;

    #ifdef VIZ_DEBUG_REID
        m_extFrameNum = 0;
    #endif

    m_inference.m_params = TRTConfigParams(configFilePath, "ReID", modelName);
    m_inference.initializeModel(&createOnnxProfile4Reid, &setRuntimeDim4Reid);

    m_bindings.resize(2, nullptr);
    m_inputInd = 0;
    m_outputInd = 1;

    m_isModelInit = true;
    m_isRuntimeInit = false;
}

/** @brief Create or load the configured inference model. */
void ReID::initializeModel()
{
}

/**
 * @brief Initialize runtime resources for the configured source.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void ReID::initializeRuntime(
    const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
    const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
    m_maxTargetsToTrack = frameInfo.maxStreams * maxTargetsPerStream;
    m_maxTargetsPerStream = maxTargetsPerStream;
    m_maxStreams = frameInfo.maxStreams;
    m_srcImageSize = frameInfo.srcImageSize;

    /// Max size of dot product between detected re-id features and feature gallery for each stream.
    m_maxReidProductSizePerStream = int(maxTargetsPerStream * 1.1 * \
			maxTargetsPerStream * 1.1 * m_params.m_reidHistorySize);

    m_targetManager = targetManager;

    if(m_params.m_useVPICropScaler)
    {
        m_subImageCropScaler = std::make_shared<ReIDCropScaler>();
        m_subImageCropScaler->initialize(
            m_params.m_debugVerboseLevel, frameInfo.maxStreams, frameInfo.maxBufSurfAddrSize,
            m_maxTargetsToTrack, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
            NvMotDataTypes::Size(m_params.m_width, m_params.m_height), m_params.m_colorFormat, m_params.m_inputOrder
        );
        m_subImageCropper2 = NULL;
    }
    else
    {
        /// Otherwise initialize a new image cropper.
        m_subImageCropper2 = std::make_shared<ReIDFrameTransformTexture>();
        m_subImageCropper2->initialize(
            m_params.m_debugVerboseLevel, frameInfo.maxStreams, frameInfo.maxBufSurfAddrSize,
            m_maxTargetsToTrack, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
            NvMotDataTypes::Size(m_params.m_width, m_params.m_height), m_params.m_colorFormat, m_params.m_inputOrder
        );
        m_subImageCropScaler = NULL;
    }

    if (m_params.m_debugVerboseLevel > 2)
    {
        cout << "[ReID::initializeRuntime] cropper init is done" << endl;
    }

    m_reidFeatureGallery = std::make_shared<ReIDFeatureGallery>();
    m_reidFeatureGallery->initialize(
        m_params.m_reidFeatureSize, m_params.m_reidHistorySize,
        frameInfo.maxStreams, maxTargetsPerStream, m_params.m_debugVerboseLevel
    );

    /// Create cuda, cublas resources.
    CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );
    CUBLAS_CHECK( cublasCreate(&m_cublasHandle) );
    CUBLAS_CHECK( cublasSetStream(m_cublasHandle, m_cuStream) );

    m_inference.setCudaStream(m_cuStream);

    /// Allocate re-id TRT output.
    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>( &m_reidFeaturesDet_d ),
            m_maxTargetsToTrack * m_params.m_reidFeatureSize * sizeof(float)
        )
    );

    /// Allocate re-id similarity computation arrays.
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_reidProduct_d),
            frameInfo.maxStreams * m_maxReidProductSizePerStream * sizeof(float)
        ))
    );
    CUDA_CHECK(
        (cudaMallocHost(
            reinterpret_cast<void **>( &m_reidProduct_h),
            frameInfo.maxStreams * m_maxReidProductSizePerStream * sizeof(float)
        ))
    );

    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_targetReidFeatureInds_d),
            m_maxTargetsToTrack * m_params.m_reidHistorySize * sizeof(int)
        ))
    );
    CUDA_CHECK(
        (cudaMallocHost(
            reinterpret_cast<void **>( &m_targetReidFeatureInds_h),
            m_maxTargetsToTrack * m_params.m_reidHistorySize * sizeof(int)
        ))
    );

    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_reidProductSize_d), m_maxStreams * sizeof(int2))));
    CUDA_CHECK((cudaMallocHost(reinterpret_cast<void **>( &m_reidProductSize_h), m_maxStreams * sizeof(int2))));

    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_targetWork_1_d), m_maxTargetsToTrack * sizeof(int))));
    CUDA_CHECK((cudaMallocHost(reinterpret_cast<void **>( &m_targetWork_1_h), m_maxTargetsToTrack * sizeof(int))));

    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_targetWork_2_d), m_maxTargetsToTrack * sizeof(int))));
    CUDA_CHECK((cudaMallocHost(reinterpret_cast<void **>( &m_targetWork_2_h), m_maxTargetsToTrack * sizeof(int))));

    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_targetWork_3_d), m_maxTargetsToTrack * sizeof(int))));
    CUDA_CHECK((cudaMallocHost(reinterpret_cast<void **>( &m_targetWork_3_h), m_maxTargetsToTrack * sizeof(int))));

    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_confidenceReidDetTargets_d),
            m_maxTargetsToTrack * m_maxTargetsToTrack * sizeof(float)
        ))
    );
    CUDA_CHECK(
        (cudaMallocHost(
            reinterpret_cast<void **>( &m_confidenceReidDetTargets_h),
            m_maxTargetsToTrack * m_maxTargetsToTrack * sizeof(float)
        ))
    );
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_nearestIndReidDetTargets_d),
            m_maxTargetsToTrack * m_maxTargetsToTrack * sizeof(int)
        ))
    );
    CUDA_CHECK(
        (cudaMallocHost(
            reinterpret_cast<void **>( &m_nearestIndReidDetTargets_h),
            m_maxTargetsToTrack * m_maxTargetsToTrack * sizeof(int)
        ))
    );

    memset(m_reidProductSize_h, 0, m_maxStreams*sizeof(int2));

    if (m_params.m_useSegmentationMask4Input)
    {
        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>( &m_inputRoisInBatch_d ),
                m_maxTargetsToTrack * m_params.m_width * m_params.m_height * sizeof(float)
            )
        );
        CUDA_CHECK(
            cudaMallocHost(
                reinterpret_cast<void**>( &m_inputRoisInBatch_h ),
                m_maxTargetsToTrack * m_params.m_width * m_params.m_height * sizeof(float)
            )
        );

        CUDA_CHECK(cudaMalloc( reinterpret_cast<void**>( &m_targetInds4Rois_d ), m_maxTargetsToTrack * sizeof(int)));
        CUDA_CHECK(
            cudaMallocHost( reinterpret_cast<void**>( &m_targetInds4Rois_h ), m_maxTargetsToTrack * sizeof(int))
        );
    }

    CUDA_CHECK(
        cudaMallocHost(
            reinterpret_cast<void**>( &m_reidFeaturesDet_h ),
            m_maxTargetsToTrack * m_params.m_reidFeatureSize * sizeof(float)
        )
    );
    #ifdef VIZ_DEBUG_REID
        m_timeTotal = 0.0;
        m_timeCnt = 0;

        if (m_params.m_useExternalImages)
        {
            CUDA_CHECK (
                cudaMalloc(
                    reinterpret_cast<void**>(&m_inputBuf_d),
                    m_maxTargetsToTrack * m_params.m_height * m_params.m_width * m_params.m_channel * sizeof(float)
                )
            );
            CUDA_CHECK (
                cudaMallocHost(
                    reinterpret_cast<void**>(&m_inputBuf_h),
                    m_maxTargetsToTrack * m_params.m_height * m_params.m_width * m_params.m_channel * sizeof(float)
                )
            );
        }
    #endif
    if ((m_params.m_reidType & REID_TYPES::NvDEEPSORT) && m_calDisMaha)
    {
        m_disMahaThreadWorker = std::make_shared<NvTrackerMultiThreadWorker<DisMahalanobisIn>>();
		{
			/// Initialize multi-thread worker to perform data association and target update.
			m_disMahaThreadWorker->initialize(
				1, [this](DisMahalanobisIn in) {            this->computeMahalanobisDist(in); }
			);
		}
    }

    if (m_params.m_addFeatureNormalization)
    {
        m_normFeatureCudaBlockSize = 64;
        while (m_normFeatureCudaBlockSize < m_params.m_reidFeatureSize)
        {
            m_normFeatureCudaBlockSize *= 2;
        }
    }

    m_isRuntimeInit = true;
}

/** @brief Release resources owned by the ReID instance. */
ReID::~ReID()
{
    if (m_isRuntimeInit)
    {
        /// ReID network buffer.
        CUDA_CHECK(cudaFree(m_reidFeaturesDet_d));

        /// ReID distance array.
        CUDA_CHECK( cudaFree(m_reidProduct_d) );
		CUDA_CHECK( cudaFreeHost(m_reidProduct_h) );

        CUDA_CHECK( cudaFree(m_targetReidFeatureInds_d) );
		CUDA_CHECK( cudaFreeHost(m_targetReidFeatureInds_h) );

        CUDA_CHECK( cudaFree(m_reidProductSize_d) );
		CUDA_CHECK( cudaFreeHost(m_reidProductSize_h) );

        CUDA_CHECK( cudaFree(m_targetWork_1_d) );
		CUDA_CHECK( cudaFreeHost(m_targetWork_1_h) );

        CUDA_CHECK( cudaFree(m_targetWork_2_d) );
		CUDA_CHECK( cudaFreeHost(m_targetWork_2_h) );

        CUDA_CHECK( cudaFree(m_targetWork_3_d) );
		CUDA_CHECK( cudaFreeHost(m_targetWork_3_h) );

        CUDA_CHECK( cudaFree(m_confidenceReidDetTargets_d) );
		CUDA_CHECK( cudaFreeHost(m_confidenceReidDetTargets_h) );

        CUDA_CHECK( cudaFree(m_nearestIndReidDetTargets_d) );
		CUDA_CHECK( cudaFreeHost(m_nearestIndReidDetTargets_h) );

        if (m_params.m_useSegmentationMask4Input)
        {
            CUDA_CHECK( cudaFree(m_inputRoisInBatch_d) );
            CUDA_CHECK( cudaFreeHost(m_inputRoisInBatch_h) );

            CUDA_CHECK( cudaFree(m_targetInds4Rois_d) );
            CUDA_CHECK( cudaFreeHost(m_targetInds4Rois_h) );
        }

        /// Finally destroy cublas and cuda stream.
		CUBLAS_CHECK( cublasDestroy(m_cublasHandle) );
        CUDA_CHECK(cudaStreamDestroy(m_cuStream));

        CUDA_CHECK(cudaFreeHost(m_reidFeaturesDet_h));
        #ifdef VIZ_DEBUG_REID
            if (m_params.m_useExternalImages)
            {
                CUDA_CHECK(cudaFree(m_inputBuf_d));
                CUDA_CHECK(cudaFreeHost(m_inputBuf_h));
            }
        #endif
        m_subImageCropper2.reset();
        m_subImageCropScaler.reset();
        m_disMahaThreadWorker.reset();
        m_reidFeatureGallery.reset();
    }
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void ReID::queryParams(NvMOTQuery &paramsQuery)
{
	paramsQuery.outputReidTensor = m_params.m_outputReidTensor;
	paramsQuery.reidFeatureSize = m_params.m_reidFeatureSize;
	paramsQuery.numTransforms = 1;
}

/**
 * @brief Run inference for the targets in the input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in,out] detectedObjects Newly detected objects in each stream.
 * @param[in] isForReAssoc Whether this inference is for reassociation.
 */
void ReID::doInference(
    const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta,
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects, const bool isForReAssoc
)
{
    /// Perform inference if reid type enables NvDeepSORT and isForReAssoc==false.
    /// Perform inference if reid type enables reassoc and isForReAssoc==true.
    /// Other cases are invalid.
    if ( ( (m_params.m_reidType & REID_TYPES::NvDEEPSORT) && (!isForReAssoc) ) ||
         ( (m_params.m_reidType & REID_TYPES::REASSOC) && isForReAssoc ) )
    {
        if (m_params.m_debugVerboseLevel > 5)
        {
            cout << "[ReID::doInference] performs inference " << endl;
        }
    }
    else
    {
        return;
    }

    /// Mahalanobis distance only for NvDeepSORT feature extraction stage, And only needs to be calculated once when the first ReID model is called.
    if ( (m_params.m_reidType & REID_TYPES::NvDEEPSORT) && (!isForReAssoc) && m_calDisMaha)
    {
        m_disMahaThreadWorker->submitInput({&inputMeta, &detectedObjects});
    }

    // Crop input frames using detected bboxes and rescale them as ReID network input size.
    #ifdef VIZ_DEBUG_REID
        if (m_params.m_useExternalImages)
        {
            loadExtImagePatchesBatch();
        }
        else
        {
            extractImagePatchesBatch(inputMeta, detectedObjects);
        }
    #else
        extractImagePatchesBatch(inputMeta, detectedObjects);
    #endif
    /// Run ReID network for each batch.
    for (int dBboxInd = 0; dBboxInd < m_numBboxes; dBboxInd += m_params.m_maxBatchSize)
    {

        /// Except the last batch size is the number of remaining boxes.
        int batchSize = std::min(m_params.m_maxBatchSize, m_numBboxes - dBboxInd);
        /// Binding input and output address to nvinfer engine.
        m_bindings[m_inputInd] = (void*) (m_inputBuf_d + dBboxInd * m_params.m_channel *\
            m_params.m_height * m_params.m_width);
        m_bindings[m_outputInd] = (void*) (m_reidFeaturesDet_d + dBboxInd * m_params.m_reidFeatureSize);

        // Timer for debug.
        #ifdef VIZ_DEBUG_REID
            auto timeStart = std::chrono::system_clock::now();
        #endif
        m_inference.submitInput(m_bindings, {batchSize}, m_cuStream);

        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );

        #ifdef VIZ_DEBUG_REID
            auto timeEnd = std::chrono::system_clock::now();
            /// In seconds.
            std::chrono::duration<double> timeElapsed = timeEnd-timeStart;
            double timeCurr = timeElapsed.count() * 1e+3;
            /// Skip first batch.
            if (m_timeCnt >= 1)
            {
                m_timeTotal += timeCurr;
            }
            m_timeCnt++;
            if (m_params.m_debugVerboseLevel > 3)
            {
                cout << "Time [ReID inference] (ms): current: " << timeCurr <<
                    "\taverage: " << m_timeTotal / (m_timeCnt - 1) << endl;
            }
        #endif
    }
    if (m_params.m_addFeatureNormalization && m_numBboxes > 0)
    {
        cuReID::l2NormalizeReIDFeatures(
            m_cuStream, m_numBboxes, m_params.m_reidFeatureSize, m_normFeatureCudaBlockSize, m_reidFeaturesDet_d
        );
    }
    if (m_params.m_debugVerboseLevel > 5)
    {
        cout << "[ReID::doInference] getFeatures2Host" << endl;
    }
    getFeatures2Host();

    if ( (m_params.m_reidType & REID_TYPES::NvDEEPSORT) && (!isForReAssoc) && m_calDisMaha)
    {
        m_disMahaThreadWorker->waitToFinish();
    }

    /// Compute feature distance between detection and targets; stored in target.
    computeReIDFeatureDist(inputMeta);
}

/** @brief Normalize the current batch of ReID feature vectors. */
void ReID::normalizaReIDFeatures()
{
}

/**
 * @brief Compute detection-to-gallery ReID similarities.
 * @param[in] inputMeta Input frames keyed by stream ID.
 */
void ReID::computeReIDFeatureDist(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta)
{
    int streamInd = 0;
    int lastTargetInd = m_targetManager->getLastTargetInd();

    /// Record in targetWork_2 whether each target is updated.
    memset(m_targetWork_2_h, 0, m_maxTargetsToTrack*sizeof(int));

    /// TODO: Process streams concurrently instead of sequentially.
    /// A shared worker framework should cover most of these CPU operations.
	for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it, ++streamInd)
	{
        NvMOTStreamId videoStreamId = it->first;
        /// Multiply detection and gallery features when this stream has both.
        if (m_reidFeatureGallery->getReIDFeaturePtrs(videoStreamId) &&
            m_reidFeatureGallery->getLastInd(videoStreamId) >= 0 &&
            m_detObjReidFeatures.find(videoStreamId) != m_detObjReidFeatures.end())
        {
            multiplyReIDFeatures(streamInd, videoStreamId);
        }

        const NvTrackedObjectList &targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);

        /// Copy target's feature gallery ind to batched arrays needed by getNearestReIDSimilarity.
        for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
            uint targetInd = targetIt->second->getTargetInd();
            std::vector<int> featureInds = m_reidFeatureGallery->getTargetReIDFeatureInds(targetInd);
            if (featureInds.size()>0)
            {
                std::copy(featureInds.begin(), featureInds.end(), \
                    m_targetReidFeatureInds_h+targetInd*m_params.m_reidHistorySize);
            }
            /// Record each target's feature gallery size in targetWork_1.
            m_targetWork_1_h[targetInd] = featureInds.size();
            m_targetWork_2_h[targetInd] = streamInd;
        }
    }

    if(lastTargetInd >= 0)
    {
        /// Copy inds and size info to cuda array needed by getNearestReIDSimilarity.
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_targetReidFeatureInds_d, m_targetReidFeatureInds_h,
                (lastTargetInd+1) * m_params.m_reidHistorySize * sizeof(int), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_reidProductSize_d, m_reidProductSize_h,
                m_maxStreams * sizeof(int2), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_targetWork_2_d, m_targetWork_2_h, (lastTargetInd+1) * sizeof(int), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_targetWork_1_d, m_targetWork_1_h, (lastTargetInd+1) * sizeof(int), cudaMemcpyHostToDevice, m_cuStream
            )
        );

        /// Given ReID similarity score between detection and feature gallery, call a batched CUDA kernel to find nearest neighbor (highest score) for each detection and target pair.
        cuReID::getNearestReIDSimilarity(
            m_cuStream, lastTargetInd+1, m_maxTargetsPerStream, m_params.m_reidHistorySize,
            m_maxReidProductSizePerStream, m_targetReidFeatureInds_d, m_reidProduct_d, m_reidProductSize_d,
            /// Stream index in targetWork_2.
            m_targetWork_2_d,
            /// Record each target's feature gallery size in targetWork_1.
            m_targetWork_1_d, m_confidenceReidDetTargets_d, m_nearestIndReidDetTargets_d
        );

        /// Copy results back to CPU.
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_confidenceReidDetTargets_h, m_confidenceReidDetTargets_d,
                (lastTargetInd+1) * m_maxTargetsPerStream * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_nearestIndReidDetTargets_h, m_nearestIndReidDetTargets_d,
                (lastTargetInd+1) * m_maxTargetsPerStream * sizeof(int), cudaMemcpyDeviceToHost, m_cuStream
            )
        );
        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );
    }

    /// Copy re-id similarity score from batched array to each target.
    for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
        /// Generate ReID distance scores for targets in this stream.
        const NvTrackedObjectList &targetList = m_targetManager->getTrackedObjectsInUseList(it->first);

        for(auto tIt = targetList.begin(); tIt != targetList.end(); ++tIt)
        {
            const std::shared_ptr<NvTrackedObject> target = tIt->second;
            target->setConfidenceReidDets(
                m_confidenceReidDetTargets_h + target->getTargetInd() * m_maxTargetsPerStream, m_maxTargetsPerStream
            );

            /// Keep track of the frame num of the nearest neighbor reid feature in feature gallery.
            std::vector<int> frameNumReidDets;
            for (int fi = 0; fi < m_maxTargetsPerStream; fi++)
            {
                int frameNum = -1;
                if (m_featureIndFrameNumMap.find(it->first) != m_featureIndFrameNumMap.end())
                {
                    auto& frameNumsPerStream = m_featureIndFrameNumMap.at(it->first);
                    /// Add the stream offset to a non-negative local index so -1 does not alias another stream.
                    const int nearestIndLocal =
                        m_nearestIndReidDetTargets_h[target->getTargetInd() * m_maxTargetsPerStream + fi];
                    if (nearestIndLocal >= 0)
                    {
                        const int streamOffset = m_reidFeatureGallery->getReidFeatureOffset(it->first);
                        if (streamOffset >= 0)
                        {
                            const int nearestInd = nearestIndLocal + streamOffset;
                            if (frameNumsPerStream.find(nearestInd) != frameNumsPerStream.end())
                            {
                                frameNum = frameNumsPerStream.at(nearestInd);
                            }
                        }
                    }
                }
                frameNumReidDets.push_back(frameNum);
            }
            target->setConfidenceReidDetsFrameNum(std::move(frameNumReidDets));
            int reidInd = outputTargetReIDInd(it->first, target->getTargetInd());
            if (reidInd >= 0)
            {
                target->setReidFeaturesDet_h(
                    m_reidFeaturesDet_h + reidInd * m_params.m_reidFeatureSize, m_params.m_reidFeatureSize
                );
            }
            else
            {
                target->setReidFeaturesDet_h(nullptr, m_params.m_reidFeatureSize);
            }
        }
    }
}

/**
 * @brief Crop target regions from the input frame batch.
 * @param[in] inputMeta Input frames keyed by stream ID.
 * @param[in] detectedObjects Newly detected objects in each stream.
 */
void ReID::extractImagePatchesBatch(
    const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta,
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
    m_inputRoisInBatch.clear();
    m_detObjReidFeatures.clear();
    m_numBboxes = 0;
    const float aspect = float(m_params.m_width) / m_params.m_height;

    /// Store detector bboxes in m_inputRoisInBatch.
    for (auto streamIt = detectedObjects.begin(); streamIt != detectedObjects.end(); ++streamIt)
    {
        const NvMOTStreamId &videoStreamId = streamIt->first;
        std::deque<DetectedObjInfo> &detObjs = streamIt->second;

        /// If using segmentation for ReID input, need to know each bbox's targetInd.
        std::map<uint16_t, uint> bboxInd2TargetInd;
        /// Save them in bboxInd2TargetInd.
        if (m_params.m_useSegmentationMask4Input)
        {
            const std::map<uint32_t, std::shared_ptr<NvTrackedObject>>& targetList = \
                m_targetManager->getTrackedObjectsInUseList(videoStreamId);
            for (auto targetIt = targetList.begin(); targetIt != targetList.end(); targetIt++)
            {
                std::shared_ptr<NvTrackedObject> target = targetIt->second;
                if (target->getMatchedDetObjInd() >= 0)
                {
                    bboxInd2TargetInd[target->getMatchedDetObjInd()] = target->getTargetInd();
                }
            }
        }

        /// For each detector bounding box in a video stream.
		for (uint dBboxInd=0; dBboxInd < detObjs.size(); dBboxInd++)
		{
            if (dBboxInd >= uint(m_maxTargetsPerStream))
            {
                break;
            }

            const DetectedObjInfo& detObj = detObjs.at(dBboxInd);
            NvMotDataTypes::Rect2f roi = detObj.bbox;

            if (m_params.m_keepAspc)
            {
                roi = CV_FUNCS::getBoxWithAspectRatio(roi, aspect);
            }

            int targetInd = m_numBboxes;
            /// If using segmentation for input, add each image patch's targetInd.
            /// If the target is not found, mark it as -1, so not used with segmentation.
            /// If not using segmentation for input, assign consecutive target indices starting at zero.
            if (m_params.m_useSegmentationMask4Input)
            {
                if (bboxInd2TargetInd.find(detObj.bboxInd) != bboxInd2TargetInd.end())
                {
                    targetInd = bboxInd2TargetInd.at(detObj.bboxInd);
                }
                else
                {
                    targetInd = -1;
                }
            }

            m_inputRoisInBatch[videoStreamId].push_back(std::make_pair(targetInd, roi));

            /// Each detected object record its det bboxInd : reid ind.
            auto &detObjReidFeaturesPerStream = m_detObjReidFeatures[videoStreamId];
            detObjReidFeaturesPerStream.second[detObj.bboxInd] = m_numBboxes;

            /// Stores the pointer to the beginning of reid feature detected for this stream.
            /// Used to compute reid similarity score for each stream.
            if (dBboxInd == 0)
            {
                detObjReidFeaturesPerStream.first = m_reidFeaturesDet_d + m_numBboxes * m_params.m_reidFeatureSize;;
            }

            m_numBboxes++;
        }
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

        #ifdef VIZ_DEBUG_REID
            if(m_numBboxes > 0)
            {
                std::map<NvMOTStreamId, std::map<uint, cv::Mat>> subImgs =
                    m_subImageCropScaler->getImagePatches2Host_OCV_batch();
                vizImagePatches(subImgs, inputMeta);
            }
        #endif
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

        m_inputBuf_d = m_subImageCropper2->getOutputImagePatchPtrs();

        #ifdef VIZ_DEBUG_REID
            if (m_numBboxes > 0)
            {
                std::map<NvMOTStreamId, std::map<uint, cv::Mat>> subImgs =
                    m_subImageCropper2->getImagePatches2Host_OCV_batch();
                vizImagePatches(subImgs, inputMeta);
            }
        #endif
    }

    if (m_params.m_useSegmentationMask4Input)
    {
        generateImagePatchesWithSegmentationMask(detectedObjects);
    }
}

/**
 * @brief Apply target masks to the cropped ReID input patches.
 * @param[in,out] detectedObjects Newly detected objects in each stream.
 */
void ReID::generateImagePatchesWithSegmentationMask(
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
	const SegmentationInBatch& segInfo = m_targetManager->getSegmentationBatchInfo();
    if (m_numBboxes == 0 || segInfo.segMaskInBatch_d == nullptr)
    {
        return;
    }

    int imgPatchInd = 0;
    for (auto it = m_inputRoisInBatch.begin(); it != m_inputRoisInBatch.end(); it++)
    {
        auto& inputRoisInBatchPeStream = it->second;
        for (auto jt = inputRoisInBatchPeStream.begin(); jt != inputRoisInBatchPeStream.end(); jt++)
        {
            const uint32_t targetInd = jt->first;
            const auto& roi = jt->second;

            m_inputRoisInBatch_h[imgPatchInd].x = roi.min()(0) / m_srcImageSize.width * segInfo.size.width;
            m_inputRoisInBatch_h[imgPatchInd].y = roi.min()(1) / m_srcImageSize.height * segInfo.size.height;
            m_inputRoisInBatch_h[imgPatchInd].w = roi.sizes()(0) / m_srcImageSize.width * segInfo.size.width;
            m_inputRoisInBatch_h[imgPatchInd].z = roi.sizes()(1) / m_srcImageSize.height * segInfo.size.height;

            m_targetInds4Rois_h[imgPatchInd] = targetInd;

            imgPatchInd++;
        }
    }

    CUDA_CHECK(
        cudaMemcpyAsync(
            m_inputRoisInBatch_d, m_inputRoisInBatch_h, m_numBboxes*sizeof(float4), cudaMemcpyHostToDevice, m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_targetInds4Rois_d, m_targetInds4Rois_h, m_numBboxes*sizeof(int), cudaMemcpyHostToDevice, m_cuStream
        )
    );

    cuReID::applySegmentaitonToImagePatches(
        m_cuStream, segInfo.segMaskInBatch_d, segInfo.size.width,
        segInfo.size.height * segInfo.size.width, m_params.m_width, m_params.m_width * m_params.m_height,
        m_numBboxes, m_targetInds4Rois_d, m_inputRoisInBatch_d, m_inputBuf_d
    );
}

/** @brief Update retained state after data association. */
void ReID::update()
{
    /// If re-assoc enabled, copy fused targets' features to their peers.
    if ( m_params.m_reidType & REID_TYPES::REASSOC )
    {
        const std::map<NvMOTStreamId, FusedObjectList>& targetFused = m_targetManager->getTrackedObjectsFused();
        for (auto it = targetFused.begin(); it != targetFused.end(); ++it)
        {
            NvMOTStreamId videoStreamId = it->first;
            for (auto targetIt = it->second.begin(); targetIt != it->second.end(); targetIt++)
            {
                auto targetKept = targetIt->first;
                auto targetFused = targetIt->second.first;
                m_reidFeatureGallery->fuseReidFeatureInd(
                    videoStreamId, targetKept->getTargetInd(), targetFused->getTargetInd()
                );
            }
        }
    }

    /// Remove the ReID features of terminated targets.
    const std::map<NvMOTStreamId, std::vector<std::shared_ptr<NvTrackedObject>>> & targetsTerminated = \
        m_targetManager->getTrackedObjectsTerminated();
    for (auto it = targetsTerminated.begin(); it != targetsTerminated.end(); ++it)
    {
        NvMOTStreamId videoStreamId = it->first;
        for (uint ti = 0; ti < it->second.size(); ti++)
        {
            removeTrackerReIDFeature(videoStreamId, it->second.at(ti));
        }
    }

    /// Record in targetWork_2 whether each target is updated.
    memset(m_targetWork_2_h, 0, m_maxTargetsToTrack*sizeof(int));

    /// Update target's ReID feature from its matched detections.
    for (auto it = m_detObjReidFeatures.begin(); it != m_detObjReidFeatures.end(); it++)
    {
        NvMOTStreamId videoStreamId = it->first;
        const NvTrackedObjectList& targetList = \
            m_targetManager->getTrackedObjectsInUseList(videoStreamId);

        if (!targetList.empty())
        {
            const auto &detFeaturesPerStream = it->second.second;
            for (auto targetIt = targetList.begin(); targetIt != targetList.end(); targetIt++)
            {
                std::shared_ptr<NvTrackedObject> target = targetIt->second;
                uint targetInd = target->getTargetInd();

                /// If target is matched with a detection, update its reid features.
                int detInd = target->getMatchedDetObjInd();

                /// If re-assoc enabled, the target (i.e. a peer) has no matched detection but has a fused re-assoc, the fused re-assoc may have a reid detection in the current batch.
                /// Fuse it to the peer.
                if ( ( m_params.m_reidType & REID_TYPES::REASSOC ) &&
                    (detInd < 0 || detFeaturesPerStream.find(detInd) == detFeaturesPerStream.end()) )
                {
                    /// Find if it (the peer) has a fused target (terminated after fusion).
                    if (m_targetManager->getTrackedObjectsFused().find(videoStreamId) != m_targetManager->getTrackedObjectsFused().end())
                    {
                        const FusedObjectList &targetsFused = m_targetManager->getTrackedObjectsFused().at(
                            videoStreamId
                        );
                        if (targetsFused.find(target) != targetsFused.end())
                        {
                            /// Use the matched detection of the fused target to update the peer's reid feature.
                            detInd = targetsFused.at(target).second;
                        }
                    }
                }

                /// Filter out low-visibility objects.
                if(target->getTargetInfo().visibility < m_params.m_minVisibility4GalleryUpdate)
                {
                    continue;
                }

                /// Only update the gallery when the detection has a reid feature.
                if (detFeaturesPerStream.find(detInd) != detFeaturesPerStream.end())
                {
                    /// Store the detection's reid feature index in targetWork_1.
                    m_targetWork_1_h[targetInd] = detFeaturesPerStream.at(detInd);
                    /// Mark this target for a gallery update.
                    m_targetWork_2_h[targetInd] = 1;
                    /// Store this target's global ReID feature index.
                    m_targetWork_3_h[targetInd] = m_reidFeatureGallery->getReidFeatureIndGlobal(
                        videoStreamId, targetInd
                    );
                    /// Update frame num.
                    m_featureIndFrameNumMap[videoStreamId][m_targetWork_3_h[targetInd]] = target->getTargetInfo().frameNum;
                }
            }
        }

        if (m_params.m_debugVerboseLevel > 3)
        {
            for (auto targetIt = targetList.begin(); targetIt != targetList.end(); targetIt++)
            {
                std::shared_ptr<NvTrackedObject> target = targetIt->second;
                uint targetInd = target->getTargetInd();
                const uint64_t targetID = target->getTargetID();
                std::cout << "Tracker [" << targetID << "] ReID feature indices: ";
                m_reidFeatureGallery->printFeatureInds(targetInd);
                std::cout << std::endl;
            }
        }
    }

    int lastTargetInd = m_targetManager->getLastTargetInd();
    if (lastTargetInd >= 0)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_targetWork_1_d, m_targetWork_1_h, (lastTargetInd+1) * sizeof(int), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_targetWork_2_d, m_targetWork_2_h, (lastTargetInd+1) * sizeof(int), cudaMemcpyHostToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_targetWork_3_d, m_targetWork_3_h, (lastTargetInd+1) * sizeof(int), cudaMemcpyHostToDevice, m_cuStream
            )
        );

        cuReID::copyDetReIDFeatureToGallery(
            m_cuStream, lastTargetInd+1, m_params.m_reidFeatureSize, m_targetWork_1_d, m_reidFeaturesDet_d,
            m_targetWork_3_d, m_targetWork_2_d, m_reidFeatureGallery->getFirstReIDFeaturePtrs()
        );
    }
}

/**
 * @brief Remove the retained ReID features for a target.
 * @param[in] videoStreamId Video stream id.
 * @param[in] target Tracker associated with the detection.
 */
void ReID::removeTrackerReIDFeature(const NvMOTStreamId videoStreamId, const std::shared_ptr<NvTrackedObject> target)
{
    m_reidFeatureGallery->remove(videoStreamId,\
        m_reidFeatureGallery->getTargetReIDFeatureInds( target->getTargetInd() ));
    m_reidFeatureGallery->resetTargetReIDFeatureInds( target->getTargetInd() );
}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void ReID::removeStream(const NvMOTStreamId videoStreamId)
{
    if (m_targetManager->getTrackedObjectsTerminated().find(videoStreamId) !=
        m_targetManager->getTrackedObjectsTerminated().end())
    {
        const std::vector<std::shared_ptr<NvTrackedObject>> & targetsTerminated = \
            m_targetManager->getTrackedObjectsTerminated().at(videoStreamId);

        for (uint targetInd = 0; targetInd < targetsTerminated.size(); targetInd++)
        {
            removeTrackerReIDFeature(videoStreamId, targetsTerminated.at(targetInd));
        }
    }

    if (m_featureIndFrameNumMap.find(videoStreamId) != m_featureIndFrameNumMap.end())
    {
        m_featureIndFrameNumMap.erase(videoStreamId);
    }

    m_reidFeatureGallery->removeStream(videoStreamId);
}

/**
 * @brief Expose the current batch of ReID features.
 * @param[out] reidTensorBatch Batch reid tensor output to be filled.
 */
void ReID::outputReIDTensor(NvDsReidTensorBatch& reidTensorBatch)
{
    if (m_params.m_outputReidTensor)
    {
        if (reidTensorBatch.ptr_dev == nullptr || reidTensorBatch.ptr_host == nullptr)
        {
            cout << "[ReID][Warning] Invalid reid output array provided. Skip reid tensor output" << endl;
            return;
        }

        reidTensorBatch.numFilled = m_numBboxes;
        reidTensorBatch.featureSize = m_params.m_reidFeatureSize;

        if (m_numBboxes == 0)
        {
            return;
        }

        CUDA_CHECK(
            cudaMemcpyAsync (
                reidTensorBatch.ptr_dev, m_reidFeaturesDet_d,
                m_numBboxes * m_params.m_reidFeatureSize * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync (
                reidTensorBatch.ptr_host, m_reidFeaturesDet_d,
                m_numBboxes * m_params.m_reidFeatureSize * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
            )
        );
        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );
    }
}

/**
 * @brief Get a target feature index in the batch tensor.
 * @param[in] streamId Stream index.
 * @param[in] targetInd The target index.
 * @return The target's ReID vector index in the batch tensor, -1 means no reid feature.
 */
int ReID::outputTargetReIDInd(NvMOTStreamId streamId, uint targetInd)
{
    int reidInd = -1;

    if (m_params.m_outputReidTensor)
    {
        /// Look up the target and its matched detection index.
        const std::shared_ptr<NvTrackedObject> target = \
            m_targetManager->getTrackedObjectsInUseList(streamId).at(targetInd);
        int detInd = target->getMatchedDetObjInd();

        /// Get the reid index in the whole batch.
        if (m_detObjReidFeatures.find(streamId) != m_detObjReidFeatures.end() &&
            m_detObjReidFeatures.at(streamId).second.find(detInd) != m_detObjReidFeatures.at(streamId).second.end())
        {
            reidInd = m_detObjReidFeatures.at(streamId).second.at(detInd);
        }
    }

    return reidInd;
}

/**
 * @brief Populate ReID metadata for a tracked target.
 * @param[in] streamId Stream index.
 * @param[in] targetInd The target index.
 * @param[in] reidTensorBatch Batch reid tensor to get the memory pointers.
 * @param[out] reidObj The target's ReID vector memory pointers and feature size.
 */
void ReID::outputTargetReIDObj(
    NvMOTStreamId streamId, uint targetInd, NvDsReidTensorBatch& reidTensorBatch, NvDsObjReid& reidObj
)
{
    int reidInd = -1;
    /// Initialize all the variables to 0 & NULL.
    reidObj.featureSize = 0;
    reidObj.ptr_dev = NULL;
    reidObj.ptr_host = NULL;
    if (m_params.m_outputReidTensor)
    {
        if (reidTensorBatch.ptr_dev == nullptr || reidTensorBatch.ptr_host == nullptr)
        {
            cout << "[ReID][Warning] Invalid reid output array provided. Skip reid tensor output" << endl;
            return;
        }
        if (m_numBboxes == 0)
        {
            return;
        }
        /// Look up the target and its matched detection index.
        const std::shared_ptr<NvTrackedObject> target = \
            m_targetManager->getTrackedObjectsInUseList(streamId).at(targetInd);
        int detInd = target->getMatchedDetObjInd();

        if(m_params.m_debugVerboseLevel > 3)
        {
            if ((target->getTargetInfo().visibility < m_params.m_minVisibility4ReidOutput ||
            target->getTargetInfo().imgBorderCroppingVisibility < m_params.m_minVisibility4ReidOutput) &&
            m_detObjReidFeatures.find(streamId) != m_detObjReidFeatures.end() &&
            m_detObjReidFeatures.at(streamId).second.find(detInd) != m_detObjReidFeatures.at(streamId).second.end())
            {
                cout << "[ReID][Warning] Target [" << target->getTargetID() << "] visibility is less than " << m_params.m_minVisibility4ReidOutput << ". Skipping reid tensor output" << endl;
            }
        }
        /// Get the reid index in the whole batch.
        if (target->getTargetInfo().visibility >= m_params.m_minVisibility4ReidOutput &&
            target->getTargetInfo().imgBorderCroppingVisibility >= m_params.m_minVisibility4ReidOutput &&
            m_detObjReidFeatures.find(streamId) != m_detObjReidFeatures.end() &&
            m_detObjReidFeatures.at(streamId).second.find(detInd) != m_detObjReidFeatures.at(streamId).second.end())
        {
            reidInd = m_detObjReidFeatures.at(streamId).second.at(detInd);
            reidObj.featureSize = m_params.m_reidFeatureSize;
            reidObj.ptr_dev = reidTensorBatch.ptr_dev + reidInd * m_params.m_reidFeatureSize;
            reidObj.ptr_host = reidTensorBatch.ptr_host + reidInd * m_params.m_reidFeatureSize;
        }
    }
}

/**
 * @brief Multiply detection features by the stream feature gallery.
 * @param[in] streamInd Index of this stream in the current input batch.
 * @param[in] videoStreamId Video stream identifier.
 */
void ReID::multiplyReIDFeatures(const int streamInd, const NvMOTStreamId videoStreamId)
{
    /// Determine reid CUDA work space offset.
    float *reidProduct_d = m_reidProduct_d + \
        m_maxReidProductSizePerStream * streamInd;

    const float alpha = 1.0;
    const float beta = 0.0;

    /// Store them in m_reidProductSize_h.
    int featureGallerySize = m_reidFeatureGallery->getLastInd(videoStreamId) + 1;
    int detObjSize = m_detObjReidFeatures.at(videoStreamId).second.size();
    m_reidProductSize_h[streamInd].x = featureGallerySize;
    m_reidProductSize_h[streamInd].y = detObjSize;

    /// Calculate cosine distance of each <feat_det, feat_gallery> pair.
    /// Store reidProduct = matmul (reidFeaturesDet_d^T, reidFeaturesGallery_d).
    /// [det size x 128] * [128 x gallery size] --> [det size x gallery size].
    CUBLAS_CHECK(
        cublasSgemm(
            m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, featureGallerySize, detObjSize,
            m_params.m_reidFeatureSize, &alpha, m_reidFeatureGallery->getReIDFeaturePtrs(videoStreamId),
            m_params.m_reidFeatureSize, m_detObjReidFeatures.at(videoStreamId).first,
            m_params.m_reidFeatureSize, &beta, reidProduct_d, featureGallerySize
        )
    );
}

/**
 * @brief Compute Mahalanobis distances for one queued work item.
 * @param[in] in Input to compute Mahalanobis distance.
 */
void ReID::computeMahalanobisDist(DisMahalanobisIn in)
{
    if (!in.first || !in.second)
    {
        return;
    }

    const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta = *(in.first);
    std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects = *(in.second);
    /// Copy re-id similarity score from batched array to each target.
    for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
        const NvMOTStreamId &videoStreamId = it->first;
        /// Generate ReID distance scores for targets in this stream.
        const NvTrackedObjectList &targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);

        if (detectedObjects.find(videoStreamId) == detectedObjects.end() ||
            detectedObjects.at(videoStreamId).empty() || targetList.empty())
        {
            continue;
        }

        const std::deque<DetectedObjInfo>& detObjsPerStream = detectedObjects.at(videoStreamId);

        for(auto tIt = targetList.begin(); tIt != targetList.end(); ++tIt)
        {
            const std::shared_ptr<NvTrackedObject> target = tIt->second;
            std::vector<float> disMaha(m_maxTargetsPerStream, 0.f);

			Eigen::VectorXd X_pred;
			Eigen::MatrixXd X_pred_cov;
			std::tie(X_pred, X_pred_cov) = target->getPredictedMeasurementsWithCov();

			/// Here assumes cov is diagonal, add eps in case cov is singular.
			Eigen::Diagonal<Eigen::MatrixXd> predCov = X_pred_cov.diagonal();

			predCov.array() += std::numeric_limits<double>::epsilon();
			Eigen::VectorXd predCovInv = predCov.cwiseInverse();

            /// Iterate over each det and tracker to get distance.
			for (uint detInd = 0; detInd < detObjsPerStream.size(); detInd++)
			{
				const DetectedObjInfo& detObj = detObjsPerStream[detInd];
				Eigen::VectorXd detMeas = CV_FUNCS::convertBboxToZ(detObj.bbox, target->getStateUseAspc());

				Eigen::VectorXd diffMeas = X_pred - detMeas;
                if (detObj.bboxInd < uint16_t(disMaha.size()))
                {
					disMaha[detObj.bboxInd] = diffMeas.array().square().matrix().dot(predCovInv);
                }
			}

            target->setDisMahalanobisDets(std::move(disMaha));
        }
    }
}

#ifdef VIZ_DEBUG_REID
/**
 * @brief Write cropped target images for visualization.
 * @param[in] subImgs Cropped target images keyed by stream and target.
 * @param[in] inputMeta Input frames keyed by stream ID.
 */
void ReID::vizImagePatches(
    const std::map<NvMOTStreamId, std::map<uint, cv::Mat > >& subImgs,
    const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta
)
{
    for(auto it = subImgs.begin(); it != subImgs.end(); ++it)
    {
        NvMOTStreamId streamId = it->first;
        auto frameNum = inputMeta.at(streamId)->frameNum;
        uint bboxInd = 0;
        for(auto jt=it->second.begin(); jt != it->second.end(); ++jt, bboxInd++)
        {
            cv::Mat subImg = jt->second;

            std::string outDir = "infer_patches/";
            std::string label = "s" + std::to_string(streamId) + \
                "_" + std::to_string(frameNum) + "_" + \
                std::to_string(bboxInd) + ".png";

            cv::imwrite(outDir + label, subImg);
        }
    }
}

/** @brief Load externally supplied image patches for inference. */
void ReID::loadExtImagePatchesBatch()
{
    m_inputRoisInBatch.clear();
    m_numBboxes = 0;

    std::vector<std::string> extImageList;
    boost::filesystem::path extImagePath(m_params.m_extImagePath);
    boost::filesystem::directory_iterator endIt;

    if (m_params.m_debugVerboseLevel > 2)
    {
        cout << "[ReID::loadExtImagePatchesBatch] m_extFrameNum: " << m_extFrameNum << endl;
    }

    for (boost::filesystem::directory_iterator it(extImagePath);
        it != endIt; it++)
    {
        if (boost::filesystem::is_regular_file(it->path()))
        {
            /// File name xxxx/MOT16-xx_frm_idx.jpg.
            const std::string &imageName = it->path().string();
            const int pos = imageName.size() - 11;
            if(std::stoi(imageName.substr(pos, 3)) == m_extFrameNum)
            {
                /// Collect the image files that belong to the current external frame.
                extImageList.push_back(it->path().string());
            }
        }
    }

    std::sort(extImageList.begin(), extImageList.end());
    m_numBboxes = std::min((int)extImageList.size(), m_maxTargetsToTrack);

    for (int i = 0; i < m_numBboxes; i++)
    {
        cv::Mat crop = cv::imread(extImageList[i]);
        cv::cvtColor(crop, crop, CV_BGR2RGB);
        cv::Mat buf(
            m_params.m_height, m_params.m_width, CV_32FC3,
            reinterpret_cast<float*>(m_inputBuf_h) + i * m_params.m_width * m_params.m_height * m_params.m_channel
        );
        crop.convertTo(buf, CV_32FC3);
    }
    CUDA_CHECK(
        cudaMemcpyAsync (
            m_inputBuf_d, m_inputBuf_h,
            m_numBboxes * m_params.m_width * m_params.m_height * m_params.m_channel * sizeof(float),
            cudaMemcpyHostToDevice, m_cuStream
        )
    );
    CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
    CUDA_CHECK( cudaPeekAtLastError() );

    /// Increase ext frame cnt by 1.
    m_extFrameNum++;
}
#endif

/** @brief Copy ReID features from device memory to host memory. */
void ReID::getFeatures2Host()
{
    memset(m_reidFeaturesDet_h, 0, m_maxTargetsToTrack * m_params.m_reidFeatureSize  * sizeof(float));
    if (m_params.m_debugVerboseLevel > 2)
    {
        cout << "Number of feature vectors:" << m_numBboxes << endl;
    }
    CUDA_CHECK(
        cudaMemcpyAsync (
            m_reidFeaturesDet_h, m_reidFeaturesDet_d,
            m_numBboxes * m_params.m_reidFeatureSize * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
        )
    );
    CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
    CUDA_CHECK( cudaPeekAtLastError() );

    if (m_params.m_debugVerboseLevel > 5)
    {
        cout << "[ReID::getFeatures2Host] ReID features: ";
        for (int bboxInd = 0; bboxInd < std::min(m_maxTargetsToTrack, m_numBboxes); bboxInd++)
        {
            for (int fInd = 0; fInd < m_params.m_reidFeatureSize; fInd++)
            {
                printf("%.8f", m_reidFeaturesDet_h[bboxInd * m_params.m_reidFeatureSize + fInd]);
                if (fInd != m_params.m_reidFeatureSize - 1)
                {
                    printf(" ");
                }
                else
                {
                    printf("\n");
                }
            }
        }
    }
}

/** @brief Construct a ReIDConfigParams instance. */
ReIDConfigParams::ReIDConfigParams()
{
    /// TensorRT work size.
    m_maxBatchSize = 1;
    m_workspaceSize = 20;

    /// Reid model tensor format.
    m_reidFeatureSize = 1024;
    m_reidHistorySize = 100;
    m_inputOrder = 1;
    m_width = 64;
    m_height = 128;
    m_channel = 3;

    m_colorFormat = NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGB;
    m_offsets = std::vector<float>(3, 0.0);
    m_netScaleFactor = 1.0;

    m_minVisibility4GalleryUpdate = 0.6f;
    m_minVisibility4ReidOutput = 0.0f;

    /// Preprocess patches.
    /// Default reid model keeps aspect ratio.
    m_keepAspc = true;
    m_outputReidTensor = false;

    /// Post process features.
    m_addFeatureNormalization = false;

    m_reidType = REID_TYPES::NvDEEPSORT;
    m_useVPICropScaler = true;
    m_useSegmentationMask4Input = false;

    m_debugVerboseLevel = 0;

    #ifdef VIZ_DEBUG_REID
        m_useExternalImages = false;
    #endif
}

/**
 * @brief Construct a ReIDConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 * @param[in] submoduleName Configuration submodule name.
 */
ReIDConfigParams::ReIDConfigParams(
    const std::string &configFilePath, const std::string moduleName, const std::string submoduleName
)
{
    /// Set the default values.
    *this = ReIDConfigParams();

    YAML::Node configRoot = getConfigRoot(configFilePath);
    /// ReID model base parameters.
    this->read(configRoot[moduleName]);

    if (!submoduleName.empty())
    {
        /// The submodule contains model specific parameters to overwrite the basic module.
        this->read(configRoot[moduleName][submoduleName]);
    }

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
 * @brief Construct a ReIDConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
ReIDConfigParams::ReIDConfigParams(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void ReIDConfigParams::read(const YAML::Node &fn)
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
		else if (paramKey == "reidType")
		{
            m_reidType = (REID_TYPES)((int)fit->second.as<float>());
        }
		else if (paramKey == "batchSize")
        {
            int maxBatchSize = (int)fit->second.as<float>();
            if (maxBatchSize > 0)
            {
                m_maxBatchSize = maxBatchSize;
            }
            else
            {
                cout << "maxBatchSize should be positive int; use default batch size" <<endl;
            }
        }
        else if (paramKey == "reidFeatureSize")
        {
            int reidFeatureSize = (int)fit->second.as<float>();
            if (reidFeatureSize > 0)
            {
                m_reidFeatureSize = reidFeatureSize;
            }
            else
            {
                throw std::runtime_error("reidFeatureSize should be positive int");
            }
        }
        else if (paramKey == "reidHistorySize")
        {
            int reidHistorySize = (int)fit->second.as<float>();
            if (reidHistorySize > 0)
            {
                m_reidHistorySize = reidHistorySize;
            }
            else
            {
                throw std::runtime_error("reidHistorySize should be positive int");
            }
        }
        else if (paramKey == "inferDims")
        {
            m_inferDims.clear();
            /// Interp later because input order unknown at this time.
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
        else if (paramKey == "netScaleFactor")
        {
			m_netScaleFactor = fit->second.as<float>();
        }
        else if (paramKey == "minVisibility4GalleryUpdate")
        {
			m_minVisibility4GalleryUpdate = fit->second.as<float>();
        }
        else if (paramKey == "minVisibility4ReidOutput")
        {
            m_minVisibility4ReidOutput = fit->second.as<float>();
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
        else if (paramKey == "keepAspc")
		{
			m_keepAspc = (int)fit->second.as<float>();
		}
        else if (paramKey == "addFeatureNormalization")
		{
			m_addFeatureNormalization = (int)fit->second.as<float>();
		}
        else if (paramKey == "outputReidTensor")
		{
			m_outputReidTensor = (int)fit->second.as<float>();
		}
        else if (paramKey == "reidType")
        {
            /// ReidType should be determined by plugin factory.
            ;
        }
        else if (paramKey == "useVPICropScaler")
        {
            m_useVPICropScaler = (int)fit->second.as<float>();
        }
        else if (paramKey == "useSegmentationMask4Input")
        {
            m_useSegmentationMask4Input = (int)fit->second.as<float>();
        }
        #ifdef VIZ_DEBUG_REID
            else if (paramKey == "extImagePath")
            {
                m_extImagePath = fit->second.as<std::string>();
                m_useExternalImages = true;
            }
        #endif
    }
}
