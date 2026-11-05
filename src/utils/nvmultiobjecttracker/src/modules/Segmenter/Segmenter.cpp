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

#include "modules/Segmenter/Segmenter.hpp"
#include "modules/Segmenter/cuSegmenter.h"
#include "modules/Segmenter/SegmenterProfile.hpp"
#include "modules/Segmenter/SegConstant.cuh"
#include "modules/utility/debug_common.hpp"
#include "modules/utility/cuda_common.hpp"

using std::cout;
using std::endl;

/**
 * @brief Broadcast-add a per-channel vector across spatial locations.
 * @param[out] dst Destination tensor to populate.
 * @param[in] src_large Larger input tensor.
 * @param[in] src_small Input vector broadcast across the larger tensor.
 */
static void broadcastAddCpu(float* dst, const float* src_large, const float* src_small)
{
    for(int i =0;i<4096;i++){
        for(int j =0;j<64;j++){
            dst[i*64+j] = src_large[i*64+j] + src_small[j];
        }
    }
};

/**
 * @brief Construct a Segmenter instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
Segmenter::Segmenter(const std::string &configFilePath)
{
    m_params = SegmenterConfigParams(configFilePath, "Segmenter");
    /// Load the segmenter model configuration referenced by the tracker config.
    YAML::Node configRoot = NvTrackerParams::getConfigRoot(configFilePath);
    if (!configRoot["Segmenter"]["segmenterConfigPath"])
    {
        throw std::runtime_error("[Segmenter::Segmenter] segmenterConfigPath not found in tracker config file");
    }
    std::string segmenterConfigPath = configRoot["Segmenter"]["segmenterConfigPath"].as<std::string>();
    segmenterConfigPath = NvTrackerParams::resolvePathRelativeToConfig(configFilePath, segmenterConfigPath);
    YAML::Node configSeg = YAML::LoadFile(segmenterConfigPath);

    /// Image encoder and mask decoder compatible with legacy names.
    if (configSeg["encoder"])
    {
        m_imageEncoder.m_params = TRTConfigParams(segmenterConfigPath, "encoder", "");
    }
    else if (configSeg["ImageEncoder"])
    {
        m_imageEncoder.m_params = TRTConfigParams(segmenterConfigPath, "ImageEncoder", "");
    }
    else
    {
        throw std::runtime_error("[Segmenter::Segmenter] ImageEncoder not found in tracker config file");
    }

    if (configSeg["decoder"])
    {
        m_maskDecoder.m_params = TRTConfigParams(segmenterConfigPath, "decoder", "");
    }
    else if (configSeg["MaskDecoder"])
    {
        m_maskDecoder.m_params = TRTConfigParams(segmenterConfigPath, "MaskDecoder", "");
    }
    else
    {
        throw std::runtime_error("[Segmenter::Segmenter] MaskDecoder not found in tracker config file");
    }

    if (m_params.m_segmenterType == SEGMENTER_TYPES::EfficientVitSAM)
    {
        m_imageEncoder.initializeModel(&createOnnxProfile4EVSamEncoder, &setRuntimeDim4EVSamEncoder);
        m_maskDecoder.initializeModel(&createOnnxProfile4EVSamDecoder, &setRuntimeDim4EVSamDecoder);
    }
    else if (m_params.m_segmenterType == SEGMENTER_TYPES::NanoSAM)
    {
        m_imageEncoder.initializeModel(nullptr, nullptr);
        m_maskDecoder.initializeModel(&createOnnxProfile4NanoSamDecoder, &setRuntimeDim4NanoSamDecoder);
    }
    else if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
    {
        m_imageEncoder.initializeModel(nullptr, nullptr);
        m_maskDecoder.initializeModel(&createOnnxProfile4Sam2MaskDecoder, &setRuntimeDim4Sam2MaskDecoder);

        if (configSeg["MemoryAttention"] && configSeg["MemoryEncoder"])
        {
            m_params.m_useMemBank = true;
        }
        else
        {
            printf(
                "[Segmenter][Warning] SAM2 memory attention or memory encoder not provided, memory bank is disabled!!"
            );
        }

        if (m_params.m_useMemBank)
        {
            m_memoryAttention.m_params = TRTConfigParams(segmenterConfigPath, "MemoryAttention", "");
            m_memoryEncoder.m_params = TRTConfigParams(segmenterConfigPath, "MemoryEncoder", "");

            m_memoryAttention.initializeModel(&createOnnxProfile4Sam2MemAttn, &setRuntimeDim4Sam2MemAttn);
            m_memoryEncoder.initializeModel(&createOnnxProfile4Sam2MemEncoder, &setRuntimeDim4Sam2MemEncoder);
        }
    }
    else
    {
        throw std::runtime_error("Unknown segmenter type");
    }
    m_isInit = false;
}

/**
 * @brief Construct a Segmenter instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] modelName Configured model name.
 */
Segmenter::Segmenter(const std::string &configFilePath, const std::string &modelName)
{
    m_isInit = false;
}

/**
 * @brief Configure segmentation inference and allocate runtime resources.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] targetManager Target manager of the tracked objects.
 */
void Segmenter::initialize(
    const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
    const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
    m_targetManager = targetManager;
    m_isForLocalization = false;

    /// Keep the source and original image sizes for the scale transformations.
    m_srcImageSize = frameInfo.srcImageSize;
    m_originalSrcImageSize = frameInfo.originalSrcImageSize;

    if (m_params.m_segmenterType == SEGMENTER_TYPES::EfficientVitSAM)
    {
        m_scaleMask2Src(0) = frameInfo.srcImageSize.width / (float)m_params.m_width/2;
        m_scaleMask2Src(1) = frameInfo.srcImageSize.height / (float)m_params.m_height/2;
    }
    else if (m_params.m_segmenterType == SEGMENTER_TYPES::NanoSAM || m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
    {
        m_scaleMask2Src(0) = frameInfo.srcImageSize.width / (float)m_params.m_width;
        m_scaleMask2Src(1) = frameInfo.srcImageSize.height / (float)m_params.m_height;
    }

    m_scaleInput2Original(0) = frameInfo.originalSrcImageSize.width / (float)frameInfo.srcImageSize.width;
    m_scaleInput2Original(1) = frameInfo.originalSrcImageSize.height / (float)frameInfo.srcImageSize.height;

    /// Set runtime parameters.
    {
        /// For SAM2, m_hiddenDim = 256, m_encoderFeatureDim = 64 featuresEncoderSize: 256 x 64 x 64.
        m_paramsRt.featuresEncoderSize          = m_params.m_hiddenDim * m_params.m_encoderFeatureDim * m_params.m_encoderFeatureDim;
        /// HighResFeaturesEncoder0Size: 32 x 256 x 256.
        m_paramsRt.highResFeaturesEncoder0Size  = m_params.m_hiddenDim / 8 * m_params.m_encoderFeatureDim * 4 * m_params.m_encoderFeatureDim * 4;
        /// HighResFeaturesEncoder1Size: 64 x 128 x 128.
        m_paramsRt.highResFeaturesEncoder1Size  = m_params.m_hiddenDim / 4 * m_params.m_encoderFeatureDim * 2 * m_params.m_encoderFeatureDim * 2;

        /// Already checked attention, decoder and memory encoder have the same batch size.
        m_paramsRt.batchSize = m_maskDecoder.m_params.m_maxBatchSize;
        /// MemChannel: 64.
        m_paramsRt.memChannel = m_params.m_encoderFeatureDim;

        /// Each target has at most 2 points: top-left and bottom-right.
        m_paramsRt.maxPointNum          = m_paramsRt.batchSize * 2;
        /// PixFeatWithMemSize (mem attn output) 256 x 64 x 64.
        m_paramsRt.pixFeatWithMemSize   = m_paramsRt.featuresEncoderSize;
        /// MaskMemFeatSize: 64 x 64 x 64.
        m_paramsRt.maskMemFeatSize      = m_params.m_encoderFeatureDim * m_params.m_encoderFeatureDim * m_paramsRt.memChannel;

        /// Min (1 x 4096 + 4 x 1 )x 64 = 4100 x 64 max (7 x 4096 + 4 x 16)x 64 = 28736 x 64.
        m_paramsRt.maxConcatMemFeatSize = m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize + 4 * m_params.m_numObjPtr * m_paramsRt.memChannel;
        /// SAM2: 256.
        m_paramsRt.objPtrSize           = m_params.m_hiddenDim;
        /// SAM2: 64.
        m_paramsRt.temporalCodeSize     = m_params.m_encoderFeatureDim;

        /// SAM2 basicMaskSize: 1 x 256 x 256.
        m_paramsRt.basicMaskSize        = m_paramsRt.batchSize * m_params.m_outputLabelNum * m_params.m_hiddenDim * m_params.m_hiddenDim;
        /// HighResMaskSize: 1 x 1024 x 1024.
        m_paramsRt.highResMaskSize      = m_paramsRt.batchSize * m_params.m_outputLabelNum * m_params.m_hiddenDim * 4 * m_params.m_hiddenDim * 4;
    }

    /// Initialize the image cropper selected by the configuration.
    if(m_params.m_useVPICropScaler)
    {
        m_subImageCropScaler = std::make_shared<ReIDCropScaler>();
        m_subImageCropScaler->initialize(
            m_params.m_debugVerboseLevel, frameInfo.maxStreams, frameInfo.maxBufSurfAddrSize,
            frameInfo.maxStreams, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
            NvMotDataTypes::Size(m_params.m_width, m_params.m_height), m_params.m_colorFormat, m_params.m_inputOrder
        );
        m_subImageCropper2 = NULL;
    }
    else
    {
        m_subImageCropper2 = std::make_shared<ReIDFrameTransformTexture>();
        m_subImageCropper2->initialize(
            m_params.m_debugVerboseLevel, frameInfo.maxStreams, frameInfo.maxBufSurfAddrSize,
            frameInfo.maxStreams, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
            NvMotDataTypes::Size(m_params.m_width, m_params.m_height), m_params.m_colorFormat, m_params.m_inputOrder
        );
        m_subImageCropScaler = nullptr;
    }

    CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );
    CUBLAS_CHECK( cublasCreate(&m_cublasHandle) );
    CUBLAS_CHECK( cublasSetStream(m_cublasHandle, m_cuStream) );

    m_imageEncoder.setCudaStream(m_cuStream);
    m_maskDecoder.setCudaStream(m_cuStream);

    /// Allocate segmentation data fields.
    /// Decoder input.
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_featuresEncoder_d),
            m_paramsRt.featuresEncoderSize * frameInfo.maxStreams * sizeof(float)
        ))
    );
    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_points_d), m_paramsRt.maxPointNum * 2 * sizeof(float))));
    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_pointLabels_d), m_paramsRt.maxPointNum * sizeof(float))));
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_inputMask_d), m_params.m_hiddenDim * m_params.m_hiddenDim * sizeof(float)
        ))
    );
    CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_inputMaskLabel_d), 1 * sizeof(float))));

    /// Decoder output.
    CUDA_CHECK(
        (cudaMalloc(reinterpret_cast<void **>( &m_targetMaskDecSize_d), m_paramsRt.basicMaskSize * sizeof(float)))
    );
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_outputIou_d), m_paramsRt.batchSize * m_params.m_outputLabelNum * sizeof(float)
        ))
    );
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_outputOccLogit_d),
            m_paramsRt.batchSize * m_params.m_outputLabelNum * sizeof(float)
        ))
    );

    /// Store batch mask result.
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_batchMaskDecSize_d),
            frameInfo.maxStreams * maxTargetsPerStream * m_params.m_hiddenDim * m_params.m_hiddenDim * sizeof(float)
        ))
    );
    /// Mask resized to original streammux resolution.
    CUDA_CHECK(
        (cudaMalloc(
            reinterpret_cast<void **>( &m_targetMaskOriginalSrcSize_d),
            frameInfo.originalSrcImageSize.width * frameInfo.originalSrcImageSize.height * sizeof(float)
        ))
    );

    CUDA_CHECK(
        (cudaMallocHost(reinterpret_cast<void **>( &m_targetMaskDecSize_h), m_paramsRt.basicMaskSize * sizeof(float)))
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            m_batchMaskDecSize_d, 0.f,
            frameInfo.maxStreams * maxTargetsPerStream * m_params.m_hiddenDim * m_params.m_hiddenDim * sizeof(float),
            m_cuStream
        )
    );

    /// Regular mask size, equals to m_hiddenDim=256.
    SegmentationInBatch segInfo = {m_batchMaskDecSize_d, {m_params.m_hiddenDim, m_params.m_hiddenDim}};
    m_targetManager->setSegmentationBatchInfo(std::move(segInfo));

    if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
    {
        /// SAM2 image encoder has high res feature output m_highResFeaturesEncoder0_d: 32 x 256 x 256.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_highResFeaturesEncoder0_d),
                m_paramsRt.highResFeaturesEncoder0Size * frameInfo.maxStreams * sizeof(float)
            ))
        );
        /// m_highResFeaturesEncoder1_d: 64 x 128 x 128.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_highResFeaturesEncoder1_d),
                m_paramsRt.highResFeaturesEncoder1Size * frameInfo.maxStreams * sizeof(float)
            ))
        );

        /// SAM2 image encoder also has image position embedding output m_visionPosEmbed_d: 4096 x 1 x 256.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_visionPosEmbed_d),
                m_paramsRt.featuresEncoderSize * frameInfo.maxStreams * sizeof(float)
            ))
        );
        /// m_visionFeats_d: 256 x 64 x 64.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_visionFeats_d),
                m_paramsRt.featuresEncoderSize * frameInfo.maxStreams * sizeof(float)
            ))
        );

        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_pix_feat_with_mem_d),
                m_paramsRt.batchSize * m_paramsRt.pixFeatWithMemSize * sizeof(float)
            ))
        );
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_batchPixFeatWithMem_d),
                frameInfo.maxStreams * maxTargetsPerStream * m_paramsRt.pixFeatWithMemSize * sizeof(float)
            ))
        );
        /// SAM2 mask decoder has more outputs m_maskForMem_d: high resolution mask for mem decoder 1 x 1024 x 1024.
        CUDA_CHECK(
            (cudaMalloc(reinterpret_cast<void **>( &m_maskForMem_d), m_paramsRt.highResMaskSize * sizeof(float)))
        );
        /// m_obj_ptr_d: 256.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_obj_ptr_d), m_paramsRt.batchSize * m_paramsRt.objPtrSize * sizeof(float)
            ))
        );
        CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_outputSize_d), 2 * sizeof(int64_t))));
        /// Regular mask size, equals to m_hiddenDim=256.
        int64_t outputSize_h[2] = {m_params.m_hiddenDim, m_params.m_hiddenDim};
        CUDA_CHECK(
            cudaMemcpyAsync( m_outputSize_d, outputSize_h, 2 * sizeof(int64_t), cudaMemcpyHostToDevice, m_cuStream)
        );
        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );
    }

    if (m_params.m_useMemBank)
    {
        /// Allocate the legacy work spaces.
        CUDA_CHECK(
            (cudaMalloc(reinterpret_cast<void **>( &m_work_0_d), m_paramsRt.featuresEncoderSize * sizeof(float)))
        );
        CUDA_CHECK(
            (cudaMalloc(reinterpret_cast<void **>( &m_work_1_d), m_paramsRt.featuresEncoderSize * sizeof(float)))
        );
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_work_2_d),
                m_params.m_maxMemVecSize_legacy * m_paramsRt.objPtrSize * sizeof(float)
            ))
        );

        /// Memory attention m_memory_d not used.
        CUDA_CHECK(
            (cudaMalloc(reinterpret_cast<void **>( &m_memory_d), m_paramsRt.maxConcatMemFeatSize * sizeof(float)))
        );
        /// m_memory_obj_ptr_d: [b x] (15 + 1) x 256.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_memory_obj_ptr_d),
                m_paramsRt.batchSize * m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1) * sizeof(float)
            ))
        );
        /// m_memory_maskmem_features_d: [b x] 7 x 64 x 64 x 64.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_memory_maskmem_features_d),
                m_paramsRt.batchSize * m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem * sizeof(float)
            ))
        );

        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_memory_pos_embed_d),
                m_paramsRt.batchSize * m_paramsRt.maxConcatMemFeatSize * sizeof(float)
            ))
        );
        /// m_cond_frame_id_diff_d: [b x] 1 x 16.
        CUDA_CHECK((cudaMalloc(reinterpret_cast<void **>( &m_cond_frame_id_diff_d), 1 * sizeof(float))));

        /// Memory encoder m_maskmem_features_d: [b x] 64 x 64 x 64.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_maskmem_features_d),
                m_paramsRt.batchSize * m_paramsRt.maskMemFeatSize * sizeof(float)
            ))
        );
        /// m_maskmem_pos_enc_d: [b x] 64 x 64 x 64.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_maskmem_pos_enc_d),
                m_paramsRt.batchSize * m_paramsRt.maskMemFeatSize * sizeof(float)
            ))
        );
        /// m_temporal_code_d: 64 x 7.
        CUDA_CHECK(
            (cudaMalloc(
                reinterpret_cast<void **>( &m_temporal_code_d),
                m_params.m_numMaskMem * m_paramsRt.temporalCodeSize * sizeof(float)
            ))
        );

        m_cudaMemBank.initialize(frameInfo, maxTargetsPerStream, m_paramsRt, m_params, m_cuStream);

        /// Set memory cuda array as 0.
        CUDA_CHECK(cudaMemsetAsync(m_memory_d, 0, m_paramsRt.maxConcatMemFeatSize * sizeof(float), m_cuStream));
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_obj_ptr_d, 0,
                m_paramsRt.batchSize * m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1) * sizeof(float), m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_maskmem_features_d, 0,
                m_paramsRt.batchSize * m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem * sizeof(float), m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_pos_embed_d, 0,
                m_paramsRt.batchSize * m_paramsRt.maxConcatMemFeatSize * sizeof(float), m_cuStream
            )
        );
        CUDA_CHECK(cudaMemsetAsync(m_cond_frame_id_diff_d, 0, 1 * sizeof(float), m_cuStream));
    }

    if (m_params.m_debugVerboseLevel > 1)
    {
        std::cout << "[Segmenter::initializeRuntime] Tensor size: (assuming batch size = 1)" << std::endl <<
        "[Image Encoder]" << std::endl <<
        "Tensor name: m_featuresEncoder_d, size: " << m_paramsRt.featuresEncoderSize << std::endl <<
        "Tensor name: m_highResFeaturesEncoder0_d, size: " << m_paramsRt.highResFeaturesEncoder0Size << std::endl <<
        "Tensor name: m_highResFeaturesEncoder1_d, size: " << m_paramsRt.highResFeaturesEncoder1Size << std::endl <<
        "Tensor name: m_visionFeats_d, size: " << m_paramsRt.featuresEncoderSize << std::endl <<
        "Tensor name: m_visionPosEmbed_d, size: " << m_paramsRt.featuresEncoderSize << std::endl <<
        "[Mask Decoder]" << std::endl <<
        "Tensor name: m_points_d, size: " << m_paramsRt.maxPointNum * 2 << std::endl <<
        "Tensor name: m_pointLabels_d, size: " << m_paramsRt.maxPointNum << std::endl <<
        "Tensor name: m_outputSize_d, size: " << 2 << std::endl <<
        "Tensor name: m_pix_feat_with_mem_d, size: " << m_paramsRt.pixFeatWithMemSize << std::endl <<
        "Tensor name: m_highResFeaturesEncoder0_d, size: " << m_paramsRt.highResFeaturesEncoder0Size << std::endl <<
        "Tensor name: m_highResFeaturesEncoder1_d, size: " << m_paramsRt.highResFeaturesEncoder1Size << std::endl <<
        "Tensor name: m_obj_ptr_d, size: " << m_paramsRt.objPtrSize << std::endl <<
        "Tensor name: m_targetMaskDecSize_d, size: " << m_paramsRt.basicMaskSize << std::endl <<
        "Tensor name: m_maskForMem_d, size: " << m_paramsRt.highResMaskSize << std::endl <<
        "[Memory Attention]" << std::endl <<
        "Tensor name: m_visionFeats_d, size: " << m_paramsRt.featuresEncoderSize << std::endl <<
        "Tensor name: m_visionPosEmbed_d, size: " << m_paramsRt.featuresEncoderSize << std::endl <<
        "Tensor name: m_memory_obj_ptr_d, size: " << m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1) << std::endl <<
        "Tensor name: m_memory_maskmem_features_d, size: " << m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem << std::endl <<
        "Tensor name: m_memory_pos_embed_d, size: " << m_paramsRt.maxConcatMemFeatSize << std::endl <<
        "Tensor name: m_cond_frame_id_diff_d, size: " << 1 << std::endl <<
        "Tensor name: m_pix_feat_with_mem_d, size: " << m_paramsRt.pixFeatWithMemSize << std::endl <<
        "[Memory Encoder]" << std::endl <<
        "Tensor name: m_maskForMem_d, size: " << m_paramsRt.highResMaskSize << std::endl <<
        "Tensor name: m_featuresEncoder_d, size: " << m_paramsRt.featuresEncoderSize << std::endl <<
        "Tensor name: m_maskmem_features_d, size: " << m_paramsRt.maskMemFeatSize << std::endl <<
        "Tensor name: m_maskmem_pos_enc_d, size: " << m_paramsRt.maskMemFeatSize << std::endl <<
        "Tensor name: m_temporal_code_d, size: " << m_params.m_numMaskMem * m_paramsRt.temporalCodeSize << std::endl;
    }

    initializeTrtBindings();
    m_isInit = true;
}

/** @brief Bind persistent buffers to the segmentation engines. */
void Segmenter::initializeTrtBindings()
{
    if (m_params.m_segmenterType == SEGMENTER_TYPES::NanoSAM || m_params.m_segmenterType == SEGMENTER_TYPES::EfficientVitSAM)
    {
        m_bindingsImgEncoder.resize(2);
        /// Leave the image input binding to be set at runtime.
        m_bindingsImgEncoder[1] = m_featuresEncoder_d;
    }
    else if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
    {
        m_bindingsImgEncoder.resize(6);
        /// Output bindings; the image input m_bindingsImgEncoder[0] is set at runtime.
        m_bindingsImgEncoder[1] = m_featuresEncoder_d;
        m_bindingsImgEncoder[2] = m_highResFeaturesEncoder0_d;
        m_bindingsImgEncoder[3] = m_highResFeaturesEncoder1_d;
        m_bindingsImgEncoder[4] = m_visionFeats_d;
        m_bindingsImgEncoder[5] = m_visionPosEmbed_d;
    }

    if (m_params.m_segmenterType == SEGMENTER_TYPES::NanoSAM)
    {
        m_bindingsMaskDecoder.resize(7);
        m_bindingsMaskDecoder[0] = m_featuresEncoder_d;
        m_bindingsMaskDecoder[1] = m_points_d;
        m_bindingsMaskDecoder[2] = m_pointLabels_d;
        m_bindingsMaskDecoder[3] = m_inputMask_d;
        m_bindingsMaskDecoder[4] = m_inputMaskLabel_d;
        m_bindingsMaskDecoder[5] = m_targetMaskDecSize_d;
        m_bindingsMaskDecoder[6] = m_outputIou_d;
    }
    else if (m_params.m_segmenterType == SEGMENTER_TYPES::EfficientVitSAM)
    {
        m_bindingsMaskDecoder.resize(5);
        m_bindingsMaskDecoder[0] = m_featuresEncoder_d;
        m_bindingsMaskDecoder[1] = m_points_d;
        m_bindingsMaskDecoder[2] = m_pointLabels_d;
        m_bindingsMaskDecoder[3] = m_targetMaskDecSize_d;
        m_bindingsMaskDecoder[4] = m_outputIou_d;
    }
    else if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
    {
        m_bindingsMaskDecoder.resize(10);
        m_bindingsMaskDecoder[0] = m_points_d;
        m_bindingsMaskDecoder[1] = m_pointLabels_d;
        /// Use features with memory if memory bank is enabled.
        m_bindingsMaskDecoder[2] = m_pix_feat_with_mem_d;
        m_bindingsMaskDecoder[3] = m_highResFeaturesEncoder0_d;
        m_bindingsMaskDecoder[4] = m_highResFeaturesEncoder1_d;
        /// Out.
        m_bindingsMaskDecoder[5] = m_obj_ptr_d;
        m_bindingsMaskDecoder[6] = m_maskForMem_d;
        m_bindingsMaskDecoder[7] = m_targetMaskDecSize_d;
        m_bindingsMaskDecoder[8] = m_outputIou_d;
        m_bindingsMaskDecoder[9] = m_outputOccLogit_d;
    }

    if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2 && m_params.m_useMemBank)
    {
        m_bindingsMemAtten.resize(7);
        m_bindingsMemAtten[0] = m_visionFeats_d;
        m_bindingsMemAtten[1] = m_visionPosEmbed_d;
        m_bindingsMemAtten[2] = m_memory_obj_ptr_d;
        m_bindingsMemAtten[3] = m_memory_maskmem_features_d;
        m_bindingsMemAtten[4] = m_memory_pos_embed_d;
        m_bindingsMemAtten[5] = m_cond_frame_id_diff_d;
        m_bindingsMemAtten[6] = m_pix_feat_with_mem_d;

        m_bindingsMemEncoder.resize(6);
        m_bindingsMemEncoder[0] = m_maskForMem_d;
        m_bindingsMemEncoder[1] = m_featuresEncoder_d;
        m_bindingsMemEncoder[2] = m_outputOccLogit_d;
        m_bindingsMemEncoder[3] = m_maskmem_features_d;
        m_bindingsMemEncoder[4] = m_maskmem_pos_enc_d;
        m_bindingsMemEncoder[5] = m_temporal_code_d;
    }
}

/**
 * @brief Select per-stream tensor addresses for inference.
 * @param[in] inputRoiInd Region index within the input batch.
 */
void Segmenter::updateTrtBindingsPerStream(const uint inputRoiInd)
{
    if (m_params.m_segmenterType != SEGMENTER_TYPES::SAM2)
    {
        throw std::runtime_error("[FAILED] updateTrtBindingsPerStream() only supports SAM2");
    }

    /// Update image encoder bindings based on stream index.
    m_bindingsImgEncoder[0] = m_inputBuf_d + inputRoiInd * m_params.m_width * m_params.m_height * m_params.m_channel;
    m_bindingsImgEncoder[1] = m_featuresEncoder_d + inputRoiInd * m_paramsRt.featuresEncoderSize;
    m_bindingsImgEncoder[2] = m_highResFeaturesEncoder0_d + inputRoiInd * m_paramsRt.highResFeaturesEncoder0Size;
    m_bindingsImgEncoder[3] = m_highResFeaturesEncoder1_d + inputRoiInd * m_paramsRt.highResFeaturesEncoder1Size;
    m_bindingsImgEncoder[4] = m_visionFeats_d + inputRoiInd * m_paramsRt.featuresEncoderSize;
    m_bindingsImgEncoder[5] = m_visionPosEmbed_d + inputRoiInd * m_paramsRt.featuresEncoderSize;

    /// Update mask decoder bindings based on stream index.
    m_bindingsMaskDecoder[3] = m_highResFeaturesEncoder0_d + inputRoiInd * m_paramsRt.highResFeaturesEncoder0Size;
    m_bindingsMaskDecoder[4] = m_highResFeaturesEncoder1_d + inputRoiInd * m_paramsRt.highResFeaturesEncoder1Size;

    if (m_params.m_useMemBank)
    {
        /// Update memory attention bindings based on stream index.
        m_bindingsMemAtten[0] = m_visionFeats_d + inputRoiInd * m_paramsRt.featuresEncoderSize;
        m_bindingsMemAtten[1] = m_visionPosEmbed_d + inputRoiInd * m_paramsRt.featuresEncoderSize;

        /// Update memory encoder bindings based on stream index.
        m_bindingsMemEncoder[1] = m_featuresEncoder_d + inputRoiInd * m_paramsRt.featuresEncoderSize;
    }
}

/** @brief Release resources owned by the Segmenter instance. */
Segmenter::~Segmenter()
{
    if (m_isInit)
    {
        CUDA_CHECK( cudaFree(m_featuresEncoder_d) );
        CUDA_CHECK( cudaFree(m_points_d) );
        CUDA_CHECK( cudaFree(m_pointLabels_d) );
        CUDA_CHECK( cudaFree(m_inputMask_d) );
        CUDA_CHECK( cudaFree(m_inputMaskLabel_d) );
        CUDA_CHECK( cudaFree(m_pix_feat_with_mem_d));
        CUDA_CHECK( cudaFree(m_batchPixFeatWithMem_d));
        CUDA_CHECK( cudaFree(m_targetMaskDecSize_d) );
        CUDA_CHECK( cudaFree(m_outputIou_d) );
        CUDA_CHECK( cudaFree(m_outputOccLogit_d) );
        CUDA_CHECK( cudaFree(m_batchMaskDecSize_d) );
        CUDA_CHECK( cudaFree(m_targetMaskOriginalSrcSize_d) );
        CUDA_CHECK( cudaFreeHost(m_targetMaskDecSize_h) );

        if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
        {
            CUDA_CHECK( cudaFree(m_highResFeaturesEncoder0_d));
            CUDA_CHECK( cudaFree(m_highResFeaturesEncoder1_d));
            CUDA_CHECK( cudaFree(m_visionPosEmbed_d));
            CUDA_CHECK( cudaFree(m_visionFeats_d));
            CUDA_CHECK( cudaFree(m_maskForMem_d));
            CUDA_CHECK( cudaFree(m_obj_ptr_d));
            CUDA_CHECK( cudaFree(m_outputSize_d));
        }
        if (m_params.m_useMemBank)
        {
            CUDA_CHECK( cudaFree(m_work_0_d));
            CUDA_CHECK( cudaFree(m_work_1_d));
            CUDA_CHECK( cudaFree(m_work_2_d));
            CUDA_CHECK( cudaFree(m_memory_d));
            CUDA_CHECK( cudaFree(m_memory_obj_ptr_d));
            CUDA_CHECK( cudaFree(m_memory_maskmem_features_d));
            CUDA_CHECK( cudaFree(m_memory_pos_embed_d));
            CUDA_CHECK( cudaFree(m_cond_frame_id_diff_d));
            CUDA_CHECK( cudaFree(m_maskmem_features_d));
            CUDA_CHECK( cudaFree(m_maskmem_pos_enc_d));
            CUDA_CHECK( cudaFree(m_temporal_code_d));
        }

        CUBLAS_CHECK( cublasDestroy(m_cublasHandle) );
        CUDA_CHECK(cudaStreamDestroy(m_cuStream));
        m_subImageCropper2.reset();
        m_subImageCropScaler.reset();
    }
}

/**
 * @brief Populate the capabilities reported to the tracker.
 * @param[out] paramsQuery The output query parameters.
 */
void Segmenter::queryParams(NvMOTQuery &paramsQuery)
{
    paramsQuery.numTransforms = 1;
}

/**
 * @brief Crop target regions from the input frame batch.
 * @param[in] inputMeta Input frames keyed by stream ID.
 */
void Segmenter::extractImagePatchesBatch(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta)
{
    m_inputRoisInBatch.clear();
    NvMotDataTypes::Rect2f roiWholeFrame = CV_FUNCS::constructRect2f(0, 0, m_srcImageSize.width, m_srcImageSize.height);
    uint numBboxes = 0;
    /// Build one whole-frame region of interest for each stream.
    for (auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
    {
        NvMOTStreamId videoStreamId = streamIt->first;
        m_inputRoisInBatch[videoStreamId].push_back(std::make_pair(numBboxes++, roiWholeFrame));
    }

    if(m_params.m_useVPICropScaler)
    {
        /// Crop and scale input frames using VPI crop scaler, perform format conversion.
        if (!m_subImageCropScaler->frameTransformInBatch(m_inputRoisInBatch,
                                                         numBboxes - 1, m_params.m_netScaleFactor, m_params.m_offsets, inputMeta))
        {
            throw std::runtime_error("[FAILED] m_subImageCropScaler->frameTransformInBatch()");
        }

        m_inputBuf_d = m_subImageCropScaler->getOutputImagePatchPtrs();
    }
    else
    {
        /// Prepare input texture object.
        m_subImageCropper2->updateTextureObject(inputMeta);

        /// Crop, rescale and convert image format.
        /// May change dimension order, perform image normalization and convert to float.
        if (m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
        {
            if (!m_subImageCropper2->frameTransformInBatch_NV12(m_inputRoisInBatch,
                                                                numBboxes - 1, m_params.m_netScaleFactor, m_params.m_offsets))
            {
                throw std::runtime_error("[FAILED] m_subImageCropper2->formatConvert_batch_NV12()");
            }
        }
        else if (m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
        {
            if (!m_subImageCropper2->frameTransformInBatch(m_inputRoisInBatch,
                                                            numBboxes - 1, m_params.m_netScaleFactor, m_params.m_offsets))
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
 * @brief Mark a single decoder prompt point as negative.
 * @param[in] pointNumPerTarget Number of prompt points allocated to each target.
 * @param[in] indInBatch Target index within the inference batch.
 */
void Segmenter::setMaskDecoderSinglePointLabelAsNegative(int pointNumPerTarget, int indInBatch)
{
    /// Empty point label -1.
    std::vector<float> pointLabels_h(pointNumPerTarget, -1.0);
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_pointLabels_d + pointNumPerTarget * indInBatch, pointLabels_h.data(),
            pointNumPerTarget * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
        )
    );
}

/**
 * @brief Populate decoder prompt labels for the configured prompt type.
 * @param[in] promptType Prompt representation selected for the decoder.
 */
void Segmenter::setMaskDecoderPointLabels(int promptType)
{
    if (promptType == 0)
    {
        /// Use foreground point as prompt.
        std::vector<float> pointLabels_h(m_maskDecoder.m_params.m_maxBatchSize, SAM_POINT_TYPE::FOREGROUND);
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pointLabels_d, pointLabels_h.data(),
                m_maskDecoder.m_params.m_maxBatchSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
            )
        );
    }
    else if (promptType == 1)
    {
        /// Use box as prompt.
        /// Labels 2 and 3 identify the top-left and bottom-right bounding-box corners.
        std::vector<float> pointLabels_h;
        for (uint oi=0; oi < (uint)m_maskDecoder.m_params.m_maxBatchSize; oi++)
        {
            pointLabels_h.push_back(SAM_POINT_TYPE::TOP_LEFT);
            pointLabels_h.push_back(SAM_POINT_TYPE::BOTTOM_RIGHT);
        }
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pointLabels_d, pointLabels_h.data(),
                2 * m_maskDecoder.m_params.m_maxBatchSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
            )
        );
    }
    else if (promptType == -1)
    {
        /// Empty point label -1.
        std::vector<float> pointLabels_h(m_maskDecoder.m_params.m_maxBatchSize, -1.0);
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pointLabels_d, pointLabels_h.data(),
                m_maskDecoder.m_params.m_maxBatchSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
            )
        );
    }

    if (m_params.m_segmenterType == SEGMENTER_TYPES::NanoSAM)
    {
        CUDA_CHECK(
            cudaMemsetAsync(
                m_inputMask_d, 0.f, m_params.m_hiddenDim * m_params.m_hiddenDim * sizeof(float), m_cuStream
            )
        );
        CUDA_CHECK(cudaMemsetAsync( m_inputMaskLabel_d, 0.f, 1 * sizeof(float), m_cuStream));
    }
}

/**
 * @brief Populate decoder point coordinates from target prompts.
 * @param[in] usePrompt Whether decoder prompt data is enabled.
 * @param[in] targetList Targets included in the inference operation.
 */
void Segmenter::setMaskDecoderPoints(const bool usePrompt, std::vector<std::shared_ptr<NvTrackedObject>> targetList)
{
    std::vector<float> points_h(m_maskDecoder.m_params.m_maxBatchSize * 4, 0.0);
    if (m_params.m_useMemBank && (!usePrompt))
    {
        /// Use memory bank to track when there is no detection input.
        /// No points are provided, so pad with an empty point (with label -1) and keep points_h at zero.
        setMaskDecoderPointLabels(-1);
    }
    else
    {
        setMaskDecoderPointLabels(m_params.m_promptType);

        /// Get prompt bounding box.
        /// If current target has detection, use detection bbox (partial bbox) as prompt.
        /// Otherwise use localized progressed bbox from prev frame as prompt.
        for (uint indInBatch = 0; indInBatch < targetList.size(); indInBatch++)
        {
            auto target = targetList.at(indInBatch);
            NvMotDataTypes::Rect2f tBbox;
            if (target->getMatchedDetObj() != nullptr && target->getMatchedDetObj()->objIn != nullptr)
            {
                const NvMOTObjToTrack *objIn = target->getMatchedDetObj()->objIn;
                tBbox = CV_FUNCS::constructRect2f(objIn->bbox.x, objIn->bbox.y, objIn->bbox.width, objIn->bbox.height);
            }
            else
            {
                if (m_params.m_useMemBank)
                {
                    setMaskDecoderSinglePointLabelAsNegative(2, indInBatch);
                    continue;
                }
                else
                {
                    tBbox = target->getTargetInfo().tBbox;
                }
            }

            if (tBbox.isEmpty())
            {
                throw std::runtime_error("[Segmenter::setMaskDecoderPoints] tBbox is empty");
            }

            if (m_params.m_promptType == 0)
            {
                /// Center.
                points_h[2 * indInBatch]     = (tBbox.min()(0) + tBbox.max()(0)) / m_scaleMask2Src(0) / 2;
                points_h[2 * indInBatch + 1] = (tBbox.min()(1) + tBbox.max()(1)) / m_scaleMask2Src(1) / 2;
            }
            else if (m_params.m_promptType == 1)
            {
                /// Bbox top-left and bottom-right.
                float pad = m_params.m_promptBboxPaddingScale * tBbox.sizes()(1);
                points_h[4 * indInBatch]     = (tBbox.min()(0)-pad)/m_scaleMask2Src(0);
                points_h[4 * indInBatch + 1] = (tBbox.min()(1)-pad)/m_scaleMask2Src(1);
                points_h[4 * indInBatch + 2] = (tBbox.max()(0)+pad)/m_scaleMask2Src(0);
                points_h[4 * indInBatch + 3] = (tBbox.max()(1)+pad)/m_scaleMask2Src(1);
            }
        }
    }

    CUDA_CHECK(
        cudaMemcpyAsync(
            m_points_d, points_h.data(), m_maskDecoder.m_params.m_maxBatchSize * 4 * sizeof(float),
            cudaMemcpyHostToDevice, m_cuStream
        )
    );
    CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
    CUDA_CHECK( cudaPeekAtLastError() );
}

/** @brief Run image-encoder inference. */
void Segmenter::imgEncoderInfer()
{
    for (auto streamIt = m_inputRoisInBatch.begin(); streamIt != m_inputRoisInBatch.end(); ++streamIt)
    {
        /// Define per frame info.
        uint inputRoiInd = streamIt->second.at(0).first;

        /// Set image as input binding.
        updateTrtBindingsPerStream(inputRoiInd);
        m_imageEncoder.submitInput(m_bindingsImgEncoder, {1}, m_cuStream);
        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );
        if (m_params.m_debugVerboseLevel > 3)
        {
            std::cout << "Image encoder submit input done" << std::endl;
        }
    }
}

/**
 * @brief Run mask-decoder inference for a target batch.
 * @param[in] usePrompt Whether decoder prompt data is enabled.
 * @param[in] targetList Targets included in the inference operation.
 * @param[out] outputLabelIndMax Index of the output label with the highest IoU.
 * @param[in] streamInd Index of the input stream being processed.
 * @param[in] frameNum Frame number associated with the inference result.
 */
void Segmenter::maskDecoderInfer(
    bool usePrompt, std::vector<std::shared_ptr<NvTrackedObject>>& targetList,
    int& outputLabelIndMax, const uint streamInd, const NvMOTFrameNum frameNum
)
{
    if (m_params.m_debugVerboseLevel > 3)
    {
        std::cout << "Mask decoder preparation" << std::endl;
    }
    bool targetStatusEmpty = isInferStatusEmpty(targetList.front()->getTargetInd());

    /// Use the prompt when the memory bank is unavailable or empty, or on a prompt-update frame during update.
    if ( (!m_params.m_useMemBank) || targetStatusEmpty ||
        ( frameNum % (m_params.m_maskPromptUpdateInterval + 1) == 0 && (!m_isForLocalization)) )
    {
        usePrompt = true;
    }
    else
    {
        usePrompt = false;
    }

    /// Set point labels for bbox only.
    /// TODO: Load external positive and negative points and masks.
    setMaskDecoderPoints(usePrompt, targetList);

    int pointNum = 0;
    /// When not using prompt and using memory bank only, there is only one point (0,0) label -1.
    if (m_params.m_promptType == 0 || (!usePrompt))
    {
        pointNum = 1;
    }
    else if (m_params.m_promptType == 1)
    {
        pointNum = 2;
    }
    m_maskDecoder.submitInput(m_bindingsMaskDecoder, {pointNum, int(targetList.size())}, m_cuStream);
    if (m_params.m_debugVerboseLevel > 3)
    {
        cout << "pointNum: " << pointNum << endl;
        std::cout << "Mask decoder submit input done" << std::endl;
    }

    outputLabelIndMax = 0;
    m_outputIou_h = std::vector<float> (m_paramsRt.batchSize * m_params.m_outputLabelNum, 0.0);
    m_outputOccLogit_h = std::vector<float> (m_paramsRt.batchSize * m_params.m_outputLabelNum, 0.0);
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_outputIou_h.data(), m_outputIou_d,
            m_paramsRt.batchSize * m_params.m_outputLabelNum * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_outputOccLogit_h.data(), m_outputOccLogit_d,
            m_paramsRt.batchSize * m_params.m_outputLabelNum * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
        )
    );
    CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
    CUDA_CHECK( cudaPeekAtLastError() );

    if (m_params.m_outputLabelNum > 1)
    {
        if (m_params.m_segmenterType == SEGMENTER_TYPES::SAM2)
        {
            throw std::runtime_error("[Segmenter::maskDecoderInfer] outputLabelNum must be 1 for SAM2");
        }

        std::vector<float> outputIou_h(m_params.m_outputLabelNum, 0.0);
        CUDA_CHECK(
            cudaMemcpyAsync(
                outputIou_h.data(), m_outputIou_d,
                m_params.m_outputLabelNum * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
            )
        );
        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );
        float outputIouMax = -1.0;
        for (auto oi = 0; oi < m_params.m_outputLabelNum; oi++)
        {
            if (outputIou_h[oi] > outputIouMax)
            {
                outputLabelIndMax = oi;
                outputIouMax = outputIou_h[oi];
            }
        }
    }
}

/**
 * @brief Run memory attention for one target.
 * @param[in] target Tracked target to process.
 * @param[in] indInBatch Target index within the inference batch.
 * @param[in] streamInd Index of the input stream being processed.
 */
void Segmenter::memAttenInfer(const std::shared_ptr<NvTrackedObject> target, const int indInBatch, const int streamInd)
{
    if (m_params.m_debugVerboseLevel > 3)
    {
        std::cout << "Memory attention target age: " << target->getAge() << std::endl;
    }

    /// If memory bank is not used or memory bank is empty, copy featuresEncoder_d (at stream ind location) to pix_feat_with_mem_d (at indInBatch location).
    if ( (!m_params.m_useMemBank) || isInferStatusEmpty(target->getTargetInd()))
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pix_feat_with_mem_d + indInBatch * m_paramsRt.pixFeatWithMemSize,
                m_featuresEncoder_d + streamInd * m_paramsRt.featuresEncoderSize,
                m_paramsRt.featuresEncoderSize * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        return;
    }

    /// For update, reuse the mem atten output from localization.
    if (!m_isForLocalization)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_pix_feat_with_mem_d + indInBatch * m_paramsRt.pixFeatWithMemSize,
                m_batchPixFeatWithMem_d + target->getTargetInd() * m_paramsRt.pixFeatWithMemSize,
                m_paramsRt.pixFeatWithMemSize * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        return;
    }

    int numObjPtrs = 0, numMaskMem = 0;
    if (m_params.m_useCudaMemBank)
    {
        MEAS_TIME(
            fillCudaMemoryConditionedFeatures(target->getTargetInd(), indInBatch),
            m_params.m_debugVerboseLevel > 2, "[Segmenter::memAttenInfer] fillCudaMemoryConditionedFeatures"
        );
        /// Here obj ptr first and obj ptr uncond are the same cuda array, and numObjPtrUncond is the sum of these two parts.
        numObjPtrs = m_cudaInferenceStatus.at(target->getTargetInd()).numObjPtrUncond;
        numMaskMem = m_cudaInferenceStatus.at(target->getTargetInd()).numStatusUncond;
    }
    else
    {
        MEAS_TIME(
            fillMemoryConditionedFeatures(target->getTargetInd(), indInBatch),
            m_params.m_debugVerboseLevel > 2, "[Segmenter::memAttenInfer] fillMemoryConditionedFeatures"
        );
        /// Construct dynamic size {numObjPtrs, numMaskMem, numFrames}.
        numObjPtrs = m_inferStatus.at(target->getTargetInd()).objPtrUncond.size() + 1;
        numMaskMem = m_inferStatus.at(target->getTargetInd()).statusUncond.size() + 1;
    }
    /// Limit the position-encoding history using the target age, which starts at one.
    std::vector<int> sizes = {1, numObjPtrs, numMaskMem, std::min(numObjPtrs, m_params.m_numObjPtr+1)};

    /// Print the runtime tensor shapes for debugging.
    if (m_params.m_debugVerboseLevel > 3)
    {
        std::cout << "[Segmenter::memAttenInfer] Set runtime tensor shapes: (" <<
        numObjPtrs * 256 << "," << numMaskMem*64*64*64 << ", " << (numMaskMem*4096 + 4*sizes[3])*64 << "), " <<
        "numObjPtrs, numMaskMem, target->getAge()-1, m_params.m_numObjPtr+1 = " <<
        numObjPtrs << ", " << numMaskMem << ", " << int(target->getAge())-1 << ", " << m_params.m_numObjPtr+1 << std::endl;
    }

    /// Set memory attention output as indInBatch location in pix_feat_with_mem_d.
    m_bindingsMemAtten[5] = m_pix_feat_with_mem_d + indInBatch * m_paramsRt.pixFeatWithMemSize;
    m_memoryAttention.submitInput(m_bindingsMemAtten, sizes, m_cuStream);
    if (m_params.m_debugVerboseLevel > 3)
    {
        std::cout << "Memory attention submit input done" << std::endl;
    }

    /// For localization, store each target's mem atten output for reuse.
    if (m_isForLocalization)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_batchPixFeatWithMem_d + target->getTargetInd() * m_paramsRt.pixFeatWithMemSize, m_bindingsMemAtten[5],
                m_paramsRt.pixFeatWithMemSize * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
    }
}

/**
 * @brief Assemble and run batched memory-attention input.
 * @param[in] targetList Targets included in the inference operation.
 * @param[in] streamInd Index of the input stream being processed.
 */
void Segmenter::memAttenInferBatchParser(std::vector<std::shared_ptr<NvTrackedObject>>& targetList, const int streamInd)
{
    if (m_params.m_useMemBank)
    {
        int numObjPtrsMax = 0, numMaskMemMax = 0;
        /// Set memory cuda array as 0.
        CUDA_CHECK(cudaMemsetAsync(m_memory_d, 0, m_paramsRt.maxConcatMemFeatSize * sizeof(float), m_cuStream));
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_obj_ptr_d, 0,
                m_paramsRt.batchSize * m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1) * sizeof(float), m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_maskmem_features_d, 0,
                m_paramsRt.batchSize * m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem * sizeof(float), m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_pos_embed_d, 0,
                m_paramsRt.batchSize * m_paramsRt.maxConcatMemFeatSize * sizeof(float), m_cuStream
            )
        );

        for (int indInBatch = 0; indInBatch < int(targetList.size()); indInBatch++)
        {
            std::shared_ptr<NvTrackedObject> target = targetList.at(indInBatch);
            if (m_params.m_debugVerboseLevel > 3)
            {
                std::cout << "Memory attention target age: " << target->getAge() << std::endl;
            }

            /// Skip memory attention when the memory bank is unused or empty.
            if ( (!m_params.m_useMemBank) || isInferStatusEmpty(target->getTargetInd()))
            {
                continue;
            }

            int numObjPtrs = 0, numMaskMem = 0;
            if (m_params.m_useCudaMemBank)
            {
                MEAS_TIME(
                    fillCudaMemoryConditionedFeatures(target->getTargetInd(), indInBatch),
                    m_params.m_debugVerboseLevel > 2, "[Segmenter::memAttenInfer] fillCudaMemoryConditionedFeatures"
                );
                /// Here obj ptr first and obj ptr uncond are the same cuda array, and numObjPtrUncond is the sum of these two parts.
                numObjPtrs = m_cudaInferenceStatus.at(target->getTargetInd()).numObjPtrUncond;
                numMaskMem = m_cudaInferenceStatus.at(target->getTargetInd()).numStatusUncond;
            }
            else
            {
                MEAS_TIME(
                    fillMemoryConditionedFeatures(target->getTargetInd(), indInBatch),
                    m_params.m_debugVerboseLevel > 2, "[Segmenter::memAttenInfer] fillMemoryConditionedFeatures"
                );
                /// Construct dynamic size {numObjPtrs, numMaskMem, numFrames}.
                numObjPtrs = m_inferStatus.at(target->getTargetInd()).objPtrUncond.size() + \
                    m_inferStatus.at(target->getTargetInd()).objPtrCond.size();
                numMaskMem = m_inferStatus.at(target->getTargetInd()).statusUncond.size();
            }

            numObjPtrsMax = std::max(numObjPtrsMax, numObjPtrs);
            numMaskMemMax = std::max(numMaskMemMax, numMaskMem);

            /// Print the runtime tensor shapes for debugging.
            if (m_params.m_debugVerboseLevel > 3)
            {
                std::cout << "[Segmenter::memAttenInfer] Set runtime tensor shapes: (" <<
                numObjPtrs * 256 << "," << numMaskMem*64*64*64 << ", " <<
                (numMaskMem*4096 + 4*std::min(numObjPtrs, m_params.m_numObjPtr+1))*64 << "), " <<
                "numObjPtrs, numMaskMem, target->getAge()-1, m_params.m_numObjPtr+1 = " <<
                numObjPtrs << ", " << numMaskMem << ", " << int(target->getAge())-1 << ", " << m_params.m_numObjPtr+1 << std::endl;
            }
        }

        if (numObjPtrsMax > 0 && numMaskMemMax > 0)
        {
            std::vector<int> sizes = {int(targetList.size()), m_params.m_numObjPtr + 1,
                m_params.m_numMaskMem, m_params.m_numObjPtr + 1};
            /// Run memory attention once for the whole batch.
            m_memoryAttention.submitInput(m_bindingsMemAtten, sizes, m_cuStream);
            if (m_params.m_debugVerboseLevel > 3)
            {
                std::cout << "Memory attention submit input done" << std::endl;
            }
        }
    }

    for (int indInBatch = 0; indInBatch < int(targetList.size()); indInBatch++)
    {
        std::shared_ptr<NvTrackedObject> target = targetList.at(indInBatch);
        /// If memory bank is not used or memory bank is empty, copy featuresEncoder_d (at stream ind location) to pix_feat_with_mem_d (at indInBatch location).
        if ( (!m_params.m_useMemBank) || isInferStatusEmpty(target->getTargetInd()))
        {
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_pix_feat_with_mem_d + indInBatch * m_paramsRt.pixFeatWithMemSize,
                    m_featuresEncoder_d + streamInd * m_paramsRt.featuresEncoderSize,
                    m_paramsRt.featuresEncoderSize * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
                )
            );
        }
    }
}

/**
 * @brief Run memory-encoder inference for the current batch.
 * @param[in] batchSizeActual Number of targets in the current batch.
 */
void Segmenter::memEncoderInfer(const int batchSizeActual)
{
    if (m_params.m_useMemBank)
    {
        /// Memory Encoder.
        m_memoryEncoder.submitInput(m_bindingsMemEncoder, {batchSizeActual}, m_cuStream);
        if (m_params.m_debugVerboseLevel > 3)
        {
            std::cout << "Memory encoder submit input done" << std::endl;
        }

        CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
        CUDA_CHECK( cudaPeekAtLastError() );
    }
}

/**
 * @brief Copy a CUDA buffer into a tensor.
 * @param[in] src The source tensor.
 * @param[out] dst Destination tensor to populate.
 * @param[in] numElements Number of tensor elements to copy.
 * @param[in] cuStream CUDA stream that runs the GPU work.
 */
void Segmenter::copyFromCuda(float* src, TensorPtr& dst, int numElements, cudaStream_t& cuStream)
{
    dst.reset();
    dst = std::make_shared<std::vector<float>>(std::vector<float>(numElements));
    CUDA_CHECK(cudaMemcpyAsync( dst->data(), src, numElements *  sizeof(float), cudaMemcpyDeviceToHost, cuStream));
}

/**
 * @brief Convert decoder output into per-target mask state.
 * @param[in] indInBatch Index of this target in the inference batch.
 * @param[in] target Tracked target to process.
 * @param[in] frameNum Frame number associated with the inference result.
 */
void Segmenter::postProcessing(
    const int indInBatch, const std::shared_ptr<NvTrackedObject> target, const NvMOTFrameNum frameNum
)
{
    int outputLabelIndMax = 0;

    /// Copy output results.
    float *selectedMaskDecSize_d = m_targetMaskDecSize_d + (indInBatch*m_params.m_outputLabelNum + outputLabelIndMax) * m_params.m_hiddenDim * m_params.m_hiddenDim;

    /// Save mask from GPU to CPU memory m_batchMaskDecSize_d to m_targetMaskDecSize_h.
    float *ptrInBatchMaskDecSize_d = m_batchMaskDecSize_d + target->getTargetInd() * m_params.m_hiddenDim * m_params.m_hiddenDim;
    CUDA_CHECK(
        cudaMemcpyAsync(
            ptrInBatchMaskDecSize_d, selectedMaskDecSize_d,
            m_params.m_hiddenDim * m_params.m_hiddenDim *  sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_targetMaskDecSize_h, ptrInBatchMaskDecSize_d,
            m_params.m_outputLabelNum * m_params.m_hiddenDim * m_params.m_hiddenDim *  sizeof(float),
            cudaMemcpyDeviceToHost, m_cuStream
        )
    );
    CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
    CUDA_CHECK( cudaPeekAtLastError() );

    /// Calculate bounding box from mask.
    if (m_params.m_useMemBank)
    {
        NvMotDataTypes::Rect2f bboxProg = CV_FUNCS::boundingRect(
            m_targetMaskDecSize_h, m_params.m_hiddenDim, m_params.m_hiddenDim
        );
        bboxProg = CV_FUNCS::constructRect2f(
            bboxProg.min()(0) / m_params.m_hiddenDim * m_srcImageSize.width,
            bboxProg.min()(1) / m_params.m_hiddenDim * m_srcImageSize.height,
            bboxProg.sizes()(0) / m_params.m_hiddenDim * m_srcImageSize.width,
            bboxProg.sizes()(1) / m_params.m_hiddenDim * m_srcImageSize.height
        );

        float confidence = m_outputIou_h.at(indInBatch*m_params.m_outputLabelNum + outputLabelIndMax);
        float occLogit = m_outputOccLogit_h.at(indInBatch*m_params.m_outputLabelNum + outputLabelIndMax);
        m_targetScores[target->getTargetInd()] = {confidence, occLogit};
        if (m_params.m_debugVerboseLevel > 3)
        {
            std::cout << "Target:" << target->getTargetInd()
                << ", SAM2 confidence level: " << confidence << ", occLogit: " << occLogit << ", bboxProg: " <<
                bboxProg.min()(0) << "," << bboxProg.min()(1) << "," << bboxProg.sizes()(0) << "," <<bboxProg.sizes()(1) << std::endl;
        }
        if ( m_isForLocalization && isGoodBboxProg(target, confidence, bboxProg, 0.8) )
        {
            target->setLocalizationResults(frameNum, confidence, bboxProg);
        }

        /// Copy inference result to per target mem bank.
        bool isGoodPrompt = (!m_isForLocalization) && (target->getMatchedDetObj()) && \
            (target->getMatchedDetObj()->confidence > m_params.m_minPromptUpdateConfidence);
        if ( ( (isGoodBboxProg(target, confidence, bboxProg, 0.4) || isGoodPrompt)  &&
            (target->getAge() % (m_params.m_maskMemoryUpdateInterval+1) == 0)) ||
            isInferStatusEmpty(target->getTargetInd())
            )
        {
            updateInferStatus(indInBatch, target, frameNum);
            if (m_params.m_debugVerboseLevel > 3)
            {
                std::cout << "Frame " << frameNum << " Update memory bank for target " << target->getTargetInd() << std::endl;
            }
        }
        else
        {
            if (m_params.m_debugVerboseLevel > 3)
            {
                std::cout << "Frame " << frameNum << " No update memory bank for target " << target->getTargetInd() << std::endl;
                std::cout << "is infer status empty: " << isInferStatusEmpty(target->getTargetInd()) << std::endl;
            }
        }
    }
}

/**
 * @brief Check whether a propagated bounding box is reliable.
 * @param[in] target Tracked target to process.
 * @param[in] confidence Target confidence used to validate propagation.
 * @param[in] bboxProg Propagated target bounding box.
 * @param[in] iouThresh Minimum intersection-over-union threshold.
 */
bool Segmenter::isGoodBboxProg(
    const std::shared_ptr<NvTrackedObject> target, const float confidence,
    const NvMotDataTypes::Rect2f bboxProg, const float iouThresh
)
{
    return (
        confidence > m_params.m_minMemoryUpdateConfidence && bboxProg.sizes()(0) > 20 && bboxProg.sizes()(1) > 20 &&
        CV_FUNCS::computeIOU(target->getTargetInfo().tBbox, bboxProg) > m_params.m_minIouPrevFrame4ProgBbox
    );
}

/**
 * @brief Record a target in the pending inference batch.
 * @param[in] indInBatch Index of this target in the inference batch.
 * @param[in] target Tracked target to process.
 * @param[in] frameNum Frame number associated with the inference result.
 */
void Segmenter::updateInferStatus(
    const int indInBatch, const std::shared_ptr<NvTrackedObject> target, const NvMOTFrameNum frameNum
)
{
    if (m_params.m_useCudaMemBank)
    {
        /// Initialize cuda inference status if not found.
        if (m_cudaInferenceStatus.find(target->getTargetInd()) == m_cudaInferenceStatus.end())
        {
            m_cudaInferenceStatus[target->getTargetInd()] = {target->getAge(), 0, 0};
            if (m_params.m_debugVerboseLevel > 4)
            {
                std::cout << "Initialize cuda inference status for target " << target->getTargetInd() << std::endl;
            }
        }
        /// Update cuda inference status.
        CudaInferenceStatus& targetStatus = m_cudaInferenceStatus[target->getTargetInd()];
        targetStatus.currentFrameNum = target->getAge();
        m_cudaMemBank.update(
            m_maskmem_features_d, m_maskmem_pos_enc_d, m_temporal_code_d, m_obj_ptr_d, target, indInBatch, targetStatus
        );
    }
    else
    {
        /// Update memory bank.
        MemoryEncoderStatus curStatus;
        copyFromCuda(
            m_maskmem_features_d + indInBatch * m_paramsRt.maskMemFeatSize,
            curStatus.maskmem_features, m_paramsRt.maskMemFeatSize, m_cuStream
        );
        copyFromCuda(
            m_maskmem_pos_enc_d + indInBatch * m_paramsRt.maskMemFeatSize,
            curStatus.maskmem_pos_enc, m_paramsRt.maskMemFeatSize, m_cuStream
        );
        copyFromCuda(
            m_temporal_code_d, curStatus.temporal_code, m_params.m_numMaskMem * m_paramsRt.temporalCodeSize, m_cuStream
        );

        InferenceStatus& targetStatus = m_inferStatus[target->getTargetInd()];
        targetStatus.currentFrameNum = target->getAge();

        if (isInferStatusEmpty(target->getTargetInd()) || int(targetStatus.objPtrCond.size()) < m_params.m_numConditionalMem)
        {
            /// Initialize conditional obj_ptr and status (with prompt).
            TensorPtr objPtrCond;
            copyFromCuda(
                m_obj_ptr_d + indInBatch * m_paramsRt.objPtrSize, objPtrCond, m_paramsRt.objPtrSize, m_cuStream
            );
            targetStatus.objPtrCond.push_back(std::move(objPtrCond));
            targetStatus.statusCond.push_back(std::move(curStatus));
        }
        else
        {
            /// Update unconditional obj_ptrs and status (without prompt).
            while (int(targetStatus.objPtrUncond.size()) >= m_params.m_numObjPtr)
            {
                targetStatus.objPtrUncond.erase(targetStatus.objPtrUncond.begin());
            }
            TensorPtr objPtrUncond;
            copyFromCuda(
                m_obj_ptr_d + indInBatch * m_paramsRt.objPtrSize, objPtrUncond, m_paramsRt.objPtrSize, m_cuStream
            );
            targetStatus.objPtrUncond.push_back(std::move(objPtrUncond));

            while (int(targetStatus.statusUncond.size()) >= m_params.m_numMaskMem - 1)
            {
                targetStatus.statusUncond.erase(targetStatus.statusUncond.begin());
            }
            targetStatus.statusUncond.push_back(std::move(curStatus));
        }
    }
}

/**
 * @brief Run the segmentation pipeline for one target.
 * @param[in] targetList Targets included in the inference operation.
 * @param[in] usePrompt Whether decoder prompt data is enabled.
 * @param[in] outputLabelIndMax Highest decoder output-label index to process.
 * @param[in] frameNum Frame number associated with the inference result.
 * @param[in] streamInd Region index within the input batch.
 */
void Segmenter::singleInfer(
    const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList, bool usePrompt,
    int outputLabelIndMax, const NvMOTFrameNum frameNum, const uint streamInd
)
{
    /// Perform inference.
    for (auto ti = targetList.begin(); ti != targetList.end(); ti++)
    {
        const std::shared_ptr<NvTrackedObject> target = ti->second;

        /// During update, only perform inference on the newly added targets with detection as prompt.
        /// Other existing targets are already done in localize.
        if (skipInfer(target, frameNum))
        {
            continue;
        }

        /// Per target inference.
        MEAS_TIME(
            memAttenInfer(target, 0, streamInd), m_params.m_debugVerboseLevel > 2,
            "[Segmenter::doInference] memAttenInfer"
        );

        std::vector<std::shared_ptr<NvTrackedObject>> singleTargetList = {target};
        MEAS_TIME(
            maskDecoderInfer(usePrompt, singleTargetList, outputLabelIndMax, streamInd, frameNum),
            m_params.m_debugVerboseLevel > 2, "[Segmenter::doInference] maskDecoderInfer"
        );

        MEAS_TIME(memEncoderInfer(1), m_params.m_debugVerboseLevel > 2, "[Segmenter::doInference] memEncoderInfer");

        /// Post processing.
        MEAS_TIME(
            postProcessing(0, target, frameNum), m_params.m_debugVerboseLevel > 2,
            "[Segmenter::doInference] postProcessing"
        );

        #ifdef VIZ_DEBUG_SEG
            updatePerTargetMaskInFrame(target, ti == targetList.begin());
        #endif
    }
}

/**
 * @brief Determine whether segmentation can be skipped for a target.
 * @param[in] target Tracked target to process.
 * @param[in] frameNum Frame number associated with the inference result.
 */
bool Segmenter::skipInfer(std::shared_ptr<NvTrackedObject> target, const NvMOTFrameNum frameNum)
{
    if ( (m_isForLocalization) || (!m_params.m_useMemBank) )
    {
        return false;
    }
    else
    {
        /// Must perform bbox mask decoding on newly added targets.
        bool doInferNewTarget = isInferStatusEmpty(target->getTargetInd()) ||
            target->getTrackerState() == TRACKER_STATE::TENTATIVE;

        if (frameNum % (m_params.m_maskPromptUpdateInterval + 1) == 0)
        {
            if (doInferNewTarget)
            {
                return false;
            }
            else if (target->getMatchedDetObj() == nullptr || target->getMatchedDetObj()->bbox.isEmpty() )
            {
                return true;
            }

            /// Perform mask decoding with bbox prompt on detections not overlapped with existing targets.
            NvMOTStreamId streamId = target->getVideoStreamId();
            const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList = \
                m_targetManager->getTrackedObjectsInUseList(streamId);
            for(auto tIt = targetList.begin(); tIt != targetList.end(); ++tIt)
            {
                if (tIt->second->getTargetInd() == target->getTargetInd())
                {
                    continue;
                }

                const float iou = CV_FUNCS::computeIOU(
                    tIt->second->getTargetInfo().tBbox, target->getMatchedDetObj()->bbox
                );
                if(iou > m_params.m_minIouDiff4MemoryUpdate)
                {
                    return true;
                }
            }
            return false;
        }
        else
        {
            /// Skip targets that are not newly added.
            return (!doInferNewTarget);
        }
    }
}

/**
 * @brief Run the segmentation pipeline for a target batch.
 * @param[in] targetMap Targets included in the inference operation.
 * @param[in] usePrompt Whether decoder prompt data is enabled.
 * @param[in] outputLabelIndMax Highest decoder output-label index to process.
 * @param[in] frameNum Frame number associated with the inference result.
 * @param[in] streamInd Region index within the input batch.
 */
void Segmenter::batchInfer(
    const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetMap, bool usePrompt,
    int outputLabelIndMax, const NvMOTFrameNum frameNum, const uint streamInd
)
{
    std::vector<std::shared_ptr<NvTrackedObject>> targetList;
    for (const auto& pair : targetMap)
    {
        const std::shared_ptr<NvTrackedObject>& target = pair.second;
        if (skipInfer(target, frameNum))
        {
            continue;
        }

        targetList.insert(targetList.begin(), target);
    }

    for (int indInList = 0; indInList < int(targetList.size()); indInList += m_paramsRt.batchSize)
    {
        int indInBatchNext = std::min(int(indInList + m_paramsRt.batchSize), int(targetList.size()));
        std::vector<std::shared_ptr<NvTrackedObject>> targetBatch = \
            std::vector<std::shared_ptr<NvTrackedObject>>(
                targetList.begin() + indInList, targetList.begin() + indInBatchNext
            );

        /// Per target inference.
        MEAS_TIME(
            memAttenInferBatchParser(targetBatch, streamInd),
            m_params.m_debugVerboseLevel > 2, "[Segmenter::doInference] memAttenInfer"
        );

        MEAS_TIME(
            maskDecoderInfer(usePrompt, targetBatch, outputLabelIndMax, streamInd, frameNum),
            m_params.m_debugVerboseLevel > 2, "[Segmenter::doInference] maskDecoderInfer"
        );

        MEAS_TIME(
            memEncoderInfer(targetBatch.size()), m_params.m_debugVerboseLevel > 2,
            "[Segmenter::doInference] memEncoderInfer"
        );

        /// Post processing.
        for (int indInBatch = 0; indInBatch < indInBatchNext - indInList; indInBatch++)
        {
            MEAS_TIME(
                postProcessing(indInBatch, targetList.at(indInList + indInBatch), frameNum),
                m_params.m_debugVerboseLevel > 2, "[Segmenter::doInference] postProcessing"
            );
            #ifdef VIZ_DEBUG_SEG
                updatePerTargetMaskInFrame(targetList.at(indInList + indInBatch), indInList==0 && indInBatch==0);
            #endif
        }
    }
}

/**
 * @brief Run inference for the targets in the input batch.
 * @param[in] inputMeta A batch of input frames.
 */
void Segmenter::doInference(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta)
{
    for (auto streamIt = m_inputRoisInBatch.begin(); streamIt != m_inputRoisInBatch.end(); ++streamIt)
    {
        /// Define per frame info.
        NvMOTStreamId videoStreamId = streamIt->first;
        uint inputRoiInd = streamIt->second.at(0).first;

        /// For SAM2, currently only the first frame uses bbox as prompt.
        /// TODO: Choose when to use a bounding-box prompt.
        bool usePrompt = m_params.m_useMemBank ? false : true;
        int outputLabelIndMax = 0;
        const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList = \
            m_targetManager->getTrackedObjectsInUseList(videoStreamId);
        if (targetList.empty())
        {
            continue;
        }

        updateTrtBindingsPerStream(inputRoiInd);

        /// Perform inference.
        if (m_paramsRt.batchSize == 1)
        {
            singleInfer(targetList, usePrompt, outputLabelIndMax, inputMeta.at(videoStreamId)->frameNum, inputRoiInd);
        }
        else
        {
            batchInfer(targetList, usePrompt, outputLabelIndMax, inputMeta.at(videoStreamId)->frameNum, inputRoiInd);
        }

        #ifdef VIZ_DEBUG_SEG
            savePerFrameMask(videoStreamId, inputMeta.at(videoStreamId)->frameNum, targetList);
        #endif
    }
}

#ifdef VIZ_DEBUG_SEG
/**
 * @brief Update visualization masks for one target.
 * @param[in] target Tracked target to process.
 * @param[in] initMask Whether the target mask is being initialized.
 */
void Segmenter::updatePerTargetMaskInFrame(const std::shared_ptr<NvTrackedObject> target, bool initMask)
{
    if ( (!m_params.m_saveFrameViz) && (!m_params.m_saveIDMask))
    {
        return;
    }

    if (initMask)
    {
        m_imageSumMask = cv::Mat(m_srcImageSize.height, m_srcImageSize.width, CV_32FC1, 0.0);
        m_imageIDMask = cv::Mat(m_srcImageSize.height, m_srcImageSize.width, CV_32SC1, -1);
    }

    cv::Mat targetMask(
        m_params.m_hiddenDim, m_params.m_hiddenDim, CV_32FC1,
        reinterpret_cast<float*>(m_targetMaskDecSize_h )
    );
    cv::resize(targetMask, targetMask, cv::Size(m_srcImageSize.width, m_srcImageSize.height));
    targetMask = (targetMask>=0) / 255;

    if (m_params.m_saveFrameViz || m_params.m_saveIDMask)
    {
        cv::Mat targetIDMask(
            m_srcImageSize.height, m_srcImageSize.width, CV_32SC1, int(target->getTargetID())
        );

        {
            targetIDMask.copyTo(m_imageIDMask, targetMask > 0);
        }
        m_imageSumMask += targetMask;
    }

    /// Calculate bounding box from mask.
    if (1)
    {
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Point> contoursAll;
        cv::findContours( targetMask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
        cv::Rect bboxLargestMask;
        for (uint ci = 0; ci < contours.size(); ci++)
        {
            cv::Rect bboxMask = cv::boundingRect(contours.at(ci));

            if (ci == 0)
            {
                bboxLargestMask = bboxMask;
            }
            else
            {
                auto x1 = std::max(bboxLargestMask.width + bboxLargestMask.x, bboxMask.width + bboxMask.x);
                auto y1 = std::max(bboxLargestMask.height + bboxLargestMask.y, bboxMask.height + bboxMask.y);
                bboxLargestMask.x = std::min(bboxLargestMask.x, bboxMask.x);
                bboxLargestMask.y = std::min(bboxLargestMask.y, bboxMask.y);
                bboxLargestMask.width = x1 - bboxLargestMask.x;
                bboxLargestMask.height = y1 - bboxLargestMask.y;
            }
        }

        if (bboxLargestMask.area() >= 0)
            target->setProgBbox(CV_FUNCS::convertCvRect2f2MotRect2f(bboxLargestMask));
    }
}

/**
 * @brief Write accumulated frame masks and overlays.
 * @param[in] videoStreamId Video stream identifier.
 * @param[in] frameNum Frame number associated with the inference result.
 * @param[in] targetList Targets included in the inference operation.
 */
void Segmenter::savePerFrameMask(
    NvMOTStreamId videoStreamId, NvMOTFrameNum frameNum,
    const std::map<uint32_t,std::shared_ptr<NvTrackedObject>> &targetList
)
{
    if (m_imageSumMask.empty() || m_imageIDMask.empty())
    {
        m_imageSumMask = cv::Mat(m_srcImageSize.height, m_srcImageSize.width, CV_32FC1, 0.0);
        m_imageIDMask = cv::Mat(m_srcImageSize.height, m_srcImageSize.width, CV_32SC1, -1);
    }

    if ( (!m_params.m_saveFrameViz) && (!m_params.m_saveIDMask))
    {
        return;
    }

    std::map<NvMOTStreamId, std::map<uint, cv::Mat>> m_vizSrcImages;
    if (m_params.m_saveFrameViz || m_params.m_saveIDMask)
    {
        if(m_params.m_useVPICropScaler)
        {
            m_vizSrcImages = m_subImageCropScaler->getImagePatches2Host_OCV_batch(
                m_params.m_netScaleFactor, m_params.m_offsets
            );
        }
        else
        {
            m_vizSrcImages = m_subImageCropper2->getImagePatches2Host_OCV_batch(
                m_params.m_netScaleFactor, m_params.m_offsets
            );
        }
    }

    cv::Mat srcImage, overlayImage;
    cv::resize(
        m_vizSrcImages.at(videoStreamId).at(0), srcImage, cv::Size(m_srcImageSize.width, m_srcImageSize.height)
    );

    cv::Mat colorMask(m_srcImageSize.height, m_srcImageSize.width, CV_8UC3, cv::Scalar(0, 255, 0));

    srcImage.copyTo(colorMask, m_imageSumMask <= 0);
    float alpha = 0.3;
    cv::addWeighted(colorMask, alpha, srcImage, 1.0 - alpha, 0.0f, overlayImage);

    for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
    {
        const std::shared_ptr<NvTrackedObject> target = targetIt->second;

        auto rect = CV_FUNCS::convertMotRect2f2CvRect2f(target->getProgBbox());

        cv::rectangle(overlayImage, rect, cv::Scalar(255, 0, 0), 2);
        cv::putText(
            overlayImage, std::to_string(int(m_targetScores[target->getTargetInd()][0]*10) ),
            cv::Point(rect.x+2, rect.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 2
        );
        cv::putText(
            overlayImage, std::to_string(int(m_targetScores[target->getTargetInd()][1]*10) ),
            cv::Point(rect.x+2, rect.y+50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2
        );

        if (!target->getTargetInfo().tBbox.isEmpty())
        {
            rect = CV_FUNCS::convertMotRect2f2CvRect2f(target->getTargetInfo().tBbox);

            cv::rectangle(overlayImage, rect, cv::Scalar(0, 0, 255), 1);
        }
    }

    char baseNameStr[1024] = { 0 };
    snprintf (
        baseNameStr, sizeof (baseNameStr) - 1, "%s/%03d_%06d_%01d",
        m_params.m_vizDir.c_str(), int(videoStreamId), frameNum, int(!m_isForLocalization)
    );
    std::string baseName = std::string(baseNameStr);
    if (m_params.m_saveFrameViz)
    {
        cv::imwrite(baseName + "_src.jpg", srcImage);
        cv::imwrite(baseName + "_mask.jpg", m_imageSumMask>0);
        cv::imwrite(baseName + "_overlay.jpg", overlayImage);
    }
    if (m_params.m_saveIDMask)
    {
        cv::Mat imageIDToSave(m_srcImageSize.height, m_srcImageSize.width, CV_8UC4, m_imageIDMask.data);
        cv::imwrite(baseName + ".png", imageIDToSave);
    }
}
#endif

/**
 * @brief Run segmentation to localize targets in the current frames.
 * @param[in] inputMeta A batch of input frames.
 */
void Segmenter::localize(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta)
{
    m_isForLocalization = true;
    /// For segmenters without memory bank, only do update.
    if ( (!m_params.m_useMemBank) || getSegmenterType() == SEGMENTER_TYPES::EfficientVitSAM ||
        getSegmenterType() == SEGMENTER_TYPES::NanoSAM )
    {
        return;
    }
    /// Encode the input frame and run SAM2 inference for localization.
    /// Transform input frame to RGB format and resize it to encoder input size.
    extractImagePatchesBatch(inputMeta);
    imgEncoderInfer();
    doInference(inputMeta);
}

/** @brief Discard completed or terminated inference state. */
void Segmenter::removeInferStatus()
{
    /// Remove terminated memory bank.
    const std::map<NvMOTStreamId, std::vector<std::shared_ptr<NvTrackedObject>>> & targetsTerminated = \
        m_targetManager->getTrackedObjectsTerminated();
    if (m_params.m_useMemBank)
    {
        for (auto it = targetsTerminated.begin(); it != targetsTerminated.end(); ++it)
        {
            for (uint ti = 0; ti < it->second.size(); ti++)
            {
                TrackerIndex targetInd = it->second.at(ti)->getTargetInd();
                if (m_cudaInferenceStatus.find(targetInd) != m_cudaInferenceStatus.end())
                {
                    m_cudaInferenceStatus.erase(targetInd);
                }
                m_cudaMemBank.reset(targetInd);
            }
        }
    }
    else
    {
        for (auto it = targetsTerminated.begin(); it != targetsTerminated.end(); ++it)
        {
            for (uint ti = 0; ti < it->second.size(); ti++)
            {
                TrackerIndex targetInd = it->second.at(ti)->getTargetInd();
                if (m_inferStatus.find(targetInd) != m_inferStatus.end())
                {
                    m_inferStatus.erase(targetInd);
                }
            }
        }
    }
}

/**
 * @brief Update retained state after data association.
 * @param[in] inputMeta A batch of input frames.
 */
void Segmenter::update(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta)
{
    m_isForLocalization = false;

    /// For segmenters without memory bank, only do update.
    if ( (!m_params.m_useMemBank) || getSegmenterType() == SEGMENTER_TYPES::EfficientVitSAM ||
        getSegmenterType() == SEGMENTER_TYPES::NanoSAM )
    {
        extractImagePatchesBatch(inputMeta);
        imgEncoderInfer();
        doInference(inputMeta);
        return;
    }

    /// Remove terminated memory bank.
    removeInferStatus();

    std::map<NvMOTStreamId, NvMOTFrame *> inferenceMeta;
    /// For SAM2, if there is detection input, generate segmentation mask in update.
    /// If there is no detection input, nothing to do here because segmentation is already done in localization.
    for (auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
    {
        /// Define per frame info.
        NvMOTStreamId videoStreamId = streamIt->first;
        if ( inputMeta.at(videoStreamId)->objectsIn.detectionDone )
        {
            inferenceMeta[videoStreamId] = inputMeta.at(videoStreamId);
        }
    }
    if (!inferenceMeta.empty())
    {
        doInference(inferenceMeta);
    }
}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void Segmenter::removeStream(const NvMOTStreamId videoStreamId)
{
}

/**
 * @brief Populate the output mask for a tracked target.
 * @param[in] target The target.
 * @param[out] maskOriginal Output mask in original-frame coordinates.
 */
void Segmenter::outputMaskObj(const std::shared_ptr<NvTrackedObject> target, NvMOTMask& maskOriginal)
{
    if (!m_params.m_outputMask)
    {
        return;
    }

    /// Crop and rescale to original frame size (streammux).
    const TargetInfo tInfo = target->getTargetInfo();
    NvMotDataTypes::Size bboxOriginalSize = {
        std::min(int(tInfo.tBbox.sizes()(0) * m_scaleInput2Original(0) + 0.5), m_originalSrcImageSize.width),
        std::min(int(tInfo.tBbox.sizes()(1) * m_scaleInput2Original(1) + 0.5), m_originalSrcImageSize.height)
    };
    if (bboxOriginalSize.width <= 0 || bboxOriginalSize.height <= 0)
    {
        return;
    }

    uint bboxOriginalElements = bboxOriginalSize.width * bboxOriginalSize.height;

    float4 bboxDecOutput;
    bboxDecOutput.x = tInfo.tBbox.min()(0) / m_srcImageSize.width * m_params.m_hiddenDim,
    bboxDecOutput.y = tInfo.tBbox.min()(1) / m_srcImageSize.height * m_params.m_hiddenDim,
    bboxDecOutput.w = tInfo.tBbox.sizes()(0) / m_srcImageSize.width * m_params.m_hiddenDim,
    bboxDecOutput.z = tInfo.tBbox.sizes()(1) / m_srcImageSize.height * m_params.m_hiddenDim;

    float *ptrInBatchMaskDecSize_d = m_batchMaskDecSize_d + target->getTargetInd() * m_params.m_hiddenDim * m_params.m_hiddenDim;

    cuSegmenter::rescaleSegmentationOutput(
        m_cuStream, ptrInBatchMaskDecSize_d, m_params.m_hiddenDim, m_params.m_hiddenDim * m_params.m_hiddenDim,
        bboxOriginalSize.width, bboxOriginalElements, bboxDecOutput, m_targetMaskOriginalSrcSize_d
    );

    /// Check if mask is allocated with enough memory.
    if (maskOriginal.data == nullptr)
    {
        CLOG(
            std::string( "!!![WARNING] Invalid pointer encountered when outputting segmentation mask. Reallocating array memory..." ).c_str()
        );
        maskOriginal.data = new float[m_originalSrcImageSize.width * m_originalSrcImageSize.height];
        maskOriginal.allocatedSize = uint32_t(m_originalSrcImageSize.width * m_originalSrcImageSize.height);
    }
    else if (m_originalSrcImageSize.width * m_originalSrcImageSize.height > int(maskOriginal.allocatedSize))
    {
        delete[] maskOriginal.data;
        maskOriginal.data = new float[m_originalSrcImageSize.width * m_originalSrcImageSize.height];
        maskOriginal.allocatedSize = uint32_t(m_originalSrcImageSize.width * m_originalSrcImageSize.height);
    }

    /// Fill mask output.
    maskOriginal.width = bboxOriginalSize.width;
    maskOriginal.height = bboxOriginalSize.height;

    CUDA_CHECK(
        cudaMemcpyAsync(
            maskOriginal.data, m_targetMaskOriginalSrcSize_d,
            bboxOriginalElements * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
        )
    );
    CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
    CUDA_CHECK( cudaPeekAtLastError() );
}

/**
 * @brief Assemble host-managed SAM2 memory features.
 * @param[in] targetInd Index of the target.
 * @param[in] indInBatch Index of this target in the inference batch.
 */
void Segmenter::fillMemoryConditionedFeatures(TrackerIndex targetInd, int indInBatch)
{
    /// Acquire current target's memory.
    if (m_inferStatus.find(targetInd) == m_inferStatus.end())
    {
        throw std::runtime_error("Unable to find target in segmentation inference status");
    }
    const InferenceStatus& targetStatus = m_inferStatus.at(targetInd);

    /// memory_obj_ptr consists of obj_ptr_cond and obj_ptr_uncond obj_ptr_cond.
    uint numObjPtrCond = targetStatus.objPtrCond.size();
    for (uint oi = 0; oi < numObjPtrCond; oi++)
    {
        const float* objPtrCond = targetStatus.objPtrCond.at(oi)->data();
        if (m_paramsRt.batchSize == 1)
        {
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_obj_ptr_d + m_paramsRt.objPtrSize * oi, objPtrCond,
                    m_paramsRt.objPtrSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
        else
        {
            int indInBatchOffset = indInBatch * m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1);
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_obj_ptr_d + indInBatchOffset + m_paramsRt.objPtrSize * oi, objPtrCond,
                    m_paramsRt.objPtrSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
    }
    /// obj_ptr_uncond contains at most 15 object pointers.
    /// TODO: Match the reference order, newest frame first.
    /// Select at most the last 15 unconditional object pointers at stride 1.
    /// Although mask allows stride > 1.
    /// Therefore, for now the obj_pos code should always be [frame_diff_with_first_cond_frame, 1,2,...,15].
    for (uint oi = 0; oi < targetStatus.objPtrUncond.size(); oi++)
    {
        /// Read unconditional object pointers from newest to oldest.
        const float* objPtrUncond = targetStatus.objPtrUncond.at(
            int (targetStatus.objPtrUncond.size()) - oi - 1
        )->data();
        if (m_paramsRt.batchSize == 1)
        {
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_obj_ptr_d + m_paramsRt.objPtrSize * (numObjPtrCond + oi), objPtrUncond,
                    m_paramsRt.objPtrSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
        else
        {
            int indInBatchOffset = indInBatch * m_paramsRt.objPtrSize * (m_params.m_numObjPtr + numObjPtrCond);
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_obj_ptr_d + indInBatchOffset + m_paramsRt.objPtrSize * (numObjPtrCond + oi),
                    objPtrUncond, m_paramsRt.objPtrSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
    }

    /// Set m_cond_frame_id_diff_d, so mem atten takes in this value and generates sine embedded obj_pos.
    /// For example, obj_pos frame id is like [m_cond_frame_id_diff_d, 1, 2, 3,...,15] -> convert to sine embedding.
    /// SAM 2.1 new feature.
    float frame_diff_with_first_cond_frame = targetStatus.currentFrameNum + 1;
    if (m_params.m_debugVerboseLevel > 3)
    {
        std::cout << "frame_diff_with_first_cond_frame: " << frame_diff_with_first_cond_frame << std::endl;
    }
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_cond_frame_id_diff_d, &frame_diff_with_first_cond_frame, sizeof(float), cudaMemcpyHostToDevice, m_cuStream
        )
    );

    /// memory_maskmem_features contains at most seven unconditional features.
    /// Each feature size = 4096 x 64.
    uint numMaskMemCond = targetStatus.statusCond.size();
    uint numMaskMemUncond = targetStatus.statusUncond.size();
    for (uint oi = 0; oi < numMaskMemUncond + numMaskMemCond; oi++)
    {
        float* maskMemFeat = nullptr;
        if (oi < numMaskMemCond)
        {
            maskMemFeat = targetStatus.statusCond.at(oi).maskmem_features->data();
        }
        else
        {
            maskMemFeat = targetStatus.statusUncond.at(oi-numMaskMemCond).maskmem_features->data();
        }
        if (m_paramsRt.batchSize == 1)
        {
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_maskmem_features_d + m_paramsRt.maskMemFeatSize * oi, maskMemFeat,
                    m_paramsRt.maskMemFeatSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
        else
        {
            int indInBatchOffset = indInBatch * m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem;
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_maskmem_features_d + indInBatchOffset + m_paramsRt.maskMemFeatSize * oi,
                    maskMemFeat, m_paramsRt.maskMemFeatSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
    }

    /// memory_pos_embed temporal_code_: size [7 x 64].
    const float* temporal_code_ = targetStatus.statusCond.front().temporal_code->data();

    std::vector<const float*> temporal_code;
    for(int oi = m_params.m_numMaskMem-1; oi>=0; oi--){
        const float* temp = temporal_code_ + oi * m_paramsRt.temporalCodeSize;
        temporal_code.push_back(temp);
    }

    /// Clear the memory position embedding before filling it.
    if (m_paramsRt.batchSize == 1)
    {
        CUDA_CHECK(
            cudaMemsetAsync( m_memory_pos_embed_d, 0.f, m_paramsRt.maxConcatMemFeatSize * sizeof(float), m_cuStream)
        );
    }
    else
    {
        int indInBatchOffset = indInBatch * m_paramsRt.maxConcatMemFeatSize;
        CUDA_CHECK(
            cudaMemsetAsync(
                m_memory_pos_embed_d + indInBatchOffset, 0.f,
                m_paramsRt.maxConcatMemFeatSize * sizeof(float), m_cuStream
            )
        );
    }

    /// Fill m_maskmem_pos_enc_d with: (maskmem_pos_enc + temporal_code).
    /// [4096 x 64] + [64] ->[4096 x 64] + [4096 x 64] ->[4096 x 64].
    for(uint oi = 0; oi < numMaskMemCond + numMaskMemUncond; oi++){
        float* maskmem_pos_enc = nullptr;
        if (oi < numMaskMemCond)
        {
            maskmem_pos_enc = targetStatus.statusCond.at(oi).maskmem_pos_enc->data();
        }
        else
        {
            maskmem_pos_enc = targetStatus.statusUncond.at(oi-numMaskMemCond).maskmem_pos_enc->data();
        }

        std::vector<float> dst_memory_pos_embed_h(m_paramsRt.maskMemFeatSize, 0.0);
        /// Broadcast add [maskmem_pos_enc: 4096 x 64] + [temporal_code: 64] ->[4096 x 64] + [4096 x 64] ->[4096 x 64].
        broadcastAddCpu(dst_memory_pos_embed_h.data(), maskmem_pos_enc, temporal_code.at(oi));

        if (m_paramsRt.batchSize == 1)
        {
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_pos_embed_d + m_paramsRt.maskMemFeatSize * oi, dst_memory_pos_embed_h.data(),
                    m_paramsRt.maskMemFeatSize * sizeof(float), cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
        else
        {
            int indInBatchOffset = indInBatch * m_paramsRt.maxConcatMemFeatSize;
            CUDA_CHECK(
                cudaMemcpyAsync(
                    m_memory_pos_embed_d + indInBatchOffset + m_paramsRt.maskMemFeatSize * oi,
                    dst_memory_pos_embed_h.data(), m_paramsRt.maskMemFeatSize * sizeof(float),
                    cudaMemcpyHostToDevice, m_cuStream
                )
            );
        }
    }
}

/**
 * @brief Assemble CUDA-managed SAM2 memory features.
 * @param[in] targetInd Index of the target.
 * @param[in] indInBatch Index of this target in the inference batch.
 */
void Segmenter::fillCudaMemoryConditionedFeatures(TrackerIndex targetInd, int indInBatch)
{
    /// MemFeatures include {targetsObjPtr_d, targetsMaskMemFeatures_d, m_targetsMaskMemPosEnc_d, m_targetsTemporalCode_d}.
    std::vector<float*> memFeatures = m_cudaMemBank.getCudaMemBank(targetInd);
    if (m_paramsRt.batchSize == 1)
    {
        m_bindingsMemAtten[2] = memFeatures.at(0);
        m_bindingsMemAtten[3] = memFeatures.at(1);
        float* targetLastTemporalCode = memFeatures.at(3) + \
            m_cudaInferenceStatus.at(targetInd).numStatusUncond * m_paramsRt.temporalCodeSize;
        cuSegmenter::addTemporalCode2PosEmbed(
            m_cuStream, targetLastTemporalCode, memFeatures.at(2), m_memory_pos_embed_d,
            m_params.m_numMaskMem, m_paramsRt.maskMemFeatSize, m_paramsRt.temporalCodeSize
        );
    }
    else
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_memory_obj_ptr_d + indInBatch * m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1), memFeatures.at(0),
                m_paramsRt.objPtrSize * (m_params.m_numObjPtr + 1) * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_memory_maskmem_features_d + indInBatch * m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem,
                memFeatures.at(1), m_paramsRt.maskMemFeatSize * m_params.m_numMaskMem * sizeof(float),
                cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        float* targetLastTemporalCode = memFeatures.at(3) + \
            m_cudaInferenceStatus.at(targetInd).numStatusUncond * m_paramsRt.temporalCodeSize;
        cuSegmenter::addTemporalCode2PosEmbed(
            m_cuStream, targetLastTemporalCode, memFeatures.at(2),
            m_memory_pos_embed_d + indInBatch * m_paramsRt.maxConcatMemFeatSize,
            m_params.m_numMaskMem, m_paramsRt.maskMemFeatSize, m_paramsRt.temporalCodeSize
        );
    }

    float frame_diff_with_first_cond_frame = m_cudaInferenceStatus.at(targetInd).currentFrameNum + 1;
    CUDA_CHECK(
        cudaMemcpyAsync(
            m_cond_frame_id_diff_d, &frame_diff_with_first_cond_frame, sizeof(float), cudaMemcpyHostToDevice, m_cuStream
        )
    );
}

/** @brief Construct a SegmenterConfigParams instance. */
SegmenterConfigParams::SegmenterConfigParams()
{
    /// Sets the default segmenter input size, color normalization, and memory-bank capacity.
    m_outputLabelNum = 1;
    m_hiddenDim = 256;
    m_encoderFeatureDim = 64;
    m_inputOrder = 0;
    m_width = 1024;
    m_height = 1024;
    m_inferDims = {3, 1024, 1024};
    m_channel = 3;
    m_gpuId = 0;
    m_netScaleFactor = 0.01742919389;
    /// Rgb.
    m_offsets = {121.514738059, 116.799107208, 103.530000058};
    m_promptType = 1;
    m_useVPICropScaler = true;
    m_useMemBank = false;
    m_promptBboxPaddingScale = 0;
    m_maxMemVecSize_legacy = 100;

    m_vizDir = "trackerViz/";
    m_saveFrameViz = false;
    m_saveIDMask = false;
    m_outputMask = true;
    m_colorFormat = NVBUF_COLOR_FORMAT_RGB;
    m_segmenterType = SEGMENTER_TYPES::EfficientVitSAM;
    m_debugVerboseLevel = 0;
    m_updateMemBankPrompt = false;
    m_useCudaMemBank = false;

    m_numMaskMem = 7;
    m_numObjPtr = 15;
    m_numConditionalMem = 1;

    m_maskMemoryUpdateInterval = 0;
    /// Update memory bank when IOU is lower than this value.
    m_minIouDiff4MemoryUpdate = 0.8;
    m_minIouPrevFrame4ProgBbox = 0.0;
    m_minMemoryUpdateConfidence = 0.5;
    m_maskPromptUpdateInterval = 0;
    m_minPromptUpdateConfidence = 0.8;
}

/**
 * @brief Construct a SegmenterConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 */
SegmenterConfigParams::SegmenterConfigParams(const std::string &configFilePath, const std::string moduleName)
{
    /// Set the default values.
    *this = SegmenterConfigParams();

    m_trackerConfigPathForPathResolve = configFilePath;
    YAML::Node configRoot = getConfigRoot(configFilePath);
    /// Read the segmenter module parameters.
    this->read(configRoot[moduleName]);
    m_trackerConfigPathForPathResolve.clear();

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
 * @brief Construct a SegmenterConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
SegmenterConfigParams::SegmenterConfigParams(const YAML::Node &fn)
{
    read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void SegmenterConfigParams::read(const YAML::Node &fn)
{
    if(!fn.size())
    {
        return;
    }

    std::string segmenterConfigPath;
    for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
        std::string paramKey = fit->first.as<std::string>();
        if (paramKey == "segmenterType")
        {
            m_segmenterType = (SEGMENTER_TYPES)((int)fit->second.as<float>());
        }
        else if (paramKey == "segmenterConfigPath")
        {
            segmenterConfigPath = fit->second.as<std::string>();
        }
        else
        {
            cout << "!! [WARNING][SegmenterConfigParams] Unknown param found: " << paramKey << endl;
        }
    }

    if(segmenterConfigPath.empty())
    {
        throw std::runtime_error("segmenterConfigPath is not set");
    }

    std::string resolvedSegmenterYaml = segmenterConfigPath;
    if (!m_trackerConfigPathForPathResolve.empty())
    {
        resolvedSegmenterYaml = NvTrackerParams::resolvePathRelativeToConfig(
            m_trackerConfigPathForPathResolve, segmenterConfigPath
        );
    }

    YAML::Node fnSeg = YAML::LoadFile(resolvedSegmenterYaml);
    if(!fnSeg.size())
    {
        return;
    }

    for (YAML::const_iterator fit = fnSeg.begin(); fit != fnSeg.end(); ++fit)
    {
        std::string paramKey = fit->first.as<std::string>();
        if (paramKey == "segmenterType")
        {
            m_segmenterType = (SEGMENTER_TYPES)((int)fit->second.as<float>());
        }
        else if (paramKey == "debugVerboseLevel")
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
        else if (paramKey == "promptType")
        {
            m_promptType = (int)fit->second.as<float>();
            if (m_promptType > 1 || m_promptType < 0)
            {
                throw std::runtime_error("promptType should be 0 or 1");
            }
        }
        else if (paramKey == "outputLabelNum")
        {
            m_outputLabelNum = fit->second.as<float>();
        }
        else if (paramKey == "useVPICropScaler")
        {
            m_useVPICropScaler = (int)fit->second.as<float>();
        }
        else if (paramKey == "vizDir")
        {
            m_vizDir = fit->second.as<std::string>();
        }
        else if (paramKey == "saveFrameViz")
        {
            m_saveFrameViz = fit->second.as<float>();
        }
        else if (paramKey == "saveIDMask")
        {
            m_saveIDMask = fit->second.as<float>();
        }
        else if (paramKey == "outputMask")
        {
            m_outputMask = (int)fit->second.as<float>();
        }
        else if (paramKey == "maxMemVecSize")
        {
            m_maxMemVecSize_legacy = (int)fit->second.as<float>();
        }
        else if (paramKey == "updateMemBankPrompt")
        {
            m_updateMemBankPrompt = (int)fit->second.as<float>();
        }
        else if (paramKey == "useCudaMemBank")
        {
            m_useCudaMemBank = (int)fit->second.as<float>();
        }
        else if (paramKey == "numMaskMem")
        {
            m_numMaskMem = (int)fit->second.as<float>();
        }
        else if (paramKey == "numObjPtr")
        {
            m_numObjPtr = (int)fit->second.as<float>();
        }
        else if (paramKey == "numConditionalMem")
        {
            m_numConditionalMem = (int)fit->second.as<float>();
        }
        else if (paramKey == "maskMemoryUpdateInterval")
        {
            m_maskMemoryUpdateInterval = (int)fit->second.as<float>();
        }
        else if (paramKey == "minIouDiff4MemoryUpdate")
        {
            m_minIouDiff4MemoryUpdate = (float)fit->second.as<float>();
        }
        else if (paramKey == "minMemoryUpdateConfidence")
        {
            m_minMemoryUpdateConfidence = (float)fit->second.as<float>();
        }
        else if (paramKey == "maskPromptUpdateInterval")
        {
            m_maskPromptUpdateInterval = (int)fit->second.as<float>();
        }
        else if (paramKey == "minPromptUpdateConfidence")
        {
            m_minPromptUpdateConfidence = (float)fit->second.as<float>();
        }
        else if (paramKey == "minIouPrevFrame4ProgBbox")
        {
            m_minIouPrevFrame4ProgBbox = (float)fit->second.as<float>();
        }
        else
        {
            std::vector<std::string> allowedKeys = {
                "encoder", "decoder", "ImageEncoder", "MaskDecoder", "MemoryAttention", "MemoryEncoder"
            };
            if (std::find(allowedKeys.begin(), allowedKeys.end(), paramKey) == allowedKeys.end())
            {
                cout << "!! [WARNING][SegmenterConfigParams] Unknown param found: " << paramKey << endl;
            }
        }
    }
}

/** @brief Construct a SegmenterCudaMemBank instance. */
SegmenterCudaMemBank::SegmenterCudaMemBank()
{
    m_isInit = false;
}

/** @brief Release resources owned by the SegmenterCudaMemBank instance. */
SegmenterCudaMemBank::~SegmenterCudaMemBank()
{
    if (m_isInit)
    {
        CUDA_CHECK( cudaFree(m_targetsMaskMemFeatures_d) );
        CUDA_CHECK( cudaFree(m_targetsMaskMemPosEnc_d) );
        CUDA_CHECK( cudaFree(m_targetsTemporalCode_d) );
        CUDA_CHECK( cudaFree(m_targetsObjPtr_d) );
        CUDA_CHECK( cudaFree(m_work_0_d) );
    }
}

/**
 * @brief Allocate device memory for per-target segmentation state.
 * @param[in] frameInfo Metadata about the source image, such as size and color.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] paramsRt Runtime parameters for the segmenter.
 * @param[in] params Configuration parameters for the segmenter.
 * @param[in] cuStream CUDA stream that runs the GPU work.
 */
void SegmenterCudaMemBank::initialize(
    const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
    const SegmenterRunTimeParams &paramsRt, const SegmenterConfigParams &params, cudaStream_t& cuStream
)
{
    m_paramsRt = paramsRt;
    m_params = params;
    m_cuStream = cuStream;
    int totalNumTargets = maxTargetsPerStream * frameInfo.maxStreams;
    CUDA_CHECK(
        cudaMalloc(
            &m_targetsMaskMemFeatures_d,
            m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize * totalNumTargets * sizeof(float)
        )
    );
    CUDA_CHECK(
        cudaMalloc(
            &m_targetsMaskMemPosEnc_d,
            m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize * totalNumTargets * sizeof(float)
        )
    );
    CUDA_CHECK(
        cudaMalloc(
            &m_targetsTemporalCode_d,
            m_params.m_numMaskMem * m_paramsRt.temporalCodeSize * totalNumTargets * sizeof(float)
        )
    );
    CUDA_CHECK(
        cudaMalloc(
            &m_targetsObjPtr_d, (m_params.m_numObjPtr + 1) * m_paramsRt.objPtrSize * totalNumTargets * sizeof(float)
        )
    );
    CUDA_CHECK(
        cudaMalloc(
            &m_work_0_d,
            std::max(m_params.m_numMaskMem, m_params.m_numObjPtr + 1) * m_paramsRt.maskMemFeatSize * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsMaskMemFeatures_d, 0,
            m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize * totalNumTargets * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsMaskMemPosEnc_d, 0,
            m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize * totalNumTargets * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsTemporalCode_d, 0,
            m_params.m_numMaskMem * m_paramsRt.temporalCodeSize * totalNumTargets * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsObjPtr_d, 0,
            (m_params.m_numObjPtr + 1) * m_paramsRt.objPtrSize * totalNumTargets * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_work_0_d, 0,
            std::max(m_params.m_numMaskMem, m_params.m_numObjPtr + 1) * m_paramsRt.maskMemFeatSize * sizeof(float),
            m_cuStream
        )
    );

    m_isInit = true;
}

/**
 * @brief Clear retained memory for one target.
 * @param[in] targetInd Target index.
 */
void SegmenterCudaMemBank::reset(const TrackerIndex targetInd)
{
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsMaskMemFeatures_d + targetInd * m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize,
            0, m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsMaskMemPosEnc_d + targetInd * m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize,
            0, m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsTemporalCode_d + targetInd * m_params.m_numMaskMem * m_paramsRt.temporalCodeSize,
            0, m_params.m_numMaskMem * m_paramsRt.temporalCodeSize * sizeof(float), m_cuStream
        )
    );
    CUDA_CHECK(
        cudaMemsetAsync(
            m_targetsObjPtr_d + targetInd * (m_params.m_numObjPtr + 1) * m_paramsRt.objPtrSize,
            0, (m_params.m_numObjPtr + 1) * m_paramsRt.objPtrSize * sizeof(float), m_cuStream
        )
    );
}

/**
 * @brief Insert one tensor into a bounded temporal bank.
 * @param[in] srcBatch Source batch containing the new tensor.
 * @param[in,out] dst Tensor bank that receives the new tensor.
 * @param[in] numFilled Number of populated tensors in the destination bank.
 * @param[in] maxNumPerTarget Maximum number of tensors stored per target.
 * @param[in] numElementsPerTensor Number of elements in each tensor.
 * @param[in] targetInd Index of the target.
 * @param[in] indInBatch Index of this target in the inference batch.
 * @param[in] keepFirst When the bank is full, keep the first tensor and drop the second oldest.
 */
void SegmenterCudaMemBank::updateTensorBank(
    float* srcBatch, float* dst, int& numFilled, const int maxNumPerTarget, const int numElementsPerTensor,
    const TrackerIndex targetInd, const int indInBatch, const bool keepFirst
)
{
    /// Get starting address of the current target.
    float* targetBank = dst + targetInd * maxNumPerTarget * numElementsPerTensor;
    /// TensorNew size is equal to numElementsPerTensor.
    float *tensorNew = srcBatch + indInBatch * numElementsPerTensor;
    if (numFilled < maxNumPerTarget)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(
                targetBank + numFilled * numElementsPerTensor, tensorNew,
                numElementsPerTensor * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
    }
    else if (keepFirst)
    {
        /// Overwrite the second oldest tensor.
        int elementsToCopy = (maxNumPerTarget-2) * numElementsPerTensor;
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_work_0_d, targetBank + 2 * numElementsPerTensor,
                elementsToCopy * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_work_0_d + elementsToCopy, tensorNew,
                numElementsPerTensor * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                targetBank + numElementsPerTensor, m_work_0_d,
                (maxNumPerTarget-1) * numElementsPerTensor * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
    }
    else
    {
        /// Overwrite the oldest tensor.
        int elementsToCopy = (maxNumPerTarget-1) * numElementsPerTensor;
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_work_0_d, targetBank + numElementsPerTensor,
                elementsToCopy * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_work_0_d + elementsToCopy, tensorNew,
                numElementsPerTensor * sizeof(float), cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
        CUDA_CHECK(
            cudaMemcpyAsync(
                targetBank, m_work_0_d, maxNumPerTarget * numElementsPerTensor * sizeof(float),
                cudaMemcpyDeviceToDevice, m_cuStream
            )
        );
    }
}

/**
 * @brief Update retained state after data association.
 * @param[in] maskMemFeatures_d Mask memory features to append.
 * @param[in] maskMemPosEnc_d Mask memory position encoding to append.
 * @param[in] temporalCode_d Device temporal-code tensor.
 * @param[in] objPtr_d Device object-pointer tensor.
 * @param[in] target Target whose memory bank is updated.
 * @param[in] indInBatch Index of this target in the inference batch.
 * @param[in,out] targetStatus CUDA inference status whose stored-tensor counts are updated.
 */
void SegmenterCudaMemBank::update(
    float* maskMemFeatures_d, float* maskMemPosEnc_d, float* temporalCode_d, float* objPtr_d,
    std::shared_ptr<NvTrackedObject> target, int indInBatch, CudaInferenceStatus& targetStatus
)
{
    updateTensorBank(
        maskMemFeatures_d, m_targetsMaskMemFeatures_d, targetStatus.numStatusUncond,
        m_params.m_numMaskMem, m_paramsRt.maskMemFeatSize, target->getTargetInd(), indInBatch
    );
    updateTensorBank(
        maskMemPosEnc_d, m_targetsMaskMemPosEnc_d, targetStatus.numStatusUncond,
        m_params.m_numMaskMem, m_paramsRt.maskMemFeatSize, target->getTargetInd(), indInBatch
    );

    /// Only 1 temporal code per batch.
    updateTensorBank(
        temporalCode_d, m_targetsTemporalCode_d, targetStatus.numStatusUncond,
        m_params.m_numMaskMem, m_paramsRt.temporalCodeSize, target->getTargetInd(), 0
    );

    if (targetStatus.numStatusUncond < m_params.m_numMaskMem)
    {
        targetStatus.numStatusUncond++;
    }

    /// Update objPtr.
    updateTensorBank(
        objPtr_d, m_targetsObjPtr_d, targetStatus.numObjPtrUncond, m_params.m_numObjPtr + 1,
        m_paramsRt.objPtrSize, target->getTargetInd(), indInBatch, true
    );
    if (targetStatus.numObjPtrUncond < m_params.m_numObjPtr + 1)
    {
        targetStatus.numObjPtrUncond++;
    }
}

/**
 * @brief Get CUDA memory-bank pointers for a target.
 * @param[in] targetInd Target index.
 */
std::vector<float*> SegmenterCudaMemBank::getCudaMemBank(const TrackerIndex targetInd)
{
    std::vector<float*> bank;
    bank.push_back(m_targetsObjPtr_d + targetInd * (m_params.m_numObjPtr + 1) * m_paramsRt.objPtrSize);
    bank.push_back(m_targetsMaskMemFeatures_d + targetInd * m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize);
    bank.push_back(m_targetsMaskMemPosEnc_d + targetInd * m_params.m_numMaskMem * m_paramsRt.maskMemFeatSize);
    bank.push_back(m_targetsTemporalCode_d + targetInd * m_params.m_numMaskMem * m_paramsRt.temporalCodeSize);
    return bank;
}
