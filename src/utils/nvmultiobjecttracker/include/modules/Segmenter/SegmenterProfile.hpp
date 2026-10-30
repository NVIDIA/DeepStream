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

#include <NvInfer.h>

/**
 * @brief Configure dynamic input shapes for the NanoSAM decoder.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4NanoSamDecoder(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

    nvinfer1::ITensor* input = network->getInput(1);
    const char* inputNamePts = input->getName();
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 1, 2));
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 10, 2));

    input = network->getInput(2);
    const char* inputNamePtLabels = input->getName();
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 1));
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 10));

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for the NanoSAM decoder.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4NanoSamDecoder(TrtModule *trtModule, const std::vector<int>& sizes)
{
    trtModule->m_context->setOptimizationProfileAsync(0, trtModule->m_cuStream);
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(1), nvinfer1::Dims3(1, 2, 2));
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(2), nvinfer1::Dims2(1, 2));
}

/**
 * @brief Configure dynamic input shapes for the EfficientViT-SAM encoder.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4EVSamEncoder(
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

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for the EfficientViT-SAM encoder.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4EVSamEncoder(TrtModule *trtModule, const std::vector<int>& sizes)
{
    for (int ind = 0; ind < trtModule->m_engine->getNbIOTensors(); ++ind)
    {
        if (trtModule->m_engine->getTensorIOMode(trtModule->m_engine->getIOTensorName(ind)) == nvinfer1::TensorIOMode::kINPUT)
        {
            nvinfer1::Dims lastInputDims = trtModule->m_context->getTensorShape(
                trtModule->m_engine->getIOTensorName(ind)
            );
            lastInputDims.d[0] = 1;
            trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(ind), lastInputDims);
        }
    }
}

/**
 * @brief Configure dynamic input shapes for the EfficientViT-SAM decoder.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4EVSamDecoder(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    int maxPointNum = batchSize * 2;
    nvinfer1::ITensor* input = network->getInput(1);
    const char* inputNamePts = input->getName();
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 2 * 2));
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, maxPointNum * 2));
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, maxPointNum * 2));

    input = network->getInput(2);
    const char* inputNamePtLabels = input->getName();
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 2));
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, maxPointNum));
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, maxPointNum));

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for the EfficientViT-SAM decoder.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4EVSamDecoder(TrtModule *trtModule, const std::vector<int>& sizes)
{
    trtModule->m_context->setOptimizationProfileAsync(0, trtModule->m_cuStream);
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(1), nvinfer1::Dims2(1, 2 * 2));
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(2), nvinfer1::Dims2(1, 2));
}

/**
 * @brief Configure dynamic input shapes for the SAM2 mask decoder.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4Sam2MaskDecoder(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

    nvinfer1::ITensor* input = network->getInput(0);
    const char* inputNamePts = input->getName();
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(batchSize, 2, 2));
    profile->setDimensions(inputNamePts, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(batchSize, 2, 2));

    input = network->getInput(1);
    const char* inputNamePtLabels = input->getName();
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(batchSize, 2));
    profile->setDimensions(inputNamePtLabels, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(batchSize, 2));

    input = network->getInput(2);
    const char* inputNamePixFeatWithMem = input->getName();
    profile->setDimensions(
        inputNamePixFeatWithMem, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 256, 64, 64)
    );
    profile->setDimensions(
        inputNamePixFeatWithMem, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(batchSize, 256, 64, 64)
    );
    profile->setDimensions(
        inputNamePixFeatWithMem, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(batchSize, 256, 64, 64)
    );

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for the SAM2 mask decoder.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4Sam2MaskDecoder(TrtModule *trtModule, const std::vector<int>& sizes)
{
    int pointNum = sizes.at(0);
    int batchSize = (sizes.size() >= 2) ? sizes.at(1) : 1;
    trtModule->m_context->setOptimizationProfileAsync(0, trtModule->m_cuStream);
    trtModule->m_context->setInputShape(
        trtModule->m_engine->getIOTensorName(0), nvinfer1::Dims3(batchSize, pointNum, 2)
    );
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(1), nvinfer1::Dims2(batchSize, pointNum));
    trtModule->m_context->setInputShape(
        trtModule->m_engine->getIOTensorName(2), nvinfer1::Dims4(batchSize, 256, 64, 64)
    );
}

/**
 * @brief Configure dynamic input shapes for SAM2 memory attention.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4Sam2MemAttn(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    int numObjPtr = 16, numMaskMem = 7;
    /// m_memory_obj_ptr_d [b x 1 x 256] to [b x 16 x 256].
    nvinfer1::ITensor* input = network->getInput(2);
    const char* inputName2 = input->getName();
    profile->setDimensions(inputName2, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 256));
    profile->setDimensions(inputName2, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(batchSize, numObjPtr, 256));
    profile->setDimensions(inputName2, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(batchSize, numObjPtr, 256));

    /// m_memory_maskmem_features_d [b x 1 x 64 x 64 x 64] to [b x 7 x 64 x 64 x 64].
    input = network->getInput(3);
    const char* inputName3 = input->getName();
    nvinfer1::Dims inputDim = {5, {1,1,64,64,64}};
    profile->setDimensions(inputName3, nvinfer1::OptProfileSelector::kMIN, inputDim);
    inputDim.d[0] = batchSize, inputDim.d[1] = numMaskMem;
    profile->setDimensions(inputName3, nvinfer1::OptProfileSelector::kOPT, inputDim);
    inputDim.d[0] = batchSize, inputDim.d[1] = numMaskMem;
    profile->setDimensions(inputName3, nvinfer1::OptProfileSelector::kMAX, inputDim);

    /// Positional encoding shape: [batch, mask memories * 4096 + object pointers * 4, 64].
    /// This gives sequence lengths of 4100 at minimum and 28736 at maximum.
    input = network->getInput(4);
    const char* inputName4 = input->getName();
    profile->setDimensions(inputName4, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 4100, 64));
    profile->setDimensions(
        inputName4, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(batchSize, numMaskMem*4096 + 4*numObjPtr, 64)
    );
    profile->setDimensions(
        inputName4, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(batchSize, numMaskMem*4096 + 4*numObjPtr, 64)
    );

    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for SAM2 memory attention.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4Sam2MemAttn(TrtModule *trtModule, const std::vector<int>& sizes)
{
    int batchSize = sizes.at(0), numObjPtrs = sizes.at(1), numMaskMem = sizes.at(2), numFrames = sizes.at(3);
    trtModule->m_context->setOptimizationProfileAsync(0, trtModule->m_cuStream);
    trtModule->m_context->setInputShape(
        trtModule->m_engine->getIOTensorName(2), nvinfer1::Dims3(batchSize, numObjPtrs, 256)
    );
    nvinfer1::Dims inputDim = {5, {batchSize,numMaskMem,64,64,64}};
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(3), inputDim);
    trtModule->m_context->setInputShape(
        trtModule->m_engine->getIOTensorName(4), nvinfer1::Dims3(batchSize, numMaskMem*4096 + 4*numFrames, 64)
    );
}

/**
 * @brief Configure dynamic input shapes for the SAM2 memory encoder.
 * @param[in,out] network TensorRT network definition.
 * @param[in] builder TensorRT engine builder.
 * @param[in,out] config TensorRT builder configuration.
 * @param[in] batchSize Maximum runtime batch size.
 */
static void createOnnxProfile4Sam2MemEncoder(
    nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int batchSize
)
{
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

    nvinfer1::ITensor* input = network->getInput(0);
    const char* inputNameMaskForMem = input->getName();
    profile->setDimensions(inputNameMaskForMem, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 1024, 1024));
    profile->setDimensions(
        inputNameMaskForMem, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(batchSize, 1, 1024, 1024)
    );
    profile->setDimensions(
        inputNameMaskForMem, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(batchSize, 1, 1024, 1024)
    );

    input = network->getInput(2);
    const char* inputNameOccLogit = input->getName();
    profile->setDimensions(inputNameOccLogit, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions(inputNameOccLogit, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(batchSize, 1));
    profile->setDimensions(inputNameOccLogit, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(batchSize, 1));
    config->addOptimizationProfile(profile);
}

/**
 * @brief Set runtime input shapes for the SAM2 memory encoder.
 * @param[in,out] trtModule TensorRT module whose runtime dimensions are configured.
 * @param[in] sizes Runtime tensor dimensions.
 */
static void setRuntimeDim4Sam2MemEncoder(TrtModule *trtModule, const std::vector<int>& sizes)
{
    int batchSize = sizes.at(0);
    trtModule->m_context->setOptimizationProfileAsync(0, trtModule->m_cuStream);
    trtModule->m_context->setInputShape(
        trtModule->m_engine->getIOTensorName(0), nvinfer1::Dims4(batchSize, 1, 1024, 1024)
    );
    trtModule->m_context->setInputShape(trtModule->m_engine->getIOTensorName(2), nvinfer1::Dims2(batchSize, 1));
}
