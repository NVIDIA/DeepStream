/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA DeepStream API for importing Transfer Learning Toolkit
 *  encoded models </b>
 *
 * @b Description: This file specifies the API to decode and create
 * a CUDA engine file from a Tranfer Learning Toolkit (TLT) encoded model.
 */

/**
 * @defgroup ee_nvdsinfer_tlt  Import Transfer Learning Toolkit Encoded Models
 *
 * Defines an API for importing Transfer Learning Toolkit encoded models.
 *
 * @ingroup NvDsInferApi
 * @{
 */

#ifndef __NVDSINFER_TLT_H__
#define __NVDSINFER_TLT_H__

#include <nvdsinfer_custom_impl.h>

/**
 * \brief  Decodes and creates a CUDA engine file from a TLT encoded model.
 *
 * This function implements the @ref NvDsInferCudaEngineGet interface. The
 * correct key and model path must be provided in the @a tltModelKey and
 * @a tltEncodedModelFilePath members of @a initParams. Other parameters
 * applicable to UFF models also apply to TLT encoded models.
 */
extern "C"
bool NvDsInferCudaEngineGetFromTltModel(nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

#endif

/** @} */
