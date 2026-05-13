/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_ibackend.h
 *
 * @brief Inference processing backend interface header file.
 *
 * This file defines the interface for the backend of inference processing
 * using the Triton Inference Server.
 *
 */

#ifndef __NVDSINFERSERVER_IBACKEND_H__
#define __NVDSINFERSERVER_IBACKEND_H__

#include "infer_common.h"
#include "infer_datatypes.h"

namespace nvdsinferserver {

/**
 * @brief Stores the information of a layer in the inference model.
 */
struct LayerDescription {
    /**
     * @brief Data type of the layer.
     */
    InferDataType dataType = InferDataType::kFp32;
    /**
     * @brief Dimensions of the layer.
     */
    InferDims inferDims;
    /**
     * @brief Index of the layer as per sequence in which the layer is added
     * to the list of layers.
     */
    int bindingIndex = 0;
    /**
     * @brief True if the layer is an input layer.
     */
    bool isInput = 0;
    /**
     * @brief Name of the model layer.
     */
    std::string name;
};

using LayerDescriptionList = std::vector<LayerDescription>;

class IBackend {
public:
    /**
     * @brief Function wrapper for post inference processing.
     */
    using InferenceDone =
        std::function<void(NvDsInferStatus, SharedBatchArray)>;
    /**
     * @brief Function wrapper called after the input buffer is consumed.
     */
    using InputsConsumed = std::function<void(SharedBatchArray)>;

    enum { kLTpLayerDesc, kTpLayerNum };
    /**
     * @brief Tuple containing pointer to layer descriptions and the number of
     * layers.
     */
    using LayersTuple = std::tuple<const LayerDescription*, int>;

    enum { kInShapeName, kInShapeDims };
    /**
     * @brief Tuple of layer name and dimensions including batch size.
     */
    using InputShapeTuple = std::tuple<std::string, InferBatchDims>;
    using InputShapes = std::vector<InputShapeTuple>;

    /**
     * @brief Constructor, default.
     */
    IBackend() = default;

    /**
     * @brief Destructor, default.
     */
    virtual ~IBackend() = default;

    /**
     * @brief Initialize the backend for processing.
     * @return Status code of the type NvDsInferStatus.
     */
    virtual NvDsInferStatus initialize() = 0;

    /**
     * @brief Specify the input layers for the backend.
     * @param shapes List of name and shapes of the input layers.
     * @return Status code of the type NvDsInferStatus.
     */
    virtual NvDsInferStatus specifyInputDims(const InputShapes& shapes) = 0;

    /**
     * @brief Check if the flag for first dimension being batch is set.
     */
    virtual bool isFirstDimBatch() const = 0;

    /**
     * @brief Get the tensor order set for the input.
     */
    virtual InferTensorOrder getInputTensorOrder() const = 0;

    /**
     * @brief Get the configured maximum batch size for this backend.
     */
    virtual int32_t maxBatchSize() const = 0;

    /**
     * @brief Get the number of layers (input and output) for the model.
     */
    virtual uint32_t getLayerSize() const = 0;

    /**
     * @brief Get the number of input layers.
     */
    virtual uint32_t getInputLayerSize() const = 0;

    /**
     * @brief Get the layer description from the layer name.
     */
    virtual const LayerDescription* getLayerInfo(
        const std::string& bindingName) const = 0;

    /**
     * @brief Get the LayersTuple for input layers.
     */
    virtual LayersTuple getInputLayers() const = 0;

    /**
     * @brief Get the LayersTuple for output layers.
     */
    virtual LayersTuple getOutputLayers() const = 0;

    /**
     * @brief Enqueue an array of input batches for inference.
     *
     * This function adds a input to the inference processing queue of the
     * backend. The post inference function and function to be called after
     * consuming input buffer is provided.
     *
     * @param[in] inputs        List of input batch buffers
     * @param[in] stream        The CUDA stream to be used in inference
     *                          processing.
     * @param[in] bufConsumed   Function to be called once input buffer is
     *                          consumed.
     * @param[in] inferenceDone Function to be called after inference.
     * @return                  Execution status code.
     */
    virtual NvDsInferStatus enqueue(
        SharedBatchArray inputs, SharedCuStream stream,
        InputsConsumed bufConsumed, InferenceDone inferenceDone) = 0;

private:
    DISABLE_CLASS_COPY(IBackend);
};

} // namespace nvdsinferserver

#endif
