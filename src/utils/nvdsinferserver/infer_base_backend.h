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
 * @file infer_base_backend.h
 *
 * @brief Header file for inference processing backend base class.
 *
 * This file declares the base class for the backend of inference processing
 * using the Triton Inference Server.
 */

#ifndef __NVDSINFER_BASE_BACKEND_H__
#define __NVDSINFER_BASE_BACKEND_H__

#include <stdarg.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "infer_datatypes.h"
#include "infer_ibackend.h"
#include "infer_utils.h"

namespace nvdsinferserver {

/**
 * @brief Base class of inference backend processing.
 */
class BaseBackend : public IBackend {
public:
    /**
     * @brief Destructor, default.
     */
    ~BaseBackend() override = default;
    /**
     * @brief Returns the input tensor order.
     */
    InferTensorOrder getInputTensorOrder() const final { return m_InputOrder; }

    /**
     * @brief Set the unique ID for the object instance.
     */
    void setUniqueId(uint32_t id) { m_UniqueId = id; }

    /**
     * @brief Get the unique ID of the object instance.
     */
    int uniqueId() const { return m_UniqueId; }

    /**
     * @brief Set the flag indicating that it is a batch input.
     */
    void setFirstDimBatch(bool flag) { m_IsFirstDimBatch = flag; }

    /**
     * @brief Returns boolean indicating if batched input is expected.
     */
    bool isFirstDimBatch() const final { return m_IsFirstDimBatch; }

    /**
     * @brief Returns the total number of layers (input + output) for the
     * model.
     */
    uint32_t getLayerSize() const final {
        assert(!m_AllLayers.empty());
        return (int)m_AllLayers.size();
    }

    /**
     * @brief Returns the number of input layers for the model.
     */
    uint32_t getInputLayerSize() const final {
        return m_InputSize;
    }

    /**
     * @brief Retrieve the layer information from the layer name.
     */
    const LayerDescription* getLayerInfo(
        const std::string& bindingName) const final;

    /**
     * @brief Get the LayersTuple for input layers.
     */
    LayersTuple getInputLayers() const final;

    /**
     * @brief Get the LayersTuple for output layers.
     */
    LayersTuple getOutputLayers() const final;

    /**
     * @brief Check that the list of input shapes have fixed dimensions
     * and corresponding layers are marked as input layers.
     */
    bool checkInputDims(const InputShapes& shapes) const;

    /**
     * @brief Returns the list of all descriptions of all layers, input and
     * output.
     */
    const LayerDescriptionList& allLayers() const { return m_AllLayers; }

    /**
     * @brief Set the flag indicating whether to keep inputs buffers.
     */
    void setKeepInputs(bool enable) {
        m_KeepInputs = enable;
    }

    /**
     * @brief Returns the maximum batch size set for the backend.
     */
    int32_t maxBatchSize() const final { return m_MaxBatchSize; }

    /**
     * @brief Checks if the batch size indicates batched processing or no.
     */
    bool isNonBatching() const { return isNonBatch(maxBatchSize()); }

protected:
    /**
     * @brief Map of layer name to layer index.
     */
    using LayerIdxMap = std::unordered_map<std::string, int>;

    /**
     * @brief Set the layer description list of the backend.
     *
     * This function sets the layer description for the backend and
     * updates the number of input layers, layer name to index map.
     *
     * @param[in] layers    The list of descriptions for all layers, input
     *                      followed by output layers.
     * @param[in] inputSize The number of input layers in the list.
     */
    void resetLayers(LayerDescriptionList layers, int inputSize);

    /**
     * @brief Get the mutable layer description structure for the layer name.
     */
    LayerDescription* mutableLayerInfo(const std::string& bindingName)
    {
        const LayerDescription* info = getLayerInfo(bindingName);
        return const_cast<LayerDescription*>(info);
    }

    /**
     * @brief Set the tensor order for the input layers.
     */
    void setInputTensorOrder(InferTensorOrder order) { m_InputOrder = order; }

    /**
     * @brief Check if the keep input flag is set.
     */
    bool needKeepInputs() const {
        return m_KeepInputs;
    }

    /**
     * @brief Set the maximum batch size to be used for the backend.
     */
    void setMaxBatchSize(uint32_t size) { m_MaxBatchSize = size; }

private:
    /**
     * @brief List of descriptions for all layers (input and output).
     */
    LayerDescriptionList m_AllLayers;
    /**
     * @brief Map of layer name to layer index.
     */
    LayerIdxMap m_LayerNameToIdx;
    /**
     * @brief Number of input layers for the model.
     */
    uint32_t m_InputSize = 0;
    /**
     * @brief Maximum batch size to be used for inference processing.
     */
    int32_t m_MaxBatchSize = 0;
    /**
     * @brief Flag to indicate if the first dimension is the batch size.
     */
    bool m_IsFirstDimBatch = false;

    /**
     * @brief The expected tensor order for the input layers.
     */
    InferTensorOrder m_InputOrder = InferTensorOrder::kNone;
    /**
     * @brief Unique ID for the instance.
     */
    uint32_t m_UniqueId = 0;
    /**
     * @brief Flag to indicate the input buffers should be retained.
     */
    bool m_KeepInputs = false;
};

using UniqBackend = std::unique_ptr<BaseBackend>;

} // namespace nvdsinferserver
#endif
