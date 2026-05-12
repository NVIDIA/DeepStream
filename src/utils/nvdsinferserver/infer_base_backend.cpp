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
 * @file infer_base_backend.cpp
 *
 * @brief Source file for inference processing backend base class.
 *
 * This file defines the base class for the backend of inference processing
 * using the Triton Inference Server.
 */

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include "infer_base_backend.h"
#include "infer_cuda_utils.h"
#include "infer_utils.h"

namespace nvdsinferserver {

bool BaseBackend::checkInputDims(const InputShapes& shapes) const {
    assert(shapes.size());
    for (const auto& bindings : shapes) {
        const LayerDescription* layerDesc =
            getLayerInfo(std::get<kInShapeName>(bindings));
        const InferBatchDims& dims = std::get<kInShapeDims>(bindings);

        RETURN_IF_FAILED(
            layerDesc && layerDesc->isInput, false,
            "Failed to specify input dimension which is not an input layer.");
        RETURN_IF_FAILED(
            dims.batchSize >= 0 && !hasWildcard(dims.dims), false,
            "Failed to specify input dimension which has wildcard values.");
    }
    return true;
}

IBackend::LayersTuple BaseBackend::getInputLayers() const {
    return std::make_tuple(
        m_AllLayers.empty() ? nullptr : m_AllLayers.data(), m_InputSize);
}

IBackend::LayersTuple BaseBackend::getOutputLayers() const {
    return std::make_tuple(
        m_AllLayers.empty() ? nullptr : &m_AllLayers.at(m_InputSize),
        m_AllLayers.size() - m_InputSize);
}

const LayerDescription*
BaseBackend::getLayerInfo(const std::string& name) const
{
    auto iter = m_LayerNameToIdx.find(name);
    if (iter == m_LayerNameToIdx.end()) {
        return nullptr;
    }
    const LayerDescriptionList& layers = allLayers();
    int idx = iter->second;
    if (idx < 0 || idx >= (int)layers.size()) {
        InferError("Internal error to get tensor:%s layer info", safeStr(name));
        return nullptr;
    }
    return &(layers.at(idx));
}

void
BaseBackend::resetLayers(LayerDescriptionList layers, int inputSize)
{
    m_InputSize = inputSize;
    m_LayerNameToIdx.clear();
    for (size_t i = 0; i < layers.size(); ++i) {
        assert(!m_LayerNameToIdx.count(layers[i].name));
        m_LayerNameToIdx[layers[i].name] = i;
    }
    m_AllLayers = std::move(layers);
}

} // namespace nvdsinferserver
