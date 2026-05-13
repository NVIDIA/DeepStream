/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cstring>
#include <iostream>
#include <cassert>
#include "nvdsinfer_custom_impl.h"

/* This is a custom parsing function for the TAO PeopleSemSegNet model
 * provided at NGC. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomPeopleSemSegNet(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo, float segmentationThreshold,
        unsigned int numClasses, int* classificationMap,
        float*& classProbabilityMap);

extern "C"
bool NvDsInferParseCustomPeopleSemSegNet(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo, float segmentationThreshold,
        unsigned int numClasses, int* classificationMap,
        float*& classProbabilityMap) {

    assert(classificationMap);

    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if ((layer.dataType == INT32 || layer.dataType == INT64) &&
                (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *classMapLayer = layerFinder("argmax_1"); // height x width x 1

    if (!classMapLayer) {
        std::cerr << "ERROR: Output layer argmax_1 not found in output tensors"
                  << " or was not of type INT32/INT64" << std::endl;
        return false;
    }

    NvDsInferDims outputDims = classMapLayer->inferDims;

    if (outputDims.numDims != 3U) {
        std::cerr << "Network output number of dims is : " <<
            outputDims.numDims << " expected is 3."<< std::endl;
        return false;
    }

   if (outputDims.d[0] != networkInfo.height
           || outputDims.d[1] != networkInfo.width
           || outputDims.d[2] != 1) {
        std::cerr << "Incorrect output layer dimensions : " <<
            outputDims.d[0] << " expected is " <<
            networkInfo.height << "x" << networkInfo.width << "x1." << std::endl;
        return false;
    }

    if (classMapLayer->dataType == INT64) {
        /* Converting INT64 layer into INT32 */
        int32_t *tmp_buf = classificationMap;
        int64_t *class_buf = (static_cast<int64_t *>(classMapLayer->buffer));
        for (unsigned int i = 0; i < networkInfo.width * networkInfo.height; i++) {
            *tmp_buf = static_cast<int32_t> (*class_buf);
            tmp_buf++;
            class_buf++;
        }
    } else {
        memcpy (classificationMap, classMapLayer->buffer, sizeof(int32_t) * networkInfo.width * networkInfo.height);
    }

    classProbabilityMap = nullptr;

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPeopleSemSegNet);
