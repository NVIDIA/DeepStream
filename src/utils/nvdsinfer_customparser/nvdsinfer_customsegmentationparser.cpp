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
