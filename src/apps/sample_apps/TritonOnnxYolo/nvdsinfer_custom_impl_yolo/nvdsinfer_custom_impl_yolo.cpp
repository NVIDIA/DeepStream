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

#include <algorithm>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

float
intersectionOverUnion(
    const NvDsInferObjectDetectionInfo& a, const NvDsInferObjectDetectionInfo& b)
{
    const float interLeft = std::max(a.left, b.left);
    const float interTop = std::max(a.top, b.top);
    const float interRight = std::min(a.left + a.width, b.left + b.width);
    const float interBottom = std::min(a.top + a.height, b.top + b.height);
    const float interArea =
        std::max(0.0F, interRight - interLeft) * std::max(0.0F, interBottom - interTop);
    const float unionArea = a.width * a.height + b.width * b.height - interArea;
    return unionArea > 0.0F ? interArea / unionArea : 0.0F;
}

const NvDsInferLayerInfo*
findLayer(const std::vector<NvDsInferLayerInfo>& layers, const char* name)
{
    for (const auto& layer : layers) {
        if (layer.layerName && std::strcmp(layer.layerName, name) == 0) {
            return &layer;
        }
    }
    return nullptr;
}

}  // namespace

extern "C" bool
NvDsInferInitializeInputLayers(
    std::vector<NvDsInferLayerInfo> const& inputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, unsigned int maxBatchSize)
{
    (void)maxBatchSize;
    for (const auto& layer : inputLayersInfo) {
        if (layer.layerName && std::strcmp(layer.layerName, "image_shape") == 0) {
            auto* imageShape = static_cast<float*>(layer.buffer);
            imageShape[0] = static_cast<float>(networkInfo.height);
            imageShape[1] = static_cast<float>(networkInfo.width);
            return true;
        }
    }
    return false;
}

extern "C" bool
NvDsInferParseCustomYoloOriginal(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    (void)networkInfo;
    const auto* boxesLayer =
        findLayer(outputLayersInfo, "yolonms_layer_1/ExpandDims_1:0");
    const auto* scoresLayer =
        findLayer(outputLayersInfo, "yolonms_layer_1/ExpandDims_3:0");
    // Do not parse yolonms_layer_1/concat_2:0 here. Gst-nvinfer exposes only
    // one dynamic NMS index row for this model, which drops most boxes.
    // Use the dense boxes/scores tensors and run NMS in this parser instead.
    if (!boxesLayer || !scoresLayer || !boxesLayer->buffer || !scoresLayer->buffer) {
        return false;
    }

    int numBoxes = 0;
    int numClasses = 0;
    if (boxesLayer->inferDims.numDims == 3 && boxesLayer->inferDims.d[2] == 4) {
        numBoxes = boxesLayer->inferDims.d[1];
    } else if (boxesLayer->inferDims.numDims == 2 && boxesLayer->inferDims.d[1] == 4) {
        numBoxes = boxesLayer->inferDims.d[0];
    } else {
        return false;
    }

    if (scoresLayer->inferDims.numDims == 3 &&
        static_cast<int>(scoresLayer->inferDims.d[2]) == numBoxes) {
        numClasses = scoresLayer->inferDims.d[1];
    } else if (scoresLayer->inferDims.numDims == 2 &&
               static_cast<int>(scoresLayer->inferDims.d[1]) == numBoxes) {
        numClasses = scoresLayer->inferDims.d[0];
    } else {
        return false;
    }

    const auto* boxes = static_cast<const float*>(boxesLayer->buffer);
    const auto* scores = static_cast<const float*>(scoresLayer->buffer);
    const float threshold = detectionParams.perClassThreshold[0];
    const float nms_iou_threshold = 0.5F;
    const int keep_top_k = 200;

    std::vector<NvDsInferObjectDetectionInfo> candidates;
    for (int classId = 0; classId < numClasses; ++classId) {
        if (detectionParams.numClassesConfigured > 0 &&
            static_cast<unsigned int>(classId) >= detectionParams.numClassesConfigured) {
            break;
        }

        candidates.clear();
        for (int boxIdx = 0; boxIdx < numBoxes; ++boxIdx) {
            const float score = scores[classId * numBoxes + boxIdx];
            if (score < threshold) {
                continue;
            }

            const float* bbox = &boxes[4 * boxIdx];
            NvDsInferObjectDetectionInfo object{};
            object.classId = static_cast<unsigned int>(classId);
            object.top = bbox[0];
            object.left = bbox[1];
            object.height = bbox[2] - object.top;
            object.width = bbox[3] - object.left;
            object.detectionConfidence = score;
            if (object.width > 0.0F && object.height > 0.0F) {
                candidates.emplace_back(object);
            }
        }

        std::sort(
            candidates.begin(), candidates.end(),
            [](const NvDsInferObjectDetectionInfo& a,
               const NvDsInferObjectDetectionInfo& b) {
                return a.detectionConfidence > b.detectionConfidence;
            });

        for (const auto& candidate : candidates) {
            bool keep = true;
            for (const auto& selected : objectList) {
                if (selected.classId == candidate.classId &&
                    intersectionOverUnion(selected, candidate) > nms_iou_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                objectList.emplace_back(candidate);
                if (objectList.size() >= static_cast<size_t>(keep_top_k)) {
                    return true;
                }
            }
        }
    }
    return true;
}
