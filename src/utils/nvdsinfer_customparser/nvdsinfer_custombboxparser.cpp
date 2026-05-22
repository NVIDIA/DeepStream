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
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple IoU calculation for NMS (axis-aligned approximation)
static float calculateIoU(const NvDsInferObjectDetectionInfo& box1, 
                   const NvDsInferObjectDetectionInfo& box2) {
    float x1_min = box1.left;
    float y1_min = box1.top;
    float x1_max = box1.left + box1.width;
    float y1_max = box1.top + box1.height;

    float x2_min = box2.left;
    float y2_min = box2.top;
    float x2_max = box2.left + box2.width;
    float y2_max = box2.top + box2.height;

    // Calculate intersection
    float inter_x1 = std::max(x1_min, x2_min);
    float inter_y1 = std::max(y1_min, y2_min);
    float inter_x2 = std::min(x1_max, x2_max);
    float inter_y2 = std::min(y1_max, y2_max);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);

    // Calculate union
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - inter_area;

    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

// NMS implementation for OBB detections
static void performNMS(std::vector<NvDsInferObjectDetectionInfo>& detections, float nms_threshold) {
    if (detections.empty()) return;

    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
              [](const NvDsInferObjectDetectionInfo& a, const NvDsInferObjectDetectionInfo& b) {
                  return a.detectionConfidence > b.detectionConfidence;
              });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        // Get class ID directly (no decoding needed)
        uint32_t class_i = detections[i].classId;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            // Get class ID directly
            uint32_t class_j = detections[j].classId;

            // Only suppress boxes of the same class
            if (class_i != class_j) continue;

            float iou = calculateIoU(detections[i], detections[j]);
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // Remove suppressed detections
    std::vector<NvDsInferObjectDetectionInfo> filtered;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            filtered.push_back(detections[i]);
        }
    }

    detections = std::move(filtered);
}

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

struct MrcnnRawDetection {
    float y1, x1, y2, x2, class_id, score;
};

/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomResnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* This is a sample bounding box parsing function for the tensorflow SSD models
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomTfSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseYoloV5CustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomMrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList);

extern "C"
bool NvDsInferParseCustomMrcnnTLTV2 (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList);

extern "C"
bool NvDsInferParseCustomEfficientDetTAO (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomDDETRTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomRTDETRTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomYoloV11OBB (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo  const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomResnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  static NvDsInferDimsCHW covLayerDims;
  static NvDsInferDimsCHW bboxLayerDims;
  static int bboxLayerIndex = -1;
  static int covLayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "output_bbox/BiasAdd:0") == 0) {
        bboxLayerIndex = i;
        getDimsCHWFromDims(bboxLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (bboxLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Find the cov layer */
  if (covLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "output_cov/Sigmoid:0") == 0) {
        covLayerIndex = i;
        getDimsCHWFromDims(covLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (covLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Warn in case of mismatch in number of classes */
  if (!classMismatchWarn) {
    if (covLayerDims.c != detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        covLayerDims.c << std::endl;
    }
    classMismatchWarn = true;
  }

  /* Calculate the number of classes to parse */
  numClassesToParse = MIN (covLayerDims.c, detectionParams.numClassesConfigured);

  int gridW = covLayerDims.w;
  int gridH = covLayerDims.h;
  int gridSize = gridW * gridH;
  float gcCentersX[gridW];
  float gcCentersY[gridH];
  float bboxNormX = 35.0;
  float bboxNormY = 35.0;
  float *outputCovBuf = (float *) outputLayersInfo[covLayerIndex].buffer;
  float *outputBboxBuf = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, bboxLayerDims.w);
  int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, bboxLayerDims.h);

  for (int i = 0; i < gridW; i++)
  {
    gcCentersX[i] = (float)(i * strideX + 0.5);
    gcCentersX[i] /= (float)bboxNormX;

  }
  for (int i = 0; i < gridH; i++)
  {
    gcCentersY[i] = (float)(i * strideY + 0.5);
    gcCentersY[i] /= (float)bboxNormY;

  }

  for (int c = 0; c < numClassesToParse; c++)
  {
    float *outputX1 = outputBboxBuf + (c * 4 * bboxLayerDims.h * bboxLayerDims.w);

    float *outputY1 = outputX1 + gridSize;
    float *outputX2 = outputY1 + gridSize;
    float *outputY2 = outputX2 + gridSize;

    float threshold = detectionParams.perClassPreclusterThreshold[c];
    for (int h = 0; h < gridH; h++)
    {
      for (int w = 0; w < gridW; w++)
      {
        int i = w + h * gridW;
        if (outputCovBuf[c * gridSize + i] >= threshold)
        {
          NvDsInferObjectDetectionInfo object{};
          float rectX1f, rectY1f, rectX2f, rectY2f;

          rectX1f = (outputX1[w + h * gridW] - gcCentersX[w]) * -bboxNormX;
          rectY1f = (outputY1[w + h * gridW] - gcCentersY[h]) * -bboxNormY;
          rectX2f = (outputX2[w + h * gridW] + gcCentersX[w]) * bboxNormX;
          rectY2f = (outputY2[w + h * gridW] + gcCentersY[h]) * bboxNormY;

          object.classId = c;
          object.detectionConfidence = outputCovBuf[c * gridSize + i];

          /* Clip object box co-ordinates to network resolution */
          object.left = CLIP(rectX1f, 0, networkInfo.width - 1);
          object.top = CLIP(rectY1f, 0, networkInfo.height - 1);
          object.width = CLIP(rectX2f, 0, networkInfo.width - 1) -
                             object.left + 1;
          object.height = CLIP(rectY2f, 0, networkInfo.height - 1) -
                             object.top + 1;

          objectList.push_back(object);
        }
      }
    }
  }
  return true;
}

extern "C"
bool NvDsInferParseCustomTfSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *numDetectionLayer = layerFinder("num_detections");
    const NvDsInferLayerInfo *scoreLayer = layerFinder("detection_scores");
    const NvDsInferLayerInfo *classLayer = layerFinder("detection_classes");
    const NvDsInferLayerInfo *boxLayer = layerFinder("detection_boxes");
    if (!scoreLayer || !classLayer || !boxLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    unsigned int numDetections = classLayer->inferDims.d[0];
    if (numDetectionLayer && numDetectionLayer->buffer) {
        numDetections = (int)((float*)numDetectionLayer->buffer)[0];
    }
    if (numDetections > classLayer->inferDims.d[0]) {
        numDetections = classLayer->inferDims.d[0];
    }
    numDetections = std::max<int>(0, numDetections);
    for (unsigned int i = 0; i < numDetections; ++i) {
        NvDsInferObjectDetectionInfo res{};
        res.detectionConfidence = ((float*)scoreLayer->buffer)[i];
        res.classId = ((float*)classLayer->buffer)[i];
        if (res.classId >= detectionParams.perClassPreclusterThreshold.size() ||
            res.detectionConfidence <
            detectionParams.perClassPreclusterThreshold[res.classId]) {
            continue;
        }
        enum {y1, x1, y2, x2};
        float rectX1f, rectY1f, rectX2f, rectY2f;
        rectX1f = ((float*)boxLayer->buffer)[i *4 + x1] * networkInfo.width;
        rectY1f = ((float*)boxLayer->buffer)[i *4 + y1] * networkInfo.height;
        rectX2f = ((float*)boxLayer->buffer)[i *4 + x2] * networkInfo.width;;
        rectY2f = ((float*)boxLayer->buffer)[i *4 + y2] * networkInfo.height;
        rectX1f = CLIP(rectX1f, 0.0f, networkInfo.width - 1);
        rectX2f = CLIP(rectX2f, 0.0f, networkInfo.width - 1);
        rectY1f = CLIP(rectY1f, 0.0f, networkInfo.height - 1);
        rectY2f = CLIP(rectY2f, 0.0f, networkInfo.height - 1);
        if (rectX2f <= rectX1f || rectY2f <= rectY1f) {
            continue;
        }
        res.left = rectX1f;
        res.top = rectY1f;
        res.width = rectX2f - rectX1f;
        res.height = rectY2f - rectY1f;
        if (res.width && res.height) {
            objectList.emplace_back(res);
        }
    }

    return true;
}

extern "C"
bool NvDsInferParseCustomMrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList) {
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *detectionLayer = layerFinder("generate_detections");
    const NvDsInferLayerInfo *maskLayer = layerFinder("mask_head/mask_fcn_logits/BiasAdd");

    if (!detectionLayer || !maskLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    if(maskLayer->inferDims.numDims != 4U) {
        std::cerr << "Network output number of dims is : " <<
            maskLayer->inferDims.numDims << " expect is 4"<< std::endl;
        return false;
    }

    const unsigned int det_max_instances = maskLayer->inferDims.d[0];
    const unsigned int num_classes = maskLayer->inferDims.d[1];
    if(num_classes != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:" <<
            detectionParams.numClassesConfigured << ", detected by network: " <<
            num_classes << std::endl;
    }
    const unsigned int mask_instance_height= maskLayer->inferDims.d[2];
    const unsigned int mask_instance_width = maskLayer->inferDims.d[3];

    auto out_det = reinterpret_cast<MrcnnRawDetection*>( detectionLayer->buffer);
    auto out_mask = reinterpret_cast<float(*)[mask_instance_width *
        mask_instance_height]>(maskLayer->buffer);

    for(auto i = 0U; i < det_max_instances; i++) {
        MrcnnRawDetection &rawDec = out_det[i];

        if(rawDec.score < detectionParams.perClassPreclusterThreshold[0])
            continue;

        NvDsInferInstanceMaskInfo obj;
        obj.left = CLIP(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
        if(obj.width <= 0 || obj.height <= 0)
            continue;
        obj.classId = static_cast<int>(rawDec.class_id);
        obj.detectionConfidence = rawDec.score;

        obj.mask_size = sizeof(float)*mask_instance_width*mask_instance_height;
        obj.mask = new float[mask_instance_width*mask_instance_height];
        obj.mask_width = mask_instance_width;
        obj.mask_height = mask_instance_height;

        float *rawMask = reinterpret_cast<float *>(out_mask + i
                         * detectionParams.numClassesConfigured + obj.classId);
        memcpy (obj.mask, rawMask, sizeof(float)*mask_instance_width*mask_instance_height);

        objectList.push_back(obj);
    }

    return true;

}

extern "C"
bool NvDsInferParseCustomNMSTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 2)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 2 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    // Host memory for "nms" which has 2 output bindings:
    // the order is bboxes and keep_count
    float* out_nms = (float *) outputLayersInfo[0].buffer;
    int * p_keep_count = (int *) outputLayersInfo[1].buffer;
    const float threshold = detectionParams.perClassThreshold[0];

    float* det;

    for (int i = 0; i < p_keep_count[0]; i++) {
        det = out_nms + i * 7;

        // Output format for each detection is stored in the below order
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        if ( det[2] < threshold) continue;
        assert((unsigned int) det[1] < detectionParams.numClassesConfigured);

#if 0
        std::cout << "id/label/conf/ x/y x/y -- "
                  << det[0] << " " << det[1] << " " << det[2] << " "
                  << det[3] << " " << det[4] << " " << det[5] << " " << det[6] << std::endl;
#endif
        NvDsInferObjectDetectionInfo object{};
            object.classId = (int) det[1];
            object.detectionConfidence = det[2];

            /* Clip object box co-ordinates to network resolution */
            object.left = CLIP(det[3] * networkInfo.width, 0, networkInfo.width - 1);
            object.top = CLIP(det[4] * networkInfo.height, 0, networkInfo.height - 1);
            object.width = CLIP((det[5] - det[3]) * networkInfo.width, 0, networkInfo.width - 1);
            object.height = CLIP((det[6] - det[4]) * networkInfo.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
    }

    return true;
}

extern "C"
bool NvDsInferParseYoloV5CustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout <<"keep cout"
              <<p_keep_count[0] << std::endl;
    }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object{};
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];

        object.left = CLIP(p_bboxes[4*i], 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1], 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2], 0, networkInfo.width - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3], 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {

	 if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout <<"keep cout"
              <<p_keep_count[0] << std::endl;
    }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object{};
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(p_bboxes[4*i] * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1] * networkInfo.height, 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2] * networkInfo.width, 0, networkInfo.width - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3] * networkInfo.height, 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomMrcnnTLTV2 (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList) {
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *detectionLayer = layerFinder("generate_detections");
    const NvDsInferLayerInfo *maskLayer = layerFinder("mask_fcn_logits/BiasAdd");

    if (!detectionLayer || !maskLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    if(maskLayer->inferDims.numDims != 4U) {
        std::cerr << "Network output number of dims is : " <<
            maskLayer->inferDims.numDims << " expect is 4"<< std::endl;
        return false;
    }

    const unsigned int det_max_instances = maskLayer->inferDims.d[0];
    const unsigned int num_classes = maskLayer->inferDims.d[1];
    if(num_classes != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:" <<
            detectionParams.numClassesConfigured << ", detected by network: " <<
            num_classes << std::endl;
    }
    const unsigned int mask_instance_height= maskLayer->inferDims.d[2];
    const unsigned int mask_instance_width = maskLayer->inferDims.d[3];

    auto out_det = reinterpret_cast<MrcnnRawDetection*>( detectionLayer->buffer);
    auto out_mask = reinterpret_cast<float(*)[mask_instance_width *
        mask_instance_height]>(maskLayer->buffer);

    for(auto i = 0U; i < det_max_instances; i++) {
        MrcnnRawDetection &rawDec = out_det[i];

        if(rawDec.score < detectionParams.perClassPreclusterThreshold[0])
            continue;

        NvDsInferInstanceMaskInfo obj;
        obj.left = CLIP(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
        if(obj.width <= 0 || obj.height <= 0)
            continue;
        obj.classId = static_cast<int>(rawDec.class_id);
        obj.detectionConfidence = rawDec.score;

        obj.mask_size = sizeof(float)*mask_instance_width*mask_instance_height;
        obj.mask = new float[mask_instance_width*mask_instance_height];
        obj.mask_width = mask_instance_width;
        obj.mask_height = mask_instance_height;

        float *rawMask = reinterpret_cast<float *>(out_mask + i
                         * detectionParams.numClassesConfigured + obj.classId);
        memcpy (obj.mask, rawMask, sizeof(float)*mask_instance_width*mask_instance_height);

        objectList.push_back(obj);
    }

    return true;

}

extern "C"
bool NvDsInferParseCustomEfficientDetTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int* p_keep_count = (int *) outputLayersInfo[0].buffer;

    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    NvDsInferDims inferDims_p_bboxes = outputLayersInfo[1].inferDims;
    int numElements_p_bboxes=inferDims_p_bboxes.numElements;

    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    float max_bbox=0;
    for (int i=0; i < numElements_p_bboxes; i++)
    {
        // std::cout <<"p_bboxes: "
              // <<p_bboxes[i] << std::endl;
        if ( max_bbox < p_bboxes[i] )
            max_bbox=p_bboxes[i];
    }

    if (p_keep_count[0] > 0)
    {
        assert (!(max_bbox < 2.0));
        for (int i = 0; i < p_keep_count[0]; i++) {


            if ( p_scores[i] < threshold) continue;
            assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);


            // std::cout << "label/conf/ x/y x/y -- "
                      // << (int)p_classes[i] << " " << p_scores[i] << " "
                      // << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;

            if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1])
                continue;

            NvDsInferObjectDetectionInfo object{};
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];


            object.left=p_bboxes[4*i+1];
            object.top=p_bboxes[4*i];
            object.width=( p_bboxes[4*i+3] - object.left);
            object.height= ( p_bboxes[4*i+2] - object.top);

            object.left=CLIP(object.left, 0, networkInfo.width - 1);
            object.top=CLIP(object.top, 0, networkInfo.height - 1);
            object.width=CLIP(object.width, 0, networkInfo.width - 1);
            object.height=CLIP(object.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
        }
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomDDETRTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *boxLayer = layerFinder("pred_boxes"); // 1 x num_queries x 4
    const NvDsInferLayerInfo *classLayer = layerFinder("pred_logits");  // 1 x num_queries x num_classes

    if (!boxLayer || !classLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    const int keep_top_k = 200;
    unsigned int numDetections = classLayer->inferDims.d[0];
    unsigned int numClasses = classLayer->inferDims.d[1];
    std::map<float, NvDsInferObjectDetectionInfo> ordered_objects;

    for (unsigned int idx = 0; idx < numDetections*4; idx += 4) {
        NvDsInferObjectDetectionInfo res{};

        res.classId = std::max_element(((float*)classLayer->buffer+idx), ((float*)classLayer->buffer+idx+numClasses)) - ((float*)classLayer->buffer+idx);
        res.detectionConfidence = ((float*)classLayer->buffer)[idx+res.classId];

        // If model does not have sigmoid layer, perform sigmoid calculation here
        res.detectionConfidence = 1.0/(1.0 + exp(-res.detectionConfidence));

        if(res.classId == 0 || res.detectionConfidence < detectionParams.perClassPreclusterThreshold[res.classId]) {
            continue;
        }
        enum {cx, cy, w, h};
        float rectX1f, rectY1f, rectX2f, rectY2f;

        rectX1f = (((float*)boxLayer->buffer)[idx + cx] - (((float*)boxLayer->buffer)[idx + w]/2)) * networkInfo.width;
        rectY1f = (((float*)boxLayer->buffer)[idx + cy] - (((float*)boxLayer->buffer)[idx + h]/2)) * networkInfo.height;
        rectX2f = rectX1f + ((float*)boxLayer->buffer)[idx + w] * networkInfo.width;
        rectY2f = rectY1f + ((float*)boxLayer->buffer)[idx + h] * networkInfo.height;

        rectX1f = CLIP(rectX1f, 0.0f, networkInfo.width - 1);
        rectX2f = CLIP(rectX2f, 0.0f, networkInfo.width - 1);
        rectY1f = CLIP(rectY1f, 0.0f, networkInfo.height - 1);
        rectY2f = CLIP(rectY2f, 0.0f, networkInfo.height - 1);

        res.left = rectX1f;
        res.top = rectY1f;
        res.width = rectX2f - rectX1f;
        res.height = rectY2f - rectY1f;

        ordered_objects[res.detectionConfidence] = res;
    }

    // Use objects within top_k range
    int jdx = 0;
    for (std::map<float, NvDsInferObjectDetectionInfo>::iterator iter=ordered_objects.end();
        iter!=ordered_objects.begin() && jdx<keep_top_k; iter--, jdx++) {
        if (iter->second.classId != 0)
            objectList.emplace_back(iter->second);
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomRTDETRTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    // Layer finder - C++11 compatible
    const NvDsInferLayerInfo *boxLayer = NULL;
    const NvDsInferLayerInfo *classLayer = NULL;
    
    for (size_t i = 0; i < outputLayersInfo.size(); i++) {
        if (outputLayersInfo[i].dataType == FLOAT &&
            (outputLayersInfo[i].layerName && std::string("pred_boxes") == outputLayersInfo[i].layerName)) {
            boxLayer = &outputLayersInfo[i];
        }
        if (outputLayersInfo[i].dataType == FLOAT &&
            (outputLayersInfo[i].layerName && std::string("pred_logits") == outputLayersInfo[i].layerName)) {
            classLayer = &outputLayersInfo[i];
        }
    }

    if (!boxLayer || !classLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }
    


    // Get dimensions - handle flattened format
    const int total_queries = classLayer->inferDims.d[0];  // batch_size * num_queries
    const int num_classes = classLayer->inferDims.d[1];
    const int num_select = 100; // Default top-k selection
    


    // Cast buffers
    float* pred_logits = (float*)classLayer->buffer;
    float* pred_boxes = (float*)boxLayer->buffer;

    // Process flattened format - all queries from all batches together
    int total_detections_found = 0;
    
    // Apply sigmoid to all logits
    std::vector<float> prob(total_queries * num_classes);
    for (int i = 0; i < total_queries * num_classes; i++) {
        prob[i] = 1.0f / (1.0f + exp(-pred_logits[i]));
    }

    // Get top-k indices (C++11 compatible)
    std::vector<std::pair<float, int> > score_index_pairs;
    for (int i = 0; i < total_queries * num_classes; i++) {
        score_index_pairs.push_back(std::make_pair(prob[i], i));
    }
        
    // Sort by score (descending) and take top-k - C++11 compatible
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { 
                  return a.first > b.first; 
              });

    // Process top-k predictions
    for (int k = 0; k < std::min(num_select, (int)score_index_pairs.size()); k++) {
        float score = score_index_pairs[k].first;
        int flat_index = score_index_pairs[k].second;
        
        // Extract query index and class index
        int query_idx = flat_index / num_classes;
        int class_id = flat_index % num_classes;
        
        // Check confidence threshold
        if (score < detectionParams.perClassPreclusterThreshold[class_id]) {
            continue;
        }

        // Get corresponding box - flattened format
        int box_offset = query_idx * 4;
        float cx = pred_boxes[box_offset + 0];
        float cy = pred_boxes[box_offset + 1];
        float w = pred_boxes[box_offset + 2];
        float h = pred_boxes[box_offset + 3];

        // Convert to x1, y1, x2, y2 format
        float x1 = (cx - 0.5f * w) * networkInfo.width;
        float y1 = (cy - 0.5f * h) * networkInfo.height;
        float x2 = (cx + 0.5f * w) * networkInfo.width;
        float y2 = (cy + 0.5f * h) * networkInfo.height;

        // Clamp coordinates
        x1 = CLIP(x1, 0.0f, networkInfo.width - 1);
        y1 = CLIP(y1, 0.0f, networkInfo.height - 1);
        x2 = CLIP(x2, 0.0f, networkInfo.width - 1);
        y2 = CLIP(y2, 0.0f, networkInfo.height - 1);

        // Create detection object
        NvDsInferObjectDetectionInfo detection{};
        detection.classId = class_id;
        detection.detectionConfidence = score;
        detection.left = x1;
        detection.top = y1;
        detection.width = x2 - x1;
        detection.height = y2 - y1;

        objectList.push_back(detection);
        total_detections_found++;
    }

    return true;
}

extern "C"
bool NvDsInferParseCustomYoloV11OBB (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.size() != 1) {
        std::cerr << "YOLO11 OBB expects 1 output layer, got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }
    const NvDsInferLayerInfo &outputLayer = outputLayersInfo[0];
    const NvDsInferDims &outputDims = outputLayer.inferDims;

    // YOLO11 OBB output format: [num_channels, num_anchors]
    // For YOLO11-OBB: [20, 21504] where 20 = 4 bbox + 15 classes + 1 angle
    if (outputDims.numDims == 2) {
        int numChannels = outputDims.d[0];  // Should be 20 for OBB (4+num_classes+1)
        int numAnchors = outputDims.d[1];   // Should be 21504 (sum of all grid cells)

        if (numChannels < 5) {
            std::cerr << "Invalid YOLO11 OBB output dimensions. Expected at least 5 channels, got " 
                      << numChannels << std::endl;
            return false;
        }

        float *outputBuffer = (float *)outputLayer.buffer;
        int numClasses = numChannels - 5;  // 4 bbox + num_classes + 1 angle

        for (int a = 0; a < numAnchors; a++) {
            // Read bbox coordinates (channels 0-3)
            // YOLO11 outputs are already in pixel coordinates relative to input size (640x640)
            float center_x = outputBuffer[0 * numAnchors + a];
            float center_y = outputBuffer[1 * numAnchors + a];
            float width = outputBuffer[2 * numAnchors + a];
            float height = outputBuffer[3 * numAnchors + a];

            // Find class with highest score (channels 4 to 4+numClasses-1)
            float max_class_score = -1.0f;
            int class_id = 0;
            for (int c = 0; c < numClasses; c++) {
                float class_score = outputBuffer[(4 + c) * numAnchors + a];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    class_id = c;
                }
            }

            // Apply confidence threshold
            if (max_class_score < detectionParams.perClassPreclusterThreshold[0]) {
                continue;
            }

            // Read rotation angle (last channel)
            float angle = outputBuffer[(numChannels - 1) * numAnchors + a];

            // Normalize angle to positive range [0, π]
            // YOLO outputs angles in radians, typically in range [-π/2, π/2] or [-π, π]
            while (angle < 0) {
                angle += M_PI;  // Add 180° to make negative angles positive
            }
            while (angle > M_PI) {
                angle -= M_PI;  // Keep in 0-180° range (OBB are symmetric)
            }

            // Clip to preprocessed bounds
            float preprocessed_size = (float)networkInfo.width;
            center_x = std::max(0.0f, std::min(center_x, preprocessed_size - 1));
            center_y = std::max(0.0f, std::min(center_y, preprocessed_size - 1));
            width = std::max(1.0f, std::min(width, preprocessed_size));
            height = std::max(1.0f, std::min(height, preprocessed_size));

            // Convert center coordinates to top-left corner
            // OSD expects left/top as corner, then calculates center as (left + width/2, top + height/2)
            float left = center_x - width / 2.0f;
            float top = center_y - height / 2.0f;

            if (width <= 0 || height <= 0) continue;

            // Convert to DeepStream format
            NvDsInferObjectDetectionInfo detection = {};  // Zero-initialize all fields
            detection.left = left;
            detection.top = top;
            detection.width = width;
            detection.height = height;
            detection.detectionConfidence = max_class_score;
            detection.classId = class_id;
            detection.rotation_angle = angle;  // Set OBB rotation angle in radians
            objectList.push_back(detection);
        }

        // Perform NMS on all detections
        //Modify this value if needed for your use case. Standard NMS threshold for YOLO models
        float nms_threshold = 0.45f;
        performNMS(objectList, nms_threshold);

        return true;
    } else if (outputDims.numDims == 3) {
        // 3D format: [batch, num_detections, 6]
        int batchSize = outputDims.d[0];
        int numDetections = outputDims.d[1];
        int numValues = outputDims.d[2];

        if (numValues == 0) {
            numValues = 6; // Assume 6 values per detection
        }

        if (numValues < 5) {
            std::cerr << "Invalid YOLO11 OBB output dimensions. Expected at least 5 values per detection, got "
                      << numValues << std::endl;
            return false;
        }

        float *outputBuffer = (float *)outputLayer.buffer;

        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < numDetections; d++) {
                int baseIdx = b * numDetections * numValues + d * numValues;

                float center_x = outputBuffer[baseIdx + 0];
                float center_y = outputBuffer[baseIdx + 1];
                float width = outputBuffer[baseIdx + 2];
                float height = outputBuffer[baseIdx + 3];
                float confidence = outputBuffer[baseIdx + 4];
                float angle = 0.0f;
                int class_id = 0;

                // YOLO11 OBB format: [x, y, w, h, confidence, angle, class_id] (7 values)
                if (numValues >= 6) {
                    angle = outputBuffer[baseIdx + 5];  // angle is at index 5
                    if (numValues >= 7) {
                        class_id = (int)outputBuffer[baseIdx + 6];  // class_id is at index 6
                    } else {
                        class_id = (int)outputBuffer[baseIdx + 5];  // if only 6 values, class_id is at index 5
                        angle = 0.0f;  // no angle in this case
                    }
                }

                // Normalize angle to positive range [0, π]
                while (angle < 0) {
                    angle += M_PI;
                }
                while (angle > M_PI) {
                    angle -= M_PI;
                }

                // Apply confidence threshold
                if (confidence < detectionParams.perClassPreclusterThreshold[0]) {
                    continue;
                }

                // "YOLOv11 outputs boxes in pixel coordinates relative to input size (640x640)"
                //
                // Model training size: 640x640
                // ONNX input size: 1024x1024
                // Model outputs: coordinates in 640x640 space
                //
                // Need to scale from 640x640 model space to 1024x1024 preprocessed space:
                float model_training_size = 640.0f;
                float preprocessed_size = (float)networkInfo.width;
                float scale_factor = preprocessed_size / model_training_size;

                float img_center_x = center_x * scale_factor;
                float img_center_y = center_y * scale_factor;
                float img_width = width * scale_factor;
                float img_height = height * scale_factor;

                // Clip to preprocessed bounds
                img_center_x = std::max(0.0f, std::min(img_center_x, preprocessed_size - 1));
                img_center_y = std::max(0.0f, std::min(img_center_y, preprocessed_size - 1));
                img_width = std::max(1.0f, std::min(img_width, preprocessed_size));
                img_height = std::max(1.0f, std::min(img_height, preprocessed_size));

                // Convert center coordinates to top-left corner
                // OSD expects left/top as corner, then calculates center as (left + width/2, top + height/2)
                float left = img_center_x - img_width / 2.0f;
                float top = img_center_y - img_height / 2.0f;

                if (img_width <= 0 || img_height <= 0) continue;

                // Convert to DeepStream format
                NvDsInferObjectDetectionInfo detection = {};  // Zero-initialize all fields
                detection.left = left;
                detection.top = top;
                detection.width = img_width;
                detection.height = img_height;
                detection.detectionConfidence = confidence;
                detection.classId = class_id;
                detection.rotation_angle = angle;  // Store angle in radians

                objectList.push_back(detection);
            }
        }

        return true;
    } else {
        std::cerr << "Invalid YOLO11 OBB output dimensions. Expected 2D or 3D tensor, got "
                  << outputDims.numDims << "D tensor" << std::endl;
        return false;
    }
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomResnet);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomTfSSD);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomNMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV5CustomBatchedNMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBatchedNMSTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnTLTV2);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEfficientDetTAO);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDDETRTAO);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRTDETRTAO);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV11OBB);
