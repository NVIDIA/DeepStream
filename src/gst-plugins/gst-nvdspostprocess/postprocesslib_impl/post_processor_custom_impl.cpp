/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <cstring>
#include <queue>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <condition_variable>
#include <yaml-cpp/yaml.h>
#include <limits.h>
#include <cassert>
#include <algorithm>
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"
#include "nvdsinfer_dbscan.h"

#include "post_processor_custom_impl.h"
#include <cassert>
#include <cmath>

const int nmsMaxOut = 300;
#define MIN1(a,b) ((a) < (b) ? (a) : (b))
#define MAX1(a,b) ((a) > (b) ? (a) : (b))
#define CLIP1(a,min,max) (MAX1(MIN1(a, max), min))
#define DIVIDE_AND_ROUND_UP1(a, b) ((a + b - 1) / b)
#define DIVUP(n, d) ((n) + (d)-1) / (d)


static float clamp(const float val, const float minVal, const float maxVal);
static float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}


struct MrcnnRawDetection {
    float y1, x1, y2, x2, class_id, score;
};

/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsPostProcessParseCustomResnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

/* This is a sample bounding box parsing function for the tensorflow SSD models
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsPostProcessParseCustomTfSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

extern "C"
bool NvDsPostProcessParseCustomNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsPostProcessParseDetectionParams const &detectionParams,
         std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

extern "C"
bool NvDsPostProcessParseCustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsPostProcessParseDetectionParams const &detectionParams,
         std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

extern "C"
bool NvDsPostProcessParseCustomMrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsPostProcessParseDetectionParams const &detectionParams,
                                   std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);

extern "C"
bool NvDsPostProcessParseCustomMrcnnTLTV2 (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsPostProcessParseDetectionParams const &detectionParams,
                                   std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);
extern "C"
bool NvDsPostProcessParseCustomFasterRCNN (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

extern "C"
bool NvDsPostProcessParseCustomResnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList)
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
      if (strcmp(outputLayersInfo[i].layerName, "output_bbox/BiasAdd") == 0) {
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
      if (strcmp(outputLayersInfo[i].layerName, "output_cov/Sigmoid") == 0) {
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
  numClassesToParse = MIN1 (covLayerDims.c, detectionParams.numClassesConfigured);

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
          NvDsPostProcessObjectDetectionInfo object;
          float rectX1f, rectY1f, rectX2f, rectY2f;

          rectX1f = (outputX1[w + h * gridW] - gcCentersX[w]) * -bboxNormX;
          rectY1f = (outputY1[w + h * gridW] - gcCentersY[h]) * -bboxNormY;
          rectX2f = (outputX2[w + h * gridW] + gcCentersX[w]) * bboxNormX;
          rectY2f = (outputY2[w + h * gridW] + gcCentersY[h]) * bboxNormY;

          object.classId = c;
          object.detectionConfidence = outputCovBuf[c * gridSize + i];

          /* Clip object box co-ordinates to network resolution */
          object.left = CLIP1(rectX1f, 0, networkInfo.width - 1);
          object.top = CLIP1(rectY1f, 0, networkInfo.height - 1);
          object.width = CLIP1(rectX2f, 0, networkInfo.width - 1) -
                             object.left + 1;
          object.height = CLIP1(rectY2f, 0, networkInfo.height - 1) -
                             object.top + 1;

          objectList.push_back(object);
        }
      }
    }
  }
  return true;
}

extern "C"
bool NvDsPostProcessParseCustomTfSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    NvDsPostProcessParseDetectionParams const &detectionParams,
    std::vector<NvDsPostProcessObjectDetectionInfo> &objectList)
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
        NvDsPostProcessObjectDetectionInfo res;
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
        rectX1f = CLIP1(rectX1f, 0.0f, networkInfo.width - 1);
        rectX2f = CLIP1(rectX2f, 0.0f, networkInfo.width - 1);
        rectY1f = CLIP1(rectY1f, 0.0f, networkInfo.height - 1);
        rectY2f = CLIP1(rectY2f, 0.0f, networkInfo.height - 1);
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
bool NvDsPostProcessParseCustomMrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsPostProcessParseDetectionParams const &detectionParams,
                                   std::vector<NvDsPostProcessInstanceMaskInfo> &objectList) {
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

        NvDsPostProcessInstanceMaskInfo obj;
        obj.left = CLIP1(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP1(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP1(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP1(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
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
bool NvDsPostProcessParseCustomNMSTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsPostProcessParseDetectionParams const &detectionParams,
                                   std::vector<NvDsPostProcessObjectDetectionInfo> &objectList) {
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
    const float threshold = detectionParams.perClassPreclusterThreshold[0];

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
        NvDsPostProcessObjectDetectionInfo object;
            object.classId = (int) det[1];
            object.detectionConfidence = det[2];

            /* Clip object box co-ordinates to network resolution */
            object.left = CLIP1(det[3] * networkInfo.width, 0, networkInfo.width - 1);
            object.top = CLIP1(det[4] * networkInfo.height, 0, networkInfo.height - 1);
            object.width = CLIP1((det[5] - det[3]) * networkInfo.width, 0, networkInfo.width - 1);
            object.height = CLIP1((det[6] - det[4]) * networkInfo.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
    }

    return true;
}

extern "C"
bool NvDsPostProcessParseCustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsPostProcessParseDetectionParams const &detectionParams,
         std::vector<NvDsPostProcessObjectDetectionInfo> &objectList) {

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

    const float threshold = detectionParams.perClassPreclusterThreshold[0];

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

        NvDsPostProcessObjectDetectionInfo object;
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP1(p_bboxes[4*i] * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP1(p_bboxes[4*i+1] * networkInfo.height, 0, networkInfo.height - 1);
        object.width = CLIP1(p_bboxes[4*i+2] * networkInfo.width, 0, networkInfo.width - 1) - object.left;
        object.height = CLIP1(p_bboxes[4*i+3] * networkInfo.height, 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}

extern "C"
bool NvDsPostProcessParseCustomMrcnnTLTV2 (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsPostProcessParseDetectionParams const &detectionParams,
                                   std::vector<NvDsPostProcessInstanceMaskInfo> &objectList) {
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

        NvDsPostProcessInstanceMaskInfo obj;
        obj.left = CLIP1(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP1(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP1(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP1(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
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

/* This is a sample classifier output parsing function from softmax layers for
 * the vehicle type classifier model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsPostProcessClassiferParseCustomSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsPostProcessAttribute> &attrList,
        std::string &descString);

static std::vector < std::vector< std:: string > > labels { {
    "coupe1", "largevehicle1", "sedan1", "suv1", "truck1", "van1"} };

extern "C"
bool NvDsPostProcessClassiferParseCustomSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsPostProcessAttribute> &attrList,
        std::string &descString)
{
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = outputLayersInfo.size();

    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numAttributes; l++)
    {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, outputLayersInfo[l].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer = (float *)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsPostProcessAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > classifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }
        }
        if (attrFound)
        {
            if (labels.size() > attr.attributeIndex &&
                    attr.attributeValue < labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    strdup(labels[attr.attributeIndex][attr.attributeValue].c_str());
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                descString.append(attr.attributeLabel).append(" ");
        }
    }

    return true;
}


/* This is a sample bounding box parsing function for the sample FasterRCNN
 * detector model provided with the TensorRT samples. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsPostProcessParseCustomFasterRCNN (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList)
{
  static int bboxPredLayerIndex = -1;
  static int clsProbLayerIndex = -1;
  static int roisLayerIndex = -1;
  static const int NUM_CLASSES_FASTER_RCNN = 21;
  static bool classMismatchWarn = false;
  int numClassesToParse;

  if (bboxPredLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "bbox_pred") == 0) {
        bboxPredLayerIndex = i;
        break;
      }
    }
    if (bboxPredLayerIndex == -1) {
    std::cerr << "Could not find bbox_pred layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (clsProbLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "cls_prob") == 0) {
        clsProbLayerIndex = i;
        break;
      }
    }
    if (clsProbLayerIndex == -1) {
    std::cerr << "Could not find cls_prob layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (roisLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "rois") == 0) {
        roisLayerIndex = i;
        break;
      }
    }
    if (roisLayerIndex == -1) {
    std::cerr << "Could not find rois layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (!classMismatchWarn) {
    if (NUM_CLASSES_FASTER_RCNN !=
        detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        NUM_CLASSES_FASTER_RCNN << std::endl;
    }
    classMismatchWarn = true;
  }

  numClassesToParse = MIN (NUM_CLASSES_FASTER_RCNN,
      detectionParams.numClassesConfigured);

  float *rois = (float *) outputLayersInfo[roisLayerIndex].buffer;
  float *deltas = (float *) outputLayersInfo[bboxPredLayerIndex].buffer;
  float *scores = (float *) outputLayersInfo[clsProbLayerIndex].buffer;

  for (int i = 0; i < nmsMaxOut; ++i)
  {
    float width = rois[i * 4 + 2] - rois[i * 4] + 1;
    float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
    float ctr_x = rois[i * 4] + 0.5f * width;
    float ctr_y = rois[i * 4 + 1] + 0.5f * height;
    float *deltas_offset = deltas + i * NUM_CLASSES_FASTER_RCNN * 4;
    for (int j = 0; j < numClassesToParse; ++j)
    {
      float confidence = scores[i * NUM_CLASSES_FASTER_RCNN + j];
      if (confidence < detectionParams.perClassPreclusterThreshold[j])
        continue;
      NvDsPostProcessObjectDetectionInfo object;

      float dx = deltas_offset[j * 4];
      float dy = deltas_offset[j * 4 + 1];
      float dw = deltas_offset[j * 4 + 2];
      float dh = deltas_offset[j * 4 + 3];
      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = exp(dw) * width;
      float pred_h = exp(dh) * height;
      float rectx1 = MIN (pred_ctr_x - 0.5f * pred_w, networkInfo.width - 1.f);
      float recty1 = MIN (pred_ctr_y - 0.5f * pred_h, networkInfo.height - 1.f);
      float rectx2 = MIN (pred_ctr_x + 0.5f * pred_w, networkInfo.width - 1.f);
      float recty2 = MIN (pred_ctr_y + 0.5f * pred_h, networkInfo.height - 1.f);


      object.classId = j;
      object.detectionConfidence = confidence;

      /* Clip object box co-ordinates to network resolution */
      object.left = CLIP1(rectx1, 0, networkInfo.width - 1);
      object.top = CLIP1(recty1, 0, networkInfo.height - 1);
      object.width = CLIP1(rectx2, 0, networkInfo.width - 1) - object.left + 1;
      object.height = CLIP1(recty2, 0, networkInfo.height - 1) - object.top + 1;

      objectList.push_back(object);
    }
  }
  return true;
}

extern "C"
bool NvDsPostProcessParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsPostProcessParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList)
{
  static int nmsLayerIndex = -1;
  static int nms1LayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;
  static const int NUM_CLASSES_SSD = 91;

  if (nmsLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "NMS") == 0) {
        nmsLayerIndex = i;
        break;
      }
    }
    if (nmsLayerIndex == -1) {
    std::cerr << "Could not find NMS layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (nms1LayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "NMS_1") == 0) {
        nms1LayerIndex = i;
        break;
      }
    }
    if (nms1LayerIndex == -1) {
    std::cerr << "Could not find NMS_1 layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (!classMismatchWarn) {
    if (NUM_CLASSES_SSD !=
        detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        NUM_CLASSES_SSD << std::endl;
    }
    classMismatchWarn = true;
  }

  numClassesToParse = MIN (NUM_CLASSES_SSD,
      detectionParams.numClassesConfigured);

  int keepCount = *((int *) outputLayersInfo[nms1LayerIndex].buffer);
  float *detectionOut = (float *) outputLayersInfo[nmsLayerIndex].buffer;

  for (int i = 0; i < keepCount; ++i)
  {
    float* det = detectionOut + i * 7;
    int classId = det[1];

    if (classId >= numClassesToParse)
      continue;

    float threshold = detectionParams.perClassPreclusterThreshold[classId];

    if (det[2] < threshold)
      continue;

    unsigned int rectx1, recty1, rectx2, recty2;
    NvDsPostProcessObjectDetectionInfo object;

    rectx1 = det[3] * networkInfo.width;
    recty1 = det[4] * networkInfo.height;
    rectx2 = det[5] * networkInfo.width;
    recty2 = det[6] * networkInfo.height;

    object.classId = classId;
    object.detectionConfidence = det[2];

    /* Clip object box co-ordinates to network resolution */
    object.left = CLIP1(rectx1, 0, networkInfo.width - 1);
    object.top = CLIP1(recty1, 0, networkInfo.height - 1);
    object.width = CLIP1(rectx2, 0, networkInfo.width - 1) -
      object.left + 1;
    object.height = CLIP1(recty2, 0, networkInfo.height - 1) -
      object.top + 1;

    objectList.push_back(object);
  }

  return true;
}


static const int NUM_CLASSES_YOLO = 80;

extern "C" bool NvDsPostProcessParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList);

extern "C" bool NvDsPostProcessParseCustomYoloV3Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList);

extern "C" bool NvDsPostProcessParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList);

extern "C" bool NvDsPostProcessParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList);

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsPostProcessParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsPostProcessParseObjectInfo b = {
        .classId = UINT32_MAX,
        .left = 0.0,
        .top = 0.0,
        .width = 0.0,
        .height = 0.0,
        .detectionConfidence = 0.0
    };
    // Restore coordinates to network input resolution
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsPostProcessParseObjectInfo>& binfo)
{
    NvDsPostProcessParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsPostProcessParseObjectInfo>
decodeYoloV2Tensor(
    const float* detections, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsPostProcessParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[b * 2];
                const float ph = anchors[b * 2 + 1];

                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)]);
                const float bh
                    = ph * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)]);

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static std::vector<NvDsPostProcessParseObjectInfo>
decodeYoloV3Tensor(
    const float* detections, const std::vector<int> &mask, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsPostProcessParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];

                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static inline std::vector<const NvDsInferLayerInfo*>
SortLayers(const std::vector<NvDsInferLayerInfo> & outputLayersInfo)
{
    std::vector<const NvDsInferLayerInfo*> outLayers;
    for (auto const &layer : outputLayersInfo) {
        outLayers.push_back (&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
        [](const NvDsInferLayerInfo* a, const NvDsInferLayerInfo* b) {
            return a->inferDims.d[1] < b->inferDims.d[1];
        });
    return outLayers;
}

static bool NvDsPostProcessParseYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList,
    const std::vector<float> &anchors,
    const std::vector<std::vector<int>> &masks)
{
    const uint kNUM_BBOXES = 3;

    const std::vector<const NvDsInferLayerInfo*> sortedLayers =
        SortLayers (outputLayersInfo);

    if (sortedLayers.size() != masks.size()) {
        std::cerr << "ERROR: yoloV3 output layer.size: " << sortedLayers.size()
                  << " does not match mask.size: " << masks.size() << std::endl;
        return false;
    }

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsPostProcessParseObjectInfo> objects;

    for (uint idx = 0; idx < masks.size(); ++idx) {
        const NvDsInferLayerInfo &layer = *sortedLayers[idx]; // 255 x Grid x Grid

        assert(layer.inferDims.numDims == 3);
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];
        const uint stride = DIVUP(networkInfo.width, gridSizeW);
        assert(stride == DIVUP(networkInfo.height, gridSizeH));

        std::vector<NvDsPostProcessParseObjectInfo> outObjs =
            decodeYoloV3Tensor((const float*)(layer.buffer), masks[idx], anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                       NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);
        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }


    objectList = objects;

    return true;
}


/* C-linkage to prevent name-mangling */
extern "C" bool NvDsPostProcessParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList)
{
    static const std::vector<float> kANCHORS = {
        10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
        45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    static const std::vector<std::vector<int>> kMASKS = {
        {6, 7, 8},
        {3, 4, 5},
        {0, 1, 2}};
    return NvDsPostProcessParseYoloV3 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}

extern "C" bool NvDsPostProcessParseCustomYoloV3Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList)
{
    static const std::vector<float> kANCHORS = {
        10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    static const std::vector<std::vector<int>> kMASKS = {
        {3, 4, 5},
        //{0, 1, 2}}; // as per output result, select {1,2,3}
        {1, 2, 3}};

    return NvDsPostProcessParseYoloV3 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}

static bool NvDsPostProcessParseYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList)
{
    // copy anchor data from yolov2.cfg file
    std::vector<float> anchors = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
        5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
    const uint kNUM_BBOXES = 5;

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    assert(layer.inferDims.numDims == 3);
    const uint gridSizeH = layer.inferDims.d[1];
    const uint gridSizeW = layer.inferDims.d[2];
    const uint stride = DIVUP(networkInfo.width, gridSizeW);
    assert(stride == DIVUP(networkInfo.height, gridSizeH));
    for (auto& anchor : anchors) {
        anchor *= stride;
    }
    std::vector<NvDsPostProcessParseObjectInfo> objects =
        decodeYoloV2Tensor((const float*)(layer.buffer), anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                   NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);

    objectList = objects;

    return true;
}

extern "C" bool NvDsPostProcessParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList)
{
    return NvDsPostProcessParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsPostProcessParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    std::vector<NvDsPostProcessParseObjectInfo>& objectList)
{
    return NvDsPostProcessParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomYoloV3);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomYoloV3Tiny);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomYoloV2);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomYoloV2Tiny);

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomSSD);
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomFasterRCNN);

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsPostProcessClassiferParseCustomSoftmax);
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomResnet);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomTfSSD);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomNMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomBatchedNMSTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomMrcnnTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsPostProcessParseCustomMrcnnTLTV2);

