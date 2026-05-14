/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: MIT
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

#include "nvdsinfer_custom_impl.h"
#include "nvtx3/nvToolsExt.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>

static const int NUM_CLASSES_YOLO = 80;
#define OBJECTLISTSIZEV7 25200
#define OBJECTLISTSIZEV8 8400
#define OBJECTLISTSIZEV11 9261

#define BLOCKSIZE 1024
thrust::device_vector<NvDsInferParseObjectInfo> objects_v7(OBJECTLISTSIZEV7);
thrust::device_vector<NvDsInferParseObjectInfo> objects_v8(OBJECTLISTSIZEV8);
thrust::device_vector<NvDsInferParseObjectInfo> objects_v11(OBJECTLISTSIZEV11);

template <bool isYoloV7>
__global__ void decodeYoloTensor_cuda(NvDsInferParseObjectInfo *binfo,
                                      float *data, int dimensions, int rows,
                                      int netW, int netH, float Threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows) {
    data = data + idx * dimensions;

    float maxProb = isYoloV7 ? data[4] : 1.0f;
    if (isYoloV7 && maxProb < Threshold) {
      binfo[idx].detectionConfidence = 0.0;
      return;
    }

    float bx = data[0];
    float by = data[1];
    float bw = data[2];
    float bh = data[3];

    float *classes_scores = (float *)(data + (isYoloV7 ? 5 : 4));
    float maxScore = 0;
    int maxIndex = 0;

#pragma unroll
    for (int j = 0; j < NUM_CLASSES_YOLO; j++) {
      if (*classes_scores > maxScore) {
        maxIndex = j;
        maxScore = *classes_scores;
      }
      classes_scores++;
    }

    float finalScore = isYoloV7 ? (maxProb * maxScore) : maxScore;
    if (finalScore < Threshold) {
      binfo[idx].detectionConfidence = 0.0;
      return;
    }

    float xCenter = bx;
    float yCenter = by;
    float x0 = xCenter - bw / 2.0f;
    float y0 = yCenter - bh / 2.0f;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    binfo[idx].left = fminf(float(netW), fmaxf(0.0f, x0));
    binfo[idx].top = fminf(float(netH), fmaxf(0.0f, y0));
    binfo[idx].width = fminf(float(netW), fmaxf(0.0f, x1 - x0));
    binfo[idx].height = fminf(float(netH), fmaxf(0.0f, y1 - y0));
    binfo[idx].detectionConfidence = finalScore;
    binfo[idx].classId = maxIndex;
  }
}

template <bool isYoloV7, int OBJECTLISTSIZE>
static bool NvDsInferParseYolo_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList,
    thrust::device_vector<NvDsInferParseObjectInfo> &objects) {

  if (outputLayersInfo.empty()) {
    std::cerr << "Could not find output layer in bbox parsing" << std::endl;
    return false;
  }
  const NvDsInferLayerInfo &layer = outputLayersInfo[0];

  if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Num classes mismatch. Configured:"
              << detectionParams.numClassesConfigured
              << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
  }

  float *data = (float *)layer.buffer;
  const int dimensions = layer.inferDims.d[1];
  int rows = layer.inferDims.numElements / layer.inferDims.d[1];

  int GRIDSIZE = ((OBJECTLISTSIZE - 1) / BLOCKSIZE) + 1;
  float min_PreclusterThreshold =
      *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
                         detectionParams.perClassPreclusterThreshold.end()));

  decodeYoloTensor_cuda<isYoloV7><<<GRIDSIZE, BLOCKSIZE>>>(
      thrust::raw_pointer_cast(objects.data()), data, dimensions, rows,
      networkInfo.width, networkInfo.height, min_PreclusterThreshold);

  objectList.resize(OBJECTLISTSIZE);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

extern "C" {

bool NvDsInferParseCustomYoloV7_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList) {
  nvtxRangePush("NvDsInferParseYoloV7");
  bool ret = NvDsInferParseYolo_cuda<true, OBJECTLISTSIZEV7>(
      outputLayersInfo, networkInfo, detectionParams, objectList, objects_v7);

  nvtxRangePop();
  return ret;
}

bool NvDsInferParseCustomYoloV8_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList) {
  nvtxRangePush("NvDsInferParseYoloV8");
  bool ret = NvDsInferParseYolo_cuda<false, OBJECTLISTSIZEV8>(
      outputLayersInfo, networkInfo, detectionParams, objectList, objects_v8);
  nvtxRangePop();

  return ret;
}

bool NvDsInferParseCustomYoloV11_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList) {
  nvtxRangePush("NvDsInferParseYoloV11");
  bool ret = NvDsInferParseYolo_cuda<false, OBJECTLISTSIZEV11>(
      outputLayersInfo, networkInfo, detectionParams, objectList, objects_v11);
  nvtxRangePop();
  return ret;
}
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV7_cuda);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8_cuda);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV11_cuda);
