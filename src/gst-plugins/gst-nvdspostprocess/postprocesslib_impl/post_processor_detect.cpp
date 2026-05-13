/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "post_processor_detect.h"

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
bool NvDsPostProcessParseCustomFasterRCNN (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsPostProcessParseDetectionParams const &detectionParams,
         std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

extern "C"
bool NvDsPostProcessParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);


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

using namespace std;

/**
 * Attach metadata for the detector. We will be adding a new metadata.
 */
void
DetectModelPostProcessor::attachMetadata (NvBufSurface *surf, gint batch_idx,
    NvDsBatchMeta  *batch_meta,
    NvDsFrameMeta  *frame_meta,
    NvDsObjectMeta  *object_meta,
    NvDsObjectMeta *parent_obj_meta,
    NvDsPostProcessFrameOutput & detection_output,
    NvDsPostProcessDetectionParams *all_params,
    std::set <gint> & filterOutClassIds,
    int32_t unique_id,
    gboolean output_instance_mask,
    gboolean process_full_frame,
    float segmentationThreshold,
    gboolean maintain_aspect_ratio,
    NvDsRoiMeta *roi_meta,
    gboolean symmetric_padding) //FIXME: handle it for detectors
{
  static gchar font_name[] = "Serif";
  NvDsObjectMeta *obj_meta = NULL;
  nvds_acquire_meta_lock (batch_meta);
  gint surf_width  = surf->surfaceList[batch_idx].width;
  gint surf_height = surf->surfaceList[batch_idx].height;
  float scale_x = (float)m_NetworkInfo.width/(float)surf_width;
  float scale_y = (float)m_NetworkInfo.height/(float)surf_height;
  float offset_left = 0;
  float offset_top = 0;
  float roi_left = 0;
  float roi_top = 0;
  float roi_height = 0;
  float roi_width = 0;

  //FIXME: Get preprocess data and scale ROI

  frame_meta->bInferDone = TRUE;
  /* Iterate through the inference output for one frame and attach the detected
   * bnounding boxes. */
  for (guint i = 0; i < detection_output.detectionOutput.numObjects; i++) {
    NvDsPostProcessObject & obj = detection_output.detectionOutput.objects[i];
    NvDsPostProcessDetectionParams & filter_params =
        all_params[obj.classIndex];
    /* Scale the bounding boxes proportionally based on how the object/frame was
     * scaled during input. */
    if(roi_meta)
    {
        scale_x = roi_meta->scale_ratio_x;
        scale_y = roi_meta->scale_ratio_y;
        offset_left = roi_meta->offset_left;
        roi_left = roi_meta->roi.left;
        offset_top = roi_meta->offset_top;
        roi_top = roi_meta->roi.top;
        roi_height = roi_meta->roi.height;
        roi_width = roi_meta->roi.height;
    }
    if (maintain_aspect_ratio){
      if (scale_x > scale_y){
        scale_y = scale_x;
      }
      else{
        scale_x = scale_y;
      }
    }
    obj.left = (obj.left - offset_left)/scale_x + roi_left;
    obj.top = (obj.top - offset_top)/scale_y + roi_top;
    obj.width /= scale_x;
    obj.height /= scale_y;

    /** Clipping the object bounding-box which lies outside the roi
     * specified by nvdspreprosess plugin. */
    if(roi_meta) {
      if (obj.left + obj.width >= roi_left + roi_width) {
        obj.width = roi_left + roi_width - obj.left;
      }
      if (obj.top + obj.height >= roi_top+ roi_height) {
        obj.height = roi_top+ roi_height - obj.top;
      }
      if (obj.left < roi_left) {
        obj.left = roi_left;
        obj.width = obj.width - roi_left + obj.left;
      }
      if (obj.top < roi_top) {
        obj.top = roi_top;
        obj.height = obj.height - roi_top+ obj.top;
      }
    }
    /* top and left should not be less than 0 */
    if(obj.top < 0)
      obj.top = 0;
    if(obj.left < 0)
      obj.left = 0;

    /* Check if the scaled box co-ordinates meet the detection filter criteria.
     * Skip the box if it does not. */
    if(filterOutClassIds.find(obj.classIndex) != filterOutClassIds.end())
        continue;
    if (obj.width < filter_params.detectionMinWidth)
      continue;
    if (obj.height < filter_params.detectionMinHeight)
      continue;
    if (filter_params.detectionMaxWidth > 0 &&
        obj.width > filter_params.detectionMaxWidth)
      continue;
    if (filter_params.detectionMaxHeight > 0 &&
        obj.height > filter_params.detectionMaxHeight)
      continue;
    if (obj.top < filter_params.roiTopOffset)
      continue;
    if (obj.top + obj.height >
        (surf_height - filter_params.roiBottomOffset))
      continue;
    obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);

    obj_meta->unique_component_id = unique_id;
    obj_meta->confidence = obj.confidence;

    /* This is an untracked object. Set tracking_id to -1. */
    obj_meta->object_id = UNTRACKED_OBJECT_ID;
    obj_meta->class_id = obj.classIndex;

    NvOSD_RectParams & rect_params = obj_meta->rect_params;
    NvOSD_TextParams & text_params = obj_meta->text_params;

    /* Assign bounding box coordinates. These can be overwritten if tracker
     * component is present in the pipeline */
    rect_params.left = obj.left;
    rect_params.top = obj.top;
    rect_params.width = obj.width;
    rect_params.height = obj.height;

    if(process_full_frame == PROCESS_MODEL_OBJECTS) {
      rect_params.left += parent_obj_meta->rect_params.left;
      rect_params.top += parent_obj_meta->rect_params.top;
    }

    /* Preserve original positional bounding box coordinates of detector in the
     * frame so that those can be accessed after tracker */
    obj_meta->detector_bbox_info.org_bbox_coords.left = rect_params.left;
    obj_meta->detector_bbox_info.org_bbox_coords.top = rect_params.top;
    obj_meta->detector_bbox_info.org_bbox_coords.width = rect_params.width;
    obj_meta->detector_bbox_info.org_bbox_coords.height = rect_params.height;

    /* Border of width 3. */
    rect_params.border_width = 3;
    rect_params.has_bg_color = filter_params.color_params.have_bg_color;
    rect_params.bg_color = filter_params.color_params.bg_color;
    rect_params.border_color = filter_params.color_params.border_color;

    if (obj.label)
      g_strlcpy (obj_meta->obj_label, obj.label, MAX_LABEL_SIZE);
    /* display_text requires heap allocated memory. */
    text_params.display_text = g_strdup (obj.label);
    /* Display text above the left top corner of the object. */
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    /* Set black background for the text. */
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    /* Font face, size and color. */
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};

    if (output_instance_mask && obj.mask) {
      float *mask = (float *)g_malloc(obj.mask_size);
      memcpy(mask, obj.mask, obj.mask_size);
      obj_meta->mask_params.data = mask;
      obj_meta->mask_params.size = obj.mask_size;
      obj_meta->mask_params.threshold = segmentationThreshold;
      obj_meta->mask_params.width = obj.mask_width;
      obj_meta->mask_params.height = obj.mask_height;
    }

    nvds_add_obj_meta_to_frame (frame_meta, obj_meta, parent_obj_meta);
  }
  nvds_release_meta_lock (batch_meta);
}



NvDsPostProcessStatus
DetectModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{
    ModelPostProcessor::initResource(initParams),

    m_ClusterMode = initParams.clusterMode;

    m_NumDetectedClasses = initParams.numDetectedClasses;
    if (initParams.numDetectedClasses > 0 &&
        initParams.perClassDetectionParams == nullptr)
    {
        printError(
            "NumDetectedClasses > 0 but PerClassDetectionParams array not "
            "specified");
        return NVDSPOSTPROCESS_CONFIG_FAILED;
    }

    if (!string_empty(initParams.customBBoxParseFuncName)){
        if (!strcmp("NvDsPostProcessParseCustomResnet",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomResnet;
        }else if (!strcmp("NvDsPostProcessParseCustomTfSSD",
              initParams.customBBoxParseFuncName))
        {
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomTfSSD;
        }else if (!strcmp("NvDsPostProcessParseCustomNMSTLT",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomNMSTLT;
        }else if (!strcmp("NvDsPostProcessParseCustomBatchedNMSTLT",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomBatchedNMSTLT;
        }else if (!strcmp("NvDsPostProcessParseCustomFasterRCNN",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomFasterRCNN;
        }
        else if (!strcmp("NvDsPostProcessParseCustomSSD",
                initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomSSD;
        }
        else if (!strcmp("NvDsPostProcessParseCustomYoloV3",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomYoloV3;
        }else if (!strcmp("NvDsPostProcessParseCustomYoloV3Tiny",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomYoloV3Tiny;
        }
        else if (!strcmp("NvDsPostProcessParseCustomYoloV2",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomYoloV2;
        }else if (!strcmp("NvDsPostProcessParseCustomYoloV2Tiny",
              initParams.customBBoxParseFuncName)){
          m_CustomBBoxParseFunc = NvDsPostProcessParseCustomYoloV2Tiny;
        }
        else {
          printError(
              "Custom parsing function %s not present "
              "specified", initParams.customBBoxParseFuncName);
          return NVDSPOSTPROCESS_CONFIG_FAILED;
        }
    }

#ifndef WITH_OPENCV
    if (initParams.clusterMode == NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES &&
        initParams.networkType == NvDsPostProcessNetworkType_Detector) {
        initParams.clusterMode = NVDSPOSTPROCESS_CLUSTER_NMS;
        for (unsigned int i = 0; i < initParams.numDetectedClasses &&
            initParams.perClassDetectionParams; i++) {
            initParams.perClassDetectionParams[i].topK = 20;
            initParams.perClassDetectionParams[i].nmsIOUThreshold = 0.5;
        }
        printWarning ("Warning, OpenCV has been deprecated. Using NMS for clustering instead of cv::groupRectangles with topK = 20 and NMS Threshold = 0.5");
    }
#endif

    m_PerClassDetectionParams.assign(initParams.perClassDetectionParams,
        initParams.perClassDetectionParams + m_NumDetectedClasses);
    m_DetectionParams.numClassesConfigured = initParams.numDetectedClasses;
    m_DetectionParams.perClassPreclusterThreshold.resize(initParams.numDetectedClasses);
    m_DetectionParams.perClassPostclusterThreshold.resize(initParams.numDetectedClasses);

    /* Resize the per class vector to the number of detected classes. */
    m_PerClassObjectList.resize(initParams.numDetectedClasses);
#ifdef WITH_OPENCV
    if (m_ClusterMode == NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES)
    {
        m_PerClassCvRectList.resize(initParams.numDetectedClasses);
    }
#endif

    /* Fill the class thresholds in the m_DetectionParams structure. This
     * will be required during parsing. */
    for (unsigned int i = 0; i < initParams.numDetectedClasses; i++)
    {
        m_DetectionParams.perClassPreclusterThreshold[i] =
            m_PerClassDetectionParams[i].preClusterThreshold;
        m_DetectionParams.perClassPostclusterThreshold[i] =
            m_PerClassDetectionParams[i].postClusterThreshold;
    }

    if (m_ClusterMode == NVDSPOSTPROCESS_CLUSTER_DBSCAN ||
        m_ClusterMode == NVDSPOSTPROCESS_CLUSTER_DBSCAN_NMS_HYBRID)
    {
        m_DBScanHandle.reset(
            NvDsInferDBScanCreate(), [](NvDsInferDBScanHandle handle) {
                if (handle)
                    NvDsInferDBScanDestroy(handle);
            });
        if (!m_DBScanHandle)
        {
            printf("Detect-postprocessor failed to create dbscan handle");
            return NVDSPOSTPROCESS_RESOURCE_ERROR;
        }
    }
    m_PreprocessorSupport = initParams.preprocessor_support;
    return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
DetectModelPostProcessor::parseEachFrame(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessFrameOutput& result)
{
    result.outputType = NvDsPostProcessNetworkType_Detector;
    fillDetectionOutput(outputLayers, result.detectionOutput);
    return NVDSPOSTPROCESS_SUCCESS;
}

bool
DetectModelPostProcessor::parseBoundingBox(vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsPostProcessParseDetectionParams const& detectionParams,
    vector<NvDsPostProcessObjectDetectionInfo>& objectList)
{

    int outputCoverageLayerIndex = -1;
    int outputBBoxLayerIndex = -1;


    for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
    {
        if (strstr(outputLayersInfo[i].layerName, "bbox") != nullptr)
        {
            outputBBoxLayerIndex = i;
        }
        if (strstr(outputLayersInfo[i].layerName, "cov") != nullptr)
        {
            outputCoverageLayerIndex = i;
        }
    }

    if (outputCoverageLayerIndex == -1)
    {
        printf("Could not find output coverage layer for parsing objects");
        return false;
    }
    if (outputBBoxLayerIndex == -1)
    {
        printf("Could not find output bbox layer for parsing objects");
        return false;
    }

    float *outputCoverageBuffer =
        (float *)outputLayersInfo[outputCoverageLayerIndex].buffer;
    float *outputBboxBuffer =
        (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;

    NvDsInferDimsCHW outputCoverageDims;
    NvDsInferDimsCHW outputBBoxDims;

    getDimsCHWFromDims(outputCoverageDims,
        outputLayersInfo[outputCoverageLayerIndex].inferDims);
    getDimsCHWFromDims(
        outputBBoxDims, outputLayersInfo[outputBBoxLayerIndex].inferDims);

    unsigned int targetShape[2] = { outputCoverageDims.w, outputCoverageDims.h };
    float bboxNorm[2] = { 35.0, 35.0 };
    float gcCenters0[targetShape[0]];
    float gcCenters1[targetShape[1]];
    int gridSize = outputCoverageDims.w * outputCoverageDims.h;
    int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, outputBBoxDims.w);
    int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, outputBBoxDims.h);

    for (unsigned int i = 0; i < targetShape[0]; i++)
    {
        gcCenters0[i] = (float)(i * strideX + 0.5);
        gcCenters0[i] /= (float)bboxNorm[0];
    }
    for (unsigned int i = 0; i < targetShape[1]; i++)
    {
        gcCenters1[i] = (float)(i * strideY + 0.5);
        gcCenters1[i] /= (float)bboxNorm[1];
    }

    unsigned int numClasses =
        std::min(outputCoverageDims.c, detectionParams.numClassesConfigured);
    for (unsigned int classIndex = 0; classIndex < numClasses; classIndex++)
    {

        /* Pointers to memory regions containing the (x1,y1) and (x2,y2) coordinates
         * of rectangles in the output bounding box layer. */
        float *outputX1 = outputBboxBuffer
            + classIndex * sizeof (float) * outputBBoxDims.h * outputBBoxDims.w;

        float *outputY1 = outputX1 + gridSize;
        float *outputX2 = outputY1 + gridSize;
        float *outputY2 = outputX2 + gridSize;

        /* Iterate through each point in the grid and check if the rectangle at that
         * point meets the minimum threshold criteria. */
        for (unsigned int h = 0; h < outputCoverageDims.h; h++)
        {
            for (unsigned int w = 0; w < outputCoverageDims.w; w++)
            {
                int i = w + h * outputCoverageDims.w;
                float confidence = outputCoverageBuffer[classIndex * gridSize + i];

                if (confidence < detectionParams.perClassPreclusterThreshold[classIndex])
                    continue;

                float rectX1Float, rectY1Float, rectX2Float, rectY2Float;

                /* Centering and normalization of the rectangle. */
                rectX1Float =
                    outputX1[w + h * outputCoverageDims.w] - gcCenters0[w];
                rectY1Float =
                    outputY1[w + h * outputCoverageDims.w] - gcCenters1[h];
                rectX2Float =
                    outputX2[w + h * outputCoverageDims.w] + gcCenters0[w];
                rectY2Float =
                    outputY2[w + h * outputCoverageDims.w] + gcCenters1[h];

                rectX1Float *= -bboxNorm[0];
                rectY1Float *= -bboxNorm[1];
                rectX2Float *= bboxNorm[0];
                rectY2Float *= bboxNorm[1];

                /* Clip parsed rectangles to frame bounds. */
                if (rectX1Float >= (int)m_NetworkInfo.width)
                    rectX1Float = m_NetworkInfo.width - 1;
                if (rectX2Float >= (int)m_NetworkInfo.width)
                    rectX2Float = m_NetworkInfo.width - 1;
                if (rectY1Float >= (int)m_NetworkInfo.height)
                    rectY1Float = m_NetworkInfo.height - 1;
                if (rectY2Float >= (int)m_NetworkInfo.height)
                    rectY2Float = m_NetworkInfo.height - 1;

                if (rectX1Float < 0)
                    rectX1Float = 0;
                if (rectX2Float < 0)
                    rectX2Float = 0;
                if (rectY1Float < 0)
                    rectY1Float = 0;
                if (rectY2Float < 0)
                    rectY2Float = 0;

                //Prevent underflows
                if(((rectX2Float - rectX1Float) < 0) || ((rectY2Float - rectY1Float) < 0))
                    continue;

                objectList.push_back({ classIndex, rectX1Float,
                         rectY1Float, (rectX2Float - rectX1Float),
                         (rectY2Float - rectY1Float), confidence});
            }
        }
    }
    return true;
}

/**
 * Filter out objects which have been specificed to be removed from the metadata
 * prior to clustering operation
 */
void DetectModelPostProcessor::preClusteringThreshold(
                           NvDsPostProcessParseDetectionParams const &detectionParams,
                           std::vector<NvDsPostProcessObjectDetectionInfo> &objectList)
{
    objectList.erase(std::remove_if(objectList.begin(), objectList.end(),
               [detectionParams](const NvDsPostProcessObjectDetectionInfo& obj)
               { return (obj.classId >= detectionParams.numClassesConfigured) ||
                        (obj.detectionConfidence <
                        detectionParams.perClassPreclusterThreshold[obj.classId])
                        ? true : false;}),objectList.end());
}

/**
 * Filter out the top k objects with the highest probability and ignore the
 * rest
 */
void DetectModelPostProcessor::filterTopKOutputs(const int topK,
                          std::vector<NvDsPostProcessObjectDetectionInfo> &objectList)
{
    if(topK < 0)
        return;

    std::stable_sort(objectList.begin(), objectList.end(),
                    [](const NvDsPostProcessObjectDetectionInfo& obj1,
                       const NvDsPostProcessObjectDetectionInfo& obj2) {
                        return obj1.detectionConfidence > obj2.detectionConfidence; });
    objectList.resize(static_cast<size_t>(topK) <= objectList.size() ? topK : objectList.size());
}

std::vector<int>
DetectModelPostProcessor::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex,
                                           std::vector<NvDsPostProcessParseObjectInfo>& bbox,
                                           const float nmsThreshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU
        = [&overlap1D](NvDsPostProcessParseObjectInfo& bbox1, NvDsPostProcessParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : scoreIndex)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(bbox.at(idx), bbox.at(kept_idx));
                keep = overlap <= nmsThreshold;
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

/** Cluster objects using Non Max Suppression */
void
DetectModelPostProcessor::clusterAndFillDetectionOutputNMS(NvDsPostProcessDetectionOutput &output)
{
    auto maxComp = [](const std::vector<NvDsPostProcessObjectDetectionInfo>& c1,
                      const std::vector<NvDsPostProcessObjectDetectionInfo>& c2) -> bool
                    { return c1.size() < c2.size(); };

    std::vector<std::pair<float, int>> scoreIndex;
    std::vector<NvDsPostProcessObjectDetectionInfo> clusteredBboxes;
    auto maxElement = *std::max_element(m_PerClassObjectList.begin(),
                            m_PerClassObjectList.end(), maxComp);
    clusteredBboxes.reserve(maxElement.size() * m_NumDetectedClasses);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        if(!m_PerClassObjectList[c].empty())
        {
            scoreIndex.reserve(m_PerClassObjectList[c].size());
            scoreIndex.clear();
            for (size_t r = 0; r < m_PerClassObjectList[c].size(); ++r)
            {
                scoreIndex.emplace_back(std::make_pair(m_PerClassObjectList[c][r].detectionConfidence, r));
            }
            std::stable_sort(scoreIndex.begin(), scoreIndex.end(),
                            [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                                return pair1.first > pair2.first; });
            // Apply NMS algorithm
            const std::vector<int> indices = nonMaximumSuppression(scoreIndex, m_PerClassObjectList[c],
                            m_PerClassDetectionParams[c].nmsIOUThreshold);

            std::vector<NvDsPostProcessObjectDetectionInfo> postNMSBboxes;
            for(auto idx : indices) {
                if(m_PerClassObjectList[c][idx].detectionConfidence >
                m_PerClassDetectionParams[c].postClusterThreshold)
                {
                    postNMSBboxes.emplace_back(m_PerClassObjectList[c][idx]);
                }
            }
            filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, postNMSBboxes);
            clusteredBboxes.insert(clusteredBboxes.end(),postNMSBboxes.begin(), postNMSBboxes.end());
        }
    }

    output.objects = new NvDsPostProcessObject[clusteredBboxes.size()];
    output.numObjects = 0;

    for(uint i=0; i < clusteredBboxes.size(); ++i)
    {
        NvDsPostProcessObject &object = output.objects[output.numObjects];
        object.left = clusteredBboxes[i].left;
        object.top = clusteredBboxes[i].top;
        object.width = clusteredBboxes[i].width;
        object.height = clusteredBboxes[i].height;
        object.classIndex = clusteredBboxes[i].classId;
        object.label = nullptr;
        object.mask = nullptr;
        if (object.classIndex < static_cast<int>(m_Labels.size()) && m_Labels[object.classIndex].size() > 0)
                object.label = strdup(m_Labels[object.classIndex][0].c_str());
        object.confidence = clusteredBboxes[i].detectionConfidence;
        output.numObjects++;
    }
}

#ifdef WITH_OPENCV
/**
 * Cluster objects using OpenCV groupRectangles and fill the output structure.
 */
void
DetectModelPostProcessor::clusterAndFillDetectionOutputCV(NvDsPostProcessDetectionOutput& output)
{
    size_t totalObjects = 0;

    for (auto & list:m_PerClassCvRectList)
        list.clear();

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (auto & object:m_ObjectList)
    {
        m_PerClassCvRectList[object.classId].emplace_back(object.left,
                object.top, object.width, object.height);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object. Refer
         * to opencv documentation of groupRectangles for more
         * information about the tuning parameters for grouping. */
        if (m_PerClassDetectionParams[c].groupThreshold > 0)
            cv::groupRectangles(m_PerClassCvRectList[c],
                    m_PerClassDetectionParams[c].groupThreshold,
                    m_PerClassDetectionParams[c].eps);
        totalObjects += m_PerClassCvRectList[c].size();
    }

    output.objects = new NvDsPostProcessObject[totalObjects];
    output.numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (auto & rect:m_PerClassCvRectList[c])
        {
            NvDsPostProcessObject &object = output.objects[output.numObjects];
            object.left = rect.x;
            object.top = rect.y;
            object.width = rect.width;
            object.height = rect.height;
            object.classIndex = c;
            object.label = nullptr;
            object.mask = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            object.confidence = -0.1;
            output.numObjects++;
        }
    }
}
#endif

/**
 * Cluster objects using DBSCAN and fill the output structure.
 */
void
DetectModelPostProcessor::clusterAndFillDetectionOutputDBSCAN(NvDsPostProcessDetectionOutput& output)
{
    size_t totalObjects = 0;
    NvDsInferDBScanClusteringParams clusteringParams;
    clusteringParams.enableATHRFilter = ATHR_ENABLED;
    clusteringParams.thresholdATHR = ATHR_THRESHOLD;
    assert(m_DBScanHandle);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        NvDsInferObjectDetectionInfo *objArray =
          (NvDsInferObjectDetectionInfo*) (m_PerClassObjectList[c].data());
        size_t numObjects = m_PerClassObjectList[c].size();
        NvDsPostProcessDetectionParams detectionParams = m_PerClassDetectionParams[c];

        clusteringParams.eps = detectionParams.eps;
        clusteringParams.minBoxes = detectionParams.minBoxes;
        clusteringParams.minScore = detectionParams.minScore;

        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object using
         * DBSCAN. */
        if (detectionParams.minBoxes > 0) {
            NvDsInferDBScanCluster(
                m_DBScanHandle.get(), &clusteringParams, objArray, &numObjects);
        }
        m_PerClassObjectList[c].resize(numObjects);
        m_PerClassObjectList[c].erase(std::remove_if(m_PerClassObjectList[c].begin(),
               m_PerClassObjectList[c].end(),
               [detectionParams](const NvDsPostProcessObjectDetectionInfo& obj)
               { return (obj.detectionConfidence <
                        detectionParams.postClusterThreshold)
                        ? true : false;}),m_PerClassObjectList[c].end());
        filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, m_PerClassObjectList.at(c));
        totalObjects += m_PerClassObjectList[c].size();
    }

    output.objects = new NvDsPostProcessObject[totalObjects];
    output.numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (size_t i = 0; i < m_PerClassObjectList[c].size(); i++)
        {
            NvDsPostProcessObject &object = output.objects[output.numObjects];
            object.left = m_PerClassObjectList[c][i].left;
            object.top = m_PerClassObjectList[c][i].top;
            object.width = m_PerClassObjectList[c][i].width;
            object.height = m_PerClassObjectList[c][i].height;
            object.classIndex = c;
            object.label = nullptr;
            object.mask = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            object.confidence = m_PerClassObjectList[c][i].detectionConfidence;
            output.numObjects++;
        }
    }
}

/**
 * Cluster objects using a hybrid algorithm of DBSCAN + NMS
 * and fill the output structure.
 */
void
DetectModelPostProcessor::clusterAndFillDetectionOutputHybrid(NvDsPostProcessDetectionOutput& output)
{
    NvDsInferDBScanClusteringParams clusteringParams;
    clusteringParams.enableATHRFilter = ATHR_ENABLED;
    clusteringParams.thresholdATHR = ATHR_THRESHOLD;
    assert(m_DBScanHandle);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        NvDsInferObjectDetectionInfo *objArray =
                (NvDsInferObjectDetectionInfo*)m_PerClassObjectList[c].data();
        size_t numObjects = m_PerClassObjectList[c].size();

        clusteringParams.eps = m_PerClassDetectionParams[c].eps;
        clusteringParams.minBoxes = m_PerClassDetectionParams[c].minBoxes;
        clusteringParams.minScore = m_PerClassDetectionParams[c].minScore;

        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object using
         * DBSCAN. */
        if (m_PerClassDetectionParams[c].minBoxes > 0) {
            NvDsInferDBScanClusterHybrid(
                m_DBScanHandle.get(), &clusteringParams, objArray, &numObjects);
        }
        m_PerClassObjectList[c].resize(numObjects);
    }

    return clusterAndFillDetectionOutputNMS(output);
}

/**
 * full the output structure without performing any clustering operations
 */

void
DetectModelPostProcessor::fillUnclusteredOutput(NvDsPostProcessDetectionOutput& output)
{
    for (auto & object:m_ObjectList)
    {
        m_PerClassObjectList[object.classId].emplace_back(object);
    }

    unsigned int totalObjects = 0;
    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, m_PerClassObjectList.at(c));
        totalObjects += m_PerClassObjectList.at(c).size();
    }

    output.objects = new NvDsPostProcessObject[totalObjects];
    output.numObjects = 0;
    for(const auto& perClassList : m_PerClassObjectList)
    {
        for(const auto& obj: perClassList)
        {
            NvDsPostProcessObject &object = output.objects[output.numObjects];
            object.left = obj.left;
            object.top = obj.top;
            object.width = obj.width;
            object.height = obj.height;
            object.classIndex = obj.classId;
            object.label = nullptr;
            object.mask = nullptr;
            if(obj.classId < m_Labels.size() && m_Labels[obj.classId].size() > 0)
                object.label = strdup(m_Labels[obj.classId][0].c_str());
            object.confidence = obj.detectionConfidence;

            ++output.numObjects;
        }
    }
}

NvDsPostProcessStatus
DetectModelPostProcessor::fillDetectionOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessDetectionOutput& output)
{
    /* Clear the object lists. */
    m_ObjectList.clear();

    /* Clear all per class object lists */
    for (auto & list:m_PerClassObjectList)
        list.clear();

    if (m_CustomBBoxParseFunc){
      if (!m_CustomBBoxParseFunc(outputLayers, m_NetworkInfo,
            m_DetectionParams, m_ObjectList)){
        printf("Failed to parse bboxes");
        return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
      }
    }
    else {
      if (!parseBoundingBox(outputLayers, m_NetworkInfo,
            m_DetectionParams, m_ObjectList)){
        printf("Failed to parse bboxes");
        return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
      }
    }

    preClusteringThreshold(m_DetectionParams, m_ObjectList);

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
#ifndef WITH_OPENCV
    if(m_ClusterMode != NVDSPOSTPROCESS_CLUSTER_NONE)
#else
    if((m_ClusterMode != NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES) &&
        (m_ClusterMode != NVDSPOSTPROCESS_CLUSTER_NONE))
#endif
    {
        for (auto & object:m_ObjectList)
        {
            m_PerClassObjectList[object.classId].emplace_back(object);
        }
    }

    switch (m_ClusterMode)
    {
        case NVDSPOSTPROCESS_CLUSTER_NMS:
            clusterAndFillDetectionOutputNMS(output);
            break;

        case NVDSPOSTPROCESS_CLUSTER_DBSCAN:
            clusterAndFillDetectionOutputDBSCAN(output);
            break;

#ifdef WITH_OPENCV
        case NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES:
            clusterAndFillDetectionOutputCV(output);
            break;
#endif

        case NVDSPOSTPROCESS_CLUSTER_DBSCAN_NMS_HYBRID:
            clusterAndFillDetectionOutputHybrid(output);
            break;

        case NVDSPOSTPROCESS_CLUSTER_NONE:
            fillUnclusteredOutput(output);
            break;

        default:
            break;
    }

    return NVDSPOSTPROCESS_SUCCESS;
}

void
DetectModelPostProcessor::releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsPostProcessNetworkType_Detector:
            for (unsigned int j = 0; j < frameOutput.detectionOutput.numObjects;
                    j++)
            {
                free(frameOutput.detectionOutput.objects[j].label);
            }
            delete[] frameOutput.detectionOutput.objects;
            break;
        default:
            break;
    }
}

