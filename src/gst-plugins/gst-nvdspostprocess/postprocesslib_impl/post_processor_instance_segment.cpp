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

#include "post_processor_instance_segment.h"

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


using namespace std;

/**
 * Attach metadata for the detector. We will be adding a new metadata.
 */
void
InstanceSegmentModelPostProcessor::attachMetadata (NvBufSurface *surf, gint batch_idx,
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
    gboolean symmetric_padding) //FIXME: handle padding
{
  static gchar font_name[] = "Serif";
  NvDsObjectMeta *obj_meta = NULL;
  nvds_acquire_meta_lock (batch_meta);
  gint surf_width  = surf->surfaceList[batch_idx].width;
  gint surf_height = surf->surfaceList[batch_idx].height;
  float scale_x =
    (float)surf_width/(float)m_NetworkInfo.width;
  float scale_y =
    (float)surf_height/(float)m_NetworkInfo.height;

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
    //FIXME: preprocess meta
    obj.left = (obj.left - 0)*scale_x + 0;
    obj.top  = (obj.top - 0)*scale_y + 0;
    obj.width *= scale_x;
    obj.height *= scale_y;

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
InstanceSegmentModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{
    ModelPostProcessor::initResource(initParams),

    m_ClusterMode = initParams.clusterMode;
    if (m_ClusterMode != NVDSPOSTPROCESS_CLUSTER_NONE) {
      printError(" cluster mode %d not supported with instance segmentation", m_ClusterMode);
      return NVDSPOSTPROCESS_CONFIG_FAILED;
    }

    m_NumDetectedClasses = initParams.numDetectedClasses;
    if (initParams.numDetectedClasses > 0 &&
        initParams.perClassDetectionParams == nullptr)
    {
        printError(
            "NumDetectedClasses > 0 but PerClassDetectionParams array not "
            "specified");
        return NVDSPOSTPROCESS_CONFIG_FAILED;
    }

    if (!string_empty(initParams.customBBoxInstanceMaskParseFuncName)){
        if (!strcmp("NvDsPostProcessParseCustomMrcnnTLT ",
              initParams.customBBoxInstanceMaskParseFuncName)){
          m_CustomParseFunc = NvDsPostProcessParseCustomMrcnnTLT ;
        }else if (!strcmp("NvDsInferParseCustomMrcnnTLTV2",
              initParams.customBBoxInstanceMaskParseFuncName))
        {
          m_CustomParseFunc = NvDsPostProcessParseCustomMrcnnTLTV2;
        }
        else {
          printError(
              "Custom parsing function %s not present "
              "specified", initParams.customBBoxParseFuncName);
          return NVDSPOSTPROCESS_CONFIG_FAILED;
        }
    }
    else {
      printError(
          "Custom parsing function not "
          "specified for instance segment post processor");
      return NVDSPOSTPROCESS_CONFIG_FAILED;
    }

    m_PerClassDetectionParams.assign(initParams.perClassDetectionParams,
        initParams.perClassDetectionParams + m_NumDetectedClasses);
    m_DetectionParams.numClassesConfigured = initParams.numDetectedClasses;
    m_DetectionParams.perClassPreclusterThreshold.resize(initParams.numDetectedClasses);
    m_DetectionParams.perClassPostclusterThreshold.resize(initParams.numDetectedClasses);

    /* Resize the per class vector to the number of detected classes. */
    m_PerClassInstanceMaskList.resize(initParams.numDetectedClasses);

    /* Fill the class thresholds in the m_DetectionParams structure. This
     * will be required during parsing. */
    for (unsigned int i = 0; i < initParams.numDetectedClasses; i++)
    {
        m_DetectionParams.perClassPreclusterThreshold[i] =
            m_PerClassDetectionParams[i].preClusterThreshold;
        m_DetectionParams.perClassPostclusterThreshold[i] =
            m_PerClassDetectionParams[i].postClusterThreshold;
    }

    return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
InstanceSegmentModelPostProcessor::parseEachFrame(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessFrameOutput& result)
{
    result.outputType = NvDsPostProcessNetworkType_InstanceSegmentation;
    fillDetectionOutput(outputLayers, result.detectionOutput);
    return NVDSPOSTPROCESS_SUCCESS;
}


/**
 * Filter out objects which have been specificed to be removed from the metadata
 * prior to clustering operation
 */
void InstanceSegmentModelPostProcessor::preClusteringThreshold(
                           NvDsPostProcessParseDetectionParams const &detectionParams,
                           std::vector<NvDsPostProcessInstanceMaskInfo> &objectList)
{
    objectList.erase(std::remove_if(objectList.begin(), objectList.end(),
               [detectionParams](const NvDsPostProcessInstanceMaskInfo& obj)
               { return (obj.classId >= detectionParams.numClassesConfigured) ||
                        (obj.detectionConfidence <
                        detectionParams.perClassPreclusterThreshold[obj.classId])
                        ? true : false;}),objectList.end());
}

/**
 * Filter out the top k objects with the highest probability and ignore the
 * rest
 */
void InstanceSegmentModelPostProcessor::filterTopKOutputs(const int topK,
                          std::vector<NvDsPostProcessInstanceMaskInfo> &objectList)
{
    if(topK < 0)
        return;

    std::stable_sort(objectList.begin(), objectList.end(),
                    [](const NvDsPostProcessInstanceMaskInfo& obj1,
                       const NvDsPostProcessInstanceMaskInfo& obj2) {
                        return obj1.detectionConfidence > obj2.detectionConfidence; });
    objectList.resize(static_cast<size_t>(topK) <= objectList.size() ? topK : objectList.size());
}



/**
 * full the output structure without performing any clustering operations
 */
void
InstanceSegmentModelPostProcessor::fillUnclusteredOutput(NvDsPostProcessDetectionOutput& output)
{
    for (auto & object:m_InstanceMaskList)
    {
        m_PerClassInstanceMaskList[object.classId].emplace_back(object);
    }

    unsigned int totalObjects = 0;
    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, m_PerClassInstanceMaskList.at(c));
        totalObjects += m_PerClassInstanceMaskList.at(c).size();
    }

    output.objects = new NvDsPostProcessObject[totalObjects];
    output.numObjects = 0;
    for(const auto& perClassList : m_PerClassInstanceMaskList)
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
            if(obj.classId < m_Labels.size() && m_Labels[obj.classId].size() > 0)
                object.label = strdup(m_Labels[obj.classId][0].c_str());
            object.confidence = obj.detectionConfidence;
            object.mask = nullptr;
            if (obj.mask) {
                object.mask = std::move(obj.mask);
                object.mask_width = obj.mask_width;
                object.mask_height = obj.mask_height;
                object.mask_size = obj.mask_size;
            }
            ++output.numObjects;
        }
    }
}

NvDsPostProcessStatus
InstanceSegmentModelPostProcessor::fillDetectionOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessDetectionOutput& output)
{
    /* Clear the object lists. */
    m_InstanceMaskList.clear();

    /* Clear all per class object lists */
    for (auto & list:m_PerClassInstanceMaskList)
        list.clear();

    if (m_CustomParseFunc){
      if (!m_CustomParseFunc(outputLayers, m_NetworkInfo,
            m_DetectionParams, m_InstanceMaskList)){
        printError("Failed to parse bboxes");
        return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
      }
    }
    else
    {
        printError("Failed to find custom parse function");
        return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
    }

    preClusteringThreshold(m_DetectionParams, m_InstanceMaskList);

    switch (m_ClusterMode)
    {
        case NVDSPOSTPROCESS_CLUSTER_NONE:
            fillUnclusteredOutput(output);
            break;
        default:
            printError("Invalid cluster mode for instance mask detection");
            return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
    }

    return NVDSPOSTPROCESS_SUCCESS;
}

void
InstanceSegmentModelPostProcessor::releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsPostProcessNetworkType_InstanceSegmentation:
            for (unsigned int j = 0; j < frameOutput.detectionOutput.numObjects;
                    j++)
            {
                free(frameOutput.detectionOutput.objects[j].label);
                delete[] frameOutput.detectionOutput.objects[j].mask;
            }
            delete[] frameOutput.detectionOutput.objects;
            break;
        default:
            break;
    }
}

