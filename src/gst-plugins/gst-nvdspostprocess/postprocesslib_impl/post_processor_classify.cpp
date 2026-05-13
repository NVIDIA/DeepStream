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

#include "post_processor_classify.h"
using namespace std;
extern "C"
bool NvDsPostProcessClassiferParseCustomSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsPostProcessAttribute> &attrList,
        std::string &descString);
NvDsPostProcessStatus
ClassifyModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{
  ModelPostProcessor::initResource(initParams);
  m_ClassifierThreshold = initParams.classifierThreshold;
  m_ClassifierType = initParams.classifier_type;
  if (!string_empty(initParams.customClassifierParseFuncName)){
    if (!strcmp(initParams.customClassifierParseFuncName,
          "NvDsPostProcessClassiferParseCustomSoftmax ")){
      m_CustomClassifierParseFunc = NvDsPostProcessClassiferParseCustomSoftmax;
    }
    else {
      printError(
          "Custom parsing function %s not present "
          "specified", initParams.customClassifierParseFuncName);
      return NVDSPOSTPROCESS_CONFIG_FAILED;
    }
  }
  return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
ClassifyModelPostProcessor::parseEachFrame(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessFrameOutput& result)
{
    result.outputType = NvDsPostProcessNetworkType_Classifier;
    fillClassificationOutput(outputLayers, result.classificationOutput);
    return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
ClassifyModelPostProcessor::fillClassificationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessClassificationOutput& output)
{
  std::string attrString;
  std::vector<NvDsPostProcessAttribute> attributes;

  if (m_CustomClassifierParseFunc){
    if (!m_CustomClassifierParseFunc(outputLayers, m_NetworkInfo,
          m_ClassifierThreshold, attributes, attrString))
    {
      printError("Failed to parse Classification output");
      return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
    }
  }
  else{
    if (!parseAttributesFromSoftmaxLayers(outputLayers, m_NetworkInfo,
          m_ClassifierThreshold, attributes, attrString))
    {
      printError("Failed to parse Classification output");
      return NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED;
    }
  }

  /* Fill the output structure with the parsed attributes. */
  output.label = strdup(attrString.c_str());
  output.numAttributes = attributes.size();
  output.attributes = new NvDsPostProcessAttribute[output.numAttributes];
  for (size_t i = 0; i < output.numAttributes; i++)
  {
    output.attributes[i].attributeIndex = attributes[i].attributeIndex;
    output.attributes[i].attributeValue = attributes[i].attributeValue;
    output.attributes[i].attributeConfidence = attributes[i].attributeConfidence;
    output.attributes[i].attributeLabel = attributes[i].attributeLabel;
  }
  return NVDSPOSTPROCESS_SUCCESS;
}


bool
ClassifyModelPostProcessor::parseAttributesFromSoftmaxLayers(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
    std::vector<NvDsPostProcessAttribute>& attrList, std::string& attrString)
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
        float *outputCoverageBuffer =
            (float *)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsPostProcessAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > m_ClassifierThreshold
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
            if (m_Labels.size() > attr.attributeIndex &&
                    attr.attributeValue < m_Labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    strdup(m_Labels[attr.attributeIndex][attr.attributeValue].c_str());
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                attrString.append(attr.attributeLabel).append(" ");
        }
    }

    return true;
}



void
ClassifyModelPostProcessor::mergeClassificationOutput (NvDsPostProcessObjectHistory & history,
    NvDsPostProcessObjectInfo &new_result)
{
  for (auto &attr : history.cached_info.attributes) {
    free(attr.attributeLabel);
  }
  history.cached_info.attributes.assign (new_result.attributes.begin (),
      new_result.attributes.end ());
  for (auto &attr : history.cached_info.attributes) {
    attr.attributeLabel =
        attr.attributeLabel ? strdup (attr.attributeLabel) : nullptr;
  }
  history.cached_info.label.assign (new_result.label);
}

void
ClassifyModelPostProcessor::attachMetadata (NvBufSurface *surf, gint batch_idx,
    NvDsBatchMeta  *batch_meta,
    NvDsFrameMeta  *frame_meta,
    NvDsObjectMeta  *object_meta,
    NvDsObjectMeta *parent_obj_meta,
    NvDsPostProcessFrameOutput & model_output,
    NvDsPostProcessDetectionParams *all_params,
    std::set <gint> & filterOutClassIds,
    int32_t unique_id,
    gboolean output_instance_mask,
    gboolean process_full_frame,
    float segmentationThreshold,
    gboolean maintain_aspect_ratio,
    NvDsRoiMeta *roi_meta,
    gboolean symmetric_padding)
{

  if (model_output.classificationOutput.numAttributes == 0 ||
      model_output.classificationOutput.label == NULL)
    return;

  if (frame_meta)
    frame_meta->bInferDone = TRUE;

  nvds_acquire_meta_lock (batch_meta);
  if (process_full_frame == PROCESS_MODEL_FULL_FRAME) {

    /* Attach only one object in the meta since this is a full frame
     * classification. */
    object_meta = nvds_acquire_obj_meta_from_pool (batch_meta);

    /* Font to be used for label text. */
    static gchar font_name[] = "Serif";

    NvOSD_RectParams & rect_params = object_meta->rect_params;
    NvOSD_TextParams & text_params = object_meta->text_params;

    //frame.object_meta = object_meta;

    /* Assign bounding box coordinates. */
    rect_params.left = 0;
    rect_params.top = 0;
    rect_params.width = surf->surfaceList[batch_idx].width;
    rect_params.height = surf->surfaceList[batch_idx].height;

    /* Semi-transparent yellow background. */
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    /* Red border of width 6. */
    rect_params.border_width = 6;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    object_meta->object_id = UNTRACKED_OBJECT_ID;
    object_meta->class_id = -1;

    /* display_text requires heap allocated memory. Actual string formation
     * is done later in the function. */
    text_params.display_text = g_strdup ("");
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

    /* Attach the NvDsFrameMeta structure as NvDsMeta to the buffer. Pass the
     * function to be called when freeing the meta_data. */
    nvds_add_obj_meta_to_frame (frame_meta, object_meta, NULL);
  }

  std::string string_label = model_output.classificationOutput.label;

  /* Fill the attribute info structure for the object. */
  guint num_attrs = model_output.classificationOutput.numAttributes;

  NvDsClassifierMeta *classifier_meta =
      nvds_acquire_classifier_meta_from_pool (batch_meta);

  classifier_meta->unique_component_id = unique_id;
  classifier_meta->classifier_type = m_ClassifierType;

  for (unsigned int i = 0; i < num_attrs; i++) {
    NvDsLabelInfo *label_info =
        nvds_acquire_label_info_meta_from_pool (batch_meta);
    NvDsPostProcessAttribute &attr = model_output.classificationOutput.attributes[i];
    label_info->label_id = attr.attributeIndex;
    label_info->result_class_id = attr.attributeValue;
    label_info->result_prob = attr.attributeConfidence;
    if (attr.attributeLabel) {
      g_strlcpy (label_info->result_label, attr.attributeLabel, MAX_LABEL_SIZE);
      if (model_output.classificationOutput.label == NULL)
        string_label.append (attr.attributeLabel).append(" ");
    }

    nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
  }

  if (string_label.length () > 0 && object_meta) {
    gchar *temp = object_meta->text_params.display_text;
    if(temp == nullptr) {
        NvOSD_TextParams & text_params = object_meta->text_params;
        NvOSD_RectParams & rect_params = object_meta->rect_params;
        /* display_text requires heap allocated memory. Actual string formation
         * is done later in the function. */
        text_params.display_text = g_strdup ("");
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams) {
        0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = const_cast<gchar*>("Serif");
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams) {
        1, 1, 1, 1};
    }
    //printf ("+%p %d %s\n",object_meta, unique_id, object_meta->text_params.display_text);
    temp = object_meta->text_params.display_text;
    object_meta->text_params.display_text =
        g_strconcat (temp, " ", string_label.c_str (), nullptr);
    //printf ("=%p %d %s\n",object_meta, unique_id, object_meta->text_params.display_text);
    g_free (temp);
  }
  nvds_add_classifier_meta_to_object (object_meta, classifier_meta);
  nvds_release_meta_lock (batch_meta);
}

void
ClassifyModelPostProcessor::releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsPostProcessNetworkType_Classifier:
            for (unsigned int j = 0; j < frameOutput.classificationOutput.numAttributes;
                    j++)
            {
                if (frameOutput.classificationOutput.attributes[j].attributeLabel)
                    free(frameOutput.classificationOutput.attributes[j].attributeLabel);
            }
            free(frameOutput.classificationOutput.label);
            delete[] frameOutput.classificationOutput.attributes;
            break;
        default:
            break;
    }
}

