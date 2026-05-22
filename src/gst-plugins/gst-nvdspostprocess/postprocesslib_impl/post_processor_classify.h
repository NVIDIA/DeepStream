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

#ifndef __POST_PROCESSOR_CLASSIFY_HPP__
#define __POST_PROCESSOR_CLASSIFY_HPP__

#include "post_processor.h"


typedef bool (* NvDsPostProcessClassiferParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsPostProcessAttribute> &attrList,
        std::string &descString);

class ClassifyModelPostProcessor : public ModelPostProcessor{

public:
  ClassifyModelPostProcessor(int id, int gpuId = 0)
    : ModelPostProcessor (NvDsPostProcessNetworkType_Classifier, id, gpuId) {}

  ~ClassifyModelPostProcessor() override = default;

  NvDsPostProcessStatus
  initResource(NvDsPostProcessContextInitParams& initParams) override;

  NvDsPostProcessStatus parseEachFrame(const std::vector <NvDsInferLayerInfo>&
       outputLayers,
       NvDsPostProcessFrameOutput &result) override;

  void
   attachMetadata (NvBufSurface *surf, gint batch_idx,
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
    gboolean symmetric_padding) override;


  void
    mergeClassificationOutput (NvDsPostProcessObjectHistory & history,
    NvDsPostProcessObjectInfo &new_result);

  void releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput) override;

private:
  NvDsPostProcessStatus fillClassificationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessClassificationOutput& output);
  bool parseAttributesFromSoftmaxLayers(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
    std::vector<NvDsPostProcessAttribute>& attrList, std::string& attrString);

  NvDsPostProcessObjectInfo m_ObjectHistory;
  float m_ClassifierThreshold = 0.51;
  const gchar *m_ClassifierType;
  uint32_t m_NumClasses = 0;
  NvDsPostProcessClassiferParseCustomFunc m_CustomClassifierParseFunc = nullptr;
};

#endif
