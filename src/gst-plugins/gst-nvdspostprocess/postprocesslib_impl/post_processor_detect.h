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

#ifndef __POST_PROCESSOR_DETECT_HPP__
#define __POST_PROCESSOR_DETECT_HPP__

#include "post_processor.h"

typedef bool (* NvDsPostProcessParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

class DetectModelPostProcessor : public ModelPostProcessor{

public:
  DetectModelPostProcessor(int id, int gpuId = 0)
    : ModelPostProcessor (NvDsPostProcessNetworkType_Detector, id, gpuId) {}

  NvDsPostProcessStatus
  initResource(NvDsPostProcessContextInitParams& initParams) override;
  ~DetectModelPostProcessor() override = default;
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

  void releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput) override;
 private:
  bool parseBoundingBox(std::vector <NvDsInferLayerInfo> const&
       outputLayersInfo,
       NvDsInferNetworkInfo const& networkInfo,
       NvDsPostProcessParseDetectionParams const& detectionParams,
       std::vector<NvDsPostProcessObjectDetectionInfo>& objectList);
  std::vector<int> nonMaximumSuppression(
      std::vector <std::pair<float,int>>& scoreIndex,
      std::vector<NvDsPostProcessParseObjectInfo> &bbox,
      float const nmsThreshold);

  void clusterAndFillDetectionOutputHybrid(NvDsPostProcessDetectionOutput& output);
  void clusterAndFillDetectionOutputNMS(NvDsPostProcessDetectionOutput& output);
  void clusterAndFillDetectionOutputDBSCAN(NvDsPostProcessDetectionOutput& output);
  void fillUnclusteredOutput(NvDsPostProcessDetectionOutput& output);
  NvDsPostProcessStatus fillDetectionOutput(const std::vector <NvDsInferLayerInfo>& outputLayers,
      NvDsPostProcessDetectionOutput& output);
  void preClusteringThreshold(NvDsPostProcessParseDetectionParams const &detectionParams,
      std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);
  void filterTopKOutputs(int const topK,
      std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

private:
  NvDsPostProcessClusterMode m_ClusterMode;

  uint32_t m_NumDetectedClasses = 0;
  std::shared_ptr<NvDsInferDBScan> m_DBScanHandle;

  std::vector <NvDsPostProcessDetectionParams> m_PerClassDetectionParams;
  NvDsPostProcessParseDetectionParams m_DetectionParams = {0, {}, {}};

  std::vector <NvDsPostProcessObjectDetectionInfo> m_ObjectList;
#ifdef WITH_OPENCV
  /* Vector of cv::Rect vectors for each class. */
  std::vector<std::vector<cv::Rect>> m_PerClassCvRectList;
#endif
   /* Vector of NvDsPostProcessObjectDetectionInfo vectors for each class. */
   std::vector<std::vector<NvDsPostProcessObjectDetectionInfo>> m_PerClassObjectList;

   NvDsPostProcessParseCustomFunc m_CustomBBoxParseFunc = nullptr;
   bool m_PreprocessorSupport = FALSE;

};


#endif
