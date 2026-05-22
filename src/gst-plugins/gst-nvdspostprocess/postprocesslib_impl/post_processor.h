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

#ifndef __POST_PROCESSOR_HPP__
#define __POST_PROCESSOR_HPP__

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
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"
#include "nvdsinfer_dbscan.h"
#include "post_processor_struct.h"


#ifndef PP_DISABLE_CLASS_COPY
#define PP_DISABLE_CLASS_COPY(NoCopyClass)  \
    NoCopyClass(const NoCopyClass&) = delete; \
    void operator=(const NoCopyClass&) = delete
#endif


class ModelPostProcessor
{

protected:
  ModelPostProcessor(NvDsPostProcessNetworkType type, int id, int gpuId)
    : m_NetworkType(type), m_UniqueID(id), m_GpuID(gpuId){}

public:
  virtual ~ModelPostProcessor() = default;

  virtual NvDsPostProcessStatus
  initResource(NvDsPostProcessContextInitParams& initParams);
  const std::vector<std::vector<std::string>>& getLabels() const
  {
    return m_Labels;
  }
  void freeBatchOutput(NvDsPostProcessBatchOutput& batchOutput);
  void setNetworkInfo (NvDsInferNetworkInfo networkInfo){
    m_NetworkInfo = networkInfo;
  }

  virtual NvDsPostProcessStatus parseEachFrame(
      const std::vector <NvDsInferLayerInfo> &outputLayers,
      NvDsPostProcessFrameOutput& result) = 0;


  virtual void
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
    gboolean symmetric_padding) = 0;


  virtual void releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput) = 0;

protected:
  NvDsPostProcessStatus parseLabelsFile(const std::string &path);

private:
  PP_DISABLE_CLASS_COPY(ModelPostProcessor);

protected:
  /* Processor type */
  NvDsPostProcessNetworkType m_NetworkType = NvDsPostProcessNetworkType_Other;

  int m_UniqueID = 0;
  uint32_t m_GpuID = 0;

  /* Network input information. */
  NvDsInferNetworkInfo m_NetworkInfo = {0};
  std::vector<NvDsInferLayerInfo> m_AllLayerInfo;
  std::vector<NvDsInferLayerInfo> m_OutputLayerInfo;

  /* Holds the string labels for classes. */
  std::vector<std::vector<std::string>> m_Labels;

};

#endif
