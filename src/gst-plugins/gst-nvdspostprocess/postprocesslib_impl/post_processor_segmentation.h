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

#ifndef __POST_PROCESSOR_SEGMENTATION_HPP__
#define __POST_PROCESSOR_SEGMENTATION_HPP__

#include "post_processor.h"
#include <functional>

class SegmentationModelPostProcessor : public ModelPostProcessor{

public:
  SegmentationModelPostProcessor(int id, int gpuId = 0)
    : ModelPostProcessor (NvDsPostProcessNetworkType_Segmentation, id, gpuId) {}

  ~SegmentationModelPostProcessor() override = default;

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


  void releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput) override;
private:
  NvDsPostProcessStatus fillSegmentationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessSegmentationOutput& output);

  float m_SegmentationThreshold = 0.50;
  NvDsPostProcessTensorOrder m_SegmentationOutputOrder = NvDsPostProcessTensorOrder_kNCHW;
  uint32_t m_NumClasses = 0;
};

#endif
