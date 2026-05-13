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

#include "post_processor_segmentation.h"

using namespace std;

static void
release_segmentation_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsInferSegmentationMeta *meta = (NvDsInferSegmentationMeta *) user_meta->user_meta_data;
  if (meta->priv_data) {
    gst_mini_object_unref (GST_MINI_OBJECT (meta->priv_data));
  } else {
    g_free (meta->class_map);
    g_free (meta->class_probabilities_map);
  }
  delete meta;
}

static gpointer
copy_segmentation_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *src_user_meta = (NvDsUserMeta *) data;
  NvDsInferSegmentationMeta *src_meta = (NvDsInferSegmentationMeta *) src_user_meta->user_meta_data;
  NvDsInferSegmentationMeta *meta = (NvDsInferSegmentationMeta *) g_malloc (sizeof (NvDsInferSegmentationMeta));

  meta->classes = src_meta->classes;
  meta->width = src_meta->width;
  meta->height = src_meta->height;
  meta->class_map = (gint *) g_memdup2(src_meta->class_map,
      meta->width * meta->height * sizeof (gint));
  meta->class_probabilities_map = (gfloat *) g_memdup2(src_meta->class_probabilities_map,
      meta->classes * meta->width * meta->height * sizeof (gfloat));
  meta->priv_data = NULL;

  return meta;
}


NvDsPostProcessStatus
SegmentationModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{
  ModelPostProcessor::initResource(initParams);
  m_SegmentationThreshold = initParams.segmentationThreshold;
  m_SegmentationOutputOrder = initParams.segmentationOutputOrder;
  return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
SegmentationModelPostProcessor::parseEachFrame(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessFrameOutput& result)
{
    result.outputType = NvDsPostProcessNetworkType_Segmentation;
    fillSegmentationOutput(outputLayers, result.segmentationOutput);
    return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
SegmentationModelPostProcessor::fillSegmentationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessSegmentationOutput& output)
{
    std::function<unsigned int(unsigned int, unsigned int, unsigned int)> indAlongChannel = nullptr;
    NvDsInferDimsCHW outputDimsCHW = {0};

    if (m_SegmentationOutputOrder == NvDsPostProcessTensorOrder_kNCHW) {
        getDimsCHWFromDims(outputDimsCHW, outputLayers[0].inferDims);
        indAlongChannel = [&outputDimsCHW](int x, int y, int c)-> int
            { return c * outputDimsCHW.w * outputDimsCHW.h + y * outputDimsCHW.w + x;};
    }
    else if (m_SegmentationOutputOrder == NvDsPostProcessTensorOrder_kNHWC) {
        getDimsHWCFromDims(outputDimsCHW, outputLayers[0].inferDims);
        indAlongChannel = [&outputDimsCHW](int x, int y, int c)-> int
            {return  outputDimsCHW.c * ( y * outputDimsCHW.w + x) + c;};
    }

    output.width = outputDimsCHW.w;
    output.height = outputDimsCHW.h;
    output.classes = outputDimsCHW.c;

    output.class_map = (gint*)g_malloc0(sizeof(gint)*output.width * output.height);

    output.class_probability_map = (gfloat *) g_memdup2((float*)outputLayers[0].buffer,
      output.classes * output.width * output.height * sizeof (gfloat));

    for (unsigned int y = 0; y < output.height; y++)
    {
        for (unsigned int x = 0; x < output.width; x++)
        {
            float max_prob = -1;
            int &cls = output.class_map[y * output.width + x] = -1;
            for (unsigned int c = 0; c < output.classes; c++)
            {
                float prob = output.class_probability_map[indAlongChannel(x,y,c)];
                if (prob > max_prob && prob > m_SegmentationThreshold)
                {
                    cls = c;
                    max_prob = prob;
                }
            }
        }
    }
    return NVDSPOSTPROCESS_SUCCESS;
}

void
SegmentationModelPostProcessor::attachMetadata (
    NvBufSurface *surf, gint batch_idx,
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
    gboolean symmetric_padding)
{
  NvDsPostProcessSegmentationOutput &segmentation_output = detection_output.segmentationOutput;
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
  NvDsInferSegmentationMeta *meta =
    (NvDsInferSegmentationMeta *) g_malloc (sizeof (NvDsInferSegmentationMeta));

  if (frame_meta)
    frame_meta->bInferDone = TRUE;
  meta->classes = segmentation_output.classes;
  meta->width = segmentation_output.width;
  meta->height = segmentation_output.height;
  meta->class_map = segmentation_output.class_map;
  meta->class_probabilities_map = segmentation_output.class_probability_map;
  meta->priv_data =  NULL;

  user_meta->user_meta_data = meta;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDSINFER_SEGMENTATION_META;
  user_meta->base_meta.release_func = release_segmentation_meta;
  user_meta->base_meta.copy_func = copy_segmentation_meta;

  if (process_full_frame == PROCESS_MODEL_FULL_FRAME) {
    nvds_add_user_meta_to_frame (frame_meta, user_meta);
  } else {
    nvds_add_user_meta_to_obj (object_meta, user_meta);
  }
}

void
SegmentationModelPostProcessor::releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsPostProcessNetworkType_Segmentation:
          //Release if meta not attached
            //delete[] frameOutput.segmentationOutput.class_map;
            break;
        default:
            break;
    }
}

