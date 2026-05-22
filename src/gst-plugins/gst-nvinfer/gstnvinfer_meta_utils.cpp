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

#include <cmath>
#include <cstring>
#include "gstnvinfer_meta_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline int
get_element_size (NvDsInferDataType data_type)
{
  switch (data_type) {
    case FLOAT:
      return 4;
    case HALF:
      return 2;
    case INT32:
      return 4;
    case UINT8:
    case INT8:
      return 1;
    case INT64:
      return 8;
    default:
      return 0;
  }
}

/**
 * Attach metadata for the detector. We will be adding a new metadata.
 */
void
attach_metadata_detector (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
    GstNvInferFrame & frame, NvDsInferDetectionOutput & detection_output, float segmentationThreshold)
{
  static gchar font_name[] = "Serif";
  NvDsObjectMeta *obj_meta = NULL;
  NvDsObjectMeta *parent_obj_meta = frame.obj_meta; /* This will be  NULL in case of primary detector */
  NvDsFrameMeta *frame_meta = frame.frame_meta;
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  nvds_acquire_meta_lock (batch_meta);

  frame_meta->bInferDone = TRUE;
  /* Iterate through the inference output for one frame and attach the detected
   * bnounding boxes. */
  srand( (unsigned int)0);
  for (guint i = 0; i < detection_output.numObjects; i++) {
    NvDsInferObject & obj = detection_output.objects[i];

    float rotation_angle_rad = obj.rotation_angle;

    GstNvInferDetectionFilterParams & filter_params =
        (*nvinfer->perClassDetectionFilterParams)[obj.classIndex];

    /* Scale the bounding boxes proportionally based on how the object/frame was
     * scaled during input. */
    obj.left = (obj.left - frame.offset_left)/frame.scale_ratio_x + frame.roi_left;
    obj.top = (obj.top - frame.offset_top)/frame.scale_ratio_y + frame.roi_top;
    obj.width /= frame.scale_ratio_x;
    obj.height /= frame.scale_ratio_y;

    /** Clipping the object bounding-box which lies outside the roi
     * specified by nvdspreprosess plugin. */
    if(nvinfer->input_tensor_from_meta && nvinfer->clip_object_outside_roi) {
      if (obj.left + obj.width > frame.roi_meta->roi.left + frame.roi_meta->roi.width) {
        obj.width = frame.roi_meta->roi.left + frame.roi_meta->roi.width - obj.left;
        // Ensure width doesn't go negative due to calculation errors
        if (obj.width < 0) obj.width = 0;
      }
      if (obj.top + obj.height > frame.roi_meta->roi.top + frame.roi_meta->roi.height) {
        obj.height = frame.roi_meta->roi.top + frame.roi_meta->roi.height - obj.top;
        // Ensure height doesn't go negative due to calculation errors
        if (obj.height < 0) obj.height = 0;
      }
      if (obj.left < frame.roi_meta->roi.left) {
        float original_left = obj.left;
        obj.left = frame.roi_meta->roi.left;
        obj.width = obj.width - (obj.left - original_left);
        // Ensure width doesn't go negative due to calculation errors
        if (obj.width < 0) obj.width = 0;
      }
      if (obj.top < frame.roi_meta->roi.top) {
        float original_top = obj.top;
        obj.top = frame.roi_meta->roi.top;
        obj.height = obj.height - (obj.top - original_top);
        // Ensure height doesn't go negative due to calculation errors
        if (obj.height < 0) obj.height = 0;
      }
    }

    /* top and left should not be less than 0 */
    if(obj.top < 0)
      obj.top = 0;
    if(obj.left < 0)
      obj.left = 0;

    /* Check if the scaled box co-ordinates meet the detection filter criteria.
     * Skip the box if it does not. */
    if(nvinfer->filter_out_class_ids->find(obj.classIndex) != nvinfer->filter_out_class_ids->end())
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
    if(nvinfer->crop_objects_to_roi_boundary){
      if (obj.top < filter_params.roiTopOffset)
        obj.top = filter_params.roiTopOffset;
      if (obj.left + obj.width >= frame.input_surf_params->width)
        obj.width = frame.input_surf_params->width - obj.left;
      if (obj.top + obj.height > (frame.input_surf_params->height - filter_params.roiBottomOffset))
        obj.height = frame.input_surf_params->height - filter_params.roiBottomOffset - obj.top;
    } else {
      if (obj.top < filter_params.roiTopOffset)
        continue;
      if (obj.top + obj.height > (frame.input_surf_params->height - filter_params.roiBottomOffset))
        continue;
    }

    obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);

    obj_meta->unique_component_id = nvinfer->unique_id;
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

    /* Set rotation angle (converted to degrees for OSD) */
    rect_params.rotation_angle = rotation_angle_rad * 57.2957795f;

    if(!nvinfer->process_full_frame && !nvinfer->input_tensor_from_meta) {
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
    if (obj.classIndex > (gint) nvinfer->perClassColorParams->size()) {
      rect_params.has_bg_color = 0;
      rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
    } else {
      GstNvInferColorParams &color_params =
          (*nvinfer->perClassColorParams)[obj.classIndex];
      rect_params.has_bg_color = color_params.have_bg_color;
      rect_params.bg_color = color_params.bg_color;
      rect_params.border_color = color_params.border_color;
    }

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

    if (nvinfer->output_instance_mask && obj.mask) {
      float *mask = (float *)g_malloc(obj.mask_size);
      memcpy(mask, obj.mask, obj.mask_size);
      obj_meta->mask_params.data = mask;
      obj_meta->mask_params.size = obj.mask_size;
      obj_meta->mask_params.threshold = segmentationThreshold;
      obj_meta->mask_params.width = obj.mask_width;
      obj_meta->mask_params.height = obj.mask_height;
      rect_params.border_color = (NvOSD_ColorParams) {(float) rand()/RAND_MAX, (float) rand()/RAND_MAX, (float) rand()/RAND_MAX, 1};
    }

    nvds_add_obj_meta_to_frame (frame_meta, obj_meta, parent_obj_meta);
  }
  nvds_release_meta_lock (batch_meta);
}

/**
 * Update string label in an existing object metadata. If processing on full
 * frames, need to attach a new metadata. Assume only one label per object is generated.
 */
void
attach_metadata_classifier (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
    GstNvInferFrame & frame, GstNvInferObjectInfo & object_info)
{
  NvDsObjectMeta *object_meta = frame.obj_meta;
  NvDsBatchMeta *batch_meta = (nvinfer->process_full_frame) ?
    frame.frame_meta->base_meta.batch_meta : object_meta->base_meta.batch_meta;

  if(frame.frame_meta)
    frame.frame_meta->bInferDone = TRUE;

  if (object_info.attributes.size () == 0 ||
          object_info.label.length() == 0)
    return;

  nvds_acquire_meta_lock (batch_meta);
  if (nvinfer->process_full_frame && !nvinfer->input_tensor_from_meta) {

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
    rect_params.width = frame.input_surf_params->width;
    rect_params.height = frame.input_surf_params->height;

    /* Semi-transparent yellow background. */
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    /* Red border of width 6. */
    rect_params.border_width = 6;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    object_meta->unique_component_id = nvinfer->unique_id;
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
    nvds_add_obj_meta_to_frame (frame.frame_meta, object_meta, NULL);
  }

  std::string string_label = object_info.label;

  /* Fill the attribute info structure for the object. */
  guint num_attrs = object_info.attributes.size ();

  NvDsClassifierMeta *classifier_meta =
      nvds_acquire_classifier_meta_from_pool (batch_meta);

  classifier_meta->unique_component_id = nvinfer->unique_id;
  classifier_meta->classifier_type = nvinfer->classifier_type;

  for (unsigned int i = 0; i < num_attrs; i++) {
    NvDsLabelInfo *label_info =
        nvds_acquire_label_info_meta_from_pool (batch_meta);
    NvDsInferAttribute &attr = object_info.attributes[i];
    label_info->label_id = attr.attributeIndex;
    label_info->result_class_id = attr.attributeValue;
    label_info->result_prob = attr.attributeConfidence;
    if (attr.attributeLabel) {
      g_strlcpy (label_info->result_label, attr.attributeLabel, MAX_LABEL_SIZE);
      if (object_info.label.length() == 0)
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
    temp = object_meta->text_params.display_text;
    object_meta->text_params.display_text =
        g_strconcat (temp, " ", string_label.c_str (), nullptr);
    g_free (temp);
  }
  if (nvinfer->input_tensor_from_meta) {
    nvds_add_classifier_meta_to_roi (frame.roi_meta, classifier_meta);
    /* if object is roi itself */
    if (object_meta) {
      nvds_add_classifier_meta_to_object (object_meta, classifier_meta);
    }
  } else {
    nvds_add_classifier_meta_to_object (object_meta, classifier_meta);
  }
  nvds_release_meta_lock (batch_meta);
}


/**
 * Given an object's history, merge the new classification results with the
 * previous cached results. This can be used to improve the results of
 * classification when reinferencing over time. Currently, the function
 * just uses the latest results.
 */
void
merge_classification_output (GstNvInferObjectHistory & history,
    GstNvInferObjectInfo &new_result)
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

  meta->unique_id = src_meta->unique_id;
  meta->classes = src_meta->classes;
  meta->width = src_meta->width;
  meta->height = src_meta->height;
  meta->class_map = (gint *) g_memdup2(src_meta->class_map, meta->width * meta->height * sizeof (gint));
  meta->class_probabilities_map = (gfloat *) g_memdup2(src_meta->class_probabilities_map, meta->classes * meta->width * meta->height * sizeof (gfloat));
  meta->priv_data = NULL;

  return meta;
}

void
attach_metadata_segmentation (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
    GstNvInferFrame & frame, NvDsInferSegmentationOutput & segmentation_output)
{
  NvDsBatchMeta *batch_meta = (nvinfer->process_full_frame) ?
    frame.frame_meta->base_meta.batch_meta : frame.obj_meta->base_meta.batch_meta;

  if(frame.frame_meta)
    frame.frame_meta->bInferDone = TRUE;

  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
  NvDsInferSegmentationMeta *meta = (NvDsInferSegmentationMeta *) g_malloc (sizeof (NvDsInferSegmentationMeta));

  meta->unique_id = nvinfer->unique_id;
  meta->classes = segmentation_output.classes;
  meta->width = segmentation_output.width;
  meta->height = segmentation_output.height;
  meta->class_map = segmentation_output.class_map;
  meta->class_probabilities_map = segmentation_output.class_probability_map;
  meta->priv_data = gst_mini_object_ref (tensor_out_object);

  user_meta->user_meta_data = meta;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDSINFER_SEGMENTATION_META;
  user_meta->base_meta.release_func = release_segmentation_meta;
  user_meta->base_meta.copy_func = copy_segmentation_meta;

  if (nvinfer->input_tensor_from_meta) {
    nvds_add_user_meta_to_roi (frame.roi_meta, user_meta);
  } else if (nvinfer->process_full_frame) {
    nvds_add_user_meta_to_frame (frame.frame_meta, user_meta);
  } else {
    nvds_add_user_meta_to_obj (frame.obj_meta, user_meta);
  }
}

/* Called when NvDsUserMeta for each frame/object is released. Reduce the
 * refcount of the mini_object by 1 and free other memory. */
static void
release_tensor_output_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if (!user_meta || !user_meta->user_meta_data) {
    return;
  }

  NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
  if (!meta) {
    return;
  }

  // Free output layers info if it exists and is valid
  if (meta->output_layers_info && meta->num_output_layers > 0) {
    g_free(meta->output_layers_info);
    meta->output_layers_info = nullptr;
  }

  // Unref the mini object if it exists and is valid
  if (meta->priv_data) {
    GstMiniObject *mini_obj = GST_MINI_OBJECT(meta->priv_data);
    if (mini_obj && mini_obj->refcount > 0) {
      gst_mini_object_unref(mini_obj);
    }
    meta->priv_data = nullptr;
  }

  // Free device and host buffer pointers if they exist and are valid
  if (meta->out_buf_ptrs_dev && meta->num_output_layers > 0) {
    delete[] meta->out_buf_ptrs_dev;
    meta->out_buf_ptrs_dev = nullptr;
  }

  if (meta->out_buf_ptrs_host && meta->num_output_layers > 0) {
    delete[] meta->out_buf_ptrs_host;
    meta->out_buf_ptrs_host = nullptr;
  }

  // Finally delete the meta structure
  delete meta;
  user_meta->user_meta_data = nullptr;
}

static gpointer
copy_tensor_output_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *src_user_meta = (NvDsUserMeta *) data;
  if (!src_user_meta || !src_user_meta->user_meta_data) {
    return nullptr;
  }

  NvDsInferTensorMeta *src_meta = (NvDsInferTensorMeta *) src_user_meta->user_meta_data;
  NvDsInferTensorMeta *tensor_output_meta = new NvDsInferTensorMeta();
  if (!tensor_output_meta) {
    return nullptr;
  }

  // Copy basic fields
  tensor_output_meta->unique_id = src_meta->unique_id;
  tensor_output_meta->num_output_layers = src_meta->num_output_layers;
  tensor_output_meta->gpu_id = src_meta->gpu_id;
  tensor_output_meta->network_info = src_meta->network_info;
  tensor_output_meta->maintain_aspect_ratio = src_meta->maintain_aspect_ratio;
  tensor_output_meta->symmetric_padding = src_meta->symmetric_padding;

  // Copy output layers info if it exists
  if (src_meta->output_layers_info && src_meta->num_output_layers > 0) {
    size_t layer_info_size = src_meta->num_output_layers * sizeof(NvDsInferLayerInfo);
    if (layer_info_size > 0 && layer_info_size < (1024 * 1024 * 1024)) { // Sanity check: less than 1GB
      tensor_output_meta->output_layers_info = (NvDsInferLayerInfo*)g_memdup2(
          src_meta->output_layers_info,
          layer_info_size);
    }
  }

  // Allocate and copy buffer pointers if needed
  if (src_meta->num_output_layers > 0 && src_meta->num_output_layers < 1024) { // Sanity check: reasonable number of layers
    tensor_output_meta->out_buf_ptrs_host = new void*[src_meta->num_output_layers];
    tensor_output_meta->out_buf_ptrs_dev = new void*[src_meta->num_output_layers];

    if (tensor_output_meta->out_buf_ptrs_host && tensor_output_meta->out_buf_ptrs_dev) {
      for (unsigned int i = 0; i < src_meta->num_output_layers; i++) {
        if (src_meta->output_layers_info && i < src_meta->num_output_layers) {
          NvDsInferLayerInfo *info = &tensor_output_meta->output_layers_info[i];
          tensor_output_meta->out_buf_ptrs_host[i] = src_meta->out_buf_ptrs_host[i];
          tensor_output_meta->out_buf_ptrs_dev[i] = src_meta->out_buf_ptrs_dev[i];
          info->buffer = tensor_output_meta->out_buf_ptrs_host[i];
        }
      }
    } else {
      // Clean up on allocation failure
      if (tensor_output_meta->out_buf_ptrs_host) {
        delete[] tensor_output_meta->out_buf_ptrs_host;
      }
      if (tensor_output_meta->out_buf_ptrs_dev) {
        delete[] tensor_output_meta->out_buf_ptrs_dev;
      }
      delete tensor_output_meta;
      return nullptr;
    }
  }

  // Handle GStreamer mini object reference counting
  if (src_meta->priv_data) {
    GstMiniObject *mini_obj = GST_MINI_OBJECT(src_meta->priv_data);
    if (mini_obj && mini_obj->refcount > 0) {
      gst_mini_object_ref(mini_obj);
      tensor_output_meta->priv_data = mini_obj;
    } else {
      tensor_output_meta->priv_data = src_meta->priv_data;
    }
  }

  return tensor_output_meta;
}

static void
release_user_meta_at_frame_level (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if (!user_meta || !user_meta->user_meta_data) {
    return;
  }
  NvDsRoiMeta *roi_meta = (NvDsRoiMeta *) user_meta->user_meta_data;
  if (!roi_meta) {
    return;
  }
  // Free classifier meta list if it exists
  if (roi_meta->classifier_meta_list) {
    GList *l_classifier = roi_meta->classifier_meta_list;
    while (l_classifier) {
      NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *) l_classifier->data;
      if (classifier_meta) {
        if (classifier_meta->label_info_list) {
          g_list_free(classifier_meta->label_info_list);
          classifier_meta->label_info_list = nullptr;
        }
        g_free(classifier_meta);
      }
      l_classifier = l_classifier->next;
    }
    g_list_free(roi_meta->classifier_meta_list);
    roi_meta->classifier_meta_list = nullptr;
  }
  // Free ROI user meta list if it exists
  if (roi_meta->roi_user_meta_list) {
    GList *l_user = roi_meta->roi_user_meta_list;
    while (l_user) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta && user_meta->user_meta_data && user_meta->base_meta.release_func) {
        user_meta->base_meta.release_func(user_meta->user_meta_data, nullptr);
      }
      l_user = l_user->next;
    }
    g_list_free(roi_meta->roi_user_meta_list);
    roi_meta->roi_user_meta_list = nullptr;
  }
  // Free object meta if it exists
  if (roi_meta->object_meta) {
    delete roi_meta->object_meta;
    roi_meta->object_meta = nullptr;
  }
  // Free ROI meta itself
  if (roi_meta) {
    delete roi_meta;
  }
  user_meta->user_meta_data = nullptr;
}

static gpointer
copy_roi_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *src_user_meta = (NvDsUserMeta *) data;
  if (!src_user_meta || !src_user_meta->user_meta_data) {
    return nullptr;
  }

  NvDsRoiMeta *src_meta = (NvDsRoiMeta *) src_user_meta->user_meta_data;
  NvDsRoiMeta *dst_meta = (NvDsRoiMeta *) g_malloc0(sizeof(NvDsRoiMeta));
  if (!dst_meta) {
    return nullptr;
  }

  // Copy ROI coordinates
  dst_meta->roi = src_meta->roi;
  memcpy(dst_meta->roi_polygon, src_meta->roi_polygon, 
      sizeof(guint) * DS_MAX_POLYGON_POINTS * 2);
  dst_meta->converted_buffer = src_meta->converted_buffer;
  dst_meta->frame_meta = src_meta->frame_meta;
  dst_meta->scale_ratio_x = src_meta->scale_ratio_x;
  dst_meta->scale_ratio_y = src_meta->scale_ratio_y;
  dst_meta->offset_left = src_meta->offset_left;
  dst_meta->offset_top = src_meta->offset_top;
  dst_meta->object_meta = src_meta->object_meta;

  // Initialize lists as NULL
  dst_meta->classifier_meta_list = nullptr;
  dst_meta->roi_user_meta_list = nullptr;

  // Copy classifier meta list if it exists
  if (src_meta->classifier_meta_list) {
    GList *l_classifier = nullptr;
    for (l_classifier = src_meta->classifier_meta_list; l_classifier != nullptr; l_classifier = l_classifier->next) {
      if (!l_classifier->data) {
        continue;
      }
      NvDsClassifierMeta *src_classifier_meta = (NvDsClassifierMeta *) l_classifier->data;
      if (src_classifier_meta) {
        NvDsClassifierMeta *dst_classifier_meta = (NvDsClassifierMeta *) g_malloc0(sizeof(NvDsClassifierMeta));
        if (dst_classifier_meta) {
          dst_classifier_meta->unique_component_id = src_classifier_meta->unique_component_id;
          dst_classifier_meta->classifier_type = src_classifier_meta->classifier_type;
          dst_classifier_meta->label_info_list = nullptr;
          dst_meta->classifier_meta_list = g_list_append(dst_meta->classifier_meta_list, dst_classifier_meta);
        }
      }
    }
  }

  // Copy ROI user meta list if it exists
  if (src_meta->roi_user_meta_list) {
    GList *l_user = nullptr;
    for (l_user = src_meta->roi_user_meta_list; l_user != nullptr; l_user = l_user->next) {
      if (!l_user->data) {
        continue;
      }
      NvDsUserMeta *src_user_meta = (NvDsUserMeta *) l_user->data;
      if (src_user_meta && src_user_meta->user_meta_data && src_user_meta->base_meta.copy_func) {
        NvDsUserMeta *dst_user_meta = (NvDsUserMeta *) g_malloc0(sizeof(NvDsUserMeta));
        if (dst_user_meta) {
          // Copy the user meta data using the copy function
          dst_user_meta->user_meta_data = src_user_meta->base_meta.copy_func(src_user_meta->user_meta_data, nullptr);
          if (dst_user_meta->user_meta_data) {
            dst_user_meta->base_meta.meta_type = src_user_meta->base_meta.meta_type;
            dst_user_meta->base_meta.release_func = src_user_meta->base_meta.release_func;
            dst_user_meta->base_meta.copy_func = src_user_meta->base_meta.copy_func;
            dst_meta->roi_user_meta_list = g_list_append(dst_meta->roi_user_meta_list, dst_user_meta);
          } else {
            g_free(dst_user_meta);
          }
        }
      }
    }
  }

  return dst_meta;
}

/* Attaches the raw tensor output to the GstBuffer as metadata. */
void
attach_tensor_output_meta (GstNvInfer *nvinfer, GstMiniObject * tensor_out_object,
    GstNvInferBatch *batch, NvDsInferContextBatchOutput *batch_output)
{
  NvDsBatchMeta *batch_meta = (nvinfer->process_full_frame
      || nvinfer->input_tensor_from_meta) ? batch->frames[0].
      frame_meta->base_meta.batch_meta : batch->frames[0].obj_meta->base_meta.
      batch_meta;

  /* Create and attach NvDsInferTensorMeta for each frame/object. Also
   * increment the refcount of GstNvInferTensorOutputObject. */
  for (size_t j = 0; j < batch->frames.size (); j++) {
    GstNvInferFrame & frame = batch->frames[j];

    /* Processing on ROIs (not frames or objects) skip attaching tensor output
     * to frames or objects. */
    NvDsInferTensorMeta *meta = new NvDsInferTensorMeta;
    meta->unique_id = nvinfer->unique_id;
    meta->num_output_layers = nvinfer->output_layers_info->size ();
    meta->output_layers_info = (NvDsInferLayerInfo*)g_memdup2(nvinfer->output_layers_info->data (),
    meta->num_output_layers * sizeof (NvDsInferLayerInfo)) ;
    meta->out_buf_ptrs_host = new void *[meta->num_output_layers];
    meta->out_buf_ptrs_dev = new void *[meta->num_output_layers];
    meta->gpu_id = nvinfer->gpu_id;
    meta->priv_data = gst_mini_object_ref (tensor_out_object);
    meta->network_info = nvinfer->network_info;
    meta->maintain_aspect_ratio = nvinfer->maintain_aspect_ratio;
    meta->symmetric_padding = nvinfer->symmetric_padding;

    for (unsigned int i = 0; i < meta->num_output_layers; i++) {
      NvDsInferLayerInfo & info = meta->output_layers_info[i];
      meta->out_buf_ptrs_dev[i] =
          (uint8_t *) batch_output->outputDeviceBuffers[i] +
          info.inferDims.numElements * get_element_size (info.dataType) * j;
      meta->out_buf_ptrs_host[i] =
          (uint8_t *) batch_output->hostBuffers[info.bindingIndex] +
          info.inferDims.numElements * get_element_size (info.dataType) * j;
    }

    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
    user_meta->user_meta_data = meta;
    user_meta->base_meta.meta_type =
        (NvDsMetaType) NVDSINFER_TENSOR_OUTPUT_META;
    user_meta->base_meta.release_func = release_tensor_output_meta;
    user_meta->base_meta.copy_func = copy_tensor_output_meta;
    user_meta->base_meta.batch_meta = batch_meta;

    if (nvinfer->input_tensor_from_meta) {
      /* Attach tensor meta as part of ROI Meta */
      nvds_add_user_meta_to_roi (frame.roi_meta, user_meta);

      /* Create a new user meta for ROI meta at frame level */
      NvDsUserMeta *roi_user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
      NvDsRoiMeta *frame_roi_meta = new NvDsRoiMeta;
      // Copy ROI data
      frame_roi_meta->roi = frame.roi_meta->roi;
      memcpy(frame_roi_meta->roi_polygon, frame.roi_meta->roi_polygon,
          sizeof(guint) * DS_MAX_POLYGON_POINTS * 2);
      frame_roi_meta->converted_buffer = frame.roi_meta->converted_buffer;
      frame_roi_meta->frame_meta = frame.roi_meta->frame_meta;
      frame_roi_meta->scale_ratio_x = frame.roi_meta->scale_ratio_x;
      frame_roi_meta->scale_ratio_y = frame.roi_meta->scale_ratio_y;
      frame_roi_meta->offset_left = frame.roi_meta->offset_left;
      frame_roi_meta->offset_top = frame.roi_meta->offset_top;
      frame_roi_meta->object_meta = nullptr;
      frame_roi_meta->classifier_meta_list = nullptr;
      frame_roi_meta->roi_user_meta_list = nullptr;

      roi_user_meta->user_meta_data = frame_roi_meta;
      roi_user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_ROI_META;
      roi_user_meta->base_meta.release_func = release_user_meta_at_frame_level;
      roi_user_meta->base_meta.copy_func = copy_roi_meta;
      roi_user_meta->base_meta.batch_meta = batch_meta;

      /* Add ROI meta to frame meta */
      nvds_add_user_meta_to_frame (frame.frame_meta, roi_user_meta);

      /* If object is ROI itself, add ROI meta to object meta */
      if (frame.obj_meta) {
        NvDsUserMeta *obj_roi_user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
        NvDsRoiMeta *obj_roi_meta = new NvDsRoiMeta;
        // Copy ROI data
        obj_roi_meta->roi = frame.roi_meta->roi;
        memcpy(obj_roi_meta->roi_polygon, frame.roi_meta->roi_polygon,
            sizeof(guint) * DS_MAX_POLYGON_POINTS * 2);
        obj_roi_meta->converted_buffer = frame.roi_meta->converted_buffer;
        obj_roi_meta->frame_meta = frame.roi_meta->frame_meta;
        obj_roi_meta->scale_ratio_x = frame.roi_meta->scale_ratio_x;
        obj_roi_meta->scale_ratio_y = frame.roi_meta->scale_ratio_y;
        obj_roi_meta->offset_left = frame.roi_meta->offset_left;
        obj_roi_meta->offset_top = frame.roi_meta->offset_top;
        obj_roi_meta->object_meta = nullptr;
        obj_roi_meta->classifier_meta_list = nullptr;
        obj_roi_meta->roi_user_meta_list = nullptr;

        obj_roi_user_meta->user_meta_data = obj_roi_meta;
        obj_roi_user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_ROI_META;
        obj_roi_user_meta->base_meta.release_func = release_user_meta_at_frame_level;
        obj_roi_user_meta->base_meta.copy_func = nullptr;
        obj_roi_user_meta->base_meta.batch_meta = batch_meta;
        nvds_add_user_meta_to_obj (frame.obj_meta, obj_roi_user_meta);
      }
    } else if (nvinfer->process_full_frame) {
      nvds_add_user_meta_to_frame (frame.frame_meta, user_meta);
    } else {
      nvds_add_user_meta_to_obj (frame.obj_meta, user_meta);
    }
  }
}
