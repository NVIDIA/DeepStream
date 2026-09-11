/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvdsmeta_internal.h"
#include "nvdsmeta.h"
#include <stdio.h>
#include <string.h>

/* Copy functions */
NvDsBatchMeta *batch_meta_copy(NvDsBatchMeta *src_batch_meta,
    gpointer user_data)
{
  /* g_print("\n\n\n*******Get Src batch meta Info************\n");
  nvds_get_current_metadata_info(src_batch_meta); */

  NvDsBatchMeta *dst_batch_meta = nvds_create_batch_meta(
      src_batch_meta->max_frames_in_batch);
  dst_batch_meta->max_frames_in_batch = src_batch_meta->max_frames_in_batch;
  memcpy(dst_batch_meta->misc_batch_info, src_batch_meta->misc_batch_info,
      sizeof(gint64) * MAX_USER_FIELDS);
  memcpy(dst_batch_meta->reserved, src_batch_meta->reserved, sizeof(gint64) *
      MAX_USER_FIELDS);

  nvds_copy_frame_meta_list(src_batch_meta->frame_meta_list, dst_batch_meta);

  nvds_copy_batch_user_meta_list( src_batch_meta->batch_user_meta_list,
      dst_batch_meta);

  /* g_print("\n\n\n******Get Dst batch meta Info************\n");
  nvds_get_current_metadata_info(dst_batch_meta); */

  return dst_batch_meta;
}

void nvds_copy_frame_meta_list (NvDsFrameMetaList *src_frame_meta_list,
    NvDsBatchMeta *dst_batch_meta)
{
  NvDsMetaList *l = NULL;
  NvDsFrameMeta *src_frame_meta = NULL;
  NvDsFrameMeta *dst_frame_meta = NULL;

  for (l = src_frame_meta_list; l != NULL; l = l->next) {
    src_frame_meta = (NvDsFrameMeta *)(l->data);
    dst_frame_meta = nvds_acquire_frame_meta_from_pool (dst_batch_meta);
    nvds_copy_frame_meta(src_frame_meta, dst_frame_meta);
    nvds_add_frame_meta_to_batch(dst_batch_meta, dst_frame_meta);
  }
}

void nvds_copy_frame_meta(NvDsFrameMeta *src_frame_meta,
    NvDsFrameMeta *dst_frame_meta)
{
  NvDsBaseMeta src_base_meta = src_frame_meta->base_meta;
  NvDsBaseMeta dst_base_meta = dst_frame_meta->base_meta;
  NvDsBatchMeta *src_batch_meta = src_base_meta.batch_meta;
  NvDsBatchMeta *dst_batch_meta = dst_base_meta.batch_meta;

  nvds_acquire_meta_lock (src_batch_meta);
  dst_frame_meta->base_meta.batch_meta = dst_batch_meta;
  dst_frame_meta->base_meta.meta_type = src_frame_meta->base_meta.meta_type;
  dst_frame_meta->base_meta.copy_func = src_frame_meta->base_meta.copy_func;
  dst_frame_meta->base_meta.release_func =
    src_frame_meta->base_meta.release_func;

  dst_frame_meta->source_id = src_frame_meta->source_id;
  dst_frame_meta->pad_index = src_frame_meta->pad_index;
  dst_frame_meta->frame_num = src_frame_meta->frame_num;
  dst_frame_meta->buf_pts = src_frame_meta->buf_pts;
  dst_frame_meta->ntp_timestamp = src_frame_meta->ntp_timestamp;
  dst_frame_meta->surface_type = src_frame_meta->surface_type;
  dst_frame_meta->surface_index = src_frame_meta->surface_index;
  dst_frame_meta->bInferDone = src_frame_meta->bInferDone;
  dst_frame_meta->source_frame_width = src_frame_meta->source_frame_width;
  dst_frame_meta->source_frame_height = src_frame_meta->source_frame_height;
  dst_frame_meta->batch_id = src_frame_meta->batch_id;
  dst_frame_meta->num_surfaces_per_frame = src_frame_meta->num_surfaces_per_frame;
  dst_frame_meta->sensorInfo_meta.source_id = src_frame_meta->sensorInfo_meta.source_id;
  if(src_frame_meta->sensorInfo_meta.sensor_id)
  {
    dst_frame_meta->sensorInfo_meta.sensor_id = g_strdup(src_frame_meta->sensorInfo_meta.sensor_id);
  }
  if(src_frame_meta->sensorInfo_meta.sensor_name)
  {
    dst_frame_meta->sensorInfo_meta.sensor_name = g_strdup(src_frame_meta->sensorInfo_meta.sensor_name);
  }
  if(src_frame_meta->sensorInfo_meta.uri)
  {
    dst_frame_meta->sensorInfo_meta.uri = g_strdup(src_frame_meta->sensorInfo_meta.uri);
  }

  memcpy(dst_frame_meta->misc_frame_info, src_frame_meta->misc_frame_info,
      sizeof(gint64) * MAX_USER_FIELDS);
  memcpy(dst_frame_meta->reserved, src_frame_meta->reserved,
      sizeof(gint64) * MAX_USER_FIELDS);

  nvds_release_meta_lock (src_batch_meta);

  nvds_copy_obj_meta_list(src_frame_meta->obj_meta_list, dst_frame_meta);

  nvds_copy_display_meta_list(src_frame_meta->display_meta_list, dst_frame_meta);

  nvds_copy_frame_user_meta_list(src_frame_meta->frame_user_meta_list,
      dst_frame_meta);
}

void nvds_copy_obj_meta_list(NvDsObjectMetaList *src_obj_meta_list,
    NvDsFrameMeta *dst_frame_meta)
{
  NvDsMetaList *l = NULL;
  NvDsObjectMeta *src_object_meta = NULL;
  NvDsObjectMeta *dst_object_meta = NULL;
  NvDsBatchMeta *dst_batch_meta = dst_frame_meta->base_meta.batch_meta;
  GHashTable *parent_obj_map = g_hash_table_new (NULL, NULL);

    /* Copy frame meta list */
  for (l = src_obj_meta_list; l != NULL; l = l->next) {
    src_object_meta = (NvDsObjectMeta *)(l->data);
    dst_object_meta = nvds_acquire_obj_meta_from_pool (dst_batch_meta);
    nvds_copy_obj_meta(src_object_meta, dst_object_meta);
    nvds_add_obj_meta_to_frame(dst_frame_meta, dst_object_meta, NULL);
    g_hash_table_insert (parent_obj_map, src_object_meta, dst_object_meta);
    if (src_object_meta->parent) {
      dst_object_meta->parent = g_hash_table_lookup (parent_obj_map,
              src_object_meta->parent);
    }
  }
  g_hash_table_unref (parent_obj_map);
}

void nvds_copy_obj_meta(NvDsObjectMeta *src_object_meta,
    NvDsObjectMeta *dst_object_meta)
{
  NvDsBaseMeta src_base_meta = src_object_meta->base_meta;
  NvDsBaseMeta dst_base_meta = dst_object_meta->base_meta;
  NvDsBatchMeta *src_batch_meta = src_base_meta.batch_meta;
  NvDsBatchMeta *dst_batch_meta = dst_base_meta.batch_meta;

  nvds_acquire_meta_lock (src_batch_meta);

  dst_object_meta->base_meta.batch_meta = dst_batch_meta;
  dst_object_meta->base_meta.meta_type = src_object_meta->base_meta.meta_type;
  dst_object_meta->base_meta.copy_func = src_object_meta->base_meta.copy_func;
  dst_object_meta->base_meta.release_func =
    src_object_meta->base_meta.release_func;

  dst_object_meta->unique_component_id = src_object_meta->unique_component_id;
  dst_object_meta->class_id = src_object_meta->class_id;
  dst_object_meta->object_id = src_object_meta->object_id;
  dst_object_meta->detector_bbox_info = src_object_meta->detector_bbox_info;
  dst_object_meta->tracker_bbox_info = src_object_meta->tracker_bbox_info;
  dst_object_meta->confidence = src_object_meta->confidence;
  dst_object_meta->tracker_confidence = src_object_meta->tracker_confidence;
  dst_object_meta->rect_params = src_object_meta->rect_params;
  dst_object_meta->text_params = src_object_meta->text_params;
  strncpy(dst_object_meta->obj_label, src_object_meta->obj_label, MAX_LABEL_SIZE);

  if(src_object_meta->text_params.display_text)
  {
    dst_object_meta->text_params.display_text = g_strdup(src_object_meta->text_params.display_text);
  }

  dst_object_meta->mask_params = src_object_meta->mask_params;
  if(src_object_meta->mask_params.size > 0 && src_object_meta->mask_params.data)
  {
    dst_object_meta->mask_params.data = g_memdup2(src_object_meta->mask_params.data,
                                                            src_object_meta->mask_params.size);
  }

  dst_object_meta->unique_component_id = src_object_meta->unique_component_id;

  memcpy(dst_object_meta->misc_obj_info, src_object_meta->misc_obj_info,
      sizeof(gint64) * MAX_USER_FIELDS);
  memcpy(dst_object_meta->reserved, src_object_meta->reserved,
      sizeof(gint64) * MAX_USER_FIELDS);

  nvds_release_meta_lock (src_batch_meta);

  nvds_copy_obj_user_meta_list(src_object_meta->obj_user_meta_list,
      dst_object_meta);

  nvds_copy_classification_list(src_object_meta->classifier_meta_list, dst_object_meta);
}

void nvds_copy_classification_list(NvDsClassifierMetaList *src_classifier_meta_list,
    NvDsObjectMeta *dst_object_meta)
{
  NvDsMetaList *l = NULL;
  NvDsClassifierMeta *src_classifier_meta = NULL;
  NvDsClassifierMeta *dst_classifier_meta = NULL;
  NvDsBatchMeta *dst_batch_meta = dst_object_meta->base_meta.batch_meta;

  //g_print("classifier_meta_list list len = %d\n", g_list_length(src_classifier_meta_list));
    /* Copy frame meta list */
  for (l = src_classifier_meta_list; l != NULL; l = l->next) {
    src_classifier_meta = (NvDsClassifierMeta *)(l->data);
    dst_classifier_meta = nvds_acquire_classifier_meta_from_pool (dst_batch_meta);
    nvds_copy_classifier_meta(src_classifier_meta, dst_classifier_meta);
    nvds_add_classifier_meta_to_object(dst_object_meta, dst_classifier_meta);
  }
}

void nvds_copy_classifier_meta(NvDsClassifierMeta *src_classifier_meta,
    NvDsClassifierMeta *dst_classifier_meta)
{
  NvDsBaseMeta src_base_meta = src_classifier_meta->base_meta;
  NvDsBaseMeta dst_base_meta = dst_classifier_meta->base_meta;
  NvDsBatchMeta *src_batch_meta = src_base_meta.batch_meta;
  NvDsBatchMeta *dst_batch_meta = dst_base_meta.batch_meta;

  nvds_acquire_meta_lock (src_batch_meta);

  dst_classifier_meta->base_meta.batch_meta = dst_batch_meta;
  dst_classifier_meta->base_meta.meta_type = src_classifier_meta->base_meta.meta_type;
  dst_classifier_meta->base_meta.copy_func = src_classifier_meta->base_meta.copy_func;
  dst_classifier_meta->base_meta.release_func =
    src_classifier_meta->base_meta.release_func;

  //dst_classifier_meta->num_labels = src_classifier_meta->num_labels;
  dst_classifier_meta->unique_component_id = src_classifier_meta->unique_component_id;

  nvds_release_meta_lock (src_batch_meta);

  nvds_copy_label_info_list(src_classifier_meta->label_info_list, dst_classifier_meta);

}

void nvds_copy_label_info_list(NvDsLabelInfoList *src_label_info_list,
    NvDsClassifierMeta *dst_classifier_meta)
{
  NvDsMetaList *l = NULL;
  NvDsLabelInfo *src_label_info = NULL;
  NvDsLabelInfo *dst_label_info = NULL;
  NvDsBatchMeta *dst_batch_meta = dst_classifier_meta->base_meta.batch_meta;

  /* Copy frame meta list */
  //g_print("label list len = %d\n", g_list_length(src_label_info_list));
  for (l = src_label_info_list; l != NULL; l = l->next) {
    src_label_info = (NvDsLabelInfo *)(l->data);
    dst_label_info = nvds_acquire_label_info_meta_from_pool (dst_batch_meta);
    nvds_copy_label_info_meta(src_label_info, dst_label_info);
    nvds_add_label_info_meta_to_classifier(dst_classifier_meta, dst_label_info);
  }
}

void nvds_copy_label_info_meta(NvDsLabelInfo *src_label_info,
    NvDsLabelInfo *dst_label_info)
{
  NvDsBaseMeta src_base_meta = src_label_info->base_meta;
  NvDsBaseMeta dst_base_meta = dst_label_info->base_meta;
  NvDsBatchMeta *src_batch_meta = src_base_meta.batch_meta;
  NvDsBatchMeta *dst_batch_meta = dst_base_meta.batch_meta;

  nvds_acquire_meta_lock (src_batch_meta);

  dst_label_info->base_meta.batch_meta = dst_batch_meta;
  dst_label_info->base_meta.meta_type = src_label_info->base_meta.meta_type;
  dst_label_info->base_meta.copy_func = src_label_info->base_meta.copy_func;
  dst_label_info->base_meta.release_func =
    src_label_info->base_meta.release_func;

  dst_label_info->num_classes = src_label_info->num_classes;

  memcpy(dst_label_info->result_label, src_label_info->result_label,
      MAX_LABEL_SIZE * sizeof(gchar));

  if(src_label_info->pResult_label)
  {
    dst_label_info->pResult_label = g_strdup(src_label_info->pResult_label);
  }

  dst_label_info->result_class_id = src_label_info->result_class_id;
  dst_label_info->label_id = src_label_info->label_id;
  dst_label_info->result_prob = src_label_info->result_prob;

  nvds_release_meta_lock (src_batch_meta);
}



#if 1
void nvds_copy_display_meta_list(NvDisplayMetaList *src_display_meta_list,
    NvDsFrameMeta *dst_frame_meta)
{
  NvDsMetaList *l = NULL;
  NvDsDisplayMeta *src_display_meta = NULL;
  NvDsDisplayMeta *dst_display_meta = NULL;
  NvDsBatchMeta *dst_batch_meta = dst_frame_meta->base_meta.batch_meta;

    /* Copy frame meta list */
  for (l = src_display_meta_list; l != NULL; l = l->next) {
    src_display_meta = (NvDsDisplayMeta *)(l->data);
    dst_display_meta = nvds_acquire_display_meta_from_pool (dst_batch_meta);
    nvds_copy_display_meta(src_display_meta, dst_display_meta);
    nvds_add_display_meta_to_frame(dst_frame_meta, dst_display_meta);
  }
}

void nvds_copy_display_meta(NvDsDisplayMeta *src_display_meta,
    NvDsDisplayMeta *dst_display_meta)
{
  int i = 0;
  NvDsBaseMeta src_base_meta = src_display_meta->base_meta;
  NvDsBaseMeta dst_base_meta = dst_display_meta->base_meta;
  NvDsBatchMeta *src_batch_meta = src_base_meta.batch_meta;
  NvDsBatchMeta *dst_batch_meta = dst_base_meta.batch_meta;

  nvds_acquire_meta_lock (src_batch_meta);
  memcpy(dst_display_meta, src_display_meta, sizeof(NvDsDisplayMeta));
  dst_display_meta->base_meta.batch_meta = dst_batch_meta;

  for(i = 0; i < MAX_ELEMENTS_IN_DISPLAY_META; i++)
  {
    if(src_display_meta->text_params[i].display_text)
    {
      dst_display_meta->text_params[i].display_text =
        g_strdup(src_display_meta->text_params[i].display_text);
    }
  }
  nvds_release_meta_lock (src_batch_meta);
}
#endif


void nvds_copy_batch_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsBatchMeta *dst_batch_meta)
{
  copy_user_meta_list(src_user_meta_list, (NvDsElementMeta *)dst_batch_meta,
      NVDS_BATCH_META);
}

void nvds_copy_frame_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsFrameMeta *dst_frame_meta)
{
  copy_user_meta_list(src_user_meta_list, (NvDsElementMeta *)dst_frame_meta,
      NVDS_FRAME_META);
}

void nvds_copy_obj_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsObjectMeta *dst_object_meta)
{
  copy_user_meta_list(src_user_meta_list, (NvDsElementMeta *)dst_object_meta,
      NVDS_OBJ_META);
}

void copy_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsElementMeta *element_meta, NvDsMetaType meta_type)
{
  NvDsUserMetaList *l = NULL;
  NvDsUserMeta *src_user_meta = NULL;
  NvDsUserMeta *dst_user_meta = NULL;
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsBatchMeta *dst_batch_meta = base_meta->batch_meta;

  for (l = src_user_meta_list; l != NULL; l = l->next) {
    src_user_meta = (NvDsUserMeta *)l->data;
    if (!src_user_meta->base_meta.copy_func)
        continue;

    dst_user_meta = nvds_acquire_user_meta_from_pool (dst_batch_meta);
    copy_user_meta(src_user_meta, dst_user_meta);

    if(meta_type == NVDS_BATCH_META)
    {
      nvds_add_user_meta_to_batch((NvDsBatchMeta *)element_meta, dst_user_meta);
    }
    if(meta_type == NVDS_FRAME_META)
    {
      nvds_add_user_meta_to_frame((NvDsFrameMeta *)element_meta, dst_user_meta);
    }
    if(meta_type == NVDS_OBJ_META)
    {
      nvds_add_user_meta_to_obj((NvDsObjectMeta *)element_meta, dst_user_meta);
    }
  }
}

void copy_user_meta(NvDsUserMeta *src_user_meta, NvDsUserMeta *dst_user_meta)
{
  NvDsBaseMeta *src_base_meta = (NvDsBaseMeta *)src_user_meta;
  NvDsBaseMeta *dst_base_meta = (NvDsBaseMeta *)dst_user_meta;
  dst_base_meta->meta_type = src_base_meta->meta_type;
  dst_base_meta->copy_func = src_base_meta->copy_func;
  dst_base_meta->release_func = src_base_meta->release_func;
  dst_base_meta->uContext = src_base_meta->uContext;
  dst_user_meta->user_meta_data = src_base_meta->copy_func(src_user_meta, NULL);
}
