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

//static void nvds_destroy_batch_user_meta_list(NvDsBatchMeta *batch_meta);

/* Acquire the lock before updating metadata */
void nvds_acquire_meta_lock (NvDsBatchMeta *batch_meta) {
  g_rec_mutex_lock (&batch_meta->meta_mutex);
}

/* Release the lock after updating metadata */
void nvds_release_meta_lock (NvDsBatchMeta *batch_meta) {
  g_rec_mutex_unlock (&batch_meta->meta_mutex);
}

/*************Batch Functions *****************/
NvDsBatchMeta *nvds_create_batch_meta(guint max_batch_size) {
  NvDsBatchMeta *batch_meta = NULL;
  gboolean ret = TRUE;

  batch_meta = (NvDsBatchMeta *)g_malloc0(sizeof(NvDsBatchMeta));
  g_rec_mutex_init(&batch_meta->meta_mutex);

  batch_meta->frame_meta_pool = nvds_create_frame_meta_pool(batch_meta,
      max_batch_size);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->obj_meta_pool = nvds_create_obj_meta_pool(batch_meta,
      8 * max_batch_size);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->classifier_meta_pool = nvds_create_classifier_meta_pool(
      batch_meta, NUM_CLASSIFIER_META * max_batch_size);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->display_meta_pool = nvds_create_display_meta_pool(batch_meta,
      NUM_DISPLAY_META);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->user_meta_pool = nvds_create_user_meta_pool(batch_meta,
      NUM_USER_META);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->label_info_meta_pool = nvds_create_label_info_meta_pool(
      batch_meta, NUM_LABEL_META * max_batch_size);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->base_meta.batch_meta = batch_meta;
  batch_meta->base_meta.meta_type = NVDS_BATCH_META;
  batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
  batch_meta->base_meta.release_func = nvds_batch_meta_release_func;

  return batch_meta;
}

gboolean nvds_destroy_batch_meta(NvDsBatchMeta *batch_meta) {
  gboolean ret = TRUE;

  ret = nvds_destroy_frame_meta_pool(batch_meta, batch_meta->frame_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  ret = nvds_destroy_obj_meta_pool(batch_meta, batch_meta->obj_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  ret = nvds_destroy_classifier_meta_pool(batch_meta,
      batch_meta->classifier_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  ret = nvds_destroy_display_meta_pool(batch_meta,
      batch_meta->display_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  ret = nvds_destroy_user_meta_pool(batch_meta, batch_meta->user_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  ret = nvds_destroy_label_info_meta_pool(batch_meta,
      batch_meta->label_info_meta_pool);
  if(ret == FALSE) {
    return ret;
  }

  g_list_free(batch_meta->frame_meta_list);
  g_list_free(batch_meta->batch_user_meta_list);

  g_rec_mutex_clear(&batch_meta->meta_mutex);
  g_free(batch_meta);
  return ret;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsFrameMeta *nvds_acquire_frame_meta_from_pool (NvDsBatchMeta *batch_meta)
{
  NvDsFrameMeta * frame_meta = NULL;
  frame_meta = (NvDsFrameMeta *)nvds_acquire_meta_from_pool(batch_meta,
      batch_meta->frame_meta_pool);
  return frame_meta;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsObjectMeta *nvds_acquire_obj_meta_from_pool (NvDsBatchMeta *batch_meta)
{
  NvDsObjectMeta * obj_meta = NULL;
  obj_meta = (NvDsObjectMeta *)nvds_acquire_meta_from_pool(batch_meta,
      batch_meta->obj_meta_pool);
  return obj_meta;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsClassifierMeta *nvds_acquire_classifier_meta_from_pool (
    NvDsBatchMeta *batch_meta)
{
  NvDsClassifierMeta * classifier_meta = NULL;
  classifier_meta = (NvDsClassifierMeta *)nvds_acquire_meta_from_pool(
      batch_meta, batch_meta->classifier_meta_pool);
  return classifier_meta;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsDisplayMeta *nvds_acquire_display_meta_from_pool (
    NvDsBatchMeta *batch_meta)
{
  NvDsDisplayMeta * display_meta = NULL;
  display_meta = (NvDsDisplayMeta *)nvds_acquire_meta_from_pool(batch_meta,
      batch_meta->display_meta_pool);
  return display_meta;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsUserMeta *nvds_acquire_user_meta_from_pool (NvDsBatchMeta *batch_meta)
{
  NvDsUserMeta * user_meta = NULL;
  user_meta = (NvDsUserMeta *)nvds_acquire_meta_from_pool(batch_meta,
      batch_meta->user_meta_pool);
  return user_meta;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsLabelInfo *nvds_acquire_label_info_meta_from_pool (
    NvDsBatchMeta *batch_meta)
{
  NvDsLabelInfo * label_info_meta = NULL;
  label_info_meta = (NvDsLabelInfo *)nvds_acquire_meta_from_pool(batch_meta,
      batch_meta->label_info_meta_pool);
  return label_info_meta;
}

void nvds_add_frame_meta_to_batch(NvDsBatchMeta * batch_meta,
    NvDsFrameMeta * frame_meta)
{
  batch_meta->frame_meta_list = nvds_add_meta_to_parent(
      batch_meta->frame_meta_list, frame_meta, TRUE);

  batch_meta->num_frames_in_batch++;
}

void nvds_add_obj_meta_to_frame(NvDsFrameMeta * frame_meta,
    NvDsObjectMeta *obj_meta, NvDsObjectMeta *obj_parent)
{
  frame_meta->obj_meta_list = nvds_add_meta_to_parent(
      frame_meta->obj_meta_list, obj_meta, FALSE);
  obj_meta->parent = obj_parent;
  frame_meta->num_obj_meta++;
}

void nvds_add_display_meta_to_frame(NvDsFrameMeta * frame_meta,
    NvDsDisplayMeta * display_meta)
{
  frame_meta->display_meta_list = nvds_add_meta_to_parent(
      frame_meta->display_meta_list, display_meta, FALSE);
}

void nvds_add_classifier_meta_to_roi(NvDsRoiMeta *roi_meta,
    NvDsClassifierMeta * classifier_meta)
{
  roi_meta->classifier_meta_list = nvds_add_meta_to_parent(
      roi_meta->classifier_meta_list, classifier_meta, FALSE);
}

void nvds_add_classifier_meta_to_object(NvDsObjectMeta *obj_meta,
    NvDsClassifierMeta * classifier_meta)
{
  obj_meta->classifier_meta_list = nvds_add_meta_to_parent(
      obj_meta->classifier_meta_list, classifier_meta, FALSE);
}

void nvds_add_label_info_meta_to_classifier(
    NvDsClassifierMeta *classifier_meta, NvDsLabelInfo * label_info_meta)
{
  classifier_meta->label_info_list = nvds_add_meta_to_parent(
      classifier_meta->label_info_list, label_info_meta, FALSE);
  classifier_meta->num_labels++;
}

void nvds_add_user_meta_to_batch(NvDsBatchMeta * batch_meta,
    NvDsUserMeta * user_meta)
{
  batch_meta->batch_user_meta_list = nvds_add_meta_to_parent(
      batch_meta->batch_user_meta_list, user_meta, TRUE);
}

void nvds_add_user_meta_to_frame(NvDsFrameMeta * frame_meta,
    NvDsUserMeta * user_meta)
{
  frame_meta->frame_user_meta_list = nvds_add_meta_to_parent(
      frame_meta->frame_user_meta_list, user_meta, FALSE);
}

void nvds_add_user_meta_to_roi(NvDsRoiMeta * roi_meta,
    NvDsUserMeta * user_meta)
{
  roi_meta->roi_user_meta_list = nvds_add_meta_to_parent(
      roi_meta->roi_user_meta_list, user_meta, FALSE);
}

void nvds_add_user_meta_to_obj(NvDsObjectMeta * obj_meta,
    NvDsUserMeta * user_meta)
{
  obj_meta->obj_user_meta_list = nvds_add_meta_to_parent(
      obj_meta->obj_user_meta_list, user_meta, FALSE);
}

/* Removes given frame_meta from FrameMetaList in the given batch metadata */
void nvds_remove_frame_meta_from_batch (NvDsBatchMeta *batch_meta,
    NvDsFrameMeta * frame_meta)
{
  batch_meta->frame_meta_list = nvds_remove_meta_from_parent (
      batch_meta->frame_meta_list, frame_meta, batch_meta->frame_meta_pool);
  batch_meta->num_frames_in_batch--;
}

/* Removes given frame_meta from FrameMetaList in the given batch metadata */
void nvds_remove_obj_meta_from_frame (NvDsFrameMeta * frame_meta,
    NvDsObjectMeta *obj_meta)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  frame_meta->obj_meta_list = nvds_remove_meta_from_parent (
      frame_meta->obj_meta_list, obj_meta, batch_meta->obj_meta_pool);
  frame_meta->num_obj_meta--;
}

/* Removes given frame_meta from FrameMetaList in the given batch metadata */
void nvds_remove_display_meta_from_frame (NvDsFrameMeta * frame_meta,
    NvDsDisplayMeta *display_meta)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  frame_meta->display_meta_list = nvds_remove_meta_from_parent (
      frame_meta->display_meta_list,
      display_meta, batch_meta->display_meta_pool);
}

/* Removes given classifier_meta from ClassifierMetaList in the given roi metadata */
void nvds_remove_classifier_meta_from_roi (NvDsRoiMeta *roi_meta,
    NvDsClassifierMeta *classifier_meta)
{
  NvDsBatchMeta *batch_meta = classifier_meta->base_meta.batch_meta;

  roi_meta->classifier_meta_list = nvds_remove_meta_from_parent (
      roi_meta->classifier_meta_list, classifier_meta,
      batch_meta->classifier_meta_pool);
}

/* Removes given frame_meta from FrameMetaList in the given batch metadata */
void nvds_remove_classifier_meta_from_obj (NvDsObjectMeta * obj_meta,
    NvDsClassifierMeta *classifier_meta)
{
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;
  obj_meta->classifier_meta_list = nvds_remove_meta_from_parent (
      obj_meta->classifier_meta_list, classifier_meta,
      batch_meta->classifier_meta_pool);
}

/* Removes given frame_meta from FrameMetaList in the given batch metadata */
void nvds_remove_label_info_meta_from_classifier (
    NvDsClassifierMeta * classifier_meta, NvDsLabelInfo *label_info_meta)
{
  NvDsBatchMeta *batch_meta = classifier_meta->base_meta.batch_meta;
  classifier_meta->label_info_list = nvds_remove_meta_from_parent (
      classifier_meta->label_info_list,
      label_info_meta, batch_meta->label_info_meta_pool);
  classifier_meta->num_labels--;
}

void nvds_remove_user_meta_from_batch(NvDsBatchMeta * batch_meta,
    NvDsUserMeta * user_meta)
{
    batch_meta->batch_user_meta_list = nvds_remove_meta_from_parent (
        batch_meta->batch_user_meta_list,
        user_meta, batch_meta->user_meta_pool);
}

void nvds_remove_user_meta_from_frame(NvDsFrameMeta * frame_meta,
    NvDsUserMeta * user_meta)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
    frame_meta->frame_user_meta_list = nvds_remove_meta_from_parent (
        frame_meta->frame_user_meta_list,
        user_meta, batch_meta->user_meta_pool);
}

void nvds_remove_user_meta_from_roi(NvDsRoiMeta * roi_meta,
    NvDsUserMeta * user_meta)
{
  NvDsBatchMeta *batch_meta = (NvDsBatchMeta *)&user_meta->base_meta;
    roi_meta->roi_user_meta_list = nvds_remove_meta_from_parent (
        roi_meta->roi_user_meta_list,
        user_meta, batch_meta->user_meta_pool);
}

void nvds_remove_user_meta_from_object(NvDsObjectMeta * obj_meta,
    NvDsUserMeta * user_meta)
{
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;
    obj_meta->obj_user_meta_list = nvds_remove_meta_from_parent (
        obj_meta->obj_user_meta_list,
        user_meta, batch_meta->user_meta_pool);
}

void nvds_batch_meta_release_func(gpointer data, gpointer user_data)
{
  nvds_destroy_batch_meta((NvDsBatchMeta *) data);
}

gpointer nvds_batch_meta_copy_func (gpointer data, gpointer user_data)
{
  return (gpointer)batch_meta_copy((NvDsBatchMeta *)data, user_data);
}

NvDsFrameMeta *nvds_get_nth_frame_meta (NvDsFrameMetaList *frame_meta_list,
    guint index)
{
  NvDsFrameMeta *frame_meta = NULL;
  frame_meta = (NvDsFrameMeta *)g_list_nth_data(frame_meta_list, index);
  return frame_meta;
}

void nvds_clear_frame_meta_list(NvDsBatchMeta *batch_meta,
    NvDsFrameMetaList *meta_list)
{
  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->frame_meta_pool);
  batch_meta->num_frames_in_batch = 0;
  batch_meta->frame_meta_list = NULL;
}

void nvds_clear_obj_meta_list(NvDsFrameMeta *frame_meta,
    NvDsObjectMetaList *meta_list)
{
  NvDsBaseMeta base_meta = frame_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->obj_meta_pool);
  frame_meta->num_obj_meta = 0;
  frame_meta->obj_meta_list = NULL;
}

void nvds_clear_classifier_meta_list(NvDsObjectMeta *obj_meta,
    NvDsClassifierMetaList *meta_list)
{
  NvDsBaseMeta base_meta = obj_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->classifier_meta_pool);
  obj_meta->classifier_meta_list = NULL;
}

void nvds_clear_label_info_meta_list(NvDsClassifierMeta *classifier_meta,
    NvDsLabelInfoList *meta_list)
{
  NvDsBaseMeta base_meta = classifier_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->label_info_meta_pool);
  classifier_meta->num_labels = 0;
  classifier_meta->label_info_list = NULL;
}

void nvds_clear_display_meta_list(NvDsFrameMeta *frame_meta,
    NvDisplayMetaList *meta_list)
{
  NvDsBaseMeta base_meta = frame_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->obj_meta_pool);
  frame_meta->display_meta_list = NULL;
}

void nvds_clear_batch_user_meta_list(NvDsBatchMeta *batch_meta,
    NvDsUserMetaList *meta_list)
{
  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->user_meta_pool);
  batch_meta->batch_user_meta_list = NULL;
}

void nvds_clear_frame_user_meta_list(NvDsFrameMeta *frame_meta,
    NvDsUserMetaList *meta_list)
{
  NvDsBaseMeta base_meta = frame_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->user_meta_pool);
  frame_meta->frame_user_meta_list = NULL;
}

void nvds_clear_obj_user_meta_list(NvDsObjectMeta *object_meta,
    NvDsUserMetaList *meta_list)
{
  NvDsBaseMeta base_meta = object_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->user_meta_pool);
  object_meta->obj_user_meta_list = NULL;
}

NvDsMetaType nvds_get_user_meta_type(gchar *meta_descriptor)
{
  NvDsMetaType user_meta_enum = NVDS_START_USER_META +
    g_quark_from_static_string(meta_descriptor);

  return user_meta_enum;
}
