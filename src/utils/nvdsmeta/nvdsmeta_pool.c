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
#include "nvds_audio_meta.h"
#include <stdio.h>
#include <string.h>

void release_frame_meta(NvDsElementMeta * element_meta, gpointer user_data){
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  NvDsBaseMeta temp_base_meta = {};

  if(frame_meta->sensorInfo_meta.sensor_id)
  {
    g_free((void*)frame_meta->sensorInfo_meta.sensor_id);
    frame_meta->sensorInfo_meta.sensor_id = NULL;
  }

  if(frame_meta->sensorInfo_meta.sensor_name)
  {
    g_free((void*)frame_meta->sensorInfo_meta.sensor_name);
    frame_meta->sensorInfo_meta.sensor_name = NULL;
  }

  if(frame_meta->sensorInfo_meta.uri)
  {
    g_free((void*)frame_meta->sensorInfo_meta.uri);
    frame_meta->sensorInfo_meta.uri = NULL;
  }

  if(frame_meta->obj_meta_list) {
    frame_meta->obj_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)frame_meta->obj_meta_list,
        (NvDsMetaPool *)batch_meta->obj_meta_pool);
  }

  if(frame_meta->display_meta_list) {
    frame_meta->display_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)frame_meta->display_meta_list,
        (NvDsMetaPool *)batch_meta->display_meta_pool);
  }

  if(frame_meta->frame_user_meta_list) {
    frame_meta->frame_user_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)frame_meta->frame_user_meta_list,
        (NvDsMetaPool *)batch_meta->user_meta_pool);
  }

  temp_base_meta = *base_meta;
  memset(element_meta, 0, sizeof(NvDsFrameMeta));
  *base_meta = temp_base_meta;
}

void release_audio_frame_meta(NvDsElementMeta * element_meta, gpointer user_data){
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  NvDsBaseMeta temp_base_meta = {};

  if(frame_meta->classifier_meta_list) {
    frame_meta->classifier_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)frame_meta->classifier_meta_list,
        (NvDsMetaPool *)batch_meta->classifier_meta_pool);
  }

  if(frame_meta->frame_user_meta_list) {
    frame_meta->frame_user_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)frame_meta->frame_user_meta_list,
        (NvDsMetaPool *)batch_meta->user_meta_pool);
  }

  temp_base_meta = *base_meta;
  memset(element_meta, 0, sizeof(NvDsAudioFrameMeta));
  *base_meta = temp_base_meta;
}

void release_obj_meta(NvDsElementMeta *element_meta, gpointer user_data){
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  NvDsBaseMeta temp_base_meta = {};

  if(obj_meta->classifier_meta_list) {
    obj_meta->classifier_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)obj_meta->classifier_meta_list,
        (NvDsMetaPool *)batch_meta->classifier_meta_pool);
  }

  if(obj_meta->obj_user_meta_list) {
    obj_meta->obj_user_meta_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)obj_meta->obj_user_meta_list,
        (NvDsMetaPool *)batch_meta->user_meta_pool);
  }

  if(obj_meta->text_params.display_text)
  {
    g_free(obj_meta->text_params.display_text);
    obj_meta->text_params.display_text = NULL;
  }

  if(obj_meta->mask_params.data && obj_meta->mask_params.size > 0)
  {
    g_free(obj_meta->mask_params.data);
    obj_meta->mask_params.data = NULL;
    obj_meta->mask_params.size = 0;
  }

  temp_base_meta = *base_meta;
  memset(element_meta, 0, sizeof(NvDsObjectMeta));
  *base_meta = temp_base_meta;
}

void release_classifier_meta(NvDsElementMeta *element_meta, gpointer user_data){
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  NvDsBaseMeta temp_base_meta = {};

  if(classifier_meta->label_info_list) {
    classifier_meta->label_info_list = nvds_clear_meta_list(batch_meta,
        (NvDsMetaList *)classifier_meta->label_info_list,
        (NvDsMetaPool *)batch_meta->label_info_meta_pool);
  }
  temp_base_meta = *base_meta;
  memset(element_meta, 0, sizeof(NvDsClassifierMeta));
  *base_meta = temp_base_meta;
}

void release_label_meta(NvDsElementMeta *element_meta, gpointer user_data){
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsLabelInfo *label_meta = (NvDsLabelInfo *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  NvDsBaseMeta temp_base_meta = {};

  nvds_acquire_meta_lock (batch_meta);
  if(label_meta->pResult_label) {
    g_free(label_meta->pResult_label);
    label_meta->pResult_label = NULL;
  }

  temp_base_meta = *base_meta;
  memset(element_meta, 0, sizeof(NvDsLabelInfo));
  *base_meta = temp_base_meta;
  nvds_release_meta_lock (batch_meta);
}

void release_display_meta(NvDsElementMeta *element_meta, gpointer user_data) {
  int i = 0;
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsDisplayMeta *display_meta = (NvDsDisplayMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  NvDsBaseMeta temp_base_meta = {};

  nvds_acquire_meta_lock (batch_meta);

  for(i = 0; i < MAX_ELEMENTS_IN_DISPLAY_META; i++)
  {
    if(display_meta->text_params[i].display_text)
    {
      g_free(display_meta->text_params[i].display_text);
      display_meta->text_params[i].display_text = NULL;
    }
  }

  temp_base_meta = *base_meta;
  memset(element_meta, 0, sizeof(NvDsDisplayMeta));
  *base_meta = temp_base_meta;

  nvds_release_meta_lock (batch_meta);
}


/* Creates frame meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */

NvDsMetaPool *nvds_create_frame_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_batch_size)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_batch_size, sizeof(NvDsFrameMeta), NULL, release_frame_meta, NVDS_FRAME_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_frame_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *frame_meta_pool)
{
  gboolean ret = FALSE;
  ret = nvds_destroy_meta_pool (batch_meta, frame_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates audio frame meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */

NvDsMetaPool *nvds_create_audio_frame_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_batch_size)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_batch_size, sizeof(NvDsAudioFrameMeta), NULL, release_audio_frame_meta, NVDS_AUDIO_FRAME_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_audio_frame_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *frame_meta_pool)
{
  gboolean ret = FALSE;
  ret = nvds_destroy_meta_pool (batch_meta, frame_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates obj meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */
NvDsMetaPool *nvds_create_obj_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_init_objs)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_init_objs, sizeof(NvDsObjectMeta), NULL, release_obj_meta, NVDS_OBJ_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_obj_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *obj_meta_pool)
{
  gboolean ret = FALSE;

  ret = nvds_destroy_meta_pool (batch_meta, obj_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates obj meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */
NvDsMetaPool *nvds_create_classifier_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_init_classifier_meta)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_init_classifier_meta, sizeof(NvDsClassifierMeta), NULL, release_classifier_meta,
      NVDS_CLASSIFIER_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_classifier_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *classifier_meta_pool)
{
  gboolean ret = FALSE;

  ret = nvds_destroy_meta_pool (batch_meta, classifier_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates obj meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */
NvDsMetaPool *nvds_create_display_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_display_meta)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_display_meta, sizeof(NvDsDisplayMeta), NULL, release_display_meta,
      NVDS_DISPLAY_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_display_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *display_meta_pool)
{
  gboolean ret = FALSE;
  ret = nvds_destroy_meta_pool (batch_meta, display_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates obj meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */
NvDsMetaPool *nvds_create_user_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_user_meta)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_user_meta, sizeof(NvDsUserMeta), NULL, NULL, NVDS_USER_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_user_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *user_meta_pool)
{
  gboolean ret = FALSE;
  ret = nvds_destroy_meta_pool (batch_meta, user_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates obj meta pool of size num_frame_meta initially for the given
 * batch size. If more frames are required, pool size will increase
 * dynamically */
NvDsMetaPool *nvds_create_label_info_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_label_info_meta)
{
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)nvds_create_meta_pool (batch_meta,
      max_label_info_meta, sizeof(NvDsLabelInfo), NULL, release_label_meta,
      NVDS_LABEL_INFO_META);
  if(meta_pool == NULL) {
    return meta_pool;
  }
  return meta_pool;
}

gboolean nvds_destroy_label_info_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *label_info_meta_pool)
{
  gboolean ret = FALSE;
  ret = nvds_destroy_meta_pool (batch_meta, label_info_meta_pool);
  if(ret == FALSE) {
    return ret;
  }
  return ret;
}

/* Creates  meta pool of size pool_size initially. If more units are required,
 * pool size will increase dynamically */
NvDsMetaPool *nvds_create_meta_pool (NvDsBatchMeta *batch_meta,
    guint pool_size, guint element_size, NvDsMetaCopyFunc copy_func,
    NvDsMetaReleaseFunc release_func, NvDsMetaType meta_type)
{
  guint i = 0;
  NvDsElementMeta *element_meta = NULL;
  NvDsMetaPool *meta_pool = (NvDsMetaPool *)g_malloc0(sizeof(NvDsMetaPool));
  if(meta_pool == NULL) {
    return meta_pool;
  }

  for(i = 0; i < pool_size; i++) {
    element_meta = (NvDsElementMeta *)g_malloc0(element_size);
    ((NvDsBaseMeta *)element_meta)->batch_meta = batch_meta;
    ((NvDsBaseMeta *)element_meta)->meta_type = meta_type;
    ((NvDsBaseMeta *)element_meta)->copy_func = copy_func;
    ((NvDsBaseMeta *)element_meta)->release_func = release_func;
    meta_pool->empty_list = g_list_prepend (meta_pool->empty_list,
        element_meta);
  }

  meta_pool->full_list = NULL;
  meta_pool->max_elements_in_pool = pool_size;
  meta_pool->element_size = element_size;
  meta_pool->num_empty_elements = pool_size;
  meta_pool->num_full_elements = 0;
  meta_pool->copy_func = copy_func;
  meta_pool->release_func = release_func;
  meta_pool->meta_type = meta_type;
  return meta_pool;
}

/* Destroys meta pool */
gboolean nvds_destroy_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *meta_pool)
{
  NvDsElementMeta *element_meta = NULL;
  NvDsMetaList *l = NULL;
  guint list_len = 0;
  NvDsBaseMeta *base_meta = NULL;

  //g_print("empty_len = %d\n",g_list_length(meta_pool->empty_list));
  //g_print("full_len = %d\n",g_list_length(meta_pool->full_list));
  list_len = g_list_length(meta_pool->empty_list) +
    g_list_length(meta_pool->full_list);

  if(list_len != meta_pool->max_elements_in_pool) {
    g_print("\nMismatch in elements in pool and in queue");
    return FALSE;
  }

  for (l = meta_pool->full_list; l != NULL; l = l->next) {
    element_meta = (NvDsElementMeta *)l->data;
    // release func
    base_meta = (NvDsBaseMeta *)(element_meta);
    if(base_meta->release_func) {
      base_meta->release_func(element_meta, NULL);
    }
    g_free(element_meta);
    element_meta = NULL;
  }

  for (l =  meta_pool->empty_list; l != NULL; l = l->next) {
    element_meta = (NvDsElementMeta *)l->data;
    g_free(element_meta);
    element_meta = NULL;
  }

  g_list_free(meta_pool->empty_list);
  meta_pool->empty_list = NULL;
  g_list_free(meta_pool->full_list);
  meta_pool->full_list = NULL;
  meta_pool->max_elements_in_pool = 0;
  meta_pool->num_empty_elements = 0;
  meta_pool->num_full_elements = 0;
  g_free(meta_pool);
  meta_pool = NULL;
  return TRUE;
}

/* Acquires a frame meta from meta_pool of given batch */
NvDsElementMeta *nvds_acquire_meta_from_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *meta_pool)
{
  NvDsMetaList *empty_list;
  NvDsMetaList *element_from_list = NULL;
  NvDsElementMeta *element_meta = NULL;
  NvDsBaseMeta *base_meta = NULL;
  NvDsBaseMeta temp_base_meta = {};

  nvds_acquire_meta_lock (batch_meta);

  empty_list = meta_pool->empty_list;
  element_from_list = g_list_first(empty_list);
  if(element_from_list == NULL) {
    increase_pool_size(batch_meta, meta_pool);
    empty_list = meta_pool->empty_list;
    element_from_list = g_list_first(empty_list);
    //return NULL;
  }
  element_meta = (NvDsElementMeta *)element_from_list->data;
  empty_list = g_list_remove_link (empty_list, element_from_list);
  meta_pool->num_empty_elements--;
  meta_pool->empty_list = empty_list;
  meta_pool->full_list = g_list_concat (element_from_list, meta_pool->full_list);
  meta_pool->num_full_elements++;

  base_meta = (NvDsBaseMeta *)(element_meta);

  temp_base_meta.batch_meta = base_meta->batch_meta;
  temp_base_meta.meta_type = base_meta->meta_type;
  temp_base_meta.copy_func = base_meta->copy_func;
  temp_base_meta.release_func = base_meta->release_func;

  memset(element_meta, 0, sizeof(meta_pool->element_size));

  base_meta->batch_meta = temp_base_meta.batch_meta;
  base_meta->meta_type = temp_base_meta.meta_type;
  base_meta->copy_func = temp_base_meta.copy_func;
  base_meta->release_func = temp_base_meta.release_func;

  nvds_release_meta_lock (batch_meta);
  return element_meta;
}

void increase_pool_size (NvDsBatchMeta *batch_meta, NvDsMetaPool *meta_pool)
{
  NvDsMetaType meta_type = meta_pool->meta_type;
  NvDsElementMeta *element_meta = NULL;
  guint i = 0;

  guint NUM_ELEM = MIN(128, 2 * meta_pool->max_elements_in_pool);

  //g_print("************Increasing Pool size *************\n");
  for(i = 0; i < NUM_ELEM; i++) {
    element_meta = (NvDsElementMeta *)g_malloc0(meta_pool->element_size);
    ((NvDsBaseMeta *)element_meta)->batch_meta = batch_meta;
    ((NvDsBaseMeta *)element_meta)->meta_type = meta_type;
    ((NvDsBaseMeta *)element_meta)->copy_func = meta_pool->copy_func;
    ((NvDsBaseMeta *)element_meta)->release_func = meta_pool->release_func;
    meta_pool->empty_list = g_list_prepend (meta_pool->empty_list,
        element_meta);
  }
  meta_pool->num_empty_elements = NUM_ELEM;
  meta_pool->max_elements_in_pool += NUM_ELEM;
}

NvDsMetaList *nvds_add_meta_to_parent(NvDsMetaList *parent_meta_list,
    NvDsElementMeta *element_meta, gboolean append)
{
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;
  nvds_acquire_meta_lock (batch_meta);

  if (append)
    parent_meta_list = g_list_append (parent_meta_list, element_meta);
  else
    parent_meta_list = g_list_prepend (parent_meta_list, element_meta);

  nvds_release_meta_lock (batch_meta);
  return parent_meta_list;
}

/* Removes given frame_meta from FrameMetaList in the given batch metadata */
NvDsMetaList * nvds_remove_meta_from_parent (NvDsMetaList *parent_list,
    NvDsElementMeta * element_meta, NvDsMetaPool *meta_pool)
{
  NvDsMetaList *temp_list = NULL;
  NvDsBaseMeta *base_meta = (NvDsBaseMeta *)element_meta;
  NvDsBatchMeta * batch_meta = base_meta->batch_meta;

  if(base_meta->release_func) {
    base_meta->release_func(element_meta, NULL);
  }

  nvds_acquire_meta_lock (batch_meta);
  /* remove from parent list */
  temp_list = g_list_find(parent_list, element_meta);
  parent_list = g_list_remove_link (parent_list, temp_list);
  g_list_free(temp_list);
  temp_list = NULL;

  /* remove from full list of pool */
  temp_list = g_list_find(meta_pool->full_list, element_meta);
  meta_pool->full_list = g_list_remove_link (meta_pool->full_list, temp_list);

  /* return to the empty list of pool */
  meta_pool->empty_list = g_list_concat (temp_list, meta_pool->empty_list);
  meta_pool->num_empty_elements++;
  meta_pool->num_full_elements--;
  nvds_release_meta_lock (batch_meta);
  return parent_list;
}

void get_meta_pool_status(NvDsMetaPool *meta_pool)
{
  int i = 0;
  NvDsElementMeta *free_meta = NULL;
  NvDsElementMeta *full_meta = NULL;
  NvDsMetaList *l = NULL;

  g_print("meta_type = %d\n", meta_pool->meta_type);
  g_print("max_elements_in_pool = %d\n",meta_pool->max_elements_in_pool);
  g_print("num_empty_elements = %d\n",meta_pool->num_empty_elements);
  g_print("num_full_elements = %d\n",meta_pool->num_full_elements);

#if 1
  for (l = meta_pool->empty_list; l != NULL; l = l->next) {
    free_meta = (NvDsElementMeta *)l->data;
    g_print("\nFREE META %d ptr = %p", i, free_meta);
    i++;
  }

 i = 0;
 for (l = meta_pool->full_list; l != NULL; l = l->next) {
    full_meta = (NvDsElementMeta *)l->data;
    g_print("\nFULL META %d ptr = %p", i, full_meta);
    if(meta_pool->meta_type == NVDS_OBJ_META)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)full_meta;
      g_print("\nx = %f", obj_meta->rect_params.left);
      g_print(" y = %f", obj_meta->rect_params.top);
      g_print(" w = %f", obj_meta->rect_params.width);
      g_print(" h = %f\n", obj_meta->rect_params.height);
    }
    i++;
  }
#endif
}

gboolean nvds_get_current_metadata_info(NvDsBatchMeta *batch_meta)
{
  NvDsFrameMetaList *frame_meta_list = batch_meta->frame_meta_list;
  NvDsMetaPool *meta_pool = batch_meta->frame_meta_pool;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsFrameMetaList *l = NULL;
  guint i = 0;
  guint j = 0;

  if(frame_meta_list == NULL)
  {
    return TRUE;
  }
  /* FRAME META POOL STATUS*/
  g_print("\n******************FRAME META POOL STATUS ***************\n");
  nvds_acquire_meta_lock (batch_meta);
  get_meta_pool_status(meta_pool);

  g_print("\n******************OBJ META POOL STATUS *****************\n");
  meta_pool = batch_meta->obj_meta_pool;
  get_meta_pool_status(meta_pool);

  for (l = frame_meta_list; l != NULL; l = l->next) {
    frame_meta = (NvDsFrameMeta *)l->data;
    if(frame_meta) {
      g_print("\nFRAME %d", i);
      g_print("\nFRAME META ptr = %p source_id = %d port id = %d\n",
          frame_meta, frame_meta->source_id, frame_meta->pad_index);
      i++;

      NvDsObjectMetaList *l = NULL;

      for (l = frame_meta->obj_meta_list; l != NULL; l = l->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l->data;
        g_print("\n  OBJ %d\n", j);
        g_print("  OBJ META ptr = %p component_id = %d,"
            "class_id = %d object id = %lu x =%f, y =%f, w =%f, h = %f\n",
            obj_meta, obj_meta->unique_component_id,
            obj_meta->class_id, obj_meta->object_id, obj_meta->rect_params.left,
            obj_meta->rect_params.top, obj_meta->rect_params.width,
            obj_meta->rect_params.height);
        j++;
      }
      j = 0;
    }
  }
  nvds_release_meta_lock (batch_meta);
  return TRUE;
}

NvDsMetaList *nvds_clear_meta_list(NvDsBatchMeta *batch_meta,
    NvDsMetaList *meta_list, NvDsMetaPool *meta_pool)
{
  NvDsMetaList *l = NULL;
  NvDsMetaList *temp_list = NULL;
  NvDsElementMeta * element_meta = NULL;
  NvDsBaseMeta *base_meta = NULL;

  nvds_acquire_meta_lock (batch_meta);
  for (l = meta_list; l != NULL; l = l->next)
  {
    element_meta = (NvDsElementMeta *)(l->data);
    base_meta = (NvDsBaseMeta *)element_meta;

    if(base_meta->release_func) {
      base_meta->release_func(element_meta, NULL);
    }

    temp_list = g_list_find(meta_pool->full_list, element_meta);
    meta_pool->full_list = g_list_remove_link (meta_pool->full_list,
        temp_list);
    /* return to the empty list of pool */
    meta_pool->empty_list = g_list_concat (temp_list, meta_pool->empty_list);
    meta_pool->num_empty_elements++;
    meta_pool->num_full_elements--;
  }
  g_list_free(meta_list);
  meta_list = NULL;
  nvds_release_meta_lock (batch_meta);
  return meta_list;
}