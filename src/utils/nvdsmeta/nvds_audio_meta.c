/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvds_audio_meta.h"
#include "nvdsmeta_internal.h"
#include <stdio.h>
#include <string.h>

/*************Batch Functions *****************/
NvDsBatchMeta *nvds_create_audio_batch_meta(guint max_batch_size) {
  NvDsBatchMeta *batch_meta = NULL;
  gboolean ret = TRUE;

  batch_meta = (NvDsBatchMeta *)g_malloc0(sizeof(NvDsBatchMeta));
  g_rec_mutex_init(&batch_meta->meta_mutex);

  batch_meta->frame_meta_pool = nvds_create_audio_frame_meta_pool(batch_meta,
      max_batch_size);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->classifier_meta_pool = nvds_create_classifier_meta_pool(
      batch_meta, NUM_CLASSIFIER_META);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->user_meta_pool = nvds_create_user_meta_pool(batch_meta,
      NUM_USER_META);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->label_info_meta_pool = nvds_create_label_info_meta_pool(
      batch_meta, NUM_LABEL_META);
  if(ret == FALSE) {
    return NULL;
  }

  batch_meta->base_meta.batch_meta = batch_meta;
  batch_meta->base_meta.meta_type = NVDS_AUDIO_BATCH_META;
  batch_meta->base_meta.copy_func = nvds_audio_batch_meta_copy_func;
  batch_meta->base_meta.release_func = nvds_audio_batch_meta_release_func;

  return batch_meta;
}

gboolean nvds_destroy_audio_batch_meta(NvDsBatchMeta *batch_meta) {
  gboolean ret = TRUE;

  if (batch_meta->base_meta.meta_type != NVDS_AUDIO_BATCH_META)
    return FALSE;

  ret = nvds_destroy_frame_meta_pool(batch_meta, batch_meta->frame_meta_pool);
  if(ret == FALSE) {
    return ret;
  }

  ret = nvds_destroy_classifier_meta_pool(batch_meta,
      batch_meta->classifier_meta_pool);
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
NvDsAudioFrameMeta *nvds_acquire_audio_frame_meta_from_pool (NvDsBatchMeta *batch_meta)
{
  NvDsAudioFrameMeta * frame_meta = NULL;

  if (batch_meta->base_meta.meta_type != NVDS_AUDIO_BATCH_META)
    return frame_meta;

  frame_meta = (NvDsAudioFrameMeta *)nvds_acquire_meta_from_pool(batch_meta,
      batch_meta->frame_meta_pool);
  return frame_meta;
}

void nvds_add_audio_frame_meta_to_audio_batch(NvDsBatchMeta * batch_meta,
    NvDsAudioFrameMeta * frame_meta)
{
  batch_meta->frame_meta_list = nvds_add_meta_to_parent(
      batch_meta->frame_meta_list, frame_meta, TRUE);

  batch_meta->num_frames_in_batch++;
}

void nvds_add_classifier_meta_to_audio_frame(NvDsAudioFrameMeta *frame_meta,
    NvDsClassifierMeta * classifier_meta)
{
  frame_meta->classifier_meta_list = nvds_add_meta_to_parent(
      frame_meta->classifier_meta_list, classifier_meta, TRUE);
}

void nvds_add_user_meta_to_audio_batch(NvDsBatchMeta * batch_meta,
    NvDsUserMeta * user_meta)
{
  batch_meta->batch_user_meta_list = nvds_add_meta_to_parent(
      batch_meta->batch_user_meta_list, user_meta, TRUE);
}

void nvds_add_user_meta_to_audio_frame(NvDsAudioFrameMeta * frame_meta,
    NvDsUserMeta * user_meta)
{
  frame_meta->frame_user_meta_list = nvds_add_meta_to_parent(
      frame_meta->frame_user_meta_list, user_meta, TRUE);
}

/* Removes given frame_meta from AudioFrameMetaList in the given batch metadata */
void nvds_remove_audio_frame_meta_from_audio_batch (NvDsBatchMeta *batch_meta,
    NvDsAudioFrameMeta * frame_meta)
{
  batch_meta->frame_meta_list = nvds_remove_meta_from_parent (
      batch_meta->frame_meta_list, frame_meta, batch_meta->frame_meta_pool);
  batch_meta->num_frames_in_batch--;
}

/* Removes given classifier_meta from ClassifierMetaList in the given audio frame metadata */
void nvds_remove_classifier_meta_from_audio_frame (NvDsAudioFrameMeta * frame_meta,
    NvDsClassifierMeta *classifier_meta)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  frame_meta->classifier_meta_list = nvds_remove_meta_from_parent (
      frame_meta->classifier_meta_list, classifier_meta,
      batch_meta->classifier_meta_pool);
}

void nvds_remove_user_meta_from_audio_batch(NvDsBatchMeta * batch_meta,
    NvDsUserMeta * user_meta)
{
    batch_meta->batch_user_meta_list = nvds_remove_meta_from_parent (
        batch_meta->batch_user_meta_list,
        user_meta, batch_meta->user_meta_pool);
}

void nvds_remove_user_meta_from_audio_frame(NvDsAudioFrameMeta * frame_meta,
    NvDsUserMeta * user_meta)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
    frame_meta->frame_user_meta_list = nvds_remove_meta_from_parent (
        frame_meta->frame_user_meta_list,
        user_meta, batch_meta->user_meta_pool);
}

void nvds_audio_batch_meta_release_func(gpointer data, gpointer user_data)
{
  nvds_destroy_audio_batch_meta((NvDsBatchMeta *)data);
}

gpointer nvds_audio_batch_meta_copy_func (gpointer data, gpointer user_data)
{
  return (gpointer)audio_batch_meta_copy((NvDsBatchMeta *)data, user_data);
}

NvDsAudioFrameMeta *nvds_get_nth_audio_frame_meta (NvDsFrameMetaList *frame_meta_list,
    guint index)
{
  NvDsAudioFrameMeta *frame_meta = NULL;
  frame_meta = (NvDsAudioFrameMeta *)g_list_nth_data(frame_meta_list, index);
  return frame_meta;
}

void nvds_clear_audio_frame_meta_list(NvDsBatchMeta *batch_meta,
    NvDsFrameMetaList *meta_list)
{
  meta_list = nvds_clear_meta_list(batch_meta, (NvDsFrameMetaList *)meta_list,
      batch_meta->frame_meta_pool);
  batch_meta->num_frames_in_batch = 0;
  batch_meta->frame_meta_list = NULL;
}

void nvds_clear_audio_classifier_meta_list(NvDsAudioFrameMeta *frame_meta,
    NvDsClassifierMetaList *meta_list)
{
  NvDsBaseMeta base_meta = frame_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->classifier_meta_pool);
  frame_meta->classifier_meta_list = NULL;
}

void nvds_clear_audio_batch_user_meta_list(NvDsBatchMeta *batch_meta,
    NvDsUserMetaList *meta_list)
{
  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->user_meta_pool);
  batch_meta->batch_user_meta_list = NULL;
}

void nvds_clear_audio_frame_user_meta_list(NvDsAudioFrameMeta *frame_meta,
    NvDsUserMetaList *meta_list)
{
  NvDsBaseMeta base_meta = frame_meta->base_meta;
  NvDsBatchMeta *batch_meta = base_meta.batch_meta;

  meta_list = nvds_clear_meta_list(batch_meta, (NvDsMetaList *)meta_list,
      batch_meta->user_meta_pool);
  frame_meta->frame_user_meta_list = NULL;
}
