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
#include "nvds_audio_meta.h"
#include <stdio.h>
#include <string.h>

/* Copy functions */
NvDsBatchMeta *audio_batch_meta_copy(NvDsBatchMeta *src_batch_meta,
    gpointer user_data)
{
  NvDsBatchMeta *dst_batch_meta = nvds_create_audio_batch_meta(
      src_batch_meta->max_frames_in_batch);
  dst_batch_meta->max_frames_in_batch = src_batch_meta->max_frames_in_batch;
  memcpy(dst_batch_meta->misc_batch_info, src_batch_meta->misc_batch_info,
      sizeof(gint64) * MAX_USER_FIELDS);
  memcpy(dst_batch_meta->reserved, src_batch_meta->reserved, sizeof(gint64) *
      MAX_USER_FIELDS);

  nvds_copy_audio_frame_meta_list(src_batch_meta->frame_meta_list, dst_batch_meta);

  nvds_copy_audio_batch_user_meta_list( src_batch_meta->batch_user_meta_list,
      dst_batch_meta);

  return dst_batch_meta;
}

void nvds_copy_audio_frame_meta_list (NvDsFrameMetaList *src_frame_meta_list,
    NvDsBatchMeta *dst_batch_meta)
{
  NvDsMetaList *l = NULL;
  NvDsAudioFrameMeta *src_frame_meta = NULL;
  NvDsAudioFrameMeta *dst_frame_meta = NULL;

  for (l = src_frame_meta_list; l != NULL; l = l->next) {
    src_frame_meta = (NvDsAudioFrameMeta *)(l->data);
    dst_frame_meta = nvds_acquire_audio_frame_meta_from_pool (dst_batch_meta);
    nvds_copy_audio_frame_meta(src_frame_meta, dst_frame_meta);
    nvds_add_audio_frame_meta_to_audio_batch(dst_batch_meta, dst_frame_meta);
  }
}

void nvds_copy_audio_frame_meta(NvDsAudioFrameMeta *src_frame_meta,
    NvDsAudioFrameMeta *dst_frame_meta)
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
  dst_frame_meta->batch_id = src_frame_meta->batch_id;
  dst_frame_meta->bInferDone = src_frame_meta->bInferDone;
  dst_frame_meta->num_samples_per_frame = src_frame_meta->num_samples_per_frame;
  dst_frame_meta->sample_rate = src_frame_meta->sample_rate;
  dst_frame_meta->num_channels = src_frame_meta->num_channels;
  dst_frame_meta->format = src_frame_meta->format;
  dst_frame_meta->layout = src_frame_meta->layout;
  dst_frame_meta->class_id = src_frame_meta->class_id;
  dst_frame_meta->confidence = src_frame_meta->confidence;

  strncpy(dst_frame_meta->class_label, src_frame_meta->class_label, MAX_LABEL_SIZE);

  memcpy(dst_frame_meta->misc_frame_info, src_frame_meta->misc_frame_info,
      sizeof(gint64) * MAX_USER_FIELDS);
  memcpy(dst_frame_meta->reserved, src_frame_meta->reserved,
      sizeof(gint64) * MAX_USER_FIELDS);

  nvds_release_meta_lock (src_batch_meta);

  nvds_copy_audio_classification_list(src_frame_meta->classifier_meta_list, dst_frame_meta);

  nvds_copy_audio_frame_user_meta_list(src_frame_meta->frame_user_meta_list,
      dst_frame_meta);
}

void nvds_copy_audio_classification_list(NvDsClassifierMetaList *src_classifier_meta_list,
    NvDsAudioFrameMeta *dst_frame_meta)
{
  NvDsMetaList *l = NULL;
  NvDsClassifierMeta *src_classifier_meta = NULL;
  NvDsClassifierMeta *dst_classifier_meta = NULL;
  NvDsBatchMeta *dst_batch_meta = dst_frame_meta->base_meta.batch_meta;

  //g_print("classifier_meta_list list len = %d\n", g_list_length(src_classifier_meta_list));
    /* Copy frame meta list */
  for (l = src_classifier_meta_list; l != NULL; l = l->next) {
    src_classifier_meta = (NvDsClassifierMeta *)(l->data);
    dst_classifier_meta = nvds_acquire_classifier_meta_from_pool (dst_batch_meta);
    nvds_copy_classifier_meta(src_classifier_meta, dst_classifier_meta);
    nvds_add_classifier_meta_to_audio_frame(dst_frame_meta, dst_classifier_meta);
  }
}

void nvds_copy_audio_batch_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsBatchMeta *dst_batch_meta)
{
  copy_audio_user_meta_list(src_user_meta_list, (NvDsElementMeta *)dst_batch_meta,
      NVDS_AUDIO_BATCH_META);
}

void nvds_copy_audio_frame_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsAudioFrameMeta *dst_frame_meta)
{
  copy_audio_user_meta_list(src_user_meta_list, (NvDsElementMeta *)dst_frame_meta,
      NVDS_AUDIO_FRAME_META);
}

void copy_audio_user_meta_list(NvDsUserMetaList *src_user_meta_list,
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

    if(meta_type == NVDS_AUDIO_BATCH_META)
    {
      nvds_add_user_meta_to_audio_batch((NvDsBatchMeta *)element_meta, dst_user_meta);
    }
    if(meta_type == NVDS_AUDIO_FRAME_META)
    {
      nvds_add_user_meta_to_audio_frame((NvDsAudioFrameMeta *)element_meta, dst_user_meta);
    }
  }
}
