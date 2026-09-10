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

#ifndef _NVDSMETA_INTERNAL_H_
#define _NVDSMETA_INTERNAL_H_

#include "nvdsmeta.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /** number of object elements in the object meta pool at init time */
#define NUM_OBJ_META 16
  /** number of classifier elements in the classifier meta pool at init time */
#define NUM_CLASSIFIER_META 8
  /** number of display elements in the display meta pool at init time */
#define NUM_DISPLAY_META 8
  /** number of user elements in the user meta pool at init time */
#define NUM_USER_META 8
  /** number of label elements in the label meta pool at init time */
#define NUM_LABEL_META 8

NvDsMetaPool *nvds_create_frame_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_batch_size);

gboolean nvds_destroy_frame_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *frame_meta_pool);

NvDsMetaPool *nvds_create_audio_frame_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_batch_size);

gboolean nvds_destroy_audio_frame_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *frame_meta_pool);

NvDsMetaPool *nvds_create_obj_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_init_objs);

gboolean nvds_destroy_obj_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *obj_meta_pool);

NvDsMetaPool *nvds_create_classifier_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_init_classifier_meta);

gboolean nvds_destroy_classifier_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *classifier_meta_pool);

NvDsMetaPool *nvds_create_display_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_display_meta);

gboolean nvds_destroy_display_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *display_meta_pool);

NvDsMetaPool *nvds_create_user_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_user_meta);

gboolean nvds_destroy_user_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *user_meta_pool);

NvDsMetaPool *nvds_create_label_info_meta_pool (NvDsBatchMeta *batch_meta,
    guint max_label_info_meta);

gboolean nvds_destroy_label_info_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *label_info_meta_pool);

NvDsMetaPool *nvds_create_meta_pool (NvDsBatchMeta *batch_meta,
    guint pool_size, guint element_size, NvDsMetaCopyFunc copy_func,
    NvDsMetaReleaseFunc release_func, NvDsMetaType meta_type);

gboolean nvds_destroy_meta_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *meta_pool);

NvDsElementMeta *nvds_acquire_meta_from_pool (NvDsBatchMeta *batch_meta,
    NvDsMetaPool *meta_pool);

NvDsMetaList *nvds_add_meta_to_parent(NvDsMetaList *parent_meta_list,
    NvDsElementMeta *element_meta, gboolean append);

NvDsMetaList *nvds_remove_meta_from_parent (NvDsMetaList *parent_list,
    NvDsElementMeta * meta, NvDsMetaPool *meta_pool);

void increase_pool_size (NvDsBatchMeta *batch_meta, NvDsMetaPool *meta_pool);

NvDsBatchMeta *batch_meta_copy(NvDsBatchMeta *src_batch_meta,
    gpointer user_data);

NvDsBatchMeta *audio_batch_meta_copy(NvDsBatchMeta *src_batch_meta,
    gpointer user_data);

void release_frame_meta(NvDsElementMeta * element_meta, gpointer user_data);

void release_audio_frame_meta(NvDsElementMeta * element_meta, gpointer user_data);

void release_obj_meta(NvDsElementMeta *element_meta, gpointer user_data);

void release_classifier_meta(NvDsElementMeta *element_meta, gpointer user_data);

void release_label_meta(NvDsElementMeta *element_meta, gpointer user_data);

void release_display_meta(NvDsElementMeta *element_meta, gpointer user_data);

void get_meta_pool_status(NvDsMetaPool *meta_pool);

void copy_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsElementMeta *element_meta, NvDsMetaType meta_type);

void copy_audio_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsElementMeta *element_meta, NvDsMetaType meta_type);

void copy_user_meta(NvDsUserMeta *src_user_meta, NvDsUserMeta *dst_user_meta);

#ifdef __cplusplus
}
#endif

#endif
