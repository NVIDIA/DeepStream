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

#include "string.h"
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvds_latency_meta.h"
#include "nvds_latency_meta_internal.h"

static GMutex g_nvds_meta_lock;
static GQuark _dsmeta_quark = 0;

static gboolean enable_latency_measurement_1 = FALSE;
static gboolean enable_component_latency_measurement_1 = FALSE;
static gboolean latency_measurement_silent_1 = FALSE;

gboolean nvds_get_enable_latency_measurement()
{
  return enable_latency_measurement_1;
}

gboolean nvds_get_enable_per_component_latency_measurement()
{
  return enable_component_latency_measurement_1;
}

gboolean nvds_get_latency_measurement_silent()
{
  return latency_measurement_silent_1;
}

GType
nvds_meta_api_get_type (void)
{
  static GType type;
  static const gchar *tags[] = { NVDS_META_STRING, NULL };

  if (g_once_init_enter (&type)) {
    GType _type = gst_meta_api_type_register ("NvDsMetaAPI", tags);
    g_once_init_leave (&type, _type);
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
  }
  return type;
}

static gboolean
gst_nvds_meta_init (GstMeta * meta, gpointer params, GstBuffer *buffer)
{
  const gchar * latency = g_getenv("NVDS_ENABLE_LATENCY_MEASUREMENT");
  const gchar * component_latency = g_getenv("NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT");
  const gchar * silent_mode = g_getenv("NVDS_LATENCY_MEASUREMENT_SILENT");

  if(latency)
  {
    enable_latency_measurement_1 = TRUE;
  }
  if(component_latency)
  {
    enable_component_latency_measurement_1 = TRUE;
  }
  if(silent_mode)
  {
    latency_measurement_silent_1 = TRUE;
  }
  return TRUE;
}

static gboolean
gst_nvds_meta_transform (GstBuffer * outbuf, GstMeta * meta,
    GstBuffer * inbuf, GQuark type, gpointer data)
{
  NvDsMeta *src_meta = (NvDsMeta *) meta;
  NvDsMeta *dst_meta = NULL;

  if (GST_META_TRANSFORM_IS_COPY (type)) {
    switch (src_meta->meta_type) {
      case NVDS_BATCH_GST_META:
        {
          NvDsBatchMeta *batch_meta = (NvDsBatchMeta *)src_meta->copyfunc(
              (NvDsBatchMeta *)src_meta->meta_data, NULL);

          if (batch_meta->base_meta.meta_type == NVDS_BATCH_META) {
              dst_meta =
                gst_buffer_add_nvds_meta (outbuf, batch_meta, NULL,
                    (NvDsMetaCopyFunc) nvds_batch_meta_copy_func,
                    (NvDsMetaReleaseFunc) nvds_batch_meta_release_func);
          } else if (batch_meta->base_meta.meta_type == NVDS_AUDIO_BATCH_META) {
              dst_meta =
                gst_buffer_add_nvds_meta (outbuf, batch_meta, NULL,
                    (NvDsMetaCopyFunc) nvds_audio_batch_meta_copy_func,
                    (NvDsMetaReleaseFunc) nvds_audio_batch_meta_release_func);
          }
          break;
        }

      default:
        {
        gpointer o_meta = (gpointer)src_meta->copyfunc(
            (gpointer)src_meta->meta_data, NULL);

        dst_meta =
          gst_buffer_add_nvds_meta (outbuf, o_meta, NULL,
              (NvDsMetaCopyFunc) src_meta->copyfunc,
              (NvDsMetaReleaseFunc) src_meta->freefunc);
        }
        break;
    }

    if (dst_meta) {
      dst_meta->meta_type = src_meta->meta_type;
      dst_meta->gst_to_nvds_meta_transform_func =
        src_meta->gst_to_nvds_meta_transform_func;
      dst_meta->gst_to_nvds_meta_release_func = src_meta->gst_to_nvds_meta_release_func;
    }
    else {
      return FALSE;
    }
  }
  return TRUE;
}

static void
gst_nvds_meta_free (GstMeta * meta, GstBuffer * buffer)
{
  NvDsMeta *emeta = (NvDsMeta *) meta;

  if (emeta->freefunc && emeta->meta_data) {
    emeta->freefunc (emeta->meta_data, emeta->user_data);
  }
}

const GstMetaInfo *
nvds_meta_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;

  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (NVDS_META_API_TYPE,
        "NvDsMeta",
        sizeof (NvDsMeta),
        gst_nvds_meta_init,
        gst_nvds_meta_free,
        gst_nvds_meta_transform);
    g_once_init_leave (&meta_info, mi);
  }
  return meta_info;
}



NvDsMeta *gst_buffer_add_nvds_meta (GstBuffer *buffer, gpointer meta_data,
    gpointer user_data, NvDsMetaCopyFunc copy_func,
    NvDsMetaReleaseFunc release_func)
{
  NvDsMeta *meta;
  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);
  g_mutex_lock (&g_nvds_meta_lock);
  GST_MINI_OBJECT_FLAG_SET (buffer, GST_MINI_OBJECT_FLAG_LOCKABLE);

  meta = (NvDsMeta *) gst_buffer_add_meta (buffer,
      NVDS_META_INFO, NULL);

  if(meta == NULL) {
      g_mutex_unlock (&g_nvds_meta_lock);
      return NULL;
  }
  meta->meta_data = meta_data;
  meta->copyfunc = copy_func;
  meta->freefunc = release_func;
  meta->user_data = user_data;

  GST_MINI_OBJECT_FLAG_UNSET (buffer, GST_MINI_OBJECT_FLAG_LOCKABLE);
  g_mutex_unlock (&g_nvds_meta_lock);

  return meta;
}

NvDsMeta *
gst_buffer_get_nvds_meta (GstBuffer *buffer)
{
    NvDsMeta *meta= (NvDsMeta *)gst_buffer_get_meta(buffer,NVDS_META_API_TYPE);
    return meta;
}

NvDsBatchMeta *
gst_buffer_get_nvds_batch_meta (GstBuffer *buffer)
{
  gpointer state = NULL;
  GstMeta *gst_meta;
  NvDsBatchMeta *batch_meta = NULL;

  while ((gst_meta = gst_buffer_iterate_meta(buffer, &state))) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      continue;
    }
    NvDsMeta *dsmeta = (NvDsMeta *) gst_meta;

    if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
      if (batch_meta != NULL) {
        GST_WARNING("Multiple NvDsBatchMeta found on buffer %p", buffer);
      }
      batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
    }
  }

  return batch_meta;
}

static void *s_set_metadata_ptr(GstBuffer* src_gst_buffer);
static gpointer s_copy_user_meta(gpointer data, gpointer user_data);
static void s_release_user_meta(gpointer data, gpointer user_data);
static GList* s_get_nvdsmeta_list(GstBuffer* buf);
static void s_remove_all_list_entries(GList* gst_meta_list);
static void s_add_nvdsmeta_list_as_usermeta_list(GstBuffer* src_buffer,
        NvDsBatchMeta* batch_meta, void* frame_meta,
        gboolean is_audio);

static gboolean gst_meta_to_copy_is_available(GstBuffer* buf);

typedef struct
{
  GstBuffer *outbuf;
} CopyMetaData;

static gboolean gst_meta_to_copy_is_available(GstBuffer* buf)
{
  GstMeta* gst_meta = NULL;
  gpointer state = NULL;
  while ((gst_meta = gst_buffer_iterate_meta (buf, &state)))
  {
     if (gst_meta_api_type_has_tag (gst_meta->info->api, _gst_meta_tag_memory)
      || gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
        /** need not copy memory specific meta and DS meta
         * Note: DS meta (NvDsMeta) copy done separately with
         * s_add_nvdsmeta_list_as_usermeta_list() */
        continue;
     }
     else {
        return TRUE;
     }
  }

  return FALSE;
}

/**
 * @brief  Gather all NvDsMeta GstMeta elements in @buf into a GList
 * @param  buf [in]
 * @return the GList [transfer-none]; the list is valid until the buf is
 */
static GList*
s_get_nvdsmeta_list(GstBuffer* buf)
{
  GList * gst_meta_list = NULL;
  gpointer state = NULL;
  GstMeta* gst_meta = NULL;
  while ((gst_meta = gst_buffer_iterate_meta (buf, &state)))
  {
    if (gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      gst_meta_list = g_list_append(gst_meta_list, gst_meta);
    }
  }
  return gst_meta_list;
}

static void
s_remove_all_list_entries(GList* gst_meta_list)
{
  while(g_list_length(gst_meta_list))
  {
    GList* l = g_list_first(gst_meta_list);
    gst_meta_list = g_list_remove_link(gst_meta_list, l);
    g_list_free (l);
  }
}

static void s_add_nvdsmeta_list_as_usermeta_list(GstBuffer* src_buffer,
        NvDsBatchMeta* batch_meta, void* frame_meta,
        gboolean is_audio)
{
  GList* gst_meta_list = NULL;

  gst_meta_list = s_get_nvdsmeta_list(src_buffer);
  NvDsMetaList *l = NULL;
  for (l = gst_meta_list; l != NULL; l = l->next)
  {
    NvDsMeta* meta = (NvDsMeta *)(l->data);
    /** We need to copy all DS meta except NVDS_BATCH_GST_META */
    if(meta->meta_type == NVDS_BATCH_GST_META) {
      continue; /**< skip batch meta */
    }

    NvDsUserMeta *user_gst_meta = nvds_acquire_user_meta_from_pool (batch_meta);
    user_gst_meta->user_meta_data = meta->copyfunc(meta->meta_data, meta->user_data);
    user_gst_meta->base_meta.meta_type = meta->meta_type;
    user_gst_meta->base_meta.copy_func = (NvDsMetaCopyFunc)meta->gst_to_nvds_meta_transform_func;
    user_gst_meta->base_meta.release_func = (NvDsMetaReleaseFunc)meta->gst_to_nvds_meta_release_func;
    if(is_audio) {
      nvds_add_user_meta_to_audio_frame((NvDsAudioFrameMeta*)frame_meta, user_gst_meta);
    }
    else {
      nvds_add_user_meta_to_frame((NvDsFrameMeta*)frame_meta, user_gst_meta);
    }
  }
  s_remove_all_list_entries(gst_meta_list);
}

static gboolean
s_foreach_metadata (GstBuffer * inbuf, GstMeta ** meta, gpointer user_data)
{
  CopyMetaData* copy_meta_ctx = (CopyMetaData*)user_data;
  const GstMetaInfo *info = (*meta)->info;
  GstMetaTransformCopy copy_data = { FALSE, 0, -1 };

  if (gst_meta_api_type_has_tag (info->api, _gst_meta_tag_memory)
      || gst_meta_api_type_has_tag (info->api, _dsmeta_quark)) {
    /* never call the transform_meta with memory specific metadata;
     * Also, no need to copy DS_META; only Gst Meta copy
     * is intended with the API nvds_copy_gst_meta_to_frame_meta /
     * nvds_copy_gst_meta_to_audio_frame_meta */
    return TRUE;
  }

  /* simply copy *meta from inbuf into outbuf */
  info->transform_func (copy_meta_ctx->outbuf, *meta, inbuf,
        _gst_meta_transform_copy, &copy_data);
  return TRUE;
}

static void *s_set_metadata_ptr(GstBuffer* src_gst_buffer)
{
  CopyMetaData copy_meta_ctx;
  /** Steps here:
   * 1) Create an empty GstBuffer, outbuf
   * 2) Copy GstMeta contents of input buffer src_gst_buffer into meta_buffer
   * 3) Now, outbuf can be saved as the user-meta pointer
   */

  /** create an empty GstBuffer to hold the metadata */
  copy_meta_ctx.outbuf = gst_buffer_new();

  /** copy ALL Gst Meta from input buffer to outbuf: */
  gboolean ret = gst_buffer_foreach_meta (src_gst_buffer, s_foreach_metadata, &copy_meta_ctx);

  if(ret == FALSE)
  {
    gst_buffer_unref(copy_meta_ctx.outbuf);
    /** return NULL indicating failure */
    copy_meta_ctx.outbuf = NULL;
  }

  return (void *)copy_meta_ctx.outbuf;
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer s_copy_user_meta(gpointer data, gpointer user_data)
{
  /**
   * Steps here:
   * 1) Input user_meta is type-casted to the GstBuffer
   * 2) Create an empty GstBuffer, outbuf
   * 3) Copy ALL Gst Meta from the input GstBuffer to outbuf
   * 4) Now, outbuf can be saved as the new user-meta pointer
   */
  CopyMetaData copy_meta_ctx;
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  GstBuffer* inbuf = (GstBuffer*)user_meta->user_meta_data;
  copy_meta_ctx.outbuf = gst_buffer_new();

  /** copy ALL Gst Meta from input buffer to outbuf: */
  gst_buffer_foreach_meta (inbuf, s_foreach_metadata, &copy_meta_ctx);

  return (gpointer)copy_meta_ctx.outbuf;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void s_release_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if(user_meta->user_meta_data) {
    GstBuffer* meta_buffer = (GstBuffer*)user_meta->user_meta_data;
    gst_buffer_unref(meta_buffer);
  }
}

void nvds_copy_gst_meta_to_frame_meta(GstBuffer* src_gst_buffer, NvDsBatchMeta* batch_meta, NvDsFrameMeta* frame_meta)
{
  /**
   * Note: Here, we add the list of N X NvDsMeta as N X frame_meta->user_meta
   * with NvDsMeta->meta_type == NvDsUserMeta->base_meta.meta_type
   */
  s_add_nvdsmeta_list_as_usermeta_list(src_gst_buffer, batch_meta,
        (void*)frame_meta, FALSE);

  /** Note: The below code ensure adding the list of plain GstMeta
   * as a single frame_meta->user_meta
   */
  if(!gst_meta_to_copy_is_available(src_gst_buffer)) {
    return;
  }
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

  /* Set NvDsUserMeta below */
  user_meta->user_meta_data = (void *)s_set_metadata_ptr(src_gst_buffer);
  user_meta->base_meta.meta_type = NVDS_BUFFER_GST_AS_FRAME_USER_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)s_copy_user_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)s_release_user_meta;

  /* We want to add NvDsUserMeta to frame level */
  nvds_add_user_meta_to_frame(frame_meta, user_meta);
}

void nvds_copy_gst_meta_to_audio_frame_meta(GstBuffer* src_gst_buffer, NvDsBatchMeta* batch_meta, NvDsAudioFrameMeta* frame_meta)
{
  /**
   * Note: Here, we add the list of N X NvDsMeta as N X frame_meta->user_meta
   * with NvDsMeta->meta_type == NvDsUserMeta->base_meta.meta_type
   */
  s_add_nvdsmeta_list_as_usermeta_list(src_gst_buffer, batch_meta,
        (void*)frame_meta, TRUE);

  /** Note: The below code ensure adding the list of plain GstMeta
   * as a single frame_meta->user_meta
   */
  if(!gst_meta_to_copy_is_available(src_gst_buffer)) {
    return;
  }
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

  /* Set NvDsUserMeta below */
  user_meta->user_meta_data = (void *)s_set_metadata_ptr(src_gst_buffer);
  user_meta->base_meta.meta_type = NVDS_BUFFER_GST_AS_FRAME_USER_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)s_copy_user_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)s_release_user_meta;

  /* We want to add NvDsUserMeta to frame level */
  nvds_add_user_meta_to_audio_frame(frame_meta, user_meta);
}
