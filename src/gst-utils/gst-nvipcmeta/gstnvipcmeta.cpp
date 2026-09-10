/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gstnvdsmeta.h"
#include "gstnvipcmeta.h"

static gboolean
gst_nv_ipc_init_func (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  GST_DEBUG ("init called on buffer %p, meta %p", buffer, meta);
  /* nothing to init really, the init function is mostly for allocating
   * additional memory or doing special setup as part of adding the metadata to
   * the buffer*/
  return TRUE;
}

static void
gst_nv_ipc_free_func (GstMeta * meta, GstBuffer * buffer)
{
  GST_DEBUG ("free called on buffer %p, meta %p", buffer, meta);
  /* nothing to free really */
}

static gboolean
gst_nv_ipc_transform_func (GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  GstNvIpcMeta *test, *tmeta = (GstNvIpcMeta *) meta;

  GST_DEBUG ("transform %s called from buffer %p to %p, meta %p",
      g_quark_to_string (type), buffer, transbuf, meta);

  if (GST_META_TRANSFORM_IS_COPY (type)) {
    test = GST_NV_IPC_META_ADD (transbuf);
    if (!test)
      return FALSE;

    GST_DEBUG ("copy ipc metadata");
    test->num_rects = tmeta->num_rects;
    for (guint k = 0; k < tmeta->num_rects; k++)
    {
      memcpy(&test->rect_params[k], &tmeta->rect_params[k], sizeof(GstNvIpcRectParams));
    }
  } else {
    /* return FALSE, if transform type is not supported */
    return FALSE;
  }

  return TRUE;
}

GType
gst_nv_ipc_meta_api_get_type (void)
{
  static GType type;
  static const gchar *tags[] = { GST_NV_IPC_META_STRING, NULL };

  if (g_once_init_enter (&type)) {
    GType _type = gst_meta_api_type_register ("GstNvIpcMetaAPI", tags);
    g_once_init_leave (&type, _type);
  }
  return type;
}

const GstMetaInfo *
gst_nv_ipc_meta_get_info (void)
{
  static const GstMetaInfo *nv_ipc_meta_info = NULL;

  if (g_once_init_enter ((GstMetaInfo **) & nv_ipc_meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (GST_NV_IPC_META_API_TYPE,
        "GstNvIpcMeta",
        sizeof (GstNvIpcMeta),
        gst_nv_ipc_init_func, gst_nv_ipc_free_func, gst_nv_ipc_transform_func);
    g_once_init_leave ((GstMetaInfo **) & nv_ipc_meta_info, (GstMetaInfo *) mi);
  }
  return nv_ipc_meta_info;
}

void
serialize_meta (GstBuffer *buffer, guint8 **data, guint *length)
{
  GstMeta *gst_meta = NULL;
  NvDsMeta *dsmeta = NULL;
  gpointer state = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsMetaList *l = NULL;
  NvDsMetaList *full_obj_meta_list = NULL;
  NvDsObjectMeta *object_meta = NULL;
  GstNvIpcMeta meta = {};
  GByteArray *array = NULL;

  if (buffer == NULL || data == NULL || length == NULL) {
    GST_ERROR ("%s: Invalid arguments for buffer %p", __func__, buffer);
    return;
  }

  while ((gst_meta = gst_buffer_iterate_meta (buffer, &state)))
  {
    if (gst_meta_api_type_has_tag(gst_meta->info->api, g_quark_from_static_string(NVDS_META_STRING)))
    {
      dsmeta = (NvDsMeta *) gst_meta;
      if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *)dsmeta->meta_data;
        break;
      }
    }
  }

  if (!batch_meta) {
    GST_DEBUG ("%s: DS batch meta not found for buffer %p", __func__, buffer);
    return;
  }

  full_obj_meta_list = batch_meta->obj_meta_pool->full_list;

  memset(&meta, 0, sizeof(GstNvIpcMeta));
  for (l = full_obj_meta_list; l != NULL; l = l->next) {
    object_meta = (NvDsObjectMeta *) (l->data);
    NvOSD_RectParams rect_params = object_meta->rect_params;
    meta.rect_params[meta.num_rects].left = rect_params.left;
    meta.rect_params[meta.num_rects].top = rect_params.top;
    meta.rect_params[meta.num_rects].width = rect_params.width;
    meta.rect_params[meta.num_rects].height = rect_params.height;
    meta.num_rects++;
  }

  array = g_byte_array_new ();
  g_byte_array_append(array, (const guint8 *)&meta.num_rects, sizeof(guint));
  for (guint k = 0; k < meta.num_rects; k++) {
    g_byte_array_append(array, (const guint8 *)&meta.rect_params[k], sizeof(GstNvIpcRectParams));
  }

  *data = (guint8 *) (g_malloc(array->len + 1));
  if (*data == NULL) {
    GST_ERROR ("%s: malloc failed for buffer %p", __func__, buffer);
    g_byte_array_unref(array);
    return;
  }
  memcpy(*data, array->data, array->len);
  *length = array->len;
}

void
deserialize_meta (GstBuffer *buffer, guint8 *data, guint length)
{
  guint32 offset = 0;
  guint *num_rects = NULL;
  GstNvIpcMeta *ipc_meta = NULL;
  GstNvIpcRectParams *rect_params = NULL;

  if (buffer == NULL || data == NULL) {
    GST_ERROR ("%s: Invalid arguments for buffer %p", __func__, buffer);
    return;
  }

  if (length == 0) {
    GST_DEBUG ("%s: meta length is zero for buffer %p", __func__, buffer);
    return;
  }

  num_rects = (guint *) data;

  if (*num_rects == 0) {
    GST_DEBUG ("%s: Num rects is zero for buffer %p", __func__, buffer);
    return;
  }

  ipc_meta = GST_NV_IPC_META_ADD (buffer);
  if (ipc_meta == NULL) {
    GST_ERROR ("%s: IPC meta is NULL for buffer %p", __func__, buffer);
    return;
  }

  ipc_meta->num_rects = *num_rects;
  offset += sizeof(guint);

  for (guint k = 0; k < ipc_meta->num_rects; k++)
  {
    rect_params = (GstNvIpcRectParams *)(data + offset);
    memcpy(&ipc_meta->rect_params[k], rect_params, sizeof(GstNvIpcRectParams));
    offset += sizeof(GstNvIpcRectParams);
  }
}
