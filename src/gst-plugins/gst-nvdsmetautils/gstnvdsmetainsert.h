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

#ifndef __GST_NVDSMETAINSERT_H__
#define __GST_NVDSMETAINSERT_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_NVDSMETAINSERT \
  (gst_nvdsmetainsert_get_type())
#define GST_NVDSMETAINSERT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSMETAINSERT,Gstnvdsmetainsert))
#define GST_NVDSMETAINSERT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSMETAINSERT,GstnvdsmetainsertClass))
#define GST_IS_NVDSMETAINSERT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSMETAINSERT))
#define GST_IS_NVDSMETAINSERT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSMETAINSERT))

typedef struct _Gstnvdsmetainsert      Gstnvdsmetainsert;
typedef struct _GstnvdsmetainsertClass GstnvdsmetainsertClass;

struct _Gstnvdsmetainsert
{
  GstBaseTransform element;

  GstPad *sinkpad, *srcpad;
  gboolean is_same_caps;

  /* source and sink pad caps */
  GstCaps *sinkcaps;
  GstCaps *srccaps;

  gchar* serialization_lib_name;
  void *lib_handle;
  void (*serialize_func)(GstBuffer *buf);
  guint meta_mem_size;
  void *meta_mem;

};

struct _GstnvdsmetainsertClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdsmetainsert_get_type (void);

gboolean nvds_metainsert_init (GstPlugin * nvdsmetainsert);

G_END_DECLS

#endif /* __GST_NVDSMETAINSERT_H__ */
