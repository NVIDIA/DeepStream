/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
