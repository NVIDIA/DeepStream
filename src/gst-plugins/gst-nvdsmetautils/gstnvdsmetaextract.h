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

#ifndef __GST_NVDSMETAEXTRACT_H__
#define __GST_NVDSMETAEXTRACT_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_NVDSMETAEXTRACT \
  (gst_nvdsmetaextract_get_type())
#define GST_NVDSMETAEXTRACT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSMETAEXTRACT,Gstnvdsmetaextract))
#define GST_NVDSMETAEXTRACT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSMETAEXTRACT,GstnvdsmetaextractClass))
#define GST_IS_NVDSMETAEXTRACT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSMETAEXTRACT))
#define GST_IS_NVDSMETAEXTRACT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSMETAEXTRACT))

typedef struct _Gstnvdsmetaextract      Gstnvdsmetaextract;
typedef struct _GstnvdsmetaextractClass GstnvdsmetaextractClass;

struct _Gstnvdsmetaextract
{
  GstBaseTransform element;

  GstPad *sinkpad, *srcpad;
  gboolean is_same_caps;

  /* source and sink pad caps */
  GstCaps *sinkcaps;
  GstCaps *srccaps;

  guint frame_width;
  guint frame_height;

  void *lib_handle;
  gchar* deserialization_lib_name;
  void (*deserialize_func)(GstBuffer *buf);
};

struct _GstnvdsmetaextractClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdsmetaextract_get_type (void);

gboolean nvds_metaextract_init (GstPlugin * nvdsmetaextract);

G_END_DECLS

#endif /* __GST_NVDSMETAEXTRACT_H__ */
