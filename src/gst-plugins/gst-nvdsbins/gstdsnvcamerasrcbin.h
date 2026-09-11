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

#ifndef __GST_DS_NVCAMERASRC_BIN_H__
#define __GST_DS_NVCAMERASRC_BIN_H__

#include <gst/video/video.h>

G_BEGIN_DECLS

typedef enum
{
  V4L2,
  NVARGUS,
} GstDsNvCameraSrcType;

typedef struct _GstDsNvCameraSrcBin
{
  GstBin bin;

  GstDsNvCameraSrcType type;
  guint width;
  guint height;
  guint fps_n;
  guint fps_d;

  guint gpu_id;
  guint memtype;

  guint sensor_id;
  gchar *v4l2_device;

  GstPad *bin_src_pad;

  GstElement *src_elem;

  gint source_id;

} GstDsNvCameraSrcBin;

typedef struct _GstDsNvCameraSrcBinClass
{
  GstBinClass parent_class;
} GstDsNvCameraSrcBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVCAMERASRC_BIN (gst_ds_nvcamerasrc_bin_get_type())
#define GST_DS_NVCAMERASRC_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVCAMERASRC_BIN,GstDsNvCameraSrcBin))
#define GST_DS_NVCAMERASRC_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVCAMERASRC_BIN,GstDsNvCameraSrcBinClass))
#define GST_DS_NVCAMERASRC_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVCAMERASRC_BIN, GstDsNvCameraSrcBinClass))
#define GST_IS_DS_NVCAMERASRC_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVCAMERASRC_BIN))
#define GST_IS_DS_NVCAMERASRC_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVCAMERASRC_BIN))
#define GST_DS_NVCAMERASRC_BIN_CAST(obj)  ((GstDsNvCameraSrcBin *)(obj))

GType gst_ds_nvcamerasrc_bin_get_type (void);

G_END_DECLS
#endif /* __GST_DS_NVCAMERASRC_BIN_H__ */
