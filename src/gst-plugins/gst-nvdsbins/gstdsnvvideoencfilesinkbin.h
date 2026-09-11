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

#ifndef __GST_DS_NVVIDEOENCFILESINK_BIN_H__
#define __GST_DS_NVVIDEOENCFILESINK_BIN_H__

#include <gst/gst.h>

#include "nvbufsurface.h"

G_BEGIN_DECLS

typedef enum
{
  FILE_CONTAINER_MP4 = 1,
  FILE_CONTAINER_MKV
} FileContainerType;

typedef enum
{
  FILE_CODEC_H264 = 1,
  FILE_CODEC_H265,
  FILE_CODEC_MPEG4
} FileCodecType;

typedef enum
{
  FILE_ENCODER_HW,
  FILE_ENCODER_SW
} FileEncodeType;

typedef enum
{
  FILE_PROFILE_BASELINE,
  FILE_PROFILE_MAIN,
  FILE_PROFILE_HIGH,
  FILE_PROFILE_MAIN10
} FileProfileType;

/* Standard GStreamer boilerplate */
typedef struct _GstDsNvVideoEncFilesinkBin
{
  GstBin bin;

  FileContainerType container;
  FileCodecType codec;
  FileEncodeType enc_type;
  FileProfileType profile;
  guint iframeinterval;
  guint bitrate;
  gchar *file_location;

  guint sync;
  guint qos;
  guint gpu_id;
  NvBufSurfaceMemType mem_type;
  gchar *create_from_config;

  GstElement *sink;
  GstElement *mux;
  GstElement *codecparse;
  GstElement *encoder;
  GstElement *cap_filter;
  GstElement *transform;
  GstElement *queue;

  GstPad *bin_sink_pad;

} GstDsNvVideoEncFilesinkBin;

typedef struct _GstDsNvVideoEncFilesinkBinClass
{
  GstBinClass parent_class;
} GstDsNvVideoEncFilesinkBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVVIDEOENCFILESINK_BIN (gst_ds_nvvideoencfilesink_bin_get_type())
#define GST_DS_NVVIDEOENCFILESINK_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVVIDEOENCFILESINK_BIN,GstDsNvVideoEncFilesinkBin))
#define GST_DS_NVVIDEOENCFILESINK_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVVIDEOENCFILESINK_BIN,GstDsNvVideoEncFilesinkBinClass))
#define GST_DS_NVVIDEOENCFILESINK_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVVIDEOENCFILESINK_BIN, GstDsNvVideoEncFilesinkBinClass))
#define GST_IS_DS_NVVIDEOENCFILESINK_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVVIDEOENCFILESINK_BIN))
#define GST_IS_DS_NVVIDEOENCFILESINK_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVVIDEOENCFILESINK_BIN))
#define GST_DS_NVVIDEOENCFILESINK_BIN_CAST(obj)  ((GstDsNvVideoEncFilesinkBin *)(obj))

GType gst_ds_nvvideoencfilesink_bin_get_type (void);

G_END_DECLS
#endif /* __GST_INFER_H__ */
