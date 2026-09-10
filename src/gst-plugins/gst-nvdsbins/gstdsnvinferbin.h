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

#ifndef __GST_DS_NVINFER_BIN_H__
#define __GST_DS_NVINFER_BIN_H__

#include <gst/video/video.h>

G_BEGIN_DECLS
/* nvinfer signals */
enum
{
  /* Signal emitted to notify app about model update completion with
   * success/error messages. */
  SIGNAL_MODEL_UPDATED,
  LAST_SIGNAL,
};

/* Standard GStreamer boilerplate */
typedef struct _GstDsNvInferBin
{
  GstBin bin;
  GstElement *queue;
  GstElement *nvinfer;
} GstDsNvInferBin;

typedef struct _GstDsNvInferBinClass
{
  GstBinClass parent_class;

  /** Signals */
  /** signal : model-update
    * err: int, error result, type NvDsInferStatus.
    * cfg_file: update cfg file.
    */
  void (*model_updated) (GstDsNvInferBin *, gint err, const gchar * cfg_file);
} GstDsNvInferBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVINFER_BIN (gst_ds_nvinfer_bin_get_type())
#define GST_DS_NVINFER_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVINFER_BIN,GstDsNvInferBin))
#define GST_DS_NVINFER_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVINFER_BIN,GstDsNvInferBinClass))
#define GST_DS_NVINFER_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVINFER_BIN, GstDsNvInferBinClass))
#define GST_IS_DS_NVINFER_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVINFER_BIN))
#define GST_IS_DS_NVINFER_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVINFER_BIN))
#define GST_DS_NVINFER_BIN_CAST(obj)  ((GstDsNvInferBin *)(obj))

GType gst_ds_nvinfer_bin_get_type (void);

G_END_DECLS
#endif /* __GST_DS_NVINFER_BIN_H__ */
