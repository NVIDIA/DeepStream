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

#ifndef __GST_DS_NVTRACKER_BIN_H__
#define __GST_DS_NVTRACKER_BIN_H__

#include <gst/video/video.h>

G_BEGIN_DECLS

/* Standard GStreamer boilerplate */
typedef struct _GstDsNvTrackerBin
{
  GstBin bin;

  GstElement *queue;
  GstElement *tracker;

} GstDsNvTrackerBin;

typedef struct _GstDsNvTrackerBinClass
{
  GstBinClass parent_class;
} GstDsNvTrackerBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVTRACKER_BIN (gst_ds_nvtracker_bin_get_type())
#define GST_DS_NVTRACKER_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVTRACKER_BIN,GstDsNvTrackerBin))
#define GST_DS_NVTRACKER_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVTRACKER_BIN,GstDsNvTrackerBinClass))
#define GST_DS_NVTRACKER_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVTRACKER_BIN, GstDsNvTrackerBinClass))
#define GST_IS_DS_NVTRACKER_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVTRACKER_BIN))
#define GST_IS_DS_NVTRACKER_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVTRACKER_BIN))
#define GST_DS_NVTRACKER_BIN_CAST(obj)  ((GstDsNvTrackerBin *)(obj))

GType gst_ds_nvtracker_bin_get_type (void);

G_END_DECLS
#endif /* __GST_DS_NVTRACKER_BIN_H__ */
