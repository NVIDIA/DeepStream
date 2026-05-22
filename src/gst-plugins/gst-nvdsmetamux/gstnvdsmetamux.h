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

#ifndef __GST_NVDSMETAMUX_H__
#define __GST_NVDSMETAMUX_H__

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>
#include "gstnvdsmeta.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <functional>

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsmetamux"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA batch meta mux plugin for integration with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "NVIDIA DeepStream batch meta mux form different model"
#define URL "http://nvidia.com/"

#ifdef __cplusplus
extern "C" {
#endif

G_BEGIN_DECLS

#define GST_TYPE_NVDSMETAMUX_PAD (gst_nvdsmetamux_pad_get_type())
#define GST_NVDSMETAMUX_PAD(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSMETAMUX_PAD, GstNvDsMetaMuxPad))
#define GST_NVDSMETAMUX_PAD_CAST(obj) ((GstNvDsMetaMuxPad *)(obj))
#define GST_NVDSMETAMUX_PAD_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSMETAMUX_PAD, GstNvDsMetaMuxPad))
#define GST_IS_NVDSMETAMUX_PAD(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSMETAMUX_PAD))
#define GST_IS_NVDSMETAMUX_PAD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSMETAMUX_PAD))

typedef struct _GstNvDsMetaMuxPad GstNvDsMetaMuxPad;
typedef struct _GstNvDsMetaMuxPadClass GstNvDsMetaMuxPadClass;
typedef struct _GstNvDsMetaMux GstNvDsMetaMux;
typedef struct _GstNvDsMetaMuxClass GstNvDsMetaMuxClass;

#define GST_TYPE_NVDSMETAMUX \
  (gst_nvdsmetamux_get_type ())
#define GST_NVDSMETAMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVDSMETAMUX, GstNvDsMetaMux))
#define GST_NVDSMETAMUX_CAST(obj) ((GstNvDsMetaMux *)obj)
#define GST_NVDSMETAMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_NVDSMETAMUX, GstNvDsMetaMuxClass))
#define GST_IS_NVDSMETAMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVDSMETAMUX))
#define GST_IS_NVDSMETAMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_NVDSMETAMUX))

typedef struct
{
  /** list of source_id vectors per model */
  std::vector<gint> source_id_vector;
} GstNvDsMetaMuxModelSource;

struct _GstNvDsMetaMuxPad
{
  GstAggregatorPad aggregator_pad;

  GstBufferList *buf_list;
  std::vector <gint> source_ids;
};

struct _GstNvDsMetaMuxPadClass {
  GstAggregatorPadClass parent;
};

struct _GstNvDsMetaMux {
  GstAggregator aggregator;

  GstPad *srcpad;
  gboolean enable;
  gchar *active_sink_pad;
  GstClockTimeDiff pts_tolerance;
  gchar *config_file_path;
  GstNvDsMetaMuxPad *active_pad;
  gboolean config_file_parse_successful;
  std::unordered_map<gint, GstNvDsMetaMuxModelSource> model_source_map;
  GHashTable *pts_table;
};

struct _GstNvDsMetaMuxClass {
  GstAggregatorClass parent;
};

GType    gst_nvdsmetamux_pad_get_type(void);
GType    gst_nvdsmetamux_get_type    (void);

G_END_DECLS

#ifdef __cplusplus
}
#endif

#endif
