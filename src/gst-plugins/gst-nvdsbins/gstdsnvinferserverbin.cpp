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

#include "gstdsnvinferserverbin.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nvinfer_get_type (void);

GST_DEBUG_CATEGORY (gst_ds_nvinferserver_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvinferserver_bin_debug

#define DEFAULT_NVINFERSERVER_BIN_PROCESS_MODE 0

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvinferserver_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvinferserver_bin_debug, "nvinferserverbin", 0, "nvinferserverbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvInferServerBin, gst_ds_nvinferserver_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvinferserver_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvinferserver_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);

static void
gst_ds_nvinferserver_bin_class_init (GstDsNvInferServerBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvinferserver_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvinferserver_bin_get_property);

  /* All properties of the bin are same as inferserver element */

  // Mimic nvinfer instead since nvinferserver plugin might not load
  // because of missing Triton dependencies.
  GType inferserver_elem_type = gst_nvinfer_get_type ();

  /* All properties of the bin are same as infer element */
  forward_properties (gobject_class, inferserver_elem_type);

  /* Set src pad capabilities same as nvinfer element */
  forward_pad_template (gstelement_class, inferserver_elem_type, "src");
  forward_pad_template (gstelement_class, inferserver_elem_type, "sink");

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvInferServer Bin",
      "NvInferServer Bin",
      "Nvidia DeepStreamSDK TensorRT Bin. Internal Pipeline: queue->nvinferserver.",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_ds_nvinferserver_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvInferServerBin *nvinferserverbin = GST_DS_NVINFERSERVER_BIN (object);

  g_object_set_property (G_OBJECT (nvinferserverbin->nvinferserver),
      pspec->name, value);
}


static void
gst_ds_nvinferserver_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvInferServerBin *nvinferserverbin = GST_DS_NVINFERSERVER_BIN (object);

  g_object_get_property (G_OBJECT (nvinferserverbin->nvinferserver),
      pspec->name, value);
}


static void
gst_ds_nvinferserver_bin_init (GstDsNvInferServerBin * nvinferserverbin)
{
  nvinferserverbin->queue =
      gst_element_factory_make ("queue", "nvinferserver_bin_queue");
  if (!nvinferserverbin->queue) {
    GST_ELEMENT_ERROR (nvinferserverbin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return;
  }

  nvinferserverbin->nvinferserver =
      gst_element_factory_make ("nvinferserver",
      "nvinferserver_bin_nvinferserver");
  if (!nvinferserverbin->nvinferserver) {
    GST_ELEMENT_ERROR (nvinferserverbin, STREAM, FAILED,
        ("Failed to create 'nvinferserver'"), (NULL));
    return;
  }

  gst_bin_add_many (GST_BIN (nvinferserverbin), nvinferserverbin->queue,
      nvinferserverbin->nvinferserver, NULL);

  NVGSTDS_LINK_ELEMENT (nvinferserverbin->queue,
      nvinferserverbin->nvinferserver);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvinferserverbin),
      nvinferserverbin->nvinferserver, "src");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvinferserverbin),
      nvinferserverbin->queue, "sink");
}
