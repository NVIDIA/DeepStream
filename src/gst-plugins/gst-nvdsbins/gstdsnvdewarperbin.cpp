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

#include "gstdsnvdewarperbin.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nvvideoconvert_get_type ();
extern "C" GType gst_nvdewarper_get_type (void);

GST_DEBUG_CATEGORY (gst_ds_nvdewarper_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvdewarper_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvdewarper_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvdewarper_bin_parent_class, "nvdewarperbin", 0, "nvdewarperbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvDewarperBin, gst_ds_nvdewarper_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvdewarper_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_ds_nvdewarper_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);

static void
gst_ds_nvdewarper_bin_class_init (GstDsNvDewarperBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvdewarper_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvdewarper_bin_get_property);

  GType dewarper_elem_type = gst_nvdewarper_get_type ();
  GType nvvidconv_elem_type = gst_nvvideoconvert_get_type ();
  /* All properties of the bin are same as dewarper element */
  forward_properties (gobject_class, dewarper_elem_type);

  forward_pad_template (gstelement_class, dewarper_elem_type, "src");
  forward_pad_template (gstelement_class, nvvidconv_elem_type, "sink");

  gst_element_class_set_details_simple (gstelement_class,
      "NvDewarper Bin",
      "NvDewarper Bin",
      "Nvidia DeepStreamSDK NvDewarper Bin. Internal Pipeline: queue->nvvidconv->queue->nvdewarper",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvdewarper_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvDewarperBin *nvdewarperbin = GST_DS_NVDEWARPER_BIN (object);
  if (!g_strcmp0 (pspec->name, "gpu-id")
      || !g_strcmp0 (pspec->name, "nvbuf-memory-type")) {
    g_object_set_property (G_OBJECT (nvdewarperbin->nvvidconv), pspec->name,
        value);
  }
  g_object_set_property (G_OBJECT (nvdewarperbin->nvdewarper), pspec->name,
      value);
}

static void
gst_ds_nvdewarper_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvDewarperBin *nvdewarperbin = GST_DS_NVDEWARPER_BIN (object);
  g_object_get_property (G_OBJECT (nvdewarperbin->nvdewarper), pspec->name,
      value);
}


static void
gst_ds_nvdewarper_bin_init (GstDsNvDewarperBin * nvdewarperbin)
{
  nvdewarperbin->nvvidconv =
      gst_element_factory_make ("nvvideoconvert", "nvdewarper_bin_nvvidconv");
  if (!nvdewarperbin->nvvidconv) {
    GST_ELEMENT_ERROR (nvdewarperbin, STREAM, FAILED,
        ("Failed to create 'nvvideoconvert'"), (NULL));
    return;
  }

  nvdewarperbin->queue =
      gst_element_factory_make ("queue", "nvdewarper_bin_queue");
  if (!nvdewarperbin->queue) {
    GST_ELEMENT_ERROR (nvdewarperbin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return;
  }

  nvdewarperbin->conv_queue =
      gst_element_factory_make ("queue", "nvdewarper_bin_conv_queue");
  if (!nvdewarperbin->conv_queue) {
    GST_ELEMENT_ERROR (nvdewarperbin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return;
  }

  nvdewarperbin->nvdewarper =
      gst_element_factory_make ("nvdewarper", "nvdewarper_bin_nvdewarper");
  if (!nvdewarperbin->nvdewarper) {
    GST_ELEMENT_ERROR (nvdewarperbin, STREAM, FAILED,
        ("Failed to create 'nvvideoconvert'"), (NULL));
    return;
  }

  gst_bin_add_many (GST_BIN (nvdewarperbin), nvdewarperbin->queue,
      nvdewarperbin->conv_queue, nvdewarperbin->nvvidconv,
      nvdewarperbin->nvdewarper, NULL);

  NVGSTDS_LINK_ELEMENT (nvdewarperbin->queue, nvdewarperbin->nvvidconv);
  NVGSTDS_LINK_ELEMENT (nvdewarperbin->nvvidconv, nvdewarperbin->conv_queue);
  NVGSTDS_LINK_ELEMENT (nvdewarperbin->conv_queue, nvdewarperbin->nvdewarper);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvdewarperbin),
      nvdewarperbin->nvdewarper, "src");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvdewarperbin), nvdewarperbin->queue,
      "sink");
}
