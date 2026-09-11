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

#include "gstdsnvtrackerbin.h"
#include "gst-nvcommon.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nv_tracker_get_type ();

GST_DEBUG_CATEGORY (gst_ds_nvtracker_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvtracker_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvtracker_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvtracker_bin_debug, "nvtrackerbin", 0, "nvtrackerbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvTrackerBin, gst_ds_nvtracker_bin, GST_TYPE_BIN,
    _do_init);

static void gst_ds_nvtracker_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_ds_nvtracker_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);

static void
gst_ds_nvtracker_bin_class_init (GstDsNvTrackerBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvtracker_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvtracker_bin_get_property);

  GType tracker_elem_type = gst_nv_tracker_get_type ();
  /* All properties of the bin are same as tracker element */
  forward_properties (gobject_class, tracker_elem_type);

  /* Set sink and src pad capabilities same as tracker plugin */
  forward_pad_template (gstelement_class, tracker_elem_type, "sink");
  forward_pad_template (gstelement_class, tracker_elem_type, "src");

  gst_element_class_set_details_simple (gstelement_class,
      "NvTracker Bin", "NvTracker Bin",
      "Nvidia DeepStreamSDK NvTracker Bin. Internal Pipeline: queue->nvtracker",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvtracker_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvTrackerBin *nvtrackerbin = GST_DS_NVTRACKER_BIN (object);

  //All properties of the nvtrackerbin are same as tracker element
  guint num_specs = 0;
  GParamSpec **spec =
      g_object_class_list_properties (G_OBJECT_CLASS (GST_ELEMENT_GET_CLASS
          (nvtrackerbin->tracker)), &num_specs);

  for (guint n = 0; n < num_specs; ++n) {
    if (!g_strcmp0 (pspec->name, spec[n]->name)) {
      g_object_set_property (G_OBJECT (nvtrackerbin->tracker), pspec->name,
          value);
    }
  }

  g_free (spec);

}

static void
gst_ds_nvtracker_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvTrackerBin *nvtrackerbin = GST_DS_NVTRACKER_BIN (object);

  //All properties of the nvtrackerbin are same as tracker element
  guint num_specs = 0;
  GParamSpec **spec =
      g_object_class_list_properties (G_OBJECT_CLASS (GST_ELEMENT_GET_CLASS
          (nvtrackerbin->tracker)), &num_specs);

  for (guint n = 0; n < num_specs; ++n) {
    if (!g_strcmp0 (pspec->name, spec[n]->name)) {
      g_object_get_property (G_OBJECT (nvtrackerbin->tracker), pspec->name,
          value);
    }
  }
  g_free (spec);

}

static void
gst_ds_nvtracker_bin_init (GstDsNvTrackerBin * nvtrackerbin)
{
  nvtrackerbin->queue =
      gst_element_factory_make ("queue", "nvtracker_bin_queue");
  if (!nvtrackerbin->queue) {
    GST_ELEMENT_ERROR (nvtrackerbin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return;
  }

  nvtrackerbin->tracker =
      gst_element_factory_make ("nvtracker", "nvtracker_bin_tracker");
  if (!nvtrackerbin->tracker) {
    GST_ELEMENT_ERROR (nvtrackerbin, STREAM, FAILED,
        ("Failed to create 'nvtracker'"), (NULL));
    return;
  }

  gst_bin_add_many (GST_BIN (nvtrackerbin), nvtrackerbin->queue,
      nvtrackerbin->tracker, NULL);
  NVGSTDS_LINK_ELEMENT (nvtrackerbin->queue, nvtrackerbin->tracker);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvtrackerbin), nvtrackerbin->queue,
      "sink");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvtrackerbin), nvtrackerbin->tracker,
      "src");
}
