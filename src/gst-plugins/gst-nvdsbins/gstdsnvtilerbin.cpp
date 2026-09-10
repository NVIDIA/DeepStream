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

#include "gstdsnvtilerbin.h"
#include "gst-nvcommon.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nvmultistreamtiler_get_type (void);


GST_DEBUG_CATEGORY (gst_ds_nvtiler_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvtiler_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvtiler_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvtiler_bin_debug, "nvtilerbin", 0, "nvtilerbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvTilerBin, gst_ds_nvtiler_bin, GST_TYPE_BIN,
    _do_init);

static void gst_ds_nvtiler_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_ds_nvtiler_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);

static void
gst_ds_nvtiler_bin_class_init (GstDsNvTilerBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvtiler_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvtiler_bin_get_property);

  GType child_elem_type = gst_nvmultistreamtiler_get_type ();
  /* All properties of the bin are same as tiler element */
  forward_properties (gobject_class, child_elem_type);

  /* Set sink and src pad capabilities same as tiler plugin */
  GstElementClass *tiler_class =
      GST_ELEMENT_GET_CLASS (GST_ELEMENT (g_object_new
          (gst_nvmultistreamtiler_get_type (), NULL)));
  const GList *tiler_pads =
      gst_element_class_get_pad_template_list (tiler_class);

  while (tiler_pads) {
    GstPadTemplate *padtemplate = (GstPadTemplate *) (tiler_pads->data);
    gst_element_class_add_pad_template (gstelement_class, padtemplate);
    tiler_pads = g_list_next (tiler_pads);
  }


  gst_element_class_set_details_simple (gstelement_class,
      "NvTiler Bin", "NvTiler Bin",
      "Tile input multistream buffer into a 2D array. Internal Pipeline: queue->nvmultistreamtiler",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvtiler_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvTilerBin *nvtilerbin = GST_DS_NVTILER_BIN (object);

  //All properties of the nvtilerbin are same as tiler element
  g_object_set_property (G_OBJECT (nvtilerbin->tiler), pspec->name, value);

}

static void
gst_ds_nvtiler_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvTilerBin *nvtilerbin = GST_DS_NVTILER_BIN (object);

  //All properties of the nvtilerbin are same as tiler element
  g_object_get_property (G_OBJECT (nvtilerbin->tiler), pspec->name, value);

}

static void
gst_ds_nvtiler_bin_init (GstDsNvTilerBin * nvtilerbin)
{
  nvtilerbin->queue = gst_element_factory_make ("queue", "nvtiler_bin_queue");
  if (!nvtilerbin->queue) {
    GST_ELEMENT_ERROR (nvtilerbin, STREAM, FAILED, ("Failed to create 'queue'"),
        (NULL));
    return;
  }

  nvtilerbin->tiler =
      gst_element_factory_make ("nvmultistreamtiler", "nvtiler_bin_tiler");
  if (!nvtilerbin->tiler) {
    GST_ELEMENT_ERROR (nvtilerbin, STREAM, FAILED,
        ("Failed to create 'nvmultistreamtiler'"), (NULL));
    return;
  }

  gst_bin_add_many (GST_BIN (nvtilerbin), nvtilerbin->queue, nvtilerbin->tiler,
      NULL);
  NVGSTDS_LINK_ELEMENT (nvtilerbin->queue, nvtilerbin->tiler);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvtilerbin), nvtilerbin->queue,
      "sink");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvtilerbin), nvtilerbin->tiler,
      "src");
}
