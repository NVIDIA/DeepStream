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

#include "gstdsnvanalyticsbin.h"
#include "gst-nvcommon.h"
#include "gstnvdsbinutils.h"

GST_DEBUG_CATEGORY (gst_ds_nvanalytics_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvanalytics_bin_debug

extern "C" GType gst_nvdsanalytics_get_type (void);

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvanalytics_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvanalytics_bin_debug, "nvdsanalyticsbin", 0, "nvdsanalyticsbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvAnalyticsBin, gst_ds_nvanalytics_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvanalytics_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvanalytics_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);

static void
gst_ds_nvanalytics_bin_class_init (GstDsNvAnalyticsBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvanalytics_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvanalytics_bin_get_property);


  GType child_elem_type = gst_nvdsanalytics_get_type ();
  /* All properties of the bin are same as analytics element */
  forward_properties (gobject_class, child_elem_type);

  /* Set sink and src pad capabilities same as analytics plugin */
  GstElementClass *analytics_class =
      GST_ELEMENT_GET_CLASS (GST_ELEMENT (g_object_new
          (gst_nvdsanalytics_get_type (),
              NULL)));
  const GList *pads = gst_element_class_get_pad_template_list (analytics_class);

  while (pads) {
    GstPadTemplate *padtemplate = (GstPadTemplate *) (pads->data);
    gst_element_class_add_pad_template (gstelement_class, padtemplate);
    pads = g_list_next (pads);
  }

  gst_element_class_set_details_simple (gstelement_class,
      "NvAnalytics Bin", "NvAnalytics Bin",
      "Process analytics algorithm on objects. Internal Pipeline: queue->nvdsanalytics",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvanalytics_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvAnalyticsBin *nvanalyticsbin = GST_DS_NVANALYTICS_BIN (object);

  //All properties of the nvanalyticsbin are same as analytics element
  g_object_set_property (G_OBJECT (nvanalyticsbin->analytics), pspec->name,
      value);
}

static void
gst_ds_nvanalytics_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvAnalyticsBin *nvanalyticsbin = GST_DS_NVANALYTICS_BIN (object);

  //All properties of the nvanalyticsbin are same as analytics element
  g_object_get_property (G_OBJECT (nvanalyticsbin->analytics), pspec->name,
      value);

}

static void
gst_ds_nvanalytics_bin_init (GstDsNvAnalyticsBin * nvanalyticsbin)
{
  nvanalyticsbin->queue =
      gst_element_factory_make ("queue", "nvanalytics_bin_queue");
  if (!nvanalyticsbin->queue) {
    GST_ELEMENT_ERROR (nvanalyticsbin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return;
  }

  nvanalyticsbin->analytics =
      gst_element_factory_make ("nvdsanalytics", "nvanalytics_bin_analytics");
  if (!nvanalyticsbin->analytics) {
    GST_ELEMENT_ERROR (nvanalyticsbin, STREAM, FAILED,
        ("Failed to create 'nvdsanalytics'"), (NULL));
    return;
  }

  gst_bin_add_many (GST_BIN (nvanalyticsbin), nvanalyticsbin->queue,
      nvanalyticsbin->analytics, NULL);
  NVGSTDS_LINK_ELEMENT (nvanalyticsbin->queue, nvanalyticsbin->analytics);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvanalyticsbin),
      nvanalyticsbin->queue, "sink");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvanalyticsbin),
      nvanalyticsbin->analytics, "src");
}
