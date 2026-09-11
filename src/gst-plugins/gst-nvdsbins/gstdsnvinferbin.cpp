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

#include "gstdsnvinferbin.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nvinfer_get_type (void);

GST_DEBUG_CATEGORY (gst_ds_nvinfer_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvinfer_bin_debug

#define DEFAULT_NVINFER_BIN_PROCESS_MODE 0

guint gst_nvinferbin_signals[LAST_SIGNAL] = { 0 };

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvinfer_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvinfer_bin_debug, "nvinferbin", 0, "nvinferbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvInferBin, gst_ds_nvinfer_bin, GST_TYPE_BIN,
    _do_init);

static void gst_ds_nvinfer_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_ds_nvinfer_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);

static void
gst_ds_nvinfer_bin_class_init (GstDsNvInferBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvinfer_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvinfer_bin_get_property);

  GType infer_elem_type = gst_nvinfer_get_type ();
  /* All properties of the bin are same as infer element */
  forward_properties (gobject_class, infer_elem_type);

  /* Set src pad capabilities same as nvinfer element */
  forward_pad_template (gstelement_class, infer_elem_type, "src");
  forward_pad_template (gstelement_class, infer_elem_type, "sink");

  gst_nvinferbin_signals[SIGNAL_MODEL_UPDATED] =
      g_signal_new ("model-updated",
      G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstDsNvInferBinClass, model_updated),
      NULL, NULL, NULL, G_TYPE_NONE, 2, G_TYPE_INT, G_TYPE_STRING);

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvInfer Bin",
      "NvInfer Bin",
      "Nvidia DeepStreamSDK TensorRT Bin. Internal Pipeline: queue->nvinfer.",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_ds_nvinfer_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvInferBin *nvinferbin = GST_DS_NVINFER_BIN (object);

  //All properties of the bin are same as nvinfer element
  g_object_set_property (G_OBJECT (nvinferbin->nvinfer), pspec->name, value);

}


static void
gst_ds_nvinfer_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvInferBin *nvinferbin = GST_DS_NVINFER_BIN (object);

  g_object_get_property (G_OBJECT (nvinferbin->nvinfer), pspec->name, value);
}

static void
infer_model_updated_cb (GstElement * gie, gint err, const gchar * config_file,
    gpointer user_data)
{
  GstDsNvInferBin *nvinferbin = (GstDsNvInferBin *) user_data;
  g_signal_emit (nvinferbin, gst_nvinferbin_signals[SIGNAL_MODEL_UPDATED], 0,
      err, config_file);
}

static void
gst_ds_nvinfer_bin_init (GstDsNvInferBin * nvinferbin)
{
  nvinferbin->queue = gst_element_factory_make ("queue", "nvinfer_bin_queue");
  if (!nvinferbin->queue) {
    GST_ELEMENT_ERROR (nvinferbin, STREAM, FAILED, ("Failed to create 'queue'"),
        (NULL));
    return;
  }

  nvinferbin->nvinfer =
      gst_element_factory_make ("nvinfer", "nvinfer_bin_nvinfer");
  if (!nvinferbin->nvinfer) {
    GST_ELEMENT_ERROR (nvinferbin, STREAM, FAILED,
        ("Failed to create 'nvinfer'"), (NULL));
    return;
  }

  g_signal_connect (nvinferbin->nvinfer, "model-updated",
      G_CALLBACK (infer_model_updated_cb), nvinferbin);

  gst_bin_add_many (GST_BIN (nvinferbin), nvinferbin->queue,
      nvinferbin->nvinfer, NULL);

  NVGSTDS_LINK_ELEMENT (nvinferbin->queue, nvinferbin->nvinfer);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvinferbin), nvinferbin->nvinfer,
      "src");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvinferbin), nvinferbin->queue,
      "sink");
}
