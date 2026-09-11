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

#include "gstdsnvbuffersyncbin.h"
#include "gst-nvcommon.h"
#include "gstnvdsbinutils.h"
#include <string>

extern "C" GType gst_nv_buffersync_get_type ();

GST_DEBUG_CATEGORY (gst_ds_nvbuffersync_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvbuffersync_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvbuffersync_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvbuffersync_bin_debug, "nvbuffersyncbin", 0, "nvbuffersyncbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvBufferSyncBin, gst_ds_nvbuffersync_bin,
    GST_TYPE_BIN, _do_init);

static GstPad *gst_ds_nvbuffersync_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps);
static GstPadProbeReturn
buffersync_probe_func (GstPad * pad, GstPadProbeInfo * info, gpointer u_data);
static GstStateChangeReturn
gst_ds_nvbuffersync_bin_change_state (GstElement * element,
    GstStateChange transition);

static void
gst_ds_nvbuffersync_bin_class_init (GstDsNvBufferSyncBinClass * klass)
{
  GstElementClass *gstelement_class;

  gstelement_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_add_pad_template (gstelement_class,
      gst_pad_template_new ("sink_%u", GST_PAD_SINK, GST_PAD_REQUEST,
          GST_CAPS_ANY)
      );

  gst_element_class_add_pad_template (gstelement_class,
      gst_pad_template_new ("sync_sink", GST_PAD_SINK, GST_PAD_ALWAYS,
          GST_CAPS_ANY)
      );

  gst_element_class_add_pad_template (gstelement_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_CAPS_ANY)
      );

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_ds_nvbuffersync_request_new_pad);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvbuffersync_bin_change_state);

  gst_element_class_set_details_simple (gstelement_class,
      "NvBufferSync Bin", "NvBufferSync Bin",
      "Ensures that processing by upstream components on buffers received on "
      "`sync_sink` pad has finished. This is useful in cases where a buffer is "
      "provided to multiple elements simultaneously using an element like "
      "`tee` and it must be ensured that the elements have finished "
      "processing before pushing it downstream.",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries"
      " @ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvbuffersync_bin_init (GstDsNvBufferSyncBin * nvbuffersyncbin)
{
  GstElement *element = GST_ELEMENT (nvbuffersyncbin);
  nvbuffersyncbin->queue =
      gst_element_factory_make ("queue", "nvbuffersync_bin_queue");
  if (!nvbuffersyncbin->queue) {
    GST_ELEMENT_ERROR (nvbuffersyncbin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return;
  }

  gst_bin_add (GST_BIN (nvbuffersyncbin), nvbuffersyncbin->queue);

  GstElementClass *klass = GST_ELEMENT_GET_CLASS(nvbuffersyncbin);
  GstPadTemplate *tmpl = gst_element_class_get_pad_template (klass, "sync_sink");
  GstPadUPtr qpad = gst_element_get_static_pad (nvbuffersyncbin->queue, "sink");
  GstPad *gpad = gst_ghost_pad_new_from_template ("sync_sink", qpad, tmpl);
  gst_element_add_pad (element, gpad);

  NVGSTDS_BIN_ADD_GHOST_PAD (element, nvbuffersyncbin->queue, "src");

  NVGSTDS_ELEM_ADD_PROBE (nvbuffersyncbin,
      nvbuffersyncbin->queue, "src", buffersync_probe_func,
      GST_PAD_PROBE_TYPE_BUFFER, nvbuffersyncbin);
}

static GstStateChangeReturn
gst_ds_nvbuffersync_bin_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvBufferSyncBin *nvbuffersyncbin = GST_DS_NVBUFFERSYNC_BIN (element);
  if (transition == GST_STATE_CHANGE_READY_TO_PAUSED) {
    GstPadUPtr pad = gst_element_get_static_pad (element, "sync_sink");
    if (!gst_pad_is_linked (pad)) {
      GST_ELEMENT_ERROR (nvbuffersyncbin, STREAM, FAILED,
          ("'sink' pad is required to be linked."), (NULL));
      return GST_STATE_CHANGE_FAILURE;
    }
  }
  return GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
}

static GstPadProbeReturn
buffersync_probe_func (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  while (GST_OBJECT_REFCOUNT_VALUE (info->data) > 1)
    g_usleep (100);
  return GST_PAD_PROBE_OK;
}

static GstPad *
gst_ds_nvbuffersync_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps)
{
  std::string pad_name;

  if (name) {
    guint pad_idx;

    if (GstPadUPtr gstpad = gst_element_get_static_pad (element, name)) {
      return nullptr;
    }

    if (sscanf (name, "sink_%u", &pad_idx) < 1) {
      GST_ERROR_OBJECT (element,
          "Pad should be named 'sink_%%u' when requesting a pad");
      return NULL;
    }
  }

  if (!name || !g_strcmp0 (name, "sink_%u")) {
    for (size_t i = 0; i < G_MAXINT; i++) {
      pad_name = "sink_" + std::to_string (i);
      GstPadUPtr gstpad =
          gst_element_get_static_pad (element, pad_name.c_str ());
      if (!gstpad) {
        name = pad_name.c_str ();
        break;
      }
    }
  }

  GstPad *bin_pad = gst_pad_new_from_template (templ, name);

  gst_pad_set_chain_function (bin_pad,[](GstPad * pad, GstObject * parent,
          GstBuffer * buf) {
        gst_buffer_unref (buf);
        return GST_FLOW_OK;
      }
  );
  gst_pad_set_event_function (bin_pad,[](GstPad *, GstObject *, GstEvent *) {
        return (gboolean) TRUE;
      }
  );
  gst_element_add_pad (element, bin_pad);

  return bin_pad;
}
