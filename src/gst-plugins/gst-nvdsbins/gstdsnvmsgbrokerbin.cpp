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

#include <glib/gprintf.h>
#include <gst/video/video.h>
#include <gst/audio/audio.h>
#include "gstdsnvmsgbrokerbin.h"
#include "gst-nvcommon.h"
#include <string.h>
#include "nvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "gstnvdsmeta.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nvmsgbroker_get_type (void);
extern "C" GType gst_nvmsgconv_get_type (void);

GST_DEBUG_CATEGORY (gst_ds_nvmsgbroker_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvmsgbroker_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvmsgbroker_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvmsgbroker_bin_debug, "nvnvmsgbrokerbin", 0, "nvnvmsgbrokerbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvMsgBrokerBin, gst_ds_nvmsgbroker_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvmsgbroker_bin_finalize (GObject * object);
static void gst_ds_nvmsgbroker_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvmsgbroker_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);
static GstStateChangeReturn
gst_ds_nvmsgbroker_change_state (GstElement * element,
    GstStateChange transition);


static GstStaticPadTemplate gst_nvmsgbrokerbin_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

enum
{
  PROP_0,
  PROP_DISABLE_MSGCONV,
  PROP_LAST
};

static const gchar *msgconv_prop_map[][2] = {
  {"config", "msg-conv-config"},
  {"msg2p-lib", "msg-conv-msg2p-lib"},
  {"payload-type", "msg-conv-payload-type"},
  {"comp-id", "msg-conv-comp-id"},
  {"debug-payload-dir", "debug-payload-dir"},
  {"multiple-payloads", "multiple-payloads"},
  {"frame-interval", "msg-conv-frame-interval"},
  {"msg2p-newapi", "msg-conv-msg2p-new-api"},
};

static const gchar *msgbroker_prop_map[][2] = {
  {"proto-lib", "msg-broker-proto-lib"},
  {"conn-str", "msg-broker-conn-str"},
  {"topic", "topic"},
  {"sync", "sync"},
  {"config", "msg-broker-config"},
  {"comp-id", "msg-broker-comp-id"},
  {"new-api", "new-api"},
};

#define MAX_TIME_STAMP_LEN 32

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

static void
gst_ds_nvmsgbroker_bin_class_init (GstDsNvMsgBrokerBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvmsgbroker_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvmsgbroker_bin_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_ds_nvmsgbroker_bin_finalize);


  g_object_class_install_property (gobject_class, PROP_DISABLE_MSGCONV,
      g_param_spec_boolean ("disable-msgconv", "Disable Msgconv",
          "Sync on the clock", FALSE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  guint num_msgconv_props =
      (sizeof (msgconv_prop_map) / sizeof (msgconv_prop_map[0]));
  guint num_msgbroker_props =
      (sizeof (msgbroker_prop_map) / sizeof (msgbroker_prop_map[0]));
  guint prop_index = PROP_LAST;

  klass->orig_prop_name_id_map = g_ptr_array_new ();

  forward_select_properties (gobject_class, msgconv_prop_map, num_msgconv_props,
      gst_nvmsgconv_get_type (), &prop_index, klass->orig_prop_name_id_map);
  klass->last_nvmsgconv_prop_id = prop_index - 1;

  forward_select_properties (gobject_class, msgbroker_prop_map,
      num_msgbroker_props, gst_nvmsgbroker_get_type (), &prop_index,
      klass->orig_prop_name_id_map);
  klass->last_nvmsgbroker_prop_id = prop_index - 1;

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvmsgbrokerbin_sink_template);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvmsgbroker_change_state);
  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvMsgBroker Bin", "NvMsgBroker Bin",
      "Nvidia DeepStreamSDK Message Broker Sink Bin. Internal Pipeline: queue->nvmsgconv->nvmsgbroker",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_ds_nvmsgbroker_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvMsgBrokerBin *nvmsgbrokerbin = GST_DS_NVMSGBROKER_BIN (object);
  GstDsNvMsgBrokerBinClass *klass =
      GST_DS_NVMSGBROKER_BIN_GET_CLASS (nvmsgbrokerbin);
  switch (prop_id) {
    case PROP_DISABLE_MSGCONV:
      nvmsgbrokerbin->disable_msgconv = g_value_get_boolean (value);
      break;
    default:
      if (prop_id <= klass->last_nvmsgconv_prop_id) {
        g_object_set_property (G_OBJECT (nvmsgbrokerbin->msgconv),
            (gchar *) g_ptr_array_index (klass->orig_prop_name_id_map, prop_id),
            value);
      } else if (prop_id <= klass->last_nvmsgbroker_prop_id) {
        g_object_set_property (G_OBJECT (nvmsgbrokerbin->msgbroker),
            (gchar *) g_ptr_array_index (klass->orig_prop_name_id_map, prop_id),
            value);
      } else {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;
  }
}


static void
gst_ds_nvmsgbroker_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvMsgBrokerBin *nvmsgbrokerbin = GST_DS_NVMSGBROKER_BIN (object);
  GstDsNvMsgBrokerBinClass *klass =
      GST_DS_NVMSGBROKER_BIN_GET_CLASS (nvmsgbrokerbin);

  switch (prop_id) {
    case PROP_DISABLE_MSGCONV:
      g_value_set_boolean (value, nvmsgbrokerbin->disable_msgconv);
      break;
    default:
      if (prop_id <= klass->last_nvmsgconv_prop_id) {
        g_object_get_property (G_OBJECT (nvmsgbrokerbin->msgconv),
            (gchar *) g_ptr_array_index (klass->orig_prop_name_id_map, prop_id),
            value);
      } else if (prop_id <= klass->last_nvmsgbroker_prop_id) {
        g_object_get_property (G_OBJECT (nvmsgbrokerbin->msgbroker),
            (gchar *) g_ptr_array_index (klass->orig_prop_name_id_map, prop_id),
            value);
      } else {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;
  }
}

static void
broker_queue_overrun (GstElement * sink_queue, gpointer user_data)
{
  GST_ELEMENT_WARNING (sink_queue, STREAM, FAILED,
      ("nvmsgbroker queue overrun; Older Message Buffer "
          "Dropped; Network bandwidth might be insufficient"), (NULL));
}

static gboolean
link_elements (GstDsNvMsgBrokerBin * mbin)
{
  if (mbin->disable_msgconv) {
    NVGSTDS_LINK_ELEMENT (mbin->queue, mbin->msgbroker, FALSE);
  } else {
    gst_bin_add (GST_BIN (mbin), mbin->msgconv);
    NVGSTDS_LINK_ELEMENT (mbin->queue, mbin->msgconv, FALSE);
    NVGSTDS_LINK_ELEMENT (mbin->msgconv, mbin->msgbroker, FALSE);
  }

  return TRUE;
}

static gboolean
populate_bin (GstDsNvMsgBrokerBin * mbin)
{
  gchar elem_name[128];
  g_snprintf (elem_name, sizeof (elem_name), "nvmsgbrokersinkbin-queue");
  mbin->queue = gst_element_factory_make ("queue", elem_name);
  if (!mbin->queue) {
    GST_ELEMENT_ERROR (mbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'queue' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (mbin), mbin->queue);
  g_object_set (G_OBJECT (mbin->queue), "leaky", 2, NULL);
  g_object_set (G_OBJECT (mbin->queue), "max-size-buffers", 20, NULL);
  g_signal_connect (G_OBJECT (mbin->queue), "overrun",
      G_CALLBACK (broker_queue_overrun), mbin);

  g_snprintf (elem_name, sizeof (elem_name), "nvmsgbrokersinkbin-nvmsgconv");
  mbin->msgconv = gst_element_factory_make ("nvmsgconv", elem_name);
  if (!mbin->msgconv) {
    GST_ELEMENT_ERROR (mbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'nvmsgconv' element"), (NULL));
    return FALSE;
  }
  g_object_ref_sink (mbin->msgconv);

  g_snprintf (elem_name, sizeof (elem_name), "nvmsgbrokersinkbin-nvmsgbroker");
  mbin->msgbroker = gst_element_factory_make ("nvmsgbroker", elem_name);
  if (!mbin->msgbroker) {
    GST_ELEMENT_ERROR (mbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'nvmsgbroker' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (mbin), mbin->msgbroker);
  g_object_set (G_OBJECT (mbin->msgbroker), "async", FALSE, NULL);

  NVGSTDS_BIN_SET_GHOST_PAD_TARGET (mbin, mbin->bin_sink_pad, mbin->queue,
      "sink", FALSE);

  return TRUE;
}

static GstStateChangeReturn
gst_ds_nvmsgbroker_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvMsgBrokerBin *nvmsgbrokerbin = GST_DS_NVMSGBROKER_BIN (element);
  GstStateChangeReturn ret;
  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    if (!link_elements (nvmsgbrokerbin)) {
      return GST_STATE_CHANGE_FAILURE;
    }
  }
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    if (nvmsgbrokerbin->disable_msgconv) {
      gst_element_unlink (nvmsgbrokerbin->queue, nvmsgbrokerbin->msgbroker);
    } else {
      gst_bin_remove (GST_BIN (nvmsgbrokerbin), nvmsgbrokerbin->msgconv);
      gst_element_unlink (nvmsgbrokerbin->queue, nvmsgbrokerbin->msgconv);
      gst_element_unlink (nvmsgbrokerbin->msgconv, nvmsgbrokerbin->msgbroker);
    }
  }

  return ret;
}

static void
gst_ds_nvmsgbroker_bin_init (GstDsNvMsgBrokerBin * nvmsgbrokerbin)
{
  nvmsgbrokerbin->bin_sink_pad =
      gst_ghost_pad_new_no_target_from_template ("sink",
      gst_static_pad_template_get (&gst_nvmsgbrokerbin_sink_template));
  gst_element_add_pad (GST_ELEMENT (nvmsgbrokerbin),
      nvmsgbrokerbin->bin_sink_pad);
  if (!populate_bin (nvmsgbrokerbin)) {
    return;
  }
}

static void
gst_ds_nvmsgbroker_bin_finalize (GObject * object)
{
  GstDsNvMsgBrokerBin *bin = (GstDsNvMsgBrokerBin *) object;
  gst_object_unref (bin->msgconv);
}
