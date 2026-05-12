/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <iostream>
#include <dlfcn.h>
#include <sys/time.h>
#include <stdio.h>


#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <gst/audio/audio.h>

#include "gstnvdsmeta.h"
#include "gstnvdsmetainsert.h"
#include "nvdscustomusermeta.h"

using namespace std;

GST_DEBUG_CATEGORY_STATIC (gst_nvdsmetainsert_debug);
#define GST_CAT_DEFAULT gst_nvdsmetainsert_debug

#ifndef PACKAGE
#define PACKAGE "nvdsmetainsert"
#endif

#define COMMON_AUDIO_CAPS \
  "channels = (int) [ 1, MAX ], " \
  "rate = (int) [ 1, MAX ]"

/* Filter signals and args */
enum
{
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SERIALIZATION_LIB_NAME,
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
/* Input capabilities. */
static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM",
            GST_VIDEO_FORMATS_ALL) "; "
            "audio/x-raw(memory:NVMM), "
            "format = (string) " GST_AUDIO_FORMATS_ALL
            ", layout = (string) {interleaved, non-interleaved}, "
            COMMON_AUDIO_CAPS));

/* Output capabilities. */
static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM",
            GST_VIDEO_FORMATS_ALL) "; "
            "audio/x-raw(memory:NVMM), "
            "format = (string) " GST_AUDIO_FORMATS_ALL
            ", layout = (string) {interleaved, non-interleaved}, "
            COMMON_AUDIO_CAPS));

#define gst_nvdsmetainsert_parent_class parent_class
G_DEFINE_TYPE (Gstnvdsmetainsert, gst_nvdsmetainsert, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdsmetainsert_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsmetainsert_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvdsmetainsert_set_caps (GstBaseTransform * trans,
        GstCaps * incaps, GstCaps * outcaps)
{
  Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (trans);

  GST_DEBUG_OBJECT (nvdsmetainsert, "set_caps");

  return TRUE;
}

static void gst_nvdsmetainsert_finalize (GObject * object)
{
  Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (object);
  GST_DEBUG_OBJECT (nvdsmetainsert, "nvdsmetainsert = %p\n", nvdsmetainsert);
}

static GstFlowReturn gst_nvdsmetainsert_transform_ip (GstBaseTransform * btrans,
    GstBuffer * buf)
{
  Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (btrans);
  nvdsmetainsert->meta_mem_size = 0;

  /* Set input timestamp for latency measurement */
  nvds_set_input_system_timestamp (buf, GST_ELEMENT_NAME(nvdsmetainsert));

  nvdsmetainsert->serialize_func(buf);

  /* Set output timestamp for latency measurement */
  nvds_set_output_system_timestamp (buf, GST_ELEMENT_NAME(nvdsmetainsert));

  return GST_FLOW_OK;
}

static gboolean gst_nvdsmetainsert_start (GstBaseTransform * btrans)
{
    Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (btrans);
    GST_INFO_OBJECT (nvdsmetainsert, " %s\n", __func__);
    char *error;

    nvdsmetainsert->lib_handle = dlopen (nvdsmetainsert->serialization_lib_name, RTLD_NOW);
    if (nvdsmetainsert->lib_handle == NULL)
    {
        GST_DEBUG_OBJECT(nvdsmetainsert, "Could not open serialiaztion library");
        return FALSE;
    }
    nvdsmetainsert->serialize_func = (void (*)(GstBuffer*))dlsym (nvdsmetainsert->lib_handle, "serialize_data");
    if ((error = dlerror()) != NULL)
    {
        GST_DEBUG_OBJECT(nvdsmetainsert, "%s", error);
        return FALSE;
    }
    return TRUE;
}

static gboolean gst_nvdsmetainsert_stop (GstBaseTransform * btrans)
{
  Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (btrans);
  GST_INFO_OBJECT (nvdsmetainsert, " %s\n", __func__);
  return TRUE;
}

/* initialize the nvdsmetainsert's class */
  static void
gst_nvdsmetainsert_class_init (GstnvdsmetainsertClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class =
      (GstBaseTransformClass *) klass;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_nvdsmetainsert_set_property;
  gobject_class->get_property = gst_nvdsmetainsert_get_property;
  gobject_class->finalize = gst_nvdsmetainsert_finalize;

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvdsmetainsert_transform_ip);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdsmetainsert_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsmetainsert_stop);
  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsmetainsert_set_caps);

  gst_element_class_set_details_simple(gstelement_class,
      "nvdsmetainsert",
      "nvdsmetainsert",
      "Gstreamer NV DS META DATA INSERTION PLUGIN",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  g_object_class_install_property (gobject_class, PROP_SERIALIZATION_LIB_NAME,
          g_param_spec_string ("serialize-lib", "Serialization library name",
            "Set serialization library Name to be used",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

static void gst_nvdsmetainsert_init (Gstnvdsmetainsert * nvdsmetainsert)
{
  nvdsmetainsert->sinkcaps = gst_static_pad_template_get_caps (&sink_factory);
  nvdsmetainsert->srccaps = gst_static_pad_template_get_caps (&src_factory);
  nvdsmetainsert->meta_mem_size = 0;
}

static void gst_nvdsmetainsert_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (object);

  switch (prop_id) {
    case PROP_SERIALIZATION_LIB_NAME:
        if (nvdsmetainsert->serialization_lib_name) {
            g_free(nvdsmetainsert->serialization_lib_name);
        }
        nvdsmetainsert->serialization_lib_name = (gchar *)g_value_dup_string (value);
        break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void gst_nvdsmetainsert_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstnvdsmetainsert *nvdsmetainsert = GST_NVDSMETAINSERT (object);

  switch (prop_id) {
    case PROP_SERIALIZATION_LIB_NAME:
      g_value_set_string (value, nvdsmetainsert->serialization_lib_name);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

gboolean nvds_metainsert_init (GstPlugin * nvdsmetainsert)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsmetainsert_debug, "nvdsmetainsert",
      0, "nvdsmetainsert");

  return gst_element_register (nvdsmetainsert, "nvdsmetainsert", GST_RANK_NONE,
      GST_TYPE_NVDSMETAINSERT);
}
