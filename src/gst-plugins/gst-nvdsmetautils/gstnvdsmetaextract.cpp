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
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <dlfcn.h>
#include <sys/time.h>
#include <stdio.h>

#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <gst/audio/audio.h>

#include "gstnvdsmeta.h"
#include "gstnvdsmetaextract.h"
#include "nvdscustomusermeta.h"

using namespace std;

GST_DEBUG_CATEGORY_STATIC (gst_nvdsmetaextract_debug);
#define GST_CAT_DEFAULT gst_nvdsmetaextract_debug

#ifndef PACKAGE
#define PACKAGE "nvdsmetaextract"
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
  PROP_DESERIALIZATION_LIB_NAME,
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

#define gst_nvdsmetaextract_parent_class parent_class
G_DEFINE_TYPE (Gstnvdsmetaextract, gst_nvdsmetaextract, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdsmetaextract_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsmetaextract_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvdsmetaextract_set_caps (GstBaseTransform * trans,
        GstCaps * incaps, GstCaps * outcaps)
{
  Gstnvdsmetaextract *nvmetaextract = GST_NVDSMETAEXTRACT (trans);
  GstVideoInfo video_info = {{0}};

  GST_DEBUG_OBJECT (nvmetaextract, "set_caps");

  nvmetaextract->frame_width = GST_VIDEO_INFO_WIDTH (&video_info);
  nvmetaextract->frame_height = GST_VIDEO_INFO_HEIGHT (&video_info);

  return TRUE;
}

static void gst_nvdsmetaextract_finalize (GObject * object)
{
  Gstnvdsmetaextract *nvdsmetaextract = GST_NVDSMETAEXTRACT (object);
  GST_DEBUG_OBJECT (nvdsmetaextract, "nvdsmetaextract = %p\n", nvdsmetaextract);
}

static GstFlowReturn gst_nvdsmetaextract_transform_ip (GstBaseTransform * btrans,
    GstBuffer * buf)
{
  Gstnvdsmetaextract *nvdsmetaextract = GST_NVDSMETAEXTRACT (btrans);

  /* Set input timestamp for latency measurement */
  nvds_set_input_system_timestamp (buf, GST_ELEMENT_NAME(nvdsmetaextract));

  nvdsmetaextract->deserialize_func(buf);

  /* Set output timestamp for latency measurement */
  nvds_set_output_system_timestamp (buf, GST_ELEMENT_NAME(nvdsmetaextract));

  return GST_FLOW_OK;
}

static gboolean gst_nvdsmetaextract_start (GstBaseTransform * btrans)
{
  Gstnvdsmetaextract *nvdsmetaextract = GST_NVDSMETAEXTRACT (btrans);
  char *error;
  nvdsmetaextract->lib_handle =
      dlopen (nvdsmetaextract->deserialization_lib_name, RTLD_NOW);
  if (nvdsmetaextract->lib_handle == NULL)
  {
      GST_DEBUG_OBJECT(nvdsmetaextract, "Could not open deserialiaztion library %s",
              nvdsmetaextract->deserialization_lib_name);
      return FALSE;
  }
  nvdsmetaextract->deserialize_func =
      (void (*)(GstBuffer*))dlsym (nvdsmetaextract->lib_handle, "deserialize_data");
  if ((error = dlerror()) != NULL)
  {
      GST_DEBUG_OBJECT(nvdsmetaextract, "%s", error);
      return FALSE;
  }
  return TRUE;
}

static gboolean gst_nvdsmetaextract_stop (GstBaseTransform * btrans)
{
  Gstnvdsmetaextract *nvdsmetaextract = GST_NVDSMETAEXTRACT (btrans);
  GST_INFO_OBJECT (nvdsmetaextract, " %s\n", __func__);
  return TRUE;
}

/* initialize the nvdsmetaextract's class */
  static void
gst_nvdsmetaextract_class_init (GstnvdsmetaextractClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class =
      (GstBaseTransformClass *) klass;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_nvdsmetaextract_set_property;
  gobject_class->get_property = gst_nvdsmetaextract_get_property;
  gobject_class->finalize = gst_nvdsmetaextract_finalize;

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvdsmetaextract_transform_ip);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdsmetaextract_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsmetaextract_stop);
  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsmetaextract_set_caps);

  gst_element_class_set_details_simple(gstelement_class,
      "nvdsmetaextract",
      "nvdsmetaextract",
      "Gstreamer NV DS META DATA EXTRACTION PLUGIN",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  g_object_class_install_property (gobject_class, PROP_DESERIALIZATION_LIB_NAME,
          g_param_spec_string ("deserialize-lib", "De-serialization library name",
            "Set de-serialization library Name to be used",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

static void gst_nvdsmetaextract_init (Gstnvdsmetaextract * filter)
{
  filter->sinkcaps = gst_static_pad_template_get_caps (&sink_factory);
  filter->srccaps = gst_static_pad_template_get_caps (&src_factory);
}

static void gst_nvdsmetaextract_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstnvdsmetaextract *nvdsmetaextract = GST_NVDSMETAEXTRACT (object);

  switch (prop_id) {
      case PROP_DESERIALIZATION_LIB_NAME:
          if (nvdsmetaextract->deserialization_lib_name) {
              g_free(nvdsmetaextract->deserialization_lib_name);
          }
          nvdsmetaextract->deserialization_lib_name =
              (gchar *)g_value_dup_string (value);
          break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
          break;
  }
}

static void gst_nvdsmetaextract_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstnvdsmetaextract *nvdsmetaextract = GST_NVDSMETAEXTRACT (object);

  switch (prop_id) {
    case PROP_DESERIALIZATION_LIB_NAME:
      g_value_set_string (value, nvdsmetaextract->deserialization_lib_name);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

gboolean nvds_metaextract_init (GstPlugin * nvdsmetaextract)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsmetaextract_debug, "nvdsmetaextract",
      0, "nvdsmetaextract");

  return gst_element_register (nvdsmetaextract, "nvdsmetaextract", GST_RANK_NONE,
      GST_TYPE_NVDSMETAEXTRACT);
}
