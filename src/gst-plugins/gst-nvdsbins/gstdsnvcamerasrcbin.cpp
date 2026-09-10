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

#include "gstdsnvcamerasrcbin.h"
#include "gst-nvcommon.h"
#include "gst-nvquery-internal.h"
#include "nvbufsurface.h"
#include "gstnvdsbinutils.h"

static GstStaticPadTemplate gst_nvcamerasrc_bin_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "I420,  NV12, P010_10LE, BGRx, RGBA, GRAY8 }") ";"
        GST_VIDEO_CAPS_MAKE ("{ "
            "I420, P010_10LE, NV12, BGRx, RGBA, GRAY8 }")));

GST_DEBUG_CATEGORY (gst_ds_nvcamerasrc_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvcamerasrc_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvcamerasrc_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvcamerasrc_bin_debug, "nvcamerasrcbin", 0, "nvcamerasrcbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvCameraSrcBin, gst_ds_nvcamerasrc_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvcamerasrc_bin_finalize (GObject * object);
static void gst_ds_nvcamerasrc_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvcamerasrc_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);
static GstStateChangeReturn gst_ds_nvcamerasrc_bin_change_state (GstElement *
    element, GstStateChange transition);
static GstPadProbeReturn src_pad_query_probe (GstPad * pad,
    GstPadProbeInfo * info, gpointer data);

enum
{
  PROP_0,
  PROP_TYPE,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_FRAMERATE,
  PROP_GPU_DEVICE_ID,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_SENSOR_ID,
  PROP_V4L2_DEVICE,
  PROP_SOURCE_ID,
};


#define DEFAULT_TYPE V4L2
#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480
#define DEFAULT_FPS_N 30
#define DEFAULT_FPS_D 1
#define DEFAULT_GPU_ID 0
#define DEFAULT_MEM_TYPE NVBUF_MEM_DEFAULT
#define DEFAULT_SENSOR_ID 0
#define DEFAULT_V4L2_DEVICE "/dev/video0"
#define DEFAULT_SOURCE_ID -1

#define GST_TYPE_NVDSCAMERA_TYPE (gst_nvdscamerasrc_type ())

static GType
gst_nvdscamerasrc_type (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue type[] = {
    {V4L2, "V4L2 Interface", "v4l2"},
    {NVARGUS, "NvArgus Interface", "nvargus"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("NvDsCameraSrcBinType",
        type);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static void
gst_ds_nvcamerasrc_bin_class_init (GstDsNvCameraSrcBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvcamerasrc_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvcamerasrc_bin_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_ds_nvcamerasrc_bin_finalize);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvcamerasrc_bin_change_state);

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvcamerasrc_bin_src_template);

  g_object_class_install_property (gobject_class, PROP_TYPE,
      g_param_spec_enum ("type", "Type",
          "Type of interface to use",
          GST_TYPE_NVDSCAMERA_TYPE, DEFAULT_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width", "Width",
          "Frame width to request from the source", 0, G_MAXUINT,
          DEFAULT_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height", "Height",
          "Frame height to request from the source", 0, G_MAXUINT,
          DEFAULT_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_FRAMERATE,
      gst_param_spec_fraction ("framerate",
          "Framerate",
          "Frame rate to request from the source",
          0, 1, G_MAXINT, 1, DEFAULT_FPS_N, DEFAULT_FPS_D,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_SENSOR_ID,
      g_param_spec_uint ("sensor-id",
          "Sensor ID",
          "Set the id of camera sensor to use.",
          0, G_MAXUINT, DEFAULT_SENSOR_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_V4L2_DEVICE,
      g_param_spec_string ("v4l2-device",
          "V4L2 Device",
          "V4L2 capture device", DEFAULT_V4L2_DEVICE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_SOURCE_ID,
      g_param_spec_int ("source-id", "Source ID",
          "Unique ID for the input source",
          -1, G_MAXINT, DEFAULT_SOURCE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));


  gst_element_class_set_details_simple (gstelement_class,
      "NvCameraSrc Bin", "NvCameraSrc Bin",
      "Nvidia DeepStreamSDK NvCameraSrc Bin",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvcamerasrc_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvCameraSrcBin *nvcamerasrcbin = GST_DS_NVCAMERASRC_BIN (object);

  switch (prop_id) {
    case PROP_TYPE:
      nvcamerasrcbin->type = (GstDsNvCameraSrcType) g_value_get_enum (value);
      break;
    case PROP_WIDTH:
      nvcamerasrcbin->width = g_value_get_uint (value);
      break;
    case PROP_HEIGHT:
      nvcamerasrcbin->height = g_value_get_uint (value);
      break;
    case PROP_FRAMERATE:
      nvcamerasrcbin->fps_n = gst_value_get_fraction_numerator (value);
      nvcamerasrcbin->fps_d = gst_value_get_fraction_denominator (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvcamerasrcbin->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvcamerasrcbin->memtype = g_value_get_enum (value);
      break;
    case PROP_SENSOR_ID:
      nvcamerasrcbin->sensor_id = g_value_get_uint (value);
      break;
    case PROP_V4L2_DEVICE:
      g_free (nvcamerasrcbin->v4l2_device);
      nvcamerasrcbin->v4l2_device = g_value_dup_string (value);
      break;
    case PROP_SOURCE_ID:
      nvcamerasrcbin->source_id = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_ds_nvcamerasrc_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvCameraSrcBin *nvcamerasrcbin = GST_DS_NVCAMERASRC_BIN (object);

  switch (prop_id) {
    case PROP_TYPE:
      g_value_set_enum (value, nvcamerasrcbin->type);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, nvcamerasrcbin->width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, nvcamerasrcbin->height);
      break;
    case PROP_FRAMERATE:
      gst_value_set_fraction (value, nvcamerasrcbin->fps_n,
          nvcamerasrcbin->fps_d);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvcamerasrcbin->gpu_id);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvcamerasrcbin->memtype);
      break;
    case PROP_SENSOR_ID:
      g_value_set_uint (value, nvcamerasrcbin->sensor_id);
      break;
    case PROP_V4L2_DEVICE:
      g_value_set_string (value, nvcamerasrcbin->v4l2_device);
      break;
    case PROP_SOURCE_ID:
      g_value_set_int (value, nvcamerasrcbin->source_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static void
gst_ds_nvcamerasrc_bin_init (GstDsNvCameraSrcBin * nvcamerasrcbin)
{
  nvcamerasrcbin->type = DEFAULT_TYPE;
  nvcamerasrcbin->width = DEFAULT_WIDTH;
  nvcamerasrcbin->height = DEFAULT_HEIGHT;
  nvcamerasrcbin->fps_n = DEFAULT_FPS_N;
  nvcamerasrcbin->fps_d = DEFAULT_FPS_D;

  nvcamerasrcbin->gpu_id = DEFAULT_GPU_ID;
  nvcamerasrcbin->memtype = DEFAULT_MEM_TYPE;

  nvcamerasrcbin->sensor_id = DEFAULT_SENSOR_ID;
  nvcamerasrcbin->v4l2_device = g_strdup (DEFAULT_V4L2_DEVICE);
  nvcamerasrcbin->source_id = DEFAULT_SOURCE_ID;

  nvcamerasrcbin->bin_src_pad =
      gst_ghost_pad_new_no_target_from_template ("src",
      gst_static_pad_template_get (&gst_nvcamerasrc_bin_src_template));
  gst_element_add_pad (GST_ELEMENT (nvcamerasrcbin),
      nvcamerasrcbin->bin_src_pad);
  gst_pad_add_probe (nvcamerasrcbin->bin_src_pad, GST_PAD_PROBE_TYPE_QUERY_BOTH,
      src_pad_query_probe, nvcamerasrcbin, NULL);

  GST_OBJECT_FLAG_SET (nvcamerasrcbin, GST_ELEMENT_FLAG_SOURCE);
}

static void
gst_ds_nvcamerasrc_bin_finalize (GObject * object)
{
  GstDsNvCameraSrcBin *bin = (GstDsNvCameraSrcBin *) object;
  g_free (bin->v4l2_device);
}

static GstPadProbeReturn
src_pad_query_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  GstDsNvCameraSrcBin *bin = (GstDsNvCameraSrcBin *) data;
  if (info->type & GST_PAD_PROBE_TYPE_QUERY_BOTH) {
    GstQuery *query = GST_QUERY (info->data);
    if (gst_nvquery_is_sourceid (query) && bin->source_id >= 0) {
      gst_nvquery_sourceid_set (query, bin->source_id);
      return GST_PAD_PROBE_HANDLED;
    }
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
populate_camera_bin (GstDsNvCameraSrcBin * csbin)
{
  GstElement *cap_filter1 = NULL, *cap_filter = NULL;

  switch (csbin->type) {
    case NVARGUS:
#ifdef __aarch64__
      csbin->src_elem =
          gst_element_factory_make ("nvarguscamerasrc", "src_elem");
      g_object_set (G_OBJECT (csbin->src_elem), "bufapi-version", TRUE, NULL);
      if (!csbin->src_elem) {
        GST_ELEMENT_ERROR (csbin, RESOURCE, NOT_FOUND,
            ("Failed to create 'nvarguscamerasrc' element"), (NULL));
        return FALSE;
      }
#else
      GST_ELEMENT_ERROR (csbin, LIBRARY, SETTINGS,
          ("ARGUS interface is not supported for dGPU"), (NULL));
      return FALSE;
#endif
      break;
    case V4L2:
      csbin->src_elem = gst_element_factory_make ("v4l2src", "src_elem");
      if (!csbin->src_elem) {
        GST_ELEMENT_ERROR (csbin, RESOURCE, NOT_FOUND,
            ("Failed to create 'v4l2src' element"), (NULL));
        return FALSE;
      }

      cap_filter1 = gst_element_factory_make ("capsfilter", "src_cap_filter1");
      if (!cap_filter1) {
        GST_ELEMENT_ERROR (csbin, RESOURCE, NOT_FOUND,
            ("Failed to create 'capsfilter' element"), (NULL));
        return FALSE;
      }

      {
        GstCaps *caps = gst_caps_new_simple ("video/x-raw",
            "width", G_TYPE_INT, csbin->width, "height", G_TYPE_INT,
            csbin->height, "framerate", GST_TYPE_FRACTION,
            csbin->fps_n, csbin->fps_d, NULL);
        g_object_set (G_OBJECT (cap_filter1), "caps", caps, NULL);
        gst_caps_unref (caps);
      }
      break;
    default:
      return FALSE;
  }

  cap_filter = gst_element_factory_make ("capsfilter", "src_cap_filter");
  if (!cap_filter) {
    GST_ELEMENT_ERROR (csbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'capsfilter' element"), (NULL));
    return FALSE;
  }

  {
    GstCaps *caps =
        gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
        "width", G_TYPE_INT, csbin->width, "height", G_TYPE_INT,
        csbin->height, "framerate", GST_TYPE_FRACTION,
        csbin->fps_n, csbin->fps_d, NULL);

    GstCapsFeatures *feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);
    g_object_set (G_OBJECT (cap_filter), "caps", caps, NULL);
    gst_caps_unref (caps);
  }

  if (csbin->type == V4L2) {
    GstElement *nvvidconv2;

#ifdef __x86_64__
    GstElement *nvvidconv1 =
        gst_element_factory_make ("videoconvert", "nvvidconv1");
    if (!nvvidconv1) {
      GST_ELEMENT_ERROR (csbin, RESOURCE, NOT_FOUND,
          ("Failed to create 'nvvideoconvert' element"), (NULL));
      return FALSE;
    }
#endif

    nvvidconv2 = gst_element_factory_make ("nvvideoconvert", "nvvidconv2");
    if (!nvvidconv2) {
      GST_ELEMENT_ERROR (csbin, RESOURCE, NOT_FOUND,
          ("Failed to create 'nvvideoconvert' element"), (NULL));
      return FALSE;
    }

    g_object_set (G_OBJECT (nvvidconv2), "gpu-id", csbin->gpu_id,
        "nvbuf-memory-type", csbin->memtype, NULL);

    gst_bin_add_many (GST_BIN (csbin), csbin->src_elem, cap_filter1,
        nvvidconv2, cap_filter, NULL);
#ifdef __x86_64__
    gst_bin_add (GST_BIN (csbin), nvvidconv1);
#else

#endif

    NVGSTDS_LINK_ELEMENT (csbin->src_elem, cap_filter1, FALSE);
#ifdef __x86_64__
    NVGSTDS_LINK_ELEMENT (cap_filter1, nvvidconv1, FALSE);
    NVGSTDS_LINK_ELEMENT (nvvidconv1, nvvidconv2, FALSE);
#else
    NVGSTDS_LINK_ELEMENT (cap_filter1, nvvidconv2, FALSE);
#endif
    NVGSTDS_LINK_ELEMENT (nvvidconv2, cap_filter, FALSE);

    g_object_set (G_OBJECT (csbin->src_elem), "device",
        csbin->v4l2_device, NULL);
  } else {
    gst_bin_add_many (GST_BIN (csbin), csbin->src_elem, cap_filter, NULL);

    NVGSTDS_LINK_ELEMENT (csbin->src_elem, cap_filter, FALSE);

    g_object_set (G_OBJECT (csbin->src_elem), "sensor-id",
        csbin->sensor_id, NULL);
  }

  NVGSTDS_BIN_SET_GHOST_PAD_TARGET (csbin, csbin->bin_src_pad, cap_filter,
      "src", FALSE);

  return TRUE;
}

static GstStateChangeReturn
gst_ds_nvcamerasrc_bin_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvCameraSrcBin *nvcamerasrcbin = GST_DS_NVCAMERASRC_BIN (element);
  GstStateChangeReturn ret;
  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    if (!populate_camera_bin (nvcamerasrcbin)) {
      return GST_STATE_CHANGE_FAILURE;
    }
  }
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    remove_all_children (GST_BIN (nvcamerasrcbin));
  }
  return ret;
}
