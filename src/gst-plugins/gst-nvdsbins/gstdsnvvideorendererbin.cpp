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

#include <gst/video/video.h>
#include <gst/video/videooverlay.h>
#include "gstdsnvvideorendererbin.h"
#include "gst-nvcommon.h"
#include "nvbufsurface.h"
#include "gstnvdsbinutils.h"
#include "cuda_runtime_api.h"

extern "C" GType gst_nvvideoconvert_get_type ();

GST_DEBUG_CATEGORY (gst_ds_nvvideorenderer_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvvideorenderer_bin_debug

static void gst_nvvideorenderer_bin_videooverlay_init (GstVideoOverlayInterface
    * iface);

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvvideorenderer_bin_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstDsNvVideoRendererBin, gst_ds_nvvideorenderer_bin,
    GST_TYPE_BIN, G_IMPLEMENT_INTERFACE (GST_TYPE_VIDEO_OVERLAY,
        gst_nvvideorenderer_bin_videooverlay_init));

static void gst_ds_nvvideorenderer_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvvideorenderer_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);

static GstStateChangeReturn
gst_ds_nvvideorenderer_change_state (GstElement * element,
    GstStateChange transition);

#define GST_TYPE_NVDS_VIDEO_RENDERER (gst_nvds_video_sink_get_type())
static GType
gst_nvds_video_sink_get_type (void)
{
  static GType video_sink_type = 0;
  static const GEnumValue video_sinks[] = {
    {VIDEO_RENDERER_FAKE, "Fakesink", "fake"},
    {VIDEO_RENDERER_EGL, "NvEglGlesSink - EGL based renderer", "egl"},
    {VIDEO_RENDERER_DRM,
          "NvDrmVideoSink - Nv Drm Video Sink (Jetson only)",
        "drmvideo"},
    {VIDEO_RENDERER_3D,
          "Nv3dSink - Nv 3D Sink (Jetson only)",
        "3d"},
    {0, NULL, NULL},
  };

  if (!video_sink_type) {
    video_sink_type = g_enum_register_static ("GstNvDsVideoRendererType",
        video_sinks);
  }
  return video_sink_type;
}

enum
{
  PROP_FIRST,
  PROP_OFFSET_X,
  PROP_OFFSET_Y,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_CONN_ID,
  PROP_PLANE_ID,
  PROP_SET_MODE,
  PROP_COLOR_RANGE,
  PROP_SYNC,
  PROP_QOS,
  PROP_GPU_DEVICE_ID,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_CREATE_FROM_CONFIG,
  PROP_VIDEO_RENDERER,
  PROP_LAST
};

#define DEFAULT_OFFSET_X 0
#define DEFAULT_OFFSET_Y 0
#define DEFAULT_WIDTH 0
#define DEFAULT_HEIGHT 0
#define DEFAULT_CONN_ID INT_MAX
#define DEFAULT_PLANE_ID INT_MAX
#define DEFAULT_SET_MODE FALSE
#define DEFAULT_COLOR_RANGE 2
#define DEFAULT_SYNC TRUE
#define DEFAULT_QOS FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_NVBUF_MEMORY_TYPE NVBUF_MEM_DEFAULT
#define DEFAULT_NVDS_VIDEO_RENDERER VIDEO_RENDERER_EGL
#define DEFAULT_CREATE_FROM_CONFIG ""

static void
gst_ds_nvvideorenderer_bin_class_init (GstDsNvVideoRendererBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvvideorenderer_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvvideorenderer_bin_get_property);

  g_object_class_install_property (gobject_class, PROP_OFFSET_X,
      g_param_spec_uint ("offset-x", "Offset X",
          "Renderer horizontal offset", 0, G_MAXUINT, DEFAULT_OFFSET_X,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OFFSET_Y,
      g_param_spec_uint ("offset-y", "Offset Y",
          "Renderer vertical offset", 0, G_MAXUINT, DEFAULT_OFFSET_Y,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width", "Width",
          "Renderer width", 0, G_MAXUINT, DEFAULT_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height", "Height",
          "Renderer height", 0, G_MAXUINT, DEFAULT_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONN_ID,
      g_param_spec_uint ("conn-id", "Connection ID",
          "Connection Index (Valid for nvdrmvideosink)", 0, G_MAXUINT,
          DEFAULT_CONN_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_PLANE_ID,
      g_param_spec_uint ("plane-id", "Plane ID",
          "Plane on which video should be rendered (Valid for nvdrmvideosink)",
          0, G_MAXUINT, DEFAULT_PLANE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SET_MODE,
      g_param_spec_uint ("set-mode", "Set mode",
          "Select default or mode of Video Stream (Valid for nvdrmvideosink)",
          0, G_MAXUINT, DEFAULT_PLANE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_COLOR_RANGE,
      g_param_spec_uint ("color-range", "Color Range",
          "Sets color range only when set-mode=1 (Valid for nvdrmvideosink)",
          0, G_MAXUINT, DEFAULT_COLOR_RANGE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SYNC,
      g_param_spec_boolean ("sync", "Sync",
          "Sync on the clock", DEFAULT_SYNC,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_QOS,
      g_param_spec_boolean ("qos", "QoS",
          "Generate Quality - of - Service events upstream", DEFAULT_QOS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_VIDEO_RENDERER,
      g_param_spec_enum ("video-sink", "Video Sink",
          "Type of Video sink to use",
          GST_TYPE_NVDS_VIDEO_RENDERER,
          DEFAULT_NVDS_VIDEO_RENDERER,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  forward_pad_template (gstelement_class, gst_nvvideoconvert_get_type (),
      "sink");

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvvideorenderer_change_state);

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvVideoRenderer Bin",
      "NvVideoRenderer Bin",
      "Nvidia DeepStreamSDK Video Sinks Bin. Internal Pipeline: queue->nvvideoconvert->renderer",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_ds_nvvideorenderer_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvVideoRendererBin *nvvideorendererbin =
      GST_DS_NVVIDEORENDERER_BIN (object);

  switch (prop_id) {
    case PROP_OFFSET_X:
      nvvideorendererbin->offset_x = g_value_get_uint (value);
      break;
    case PROP_OFFSET_Y:
      nvvideorendererbin->offset_y = g_value_get_uint (value);
      break;
    case PROP_WIDTH:
      nvvideorendererbin->width = g_value_get_uint (value);
      break;
    case PROP_HEIGHT:
      nvvideorendererbin->height = g_value_get_uint (value);
      break;
    case PROP_COLOR_RANGE:
      nvvideorendererbin->color_range = g_value_get_uint (value);
      break;
    case PROP_CONN_ID:
      nvvideorendererbin->conn_id = g_value_get_uint (value);
      break;
    case PROP_PLANE_ID:
      nvvideorendererbin->plane_id = g_value_get_uint (value);
      break;
    case PROP_SET_MODE:
      nvvideorendererbin->set_mode = g_value_get_boolean (value);
      break;
    case PROP_SYNC:
      nvvideorendererbin->sync = g_value_get_boolean (value);
      break;
    case PROP_QOS:
      nvvideorendererbin->qos = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvvideorendererbin->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvvideorendererbin->mem_type =
          (NvBufSurfaceMemType) g_value_get_enum (value);
      break;
    case PROP_CREATE_FROM_CONFIG:
      g_free (nvvideorendererbin->create_from_config);
      nvvideorendererbin->create_from_config = g_value_dup_string (value);
      break;
    case PROP_VIDEO_RENDERER:
      nvvideorendererbin->vrenderer_type =
          (VideoRendererType) g_value_get_enum (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static void
gst_ds_nvvideorenderer_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvVideoRendererBin *nvvideorendererbin =
      GST_DS_NVVIDEORENDERER_BIN (object);

  switch (prop_id) {
    case PROP_OFFSET_X:
      g_value_set_uint (value, nvvideorendererbin->offset_x);
      break;
    case PROP_OFFSET_Y:
      g_value_set_uint (value, nvvideorendererbin->offset_y);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, nvvideorendererbin->width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, nvvideorendererbin->height);
      break;
    case PROP_COLOR_RANGE:
      g_value_set_uint (value, nvvideorendererbin->color_range);
      break;
    case PROP_CONN_ID:
      g_value_set_uint (value, nvvideorendererbin->conn_id);
      break;
    case PROP_PLANE_ID:
      g_value_set_uint (value, nvvideorendererbin->plane_id);
      break;
    case PROP_SET_MODE:
      g_value_set_boolean (value, nvvideorendererbin->set_mode);
      break;
    case PROP_SYNC:
      g_value_set_boolean (value, nvvideorendererbin->sync);
      break;
    case PROP_QOS:
      g_value_set_boolean (value, nvvideorendererbin->qos);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvvideorendererbin->gpu_id);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvvideorendererbin->mem_type);
      break;
    case PROP_VIDEO_RENDERER:
      g_value_set_enum (value, nvvideorendererbin->vrenderer_type);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
populate_bin (GstDsNvVideoRendererBin * nvvideorendererbin)
{
  gchar elem_name[128];
  g_snprintf (elem_name, sizeof (elem_name), "%s-sink",
      GST_ELEMENT_NAME (nvvideorendererbin));
  GstElement *nvvidconv = NULL;

  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      is_nvgpu = 1;
    }
  }

  switch (nvvideorendererbin->vrenderer_type) {
    case VIDEO_RENDERER_EGL:
      nvvideorendererbin->sink =
          gst_element_factory_make ("nveglglessink", elem_name);
      g_object_set (G_OBJECT (nvvideorendererbin->sink),
          "window-x", nvvideorendererbin->offset_x,
          "window-y", nvvideorendererbin->offset_y,
          "window-width", nvvideorendererbin->width,
          "window-height", nvvideorendererbin->height,
          "enable-last-sample", FALSE, NULL);

      if (nvvideorendererbin->render_rect_set)
        gst_video_overlay_set_render_rectangle (GST_VIDEO_OVERLAY
            (nvvideorendererbin->sink), nvvideorendererbin->render_rect_x,
            nvvideorendererbin->render_rect_y,
            nvvideorendererbin->render_rect_w,
            nvvideorendererbin->render_rect_h);

      if (nvvideorendererbin->window_id_set)
        gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY
            (nvvideorendererbin->sink), nvvideorendererbin->window_id);
      break;
    case VIDEO_RENDERER_DRM:
#ifndef __aarch64__
      GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
          ("'Nvdrvvideosink' is supported only on Jetson"), (NULL));
      return FALSE;
#endif

      nvvideorendererbin->sink =
          gst_element_factory_make ("nvdrmvideosink", elem_name);
      g_object_set (G_OBJECT (nvvideorendererbin->sink),
          "color-range", nvvideorendererbin->color_range,
          "conn-id", nvvideorendererbin->conn_id,
          "plane-id", nvvideorendererbin->plane_id,
          "set-mode", nvvideorendererbin->set_mode, NULL);
      break;
    case VIDEO_RENDERER_3D:
#ifndef __aarch64__
      GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
          ("'Nv3dsink' is supported only on Jetson"), (NULL));
      return FALSE;
#endif

      nvvideorendererbin->sink =
          gst_element_factory_make ("nv3dsink", elem_name);
      break;
    case VIDEO_RENDERER_FAKE:
      nvvideorendererbin->sink =
          gst_element_factory_make ("fakesink", elem_name);
      g_object_set (G_OBJECT (nvvideorendererbin->sink), "enable-last-sample",
          FALSE, NULL);
      break;
    default:
      GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
          ("A type of renderer must be selected"), (NULL));
      return FALSE;
  }
  if (!nvvideorendererbin->sink) {
    GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
        ("Failed to create sink element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (nvvideorendererbin), nvvideorendererbin->sink);
  g_object_set (G_OBJECT (nvvideorendererbin->sink), "sync",
      nvvideorendererbin->sync, "max-lateness", -1, "async", FALSE, "qos",
      nvvideorendererbin->qos, NULL);
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (nvvideorendererbin->sink),
          "gpu-id")) {
    g_object_set (G_OBJECT (nvvideorendererbin->sink), "gpu-id",
        nvvideorendererbin->gpu_id, NULL);
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-capfiler",
      GST_ELEMENT_NAME (nvvideorendererbin));

  if (!is_nvgpu
      && nvvideorendererbin->vrenderer_type == VIDEO_RENDERER_EGL) {
    nvvideorendererbin->cap_filter =
        gst_element_factory_make ("capsfilter", elem_name);
    if (!nvvideorendererbin->cap_filter) {
      GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
          ("Failed to create 'capsfilter' element"), (NULL));
      return FALSE;
    }
    gst_bin_add (GST_BIN (nvvideorendererbin), nvvideorendererbin->cap_filter);

    GstCaps *caps = gst_caps_from_string ("video/x-raw(memory:NVMM)");
    g_object_set (G_OBJECT (nvvideorendererbin->cap_filter), "caps", caps,
        NULL);
    gst_caps_unref (caps);
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-transform",
      GST_ELEMENT_NAME (nvvideorendererbin));
  if (nvvideorendererbin->vrenderer_type == VIDEO_RENDERER_EGL) {
    nvvideorendererbin->transform =
        gst_element_factory_make (is_nvgpu ? "nvegltransform" :
        "nvvideoconvert", elem_name);
    if (!nvvideorendererbin->transform) {
      GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
          ("Failed to create '%s' element",
              is_nvgpu ? "nvegltransform" : "nvvideoconvert"), (NULL));
      return FALSE;
    }
    gst_bin_add (GST_BIN (nvvideorendererbin), nvvideorendererbin->transform);
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (nvvideorendererbin->
                transform), "gpu-id")) {
      g_object_set (G_OBJECT (nvvideorendererbin->transform), "gpu-id",
          nvvideorendererbin->gpu_id, NULL);
    }
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (nvvideorendererbin->
                transform), "nvbuf-memory-type")) {
      g_object_set (G_OBJECT (nvvideorendererbin->transform),
          "nvbuf-memory-type", nvvideorendererbin->mem_type, NULL);
    }
    if (is_nvgpu) {
      g_snprintf (elem_name, sizeof (elem_name), "%s-nvvidconv",
          GST_ELEMENT_NAME (nvvideorendererbin));
      nvvidconv =
          gst_element_factory_make ("nvvideoconvert", elem_name);
      if (!nvvidconv) {
        GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
            ("Failed to create 'nvvideoconvert' element"), (NULL));
        return FALSE;
      }
      gst_bin_add (GST_BIN (nvvideorendererbin), nvvidconv);
      if (g_object_class_find_property (G_OBJECT_GET_CLASS (nvvidconv), "gpu-id")) {
        g_object_set (G_OBJECT (nvvidconv), "gpu-id",
            nvvideorendererbin->gpu_id, NULL);
      }
      if (g_object_class_find_property (G_OBJECT_GET_CLASS (nvvidconv), "nvbuf-memory-type")) {
        g_object_set (G_OBJECT (nvvidconv),
            "nvbuf-memory-type", nvvideorendererbin->mem_type, NULL);
      }
    }
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-queue",
      GST_ELEMENT_NAME (nvvideorendererbin));
  nvvideorendererbin->queue = gst_element_factory_make ("queue", elem_name);
  if (!nvvideorendererbin->queue) {
    GST_ELEMENT_ERROR (nvvideorendererbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'queue' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (nvvideorendererbin), nvvideorendererbin->queue);

  GstElement *connect_to = nvvideorendererbin->sink;
  if (nvvideorendererbin->cap_filter) {
    NVGSTDS_LINK_ELEMENT (nvvideorendererbin->cap_filter, connect_to, FALSE);
    connect_to = nvvideorendererbin->cap_filter;
  }

  if (nvvideorendererbin->transform) {
    NVGSTDS_LINK_ELEMENT (nvvideorendererbin->transform, connect_to, FALSE);
    connect_to = nvvideorendererbin->transform;
  }

  if (nvvidconv) {
    NVGSTDS_LINK_ELEMENT (nvvidconv, connect_to, FALSE);
    connect_to = nvvidconv;
  }

  NVGSTDS_LINK_ELEMENT (nvvideorendererbin->queue, connect_to, FALSE);

  NVGSTDS_BIN_SET_GHOST_PAD_TARGET (nvvideorendererbin,
      nvvideorendererbin->bin_sink_pad, nvvideorendererbin->queue, "sink",
      FALSE);

  return TRUE;
}

static GstStateChangeReturn
gst_ds_nvvideorenderer_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvVideoRendererBin *nvvideorendererbin =
      GST_DS_NVVIDEORENDERER_BIN (element);
  GstStateChangeReturn ret;

  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    if (!populate_bin (nvvideorendererbin)) {
      return GST_STATE_CHANGE_FAILURE;
    }
  }
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    remove_all_children (GST_BIN (nvvideorendererbin));
  }

  return ret;
}

static void
gst_ds_nvvideorenderer_bin_init (GstDsNvVideoRendererBin * nvvideorendererbin)
{
  nvvideorendererbin->offset_x = DEFAULT_OFFSET_X;
  nvvideorendererbin->offset_y = DEFAULT_OFFSET_Y;
  nvvideorendererbin->width = DEFAULT_WIDTH;
  nvvideorendererbin->height = DEFAULT_HEIGHT;
  nvvideorendererbin->conn_id = DEFAULT_CONN_ID;
  nvvideorendererbin->plane_id = DEFAULT_PLANE_ID;
  nvvideorendererbin->set_mode = DEFAULT_SET_MODE;
  nvvideorendererbin->color_range = DEFAULT_COLOR_RANGE;
  nvvideorendererbin->sync = DEFAULT_SYNC;
  nvvideorendererbin->qos = DEFAULT_QOS;
  nvvideorendererbin->gpu_id = DEFAULT_GPU_ID;
  nvvideorendererbin->mem_type = DEFAULT_NVBUF_MEMORY_TYPE;
  nvvideorendererbin->create_from_config =
      g_strdup (DEFAULT_CREATE_FROM_CONFIG);
  nvvideorendererbin->vrenderer_type = DEFAULT_NVDS_VIDEO_RENDERER;

  nvvideorendererbin->bin_sink_pad =
      gst_ghost_pad_new_no_target_from_template ("sink",
      gst_element_class_get_pad_template (GST_ELEMENT_GET_CLASS
          (nvvideorendererbin), "sink"));
  gst_element_add_pad (GST_ELEMENT (nvvideorendererbin),
      nvvideorendererbin->bin_sink_pad);

  GST_OBJECT_FLAG_SET (nvvideorendererbin, GST_ELEMENT_FLAG_SINK);
}

static void
gst_nvvideorenderer_bin_expose (GstVideoOverlay * overlay)
{
  GstDsNvVideoRendererBin *bin = GST_DS_NVVIDEORENDERER_BIN (overlay);
  if (bin->vrenderer_type == VIDEO_RENDERER_EGL && bin->sink)
    gst_video_overlay_expose (GST_VIDEO_OVERLAY (bin->sink));
}

static void
gst_nvvideorenderer_bin_set_window_handle (GstVideoOverlay * overlay,
    guintptr id)
{
  GstDsNvVideoRendererBin *bin = GST_DS_NVVIDEORENDERER_BIN (overlay);
  bin->window_id = id;
  bin->window_id_set = TRUE;

  if (bin->vrenderer_type == VIDEO_RENDERER_EGL && bin->sink)
    gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY (bin->sink), id);
}

static void
gst_nvvideorenderer_bin_set_render_rectangle (GstVideoOverlay * overlay, gint x,
    gint y, gint width, gint height)
{
  GstDsNvVideoRendererBin *bin = GST_DS_NVVIDEORENDERER_BIN (overlay);

  bin->render_rect_h = height;
  bin->render_rect_w = width;
  bin->render_rect_x = x;
  bin->render_rect_y = y;
  bin->render_rect_set = TRUE;

  if (bin->vrenderer_type == VIDEO_RENDERER_EGL && bin->sink)
    gst_video_overlay_set_render_rectangle (GST_VIDEO_OVERLAY (bin->sink), x, y,
        width, height);
}

static void
gst_nvvideorenderer_bin_videooverlay_init (GstVideoOverlayInterface * iface)
{
  iface->set_window_handle = gst_nvvideorenderer_bin_set_window_handle;
  iface->expose = gst_nvvideorenderer_bin_expose;
  iface->set_render_rectangle = gst_nvvideorenderer_bin_set_render_rectangle;
}
