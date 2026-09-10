/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/time.h>

#include "gstnvdsbufferpool.h"
#include "gstnvsegvisual.h"
#include "nvbufsurface.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvbufsurftransform.h"
#include "nvds_segvis.h"
#include "gst-nvquery.h"

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

GST_DEBUG_CATEGORY_STATIC (gst_nvseg_visual_debug);
#define GST_CAT_DEFAULT gst_nvseg_visual_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_GPU_DEVICE_ID,
    PROP_BATCH_SIZE,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_OPERATE_ON_SEG_META_ID,
    PROP_ORIGINAL_BACKGROUND,
    PROP_CLASS_ID,
    PROP_ALPHA,
    PROP_GPU,
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 0
#define DEFAULT_OUTPUT_WIDTH 1280
#define DEFAULT_OUTPUT_HEIGHT 720
#define DEFAULT_GPU_ID 0
#define DEFAULT_GRID_SIZE 0
#define DEFAULT_OPERATE_ON_SEG_META_ID -1
#define DEFAULT_ORIGINAL_BACKGROUND FALSE
#define DEFAULT_GPU_ON TRUE
#define DEFAULT_CLASS_ID 0
#define DEFAULT_ALPHA 1.0f

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvseg_visual_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvseg_visual_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvseg_visual_parent_class parent_class
G_DEFINE_TYPE (GstNvSegVisual, gst_nvseg_visual, GST_TYPE_BASE_TRANSFORM);

static void gst_nvseg_visual_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvseg_visual_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvseg_visual_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize);

static GstCaps* gst_nvseg_visual_fixate_caps(GstBaseTransform* btrans,
        GstPadDirection direction, GstCaps* caps, GstCaps* othercaps);

static gboolean gst_nvseg_visual_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);

static GstCaps* gst_nvseg_visual_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
    GstCaps* caps, GstCaps* filter);

static gboolean gst_nvseg_visual_start (GstBaseTransform * btrans);
static gboolean gst_nvseg_visual_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_nvseg_visual_transform(GstBaseTransform* btrans,
    GstBuffer* inbuf, GstBuffer* outbuf);

static GstFlowReturn
gst_nvseg_visual_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf);

/* For segmentation visulization */
static unsigned char class2BGR[] = {
  0, 0, 0,        0, 0, 128,      128, 128, 128,
  0, 128, 128,    128, 0, 0,      128, 0, 128,
  128, 128, 0,    0, 128, 0,      0, 0, 64,
  0, 0, 192,      0, 128, 64,     0, 128, 192,
  128, 0, 64,     128, 0, 192,    128, 128, 64,
  128, 128, 192,  0, 64, 0,       0, 64, 128,
  0, 192, 0,     0, 192, 128,    128, 64, 0,
  192, 192, 0
};

static void overlayColor(int* mask, unsigned char* buffer,
                             int height, int width,
			     int stream_num, int frame_num,
			     gboolean original_background, gint class_id, float alpha)
{
  unsigned char* buffer_R;
  unsigned char* buffer_G;
  unsigned char* buffer_B;

  if(!original_background) {
    for(int pix_id = 0; pix_id < width * height; pix_id++) {
      unsigned char* color = class2BGR + (mask[pix_id] + 3) * 3;
      buffer_R = buffer + pix_id * 4;
      buffer_G = buffer + pix_id * 4 + 1;
      buffer_B = buffer + pix_id * 4 + 2;
      *buffer_R = color[0];
      *buffer_G = color[1];
      *buffer_B = color[2];
    }
  } else {
    for(int pix_id = 0; pix_id < width * height; pix_id++) {
      if (mask[pix_id] != class_id) {
        unsigned char* color = class2BGR + (mask[pix_id] + 3) * 3;
        buffer_R = buffer + pix_id * 4;
        buffer_G = buffer + pix_id * 4 + 1;
        buffer_B = buffer + pix_id * 4 + 2;
        *buffer_R = (unsigned char)((color[0] * alpha) + (*buffer_R * (1-alpha)));
        *buffer_G = (unsigned char)((color[1] * alpha) + (*buffer_G * (1-alpha)));
        *buffer_B = (unsigned char)((color[2] * alpha) + (*buffer_B * (1-alpha)));
      }
    }
  }
#if 0
  char file_name[128];
  sprintf(file_name, "dump_map_stream%d_frame%03d.rgba", stream_num, frame_num);
  FILE* fp = fopen(file_name, "ab");
  fwrite(buffer, 4*height*width, 1, fp);
  fclose(fp);
#endif
}


/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvseg_visual_class_init (GstNvSegVisualClass * klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;
    gobject_class = (GObjectClass *) klass;
    gstelement_class = (GstElementClass *) klass;
    gstbasetransform_class = (GstBaseTransformClass *) klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvseg_visual_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvseg_visual_get_property);

    gstbasetransform_class->transform_size = GST_DEBUG_FUNCPTR(gst_nvseg_visual_transform_size);
    gstbasetransform_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_nvseg_visual_fixate_caps);
    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvseg_visual_set_caps);
    gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_nvseg_visual_transform_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvseg_visual_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvseg_visual_stop);

    gstbasetransform_class->transform = GST_DEBUG_FUNCPTR (gst_nvseg_visual_transform);

    gstbasetransform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR (gst_nvseg_visual_prepare_output_buffer);

    gstbasetransform_class->passthrough_on_same_caps = TRUE;

    /* Install properties */
    g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint ("unique-id",
            "Unique ID",
            "Unique ID for the element. Can be used to identify output of the"
            " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint ("gpu-id",
            "Set GPU Device ID",
            "Set GPU Device ID", 0,
            G_MAXUINT, 0,
            GParamFlags
            (G_PARAM_READWRITE |
                G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
        g_param_spec_uint ("batch-size", "Batch Size",
            "Maximum batch size for inference",
            1, G_MAXUINT, 1,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_WIDTH,
        g_param_spec_uint ("width", "Width",
            "Width of each frame in output batched buffer.",
            0, G_MAXUINT, DEFAULT_OUTPUT_WIDTH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_HEIGHT,
        g_param_spec_uint ("height", "Height",
            "Height of each frame in output batched buffer.",
            0, G_MAXUINT, DEFAULT_OUTPUT_HEIGHT,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_OPERATE_ON_SEG_META_ID,
      g_param_spec_int ("operate-on-seg-meta-id", "Visualization on Seg Meta ID",
          "visualize segmentation on seg-metadata with this unique ID.\n"
          "\t\t\tSet to -1 to visualize on all metadata.",
          -1, G_MAXINT, DEFAULT_OPERATE_ON_SEG_META_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_ORIGINAL_BACKGROUND,
      g_param_spec_boolean ("original-background", "Display original background",
          "Instead of masked background show original background.",
          DEFAULT_ORIGINAL_BACKGROUND,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_CLASS_ID,
        g_param_spec_uint ("class-id", "Background Class ID",
          "Class ID of the background, should be set if original-background is set to TRUE",
          0, G_MAXUINT, DEFAULT_CLASS_ID,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_ALPHA,
        g_param_spec_float ("alpha", "Alpha Value",
         "Alpha Value for per pixel blending.", 0.0, 1.0, DEFAULT_ALPHA,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_GPU,
      g_param_spec_boolean ("gpu-on", "GPU On/Off setting",
          "Switch between device and host memory",
          DEFAULT_GPU_ON,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_nvseg_visual_src_template));
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_nvseg_visual_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(gstelement_class,
          "nvsegvisual",
          "nvsegvisual",
          "Gstreamer NV Segmantation Visualization Plugin",
          "NVIDIA Corporation. Post on Deepstream for Jetson/Tesla forum for any queries "
          "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvseg_visual_init (GstNvSegVisual * segvisual)
{
    int class2BGR_size = sizeof(class2BGR);
    segvisual->sinkcaps =
      gst_static_pad_template_get_caps (&gst_nvseg_visual_sink_template);
    segvisual->srccaps =
      gst_static_pad_template_get_caps (&gst_nvseg_visual_src_template);

    /* Initialize all property variables to default values */
    segvisual->unique_id = DEFAULT_UNIQUE_ID;
    segvisual->output_width = DEFAULT_OUTPUT_WIDTH;
    segvisual->output_height = DEFAULT_OUTPUT_HEIGHT;
    segvisual->gpu_id = DEFAULT_GPU_ID;
    segvisual->operate_on_seg_meta_id = DEFAULT_OPERATE_ON_SEG_META_ID;
    segvisual->batch_size = 1;
    segvisual->original_background = DEFAULT_ORIGINAL_BACKGROUND;
    segvisual->flag_alloc_dev_memory = TRUE;
    segvisual->class_id = DEFAULT_CLASS_ID;
    segvisual->alpha = DEFAULT_ALPHA;
    segvisual->gpu_on = DEFAULT_GPU_ON;

    int is_nvgpu = 0;
    NvBufSurfaceDeviceInfo dev_info{};
    if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
        is_nvgpu = 1;
      }
    }
    if(is_nvgpu) {
        segvisual->cuda_mem_type = NVBUF_MEM_SURFACE_ARRAY;
    }
    else {
        segvisual->cuda_mem_type = NVBUF_MEM_CUDA_DEVICE;
    }

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
      _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

    cudaMalloc((void**)&segvisual->class2BGR_device, class2BGR_size);
    // Copy the class2BGR array from host to device
    cudaMemcpy(segvisual->class2BGR_device, class2BGR, class2BGR_size, cudaMemcpyHostToDevice);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvseg_visual_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (object);
    switch (prop_id) {
      case PROP_UNIQUE_ID:
        segvisual->unique_id = g_value_get_uint (value);
        break;
      case PROP_GPU_DEVICE_ID:
        segvisual->gpu_id = g_value_get_uint (value);
        break;
      case PROP_BATCH_SIZE:
        segvisual->batch_size = g_value_get_uint (value);
        break;
      case PROP_WIDTH:
        segvisual->output_width = g_value_get_uint (value);
        break;
      case PROP_HEIGHT:
        segvisual->output_height = g_value_get_uint (value);
        break;
      case PROP_OPERATE_ON_SEG_META_ID:
        segvisual->operate_on_seg_meta_id = g_value_get_int (value);
        break;
      case PROP_ORIGINAL_BACKGROUND:
        segvisual->original_background = g_value_get_boolean (value);
        break;
      case PROP_CLASS_ID:
        segvisual->class_id = g_value_get_uint (value);
        break;
      case PROP_ALPHA:
	segvisual->alpha = g_value_get_float (value);
	break;
      case PROP_GPU:
        segvisual->gpu_on = g_value_get_boolean (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvseg_visual_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
    GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (object);
    switch (prop_id) {
      case PROP_UNIQUE_ID:
        g_value_set_uint (value, segvisual->unique_id);
        break;
      case PROP_GPU_DEVICE_ID:
        g_value_set_uint (value, segvisual->gpu_id);
        break;
      case PROP_BATCH_SIZE:
        g_value_set_uint (value, segvisual->batch_size);
        break;
      case PROP_WIDTH:
        g_value_set_uint (value, segvisual->output_width);
        break;
      case PROP_HEIGHT:
        g_value_set_uint (value, segvisual->output_height);
        break;
      case PROP_OPERATE_ON_SEG_META_ID:
        g_value_set_int (value, segvisual->operate_on_seg_meta_id);
        break;
      case PROP_ORIGINAL_BACKGROUND:
        g_value_set_boolean (value, segvisual->original_background);
        break;
      case PROP_CLASS_ID:
        g_value_set_uint (value, segvisual->class_id);
        break;
      case PROP_ALPHA:
	g_value_set_float (value, segvisual->alpha);
        break;
      case PROP_GPU:
        g_value_set_boolean (value, segvisual->gpu_on);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvseg_visual_start(GstBaseTransform *btrans)
{
  GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (btrans);
  GST_DEBUG_OBJECT (segvisual, "gst_nvseg_visual_start\n");
  if (segvisual->stream == NULL) {
    cudaStreamCreate (&segvisual->stream);
  }
	return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvseg_visual_stop (GstBaseTransform * btrans)
{
  GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (btrans);

  if (segvisual->pool) {
    gst_buffer_pool_set_active (segvisual->pool, FALSE);
    gst_object_unref(segvisual->pool);
    segvisual->pool = NULL;
  }

  cudaFree(segvisual->class2BGR_device);
  cudaFree(segvisual->class_map_device);

   if (segvisual->stream)
   {
      cudaStreamDestroy (segvisual->stream);
      segvisual->stream = NULL;
   }
  GST_DEBUG_OBJECT (segvisual, "gst_nvseg_visual_stop\n");
  return TRUE;
}

static gboolean
gst_nvseg_visual_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize)
{
    gboolean ret = TRUE;
    GstVideoInfo info = {0};

    ret = gst_video_info_from_caps(&info, othercaps);
    if (ret) *othersize = info.size;

    return ret;
}

static GstCaps *
gst_nvseg_visual_transform_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCapsFeatures *feature = NULL;
  GstCaps *new_caps = NULL;
  GstCaps *temp_caps = NULL;

  if (direction == GST_PAD_SINK)
  {
    new_caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT, "height", GST_TYPE_INT_RANGE, 1,G_MAXINT, NULL);

  }
  else if (direction == GST_PAD_SRC)
  {
    new_caps = gst_caps_new_simple ("video/x-raw",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT, "height", GST_TYPE_INT_RANGE, 1,G_MAXINT, NULL);
  }

  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (new_caps, 0, feature);

  if(gst_caps_is_fixed (caps))
  {
    GstStructure *fs = gst_caps_get_structure (caps, 0);
    const GValue *fps_value;
    guint i, n = gst_caps_get_size(new_caps);

    fps_value = gst_structure_get_value (fs, "framerate");

    // We cannot change framerate
    for (i = 0; i < n; i++)
    {
      fs = gst_caps_get_structure (new_caps, i);
      gst_structure_set_value (fs, "framerate", fps_value);
    }
  }
  if (filter)
  {
    temp_caps = gst_caps_intersect(new_caps, filter);
    gst_caps_unref(new_caps);
    new_caps = temp_caps;
  }
  return new_caps;
}

/* fixate the caps on the other side */
static GstCaps* gst_nvseg_visual_fixate_caps(GstBaseTransform* btrans,
    GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
  GstNvSegVisual* segvisual = GST_NV_SEG_VISUAL(btrans);
  GstStructure *s2;
  GstCaps* result;

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);
  s2 = gst_caps_get_structure(othercaps, 0);

  {
    /* otherwise the dimension of the output heatmap needs to be fixated */
    gst_structure_fixate_field_nearest_int(s2, "width", segvisual->output_width);
    gst_structure_fixate_field_nearest_int(s2, "height", segvisual->output_height);

    gst_structure_remove_fields (s2, "width", "height", NULL);

    gst_structure_set (s2, "width", G_TYPE_INT, segvisual->output_width,
        "height", G_TYPE_INT, segvisual->output_height, NULL);

    result = gst_caps_ref(othercaps);
  }

  gst_caps_unref(othercaps);

  GST_INFO_OBJECT(segvisual, "CAPS fixate: %" GST_PTR_FORMAT ", direction %d",
      result, direction);

  return result;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvseg_visual_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (btrans);
  GstStructure *config = NULL;
  GstQuery *bsquery = NULL;
  guint batch_size = 0;

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps(&segvisual->video_info, incaps);

  if (segvisual->batch_size == 0)
  {
    g_print ("NvSegVisual: Received invalid batch_size i.e. 0\n");
    return FALSE;
  }

  bsquery = gst_nvquery_batch_size_new ();
  if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), bsquery))
  {
    gst_nvquery_batch_size_parse (bsquery, &batch_size);
    segvisual->batch_size = batch_size;
  }
  gst_query_unref (bsquery);

  if (!gst_video_info_from_caps (&segvisual->out_info, outcaps)) {
    GST_ERROR ("invalid output caps");
    return FALSE;
  }
  segvisual->output_fmt = GST_VIDEO_FORMAT_INFO_FORMAT (segvisual->out_info.finfo);

  if (!segvisual->pool)
  {
    segvisual->pool = gst_nvds_buffer_pool_new ();
    config = gst_buffer_pool_get_config (segvisual->pool);

    g_print ("in videoconvert caps = %s\n", gst_caps_to_string(outcaps));
    gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), 4, 4); // TODO: remove 4 hardcoding

    gst_structure_set (config,
        "memtype", G_TYPE_UINT, segvisual->cuda_mem_type,
        "gpu-id", G_TYPE_UINT, segvisual->gpu_id,
        "batch-size", G_TYPE_UINT, segvisual->batch_size, NULL);

    GST_INFO_OBJECT (segvisual, " %s Allocating Buffers in NVM Buffer Pool for Max_Views=%d\n",
        __func__, segvisual->batch_size);

    /* set config for the created buffer pool */
    if (!gst_buffer_pool_set_config (segvisual->pool, config)) {
      GST_WARNING ("bufferpool configuration failed");
      return FALSE;
    }

    gboolean is_active = gst_buffer_pool_set_active (segvisual->pool, TRUE);
    if (!is_active) {
      GST_WARNING (" Failed to allocate the buffers inside the output pool");
      return FALSE;
    } else {
      GST_DEBUG (" Output buffer pool (%p) successfully created",
                  segvisual->pool);
    }
  }

  return TRUE;
}

static GstFlowReturn
gst_nvseg_visual_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf)
{
  GstBuffer *gstOutBuf = NULL;
  GstFlowReturn result = GST_FLOW_OK;
  GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (trans);

  result = gst_buffer_pool_acquire_buffer (segvisual->pool, &gstOutBuf, NULL);
  GST_DEBUG_OBJECT (segvisual, "%s : Gst-OutBuf=%p\n",
		  __func__, gstOutBuf);

  if (result != GST_FLOW_OK)
  {
    GST_ERROR_OBJECT (segvisual, "gst_segvisual_prepare_output_buffer failed");
    return result;
  }

  *outbuf = gstOutBuf;
  return result;
}

static GstFlowReturn
gst_nvseg_visual_transform_internal(GstBaseTransform *btrans,
                                       GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstNvSegVisual *segvisual = GST_NV_SEG_VISUAL (btrans);
  GstFlowReturn flow_ret = GST_FLOW_OK;
  gpointer state = NULL;
  GstMeta *gst_meta = NULL;
  NvDsMeta *dsmeta = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  guint i = 0;
  GstMapInfo outmap = GST_MAP_INFO_INIT;
  void * dst = NULL;
  NvBufSurfTransformConfigParams config_params;
  int err = -1;

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
  {
	  g_print ("%s output buf map failed\n", __func__);
	  return GST_FLOW_ERROR;
  }

  config_params.compute_mode = NvBufSurfTransformCompute_Default;
  config_params.gpu_id = segvisual->gpu_id;
  config_params.cuda_stream = segvisual->stream;
  err = NvBufSurfTransformSetSessionParams(&config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ERROR_OBJECT(segvisual,"Set session params failed \n");
    return GST_FLOW_ERROR;
  }

  NvBufSurface *dstSurf = (NvBufSurface *)outmap.data;

  NvBufSurfTransform_Error error;
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ))
  {
     g_print ("%s output buf map failed\n", __func__);
     return GST_FLOW_ERROR;
  }
  NvBufSurface *srcSurf = (NvBufSurface *)inmap.data;

  NvBufSurfTransformParams transform_params = { 0 };
  transform_params.transform_flip = NvBufSurfTransform_None;
  transform_params.transform_filter = NvBufSurfTransformInter_Default;
  
  /* Batched tranformation. */
  error = NvBufSurfTransform(srcSurf, dstSurf, &transform_params);
  if (error != NvBufSurfTransformError_Success)
  {
    gst_buffer_unmap (inbuf, &inmap);
    gst_buffer_unmap (outbuf, &outmap);
    return GST_FLOW_ERROR;
  }

  // Required in the case of tiler
  if (!gst_buffer_copy_into (outbuf, inbuf, GST_BUFFER_COPY_META, 0, -1)) {
	  GST_DEBUG ("Buffer metadata copy failed \n");
  }
  GST_BUFFER_PTS (outbuf) = GST_BUFFER_PTS(inbuf);

  if (cudaSetDevice(segvisual->gpu_id) != cudaSuccess)
  {
    g_printerr("Error: failed to set GPU to %d\n", segvisual->gpu_id);
    return GST_FLOW_ERROR;
  }

  while ((gst_meta = gst_buffer_iterate_meta (inbuf, &state)))
  {
    if (gst_meta_api_type_has_tag(gst_meta->info->api, _dsmeta_quark))
    {
      dsmeta = (NvDsMeta *) gst_meta;
      if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *)dsmeta->meta_data;
        break;
      }
    }
  }

  if (batch_meta == NULL)
  {
    g_print ("batch_meta not found, skipping optical flow visual draw execution\n");
    return GST_FLOW_ERROR;
  }

  dstSurf->numFilled = batch_meta->num_frames_in_batch;

  // TODO: Improve by mapping and unmapping buffer at the time of buffer creation
  if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    NvBufSurfaceMap (dstSurf, -1, -1, NVBUF_MAP_WRITE);
  }

  static int frame_n = 0;
  for (i=0; i < batch_meta->num_frames_in_batch; i++)
  {
    NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
    if (frame_meta->frame_user_meta_list)
    {
      NvDsFrameMetaList *fmeta_list = NULL;
      NvDsUserMeta *of_user_meta = NULL;

      for (fmeta_list = frame_meta->frame_user_meta_list; fmeta_list != NULL; fmeta_list = fmeta_list->next)
      {
        of_user_meta = (NvDsUserMeta *)fmeta_list->data;
        if (of_user_meta && of_user_meta->base_meta.meta_type == NVDSINFER_SEGMENTATION_META) {
          NvDsInferSegmentationMeta *segmeta = (NvDsInferSegmentationMeta *) (of_user_meta->user_meta_data);
          GST_DEBUG("unique_id/classes/width/height=%d/%d/%d/%d\n",
                    segmeta->unique_id,
                    segmeta->classes,
                    segmeta->width,
                    segmeta->height);
          GST_DEBUG("dstSurf [%d] dataSize=%d\n", i, dstSurf->surfaceList[i].dataSize);

          /* skip the segmentation meta visualization if seg-meta-id is not matched with operate_on_seg_meta_id */
          if (segvisual->operate_on_seg_meta_id>0 && segvisual->operate_on_seg_meta_id != segmeta->unique_id) {
            GST_DEBUG("skipping the NvDsInferSegmentationMeta\n");
            continue;
          }

          if (segvisual->flag_alloc_dev_memory) {
            segvisual->flag_alloc_dev_memory = FALSE;
            cudaMalloc((void**)&segvisual->class_map_device, segmeta->width * segmeta->height * sizeof(int));
          }
          cudaMemcpyAsync(segvisual->class_map_device, segmeta->class_map, segmeta->width * segmeta->height* sizeof(int), cudaMemcpyHostToDevice, segvisual->stream);
          dst = (void*) dstSurf->surfaceList[i].dataPtr;
#if defined(__aarch64__)
          CUresult status;
          CUeglFrame eglFrame;
          memset(&eglFrame, 0, sizeof(CUeglFrame));
          CUgraphicsResource pResource = NULL;
          EGLImageKHR eglimage_dst = NULL;
          if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {

            if (dstSurf->surfaceList[i].mappedAddr.eglImage == NULL) {
              NvBufSurfaceMapEglImage (dstSurf, -1);
            }
            eglimage_dst = dstSurf->surfaceList[i].mappedAddr.eglImage;

            status = cuGraphicsEGLRegisterImage(&pResource, eglimage_dst,
                CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
            if (status != CUDA_SUCCESS) {
              printf("cuGraphicsEGLRegisterImage failed: %d\n", status);
              exit (-1);
            }
            status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
            if (status != CUDA_SUCCESS) {
              printf("cuGraphicsSubResourceGetMappedArray failed\n");
            }

            dst = (void*) eglFrame.frame.pPitch[0];
          }
#endif
         if(segvisual->gpu_on) {
          // Launch the CUDA kernel to update the pixel buffer for the entire 2D grid
          updatePixelBuffer((unsigned char*) dst, segvisual->class_map_device, segvisual->class2BGR_device, segmeta->width,
                 segmeta->height, segvisual->original_background, segvisual->class_id, segvisual->alpha,
                 dstSurf->surfaceList[i].planeParams.pitch[0], segvisual->stream);
          cudaStreamSynchronize(segvisual->stream);
         }
         else {
          int rgba_bytes = 4;
          unsigned char* buffer = (unsigned char*)(malloc(rgba_bytes * segmeta->height * segmeta->width));
          char dummyByte;
          cudaError_t err;

          cudaMemcpy2D((void*)buffer, rgba_bytes * segmeta->width,
                      (void*)dstSurf->surfaceList[i].dataPtr,
                      dstSurf->surfaceList[i].planeParams.pitch[0],
                      rgba_bytes * segmeta->width, segmeta->height,
                      cudaMemcpyDeviceToHost);

          overlayColor(segmeta->class_map, buffer, segmeta->height, segmeta->width, i, frame_n, segvisual->original_background, (gint)segvisual->class_id, segvisual->alpha);

          if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
#if defined(__aarch64__)
            for (unsigned int h = 0; h < segmeta->height; h++) {
              memcpy((char *)dstSurf->surfaceList[i].mappedAddr.addr[0] +
                        h * dstSurf->surfaceList[i].planeParams.pitch[0],
                    buffer + h * segmeta->width * 4,
                    segmeta->width * 4);
            }
#endif
          }
          else {
            cudaMemcpy2D((void*)dstSurf->surfaceList[i].dataPtr,
                      dstSurf->surfaceList[i].planeParams.pitch[0],
                      (void*)buffer, rgba_bytes * segmeta->width,
                      rgba_bytes * segmeta->width, segmeta->height,
                      cudaMemcpyHostToDevice);

            //copy one byte to make sure above copy is complete.
            {
                err = cudaMemcpy (&dummyByte, (void*)dstSurf->surfaceList[i].dataPtr,
                        1, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    printf ("nvbufsurface: NvBufSurfaceSysToHWCopy: failed in mem copy\n");
                    free(buffer);
                    return GST_FLOW_ERROR;
                }
            }
          }
          free(buffer);
        }
#if defined(__aarch64__)
          if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
            status = cuGraphicsUnregisterResource(pResource);
            if (status != CUDA_SUCCESS) {
              printf ("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
            }
          }
#endif
        }
      }
    }
  }

  frame_n++;

  if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    NvBufSurfaceSyncForDevice (dstSurf, -1, -1);
    NvBufSurfaceUnMap (dstSurf, -1, -1);
  }

  gst_buffer_unmap (inbuf, &inmap);
  gst_buffer_unmap (outbuf, &outmap);
  return flow_ret;
}

/**
 * Called when the plugin works in non-passthough mode
 */
static GstFlowReturn
gst_nvseg_visual_transform(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
  return gst_nvseg_visual_transform_internal(btrans, inbuf, outbuf);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvseg_visual_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvseg_visual_debug, "nvsegvisual", 0,
      "nvsegvisual plugin");

  return gst_element_register (plugin, "nvsegvisual", GST_RANK_PRIMARY,
          GST_TYPE_NV_SEG_VISUAL);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_segvisual,
    DESCRIPTION, nvseg_visual_plugin_init, "4.0", LICENSE, BINARY_PACKAGE, URL)
