/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gstnvofvisual.h"
#include "gstnvdsmeta.h"

#include "nvofvisual_draw.h"
#include "nvbufsurface.h"
#include "nvds_opticalflow_meta.h"

#include "nvtx_helper.h"

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

GST_DEBUG_CATEGORY_STATIC (gst_nvof_visual_debug);
#define GST_CAT_DEFAULT gst_nvof_visual_debug

static GQuark _dsmeta_quark = 0;

//#define MEASURE_TIME
#ifdef MEASURE_TIME
#include <sys/time.h>
#include <stdio.h>

#define START_PROFILE \
    { \
  struct timeval t1, t2; \
  double elapsedTime = 0; \
  double totalReadTime = 0; \
  gettimeofday(&t1, NULL);

#define STOP_PROFILE(X) \
    gettimeofday(&t2, NULL); \
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      \
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   \
    totalReadTime += elapsedTime; \
    printf("(%s)  %p : BS %d %s ElaspedTime=%f TotalTime=%f ms\n", \
        GST_ELEMENT_NAME(ofvisual), ofvisual, batch_meta->num_frames_in_batch, \
        X, elapsedTime, totalReadTime); \
    }

#else
#define START_PROFILE
#define STOP_PROFILE(X)
#endif

/* Enum to identify properties */
enum
{
    PROP_0,
    PROP_GPU_DEVICE_ID,
};

/* Default values for properties */
#define DEFAULT_OUTPUT_WIDTH 1280
#define DEFAULT_OUTPUT_HEIGHT 720
#define DEFAULT_GPU_ID 0
#define DEFAULT_GRID_SIZE 0

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvof_visual_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvof_visual_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ RGBA }")));


inline bool CUDA_CHECK_(gint e, gint iLine, const gchar *szFile) {
  if (e != cudaSuccess) {
    std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile << std::endl;
    return false;
  }
  return true;
}

#define cuda_ck(call) CUDA_CHECK_(call, __LINE__, __FILE__)

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvof_visual_parent_class parent_class
G_DEFINE_TYPE (GstNvOFVisual, gst_nvof_visual, GST_TYPE_BASE_TRANSFORM);

static void gst_nvof_visual_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvof_visual_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvof_visual_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize);

static GstCaps* gst_nvof_visual_fixate_caps(GstBaseTransform* btrans,
        GstPadDirection direction, GstCaps* caps, GstCaps* othercaps);

static gboolean gst_nvof_visual_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);

static GstCaps* gst_nvof_visual_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
    GstCaps* caps, GstCaps* filter);

static gboolean gst_nvof_visual_start (GstBaseTransform * btrans);
static gboolean gst_nvof_visual_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_nvof_visual_transform(GstBaseTransform* btrans, 
    GstBuffer* inbuf, GstBuffer* outbuf);
static gboolean
gst_nvof_visual_accept_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps);

static GstFlowReturn
gst_nvof_visual_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvof_visual_class_init (GstNvOFVisualClass * klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;
    gobject_class = (GObjectClass *) klass;
    gstelement_class = (GstElementClass *) klass;
    gstbasetransform_class = (GstBaseTransformClass *) klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvof_visual_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvof_visual_get_property);

    gstbasetransform_class->transform_size = GST_DEBUG_FUNCPTR(gst_nvof_visual_transform_size);
    gstbasetransform_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_nvof_visual_fixate_caps);
    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvof_visual_set_caps);
    gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_nvof_visual_transform_caps);
    gstbasetransform_class->accept_caps = GST_DEBUG_FUNCPTR (gst_nvof_visual_accept_caps);
    //gstbasetransform_class->decide_allocation = GST_DEBUG_FUNCPTR (gst_nvof_visual_decide_allocation);

    gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvof_visual_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvof_visual_stop);

    gstbasetransform_class->transform = GST_DEBUG_FUNCPTR (gst_nvof_visual_transform);

    gstbasetransform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR (gst_nvof_visual_prepare_output_buffer);

    gstbasetransform_class->passthrough_on_same_caps = TRUE;

    /* Install properties */
    g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint ("gpu-id",
            "Set GPU Device ID",
            "Set GPU Device ID", 0,
            G_MAXUINT, 0,
            GParamFlags
            (G_PARAM_READWRITE |
                G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_nvof_visual_src_template));
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_nvof_visual_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(gstelement_class,
          "nvofvisual",
          "nvofvisual",
          "Gstreamer NV Optical Flow Visualization Plugin",
          "NVIDIA Corporation. Post on Deepstream for Jetson/Tesla forum for any queries "
          "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvof_visual_init (GstNvOFVisual * ofvisual)
{
    ofvisual->sinkcaps =
      gst_static_pad_template_get_caps (&gst_nvof_visual_sink_template);
    ofvisual->srccaps =
      gst_static_pad_template_get_caps (&gst_nvof_visual_src_template);

    /* Initialize all property variables to default values */
    ofvisual->output_width = DEFAULT_OUTPUT_WIDTH;
    ofvisual->output_height = DEFAULT_OUTPUT_HEIGHT;
    ofvisual->gpu_id = DEFAULT_GPU_ID;

    // TODO:
    ofvisual->block_size_x = 4;
    ofvisual->block_size_y = 4;
    ofvisual->max_buffers = 4;
    ofvisual->batch_size = 0;

    int is_nvgpu = 0;
    NvBufSurfaceDeviceInfo dev_info{};
    if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
        is_nvgpu = 1;
      }
    }
    if(is_nvgpu) {
      ofvisual->cuda_mem_type = NVBUF_MEM_SURFACE_ARRAY;
    }
    else {
      ofvisual->cuda_mem_type = NVBUF_MEM_CUDA_DEVICE;
    }

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
      _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvof_visual_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (object);
    switch (prop_id) {
      case PROP_GPU_DEVICE_ID:
        ofvisual->gpu_id = g_value_get_uint (value);
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
gst_nvof_visual_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
    GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (object);
    switch (prop_id) {
      case PROP_GPU_DEVICE_ID:
        g_value_set_uint (value, ofvisual->gpu_id);
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
gst_nvof_visual_start(GstBaseTransform *btrans)
{
  GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (btrans);

  GST_DEBUG_OBJECT (ofvisual, "gst_nvof_visual_start\n");

  return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvof_visual_stop (GstBaseTransform * btrans)
{
    GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (btrans);
    gint i = 0;

    if (ofvisual->pool) {
      gst_buffer_pool_set_active (ofvisual->pool, FALSE);
      gst_object_unref(ofvisual->pool);
      ofvisual->pool = NULL;
    }

    for (i=0; i < ofvisual->batch_size; i++)
    {
        if (ofvisual->streams_array[i])
        {
            cuda_ck(cudaStreamDestroy(ofvisual->streams_array[i]));
        }
    }
    GST_DEBUG_OBJECT (ofvisual, "gst_nvof_visual_stop\n");
    return TRUE;
}

static gboolean
gst_nvof_visual_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize)
{
    gboolean ret = TRUE;
    GstVideoInfo info = {0};

    ret = gst_video_info_from_caps(&info, othercaps);
    if (ret) *othersize = info.size;

    return ret;
}

static gboolean
gst_nvof_visual_accept_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps)
{
  gboolean ret = TRUE;
  GstCaps *allowed = NULL;
  GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (btrans);

  GST_DEBUG_OBJECT (ofvisual, "accept caps %" GST_PTR_FORMAT, caps);

  /* get all the formats we can handle on this pad */
  if (direction == GST_PAD_SINK)
    allowed = ofvisual->sinkcaps;
  else
    allowed = ofvisual->srccaps;

  if (!allowed) {
    GST_DEBUG_OBJECT (ofvisual, "failed to get allowed caps");
    GST_DEBUG_OBJECT (ofvisual,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    return FALSE;
  }

  GST_DEBUG_OBJECT (ofvisual, "allowed caps %" GST_PTR_FORMAT, allowed);

  /* intersect with the requested format */
  ret = gst_caps_is_subset (caps, allowed);
  if (!ret) {
    GST_DEBUG_OBJECT (ofvisual,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    return FALSE;
  }

  return ret;
}

static GstCaps *
gst_nvof_visual_transform_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (btrans);
  GstCapsFeatures *feature = NULL;
  GstCaps *new_caps = NULL;
  GstCaps *temp_caps = NULL;

  if (direction == GST_PAD_SINK)
  {
	  GST_INFO_OBJECT (ofvisual, "\n%s SINK INPUT CAPS = %" GST_PTR_FORMAT "\n\n", __func__, caps);
     new_caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT, "height", GST_TYPE_INT_RANGE, 1,G_MAXINT, NULL);

  }
  else if (direction == GST_PAD_SRC)
  {
	  GST_INFO_OBJECT (ofvisual, "\n%s SRC INPUT CAPS = %" GST_PTR_FORMAT "\n\n", __func__, caps);
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
static GstCaps* gst_nvof_visual_fixate_caps(GstBaseTransform* btrans,
    GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
  GstNvOFVisual* ofvisual = GST_NV_OF_VISUAL(btrans);
  GstStructure *s1, *s2;
  GstCaps* result;
  gint num, denom;

  GST_INFO_OBJECT (ofvisual, "%s : CAPS = %" GST_PTR_FORMAT "\n\n", __func__, caps);
  GST_INFO_OBJECT (ofvisual, "%s : OTHER CAPS = %" GST_PTR_FORMAT "\n\n", __func__, othercaps);

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);
  s2 = gst_caps_get_structure(othercaps, 0);

  {
    s1 = gst_caps_get_structure(caps, 0);
    gst_structure_get_int (s1, "width", &ofvisual->output_width);
    gst_structure_get_int (s1, "height", &ofvisual->output_height);
    gst_structure_get_int (s1, "of-block-size-x", &ofvisual->block_size_x);
    gst_structure_get_int (s1, "of-block-size-y", &ofvisual->block_size_y);

    ofvisual->output_width  = ofvisual->output_width / ofvisual->block_size_x;
    ofvisual->output_height = ofvisual->output_height / ofvisual->block_size_y;

    /* otherwise the dimension of the output heatmap needs to be fixated */
    gst_structure_fixate_field_nearest_int(s2, "width", ofvisual->output_width);
    gst_structure_fixate_field_nearest_int(s2, "height", ofvisual->output_height);
    if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
    {
      gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);
    }

    gst_structure_remove_fields (s2, "width", "height", NULL);

    gst_structure_set (s2, "width", G_TYPE_INT, ofvisual->output_width,
        "height", G_TYPE_INT, ofvisual->output_height, NULL);

    result = gst_caps_ref(othercaps);
  }

  gst_caps_unref(othercaps);

  GST_INFO_OBJECT (ofvisual, "%s : CAPS = %" GST_PTR_FORMAT "\n\n", __func__, othercaps);

  GST_INFO_OBJECT(ofvisual, "CAPS fixate: %" GST_PTR_FORMAT ", direction %d",
      result, direction);

  return result;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvof_visual_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (btrans);
  GstStructure *s1, *s2;
  GstStructure *config = NULL;
  gint i = 0;

  GST_INFO_OBJECT (ofvisual, "%s : IN CAPS = %" GST_PTR_FORMAT "\n\n", __func__, incaps);
  GST_INFO_OBJECT (ofvisual, "%s : OUT CAPS = %" GST_PTR_FORMAT"\n\n", __func__, outcaps);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps(&ofvisual->video_info, incaps);

  s1 = gst_caps_get_structure(outcaps, 0);
  gst_structure_get_int (s1, "width", &ofvisual->output_width);
  gst_structure_get_int (s1, "height", &ofvisual->output_height);

  s2 = gst_caps_get_structure(incaps, 0);
  gst_structure_get_int (s2, "batch-size", &ofvisual->batch_size);

  if (ofvisual->batch_size == 0)
  {
	  g_print ("nvofvisual: Received invalid batch_size i.e. 0\n");
	  return FALSE;
  }

  if (cudaSetDevice(ofvisual->gpu_id) != cudaSuccess)
  {
	  g_printerr("Error: failed to set GPU to %d\n", ofvisual->gpu_id);
	  return GST_FLOW_ERROR;
  }

  // Allocate seperate cudaStream for each source
  ofvisual->streams_array = (cudaStream_t *) calloc (1, sizeof (cudaStream_t) * ofvisual->batch_size);
  for (i=0; i < ofvisual->batch_size; i++)
  {
	  cuda_ck(cudaStreamCreateWithFlags(&ofvisual->streams_array[i], cudaStreamNonBlocking));
  }

  if (!gst_video_info_from_caps (&ofvisual->out_info, outcaps)) {
	  GST_ERROR ("invalid output caps");
	  return FALSE;
  }
  ofvisual->output_fmt = GST_VIDEO_FORMAT_INFO_FORMAT (ofvisual->out_info.finfo);

  if (!ofvisual->pool)
  {
	  ofvisual->pool = gst_nvds_buffer_pool_new ();

	  config = gst_buffer_pool_get_config (ofvisual->pool);

	  GST_INFO_OBJECT (ofvisual, "in videoconvert caps = %" GST_PTR_FORMAT "\n", outcaps);
	  gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), 4, 4); // TODO: remove 4 hardcoding

	  gst_structure_set (config,
			  "memtype", G_TYPE_UINT, ofvisual->cuda_mem_type,
			  "gpu-id", G_TYPE_UINT, ofvisual->gpu_id,
			  "batch-size", G_TYPE_UINT, ofvisual->batch_size, NULL);

	  GST_INFO_OBJECT (ofvisual, " %s Allocating Buffers in NVM Buffer Pool for Max_Views=%d\n",
			  __func__, ofvisual->batch_size);

	  /* set config for the created buffer pool */
	  if (!gst_buffer_pool_set_config (ofvisual->pool, config)) {
		  GST_WARNING ("bufferpool configuration failed");
		  return FALSE;
	  }

	  gboolean is_active = gst_buffer_pool_set_active (ofvisual->pool, TRUE);
	  if (!is_active) {
		  GST_WARNING (" Failed to allocate the buffers inside the output pool");
		  return FALSE;
	  } else {
		  GST_DEBUG (" Output buffer pool (%p) successfully created with %d buffers",
				  ofvisual->pool, ofvisual->max_buffers);
	  }
  }

  return TRUE;
}

static GstFlowReturn
gst_nvof_visual_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf)
{
  GstBuffer *gstOutBuf = NULL;
  GstFlowReturn result = GST_FLOW_OK;
  GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (trans);

#if 0
  if (ofvisual->pool_created == FALSE)
  {
	  ofvisual->pool_created = gst_nvof_visual_allocate_output_buffer (ofvisual);
	  if (ofvisual->pool_created == FALSE)
	  {
		  g_print ("gst_nvof_visual_allocate_output_buffer failed\n");
		  return GST_FLOW_ERROR;
	  }
  }
#endif
  result = gst_buffer_pool_acquire_buffer (ofvisual->pool, &gstOutBuf, NULL);
  GST_DEBUG_OBJECT (ofvisual, "%s : Frame=%lu Gst-OutBuf=%p\n",
		  __func__, ofvisual->frame_num, gstOutBuf);

  if (result != GST_FLOW_OK)
  {
    GST_ERROR_OBJECT (ofvisual, "gst_ofvisual_prepare_output_buffer failed");
    return result;
  }

  *outbuf = gstOutBuf;
  return result;
}

static GstFlowReturn
gst_nvof_visual_transform_internal(GstBaseTransform *btrans,
                                       GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstNvOFVisual *ofvisual = GST_NV_OF_VISUAL (btrans);
  GstFlowReturn flow_ret = GST_FLOW_OK;
  gpointer state = NULL;
  gboolean of_metadata_found = FALSE;
  GstMeta *gst_meta = NULL;
  NvDsMeta *dsmeta = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  guint i = 0;

  GstMapInfo outmap = GST_MAP_INFO_INIT;

  START_PROFILE;

  char context_name[100];
  snprintf(context_name, sizeof(context_name), "%s_(Frame=%" G_GUINT64_FORMAT ")",
      GST_ELEMENT_NAME(ofvisual), ofvisual->frame_num);
  nvtx_helper_push_pop(context_name);

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
  {
	  g_print ("%s output buf map failed\n", __func__);
	  return GST_FLOW_ERROR;
  }

  NvBufSurface *dstSurf = (NvBufSurface *)outmap.data;
  gst_buffer_unmap (outbuf, &outmap);

  // Required in the case of tiler
  if (!gst_buffer_copy_into (outbuf, inbuf, GST_BUFFER_COPY_META, 0, -1)) {
	  GST_DEBUG ("Buffer metadata copy failed \n");
  }

  GST_BUFFER_PTS (outbuf) = GST_BUFFER_PTS (inbuf);

  if (cudaSetDevice(ofvisual->gpu_id) != cudaSuccess)
  {
	  g_printerr("Error: failed to set GPU to %d\n", ofvisual->gpu_id);
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


#if 1
  if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    NvBufSurfaceSyncForCpu (dstSurf, -1, -1);
  }

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
			  if (of_user_meta && of_user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META)
			  {
				  NvDsOpticalFlowMeta *ofmeta = (NvDsOpticalFlowMeta *) (of_user_meta->user_meta_data);
				  void * dst = NULL;

          if (ofmeta)
          {
            dst = (void*) dstSurf->surfaceList[i].dataPtr;
#if defined(__aarch64__)
            CUresult status;
            CUeglFrame eglFrame;
            memset(&eglFrame, 0, sizeof(CUeglFrame));
            CUgraphicsResource pResource = NULL;
            EGLImageKHR eglimage_dst = NULL;
            if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {

              if (dstSurf->surfaceList[i].mappedAddr.eglImage == NULL)
              {
                NvBufSurfaceMapEglImage (dstSurf, -1);
              }
              eglimage_dst = dstSurf->surfaceList[i].mappedAddr.eglImage;

              status = cuGraphicsEGLRegisterImage(&pResource, eglimage_dst,
                  CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
              if (status != CUDA_SUCCESS)
              {
                printf("cuGraphicsEGLRegisterImage failed: %d\n", status);
                exit (-1);
              }

              status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
              if (status != CUDA_SUCCESS)
              {
                printf("cuGraphicsSubResourceGetMappedArray failed\n");
              }

              dst = (void*) eglFrame.frame.pPitch[0];
            }
#endif

#if 1
					  // Cuda Kernel based Drawing
					  DrawOpticalFlow_Cuda (ofmeta->data, dst, ofmeta->cols, ofmeta->rows, dstSurf->surfaceList[i].planeParams.pitch[0], 10.0, ofvisual->streams_array[i]);
#else
					  // CPU based Drawing
					  DrawOpticalFlow (ofmeta->data, dst, ofmeta->cols, ofmeta->rows, 10.0);
#endif
					  of_metadata_found = TRUE;

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
  }

  for (i=0; i < batch_meta->num_frames_in_batch; i++)
  {
	  cudaStreamSynchronize (ofvisual->streams_array[i]);
  }
  if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    NvBufSurfaceSyncForDevice (dstSurf, -1, -1);
  }
#else
  static int add = 0xf;
  NvBufSurfaceMemSet (dstSurf, -1, -1, add);
  add += 1;
#endif
  if(dstSurf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    NvBufSurfaceUnMap (dstSurf, -1, -1);
  }

  if (of_metadata_found == FALSE)
  {
    GST_WARNING_OBJECT (ofvisual, "OF METADATA NOT FOUND\n");
  }

  nvtx_helper_push_pop(NULL);
  STOP_PROFILE("+++++++++ NVOFVISUAL BUFFER PROCESSED +++++++++");

  return flow_ret;
}

/**
 * Called when the plugin works in non-passthough mode
 */
static GstFlowReturn
gst_nvof_visual_transform(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
    return gst_nvof_visual_transform_internal(btrans, inbuf, outbuf);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvof_visual_plugin_init (GstPlugin * plugin)
{
    GST_DEBUG_CATEGORY_INIT (gst_nvof_visual_debug, "nvofvisual", 0,
      "nvofvisual plugin");

    return gst_element_register (plugin, "nvofvisual", GST_RANK_PRIMARY,
          GST_TYPE_NV_OF_VISUAL);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_ofvisual,
    DESCRIPTION, nvof_visual_plugin_init, "4.0", LICENSE, BINARY_PACKAGE, URL)
