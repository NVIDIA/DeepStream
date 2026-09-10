/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sys/time.h>
#include <sys/time.h>
#include <stdio.h>
#include <string>

#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <memory>

#include "gst-nvevent.h"
#include "gst-nvquery.h"
#include "gst-nvcommon.h"
#include "gstnvdsbufferpool.h"

#include "gstnvdsxfer.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvdsxfer_debug);
#define GST_CAT_DEFAULT gst_nvdsxfer_debug

//#define MEASURE_TIME
#ifdef MEASURE_TIME
#define START_PROFILE \
  { \
    struct timeval t1, t2; \
    double elapsedTime = 0; \
    double totalReadTime = 0; \
    gettimeofday(&t1, NULL);

#define STOP_PROFILE(ele, X) \
    gettimeofday(&t2, NULL); \
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      \
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   \
    totalReadTime += elapsedTime; \
    GST_INFO_OBJECT(ele, " %p : #%d %s ElaspedTime=%f TotalTime=%f ms", \
        ele, ele->frame_num, \
        X, elapsedTime, totalReadTime); \
  }
#else
#define START_PROFILE
#define STOP_PROFILE(ele, X)
#endif

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_GPU_DEVICE_ID,
  PROP_BATCH_SIZE,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_BUFFER_POOL_SIZE,
  PROP_ENABLE_PEER_TO_DEVICE
};

#define FORMAT_NV12 "NV12"
#define FORMAT_I420 "I420"
#define FORMAT_RGBA "RGBA"

/* Default values for properties */
#define DEFAULT_GPU_ID 0

#define MAX_P2P_DEVICES 1024

#define MAX_NVBUFFERS 16
#define MIN_POOL_BUFFERS 2
#define MAX_POOL_BUFFERS (MAX_NVBUFFERS)

#define DEFAULT_BATCH_SIZE (1)
#define DEFAULT_BUFFER_POOL_SIZE (4)
#define DEFAULT_ENABLE_PEER_DEVICE (-1)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess && (cudaErrorPeerAccessAlreadyEnabled != cuda_status) ) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

static GQuark _dsmeta_quark = 0;

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvdsxfer_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "NV12, I420, RGBA }") ";"));

static GstStaticPadTemplate gst_nvdsxfer_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, I420, RGBA }") ";"));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvdsxfer_parent_class parent_class
G_DEFINE_TYPE (GstNvXfer, gst_nvdsxfer, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdsxfer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsxfer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean
gst_nvdsxfer_sink_event (GstBaseTransform * btrans, GstEvent *event);

static gboolean gst_nvdsxfer_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvdsxfer_start (GstBaseTransform * btrans);
static gboolean gst_nvdsxfer_stop (GstBaseTransform * btrans);
static GstFlowReturn gst_nvdsxfer_transform (GstBaseTransform* btrans,
    GstBuffer* inbuf, GstBuffer* outbuf);

static GstFlowReturn
gst_nvdsxfer_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf);

/* fixate the caps on the other side */
static GstCaps* gst_nvdsxfer_fixate_caps(GstBaseTransform* btrans,
    GstPadDirection direction, GstCaps* in_caps, GstCaps* othercaps)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  GstCaps* result = NULL;
  GstStructure *s1, *s2;
  gint width, height;
  gint i, num, denom;
  const gchar *inputFmt = NULL;

  GST_DEBUG_OBJECT (nvdsxfer, "%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

  // Check if othercaps has format NV12, I420 and RGBA
  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

  int num_output_caps = gst_caps_get_size (othercaps);

  s1 = gst_caps_get_structure(in_caps, 0);
  for (i=0; i<num_output_caps; i++)
  {
    s2 = gst_caps_get_structure(othercaps, i);
    inputFmt = gst_structure_get_string (s1, "format");

    GST_DEBUG_OBJECT (nvdsxfer, "InputFMT = %s \n\n", inputFmt);
    // Check for desired color format
    if ((strncmp(inputFmt, FORMAT_NV12, strlen(FORMAT_NV12)) == 0) ||
        (strncmp(inputFmt, FORMAT_I420, strlen(FORMAT_I420)) == 0) ||
        (strncmp(inputFmt, FORMAT_RGBA, strlen(FORMAT_RGBA)) == 0))
    {
      //Set these output caps
      gst_structure_get_int (s1, "width", &width);
      gst_structure_get_int (s1, "height", &height);

      /* otherwise the dimension of the output needs to be fixated */

      // Here change the width and height on output caps based on the input caps
      gst_structure_fixate_field_nearest_int(s2, "width", width);
      gst_structure_fixate_field_nearest_int(s2, "height", height);
      if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
      {
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);
      }
      gst_structure_set (s2, "format", G_TYPE_STRING, inputFmt, NULL);

      result = gst_caps_ref(othercaps);
      gst_caps_unref(othercaps);
      GST_DEBUG_OBJECT (nvdsxfer, "%s : Updated OTHERCAPS = %s \n\n", __func__, gst_caps_to_string(othercaps));
    }
    else {
      continue;
    }
  }
  return result;
}

static GstCaps *
gst_nvdsxfer_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (trans);
  GstCaps *ret = gst_caps_copy (caps);

  GST_DEBUG_OBJECT (nvdsxfer, "Inside Transform_Caps \ncaps = %s\n", gst_caps_to_string(caps));
  GST_DEBUG_OBJECT (nvdsxfer, "filter_caps = %s\n\n", gst_caps_to_string(filter));

  if (!ret)
    return nullptr;

  if (filter) {
    GstCaps *tmp = gst_caps_intersect (ret, filter);
    GST_DEBUG_OBJECT (nvdsxfer, "intersect_caps = %s\n\n", gst_caps_to_string(tmp));
    gst_caps_unref (ret);
    return tmp;
  }

  return ret;
}

static gboolean gst_nvdsxfer_accept_caps (GstBaseTransform * btrans, GstPadDirection direction,
        GstCaps * caps)
{
  gboolean ret = TRUE;
  GstNvXfer *space = NULL;
  GstCaps *allowed = NULL;
  GstCapsFeatures *features;

  space = GST_NVXFER (btrans);

  GST_DEBUG_OBJECT (btrans, "accept caps %" GST_PTR_FORMAT, caps);

  /* get all the formats we can handle on this pad */
  if (direction == GST_PAD_SINK)
    allowed = space->sinkcaps;
  else
    allowed = space->srccaps;

  if (!allowed) {
    GST_DEBUG_OBJECT (btrans, "failed to get allowed caps");
    goto no_transform_possible;
  }

  features = gst_caps_get_features (caps, 0);
  if (!gst_caps_features_contains (features, GST_CAPS_FEATURE_MEMORY_NVMM))
  {
      GST_DEBUG_OBJECT (btrans, "failed to find HW memory feature");
      goto no_transform_possible;
  }

  GST_DEBUG_OBJECT (btrans, "allowed caps %" GST_PTR_FORMAT, allowed);

  /* intersect with the requested format */
  ret = gst_caps_is_subset (caps, allowed);
  if (!ret) {
    goto no_transform_possible;
  }

  return ret;

  /* ERRORS */
no_transform_possible:
  {
    GST_DEBUG_OBJECT (btrans,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    ret = FALSE;
    return ret;
  }
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvdsxfer_class_init (GstNvXferClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  gstbasetransform_class->passthrough_on_same_caps = FALSE;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvdsxfer_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvdsxfer_get_property);

  gstbasetransform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsxfer_transform_caps);

  gstbasetransform_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsxfer_fixate_caps);
  gstbasetransform_class->accept_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsxfer_accept_caps);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvdsxfer_set_caps);
  gstbasetransform_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvdsxfer_sink_event);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdsxfer_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsxfer_stop);

  gstbasetransform_class->transform = GST_DEBUG_FUNCPTR (gst_nvdsxfer_transform);
  gstbasetransform_class->prepare_output_buffer =
      GST_DEBUG_FUNCPTR (gst_nvdsxfer_prepare_output_buffer);

  /* Install properties */
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
          "Maximum number of buffers in a batch",
          0, G_MAXUINT, DEFAULT_BATCH_SIZE,
          GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BUFFER_POOL_SIZE,
      g_param_spec_uint ("buffer-pool-size", "Buffer Pool Size",
          "Maximum number of buffers in muxer's internal pool",
          MIN_POOL_BUFFERS, MAX_POOL_BUFFERS, DEFAULT_BUFFER_POOL_SIZE,
          GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  PROP_NVBUF_MEMORY_TYPE_INSTALL(gobject_class);

  g_object_class_install_property (gobject_class, PROP_ENABLE_PEER_TO_DEVICE,
      g_param_spec_int ("p2p-gpu-id", "Set value of gpu-id to enable P2P access with",
          "Default P2P access between GPUs is disabled. Set P2P GPU ID to enable P2P access. ",
          -1, MAX_P2P_DEVICES, -1,
          GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdsxfer_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdsxfer_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvDsXfer plugin",
      "For Multi-GPU scenarios",
      "DS plugin for transferring data between GPUs",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvdsxfer_init (GstNvXfer * nvdsxfer)
{
  /* Initialize all property variables to default values */
  nvdsxfer->gpu_id = DEFAULT_GPU_ID;
  nvdsxfer->batch_size = DEFAULT_BATCH_SIZE;
  nvdsxfer->p2p_gpu_id = -1;
  nvdsxfer->can_peer_access = 0;
  nvdsxfer->buffer_pool_size = DEFAULT_BUFFER_POOL_SIZE;
  nvdsxfer->nvbuf_mem_type = NVBUF_MEM_DEFAULT;

  nvdsxfer->sinkcaps =
      gst_static_pad_template_get_caps (&gst_nvdsxfer_sink_template);
  nvdsxfer->srccaps =
      gst_static_pad_template_get_caps (&gst_nvdsxfer_src_template);

  /* This quark is required to identify NvDsMeta when iterating through
  * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvdsxfer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (object);
  switch (prop_id) {
    case PROP_GPU_DEVICE_ID:
      nvdsxfer->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      nvdsxfer->batch_size = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvdsxfer->nvbuf_mem_type = g_value_get_enum (value);
      break;
    case PROP_BUFFER_POOL_SIZE:
      nvdsxfer->buffer_pool_size = g_value_get_uint (value);
      break;
    case PROP_ENABLE_PEER_TO_DEVICE:
      nvdsxfer->p2p_gpu_id = g_value_get_int (value);
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
gst_nvdsxfer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (object);

  switch (prop_id) {
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvdsxfer->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, nvdsxfer->batch_size);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvdsxfer->nvbuf_mem_type);
      break;
    case PROP_BUFFER_POOL_SIZE:
      g_value_set_uint (value, nvdsxfer->buffer_pool_size);
      break;
    case PROP_ENABLE_PEER_TO_DEVICE:
      g_value_set_int (value, nvdsxfer->p2p_gpu_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the process thread
 */
static gboolean
gst_nvdsxfer_start (GstBaseTransform * btrans)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  std::string nvtx_str("GstNvXfer ");
  int current_device = -1;

  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  /* Retrive current GPU-id which is set for this thread */
  CHECK_CUDA_STATUS (cudaGetDevice(&current_device), "Unable to get cuda device");

  CHECK_CUDA_STATUS (cudaSetDevice (nvdsxfer->gpu_id), "Unable to set cuda device");

  nvdsxfer->nvtx_domain = nvtx_domain_ptr.release ();

  cudaStreamCreateWithFlags (&(nvdsxfer->cuda_xfer_stream), cudaStreamNonBlocking);

  // Enable P2P Access
  if ((nvdsxfer->p2p_gpu_id >= 0) && (nvdsxfer->gpu_id != (guint) nvdsxfer->p2p_gpu_id)) {
    // Enable P2P Access between two distinct devices
    CHECK_CUDA_STATUS (cudaDeviceCanAccessPeer(&nvdsxfer->can_peer_access, nvdsxfer->gpu_id,
      nvdsxfer->p2p_gpu_id), "CanAccessPeer Failed");

    g_print ("\n*** cudaDeviceCanAccessPeer (%d -> %d) = %s ***\n",
      nvdsxfer->gpu_id, nvdsxfer->p2p_gpu_id, nvdsxfer->can_peer_access? "YES":"NO");

    if (nvdsxfer->can_peer_access) {
      // Access between gpu_id ---> p2p_gpu_id GPUs
      CHECK_CUDA_STATUS (cudaDeviceEnablePeerAccess(nvdsxfer->p2p_gpu_id,0), "Unable To Set P2P access");

      // Access between p2p_gpu_id ---> gpu_id GPUs
      CHECK_CUDA_STATUS (cudaSetDevice (nvdsxfer->p2p_gpu_id), "Unable to set cuda device");
      CHECK_CUDA_STATUS (cudaDeviceEnablePeerAccess(nvdsxfer->gpu_id, 0), "Unable To Set P2P access");

      CHECK_CUDA_STATUS (cudaSetDevice (nvdsxfer->gpu_id), "Unable to set cuda device");
    } else {
      g_print ("*** P2P Access Not Possible\n\n");
      goto error;
    }
  }

  /* Set the previous GPU-id of this thread */
  CHECK_CUDA_STATUS (cudaSetDevice (current_device), "Unable to set cuda device");

  return TRUE;

error:
  /* Set the previous GPU-id of this thread */
  cudaSetDevice (current_device);

  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_nvdsxfer_stop (GstBaseTransform * btrans)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  int current_device = -1;

  /* Retrive current GPU-id which is set for this thread */
  CHECK_CUDA_STATUS (cudaGetDevice(&current_device), "Unable to get cuda device");

  CHECK_CUDA_STATUS (cudaSetDevice (nvdsxfer->gpu_id), "Unable to set cuda device");
  nvdsxfer->stop = TRUE;
  if (nvdsxfer->pool) {
      gst_buffer_pool_set_active (nvdsxfer->pool, FALSE);
      gst_object_unref(nvdsxfer->pool);
      nvdsxfer->pool = NULL;
  }

  if (nvdsxfer->cuda_xfer_stream) {
      cudaStreamDestroy(nvdsxfer->cuda_xfer_stream);
      nvdsxfer->cuda_xfer_stream = NULL;
  }

  GST_DEBUG_OBJECT (nvdsxfer, "ctx lib released \n");

  /* Set the previous GPU-id of this thread */
  CHECK_CUDA_STATUS (cudaSetDevice (current_device), "Unable to set cuda device");
  return TRUE;

error:
  /* Set the previous GPU-id of this thread */
  cudaSetDevice (current_device);
  return FALSE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvdsxfer_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstQuery *bsquery = NULL;
  guint batch_size = 0;
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  GstStructure *config = NULL;
  int current_device = -1;

  /* Retrive current GPU-id which is set for this thread */
  CHECK_CUDA_STATUS (cudaGetDevice(&current_device), "Unable to get cuda device");

  if (nvdsxfer->batch_size == 1)
  {
    bsquery = gst_nvquery_batch_size_new ();
    if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), bsquery))
    {
      gst_nvquery_batch_size_parse (bsquery, &batch_size);
      nvdsxfer->batch_size = batch_size;
    }
    gst_query_unref (bsquery);
  }

  /* Save the input & output video information, since this will be required later. */
  gst_video_info_from_caps (&nvdsxfer->in_video_info, incaps);
  gst_video_info_from_caps (&nvdsxfer->out_video_info, outcaps);

  CHECK_CUDA_STATUS (cudaSetDevice (nvdsxfer->gpu_id), "Unable to set cuda device");

  // TODO: Handle buffer pool incase of different gpu-ids, else make this run under bypass mode
  // Explore more on query to get the GPU-ID from upstream buffer allocator

  // Destroy previously allocated pool and create new pool
  // TODO: Check if the width, height and color format is same then use the same pool
  if (nvdsxfer->pool) {
      gst_buffer_pool_set_active (nvdsxfer->pool, FALSE);
      gst_object_unref(nvdsxfer->pool);
      nvdsxfer->pool = NULL;
  }

  if (!nvdsxfer->pool)
  {
    nvdsxfer->pool = gst_nvds_buffer_pool_new ();

    config = gst_buffer_pool_get_config (nvdsxfer->pool);

    GST_DEBUG_OBJECT (nvdsxfer, "OutputCaps = %" GST_PTR_FORMAT "\n", outcaps);
    gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), nvdsxfer->buffer_pool_size, nvdsxfer->buffer_pool_size);

    // contiguous-alloc default set for buffer pool config
    gst_structure_set (config,
                       "memtype", G_TYPE_UINT, nvdsxfer->nvbuf_mem_type,
                       "gpu-id", G_TYPE_UINT, nvdsxfer->gpu_id,
                       "batch-size", G_TYPE_UINT, nvdsxfer->batch_size,
                       "contiguous-alloc", G_TYPE_BOOLEAN, TRUE, NULL);

    GST_DEBUG_OBJECT (nvdsxfer, "%s Allocating Buffers in NVM Buffer Pool for Batch-Size=%d\n",
        __func__, nvdsxfer->batch_size);

    /* set config for the created buffer pool */
    if (!gst_buffer_pool_set_config (nvdsxfer->pool, config)) {
      GST_WARNING_OBJECT (nvdsxfer, "Bufferpool configuration failed");
      goto error;
    }

    gboolean is_active = gst_buffer_pool_set_active (nvdsxfer->pool, TRUE);
    if (!is_active) {
      GST_WARNING_OBJECT (nvdsxfer, "Failed to allocate the buffers inside the output pool");
      goto error;
    } else {
      GST_WARNING_OBJECT (nvdsxfer, "Output buffer pool (%p) successfully created with %d buffers",
       nvdsxfer->pool, nvdsxfer->buffer_pool_size);
    }
  }

  /* Set the previous GPU-id of this thread */
  CHECK_CUDA_STATUS (cudaSetDevice (current_device), "Unable to set cuda device");
  return TRUE;
error:

  /* Set the previous GPU-id of this thread */
  cudaSetDevice (current_device);
  return FALSE;
}

static gboolean
gst_nvdsxfer_sink_event (GstBaseTransform * btrans, GstEvent *event)
{
    return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event(btrans, event);
}

static GstFlowReturn
gst_nvdsxfer_prepare_output_buffer (GstBaseTransform * btrans,
    GstBuffer * inbuf, GstBuffer ** outbuf)
{
  GstBuffer *gstOutBuf = NULL;
  GstFlowReturn result = GST_FLOW_OK;
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);

  nvdsxfer->frame_num++;

  result = gst_buffer_pool_acquire_buffer (nvdsxfer->pool, &gstOutBuf, NULL);
  GST_DEBUG_OBJECT (nvdsxfer, "%s : Frame=%d Gst-OutBuf=%p\n", __func__,
      nvdsxfer->frame_num, gstOutBuf);

  if (result != GST_FLOW_OK) {
    GST_ERROR_OBJECT (nvdsxfer,
        "gst_nvdsxfer_prepare_output_buffer failed");
    return result;
  }

  *outbuf = gstOutBuf;
  return result;
}

static GstFlowReturn
gst_nvdsxfer_copy_metadata(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  gpointer state = NULL;
  GstMeta *gst_meta = NULL;
  NvDsMeta *dsmeta = NULL;
  NvDsBatchMeta *batch_meta = NULL;

  GstMapInfo outmap = GST_MAP_INFO_INIT;

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
  {
    g_print ("%s output buf map failed\n", __func__);
    return GST_FLOW_ERROR;
  }

  NvBufSurface *dstSurf = (NvBufSurface *)outmap.data;
  gst_buffer_unmap (outbuf, &outmap);

  // Required in the case of tiler
  if (!gst_buffer_copy_into (outbuf, inbuf, GST_BUFFER_COPY_META, 0, -1)) {
    GST_DEBUG_OBJECT (nvdsxfer, "Buffer metadata copy failed \n");
  }

  GST_BUFFER_PTS (outbuf) = GST_BUFFER_PTS (inbuf);

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
    dstSurf->numFilled = 1;
  } else {
    dstSurf->numFilled = batch_meta->num_frames_in_batch;
  }

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_nvdsxfer_copy_buffer(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  GstFlowReturn result = GST_FLOW_OK;
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  GstMapInfo outmap = GST_MAP_INFO_INIT;

  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ))
  {
    GST_DEBUG_OBJECT (nvdsxfer, "%s Input buffer map failed \n", __func__);
    return GST_FLOW_ERROR;
  }
  NvBufSurface *srcSurf = (NvBufSurface *)inmap.data;
  gst_buffer_unmap (inbuf, &inmap);

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
  {
    GST_DEBUG_OBJECT (nvdsxfer, "%s output buf map failed\n", __func__);
    return GST_FLOW_ERROR;
  }
  NvBufSurface *dstSurf = (NvBufSurface *)outmap.data;
  gst_buffer_unmap (outbuf, &outmap);

  // Check Inputbuffer and output buffer batch-size, may not be same
  if (srcSurf->batchSize != dstSurf->batchSize) {
    GST_DEBUG_OBJECT (nvdsxfer, "Input (%d) and Output (%d) buffer batchsize mismatch \n\n", srcSurf->batchSize, dstSurf->batchSize);
  }

  // Copy Input Buffer Memory into Output Buffer Memory
  {
    cudaError_t err;
    unsigned int i = 0;
#if 0
    // TODO : Check for contiguous memory transfer
    if (srcSurf->isContiguous == 1 && dstSurf->isContiguous == 1) {
      NvBufSurfaceParams *srcParams = &srcSurf->surfaceList[0];
      NvBufSurfaceParams *dstParams = &dstSurf->surfaceList[0];
      if (srcParams->width != dstParams->width ||
              srcParams->height != dstParams->height) {
            printf ("nvdsxfer: Src and Dst NvBufSurface size mismatch\n");
            return GST_FLOW_ERROR;
      }
      unsigned long int data_size = srcSurf->numFilled * srcParams->dataSize;
      //printf("nvdsxfer: Frame : %d numFilled = %d isContiguous = %d dataSize = %ld\n",
      // nvdsxfer->frame_num, srcSurf->numFilled, srcSurf->isContiguous, data_size);
      err = cudaMemcpyAsync (dstParams->dataPtr, srcParams->dataPtr,
              data_size, cudaMemcpyDeviceToDevice, nvdsxfer->cuda_xfer_stream);
    }
    else
#endif
    {
      for (i = 0; i < srcSurf->numFilled; i++) {
        NvBufSurfaceParams *srcParams = &srcSurf->surfaceList[i];
        NvBufSurfaceParams *dstParams = &dstSurf->surfaceList[i];
        if (srcParams->width != dstParams->width ||
              srcParams->height != dstParams->height) {
            printf ("nvdsxfer: Src and Dst NvBufSurface size mismatch\n");
            return GST_FLOW_ERROR;
        }

        if (nvdsxfer->can_peer_access) {
          GST_DEBUG_OBJECT(nvdsxfer, "Calling cudaMemcpyPeerAsync for frame # %d surface # %d", nvdsxfer->frame_num, i);
          err = cudaMemcpyPeerAsync (dstParams->dataPtr,
            nvdsxfer->p2p_gpu_id, srcParams->dataPtr, nvdsxfer->gpu_id,
            dstParams->dataSize, nvdsxfer->cuda_xfer_stream);
        }
        else {
          GST_DEBUG_OBJECT(nvdsxfer, "Calling cudaMemcpyAsync for frame # %d surface # %d", nvdsxfer->frame_num, i);
          err = cudaMemcpyAsync (dstParams->dataPtr,
              srcParams->dataPtr, dstParams->dataSize,
              cudaMemcpyDeviceToDevice, nvdsxfer->cuda_xfer_stream);
        }

        if (err != cudaSuccess) {
              printf ("nvdsxfer: %s: failed\n", nvdsxfer->can_peer_access ? "cudaMemcpyPeerAsync" : "cudaMemcpyAsync");
              return GST_FLOW_ERROR;
        }
      }
    }
    cudaStreamSynchronize(nvdsxfer->cuda_xfer_stream);
  }
  return result;
}

/**
 * Called when the plugin works in non-passthough mode
 */
static GstFlowReturn
gst_nvdsxfer_transform(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
  GstNvXfer *nvdsxfer = GST_NVXFER (btrans);
  GstFlowReturn result = GST_FLOW_OK;
  int current_device = -1;

  /* Retrive current GPU-id which is set for this thread */
  CHECK_CUDA_STATUS (cudaGetDevice(&current_device), "Unable to get cuda device");

  CHECK_CUDA_STATUS (cudaSetDevice (nvdsxfer->gpu_id), "Unable to set cuda device");

  START_PROFILE

  // Copy buffer memory from inbuf to outbuf
  result = gst_nvdsxfer_copy_buffer(btrans, inbuf, outbuf);
  if (result != GST_FLOW_OK)
    goto error;

  // Copy metadata
  result = gst_nvdsxfer_copy_metadata(btrans, inbuf, outbuf);
  if (result != GST_FLOW_OK)
    goto error;

  STOP_PROFILE(nvdsxfer, "Buffer+Meta Copy ")

  /* Set the previous GPU-id of this thread */
  CHECK_CUDA_STATUS (cudaSetDevice (current_device), "Unable to set cuda device");
  return GST_FLOW_OK;

error:
  /* Set the previous GPU-id of this thread */
  cudaSetDevice (current_device);
  return GST_FLOW_ERROR;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdsxfer_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsxfer_debug, "nvdsxfer", 0,
      "nvdsxfer plugin");

  return gst_element_register (plugin, "nvdsxfer", GST_RANK_PRIMARY,
      GST_TYPE_NVDSXFER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_xfer,
    DESCRIPTION, nvdsxfer_plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE,
    URL)
