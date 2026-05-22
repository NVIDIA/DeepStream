/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gstdsexample.h"
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC (gst_dsexample_debug);
#define GST_CAT_DEFAULT gst_dsexample_debug
#define USE_EGLIMAGE 1
/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_BATCH_SIZE,
  PROP_BLUR_OBJECTS,
  PROP_GPU_DEVICE_ID
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)  \
  ({ int _errtype=0;\
   do {  \
    if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) && \
        (surface->gpuId != object->gpu_id))  { \
    GST_ELEMENT_ERROR (object, RESOURCE, FAILED, \
        ("Input surface gpu-id doesnt match with configured gpu-id for element," \
         " please allocate input using unified memory, or use same gpu-ids"),\
        ("surface-gpu-id=%d,%s-gpu-id=%d",surface->gpuId,GST_ELEMENT_NAME(object),\
         object->gpu_id)); \
    _errtype = 1;\
    } \
    } while(0); \
    _errtype; \
  })


/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_BLUR_OBJECTS FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 1
#define MIN_INPUT_OBJECT_HEIGHT 1

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsexample_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsexample_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsexample_parent_class parent_class
G_DEFINE_TYPE (GstDsExample, gst_dsexample, GST_TYPE_BASE_TRANSFORM);

static void gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dsexample_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsexample_start (GstBaseTransform * btrans);
static gboolean gst_dsexample_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_dsexample_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);

static void
attach_metadata_full_frame (GstDsExample * dsexample, NvDsFrameMeta *frame_meta,
    gdouble scale_ratio, DsExampleOutput * output, guint batch_id);
static void attach_metadata_object (GstDsExample * dsexample,
    NvDsObjectMeta * obj_meta, DsExampleOutput * output);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsexample_class_init (GstDsExampleClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  /* Indicates we want to use DS buf api */
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsexample_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsexample_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsexample_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsexample_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsexample_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dsexample_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width",
          "Processing Width",
          "Width of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_WIDTH, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height",
          "Processing Height",
          "Height of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_HEIGHT, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame",
          "Full frame",
          "Enable to process full frame or disable to process objects detected"
          "by primary detector", DEFAULT_PROCESS_FULL_FRAME, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BLUR_OBJECTS,
      g_param_spec_boolean ("blur-objects",
          "Blur Objects",
          "Enable to blur the objects detected in full-frame=0 mode"
          "by primary detector", DEFAULT_BLUR_OBJECTS, (GParamFlags)
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
          "Maximum batch size for processing",
          1, NVDSEXAMPLE_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));
  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsexample_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsexample_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsExample plugin",
      "DsExample Plugin",
      "Process a 3rdparty example algorithm on objects / full frame",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dsexample_init (GstDsExample * dsexample)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsexample);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dsexample->unique_id = DEFAULT_UNIQUE_ID;
  dsexample->processing_width = DEFAULT_PROCESSING_WIDTH;
  dsexample->processing_height = DEFAULT_PROCESSING_HEIGHT;
  dsexample->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  dsexample->blur_objects = DEFAULT_BLUR_OBJECTS;
  dsexample->gpu_id = DEFAULT_GPU_ID;
  dsexample->max_batch_size = DEFAULT_BATCH_SIZE;

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dsexample->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      dsexample->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      dsexample->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      dsexample->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_BLUR_OBJECTS:
      dsexample->blur_objects = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dsexample->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      dsexample->max_batch_size = g_value_get_uint (value);
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
gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dsexample->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, dsexample->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, dsexample->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, dsexample->process_full_frame);
      break;
    case PROP_BLUR_OBJECTS:
      g_value_set_boolean (value, dsexample->blur_objects);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dsexample->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, dsexample->max_batch_size);
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
gst_dsexample_start (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  NvBufSurfaceCreateParams create_params = { 0 };
  DsExampleInitParams init_params =
      { dsexample->processing_width, dsexample->processing_height,
    dsexample->process_full_frame
  };
  NvBufSurfaceDeviceInfo dev_info{};

  /* Algorithm specific initializations and resource allocation. */
  dsexample->dsexamplelib_ctx = DsExampleCtxInit (&init_params);

  GST_DEBUG_OBJECT (dsexample, "ctx lib %p \n", dsexample->dsexamplelib_ctx);

  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  dsexample->is_integrated = 0;
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      dsexample->is_integrated = 1;
    }
  }

  GST_DEBUG_OBJECT (dsexample, "Setting batch-size %d \n",
      dsexample->max_batch_size);

  if (dsexample->process_full_frame && dsexample->blur_objects) {
    GST_ERROR ("Error: does not support blurring while processing full frame");
    goto error;
  }

#ifndef WITH_OPENCV
  if (dsexample->blur_objects) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("OpenCV has been deprecated, hence object blurring will not work."
          "Enable OpenCV compilation in gst-dsexample Makefile by setting 'WITH_OPENCV:=1"), (NULL));
    goto error;
  }
#endif

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsexample->cuda_stream),
      "Could not create cuda stream");

  if (dsexample->inter_buf)
    NvBufSurfaceDestroy (dsexample->inter_buf);
  dsexample->inter_buf = NULL;

  /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
   * required. Can be skipped if custom algorithm can work directly on NV12/RGBA. */
  create_params.gpuId  = dsexample->gpu_id;
  create_params.width  = dsexample->processing_width;
  create_params.height = dsexample->processing_height;
  create_params.size = 0;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  create_params.layout = NVBUF_LAYOUT_PITCH;

  if(dsexample->is_integrated) {
    create_params.memType = NVBUF_MEM_DEFAULT;
  }
  else {
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
  }

  if (NvBufSurfaceCreate (&dsexample->inter_buf, 1,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer for dsexample");
    goto error;
  }

  /* Create host memory for storing converted/scaled interleaved RGB data */
  CHECK_CUDA_STATUS (cudaMallocHost (&dsexample->host_rgb_buf,
          dsexample->processing_width * dsexample->processing_height *
          RGB_BYTES_PER_PIXEL), "Could not allocate cuda host buffer");

  GST_DEBUG_OBJECT (dsexample, "allocated cuda buffer %p \n",
      dsexample->host_rgb_buf);

#ifdef WITH_OPENCV
  /* CV Mat containing interleaved RGB data. This call does not allocate memory.
   * It uses host_rgb_buf as data. */
  dsexample->cvmat =
      new cv::Mat (dsexample->processing_height, dsexample->processing_width,
      CV_8UC3, dsexample->host_rgb_buf,
      dsexample->processing_width * RGB_BYTES_PER_PIXEL);

  if (!dsexample->cvmat)
    goto error;

  GST_DEBUG_OBJECT (dsexample, "created CV Mat\n");
#endif

  /* Set the NvBufSurfTransform config parameters. */
  dsexample->transform_config_params.compute_mode =
      NvBufSurfTransformCompute_Default;
  dsexample->transform_config_params.gpu_id = dsexample->gpu_id;

  return TRUE;
error:
  if (dsexample->host_rgb_buf) {
    cudaFreeHost (dsexample->host_rgb_buf);
    dsexample->host_rgb_buf = NULL;
  }

  if (dsexample->cuda_stream) {
    cudaStreamDestroy (dsexample->cuda_stream);
    dsexample->cuda_stream = NULL;
  }
  if (dsexample->dsexamplelib_ctx)
    DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_dsexample_stop (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);

  if (dsexample->inter_buf)
    NvBufSurfaceDestroy(dsexample->inter_buf);
  dsexample->inter_buf = NULL;

  if (dsexample->cuda_stream)
    cudaStreamDestroy (dsexample->cuda_stream);
  dsexample->cuda_stream = NULL;

#ifdef WITH_OPENCV
  delete dsexample->cvmat;
  dsexample->cvmat = NULL;
#endif

  if (dsexample->host_rgb_buf) {
    cudaFreeHost (dsexample->host_rgb_buf);
    dsexample->host_rgb_buf = NULL;
  }

  GST_DEBUG_OBJECT (dsexample, "deleted CV Mat \n");

  /* Deinit the algorithm library */
  DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
  dsexample->dsexamplelib_ctx = NULL;

  GST_DEBUG_OBJECT (dsexample, "ctx lib released \n");

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dsexample_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dsexample->video_info, incaps);

  if (dsexample->blur_objects && !dsexample->process_full_frame) {
    /* requires RGBA format for blurring the objects in opencv */
     if (dsexample->video_info.finfo->format != GST_VIDEO_FORMAT_RGBA) {
      GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("input format should be RGBA when using blur-objects property"), (NULL));
      goto error;
      }
  }

  return TRUE;

error:
  return FALSE;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio. Remove the padding required by hardware and convert from RGBA to RGB
 * using openCV. These steps can be skipped if the algorithm can work with
 * padded data and/or can work with RGBA.
 */
static GstFlowReturn
get_converted_mat (GstDsExample * dsexample, NvBufSurface *input_buf, gint idx,
    NvOSD_RectParams * crop_rect_params, gdouble & ratio, gint input_width,
    gint input_height)
{
  NvBufSurfTransform_Error err;
  NvBufSurfTransformConfigParams transform_config_params;
  NvBufSurfTransformParams transform_params;
  NvBufSurfTransformRect src_rect;
  NvBufSurfTransformRect dst_rect;
  NvBufSurface ip_surf;
#ifdef WITH_OPENCV
  cv::Mat in_mat;
#endif
  ip_surf = *input_buf;

  ip_surf.numFilled = ip_surf.batchSize = 1;
  ip_surf.surfaceList = &(input_buf->surfaceList[idx]);

  gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
  gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
  gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
  gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);

  /* Maintain aspect ratio */
  double hdest = dsexample->processing_width * src_height / (double) src_width;
  double wdest = dsexample->processing_height * src_width / (double) src_height;
  guint dest_width, dest_height;

  if (hdest <= dsexample->processing_height) {
    dest_width = dsexample->processing_width;
    dest_height = hdest;
  } else {
    dest_width = wdest;
    dest_height = dsexample->processing_height;
  }

  /* Configure transform session parameters for the transformation */
  transform_config_params.compute_mode = dsexample->transform_config_params.compute_mode;
  transform_config_params.gpu_id = dsexample->gpu_id;
  transform_config_params.cuda_stream = dsexample->cuda_stream;

  /* Set the transform session parameters for the conversions executed in this
   * thread. */
  err = NvBufSurfTransformSetSessionParams (&transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
    goto error;
  }

  /* Calculate scaling ratio while maintaining aspect ratio */
  ratio = MIN (1.0 * dest_width/ src_width, 1.0 * dest_height / src_height);

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero",__func__), (NULL));
    goto error;
  }

#ifdef __aarch64__
  if (ratio <= 1.0 / 16 || ratio >= 16.0) {
    /* Currently cannot scale by ratio > 16 or < 1/16 for Jetson */
    goto error;
  }
#endif
  /* Set the transform ROIs for source and destination */
  src_rect = {(guint)src_top, (guint)src_left, (guint)src_width, (guint)src_height};
  dst_rect = {0, 0, (guint)dest_width, (guint)dest_height};

  /* Set the transform parameters */
  transform_params.src_rect = &src_rect;
  transform_params.dst_rect = &dst_rect;
  transform_params.transform_flag =
    NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  transform_params.transform_filter = NvBufSurfTransformInter_Default;

  /* Memset the memory */
  NvBufSurfaceMemSet (dsexample->inter_buf, 0, 0, 0);

  GST_DEBUG_OBJECT (dsexample, "Scaling and converting input buffer\n");

  /* Transformation scaling+format conversion if any. */
  err = NvBufSurfTransform (&ip_surf, dsexample->inter_buf, &transform_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer", err),
        (NULL));
    goto error;
  }
  /* Map the buffer so that it can be accessed by CPU */
  if (NvBufSurfaceMap (dsexample->inter_buf, 0, 0, NVBUF_MAP_READ) != 0){
    goto error;
  }
  if(dsexample->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    /* Cache the mapped data for CPU access */
    NvBufSurfaceSyncForCpu (dsexample->inter_buf, 0, 0);
  }

#ifdef WITH_OPENCV
  /* Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
   * algorithm can handle padded RGBA data. */
  in_mat =
      cv::Mat (dsexample->processing_height, dsexample->processing_width,
      CV_8UC4, dsexample->inter_buf->surfaceList[0].mappedAddr.addr[0],
      dsexample->inter_buf->surfaceList[0].pitch);

#if (CV_MAJOR_VERSION >= 4)
  cv::cvtColor (in_mat, *dsexample->cvmat, cv::COLOR_RGBA2BGR);
#else
  cv::cvtColor (in_mat, *dsexample->cvmat, CV_RGBA2BGR);
#endif
#endif

    if (NvBufSurfaceUnMap (dsexample->inter_buf, 0, 0)){
      goto error;
    }

  if(dsexample->is_integrated) {
#ifdef __aarch64__
    /* To use the converted buffer in CUDA, create an EGLImage and then use
    * CUDA-EGL interop APIs */
    if (USE_EGLIMAGE) {
      if (NvBufSurfaceMapEglImage (dsexample->inter_buf, 0) !=0 ) {
        goto error;
      }

      /* dsexample->inter_buf->surfaceList[0].mappedAddr.eglImage
      * Use interop APIs cuGraphicsEGLRegisterImage and
      * cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA */

      /* Destroy the EGLImage */
      NvBufSurfaceUnMapEglImage (dsexample->inter_buf, 0);
    }
#endif
  }

  /* We will first convert only the Region of Interest (the entire frame or the
   * object bounding box) to RGB and then scale the converted RGB frame to
   * processing resolution. */
  return GST_FLOW_OK;

error:
  return GST_FLOW_ERROR;
}

#ifdef WITH_OPENCV
/*
 * Blur the detected objects when processing in object mode (full-frame=0)
 */
static GstFlowReturn
blur_objects (GstDsExample * dsexample, gint idx,
    NvOSD_RectParams * crop_rect_params, cv::Mat in_mat)
{
  cv::Rect crop_rect;

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero",__func__), (NULL));
    return GST_FLOW_ERROR;
  }

/* rectangle for cropped objects */
  crop_rect = cv::Rect (crop_rect_params->left, crop_rect_params->top,
  crop_rect_params->width, crop_rect_params->height);

/* apply gaussian blur to the detected objects */
  // Check if width or height is less than 1 pixel, which would cause a crash
  if(crop_rect.width < 1 || crop_rect.height < 1) {
    g_print("WARNING: Skipping GaussianBlur due to invalid rect dimensions: %dx%d\n",
      crop_rect.width, crop_rect.height);
    return GST_FLOW_OK;
  }

  GaussianBlur(in_mat(crop_rect), in_mat(crop_rect), cv::Size(15,15), 4);

  return GST_FLOW_OK;
}
#endif

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsexample_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  gdouble scale_ratio = 1.0;
  DsExampleOutput *output;

  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList * l_frame = NULL;
  guint i = 0;

  dsexample->frame_num++;
  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    goto error;
  }

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));
  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (dsexample,
      "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
      dsexample->frame_num, surface);

  if (CHECK_NVDS_MEMORY_AND_GPUID (dsexample, surface))
    goto error;

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  if (dsexample->process_full_frame) {
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      NvOSD_RectParams rect_params;

      /* Scale the entire frame to processing resolution */
      rect_params.left = 0;
      rect_params.top = 0;
      rect_params.width = dsexample->video_info.width;
      rect_params.height = dsexample->video_info.height;

      /* Scale and convert the frame */
      if (get_converted_mat (dsexample, surface, i, &rect_params,
            scale_ratio, dsexample->video_info.width,
            dsexample->video_info.height) != GST_FLOW_OK) {
        goto error;
      }

      /* Process to get the output */
#ifdef WITH_OPENCV
      output =
          DsExampleProcess (dsexample->dsexamplelib_ctx,
          dsexample->cvmat->data);
#else
      output =
          DsExampleProcess (dsexample->dsexamplelib_ctx,
          (unsigned char *)dsexample->inter_buf->surfaceList[0].mappedAddr.addr[0]);
#endif
      /* Attach the metadata for the full frame */
      attach_metadata_full_frame (dsexample, frame_meta, scale_ratio, output, i);
      i++;
      free (output);
    }

  } else {
    /* Using object crops as input to the algorithm. The objects are detected by
     * the primary detector */
    NvDsMetaList * l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;

    if(!dsexample->is_integrated) {
      if (dsexample->blur_objects) {
        if (!(surface->memType == NVBUF_MEM_CUDA_UNIFIED || surface->memType == NVBUF_MEM_CUDA_PINNED)){
          GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
              ("%s:need NVBUF_MEM_CUDA_UNIFIED or NVBUF_MEM_CUDA_PINNED memory for opencv blurring",__func__), (NULL));
          return GST_FLOW_ERROR;
        }
      }
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);

#ifdef WITH_OPENCV
      cv::Mat in_mat;

      if (dsexample->blur_objects) {
        /* Map the buffer so that it can be accessed by CPU */
        if (surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0] == NULL){
          if (NvBufSurfaceMap (surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE) != 0){
            GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
                ("%s:buffer map to be accessed by CPU failed", __func__), (NULL));
            return GST_FLOW_ERROR;
          }
        }

        /* Cache the mapped data for CPU access */
        if(dsexample->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY)
          NvBufSurfaceSyncForCpu (surface, frame_meta->batch_id, 0);

        in_mat =
            cv::Mat (surface->surfaceList[frame_meta->batch_id].planeParams.height[0],
            surface->surfaceList[frame_meta->batch_id].planeParams.width[0], CV_8UC4,
            surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
            surface->surfaceList[frame_meta->batch_id].planeParams.pitch[0]);
      }
#endif

      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next)
      {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);

        if (dsexample->blur_objects) {
          /* gaussian blur the detected objects using opencv */
#ifdef WITH_OPENCV
          if (blur_objects (dsexample, frame_meta->batch_id,
            &obj_meta->rect_params, in_mat) != GST_FLOW_OK) {
          /* Error in blurring, skip processing on object. */
            GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
            ("blurring the object failed"), (NULL));
            if (NvBufSurfaceUnMap (surface, frame_meta->batch_id, 0)){
              GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
                ("%s:buffer unmap to be accessed by CPU failed", __func__), (NULL));
            }
            return GST_FLOW_ERROR;
          }
          continue;
#else
          GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("OpenCV has been deprecated, hence object blurring will not work."
          "Enable OpenCV compilation in gst-dsexample Makefile by setting 'WITH_OPENCV:=1"), (NULL));
          return GST_FLOW_ERROR;
#endif
        }

        /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT */
        if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
            obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
          continue;

        /* Extra check for Jetson devices as default compute mode on Jetson is VIC which supports min 16x16 */
        if (dsexample->is_integrated) {
          if (dsexample->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC ||
              dsexample->transform_config_params.compute_mode == NvBufSurfTransformCompute_Default)
          {
            if (obj_meta->rect_params.width < 16 ||
                obj_meta->rect_params.height < 16)
              continue;
          }
        }

        /* Crop and scale the object */
        if (get_converted_mat (dsexample,
              surface, frame_meta->batch_id, &obj_meta->rect_params,
              scale_ratio, dsexample->video_info.width,
              dsexample->video_info.height) != GST_FLOW_OK) {
          /* Error in conversion, skip processing on object. */
          continue;
        }

#ifdef WITH_OPENCV
        /* Process the object crop to obtain label */
        output = DsExampleProcess (dsexample->dsexamplelib_ctx,
            dsexample->cvmat->data);
#else
        /* Process the object crop to obtain label */
        output = DsExampleProcess (dsexample->dsexamplelib_ctx,
            (unsigned char *)dsexample->inter_buf->surfaceList[0].mappedAddr.addr[0]);
#endif

        /* Attach labels for the object */
        attach_metadata_object (dsexample, obj_meta, output);

        free (output);
      }

      if (dsexample->blur_objects) {
      /* Cache the mapped data for device access */
        if(dsexample->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY) 
          NvBufSurfaceSyncForDevice (surface, frame_meta->batch_id, 0);

#ifdef WITH_OPENCV
#ifdef DSEXAMPLE_DEBUG
        /* Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
        * algorithm can handle padded RGBA data. */
#if (CV_MAJOR_VERSION >= 4)
        cv::cvtColor (in_mat, *dsexample->cvmat, cv::COLOR_RGBA2BGR);
#else
        cv::cvtColor (in_mat, *dsexample->cvmat, CV_RGBA2BGR);
#endif
        /* used to dump the converted mat to files for debug */
        static guint cnt = 0;
        cv::imwrite("out_" + std::to_string (cnt) + ".jpeg", *dsexample->cvmat);
        cnt++;
#endif
#endif
      }
    }
  }
  flow_ret = GST_FLOW_OK;

error:

  nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Attach metadata for the full frame. We will be adding a new metadata.
 */
static void
attach_metadata_full_frame (GstDsExample * dsexample, NvDsFrameMeta *frame_meta,
    gdouble scale_ratio, DsExampleOutput * output, guint batch_id)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  NvDsObjectMeta *object_meta = NULL;
  static gchar font_name[] = "Serif";
  GST_DEBUG_OBJECT (dsexample, "Attaching metadata %d\n", output->numObjects);

  for (gint i = 0; i < output->numObjects; i++) {
    DsExampleObject *obj = &output->object[i];
    object_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
    NvOSD_RectParams & rect_params = object_meta->rect_params;
    NvOSD_TextParams & text_params = object_meta->text_params;

    /* Assign bounding box coordinates */
    rect_params.left = obj->left;
    rect_params.top = obj->top;
    rect_params.width = obj->width;
    rect_params.height = obj->height;

    /* Semi-transparent yellow background */
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    /* Red border of width 6 */
    rect_params.border_width = 3;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    /* Scale the bounding boxes proportionally based on how the object/frame was
     * scaled during input */
    rect_params.left /= scale_ratio;
    rect_params.top /= scale_ratio;
    rect_params.width /= scale_ratio;
    rect_params.height /= scale_ratio;
    GST_DEBUG_OBJECT (dsexample, "Attaching rect%d of batch%u"
        "  left->%f top->%f width->%f"
        " height->%f label->%s\n", i, batch_id, rect_params.left,
        rect_params.top, rect_params.width, rect_params.height, obj->label);

    object_meta->object_id = UNTRACKED_OBJECT_ID;
    g_strlcpy (object_meta->obj_label, obj->label, MAX_LABEL_SIZE);
    /* display_text required heap allocated memory */
    text_params.display_text = g_strdup (obj->label);
    /* Display text above the left top corner of the object */
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    /* Set black background for the text */
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    /* Font face, size and color */
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};

    nvds_add_obj_meta_to_frame(frame_meta, object_meta, NULL);
    frame_meta->bInferDone = TRUE;
  }
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void
attach_metadata_object (GstDsExample * dsexample, NvDsObjectMeta * obj_meta,
    DsExampleOutput * output)
{
  if (output->numObjects == 0)
    return;
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

  NvDsClassifierMeta *classifier_meta =
    nvds_acquire_classifier_meta_from_pool (batch_meta);

  classifier_meta->unique_component_id = dsexample->unique_id;

  NvDsLabelInfo *label_info =
    nvds_acquire_label_info_meta_from_pool (batch_meta);
  g_strlcpy (label_info->result_label, output->object[0].label, MAX_LABEL_SIZE);
  nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
  nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);

  nvds_acquire_meta_lock (batch_meta);
  NvOSD_TextParams & text_params = obj_meta->text_params;
  NvOSD_RectParams & rect_params = obj_meta->rect_params;

  /* Below code to display the result */
  /* Set black background for the text
   * display_text required heap allocated memory */
  if (text_params.display_text) {
    gchar *conc_string = g_strconcat (text_params.display_text, " ",
        output->object[0].label, NULL);
    g_free (text_params.display_text);
    text_params.display_text = conc_string;
  } else {
    /* Display text above the left top corner of the object */
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    text_params.display_text = g_strdup (output->object[0].label);
    /* Font face, size and color */
    text_params.font_params.font_name = (char *)"Serif";
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    /* Set black background for the text */
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
  }
  nvds_release_meta_lock (batch_meta);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsexample_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsexample_debug, "dsexample", 0,
      "dsexample plugin");

  return gst_element_register (plugin, "dsexample", GST_RANK_PRIMARY,
      GST_TYPE_DSEXAMPLE);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsexample,
    DESCRIPTION, dsexample_plugin_init, "9.0", LICENSE, BINARY_PACKAGE, URL)
