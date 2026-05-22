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

/**
 * There are two threads in the optimized code. input thread and Processing thread.
 * The pre-procesing as required by the algorithm like scaling and color
 * conversion of data is done in input thread. This is done using NvBufSurfTransform's
 * batch conversion APIs to improve performance. The processing of data using custom
 * algorithm and parsing the output and  metadata attachment is done in separate processing
 * thread.
 *
 * There are two queues used for buffering and transferring data between thread:
 * Process_queue and buf_queue Process_queue is used to send filled batched data to
 * process thread and buf_queue is used to get return empty processed buffers from
 * process thread to input thread.  Two buffers are used in a ping pong manner between
 * the two threads for parallel processing.
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>

#include "gstdsexample_optimized.h"

#include <sys/time.h>
#include <condition_variable>
#include <mutex>
#include <thread>

GST_DEBUG_CATEGORY_STATIC (gst_dsexample_debug);
#define GST_CAT_DEFAULT gst_dsexample_debug
#define USE_EGLIMAGE 1

#ifdef WITH_OPENCV
//enable to write transformed cvmat to files
//#define DSEXAMPLE_DEBUG
#ifdef DSEXAMPLE_DEBUG
#include "opencv2/imgcodecs.hpp"
#endif
#endif

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

static GstFlowReturn
gst_dsexample_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_dsexample_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);

static void
attach_metadata_full_frame (GstDsExample * dsexample,
    NvDsFrameMeta * frame_meta, gdouble scale_ratio, DsExampleOutput * output,
    guint batch_id);
static void attach_metadata_object (GstDsExample * dsexample,
    NvDsObjectMeta * obj_meta, DsExampleOutput * output);

static gpointer gst_dsexample_output_loop (gpointer data);

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

  // Indicates we want to use DS buf api
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

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_dsexample_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_dsexample_generate_output);

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

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for processing",
          1, NVDSEXAMPLE_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

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
 * Initialize all resources and start the process thread
 */
static gboolean
gst_dsexample_start (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  std::string nvtx_str;
#ifdef WITH_OPENCV
  // OpenCV mat containing RGB data
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif
  NvBufSurfaceCreateParams create_params = { 0 };
  DsExampleInitParams init_params =
      { dsexample->processing_width, dsexample->processing_height,
    dsexample->process_full_frame };

  /* Algorithm specific initializations and resource allocation. */
  dsexample->dsexamplelib_ctx = DsExampleCtxInit (&init_params);

  GST_DEBUG_OBJECT (dsexample, "ctx lib %p \n", dsexample->dsexamplelib_ctx);

  nvtx_str = "GstNvDsExample: UID=" + std::to_string(dsexample->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  dsexample->is_integrated = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      dsexample->is_integrated = 1;
    }
  }

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsexample->cuda_stream),
      "Could not create cuda stream");

#ifdef WITH_OPENCV
  if (dsexample->inter_buf)
    NvBufSurfaceDestroy (dsexample->inter_buf);
  dsexample->inter_buf = NULL;
#endif

  /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
   * required. Can be skipped if custom algorithm can work directly on NV12/RGBA. */
  create_params.gpuId = dsexample->gpu_id;
  create_params.width = dsexample->processing_width;
  create_params.height = dsexample->processing_height;
  create_params.size = 0;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  if(dsexample->is_integrated) {
    create_params.memType = NVBUF_MEM_DEFAULT;
  }
  else {
    create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
  }

#ifdef WITH_OPENCV
  if (NvBufSurfaceCreate (&dsexample->inter_buf, dsexample->max_batch_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer for dsexample");
    goto error;
  }
#endif

  /* Create process queue and cvmat queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  dsexample->process_queue = g_queue_new ();
  dsexample->buf_queue = g_queue_new ();

#ifdef WITH_OPENCV
  /* Push cvmat buffer twice on the buf_queue which will handle the
   * different processing speed between input thread and process thread
   * cvmat queue is used for getting processed data from the process thread*/
  for (int i = 0; i < 2; i++) {
    // CV Mat containing interleaved RGB data.
    cvmat = new cv::Mat[dsexample->max_batch_size];

    for (guint j = 0; j < dsexample->max_batch_size; j++) {
      cvmat[j] =
          cv::Mat (dsexample->processing_height, dsexample->processing_width,
          CV_8UC3);
    }

    if (!cvmat)
      goto error;

    g_queue_push_tail (dsexample->buf_queue, cvmat);
  }

  GST_DEBUG_OBJECT (dsexample, "created CV Mat\n");
#else
  for (int i = 0; i < 2; i++) {
    if (NvBufSurfaceCreate (&inter_buf, dsexample->max_batch_size,
          &create_params) != 0) {
      GST_ERROR ("Error: Could not allocate internal buffer for dsexample");
      goto error;
    }

    g_queue_push_tail (dsexample->buf_queue, inter_buf);
  }
#endif

  /* Set the NvBufSurfTransform config parameters. */
  dsexample->transform_config_params.compute_mode =
      NvBufSurfTransformCompute_Default;
  dsexample->transform_config_params.gpu_id = dsexample->gpu_id;

  /* Create the intermediate NvBufSurface structure for holding an array of input
   * NvBufSurfaceParams for batched transforms. */
  dsexample->batch_insurf.surfaceList =
      new NvBufSurfaceParams[dsexample->max_batch_size];
  dsexample->batch_insurf.batchSize = dsexample->max_batch_size;
  dsexample->batch_insurf.gpuId = dsexample->gpu_id;

  /* Set up the NvBufSurfTransformParams structure for batched transforms. */
  dsexample->transform_params.src_rect =
      new NvBufSurfTransformRect[dsexample->max_batch_size];
  dsexample->transform_params.dst_rect =
      new NvBufSurfTransformRect[dsexample->max_batch_size];
  dsexample->transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  dsexample->transform_params.transform_flip = NvBufSurfTransform_None;
  dsexample->transform_params.transform_filter =
      NvBufSurfTransformInter_Default;

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  dsexample->process_thread =
      g_thread_new ("dsexample-process-thread", gst_dsexample_output_loop,
      dsexample);

  dsexample->nvtx_domain = nvtx_domain_ptr.release ();

  return TRUE;
error:

  delete[]dsexample->transform_params.src_rect;
  delete[]dsexample->transform_params.dst_rect;
  delete[]dsexample->batch_insurf.surfaceList;

  if (dsexample->cuda_stream) {
    cudaStreamDestroy (dsexample->cuda_stream);
    dsexample->cuda_stream = NULL;
  }
  if (dsexample->dsexamplelib_ctx)
    DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_dsexample_stop (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);

#ifdef WITH_OPENCV
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif

  g_mutex_lock (&dsexample->process_lock);

  /* Wait till all the items in the queue are handled. */
  while (!g_queue_is_empty (dsexample->process_queue)) {
    g_cond_wait (&dsexample->process_cond, &dsexample->process_lock);
  }

#ifdef WITH_OPENCV
  while (!g_queue_is_empty (dsexample->buf_queue)) {
    cvmat = (cv::Mat *) g_queue_pop_head (dsexample->buf_queue);
    delete[]cvmat;
    cvmat = NULL;
  }
#else
  while (!g_queue_is_empty (dsexample->buf_queue)) {
    inter_buf = (NvBufSurface *) g_queue_pop_head (dsexample->buf_queue);
    if (inter_buf)
      NvBufSurfaceDestroy (inter_buf);
    inter_buf = NULL;
  }
#endif
  dsexample->stop = TRUE;

  g_cond_broadcast (&dsexample->process_cond);
  g_mutex_unlock (&dsexample->process_lock);

  g_thread_join (dsexample->process_thread);

#ifdef WITH_OPENCV
  if (dsexample->inter_buf)
    NvBufSurfaceDestroy (dsexample->inter_buf);
  dsexample->inter_buf = NULL;
#endif

  if (dsexample->cuda_stream)
    cudaStreamDestroy (dsexample->cuda_stream);
  dsexample->cuda_stream = NULL;

  delete[]dsexample->transform_params.src_rect;
  delete[]dsexample->transform_params.dst_rect;
  delete[]dsexample->batch_insurf.surfaceList;

#ifdef WITH_OPENCV
  GST_DEBUG_OBJECT (dsexample, "deleted CV Mat \n");
#endif

  // Deinit the algorithm library
  DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
  dsexample->dsexamplelib_ctx = NULL;

  GST_DEBUG_OBJECT (dsexample, "ctx lib released \n");

  g_queue_free (dsexample->process_queue);

  g_queue_free (dsexample->buf_queue);

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

  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio and fills data for batched conversation */
static GstFlowReturn
scale_and_fill_data(GstDsExample * dsexample,
    NvBufSurfaceParams * src_frame, NvOSD_RectParams * crop_rect_params,
    gdouble & ratio, gint input_width, gint input_height)
{

  gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
  gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
  gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
  gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);

  // Maintain aspect ratio
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

  // Calculate scaling ratio while maintaining aspect ratio
  ratio = MIN (1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  if(dsexample->is_integrated) {
    if (ratio <= 1.0 / 16 || ratio >= 16.0) {
      // Currently cannot scale by ratio > 16 or < 1/16 for Jetson
      return GST_FLOW_ERROR;
    }
  }

  /* We will first convert only the Region of Interest (the entire frame or the
   * object bounding box) to RGB and then scale the converted RGB frame to
   * processing resolution. */
  GST_DEBUG_OBJECT (dsexample, "Scaling and converting input buffer\n");

  /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
  dsexample->batch_insurf.surfaceList[dsexample->batch_insurf.numFilled] = *src_frame;

  /* Set the source ROI. Could be entire frame or an object. */
  dsexample->transform_params.src_rect[dsexample->batch_insurf.numFilled] = {
  (guint) src_top, (guint) src_left, (guint) src_width, (guint) src_height};
  /* Set the dest ROI. Could be the entire destination frame or part of it to
   * maintain aspect ratio. */
  dsexample->transform_params.dst_rect[dsexample->batch_insurf.numFilled] = {
  0, 0, dest_width, dest_height};

  dsexample->batch_insurf.numFilled++;

  return GST_FLOW_OK;
}

static gboolean
convert_batch_and_push_to_process_thread (GstDsExample * dsexample,
    GstDsExampleBatch * batch)
{

  NvBufSurfTransform_Error err;
  NvBufSurfTransformConfigParams transform_config_params;
  std::string nvtx_str;
#ifdef WITH_OPENCV
  cv::Mat in_mat;
#endif

  // Configure transform session parameters for the transformation
  transform_config_params.compute_mode = dsexample->transform_config_params.compute_mode;
  transform_config_params.gpu_id = dsexample->gpu_id;
  transform_config_params.cuda_stream = dsexample->cuda_stream;

  err = NvBufSurfTransformSetSessionParams (&transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err),
        (NULL));
    return FALSE;
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(dsexample->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(dsexample->nvtx_domain, &eventAttrib);

  g_mutex_lock (&dsexample->process_lock);

  /* Wait if buf queue is empty. */
  while (g_queue_is_empty (dsexample->buf_queue)) {
    g_cond_wait (&dsexample->buf_cond, &dsexample->process_lock);
  }

#ifdef WITH_OPENCV
  /* Pop a buffer from the element's buf queue. */
  batch->cvmat = (cv::Mat *) g_queue_pop_head (dsexample->buf_queue);
#else
  /* Pop a buffer from the element's buf queue. */
  batch->inter_buf = (NvBufSurface *) g_queue_pop_head (dsexample->buf_queue);
  dsexample->inter_buf = batch->inter_buf;
#endif

  g_mutex_unlock (&dsexample->process_lock);

  //Memset the memory
  for (uint i = 0; i < dsexample->batch_insurf.numFilled; i++)
    NvBufSurfaceMemSet (dsexample->inter_buf, i, 0, 0);

  /* Batched tranformation. */
  err = NvBufSurfTransform (&dsexample->batch_insurf, dsexample->inter_buf,
      &dsexample->transform_params);

  nvtxDomainRangePop (dsexample->nvtx_domain);

  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer",
            err), (NULL));
    return FALSE;
  }

  // Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
  // algorithm can handle padded RGBA data.
  for (guint i = 0; i < dsexample->batch_insurf.numFilled; i++) {
    // Map the buffer so that it can be accessed by CPU
    if (NvBufSurfaceMap (dsexample->inter_buf, i, 0, NVBUF_MAP_READ) != 0) {
      GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("%s:buffer map to be accessed by CPU failed", __func__), (NULL));
      return FALSE;
    }
    if(dsexample->is_integrated) {
      // sync mapped data for CPU access
      NvBufSurfaceSyncForCpu (dsexample->inter_buf, i,0);
    }

#ifdef WITH_OPENCV
    in_mat =
        cv::Mat (dsexample->processing_height, dsexample->processing_width,
        CV_8UC4,dsexample->inter_buf->surfaceList[i].mappedAddr.addr[0],
      dsexample->inter_buf->surfaceList[i].pitch);

#if (CV_MAJOR_VERSION >= 4)
  cv::cvtColor (in_mat, batch->cvmat[i], cv::COLOR_RGBA2BGR);
#else
  cv::cvtColor (in_mat, batch->cvmat[i], CV_RGBA2BGR);
#endif

#ifdef DSEXAMPLE_DEBUG
  static guint cnt = 0;
  cv::imwrite("out_" + std::to_string (cnt) + ".jpeg", batch->cvmat[i]);
  cnt++;
#endif
#endif

    if (NvBufSurfaceUnMap (dsexample->inter_buf, i,0)) {
      GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("%s:buffer unmap to be accessed by CPU failed", __func__), (NULL));
      return FALSE;
    }

    if(dsexample->is_integrated) {
      // To use the converted buffer in CUDA, create an EGLImage and then use
      // CUDA-EGL interop APIs
      if (USE_EGLIMAGE) {
        if (NvBufSurfaceMapEglImage (dsexample->inter_buf, 0) != 0) {
          GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
            ("%s:buffer map eglimage failed", __func__), (NULL));
          return FALSE;
        }
        // dsexample->inter_buf->surfaceList[0].mappedAddr.eglImage
        // Use interop APIs cuGraphicsEGLRegisterImage and
        // cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA

        // Destroy the EGLImage
        NvBufSurfaceUnMapEglImage (dsexample->inter_buf, 0);
      }
    }
  }

  /* Push the batch info structure in the processing queue and notify the process
   * thread that a new batch has been queued. */
  g_mutex_lock (&dsexample->process_lock);

  g_queue_push_tail (dsexample->process_queue, batch);
  g_cond_broadcast (&dsexample->process_cond);

  g_mutex_unlock (&dsexample->process_lock);

  return TRUE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsexample_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  GstDsExampleBatch *buf_push_batch;
  GstFlowReturn flow_ret;
  std::string nvtx_str;
  std::unique_ptr < GstDsExampleBatch > batch = nullptr;

  NvDsBatchMeta *batch_meta = NULL;
  guint i = 0;
  gdouble scale_ratio = 1.0;
  guint num_filled = 0;

  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      is_nvgpu = 1;
    }
  }
  dsexample->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(dsexample->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(dsexample->nvtx_domain, &eventAttrib);

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  num_filled = batch_meta->num_frames_in_batch;

  if (dsexample->process_full_frame) {
    for (guint i = 0; i < num_filled; i++) {
      NvOSD_RectParams rect_params;

      // Scale the entire frame to processing resolution
      rect_params.left = 0;
      rect_params.top = 0;
      rect_params.width = in_surf->surfaceList[i].width;
      rect_params.height = in_surf->surfaceList[i].height;

      // Scale the frame maintaining aspect ratio
      if (scale_and_fill_data (dsexample, in_surf->surfaceList + i,
              &rect_params, scale_ratio, dsexample->video_info.width,
              dsexample->video_info.height) != GST_FLOW_OK) {
        goto error;
      }

      if (batch == nullptr) {
        batch.reset (new GstDsExampleBatch);
        batch->push_buffer = FALSE;
        batch->inbuf = inbuf;
        batch->inbuf_batch_num = dsexample->current_batch_num;
      }

      /* Adding a frame to the current batch. Set the frames members. */
      GstDsExampleFrame frame;
      frame.scale_ratio_x = scale_ratio;
      frame.scale_ratio_y = scale_ratio;
      frame.obj_meta = nullptr;
      frame.frame_meta =
          nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
      frame.frame_num = frame.frame_meta->frame_num;
      frame.batch_index = i;
      frame.input_surf_params = in_surf->surfaceList + i;
      batch->frames.push_back (frame);

      // Set the transform session parameters for the conversions executed in this
      // thread.
      if (batch->frames.size () == dsexample->max_batch_size || i == num_filled) {
        if (!convert_batch_and_push_to_process_thread (dsexample, batch.get ())) {
          return GST_FLOW_ERROR;
        }
        /* Batch submitted. Set batch to nullptr so that a new GstDsExampleBatch
         * structure can be allocated if required. */
        batch.release ();
        dsexample->batch_insurf.numFilled = 0;
      }
    }
  } else {
    // Using object crops as input to the algorithm. The objects are detected by
    // the primary detector
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_obj = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);

        /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT */
        if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
            obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
          continue;

        /* Extra check for Jetson devices as default compute mode on Jetson is VIC which supports min 16x16 */
        if (is_nvgpu) {
          if (dsexample->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC ||
              dsexample->transform_config_params.compute_mode == NvBufSurfTransformCompute_Default)
          {
            if (obj_meta->rect_params.width < 16 ||
                obj_meta->rect_params.height < 16)
              continue;
          }
        }

        // Crop and scale the object maintainig aspect ratio
        if (scale_and_fill_data (dsexample,
                in_surf->surfaceList + frame_meta->batch_id,
                &obj_meta->rect_params, scale_ratio,
                dsexample->video_info.width,
                dsexample->video_info.height) != GST_FLOW_OK) {
          // Error in conversion, skip processing on object. */
          continue;
        }

        if (batch == nullptr) {
          batch.reset (new GstDsExampleBatch);
          batch->push_buffer = FALSE;
          batch->inbuf = inbuf;
          batch->inbuf_batch_num = dsexample->current_batch_num;
          batch->nvtx_complete_buf_range = buf_process_range;
        }

        /* Adding a frame to the current batch. Set the frames members. */
        GstDsExampleFrame frame;
        frame.scale_ratio_x = scale_ratio;
        frame.scale_ratio_y = scale_ratio;
        frame.obj_meta = obj_meta;
        frame.frame_meta =
            nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
        frame.frame_num = frame.frame_meta->frame_num;
        frame.batch_index = i;
        frame.input_surf_params = in_surf->surfaceList + i;
        batch->frames.push_back (frame);

        i++;

        // Convert batch and push to process thread
        if (batch->frames.size () == dsexample->max_batch_size
            || i == num_filled) {
          if (!convert_batch_and_push_to_process_thread (dsexample,
                  batch.get ())) {
            return GST_FLOW_ERROR;
          }
          /* Batch submitted. Set batch to nullptr so that a new GstDsExampleBatch
           * structure can be allocated if required. */
          i = 0;
          batch.release ();
          dsexample->batch_insurf.numFilled = 0;
        }
      }
    }
  }
    /* Submit a non-full batch. */
  if (batch) {
    if (!convert_batch_and_push_to_process_thread (dsexample, batch.get ())) {
      return GST_FLOW_ERROR;
    }
    batch.release ();
    dsexample->batch_insurf.numFilled = 0;
  }

  nvtxDomainRangeEnd(dsexample->nvtx_domain, buf_process_range);

  /* Queue a push buffer batch. This batch is not inferred. This batch is to
   * signal the process thread that there are no more batches
   * belonging to this input buffer and this GstBuffer can be pushed to
   * downstream element once all the previous processing is done. */
  buf_push_batch = new GstDsExampleBatch;
  buf_push_batch->inbuf = inbuf;
  buf_push_batch->push_buffer = TRUE;
  buf_push_batch->nvtx_complete_buf_range = buf_process_range;

  g_mutex_lock (&dsexample->process_lock);
  /* Check if this is a push buffer or event marker batch. If yes, no need to
   * queue the input for inferencing. */
  if (buf_push_batch->push_buffer) {
    /* Push the batch info structure in the processing queue and notify the
     * process thread that a new batch has been queued. */
    g_queue_push_tail (dsexample->process_queue, buf_push_batch);
    g_cond_broadcast (&dsexample->process_cond);
  }
  g_mutex_unlock (&dsexample->process_lock);

  flow_ret = GST_FLOW_OK;

error:
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_dsexample_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  return dsexample->last_flow_ret;
}

/**
 * Attach metadata for the full frame. We will be adding a new metadata.
 */
static void
attach_metadata_full_frame (GstDsExample * dsexample,
    NvDsFrameMeta * frame_meta, gdouble scale_ratio, DsExampleOutput * output,
    guint batch_id)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  NvDsObjectMeta *object_meta = NULL;
  static gchar font_name[] = "Serif";
  GST_DEBUG_OBJECT (dsexample, "Attaching metadata %d\n", output->numObjects);

  for (gint i = 0; i < output->numObjects; i++) {
    DsExampleObject *obj = &output->object[i];
    object_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
    NvOSD_RectParams & rect_params = object_meta->rect_params;
    NvOSD_TextParams & text_params = object_meta->text_params;

    // Assign bounding box coordinates
    rect_params.left = obj->left;
    rect_params.top = obj->top;
    rect_params.width = obj->width;
    rect_params.height = obj->height;

    // Semi-transparent yellow background
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    // Red border of width 6
    rect_params.border_width = 3;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    // Scale the bounding boxes proportionally based on how the object/frame was
    // scaled during input
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
    // display_text required heap allocated memory
    text_params.display_text = g_strdup (obj->label);
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    // Font face, size and color
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};

    nvds_add_obj_meta_to_frame (frame_meta, object_meta, NULL);
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
  nvds_add_label_info_meta_to_classifier (classifier_meta, label_info);
  nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);

  nvds_acquire_meta_lock (batch_meta);
  NvOSD_TextParams & text_params = obj_meta->text_params;
  NvOSD_RectParams & rect_params = obj_meta->rect_params;

  /* Below code to display the result */
  // Set black background for the text
  // display_text required heap allocated memory
  if (text_params.display_text) {
    gchar *conc_string = g_strconcat (text_params.display_text, " ",
        output->object[0].label, NULL);
    g_free (text_params.display_text);
    text_params.display_text = conc_string;
  } else {
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    text_params.display_text = g_strdup (output->object[0].label);
    // Font face, size and color
    text_params.font_params.font_name = (char *) "Serif";
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
  }
  nvds_release_meta_lock (batch_meta);
}

/**
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_dsexample_output_loop (gpointer data)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (data);
  DsExampleOutput *output;
  NvDsObjectMeta *obj_meta = NULL;
  gdouble scale_ratio = 1.0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;

  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      is_nvgpu = 1;
    }
  }
  nvtx_str =
      "gst-dsexample_output-loop_uid=" + std::to_string (dsexample->unique_id);

  g_mutex_lock (&dsexample->process_lock);

  /* Run till signalled to stop. */
  while (!dsexample->stop) {
    std::unique_ptr < GstDsExampleBatch > batch = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (dsexample->process_queue)) {
      g_cond_wait (&dsexample->process_cond, &dsexample->process_lock);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((GstDsExampleBatch *)
        g_queue_pop_head (dsexample->process_queue));
    g_cond_broadcast (&dsexample->process_cond);

    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    g_mutex_unlock (&dsexample->process_lock);

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(dsexample->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp (batch->inbuf,
          GST_ELEMENT_NAME (dsexample));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (dsexample),
          batch->inbuf);
      if (dsexample->last_flow_ret != flow_ret) {
        switch (flow_ret) {
            /* Signal the application for pad push errors by posting a error message
             * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)",
                    gst_flow_get_name (flow_ret), flow_ret));
            break;
          default:
            break;
        }
      }
      dsexample->last_flow_ret = flow_ret;
      g_mutex_lock (&dsexample->process_lock);
      continue;
    }

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(dsexample->nvtx_domain, &eventAttrib);

    /* For each frame attach metadata output. */
    for (guint i = 0; i < batch->frames.size (); i++) {
      if (dsexample->process_full_frame) {
        // Process to get the output
#ifdef WITH_OPENCV
        output =
            DsExampleProcess (dsexample->dsexamplelib_ctx,
            batch->cvmat[i].data);
#else
        output =
            DsExampleProcess (dsexample->dsexamplelib_ctx,
            (unsigned char *)batch->inter_buf->surfaceList[i].mappedAddr.addr[0]);
#endif
        // Attach the metadata for the full frame
        attach_metadata_full_frame (dsexample, batch->frames[i].frame_meta,
            scale_ratio, output, i);
        free (output);
      } else {
        GstDsExampleFrame & frame = batch->frames[i];

        obj_meta = frame.obj_meta;

        /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT */
        if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
            obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
          continue;

        /* Extra check for Jetson devices as default compute mode on Jetson is VIC which supports min 16x16 */
        if (is_nvgpu) {
          if (dsexample->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC ||
              dsexample->transform_config_params.compute_mode == NvBufSurfTransformCompute_Default)
          {
            if (obj_meta->rect_params.width < 16 ||
                obj_meta->rect_params.height < 16)
              continue;
          }
        }

        // Process the object crop to obtain label
#ifdef WITH_OPENCV
        output = DsExampleProcess (dsexample->dsexamplelib_ctx,
            batch->cvmat[i].data);
#else
        output = DsExampleProcess (dsexample->dsexamplelib_ctx,
            (unsigned char *)batch->inter_buf->surfaceList[i].mappedAddr.addr[0]);
#endif

        // Attach labels for the object
        attach_metadata_object (dsexample, obj_meta, output);

        free (output);
      }
    }

    g_mutex_lock (&dsexample->process_lock);

#ifdef WITH_OPENCV
    g_queue_push_tail (dsexample->buf_queue, batch->cvmat);
#else
    g_queue_push_tail (dsexample->buf_queue, batch->inter_buf);
#endif
    g_cond_broadcast (&dsexample->buf_cond);

    nvtxDomainRangePop (dsexample->nvtx_domain);
  }
  g_mutex_unlock (&dsexample->process_lock);

  return nullptr;
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
    DESCRIPTION, dsexample_plugin_init, "9.0", LICENSE, BINARY_PACKAGE,
    URL)
