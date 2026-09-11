/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sstream>
#include <sys/time.h>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <list>
#include <thread>

#include "gst-nvevent.h"
#include "gstnvdsmeta.h"

#include "gstnvinferbase.h"
#include "gstnvinfer_property_parser.h"
#include "gstnvinfer_impl.h"

using namespace gstnvinfer;
using namespace nvdsinfer;

GST_DEBUG_CATEGORY (gst_nvinferbase_debug);
#define GST_CAT_DEFAULT gst_nvinferbase_debug

#define DS_NVINFER_IMPL(gst_nvinfer) reinterpret_cast<DsNvInferImpl*>((gst_nvinfer)->impl)

/* Gst-nvinfer supports runtime model updates. Refer to gstnvinfer_impl.h
 * for details. */

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_OUTPUT_WRITE_TO_FILE FALSE
#define DEFAULT_OUTPUT_TENSOR_META FALSE
#define DEFAULT_CLASSIFIER_TYPE ""

guint gst_nvinfer_signals[LAST_SIGNAL] = { 0 };

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvinferbase_parent_class parent_class
G_DEFINE_TYPE (GstNvInferBase, gst_nvinferbase, GST_TYPE_BASE_TRANSFORM);

/* Implementation of the GObject/GstBaseTransform interfaces. */
static void gst_nvinfer_finalize (GObject * object);
static void gst_nvinfer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvinfer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvinfer_start (GstBaseTransform * btrans);
static gboolean gst_nvinfer_stop (GstBaseTransform * btrans);
static gboolean gst_nvinfer_sink_event (GstBaseTransform * trans,
    GstEvent * event);

static gboolean gst_nvinfer_set_caps (GstBaseTransform * btrans,
         GstCaps * incaps, GstCaps * outcaps);

static GstFlowReturn gst_nvinfer_submit_input_buffer (GstBaseTransform *
    btrans, gboolean discont, GstBuffer * inbuf);
static GstFlowReturn gst_nvinfer_generate_output (GstBaseTransform *
    btrans, GstBuffer ** outbuf);

static gpointer gst_nvinfer_input_queue_loop (gpointer data);
static gpointer gst_nvinfer_output_loop (gpointer data);

static void gst_nvinfer_reset_init_params (GstNvInferBase * nvinfer);


typedef struct {
  guint batch_index = 0;

  gulong tensor_num = 0;
  gpointer tensor_ptr = nullptr;
} GstNvInferTensor;

typedef struct {
  gulong inbuf_batch_num = 0;

  gboolean push_buffer = FALSE;
  gboolean event_marker = FALSE;

  GstBuffer *inbuf = nullptr;
  GstBuffer *conv_buf = nullptr;

  std::vector<GstNvInferTensor> tensors;

  nvtxRangeId_t nvtx_complete_buf_range = 0;
} GstNvInferTensorBatch;

static inline int
get_element_size (NvDsInferDataType data_type)
{
  switch (data_type) {
    case FLOAT:
      return 4;
    case HALF:
      return 2;
    case INT32:
      return 4;
    case INT8:
    case UINT8:
      return 1;
    case INT64:
      return 8;
    default:
      return 0;
  }
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvinferbase_class_init (GstNvInferBaseClass * klass)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *gstbasetransform_class;

  GST_DEBUG_CATEGORY_INIT (gst_nvinferbase_debug, "nvinferbase", 0, "nvinferbase transform");

  gobject_class = (GObjectClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvinfer_finalize);
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvinfer_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvinfer_get_property);

  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvinfer_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvinfer_stop);
  gstbasetransform_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvinfer_sink_event);
  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvinfer_set_caps);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_nvinfer_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_nvinfer_generate_output);

  /* Install properties. Values set through these properties override the ones in
   * the config file. */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element. Can be used to "
          "identify output of the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Config File Path",
          "Path to the configuration file for this instance of nvinfer",
          DEFAULT_CONFIG_FILE_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_CLASSIFIER_TYPE,
      g_param_spec_string ("classifier-type", "Classifier Type",
          "Type of classifier of this instance of nvinfer",
          DEFAULT_CLASSIFIER_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for inference",
          1, NVDSINFER_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_MODEL_ENGINEFILE,
      g_param_spec_string ("model-engine-file", "Model Engine File",
          "Absolute path to the pre-generated serialized engine file for the model",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, DEFAULT_GPU_DEVICE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_WRITE_TO_FILE,
      g_param_spec_boolean ("raw-output-file-write", "Raw Output File Write",
          "Write raw inference output to file",
          DEFAULT_OUTPUT_WRITE_TO_FILE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_CALLBACK,
      g_param_spec_pointer ("raw-output-generated-callback",
          "Raw Output Generated Callback",
          "Pointer to the raw output generated callback funtion\n"
          "\t\t\t(type: gst_nvinfer_raw_output_generated_callback in 'gstnvdsinfer.h')",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_CALLBACK_USERDATA,
      g_param_spec_pointer ("raw-output-generated-userdata",
          "Raw Output Generated UserData",
          "Pointer to the userdata to be supplied with raw output generated callback",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_TENSOR_META,
      g_param_spec_boolean ("output-tensor-meta", "Output Tensor Meta",
          "Attach inference tensor outputs as buffer metadata",
          DEFAULT_OUTPUT_TENSOR_META,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  /** install signal MODEL_UPDATED */
  gst_nvinfer_signals[SIGNAL_MODEL_UPDATED] =
      g_signal_new ("model-updated",
      G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstNvInferBaseClass, model_updated),
      NULL, NULL, NULL,
      G_TYPE_NONE, 2, G_TYPE_INT, G_TYPE_STRING);
}

static void
gst_nvinferbase_init (GstNvInferBase * nvinfer)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvinfer);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  nvinfer->impl = reinterpret_cast<GstNvInferImpl*>(new DsNvInferImpl(nvinfer));
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);

  /* Initialize all property variables to default values */
  nvinfer->unique_id = DEFAULT_UNIQUE_ID;
  nvinfer->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);
  nvinfer->classifier_type = g_strdup (DEFAULT_CLASSIFIER_TYPE);
  nvinfer->output_tensor_meta = DEFAULT_OUTPUT_TENSOR_META;

  nvinfer->max_batch_size = impl->m_InitParams->maxBatchSize =
      DEFAULT_BATCH_SIZE;
  nvinfer->gpu_id = impl->m_InitParams->gpuID = DEFAULT_GPU_DEVICE_ID;
  nvinfer->is_prop_set = new std::vector < gboolean > (PROP_LAST, FALSE);

  /* Create processing lock and condition for synchronization.*/
  g_mutex_init (&nvinfer->process_lock);
  g_cond_init (&nvinfer->process_cond);
}

/* Free resources allocated during init. */
static void
gst_nvinfer_finalize (GObject * object)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (object);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);

  g_mutex_clear (&nvinfer->process_lock);
  g_cond_clear (&nvinfer->process_cond);

  delete nvinfer->is_prop_set;
  g_free (nvinfer->config_file_path);

  if (klass->finalize)
    klass->finalize (object);

  delete DS_NVINFER_IMPL(nvinfer);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvinfer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (object);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);

  if (prop_id < PROP_LAST) {
    /* Mark the property as being set through g_object_set. */
    (*nvinfer->is_prop_set)[prop_id] = TRUE;
  }

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      impl->m_InitParams->uniqueID = nvinfer->unique_id =
          g_value_get_uint (value);
      break;
    case PROP_CONFIG_FILE_PATH:
      {
        LockGMutex lock (nvinfer->process_lock);
        const std::string cfg_path (g_value_get_string (value));
        if (impl->isContextReady ()) {
            /* A NvDsInferContext is being used. Trigger a new model update. */
            impl->triggerNewModel (cfg_path, MODEL_LOAD_FROM_CONFIG);
            break;
        }
        g_free (nvinfer->config_file_path);
        nvinfer->config_file_path = g_value_dup_string (value);
        gst_nvinfer_reset_init_params (nvinfer);
        /* Parse the initialization parameters from the config file. This function
         * gives preference to values set through the set_property function over
         * the values set in the config file. */
        nvinfer->config_file_parse_successful =
            gst_nvinfer_parse_config_file (nvinfer, impl->m_InitParams.get(),
                nvinfer->config_file_path);
      }
      break;
    case PROP_CLASSIFIER_TYPE:
      nvinfer->classifier_type = g_value_dup_string (value);
      break;
    case PROP_BATCH_SIZE:
      nvinfer->max_batch_size = impl->m_InitParams->maxBatchSize =
          g_value_get_uint (value);
      break;
    case PROP_MODEL_ENGINEFILE:
      {
        LockGMutex lock (nvinfer->process_lock);
        const std::string engine_path (g_value_get_string (value));
        if (impl->isContextReady ()) {
            /* A NvDsInferContext is being used. Trigger a new model update. */
            impl->triggerNewModel (engine_path, MODEL_LOAD_FROM_ENGINE);
            break;
        }
        g_strlcpy (impl->m_InitParams->modelEngineFilePath,
            g_value_get_string (value), _PATH_MAX);
      }
      break;
    case PROP_GPU_DEVICE_ID:
      nvinfer->gpu_id = impl->m_InitParams->gpuID = g_value_get_uint (value);
      break;
    case PROP_OUTPUT_WRITE_TO_FILE:
      nvinfer->write_raw_buffers_to_file = g_value_get_boolean (value);
      break;
    case PROP_OUTPUT_CALLBACK:
      nvinfer->output_generated_callback =
          (gst_nvinfer_raw_output_generated_callback)
          g_value_get_pointer (value);
      break;
    case PROP_OUTPUT_CALLBACK_USERDATA:
      nvinfer->output_generated_userdata = g_value_get_pointer (value);
      break;
    case PROP_OUTPUT_TENSOR_META:
      nvinfer->output_tensor_meta = g_value_get_boolean (value);
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
gst_nvinfer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (object);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, nvinfer->unique_id);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, nvinfer->config_file_path);
      break;
    case PROP_CLASSIFIER_TYPE:
      g_value_set_string (value, nvinfer->classifier_type);
      break;
    case PROP_MODEL_ENGINEFILE:
      g_value_set_string (value, impl->m_InitParams->modelEngineFilePath);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, nvinfer->max_batch_size);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvinfer->gpu_id);
      break;
    case PROP_OUTPUT_WRITE_TO_FILE:
      g_value_set_boolean (value, nvinfer->write_raw_buffers_to_file);
      break;
    case PROP_OUTPUT_CALLBACK:
      g_value_set_pointer (value,
          (gpointer) nvinfer->output_generated_callback);
      break;
    case PROP_OUTPUT_CALLBACK_USERDATA:
      g_value_set_pointer (value, nvinfer->output_generated_userdata);
      break;
    case PROP_OUTPUT_TENSOR_META:
      g_value_set_boolean (value, nvinfer->output_tensor_meta);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

void gst_nvinfer_logger(NvDsInferContextHandle handle, unsigned int unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx) {
    GstNvInferBase *nvinfer = GST_NVINFERBASE(user_ctx);

    switch (log_level) {
    case NVDSINFER_LOG_ERROR:
        GST_ERROR_OBJECT(nvinfer, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_WARNING:
        GST_WARNING_OBJECT(nvinfer, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_INFO:
        GST_INFO_OBJECT(nvinfer, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_DEBUG:
        GST_DEBUG_OBJECT(nvinfer, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
  }
}

/**
 * Reset m_InitParams structure while preserving property values set through
 * GObject set method. */
static void
gst_nvinfer_reset_init_params (GstNvInferBase * nvinfer)
{
  DsNvInferImpl *impl = DS_NVINFER_IMPL(nvinfer);
  auto prev_params = std::move(impl->m_InitParams);
  impl->m_InitParams.reset (new NvDsInferContextInitParams);
  assert (impl->m_InitParams);
  NvDsInferContext_ResetInitParams (impl->m_InitParams.get ());

  if (nvinfer->is_prop_set->at (PROP_MODEL_ENGINEFILE))
    g_strlcpy (impl->m_InitParams->modelEngineFilePath,
        prev_params->modelEngineFilePath, _PATH_MAX);

  if (nvinfer->is_prop_set->at (PROP_BATCH_SIZE))
    impl->m_InitParams->maxBatchSize = prev_params->maxBatchSize;

  if (nvinfer->is_prop_set->at (PROP_GPU_DEVICE_ID))
    impl->m_InitParams->gpuID = prev_params->gpuID;

  delete prev_params->perClassDetectionParams;
  g_strfreev (prev_params->outputLayerNames);
}

/**
 * Called when an event is recieved on the sink pad. We need to make sure
 * serialized events and buffers are pushed downstream while maintaining the order.
 * To ensure this, we push all the buffers in the internal queue to the
 * downstream element before forwarding the serialized event to the downstream element.
 */
static gboolean
gst_nvinfer_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (trans);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);
  gboolean ignore_serialized_event = FALSE;

  /** The TAG event is sent many times leading to drop in performance because of
   * buffer/event serialization. We can ignore such events which won't cause
   * issues if we don't serialize the events. */
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_TAG:
      ignore_serialized_event = TRUE;
      break;
    default:
      break;
  }

  /* Serialize events. Wait for pending buffers to be processed and pushed
   * downstream. No need to wait in case of classifier async mode since all
   * the buffers are already pushed downstream. */
  if (GST_EVENT_IS_SERIALIZED (event) && !ignore_serialized_event &&
      !klass->get_classifier_async_mode(nvinfer)) {
    GstNvInferBatch *batch = new GstNvInferBatch;
    batch->event_marker = TRUE;

    g_mutex_lock (&nvinfer->process_lock);
    /* Push the event marker batch in the processing queue. */
    g_queue_push_tail (nvinfer->input_queue, batch);
    g_cond_broadcast (&nvinfer->process_cond);

    /* Wait for all the remaining batches in the queue including the event
     * marker to be processed. */
    while (!g_queue_is_empty (nvinfer->input_queue)) {
      g_cond_wait (&nvinfer->process_cond, &nvinfer->process_lock);
    }
    while (!g_queue_is_empty (nvinfer->process_queue)) {
      g_cond_wait (&nvinfer->process_cond, &nvinfer->process_lock);
    }
    g_mutex_unlock (&nvinfer->process_lock);
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
    nvinfer->interval_counter = 0;
  }

  if (klass->sink_event)
    klass->sink_event (nvinfer, event);

  /* Call the sink event handler of the base class. */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvinfer_start (GstBaseTransform * btrans)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (btrans);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);
  NvDsInferStatus status;
  std::string nvtx_str;
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  NvDsInferContextHandle infer_context = nullptr;

  LockGMutex lock (nvinfer->process_lock);
  NvDsInferContextInitParams *init_params = impl->m_InitParams.get ();
  assert (init_params);

  nvtx_str = "GstNvInfer: UID=" + std::to_string(nvinfer->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  /* Providing a valid config file is mandatory. */
  if (!nvinfer->config_file_path || strlen (nvinfer->config_file_path) == 0) {
    GST_ELEMENT_ERROR (nvinfer, LIBRARY, SETTINGS,
        ("Configuration file not provided"), (nullptr));
    return FALSE;
  }
  if (nvinfer->config_file_parse_successful == FALSE) {
    GST_ELEMENT_ERROR (nvinfer, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", nvinfer->config_file_path));
    return FALSE;
  }

  nvinfer->interval_counter = 0;

  /* Ask NvDsInferContext to copy the input layer contents to host memory if
   * CPU needs to access it. */
  init_params->copyInputToHostBuffers =
      (nvinfer->write_raw_buffers_to_file ||
      (nvinfer->output_generated_callback != nullptr));

  /* Set the number of output buffers that should be allocated by NvDsInferContext.
   * Should allocate more buffers if the output tensor buffers will be attached
   * as meta to GstBuffers and pushed downstream. */
  if (klass->get_output_buffer_pool_size)
    init_params->outputBufferPoolSize = klass->get_output_buffer_pool_size (nvinfer);

  /* Create the NvDsInferContext instance. */
  status =
      createNvDsInferContext (&infer_context, *init_params,
      nvinfer, gst_nvinfer_logger);
  if (status != NVDSINFER_SUCCESS) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to create NvDsInferContext instance"),
        ("Config file path: %s, NvDsInfer Error: %s", nvinfer->config_file_path,
            NvDsInferStatus2Str (status)));
    return FALSE;
  }
  std::unique_ptr<INvDsInferContext> ctx_ptr (infer_context);

  /* Get the network resolution. */
  ctx_ptr->getNetworkInfo (nvinfer->network_info);
  nvinfer->network_width = nvinfer->network_info.width;
  nvinfer->network_height = nvinfer->network_info.height;

  /* Get information on all the bound layers. */
  nvinfer->layers_info = new std::vector < NvDsInferLayerInfo > ();
  ctx_ptr->fillLayersInfo (*nvinfer->layers_info);

  nvinfer->output_layers_info = new std::vector < NvDsInferLayerInfo > ();
  for (auto & layer:*(nvinfer->layers_info)) {
    if (!layer.isInput)
      nvinfer->output_layers_info->push_back (layer);
  }

  nvinfer->file_write_batch_num = 0;

  /* Create process queue and input queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  nvinfer->process_queue = g_queue_new ();
  nvinfer->input_queue = g_queue_new ();

  if (klass->start)
    klass->start(nvinfer, init_params);

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  nvinfer->output_thread =
      g_thread_new ("nvinfer-output-thread", gst_nvinfer_output_loop, nvinfer);

  /* Start a thread which will queue input to the NvDsInfer context since
   * queueInputBatch is a blocking function. This is done to parallelize
   * input conversion and queueInputBatch. */
  nvinfer->input_queue_thread =
      g_thread_new ("nvinfer-input-queue-thread", gst_nvinfer_input_queue_loop,
          nvinfer);

  /* nvinfer internal resource start for loading models */
  impl->m_InferCtx = std::move (ctx_ptr);
  if (impl->start () != NVDSINFER_SUCCESS) {
      GST_ELEMENT_WARNING (nvinfer, RESOURCE, FAILED,
          ("NvInfer start loading model thread failed."), (nullptr));
      return FALSE;
  }

  nvinfer->nvtx_domain = nvtx_domain_ptr.release ();
  lock.unlock ();

  impl->notifyLoadModelStatus (
      ModelStatus {NVDSINFER_SUCCESS, nvinfer->config_file_path,
      "Model loaded successfully"});
  return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvinfer_stop (GstBaseTransform * btrans)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (btrans);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);

  LockGMutex locker (nvinfer->process_lock);
  /* Wait till all the items in the two queues are handled. */
  while (!g_queue_is_empty (nvinfer->input_queue)) {
    locker.wait (nvinfer->process_cond);
  }
  while (!g_queue_is_empty (nvinfer->process_queue)) {
    locker.wait (nvinfer->process_cond);
  }
  nvinfer->stop = TRUE;

  g_cond_broadcast (&nvinfer->process_cond);
  locker.unlock ();

  impl->stop ();

  g_thread_join (nvinfer->input_queue_thread);
  g_thread_join (nvinfer->output_thread);

  nvinfer->stop = FALSE;

  delete nvinfer->layers_info;
  delete nvinfer->output_layers_info;

  if (klass->stop)
    klass->stop(nvinfer);

  g_queue_free (nvinfer->process_queue);
  g_queue_free (nvinfer->input_queue);

  return TRUE;
}


static gboolean
gst_nvinfer_set_caps (GstBaseTransform * btrans,
         GstCaps * incaps, GstCaps * outcaps)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (btrans);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);

  if (klass->set_caps)
    klass->set_caps(nvinfer, incaps, outcaps);

  return TRUE;
}

/* Helper function to queue a batch for inferencing and push it to the element's
 * processing queue. */
static gpointer
gst_nvinfer_input_queue_loop (gpointer data)
{
  GstNvInferBase *nvinfer = (GstNvInferBase *) data;
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  std::string nvtx_str;
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;

  LockGMutex locker (nvinfer->process_lock);

  while (nvinfer->stop == FALSE) {
    GstNvInferBatch *batch;
    NvDsInferContextBatchInput input_batch;
    std::vector < void *>input_frames;
    unsigned int i;
    NvDsInferStatus status;

    /* Wait if input queue is empty. */
    if (g_queue_is_empty (nvinfer->input_queue)) {
      locker.wait (nvinfer->process_cond);
      continue;
    }
    batch = (GstNvInferBatch *) g_queue_pop_head (nvinfer->input_queue);
    NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;

    /* Check if this is a push buffer or event marker batch. If yes, no need to
     * queue the input for inferencing. */
    if (batch->push_buffer || batch->event_marker || batch->frames.size() == 0) {
      goto queue_batch;
    }

    if (klass->get_input_format)
      input_batch.inputFormat = klass->get_input_format(nvinfer, batch);
    if (klass->get_input_pitch)
      input_batch.inputPitch = klass->get_input_pitch(nvinfer, batch);

    /* Form the vector of input frame pointers. */
    for (i = 0; i < batch->frames.size (); i++) {
      input_frames.push_back (batch->frames[i].converted_frame_ptr);
    }

    input_batch.inputFrames = input_frames.data ();
    input_batch.numInputFrames = input_frames.size ();

    input_batch.returnInputFunc =
        (NvDsInferContextReturnInputAsyncFunc) gst_buffer_unref;
    input_batch.returnFuncData = batch->conv_buf;

    locker.unlock ();

    nvtx_str = "queueInput batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(nvinfer->nvtx_domain, &eventAttrib);

    status = nvdsinfer_ctx->queueInputBatch (input_batch);
    nvtxDomainRangePop(nvinfer->nvtx_domain);

    locker.lock ();

    if (status != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
          ("Failed to queue input batch for inferencing"), (nullptr));
      continue;
    }

queue_batch:
    /* Push the batch info structure in the processing queue and notify the
     * output thread that a new batch has been queued. */
    g_queue_push_tail (nvinfer->process_queue, batch);
    g_cond_broadcast (&nvinfer->process_cond);
  }

  return NULL;
}

void
gst_nvinfer_push_to_input_queue (GstNvInferBase *nvinfer, GstNvInferBatch *batch)
{
  LockGMutex locker (nvinfer->process_lock);
  /* Push the batch info structure in the processing queue and notify the output
   * thread that a new batch has been queued. */
  g_queue_push_tail (nvinfer->input_queue, batch);
  g_cond_broadcast (&nvinfer->process_cond);
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvinfer_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (btrans);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  GstNvInferBatch *buf_push_batch;
  std::string nvtx_str;
  GstFlowReturn flow_ret;

  /* Check for model updates and replace the model if a new model is loaded. */
  if (impl->ensureReplaceNextContext () != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
              ("Ensure next context failed."),
              ("streaming stopped"));
      return GST_FLOW_ERROR;
  }

  nvinfer->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(nvinfer->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(nvinfer->nvtx_domain, &eventAttrib);

  nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinfer));

  if (!klass->process_input)
    return GST_FLOW_ERROR;

  flow_ret = klass->process_input(nvinfer, inbuf);
  if (flow_ret == GST_FLOW_ERROR)
    return GST_FLOW_ERROR;

  if (klass->get_classifier_async_mode(nvinfer)) {
    /* Asynchronous mode. Push the buffer immediately instead of waiting for
     * the results. */
    nvtxDomainRangeEnd(nvinfer->nvtx_domain, buf_process_range);

    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinfer));

    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvinfer),
        inbuf);
    if (nvinfer->last_flow_ret != flow_ret) {
      switch (flow_ret) {
        /* Signal the application for pad push errors by posting a error message
         * on the pipeline bus. */
        case GST_FLOW_ERROR:
        case GST_FLOW_NOT_LINKED:
        case GST_FLOW_NOT_NEGOTIATED:
          GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
              ("Internal data stream error."),
              ("streaming stopped, reason %s (%d)", gst_flow_get_name (flow_ret),
                  flow_ret));
          break;
        default:
          break;
      }
    }
    nvinfer->last_flow_ret = flow_ret;

    return flow_ret;
  } else {
    /* Queue a push buffer batch. This batch is not inferred. This batch is to
     * signal the input-queue and output thread that there are no more batches
     * belonging to this input buffer and this GstBuffer can be pushed to
     * downstream element once all the previous processing is done. */
    buf_push_batch = new GstNvInferBatch;
    buf_push_batch->inbuf = inbuf;
    buf_push_batch->push_buffer = TRUE;
    buf_push_batch->nvtx_complete_buf_range = buf_process_range;

    g_mutex_lock (&nvinfer->process_lock);
    g_queue_push_tail (nvinfer->input_queue, buf_push_batch);
    g_cond_broadcast (&nvinfer->process_cond);
    g_mutex_unlock (&nvinfer->process_lock);
  }

  return GST_FLOW_OK;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_nvinfer_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstNvInferBase *nvinfer = GST_NVINFERBASE (btrans);
  return nvinfer->last_flow_ret;
}

/** Writes contents of the bound input and output layers to files. */
static void
gst_nvinfer_output_generated_file_write (GstBuffer * buf,
    NvDsInferNetworkInfo * network_info, NvDsInferLayerInfo * layers_info,
    guint num_layers, guint batch_size, GstNvInferBase * nvinfer)
{
  guint i;
  gchar file_name[256];
  gchar *iter;

  for (i = 0; i < num_layers; i++) {
    NvDsInferLayerInfo *info = &layers_info[i];
    gsize layer_size = info->inferDims.numElements * batch_size;
    FILE *file;

    g_snprintf (file_name, 256,
        "gstnvdsinfer_uid-%02d_layer-%s_batch-%010lu_batchsize-%02d.bin",
        nvinfer->unique_id, info->layerName,
        nvinfer->file_write_batch_num, batch_size);
    file_name[255] = '\0';

    /* Replace '/' in a layer name with '_' */
    for (iter = file_name; *iter != '\0'; iter++) {
      if (*iter == '/')
        *iter = '_';
    }

    file = fopen (file_name, "w");
    if (!file) {
      g_printerr ("Could not open file '%s' for writing:%s\n",
          file_name, strerror (errno));
      continue;
    }
    fwrite (info->buffer, get_element_size (info->dataType), layer_size, file);
    fclose (file);
  }
  nvinfer->file_write_batch_num++;
}

/* Called when the last ref on the GstMiniObject inside
 * GstNvInferTensorOutputObject is removed. The batch output can be released
 * back to the NvDsInferContext. */
static void
gst_nvinfer_tensoroutput_free (GstMiniObject * obj)
{
  GstNvInferTensorOutputObject *output_obj =
      (GstNvInferTensorOutputObject *) obj;
  assert (output_obj->infer_context.get());
  output_obj->infer_context->releaseBatchOutput (output_obj->
      batch_output);
  output_obj->infer_context.reset ();
  delete output_obj;
}

/**
 * Output loop used to pop output from inference, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_nvinfer_output_loop (gpointer data)
{
  std::unique_ptr<GstNvInferBatch> batch = nullptr;
  GstNvInferBase *nvinfer = GST_NVINFERBASE (data);
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);
  NvDsInferStatus status = NVDSINFER_SUCCESS;
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;

  nvtx_str = "gst-nvinfer_output-loop_uid=" + std::to_string(nvinfer->unique_id);

  LockGMutex locker (nvinfer->process_lock);
  /* Run till signalled to stop. */
  while (!nvinfer->stop) {
    std::unique_ptr<GstNvInferBatch> batch = nullptr;
    NvDsInferContextBatchOutput *batch_output = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (nvinfer->process_queue)) {
      locker.wait (nvinfer->process_cond);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((GstNvInferBatch *) g_queue_pop_head (nvinfer->process_queue));
    g_cond_broadcast (&nvinfer->process_cond);
    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    /* Attach latest available classification metadata for objects that have
     * not been inferred on in the current frame. */
    if (batch->frames.size() == 0 && !batch->push_buffer) {
      if (klass->attach_previous_metadata)
        klass->attach_previous_metadata (nvinfer, batch.get());
      continue;
    }

    locker.unlock ();

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(nvinfer->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp(batch->inbuf, GST_ELEMENT_NAME(nvinfer));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvinfer),
          batch->inbuf);
      if (nvinfer->last_flow_ret != flow_ret) {
        switch (flow_ret) {
          /* Signal the application for pad push errors by posting a error message
           * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)", gst_flow_get_name (flow_ret),
                    flow_ret));
            break;
          default:
          break;
        }
      }
      nvinfer->last_flow_ret = flow_ret;
      locker.lock ();
      continue;
    }

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(nvinfer->nvtx_domain, &eventAttrib);

    NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;

    /* Create and initialize the object for managing the usage of batch_output. */
    auto tensor_deleter = [] (GstNvInferTensorOutputObject *o) {
      if (o)
        gst_mini_object_unref (GST_MINI_OBJECT (o));
    };
    std::unique_ptr<GstNvInferTensorOutputObject, decltype(tensor_deleter)>
        tensor_out_object (new GstNvInferTensorOutputObject, tensor_deleter);
    gst_mini_object_init (GST_MINI_OBJECT (tensor_out_object.get()), 0, G_TYPE_POINTER, NULL,
        NULL, gst_nvinfer_tensoroutput_free);
    tensor_out_object->infer_context = nvdsinfer_ctx;

    batch_output = &tensor_out_object->batch_output;
    /* Dequeue inferencing output from NvDsInferContext */
    status = nvdsinfer_ctx->dequeueOutputBatch (*batch_output);

    locker.lock ();

    if (status != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
          ("Failed to dequeue output from inferencing. NvDsInferContext error: %s",
              NvDsInferStatus2Str (status)), (nullptr));
      continue;
    }

    /* Get the host buffer pointers from the latest dequeued output. */
    for (auto & layer:*nvinfer->layers_info) {
      layer.buffer = batch_output->hostBuffers[layer.bindingIndex];
    }

    /* Write layer contents to file if enabled. */
    if (nvinfer->write_raw_buffers_to_file) {
      gst_nvinfer_output_generated_file_write (batch->inbuf,
          &nvinfer->network_info,
          nvinfer->layers_info->data (),
          nvinfer->layers_info->size (), batch->frames.size (), nvinfer);
    }

    /* Call the output generated callback if specified. */
    if (nvinfer->output_generated_callback) {
      nvinfer->output_generated_callback (batch->inbuf,
          &nvinfer->network_info,
          nvinfer->layers_info->data (),
          nvinfer->layers_info->size (),
          batch->frames.size (), nvinfer->output_generated_userdata);
    }

    if (klass->process_output)
      klass->process_output (nvinfer, batch.get(), batch_output, GST_MINI_OBJECT (tensor_out_object.get()));

    nvtxDomainRangePop (nvinfer->nvtx_domain);
  }

  return nullptr;
}
