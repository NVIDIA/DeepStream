/**
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
#include <sstream>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <list>
#include <thread>
#include <vector>

#include "gst-nvevent.h"
#include "gst-nvdscustomevent.h"
#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"

#include "gstnvinfer.h"
#include "gstnvinfer_allocator.h"
#include "gstnvinfer_meta_utils.h"
#include "gstnvinfer_yaml_parser.h"
#include "gstnvinfer_property_parser.h"
#include "gstnvinfer_impl.h"

using namespace gstnvinfer;
using namespace nvdsinfer;

GST_DEBUG_CATEGORY (gst_nvinfer_debug);
#define GST_CAT_DEFAULT gst_nvinfer_debug

#define INTERNAL_BUF_POOL_SIZE 3

#define NVDSINFER_CTX_OUT_POOL_SIZE_FLOW_META 6

/* Tracked objects will be reinferred only when their area in terms of pixels
 * increase by this ratio. */
#define REINFER_AREA_THRESHOLD 0.2

/* Tracked objects in the infer history map will be removed if they have not
 * been accessed for at least this number of frames. The tracker would definitely
 * have dropped references to an unseen object by 150 frames. */
#define CLEANUP_ACCESS_CRITERIA 150

/* Object history map cleanup interval. 1800 frames is a minute with a 30fps input */
#define MAP_CLEANUP_INTERVAL 1800

#define PROCESS_MODEL_FULL_FRAME 1
#define PROCESS_MODEL_OBJECTS 2

/* Warn about untracked objects in async mode every 5 minutes. */
#define UNTRACKED_OBJECT_WARN_INTERVAL (GST_SECOND * 60 * 5)

#define MIN_INPUT_OBJECT_WIDTH 1
#define MIN_INPUT_OBJECT_HEIGHT 1

/* Default source width and height for warmup. */
#define NVINFER_WARMUP_DEFAULT_SRC_WIDTH  1920
#define NVINFER_WARMUP_DEFAULT_SRC_HEIGHT 1080

extern const int DEFAULT_REINFER_INTERVAL = G_MAXINT;

#define DS_NVINFER_IMPL(gst_nvinfer) reinterpret_cast<DsNvInferImpl*>((gst_nvinfer)->impl)

#define IS_DETECTOR_INSTANCE(nvinfer) \
  (DS_NVINFER_IMPL(nvinfer)->m_InitParams->networkType == NvDsInferNetworkType_Detector)
#define IS_CLASSIFIER_INSTANCE(nvinfer) \
  (DS_NVINFER_IMPL(nvinfer)->m_InitParams->networkType == NvDsInferNetworkType_Classifier)
#define IS_SEGMENTATION_INSTANCE(nvinfer) \
  (DS_NVINFER_IMPL(nvinfer)->m_InitParams->networkType == NvDsInferNetworkType_Segmentation)
#define IS_INSTANCE_SEGMENTATION_INSTANCE(nvinfer) \
  (DS_NVINFER_IMPL(nvinfer)->m_InitParams->networkType == NvDsInferNetworkType_InstanceSegmentation)

static GQuark _dsmeta_quark = 0;

/* Gst-nvinfer supports runtime model updates. Refer to gstnvinfer_impl.h
 * for details. */

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESS_MODE PROCESS_MODEL_FULL_FRAME
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_INTERVAL 0
#define DEFAULT_OPERATE_ON_GIE_ID -1
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_OUTPUT_WRITE_TO_FILE FALSE
#define DEFAULT_OUTPUT_TENSOR_META FALSE
#define DEFAULT_OUTPUT_INSTANCE_MASK FALSE
#define DEFAULT_INPUT_TENSOR_META FALSE
#define DEFAULT_CLIP_OBJECT_OUTSIDE_ROI TRUE
#define DEFAULT_CLIP_OBJECT_TO_ROI_BOUNDARY FALSE
#define DEFAULT_STRONGLY_TYPED FALSE
#define DEFAULT_DYNAMIC_PROPERTIES ""

#define NVTX_DEEPBLUE_COLOR 0xFF667EBE

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvinfer_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvinfer_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

guint gst_nvinfer_signals[LAST_SIGNAL] = { 0 };

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvinfer_parent_class parent_class
G_DEFINE_TYPE (GstNvInfer, gst_nvinfer, GST_TYPE_BASE_TRANSFORM);

/* Implementation of the GObject/GstBaseTransform interfaces. */
static void gst_nvinfer_finalize (GObject * object);
static void gst_nvinfer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvinfer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_nvinfer_parse_dynamic_properties(GstNvInfer *nvinfer, const gchar *properties_string);

static gboolean gst_nvinfer_start (GstBaseTransform * btrans);
static gboolean gst_nvinfer_stop (GstBaseTransform * btrans);
static gboolean gst_nvinfer_sink_event (GstBaseTransform * trans,
    GstEvent * event);

static GstFlowReturn gst_nvinfer_submit_input_buffer (GstBaseTransform *
    btrans, gboolean discont, GstBuffer * inbuf);
static GstFlowReturn gst_nvinfer_generate_output (GstBaseTransform *
    btrans, GstBuffer ** outbuf);

static gpointer gst_nvinfer_input_queue_loop (gpointer data);
static gpointer gst_nvinfer_output_loop (gpointer data);

static void gst_nvinfer_reset_init_params (GstNvInfer * nvinfer);

/* Create enum type for the process mode property. */
#define GST_TYPE_NVDSINFER_PROCESS_MODE (gst_nvinfer_process_mode_get_type ())

static GType
gst_nvinfer_process_mode_get_type (void)
{
  static gsize process_mode_type = 0;
  static const GEnumValue process_mode[] = {
    {PROCESS_MODEL_FULL_FRAME, "Primary (Full Frame)", "primary"},
    {PROCESS_MODEL_OBJECTS, "Secondary (Objects)", "secondary"},
    {0, nullptr, nullptr}
  };

  if (g_once_init_enter (&process_mode_type)) {
    GType tmp = g_enum_register_static ("GstNvInferProcessModeType",
        process_mode);
    g_once_init_leave (&process_mode_type, tmp);
  }

  return (GType) process_mode_type;
}

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
gst_nvinfer_class_init (GstNvInferClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvinfer_finalize);
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvinfer_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvinfer_get_property);

  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvinfer_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvinfer_stop);
  gstbasetransform_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvinfer_sink_event);

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

  g_object_class_install_property (gobject_class, PROP_PROCESS_MODE,
      g_param_spec_enum ("process-mode", "Process Mode",
          "Infer processing mode", GST_TYPE_NVDSINFER_PROCESS_MODE,
          DEFAULT_PROCESS_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Config File Path",
          "Path to the configuration file for this instance of nvinfer",
          DEFAULT_CONFIG_FILE_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for inference",
          1, NVDSINFER_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_INTERVAL,
      g_param_spec_uint ("interval", "Interval",
          "Specifies number of consecutive batches to be skipped for inference",
          0, G_MAXINT, DEFAULT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_GIE_ID,
      g_param_spec_int ("infer-on-gie-id", "Infer on Gie ID",
          "Infer on metadata generated by GIE with this unique ID.\n"
          "\t\t\tSet to -1 to infer on all metadata.",
          -1, G_MAXINT, DEFAULT_OPERATE_ON_GIE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_CLASS_IDS,
      g_param_spec_string ("infer-on-class-ids", "Operate on Class ids",
          "Operate on objects with specified class ids\n"
          "\t\t\tUse string with values of class ids in ClassID (int) to set the property.\n"
          "\t\t\t e.g. 0:2:3",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property(gobject_class, PROP_FILTER_OUT_CLASS_IDS,
      g_param_spec_string ("filter-out-class-ids", "Ignore metadata for class ids",
            "Ignore metadata for objects of specified class ids\n"
            "\t\t\tUse string with values of class ids in ClassID (int) to set the property.\n"
            "\t\t\t e.g. 0;2;3",
            "",
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

  g_object_class_install_property (gobject_class, PROP_OUTPUT_INSTANCE_MASK,
      g_param_spec_boolean ("output-instance-mask", "Output Instance Mask",
          "Instance mask expected in network output and attach it to metadata",
          DEFAULT_OUTPUT_INSTANCE_MASK,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_INPUT_TENSOR_META,
      g_param_spec_boolean ("input-tensor-meta", "Input Tensor Meta",
          "Use preprocessed input tensors attached as metadata instead of preprocessing inside the plugin",
          DEFAULT_INPUT_TENSOR_META,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CLIP_OBJECT_OUTSIDE_ROI,
      g_param_spec_boolean ("clip-object-outside-roi", "Clip Object Outside Roi",
          "Clip the object bounding-box which lies outside the roi specified by nvdspreprosess plugin",
          DEFAULT_CLIP_OBJECT_OUTSIDE_ROI,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CROP_OBJECTS_TO_ROI_BOUNDARY,
      g_param_spec_boolean ("crop-objects-to-roi-boundary", "Crop Object to Roi Boundary",
          "Clip the object bounding-box which lies outside the roi boundary",
          DEFAULT_CLIP_OBJECT_TO_ROI_BOUNDARY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_STRONGLY_TYPED,
      g_param_spec_boolean ("strongly-typed", "Strongly Typed",
          "Enable strongly typed network mode for enhanced type inference. "
          "Corresponds to --stronglyTyped flag in trtexec.",
          DEFAULT_STRONGLY_TYPED,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_DYNAMIC_PROPERTIES,
      g_param_spec_string ("dynamic-properties", "Dynamic Properties",
          "Dynamic properties for TensorRT/trtexec flags as key=value pairs separated by semicolons. "
          "Example: 'data_max=100;init_max=200;calibration_data=/path/to/data'",
          DEFAULT_DYNAMIC_PROPERTIES,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  /** install signal MODEL_UPDATED */
  gst_nvinfer_signals[SIGNAL_MODEL_UPDATED] =
      g_signal_new ("model-updated",
      G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstNvInferClass, model_updated),
      NULL, NULL, NULL,
      G_TYPE_NONE, 2, G_TYPE_INT, G_TYPE_STRING);

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvinfer_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvinfer_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvInfer plugin",
      "NvInfer Plugin",
      "Nvidia DeepStreamSDK TensorRT plugin",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_nvinfer_init (GstNvInfer * nvinfer)
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
  nvinfer->process_full_frame = DEFAULT_PROCESS_MODE;
  nvinfer->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);
  nvinfer->operate_on_class_ids = new std::vector < gboolean >;
  nvinfer->filter_out_class_ids = new std::set<uint>;
  nvinfer->output_tensor_meta = DEFAULT_OUTPUT_TENSOR_META;
  nvinfer->output_instance_mask = DEFAULT_OUTPUT_INSTANCE_MASK;
  nvinfer->clip_object_outside_roi = DEFAULT_CLIP_OBJECT_OUTSIDE_ROI;
  nvinfer->crop_objects_to_roi_boundary = DEFAULT_CLIP_OBJECT_TO_ROI_BOUNDARY;
  nvinfer->dynamic_properties_string = g_strdup (DEFAULT_DYNAMIC_PROPERTIES);
  nvinfer->stabilize_buffer_count = 0;

  nvinfer->max_batch_size = impl->m_InitParams->maxBatchSize =
      DEFAULT_BATCH_SIZE;
  nvinfer->interval = DEFAULT_INTERVAL;
  nvinfer->operate_on_gie_id = DEFAULT_OPERATE_ON_GIE_ID;
  nvinfer->gpu_id = impl->m_InitParams->gpuID = DEFAULT_GPU_DEVICE_ID;
  nvinfer->is_prop_set = new std::vector < gboolean > (PROP_LAST, FALSE);

  nvinfer->untracked_object_warn_pts = GST_CLOCK_TIME_NONE;

  /* Set the default pre-processing transform params. */
  nvinfer->transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  nvinfer->transform_params.transform_filter = NvBufSurfTransformInter_Default;

  /* Create processing lock and condition for synchronization.*/
  g_mutex_init (&nvinfer->process_lock);
  g_cond_init (&nvinfer->process_cond);

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Free resources allocated during init. */
static void
gst_nvinfer_finalize (GObject * object)
{
  GstNvInfer *nvinfer = GST_NVINFER (object);

  g_mutex_clear (&nvinfer->process_lock);
  g_cond_clear (&nvinfer->process_cond);

  delete nvinfer->perClassDetectionFilterParams;
  delete nvinfer->perClassColorParams;
  delete nvinfer->is_prop_set;
  g_free (nvinfer->config_file_path);
  delete nvinfer->operate_on_class_ids;
  delete nvinfer->filter_out_class_ids;

  delete DS_NVINFER_IMPL(nvinfer);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvinfer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvInfer *nvinfer = GST_NVINFER (object);
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
    case PROP_PROCESS_MODE:
    {
      guint val = g_value_get_enum (value);
      nvinfer->process_full_frame = (val == PROCESS_MODEL_FULL_FRAME);
    }
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
        if (g_str_has_suffix(nvinfer->config_file_path, ".yml") ||
                g_str_has_suffix(nvinfer->config_file_path, ".yaml")) {
          nvinfer->config_file_parse_successful =
              gst_nvinfer_parse_config_file_yaml (nvinfer, impl->m_InitParams.get(),
                  nvinfer->config_file_path);
        } else {
          nvinfer->config_file_parse_successful =
              gst_nvinfer_parse_config_file (nvinfer, impl->m_InitParams.get(),
                  nvinfer->config_file_path);
        }
      }
      break;
    case PROP_OPERATE_ON_GIE_ID:
      nvinfer->operate_on_gie_id = g_value_get_int (value);
      break;
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str (g_value_get_string (value));
      std::vector < gint > class_ids;
      gint max_class_id = -1;

      while (str.peek () != EOF) {
        gint class_id;
        str >> class_id;
        if(class_id < 0)
          continue;
        class_ids.push_back (class_id);
        max_class_id = MAX (max_class_id, class_id);
        str.get ();
      }
      nvinfer->operate_on_class_ids->assign (max_class_id + 1, FALSE);
    for (auto & cid:class_ids)
        nvinfer->operate_on_class_ids->at (cid) = TRUE;
    }
      break;
    case PROP_FILTER_OUT_CLASS_IDS:
    {
        std::stringstream str(g_value_get_string(value));
        nvinfer->filter_out_class_ids->clear();
        while(str.peek() != EOF) {
            gint class_id;
            str >> class_id;
            nvinfer->filter_out_class_ids->insert(class_id);
            str.get();
        }
    }
      break;
    case PROP_BATCH_SIZE:
      nvinfer->max_batch_size = impl->m_InitParams->maxBatchSize =
          g_value_get_uint (value);
      break;
    case PROP_INTERVAL:
      nvinfer->interval = g_value_get_uint (value);
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
    case PROP_OUTPUT_INSTANCE_MASK:
      nvinfer->output_instance_mask = g_value_get_boolean (value);
      break;
    case PROP_INPUT_TENSOR_META:
      nvinfer->input_tensor_from_meta = g_value_get_boolean (value);
      impl->m_InitParams->inputFromPreprocessedTensor =
          g_value_get_boolean (value);
      break;
    case PROP_CLIP_OBJECT_OUTSIDE_ROI:
      nvinfer->clip_object_outside_roi = g_value_get_boolean (value);
      break;
    case PROP_CROP_OBJECTS_TO_ROI_BOUNDARY:
      nvinfer->crop_objects_to_roi_boundary = g_value_get_boolean (value);
      break;
    case PROP_STRONGLY_TYPED:
      impl->m_InitParams->useStronglyTyped = g_value_get_boolean (value);
      break;
    case PROP_DYNAMIC_PROPERTIES:
      g_free (nvinfer->dynamic_properties_string);
      nvinfer->dynamic_properties_string = g_value_dup_string (value);
      /* Parse dynamic properties and apply to init params */
      gst_nvinfer_parse_dynamic_properties (nvinfer, nvinfer->dynamic_properties_string);
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
  GstNvInfer *nvinfer = GST_NVINFER (object);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, nvinfer->unique_id);
      break;
    case PROP_PROCESS_MODE:
      g_value_set_enum (value,
          nvinfer->process_full_frame ? PROCESS_MODEL_FULL_FRAME :
          PROCESS_MODEL_OBJECTS);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, nvinfer->config_file_path);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      g_value_set_int (value, nvinfer->operate_on_gie_id);
      break;
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str;
      for (size_t i = 0; i < nvinfer->operate_on_class_ids->size (); i++) {
        if (nvinfer->operate_on_class_ids->at (i))
          str << i << ":";
      }
      g_value_set_string (value, str.str ().c_str ());
    }
      break;
    case PROP_FILTER_OUT_CLASS_IDS:
    {
        std::stringstream str;
        for(const auto id : *nvinfer->filter_out_class_ids)
            str << id << ";";
        g_value_set_string (value, str.str ().c_str ());
    }
        break;
    case PROP_MODEL_ENGINEFILE:
      g_value_set_string (value, impl->m_InitParams->modelEngineFilePath);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, nvinfer->max_batch_size);
      break;
    case PROP_INTERVAL:
      g_value_set_uint (value, nvinfer->interval);
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
    case PROP_OUTPUT_INSTANCE_MASK:
      g_value_set_boolean (value, nvinfer->output_instance_mask);
      break;
    case PROP_INPUT_TENSOR_META:
      g_value_set_boolean (value, nvinfer->input_tensor_from_meta);
      break;
    case PROP_CLIP_OBJECT_OUTSIDE_ROI:
      g_value_set_boolean (value, nvinfer->clip_object_outside_roi);
      break;
    case PROP_CROP_OBJECTS_TO_ROI_BOUNDARY:
      g_value_set_boolean (value, nvinfer->crop_objects_to_roi_boundary);
      break;
    case PROP_STRONGLY_TYPED:
      g_value_set_boolean (value, impl->m_InitParams->useStronglyTyped);
      break;
    case PROP_DYNAMIC_PROPERTIES:
      g_value_set_string (value, nvinfer->dynamic_properties_string);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

void gst_nvinfer_logger(NvDsInferContextHandle handle, unsigned int unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx) {
    GstNvInfer* nvinfer = GST_NVINFER(user_ctx);

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
gst_nvinfer_reset_init_params (GstNvInfer * nvinfer)
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

  if (nvinfer->is_prop_set->at (PROP_UNIQUE_ID))
    impl->m_InitParams->uniqueID = prev_params->uniqueID;
  if (nvinfer->is_prop_set->at (PROP_INPUT_TENSOR_META))
    impl->m_InitParams->inputFromPreprocessedTensor =
        prev_params->inputFromPreprocessedTensor;
  if (nvinfer->is_prop_set->at (PROP_STRONGLY_TYPED))
    impl->m_InitParams->useStronglyTyped = prev_params->useStronglyTyped;

  delete prev_params->perClassDetectionParams;
  g_strfreev (prev_params->outputLayerNames);
  g_strfreev (prev_params->outputIOFormats);
  g_strfreev (prev_params->layerDevicePrecisions);
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
  GstNvInfer *nvinfer = GST_NVINFER (trans);
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
      !nvinfer->classifier_async_mode) {
    GstNvInferBatch *batch = new GstNvInferBatch;
    batch->event_marker = TRUE;

    g_mutex_lock (&nvinfer->process_lock);
    /* Push the event marker batch in the processing queue. */
    if (nvinfer->input_queue_thread)
      g_queue_push_tail (nvinfer->input_queue, batch);
    else
      g_queue_push_tail (nvinfer->process_queue, batch);
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

  if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_ADDED) {
    /* New source added in the pipeline. Create a source info instance for it. */
    guint source_id = 0;
    gst_nvevent_parse_pad_added (event, &source_id);
    nvinfer->source_info->emplace (source_id, GstNvInferSourceInfo ());
  }

  if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_DELETED) {
    /* Source removed from the pipeline. Remove the related structure. */
    guint source_id = 0;
    gst_nvevent_parse_pad_deleted (event, &source_id);
    nvinfer->source_info->erase (source_id);
  }

  if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_STREAM_EOS) {
    /* Got EOS from a source. Clean up the object history map. */
    guint source_id = 0;
    gst_nvevent_parse_stream_eos (event, &source_id);
    auto result = nvinfer->source_info->find (source_id);
    if (result != nvinfer->source_info->end ())
      result->second.object_history_map.clear ();
  }

  if ((GstNvDsCustomEventType)GST_EVENT_TYPE (event) == GST_NVEVENT_INFER_INTERVAL_UPDATE) {
    gchar* stream_id = NULL;

    g_mutex_lock (&nvinfer->process_lock);
    gst_nvevent_parse_infer_interval_update (event, &stream_id, &nvinfer->interval);
    g_mutex_unlock (&nvinfer->process_lock);
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
    nvinfer->interval_counter = 0;
  }

  /* Call the sink event handler of the base class. */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * Run NvBufSurfTransform warmup when warmup-engine is enabled. Runs dummy
 * transforms for batch sizes 1 to max_batch_size so JIT/caches are warm.
 * The batched API expects src_rect/dst_rect to be arrays of size numFilled;
 * we allocate rect arrays of size max_batch_size and pass them so all batch
 * sizes are valid. Source at input resolution (from sink caps or default),
 * destination at network resolution. Non-fatal on failure per batch.
 *
 * \param nvinfer The nvinfer element.
 * \param color_format Destination surface color format (network input).
 * \param src_width Source width (0 to use default 1920).
 * \param src_height Source height (0 to use default 1080).
 */
static void
gst_nvinfer_warmup_nvbuf_transform (GstNvInfer *nvinfer,
    NvBufSurfaceColorFormat color_format, guint src_width, guint src_height)
{
  NvBufSurface *warmup_src = nullptr;
  NvBufSurface *warmup_dst = nullptr;
  guint max_bs = nvinfer->max_batch_size;

  if (max_bs == 0 ||
      nvinfer->network_width <= 0 || nvinfer->network_height <= 0) {
    GST_WARNING_OBJECT (nvinfer,
        "NvBufSurfTransform warmup skipped (invalid batch/dimensions)");
    return;
  }

  if (src_width == 0)
    src_width = NVINFER_WARMUP_DEFAULT_SRC_WIDTH;
  if (src_height == 0)
    src_height = NVINFER_WARMUP_DEFAULT_SRC_HEIGHT;

  /* Batched API expects src_rect/dst_rect to point to arrays of size
   * numFilled. Allocate one rect per possible batch index. */
  std::vector<NvBufSurfTransformRect> src_rects (max_bs);
  std::vector<NvBufSurfTransformRect> dst_rects (max_bs);
  for (guint i = 0; i < max_bs; i++) {
    src_rects[i] = {0, 0, src_width, src_height};
    dst_rects[i] = {0, 0, (guint) nvinfer->network_width,
        (guint) nvinfer->network_height};
  }

  /* Source at input resolution (from sink caps or default). */
  NvBufSurfaceCreateParams src_create_params = {0};
  src_create_params.gpuId = nvinfer->gpu_id;
  src_create_params.width = src_width;
  src_create_params.height = src_height;
  src_create_params.size = 0;
  src_create_params.isContiguous = 1;
  src_create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  src_create_params.layout = NVBUF_LAYOUT_PITCH;
  src_create_params.memType = NVBUF_MEM_CUDA_DEVICE;

  /* Destination at network resolution (dynamic from config). */
  NvBufSurfaceCreateParams dst_create_params = {0};
  dst_create_params.gpuId = nvinfer->gpu_id;
  dst_create_params.width = nvinfer->network_width;
  dst_create_params.height = nvinfer->network_height;
  dst_create_params.size = 0;
  dst_create_params.isContiguous = 1;
  dst_create_params.colorFormat = color_format;
  dst_create_params.layout = NVBUF_LAYOUT_PITCH;
  dst_create_params.memType = NVBUF_MEM_CUDA_DEVICE;

  NvBufSurfTransformSetSessionParams (&nvinfer->transform_config_params);

  NvBufSurfTransformParams warmup_xform_params;
  memset (&warmup_xform_params, 0, sizeof (warmup_xform_params));
  warmup_xform_params.src_rect = src_rects.data ();
  warmup_xform_params.dst_rect = dst_rects.data ();
  warmup_xform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  warmup_xform_params.transform_filter = NvBufSurfTransformInter_Default;

  for (guint bs = 1; bs <= max_bs; bs++)
  {
    gint src_ok = NvBufSurfaceCreate (&warmup_src, bs, &src_create_params);
    gint dst_ok = NvBufSurfaceCreate (&warmup_dst, bs, &dst_create_params);

    if (src_ok != 0 || dst_ok != 0 || !warmup_src || !warmup_dst) {
      GST_WARNING_OBJECT (nvinfer,
          "NvBufSurfTransform warmup: surface create failed for batch %u "
          "(non-fatal)", bs);
      if (warmup_src) {
        NvBufSurfaceDestroy (warmup_src);
        warmup_src = nullptr;
      }
      if (warmup_dst) {
        NvBufSurfaceDestroy (warmup_dst);
        warmup_dst = nullptr;
      }
      continue;
    }

    warmup_src->numFilled = bs;
    warmup_dst->numFilled = bs;
    NvBufSurfaceMemSet (warmup_src, 0, 0, 0);

    NvBufSurfTransformSyncObj_t sync_obj = nullptr;
    NvBufSurfTransform_Error xform_err =
        NvBufSurfTransformAsync (warmup_src, warmup_dst,
            &warmup_xform_params, &sync_obj);
    if (xform_err == NvBufSurfTransformError_Success && sync_obj) {
      NvBufSurfTransformSyncObjWait (sync_obj, -1);
      NvBufSurfTransformSyncObjDestroy (&sync_obj);
    } else if (xform_err != NvBufSurfTransformError_Success) {
      if (bs == 1) {
        GST_WARNING_OBJECT (nvinfer,
            "NvBufSurfTransform warmup failed for batch 1 (error %d, non-fatal)",
            xform_err);
      } else {
        GST_WARNING_OBJECT (nvinfer,
            "NvBufSurfTransform warmup failed for batch sizes %u to %u "
            "(error %d, non-fatal). Only batch 1 was warmed.",
            bs, max_bs, xform_err);
        NvBufSurfaceDestroy (warmup_src);
        NvBufSurfaceDestroy (warmup_dst);
        warmup_src = nullptr;
        warmup_dst = nullptr;
        break;
      }
    }

    NvBufSurfaceDestroy (warmup_src);
    NvBufSurfaceDestroy (warmup_dst);
    warmup_src = nullptr;
    warmup_dst = nullptr;
  }

  GST_INFO_OBJECT (nvinfer, "NvBufSurfTransform warmup completed");
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvinfer_start (GstBaseTransform * btrans)
{
  GstNvInfer *nvinfer = GST_NVINFER (btrans);
  GstAllocationParams allocation_params;
  cudaError_t cudaReturn;
  NvBufSurfaceColorFormat color_format;
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

  if (nvinfer->output_instance_mask == TRUE &&
                 init_params->clusterMode != NVDSINFER_CLUSTER_NONE)
  {
    GST_ELEMENT_ERROR (nvinfer, LIBRARY, SETTINGS,
        ("Instance mask output not supported with cluster mode %d",
                               init_params->clusterMode), (nullptr));
    return FALSE;
  }

  nvinfer->interval_counter = 0;
  /* Should not infer on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT. */
  nvinfer->min_input_object_width =
      MAX(MIN_INPUT_OBJECT_WIDTH, nvinfer->min_input_object_width);
  nvinfer->min_input_object_height =
      MAX(MIN_INPUT_OBJECT_HEIGHT, nvinfer->min_input_object_height);

  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      is_nvgpu = 1;
    }
  }
  if (is_nvgpu) {
    if (nvinfer->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC ||
        nvinfer->transform_config_params.compute_mode == NvBufSurfTransformCompute_Default)
    {
      g_print("Setting min object dimensions as 16x16 instead of 1x1 to support VIC compute mode.\n");
      nvinfer->min_input_object_width =
          MAX(16, nvinfer->min_input_object_width);
      nvinfer->min_input_object_height =
          MAX(16, nvinfer->min_input_object_height);
    }
  }

  /* Ask NvDsInferContext to copy the input layer contents to host memory if
   * CPU needs to access it. */
  init_params->copyInputToHostBuffers =
      (nvinfer->write_raw_buffers_to_file ||
      (nvinfer->output_generated_callback != nullptr));

  /* Set the number of output buffers that should be allocated by NvDsInferContext.
   * Should allocate more buffers if the output tensor buffers will be attached
   * as meta to GstBuffers and pushed downstream. */

  init_params->outputBufferPoolSize = std::max<uint>(init_params->outputBufferPoolSize,
                                        NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE);

  if (nvinfer->output_tensor_meta || IS_SEGMENTATION_INSTANCE (nvinfer))
    init_params->outputBufferPoolSize = std::max<uint>(init_params->outputBufferPoolSize,
                                        NVDSINFER_CTX_OUT_POOL_SIZE_FLOW_META);


  if (nvinfer->output_tensor_meta && init_params->autoIncMem == 0) {
      GST_ELEMENT_WARNING (nvinfer, LIBRARY, SETTINGS,
          ("NvInfer output-tensor-meta is enabled but init_params auto "
              "increase memory (auto-inc-mem) is disabled. The bufferpool "
              "will not be automatically resized."), (nullptr));
    }

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

  /* Set the NvBufSurfTransform config parameters. */
  nvinfer->transform_config_params.gpu_id = nvinfer->gpu_id;

  NvBufSurfTransformSetSessionParams (&nvinfer->transform_config_params);

  /* Based on the network input requirements decide the buffer pool color format. */
  switch (init_params->networkInputFormat) {
    case NvDsInferFormat_RGB:
    case NvDsInferFormat_BGR:
    if(nvinfer->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC) {
      color_format = NVBUF_COLOR_FORMAT_RGBA;
    }
    else {
      color_format = NVBUF_COLOR_FORMAT_RGB;
    }
    break;
    case NvDsInferFormat_GRAY:
    if(nvinfer->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC) {
      color_format = NVBUF_COLOR_FORMAT_NV12;
    }
    else {
      color_format = NVBUF_COLOR_FORMAT_GRAY8;
    }
    break;
    default:
      GST_ELEMENT_ERROR (nvinfer, LIBRARY, SETTINGS,
          ("Unsupported network input format: %d",
              init_params->networkInputFormat), (nullptr));
      return FALSE;
  }

  if (!nvinfer->input_tensor_from_meta) {
    /* Create a buffer pool for internal memory required for scaling frames to
    * network resolution / cropping objects. The pool allocates
    * INTERNAL_BUF_POOL_SIZE buffers at start and keeps reusing them. */
    auto pool_deleter = [](GstBufferPool *p) { if (p) gst_object_unref (p); };
    std::unique_ptr<GstBufferPool, decltype(pool_deleter)> pool_ptr (
        gst_buffer_pool_new (), pool_deleter);

    auto config_deleter = [](GstStructure *s) { if (s) gst_structure_free (s); };
    std::unique_ptr<GstStructure, decltype(config_deleter)> config_ptr (
        gst_buffer_pool_get_config (pool_ptr.get()), config_deleter);
    gst_buffer_pool_config_set_params (config_ptr.get(), nullptr,
        sizeof (GstNvInferMemory), INTERNAL_BUF_POOL_SIZE, INTERNAL_BUF_POOL_SIZE);

    /* Create a new GstNvInferAllocator instance. Allocator has methods to allocate
    * and free custom memories. */
    auto allocator_deleter = [](GstAllocator *a) { if (a) gst_object_unref (a); };
    std::unique_ptr<GstAllocator, decltype(allocator_deleter)> allocator_ptr (
        gst_nvinfer_allocator_new (nvinfer->network_width,
        nvinfer->network_height, color_format, nvinfer->max_batch_size,
        nvinfer->gpu_id),
        allocator_deleter);
    memset (&allocation_params, 0, sizeof (allocation_params));
    gst_buffer_pool_config_set_allocator (config_ptr.get (), allocator_ptr.get (),
        &allocation_params);

    if (!gst_buffer_pool_set_config (pool_ptr.get(), config_ptr.release())) {
      GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
          ("Failed to set config on buffer pool"), (nullptr));
      return FALSE;
    }

    /* Start the buffer pool and allocate all internal buffers. */
    if (!gst_buffer_pool_set_active (pool_ptr.get(), TRUE)) {
      GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
          ("Failed to set buffer pool to active"), (nullptr));
      return FALSE;
    }
    nvinfer->pool = pool_ptr.release ();
  }

  cudaReturn = cudaSetDevice (nvinfer->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinfer->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
    return FALSE;
  }

  cudaReturn =
      cudaStreamCreateWithFlags (&nvinfer->convertStream,
      cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to create cuda stream"),
        ("cudaStreamCreateWithFlags failed with error %s",
            cudaGetErrorName (cudaReturn)));
    return FALSE;
  }

  nvinfer->transform_config_params.cuda_stream = nvinfer->convertStream;

  /* Create the intermediate NvBufSurface structure for holding an array of input
   * NvBufSurfaceParams for batched transforms. */
  nvinfer->tmp_surf.surfaceList = new NvBufSurfaceParams[nvinfer->max_batch_size];
  nvinfer->tmp_surf.batchSize = nvinfer->max_batch_size;
  nvinfer->tmp_surf.gpuId = nvinfer->gpu_id;

  /* Set up the NvBufSurfTransformParams structure for batched transforms. */
  nvinfer->transform_params.src_rect =
      new NvBufSurfTransformRect[nvinfer->max_batch_size];
  nvinfer->transform_params.dst_rect =
      new NvBufSurfTransformRect[nvinfer->max_batch_size];
  nvinfer->transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST | NVBUFSURF_TRANSFORM_FLIP |
      NVBUFSURF_TRANSFORM_ALLOW_ODD_CROP;
  nvinfer->transform_params.transform_flip = NvBufSurfTransform_None;

  if (init_params->warmupEngine) {
    guint warmup_src_width = 0, warmup_src_height = 0;
    GstCaps *sink_caps =
        gst_pad_get_current_caps (GST_BASE_TRANSFORM_SINK_PAD (btrans));
    if (sink_caps && gst_caps_get_size (sink_caps) > 0) {
      GstStructure *s = gst_caps_get_structure (sink_caps, 0);
      gint w = 0, h = 0;
      if (s && gst_structure_get_int (s, "width", &w) &&
          gst_structure_get_int (s, "height", &h) && w > 0 && h > 0) {
        warmup_src_width = (guint) w;
        warmup_src_height = (guint) h;
      }
      gst_caps_unref (sink_caps);
    }
    gst_nvinfer_warmup_nvbuf_transform (nvinfer, color_format,
        warmup_src_width, warmup_src_height);
  }

  /* Initialize the object history map for source 0. */
  nvinfer->source_info = new std::unordered_map < gint, GstNvInferSourceInfo >;
  nvinfer->source_info->emplace (0, GstNvInferSourceInfo {
      GstNvInferObjectHistoryMap (), 0}
  );

  if (nvinfer->classifier_async_mode) {
    if (nvinfer->process_full_frame || !IS_CLASSIFIER_INSTANCE (nvinfer)) {
      GST_ELEMENT_WARNING (nvinfer, LIBRARY, SETTINGS,
          ("NvInfer asynchronous mode is applicable for secondary"
              "classifiers only. Turning off asynchronous mode"), (nullptr));
      nvinfer->classifier_async_mode = FALSE;
    }
  }

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  nvinfer->output_thread =
      g_thread_new ("nvinfer-output-thread", gst_nvinfer_output_loop, nvinfer);

  /* Start a thread which will queue input to the NvDsInfer context since
   * queueInputBatch is a blocking function. This is done to parallelize
   * input conversion and queueInputBatch. */
  if (!nvinfer->input_tensor_from_meta) {
    nvinfer->input_queue_thread =
        g_thread_new ("nvinfer-input-queue-thread", gst_nvinfer_input_queue_loop,
            nvinfer);
  }

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
  GstNvInfer *nvinfer = GST_NVINFER (btrans);
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

  if (nvinfer->input_queue_thread)
    g_thread_join (nvinfer->input_queue_thread);
  g_thread_join (nvinfer->output_thread);

  nvinfer->stop = FALSE;

  delete nvinfer->source_info;
  delete nvinfer->layers_info;
  delete nvinfer->output_layers_info;

  delete[] nvinfer->transform_params.src_rect;
  delete[] nvinfer->transform_params.dst_rect;
  delete[] nvinfer->tmp_surf.surfaceList;

  cudaSetDevice (nvinfer->gpu_id);

  if (nvinfer->convertStream)
    cudaStreamDestroy (nvinfer->convertStream);

  /* Free up the memory allocated by pool. */
  if (nvinfer->pool)
    gst_object_unref (nvinfer->pool);

  g_queue_free (nvinfer->process_queue);
  g_queue_free (nvinfer->input_queue);

  return TRUE;
}

/**
 * Calls the one of the required conversion functions based on the network
 * input format.
 */
static GstFlowReturn
get_converted_buffer (GstNvInfer * nvinfer, NvBufSurface * src_surf,
    NvBufSurfaceParams * src_frame, NvOSD_RectParams * crop_rect_params,
    NvBufSurface * dest_surf, NvBufSurfaceParams * dest_frame,
    gdouble & ratio_x, gdouble & ratio_y, guint & offset_left,
    guint & offset_top, void *destCudaPtr)
{
  guint src_left = GST_ROUND_UP_2 ((unsigned int)crop_rect_params->left);
  guint src_top = GST_ROUND_UP_2 ((unsigned int)crop_rect_params->top);
  guint src_width = GST_ROUND_DOWN_2 ((unsigned int)crop_rect_params->width);
  guint src_height = GST_ROUND_DOWN_2 ((unsigned int)crop_rect_params->height);
  guint dest_width, dest_height;

  guint offset_right = 0, offset_bottom = 0;
  offset_left = 0;
  offset_top = 0;
  if (nvinfer->maintain_aspect_ratio) {
    /* Calculate the destination width and height required to maintain
     * the aspect ratio. */
    double hdest = dest_frame->width * src_height / (double) src_width;
    double wdest = dest_frame->height * src_width / (double) src_height;
    int pixel_size;
    cudaError_t cudaReturn;

    if (hdest <= dest_frame->height) {
      dest_width = dest_frame->width;
      dest_height = hdest;
    } else {
      dest_width = wdest;
      dest_height = dest_frame->height;
    }

    switch (dest_frame->colorFormat) {
      case NVBUF_COLOR_FORMAT_RGBA:
        pixel_size = 4;
        break;
      case NVBUF_COLOR_FORMAT_RGB:
        pixel_size = 3;
        break;
      case NVBUF_COLOR_FORMAT_GRAY8:
      case NVBUF_COLOR_FORMAT_NV12:
        pixel_size = 1;
        break;
      default:
        g_assert_not_reached ();
        break;
    }
    /* Pad the scaled image with black color. */
    if (!nvinfer->symmetric_padding) {
      /* Right-Bottom Padding. */
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr + pixel_size * dest_width,
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * (dest_frame->width - dest_width), dest_frame->height,
          nvinfer->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinfer,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr +
          dest_frame->planeParams.pitch[0] * dest_height,
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          dest_frame->height - dest_height, nvinfer->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinfer,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
    } else {
      /* Symmetric Padding. */
      offset_left = (dest_frame->width - dest_width) / 2;
      offset_right = dest_frame->width - dest_width - offset_left;

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr,
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_left, dest_frame->height, nvinfer->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinfer,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr + pixel_size *
          (dest_width + offset_left),
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_right, dest_frame->height,
          nvinfer->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinfer,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }

      offset_top = (dest_frame->height - dest_height) / 2;
      offset_bottom = dest_frame->height - dest_height - offset_top;

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr,
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_top, nvinfer->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinfer,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr +
          dest_frame->planeParams.pitch[0] * (dest_height + offset_top),
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_bottom, nvinfer->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinfer,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
    }
  } else {
    dest_width = nvinfer->network_width;
    dest_height = nvinfer->network_height;
  }
  /* Calculate the scaling ratio of the frame / object crop. This will be
   * required later for rescaling the detector output boxes to input resolution.
   */
  ratio_x = (double) dest_width / src_width;
  ratio_y = (double) dest_height / src_height;

  /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
  nvinfer->tmp_surf.surfaceList[nvinfer->tmp_surf.numFilled] = *src_frame;

  /* Set the source ROI. Could be entire frame or an object. */
  nvinfer->transform_params.src_rect[nvinfer->tmp_surf.numFilled] =
      { src_top, src_left, src_width, src_height };
  /* Set the dest ROI. Could be the entire destination frame or part of it to
   * maintain aspect ratio. */
  if (!nvinfer->symmetric_padding) {
    nvinfer->transform_params.dst_rect[nvinfer->tmp_surf.numFilled] =
        { 0, 0, dest_width, dest_height };
  } else {
    nvinfer->transform_params.dst_rect[nvinfer->tmp_surf.numFilled] =
        { offset_top, offset_left, dest_width, dest_height };
  }

  nvinfer->tmp_surf.numFilled++;
  nvinfer->tmp_surf.memType = src_surf->memType;

  return GST_FLOW_OK;
}

/* Helper function to queue a batch for inferencing and push it to the element's
 * processing queue. */
static gpointer
gst_nvinfer_input_queue_loop (gpointer data)
{
  GstNvInfer *nvinfer = (GstNvInfer *) data;
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  std::string nvtx_str;
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinfer->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinfer->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  LockGMutex locker (nvinfer->process_lock);

  while (nvinfer->stop == FALSE) {
    GstNvInferBatch *batch;
    GstNvInferMemory *mem;
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

    mem = gst_nvinfer_buffer_get_memory (batch->conv_buf);

    /* Form the vector of input frame pointers. */
    for (i = 0; i < batch->frames.size (); i++) {
      input_frames.push_back (batch->frames[i].converted_frame_ptr);
    }

    input_batch.inputFrames = input_frames.data ();
    input_batch.numInputFrames = input_frames.size ();

    switch (mem->surf->surfaceList[0].colorFormat) {
      case NVBUF_COLOR_FORMAT_RGBA:
        input_batch.inputFormat = NvDsInferFormat_RGBA;
        break;
      case NVBUF_COLOR_FORMAT_RGB:
        input_batch.inputFormat = NvDsInferFormat_RGB;
        break;
      case NVBUF_COLOR_FORMAT_GRAY8:
      case NVBUF_COLOR_FORMAT_NV12:
        input_batch.inputFormat = NvDsInferFormat_GRAY;
        break;
      default:
        input_batch.inputFormat = NvDsInferFormat_Unknown;
        break;
    }
    input_batch.inputPitch = mem->surf->surfaceList[0].planeParams.pitch[0];

    input_batch.returnInputFunc =
        (NvDsInferContextReturnInputAsyncFunc) gst_buffer_unref;
    input_batch.returnFuncData = batch->conv_buf;

    locker.unlock ();

    nvtx_str = "queueInput batch_num=" + std::to_string(nvinfer->current_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(nvinfer->nvtx_domain, &eventAttrib);

    // Moved outside the lock, blocking preprocess thread
    if (batch->sync_obj){
      NvBufSurfTransformSyncObjWait (batch->sync_obj, -1);
      NvBufSurfTransformSyncObjDestroy (&(batch->sync_obj));
    }

    status = nvdsinfer_ctx->queueInputBatch (input_batch);

    nvtxDomainRangePop(nvinfer->nvtx_domain);

    locker.lock ();

    if (status == NVDSINFER_MEM_ERROR) {
      delete batch;
      batch = NULL;
      continue;
    }

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

static gboolean
convert_batch_and_push_to_input_thread (GstNvInfer *nvinfer,
    GstNvInferBatch *batch, GstNvInferMemory *mem)
{
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
  std::string nvtx_str;
  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinfer->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinfer->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  /* Set the transform session parameters for the conversions executed in this
   * thread. */
  err = NvBufSurfTransformSetSessionParams (&nvinfer->transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
    return FALSE;
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(nvinfer->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(nvinfer->nvtx_domain, &eventAttrib);

  if (batch->frames.size() > 0) {
    /* Batched tranformation. */
    err = NvBufSurfTransformAsync (&nvinfer->tmp_surf, mem->surf,
              &nvinfer->transform_params, &batch->sync_obj);
  }

  nvtxDomainRangePop (nvinfer->nvtx_domain);

  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer", err),
        (NULL));
    return FALSE;
  }

  LockGMutex locker (nvinfer->process_lock);
  /* Push the batch info structure in the processing queue and notify the output
   * thread that a new batch has been queued. */
  g_queue_push_tail (nvinfer->input_queue, batch);
  g_cond_broadcast (&nvinfer->process_cond);

  return TRUE;
}

/* Process entire frames in the batched buffer. */
static GstFlowReturn
gst_nvinfer_process_full_frame (GstNvInfer * nvinfer, GstBuffer * inbuf,
    NvBufSurface * in_surf)
{
  NvOSD_RectParams rect_params;
  NvDsBatchMeta *batch_meta = NULL;
  guint num_filled = 0;
  std::unique_ptr<GstNvInferBatch> batch = nullptr;
  GstBuffer *conv_gst_buf = nullptr;
  GstFlowReturn flow_ret;
  GstNvInferMemory *memory = nullptr;
  gdouble scale_ratio_x, scale_ratio_y;
  guint offset_left = 0, offset_top = 0;
  gboolean skip_batch;

  /* Process batch only when interval_counter is 0. */
  skip_batch = (nvinfer->interval_counter++ % (nvinfer->interval + 1) > 0);

  if (skip_batch) {
    return GST_FLOW_OK;
  }

  if (((in_surf->memType == NVBUF_MEM_DEFAULT || in_surf->memType == NVBUF_MEM_CUDA_DEVICE) &&
       ((int)in_surf->gpuId != (int)nvinfer->gpu_id)) ||
      (((int)in_surf->gpuId == (int)nvinfer->gpu_id) && (in_surf->memType == NVBUF_MEM_SYSTEM)))  {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Memory Compatibility Error:Input surface gpu-id doesnt match with configured gpu-id for element,"
         " please allocate input using unified memory, or use same gpu-ids OR,"
         " if same gpu-ids are used ensure appropriate Cuda memories are used"),
        ("surface-gpu-id=%d,%s-gpu-id=%d",in_surf->gpuId,GST_ELEMENT_NAME(nvinfer),
         nvinfer->gpu_id)); \
      return GST_FLOW_ERROR;
  }


  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  num_filled = batch_meta->num_frames_in_batch;

  /* Processing on full frames. Iterate through all the frames in the batched
   * input buffer. */
  for (guint i = 0; i < num_filled; i++) {
    guint idx;

    /* No existing GstNvInferBatch structure. Allocate a new structure,
     * acquire a buffer from our internal pool for conversions. */
    if (batch == nullptr) {
      batch.reset (new GstNvInferBatch);
      batch->push_buffer = FALSE;
      batch->inbuf = inbuf;
      batch->inbuf_batch_num = nvinfer->current_batch_num;

      flow_ret =
          gst_buffer_pool_acquire_buffer (nvinfer->pool, &conv_gst_buf,
          nullptr);
      if (flow_ret != GST_FLOW_OK) {
        return flow_ret;
      }
      memory = gst_nvinfer_buffer_get_memory (conv_gst_buf);
      if (!memory) {
        return GST_FLOW_ERROR;
      }
      batch->conv_buf = conv_gst_buf;
    }

    idx = batch->frames.size ();
    /* Scale the entire frame to network resolution. */
    rect_params.left = 0;
    rect_params.top = 0;
    rect_params.width = in_surf->surfaceList[i].width;
    rect_params.height = in_surf->surfaceList[i].height;

    /* Scale and convert the buffer. */
    if (get_converted_buffer (nvinfer, in_surf, in_surf->surfaceList + i,
            &rect_params, memory->surf, memory->surf->surfaceList + idx,
            scale_ratio_x, scale_ratio_y, offset_left, offset_top,
            memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
      GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED, ("Buffer conversion failed"),
          (NULL));
      return GST_FLOW_ERROR;
    }

    /* Adding a frame to the current batch. Set the frames members. */
    GstNvInferFrame frame;
    frame.converted_frame_ptr = memory->frame_memory_ptrs[idx];
    frame.scale_ratio_x = scale_ratio_x;
    frame.scale_ratio_y = scale_ratio_y;
    frame.offset_left = offset_left;
    frame.offset_top = offset_top;
    frame.obj_meta = nullptr;
    frame.frame_meta = nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
    frame.frame_num = frame.frame_meta->frame_num;
    frame.batch_index = i;
    //frame.history = nullptr;
    frame.input_surf_params = in_surf->surfaceList + i;
    batch->frames.push_back (frame);

    /* Submit batch if the batch size has reached max_batch_size or
     * if this is the last frame in the input batched buffer. */
    if (batch->frames.size () == nvinfer->max_batch_size || i == num_filled - 1) {
      if (!convert_batch_and_push_to_input_thread (nvinfer, batch.release(), memory)) {
        return GST_FLOW_ERROR;
      }

      /* Batch submitted. Set batch to nullptr so that a new GstNvInferBatch
       * structure can be allocated if required. */
      conv_gst_buf = nullptr;
      nvinfer->tmp_surf.numFilled = 0;
    }
  }
  return GST_FLOW_OK;
}

/* The object history map should be trimmed periodically to keep the map size
 * in check. */
static void
cleanup_history_map (GstNvInfer * nvinfer, GstBuffer * inbuf)
{
  LockGMutex locker (nvinfer->process_lock);
  /* Find the history map for each source whose frames are present in the batch
   * and trim the map. */
  for (auto &source_iter : *(nvinfer->source_info)) {
    GstNvInferSourceInfo &source_info = source_iter.second;
    if (source_info.last_seen_frame_num - source_info.last_cleanup_frame_num <
        MAP_CLEANUP_INTERVAL)
      continue;
    source_info.last_cleanup_frame_num = source_info.last_seen_frame_num;

    /* Remove entries for objects which have not been seen for
     * CLEANUP_ACCESS_CRITERIA */
    auto iterator = source_info.object_history_map.begin ();
    while (iterator != source_info.object_history_map.end ()) {
      auto history = iterator->second;
      if (!history->under_inference &&
          source_info.last_seen_frame_num - history->last_accessed_frame_num >
          CLEANUP_ACCESS_CRITERIA) {
        iterator = source_info.object_history_map.erase (iterator);
      } else {
        ++iterator;
      }
    }
  }
}


/* Function to decide if object should be inferred on. */
static inline gboolean
should_infer_object (GstNvInfer * nvinfer, GstBuffer * inbuf,
    NvDsObjectMeta * obj_meta, gulong frame_num,
    GstNvInferObjectHistory * history)
{
  if (nvinfer->operate_on_gie_id > -1 &&
      obj_meta->unique_component_id != nvinfer->operate_on_gie_id)
    return FALSE;

  if (obj_meta->rect_params.width < nvinfer->min_input_object_width)
    return FALSE;

  if (obj_meta->rect_params.height < nvinfer->min_input_object_height)
    return FALSE;

  if (nvinfer->max_input_object_width > 0 &&
      obj_meta->rect_params.width > nvinfer->max_input_object_width)
    return FALSE;

  if (nvinfer->max_input_object_height > 0 &&
      obj_meta->rect_params.height > nvinfer->max_input_object_height)
    return FALSE;

  /* Infer on object if the operate_on_class_ids list is empty or if
   * the flag at index  class_id is TRUE. */
  if (!nvinfer->operate_on_class_ids->empty () &&
      ((int) nvinfer->operate_on_class_ids->size () <= obj_meta->class_id ||
          nvinfer->operate_on_class_ids->at (obj_meta->class_id) == FALSE)) {
    return FALSE;
  }

  if (history && IS_CLASSIFIER_INSTANCE (nvinfer)) {
    gboolean should_reinfer = FALSE;

    /* Do not reinfer if the object area has not grown by the reinference area
     * threshold and reinfer interval criteria is not met. */
    if ((history->last_inferred_coords.width *
          history->last_inferred_coords.height * (1 +
            REINFER_AREA_THRESHOLD)) <
        (obj_meta->rect_params.width * obj_meta->rect_params.height))
      should_reinfer = TRUE;

    if (frame_num - history->last_inferred_frame_num >
         nvinfer->secondary_reinfer_interval)
      should_reinfer = TRUE;

    return should_reinfer;
  }

  if (history && IS_DETECTOR_INSTANCE (nvinfer)) {
    gboolean should_reinfer = FALSE;

    if (frame_num - history->last_inferred_frame_num >
         nvinfer->secondary_reinfer_interval ||
         nvinfer->secondary_reinfer_interval == DEFAULT_REINFER_INTERVAL)
      should_reinfer = TRUE;

    return should_reinfer;
  }

  return TRUE;
}

/* Process on objects detected by upstream detectors.
 *
 * Secondary classifiers can work in asynchronous mode as well. In this mode,
 * tracked objects are cropped and queued for inferencing. The input buffer
 * is pushed downstream (from the input thread itself) without waiting for results.
 * When the infer results are available they are stored in the object history map
 * in the output loop. After the results are available the new/updated results
 * are attached (in the input thread) to the object whenever it is found in the
 * frame again. */
static GstFlowReturn
gst_nvinfer_process_objects (GstNvInfer * nvinfer, GstBuffer * inbuf,
    NvBufSurface * in_surf)
{
  std::unique_ptr<GstNvInferBatch> batch (nullptr);
  GstBuffer *conv_gst_buf = nullptr;
  GstNvInferMemory *memory = nullptr;
  GstFlowReturn flow_ret;
  gdouble scale_ratio_x, scale_ratio_y;
  guint offset_left = 0, offset_top = 0;
  gboolean warn_untracked_object = FALSE;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    GstNvInferSourceInfo *source_info = nullptr;

    /* Find the source info instance. */
    auto iter = nvinfer->source_info->find (frame_meta->pad_index);
    if (iter == nvinfer->source_info->end ()) {
      GST_WARNING_OBJECT
          (nvinfer,
          "Source info not found for source %d. Maybe the GST_NVEVENT_PAD_ADDED"
          " event was never generated for the source.", frame_meta->pad_index);
      continue;
    } else {
      source_info = &iter->second;
    }
    source_info->last_seen_frame_num = frame_meta->frame_num;

    /* Iterate through all the objects. */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *object_meta = (NvDsObjectMeta *) (l_obj->data);
      guint idx;
      std::shared_ptr<GstNvInferObjectHistory> obj_history;
      gulong frame_num = frame_meta->frame_num;

      /* Cannot infer on untracked objects in asynchronous mode. */
      if (nvinfer->classifier_async_mode && object_meta->object_id == UNTRACKED_OBJECT_ID) {
        if (!warn_untracked_object) {
          /* Warn periodically about untracked objects in the metadata. */
          if (nvinfer->untracked_object_warn_pts == GST_CLOCK_TIME_NONE ||
              (GST_BUFFER_PTS(inbuf) - nvinfer->untracked_object_warn_pts >
                   UNTRACKED_OBJECT_WARN_INTERVAL)) {
            GST_WARNING_OBJECT (nvinfer, "Untracked objects in metadata. Cannot"
                " infer on untracked objects in asynchronous mode.");
            nvinfer->untracked_object_warn_pts = GST_BUFFER_PTS(inbuf);
          }
        }
        warn_untracked_object = TRUE;
        continue;
      }

      LockGMutex locker (nvinfer->process_lock);

      /* Find the object history if it exists only when tracking id is valid. */
      if (source_info != nullptr && object_meta->object_id != UNTRACKED_OBJECT_ID) {
        auto search =
            source_info->object_history_map.find (object_meta->object_id);
        if (search != source_info->object_history_map.end ()) {
          obj_history = search->second;
        }
      }

      bool needs_infer = should_infer_object (nvinfer, inbuf, object_meta, frame_num,
              obj_history.get());
      if (!needs_infer) {
        /* Should not infer again. */

        if (IS_CLASSIFIER_INSTANCE (nvinfer) && obj_history != nullptr) {
          /* Working in synchronous mode. Defer attachment of classifier metadata
           * in the object history to the output thread. */
          if (!nvinfer->classifier_async_mode) {
            /* No existing GstNvInferBatch structure. Allocate a new structure,
             * acquire a buffer from our internal pool for conversions. */
            if (batch == nullptr) {
              batch.reset (new GstNvInferBatch);
              batch->push_buffer = FALSE;
              batch->event_marker = FALSE;
              batch->inbuf = inbuf;
              batch->inbuf_batch_num = nvinfer->current_batch_num;
              locker.unlock ();
              flow_ret =
                gst_buffer_pool_acquire_buffer (nvinfer->pool, &conv_gst_buf,
                    nullptr);
              locker.lock ();
              if (flow_ret != GST_FLOW_OK) {
                return flow_ret;
              }
              memory = gst_nvinfer_buffer_get_memory (conv_gst_buf);
              if (!memory) {
                return GST_FLOW_ERROR;
              }
              batch->conv_buf = conv_gst_buf;
            }
            obj_history->last_accessed_frame_num = frame_meta->frame_num;
            /* Let the output thread know to attach latest available classifier
             * metadata for this object. */
            batch->objs_pending_meta_attach.emplace_back(obj_history, object_meta);
            continue;
          }
        }
      }


      /* Asynchronous mode. If we have previous results for the tracked object,
       * attach the results. New results will be attached when inference on the
       * object is complete and the object is present in the frame after that. */
      if (obj_history && nvinfer->classifier_async_mode) {
        GstNvInferFrame frame;
        frame.obj_meta = object_meta;
        attach_metadata_classifier (nvinfer, nullptr, frame,
            obj_history->cached_info);
        obj_history->last_accessed_frame_num = frame_meta->frame_num;
      }

      if (!needs_infer) {
        continue;
      }

      /* Object has a valid tracking id but does not have any history. Create
       * an entry in the map for the object. */
      if (source_info != nullptr && object_meta->object_id != UNTRACKED_OBJECT_ID &&
          obj_history == nullptr) {
        auto ret_iter =
            source_info->object_history_map.emplace (object_meta->object_id,
            std::make_shared<GstNvInferObjectHistory> ());
        obj_history = ret_iter.first->second;
      }

      /* Update the object history if it is found. */
      if (obj_history != nullptr) {
        obj_history->under_inference = TRUE;
        obj_history->last_inferred_frame_num = frame_num;
        obj_history->last_accessed_frame_num = frame_num;
        obj_history->last_inferred_coords = object_meta->rect_params;
      }

      locker.unlock ();

      /* No existing GstNvInferBatch structure. Allocate a new structure,
       * acquire a buffer from our internal pool for conversions. */
      if (batch == nullptr) {
        batch.reset (new GstNvInferBatch);
        batch->push_buffer = FALSE;
        batch->inbuf = (nvinfer->classifier_async_mode) ? nullptr : inbuf;
        batch->inbuf_batch_num = nvinfer->current_batch_num;

        flow_ret =
            gst_buffer_pool_acquire_buffer (nvinfer->pool, &conv_gst_buf,
            nullptr);
        if (flow_ret != GST_FLOW_OK) {
          return flow_ret;
        }
        memory = gst_nvinfer_buffer_get_memory (conv_gst_buf);
        if (!memory) {
          return GST_FLOW_ERROR;
        }
        batch->conv_buf = conv_gst_buf;
      }
      idx = batch->frames.size ();

      /* Crop, scale and convert the buffer. */
      if (get_converted_buffer (nvinfer, in_surf,
              in_surf->surfaceList + frame_meta->batch_id,
              &object_meta->rect_params, memory->surf,
              memory->surf->surfaceList + idx, scale_ratio_x, scale_ratio_y,
              offset_left, offset_top,
              memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
        GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
            ("Buffer conversion failed"), (NULL));
        return GST_FLOW_ERROR;
      }

      /* Adding a frame to the current batch. Set the frames members. */
      GstNvInferFrame frame;
      frame.converted_frame_ptr = memory->frame_memory_ptrs[idx];
      frame.scale_ratio_x = scale_ratio_x;
      frame.scale_ratio_y = scale_ratio_y;
      frame.offset_left = offset_left;
      frame.offset_top = offset_top;
      frame.obj_meta = (nvinfer->classifier_async_mode) ? nullptr : object_meta;
      frame.frame_meta = frame_meta;
      frame.frame_num = frame_num;
      frame.batch_index = frame_meta->batch_id;
      frame.history = obj_history;
      frame.input_surf_params =
          (nvinfer->classifier_async_mode) ? nullptr : (in_surf->surfaceList +
          frame_meta->batch_id);
      batch->frames.push_back (frame);

      /* Submit batch if the batch size has reached max_batch_size. */
      if (batch->frames.size () == nvinfer->max_batch_size) {
      if (!convert_batch_and_push_to_input_thread (nvinfer, batch.release(), memory)) {
        return GST_FLOW_ERROR;
      }
      /* Batch submitted. Set batch to nullptr so that a new GstNvInferBatch
       * structure can be allocated if required. */
      conv_gst_buf = nullptr;
      nvinfer->tmp_surf.numFilled = 0;
      }
    }
  }

  /* Submit a non-full batch. */
  if (batch) {
    /* No frames to infer in this batch. It might contain objects that
     * have been deferred for classification metadata attachment. Return
     * intermediate memory to pool. */
    if (batch->frames.size() == 0)
      gst_buffer_unref (batch->conv_buf);

    if (!convert_batch_and_push_to_input_thread (nvinfer, batch.release(), memory)) {
      return GST_FLOW_ERROR;
    }
    conv_gst_buf = nullptr;
    nvinfer->tmp_surf.numFilled = 0;
  }

  if (nvinfer->current_batch_num -
      nvinfer->last_map_cleanup_frame_num > MAP_CLEANUP_INTERVAL) {
    cleanup_history_map (nvinfer, inbuf);
    nvinfer->last_map_cleanup_frame_num = nvinfer->current_batch_num;
  }

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_nvinfer_process_tensor_input (GstNvInfer * nvinfer, GstBuffer * inbuf,
    NvBufSurface * in_surf)
{
  typedef struct{
    guint batch_size = 0;
    std::vector<GstNvInferFrame> frames;

    std::unique_ptr<GstNvInferBatch> batch = nullptr;
    std::vector<NvDsInferLayerInfo> tensors;
  } TensorInputBatch;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  std::unordered_map<guint, TensorInputBatch> tensormeta_map;

  for (NvDsMetaList * l_user = batch_meta->batch_user_meta_list; l_user != NULL;
      l_user = l_user->next) {

    NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
    if (user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META)
      continue;
    GstNvDsPreProcessBatchMeta *preproc_meta =
        (GstNvDsPreProcessBatchMeta *) user_meta->user_meta_data;

    const auto & uids = preproc_meta->target_unique_ids;
    if (std::find (uids.begin (), uids.end (),
            nvinfer->unique_id) == uids.end ())
      continue;

    if (!preproc_meta->tensor_meta)
      continue;

    guint meta_id = preproc_meta->tensor_meta->meta_id;
    nvinfer->maintain_aspect_ratio = preproc_meta->tensor_meta->maintain_aspect_ratio;

    if (preproc_meta->tensor_meta->gpu_id != nvinfer->gpu_id) {
      GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
          ("nvinfer configured with gpu-id %d but received input tensor allocated on gpu %d",
              nvinfer->gpu_id, preproc_meta->tensor_meta->gpu_id), (NULL));
      return GST_FLOW_ERROR;
    }

    bool layerExists = false;

    for (auto &layer : *nvinfer->layers_info) {
        if (layer.isInput && preproc_meta->tensor_meta->tensor_name == layer.layerName) {
          layerExists = true;
          break;
        }
    }

    if (!layerExists) {
        GST_ELEMENT_WARNING (nvinfer, STREAM, FAILED,
            ("nvinfer could not find input layer with name = %s\n",
              preproc_meta->tensor_meta->tensor_name.c_str()), (NULL));
      continue;
    }

    if (tensormeta_map[meta_id].batch == nullptr) {
      for (auto &roi : preproc_meta->roi_vector) {
        GstNvInferFrame frame;
        frame.batch_index = roi.frame_meta->batch_id;
        frame.scale_ratio_x = roi.scale_ratio_x;
        frame.scale_ratio_y = roi.scale_ratio_y;
        frame.offset_left = roi.offset_left;
        frame.offset_top = roi.offset_top;
        frame.roi_left = roi.roi.left;
        frame.roi_top = roi.roi.top;
        frame.roi_meta = &roi;
        frame.frame_meta = roi.frame_meta;
        frame.obj_meta = roi.object_meta;
        frame.frame_num = roi.frame_meta->frame_num;
        frame.input_surf_params =
            &in_surf->surfaceList[roi.frame_meta->batch_id];
        tensormeta_map[meta_id].frames.push_back (frame);
      }
    }

    NvDsInferLayerInfo tensor;
    tensor.isInput = TRUE;
    tensor.bindingIndex = -1;
    tensor.buffer = preproc_meta->tensor_meta->raw_tensor_buffer;
    tensor.layerName = preproc_meta->tensor_meta->tensor_name.c_str();
    tensor.inferDims.numDims = preproc_meta->tensor_meta->tensor_shape.size();
    tensor.inferDims.numElements = 1;
    for (unsigned int i = 0; i < tensor.inferDims.numDims; i++) {
      tensor.inferDims.numElements *=
          preproc_meta->tensor_meta->tensor_shape[i];
      tensor.inferDims.d[i] = preproc_meta->tensor_meta->tensor_shape[i];
    }

    if (tensormeta_map[meta_id].batch_size == 0) {
      tensormeta_map[meta_id].batch_size = tensor.inferDims.d[0];
    }
    else if (tensormeta_map[meta_id].batch_size != tensor.inferDims.d[0]) {
      GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
          ("Mismatch in input tensor batch sizes %d vs %d", tensormeta_map[meta_id].batch_size,
              tensor.inferDims.d[0]), (nullptr));
      return GST_FLOW_ERROR;
    }

    switch (preproc_meta->tensor_meta->data_type) {
      case NvDsDataType_FP32:
        tensor.dataType = FLOAT;
        break;
      case NvDsDataType_FP16:
        tensor.dataType = HALF;
        break;
      case NvDsDataType_UINT8:
      case NvDsDataType_INT8:
        tensor.dataType = INT8;
        break;
      case NvDsDataType_UINT32:
      case NvDsDataType_INT32:
        tensor.dataType = INT32;
        break;
      case NvDsDataType_UINT64:
      case NvDsDataType_INT64:
        tensor.dataType = INT64;
        break;
      default:
        return GST_FLOW_ERROR;
    }

    tensormeta_map[meta_id].tensors.push_back (tensor);
  }

  for (auto & it : tensormeta_map) {
    auto & tensor_input_batch = it.second;
    for (size_t i = 0; i < tensor_input_batch.frames.size (); i++) {
      if (!tensor_input_batch.batch) {
        tensor_input_batch.batch.reset (new GstNvInferBatch);
        tensor_input_batch.batch->push_buffer = FALSE;
        tensor_input_batch.batch->inbuf = inbuf;
        tensor_input_batch.batch->inbuf_batch_num = nvinfer->current_batch_num;
      }
      tensor_input_batch.batch->frames.push_back (tensor_input_batch.frames[i]);

      if (i == tensor_input_batch.frames.size () - 1
          || tensor_input_batch.batch->frames.size () == nvinfer->max_batch_size) {
        NvDsInferContextBatchPreprocessedInput input_batch;
        input_batch.tensors = tensor_input_batch.tensors.data ();
        input_batch.numInputTensors = tensor_input_batch.tensors.size ();
        input_batch.returnFuncData = nullptr;
        input_batch.returnInputFunc = nullptr;

        for (auto &tensor : tensor_input_batch.tensors)
          tensor.dims.d[0] = tensor_input_batch.batch->frames.size();

        NvDsInferStatus status =
            DS_NVINFER_IMPL (nvinfer)->m_InferCtx->
            queueInputBatchPreprocessed (input_batch);

        if (status == NVDSINFER_MEM_ERROR) {
          GstNvInferBatch *raw_batch_ptr = tensor_input_batch.batch.release();
          delete raw_batch_ptr;
          raw_batch_ptr = NULL;
          continue;
        }

        if (status != NVDSINFER_SUCCESS) {
          GST_ELEMENT_ERROR (nvinfer, STREAM, FAILED,
              ("Failed to queue input batch for inferencing"), (nullptr));
          return GST_FLOW_ERROR;
        }

        g_mutex_lock (&nvinfer->process_lock);
        g_queue_push_tail (nvinfer->process_queue, tensor_input_batch.batch.release ());
        g_cond_broadcast (&nvinfer->process_cond);
        g_mutex_unlock (&nvinfer->process_lock);

        for (auto &tensor : tensor_input_batch.tensors) {
          gsize offset = 0;
          switch (tensor.dataType) {
            case FLOAT:
              offset = 4;
              break;
            case HALF:
              offset = 2;
              break;
            case INT8:
            case UINT8:
              offset = 1;
              break;
            case INT32:
              offset = 4;
              break;
            case INT64:
              offset = 8;
              break;
            default:
              return GST_FLOW_ERROR;
          }
          for (guint j = 0; j < tensor.inferDims.numDims; j++)
            offset *= tensor.inferDims.d[j];

          tensor.buffer = (char *) tensor.buffer + offset;
        }
      }
    }
  }

  return GST_FLOW_OK;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvinfer_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstNvInfer *nvinfer = GST_NVINFER (btrans);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  GstNvInferBatch *buf_push_batch;
  GstFlowReturn flow_ret;
  std::string nvtx_str;

  /* Check for model updates and replace the model if a new model is loaded. */
  if (impl->ensureReplaceNextContext () != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
              ("Ensure next context failed."),
              ("streaming stopped"));
      return GST_FLOW_ERROR;
  }

  nvinfer->current_batch_num++;

  /* Pass through first stabilize_buffer_count buffers without inference
   * (pipeline stabilization). Config: stabilize-buffer-count; default 0. */
  if (nvinfer->current_batch_num <= nvinfer->stabilize_buffer_count) {
    gst_buffer_unref (inbuf);
    return GST_FLOW_OK;
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(nvinfer->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(nvinfer->nvtx_domain, &eventAttrib);

  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinfer->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinfer->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinfer));

  if (nvinfer->input_tensor_from_meta) {
   flow_ret = gst_nvinfer_process_tensor_input (nvinfer, inbuf, in_surf);
  } else if (nvinfer->process_full_frame) {
   flow_ret = gst_nvinfer_process_full_frame (nvinfer, inbuf, in_surf);
  } else {
    flow_ret = gst_nvinfer_process_objects (nvinfer, inbuf, in_surf);
  }

  /* Unmap the input buffer contents. */
  if (in_map_info.data)
    gst_buffer_unmap (inbuf, &in_map_info);

  if (flow_ret == GST_FLOW_ERROR)
    return GST_FLOW_ERROR;

  if (nvinfer->classifier_async_mode) {
    /* Asynchronous mode. Push the buffer immediately instead of waiting for
     * the results. */
    nvtxDomainRangeEnd(nvinfer->nvtx_domain, buf_process_range);

    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinfer));

    GstFlowReturn flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvinfer),
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
    if (nvinfer->input_queue_thread)
      g_queue_push_tail (nvinfer->input_queue, buf_push_batch);
    else
      g_queue_push_tail (nvinfer->process_queue, buf_push_batch);
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
  GstNvInfer *nvinfer = GST_NVINFER (btrans);
  return nvinfer->last_flow_ret;
}

/** Writes contents of the bound input and output layers to files. */
static void
gst_nvinfer_output_generated_file_write (GstBuffer * buf,
    NvDsInferNetworkInfo * network_info, NvDsInferLayerInfo * layers_info,
    guint num_layers, guint batch_size, GstNvInfer * nvinfer)
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
  GstNvInfer *nvinfer = GST_NVINFER (data);
  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  NvDsInferContextInitParams *init_params = impl->m_InitParams.get ();
  NvDsInferStatus status = NVDSINFER_SUCCESS;
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;
  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinfer->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinfer, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinfer->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

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
      for (auto &hist : batch->objs_pending_meta_attach) {
        GstNvInferFrame frame;
        frame.obj_meta = hist.second;
        std::weak_ptr<GstNvInferObjectHistory> obj_history = hist.first;
        attach_metadata_classifier (nvinfer, nullptr, frame,
            obj_history.lock()->cached_info);
      }
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



    /* For each frame attach metadata output. */
    for (guint i = 0; i < batch->frames.size (); i++) {
      GstNvInferFrame & frame = batch->frames[i];
      NvDsInferFrameOutput &frame_output = batch_output->frames[i];
      auto obj_history = frame.history.lock ();

      /* If we have an object's history and the buffer PTS is same as last
       * inferred PTS mark the object as not being inferred. This check could be
       * useful if object is inferred multiple times before completion of an
       * existing inference. */
      if (obj_history != nullptr) {
        if (obj_history->last_inferred_frame_num == frame.frame_num)
          obj_history->under_inference = FALSE;
      }

      if (IS_DETECTOR_INSTANCE (nvinfer) || IS_INSTANCE_SEGMENTATION_INSTANCE (nvinfer)) {
        attach_metadata_detector (nvinfer, GST_MINI_OBJECT (tensor_out_object.get()),
                frame, frame_output.detectionOutput, init_params->segmentationThreshold);
      } else if (IS_CLASSIFIER_INSTANCE (nvinfer)) {
        NvDsInferClassificationOutput &classification_output = frame_output.classificationOutput;
        GstNvInferObjectInfo new_info;
        new_info.attributes.assign(classification_output.attributes,
            classification_output.attributes + classification_output.numAttributes);
        new_info.label.assign(classification_output.label);
        for (guint i = 0; i < classification_output.numAttributes; i++) {
          classification_output.attributes[i].attributeLabel = nullptr;
        }

        /* Object history is available merge the old and new classification
         * results. */
        if (obj_history != nullptr) {
          merge_classification_output (*obj_history, new_info);
        }

        /* Use the merged classification results if available otherwise use
         * the new results. */
        auto &  info = (obj_history) ? obj_history->cached_info : new_info;

        /* Attach metadata only if not operating in async mode. In async mode,
         * the GstBuffer and the associated metadata are not valid here, since
         * the buffer is already pushed downstream. The metadata will be updated
         * in the input thread. */
        if (nvinfer->classifier_async_mode == FALSE) {
          attach_metadata_classifier (nvinfer, GST_MINI_OBJECT (tensor_out_object.get()),
                  frame, info);
        }
      } else if (IS_SEGMENTATION_INSTANCE (nvinfer)) {
        attach_metadata_segmentation (nvinfer, GST_MINI_OBJECT (tensor_out_object.get()),
            frame, frame_output.segmentationOutput);
      }
    }

    /* Attach latest available classification metadata for objects that have
     * not been inferred on in the current frame. */
    for (auto &hist : batch->objs_pending_meta_attach) {
      GstNvInferFrame frame;
      frame.obj_meta = hist.second;
      std::weak_ptr<GstNvInferObjectHistory> obj_history = hist.first;
      attach_metadata_classifier (nvinfer, nullptr, frame,
          obj_history.lock()->cached_info);
    }

    if (nvinfer->output_tensor_meta && !nvinfer->classifier_async_mode) {
      /* Attach the tensor output as meta. */
      attach_tensor_output_meta (nvinfer, GST_MINI_OBJECT(tensor_out_object.get()),
          batch.get(), batch_output);
    }
    nvtxDomainRangePop (nvinfer->nvtx_domain);

  }
  return nullptr;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvinfer_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvinfer_debug, "nvinfer", 0, "nvinfer plugin");
  gst_debug_category_set_threshold (gst_nvinfer_debug, GST_LEVEL_INFO);

  return gst_element_register (plugin, "nvinfer", GST_RANK_PRIMARY,
      GST_TYPE_NVINFER);
}

/**
 * Parse dynamic properties string and apply to init params
 * Format: "key1=value1;key2=value2;key3=value3"
 */
static void 
gst_nvinfer_parse_dynamic_properties(GstNvInfer *nvinfer, const gchar *properties_string)
{
  if (!nvinfer || !properties_string || strlen(properties_string) == 0) {
    return;
  }

  DsNvInferImpl *impl = DS_NVINFER_IMPL (nvinfer);
  if (!impl || !impl->m_InitParams) {
    return;
  }

  /* Clear existing dynamic properties */
  NvDsInferContext_ClearDynamicProperties(impl->m_InitParams.get());

  /* Parse the string: key1=value1;key2=value2;... */
  gchar **pairs = g_strsplit(properties_string, ";", -1);
  if (!pairs) {
    return;
  }

  for (gchar **pair = pairs; *pair != NULL; pair++) {
    gchar *trimmed = g_strstrip(*pair);
    if (strlen(trimmed) == 0) {
      continue;
    }

    gchar **kv = g_strsplit(trimmed, "=", 2);
    if (kv && kv[0] && kv[1]) {
      gchar *key = g_strstrip(kv[0]);
      gchar *value = g_strstrip(kv[1]);
      
      if (strlen(key) > 0 && strlen(value) > 0) {
        /* Add the dynamic property */
        if (!NvDsInferContext_SetDynamicProperty(impl->m_InitParams.get(), key, value)) {
          g_printerr("Warning: Failed to set dynamic property %s = %s\n", key, value);
        } else {
          g_print("Info: Set dynamic property %s = %s\n", key, value);
        }
      }
    }
    g_strfreev(kv);
  }

  g_strfreev(pairs);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsgst_infer,
    DESCRIPTION, nvinfer_plugin_init, "9.0", LICENSE, BINARY_PACKAGE, URL)
