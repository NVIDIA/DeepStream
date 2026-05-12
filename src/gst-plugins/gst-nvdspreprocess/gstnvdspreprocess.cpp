/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <functional>

#include "gstnvdspreprocess.h"
#include "nvdspreprocess_property_parser.h"
#include "nvdspreprocess_yaml_parser.h"
#include "gstnvdspreprocess_allocator.h"

#include <sys/time.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "gst-nvdscustomevent.h"
#include <stdint.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvdspreprocess_debug);
#define GST_CAT_DEFAULT gst_nvdspreprocess_debug
#define USE_EGLIMAGE 1

/** compile makefile WITH_OPENCV:=1
 * and enable this to write transformed ROIs to files
 */
//#define DUMP_ROIS

#ifdef DUMP_ROIS
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif

/**
 * enable to debug tensor prepared by this plugin
 * and dump it in .bin files
 */
//#define DEBUG_TENSOR

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_ENABLE,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_GPU_DEVICE_ID,
  PROP_PROCESS_ON_FRAME,
  PROP_OPERATE_ON_GIE_ID,
  PROP_TARGET_UNIQUE_IDS,
  PROP_CONFIG_FILE
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
#define DEFAULT_PROCESS_ON_FRAME 1
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_OPERATE_ON_GIE_ID -1
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_SCALING_POOL_COMPUTE_HW NvBufSurfTransformCompute_Default
#define DEFAULT_SCALING_BUF_POOL_SIZE 6 /** Inter Buffer Pool Size for Scale & Converted ROIs */
#define DEFAULT_TENSOR_BUF_POOL_SIZE 6 /** Tensor Buffer Pool Size */
#define DEFAULT_TARGET_UNIQUE_IDS ""

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define NVTX_TEAL_COLOR  0xFF008080

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
static GstStaticPadTemplate gst_nvdspreprocess_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_nvdspreprocess_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvdspreprocess_parent_class parent_class
G_DEFINE_TYPE (GstNvDsPreProcess, gst_nvdspreprocess, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdspreprocess_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdspreprocess_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvdspreprocess_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvdspreprocess_start (GstBaseTransform * btrans);
static gboolean gst_nvdspreprocess_stop (GstBaseTransform * btrans);

static GstFlowReturn
gst_nvdspreprocess_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_nvdspreprocess_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);
static gpointer gst_nvdspreprocess_output_loop (gpointer data);

static gboolean gst_nvdspreprocess_sink_event(
    GstBaseTransform* trans, GstEvent* event);

static void gst_nvdspreprocess_finalize(GObject *object);
static gboolean
gst_nvdspreprocess_src_query (GstPad * pad, GstObject * parent, GstQuery * query);

template<class T>
  T* dlsym_ptr(void* handle, char const* name) {
    return reinterpret_cast<T*>(dlsym(handle, name));
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvdspreprocess_class_init (GstNvDsPreProcessClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_finalize);
  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_stop);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_nvdspreprocess_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_nvdspreprocess_generate_output);
  gstbasetransform_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvdspreprocess_sink_event);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Enable gst-nvdspreprocess plugin, or set in passthrough mode",
          TRUE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, DEFAULT_GPU_ID,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_ON_FRAME,
      g_param_spec_boolean ("process-on-frame", "Process On Frame",
          "Process On Frame or Objects",
          DEFAULT_PROCESS_ON_FRAME,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_GIE_ID,
    g_param_spec_int ("operate-on-gie-id", "Preprocess on Gie ID",
        "Preprocess on metadata generated by GIE with this unique ID.\n"
        "\t\t\tSet to -1 to infer on all metadata.",
        -1, G_MAXINT, DEFAULT_OPERATE_ON_GIE_ID,
        (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
            GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_TARGET_UNIQUE_IDS,
        g_param_spec_string ("target-unique-ids", "Target Unique Ids",
            "list of component gie-id for which tensor is prepared\n"
            "\t\t\tUse string with values of gie-id of infer components (int) to set the property.\n"
            "\t\t\t e.g. 3;4;5",
            DEFAULT_TARGET_UNIQUE_IDS,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
    g_param_spec_string ("config-file", "Preprocess Config File",
        "Preprocess Config File",
        DEFAULT_CONFIG_FILE_PATH,
        (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdspreprocess_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdspreprocess_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "gst-nvdspreprocess plugin",
      "gst-nvdspreprocess plugin",
      "Preprocessing using custom algorithms for different streams",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvdspreprocess_init (GstNvDsPreProcess * nvdspreprocess)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvdspreprocess);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  nvdspreprocess->unique_id = DEFAULT_UNIQUE_ID;
  nvdspreprocess->process_on_frame= DEFAULT_PROCESS_ON_FRAME;
  nvdspreprocess->enable = TRUE;
  nvdspreprocess->processing_width = DEFAULT_PROCESSING_WIDTH;
  nvdspreprocess->processing_height = DEFAULT_PROCESSING_HEIGHT;
  nvdspreprocess->gpu_id = DEFAULT_GPU_ID;
  nvdspreprocess->max_batch_size = DEFAULT_BATCH_SIZE;
  nvdspreprocess->operate_on_gie_id = DEFAULT_OPERATE_ON_GIE_ID;
  nvdspreprocess->scaling_pool_compute_hw = DEFAULT_SCALING_POOL_COMPUTE_HW;
  nvdspreprocess->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);
  nvdspreprocess->config_file_parse_successful = FALSE;
  nvdspreprocess->scaling_buf_pool_size = DEFAULT_SCALING_BUF_POOL_SIZE;
  nvdspreprocess->tensor_buf_pool_size = DEFAULT_TENSOR_BUF_POOL_SIZE;
  nvdspreprocess->src_to_group_map= new std::unordered_map<gint, gint>;

   /* Set the default pre-processing transform params. */
  nvdspreprocess->transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  nvdspreprocess->transform_params.transform_filter = NvBufSurfTransformInter_Default;

  gst_pad_set_query_function (GST_BASE_TRANSFORM_SRC_PAD(nvdspreprocess), GST_DEBUG_FUNCPTR (gst_nvdspreprocess_src_query));
}

static void
gst_nvdspreprocess_finalize(GObject* object)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (object);

  /* Deinitialize custom library */
  if (nvdspreprocess->custom_lib_path) {
    std::function<void(void *)> deInitLib;

    if (nvdspreprocess->custom_lib_handle) {
      deInitLib = dlsym_ptr<void(void *)>(nvdspreprocess->custom_lib_handle, "deInitLib");
      deInitLib(nvdspreprocess->custom_lib_ctx);
      dlclose(nvdspreprocess->custom_lib_handle);
      nvdspreprocess->custom_lib_handle = NULL;
    }

    delete[] (nvdspreprocess->custom_lib_path);
    nvdspreprocess->custom_lib_path = NULL;
  }
  nvdspreprocess->acquire_impl.reset();

  GST_DEBUG_OBJECT (nvdspreprocess, "Successfully Closed Custom Library\n");
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvdspreprocess_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      nvdspreprocess->unique_id = g_value_get_uint (value);
      break;
    case PROP_ENABLE:
      nvdspreprocess->enable = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvdspreprocess->gpu_id = g_value_get_uint (value);
      break;
    case PROP_PROCESS_ON_FRAME:
      nvdspreprocess->process_on_frame = g_value_get_boolean (value);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      nvdspreprocess->operate_on_gie_id = g_value_get_int (value);
      break;
    case PROP_TARGET_UNIQUE_IDS:
    {
      std::stringstream str(g_value_get_string(value)? g_value_get_string(value) : "");
      nvdspreprocess->target_unique_ids.clear();
      while(str.peek() != EOF) {
          gint gie_id;
          str >> gie_id;
          nvdspreprocess->target_unique_ids.push_back(gie_id);
          str.get();
      }
    }
      break;
    case PROP_CONFIG_FILE:
    {
        g_mutex_lock (&nvdspreprocess->preprocess_lock);
        g_free (nvdspreprocess->config_file_path);
        nvdspreprocess->config_file_path = g_value_dup_string (value);
        std::string lower(nvdspreprocess->config_file_path);
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        bool is_yaml =  (lower.length() >= 4 && lower.substr(lower.length() - 4) == ".yml") ||
           (lower.length() >= 5 && lower.substr(lower.length() - 5) == ".yaml");
        /* Parse the initialization parameters from the config file. This function
         * gives preference to values set through the set_property function over
         * the values set in the config file. */
        if (is_yaml) {
          nvdspreprocess->config_file_parse_successful =
          gst_nvdspreprocess_parse_config_file_yaml(nvdspreprocess, nvdspreprocess->config_file_path);
        } else {
          nvdspreprocess->config_file_parse_successful =
            nvdspreprocess_parse_config_file (nvdspreprocess,
                                                  nvdspreprocess->config_file_path);
        }

        if (nvdspreprocess->config_file_parse_successful) {
          GST_DEBUG_OBJECT (nvdspreprocess, "Successfully Parsed Config file\n");
        }
        g_mutex_unlock (&nvdspreprocess->preprocess_lock);
      }
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
gst_nvdspreprocess_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, nvdspreprocess->unique_id);
      break;
    case PROP_ENABLE:
      g_value_set_boolean (value, nvdspreprocess->enable);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvdspreprocess->gpu_id);
      break;
    case PROP_PROCESS_ON_FRAME:
      g_value_set_boolean (value, nvdspreprocess->process_on_frame);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      g_value_set_int (value, nvdspreprocess->operate_on_gie_id);
      break;
    case PROP_TARGET_UNIQUE_IDS:
    {
      std::stringstream str;
      for(const auto id : nvdspreprocess->target_unique_ids)
          str << id << ";";
      g_value_set_string (value, str.str ().c_str ());
    }
      break;
    case PROP_CONFIG_FILE:
      g_value_set_string (value, nvdspreprocess->config_file_path);
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
gst_nvdspreprocess_start (GstBaseTransform * btrans)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (btrans);
  std::string nvtx_str;
  NvBufSurfaceColorFormat color_format;
  cudaError_t cudaReturn;
  /** scaling pool params */
  GstStructure *scaling_pool_config;
  GstAllocator *scaling_pool_allocator;
  GstAllocationParams allocation_params;
  /** tensor pool params */
  GstStructure *tensor_pool_config;
  GstAllocator *tensor_pool_allocator;
  GstAllocationParams tensor_pool_allocation_params;

#ifdef DUMP_ROIS
  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo (&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
          is_nvgpu = 1;
      }
  }
#endif
  if (!nvdspreprocess->config_file_path || strlen (nvdspreprocess->config_file_path) == 0) {
    GST_ELEMENT_ERROR (nvdspreprocess, LIBRARY, SETTINGS,
        ("Configuration file not provided"), (nullptr));
    return FALSE;
  }

  if (nvdspreprocess->config_file_parse_successful == FALSE) {
    GST_ELEMENT_ERROR (nvdspreprocess, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", nvdspreprocess->config_file_path));
    return FALSE;
  }

  nvtx_str = "GstNvDsPreProcess: UID=" + std::to_string(nvdspreprocess->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  nvdspreprocess->custom_initparams.tensor_params = nvdspreprocess->tensor_params;
  nvdspreprocess->custom_initparams.unique_id = nvdspreprocess->unique_id;
  nvdspreprocess->custom_initparams.config_file_path = nvdspreprocess->config_file_path;

  /* Initialize custom library */
  if (nvdspreprocess->custom_lib_path) {
    if (nvdspreprocess->custom_lib_handle) {
      GST_DEBUG_OBJECT (nvdspreprocess, "Custom library Has Been Loaded\n");
      return TRUE;
    }
    nvdspreprocess->custom_lib_handle = dlopen(nvdspreprocess->custom_lib_path, RTLD_NOW);
    std::function<CustomCtx*(CustomInitParams)> initLib;

    if (nvdspreprocess->custom_lib_handle) {
      GST_DEBUG_OBJECT (nvdspreprocess, "Custom Library Opened Successfully\n");
      initLib = dlsym_ptr<CustomCtx*(CustomInitParams)>(nvdspreprocess->custom_lib_handle, "initLib");
      nvdspreprocess->custom_lib_ctx = initLib(nvdspreprocess->custom_initparams);
      if (nvdspreprocess->custom_lib_ctx == nullptr) {
        GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                          ("Error while initializing Custom Library\n"), (NULL));
        return FALSE;
      }

      if (!nvdspreprocess->custom_tensor_function_name.empty()) {
        nvdspreprocess->custom_tensor_function =
            dlsym_ptr<NvDsPreProcessStatus(CustomCtx *, NvDsPreProcessBatch *, NvDsPreProcessCustomBuf *&,
                                      CustomTensorParams &, NvDsPreProcessAcquirer *)>
            (nvdspreprocess->custom_lib_handle, nvdspreprocess->custom_tensor_function_name.c_str());

        if (!nvdspreprocess->custom_tensor_function) {
          GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                            ("Error while loading Custom Tensor Preparation function\n"), (NULL));
          return FALSE;
        }
      }

      for (guint gcnt = 0; gcnt < nvdspreprocess->nvdspreprocess_groups.size(); gcnt ++) {
        if (!nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform_function_name.empty()) {
          nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform =
              dlsym_ptr<NvDsPreProcessStatus(NvBufSurface*, NvBufSurface*, CustomTransformParams &)>
              (nvdspreprocess->custom_lib_handle, nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform_function_name.c_str());

          if (!nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform) {
            GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                    ("Custom Transformation function not found"), (NULL));
            return FALSE;
          }
        }
      }
    }
    else {
      GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
            ("Could not open custom library\n"), (NULL));
      return FALSE;
    }
  }

  GST_DEBUG_OBJECT (nvdspreprocess, "Initialized Custom Library Context\n");

  /* Create the intermediate NvBufSurface structure for holding an array of input
   * NvBufSurfaceParams for batched transforms. */
  nvdspreprocess->batch_insurf.surfaceList =
      new NvBufSurfaceParams[nvdspreprocess->max_batch_size];
  nvdspreprocess->batch_insurf.batchSize = nvdspreprocess->max_batch_size;
  nvdspreprocess->batch_insurf.gpuId = nvdspreprocess->gpu_id;

  /* Holds output of batched transforms. */
  nvdspreprocess->batch_outsurf.surfaceList =
      new NvBufSurfaceParams[nvdspreprocess->max_batch_size];
  nvdspreprocess->batch_outsurf.batchSize = nvdspreprocess->max_batch_size;
  nvdspreprocess->batch_outsurf.gpuId = nvdspreprocess->gpu_id;

  /* Set the NvBufSurfTransform config parameters. */
  cudaReturn = cudaSetDevice (nvdspreprocess->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvdspreprocess->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
    goto error;
  }

  cudaReturn =
      cudaStreamCreateWithFlags (&nvdspreprocess->convert_stream,
      cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to create cuda stream for conversion/transformation"),
        ("cudaStreamCreateWithFlags failed with error %s",
            cudaGetErrorName (cudaReturn)));
    goto error;
  }

  nvdspreprocess->transform_config_params.gpu_id = nvdspreprocess->gpu_id;
  nvdspreprocess->transform_config_params.cuda_stream = nvdspreprocess->convert_stream;
  nvdspreprocess->transform_config_params.compute_mode = nvdspreprocess->scaling_pool_compute_hw;

  /* Set up the NvBufSurfTransformParams structure for batched transforms. */
  nvdspreprocess->transform_params.src_rect =
      new NvBufSurfTransformRect[nvdspreprocess->max_batch_size];
  nvdspreprocess->transform_params.dst_rect =
      new NvBufSurfTransformRect[nvdspreprocess->max_batch_size];
  nvdspreprocess->transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  nvdspreprocess->transform_params.transform_flip = NvBufSurfTransform_None;
  nvdspreprocess->transform_params.transform_filter = nvdspreprocess->scaling_pool_interpolation_filter;

   /* Create a buffer pool for internal memory required for scaling frames to
   * network resolution / cropping objects. The pool allocates
   * nvdspreprocess->scaling_buf_pool_size buffers at start and keeps reusing them. */
  nvdspreprocess->scaling_pool = gst_buffer_pool_new();
  scaling_pool_config = gst_buffer_pool_get_config(nvdspreprocess->scaling_pool);
  gst_buffer_pool_config_set_params(scaling_pool_config, nullptr,
      sizeof (GstNvDsPreProcessMemory), nvdspreprocess->scaling_buf_pool_size,
      nvdspreprocess->scaling_buf_pool_size);

  /* Based on the network input requirements decide the buffer pool color format. */
  switch (nvdspreprocess->tensor_params.network_color_format) {
    case NvDsPreProcessFormat_RGB:
    case NvDsPreProcessFormat_BGR:
      color_format = NVBUF_COLOR_FORMAT_RGBA;
      nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_RGBA;
    break;
    case NvDsPreProcessFormat_GRAY:
    if(nvdspreprocess->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC
#ifdef __aarch64__
      || nvdspreprocess->transform_config_params.compute_mode == NvBufSurfTransformCompute_Default
#endif
    ) {
      g_print("Warning: converting ROIs to NV12 for VIC mode\n");
      color_format = NVBUF_COLOR_FORMAT_NV12;
      nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_GRAY;
    }
    else {
      color_format = NVBUF_COLOR_FORMAT_GRAY8;
      nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_GRAY;
    }
    break;
    default:
      GST_ELEMENT_ERROR (nvdspreprocess, LIBRARY, SETTINGS,
          ("Unsupported network color format: %d",
              nvdspreprocess->tensor_params.network_color_format), (nullptr));
      goto error;
  }

  GST_DEBUG_OBJECT (nvdspreprocess, "network-input-order = %d\n",
        nvdspreprocess->custom_initparams.tensor_params.network_input_order);

  if (nvdspreprocess->tensor_params.network_input_order == NvDsPreProcessNetworkInputOrder_kNCHW) {
    if (nvdspreprocess->processing_height != nvdspreprocess->tensor_params.network_input_shape[2] ||
        nvdspreprocess->processing_width != nvdspreprocess->tensor_params.network_input_shape[3]) {
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
          ("Processing height = %d and width =%d should be same as network height = %d and width =%d\n",
              nvdspreprocess->processing_height, nvdspreprocess->processing_width,
              nvdspreprocess->tensor_params.network_input_shape[2],
              nvdspreprocess->tensor_params.network_input_shape[3]), (nullptr));
    }
  } else if (nvdspreprocess->tensor_params.network_input_order == NvDsPreProcessNetworkInputOrder_kNHWC) {
    if (nvdspreprocess->processing_height != nvdspreprocess->tensor_params.network_input_shape[1] ||
        nvdspreprocess->processing_width != nvdspreprocess->tensor_params.network_input_shape[2]) {
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
          ("Processing height = %d and width =%d should be same as network height = %d and width =%d\n",
              nvdspreprocess->processing_height, nvdspreprocess->processing_width,
              nvdspreprocess->tensor_params.network_input_shape[1],
              nvdspreprocess->tensor_params.network_input_shape[2]), (nullptr));
    }
  } else if (nvdspreprocess->tensor_params.network_input_order == NvDsPreProcessNetworkInputOrder_CUSTOM) {
    g_print("Using user provided processing height = %d and processing width = %d\n",
            nvdspreprocess->processing_height, nvdspreprocess->processing_width);
  } else {
    GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
      ("Unsupported network input order: %d",
          nvdspreprocess->tensor_params.network_input_order), (nullptr));
  }

  /* Create a new GstNvDsPreProcess Allocator instance. Allocator has methods to allocate
   * and free custom memories. */
  GstNvDsPreProcessVideoBufferAllocatorInfo allocator_info;
  allocator_info.width = nvdspreprocess->processing_width;
  allocator_info.height = nvdspreprocess->processing_height;
  allocator_info.color_format = color_format;
  allocator_info.batch_size = nvdspreprocess->max_batch_size;

#ifdef DUMP_ROIS
  allocator_info.memory_type = NVBUF_MEM_CUDA_UNIFIED;
  if (is_nvgpu) {
    allocator_info.memory_type = NVBUF_MEM_DEFAULT;
  }
#else
  allocator_info.memory_type = nvdspreprocess->scaling_pool_memory_type;
#endif

  GST_DEBUG_OBJECT (nvdspreprocess, "Scaling pool batch-size = %d\n", nvdspreprocess->max_batch_size);

  scaling_pool_allocator = gst_nvdspreprocess_allocator_new (&allocator_info, 1, nvdspreprocess->gpu_id, FALSE);

  memset (&allocation_params, 0, sizeof (allocation_params));

  gst_buffer_pool_config_set_allocator (scaling_pool_config, scaling_pool_allocator,
      &allocation_params);

  if (!gst_buffer_pool_set_config (nvdspreprocess->scaling_pool, scaling_pool_config)) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set config on buffer scaling_pool"), (nullptr));
    goto error;
  }

  /* Start the buffer pool and allocate all internal buffers. */
  if (!gst_buffer_pool_set_active (nvdspreprocess->scaling_pool, TRUE)) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set buffer pool to active"), (nullptr));
    goto error;
  }

   /* Create a buffer pool for internal memory required for scaling frames to
   * network resolution / cropping objects. The pool allocates
   * INTERNAL_BUF_POOL_SIZE buffers at start and keeps reusing them. */

  nvdspreprocess->tensor_pool = gst_buffer_pool_new();
  tensor_pool_config = gst_buffer_pool_get_config(nvdspreprocess->tensor_pool);
  gst_buffer_pool_config_set_params(tensor_pool_config, nullptr,
      sizeof (GstNvDsPreProcessMemory), nvdspreprocess->tensor_buf_pool_size,
      nvdspreprocess->tensor_buf_pool_size);

  nvdspreprocess->tensor_params.buffer_size = 1;
  for (auto& p : nvdspreprocess->tensor_params.network_input_shape) {
   nvdspreprocess->tensor_params.buffer_size *= p;
  }

  switch (nvdspreprocess->tensor_params.data_type) {
    case NvDsDataType_FP32:
    case NvDsDataType_UINT32:
    case NvDsDataType_INT32:
      nvdspreprocess->tensor_params.buffer_size *= 4;
    break;
    case NvDsDataType_UINT8:
    case NvDsDataType_INT8:
      nvdspreprocess->tensor_params.buffer_size *= 1;
    break;
    case NvDsDataType_UINT64:
    case NvDsDataType_INT64:
      nvdspreprocess->tensor_params.buffer_size *= 8;
    break;
    case NvDsDataType_FP16:
      nvdspreprocess->tensor_params.buffer_size *= 2;
    break;
    default:
      GST_ELEMENT_ERROR (nvdspreprocess, LIBRARY, SETTINGS,
          ("Tensor data type : %d is not Supported\n",
              (int )nvdspreprocess->tensor_params.data_type), (nullptr));
      goto error;
  }

  GST_DEBUG_OBJECT(nvdspreprocess, "Tensor Buffer Pool size = %ld data-type=%d\n", nvdspreprocess->tensor_params.buffer_size,
          nvdspreprocess->tensor_params.data_type);
#ifdef DEBUG_TENSOR
  tensor_pool_allocator = gst_nvdspreprocess_allocator_new (NULL, nvdspreprocess->tensor_params.buffer_size,
       nvdspreprocess->gpu_id, TRUE);
#else
  tensor_pool_allocator = gst_nvdspreprocess_allocator_new (NULL, nvdspreprocess->tensor_params.buffer_size,
       nvdspreprocess->gpu_id, FALSE);
#endif
  memset (&tensor_pool_allocation_params, 0, sizeof (tensor_pool_allocation_params));

  gst_buffer_pool_config_set_allocator (tensor_pool_config, tensor_pool_allocator,
      &tensor_pool_allocation_params);

  if (!gst_buffer_pool_set_config (nvdspreprocess->tensor_pool, tensor_pool_config)) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set config on tensor buffer pool"), (nullptr));
    goto error;
  }

  /* Start the buffer pool and allocate all internal buffers. */
  if (!gst_buffer_pool_set_active (nvdspreprocess->tensor_pool, TRUE)) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set tensor buffer pool to active"), (nullptr));
    goto error;
  }

  /** class for acquiring/releasing buffer from tensor pool */
  nvdspreprocess->acquire_impl = std::make_unique <NvDsPreProcessAcquirerImpl> (nvdspreprocess->tensor_pool);

  nvdspreprocess->nvtx_domain = nvtx_domain_ptr.release ();

  /* Create process queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  nvdspreprocess->preprocess_queue = g_queue_new ();

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  nvdspreprocess->output_thread =
      g_thread_new ("nvdspreprocess-process-thread", gst_nvdspreprocess_output_loop,
      nvdspreprocess);

  return TRUE;

error:
  delete[] nvdspreprocess->transform_params.src_rect;
  delete[] nvdspreprocess->transform_params.dst_rect;

  delete[] nvdspreprocess->batch_insurf.surfaceList;
  delete[] nvdspreprocess->batch_outsurf.surfaceList;

  if (nvdspreprocess->convert_stream) {
    cudaStreamDestroy (nvdspreprocess->convert_stream);
    nvdspreprocess->convert_stream = NULL;
  }

  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_nvdspreprocess_stop (GstBaseTransform * btrans)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (btrans);

  g_mutex_lock (&nvdspreprocess->preprocess_lock);

  /* Wait till all the items in the queue are handled. */
  while (!g_queue_is_empty (nvdspreprocess->preprocess_queue)) {
    g_cond_wait (&nvdspreprocess->preprocess_cond, &nvdspreprocess->preprocess_lock);
  }

  nvdspreprocess->stop = TRUE;

  g_cond_broadcast (&nvdspreprocess->preprocess_cond);
  g_mutex_unlock (&nvdspreprocess->preprocess_lock);

  g_thread_join (nvdspreprocess->output_thread);

  cudaSetDevice (nvdspreprocess->gpu_id);

  if (nvdspreprocess->convert_stream)
    cudaStreamDestroy (nvdspreprocess->convert_stream);
  nvdspreprocess->convert_stream = NULL;

  delete[] nvdspreprocess->transform_params.src_rect;
  delete[] nvdspreprocess->transform_params.dst_rect;

  delete[] nvdspreprocess->batch_insurf.surfaceList;
  delete[] nvdspreprocess->batch_outsurf.surfaceList;

  g_queue_free (nvdspreprocess->preprocess_queue);

  if (nvdspreprocess->config_file_path) {
    g_free (nvdspreprocess->config_file_path);
    nvdspreprocess->config_file_path = NULL;
  }

  /* delete the heap allocated memory */
  for (auto &group : nvdspreprocess->nvdspreprocess_groups) {
    group->framemeta_map.clear ();
    delete group;
    group = NULL;
  }

  /* Free up the memory allocated by pool. */
  gst_object_unref (nvdspreprocess->scaling_pool);
  gst_object_unref (nvdspreprocess->tensor_pool);

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvdspreprocess_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (btrans);
  /* Save the input video information, since this will be required later. */
  //gst_video_info_from_caps (&nvdspreprocess->video_info, incaps);

  CHECK_CUDA_STATUS (cudaSetDevice (nvdspreprocess->gpu_id),
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
scale_and_fill_data(GstNvDsPreProcess * nvdspreprocess,
    NvBufSurfaceParams * src_frame, NvOSD_RectParams * crop_rect_params,
    gdouble & ratio_x, gdouble & ratio_y, guint & offset_left, guint & offset_top,
    NvBufSurface * dest_surf, NvBufSurfaceParams * dest_frame,
    void *destCudaPtr)
{
  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  if (crop_rect_params->left > src_frame->width || crop_rect_params->top > src_frame->height) {
    GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("%s:crop_rect_params [left: %.02f or top:%.02f] are out of boundary", __func__, crop_rect_params->left, crop_rect_params->top), (NULL));
    return GST_FLOW_ERROR;
  }

  /* Clipping the excess width of rectangle */
  if(crop_rect_params->left+crop_rect_params->width > src_frame->width){
    crop_rect_params->width = src_frame->width-crop_rect_params->left;
  }

  /* Clipping the excess height of rectangle */
  if(crop_rect_params->top+crop_rect_params->height > src_frame->height){
    crop_rect_params->height = src_frame->height-crop_rect_params->top;
  }

  gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
  gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
  gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
  gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);
  guint dest_width, dest_height;

  guint offset_right = 0, offset_bottom = 0;
  offset_left = 0;
  offset_top = 0;

  if (nvdspreprocess->maintain_aspect_ratio) {
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
    if (!nvdspreprocess->symmetric_padding) {
      /* Non Symmetric Padding. */
      /* Right side Padding. */
      offset_right=(dest_frame->width - dest_width);
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr + pixel_size * dest_width,
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_right, dest_frame->height,
          nvdspreprocess->convert_stream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvdspreprocess,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
      /* Bottom side Padding. */
      offset_bottom =dest_frame->height - dest_height;
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr +
          dest_frame->planeParams.pitch[0] * dest_height,
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_bottom, nvdspreprocess->convert_stream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvdspreprocess,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
    } else {
      /* Symmetric Padding. */
      /* Left side Half Padding. */
      offset_left = (dest_frame->width - dest_width) / 2;
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr,
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_left, dest_frame->height, nvdspreprocess->convert_stream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvdspreprocess,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
      /* Right side Half Padding. */
      offset_right = dest_frame->width - dest_width - offset_left;
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr + pixel_size *
          (dest_width + offset_left),
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_right, dest_frame->height,
          nvdspreprocess->convert_stream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvdspreprocess,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
      /* Top side Half Padding. */
      offset_top = (dest_frame->height - dest_height) / 2;
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr,
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_top, nvdspreprocess->convert_stream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvdspreprocess,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
      /* Bottom side Half Padding. */
      offset_bottom = dest_frame->height - dest_height - offset_top;
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr +
          dest_frame->planeParams.pitch[0] * (dest_height + offset_top),
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_bottom, nvdspreprocess->convert_stream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvdspreprocess,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
    }
  } else {
    GST_DEBUG_OBJECT (nvdspreprocess, "scaling at processing width & height\n");
    dest_width = nvdspreprocess->processing_width;
    dest_height = nvdspreprocess->processing_height;
  }

  /* Calculate the scaling ratio of the frame / object crop. This will be
   * required later for rescaling the detector output boxes to input resolution.
   */
  ratio_x = (double) dest_width / src_width;
  ratio_y = (double) dest_height / src_height;

#ifdef __aarch64__
  if (nvdspreprocess->scaling_pool_compute_hw != NvBufSurfTransformCompute_GPU) {
    if (ratio_y <= 1.0 / 16 || ratio_y >= 16.0) {
      GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("%s:Currently cannot scale by ratio > 16 or < 1/16 for Jetson \n Use NvBufSurfTransformCompute_GPU compute mode to resolve the issue in the config file.\n", __func__), (NULL));
      return GST_FLOW_ERROR;
    }
  }
#endif

  /* We will first convert only the Region of Interest (the entire frame or the
   * object bounding box) to RGB and then scale the converted RGB frame to
   * processing resolution. */
  GST_DEBUG_OBJECT (nvdspreprocess, "Fill the input buffer for batch transformation\n");

  /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
  nvdspreprocess->batch_insurf.surfaceList[nvdspreprocess->batch_insurf.numFilled] = *src_frame;

  nvdspreprocess->batch_outsurf.surfaceList[nvdspreprocess->batch_outsurf.numFilled] = *dest_frame;

  /* Set the source ROI. Could be entire frame or an object. */
  nvdspreprocess->transform_params.src_rect[nvdspreprocess->batch_insurf.numFilled] = {
  (guint) src_top, (guint) src_left, (guint) src_width, (guint) src_height};
  /* Set the dest ROI. Could be the entire destination frame or part of it to
   * maintain aspect ratio. */
  nvdspreprocess->transform_params.dst_rect[nvdspreprocess->batch_outsurf.numFilled] = {
  offset_top, offset_left, dest_width, dest_height};

  nvdspreprocess->batch_insurf.numFilled++;
  nvdspreprocess->batch_outsurf.numFilled++;

  nvdspreprocess->batch_insurf.batchSize = nvdspreprocess->batch_insurf.numFilled;
  nvdspreprocess->batch_outsurf.batchSize = nvdspreprocess->batch_outsurf.numFilled;

  return GST_FLOW_OK;
}

#ifdef DUMP_ROIS
static gboolean dump_rois (GstNvDsPreProcess * nvdspreprocess, NvDsPreProcessBatch *batch,
    NvBufSurface * outsurf)
{
  cv::Mat in_mat, out_mat;
  static guint cnt = 0;
  guint src_id = G_MAXUINT;
  guint roi_cnt = 0;

  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo (&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
          is_nvgpu = 1;
      }
  }

  for (guint i = 0; i < batch->units.size(); i++) {

    if (src_id == batch->units[i].frame_meta->source_id) {
      roi_cnt ++;
    } else {
      roi_cnt = 0;
    }
    src_id = batch->units[i].frame_meta->source_id;

    // Map the buffer so that it can be accessed by CPU
    if (NvBufSurfaceMap (outsurf, i, 0, NVBUF_MAP_READ) != 0) {
      GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
          ("%s:buffer map to be accessed by CPU failed", __func__), (NULL));
      return FALSE;
    }

    if (is_nvgpu) {
      // sync mapped data for CPU access
      // TODO: outsurf->numFilled needs to be set correctly when function is called.
      NvBufSurfaceSyncForCpu (outsurf, i, 0);
    }

    in_mat =
        cv::Mat (nvdspreprocess->processing_height, nvdspreprocess->processing_width,
        CV_8UC4, outsurf->surfaceList[i].mappedAddr.addr[0],
        outsurf->surfaceList[i].pitch);

#ifdef __aarch64__
#if (CV_MAJOR_VERSION >= 4)
    cv::cvtColor (in_mat, out_mat, cv::COLOR_RGBA2BGR);
#else
    cv::cvtColor (in_mat, out_mat, CV_RGBA2BGR);
#endif
#else
    cv::cvtColor (in_mat, out_mat, cv::COLOR_RGBA2BGR);
#endif
    cv::imwrite("out_" + std::to_string (cnt) + "__src__" + std::to_string (src_id) +
        "__roi__" + std::to_string (roi_cnt) + ".jpeg", out_mat);

    if (NvBufSurfaceUnMap (outsurf, i,0)) {
      GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("%s:buffer unmap to be accessed by CPU failed", __func__), (NULL));
      return FALSE;
    }

    if (is_nvgpu) {
      // To use the converted buffer in CUDA, create an EGLImage and then use
      // CUDA-EGL interop APIs
      if (USE_EGLIMAGE) {
        if (NvBufSurfaceMapEglImage (outsurf, 0) != 0) {
          GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
            ("%s:buffer map eglimage failed", __func__), (NULL));
          return FALSE;
        }
        // outsurf->surfaceList[0].mappedAddr.eglImage
        // Use interop APIs cuGraphicsEGLRegisterImage and
        // cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA
        // Destroy the EGLImage
        NvBufSurfaceUnMapEglImage (outsurf, 0);
      }
    }
  }
  cnt++;

  return TRUE;
}
#endif

/** As an custom example we perform async batched transformation here. */
static gboolean batch_transformation (NvBufSurface *in_surf,
      NvBufSurface *out_surf, CustomTransformParams &params)
{
  NvBufSurfTransform_Error err;

  err = NvBufSurfTransformSetSessionParams (&params.transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ERROR ("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
    return FALSE;
  }

  /* Batched tranformation. */
  err = NvBufSurfTransformAsync (in_surf, out_surf, &params.transform_params, &params.sync_obj);

  if (err != NvBufSurfTransformError_Success) {
    GST_ERROR ("NvBufSurfTransform failed with error %d\n", err);
    return FALSE;
  }

  return TRUE;
}

static gboolean gst_nvdspreprocess_sink_event(
    GstBaseTransform* trans, GstEvent* event) {
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (trans);

  if ((GstNvDsCustomEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_ROI_UPDATE) {
    gchar* stream_id = NULL;
    guint roi_count = 0;
    RoiDimension *roi_dim;
    int source_id;

    gst_nvevent_parse_roi_update(event, &stream_id, &roi_count, &roi_dim);
    source_id=atoi(stream_id);

    // if stream exists
    if (nvdspreprocess->src_to_group_map->find(source_id)!= nvdspreprocess->src_to_group_map->end()) {
      int group_id = nvdspreprocess->src_to_group_map->at(source_id);

      GstNvDsPreProcessGroup *preprocess_group = nvdspreprocess->nvdspreprocess_groups[group_id];
      GstNvDsPreProcessFrame preprocess_frame;

      for (int i =0;i<(int)roi_count;i++){
        NvDsRoiMeta roi_info;
        roi_info.roi.left = roi_dim[i].left;
        roi_info.roi.top = roi_dim[i].top;
        roi_info.roi.width = roi_dim[i].width;
        roi_info.roi.height = roi_dim[i].height;

        //update the process frame with new roi
        preprocess_frame.roi_vector.push_back(roi_info);
      }
      //update the framemeta_map
      g_mutex_lock (&nvdspreprocess->framemeta_map_lock);
      preprocess_group->framemeta_map[source_id]=preprocess_frame;
      g_mutex_unlock (&nvdspreprocess->framemeta_map_lock);
    }
    g_free(stream_id); // free the stream_id post usage.
    g_free(roi_dim); // free the roi_dim post usage.
  }

  /* Serialize events. Wait for pending buffers to be processed and pushed downstream*/
  if (GST_EVENT_IS_SERIALIZED (event)) {
    NvDsPreProcessBatch * batch = new NvDsPreProcessBatch;
    batch->event_marker = TRUE;

    /* Push the event marker batch in the preprocessing queue. */
    g_mutex_lock (&nvdspreprocess->preprocess_lock);
    g_queue_push_tail (nvdspreprocess->preprocess_queue, batch);
    g_cond_broadcast (&nvdspreprocess->preprocess_cond);

    /* Wait for all the remaining batches in the preprocessing queue including
      * the event marker to be processed. */
    while (!g_queue_is_empty (nvdspreprocess->preprocess_queue)) {
      g_cond_wait (&nvdspreprocess->preprocess_cond, &nvdspreprocess->preprocess_lock);
    }
    g_mutex_unlock (&nvdspreprocess->preprocess_lock);
  }

  /* Call the sink event handler of the base class. */
  return GST_BASE_TRANSFORM_CLASS(parent_class)->sink_event(trans, event);
}

static gboolean
gst_nvdspreprocess_src_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (parent);
  guint gieid = 0;
  if (gst_nvquery_is_preprocess_poolsize (query) &&
      gst_nvquery_preprocess_poolsize_gieid_parse(query, &gieid)) {

    if (std::find (nvdspreprocess->target_unique_ids.begin (),
                   nvdspreprocess->target_unique_ids.end (),
                   gieid) != nvdspreprocess->target_unique_ids.end ())
    {
      gst_nvquery_preprocess_poolsize_set (query, nvdspreprocess->scaling_buf_pool_size);
      return TRUE;
    }
  }
  return gst_pad_query_default (pad, parent, query);
}


static gboolean
group_transformation (GstNvDsPreProcess *nvdspreprocess, GstNvDsPreProcessGroup *&group)
{
  std::string nvtx_str;
  CustomTransformParams params;
  gboolean ret = 0;

  /** Configure transform session parameters for the transformation */
  params.transform_config_params = nvdspreprocess->transform_config_params;
  params.transform_params = nvdspreprocess->transform_params;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_TEAL_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(nvdspreprocess->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(nvdspreprocess->nvtx_domain, &eventAttrib);

  if (nvdspreprocess->custom_lib_path && nvdspreprocess->custom_lib_handle && !group->custom_transform_function_name.empty()) {
    NvDsPreProcessStatus status;
    status = group->custom_transform(&nvdspreprocess->batch_insurf,
                            &nvdspreprocess->batch_outsurf, params);

    if (status != NVDSPREPROCESS_SUCCESS) {
      GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                        ("Custom Transformation from library failed\n"), (NULL));
      return FALSE;
    }
    ret = TRUE;
    GST_DEBUG_OBJECT (nvdspreprocess, "Custom tranformation from library successfull\n");
  }
  else {
    ret = batch_transformation (&nvdspreprocess->batch_insurf, &nvdspreprocess->batch_outsurf, params);
    if (!ret) {
      GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
              ("Transformation from plugin failed with error"), (NULL));
      return FALSE;
    }
    GST_DEBUG_OBJECT (nvdspreprocess, "Custom tranformation from plugin successfull\n");
  }

  group->sync_obj = params.sync_obj;

  nvtxDomainRangePop (nvdspreprocess->nvtx_domain);

  return ret;
}

static void
release_user_meta_at_batch_level (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  GstNvDsPreProcessBatchMeta *preprocess_batchmeta = (GstNvDsPreProcessBatchMeta *) user_meta->user_meta_data;

  if (preprocess_batchmeta->tensor_meta != nullptr) {
    auto private_data_pair =
        ( std::pair<GstNvDsPreProcess*, NvDsPreProcessCustomBuf*> *) preprocess_batchmeta->tensor_meta->private_data;

    GstNvDsPreProcess* nvdspreprocess = private_data_pair->first;
    NvDsPreProcessCustomBuf* buf = private_data_pair->second;

    NvDsPreProcessAcquirerImpl * acquire_impl = (NvDsPreProcessAcquirerImpl*)nvdspreprocess->acquire_impl.get();
    if (acquire_impl != nullptr)
      acquire_impl->release(buf);
    delete private_data_pair;
    delete preprocess_batchmeta->tensor_meta;
  }
  gst_buffer_unref ((GstBuffer *)preprocess_batchmeta->private_data); //unref conversion pool buffer

  for (auto &roi_meta : preprocess_batchmeta->roi_vector) {
    if (roi_meta.classifier_meta_list) {
      g_list_free(roi_meta.classifier_meta_list);
      roi_meta.classifier_meta_list = nullptr;
    }
    if (roi_meta.roi_user_meta_list) {
      g_list_free(roi_meta.roi_user_meta_list);
      roi_meta.roi_user_meta_list = nullptr;
    }
  }

  delete preprocess_batchmeta;
}

static void
attach_user_meta_at_batch_level (GstNvDsPreProcess * nvdspreprocess,
    NvDsPreProcessBatch * batch, CustomTensorParams custom_tensor_params,
    NvDsPreProcessStatus status)
{
  NvDsBatchMeta *batch_meta = NULL;
  NvDsUserMeta *user_meta = NULL;

  /** attach preprocess batchmeta as user meta at batch level */
  GstNvDsPreProcessBatchMeta *preprocess_batchmeta = new GstNvDsPreProcessBatchMeta;

  /* If tensor is prepared */
  if (status == NVDSPREPROCESS_SUCCESS) {
    preprocess_batchmeta->roi_vector.clear();
    preprocess_batchmeta->roi_vector = custom_tensor_params.seq_params.roi_vector;

    preprocess_batchmeta->tensor_meta = new NvDsPreProcessTensorMeta;

    preprocess_batchmeta->tensor_meta->gpu_id = nvdspreprocess->gpu_id;

    preprocess_batchmeta->tensor_meta->private_data = new std::pair(nvdspreprocess, nvdspreprocess->tensor_buf);

    preprocess_batchmeta->tensor_meta->meta_id = nvdspreprocess->meta_id;
    nvdspreprocess->meta_id ++;

    preprocess_batchmeta->tensor_meta->maintain_aspect_ratio = nvdspreprocess->maintain_aspect_ratio;

    preprocess_batchmeta->tensor_meta->raw_tensor_buffer =
              ((NvDsPreProcessCustomBufImpl *)nvdspreprocess->tensor_buf)->memory->dev_memory_ptr;
    preprocess_batchmeta->tensor_meta->tensor_shape = custom_tensor_params.params.network_input_shape;
    preprocess_batchmeta->tensor_meta->buffer_size =  custom_tensor_params.params.buffer_size;
    preprocess_batchmeta->tensor_meta->data_type = custom_tensor_params.params.data_type;
    preprocess_batchmeta->tensor_meta->tensor_name = custom_tensor_params.params.tensor_name;

    GST_DEBUG_OBJECT (nvdspreprocess, "attached network shape in tensor %d : %d : %d : %d\n",
      preprocess_batchmeta->tensor_meta->tensor_shape[0], preprocess_batchmeta->tensor_meta->tensor_shape[1],
      preprocess_batchmeta->tensor_meta->tensor_shape[2], preprocess_batchmeta->tensor_meta->tensor_shape[3]);
  } else {
    preprocess_batchmeta->roi_vector.clear();
    for (guint i = 0; i < batch->units.size(); i++) {
      preprocess_batchmeta->roi_vector.push_back(batch->units[i].roi_meta);
    }
    preprocess_batchmeta->tensor_meta = nullptr; //set tensor meta to nullptr
  }

  preprocess_batchmeta->private_data = batch->converted_buf;
  preprocess_batchmeta->target_unique_ids = nvdspreprocess->target_unique_ids;

  batch_meta = batch->batch_meta;

  /* Acquire user meta from pool */
  user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

  /* Set user meta below */
  user_meta->user_meta_data = preprocess_batchmeta;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_PREPROCESS_BATCH_META;
  user_meta->base_meta.copy_func = NULL;
  user_meta->base_meta.release_func = release_user_meta_at_batch_level;
  user_meta->base_meta.batch_meta = batch_meta;

  nvds_add_user_meta_to_batch (batch_meta, user_meta);

  return;
}

/* Process entire frames in the batched buffer. */
static GstFlowReturn
gst_nvdspreprocess_on_frame (GstNvDsPreProcess * nvdspreprocess, GstBuffer * inbuf,
    NvBufSurface * in_surf)
{
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  std::string nvtx_str;
  std::unique_ptr < NvDsPreProcessBatch > batch = nullptr;

  GstNvDsPreProcessMemory *memory = nullptr;
  GstBuffer *conv_gst_buf = nullptr;

  NvDsBatchMeta *batch_meta = NULL;
  guint num_groups = 0;
  gdouble scale_ratio_x, scale_ratio_y;
  guint  offset_left, offset_top;
  gint idx = 0;

  if (((in_surf->memType == NVBUF_MEM_DEFAULT || in_surf->memType == NVBUF_MEM_CUDA_DEVICE) &&
       ((int)in_surf->gpuId != (int)nvdspreprocess->gpu_id)) ||
      (((int)in_surf->gpuId == (int)nvdspreprocess->gpu_id) && (in_surf->memType == NVBUF_MEM_SYSTEM)))  {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Memory Compatibility Error:Input surface gpu-id doesnt match with configured gpu-id for element,"
         " please allocate input using unified memory, or use same gpu-ids OR,"
         " if same gpu-ids are used ensure appropriate Cuda memories are used"),
        ("surface-gpu-id=%d,%s-gpu-id=%d",in_surf->gpuId,GST_ELEMENT_NAME(nvdspreprocess),
         nvdspreprocess->gpu_id)); \
      return GST_FLOW_ERROR;
  }

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }


  num_groups = nvdspreprocess->nvdspreprocess_groups.size();
  GST_DEBUG_OBJECT(nvdspreprocess, "Num Groups = %d\n", num_groups);
  std::vector<bool> group_present(num_groups, 0);

  for (guint gcnt = 0; gcnt < num_groups; gcnt ++) {
    GstNvDsPreProcessGroup *& preprocess_group = nvdspreprocess->nvdspreprocess_groups[gcnt];
    GST_DEBUG_OBJECT(nvdspreprocess, "num filled in batch meta = %d\n", batch_meta->num_frames_in_batch);
    NvDsMetaList *l_frame = NULL;

    /* Process group only when interval_counter is 0. */
    if (preprocess_group->interval_counter++ % (preprocess_group->interval + 1) > 0) {
      continue;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {

      NvDsFrameMeta *frame_meta = NULL;
      frame_meta = (NvDsFrameMeta *) (l_frame->data);

      gint source_id = frame_meta->source_id;     /* source id of incoming buffer */
      gint batch_index = frame_meta->batch_id;    /* batch id of incoming buffer */
      gint framemeta_map_idx = 0;
      GstNvDsPreProcessFrame preprocess_frame;
      std::vector<NvDsRoiMeta> roi_vector;
      NvDsRoiMeta roi_meta;
      NvOSD_RectParams rect_params;

      std::vector <gint> src_ids = preprocess_group->src_ids;

      if (src_ids[0] == -1) {
        framemeta_map_idx=preprocess_group->replicated_src_id;
      }

      if (std::find(src_ids.begin(), src_ids.end(), source_id) == src_ids.end() && src_ids[0] != -1) {
        GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : No Source %d => skipping\n", gcnt, source_id);
        continue;
      }

      /*if source_id is present in config file */
      if (nvdspreprocess->src_to_group_map->find(source_id)!= nvdspreprocess->src_to_group_map->end()) {
        /*if source_id belongs to different */
        if(nvdspreprocess->src_to_group_map->at(source_id)!=gint(gcnt)){
          GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : No Source %d => skipping\n", gcnt, source_id);
          continue;
        }
        /*if source_id belongs to same group */
        else {
          framemeta_map_idx=source_id;
        }
      }

      GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Processsing Source ID = %d \n", gcnt, source_id);

      g_mutex_lock (&nvdspreprocess->framemeta_map_lock);
      auto get_preprocess_frame_meta = preprocess_group->framemeta_map.find(framemeta_map_idx);

      if (get_preprocess_frame_meta == preprocess_group->framemeta_map.end() && src_ids[0] != -1) {
        g_print("Group %d : Configuration for Source ID = %d not found\n", gcnt, source_id);
        flow_ret = GST_FLOW_ERROR;
        g_mutex_unlock (&nvdspreprocess->framemeta_map_lock);
        return flow_ret;
      }
      else {
        preprocess_frame = get_preprocess_frame_meta->second;
        g_mutex_unlock (&nvdspreprocess->framemeta_map_lock);

        roi_vector = preprocess_frame.roi_vector;

        GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Source ID %d : Got roi-vecsize = %ld\n",
            gcnt, source_id, roi_vector.size());

        for (guint n = 0; n < preprocess_frame.roi_vector.size(); n++) {
          roi_meta = roi_vector[n];

          if (preprocess_group->process_on_roi) {
            GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Source ID %d : Processing on ROIS\n", gcnt, source_id);

            /* Clipping the excess width of roi's */
            if(roi_meta.roi.left+roi_meta.roi.width > (in_surf->surfaceList+batch_index)->width){
              roi_meta.roi.width = (in_surf->surfaceList+batch_index)->width-roi_meta.roi.left;
            }

            /* Clipping the excess height of roi's */
            if(roi_meta.roi.top+roi_meta.roi.height > (in_surf->surfaceList+batch_index)->height){
              roi_meta.roi.height = (in_surf->surfaceList+batch_index)->height-roi_meta.roi.top;
            }

            /** Process on ROIs provided from config file */
            rect_params = roi_meta.roi;

            GST_DEBUG_OBJECT(nvdspreprocess, "filling ROI left=%f top=%f width=%f height=%f\n",
                rect_params.left, rect_params.top, rect_params.width, rect_params.height);
          } else {
            GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Source ID %d : Processing on Full Frames\n", gcnt, source_id);

            /** Process on Full Frames */
            rect_params.left = 0;
            rect_params.top = 0;
            rect_params.width = in_surf->surfaceList[batch_index].width;
            rect_params.height = in_surf->surfaceList[batch_index].height;

            roi_meta.roi = rect_params;
            GST_DEBUG_OBJECT(nvdspreprocess, "filling FULL FRAME left=%f top=%f width=%f height=%f\n",
                rect_params.left, rect_params.top, rect_params.width, rect_params.height);
          }

          /* batch is empty*/
          if (batch == nullptr) {
            /* Initialising a new batch*/
            batch.reset (new NvDsPreProcessBatch);
            batch->push_buffer = FALSE;
            batch->event_marker = FALSE;
            batch->inbuf = inbuf;
            batch->inbuf_batch_num = nvdspreprocess->current_batch_num;
            batch->batch_meta = batch_meta;
            batch->scaling_pool_format = nvdspreprocess->scaling_pool_format;

            /* acquiing the conv_gst_buf buffer from scaling_pool which store the transformed output buffer */
            flow_ret =
                gst_buffer_pool_acquire_buffer (nvdspreprocess->scaling_pool, &conv_gst_buf,
                nullptr);

            if (flow_ret != GST_FLOW_OK) {
              return flow_ret;
            }

            /* taking memory from buffer pool */
            memory = gst_nvdspreprocess_buffer_get_memory (conv_gst_buf);
            if (!memory) {
              return GST_FLOW_ERROR;
            }

            /* assigning the pointer to the buffer pool memory to batch */
            batch->converted_buf = conv_gst_buf;
            batch->pitch = memory->surf->surfaceList[0].planeParams.pitch[0];
          }

          idx = batch->units.size ();

          /** Scale the roi part to the network resolution maintaining aspect ratio */
          if (scale_and_fill_data (nvdspreprocess, in_surf->surfaceList + batch_index,
                  &rect_params, scale_ratio_x, scale_ratio_y, offset_left, offset_top,
                  memory->surf, memory->surf->surfaceList + idx,
                  memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
            flow_ret = GST_FLOW_ERROR;
            return flow_ret;
          }
          nvdspreprocess->batch_insurf.memType = in_surf->memType;
          nvdspreprocess->batch_outsurf.memType = memory->surf->memType;
          roi_meta.converted_buffer = (NvBufSurfaceParams *)memory->surf->surfaceList + idx;
          roi_meta.scale_ratio_x = scale_ratio_x;
          roi_meta.scale_ratio_y = scale_ratio_y;
          roi_meta.offset_left = offset_left;
          roi_meta.offset_top = offset_top;
          roi_meta.frame_meta = frame_meta;
          roi_meta.object_meta = NULL;

          if (preprocess_group->draw_roi) {
            NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
            display_meta->num_rects = 1;
            display_meta->rect_params[0].left = rect_params.left;
            display_meta->rect_params[0].top = rect_params.top;
            display_meta->rect_params[0].width = rect_params.width;
            display_meta->rect_params[0].height = rect_params.height;
            display_meta->rect_params[0].border_width = 2;
            display_meta->rect_params[0].border_color = preprocess_group->roi_color;
            nvds_add_display_meta_to_frame(frame_meta, display_meta);
          }

          /* Adding a Unit (ROI/Crop/Full Frame) to the current batch. Set the frames members. */
          NvDsPreProcessUnit unit;
          unit.converted_frame_ptr = memory->frame_memory_ptrs[idx];
          unit.obj_meta = nullptr;
          unit.frame_meta = frame_meta;
          unit.frame_num = unit.frame_meta->frame_num;
          unit.batch_index = batch_index;
          unit.input_surf_params = in_surf->surfaceList + batch_index;
          unit.roi_meta = roi_meta;
          unit.roi_meta.classifier_meta_list = NULL;
          unit.roi_meta.roi_user_meta_list = NULL;

          batch->units.push_back (unit);

          if (preprocess_group->process_on_roi) {
            GST_DEBUG_OBJECT(nvdspreprocess,
              "Group %d : Source ID %d : ROI : max-batch-size = %d batch-units-size = %ld batch_index = %d idx = %d\n",
                gcnt, source_id, nvdspreprocess->max_batch_size, batch->units.size (), batch_index, idx);
          }
          else {
            GST_DEBUG_OBJECT(nvdspreprocess,
              "Group %d : Source ID %d : FULL FRAME : max-batch-size = %d batch-units-size = %ld batch_index = %d idx = %d\n",
                gcnt, source_id, nvdspreprocess->max_batch_size, batch->units.size (), batch_index, idx);
          }

          /** push the batch to the queue if batch units exceeds nvdspreprocess batch size **/
          if (batch->units.size() == nvdspreprocess->max_batch_size) {
            /** transform the group according to num filled from batch_meta */
            if (!group_transformation (nvdspreprocess, preprocess_group)) {
              GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                    ("Group %d : group transformation failed\n", gcnt), (NULL));
              return GST_FLOW_ERROR;
            }

            group_present[gcnt] = 1;
            nvdspreprocess->batch_insurf.numFilled = 0;
            nvdspreprocess->batch_outsurf.numFilled = 0;

            /** wait for async transformation */
            for (guint g_count = 0; g_count < num_groups; g_count ++) {
              GstNvDsPreProcessGroup * group = nvdspreprocess->nvdspreprocess_groups[g_count];
              if (group->custom_transform_function_name.empty() ||
                  !g_strcmp0(group->custom_transform_function_name.c_str(), "CustomAsyncTransformation")) {
                if (group_present[g_count] == 1  && group->sync_obj!=NULL) {
                  batch->sync_objects.push_back(group->sync_obj);
                  group_present[g_count] = 0;
                }
              }
            }

            /* Push the batch info structure in the processing queue and notify the process
             * thread that a new batch has been queued. */
            g_mutex_lock (&nvdspreprocess->preprocess_lock);
            g_queue_push_tail (nvdspreprocess->preprocess_queue, batch.get());
            g_cond_broadcast (&nvdspreprocess->preprocess_cond);
            g_mutex_unlock (&nvdspreprocess->preprocess_lock);

            /* Set batch to nullptr so that a new NvDsPreProcessBatch structure can be allocated if required. */
            conv_gst_buf = nullptr;
            batch.release ();
          }
        }
      }
    }

    /** transform the group according to num filled from batch_meta */
    if (nvdspreprocess->batch_insurf.numFilled > 0) {
      if (!group_transformation (nvdspreprocess, preprocess_group)) {
        GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
              ("Group %d : group transformation failed\n", gcnt), (NULL));
        return GST_FLOW_ERROR;
      }
      group_present[gcnt] = 1;
    }
    nvdspreprocess->batch_insurf.numFilled = 0;
    nvdspreprocess->batch_outsurf.numFilled = 0;
  }

  /* Processing last batch whose size is lesser than max batch size*/
  if (batch != nullptr) {
    /** wait for async transformation */
    for (guint g_count = 0; g_count < num_groups; g_count ++) {
      GstNvDsPreProcessGroup * group = nvdspreprocess->nvdspreprocess_groups[g_count];
      if (group->custom_transform_function_name.empty() ||
          !g_strcmp0(group->custom_transform_function_name.c_str(), "CustomAsyncTransformation")) {
        if (group_present[g_count] == 1 && group->sync_obj!=NULL ) {
          batch->sync_objects.push_back(group->sync_obj);
        }
      }
    }

    /* Push the last batch info structure in the processing queue and notify the process
    * thread that a new batch has been queued. */
    g_mutex_lock (&nvdspreprocess->preprocess_lock);

    g_queue_push_tail (nvdspreprocess->preprocess_queue, batch.get());
    g_cond_broadcast (&nvdspreprocess->preprocess_cond);
    g_mutex_unlock (&nvdspreprocess->preprocess_lock);

    /* Batch subm-itted. Set last batch to nullptr so that a new NvDsPreProcessBatch
    * structure can be allocated if required. */
    conv_gst_buf = nullptr;
    batch.release ();
  }
  return GST_FLOW_OK;
}

/* Function to decide if object should be processed further or not. */
static inline gboolean
should_process_object (GstNvDsPreProcess * nvdspreprocess,GstNvDsPreProcessGroup *& preprocess_group,
    NvDsObjectMeta *object_meta,NvOSD_RectParams roi)
{
  NvOSD_RectParams rect_params = object_meta->rect_params;

  /* skip the object preprocessing if it belongs to different gie id */
  if (nvdspreprocess->operate_on_gie_id > -1 && object_meta->unique_component_id != nvdspreprocess->operate_on_gie_id) {
    return FALSE;
  }

  /* skip the object if object does not belong to secondary classifier class */
  std::vector <gint> operate_on_class_ids = preprocess_group->operate_on_class_ids;
  if (operate_on_class_ids.size() > 0 && (std::find(operate_on_class_ids.begin(), operate_on_class_ids.end(), object_meta->class_id) == operate_on_class_ids.end())) {
    return FALSE;
  }

  /** Skipping the object which are too small for secondary inference */
  if ((preprocess_group->min_input_object_width > 0) &&
      (rect_params.width < preprocess_group->min_input_object_width)) {
    return FALSE;
  }
  if ((preprocess_group->min_input_object_height > 0) &&
      (rect_params.height < preprocess_group->min_input_object_height)) {
    return FALSE;
  }
  /** Skipping the object which are too big for secondary inference */
  if ((preprocess_group->max_input_object_width > 0) &&
      (rect_params.width > preprocess_group->max_input_object_width)) {
    return FALSE;
  }
  if ((preprocess_group->max_input_object_height > 0) &&
      (rect_params.height > preprocess_group->max_input_object_height)) {
    return FALSE;
  }

  /* Process on ROIs provided from config file */
  if (!preprocess_group->process_on_all_objects) {
    /* skip the object if object does not overlap with roi provided */
    if (roi.left + roi.width <= rect_params.left ||
        roi.top + roi.height <= rect_params.top ||
        rect_params.left + rect_params.width <= roi.left ||
        rect_params.top + rect_params.height <= roi.top) {
      return FALSE;
    }
  }

  return TRUE;
}

/* Process on objects in the batched buffer. */
static GstFlowReturn
gst_nvdspreprocess_on_objects (GstNvDsPreProcess * nvdspreprocess, GstBuffer * inbuf,
    NvBufSurface * in_surf)
{
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  std::string nvtx_str;
  std::unique_ptr < NvDsPreProcessBatch > batch = nullptr;

  GstNvDsPreProcessMemory *memory = nullptr;
  GstBuffer *conv_gst_buf = nullptr;

  NvDsBatchMeta *batch_meta = NULL;
  guint num_groups = 0;
  gdouble scale_ratio_x, scale_ratio_y;
  guint  offset_left, offset_top;
  gint idx = 0;

  if (((in_surf->memType == NVBUF_MEM_DEFAULT || in_surf->memType == NVBUF_MEM_CUDA_DEVICE) &&
       ((int)in_surf->gpuId != (int)nvdspreprocess->gpu_id)) ||
      (((int)in_surf->gpuId == (int)nvdspreprocess->gpu_id) && (in_surf->memType == NVBUF_MEM_SYSTEM)))  {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Memory Compatibility Error:Input surface gpu-id doesnt match with configured gpu-id for element,"
         " please allocate input using unified memory, or use same gpu-ids OR,"
         " if same gpu-ids are used ensure appropriate Cuda memories are used"),
        ("surface-gpu-id=%d,%s-gpu-id=%d",in_surf->gpuId,GST_ELEMENT_NAME(nvdspreprocess),
         nvdspreprocess->gpu_id)); \
      return GST_FLOW_ERROR;
  }

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }


  num_groups = nvdspreprocess->nvdspreprocess_groups.size();
  GST_DEBUG_OBJECT(nvdspreprocess, "Num Groups = %d\n", num_groups);
  std::vector<bool> group_present(num_groups, 0);

  for (guint gcnt = 0; gcnt < num_groups; gcnt ++) {
    GstNvDsPreProcessGroup *& preprocess_group = nvdspreprocess->nvdspreprocess_groups[gcnt];
    GST_DEBUG_OBJECT(nvdspreprocess, "num filled in batch meta = %d\n", batch_meta->num_frames_in_batch);
    NvDsMetaList *l_frame = NULL;

    /* Process group only when interval_counter is 0. */
    if (preprocess_group->interval_counter++ % (preprocess_group->interval + 1) > 0) {
      continue;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {

      NvDsFrameMeta *frame_meta = NULL;
      frame_meta = (NvDsFrameMeta *) (l_frame->data);

      gint source_id = frame_meta->source_id;     /* source id of incoming buffer */
      gint batch_index = frame_meta->batch_id;    /* batch id of incoming buffer */
      gint framemeta_map_idx = 0;

      GstNvDsPreProcessFrame preprocess_frame;
      std::vector<NvDsRoiMeta> roi_vector;
      NvDsRoiMeta roi_meta;
      std::vector <gint> src_ids = preprocess_group->src_ids;

      if (src_ids[0] == -1) {
        framemeta_map_idx=preprocess_group->replicated_src_id;
      }

      if (std::find(src_ids.begin(), src_ids.end(), source_id) == src_ids.end() && src_ids[0] != -1) {
        GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : No Source %d => skipping\n", gcnt, source_id);
        continue;
      }

      /*if source_id is present in config file */
      if (nvdspreprocess->src_to_group_map->find(source_id)!= nvdspreprocess->src_to_group_map->end()) {
        /*if source_id belongs to different */
        if(nvdspreprocess->src_to_group_map->at(source_id)!=gint(gcnt)){
          GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : No Source %d => skipping\n", gcnt, source_id);
          continue;
        }
        /*if source_id belongs to same group */
        else {
          framemeta_map_idx=source_id;
        }
      }

      GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Processsing Source ID = %d \n", gcnt, source_id);

      g_mutex_lock (&nvdspreprocess->framemeta_map_lock);
      auto get_preprocess_frame_meta = preprocess_group->framemeta_map.find(framemeta_map_idx);

      if (get_preprocess_frame_meta == preprocess_group->framemeta_map.end() && src_ids[0] != -1) {
        g_print("Group %d : Configuration for Source ID = %d not found\n", gcnt, source_id);
        flow_ret = GST_FLOW_ERROR;
        g_mutex_unlock (&nvdspreprocess->framemeta_map_lock);
        return flow_ret;
      }
      else {
        preprocess_frame = get_preprocess_frame_meta->second;
        g_mutex_unlock (&nvdspreprocess->framemeta_map_lock);

        roi_vector = preprocess_frame.roi_vector;

        GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Source ID %d : Got roi-vecsize = %ld\n",
            gcnt, source_id, roi_vector.size());

        for (guint n = 0; n < preprocess_frame.roi_vector.size(); n++) {
          roi_meta = roi_vector[n];

          /* Secondary Classification only on selected roi's */
          if (!preprocess_group->process_on_all_objects) {
            GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Source ID %d : Processing on ROIS\n", gcnt, source_id);

            /* Clipping the excess width of roi's */
            if(roi_meta.roi.left+roi_meta.roi.width > (in_surf->surfaceList+batch_index)->width){
              roi_meta.roi.width = (in_surf->surfaceList+batch_index)->width-roi_meta.roi.left;
            }

            /* Clipping the excess height of roi's */
            if(roi_meta.roi.top+roi_meta.roi.height > (in_surf->surfaceList+batch_index)->height){
              roi_meta.roi.height = (in_surf->surfaceList+batch_index)->height-roi_meta.roi.top;
            }

            GST_DEBUG_OBJECT(nvdspreprocess, "filling ROI left=%f top=%f width=%f height=%f\n",
                roi_meta.roi.left, roi_meta.roi.top, roi_meta.roi.width, roi_meta.roi.height);
          } else {
            GST_DEBUG_OBJECT (nvdspreprocess, "Group %d : Source ID %d : Processing on Full Frames\n", gcnt, source_id);

            /** Process on Full Frames */
            NvOSD_RectParams full_frame;
            full_frame.left = 0;
            full_frame.top = 0;
            full_frame.width = in_surf->surfaceList[batch_index].width;
            full_frame.height = in_surf->surfaceList[batch_index].height;

            roi_meta.roi = full_frame;
            GST_DEBUG_OBJECT(nvdspreprocess, "filling FULL FRAME left=%f top=%f width=%f height=%f\n",
                roi_meta.roi.left, roi_meta.roi.top, roi_meta.roi.width, roi_meta.roi.height);
          }

          if (preprocess_group->draw_roi) {
            /* drawing roi rectangle*/
            NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
            display_meta->num_rects = 1;
            display_meta->rect_params[0].left = roi_meta.roi.left;
            display_meta->rect_params[0].top = roi_meta.roi.top;
            display_meta->rect_params[0].width = roi_meta.roi.width;
            display_meta->rect_params[0].height = roi_meta.roi.height;
            display_meta->rect_params[0].border_width = 2;
            display_meta->rect_params[0].border_color = preprocess_group->roi_color;
            nvds_add_display_meta_to_frame(frame_meta, display_meta);
          }

          for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
            l_obj = l_obj->next) {
            NvDsObjectMeta *object_meta = (NvDsObjectMeta *) (l_obj->data);
            NvOSD_RectParams rect_params = object_meta->rect_params;

            /* skip the object if it should not be processed further*/
            if (!should_process_object (nvdspreprocess,preprocess_group,object_meta,roi_meta.roi)) {
              continue;
            }

            /* batch is empty*/
            if (batch == nullptr) {
              /* Initialising a new batch*/
              batch.reset (new NvDsPreProcessBatch);
              batch->push_buffer = FALSE;
              batch->event_marker = FALSE;
              batch->inbuf = inbuf;
              batch->inbuf_batch_num = nvdspreprocess->current_batch_num;
              batch->batch_meta = batch_meta;
              batch->scaling_pool_format = nvdspreprocess->scaling_pool_format;

              /* acquiing the conv_gst_buf buffer from scaling_pool which store the transformed output buffer */
              flow_ret =
                  gst_buffer_pool_acquire_buffer (nvdspreprocess->scaling_pool, &conv_gst_buf,
                  nullptr);

              if (flow_ret != GST_FLOW_OK) {
                return flow_ret;
              }

              /* taking memory from buffer pool */
              memory = gst_nvdspreprocess_buffer_get_memory (conv_gst_buf);
              if (!memory) {
                return GST_FLOW_ERROR;
              }

              /* assigning the pointer to the buffer pool memory to batch */
              batch->converted_buf = conv_gst_buf;
              batch->pitch = memory->surf->surfaceList[0].planeParams.pitch[0];
            }

            idx = batch->units.size ();

            /** Scale the object part to the network resolution maintaining aspect ratio */
            if (scale_and_fill_data (nvdspreprocess, in_surf->surfaceList + batch_index,
                    &rect_params, scale_ratio_x, scale_ratio_y, offset_left, offset_top,
                    memory->surf, memory->surf->surfaceList + idx,
                    memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
              flow_ret = GST_FLOW_ERROR;
              return flow_ret;
            }
            nvdspreprocess->batch_insurf.memType = in_surf->memType;
            nvdspreprocess->batch_outsurf.memType = memory->surf->memType;

            NvDsRoiMeta obj_roi_meta;
            obj_roi_meta.roi = rect_params;
            obj_roi_meta.converted_buffer = (NvBufSurfaceParams *)memory->surf->surfaceList + idx;
            obj_roi_meta.scale_ratio_x = scale_ratio_x;
            obj_roi_meta.scale_ratio_y = scale_ratio_y;
            obj_roi_meta.offset_left = offset_left;
            obj_roi_meta.offset_top = offset_top;
            obj_roi_meta.frame_meta = frame_meta;
            obj_roi_meta.object_meta = object_meta;

            /* Adding a Unit (Full objects/Cropped objects) to the current batch. Set the frames members. */
            NvDsPreProcessUnit unit;
            unit.converted_frame_ptr = memory->frame_memory_ptrs[idx];
            unit.obj_meta = nullptr;
            unit.frame_meta = frame_meta;
            unit.frame_num = unit.frame_meta->frame_num;
            unit.batch_index = batch_index;
            unit.input_surf_params = in_surf->surfaceList + batch_index;
            unit.roi_meta = obj_roi_meta;
            unit.roi_meta.classifier_meta_list = NULL;
            unit.roi_meta.roi_user_meta_list = NULL;

            batch->units.push_back (unit);

            if (preprocess_group->process_on_roi) {
              GST_DEBUG_OBJECT(nvdspreprocess,
                "Group %d : Source ID %d : ROI : max-batch-size = %d batch-units-size = %ld batch_index = %d idx = %d\n",
                  gcnt, source_id, nvdspreprocess->max_batch_size, batch->units.size (), batch_index, idx);
            }
            else {
              GST_DEBUG_OBJECT(nvdspreprocess,
                "Group %d : Source ID %d : FULL FRAME : max-batch-size = %d batch-units-size = %ld batch_index = %d idx = %d\n",
                  gcnt, source_id, nvdspreprocess->max_batch_size, batch->units.size (), batch_index, idx);
            }

            /** push the batch to the queue if batch units exceeds nvdspreprocess batch size **/
            if (batch->units.size() == nvdspreprocess->max_batch_size) {
              /** transform the group according to num filled from batch_meta */
              if (!group_transformation (nvdspreprocess, preprocess_group)) {
                GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                      ("Group %d : group transformation failed\n", gcnt), (NULL));
                return GST_FLOW_ERROR;
              }

              group_present[gcnt] = 1;
              nvdspreprocess->batch_insurf.numFilled = 0;
              nvdspreprocess->batch_outsurf.numFilled = 0;

              /** wait for async transformation */
              for (guint g_count = 0; g_count < num_groups; g_count ++) {
                GstNvDsPreProcessGroup * group = nvdspreprocess->nvdspreprocess_groups[g_count];
                if (group->custom_transform_function_name.empty() ||
                    !g_strcmp0(group->custom_transform_function_name.c_str(), "CustomAsyncTransformation")) {
                  if (group_present[g_count] == 1  && group->sync_obj!=NULL) {
                    batch->sync_objects.push_back(group->sync_obj);
                    group_present[g_count] = 0;
                  }
                }
              }

              /* Push the batch info structure in the processing queue and notify the process
              * thread that a new batch has been queued. */
              g_mutex_lock (&nvdspreprocess->preprocess_lock);
              g_queue_push_tail (nvdspreprocess->preprocess_queue, batch.get());
              g_cond_broadcast (&nvdspreprocess->preprocess_cond);
              g_mutex_unlock (&nvdspreprocess->preprocess_lock);

              /* Set batch to nullptr so that a new NvDsPreProcessBatch structure can be allocated if required. */
              conv_gst_buf = nullptr;
              batch.release ();
            }
          }
        }
      }
    }

    /** transform the group according to num filled from batch_meta */
    if (nvdspreprocess->batch_insurf.numFilled > 0) {
      if (!group_transformation (nvdspreprocess, preprocess_group)) {
        GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
              ("Group %d : group transformation failed\n", gcnt), (NULL));
        return GST_FLOW_ERROR;
      }
      group_present[gcnt] = 1;
    }
    nvdspreprocess->batch_insurf.numFilled = 0;
    nvdspreprocess->batch_outsurf.numFilled = 0;
  }

  /* Processing last batch whose size is lesser than max batch size*/
  if (batch != nullptr) {
    /** wait for async transformation */
    for (guint g_count = 0; g_count < num_groups; g_count ++) {
      GstNvDsPreProcessGroup * group = nvdspreprocess->nvdspreprocess_groups[g_count];
      if (group->custom_transform_function_name.empty() ||
          !g_strcmp0(group->custom_transform_function_name.c_str(), "CustomAsyncTransformation")) {
        if (group_present[g_count] == 1 && group->sync_obj!=NULL ) {
          batch->sync_objects.push_back(group->sync_obj);
        }
      }
    }

    /* Push the last batch info structure in the processing queue and notify the process
    * thread that a new batch has been queued. */
    g_mutex_lock (&nvdspreprocess->preprocess_lock);

    g_queue_push_tail (nvdspreprocess->preprocess_queue, batch.get());
    g_cond_broadcast (&nvdspreprocess->preprocess_cond);
    g_mutex_unlock (&nvdspreprocess->preprocess_lock);

    /* Batch subm-itted. Set last batch to nullptr so that a new NvDsPreProcessBatch
    * structure can be allocated if required. */
    conv_gst_buf = nullptr;
    batch.release ();
  }
  return GST_FLOW_OK;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvdspreprocess_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (btrans);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  NvDsPreProcessBatch *buf_push_batch;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  std::string nvtx_str;
  std::unique_ptr < NvDsPreProcessBatch > batch = nullptr;
  cudaError_t cudaReturn;

  nvdspreprocess->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_TEAL_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(nvdspreprocess->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(nvdspreprocess->nvtx_domain, &eventAttrib);

  cudaReturn = cudaSetDevice (nvdspreprocess->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvdspreprocess->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
    return GST_FLOW_ERROR;
  }

  if (FALSE == nvdspreprocess->config_file_parse_successful) {
    GST_ELEMENT_ERROR (nvdspreprocess, LIBRARY, SETTINGS,
        ("Configuration file parsing failed\n"),
        ("Config file path: %s\n", nvdspreprocess->config_file_path));
    return flow_ret;
  }

  if (FALSE == nvdspreprocess->enable){
    GST_DEBUG_OBJECT (nvdspreprocess, "nvdspreprocess in passthrough mode\n");
    flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD (nvdspreprocess), inbuf);
    return flow_ret;
  }

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (nvdspreprocess));

  /** Preprocess on Frames */
  if (nvdspreprocess->process_on_frame) {
    flow_ret = gst_nvdspreprocess_on_frame (nvdspreprocess, inbuf, in_surf);
  } else {
    flow_ret = gst_nvdspreprocess_on_objects (nvdspreprocess, inbuf, in_surf);
  }

  if (flow_ret != GST_FLOW_OK)
    goto error;

  nvtxDomainRangeEnd(nvdspreprocess->nvtx_domain, buf_process_range);

  /* Queue a push buffer batch. This batch is not inferred. This batch is to
   * signal the process thread that there are no more batches
   * belonging to this input buffer and this GstBuffer can be pushed to
   * downstream element once all the previous processing is done. */
  buf_push_batch = new NvDsPreProcessBatch;
  buf_push_batch->inbuf = inbuf;
  buf_push_batch->push_buffer = TRUE;
  buf_push_batch->nvtx_complete_buf_range = buf_process_range;

  g_mutex_lock (&nvdspreprocess->preprocess_lock);
  /* Check if this is a push buffer or event marker batch. If yes, no need to
   * queue the input for inferencing. */
  if (buf_push_batch->push_buffer) {
    /* Push the batch info structure in the processing queue and notify the
     * process thread that a new batch has been queued. */
    g_queue_push_tail (nvdspreprocess->preprocess_queue, buf_push_batch);
    g_cond_broadcast (&nvdspreprocess->preprocess_cond);
  }
  g_mutex_unlock (&nvdspreprocess->preprocess_lock);

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
gst_nvdspreprocess_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (btrans);
  return nvdspreprocess->last_flow_ret;
}

NvDsPreProcessAcquirerImpl::NvDsPreProcessAcquirerImpl(GstBufferPool *pool)
{
  m_gstpool = pool;
}

NvDsPreProcessCustomBuf* NvDsPreProcessAcquirerImpl::acquire()
{
  GstBuffer *gstbuf;
  GstNvDsPreProcessMemory *memory;
  GstFlowReturn flow_ret;

  flow_ret =
      gst_buffer_pool_acquire_buffer (m_gstpool, &gstbuf,
      nullptr);

  if (flow_ret != GST_FLOW_OK) {
    GST_ERROR ("error while acquiring buffer from tensor pool\n");
    return nullptr;
  }

  memory = gst_nvdspreprocess_buffer_get_memory (gstbuf);
  if (!memory) {
    GST_ERROR ("error while getting memory from tensor pool\n");
    return nullptr;
  }

  return new NvDsPreProcessCustomBufImpl {{memory->dev_memory_ptr}, gstbuf, memory};
}

gboolean NvDsPreProcessAcquirerImpl::release(NvDsPreProcessCustomBuf* buf)
{
  NvDsPreProcessCustomBufImpl *implBuf = (NvDsPreProcessCustomBufImpl*)(buf);
  gst_buffer_unref(implBuf->gstbuf);
  delete implBuf;
  return TRUE;
}

/**
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_nvdspreprocess_output_loop (gpointer data)
{
  GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS (data);
  NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;
  cudaError_t cudaReturn;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_TEAL_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;
  nvtx_str =
      "gst-nvdspreprocess_output-loop_uid=" + std::to_string (nvdspreprocess->unique_id);

  cudaReturn = cudaSetDevice (nvdspreprocess->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdspreprocess, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvdspreprocess->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  g_mutex_lock (&nvdspreprocess->preprocess_lock);

  /* Run till signalled to stop. */
  while (!nvdspreprocess->stop) {
    std::unique_ptr < NvDsPreProcessBatch > batch = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (nvdspreprocess->preprocess_queue)) {
      g_cond_wait (&nvdspreprocess->preprocess_cond, &nvdspreprocess->preprocess_lock);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((NvDsPreProcessBatch *)
        g_queue_pop_head (nvdspreprocess->preprocess_queue));
    g_cond_broadcast (&nvdspreprocess->preprocess_cond);

    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    g_mutex_unlock (&nvdspreprocess->preprocess_lock);

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(nvdspreprocess->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp (batch->inbuf,
          GST_ELEMENT_NAME (nvdspreprocess));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvdspreprocess),
          batch->inbuf);
      if (nvdspreprocess->last_flow_ret != flow_ret) {
        switch (flow_ret) {
            /* Signal the application for pad push errors by posting a error message
             * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (nvdspreprocess, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)",
                    gst_flow_get_name (flow_ret), flow_ret));
            break;
          default:
            break;
        }
      }
      nvdspreprocess->last_flow_ret = flow_ret;
      nvdspreprocess->meta_id = 0;
      g_mutex_lock (&nvdspreprocess->preprocess_lock);
      continue;
    }

    for(auto sync_object: batch->sync_objects){
      NvBufSurfTransformSyncObjWait(sync_object, -1);
      NvBufSurfTransformSyncObjDestroy(&sync_object);
    }

#ifdef DUMP_ROIS
    GstNvDsPreProcessMemory *memory = nullptr;
    memory = gst_nvdspreprocess_buffer_get_memory (batch->converted_buf);
    dump_rois(nvdspreprocess, batch.get(), memory->surf);
#endif

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(nvdspreprocess->nvtx_domain, &eventAttrib);

    CustomTensorParams custom_tensor_params;
    /** Tensor Preparation from Custom Library */
    if (nvdspreprocess->custom_lib_path && nvdspreprocess->custom_lib_handle && nvdspreprocess->custom_tensor_function) {
      custom_tensor_params.params = nvdspreprocess->tensor_params;
      custom_tensor_params.params.buffer_size = custom_tensor_params.params.buffer_size * batch->units.size()/nvdspreprocess->tensor_params.network_input_shape[0];
      custom_tensor_params.seq_params.roi_vector.clear();

      for (guint i = 0; i < batch->units.size(); i++) {
        custom_tensor_params.seq_params.roi_vector.push_back(batch->units[i].roi_meta);
      }

      status = nvdspreprocess->custom_tensor_function(nvdspreprocess->custom_lib_ctx, batch.get(),
                                                      nvdspreprocess->tensor_buf, custom_tensor_params,
                                                      nvdspreprocess->acquire_impl.get());
    }

#ifdef DEBUG_TENSOR
    static int batch_num = 0;
    std::ofstream outfile("tensorout_batch_" + std::to_string(batch_num) + ".bin");
    outfile.write((char *)((NvDsPreProcessCustomBufImpl *)nvdspreprocess->tensor_buf)->memory->dev_memory_ptr,
          custom_tensor_params.params.buffer_size);
    outfile.close();
    batch_num ++;
#endif

    /** attach user meta at batch level */
    attach_user_meta_at_batch_level (nvdspreprocess, batch.get(), custom_tensor_params, status);

    g_mutex_lock (&nvdspreprocess->preprocess_lock);

    nvtxDomainRangePop (nvdspreprocess->nvtx_domain);
  }
  g_mutex_unlock (&nvdspreprocess->preprocess_lock);

  return nullptr;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdspreprocess_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdspreprocess_debug, "nvdspreprocess", 0,
      "preprocess plugin");

  return gst_element_register (plugin, "nvdspreprocess", GST_RANK_PRIMARY,
      GST_TYPE_NVDSPREPROCESS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_preprocess,
    DESCRIPTION, nvdspreprocess_plugin_init, "9.0", LICENSE, BINARY_PACKAGE,
    URL)
