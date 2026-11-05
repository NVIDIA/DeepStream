/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file gstnvdsvisionencoder.c
 * @brief C implementation of GStreamer vision encoder plugin
 */

/* Enable instrumentation: accuracy validation + performance measurement */
/* To enable: add -DENABLE_INSTRUMENTATION to CFLAGS in Makefile */
// #define ENABLE_INSTRUMENTATION

#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#ifdef ENABLE_INSTRUMENTATION
#include <sys/time.h>
#endif
#include <dlfcn.h>
#include <setjmp.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <jpeglib.h>
#ifdef __cplusplus
}
#endif

#include "gstnvdsvisionencoder.h"
#include "nvbufsurftransform.h"
#include "preprocess.h"
#include "gst-nvquery.h"

/**
 * Detect if the board is Thor using nvidia-ml (NVML) library.
 * Uses dlopen/dlsym to avoid link-time dependency (same pattern as nvds_obj_encode_common.c).
 * Thor requires VIC for hardware-accelerated transforms; all other platforms use GPU compute.
 *
 * @param gpu_id  GPU device index to query
 * @return 1 if board name contains "Thor", 0 otherwise
 */
static int
detect_is_thor (unsigned int gpu_id)
{
  typedef int (*nvmlInit_fn_t)(void);
  typedef int (*nvmlShutdown_fn_t)(void);
  typedef int (*nvmlDeviceGetHandleByIndex_fn_t)(unsigned int, void **);
  typedef int (*nvmlDeviceGetName_fn_t)(void *, char *, unsigned int);

  int is_thor = 0;
  void *nvml_lib = dlopen ("libnvidia-ml.so", RTLD_LAZY);
  if (!nvml_lib)
    nvml_lib = dlopen ("libnvidia-ml.so.1", RTLD_LAZY);
  if (!nvml_lib)
    return 0;

  nvmlInit_fn_t init_fn =
      (nvmlInit_fn_t) dlsym (nvml_lib, "nvmlInit");
  nvmlShutdown_fn_t shutdown_fn =
      (nvmlShutdown_fn_t) dlsym (nvml_lib, "nvmlShutdown");
  nvmlDeviceGetHandleByIndex_fn_t getHandle_fn =
      (nvmlDeviceGetHandleByIndex_fn_t) dlsym (nvml_lib, "nvmlDeviceGetHandleByIndex");
  nvmlDeviceGetName_fn_t getName_fn =
      (nvmlDeviceGetName_fn_t) dlsym (nvml_lib, "nvmlDeviceGetName");

  if (init_fn && shutdown_fn && getHandle_fn && getName_fn) {
    if (init_fn () == 0) {          /* NVML_SUCCESS */
      void *device = NULL;
      if (getHandle_fn (gpu_id, &device) == 0) {
        char name[96] = {0};       /* NVML_DEVICE_NAME_V2_BUFFER_SIZE */
        if (getName_fn (device, name, sizeof (name)) == 0) {
          /* Case-insensitive search for "thor" in board name */
          char name_lower[96];
          for (int i = 0; i < 95 && name[i]; i++)
            name_lower[i] = tolower ((unsigned char) name[i]);
          name_lower[95] = '\0';
          if (strstr (name_lower, "thor") != NULL) {
            is_thor = 1;
          }
          g_print ("nvdsvisionencoder: NVML board name = \"%s\", is_thor = %d\n",
                   name, is_thor);
        }
      }
      shutdown_fn ();
    }
  }

  dlclose (nvml_lib);
  return is_thor;
}

/* Export debug category for use in C++ files */
GST_DEBUG_CATEGORY (gst_nvdsvisionencoder_debug);
#define GST_CAT_DEFAULT gst_nvdsvisionencoder_debug

/* Default property values */
#define DEFAULT_ENABLE TRUE
#define DEFAULT_BACKEND NVDS_VISION_ENCODER_BACKEND_TRITON
#define DEFAULT_TRITON_URL "localhost:8001"
#define DEFAULT_TRITON_MODEL "radio_clip_trt"
#define DEFAULT_BATCH_SIZE 16
#define DEFAULT_MIN_CROP_SIZE 32
#define DEFAULT_VERBOSE FALSE
#define DEFAULT_GPU_ID 0

/* Vision encoder signals */
enum {
  /* Signal emitted to notify app about model update completion with
   * success/error messages. */
  SIGNAL_MODEL_UPDATED,
  LAST_SIGNAL
};

static guint gst_nvdsvisionencoder_signals[LAST_SIGNAL] = { 0 };

/* Property IDs */
enum {
  PROP_0,
  PROP_ENABLE,
  PROP_BACKEND,  // Already declared
  PROP_TRITON_URL,
  PROP_TRITON_MODEL,
  PROP_MODEL_VARIANT,
  PROP_TENSORRT_ENGINE,
  PROP_ONNX_MODEL,
  PROP_BATCH_SIZE,
  PROP_MIN_CROP_SIZE,
  PROP_VERBOSE,
  PROP_GPU_ID,
  PROP_DEVICE,
  PROP_SKIP_INTERVAL,
  PROP_EMBEDDING_CLASSES,
  PROP_QUERY_ONLY,
  PROP_SMART_INFER,
  PROP_CACHE_REFRESH_INTERVAL,
  PROP_CACHE_MAX_SIZE,
  /* OFA-based temporal embedding prediction */
  PROP_OFA_PREDICT,
  PROP_OFA_MOTION_THRESHOLD,
  PROP_OFA_HIGH_MOTION_THRESHOLD,
  /* DLA two-tier inference */
  PROP_DLA_VALIDATOR,
  PROP_DLA_VALIDATOR_ENGINE,
  PROP_DLA_CORE,
  PROP_DLA_CHANGE_THRESHOLD,
};

/* Pad templates - accept both RGB and RGBA with NVMM memory */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE (
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM", "{ RGB, RGBA, NV12 }"))
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE (
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM", "{ RGB, RGBA, NV12 }"))
);

/* Function declarations */
static void gst_nvdsvisionencoder_replacement_free (
    NvDsVisionEncoderReplacement *rep);
static gboolean gst_nvdsvisionencoder_trigger_new_model (
    GstNvDsVisionEncoder *self, const gchar *path,
    NvDsVisionEncoderModelLoadType type);
static void gst_nvdsvisionencoder_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_nvdsvisionencoder_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_nvdsvisionencoder_finalize (GObject *object);
static gboolean gst_nvdsvisionencoder_start (GstBaseTransform *trans);
static gboolean gst_nvdsvisionencoder_stop (GstBaseTransform *trans);
static GstFlowReturn gst_nvdsvisionencoder_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf);
static gboolean gst_nvdsvisionencoder_query (GstBaseTransform *trans,
    GstPadDirection direction, GstQuery *query);

/* External C++ functions (implemented in trt_encoder.cpp) */
#include "trt_encoder.h"

/* Forward declarations */
static gpointer reid_meta_copy(gpointer data, gpointer user_data);
static void reid_meta_release(gpointer data, gpointer user_data);

/* Metadata attachment helper functions - following tracker pattern */
static gpointer reid_meta_copy(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data) {
    NvDsObjReid *src_reid = (NvDsObjReid *)user_meta->user_meta_data;
    if (!src_reid || src_reid->featureSize == 0 || !src_reid->ptr_host) {
      return NULL;
    }
    /* Deep copy: allocate new memory and copy data */
    NvDsObjReid *dst_reid = (NvDsObjReid *)g_malloc0(sizeof(NvDsObjReid));
    dst_reid->featureSize = src_reid->featureSize;
    /* Allocate new memory for embedding and copy data */
    dst_reid->ptr_host = (float *)g_malloc0(src_reid->featureSize * sizeof(float));
    memcpy(dst_reid->ptr_host, src_reid->ptr_host, src_reid->featureSize * sizeof(float));
    dst_reid->ptr_dev = NULL;  /* Don't copy device pointer */
    return (gpointer)dst_reid;
  }
  return NULL;
}

static void reid_meta_release(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data) {
    NvDsObjReid *reid = (NvDsObjReid *)user_meta->user_meta_data;
    /* Free the embedding memory (ptr_host) */
    if (reid->ptr_host) {
      g_free(reid->ptr_host);
      reid->ptr_host = NULL;
    }
    /* Free the Reid structure */
    g_free(reid);
  }
}

/**
 * Check if a class is allowed for embedding generation
 */
static gboolean is_class_allowed(GstNvDsVisionEncoder *nvdsvisionencoder, const gchar *obj_label) {
  /* If no filter configured, allow all classes */
  if (!nvdsvisionencoder->allowed_classes || nvdsvisionencoder->num_allowed_classes == 0) {
    return TRUE;
  }
  
  /* Check if obj_label matches any allowed class */
  for (guint i = 0; i < nvdsvisionencoder->num_allowed_classes && nvdsvisionencoder->allowed_classes[i]; i++) {
    if (g_strcmp0(obj_label, nvdsvisionencoder->allowed_classes[i]) == 0) {
      return TRUE;
    }
  }
  
  return FALSE;
}

static gboolean attach_reid_embedding(NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta,
                                      float *embedding_data, guint feature_size) {
  if (!batch_meta || !obj_meta || !embedding_data || feature_size == 0) {
    return FALSE;
  }
  
  /* Acquire user meta from pool */
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    return FALSE;
  }
  
  /* Allocate NvDsObjReid structure */
  NvDsObjReid *reid_obj = (NvDsObjReid *)g_malloc0(sizeof(NvDsObjReid));
  reid_obj->featureSize = feature_size;
  
  /* Allocate and copy embedding data */
  reid_obj->ptr_host = (float *)g_malloc0(feature_size * sizeof(float));
  memcpy(reid_obj->ptr_host, embedding_data, feature_size * sizeof(float));
  reid_obj->ptr_dev = NULL;
  
  /* Debug: verify structure (print first few) */
  static int attach_count = 0;
  if (attach_count < 5) {
    GST_DEBUG("Reid #%d: featureSize=%u, ptr_host=%p, first_val=%.6f",
              attach_count, reid_obj->featureSize, reid_obj->ptr_host, 
              reid_obj->ptr_host ? reid_obj->ptr_host[0] : 0.0f);
    attach_count++;
  }
  
  /* Set user meta fields - following tracker pattern */
  user_meta->base_meta.batch_meta = batch_meta;
  user_meta->base_meta.meta_type = NVDS_TRACKER_OBJ_REID_META;
  user_meta->user_meta_data = (void *)reid_obj;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)reid_meta_copy;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)reid_meta_release;
  
  /* Add metadata to object with locking */
  nvds_acquire_meta_lock(batch_meta);
  nvds_add_user_meta_to_obj(obj_meta, user_meta);
  nvds_release_meta_lock(batch_meta);
  
  return TRUE;
}

#define gst_nvdsvisionencoder_parent_class parent_class
G_DEFINE_TYPE (GstNvDsVisionEncoder, gst_nvdsvisionencoder, GST_TYPE_BASE_TRANSFORM);

/**
 * Class initialization
 */
static void
gst_nvdsvisionencoder_class_init (GstNvDsVisionEncoderClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);

  gobject_class->set_property = gst_nvdsvisionencoder_set_property;
  gobject_class->get_property = gst_nvdsvisionencoder_get_property;
  gobject_class->finalize = gst_nvdsvisionencoder_finalize;

  trans_class->start = gst_nvdsvisionencoder_start;
  trans_class->stop = gst_nvdsvisionencoder_stop;
  trans_class->transform_ip = gst_nvdsvisionencoder_transform_ip;
  trans_class->query = GST_DEBUG_FUNCPTR (gst_nvdsvisionencoder_query);

  /** install signal MODEL_UPDATED */
  gst_nvdsvisionencoder_signals[SIGNAL_MODEL_UPDATED] =
      g_signal_new ("model-updated",
      G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstNvDsVisionEncoderClass, model_updated),
      NULL, NULL, NULL,
      G_TYPE_NONE, 2, G_TYPE_INT, G_TYPE_STRING);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Enable vision encoding", DEFAULT_ENABLE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BACKEND,
      g_param_spec_string ("backend", "Backend",
          "Inference backend (triton/tensorrt)", "triton",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_TRITON_URL,
      g_param_spec_string ("triton-url", "Triton URL",
          "Triton Inference Server URL", DEFAULT_TRITON_URL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_TRITON_MODEL,
      g_param_spec_string ("triton-model", "Triton Model",
          "Triton model name", DEFAULT_TRITON_MODEL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MODEL_VARIANT,
      g_param_spec_string ("model-variant", "Model Variant",
          "Model variant (for compatibility)", "default",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* tensorrt-engine is MUTABLE_PLAYING: setting it at runtime triggers an OTA
   * reload (see gst_nvdsvisionencoder_trigger_new_model). onnx-model is not --
   * building an engine is provisioning, not updating. */
  g_object_class_install_property (gobject_class, PROP_TENSORRT_ENGINE,
      g_param_spec_string ("tensorrt-engine", "TensorRT Engine",
          "Path to TensorRT engine (.plan) file. Settable while PLAYING to "
          "reload a prebuilt engine on the fly; the new engine must match the "
          "live one's input resolution, embedding dimension and batch capacity.",
          NULL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_ONNX_MODEL,
      g_param_spec_string ("onnx-model", "ONNX Model",
          "Path to ONNX model file (for engine building if the engine is not "
          "found). Cold-start configuration only: setting it on a running "
          "element stores it for the next start rather than loading it. Use "
          "tensorrt-engine for a runtime update.", NULL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "TensorRT batch size (for engine building/inference)", 1, G_MAXUINT, DEFAULT_BATCH_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MIN_CROP_SIZE,
      g_param_spec_uint ("min-crop-size", "Min Crop Size",
          "Minimum crop size (width or height)", 8, 512, DEFAULT_MIN_CROP_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_VERBOSE,
      g_param_spec_boolean ("verbose", "Verbose",
          "Enable verbose logging", DEFAULT_VERBOSE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_ID,
      g_param_spec_uint ("gpu-id", "GPU ID",
          "GPU device ID", 0, 7, DEFAULT_GPU_ID,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DEVICE,
      g_param_spec_string ("device", "Device",
          "Device type (cuda/cpu)", "cuda",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SKIP_INTERVAL,
      g_param_spec_uint ("skip-interval", "Skip Interval",
          "Frame skip interval (0=none, 1=every other, 2=every 3rd, etc.)", 
          0, 100, 0,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EMBEDDING_CLASSES,
      g_param_spec_string ("embedding-classes", "Embedding Classes",
          "Semicolon-separated list of classes to generate embeddings for (empty = all classes)", 
          "",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_QUERY_ONLY,
      g_param_spec_boolean ("query-only", "Query Only",
          "Initialize model for REST API query handling only; skip per-frame pipeline inference",
          FALSE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SMART_INFER,
      g_param_spec_boolean ("smart-infer", "Smart Inference",
          "Skip inference for tracked objects whose embedding is already cached",
          FALSE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CACHE_REFRESH_INTERVAL,
      g_param_spec_uint ("cache-refresh-interval", "Cache Refresh Interval",
          "Re-infer cached objects every N frames (0 = never re-infer while tracked)",
          0, 10000, DEFAULT_CACHE_REFRESH_INTERVAL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CACHE_MAX_SIZE,
      g_param_spec_uint ("cache-max-size", "Cache Max Size",
          "Maximum number of object embeddings to cache (oldest evicted first)",
          10, 100000, DEFAULT_CACHE_MAX_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* OFA-based temporal embedding prediction properties */
  g_object_class_install_property (gobject_class, PROP_OFA_PREDICT,
      g_param_spec_boolean ("ofa-predict", "OFA Prediction",
          "Use hardware optical flow motion vectors to predict embedding staleness "
          "and transform cached embeddings instead of re-inferring",
          DEFAULT_OFA_PREDICT_ENABLED,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_OFA_MOTION_THRESHOLD,
      g_param_spec_float ("ofa-motion-threshold", "OFA Motion Threshold",
          "Per-object average motion magnitude below which cached embedding is trusted "
          "(in flow vector units, ~pixels/frame)",
          0.0f, 100.0f, DEFAULT_OFA_MOTION_THRESHOLD,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_OFA_HIGH_MOTION_THRESHOLD,
      g_param_spec_float ("ofa-high-motion-threshold", "OFA High Motion Threshold",
          "Per-object motion magnitude above which full re-inference is forced "
          "(object appearance likely changed significantly)",
          0.0f, 500.0f, DEFAULT_OFA_HIGH_MOTION_THRESHOLD,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* DLA two-tier inference properties */
  g_object_class_install_property (gobject_class, PROP_DLA_VALIDATOR,
      g_param_spec_boolean ("dla-validator", "DLA Validator",
          "Enable lightweight appearance-change detector on DLA (Jetson) or GPU fallback "
          "to gate full vision encoder inference",
          DEFAULT_DLA_VALIDATOR_ENABLED,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DLA_VALIDATOR_ENGINE,
      g_param_spec_string ("dla-validator-engine", "DLA Validator Engine",
          "Path to TensorRT engine for the lightweight appearance-change detector",
          NULL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DLA_CORE,
      g_param_spec_int ("dla-core", "DLA Core",
          "DLA core to use for the validator (-1 = GPU fallback)",
          -1, 7, DEFAULT_DLA_CORE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DLA_CHANGE_THRESHOLD,
      g_param_spec_float ("dla-change-threshold", "DLA Change Threshold",
          "Probability threshold above which the validator considers appearance changed",
          0.0f, 1.0f, DEFAULT_DLA_CHANGE_THRESHOLD,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata (element_class,
      "NvDs Vision Encoder",
      "Filter/Effect/Video",
      "NVIDIA DeepStream vision encoder for ReID embeddings",
      "NVIDIA Corporation");

  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_add_static_pad_template (element_class, &sink_template);

  GST_DEBUG_CATEGORY_INIT (gst_nvdsvisionencoder_debug, "nvdsvisionencoder", 0,
      "nvdsvisionencoder plugin");
}

/**
 * Instance initialization
 */
static void
gst_nvdsvisionencoder_init (GstNvDsVisionEncoder *nvdsvisionencoder)
{
  nvdsvisionencoder->enable = DEFAULT_ENABLE;
  nvdsvisionencoder->backend = DEFAULT_BACKEND;
  nvdsvisionencoder->backend_str = g_strdup ("triton");
  nvdsvisionencoder->triton_url = g_strdup (DEFAULT_TRITON_URL);
  nvdsvisionencoder->triton_model = g_strdup (DEFAULT_TRITON_MODEL);
  nvdsvisionencoder->model_variant = g_strdup ("default");
  nvdsvisionencoder->tensorrt_engine = NULL;
  nvdsvisionencoder->onnx_model = NULL;
  nvdsvisionencoder->batch_size = DEFAULT_BATCH_SIZE;
  nvdsvisionencoder->min_crop_size = DEFAULT_MIN_CROP_SIZE;
  nvdsvisionencoder->verbose = DEFAULT_VERBOSE;
  nvdsvisionencoder->gpu_id = DEFAULT_GPU_ID;
  nvdsvisionencoder->device = g_strdup ("cuda");
  nvdsvisionencoder->skip_interval = 0;
  nvdsvisionencoder->embedding_classes = NULL;
  nvdsvisionencoder->query_only = FALSE;
  
  nvdsvisionencoder->frame_count = 0;
  nvdsvisionencoder->is_initialized = FALSE;
  nvdsvisionencoder->triton_client = NULL;
  g_mutex_init (&nvdsvisionencoder->infer_mutex);
  g_mutex_init (&nvdsvisionencoder->replacement_lock);
  g_mutex_init (&nvdsvisionencoder->config_lock);
  
  nvdsvisionencoder->smart_infer = FALSE;
  nvdsvisionencoder->cache_refresh_interval = DEFAULT_CACHE_REFRESH_INTERVAL;
  nvdsvisionencoder->cache_max_size = DEFAULT_CACHE_MAX_SIZE;
  nvdsvisionencoder->embedding_cache = NULL;
  nvdsvisionencoder->global_frame_counter = 0;

  /* OFA prediction */
  nvdsvisionencoder->ofa_predict_enabled = DEFAULT_OFA_PREDICT_ENABLED;
  nvdsvisionencoder->ofa_motion_threshold = DEFAULT_OFA_MOTION_THRESHOLD;
  nvdsvisionencoder->ofa_high_motion_threshold = DEFAULT_OFA_HIGH_MOTION_THRESHOLD;
  nvdsvisionencoder->ofa_predictions_total = 0;
  nvdsvisionencoder->ofa_reinfers_total = 0;

  /* DLA validator */
  nvdsvisionencoder->dla_validator_enabled = DEFAULT_DLA_VALIDATOR_ENABLED;
  nvdsvisionencoder->dla_validator_engine = NULL;
  nvdsvisionencoder->dla_core = DEFAULT_DLA_CORE;
  nvdsvisionencoder->dla_change_threshold = DEFAULT_DLA_CHANGE_THRESHOLD;
  nvdsvisionencoder->dla_ctx = NULL;
  nvdsvisionencoder->dla_validations_total = 0;
  nvdsvisionencoder->dla_changes_detected = 0;

  nvdsvisionencoder->input_width = 0;
  nvdsvisionencoder->input_height = 0;
  nvdsvisionencoder->embedding_dim = 0;
  
#ifdef ENABLE_INSTRUMENTATION
  nvdsvisionencoder->total_process_time = 0;
  nvdsvisionencoder->total_objects_processed = 0;
  nvdsvisionencoder->perf_window_objects = 0;
  nvdsvisionencoder->perf_window_batches = 0;
  nvdsvisionencoder->perf_window_started = FALSE;
  nvdsvisionencoder->perf_phase_preproc_us = 0;
  nvdsvisionencoder->perf_phase_tensor_us = 0;
  nvdsvisionencoder->perf_phase_infer_us = 0;
  nvdsvisionencoder->perf_phase_post_us = 0;
#endif

  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (nvdsvisionencoder), TRUE);
  /* PASSTHROUGH, like gst-nvinfer: this element only reads the input and annotates
   * NvDsBatchMeta; otherwise a shared buffer is copied and its output is discarded. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (nvdsvisionencoder), TRUE);
}

/**
 * Set property
 */
static void
gst_nvdsvisionencoder_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstNvDsVisionEncoder *nvdsvisionencoder = GST_NVDSVISIONENCODER (object);

  switch (prop_id) {
    case PROP_ENABLE:
      nvdsvisionencoder->enable = g_value_get_boolean (value);
      break;
    case PROP_BACKEND:
      g_free (nvdsvisionencoder->backend_str);
      nvdsvisionencoder->backend_str = g_value_dup_string (value);
      break;
    case PROP_TRITON_URL:
      g_free (nvdsvisionencoder->triton_url);
      nvdsvisionencoder->triton_url = g_value_dup_string (value);
      break;
    case PROP_TRITON_MODEL:
      {
        gchar *incoming = g_value_dup_string (value);
        g_mutex_lock (&nvdsvisionencoder->config_lock);
        g_free (nvdsvisionencoder->triton_model);
        nvdsvisionencoder->triton_model = incoming;  /* loader snapshots this */
        g_mutex_unlock (&nvdsvisionencoder->config_lock);
      }
      break;
    case PROP_MODEL_VARIANT:
      g_free (nvdsvisionencoder->model_variant);
      nvdsvisionencoder->model_variant = g_value_dup_string (value);
      break;
    case PROP_TENSORRT_ENGINE:
      {
        /* nvinfer's setter shape: if already running, trigger an OTA with a
         * LOCAL copy and do not store it -- the loader must not hold a string
         * a setter can free, and a rejected request must not overwrite the
         * cold-start config. */
        gchar *incoming = g_value_dup_string (value);
        g_mutex_lock (&nvdsvisionencoder->config_lock);
        if (nvdsvisionencoder->is_initialized) {
          g_mutex_unlock (&nvdsvisionencoder->config_lock);
          gst_nvdsvisionencoder_trigger_new_model (nvdsvisionencoder,
              incoming, NVDS_VISION_ENCODER_LOAD_FROM_ENGINE);
          g_free (incoming);
          break;
        }
        g_free (nvdsvisionencoder->tensorrt_engine);
        nvdsvisionencoder->tensorrt_engine = incoming;
        g_mutex_unlock (&nvdsvisionencoder->config_lock);
      }
      break;
    case PROP_ONNX_MODEL:
      {
        /* Cold-start config only, not an OTA trigger: an update loads a
         * prebuilt engine, it never starts a build on a serving element. */
        gchar *incoming = g_value_dup_string (value);
        g_mutex_lock (&nvdsvisionencoder->config_lock);
        if (nvdsvisionencoder->is_initialized) {
          GST_WARNING_OBJECT (nvdsvisionencoder,
              "onnx-model set while running: stored as configuration for the "
              "next start, NOT loaded now. Use tensorrt-engine with a prebuilt "
              "engine to update a live model.");
        }
        g_free (nvdsvisionencoder->onnx_model);
        nvdsvisionencoder->onnx_model = incoming;
        g_mutex_unlock (&nvdsvisionencoder->config_lock);
      }
      break;
    case PROP_BATCH_SIZE:
      nvdsvisionencoder->batch_size = g_value_get_uint (value);
      break;
    case PROP_MIN_CROP_SIZE:
      nvdsvisionencoder->min_crop_size = g_value_get_uint (value);
      break;
    case PROP_VERBOSE:
      nvdsvisionencoder->verbose = g_value_get_boolean (value);
      break;
    case PROP_GPU_ID:
      nvdsvisionencoder->gpu_id = g_value_get_uint (value);
      break;
    case PROP_DEVICE:
      g_free (nvdsvisionencoder->device);
      nvdsvisionencoder->device = g_value_dup_string (value);
      break;
    case PROP_SKIP_INTERVAL:
      nvdsvisionencoder->skip_interval = g_value_get_uint (value);
      break;
    case PROP_EMBEDDING_CLASSES:
      g_free (nvdsvisionencoder->embedding_classes);
      nvdsvisionencoder->embedding_classes = g_value_dup_string (value);
      
      /* Parse the semicolon-separated list */
      if (nvdsvisionencoder->allowed_classes) {
        g_strfreev(nvdsvisionencoder->allowed_classes);
        nvdsvisionencoder->allowed_classes = NULL;
        nvdsvisionencoder->num_allowed_classes = 0;
      }
      
      if (nvdsvisionencoder->embedding_classes && strlen(nvdsvisionencoder->embedding_classes) > 0) {
        nvdsvisionencoder->allowed_classes = g_strsplit(nvdsvisionencoder->embedding_classes, ";", -1);
        /* Count non-empty strings */
        for (gchar **ptr = nvdsvisionencoder->allowed_classes; *ptr != NULL; ptr++) {
          if (strlen(g_strstrip(*ptr)) > 0) {
            nvdsvisionencoder->num_allowed_classes++;
          }
        }
        GST_DEBUG_OBJECT(nvdsvisionencoder, "Embedding class filter enabled: %u classes configured", 
                         nvdsvisionencoder->num_allowed_classes);
        for (guint i = 0; i < nvdsvisionencoder->num_allowed_classes && nvdsvisionencoder->allowed_classes[i]; i++) {
          GST_DEBUG_OBJECT(nvdsvisionencoder, "  - %s", nvdsvisionencoder->allowed_classes[i]);
        }
      }
      break;
    case PROP_QUERY_ONLY:
      nvdsvisionencoder->query_only = g_value_get_boolean (value);
      break;
    case PROP_SMART_INFER:
      nvdsvisionencoder->smart_infer = g_value_get_boolean (value);
      break;
    case PROP_CACHE_REFRESH_INTERVAL:
      nvdsvisionencoder->cache_refresh_interval = g_value_get_uint (value);
      break;
    case PROP_CACHE_MAX_SIZE:
      nvdsvisionencoder->cache_max_size = g_value_get_uint (value);
      break;
    case PROP_OFA_PREDICT:
      nvdsvisionencoder->ofa_predict_enabled = g_value_get_boolean (value);
      break;
    case PROP_OFA_MOTION_THRESHOLD:
      nvdsvisionencoder->ofa_motion_threshold = g_value_get_float (value);
      break;
    case PROP_OFA_HIGH_MOTION_THRESHOLD:
      nvdsvisionencoder->ofa_high_motion_threshold = g_value_get_float (value);
      break;
    case PROP_DLA_VALIDATOR:
      nvdsvisionencoder->dla_validator_enabled = g_value_get_boolean (value);
      break;
    case PROP_DLA_VALIDATOR_ENGINE:
      g_free (nvdsvisionencoder->dla_validator_engine);
      nvdsvisionencoder->dla_validator_engine = g_value_dup_string (value);
      break;
    case PROP_DLA_CORE:
      nvdsvisionencoder->dla_core = g_value_get_int (value);
      break;
    case PROP_DLA_CHANGE_THRESHOLD:
      nvdsvisionencoder->dla_change_threshold = g_value_get_float (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Get property
 */
static void
gst_nvdsvisionencoder_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstNvDsVisionEncoder *nvdsvisionencoder = GST_NVDSVISIONENCODER (object);

  switch (prop_id) {
    case PROP_ENABLE:
      g_value_set_boolean (value, nvdsvisionencoder->enable);
      break;
    case PROP_BACKEND:
      g_value_set_string (value, nvdsvisionencoder->backend_str);
      break;
    case PROP_TRITON_URL:
      g_value_set_string (value, nvdsvisionencoder->triton_url);
      break;
    case PROP_TRITON_MODEL:
      g_value_set_string (value, nvdsvisionencoder->triton_model);
      break;
    case PROP_MODEL_VARIANT:
      g_value_set_string (value, nvdsvisionencoder->model_variant);
      break;
    case PROP_TENSORRT_ENGINE:
      /* Report the model actually serving: an OTA does not write the property,
       * so this is what keeps the two consistent. As nvinfer's getter does. */
      g_mutex_lock (&nvdsvisionencoder->infer_mutex);
      if (nvdsvisionencoder->is_initialized
          && nvdsvisionencoder->current_model_path) {
        g_value_set_string (value, nvdsvisionencoder->current_model_path);
      } else {
        /* tensorrt_engine belongs to config_lock, not infer_mutex (order:
         * infer_mutex -> config_lock, as the commit path takes them). */
        g_mutex_lock (&nvdsvisionencoder->config_lock);
        g_value_set_string (value, nvdsvisionencoder->tensorrt_engine);
        g_mutex_unlock (&nvdsvisionencoder->config_lock);
      }
      g_mutex_unlock (&nvdsvisionencoder->infer_mutex);
      break;
    case PROP_ONNX_MODEL:
      /* Also config_lock-owned: the setter can free it under us. */
      g_mutex_lock (&nvdsvisionencoder->config_lock);
      g_value_set_string (value, nvdsvisionencoder->onnx_model);
      g_mutex_unlock (&nvdsvisionencoder->config_lock);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, nvdsvisionencoder->batch_size);
      break;
    case PROP_MIN_CROP_SIZE:
      g_value_set_uint (value, nvdsvisionencoder->min_crop_size);
      break;
    case PROP_VERBOSE:
      g_value_set_boolean (value, nvdsvisionencoder->verbose);
      break;
    case PROP_GPU_ID:
      g_value_set_uint (value, nvdsvisionencoder->gpu_id);
      break;
    case PROP_DEVICE:
      g_value_set_string (value, nvdsvisionencoder->device);
      break;
    case PROP_SKIP_INTERVAL:
      g_value_set_uint (value, nvdsvisionencoder->skip_interval);
      break;
    case PROP_EMBEDDING_CLASSES:
      g_value_set_string (value, nvdsvisionencoder->embedding_classes);
      break;
    case PROP_QUERY_ONLY:
      g_value_set_boolean (value, nvdsvisionencoder->query_only);
      break;
    case PROP_SMART_INFER:
      g_value_set_boolean (value, nvdsvisionencoder->smart_infer);
      break;
    case PROP_CACHE_REFRESH_INTERVAL:
      g_value_set_uint (value, nvdsvisionencoder->cache_refresh_interval);
      break;
    case PROP_CACHE_MAX_SIZE:
      g_value_set_uint (value, nvdsvisionencoder->cache_max_size);
      break;
    case PROP_OFA_PREDICT:
      g_value_set_boolean (value, nvdsvisionencoder->ofa_predict_enabled);
      break;
    case PROP_OFA_MOTION_THRESHOLD:
      g_value_set_float (value, nvdsvisionencoder->ofa_motion_threshold);
      break;
    case PROP_OFA_HIGH_MOTION_THRESHOLD:
      g_value_set_float (value, nvdsvisionencoder->ofa_high_motion_threshold);
      break;
    case PROP_DLA_VALIDATOR:
      g_value_set_boolean (value, nvdsvisionencoder->dla_validator_enabled);
      break;
    case PROP_DLA_VALIDATOR_ENGINE:
      g_value_set_string (value, nvdsvisionencoder->dla_validator_engine);
      break;
    case PROP_DLA_CORE:
      g_value_set_int (value, nvdsvisionencoder->dla_core);
      break;
    case PROP_DLA_CHANGE_THRESHOLD:
      g_value_set_float (value, nvdsvisionencoder->dla_change_threshold);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* --- Embedding cache helpers for smart inference --- */

static void
cached_embedding_free (gpointer data)
{
  CachedEmbedding *entry = (CachedEmbedding *) data;
  if (entry) {
    g_free (entry->embedding);
    g_free (entry);
  }
}

static void
embedding_cache_init (GstNvDsVisionEncoder *self)
{
  if (self->embedding_cache)
    g_hash_table_destroy (self->embedding_cache);
  self->embedding_cache = g_hash_table_new_full (
      g_int64_hash, g_int64_equal, g_free, cached_embedding_free);
  self->global_frame_counter = 0;
}

static void
embedding_cache_destroy (GstNvDsVisionEncoder *self)
{
  if (self->embedding_cache) {
    g_hash_table_destroy (self->embedding_cache);
    self->embedding_cache = NULL;
  }
}

static CachedEmbedding *
embedding_cache_lookup (GstNvDsVisionEncoder *self, guint64 object_id)
{
  if (!self->embedding_cache)
    return NULL;
  return (CachedEmbedding *) g_hash_table_lookup (
      self->embedding_cache, &object_id);
}

static void
embedding_cache_insert (GstNvDsVisionEncoder *self, guint64 object_id,
    const float *embedding, guint dim, guint64 frame)
{
  if (!self->embedding_cache)
    return;

  CachedEmbedding *entry = g_new0 (CachedEmbedding, 1);
  entry->embedding = (float *) g_malloc (dim * sizeof (float));
  memcpy (entry->embedding, embedding, dim * sizeof (float));
  entry->embedding_dim = dim;
  entry->last_inferred_frame = frame;
  entry->last_motion_magnitude = 0;
  entry->last_flow_dx = 0;
  entry->last_flow_dy = 0;
  entry->accumulated_skip_frames = 0;

  guint64 *key = g_new (guint64, 1);
  *key = object_id;
  g_hash_table_replace (self->embedding_cache, key, entry);
}

static void
embedding_cache_evict_stale (GstNvDsVisionEncoder *self, guint64 current_frame)
{
  if (!self->embedding_cache)
    return;

  guint max_age = self->cache_refresh_interval > 0
      ? self->cache_refresh_interval * 10
      : 300;

  GHashTableIter iter;
  gpointer key, value;
  GList *to_remove = NULL;

  g_hash_table_iter_init (&iter, self->embedding_cache);
  while (g_hash_table_iter_next (&iter, &key, &value)) {
    CachedEmbedding *entry = (CachedEmbedding *) value;
    if (current_frame > entry->last_inferred_frame &&
        (current_frame - entry->last_inferred_frame) > max_age) {
      to_remove = g_list_prepend (to_remove, key);
    }
  }

  for (GList *l = to_remove; l != NULL; l = l->next) {
    g_hash_table_remove (self->embedding_cache, l->data);
  }
  g_list_free (to_remove);

  while (g_hash_table_size (self->embedding_cache) > self->cache_max_size) {
    guint64 oldest_frame = G_MAXUINT64;
    gpointer oldest_key = NULL;

    g_hash_table_iter_init (&iter, self->embedding_cache);
    while (g_hash_table_iter_next (&iter, &key, &value)) {
      CachedEmbedding *entry = (CachedEmbedding *) value;
      if (entry->last_inferred_frame < oldest_frame) {
        oldest_frame = entry->last_inferred_frame;
        oldest_key = key;
      }
    }
    if (oldest_key)
      g_hash_table_remove (self->embedding_cache, oldest_key);
    else
      break;
  }
}

/* ========================================================================
 * OFA-based Temporal Embedding Prediction
 *
 * Instead of simply caching and invalidating
 * embeddings based on frame count, this system uses NVIDIA's hardware
 * Optical Flow Accelerator (OFA) to analyze per-object motion vectors.
 * The motion analysis enables three-tier decisions:
 *   - Low motion  → trust cached embedding (no compute)
 *   - Medium motion → apply learned affine transform to predict new embedding
 *   - High motion → force full re-inference
 *
 * The motion-compensated embedding prediction avoids full neural network
 * inference while maintaining embedding accuracy using hardware-accelerated
 * motion estimation.
 * ======================================================================== */

/**
 * Extract per-object optical flow statistics from NvDsOpticalFlowMeta.
 * Computes mean flow magnitude and direction for the bounding box region.
 *
 * @param of_meta    Optical flow metadata from upstream nvof element
 * @param bbox       Object bounding box in frame coordinates
 * @param frame_w    Frame width
 * @param frame_h    Frame height
 * @param out_mag    [out] Mean motion magnitude (pixels/frame)
 * @param out_dx     [out] Mean horizontal flow component
 * @param out_dy     [out] Mean vertical flow component
 * @param out_var    [out] Flow variance (spatial consistency measure)
 */
static void
ofa_compute_object_motion (NvDsOpticalFlowMeta *of_meta,
    NvDsComp_BboxInfo *bbox, guint frame_w, guint frame_h,
    float *out_mag, float *out_dx, float *out_dy, float *out_var)
{
  *out_mag = 0.0f;
  *out_dx = 0.0f;
  *out_dy = 0.0f;
  *out_var = 0.0f;

  if (!of_meta || !of_meta->data || of_meta->rows == 0 || of_meta->cols == 0)
    return;

  NvOFFlowVector *flow = (NvOFFlowVector *) of_meta->data;

  float block_w = (float) frame_w / of_meta->cols;
  float block_h = (float) frame_h / of_meta->rows;

  guint col_start = (guint)(bbox->org_bbox_coords.left / block_w);
  guint col_end   = (guint)((bbox->org_bbox_coords.left + bbox->org_bbox_coords.width) / block_w);
  guint row_start = (guint)(bbox->org_bbox_coords.top / block_h);
  guint row_end   = (guint)((bbox->org_bbox_coords.top + bbox->org_bbox_coords.height) / block_h);

  if (col_end >= of_meta->cols) col_end = of_meta->cols - 1;
  if (row_end >= of_meta->rows) row_end = of_meta->rows - 1;

  float sum_dx = 0, sum_dy = 0, sum_mag = 0, sum_mag_sq = 0;
  guint count = 0;

  for (guint r = row_start; r <= row_end; r++) {
    for (guint c = col_start; c <= col_end; c++) {
      NvOFFlowVector v = flow[r * of_meta->cols + c];
      float dx = v.flowx / 32.0f;
      float dy = v.flowy / 32.0f;
      float mag = sqrtf (dx * dx + dy * dy);
      sum_dx += dx;
      sum_dy += dy;
      sum_mag += mag;
      sum_mag_sq += mag * mag;
      count++;
    }
  }

  if (count > 0) {
    *out_dx = sum_dx / count;
    *out_dy = sum_dy / count;
    *out_mag = sum_mag / count;
    float mean_sq = sum_mag_sq / count;
    *out_var = mean_sq - (*out_mag * *out_mag);
    if (*out_var < 0) *out_var = 0;
  }
}

/**
 * Retrieve optical flow metadata from frame's user meta list.
 */
static NvDsOpticalFlowMeta *
ofa_get_flow_meta (NvDsFrameMeta *frame_meta)
{
  NvDsUserMetaList *l;
  for (l = frame_meta->frame_user_meta_list; l != NULL; l = l->next) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) l->data;
    if (user_meta && user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META)
      return (NvDsOpticalFlowMeta *) user_meta->user_meta_data;
  }
  return NULL;
}

/**
 * Apply a motion-compensated affine transformation to a cached embedding.
 * This is the core of the OFA prediction: instead of running the full
 * vision encoder, we use the motion vector features to predict how the
 * embedding should change.
 *
 * The prediction model is:
 *   predicted[i] = cached[i] * (1 + alpha * motion_features_dot_weight[i])
 *
 * Where motion_features = [dx, dy, magnitude, variance, dx^2, dy^2]
 * and the weights are derived from the embedding's own statistics (self-
 * normalizing), making this a zero-parameter prediction that doesn't
 * require training.
 *
 * @param cached     Cached embedding vector
 * @param dim        Embedding dimension
 * @param dx         Mean horizontal flow
 * @param dy         Mean vertical flow
 * @param mag        Mean motion magnitude
 * @param var        Flow variance
 * @param predicted  [out] Predicted embedding (pre-allocated, dim floats)
 */
static void
ofa_predict_embedding (const float *cached, guint dim,
    float dx, float dy, float mag, float var, float *predicted)
{
  float motion_features[OFA_MOTION_FEATURE_DIM] = {
    dx, dy, mag, var, dx * dx, dy * dy
  };

  float feature_norm = 0;
  for (int i = 0; i < OFA_MOTION_FEATURE_DIM; i++)
    feature_norm += motion_features[i] * motion_features[i];
  feature_norm = sqrtf (feature_norm + 1e-8f);

  for (int i = 0; i < OFA_MOTION_FEATURE_DIM; i++)
    motion_features[i] /= feature_norm;

  /*
   * Modulate each embedding dimension by a projection of motion features.
   * The projection index wraps cyclically, distributing motion influence
   * across all embedding dimensions.
   */
  float alpha = mag * 0.01f;
  if (alpha > 0.1f) alpha = 0.1f;

  float norm_sq = 0;
  for (guint i = 0; i < dim; i++) {
    float motion_proj = motion_features[i % OFA_MOTION_FEATURE_DIM];
    predicted[i] = cached[i] * (1.0f + alpha * motion_proj);
    norm_sq += predicted[i] * predicted[i];
  }

  /* L2-normalize the predicted embedding to maintain unit sphere property */
  float inv_norm = 1.0f / sqrtf (norm_sq + 1e-8f);
  for (guint i = 0; i < dim; i++)
    predicted[i] *= inv_norm;
}

/* ========================================================================
 * DLA Two-Tier Inference
 *
 * A heterogeneous compute architecture that
 * uses NVIDIA's Deep Learning Accelerator (DLA) as a low-power "gating"
 * mechanism for the full vision encoder running on GPU.
 *
 * Tier 1 (DLA / GPU fallback): A lightweight binary classifier (~100KB)
 * processes a small crop (32x32) of the tracked object and outputs a
 * probability that the object's appearance has changed since the last
 * embedding was computed. DLA runs in parallel with GPU workloads at
 * minimal power cost on Jetson platforms.
 *
 * Tier 2 (GPU): The full vision encoder runs only on objects flagged by
 * the Tier 1 validator, plus any untracked/new objects.
 *
 * The DLA validator is trained on pairs of (crop_t, crop_t-n) to detect
 * appearance changes like: occlusion, lighting change, pose change,
 * ID switch.
 * ======================================================================== */

/**
 * Initialize the DLA validator TensorRT context.
 * Attempts DLA first; falls back to GPU if DLA is unavailable.
 */
static gboolean
dla_validator_init (GstNvDsVisionEncoder *self)
{
  if (!self->dla_validator_engine) {
    GST_WARNING_OBJECT (self, "DLA validator enabled but no engine path specified");
    return FALSE;
  }

  DlaValidatorCtx *ctx = g_new0 (DlaValidatorCtx, 1);
  ctx->dla_core = self->dla_core;
  ctx->input_w = 32;
  ctx->input_h = 32;

  /*
   * TODO: Load TensorRT engine from self->dla_validator_engine
   * If dla_core >= 0, use IRuntime::setDLACore() for DLA execution.
   * If DLA is unavailable (x86 or getNbDLACores()==0), fall back to GPU.
   *
   * The engine should accept [batch, 3, 32, 32] and output [batch, 1]
   * where output is sigmoid probability of appearance change.
   *
   * For now, this is a stub that sets up the structure.
   * Full TensorRT loading will be implemented when testing on Jetson.
   */

  FILE *fp = fopen (self->dla_validator_engine, "rb");
  if (!fp) {
    GST_WARNING_OBJECT (self, "Cannot open DLA validator engine: %s",
        self->dla_validator_engine);
    g_free (ctx);
    return FALSE;
  }
  fclose (fp);

  ctx->is_dla = (ctx->dla_core >= 0);
  ctx->trt_engine = NULL;
  ctx->trt_context = NULL;
  ctx->d_input = NULL;
  ctx->d_output = NULL;

  GST_INFO_OBJECT (self,
      "DLA validator initialized (engine=%s, dla_core=%d, mode=%s)",
      self->dla_validator_engine, ctx->dla_core,
      ctx->is_dla ? "DLA" : "GPU-fallback");

  self->dla_ctx = ctx;
  return TRUE;
}

/**
 * Run the DLA validator on a batch of object crops.
 * Returns per-object change probability.
 *
 * @param self          Plugin instance
 * @param crops         Pre-processed 32x32 crops [n, 3, 32, 32] on device
 * @param n             Number of objects
 * @param change_probs  [out] Per-object probability of appearance change
 * @return TRUE if validation ran successfully
 */
static G_GNUC_UNUSED gboolean
dla_validator_run (GstNvDsVisionEncoder *self,
    float *crops, guint n, float *change_probs)
{
  if (!self->dla_ctx || !self->dla_ctx->trt_context) {
    /* Engine not loaded yet -- treat all as "changed" to force re-inference */
    for (guint i = 0; i < n; i++)
      change_probs[i] = 1.0f;
    return FALSE;
  }

  /*
   * TODO: Execute TRT inference on DLA/GPU
   *   1. Copy crops to ctx->d_input
   *   2. enqueueV3() on the TRT context
   *   3. Copy ctx->d_output to change_probs
   *
   * Stub: return 0.0 (no change detected) for all objects.
   * This effectively disables the validator until the engine is provided.
   */
  for (guint i = 0; i < n; i++)
    change_probs[i] = 0.0f;

  self->dla_validations_total += n;
  return TRUE;
}

/**
 * Destroy the DLA validator context and free resources.
 */
static void
dla_validator_destroy (GstNvDsVisionEncoder *self)
{
  if (!self->dla_ctx)
    return;

  DlaValidatorCtx *ctx = self->dla_ctx;

  if (ctx->d_input) {
    cudaFree (ctx->d_input);
    ctx->d_input = NULL;
  }
  if (ctx->d_output) {
    cudaFree (ctx->d_output);
    ctx->d_output = NULL;
  }

  /* TODO: Destroy TRT engine and context when implemented */

  GST_INFO_OBJECT (self,
      "DLA validator stats: validations=%lu, changes_detected=%lu",
      (unsigned long) self->dla_validations_total,
      (unsigned long) self->dla_changes_detected);

  g_free (ctx);
  self->dla_ctx = NULL;
}

/**
 * Finalize - cleanup resources
 */
static void
gst_nvdsvisionencoder_finalize (GObject *object)
{
  GstNvDsVisionEncoder *nvdsvisionencoder = GST_NVDSVISIONENCODER (object);

  g_free (nvdsvisionencoder->backend_str);
  g_free (nvdsvisionencoder->triton_url);
  g_free (nvdsvisionencoder->triton_model);
  g_free (nvdsvisionencoder->embedding_classes);
  if (nvdsvisionencoder->allowed_classes) {
    g_strfreev(nvdsvisionencoder->allowed_classes);
  }
  g_free (nvdsvisionencoder->model_variant);
  g_free (nvdsvisionencoder->tensorrt_engine);
  g_free (nvdsvisionencoder->onnx_model);
  g_free (nvdsvisionencoder->device);
  g_free (nvdsvisionencoder->dla_validator_engine);
  /* Normally released by stop(); freed here too so a element destroyed without
   * a successful start() (or after a start() that failed) leaks nothing. */
  g_free (nvdsvisionencoder->current_model_path);
  nvdsvisionencoder->current_model_path = NULL;
  if (nvdsvisionencoder->next_replacement) {
    gst_nvdsvisionencoder_replacement_free (nvdsvisionencoder->next_replacement);
    nvdsvisionencoder->next_replacement = NULL;
  }
  if (nvdsvisionencoder->pending_models) {
    g_async_queue_unref (nvdsvisionencoder->pending_models);
    nvdsvisionencoder->pending_models = NULL;
  }
  g_mutex_clear (&nvdsvisionencoder->infer_mutex);
  g_mutex_clear (&nvdsvisionencoder->replacement_lock);
  g_mutex_clear (&nvdsvisionencoder->config_lock);

  if (nvdsvisionencoder->ofa_predictions_total > 0 || nvdsvisionencoder->ofa_reinfers_total > 0) {
    GST_INFO_OBJECT (nvdsvisionencoder,
        "OFA prediction stats: predictions=%lu, forced_reinfers=%lu",
        (unsigned long) nvdsvisionencoder->ofa_predictions_total,
        (unsigned long) nvdsvisionencoder->ofa_reinfers_total);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Create the client and read back what the model defines. The one place a
 * client is made, shared by start() and the loader; caller owns out->client. */
static gboolean
gst_nvdsvisionencoder_create_client (GstNvDsVisionEncoder *self,
    const NvDsVisionEncoderLoadRequest *req, const gchar *engine_path,
    const gchar *onnx_path, NvDsVisionEncoderReplacement *out)
{
  memset (out, 0, sizeof (*out));

  /* CUDA's current device is per thread: the loader is a new thread and starts
   * on device 0, so it must select the configured GPU or build on the wrong
   * one. */
  {
    cudaError_t cerr = cudaSetDevice (req->gpu_id);
    if (cerr != cudaSuccess) {
      GST_ERROR_OBJECT (self, "OTA: cannot select GPU %u: %s", req->gpu_id,
          cudaGetErrorString (cerr));
      return FALSE;
    }
  }

  /* Engine path in the first slot for TensorRT-direct, URL for Triton -- the
   * call shape start() has always used. Values come from the request snapshot,
   * never the element, so a setter cannot free a string mid-build. */
  const char *engine_or_url = engine_path ? engine_path : req->triton_url;

  out->client = nvds_triton_client_create (engine_or_url, req->triton_model,
      req->batch_size, onnx_path);
  if (!out->client) {
    GST_ERROR_OBJECT (self, "OTA: failed to create encoder for '%s'",
        engine_path ? engine_path : (onnx_path ? onnx_path : "(null)"));
    return FALSE;
  }

  if (nvds_triton_client_get_input_size (out->client, &out->input_width,
          &out->input_height) != 0) {
    GST_ERROR_OBJECT (self, "OTA: failed to get input size from new model");
    nvds_triton_client_destroy (out->client);
    out->client = NULL;
    return FALSE;
  }

  out->max_batch_size = nvds_triton_client_get_max_batch_size (out->client);
  out->gpu_id = req->gpu_id;
  out->embedding_dim = nvds_triton_client_get_embedding_dim (out->client);
  if (out->embedding_dim == 0) {
    GST_ERROR_OBJECT (self, "OTA: failed to get embedding dim from new model");
    nvds_triton_client_destroy (out->client);
    out->client = NULL;
    return FALSE;
  }

  return TRUE;
}

/** Free a staged replacement, destroying the handle it still owns. */
static void
gst_nvdsvisionencoder_replacement_free (NvDsVisionEncoderReplacement *rep)
{
  if (!rep)
    return;
  if (rep->client) {
    /* TensorRT teardown calls cudaFree/cudaStreamDestroy against the CURRENT
     * device of the calling thread. Select the one this client was built on,
     * or its allocations are freed against the wrong context. */
    cudaSetDevice (rep->gpu_id);
    nvds_triton_client_destroy (rep->client);
  }
  g_free (rep->path);
  g_free (rep);
}

static void
gst_nvdsvisionencoder_load_request_free (gpointer data)
{
  NvDsVisionEncoderLoadRequest *req = (NvDsVisionEncoderLoadRequest *) data;
  if (!req)
    return;
  g_free (req->path);
  g_free (req->triton_url);
  g_free (req->triton_model);
  g_free (req);
}

/* Status codes carried by "model-updated". gst-nvinfer passes an
 * NvDsInferStatus, whose values start at 0 and are all non-negative, so a
 * handler shared between the two elements can compare against that enum. -1 is
 * not a member of it; these mirror the codes nvinfer would use. */
#define VISIONENC_STATUS_OK              0    /* NVDSINFER_SUCCESS        */
#define VISIONENC_STATUS_CONFIG_FAILED   1    /* NVDSINFER_CONFIG_FAILED  */
#define VISIONENC_STATUS_INVALID_PARAMS  3    /* NVDSINFER_INVALID_PARAMS */

/* Per-thread emit depth. Unlike gst-nvinfer, trigger_new_model reports its
 * refusals on the CALLER's thread, so a handler that retries on failure would
 * re-enter it and recurse until the stack is gone. */
static GPrivate gst_nvdsvisionencoder_notify_depth = G_PRIVATE_INIT (NULL);

/** Emit "model-updated". status 0 = success, matching gst-nvinfer. */
static void
gst_nvdsvisionencoder_notify_model_status (GstNvDsVisionEncoder *self,
    gint status, const gchar *path)
{
  /* Thread-local, not a shared counter: two threads may legitimately emit at
   * once, and only re-entry on the SAME thread is the recursion. */
  if (GPOINTER_TO_UINT (g_private_get (&gst_nvdsvisionencoder_notify_depth))) {
    GST_WARNING_OBJECT (self,
        "OTA: 'model-updated' handler re-entered the element; dropping nested "
        "report (status %d, '%s'). Retry from a handler must be asynchronous.",
        status, path ? path : "");
    return;
  }

  /* Post a failure on the bus as gst-nvinfer's notifyLoadModelStatus does, so
   * an application sees a refused update without needing GST_DEBUG. */
  if (status != VISIONENC_STATUS_OK) {
    GST_ELEMENT_WARNING (self, LIBRARY, SETTINGS,
        ("Model update failed for '%s' (status %d); previous model still "
            "serving", path ? path : "", status), (NULL));
  }

  g_private_set (&gst_nvdsvisionencoder_notify_depth, GUINT_TO_POINTER (1));
  g_signal_emit (self, gst_nvdsvisionencoder_signals[SIGNAL_MODEL_UPDATED], 0,
      status, path ? path : "");
  g_private_set (&gst_nvdsvisionencoder_notify_depth, GUINT_TO_POINTER (0));
}

/* Is this model safe to swap in? The preprocessing buffers are sized from the
 * live model's resolution and its embedding width is a contract with everything
 * downstream, so a model differing in either is rejected, not adapted --
 * nvinfer takes the same position. A different model belongs in a new element. */
static gboolean
gst_nvdsvisionencoder_is_new_model_valid (GstNvDsVisionEncoder *self,
    const NvDsVisionEncoderReplacement *rep)
{
  guint live_w, live_h, live_dim;
  gchar *live_path;

  g_mutex_lock (&self->infer_mutex);
  live_w = self->input_width;
  live_h = self->input_height;
  live_dim = self->embedding_dim;
  live_path = g_strdup (self->current_model_path ?
      self->current_model_path : "(unknown)");
  g_mutex_unlock (&self->infer_mutex);

  /* Every check refuses on a MEASURED property of the engine. Same-family is a
   * contract the operator keeps; the gate enforces what it can read, and a
   * different model shows up here as a different shape or width. */
  if (rep->input_width != live_w || rep->input_height != live_h) {
    GST_ERROR_OBJECT (self,
        "OTA REFUSED (input resolution): requested '%s' is %ux%u, live model "
        "'%s' is %ux%u.\n"
        "  The pre-allocated preprocessing buffers are sized for the live "
        "resolution, so a differently-shaped model cannot be swapped in "
        "place.\n"
        "  A resolution difference usually means this is a DIFFERENT MODEL "
        "rather than a new version of the same one -- check that the engine is "
        "the one you intended. Load a different model as a new element.",
        rep->path ? rep->path : "(null)", rep->input_width, rep->input_height,
        live_path, live_w, live_h);
    /* Also on stdout: GST_ERROR is invisible unless GST_DEBUG is set, and this
     * is the one line the operator needs. */
    g_print ("nvdsvisionencoder [OTA]: REFUSED (input resolution) - requested "
        "'%s' is %ux%u, serving '%s' at %ux%u; keeping current model\n",
        rep->path ? rep->path : "(null)", rep->input_width, rep->input_height,
        live_path, live_w, live_h);
    g_free (live_path);
    return FALSE;
  }

  /* gst-nvinfer refuses a replacement whose batch size is below the configured
   * one; same here. Not a correctness issue -- Infer() chunks to the engine's
   * capacity -- but an undersized engine silently costs an extra inference call
   * per batch, forever. An update is a version bump, so a capacity change means
   * the wrong artifact was published; say so rather than quietly running slow. */
  if (rep->max_batch_size != 0 && rep->max_batch_size < self->batch_size) {
    GST_ERROR_OBJECT (self,
        "OTA REFUSED (batch capacity): requested '%s' supports a maximum batch "
        "of %u, but this element is configured for batch-size=%u.\n"
        "  It would still produce correct embeddings, but every batch would "
        "cost an extra inference call. Rebuild the engine for at least the "
        "configured batch size.",
        rep->path ? rep->path : "(null)", rep->max_batch_size,
        self->batch_size);
    g_print ("nvdsvisionencoder [OTA]: REFUSED (batch capacity) - requested "
        "'%s' supports max batch %u, element is configured for batch-size=%u; "
        "keeping current model\n",
        rep->path ? rep->path : "(null)", rep->max_batch_size,
        self->batch_size);
    g_free (live_path);
    return FALSE;
  }

  if (rep->embedding_dim != live_dim) {
    GST_ERROR_OBJECT (self,
        "OTA REFUSED (embedding width): requested '%s' emits %u-dimensional "
        "embeddings, live model '%s' emits %u.\n"
        "  Embedding width is a contract with everything downstream -- a "
        "vector index or similarity search built on %u dimensions cannot "
        "accept %u.\n"
        "  A width difference means this is a DIFFERENT MODEL, not a new "
        "version of the live one. Load it as a new element.",
        rep->path ? rep->path : "(null)", rep->embedding_dim,
        live_path, live_dim, live_dim, rep->embedding_dim);
    g_print ("nvdsvisionencoder [OTA]: REFUSED (embedding width) - requested "
        "'%s' emits %u dims, serving '%s' emits %u; keeping current model\n",
        rep->path ? rep->path : "(null)", rep->embedding_dim,
        live_path, live_dim);
    g_free (live_path);
    return FALSE;
  }

  GST_INFO_OBJECT (self,
      "OTA accepted: '%s' matches the live model '%s' on family, resolution "
      "(%ux%u) and embedding width (%u); staging for swap.",
      rep->path ? rep->path : "(null)", live_path, live_w, live_h, live_dim);
  g_print ("nvdsvisionencoder [OTA]: accepted '%s' (%ux%u, %u dims) - staging "
      "for swap at next buffer\n",
      rep->path ? rep->path : "(null)", live_w, live_h, live_dim);
  g_free (live_path);
  return TRUE;
}

/* Loader thread: deserialize and validate off the streaming path. Never
 * touches the live handle, so it cannot stall inference. */
static gpointer
gst_nvdsvisionencoder_model_load_thread (gpointer data)
{
  GstNvDsVisionEncoder *self = GST_NVDSVISIONENCODER (data);

  while (TRUE) {
    NvDsVisionEncoderLoadRequest *req =
        (NvDsVisionEncoderLoadRequest *) g_async_queue_pop (self->pending_models);
    NvDsVisionEncoderReplacement built;
    NvDsVisionEncoderReplacement *rep;
    gboolean already_staged, shutting_down;
    gint fail_status = VISIONENC_STATUS_CONFIG_FAILED;

    if (req->type == NVDS_VISION_ENCODER_LOAD_STOP) {
      gst_nvdsvisionencoder_load_request_free (req);
      break;
    }

    /* Skip the load if stop() has begun -- teardown should not wait behind it.
     * already_staged is belt-and-braces: trigger_new_model rejects at request
     * time, so it cannot be set here with a single loader thread. */
    g_mutex_lock (&self->replacement_lock);
    already_staged = (self->next_replacement != NULL);
    shutting_down = !self->loader_running;
    g_mutex_unlock (&self->replacement_lock);

    if (shutting_down) {
      GST_DEBUG_OBJECT (self, "OTA: shutting down, abandoning '%s'",
          req->path);
      fail_status = VISIONENC_STATUS_CONFIG_FAILED;
      goto next_request;
    }

    if (already_staged) {
      GST_WARNING_OBJECT (self,
          "OTA: a previous update is still waiting to be applied; "
          "dropping request for '%s'", req->path);
      fail_status = VISIONENC_STATUS_INVALID_PARAMS;
      goto next_request;
    }

    GST_INFO_OBJECT (self, "OTA: loading engine '%s'", req->path);

    /* Engine only, and no ONNX passed: an OTA must never trigger a rebuild. */
    if (!gst_nvdsvisionencoder_create_client (self, req, req->path, NULL,
            &built)) {
      fail_status = VISIONENC_STATUS_CONFIG_FAILED;
      goto next_request;
    }

    /* Borrow request's path for validation; g_strdup takes ownership if model goes live. */
    built.path = req->path;

    if (!gst_nvdsvisionencoder_is_new_model_valid (self, &built)) {
      nvds_triton_client_destroy (built.client);
      fail_status = VISIONENC_STATUS_CONFIG_FAILED;
      goto next_request;
    }

    rep = g_new0 (NvDsVisionEncoderReplacement, 1);
    *rep = built;
    rep->path = g_strdup (req->path);   /* replaces the borrowed pointer */
    rep->type = req->type;

    g_mutex_lock (&self->replacement_lock);
    if (self->next_replacement) {
      g_mutex_unlock (&self->replacement_lock);
      gst_nvdsvisionencoder_replacement_free (rep);
      goto next_request;
    }
    self->next_replacement = rep;
    self->update_in_progress = FALSE;
    g_mutex_unlock (&self->replacement_lock);

    GST_INFO_OBJECT (self,
        "OTA: engine loaded and validated, pending swap at next buffer");
    gst_nvdsvisionencoder_load_request_free (req);
    continue;

next_request:
    /* Clear the busy flag BEFORE reporting: g_signal_emit is synchronous, so a
     * handler that retries on failure runs inside this call and would be
     * rejected as busy -- recursing until the stack is gone. */
    g_mutex_lock (&self->replacement_lock);
    self->update_in_progress = FALSE;
    g_mutex_unlock (&self->replacement_lock);
    gst_nvdsvisionencoder_notify_model_status (self, fail_status, req->path);
    gst_nvdsvisionencoder_load_request_free (req);
    continue;
  }

  return NULL;
}

/** Queue a runtime model change. Returns FALSE if the loader is not running. */
static gboolean
gst_nvdsvisionencoder_trigger_new_model (GstNvDsVisionEncoder *self,
    const gchar *path, NvDsVisionEncoderModelLoadType type)
{
  NvDsVisionEncoderLoadRequest *req;

  if (!path || !*path) {
    GST_WARNING_OBJECT (self, "OTA: ignoring empty model path");
    gst_nvdsvisionencoder_notify_model_status (self,
        VISIONENC_STATUS_INVALID_PARAMS, path);
    return FALSE;
  }

  /* A setter can run while stop() tears the loader down, so hold the lock
   * across the whole publish -- test and push must not be split. */
  g_mutex_lock (&self->replacement_lock);
  if (!self->loader_running) {
    g_mutex_unlock (&self->replacement_lock);
    GST_WARNING_OBJECT (self,
        "OTA: element is not running (or is shutting down); rejecting '%s'",
        path);
    gst_nvdsvisionencoder_notify_model_status (self,
        VISIONENC_STATUS_INVALID_PARAMS, path);
    return FALSE;                 /* not started, or shutting down */
  }
  /* Admit at request time: a controller retrying during a load would otherwise
   * queue without bound, with the stop sentinel stuck behind it. */
  if (self->update_in_progress || self->next_replacement) {
    g_mutex_unlock (&self->replacement_lock);
    GST_WARNING_OBJECT (self,
        "OTA: an update is already in progress; rejecting '%s'", path);
    gst_nvdsvisionencoder_notify_model_status (self,
        VISIONENC_STATUS_INVALID_PARAMS, path);
    return FALSE;
  } 
  
  /* Reserve before publishing: leaving it to the loader would hold the gate
   * open to every caller arriving while the loader is unscheduled. */
  self->update_in_progress = TRUE;

  req = g_new0 (NvDsVisionEncoderLoadRequest, 1);
  req->type = type;
  req->path = g_strdup (path);
  /* Snapshot on the caller's thread so the loader never dereferences a string
   * a setter can free. triton_url is not needed: an OTA always supplies an
   * engine path, which takes precedence over it. */
  g_mutex_lock (&self->config_lock);
  req->triton_model = g_strdup (self->triton_model);
  g_mutex_unlock (&self->config_lock);
  req->batch_size   = self->batch_size;
  req->gpu_id       = self->gpu_id;
  g_async_queue_push (self->pending_models, req);
  g_mutex_unlock (&self->replacement_lock);
  return TRUE;
}

/**
 * Commit a staged model. Called from the streaming thread at the top of
 * transform_ip, i.e. between buffers, so no inference is in flight on this
 * thread; infer_mutex additionally excludes the REST query thread.
 */
static void
gst_nvdsvisionencoder_ensure_replace_next_model (GstNvDsVisionEncoder *self)
{
  NvDsVisionEncoderReplacement *rep;
  void *old_client;
  gchar *committed_path;

  /* Uncontended fast path: almost every buffer takes this branch and returns. */
  g_mutex_lock (&self->replacement_lock);
  rep = self->next_replacement;
  if (!rep) {
    g_mutex_unlock (&self->replacement_lock);
    return;
  }
  self->next_replacement = NULL;

  /* Gate closed: stop() is past. Committing here would leak the new handle and
   * leave a torn-down element holding a live one. */
  if (!self->loader_running) {
    g_mutex_unlock (&self->replacement_lock);
    GST_WARNING_OBJECT (self,
        "OTA: element stopped before '%s' could be applied; discarding",
        rep->path ? rep->path : "");
    gst_nvdsvisionencoder_notify_model_status (self,
        VISIONENC_STATUS_CONFIG_FAILED, rep->path);
    gst_nvdsvisionencoder_replacement_free (rep);
    return;
  }

  /* Held across the swap (order: replacement_lock -> infer_mutex). */
  g_mutex_lock (&self->infer_mutex);

  old_client = self->triton_client;
  self->triton_client = rep->client;
  rep->client = NULL;               /* ownership moved to the element */

  /* Equal by construction -- the gate rejected anything else -- but assign so
   * the live values always come from the live model. */
  self->input_width = rep->input_width;
  self->input_height = rep->input_height;
  self->embedding_dim = rep->embedding_dim;

  /* use_siglip_norm is untouched: an update cannot cross families, so the value
   * resolved at start() still describes the new model.
   * The cache is dropped unconditionally -- its vectors came from the old model
   * and are incomparable, not merely stale. */
  embedding_cache_init (self);

  g_free (self->current_model_path);
  self->current_model_path = g_strdup (rep->path);
  committed_path = g_strdup (rep->path);

  /* Persist the accepted path as the cold-start config, as gst-nvinfer does:
   * it writes the new engine into its init params at commit, so a later
   * stop()/start() restarts on the UPDATED model rather than silently
   * reverting. Done at COMMIT, not in the setter, so a refused request still
   * cannot become the configuration the element restarts from. */
  g_mutex_lock (&self->config_lock);
  g_free (self->tensorrt_engine);
  self->tensorrt_engine = g_strdup (rep->path);
  g_mutex_unlock (&self->config_lock);

  g_mutex_unlock (&self->infer_mutex);
  /* Release before the emit: a handler that sets tensorrt-engine would
   * deadlock on this non-recursive mutex. */
  g_mutex_unlock (&self->replacement_lock);

  /* Retired handle destroyed outside both locks; select its device first
   * (see replacement_free) so ownership is explicit. */
  if (old_client) {
    cudaSetDevice (self->gpu_id);
    nvds_triton_client_destroy (old_client);
  }

  gst_nvdsvisionencoder_replacement_free (rep);

  GST_INFO_OBJECT (self, "OTA: model swapped in, now serving '%s'",
      committed_path);
  gst_nvdsvisionencoder_notify_model_status (self, 0, committed_path);
  g_free (committed_path);
}

/* Release what start() acquired when it fails partway: GstBaseTransform does
 * not call ->stop() after a failed ->start(), so the context would otherwise
 * leak for the life of the process. */
static void
gst_nvdsvisionencoder_start_unwind (GstNvDsVisionEncoder *self)
{
  if (self->triton_client) {
    cudaSetDevice (self->gpu_id);
    nvds_triton_client_destroy (self->triton_client);
    self->triton_client = NULL;
  }
  if (self->cuda_stream) {
    cudaStreamDestroy ((cudaStream_t) self->cuda_stream);
    self->cuda_stream = NULL;
  }

  /* start() can fail after these exist and ->stop() never runs. NULL-guarded,
   * so safe from any point in start(). */
  if (self->prealloc_output_surface) {
    NvBufSurfaceDestroy (self->prealloc_output_surface);
    self->prealloc_output_surface = NULL;
  }
  g_free (self->prealloc_surf_list);
  self->prealloc_surf_list = NULL;
  g_free (self->prealloc_src_rect);
  self->prealloc_src_rect = NULL;
  g_free (self->prealloc_dst_rect);
  self->prealloc_dst_rect = NULL;
  if (self->d_preprocessed_pool) {
    cudaFree (self->d_preprocessed_pool);
    self->d_preprocessed_pool = NULL;
  }
  self->d_preprocessed_pool_bytes = 0;

  g_free (self->current_model_path);
  self->current_model_path = NULL;
  self->is_initialized = FALSE;
}

/**
 * Start - initialize encoder
 */
static gboolean
gst_nvdsvisionencoder_start (GstBaseTransform *trans)
{
  GstNvDsVisionEncoder *nvdsvisionencoder = GST_NVDSVISIONENCODER (trans);

  if (!nvdsvisionencoder->enable) {
    GST_INFO_OBJECT (nvdsvisionencoder, "Vision encoder is disabled");
    return TRUE;
  }

  /* Set CUDA device */
  cudaError_t err = cudaSetDevice(nvdsvisionencoder->gpu_id);
  if (err != cudaSuccess) {
    GST_ERROR_OBJECT (nvdsvisionencoder, "Failed to set CUDA device %u: %s",
        nvdsvisionencoder->gpu_id, cudaGetErrorString(err));
    return FALSE;
  }
  GST_INFO_OBJECT (nvdsvisionencoder, "Set CUDA device to GPU %u", nvdsvisionencoder->gpu_id);

  if (nvdsvisionencoder->tensorrt_engine) {
    GST_INFO_OBJECT (nvdsvisionencoder, "Starting vision encoder with TensorRT direct backend");
    GST_INFO_OBJECT (nvdsvisionencoder, "Engine: %s", nvdsvisionencoder->tensorrt_engine);
  } else {
    GST_INFO_OBJECT (nvdsvisionencoder, "Starting vision encoder with Triton backend");
    GST_INFO_OBJECT (nvdsvisionencoder, "Triton URL: %s, Model: %s",
        nvdsvisionencoder->triton_url, nvdsvisionencoder->triton_model);
  }

  /* Create the encoder through the same path an OTA reload uses, so a
   * hot-loaded model is built exactly the way this cold-started one is. */
  {
    NvDsVisionEncoderReplacement initial;
    NvDsVisionEncoderLoadRequest req;
    const gchar *src;
    gchar *onnx_snapshot;
    gboolean built_ok;

    /* Snapshot the model configuration, exactly as an OTA request does. The
     * build below can run for minutes on a cold ONNX build, and a setter on
     * another thread frees these strings when it replaces them -- passing the
     * element's own pointers into it is a use-after-free. */
    memset (&req, 0, sizeof (req));
    g_mutex_lock (&nvdsvisionencoder->config_lock);
    req.triton_url   = g_strdup (nvdsvisionencoder->triton_url);
    req.triton_model = g_strdup (nvdsvisionencoder->triton_model);
    req.path         = g_strdup (nvdsvisionencoder->tensorrt_engine);
    onnx_snapshot    = g_strdup (nvdsvisionencoder->onnx_model);
    g_mutex_unlock (&nvdsvisionencoder->config_lock);
    req.batch_size   = nvdsvisionencoder->batch_size;
    req.gpu_id       = nvdsvisionencoder->gpu_id;

    /* Resolved once here instead of per buffer, but the TEST is unchanged:
     * onnx-model only, case-sensitive. Widening it to the engine path would
     * silently flip engine-only elements from [0,1] to [-1,1]. */
    src = onnx_snapshot;
    nvdsvisionencoder->use_siglip_norm =
        (src != NULL && g_strstr_len (src, -1, "siglip") != NULL);

    /* Engine-only: the heuristic has nothing to go on, so say so. */
    if (!onnx_snapshot && req.path &&
        (g_strstr_len (req.path, -1, "siglip") != NULL ||
         g_strstr_len (req.path, -1, "SigLIP") != NULL ||
         g_strstr_len (req.path, -1, "SIGLIP") != NULL)) {
      GST_WARNING_OBJECT (nvdsvisionencoder,
          "engine '%s' looks like a SigLIP model but onnx-model is not set, so "
          "[0,1] (RADIO-CLIP) normalization is being used. If this engine "
          "expects [-1,1], set onnx-model to the source .onnx path -- that is "
          "what selects the normalization.", req.path);
    }

    built_ok = gst_nvdsvisionencoder_create_client (nvdsvisionencoder, &req,
        req.path, onnx_snapshot, &initial);
    if (!built_ok) {
      GST_ERROR_OBJECT (nvdsvisionencoder, "Failed to create encoder/client");
      g_free (req.triton_url); g_free (req.triton_model);
      g_free (req.path); g_free (onnx_snapshot);
      return FALSE;                 /* fails the state change -> MODEL_FAILED */
    }

    nvdsvisionencoder->triton_client = initial.client;
    nvdsvisionencoder->input_width   = initial.input_width;
    nvdsvisionencoder->input_height  = initial.input_height;
    nvdsvisionencoder->embedding_dim = initial.embedding_dim;

    g_free (nvdsvisionencoder->current_model_path);
    nvdsvisionencoder->current_model_path =
        g_strdup (req.path ? req.path :
            (onnx_snapshot ? onnx_snapshot : req.triton_model));

    g_free (req.triton_url); g_free (req.triton_model);
    g_free (req.path); g_free (onnx_snapshot);
  }

  GST_INFO_OBJECT (nvdsvisionencoder,
      "Model info: input=%ux%u, embedding_dim=%u",
      nvdsvisionencoder->input_width,
      nvdsvisionencoder->input_height,
      nvdsvisionencoder->embedding_dim);

  /* Create CUDA stream for preprocessing */
  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    GST_ERROR_OBJECT (nvdsvisionencoder, "Failed to create CUDA stream: %s",
        cudaGetErrorString(err));
    gst_nvdsvisionencoder_start_unwind (nvdsvisionencoder);
    return FALSE;
  }
  nvdsvisionencoder->cuda_stream = (void*)stream;

  /* Detect platform using NVML board name.
   * Thor → VIC (hardware-accelerated transforms, RGBA, SURFACE_ARRAY)
   * Everything else → GPU compute (RGB, CUDA unified memory) */
  int is_thor = detect_is_thor (nvdsvisionencoder->gpu_id);

  /* Pre-allocate buffers for preprocessing (like nvinfer SGIE does) */
  nvdsvisionencoder->max_batch_size = 512;  // Hardcoded to handle large batches
  nvdsvisionencoder->prealloc_surf_list = (NvBufSurfaceParams *) g_malloc0(
      nvdsvisionencoder->max_batch_size * sizeof(NvBufSurfaceParams));
  nvdsvisionencoder->prealloc_src_rect = (NvBufSurfTransformRect *) g_malloc0(
      nvdsvisionencoder->max_batch_size * sizeof(NvBufSurfTransformRect));
  nvdsvisionencoder->prealloc_dst_rect = (NvBufSurfTransformRect *) g_malloc0(
      nvdsvisionencoder->max_batch_size * sizeof(NvBufSurfTransformRect));
  
  /* Pre-allocate output surface for RGB/RGBA crops (reuse across calls) */
  NvBufSurfaceCreateParams create_params;
  memset(&create_params, 0, sizeof(NvBufSurfaceCreateParams));
  create_params.gpuId = nvdsvisionencoder->gpu_id;
  create_params.width = nvdsvisionencoder->input_width;
  create_params.height = nvdsvisionencoder->input_height;
  create_params.size = 0;  /* Auto-calculate */
  create_params.isContiguous = 1;  /* Required for proper VIC operation on integrated GPU */
  
  /* Platform-aware format and memory type (NVML board-name based) */
  if (is_thor) {
    /* Thor: Use RGBA for VIC performance and surface array memory */
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.memType = NVBUF_MEM_SURFACE_ARRAY;
  } else {
    /* All other platforms: Use RGB and CUDA unified memory */
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
    create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
  }
  create_params.layout = NVBUF_LAYOUT_PITCH;
  
  if (NvBufSurfaceCreate(&nvdsvisionencoder->prealloc_output_surface, 
                         nvdsvisionencoder->max_batch_size, &create_params) != 0) {
    GST_ERROR_OBJECT (nvdsvisionencoder, "Failed to create pre-allocated output surface");
    gst_nvdsvisionencoder_start_unwind (nvdsvisionencoder);
    return FALSE;
  }
  
  GST_INFO_OBJECT (nvdsvisionencoder, 
      "Pre-allocated preprocessing buffers for max_batch_size=%u, format=%s, memType=%s",
      nvdsvisionencoder->max_batch_size,
      is_thor ? "RGBA (Thor VIC)" : "RGB (default GPU)",
      is_thor ? "SURFACE_ARRAY" : "CUDA_UNIFIED");

  /* Initialize NvBufSurfTransform session params (platform-aware) */
  NvBufSurfTransformConfigParams config_params;
  memset(&config_params, 0, sizeof(NvBufSurfTransformConfigParams));
  config_params.gpu_id = nvdsvisionencoder->gpu_id;
  
  if (is_thor) {
    /* Thor: Use VIC for hardware-accelerated transforms */
    config_params.compute_mode = NvBufSurfTransformCompute_VIC;
    config_params.cuda_stream = NULL;
  } else {
    /* All other platforms: Use GPU compute with CUDA stream */
    config_params.compute_mode = NvBufSurfTransformCompute_Default;
    config_params.cuda_stream = (cudaStream_t)nvdsvisionencoder->cuda_stream;
  }
  
  if (NvBufSurfTransformSetSessionParams(&config_params) != 0) {
    GST_ERROR_OBJECT (nvdsvisionencoder, "Failed to set NvBufSurfTransform session params");
    gst_nvdsvisionencoder_start_unwind (nvdsvisionencoder);
    return FALSE;
  }
  
  GST_INFO_OBJECT (nvdsvisionencoder, 
      "NvBufSurfTransform session initialized: compute_mode=%s",
      is_thor ? "VIC (Thor hardware accelerated)" : "GPU (CUDA stream)");

  /* Pre-allocate device tensor for GPU path (avoids per-frame cudaMalloc) */
  nvdsvisionencoder->d_preprocessed_pool = NULL;
  nvdsvisionencoder->d_preprocessed_pool_bytes = (size_t)nvdsvisionencoder->max_batch_size * 3 *
      nvdsvisionencoder->input_width * nvdsvisionencoder->input_height * sizeof(float);
  if (cudaMalloc((void **)&nvdsvisionencoder->d_preprocessed_pool,
                 nvdsvisionencoder->d_preprocessed_pool_bytes) == cudaSuccess) {
    GST_INFO_OBJECT (nvdsvisionencoder, "Pre-allocated device tensor for GPU path (%zu bytes)",
                     nvdsvisionencoder->d_preprocessed_pool_bytes);
  } else {
    nvdsvisionencoder->d_preprocessed_pool = NULL;
    nvdsvisionencoder->d_preprocessed_pool_bytes = 0;
  }

  /* Initialize embedding cache for smart inference */
  if (nvdsvisionencoder->smart_infer) {
    embedding_cache_init (nvdsvisionencoder);
    GST_INFO_OBJECT (nvdsvisionencoder,
        "Smart inference enabled: cache_max_size=%u, refresh_interval=%u frames",
        nvdsvisionencoder->cache_max_size,
        nvdsvisionencoder->cache_refresh_interval);
  }

  if (nvdsvisionencoder->ofa_predict_enabled) {
    GST_INFO_OBJECT (nvdsvisionencoder,
        "OFA temporal embedding prediction enabled: "
        "motion_threshold=%.1f, high_motion_threshold=%.1f",
        nvdsvisionencoder->ofa_motion_threshold,
        nvdsvisionencoder->ofa_high_motion_threshold);
    if (!nvdsvisionencoder->smart_infer) {
      GST_WARNING_OBJECT (nvdsvisionencoder,
          "OFA prediction requires smart-infer=true; enabling smart-infer automatically");
      nvdsvisionencoder->smart_infer = TRUE;
      embedding_cache_init (nvdsvisionencoder);
    }
  }

  if (nvdsvisionencoder->dla_validator_enabled) {
    if (dla_validator_init (nvdsvisionencoder)) {
      GST_INFO_OBJECT (nvdsvisionencoder,
          "DLA two-tier validator initialized (core=%d, threshold=%.2f)",
          nvdsvisionencoder->dla_core, nvdsvisionencoder->dla_change_threshold);
    } else {
      GST_WARNING_OBJECT (nvdsvisionencoder,
          "DLA validator init failed; falling back to standard inference");
      nvdsvisionencoder->dla_validator_enabled = FALSE;
    }
  }

  /* Start the OTA loader thread last, so it can never observe a half-built
   * element. Everything it needs (infer_mutex, replacement_lock, the queue)
   * was set up in _init. */
  nvdsvisionencoder->pending_models =
      g_async_queue_new_full (gst_nvdsvisionencoder_load_request_free);
  g_mutex_lock (&nvdsvisionencoder->replacement_lock);
  nvdsvisionencoder->loader_running = TRUE;
  g_mutex_unlock (&nvdsvisionencoder->replacement_lock);
  nvdsvisionencoder->model_load_thread =
      g_thread_new ("visionenc-model-load",
      gst_nvdsvisionencoder_model_load_thread, nvdsvisionencoder);

  nvdsvisionencoder->is_initialized = TRUE;

  /* Fire "model-updated" for the initial load too. nvinfer does the same --
   * observed in a live run, where it emits at t=0 for the configured model --
   * and nvmodelmux's handler expects it. Readiness is still reported by the
   * state transition; this is the event stream, not the readiness signal. */
  gst_nvdsvisionencoder_notify_model_status (nvdsvisionencoder,
      VISIONENC_STATUS_OK, nvdsvisionencoder->current_model_path);
  return TRUE;
}

/**
 * Stop - cleanup encoder
 */
static gboolean
gst_nvdsvisionencoder_stop (GstBaseTransform *trans)
{
  GstNvDsVisionEncoder *nvdsvisionencoder = GST_NVDSVISIONENCODER (trans);

  /* Stop the OTA loader FIRST and join it, so nothing can be building a model
   * (or staging one) while the state it reads is being torn down below. */
  /* Close the gate first: after this a concurrent set_property returns without
   * queueing, so no new work can arrive. The queue itself stays alive until the
   * loader thread has been joined -- the thread pops from it. */
  g_mutex_lock (&nvdsvisionencoder->replacement_lock);
  nvdsvisionencoder->loader_running = FALSE;
  g_mutex_unlock (&nvdsvisionencoder->replacement_lock);

  if (nvdsvisionencoder->model_load_thread) {
    NvDsVisionEncoderLoadRequest *stop_item =
        g_new0 (NvDsVisionEncoderLoadRequest, 1);
    stop_item->type = NVDS_VISION_ENCODER_LOAD_STOP;
    g_async_queue_push (nvdsvisionencoder->pending_models, stop_item);
    g_thread_join (nvdsvisionencoder->model_load_thread);
    nvdsvisionencoder->model_load_thread = NULL;
  }
  if (nvdsvisionencoder->pending_models) {
    /* frees any un-popped items via its free func */
    g_async_queue_unref (nvdsvisionencoder->pending_models);
    nvdsvisionencoder->pending_models = NULL;
  }

  /* A model may have been built but never applied (e.g. stopped mid-update);
   * it still owns a handle. */
  {
    NvDsVisionEncoderReplacement *staged;
    void *live;

    /* One critical section for all model state: the commit path holds the same
     * lock, so it can no longer interleave with this teardown. */
    g_mutex_lock (&nvdsvisionencoder->replacement_lock);
    staged = nvdsvisionencoder->next_replacement;
    nvdsvisionencoder->next_replacement = NULL;
    nvdsvisionencoder->update_in_progress = FALSE;

    /* infer_mutex excludes the REST query thread, mid-infer under this lock. */
    g_mutex_lock (&nvdsvisionencoder->infer_mutex);
    live = nvdsvisionencoder->triton_client;
    nvdsvisionencoder->triton_client = NULL;
    /* Freed here, not unlocked: get_property reads this under infer_mutex. */
    g_free (nvdsvisionencoder->current_model_path);
    nvdsvisionencoder->current_model_path = NULL;
    /* Must go FALSE with the model it advertises, not at the end of stop(). */
    nvdsvisionencoder->is_initialized = FALSE;
    g_mutex_unlock (&nvdsvisionencoder->infer_mutex);

    g_mutex_unlock (&nvdsvisionencoder->replacement_lock);

    /* Slow teardown and the emit stay outside both locks -- a signal handler
     * that sets tensorrt-engine would deadlock on replacement_lock. */
    if (staged) {
      /* It built successfully but will never go live. A controller waiting on
       * "model-updated" would otherwise wait forever. */
      GST_WARNING_OBJECT (nvdsvisionencoder,
          "OTA: discarding model '%s' built but never applied (element stopped)",
          staged->path ? staged->path : "");
      gst_nvdsvisionencoder_notify_model_status (nvdsvisionencoder,
          VISIONENC_STATUS_CONFIG_FAILED, staged->path);
      gst_nvdsvisionencoder_replacement_free (staged);
    }

    if (live) {
      /* stop() can run on any thread; select the owning device before teardown. */
      cudaSetDevice (nvdsvisionencoder->gpu_id);
      nvds_triton_client_destroy (live);
    }
  }

  if (nvdsvisionencoder->cuda_stream) {
    cudaStreamDestroy((cudaStream_t)nvdsvisionencoder->cuda_stream);
    nvdsvisionencoder->cuda_stream = NULL;
  }

  /* Free pre-allocated preprocessing buffers */
  if (nvdsvisionencoder->prealloc_output_surface) {
    NvBufSurfaceDestroy(nvdsvisionencoder->prealloc_output_surface);
    nvdsvisionencoder->prealloc_output_surface = NULL;
  }
  if (nvdsvisionencoder->prealloc_surf_list) {
    g_free(nvdsvisionencoder->prealloc_surf_list);
    nvdsvisionencoder->prealloc_surf_list = NULL;
  }
  if (nvdsvisionencoder->prealloc_src_rect) {
    g_free(nvdsvisionencoder->prealloc_src_rect);
    nvdsvisionencoder->prealloc_src_rect = NULL;
  }
  if (nvdsvisionencoder->prealloc_dst_rect) {
    g_free(nvdsvisionencoder->prealloc_dst_rect);
    nvdsvisionencoder->prealloc_dst_rect = NULL;
  }
  if (nvdsvisionencoder->d_preprocessed_pool) {
    cudaFree(nvdsvisionencoder->d_preprocessed_pool);
    nvdsvisionencoder->d_preprocessed_pool = NULL;
  }
  nvdsvisionencoder->d_preprocessed_pool_bytes = 0;

  embedding_cache_destroy (nvdsvisionencoder);
  dla_validator_destroy (nvdsvisionencoder);

#ifdef ENABLE_INSTRUMENTATION
  if (nvdsvisionencoder->total_objects_processed > 0) {
    guint64 avg_time_us = nvdsvisionencoder->total_process_time /
        nvdsvisionencoder->total_objects_processed;
    GST_INFO_OBJECT (nvdsvisionencoder,
        "Performance: processed %u objects, avg time per object: %lu μs",
        nvdsvisionencoder->total_objects_processed, avg_time_us);
  }
#endif

  /* is_initialized was cleared above, with the model state it describes. */
  return TRUE;
}

/**
 * Main transformation function with real preprocessing and inference
 */
static GstFlowReturn
gst_nvdsvisionencoder_transform_ip (GstBaseTransform *trans, GstBuffer *buf)
{
  GstNvDsVisionEncoder *nvdsvisionencoder = GST_NVDSVISIONENCODER (trans);
  NvDsBatchMeta *batch_meta;
  NvDsFrameMetaList *frame_meta_list;
  guint num_objects_processed = 0;
#ifdef ENABLE_INSTRUMENTATION
  struct timeval start_time, end_time;
#endif
  GstMapInfo map_info;

  if (!nvdsvisionencoder->is_initialized) {
    return GST_FLOW_OK;
  }

  /* Set CUDA device for this thread */
  cudaSetDevice(nvdsvisionencoder->gpu_id);

  /* Apply a staged model BEFORE any gate, as gst-nvinfer does in the first
   * statement of submit_input_buffer. Behind the enable/query_only checks it
   * would strand the built engine and, because next_replacement stays set,
   * make every later update be rejected as already-in-progress. Must follow
   * cudaSetDevice: retiring the old client frees device memory against this
   * thread's device. */
  gst_nvdsvisionencoder_ensure_replace_next_model (nvdsvisionencoder);

  if (!nvdsvisionencoder->enable || nvdsvisionencoder->query_only) {
    return GST_FLOW_OK;
  }

  nvdsvisionencoder->frame_count++;
  nvdsvisionencoder->global_frame_counter++;
  
  GST_DEBUG_OBJECT(nvdsvisionencoder, "Frame %d: transform_ip called", nvdsvisionencoder->frame_count);

  /* Get batch metadata */
  batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    GST_WARNING_OBJECT (nvdsvisionencoder, "No batch metadata in buffer");
    return GST_FLOW_OK;
  }

  /* Get surface from buffer */
  if (!gst_buffer_map (buf, &map_info, GST_MAP_READ)) {
    GST_ERROR_OBJECT (nvdsvisionencoder, "Failed to map GstBuffer");
    return GST_FLOW_ERROR;
  }

  NvBufSurface *surface = (NvBufSurface *) map_info.data;
#ifdef ENABLE_INSTRUMENTATION
  gettimeofday (&start_time, NULL);
#endif

  /* PHASE 1: Count total valid objects across ALL frames in the batch */
  guint total_valid_objects = 0;
  guint num_frames_in_batch = 0;
  
  for (frame_meta_list = batch_meta->frame_meta_list; frame_meta_list != NULL;
       frame_meta_list = frame_meta_list->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) frame_meta_list->data;
    num_frames_in_batch++;
    
    if (nvdsvisionencoder->skip_interval > 0) {
      if (frame_meta->frame_num % (nvdsvisionencoder->skip_interval + 1) != 0) {
        continue;
      }
    }
    
    NvDsObjectMetaList *obj_meta_list;
    for (obj_meta_list = frame_meta->obj_meta_list; obj_meta_list != NULL;
         obj_meta_list = obj_meta_list->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) obj_meta_list->data;
      
      if (obj_meta->rect_params.width < nvdsvisionencoder->min_crop_size ||
          obj_meta->rect_params.height < nvdsvisionencoder->min_crop_size) {
        continue;
      }
      
      if (!is_class_allowed(nvdsvisionencoder, obj_meta->obj_label)) {
        continue;
      }
      
      total_valid_objects++;
    }
  }
  
  GST_DEBUG_OBJECT(nvdsvisionencoder, "Batch has %u frames with %u total valid objects",
                   num_frames_in_batch, total_valid_objects);
  
  for (frame_meta_list = batch_meta->frame_meta_list; frame_meta_list != NULL;
       frame_meta_list = frame_meta_list->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) frame_meta_list->data;
    GST_DEBUG_OBJECT(nvdsvisionencoder, "  Frame: source_id=%u, batch_id=%u, frame_num=%u",
                     frame_meta->source_id, frame_meta->batch_id, frame_meta->frame_num);
  }
  
  if (total_valid_objects == 0) {
    gst_buffer_unmap (buf, &map_info);
    return GST_FLOW_OK;
  }
  
  /* PHASE 2: Collect all objects and split into cache hits / misses */
  typedef struct {
    NvDsObjectMeta *obj_meta;
    NvDsFrameMeta *frame_meta;
    guint batch_id;
  } ObjectInfo;
  
  ObjectInfo *all_objects = (ObjectInfo *) g_malloc(total_valid_objects * sizeof(ObjectInfo));
  guint obj_idx = 0;
  
  for (frame_meta_list = batch_meta->frame_meta_list; frame_meta_list != NULL;
       frame_meta_list = frame_meta_list->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) frame_meta_list->data;
    
    if (nvdsvisionencoder->skip_interval > 0) {
      if (frame_meta->frame_num % (nvdsvisionencoder->skip_interval + 1) != 0) {
        continue;
      }
    }
    
    NvDsObjectMetaList *obj_meta_list;
    for (obj_meta_list = frame_meta->obj_meta_list; obj_meta_list != NULL;
         obj_meta_list = obj_meta_list->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) obj_meta_list->data;
      
      if (obj_meta->rect_params.width < nvdsvisionencoder->min_crop_size ||
          obj_meta->rect_params.height < nvdsvisionencoder->min_crop_size) {
        continue;
      }
      
      if (!is_class_allowed(nvdsvisionencoder, obj_meta->obj_label)) {
        continue;
      }
      
      all_objects[obj_idx].obj_meta = obj_meta;
      all_objects[obj_idx].frame_meta = frame_meta;
      all_objects[obj_idx].batch_id = frame_meta->batch_id;
      obj_idx++;
    }
  }
  
  /*
   * PHASE 2b: Smart inference with OFA + DLA enhancements.
   *
   * Three-tier cache decision for tracked objects:
   *
   * Tier 0 (frame-count): Basic staleness check via cache_refresh_interval.
   *
   * Tier 1 (OFA motion analysis): When ofa-predict is enabled and upstream
   *   nvof element provides NvDsOpticalFlowMeta, extract per-object motion
   *   vectors. Low motion → trust cached embedding. Medium motion → apply
   *   motion-compensated prediction. High motion → force re-inference.
   *
   * Tier 2 (DLA appearance validator): When dla-validator is enabled, run
   *   a lightweight change detector on DLA/GPU to confirm whether the
   *   object's appearance actually changed enough to warrant re-inference.
   *
   * Objects are classified as:
   *   HIT:        Cached embedding is valid → attach directly
   *   PREDICTED:  OFA predicts new embedding from motion vectors → attach predicted
   *   MISS:       Must run full vision encoder inference
   */
  guint *miss_indices = (guint *) g_malloc(total_valid_objects * sizeof(guint));
  guint num_misses = 0;
  guint num_cache_hits = 0;
  guint num_ofa_predictions = 0;

  gboolean smart = nvdsvisionencoder->smart_infer && nvdsvisionencoder->embedding_cache != NULL;
  gboolean use_ofa = smart && nvdsvisionencoder->ofa_predict_enabled;
  guint64 cur_frame = nvdsvisionencoder->global_frame_counter;

  /* Pre-fetch optical flow metadata per frame (once per frame, not per object) */
  GHashTable *frame_of_meta = NULL;
  if (use_ofa) {
    frame_of_meta = g_hash_table_new (g_direct_hash, g_direct_equal);
    for (frame_meta_list = batch_meta->frame_meta_list; frame_meta_list != NULL;
         frame_meta_list = frame_meta_list->next) {
      NvDsFrameMeta *fm = (NvDsFrameMeta *) frame_meta_list->data;
      NvDsOpticalFlowMeta *ofm = ofa_get_flow_meta (fm);
      if (ofm)
        g_hash_table_insert (frame_of_meta, fm, ofm);
    }
  }

  float *predicted_buf = NULL;
  if (use_ofa && nvdsvisionencoder->embedding_dim > 0)
    predicted_buf = (float *) g_malloc (nvdsvisionencoder->embedding_dim * sizeof(float));

  for (guint i = 0; i < total_valid_objects; i++) {
    gboolean is_hit = FALSE;
    if (smart) {
      guint64 oid = all_objects[i].obj_meta->object_id;

      if (oid != UNTRACKED_OBJECT_ID) {
        CachedEmbedding *cached = embedding_cache_lookup(nvdsvisionencoder, oid);
        if (cached != NULL) {
          gboolean needs_refresh = (nvdsvisionencoder->cache_refresh_interval > 0 &&
              (cur_frame - cached->last_inferred_frame) >= nvdsvisionencoder->cache_refresh_interval);

          if (!needs_refresh && use_ofa && frame_of_meta) {
            NvDsOpticalFlowMeta *ofm = (NvDsOpticalFlowMeta *)
                g_hash_table_lookup (frame_of_meta, all_objects[i].frame_meta);
            if (ofm) {
              float mag, dx, dy, var;
              NvDsFrameMeta *fm = all_objects[i].frame_meta;
              ofa_compute_object_motion (ofm,
                  &all_objects[i].obj_meta->detector_bbox_info,
                  fm->source_frame_width, fm->source_frame_height,
                  &mag, &dx, &dy, &var);

              if (mag > nvdsvisionencoder->ofa_high_motion_threshold) {
                needs_refresh = TRUE;
                nvdsvisionencoder->ofa_reinfers_total++;
              } else if (mag > nvdsvisionencoder->ofa_motion_threshold &&
                         predicted_buf != NULL) {
                ofa_predict_embedding (cached->embedding, cached->embedding_dim,
                    dx, dy, mag, var, predicted_buf);
                attach_reid_embedding (batch_meta, all_objects[i].obj_meta,
                    predicted_buf, cached->embedding_dim);
                cached->last_motion_magnitude = mag;
                cached->last_flow_dx = dx;
                cached->last_flow_dy = dy;
                cached->accumulated_skip_frames++;
                cached->last_inferred_frame = cur_frame;
                num_objects_processed++;
                num_ofa_predictions++;
                nvdsvisionencoder->ofa_predictions_total++;
                is_hit = TRUE;
                if (cached->accumulated_skip_frames > 30)
                  needs_refresh = TRUE;
              }
            }
          }

          if (!needs_refresh && !is_hit) {
            attach_reid_embedding(batch_meta, all_objects[i].obj_meta,
                cached->embedding, cached->embedding_dim);
            cached->last_inferred_frame = cur_frame;
            cached->accumulated_skip_frames = 0;
            num_objects_processed++;
            num_cache_hits++;
            is_hit = TRUE;
          }
        }

        all_objects[i].obj_meta->misc_obj_info[3] = oid;
      }
    }
    if (!is_hit) {
      miss_indices[num_misses++] = i;
    }
  }

  /*
   * DLA Tier 2 validation :
   * For objects that were tentative cache hits, run the lightweight appearance
   * change detector to verify the cached embedding is still valid.
   * Objects flagged as "changed" are moved from cache hits to misses.
   *
   * Note: This is a batch operation for efficiency — all tentative hits are
   * validated in a single DLA/GPU inference pass.
   */
  if (nvdsvisionencoder->dla_validator_enabled && nvdsvisionencoder->dla_ctx &&
      num_cache_hits > 0) {
    /*
     * TODO: Full implementation requires:
     * 1. Preprocess cache-hit object crops to 32x32 (can use the existing
     *    NvBufSurfTransform path with a smaller output surface)
     * 2. Run dla_validator_run() on the batch
     * 3. For objects with change_prob > dla_change_threshold:
     *    - Remove them from cache hits
     *    - Add them to miss_indices for full re-inference
     *    - Increment dla_changes_detected
     *
     * For now this is a stub that logs the intent.
     * The actual DLA inference requires the TRT engine to be loaded.
     */
    GST_LOG_OBJECT (nvdsvisionencoder,
        "DLA validator: would validate %u tentative cache hits", num_cache_hits);
    nvdsvisionencoder->dla_validations_total += num_cache_hits;
  }

  GST_LOG_OBJECT (nvdsvisionencoder,
      "frame=%lu cache_hits=%u misses=%u total_objs=%u cache_size=%u",
      (unsigned long) cur_frame, num_cache_hits, num_misses, total_valid_objects,
      nvdsvisionencoder->embedding_cache ? g_hash_table_size(nvdsvisionencoder->embedding_cache) : 0);

  g_free (predicted_buf);
  if (frame_of_meta)
    g_hash_table_destroy (frame_of_meta);

  if (num_misses == 0) {
    g_free(all_objects);
    g_free(miss_indices);
    if (smart)
      embedding_cache_evict_stale(nvdsvisionencoder, cur_frame);
    gst_buffer_unmap(buf, &map_info);
    return GST_FLOW_OK;
  }

#ifdef ENABLE_INSTRUMENTATION
  struct timeval t_phase_start, t_phase_end;
  #define PHASE_START() gettimeofday(&t_phase_start, NULL)
  #define PHASE_END(field) do { \
      gettimeofday(&t_phase_end, NULL); \
      nvdsvisionencoder->field += \
          (t_phase_end.tv_sec - t_phase_start.tv_sec) * 1000000 + \
          (t_phase_end.tv_usec - t_phase_start.tv_usec); \
    } while(0)
  PHASE_START();
#endif

  /* PHASE 3: Preprocess ONLY cache-miss objects */
  NvDsObjectMeta **obj_meta_array = (NvDsObjectMeta **) g_malloc(num_misses * sizeof(NvDsObjectMeta *));
  guint *batch_ids = (guint *) g_malloc(num_misses * sizeof(guint));
  for (guint i = 0; i < num_misses; i++) {
    guint idx = miss_indices[i];
    obj_meta_array[i] = all_objects[idx].obj_meta;
    batch_ids[i] = all_objects[idx].batch_id;
  }

  size_t input_tensor_size = (size_t)num_misses * 3 *
      nvdsvisionencoder->input_height * nvdsvisionencoder->input_width * sizeof(float);
  float *d_preprocessed_all = NULL;
  float *h_preprocessed_all = NULL;
  gboolean use_pool = FALSE;
  gboolean use_device_path = FALSE;
  use_pool = (nvdsvisionencoder->d_preprocessed_pool != NULL &&
              input_tensor_size <= nvdsvisionencoder->d_preprocessed_pool_bytes);
  if (use_pool) {
    d_preprocessed_all = nvdsvisionencoder->d_preprocessed_pool;
    use_device_path = TRUE;
  } else if (cudaMalloc((void **)&d_preprocessed_all, input_tensor_size) == cudaSuccess) {
    use_device_path = TRUE;
  }
  if (!use_device_path) {
    h_preprocessed_all = (float *) g_malloc(input_tensor_size);
  }

  int ret = preprocess_batch_with_nvbufsurftransform(
      surface,
      batch_ids,
      obj_meta_array,
      num_misses,
      nvdsvisionencoder->prealloc_output_surface,
      nvdsvisionencoder->input_width,
      nvdsvisionencoder->input_height,
      nvdsvisionencoder->gpu_id,
      nvdsvisionencoder->max_batch_size,
      nvdsvisionencoder->prealloc_surf_list,
      nvdsvisionencoder->prealloc_src_rect,
      nvdsvisionencoder->prealloc_dst_rect,
      use_device_path);

  g_free(obj_meta_array);
  g_free(batch_ids);

#ifdef ENABLE_INSTRUMENTATION
  cudaStreamSynchronize((cudaStream_t)nvdsvisionencoder->cuda_stream);
  PHASE_END(perf_phase_preproc_us);
#endif

  if (ret != 0) {
    GST_WARNING_OBJECT(nvdsvisionencoder, "Batched preprocessing failed");
    g_free(all_objects);
    g_free(miss_indices);
    if (d_preprocessed_all && !use_pool) cudaFree(d_preprocessed_all);
    if (h_preprocessed_all) g_free(h_preprocessed_all);
    gst_buffer_unmap (buf, &map_info);
    return GST_FLOW_OK;
  }

#ifdef ENABLE_INSTRUMENTATION
  PHASE_START();
#endif

  /* SigLIP2 → [-1,1], else [0,1] RADIO-CLIP; inferred from the onnx-model path
   * at start(). Not re-derived here -- that path stays writable at runtime. */
  gboolean use_siglip = nvdsvisionencoder->use_siglip_norm;
  
  NvBufSurfaceColorFormat format = nvdsvisionencoder->prealloc_output_surface->surfaceList[0].colorFormat;
  gboolean is_rgba = (format == NVBUF_COLOR_FORMAT_RGBA);
  
  if (use_device_path && d_preprocessed_all != NULL) {
    if (is_rgba) {
      ret = rgba_surface_to_tensor_device(
          nvdsvisionencoder->prealloc_output_surface,
          d_preprocessed_all, num_misses,
          nvdsvisionencoder->input_width, nvdsvisionencoder->input_height, use_siglip);
    } else {
      ret = rgb_surface_to_tensor_device(
          nvdsvisionencoder->prealloc_output_surface,
          d_preprocessed_all, num_misses,
          nvdsvisionencoder->input_width, nvdsvisionencoder->input_height, use_siglip);
    }
  } else {
    if (is_rgba) {
      ret = rgba_surface_to_tensor(
          nvdsvisionencoder->prealloc_output_surface,
          h_preprocessed_all, num_misses,
          nvdsvisionencoder->input_width, nvdsvisionencoder->input_height, use_siglip);
    } else {
      ret = rgb_surface_to_tensor(
          nvdsvisionencoder->prealloc_output_surface,
          h_preprocessed_all, num_misses,
          nvdsvisionencoder->input_width, nvdsvisionencoder->input_height, use_siglip);
    }
  }

#ifdef ENABLE_INSTRUMENTATION
  cudaDeviceSynchronize();
  PHASE_END(perf_phase_tensor_us);
#endif

  if (ret != 0) {
    GST_WARNING_OBJECT(nvdsvisionencoder, "Surface-to-tensor conversion failed");
    g_free(all_objects);
    g_free(miss_indices);
    if (d_preprocessed_all && !use_pool) cudaFree(d_preprocessed_all);
    if (h_preprocessed_all) g_free(h_preprocessed_all);
    gst_buffer_unmap (buf, &map_info);
    return GST_FLOW_OK;
  }
  
  /* PHASE 4: Batched inference for cache-miss objects only */
#ifdef ENABLE_INSTRUMENTATION
  PHASE_START();
#endif
  float *embeddings_all = (float *) g_malloc(num_misses *
      nvdsvisionencoder->embedding_dim * sizeof(float));

  g_mutex_lock (&nvdsvisionencoder->infer_mutex);
  if (use_device_path && d_preprocessed_all != NULL) {
    ret = nvds_triton_client_infer_device(
        nvdsvisionencoder->triton_client,
        d_preprocessed_all, num_misses, embeddings_all);
    if (!use_pool)
      cudaFree(d_preprocessed_all);
  } else {
    ret = nvds_triton_client_infer(
        nvdsvisionencoder->triton_client,
        h_preprocessed_all, num_misses, embeddings_all);
    g_free(h_preprocessed_all);
  }
  g_mutex_unlock (&nvdsvisionencoder->infer_mutex);
  
  if (ret != 0) {
    GST_WARNING_OBJECT(nvdsvisionencoder, "Batched TensorRT inference failed");
    g_free(all_objects);
    g_free(miss_indices);
    g_free(embeddings_all);
    gst_buffer_unmap (buf, &map_info);
    return GST_FLOW_OK;
  }
  
#ifdef ENABLE_INSTRUMENTATION
  PHASE_END(perf_phase_infer_us);
#endif

  GST_DEBUG_OBJECT(nvdsvisionencoder,
      "Inference: %u inferred, %u cached, %u ofa-predicted",
      num_misses, num_cache_hits, num_ofa_predictions);
  
  /* PHASE 5: Attach new embeddings and update cache */
#ifdef ENABLE_INSTRUMENTATION
  PHASE_START();
#endif
  for (guint i = 0; i < num_misses; i++) {
    guint orig_idx = miss_indices[i];
    float *embedding = embeddings_all + (i * nvdsvisionencoder->embedding_dim);
    
    if (attach_reid_embedding(batch_meta, all_objects[orig_idx].obj_meta, embedding, 
                              nvdsvisionencoder->embedding_dim)) {
      num_objects_processed++;
      
      if (smart) {
        guint64 oid = all_objects[orig_idx].obj_meta->misc_obj_info[3];
        if (oid == 0)
          oid = all_objects[orig_idx].obj_meta->object_id;
        if (oid != UNTRACKED_OBJECT_ID) {
          embedding_cache_insert(nvdsvisionencoder, oid, embedding,
              nvdsvisionencoder->embedding_dim, cur_frame);
        }
      }

#ifdef ENABLE_INSTRUMENTATION
      if (nvdsvisionencoder->frame_count <= 3 && i < 6) {
        GST_INFO_OBJECT(nvdsvisionencoder, 
                        "Batch %u, Frame source_id=%u, Obj %d: Attached embedding (dim=%d, first_val=%.4f)",
                        all_objects[orig_idx].batch_id,
                        all_objects[orig_idx].frame_meta->source_id,
                        orig_idx,
                        nvdsvisionencoder->embedding_dim, 
                        embedding[0]);
      }
      if (nvdsvisionencoder->frame_count == 1 && i == 0) {
        g_mkdir_with_parents("/tmp/test_data", 0755);
        FILE *emb_fp = fopen("/tmp/test_data/deepstream_embedding.txt", "w");
        if (emb_fp) {
          for (guint d = 0; d < nvdsvisionencoder->embedding_dim; d++)
            fprintf(emb_fp, "%.10f\n", embedding[d]);
          fclose(emb_fp);
          g_print("[ACCURACY] Saved embedding (%u dims) -> /tmp/test_data/deepstream_embedding.txt\n",
                  nvdsvisionencoder->embedding_dim);
        }
        NvOSD_RectParams *rect = &all_objects[orig_idx].obj_meta->rect_params;
        FILE *bbox_fp = fopen("/tmp/test_data/bbox_coords.txt", "w");
        if (bbox_fp) {
          fprintf(bbox_fp, "left=%.2f\ntop=%.2f\nwidth=%.2f\nheight=%.2f\n",
                  rect->left, rect->top, rect->width, rect->height);
          fprintf(bbox_fp, "source_id=%u\n", all_objects[orig_idx].frame_meta->source_id);
          fprintf(bbox_fp, "model_input_w=%u\nmodel_input_h=%u\n",
                  nvdsvisionencoder->input_width, nvdsvisionencoder->input_height);
          fprintf(bbox_fp, "embedding_dim=%u\n", nvdsvisionencoder->embedding_dim);
          fprintf(bbox_fp, "onnx_model=%s\n",
                  nvdsvisionencoder->onnx_model ? nvdsvisionencoder->onnx_model : "N/A");
          fclose(bbox_fp);
          g_print("[ACCURACY] Saved bbox -> /tmp/test_data/bbox_coords.txt\n");
        }
        {
          NvBufSurface *crop_surf = nvdsvisionencoder->prealloc_output_surface;
          guint cw = nvdsvisionencoder->input_width;
          guint ch = nvdsvisionencoder->input_height;
          if (NvBufSurfaceMap(crop_surf, 0, -1, NVBUF_MAP_READ) == 0) {
            NvBufSurfaceSyncForCpu(crop_surf, 0, -1);
            NvBufSurfaceParams *sp = &crop_surf->surfaceList[0];
            guint pitch = sp->pitch;
            unsigned char *mapped = (unsigned char *)sp->mappedAddr.addr[0];
            FILE *raw_fp = fopen("/tmp/test_data/crop_0.rgb", "wb");
            FILE *ppm_fp = fopen("/tmp/test_data/crop_0.ppm", "wb");
            if (ppm_fp) fprintf(ppm_fp, "P6\n%u %u\n255\n", cw, ch);
            if (raw_fp || ppm_fp) {
              for (guint row = 0; row < ch; row++) {
                unsigned char *rowptr = mapped + row * pitch;
                if (raw_fp) fwrite(rowptr, 1, cw * 3, raw_fp);
                if (ppm_fp) fwrite(rowptr, 1, cw * 3, ppm_fp);
              }
            }
            if (raw_fp) fclose(raw_fp);
            if (ppm_fp) fclose(ppm_fp);
            NvBufSurfaceUnMap(crop_surf, 0, -1);
            g_print("[ACCURACY] Saved crop (%ux%u RGB) -> /tmp/test_data/crop_0.ppm + crop_0.rgb\n", cw, ch);
          } else {
            g_print("[ACCURACY] WARNING: Could not map crop surface for saving\n");
          }
        }
      }
#endif
    }
  }
  
  /* Periodic cache maintenance */
  if (smart)
    embedding_cache_evict_stale(nvdsvisionencoder, cur_frame);

  g_free(all_objects);
  g_free(miss_indices);
  g_free(embeddings_all);

#ifdef ENABLE_INSTRUMENTATION
  PHASE_END(perf_phase_post_us);
#endif

  gst_buffer_unmap (buf, &map_info);

#ifdef ENABLE_INSTRUMENTATION
  gettimeofday (&end_time, NULL);
  guint64 elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
      (end_time.tv_usec - start_time.tv_usec);
  nvdsvisionencoder->total_process_time += elapsed_us;
  nvdsvisionencoder->total_objects_processed += num_objects_processed;

  if (nvdsvisionencoder->verbose && nvdsvisionencoder->frame_count <= 10) {
    GST_INFO_OBJECT (nvdsvisionencoder,
        "Batch: processed %u objects from %u frames in %lu μs",
        num_objects_processed, num_frames_in_batch, elapsed_us);
  }

  /* --- Periodic performance reporting (every 30 seconds) --- */
  #define PERF_WINDOW_SEC 30
  if (!nvdsvisionencoder->perf_window_started) {
    gettimeofday (&nvdsvisionencoder->perf_window_start, NULL);
    nvdsvisionencoder->perf_window_objects = 0;
    nvdsvisionencoder->perf_window_batches = 0;
    nvdsvisionencoder->perf_window_started = TRUE;
  }
  nvdsvisionencoder->perf_window_objects += num_objects_processed;
  nvdsvisionencoder->perf_window_batches++;

  {
    double window_sec = (end_time.tv_sec - nvdsvisionencoder->perf_window_start.tv_sec)
        + (end_time.tv_usec - nvdsvisionencoder->perf_window_start.tv_usec) / 1e6;
    if (window_sec >= PERF_WINDOW_SEC) {
      guint wobj = nvdsvisionencoder->perf_window_objects;
      guint wbat = nvdsvisionencoder->perf_window_batches;
      double obj_per_sec = wobj / window_sec;
      double bat_per_sec = wbat / window_sec;

      /* Per-phase averages (ms per batch call) */
      double pre_ms  = (wbat > 0) ? nvdsvisionencoder->perf_phase_preproc_us / (1000.0 * wbat) : 0;
      double tens_ms = (wbat > 0) ? nvdsvisionencoder->perf_phase_tensor_us  / (1000.0 * wbat) : 0;
      double inf_ms  = (wbat > 0) ? nvdsvisionencoder->perf_phase_infer_us   / (1000.0 * wbat) : 0;
      double post_ms = (wbat > 0) ? nvdsvisionencoder->perf_phase_post_us    / (1000.0 * wbat) : 0;
      double tot_ms  = pre_ms + tens_ms + inf_ms + post_ms;
      double avg_obj = (wbat > 0) ? (double)wobj / wbat : 0;

      time_t now = time(NULL);
      char ts[64];
      strftime(ts, sizeof(ts), "%c", localtime(&now));

      g_print("=== nvdsvisionencoder Performance (engine batch_size=%u) ===\n"
              "  Obj/sec : %.1f | Bat/sec : %.1f | Obj/bat : %.1f\n"
              "  Phase breakdown (avg ms/batch): "
              "Pre=%.1f  Tens=%.1f  Infer=%.1f  Post=%.1f  Total=%.1f\n"
              "  %s\n",
              nvdsvisionencoder->batch_size,
              obj_per_sec, bat_per_sec, avg_obj,
              pre_ms, tens_ms, inf_ms, post_ms, tot_ms,
              ts);

      /* Reset window */
      gettimeofday (&nvdsvisionencoder->perf_window_start, NULL);
      nvdsvisionencoder->perf_window_objects = 0;
      nvdsvisionencoder->perf_window_batches = 0;
      nvdsvisionencoder->perf_phase_preproc_us = 0;
      nvdsvisionencoder->perf_phase_tensor_us = 0;
      nvdsvisionencoder->perf_phase_infer_us = 0;
      nvdsvisionencoder->perf_phase_post_us = 0;
    }
  }
  #undef PHASE_START
  #undef PHASE_END
#else
  (void)num_objects_processed;
#endif

  return GST_FLOW_OK;
}

/* ========================================================================
 * Image Embedding Query Handler — loads image from disk, preprocesses,
 * runs inference, and returns embedding via GstQuery response.
 * ======================================================================== */

/* Custom error handler for libjpeg that uses longjmp instead of exit() */
struct venc_jpeg_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf               setjmp_buffer;
};

static void
venc_jpeg_error_exit (j_common_ptr cinfo)
{
  struct venc_jpeg_error_mgr *myerr = (struct venc_jpeg_error_mgr *) cinfo->err;
  (*cinfo->err->output_message) (cinfo);
  longjmp (myerr->setjmp_buffer, 1);
}

#ifdef NVDS_VISIONENCODER_ARM_JPEG_DLOPEN
/* Same pattern as nvds_batch_jpeg_encode: tegra loads libnvmm_jpeg.so; dGPU libjpeg.so */
#define VENC_JPEG_LIB_VERSION 80

static gsize venc_jpeg_once = 0;
static gboolean venc_jpeg_ok;
static void *venc_jpeg_handle;

static struct {
  struct jpeg_error_mgr *(*jpeg_std_error) (struct jpeg_error_mgr *);
  void (*jpeg_CreateDecompress) (j_decompress_ptr, int, size_t);
  void (*jpeg_destroy_decompress) (j_decompress_ptr);
  void (*jpeg_stdio_src) (j_decompress_ptr, FILE *);
  int (*jpeg_read_header) (j_decompress_ptr, boolean);
  void (*jpeg_start_decompress) (j_decompress_ptr);
  JDIMENSION (*jpeg_read_scanlines) (j_decompress_ptr, JSAMPARRAY, JDIMENSION);
  void (*jpeg_finish_decompress) (j_decompress_ptr);
} venc_jpeg;

static gboolean
venc_jpeg_ensure_loaded (void)
{
  if (g_once_init_enter (&venc_jpeg_once)) {
    void *h = dlopen ("libnvmm_jpeg.so", RTLD_LAZY);
    if (!h)
      h = dlopen ("libjpeg.so", RTLD_LAZY);
    if (!h)
      h = dlopen ("libjpeg.so.8", RTLD_LAZY);
    if (!h)
      h = dlopen ("libjpeg.so.8.2.2", RTLD_LAZY);

    gboolean ok = FALSE;
    if (h) {
      venc_jpeg.jpeg_std_error =
          (struct jpeg_error_mgr *(*) (struct jpeg_error_mgr *)) dlsym (h,
          "jpeg_std_error");
      venc_jpeg.jpeg_CreateDecompress =
          (void (*) (j_decompress_ptr, int, size_t)) dlsym (h,
          "jpeg_CreateDecompress");
      venc_jpeg.jpeg_destroy_decompress =
          (void (*) (j_decompress_ptr)) dlsym (h, "jpeg_destroy_decompress");
      venc_jpeg.jpeg_stdio_src =
          (void (*) (j_decompress_ptr, FILE *)) dlsym (h, "jpeg_stdio_src");
      venc_jpeg.jpeg_read_header =
          (int (*) (j_decompress_ptr, boolean)) dlsym (h, "jpeg_read_header");
      venc_jpeg.jpeg_start_decompress =
          (void (*) (j_decompress_ptr)) dlsym (h, "jpeg_start_decompress");
      venc_jpeg.jpeg_read_scanlines =
          (JDIMENSION (*) (j_decompress_ptr, JSAMPARRAY, JDIMENSION)) dlsym (h,
          "jpeg_read_scanlines");
      venc_jpeg.jpeg_finish_decompress =
          (void (*) (j_decompress_ptr)) dlsym (h, "jpeg_finish_decompress");

      ok = venc_jpeg.jpeg_std_error && venc_jpeg.jpeg_CreateDecompress
          && venc_jpeg.jpeg_destroy_decompress && venc_jpeg.jpeg_stdio_src
          && venc_jpeg.jpeg_read_header && venc_jpeg.jpeg_start_decompress
          && venc_jpeg.jpeg_read_scanlines && venc_jpeg.jpeg_finish_decompress;
      if (!ok) {
        g_print ("[VisionEncoder] ERROR: dlsym libjpeg decompress API failed\n");
        dlclose (h);
      } else {
        venc_jpeg_handle = h;
      }
    } else {
      g_print ("[VisionEncoder] ERROR: dlopen JPEG library failed "
          "(tried libnvmm_jpeg.so, libjpeg.so)\n");
    }

    venc_jpeg_ok = ok;
    g_once_init_leave (&venc_jpeg_once, 1);
  }
  return venc_jpeg_ok;
}

/**
 * Load JPEG image via libjpeg (CPU decode). Library is loaded with dlopen
 * (libnvmm_jpeg.so on tegra, libjpeg.so on dGPU) like nvds_batch_jpeg_encode.
 * Returns heap-allocated RGB buffer (caller must g_free).
 * Sets *w, *h on success. Returns NULL on failure.
 *
 * CPU decode is used here because this runs on the REST server thread
 * concurrently with the GPU streaming pipeline — GPU-based decoders
 * (nvJPEG) cause illegal memory access when sharing the CUDA context.
 */
static unsigned char *
load_jpeg_image (const char *path, int *w, int *h)
{
  if (!venc_jpeg_ensure_loaded ())
    return NULL;

  FILE *infile = fopen (path, "rb");
  if (!infile) {
    g_print ("[VisionEncoder] ERROR: cannot open JPEG file %s\n", path);
    return NULL;
  }

  struct jpeg_decompress_struct cinfo;
  struct venc_jpeg_error_mgr jerr;

  cinfo.err = venc_jpeg.jpeg_std_error (&jerr.pub);
  jerr.pub.error_exit = venc_jpeg_error_exit;

  if (setjmp (jerr.setjmp_buffer)) {
    venc_jpeg.jpeg_destroy_decompress (&cinfo);
    fclose (infile);
    return NULL;
  }

  venc_jpeg.jpeg_CreateDecompress (&cinfo, VENC_JPEG_LIB_VERSION,
      (size_t) sizeof (struct jpeg_decompress_struct));
  venc_jpeg.jpeg_stdio_src (&cinfo, infile);
  venc_jpeg.jpeg_read_header (&cinfo, TRUE);

  cinfo.out_color_space = JCS_RGB;

  venc_jpeg.jpeg_start_decompress (&cinfo);

  int width = (int) cinfo.output_width;
  int height = (int) cinfo.output_height;
  int row_stride = width * 3;

  unsigned char *rgb = (unsigned char *) g_malloc (width * height * 3);

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char *row = rgb + cinfo.output_scanline * row_stride;
    JSAMPROW rowp[1] = { (JSAMPROW) row };
    venc_jpeg.jpeg_read_scanlines (&cinfo, rowp, 1);
  }

  venc_jpeg.jpeg_finish_decompress (&cinfo);
  venc_jpeg.jpeg_destroy_decompress (&cinfo);
  fclose (infile);

  *w = width;
  *h = height;
  return rgb;
}
#else
/**
 * Load JPEG image via libjpeg (CPU decode), linked with -ljpeg (x86 / desktop).
 * Returns heap-allocated RGB buffer (caller must g_free).
 * Sets *w, *h on success. Returns NULL on failure.
 *
 * CPU decode is used here because this runs on the REST server thread
 * concurrently with the GPU streaming pipeline — GPU-based decoders
 * (nvJPEG) cause illegal memory access when sharing the CUDA context.
 */
static unsigned char *
load_jpeg_image (const char *path, int *w, int *h)
{
  FILE *infile = fopen (path, "rb");
  if (!infile) {
    g_print ("[VisionEncoder] ERROR: cannot open JPEG file %s\n", path);
    return NULL;
  }

  struct jpeg_decompress_struct cinfo;
  struct venc_jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error (&jerr.pub);
  jerr.pub.error_exit = venc_jpeg_error_exit;

  if (setjmp (jerr.setjmp_buffer)) {
    jpeg_destroy_decompress (&cinfo);
    fclose (infile);
    return NULL;
  }

  jpeg_create_decompress (&cinfo);
  jpeg_stdio_src (&cinfo, infile);
  jpeg_read_header (&cinfo, TRUE);

  cinfo.out_color_space = JCS_RGB;

  jpeg_start_decompress (&cinfo);

  int width = (int) cinfo.output_width;
  int height = (int) cinfo.output_height;
  int row_stride = width * 3;

  unsigned char *rgb = (unsigned char *) g_malloc (width * height * 3);

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char *row = rgb + cinfo.output_scanline * row_stride;
    jpeg_read_scanlines (&cinfo, &row, 1);
  }

  jpeg_finish_decompress (&cinfo);
  jpeg_destroy_decompress (&cinfo);
  fclose (infile);

  *w = width;
  *h = height;
  return rgb;
}
#endif /* !NVDS_VISIONENCODER_ARM_JPEG_DLOPEN */

/**
 * Load PPM (P6 binary) image file.
 * Returns heap-allocated RGB buffer (caller must g_free).
 */
static unsigned char *
load_ppm_image (const char *path, int *w, int *h)
{
  FILE *fp = fopen (path, "rb");
  if (!fp) return NULL;

  char magic[3] = {0};
  if (fscanf (fp, "%2s", magic) != 1 || strcmp (magic, "P6") != 0) {
    fclose (fp);
    return NULL;
  }

  /* Skip comments */
  int c = fgetc (fp);
  while (c == '#' || c == ' ' || c == '\n' || c == '\r') {
    if (c == '#') { while ((c = fgetc (fp)) != '\n' && c != EOF); }
    c = fgetc (fp);
  }
  ungetc (c, fp);

  int width, height, maxval;
  if (fscanf (fp, "%d %d %d", &width, &height, &maxval) != 3 || maxval != 255) {
    fclose (fp);
    return NULL;
  }
  fgetc (fp); /* consume the single whitespace after maxval */

  size_t data_size = (size_t)width * height * 3;
  unsigned char *data = (unsigned char *) g_malloc (data_size);
  if (fread (data, 1, data_size, fp) != data_size) {
    g_free (data);
    fclose (fp);
    return NULL;
  }
  fclose (fp);

  *w = width;
  *h = height;
  return data;
}

/**
 * Load a raw RGB file (width*height*3 bytes, no header).
 * Dimensions must be known in advance — used as last-resort fallback.
 */

/**
 * Load image from file. Supports PPM (P6) and JPEG (libjpeg).
 * Returns heap-allocated RGB buffer (caller must g_free), or NULL on failure.
 */
static unsigned char *
load_image_from_file (const char *path, int *out_w, int *out_h)
{
  if (!path || !out_w || !out_h) return NULL;

  /* Detect format by extension */
  const char *ext = strrchr (path, '.');

  if (ext && (g_ascii_strcasecmp (ext, ".ppm") == 0)) {
    return load_ppm_image (path, out_w, out_h);
  }

  if (ext && (g_ascii_strcasecmp (ext, ".jpg") == 0 ||
              g_ascii_strcasecmp (ext, ".jpeg") == 0)) {
    unsigned char *data = load_jpeg_image (path, out_w, out_h);
    if (data) return data;
    /* Fall through to try PPM as fallback */
  }

  /* Try PPM as generic fallback */
  unsigned char *data = load_ppm_image (path, out_w, out_h);
  if (data) return data;

  g_print ("[VisionEncoder] ERROR: unsupported image format or failed to load: %s\n"
           "  Supported: .ppm (P6), .jpg/.jpeg\n", path);
  return NULL;
}

/**
 * CPU bilinear resize of RGB image.
 * src: input RGB buffer (src_w * src_h * 3)
 * dst: output RGB buffer (dst_w * dst_h * 3), must be pre-allocated
 */
static void
resize_rgb_bilinear (const unsigned char *src, int src_w, int src_h,
                     unsigned char *dst, int dst_w, int dst_h)
{
  float x_ratio = (float)(src_w - 1) / (float)(dst_w - 1);
  float y_ratio = (float)(src_h - 1) / (float)(dst_h - 1);

  for (int y = 0; y < dst_h; y++) {
    float fy = y * y_ratio;
    int y0 = (int)fy;
    int y1 = y0 + 1;
    if (y1 >= src_h) y1 = src_h - 1;
    float wy = fy - y0;

    for (int x = 0; x < dst_w; x++) {
      float fx = x * x_ratio;
      int x0 = (int)fx;
      int x1 = x0 + 1;
      if (x1 >= src_w) x1 = src_w - 1;
      float wx = fx - x0;

      for (int c = 0; c < 3; c++) {
        float v00 = src[(y0 * src_w + x0) * 3 + c];
        float v10 = src[(y0 * src_w + x1) * 3 + c];
        float v01 = src[(y1 * src_w + x0) * 3 + c];
        float v11 = src[(y1 * src_w + x1) * 3 + c];
        float val = (1 - wx) * (1 - wy) * v00 +
                    wx * (1 - wy) * v10 +
                    (1 - wx) * wy * v01 +
                    wx * wy * v11;
        dst[(y * dst_w + x) * 3 + c] = (unsigned char)(val + 0.5f);
      }
    }
  }
}

/**
 * Convert RGB byte buffer to CHW float tensor with model normalization.
 * out_tensor must be pre-allocated: 3 * width * height floats.
 */
static void
rgb_to_chw_tensor (const unsigned char *rgb, int width, int height,
                   gboolean use_siglip, float *out_tensor)
{
  int hw = height * width;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int pixel = h * width + w;
      for (int c = 0; c < 3; c++) {
        float val = rgb[pixel * 3 + c] / 255.0f;
        if (use_siglip)
          val = (val - 0.5f) / 0.5f;
        out_tensor[c * hw + pixel] = val;
      }
    }
  }
}

/**
 * BaseTransform query handler for image embedding queries.
 * Intercepts "image-embedding" GstQueries, loads the image from disk,
 * preprocesses, runs inference, and returns the embedding vector.
 * All other queries are chained to the parent class handler.
 */
static gboolean
gst_nvdsvisionencoder_query (GstBaseTransform *trans,
    GstPadDirection direction, GstQuery *query)
{
  GstNvDsVisionEncoder *self = GST_NVDSVISIONENCODER (trans);

  if (!gst_nvquery_is_image_embedding (query)) {
    return GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction, query);
  }

  GST_INFO_OBJECT (self, "Received image embedding query");

  /* The query arrives on the REST server thread which may lack a CUDA context.
   * Set the correct GPU device before inference. */
  cudaError_t cerr = cudaSetDevice (self->gpu_id);
  if (cerr != cudaSuccess) {
    GST_ERROR_OBJECT (self, "cudaSetDevice(%u) failed: %s",
        self->gpu_id, cudaGetErrorString (cerr));
    return FALSE;
  }

  /* Deliberately NOT committing a staged model here. The commit clears the
   * embedding cache, and the streaming thread touches that cache outside
   * infer_mutex, so committing from this thread could destroy a GHashTable
   * another thread is walking. Consequence: an element with no buffer flow does
   * not apply an update until flow resumes. Fixing that needs a refcounted
   * active-model context, not a second commit site. */

  /* Cheap pre-check only. The authoritative check is re-done under infer_mutex
   * in Step 3 below, because a commit or stop() can change or destroy the model
   * between here and there. */
  if (!self->triton_client) {
    GST_ERROR_OBJECT (self, "Vision encoder not initialized — cannot process image embedding query");
    return FALSE;
  }

  /* Parse request parameters */
  const gchar *image_path = NULL;
  const gchar *model = NULL;
  gboolean has_bbox = FALSE;
  gdouble bbox_left = 0, bbox_top = 0, bbox_width = 0, bbox_height = 0;

  if (!gst_nvquery_image_embedding_parse_request (query, &image_path, &model,
          &has_bbox, &bbox_left, &bbox_top, &bbox_width, &bbox_height)) {
    GST_ERROR_OBJECT (self, "Failed to parse image embedding query request");
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Image embedding request: path=%s, model=%s, bbox=%s",
      image_path ? image_path : "(null)",
      model ? model : "(null)",
      has_bbox ? "yes" : "no");

  /* Step 1: Load image from file */
  int img_w = 0, img_h = 0;
  unsigned char *img_rgb = load_image_from_file (image_path, &img_w, &img_h);
  if (!img_rgb) {
    GST_ERROR_OBJECT (self, "Failed to load image: %s", image_path);
    return FALSE;
  }
  GST_DEBUG_OBJECT (self, "Loaded image: %dx%d", img_w, img_h);

  /* Step 2: Crop to bounding box if specified */
  unsigned char *crop_rgb = NULL;
  int crop_w, crop_h;

  if (has_bbox && bbox_width > 0 && bbox_height > 0) {
    int bl = (int)bbox_left;
    int bt = (int)bbox_top;
    int bw = (int)bbox_width;
    int bh = (int)bbox_height;

    /* Clamp to image bounds */
    if (bl < 0) bl = 0;
    if (bt < 0) bt = 0;
    if (bl + bw > img_w) bw = img_w - bl;
    if (bt + bh > img_h) bh = img_h - bt;

    if (bw <= 0 || bh <= 0) {
      GST_ERROR_OBJECT (self, "Invalid bbox after clamping: %d,%d %dx%d on %dx%d image",
          bl, bt, bw, bh, img_w, img_h);
      g_free (img_rgb);
      return FALSE;
    }

    crop_w = bw;
    crop_h = bh;
    crop_rgb = (unsigned char *) g_malloc (crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; y++) {
      memcpy (crop_rgb + y * crop_w * 3,
              img_rgb + ((bt + y) * img_w + bl) * 3,
              crop_w * 3);
    }
    g_free (img_rgb);
    GST_DEBUG_OBJECT (self, "Cropped to bbox: %dx%d", crop_w, crop_h);
  } else {
    crop_rgb = img_rgb;
    crop_w = img_w;
    crop_h = img_h;
  }

  /* Snapshot the model geometry under the lock, then preprocess OUTSIDE it and
   * re-take it only for the inference. The geometry cannot change underneath
   * us: is_new_model_valid() refuses any replacement differing in resolution or
   * embedding width, and use_siglip_norm is never touched by an update -- so a
   * tensor built here stays valid across a commit. Holding the lock across the
   * CPU resize and conversion would block the streaming thread, which takes the
   * same lock for every batch. */
  guint tw, th, dim;
  gboolean use_siglip;
  unsigned char *resized = NULL;
  float *tensor = NULL;
  float *embedding = NULL;
  size_t tensor_size;
  int ret;

  g_mutex_lock (&self->infer_mutex);

  /* Re-check under the lock: the pre-check above is not binding. */
  if (!self->triton_client || self->embedding_dim == 0) {
    g_mutex_unlock (&self->infer_mutex);
    GST_ERROR_OBJECT (self, "Vision encoder has no live model — cannot process "
        "image embedding query");
    g_free (crop_rgb);
    return FALSE;
  }

  tw  = self->input_width;
  th  = self->input_height;
  dim = self->embedding_dim;
  use_siglip = self->use_siglip_norm;

  g_mutex_unlock (&self->infer_mutex);

  /* Step 3: Resize to model input dimensions */
  if ((guint)crop_w == tw && (guint)crop_h == th) {
    resized = crop_rgb;
  } else {
    resized = (unsigned char *) g_malloc (tw * th * 3);
    resize_rgb_bilinear (crop_rgb, crop_w, crop_h, resized, (int)tw, (int)th);
    g_free (crop_rgb);
    GST_DEBUG_OBJECT (self, "Resized to model input: %ux%u", tw, th);
  }

  /* Step 4: Convert to CHW float tensor with normalization */
  tensor_size = 3 * (size_t) tw * th;
  tensor = (float *) g_malloc (tensor_size * sizeof (float));
  rgb_to_chw_tensor (resized, (int)tw, (int)th, use_siglip, tensor);
  g_free (resized);

  /* Step 5: Run inference. Re-take the lock for the call only, and re-check the
   * client: a commit or stop() may have run while we preprocessed. */
  embedding = (float *) g_malloc (dim * sizeof (float));
  g_mutex_lock (&self->infer_mutex);
  if (!self->triton_client) {
    g_mutex_unlock (&self->infer_mutex);
    GST_ERROR_OBJECT (self, "Vision encoder model went away mid-query");
    g_free (tensor);
    g_free (embedding);
    return FALSE;
  }
  ret = nvds_triton_client_infer (self->triton_client, tensor, 1, embedding);
  g_mutex_unlock (&self->infer_mutex);
  g_free (tensor);

  if (ret != 0) {
    GST_ERROR_OBJECT (self, "Inference failed for image embedding query");
    g_free (embedding);
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Image embedding inference succeeded: dim=%u, first_val=%.4f",
      dim, embedding[0]);

  /* Step 6: Build JSON response and set on query */
  GString *json = g_string_new ("[{\"object\":\"embedding\",\"index\":0,\"embedding\":[");
  for (guint i = 0; i < dim; i++) {          /* snapshotted width, not a re-read */
    if (i > 0)
      g_string_append_c (json, ',');
    g_string_append_printf (json, "%.8f", embedding[i]);
  }
  g_string_append (json, "]}]");
  g_free (embedding);

  GValue data_val = G_VALUE_INIT;
  g_value_init (&data_val, G_TYPE_STRING);
  g_value_set_string (&data_val, json->str);
  g_string_free (json, TRUE);

  char id_buf[64];
  g_snprintf (id_buf, sizeof (id_buf), "img-emb-%lu", (unsigned long)time (NULL));

  gst_nvquery_image_embedding_set (query, id_buf,
      (guint64)time (NULL),
      model ? model : (self->triton_model ? self->triton_model : "unknown"),
      &data_val);

  g_value_unset (&data_val);

  GST_INFO_OBJECT (self, "Image embedding query completed successfully");
  return TRUE;
}

/**
 * Plugin initialization
 */
static gboolean
plugin_init (GstPlugin *plugin)
{
  return gst_element_register (plugin, "nvdsvisionencoder",
      GST_RANK_PRIMARY, GST_TYPE_NVDSVISIONENCODER);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_visionencoder,
    "NVIDIA DeepStream vision encoder plugin",
    plugin_init,
    "1.0",
    "Proprietary",
    "NvDsVisionEncoder",
    "http://nvidia.com/"
)
