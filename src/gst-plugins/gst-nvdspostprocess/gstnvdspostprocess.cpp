/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <string.h>
#include <sys/time.h>

#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <memory>

#include "gst-nvevent.h"
#include "gstnvdspostprocess.h"
#include "gst-nvquery.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvdspostprocess_debug);
#define GST_CAT_DEFAULT gst_nvdspostprocess_debug
#define DEFAULT_PROP_PREPROCESSOR_SUPPORT FALSE

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_POSTPROCESSLIB_NAME,
  PROP_GPU_DEVICE_ID,
  PROP_POSTPROCESSLIB_CONFIG_FILE,
  PROP_PREPROCESSOR_SUPPORT
};

/* Default values for properties */
#define DEFAULT_GPU_ID 0

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
static GstStaticPadTemplate gst_nvdspostprocess_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_nvdspostprocess_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvdspostprocess_parent_class parent_class
G_DEFINE_TYPE (GstNvDsPostProcess, gst_nvdspostprocess, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdspostprocess_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdspostprocess_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean
gst_nvdspostprocess_sink_event (GstBaseTransform * btrans, GstEvent *event);
/*
static gboolean gst_nvdspostprocess_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);*/
static gboolean gst_nvdspostprocess_start (GstBaseTransform * btrans);
static gboolean gst_nvdspostprocess_stop (GstBaseTransform * btrans);

static GstFlowReturn
gst_nvdspostprocess_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_nvdspostprocess_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);



/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvdspostprocess_class_init (GstNvDsPostProcessClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  //gstbasetransform_class->passthrough_on_same_caps = TRUE;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvdspostprocess_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvdspostprocess_get_property);
/*
  gstbasetransform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvdspostprocess_transform_caps);

  gstbasetransform_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_nvdspostprocess_fixate_caps);
  gstbasetransform_class->accept_caps =
      GST_DEBUG_FUNCPTR (gst_nvdspostprocess_accept_caps);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvdspostprocess_set_caps);
  */
  gstbasetransform_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvdspostprocess_sink_event);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdspostprocess_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdspostprocess_stop);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_nvdspostprocess_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_nvdspostprocess_generate_output);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_POSTPROCESSLIB_NAME,
          g_param_spec_string ("postprocesslib-name", "Post Process library",
            "Set postprocess library to be used",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_POSTPROCESSLIB_CONFIG_FILE,
          g_param_spec_string ("postprocesslib-config-file", "Post Process library config file",
            "Set postprocess library config file to be used",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_PREPROCESSOR_SUPPORT,
      g_param_spec_boolean ("preprocessor-support", "Preprocessor Support",
          "Postprocessor parsing support for custom preprocessor tensors",
          DEFAULT_PROP_PREPROCESSOR_SUPPORT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));


  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdspostprocess_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdspostprocess_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvDsPostProcess plugin for Transform/In-Place use-cases",
      "NvDsPostProcess Plugin for Transform/In-Place use-cases",
      "A postprocess algorithm can be hooked for Transform/In-Place use-cases",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvdspostprocess_init (GstNvDsPostProcess * nvdspostprocess)
{
   GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvdspostprocess);
  /* Initialize all property variables to default values */
  nvdspostprocess->gpu_id = DEFAULT_GPU_ID;
  nvdspostprocess->preprocessor_support = DEFAULT_PROP_PREPROCESSOR_SUPPORT;
  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  nvdspostprocess->sinkcaps =
      gst_static_pad_template_get_caps (&gst_nvdspostprocess_sink_template);
  nvdspostprocess->srccaps =
      gst_static_pad_template_get_caps (&gst_nvdspostprocess_src_template);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvdspostprocess_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsPostProcess *nvdspostprocess = GST_NVDSPOSTPROCESS (object);
  switch (prop_id) {
    case PROP_GPU_DEVICE_ID:
      nvdspostprocess->gpu_id = g_value_get_uint (value);
      break;
    case PROP_POSTPROCESSLIB_NAME:
      if (nvdspostprocess->postprocess_lib_name) {
        g_free(nvdspostprocess->postprocess_lib_name);
      }
      nvdspostprocess->postprocess_lib_name = (gchar *)g_value_dup_string (value);
      break;
    case PROP_POSTPROCESSLIB_CONFIG_FILE:
      if (nvdspostprocess->postprocess_lib_config_file) {
        g_free(nvdspostprocess->postprocess_lib_config_file);
      }
      nvdspostprocess->postprocess_lib_config_file = (gchar *)g_value_dup_string (value);

      if(nullptr != nvdspostprocess->algo_ctx) {
         nvdspostprocess->algo_ctx->SetConfigFile(nvdspostprocess->postprocess_lib_config_file);
      }
      break;
    case PROP_PREPROCESSOR_SUPPORT:
      nvdspostprocess->preprocessor_support = g_value_get_boolean (value);
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
gst_nvdspostprocess_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsPostProcess *nvdspostprocess = GST_NVDSPOSTPROCESS (object);

  switch (prop_id) {
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvdspostprocess->gpu_id);
      break;
    case PROP_POSTPROCESSLIB_NAME:
      g_value_set_string (value, nvdspostprocess->postprocess_lib_name);
      break;
    case PROP_POSTPROCESSLIB_CONFIG_FILE:
      g_value_set_string (value, nvdspostprocess->postprocess_lib_config_file);
      break;
    case PROP_PREPROCESSOR_SUPPORT:
      g_value_set_boolean (value, nvdspostprocess->preprocessor_support);
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
gst_nvdspostprocess_start (GstBaseTransform * btrans)
{
  GstNvDsPostProcess *nvdspostprocess = GST_NVDSPOSTPROCESS (btrans);
  std::string nvtx_str("GstNvDsPostProcess ");

  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (nvdspostprocess->gpu_id),
      "Unable to set cuda device");

  nvdspostprocess->nvtx_domain = nvtx_domain_ptr.release ();

  if (!nvdspostprocess->postprocess_lib_name){
    GST_ELEMENT_ERROR(nvdspostprocess,RESOURCE,NOT_FOUND,("property postprocesslib-name is not set"),
        ("Postprocess lib  is NULL"));
    return FALSE;
  }

  cudaStreamCreateWithFlags (&(nvdspostprocess->cu_nbstream), cudaStreamNonBlocking);

  {
      DSPostProcess_CreateParams params;

      params.m_element = btrans;
      params.m_gpuId = nvdspostprocess->gpu_id;
      params.m_cudaStream = nvdspostprocess->cu_nbstream;
      params.m_preprocessor_support = nvdspostprocess->preprocessor_support;

      nvdspostprocess->algo_factory = new DSPostProcessLibrary_Factory();
      nvdspostprocess->algo_ctx =
        nvdspostprocess->algo_factory->CreateCustomAlgoCtx(nvdspostprocess->postprocess_lib_name,
         (DSPostProcess_CreateParams*) &params);
      if (nvdspostprocess->algo_ctx == nullptr){

        GST_ELEMENT_ERROR(nvdspostprocess,RESOURCE,FAILED,("Failed to open postprocesslib %s",nvdspostprocess->postprocess_lib_name),
        ("Failed to open postprocesslib %s",nvdspostprocess->postprocess_lib_name));
        goto error;
      }

      if (nvdspostprocess->postprocess_lib_config_file){
       if (!nvdspostprocess->algo_ctx->SetConfigFile
           (nvdspostprocess->postprocess_lib_config_file)){
        GST_ELEMENT_ERROR(nvdspostprocess,LIBRARY,SETTINGS,(" Library config file %s parsing failed", nvdspostprocess->postprocess_lib_config_file),
            ("Postprocess lib config file %s parsing failed", nvdspostprocess->postprocess_lib_config_file));
         goto error;
       }
      }
      else{
        GST_ELEMENT_ERROR(nvdspostprocess,LIBRARY,SETTINGS,(" Library config file for postprocess lib not set!"),
            ("Postprocess lib config file is NULL"));
        goto error;
      }
  }


  return TRUE;

error:
  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_nvdspostprocess_stop (GstBaseTransform * btrans)
{
  GstNvDsPostProcess *nvdspostprocess = GST_NVDSPOSTPROCESS (btrans);

  nvdspostprocess->stop = TRUE;

  if (nvdspostprocess->cu_nbstream)
  {
      cudaStreamDestroy(nvdspostprocess->cu_nbstream);
      nvdspostprocess->cu_nbstream = NULL;
  }

  if (nvdspostprocess->algo_ctx)
    delete nvdspostprocess->algo_ctx;

  if (nvdspostprocess->algo_factory)
    delete nvdspostprocess->algo_factory;

  if (nvdspostprocess->vecProp)
    delete nvdspostprocess->vecProp;

  if (nvdspostprocess->postprocess_lib_name) {
    g_free(nvdspostprocess->postprocess_lib_name);
    nvdspostprocess->postprocess_lib_name = NULL;
  }
  if (nvdspostprocess->postprocess_lib_config_file) {
    g_free(nvdspostprocess->postprocess_lib_config_file);
    nvdspostprocess->postprocess_lib_config_file = NULL;
  }
  if (nvdspostprocess->postprocess_prop_string) {
    g_free(nvdspostprocess->postprocess_prop_string);
    nvdspostprocess->postprocess_prop_string = NULL;
  }

  GST_DEBUG_OBJECT (nvdspostprocess, "ctx lib released \n");
  return TRUE;
}

static gboolean
gst_nvdspostprocess_sink_event (GstBaseTransform * btrans, GstEvent *event)
{
    gboolean ret = TRUE;
    GstNvDsPostProcess *nvdspostprocess = GST_NVDSPOSTPROCESS (btrans);

    ret  = nvdspostprocess->algo_ctx->HandleEvent(event);
    if (!ret)
        return ret;

    return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event(btrans, event);
}


/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvdspostprocess_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstFlowReturn flow_ret;
  GstNvDsPostProcess *nvdspostprocess = GST_NVDSPOSTPROCESS (btrans);

  BufferResult result = BufferResult::Buffer_Async;
  GST_DEBUG("nvdspostprocess: Inside %s \n", __func__);

  // Call the callback of user provided library
  // check the return type
  // based on return type push the buffer or just return as use is going to handle pushing of data
  if (nvdspostprocess->algo_ctx)
  {
    cudaError_t cuErr = cudaSetDevice(nvdspostprocess->gpu_id);
    if(cuErr != cudaSuccess) {
      GST_ERROR_OBJECT(nvdspostprocess, "Unable to set cuda device");
      return GST_FLOW_ERROR;
    }
    result = nvdspostprocess->algo_ctx->ProcessBuffer (inbuf);
    nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME(nvdspostprocess));

    if (result == BufferResult::Buffer_Ok) {
      flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvdspostprocess), inbuf);
      GST_DEBUG("nvdspostprocess: -- Forwarding Buffer to downstream, flow_ret = %d\n", flow_ret);
      return flow_ret;
    } else if (result == BufferResult::Buffer_Drop) {
        GST_DEBUG("nvdspostprocess: -- Dropping Buffer");
        return GST_FLOW_OK;
    } else if (result == BufferResult::Buffer_Error) {
        GST_DEBUG ("nvdspostprocess: -- Buffer_Error Buffer");
        return GST_FLOW_ERROR;
    } else if (result == BufferResult::Buffer_Async) {
        GST_DEBUG ("nvdspostprocess: -- Buffer_Async Received, postprocess lib to push the Buffer to downstream\n");
        return GST_FLOW_OK;
    }
  }

  flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvdspostprocess), inbuf);
  GST_DEBUG ("nvdspostprocess: -- Sending Buffer to downstream, flow_ret = %d\n", flow_ret);
  return GST_FLOW_OK;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_nvdspostprocess_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  return GST_FLOW_OK;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdspostprocess_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdspostprocess_debug, "nvdspostprocess", 0,
      "nvdspostprocess plugin");

  return gst_element_register (plugin, "nvdspostprocess", GST_RANK_PRIMARY,
      GST_TYPE_NVDSPOSTPROCESS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_postprocess,
    DESCRIPTION, nvdspostprocess_plugin_init, "9.0", LICENSE, BINARY_PACKAGE,
    URL)
