/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <ostream>
#include <fstream>

#include <cuda_runtime.h>

#include <gst/gst.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>

#include "nvbufsurface.h"
#include "gstnvbufsurfacewrite.h"
#include "gstnvdsmeta.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvbufsurfacewrite_debug);
#define GST_CAT_DEFAULT gst_nvbufsurfacewrite_debug

#define DEFAULT_GPU_ID 0

enum
{
  /* FILL ME */
  MEM_FEATURE_NVMM,
  MEM_FEATURE_RAW

};
#ifndef PACKAGE
#define PACKAGE "nvbufsurfacewrite"
#endif

#define VERSION "1.8.2"
#define PACKAGE_DESCRIPTION "Gstreamer plugin for NvBufSurface writing"
/* Define under which licence the package has been released */
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer NvBufSurface Write Plugin"
/* Define to the home page for this package. */
#define PACKAGE_URL "http://nvidia.com/"

#define MEASURE_TIME

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
  printf("(%s): #%d %s ElaspedTime=%f TotalTime=%f\n", \
      GST_ELEMENT_NAME(nvbufsurfacewrite), nvbufsurfacewrite->frame_num, \
      X, elapsedTime, totalReadTime); \
}

#else
#define START_PROFILE
#define STOP_PROFILE(X)
#endif

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_GPU_DEVICE_ID,
  PROP_SILENT,
};

#define GST_CAPS_FEATURE_MEMORY_NVMM      "memory:NVMM"


inline bool CHECK_(int e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
      std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile;
        exit (-1);
        return false;
    }
    return true;
}

#define ck(call) CHECK_(call, __LINE__, __FILE__)

static GQuark _dsmeta_quark = 0;

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
/* Input capabilities. */
static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "NV12, I420, UYVY, YUY2, YVYU, GRAY8, BGRx, RGBA }")));

/* Output capabilities. */
static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "NV12, I420, UYVY, YUY2, YVYU, GRAY8, BGRx, RGBA }")));

#define gst_nvbufsurfacewrite_parent_class parent_class
G_DEFINE_TYPE (GstNvBufSurfaceWrite, gst_nvbufsurfacewrite, GST_TYPE_BASE_TRANSFORM);

static void gst_nvbufsurfacewrite_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvbufsurfacewrite_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

#if 1
static gboolean gst_nvbufsurfacewrite_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvBufSurfaceWrite *nvbufsurfacewrite = GST_NVBUFSURFACEWRITE (trans);
  GstCapsFeatures *ift = NULL;

  GST_DEBUG_OBJECT (nvbufsurfacewrite, "set_caps");

  if (!gst_video_info_from_caps (&nvbufsurfacewrite->in_info, incaps)) {
    GST_ERROR ("invalid input caps");
    return FALSE;
  }
  nvbufsurfacewrite->input_width = GST_VIDEO_INFO_WIDTH (&nvbufsurfacewrite->in_info);
  nvbufsurfacewrite->input_height = GST_VIDEO_INFO_HEIGHT (&nvbufsurfacewrite->in_info);
  nvbufsurfacewrite->input_fmt = GST_VIDEO_FORMAT_INFO_FORMAT (nvbufsurfacewrite->in_info.finfo);

  if (!gst_video_info_from_caps (&nvbufsurfacewrite->out_info, outcaps)) {
    GST_ERROR ("invalid output caps");
    return FALSE;
  }
  nvbufsurfacewrite->output_width = GST_VIDEO_INFO_WIDTH (&nvbufsurfacewrite->out_info);
  nvbufsurfacewrite->output_height = GST_VIDEO_INFO_HEIGHT (&nvbufsurfacewrite->out_info);
  nvbufsurfacewrite->output_fmt = GST_VIDEO_FORMAT_INFO_FORMAT (nvbufsurfacewrite->out_info.finfo);

  if (nvbufsurfacewrite->input_feature == nvbufsurfacewrite->output_feature)
  {
    // Check if in and out caps are same then set the passtrough mode TRUE
    if ( (nvbufsurfacewrite->output_width == nvbufsurfacewrite->input_width) &&
        (nvbufsurfacewrite->output_height == nvbufsurfacewrite->input_height) &&
        (GST_VIDEO_FORMAT_INFO_FORMAT(nvbufsurfacewrite->in_info.finfo) == GST_VIDEO_FORMAT_INFO_FORMAT(nvbufsurfacewrite->out_info.finfo)))
    {
      // Same caps
      nvbufsurfacewrite->is_same_caps = TRUE;
      g_print ("nvbufsurfacewrite: %s : INPUT W=%d H=%d F=%d\n", __func__, nvbufsurfacewrite->input_width,
          nvbufsurfacewrite->input_height, GST_VIDEO_FORMAT_INFO_FORMAT(nvbufsurfacewrite->in_info.finfo));
      g_print ("nvbufsurfacewrite: %s : OUTPUT W=%d H=%d F=%d\n", __func__, nvbufsurfacewrite->output_width,
          nvbufsurfacewrite->output_height, GST_VIDEO_FORMAT_INFO_FORMAT(nvbufsurfacewrite->out_info.finfo));
    }
  }

  if (nvbufsurfacewrite->is_same_caps == TRUE)
  {
    // Check for same caps, if yes then set passtrough as TRUE
    gst_base_transform_set_passthrough (trans, TRUE);
  }

  return TRUE;
}
#endif

static GstFlowReturn gst_nvbufsurfacewrite_transform_ip (GstBaseTransform * btrans,
    GstBuffer * buf)
{
  GstNvBufSurfaceWrite *nvbufsurfacewrite = GST_NVBUFSURFACEWRITE (btrans);
  GstMapInfo map;
  void *input_data = NULL;
  NvBufSurface *in_surface = NULL;
  unsigned int input_size = 0;
  guint source_id = 0;
  unsigned int i = 0;
  cudaError_t CUerr = cudaSuccess;
  std::ofstream outfile;
  int size = 0;
  int numFilled = 1;
  GstMeta *gst_meta = NULL;
  NvDsMeta *dsmeta = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  gpointer state = NULL;

  while ((gst_meta = gst_buffer_iterate_meta (buf, &state)))
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

  if (!gst_buffer_map (buf, &map, GST_MAP_READ))
  {
    GST_ERROR ("NvBufSurface Map Failed");
    gst_buffer_unmap (buf, &map);
    return GST_FLOW_ERROR;
  }

  GST_DEBUG_OBJECT (nvbufsurfacewrite, "transform_ip");

  nvbufsurfacewrite->frame_num++;
  //g_print ("%s: inside %s()\n", GST_ELEMENT_NAME(nvbufsurfacewrite), __func__);

  if (nvbufsurfacewrite->input_feature == MEM_FEATURE_NVMM)
  {
    in_surface = (NvBufSurface *) map.data;
    numFilled = in_surface->numFilled;

    // TODO : Fix this
    if (numFilled == 0)
      numFilled = 2;
  }
  else
  {
    input_data = map.data;
    input_size = map.size;
  }

  GST_LOG_OBJECT (nvbufsurfacewrite, "SETTING CUDA DEVICE = %d in nvbufsurfacewrite func=%s\n", nvbufsurfacewrite->gpu_id, __func__);
  CUerr = cudaSetDevice(nvbufsurfacewrite->gpu_id);
  if(CUerr != cudaSuccess)
  {
    printf ("\n *** Unable to set device in %s Line %d\n", __func__, __LINE__);
    exit (-1);
  }


  for (i=0; i < numFilled; i++)
  {
      if (batch_meta)
      {
	    NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
	    source_id = frame_meta->pad_index;
      }

	  std::string tmp = "_" + std::to_string(in_surface->surfaceList[i].width)
	  + "x" + std::to_string(in_surface->surfaceList[i].height) + "_" +
	    "BS-" + std::to_string(source_id);

	  input_data = in_surface->surfaceList[i].dataPtr;
	  input_size = in_surface->surfaceList[i].dataSize;

	  switch (nvbufsurfacewrite->input_fmt)
	  {
	  case GST_VIDEO_FORMAT_NV12:
	  {
		  std::string fname = GST_ELEMENT_NAME(nvbufsurfacewrite) + tmp + ".nv12";

		  size = (in_surface->surfaceList[i].width * in_surface->surfaceList[i].height * 3) / 2;

		  if (nvbufsurfacewrite->input_nvbufsurface == NULL)
		  {
			  ck (cudaMallocHost (&nvbufsurfacewrite->input_nvbufsurface, size));
			  outfile.open (fname, std::ofstream::out);
		  }
		  else
		  {
			  outfile.open (fname, std::ofstream::out | std::ofstream::app);
		  }

		  cudaMemcpy2D (nvbufsurfacewrite->input_nvbufsurface,
				  in_surface->surfaceList[i].width, input_data, in_surface->surfaceList[i].width,
				  in_surface->surfaceList[i].width,
				  (in_surface->surfaceList[i].height*3)/2, cudaMemcpyDeviceToHost);

		  outfile.write(reinterpret_cast<char*>(nvbufsurfacewrite->input_nvbufsurface), size);
		  outfile.close();
	  }
	  break;

	  case GST_VIDEO_FORMAT_RGBA:
	  case GST_VIDEO_FORMAT_BGRx:
	  {
		  std::string fname = GST_ELEMENT_NAME(nvbufsurfacewrite) + tmp;

		  if (nvbufsurfacewrite->input_fmt == GST_VIDEO_FORMAT_RGBA)
			  fname.append(".rgba");
		  else if (nvbufsurfacewrite->input_fmt == GST_VIDEO_FORMAT_BGRx)
			  fname.append(".bgrx");

		  size = in_surface->surfaceList[i].width *in_surface->surfaceList[i].height*4;

		  if (nvbufsurfacewrite->input_nvbufsurface == NULL)
		  {
			  ck (cudaMallocHost (&nvbufsurfacewrite->input_nvbufsurface, size));
			  outfile.open (fname, std::ofstream::out);
		  }
		  else
		  {
			  outfile.open (fname, std::ofstream::out | std::ofstream::app);
		  }

		  cudaMemcpy2D (nvbufsurfacewrite->input_nvbufsurface,
				  in_surface->surfaceList[i].width*4, input_data,
				  in_surface->surfaceList[i].width*4, in_surface->surfaceList[i].width*4,
				  in_surface->surfaceList[i].height, cudaMemcpyDeviceToHost);

		  outfile.write(reinterpret_cast<char*>(nvbufsurfacewrite->input_nvbufsurface), size);
		  outfile.close();
	  }
	  break;

	  default:
		  g_print ("%s : NOT SUPPORTED FORMAT\n", GST_ELEMENT_NAME(nvbufsurfacewrite));
		  break;
	  }
  }

  return GST_FLOW_OK;
}

static gboolean gst_nvbufsurfacewrite_start (GstBaseTransform * btrans)
{
  GstNvBufSurfaceWrite *nvbufsurfacewrite = GST_NVBUFSURFACEWRITE (btrans);
  CUcontext cuContext = NULL;
  CUresult result = CUDA_SUCCESS;
  cudaError_t CUerr = cudaSuccess;

  nvbufsurfacewrite->frame_num = 0;

  GST_LOG_OBJECT (nvbufsurfacewrite, "SETTING CUDA DEVICE = %d in nvbufsurfacewrite func=%s\n", nvbufsurfacewrite->gpu_id, __func__);
  CUerr = cudaSetDevice(nvbufsurfacewrite->gpu_id);
  if(CUerr != cudaSuccess)
  {
    printf ("\n *** Unable to set device in %s Line %d\n", __func__, __LINE__);
    exit (-1);
  }

  result = cuCtxGetCurrent(&cuContext);

  if((result != CUDA_SUCCESS) || (cuContext == NULL))
  {
    g_print ("Unable to create cuda context in the %s", GST_ELEMENT_NAME(nvbufsurfacewrite));
    exit (-1);
    return FALSE;
  }
  GST_LOG_OBJECT (nvbufsurfacewrite, "cuContext in %s is %p\n", __func__, cuContext);

  ck(cudaStreamCreateWithFlags(&(nvbufsurfacewrite->stream), cudaStreamNonBlocking));

  return TRUE;
}

static gboolean gst_nvbufsurfacewrite_stop (GstBaseTransform * btrans)
{
  GstNvBufSurfaceWrite *nvbufsurfacewrite = GST_NVBUFSURFACEWRITE (btrans);

  ck(cudaStreamDestroy(nvbufsurfacewrite->stream));

  GST_DEBUG_OBJECT (nvbufsurfacewrite, "gst_nvbufsurfacewrite_stop");
  return TRUE;
}

/* initialize the nvbufsurfacewrite's class */
  static void
gst_nvbufsurfacewrite_class_init (GstNvBufSurfaceWriteClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class = (GstBaseTransformClass *) klass;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_nvbufsurfacewrite_set_property;
  gobject_class->get_property = gst_nvbufsurfacewrite_get_property;

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvbufsurfacewrite_set_caps);
  gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_nvbufsurfacewrite_transform_ip);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvbufsurfacewrite_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvbufsurfacewrite_stop);

  gstbasetransform_class->passthrough_on_same_caps = TRUE;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
        FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
        "Set GPU Device ID",
        0, G_MAXUINT, DEFAULT_GPU_ID,
        GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  gst_element_class_set_details_simple(gstelement_class,
      "nvbufsurfacewrite",
      "nvbufsurfacewrite",
      "Gstreamer NvBufSurfaceWrite Element",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

static void gst_nvbufsurfacewrite_init (GstNvBufSurfaceWrite * filter)
{
  filter->sinkcaps =
    gst_static_pad_template_get_caps (&sink_factory);
  filter->srccaps =
    gst_static_pad_template_get_caps (&src_factory);

  filter->silent = FALSE;
  filter->input_nvbufsurface = NULL;

  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

static void gst_nvbufsurfacewrite_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvBufSurfaceWrite *filter = GST_NVBUFSURFACEWRITE (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      filter->gpu_id = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void gst_nvbufsurfacewrite_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvBufSurfaceWrite *filter = GST_NVBUFSURFACEWRITE (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, filter->gpu_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean nvbufsurfacewrite_init (GstPlugin * nvbufsurfacewrite)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvbufsurfacewrite_debug, "nvbufsurfacewrite",
      0, "Template nvbufsurfacewrite");

  return gst_element_register (nvbufsurfacewrite, "nvbufsurfacewrite", GST_RANK_NONE,
      GST_TYPE_NVBUFSURFACEWRITE);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvbufsurfacewrite,
    PACKAGE_DESCRIPTION,
    nvbufsurfacewrite_init,
    DS_VERSION,
    PACKAGE_LICENSE,
    PACKAGE_NAME,
    PACKAGE_URL
)
