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

#ifndef __GST_NVDSPOSTPROCESS_H__
#define __GST_NVDSPOSTPROCESS_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <glib-object.h>
#include <vector>

#include <cuda_runtime.h>
#include "gstnvdsmeta.h"
#include "nvtx3/nvToolsExt.h"

#include "nvdspostprocesslib_factory.hpp"
#include "nvdspostprocesslib_interface.hpp"
#include "nvbufsurftransform.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvdspostprocess"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA nvdspostprocess plugin for parsing inference output from nvdsinfer/nvdsinferserver with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "NVIDIA DeepStream Post Processing Plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstNvDsPostProcess GstNvDsPostProcess;
typedef struct _GstNvDsPostProcessClass GstNvDsPostProcessClass;

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSPOSTPROCESS (gst_nvdspostprocess_get_type())
#define GST_NVDSPOSTPROCESS(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSPOSTPROCESS,GstNvDsPostProcess))
#define GST_NVDSPOSTPROCESS_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSPOSTPROCESS,GstNvDsPostProcessClass))
#define GST_NVDSPOSTPROCESS_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSPOSTPROCESS, GstNvDsPostProcessClass))
#define GST_IS_NVDSPOSTPROCESS(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSPOSTPROCESS))
#define GST_IS_NVDSPOSTPROCESS_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSPOSTPROCESS))
#define GST_NVDSPOSTPROCESS_CAST(obj)  ((GstNvDsPostProcess *)(obj))

struct _GstNvDsPostProcess
{
  GstBaseTransform base_trans;

  /** Custom Library Factory and Interface */
  DSPostProcessLibrary_Factory *algo_factory;
  IDSPostProcessLibrary *algo_ctx;

  /** Custom Library Name and output caps string */
  gchar* postprocess_lib_name;

  /** Custom Library config file path */
  gchar* postprocess_lib_config_file;

  /* Store postprocess lib property values */
  std::vector<Property> *vecProp;
  gchar *postprocess_prop_string;

  /** Boolean to signal output thread to stop. */
  gboolean stop;

  /** Input and Output video info (resolution, color format, framerate, etc) */
  GstVideoInfo in_video_info;
  GstVideoInfo out_video_info;

  /** GPU ID on which we expect to execute the task */
  guint gpu_id;

  /** NVTX Domain. */
  nvtxDomainHandle_t nvtx_domain;

  GstCaps *sinkcaps;
  GstCaps *srccaps;
  NvBufSurfTransformConfigParams config_params;
  gint compute_hw;
  cudaStream_t cu_nbstream;
  /** Boolean to indicate whether postprocessor support for preprocesor should be enabled */
  gboolean preprocessor_support;
};


/** Boiler plate stuff */
struct _GstNvDsPostProcessClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdspostprocess_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSPOSTPROCESS_H__ */
