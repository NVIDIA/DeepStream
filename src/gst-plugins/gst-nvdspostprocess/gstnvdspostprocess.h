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
