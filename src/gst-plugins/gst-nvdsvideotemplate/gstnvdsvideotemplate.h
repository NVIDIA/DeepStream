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

#ifndef __GST_NVDSVIDEOTEMPLATE_H__
#define __GST_NVDSVIDEOTEMPLATE_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <glib-object.h>
#include <vector>

#include <cuda_runtime.h>
#include "gstnvdsmeta.h"
#include "nvtx3/nvToolsExt.h"

#include "nvdscustomlib_factory.hpp"
#include "nvdscustomlib_interface.hpp"
#include "nvbufsurftransform.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsvideotemplate"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA nvdsvideotemplate plugin for integration with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "NVIDIA DeepStream Template Plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstNvDsVideoTemplate GstNvDsVideoTemplate;
typedef struct _GstNvDsVideoTemplateClass GstNvDsVideoTemplateClass;

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSVIDEOTEMPLATE (gst_nvdsvideotemplate_get_type())
#define GST_NVDSVIDEOTEMPLATE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSVIDEOTEMPLATE,GstNvDsVideoTemplate))
#define GST_NVDSVIDEOTEMPLATE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSVIDEOTEMPLATE,GstNvDsVideoTemplateClass))
#define GST_NVDSVIDEOTEMPLATE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSVIDEOTEMPLATE, GstNvDsVideoTemplateClass))
#define GST_IS_NVDSVIDEOTEMPLATE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSVIDEOTEMPLATE))
#define GST_IS_NVDSVIDEOTEMPLATE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSVIDEOTEMPLATE))
#define GST_NVDSVIDEOTEMPLATE_CAST(obj)  ((GstNvDsVideoTemplate *)(obj))

struct _GstNvDsVideoTemplate
{
  GstBaseTransform base_trans;

  /** Custom Library Factory and Interface */
  DSCustomLibrary_Factory *algo_factory;
  IDSCustomLibrary *algo_ctx;

  /** Custom Library Name and output caps string */
  gchar* custom_lib_name;

  /* Store custom lib property values */
  std::vector<Property> *vecProp;
  gchar *custom_prop_string;

  /** Boolean to signal output thread to stop. */
  gboolean stop;

  /** Input and Output video info (resolution, color format, framerate, etc) */
  GstVideoInfo in_video_info;
  GstVideoInfo out_video_info;

  /** GPU ID on which we expect to execute the task */
  guint gpu_id;

  gboolean dummy_meta_insert;

  gboolean fill_dummy_batch_meta;

  /** NVTX Domain. */
  nvtxDomainHandle_t nvtx_domain;

  GstCaps *sinkcaps;
  GstCaps *srccaps;
  guint num_batch_buffers;
  NvBufSurfTransformConfigParams config_params;
  gint compute_hw;
  cudaStream_t cu_nbstream;

  // NVIDIA Video Memory
  void *inter_buf;

  // OpenCV related
  void *cvmat;

  // Config file path
  gchar *config_file_path;

  // Input caps width and height
  gint in_width;
  gint in_height;

  // Output caps width and height
  gint out_width;
  gint out_height;
};


/** Boiler plate stuff */
struct _GstNvDsVideoTemplateClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdsvideotemplate_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSVIDEOTEMPLATE_H__ */
