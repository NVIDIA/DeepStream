/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
