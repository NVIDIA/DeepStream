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

#include <gst/gst.h>
#include "gstnvinfereval.h"

GST_DEBUG_CATEGORY_EXTERN (gst_nv_infer_eval_debug);

#define PACKAGE         "nvdsgst_infereval"
#define LICENSE         "Proprietary"
#define DESCRIPTION     "NVIDIA DeepStream live inference accuracy evaluator with A/B promotion gating"
#define BINARY_PACKAGE  "NVIDIA DeepStream nvinfereval"
#define URL             "http://nvidia.com/"

static gboolean
nvinfereval_plugin_init (GstPlugin *plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nv_infer_eval_debug, "nvinfereval", 0,
      "NVIDIA DeepStream inference evaluator");

  return gst_element_register (plugin, "nvinfereval",
      GST_RANK_NONE, GST_TYPE_NVINFEREVAL);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_infereval,
    DESCRIPTION, nvinfereval_plugin_init, DS_VERSION, LICENSE,
    BINARY_PACKAGE, URL)
