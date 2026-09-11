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
#include "gstnvmodelmux.h"

/* The plugin's debug category is DEFINED in gstnvmodelmux.c and shared
 * by the bin/config translation units; we register it here, in plugin_init. */
GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);

#define PACKAGE         "nvdsgst_modelmux"
#define LICENSE         "Proprietary"
#define DESCRIPTION     "NVIDIA DeepStream multi-model per-stream inference bin"
#define BINARY_PACKAGE  "NVIDIA DeepStream nvmodelmux Bin"
#define URL             "http://nvidia.com/"

/* Boilerplate: register the debug category and the element factory. */
static gboolean
nvmodelmux_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_modelmux_debug_cat, "nvmodelmux", 0,
      "NVIDIA DeepStream multi-model per-stream inference bin");

  return gst_element_register (plugin, "nvmodelmux",
      GST_RANK_PRIMARY, GST_TYPE_NVMODELMUX);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_modelmux,
    DESCRIPTION, nvmodelmux_plugin_init, DS_VERSION, LICENSE,
    BINARY_PACKAGE, URL)
