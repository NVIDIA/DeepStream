/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "gstucxserversink.h"
#include "gstucxclientsrc.h"
#include "gstucxclientsink.h"
#include "gstucxserversrc.h"

GST_DEBUG_CATEGORY (ucx_debug);

static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvdsucxserversink", GST_RANK_NONE,
          GST_TYPE_UCX_SERVER_SINK))
    return FALSE;

  if (!gst_element_register (plugin, "nvdsucxclientsrc", GST_RANK_NONE,
          GST_TYPE_UCX_CLIENT_SRC))
    return FALSE;
  if (!gst_element_register (plugin, "nvdsucxclientsink", GST_RANK_NONE,
          GST_TYPE_UCX_CLIENT_SINK))
    return FALSE;

  if (!gst_element_register (plugin, "nvdsucxserversrc", GST_RANK_NONE,
          GST_TYPE_UCX_SERVER_SRC))
    return FALSE;

  GST_DEBUG_CATEGORY_INIT (ucx_debug, "ucx", 0, "UCX calls");

  return TRUE;
}

#ifndef VERSION
#define VERSION DS_VERSION
#endif
#ifndef PACKAGE
#define PACKAGE "nvucx"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Nvidia DeepStreamSDK UCX plugins"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://nvidia.com/"
#endif
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_ucx,
    "NVIDIA UCX sink/source plugin",
    plugin_init, VERSION, "Proprietary", PACKAGE_NAME, GST_PACKAGE_ORIGIN)
