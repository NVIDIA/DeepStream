/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PACKAGE
#define PACKAGE "nvmultistream"
#endif

#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA Multistream mux/demux plugin"
#define BINARY_PACKAGE "NVIDIA Multistream Plugins"
#define URL "http://nvidia.com/"

#include "gstnvstreammux.h"
#include "gstnvstreamdemux.h"

gboolean plugin_init (GstPlugin * plugin);

gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvstreammux", GST_RANK_PRIMARY,
          GST_TYPE_NVSTREAMMUX))
    return FALSE;

  if (!gst_element_register (plugin, "nvstreamdemux", GST_RANK_PRIMARY,
          GST_TYPE_NVSTREAMDEMUX))
    return FALSE;

  return TRUE;
}
#if 0
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistream,
    DESCRIPTION, plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE, URL)
#endif
