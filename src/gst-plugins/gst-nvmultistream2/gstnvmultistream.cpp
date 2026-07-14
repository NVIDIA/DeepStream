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

gboolean plugin_init (GstPlugin *plugin);

static gboolean
plugin_init_2 (GstPlugin * plugin)
{
  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

#ifndef ENABLE_GST_NVSTREAMMUX_UNIT_TESTS
  if (!use_new_mux) {
    return plugin_init (plugin);
  } else
#endif
  {
  if (!gst_element_register (plugin, "nvstreammux", GST_RANK_PRIMARY,
          GST_TYPE_NVSTREAMMUX))
    return FALSE;

  if (!gst_element_register (plugin, "nvstreamdemux", GST_RANK_PRIMARY,
          GST_TYPE_NVSTREAMDEMUX))
    return FALSE;
   }

  return TRUE;
}

#if 0
/** NOTE: Disabling all static Gst APIs for loading streammux2
 * based on ENV var: USE_NEW_NVSTREAMMUX
 * TODO: Revert the change that disabled these static Gst APIs
 * when we are ready to drop legacy muxer
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistream,
    DESCRIPTION, plugin_init, "9.1", LICENSE, BINARY_PACKAGE, URL)
#endif

#ifdef ENABLE_GST_NVSTREAMMUX_UNIT_TESTS
extern "C" gboolean gGstNvMultistream2StaticInit();
gboolean gGstNvMultistream2StaticInit()
{
  return gst_plugin_register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                 "nvdsgst_multistream",
                 DESCRIPTION, plugin_init_2, "9.1", LICENSE, BINARY_PACKAGE, PACKAGE, URL);
}
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistream,
    DESCRIPTION, plugin_init_2, "9.1", LICENSE, BINARY_PACKAGE, URL)

