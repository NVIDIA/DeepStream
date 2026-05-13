/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
 * TODO: Revert https://git-master.nvidia.com/r/#/c/2127642/
 * when we are ready to drop legacy muxer
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistream,
    DESCRIPTION, plugin_init, "9.0", LICENSE, BINARY_PACKAGE, URL)
#endif

#ifdef ENABLE_GST_NVSTREAMMUX_UNIT_TESTS
extern "C" gboolean gGstNvMultistream2StaticInit();
gboolean gGstNvMultistream2StaticInit()
{
  return gst_plugin_register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                 "nvdsgst_multistream",
                 DESCRIPTION, plugin_init_2, "9.0", LICENSE, BINARY_PACKAGE, PACKAGE, URL);
}
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistream,
    DESCRIPTION, plugin_init_2, "9.0", LICENSE, BINARY_PACKAGE, URL)

