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

#include "gstdeepstreambins.h"
#include "gstdsnvinferbin.h"
#include "gstdsnvinferserverbin.h"
#include "gstdsnvosdbin.h"
#include "gstdsnvdewarperbin.h"
#include "gstdsnvtilerbin.h"
#include "gstdsnvtrackerbin.h"
#include "gstdsnvanalyticsbin.h"
#include "gstdsnvvideorendererbin.h"
#include "gstdsnvvideoencfilesinkbin.h"
#include "gstdsnvrtspoutbin.h"
#include "gstdsnvmsgbrokerbin.h"
#include "gstdsnvcamerasrcbin.h"
#include "gstdsnvbuffersyncbin.h"

#include <string.h>
#include <gst/gst.h>

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
deepstream_bins_plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvinferbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVINFER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvinferserverbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVINFERSERVER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvosdbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVOSD_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvdewarperbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVDEWARPER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvtilerbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVTILER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvtrackerbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVTRACKER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvcamerasrcbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVCAMERASRC_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvanalyticsbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVANALYTICS_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvvideorenderersinkbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVVIDEORENDERER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvvideoencfilesinkbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVVIDEOENCFILESINK_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvrtspoutsinkbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVRTSPOUT_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvmsgbrokersinkbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVMSGBROKER_BIN))
    return FALSE;

  if (!gst_element_register (plugin, "nvdsbuffersyncbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVBUFFERSYNC_BIN))
    return FALSE;

  return TRUE;

}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_deepstream_bins,
    DESCRIPTION, deepstream_bins_plugin_init, DS_VERSION, LICENSE,
    BINARY_PACKAGE, URL)
