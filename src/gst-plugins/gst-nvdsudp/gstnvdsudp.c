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

#include "gstnvdsudpsrc.h"
#include "gstnvdsudpsink.h"

static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvdsudpsrc", GST_RANK_PRIMARY,
      GST_TYPE_NVDSUDPSRC))
    return FALSE;

  if (!gst_element_register (plugin, "nvdsudpsink", GST_RANK_PRIMARY,
      GST_TYPE_NVDSUDPSINK))
    return FALSE;

  return TRUE;
}

#define PACKAGE "nvdsudp"
#define PACKAGE_NAME "Nvidia DeepstreamSDK UDP plugins"
#define DESCRIPTION "Transfer data via UDP using Rivermax APIs"
#define PACKAGE_ORIGIN "http://nvidia.com"

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvdsgst_udp, DESCRIPTION, plugin_init, "9.0",
    "Proprietary", PACKAGE_NAME, PACKAGE_ORIGIN)
