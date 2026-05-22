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
#include "gstdsnvmultiurisrcbin.h"

/* Package and library details required for plugin_init */
#define PACKAGE "DeepStream SDK nvmultiurisrcbin Bin"
#define LICENSE "Proprietary"
#define DESCRIPTION "Deepstream SDK nvmultiurisrcbin Bin"
#define BINARY_PACKAGE "Deepstream SDK nvmultiurisrcbin Bin"
#define URL "http://nvidia.com/"

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvmultiurisrcbin_plugin_init (GstPlugin * plugin)
{

  if (!gst_element_register (plugin, "nvmultiurisrcbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVMULTIURISRC_BIN))
    return FALSE;

  return TRUE;

}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_nvmultiurisrcbin,
    DESCRIPTION, nvmultiurisrcbin_plugin_init, "9.0", LICENSE,
    BINARY_PACKAGE, URL)
