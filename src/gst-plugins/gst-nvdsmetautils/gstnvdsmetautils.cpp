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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>

#include "gstnvdsmetainsert.h"
#include "gstnvdsmetaextract.h"

#define PACKAGE_LICENSE     "Proprietary"
#define PACKAGE_NAME        "GStreamer NV DS META Data Processor Plugins"
#define PACKAGE_URL         "http://nvidia.com/"
#define PACKAGE_DESCRIPTION "DS Elements for META insertion & extraction"

#ifndef PACKAGE
#define PACKAGE "nvdsmetautils"
#endif

static gboolean plugin_init (GstPlugin * plugin)
{
    gboolean ret = TRUE;
    nvds_metainsert_init (plugin);
    nvds_metaextract_init (plugin);
    return ret;
}


GST_PLUGIN_DEFINE (
        GST_VERSION_MAJOR,
        GST_VERSION_MINOR,
        nvdsgst_metautils,
        PACKAGE_DESCRIPTION,
        plugin_init,
        "9.0",
        PACKAGE_LICENSE,
        PACKAGE_NAME,
        PACKAGE_URL)
