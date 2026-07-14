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

#include "gstnvimagedec.h"
#include "gstnvimageenc.h"

#ifndef PACKAGE
#define PACKAGE "nvimage"
#endif

#define PACKAGE_DESCRIPTION "nvidia image decoder encoder plugin"
#define PACKAGE_LICENSE     "Proprietary"
#define PACKAGE_NAME        "GStreamer nVidia Image Decoder Encoder Plugins"
#define PACKAGE_URL         "http://nvidia.com/"

#define DS_VERSION_STR_(x) #x
#define DS_VERSION_STR(x) DS_VERSION_STR_(x)

static gboolean
plugin_init (GstPlugin * plugin)
{
  /* FIXME: Keeping rank of nvimagedec/enc just below nvjpegdec/enc for now,
   * to avoid breaking any existing usecases. Increase the rank later, once
   * nvjpegdec/enc is deprecated. */
  if (!gst_element_register (plugin, "nvimagedec", GST_RANK_PRIMARY+14,
          GST_TYPE_NVIMAGE_DEC))
    return FALSE;

  if (!gst_element_register (plugin, "nvimageenc", GST_RANK_PRIMARY+9,
          GST_TYPE_NVIMAGE_ENC))
    return FALSE;

  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_image,
    PACKAGE_DESCRIPTION,
    plugin_init,
    DS_VERSION_STR("9.1"),
    PACKAGE_LICENSE,
    PACKAGE_NAME,
    PACKAGE_URL)
