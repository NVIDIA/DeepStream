/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
