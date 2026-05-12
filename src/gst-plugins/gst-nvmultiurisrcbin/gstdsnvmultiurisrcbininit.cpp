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
