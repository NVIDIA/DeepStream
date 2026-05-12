/*
 * Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
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
