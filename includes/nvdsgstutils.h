/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>Defines NVIDIA DeepStream GStreamer Utilities</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer utility
 * functions.
 *
 */
/**
 * @defgroup  gstreamer_utils  Utilities: Gstreamer utilities API
 *
 * Specifies GStreamer utilities functions, used to configure the source to generate NTP Sync values.
 *
 * @ingroup NvDsUtilsApi
 * @{
 */
#ifndef __NVDS_GSTUTILS_H__
#define __NVDS_GSTUTILS_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <gst/gst.h>

/**
 * Configure the source to generate NTP sync values for RTSP sources.
 *
 * These values are used by the DeepStream GStreamer element NvStreamMux to
 * calculate the NTP time of the frames at the source.
 *
 * This functionality is dependent on the RTSP sending the RTCP Sender Reports.
 * source.
 *
 * This function only works for RTSP sources i.e. GStreamer elements "rtspsrc"
 * or "uridecodebin" with an RTSP uri.
 *
 * params[in] src_elem GStreamer source element to be configured.
 */
void configure_source_for_ntp_sync (GstElement *src_elem);

#ifdef __cplusplus
}
#endif

#endif

/** @} */
