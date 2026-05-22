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
