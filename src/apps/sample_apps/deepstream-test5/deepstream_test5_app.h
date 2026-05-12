/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __DEEPSTREAM_TEST5_APP_H__
#define __DEEPSTREAM_TEST5_APP_H__

#include <gst/gst.h>
#include "deepstream_config.h"

typedef struct
{
  gint anomaly_count;
  gint meta_number;
  struct timespec timespec_first_frame;
  GstClockTime gst_ts_first_frame;
  GMutex lock_stream_rtcp_sr;
  guint32 id;
  gint frameCount;
  GstClockTime last_ntp_time;
} StreamSourceInfo;

typedef struct
{
  StreamSourceInfo streams[MAX_SOURCE_BINS];
} TestAppCtx;

struct timespec extract_utc_from_uri (gchar * uri);

#endif /**< __DEEPSTREAM_TEST5_APP_H__ */
