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

#ifndef __GSTNVSTREAMMUX_NTP__
#define __GSTNVSTREAMMUX_NTP__

#include <glib.h>
#include <gst/gst.h>

G_BEGIN_DECLS

#define NVDS_RFC3339_STR_BUF_LEN 32

void generate_rfc3339_str_from_ts (gchar *buf, GstClockTime ts);

typedef enum
{
  GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME,
  GST_NVDS_NTP_CALC_MODE_RTCP
} GstNvDsNtpCalculatorMode;

/* Modes of NTP timestamp correction based on frame rate */
typedef enum
{
  /* Frame rate based correction disabled */
  GST_NVDS_NTP_CORRECTION_DISABLED,
  /* Use the average frame rate provided in the NTP query to correct
   * NTP timestamp. (PTS based frame rate calculation at rtpjitterbuffer.) */
  GST_NVDS_NTP_CORRECTION_AUTOMATIC,
  /* Use the frame rate provided by application to correct NTP timestamp.
   * This mode is required frame rate is different at rtpjitterbuffer and
   * nvstreammux.  Bug 3620472, 3626628  */
  GST_NVDS_NTP_CORRECTION_USER_FRAME_RATE
} GstNvDsNtpCorrectionMode;

typedef struct _GstNvDsNtpCalculator GstNvDsNtpCalculator;

GstNvDsNtpCalculator * gst_nvds_ntp_calculator_new (GstNvDsNtpCalculatorMode mode,
        GstClockTime frame_duration, GstElement *elem, guint source_id);

void gst_nvds_ntp_calculator_add_ntp_sync_values (GstNvDsNtpCalculator *calc,
        GstClockTime ntp_time_epoch_ns, GstClockTime ntp_frame_timestamp,
        GstClockTime avg_frame_time);

gboolean gst_nvds_ntp_calculator_have_ntp_sync_values (GstNvDsNtpCalculator *calc);

void gst_nvds_ntp_calculator_get_ntp_sync_values (GstNvDsNtpCalculator *calc,
        GstClockTime *ntp_time_epoch_ns, GstClockTime *ntp_frame_timestamp,
        GstClockTime *avg_frame_time, GstClockTime *ntp_time_epoch_ns_next,
        GstClockTime *ntp_frame_timestamp_next);

GstClockTime gst_nvds_ntp_calculator_get_buffer_ntp (GstNvDsNtpCalculator *calc,
        GstClockTime buf_pts);

void gst_nvds_ntp_calculator_reset (GstNvDsNtpCalculator *calc);

void gst_nvds_ntp_calculator_free (GstNvDsNtpCalculator *calc);

G_END_DECLS

#endif
