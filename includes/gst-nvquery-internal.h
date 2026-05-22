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

#ifndef __GST_NVQUERY_INT_H__
#define __GST_NVQUERY_INT_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  GstClockTime ntp_time_epoch_ns;
  GstClockTime frame_timestamp;
  GstClockTime avg_frame_time;
} _NtpData;

GstQuery * gst_nvquery_nppstream_new (void);
gboolean gst_nvquery_is_nppstream (GstQuery * query);
void gst_nvquery_nppstream_set (GstQuery * query, gpointer cudastream);
gboolean gst_nvquery_nppstream_parse (GstQuery * query, gpointer cudastreamptr);

GstQuery * gst_nvquery_resolution_new (void);
gboolean gst_nvquery_is_resolution (GstQuery * query);
void gst_nvquery_resolution_set (GstQuery * query, guint width, guint height);
gboolean gst_nvquery_resolution_parse (GstQuery * query, guint * width, guint * height);

GstQuery * gst_nvquery_stream_caps_new (guint streamId);
gboolean gst_nvquery_is_stream_caps (GstQuery * query);
void gst_nvquery_stream_caps_set (GstQuery * query, GstCaps* caps);
gboolean gst_nvquery_stream_caps_parse_streamid (GstQuery * query, guint * streamid);
gboolean gst_nvquery_stream_caps_parse (GstQuery * query, GstStructure** str);

GstQuery * gst_nvquery_num_surfaces_per_buffer_new (void);
gboolean gst_nvquery_is_num_surfaces_per_buffer (GstQuery * query);
void gst_nvquery_num_surfaces_per_buffer_set (GstQuery * query, guint num_surfaces_per_buffers);
gboolean gst_nvquery_num_surfaces_per_buffer_parse (GstQuery * query, guint * num_surfaces_per_buffers);

GstQuery * gst_nvquery_ntp_sync_new (void);
gboolean gst_nvquery_is_ntp_sync (GstQuery * query);
gboolean gst_nvquery_ntp_sync_parse (GstQuery * query, _NtpData *ntp_data);
void gst_nvquery_ntp_sync_set (GstQuery * query, _NtpData *ntp_data);

GstQuery * gst_nvquery_uri_from_streamid_new (guint streamid);
gboolean gst_nvquery_is_uri_from_streamid (GstQuery * query);
void gst_nvquery_uri_from_streamid_set (GstQuery * query, const gchar *uri);
gboolean gst_nvquery_uri_from_streamid_parse (GstQuery * query, const gchar **uri);
gboolean gst_nvquery_uri_from_streamid_parse_streamid (GstQuery * query, guint *streamid);

GstQuery * gst_nvquery_sourceid_new (void);
gboolean gst_nvquery_is_sourceid (GstQuery * query);
gboolean gst_nvquery_sourceid_parse (GstQuery * query, guint *sourceid);
void gst_nvquery_sourceid_set (GstQuery * query, guint sourceid);

#ifdef __cplusplus
}
#endif

#endif
