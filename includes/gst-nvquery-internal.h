/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
