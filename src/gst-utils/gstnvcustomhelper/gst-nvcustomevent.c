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

#include "gst-nvcustomevent.h"

GstEvent *
gst_nvevent_dec_drop_frame_interval_update (gchar* stream_id, guint interval)
{
  GstStructure *str = gst_structure_new_empty ("nv-dec-drop-frame-interval-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "interval", G_TYPE_UINT, interval,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_DEC_DROP_FRAME_INTERVAL_UPDATE, str);
}


void
gst_nvevent_parse_dec_drop_frame_interval_update (GstEvent * event, gchar** stream_id, guint *interval)
{
  if((GstEventType) GST_NVEVENT_DEC_DROP_FRAME_INTERVAL_UPDATE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "interval", G_TYPE_UINT, interval,
                      NULL);
  }
}

GstEvent *
gst_nvevent_dec_skip_frame_update (gchar* stream_id, guint frame_type)
{
  GstStructure *str = gst_structure_new_empty ("nv-dec-skip-frame-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "frame_type", G_TYPE_UINT, frame_type,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_DEC_SKIP_FRAME_UPDATE, str);
}


void
gst_nvevent_parse_dec_skip_frame_update (GstEvent * event, gchar** stream_id, guint *frame_type)
{
  if((GstEventType) GST_NVEVENT_DEC_SKIP_FRAME_UPDATE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "frame_type", G_TYPE_UINT, frame_type,
                      NULL);
  }
}

GstEvent *
gst_nvevent_dec_enable_low_latency_mode (gchar* stream_id, gint enable)
{
  GstStructure *str = gst_structure_new_empty ("nv-dec-enable-low-latency-mode");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "enable", G_TYPE_INT, enable,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_DEC_ENABLE_LOW_LATENCY_MODE, str);
}


void
gst_nvevent_parse_dec_enable_low_latency_mode (GstEvent * event, gchar** stream_id, gint *enable)
{
  if((GstEventType) GST_NVEVENT_DEC_ENABLE_LOW_LATENCY_MODE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "enable", G_TYPE_INT, enable,
                      NULL);
  }
}

GstEvent *
gst_nvevent_enc_bitrate_update (gchar* stream_id, guint bitrate)
{
  GstStructure *str = gst_structure_new_empty ("nv-enc-bitrate-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "bitrate", G_TYPE_UINT, bitrate,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_ENC_BITRATE_UPDATE, str);
}


void
gst_nvevent_parse_enc_bitrate_update (GstEvent * event, gchar** stream_id, guint *bitrate)
{
  if((GstEventType) GST_NVEVENT_ENC_BITRATE_UPDATE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "bitrate", G_TYPE_UINT, bitrate,
                      NULL);
  }
}

GstEvent *
gst_nvevent_enc_force_idr (gchar* stream_id, gint force)
{
  GstStructure *str = gst_structure_new_empty ("nv-enc-force-idr");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "force", G_TYPE_INT, force,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_ENC_FORCE_IDR, str);
}

void
gst_nvevent_parse_enc_force_idr (GstEvent * event, gchar** stream_id, gint *force)
{
  if((GstEventType) GST_NVEVENT_ENC_FORCE_IDR == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "force", G_TYPE_INT, force,
                      NULL);
  }
}

GstEvent *
gst_nvevent_enc_force_intra (gchar* stream_id, gint force)
{
  GstStructure *str = gst_structure_new_empty ("nv-enc-force-intra");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "force", G_TYPE_INT, force,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_ENC_FORCE_INTRA, str);
}

void
gst_nvevent_parse_enc_force_intra (GstEvent * event, gchar** stream_id, gint *force)
{
  if((GstEventType) GST_NVEVENT_ENC_FORCE_INTRA == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "force", G_TYPE_INT, force,
                      NULL);
  }
}

GstEvent *
gst_nvevent_enc_iframeinterval_update (gchar* stream_id, guint interval)
{
  GstStructure *str = gst_structure_new_empty ("nv-enc-iframeinterval-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
                    "interval", G_TYPE_UINT, interval,
                    NULL);

  return gst_event_new_custom (GST_NVEVENT_ENC_IFRAME_INTERVAL_UPDATE, str);
}


void
gst_nvevent_parse_enc_iframeinterval_update (GstEvent * event, gchar** stream_id, guint *interval)
{
  if((GstEventType) GST_NVEVENT_ENC_IFRAME_INTERVAL_UPDATE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
                      "interval", G_TYPE_UINT, interval,
                      NULL);
  }
}
