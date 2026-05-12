/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
