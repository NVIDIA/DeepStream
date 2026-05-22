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
 * <b>NVIDIA GStreamer: Custom Events</b>
 *
 * @b Description: This file specifies the NVIDIA GStreamer custom
 * event functions.
 *
 */

/**
 * @defgroup  gstreamer_nvevent  Events: Custom Events API
 *
 * Specifies GStreamer custom event functions.
 *
 * @ingroup gst_mess_evnt_qry
 * @{
 */

#ifndef __GST_NVCUSTOMEVENT_H__
#define __GST_NVCUSTOMEVENT_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLAG(name) GST_EVENT_TYPE_##name

/** Defines supported types of custom events. */
typedef enum {
  /** Specifies a custom event to indicate decoder drop frame interval update
   of a particular stream. */
  GST_NVEVENT_DEC_DROP_FRAME_INTERVAL_UPDATE
    = GST_EVENT_MAKE_TYPE (500, FLAG(DOWNSTREAM) | FLAG(SERIALIZED)),
  /** Specifies a custom event to indicate decoder skip frame update
   of a particular stream. */
  GST_NVEVENT_DEC_SKIP_FRAME_UPDATE
    = GST_EVENT_MAKE_TYPE (501, FLAG(DOWNSTREAM) | FLAG(SERIALIZED)),
  /** Specifies a custom event to enable decoder low-latency-mode
   of a particular stream. */
  GST_NVEVENT_DEC_ENABLE_LOW_LATENCY_MODE
    = GST_EVENT_MAKE_TYPE (502, FLAG(DOWNSTREAM) | FLAG(SERIALIZED)),

  /** Specifies a custom event to indicate encoder bitrate update
   of a particular stream. */
  GST_NVEVENT_ENC_BITRATE_UPDATE
    = GST_EVENT_MAKE_TYPE (503, FLAG(DOWNSTREAM) | FLAG(SERIALIZED)),
  /** Specifies a custom event to indicate encoder force IDR frame
   of a particular stream. */
  GST_NVEVENT_ENC_FORCE_IDR
    = GST_EVENT_MAKE_TYPE (504, FLAG(DOWNSTREAM) | FLAG(SERIALIZED)),
  /** Specifies a custom event to indicate encoder force Intra frame
   of a particular stream. */
  GST_NVEVENT_ENC_FORCE_INTRA
    = GST_EVENT_MAKE_TYPE (505, FLAG(DOWNSTREAM) | FLAG(SERIALIZED)),
  /** Specifies a custom event to indicate iframe interval update
   of a particular stream. */
  GST_NVEVENT_ENC_IFRAME_INTERVAL_UPDATE
    = GST_EVENT_MAKE_TYPE (506, FLAG(DOWNSTREAM) | FLAG(SERIALIZED))
} GstNvCustomEventType;
#undef FLAG

/**
 * Creates a new "nv-dec-drop-frame-interval-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which decoder-drop-frame-interval is to be sent
 * @param[out] interval     The decoder drop-frame interval obtained corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_dec_drop_frame_interval_update (gchar* stream_id, guint interval);

/**
 * Parses a "nv-dec-drop-frame-interval-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a dec-drop-frame-interval-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] interval     A pointer to the parsed interval
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_dec_drop_frame_interval_update (GstEvent * event, gchar** stream_id, guint *interval);

/**
 * Creates a new "nv-dec-skip-frame-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which decoder-skip-frame-update is to be sent
 * @param[out] frame_type   The decoder frame-type to be skipped obtained corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_dec_skip_frame_update (gchar* stream_id, guint frame_type);

/**
 * Parses a "nv-dec-skip-frame-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a skip-frame-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] frame_type   A pointer to the parsed frame_type
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_dec_skip_frame_update (GstEvent * event, gchar** stream_id, guint *frame_type);


/**
 * Creates a new "nv-dec-enable-low-latency-mode" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which decoder-low-latenct-mode  is to be sent
 * @param[out] enable       The decoder low latency mode to be enabled corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_dec_enable_low_latency_mode (gchar* stream_id, gint enable);

/**
 * Parses a "nv-dec-enable-low-latency-mode" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a enable-low-latency-mode event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] enable       A pointer to the parsed enable flag
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_dec_enable_low_latency_mode (GstEvent * event, gchar** stream_id, gint *enable);

/**
 * Creates a new "nv-enc-bitrate-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which encoder-bitrate-update is to be sent
 * @param[out] bitrate      The encoder bitrate to be set corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_enc_bitrate_update (gchar* stream_id, guint bitrate);

/**
 * Parses a "nv-enc-bitrate-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a bitrate-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] bitrate      A pointer to the parsed bitrate value
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_enc_bitrate_update (GstEvent * event, gchar** stream_id, guint *bitrate);

/**
 * Creates a new "nv-enc-force-idr" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which encoder-force-idr is to be sent
 * @param[out] force        The encoder force IDR frame corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_enc_force_idr (gchar* stream_id, gint force);

/**
 * Parses a "nv-enc-force-idr" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a force-idr event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] force        A pointer to the parsed force value
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_enc_force_idr (GstEvent * event, gchar** stream_id, gint *force);

/**
 * Creates a new "nv-enc-force-intra" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which encoder-force-intra is to be sent
 * @param[out] force        The encoder force Intra frame corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_enc_force_intra (gchar* stream_id, gint force);

/**
 * Parses a "nv-enc-force-intra" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a force-intra event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] force        A pointer to the parsed force value
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_enc_force_intra (GstEvent * event, gchar** stream_id, gint *force);

/**
 * Creates a new "nv-enc-iframeinterval-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which encoder-iframeinterval-update is to be sent
 * @param[out] interval     The encoder iframeinterval to be set corresponding to stream ID for the event.
 */
GstEvent * gst_nvevent_enc_iframeinterval_update (gchar* stream_id, guint interval);

/**
 * Parses a "nv-enc-iframeinterval-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a iframeinterval-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] bitrate      A pointer to the parsed interval value
 *                          corresponding to stream ID for the event.
 */
void gst_nvevent_parse_enc_iframeinterval_update (GstEvent * event, gchar** stream_id, guint *interval);


#ifdef __cplusplus
}
#endif

#endif

/** @} */
