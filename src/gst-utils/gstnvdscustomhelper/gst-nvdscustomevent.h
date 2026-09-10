/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA GStreamer DeepStream: Custom Events</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer custom
 * event functions, used to map events to individual sources which
 * are batched together by Gst-nvstreammux.
 *
 */

/**
 * @defgroup  gstreamer_nvevent  Events: Custom Events API
 *
 * Specifies GStreamer custom event functions, used to map events
 * to individual sources which are batched together by Gst-nvstreammux.
 *
 * @ingroup gst_mess_evnt_qry
 * @{
 */

#ifndef __GST_NVDSCUSTOMEVENT_H__
#define __GST_NVDSCUSTOMEVENT_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif


/** Defines Roi structure for GST_NVCUSTOMEVENT_ROI_UPDATE custom event */
  typedef struct RoiDimension
  {
    gchar roi_id[128];
    guint left;
    guint top;
    guint width;
    guint height;
  } RoiDimension;

#define FLAG(name) GST_EVENT_TYPE_##name

/** Defines supported types of custom events. */
  typedef enum
  {
  /** Specifies a custom event to indicate ROI update for preprocess
   of a particular stream in a batch. */
    GST_NVEVENT_ROI_UPDATE
        = GST_EVENT_MAKE_TYPE (406, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event to indicate infer interval update
   of a particular stream in a batch. */
    GST_NVEVENT_INFER_INTERVAL_UPDATE
        = GST_EVENT_MAKE_TYPE (407, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event to indicate osd process mode update
   of a particular stream in a batch. */
    GST_NVEVENT_OSD_PROCESS_MODE_UPDATE
        = GST_EVENT_MAKE_TYPE (408, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event to indicate analytics reload_config update
   of a particular stream in a batch. */
    GST_NVEVENT_ANALYTICS_RELOAD_CONFIG_UPDATE
        = GST_EVENT_MAKE_TYPE (409, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),

  /** Specifies a custom event to indicate nvTracker config update
   for all streams . */
    GST_NVEVENT_NVTRACKER_CONFIG_UPDATE
        = GST_EVENT_MAKE_TYPE (410, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),

  /** Specifies a custom event carrying a REST model/load request to a
   downstream model-managing element (e.g. nvmodelmux). Delivered
   in-band on the data plane so it reaches the element in ANY host app,
   independent of who owns the pipeline bus. Payload = the request "value"
   object as verbatim JSON (one schema end-to-end; see NvDsModelInfo). */
    GST_NVEVENT_MODEL_LOAD
        = GST_EVENT_MAKE_TYPE (411, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event carrying a REST model/unload request (verbatim
   JSON payload). */
    GST_NVEVENT_MODEL_UNLOAD
        = GST_EVENT_MAKE_TYPE (412, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event carrying a REST model/update request: an
   IN-PLACE checkpoint transition on the live instance group (same network,
   new weights, NEW version). Model plane only -- never routes streams
   (routing is GST_NVEVENT_STREAM_ROUTE). Verbatim JSON payload. */
    GST_NVEVENT_MODEL_UPDATE
        = GST_EVENT_MAKE_TYPE (413, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event carrying a REST stream/add per-stream model
   binding (source_id + camera_id + raw metadata JSON) to a downstream
   model-managing element, so per-stream model selection works in-band even
   when the host owns the pipeline bus. */
    GST_NVEVENT_STREAM_MODEL_BIND
        = GST_EVENT_MAKE_TYPE (414, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event carrying a REST stream/route request: the
   declarative routing document (routes[] + default{}), with each route
   AUGMENTED by nvmultiurisrcbin with the resolved source_ids. Stream plane
   only -- never loads or destroys models. Verbatim JSON payload. */
    GST_NVEVENT_STREAM_ROUTE
        = GST_EVENT_MAKE_TYPE (415, FLAG (DOWNSTREAM) | FLAG (SERIALIZED))
  } GstNvDsCustomEventType;
#undef FLAG

/**
 * Creates a new "roi-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which nv-roi-update is to be sent
 * @param[out] roi_count    The roi_count obtained corresponding to stream ID for the event.
 * @param[out] roi_dim      The RoiDimension structure of size roi_count.
 */
  GstEvent *gst_nvevent_new_roi_update (gchar * stream_id, guint roi_count,
      RoiDimension * roi_dim);

/**
 * Parses a "roi-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a nv-roi-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] roi_count    A pointer to the parsed number of roi(s)
 *                          corresponding to stream ID for the event.
 * @param[out] roi_dim      A double pointer to the parsed RoiDimension structure of size roi_count.
 *                          User MUST free roi_dim memory using g_free post usage.
 */
  void gst_nvevent_parse_roi_update (GstEvent * event, gchar ** stream_id,
      guint * roi_count, RoiDimension ** roi_dim);

/**
 * Creates a new "nv-infer-interval-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which infer-interval-update is to be sent
 * @param[out] interval     The infer interval obtained corresponding to stream ID for the event.
 */
  GstEvent *gst_nvevent_infer_interval_update (gchar * stream_id,
      guint interval);

/**
 * Parses a "nv-infer-interval-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a infer-interval-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] interval     A pointer to the parsed interval
 *                          corresponding to stream ID for the event.
 */
  void gst_nvevent_parse_infer_interval_update (GstEvent * event,
      gchar ** stream_id, guint * interval);


/**
 * Creates a new "nv-tracker-config-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which infer-interval-update is to be sent
 * @param[out] configStr    A reference to the parsed char string in the tracker plugin
 */
  GstEvent *gst_nvevent_nvtracker_config_update (gchar * stream_id, gchar * configStr);

/**
 * Parses a "nv-tracker-config-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a infer-interval-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] configStr    A reference to the parsed char string
 *                          in the tracker plugin
 */
  void gst_nvevent_parse_nvtracker_config_update (GstEvent * event,
      gchar ** stream_id, gchar ** configStr);


/**
 * Creates a new "nv-osd-process-mode-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which osd-process-mode-update is to be sent
 * @param[out] process_mode The infer interval obtained corresponding to stream ID for the event.
 */
  GstEvent *gst_nvevent_osd_process_mode_update (gchar * stream_id,
      guint process_mode);

/**
 * Parses a "nv-osd-process-mode-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a osd-process-mode-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] process_mode A pointer to the parsed interval
 *                          corresponding to stream ID for the event.
 */
  void gst_nvevent_parse_osd_process_mode_update (GstEvent * event,
      gchar ** stream_id, guint * process_mode);
/**
 * Creates a new "nv-analytics-reload_config-update" event.
 *
 * @param[out] config_file_path   The config file path
 */
GstEvent *gst_nvevent_analytics_reload_config_update (gchar * config_file_path);

/**
 * Parses a "nv-analytics-reload_config-update" event.
 *
 * @param[in] event         The event received on the sinkpad
 * @param[out] config_file_path   A pointer to the parsed config file path
 */
void gst_nvevent_parse_analytics_reload_config_update (GstEvent * event,
    gchar ** config_file_path);

/**
 * Model-plane + stream-routing control events carry ONE payload: the request
 * "value" object as verbatim JSON (the same document a REST client POSTs and
 * an action-signal caller passes). Positional argument lists cannot grow with
 * the schema (engine_files{}, instances[], nested refs), so they do not exist
 * here: one string in, one string out, one schema end-to-end.
 */

/** Creates a new "nv-model-load" event. @value_json = the model/load request
 * "value" object, serialized JSON (see NvDsModelInfo). Must be non-NULL. */
GstEvent *gst_nvevent_new_model_load (const gchar * value_json);

/** Parses a "nv-model-load" event. *value_json is newly allocated (caller
 * frees with g_free) or NULL when absent. */
void gst_nvevent_parse_model_load (GstEvent * event, gchar ** value_json);

/** Creates a new "nv-model-unload" event (verbatim JSON payload). */
GstEvent *gst_nvevent_new_model_unload (const gchar * value_json);

/** Parses a "nv-model-unload" event. Caller frees *value_json. */
void gst_nvevent_parse_model_unload (GstEvent * event, gchar ** value_json);

/** Creates a new "nv-model-update" event: IN-PLACE checkpoint transition
 * (model plane only -- never routes streams). Verbatim JSON payload. */
GstEvent *gst_nvevent_new_model_update (const gchar * value_json);

/** Parses a "nv-model-update" event. Caller frees *value_json. */
void gst_nvevent_parse_model_update (GstEvent * event, gchar ** value_json);

/** Creates a new "nv-stream-route" event: the declarative routing document
 * (routes[] + default{}), each route augmented with resolved source_ids by
 * nvmultiurisrcbin. Verbatim JSON payload. */
GstEvent *gst_nvevent_new_stream_route (const gchar * value_json);

/** Parses a "nv-stream-route" event. Caller frees *value_json. */
void gst_nvevent_parse_stream_route (GstEvent * event, gchar ** value_json);

/**
 * Creates a new "nv-stream-model-bind" event carrying a stream/add's per-stream
 * binding: the assigned source_id, camera_id, camera_name (display name) and the
 * raw metadata JSON (which may name primary/shadow models). Any string may be
 * NULL/"".
 */
GstEvent *gst_nvevent_new_stream_model_bind (guint source_id,
    const gchar * camera_id, const gchar * camera_name,
    const gchar * metadata_json);

/** Parses a "nv-stream-model-bind" event. Caller frees each out string. */
void gst_nvevent_parse_stream_model_bind (GstEvent * event, guint * source_id,
    gchar ** camera_id, gchar ** camera_name, gchar ** metadata_json);

#ifdef __cplusplus
}
#endif

#endif

/** @} */
