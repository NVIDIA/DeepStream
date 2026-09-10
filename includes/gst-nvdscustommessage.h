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
 * <b>NVIDIA GStreamer DeepStream: Custom Message Functions</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer custom
 * message functions.
 *
 */
 /**
 * @defgroup gst_mess_evnt_qry Events, Messages and Query based APIs
 *
 * Defines Events, Messages and Query-based APIs
 *
 */

#ifndef __GST_NVDSCUSTOMMESSAGE_H__
#define __GST_NVDSCUSTOMMESSAGE_H__

#include <gst/gst.h>
#include "gst-nvdscommonconfig.h"
#include "deepstream_perf.h"
G_BEGIN_DECLS
/**
 * Creates a new Stream ADD message - denoting a new stream getting added to
 * nvmultiurisrcbin.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] sensor_info   Sensor info of the stream which is getting added
 *                          into nvmultiurisrcbin
 *
 * @return  A pointer to the new message.
 */
    GstMessage * gst_nvmessage_new_stream_add (GstObject * obj,
    NvDsSensorInfo * sensor_info);

/**
 * Determines whether a message is a stream ADD message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a stream ADD message.
 */
gboolean gst_nvmessage_is_stream_add (GstMessage * message);

/**
 * \brief  Parses the stream ID from a stream ADD message.
 *
 * The stream ID is the index of the stream which is getting added
 * to nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream ADD message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_stream_add (GstMessage * message,
    NvDsSensorInfo * sensor_info);

/**
 * Creates a new Stream REMOVE message - denoting a new stream getting removed
 * from nvmultiurisrcbin.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] sensor_info   Sensor info of the stream which is getting removed
 *                          from nvmultiurisrcbin
 *
 * @return  A pointer to the new message.
 */
GstMessage *gst_nvmessage_new_stream_remove (GstObject * obj,
    NvDsSensorInfo * sensor_info);

/**
 * Determines whether a message is a stream REMOVE message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a stream REMOVE message.
 */
gboolean gst_nvmessage_is_stream_remove (GstMessage * message);

/**
 * \brief  Parses the stream ID from a stream REMOVE message.
 *
 * The stream ID is the index of the stream which is getting removed
 * from nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream REMOVE message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_stream_remove (GstMessage * message,
    NvDsSensorInfo * sensor_info);

/**
 * \brief  Parses the stream ID from a stream ADD message.
 *
 * The stream ID is the index of the stream which is getting added
 * to nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream ADD message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsFPSSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */

gboolean gst_nvmessage_parse_fps_stream_add (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info);

/**
 * \brief  Parses the stream ID from a stream REMOVE message.
 *
 * The stream ID is the index of the stream which is getting removed
 * from nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream REMOVE message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsFPSSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */

gboolean gst_nvmessage_parse_fps_stream_remove (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info);

/* ---- model plane (native REST /api/v1/model/load|unload|update) ---- */

/**
 * Creates a new model-LOAD message - denoting a model getting loaded into
 * nvmodelmux.
 *
 * The message carries the request "value" object as verbatim JSON, so one
 * schema is used end-to-end and the consumer parses the document.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] model_info    A pointer to @ref NvDsModelInfo holding value_json.
 *
 * @return  A pointer to the new message.
 */
GstMessage * gst_nvmessage_new_model_load (GstObject * obj,
    NvDsModelInfo * model_info);

/**
 * Determines whether a message is a model-LOAD message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a model-LOAD message.
 */
gboolean gst_nvmessage_is_model_load (GstMessage * message);

/**
 * \brief  Parses the model info from a model-LOAD message.
 *
 * params[in] message        A pointer to a model-LOAD message.
 * params[out] model_info    A pointer to @ref NvDsModelInfo
 *    The string NvDsModelInfo->value_json should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_model_load (GstMessage * message,
    NvDsModelInfo * model_info);

/**
 * Creates a new model-UNLOAD message - denoting a model getting unloaded from
 * nvmodelmux.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] model_info    A pointer to @ref NvDsModelInfo holding value_json.
 *
 * @return  A pointer to the new message.
 */
GstMessage * gst_nvmessage_new_model_unload (GstObject * obj,
    NvDsModelInfo * model_info);

/**
 * Determines whether a message is a model-UNLOAD message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a model-UNLOAD message.
 */
gboolean gst_nvmessage_is_model_unload (GstMessage * message);

/**
 * \brief  Parses the model info from a model-UNLOAD message.
 *
 * params[in] message        A pointer to a model-UNLOAD message.
 * params[out] model_info    A pointer to @ref NvDsModelInfo
 *    The string NvDsModelInfo->value_json should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_model_unload (GstMessage * message,
    NvDsModelInfo * model_info);

/* ---- model-plane COMPLETION event (nvmodelmux -> app bus) ---- */

/**
 * Creates a new model-event message - denoting an asynchronous model-plane
 * operation that has actually completed, such as an engine warmed, a version
 * torn down, an OTA committed or a reroute settled.
 *
 * This is the stream ADD/REMOVE notification's counterpart for the model
 * plane: an app subscribes on the bus and reacts in the same way.
 *
 * params[in] obj     The GStreamer object creating the message.
 * params[in] info    A pointer to @ref NvDsModelEventInfo describing the
 *                    operation which completed.
 *
 * @return  A pointer to the new message.
 */
GstMessage * gst_nvmessage_new_model_event (GstObject * obj,
    NvDsModelEventInfo * info);

/**
 * Determines whether a message is a model-event message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a model-event message.
 */
gboolean gst_nvmessage_is_model_event (GstMessage * message);

/**
 * \brief  Parses the event info from a model-event message.
 *
 * params[in] message   A pointer to a model-event message.
 * params[out] info     A pointer to @ref NvDsModelEventInfo
 *    The strings in NvDsModelEventInfo should not be modified,
 *    and remain valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_model_event (GstMessage * message,
    NvDsModelEventInfo * info);

/* ---- model status (native REST GET /api/v1/model/status) ---- */

/**
 * Creates a new empty (unfiltered) model-status query, to be sent downstream
 * via peer_query.
 *
 * The app answers with a JSON status string at its bin boundary, which
 * decouples the in-plugin REST thread from the app-owned model state.
 *
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_model_status_new (void);

/**
 * \brief  Creates a model-status query carrying an optional request filter.
 *
 * Pass NULL, NULL, NULL and -1 for no filter (full dump). The response schema
 * is identical either way.
 *
 * params[in] model_name       Only that model, or NULL for every model.
 * params[in] model_version    Only that (name, version) instance, or NULL for
 *                             every version of the name. Optional, and needs
 *                             model_name to be set.
 * params[in] stream_name      Only that stream, or NULL for every stream.
 * params[in] source_id        Only that stream, or -1 for every stream.
 *
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_model_status_new_filtered (const gchar * model_name,
    const gchar * model_version, const gchar * stream_name, gint source_id);

/**
 * \brief  Parses the request filter from a model-status query.
 *
 * params[in] query             A pointer to a model-status query.
 * params[out] model_name       A pointer to the model name; "" if unset.
 * params[out] model_version    A pointer to the model version; "" if unset.
 * params[out] stream_name      A pointer to the stream name; "" if unset.
 * params[out] source_id        A pointer to the source ID; -1 if unset.
 *    The strings should not be modified, and remain valid until the query is
 *    unreffed. Please use or make copy within scope.
 *
 * @return  A Boolean; true if the query was successfully parsed.
 */
gboolean gst_nvquery_model_status_parse_request (GstQuery * query,
    const gchar ** model_name, const gchar ** model_version,
    const gchar ** stream_name, gint * source_id);

/**
 * Determines whether a query is a model-status query.
 *
 * params[in] query   A pointer to the query to be checked.
 *
 * @return  A Boolean; true if the query is a model-status query.
 */
gboolean gst_nvquery_is_model_status (GstQuery * query);

/**
 * Sets the JSON status response on a model-status query.
 *
 * params[in] query   A pointer to a model-status query.
 * params[in] json    The JSON status response.
 */
void gst_nvquery_model_status_set_response (GstQuery * query, const gchar * json);

/**
 * \brief  Parses the JSON status response from a model-status query.
 *
 * params[in] query     A pointer to a model-status query.
 * params[out] json     A pointer to the JSON response. The string should not be
 *                      modified, and remains valid until the query is unreffed.
 *
 * @return  A Boolean; true if a response was set.
 */
gboolean gst_nvquery_model_status_parse_response (GstQuery * query,
    const gchar ** json);

/* ---- stream route (native REST GET /api/v1/stream/route) ---- */

/**
 * Creates a new stream-route query - the read side of the routing plane, which
 * the REST server sends downstream for the model-managing element to answer.
 *
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_stream_route_new (void);

/**
 * \brief  Creates a stream-route query carrying an optional point lookup.
 *
 * The filter is a camera_id, the same identifier that
 * POST /api/v1/stream/route accepts. Pass NULL, and only NULL, for no filter;
 * any non-NULL value, including an empty string, is a filter which is present,
 * so a malformed point lookup matches nothing instead of returning every route.
 *
 * params[in] camera_id     The camera ID to look up, or NULL for no filter.
 *
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_stream_route_new_filtered (const gchar * camera_id);

/**
 * \brief  Parses the request filter from a stream-route query.
 *
 * params[in] query          A pointer to a stream-route query.
 * params[out] camera_id     A pointer to the camera ID. The string should not
 *                           be modified, and remains valid until the query is
 *                           unreffed. Please use or make a copy within scope.
 * params[out] filtered      A pointer to the filter-present flag. The flag is
 *                           authoritative, not the string: true with an empty
 *                           camera_id is a present filter matching nothing.
 *
 * @return  A Boolean; true if the query was successfully parsed.
 */
gboolean gst_nvquery_stream_route_parse_request (GstQuery * query,
    const gchar ** camera_id, gboolean * filtered);

/**
 * Determines whether a query is a stream-route query.
 *
 * params[in] query   A pointer to the query to be checked.
 *
 * @return  A Boolean; true if the query is a stream-route query.
 */
gboolean gst_nvquery_is_stream_route (GstQuery * query);

/**
 * Sets the JSON routing response on a stream-route query.
 *
 * params[in] query   A pointer to a stream-route query.
 * params[in] json    The JSON routing response.
 */
void gst_nvquery_stream_route_set_response (GstQuery * query, const gchar * json);

/**
 * \brief  Parses the JSON routing response from a stream-route query.
 *
 * params[in] query     A pointer to a stream-route query.
 * params[out] json     A pointer to the JSON response. The string should not be
 *                      modified, and remains valid until the query is unreffed.
 *
 * @return  A Boolean; true if a response was set.
 */
gboolean gst_nvquery_stream_route_parse_response (GstQuery * query,
    const gchar ** json);

/* ---- generic control op: SYNCHRONOUS admission query ---- */

/**
 * \brief  Creates a new control-op admission query.
 *
 * The query is sent downstream via peer_query by the REST layer, so that the
 * model-managing element returns its admission verdict in the same call. It is
 * used for model/unload; the asynchronous ops keep the in-band event instead.
 *
 * params[in] op            The plane path, for example "model/unload".
 * params[in] value_json    The request "value" object, as verbatim JSON.
 *
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_control_new (const gchar * op, const gchar * value_json);

/**
 * Determines whether a query is a control-op admission query.
 *
 * params[in] query   A pointer to the query to be checked.
 *
 * @return  A Boolean; true if the query is a control-op admission query.
 */
gboolean gst_nvquery_is_control (GstQuery * query);

/**
 * \brief  Parses the op and value JSON from a control-op admission query.
 *
 * params[in] query          A pointer to a control-op admission query.
 * params[out] op            A pointer to the plane path.
 * params[out] value_json    A pointer to the request "value" object.
 *    The strings should not be modified, and remain valid until the query is
 *    unreffed. Please use or make copy within scope.
 *
 * @return  A Boolean; true if the query was successfully parsed.
 */
gboolean gst_nvquery_control_parse_request (GstQuery * query, const gchar ** op,
    const gchar ** value_json);

/**
 * \brief  Writes the admission verdict on a control-op admission query.
 *
 * params[in] query       A pointer to a control-op admission query.
 * params[in] http_code   The verdict: 202 if accepted, 400 if rejected.
 * params[in] err_code    A stable error ID; NULL or "" when accepted.
 * params[in] reason      A human-readable reason for the verdict.
 * params[in] hint        An optional hint; may be NULL.
 */
void gst_nvquery_control_set_response (GstQuery * query, gint http_code,
    const gchar * err_code, const gchar * reason, const gchar * hint);

/**
 * \brief  Parses the admission verdict from a control-op admission query.
 *
 * params[in] query         A pointer to a control-op admission query.
 * params[out] http_code    A pointer to the verdict HTTP code.
 * params[out] err_code     A pointer to the stable error ID.
 * params[out] reason       A pointer to the reason for the verdict.
 * params[out] hint         A pointer to the optional hint.
 *    The strings should not be modified, and remain valid until the query is
 *    unreffed. Please use or make copy within scope.
 *
 * @return  A Boolean; false if no handler answered, because the element is
 *          absent or the query is not a control-op query.
 */
gboolean gst_nvquery_control_parse_response (GstQuery * query, gint * http_code,
    const gchar ** err_code, const gchar ** reason, const gchar ** hint);

/* ---- model/update: IN-PLACE checkpoint transition ---- */

/**
 * Creates a new model-update message - denoting an in-place checkpoint
 * transition on the model plane.
 *
 * This never routes streams; routing is the stream-route message below.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] model_info    A pointer to @ref NvDsModelInfo holding value_json.
 *
 * @return  A pointer to the new message.
 */
GstMessage * gst_nvmessage_new_model_update (GstObject * obj,
    NvDsModelInfo * model_info);

/**
 * Determines whether a message is a model-update message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a model-update message.
 */
gboolean gst_nvmessage_is_model_update (GstMessage * message);

/**
 * \brief  Parses the model info from a model-update message.
 *
 * params[in] message        A pointer to a model-update message.
 * params[out] model_info    A pointer to @ref NvDsModelInfo
 *    The string NvDsModelInfo->value_json should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_model_update (GstMessage * message,
    NvDsModelInfo * model_info);

/* ---- stream plane (native REST /api/v1/stream/route) ---- */

/**
 * \brief  Creates a new stream-route message - the write side of the routing
 * plane.
 *
 * The message carries the declarative routing document as verbatim JSON, with
 * its routes augmented with the source IDs resolved by nvmultiurisrcbin.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] route_info    A pointer to @ref NvDsRouteInfo holding value_json.
 *
 * @return  A pointer to the new message.
 */
GstMessage * gst_nvmessage_new_stream_route (GstObject * obj,
    NvDsRouteInfo * route_info);

/**
 * Determines whether a message is a stream-route message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a stream-route message.
 */
gboolean gst_nvmessage_is_stream_route (GstMessage * message);

/**
 * \brief  Parses the route info from a stream-route message.
 *
 * params[in] message        A pointer to a stream-route message.
 * params[out] route_info    A pointer to @ref NvDsRouteInfo
 *    The string NvDsRouteInfo->value_json should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_stream_route (GstMessage * message,
    NvDsRouteInfo * route_info);

/**
 * \brief  Sends custom message to force-eos on the pipeline
 *
 * The element on which custom message related to eos is to be sent.
 *
 * params[in] object           A gboject on which custom message to be sent
 * params[in] force_eos        A force_eos variable 
 * @return  A GstMessage;      The GstMessage which was successfully sent
 */
GstMessage *gst_nvmessage_force_pipeline_eos (GstObject * obj,
    gboolean force_eos);


/**
 * \brief  Parses the force_eos from a force eos message.
 *
 *
 * params[in] message           A pointer to a force eos message.
 * params[out] force_eos        A pointer to force_eos variable
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean
gst_nvmessage_parse_force_pipeline_eos (GstMessage * message,
    gboolean * force_eos);


/**
 * Determines whether a message is a force pipeline eos message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a force pipeline message.
 */
gboolean gst_nvmessage_is_force_pipeline_eos (GstMessage * message);

/**
 * Creates a new attempt-exceeded message - denoting a reconnection attempt
 * is exceeded for a rtsp uri
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] rtsp_info     rtsp info of the stream which exceeds rtsp
 *                          reconnection attempt
 *
 * @return  A pointer to the new message.
 */
GstMessage *gst_nvmessage_reconnect_attempt_exceeded (GstObject * obj,
      NvDsRtspAttemptsInfo *rtsp_info);

/**
 * \brief  Parses the NvDsRtspAttemptsInfo from a attempt-exceeded message.
 *
 *
 * params[in] message                A pointer to a attempt-exceeded message.
 * params[out] NvDsRtspAttemptsInfo  A pointer to @ref NvDsRtspAttemptsInfo
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean
gst_nvmessage_parse_reconnect_attempt_exceeded (GstMessage * message,
    NvDsRtspAttemptsInfo *rtsp_info);

/**
 * Determines whether a message is a attempt-exceeded message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a attempt exceeded message.
 */
gboolean gst_nvmessage_is_reconnect_attempt_exceeded (GstMessage * message);

/** @} */

G_END_DECLS
#endif
