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

#include "gst-nvdscustommessage.h"

#define STREAM_ADD_STRUCT_NAME "stream-add"
#define STREAM_REMOVE_STRUCT_NAME "stream-remove"
#define MODEL_LOAD_STRUCT_NAME "model-load"
#define MODEL_UNLOAD_STRUCT_NAME "model-unload"
#define MODEL_UPDATE_STRUCT_NAME "model-update"
#define STREAM_ROUTE_STRUCT_NAME "stream-route"
#define PIPELINE_EOS_STRUCT_NAME "force-pipeline-eos"
#define PIPELINE_RTSP_RECONNECT_ATTEMPT_STRUCT_NAME "attempt-exceeded"
#define MODEL_STATUS_QUERY_NAME "nv-model-status"
/* distinct from STREAM_ROUTE_STRUCT_NAME above: that names the control
 * MESSAGE (the write), this names the read-back QUERY. */
#define STREAM_ROUTE_QUERY_NAME "nv-stream-route"
#define CONTROL_QUERY_NAME "nv-control"
#define MODEL_EVENT_STRUCT_NAME "nv-model-event"

#define CHECK_MESSAGE_TYPE(message,type) do { \
  const GstStructure *str; \
  if (GST_MESSAGE_TYPE (message) != GST_MESSAGE_ELEMENT) \
    return FALSE; \
  str = gst_message_get_structure (message); \
  return (str != NULL) && gst_structure_has_name (str,type); \
} while (0)

/* gst_structure_get_string() returns NULL for an absent/NULL field. Normalize to
 * "" so downstream callers never deref NULL, and so the bus-message parse path
 * matches the REST/JSON parse path (which yields "" for absent fields). */
static inline const gchar *
s_str_or_empty (const gchar * s)
{
  return s ? s : "";
}

GstMessage *
gst_nvmessage_new_stream_add (GstObject * obj, NvDsSensorInfo * sensor_info)
{
  GstStructure *str =
      gst_structure_new (STREAM_ADD_STRUCT_NAME, "source-id", G_TYPE_UINT,
      sensor_info->source_id,
      "sensor-id", G_TYPE_STRING, sensor_info->sensor_id,
      "sensor-name", G_TYPE_STRING, sensor_info->sensor_name,
      "uri", G_TYPE_STRING, sensor_info->uri,
      "sensor-metadata", G_TYPE_STRING, sensor_info->sensor_metadata,  NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_is_force_pipeline_eos (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, PIPELINE_EOS_STRUCT_NAME);
}

GstMessage *
gst_nvmessage_force_pipeline_eos (GstObject * obj, gboolean force_eos)
{
  GstStructure *str = gst_structure_new (PIPELINE_EOS_STRUCT_NAME,
      "force_eos", G_TYPE_BOOLEAN, force_eos, NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_parse_force_pipeline_eos (GstMessage * message,
    gboolean * force_eos)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_force_pipeline_eos (message))
    return FALSE;

  str = gst_message_get_structure (message);

  gst_structure_get_boolean (str, "force_eos", force_eos);
  return TRUE;
}

gboolean
gst_nvmessage_is_stream_add (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, STREAM_ADD_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_stream_add (GstMessage * message,
    NvDsSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_add (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = s_str_or_empty (gst_structure_get_string (str, "sensor-id"));
  sensor_info->sensor_name = s_str_or_empty (gst_structure_get_string (str, "sensor-name"));
  sensor_info->uri = s_str_or_empty (gst_structure_get_string (str, "uri"));
  sensor_info->sensor_metadata = s_str_or_empty (gst_structure_get_string (str, "sensor-metadata"));
  return TRUE;
}

gboolean
gst_nvmessage_parse_fps_stream_add (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_add (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = s_str_or_empty (gst_structure_get_string (str, "sensor-id"));
  sensor_info->sensor_name = s_str_or_empty (gst_structure_get_string (str, "sensor-name"));
  sensor_info->uri = s_str_or_empty (gst_structure_get_string (str, "uri"));
  return TRUE;
}

GstMessage *
gst_nvmessage_new_stream_remove (GstObject * obj, NvDsSensorInfo * sensor_info)
{
  GstStructure *str =
      gst_structure_new (STREAM_REMOVE_STRUCT_NAME, "source-id", G_TYPE_UINT,
      sensor_info->source_id,
      "sensor-id", G_TYPE_STRING, sensor_info->sensor_id,
      "sensor-name", G_TYPE_STRING, sensor_info->sensor_name,
      "uri", G_TYPE_STRING, sensor_info->uri,
      "sensor-metadata", G_TYPE_STRING, sensor_info->sensor_metadata, NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_is_stream_remove (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, STREAM_REMOVE_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_stream_remove (GstMessage * message,
    NvDsSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_remove (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = s_str_or_empty (gst_structure_get_string (str, "sensor-id"));
  sensor_info->sensor_name = s_str_or_empty (gst_structure_get_string (str, "sensor-name"));
  sensor_info->uri = s_str_or_empty (gst_structure_get_string (str, "uri"));
  sensor_info->sensor_metadata = s_str_or_empty (gst_structure_get_string (str, "sensor-metadata"));
  return TRUE;
}

gboolean
gst_nvmessage_parse_fps_stream_remove (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_remove (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = s_str_or_empty (gst_structure_get_string (str, "sensor-id"));
  sensor_info->sensor_name = s_str_or_empty (gst_structure_get_string (str, "sensor-name"));
  sensor_info->uri = s_str_or_empty (gst_structure_get_string (str, "uri"));
  return TRUE;
}

/* ---- model load / unload ---- */

/* Model-plane + stream-route messages: ONE payload -- the request "value"
 * object as verbatim JSON (NvDsModelInfo/NvDsRouteInfo value_json). One schema
 * end-to-end; the consumer parses the document. */

static GstMessage *
s_new_json_payload_message (GstObject * obj, const gchar * struct_name,
    const gchar * value_json)
{
  GstStructure *str = gst_structure_new (struct_name,
      "value-json", G_TYPE_STRING, s_str_or_empty (value_json), NULL);
  return gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);
}

static gboolean
s_parse_json_payload_message (GstMessage * message, const gchar ** value_json)
{
  const GstStructure *str = gst_message_get_structure (message);
  *value_json = s_str_or_empty (gst_structure_get_string (str, "value-json"));
  return TRUE;
}

GstMessage *
gst_nvmessage_new_model_load (GstObject * obj, NvDsModelInfo * model_info)
{
  return s_new_json_payload_message (obj, MODEL_LOAD_STRUCT_NAME,
      model_info->value_json);
}

gboolean
gst_nvmessage_is_model_load (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, MODEL_LOAD_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_model_load (GstMessage * message, NvDsModelInfo * model_info)
{
  if (!gst_nvmessage_is_model_load (message))
    return FALSE;
  return s_parse_json_payload_message (message, &model_info->value_json);
}

GstMessage *
gst_nvmessage_new_model_unload (GstObject * obj, NvDsModelInfo * model_info)
{
  return s_new_json_payload_message (obj, MODEL_UNLOAD_STRUCT_NAME,
      model_info->value_json);
}

gboolean
gst_nvmessage_is_model_unload (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, MODEL_UNLOAD_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_model_unload (GstMessage * message, NvDsModelInfo * model_info)
{
  if (!gst_nvmessage_is_model_unload (message))
    return FALSE;
  return s_parse_json_payload_message (message, &model_info->value_json);
}

/* ---- model-plane COMPLETION event (nvmodelmux -> app bus) ----
 * Exactly the stream-add/remove notification pattern: nvmodelmux POSTS this on
 * the pipeline bus when an async op actually finishes (warmed / torn down / OTA
 * committed / reroute settled); the app's bus handler prints or reacts. */
GstMessage *
gst_nvmessage_new_model_event (GstObject * obj, NvDsModelEventInfo * info)
{
  GstStructure *str = gst_structure_new (MODEL_EVENT_STRUCT_NAME,
      "event", G_TYPE_STRING, s_str_or_empty (info->event),
      "name", G_TYPE_STRING, s_str_or_empty (info->name),
      "version", G_TYPE_STRING, s_str_or_empty (info->version),
      "gpu", G_TYPE_INT, info->gpu,
      "ok", G_TYPE_BOOLEAN, info->ok,
      "detail", G_TYPE_STRING, s_str_or_empty (info->detail), NULL);
  return gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);
}

gboolean
gst_nvmessage_is_model_event (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, MODEL_EVENT_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_model_event (GstMessage * message, NvDsModelEventInfo * info)
{
  const GstStructure *str;
  if (!gst_nvmessage_is_model_event (message))
    return FALSE;
  str = gst_message_get_structure (message);
  info->event = s_str_or_empty (gst_structure_get_string (str, "event"));
  info->name = s_str_or_empty (gst_structure_get_string (str, "name"));
  info->version = s_str_or_empty (gst_structure_get_string (str, "version"));
  if (!gst_structure_get_int (str, "gpu", &info->gpu))
    info->gpu = -1;
  if (!gst_structure_get_boolean (str, "ok", &info->ok))
    info->ok = TRUE;
  info->detail = s_str_or_empty (gst_structure_get_string (str, "detail"));
  return TRUE;
}

/* ---- model/update: IN-PLACE checkpoint transition (never routes streams) ---- */

GstMessage *
gst_nvmessage_new_model_update (GstObject * obj, NvDsModelInfo * model_info)
{
  return s_new_json_payload_message (obj, MODEL_UPDATE_STRUCT_NAME,
      model_info->value_json);
}

gboolean
gst_nvmessage_is_model_update (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, MODEL_UPDATE_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_model_update (GstMessage * message, NvDsModelInfo * model_info)
{
  if (!gst_nvmessage_is_model_update (message))
    return FALSE;
  return s_parse_json_payload_message (message, &model_info->value_json);
}

/* ---- stream/route: the declarative routing document (stream plane) ---- */

GstMessage *
gst_nvmessage_new_stream_route (GstObject * obj, NvDsRouteInfo * route_info)
{
  return s_new_json_payload_message (obj, STREAM_ROUTE_STRUCT_NAME,
      route_info->value_json);
}

gboolean
gst_nvmessage_is_stream_route (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, STREAM_ROUTE_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_stream_route (GstMessage * message, NvDsRouteInfo * route_info)
{
  if (!gst_nvmessage_is_stream_route (message))
    return FALSE;
  return s_parse_json_payload_message (message, &route_info->value_json);
}

/* ---- model status (synchronous custom query) ---- */

GstQuery *
gst_nvquery_model_status_new_filtered (const gchar * model_name,
    const gchar * model_version, const gchar * stream_name, gint source_id)
{
  /* optional request filter carried on the query: empty model/version/stream + source_id<0
   * means "no filter" (full dump). model_version refines model_name (ignored without a name).
   * The app reads these with parse_request. */
  GstStructure *str = gst_structure_new (MODEL_STATUS_QUERY_NAME,
      "req-model", G_TYPE_STRING, s_str_or_empty (model_name),
      "req-model-version", G_TYPE_STRING, s_str_or_empty (model_version),
      "req-stream", G_TYPE_STRING, s_str_or_empty (stream_name),
      "req-source-id", G_TYPE_INT, source_id, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

GstQuery *
gst_nvquery_model_status_new (void)
{
  return gst_nvquery_model_status_new_filtered (NULL, NULL, NULL, -1);
}

gboolean
gst_nvquery_model_status_parse_request (GstQuery * query,
    const gchar ** model_name, const gchar ** model_version,
    const gchar ** stream_name, gint * source_id)
{
  const GstStructure *str;
  if (!gst_nvquery_is_model_status (query))
    return FALSE;
  str = gst_query_get_structure (query);
  if (model_name)
    *model_name = gst_structure_get_string (str, "req-model");
  if (model_version)
    *model_version = gst_structure_get_string (str, "req-model-version");
  if (stream_name)
    *stream_name = gst_structure_get_string (str, "req-stream");
  if (source_id) {
    if (!gst_structure_get_int (str, "req-source-id", source_id))
      *source_id = -1;
  }
  return TRUE;
}

gboolean
gst_nvquery_is_model_status (GstQuery * query)
{
  const GstStructure *str;
  if (!query || GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;
  str = gst_query_get_structure (query);
  return (str != NULL) && gst_structure_has_name (str, MODEL_STATUS_QUERY_NAME);
}

void
gst_nvquery_model_status_set_response (GstQuery * query, const gchar * json)
{
  GstStructure *str;
  if (!gst_nvquery_is_model_status (query))
    return;
  str = gst_query_writable_structure (query);
  gst_structure_set (str, "response-json", G_TYPE_STRING, s_str_or_empty (json), NULL);
}

gboolean
gst_nvquery_model_status_parse_response (GstQuery * query, const gchar ** json)
{
  const GstStructure *str;
  if (!gst_nvquery_is_model_status (query) || !json)
    return FALSE;
  str = gst_query_get_structure (query);
  *json = gst_structure_get_string (str, "response-json");
  return (*json != NULL);
}

GstQuery *
gst_nvquery_stream_route_new_filtered (const gchar * camera_id)
{
  GstStructure *str = gst_structure_new (STREAM_ROUTE_QUERY_NAME,
      "req-filtered", G_TYPE_BOOLEAN, (camera_id != NULL),
      "req-camera-id", G_TYPE_STRING, s_str_or_empty (camera_id), NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

GstQuery *
gst_nvquery_stream_route_new (void)
{
  return gst_nvquery_stream_route_new_filtered (NULL);
}

gboolean
gst_nvquery_is_stream_route (GstQuery * query)
{
  const GstStructure *str;
  if (!query || GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;
  str = gst_query_get_structure (query);
  return (str != NULL) && gst_structure_has_name (str, STREAM_ROUTE_QUERY_NAME);
}

gboolean
gst_nvquery_stream_route_parse_request (GstQuery * query,
    const gchar ** camera_id, gboolean * filtered)
{
  const GstStructure *str;
  if (!gst_nvquery_is_stream_route (query))
    return FALSE;
  str = gst_query_get_structure (query);
  if (camera_id)
    *camera_id = gst_structure_get_string (str, "req-camera-id");
  if (filtered && !gst_structure_get_boolean (str, "req-filtered", filtered))
    *filtered = FALSE;
  return TRUE;
}

void
gst_nvquery_stream_route_set_response (GstQuery * query, const gchar * json)
{
  GstStructure *str;
  if (!gst_nvquery_is_stream_route (query))
    return;
  str = gst_query_writable_structure (query);
  gst_structure_set (str, "response-json", G_TYPE_STRING, s_str_or_empty (json), NULL);
}

gboolean
gst_nvquery_stream_route_parse_response (GstQuery * query, const gchar ** json)
{
  const GstStructure *str;
  if (!gst_nvquery_is_stream_route (query) || !json)
    return FALSE;
  str = gst_query_get_structure (query);
  *json = gst_structure_get_string (str, "response-json");
  return (*json != NULL);
}

/* ---- generic control op (synchronous ADMISSION query) ----
 * Carries a control request downstream to the model-managing element (nvmodelmux),
 * which decides admission SYNCHRONOUSLY and writes back {http, err_code, reason,
 * hint} -- so the REST caller gets the element's real verdict in the same call
 * instead of a blind 202. `op` is the plane path, e.g. "model/unload". */
GstQuery *
gst_nvquery_control_new (const gchar * op, const gchar * value_json)
{
  GstStructure *str = gst_structure_new (CONTROL_QUERY_NAME,
      "op", G_TYPE_STRING, s_str_or_empty (op),
      "value-json", G_TYPE_STRING, s_str_or_empty (value_json), NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_control (GstQuery * query)
{
  const GstStructure *str;
  if (!query || GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;
  str = gst_query_get_structure (query);
  return (str != NULL) && gst_structure_has_name (str, CONTROL_QUERY_NAME);
}

gboolean
gst_nvquery_control_parse_request (GstQuery * query, const gchar ** op,
    const gchar ** value_json)
{
  const GstStructure *str;
  if (!gst_nvquery_is_control (query))
    return FALSE;
  str = gst_query_get_structure (query);
  if (op)
    *op = gst_structure_get_string (str, "op");
  if (value_json)
    *value_json = gst_structure_get_string (str, "value-json");
  return TRUE;
}

/* App (nvmodelmux) side: write the admission verdict. `resp-set` distinguishes a
 * real answer from a query that merely passed through unhandled. */
void
gst_nvquery_control_set_response (GstQuery * query, gint http_code,
    const gchar * err_code, const gchar * reason, const gchar * hint)
{
  GstStructure *str;
  if (!gst_nvquery_is_control (query))
    return;
  str = gst_query_writable_structure (query);
  gst_structure_set (str,
      "resp-set", G_TYPE_BOOLEAN, TRUE,
      "resp-http", G_TYPE_INT, http_code,
      "resp-err-code", G_TYPE_STRING, s_str_or_empty (err_code),
      "resp-reason", G_TYPE_STRING, s_str_or_empty (reason),
      "resp-hint", G_TYPE_STRING, s_str_or_empty (hint), NULL);
}

/* Server (nvmultiurisrcbin) side: read the verdict. @return FALSE if the element
 * did not answer (no handler downstream). Strings borrowed until query unref. */
gboolean
gst_nvquery_control_parse_response (GstQuery * query, gint * http_code,
    const gchar ** err_code, const gchar ** reason, const gchar ** hint)
{
  const GstStructure *str;
  gboolean set = FALSE;
  if (!gst_nvquery_is_control (query))
    return FALSE;
  str = gst_query_get_structure (query);
  if (!gst_structure_get_boolean (str, "resp-set", &set) || !set)
    return FALSE;
  if (http_code && !gst_structure_get_int (str, "resp-http", http_code))
    *http_code = 500;
  if (err_code)
    *err_code = gst_structure_get_string (str, "resp-err-code");
  if (reason)
    *reason = gst_structure_get_string (str, "resp-reason");
  if (hint)
    *hint = gst_structure_get_string (str, "resp-hint");
  return TRUE;
}


GstMessage *
gst_nvmessage_reconnect_attempt_exceeded (GstObject * obj, NvDsRtspAttemptsInfo * rtsp_info)
{
  GstStructure *str = gst_structure_new (PIPELINE_RTSP_RECONNECT_ATTEMPT_STRUCT_NAME,
      "source-id", G_TYPE_UINT, rtsp_info->source_id,
      "attempt_exceeded", G_TYPE_BOOLEAN, rtsp_info->attempt_exceeded, NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}


gboolean
gst_nvmessage_is_reconnect_attempt_exceeded (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, PIPELINE_RTSP_RECONNECT_ATTEMPT_STRUCT_NAME);
}


gboolean
gst_nvmessage_parse_reconnect_attempt_exceeded (GstMessage * message,
    NvDsRtspAttemptsInfo * rtsp_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_reconnect_attempt_exceeded (message))
    return FALSE;

  str = gst_message_get_structure (message);

  gst_structure_get_boolean (str, "attempt_exceeded", &rtsp_info->attempt_exceeded);
  gst_structure_get_uint (str, "source-id", &rtsp_info->source_id);
  return TRUE;
}
