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

#include "gst-nvdscustomevent.h"

GstEvent *
gst_nvevent_new_roi_update (gchar * stream_id, guint roi_count,
    RoiDimension * roi_dim)
{
  GstStructure *str = gst_structure_new_empty ("nv-roi-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
      "roi-count", G_TYPE_UINT, roi_count, NULL);

  for (int i = 0; i < (int) roi_count; i++) {
    char key[128];
    g_snprintf (key, sizeof (key), "roi_id_%d", i);
    gst_structure_set (str, key, G_TYPE_STRING, (char *) roi_dim[i].roi_id,
        NULL);
    g_snprintf (key, sizeof (key), "left_%d", i);
    gst_structure_set (str, key, G_TYPE_UINT, roi_dim[i].left, NULL);
    g_snprintf (key, sizeof (key), "top_%d", i);
    gst_structure_set (str, key, G_TYPE_UINT, roi_dim[i].top, NULL);
    g_snprintf (key, sizeof (key), "width_%d", i);
    gst_structure_set (str, key, G_TYPE_UINT, roi_dim[i].width, NULL);
    g_snprintf (key, sizeof (key), "height_%d", i);
    gst_structure_set (str, key, G_TYPE_UINT, roi_dim[i].height, NULL);
  }

  return gst_event_new_custom ((GstEventType) GST_NVEVENT_ROI_UPDATE, str);
}

GstEvent *
gst_nvevent_infer_interval_update (gchar * stream_id, guint interval)
{
  GstStructure *str = gst_structure_new_empty ("nv-infer-interval-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
      "interval", G_TYPE_UINT, interval, NULL);

  return gst_event_new_custom ((GstEventType) GST_NVEVENT_INFER_INTERVAL_UPDATE,
      str);
}


GstEvent *
gst_nvevent_nvtracker_config_update (gchar * stream_id, gchar * configPath)
{
  GstStructure *str = gst_structure_new_empty ("nv-tracker-config-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
      "config_path", G_TYPE_STRING, configPath, NULL);

  return gst_event_new_custom ((GstEventType) GST_NVEVENT_NVTRACKER_CONFIG_UPDATE,
      str);
}

void
gst_nvevent_parse_roi_update (GstEvent * event, gchar ** stream_id,
    guint * roi_count, RoiDimension ** roi_dim)
{
  if ((GstEventType) GST_NVEVENT_ROI_UPDATE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
        "roi-count", G_TYPE_UINT, roi_count, NULL);

    *roi_dim = (RoiDimension *) g_malloc (sizeof (RoiDimension) * (*roi_count));

    for (int i = 0; i < (int) *roi_count; i++) {

      char key[128];
      gchar *roi_id;
      g_snprintf (key, sizeof (key), "roi_id_%d", i);
      gst_structure_get (str, key, G_TYPE_STRING, &(roi_id), NULL);
      g_strlcpy (((*roi_dim)[i].roi_id), roi_id,
          sizeof (((*roi_dim)[i].roi_id)));
      g_free (roi_id);
      g_snprintf (key, sizeof (key), "left_%d", i);
      gst_structure_get (str, key, G_TYPE_UINT, &((*roi_dim)[i].left), NULL);
      g_snprintf (key, sizeof (key), "top_%d", i);
      gst_structure_get (str, key, G_TYPE_UINT, &((*roi_dim)[i].top), NULL);
      g_snprintf (key, sizeof (key), "width_%d", i);
      gst_structure_get (str, key, G_TYPE_UINT, &((*roi_dim)[i].width), NULL);
      g_snprintf (key, sizeof (key), "height_%d", i);
      gst_structure_get (str, key, G_TYPE_UINT, &((*roi_dim)[i].height), NULL);

    }
  }
}

void
gst_nvevent_parse_infer_interval_update (GstEvent * event, gchar ** stream_id,
    guint * interval)
{
  if ((GstEventType) GST_NVEVENT_INFER_INTERVAL_UPDATE ==
      GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
        "interval", G_TYPE_UINT, interval, NULL);
  }
}

void
gst_nvevent_parse_nvtracker_config_update (GstEvent * event, gchar ** stream_id,
    gchar** configStr)
{
  if ((GstEventType) GST_NVEVENT_NVTRACKER_CONFIG_UPDATE == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
        "config_path", G_TYPE_STRING, configStr, NULL);
  }
}

GstEvent *
gst_nvevent_osd_process_mode_update (gchar * stream_id, guint process_mode)
{
  GstStructure *str = gst_structure_new_empty ("nv-osd-process-mode-update");

  gst_structure_set (str, "stream_id", G_TYPE_STRING, stream_id,
      "process_mode", G_TYPE_UINT, process_mode, NULL);

  return gst_event_new_custom ((GstEventType)
      GST_NVEVENT_OSD_PROCESS_MODE_UPDATE, str);
}

void
gst_nvevent_parse_osd_process_mode_update (GstEvent * event, gchar ** stream_id,
    guint * process_mode)
{
  if ((GstEventType) GST_NVEVENT_OSD_PROCESS_MODE_UPDATE ==
      GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "stream_id", G_TYPE_STRING, stream_id,
        "process_mode", G_TYPE_UINT, process_mode, NULL);
  }
}

GstEvent*
gst_nvevent_analytics_reload_config_update(gchar* config_file_path)
{
  GstStructure* str = gst_structure_new_empty("nv-analytics-reload_config-update");

  gst_structure_set(str, "config_file_path", G_TYPE_STRING, config_file_path, NULL);

  return gst_event_new_custom((GstEventType)
      GST_NVEVENT_ANALYTICS_RELOAD_CONFIG_UPDATE, str);
}

void
gst_nvevent_parse_analytics_reload_config_update(GstEvent* event, gchar** config_file_path)
{
  if ((GstEventType)GST_NVEVENT_ANALYTICS_RELOAD_CONFIG_UPDATE ==
      GST_EVENT_TYPE(event)) {
    const GstStructure* str = gst_event_get_structure(event);
    gst_structure_get(str, "config_file_path", G_TYPE_STRING, config_file_path, NULL);
  }
}


/* ------------------------------------------------------------------ */
/* Model control events (model/load|unload|update)                     */
/*                                                                      */
/* These carry a REST model request in-band on the data plane so a     */
/* downstream model-managing element receives it regardless of who     */
/* owns the pipeline bus. All payload fields are plain strings; a NULL  */
/* input is encoded as "" and surfaces as NULL on parse.                */
/* ------------------------------------------------------------------ */

/* Model-plane + stream-routing control events: ONE payload -- the request
 * "value" object as verbatim JSON. Positional argument lists cannot grow with
 * the schema (engine_files{}, instances[], nested refs), so they do not exist
 * here; the consuming element parses the document (nvmodelmux: json-glib). */

/* Copy a structure string field into *out (newly allocated), or NULL when
 * the field is absent/empty. */
static void
mm_get_str (const GstStructure * str, const gchar * field, gchar ** out)
{
  const gchar *v = (out && str) ? gst_structure_get_string (str, field) : NULL;
  if (out)
    *out = (v && *v) ? g_strdup (v) : NULL;
}

static GstEvent *
s_new_json_payload_event (GstNvDsCustomEventType type, const gchar * name,
    const gchar * value_json)
{
  GstStructure *str;

  g_return_val_if_fail (value_json && *value_json, NULL);
  str = gst_structure_new (name, "value_json", G_TYPE_STRING, value_json, NULL);
  return gst_event_new_custom ((GstEventType) type, str);
}

static void
s_parse_json_payload_event (GstEvent * event, GstNvDsCustomEventType type,
    gchar ** value_json)
{
  if ((GstEventType) type == GST_EVENT_TYPE (event))
    mm_get_str (gst_event_get_structure (event), "value_json", value_json);
  else if (value_json)
    *value_json = NULL;
}

GstEvent *
gst_nvevent_new_model_load (const gchar * value_json)
{
  return s_new_json_payload_event (GST_NVEVENT_MODEL_LOAD, "nv-model-load",
      value_json);
}

void
gst_nvevent_parse_model_load (GstEvent * event, gchar ** value_json)
{
  s_parse_json_payload_event (event, GST_NVEVENT_MODEL_LOAD, value_json);
}

GstEvent *
gst_nvevent_new_model_unload (const gchar * value_json)
{
  return s_new_json_payload_event (GST_NVEVENT_MODEL_UNLOAD, "nv-model-unload",
      value_json);
}

void
gst_nvevent_parse_model_unload (GstEvent * event, gchar ** value_json)
{
  s_parse_json_payload_event (event, GST_NVEVENT_MODEL_UNLOAD, value_json);
}

GstEvent *
gst_nvevent_new_model_update (const gchar * value_json)
{
  return s_new_json_payload_event (GST_NVEVENT_MODEL_UPDATE, "nv-model-update",
      value_json);
}

void
gst_nvevent_parse_model_update (GstEvent * event, gchar ** value_json)
{
  s_parse_json_payload_event (event, GST_NVEVENT_MODEL_UPDATE, value_json);
}

GstEvent *
gst_nvevent_new_stream_route (const gchar * value_json)
{
  return s_new_json_payload_event (GST_NVEVENT_STREAM_ROUTE, "nv-stream-route",
      value_json);
}

void
gst_nvevent_parse_stream_route (GstEvent * event, gchar ** value_json)
{
  s_parse_json_payload_event (event, GST_NVEVENT_STREAM_ROUTE, value_json);
}

/* ------------------------------------------------------------------ */
/* Stream model-binding event (stream/add per-stream model selection)  */
/*                                                                      */
/* Carries the assigned source_id + camera_id + raw metadata JSON in    */
/* band on the data plane, so per-stream model routing works even when  */
/* the host owns the pipeline bus (the stream-add bus message does not   */
/* reach the element then; PAD_ADDED carries only source_id).           */
/* ------------------------------------------------------------------ */

GstEvent *
gst_nvevent_new_stream_model_bind (guint source_id, const gchar * camera_id,
    const gchar * camera_name, const gchar * metadata_json)
{
  GstStructure *str = gst_structure_new_empty ("nv-stream-model-bind");
  gst_structure_set (str,
      "source_id", G_TYPE_UINT, source_id,
      "camera_id", G_TYPE_STRING, camera_id ? camera_id : "",
      "camera_name", G_TYPE_STRING, camera_name ? camera_name : "",
      "metadata_json", G_TYPE_STRING, metadata_json ? metadata_json : "",
      NULL);
  return gst_event_new_custom ((GstEventType) GST_NVEVENT_STREAM_MODEL_BIND, str);
}

void
gst_nvevent_parse_stream_model_bind (GstEvent * event, guint * source_id,
    gchar ** camera_id, gchar ** camera_name, gchar ** metadata_json)
{
  if ((GstEventType) GST_NVEVENT_STREAM_MODEL_BIND == GST_EVENT_TYPE (event)) {
    const GstStructure *str = gst_event_get_structure (event);
    if (source_id)
      gst_structure_get_uint (str, "source_id", source_id);
    mm_get_str (str, "camera_id", camera_id);
    mm_get_str (str, "camera_name", camera_name);
    mm_get_str (str, "metadata_json", metadata_json);
  }
}
