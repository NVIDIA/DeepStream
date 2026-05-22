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

