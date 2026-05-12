/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#define __USE_XOPEN
#include <time.h>
#include <json-glib/json-glib.h>
#include "deepstream_c2d_msg_util.h"
#include "deepstream_common.h"

#define CONFIG_GROUP_SENSOR "sensor"
#define CONFIG_KEY_ENABLE "enable"
#define CONFIG_KEY_ID "id"

/**
 * This function assumes the UTC time and string in the following format.
 * "2020-05-18T20:02:00.051Z" - Milliseconds are optional.
 */
static time_t
nvds_c2d_str_to_second (const gchar * str)
{
  char *err;
  struct tm tm_log = { 0 };
  time_t t1;

  g_return_val_if_fail (str, -1);

  err = strptime (str, "%Y-%m-%dT%H:%M:%S", &tm_log);
  if (err == NULL) {
    NVGSTDS_ERR_MSG_V ("Error in parsing time string");
    return -1;
  }

  t1 = mktime (&tm_log);
  return t1;
}

static time_t
nvds_get_current_utc_time (void)
{
  struct timespec ts;
  time_t tloc, t1;
  struct tm tm_log;

  clock_gettime (CLOCK_REALTIME, &ts);

  memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
  gmtime_r (&tloc, &tm_log);
  t1 = mktime (&tm_log);
  return t1;
}

NvDsC2DMsg *
nvds_c2d_parse_cloud_message (gpointer data, guint size)
{
  JsonNode *rootNode = NULL;
  GError *error = NULL;
  NvDsC2DMsg *msg = NULL;
  gchar *sensorStr = NULL;
  gint start, duration;
  gboolean startRec, ret;

  /**
   * Following minimum json message is expected to trigger the start / stop
   * of smart record.
   * {
   *   command: string   // <start-recording / stop-recording>
   *   start: string     // "2020-05-18T20:02:00.051Z"
   *   end: string       // "2020-05-18T20:02:02.851Z",
   *   sensor: {
   *     id: string
   *   }
   * }
   */
  NvDsC2DMsgSR *srMsg = g_new0 (NvDsC2DMsgSR, 1);
  JsonParser *parser = json_parser_new ();
  ret = json_parser_load_from_data (parser, data, size, &error);
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("Error in parsing json message %s", error->message);
    g_error_free (error);
    g_object_unref (parser);
    return NULL;
  }

  rootNode = json_parser_get_root (parser);
  if (JSON_NODE_HOLDS_OBJECT (rootNode)) {
    JsonObject *object;

    object = json_node_get_object (rootNode);
    if (json_object_has_member (object, "command")) {
      const gchar *type = json_object_get_string_member (object, "command");
      if (!g_strcmp0 (type, "start-recording"))
        startRec = TRUE;
      else if (!g_strcmp0 (type, "stop-recording"))
        startRec = FALSE;
      else {
        NVGSTDS_WARN_MSG_V ("wrong command %s", type);
        goto error;
      }
    } else {
      // 'command' field not provided, assume it to be start-recording.
      startRec = TRUE;
    }

    if (json_object_has_member (object, "sensor")) {
      JsonObject *tempObj = json_object_get_object_member (object, "sensor");
      if (json_object_has_member (tempObj, "id")) {
        sensorStr = g_strdup (json_object_get_string_member (tempObj, "id"));
        if (!sensorStr) {
          NVGSTDS_WARN_MSG_V ("wrong sensor.id value");
          goto error;
        }

        g_strstrip (sensorStr);
        if (!g_strcmp0 (sensorStr, "")) {
          NVGSTDS_WARN_MSG_V ("empty sensor.id value");
          goto error;
        }
      } else {
        NVGSTDS_WARN_MSG_V ("wrong message format, missing 'sensor.id' field.");
        goto error;
      }
    } else {
      NVGSTDS_WARN_MSG_V ("wrong message format, missing 'sensor.id' field.");
      goto error;
    }

    if (startRec) {
      time_t startUtc, endUtc, curUtc;
      const gchar *timeStr;
      if (json_object_has_member (object, "start")) {
        timeStr = json_object_get_string_member (object, "start");
        startUtc = nvds_c2d_str_to_second (timeStr);
        if (startUtc < 0) {
          NVGSTDS_WARN_MSG_V ("Error in parsing 'start' time - %s", timeStr);
          goto error;
        }
        curUtc = nvds_get_current_utc_time ();
        start = curUtc - startUtc;
        if (start < 0) {
          start = 0;
          NVGSTDS_WARN_MSG_V
              ("start time is in future, setting it to current time");
        }
      } else {
        NVGSTDS_WARN_MSG_V ("wrong message format, missing 'start' field.");
        goto error;
      }
      if (json_object_has_member (object, "end")) {
        timeStr = json_object_get_string_member (object, "end");
        endUtc = nvds_c2d_str_to_second (timeStr);
        if (endUtc < 0) {
          NVGSTDS_WARN_MSG_V ("Error in parsing 'end' time - %s", timeStr);
          goto error;
        }
        duration = endUtc - startUtc;
        if (duration < 0) {
          NVGSTDS_WARN_MSG_V ("Negative duration (%d), setting it to zero",
              duration);
          duration = 0;
        }
      } else {
        // Duration is not specified that means stop event will be received later.
        duration = 0;
      }
    }
  } else {
    NVGSTDS_WARN_MSG_V ("wrong message format - no json object");
    goto error;
  }

  srMsg->sensorStr = sensorStr;
  if (startRec) {
    srMsg->startTime = start;
    srMsg->duration = duration;
  }

  msg = g_new0 (NvDsC2DMsg, 1);
  if (startRec)
    msg->type = NVDS_C2D_MSG_SR_START;
  else
    msg->type = NVDS_C2D_MSG_SR_STOP;

  msg->message = (gpointer) srMsg;
  msg->msgSize = sizeof (NvDsC2DMsgSR);

  g_object_unref (parser);
  return msg;

error:
  g_object_unref (parser);
  g_free (sensorStr);
  g_free (srMsg);
  return NULL;
}

void
nvds_c2d_release_message (NvDsC2DMsg * msg)
{
  if (msg->type == NVDS_C2D_MSG_SR_STOP || msg->type == NVDS_C2D_MSG_SR_START) {

    NvDsC2DMsgSR *srMsg = (NvDsC2DMsgSR *) msg->message;
    g_free (srMsg->sensorStr);
  }
  g_free (msg->message);
  g_free (msg);
}

gboolean
nvds_c2d_parse_sensor (NvDsC2DContext * ctx, const gchar * file)
{
  gboolean ret = FALSE;
  GKeyFile *cfgFile = NULL;
  GError *error = NULL;
  gchar **groups = NULL;
  gchar **group;
  gboolean isEnabled = FALSE;
  gint sensorId;
  gchar *sensorStr = NULL;

  GHashTable *hashMap = ctx->hashMap;

  g_return_val_if_fail (hashMap, FALSE);

  cfgFile = g_key_file_new ();
  if (!g_key_file_load_from_file (cfgFile, file, G_KEY_FILE_NONE, &error)) {
    g_message ("Failed to load file: %s", error->message);
    goto done;
  }

  groups = g_key_file_get_groups (cfgFile, NULL);

  for (group = groups; *group; group++) {
    if (!strncmp (*group, CONFIG_GROUP_SENSOR, strlen (CONFIG_GROUP_SENSOR))) {

      if (sscanf (*group, CONFIG_GROUP_SENSOR "%u", &sensorId) < 1) {
        NVGSTDS_ERR_MSG_V ("Wrong sensor group name %s", *group);
        goto done;
      }

      isEnabled =
          g_key_file_get_boolean (cfgFile, *group, CONFIG_KEY_ENABLE, &error);
      if (!isEnabled) {
        // Not enabled, skip the parsing of source id.
        continue;
      } else {
        gpointer hashVal = NULL;
        sensorStr =
            g_key_file_get_string (cfgFile, *group, CONFIG_KEY_ID, &error);
        if (error) {
          NVGSTDS_ERR_MSG_V ("Error: %s", error->message);
          goto done;
        }

        hashVal = g_hash_table_lookup (hashMap, sensorStr);
        if (hashVal != NULL) {
          NVGSTDS_ERR_MSG_V ("Duplicate entries for key %s", sensorStr);
          goto done;
        }
        g_hash_table_insert (hashMap, sensorStr, &sensorId);
      }
    }
  }

  ret = TRUE;

done:
  if (error)
    g_error_free (error);

  if (groups)
    g_strfreev (groups);

  if (cfgFile)
    g_key_file_free (cfgFile);

  return ret;
}
