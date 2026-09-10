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

#include "msgbroker_c2d_receiver.hpp"
#include "nvmsgbroker.h"
#include "json-glib/json-glib.h"

#define CONFIG_GROUP_SENSOR "sensor"

using namespace deepstream;

typedef guint32 NvDsSRSessionId;

typedef struct NvDsC2DContext
{
  NvMsgBrokerClientHandle connHandle;
} NvDsC2DContext;

static time_t nvds_c2d_str_to_second(const gchar *str)
{
  char *err;
  tm tm_log;
  time_t t1;

  g_return_val_if_fail(str, -1);
  err = strptime(str, "%Y-%m-%dT%H:%M:%S%Z", &tm_log);
  if (err == NULL)
  {
    g_printerr("Error in parsing time string");
    return -1;
  }

  tm_log.tm_isdst = 0;
  t1 = mktime(&tm_log);
  if (t1 < 0)
  {
    perror("mktime");
  }
  return t1;
}

static time_t nvds_get_current_utc_time(void)
{
  struct timespec ts;
  time_t tloc, t1;
  tm tm_log;

  clock_gettime(CLOCK_REALTIME, &ts);

  memcpy(&tloc, &ts.tv_sec, sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  t1 = mktime(&tm_log);
  if (t1 < 0)
  {
    perror("mktime: ");
  }
  return t1;
}

Cloud2DeviceReceiver &Cloud2DeviceReceiver::getInstance()
{
  static Cloud2DeviceReceiver instance; // create a single instance on first use
  return instance;
}

static void nv_msgbroker_subscribe_cb(
    NvMsgBrokerErrorType flag, void *msg, int msglen, char *topic,
    void *user_ptr)
{
  Cloud2DeviceReceiver *self =
      reinterpret_cast<Cloud2DeviceReceiver *>(user_ptr);
  g_print("Message: %s\n", (char *)msg);
  if (!self->handleMessage(topic, (char *)msg, msglen))
  {
    // error
    return;
  }
}

static std::vector<std::string> parse_topics(const std::string &topics)
{
  std::vector<std::string> topics_list_parsed;
  // auto p_topics = topics_.try_get();
  if (!topics.empty())
  {
    // std::string topics_list_str = p_topics.value();
    std::string topics_list_str = topics;
    for (size_t begin = 0;;)
    {
      const size_t end = topics_list_str.find(';', begin);
      if (end == std::string::npos)
      {
        topics_list_parsed.push_back(topics_list_str.substr(begin));
        break;
      }
      else
      {
        topics_list_parsed.push_back(
            topics_list_str.substr(begin, end - begin));
        begin = end + 1;
      }
    }
  }
  return topics_list_parsed;
}

bool Cloud2DeviceReceiver::parse_msgconv_config(const std::string &file_path)
{
  GKeyFile *cfgFile = NULL;
  GError *error = NULL;
  gchar **groups = NULL;
  gchar **group;
  gint sensorId;
  gchar *sensorStr = NULL;
  cfgFile = g_key_file_new();
  if (!g_key_file_load_from_file(cfgFile, file_path.c_str(), G_KEY_FILE_NONE, &error))
  {
    printf("%s: Failed to load file: %s", "Cloud2DeviceReceiver", error->message);
    g_error_free(error);
    return false;
  }
  groups = g_key_file_get_groups(cfgFile, NULL);
  for (group = groups; *group; group++)
  {
    if (!strncmp(*group, CONFIG_GROUP_SENSOR, strlen(CONFIG_GROUP_SENSOR)))
    {
      if (sscanf(*group, CONFIG_GROUP_SENSOR "%u", &sensorId) < 1)
      {
        printf("Cloud2DeviceReceiver: Wrong sensor group name %s", *group);
        return false;
      }
      sensorStr = g_key_file_get_string(cfgFile, *group, "id", &error);
      if (error)
      {
        printf("Cloud2DeviceReceiver: %s", error->message);
        g_error_free(error);
        return false;
      }
      sensor_name_id_map_[sensorStr] = sensorId;
      g_free(sensorStr);
    }
  }
  if (groups)
    g_strfreev(groups);
  if (cfgFile)
    g_key_file_free(cfgFile);
  return true;
}

Cloud2DeviceReceiver::Cloud2DeviceReceiver() {}

Cloud2DeviceReceiver::~Cloud2DeviceReceiver() {}

void Cloud2DeviceReceiver::connect(Config &config)
{
  // connnect
  auto topics_vec = parse_topics(config.topicList);
  if (topics_vec.size() == 0)
  {
    g_printerr("No topics specified. Not connecting to the cloud\n");
    return;
  }

  if (!parse_msgconv_config(config.sensor_list_file))
  {
    g_printerr("Failed to parse the sensor list configurations\n");
    return;
  }

  context_ = new NvDsC2DContext;
  context_->connHandle = nv_msgbroker_connect(const_cast<char *>(config.conn_str.c_str()),
                                              const_cast<char *>(config.proto_lib.c_str()),
                                              NULL,
                                              const_cast<char *>(config.config_file_path.c_str()));

  char *topics[topics_vec.size()];
  for (size_t i = 0; i < topics_vec.size(); i++)
  {
    topics[i] = const_cast<char *>(topics_vec[i].c_str());
  }

  if (nv_msgbroker_subscribe(context_->connHandle, topics, topics_vec.size(),
                             nv_msgbroker_subscribe_cb,
                             this) != NV_MSGBROKER_API_OK)
  {
    g_printerr("Failed to subscribe to topics\n");
    return;
  }
}

void Cloud2DeviceReceiver::disconnect()
{
  // disconnect
  if (context_)
  {
    nv_msgbroker_disconnect(context_->connHandle);
    delete context_;
    context_ = nullptr;
  }
}

bool Cloud2DeviceReceiver::handleMessage(
    const char *topic, const char *payload, unsigned int size)
{
  GError *error = NULL;
  gboolean startRec, ret;
  std::string sensorStr;
  gint start = 0, duration = 0;
  JsonParser *parser = json_parser_new();
  ret = json_parser_load_from_data(parser, payload, size, &error);
  if (!ret)
  {
    g_printerr("Error while parsing JSON message %s\n", error->message);
    g_error_free(error);
    g_object_unref(parser);
    return false;
  }

  JsonNode *rootNode = json_parser_get_root(parser);
  if (JSON_NODE_HOLDS_OBJECT(rootNode))
  {
    JsonObject *object;

    object = json_node_get_object(rootNode);
    if (json_object_has_member(object, "command"))
    {
      const gchar *type = json_object_get_string_member(object, "command");
      if (!g_strcmp0(type, "start-recording"))
      {
        startRec = TRUE;
      }
      else if (!g_strcmp0(type, "stop-recording"))
      {
        startRec = FALSE;
      }
      else
      {
        g_printerr("Error while parsing JSON message. Wrong command %s\n", type);
        g_object_unref(parser);
        return false;
      }
    }
    else
    {
      // 'command' field not provided, assume it to be start-recording.
      startRec = TRUE;
    }

    if (json_object_has_member(object, "sensor"))
    {
      JsonObject *tempObj = json_object_get_object_member(object, "sensor");
      if (json_object_has_member(tempObj, "id"))
      {
        if (!json_object_get_string_member(tempObj, "id"))
        {
          g_printerr("Error while parsing JSON message. wrong sensor.id value\n");
          g_object_unref(parser);
          return false;
        }
        sensorStr = json_object_get_string_member(tempObj, "id");

        if (sensorStr.empty())
        {
          g_printerr("Error while parsing JSON message. empty sensor.id value\n");
          g_object_unref(parser);
          return false;
        }
      }
      else
      {
        g_printerr("Error while parsing JSON message. missing sensor.id value\n");
        g_object_unref(parser);
        return false;
      }
    }
    else
    {
      g_printerr("Error while parsing JSON message. missing sensor.id value\n");
      g_object_unref(parser);
      return false;
    }

    if (startRec)
    {
      time_t startUtc, endUtc, curUtc;
      const gchar *timeStr;

      if (json_object_has_member(object, "start"))
      {
        timeStr = json_object_get_string_member(object, "start");
        startUtc = nvds_c2d_str_to_second(timeStr);
        if (startUtc < 0)
        {
          g_printerr("Error while parsing JSON message. Failed to parse 'start' "
                     "time - %s, utc sec %ld\n",
                     timeStr, startUtc);
          g_object_unref(parser);
          return false;
        }
        curUtc = nvds_get_current_utc_time();
        start = curUtc - startUtc;
        if (start < 0)
        {
          start = 0;
          g_printerr("start time is in future, setting it to current time\n");
        }
      }
      else
      {
        g_printerr("Error while parsing JSON message.  missing 'start' field.\n");
        g_object_unref(parser);
        return false;
      }

      if (json_object_has_member(object, "end"))
      {
        timeStr = json_object_get_string_member(object, "end");
        endUtc = nvds_c2d_str_to_second(timeStr);
        if (endUtc < 0)
        {
          g_printerr("Error while parsing JSON message. Failed to parse 'end' "
                     "time - %s\n",
                     timeStr);
          g_object_unref(parser);
          return false;
        }
        duration = endUtc - startUtc;
        if (duration < 0)
        {
          g_printerr("Negative duration (%d), setting it to zero\n", duration);
          duration = 0;
        }
      }
      else
      {
        // Duration is not specified that means stop event will be received
        // later.
        duration = 0;
      }
    }
  }
  else
  {
    g_printerr("Error while parsing JSON message. No JSON object.\n");
    g_object_unref(parser);
    return false;
  }

  auto result = sensor_name_id_map_.find(sensorStr);
  if (result == sensor_name_id_map_.end())
  {
    g_printerr("Could not find sensor with name %s\n", sensorStr.c_str());
    return false;
  }
  int64_t source_id = result->second;
  g_object_unref(parser);

  NvDsSRSessionId sessId = -1;

  for (auto &handler : handlers_)
  {
    ISmartRecordingController *sr_controller = dynamic_cast<ISmartRecordingController *>(
        handler);
    // smart recording actions
    if (sr_controller)
    {
      if (startRec)
      {
        g_print("Starting smart record for source %ld: %s\n", source_id, sensorStr.c_str());
        sr_controller->startSmartRecord(source_id, &sessId, start, duration, nullptr);
        if (sessId != static_cast<NvDsSRSessionId>(-1))
        {
          sensor_sr_session_id_map_[sensorStr] = sessId;
        }
      }
      else
      {
        auto result = sensor_sr_session_id_map_.find(sensorStr);
        if (result != sensor_sr_session_id_map_.end())
          sessId = result->second;
        g_print("Stoping smart record for %ld\n", source_id);
        sr_controller->stopSmartRecord(source_id, sessId);
      }
    }
  }

  return true;
}