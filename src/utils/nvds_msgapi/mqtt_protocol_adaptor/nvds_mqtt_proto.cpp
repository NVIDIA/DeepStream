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

#include "nvds_mqtt_proto.h"

#include "nvds_logger.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <thread>
#include <unistd.h>
#include <openssl/sha.h>
#include <glib.h>

using namespace std;


#define UNUSED(A) (void)(A)
#define NVDS_MQTT_LOG_CAT "DSLOG:NVDS_MQTT_PROTO"
#define CONFIG_GROUP_MSG_BROKER_MQTT_USERNAME_CFG "username"
#define CONFIG_GROUP_MSG_BROKER_MQTT_PASSWORD_CFG "password"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CLIENT_ID_CFG "client-id"
#define CONFIG_GROUP_MSG_BROKER_MQTT_LOOP_TIMEOUT_CFG "loop-timeout"
#define CONFIG_GROUP_MSG_BROKER_MQTT_KEEP_ALIVE_CFG "keep-alive"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CAFILE_CFG "tls-cafile"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CAPATH_CFG "tls-capath"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CERTFILE_CFG "tls-certfile"
#define CONFIG_GROUP_MSG_BROKER_MQTT_KEYFILE_CFG "tls-keyfile"

#define NVDS_MSGAPI_VERSION "4.0"

// Note: The NVDS_MSGAPI_PROTOCOL name "MQTT" is hard-coded for a check in the nvmsgbroker lib
// See /opt/nvidia/deepstream/deepstream/sources/libs/nvmsgbroker/nvmsgbroker.cpp
#define NVDS_MSGAPI_PROTOCOL "MQTT"

#define PASSWORD_VAR "PASSWORD_MQTT"
#define USERNAME_VAR "USER_MQTT"

// Helper function to extract sensor ID from protobuf payload

ExtractedData extract_data_from_payload(const uint8_t *payload, size_t nbuf) {
    ExtractedData result = {std::string(""), std::string("")};

    // Check if extraction is enabled via environment variable
    const char* enable_extraction = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
    if (enable_extraction == NULL || strcmp(enable_extraction, "1") != 0) {
        // If not enabled, return empty strings
        return result;
    }

    // Try to extract data from protobuf payload
    if (payload != NULL && nbuf > 0) {
        const uint8_t *data = payload;
        size_t i = 0;

        while (i < nbuf - 1) {
            uint8_t tag = data[i];
            i++;

            // Parse varint length for string fields
            uint32_t length = 0;
            uint32_t shift = 0;
            while (i < nbuf && (data[i] & 0x80)) {
                length |= (data[i] & 0x7F) << shift;
                shift += 7;
                i++;
            }
            if (i < nbuf) {
                length |= (data[i] & 0x7F) << shift;
                i++;
            }

            // Check if we have enough data for the string
            if (i + length > nbuf) {
                break;
            }

            // Extract field number and wire type from tag
            uint32_t field_number = tag >> 3;
            uint32_t wire_type = tag & 0x07;

            if (wire_type == 2) { // String field
                std::string field_value((char*)&data[i], length);

                // Field 4: sensorId (from DeepStream protobuf schema)
                if (field_number == 4) {
                    // Validate it looks like a sensor ID
                    if (field_value.find("sensor") != std::string::npos || 
                        field_value.find("bev") != std::string::npos ||
                        field_value.find("camera") != std::string::npos ||
                        field_value.find("0") != std::string::npos ||
                        field_value.length() > 0) {
                        result.sensor_id = field_value;
                        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Extracted sensor_id: %s", field_value.c_str());
                    }
                }
                // Note: We only extract sensorId from metadata, timestamp is always current system time
            }

            i += length;
        }
    }

    return result;
}

// Helper function to create protobuf wrapper
std::string create_protobuf_wrapper(const uint8_t *payload, size_t nbuf) {
    // Check if DeepStream format is enabled
    const char* enable_deepstream_format = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
    if (enable_deepstream_format == NULL || strcmp(enable_deepstream_format, "1") != 0) {
        // If not enabled, return original payload
        return std::string((char*)payload, nbuf);
    }

    // Extract sensor ID from payload
    ExtractedData extracted_data = extract_data_from_payload(payload, nbuf);
    std::string sensor_id = extracted_data.sensor_id;

    // Always use current system timestamp
    time_t now = time(0);
    char current_time[64];
    struct tm tm_buf;
    strftime(current_time, sizeof(current_time), "%Y-%m-%dT%H:%M:%SZ", gmtime_r(&now, &tm_buf));
    std::string timestamp_str = std::string(current_time);

    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Creating protobuf wrapper with sensor_id: %s, current_timestamp: %s", 
             sensor_id.c_str(), timestamp_str.c_str());

    // Create protobuf wrapper structure
    ProtobufWrapper wrapper;
    wrapper.payload = std::string((char*)payload, nbuf);  // Direct binary data, no Base64 encoding needed
    wrapper.key = sensor_id;
    wrapper.timestamp = timestamp_str;

    // Serialize the wrapper to protobuf format
    return serialize_protobuf_wrapper(wrapper);
}

// Helper function to serialize ProtobufWrapper to protobuf format
std::string serialize_protobuf_wrapper(const ProtobufWrapper& wrapper) {
    // Create a simple protobuf-like structure using manual serialization
    // This creates a protobuf message with the following structure:
    // message MqttWrapper {
    //   bytes payload = 1;     // Raw original protobuf binary data (no encoding needed)
    //   string key = 2;        // sensorId extracted from metadata
    //   string timestamp = 3;  // current system timestamp when data was sent
    // }

    std::string result;

    // Field 1: payload (bytes)
    if (!wrapper.payload.empty()) {
        // Protobuf wire format: tag (field_number << 3 | wire_type)
        // For bytes: field_number=1, wire_type=2 (length-delimited)
        result += static_cast<char>(0x0A); // tag: (1 << 3) | 2 = 10 = 0x0A

        // Length of the binary data
        size_t payload_len = wrapper.payload.length();
        if (payload_len < 128) {
            result += static_cast<char>(payload_len);
        } else {
            // For longer data, use varint encoding
            while (payload_len >= 128) {
                result += static_cast<char>((payload_len & 0x7F) | 0x80);
                payload_len >>= 7;
            }
            result += static_cast<char>(payload_len);
        }

        // The actual binary data
        result += wrapper.payload;
    }

    // Field 2: key (string)
    if (!wrapper.key.empty()) {
        result += static_cast<char>(0x12); // tag: (2 << 3) | 2 = 18 = 0x12

        size_t key_len = wrapper.key.length();
        if (key_len < 128) {
            result += static_cast<char>(key_len);
        } else {
            while (key_len >= 128) {
                result += static_cast<char>((key_len & 0x7F) | 0x80);
                key_len >>= 7;
            }
            result += static_cast<char>(key_len);
        }

        result += wrapper.key;
    }

    // Field 3: timestamp (string)
    if (!wrapper.timestamp.empty()) {
        result += static_cast<char>(0x1A); // tag: (3 << 3) | 2 = 26 = 0x1A

        size_t timestamp_len = wrapper.timestamp.length();
        if (timestamp_len < 128) {
            result += static_cast<char>(timestamp_len);
        } else {
            while (timestamp_len >= 128) {
                result += static_cast<char>((timestamp_len & 0x7F) | 0x80);
                timestamp_len >>= 7;
            }
            result += static_cast<char>(timestamp_len);
        }

        result += wrapper.timestamp;
    }

    return result;
}

// Helper function to create enhanced payload (now returns protobuf instead of JSON)
std::string create_enhanced_payload(const uint8_t *payload, size_t nbuf) {
    return create_protobuf_wrapper(payload, nbuf);
}

/* Parse the config file into the mosquitto handle
*/
NvDsMsgApiErrorType nvds_mqtt_read_config(NvDsMqttClientHandle *mh, char *config_path)
{
    char cfg_uname[MAX_FIELD_LEN]="", cfg_pwd[MAX_FIELD_LEN]="", cfg_client_id[MAX_FIELD_LEN]="";
    char cfg_timeout[MAX_FIELD_LEN]="", cfg_keep_alive[MAX_FIELD_LEN]="";

    const gchar* var_user = NULL;
    const gchar* var_pwd = NULL;

    // Read config to fetch username -- to be deprecated in favor of env variable
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_USERNAME_CFG, cfg_uname, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_uname,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->username, cfg_uname, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT username = %s \n", mh->username);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Username not specified in cfg.\n");
    }
    // Read config to fetch password -- to be deprecated in favor of env variable
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_PASSWORD_CFG, cfg_pwd, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_pwd,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->password, cfg_pwd, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT password = %s \n", mh->password);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Password not specified in cfg.\n");
    }
    // Read config to fetch client ID
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CLIENT_ID_CFG, cfg_client_id, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_client_id,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->client_id, cfg_client_id, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT Client ID = %s \n", mh->client_id);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Client ID not specified in cfg.\n");
    }
    // Read config to fetch loop_timeout value. Uses default if not present.
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_LOOP_TIMEOUT_CFG, cfg_timeout, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_timeout,"", MAX_FIELD_LEN)) {
        string timeout(cfg_timeout);
        try {
          mh->loop_timeout = stoi(timeout);
        } catch (const std::invalid_argument & e) {
          mh->loop_timeout = DEFAULT_LOOP_TIMEOUT;
          g_print("parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
        }
        catch (const std::out_of_range & e) {
          mh->loop_timeout = DEFAULT_LOOP_TIMEOUT;
          g_print("parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
        }
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Loop timeout = %d \n", mh->loop_timeout);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Loop timeout not specified in cfg. Using default of %d\n", mh->loop_timeout);
    }
    // Read config to fetch keep-alive value. Uses default if not present.
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_KEEP_ALIVE_CFG, cfg_keep_alive, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_keep_alive,"", MAX_FIELD_LEN)) {
        string keep_alive(cfg_keep_alive);
        try {
          mh->keep_alive = stoi(keep_alive);
        } catch (const std::invalid_argument & e) {
          mh->keep_alive = DEFAULT_KEEP_ALIVE;
          g_print("parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
        }
        catch (const std::out_of_range & e) {
          mh->keep_alive = DEFAULT_KEEP_ALIVE;
          g_print("parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
        }
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "keep_alive = %d \n", mh->keep_alive);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Keep alive interval not specified in cfg. Using default of %d\n", mh->keep_alive);
    }
    // Read config file to check if TLS is enabled
    char enable_tls[16]="";
    if(fetch_config_value(config_path, "enable-tls", enable_tls, sizeof(enable_tls), NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) {
        if (strncmp(enable_tls, "1", 1) == 0) {
          mh->enable_tls = true;
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "TLS encryption enabled\n");
        }
        else {
          mh->enable_tls = false;
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "TLS encryption disabled\n");
        }
    }
    else {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "parse_config: Error parsing mqtt enable-tls config option. Disabled by default.\n");
    }
    // If TLS is enabled, read config to fetch TLS CA, cert, and key file paths
    if (mh->enable_tls) {
      char cfg_cafile[MAX_FIELD_LEN]="", cfg_capath[MAX_FIELD_LEN]="", cfg_certfile[MAX_FIELD_LEN]="", cfg_keyfile[MAX_FIELD_LEN]="";
      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CAFILE_CFG, cfg_cafile, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_cafile,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->cafile, cfg_cafile, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS CA certificate filepath = %s \n", mh->cafile);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "CA certificate filepath is not provided\n");
      }

      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CAPATH_CFG, cfg_capath, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_capath,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->capath, cfg_capath, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS CA certificate directory path = %s \n", mh->capath);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "CA certificate directory path is not provided\n");
      }

      if (strncmp(mh->cafile,"", MAX_FIELD_LEN) == 0 && strncmp(mh->capath,"", MAX_FIELD_LEN) == 0) {
        g_print("parse_config: Error: TLS encyption is enabled but neither CA certificate filepath nor directory path were provided\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "parse_config: Error: TLS encyption is enabled but neither CA certificate filepath nor directory path were provided\n");
        return NVDS_MSGAPI_ERR;
      }

      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CERTFILE_CFG, cfg_certfile, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_certfile,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->certfile, cfg_certfile, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client certificate filepath = %s \n", mh->certfile);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client certificate not provided.\n");
      }

      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_KEYFILE_CFG, cfg_keyfile, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_keyfile,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->keyfile, cfg_keyfile, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client key filepath = %s \n", mh->keyfile);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client key not provided.\n");
      }
    }

    // Fetch username and password from environmental variable
    var_user = g_getenv(USERNAME_VAR);
    var_pwd = g_getenv(PASSWORD_VAR);
    // Check that username and password do not exceed maximum length
    // Override user and password with environmental variable if given
    if (var_user != NULL) {
      if (MAX_FIELD_LEN < (int) strlen(var_user) + 1) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Username length exceeds capacity of %d", MAX_FIELD_LEN);
        return NVDS_MSGAPI_ERR;
      }
      else{
        g_strlcpy(mh->username, var_user, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT username = %s \n", mh->username);
      }
    }
    else {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Username not provided through environmental variable.\n");
    }

    if (var_pwd != NULL) {
      if(MAX_FIELD_LEN < (int) strlen(var_pwd) + 1) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Password length exceeds capacity of %d \n", MAX_FIELD_LEN);
        return NVDS_MSGAPI_ERR;
      }
      else{
        g_strlcpy(mh->password, var_pwd, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT password = %s \n", mh->password);
      }
    }
    else {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Password not provided through environmental variable.\n");
    }

  // Read config file to check for set_threaded
  char set_threaded[MAX_FIELD_LEN]="";
  if(fetch_config_value(config_path, "set-threaded", set_threaded, sizeof(set_threaded), NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK && strncmp(set_threaded,"", MAX_FIELD_LEN)) {
      if (strncmp(set_threaded, "1", 1) == 0) {
        mh->set_threaded = true;
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Threaded set in mqtt_protocol adaptor. Use with nvmsgbroker plugin or do_work loop implementation.\n");
      }
      else {
        mh->set_threaded = false;
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Threaded not set in mqtt_protocol_adaptor. Use with mosquitto_loop_start().\n");
      }
  }
  else {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO , "parse_config: Error parsing mqtt set-threaded config option. Enabled by default.\n");
  }

  return NVDS_MSGAPI_OK;
}

/* Mosquitto log callback
*/
void mosq_mqtt_log_callback(struct mosquitto *mosq, void *obj, int level, const char *str)
{
  UNUSED(mosq);
  UNUSED(obj);
  UNUSED(level);

  // g_print("[%s] %s\n", __func__, str);
  // nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "[%s] %s\n", __func__, str);
}

/* Mosquitto disconnect callback, called when disconnect is received from the broker
*/
void my_disconnect_callback(struct mosquitto *mosq, void *obj, int rc, const mosquitto_property *properties)
{
  UNUSED(mosq);
  UNUSED(obj);
  UNUSED(rc);
  UNUSED(properties);

  if(rc == 0){
    g_print("mqtt disconnected\n");
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt disconnected\n");
  }
  else {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt disconnect failed rc=%d\n", rc);
  }
}

/* Mosquitto connect callback, called when CONNACK is recieved from the broker
*/
void my_connect_callback(struct mosquitto *mosq, void *obj, int result, int flags, const mosquitto_property *properties)
{
  UNUSED(obj);
  UNUSED(flags);
  UNUSED(properties);
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)mosquitto_userdata(mosq);

  // When connection succeeds, call the user connect callback with success flag
  if(!result){
    g_print("mqtt connection success; ready to send data\n");
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt connection success; ready to send data\n");
    if (mqttH->connect_cb)
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SUCCESS);
  // If connection failed, call the user connect callback with failure flag
  }else{
    if(result){
      { //we are always using MQTT v5
        if(result == MQTT_RC_UNSUPPORTED_PROTOCOL_VERSION){
          g_print("Connection error: %s. Try connecting to an MQTT v5 broker; Other versions are not supported\n", mosquitto_reason_string(result));
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Connection error: %s. Try connecting to an MQTT v5 broker; Other versions are not supported\n", mosquitto_reason_string(result));
        }else{
          g_print("Connection error: %s\n", mosquitto_reason_string(result));
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Connection error: %s\n", mosquitto_reason_string(result));
        }
        if (mqttH->connect_cb)
          mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
      }
    }
  }
}

/* Mosquitto publish callback, called on PUBACK from broker
*/
void my_publish_callback(struct mosquitto *mosq, void *obj, int mid, int reason_code, const mosquitto_property *properties)
{
  UNUSED(obj);
  UNUSED(properties);

  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)mosquitto_userdata(mosq);

  nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Publish callback with reason code: %s.\n", mosquitto_reason_string(reason_code));

  nvds_msgapi_send_cb_t send_callback = NULL;
  void* user_ptr = NULL;
  // Retrieve the message info corresponding to the message id
  // Retrieve the message info corresponding to the message id
  std::unique_lock<std::mutex> mapLock(mqttH->map_lock);
  std::unordered_map<int , send_msg_info_t>::iterator it = mqttH->send_msg_info_map.find(mid);
  if(it != mqttH->send_msg_info_map.end()) {
    send_msg_info_t& msg_info = it->second;
    send_callback = msg_info.send_callback;
    user_ptr = msg_info.user_ptr;
    // Remove the msg_info from the map
    mqttH->send_msg_info_map.erase(it);
  }
  mapLock.unlock();

  if (send_callback) {
    // Call the user send callback function associated with the message ID to communicate publish status
    //MQTTv5 reason code check:
    if(reason_code > 127){
      g_print("Warning: Publish %d failed: %s.\n", mid, mosquitto_reason_string(reason_code));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Warning: Publish %d failed: %s.\n", mid, mosquitto_reason_string(reason_code));
      send_callback(user_ptr, NVDS_MSGAPI_ERR);
    } else {
      send_callback(user_ptr, NVDS_MSGAPI_OK);
    }
  }

}

/* Mosquitto message callback, called when a message is received from the broker
*/
static void my_message_callback(struct mosquitto *mosq, void *obj, const struct mosquitto_message *message, const mosquitto_property *properties)
{
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)mosquitto_userdata(mosq);
  // Call the user sub callback with the message and user pointer
  if(mqttH->sub_callback) {
    mqttH->sub_callback(NVDS_MSGAPI_OK, message->payload,message->payloadlen, message->topic, mqttH->user_ctx);
  }
}

/* Mosquitto subscribe callback, called when SUBACK is received from the broker
*/
static void my_subscribe_callback(struct mosquitto *mosq, void *obj, int mid, int qos_count, const int *granted_qos, const mosquitto_property *properties)
{
  int i;
  UNUSED(obj);
  for(i=0; i<qos_count; i++){
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Subscription granted qos %d", granted_qos[i]);
  }
  return;
}

/*
 * Validate mqtt connection string format
 * Valid format host;port or (host;port;topic to support backward compatibility)
 */
bool is_valid_mqtt_connection_str(char *connection_str, std::string &burl, std::string &bport) {
  if(connection_str == NULL) {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection string cant be NULL");
    return false;
  }

  std::string str(connection_str);
  size_t n = count(str.begin(), str.end(), ';');
  if(n>2) {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "mqtt connection string format is invalid");
    return false;
  }

  istringstream iss(connection_str);
  getline(iss, burl, ';');
  getline(iss, bport,';');

  if(burl =="" || bport == "") {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection string is invalid. hostname or port is empty\n");
      return false;
  }
  return true;
}

/* nvds_msgapi function for connect to broker
*/
NvDsMsgApiHandle nvds_msgapi_connect(char *connection_str,
        nvds_msgapi_connect_cb_t connect_cb, char *config_path)
{
  int rc;
  std::string burl="", bport="";
  int port = 0;
  char* err;

  // Retrieve host and port from connection string
  if(!is_valid_mqtt_connection_str(connection_str, burl, bport))
    return NULL;

  try {
    port = stoi(bport);
  } catch (const std::invalid_argument & e) {
    g_print("stoi exception %s\n", e.what());
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "stoi exception %s\n", e.what());
  }
  catch (const std::out_of_range & e) {
    g_print("stoi exception %s\n", e.what());
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "stoi exception %s\n", e.what());
  }

  if(port <= 0)
  {
    return NULL;
  }

  // Mosquitto library initialization, must be called before any other mosquitto function
  mosquitto_lib_init();
  // Create handle
  NvDsMqttClientHandle* mqttH = new NvDsMqttClientHandle;
  g_strlcpy(mqttH->connection_str, connection_str, MAX_FIELD_LEN);
  if (strlen(connection_str) >= MAX_FIELD_LEN) {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Warning: length of connection string is longer than maximum length stored in the client handle. Only up to %d characters are stored.\n", MAX_FIELD_LEN);
  }

  // Set user connect callback
  mqttH->connect_cb = connect_cb;
  // Read config
  if (config_path) {
    if(nvds_mqtt_read_config(mqttH, config_path) == NVDS_MSGAPI_ERR) {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "MQTT config parsing failed. Connection failed \n");
      delete mqttH;
      return NULL;
    }
  }
  if (mqttH->client_id[0] != 0) // If client ID is set
    mqttH->mosq = mosquitto_new(mqttH->client_id, true, NULL); // Instantiate client
  else
    mqttH->mosq = mosquitto_new(NULL, true, NULL); // Instantiate client with NULL client ID -- random client ID will be generated
  if(!mqttH->mosq){
    switch(errno){
      case ENOMEM:
        g_print("Error: Out of memory.\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Out of memory.\n");
        break;
      case EINVAL:
        g_print("Error: Invalid id.\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Invalid id.\n");
        break;
    }
    delete mqttH;
    return NULL;
  }

  // Set TLS if enabled
  if (mqttH->enable_tls) {
    // Set cafile and capath to null if empty
    char* cafile = (mqttH->cafile[0] != 0) ? (char*)mqttH->cafile : NULL;
    char* capath = (mqttH->capath[0] != 0) ? (char*)mqttH->capath : NULL;
    // certfile and keyfile must both be provided or both be NULL
    char* certfile = (mqttH->certfile[0] != 0 && mqttH->keyfile[0] != 0) ? (char*)mqttH->certfile : NULL;
    char* keyfile = (mqttH->certfile[0] != 0 && mqttH->keyfile[0] != 0) ? (char*)mqttH->keyfile : NULL;
    rc = mosquitto_tls_set(mqttH->mosq, cafile, capath, certfile, keyfile, NULL);
    if(rc){
      if(rc == MOSQ_ERR_INVAL){
        g_print("Error: Problem setting TLS options: File not found.\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Problem setting TLS options: File not found.\n");
			}
      else{
        g_print("Error: Problem setting TLS options: %s.\n", mosquitto_strerror(rc));
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Problem setting TLS options: %s.\n", mosquitto_strerror(rc));
      }
      delete mqttH;
      return NULL;
    }
  }

  if (mqttH->set_threaded)
    mosquitto_threaded_set(mqttH->mosq, true); // Must be set to avoid race conditions if using with nvmsgbroker plugin or managing own do_work calls. Do not set if using mosquitto_loop_start in app.

  mosquitto_user_data_set(mqttH->mosq, (void*)mqttH); // Set user data so handle can be retrieved in mosquitto callbacks
  // Set mosquitto callbacks
  mosquitto_log_callback_set(mqttH->mosq, mosq_mqtt_log_callback);
  mosquitto_connect_v5_callback_set(mqttH->mosq, my_connect_callback);
  mosquitto_disconnect_v5_callback_set(mqttH->mosq, my_disconnect_callback);
  mosquitto_publish_v5_callback_set(mqttH->mosq, my_publish_callback);
  mosquitto_subscribe_v5_callback_set(mqttH->mosq, my_subscribe_callback);
  mosquitto_message_v5_callback_set(mqttH->mosq, my_message_callback);

  // Authenticate with user credentials if provided
  if (strncmp(mqttH->username,"", MAX_FIELD_LEN) || strncmp(mqttH->password, "", MAX_FIELD_LEN)) {
    if(mosquitto_username_pw_set(mqttH->mosq, mqttH->username, mqttH->password) == MOSQ_ERR_SUCCESS) {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Successfully set MQTT username = %s and password = %s\n", mqttH->username, mqttH->password);
    }
  }
  // Call connect
  rc = mosquitto_connect_bind_v5(mqttH->mosq, burl.c_str(), port, mqttH->keep_alive, NULL, NULL);
  if(rc>0){
    printf("error=%d\n", rc);
    if(rc == MOSQ_ERR_ERRNO){
      err = strerror(errno);
      g_print("Error: %s\n", err);
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "Error: %s\n", err);

    }else{
      g_print("Unable to connect (%s).\n", mosquitto_strerror(rc));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "Unable to connect (%s).\n", mosquitto_strerror(rc));
    }
    if (mqttH->connect_cb)
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    delete mqttH;
    return NULL;
  }

  // If not using mosquitto_loop_start, manually call mosquitto_loop to send connect right away.
  if (mqttH->set_threaded)
    rc = mosquitto_loop(mqttH->mosq, mqttH->loop_timeout /*< ms select timeout */, 1);
  if (mqttH->connect_cb && (rc != MOSQ_ERR_SUCCESS)) {
    mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    g_print("Connect loop rc: %s", mosquitto_strerror(rc));
    delete mqttH;
    return NULL;
  }

  return (NvDsMsgApiHandle)mqttH;
}

/* nvds_msgapi function for synchronous send, not supported for MQTT
*/
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf)
{
  g_print("nvds_msgapi_send() is not currently supported for MQTT.");
  nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "nvds_msgapi_send() is not currently supported for MQTT.");
  return NVDS_MSGAPI_ERR;
}

/* nvds_msgapi function for asynchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char  *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr)
{
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)(h_ptr);
  int rc = MOSQ_ERR_SUCCESS;

  // Create msg_info from the user send callback and pointer
  struct send_msg_info_t msg_info = {send_callback, user_ptr};

  // message id, will be assigned by call to publish
  int mid;

  if(mqttH == NULL)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection handle passed for nvds_msgapi_send_async() = NULL. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }
  if(!topic)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Topic not specified for send. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }
  if(!payload || nbuf <= 0)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Message not specified for send. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }
  if(nbuf <= 0)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Invalid message size (<=0) for send. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }

  // Create enhanced payload if environment variable is enabled
  std::string enhanced_payload = create_enhanced_payload(payload, nbuf);

  // Publish with enhanced payload (no properties needed)
  rc = mosquitto_publish_v5(mqttH->mosq, &mid, topic, enhanced_payload.length(), (uint8_t*)enhanced_payload.c_str(), 0, false, NULL);

  if(rc != MOSQ_ERR_SUCCESS){
    g_print("Error sending repeat publish: %s", mosquitto_strerror(rc));
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error sending repeat publish: %s\n", mosquitto_strerror(rc));
    if (mqttH->connect_cb && rc == MOSQ_ERR_NO_CONN) {
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    }
  }
  // Map msg_info to message id
  else {
    std::unique_lock<std::mutex> mapLock(mqttH->map_lock);
    mqttH->send_msg_info_map[mid] = msg_info;
    mapLock.unlock();
  }

  // No properties to clean up - using enhanced payload instead

  return (rc == MOSQ_ERR_SUCCESS) ? NVDS_MSGAPI_OK : NVDS_MSGAPI_ERR;
}

/* nvds_msgapi function for subscribe to topics
*/
NvDsMsgApiErrorType nvds_msgapi_subscribe (NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx)
{
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle *)h_ptr;
    if(mqttH == NULL ) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection handle passed for nvds_msgapi_subscribe() = NULL. Subscribe failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(topics == NULL || num_topics <=0 ) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Topics not specified for subscribe. Subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(!cb) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Subscribe callback cannot be NULL. Subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(mqttH->subscription_on) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "MQTT Subscription already exists. Ignoring this subscription call\n\n");
        return NVDS_MSGAPI_ERR;
    }

  // Set user callback and pointer
  mqttH->sub_callback = cb;
  mqttH->user_ctx = user_ctx;

  // Subscribe to each topic
  for (int i = 0 ; i < num_topics ; i++){
    int rc;
    // Subscribe with QoS 0 and default options
    rc = mosquitto_subscribe_v5(mqttH->mosq, NULL, topics[i], 0, 0, NULL);
    if(rc != MOSQ_ERR_SUCCESS) {
      g_print("Error subscribing: %s", mosquitto_strerror(rc));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt: Failed to subscribe to topic %s. Error: %s\n", std::string(topics[i]).c_str(), mosquitto_strerror(rc));
      if (mqttH->connect_cb && (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST)) {
          mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
      }
      return NVDS_MSGAPI_ERR;
    }

    // If not using mosquitto_loop_start, manually call mosquitto_loop to send subscribe right away
    if (mqttH->set_threaded)
      rc = mosquitto_loop(mqttH->mosq, mqttH->loop_timeout /*< ms select timeout */, 1);
    if (mqttH->connect_cb && rc == MOSQ_ERR_NO_CONN) {
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    }
  }

  mqttH->subscription_on=true;
  return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for performing work on thread in nvmsgbroker plugin. In the case of mqtt, we call mosquitto_loop to send outgoing messages
* and read incoming messages
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr)
{
  int rc;
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)(h_ptr);
  // mosquitto_loop is called to process incoming data and send outgoing messages
  rc = mosquitto_loop(mqttH->mosq, mqttH->loop_timeout, 1);
  if (mqttH->connect_cb && (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST)) {
    mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
  }
}

/* nvds_msgapi function for disconnect from broker
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr)
{
  int rc;
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)(h_ptr);
  // Call disconnect
  rc = mosquitto_disconnect_v5(mqttH->mosq, 0, NULL);
  if(rc>0){
    g_print("Error disconnecting: (%s).\n", mosquitto_strerror(rc));
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error disconnecting: (%s).\n", mosquitto_strerror(rc));
  }

  // If not using mosquitto_loop_start, manually call mosquitto_loop to send disconnect
  if (mqttH->set_threaded)
    mosquitto_loop(mqttH->mosq, mqttH->loop_timeout, 1);
  // Standard mosquitto clean-up procedure

  mqttH->disconnect = true;
  mosquitto_destroy(mqttH->mosq);
  mosquitto_lib_cleanup();
  delete mqttH;
  return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for retrieving nvds_msgapi version
*/
char *nvds_msgapi_getversion() {
  return (char *) NVDS_MSGAPI_VERSION;
}

/* nvds_msgapi function for retrieving protocol name
*/
char *nvds_msgapi_get_protocol_name(void) {
  return (char *)NVDS_MSGAPI_PROTOCOL;
}

/* nvds_msgapi function for retrieving connection signature, used for connection sharing
*/
NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str, char *cfg, char *output_str, int max_len) {
  g_strlcpy(output_str,"", max_len);

    //check if share-connection config option is turned ON
    char reuse_connection[16]="";
    if(fetch_config_value(cfg, "share-connection", reuse_connection, sizeof(reuse_connection), NVDS_MQTT_LOG_CAT) != NVDS_MSGAPI_OK) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing mqtt share-connection config option");
        return NVDS_MSGAPI_ERR;
    }
    if(strncmp(reuse_connection, "1", 1)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "nvds_msgapi_connection_signature: mqtt connection sharing disabled. Hence connection signature cant be returned");
        return NVDS_MSGAPI_OK;
    }

    if(broker_str == NULL && cfg == NULL) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Both mqtt broker_str and path to cfg cant be NULL");
        return NVDS_MSGAPI_ERR;
    }
    if(max_len < 2 * SHA256_DIGEST_LENGTH + 1) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: output string length allocated not sufficient");
        return NVDS_MSGAPI_ERR;
    }

    char mqtt_connection_str[MAX_FIELD_LEN]="";
    char uname[MAX_FIELD_LEN]="", pwd[MAX_FIELD_LEN]="";

    std::string burl="", bport="";
    if(!is_valid_mqtt_connection_str(broker_str, burl, bport)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Invalid connection string");
        return NVDS_MSGAPI_ERR;
    }

    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_MQTT_USERNAME_CFG, uname, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) != NVDS_MSGAPI_OK)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing mqtt username");
    }

    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_MQTT_PASSWORD_CFG, pwd, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) != NVDS_MSGAPI_OK)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing mqtt password");
    }

    // Generate connection signature from connection string and user credentials
    g_snprintf(mqtt_connection_str, sizeof (mqtt_connection_str), "%s;%s;%s;%s", burl.c_str(),bport.c_str(),uname,pwd);
    std::string hashval = generate_sha256_hash(std::string(mqtt_connection_str, max_len));
    g_strlcpy(output_str, hashval.c_str(), max_len);
    return NVDS_MSGAPI_OK;
}
