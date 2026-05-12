/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "mqtt_protocol.h"
#include "mosquitto.h"

#include <mutex>
#include <unordered_map>
#include <errno.h>
#include <syslog.h>
#include "glib.h"
#include <string.h>
#include <string>
#include "nvds_msgapi.h"
#include "nvds_utils.h"

using namespace std;

#define MAX_FIELD_LEN 1024

// Structure to hold extracted data from protobuf payload
struct ExtractedData {
    std::string sensor_id;  // Extracted from metadata
    std::string timestamp;  // Not used - timestamp is always current system time
};

// Structure to hold protobuf wrapper message
struct ProtobufWrapper {
    std::string payload;        // Raw protobuf binary data (no encoding needed)
    std::string key;           // sensorId extracted from metadata
    std::string timestamp;     // current system timestamp when data was sent
};
#define DEFAULT_LOOP_TIMEOUT 2000
#define DEFAULT_KEEP_ALIVE 60
/* Message details:
 * send_callback = user callback func
 * user_ptr = user pointer passed by async send
 */
struct send_msg_info_t {
  nvds_msgapi_send_cb_t send_callback;
  void *user_ptr;
};

/* Details of mqtt connection handle:
 * mosq : mosquitto client object
 * sub_callback : user subscription callback func
 * connect_cb : user connection callback func
 * user_ctx : user pointer passed by sub
 * username: username for login to server
 * password: password for login to server
 * client_id: name of MQTT client
 * loop_timeout : time in ms for the call to loop to wait for network activity
 * keep_alive : number of seconds after which broker should send PING if no messages have been exchanged
 * subscription_on : Flag to check if subscription is ON
 * send_msg_info_map : map message info to id assigned by mosquitto broker
 * map_lock : mutex lock for accessing above map
 * enable_tls : flag to check if TLS encryption is enabled by the broker
 * cafile : path to a TLS certificate authority file
 * capath : path to a directory containing TLS CA files
 * certfile : path to the client TLS certificate file
 * keyfile : path to the client TLS key file
 * disconnect : bool for checking if disconnect has been called
 * set_threaded : bool for setting mosquitto_threaded_set in proto adaptor
 */
typedef struct {
   struct mosquitto *mosq = NULL;
   nvds_msgapi_subscribe_request_cb_t  sub_callback;
   nvds_msgapi_connect_cb_t connect_cb;
   void* user_ctx;
   char connection_str[MAX_FIELD_LEN] = {0};
   char username[MAX_FIELD_LEN] = {0};
   char password[MAX_FIELD_LEN] = {0};
   char client_id[MAX_FIELD_LEN] = {0};
   int loop_timeout = DEFAULT_LOOP_TIMEOUT;
   int keep_alive = DEFAULT_KEEP_ALIVE;
   bool subscription_on = false;
   std::unordered_map<int , send_msg_info_t> send_msg_info_map;
   std::mutex map_lock;
   bool enable_tls = false;
   char cafile[MAX_FIELD_LEN] = {0};
   char capath[MAX_FIELD_LEN] = {0};
   char certfile[MAX_FIELD_LEN] = {0};
   char keyfile[MAX_FIELD_LEN] = {0};
   bool disconnect = false;
   bool set_threaded = true;
} NvDsMqttClientHandle;

NvDsMsgApiErrorType nvds_mqtt_read_config(NvDsMqttClientHandle *mh, char *config_path);
void mosq_mqtt_log_callback(struct mosquitto *mosq, void *obj, int level, const char *str);
void my_disconnect_callback(struct mosquitto *mosq, void *obj, int rc, const mosquitto_property *properties);
void my_connect_callback(struct mosquitto *mosq, void *obj, int result, int flags, const mosquitto_property *properties);
void my_publish_callback(struct mosquitto *mosq, void *obj, int mid, int reason_code, const mosquitto_property *properties);
bool is_valid_mqtt_connection_str(char *connection_str, std::string &burl, std::string &bport);

// Function declarations for protobuf wrapper
ExtractedData extract_data_from_payload(const uint8_t *payload, size_t nbuf);
std::string create_protobuf_wrapper(const uint8_t *payload, size_t nbuf);
std::string create_enhanced_payload(const uint8_t *payload, size_t nbuf);
std::string serialize_protobuf_wrapper(const ProtobufWrapper& wrapper);