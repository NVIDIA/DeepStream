/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <openssl/sha.h>
#include <unistd.h>
#include <string.h>
#include <glib.h>
#include "hiredis.h"
#include "nvds_msgapi.h"
#include "nvds_logger.h"
#include "nvds_utils.h"

using namespace std;

#define LOG_CAT "DSLOG:NVDS_REDIS_PROTO"
#define NVDS_MSGAPI_VERSION "1.0"
#define NVDS_MSGAPI_PROTOCOL "REDIS"
#define DFLT_PAYLOAD_KEY "metadata"
#define DFLT_CONSUMER_GRP "redis-consumer-group"
#define DFLT_CONSUMER_NAME "redis-consumer"

#define PASSWORD_VAR "PASSWORD_REDIS"

// Forward declarations
string get_default_sensor_id();
string extract_sensor_id_from_payload(const uint8_t *payload, size_t nbuf);

// Parse connection string
NvDsMsgApiErrorType parse_config(char *str, char *config, string &ip_addr, string &password, string &port, string &stream_size, string &stream_key, string &cgrp, string &cname);
void consumer(NvDsMsgApiHandle h_ptr, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx);

/* send message info
 * msg : payload
 * msg_len : length of payload
 * topic : topicname
 * user_cb_func :  user callback function to notify status
 * user_ctx : user data
 */
typedef struct {
  void *msg;
  size_t msg_len;
  string topic;
  nvds_msgapi_send_cb_t user_cb_func;
  void *user_ctx;
} send_msg_info_t;

/* Details of redis connection handle:
 * host : redis server hostname
 * port : port number
 * c : redis context for producer
 * sc: redis context for subscriber
 * stream_size : Max size of the redis stream
 * payload_key : key name for the redis stream payload
 * consumer_grp : consumer group name
 * consumer_name : consumer name
 * subscribe_thread : thread used for the redis subscriber
 * disconnect : flag used to notify consumer thread disconnect or not
 * subscription: flag to indicate if subscription is on
 * primary_queue : main queue where publisher place msgs
 * secondary_queue : secondary queue used for double buffering
 * publisher_mutex : mutex used by publisher to place outgoing msgs in queue
 */
typedef struct {
  string host;
  int port;
  redisContext *c;
  redisContext *sc;
  string stream_size;
  string payload_key;
  string consumer_grp;
  string consumer_name;
  string sensor_id;  // Add sensor ID for DeepStream format
  thread subscribe_thread;
  unordered_map<string , string> subscribe_topics;
  bool disconnect;
  bool subscription;
  nvds_msgapi_connect_cb_t connect_cb;
  queue<send_msg_info_t> primary_queue;
  queue<send_msg_info_t> secondary_queue;
  mutex publisher_mutex;
} nvds_redis_context_data_t;

// Alternative approach: Use a global default sensor ID that can be set via environment variable
string get_default_sensor_id() {
    const char* env_sensor_id = getenv("DEEPSTREAM_SENSOR_ID");
    if (env_sensor_id != NULL && strlen(env_sensor_id) > 0) {
        return string(env_sensor_id);
    }
    return "unknown-sensor";
}

// Helper function to extract sensor ID from protobuf payload
string extract_sensor_id_from_payload(const uint8_t *payload, size_t nbuf) {
    // Check if sensor ID extraction is enabled via environment variable
    const char* enable_extraction = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
    if (enable_extraction == NULL || strcmp(enable_extraction, "1") != 0) {
        // If not enabled, return default without any extraction
        return get_default_sensor_id();
    }
    
    // Try to get default sensor ID from environment variable first
    string sensor_id = get_default_sensor_id();
    
    // Try to extract sensor ID from protobuf payload
    if (payload != NULL && nbuf > 0) {
        // For DeepStream protobuf messages, the sensor ID is in field 4 (sensorId)
        // We need to parse the protobuf to extract the sensorId field
        
        // Look for protobuf field patterns that indicate sensor ID
        // In protobuf, field 4 (sensorId) is encoded as 0x22 followed by length and data
        // Tag: 0x22 (field 4, wire type 2 = string)
        const uint8_t *data = payload;
        for (size_t i = 0; i < nbuf - 5; i++) {
            // Look for protobuf field tag for sensorId (field 4, wire type 2 = string)
            // Tag: 0x22 (field 4, wire type 2)
            if (data[i] == 0x22) {
                // Next byte should be the length of the string
                if (i + 1 < nbuf) {
                    uint8_t length = data[i + 1];
                    if (length > 0 && length < 50 && i + 2 + length <= nbuf) {
                        // Extract the sensor ID string
                        string extracted_id((char*)&data[i + 2], length);
                        // Validate it looks like a sensor ID
                        if (extracted_id.find("sensor") != string::npos || 
                            extracted_id.find("bev") != string::npos ||
                            extracted_id.find("camera") != string::npos ||
                            extracted_id.find("0") != string::npos ||
                            extracted_id.length() > 0) {
                            sensor_id = extracted_id;
                            nvds_log(LOG_CAT, LOG_DEBUG, "Extracted sensor ID from protobuf field 4: %s", sensor_id.c_str());
                            break;
                        }
                    }
                }
            }
            // Also look for field 1 (version) with tag 0x0A to validate we're in the right message
            else if (data[i] == 0x0A) {
                if (i + 1 < nbuf) {
                    uint8_t length = data[i + 1];
                    if (length > 0 && length < 10 && i + 2 + length <= nbuf) {
                        string version((char*)&data[i + 2], length);
                        if (version.find("4.0") != string::npos || version.find("3.0") != string::npos) {
                            nvds_log(LOG_CAT, LOG_DEBUG, "Found DeepStream Frame message with version: %s", version.c_str());
                        }
                    }
                }
            }
        }
    }
    
    nvds_log(LOG_CAT, LOG_DEBUG, "Using sensor ID: %s", sensor_id.c_str());
    return sensor_id;
}

// Parse connection string
NvDsMsgApiErrorType parse_config(char *str, char *config, string &ip_addr, string &password, string &port, string &stream_size, string &stream_key, string &cgrp, string &cname) {
    string cfg_ipaddr;
    string cfg_port;
    string cfg_pwd;
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    gchar *val = NULL;
    const gchar *var_pwd = NULL;


    // parse config if not null and file length is not 0
    if((config != NULL) && (strlen(config))) {
        GKeyFile *gcfg_file = g_key_file_new();
        if (!g_key_file_load_from_file (gcfg_file, config, G_KEY_FILE_NONE, &error)) {
            nvds_log(LOG_CAT, LOG_ERR , "REDIS adaptor: Failed to parse configfile %s\n", config);
            free_gobjs(gcfg_file, error, keys, val);
        }
        else {
            const char *grpname = "message-broker";
            keys = g_key_file_get_keys(gcfg_file, grpname, NULL, &error);
            for (key = keys; *key; key++) {
                if (!g_strcmp0(*key, "hostname")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "hostname",&error);
                    cfg_ipaddr = string(val);
                }
                else if (!g_strcmp0(*key, "password")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "password", &error);
                    cfg_pwd = string(val);
                }
                else if (!g_strcmp0(*key, "port")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "port", &error);
                    cfg_port = string(val);
                }
                else if (!g_strcmp0(*key, "streamsize")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "streamsize", &error);
                    stream_size = string(val);
                }
                else if (!g_strcmp0(*key, "payloadkey")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "payloadkey", &error);
                    stream_key = string(val);
                }
                else if (!g_strcmp0(*key, "consumergroup")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "consumergroup", &error);
                    cgrp = string(val);
                }
                else if (!g_strcmp0(*key, "consumername")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "consumername", &error);
                    cname = string(val);
                }
                else if (!g_strcmp0(*key, "sensorid")) {
                    val = g_key_file_get_string(gcfg_file, grpname, "sensorid", &error);
                    // Only set sensor ID if extraction is enabled
                    const char* enable_extraction = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
                    if (val != NULL && (enable_extraction != NULL && strcmp(enable_extraction, "1") == 0)) {
                        g_setenv("DEEPSTREAM_SENSOR_ID", val, TRUE);
                        nvds_log(LOG_CAT, LOG_DEBUG, "Sensor ID from config file: %s", val);
                    }
                }
                if (val != NULL) {
                    g_free(val);
                    val = NULL;
                }
            }
            free_gobjs(gcfg_file, error, keys, NULL);
        }
    }

    //Look for connection details provided(if any) in params to nvds_msgapi_connect [hostname;port]
    if((str != NULL) && (strlen(str) > 3)) {
        istringstream iss(str);
        getline(iss, ip_addr, ';');
        getline(iss, port,';');
    }
    //Fetch connection details were provided in cfg file
    else {
        port = cfg_port;
        ip_addr = cfg_ipaddr;
    }

    password = cfg_pwd;

    // Fetch password from environmental variable
    var_pwd = g_getenv(PASSWORD_VAR);
    // Override password with environmental variable if given
    if (var_pwd != NULL)
      password = string(var_pwd);

    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function to connect to redis server
*/
NvDsMsgApiHandle nvds_msgapi_connect(char *connection_str, nvds_msgapi_connect_cb_t connect_cb, char *config_path) {
    string hostname="";
    string password="";
    string port="";
    string stream_size="";
    string payloadkey=DFLT_PAYLOAD_KEY;
    string consumer_grp=DFLT_CONSUMER_GRP;
    string consumer_name=DFLT_CONSUMER_NAME;

    // If file dump is enabled, skip actual Redis connection and return a dummy handle
    // This allows the pipeline to start even when Redis server is not available
    const char *json_dump_env = getenv("NVDS_JSON_DUMP_FILE");
    if (json_dump_env) {
        nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_connect: File dump mode enabled (NVDS_JSON_DUMP_FILE=%s), skipping Redis connection", json_dump_env);
        // Allocate a dummy context to return as handle
        nvds_redis_context_data_t *rh = new nvds_redis_context_data_t();
        rh->c = NULL;  // No actual Redis connection
        rh->sc = NULL;
        rh->connect_cb = connect_cb;
        rh->disconnect = false;
        rh->stream_size = "";
        rh->payload_key = DFLT_PAYLOAD_KEY;
        rh->consumer_grp = DFLT_CONSUMER_GRP;
        rh->consumer_name = DFLT_CONSUMER_NAME;
        return (NvDsMsgApiHandle) rh;
    }

    if(connection_str == NULL && config_path == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Both redis broker_str and path to cfg cant be NULL");
        return NULL;
    }

    // Parse config and connection string
    if(parse_config(connection_str, config_path, hostname, password, port, stream_size, payloadkey, consumer_grp, consumer_name)) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure to fetch login credentails");
        return NULL;
    }

    int port_num= atoi(port.c_str());
    // Connect to redis server with hostname and port
    redisContext *c = redisConnect(hostname.c_str(), port_num);
    if (c == NULL || c->err) {
        if (c) {
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Redis Connection error: %s\n", c->errstr);
            redisFree(c);
        }
        else {
            nvds_log(LOG_CAT, LOG_ERR , "Redis Connection error: can't allocate redis context");
        }
        return NULL;
    }

    // If password is provided, send authentication command
    if (!password.empty()) {
        string command = "AUTH " + password;
        redisReply *reply;
        reply = (redisReply*) redisCommand(c, command.c_str());
        if (reply == NULL) {
            g_print("nvds_msgapi_connect: Authentication has failed.");
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Authentication has failed.");
            redisFree(c);
            return NULL;
        }
        else if (c->err || reply->type == REDIS_REPLY_ERROR) {
            g_print("nvds_msgapi_connect: %s\n", reply->str);
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: %s\n", reply->str);
            freeReplyObject(reply);
            redisFree(c);
            return NULL;
        }
        freeReplyObject(reply);
    }

    //Check connection status with PING
    redisReply* reply = (redisReply*) redisCommand(c, "PING");
    if (reply == NULL) {
        g_print("nvds_msgapi_connect: Unable to ping server.");
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Unable to ping server.");
        redisFree(c);
        return NULL;
    }
    if (c->err || reply->type == REDIS_REPLY_ERROR) {
        g_print("nvds_msgapi_connect: %s\n", reply->str);
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: %s\n", reply->str);
        freeReplyObject(reply);
        redisFree(c);
        return NULL;
    }

    // Create redis handle
    nvds_redis_context_data_t *rh = new (nothrow) nvds_redis_context_data_t;
    if(rh == NULL){
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: malloc failed for creating handle. Exiting..");
        redisFree(c);
        return NULL;
    }
    // Set handle members
    rh->host=hostname;
    rh->port=port_num;
    rh->stream_size=stream_size;
    rh->payload_key=payloadkey;
    rh->consumer_grp=consumer_grp;
    rh->consumer_name=consumer_name;
    rh->c = c;
    rh->disconnect=false;
    rh->subscription=false;
    rh->connect_cb=connect_cb;

    nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_connect: Connection Success.");
    nvds_log(LOG_CAT, LOG_INFO , "Connection details: Host[%s], port[%s], consumer grp[%s], consumer name[%s]", \
                                  hostname.c_str(), port.c_str(), consumer_grp.c_str(), consumer_name.c_str());
    return (NvDsMsgApiHandle) rh;
}

/* nvds_msgapi function for synchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf) {
     if(h_ptr == NULL ) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis connection handle passed for send() = NULL. Send failed\n");
        return NVDS_MSGAPI_ERR;
    }
    if(topic == NULL || !strcmp(topic, "")) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis send topic not specified. Send failed\n");
        return NVDS_MSGAPI_ERR;
    }
    if(payload == NULL || nbuf <=0) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis: Either send payload is NULL or payload length <=0. Send failed\n");
        return NVDS_MSGAPI_ERR;
    }
     // Retrieve handle
    nvds_redis_context_data_t *rh = (nvds_redis_context_data_t *) h_ptr;
    redisReply *reply;

    // Check if DeepStream format is enabled
    const char* enable_deepstream_format = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
    if (enable_deepstream_format != NULL && strcmp(enable_deepstream_format, "1") == 0) {
        // DeepStream format: key, value, headers
        string sensor_id = extract_sensor_id_from_payload(payload, nbuf);
        
        // Send redis command for publishing a message with DeepStream format
        if(rh->stream_size == "")
            reply = (redisReply *) redisCommand(rh->c, "XADD %s * key %s value %b headers {}", topic, sensor_id.c_str(), (void *)payload, nbuf);
        else
            reply = (redisReply *) redisCommand(rh->c, "XADD %s MAXLEN ~ %s * key %s value %b headers {}", topic, rh->stream_size.c_str(), sensor_id.c_str(), (void *)payload, nbuf);
    } else {
        // Original format: only value field
        if(rh->stream_size == "")
            reply = (redisReply *) redisCommand(rh->c, "XADD %s * %s %b", topic, rh->payload_key.c_str(), (void *)payload, nbuf);
        else
            reply = (redisReply *) redisCommand(rh->c, "XADD %s MAXLEN ~ %s * %s %b", topic, rh->stream_size.c_str(), rh->payload_key.c_str(), (void *)payload, nbuf);
    }

    if(reply == NULL) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis send failed");
        return NVDS_MSGAPI_ERR;
    }
    freeReplyObject(reply);
    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for asynchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char  *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr){
    if(h_ptr == NULL ) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis connection handle passed for send() = NULL. Send failed\n");
        return NVDS_MSGAPI_ERR;
    }
    if(topic == NULL || !strcmp(topic, "")) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis send topic not specified. Send failed\n");
        return NVDS_MSGAPI_ERR;
    }
    if(payload == NULL || nbuf <=0) {
        nvds_log(LOG_CAT, LOG_ERR, "Redis: Either send payload is NULL or payload length <=0. Send failed\n");
        return NVDS_MSGAPI_ERR;
    }
    // Retrieve handle
    nvds_redis_context_data_t *rh = (nvds_redis_context_data_t *) h_ptr;

    // Copy message into data
    uint8_t *data = new (nothrow) uint8_t[nbuf];
    memset(data,'\0', nbuf);
    memcpy(data, payload, nbuf);

    // Create msg_info
    send_msg_info_t msg_info = {(void *) data, nbuf, topic, send_callback, user_ptr};
    // Add msg_info to message queue
    rh->publisher_mutex.lock();
    rh->primary_queue.push(msg_info);
    rh->publisher_mutex.unlock();

    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for performing work. In redis case, send the messages added to queue for async send
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) {
    if(h_ptr == NULL ) {
        nvds_log(LOG_CAT, LOG_ERR, "Null connection handle passed in do_work(). Error\n");
        return;
    }

    nvds_redis_context_data_t *rh = (nvds_redis_context_data_t *) h_ptr;

    //swap buffers
    rh->publisher_mutex.lock();
    rh->primary_queue.swap(rh->secondary_queue);
    rh->publisher_mutex.unlock();

    // Check if any async send operations pending
    while(!rh->secondary_queue.empty()) {
        // Send messages in queue
        send_msg_info_t &node = rh->secondary_queue.front();
        redisReply *reply;

        // Check if DeepStream format is enabled
        const char* enable_deepstream_format = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
        if (enable_deepstream_format != NULL && strcmp(enable_deepstream_format, "1") == 0) {
            // DeepStream format: key, value, headers
            string sensor_id = extract_sensor_id_from_payload((const uint8_t *)node.msg, node.msg_len);
            
            if(rh->stream_size == ""){
                reply = (redisReply *) redisCommand(rh->c, "XADD %s * key %s value %b headers {}", node.topic.c_str(), sensor_id.c_str(), node.msg, node.msg_len);
            }
            else {
                reply = (redisReply *) redisCommand(rh->c, "XADD %s MAXLEN ~ %s * key %s value %b headers {}", node.topic.c_str(), rh->stream_size.c_str(), sensor_id.c_str(), node.msg, node.msg_len);
            }
        } else {
            // Original format: only value field
            if(rh->stream_size == ""){
                reply = (redisReply *) redisCommand(rh->c, "XADD %s * %s %b", node.topic.c_str(), rh->payload_key.c_str(), node.msg, node.msg_len);
            }
            else {
                reply = (redisReply *) redisCommand(rh->c, "XADD %s MAXLEN ~ %s * %s %b", node.topic.c_str(), rh->stream_size.c_str(), rh->payload_key.c_str(), node.msg, node.msg_len);
            }
        }
        
        if(reply == NULL) {
            nvds_log(LOG_CAT, LOG_ERR, "nvds_msgapi_do_work: Redis send failed");
            (node.user_cb_func) (node.user_ctx , NVDS_MSGAPI_ERR);
            //Treat any send/consume failure as connection failure
            if(rh->connect_cb)
                (rh->connect_cb) ((NvDsMsgApiHandle) rh , NVDS_MSGAPI_EVT_SERVICE_DOWN);
        }
        else {
            nvds_log(LOG_CAT, LOG_DEBUG , "nvds_msgapi_do_work: Message sent = %.*s\n", node.msg_len, node.msg);
            (node.user_cb_func) (node.user_ctx , NVDS_MSGAPI_OK);
        }
        freeReplyObject(reply);
        delete[] (uint *)node.msg;
        rh->secondary_queue.pop();
    }
}

//Thread to consume messages on redis stream
//Single topic subscription supported
void consumer(NvDsMsgApiHandle h_ptr, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx) {
    nvds_redis_context_data_t *rctx = (nvds_redis_context_data_t *) h_ptr;

    while(!rctx->disconnect) {
        // Redis command for reading data from the consumer group
        string redis_read_cmd = "XREADGROUP GROUP " + rctx->consumer_grp + " " + rctx->consumer_name + " BLOCK 100 COUNT 1 STREAMS ";
        // Add stream/topic names to command
        for(auto it:rctx->subscribe_topics)
            redis_read_cmd += it.first + " ";
        // Add stream IDs to command
        for(auto it:rctx->subscribe_topics)
            redis_read_cmd += it.second + " ";
        redis_read_cmd = redis_read_cmd.substr(0, redis_read_cmd.length()-1);

        // Send command
        redisReply *reply = (redisReply *) redisCommand(rctx->sc, redis_read_cmd.c_str());

        if(reply == NULL) {
            nvds_log(LOG_CAT, LOG_ERR, "Redis consume failed");
            //Treat any send/consume failure as connection failure
            if(rctx->connect_cb)
                (rctx->connect_cb) ((NvDsMsgApiHandle) rctx , NVDS_MSGAPI_EVT_SERVICE_DOWN);
        }
        else {
            if(reply->type == REDIS_REPLY_ARRAY){
                /*example reply msg
                127.0.0.1:6379> XREADGROUP GROUP redis-consumer-group redis-consumer BLOCK 10 COUNT 1 STREAMS topic11 >
                    1) 1) "topic11"
                       2) 1) 1) "1601055190555-0"
                             2) 1) "sensor.id"
                                2) "{    \"messageid\" : \"84a3a0ad-7eb8" }
                */

                //Parse the reply message as per the above format
                redisReply *streams_arr;
                for(size_t i=0; i<reply->elements; i++) {
                    streams_arr = reply->element[i];
                    if(streams_arr->type == REDIS_REPLY_ARRAY) {
                        char *stream_name, *payload_id;
                        redisReply *data_arr, *sub_arr, *msg_arr;

                        stream_name = streams_arr->element[0]->str;
                        data_arr = streams_arr->element[1];

                        if(data_arr->type == REDIS_REPLY_ARRAY) {
                            sub_arr = data_arr->element[0];
                            payload_id = sub_arr->element[0]->str;
                            msg_arr = sub_arr->element[1];
                            if(msg_arr->type == REDIS_REPLY_ARRAY) {
                                char *value;
                                size_t value_len;
                                
                                // Check if DeepStream format is enabled
                                const char* enable_deepstream_format = getenv("DEEPSTREAM_ENABLE_SENSOR_ID_EXTRACTION");
                                if (enable_deepstream_format != NULL && strcmp(enable_deepstream_format, "1") == 0) {
                                    // DeepStream format: key {sensor_id} value {payload} headers {}
                                    // msg_arr structure: [0]="key", [1]={sensor_id}, [2]="value", [3]={payload}, [4]="headers", [5]={}
                                    if (msg_arr->elements >= 4 && msg_arr->element[2]->type == REDIS_REPLY_STRING && 
                                        strcmp(msg_arr->element[2]->str, "value") == 0 && 
                                        msg_arr->element[3]->type == REDIS_REPLY_STRING) {
                                        value = msg_arr->element[3]->str;
                                        value_len = msg_arr->element[3]->len;
                                        nvds_log(LOG_CAT, LOG_DEBUG, "Redis consume DeepStream format message: sensor_id=%s, payload=%s", 
                                                 msg_arr->element[1]->str, value);
                                    } else {
                                        nvds_log(LOG_CAT, LOG_ERR, "Redis consume: Invalid DeepStream format message structure");
                                        continue;
                                    }
                                } else {
                                    // Original format: {payload_key} {payload}
                                    // msg_arr structure: [0]={payload_key}, [1]={payload}
                                    if (msg_arr->elements >= 2 && msg_arr->element[1]->type == REDIS_REPLY_STRING) {
                                        value = msg_arr->element[1]->str;
                                        value_len = msg_arr->element[1]->len;
                                        nvds_log(LOG_CAT, LOG_DEBUG, "Redis consume original format message: key=%s, payload=%s", 
                                                 msg_arr->element[0]->str, value);
                                    } else {
                                        nvds_log(LOG_CAT, LOG_ERR, "Redis consume: Invalid original format message structure");
                                        continue;
                                    }
                                }
                                
                                nvds_log(LOG_CAT, LOG_INFO, "Redis consume message: %s ", value);
                                cb(NVDS_MSGAPI_OK, (void *) value, value_len, stream_name, user_ctx);

                                string ack_cmd = "XACK " + string(stream_name) + " " + rctx->consumer_grp + " " + string(payload_id);

                                //Send ACK for the msg read
                                redisReply *ack_reply = (redisReply *) redisCommand(rctx->sc, ack_cmd.c_str());
                                if(ack_reply == NULL) {
                                    nvds_log(LOG_CAT, LOG_ERR, "Redis consumer XACK failed for stream[%s] , consumer grp[%s]", stream_name, rctx->consumer_grp.c_str());
                                    //Treat any send/consume failure as connection failure
                                    if(rctx->connect_cb)
                                        (rctx->connect_cb) ((NvDsMsgApiHandle) rctx , NVDS_MSGAPI_EVT_SERVICE_DOWN);
                                }
                                freeReplyObject(ack_reply);
                            }
                        }
                    }
                }
            }
        }

        freeReplyObject(reply);

        if(rctx->disconnect) {
            break;
        }
    }
    nvds_log(LOG_CAT, LOG_INFO, "Redis Consumer thread exited. Message consumption stopped\n");
}

/* nvds_msgapi function for subscribe to topics
*/
NvDsMsgApiErrorType nvds_msgapi_subscribe (NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx) {
    if (!h_ptr) {
        nvds_log(LOG_CAT, LOG_ERR, "nvds_msgapi_subscribe called with null handle\n");
        return NVDS_MSGAPI_ERR;
    }

    if(topics == NULL || num_topics <=0 ) {
        nvds_log(LOG_CAT, LOG_ERR, "Topics not specified for subscribe. Subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(!cb) {
        nvds_log(LOG_CAT, LOG_ERR, "Subscribe callback cannot be NULL. subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }
    // Retrieve handle
    nvds_redis_context_data_t *rctx = (nvds_redis_context_data_t *) h_ptr;

    // Create consumer group for each topic
    for(int i=0; i<num_topics; i++) {
        if(!rctx->subscribe_topics.count(string(topics[i]))) {
            // Set ID for the group to '>' : this means that the consumer wants to receive only new messages
            rctx->subscribe_topics[string(topics[i])] = ">";
            // Redis command for creating consumer group for a topic name
            string cmd = "XGROUP CREATE " + string(topics[i]) + " " + rctx->consumer_grp + " $ MKSTREAM";
            redisReply *reply = (redisReply *) redisCommand(rctx->c, cmd.c_str());
            if(reply == NULL) {
                nvds_log(LOG_CAT, LOG_ERR, "Redis xgroup create failed on consumer group[%s] for stream %s", rctx->consumer_grp.c_str(), topics[i]);
            }
            freeReplyObject(reply);
        }
    }
    //if there's an already existing subscription
    if(rctx->subscription) {
        nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_subscribe : subscribe topics updated.");
    }
    else {
        redisContext *sc = redisConnect(rctx->host.c_str(), rctx->port);
        if (sc == NULL || sc->err) {
            if (sc) {
                nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_subscribe: Redis subscribe context creation error: %s\n", sc->errstr);
                redisFree(sc);
            }
            else {
                nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_subscribe: Redis subscribe: can't allocate redis context");
            }
            return NVDS_MSGAPI_ERR;
        }
        rctx->sc = sc;
        rctx->subscription=true;
        rctx->subscribe_thread = thread(consumer, h_ptr, cb , user_ctx);
    }
    nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_subscribe: Redis subscribe success");
    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for disconnect
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr) {
    if (!h_ptr) {
        nvds_log(LOG_CAT, LOG_DEBUG, "nvds_msgapi_disconnect called with null handle\n");
        return NVDS_MSGAPI_ERR;
    }
    //Retrieve handle
    nvds_redis_context_data_t *rctx = (nvds_redis_context_data_t *) h_ptr;

    // If in file dump mode (no actual Redis connection), just free the context
    if (rctx->c == NULL) {
        nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_disconnect: File dump mode, no Redis connection to close");
        delete rctx;
        h_ptr = NULL;
        return NVDS_MSGAPI_OK;
    }

    rctx->disconnect=true;
    if(rctx->subscription) {
        //join subscribe thread
        rctx->subscribe_thread.join();
        //free redis consumer context
        redisFree(rctx->sc);
        //destroy consumer group created on streams
        for(auto it:rctx->subscribe_topics) {
            string cmd = "XGROUP DESTROY " + it.first + " " + rctx->consumer_grp;
            redisReply *reply = (redisReply *) redisCommand(rctx->c, cmd.c_str());
            if(reply == NULL) {
                nvds_log(LOG_CAT, LOG_ERR, "Redis xgroup destroy failed on consumer group[%s] for stream %s",rctx->consumer_grp.c_str(), it.first.c_str());
            }
            freeReplyObject(reply);
        }
    }

    //free redis producer context
    redisFree(rctx->c);

    //free user data
    delete rctx;

    h_ptr=NULL;
    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for retrieving nvds_msgapi version
*/
char *nvds_msgapi_getversion(void){
  return (char *)NVDS_MSGAPI_VERSION;
}

/* nvds_msgapi function for retrieving protocol name
*/
char *nvds_msgapi_get_protocol_name(void) {
  return (char *)NVDS_MSGAPI_PROTOCOL;
}

/* nvds_msgapi function for retrieving connection signature, used for connection sharing
*/
NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str, char *cfg, char *output_str, int max_len) {
    strcpy(output_str,"");

    //check if share-connection config option is turned ON
    char reuse_connection[16]="";
    if(fetch_config_value(cfg, "share-connection", reuse_connection, sizeof(reuse_connection), LOG_CAT) != NVDS_MSGAPI_OK) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing redis share-connection config option");
        return NVDS_MSGAPI_ERR;
    }
    if(strcmp(reuse_connection, "1")) {
        nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_connection_signature: redis connection sharing disabled. Hence connection signature cant be returned");
        return NVDS_MSGAPI_OK;
    }

    if(broker_str == NULL && cfg == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Both redis broker_str and path to cfg cant be NULL");
        return NVDS_MSGAPI_ERR;
    }
    if(max_len < 2 * SHA256_DIGEST_LENGTH + 1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: output string length allocated not sufficient");
        return NVDS_MSGAPI_ERR;
    }
    string ip_addr;
    string password;
    string port;
    string csize, cgrp, cname, key;

    if(parse_config(broker_str, cfg, password, ip_addr, port, csize, key, cgrp, cname) == -1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Failure in fetching redis connection string");
        return NVDS_MSGAPI_ERR;
    }
    // Generate connection signature from connection string
    string redis_connection_str = ip_addr + ";" + port;
    string hashval = generate_sha256_hash(redis_connection_str);
    strcpy(output_str, hashval.c_str());
    return NVDS_MSGAPI_OK;
}
