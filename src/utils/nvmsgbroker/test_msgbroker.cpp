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

#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <map>
#include <string>
#include "nvmsgbroker.h"

/* MODIFY: to reflect your own paths */
#define SO_PATH "/opt/nvidia/deepstream/deepstream/lib/"

#define KAFKA_KEY "kafka"
#define KAFKA_PROTO_SO "libnvds_kafka_proto.so"
#define KAFKA_PROTO_PATH SO_PATH KAFKA_PROTO_SO
#define KAFKA_CFG_FILE "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
#define KAFKA_CONN_STR "localhost;9092" // broker;port

#define AMQP_KEY "amqp"
#define AMQP_PROTO_SO "libnvds_amqp_proto.so"
#define AMQP_PROTO_PATH SO_PATH AMQP_PROTO_SO
#define AMQP_CFG_FILE "/opt/nvidia/deepstream/deepstream/sources/libs/amqp_protocol_adaptor/cfg_amqp.txt"
#define AMQP_CONN_STR "localhost;5672;guest;guest" // broker;port;username;password

#define AZURE_KEY "azure"
#define AZURE_PROTO_SO "libnvds_azure_proto.so"
#define AZURE_PROTO_PATH SO_PATH AZURE_PROTO_SO
#define AZURE_CFG_FILE "/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/device_client/cfg_azure.txt"

#define REDIS_KEY "redis"
#define REDIS_PROTO_SO "libnvds_redis_proto.so"
#define REDIS_PROTO_PATH SO_PATH REDIS_PROTO_SO
#define REDIS_CFG_FILE "/opt/nvidia/deepstream/deepstream/sources/libs/redis_protocol_adaptor/cfg_redis.txt"
#define REDIS_CONN_STR "localhost;6379" // broker;port

#define MQTT_KEY "mqtt"
#define MQTT_PROTO_SO "libnvds_mqtt_proto.so"
#define MQTT_PROTO_PATH SO_PATH MQTT_PROTO_SO
#define MQTT_CFG_FILE "/opt/nvidia/deepstream/deepstream/sources/libs/mqtt_protocol_adaptor/cfg_mqtt.txt"
#define MQTT_CONN_STR "localhost;1883" // broker;port

//There are 2 options to provide Azure connection string
//option 1: The full device connection string is provided in nv_msgbroker_connect().
//option 2: The full device connection string is provided in config file.

//In this example: option 2 is used. full connection string should be provided in cfg_azure.txt
#define AZURE_CONN_STR nullptr  // "HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>"

#define MAX_FIELD_LEN 1024

struct test_info
{
    int test_id;
    char *test_key;
    char *proto_key;
    char *proto_path;
    char *cfg_file;
    char *conn_str;
    int cb_count;
    int consumed_count;

    test_info() {}
    test_info(int id, const char* test_key, const char* proto_key, const char* proto_path, const char* cfg_file, const char* conn_str)
    {
        load_info(id, test_key, proto_key, proto_path, cfg_file, conn_str);
    }

    void load_info(int id, const char* test_key, const char* proto_key, const char* proto_path, const char* cfg_file, const char* conn_str)
    {
        this->test_id = id;
        this->test_key = strndup(test_key, MAX_FIELD_LEN);
        this->proto_key = strndup(proto_key, MAX_FIELD_LEN);
        this->proto_path = strndup(proto_path, MAX_FIELD_LEN);
        this->cfg_file = (cfg_file != nullptr) ? strndup(cfg_file, MAX_FIELD_LEN) : nullptr;
        this->conn_str = (conn_str != nullptr) ? strndup(conn_str, MAX_FIELD_LEN) : nullptr;
        this->cb_count = 0;
        this->consumed_count = 0;
    }

    /** copy constructor for when the user may create a new object from
     * extant object */
    test_info(const test_info &src)
    {
        test_id = src.test_id;
        test_key = strndup(src.test_key, MAX_FIELD_LEN);
        proto_key = strndup(src.proto_key, MAX_FIELD_LEN);
        proto_path = strndup(src.proto_path, MAX_FIELD_LEN);
        cfg_file = (src.cfg_file != nullptr) ? strndup(src.cfg_file, MAX_FIELD_LEN) : nullptr;
        conn_str = (src.conn_str != nullptr) ? strndup(src.conn_str, MAX_FIELD_LEN) : nullptr;
        cb_count = src.cb_count;
        consumed_count = src.consumed_count;
    }

    /** copy assignment operator for when user uses assignment operator
     * to copy data into a new object */
    test_info& operator=(test_info const & src)
    {
        test_id = src.test_id;
        test_key = strndup(src.test_key, MAX_FIELD_LEN);
        proto_key = strndup(src.proto_key, MAX_FIELD_LEN);
        proto_path = strndup(src.proto_path, MAX_FIELD_LEN);
        cfg_file = (src.cfg_file != nullptr) ? strndup(src.cfg_file, MAX_FIELD_LEN) : nullptr;
        conn_str = (src.conn_str != nullptr) ? strndup(src.conn_str, MAX_FIELD_LEN) : nullptr;
        cb_count = src.cb_count;
        consumed_count = src.consumed_count;
        return *this;
    }

    ~test_info()
    {
        free(test_key);
        free(proto_key);
        free(proto_path);
        if (cfg_file != nullptr)
            free(cfg_file);
        if (conn_str != nullptr)
            free(conn_str);
    }
};

struct key_cmp
{
   bool operator()(char const *a, char const *b) const
   {
      return strncmp(a, b, MAX_FIELD_LEN) < 0;
   }
};

// Global info map for tests
// The entry for each test holds information for launching
// the test as well as keeping track of callback counts etc.
std::map<char*, test_info, key_cmp> g_info_map;

void test_connect_cb(NvMsgBrokerClientHandle h_ptr, NvMsgBrokerErrorType status)
{
    if (status == NV_MSGBROKER_API_OK)
        printf("Connect succeeded\n");
    else if(status == NV_MSGBROKER_API_RECONNECTING)
        printf("Reconnection in progress\n");
    else
        printf("Connect failed\n");
}

void test_send_cb(void *user_ptr,  NvMsgBrokerErrorType flag)
{
    int count = -1;
    int id = -1;
    char *key = (char*)user_ptr;
    std::map<char*, test_info>::iterator iter = g_info_map.find(key);
    if (iter != g_info_map.end()) {
        test_info &ti = iter->second;
        count = ++ti.cb_count;
        id = ti.test_id;
    } else {
        printf("test_send_cb: Failed to find test info for %s\n", key);
    }

    if (flag == NV_MSGBROKER_API_OK)
        printf("Test %d: async send[%d] succeeded for %s\n", id, count, key);
    else
        printf("Test %d: async send[%d] failed for %s\n", id, count, key);
}

void test_subscribe_cb(NvMsgBrokerErrorType flag, void *msg, int msglen, char *topic, void *user_ptr)
{
    int count = -1;
    int id = -1;
    char *key = (char*)user_ptr;
    std::map<char*, test_info>::iterator iter = g_info_map.find(key);
    if (iter != g_info_map.end()) {
        test_info &ti = iter->second;
        count = ++ti.consumed_count;
        id = ti.test_id;
    } else {
        printf("subscribe_cb: Failed to find test info for %s\n", key);
    }

    if(flag == NV_MSGBROKER_API_ERR) {
        printf("Test %d: Error in consuming message[%d] from broker\n", id, count);
    }
    else {
        printf("Test %d: Consuming message[%d], on topic[%s]. Payload= %.*s\n", id, count, topic, msglen, (const char *) msg);
    }
}

int run_test(char* test_key)
{
    std::map<char*, test_info>::iterator iter = g_info_map.find(test_key);
    if (iter == g_info_map.end()) {
        printf("Failed to find test info for %s\n", test_key);
        return -1;
    }

    test_info &ti = iter->second;

    NvMsgBrokerClientHandle conn_handle;
    char *error;
    char SEND_MSG[]= "{ \
    \"messageid\" : \"84a3a0ad-7eb8-49a2-9aa7-104ded6764d0_c788ea9efa50\", \
    \"mdsversion\" : \"1.0\", \
    \"@timestamp\" : \"\", \
    \"place\" : { \
      \"id\" : \"1\", \
      \"name\" : \"HQ\", \
      \"type\" : \"building/garage\", \
      \"location\" : { \
        \"lat\" : 0, \
        \"lon\" : 0, \
        \"alt\" : 0 \
    }, \
    \"aisle\" : { \
      \"id\" : \"C_126_135\", \
      \"name\" : \"Lane 1\", \
      \"level\" : \"P1\", \
      \"coordinate\" : { \
        \"x\" : 1, \
        \"y\" : 2, \
        \"z\" : 3 \
      } \
     }\
    },\
    \"sensor\" : { \
      \"id\" : \"10_110_126_135_A0\", \
      \"type\" : \"Camera\", \
      \"description\" : \"Aisle Camera\", \
      \"location\" : { \
        \"lat\" : 0, \
        \"lon\" : 0, \
        \"alt\" : 0 \
     }, \
     \"coordinate\" : { \
       \"x\" : 0, \
       \"y\" : 0, \
       \"z\" : 0 \
      } \
     } \
    }";

   
    conn_handle = nv_msgbroker_connect(ti.conn_str, ti.proto_path, test_connect_cb, ti.cfg_file);
    if (!conn_handle) {
        printf("Test %d: Connect failed for %s [%s:%s].\n", ti.test_id, ti.conn_str, ti.test_key, ti.proto_key);
        return -1;
    }

    //Subscribe to topics
    const char *topics[] = {"topic1", "topic2"};
    int num_topics=2;
    NvMsgBrokerErrorType ret = nv_msgbroker_subscribe(conn_handle, (char **)topics, num_topics, test_subscribe_cb, ti.test_key);
    switch (ret) {
        case NV_MSGBROKER_API_ERR:
            printf("Test %d: Subscription to topic[s] failed for %s(%s)\n", ti.test_id, ti.test_key, ti.proto_key);
            return -1;
        case NV_MSGBROKER_API_NOT_SUPPORTED:
            printf("Test %d: Subscription not supported for %s(%s). Skipping subscription.\n", ti.test_id, ti.test_key, ti.proto_key);
            break;
    }

    NvMsgBrokerClientMsg msg;
    msg.topic = strndup("topic1", MAX_FIELD_LEN);
    msg.payload = SEND_MSG;
    msg.payload_len = strlen(SEND_MSG);
    for(int i = 0; i < 5; i++) {
      if (nv_msgbroker_send_async(conn_handle, msg, test_send_cb, ti.test_key) != NV_MSGBROKER_API_OK)
	    printf("Test %d: send [%d] failed for %s(%s)\n", ti.test_id, i, ti.test_key, ti.proto_key);
      else
	    printf("Test %d: sending [%d] asynchronously for %s(%s)\n", ti.test_id, i, ti.test_key, ti.proto_key);
      usleep(10000);    //10ms sleep
    }
    free(msg.topic);

    printf("Test %d: Disconnecting... in 3 secs\n", ti.test_id);
    sleep(3);
    nv_msgbroker_disconnect(conn_handle);
    return 0;
}

int main(int argc, char* argv[])
{
    int test_id = 0;
    test_info kafka_test(test_id++, KAFKA_KEY, KAFKA_KEY, KAFKA_PROTO_PATH, KAFKA_CFG_FILE, KAFKA_CONN_STR);
    test_info amqp_test(test_id++, AMQP_KEY, AMQP_KEY, AMQP_PROTO_PATH, AMQP_CFG_FILE, AMQP_CONN_STR);
    test_info azure_test(test_id++, AZURE_KEY, AZURE_KEY, AZURE_PROTO_PATH, AZURE_CFG_FILE, AZURE_CONN_STR);
    test_info redis_test(test_id++, REDIS_KEY, REDIS_KEY, REDIS_PROTO_PATH, REDIS_CFG_FILE, REDIS_CONN_STR);
    test_info mqtt_test(test_id++, MQTT_KEY, MQTT_KEY, MQTT_PROTO_PATH, MQTT_CFG_FILE, MQTT_CONN_STR);

    // To disable a test, comment out the line below that adds it to g_info_map.
    g_info_map[kafka_test.test_key] = kafka_test;
    g_info_map[amqp_test.test_key] = amqp_test;
    g_info_map[azure_test.test_key] = azure_test;
    g_info_map[redis_test.test_key] = redis_test;
    g_info_map[mqtt_test.test_key] = mqtt_test;

    printf("Refer to nvds log file for log output\n");

    std::vector<std::thread> threads;
    for (auto iter = g_info_map.begin(); iter != g_info_map.end(); ++iter) {
        printf("Starting %s: %s\n", iter->second.test_key, iter->second.proto_key);
        threads.push_back(std::thread(run_test, iter->first));
    }

    // Wait for all the threads to terminate.
    for (auto& t : threads) {
        t.join();
    }
    printf("Done. All tests finished successfully\n");
    return 0;
}
