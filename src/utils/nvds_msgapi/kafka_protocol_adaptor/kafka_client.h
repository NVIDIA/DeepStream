/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include <string>
#include <sstream>
using namespace std;

#include "rdkafka.h"
#include "nvds_msgapi.h"

#define MAX_FIELD_LEN 1024
#define MAX_TOPIC_LEN 255 //maximum topic length supported by kafka is 255
#define NVDS_KAFKA_LOG_CAT "DSLOG:NVDS_KAFKA_PROTO"

class NvDsKafkaSendCompl {
 public:
  virtual void sendcomplete(NvDsMsgApiErrorType);
  NvDsMsgApiErrorType get_err();
  virtual ~NvDsKafkaSendCompl() = default;
};

class NvDsKafkaSyncSendCompl: public NvDsKafkaSendCompl {
 private:
  uint8_t *compl_flag;
  NvDsMsgApiErrorType err;

 public:
  NvDsKafkaSyncSendCompl(uint8_t *);
  void sendcomplete(NvDsMsgApiErrorType);
  NvDsMsgApiErrorType get_err();
};

class NvDsKafkaAsyncSendCompl: public NvDsKafkaSendCompl {
 private:
  void *user_ptr;
  nvds_msgapi_send_cb_t async_send_cb;

 public:
  NvDsKafkaAsyncSendCompl(void *ctx, nvds_msgapi_send_cb_t cb);
  void sendcomplete(NvDsMsgApiErrorType);
};

typedef struct {
  pthread_t consumer_tid;                              /* Thread which waits on incoming msg from cloud*/
  nvds_msgapi_subscribe_request_cb_t  subscribe_req_cb;/* User callback function */
  void *user_ptr;                                      /* User context pointer */
} consumer_thread_info;

typedef struct {
   rd_kafka_t *consumer;                               /* Consumer instance handle */
   char consumer_grp_id[MAX_FIELD_LEN];                /* Consumer group id */
   consumer_thread_info cinfo;                         /* Consumer thread info*/
   bool disconnect;                                    /* variable to notify consume thread to quit */
   string config;                                      /* config options for consumer instance */
}consumer_instance_t;

typedef struct {
  char partition_key_field[MAX_FIELD_LEN];	       /* partition key for messages */
  rd_kafka_t *producer;                                /* Producer instance handle */
}producer_instance_t;

typedef struct {
   char brokers[MAX_FIELD_LEN];                        /* Broker string - comma separated host:port */
   producer_instance_t  p_instance;                    /* Producer instance details */
   consumer_instance_t  c_instance;                    /* consumer instance details */
} NvDsKafkaClientHandle;

void *nvds_kafka_client_init(NvDsKafkaClientHandle *kh);
NvDsMsgApiErrorType nvds_kafka_producer_launch(void *kh, rd_kafka_conf_t *conf);
NvDsMsgApiErrorType nvds_kafka_client_send(void *kh, const uint8_t *payload, int len, char *topic, int sync, void *ctx, nvds_msgapi_send_cb_t cb,  char *key, int keylen);
NvDsMsgApiErrorType nvds_kafka_client_setconf(rd_kafka_conf_t *conf, char *key, char *val);
void nvds_kafka_client_poll(void *kv);
void nvds_kafka_client_finish(void *kv);
