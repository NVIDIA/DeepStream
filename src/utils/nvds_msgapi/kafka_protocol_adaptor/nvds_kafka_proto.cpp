/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <netdb.h>
#include <openssl/sha.h>
#include <glib.h>
#include <algorithm>
#include "nvds_logger.h"
#include "nvds_msgapi.h"
#include "kafka_client.h"
#include "nvds_kafka_proto.h"
#include "nvds_utils.h"
#include "schema.pb.h"

int json_get_key_value(const char *msg, int msglen, const char *key, char *value, int nbuf);
NvDsMsgApiErrorType nvds_kafka_parse_proto_cfg(char *confptr, rd_kafka_conf_t *conf);
NvDsMsgApiErrorType nvds_kafka_read_config(NvDsKafkaClientHandle *kh, char *config_path, char *partition_key_field, int field_len, rd_kafka_conf_t *conf);
NvDsMsgApiErrorType kafka_create_topic_subscription(NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx);
void *consume(void *ptr);
bool is_valid_kafka_connection_str(char *connection_str, string &burl, string &bport);

//This function will parse the proto-cfg key=value pairs
//Apply the config options to the rdkafka conf variable
NvDsMsgApiErrorType nvds_kafka_parse_proto_cfg(char *confptr, rd_kafka_conf_t *conf) {
  char *equalptr, *semiptr;
  int keylen, vallen, conflen = strlen(confptr);
  char confkey[MAX_FIELD_LEN], confval[MAX_FIELD_LEN];
  char *curptr = confptr;
  while (((equalptr = strchr(curptr, '=')) != NULL) && ((curptr - confptr) < conflen)) {
     keylen = (equalptr - curptr);
     if (keylen >= MAX_FIELD_LEN)
       keylen = MAX_FIELD_LEN - 1;

     memcpy(confkey, curptr, keylen);
     confkey[keylen] = '\0';

     if (equalptr >= (confptr + conflen)) //no more string; dangling key
       return NVDS_MSGAPI_ERR;

     semiptr = strchr(equalptr+1, ';');

     if (!semiptr) {
       vallen = (confptr + conflen - equalptr - 1);//end of strng case
       curptr = (confptr + conflen);
     }
     else {
       curptr = semiptr + 1;
       vallen = (semiptr - equalptr - 1);
     }

     if (vallen >= MAX_FIELD_LEN)
       vallen = MAX_FIELD_LEN - 1;

     memcpy(confval, (equalptr + 1), vallen);
     confval[vallen] = '\0';

     nvds_kafka_client_setconf(conf, confkey, confval);
  }
  return NVDS_MSGAPI_OK;
}

/**
 * internal function to read settings from config file
 * Documentation needs to indicate that kafka config parameters are:
  (1) located within application level config file passed to connect
  (2) within the message broker group of the config file
  (3) specified based on 'proto-cfg' key
  (4) the various options to rdkafka are specified based on 'key=value' format, within various entries semi-colon separated
Eg:
[message-broker]
enable=1
broker-proto-lib=/opt/nvidia/deepstream/deepstream-<version>/lib/libnvds_kafka_proto.so
broker-conn-str=hostname;9092
proto-cfg="message.timeout.ms=2000"

*/
NvDsMsgApiErrorType nvds_kafka_read_config(NvDsKafkaClientHandle *kh, char *config_path, char *partition_key_field, int field_len, rd_kafka_conf_t *conf)
{
    char proto_cfg[MAX_FIELD_LEN]="", producer_proto_cfg[MAX_FIELD_LEN]="", consumer_proto_cfg[MAX_FIELD_LEN]="";
    char part_key[MAX_FIELD_LEN]="", cgrp[MAX_FIELD_LEN]="";

    //Read config to fetch proto-cfg ,a generic setting applied for both producer & consumer
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_RDKAFKA_CFG, proto_cfg, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) \
        || \
       (strip_quote(proto_cfg, "proto_cfg", NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK))
            return NVDS_MSGAPI_ERR;

    //Read config to fetch producer-proto-cfg specific for producer instance
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_RDKAFKA_PRODUCER_CFG, producer_proto_cfg, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) \
        || \
       (strip_quote(producer_proto_cfg, "producer-proto-cfg", NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK))
            return NVDS_MSGAPI_ERR;

    //Read config to fetch consumer-proto-cfg specific for consumer instance
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_RDKAFKA_CONSUMER_CFG, consumer_proto_cfg, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) \
        || \
       (strip_quote(consumer_proto_cfg, "consumer-proto-cfg", NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK))
            return NVDS_MSGAPI_ERR;

    // Read config to fetch partition key field
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_PARTITION_KEY, part_key, field_len,NVDS_KAFKA_LOG_CAT) == NVDS_MSGAPI_OK) && strcmp(part_key, "")) {
        strcpy(partition_key_field, part_key);
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO,  "kafka partition key field name = %s\n", partition_key_field);
    }
    else {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR,  "kafka partition key not specified in cfg. Using default partition: %s \n", DEFAULT_PARTITION_NAME);
    }

    //Read config to fetch consumer group name
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_CONSUMER_GROUP, cgrp, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) == NVDS_MSGAPI_OK) && strcmp(cgrp,"")) {
        strcpy(kh->c_instance.consumer_grp_id, cgrp);
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO,  "Kafka Consumer group = %s \n", kh->c_instance.consumer_grp_id);
    }
    else {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO,  "Consumer group id not specified in cfg. Using default group id: %s \n", DEFAULT_KAFKA_CONSUMER_GROUP);
    }

  if (strcmp(proto_cfg , "")) {
      //Apply proto-cfg options to producer conf object
      //store the proto-cfg option in consumer string
      //to be used later when subscribe is called
      if(nvds_kafka_parse_proto_cfg(proto_cfg, conf) == NVDS_MSGAPI_OK) {
          kh->c_instance.config = string(proto_cfg);
      }
      else {
          nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Error seen while parsing proto-cfg options");
          return NVDS_MSGAPI_ERR;
      }
  }
  if (strcmp(producer_proto_cfg, "")) {
      if(nvds_kafka_parse_proto_cfg(producer_proto_cfg, conf) != NVDS_MSGAPI_OK) {
          nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Error seen while parsing producer-proto-cfg options");
          return NVDS_MSGAPI_ERR;
      }
  }
  if (strcmp(consumer_proto_cfg, "")) {
      if(kh->c_instance.config != "")
          kh->c_instance.config = kh->c_instance.config + ";" + string(consumer_proto_cfg);
      else
          kh->c_instance.config = string(consumer_proto_cfg);
  }
  return NVDS_MSGAPI_OK;
}


/**
 * connects to a broker based on given url and port to check if address is valid
 * Returns 0 if valid, and non-zero if invalid.
 * Also returns 0 if there is trouble resolving  the address or creating connection
 */
static int test_kafka_broker_endpoint(const char *burl, const char *bport) {
  int sockid;
  int port = atoi(bport);
  int flags;
  fd_set wfds;
  int error;
  struct addrinfo *res, *rp, hints;

  if (!port)
    return -1;

  memset(&hints, 0, sizeof (hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  //resolve the given url
  if ((error = getaddrinfo(burl, bport, &hints, &res))) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "getaddrinfo returned error %d\n", error);

    if ((error == EAI_FAIL) || (error == EAI_NONAME) ||  (error == EAI_NODATA)){
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "count not resolve addr - permanent failure\n");
      return error; //permanent failure to resolve
    }
    else 
      return 0; //unknown error during resolve; can't invalidate address
  }

  //iterate through all ip addresses resolved for the url
  for(rp = res; rp != NULL; rp = rp->ai_next) {
    sockid = socket(AF_INET, SOCK_STREAM, 0); //tcp socket

    //make socket non-blocking
    flags = fcntl(sockid, F_GETFL);
    if (fcntl(sockid, F_SETFL, flags | O_NONBLOCK) == -1) {
      /* having trouble making socket non-blocking;
        can't check network address, and so assume it is valid
      */
      close(sockid);
      freeaddrinfo(res);
      return 0;
    }

    if (!connect(sockid, (struct sockaddr *)res->ai_addr, res->ai_addrlen)) {
      close(sockid);
      freeaddrinfo(res);
      return 0; //connection succeeded right away
    }
    else {
      if (errno == EINPROGRESS) { //normal for non-blocking socker
        struct timeval conn_timeout;
        int optval;
        socklen_t optlen;

        conn_timeout.tv_sec = 5; //give 5 sec for connection to go through
        conn_timeout.tv_usec = 0;
        FD_ZERO(&wfds);
        FD_SET(sockid, &wfds);

        int err = select(sockid+1, NULL, & wfds, NULL, &conn_timeout);
        switch(err) {
          case 0: //timeout
            close(sockid);
            freeaddrinfo(res);
            return ETIMEDOUT;

          case 1: //socket unblocked; now figure out why
            optval = -1;
            optlen = sizeof(optval);
            if (getsockopt(sockid, SOL_SOCKET, SO_ERROR, &optval, &optlen) == -1) {
              /* error getting socket options; can't invalidate address */
              close(sockid);
              freeaddrinfo(res);
              return 0;
            }
            if (optval == 0) {
              close(sockid);
              freeaddrinfo(res);
              return 0; //no error; connection succeeded
            }
            else {
              close(sockid);
              freeaddrinfo(res);
              return optval;        //connection failed; something wrong with address
            }

          case -1: { //error in select, can't invalidate address 
            close(sockid);
            freeaddrinfo(res);
            return 0;
          }
        }
      } else {
        close(sockid);
        freeaddrinfo(res);
        return 0; // error in connect; can't invalidate address
      }
    } // non-blocking connect did not succeed
  }
  freeaddrinfo(res);
  return 0; //if we got here then can't invalidate
}

/*
 * Validate kafka connection string format
 * Valid format host;port or (host;port;topic to support backward compatibility)
 */
bool is_valid_kafka_connection_str(char *connection_str, string &burl, string &bport) {
  if(connection_str == NULL) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka connection string cant be NULL");
    return false;
  }

  string str(connection_str);
  size_t n = count(str.begin(), str.end(), ';');
  if(n>2) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "Kafka connection string format is invalid");
    return false;
  }

  istringstream iss(connection_str);
  getline(iss, burl, ';');
  getline(iss, bport,';');

  if(burl =="" || bport == "") {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka connection string is invalid. hostname or port is empty\n");
      return false;
  }
  return true;
}

/**
 * Connects to a remote kafka broker based on connection string.
 */
NvDsMsgApiHandle nvds_msgapi_connect(char *connection_str,  nvds_msgapi_connect_cb_t connect_cb, char *config_path)
{
  nvds_log_open();

  // If file dump is enabled, skip actual Kafka connection and return a dummy handle
  // This allows the pipeline to start even when Kafka server is not available
  const char *json_dump_env = getenv("NVDS_JSON_DUMP_FILE");
  if (json_dump_env) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "nvds_msgapi_connect: File dump mode enabled (NVDS_JSON_DUMP_FILE=%s), skipping Kafka connection", json_dump_env);
    // Allocate a dummy context to return as handle
    NvDsKafkaClientHandle *kh = new(std::nothrow) NvDsKafkaClientHandle();
    if (kh == NULL) {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka nvds_msgapi_connect: Malloc failed for dummy handle\n");
      return NULL;
    }
    kh->p_instance.producer = NULL;  // No actual Kafka connection
    kh->c_instance.consumer = NULL;
    kh->c_instance.disconnect = true;
    return (NvDsMsgApiHandle)(kh);
  }

  string burl="", bport="";
  if(!is_valid_kafka_connection_str(connection_str, burl, bport))
    return NULL;

  if (test_kafka_broker_endpoint(burl.c_str(), bport.c_str())) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Invalid address or network endpoint down. kafka connect failed\n");
    return NULL;
  }

  NvDsKafkaClientHandle *kh = new(std::nothrow) NvDsKafkaClientHandle();
  if (kh == NULL) {
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka nvds_msgapi_connect : Malloc failed\n");
       return NULL;
  }

  g_strlcpy(kh->brokers , string(burl + ":" + bport).c_str(), MAX_FIELD_LEN);
  /*----Producer initialize----*/
  kh->p_instance.producer = NULL;
  g_strlcpy(kh->p_instance.partition_key_field, DEFAULT_PARTITION_NAME, MAX_FIELD_LEN);
  /*----Consumer initialize----*/
  kh->c_instance.consumer = NULL;
  g_strlcpy(kh->c_instance.consumer_grp_id, DEFAULT_KAFKA_CONSUMER_GROUP, MAX_FIELD_LEN);
  kh->c_instance.disconnect=true;
  kh->c_instance.config="";

  //Create kafka conf object for producer instance
  rd_kafka_conf_t *conf = (rd_kafka_conf_t *) nvds_kafka_client_init(kh);
  if(conf == NULL) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka client initialization for producer fail. Connection failed \n");
    delete kh;
    return NULL;
  }

  if (config_path)
    if(nvds_kafka_read_config(kh, config_path, kh->p_instance.partition_key_field, sizeof(kh->p_instance.partition_key_field), conf) == NVDS_MSGAPI_ERR) {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka config parsing failed. Connection failed \n");
      rd_kafka_conf_destroy(conf);
      delete kh;
      return NULL;
    }

  //Launch the producer instance
  if(nvds_kafka_producer_launch(kh, conf) == NVDS_MSGAPI_ERR) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka client producer launch fail. Connection failed \n");
    delete kh;
    return NULL;
  }
  nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "Kafka connection successful\n");
  return (NvDsMsgApiHandle)(kh);
}

/* Function used by kafka consumer thread to listen & consume incoming messages
 * User callback will be called to forward consumed messages or errors(if any)
 */
void *consume(void *ptr) {
    NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)ptr;

    /* Subscribing to topics will trigger a group rebalance
     * which may take some time to finish, but there is no need
     * for the application to handle this idle period in a special way
     * since a rebalance may happen at any time.
     * Start polling for messages.
     */
    while(!kh->c_instance.disconnect) {
       /* poll for 0ms */
       rd_kafka_message_t *rkm = rd_kafka_consumer_poll(kh->c_instance.consumer, 0);
       if (!rkm) {
            if(kh->c_instance.disconnect)
                break;
            else {
                //If there's no message, sleep for 5ms
                usleep(5000);
                continue;
            }
       }
        /* Consumer_poll() will return either a proper message
         * or a consumer error (rkm->err is set).
        */
        if (rkm->err) {

            /* Consumer errors are generally to be considered
             * informational as the consumer will automatically
             * try to recover from all types of errors.
            */
            if (rkm->err != RD_KAFKA_RESP_ERR__PARTITION_EOF) {
                nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka Consumer error: %s\n", rd_kafka_message_errstr(rkm));
                kh->c_instance.cinfo.subscribe_req_cb(NVDS_MSGAPI_ERR, (void *) rkm->payload, (int) rkm->len, (char *) rd_kafka_topic_name(rkm->rkt), kh->c_instance.cinfo.user_ptr);
           }
            rd_kafka_message_destroy(rkm);
            continue;
        }
       /* Print the message value/payload. */
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "Kafka Message received: topic[%s],\npayload=%s\n",rd_kafka_topic_name(rkm->rkt), (const char *)rkm->payload);
       kh->c_instance.cinfo.subscribe_req_cb(NVDS_MSGAPI_OK, (void *) rkm->payload, (int) rkm->len, (char *) rd_kafka_topic_name(rkm->rkt), kh->c_instance.cinfo.user_ptr);
       rd_kafka_message_destroy(rkm);
    }
    if(!kh->c_instance.disconnect)
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "ERROR: Consumer thread exits. Message consumption stopped\n");
    return NULL;
}

/* Function to subscribe to list of topics
*/
NvDsMsgApiErrorType kafka_create_topic_subscription(NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx) {
    NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)h_ptr;

    /* Convert the list of topics to a format suitable for librdkafka */
    rd_kafka_topic_partition_list_t *subscription = rd_kafka_topic_partition_list_new(num_topics);

    for (int i = 0 ; i < num_topics ; i++)
        rd_kafka_topic_partition_list_add(subscription,
                                                  topics[i],
                                                  /* the partition is ignored
                                                   * by subscribe() */
                                                  RD_KAFKA_PARTITION_UA);

    /* Subscribe to the list of topics */
    rd_kafka_resp_err_t err = rd_kafka_subscribe(kh->c_instance.consumer, subscription);
    if (err) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka : Failed to subscribe to topics. Error: %s\n", rd_kafka_err2str(err));
        rd_kafka_topic_partition_list_destroy(subscription);
        return NVDS_MSGAPI_ERR;
    }

    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "Kafka: Successfully Subscribed to below topic(s):\n");
    for(int i=0; i<num_topics; i++)
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "%s",topics[i]);

    rd_kafka_topic_partition_list_destroy(subscription);

    kh->c_instance.cinfo.subscribe_req_cb = cb;
    kh->c_instance.cinfo.user_ptr         = user_ctx;
    return NVDS_MSGAPI_OK;
}

/* This api will be used to create & intialize kafka consumer
 * A consumer thread is spawned to listen & consume messages on topics specified in the api
 */
NvDsMsgApiErrorType nvds_msgapi_subscribe (NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx) {
    char errstr[512];
    NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)h_ptr;
    if(kh == NULL ) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka connection handle passed for nvds_msgapi_subscribe() = NULL. Subscribe failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(topics == NULL || num_topics <=0 ) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Topics not specified for subscribe. Subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(!cb) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Subscribe callback cannot be NULL. subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(kh->c_instance.consumer) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "Kafka Subscription already exists. Replacing the existing one with newer topics provided\n");
        if(kafka_create_topic_subscription(h_ptr, topics, num_topics, cb, user_ctx) == NVDS_MSGAPI_ERR) {
            return NVDS_MSGAPI_ERR;
        }
        return NVDS_MSGAPI_OK;
    }

    /*Initialization for kafka consumer*/
    kh->c_instance.disconnect = false;

    rd_kafka_conf_t *conf = (rd_kafka_conf_t *) nvds_kafka_client_init(kh);
    if(conf == NULL) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka subscribe() : conf object creation fail. Subscribe failed\n");
    }

  /* Set the consumer group id.
   * All consumers sharing the same group id will join the same
   * group, and the subscribed topic' partitions will be assigned
   * according to the partition.assignment.strategy
   * (consumer config property) to the consumers in the group. */
   if (rd_kafka_conf_set(conf, "group.id", kh->c_instance.consumer_grp_id, errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Error setting consumer group id[%s]. Error string: %s\n", kh->c_instance.consumer_grp_id, errstr);
      rd_kafka_conf_destroy(conf);
      return NVDS_MSGAPI_ERR;
   }

  /* If there is no previously committed offset for a partition
   * the auto.offset.reset strategy will be used to decide where
   * in the partition to start fetching messages.
   * By setting this to earliest the consumer will read all messages
   * in the partition if there was no previously committed offset.
   * */
   if (rd_kafka_conf_set(conf, "auto.offset.reset", "earliest", errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
     nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka subscribe(): Error setting offset configuration - %s\n", errstr);
     rd_kafka_conf_destroy(conf);
     return NVDS_MSGAPI_ERR;
   }

   // Apply user specified config options for consumer instance
   if(nvds_kafka_parse_proto_cfg((char *)kh->c_instance.config.c_str() , conf) != NVDS_MSGAPI_OK){
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Error seen while parsing consumer-proto-cfg options");
       rd_kafka_conf_destroy(conf);
       return NVDS_MSGAPI_ERR;
   }

  /*
   * Create consumer instance.
   *
   * NOTE: rd_kafka_new() takes ownership of the conf object
   *       and the application must not reference it again after
   *       this call.
   */
   kh->c_instance.consumer = rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
   if (kh->c_instance.consumer == NULL) {
     nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka subscribe: Failed to create new consumer: %s\n", errstr);
     rd_kafka_conf_destroy(conf);
     return NVDS_MSGAPI_ERR;
   }


   conf = NULL; /* Configuration object is now owned, and freed,
                 * by the rd_kafka_t consumer instance. */

    /* Redirect all messages from per-partition queues to
     * the main queue so that messages can be consumed with one
     * call from all assigned partitions.
     *
     * The alternative is to poll the main queue (for events)
     * and each partition queue separately, which requires setting
     * up a rebalance callback and keeping track of the assignment:
     * but that is more complex and typically not recommended.
     * */
    rd_kafka_poll_set_consumer(kh->c_instance.consumer);

    if(kafka_create_topic_subscription(h_ptr, topics, num_topics, cb, user_ctx)==NVDS_MSGAPI_ERR) {
        rd_kafka_consumer_close(kh->c_instance.consumer);
        rd_kafka_destroy(kh->c_instance.consumer);
        kh->c_instance.consumer = NULL;
        return NVDS_MSGAPI_ERR;
    }

    //Create a new thread
    int rv = pthread_create(&kh->c_instance.cinfo.consumer_tid, NULL, &consume, kh);
    if(rv) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka nvds_msgapi_subscribe(): Consumer Thread creation failed. Subscription to topics failed\n");
        rd_kafka_consumer_close(kh->c_instance.consumer);
        rd_kafka_destroy(kh->c_instance.consumer);
        kh->c_instance.consumer = NULL;
        return NVDS_MSGAPI_ERR;
    }
    return NVDS_MSGAPI_OK;
}

//There could be several synchronous and asychronous send operations in flight.
//Once a send operation callback is received the course of action  depends on if it's synch or async
// -- if it's sync then the associated complletion flag should  be set
// -- if it's asynchronous then completion callback from the user should be called
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf)
{
  NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)h_ptr;
  if(h_ptr == NULL ) {
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka connection handle passed for send() = NULL. Send failed\n");
       return NVDS_MSGAPI_ERR;
  }
  if(topic == NULL || !strcmp(topic, "")) {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka topic not specified.Send failed\n");
      return NVDS_MSGAPI_ERR;
  }
  if(payload == NULL || nbuf <=0) {
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka: Either send payload is NULL or payload length <=0. Send failed\n");
       return NVDS_MSGAPI_ERR;
  }

  char idval[100];
  int retval;

  nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "nvds_msgapi_send: payload=%.*s \n topic = %s\n", nbuf, payload, topic);

  // parition key retrieved from config file
  retval = json_get_key_value((const char *)payload, nbuf, kh->p_instance.partition_key_field , idval, sizeof(idval));

  if (retval)
    return nvds_kafka_client_send(kh, payload, nbuf, topic, 1, NULL, NULL, idval, strlen(idval));
  else {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "nvds_msgapi_send: \
                  no matching json field found based on kafka key config; \
                  using default partition\n");

      return nvds_kafka_client_send(kh, payload, nbuf, topic, 1, NULL, NULL, NULL, 0);
  }
}

/* nvds_msgapi function for asynchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf,  nvds_msgapi_send_cb_t send_callback, void *user_ptr)
{
  NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)h_ptr;
  if(h_ptr == NULL ) {
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka connection handle passed for send_async() = NULL. Send failed\n");
       return NVDS_MSGAPI_ERR;
  }
  if(topic == NULL || !strcmp(topic, "")) {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka topic not specified.Send failed\n");
      return NVDS_MSGAPI_ERR;
  }
  if(payload == NULL || nbuf <=0) {
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "kafka: send_async() either payload is NULL or payload length <=0. Send failed\n");
       return NVDS_MSGAPI_ERR;
  }

  char idval[100];
  int retval;

  nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "nvds_msgapi_send_async: payload=%.*s \n topic = %s\n", nbuf, payload, topic);

  // parition key retrieved from config file
  retval = json_get_key_value((const char *)payload, nbuf, kh->p_instance.partition_key_field , idval, sizeof(idval));

  if (retval) {
    return nvds_kafka_client_send(kh, payload, nbuf, topic, 0, user_ptr, send_callback, idval, strlen(idval));
  }
  else {
    nv::Frame protomsg;
    protomsg.ParseFromString(string((char *) payload, nbuf));
    string proto_sensor_id="";
    proto_sensor_id = protomsg.sensorid();
    if(proto_sensor_id != "") {
        return nvds_kafka_client_send(kh, payload, nbuf, topic, 0, user_ptr, send_callback, (char *)proto_sensor_id.c_str(), proto_sensor_id.length());
    }
    else {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "no matching json field found \
                based on kafka key config; using default partition\n");
        return nvds_kafka_client_send(kh, payload, nbuf, topic, 0, user_ptr, send_callback, NULL, 0);
    }
  }
}

/* nvds_msgapi function for performing work in the nvmsgbroker plugin. In kafka case, poll the kafka producer
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr)
{
  if(h_ptr == NULL ) {
       nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Kafka connection handle passed for dowork() = NULL. No actions taken\n");
       return;
  }

  nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "nvds_msgapi_do_work\n");
  nvds_kafka_client_poll((NvDsKafkaClientHandle *) h_ptr);
}

/* nvds_msgapi function for disconnect
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr)
{
  if (!h_ptr) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "nvds_msgapi_disconnect called with null handle\n");
    return NVDS_MSGAPI_ERR;
  }

  NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)h_ptr;

  // If in file dump mode (no actual Kafka connection), just free the context
  if (kh->p_instance.producer == NULL) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "nvds_msgapi_disconnect: File dump mode, no Kafka connection to close");
    delete kh;
    kh = NULL;
    nvds_log_close();
    return NVDS_MSGAPI_OK;
  }

  nvds_kafka_client_finish(kh);
  delete kh;
  kh = NULL;
  nvds_log_close();
  return NVDS_MSGAPI_OK;
}

/**
  * Returns version of API supported by this adaptor
  */
char *nvds_msgapi_getversion()
{
  return (char *)NVDS_MSGAPI_VERSION;
}

//Returns underlying framework/protocol name
char *nvds_msgapi_get_protocol_name()
{
  return (char *)NVDS_MSGAPI_PROTOCOL;
}

//Query connection signature
NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str, char *cfg, char *output_str, int max_len) {
    strcpy(output_str,"");

    if(broker_str == NULL || cfg == NULL) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: broker_str or cfg path cant be NULL");
        return NVDS_MSGAPI_ERR;
    }

    //check if share-connection config option is turned ON
    char reuse_connection[MAX_FIELD_LEN]="";
    if(fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_RDKAFKA_SHARE_CONNECTION, reuse_connection, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing kafka share-connection config option");
        return NVDS_MSGAPI_ERR;
    }
    if(strcmp(reuse_connection, "1")) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "nvds_msgapi_connection_signature: Kafka connection sharing disabled. Hence connection signature cant be returned");
        return NVDS_MSGAPI_OK;
    }

    if(max_len < 2 * SHA256_DIGEST_LENGTH + 1) {
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: insufficient output string length. Atleast %d bytes needed", 2 * SHA256_DIGEST_LENGTH + 1);
        return NVDS_MSGAPI_ERR;
    }

    string burl="", bport="";
    if(!is_valid_kafka_connection_str(broker_str, burl, bport))
        return NVDS_MSGAPI_ERR;

    char proto_cfg[MAX_FIELD_LEN]="", producer_proto_cfg[MAX_FIELD_LEN]="", consumer_proto_cfg[MAX_FIELD_LEN]="";

    //Read config to fetch proto-cfg ,a generic setting applied for both producer & consumer
    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_RDKAFKA_CFG, proto_cfg, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) || \
       (strip_quote(proto_cfg, "proto-cfg", NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK)){
            nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing kafka proto-cfg");
            return NVDS_MSGAPI_ERR;
    }

    //Read config to fetch producer-proto-cfg specific for producer instance
    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_RDKAFKA_PRODUCER_CFG, producer_proto_cfg, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) || \
       (strip_quote(producer_proto_cfg, "producer-proto-cfg", NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK)){
            nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing kafka producer-proto-cfg");
            return NVDS_MSGAPI_ERR;
    }

    //Read config to fetch consumer-proto-cfg specific for consumer instance
    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_RDKAFKA_CONSUMER_CFG, consumer_proto_cfg, MAX_FIELD_LEN, NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK) || \
       (strip_quote(consumer_proto_cfg, "consumer-proto-cfg", NVDS_KAFKA_LOG_CAT) != NVDS_MSGAPI_OK)){
            nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing kafka consumer-proto-cfg");
            return NVDS_MSGAPI_ERR;
    }

    string unsorted_cfg = string(proto_cfg) + string(producer_proto_cfg) + string(consumer_proto_cfg);
    string sorted_cfg = sort_key_value_pairs(unsorted_cfg);
    string kafka_connection_signature = generate_sha256_hash(burl + bport + sorted_cfg);
    strcpy(output_str, kafka_connection_signature.c_str());
    return NVDS_MSGAPI_OK;
}
