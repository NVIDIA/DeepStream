/*
 * SPDX-FileCopyrightText: Copyright (c) 2018 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
/*
 * librdkafka - Apache Kafka C library
 *
 * Copyright (c) 2012-2022, Magnus Edenhill
 *               2023, Confluent Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * kafka client based on Simple Apache Kafka producer from librdkafka
 * (https://github.com/confluentinc/librdkafka)
 */

#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <glib.h>
#include "rdkafka.h"
#include "nvds_logger.h"
#include "kafka_client.h"

/**
 * @brief Message delivery report callback.
 *
 * This callback is called exactly once per message, indicating if
 * the message was succesfully delivered
 * (rkmessage->err == RD_KAFKA_RESP_ERR_NO_ERROR) or permanently
 * failed delivery (rkmessage->err != RD_KAFKA_RESP_ERR_NO_ERROR).
 *
 * The callback is triggered from rd_kafka_poll() and executes on
 * the application's thread.
 */
static void dr_msg_cb (rd_kafka_t *rk,
                       const rd_kafka_message_t *rkmessage, void *opaque) {
  NvDsMsgApiErrorType dserr;
  if (rkmessage->err) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Message delivery failed: %s\n", \
            rd_kafka_err2str(rkmessage->err));
  }
  else
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "Message delivered (%zd bytes, " \
                        "partition %d)\n", rkmessage->len, rkmessage->partition);

  switch (rkmessage->err) {
    case RD_KAFKA_RESP_ERR_NO_ERROR:
        dserr = NVDS_MSGAPI_OK;
        break;

   case RD_KAFKA_RESP_ERR_UNKNOWN_TOPIC_OR_PART:
        dserr = NVDS_MSGAPI_UNKNOWN_TOPIC;
        break;

    default:
        dserr = NVDS_MSGAPI_ERR;
        break;
  };
  ((NvDsKafkaSendCompl *)(rkmessage->_private))->sendcomplete(dserr);

  delete ((NvDsKafkaSendCompl *)(rkmessage->_private));

}

NvDsKafkaSyncSendCompl::NvDsKafkaSyncSendCompl(uint8_t *cflag) {
  compl_flag = cflag;
}

/**
 * Method that gets invoked when sync send operation is completed
 */
void NvDsKafkaSyncSendCompl::sendcomplete(NvDsMsgApiErrorType senderr) {
  *compl_flag = 1;
  err = senderr;
}


void NvDsKafkaSendCompl::sendcomplete(NvDsMsgApiErrorType senderr) {
  printf("wrong class\n");
}

NvDsMsgApiErrorType NvDsKafkaSendCompl::get_err() {
  return NVDS_MSGAPI_OK;
}

/**
 * Method that gets invoked when sync send operation is completed
 */
NvDsMsgApiErrorType NvDsKafkaSyncSendCompl::get_err() {
  return err;
}


NvDsKafkaAsyncSendCompl::NvDsKafkaAsyncSendCompl(void *ctx, nvds_msgapi_send_cb_t cb) {
  user_ptr = ctx;
  async_send_cb = cb;
}

/**
 * Method that gets invoked when async send operation is completed
 */
void NvDsKafkaAsyncSendCompl::sendcomplete(NvDsMsgApiErrorType senderr) {
  // simply call any registered callback
  if (async_send_cb)
    async_send_cb(user_ptr, senderr);
}

/*
 * The kafka protocol adaptor expects the client to manage handle usage and retirement.
 * Specifically, client has to ensure that once a handle is retired through disconnect,
 * that it does not get used for either send or do_work.
 * While the library iplements a best effort mechanism to ensure that calling into these
 * functions with retired handles is  gracefully handled, this is not done in a thread-safe
 * manner
 *
 * Also, note thatrdkafka is inherently thread safe and therefore there is no need to implement separate
 * locking mechanisms in the kafka protocol adaptor for methods calling directly in rdkafka
 *
 */

void * nvds_kafka_client_init(NvDsKafkaClientHandle *kh) {
    char errstr[512];

    //Create Kafka client configuration place-holder
    rd_kafka_conf_t *conf = rd_kafka_conf_new();

    /* Set bootstrap broker(s) as a comma-separated list of host or host:port (default port 9092).
     * librdkafka will use the bootstrap brokers to acquire the full set of brokers from the cluster. */
    if (rd_kafka_conf_set(conf, "bootstrap.servers", kh->brokers, errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
      nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Error connecting kafka broker: %s\n",errstr);
      rd_kafka_conf_destroy(conf);
      return NULL;
    }

    return conf;
}


//There could be several synchronous and asychronous send operations in flight.
//Once a send operation callback is received the course of action  depends on if it's sync or async
// -- if it's sync then the associated completion flag should  be set
// -- if it's asynchronous then completion callback from the user should be called along with context
NvDsMsgApiErrorType nvds_kafka_client_send(void *kv,  const uint8_t *payload, int len, char *topicname, int sync, void *ctx, nvds_msgapi_send_cb_t cb, char *key, int keylen)
{
  NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)kv;
  uint8_t done = 0;

  NvDsKafkaSendCompl *scd;
  if (sync) {
    NvDsKafkaSyncSendCompl *sc= new NvDsKafkaSyncSendCompl(&done);
    scd = sc;
  } else {
    NvDsKafkaAsyncSendCompl *sc= new NvDsKafkaAsyncSendCompl(ctx, cb);
    scd = sc;
  }

  if (!kh) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "send called on NULL handle \n");
    return NVDS_MSGAPI_ERR;
  }

  rd_kafka_resp_err_t err = rd_kafka_producev(
                        /* Producer handle */
                        kh->p_instance.producer,
                        /* Topic name */
                        RD_KAFKA_V_TOPIC(topicname),
                        /* Make a copy of the payload. */
                        RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                        /* Message value and length */
                        RD_KAFKA_V_VALUE((void *)payload, len),
                        /* Message Key and key length */
                        RD_KAFKA_V_KEY(key, keylen),
                        /* Per-Message opaque, provided in
                         * delivery report callback as
                         * msg_opaque. */
                        RD_KAFKA_V_OPAQUE(scd),
                        /* End sentinel */
                        RD_KAFKA_V_END);

  if(err) {
    if (err == RD_KAFKA_RESP_ERR__QUEUE_FULL) {
       /* If the internal queue is full, discard the
        * message and log the error.
        * The internal queue represents both
        * messages to be sent and messages that have
        * been sent or failed, awaiting their
        * delivery report callback to be called.
        *
        * The internal queue is limited by the
        * configuration property
        * queue.buffering.max.messages */
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR,"rd_kafka_produce: Internal queue is full, discarding payload\n");
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "rd_kafkaproduce: Discarding payload=%.*s \n topic = %s\n",len, payload, topicname);
     }
     else
     {
       /**
        * Failed to *enqueue* message for producing.
        */
        nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR,"Failed to schedule kafka send: Error= [%s] on topic <%s>\n", rd_kafka_err2str(err), topicname);
     }
     return NVDS_MSGAPI_ERR;
  }
  else
  {
    NvDsMsgApiErrorType err;
    if  (!sync)
      return NVDS_MSGAPI_OK;
    else
    {
      while (sync && !done) {
        usleep(1000);
        rd_kafka_poll(kh->p_instance.producer, 0/*non-blocking*/);
      }
      err = (scd)->get_err();
      return err;
    }
  }
}

/* This function used to set kafka key=value config to a kafka instance
 * @param conf : rdkafka conf object
 * @param key
 * @param value
 */
NvDsMsgApiErrorType nvds_kafka_client_setconf(rd_kafka_conf_t *conf, char *key, char *val)
{
  char errstr[512];
  
  if (rd_kafka_conf_set(conf, key, val , errstr, sizeof(errstr)) \
           != RD_KAFKA_CONF_OK) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Error setting config setting %s; %s\n", key, errstr );
    return NVDS_MSGAPI_ERR;
  } else {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_INFO, "set config setting %s to %s\n", key, val);
    return NVDS_MSGAPI_OK;
  }
}

/* This function used to launch kafka producer instance
 * @param kv   : nvds_Kafka connection handle
 * @param conf : temporary rdkafka conf object
 */

NvDsMsgApiErrorType nvds_kafka_producer_launch(void *kv, rd_kafka_conf_t *conf)
{
   rd_kafka_t *rk = NULL;
   NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)kv;
   char errstr[512];

   /* Set the delivery report callback.
    * This callback will be called once per message to inform the application if delivery succeeded or failed.
    * See dr_msg_cb() above. */
    rd_kafka_conf_set_dr_msg_cb(conf, dr_msg_cb);

   /*
    * Create producer instance.
    * NOTE: rd_kafka_new() takes ownership of the conf object
    *       and the application must not reference it again after this call.
    */
   rk = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
   if (!rk) {
     nvds_log(NVDS_KAFKA_LOG_CAT, LOG_ERR, "Failed to create new producer: %s\n", errstr);
     rd_kafka_conf_destroy(conf);
     return NVDS_MSGAPI_ERR;
   }

   conf = NULL; /* Configuration object is now owned, and freed,
                 * by the rd_kafka_t producer instance. */

   kh->p_instance.producer = rk;

   return NVDS_MSGAPI_OK;
}

/* This function called during disconnect operation
 * @param kv   : nvds_Kafka connection handle
 */

void nvds_kafka_client_finish(void *kv)
{
  NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)kv;

  /* Destroy the producer instance */
  if(kh->p_instance.producer != NULL) {
    rd_kafka_flush (kh->p_instance.producer, 10000);
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "kafka: Closing producer instance\n");
    rd_kafka_destroy(kh->p_instance.producer);
  }

  if(kh->c_instance.consumer != NULL) {
    nvds_log(NVDS_KAFKA_LOG_CAT, LOG_DEBUG, "kafka: Closing consumer instance\n");
    kh->c_instance.disconnect = true;
    pthread_join(kh->c_instance.cinfo.consumer_tid, NULL);

    /* Close the consumer: commit final offsets and leave the group.*/
    rd_kafka_consumer_close(kh->c_instance.consumer);
    /* Destroy the consumer */
    rd_kafka_destroy(kh->c_instance.consumer);
  }

}

/* This function used to periodically poll on kafka producer instance
 * @param kv   : nvds_Kafka connection handle
 */
void nvds_kafka_client_poll(void *kv)
{
  NvDsKafkaClientHandle *kh = (NvDsKafkaClientHandle *)kv;
  if (kh)
    rd_kafka_poll(kh->p_instance.producer, 0/*non-blocking*/);
}
