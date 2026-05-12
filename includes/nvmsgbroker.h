/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * This file defines the NvMsgBroker interface.
 * The interfaces is used by applications to send and receive
 * messages from remote entities and services to deliver events, allow
 * configuration of settings etc.
 */

#ifndef __NV_MSGBROKER_H__
#define __NV_MSGBROKER_H__

#ifdef __cplusplus
extern "C"
{
#endif

#define NV_MSGBROKER_VERSION "2.0"
#define LOG_CAT "DSLOG:NV_MSGBROKER"

/*
 * Defines status for operations in the NvMsgBroker interface
 */
typedef enum {
    NV_MSGBROKER_API_OK,
    NV_MSGBROKER_API_ERR,
    NV_MSGBROKER_API_RECONNECTING,
    NV_MSGBROKER_API_NOT_SUPPORTED
} NvMsgBrokerErrorType;

/*
 * Defines structure of a client message packet
 */

typedef struct {
    char *topic;
    void *payload;
    size_t payload_len;
} NvMsgBrokerClientMsg;

/* 
 * Handle to be used in NvMsgBroker API interface
 */
typedef void *NvMsgBrokerClientHandle;


/** Connect callback method registered during connect
  *
  * @param[in] h_ptr             Msgbroker Connection handle
  * @param[in] status            Connection status
  */
typedef void (*nv_msgbroker_connect_cb_t)(NvMsgBrokerClientHandle h_ptr, NvMsgBrokerErrorType status );

/** Send callback method registered during send_async
  * @param[in] user_ptr          Pointer passed during send_async for context
  * @param[in] status            Completion status of send operation
  */
typedef void (*nv_msgbroker_send_cb_t)(void *user_ptr,  NvMsgBrokerErrorType status);


/** subscribe callback method registered during subscribe
  * @param[in] status            Completion status of send operation
  * @param[in] msg               Consumer Message. Should be copied before function return
  * @param[in] msglen            Length of message consumed
  * @param[in] topic             Subscribed Topic where message is received
  * @param[in] user_ptr          Pointer passed during send_async for context
  */
typedef void (*nv_msgbroker_subscribe_cb_t)(NvMsgBrokerErrorType status, void *msg, int msglen, char *topic, void *user_ptr);


/** Connect to a remote entity by calling into msgbroker library
  *
  * @param[in] broker_conn_str   A connection string (optional) with format
  * @param[in] broker_proto_lib  Full Path to Message protocol adapter library
  * @param[in] connect_cb        A pointer to a callback function for events associated with the connection.
  * @param[in] cfg               Pathname of a configuration file passed to be passed to the message protocol adapter
  *
  * @return                      A connection handle.
 */
NvMsgBrokerClientHandle   nv_msgbroker_connect(char *broker_conn_str, char *broker_proto_lib, nv_msgbroker_connect_cb_t connect_cb, char *cfg);

 /* Send a message asynchronously based on non-blocking semantics
  * @param[in] h_ptr             connection handle to Message Broker library
  * @param[in] message           Message packet which has details of message, topic, priority ..etc
  * @param[in] cb                callback to be invoked to notify status of send
  * @param[in] user_ctx          pointer to pass to callback for context
  *
  * @return                      Completion status of send operation */
NvMsgBrokerErrorType nv_msgbroker_send_async (NvMsgBrokerClientHandle h_ptr, NvMsgBrokerClientMsg message, nv_msgbroker_send_cb_t cb, void *user_ctx);

 /** Subscribe to a remote entity for receiving messages on particular topic(s)
  * @param[in] h_ptr             Connection handle
  * @param[in] topics            pointer to array of topics  (cannot be NULL)
  * @param[in] num_topics        number of topics
  * @param[in] cb                A pointer to a callback function to forward consumed message
  * @param[in] user_ctx          user ptr to be passed to callback for context
  * @return Completion status of send operation
  **/
NvMsgBrokerErrorType nv_msgbroker_subscribe(NvMsgBrokerClientHandle h_ptr, char ** topics, int num_topics,  nv_msgbroker_subscribe_cb_t cb, void *user_ctx);


 /* Disconnect call to notify msgbroker library for connection termination
  * @param[in] h_ptr             Connection handle
  *
  * @return                      status of disconnect
  */
NvMsgBrokerErrorType nv_msgbroker_disconnect(NvMsgBrokerClientHandle h_ptr);

/* Version of Nvmsgbroker interface
 *
 * @return [out]                 version of Nvmsgbroker interface supported by msgbroker library in MAJOR.MINOR format
 */
char *nv_msgbroker_version(void);

#ifdef __cplusplus
}
#endif

#endif
