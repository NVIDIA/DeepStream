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

#ifndef __NV_MSGBROKER_INTERNAL_H__
#define __NV_MSGBROKER_INTERNAL_H__

#ifdef __cplusplus
extern "C"
{
#endif

#define DEFAULT_RETRY_INTERVAL 2
#define DEFAULT_MAX_RETRY_LIMIT 360
#define DEFAULT_WORK_INTERVAL 10000

//config settings applicable for nvmsgbroker library
const char *nvmsgbrokerConfig = "/opt/nvidia/deepstream/deepstream/sources/libs/nvmsgbroker/cfg_nvmsgbroker.txt";

bool auto_reconnect;          //Flag to indicate autoreconnection
unsigned retry_interval;      //connection retry interval in secs
unsigned max_retry_limit;     //max retry limit in secs
unsigned work_interval;       //interval at which to perform work, in microseconds

typedef enum {
    SUCCESS,
    FAIL,
    ERROR
}NvMsgBrokerReturnVal;

//Structure used to store synchronization primitives
typedef struct {
    pthread_t  tid;            //thread id
    pthread_mutex_t lock;      //lock
    pthread_cond_t  cv;        //condition variable
}NvMsgBrokerThread_t ;

//Pointer to Adapter Api's
typedef struct proto_adapter{
    string protocol;           //Protocol name
    void *so_handle;           //Adapter shared library handle
    int ref_count;             //count num of lib open references
    bool subscribe_capability; //Flag to indicate if the adapter lib supports topic subscription
    bool signature_capability; //Indicate if adapter lib implements connection signature query api
    NvDsMsgApiHandle (*nvds_msgapi_connect_ptr)(char *connection_str, nvds_msgapi_connect_cb_t connect_cb, char *config_path);
    NvDsMsgApiErrorType (*nvds_msgapi_disconnect_ptr)(NvDsMsgApiHandle h_ptr);
    NvDsMsgApiErrorType (*nvds_msgapi_send_async_ptr)(NvDsMsgApiHandle conn, char *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr);
    void (*nvds_msgapi_do_work_ptr) (NvDsMsgApiHandle h_ptr);
    NvDsMsgApiErrorType (*nvds_msgapi_subscribe_ptr)(NvDsMsgApiHandle conn, char **topics, int num_topics, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx);
    char* (*nvds_msgapi_getversion_ptr)(void);
    char* (*nvds_msgapi_get_protocol_name_ptr)(void);
    NvDsMsgApiErrorType (*nvds_msgapi_connection_signature_ptr)(char *connection_str, char *config_path, char *output_str, int max_len);

    proto_adapter():protocol(""), so_handle(NULL), ref_count(0), subscribe_capability(true), signature_capability(false) {}
}AdapterLib;

//Structure to hold info used within a NvMsgBrokerHandle
typedef struct {
    int ref_count;                                                                              //count num of components calling connect
    string broker_proto_so;                                                                     //Adapter library so name
    string broker_conn_str;                                                                     //Broker connection string(hostname;port)
    string broker_cfg_path;                                                                     //Broker config file path
    string broker_signature;                                                                    //Broker connection string signature
    NvDsMsgApiHandle adapter_h_ptr;                                                             //Adapter Connection handle
    AdapterLib *adapter_lib;                                                                    //Adapter shared library handle
    nvds_msgapi_subscribe_request_cb_t subscribe_cb;                                            //callback for consuming messages from adapter lib
    unordered_set<nv_msgbroker_connect_cb_t>  connect_cb_set;                                   //saved list of connect cb() for the handle
    unordered_set<nv_msgbroker_send_cb_t>  send_cb_set;                                         //saved list of send cb() for the handle
    unordered_map<string, set<pair<nv_msgbroker_subscribe_cb_t, void *>>> subscribe_topic_map;  //topic → {subscribe_cb(), user_ctx}
    NvMsgBrokerThread_t do_work_thread;                                                         //Used for calling adapter do_work()
    pthread_mutex_t subscribe_lock;                                                             //Used for calling adapter subscribe()
    int pending_send_cnt;                                                                       //num of msgs waiting to be sent
    bool disconnect;                                                                            //Flag used to notify dowork thread to quit
    bool reconnecting;                                                                          //Flag to notify if reconnection attempt is in progress
}NvMsgBrokerHandle;

//Structure to store user data during a send_async operation
typedef struct {
    nv_msgbroker_send_cb_t send_cb;                                                             //User send callback func pointer
    void *user_ptr;                                                                             //Pointer to user context
    NvMsgBrokerHandle *h_ptr;                                                                   //Pointer to msgbroker connection handle
}NvMsgBrokerSendCB_info;

//Containers used by nvmsgbroker library
map<pair<string, AdapterLib *>, NvMsgBrokerHandle *> conn_string_handle_map;                    //Map connection string --> connection  handle
unordered_set<NvMsgBrokerHandle *> conn_Handle_list;                                            //List of connection handles
unordered_map<NvDsMsgApiHandle, NvMsgBrokerHandle *> adapter_msgbroker_map;                     //Map Adapter handle --> msgbroker handle
unordered_map<void *, AdapterLib *> so_handle_map;                                              //Map so_handle --> adapter lib

//Lock to use msgbroker connection handle h_ptr
pthread_mutex_t h_ptr_lock;

//Callbacks and functions used internally within nvmsgbroker lib
void *do_work_func(void *);
void adapter_connect_cb(NvDsMsgApiHandle *adapter_h_ptr, NvDsMsgApiEventType ds_evt);
void adapter_send_cb(void *user_ptr,  NvDsMsgApiErrorType completion_flag);
void adapter_subscribe_cb(NvDsMsgApiErrorType flag, void *msg, int msglen, char *topic, void *user_ptr);
void handle_error(void *handle, pthread_mutex_t *lock, AdapterLib * lib, const char *log, const char *error);
bool fetch_adapter_api(void *so_handle, AdapterLib *LIB);
NvMsgBrokerReturnVal reconnect(NvMsgBrokerHandle *h_ptr);
void __attribute__ ((constructor)) nvmsgbroker_init(void);
void __attribute__ ((destructor)) nvmsgbroker_deinit(void);

#ifdef __cplusplus
}
#endif

#endif