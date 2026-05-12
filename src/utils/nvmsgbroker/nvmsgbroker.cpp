/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <sys/time.h>
using namespace std;

#include "nvmsgbroker.h"
#include "nvds_msgapi.h"
#include "nvds_logger.h"
#include "nvmsgbroker_internal.h"

void nvmsgbroker_init(){
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    //Parse nvmsgbroker config file
    GKeyFile *gcfg_file = g_key_file_new();
    if (g_key_file_load_from_file (gcfg_file, nvmsgbrokerConfig, G_KEY_FILE_NONE, &error)) {
        const char *grpname = "nvmsgbroker";
        keys = g_key_file_get_keys(gcfg_file, grpname, NULL, &error);
        for (key = keys; *key; key++) {
            if (!g_strcmp0(*key, "auto-reconnect")) {
                auto_reconnect = g_key_file_get_boolean(gcfg_file, grpname, "auto-reconnect",&error);
            }
            else if (!g_strcmp0(*key, "retry-interval")) {
                retry_interval = g_key_file_get_integer(gcfg_file, grpname, "retry-interval", &error);
            }
            else if (!g_strcmp0(*key, "max-retry-limit")) {
                max_retry_limit = g_key_file_get_integer(gcfg_file, grpname, "max-retry-limit", &error);
            }
            else if (!g_strcmp0(*key, "work-interval")) {
                work_interval = g_key_file_get_integer(gcfg_file, grpname, "work-interval", &error);
            }
        }
    }
    else {
        nvds_log(LOG_CAT, LOG_ERR , "nvmsgbroker init: Failed to parse configfile [%s]. Using default configs\n", nvmsgbrokerConfig);
        auto_reconnect  = true;
        retry_interval  = DEFAULT_RETRY_INTERVAL;
        max_retry_limit = DEFAULT_MAX_RETRY_LIMIT;
        work_interval = DEFAULT_WORK_INTERVAL;
    }

    //Free allocations
    if(keys != NULL)
        g_strfreev(keys);
    if(error != NULL)
        g_error_free (error);
    if(gcfg_file != NULL)
        g_key_file_free(gcfg_file);
}

void nvmsgbroker_deinit(){}

/*
 * Receive connect status callbacks from Message adapter
 */
void adapter_connect_cb(NvDsMsgApiHandle *adapter_h_ptr, NvDsMsgApiEventType ds_evt) {
    //Access adapter handle and notify all the callers of connect status
    if(adapter_msgbroker_map.count(adapter_h_ptr)) {
        NvMsgBrokerHandle *h_ptr = adapter_msgbroker_map[adapter_h_ptr];
        if(ds_evt != NVDS_MSGAPI_EVT_SUCCESS){
            if(auto_reconnect) {
                pthread_mutex_lock(&h_ptr_lock);
                if(!h_ptr->reconnecting){
                    h_ptr->reconnecting = true;
                    //notify callers of state changed to reconnecting
                    for(auto &user_cb : h_ptr->connect_cb_set)
                        user_cb(h_ptr, NV_MSGBROKER_API_RECONNECTING);
                    //signal do_work thread
                    pthread_mutex_lock(&h_ptr->do_work_thread.lock);
                    pthread_cond_signal(&h_ptr->do_work_thread.cv);
                    pthread_mutex_unlock(&h_ptr->do_work_thread.lock);
                    pthread_mutex_unlock(&h_ptr_lock);
                }
                else
                    pthread_mutex_unlock(&h_ptr_lock);
            }
            else {
                for(auto &user_cb : h_ptr->connect_cb_set)
                    user_cb(h_ptr, NV_MSGBROKER_API_ERR);
            }
        }
    }
}

//attempt reconnection
NvMsgBrokerReturnVal reconnect(NvMsgBrokerHandle *h_ptr){
    NvDsMsgApiHandle adapter_conn_handle;

    adapter_conn_handle = h_ptr->adapter_lib->nvds_msgapi_connect_ptr((char *) h_ptr->broker_conn_str.c_str(),
                          (nvds_msgapi_connect_cb_t) adapter_connect_cb, (char *) h_ptr->broker_cfg_path.c_str());
    if(adapter_conn_handle) {
        //create new subsrciption using stored subscription details
        if(h_ptr->adapter_lib->subscribe_capability && h_ptr->subscribe_topic_map.size()) {
            size_t max_topic_len=0;
            for(auto const &it: h_ptr->subscribe_topic_map)
            if(it.first.length() > max_topic_len)
                max_topic_len = it.first.length();

            char *full_topic_list[h_ptr->subscribe_topic_map.size()];
            int i=0;
            for(auto const& it:h_ptr->subscribe_topic_map) {
                full_topic_list[i++] = (char *) it.first.c_str();
            }
            if(h_ptr->adapter_lib->nvds_msgapi_subscribe_ptr(adapter_conn_handle, full_topic_list,
               h_ptr->subscribe_topic_map.size(), adapter_subscribe_cb, h_ptr) != NVDS_MSGAPI_OK) {

                //when subscription fails : try to close adapter context created above
                if(h_ptr->adapter_lib->nvds_msgapi_disconnect_ptr(adapter_conn_handle) != NVDS_MSGAPI_OK) {
                    //Adapter Disconnection fail. Fatal error
                    nvds_log(LOG_CAT, LOG_ERR,"nvmsgbroker reconnect: Error disconnecting adapter ctx");
                    for(auto &user_cb : h_ptr->connect_cb_set) {
                        user_cb(h_ptr, NV_MSGBROKER_API_ERR);
                    }
                    return ERROR;
                }
                else {
                    //Connection success but subscription fail. Overall Reconnection attempt failed
                    return FAIL;
                }
            }
        }
        //Success. save newly created adapter context details
        adapter_msgbroker_map[adapter_conn_handle] = h_ptr;
        h_ptr->adapter_h_ptr = adapter_conn_handle;
        for(auto &user_cb : h_ptr->connect_cb_set) {
            user_cb(h_ptr, NV_MSGBROKER_API_OK);
        }
        return SUCCESS;
    }
    else //New adapter ctx creation fail
        return FAIL;
}

void handle_error(void *handle, pthread_mutex_t *lock, AdapterLib * lib, const char *log, const char *error){
    nvds_log(LOG_CAT, LOG_ERR , "%s %s", log, error);
    if(handle)  dlclose(handle);
    if(lib) {
        lib->ref_count--;
        if(!lib->ref_count) {
            so_handle_map.erase(lib->so_handle);
            delete(lib);
        }
    }
    if(lock)    pthread_mutex_unlock(lock);
}

bool fetch_adapter_api(void *so_handle, AdapterLib *adapter_lib){
    char* error=NULL;

    *(void **) (&adapter_lib->nvds_msgapi_connect_ptr) = dlsym(so_handle, "nvds_msgapi_connect");
    if ((error = dlerror()) != NULL) {
        handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load api nvds_msgapi_connect. Error:", error);
        return false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_disconnect_ptr) = dlsym(so_handle, "nvds_msgapi_disconnect");
    if ((error = dlerror()) != NULL) {
        handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load api nvds_msgapi_disconnect. Error:", error);
        return false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_send_async_ptr) = dlsym(so_handle, "nvds_msgapi_send_async");
    if ((error = dlerror()) != NULL) {
        handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load api nvds_msgapi_send_async. Error:", error);
        return false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_do_work_ptr) = dlsym(so_handle, "nvds_msgapi_do_work");
    if ((error = dlerror()) != NULL) {
        handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load api nvds_msgapi_do_work. Error:", error);
        return false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_subscribe_ptr) = dlsym(so_handle, "nvds_msgapi_subscribe");
    if ((error = dlerror()) != NULL) {
        nvds_log(LOG_CAT, LOG_DEBUG, "subscription to topics not enabled in adapter lib");
        adapter_lib->subscribe_capability = false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_getversion_ptr) = dlsym(so_handle, "nvds_msgapi_getversion");
    if ((error = dlerror()) != NULL) {
        handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load api nvds_msgapi_getversion. Error:", error);
        return false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_get_protocol_name_ptr) = dlsym(so_handle, "nvds_msgapi_get_protocol_name");
    if ((error = dlerror()) != NULL) {
        handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load api nvds_msgapi_get_protocol_name. Error:", error);
        return false;
    }
    *(void **) (&adapter_lib->nvds_msgapi_connection_signature_ptr) = dlsym(so_handle, "nvds_msgapi_connection_signature");
    if ((error = dlerror()) != NULL) {
        nvds_log(LOG_CAT, LOG_DEBUG, "nvds_msgapi_connection_signature api not implemented in adapter lib");
    }
    else        adapter_lib->signature_capability = true;
    return true;
}

NvMsgBrokerClientHandle nv_msgbroker_connect(char *broker_conn_str, char *broker_proto_lib, nv_msgbroker_connect_cb_t connect_cb, char *cfg){
    //check adapter library for dlopen
    if(!broker_proto_lib) {
        nvds_log(LOG_CAT, LOG_ERR, "Path to broker proto/adapter lib cant be NULL. Connection establishment failed\n");
        return NULL;
    }
    else {
        pthread_mutex_lock(&h_ptr_lock);
        AdapterLib *adapter_lib=NULL;
        bool share_connection = false;
        char* error=NULL;
        void *so_handle = dlopen(broker_proto_lib, RTLD_LAZY);
        if (!so_handle) {
            pthread_mutex_unlock(&h_ptr_lock);
            nvds_log(LOG_CAT, LOG_ERR, "unable to dlopen message adapter shared lib. Connection establishment failed. Error=%s\n", dlerror());
            return NULL;
        }

        //reuse handle if already opened
        if(so_handle_map.count(so_handle)) {
            adapter_lib = so_handle_map[so_handle];
            adapter_lib->ref_count++;
            nvds_log(LOG_CAT, LOG_DEBUG,"Reusing already opened adapter lib for [%s]", adapter_lib->protocol.c_str());
        }
        //else create a new adapter lib object
        else {
            char* (*my_ptr)(void);
            *(void **) (&my_ptr) = dlsym(so_handle, "nvds_msgapi_get_protocol_name");
            if ((error = dlerror()) != NULL) {
                handle_error(so_handle, &h_ptr_lock, adapter_lib, "Failed to load Adapter api nvds_msgapi_get_protocol_name(). Error:", error);
                return NULL;
            }

            string str(my_ptr());
            adapter_lib = new(nothrow) AdapterLib();
            if(!adapter_lib) {
                handle_error(so_handle, &h_ptr_lock, adapter_lib, "Error: malloc failed while creating AdapterLib:", str.c_str());
                return NULL;
            }
            if(!fetch_adapter_api(so_handle, adapter_lib)) {
                handle_error(so_handle, &h_ptr_lock, adapter_lib, "Error: while fetching adapter API symbols for:", str.c_str());
                return NULL;
            }
            adapter_lib->protocol = str;
            adapter_lib->so_handle = so_handle;
            adapter_lib->ref_count = 1;
            so_handle_map[so_handle] = adapter_lib;
        }

        NvMsgBrokerHandle *h_ptr=NULL;

        //Query connection signature from the adapter library
        char signature[1024];
        if((adapter_lib->signature_capability && \
            adapter_lib->nvds_msgapi_connection_signature_ptr(broker_conn_str, cfg, signature, sizeof(signature)) == NVDS_MSGAPI_OK) && \
            strcmp(signature, "")) {
                share_connection = true;
                nvds_log(LOG_CAT, LOG_DEBUG,"Connection signature queried = %s",signature);
        }

        //If connection sharing is allowed and a handle already present , return it to the caller
        if(share_connection && conn_string_handle_map.count(make_pair(string(signature), adapter_lib)) > 0) {
            h_ptr = conn_string_handle_map[make_pair(string(signature), adapter_lib)];

            //if existing connection is down and reconnection attempt is in progress, return error
            if(h_ptr->reconnecting){
                handle_error(so_handle, &h_ptr_lock, adapter_lib, "nvmsgbroker_connect: Connection to broker is down. Try later", "");
                return NULL;
            }
            nvds_log(LOG_CAT, LOG_INFO,"nvmsgbroker_connect: Reusing an already created connection with signature[%s] for [%s]",signature, adapter_lib->protocol.c_str());
        }
        //else if its a new connection request, create a new connection
        else{
            h_ptr = new (nothrow) NvMsgBrokerHandle();
            if(!h_ptr) {
                handle_error(so_handle, &h_ptr_lock, adapter_lib, "Error - malloc failed while creating nvmsgbroker handle", "");
                return NULL;
            }

            NvDsMsgApiHandle adapter_conn_handle = adapter_lib->nvds_msgapi_connect_ptr(broker_conn_str, (nvds_msgapi_connect_cb_t) adapter_connect_cb, cfg);
            if(!adapter_conn_handle) {
                handle_error(so_handle, &h_ptr_lock, adapter_lib, "Connection creation failed for adapter library :", adapter_lib->protocol.c_str());
                delete h_ptr;
                return NULL;
            }
            else {
                conn_Handle_list.insert(h_ptr);
                adapter_msgbroker_map[adapter_conn_handle] = h_ptr;
                if(share_connection)    conn_string_handle_map[make_pair(string(signature), adapter_lib)] = h_ptr;
                h_ptr->adapter_lib = adapter_lib;
                h_ptr->adapter_h_ptr = adapter_conn_handle;
                //create do work thread
                pthread_create(&h_ptr->do_work_thread.tid, NULL, &do_work_func, h_ptr);
                //initialize handle to default values
                h_ptr->broker_proto_so = string(broker_proto_lib);
                if(broker_conn_str)
                    h_ptr->broker_conn_str = string(broker_conn_str);
                if(cfg)
                    h_ptr->broker_cfg_path = string(cfg);
                h_ptr->broker_signature = share_connection? string(signature) : "";
                h_ptr->ref_count = 0;
                h_ptr->pending_send_cnt = 0;
                h_ptr->disconnect = false;

                nvds_log(LOG_CAT, LOG_INFO,"New Connection created successfully");
            }
        }
        //Increment ref count
        h_ptr->ref_count++;
        //Add connect callback to the list
        if(connect_cb)
            h_ptr->connect_cb_set.insert(connect_cb);

        pthread_mutex_unlock(&h_ptr_lock);
        return (NvMsgBrokerClientHandle) h_ptr;
    }
}

void *do_work_func (void *ptr) {
    NvMsgBrokerHandle *h_ptr = (NvMsgBrokerHandle *) ptr;
    unsigned cnt=0;
    while(h_ptr->disconnect == false) {
        pthread_mutex_lock(&h_ptr_lock);
        //attempt to reconnect if necessary
        if(h_ptr->reconnecting){
            if(h_ptr->adapter_h_ptr) {
                //disconnect adapter context
                if(h_ptr->adapter_lib->nvds_msgapi_disconnect_ptr(h_ptr->adapter_h_ptr) != NVDS_MSGAPI_OK) {
                    nvds_log(LOG_CAT, LOG_ERR, "Error: Call to disconnect adapter failed for [%s].\
                         Reconnection attempt failed", h_ptr->adapter_lib->protocol.c_str());
                    for(auto &user_cb : h_ptr->connect_cb_set) {
                        user_cb(h_ptr, NV_MSGBROKER_API_ERR);
                    }
                    h_ptr->reconnecting = false;
                }
                else {
                    //erase disconnected adapter context details
                    adapter_msgbroker_map.erase(h_ptr->adapter_h_ptr);
                    h_ptr->adapter_h_ptr = NULL;
                }
            }
            if(cnt > max_retry_limit){
                for(auto &user_cb : h_ptr->connect_cb_set) {
                    user_cb(h_ptr, NV_MSGBROKER_API_ERR);
                }
                nvds_log(LOG_CAT, LOG_INFO, "nvmsgbroker reconnect: Reached max reconnect retries");
                h_ptr->reconnecting = false;
                pthread_mutex_unlock(&h_ptr_lock);
            }
            else if(h_ptr->disconnect){
                nvds_log(LOG_CAT, LOG_INFO, "nvmsgbroker reconnect: Disconnection signal received.Quitting reconnection attempts");
                pthread_mutex_unlock(&h_ptr_lock);
                break;
            }
            else {
                int rv = reconnect(h_ptr);
                if(rv == FAIL) {
                    pthread_mutex_unlock(&h_ptr_lock);
                    sleep(retry_interval);
                    cnt += retry_interval;
                }
                else {
                    h_ptr->reconnecting = false;
                    pthread_mutex_unlock(&h_ptr_lock);
                    if(rv == SUCCESS) {
                        nvds_log(LOG_CAT, LOG_INFO,"nvmsgbroker reconnect: Success");
                        cnt=0;
                    }
                    else if(rv == ERROR){
                        nvds_log(LOG_CAT, LOG_ERR,"nvmsgbroker reconnect: Error");
                    }
                }
            }
        }
        //do work when there are pending messages
        else {
            pthread_mutex_unlock(&h_ptr_lock);
            // Don't wait for new messages in nvmsgbroker in case of MQTT--MQTT relies on continuous calls to mosquitto_loop()
            // as called by do_work() to keep connection alive. The call blocks until a messages is received or until
            // timeout is met as set in the mqtt config
            if (!g_strrstr(h_ptr->adapter_lib->protocol.c_str(), "MQTT")) {
                pthread_mutex_lock(&h_ptr->do_work_thread.lock);
                while(!h_ptr->disconnect && h_ptr->pending_send_cnt <=0 && h_ptr->reconnecting == false) {
                    pthread_cond_wait(&h_ptr->do_work_thread.cv, &h_ptr->do_work_thread.lock);
                }
                pthread_mutex_unlock(&h_ptr->do_work_thread.lock);
            }
            if(h_ptr->disconnect) {
                nvds_log(LOG_CAT, LOG_DEBUG, "do_work(): Disconnect signal received for [%s], exiting thread", h_ptr->adapter_lib->protocol.c_str());
                break;
            }
            if(h_ptr->adapter_h_ptr)
                h_ptr->adapter_lib->nvds_msgapi_do_work_ptr(h_ptr->adapter_h_ptr);
            //wait according to work-interval config
            usleep(work_interval);
        }
    }
    return NULL;
}

/*
 * Receive send status callbacks from Message adapter to the user
 */
void adapter_send_cb(void *user_ptr,  NvDsMsgApiErrorType completion_flag) {
    //extract user_ptr of type NvMsgBrokerSendCB_info and notify cb of send status

    NvMsgBrokerSendCB_info *myinfo = (NvMsgBrokerSendCB_info*) user_ptr;

    pthread_mutex_lock(&myinfo->h_ptr->do_work_thread.lock);
    myinfo->h_ptr->pending_send_cnt--;
    pthread_mutex_unlock(&myinfo->h_ptr->do_work_thread.lock);

    if(completion_flag == NVDS_MSGAPI_OK) {
        if(myinfo->send_cb)     myinfo->send_cb(myinfo->user_ptr, NV_MSGBROKER_API_OK);
    }
    else {
        if(myinfo->send_cb)     myinfo->send_cb(myinfo->user_ptr, NV_MSGBROKER_API_ERR);
        nvds_log(LOG_CAT, LOG_ERR, "send failed. Error returned from adapter lib send callback for [%s]", myinfo->h_ptr->adapter_lib->protocol.c_str());
    }

    delete myinfo;
}

NvMsgBrokerErrorType nv_msgbroker_send_async (NvMsgBrokerClientHandle ptr, NvMsgBrokerClientMsg message, nv_msgbroker_send_cb_t cb, void *user_ctx) {
    //Validate connection handle
    if(!ptr) {
        nvds_log(LOG_CAT, LOG_DEBUG, "send_async() called with null handle\n");
        return NV_MSGBROKER_API_ERR;
    }
    NvMsgBrokerHandle *h_ptr = (NvMsgBrokerHandle *) ptr;
    if(!conn_Handle_list.count(h_ptr)){
        nvds_log(LOG_CAT, LOG_ERR, "send_async() called with invalid handle\n");
        return NV_MSGBROKER_API_ERR;
    }

    if(h_ptr->reconnecting){
        nvds_log(LOG_CAT, LOG_ERR, "send_async() called while reconnection in progress for [%s]", h_ptr->adapter_lib->protocol.c_str());
        return NV_MSGBROKER_API_ERR;
    }

    if(h_ptr->adapter_h_ptr == NULL){
        nvds_log(LOG_CAT, LOG_ERR, "send_async() :Error adapter handle is NULL for [%s]", h_ptr->adapter_lib->protocol.c_str());
        return NV_MSGBROKER_API_ERR;
    }

    //Validate Message packet - payload pointer, payload len and topic pointer
    if(message.payload == NULL || message.payload_len <=0) {
        nvds_log(LOG_CAT, LOG_ERR, "send_async() invalid message packet sent for [%s]", h_ptr->adapter_lib->protocol.c_str());
        return NV_MSGBROKER_API_ERR;
    }

    NvMsgBrokerSendCB_info *myinfo = new(nothrow) NvMsgBrokerSendCB_info();
    if(myinfo == NULL) {
        nvds_log(LOG_CAT, LOG_ERR, "send_async() Malloc failed while creating sendcallback info object");
        return NV_MSGBROKER_API_ERR;
    }
    myinfo->send_cb = cb;
    myinfo->user_ptr = user_ctx;
    myinfo->h_ptr = h_ptr;

    pthread_mutex_lock(&h_ptr->do_work_thread.lock);
    //call adapter send_async
    if (h_ptr->adapter_lib->nvds_msgapi_send_async_ptr(h_ptr->adapter_h_ptr, message.topic, (const uint8_t*) message.payload, \
                                message.payload_len, adapter_send_cb, myinfo) != NVDS_MSGAPI_OK) {
        nvds_log(LOG_CAT, LOG_INFO, "send_async() : send failed. Error returned from adapter lib [%s]", h_ptr->adapter_lib->protocol.c_str());
        delete myinfo;
        pthread_mutex_unlock(&h_ptr->do_work_thread.lock);
        return NV_MSGBROKER_API_ERR;
    }
    //if success, signal the do_work thread to call adapter lib api do_work()
    h_ptr->pending_send_cnt++;
    pthread_cond_signal(&h_ptr->do_work_thread.cv);
    pthread_mutex_unlock(&h_ptr->do_work_thread.lock);

    //return send status msg
    return NV_MSGBROKER_API_OK;
}

/*
 * Receive subscibe callbacks for each message consumed from Message adapter
 */
void adapter_subscribe_cb(NvDsMsgApiErrorType flag, void *msg, int msglen, char *topic, void *user_ptr) {
    NvMsgBrokerHandle *h_ptr = (NvMsgBrokerHandle *) user_ptr;
    //Lookup topic map, For each topic --> user_cb in the map - call the cb with consume msg
    if(h_ptr->subscribe_topic_map.count(string(topic))) {
        for(auto const& it : h_ptr->subscribe_topic_map[string(topic)]) {
            if(flag == NVDS_MSGAPI_OK)
                it.first(NV_MSGBROKER_API_OK, msg, msglen, topic, it.second);
            else
                it.first(NV_MSGBROKER_API_ERR, msg, msglen, topic, it.second);
        }
    }
}

NvMsgBrokerErrorType nv_msgbroker_subscribe(NvMsgBrokerClientHandle ptr, char ** topics, int num_topics,  nv_msgbroker_subscribe_cb_t cb, void *user_ctx) {
    //Validate connection handle
    if(!ptr) {
        nvds_log(LOG_CAT, LOG_ERR, "Error: subscribe called with null handle");
        return NV_MSGBROKER_API_ERR;
    }
    //validate cb - cant be NULL
    if(!cb) {
        nvds_log(LOG_CAT, LOG_ERR, "subscribe: Error. callback pointer cant be NULL");
        return NV_MSGBROKER_API_ERR;
    }

    NvMsgBrokerHandle *h_ptr = (NvMsgBrokerHandle *) ptr;
    if(!conn_Handle_list.count(h_ptr)){
        nvds_log(LOG_CAT, LOG_ERR, "subscribe: subscribe called with invalid handle\n");
        return NV_MSGBROKER_API_ERR;
    }
    if(h_ptr->reconnecting) {
        nvds_log(LOG_CAT, LOG_ERR, "subscribe: Connection Error. Reconnection attempt in progress for [%s].try later", h_ptr->adapter_lib->protocol.c_str());
        return NV_MSGBROKER_API_ERR;
    }
    if(h_ptr->adapter_h_ptr == NULL) {
        nvds_log(LOG_CAT, LOG_ERR, "subscribe: Error. Adapter context is NULL for [%s]",h_ptr->adapter_lib->protocol.c_str());
        return NV_MSGBROKER_API_ERR;
    }
    //Validate topic pointer, num_topics
    if(topics==NULL || num_topics <=0) {
        nvds_log(LOG_CAT, LOG_ERR, "subscribe: Error. topics pointer is NULL or num topics <=0 for [%s]", h_ptr->adapter_lib->protocol.c_str());
        return NV_MSGBROKER_API_ERR;
    }

    pthread_mutex_lock(&h_ptr->subscribe_lock);

    if(!h_ptr->adapter_lib->subscribe_capability) {
        nvds_log(LOG_CAT, LOG_ERR, "subscribe: Error - Topic Subscription capability not supported in adapter lib[%s]", h_ptr->adapter_lib->protocol.c_str());
        pthread_mutex_unlock(&h_ptr->subscribe_lock);
        return NV_MSGBROKER_API_NOT_SUPPORTED;
    }

    size_t oldsize = h_ptr->subscribe_topic_map.size();

    //for each topic in topics - store the callback and user_ctx
    for (int i = 0 ; i < num_topics ; i++) {
            h_ptr->subscribe_topic_map[string(topics[i])].insert(make_pair(cb, user_ctx));
    }
    //if no new topics are requested to subscribe, then just return back
    if(h_ptr->subscribe_topic_map.size() == oldsize) {
        pthread_mutex_unlock(&h_ptr->subscribe_lock);
        return NV_MSGBROKER_API_OK;
    }

    size_t max_topic_len=0;
    for(auto const &it: h_ptr->subscribe_topic_map)
        if(it.first.length() > max_topic_len)
            max_topic_len = it.first.length();

    char *full_topic_list[h_ptr->subscribe_topic_map.size()];
    int i=0;
    for(auto const& it:h_ptr->subscribe_topic_map) {
        full_topic_list[i++] = (char *) it.first.c_str();
    }

    if(h_ptr->adapter_lib->nvds_msgapi_subscribe_ptr(h_ptr->adapter_h_ptr, full_topic_list, h_ptr->subscribe_topic_map.size(), adapter_subscribe_cb, h_ptr) != NVDS_MSGAPI_OK) {
        h_ptr->subscribe_topic_map.clear();
        nvds_log(LOG_CAT, LOG_ERR, "Error. subscribe call to adapter library failed for adapter[%s]", h_ptr->broker_proto_so.c_str());
        pthread_mutex_unlock(&h_ptr->subscribe_lock);
        return NV_MSGBROKER_API_ERR;
    }
    pthread_mutex_unlock(&h_ptr->subscribe_lock);
    return NV_MSGBROKER_API_OK;
}

NvMsgBrokerErrorType nv_msgbroker_disconnect(NvMsgBrokerClientHandle ptr){
    //Validate connection handle
    if(!ptr) {
        nvds_log(LOG_CAT, LOG_ERR, "Error: disconnect called with null handle\n");
        return NV_MSGBROKER_API_ERR;
    }

    pthread_mutex_lock(&h_ptr_lock);

    NvMsgBrokerHandle *h_ptr = (NvMsgBrokerHandle *) ptr;
    if(conn_Handle_list.count(h_ptr) == 0) {
        nvds_log(LOG_CAT, LOG_ERR, "Error: Invalid handle passed to nv_msgbroker_disconnect\n");
        pthread_mutex_unlock(&h_ptr_lock);
        return NV_MSGBROKER_API_ERR;
    }

    //decrement ref_count for number of references to nvmsgbroker connection handle
    h_ptr->ref_count--;

    //decrement ref_count for number of references to adapter shared library
    h_ptr->adapter_lib->ref_count--;

    if(h_ptr->ref_count == 0) {
        h_ptr->disconnect = true;
        //signal disconnect
        pthread_mutex_lock(&h_ptr->do_work_thread.lock);
        pthread_cond_signal(&h_ptr->do_work_thread.cv);
        pthread_mutex_unlock(&h_ptr->do_work_thread.lock);

        //join on do_work thread
        pthread_join(h_ptr->do_work_thread.tid, NULL);

        //Erase adapter handle --> msgbroker handle mapping
        adapter_msgbroker_map.erase(h_ptr->adapter_h_ptr);

        //Call adapter disconnect
        nvds_log(LOG_CAT, LOG_DEBUG, "Attempting Disconnect for [%s]", h_ptr->adapter_lib->protocol.c_str());
        if(h_ptr->adapter_h_ptr && h_ptr->adapter_lib->nvds_msgapi_disconnect_ptr(h_ptr->adapter_h_ptr) != NVDS_MSGAPI_OK) {
            nvds_log(LOG_CAT, LOG_ERR, "Error: Call to disconnect nvmsgbroker failed for [%s]", h_ptr->adapter_lib->protocol.c_str());
        }

        //Free pointer to adapter library info
        if(h_ptr->adapter_lib->ref_count==0) {
            so_handle_map.erase(h_ptr->adapter_lib->so_handle);
            //close shared lib
            dlclose(h_ptr->adapter_lib->so_handle);
            delete h_ptr->adapter_lib;
        }

        //Erase handle from the containers
        if(h_ptr->broker_signature != "")     conn_string_handle_map.erase(make_pair(h_ptr->broker_signature, h_ptr->adapter_lib));
        conn_Handle_list.erase(h_ptr);

        //free handle
        delete h_ptr;
        h_ptr = NULL;
        nvds_log(LOG_CAT, LOG_INFO, "Disconnection Success");
    }
    pthread_mutex_unlock(&h_ptr_lock);
    return NV_MSGBROKER_API_OK;
}

char *nv_msgbroker_version(){
     return (char *)NV_MSGBROKER_VERSION;
}
