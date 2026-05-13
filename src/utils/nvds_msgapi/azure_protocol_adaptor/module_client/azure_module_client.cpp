/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <glib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "iothub.h"
#include "iothub_module_client.h"
#include "iothub_client_options.h"
#include "iothub_message.h"
#include "azure_c_shared_utility/threadapi.h"
#include "azure_c_shared_utility/crt_abstractions.h"
#include "azure_c_shared_utility/shared_util_options.h"
#include "iothubtransportmqtt.h"
#include "nvds_msgapi.h"

/* This sample uses the convenience APIs of iothub_client */

#define NVDS_MSGAPI_VERSION "1.0"
#define NVDS_MSGAPI_PROTOCOL "AZURE_MODULE_CLIENT"
#define MAX_WAIT_TIME_FOR_SEND 200

/* custom message properties */
std::unordered_map<std::string , std::string> msg_properties;

/* Azure module client handle type */
typedef struct {
    IOTHUB_MODULE_CLIENT_HANDLE module_handle;
} nvds_azure_client_handle_t;


/* structure to store params needed for send_sync
 * send_cb : user callback function for send
 * user_cb_ptr : user pointer for callback context
 */
struct send_async_info_t {
    nvds_msgapi_send_cb_t send_cb;
    void *user_cb_ptr;
    IOTHUB_MESSAGE_HANDLE mHandle;
};


void free_gobjs(GKeyFile *gcfg_file, GError *error, gchar **keys);
int parse_config(char *config);

/* nvds_msgapi function for retrieving nvds_msgapi version
*/
char *nvds_msgapi_getversion() {
  return (char *) NVDS_MSGAPI_VERSION;
}

// When a message is sent with synchronous send , this callback will get invoked
static void send_confirm_callback(IOTHUB_CLIENT_CONFIRMATION_RESULT result, void *userContextCallback) {
    if(!result) {
        *(int *) (userContextCallback) = 1;
    }
    else {
        printf( "send_confirm_callback : send failed with error code %d\n", result);
        *(int *) (userContextCallback) = -1;
    }
}

// When a message is sent with send async, this callback will get invoked
static void send_async_callback(IOTHUB_CLIENT_CONFIRMATION_RESULT result, void *userContextCallback) {
    struct send_async_info_t *ptr = (struct send_async_info_t *) userContextCallback;
    if(!result) {
        ((nvds_msgapi_send_cb_t) ptr->send_cb) (ptr->user_cb_ptr , NVDS_MSGAPI_OK);
    }
    else {
        ((nvds_msgapi_send_cb_t) ptr->send_cb) (ptr->user_cb_ptr , NVDS_MSGAPI_ERR);
        printf( "send_async_callback : send async failed with error code %d\n", result);
    }
    IoTHubMessage_Destroy(ptr->mHandle);
    free(ptr);
}

// Release memory allocated for gobjects
void free_gobjs(GKeyFile *gcfg_file, GError *error, gchar **keys) {
    if(keys != NULL)
        g_strfreev(keys);
    if(error != NULL)
        g_error_free (error);
    if(gcfg_file != NULL)
        g_key_file_free(gcfg_file);
}

// Parse the config file
int parse_config(char *config) {
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    gchar *val = NULL;
    GKeyFile *gcfg_file = g_key_file_new ();
    if (!g_key_file_load_from_file (gcfg_file, config, G_KEY_FILE_NONE, &error)) {
        printf("Azure edge adaptor: Failed to parse configfile %s\n", config);
        g_free(val);
        free_gobjs(gcfg_file, error, keys);
        return 0;
    }
    unsigned int max_len = 512;
    char grpname[15] = "message-broker";
    char cust_msg_properties[max_len]="";
    keys = g_key_file_get_keys(gcfg_file, grpname, NULL, &error);
    for (key = keys; *key; key++) {
        if (!g_strcmp0(*key, "custom_msg_properties")) {
            val = g_key_file_get_string (gcfg_file, grpname, "custom_msg_properties",&error);
            g_strlcpy(cust_msg_properties, val, max_len);
            g_free(val);
            val = NULL;
        }
    }

    free_gobjs(gcfg_file, error, keys);

    //do some message properties validation
    //string length needs to be at least 4. key=value;
    if((strlen(cust_msg_properties) > 0) && (strlen(cust_msg_properties) < 4))
        printf("Azure edge: custom message property has invalid format. Ignoring message properties\n");
    else if(strnlen(cust_msg_properties, max_len) == max_len)
       printf("Azure custom message property string len exceeds max len of %dbytes. Ignoring message properties\n", max_len);
    else {
        std::string key_val_token;
        std::istringstream iss(cust_msg_properties);
        while(std::getline(iss, key_val_token, ';')) {
            size_t pos = key_val_token.find('=');
            //validate if each key value pair is delimited by "="
            if(pos == std::string::npos) {
                printf("Azure edge: custom message property. key value pairs not separated by =");
                return -1;
            }
            std::string key = key_val_token.substr(0, pos);
            std::string val = key_val_token.substr(pos + 1);
            msg_properties[key] = val;
        }
    }
    return 1;
}

/* nvds_msgapi function to connect to azure iothub
*/
NvDsMsgApiHandle nvds_msgapi_connect(char *str, nvds_msgapi_connect_cb_t connect_cb, char *config_path) {
    nvds_azure_client_handle_t *ah = (nvds_azure_client_handle_t *) malloc(sizeof(nvds_azure_client_handle_t));
    if(ah == NULL) {
        printf( "nvds_msgapi_connect: Error - malloc failed for azure handle\n");
        return NULL;
    }

    if((config_path != NULL) && strlen(config_path)) {
        if(!parse_config(config_path)) {
            printf("nvds_msgapi_connect: Unable to parse config file\n");
            free(ah);
            return NULL;
        }
    }
    // Used to initialize IoTHub SDK subsystem
    if (IoTHub_Init() != 0) {
        printf( "nvds_msgapi_connect: Failed to initialize the platform.\r\n");
        free(ah);
        return NULL;
    }

    // Create edge module with MQTT protocol as transport provider
    ah->module_handle = IoTHubModuleClient_CreateFromEnvironment(MQTT_Protocol);
    if (ah->module_handle == NULL) {
        printf( "ERROR: iotHubModuleClientHandle is NULL! connect failed\n");
        connect_cb((NvDsMsgApiHandle) ah , NVDS_MSGAPI_EVT_DISCONNECT);
        IoTHub_Deinit();
        free(ah);
        return NULL;
    }
    else {
        connect_cb((NvDsMsgApiHandle) ah , (NvDsMsgApiEventType) 2);
        printf( "nvds_msgapi_connect : connect success\n");
        return (NvDsMsgApiHandle) ah;
    }
}

/* nvds_msgapi function for synchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *msg, size_t nbuf) {
    NvDsMsgApiErrorType rv = NVDS_MSGAPI_ERR;
    // done : {0 = send in progress, 1 = send success, -1 = send failed}
    int done=0;
    // Construct the iothub message from byte array
    IOTHUB_MESSAGE_HANDLE message_handle = IoTHubMessage_CreateFromByteArray((const unsigned char*)msg, nbuf);
    if(message_handle == NULL) {
        printf("nvds_msgapi_send: Error creating message handle. Send to azure failed");
        return rv;
    }
    // Apply custom message properties
    for(auto it = msg_properties.begin(); it != msg_properties.end(); it++) {
        IOTHUB_MESSAGE_RESULT res = IoTHubMessage_SetProperty(message_handle, it->first.c_str(), it->second.c_str());
        if(res != IOTHUB_MESSAGE_OK) {
            printf("nvds_msgapi_send: Applying msg property failed. key=%s, val=%s", it->first.c_str(), it->second.c_str());
        }
    }

    // Send message--done will be set when send_confirm_callback is invoked
    IOTHUB_CLIENT_RESULT result = IoTHubModuleClient_SendEventToOutputAsync(((nvds_azure_client_handle_t *)h_ptr)->module_handle,  message_handle, (const char *) topic, send_confirm_callback, &done);
    if(result != IOTHUB_CLIENT_OK) {
        printf("nvds_msgapi_send: send failed, error code = %d\n", result);
        IoTHubMessage_Destroy(message_handle);
        return rv;
    }

    int cnt=0;
    // Wait for send to be confirmed, i.e. done is set by the callback
    while(cnt < MAX_WAIT_TIME_FOR_SEND) {
        if(done == 1) {
            rv = NVDS_MSGAPI_OK;
            printf("Message sent : %.*s\n", (int) nbuf, (char *) msg);
            break;
        }
        else if(done == -1) {
            break;
        }
        else	ThreadAPI_Sleep(10);
        cnt++;
    }
    // If send confirmation not received within MAX_WAIT_TIME_FOR_SEND * 10ms, report send failure
    if(cnt == MAX_WAIT_TIME_FOR_SEND) {
        printf("nvds_msgapi_send: Exceeds max wait time to receive send confirmation. Discarding message\n");
    }
    IoTHubMessage_Destroy(message_handle);
    return rv;
}

NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char  *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr) {
    NvDsMsgApiErrorType rv = NVDS_MSGAPI_ERR;
    // Create message info
    struct send_async_info_t *myinfo = (struct send_async_info_t *) malloc (sizeof(struct send_async_info_t));
    if(myinfo == NULL) {
        printf( "nvds_msgapi_send_async: Malloc fail\n");
        return rv;
    }
    // Construct the iothub message from byte array
    IOTHUB_MESSAGE_HANDLE message_handle = IoTHubMessage_CreateFromByteArray((const unsigned char*)payload, nbuf);
    if(message_handle == NULL) {
        printf("nvds_msgapi_send_async: Error creating message handle. Send to azure failed");
        free(myinfo);
        return rv;
    }
    // Apply custom message properties
    for(auto it = msg_properties.begin(); it != msg_properties.end(); it++) {
        IOTHUB_MESSAGE_RESULT res = IoTHubMessage_SetProperty(message_handle, it->first.c_str(), it->second.c_str());
        if(res != IOTHUB_MESSAGE_OK) {
            printf("nvds_msgapi_send_async: Applying msg property failed. key=%s, val=%s", it->first.c_str(), it->second.c_str());
        }
    }

    // Fill message info
    myinfo->send_cb = send_callback;
    myinfo->user_cb_ptr = user_ptr;
    myinfo->mHandle = message_handle;

    // Send message without waiting
    IOTHUB_CLIENT_RESULT result = IoTHubModuleClient_SendEventToOutputAsync(((nvds_azure_client_handle_t *)h_ptr)->module_handle, message_handle, (const char *) topic, send_async_callback, myinfo);
    if(result == IOTHUB_CLIENT_OK) {
        rv = NVDS_MSGAPI_OK;
        printf("Message sent : %.*s\n", (int) nbuf, (char *) payload);
    }
    else {
        printf("nvds_msgapi_send_async: send failed, error code = %d\n", result);
        IoTHubMessage_Destroy(message_handle);
    }
    return rv;
}

/* nvds_msgapi function for performing work in nvmsgbroker plugin. No op for azure device client
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) {
}

/* nvds_msgapi function for disconnect
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr) {
    // Clean up the iothub sdk handle
    IoTHubModuleClient_Destroy(((nvds_azure_client_handle_t *) h_ptr)->module_handle);
    // Free all the sdk subsystem
    IoTHub_Deinit();
    free((nvds_azure_client_handle_t *) h_ptr);
    printf( "Disconnecting Azure..\n");
    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for retrieving protocol name
*/
char *nvds_msgapi_get_protocol_name()
{
  return (char *)NVDS_MSGAPI_PROTOCOL;
}

/* nvds_msgapi function for retrieving connection signature, used for connection sharing
*/
NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str, char *cfg, char *output_str, int max_len) {
    strcpy(output_str, "");
    return NVDS_MSGAPI_OK;
}
