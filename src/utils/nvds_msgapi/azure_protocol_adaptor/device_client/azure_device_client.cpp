/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
/* This sample uses the convenience APIs of iothub_client */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <glib.h>
#include <openssl/sha.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <atomic>
#include "iothub.h"
#include "iothub_device_client.h"
#include "iothub_client_options.h"
#include "iothub_message.h"
#include "azure_c_shared_utility/threadapi.h"
#include "azure_c_shared_utility/crt_abstractions.h"
#include "azure_c_shared_utility/shared_util_options.h"
#include "iothubtransportmqtt.h"
#include "nvds_msgapi.h"
#include "nvds_logger.h"
#include "nvds_utils.h"

using namespace std;
#define NVDS_MSGAPI_VERSION "2.0"
#define NVDS_MSGAPI_PROTOCOL "AZURE_DEVICE_CLIENT"
#define LOG_CAT "DSLOG:NVDS_AZURE_PROTO"
#define CONNECT_COUNTER 300
#define MAX_WAIT_TIME_FOR_SEND 500

/* custom message properties */
std::unordered_map<std::string , std::string> msg_properties;

/* Global variables for message tracking */
static std::atomic<int> pending_messages(0);
static bool is_disconnecting = false;

/* NVDS Azure connection handle
 * device_handle : device client handle
 * connect_cb_func  : user callback function for connect
 * connection_state : 0=default, 1 = success, -1 = fail
 * broker_cfg_path : path to broker config file
 * */
typedef struct {
    IOTHUB_DEVICE_CLIENT_HANDLE device_handle;
    nvds_msgapi_connect_cb_t connect_cb_func;
    int connection_state;
    char *broker_cfg_path;
} nvds_azure_client_handle_t;

/* structure to store params needed for send_sync
 * send_cb : user callback function for send
 * user_cb_ptr : user pointer for callback context
 * message_handle : of type IOTHUB_MESSAGE_HANDLE
 */
struct send_async_info_t {
    nvds_msgapi_send_cb_t send_cb;
    void *user_cb_ptr;
};

static bool validate_connection_str(char *s);
int fetch_connect_str(char*, char*, char*, unsigned int);
void CLEANUP_CLIENT_HANDLE(nvds_azure_client_handle_t *ah);

/* Function performs standard iothub device cleanup
*/
void CLEANUP_CLIENT_HANDLE(nvds_azure_client_handle_t *ah){
    // Clean up the iothub sdk handle
    IoTHubDeviceClient_Destroy(ah->device_handle);
    // Free all the sdk subsystem
    IoTHub_Deinit();
    free(ah);
}

/* nvds_msgapi function for retrieving nvds_msgapi version
*/
char *nvds_msgapi_getversion()
{
  return (char *) NVDS_MSGAPI_VERSION;
}

// When a message is sent synchronously this callback will get invoked
static void send_confirm_callback(IOTHUB_CLIENT_CONFIRMATION_RESULT result, void *userContextCallback)
{
    if(!result) {
        *(int *) (userContextCallback) = 1;
    }
    else {
        nvds_log(LOG_CAT, LOG_ERR , "Azure: send callback received error code %d\n", result);
        *(int *) (userContextCallback) = -1;
    }
}

// When a message is sent asynchronously this callback will get invoked
static void send_async_callback(IOTHUB_CLIENT_CONFIRMATION_RESULT result, void *userContextCallback)
{
    // Call user send callback with send status
    struct send_async_info_t *ptr = (struct send_async_info_t *) userContextCallback;

    // Decrement pending messages counter
    pending_messages--;

    // If we're disconnecting or the result indicates the client was destroyed,
    // treat it as a successful send to prevent errors
    if (is_disconnecting || result == IOTHUB_CLIENT_CONFIRMATION_BECAUSE_DESTROY) {
        if (ptr->send_cb) {
            ((nvds_msgapi_send_cb_t) ptr->send_cb) (ptr->user_cb_ptr, NVDS_MSGAPI_OK);
        }
        free(ptr);
        return;
    }

    if(!result) {
        ((nvds_msgapi_send_cb_t) ptr->send_cb) (ptr->user_cb_ptr , NVDS_MSGAPI_OK);
    }
    else {
        ((nvds_msgapi_send_cb_t) ptr->send_cb) (ptr->user_cb_ptr , NVDS_MSGAPI_ERR);
        nvds_log(LOG_CAT, LOG_ERR , "Azure: send async callback received error code %d\n", result);
        // Add more detailed error logging
        switch(result) {
            case IOTHUB_CLIENT_CONFIRMATION_ERROR:
                nvds_log(LOG_CAT, LOG_ERR, "Azure: General error occurred while sending message");
                break;
            case IOTHUB_CLIENT_CONFIRMATION_BECAUSE_DESTROY:
                nvds_log(LOG_CAT, LOG_ERR, "Azure: Message delivery failed because client was destroyed");
                break;
            case IOTHUB_CLIENT_CONFIRMATION_MESSAGE_TIMEOUT:
                nvds_log(LOG_CAT, LOG_ERR, "Azure: Message timeout occurred");
                break;
            default:
                nvds_log(LOG_CAT, LOG_ERR, "Azure: Unknown error occurred");
                break;
        }
    }

    free(ptr);
}

// Callback for connection status
static void connection_status_callback(IOTHUB_CLIENT_CONNECTION_STATUS result, IOTHUB_CLIENT_CONNECTION_STATUS_REASON reason, void* user_context)
{
    nvds_azure_client_handle_t *ah = (nvds_azure_client_handle_t *) user_context;
    // Set connection state and call user connect callback according to status
    // This DOES NOT take into consideration network outages.
    if (result == IOTHUB_CLIENT_CONNECTION_AUTHENTICATED)
    {
        (ah->connect_cb_func) ((NvDsMsgApiHandle) ah , NVDS_MSGAPI_EVT_SUCCESS);
        ah->connection_state = 1;
        nvds_log(LOG_CAT, LOG_INFO , "The device client is connected to azure iothub");
    }
    else
    {
        (ah->connect_cb_func) ((NvDsMsgApiHandle) ah , NVDS_MSGAPI_EVT_DISCONNECT);
        ah->connection_state = -1;
        nvds_log(LOG_CAT,LOG_ERR,"Unable to connect device client to azure iothub. error code : %d", result);
    }
}

/* Function to validate format of connection string
*/
static bool validate_connection_str(char *s) {
    if((strstr(s, "HostName=") != NULL) &&
       (strstr(s, "DeviceId=") != NULL) &&
       (strstr(s, "SharedAccessKey=") != NULL))
            return true;
    else
            return false;
}

/*
  Function to fetch connection string provided for azure authentication.
  option 1: Provide full azure conenction string as connection params in nvds_msgapi_connect() with format: HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>
  option 2: The full device connection string is provided in Azure specific config file. ex: HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>
  inputs  : char *str - Azure connection string passed as param to nvds_msgapi_connect()
            char *config - config file passed to nvds_msgapi_conenct() to parse Azure adapter related information
            char *res - resulting connection string to be used for authentication
            unsigned int size - max size of connection string
  return  : success = 0 , fail = -1. Resulting connection string will be stored in char *res
*/
int fetch_connect_str(char *str, char *config, char *res, unsigned int size) {
    unsigned int max_len = 512;
    char cust_msg_properties[max_len]="", connection_string[size]="";
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    gchar *val = NULL;

    // parse config if not null and file length  is not 0
    if((config != NULL) && (strlen(config))) {
        GKeyFile *gcfg_file = g_key_file_new ();
        if (!g_key_file_load_from_file (gcfg_file, config, G_KEY_FILE_NONE, &error)) {
            nvds_log(LOG_CAT, LOG_ERR , "Azure adaptor: Failed to parse configfile %s\n", config);
            free_gobjs(gcfg_file, error, keys, val);
            return -1;
        }

        char grpname[15] = "message-broker";
        keys = g_key_file_get_keys(gcfg_file, grpname, NULL, &error);
        for (key = keys; *key; key++) {
            if (!g_strcmp0(*key, "connection_str")) {
                val = g_key_file_get_string (gcfg_file, grpname, "connection_str",&error);
                unsigned int conn_string_size = g_strlcpy(connection_string, val, size);
                g_free(val);
                if(conn_string_size >= size) {
                    nvds_log(LOG_CAT, LOG_ERR , "Azure connection string size exceeds max len of %dbytes\n", size);
                    free_gobjs(gcfg_file, error, keys, NULL);
                    return -1;
                }
            }
            if (!g_strcmp0(*key, "custom_msg_properties")) {
                val = g_key_file_get_string (gcfg_file, grpname, "custom_msg_properties",&error);
                g_strlcpy(cust_msg_properties, val, max_len);
                g_free(val);
            }
        }
        free_gobjs(gcfg_file, error, keys, NULL);
    }
    //do some message properties validation
    //string length needs to be at least 4. key=value;
    if((strlen(cust_msg_properties) > 0) && (strlen(cust_msg_properties) < 4))
        nvds_log(LOG_CAT, LOG_ERR , "Azure custom message property has invalid format");
    else if (strnlen(cust_msg_properties, max_len) == max_len)
        nvds_log(LOG_CAT, LOG_ERR , "Azure custom message property string len exceeds max len of %dbytes. Ignoring message properties\n", max_len);
    else {
        std::string key_val_token;
        std::istringstream iss(cust_msg_properties);
        while(std::getline(iss, key_val_token, ';')) {
            size_t pos = key_val_token.find('=');
            //validate if each key value pair is delimited by "="
            if(pos == std::string::npos) {
                nvds_log(LOG_CAT, LOG_ERR , "Azure custom message property has invalid format. key value pairs not separated by =");
                nvds_log(LOG_CAT, LOG_ERR, "Ignoring Azure custom message property");
                break;
            }
            std::string key = key_val_token.substr(0, pos);
            std::string value = key_val_token.substr(pos + 1);
            msg_properties[key] = std::move(value);
        }
    }
    // Check & validate connection details provided
    if(str != NULL && (strlen(str) > 0)) {
        // Option 1 : Check if connection params to nvds_msgapi_connect() has the full azure connection string
        if(validate_connection_str(str)) {
            strcpy(res, str);
            return 0;
        }
        else {
            nvds_log(LOG_CAT, LOG_ERR , "Azure connection string provided as params to nvds_msgapi_connect is of invalid format");
            return -1;
        }
    }
    else if(config != NULL && (strlen(connection_string) > 0)) {
        // Option 2: else check if the full connection string provided within azure specific cfg file
        if(validate_connection_str(connection_string)) {
            strcpy(res, connection_string);
	    return 0;
        }
        else {
            nvds_log(LOG_CAT, LOG_ERR , "Azure connection string provided in cfg file is of invalid format");
            return -1;
        }
    }
    else {
        nvds_log(LOG_CAT, LOG_ERR ,"Error. Azure connection string not provided\n");
        return -1;
    }
}

/* nvds_msgapi function to connect to azure iothub
*/
NvDsMsgApiHandle nvds_msgapi_connect(char *str, nvds_msgapi_connect_cb_t connect_cb, char *config_path) {
    NvDsMsgApiHandle myhandle = NULL;
    char connection_str[1024];
    nvds_log_open();
    // Validate connection string
    if(fetch_connect_str(str , config_path, connection_str, sizeof(connection_str)) == -1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure in fetching azure connection string");
        nvds_log_close();
        return myhandle;
    }
    // Create azure client handle
    nvds_azure_client_handle_t *ah = (nvds_azure_client_handle_t *) malloc(sizeof(nvds_azure_client_handle_t));
    if(ah == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Error - malloc failed for azure handle");
        nvds_log_close();
        return myhandle;
    }

    // Store config path if provided
    ah->broker_cfg_path = config_path;

    // Using mqtt protocol to communicate with azure iothub
    IOTHUB_CLIENT_TRANSPORT_PROVIDER protocol = MQTT_Protocol;

    // Used to initialize IoTHub SDK subsystem
    (void)IoTHub_Init();
    ah->connection_state = 0;
    ah->connect_cb_func = connect_cb;
    // Create IoTHub device client from connection string
    ah->device_handle = IoTHubDeviceClient_CreateFromConnectionString(connection_str, protocol);
    if (ah->device_handle == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure creating Iothub device. Hint: Check your connection string");
        IoTHub_Deinit();
        free(ah);
        nvds_log_close();
        return myhandle;
    }
    else {
        // Set azure connection callback
        IoTHubDeviceClient_SetConnectionStatusCallback(ah->device_handle, connection_status_callback, ah);
        int cnt=0;
        // Wait for connection results
        while(cnt < CONNECT_COUNTER) {
            // Check connection state, set in azure connection callback
            if(ah->connection_state == -1) {
                nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure to connect Azure Iot hub.");
                CLEANUP_CLIENT_HANDLE(ah);
                nvds_log_close();
                break;
            }
            // Successful connection, break loop
            else if(ah->connection_state == 1) {
                nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_connect: Connect to Azure Iot hub success");
                myhandle = (NvDsMsgApiHandle) ah;
                break;
            }
            else        ThreadAPI_Sleep(10);
            cnt++;
        }
        // if connection results wont appear within CONNECT_COUNTER * 10ms, report connection failure
        if(cnt == CONNECT_COUNTER) {
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: connection timeout. Failure to connect azure");
            CLEANUP_CLIENT_HANDLE(ah);
            nvds_log_close();
        }
        return myhandle;
    }
}

/* nvds_msgapi function for synchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *msg, size_t nbuf) {
    NvDsMsgApiErrorType rv = NVDS_MSGAPI_ERR;

    // done : {0 = send in progress, 1 = send success, -1 = send failed}
    int done = 0;

    // Construct the iothub message from byte array
    IOTHUB_MESSAGE_HANDLE message_handle = IoTHubMessage_CreateFromByteArray((const unsigned char*)msg, nbuf);
    if(message_handle == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send: Error creating message handle. Send to azure failed");
        return rv;
    }
    // Apply custom message properties
    for(auto it = msg_properties.begin(); it != msg_properties.end(); it++) {
        IOTHUB_MESSAGE_RESULT res = IoTHubMessage_SetProperty(message_handle, it->first.c_str(), it->second.c_str());
        if(res != IOTHUB_MESSAGE_OK) {
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send: Applying msg property failed. key=%s, val=%s", it->first, it->second);
        }
    }

    // Send message--done will be set when send_confirm_callback is invoked
    IOTHUB_CLIENT_RESULT result = IoTHubDeviceClient_SendEventAsync(((nvds_azure_client_handle_t *)h_ptr)->device_handle, message_handle, send_confirm_callback, &done);
    if(result != IOTHUB_CLIENT_OK) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send: Send to azure failed");
        IoTHubMessage_Destroy(message_handle);
        return rv;
    }
    int cnt=0;
    // Wait for send to be confirmed, i.e. done is set by the callback
    while(cnt < MAX_WAIT_TIME_FOR_SEND) {
        if(done == 1) {
            rv = NVDS_MSGAPI_OK;
            nvds_log(LOG_CAT, LOG_DEBUG , "Message sent to Azure IoT : %.*s\n", nbuf, (char *) msg);
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
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send: Exceeds max wait time to receive send confirmation. Discarding message");
    }
    IoTHubMessage_Destroy(message_handle);
    return rv;
}

/* nvds_msgapi function for asynchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr) {
    NvDsMsgApiErrorType rv = NVDS_MSGAPI_ERR;
    nvds_azure_client_handle_t *ah = (nvds_azure_client_handle_t *)h_ptr;

    if (is_disconnecting) {
        nvds_log(LOG_CAT, LOG_WARNING, "Cannot send message - client is disconnecting");
        return NVDS_MSGAPI_ERR;
    }

    // Create message info
    struct send_async_info_t *myinfo = (struct send_async_info_t *) malloc (sizeof(struct send_async_info_t));
    if(myinfo == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "Malloc fail");
        return rv;
    }

    // Set user send callback and user pointer for message info
    myinfo->send_cb = send_callback;
    myinfo->user_cb_ptr = user_ptr;

    // Construct the iothub message from byte array
    IOTHUB_MESSAGE_HANDLE message_handle = IoTHubMessage_CreateFromByteArray((const unsigned char*)payload, nbuf);
    if(message_handle == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send_async: Error creating message handle. Send to azure failed");
        free(myinfo);
        return rv;
    }

    // Store client handle in message properties for callback
    char handle_str[32];
    snprintf(handle_str, sizeof(handle_str), "%p", ah);
    IoTHubMessage_SetProperty(message_handle, "client_handle", handle_str);

    // Apply custom message properties
    for(auto it=msg_properties.begin(); it != msg_properties.end(); it++) {
        IOTHUB_MESSAGE_RESULT res = IoTHubMessage_SetProperty(message_handle, it->first.c_str(), it->second.c_str());
        if(res != IOTHUB_MESSAGE_OK) {
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send_async: Applying msg property failed. key=%s, val=%s", it->first, it->second);
        }
    }

    // Increment pending messages counter
    pending_messages++;

    // Send message without waiting
    IOTHUB_CLIENT_RESULT result = IoTHubDeviceClient_SendEventAsync(ah->device_handle, message_handle, send_async_callback, myinfo);
    if(result == IOTHUB_CLIENT_OK) {
        rv = NVDS_MSGAPI_OK;
        nvds_log(LOG_CAT, LOG_DEBUG , "Message sent to Azure IoT : %.*s\n", nbuf, (char *) payload);
    }
    else {
        pending_messages--; // Decrement on failure
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send_async: send to azure failed");
    }
    IoTHubMessage_Destroy(message_handle);
    return rv;
}

/* nvds_msgapi function for performing work in nvmsgbroker plugin. No op for azure device client
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) {
}

/* nvds_msgapi function for disconnect
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr) {
    nvds_azure_client_handle_t *ah = (nvds_azure_client_handle_t *)h_ptr;

    if (!ah) {
        nvds_log(LOG_CAT, LOG_ERR, "Invalid handle in disconnect");
        return NVDS_MSGAPI_ERR;
    }

    nvds_log(LOG_CAT, LOG_INFO, "Starting graceful disconnect...");

    // Set disconnecting flag to prevent new messages
    is_disconnecting = true;

    // First check if we're still connected
    IOTHUB_CLIENT_STATUS status;
    IOTHUB_CLIENT_RESULT result = IoTHubDeviceClient_GetSendStatus(ah->device_handle, &status);

    if (result != IOTHUB_CLIENT_OK) {
        nvds_log(LOG_CAT, LOG_WARNING, "Failed to get send status, proceeding with cleanup");
        CLEANUP_CLIENT_HANDLE(ah);
        nvds_log_close();
        return NVDS_MSGAPI_ERR;
    }

    // Get cleanup timeout from config
    int cleanup_timeout = 10; // Default 10 seconds
    GError *error = NULL;
    if (ah->broker_cfg_path != NULL) {
        GKeyFile *gcfg_file = g_key_file_new();
        if (g_key_file_load_from_file(gcfg_file, ah->broker_cfg_path, G_KEY_FILE_NONE, &error)) {
            cleanup_timeout = g_key_file_get_integer(gcfg_file, "message-broker", "cleanup-timeout", &error);
            if (error) {
                nvds_log(LOG_CAT, LOG_WARNING, "Failed to read cleanup-timeout from config, using default of 10 seconds");
                g_error_free(error);
                error = NULL; // Reset error pointer after freeing
            }
        } else {
            // g_key_file_load_from_file failed, error is set
            if (error) {
                nvds_log(LOG_CAT, LOG_WARNING, "Failed to load azure config file inside nvds_msgapi_disconnect");
                g_error_free(error);
                error = NULL; // Reset error pointer after freeing
            }
        }
        if(gcfg_file != NULL) {
            g_key_file_free(gcfg_file);
        }
    }

    // Calculate number of retries based on timeout (100ms sleep per retry)
    const int CLEANUP_SLEEP_MS = 100;
    const int MAX_CLEANUP_RETRIES = (cleanup_timeout * 1000) / CLEANUP_SLEEP_MS;
    const int BATCH_SIZE = 100;   // Process messages in batches

    int retry_count = 0;
    while (retry_count < MAX_CLEANUP_RETRIES && pending_messages > 0) {
        int current_pending = pending_messages.load();
        int remaining_timeout = cleanup_timeout - (retry_count * CLEANUP_SLEEP_MS / 1000);
        nvds_log(LOG_CAT, LOG_WARNING, "Azure disconnect: %d pending messages, cleanup timeout remaining: %d seconds (attempt %d/%d)",
                 current_pending, remaining_timeout, retry_count + 1, MAX_CLEANUP_RETRIES);

        // If we have a large number of pending messages, log a warning
        if (current_pending > BATCH_SIZE) {
            nvds_log(LOG_CAT, LOG_WARNING, "Azure disconnect:Large number of pending messages (%d), holding off cleanup until all messages are sent", current_pending);
        }

        // Give more time for messages to complete if we have fewer pending
        int sleep_time = (current_pending > BATCH_SIZE) ? CLEANUP_SLEEP_MS : CLEANUP_SLEEP_MS * 2;
        ThreadAPI_Sleep(sleep_time);
        retry_count++;
    }

    if (pending_messages > 0) {
        nvds_log(LOG_CAT, LOG_WARNING, "Cleanup Timeout(%d seconds) over. %d messages still pending shall be discarded, forcing cleanup",
                 cleanup_timeout, pending_messages.load());
    } else {
        nvds_log(LOG_CAT, LOG_INFO, "All messages completed successfully");
    }

    // Give a small grace period for any final callbacks
    ThreadAPI_Sleep(100);

    // Final cleanup
    if (ah->device_handle) {
        nvds_log(LOG_CAT, LOG_INFO, "Cleaning up Azure IoT device handle");
        CLEANUP_CLIENT_HANDLE(ah);
    }

    // Reset the disconnecting flag
    is_disconnecting = false;

    nvds_log(LOG_CAT, LOG_INFO, "Disconnect completed");
    nvds_log_close();
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
    strcpy(output_str,"");
     //check if share-connection config option is turned ON
    char reuse_connection[16]="";
    if(fetch_config_value(cfg, "share-connection", reuse_connection, sizeof(reuse_connection), LOG_CAT) != NVDS_MSGAPI_OK) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing Azure share-connection config option");
        return NVDS_MSGAPI_ERR;
    }
    if(strcmp(reuse_connection, "1")) {
        nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_connection_signature: Azure connection sharing disabled. Hence connection signature cant be returned");
        return NVDS_MSGAPI_OK;
    }
    if(broker_str == NULL && cfg == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Both broker_str and path to cfg cant be NULL");
        return NVDS_MSGAPI_ERR;
    }
    if(max_len < 2 * SHA256_DIGEST_LENGTH + 1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: output string length allocated not sufficient");
        return NVDS_MSGAPI_ERR;
    }
    char azure_conn_str[1024];
    if(fetch_connect_str(broker_str, cfg, azure_conn_str, sizeof(azure_conn_str)) == -1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Failure in fetching azure connection signature");
        return NVDS_MSGAPI_ERR;
    }
    string hashval = generate_sha256_hash(string(azure_conn_str));
    strcpy(output_str , hashval.c_str());
    return NVDS_MSGAPI_OK;
}
