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
/* Amqp adaptor used to conenct/send/disconnect to amqp broker */

#include <syslog.h>
#include <time.h>
#include <sys/timeb.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <amqp_tcp_socket.h>
#include <amqp.h>
#include <amqp_framing.h>
#include <glib.h>
#include <iostream>
#include <list>
#include <algorithm>
#include <vector>
#include <thread>
#include <openssl/sha.h>
#include "nvds_msgapi.h"
#include "nvds_logger.h"
#include "nvds_utils.h"

using namespace std;

#define NVDS_MSGAPI_VERSION "4.0"
#define NVDS_MSGAPI_PROTOCOL "AMQP"
#define LOG_CAT "DSLOG:NVDS_AMQP_PROTO"

#define DFLT_RMQ_EXCHANGE_NAME "amq.topic"
#define DFLT_PORT 5672
#define DFLT_TOPIC "deepstream_topic"
#define MAX_LEN 512
#define MAX_QUEUE_NAME_LEN 100

#define PASSWORD_VAR "PASSWORD_AMQP"
#define USERNAME_VAR "USER_AMQP"

static int AMQP_FRAME_SIZE=131072;
static int AMQP_HEARTBEAT=0; // The number of seconds between heartbeat frames to request of the broker. 0 to disable

/* Message details:
 * topic = message topic name
 * msg   = payload
 * send_cb = user callback func
 * user_ctx = user pointer
 */
struct send_msg_info_t {
  string topic;
  string msg;
  nvds_msgapi_send_cb_t send_cb;
  void *user_ctx;
};

/* Details of amqp connection handle:
 * conn : amqp connection object
 * conn_consume: amqp consume connection object
 * subscription_on: Flag to check if subscription is ON
 * c_thread: consumer thread
 * RMQ_EXCHANGE_NAME : Exchange name
 * TOPIC_NAME : Topic name for messages
 * send_cb_list: List of pending outgoing messages
 * disconnect: Flag to indicate if connection msut be terminated
 * consume_queue_list: List of consumer queues
 */
typedef struct {
  amqp_connection_state_t conn;
  amqp_connection_state_t conn_consume;
  bool subscription_on;
  thread c_thread;
  char RMQ_EXCHANGE_NAME[MAX_LEN];
  char TOPIC_NAME[MAX_LEN];
  nvds_msgapi_connect_cb_t connect_cb;
  list<struct send_msg_info_t> send_cb_list;
  bool disconnect;
  vector<amqp_bytes_t> consume_queue_list;
} nvds_rmq_client_handle_t;

int parse_config(nvds_rmq_client_handle_t *rh, char *str, char *config, char *ip_addr, int *port, char *username, char *pwd);
void CLEANUP(amqp_connection_state_t conn);
amqp_connection_state_t create_amqp_ctx(char *ip_addr, char *username, char *pwd, int port);
void consume(nvds_rmq_client_handle_t *rh, nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx);

// Free resources
void CLEANUP(amqp_connection_state_t conn) {
  if(conn) {
    amqp_channel_close(conn, 1, AMQP_REPLY_SUCCESS);
    amqp_connection_close(conn, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(conn);
  }
}

/* nvds_msgapi function for retrieving nvds_msgapi version
*/
char *nvds_msgapi_getversion() {
  return (char *) NVDS_MSGAPI_VERSION;
}

/*Function to fetch connection string provided for amqp authentication
  option 1: Connection string param provided in nvds_msgapi_connect() with format "host;port;username;password"
  option 2: Part of connection string param provided in nvds_msgapi_connect() with format "host;port;username"
            and password is separately provided in config file (This option to be depracated soon)
  option 3: Connection string can be NULL, all connection details to be provided in config file
  return  : success = 0 , fail = -1
*/
int parse_config(nvds_rmq_client_handle_t *rh, char *str, char *config, char *ip_addr, int *port, char *username, char *pwd) {
    char cfg_ipaddr[MAX_LEN]="";
    char cfg_uname[MAX_LEN]="";
    char cfg_pwd[MAX_LEN]="";
    int cfg_port =  DFLT_PORT;
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    gchar* val = NULL;
    const gchar* var_user = NULL;
    const gchar* var_pwd = NULL;

    // Fetch username and password from environmental variable
    var_user = g_getenv(USERNAME_VAR);
    var_pwd = g_getenv(PASSWORD_VAR);

    // parse config if not null and file length is not 0
    if((config != NULL) && (strlen(config))) {
        GKeyFile *gcfg_file = g_key_file_new();
        if (!g_key_file_load_from_file (gcfg_file, config, G_KEY_FILE_NONE, &error)) {
            nvds_log(LOG_CAT, LOG_ERR , "AMQP adaptor: Failed to parse configfile %s\n", config);
            free_gobjs(gcfg_file, error, keys, val);
            return -1;
        }

        char grpname[15] = "message-broker";
        keys = g_key_file_get_keys(gcfg_file, grpname, NULL, &error);
        for (key = keys; *key; key++) {
            // Read config to fetch username for authentication -- to be deprecated in favor of env variable
            if (!g_strcmp0(*key, "username")) {
                val = g_key_file_get_string (gcfg_file, grpname, "username",&error);
                int size = g_strlcpy(cfg_uname, val, MAX_LEN);
                if(size >= MAX_LEN) {
                    nvds_log(LOG_CAT, LOG_ERR , "AMQP username string size exceeds max len of %dbytes\n", MAX_LEN);
                    free_gobjs(gcfg_file, error, keys, val);
                    return -1;
                }
            }
            // Read config to fetch password for authentication -- to be deprecated in favor of env variable
            else if (!g_strcmp0(*key, "password")) {
                val = g_key_file_get_string (gcfg_file, grpname, "password",&error);
                int size = g_strlcpy(cfg_pwd, val, MAX_LEN);
                if(size >= MAX_LEN) {
                    nvds_log(LOG_CAT, LOG_ERR , "AMQP password string size exceeds max len of %dbytes\n", MAX_LEN);
                    free_gobjs(gcfg_file, error, keys, val);
                    return -1;
                }
            }
            // Read config to fetch hostname for connection
            else if (!g_strcmp0(*key, "hostname")) {
                val = g_key_file_get_string (gcfg_file, grpname, "hostname",&error);
                int size = g_strlcpy(cfg_ipaddr, val, MAX_LEN);
                if(size >= MAX_LEN) {
                    nvds_log(LOG_CAT, LOG_ERR , "AMQP hostname string size exceeds max len of %dbytes\n", MAX_LEN);
                    free_gobjs(gcfg_file, error, keys, val);
                    return -1;
                }
            }
            // Read config to fetch name of the exchange to be used by the adaptor
            else if (!g_strcmp0(*key, "exchange")) {
                if(rh) {
                    val = g_key_file_get_string (gcfg_file, grpname, "exchange",&error);
                    int size = g_strlcpy(rh->RMQ_EXCHANGE_NAME, val, MAX_LEN);
                    if(size >= MAX_LEN) {
                        nvds_log(LOG_CAT, LOG_ERR , "AMQP exchange string size exceeds max len of %dbytes\n", MAX_LEN);
                        free_gobjs(gcfg_file, error, keys, val);
                        return -1;
                    }
                }
            }
            // Read config to fetch message topic
            else if (!g_strcmp0(*key, "topic")) {
                if(rh) {
                    val = g_key_file_get_string (gcfg_file, grpname, "topic",&error);
                    int size = g_strlcpy(rh->TOPIC_NAME, val, MAX_LEN);
                    if(size >= MAX_LEN) {
                        nvds_log(LOG_CAT, LOG_ERR , "AMQP topic string size exceeds max len of %dbytes\n", MAX_LEN);
                        free_gobjs(gcfg_file, error, keys, val);
                        return -1;
                    }
                }
            }
            // Read config to fetch port number
            else if (!g_strcmp0(*key, "port")) {
                cfg_port = g_key_file_get_integer(gcfg_file, grpname, "port", &error);
            }
            // Read config to fetch the maximum AMQP frame size in bytes
            else if (!g_strcmp0(*key, "amqp-framesize")) {
                AMQP_FRAME_SIZE = g_key_file_get_integer(gcfg_file, grpname, "amqp-framesize", &error);
            }
            // Read config to fetch the time in seconds between heartbeat frames
            else if (!g_strcmp0(*key, "amqp-heartbeat")) {
                AMQP_HEARTBEAT = g_key_file_get_integer(gcfg_file, grpname, "amqp-heartbeat", &error);
            }

            if (val != NULL) {
                g_free(val);
                val = NULL;
            }
        }
        free_gobjs(gcfg_file, error, keys, NULL);
    }

    //Look for connection details provided(if any) in params to nvds_msgapi_connect
    if((str != NULL) && (strlen(str) >= 5)) {
        char port_str[16];
        string connect_str(str);
        size_t n = count(connect_str.begin(), connect_str.end(), ';');
        // Format is host;port;username;password, host;port;username, or host;port (PREFERRED) -- first two options are to be deprecated
        if(n==3)
            // To be deprecated
            sscanf(str, "%[^';'];%[^';'];%[^';'];%s", ip_addr, port_str, username, pwd);
        else if(n==2) {
            // To be deprecated
            if(!strcmp(cfg_pwd,"") && var_pwd == NULL) {
                nvds_log(LOG_CAT, LOG_ERR , "AMQP authentication password not provided in cfg file or environmental variables\n");
                return -1;
            }
            sscanf(str, "%[^';'];%[^';'];%s", ip_addr, port_str, username);
            g_strlcpy(pwd, cfg_pwd, MAX_LEN);
        }
        else if (n==1) {
            // Preferred format, host;port
            if((!strcmp(cfg_pwd,"") || !strcmp(cfg_uname,"")) && (var_pwd == NULL || var_user == NULL)) {
                nvds_log(LOG_CAT, LOG_ERR , "AMQP authentication user and pass not provided in cfg file or environmental variables\n");
                return -1;
            }
            sscanf(str, "%[^';'];%s", ip_addr, port_str);
            g_strlcpy(username, cfg_uname, MAX_LEN);
            g_strlcpy(pwd, cfg_pwd, MAX_LEN);
        }
        else {
            nvds_log(LOG_CAT, LOG_ERR , "AMQP connection string format is invalid\n");
            return -1;
        }
        *port = atoi(port_str);
    }
    //Fetch connection details were provided in cfg file
    else {
        *port = cfg_port;
        g_strlcpy(ip_addr, cfg_ipaddr, MAX_LEN);
        // Username and password in config file to be deprecated
        g_strlcpy(username, cfg_uname, MAX_LEN);
        g_strlcpy(pwd, cfg_pwd, MAX_LEN);
    }

    // Check that username and password do not exceed maximum length
    // Override user and password with environmental variable if given
    if (var_user != NULL) {
        if (MAX_LEN < (int) strlen(var_user) + 1) {
            nvds_log(LOG_CAT, LOG_ERR, "Username length exceeds capacity of %d", MAX_LEN);
            return -1;
        }
        else{
            g_strlcpy(username, var_user, MAX_LEN);
        }
    } 
    else {
        nvds_log(LOG_CAT, LOG_INFO,  "Username not provided through environmental variable.\n");
    }

    if (var_pwd != NULL) {
        if(MAX_LEN < (int) strlen(var_pwd) + 1) {
            nvds_log(LOG_CAT, LOG_ERR, "Password length exceeds capacity of %d", MAX_LEN);
            return -1;
        }
        else{
            g_strlcpy(pwd, var_pwd, MAX_LEN);
        }
    }
    else {
        nvds_log(LOG_CAT, LOG_INFO,  "Password not provided through environmental variable.\n");
    }

    return 0;
}

/* Function to create amqp context
*/ 
amqp_connection_state_t create_amqp_ctx(char *ip_addr, char *username, char *pwd, int port){
   // Create connection
   amqp_connection_state_t conn = amqp_new_connection();
    if(conn == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "create_amqp_ctx: Error creating amqp connection");
        return NULL;
    }
    // Create socket
    amqp_socket_t *socket = amqp_tcp_socket_new(conn);
    if (!socket) {
        CLEANUP(conn);
        nvds_log(LOG_CAT, LOG_ERR , "create_amqp_ctx: Error creating socket");
        return NULL;
    }
    // Open socket
    int status = amqp_socket_open(socket, ip_addr , port);
    if (status) {
        CLEANUP(conn);
        nvds_log(LOG_CAT, LOG_ERR , "create_amqp_ctx: Error opening socket. Perhaps invalid ip addr or port or amqp broker service not running\n");
        return NULL;
    }
    // Login with credentials and other configurations
    amqp_rpc_reply_t login_reply = amqp_login(conn, "/", 0, AMQP_FRAME_SIZE, AMQP_HEARTBEAT, AMQP_SASL_METHOD_PLAIN, username, pwd);
    if (login_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        CLEANUP(conn);
        nvds_log(LOG_CAT, LOG_ERR , "create_amqp_ctx: Login credentials error. Check username & password");
        nvds_log_close();
        return NULL;
    }

    // Open channel 1
    amqp_channel_open(conn, 1);
    // Return connection state
    return conn;
}

/* nvds_msgapi function to connect to the AMQP broker
*/
NvDsMsgApiHandle nvds_msgapi_connect(char *str, nvds_msgapi_connect_cb_t connect_cb, char *config_path) {
    char ip_addr[MAX_LEN], username[MAX_LEN], pwd[MAX_LEN];
    int port;
    nvds_log_open();
    // Create amqp client handle
    nvds_rmq_client_handle_t *rh = new (nothrow) nvds_rmq_client_handle_t;
    if(rh == NULL){
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: malloc failed for creating handle. Exiting..");
        nvds_log_close();
        return NULL;
    }
    // Set exchange name and topic name in handle to defaults
    g_strlcpy(rh->RMQ_EXCHANGE_NAME , DFLT_RMQ_EXCHANGE_NAME, MAX_LEN);
    g_strlcpy(rh->TOPIC_NAME , DFLT_TOPIC, MAX_LEN);
    // Parse config file and connection string 
    if(parse_config(rh, str, config_path, ip_addr, &port, username, pwd)) {
        delete rh;
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure to fetch login credentails");
        nvds_log_close();
        return NULL;
    }
    // Create amqp producer context
    rh->conn = create_amqp_ctx(ip_addr, username, pwd, port);
    if(rh->conn == NULL) {
        delete rh;
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure to create AMQP context");
        nvds_log_close();
        return NULL;
    }
    // Create amqp consumer context
    rh->conn_consume=create_amqp_ctx(ip_addr, username, pwd, port);
    if(rh->conn_consume == NULL) {
        delete rh;
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connect: Failure to create AMQP consume context");
        nvds_log_close();
        return NULL;
    }
    rh->subscription_on=false;
    rh->connect_cb = connect_cb;
    rh->disconnect=false;
    nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_connect: Success");
    return (NvDsMsgApiHandle) rh;
}

/* Function used by AMQP consumer thread to listen & consume incoming messages
 * User callback will be called to forward consumed messages or errors(if any)
 */
void consume(nvds_rmq_client_handle_t *rh, nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx) {
    while(!rh->disconnect) {
        amqp_rpc_reply_t ret;
        amqp_envelope_t envelope;
        struct timeval timeout = {0, 0};
        /* Release any existing amqp_connection_state_t owned memory */
        amqp_maybe_release_buffers(rh->conn_consume);
        /* Consume message */
        ret = amqp_consume_message(rh->conn_consume, &envelope, &timeout, 0);
        if (AMQP_RESPONSE_NORMAL == ret.reply_type) {
        /* Print the message value/payload. */
            if (envelope.message.properties._flags & AMQP_BASIC_CONTENT_TYPE_FLAG) {
                string msg=string((char *) envelope.message.body.bytes, envelope.message.body.len);
                int msg_len=(int) envelope.message.body.len;
                string topic=string((char *) envelope.routing_key.bytes, envelope.routing_key.len);
                cb(NVDS_MSGAPI_OK, (void *) msg.c_str(), msg_len, (char *)topic.c_str(), user_ctx);
                nvds_log(LOG_CAT, LOG_INFO, "Amqp Message consumed:[%.*s] on topic[ %.*s]\n",
                        (int) envelope.message.body.len,
                        (char *) envelope.message.body.bytes,
                        (int) envelope.routing_key.len, (char *) envelope.routing_key.bytes);
            }
        }
        amqp_destroy_envelope(&envelope);
        usleep(5000);
    }
    for(auto item: rh->consume_queue_list){
        amqp_bytes_free(item);
    }
}

/* This api will be used to create & intialize amqp consumer
 * A consumer thread is spawned to listen & consume messages on topics specified in the api
 */
NvDsMsgApiErrorType nvds_msgapi_subscribe (NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx) {
    nvds_rmq_client_handle_t *rh= (nvds_rmq_client_handle_t *)h_ptr;
    if(rh == NULL ) {
        nvds_log(LOG_CAT, LOG_ERR, "Amqp connection handle passed for nvds_msgapi_subscribe() = NULL. Subscribe failed\n");
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

    if(rh->subscription_on) {
        nvds_log(LOG_CAT, LOG_INFO, "AMQP Subscription already exists. Ignoring this subscription call\n");
        return NVDS_MSGAPI_ERR;
    }

    // Subscribe to each topic
    for (int i = 0 ; i < num_topics ; i++){
        // Declare queue
        amqp_queue_declare_ok_t *r = amqp_queue_declare(rh->conn_consume, 1, amqp_empty_bytes, 0, 0, 0, 1, amqp_empty_table);
        if(r == NULL) {
            nvds_log(LOG_CAT, LOG_ERR, "nvds_msgapi_subscribe: amqp_queue_declare failed\n");
            return NVDS_MSGAPI_ERR;
        }
        // Retrieve queue name
        amqp_bytes_t queuename = amqp_bytes_malloc_dup(r->queue);
        if (queuename.bytes == NULL) {
          nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_subscribe: Out of memory while copying queue name");
          return NVDS_MSGAPI_ERR;
        }
        // Add queue to list of queues to be consumed
        rh->consume_queue_list.push_back(queuename);
        // Bind the queue to the exchange with topic as routing key
        amqp_queue_bind(rh->conn_consume, 1, queuename, amqp_cstring_bytes(rh->RMQ_EXCHANGE_NAME), amqp_cstring_bytes(topics[i]), amqp_empty_table);
        // Register consumer 
        amqp_basic_consume(rh->conn_consume, 1, queuename, amqp_empty_bytes, 0, 1, 0, amqp_empty_table);
    }
    // Construct consumer thread
    rh->c_thread = thread(consume, rh, cb, user_ctx);
    rh->subscription_on=true;
    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for synchronous message send
*/
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *msg, size_t nbuf) {
    nvds_rmq_client_handle_t *rh = (nvds_rmq_client_handle_t *) h_ptr;
    if((topic == NULL) || (!strlen(topic)) || (strlen(topic) >= MAX_LEN)) {
        nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_send_async: using msg topic %s\n", rh->TOPIC_NAME);
    }
    else
        g_strlcpy(rh->TOPIC_NAME, topic, MAX_LEN);

    // Initialize properties
    amqp_basic_properties_t props;
    props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG | AMQP_BASIC_DELIVERY_MODE_FLAG;
    props.content_type = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2; /* persistent delivery mode */
    //The publish api is Blocking
    int retval = amqp_basic_publish(rh->conn,
                                    1,
                                    amqp_cstring_bytes(rh->RMQ_EXCHANGE_NAME),
                                    amqp_cstring_bytes(rh->TOPIC_NAME),
                                    0,
                                    0,
                                    &props,
                       amqp_cstring_bytes((const char *) msg));
    if (retval) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_send: Error performing send. value returned = %d\n", retval);

        //check if connection is down and if yes report back
        if(rh->connect_cb && (retval == AMQP_STATUS_CONNECTION_CLOSED ||
                              retval == AMQP_STATUS_SOCKET_ERROR ||
                              retval == AMQP_STATUS_SOCKET_CLOSED )){
                rh->connect_cb(h_ptr, NVDS_MSGAPI_EVT_SERVICE_DOWN);
        }
        return NVDS_MSGAPI_ERR;
    }
    else {
        nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_send: Success sent msg %.*s\n", nbuf, (const char *) msg);
        return NVDS_MSGAPI_OK;
    }
}

/* nvds_msgapi function for asynchronous message send
*/
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char  *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr) {
    nvds_rmq_client_handle_t *rh = (nvds_rmq_client_handle_t *) h_ptr;

    if((topic == NULL) || (!strlen(topic)) || (strlen(topic) >= MAX_LEN)) {
        nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_send_async: using msg topic %s\n", rh->TOPIC_NAME);
    }
    else
        g_strlcpy(rh->TOPIC_NAME, topic, MAX_LEN);
    
    // Create msg_info struct and add to list of messages to be sent
    struct send_msg_info_t msg_info = {(const char *) rh->TOPIC_NAME, string((const char *) payload, nbuf), send_callback, user_ptr};
    rh->send_cb_list.push_back(msg_info);
    return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for performing work in nvmsgbroker plugin. In the case of AMQP, this work is asynchronous send
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) {
    nvds_rmq_client_handle_t *rh = (nvds_rmq_client_handle_t *) h_ptr;

    // Check if any async send operations pending
    if(! rh->send_cb_list.empty()) {
        struct send_msg_info_t node = rh->send_cb_list.front();
        // Initialize properties
        amqp_basic_properties_t props;
        props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG | AMQP_BASIC_DELIVERY_MODE_FLAG;
        props.content_type = amqp_cstring_bytes("text/plain");
        props.delivery_mode = 2; /* persistent delivery mode */
        //The publish api is Blocking
        int retval = amqp_basic_publish(rh->conn,
                                    1,
                                    amqp_cstring_bytes(rh->RMQ_EXCHANGE_NAME),
                                    amqp_cstring_bytes(node.topic.c_str()),
                                    0,
                                    0,
                                    &props,
                       amqp_cstring_bytes(node.msg.c_str()));
        if (retval) {
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_do_work: Error performing send. value returned = %d\n", retval);
            ((nvds_msgapi_send_cb_t) node.send_cb) (node.user_ctx , NVDS_MSGAPI_ERR);

            //check if connection is down and if yes report back
            if(rh->connect_cb && (retval == AMQP_STATUS_CONNECTION_CLOSED ||
                              retval == AMQP_STATUS_SOCKET_ERROR ||
                              retval == AMQP_STATUS_SOCKET_CLOSED )){
                rh->connect_cb(h_ptr, NVDS_MSGAPI_EVT_SERVICE_DOWN);
            }
        }
        else {
            nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_do_work: Message sent = %.*s\n", node.msg.size(), node.msg.c_str());
            ((nvds_msgapi_send_cb_t) node.send_cb) (node.user_ctx , NVDS_MSGAPI_OK);
        }
        rh->send_cb_list.pop_front();
    }
}

/* nvds_msgapi function for disconnecting from the broker
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr) {
    nvds_rmq_client_handle_t *rh = (nvds_rmq_client_handle_t *) h_ptr;
    if(rh==NULL){
        nvds_log(LOG_CAT, LOG_INFO , "nvds_msgapi_disconnect: Disconnect called with empty handle. Ignoring..");
        return NVDS_MSGAPI_OK;
    }
    rh->disconnect=true;
    // Wait for consumer thread to complete
    if(rh->c_thread.joinable())
        rh->c_thread.join();
    // AMQP cleanup
    CLEANUP(rh->conn);
    CLEANUP(rh->conn_consume);
    delete rh;
    rh=NULL;
    nvds_log(LOG_CAT, LOG_INFO , "Disconnecting Amqp adapter..");
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
    g_strlcpy(output_str,"", MAX_LEN);

    //check if share-connection config option is turned ON
    char reuse_connection[16]="";
    if(fetch_config_value(cfg, "share-connection", reuse_connection, sizeof(reuse_connection), LOG_CAT) != NVDS_MSGAPI_OK) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing amqp share-connection config option");
        return NVDS_MSGAPI_ERR;
    }
    if(strcmp(reuse_connection, "1")) {
        nvds_log(LOG_CAT, LOG_INFO, "nvds_msgapi_connection_signature: amqp connection sharing disabled. Hence connection signature cant be returned");
        return NVDS_MSGAPI_OK;
    }

    if(broker_str == NULL && cfg == NULL) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Both AMQP broker_str and path to cfg cant be NULL");
        return NVDS_MSGAPI_ERR;
    }
    if(max_len < 2 * SHA256_DIGEST_LENGTH + 1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: output string length allocated not sufficient");
        return NVDS_MSGAPI_ERR;
    }
    char amqp_connection_str[MAX_LEN];
    char ip_addr[MAX_LEN], username[MAX_LEN], pwd[MAX_LEN];
    int port;

    if(parse_config(NULL, broker_str, cfg, ip_addr, &port, username, pwd) == -1) {
        nvds_log(LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Failure in fetching amqp connection string");
        return NVDS_MSGAPI_ERR;
    }

    // Generate signature from connection string
    g_snprintf(amqp_connection_str, sizeof (amqp_connection_str), "%s;%d;%s;%s", ip_addr,port,username,pwd);
    string hashval = generate_sha256_hash(string(amqp_connection_str, MAX_LEN));
    g_strlcpy(output_str, hashval.c_str(), MAX_LEN);
    return NVDS_MSGAPI_OK;
}
