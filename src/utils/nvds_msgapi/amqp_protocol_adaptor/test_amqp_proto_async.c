/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

//This is a test program to perform connect, disconnect , send messages to amqp broker
//Use a single thread to connect and perform asynchronous send

#include<stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <dlfcn.h>
#include "nvds_msgapi.h"

#define NUM_THREADS 5
//Modify to reflect your own path
#define SO_PATH "/opt/nvidia/deepstream/deepstream/lib/"
#define AMQP_PROTO_SO "libnvds_amqp_proto.so"
#define AMQP_PROTO_PATH SO_PATH AMQP_PROTO_SO
#define AMQP_CFG_FILE "./cfg_amqp.txt"
//connection string format: host;port;username;password
#define AMQP_CONNECT_STR "localhost;5672;guest;guest"
#define MAX_LEN 256

NvDsMsgApiHandle (*nvds_msgapi_connect_ptr)(char *connection_str, nvds_msgapi_connect_cb_t connect_cb, char *config_path);
NvDsMsgApiErrorType (*nvds_msgapi_send_async_ptr)(NvDsMsgApiHandle conn, char *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr);
NvDsMsgApiErrorType (*nvds_msgapi_disconnect_ptr)(NvDsMsgApiHandle h_ptr);
void (*nvds_msgapi_do_work_ptr)(NvDsMsgApiHandle h_ptr);
NvDsMsgApiErrorType (*nvds_msgapi_subscribe_ptr)(NvDsMsgApiHandle conn, char **topics, int num_topics, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx);
char* (*nvds_msgapi_getversion_ptr)(void);
char* (*nvds_msgapi_get_protocol_name_ptr)(void);
NvDsMsgApiErrorType (*nvds_msgapi_connection_signature_ptr)(char *connection_str, char *config_path, char *output_str, int max_len);

int consumed_cnt = 0;

void connect_cb(NvDsMsgApiHandle h_ptr, NvDsMsgApiEventType evt) {
    if(evt == NVDS_MSGAPI_EVT_SUCCESS)
        printf("In sample prog: connect success \n");
    else
        printf("In sample prog: connect failed \n");
}

void send_callback (void *user_ptr, NvDsMsgApiErrorType completion_flag) {
    if(completion_flag == NVDS_MSGAPI_OK)
        printf("Message num %d : send success\n", *((int *) user_ptr));
    else
        printf("Message num %d : send failed\n", *((int *) user_ptr));
}

void subscribe_cb(NvDsMsgApiErrorType flag, void *msg, int len, char *topic, void *user_ptr) {
  int *ptr = (int *) user_ptr;
  if(flag == NVDS_MSGAPI_ERR) {
    printf("Error in consuming message[%d] from AMQP broker\n", *ptr);
  }
  else {
    printf("Consuming message[%d], on topic[%s]. Payload= %.*s\n", *ptr, topic, len, (const char *) msg);
  }
  consumed_cnt++;
}


int main(int argc, char **argv) {
    void *so_handle;
    if(argc < 2)
        so_handle = dlopen(AMQP_PROTO_PATH, RTLD_LAZY);
    else if(argc == 2)
        so_handle = dlopen(argv[1], RTLD_LAZY);
    else {
        printf("Invalid arguments to sample applicaiton\n");
        printf("Usage: \n\t./test_async [optional path_to_so_lib] \n\n");
        exit(1);
    }
    char *error;
    if (!so_handle) {
        printf("unable to open shared library\n");
        exit(-1);
    }
    *(void **) (&nvds_msgapi_connect_ptr) = dlsym(so_handle, "nvds_msgapi_connect");
    *(void **) (&nvds_msgapi_send_async_ptr) = dlsym(so_handle, "nvds_msgapi_send_async");
    *(void **) (&nvds_msgapi_disconnect_ptr) = dlsym(so_handle, "nvds_msgapi_disconnect");
    *(void **) (&nvds_msgapi_do_work_ptr) = dlsym(so_handle, "nvds_msgapi_do_work");
    *(void **) (&nvds_msgapi_subscribe_ptr) = dlsym(so_handle, "nvds_msgapi_subscribe");
    *(void **) (&nvds_msgapi_getversion_ptr) = dlsym(so_handle, "nvds_msgapi_getversion");
    *(void **) (&nvds_msgapi_get_protocol_name_ptr) = dlsym(so_handle, "nvds_msgapi_get_protocol_name");
    *(void **) (&nvds_msgapi_connection_signature_ptr) = dlsym(so_handle, "nvds_msgapi_connection_signature");

    if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "%s\n", error);
        exit(-1);
    }

    printf("Adapter protocol=%s , version=%s\n", nvds_msgapi_get_protocol_name_ptr(), nvds_msgapi_getversion_ptr());

    char query_conn_signature[MAX_LEN];
    if(nvds_msgapi_connection_signature_ptr((char *) AMQP_CONNECT_STR, (char *) AMQP_CFG_FILE, query_conn_signature, MAX_LEN) != NVDS_MSGAPI_OK) {
        printf("Error querying connection signature string. Exiting\n");
    }
    printf("connection string queried= %s\n", query_conn_signature);

    //There are 2 options to provide connection string
    //option 1: provide connection string as param to nvds_msgapi_connect()
    //option 2: The full connection details in config file and connection params provided in nvds_msgapi_connect() as NULL

    NvDsMsgApiHandle ah = nvds_msgapi_connect_ptr((char *) AMQP_CONNECT_STR, connect_cb, (char *) AMQP_CFG_FILE);
    if(ah == NULL) {
        printf("Connect to amqp broker failed\n");
        exit(0);
    }
    printf("Connect Success\n");

    //Subscribe to topics
    const char *topics[] = {"topic1", "topic2"};
    int num_topics=2;
    if(nvds_msgapi_subscribe_ptr(ah, (char **)topics, num_topics, subscribe_cb, &consumed_cnt) != NVDS_MSGAPI_OK) {
      printf("AMQP subscription to topic[s] failed. Exiting \n");
      exit(-1);
    }

    for(int i=0; i < 10; i++) {
        char msg[100];
        sprintf(msg, "Hello%d\n", i);
        if(nvds_msgapi_send_async_ptr(ah, (char *) "topic1", (const uint8_t *)msg, strlen(msg), send_callback, &i) != NVDS_MSGAPI_OK)
            printf("Message send failed\n");
        nvds_msgapi_do_work_ptr(ah);
    }
    sleep(1);
    nvds_msgapi_disconnect_ptr(ah);
}
