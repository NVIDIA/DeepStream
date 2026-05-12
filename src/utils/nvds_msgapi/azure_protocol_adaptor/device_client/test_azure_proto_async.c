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

//This is a test program to perform connect, disconnect , send messages to Azure Iothub
//Use main thread to connect and multiple threads to perform asynchronous send

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
#define AZURE_PROTO_SO "libnvds_azure_proto.so"
#define AZURE_PROTO_PATH SO_PATH AZURE_PROTO_SO
#define AZURE_CFG_FILE "./cfg_azure.txt"
//connection string format: HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>
#define AZURE_CONNECT_STR NULL
#define MAX_LEN 1024

NvDsMsgApiHandle (*nvds_msgapi_connect_ptr)(char *connection_str, nvds_msgapi_connect_cb_t connect_cb, char *config_path);
NvDsMsgApiErrorType (*nvds_msgapi_send_async_ptr)(NvDsMsgApiHandle conn, char *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr);
NvDsMsgApiErrorType (*nvds_msgapi_disconnect_ptr)(NvDsMsgApiHandle h_ptr);
char* (*nvds_msgapi_getversion_ptr)(void);
char* (*nvds_msgapi_get_protocol_name_ptr)(void);
NvDsMsgApiErrorType (*nvds_msgapi_connection_signature_ptr)(char *connection_str, char *config_path, char *output_str, int max_len);

struct send_info_t {
    pid_t tid;
    int num;
};

void connect_cb(NvDsMsgApiHandle h_ptr, NvDsMsgApiEventType evt) {
    if(evt == NVDS_MSGAPI_EVT_DISCONNECT)
        printf("In sample prog: connect failed \n");
    else
        printf("In sample prog: connect success \n");
}

void send_callback (void *user_ptr, NvDsMsgApiErrorType completion_flag) {
    struct send_info_t *info = (struct send_info_t *) user_ptr;
    if(completion_flag == NVDS_MSGAPI_OK)
        printf("Thread [%d] , Message num %d : send success\n", info->tid, info->num);
    else
        printf("Thread [%d] , Message num %d : send failed\n", info->tid, info->num);
}

void *func(void *ptr) {
    NvDsMsgApiHandle ah = (NvDsMsgApiHandle) ptr;
    const char *msg = "Hello world";
    pid_t myid = syscall(SYS_gettid);
    for(int i=0; i < 10; i++) {
        struct send_info_t myinfo = {myid, i};
        nvds_msgapi_send_async_ptr(ah, NULL, (const uint8_t *)msg, strlen(msg), send_callback, &myinfo);
        sleep(1);
    }
    return NULL;
}

int main(int argc, char **argv) {
    void *so_handle;
    if(argc < 2)
        so_handle = dlopen(AZURE_PROTO_PATH, RTLD_LAZY);
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
    *(void **) (&nvds_msgapi_getversion_ptr) = dlsym(so_handle, "nvds_msgapi_getversion");
    *(void **) (&nvds_msgapi_get_protocol_name_ptr) = dlsym(so_handle, "nvds_msgapi_get_protocol_name");
    *(void **) (&nvds_msgapi_connection_signature_ptr) = dlsym(so_handle, "nvds_msgapi_connection_signature");

    if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "%s\n", error);
        exit(-1);
    }

    //There are 2 options to provide connection string
    //          format: HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>
    //option 1: Full connection string provided as a param in nvds_msgapi_connect()
    //option 2: The full device connection string is provided in config file.

    printf("Adapter protocol=%s , version=%s\n", nvds_msgapi_get_protocol_name_ptr(), nvds_msgapi_getversion_ptr());
    char query_conn_signature[MAX_LEN];
    if(nvds_msgapi_connection_signature_ptr((char *) AZURE_CONNECT_STR, (char *) AZURE_CFG_FILE, query_conn_signature, MAX_LEN) != NVDS_MSGAPI_OK) {
        printf("Error querying connection signature string. Exiting\n");
    }
    printf("connection signature queried= %s\n", query_conn_signature);

    NvDsMsgApiHandle ah = nvds_msgapi_connect_ptr((char *) AZURE_CONNECT_STR, connect_cb, (char *) AZURE_CFG_FILE);
    if(ah == NULL) {
        printf("Connect to Azure failed\n");
        exit(0);
    }
    printf("Azure: connect Success\n");
    pthread_t tid[NUM_THREADS];
    for(int i=0; i<NUM_THREADS; i++)
        pthread_create(&tid[i], NULL, &func, ah);

    for(int i=0; i<NUM_THREADS; i++)
        pthread_join(tid[i] , NULL);
    nvds_msgapi_disconnect_ptr(ah);
}
