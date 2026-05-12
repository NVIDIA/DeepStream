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
const char * AZURE_PROTO_SO = "./libnvds_azure_edge_proto.so";

NvDsMsgApiHandle (*nvds_msgapi_connect_ptr)(char *connection_str, nvds_msgapi_connect_cb_t connect_cb, char *config_path);
NvDsMsgApiErrorType (*nvds_msgapi_send_async_ptr)(NvDsMsgApiHandle conn, char *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr);
NvDsMsgApiErrorType (*nvds_msgapi_disconnect_ptr)(NvDsMsgApiHandle h_ptr);

struct send_info_t {
    pid_t tid;
    int num;
    const char *msg;
};

void connect_cb(NvDsMsgApiHandle h_ptr, NvDsMsgApiEventType evt) {
    if(evt == NVDS_MSGAPI_EVT_DISCONNECT)
        printf("In sample prog: Azure connect failed \n");
    else
        printf("In sample prog: Azure connect success \n");
}

void send_callback (void *user_ptr, NvDsMsgApiErrorType completion_flag) {
    struct send_info_t *info = (struct send_info_t *) user_ptr;
    if(completion_flag == NVDS_MSGAPI_OK)
        printf("Thread [%d] , Message num %d : send success. message sent=%s\n", info->tid, info->num, info->msg);
    else
        printf("Thread [%d] , Message num %d : send failed. message = %s\n", info->tid, info->num, info->msg);
}

void *func(void *ptr) {
    NvDsMsgApiHandle ah = (NvDsMsgApiHandle) ptr;
    const char *msg = "Hello world";
    pid_t myid = syscall(SYS_gettid);
    for(int i=0; i < 200000; i++) {
        struct send_info_t myinfo = {myid, i, msg};
        nvds_msgapi_send_async_ptr(ah, (char *)"sample_topic", (const uint8_t *)msg, strlen(msg), send_callback, &myinfo);
        sleep(1);
    }
    return NULL;
}

int main(int argc, char **argv) {
    void *so_handle;
    if(argc < 2)
        so_handle = dlopen(AZURE_PROTO_SO, RTLD_LAZY);
    else if(argc == 2)
        so_handle = dlopen(argv[1], RTLD_LAZY);
    else {
        printf("Invalid arguments to sample applicaiton\n");
        printf("Usage: \n\t./test_async <path_to_libnvds_azure_edge_proto.so_lib] \n\n");
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

    if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "%s\n", error);
        exit(-1);
    }

    //For azure module client - the edge device connection details should be mentioned in /etc/iotedge/config.yaml
    NvDsMsgApiHandle ah = nvds_msgapi_connect_ptr(NULL, connect_cb, (char *)"/root/cfg_azure.txt");
    if(ah == NULL) {
        printf("COnnect to Azure failed\n");
        exit(0);
    }
    printf("main: after conenct\n");
    pthread_t tid[NUM_THREADS];
    for(int i=0; i<NUM_THREADS; i++)
        pthread_create(&tid[i], NULL, &func, ah);

    for(int i=0; i<NUM_THREADS; i++)
        pthread_join(tid[i] , NULL);
    nvds_msgapi_disconnect_ptr(ah);
}
