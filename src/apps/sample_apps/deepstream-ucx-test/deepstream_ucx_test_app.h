/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DEEPSTREAM_UCX_TEST_APP_H
#define DEEPSTREAM_UCX_TEST_APP_H

#include <glib.h>
#include <stdio.h>

/* Constants definitions */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"


/* Server Arguments */

typedef struct {
  gchar *addr;
  gint64 port;
  gint64 width;
  gint64 height;
  gchar *uri;
} ServerTest1Args;

typedef struct {
  gchar *addr;
  gint64 port;
  gint64 width;
  gint64 height;
  gchar *libpath;
  gchar *output;
} ServerTest2Args;

typedef struct {
  gchar *addr;
  gint64 port;
  gchar *libpath;
  gchar *uri;
} ServerTest3Args;

/* Client Arguments */

typedef struct {
    gchar *addr;
    gint64 port;
    gint64 width;
    gint64 height;
    gchar *outfile;
} ClientTest1Args;

typedef struct {
    gchar *addr;
    gint64 port;
    gchar *inferpath;
    gchar *libpath;
    gchar *uri;
} ClientTest2Args;

typedef struct {
    gchar *addr;
    gint64 port;
    gchar *libpath;
    gchar *outfile;
} ClientTest3Args;

/* Common Structs */

typedef enum {
  SERVER,
  CLIENT,
  INIT
} Role;

typedef struct {
  int test_id;
  GMainLoop *loop;
  GstElement *pipeline;
  Role role;
  gboolean is_nvinfer_server;

  union {
    ServerTest1Args server_test1;
    ServerTest2Args server_test2;
    ServerTest3Args server_test3;
    ClientTest1Args client_test1;
    ClientTest2Args client_test2;
    ClientTest3Args client_test3;
  } args;
} UcxTest;

/* Function prototypes */

// Test 1
int test1_client_parse_args(int argc, char *argv[], UcxTest *t);
int test1_client_setup_pipeline(UcxTest *t);

int test1_server_parse_args(int argc, char *argv[], UcxTest *t);
int test1_server_setup_pipeline(UcxTest *t);

// Test 2
int test2_client_parse_args(int argc, char *argv[], UcxTest *t);
int test2_client_setup_pipeline(UcxTest *t);

int test2_server_parse_args(int argc, char *argv[], UcxTest *t);
int test2_server_setup_pipeline(UcxTest *t);

// Test 3
int test3_client_parse_args(int argc, char *argv[], UcxTest *t);
int test3_client_setup_pipeline(UcxTest *t);

int test3_server_parse_args(int argc, char *argv[], UcxTest *t);
int test3_server_setup_pipeline(UcxTest *t);


typedef struct {
  int (*parse_args)(int argc, char *argv[], UcxTest *t);
  int (*setup_pipeline)(UcxTest *t);
} OperationFunctions;

typedef struct {
  OperationFunctions client;
  OperationFunctions server;
} TestOps;

TestOps testOperations[] = {
  {
    {test1_client_parse_args, test1_client_setup_pipeline},
    {test1_server_parse_args, test1_server_setup_pipeline}
  },
  {
    {test2_client_parse_args, test2_client_setup_pipeline},
    {test2_server_parse_args, test2_server_setup_pipeline}
  },
  {
    {test3_client_parse_args, test3_client_setup_pipeline},
    {test3_server_parse_args, test3_server_setup_pipeline}
  }
};



#endif // DEEPSTREAM_UCX_TEST_APP_H
