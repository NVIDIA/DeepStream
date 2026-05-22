/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPSTREAM_NMOS_APP_H__
#define __DEEPSTREAM_NMOS_APP_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_SOURCE_NUM  16
#define MAX_SINK_NUM    16

typedef enum NvDsNmosAppMode {
  NVDS_NMOS_APP_MODE_RECEIVE,
  NVDS_NMOS_APP_MODE_SEND,
  NVDS_NMOS_APP_MODE_RECVSEND
} NvDsNmosAppMode;

typedef enum NvDsNmosSrcType {
  NMOS_UDP_SRC_OSS = 1,
  NMOS_UDP_SRC_NV,
} NvDsNmosSrcType;

typedef enum NvDsNmosSinkType {
  NMOS_UDP_SINK_OSS = 1,
  NMOS_UDP_SINK_NV,
  NMOS_SRT_SINK,
  NMOS_XVIMAGE_SINK
} NvDsNmosSinkType;

typedef enum NvDsNmosSrtMode {
  NMOS_SRT_MODE_CALLER,
  NMOS_SRT_MODE_LISTENER,
  NMOS_SRT_MODE_RENDEZVOUS
} NvDsNmosSrtMode;

typedef struct NvDsNmosSrcConfig {
  gboolean enable;
  guint index;
  guint type;
  guint sinkType;
  guint packetsPerLine;
  guint payloadSize;
  gchar *sdpFile;
  gchar *sinkSdpFile;
  gchar *localIfaceIp;
  gchar *id;
  gchar *sinkId;
  gchar *srcSdpTxt;
  gchar *sinkSdpTxt;
  gchar *srtUri;
  guint srtMode;
  guint srtLatency;
  gchar *srtPassphrase;
  guint bitrate;
  guint iframeinterval;
  gchar *encodeCapsFilter;
  guint flipMethod;
} NvDsNmosSrcConfig;

typedef struct NvDsNmosSinkConfig {
  gboolean enable;
  guint index;
  guint type;
  guint packetsPerLine;
  guint payloadSize;
  gchar *sdpFile;
  gchar *localIfaceIp;
  gchar *id;
  gchar *sdpTxt;
} NvDsNmosSinkConfig;

typedef struct NvDsNmosAppConfig {
  gboolean enablePgie;
  guint httpPort;
  guint numSrc;
  guint numSink;
  gchar *seed;
  gchar *hostName;
  gchar *pgieConfFile;
  gboolean pluginType;
  NvDsNmosSrcConfig srcConfigs[MAX_SOURCE_NUM];
  NvDsNmosSinkConfig sinkConfigs[MAX_SINK_NUM];
} NvDsNmosAppConfig;

typedef struct NvDsNmosSrcBin
{
  GstElement *bin;
  GstElement *src;
  GstElement *queue;
  GstElement *payloader;
  GstElement *sink;
  gchar *mediaType;
  gchar *srcId;
  guint srcIndex;
} NvDsNmosSrcBin;

typedef struct NvDsNmosSinkBin
{
  GstElement *bin;
  GstElement *queue;
  GstElement *payloader;
  GstElement *sink;
  gchar *mediaType;
  gchar *id;
  guint index;
} NvDsNmosSinkBin;

typedef struct NvDsNmosAppCtx {
  GstElement *pipeline;
  guint watchId;
  gboolean isPipelineActive;
  GMainLoop *loop;
  NvDsNmosAppConfig config;
  GHashTable *sources;
  GHashTable *sinks;
} NvDsNmosAppCtx;

#ifdef __cplusplus
}
#endif

#endif