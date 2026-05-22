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


#ifndef __NVGSTDS_C2D_MSG_H__
#define __NVGSTDS_C2D_MSG_H__

#include <gst/gst.h>
#include "nvmsgbroker.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NvDsC2DContext {
  gpointer libHandle;
  gchar *protoLib;
  gchar *connStr;
  gchar *configFile;
  gpointer uData;
  GHashTable *hashMap;
  NvMsgBrokerClientHandle connHandle;
  nv_msgbroker_subscribe_cb_t subscribeCb;
} NvDsC2DContext;

typedef struct NvDsMsgConsumerConfig {
  gboolean enable;
  gchar *proto_lib;
  gchar *conn_str;
  gchar *config_file_path;
  GPtrArray *topicList;
  gchar *sensor_list_file;
} NvDsMsgConsumerConfig;

NvDsC2DContext*
start_cloud_to_device_messaging (NvDsMsgConsumerConfig *config,
                                 nv_msgbroker_subscribe_cb_t cb,
                                 void *uData);
gboolean stop_cloud_to_device_messaging (NvDsC2DContext* uCtx);

#ifdef __cplusplus
}
#endif
#endif
