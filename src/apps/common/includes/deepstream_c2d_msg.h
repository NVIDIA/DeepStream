/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
