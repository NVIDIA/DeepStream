/**
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

#ifndef _GST_NVMSGBROKER_H_
#define _GST_NVMSGBROKER_H_

#include <gst/base/gstbasesink.h>
#include "nvds_msgapi.h"
#include "nvmsgbroker.h"

G_BEGIN_DECLS

#define GST_TYPE_NVMSGBROKER   (gst_nvmsgbroker_get_type())
#define GST_NVMSGBROKER(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVMSGBROKER,GstNvMsgBroker))
#define GST_NVMSGBROKER_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVMSGBROKER,GstNvMsgBrokerClass))
#define GST_IS_NVMSGBROKER(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVMSGBROKER))
#define GST_IS_NVMSGBROKER_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVMSGBROKER))

typedef struct _GstNvMsgBroker GstNvMsgBroker;
typedef struct _GstNvMsgBrokerClass GstNvMsgBrokerClass;

typedef NvDsMsgApiHandle (*nvds_msgapi_connect_ptr)(const char *connection_str,
    nvds_msgapi_connect_cb_t connect_cb, const char *config_path);

typedef NvDsMsgApiErrorType (*nvds_msgapi_send_ptr)(NvDsMsgApiHandle conn, const char *topic,
    const uint8_t *payload, size_t nbuf);

typedef NvDsMsgApiErrorType (*nvds_msgapi_send_async_ptr)(NvDsMsgApiHandle h_ptr,
    char *topic, const uint8_t *payload, size_t nbuf,
    nvds_msgapi_send_cb_t send_callback, void *user_ptr);

typedef void (*nvds_msgapi_do_work_ptr) (NvDsMsgApiHandle h_ptr);

typedef NvDsMsgApiErrorType (*nvds_msgapi_disconnect_ptr)(NvDsMsgApiHandle conn);

struct _GstNvMsgBroker
{
  GstBaseSink parent;

  GQuark dsMetaQuark;

  // Classic support via NvDsMsgApi
  gpointer libHandle;
  gchar *configFile;
  gchar *protoLib;
  gchar *connStr;
  gchar *topic;
  guint compId;
  guint sleepTime;
  NvMsgBrokerClientHandle connHandle;
  /** Identifies from input cap capability if the incoming data
   * is video/audio */
  gboolean is_video;
  GMutex flowLock;
  GCond flowCond;
  GThread *doWorkThread;
  gboolean isRunning;
  gboolean asyncSend;
  gint pendingCbCount;
  NvDsMsgApiErrorType lastError;
  nvds_msgapi_connect_ptr nvds_msgapi_connect;
  nvds_msgapi_send_ptr nvds_msgapi_send;
  nvds_msgapi_send_async_ptr nvds_msgapi_send_async;
  nvds_msgapi_do_work_ptr nvds_msgapi_do_work;
  nvds_msgapi_disconnect_ptr nvds_msgapi_disconnect;

  // New experimental support via NvMsgBroker lib
  gboolean newAPI;
  NvMsgBrokerClientHandle newConnHandle;
  NvMsgBrokerErrorType newLastError;
};

struct _GstNvMsgBrokerClass
{
  GstBaseSinkClass parent_class;
};

GType gst_nvmsgbroker_get_type (void);

G_END_DECLS

#endif
