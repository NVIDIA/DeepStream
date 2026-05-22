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

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <dlfcn.h>
#include "gstnvmsgbroker.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvdsmeta.h"
#include <string.h>
#include <iostream>
#include <unordered_map>

using namespace std;

GST_DEBUG_CATEGORY_STATIC (gst_nvmsgbroker_debug_category);
#define GST_CAT_DEFAULT gst_nvmsgbroker_debug_category

//Store NvMsgBroker context handle Status
typedef struct
{
  bool connection_alive;
  bool broker_disconnect;
} NvMsgBrokerHandleStatus;

//Map to store the state of NvMsgBrokerClientHandle
unordered_map < NvMsgBrokerClientHandle,
    NvMsgBrokerHandleStatus * >NvMsgBrokerHandleMap;

//Lock to change NvMsgBrokerClientHandle state
GMutex NvMsgBrokerClientHandleLock;

static void gst_nvmsgbroker_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_nvmsgbroker_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_nvmsgbroker_finalize (GObject * object);

static gboolean gst_nvmsgbroker_set_caps (GstBaseSink * sink, GstCaps * caps);
static gboolean gst_nvmsgbroker_start (GstBaseSink * sink);
static gboolean gst_nvmsgbroker_stop (GstBaseSink * sink);
static GstFlowReturn gst_nvmsgbroker_render (GstBaseSink * sink,
    GstBuffer * buffer);

static gboolean
gst_nvmsgbroker_is_video (GstCaps * caps)
{
  GstStructure *caps_str = gst_caps_get_structure (caps, 0);
  const gchar *mimetype = gst_structure_get_name (caps_str);

  if (strcmp (mimetype, "video/x-raw") == 0) {
    return TRUE;
  }
  return FALSE;
}

/**********************************************
 * Experimental functions to support libnvds_msgbroker
 */
static void nvmsgbroker_connect_callback (NvMsgBrokerClientHandle h_ptr,
    NvMsgBrokerErrorType status);
static void nvmsgbroker_send_callback (void *data, NvMsgBrokerErrorType status);
static gboolean new_gst_nvmsgbroker_start (GstBaseSink * sink);
static gboolean new_gst_nvmsgbroker_stop (GstBaseSink * sink);
static GstFlowReturn new_gst_nvmsgbroker_render (GstBaseSink * sink,
    GstBuffer * buf);
/**********************************************
 * Legacy functions to support NvDsMsgApi
 */
static gboolean legacy_gst_nvmsgbroker_start (GstBaseSink * sink);
static gboolean legacy_gst_nvmsgbroker_stop (GstBaseSink * sink);
static GstFlowReturn legacy_gst_nvmsgbroker_render (GstBaseSink * sink,
    GstBuffer * buf);

static void
nvds_msgapi_connect_callback (NvDsMsgApiHandle h_ptr,
    NvDsMsgApiEventType ds_evt)
{

}

static void
nvds_msgapi_send_callback (void *data, NvDsMsgApiErrorType status)
{
  GstNvMsgBroker *self = (GstNvMsgBroker *) data;

  g_mutex_lock (&self->flowLock);
  self->pendingCbCount--;
  self->lastError = status;

  if (status != NVDS_MSGAPI_OK) {
    GST_ERROR_OBJECT (self, "error(%d) in sending data", status);
  }
  g_mutex_unlock (&self->flowLock);
}

enum
{
  PROP_0,
  PROP_CONNECTION_STRING,
  PROP_CONFIG_FILE,
  PROP_PROTOCOL_LIBRARY,
  PROP_TOPIC,
  PROP_COMPONENT_ID,
  PROP_SLEEP_TIME,
  PROP_NEW_API
};
#define DEFAULT_USE_NEW_API FALSE
#define DEFAULT_SLEEP_TIME 0

static GstStaticPadTemplate gst_nvmsgbroker_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

G_DEFINE_TYPE_WITH_CODE (GstNvMsgBroker, gst_nvmsgbroker, GST_TYPE_BASE_SINK,
    GST_DEBUG_CATEGORY_INIT (gst_nvmsgbroker_debug_category, "nvmsgbroker", 0,
        "debug category for nvmsgbroker element"));

static gpointer
gst_nvmsgbroker_do_work (gpointer data)
{
  GstNvMsgBroker *self = (GstNvMsgBroker *) data;
  while (self->isRunning) {
    // Don't wait for new messages in nvmsgbroker in case of MQTT--MQTT relies on continuous calls to mosquitto_loop()
    // as called by do_work() to keep connection alive. The call blocks until a messages is received or until
    // timeout is met as set in the mqtt config
    if (!g_strrstr(self->protoLib, "libnvds_mqtt_proto.so")) {
      g_mutex_lock (&self->flowLock);
      while (self->isRunning && self->pendingCbCount <= 0) {
        g_cond_wait(&self->flowCond, &self->flowLock);
      }
      g_mutex_unlock (&self->flowLock);
    }

    if (!self->isRunning) {
      return NULL;
    }
    self->nvds_msgapi_do_work (self->connHandle);
    g_usleep (self->sleepTime * 1000);
  }
  return self;
}

static void
gst_nvmsgbroker_class_init (GstNvMsgBrokerClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_nvmsgbroker_sink_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "Message Broker", "Sink/Metadata",
      "Sends payload metadata to remote server",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_nvmsgbroker_set_property;
  gobject_class->get_property = gst_nvmsgbroker_get_property;
  gobject_class->finalize = gst_nvmsgbroker_finalize;
  base_sink_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvmsgbroker_set_caps);
  base_sink_class->start = GST_DEBUG_FUNCPTR (gst_nvmsgbroker_start);
  base_sink_class->stop = GST_DEBUG_FUNCPTR (gst_nvmsgbroker_stop);
  base_sink_class->render = GST_DEBUG_FUNCPTR (gst_nvmsgbroker_render);

  g_object_class_install_property (gobject_class, PROP_PROTOCOL_LIBRARY,
      g_param_spec_string ("proto-lib", "Protocol library name",
          "Name of protocol adaptor library with absolute path.",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONNECTION_STRING,
      g_param_spec_string ("conn-str", "connection string",
          "connection string of backend server (e.g. foo.bar.com;80;dsapp1)",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string ("config", "configuration file name",
          "Name of configuration file with absolute path.",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_TOPIC,
      g_param_spec_string ("topic", "topic name", "Name of the message topic",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_COMPONENT_ID,
      g_param_spec_uint ("comp-id", "Component Id ",
          "By default this element operates on all NvDsPayload type meta\n"
          "\t\t\tBut it can be restricted to a specific NvDsPayload meta\n"
          "\t\t\thaving this component id",
          0, G_MAXUINT, 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NEW_API,
      g_param_spec_boolean ("new-api", "Use new libnvds_msgbroker API",
          "Use new libnvds_msgbroker API",
          DEFAULT_USE_NEW_API, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SLEEP_TIME,
      g_param_spec_uint ("sleep-time", "Sleep time ",
          "Sleep time between consecutive do_work calls in millisecond.", 0,
          G_MAXUINT, DEFAULT_SLEEP_TIME,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

}

static void
gst_nvmsgbroker_init (GstNvMsgBroker * self)
{
  self->dsMetaQuark = g_quark_from_static_string (NVDS_META_STRING);
  self->libHandle = NULL;
  self->connHandle = NULL;
  self->connStr = NULL;
  self->topic = NULL;
  self->protoLib = NULL;
  self->configFile = NULL;
  self->isRunning = FALSE;
  self->pendingCbCount = 0;
  self->asyncSend = TRUE;
  self->lastError = NVDS_MSGAPI_OK;
  self->compId = 0;
  self->is_video = FALSE;
  self->newAPI = DEFAULT_USE_NEW_API;
  self->newConnHandle = NULL;
  self->newLastError = NV_MSGBROKER_API_OK;
  self->sleepTime = DEFAULT_SLEEP_TIME;

  g_mutex_init (&self->flowLock);
  g_cond_init (&self->flowCond);
}

void
gst_nvmsgbroker_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (object);

  GST_DEBUG_OBJECT (self, "set_property");

  switch (property_id) {
    case PROP_CONFIG_FILE:
      if (self->configFile)
        g_free (self->configFile);
      self->configFile = (gchar *) g_value_dup_string (value);
      break;
    case PROP_CONNECTION_STRING:
      if (self->connStr)
        g_free (self->connStr);
      self->connStr = (gchar *) g_value_dup_string (value);
      break;
    case PROP_PROTOCOL_LIBRARY:
      if (self->protoLib)
        g_free (self->protoLib);
      self->protoLib = (gchar *) g_value_dup_string (value);
      break;
    case PROP_TOPIC:
      if (self->topic)
        g_free (self->topic);
      self->topic = (gchar *) g_value_dup_string (value);
      g_strstrip (self->topic);
      if (!g_strcmp0 (self->topic, "")) {
        g_free (self->topic);
        self->topic = NULL;
      }
      break;
    case PROP_COMPONENT_ID:
      self->compId = g_value_get_uint (value);
      break;
    case PROP_NEW_API:
      self->newAPI = g_value_get_boolean (value);
      break;
    case PROP_SLEEP_TIME:
      self->sleepTime = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvmsgbroker_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (object);

  GST_DEBUG_OBJECT (self, "get_property");

  switch (property_id) {
    case PROP_CONFIG_FILE:
      g_value_set_string (value, self->configFile);
      break;
    case PROP_CONNECTION_STRING:
      g_value_set_string (value, self->connStr);
      break;
    case PROP_PROTOCOL_LIBRARY:
      g_value_set_string (value, self->protoLib);
      break;
    case PROP_TOPIC:
      g_value_set_string (value, self->topic);
      break;
    case PROP_COMPONENT_ID:
      g_value_set_uint (value, self->compId);
      break;
    case PROP_NEW_API:
      g_value_set_boolean (value, self->newAPI);
      break;
    case PROP_SLEEP_TIME:
      g_value_set_uint (value, self->sleepTime);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvmsgbroker_finalize (GObject * object)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (object);

  GST_DEBUG_OBJECT (self, "finalize");

  if (self->configFile)
    g_free (self->configFile);

  if (self->connStr)
    g_free (self->connStr);

  if (self->topic)
    g_free (self->topic);

  if (self->protoLib)
    g_free (self->protoLib);

  g_mutex_clear (&self->flowLock);
  g_cond_clear (&self->flowCond);

  G_OBJECT_CLASS (gst_nvmsgbroker_parent_class)->finalize (object);
}

static gboolean
gst_nvmsgbroker_set_caps (GstBaseSink * sink, GstCaps * caps)
{
  GstNvMsgBroker *nvmsgbroker = GST_NVMSGBROKER (sink);

  GST_DEBUG_OBJECT (nvmsgbroker, "set_caps");

  nvmsgbroker->is_video = gst_nvmsgbroker_is_video (caps);

  return TRUE;
}

static gboolean
gst_nvmsgbroker_start (GstBaseSink * sink)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);

  if (self->protoLib && g_strrstr(self->protoLib, "libnvds_azure_proto.so") && self->sleepTime < 10) {
    g_print("Warning: sleep-time value < 10ms fails for azure protocol adaptor. Setting to 10ms. You may require even higher sleep time depending on your service tier.\n");
    self->sleepTime = 10;
  }

  if (self->newAPI)
    return new_gst_nvmsgbroker_start (sink);
  else
    return legacy_gst_nvmsgbroker_start (sink);
}

static gboolean
legacy_gst_nvmsgbroker_start (GstBaseSink * sink)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);
  gchar *error;
  gchar *temp = NULL;

  GST_DEBUG_OBJECT (self, "start");

  if (!self->protoLib) {
    GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND, (NULL),
        ("No protocol adaptor library provided"));
    return FALSE;
  }

  temp = g_strrstr (self->connStr, ";");
  if (temp)
    if (!self->topic)
      self->topic = g_strdup (temp + 1);

  self->libHandle = dlopen (self->protoLib, RTLD_LAZY);
  if (!self->libHandle) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT, (NULL),
        ("unable to open shared library"));
    return FALSE;
  }

  dlerror ();                   /* Clear any existing error */

  self->nvds_msgapi_connect =
      (nvds_msgapi_connect_ptr) dlsym (self->libHandle, "nvds_msgapi_connect");
  self->nvds_msgapi_send =
      (nvds_msgapi_send_ptr) dlsym (self->libHandle, "nvds_msgapi_send");
  self->nvds_msgapi_disconnect =
      (nvds_msgapi_disconnect_ptr) dlsym (self->libHandle,
      "nvds_msgapi_disconnect");
  if (self->asyncSend) {
    self->nvds_msgapi_send_async =
        (nvds_msgapi_send_async_ptr) dlsym (self->libHandle,
        "nvds_msgapi_send_async");
    self->nvds_msgapi_do_work =
        (nvds_msgapi_do_work_ptr) dlsym (self->libHandle,
        "nvds_msgapi_do_work");
  }

  if ((error = dlerror ()) != NULL) {
    GST_ELEMENT_ERROR (self, LIBRARY, FAILED, (NULL), ("%s", error));
    return FALSE;
  }
  self->connHandle = self->nvds_msgapi_connect (self->connStr,
      (nvds_msgapi_connect_cb_t) nvds_msgapi_connect_callback,
      self->configFile);
  if (!self->connHandle) {
    if (self->libHandle) {
      dlclose (self->libHandle);
      self->libHandle = NULL;
    }
    GST_ELEMENT_ERROR (self, LIBRARY, SETTINGS, (NULL),
        ("unable to connect to broker library"));
    return FALSE;
  }

  self->isRunning = TRUE;
  if (self->asyncSend) {
    self->doWorkThread = g_thread_new ("doWork_thread",
        gst_nvmsgbroker_do_work, (gpointer) self);
  }

  return TRUE;
}

static gboolean
gst_nvmsgbroker_stop (GstBaseSink * sink)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);

  if (self->newAPI)
    return new_gst_nvmsgbroker_stop (sink);
  else
    return legacy_gst_nvmsgbroker_stop (sink);
}

static gboolean
legacy_gst_nvmsgbroker_stop (GstBaseSink * sink)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);
  NvDsMsgApiErrorType err;

  GST_DEBUG_OBJECT (self, "stop");

  self->isRunning = FALSE;

  if (self->asyncSend) {
    g_cond_signal (&self->flowCond);
    g_thread_join (self->doWorkThread);
  }

  if (self->nvds_msgapi_disconnect) {
    err = self->nvds_msgapi_disconnect (self->connHandle);
    if (err != NVDS_MSGAPI_OK)
      GST_ERROR_OBJECT (self, "error(%d) in disconnect", err);
    self->connHandle = NULL;
  }

  if (self->libHandle) {
    dlclose (self->libHandle);
    self->libHandle = NULL;
  }
  return TRUE;
}

static GstFlowReturn
gst_nvmsgbroker_render (GstBaseSink * sink, GstBuffer * buf)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);

  if (self->newAPI)
    return new_gst_nvmsgbroker_render (sink, buf);
  else
    return legacy_gst_nvmsgbroker_render (sink, buf);
}

static GstFlowReturn
legacy_gst_nvmsgbroker_render (GstBaseSink * sink, GstBuffer * buf)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);
  NvDsMeta *meta = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  GstMeta *gstMeta = NULL;
  gpointer state = NULL;
  NvDsMsgApiErrorType err;
  NvDsPayload *payload;

  GST_DEBUG_OBJECT (self, "render");

  while ((gstMeta = gst_buffer_iterate_meta (buf, &state))) {
    if (gst_meta_api_type_has_tag (gstMeta->info->api, self->dsMetaQuark)) {
      meta = (NvDsMeta *) gstMeta;
      if (meta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *) meta->meta_data;
        break;
      }
    }
  }

  if (batch_meta) {
    NvDsMetaList *l = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *user_meta_list = NULL;
    void *frame_meta = NULL;
    NvDsUserMeta *user_meta = NULL;

    if(batch_meta->misc_batch_info[0]==NVDS_CUSTOM_MSG_SPARSE4D) {
      user_meta_list = batch_meta->batch_user_meta_list;
      for (l = user_meta_list; l; l = l->next) {
        user_meta = (NvDsUserMeta *) (l->data);

        if (user_meta && user_meta->base_meta.meta_type == NVDS_PAYLOAD_META) {
          payload = (NvDsPayload *) user_meta->user_meta_data;

          if (self->compId && payload->componentId != self->compId)
            continue;

          if (self->asyncSend) {
            g_mutex_lock (&self->flowLock);
            err = self->nvds_msgapi_send_async (self->connHandle, self->topic,
                (uint8_t *) payload->payload,
                payload->payloadSize, nvds_msgapi_send_callback, self);

            if (err != NVDS_MSGAPI_OK) {
              GST_ELEMENT_ERROR (self, LIBRARY, FAILED, (NULL),
                  ("failed to send the message. err(%d)", err));

              g_mutex_unlock (&self->flowLock);
              return GST_FLOW_ERROR;
            }
            self->pendingCbCount++;
            g_cond_signal (&self->flowCond);
            g_mutex_unlock (&self->flowLock);
          } else {
            err = self->nvds_msgapi_send (self->connHandle, self->topic,
                (uint8_t *) payload->payload, payload->payloadSize);

            if (err != NVDS_MSGAPI_OK) {
              GST_ELEMENT_ERROR (self, LIBRARY, FAILED, (NULL),
                  ("failed to send the message. err(%d)", err));

              return GST_FLOW_ERROR;
            }
          }
        }
      }
    }
    else {
      for (l_frame = batch_meta->frame_meta_list; l_frame;
        l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) l_frame->data;
        if (self->is_video) {
          user_meta_list = ((NvDsFrameMeta *) frame_meta)->frame_user_meta_list;
        } else {
          user_meta_list =
            ((NvDsAudioFrameMeta *) frame_meta)->frame_user_meta_list;
        }

        for (l = user_meta_list; l; l = l->next) {
          user_meta = (NvDsUserMeta *) (l->data);

          if (user_meta && user_meta->base_meta.meta_type == NVDS_PAYLOAD_META) {
            payload = (NvDsPayload *) user_meta->user_meta_data;

            if (self->compId && payload->componentId != self->compId)
              continue;

            if (self->asyncSend) {
              g_mutex_lock (&self->flowLock);
              err = self->nvds_msgapi_send_async (self->connHandle, self->topic,
                (uint8_t *) payload->payload,
                payload->payloadSize, nvds_msgapi_send_callback, self);

              if (err != NVDS_MSGAPI_OK) {
                GST_ELEMENT_ERROR (self, LIBRARY, FAILED, (NULL),
                  ("failed to send the message. err(%d)", err));

                g_mutex_unlock (&self->flowLock);
                return GST_FLOW_ERROR;
              }
              self->pendingCbCount++;
              g_cond_signal (&self->flowCond);
              g_mutex_unlock (&self->flowLock);
            } else {
                err = self->nvds_msgapi_send (self->connHandle, self->topic,
                (uint8_t *) payload->payload, payload->payloadSize);

              if (err != NVDS_MSGAPI_OK) {
                GST_ELEMENT_ERROR (self, LIBRARY, FAILED, (NULL),
                  ("failed to send the message. err(%d)", err));

                return GST_FLOW_ERROR;
              }
            }
          }
        }
      }
    }
  }
  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "nvmsgbroker", GST_RANK_PRIMARY,
      GST_TYPE_NVMSGBROKER);
}

/**********************************************
 * Experimental support for libnvds_msgbroker
 */

static void
nvmsgbroker_connect_callback (NvMsgBrokerClientHandle h_ptr,
    NvMsgBrokerErrorType status)
{
  g_mutex_lock (&NvMsgBrokerClientHandleLock);
  if (status == NV_MSGBROKER_API_OK)
    NvMsgBrokerHandleMap[h_ptr]->connection_alive = true;
  else {
    NvMsgBrokerHandleMap[h_ptr]->connection_alive = false;
    if (status == NV_MSGBROKER_API_ERR) {
      NvMsgBrokerHandleMap[h_ptr]->broker_disconnect = true;
    }
  }
  g_mutex_unlock (&NvMsgBrokerClientHandleLock);
}

static void
nvmsgbroker_send_callback (void *data, NvMsgBrokerErrorType status)
{
  GstNvMsgBroker *self = (GstNvMsgBroker *) data;

  self->newLastError = status;

  if (status != NV_MSGBROKER_API_OK) {
    GST_DEBUG_OBJECT (self,
        "gstmsgbroker send callback: error(%d) in sending data", status);
  }
}

static gboolean
new_gst_nvmsgbroker_start (GstBaseSink * sink)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);

  GST_DEBUG_OBJECT (self, "start");

  if (!self->protoLib) {
    GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND, (NULL),
        ("No protocol adaptor library provided"));
    return FALSE;
  }

  self->newConnHandle = nv_msgbroker_connect (self->connStr, self->protoLib,
      nvmsgbroker_connect_callback, self->configFile);
  if (!self->newConnHandle) {
    GST_ELEMENT_ERROR (self, LIBRARY, SETTINGS, (NULL),
        ("unable to connect to nvmsgbroker library"));
    return FALSE;
  }

  g_mutex_lock (&NvMsgBrokerClientHandleLock);
  if (!NvMsgBrokerHandleMap.count (self->newConnHandle))
    NvMsgBrokerHandleMap[self->newConnHandle] = new NvMsgBrokerHandleStatus {
    true, false};
  g_mutex_unlock (&NvMsgBrokerClientHandleLock);

  return TRUE;
}

static gboolean
new_gst_nvmsgbroker_stop (GstBaseSink * sink)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);
  NvMsgBrokerErrorType err;

  GST_DEBUG_OBJECT (self, "stop");

  g_mutex_lock (&NvMsgBrokerClientHandleLock);
  if (NvMsgBrokerHandleMap.count (self->newConnHandle)) {
    delete NvMsgBrokerHandleMap[self->newConnHandle];
    NvMsgBrokerHandleMap.erase (self->newConnHandle);
  }
  g_mutex_unlock (&NvMsgBrokerClientHandleLock);

  err = nv_msgbroker_disconnect (self->newConnHandle);
  if (err != NV_MSGBROKER_API_OK) {
    GST_ERROR_OBJECT (self, "error(%d) in disconnect", err);
    self->newConnHandle = NULL;
  }
  return TRUE;
}

static GstFlowReturn
new_gst_nvmsgbroker_render_user_meta_list (GstNvMsgBroker * self,
    NvDsMetaList * user_meta_list)
{
  NvMsgBrokerErrorType err;
  NvDsPayload *payload;
  NvDsMetaList *l = NULL;
  for (l = user_meta_list; l; l = l->next) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) (l->data);

    if (user_meta && user_meta->base_meta.meta_type == NVDS_PAYLOAD_META) {
      payload = (NvDsPayload *) user_meta->user_meta_data;

      if (self->compId && payload->componentId != self->compId)
        continue;

      if (self->newConnHandle
          && NvMsgBrokerHandleMap[self->newConnHandle]->connection_alive ==
          true) {
        NvMsgBrokerClientMsg msg;
        msg.topic = self->topic;
        msg.payload = (void *) payload->payload;
        msg.payload_len = payload->payloadSize;
        err = nv_msgbroker_send_async (self->newConnHandle, msg,
            nvmsgbroker_send_callback, self);

        if (err != NV_MSGBROKER_API_OK) {
          GST_ERROR_OBJECT (self,
              "gstmsgbroker send callback: error(%d) in sending data", err);
        }
      } else
        GST_ERROR_OBJECT (self,
            "gstmsgbroker send : connection not alive, message not published");

      if (self->newConnHandle
          && NvMsgBrokerHandleMap[self->newConnHandle]->broker_disconnect ==
          true) {
        //Disconnect the connection handle (ex: after max connection retry attempt, fatal errors)
        //Signal msgbroker disconnect
        GST_ELEMENT_ERROR (self, LIBRARY, FAILED, (NULL),
            ("disconnecting nvmsgbroker"));
        return GST_FLOW_ERROR;
      }
    }
  }
  return GST_FLOW_OK;
}

static GstFlowReturn
new_gst_nvmsgbroker_render (GstBaseSink * sink, GstBuffer * buf)
{
  GstNvMsgBroker *self = GST_NVMSGBROKER (sink);
  NvDsMeta *meta = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  GstMeta *gstMeta = NULL;
  gpointer state = NULL;
  GstFlowReturn ret = GST_FLOW_OK;

  GST_DEBUG_OBJECT (self, "render");

  while ((gstMeta = gst_buffer_iterate_meta (buf, &state))) {
    if (gst_meta_api_type_has_tag (gstMeta->info->api, self->dsMetaQuark)) {
      meta = (NvDsMeta *) gstMeta;
      if (meta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *) meta->meta_data;
        break;
      }
    }
  }

  if (batch_meta) {
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *user_meta_list = NULL;
    void *frame_meta = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame;
        l_frame = l_frame->next) {
      frame_meta = l_frame->data;
      if (self->is_video) {
        user_meta_list = ((NvDsFrameMeta *) frame_meta)->frame_user_meta_list;
      } else {
        user_meta_list =
            ((NvDsAudioFrameMeta *) frame_meta)->frame_user_meta_list;
      }
      ret = new_gst_nvmsgbroker_render_user_meta_list (self, user_meta_list);
      if (ret != GST_FLOW_OK) {
        return ret;
      }
    }

    ret =
        new_gst_nvmsgbroker_render_user_meta_list (self,
        batch_meta->batch_user_meta_list);

  }
  return ret;
}

#define PACKAGE "nvmsgbroker"

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_msgbroker,
    "Message broker",
    plugin_init, "9.0", "Proprietary", "NvMsgBroker", "http://nvidia.com")
