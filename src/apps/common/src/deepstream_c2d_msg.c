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

#include <dlfcn.h>
#include <stdlib.h>
#include "deepstream_c2d_msg.h"
#include "deepstream_c2d_msg_util.h"
#include "deepstream_common.h"
#include "deepstream_sources.h"
#include "gst-nvdssr.h"


static void
connect_cb (NvMsgBrokerClientHandle h_ptr, NvMsgBrokerErrorType status)
{

}

static void
subscribe_cb (NvMsgBrokerErrorType flag, void *msg, int msg_len, char *topic,
    void *uData)
{
  NvDsC2DMsg *parsedMsg = NULL;

  if (flag == NV_MSGBROKER_API_ERR) {
    NVGSTDS_ERR_MSG_V ("Error in consuming message.");
  } else {
    GST_DEBUG ("Consuming message, on topic[%s]. Payload =%.*s\n\n", topic,
        msg_len, (char *) msg);
  }

  if (uData) {
    NvDsC2DContext *ctx = (NvDsC2DContext *) uData;

    if (ctx->subscribeCb) {
      return ctx->subscribeCb (flag, msg, msg_len, topic, ctx->uData);
    } else {
      NvDsSrcParentBin *pBin = (NvDsSrcParentBin *) ctx->uData;
      if (!pBin) {
        NVGSTDS_WARN_MSG_V ("Null user data");
        return;
      }
      parsedMsg = nvds_c2d_parse_cloud_message (msg, msg_len);
      if (parsedMsg == NULL) {
        NVGSTDS_WARN_MSG_V ("error in message parsing \n");
        return;
      }

      if (parsedMsg->type == NVDS_C2D_MSG_SR_START ||
          parsedMsg->type == NVDS_C2D_MSG_SR_STOP) {

        NvDsSRSessionId sessId = 0;
        gint sensorId;
        NvDsSRContext *srCtx = NULL;
        NvDsC2DMsgSR *msgSR = (NvDsC2DMsgSR *) parsedMsg->message;

        if (ctx->hashMap) {
          gpointer keyVal =
              g_hash_table_lookup (ctx->hashMap, msgSR->sensorStr);
          if (keyVal) {
            sensorId = *(int *) keyVal;
          } else {
            NVGSTDS_WARN_MSG_V ("%s: Sensor id not found", msgSR->sensorStr);
            goto error;
          }
        } else {
          sensorId = atoi (msgSR->sensorStr);
        }

        srCtx = (NvDsSRContext *) pBin->sub_bins[sensorId].recordCtx;
        if (!srCtx) {
          NVGSTDS_WARN_MSG_V ("Null SR context handle.");
          goto error;
        }

        if (parsedMsg->type == NVDS_C2D_MSG_SR_START) {
          NvDsSRStart (srCtx, &sessId, msgSR->startTime, msgSR->duration, NULL);
        } else {
          NvDsSRStop (srCtx, sessId);
        }
      } else {
        NVGSTDS_WARN_MSG_V ("Unknown message type.");
      }
    }
  }

error:
  if (parsedMsg) {
    nvds_c2d_release_message (parsedMsg);
    parsedMsg = NULL;
  }
  return;
}

NvDsC2DContext *
start_cloud_to_device_messaging (NvDsMsgConsumerConfig * config,
    nv_msgbroker_subscribe_cb_t subscribeCb, void *uData)
{
  NvDsC2DContext *ctx = NULL;
  gchar **topicList = NULL;
  gint i, numTopic;

  if (!config->conn_str || !config->proto_lib || !config->topicList) {
    NVGSTDS_ERR_MSG_V ("NULL parameters");
    return NULL;
  }

  ctx = g_new0 (NvDsC2DContext, 1);
  ctx->protoLib = g_strdup (config->proto_lib);
  ctx->connStr = g_strdup (config->conn_str);
  ctx->configFile = g_strdup (config->config_file_path);

  if (subscribeCb)
    ctx->subscribeCb = subscribeCb;

  if (uData)
    ctx->uData = uData;

  if (config->sensor_list_file) {
    ctx->hashMap =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
    if (!nvds_c2d_parse_sensor (ctx, config->sensor_list_file)) {
      NVGSTDS_ERR_MSG_V ("Failed to parse sensor list file");
      goto error;
    }
  }

  ctx->connHandle = nv_msgbroker_connect (ctx->connStr,
      ctx->protoLib, connect_cb, ctx->configFile);
  if (!ctx->connHandle) {
    NVGSTDS_ERR_MSG_V ("Failed to connect to broker.");
    goto error;
  }

  numTopic = config->topicList->len;
  topicList = g_new0 (gchar *, numTopic);
  for (i = 0; i < numTopic; i++) {
    topicList[i] = (gchar *) g_ptr_array_index (config->topicList, i);
  }

  if (nv_msgbroker_subscribe (ctx->connHandle, topicList, numTopic,
          subscribe_cb, (gpointer) ctx) != NV_MSGBROKER_API_OK) {
    NVGSTDS_ERR_MSG_V ("Subscription to topic[s] failed\n");
    goto error;
  }

  g_free (topicList);
  topicList = NULL;

  return ctx;

error:
  if (ctx) {
    if (ctx->hashMap) {
      g_hash_table_unref (ctx->hashMap);
      ctx->hashMap = NULL;
    }

    g_free (ctx->configFile);
    g_free (ctx->connStr);
    g_free (ctx->protoLib);
    g_free (ctx);
    ctx = NULL;
  }

  if (topicList)
    g_free (topicList);

  return ctx;
}

gboolean
stop_cloud_to_device_messaging (NvDsC2DContext * ctx)
{
  NvMsgBrokerErrorType err;
  gboolean ret = TRUE;

  g_return_val_if_fail (ctx, FALSE);

  err = nv_msgbroker_disconnect (ctx->connHandle);
  if (err != NV_MSGBROKER_API_OK) {
    NVGSTDS_ERR_MSG_V ("error(%d) in disconnect\n", err);
    ret = FALSE;
  }
  ctx->connHandle = NULL;

  if (ctx->hashMap) {
    g_hash_table_unref (ctx->hashMap);
    ctx->hashMap = NULL;
  }

  g_free (ctx->configFile);
  g_free (ctx->connStr);
  g_free (ctx->protoLib);
  g_free (ctx);
  return ret;
}
