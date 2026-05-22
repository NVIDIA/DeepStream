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

#ifndef __NVGSTDS_C2D_MSG_UTIL_H__
#define __NVGSTDS_C2D_MSG_UTIL_H__

#include <glib.h>
#include "deepstream_c2d_msg.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
  NVDS_C2D_MSG_SR_START,
  NVDS_C2D_MSG_SR_STOP
} NvDsC2DMsgType;

typedef struct NvDsC2DMsg {
  NvDsC2DMsgType type;
  gpointer message;
  guint msgSize;
} NvDsC2DMsg;

typedef struct NvDsC2DMsgSR {
  gchar *sensorStr;
  gint startTime;
  guint duration;
} NvDsC2DMsgSR;

NvDsC2DMsg* nvds_c2d_parse_cloud_message (gpointer data, guint size);

void nvds_c2d_release_message (NvDsC2DMsg *msg);

gboolean nvds_c2d_parse_sensor (NvDsC2DContext *ctx, const gchar *file);

#ifdef __cplusplus
}
#endif
#endif
