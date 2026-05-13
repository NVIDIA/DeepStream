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
