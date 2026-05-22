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

#ifndef __NVGSTDS_PERF_H__
#define __NVGSTDS_PERF_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "deepstream_config.h"

typedef struct
{
  guint source_id;
  gchar const* uri;
  gchar const* sensor_id;
  gchar const* sensor_name;
}NvDsFPSSensorInfo;

typedef struct
{
  guint source_id;
  char *stream_name;
  gchar const* sensor_id;
  gchar const* sensor_name;
} NvDsAppSourceDetail;

typedef struct
{
  gdouble fps[MAX_SOURCE_BINS];
  gdouble fps_avg[MAX_SOURCE_BINS];
  guint num_instances;
  NvDsAppSourceDetail source_detail[MAX_SOURCE_BINS];
  guint active_source_size;
  gboolean stream_name_display;
  gboolean use_nvmultiurisrcbin;
} NvDsAppPerfStruct;

typedef void (*perf_callback) (gpointer ctx, NvDsAppPerfStruct * str);

typedef struct
{
  guint buffer_cnt;
  guint total_buffer_cnt;
  struct timeval total_fps_time;
  struct timeval start_fps_time;
  struct timeval last_fps_time;
  struct timeval last_sample_fps_time;
} NvDsInstancePerfStruct;

typedef struct
{
  gulong measurement_interval_ms;
  gulong perf_measurement_timeout_id;
  guint num_instances;
  gboolean stop;
  gpointer context;
  GMutex struct_lock;
  perf_callback callback;
  GstPad *sink_bin_pad;
  gulong fps_measure_probe_id;
  NvDsInstancePerfStruct instance_str[MAX_SOURCE_BINS];
  guint dewarper_surfaces_per_frame;
  GHashTable *FPSInfoHash;
  gboolean stream_name_display;
  gboolean use_nvmultiurisrcbin;
} NvDsAppPerfStructInt;

gboolean enable_perf_measurement (NvDsAppPerfStructInt *str,
    GstPad *sink_bin_pad, guint num_sources, gulong interval_sec,
    guint num_surfaces_per_frame, perf_callback callback);

void pause_perf_measurement (NvDsAppPerfStructInt *str);
void resume_perf_measurement (NvDsAppPerfStructInt *str);

#ifdef __cplusplus
}
#endif

#endif
