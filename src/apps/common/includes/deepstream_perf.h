/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
