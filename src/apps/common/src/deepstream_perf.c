/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "gstnvdsmeta.h"
#include "deepstream_perf.h"

#define TIMESPEC_DIFF_USEC(timespec1, timespec2) \
    (timespec1.tv_sec - timespec2.tv_sec) * 1000000.0 + \
    (timespec1.tv_nsec - timespec2.tv_nsec) / 1000.0

/**
 * Buffer probe function on sink element.
 */
static GstPadProbeReturn
sink_bin_buf_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsAppPerfStructInt *str = (NvDsAppPerfStructInt *) u_data;
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  if (!batch_meta)
    return GST_PAD_PROBE_OK;

  if (!str->stop) {
    g_mutex_lock (&str->struct_lock);
    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame;
        l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      NvDsInstancePerfStruct *str1 = &str->instance_str[frame_meta->pad_index];
      gettimeofday (&str1->last_fps_time, NULL);
      if (str1->start_fps_time.tv_sec == 0 && str1->start_fps_time.tv_usec == 0) {
        str1->start_fps_time = str1->last_fps_time;
      } else {
        str1->buffer_cnt++;
      }
    }
    g_mutex_unlock (&str->struct_lock);
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
perf_measurement_callback (gpointer data)
{
  NvDsAppPerfStructInt *str = (NvDsAppPerfStructInt *) data;
  guint buffer_cnt[MAX_SOURCE_BINS];
  NvDsAppPerfStruct perf_struct;
  struct timeval current_fps_time;
  guint i;
  g_mutex_lock (&str->struct_lock);
  if (str->stop) {
    g_mutex_unlock (&str->struct_lock);
    return FALSE;
  }
  perf_struct.use_nvmultiurisrcbin = str->use_nvmultiurisrcbin;
  perf_struct.stream_name_display = str->stream_name_display;

  if (!str->use_nvmultiurisrcbin) {
    for (i = 0; i < str->num_instances; i++) {
      buffer_cnt[i] =
        str->instance_str[i].buffer_cnt / str->dewarper_surfaces_per_frame;
      str->instance_str[i].buffer_cnt = 0;
    }
  } else {
    GList *active_source_id_list = g_hash_table_get_keys(str->FPSInfoHash);
    GList *temp = active_source_id_list;
    for (guint i = 0; i < g_hash_table_size(str->FPSInfoHash); i++) {
          (perf_struct.source_detail[i]).source_id= GPOINTER_TO_UINT(temp->data);
	  NvDsFPSSensorInfo* sensorInfo = (NvDsFPSSensorInfo*)g_hash_table_lookup(str->FPSInfoHash,
            GUINT_TO_POINTER((perf_struct.source_detail[i]).source_id));
          (perf_struct.source_detail[i]).stream_name= (gchar*)sensorInfo->uri;
          (perf_struct.source_detail[i]).sensor_id= (gchar*)sensorInfo->sensor_id;
          (perf_struct.source_detail[i]).sensor_name= (gchar*)sensorInfo->sensor_name;
          temp=temp->next;
    }

    if (temp)
      g_list_free(temp);

    if (active_source_id_list)
      g_list_free(active_source_id_list);
    perf_struct.active_source_size = g_hash_table_size(str->FPSInfoHash);

    for (guint j = 0; j < g_hash_table_size(str->FPSInfoHash); j++){
      i = perf_struct.source_detail[j].source_id;
      buffer_cnt[i] =
        str->instance_str[i].buffer_cnt / str->dewarper_surfaces_per_frame;
      str->instance_str[i].buffer_cnt = 0;
    }
  }

  perf_struct.num_instances = str->num_instances;
  gettimeofday (&current_fps_time, NULL);
  if (!str->use_nvmultiurisrcbin) {
   for (i = 0; i < str->num_instances; i++) {
    NvDsInstancePerfStruct *str1 = &str->instance_str[i];
    gdouble time1 =
        (str1->total_fps_time.tv_sec +
        str1->total_fps_time.tv_usec / 1000000.0) +
        (current_fps_time.tv_sec + current_fps_time.tv_usec / 1000000.0) -
        (str1->start_fps_time.tv_sec +
        str1->start_fps_time.tv_usec / 1000000.0);

        str1->total_buffer_cnt += buffer_cnt[i];

#if 0
    gdouble time2;

    if (str1->last_sample_fps_time.tv_sec == 0 &&
        str1->last_sample_fps_time.tv_usec == 0) {
      time2 =
          (str1->last_fps_time.tv_sec +
          str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->start_fps_time.tv_sec +
          str1->start_fps_time.tv_usec / 1000000.0);
    } else {
      time2 =
          (str1->last_fps_time.tv_sec +
          str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->last_sample_fps_time.tv_sec +
          str1->last_sample_fps_time.tv_usec / 1000000.0);
    }
    perf_struct.fps[i] = buffer_cnt[i] / time2;
#else
    perf_struct.fps[i] = buffer_cnt[i] / ((str->measurement_interval_ms)/1000.0);
#endif

    if (isnan (perf_struct.fps[i]))
      perf_struct.fps[i] = 0;

    perf_struct.fps_avg[i] = str1->total_buffer_cnt / time1;
    if (isnan (perf_struct.fps_avg[i]))
      perf_struct.fps_avg[i] = 0;

    str1->last_sample_fps_time = str1->last_fps_time;
   }
  } else {
    for (guint j = 0; j < g_hash_table_size(str->FPSInfoHash); j++){
      i = perf_struct.source_detail[j].source_id;
      NvDsInstancePerfStruct *str1 = &str->instance_str[i];
      gdouble time1 =
        (str1->total_fps_time.tv_sec +
        str1->total_fps_time.tv_usec / 1000000.0) +
        (current_fps_time.tv_sec + current_fps_time.tv_usec / 1000000.0) -
        (str1->start_fps_time.tv_sec +
        str1->start_fps_time.tv_usec / 1000000.0);

        str1->total_buffer_cnt += buffer_cnt[i];

#if 0
      gdouble time2;

      if (str1->last_sample_fps_time.tv_sec == 0 &&
        str1->last_sample_fps_time.tv_usec == 0) {
        time2 =
          (str1->last_fps_time.tv_sec +
          str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->start_fps_time.tv_sec +
          str1->start_fps_time.tv_usec / 1000000.0);
      } else {
        time2 =
          (str1->last_fps_time.tv_sec +
          str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->last_sample_fps_time.tv_sec +
          str1->last_sample_fps_time.tv_usec / 1000000.0);
      }
      perf_struct.fps[i] = buffer_cnt[i] / time2;
#else
      perf_struct.fps[i] = buffer_cnt[i] / ((str->measurement_interval_ms)/1000.0);
#endif

      if (isnan (perf_struct.fps[i]))
        perf_struct.fps[i] = 0;

      perf_struct.fps_avg[i] = str1->total_buffer_cnt / time1;
      if (isnan (perf_struct.fps_avg[i]))
        perf_struct.fps_avg[i] = 0;

      str1->last_sample_fps_time = str1->last_fps_time;
    }

  }
  g_mutex_unlock (&str->struct_lock);

  str->callback (str->context, &perf_struct);

  return TRUE;
}

void
pause_perf_measurement (NvDsAppPerfStructInt * str)
{
  guint i;

  g_mutex_lock (&str->struct_lock);
  str->stop = TRUE;

  for (i = 0; i < str->num_instances; i++) {
    NvDsInstancePerfStruct *str1 = &str->instance_str[i];
    str1->total_fps_time.tv_sec +=
        str1->last_fps_time.tv_sec - str1->start_fps_time.tv_sec;
    str1->total_fps_time.tv_usec +=
        str1->last_fps_time.tv_usec - str1->start_fps_time.tv_usec;
    if (str1->total_fps_time.tv_usec < 0) {
      str1->total_fps_time.tv_sec--;
      str1->total_fps_time.tv_usec += 1000000;
    }
    str1->start_fps_time.tv_sec = str1->start_fps_time.tv_usec = 0;
  }

  g_mutex_unlock (&str->struct_lock);
}

void
resume_perf_measurement (NvDsAppPerfStructInt * str)
{
  guint i;

  g_mutex_lock (&str->struct_lock);
  if (!str->stop) {
    g_mutex_unlock (&str->struct_lock);
    return;
  }

  str->stop = FALSE;

  for (i = 0; i < str->num_instances; i++) {
    str->instance_str[i].buffer_cnt = 0;
  }

  if (str->perf_measurement_timeout_id) {
    g_source_remove_by_user_data(str);
    str->perf_measurement_timeout_id = 0;
  }

  if(!str->perf_measurement_timeout_id){
    str->perf_measurement_timeout_id =
      g_timeout_add (str->measurement_interval_ms, perf_measurement_callback,
      str);
  }

  g_mutex_unlock (&str->struct_lock);
}

gboolean
enable_perf_measurement (NvDsAppPerfStructInt * str,
    GstPad * sink_bin_pad, guint num_sources,
    gulong interval_sec, guint num_surfaces_per_frame, perf_callback callback)
{
  guint i;

  if (!callback) {
    return FALSE;
  }

  str->num_instances = num_sources;

  str->measurement_interval_ms = interval_sec * 1000;
  str->callback = callback;
  str->stop = TRUE;

  if (num_surfaces_per_frame) {
    str->dewarper_surfaces_per_frame = num_surfaces_per_frame;
  } else {
    str->dewarper_surfaces_per_frame = 1;
  }

  for (i = 0; i < num_sources; i++) {
    str->instance_str[i].buffer_cnt = 0;
  }
  str->sink_bin_pad = sink_bin_pad;
  str->fps_measure_probe_id =
      gst_pad_add_probe (sink_bin_pad, GST_PAD_PROBE_TYPE_BUFFER,
      sink_bin_buf_probe, str, NULL);

  resume_perf_measurement (str);

  return TRUE;
}
