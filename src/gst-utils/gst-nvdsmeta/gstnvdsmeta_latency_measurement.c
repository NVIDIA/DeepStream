/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* localtime_r() requires POSIX.1-2001; needed for -std=c99 + glibc. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gstnvdsmeta.h"
#include "nvds_latency_meta.h"
#include "nvds_latency_meta_internal.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>


#define DUMMY_FRAME_NUM 0
#define WRITE_COMP_CSV_LINE(latency_metadata, batch_num,component_latency) \
  fprintf(g_comp_csv_file,"\n%d,%s,%s,source_%d,%d,%d,%d,%lf,%lf,%s,%lf",getpid()\
    ,latency_metadata->component_name,"full",latency_metadata->source_id\
    ,latency_metadata->pad_index,batch_num,DUMMY_FRAME_NUM,latency_metadata->in_system_timestamp\
    ,latency_metadata->out_system_timestamp,"latency",component_latency);

#define WRITE_SUB_COMP_CSV_LINE(latency_metadata, batch_num) \
    do { \
      for (unsigned int sub_comp_id = 0; sub_comp_id < latency_metadata->num_sub_comps; sub_comp_id++) \
      { \
        NvDsMetaSubCompLatency *subl = &latency_metadata->sub_comp_latencies[sub_comp_id]; \
        gchar *sub_comp_name = subl->sub_comp_name; \
        fprintf(g_comp_csv_file,"\n%d,%s,%s-%s,source_%d,%d,%d,%d,%lf,%lf,%s,%lf",getpid()\
          ,latency_metadata->component_name,latency_metadata->component_name,sub_comp_name,latency_metadata->source_id\
          ,latency_metadata->pad_index,batch_num,DUMMY_FRAME_NUM, \
          subl->in_system_timestamp \
          ,subl->out_system_timestamp,"latency", \
          subl->out_system_timestamp - subl->in_system_timestamp); \
      } \
    } while (0)

#define WRITE_FRAME_CSV_LINE(latency_info, batch_num,fps_val,src_cnt) \
    do { \
      for (unsigned int i = 0; i < src_cnt; i++) \
      { \
        fprintf(g_frame_csv_file,"\n%d,%d,source_%d,%d,%lf,%lf,%s,%lf",getpid(),batch_num,latency_info[i].source_id \
        ,latency_info[i].frame_num,latency_info[i].comp_in_timestamp,fps_val,"latency",latency_info[i].latency); \
      } \
    } while (0)

static FILE *g_comp_csv_file = NULL;
static FILE *g_frame_csv_file= NULL;

void __attribute__((constructor)) gstnvdsmeta_latency_measurement_int(void);

void __attribute__((constructor)) gstnvdsmeta_latency_measurement_int(void)
{
  const gchar *csv_env = NULL;
  csv_env = g_getenv("NVDS_PERF_CSV_PREFIX");
  if(csv_env != NULL)
  {
    char file_id[15];
    char comp_suffix[100];
    char comp_prefix[100];
    char frame_suffix[100];
    char frame_prefix[100];
    const char *file_prefix = csv_env;
    time_t t;
    struct tm tm_buf;
    struct tm *time_val;
    time (&t);
    time_val = localtime_r (&t, &tm_buf);

    sprintf(file_id,"_id_%d", getpid());

    strcpy(comp_prefix, file_prefix);
    strcat(comp_prefix, file_id);
    strcpy(frame_prefix, file_prefix);
    strcat(frame_prefix, file_id);

    strftime(comp_suffix, 100, "_component_%Y%m%d-%H%M%S.csv", time_val);
    strftime(frame_suffix, 100, "_frame_%Y%m%d-%H%M%S.csv", time_val);
    gchar *file_comp_name = g_strdup_printf("%s%s", comp_prefix,comp_suffix);
    gchar *file_frame_name = g_strdup_printf("%s%s", frame_prefix,frame_suffix);


    g_comp_csv_file = fopen(file_comp_name,"w");
    if(g_comp_csv_file == NULL)
    {
      g_print("WARNING :COMPONENT FILE CANNOT BE OPENED");
    }
    else
    {
      fprintf(g_comp_csv_file,"file_id,comp_name,stage,source_id,pad_index,batch_num,frame_num,in_system_timestamp,out_system_timestamp,measurement_type,measurement_val");
    }

    g_frame_csv_file = fopen(file_frame_name,"w");
    if(g_frame_csv_file == NULL)
    {
      g_print("WARNING :FRAME FILE CANNOT BE OPENED");
    }
    else
    {
      fprintf(g_frame_csv_file,"file_id,batch_num,source_id,frame_num,in_system_timestamp,fps,measurement_type,measurement_val");
    }
    g_free(file_comp_name);
    g_free(file_frame_name);
  }
}

void *nvds_set_latency_metadata_ptr()
{
  NvDsMetaCompLatency *latency_metadata = (NvDsMetaCompLatency*)
    g_malloc0(sizeof(NvDsMetaCompLatency));

  return (void *)latency_metadata;
}

gpointer nvds_copy_latency_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDsMetaCompLatency *src_user_metadata =
    (NvDsMetaCompLatency*)user_meta->user_meta_data;
  NvDsMetaCompLatency *dst_user_metadata =
    (NvDsMetaCompLatency*)g_malloc0(sizeof(NvDsMetaCompLatency));
  memcpy(dst_user_metadata, src_user_metadata, sizeof(NvDsMetaCompLatency));
  return (gpointer)dst_user_metadata;
}

void nvds_release_latency_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if(user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}

gdouble nvds_get_current_system_timestamp()
{
  struct timeval t1;
  double elapsedTime = 0;
  gettimeofday(&t1, NULL);
  elapsedTime = (t1.tv_sec) * 1000.0;
  elapsedTime += (t1.tv_usec) / 1000.0;
  return elapsedTime;
}

guint nvds_measure_buffer_latency(GstBuffer *buf,
    NvDsFrameLatencyInfo *latency_info)
{
  NvDsUserMetaList *latency_meta_list = NULL;
  NvDsUserMetaList *l = NULL;
  NvDsUserMeta *user_latency_meta = NULL;
  NvDsMetaCompLatency *latency_metadata = NULL;
  gdouble component_latency = 0.0;
  gdouble frame_in_time = 0;
  gdouble batch_out_time = 0;
  guint i = 0, src_cnt = 0;
  gdouble curr_time = 0.0;
  gdouble time_diff = 0.0;
  gdouble fps_val = 0.0;

  static guint batch_num = 0;
  static gdouble last_time= 0.0;

  curr_time = nvds_get_current_system_timestamp();
  if(last_time == 0)
  {
    fps_val = 0.0;
  }
  else
  {
    time_diff = curr_time - last_time;
    fps_val = (1/time_diff) * 1000;
  }
  last_time = curr_time;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    g_print ("Batch meta not found for buffer %p", buf);
    return FALSE;
  }
  nvds_acquire_meta_lock (batch_meta);
  memset(latency_info, 0, batch_meta->max_frames_in_batch *
      sizeof(NvDsFrameLatencyInfo));

  batch_out_time = nvds_get_current_system_timestamp();

  latency_meta_list = batch_meta->batch_user_meta_list;
  for (l = latency_meta_list; l != NULL; l = l->next) {
    user_latency_meta = (NvDsUserMeta *)l->data;
    if(user_latency_meta->base_meta.meta_type == NVDS_LATENCY_MEASUREMENT_META)
    {
      latency_metadata = user_latency_meta->user_meta_data;
      component_latency = latency_metadata->out_system_timestamp -
        latency_metadata->in_system_timestamp;

      if(!strncmp(latency_metadata->component_name, "nvv4l2decoder", 12))
      {
        frame_in_time = latency_metadata->in_system_timestamp;
        latency_info[src_cnt].comp_in_timestamp = frame_in_time;
      }
      if(!strncmp(latency_metadata->component_name, "audiodecoder", 12))
      {
        frame_in_time = latency_metadata->in_system_timestamp;
        latency_info[src_cnt].comp_in_timestamp = frame_in_time;
      }
      if(g_str_has_prefix(latency_metadata->component_name, "nvstreammux-"))
      {
        if(nvds_enable_component_latency_measurement)
        {
          g_print("Comp name = %s source_id = %d pad_index = %d frame_num = %d \
              in_system_timestamp = %lf out_system_timestamp = %lf \
              component_latency = %lf\n",
              latency_metadata->component_name, latency_metadata->source_id,
              latency_metadata->pad_index, latency_metadata->frame_num,
              latency_metadata->in_system_timestamp,
              latency_metadata->out_system_timestamp, component_latency);


          if(g_comp_csv_file != NULL)
          {
              WRITE_COMP_CSV_LINE(latency_metadata, batch_num,component_latency);
              WRITE_SUB_COMP_CSV_LINE(latency_metadata, batch_num);
          }
        }
        latency_info[src_cnt].source_id = latency_metadata->source_id;
        latency_info[src_cnt++].frame_num = latency_metadata->frame_num;
      }
      else
      {
        if(nvds_enable_component_latency_measurement)
        {
          g_print("Comp name = %s in_system_timestamp = %lf out_system_timestamp = %lf \
              component latency= %lf\n",
              latency_metadata->component_name,
              latency_metadata->in_system_timestamp,
              latency_metadata->out_system_timestamp, component_latency);

          if(g_comp_csv_file != NULL)
          {
              WRITE_COMP_CSV_LINE(latency_metadata, batch_num,component_latency);
              WRITE_SUB_COMP_CSV_LINE(latency_metadata, batch_num);
          }
        }
      }
    }
  }

  for(i = 0; i < src_cnt; i++)
  {
    latency_info[i].latency = batch_out_time - latency_info[i].comp_in_timestamp;
    /*
    g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
        latency_info[i].source_id,
        latency_info[i].frame_num,
        latency_info[i].latency);
        */
  }
  if(g_frame_csv_file != NULL)
  {
      WRITE_FRAME_CSV_LINE(latency_info, batch_num,fps_val,src_cnt);
  }
  batch_num++;
  nvds_release_meta_lock (batch_meta);
  return src_cnt;
}

NvDsUserMeta *nvds_set_input_system_timestamp(GstBuffer * buffer, gchar *element_name)
{
  NvDsMetaCompLatency *latency_metadata = NULL;
  NvDsUserMeta *user_latency_meta = NULL;
  if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
  {
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
    if (batch_meta == NULL) {
      return FALSE;
    }
    user_latency_meta = nvds_acquire_user_meta_from_pool (batch_meta);
    user_latency_meta->user_meta_data = (void *)nvds_set_latency_metadata_ptr();
    user_latency_meta->base_meta.meta_type = NVDS_LATENCY_MEASUREMENT_META;
    user_latency_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_latency_meta;
    user_latency_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_latency_meta;
    latency_metadata = (NvDsMetaCompLatency *)user_latency_meta->user_meta_data;
    g_strlcpy(latency_metadata->component_name,
        element_name, MAX_COMPONENT_LEN);
    latency_metadata->in_system_timestamp = nvds_get_current_system_timestamp();

    gst_mini_object_set_qdata((GstMiniObject *)buffer,
        g_quark_from_string(element_name), latency_metadata, NULL);

    nvds_add_user_meta_to_batch(batch_meta, user_latency_meta);
  }
  return user_latency_meta;
}

gboolean nvds_set_output_system_timestamp(GstBuffer * buffer, gchar *element_name)
{
  if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
  {
    NvDsMetaCompLatency *latency_metadata = NULL;
    latency_metadata = (NvDsMetaCompLatency *)gst_mini_object_get_qdata(
        (GstMiniObject *)buffer, g_quark_from_string(element_name));

    if(latency_metadata)
    {
      latency_metadata->out_system_timestamp = nvds_get_current_system_timestamp();
      return TRUE;
    }
  }
  return FALSE;
}

#if 0
gboolean nvds_add_sub_time(GstBuffer * buffer, gchar *element_name, gchar *name, gdouble start_time, gdouble end_time)
{
  if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
  {
    NvDsMetaCompLatency *latency_metadata = NULL;
    latency_metadata = (NvDsMetaCompLatency *)gst_mini_object_get_qdata(
        (GstMiniObject *)buffer, g_quark_from_string(element_name));

    if(latency_metadata)
    {
      NvDsMetaSubCompLatency *lat = &latency_metadata->sub_comp_latencies[latency_metadata->num_sub_comps];
      latency_metadata->num_sub_comps++;
      lat->in_system_timestamp = start_time;
      lat->out_system_timestamp = end_time;
      g_snprintf(lat->sub_comp_name, MAX_COMPONENT_LEN - 1, "%s", name);
      return TRUE;
    }
  }
  return FALSE;

}
#endif

void nvds_add_reference_timestamp_meta(GstBuffer * buffer,
    gchar *element_name, guint frame_id)
{
  gdouble current_sys_time = nvds_get_current_system_timestamp();
  GstCaps *reference = gst_caps_new_simple ("any",
          "component_name", G_TYPE_STRING, element_name,
          "frame_num", G_TYPE_INT, frame_id,
          "in_timestamp", G_TYPE_DOUBLE, current_sys_time,
          "out_timestamp", G_TYPE_DOUBLE, current_sys_time,
          NULL);
  GstReferenceTimestampMeta * latency_meta =
        gst_buffer_add_reference_timestamp_meta (buffer, reference,
            0, 0);
  gst_caps_unref(reference);
  (void)latency_meta;
}
