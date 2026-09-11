/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "element.hpp"
#include "perf_monitor.hpp"
#include "deepstream_perf.h"

#include <string>
#include <mutex>
#include <gst/gst.h>

using namespace deepstream;
using namespace std;

typedef struct {
  gdouble fps[MAX_SOURCE_BINS] = {0.0};
  gdouble fps_avg[MAX_SOURCE_BINS] = {0.0};
  mutex lock;
} FpsData;

static void
perf_cb (gpointer context, NvDsAppPerfStruct * str) {
  deepstream::PerfMonitor* monitor = (deepstream::PerfMonitor*) context;
  monitor->print(str);
}

void PerfMonitor::print(void* info) {
  NvDsAppPerfStruct * str = (NvDsAppPerfStruct*) info;
  static guint header_print_cnt = 0;
  guint i;
  guint numf = str->num_instances;
  FpsData* fps_data = (FpsData*) this->fps_data_;
  gdouble* fps = &fps_data->fps[0];
  gdouble* fps_avg = &fps_data->fps_avg[0];

  lock_guard<mutex> lock(fps_data->lock);
  guint active_src_count = 0;

  if (!str->use_nvmultiurisrcbin) {
    for (i = 0; i < numf; i++) {
      fps[i] = str->fps[i];
      if (fps[i]){
        active_src_count++;
      }
      fps_avg[i] = str->fps_avg[i];
    }
    g_print("Active sources : %u\n", active_src_count);
    if (header_print_cnt % 20 == 0) {
      g_print ("\n**PERF:  ");
      for (i = 0; i < numf; i++) {
        g_print ("FPS %d (Avg)\t", i);
      }
      g_print ("\n");
      header_print_cnt = 0;
    }
    header_print_cnt++;

    time_t t = time (NULL);
    struct tm tm;
    char time_buf[26];
    localtime_r (&t, &tm);
    printf ("%s", asctime_r (&tm, time_buf));
    g_print ("**PERF:  ");

    for (i = 0; i < numf; i++) {
      g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
    }
  } else {
    for (guint j = 0; j < str->active_source_size; j++) {
      i = str->source_detail[j].source_id;
      fps[i] = str->fps[i];
      if (fps[i]){
        active_src_count++;
      }
      fps_avg[i] = str->fps_avg[i];
    }
    g_print("Active sources : %u\n", active_src_count);
    if (header_print_cnt % 20 == 0) {
      g_print ("\n**PERF:  ");
      for (guint j = 0; j < str->active_source_size; j++) {
        i = str->source_detail[j].source_id;
        g_print ("FPS %d (Avg)\t", i);
      }
      g_print ("\n");
      header_print_cnt = 0;
    }
    header_print_cnt++;

    time_t t = time (NULL);
    struct tm tm;
    char time_buf[26];
    localtime_r (&t, &tm);
    printf ("%s", asctime_r (&tm, time_buf));
    g_print ("**PERF:  ");

    g_print("\n");
    for (guint j = 0; j < str->active_source_size; j++) {
      i = str->source_detail[j].source_id;
      if (!str->stream_name_display){
        g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
      }
      else {
        g_print("%s[%s] %.2f (%.2f)\t", str->source_detail[j].sensor_id,str->source_detail[j].sensor_name,fps[i], fps_avg[i]);
      }
    }
  }
  g_print ("\n");
}



PerfMonitor::PerfMonitor(
    unsigned int batch_size, uint64_t interval, string src_type, bool show_name
): batch_size_(batch_size),
   interval_sec_(interval) {
    NvDsAppPerfStructInt* perf_struct = new NvDsAppPerfStructInt;
    memset(perf_struct, 0, sizeof(NvDsAppPerfStructInt));
    perf_struct->stream_name_display = show_name?1:0;
    perf_struct->use_nvmultiurisrcbin = (src_type == "nvmultiurisrcbin")?1:0;
    perf_struct->context = this;
    g_mutex_init(&perf_struct->struct_lock);
    perf_struct->FPSInfoHash = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, NULL);
    priv_ = perf_struct;
    fps_data_= new FpsData;
}

void PerfMonitor::apply(Element& element, const string& tips) {
  NvDsAppPerfStructInt* perf_struct = (NvDsAppPerfStructInt*) priv_;
  GstPad* pad = gst_element_get_static_pad(GST_ELEMENT(element.getGObject()), tips.c_str());

  enable_perf_measurement(perf_struct, pad, batch_size_, interval_sec_, 1, perf_cb);
}

void PerfMonitor::pause() {
  NvDsAppPerfStructInt* perf_struct = (NvDsAppPerfStructInt*) priv_;
  pause_perf_measurement(perf_struct);
}

void PerfMonitor::resume() {
  NvDsAppPerfStructInt* perf_struct = (NvDsAppPerfStructInt*) priv_;
  resume_perf_measurement(perf_struct);
}


void PerfMonitor::addStream(uint32_t source_id, const char* uri, const char* sensor_id, const char* sensor_name) {
  NvDsAppPerfStructInt* perf_struct = (NvDsAppPerfStructInt*) priv_;
  NvDsFPSSensorInfo* fpssensorInfoToHash = (NvDsFPSSensorInfo*)g_malloc0(sizeof(NvDsFPSSensorInfo));
  fpssensorInfoToHash->uri = (gchar const*)g_strdup(uri);
  fpssensorInfoToHash->source_id = source_id;
  fpssensorInfoToHash->sensor_id = (gchar const*)g_strdup(sensor_id);
  fpssensorInfoToHash->sensor_name = (gchar const*)g_strdup(sensor_name);
  g_hash_table_insert (perf_struct->FPSInfoHash, GUINT_TO_POINTER(source_id), fpssensorInfoToHash);
}

void PerfMonitor::removeStream(uint32_t source_id) {
  NvDsAppPerfStructInt* perf_struct = (NvDsAppPerfStructInt*) priv_;
  NvDsFPSSensorInfo* sensorInfo = (NvDsFPSSensorInfo*)g_hash_table_lookup(perf_struct->FPSInfoHash,
        GUINT_TO_POINTER(source_id));
  if (sensorInfo) {
    g_hash_table_remove(perf_struct->FPSInfoHash, GUINT_TO_POINTER(source_id));
    if(sensorInfo->sensor_id) {
        g_free((void*)sensorInfo->sensor_id);
    }
    if(sensorInfo->sensor_name) {
        g_free((void*)sensorInfo->sensor_name);
    }
    if(sensorInfo->uri) {
        g_free((void*)sensorInfo->uri);
    }
    g_free(sensorInfo);
  }
}

PerfMonitor::~PerfMonitor() {
  NvDsAppPerfStructInt* perf_struct = (NvDsAppPerfStructInt*) priv_;
  GList *keys = g_hash_table_get_keys(perf_struct->FPSInfoHash);
  // Iterate over the keys using a for loop
  for (GList *iter = keys; iter != NULL; iter = iter->next) {
      uint32_t source_id = GPOINTER_TO_UINT(iter->data);
      this->removeStream(source_id);
  }
  g_hash_table_destroy(perf_struct->FPSInfoHash);
  g_list_free(keys);
  delete perf_struct;
  FpsData* fps_data = (FpsData*) fps_data_;
  delete fps_data;
}