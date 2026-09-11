/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "nvds_stats.h"
#include <stdio.h>
#include <unistd.h>

static gboolean nvml_initialized = FALSE;
static GMutex nvml_mutex;

static gboolean ensure_nvml_initialized()
{
  g_mutex_lock(&nvml_mutex);
  if (!nvml_initialized) {
    if (nvmlInit() == NVML_SUCCESS) {
      nvml_initialized = TRUE;
    }
  }
  gboolean result = nvml_initialized;
  g_mutex_unlock(&nvml_mutex);
  return result;
}

gdouble get_gpu_utilization(guint gpu_id)
{
  nvmlDevice_t device = NULL;
  nvmlReturn_t result;
  nvmlUtilization_t utilization = {};

  if (!ensure_nvml_initialized()) {
    return -1.0;
  }

  result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
  if (result != NVML_SUCCESS) {
    return -1.0;
  }
  result = nvmlDeviceGetUtilizationRates(device, &utilization);
  if(result != NVML_SUCCESS) {
    return -1.0;
  }
  return utilization.gpu;
}

gdouble get_gpu_memory(guint gpu_id)
{
  nvmlDevice_t device = NULL;
  nvmlReturn_t result;
  nvmlMemory_t memory = {};

  if (!ensure_nvml_initialized()) {
    return -1.0;
  }

  result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
  if (result != NVML_SUCCESS) {
    return -1.0;
  }

  result = nvmlDeviceGetMemoryInfo(device, &memory);
  if(result != NVML_SUCCESS) {
    return -1.0;
  }

  gdouble gpu_memory_gb = gdouble(memory.used)/(1024*1024*1024); // bytes --> GB

  return gpu_memory_gb;
}

gdouble get_ram_usage()
{
  FILE *file = fopen("/proc/meminfo", "r");
  if (!file) {
    return 0.0;
  }

  char line[256];
  guint64 mem_total = 0;
  guint64 mem_available = 0;
  gboolean found_total = FALSE;
  gboolean found_available = FALSE;

  // Read file line by line to find MemTotal and MemAvailable
  while (fgets(line, sizeof(line), file)) {
    if (!found_total && g_str_has_prefix(line, "MemTotal:")) {
      if (sscanf(line, "MemTotal: %lu kB", &mem_total) == 1) {
        found_total = TRUE;
      }
    }
    else if (!found_available && g_str_has_prefix(line, "MemAvailable:")) {
      if (sscanf(line, "MemAvailable: %lu kB", &mem_available) == 1) {
        found_available = TRUE;
      }
    }

    // Exit early if we found both values
    if (found_total && found_available) {
      break;
    }
  }

  fclose(file);

  // Check if we successfully read both values
  if (!found_total || !found_available) {
    return 0.0;
  }

  // Calculate used memory (MemTotal - MemAvailable) and convert from kB to GB
  guint64 mem_used_kb = mem_total - mem_available;
  gdouble mem_used_gb = (gdouble)mem_used_kb / (1024.0 * 1024.0); // kB to GB

  return mem_used_gb;
}

gdouble get_cpu_utilization()
{
  FILE *file;
  char line[256];
  guint64 user, nice, system, idle, iowait, irq, softirq, steal;
  // First reading
  file = fopen("/proc/stat", "r");
  if (!file) {
    return 0.0;
  }

  if (!fgets(line, sizeof(line), file)) {
    fclose(file);
    return 0.0;
  }
  fclose(file);

  // Parse the CPU line: cpu user nice system idle iowait irq softirq steal
  if (sscanf(line, "cpu %lu %lu %lu %lu %lu %lu %lu %lu",
             &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) != 8) {
    return 0.0;
  }

  guint64 total1 = user + nice + system + idle + iowait + irq + softirq + steal;
  guint64 idle1 = idle + iowait;

  // Sleep for a short time to get a second reading
  usleep(100000); // 100ms

  // Second reading
  file = fopen("/proc/stat", "r");
  if (!file) {
    return 0.0;
  }

  if (!fgets(line, sizeof(line), file)) {
    fclose(file);
    return 0.0;
  }
  fclose(file);

  // Parse the CPU line again
  if (sscanf(line, "cpu %lu %lu %lu %lu %lu %lu %lu %lu",
             &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) != 8) {
    return 0.0;
  }

  guint64 total2 = user + nice + system + idle + iowait + irq + softirq + steal;
  guint64 idle2 = idle + iowait;

  // Calculate the differences
  guint64 total_diff = total2 - total1;
  guint64 idle_diff = idle2 - idle1;

  if (total_diff == 0) {
    return 0.0;
  }

  // Calculate CPU utilization percentage
  gdouble cpu_usage = (gdouble)(total_diff - idle_diff) / total_diff * 100.0;

  return cpu_usage;
}