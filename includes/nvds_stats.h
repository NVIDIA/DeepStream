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

#ifndef __NVDS_STATS_H__
#define __NVDS_STATS_H__

#include <glib.h>
#include <nvml.h>

#ifdef __cplusplus
extern "C" {
#endif

gdouble get_ram_usage();
gdouble get_cpu_utilization();
gdouble get_gpu_memory(guint gpu_id);
gdouble get_gpu_utilization(guint gpu_id);

#ifdef __cplusplus
}
#endif

#endif /* __NVDS_STATS_H__ */