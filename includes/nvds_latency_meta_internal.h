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

#ifndef _NVDSMETA_LATENCY_INTERNAL_H_
#define _NVDSMETA_LATENCY_INTERNAL_H_

#include "glib.h"

#ifdef __cplusplus
extern "C" {
#endif

void *nvds_set_latency_metadata_ptr(void);

gpointer nvds_copy_latency_meta(gpointer data, gpointer user_data);

void nvds_release_latency_meta(gpointer data, gpointer user_data);

gdouble nvds_get_current_system_timestamp(void);

gboolean nvds_get_enable_per_component_latency_measurement(void);

#define nvds_enable_component_latency_measurement (nvds_get_enable_per_component_latency_measurement())

#ifdef __cplusplus
}
#endif
#endif
