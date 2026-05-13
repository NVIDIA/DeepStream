/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
