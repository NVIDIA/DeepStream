/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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