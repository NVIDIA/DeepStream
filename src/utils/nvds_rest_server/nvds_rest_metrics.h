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

#ifndef _NVDS_REST_METRICS_H_
#define _NVDS_REST_METRICS_H_

#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_COMPONENT_LEN 64

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

/**
 * Holds information about the latency of a given frame.
 */
typedef struct _NvDsMetricsFrameLatency {
  guint source_id;            // Source ID, e.g., camera ID
  guint frame_num;            // Frame number for which latency is measured
  gdouble comp_in_timestamp;  // System timestamp when buffer arrives at the first component
  gdouble latency;            // Latency of the frame in milliseconds
} NvDsMetricsFrameLatency;

/**
 * Holds information about latency of a given component
 */
typedef struct _NvDsMetricsCompLatency {
  gchar component_name[MAX_COMPONENT_LEN]; // Component name
  gdouble in_system_timestamp;             // Timestamp when buffer arrives at component input
  gdouble out_system_timestamp;            // Timestamp when buffer is sent downstream
  guint source_id;                          // Source ID, e.g., camera ID
  guint frame_num;                          // Frame number
  guint pad_index;                          // Pad index for the frame in the batch
  guint num_sub_comps;                      // Number of subcomponents
  gdouble latency;                          // Latency in milliseconds
} NvDsMetricsCompLatency;

/**
 * Holds FPS data for a source
 */
typedef struct {
  guint source_id;
  gdouble fps_val;
  gchar source_name[256];
} NvDsMetricsFpsData;

/**
 * Shared data structure for frame latency metrics
 */
typedef struct {
  NvDsMetricsFrameLatency* frame_latency_data;
  NvDsMetricsFpsData* fps_data;

  guint frame_latency_count;
  guint fps_count;

  gboolean frame_latency_available;
  gboolean fps_data_available;

  GMutex mutex;                // Global lock
  GMutex frame_metrics_mutex;  // Additional lock for frame metrics
  void* user_data;
} NvDsFrameLatencySharedData;

/**
 * Shared data structure for component latency metrics
 */
typedef struct {
  NvDsMetricsCompLatency* comp_latency_data;
  guint comp_latency_count;

  gboolean comp_latency_available;

  GMutex mutex;                 // Global lock
  GMutex comp_metrics_mutex;    // Additional lock for component metrics
  void* user_data;
} NvDsCompLatencySharedData;

// -----------------------------------------------------------------------------
// Global Instances
// -----------------------------------------------------------------------------
extern NvDsFrameLatencySharedData g_shared_frame_latency_data;
extern NvDsCompLatencySharedData g_shared_comp_latency_data;

// -----------------------------------------------------------------------------
// API Functions
// -----------------------------------------------------------------------------

// Frame Latency APIs
void nvds_init_shared_frame_latency_data(void);
void nvds_cleanup_shared_frame_latency_data(void);
void nvds_update_shared_frame_latency_data(NvDsMetricsFrameLatency* latency_data, guint count);
NvDsMetricsFrameLatency* nvds_get_shared_frame_latency_data(guint* count);

// Component Latency APIs
void nvds_init_shared_comp_latency_data(void);
void nvds_cleanup_shared_comp_latency_data(void);
void nvds_update_shared_comp_latency_data(NvDsMetricsCompLatency* latency_data, guint count);
NvDsMetricsCompLatency* nvds_get_shared_comp_latency_data(guint* count);

// FPS APIs
void nvds_update_shared_fps_data(NvDsMetricsFpsData* data, guint count);
NvDsMetricsFpsData* nvds_get_shared_fps_data(guint* count);

#ifdef __cplusplus
}
#endif

#endif // _NVDS_REST_METRICS_H_
