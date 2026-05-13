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

#include "nvds_rest_metrics.h"
#include <string.h>

// ============================================================================
// Global shared data instances
// ============================================================================
NvDsFrameLatencySharedData g_shared_frame_latency_data = {0};
NvDsCompLatencySharedData g_shared_comp_latency_data = {0};

// ============================================================================
// Frame Latency Functions
// ============================================================================

/**
 * Initialize the shared frame latency data structure
 */
__attribute__((constructor))
void nvds_init_shared_frame_latency_data(void)
{
    g_mutex_init(&g_shared_frame_latency_data.mutex);
    g_mutex_init(&g_shared_frame_latency_data.frame_metrics_mutex);

    g_shared_frame_latency_data.frame_latency_data = NULL;
    g_shared_frame_latency_data.fps_data = NULL;

    g_shared_frame_latency_data.frame_latency_count = 0;
    g_shared_frame_latency_data.fps_count = 0;

    g_shared_frame_latency_data.frame_latency_available = FALSE;
    g_shared_frame_latency_data.fps_data_available = FALSE;
}

/**
 * Cleanup the shared frame latency data structure
 */
void nvds_cleanup_shared_frame_latency_data(void)
{
    g_mutex_lock(&g_shared_frame_latency_data.mutex);

    if (g_shared_frame_latency_data.frame_latency_data) {
        g_free(g_shared_frame_latency_data.frame_latency_data);
        g_shared_frame_latency_data.frame_latency_data = NULL;
    }

    if (g_shared_frame_latency_data.fps_data) {
        g_free(g_shared_frame_latency_data.fps_data);
        g_shared_frame_latency_data.fps_data = NULL;
    }

    g_shared_frame_latency_data.frame_latency_count = 0;
    g_shared_frame_latency_data.fps_count = 0;
    g_shared_frame_latency_data.frame_latency_available = FALSE;
    g_shared_frame_latency_data.fps_data_available = FALSE;

    g_mutex_unlock(&g_shared_frame_latency_data.mutex);

    // Properly clear both mutexes
    g_mutex_clear(&g_shared_frame_latency_data.mutex);
    g_mutex_clear(&g_shared_frame_latency_data.frame_metrics_mutex);
}

/**
 * Update shared frame latency data from nvdslogger
 */
void nvds_update_shared_frame_latency_data(NvDsMetricsFrameLatency* latency_data, guint count)
{
    if (!latency_data || count == 0) {
        return;
    }

    g_mutex_lock(&g_shared_frame_latency_data.mutex);

    // Free existing data
    if (g_shared_frame_latency_data.frame_latency_data) {
        g_free(g_shared_frame_latency_data.frame_latency_data);
        g_shared_frame_latency_data.frame_latency_data = NULL;
    }

    // Allocate new data
    g_shared_frame_latency_data.frame_latency_data =
        (NvDsMetricsFrameLatency*)g_malloc0(sizeof(NvDsMetricsFrameLatency) * count);

    if (g_shared_frame_latency_data.frame_latency_data) {
        memcpy(g_shared_frame_latency_data.frame_latency_data, latency_data,
               sizeof(NvDsMetricsFrameLatency) * count);
        g_shared_frame_latency_data.frame_latency_count = count;
        g_shared_frame_latency_data.frame_latency_available = TRUE;
    } else {
        g_shared_frame_latency_data.frame_latency_count = 0;
        g_shared_frame_latency_data.frame_latency_available = FALSE;
    }

    g_mutex_unlock(&g_shared_frame_latency_data.mutex);
}

/**
 * Get shared frame latency data
 */
NvDsMetricsFrameLatency* nvds_get_shared_frame_latency_data(guint* count)
{
    if (!count) {
        return NULL;
    }

    g_mutex_lock(&g_shared_frame_latency_data.mutex);

    if (g_shared_frame_latency_data.frame_latency_available &&
        g_shared_frame_latency_data.frame_latency_data) {
        *count = g_shared_frame_latency_data.frame_latency_count;
        g_mutex_unlock(&g_shared_frame_latency_data.mutex);
        return g_shared_frame_latency_data.frame_latency_data;
    } else {
        *count = 0;
        g_mutex_unlock(&g_shared_frame_latency_data.mutex);
        return NULL;
    }
}

// ============================================================================
// Component Latency Functions
// ============================================================================

/**
 * Initialize the shared component latency data structure
 */
__attribute__((constructor))
void nvds_init_shared_comp_latency_data(void)
{
    g_mutex_init(&g_shared_comp_latency_data.mutex);
    g_mutex_init(&g_shared_comp_latency_data.comp_metrics_mutex);

    g_shared_comp_latency_data.comp_latency_data = NULL;
    g_shared_comp_latency_data.comp_latency_count = 0;
    g_shared_comp_latency_data.comp_latency_available = FALSE;
}

/**
 * Cleanup the shared component latency data structure
 */
void nvds_cleanup_shared_comp_latency_data(void)
{
    g_mutex_lock(&g_shared_comp_latency_data.mutex);

    if (g_shared_comp_latency_data.comp_latency_data) {
        g_free(g_shared_comp_latency_data.comp_latency_data);
        g_shared_comp_latency_data.comp_latency_data = NULL;
    }

    g_shared_comp_latency_data.comp_latency_count = 0;
    g_shared_comp_latency_data.comp_latency_available = FALSE;

    g_mutex_unlock(&g_shared_comp_latency_data.mutex);

    // Properly clear both mutexes
    g_mutex_clear(&g_shared_comp_latency_data.mutex);
    g_mutex_clear(&g_shared_comp_latency_data.comp_metrics_mutex);
}

/**
 * Update shared component latency data
 */
void nvds_update_shared_comp_latency_data(NvDsMetricsCompLatency* latency_data, guint count)
{
    if (!latency_data || count == 0) {
        return;
    }

    g_mutex_lock(&g_shared_comp_latency_data.mutex);

    // Free existing data
    if (g_shared_comp_latency_data.comp_latency_data) {
        g_free(g_shared_comp_latency_data.comp_latency_data);
        g_shared_comp_latency_data.comp_latency_data = NULL;
    }

    // Allocate new data
    g_shared_comp_latency_data.comp_latency_data =
        (NvDsMetricsCompLatency*)g_malloc0(sizeof(NvDsMetricsCompLatency) * count);

    if (g_shared_comp_latency_data.comp_latency_data) {
        memcpy(g_shared_comp_latency_data.comp_latency_data, latency_data,
               sizeof(NvDsMetricsCompLatency) * count);
        g_shared_comp_latency_data.comp_latency_count = count;
        g_shared_comp_latency_data.comp_latency_available = TRUE;
    } else {
        g_shared_comp_latency_data.comp_latency_count = 0;
        g_shared_comp_latency_data.comp_latency_available = FALSE;
    }

    g_mutex_unlock(&g_shared_comp_latency_data.mutex);
}

/**
 * Get shared component latency data
 */
NvDsMetricsCompLatency* nvds_get_shared_comp_latency_data(guint* count)
{
    if (!count) {
        return NULL;
    }

    g_mutex_lock(&g_shared_comp_latency_data.mutex);

    if (g_shared_comp_latency_data.comp_latency_available &&
        g_shared_comp_latency_data.comp_latency_data) {
        *count = g_shared_comp_latency_data.comp_latency_count;
        g_mutex_unlock(&g_shared_comp_latency_data.mutex);
        return g_shared_comp_latency_data.comp_latency_data;
    } else {
        *count = 0;
        g_mutex_unlock(&g_shared_comp_latency_data.mutex);
        return NULL;
    }
}

// ============================================================================
// FPS Functions
// ============================================================================

/**
 * Update shared FPS data
 */
void nvds_update_shared_fps_data(NvDsMetricsFpsData* data, guint count)
{
    if (!data || count == 0) {
        return;
    }

    g_mutex_lock(&g_shared_frame_latency_data.mutex);

    // Free existing data
    if (g_shared_frame_latency_data.fps_data) {
        g_free(g_shared_frame_latency_data.fps_data);
        g_shared_frame_latency_data.fps_data = NULL;
    }

    // Allocate new data
    g_shared_frame_latency_data.fps_data =
        (NvDsMetricsFpsData*)g_malloc0(sizeof(NvDsMetricsFpsData) * count);

    if (g_shared_frame_latency_data.fps_data) {
        memcpy(g_shared_frame_latency_data.fps_data, data,
               sizeof(NvDsMetricsFpsData) * count);
        g_shared_frame_latency_data.fps_count = count;
        g_shared_frame_latency_data.fps_data_available = TRUE;
    } else {
        g_shared_frame_latency_data.fps_count = 0;
        g_shared_frame_latency_data.fps_data_available = FALSE;
    }

    g_mutex_unlock(&g_shared_frame_latency_data.mutex);
}

/**
 * Get shared FPS data
 */
NvDsMetricsFpsData* nvds_get_shared_fps_data(guint* count)
{
    if (!count) {
        return NULL;
    }

    g_mutex_lock(&g_shared_frame_latency_data.mutex);

    if (g_shared_frame_latency_data.fps_data_available &&
        g_shared_frame_latency_data.fps_data) {
        *count = g_shared_frame_latency_data.fps_count;
        g_mutex_unlock(&g_shared_frame_latency_data.mutex);
        return g_shared_frame_latency_data.fps_data;
    } else {
        *count = 0;
        g_mutex_unlock(&g_shared_frame_latency_data.mutex);
        return NULL;
    }
}
