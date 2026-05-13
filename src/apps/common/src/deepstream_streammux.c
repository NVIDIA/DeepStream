/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_common.h"
#include "deepstream_streammux.h"
#include <string.h>

// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean
set_streammux_properties (NvDsStreammuxConfig * config, GstElement * element)
{
  gboolean ret = FALSE;
  const gchar *new_mux_str = g_getenv ("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0 (new_mux_str, "yes");

  if (!use_new_mux) {
    g_object_set (G_OBJECT (element), "gpu-id", config->gpu_id, NULL);

    g_object_set (G_OBJECT (element), "nvbuf-memory-type",
        config->nvbuf_memory_type, NULL);

    g_object_set (G_OBJECT (element), "live-source", config->live_source, NULL);

    g_object_set (G_OBJECT (element),
        "batched-push-timeout", config->batched_push_timeout, NULL);

    g_object_set (G_OBJECT (element), "compute-hw", config->compute_hw, NULL);
    g_object_set (G_OBJECT (element), "sort-batch", config->sort_batch, NULL);
    g_object_set (G_OBJECT (element), "cache-buffer", config->buffer_cache, NULL);
    g_object_set (G_OBJECT (element), "cache-buffer-timeout", config->buffer_cache_timeout, NULL);
    g_object_set (G_OBJECT (element), "sync-inputs-ntp", config->sync_inputs_ntp, NULL);

    if (config->buffer_pool_size >= 4) {
      g_object_set (G_OBJECT (element),
          "buffer-pool-size", config->buffer_pool_size, NULL);
    }

    g_object_set (G_OBJECT (element), "enable-padding",
        config->enable_padding, NULL);

    if (config->pipeline_width && config->pipeline_height) {
      g_object_set (G_OBJECT (element), "width", config->pipeline_width, NULL);
      g_object_set (G_OBJECT (element), "height",
          config->pipeline_height, NULL);
    }
    if (!config->use_nvmultiurisrcbin) {
      g_object_set (G_OBJECT (element), "async-process",
          config->async_process, NULL);
    }

  }

  if (config->batch_size && !config->use_nvmultiurisrcbin) {
    g_object_set (G_OBJECT (element), "batch-size", config->batch_size, NULL);
  }

  g_object_set (G_OBJECT (element), "attach-sys-ts",
      config->attach_sys_ts_as_ntp, NULL);

  if (config->config_file_path) {
    g_object_set (G_OBJECT (element),
        "config-file-path", GET_FILE_PATH (config->config_file_path), NULL);
  }

  g_object_set (G_OBJECT (element), "frame-duration",
      config->frame_duration, NULL);

  g_object_set (G_OBJECT (element), "frame-num-reset-on-stream-reset",
      config->frame_num_reset_on_stream_reset, NULL);

  g_object_set (G_OBJECT (element), "sync-inputs", config->sync_inputs, NULL);

  g_object_set (G_OBJECT (element), "max-latency", config->max_latency, NULL);
  g_object_set (G_OBJECT (element), "frame-num-reset-on-eos",
      config->frame_num_reset_on_eos, NULL);
  g_object_set (G_OBJECT (element), "drop-pipeline-eos", config->no_pipeline_eos,
      NULL);

  if (config->extract_sei_type5_data) {
    g_object_set (G_OBJECT (element), "extract-sei-type5-data", config->extract_sei_type5_data,
        NULL);
  }
  if (config->extract_sei_sim_time) {
    g_object_set (G_OBJECT (element), "extract-sei-sim-time", config->extract_sei_sim_time,
        NULL);
  }
  if (config->align_first_buffer) {
    g_object_set (G_OBJECT (element), "align-first-buffer", config->align_first_buffer,
        NULL);
  }
  if (config->failsafe_flush_count > 0) {
    g_object_set (G_OBJECT (element), "failsafe-flush-count", config->failsafe_flush_count,
        NULL);
  }
  if (config->drop_backward_sei) {
    g_object_set (G_OBJECT (element), "drop-backward-sei", config->drop_backward_sei,
        NULL);
  }
  if (config->num_surface_per_frame > 1) {
      g_object_set (G_OBJECT (element), "num-surfaces-per-frame",
          config->num_surface_per_frame, NULL);
  }

  ret = TRUE;

  return ret;
}
