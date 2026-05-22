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

#ifndef _NVGSTDS_STREAMMUX_H_
#define _NVGSTDS_STREAMMUX_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  // Struct members to store config / properties for the element
  gint pipeline_width;
  gint pipeline_height;
  gint buffer_pool_size;
  gint batch_size;
  gint batched_push_timeout;
  gint compute_hw;
  gboolean sort_batch;
  gboolean buffer_cache;
  gint buffer_cache_timeout;
  gint num_surface_per_frame;
  gint interpolation_method;
  guint64 frame_duration;
  guint gpu_id;
  guint nvbuf_memory_type;
  gboolean live_source;
  gboolean enable_padding;
  gboolean is_parsed;
  gboolean attach_sys_ts_as_ntp;
  gchar* config_file_path;
  gboolean sync_inputs;
  guint64 max_latency;
  gboolean frame_num_reset_on_eos;
  gboolean frame_num_reset_on_stream_reset;
  gboolean async_process;
  gboolean no_pipeline_eos;
  gboolean use_nvmultiurisrcbin;
  gboolean extract_sei_type5_data;
  gboolean extract_sei_sim_time;
  gboolean align_first_buffer;
  guint sync_inputs_ntp;
  guint failsafe_flush_count;
  gboolean drop_backward_sei;
} NvDsStreammuxConfig;

// Function to create the bin and set properties
gboolean
set_streammux_properties (NvDsStreammuxConfig *config, GstElement *streammux);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSEXAMPLE_H_ */
