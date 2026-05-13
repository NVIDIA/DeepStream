/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "deepstream_config_yaml.h"
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_streammux_yaml (NvDsStreammuxConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;

  config->frame_duration = -1;
  config->batched_push_timeout = -1;
  config->attach_sys_ts_as_ntp = TRUE;
  config->async_process = TRUE;
  config->no_pipeline_eos = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml["streammux"].begin(); itr != configyml["streammux"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if(paramKey == "width") {
      config->pipeline_width = itr->second.as<gint>();
    }
    else if (paramKey == "height")  {
      config->pipeline_height = itr->second.as<gint>();
    }
    else if (paramKey == "gpu-id")  {
      config->gpu_id = itr->second.as<guint>();
    }
    else if (paramKey == "live-source")  {
      config->live_source = itr->second.as<gboolean>();
    }
    else if (paramKey == "buffer-pool-size")  {
      config->buffer_pool_size = itr->second.as<gint>();
    }
    else if (paramKey == "batch-size")  {
      config->batch_size = itr->second.as<gint>();
    }
    else if (paramKey == "batched-push-timeout")  {
      config->batched_push_timeout = itr->second.as<gint>();
    }
    else if (paramKey == "enable-padding")  {
      config->enable_padding = itr->second.as<gboolean>();
    }
    else if (paramKey == "frame-duration")  {
      config->frame_duration = itr->second.as<guint64>();
    }
    else if (paramKey == "nvbuf-memory-type")  {
      config->nvbuf_memory_type = itr->second.as<guint>();
    }
    else if (paramKey == "config-file-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->config_file_path)) {
        g_printerr ("Error: Could not parse config-file in streammux\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    }
    else if (paramKey == "compute-hw")  {
      config->compute_hw = itr->second.as<gint>();
    }
    else if (paramKey == "attach-sys-ts")  {
      config->attach_sys_ts_as_ntp = itr->second.as<gboolean>();
    }
    else if (paramKey == "frame-num-reset-on-stream-reset")  {
      config->frame_num_reset_on_stream_reset = itr->second.as<gboolean>();
    }
    else if (paramKey == "frame-num-reset-on-eos")  {
      config->frame_num_reset_on_eos = itr->second.as<gboolean>();
    }
    else if (paramKey == "num-surfaces-per-frame")  {
      config->num_surface_per_frame = itr->second.as<gint>();
    }
    else if (paramKey == "interpolation-method")  {
      config->interpolation_method = itr->second.as<gint>();
    }
    else if (paramKey == "sync-inputs")  {
      config->sync_inputs = itr->second.as<gboolean>();
    }
    else if (paramKey == "max-latency")  {
      config->max_latency = itr->second.as<guint64>();
    }
    else if (paramKey == "async-process")  {
      config->async_process = itr->second.as<gboolean>();
    }
    else if (paramKey == "drop-pipeline-eos")  {
      config->no_pipeline_eos = itr->second.as<gboolean>();
    }
    else if (paramKey == "failsafe-flush-count")  {
      config->failsafe_flush_count = itr->second.as<guint>();
    }
    else {
      cout << "[WARNING] Unknown param found in streammux: " << paramKey << endl;
      goto done;
    }
  }
  config->is_parsed = TRUE;

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
