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

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include <string>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_tiled_display_yaml (NvDsTiledDisplayConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml["tiled-display"].begin();
     itr != configyml["tiled-display"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable =
          (NvDsTiledDisplayEnable) itr->second.as<int>();
    } else if (paramKey == "rows") {
      config->rows = itr->second.as<guint>();
    } else if (paramKey == "columns") {
      config->columns = itr->second.as<guint>();
    } else if (paramKey == "width") {
      config->width = itr->second.as<guint>();
    } else if (paramKey == "height") {
      config->height = itr->second.as<guint>();
    } else if (paramKey == "gpu-id") {
      config->gpu_id = itr->second.as<guint>();
    } else if (paramKey == "nvbuf-memory-type") {
      config->nvbuf_memory_type = itr->second.as<guint>();
    } else if (paramKey == "compute-hw") {
      config->compute_hw = itr->second.as<guint>();
    } else if (paramKey == "buffer-pool-size") {
      config->buffer_pool_size = itr->second.as<guint>();
    } else if (paramKey == "square-seq-grid") {
      config->square_seq_grid = itr->second.as<gboolean>();
    } else {
      cout << "[WARNING] Unknown param found in tiled-display: " << paramKey << endl;
    }
  }
  ret = TRUE;

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
