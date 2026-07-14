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

#include <iostream>

#include "nvds_rest_server.h"
#include "nvds_parse.h"

bool
nvds_rest_nvtracker_parse (const Json::Value & in, NvDsServerNvTrackerInfo * trackerInfo)
{
  if (trackerInfo->uri.find("/api/v1/") != std::string::npos)
  {
    for (Json::ValueConstIterator it = in.begin(); it != in.end(); ++it)
    {
      std::string root_val = it.key().asString ().c_str();
      trackerInfo->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      trackerInfo->stream_id =
          sub_root_val.get ("stream_id", "").asString().c_str();

      if (trackerInfo->nvTracker_flag == NVTRACKER_CONFIG)
      {
        try
        {
          trackerInfo->config_path = sub_root_val.get("config_path", "").asString();
        }
        catch (const std::exception& e)
        {
          // Error handling: other exceptions
          trackerInfo->nvTracker_log = "NVTRACKER_CONFIG_UPDATE_FAIL, error: "
                                       + std::string(e.what());
          trackerInfo->status = NVTRACKER_CONFIG_UPDATE_FAIL;
          trackerInfo->err_info.code = StatusBadRequest;
          return false;
        }
      }
    }
  }
  else
  {
    g_print ("Unsupported REST API version\n");
  }
  return true;
}
