/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
