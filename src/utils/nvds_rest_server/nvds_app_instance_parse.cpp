/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "nvds_rest_server.h"
#include "nvds_parse.h"

bool
nvds_rest_app_instance_parse (const Json::Value & in,
    NvDsServerAppInstanceInfo * appinstance_info)
{
  if (appinstance_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      appinstance_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key
      if (appinstance_info->appinstance_flag == QUIT_APP){
        try {
          appinstance_info->app_quit = sub_root_val.get ("app_quit", 0).asBool ();
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            appinstance_info->app_log = "QUIT_FAIL, error: " + std::string(e.what());
            appinstance_info->status = QUIT_FAIL;
            appinstance_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
