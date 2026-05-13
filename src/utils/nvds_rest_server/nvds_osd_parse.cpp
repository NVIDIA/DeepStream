/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define EMPTY_STRING ""

bool
nvds_rest_osd_parse (const Json::Value & in, NvDsServerOsdInfo * osd_info)
{
  if (osd_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      osd_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      osd_info->stream_id =
          sub_root_val.get ("stream_id", EMPTY_STRING).asString ().c_str ();

      if (osd_info->osd_flag == PROCESS_MODE) {
        try {
          osd_info->process_mode = sub_root_val.get ("process_mode", 0).asInt ();
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, error: " + std::string(e.what());
            osd_info->status = PROCESS_MODE_UPDATE_FAIL;
            osd_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
