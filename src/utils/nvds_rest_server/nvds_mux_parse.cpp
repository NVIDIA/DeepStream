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
nvds_rest_mux_parse (const Json::Value & in, NvDsServerMuxInfo * mux_info)
{
  if (mux_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      mux_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      if (mux_info->mux_flag == BATCHED_PUSH_TIMEOUT) {
        try {
            mux_info->batched_push_timeout = sub_root_val.get("batched_push_timeout", -1).asInt();
            if (mux_info->batched_push_timeout < -1
                    || mux_info->batched_push_timeout > INT_MAX) {
                  mux_info->mux_log =
                      "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL, batched_push_timeout value not parsed correctly,  Range: -1 - 2147483647";
                  mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
                  mux_info->err_info.code = StatusBadRequest;
                  return false;
                }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL, error: " + std::string(e.what());
            mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
            mux_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (mux_info->mux_flag == MAX_LATENCY) {
        mux_info->max_latency = sub_root_val.get ("max_latency", 0).asUInt ();
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
