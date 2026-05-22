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
