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
nvds_rest_infer_parse (const Json::Value & in, NvDsServerInferInfo * infer_info)
{
  if (infer_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      infer_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

        infer_info->stream_id =
            sub_root_val.get ("stream_id", EMPTY_STRING).asString ().c_str ();

        if (infer_info->infer_flag == INFER_INTERVAL) {
          try {
            infer_info->interval =
                sub_root_val.get ("interval", 0).asUInt ();
            if (infer_info->interval > 32) {
              infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, interval value not parsed correctly, Unsigned Integer. Range: 0 - 32 ";
              infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
              infer_info->err_info.code = StatusBadRequest;
              return false;
            }
          } catch (const std::exception& e) {
            // Error handling: other exceptions
            infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, error: " + std::string(e.what());
            infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
            infer_info->err_info.code = StatusBadRequest;
            return false;
          }
        }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
