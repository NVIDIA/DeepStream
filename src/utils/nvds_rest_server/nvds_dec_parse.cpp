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
nvds_rest_dec_parse (const Json::Value & in, NvDsServerDecInfo * dec_info)
{
  if (dec_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      dec_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      dec_info->stream_id =
          sub_root_val.get ("stream_id", EMPTY_STRING).asString ().c_str ();
      if (dec_info->dec_flag == DROP_FRAME_INTERVAL) {
        try {
          dec_info->drop_frame_interval =
              sub_root_val.get ("drop_frame_interval", 0).asUInt ();

          if (dec_info->drop_frame_interval > 30) {
            dec_info->dec_log =
                "DROP_FRAME_INTERVAL_UPDATE_FAIL, drop_frame_interval value not parsed correctly, Range: 0 - 30";
            dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
            dec_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, error: " + std::string(e.what());
            dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
            dec_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (dec_info->dec_flag == SKIP_FRAMES) {
        try {
          dec_info->skip_frames = sub_root_val.get ("skip_frames", 0).asUInt ();

          if (dec_info->skip_frames > 2) {
            dec_info->dec_log =
                "SKIP_FRAMES_UPDATE_FAIL, skip_frames value not parsed correctly, Range: 0-2";
            dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
            dec_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, error: " + std::string(e.what());
            dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
            dec_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (dec_info->dec_flag == LOW_LATENCY_MODE) {
        try {
          dec_info->low_latency_mode =
            sub_root_val.get ("low_latency_mode", 0).asBool ();
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, error: " + std::string(e.what());
            dec_info->status = LOW_LATENCY_MODE_UPDATE_FAIL;
            dec_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
