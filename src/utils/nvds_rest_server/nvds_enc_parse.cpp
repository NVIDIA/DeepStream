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
nvds_rest_enc_parse (const Json::Value & in, NvDsServerEncInfo * enc_info)
{
  if (enc_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      enc_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      enc_info->stream_id =
          sub_root_val.get ("stream_id", EMPTY_STRING).asString ().c_str ();
      if (enc_info->enc_flag == BITRATE) {
        try {
          enc_info->bitrate = sub_root_val.get ("bitrate", 0).asUInt ();
          if (enc_info->bitrate > UINT_MAX) {
            enc_info->enc_log =
                "BITRATE_UPDATE_FAIL, bitrate value not parsed correctly, Unsigned Integer. Range: 0 - 4294967295";
            enc_info->status = BITRATE_UPDATE_FAIL;
            enc_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            enc_info->enc_log = "BITRATE_UPDATE_FAIL, error: " + std::string(e.what());
            enc_info->status = BITRATE_UPDATE_FAIL;
            enc_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (enc_info->enc_flag == FORCE_IDR) {
        try {
          enc_info->force_idr = sub_root_val.get ("force_idr", 0).asBool ();
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            enc_info->enc_log = "FORCE_IDR_UPDATE_FAIL, error: " + std::string(e.what());
            enc_info->status = FORCE_IDR_UPDATE_FAIL;
            enc_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (enc_info->enc_flag == FORCE_INTRA) {
        try {
          enc_info->force_intra = sub_root_val.get ("force_intra", 0).asBool ();
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            enc_info->enc_log = "FORCE_INTRA_UPDATE_FAIL, error: " + std::string(e.what());
            enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            enc_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (enc_info->enc_flag == IFRAME_INTERVAL) {
        try {
          enc_info->iframeinterval =
              sub_root_val.get ("iframeinterval", 0).asUInt ();
          if (enc_info->iframeinterval < 0 || enc_info->iframeinterval > 256) {
            enc_info->enc_log =
                "IFRAME_INTERVAL_UPDATE_FAIL, iframeinterval value not parsed correctly, Unsigned Integer Range: 0 - 4294967295";
            enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            enc_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            enc_info->enc_log = "IFRAME_INTERVAL_UPDATE_FAIL, error: " + std::string(e.what());
            enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            enc_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
