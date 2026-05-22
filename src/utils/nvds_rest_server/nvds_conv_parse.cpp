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
nvds_rest_conv_parse (const Json::Value & in, NvDsServerConvInfo * conv_info)
{
  if (conv_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      conv_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      conv_info->stream_id =
          sub_root_val.get ("stream_id", EMPTY_STRING).asString ().c_str ();
      if (conv_info->stream_id == "") {
        if (conv_info->conv_flag == SRC_CROP){
          conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, stream_id value not parsed correctly";
          conv_info->status = SRC_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusBadRequest;
          return false;
        }
        else if (conv_info->conv_flag == DEST_CROP){
          conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, stream_id value not parsed correctly";
          conv_info->status = DEST_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusBadRequest;
          return false;
        }
        else if (conv_info->conv_flag == FLIP_METHOD){
          conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, stream_id value not parsed correctly";
          conv_info->status = FLIP_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusBadRequest;
          return false;
        }
        else if (conv_info->conv_flag == INTERPOLATION_METHOD){
          conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, stream_id value not parsed correctly";
          conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusBadRequest;
          return false;
        }
      }
      if (conv_info->conv_flag == SRC_CROP) {
        try {
          conv_info->src_crop =
              sub_root_val.get ("src_crop", EMPTY_STRING).asString ().c_str ();

          if (conv_info->src_crop == "") {
            conv_info->conv_log =
                "src_crop value not parsed correctly,  Use string with values of crop location to set the property. e.g. 20:20:40:50";
            conv_info->status = SRC_CROP_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, error: " + std::string(e.what());
            conv_info->status = SRC_CROP_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (conv_info->conv_flag == DEST_CROP) {
        try {
          conv_info->dest_crop =
              sub_root_val.get ("dest_crop", EMPTY_STRING).asString ().c_str ();

          if (conv_info->dest_crop == "") {
            conv_info->conv_log =
                "dest_crop value not parsed correctly, Use string with values of crop location to set the property. e.g. 20:20:40:50";
            conv_info->status = DEST_CROP_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, error: " + std::string(e.what());
            conv_info->status = DEST_CROP_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (conv_info->conv_flag == FLIP_METHOD) {
        try {
          conv_info->flip_method = sub_root_val.get ("flip_method", 0).asUInt ();

          if (conv_info->flip_method > 7) {
            conv_info->conv_log =
                "flip_method value not parsed correctly, Enum value range 0-7";
            conv_info->status = FLIP_METHOD_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, error: " + std::string(e.what());
            conv_info->status = FLIP_METHOD_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
      if (conv_info->conv_flag == INTERPOLATION_METHOD) {
        try {
          conv_info->interpolation_method =
              sub_root_val.get ("interpolation_method", 0).asUInt ();
          if (conv_info->interpolation_method > 6) {
            conv_info->conv_log =
                "interpolation_method value not parsed correctly, Enum value range 0-6";
            conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
          }
        } catch (const std::exception& e) {
            // Error handling: other exceptions
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, error: " + std::string(e.what());
            conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
            conv_info->err_info.code = StatusBadRequest;
            return false;
        }
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
