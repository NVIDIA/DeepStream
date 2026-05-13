/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>

#define EMPTY_STRING ""

bool
nvds_rest_stream_parse (const Json::Value & in, NvDsServerStreamInfo * stream_info)
{
  if (stream_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {
        try {

        std::string root_val = it.key ().asString ().c_str ();

        const Json::Value sub_root_val = in[root_val];      //object values of root_key
        if (root_val == "key") {
          stream_info->key = in.get ("key", EMPTY_STRING).asString ().c_str ();
        }
        if (root_val == "value" || root_val == "event") {

          for (Json::ValueConstIterator it_sr = sub_root_val.begin ();
              it_sr != sub_root_val.end (); ++it_sr) {

            if (it_sr.key ().asString () == "metadata") {

              const Json::Value metadata_in =
                  sub_root_val[it_sr.key ().asString ().c_str ()];
              stream_info->metadata_resolution =
                  metadata_in.get ("resolution", EMPTY_STRING).asString ().c_str ();
              stream_info->metadata_codec =
                  metadata_in.get ("codec", EMPTY_STRING).asString ().c_str ();
              stream_info->metadata_framerate =
                  metadata_in.get ("framerate", EMPTY_STRING).asString ().c_str ();

            } else {
              stream_info->value_camera_id =
                  sub_root_val.get ("camera_id", EMPTY_STRING).asString ().c_str ();
              stream_info->value_camera_name =
                  sub_root_val.get ("camera_name",
                  EMPTY_STRING).asString ().c_str ();
              stream_info->value_camera_url =
                  sub_root_val.get ("camera_url",
                  EMPTY_STRING).asString ().c_str ();
              stream_info->value_change =
                  sub_root_val.get ("change", EMPTY_STRING).asString ().c_str ();
              stream_info->value_creation_time =
                  sub_root_val.get ("creation_time", EMPTY_STRING).asString ().c_str ();
              if (stream_info->value_camera_url == "") {
                stream_info->status =
                    stream_info->value_change.find ("add") !=
                    std::string::npos ? STREAM_ADD_FAIL : STREAM_REMOVE_FAIL;
                stream_info->stream_log = stream_info->status == STREAM_ADD_FAIL ?
                                          "STREAM_ADD_FAIL, Source url empty" :
                                          "STREAM_REMOVE_FAIL, Source url empty" ;
                stream_info->err_info.code = StatusBadRequest;
                return false;
              }
              if (stream_info->value_camera_id == "") {
                stream_info->status =
                    stream_info->value_change.find ("add") !=
                    std::string::npos ? STREAM_ADD_FAIL : STREAM_REMOVE_FAIL;
                stream_info->stream_log = stream_info->status == STREAM_ADD_FAIL ?
                                          "STREAM_ADD_FAIL, Source id empty" :
                                          "STREAM_REMOVE_FAIL, Source id empty" ;
                stream_info->err_info.code = StatusBadRequest;
                return false;
              }
            }


          }
        }
        if (root_val == "headers") {
          for (Json::ValueConstIterator it_sr = sub_root_val.begin ();
              it_sr != sub_root_val.end (); ++it_sr) {

            stream_info->headers_source =
                sub_root_val.get ("source", EMPTY_STRING).asString ().c_str ();
            stream_info->headers_created_at =
                sub_root_val.get ("created_at", EMPTY_STRING).asString ().c_str ();

          }
        }
      } catch (const std::exception& e) {
            // Error handling: other exceptions
            stream_info->stream_log = "STREAM ADD/REMOVE_FAIL, error: " + std::string(e.what());
            stream_info->status = stream_info->value_change.find ("add") !=
                    std::string::npos ? STREAM_ADD_FAIL : STREAM_REMOVE_FAIL;
            stream_info->err_info.code = StatusBadRequest;
            return false;
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
