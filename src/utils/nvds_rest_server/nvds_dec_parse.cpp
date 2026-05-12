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
