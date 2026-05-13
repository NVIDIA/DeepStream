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
nvds_rest_roi_parse (const Json::Value & in, NvDsServerRoiInfo * roi_info)
{
  if (roi_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {
      try {
        std::string root_val = it.key ().asString ().c_str ();
        roi_info->root_key = root_val;

        const Json::Value sub_root_val = in[root_val];      //object values of root_key

        roi_info->stream_id =
            sub_root_val.get ("stream_id", EMPTY_STRING).asString ().c_str ();
        if (roi_info->stream_id == "") {
          roi_info->roi_log = "ROI_UPDATE_FAIL, stream_id value not parsed correctly";
          roi_info->status = ROI_UPDATE_FAIL;
          roi_info->err_info.code = StatusBadRequest;
          return false;
        }
        roi_info->roi_count = sub_root_val.get ("roi_count", 0).asUInt ();
        if (roi_info->roi_count == 0) {
          roi_info->roi_log = "ROI_UPDATE_FAIL, roi id is 0";
          roi_info->status = ROI_UPDATE_FAIL;
          roi_info->err_info.code = StatusBadRequest;
          return false;
        }

        const Json::Value roi_arr = sub_root_val.get ("roi", EMPTY_STRING);
        if (roi_arr == "") {
          roi_info->roi_log = "ROI_UPDATE_FAIL, roi is empty";
          roi_info->status = ROI_UPDATE_FAIL;
          roi_info->err_info.code = StatusBadRequest;
          return false;
        }

        for (guint i = 0; i < roi_arr.size (); i++) {
          RoiDimension roi_dim;

          g_strlcpy (roi_dim.roi_id, roi_arr[i]["roi_id"].asString ().c_str (),
              sizeof (roi_dim.roi_id));
          roi_dim.left = roi_arr[i]["left"].asUInt ();
          roi_dim.top = roi_arr[i]["top"].asUInt ();
          roi_dim.width = roi_arr[i]["width"].asUInt ();
          roi_dim.height = roi_arr[i]["height"].asUInt ();
          roi_info->vect.push_back (roi_dim);
        }
      } catch (const std::exception& e) {
            // Error handling: other exceptions
            roi_info->roi_log = "ROI_UPDATE_FAIL, error: " + std::string(e.what());
            roi_info->status = ROI_UPDATE_FAIL;
            roi_info->err_info.code = StatusBadRequest;
            return false;
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
