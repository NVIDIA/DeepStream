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

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_sink_yaml (NvDsSinkSubBinConfig *config, std::string group_str, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  config->encoder_config.rtsp_port = 8554;
  config->encoder_config.udp_port = 5000;
  config->encoder_config.codec = NV_DS_ENCODER_H264;
  config->encoder_config.container = NV_DS_CONTAINER_MP4;
  config->encoder_config.compute_hw = 0;
  config->render_config.qos = FALSE;
  config->link_to_demux = FALSE;
  config->msg_conv_broker_config.new_api = FALSE;
  config->msg_conv_broker_config.conv_msg2p_new_api = FALSE;
  config->msg_conv_broker_config.conv_frame_interval = 30;
  config->nvdslogger = FALSE;
  config->show_fps = FALSE;

  if (configyml[group_str]["enable"]) {
    gboolean val= configyml[group_str]["enable"].as<gboolean>();
    if(val == FALSE)
      return TRUE;
  }

  for(YAML::const_iterator itr = configyml[group_str].begin();
     itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();

    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "type") {
      config->type =
          (NvDsSinkType) itr->second.as<int>();
    } else if (paramKey == "link-to-demux") {
      config->link_to_demux = itr->second.as<gboolean>();
    } else if (paramKey == "nvdslogger") {
      config->nvdslogger = itr->second.as<gboolean>();
    } else if (paramKey == "show-fps") {
      config->show_fps = itr->second.as<gboolean>();
    } else if (paramKey == "show-latency") {
      config->show_latency = itr->second.as<gboolean>();
    } else if (paramKey == "width") {
      config->render_config.width = itr->second.as<gint>();
    } else if (paramKey == "height") {
      config->render_config.height = itr->second.as<gint>();
    } else if (paramKey == "qos") {
      config->render_config.qos = itr->second.as<gboolean>();
      config->render_config.qos_value_specified = TRUE;
    } else if (paramKey == "sync") {
      config->sync = itr->second.as<gint>();
    } else if (paramKey == "nvbuf-memory-type") {
      config->render_config.nvbuf_memory_type =
          itr->second.as<guint>();
    } else if (paramKey == "container") {
      config->encoder_config.container =
          (NvDsContainerType) itr->second.as<int>();
    } else if (paramKey == "codec") {
      config->encoder_config.codec =
          (NvDsEncoderType) itr->second.as<int>();
    } else if (paramKey == "compute-hw") {
      config->encoder_config.compute_hw =
          itr->second.as<int>();
    } else if (paramKey == "enc-type") {
      config->encoder_config.enc_type =
          (NvDsEncHwSwType) itr->second.as<int>();
    } else if (paramKey == "bitrate") {
      config->encoder_config.bitrate =
          itr->second.as<gint>();
    } else if (paramKey == "profile") {
      config->encoder_config.profile =
          itr->second.as<guint>();
    } else if (paramKey == "iframeinterval") {
      config->encoder_config.iframeinterval =
          itr->second.as<guint>();
    } else if (paramKey == "output-file") {
      std::string temp = itr->second.as<std::string>();
      config->encoder_config.output_file_path = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->encoder_config.output_file_path, temp.c_str(), 1023);
    } else if (paramKey == "source-id") {
      config->source_id =
          itr->second.as<guint>();
    } else if (paramKey == "rtsp-port") {
      config->encoder_config.rtsp_port =
          itr->second.as<guint>();
    } else if (paramKey == "udp-port") {
      config->encoder_config.udp_port =
          itr->second.as<guint>();
    } else if (paramKey == "udp-buffer-size") {
      config->encoder_config.udp_buffer_size =
          itr->second.as<guint64>();
    } else if (paramKey == "color-range") {
      config->render_config.color_range =
          itr->second.as<guint>();
    } else if (paramKey == "conn-id") {
      config->render_config.conn_id =
          itr->second.as<guint>();
    } else if (paramKey == "plane-id") {
      config->render_config.plane_id =
          itr->second.as<guint>();
    } else if (paramKey == "set-mode") {
      config->render_config.set_mode =
          itr->second.as<gboolean>();
    } else if (paramKey == "gpu-id") {
      config->encoder_config.gpu_id = config->render_config.gpu_id =
          itr->second.as<guint>();
    } else if (paramKey == "msg-conv-config" ||
               paramKey == "msg-conv-payload-type" ||
               paramKey == "msg-conv-msg2p-lib" ||
               paramKey == "msg-conv-comp-id" ||
               paramKey == "debug-payload-dir" ||
               paramKey == "multiple-payloads" ||
               paramKey == "msg-conv-msg2p-new-api" ||
               paramKey == "msg-conv-frame-interval" ||
               paramKey == "msg-conv-dummy-payload") {
      ret = parse_msgconv_yaml (&config->msg_conv_broker_config, group_str, cfg_file_path);
      if (!ret)
        goto done;
    } else if (paramKey == "msg-broker-proto-lib") {
      std::string temp = itr->second.as<std::string>();
      config->msg_conv_broker_config.proto_lib = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->msg_conv_broker_config.proto_lib, temp.c_str(), 1023);
    } else if (paramKey == "msg-broker-conn-str") {
      std::string temp = itr->second.as<std::string>();
      config->msg_conv_broker_config.conn_str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->msg_conv_broker_config.conn_str, temp.c_str(), 1023);
    } else if (paramKey == "topic") {
      std::string temp = itr->second.as<std::string>();
      config->msg_conv_broker_config.topic = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->msg_conv_broker_config.topic, temp.c_str(), 1023);
    } else if (paramKey == "msg-broker-config") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->msg_conv_broker_config.broker_config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->msg_conv_broker_config.broker_config_file_path)) {
        g_printerr ("Error: Could not parse msg-broker-config in sink.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "msg-broker-comp-id") {
      config->msg_conv_broker_config.broker_comp_id =
          itr->second.as<guint>();
    } else if (paramKey == "disable-msgconv") {
      config->msg_conv_broker_config.disable_msgconv =
          itr->second.as<gboolean>();
    } else if (paramKey == "new-api") {
      config->msg_conv_broker_config.new_api =
          itr->second.as<gboolean>();
    } else {
      cout << "[WARNING] Unknown param found in sink: " << paramKey << endl;
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
