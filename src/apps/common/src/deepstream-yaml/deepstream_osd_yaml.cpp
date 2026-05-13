/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
parse_osd_yaml (NvDsOSDConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;

  /** Default values */
  config->draw_text = TRUE;
  config->draw_bbox = TRUE;
  config->draw_mask = FALSE;
  config->mode = MODE_GPU;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml["osd"].begin();
     itr != configyml["osd"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();

    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "process-mode") {
      config->mode =
          (NvOSD_Mode) itr->second.as<int>();
    } else if (paramKey == "border-width") {
      config->border_width = itr->second.as<gint>();
    } else if (paramKey == "text-size") {
      config->text_size = itr->second.as<gint>();
    } else if (paramKey == "text-color") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 4) {
        NVGSTDS_ERR_MSG_V
            ("Color params should be exactly 4 floats {r, g, b, a} between 0 and 1");
        goto done;
      }
      std::vector<int> temp;
      for(int i = 0; i < 4; i++) {
        int temp1 = std::stoi(vec[i]);
        temp.push_back(temp1);
      }
      config->text_color.red = temp[0];
      config->text_color.green = temp[1];
      config->text_color.blue = temp[2];
      config->text_color.alpha = temp[3];
    } else if (paramKey == "text-bg-color") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 4) {
        NVGSTDS_ERR_MSG_V
            ("Color params should be exactly 4 floats {r, g, b, a} between 0 and 1");
        goto done;
      }
      std::vector<int> temp;
      for(int i = 0; i < 4; i++) {
        int temp1 = std::stoi(vec[i]);
        temp.push_back(temp1);
      }
      config->text_bg_color.red = temp[0];
      config->text_bg_color.green = temp[1];
      config->text_bg_color.blue = temp[2];
      config->text_bg_color.alpha = temp[3];

      if (config->text_bg_color.red > 0 || config->text_bg_color.green > 0
          || config->text_bg_color.blue > 0 || config->text_bg_color.alpha > 0)
        config->text_has_bg = TRUE;
    } else if (paramKey == "font") {
      std::string temp = itr->second.as<std::string>();
      config->font = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->font, temp.c_str(), 1023);
    } else if (paramKey == "show-clock") {
      config->enable_clock = itr->second.as<gboolean>();
    } else if (paramKey == "clock-x-offset") {
      config->clock_x_offset = itr->second.as<gint>();
    } else if (paramKey == "clock-y-offset") {
      config->clock_y_offset = itr->second.as<gint>();
    } else if (paramKey == "clock-text-size") {
      config->clock_text_size = itr->second.as<gint>();
    } else if (paramKey == "hw-blend-color-attr") {
      std::string temp = itr->second.as<std::string>();
      config->hw_blend_color_attr = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->hw_blend_color_attr, temp.c_str(), 1023);
    } else if (paramKey == "nvbuf-memory-type") {
      config->nvbuf_memory_type = itr->second.as<guint>();
    } else if (paramKey == "clock-color") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 4) {
        NVGSTDS_ERR_MSG_V
            ("Color params should be exactly 4 floats {r, g, b, a} between 0 and 1");
        goto done;
      }
      std::vector<int> temp;
      for(int i = 0; i < 4; i++) {
        int temp1 = std::stoi(vec[i]);
        temp.push_back(temp1);
      }
      config->clock_color.red = temp[0];
      config->clock_color.green = temp[1];
      config->clock_color.blue = temp[2];
      config->clock_color.alpha = temp[3];
    } else if (paramKey == "gpu-id") {
      config->gpu_id = itr->second.as<guint>();
    } else if (paramKey == "display-text") {
      config->draw_text = itr->second.as<gboolean>();
    } else if (paramKey == "display-bbox") {
      config->draw_bbox = itr->second.as<gboolean>();
    } else if (paramKey == "display-mask") {
      config->draw_mask = itr->second.as<gboolean>();
    } else {
      cout << "[WARNING] Unknown param found in osd: " << paramKey << endl;
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
