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

#ifndef _DS_YAML_PARSER_H_
#define _DS_YAML_PARSER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include "nvds_yml_parser.h"
#define _PATH_MAX 1024

typedef enum
{
  ENCODER_TYPE_HW,
  ENCODER_TYPE_SW
} EncHwSwType;

NvDsYamlParserStatus
ds_parse_rtsp_output(GstElement *sink,
  GstRTSPServer *server, GstRTSPMediaFactory *factory,
  gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_enc_config(GstElement *encoder, 
  gchar *cfg_file_path, const char* group);

guint
ds_parse_group_type(gchar *cfg_file_path, const char* group);

guint
ds_parse_enc_type(gchar *cfg_file_path, const char* group);

guint
ds_parse_enc_codec(gchar *cfg_file_path, const char* group);

GString *
ds_parse_file_name(gchar *cfg_file_path, const char* group);

GString *
ds_parse_config_yml_filepath(gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_videotemplate_config(GstElement *vtemplate, 
  gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_ocdr_videotemplate_config(GstElement *vtemplate, 
  gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_nvdsanalytics(GstElement *element, gchar *cfg_file_path, const char* group);

void
create_video_encoder(bool isH264, int enc_type, GstElement** conv_capfilter,
  GstElement** outenc, GstElement** encparse, GstElement** rtppay);

  /** Function to get the absolute path of a file.*/
gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str);

/** Parse preprocess configurations. */
NvDsYamlParserStatus
nvds_parse_preprocess (GstElement *element, gchar* app_cfg_file_path, const char* group);

/** Parse postprocess configurations. */
NvDsYamlParserStatus
nvds_parse_postprocess (GstElement *element, gchar* app_cfg_file_path, const char* group);

/** Parse width and height of nvstreammux. */
void
parse_streammux_width_height_yaml (gint *width, gint *height, gchar *cfg_file_path);

/** Parse type of sink. */
void
parse_sink_type_yaml (gint *type, gchar *cfg_file_path);

/** Parse enc type of sink. */
void
parse_sink_enc_type_yaml (gint *enc_type, gchar *cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif
