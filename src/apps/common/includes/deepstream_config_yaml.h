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

#ifndef _NVGSTDS_STREAMMUX_YAML_H_
#define _NVGSTDS_STREAMMUX_YAML_H_


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <yaml-cpp/yaml.h>
#pragma GCC diagnostic pop

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "deepstream_streammux.h"
#include "deepstream_tiled_display.h"
#include "deepstream_osd.h"
#include "deepstream_segvisual.h"
#include "deepstream_image_save.h"
#include "deepstream_c2d_msg.h"
#include "deepstream_sinks.h"
#include "deepstream_sources.h"
#include "deepstream_tracker.h"
#include "deepstream_replay.h"
#include "deepstream_gie.h"
#include "deepstream_preprocess.h"
#include "deepstream_dewarper.h"
#include "deepstream_dsanalytics.h"
#include "deepstream_dsexample.h"

#define _MAX_STR_LENGTH 1024

std::vector<std::string> split_string (std::string input);

gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str);

gboolean
parse_streammux_yaml (NvDsStreammuxConfig *config, gchar *cfg_file_path);

gboolean
parse_tiled_display_yaml (NvDsTiledDisplayConfig *config, gchar *cfg_file_path);

gboolean
parse_osd_yaml (NvDsOSDConfig *config, gchar *cfg_file_path);

gboolean
parse_segvisual_yaml (NvDsSegVisualConfig *config, gchar *cfg_file_path);

gboolean
parse_image_save_yaml (NvDsImageSave *config, gchar *cfg_file_path);

gboolean
parse_msgconsumer_yaml (NvDsMsgConsumerConfig *config, std::string group, gchar *cfg_file_path);

gboolean
parse_msgconv_yaml (NvDsSinkMsgConvBrokerConfig *config, std::string group, gchar *cfg_file_path);

gboolean
parse_sink_yaml (NvDsSinkSubBinConfig *config, std::string group, gchar * cfg_file_path);

gboolean
parse_source_yaml (NvDsSourceConfig *config, std::vector<std::string> headers,
                    std::vector<std::string> source_values,  gchar *cfg_file_path);

gboolean
parse_tracker_yaml (NvDsTrackerConfig *config, gchar *cfg_file_path);

gboolean
parse_replay_yaml (NvDsReplayConfig *config, gchar *cfg_file_path);

gboolean
parse_gie_yaml (NvDsGieConfig *config, std::string group, gchar *cfg_file_path);

gboolean
parse_preprocess_yaml (NvDsPreProcessConfig *config, gchar* cfg_file_path);

gboolean
parse_dewarper_yaml (NvDsDewarperConfig * config, std::string group_str, gchar *cfg_file_path);

gboolean
parse_dsexample_yaml (NvDsDsExampleConfig *config, gchar *cfg_file_path);

gboolean
parse_dsanalytics_yaml (NvDsDsAnalyticsConfig *config, gchar* cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSEXAMPLE_H_ */
