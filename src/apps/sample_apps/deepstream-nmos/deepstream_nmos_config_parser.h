/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __DEEPSTREAM_NMOS_CONFIG_PARSER_H__
#define __DEEPSTREAM_NMOS_CONFIG_PARSER_H__

#include "deepstream_nmos_app.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define CONFIG_GROUP_APP "application"
#define CONFIG_GROUP_APP_SEED "seed"
#define CONFIG_GROUP_APP_HOST_NAME "host-name"
#define CONFIG_GROUP_APP_HTTP_PORT "http-port"
#define CONFIG_GROUP_RECEIVER "receiver"
#define CONFIG_GROUP_SENDER "sender"
#define CONFIG_GROUP_PGIE "primary-gie"
#define CONFIG_GROUP_PGIE_CONFIG_FILE "config-file"
#define CONFIG_GROUP_ENABLE "enable"
#define CONFIG_GROUP_PLUGIN_TYPE "plugin-type"
#define CONFIG_GROUP_TYPE "type"
#define CONFIG_GROUP_SDPFILE "sdp-file"
#define CONFIG_GROUP_SINK_SDPFILE "sink-sdp-file"
#define CONFIG_GROUP_SINK_TYPE "sink-type"
#define CONFIG_GROUP_SINK_PAYLOAD_SIZE "payload-size"
#define CONFIG_GROUP_SINK_PACKET_PER_LINE "packets-per-line"
#define CONFIG_GROUP_SINK_SRT_MODE "srt-mode"
#define CONFIG_GROUP_SINK_SRT_LATENCY "srt-latency"
#define CONFIG_GROUP_SINK_SRT_PASSPHRASE "srt-passphrase"
#define CONFIG_GROUP_SINK_SRT_URI "srt-uri"
#define CONFIG_GROUP_SINK_ENCODE_BITRATE "bitrate"
#define CONFIG_GROUP_SINK_ENCODE_IFRAMEINTERVAL "iframeinterval"
#define CONFIG_GROUP_SINK_ENCODE_CAPSFILTER "encode-caps"
#define CONFIG_GROUP_SINK_FLIP_METHOD "flip-method"

gboolean parse_config_file (NvDsNmosAppCtx *appCtx, gchar *cfgFilePath);

gboolean
parse_gie (NvDsNmosAppConfig *appConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath);

gboolean
parse_app (NvDsNmosAppConfig *appConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath);

gboolean
parse_sender (NvDsNmosSinkConfig *sinkConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath);

gboolean
parse_receiver (NvDsNmosSrcConfig *srcConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath);

#ifdef __cplusplus
}
#endif
#endif