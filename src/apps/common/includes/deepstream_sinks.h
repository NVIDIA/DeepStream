/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_SINKS_H__
#define __NVGSTDS_SINKS_H__

#ifdef __aarch64__
#define IS_TEGRA
#endif

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
  NV_DS_SINK_FAKE = 1,
#ifndef IS_TEGRA
  NV_DS_SINK_RENDER_EGL,
#else
  NV_DS_SINK_RENDER_3D,
#endif
  NV_DS_SINK_ENCODE_FILE,
  NV_DS_SINK_UDPSINK,
  NV_DS_SINK_RENDER_DRM,
  NV_DS_SINK_MSG_CONV_BROKER,
} NvDsSinkType;

typedef enum
{
  NV_DS_CONTAINER_MP4 = 1,
  NV_DS_CONTAINER_MKV
} NvDsContainerType;

typedef enum
{
  NV_DS_ENCODER_H264 = 1,
  NV_DS_ENCODER_H265,
  NV_DS_ENCODER_MPEG4
} NvDsEncoderType;

typedef enum
{
  NV_DS_ENCODER_TYPE_HW,
  NV_DS_ENCODER_TYPE_SW
} NvDsEncHwSwType;

typedef enum
{
    NV_DS_ENCODER_OUTPUT_IO_MODE_MMAP = 2,
    NV_DS_ENCODER_OUTPUT_IO_MODE_DMABUF_IMPORT = 5,
}NvDsEncOutputIOMode;

typedef struct
{
  NvDsSinkType type;
  NvDsContainerType container;
  NvDsEncoderType codec;
  NvDsEncHwSwType enc_type;
  guint compute_hw;
  gint bitrate;
  guint profile;
  gint sync;
  gchar *output_file_path;
  guint gpu_id;
  guint rtsp_port;
  guint udp_port;
  guint64 udp_buffer_size;
  guint iframeinterval;
  guint copy_meta;
  NvDsEncOutputIOMode output_io_mode;
  gint sw_preset;
} NvDsSinkEncoderConfig;

typedef struct
{
  NvDsSinkType type;
  gint width;
  gint height;
  gint sync;
  gboolean qos;
  gboolean qos_value_specified;
  guint gpu_id;
  guint nvbuf_memory_type;
  guint offset_x;
  guint offset_y;
  guint color_range;
  guint conn_id;
  guint plane_id;
  gboolean set_mode;
} NvDsSinkRenderConfig;

typedef struct
{
  gboolean enable;
  /** MsgConv settings */
  gchar*    config_file_path;
  guint     conv_payload_type;
  gchar*    conv_msg2p_lib;
  guint     conv_comp_id;
  gchar*    debug_payload_dir;
  gboolean  multiple_payloads;
  gboolean  conv_msg2p_new_api;
  guint     conv_frame_interval;
  gboolean  conv_dummy_payload;
  gchar     *embedding_filter; 
  /** Broker settings */
  gchar*    proto_lib;
  gchar*    conn_str;
  gchar*    topic;
  gchar*    broker_config_file_path;
  guint     broker_comp_id;
  gboolean  disable_msgconv;
  gint sync;
  gboolean  new_api;
  guint broker_sleep_time;
} NvDsSinkMsgConvBrokerConfig;

typedef struct
{
  gboolean enable;
  guint source_id;
  gboolean link_to_demux;
  gboolean nvdslogger;
  gboolean show_fps;
  gboolean show_latency;
  NvDsSinkType type;
  gint sync;
  NvDsSinkEncoderConfig encoder_config;
  NvDsSinkRenderConfig render_config;
  NvDsSinkMsgConvBrokerConfig msg_conv_broker_config;
} NvDsSinkSubBinConfig;

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *transform;
  GstElement *cap_filter;
  GstElement *enc_caps_filter;
  GstElement *encoder;
  GstElement *codecparse;
  GstElement *mux;
  GstElement *sink;
  GstElement *rtppay;
  gulong sink_buffer_probe;
} NvDsSinkBinSubBin;

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *tee;
  GstElement *nvdslogger;
  GstElement *nvdslogger_queue;

  gint num_bins;
  NvDsSinkBinSubBin sub_bins[MAX_SINK_BINS];
} NvDsSinkBin;

/**
 * Initialize @ref NvDsSinkBin. It creates and adds sink and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_SINK
 *
 * @param[in] num_sub_bins number of sink elements.
 * @param[in] config_array array of pointers of type @ref NvDsSinkSubBinConfig
 *            parsed from configuration file.
 * @param[in] bin pointer to @ref NvDsSinkBin to be filled.
 * @param[in] index id of source element.
 *
 * @return true if bin created successfully.
 */
gboolean create_sink_bin (guint num_sub_bins,
    NvDsSinkSubBinConfig *config_array, NvDsSinkBin *bin, guint index);

void destroy_sink_bin (void);
gboolean create_demux_sink_bin (guint num_sub_bins,
    NvDsSinkSubBinConfig *config_array, NvDsSinkBin *bin, guint index);

void set_rtsp_udp_port_num (guint rtsp_port_num, guint udp_port_num);

#ifdef __cplusplus
}
#endif

#endif
