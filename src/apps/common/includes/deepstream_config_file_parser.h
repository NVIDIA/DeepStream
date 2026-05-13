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

#ifndef __NVGSTDS_CONFIG_PARSER_H__
#define  __NVGSTDS_CONFIG_PARSER_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "deepstream_config.h"
#include "deepstream_sources.h"
#include "deepstream_preprocess.h"
#include "deepstream_primary_gie.h"
#include "deepstream_audio_classifier.h"
#include "deepstream_tiled_display.h"
#include "deepstream_gie.h"
#include "deepstream_sinks.h"
#include "deepstream_osd.h"
#include "deepstream_segvisual.h"
#include "deepstream_sources.h"
#include "deepstream_dsanalytics.h"
#include "deepstream_dsexample.h"
#include "deepstream_streammux.h"
#include "deepstream_tracker.h"
#include "deepstream_text_embedder.h"
#include "deepstream_visionencoder.h"
#include "deepstream_replay.h"
#include "deepstream_dewarper.h"
#include "deepstream_c2d_msg.h"
#include "deepstream_image_save.h"

#define CONFIG_GROUP_SOURCE_LIST "source-list"
#define CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS "num-source-bins"
#define CONFIG_GROUP_SOURCE_LIST_URI_LIST "list"
/** this vector is one to one mapped with the uri-list/list */
#define CONFIG_GROUP_SOURCE_LIST_SENSOR_ID_LIST "sensor-id-list"
#define CONFIG_GROUP_SOURCE_LIST_SENSOR_NAME_LIST "sensor-name-list"

/** additional configs to support nvmultiurisrcbin usage */
#define CONFIG_GROUP_SOURCE_LIST_USE_NVMULTIURISRCBIN "use-nvmultiurisrcbin"
#define CONFIG_GROUP_SOURCE_LIST_DUMMY_PAYLOAD "dummy-payload"
#define CONFIG_GROUP_SOURCE_LIST_CUSTOM_TS_TO_RFC "custom-ts-to-rfc"
#define CONFIG_GROUP_SOURCE_LIST_BATCHED_SENSOR_NAME "batched-sensor-name"
#define CONFIG_GROUP_SOURCE_LIST_STREAM_NAME_DISPLAY "stream-name-display"
#define CONFIG_GROUP_SOURCE_LIST_MAX_BATCH_SIZE "max-batch-size"
#define CONFIG_GROUP_SOURCE_LIST_HTTP_IP "http-ip"
#define CONFIG_GROUP_SOURCE_LIST_HTTP_PORT "http-port"
#define CONFIG_GROUP_SOURCE_EXTRACT_SEI_TYPE5_DATA "extract-sei-type5-data"
#define CONFIG_GROUP_SOURCE_SEI_UUID "sei-uuid"
#define CONFIG_GROUP_SOURCE_LIST_LOW_LATENCY_MODE "low-latency-mode"

/** Error propagation configs for nvmultiurisrcbin */
#define CONFIG_GROUP_SOURCE_LIST_ENABLE_ERROR_PROPAGATION "enable-error-propagation"
#define CONFIG_GROUP_SOURCE_LIST_PROTO_LIB "proto-lib"
#define CONFIG_GROUP_SOURCE_LIST_CONN_STR "conn-str"
#define CONFIG_GROUP_SOURCE_LIST_TOPIC "topic"


#define CONFIG_GROUP_SOURCE_ALL "source-attr-all"

#define CONFIG_GROUP_SOURCE "source"
#define CONFIG_GROUP_OSD "osd"
#define CONFIG_GROUP_SEGVISUAL "segvisual"
#define CONFIG_GROUP_PREPROCESS "pre-process"
#define CONFIG_GROUP_SECONDARY_PREPROCESS "secondary-pre-process"
#define CONFIG_GROUP_PRIMARY_GIE "primary-gie"
#define CONFIG_GROUP_SECONDARY_GIE "secondary-gie"
#define CONFIG_GROUP_TRACKER "tracker"

#define CONFIG_GROUP_TEXT_EMBEDDER "text-embedder"

#define CONFIG_GROUP_VISIONENCODER "visionencoder"

#define CONFIG_GROUP_REPLAY "replay"
#define CONFIG_GROUP_SINK "sink"
#define CONFIG_GROUP_TILED_DISPLAY "tiled-display"
#define CONFIG_GROUP_DSANALYTICS "nvds-analytics"
#define CONFIG_GROUP_DSEXAMPLE "ds-example"
#define CONFIG_GROUP_STREAMMUX "streammux"
#define CONFIG_GROUP_DEWARPER "dewarper"
#define CONFIG_GROUP_MSG_CONVERTER  "message-converter"
#define CONFIG_GROUP_MSG_CONSUMER  "message-consumer"
#define CONFIG_GROUP_IMG_SAVE "img-save"
#define CONFIG_GROUP_AUDIO_TRANSFORM "audio-transform"
#define CONFIG_GROUP_AUDIO_CLASSIFIER "audio-classifier"

#define CONFIG_GROUP_SOURCE_GPU_ID "gpu-id"
#define CONFIG_GROUP_SOURCE_TYPE "type"
#define CONFIG_GROUP_SOURCE_CUDA_MEM_TYPE "nvbuf-memory-type"
#define CONFIG_GROUP_SOURCE_CAMERA_WIDTH "camera-width"
#define CONFIG_GROUP_SOURCE_CAMERA_HEIGHT "camera-height"
#define CONFIG_GROUP_SOURCE_CAMERA_FPS_N "camera-fps-n"
#define CONFIG_GROUP_SOURCE_CAMERA_FPS_D "camera-fps-d"
#define CONFIG_GROUP_SOURCE_CAMERA_CSI_SID "camera-csi-sensor-id"
#define CONFIG_GROUP_SOURCE_CAMERA_V4L2_DEVNODE "camera-v4l2-dev-node"
#define CONFIG_GROUP_SOURCE_URI "uri"
#define CONFIG_GROUP_SOURCE_LIVE_SOURCE "live-source"
#define CONFIG_GROUP_SOURCE_FILE_LOOP "file-loop"
#define CONFIG_GROUP_SOURCE_LATENCY "latency"
#define CONFIG_GROUP_SOURCE_NUM_SOURCES "num-sources"
#define CONFIG_GROUP_SOURCE_INTRA_DECODE "intra-decode-enable"
#define CONFIG_GROUP_SOURCE_DEC_SKIP_FRAMES "dec-skip-frames"
#define CONFIG_GROUP_SOURCE_NUM_DECODE_SURFACES "num-decode-surfaces"
#define CONFIG_GROUP_SOURCE_NUM_EXTRA_SURFACES "num-extra-surfaces"
#define CONFIG_GROUP_SOURCE_DROP_FRAME_INTERVAL "drop-frame-interval"
#define CONFIG_GROUP_SOURCE_DROP_ON_LATENCY "drop-on-latency"
#define CONFIG_GROUP_SOURCE_CAMERA_ID "camera-id"
#define CONFIG_GROUP_SOURCE_ID "source-id"
#define CONFIG_GROUP_SOURCE_SELECT_RTP_PROTOCOL "select-rtp-protocol"
#define CONFIG_GROUP_SOURCE_LEAKY "leaky"
#define CONFIG_GROUP_SOURCE_MAX_SIZE_BUFFERS "max-size-buffers"
#define CONFIG_GROUP_SOURCE_RTSP_RECONNECT_INTERVAL_SEC "rtsp-reconnect-interval-sec"
#define CONFIG_GROUP_SOURCE_RTSP_RECONNECT_ATTEMPTS "rtsp-reconnect-attempts"
#define CONFIG_GROUP_SOURCE_RTSP_INIT_RECONNECT_INTERVAL_SEC "init-rtsp-reconnect-interval-sec"
#define CONFIG_GROUP_SOURCE_RTSP_BUFFER_MODE "buffer-mode"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_ENABLE "smart-record"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_DIRPATH "smart-rec-dir-path"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_FILE_PREFIX "smart-rec-file-prefix"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE_LEGACY "smart-rec-video-cache"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE "smart-rec-cache"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_CONTAINER "smart-rec-container"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_START_TIME "smart-rec-start-time"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_DEFAULT_DURATION "smart-rec-default-duration"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_DURATION "smart-rec-duration"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_INTERVAL "smart-rec-interval"

/**
 * Function to parse class label file. Parses the labels into a 2D-array of
 * strings. Refer the SDK documentation for format of the labels file.
 *
 * @param[in] config pointer to @ref NvDsGieConfig
 *
 * @return true if file parsed successfully else returns false.
 */
gboolean
parse_labels_file (NvDsGieConfig *config);

/**
 * Function to read properties of source element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsDewarperConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_DEWARPER
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_dewarper (NvDsDewarperConfig * config, GKeyFile * key_file, gchar *cfg_file_path, gchar * group);

/**
 * Function to read properties of source element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsSourceConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_SOURCE
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_source (NvDsSourceConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path);

/**
 * Function to read properties of NvSegVisual element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsSegVisualConfig
 * @param[in] key_file pointer to file having key value pairs.
 *
 * @return true if parsed successfully.
 */
gboolean parse_segvisual (NvDsSegVisualConfig * config, GKeyFile * key_file);

/**
 * Function to read properties of OSD element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsOSDConfig
 * @param[in] key_file pointer to file having key value pairs.
 *
 * @return true if parsed successfully.
 */
gboolean parse_osd (NvDsOSDConfig * config, GKeyFile * key_file);

/**
 * Function to read properties of nvdspreprocess element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsPreProcessConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_PREPROCESS and
 *            @ref CONFIG_GROUP_SECONDARY_PREPROCESS
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_preprocess (NvDsPreProcessConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path);

/**
 * Function to read properties of infer element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsGieConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_PRIMARY_GIE and
 *            @ref CONFIG_GROUP_SECONDARY_GIE
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_gie (NvDsGieConfig * config, GKeyFile * key_file, gchar * group,
    gchar * cfg_file_path);

/**
 * Function to read properties of tracker element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsTrackerConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_tracker (NvDsTrackerConfig * config, GKeyFile * key_file, gchar * cfg_file_path);

/**
 * Function to read properties of text embedder element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsTextEmbedderConfig
 * @param[in] key_file pointer to file having key value pairs.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_text_embedder (NvDsTextEmbedderConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path);

/**
 * Function to read properties of visionencoder element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsVisionEncoderConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_VISIONENCODER
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_visionencoder (NvDsVisionEncoderConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path);

/**
 * Function to read properties of replay element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsReplayConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_replay (NvDsReplayConfig * config, GKeyFile * key_file, gchar * cfg_file_path);

/**
 * Function to read properties of sink element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsSinkSubBinConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_SINK
 *
 * @return true if parsed successfully.
 */
gboolean
parse_sink (NvDsSinkSubBinConfig * config, GKeyFile * key_file, gchar * group, gchar * cfg_file_path);

/**
 * Function to read properties of tiler element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsTiledDisplayConfig
 * @param[in] key_file pointer to file having key value pairs.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_tiled_display (NvDsTiledDisplayConfig * config, GKeyFile * key_file);

/**
 * Function to read properties of dsanalytics element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsDsAnalyticsConfig
 * @param[in] key_file pointer to file having key value pairs.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_dsanalytics (NvDsDsAnalyticsConfig * config, GKeyFile * key_file, gchar* cfg_file_path);

/**
 * Function to read properties of dsexample element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsDsExampleConfig
 * @param[in] key_file pointer to file having key value pairs.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_dsexample (NvDsDsExampleConfig * config, GKeyFile * key_file);

/**
 * Function to read properties of streammux element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsStreammuxConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_streammux (NvDsStreammuxConfig * config, GKeyFile * key_file, gchar * cfg_file_path);

/**
 * Function to read properties of message converter element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsSinkMsgConvBrokerConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_MSG_CONVERTER
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_msgconv (NvDsSinkMsgConvBrokerConfig *config, GKeyFile *key_file, gchar *group, gchar *cfg_file_path);

/**
 * Function to read properties of message consumer element from configuration file.
 *
 * @param[in] config pointer to @ref NvDsMsgConsumerConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_MSG_CONSUMER
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_msgconsumer (NvDsMsgConsumerConfig *config, GKeyFile *key_file, gchar *group, gchar *cfg_file_path);

/**
 * Function to read properties of image save from configuration file.
 *
 * @param[in] config pointer to @ref NvDsMsgConsumerConfig
 * @param[in] key_file pointer to file having key value pairs.
 * @param[in] group name of property group @ref CONFIG_GROUP_MSG_CONSUMER
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_image_save (NvDsImageSave *config, GKeyFile *key_file,
                   gchar *group, gchar *cfg_file_path);

/**
 * Utility function to convert relative path in configuration file
 * with absolute path.
 *
 * @param[in] cfg_file_path path of configuration file.
 * @param[in] file_path relative path of file.
 */
gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar * file_path);

#ifdef __cplusplus
}
#endif

#endif
