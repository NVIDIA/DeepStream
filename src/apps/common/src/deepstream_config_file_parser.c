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

#include "deepstream_common.h"
#include "deepstream_config_file_parser.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

GST_DEBUG_CATEGORY (APP_CFG_PARSER_CAT);

#define CONFIG_GPU_ID "gpu-id"
#define CONFIG_NVBUF_MEMORY_TYPE "nvbuf-memory-type"
#define CONFIG_GROUP_ENABLE "enable"

#define CONFIG_GROUP_SOURCE_TYPE "type"
#define CONFIG_GROUP_SOURCE_CAMERA_WIDTH "camera-width"
#define CONFIG_GROUP_SOURCE_CAMERA_HEIGHT "camera-height"
#define CONFIG_GROUP_SOURCE_CAMERA_FPS_N "camera-fps-n"
#define CONFIG_GROUP_SOURCE_CAMERA_FPS_D "camera-fps-d"
#define CONFIG_GROUP_SOURCE_CAMERA_CSI_SID "camera-csi-sensor-id"
#define CONFIG_GROUP_SOURCE_CAMERA_V4L2_DEVNODE "camera-v4l2-dev-node"
#define CONFIG_GROUP_SOURCE_URI "uri"
#define CONFIG_GROUP_SOURCE_LATENCY "latency"
#define CONFIG_GROUP_SOURCE_NUM_SOURCES "num-sources"
#define CONFIG_GROUP_SOURCE_INTRA_DECODE "intra-decode-enable"
#define CONFIG_GROUP_SOURCE_LOW_LATENCY_DECODE "low-latency-decode"
#define CONFIG_GROUP_SOURCE_NUM_DECODE_SURFACES "num-decode-surfaces"
#define CONFIG_GROUP_SOURCE_NUM_EXTRA_SURFACES "num-extra-surfaces"
#define CONFIG_GROUP_SOURCE_DROP_FRAME_INTERVAL "drop-frame-interval"
#define CONFIG_GROUP_SOURCE_DROP_ON_LATENCY "drop-on-latency"
#define CONFIG_GROUP_SOURCE_EXTRACT_SEI_TYPE5_DATA "extract-sei-type5-data"
#define CONFIG_GROUP_SOURCE_CAMERA_ID "camera-id"
#define CONFIG_GROUP_SOURCE_ID "source-id"
#define CONFIG_GROUP_SOURCE_SELECT_RTP_PROTOCOL "select-rtp-protocol"
#define CONFIG_GROUP_SOURCE_SET_RTP_PROTOCOL "set-rtp-protocol"
#define CONFIG_GROUP_SOURCE_LEAKY "leaky"
#define CONFIG_GROUP_SOURCE_MAX_SIZE_BUFFERS "max-size-buffers"
#define CONFIG_GROUP_SOURCE_BUFFER_MODE "buffer-mode"
#define CONFIG_GROUP_SOURCE_RTSP_RECONNECT_INTERVAL_SEC "rtsp-reconnect-interval-sec"
#define CONFIG_GROUP_SOURCE_RTSP_INIT_RECONNECT_INTERVAL_SEC "init-rtsp-reconnect-interval-sec"
#define CONFIG_GROUP_SOURCE_SIMULATE_FPS_INTERVAL_MS "simulate-fps-interval-ms"
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
#define CONFIG_GROUP_SOURCE_ALSA_DEVICE "alsa-device"
#define CONFIG_GROUP_SOURCE_UDP_BUFFER_SIZE "udp-buffer-size"
#define CONFIG_GROUP_SOURCE_SENSORID_PADID_MAPPING "sensorID-padID-mapping"
#define CONFIG_GROUP_SOURCE_VIDEO_FORMAT "video-format"
#if defined(__aarch64__) && !defined(AARCH64_IS_SBSA)
/* copy-hw as VIC applicable only for Jetson */
#define CONFIG_GROUP_SOURCE_NVVIDEOCONVERT_COPY_HW "copy-hw"
#endif

#define CONFIG_GROUP_STREAMMUX_ENABLE_PADDING "enable-padding"
#define CONFIG_GROUP_STREAMMUX_COMPUTE_HW "compute-hw"
#define CONFIG_GROUP_STREAMMUX_SORT_BATCH "sort-batch"
#define CONFIG_GROUP_STREAMMUX_CACHE_BUFFER "cache-buffer"
#define CONFIG_GROUP_STREAMMUX_CACHE_BUFFER_TIMEOUT "cache-buffer-timeout"
#define CONFIG_GROUP_STREAMMUX_SYNC_INPUTS_NTP "sync-inputs-ntp"
#define CONFIG_GROUP_STREAMMUX_WIDTH "width"
#define CONFIG_GROUP_STREAMMUX_HEIGHT "height"
#define CONFIG_GROUP_STREAMMUX_BUFFER_POOL_SIZE "buffer-pool-size"
#define CONFIG_GROUP_STREAMMUX_BATCH_SIZE "batch-size"
#define CONFIG_GROUP_STREAMMUX_BATCHED_PUSH_TIMEOUT "batched-push-timeout"
#define CONFIG_GROUP_STREAMMUX_LIVE_SOURCE "live-source"
#define CONFIG_GROUP_STREAMMUX_ATTACH_SYS_TS_AS_NTP "attach-sys-ts-as-ntp"
#define CONFIG_GROUP_STREAMMUX_FRAME_NUM_RESET_ON_STREAM_RESET "frame-num-reset-on-stream-reset"
#define CONFIG_GROUP_STREAMMUX_FRAME_NUM_RESET_ON_EOS "frame-num-reset-on-eos"
#define CONFIG_GROUP_STREAMMUX_FRAME_DURATION "frame-duration"
#define CONFIG_GROUP_STREAMMUX_CONFIG_FILE_PATH "config-file-path"
#define CONFIG_GROUP_STREAMMUX_SYNC_INPUTS "sync-inputs"
#define CONFIG_GROUP_STREAMMUX_MAX_LATENCY "max-latency"
#define CONFIG_GROUP_STREAMMUX_ASYNC_PROCESS "async-process"
#define CONFIG_GROUP_STREAMMUX_DROP_PIPELINE_EOS "drop-pipeline-eos"
#define CONFIG_GROUP_STREAMMUX_NUM_SURFACES_PER_FRAME "num-surfaces-per-frame"
#define CONFIG_GROUP_STREAMMUX_EXTRACT_SEI_TYPE5_DATA "extract-sei-type5-data"
#define CONFIG_GROUP_STREAMMUX_EXTRACT_SEI_SIM_TIME "extract-sei-sim-time"
#define CONFIG_GROUP_STREAMMUX_ALIGN_FIRST_BUFFER "align-first-buffer"
#define CONFIG_GROUP_STREAMMUX_FAILSAFE_FLUSH_COUNT "failsafe-flush-count"
#define CONFIG_GROUP_STREAMMUX_DROP_BACKWARD_SEI "drop-backward-sei"
#define CONFIG_GROUP_SEGVISUAL_WIDTH "width"
#define CONFIG_GROUP_SEGVISUAL_HEIGHT "height"
#define CONFIG_GROUP_SEGVISUAL_BATCH_SIZE "batch-size"
#define CONFIG_GROUP_OSD_MODE "process-mode"
#define CONFIG_GROUP_OSD_BORDER_WIDTH "border-width"
#define CONFIG_GROUP_OSD_BORDER_COLOR "border-color"
#define CONFIG_GROUP_OSD_TEXT_SIZE "text-size"
#define CONFIG_GROUP_OSD_TEXT_COLOR "text-color"
#define CONFIG_GROUP_OSD_TEXT_BG_COLOR "text-bg-color"
#define CONFIG_GROUP_OSD_FONT "font"
#define CONFIG_GROUP_OSD_HW_BLEND_COLOR_ATTR "hw-blend-color-attr"
#define CONFIG_GROUP_OSD_CLOCK_ENABLE "show-clock"
#define CONFIG_GROUP_OSD_CLOCK_X_OFFSET "clock-x-offset"
#define CONFIG_GROUP_OSD_CLOCK_Y_OFFSET "clock-y-offset"
#define CONFIG_GROUP_OSD_CLOCK_TEXT_SIZE "clock-text-size"
#define CONFIG_GROUP_OSD_CLOCK_COLOR "clock-color"
#define CONFIG_GROUP_OSD_SHOW_TEXT "display-text"
#define CONFIG_GROUP_OSD_SHOW_BBOX "display-bbox"
#define CONFIG_GROUP_OSD_SHOW_MASK "display-mask"

#define CONFIG_GROUP_DEWARPER_CONFIG_FILE "config-file"
#define CONFIG_GROUP_DEWARPER_SOURCE_ID "source-id"
#define CONFIG_GROUP_DEWARPER_NUM_SURFACES_PER_FRAME "num-surfaces-per-frame"
#define CONFIG_GROUP_DEWARPER_NUM_OUTPUT_BUFFERS "num-output-buffers"

#define CONFIG_GROUP_PREPROCESS_CONFIG_FILE "config-file"

#define CONFIG_GROUP_GIE_INPUT_TENSOR_META "input-tensor-meta"
#define CONFIG_GROUP_GIE_BATCH_SIZE "batch-size"
#define CONFIG_GROUP_GIE_MODEL_ENGINE "model-engine-file"
#define CONFIG_GROUP_GIE_CONFIG_FILE "config-file"
#define CONFIG_GROUP_GIE_LABEL "labelfile-path"
#define CONFIG_GROUP_GIE_PLUGIN_TYPE "plugin-type"
#define CONFIG_GROUP_GIE_UNIQUE_ID "gie-unique-id"
#define CONFIG_GROUP_GIE_ID_FOR_OPERATION "operate-on-gie-id"
#define CONFIG_GROUP_GIE_BBOX_BORDER_COLOR "bbox-border-color"
#define CONFIG_GROUP_GIE_BBOX_BG_COLOR "bbox-bg-color"
#define CONFIG_GROUP_GIE_CLASS_IDS_FOR_OPERATION "operate-on-class-ids"
#define CONFIG_GROUP_GIE_INTERVAL "interval"
#define CONFIG_GROUP_GIE_RAW_OUTPUT_DIR "infer-raw-output-dir"
#define CONFIG_GROUP_GIE_AUDIO_TRANSFORM "audio-transform"
#define CONFIG_GROUP_GIE_FRAME_SIZE "audio-framesize"
#define CONFIG_GROUP_GIE_HOP_SIZE "audio-hopsize"
/** desired input rate of audio to nvinferaudio in samples per second */
#define CONFIG_GROUP_GIE_AUDIO_RATE "audio-input-rate"

// Vision encoder config keys
#define CONFIG_GROUP_GIE_MODEL_VARIANT "model-variant"
#define CONFIG_GROUP_GIE_DEVICE "device"
#define CONFIG_GROUP_GIE_MIN_CROP_SIZE "min-crop-size"
#define CONFIG_GROUP_GIE_VERBOSE "verbose"
#define CONFIG_GROUP_GIE_BACKEND "backend"
#define CONFIG_GROUP_GIE_TRITON_URL "triton-url"
#define CONFIG_GROUP_GIE_TRITON_MODEL "triton-model"

#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_ALGORITHM "tracker-algorithm"
#define CONFIG_GROUP_TRACKER_IOU_THRESHOLD "iou-threshold"
#define CONFIG_GROUP_TRACKER_SURFACE_TYPE "tracker-surface-type"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_TRACKING_SURFACE_TYPE "tracking-surface-type"
#define CONFIG_GROUP_TRACKER_DISPLAY_TRACKING_ID "display-tracking-id"
#define CONFIG_GROUP_TRACKER_TRACKING_ID_RESET_MODE "tracking-id-reset-mode"
#define CONFIG_GROUP_TRACKER_INPUT_TENSOR_META "input-tensor-meta"
#define CONFIG_GROUP_TRACKER_TENSOR_META_GIE_ID "tensor-meta-gie-id"
#define CONFIG_GROUP_TRACKER_COMPUTE_HW "compute-hw"
#define CONFIG_GROUP_TRACKER_USER_META_POOL_SIZE "user-meta-pool-size"
#define CONFIG_GROUP_TRACKER_SUB_BATCHES "sub-batches"
#define CONFIG_GROUP_TRACKER_SUB_BATCH_ERR_RECOVERY_TRIAL_CNT "sub-batch-err-recovery-trial-cnt"
#define CONFIG_GROUP_TRACKER_OPERATE_ON_CLASS_IDS "operate-on-class-ids"

#define CONFIG_GROUP_VISIONENCODER_MODEL_VARIANT "model-variant"
#define CONFIG_GROUP_VISIONENCODER_BATCH_SIZE "batch-size"
#define CONFIG_GROUP_VISIONENCODER_DEVICE "device"
#define CONFIG_GROUP_VISIONENCODER_MIN_CROP_SIZE "min-crop-size"
#define CONFIG_GROUP_VISIONENCODER_VERBOSE "verbose"
#define CONFIG_GROUP_VISIONENCODER_BACKEND "backend"
#define CONFIG_GROUP_VISIONENCODER_TRITON_URL "triton-url"
#define CONFIG_GROUP_VISIONENCODER_TRITON_MODEL "triton-model"
#define CONFIG_GROUP_VISIONENCODER_SKIP_INTERVAL "skip-interval"
#define CONFIG_GROUP_VISIONENCODER_EMBEDDING_CLASSES "embedding-classes"
#define CONFIG_GROUP_VISIONENCODER_ONNX_MODEL "onnx-model"
#define CONFIG_GROUP_VISIONENCODER_TENSORRT_ENGINE "tensorrt-engine"

#define CONFIG_GROUP_TEXT_EMBEDDER_MODEL_NAME "model-name"
#define CONFIG_GROUP_TEXT_EMBEDDER_ONNX_MODEL_PATH "onnx-model-path"
#define CONFIG_GROUP_TEXT_EMBEDDER_TOKENIZER_DIR "tokenizer-dir"

#define CONFIG_GROUP_REPLAY_LABEL_DIR "label-dir"
#define CONFIG_GROUP_REPLAY_FILE_NAMES "file-names"
#define CONFIG_GROUP_REPLAY_MAX_FRAME_NUMS "max-frame-nums"
#define CONFIG_GROUP_REPLAY_LABEL_WIDTH "label-width"
#define CONFIG_GROUP_REPLAY_LABEL_HEIGHT "label-height"
#define CONFIG_GROUP_REPLAY_INTERVAL "interval"

#define CONFIG_GROUP_SINK_TYPE "type"
#define CONFIG_GROUP_SINK_WIDTH "width"
#define CONFIG_GROUP_SINK_HEIGHT "height"
#define CONFIG_GROUP_SINK_SYNC "sync"
#define CONFIG_GROUP_SINK_QOS "qos"
#define CONFIG_GROUP_SINK_CONTAINER "container"
#define CONFIG_GROUP_SINK_CODEC "codec"
#define CONFIG_GROUP_SINK_ENC_TYPE "enc-type"
#define CONFIG_GROUP_SINK_BITRATE "bitrate"
#define CONFIG_GROUP_SINK_PROFILE "profile"
#define CONFIG_GROUP_SINK_IFRAMEINTERVAL "iframeinterval"
#define CONFIG_GROUP_SINK_COPY_META "copy-meta"
#define CONFIG_GROUP_SINK_OUTPUT_IO_MODE "output-io-mode"
#define CONFIG_GROUP_SINK_SW_PRESET "sw-preset"
#define CONFIG_GROUP_SINK_OUTPUT_FILE "output-file"
#define CONFIG_GROUP_SINK_SOURCE_ID "source-id"
#define CONFIG_GROUP_SINK_RTSP_PORT "rtsp-port"
#define CONFIG_GROUP_SINK_UDP_PORT "udp-port"
#define CONFIG_GROUP_SINK_UDP_BUFFER_SIZE "udp-buffer-size"
#define CONFIG_GROUP_SINK_COLOR_RANGE "color-range"
#define CONFIG_GROUP_SINK_CONN_ID "conn-id"
#define CONFIG_GROUP_SINK_PLANE_ID "plane-id"
#define CONFIG_GROUP_SINK_SET_MODE "set-mode"
#define CONFIG_GROUP_SINK_ONLY_FOR_DEMUX "link-to-demux"
#define CONFIG_GROUP_SINK_COMPUTE_HW "compute-hw"
#define CONFIG_GROUP_SINK_NVDSLOGGER "nvdslogger"
#define CONFIG_GROUP_SINK_SHOW_FPS "show-fps"
#define CONFIG_GROUP_SINK_SHOW_LATENCY "show-latency"

#define CONFIG_GROUP_SINK_MSG_CONV_CONFIG "msg-conv-config"
#define CONFIG_GROUP_SINK_MSG_CONV_PAYLOAD_TYPE "msg-conv-payload-type"
#define CONFIG_GROUP_SINK_MSG_CONV_MSG2P_LIB "msg-conv-msg2p-lib"
#define CONFIG_GROUP_SINK_MSG_CONV_COMP_ID "msg-conv-comp-id"
#define CONFIG_GROUP_SINK_MSG_CONV_DEBUG_PAYLOAD_DIR "debug-payload-dir"
#define CONFIG_GROUP_SINK_MSG_CONV_MULTIPLE_PAYLOADS "multiple-payloads"
#define CONFIG_GROUP_SINK_MSG_CONV_MSG2P_NEW_API "msg-conv-msg2p-new-api"
#define CONFIG_GROUP_SINK_MSG_CONV_FRAME_INTERVAL "msg-conv-frame-interval"
#define CONFIG_GROUP_SINK_MSG_CONV_DUMMY_PAYLOAD "msg-conv-dummy-payload"
#define CONFIG_GROUP_SINK_MSG_CONV_EMBEDDING_FILTER "msg-conv-embedding-filter"

#define CONFIG_GROUP_SINK_MSG_BROKER_PROTO_LIB "msg-broker-proto-lib"
#define CONFIG_GROUP_SINK_MSG_BROKER_CONN_STR "msg-broker-conn-str"
#define CONFIG_GROUP_SINK_MSG_BROKER_TOPIC "topic"
#define CONFIG_GROUP_SINK_MSG_BROKER_CONFIG_FILE "msg-broker-config"
#define CONFIG_GROUP_SINK_MSG_BROKER_COMP_ID "msg-broker-comp-id"
#define CONFIG_GROUP_SINK_MSG_BROKER_DISABLE_MSG_CONVERTER "disable-msgconv"
#define CONFIG_GROUP_SINK_MSG_BROKER_NEW_API "new-api"
#define CONFIG_GROUP_SINK_MSG_BROKER_SLEEP_TIME "sleep-time"

#define CONFIG_GROUP_MSG_CONSUMER_CONFIG "config-file"
#define CONFIG_GROUP_MSG_CONSUMER_PROTO_LIB "proto-lib"
#define CONFIG_GROUP_MSG_CONSUMER_CONN_STR "conn-str"
#define CONFIG_GROUP_MSG_CONSUMER_SENSOR_LIST_FILE "sensor-list-file"
#define CONFIG_GROUP_MSG_CONSUMER_SUBSCRIBE_TOPIC_LIST "subscribe-topic-list"

#define CONFIG_GROUP_TILED_DISPLAY_ROWS "rows"
#define CONFIG_GROUP_TILED_DISPLAY_COLUMNS "columns"
#define CONFIG_GROUP_TILED_DISPLAY_WIDTH "width"
#define CONFIG_GROUP_TILED_DISPLAY_HEIGHT "height"
#define CONFIG_GROUP_TILED_COMPUTE_HW "compute-hw"
#define CONFIG_GROUP_TILED_DISPLAY_BUFFER_POOL_SIZE "buffer-pool-size"
#define CONFIG_GROUP_TILED_DISPLAY_SQUARE_SEQ_GRID "square-seq-grid"

#define CONFIG_GROUP_DSANALYTICS_CONFIG_FILE "config-file"
#define CONFIG_GROUP_IMG_SAVE_OUTPUT_FOLDER_PATH "output-folder-path"
#define CONFIG_GROUP_IMG_SAVE_FULL_FRAME_IMG_SAVE "save-img-full-frame"
#define CONFIG_GROUP_IMG_SAVE_CROPPED_OBJECT_IMG_SAVE "save-img-cropped-obj"
#define CONFIG_GROUP_IMG_SAVE_CSV_TIME_RULES_PATH "frame-to-skip-rules-path"
#define CONFIG_GROUP_IMG_SAVE_SECOND_TO_SKIP_INTERVAL "second-to-skip-interval"
#define CONFIG_GROUP_IMG_SAVE_MIN_CONFIDENCE "min-confidence"
#define CONFIG_GROUP_IMG_SAVE_MAX_CONFIDENCE "max-confidence"
#define CONFIG_GROUP_IMG_SAVE_MIN_BOX_WIDTH "min-box-width"
#define CONFIG_GROUP_IMG_SAVE_MIN_BOX_HEIGHT "min-box-height"

// To add configuration parsing for any element, you need to:
// 1. Define a group name and set of key strings for the config options
// 2. Create a function to parse these configs (refer parse_dsexample)
// 3. Call this function in

// Add group name for set of configs of dsexample element
#define CONFIG_GROUP_DSEXAMPLE "ds-example"
// Refer to gst-dsexample element source code for the meaning of these
// configs
#define CONFIG_GROUP_DSEXAMPLE_FULL_FRAME "full-frame"
#define CONFIG_GROUP_DSEXAMPLE_PROCESSING_WIDTH "processing-width"
#define CONFIG_GROUP_DSEXAMPLE_PROCESSING_HEIGHT "processing-height"
#define CONFIG_GROUP_DSEXAMPLE_BLUR_OBJECTS "blur-objects"
#define CONFIG_GROUP_DSEXAMPLE_UNIQUE_ID "unique-id"
#define CONFIG_GROUP_DSEXAMPLE_GPU_ID "gpu-id"
#define CONFIG_GROUP_DSEXAMPLE_BATCH_SIZE "batch-size"

#define CHECK_ERROR(error) \
    if (error) { \
        GST_CAT_ERROR (APP_CFG_PARSER_CAT, "%s", error->message); \
        goto done; \
    }

#define N_DECODE_SURFACES 16
#define N_EXTRA_SURFACES 1

#define SEG_OUTPUT_WIDTH 1280
#define SEG_OUTPUT_HEIGHT 720

gchar *
get_absolute_file_path (gchar * cfg_file_path, gchar * file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }
  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

/**
 * Function to parse class label file. Parses the labels into a 2D-array of
 * strings. Refer the SDK documentation for format of the labels file.
 *
 * @param[in] config pointer to @ref NvDsGieConfig
 *
 * @return true if file parsed successfully else returns false.
 */
gboolean
parse_labels_file (NvDsGieConfig * config)
{
  GList *labels_list = NULL;
  GList *label_outputs_length_list = NULL;
  GIOChannel *label_file = NULL;
  GError *err = NULL;
  gsize term_pos;
  GIOStatus status;
  gboolean ret = FALSE;
  GList *labels_iter;
  GList *label_outputs_length_iter;
  guint j;

  label_file =
      g_io_channel_new_file (GET_FILE_PATH (config->label_file_path), "r",
      &err);

  if (err) {
    NVGSTDS_ERR_MSG_V ("Failed to open label file '%s':%s",
        config->label_file_path, err->message);
    goto done;
  }

  /* Iterate over each line */
  do {
    gchar *temp;
    gchar *line_str = NULL;
    gchar *iter;
    GList *label_outputs_list = NULL;
    guint label_outputs_length = 0;
    GList *list_iter;
    guint i;
    gchar **label_outputs;

    /* Read the line into `line_str` char array */
    status =
        g_io_channel_read_line (label_file,
        (gchar **) & line_str, NULL, &term_pos, &err);

    if (line_str == NULL)
      continue;

    temp = line_str;

    temp[term_pos] = '\0';

    /* Parse ';' delimited strings and prepend to the `labels_output_list`.
     * Prepending to a list and reversing after adding all strings is faster
     * than appending every string.
     * https://developer.gnome.org/glib/stable/glib-Doubly-Linked-Lists.html#g-list-append
     */
    while ((iter = g_strstr_len (temp, -1, ";"))) {
      *iter = '\0';
      label_outputs_list = g_list_prepend (label_outputs_list, g_strdup (temp));
      label_outputs_length++;
      temp = iter + 1;
    }

    if (*temp != '\0') {
      label_outputs_list = g_list_prepend (label_outputs_list, g_strdup (temp));
      label_outputs_length++;
    }

    /* All labels in one line parsed and added. Now reverse the list. */
    label_outputs_list = g_list_reverse (label_outputs_list);
    list_iter = label_outputs_list;

    /* Convert the list to an array for faster access. */
    label_outputs =
        (gchar **) g_malloc0 (sizeof (gchar *) * label_outputs_length);
    for (i = 0; i < label_outputs_length; i++) {
      label_outputs[i] = (gchar *) list_iter->data;
      list_iter = list_iter->next;
    }
    g_list_free (label_outputs_list);

    /* Prepend the pointer to array of labels in one line to `labels_list`. */
    labels_list = g_list_prepend (labels_list, label_outputs);
    /* Prepend the corresponding array size to `label_outputs_length_list`. */
    label_outputs_length_list =
        g_list_prepend (label_outputs_length_list,
        (gchar *) NULL + label_outputs_length);

    /* Maintain the number of labels(lines). */
    config->n_labels++;
    g_free (line_str);

  } while (status == G_IO_STATUS_NORMAL);
  g_io_channel_unref (label_file);
  ret = TRUE;

  /* Convert the `labels_list` and the `label_outputs_length_list` to arrays
   * for faster access. */
  config->n_label_outputs =
      (guint *) g_malloc (config->n_labels * sizeof (guint));
  config->labels = (gchar ***) g_malloc (config->n_labels * sizeof (gchar **));

  labels_list = g_list_reverse (labels_list);
  label_outputs_length_list = g_list_reverse (label_outputs_length_list);

  labels_iter = labels_list;
  label_outputs_length_iter = label_outputs_length_list;
  for (j = 0; j < config->n_labels; j++) {
    config->labels[j] = (gchar **) labels_iter->data;
    labels_iter = labels_iter->next;
    config->n_label_outputs[j] =
        (gchar *) label_outputs_length_iter->data - (gchar *) NULL;
    label_outputs_length_iter = label_outputs_length_iter->next;
  }

  g_list_free (labels_list);
  g_list_free (label_outputs_length_list);

done:
  return ret;
}

gboolean
parse_source (NvDsSourceConfig * config, GKeyFile * key_file, gchar * group,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  static GList *camera_id_list = NULL;

  if (g_strcmp0 (group, CONFIG_GROUP_SOURCE_ALL)) {
    if (g_key_file_get_integer (key_file, group,
            CONFIG_GROUP_ENABLE, &error) == FALSE || error != NULL) {
      return TRUE;
    }

    gchar *source_id_start_ptr = group + strlen (CONFIG_GROUP_SOURCE);
    gchar *source_id_end_ptr = NULL;
    config->camera_id =
        g_ascii_strtoull (source_id_start_ptr, &source_id_end_ptr, 10);

    config->rtsp_reconnect_attempts = -1;

    // Source group name should be of the form [source<%u>]. If
    // *source_id_end_ptr is not the string terminating character '\0' or if
    // the pointer has the same value as source_id_start_ptr, then the group
    // name does not conform to the specs.
    if (source_id_start_ptr == source_id_end_ptr || *source_id_end_ptr != '\0') {
      NVGSTDS_ERR_MSG_V
          ("Source group \"[%s]\" is not in the form \"[source<%%d>]\"", group);
      return FALSE;
    }
    // Check if a source with same source_id has already been parsed.
    if (g_list_find (camera_id_list,
            GUINT_TO_POINTER (config->camera_id)) != NULL) {
      NVGSTDS_ERR_MSG_V
          ("Did not parse source group \"[%s]\". Another source group"
          " with source-id %d already exists", group, config->camera_id);
      return FALSE;
    }
    camera_id_list = g_list_prepend (camera_id_list,
        GUINT_TO_POINTER (config->camera_id));
  }
  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  config->latency = 100;
  config->num_decode_surfaces = N_DECODE_SURFACES;
  config->num_extra_surfaces = N_EXTRA_SURFACES;
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_TYPE)) {
      config->type =
          (NvDsSourceType) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_WIDTH)) {
      config->source_width =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_HEIGHT)) {
      config->source_height =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_FPS_N)) {
      config->source_fps_n =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_FPS_N, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_FPS_D)) {
      config->source_fps_d =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_FPS_D, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_CSI_SID)) {
      config->camera_csi_sensor_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_CSI_SID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_V4L2_DEVNODE)) {
      config->camera_v4l2_dev_node =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_V4L2_DEVNODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_ALSA_DEVICE)) {
      config->alsa_device =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SOURCE_ALSA_DEVICE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_UDP_BUFFER_SIZE)) {
      config->udp_buffer_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_UDP_BUFFER_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SENSORID_PADID_MAPPING)) {
      config->sensorIdToPadIdMapping =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SOURCE_SENSORID_PADID_MAPPING, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_VIDEO_FORMAT)) {
      config->video_format =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SOURCE_VIDEO_FORMAT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_URI)) {
      gchar *uri = g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SOURCE_URI, &error);
      CHECK_ERROR (error);
      if (g_str_has_prefix (uri, "file://")) {
        config->uri = g_strdup (uri + 7);
        config->uri =
            g_strdup_printf ("file://%s", get_absolute_file_path (cfg_file_path,
                config->uri));
        g_free (uri);
      } else
        config->uri = uri;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LATENCY)) {
      config->latency =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_LATENCY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_NUM_SOURCES)) {
      config->num_sources =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_NUM_SOURCES, &error);
      CHECK_ERROR (error);
      if (config->num_sources < 1) {
        config->num_sources = 1;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, group, CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_NUM_DECODE_SURFACES)) {
      config->num_decode_surfaces =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_NUM_DECODE_SURFACES, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_NUM_EXTRA_SURFACES)) {
      config->num_extra_surfaces =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_NUM_EXTRA_SURFACES, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_DROP_FRAME_INTERVAL)) {
      config->drop_frame_interval =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_DROP_FRAME_INTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_DROP_ON_LATENCY)) {
      config->drop_on_latency =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_DROP_ON_LATENCY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_CAMERA_ID)) {
      config->camera_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_CAMERA_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_SOURCE_RTSP_RECONNECT_INTERVAL_SEC)) {
      config->rtsp_reconnect_interval_sec =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_RTSP_RECONNECT_INTERVAL_SEC, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_SOURCE_RTSP_INIT_RECONNECT_INTERVAL_SEC)) {
      config->init_rtsp_reconnect_interval_sec =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_RTSP_INIT_RECONNECT_INTERVAL_SEC, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_RTSP_RECONNECT_ATTEMPTS)) {
      config->rtsp_reconnect_attempts =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_RTSP_RECONNECT_ATTEMPTS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SIMULATE_FPS_INTERVAL_MS)) {
      config->simulate_fps_interval_ms =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SIMULATE_FPS_INTERVAL_MS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_INTRA_DECODE)) {
      config->Intra_decode =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_INTRA_DECODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LOW_LATENCY_DECODE)) {
      config->low_latency_mode =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_LOW_LATENCY_DECODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, "cudadec-memtype")) {
      config->cuda_memory_type =
          g_key_file_get_integer (key_file, group, "cudadec-memtype", &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, group,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SELECT_RTP_PROTOCOL)) {
      config->select_rtp_protocol =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SELECT_RTP_PROTOCOL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SET_RTP_PROTOCOL)) {
      config->select_rtp_protocol =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SET_RTP_PROTOCOL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LEAKY)) {
      config->leaky =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_LEAKY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_MAX_SIZE_BUFFERS)) {
      config->max_size_buffers =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_MAX_SIZE_BUFFERS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_BUFFER_MODE)) {
      config->buffer_mode =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_BUFFER_MODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_ID)) {
      config->source_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_ENABLE)) {
      config->smart_record =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_DIRPATH)) {
      config->dir_path =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_DIRPATH, &error);
      CHECK_ERROR (error);

      if (access (config->dir_path, W_OK)) {
        if (errno == ENOENT || errno == ENOTDIR) {
          g_print ("ERROR: Directory (%s) doesn't exist.\n", config->dir_path);
        } else if (errno == EACCES) {
          g_print ("ERROR: No write permission in %s\n", config->dir_path);
        }
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_FILE_PREFIX)) {
      config->file_prefix =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_FILE_PREFIX, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE_LEGACY)) {
      NVGSTDS_WARN_MSG_V
          ("Deprecated config '%s' used in group [%s]. Use '%s' instead",
          CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE_LEGACY, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE);

      config->smart_rec_cache_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE_LEGACY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE)) {
      config->smart_rec_cache_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_CACHE_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_CONTAINER)) {
      config->smart_rec_container =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_CONTAINER, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_START_TIME)) {
      config->smart_rec_start_time =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_START_TIME, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_SOURCE_SMART_RECORD_DEFAULT_DURATION)) {
      config->smart_rec_def_duration =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_DEFAULT_DURATION, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_DURATION)) {
      config->smart_rec_duration =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_DURATION, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SMART_RECORD_INTERVAL)) {
      config->smart_rec_interval =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SOURCE_SMART_RECORD_INTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_EXTRACT_SEI_TYPE5_DATA)) {
        config->extract_sei_type5_data =
            g_key_file_get_integer(key_file, CONFIG_GROUP_SOURCE,
            CONFIG_GROUP_SOURCE_EXTRACT_SEI_TYPE5_DATA, &error);
        CHECK_ERROR(error);
    }
#if defined(__aarch64__) && !defined(AARCH64_IS_SBSA)
    else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_NVVIDEOCONVERT_COPY_HW)) {
        config->nvvideoconvert_copy_hw =
            g_key_file_get_integer(key_file, group,
            CONFIG_GROUP_SOURCE_NVVIDEOCONVERT_COPY_HW, &error);
        CHECK_ERROR(error);
    }
#endif
    else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}


gboolean
parse_streammux (NvDsStreammuxConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_STREAMMUX, NULL, &error);
  CHECK_ERROR (error);

  config->frame_duration = -1;
  config->batched_push_timeout = -1;
  config->buffer_cache_timeout = -1;
  config->attach_sys_ts_as_ntp = TRUE;
  config->async_process = TRUE;
  config->no_pipeline_eos = FALSE;
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_COMPUTE_HW)) {
      config->compute_hw =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_COMPUTE_HW, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_SORT_BATCH)) {
      config->sort_batch =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_SORT_BATCH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_CACHE_BUFFER)) {
      config->buffer_cache =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_CACHE_BUFFER, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_CACHE_BUFFER_TIMEOUT)) {
      config->buffer_cache_timeout =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_CACHE_BUFFER_TIMEOUT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_SYNC_INPUTS_NTP)) {
      config->sync_inputs_ntp =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_SYNC_INPUTS_NTP, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_WIDTH)) {
      config->pipeline_width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_HEIGHT)) {
      config->pipeline_height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_ENABLE_PADDING)) {
      config->enable_padding =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_ENABLE_PADDING, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_FRAME_DURATION)) {
      config->frame_duration =
          g_key_file_get_uint64 (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_FRAME_DURATION, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_BUFFER_POOL_SIZE)) {
      config->buffer_pool_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_BUFFER_POOL_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_BATCH_SIZE)) {
      config->batch_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_BATCH_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_LIVE_SOURCE)) {
      config->live_source =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_LIVE_SOURCE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_ATTACH_SYS_TS_AS_NTP)) {
      config->attach_sys_ts_as_ntp =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_ATTACH_SYS_TS_AS_NTP, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_STREAMMUX_FRAME_NUM_RESET_ON_STREAM_RESET)) {
      config->frame_num_reset_on_stream_reset =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_FRAME_NUM_RESET_ON_STREAM_RESET, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_BATCHED_PUSH_TIMEOUT)) {
      config->batched_push_timeout =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_BATCHED_PUSH_TIMEOUT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_CONFIG_FILE_PATH)) {
      config->config_file_path =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_STREAMMUX,
              CONFIG_GROUP_STREAMMUX_CONFIG_FILE_PATH, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_SYNC_INPUTS)) {
      config->sync_inputs =
          g_key_file_get_integer (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_SYNC_INPUTS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_MAX_LATENCY)) {
      config->max_latency =
          g_key_file_get_uint64 (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_MAX_LATENCY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_FRAME_NUM_RESET_ON_EOS)) {
      config->frame_num_reset_on_eos =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_FRAME_NUM_RESET_ON_EOS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_ASYNC_PROCESS)) {
      config->async_process =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_ASYNC_PROCESS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_STREAMMUX_DROP_PIPELINE_EOS)) {
      config->no_pipeline_eos =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_STREAMMUX,
          CONFIG_GROUP_STREAMMUX_DROP_PIPELINE_EOS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_STREAMMUX_NUM_SURFACES_PER_FRAME)) {
        config->num_surface_per_frame =
            g_key_file_get_integer(key_file, CONFIG_GROUP_STREAMMUX,
            CONFIG_GROUP_STREAMMUX_NUM_SURFACES_PER_FRAME, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_STREAMMUX_EXTRACT_SEI_TYPE5_DATA)) {
        config->extract_sei_type5_data =
            g_key_file_get_integer(key_file, CONFIG_GROUP_STREAMMUX,
            CONFIG_GROUP_STREAMMUX_EXTRACT_SEI_TYPE5_DATA, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_STREAMMUX_EXTRACT_SEI_SIM_TIME)) {
        config->extract_sei_sim_time =
            g_key_file_get_integer(key_file, CONFIG_GROUP_STREAMMUX,
            CONFIG_GROUP_STREAMMUX_EXTRACT_SEI_SIM_TIME, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_STREAMMUX_ALIGN_FIRST_BUFFER)) {
        config->align_first_buffer =
            g_key_file_get_integer(key_file, CONFIG_GROUP_STREAMMUX,
            CONFIG_GROUP_STREAMMUX_ALIGN_FIRST_BUFFER, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_STREAMMUX_FAILSAFE_FLUSH_COUNT)) {
        config->failsafe_flush_count =
            g_key_file_get_integer(key_file, CONFIG_GROUP_STREAMMUX,
            CONFIG_GROUP_STREAMMUX_FAILSAFE_FLUSH_COUNT, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_STREAMMUX_DROP_BACKWARD_SEI)) {
        config->drop_backward_sei =
            g_key_file_get_integer(key_file, CONFIG_GROUP_STREAMMUX,
            CONFIG_GROUP_STREAMMUX_DROP_BACKWARD_SEI, &error);
        CHECK_ERROR(error);
    }
    else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_STREAMMUX);
    }
  }

  config->is_parsed = TRUE;

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_dsanalytics (NvDsDsAnalyticsConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_DSANALYTICS, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSANALYTICS,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSANALYTICS_CONFIG_FILE)) {
      config->config_file_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_DSANALYTICS,
              CONFIG_GROUP_DSANALYTICS_CONFIG_FILE, &error));
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_DSANALYTICS);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_dsexample (NvDsDsExampleConfig * config, GKeyFile * key_file)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_DSEXAMPLE, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_FULL_FRAME)) {
      config->full_frame =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_FULL_FRAME, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_PROCESSING_WIDTH)) {
      config->processing_width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_PROCESSING_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_PROCESSING_HEIGHT)) {
      config->processing_height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_PROCESSING_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_BLUR_OBJECTS)) {
      config->blur_objects =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_BLUR_OBJECTS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_UNIQUE_ID)) {
      config->unique_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_UNIQUE_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DSEXAMPLE_BATCH_SIZE)) {
      config->batch_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_DSEXAMPLE,
          CONFIG_GROUP_DSEXAMPLE_BATCH_SIZE, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_DSEXAMPLE);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_segvisual (NvDsSegVisualConfig * config, GKeyFile * key_file)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  /** Default values */
  config->height = SEG_OUTPUT_HEIGHT;
  config->width = SEG_OUTPUT_WIDTH;
  config->gpu_id = 0;
  config->max_batch_size = 1;
  config->nvbuf_memory_type = 0;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_SEGVISUAL, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SEGVISUAL,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SEGVISUAL_WIDTH)) {
      config->width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SEGVISUAL,
          CONFIG_GROUP_SEGVISUAL_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SEGVISUAL_HEIGHT)) {
      config->height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SEGVISUAL,
          CONFIG_GROUP_SEGVISUAL_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SEGVISUAL_BATCH_SIZE)) {
      config->max_batch_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SEGVISUAL,
          CONFIG_GROUP_SEGVISUAL_BATCH_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SEGVISUAL,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SEGVISUAL,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_SEGVISUAL);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_osd (NvDsOSDConfig * config, GKeyFile * key_file)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  /** Default values */
  config->draw_text = TRUE;
  config->draw_bbox = TRUE;
  config->draw_mask = FALSE;
  config->mode = MODE_GPU;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_OSD, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_MODE)) {
      config->mode =
          (NvOSD_Mode) g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_MODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_BORDER_WIDTH)) {
      config->border_width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_BORDER_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_TEXT_SIZE)) {
      config->text_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_TEXT_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_TEXT_COLOR)) {
      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_TEXT_COLOR, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        NVGSTDS_ERR_MSG_V
            ("Color params should be exactly 4 floats {r, g, b, a} between 0 and 1");
        goto done;
      }
      config->text_color.red = list[0];
      config->text_color.green = list[1];
      config->text_color.blue = list[2];
      config->text_color.alpha = list[3];
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_TEXT_BG_COLOR)) {
      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_TEXT_BG_COLOR, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        NVGSTDS_ERR_MSG_V
            ("Color params should be exactly 4 floats {r, g, b, a} between 0 and 1");
        goto done;
      }
      config->text_bg_color.red = list[0];
      config->text_bg_color.green = list[1];
      config->text_bg_color.blue = list[2];
      config->text_bg_color.alpha = list[3];

      if (config->text_bg_color.red > 0 || config->text_bg_color.green > 0
          || config->text_bg_color.blue > 0 || config->text_bg_color.alpha > 0)
        config->text_has_bg = TRUE;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_FONT)) {
      config->font =
          g_key_file_get_string (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_FONT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_CLOCK_ENABLE)) {
      config->enable_clock =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_CLOCK_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_CLOCK_X_OFFSET)) {
      config->clock_x_offset =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_CLOCK_X_OFFSET, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_CLOCK_Y_OFFSET)) {
      config->clock_y_offset =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_CLOCK_Y_OFFSET, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_CLOCK_TEXT_SIZE)) {
      config->clock_text_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_CLOCK_TEXT_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_HW_BLEND_COLOR_ATTR)) {
      config->hw_blend_color_attr =
          g_key_file_get_string (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_HW_BLEND_COLOR_ATTR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_CLOCK_COLOR)) {
      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_CLOCK_COLOR, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        NVGSTDS_ERR_MSG_V
            ("Color params should be exactly 4 floats {r, g, b, a} between 0 and 1");
        goto done;
      }
      config->clock_color.red = list[0];
      config->clock_color.green = list[1];
      config->clock_color.blue = list[2];
      config->clock_color.alpha = list[3];
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_OSD,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_SHOW_TEXT)) {
      config->draw_text =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_SHOW_TEXT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_SHOW_BBOX)) {
      config->draw_bbox =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_SHOW_BBOX, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_OSD_SHOW_MASK)) {
      config->draw_mask =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_OSD,
          CONFIG_GROUP_OSD_SHOW_MASK, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_OSD);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_dewarper (NvDsDewarperConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path, gchar * dewarper)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  //gchar *dewarper = g_strdup_printf("dewarper%d", config->source_id);

  keys = g_key_file_get_keys (key_file, dewarper, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, dewarper,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, dewarper,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_CONFIG_FILE)) {
      config->config_file = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file,
              dewarper,
              CONFIG_GROUP_DEWARPER_CONFIG_FILE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, dewarper,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_NUM_SURFACES_PER_FRAME)) {
      config->num_surfaces_per_frame =
          g_key_file_get_integer (key_file, dewarper,
          CONFIG_GROUP_DEWARPER_NUM_SURFACES_PER_FRAME, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SOURCE_ID)) {
      guint source_id =
          g_key_file_get_integer (key_file, dewarper,
          CONFIG_GROUP_DEWARPER_SOURCE_ID, &error);
      config->source_id = source_id;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_NUM_OUTPUT_BUFFERS)) {
      config->num_out_buffers =
          g_key_file_get_integer (key_file, dewarper,
          CONFIG_GROUP_DEWARPER_NUM_OUTPUT_BUFFERS, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          dewarper);
    }
  }
  //g_free(dewarper);
  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_preprocess (NvDsPreProcessConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_ID_FOR_OPERATION)) {
      config->operate_on_gie_id = g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_ID_FOR_OPERATION, &error);
      CHECK_ERROR (error);
      config->is_operate_on_gie_id_set = TRUE;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_PREPROCESS_CONFIG_FILE)) {
      config->config_file_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_PREPROCESS_CONFIG_FILE, &error));
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_gie (NvDsGieConfig * config, GKeyFile * key_file, gchar * group,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  if (g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_ENABLE, &error) == FALSE || error != NULL)
    return TRUE;

  config->bbox_border_color_table = g_hash_table_new (NULL, NULL);
  config->bbox_bg_color_table = g_hash_table_new (NULL, NULL);
  config->bbox_border_color = (NvOSD_ColorParams) {
  1, 0, 0, 1};

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_INPUT_TENSOR_META)) {
      config->input_tensor_meta =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_GIE_INPUT_TENSOR_META, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_CLASS_IDS_FOR_OPERATION)) {
      gsize length = 0;
      config->list_operate_on_class_ids =
          g_key_file_get_integer_list (key_file, group,
          CONFIG_GROUP_GIE_CLASS_IDS_FOR_OPERATION, &length, &error);
      config->num_operate_on_class_ids = length;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_BATCH_SIZE)) {
      config->batch_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_BATCH_SIZE, &error);
      config->is_batch_size_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_MODEL_ENGINE)) {
      config->model_engine_file_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_GIE_MODEL_ENGINE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_PLUGIN_TYPE)) {
      config->plugin_type =
          (NvDsGiePluginType) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_PLUGIN_TYPE, &error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_AUDIO_TRANSFORM)) {
      config->audio_transform =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_GIE_AUDIO_TRANSFORM, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_FRAME_SIZE)) {
      config->frame_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_FRAME_SIZE, &error);
      config->is_frame_size_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_HOP_SIZE)) {
      config->hop_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_HOP_SIZE, &error);
      config->is_hop_size_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_AUDIO_RATE)) {
      config->input_audio_rate =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_AUDIO_RATE, &error);
      config->is_hop_size_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_MODEL_VARIANT)) {
      config->model_variant =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_GIE_MODEL_VARIANT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_DEVICE)) {
      config->device =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_GIE_DEVICE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_MIN_CROP_SIZE)) {
      config->min_crop_size =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_MIN_CROP_SIZE, &error);
      config->is_min_crop_size_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_VERBOSE)) {
      config->verbose =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_VERBOSE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_BACKEND)) {
      config->backend =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_GIE_BACKEND, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_TRITON_URL)) {
      config->triton_url =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_GIE_TRITON_URL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_TRITON_MODEL)) {
      config->triton_model =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_GIE_TRITON_MODEL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_LABEL)) {
      config->label_file_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_GIE_LABEL, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_CONFIG_FILE)) {
      config->config_file_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_GIE_CONFIG_FILE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_INTERVAL)) {
      config->interval =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_INTERVAL, &error);
      config->is_interval_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_UNIQUE_ID)) {
      config->unique_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_UNIQUE_ID, &error);
      config->is_unique_id_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_ID_FOR_OPERATION)) {
      config->operate_on_gie_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_GIE_ID_FOR_OPERATION, &error);
      config->is_operate_on_gie_id_set = TRUE;
      CHECK_ERROR (error);
    } else if (!strncmp (*key, CONFIG_GROUP_GIE_BBOX_BORDER_COLOR,
            sizeof (CONFIG_GROUP_GIE_BBOX_BORDER_COLOR) - 1)) {
      NvOSD_ColorParams *clr_params;
      gchar *key1 = *key + sizeof (CONFIG_GROUP_GIE_BBOX_BORDER_COLOR) - 1;
      gchar *endptr = NULL;
      gint64 class_index = -1;

      /* Check if the key is specified for a particular class or for all classes.
       * For generic key "bbox-border-color", strlen (key1) will return 0 and·
       * class_index will be -1.
       * For class-specific key "bbox-border-color<class-id>", strlen (key1)
       * will return a positive value and·class_index will have a value >= 0.
       */
      if (strlen (key1) > 0) {
        class_index = g_ascii_strtoll (key1, &endptr, 10);
        if (class_index == 0 && endptr == key1) {
          NVGSTDS_WARN_MSG_V
              ("BBOX colors should be specified with key '%s%%d'",
              CONFIG_GROUP_GIE_BBOX_BORDER_COLOR);
          continue;
        }
      }

      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, group,
          *key, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        NVGSTDS_ERR_MSG_V
            ("Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
        goto done;
      }

      if (class_index == -1) {
        clr_params = &config->bbox_border_color;
      } else {
        clr_params =
            (NvOSD_ColorParams *) g_malloc (sizeof (NvOSD_ColorParams));
        g_hash_table_insert (config->bbox_border_color_table,
            class_index + (gchar *) NULL, clr_params);
      }

      clr_params->red = list[0];
      clr_params->green = list[1];
      clr_params->blue = list[2];
      clr_params->alpha = list[3];

    } else if (!strncmp (*key, CONFIG_GROUP_GIE_BBOX_BG_COLOR,
            sizeof (CONFIG_GROUP_GIE_BBOX_BG_COLOR) - 1)) {
      NvOSD_ColorParams *clr_params;
      gchar *key1 = *key + sizeof (CONFIG_GROUP_GIE_BBOX_BG_COLOR) - 1;
      gchar *endptr = NULL;
      gint64 class_index = -1;

      /* Check if the key is specified for a particular class or for all classes.
       * For generic key "bbox-bg-color", strlen (key1) will return 0 and·
       * class_index will be -1.
       * For class-specific key "bbox-bg-color<class-id>", strlen (key1)
       * will return a positive value and·class_index will have a value >= 0.
       */
      if (strlen (key1) > 0) {
        class_index = g_ascii_strtoll (key1, &endptr, 10);
        if (class_index == 0 && endptr == key1) {
          NVGSTDS_WARN_MSG_V
              ("BBOX background colors should be specified with key '%s%%d'",
              CONFIG_GROUP_GIE_BBOX_BG_COLOR);
          continue;
        }
      }

      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, group,
          *key, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        NVGSTDS_ERR_MSG_V
            ("Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
        goto done;
      }

      if (class_index == -1) {
        clr_params = &config->bbox_bg_color;
        config->have_bg_color = TRUE;
      } else {
        clr_params =
            (NvOSD_ColorParams *) g_malloc (sizeof (NvOSD_ColorParams));
        g_hash_table_insert (config->bbox_bg_color_table,
            class_index + (gchar *) NULL, clr_params);
      }

      clr_params->red = list[0];
      clr_params->green = list[1];
      clr_params->blue = list[2];
      clr_params->alpha = list[3];
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_GIE_RAW_OUTPUT_DIR)) {
      config->raw_output_directory = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_GIE_RAW_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, group, CONFIG_GPU_ID, &error);
      config->is_gpu_id_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, group,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }
  if (config->enable && config->label_file_path && !parse_labels_file (config)) {
    NVGSTDS_ERR_MSG_V ("Failed while parsing label file '%s'",
        config->label_file_path);
    goto done;
  }
  if (!config->config_file_path) {
    NVGSTDS_ERR_MSG_V ("Config file not provided for group '%s'", group);
    goto done;
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_tracker (NvDsTrackerConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  config->display_tracking_id = TRUE;
  config->tracking_id_reset_mode = 0;
  config->input_tensor_meta = FALSE;
  config->input_tensor_gie_id = 0;
  config->compute_hw = 0;
  config->user_meta_pool_size = 32;
  config->sub_batches = NULL;
  config->sub_batch_err_recovery_trial_cnt = 0;
  config->operate_on_class_ids = NULL;

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      config->width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      config->height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_SURFACE_TYPE)) {
      config->tracking_surf_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_SURFACE_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      gchar* temp = g_key_file_get_string (key_file,
              CONFIG_GROUP_TRACKER,
              CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error);
      CHECK_ERROR (error);
      gchar **configFileList;
      gchar *single_config_file_path;
      configFileList = g_strsplit (temp, ";", 0);
      gchar *temp_list1 = NULL, *temp_list2 = NULL;
      if (g_strv_length (configFileList) == 1)
      {
        // These is a single config file
        config->ll_config_file = get_absolute_file_path (cfg_file_path, temp);
      }
      else
      {
        single_config_file_path = get_absolute_file_path (cfg_file_path, configFileList[0]);
        temp_list1 = g_strconcat (single_config_file_path, ";", NULL);
        g_free(single_config_file_path);
        for(int i = 1; i < (int)g_strv_length (configFileList); i++)
        {
          single_config_file_path = get_absolute_file_path (cfg_file_path, configFileList[i]);
          temp_list2 = g_strconcat (temp_list1, single_config_file_path, ";", NULL);
          g_free(temp_list1); g_free(single_config_file_path);
          temp_list1 = temp_list2;
        }
        config->ll_config_file = temp_list1;
      }
      temp = NULL;
      configFileList = NULL;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      config->ll_lib_file = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file,
              CONFIG_GROUP_TRACKER, CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_TRACKING_SURFACE_TYPE)) {
      config->tracking_surface_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_TRACKING_SURFACE_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_DISPLAY_TRACKING_ID)) {
      config->display_tracking_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_DISPLAY_TRACKING_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_TRACKING_ID_RESET_MODE)) {
      config->tracking_id_reset_mode =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_TRACKING_ID_RESET_MODE, &error);
      CHECK_ERROR (error);
    }
    else if(!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_INPUT_TENSOR_META)){
      config->input_tensor_meta =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_INPUT_TENSOR_META, &error);
      CHECK_ERROR (error);
    }
    else if(!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_TENSOR_META_GIE_ID)){
      config->input_tensor_gie_id =
        g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
            CONFIG_GROUP_TRACKER_TENSOR_META_GIE_ID, &error);
      CHECK_ERROR (error);
    }
    else if(!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_COMPUTE_HW)){
      config->compute_hw =
        g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
            CONFIG_GROUP_TRACKER_COMPUTE_HW, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_USER_META_POOL_SIZE)) {
      config->user_meta_pool_size =
        g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
            CONFIG_GROUP_TRACKER_USER_META_POOL_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_SUB_BATCHES)) {
      config->sub_batches =
        g_key_file_get_string (key_file, CONFIG_GROUP_TRACKER,
            CONFIG_GROUP_TRACKER_SUB_BATCHES, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_SUB_BATCH_ERR_RECOVERY_TRIAL_CNT)) {
      config->sub_batch_err_recovery_trial_cnt =
        g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
            CONFIG_GROUP_TRACKER_SUB_BATCH_ERR_RECOVERY_TRIAL_CNT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_OPERATE_ON_CLASS_IDS)) {
      config->operate_on_class_ids =
        g_key_file_get_string (key_file, CONFIG_GROUP_TRACKER,
            CONFIG_GROUP_TRACKER_OPERATE_ON_CLASS_IDS, &error);
      CHECK_ERROR (error);
    }
    else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_visionencoder (NvDsVisionEncoderConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  /* Set defaults */
  config->enable = FALSE;
  config->model_variant = g_strdup("so400m-384");
  config->batch_size = 16;
  config->device = g_strdup("cuda");
  config->min_crop_size = 32;
  config->verbose = FALSE;
  config->gpu_id = 0;
  config->backend = g_strdup("transformers");
  config->triton_url = g_strdup("localhost:8001");
  config->triton_model = g_strdup("cradio_trt");
  config->skip_interval = 0;  // Default: no skipping
  config->embedding_classes = g_strdup("");  // Default: all classes
  config->onnx_model = NULL;  // Optional: ONNX model for engine building
  config->tensorrt_engine = NULL;  // Optional: explicit TensorRT engine path
  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_MODEL_VARIANT)) {
      config->model_variant = g_key_file_get_string (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_BATCH_SIZE)) {
      config->batch_size = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_DEVICE)) {
      config->device = g_key_file_get_string (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_MIN_CROP_SIZE)) {
      config->min_crop_size = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_VERBOSE)) {
      config->verbose = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_BACKEND)) {
      config->backend = g_key_file_get_string (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_TRITON_URL)) {
      config->triton_url = g_key_file_get_string (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_TRITON_MODEL)) {
      config->triton_model = g_key_file_get_string (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_SKIP_INTERVAL)) {
      config->skip_interval = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_EMBEDDING_CLASSES)) {
      config->embedding_classes = g_key_file_get_string (key_file, group, *key, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_ONNX_MODEL)) {
      config->onnx_model = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group, *key, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_VISIONENCODER_TENSORRT_ENGINE)) {
      config->tensorrt_engine = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group, *key, &error));
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }
  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_text_embedder (NvDsTextEmbedderConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  config->model_name = g_strdup("siglip2-giant");
  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TEXT_EMBEDDER, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TEXT_EMBEDDER,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TEXT_EMBEDDER_MODEL_NAME)) {
      g_free (config->model_name);
      config->model_name =
          g_key_file_get_string (key_file, CONFIG_GROUP_TEXT_EMBEDDER,
          CONFIG_GROUP_TEXT_EMBEDDER_MODEL_NAME, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TEXT_EMBEDDER_ONNX_MODEL_PATH)) {
      config->onnx_model_path =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_TEXT_EMBEDDER,
          CONFIG_GROUP_TEXT_EMBEDDER_ONNX_MODEL_PATH, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TEXT_EMBEDDER_TOKENIZER_DIR)) {
      config->tokenizer_dir =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_TEXT_EMBEDDER,
          CONFIG_GROUP_TEXT_EMBEDDER_TOKENIZER_DIR, &error));
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TEXT_EMBEDDER);
    }
  }
  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_replay (NvDsReplayConfig *config, GKeyFile *key_file, gchar* cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_REPLAY, NULL, &error);
  CHECK_ERROR (error);

  config->label_dir = NULL;
  config->file_names = NULL;
  config->max_frame_nums = NULL;
  config->label_width = 1920;
  config->label_height = 1080;
  config->interval = 0;
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, CONFIG_GROUP_REPLAY,
          CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_REPLAY_LABEL_DIR)) {
      config->label_dir = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file,
              CONFIG_GROUP_REPLAY,
              CONFIG_GROUP_REPLAY_LABEL_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_REPLAY_FILE_NAMES)) {
      config->file_names =
          g_key_file_get_string (key_file,
              CONFIG_GROUP_REPLAY,
              CONFIG_GROUP_REPLAY_FILE_NAMES, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_REPLAY_MAX_FRAME_NUMS)) {
      config->max_frame_nums =
          g_key_file_get_string (key_file,
              CONFIG_GROUP_REPLAY,
              CONFIG_GROUP_REPLAY_MAX_FRAME_NUMS, &error);
      CHECK_ERROR (error);
    } else if(!g_strcmp0 (*key, CONFIG_GROUP_REPLAY_LABEL_WIDTH)){
      config->label_width =
          g_key_file_get_integer (key_file,
              CONFIG_GROUP_REPLAY,
              CONFIG_GROUP_REPLAY_LABEL_WIDTH, &error);
      CHECK_ERROR (error);
    } else if(!g_strcmp0 (*key, CONFIG_GROUP_REPLAY_LABEL_HEIGHT)){
      config->label_height =
          g_key_file_get_integer (key_file,
              CONFIG_GROUP_REPLAY,
              CONFIG_GROUP_REPLAY_LABEL_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if(!g_strcmp0(*key, CONFIG_GROUP_REPLAY_INTERVAL)) {
      config->interval =
          g_key_file_get_integer (key_file,
              CONFIG_GROUP_REPLAY,
              CONFIG_GROUP_REPLAY_INTERVAL, &error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_REPLAY);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_sink (NvDsSinkSubBinConfig *config, GKeyFile *key_file, gchar *group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  config->encoder_config.rtsp_port = 8554;
  config->encoder_config.udp_port = 5000;
  config->encoder_config.codec = NV_DS_ENCODER_H264;
  config->encoder_config.container = NV_DS_CONTAINER_MP4;
  config->encoder_config.compute_hw = 0;
  config->render_config.qos = FALSE;
  config->render_config.color_range = -1;
  config->render_config.set_mode = -1;
  config->link_to_demux = FALSE;
  config->msg_conv_broker_config.new_api = FALSE;
  config->msg_conv_broker_config.conv_msg2p_new_api = FALSE;
  config->msg_conv_broker_config.conv_frame_interval = 30;
  config->nvdslogger = FALSE;
  config->show_fps = FALSE;

  if (g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_ENABLE, &error) == FALSE || error != NULL)
    return TRUE;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_TYPE)) {
      config->type =
          (NvDsSinkType) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_ONLY_FOR_DEMUX)) {
      config->link_to_demux =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_ONLY_FOR_DEMUX, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_NVDSLOGGER)) {
      config->nvdslogger =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_NVDSLOGGER, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SHOW_FPS)) {
      config->show_fps =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_SHOW_FPS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SHOW_LATENCY)) {
      config->show_latency =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_SHOW_LATENCY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_WIDTH)) {
      config->render_config.width =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_HEIGHT)) {
      config->render_config.height =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_QOS)) {
      config->render_config.qos =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_QOS, &error);
      config->render_config.qos_value_specified = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SYNC)) {
      config->sync =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_SYNC, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->render_config.nvbuf_memory_type =
          g_key_file_get_integer (key_file, group,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_CONTAINER)) {
      config->encoder_config.container =
          (NvDsContainerType) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_CONTAINER, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_CODEC)) {
      config->encoder_config.codec =
          (NvDsEncoderType) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_CODEC, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_ENC_TYPE)) {
      config->encoder_config.enc_type =
          (NvDsEncHwSwType) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_ENC_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_COPY_META)) {
      config->encoder_config.copy_meta =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_COPY_META, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_OUTPUT_IO_MODE)) {
      config->encoder_config.output_io_mode =
          (NvDsEncOutputIOMode) g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_OUTPUT_IO_MODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SW_PRESET)) {
      config->encoder_config.sw_preset =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_SW_PRESET, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_BITRATE)) {
      config->encoder_config.bitrate =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_BITRATE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_PROFILE)) {
      config->encoder_config.profile =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_PROFILE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_IFRAMEINTERVAL)) {
      config->encoder_config.iframeinterval =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_IFRAMEINTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_OUTPUT_FILE)) {
      config->encoder_config.output_file_path =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SINK_OUTPUT_FILE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SOURCE_ID)) {
      config->source_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_SOURCE_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_RTSP_PORT)) {
      config->encoder_config.rtsp_port =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_RTSP_PORT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_UDP_PORT)) {
      config->encoder_config.udp_port =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_UDP_PORT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_UDP_BUFFER_SIZE)) {
      config->encoder_config.udp_buffer_size =
          g_key_file_get_uint64 (key_file, group,
          CONFIG_GROUP_SINK_UDP_BUFFER_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_COLOR_RANGE)) {
      config->render_config.color_range =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_COLOR_RANGE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_CONN_ID)) {
      config->render_config.conn_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_CONN_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_PLANE_ID)) {
      config->render_config.plane_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_PLANE_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SET_MODE)) {
      config->render_config.set_mode =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_SET_MODE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_COMPUTE_HW)) {
      config->encoder_config.compute_hw =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_SINK_COMPUTE_HW, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->encoder_config.gpu_id = config->render_config.gpu_id =
          g_key_file_get_integer (key_file, group, CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_CONFIG) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_PAYLOAD_TYPE) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_MSG2P_LIB) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_COMP_ID) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_DEBUG_PAYLOAD_DIR) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_MULTIPLE_PAYLOADS) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_MSG2P_NEW_API) ||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_FRAME_INTERVAL)||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_EMBEDDING_FILTER)||
        !g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_DUMMY_PAYLOAD)) {
      ret =
          parse_msgconv (&config->msg_conv_broker_config, key_file, group,
          cfg_file_path);
      if (!ret)
        goto done;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_PROTO_LIB)) {
      config->msg_conv_broker_config.proto_lib =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_PROTO_LIB, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_CONN_STR)) {
      config->msg_conv_broker_config.conn_str =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_CONN_STR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_TOPIC)) {
      config->msg_conv_broker_config.topic =
          g_key_file_get_string (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_TOPIC, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_CONFIG_FILE)) {
      config->msg_conv_broker_config.broker_config_file_path =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_SINK_MSG_BROKER_CONFIG_FILE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_COMP_ID)) {
      config->msg_conv_broker_config.broker_comp_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_COMP_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_SINK_MSG_BROKER_DISABLE_MSG_CONVERTER)) {
      config->msg_conv_broker_config.disable_msgconv =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_DISABLE_MSG_CONVERTER, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_NEW_API)) {
      config->msg_conv_broker_config.new_api =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_NEW_API, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_BROKER_SLEEP_TIME)) {
      config->msg_conv_broker_config.broker_sleep_time =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_MSG_BROKER_SLEEP_TIME, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_tiled_display (NvDsTiledDisplayConfig * config, GKeyFile * key_file)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys =
      g_key_file_get_keys (key_file, CONFIG_GROUP_TILED_DISPLAY, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          (NvDsTiledDisplayEnable) g_key_file_get_integer (key_file,
          CONFIG_GROUP_TILED_DISPLAY, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_DISPLAY_ROWS)) {
      config->rows =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_DISPLAY_ROWS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_DISPLAY_COLUMNS)) {
      config->columns =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_DISPLAY_COLUMNS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_DISPLAY_WIDTH)) {
      config->width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_DISPLAY_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_DISPLAY_HEIGHT)) {
      config->height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_DISPLAY_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_NVBUF_MEMORY_TYPE)) {
      config->nvbuf_memory_type =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_NVBUF_MEMORY_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_COMPUTE_HW)) {
      config->compute_hw =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_COMPUTE_HW, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_DISPLAY_BUFFER_POOL_SIZE)) {
      config->buffer_pool_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_DISPLAY_BUFFER_POOL_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TILED_DISPLAY_SQUARE_SEQ_GRID)) {
      config->square_seq_grid =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_TILED_DISPLAY,
          CONFIG_GROUP_TILED_DISPLAY_SQUARE_SEQ_GRID, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TILED_DISPLAY);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_msgconv (NvDsSinkMsgConvBrokerConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_CONFIG)) {
      config->config_file_path =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_SINK_MSG_CONV_CONFIG, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_PAYLOAD_TYPE)) {
      config->conv_payload_type =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_MSG_CONV_PAYLOAD_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_MSG2P_LIB)) {
      config->conv_msg2p_lib =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_SINK_MSG_CONV_MSG2P_LIB, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_COMP_ID)) {
      config->conv_comp_id =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_MSG_CONV_COMP_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_DEBUG_PAYLOAD_DIR)) {
      config->debug_payload_dir = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_SINK_MSG_CONV_DEBUG_PAYLOAD_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_MULTIPLE_PAYLOADS)) {
      config->multiple_payloads =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_MSG_CONV_MULTIPLE_PAYLOADS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_MSG2P_NEW_API)) {
      config->conv_msg2p_new_api =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_MSG_CONV_MSG2P_NEW_API, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_FRAME_INTERVAL)) {
      config->conv_frame_interval =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_SINK_MSG_CONV_FRAME_INTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_DUMMY_PAYLOAD)) {
      config->conv_dummy_payload =
          g_key_file_get_boolean (key_file, group,
          CONFIG_GROUP_SINK_MSG_CONV_DUMMY_PAYLOAD, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_MSG_CONV_EMBEDDING_FILTER)) {
      config->embedding_filter =
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_SINK_MSG_CONV_EMBEDDING_FILTER, &error);
      CHECK_ERROR (error);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}




gboolean
parse_msgconsumer (NvDsMsgConsumerConfig * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_MSG_CONSUMER_CONFIG)) {
      config->config_file_path =
          get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_MSG_CONSUMER_CONFIG, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_MSG_CONSUMER_PROTO_LIB)) {
      config->proto_lib = g_key_file_get_string (key_file, group,
          CONFIG_GROUP_MSG_CONSUMER_PROTO_LIB, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_MSG_CONSUMER_CONN_STR)) {
      config->conn_str = g_key_file_get_string (key_file, group,
          CONFIG_GROUP_MSG_CONSUMER_CONN_STR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_MSG_CONSUMER_SENSOR_LIST_FILE)) {
      config->sensor_list_file = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_MSG_CONSUMER_SENSOR_LIST_FILE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_MSG_CONSUMER_SUBSCRIBE_TOPIC_LIST)) {
      gsize length = 0;
      guint i;
      gchar **topicList;

      topicList = g_key_file_get_string_list (key_file, group,
          CONFIG_GROUP_MSG_CONSUMER_SUBSCRIBE_TOPIC_LIST, &length, &error);

      CHECK_ERROR (error);
      if (length < 1) {
        NVGSTDS_ERR_MSG_V ("%s at least one topic must be provided", __func__);
	g_strfreev (topicList);
        goto done;
      }
      if (config->topicList)
        g_ptr_array_unref (config->topicList);

      config->topicList = g_ptr_array_new_full (length, g_free);
      for (i = 0; i < length; i++) {
        g_ptr_array_add (config->topicList, g_strdup (topicList[i]));
      }
      g_strfreev (topicList);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
parse_image_save (NvDsImageSave * config, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  config->enable = FALSE;
  config->gpu_id = 0;
  config->output_folder_path = NULL;
  config->frame_to_skip_rules_path = NULL;
  config->min_confidence = 0.0;
  config->max_confidence = 1.0;
  config->min_box_width = 1;
  config->min_box_height = 1;
  config->save_image_full_frame = TRUE;
  config->save_image_cropped_object = FALSE;
  config->second_to_skip_interval = 600;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);
  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      config->enable =
          g_key_file_get_integer (key_file, group, CONFIG_GROUP_ENABLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      config->gpu_id = g_key_file_get_integer (key_file, group,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_OUTPUT_FOLDER_PATH)) {
      config->output_folder_path = g_key_file_get_string (key_file, group,
          CONFIG_GROUP_IMG_SAVE_OUTPUT_FOLDER_PATH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_CSV_TIME_RULES_PATH)) {
      config->frame_to_skip_rules_path = g_key_file_get_string (key_file, group,
          CONFIG_GROUP_IMG_SAVE_CSV_TIME_RULES_PATH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_FULL_FRAME_IMG_SAVE)) {
      config->save_image_full_frame = g_key_file_get_double (key_file, group,
          CONFIG_GROUP_IMG_SAVE_FULL_FRAME_IMG_SAVE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_CROPPED_OBJECT_IMG_SAVE)) {
      config->save_image_cropped_object =
          g_key_file_get_double (key_file, group,
          CONFIG_GROUP_IMG_SAVE_CROPPED_OBJECT_IMG_SAVE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_SECOND_TO_SKIP_INTERVAL)) {
      config->second_to_skip_interval = g_key_file_get_double (key_file, group,
          CONFIG_GROUP_IMG_SAVE_SECOND_TO_SKIP_INTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_MIN_CONFIDENCE)) {
      config->min_confidence = g_key_file_get_double (key_file, group,
          CONFIG_GROUP_IMG_SAVE_MIN_CONFIDENCE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_MAX_CONFIDENCE)) {
      config->max_confidence = g_key_file_get_double (key_file, group,
          CONFIG_GROUP_IMG_SAVE_MAX_CONFIDENCE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_MIN_BOX_WIDTH)) {
      config->min_box_width = g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_IMG_SAVE_MIN_BOX_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_IMG_SAVE_MIN_BOX_HEIGHT)) {
      config->min_box_height = g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_IMG_SAVE_MIN_BOX_HEIGHT, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key, group);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
