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
#include <string.h>
#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"

#define CONFIG_GROUP_APP "application"
#define CONFIG_GROUP_APP_ENABLE_PERF_MEASUREMENT "enable-perf-measurement"
#define CONFIG_GROUP_APP_PERF_MEASUREMENT_INTERVAL "perf-measurement-interval-sec"
#define CONFIG_GROUP_APP_GIE_OUTPUT_DIR "gie-kitti-output-dir"
#define CONFIG_GROUP_APP_GIE_TRACK_OUTPUT_DIR "kitti-track-output-dir"
#define CONFIG_GROUP_APP_REID_TRACK_OUTPUT_DIR "reid-track-output-dir"

#define CONFIG_GROUP_APP_GLOBAL_GPU_ID "global-gpu-id"

#define CONFIG_GROUP_APP_TERMINATED_TRACK_OUTPUT_DIR "terminated-track-output-dir"
#define CONFIG_GROUP_APP_SHADOW_TRACK_OUTPUT_DIR "shadow-track-output-dir"

#define CONFIG_GROUP_TESTS "tests"
#define CONFIG_GROUP_TESTS_FILE_LOOP "file-loop"
#define CONFIG_GROUP_TESTS_PIPELINE_RECREATE_SEC "pipeline-recreate-sec"

#define CONFIG_GROUP_SOURCE_SGIE_BATCH_SIZE "sgie-batch-size"

GST_DEBUG_CATEGORY_EXTERN (APP_CFG_PARSER_CAT);


#define CHECK_ERROR(error) \
    if (error) { \
        GST_CAT_ERROR (APP_CFG_PARSER_CAT, "%s", error->message); \
        goto done; \
    }

static gboolean
parse_source_list (NvDsConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  gsize num_strings_uri = 0;
  gsize num_strings_sensor_id = 0;
  gsize num_strings_sensor_name = 0;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_SOURCE_LIST, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS)) {
      config->total_num_sources =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_DUMMY_PAYLOAD)) {
      config->dummy_payload =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_DUMMY_PAYLOAD, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_CUSTOM_TS_TO_RFC)) {
      config->custom_ts_to_rfc =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_CUSTOM_TS_TO_RFC, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_BATCHED_SENSOR_NAME)) {
      config->batched_sensor_name =
          g_key_file_get_string (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_BATCHED_SENSOR_NAME, &error);
      CHECK_ERROR (error);
    }
	 else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_URI_LIST)) {
      config->uri_list =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_URI_LIST, &num_strings_uri, &error);
      if (num_strings_uri > MAX_SOURCE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
        goto done;
      }
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_SENSOR_ID_LIST)) {
      config->sensor_id_list =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_SENSOR_ID_LIST, &num_strings_sensor_id, &error);
      if (num_strings_sensor_id > MAX_SOURCE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
        goto done;
      }
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_SENSOR_NAME_LIST)) {
      config->sensor_name_list =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_SENSOR_NAME_LIST, &num_strings_sensor_name, &error);
      if (num_strings_sensor_name > MAX_SOURCE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
        goto done;
      }
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_USE_NVMULTIURISRCBIN)) {
      config->use_nvmultiurisrcbin =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_USE_NVMULTIURISRCBIN, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_STREAM_NAME_DISPLAY)) {
      config->stream_name_display =
          g_key_file_get_boolean (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_STREAM_NAME_DISPLAY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_HTTP_IP)) {
      config->http_ip =
          g_key_file_get_string (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_HTTP_IP, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_HTTP_PORT)) {
      config->http_port =
          g_key_file_get_string (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_HTTP_PORT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_MAX_BATCH_SIZE)) {
      config->max_batch_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_MAX_BATCH_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SGIE_BATCH_SIZE)) {
      config->sgie_batch_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_SGIE_BATCH_SIZE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_EXTRACT_SEI_TYPE5_DATA)) {
        config->extract_sei_type5_data =
            g_key_file_get_integer(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_EXTRACT_SEI_TYPE5_DATA, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_LIST_LOW_LATENCY_MODE)) {
        config->low_latency_mode =
            g_key_file_get_integer(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_LOW_LATENCY_MODE, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_SEI_UUID)) {
        config->sei_uuid =
            g_key_file_get_string(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_SEI_UUID, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_LIST_ENABLE_ERROR_PROPAGATION)) {
        config->enable_error_propagation =
            g_key_file_get_boolean(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_ENABLE_ERROR_PROPAGATION, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_LIST_PROTO_LIB)) {
        config->proto_lib =
            g_key_file_get_string(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_PROTO_LIB, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_LIST_CONN_STR)) {
        config->conn_str =
            g_key_file_get_string(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_CONN_STR, &error);
        CHECK_ERROR(error);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_LIST_TOPIC)) {
        config->topic =
            g_key_file_get_string(key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_TOPIC, &error);
        CHECK_ERROR(error);
    }
    else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_SOURCE_LIST);
    }
  }

  if (g_key_file_has_key (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_URI_LIST, &error)) {
    if (g_key_file_has_key (key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS, &error)) {
      if (num_strings_uri != config->total_num_sources) {
        NVGSTDS_ERR_MSG_V ("Mismatch in URIs provided and num-source-bins.");
        goto done;
      }
      if (num_strings_sensor_id != config->total_num_sources) {
        NVGSTDS_ERR_MSG_V ("Mismatch in Sensor IDs provided and num-source-bins.");
        goto done;
      }
      if (num_strings_sensor_name != config->total_num_sources) {
        NVGSTDS_ERR_MSG_V ("Mismatch in Sensor Names provided and num-source-bins.");
        goto done;
      }
    } else {
      config->total_num_sources = num_strings_uri;
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

static gboolean
set_source_all_configs (NvDsConfig * config, gchar * cfg_file_path)
{
  guint i = 0;
  for (i = 0; i < config->total_num_sources; i++) {
    config->multi_source_config[i] = config->source_attr_all_config;
    config->multi_source_config[i].camera_id = i;
    if (config->uri_list) {
      char *uri = config->uri_list[i];
      if (!uri) {
        NVGSTDS_ERR_MSG_V ("uri %d entry of list is NULL, use valid uri separated by ';' with the source-list section", (i+1));
        return FALSE;
      }
      if (g_str_has_prefix (config->uri_list[i], "file://")) {
        config->multi_source_config[i].type = NV_DS_SOURCE_URI;
        config->multi_source_config[i].uri = g_strdup (uri + 7);
        config->multi_source_config[i].uri =
            g_strdup_printf ("file://%s",
            get_absolute_file_path (cfg_file_path,
                config->multi_source_config[i].uri));
      } else if (g_str_has_prefix (config->uri_list[i], "rtsp://")) {
        config->multi_source_config[i].type = NV_DS_SOURCE_RTSP;
        config->multi_source_config[i].uri = config->uri_list[i];
      } else {
        gchar *source_id_start_ptr = uri + 4;
        gchar *source_id_end_ptr = NULL;
        long camera_id =
            g_ascii_strtoull (source_id_start_ptr, &source_id_end_ptr, 10);
        if (source_id_start_ptr == source_id_end_ptr
            || *source_id_end_ptr != '\0') {
          NVGSTDS_ERR_MSG_V
              ("Incorrect URI for camera source %s. FORMAT: <usb/csi>:<dev_node/sensor_id>",
              uri);
          return FALSE;
        }
        if (g_str_has_prefix (config->uri_list[i], "csi:")) {
          config->multi_source_config[i].type = NV_DS_SOURCE_CAMERA_CSI;
          config->multi_source_config[i].camera_csi_sensor_id = camera_id;
        } else if (g_str_has_prefix (config->uri_list[i], "usb:")) {
          config->multi_source_config[i].type = NV_DS_SOURCE_CAMERA_V4L2;
          config->multi_source_config[i].camera_v4l2_dev_node = camera_id;
        } else {
          NVGSTDS_ERR_MSG_V ("URI %d (%s) not in proper format.", i,
              config->uri_list[i]);
          return FALSE;
        }
      }
    }
  }
  return TRUE;
}

static gboolean
parse_tests (NvDsConfig * config, GKeyFile * key_file)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TESTS, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TESTS_FILE_LOOP)) {
      config->file_loop =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TESTS,
          CONFIG_GROUP_TESTS_FILE_LOOP, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TESTS_PIPELINE_RECREATE_SEC)) {
      config->pipeline_recreate_sec =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TESTS,
          CONFIG_GROUP_TESTS_PIPELINE_RECREATE_SEC, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TESTS);
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

static gboolean
parse_app (NvDsConfig * config, GKeyFile * key_file, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_APP, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_APP_ENABLE_PERF_MEASUREMENT)) {
      config->enable_perf_measurement =
          g_key_file_get_integer (key_file, CONFIG_GROUP_APP,
          CONFIG_GROUP_APP_ENABLE_PERF_MEASUREMENT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_PERF_MEASUREMENT_INTERVAL)) {
      config->perf_measurement_interval_sec =
          g_key_file_get_integer (key_file, CONFIG_GROUP_APP,
          CONFIG_GROUP_APP_PERF_MEASUREMENT_INTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_GIE_OUTPUT_DIR)) {
      config->bbox_dir_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
              CONFIG_GROUP_APP_GIE_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_GIE_TRACK_OUTPUT_DIR)) {
      config->kitti_track_dir_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
              CONFIG_GROUP_APP_GIE_TRACK_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_REID_TRACK_OUTPUT_DIR)) {
      config->reid_track_dir_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
              CONFIG_GROUP_APP_REID_TRACK_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_GLOBAL_GPU_ID)) {
      /** App Level GPU ID is set here if it is present in APP LEVEL config group
       * if gpu_id prop is not set for any component, this global_gpu_id will be used */
      config->global_gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_APP,
            CONFIG_GROUP_APP_GLOBAL_GPU_ID, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_TERMINATED_TRACK_OUTPUT_DIR)) {
      config->terminated_track_output_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
              CONFIG_GROUP_APP_TERMINATED_TRACK_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_SHADOW_TRACK_OUTPUT_DIR)) {
      config->shadow_track_output_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
              CONFIG_GROUP_APP_SHADOW_TRACK_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    }
    else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_APP);
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
parse_config_file (NvDsConfig * config, gchar * cfg_file_path)
{
  GKeyFile *cfg_file = g_key_file_new ();
  GError *error = NULL;
  gboolean ret = FALSE;
  gchar **groups = NULL;
  gchar **group;
  guint i, j;
  guint num_dewarper_source = 0;

  config->source_list_enabled = FALSE;
  config->source_attr_all_parsed = FALSE;

  /** Initialize global gpu id to -1 */
  config->global_gpu_id = -1;

  if (!APP_CFG_PARSER_CAT) {
    GST_DEBUG_CATEGORY_INIT (APP_CFG_PARSER_CAT, "NVDS_CFG_PARSER", 0, NULL);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to load uri file: %s",
        error->message);
    goto done;
  }

  /** App group parsing at top level to set global_gpu_id (if available)
   * before any other group parsing */
  if (g_key_file_has_group (cfg_file, CONFIG_GROUP_APP)) {
    if (!parse_app (config, cfg_file, cfg_file_path)) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group",
          CONFIG_GROUP_APP);
      goto done;
    }
  }

  if (g_key_file_has_group (cfg_file, CONFIG_GROUP_SOURCE_LIST)) {
    if (!parse_source_list (config, cfg_file, cfg_file_path)) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group",
          CONFIG_GROUP_SOURCE_LIST);
      goto done;
    }
    config->num_source_sub_bins = config->total_num_sources;
    config->source_list_enabled = TRUE;
    if (!g_key_file_has_group (cfg_file, CONFIG_GROUP_SOURCE_ALL)) {
      NVGSTDS_ERR_MSG_V ("[source-attr-all] group not present.");
      ret = FALSE;
      goto done;
    }
    g_key_file_remove_group (cfg_file, CONFIG_GROUP_SOURCE_LIST, &error);
  }
  if (g_key_file_has_group (cfg_file, CONFIG_GROUP_SOURCE_ALL)) {
    /** set gpu_id for source component using global_gpu_id(if available) */
    if (config->global_gpu_id != -1) {
      config->source_attr_all_config.gpu_id = config->global_gpu_id;
    }
    /** if gpu_id for source component is present,
     * it will override the value set using global_gpu_id in parse_source function */
    if (!parse_source (&config->source_attr_all_config,
            cfg_file, (gchar*)CONFIG_GROUP_SOURCE_ALL, cfg_file_path)) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group",
          CONFIG_GROUP_SOURCE_LIST);
      goto done;
    }
    config->source_attr_all_parsed = TRUE;
    if (!set_source_all_configs (config, cfg_file_path)) {
      ret = FALSE;
      goto done;
    }
    g_key_file_remove_group (cfg_file, CONFIG_GROUP_SOURCE_ALL, &error);
  }

  groups = g_key_file_get_groups (cfg_file, NULL);
  for (group = groups; *group; group++) {
    gboolean parse_err = FALSE;
    GST_CAT_DEBUG (APP_CFG_PARSER_CAT, "Parsing group: %s", *group);
    if (!strncmp (*group, CONFIG_GROUP_SOURCE,
            sizeof (CONFIG_GROUP_SOURCE) - 1)) {
      if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
        ret = FALSE;
        goto done;
      }
      gchar *source_id_start_ptr = *group + strlen (CONFIG_GROUP_SOURCE);
      gchar *source_id_end_ptr = NULL;
      guint index =
          g_ascii_strtoull (source_id_start_ptr, &source_id_end_ptr, 10);
      if (source_id_start_ptr == source_id_end_ptr
          || *source_id_end_ptr != '\0') {
        NVGSTDS_ERR_MSG_V
            ("Source group \"[%s]\" is not in the form \"[source<%%d>]\"",
            *group);
        ret = FALSE;
        goto done;
      }
      guint source_id = 0;
      if (config->source_list_enabled) {
        if (index >= config->total_num_sources) {
          NVGSTDS_ERR_MSG_V
              ("Invalid source group index %d, index cannot exceed %d", index,
              config->total_num_sources);
          ret = FALSE;
          goto done;
        }
        source_id = index;
        NVGSTDS_INFO_MSG_V ("Some parameters to be overwritten for group [%s]",
            *group);
      } else {
        source_id = config->num_source_sub_bins;
      }
      /**  set gpu_id for source component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->multi_source_config[source_id].gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for source component is present,
       * it will override the value set using global_gpu_id in parse_source function */
      parse_err = !parse_source (&config->multi_source_config[source_id],
          cfg_file, *group, cfg_file_path);
      if (config->source_list_enabled
          && config->multi_source_config[source_id].type ==
          NV_DS_SOURCE_URI_MULTIPLE) {
        NVGSTDS_ERR_MSG_V
            ("MultiURI support not available if [source-list] is provided");
        ret = FALSE;
        goto done;
      }
      if (config->multi_source_config[source_id].enable
          && !config->source_list_enabled) {
        config->num_source_sub_bins++;
      }
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_STREAMMUX)) {
      /** set gpu_id for streammux component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->streammux_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for streammux component is present,
       * it will override the value set using global_gpu_id in parse_streammux function */
      parse_err =
          !parse_streammux (&config->streammux_config, cfg_file, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_OSD)) {
      /** set gpu_id for osd component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->osd_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for osd component is present,
       * it will override the value set using global_gpu_id in parse_osd function */
      parse_err = !parse_osd (&config->osd_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_SEGVISUAL)) {
      /** set gpu_id for segvisual component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->segvisual_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for segvisual component is present,
       * it will override the value set using global_gpu_id in parse_segvisual function */
      parse_err = !parse_segvisual(&config->segvisual_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_PREPROCESS)) {
      parse_err =
          !parse_preprocess (&config->preprocess_config, cfg_file,
          *group, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_PRIMARY_GIE)) {
      /** set gpu_id for primary gie component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->primary_gie_config.gpu_id = config->global_gpu_id;
        config->primary_gie_config.is_gpu_id_set = TRUE;
      }
      /** if gpu_id for primary gie component is present,
       * it will override the value set using global_gpu_id in parse_gie function */
      parse_err =
          !parse_gie (&config->primary_gie_config, cfg_file,
          (gchar*)CONFIG_GROUP_PRIMARY_GIE, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TRACKER)) {
      /** set gpu_id for tracker component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->tracker_config.gpu_id = config->global_gpu_id;
      }
      /**  if gpu_id for tracker component is present,
       * it will override the value set using global_gpu_id in parse_tracker function */
      parse_err =
          !parse_tracker (&config->tracker_config, cfg_file, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_VISIONENCODER)) {
      /** set gpu_id for visionencoder component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->visionencoder_config.gpu_id = config->global_gpu_id;
      }
      parse_err = !parse_visionencoder (&config->visionencoder_config, cfg_file, *group, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TEXT_EMBEDDER)) {
      parse_err =
          !parse_text_embedder (&config->text_embedder_config, cfg_file,
          cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_REPLAY)) {
      parse_err = !parse_replay (&config->replay_config, cfg_file, cfg_file_path);
    }

    if (!strncmp (*group, CONFIG_GROUP_SECONDARY_GIE,
            sizeof (CONFIG_GROUP_SECONDARY_GIE) - 1)) {
      if (config->num_secondary_gie_sub_bins == MAX_SECONDARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs",
            MAX_SECONDARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }
      /** set gpu_id for secondary gie component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].gpu_id = config->global_gpu_id;
        config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].is_gpu_id_set = TRUE;
      }
      /**  if gpu_id for secondary gie component is present,
       * it will override the value set using global_gpu_id in parse_gie function */
      parse_err =
          !parse_gie (&config->
          secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins],
          cfg_file, *group, cfg_file_path);
      if (config->secondary_gie_sub_bin_config[config->
              num_secondary_gie_sub_bins].enable) {
        config->num_secondary_gie_sub_bins++;
      }
    }

    if (!strncmp (*group, CONFIG_GROUP_SECONDARY_PREPROCESS,
            sizeof (CONFIG_GROUP_SECONDARY_PREPROCESS) - 1)) {
      if (config->num_secondary_preprocess_sub_bins ==
          MAX_SECONDARY_PREPROCESS_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary PREPROCESSs",
            MAX_SECONDARY_PREPROCESS_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_preprocess (&config->
          secondary_preprocess_sub_bin_config
          [config->num_secondary_preprocess_sub_bins], cfg_file, *group,
          cfg_file_path);

      if (config->secondary_preprocess_sub_bin_config[config->
              num_secondary_preprocess_sub_bins].enable) {
        config->num_secondary_preprocess_sub_bins++;
      }
    }

    if (!strncmp (*group, CONFIG_GROUP_SINK, sizeof (CONFIG_GROUP_SINK) - 1)) {
      if (config->num_sink_sub_bins == MAX_SINK_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sinks", MAX_SINK_BINS);
        ret = FALSE;
        goto done;
      }
      /** set gpu_id for sink component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        GError *error = NULL;
        if (g_key_file_get_integer (cfg_file, *group,
            "enable", &error) == TRUE && error == NULL) {
              config->sink_bin_sub_bin_config[config->num_sink_sub_bins].encoder_config.gpu_id = config->sink_bin_sub_bin_config[config->num_sink_sub_bins].render_config.gpu_id = config->global_gpu_id;
        }
      }
      /** if gpu_id for sink component is present,
       * it will override the value set using global_gpu_id in parse_sink function */
      parse_err =
          !parse_sink (&config->sink_bin_sub_bin_config[config->
              num_sink_sub_bins], cfg_file, *group, cfg_file_path);
      if (config->sink_bin_sub_bin_config[config->num_sink_sub_bins].enable) {
        config->num_sink_sub_bins++;
      }
    }

    if (!strncmp (*group, CONFIG_GROUP_MSG_CONSUMER,
            sizeof (CONFIG_GROUP_MSG_CONSUMER) - 1)) {
      if (config->num_message_consumers == MAX_MESSAGE_CONSUMERS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d consumers",
            MAX_MESSAGE_CONSUMERS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_msgconsumer (&config->message_consumer_config[config->
              num_message_consumers], cfg_file, *group, cfg_file_path);

      if (config->message_consumer_config[config->num_message_consumers].enable) {
        config->num_message_consumers++;
      }
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TILED_DISPLAY)) {
      /** set gpu_id for tiled display component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->tiled_display_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for tiled display component is present,
       * it will override the value set using global_gpu_id in parse_tiled_display function */
      parse_err =
          !parse_tiled_display (&config->tiled_display_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_IMG_SAVE)) {
      /** set gpu_id for image save component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->image_save_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for image save component is present,
       * it will override the value set using global_gpu_id in parse_image_save function */
      parse_err =
          !parse_image_save (&config->image_save_config, cfg_file, *group,
          cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_DSANALYTICS)) {
      parse_err =
          !parse_dsanalytics (&config->dsanalytics_config, cfg_file,
          cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_DSEXAMPLE)) {
      /**  set gpu_id for dsexample component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->dsexample_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for dsexample component is present,
       * it will override the value set using global_gpu_id in parse_dsexample function */
      parse_err = !parse_dsexample (&config->dsexample_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_MSG_CONVERTER)) {
      parse_err =
          !parse_msgconv (&config->msg_conv_config, cfg_file, *group,
          cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TESTS)) {
      parse_err = !parse_tests (config, cfg_file);
    }

    if (!strncmp (*group, CONFIG_GROUP_DEWARPER, strlen(*group)-1)) {
      guint source_id = 0;
      {
        gchar *source_id_start_ptr = *group + strlen (CONFIG_GROUP_DEWARPER);
        gchar *source_id_end_ptr = NULL;
        source_id = g_ascii_strtoull (source_id_start_ptr, &source_id_end_ptr, 10);
        if (source_id_start_ptr == source_id_end_ptr || *source_id_end_ptr != '\0') {
        NVGSTDS_ERR_MSG_V
            ("dewarper group \"[%s]\" is not in the form \"[dewarper<%%d>]\"", *group);
        ret = FALSE;
        goto done;}
      }
      /** set gpu_id for dewarper component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->multi_source_config[source_id].dewarper_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for dewarper component is present,
       * it will override the value set using global_gpu_id in parse_dewarper function */
      parse_err = !parse_dewarper (&config->multi_source_config[source_id].dewarper_config,
        cfg_file, cfg_file_path, *group);
      if(config->multi_source_config[source_id].dewarper_config.enable)
        num_dewarper_source++;
      if( num_dewarper_source > config->num_source_sub_bins) {
        NVGSTDS_ERR_MSG_V ("Dewarper max numbers %u should be less than number of sources %u",
          num_dewarper_source, config->num_source_sub_bins);
        ret = FALSE;
        goto done;
      }
    }

    if (parse_err) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group", *group);
      goto done;
    }
  }

  /* Updating batch size when source list is enabled */
  if (config->source_list_enabled == TRUE) {
    /* For streammux and pgie, batch size is set to number of sources */
    config->streammux_config.batch_size = config->num_source_sub_bins;
    config->primary_gie_config.batch_size = config->num_source_sub_bins;
    if (config->sgie_batch_size != 0) {
      for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
        config->secondary_gie_sub_bin_config[i].batch_size =
            config->sgie_batch_size;
      }
    }
  }

  for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    if (config->secondary_gie_sub_bin_config[i].unique_id ==
        config->primary_gie_config.unique_id) {
      NVGSTDS_ERR_MSG_V ("Non unique gie ids found");
      ret = FALSE;
      goto done;
    }
  }

  for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    for (j = i + 1; j < config->num_secondary_gie_sub_bins; j++) {
      if (config->secondary_gie_sub_bin_config[i].unique_id ==
          config->secondary_gie_sub_bin_config[j].unique_id) {
        NVGSTDS_ERR_MSG_V ("Non unique gie id %d found",
            config->secondary_gie_sub_bin_config[i].unique_id);
        ret = FALSE;
        goto done;
      }
    }
  }

  for (i = 0; i < config->num_source_sub_bins; i++) {
    if (config->multi_source_config[i].type == NV_DS_SOURCE_URI_MULTIPLE) {
      if (config->multi_source_config[i].num_sources < 1) {
        config->multi_source_config[i].num_sources = 1;
      }
      for (j = 1; j < config->multi_source_config[i].num_sources; j++) {
        if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
          NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
          ret = FALSE;
          goto done;
        }
        memcpy (&config->multi_source_config[config->num_source_sub_bins],
            &config->multi_source_config[i],
            sizeof (config->multi_source_config[i]));
        config->multi_source_config[config->num_source_sub_bins].type =
            NV_DS_SOURCE_URI;
        config->multi_source_config[config->num_source_sub_bins].uri =
            g_strdup_printf (config->
            multi_source_config[config->num_source_sub_bins].uri, j);
        config->num_source_sub_bins++;
      }
      config->multi_source_config[i].type = NV_DS_SOURCE_URI;
      config->multi_source_config[i].uri =
          g_strdup_printf (config->multi_source_config[i].uri, 0);
    }
  }
  ret = TRUE;

done:
  if (cfg_file) {
    g_key_file_free (cfg_file);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
