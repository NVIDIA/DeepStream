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

#include <gst/gst.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "deepstream_app.h"
#include "nvds_tracker_meta.h"
#include "nvds_msgapi.h"
#include "nvds_utils.h"

#define MAX_DISPLAY_LEN 64
static guint demux_batch_num = 0;

GST_DEBUG_CATEGORY_EXTERN (NVDS_APP);

GQuark _dsmeta_quark;

#define CEIL(a,b) ((a + b - 1) / b)

/** MSGAPI function pointers for error propagation */
static NvDsMsgApiHandle (*msgapi_connect_fn)(char *connection_str, nvds_msgapi_connect_cb_t connect_cb, char *config_path) = NULL;
static NvDsMsgApiErrorType (*msgapi_send_fn)(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf) = NULL;
static NvDsMsgApiErrorType (*msgapi_disconnect_fn)(NvDsMsgApiHandle h_ptr) = NULL;

/**
 * @brief  Add the (nvmsgconv->nvmsgbroker) sink-bin to the
 *         overall DS pipeline (if any configured) and link the same to
 *         common_elements.tee (This tee connects
 *         the common analytics path to Tiler/display-sink and
 *         to configured broker sink if any)
 *         NOTE: This API shall return TRUE if there are no
 *         broker sinks to add to pipeline
 *
 * @param  appCtx [IN]
 * @return TRUE if succussful; FALSE otherwise
 */
static gboolean add_and_link_broker_sink (AppCtx * appCtx);

/**
 * @brief  Checks if there are any [sink] groups
 *         configured for source_id=provided source_id
 *         NOTE: source_id key and this API is valid only when we
 *         disable [tiler] and thus use demuxer for individual
 *         stream out
 * @param  config [IN] The DS Pipeline configuration struct
 * @param  source_id [IN] Source ID for which a specific [sink]
 *         group is searched for
 */
static gboolean is_sink_available_for_source_id (NvDsConfig * config,
    guint source_id);

static NvDsSensorInfo* s_sensor_info_create(NvDsSensorInfo* sensor_info);
static void s_sensor_info_destroy(NvDsSensorInfo* sensor_info);

static NvDsSensorInfo* s_sensor_info_create(NvDsSensorInfo* sensor_info) {
  NvDsSensorInfo* sensorInfoToHash = (NvDsSensorInfo*)g_malloc0(sizeof(NvDsSensorInfo));
  *sensorInfoToHash = *sensor_info;
  sensorInfoToHash->sensor_id = (gchar const*)g_strdup(sensor_info->sensor_id);
  sensorInfoToHash->sensor_name = (gchar const*)g_strdup(sensor_info->sensor_name);
  sensorInfoToHash->uri = (gchar const*)g_strdup(sensor_info->uri);
  return sensorInfoToHash;
}

static void s_sensor_info_destroy(NvDsSensorInfo* sensor_info) {
  if(!sensor_info)
    return;
  if(sensor_info->sensor_id) {
    g_free((void*)sensor_info->sensor_id);
  }
  if(sensor_info->sensor_name) {
    g_free((void*)sensor_info->sensor_name);
  }

  g_free(sensor_info);
}

static void s_sensor_info_callback_stream_added (AppCtx *appCtx, NvDsSensorInfo* sensorInfo) {

  NvDsSensorInfo* sensorInfoToHash = s_sensor_info_create(sensorInfo);
  /** save the sensor info into the hash map */
  g_hash_table_insert (appCtx->sensorInfoHash, sensorInfo->source_id + (char *)NULL, sensorInfoToHash);
}

static void s_sensor_info_callback_stream_removed (AppCtx *appCtx, NvDsSensorInfo* sensorInfo) {

  NvDsSensorInfo* sensorInfoFromHash = get_sensor_info(appCtx, sensorInfo->source_id);
  /** remove the sensor info from the hash map */
  if(sensorInfoFromHash) {
    g_hash_table_remove(appCtx->sensorInfoHash, sensorInfo->source_id + (gchar*)NULL);
    s_sensor_info_destroy(sensorInfoFromHash);
  }
}

NvDsSensorInfo* get_sensor_info(AppCtx* appCtx, guint source_id) {
  NvDsSensorInfo* sensorInfo = (NvDsSensorInfo*)g_hash_table_lookup(appCtx->sensorInfoHash,
        source_id + (gchar*)NULL);
  return sensorInfo;
}

/*Note: Below callbacks/functions defined for FPS logging,
 *  when nvmultiurisrcbin is being used*/
static NvDsFPSSensorInfo* s_fps_sensor_info_create(NvDsFPSSensorInfo* sensor_info);
NvDsFPSSensorInfo* get_fps_sensor_info(AppCtx* appCtx, guint source_id);
static void s_fps_sensor_info_destroy(NvDsFPSSensorInfo* sensor_info);

static NvDsFPSSensorInfo* s_fps_sensor_info_create(NvDsFPSSensorInfo* sensor_info) {
  NvDsFPSSensorInfo* fpssensorInfoToHash = (NvDsFPSSensorInfo*)g_malloc0(sizeof(NvDsFPSSensorInfo));
  *fpssensorInfoToHash = *sensor_info;
  fpssensorInfoToHash->uri = (gchar const*)g_strdup(sensor_info->uri);
  fpssensorInfoToHash->source_id = sensor_info->source_id;
  fpssensorInfoToHash->sensor_id = (gchar const*)g_strdup(sensor_info->sensor_id);
  fpssensorInfoToHash->sensor_name = (gchar const*)g_strdup(sensor_info->sensor_name);
  return fpssensorInfoToHash;
}

static void s_fps_sensor_info_destroy(NvDsFPSSensorInfo* sensor_info) {
  if(!sensor_info)
    return;
  if(sensor_info->sensor_id) {
    g_free((void*)sensor_info->sensor_id);
  }
  if(sensor_info->sensor_name) {
    g_free((void*)sensor_info->sensor_name);
  }
  if(sensor_info->uri) {
    g_free((void*)sensor_info->uri);
  }

  g_free(sensor_info);
}

static void s_fps_sensor_info_callback_stream_added (AppCtx *appCtx, NvDsFPSSensorInfo* sensorInfo) {

  NvDsFPSSensorInfo* fpssensorInfoToHash = s_fps_sensor_info_create(sensorInfo);
  /** save the sensor info into the hash map */
  g_hash_table_insert (appCtx->perf_struct.FPSInfoHash, GUINT_TO_POINTER(sensorInfo->source_id), fpssensorInfoToHash);
}

NvDsFPSSensorInfo* get_fps_sensor_info(AppCtx* appCtx, guint source_id) {
  NvDsFPSSensorInfo* sensorInfo = (NvDsFPSSensorInfo*)g_hash_table_lookup(appCtx->perf_struct.FPSInfoHash,
        GUINT_TO_POINTER(source_id));
  return sensorInfo;
}

static void s_fps_sensor_info_callback_stream_removed (AppCtx *appCtx, NvDsFPSSensorInfo* sensorInfo) {

  NvDsFPSSensorInfo* fpsensorInfoFromHash = get_fps_sensor_info(appCtx, sensorInfo->source_id);
  /** remove the sensor info from the hash map */
  if(fpsensorInfoFromHash) {
    g_hash_table_remove(appCtx->perf_struct.FPSInfoHash, GUINT_TO_POINTER(sensorInfo->source_id));
    s_fps_sensor_info_destroy(fpsensorInfoFromHash);
  }
}

/**
 * msgapi_init:
 * @appCtx: Application context with error propagation configuration
 *
 * Initializes the message API for sending error events to message broker.
 * Loads the protocol library, connects to broker, and sets up function pointers.
 *
 * Returns: TRUE if initialization is successful, FALSE otherwise
 */
static gboolean msgapi_init(AppCtx *appCtx) {
  if (!appCtx->config.enable_error_propagation) {
    return FALSE;
  }

  // Check if required properties are set
  if (!appCtx->config.proto_lib || !appCtx->config.conn_str) {
    g_print("[MSGAPI_ERROR] Initialization failed: proto-lib or conn-str not set\n");
    return FALSE;
  }

  // Load the MSGAPI library
  appCtx->msgapi_lib_handle = dlopen(appCtx->config.proto_lib, RTLD_LAZY);
  if (!appCtx->msgapi_lib_handle) {
    g_print("[MSGAPI_ERROR] Failed to load library: %s\n", dlerror());
    g_print("[MSGAPI_ERROR] Make sure library exists at: %s\n", appCtx->config.proto_lib);
    return FALSE;
  }

  // Resolve function symbols
  *(void**)(&msgapi_connect_fn) = dlsym(appCtx->msgapi_lib_handle, "nvds_msgapi_connect");
  *(void**)(&msgapi_send_fn) = dlsym(appCtx->msgapi_lib_handle, "nvds_msgapi_send");
  *(void**)(&msgapi_disconnect_fn) = dlsym(appCtx->msgapi_lib_handle, "nvds_msgapi_disconnect");

  char *error = dlerror();
  if (error != NULL) {
    g_print("[MSGAPI_ERROR] Failed to resolve symbols: %s\n", error);
    dlclose(appCtx->msgapi_lib_handle);
    appCtx->msgapi_lib_handle = NULL;
    return FALSE;
  }

  appCtx->msgapi_conn_handle = msgapi_connect_fn(appCtx->config.conn_str, NULL, NULL);
  if (!appCtx->msgapi_conn_handle) {
    g_print("[MSGAPI_ERROR] Failed to connect to message broker: %s\n", appCtx->config.conn_str);
    dlclose(appCtx->msgapi_lib_handle);
    appCtx->msgapi_lib_handle = NULL;
    return FALSE;
  }

  return TRUE;
}

/**
 * msgapi_send_message:
 * @appCtx: Application context
 * @msg_type: GStreamer message type (GST_MESSAGE_ERROR or GST_MESSAGE_WARNING)
 * @sensor_id: Sensor ID (or NULL for generic messages)
 * @source: Name of the component reporting the message
 * @event: Detailed description of the event
 *
 * Sends a GStreamer event message to the message broker.
 * Topic naming: error_events_<topic> where <topic> is from config
 * (defaults to "deepstream" if not specified)
 *
 * Message JSON Schema:
 * {
 *   "stream_id": "<sensor_id or N/A>",
 *   "timestamp": "YYYY-MM-DD HH:MM:SS",
 *   "type": "CRITICAL" (for errors) or "FUNCTIONAL" (for warnings),
 *   "source": "<source>",
 *   "event": "<event_description>"
 * }
 *
 * Returns: TRUE if message was sent successfully, FALSE otherwise
 */
static gboolean msgapi_send_message(AppCtx *appCtx, GstMessageType msg_type, const char *sensor_id, const char *source, const char *event) {
  if (!msgapi_send_fn || !appCtx->msgapi_conn_handle) {
    return FALSE;
  }

  // Determine message type string based on GstMessageType
  const char *type_str;
  const char *log_prefix;
  if (msg_type == GST_MESSAGE_ERROR) {
    type_str = "CRITICAL";
    log_prefix = "error";
  } else if (msg_type == GST_MESSAGE_WARNING) {
    type_str = "FUNCTIONAL";
    log_prefix = "warning";
  } else {
    g_print("[MSGAPI_ERROR] Unsupported message type: %d\n", msg_type);
    return FALSE;
  }

  GDateTime *now = g_date_time_new_now_local();
  gchar *timestamp = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
  g_date_time_unref(now);

  // Determine stream_id for JSON message
  const gchar *stream_id_str = sensor_id ? sensor_id : "N/A";

  // Get hostname with IP for source field
  gchar *hostname = nvds_get_hostname_with_ip();

  gchar *message_payload = g_strdup_printf(
    "{\n"
    "  \"stream_id\" : \"%s\",\n"
    "  \"timestamp\" : \"%s\",\n"
    "  \"type\" : \"%s\",\n"
    "  \"source\" : \"%s\",\n"
    "  \"event\" : \"%s\"\n"
    "}\n",
    stream_id_str,
    timestamp,
    type_str,
    hostname,
    event
  );

  // Determine topic name: use user-defined topic directly for all error messages
  gchar *topic_cstr;
  if (appCtx->config.topic) {
    topic_cstr = g_strdup(appCtx->config.topic);
  } else {
    topic_cstr = g_strdup("deepstream");
  }

  NvDsMsgApiErrorType result = msgapi_send_fn(appCtx->msgapi_conn_handle,
                                               topic_cstr,
                                               (const uint8_t*)message_payload,
                                               strlen(message_payload));

  if (result != NVDS_MSGAPI_OK) {
    g_print("[MSGAPI_%s] Failed to send %s message to topic: %s\n",
            msg_type == GST_MESSAGE_ERROR ? "ERROR" : "WARNING", log_prefix, topic_cstr);
    g_free(timestamp);
    g_free(hostname);
    g_free(message_payload);
    g_free(topic_cstr);
    return FALSE;
  }

  g_print("[MSGAPI] Sent %s to topic '%s': %s - %s\n", log_prefix, topic_cstr, hostname, event);
  g_free(timestamp);
  g_free(hostname);
  g_free(message_payload);
  g_free(topic_cstr);
  return TRUE;
}

/**
 * msgapi_cleanup:
 * @appCtx: Application context
 *
 * Cleans up the message API resources
 */
static void msgapi_cleanup(AppCtx *appCtx) {
  if (appCtx->msgapi_conn_handle && msgapi_disconnect_fn) {
    msgapi_disconnect_fn(appCtx->msgapi_conn_handle);
    appCtx->msgapi_conn_handle = NULL;
    g_print("[MSGAPI] Disconnected from message broker\n");
  }

  if (appCtx->msgapi_lib_handle) {
    dlclose(appCtx->msgapi_lib_handle);
    appCtx->msgapi_lib_handle = NULL;
  }

  msgapi_connect_fn = NULL;
  msgapi_send_fn = NULL;
  msgapi_disconnect_fn = NULL;
}

/**
 * parse_msgapi_settings_only:
 * @appCtx: Application context to populate with msgapi settings
 * @cfg_file_path: Path to configuration file
 *
 * Lightweight config parser that extracts only msgapi-related settings.
 * This allows msgapi initialization even if full config parsing fails later.
 * Parses: enable-error-propagation, proto-lib, conn-str, topic
 *
 * Returns: TRUE if at least some msgapi settings were found, FALSE otherwise
 */
static gboolean parse_msgapi_settings_only (AppCtx *appCtx, const gchar *cfg_file_path)
{
  GKeyFile *key_file = g_key_file_new ();
  GError *error = NULL;
  gboolean ret = FALSE;

  if (!g_key_file_load_from_file (key_file, cfg_file_path, G_KEY_FILE_NONE, &error)) {
    g_print ("[MSGAPI] Cannot pre-parse config file: %s\n", error ? error->message : "unknown error");
    goto done;
  }

  /* Try to parse msgapi settings from [source-list] or [application] group */
  if (g_key_file_has_group (key_file, "source-list")) {
    if (g_key_file_has_key (key_file, "source-list", "enable-error-propagation", NULL)) {
      appCtx->config.enable_error_propagation =
          g_key_file_get_boolean (key_file, "source-list", "enable-error-propagation", NULL);
    }
    if (g_key_file_has_key (key_file, "source-list", "proto-lib", NULL)) {
      appCtx->config.proto_lib =
          g_key_file_get_string (key_file, "source-list", "proto-lib", NULL);
    }
    if (g_key_file_has_key (key_file, "source-list", "conn-str", NULL)) {
      appCtx->config.conn_str =
          g_key_file_get_string (key_file, "source-list", "conn-str", NULL);
    }
    if (g_key_file_has_key (key_file, "source-list", "topic", NULL)) {
      appCtx->config.topic =
          g_key_file_get_string (key_file, "source-list", "topic", NULL);
    }
    ret = TRUE;
  }

done:
  if (error) {
    g_error_free (error);
  }
  if (key_file) {
    g_key_file_free (key_file);
  }
  return ret;
}

/**
 * msgapi_init_early:
 * @appCtx: Application context
 * @cfg_file_path: Configuration file path
 *
 * Generic helper to initialize msgapi BEFORE full config parsing.
 * This enables error reporting even if config parsing fails.
 * Combines pre-parsing and initialization in one call.
 *
 * Returns: TRUE if msgapi was successfully initialized, FALSE otherwise
 */
gboolean msgapi_init_early (AppCtx *appCtx, const gchar *cfg_file_path)
{
  /* Pre-parse only msgapi settings */
  if (!parse_msgapi_settings_only (appCtx, cfg_file_path)) {
    /* Config file doesn't exist or has no msgapi settings - this is OK, not an error */
    g_print("[MSGAPI] Skipping early initialization - config file has no msgapi settings or could not be parsed\n");
    return FALSE;
  }

  /* Try to initialize msgapi */
  if (appCtx->config.enable_error_propagation) {
    if (!msgapi_init(appCtx)) {
      g_print("[MSGAPI_WARN] Failed to initialize message API for error propagation\n");
      return FALSE;
    } else {
      g_print("[MSGAPI] Successfully initialized message API before full config parsing\n");
      return TRUE;
    }
  }

  return FALSE;
}

/**
 * msgapi_report_error_and_cleanup:
 * @appCtx: Application context
 * @error_message: Error description to send (or NULL for generic message)
 *
 * Generic error reporting function called before application exit.
 * Sends error to message broker (if initialized) and cleans up resources.
 * This catches ALL errors from ANY point in the code.
 *
 * Usage: Call this from the done: label in main() when return_value == -1
 */
void msgapi_report_error_and_cleanup (AppCtx *appCtx, const gchar *error_message)
{
  if (appCtx->config.enable_error_propagation && appCtx->msgapi_conn_handle) {
    const gchar *msg = error_message ? error_message : "Application terminated due to error";
    msgapi_send_message(appCtx, GST_MESSAGE_ERROR, NULL, "deepstream-app", msg);
  }

  msgapi_cleanup(appCtx);
}

/**
 * callback function to receive messages from components
 * in the pipeline.
 */
static gboolean
bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  AppCtx *appCtx = (AppCtx *) data;
  GST_CAT_DEBUG (NVDS_APP,
      "Received message on bus: source %s, msg_type %s",
      GST_MESSAGE_SRC_NAME (message), GST_MESSAGE_TYPE_NAME (message));
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_INFO:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_info (message, &error, &debuginfo);
      g_printerr ("INFO from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_WARNING:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_warning (message, &error, &debuginfo);
      g_printerr ("WARNING from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }

      // Send warning to message broker if error propagation is enabled
      if (appCtx->config.enable_error_propagation && appCtx->msgapi_conn_handle) {
        // Filter out specific warnings that are informational/normal operational states
        // and shouldn't flood the message broker
        if (strstr(error->message, "No Sources found at the input of muxer. Waiting for sources")) {
          // Skip sending this warning - it's a normal state when waiting for sources
          g_error_free (error);
          g_free (debuginfo);
          break;
        }

        NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
        GstElement *msg_src_elem = (GstElement *) GST_MESSAGE_SRC (message);
        gboolean bin_found = FALSE;
        const gchar *sensor_id = NULL;
        guint i = 0;

        /* Find the source bin which generated the warning to get sensor_id. */
        while (msg_src_elem && !bin_found) {
          for (i = 0; i < bin->num_bins && !bin_found; i++) {
            if (bin->sub_bins[i].src_elem == msg_src_elem ||
                bin->sub_bins[i].bin == msg_src_elem) {
              bin_found = TRUE;
              // Get sensor info if available (REST API sources)
              NvDsSensorInfo* sensorInfo = get_sensor_info(appCtx, i);
              if (sensorInfo && sensorInfo->sensor_id) {
                sensor_id = sensorInfo->sensor_id;
              }
              break;
            }
          }
          msg_src_elem = GST_ELEMENT_PARENT (msg_src_elem);
        }

        // Send only the warning message without the element name prefix
        msgapi_send_message(appCtx, GST_MESSAGE_WARNING, sensor_id, "deepstream-app", error->message);
      }

      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_ERROR:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      const gchar *attempts_error =
          "Reconnection attempts exceeded for all sources or EOS received.";
      guint i = 0;
      gst_message_parse_error (message, &error, &debuginfo);

      if (strstr (error->message, attempts_error)) {
        g_print
            ("Reconnection attempt  exceeded or EOS received for all sources."
            " Exiting.\n");
        g_error_free (error);
        g_free (debuginfo);
        appCtx->return_value = 0;
        appCtx->quit = TRUE;
        return TRUE;
      }

      g_printerr ("ERROR from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }

      NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
      GstElement *msg_src_elem = (GstElement *) GST_MESSAGE_SRC (message);
      gboolean bin_found = FALSE;
      const gchar *sensor_id = NULL;

      /* Find the source bin which generated the error to get sensor_id. */
      while (msg_src_elem && !bin_found) {
        for (i = 0; i < bin->num_bins && !bin_found; i++) {
          if (bin->sub_bins[i].src_elem == msg_src_elem ||
              bin->sub_bins[i].bin == msg_src_elem) {
            bin_found = TRUE;
            // Get sensor info if available (REST API sources)
            NvDsSensorInfo* sensorInfo = get_sensor_info(appCtx, i);
            if (sensorInfo && sensorInfo->sensor_id) {
              sensor_id = sensorInfo->sensor_id;
            }
            break;
          }
        }
        msg_src_elem = GST_ELEMENT_PARENT (msg_src_elem);
      }

      // Send error to message broker if error propagation is enabled
      if (appCtx->config.enable_error_propagation && appCtx->msgapi_conn_handle) {
        // Send only the error message without the element name prefix
        msgapi_send_message(appCtx, GST_MESSAGE_ERROR, sensor_id, "deepstream-app", error->message);
      }

      if ((i != bin->num_bins) &&
          (appCtx->config.multi_source_config[0].type == NV_DS_SOURCE_RTSP)) {
        // Error from one of RTSP source.
        NvDsSrcBin *subBin = &bin->sub_bins[i];

        if (!subBin->reconfiguring ||
            g_strrstr (debuginfo, "500 (Internal Server Error)")) {
          subBin->reconfiguring = TRUE;
          g_timeout_add (0, reset_source_pipeline, subBin);
        }
        g_error_free (error);
        g_free (debuginfo);
        return TRUE;
      }

      if (appCtx->config.multi_source_config[0].type ==
          NV_DS_SOURCE_CAMERA_V4L2) {
        if (g_strrstr (debuginfo, "reason not-negotiated (-4)")) {
          NVGSTDS_INFO_MSG_V
              ("incorrect camera parameters provided, please provide supported resolution and frame rate\n");
        }

        if (g_strrstr (debuginfo, "Buffer pool activation failed")) {
          NVGSTDS_INFO_MSG_V ("usb bandwidth might be saturated\n");
        }
      }

      g_error_free (error);
      g_free (debuginfo);
      appCtx->return_value = -1;
      appCtx->quit = TRUE;
      break;
    }
    case GST_MESSAGE_STATE_CHANGED:{
      GstState oldstate, newstate;
      gst_message_parse_state_changed (message, &oldstate, &newstate, NULL);
      if (GST_ELEMENT (GST_MESSAGE_SRC (message)) == appCtx->pipeline.pipeline) {
        switch (newstate) {
          case GST_STATE_PLAYING:
            NVGSTDS_INFO_MSG_V ("Pipeline running\n");
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->pipeline.
                    pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-playing");
            break;
          case GST_STATE_PAUSED:
            if (oldstate == GST_STATE_PLAYING) {
              NVGSTDS_INFO_MSG_V ("Pipeline paused\n");
            }
            break;
          case GST_STATE_READY:
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->
                    pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                "ds-app-ready");
            if (oldstate == GST_STATE_NULL) {
              NVGSTDS_INFO_MSG_V ("Pipeline ready\n");
            } else {
              NVGSTDS_INFO_MSG_V ("Pipeline stopped\n");
            }
            break;
          case GST_STATE_NULL:
            g_mutex_lock (&appCtx->app_lock);
            g_cond_broadcast (&appCtx->app_cond);
            g_mutex_unlock (&appCtx->app_lock);
            break;
          default:
            break;
        }
      }
      break;
    }
    case GST_MESSAGE_EOS:{
      /*
       * In normal scenario, this would use g_main_loop_quit() to exit the
       * loop and release the resources. Since this application might be
       * running multiple pipelines through configuration files, it should wait
       * till all pipelines are done.
       */
      if (appCtx->config.use_nvmultiurisrcbin) {
        appCtx->eos_received = TRUE;
        gboolean app_quit = TRUE;
        for (int i = 0; i < MAX_SOURCE_BINS; i++) {
          if (appCtx->config.multi_source_config[i].type == NV_DS_SOURCE_RTSP) {
              if (appCtx->config.multi_source_config[i].rtsp_reconnect_attempt_exceeded != TRUE) {
                app_quit = FALSE;
              } else {
                app_quit = TRUE;
              }
          }
        }
        if (app_quit) {
          appCtx->quit = TRUE;
          NVGSTDS_INFO_MSG_V ("Received EOS. Exiting ...\n");
          return FALSE;
        } else {
          NVGSTDS_INFO_MSG_V ("Received EOS ...\n");
        }
      } else {
        NVGSTDS_INFO_MSG_V ("Received EOS. Exiting ...\n");
        appCtx->quit = TRUE;
        return FALSE;
      }
      break;
    }
    case GST_MESSAGE_ELEMENT:{
      if (gst_nvmessage_is_force_pipeline_eos (message)) {
        gboolean app_quit = FALSE;
        if (gst_nvmessage_parse_force_pipeline_eos (message, &app_quit)) {
          if (app_quit)
            appCtx->quit = TRUE;
        }
      }
      if(gst_nvmessage_is_stream_add(message)) {
        g_mutex_lock (&(appCtx->perf_struct).struct_lock);

        appCtx->config.num_source_sub_bins++;
        NvDsSensorInfo sensorInfo = {0};
        gst_nvmessage_parse_stream_add(message, &sensorInfo);
        g_print("new stream added [%d:%s:%s]\n\n\n\n", sensorInfo.source_id, sensorInfo.sensor_id, sensorInfo.sensor_name);
        /** Callback */
        s_sensor_info_callback_stream_added(appCtx, &sensorInfo);
        gboolean is_rtsp = g_str_has_prefix (sensorInfo.uri, "rtsp://");
        gboolean is_ipc = g_str_has_prefix (sensorInfo.uri, "ipc://");
        appCtx->config.multi_source_config[sensorInfo.source_id].uri = g_strdup(sensorInfo.uri);
        if (is_rtsp) {
          appCtx->config.multi_source_config[sensorInfo.source_id].type = NV_DS_SOURCE_RTSP;
        }
        else if (is_ipc) {
          appCtx->config.multi_source_config[sensorInfo.source_id].type = NV_DS_SOURCE_IPC;
        }
        else {
          appCtx->config.multi_source_config[sensorInfo.source_id].type = NV_DS_SOURCE_URI;
        }
        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->
                    pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                "ds-app-added");
        NvDsFPSSensorInfo fpssensorInfo = {0};
        gst_nvmessage_parse_fps_stream_add(message, &fpssensorInfo);
        s_fps_sensor_info_callback_stream_added(appCtx, &fpssensorInfo);

        g_mutex_unlock (&(appCtx->perf_struct).struct_lock);
      }
      if(gst_nvmessage_is_stream_remove(message)) {
        g_mutex_lock (&(appCtx->perf_struct).struct_lock);
        appCtx->config.num_source_sub_bins--;
        NvDsSensorInfo sensorInfo = {0};
        gst_nvmessage_parse_stream_remove(message, &sensorInfo);
        g_print("new stream removed [%d:%s]\n\n\n\n", sensorInfo.source_id, sensorInfo.sensor_id);
        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->
                    pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                "ds-app-removed");
        /** Callback */
        s_sensor_info_callback_stream_removed(appCtx, &sensorInfo);
        NvDsFPSSensorInfo fpssensorInfo = {0};
        gst_nvmessage_parse_fps_stream_remove(message, &fpssensorInfo);
        s_fps_sensor_info_callback_stream_removed(appCtx, &fpssensorInfo);
        g_mutex_unlock (&(appCtx->perf_struct).struct_lock);
      }
      if (gst_nvmessage_is_reconnect_attempt_exceeded (message)) {
        NvDsRtspAttemptsInfo rtsp_info = {0};
        gboolean rec_attempt_exceeded_for_all = TRUE;
        if (gst_nvmessage_parse_reconnect_attempt_exceeded (message, &rtsp_info)) {
          if (rtsp_info.attempt_exceeded) {
            appCtx->config.multi_source_config[rtsp_info.source_id].rtsp_reconnect_attempt_exceeded =
                       rtsp_info.attempt_exceeded;
            NVGSTDS_INFO_MSG_V ("rtsp reconnect attempt exceeded for source_id : %d\n",rtsp_info.source_id);
          }

          if (appCtx->eos_received) {
            for (int i = 0; i < MAX_SOURCE_BINS; i++) {
              if (appCtx->config.multi_source_config[i].type == NV_DS_SOURCE_RTSP) {
                  if (appCtx->config.multi_source_config[i].rtsp_reconnect_attempt_exceeded != TRUE) {
                    rec_attempt_exceeded_for_all = FALSE;
                  }
              }
            }

            if (rec_attempt_exceeded_for_all) {
              NVGSTDS_INFO_MSG_V ("Exiting ...\n");
              appCtx->quit = TRUE;
              return FALSE;
            }
          }
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

/**
 * Function to dump bounding box data in kitti format. For this to work,
 * property "gie-kitti-output-dir" must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_kitti_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  gchar bbox_file[1024] = { 0 };
  FILE *bbox_params_dump_file = NULL;

  if (!appCtx->config.bbox_dir_path)
    return;

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    guint stream_id = frame_meta->pad_index;
    g_snprintf (bbox_file, sizeof (bbox_file) - 1,
        "%s/%02u_%03u_%06lu.txt", appCtx->config.bbox_dir_path,
        appCtx->index, stream_id, (gulong) frame_meta->frame_num);
    bbox_params_dump_file = fopen (bbox_file, "w");
    if (!bbox_params_dump_file)
      continue;

    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      float left = obj->rect_params.left;
      float top = obj->rect_params.top;
      float right = left + obj->rect_params.width;
      float bottom = top + obj->rect_params.height;
      // Here confidence stores detection confidence, since dump gie output
      // is before tracker plugin
      float confidence = obj->confidence;
      fprintf (bbox_params_dump_file,
          "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
          obj->obj_label, left, top, right, bottom, confidence);
    }
    fclose (bbox_params_dump_file);
  }
}

/**
 * Function to dump past frame objs in kitti format.
 */
static void
write_kitti_past_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  if (!appCtx->config.kitti_track_dir_path)
    return;

  // dump past frame tracked objects appending current frame objects
  gchar bbox_file[1024] = { 0 };
  FILE *bbox_params_dump_file = NULL;

  NvDsTargetMiscDataBatch *pPastFrameObjBatch = NULL;
  NvDsUserMetaList *bmeta_list = NULL;
  NvDsUserMeta *user_meta = NULL;
  for (bmeta_list = batch_meta->batch_user_meta_list; bmeta_list != NULL;
      bmeta_list = bmeta_list->next) {
    user_meta = (NvDsUserMeta *) bmeta_list->data;
    if (user_meta
        && user_meta->base_meta.meta_type == NVDS_TRACKER_PAST_FRAME_META) {
      pPastFrameObjBatch =
          (NvDsTargetMiscDataBatch *) (user_meta->user_meta_data);
      for (uint si = 0; si < pPastFrameObjBatch->numFilled; si++) {
        NvDsTargetMiscDataStream *objStream = (pPastFrameObjBatch->list) + si;
        guint stream_id = (guint) (objStream->streamID);
        for (uint li = 0; li < objStream->numFilled; li++) {
          NvDsTargetMiscDataObject *objList = (objStream->list) + li;
          for (uint oi = 0; oi < objList->numObj; oi++) {
            NvDsTargetMiscDataFrame *obj = (objList->list) + oi;
            g_snprintf (bbox_file, sizeof (bbox_file) - 1,
                "%s/%02u_%03u_%06lu.txt", appCtx->config.kitti_track_dir_path,
                appCtx->index, stream_id, (gulong) obj->frameNum);

            float left = obj->tBbox.left;
            float right = left + obj->tBbox.width;
            float top = obj->tBbox.top;
            float bottom = top + obj->tBbox.height;
            // Past frame object confidence given by tracker
            float confidence = obj->confidence;
            float x_world_foot = obj->ptWorldFeet[0];
            float y_world_foot = obj->ptWorldFeet[1];
            bbox_params_dump_file = fopen (bbox_file, "a");
            if (!bbox_params_dump_file) {
              continue;
            }
            fprintf (bbox_params_dump_file,
                "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 %f %f 0.0 0.0 %f\n",
                objList->objLabel, objList->uniqueId, left, top, right, bottom,
                x_world_foot, y_world_foot, confidence);
            fclose (bbox_params_dump_file);
          }
        }
      }
    }
  }
}

/**
 * Function to dump bounding box data in kitti format with tracking ID added.
 * For this to work, property "kitti-track-output-dir" must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_kitti_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  gchar bbox_file[1024] = { 0 };
  FILE *bbox_params_dump_file = NULL;

  if (!appCtx->config.kitti_track_dir_path)
    return;

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    guint stream_id = frame_meta->pad_index;
    g_snprintf (bbox_file, sizeof (bbox_file) - 1,
        "%s/%02u_%03u_%06lu.txt", appCtx->config.kitti_track_dir_path,
        appCtx->index, stream_id, (gulong) frame_meta->frame_num);
    bbox_params_dump_file = fopen (bbox_file, "w");
    if (!bbox_params_dump_file)
      continue;

    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      float left = obj->tracker_bbox_info.org_bbox_coords.left;
      float top = obj->tracker_bbox_info.org_bbox_coords.top;
      float right = left + obj->tracker_bbox_info.org_bbox_coords.width;
      float bottom = top + obj->tracker_bbox_info.org_bbox_coords.height;
      // Here confidence stores tracker confidence value for tracker output
      float confidence = obj->tracker_confidence;
      guint64 id = obj->object_id;
      bool write_proj_info = false;
      float visibility = -1.0;
      float x_img_foot = -1.0, y_img_foot = -1.0;
      float x_world_foot = -1.0, y_world_foot = -1.0;
      bool have_convex_hull = false;
      NvDsObjConvexHull *convex_hull_meta = NULL;
      // Attach projected object info if stored in user meta
      for (NvDsUserMetaList * l_obj_user = obj->obj_user_meta_list; l_obj_user != NULL;
          l_obj_user = l_obj_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_obj_user->data;
        if (user_meta && user_meta->base_meta.meta_type == NVDS_OBJ_VISIBILITY
            && user_meta->user_meta_data) {
          write_proj_info = true;
          visibility = *((float *) (user_meta->user_meta_data));
        } else if (user_meta && user_meta->base_meta.meta_type == NVDS_OBJ_IMAGE_FOOT_LOCATION
            && user_meta->user_meta_data) {
          write_proj_info = true;
          x_img_foot = ((float *) (user_meta->user_meta_data))[0];
          y_img_foot = ((float *) (user_meta->user_meta_data))[1];
        } else if (user_meta && user_meta->base_meta.meta_type == NVDS_OBJ_WORLD_FOOT_LOCATION
            && user_meta->user_meta_data) {
          write_proj_info = true;
          x_world_foot = ((float *) (user_meta->user_meta_data))[0];
          y_world_foot = ((float *) (user_meta->user_meta_data))[1];
        }
        else if (user_meta && user_meta->base_meta.meta_type == NVDS_OBJ_IMAGE_CONVEX_HULL
            && user_meta->user_meta_data) {
          have_convex_hull = true;
          convex_hull_meta = (NvDsObjConvexHull *)(user_meta->user_meta_data);
        }
      }

      // Build convex hull suffix in the same format as trajectory manager
      char hull_suffix[4096] = {' '};
      if (have_convex_hull && convex_hull_meta && convex_hull_meta->numPoints > 0 && convex_hull_meta->list) {
        size_t offset = 1;
        for (uint32_t i = 0; i < 2*convex_hull_meta->numPoints; i+=2) {
          int n = snprintf(hull_suffix + offset, sizeof(hull_suffix) - offset, "%d|%d|", convex_hull_meta->list[i], convex_hull_meta->list[i + 1]);
          if (n < 0 || (size_t)n >= sizeof(hull_suffix) - offset) break;
          offset += (size_t)n;
        }
        hull_suffix[offset - 1] = '\0';
      }

      /* KITTI format:
            label
            id
            ~truncated
            ~occluded
            ~alpha
            left
            top
            right
            bottom
            ~height
            ~width
            ~length
            x_feet
            y_feet
            ~z
            ~rotation_y
            confidence
            [visibility]
            [x_img_foot]
            [y_img_foot]
            [convex_hull]
      */
      if (write_proj_info)
      {
        fprintf (bbox_params_dump_file,
          "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 %f %f 0.0 0.0 %f %f %f %f",
          obj->obj_label, id, left, top, right, bottom, x_world_foot, y_world_foot, confidence, visibility, x_img_foot, y_img_foot);
        if (have_convex_hull) fputs(hull_suffix, bbox_params_dump_file);
        fputc('\n', bbox_params_dump_file);
      }
      else
      {
        fprintf (bbox_params_dump_file,
          "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
          obj->obj_label, id, left, top, right, bottom, confidence);
      }
    }
    fclose (bbox_params_dump_file);
  }
}

/**
 * Function to dump object ReID embeddings to files when the tracker outputs
 * ReID embeddings into user meta. For this to work, property "reid-track-output-dir"
 * must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_reid_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  if (!appCtx->config.reid_track_dir_path)
    return;

  gchar reid_file[1024] = { 0 };
  FILE *reid_params_dump_file = NULL;

  /** Save the reid embedding for each frame. */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /** Create dump file name. */
    guint stream_id = frame_meta->pad_index;
    g_snprintf (reid_file, sizeof (reid_file) - 1,
        "%s/%02u_%03u_%06lu.txt", appCtx->config.reid_track_dir_path,
        appCtx->index, stream_id, (gulong) frame_meta->frame_num);
    reid_params_dump_file = fopen (reid_file, "w");
    if (!reid_params_dump_file)
      continue;

    /** Save the reid embedding for each object. */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      guint64 id = obj->object_id;

      for (NvDsUserMetaList * l_obj_user = obj->obj_user_meta_list; l_obj_user != NULL;
          l_obj_user = l_obj_user->next) {

        /** Find the object's reid embedding index in user meta. */
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_obj_user->data;
        if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_OBJ_REID_META
            && user_meta->user_meta_data) {

          NvDsObjReid *pReidObj = (NvDsObjReid *) (user_meta->user_meta_data);
          if (pReidObj != NULL && pReidObj->ptr_host != NULL && pReidObj->featureSize > 0) {
            fprintf (reid_params_dump_file, "%lu", id);
            for (guint ele_i = 0; ele_i < pReidObj->featureSize; ele_i++) {
              fprintf (reid_params_dump_file, " %f", pReidObj->ptr_host[ele_i]);
            }
            fprintf (reid_params_dump_file, "\n");
          }
        }
      }
    }
    fclose (reid_params_dump_file);
  }
}


/**
 * Function to dump terminated object information to files when the tracker outputs
 * terminated track info into user meta. For this to work, property "terminated_track_output_path"
 * must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_terminated_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  if (!appCtx->config.terminated_track_output_path)
    return;

  gchar term_file[1024] = { 0 };
  FILE *term_params_dump_file = NULL;
  /** Find batch terminted tensor in batch user meta. */
  GList *pTerminatedTrackList = NULL;  // list of pointers to NvDsTargetMiscDataBatch
  NvDsTargetMiscDataBatch *pTerminatedTrackBatch = NULL;
  for (NvDsUserMetaList *l_batch_user = batch_meta->batch_user_meta_list;
       l_batch_user != NULL;
       l_batch_user = l_batch_user->next
      )
  {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) l_batch_user->data;
    if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_TERMINATED_LIST_META)
    {
      pTerminatedTrackBatch = (NvDsTargetMiscDataBatch *) (user_meta->user_meta_data);
      pTerminatedTrackList = g_list_append (pTerminatedTrackList, pTerminatedTrackBatch);
    }
  }

  if (!pTerminatedTrackList)
    return;

  /** Save the Terminated data for each frame. */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    for (GList *l = pTerminatedTrackList; l != NULL; l = l->next)
    {
      pTerminatedTrackBatch = (NvDsTargetMiscDataBatch *) (l->data);

      for (uint si = 0; si < pTerminatedTrackBatch->numFilled; si++)
      {
        NvDsTargetMiscDataStream *objStream = (pTerminatedTrackBatch->list) + si;
        guint stream_id = (guint)(objStream->streamID);

        if (frame_meta->pad_index != stream_id)
          continue;

        g_snprintf(term_file, sizeof(term_file) - 1,
                   "%s/%02u_%03u_%06lu.txt", appCtx->config.terminated_track_output_path,
                   appCtx->index, stream_id, (gulong)frame_meta->frame_num);

        term_params_dump_file = fopen(term_file, "w");

        if (!term_params_dump_file)
          continue;

        for (uint li = 0; li < objStream->numFilled; li++)
        {

          NvDsTargetMiscDataObject *objList = (objStream->list) + li;
          fprintf(term_params_dump_file,
                  "Target: %ld,%d,%hu\n",
                  objList->uniqueId,
                  objList->classId,
                  stream_id);

          for (uint oi = 0; oi < objList->numObj; oi++)
          {
            NvDsTargetMiscDataFrame *obj = (objList->list) + oi;

            float left = obj->tBbox.left;
            float right = left + obj->tBbox.width;
            float top = obj->tBbox.top;
            float bottom = top + obj->tBbox.height;

            fprintf(term_params_dump_file,
                    "%u %lu %u 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f %d %f\n",
                    obj->frameNum, objList->uniqueId, objList->classId, left, top, right, bottom,
                    obj->confidence, obj->trackerState, obj->visibility);
          }
        }
        fprintf(term_params_dump_file, "\n");

        fclose(term_params_dump_file);
      }
    }
  }
  g_list_free(pTerminatedTrackList);
}

/**
 * Function to dump terminated object information to files when the tracker outputs
 * terminated track info into user meta. For this to work, property "terminated_track_output_path"
 * must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_shadow_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  if (!appCtx->config.shadow_track_output_path)
    return;


  gchar term_file[1024] = { 0 };
  FILE *shadow_dump_file = NULL;
  /** Find shadow tracked tensor in batch user meta. */
  GList *pShadowTrackList = NULL;  // list of pointers to NvDsTargetMiscDataBatch
  NvDsTargetMiscDataBatch *pShadowTrackBatch = NULL;
  for (NvDsUserMetaList *l_batch_user = batch_meta->batch_user_meta_list;
       l_batch_user != NULL;
       l_batch_user = l_batch_user->next
      )
  {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) l_batch_user->data;
    if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_SHADOW_LIST_META)
    {
      // std::cout << "Found Shadow Data" << std::endl;
      pShadowTrackBatch = (NvDsTargetMiscDataBatch *) (user_meta->user_meta_data);
      pShadowTrackList = g_list_append (pShadowTrackList, pShadowTrackBatch);
    }
  }

  if (!pShadowTrackList)
    return;

  /** Save the Terminated data for each frame. */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    for (GList *l = pShadowTrackList; l != NULL; l = l->next)
    {
      pShadowTrackBatch = (NvDsTargetMiscDataBatch *) (l->data);
      for (uint si = 0; si < pShadowTrackBatch->numFilled; si++)
      {
        NvDsTargetMiscDataStream *objStream = (pShadowTrackBatch->list) + si;
        guint stream_id = (guint)(objStream->streamID);

        if (frame_meta->pad_index != stream_id)
          continue;

        g_snprintf(term_file, sizeof(term_file) - 1,
                   "%s/%02u_%03u_%06lu.txt", appCtx->config.shadow_track_output_path,
                   appCtx->index, stream_id, (gulong)frame_meta->frame_num);

        shadow_dump_file = fopen(term_file, "w");

        if (!shadow_dump_file)
          continue;

        for (uint li = 0; li < objStream->numFilled; li++)
        {
          NvDsTargetMiscDataObject *objList = (objStream->list) + li;

          if (objList->numObj > 0)
          {
            NvDsTargetMiscDataFrame *obj = (objList->list); // get first element only

            float left = obj->tBbox.left;
            float right = left + obj->tBbox.width;
            float top = obj->tBbox.top;
            float bottom = top + obj->tBbox.height;

            fprintf(shadow_dump_file,
                    "%u %lu %u 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f %d %f\n",
                    obj->frameNum, objList->uniqueId, objList->classId, left, top, right, bottom,
                    obj->confidence, obj->trackerState, obj->visibility);
          }
        }

        fclose(shadow_dump_file);
      }
    }
  }
  g_list_free(pShadowTrackList);
}

static gint
component_id_compare_func (gconstpointer a, gconstpointer b)
{
  NvDsClassifierMeta *cmetaa = (NvDsClassifierMeta *) a;
  NvDsClassifierMeta *cmetab = (NvDsClassifierMeta *) b;

  if (cmetaa->unique_component_id < cmetab->unique_component_id)
    return -1;
  if (cmetaa->unique_component_id > cmetab->unique_component_id)
    return 1;
  return 0;
}

/**
 * Function to process the attached metadata. This is just for demonstration
 * and can be removed if not required.
 * Here it demonstrates to use bounding boxes of different color and size for
 * different type / class of objects.
 * It also demonstrates how to join the different labels(PGIE + SGIEs)
 * of an object to form a single string.
 */
static void
process_meta (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  // For single source always display text either with demuxer or with tiler
  if (!appCtx->config.tiled_display_config.enable ||
      appCtx->config.num_source_sub_bins == 1) {
    appCtx->show_bbox_text = 1;
  }

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      gint class_index = obj->class_id;
      NvDsGieConfig *gie_config = NULL;
      gchar *str_ins_pos = NULL;

      if (obj->unique_component_id ==
          (gint) appCtx->config.primary_gie_config.unique_id) {
        gie_config = &appCtx->config.primary_gie_config;
      } else {
        for (gint i = 0; i < (gint) appCtx->config.num_secondary_gie_sub_bins;
            i++) {
          gie_config = &appCtx->config.secondary_gie_sub_bin_config[i];
          if (obj->unique_component_id == (gint) gie_config->unique_id) {
            break;
          }
          gie_config = NULL;
        }
      }
      g_free (obj->text_params.display_text);
      obj->text_params.display_text = NULL;

      if (gie_config != NULL) {
        if (g_hash_table_contains (gie_config->bbox_border_color_table,
                class_index + (gchar *) NULL)) {
          obj->rect_params.border_color = *((NvOSD_ColorParams *)
              g_hash_table_lookup (gie_config->bbox_border_color_table,
                  class_index + (gchar *) NULL));
        } else {
          obj->rect_params.border_color = gie_config->bbox_border_color;
        }
        obj->rect_params.border_width = appCtx->config.osd_config.border_width;

        if (g_hash_table_contains (gie_config->bbox_bg_color_table,
                class_index + (gchar *) NULL)) {
          obj->rect_params.has_bg_color = 1;
          obj->rect_params.bg_color = *((NvOSD_ColorParams *)
              g_hash_table_lookup (gie_config->bbox_bg_color_table,
                  class_index + (gchar *) NULL));
        } else {
          obj->rect_params.has_bg_color = 0;
        }
      }

      if (!appCtx->show_bbox_text)
        continue;

      obj->text_params.x_offset = obj->rect_params.left;
      obj->text_params.y_offset = obj->rect_params.top - 30;
      obj->text_params.font_params.font_color =
          appCtx->config.osd_config.text_color;
      obj->text_params.font_params.font_size =
          appCtx->config.osd_config.text_size;
      obj->text_params.font_params.font_name = appCtx->config.osd_config.font;
      if (appCtx->config.osd_config.text_has_bg) {
        obj->text_params.set_bg_clr = 1;
        obj->text_params.text_bg_clr = appCtx->config.osd_config.text_bg_color;
      }

      obj->text_params.display_text = (char *) g_malloc (128);
      obj->text_params.display_text[0] = '\0';
      str_ins_pos = obj->text_params.display_text;

      if (obj->obj_label[0] != '\0')
        sprintf (str_ins_pos, "%s", obj->obj_label);
      str_ins_pos += strlen (str_ins_pos);

      if (obj->object_id != UNTRACKED_OBJECT_ID) {
        /** object_id is a 64-bit sequential value;
         * but considering the display aesthetic,
         * trimming to lower 32-bits */
        if (appCtx->config.tracker_config.display_tracking_id) {
          guint64 const LOW_32_MASK = 0x00000000FFFFFFFF;
          sprintf (str_ins_pos, " %lu", (obj->object_id & LOW_32_MASK));
          str_ins_pos += strlen (str_ins_pos);
        }
      }

      obj->classifier_meta_list =
          g_list_sort (obj->classifier_meta_list, component_id_compare_func);
      for (NvDsMetaList * l_class = obj->classifier_meta_list; l_class != NULL;
          l_class = l_class->next) {
        NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *) l_class->data;
        for (NvDsMetaList * l_label = cmeta->label_info_list; l_label != NULL;
            l_label = l_label->next) {
          NvDsLabelInfo *label = (NvDsLabelInfo *) l_label->data;
          if (label->pResult_label) {
            sprintf (str_ins_pos, " %s", label->pResult_label);
          } else if (label->result_label[0] != '\0') {
            sprintf (str_ins_pos, " %s", label->result_label);
          }
          str_ins_pos += strlen (str_ins_pos);
        }

      }
    }
  }
}

/**
 * Function which processes the inferred buffer and its metadata.
 * It also gives opportunity to attach application specific
 * metadata (e.g. clock, analytics output etc.).
 */
static void
process_buffer (GstBuffer * buf, AppCtx * appCtx, guint index)
{
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    NVGSTDS_WARN_MSG_V ("Batch meta not found for buffer %p", buf);
    return;
  }
  process_meta (appCtx, batch_meta);
  //NvDsInstanceData *data = &appCtx->instance_data[index];
  //guint i;

  //  data->frame_num++;

  /* Opportunity to modify the processed metadata or do analytics based on
   * type of object e.g. maintaining count of particular type of car.
   */
  if (appCtx->all_bbox_generated_cb) {
    appCtx->all_bbox_generated_cb (appCtx, buf, batch_meta, index);
  }
  //data->bbox_list_size = 0;

  /*
   * callback to attach application specific additional metadata.
   */
  if (appCtx->overlay_graphics_cb) {
    appCtx->overlay_graphics_cb (appCtx, buf, batch_meta, index);
  }
}

/**
 * Buffer probe function to get the results of primary infer.
 * Here it demonstrates the use by dumping bounding box coordinates in
 * kitti format.
 */
static GstPadProbeReturn
gie_primary_processing_done_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  AppCtx *appCtx = (AppCtx *) u_data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    NVGSTDS_WARN_MSG_V ("Batch meta not found for buffer %p", buf);
    return GST_PAD_PROBE_OK;
  }

  write_kitti_output (appCtx, batch_meta);

  return GST_PAD_PROBE_OK;
}

/**
 * Probe function to get results after all inferences(Primary + Secondary)
 * are done. This will be just before OSD or sink (in case OSD is disabled).
 */
static GstPadProbeReturn
gie_processing_done_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsInstanceBin *bin = (NvDsInstanceBin *) u_data;
  guint index = bin->index;
  AppCtx *appCtx = bin->appCtx;

  if (gst_buffer_is_writable (buf))
    process_buffer (buf, appCtx, index);
  return GST_PAD_PROBE_OK;
}

/**
 * Buffer probe function after tracker.
 */
static GstPadProbeReturn
analytics_done_buf_prob (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsInstanceBin *bin = (NvDsInstanceBin *) u_data;
  guint index = bin->index;
  AppCtx *appCtx = bin->appCtx;
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    NVGSTDS_WARN_MSG_V ("Batch meta not found for buffer %p", buf);
    return GST_PAD_PROBE_OK;
  }

  /*
   * Output KITTI labels with tracking ID if configured to do so.
   */
  write_kitti_track_output (appCtx, batch_meta);
  write_kitti_past_track_output (appCtx, batch_meta);
  write_reid_track_output(appCtx, batch_meta);
  write_terminated_track_output(appCtx, batch_meta);
  write_shadow_track_output(appCtx, batch_meta);

  if (appCtx->bbox_generated_post_analytics_cb) {
    appCtx->bbox_generated_post_analytics_cb (appCtx, buf, batch_meta, index);
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
latency_measurement_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  AppCtx *appCtx = (AppCtx *) u_data;
  guint i = 0, num_sources_in_batch = 0;

  if (nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameLatencyInfo *latency_info = NULL;

    g_mutex_lock (&appCtx->latency_lock);
    latency_info = appCtx->latency_info;
    guint64 batch_num = GPOINTER_TO_SIZE(g_object_get_data(G_OBJECT(pad), "latency-batch-num"));

    if (nvds_enable_latency_measurement) {
      g_print("\n************BATCH-NUM = %lu**************\n", batch_num);
    }

    num_sources_in_batch = nvds_measure_buffer_latency (buf, latency_info);

    if (nvds_enable_latency_measurement) {
      for (i = 0; i < num_sources_in_batch; i++) {
        g_print ("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
            latency_info[i].source_id,
            latency_info[i].frame_num, latency_info[i].latency);
      }
    }

    g_mutex_unlock (&appCtx->latency_lock);
    g_object_set_data(G_OBJECT(pad), "latency-batch-num", GSIZE_TO_POINTER(batch_num + 1));
  }

  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
demux_latency_measurement_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  AppCtx *appCtx = (AppCtx *) u_data;
  guint i = 0, num_sources_in_batch = 0;

  if (nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameLatencyInfo *latency_info = NULL;

    g_mutex_lock (&appCtx->latency_lock);
    latency_info = appCtx->latency_info;

    if (nvds_enable_latency_measurement) {
      g_print ("\n************DEMUX BATCH-NUM = %d**************\n",
          demux_batch_num);
    }

    num_sources_in_batch = nvds_measure_buffer_latency (buf, latency_info);

    if (nvds_enable_latency_measurement) {
      for (i = 0; i < num_sources_in_batch; i++) {
        g_print ("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
            latency_info[i].source_id,
            latency_info[i].frame_num, latency_info[i].latency);
      }
    }

    g_mutex_unlock (&appCtx->latency_lock);
    demux_batch_num++;
  }

  return GST_PAD_PROBE_OK;
}

static gboolean
add_and_link_broker_sink (AppCtx * appCtx)
{
  NvDsConfig *config = &appCtx->config;
  /** Only first instance_bin broker sink
   * employed as there's only one analytics path for N sources
   * NOTE: There shall be only one [sink] group
   * with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
   * a) Multiple of them does not make sense as we have only
   * one analytics pipe generating the data for broker sink
   * b) If Multiple broker sinks are configured by the user
   * in config file, only the first in the order of
   * appearance will be considered
   * and others shall be ignored
   * c) Ideally it should be documented (or obvious) that:
   * multiple [sink] groups with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
   * is invalid
   */
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[0];
  NvDsPipeline *pipeline = &appCtx->pipeline;

  for (guint i = 0; i < config->num_sink_sub_bins; i++) {
    if (config->sink_bin_sub_bin_config[i].type == NV_DS_SINK_MSG_CONV_BROKER) {
      if (!pipeline->common_elements.tee) {
        NVGSTDS_ERR_MSG_V
            ("%s failed; broker added without analytics; check config file\n",
            __func__);
        return FALSE;
      }
      /** add the broker sink bin to pipeline */
      if (!gst_bin_add (GST_BIN (pipeline->pipeline),
              instance_bin->sink_bin.sub_bins[i].bin)) {
        return FALSE;
      }
      /** link the broker sink bin to the common_elements tee
       * (The tee after nvinfer -> tracker (optional) -> sgies (optional) block) */
      if (!link_element_to_tee_src_pad (pipeline->common_elements.tee,
              instance_bin->sink_bin.sub_bins[i].bin)) {
        return FALSE;
      }
    }
  }
  return TRUE;
}

static gboolean
create_demux_pipeline (AppCtx * appCtx, guint index)
{
  gboolean ret = FALSE;
  NvDsConfig *config = &appCtx->config;
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.demux_instance_bins[index];
  GstElement *last_elem;
  gchar elem_name[32];

  instance_bin->index = index;
  instance_bin->appCtx = appCtx;

  g_snprintf (elem_name, 32, "processing_demux_bin_%d", index);
  instance_bin->bin = gst_bin_new (elem_name);

  if (!create_demux_sink_bin (config->num_sink_sub_bins,
          config->sink_bin_sub_bin_config, &instance_bin->demux_sink_bin,
          config->sink_bin_sub_bin_config[index].source_id)) {
    goto done;
  }

  gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->demux_sink_bin.bin);
  last_elem = instance_bin->demux_sink_bin.bin;

  if (config->osd_config.enable) {
    if (!create_osd_bin (&config->osd_config, &instance_bin->osd_bin)) {
      goto done;
    }

    gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->osd_bin.bin);

    NVGSTDS_LINK_ELEMENT (instance_bin->osd_bin.bin, last_elem);

    last_elem = instance_bin->osd_bin.bin;
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (instance_bin->bin, last_elem, "sink");
  if (config->osd_config.enable) {
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->osd_bin.nvosd, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
  } else {
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->demux_sink_bin.bin, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
  }

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * Function to add components to pipeline which are dependent on number
 * of streams. These components work on single buffer. If tiling is being
 * used then single instance will be created otherwise < N > such instances
 * will be created for < N > streams
 */
static gboolean
create_processing_instance (AppCtx * appCtx, guint index)
{
  gboolean ret = FALSE;
  NvDsConfig *config = &appCtx->config;
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[index];
  GstElement *last_elem;
  gchar elem_name[32];

  instance_bin->index = index;
  instance_bin->appCtx = appCtx;

  g_snprintf (elem_name, 32, "processing_bin_%d", index);
  instance_bin->bin = gst_bin_new (elem_name);

  if (!create_sink_bin (config->num_sink_sub_bins,
          config->sink_bin_sub_bin_config, &instance_bin->sink_bin, index)) {
    goto done;
  }

  gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->sink_bin.bin);
  last_elem = instance_bin->sink_bin.bin;

  if (config->osd_config.enable) {
    if (!create_osd_bin (&config->osd_config, &instance_bin->osd_bin)) {
      goto done;
    }

    gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->osd_bin.bin);

    NVGSTDS_LINK_ELEMENT (instance_bin->osd_bin.bin, last_elem);

    last_elem = instance_bin->osd_bin.bin;
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (instance_bin->bin, last_elem, "sink");
  if (config->osd_config.enable) {
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->osd_bin.nvosd, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
  } else {
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->sink_bin.bin, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
  }

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * Function to create common elements(Primary infer, tracker, secondary infer)
 * of the pipeline. These components operate on muxed data from all the
 * streams. So they are independent of number of streams in the pipeline.
 */
static gboolean
create_common_elements (NvDsConfig * config, NvDsPipeline * pipeline,
    GstElement ** sink_elem, GstElement ** src_elem,
    bbox_generated_callback bbox_generated_post_analytics_cb)
{
  gboolean ret = FALSE;
  *sink_elem = *src_elem = NULL;

  if (config->segvisual_config.enable) {
    if (!create_segvisual_bin (&config->segvisual_config,
                              &pipeline->common_elements.segvisual_bin)) {
      goto done;
    }

    gst_bin_add (GST_BIN (pipeline->pipeline),
                         pipeline->common_elements.segvisual_bin.bin);

    if (!*src_elem) {
        *src_elem = pipeline->common_elements.segvisual_bin.bin;
    }
    if (*sink_elem) {
        NVGSTDS_LINK_ELEMENT (pipeline->common_elements.segvisual_bin.bin,
                             *sink_elem);
    }
    *sink_elem = pipeline->common_elements.segvisual_bin.bin;
  }

  if (config->primary_gie_config.enable) {
    if (config->num_secondary_gie_sub_bins > 0) {
      /** if using nvmultiurisrcbin, override batch-size config for sgie */
      if (config->use_nvmultiurisrcbin) {
        for (guint i = 0; i < config->num_secondary_gie_sub_bins; i++) {
          config->secondary_gie_sub_bin_config[i].batch_size =
              config->sgie_batch_size;
        }
      }
      if (!create_secondary_gie_bin (config->num_secondary_gie_sub_bins,
              config->primary_gie_config.unique_id,
              config->secondary_gie_sub_bin_config,
              &pipeline->common_elements.secondary_gie_bin)) {
        goto done;
      }
      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->common_elements.secondary_gie_bin.bin);
      if (!*src_elem) {
        *src_elem = pipeline->common_elements.secondary_gie_bin.bin;
      }
      if (*sink_elem) {
        NVGSTDS_LINK_ELEMENT (pipeline->common_elements.secondary_gie_bin.bin,
            *sink_elem);
      }
      *sink_elem = pipeline->common_elements.secondary_gie_bin.bin;
    }
  }

  if (config->primary_gie_config.enable) {
    if (config->num_secondary_preprocess_sub_bins > 0) {
      if (!create_secondary_preprocess_bin (config->
              num_secondary_preprocess_sub_bins,
              config->primary_gie_config.unique_id,
              config->secondary_preprocess_sub_bin_config,
              &pipeline->common_elements.secondary_preprocess_bin)) {
        g_print ("creating secondary_preprocess bin failed\n");
        goto done;
      }
      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->common_elements.secondary_preprocess_bin.bin);

      if (!*src_elem) {
        *src_elem = pipeline->common_elements.secondary_preprocess_bin.bin;
      }
      if (*sink_elem) {
        NVGSTDS_LINK_ELEMENT (pipeline->common_elements.
            secondary_preprocess_bin.bin, *sink_elem);
      }

      *sink_elem = pipeline->common_elements.secondary_preprocess_bin.bin;
    }
  }

  if (config->dsanalytics_config.enable) {
    if (!create_dsanalytics_bin (&config->dsanalytics_config,
            &pipeline->common_elements.dsanalytics_bin)) {
      g_print ("creating dsanalytics bin failed\n");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.dsanalytics_bin.bin);

    if (!*src_elem) {
      *src_elem = pipeline->common_elements.dsanalytics_bin.bin;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.dsanalytics_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.dsanalytics_bin.bin;
  }

  if (config->text_embedder_config.enable) {
    pipeline->common_elements.text_embedder =
        gst_element_factory_make (NVDS_ELEM_TEXT_EMBEDDER, "text_embedder");
    if (!pipeline->common_elements.text_embedder) {
      NVGSTDS_ERR_MSG_V ("Failed to create 'text_embedder'");
      goto done;
    }

    if (config->text_embedder_config.model_name) {
      g_object_set (G_OBJECT (pipeline->common_elements.text_embedder),
          "model-name", config->text_embedder_config.model_name, NULL);
      g_print ("text_embedder: model-name set to '%s'\n",
          config->text_embedder_config.model_name);
    }
    if (config->text_embedder_config.onnx_model_path) {
      g_object_set (G_OBJECT (pipeline->common_elements.text_embedder),
          "onnx-model-path", config->text_embedder_config.onnx_model_path, NULL);
      g_print ("text_embedder: onnx-model-path set to '%s'\n",
          config->text_embedder_config.onnx_model_path);
    }
    if (config->text_embedder_config.tokenizer_dir) {
      g_object_set (G_OBJECT (pipeline->common_elements.text_embedder),
          "tokenizer-dir", config->text_embedder_config.tokenizer_dir, NULL);
      g_print ("text_embedder: tokenizer-dir set to '%s'\n",
          config->text_embedder_config.tokenizer_dir);
    }

    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.text_embedder);
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.text_embedder;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.text_embedder,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.text_embedder;
  }

  if (config->tracker_config.enable) {
    if (!create_tracking_bin (&config->tracker_config,
            &pipeline->common_elements.tracker_bin)) {
      g_print ("creating tracker bin failed\n");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.tracker_bin.bin);
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.tracker_bin.bin;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.tracker_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.tracker_bin.bin;
  }

  /* Vision Encoder - placed after tracker (SGIE-style separate bin with internal
   * drop of empty batches so nvvidconv_visionencoder_in is never called for removed-source buffers) */
  if (config->visionencoder_config.enable) {
    if (!create_visionencoder_bin (&config->visionencoder_config,
            &pipeline->common_elements.visionencoder_bin)) {
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.visionencoder_bin.bin);
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.visionencoder_bin.bin;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.visionencoder_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.visionencoder_bin.bin;

    GST_CAT_INFO (NVDS_APP, "✓ Vision encoder enabled: model=%s, batch-size=%u, device=%s (NV12→RGB→NV12, all NVMM)",
                  config->visionencoder_config.model_variant,
                  config->visionencoder_config.batch_size,
                  config->visionencoder_config.device);
  }

  if (config->replay_config.enable) {
    if (!create_replay_bin (&config->replay_config,
            &pipeline->common_elements.replay_bin)) {
      g_print ("creating replay bin failed\n");
      goto done;
    }
    else
    {
      g_print ("creating replay bin succeed\n");
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.replay_bin.bin);
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.replay_bin.bin;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.replay_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.replay_bin.bin;
    NVGSTDS_ELEM_ADD_PROBE (pipeline->common_elements.
        primary_bbox_buffer_probe_id,
        pipeline->common_elements.replay_bin.bin, "src",
        gie_primary_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
        pipeline->common_elements.appCtx);
  }

  if (config->primary_gie_config.enable && (!config->replay_config.enable)) {
    if (config->use_nvmultiurisrcbin) {
      config->primary_gie_config.batch_size = config->max_batch_size;
    }
    if (!create_primary_gie_bin (&config->primary_gie_config,
            &pipeline->common_elements.primary_gie_bin)) {
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.primary_gie_bin.bin);
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.primary_gie_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.primary_gie_bin.bin;
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.primary_gie_bin.bin;
    }
    NVGSTDS_ELEM_ADD_PROBE (pipeline->
        common_elements.primary_bbox_buffer_probe_id,
        pipeline->common_elements.primary_gie_bin.bin, "src",
        gie_primary_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
        pipeline->common_elements.appCtx);
  }

  if (config->preprocess_config.enable) {
    if (!create_preprocess_bin (&config->preprocess_config,
            &pipeline->common_elements.preprocess_bin)) {
      g_print ("creating preprocess bin failed\n");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.preprocess_bin.bin);

    if (!*src_elem) {
      *src_elem = pipeline->common_elements.preprocess_bin.bin;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.preprocess_bin.bin,
          *sink_elem);
    }

    *sink_elem = pipeline->common_elements.preprocess_bin.bin;
  }

  if (*src_elem) {
    NVGSTDS_ELEM_ADD_PROBE (pipeline->
        common_elements.primary_bbox_buffer_probe_id, *src_elem, "src",
        analytics_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
        &pipeline->common_elements);

    /* Add common message converter */
    if (config->msg_conv_config.enable) {
      NvDsSinkMsgConvBrokerConfig *convConfig = &config->msg_conv_config;
      pipeline->common_elements.msg_conv =
          gst_element_factory_make (NVDS_ELEM_MSG_CONV, "common_msg_conv");
      if (!pipeline->common_elements.msg_conv) {
        NVGSTDS_ERR_MSG_V ("Failed to create element 'common_msg_conv'");
        goto done;
      }

      g_object_set (G_OBJECT (pipeline->common_elements.msg_conv),
          "config", convConfig->config_file_path,
          "msg2p-lib",
          (convConfig->conv_msg2p_lib ? convConfig->conv_msg2p_lib : "null"),
          "payload-type", convConfig->conv_payload_type, "comp-id",
          convConfig->conv_comp_id, "debug-payload-dir",
          convConfig->debug_payload_dir, "multiple-payloads",
          convConfig->multiple_payloads,"msg2p-newapi", convConfig->conv_msg2p_new_api,
          "frame-interval", convConfig->conv_frame_interval,
          "embedding-filter", convConfig->embedding_filter, NULL);

      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->common_elements.msg_conv);

      NVGSTDS_LINK_ELEMENT (*src_elem, pipeline->common_elements.msg_conv);
      *src_elem = pipeline->common_elements.msg_conv;
    }
    pipeline->common_elements.tee =
        gst_element_factory_make (NVDS_ELEM_TEE, "common_analytics_tee");
    if (!pipeline->common_elements.tee) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'common_analytics_tee'");
      goto done;
    }

    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->common_elements.tee);

    NVGSTDS_LINK_ELEMENT (*src_elem, pipeline->common_elements.tee);
    *src_elem = pipeline->common_elements.tee;
  }

  ret = TRUE;
done:
  return ret;
}

static gboolean
is_sink_available_for_source_id (NvDsConfig * config, guint source_id)
{
  for (guint j = 0; j < config->num_sink_sub_bins; j++) {
    if (config->sink_bin_sub_bin_config[j].enable &&
        config->sink_bin_sub_bin_config[j].source_id == source_id &&
        config->sink_bin_sub_bin_config[j].link_to_demux == FALSE) {
      return TRUE;
    }
  }
  return FALSE;
}

/**
 * Main function to create the pipeline.
 */
gboolean
create_pipeline (AppCtx * appCtx,
    bbox_generated_callback bbox_generated_post_analytics_cb,
    bbox_generated_callback all_bbox_generated_cb, perf_callback perf_cb,
    overlay_graphics_callback overlay_graphics_cb)
{
  gboolean ret = FALSE;
  NvDsPipeline *pipeline = &appCtx->pipeline;
  NvDsConfig *config = &appCtx->config;
  GstBus *bus;
  GstElement *last_elem;
  GstElement *tmp_elem1;
  GstElement *tmp_elem2;
  guint i;
  GstPad *fps_pad = NULL;
  gulong latency_probe_id;

  _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

  appCtx->all_bbox_generated_cb = all_bbox_generated_cb;
  appCtx->bbox_generated_post_analytics_cb = bbox_generated_post_analytics_cb;
  appCtx->overlay_graphics_cb = overlay_graphics_cb;
  appCtx->sensorInfoHash = g_hash_table_new (NULL, NULL);
  appCtx->perf_struct.FPSInfoHash = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, NULL);

  if (config->osd_config.num_out_buffers < 8) {
    config->osd_config.num_out_buffers = 8;
  }

  pipeline->pipeline = gst_pipeline_new ("pipeline");
  if (!pipeline->pipeline) {
    NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
    goto done;
  }

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline->pipeline));
  pipeline->bus_id = gst_bus_add_watch (bus, bus_callback, appCtx);
  gst_object_unref (bus);

  /** Initialize msgapi for error propagation if enabled */
  /** Skip if already initialized early (e.g., by msgapi_init_early in main) */
  if (config->enable_error_propagation && !appCtx->msgapi_conn_handle) {
    if (!msgapi_init(appCtx)) {
      g_print("[MSGAPI_WARN] Failed to initialize message API for error propagation\n");
      g_print("[MSGAPI_WARN] Continuing without error propagation...\n");
    }
  } else if (appCtx->msgapi_conn_handle) {
    g_print("[MSGAPI] Message API already initialized, skipping duplicate initialization\n");
  }

  if (config->file_loop) {
    /* Let each source bin know it needs to loop. */
    guint i;
    for (i = 0; i < config->num_source_sub_bins; i++)
      config->multi_source_config[i].loop = TRUE;
  }

  for (guint i = 0; i < config->num_sink_sub_bins; i++) {
    NvDsSinkSubBinConfig *sink_config = &config->sink_bin_sub_bin_config[i];
    switch (sink_config->type) {
      case NV_DS_SINK_FAKE:
#ifndef IS_TEGRA
      case NV_DS_SINK_RENDER_EGL:
#else
      case NV_DS_SINK_RENDER_3D:
#endif
      case NV_DS_SINK_RENDER_DRM:
        /* Set the "qos" property of sink, if not explicitly specified in the
           config. */
        if (!sink_config->render_config.qos_value_specified) {
          sink_config->render_config.qos = FALSE;
        }
      default:
        break;
    }
  }
  /*
   * Add muxer and < N > source components to the pipeline based
   * on the settings in configuration file.
   */
  if (config->use_nvmultiurisrcbin) {
    if (config->num_source_sub_bins >0){
      if (!create_nvmultiurisrcbin_bin (config->num_source_sub_bins,
              &config->source_attr_all_config, &pipeline->multi_src_bin))
        goto done;
    }
    else {
      if (!config->source_attr_all_parsed) {
        NVGSTDS_ERR_MSG_V("[source-attr-all] config group not set, needs to be configured");
        goto done;
      }
      if (!create_nvmultiurisrcbin_bin (config->num_source_sub_bins,
              &config->source_attr_all_config, &pipeline->multi_src_bin))
        goto done;
      //[source-list] added with num-source-bins=0; This means source-bin
      //will be created and be waiting for source adds over REST API
      //mark num-source-bins=1 as one source-bin is indeed created
      config->num_source_sub_bins = 1;
    }
    /** set properties for nvmultiurisrcbin */
    if (config->uri_list) {
      gchar *uri_list_comma_sep = g_strjoinv (",", config->uri_list);
      g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "uri-list",
          uri_list_comma_sep, NULL);
      g_free (uri_list_comma_sep);
    }
    if (config->sensor_id_list) {
      gchar *uri_list_comma_sep = g_strjoinv (",", config->sensor_id_list);
      g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "sensor-id-list",
          uri_list_comma_sep, NULL);
      g_free (uri_list_comma_sep);
    }
    if (config->sensor_name_list) {
      gchar *uri_list_comma_sep = g_strjoinv (",", config->sensor_name_list);
      g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "sensor-name-list",
          uri_list_comma_sep, NULL);
      g_free (uri_list_comma_sep);
    }
    g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "max-batch-size",
        config->max_batch_size, NULL);
    g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "ip-address",
        config->http_ip, NULL);
    g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "port",
        config->http_port, NULL);
    g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "extract-sei-type5-data-dec",
        config->extract_sei_type5_data, NULL);
    g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "low-latency-mode",
        config->low_latency_mode, NULL);
    g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "sei-uuid",
        config->sei_uuid, NULL);

    /** Set error propagation properties if enabled */
    if (config->enable_error_propagation) {
      g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "enable-error-propagation",
          config->enable_error_propagation, NULL);
      if (config->proto_lib) {
        g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "proto-lib",
            config->proto_lib, NULL);
      }
      if (config->conn_str) {
        g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "conn-str",
            config->conn_str, NULL);
      }
      if (config->topic) {
        g_object_set (pipeline->multi_src_bin.nvmultiurisrcbin, "topic",
            config->topic, NULL);
      }
    }
  } else {
    if (!create_multi_source_bin (config->num_source_sub_bins,
            config->multi_source_config, &pipeline->multi_src_bin))
      goto done;
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->multi_src_bin.bin);


  if (config->streammux_config.is_parsed) {
    if (config->use_nvmultiurisrcbin) {
      config->streammux_config.use_nvmultiurisrcbin = TRUE;
      /** overriding mux_config.batch_size to max_batch_size */
      config->streammux_config.batch_size = config->max_batch_size;
    }

    if (!set_streammux_properties (&config->streammux_config,
            pipeline->multi_src_bin.streammux)) {
      NVGSTDS_WARN_MSG_V ("Failed to set streammux properties");
    }
  }


  if (appCtx->latency_info == NULL) {
    appCtx->latency_info = (NvDsFrameLatencyInfo *)
        calloc (1, config->streammux_config.batch_size *
        sizeof (NvDsFrameLatencyInfo));
  }

  /** a tee after the tiler which shall be connected to sink(s) */
  pipeline->tiler_tee = gst_element_factory_make (NVDS_ELEM_TEE, "tiler_tee");
  if (!pipeline->tiler_tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'tiler_tee'");
    goto done;
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->tiler_tee);

  /** Tiler + Demux in Parallel Use-Case */
  if (config->tiled_display_config.enable ==
      NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX) {
    pipeline->demuxer =
        gst_element_factory_make (NVDS_ELEM_STREAM_DEMUX, "demuxer");
    if (!pipeline->demuxer) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'demuxer'");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->demuxer);

    /** NOTE:
     * demux output is supported for only one source
     * If multiple [sink] groups are configured with
     * link_to_demux=1, only the first [sink]
     * shall be constructed for all occurences of
     * [sink] groups with link_to_demux=1
     */
    {
      gchar pad_name[16];
      GstPad *demux_src_pad;

      i = 0;
      if (!create_demux_pipeline (appCtx, i)) {
        goto done;
      }

      for (i = 0; i < config->num_sink_sub_bins; i++) {
        if (config->sink_bin_sub_bin_config[i].link_to_demux == TRUE) {
          g_snprintf (pad_name, 16, "src_%02d",
              config->sink_bin_sub_bin_config[i].source_id);
          break;
        }
      }

      if (i >= config->num_sink_sub_bins) {
        g_print
            ("\n\nError : sink for demux (use link-to-demux-only property) is not provided in the config file\n\n");
        goto done;
      }

      i = 0;

      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->demux_instance_bins[i].bin);

      demux_src_pad = gst_element_request_pad_simple (pipeline->demuxer, pad_name);
      NVGSTDS_LINK_ELEMENT_FULL (pipeline->demuxer, pad_name,
          pipeline->demux_instance_bins[i].bin, "sink");
      gst_object_unref (demux_src_pad);

      NVGSTDS_ELEM_ADD_PROBE (latency_probe_id,
          appCtx->pipeline.demux_instance_bins[i].demux_sink_bin.bin,
          "sink",
          demux_latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
          appCtx);
      latency_probe_id = latency_probe_id;
    }

    last_elem = pipeline->demuxer;
    link_element_to_tee_src_pad (pipeline->tiler_tee, last_elem);
    last_elem = pipeline->tiler_tee;
  }

  if (config->tiled_display_config.enable) {

    /* Tiler will generate a single composited buffer for all sources. So need
     * to create only one processing instance. */
    if (!create_processing_instance (appCtx, 0)) {
      goto done;
    }
    // create and add tiling component to pipeline.
    if (config->tiled_display_config.columns *
        config->tiled_display_config.rows < config->num_source_sub_bins) {
      if (config->tiled_display_config.columns == 0) {
        config->tiled_display_config.columns =
            (guint) (sqrt (config->num_source_sub_bins) + 0.5);
      }
      config->tiled_display_config.rows =
          (guint) ceil (1.0 * config->num_source_sub_bins /
          config->tiled_display_config.columns);
      NVGSTDS_WARN_MSG_V
          ("Num of Tiles less than number of sources, readjusting to "
          "%u rows, %u columns", config->tiled_display_config.rows,
          config->tiled_display_config.columns);
    }

    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->instance_bins[0].bin);
    last_elem = pipeline->instance_bins[0].bin;

    if (!create_tiled_display_bin (&config->tiled_display_config,
            &pipeline->tiled_display_bin)) {
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->tiled_display_bin.bin);
    NVGSTDS_LINK_ELEMENT (pipeline->tiled_display_bin.bin, last_elem);
    last_elem = pipeline->tiled_display_bin.bin;

    link_element_to_tee_src_pad (pipeline->tiler_tee,
        pipeline->tiled_display_bin.bin);
    last_elem = pipeline->tiler_tee;

    NVGSTDS_ELEM_ADD_PROBE (latency_probe_id,
        pipeline->instance_bins->sink_bin.sub_bins[0].sink, "sink",
        latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, appCtx);
    latency_probe_id = latency_probe_id;
  } else {
    /*
     * Create demuxer only if tiled display is disabled.
     */
    pipeline->demuxer =
        gst_element_factory_make (NVDS_ELEM_STREAM_DEMUX, "demuxer");
    if (!pipeline->demuxer) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'demuxer'");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->demuxer);

    for (i = 0; i < config->num_source_sub_bins; i++) {
      gchar pad_name[16];
      GstPad *demux_src_pad;

      /* Check if any sink has been configured to render/encode output for
       * source index `i`. The processing instance for that source will be
       * created only if atleast one sink has been configured as such.
       */
      if (!is_sink_available_for_source_id (config, i))
        continue;

      if (!create_processing_instance (appCtx, i)) {
        goto done;
      }
      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->instance_bins[i].bin);

      g_snprintf (pad_name, 16, "src_%02d", i);
      demux_src_pad = gst_element_request_pad_simple (pipeline->demuxer, pad_name);
      NVGSTDS_LINK_ELEMENT_FULL (pipeline->demuxer, pad_name,
          pipeline->instance_bins[i].bin, "sink");
      gst_object_unref (demux_src_pad);

      for (int k = 0; k < MAX_SINK_BINS; k++) {
        if (pipeline->instance_bins[i].sink_bin.sub_bins[k].sink) {
          NVGSTDS_ELEM_ADD_PROBE (latency_probe_id,
              pipeline->instance_bins[i].sink_bin.sub_bins[k].sink, "sink",
              latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, appCtx);
          break;
        }
      }

      latency_probe_id = latency_probe_id;
    }
    last_elem = pipeline->demuxer;
  }

  if (config->tiled_display_config.enable == NV_DS_TILED_DISPLAY_DISABLE) {
    fps_pad = gst_element_get_static_pad (pipeline->demuxer, "sink");
  } else {
    fps_pad =
        gst_element_get_static_pad (pipeline->tiled_display_bin.bin, "sink");
  }

  pipeline->common_elements.appCtx = appCtx;
  // Decide where in the pipeline the element should be added and add only if
  // enabled
  if (config->dsexample_config.enable) {
    // Create dsexample element bin and set properties
    if (!create_dsexample_bin (&config->dsexample_config,
            &pipeline->dsexample_bin)) {
      goto done;
    }
    // Add dsexample bin to instance bin
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->dsexample_bin.bin);

    // Link this bin to the last element in the bin
    NVGSTDS_LINK_ELEMENT (pipeline->dsexample_bin.bin, last_elem);

    // Set this bin as the last element
    last_elem = pipeline->dsexample_bin.bin;
  }
  // create and add common components to pipeline.
  if (!create_common_elements (config, pipeline, &tmp_elem1, &tmp_elem2,
          bbox_generated_post_analytics_cb)) {
    goto done;
  }

  if (!add_and_link_broker_sink (appCtx)) {
    goto done;
  }

  if (tmp_elem2) {
    NVGSTDS_LINK_ELEMENT (tmp_elem2, last_elem);
    last_elem = tmp_elem1;
  }

  NVGSTDS_LINK_ELEMENT (pipeline->multi_src_bin.bin, last_elem);

  // enable performance measurement and add call back function to receive
  // performance data.
  if (config->enable_perf_measurement) {
    appCtx->perf_struct.context = appCtx;
    if (config->use_nvmultiurisrcbin) {
      appCtx->perf_struct.stream_name_display = config->stream_name_display;
      appCtx->perf_struct.use_nvmultiurisrcbin = config->use_nvmultiurisrcbin;
      enable_perf_measurement (&appCtx->perf_struct, fps_pad,
          config->max_batch_size,
          config->perf_measurement_interval_sec,
          config->multi_source_config[0].dewarper_config.num_surfaces_per_frame,
          perf_cb);
    } else {
      enable_perf_measurement (&appCtx->perf_struct, fps_pad,
          pipeline->multi_src_bin.num_bins,
          config->perf_measurement_interval_sec,
          config->multi_source_config[0].dewarper_config.num_surfaces_per_frame,
          perf_cb);
    }
  }

  latency_probe_id = latency_probe_id;

  if (config->num_message_consumers) {
    for (i = 0; i < config->num_message_consumers; i++) {
      appCtx->c2d_ctx[i] =
          start_cloud_to_device_messaging (&config->message_consumer_config[i],
          NULL, &appCtx->pipeline.multi_src_bin);
      if (appCtx->c2d_ctx[i] == NULL) {
        NVGSTDS_ERR_MSG_V ("Failed to create message consumer");
        goto done;
      }
    }
  }

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->pipeline.pipeline),
      GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-null");

  g_mutex_init (&appCtx->app_lock);
  g_cond_init (&appCtx->app_cond);
  g_mutex_init (&appCtx->latency_lock);

  ret = TRUE;
done:
  if (fps_pad)
    gst_object_unref (fps_pad);

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * Function to destroy pipeline and release the resources, probes etc.
 */
void
destroy_pipeline (AppCtx * appCtx)
{
  gint64 end_time;
  NvDsConfig *config = &appCtx->config;
  guint i;
  GstBus *bus = NULL;

  end_time = g_get_monotonic_time () + G_TIME_SPAN_SECOND;

  if (!appCtx)
    return;

  gst_element_send_event(appCtx->pipeline.pipeline, gst_event_new_eos());
  sleep (1);

  g_mutex_lock (&appCtx->app_lock);
  if (appCtx->pipeline.pipeline) {
    destroy_smart_record_bin (&appCtx->pipeline.multi_src_bin);
    bus = gst_pipeline_get_bus (GST_PIPELINE (appCtx->pipeline.pipeline));

    while (TRUE) {
      GstMessage *message = gst_bus_pop (bus);
      if (message == NULL || GST_MESSAGE_TYPE (message) == GST_MESSAGE_EOS)
        break;
      else if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR)
        bus_callback (bus, message, appCtx);
      else
        gst_message_unref (message);
    }
    gst_object_unref (bus);
    gst_element_set_state (appCtx->pipeline.pipeline, GST_STATE_NULL);
  }
  g_cond_wait_until (&appCtx->app_cond, &appCtx->app_lock, end_time);
  g_mutex_unlock (&appCtx->app_lock);

  /** Cleanup msgapi resources if error propagation was enabled */
  if (appCtx->msgapi_conn_handle) {
    msgapi_cleanup(appCtx);
  }

  for (i = 0; i < appCtx->config.num_source_sub_bins; i++) {
    NvDsInstanceBin *bin = &appCtx->pipeline.instance_bins[i];
    if (config->osd_config.enable) {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->all_bbox_buffer_probe_id,
          bin->osd_bin.nvosd, "sink");
    } else {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->all_bbox_buffer_probe_id,
          bin->sink_bin.bin, "sink");
    }

    if (config->primary_gie_config.enable && (!config->replay_config.enable)) {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->primary_bbox_buffer_probe_id,
          bin->primary_gie_bin.bin, "src");
    }

    if (config->replay_config.enable) {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->primary_bbox_buffer_probe_id,
          bin->replay_bin.bin, "src");
    }

  }
  if (appCtx->latency_info == NULL) {
    free (appCtx->latency_info);
    appCtx->latency_info = NULL;
  }
  if (appCtx->sensorInfoHash) {
    g_hash_table_destroy(appCtx->sensorInfoHash);
  }
  if (appCtx->perf_struct.FPSInfoHash) {
    g_hash_table_destroy(appCtx->perf_struct.FPSInfoHash);
  }

  destroy_sink_bin ();
  g_mutex_clear (&appCtx->latency_lock);

  if (appCtx->pipeline.pipeline) {
    bus = gst_pipeline_get_bus (GST_PIPELINE (appCtx->pipeline.pipeline));
    gst_bus_remove_watch (bus);
    gst_object_unref (bus);
    gst_object_unref (appCtx->pipeline.pipeline);
    appCtx->pipeline.pipeline = NULL;
    pause_perf_measurement (&appCtx->perf_struct);

    //for pipeline-recreate, reset rtsp srouce's depay, such as rtph264depay.
    NvDsSrcParentBin *pbin = &appCtx->pipeline.multi_src_bin;
    if(pbin){
        NvDsSrcBin *src_bin;
        for (i = 0; i < MAX_SOURCE_BINS; i++) {
          src_bin = &pbin->sub_bins[i];
          if (src_bin && src_bin->config
              && src_bin->config->type == NV_DS_SOURCE_RTSP){
              src_bin->depay = NULL;
          }
        }
    }
  }

  if (config->num_message_consumers) {
    for (i = 0; i < config->num_message_consumers; i++) {
      if (appCtx->c2d_ctx[i])
        stop_cloud_to_device_messaging (appCtx->c2d_ctx[i]);
    }
  }
}

gboolean
pause_pipeline (AppCtx * appCtx)
{
  GstState cur;
  GstState pending;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret =
      gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
      timeout);

  if (ret == GST_STATE_CHANGE_ASYNC) {
    return FALSE;
  }

  if (cur == GST_STATE_PAUSED) {
    return TRUE;
  } else if (cur == GST_STATE_PLAYING) {
    gst_element_set_state (appCtx->pipeline.pipeline, GST_STATE_PAUSED);
    gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
        GST_CLOCK_TIME_NONE);
    pause_perf_measurement (&appCtx->perf_struct);
    return TRUE;
  } else {
    return FALSE;
  }
}

gboolean
resume_pipeline (AppCtx * appCtx)
{
  GstState cur;
  GstState pending;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret =
      gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
      timeout);

  if (ret == GST_STATE_CHANGE_ASYNC) {
    return FALSE;
  }

  if (cur == GST_STATE_PLAYING) {
    return TRUE;
  } else if (cur == GST_STATE_PAUSED) {
    gst_element_set_state (appCtx->pipeline.pipeline, GST_STATE_PLAYING);
    gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
        GST_CLOCK_TIME_NONE);
    resume_perf_measurement (&appCtx->perf_struct);
    return TRUE;
  } else {
    return FALSE;
  }
}
