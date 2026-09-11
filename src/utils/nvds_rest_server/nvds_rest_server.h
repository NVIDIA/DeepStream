/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _NVDS_SERVER_H_
#define _NVDS_SERVER_H_

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <functional>
#include "gst-nvdscustomevent.h"
#include "gst-nvdscommonconfig.h"
#include <json/json.h>
#define UNKNOWN_STRING "unknown"
#define EMPTY_STRING ""

typedef enum
{
  DROP_FRAME_INTERVAL = 1 << 0,
  SKIP_FRAMES = 1 << 1,
  LOW_LATENCY_MODE = 1 << 2,
} NvDsServerDecPropFlag;

typedef enum
{
  BITRATE = 1 << 0,
  FORCE_IDR = 1 << 1,
  FORCE_INTRA = 1 << 2,
  IFRAME_INTERVAL = 1 << 3,
} NvDsServerEncPropFlag;

typedef enum
{
  SRC_CROP = 1 << 0,
  DEST_CROP = 1 << 1,
  FLIP_METHOD = 1 << 2,
  INTERPOLATION_METHOD = 1 << 3,
} NvDsServerConvPropFlag;

typedef enum
{
  BATCHED_PUSH_TIMEOUT = 1 << 0,
  MAX_LATENCY = 1 << 1,
} NvDsServerMuxPropFlag;

typedef enum
{
  INFER_INTERVAL = 1 << 0,
} NvDsServerInferPropFlag;

typedef enum
{
  INFERSERVER_INTERVAL = 1 << 0,
} NvDsServerInferServerPropFlag;

typedef enum
{
  NVTRACKER_CONFIG = 1 << 0,
} NvDsServerNvTrackerPropFlag;

/**
 * Flags for GET request types supported by the REST server
 *
 * GET_LIVE_STREAM_INFO - Query stream information via /api/v1/stream/get-stream-info
 * GET_READY_INFO       - Application readiness health check via /ready
 * GET_LIVE_INFO        - Application liveness health check via /live
 * GET_STARTUP_INFO     - Application startup state via /startup
 * GET_METRICS_INFO     - Query application metrics via /api/v1/metrics
 * GET_METADATA_INFO    - Query application metadata via /api/v1/metadata
 * GET_MODEL_STATUS_INFO - Query the MODEL plane via /api/v1/model/status
 * GET_STREAM_ROUTE_INFO - Query the ROUTING plane via GET /api/v1/stream/route.
 *                        Optional ?stream=<camera_id> point lookup; an absent
 *                        filter returns every route, a present-but-empty one
 *                        matches nothing. Unlike the flags above, a backend
 *                        failure is propagated to the HTTP status line rather
 *                        than reported only in the body.
 */
typedef enum
{
  GET_LIVE_STREAM_INFO = 1 << 0,
  GET_READY_INFO = 1 << 1,
  GET_LIVE_INFO = 1 << 2,
  GET_STARTUP_INFO = 1 << 3,
  GET_METRICS_INFO = 1 << 4,
  GET_METADATA_INFO = 1 << 5,
  GET_MODEL_STATUS_INFO = 1 << 6,   /* /api/v1/model/status -- app model-pool state */
  GET_STREAM_ROUTE_INFO = 1 << 7,   /* GET /api/v1/stream/route -- routing plane only */
} NvDsServerGetRequestPropFlag;

typedef enum
{
  PROCESS_MODE = 1 << 0,
} NvDsServerOsdPropFlag;

typedef enum
{
  RELOAD_CONFIG = 1 << 0,
} NvDsServerAnalyticsPropFlag;

typedef enum
{
  TEXT_EMBEDDING_GENERATE = 1 << 0,
} NvDsServerTextEmbeddingPropFlag;

typedef enum
{
  IMAGE_EMBEDDING_GENERATE = 1 << 0,
} NvDsServerImageEmbeddingPropFlag;

typedef enum
{
  ROI_UPDATE = 1 << 0,
} NvDsServerRoiPropFlag;

typedef enum
{
  QUIT_APP = 1 << 0,
} NvDsServerAppInstanceFlag;

typedef enum
{
  QUIT_SUCCESS = 0,
  QUIT_FAIL,
} NvDsServerAppInstanceStatus;

typedef enum
{
  STREAM_ADD_SUCCESS = 0,
  STREAM_ADD_FAIL,
  STREAM_REMOVE_SUCCESS,
  STREAM_REMOVE_FAIL,
} NvDsServerStreamStatus;

typedef enum
{
  MODEL_LOAD_SUCCESS = 0,
  MODEL_LOAD_FAIL,
  MODEL_UNLOAD_SUCCESS,
  MODEL_UNLOAD_FAIL,
  /* model/update = IN-PLACE checkpoint transition on the live instance group
   * (same network, new weights, NEW version). Stream routing is a different
   * plane: see NvDsServerRouteStatus / /api/v1/stream/route. */
  MODEL_UPDATE_SUCCESS,
  MODEL_UPDATE_FAIL,
} NvDsServerModelStatus;

typedef enum
{
  STREAM_ROUTE_SUCCESS = 0,
  STREAM_ROUTE_FAIL,
} NvDsServerRouteStatus;

typedef enum
{
  GET_LIVE_STREAM_INFO_SUCCESS = 0,
  GET_LIVE_STREAM_INFO_FAIL,
  GET_READY_INFO_SUCCESS,
  GET_READY_INFO_FAIL,
  GET_LIVE_INFO_SUCCESS,
  GET_LIVE_INFO_FAIL,
  GET_STARTUP_INFO_SUCCESS,
  GET_STARTUP_INFO_FAIL,
  GET_METRICS_INFO_SUCCESS,
  GET_METRICS_INFO_FAIL,
  GET_METADATA_INFO_SUCCESS,
  GET_METADATA_INFO_FAIL,
  GET_MODEL_STATUS_INFO_SUCCESS,
  GET_MODEL_STATUS_INFO_FAIL,
  GET_STREAM_ROUTE_INFO_SUCCESS,
  GET_STREAM_ROUTE_INFO_FAIL,
} NvDsServerGetRequestStatus;

typedef enum
{
  ROI_UPDATE_SUCCESS = 0,
  ROI_UPDATE_FAIL,
} NvDsServerRoiStatus;

typedef enum
{
  DROP_FRAME_INTERVAL_UPDATE_SUCCESS = 0,
  DROP_FRAME_INTERVAL_UPDATE_FAIL,
  SKIP_FRAMES_UPDATE_SUCCESS,
  SKIP_FRAMES_UPDATE_FAIL,
  LOW_LATENCY_MODE_UPDATE_SUCCESS,
  LOW_LATENCY_MODE_UPDATE_FAIL,
} NvDsServerDecStatus;

typedef enum
{
  BITRATE_UPDATE_SUCCESS = 0,
  BITRATE_UPDATE_FAIL,
  FORCE_IDR_UPDATE_SUCCESS,
  FORCE_IDR_UPDATE_FAIL,
  FORCE_INTRA_UPDATE_SUCCESS,
  FORCE_INTRA_UPDATE_FAIL,
  IFRAME_INTERVAL_UPDATE_SUCCESS,
  IFRAME_INTERVAL_UPDATE_FAIL,
} NvDsServerEncStatus;

typedef enum
{
  DEST_CROP_UPDATE_SUCCESS = 0,
  DEST_CROP_UPDATE_FAIL,
  SRC_CROP_UPDATE_SUCCESS,
  SRC_CROP_UPDATE_FAIL,
  INTERPOLATION_METHOD_UPDATE_SUCCESS,
  INTERPOLATION_METHOD_UPDATE_FAIL,
  FLIP_METHOD_UPDATE_SUCCESS,
  FLIP_METHOD_UPDATE_FAIL,
} NvDsServerConvStatus;

typedef enum
{
  BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS = 0,
  BATCHED_PUSH_TIMEOUT_UPDATE_FAIL,
  MAX_LATENCY_UPDATE_SUCCESS,
  MAX_LATENCY_UPDATE_FAIL,
} NvDsServerMuxStatus;

typedef enum
{
  INFER_INTERVAL_UPDATE_SUCCESS = 0,
  INFER_INTERVAL_UPDATE_FAIL,
} NvDsServerInferStatus;

typedef enum
{
  INFERSERVER_INTERVAL_UPDATE_SUCCESS = 0,
  INFERSERVER_INTERVAL_UPDATE_FAIL,
} NvDsServerInferServerStatus;

typedef enum
{
  NVTRACKER_CONFIG_UPDATE_SUCCESS = 0,
  NVTRACKER_CONFIG_UPDATE_FAIL,
} NvDsServerNvTrackerStatus;

typedef enum
{
  PROCESS_MODE_UPDATE_SUCCESS = 0,
  PROCESS_MODE_UPDATE_FAIL,
} NvDsServerOsdStatus;

typedef enum
{
  RELOAD_CONFIG_UPDATE_SUCCESS = 0,
  RELOAD_CONFIG_UPDATE_FAIL,
} NvDsServerAnalyticsStatus;

typedef enum
{
  TEXT_EMBEDDING_GENERATE_SUCCESS = 0,
  TEXT_EMBEDDING_GENERATE_FAIL,
} NvDsServerTextEmbeddingStatus;

typedef enum
{
  IMAGE_EMBEDDING_GENERATE_SUCCESS = 0,
  IMAGE_EMBEDDING_GENERATE_FAIL,
} NvDsServerImageEmbeddingStatus;

typedef enum
{
  StatusOk = 0,                         // HTTP error code : 200
  StatusAccepted,                       // HTTP error code : 202
  StatusBadRequest,                     // HTTP error code : 400
  StatusUnauthorized,                   // HTTP error code : 401
  StatusForbidden,                      // HTTP error code : 403
  StatusMethodNotAllowed,               // HTTP error code : 405
  StatusNotAcceptable,                  // HTTP error code : 406
  StatusProxyAuthenticationRequired,    // HTTP error code : 407
  StatusRequestTimeout,                 // HTTP error code : 408
  StatusPreconditionFailed,             // HTTP error code : 412
  StatusPayloadTooLarge,                // HTTP error code : 413
  StatusUriTooLong,                     // HTTP error code : 414
  StatusUnsupportedMediaType,           // HTTP error code : 415
  StatusInternalServerError,            // HTTP error code : 500
  StatusNotImplemented,                 // HTTP error code : 501
  StatusServiceUnavailable              // HTTP error code : 503
} NvDsServerStatusCode;

typedef struct NvDsServerErrorInfo
{
  std::pair < int, std::string > err_log;
  NvDsServerStatusCode code;
  /* Machine-readable error envelope (emitted as error{code,message,hint} in the
   * HTTP response body when code != StatusOk). Clients must never parse prose:
   * `err_code` is a stable identifier (e.g. "MODEL_NOT_LOADED",
   * "SHADOW_WITHOUT_MODEL"); `hint` is the actionable next step. */
  std::string err_code;
  std::string hint;
} NvDsServerErrorInfo;

typedef struct NvDsServerDecInfo
{
  std::string root_key;
  std::string stream_id;
  guint drop_frame_interval;
  guint skip_frames;
  gboolean low_latency_mode;
  NvDsServerDecStatus status;
  NvDsServerDecPropFlag dec_flag;
  std::string dec_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerDecInfo;

typedef struct NvDsServerEncInfo
{
  std::string root_key;
  std::string stream_id;
  guint bitrate;
  gboolean force_idr;
  gboolean force_intra;
  guint iframeinterval;
  NvDsServerEncStatus status;
  NvDsServerEncPropFlag enc_flag;
  std::string enc_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerEncInfo;

typedef struct NvDsServerConvInfo
{
  std::string root_key;
  std::string stream_id;
  std::string src_crop;
  std::string dest_crop;
  guint flip_method;
  guint interpolation_method;
  NvDsServerConvStatus status;
  NvDsServerConvPropFlag conv_flag;
  std::string conv_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerConvInfo;

typedef struct NvDsServerMuxInfo
{
  std::string root_key;
  gint batched_push_timeout;
  guint max_latency;
  NvDsServerMuxStatus status;
  NvDsServerMuxPropFlag mux_flag;
  std::string mux_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerMuxInfo;

typedef struct NvDsServerRoiInfo
{
  std::string root_key;
  std::string stream_id;
  guint roi_count;
  std::vector < RoiDimension > vect;
  NvDsServerRoiStatus status;
  NvDsServerRoiPropFlag roi_flag;
  std::string roi_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerRoiInfo;

typedef struct NvDsServerStreamInfo
{
  std::string key;
  std::string value_camera_id;
  std::string value_camera_name;
  std::string value_camera_url;
  std::string value_change;
  std::string value_creation_time;

  std::string metadata_resolution;
  std::string metadata_codec;
  std::string metadata_framerate;
  /* Full metadata object forwarded verbatim as a JSON string (variable keys),
   * so custom per-stream metadata (e.g. model selection) reaches the app. */
  std::string metadata_json;

  std::string headers_source;
  std::string headers_created_at;
  NvDsServerStreamStatus status;
  std::string stream_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerStreamInfo;

/* MODEL PLANE (lifecycle) request info -- /api/v1/model/load|unload|update.
 *
 *   load:   register + warm one immutable (name, version); deploy its
 *           instance set (declarative full set).
 *   unload: reclaim an idle version (version "" => all non-serving versions).
 *   update: IN-PLACE checkpoint transition on the LIVE instance group -- same
 *           network, new weights, NEW version. from_version is a REQUIRED CAS
 *           guard. Engines: `engine_file` (single-instance groups only) XOR
 *           `engine_files` (exact per-GPU map).
 *
 * Stream routing is a different plane (never mixed in here): NvDsServerRouteInfo.
 * The raw request `value` object is also carried verbatim in `value_json` --
 * that is what travels in-band to the consuming element (one schema end-to-end). */
typedef struct NvDsServerModelInfo
{
  std::string key;
  std::string name;                /**< logical model name (no '@' / ';')          */
  std::string version;             /**< positive-integer version (required: load/update) */
  std::string from_version;        /**< update only: REQUIRED CAS guard             */
  std::string config_file;         /**< load: required on first load of a name      */
  std::string engine_file;         /**< SEED engine (load) / single-instance (update) */
  std::map<guint, std::string> engine_files;  /**< EXACT per-GPU engine map (gpu -> path) */
  std::vector<guint> gpus;         /**< gpu id set: load = instance placement (one per gpu), unload = subset to drain */
  gboolean has_gpus;               /**< 'gpus' present in the request               */
  std::string value_json;          /**< the request "value" object, serialized verbatim */
  std::string value_change;        /**< set by handler: "model_load"|"model_unload"|"model_update" */
  NvDsServerModelStatus status;
  std::string model_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerModelInfo;

/* STREAM PLANE (routing) request info -- /api/v1/stream/route.
 * Declarative routing request: per-group desired (model, shadow) state plus an
 * optional default switch. Parse/admission constraints enforced at this layer:
 *   - at least one of routes[] / default present (EMPTY_REQUEST);
 *   - every route carries `streams`: an explicit camera_id list or the literal
 *     "all" (an omitted list must fail validation, never mean "everything");
 *   - no stream named in two routes (STREAM_DUPLICATE_ROUTE);
 *   - a route ALWAYS resolves to a serving model: `"model": null` (explicit
 *     passthrough) with a shadow present is rejected (SHADOW_WITHOUT_MODEL);
 *   - refs are structured {name, version[, gpu]}; version is a positive integer.
 * Whether a stream exists is checked by the nvmultiurisrcbin callback (it owns
 * the camera_id -> source_id map); model-loaded/state checks belong to the
 * consuming element. The raw `value` travels verbatim in `value_json`. */
typedef struct NvDsServerRouteRef
{
  std::string name;
  std::string version;
  gint gpu;                        /**< -1 = no placement hint                     */
} NvDsServerRouteRef;

typedef struct NvDsServerRouteEntry
{
  std::vector<std::string> camera_ids;  /**< empty + all_streams=TRUE => "all"     */
  gboolean all_streams;
  gboolean has_model;              /**< "model" key present                        */
  gboolean model_null;             /**< "model": null => PASSTHROUGH               */
  NvDsServerRouteRef model;
  gboolean has_shadow;             /**< "shadow" key present                       */
  gboolean shadow_null;            /**< "shadow": null => clear                    */
  NvDsServerRouteRef shadow;
} NvDsServerRouteEntry;

typedef struct NvDsServerRouteInfo
{
  std::string key;
  std::vector<NvDsServerRouteEntry> routes;
  gboolean has_default;
  gboolean default_has_model;
  NvDsServerRouteRef default_model;
  gboolean default_has_shadow;
  gboolean default_shadow_null;
  NvDsServerRouteRef default_shadow;
  gint64 if_revision;              /**< -1 = no optimistic-concurrency guard       */
  std::string value_json;          /**< the request "value" object, serialized verbatim */
  std::string value_change;        /**< set by handler: "stream_route"             */
  NvDsServerRouteStatus status;
  std::string route_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerRouteInfo;

typedef struct NvDsGetRequestInfo
{
  std::string root_key;
  std::string stream_id;
  gboolean is_text;
  NvDsServerGetRequestStatus status;
  NvDsServerGetRequestPropFlag get_request_flag;
  std::string get_request_log;
  std::string uri;
  Json::Value stream_info;
  std::vector<NvDsSensorInfo*> sensorInfo_vec;
  NvDsServerErrorInfo err_info;
} NvDsServerGetRequestInfo;

typedef struct NvDsServerInferInfo
{
  std::string root_key;
  std::string stream_id;
  guint interval;
  NvDsServerInferStatus status;
  NvDsServerInferPropFlag infer_flag;
  std::string infer_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerInferInfo;
typedef struct NvDsServerOsdInfo
{
  std::string root_key;
  std::string stream_id;
  guint process_mode;
  NvDsServerOsdStatus status;
  NvDsServerOsdPropFlag osd_flag;
  std::string osd_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerOsdInfo;

typedef struct NvDsServerAnalyticsInfo
{
  std::string root_key;
  std::string stream_id;
  std::string config_file_path;
  NvDsServerAnalyticsStatus status;
  NvDsServerAnalyticsPropFlag analytics_flag;
  std::string analytics_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerAnalyticsInfo;

typedef struct NvDsServerTextEmbeddingInfo
{
  std::string text_input;
  std::string model;
  std::string id;
  long created;
  Json::Value data;
  NvDsServerTextEmbeddingStatus status;
  NvDsServerTextEmbeddingPropFlag text_embedding_flag;
  std::string text_embedding_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerTextEmbeddingInfo;

typedef struct NvDsServerImageEmbeddingInfo
{
  std::string image_path;
  std::string model;
  gboolean has_bbox;
  gdouble bbox_left;
  gdouble bbox_top;
  gdouble bbox_width;
  gdouble bbox_height;
  std::string id;
  long created;
  Json::Value data;
  NvDsServerImageEmbeddingStatus status;
  NvDsServerImageEmbeddingPropFlag image_embedding_flag;
  std::string image_embedding_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerImageEmbeddingInfo;

typedef struct NvDsServerAppInstanceInfo
{
  std::string root_key;
  gboolean app_quit;
  NvDsServerAppInstanceStatus status;
  NvDsServerAppInstanceFlag appinstance_flag;
  std::string app_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerAppInstanceInfo;

typedef struct NvDsServerInferServerInfo
{
  std::string root_key;
  std::string stream_id;
  guint interval;
  NvDsServerInferServerStatus status;
  NvDsServerInferServerPropFlag inferserver_flag;
  std::string inferserver_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerInferServerInfo;

typedef struct NvDsServerNvTrackerInfo
{
  std::string root_key;
  std::string stream_id;
  std::string config_path;
  NvDsServerNvTrackerStatus status;
  NvDsServerNvTrackerPropFlag nvTracker_flag;
  std::string nvTracker_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerNvTrackerInfo;
typedef struct NvDsServerResponseInfo
{
  std::string status;
  std::string reason;
  Json::Value stream_info;
} NvDsServerResponseInfo;

typedef struct NvDsServerConfig
{
  std::string ip;
  std::string port;
} NvDsServerConfig;

using cb_func = std::function < NvDsServerStatusCode (const Json::Value & req_info,
      const Json::Value & in,
      Json::Value & out, struct mg_connection * conn, void *ctx)>;

typedef struct NvDsServerCallbacks
{
  std::function < void (NvDsServerRoiInfo * roi_info, void *ctx) > roi_cb;
  std::function < void (NvDsServerDecInfo * dec_info, void *ctx) > dec_cb;
  std::function < void (NvDsServerEncInfo * enc_info, void *ctx) > enc_cb;
  std::function < void (NvDsServerStreamInfo * stream_info,
    void *ctx) > stream_cb;
  std::function < void (NvDsServerModelInfo * model_info,
    void *ctx) > model_cb;
  std::function < void (NvDsServerRouteInfo * route_info,
    void *ctx) > route_cb;
  std::function < void (NvDsServerInferInfo * infer_info,
    void *ctx) > infer_cb;
  std::function < void (NvDsServerConvInfo * conv_info, void *ctx) > conv_cb;
  std::function < void (NvDsServerMuxInfo * mux_info, void *ctx) > mux_cb;
  std::function < void (NvDsServerInferServerInfo * inferserver_info,
    void *ctx) > inferserver_cb;
  std::function < void (NvDsServerNvTrackerInfo * nvTracker_info,
    void *ctx) > nvTracker_cb;
  std::function < void (NvDsServerOsdInfo * osd_info, void *ctx) > osd_cb;
  std::function < void (NvDsServerAppInstanceInfo * appinstance_info,
    void *ctx) > appinstance_cb;
  std::function < void (NvDsServerAnalyticsInfo * analytics_info, void *ctx) > analytics_cb;
  std::function < void (NvDsServerTextEmbeddingInfo * text_embedding_info, void *ctx) > text_embedding_cb;
  std::function < void (NvDsServerImageEmbeddingInfo * image_embedding_info, void *ctx) > image_embedding_cb;
  std::function < void (NvDsServerGetRequestInfo * get_request_info,
    void *ctx) > get_request_cb;
  std::unordered_map <std::string, cb_func> custom_cb_endpt;
} NvDsServerCallbacks;

class NvDsRestServer;
NvDsRestServer* nvds_rest_server_start (NvDsServerConfig * server_config, NvDsServerCallbacks * server_cb, void* custom_ctx);
void nvds_rest_server_stop (NvDsRestServer *ctx);
bool iequals (const std::string & a, const std::string & b);

#endif
