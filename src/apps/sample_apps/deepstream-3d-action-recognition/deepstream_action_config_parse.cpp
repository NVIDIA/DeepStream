/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_action.h"

constexpr const char* kActionRecognition = "action-recognition";
constexpr const char* kUriList = "uri-list";
constexpr const char* kPreprocessConfig = "preprocess-config";
constexpr const char* kInferenceConfig = "infer-config";
constexpr const char* kTritonInferenceConfig = "triton-infer-config";
constexpr const char* kMuxerHeight = "muxer-height";
constexpr const char* kMuxerWidth = "muxer-width";
constexpr const char* kMuxerBatchTimeout = "muxer-batch-timeout"; // usec

constexpr const char* kTilerHeight = "tiler-height";
constexpr const char* kTilerWidth = "tiler-width";
constexpr const char* kDisplaySync = "display-sync";

constexpr const char* kDebug = "debug";
constexpr const char* kEnableFps = "enable-fps";
constexpr const char* kUseFakeSink = "fakesink";

#define PARSE_FAILED(statement, fmt, ...)               \
    if (!(statement)) {                                 \
      LOG_ERROR(fmt, ##__VA_ARGS__);                    \
      return false;                                     \
    }

#define PARSE_WITH_ERROR(statement, fmt, ...)           \
    do {                                                \
      statement;                                        \
      SafePtr<GError> ptrErr__(error, g_error_free); \
      if (error) {                                     \
        LOG_ERROR(fmt ", error msg: %s", ##__VA_ARGS__, ptrErr__->message); \
        return false;                                   \
      }                                                 \
    } while(0)

bool parse_action_config(const char* path, NvDsARConfig& config) {
  SafePtr<GKeyFile> keyfile(g_key_file_new(), g_key_file_free);
  SafePtr<gchar*> safeKeys(nullptr, g_strfreev);
  GError *error = nullptr;

  PARSE_WITH_ERROR(
    g_key_file_load_from_file (keyfile.get(), path, G_KEY_FILE_NONE, &error),
    "load config: %s failed", path);
  PARSE_FAILED(
    g_key_file_has_group (keyfile.get(), kActionRecognition),
    "parse config: %s failed, group: %s is missing", path, kActionRecognition);

  PARSE_WITH_ERROR(
    safeKeys.reset(g_key_file_get_keys (keyfile.get(), kActionRecognition, NULL, &error)),
    "parse keys of group: %s failed in config: %s", kActionRecognition, path
  );

  PARSE_WITH_ERROR(
    config.debug = (DebugLevel)g_key_file_get_integer (keyfile.get(), kActionRecognition,
      kDebug, &error),
    "parse key: %s failed in config: %s", kDebug, path);

  SafePtr<gchar*> uri_list(nullptr, g_strfreev);
  gsize num_strings = 0;
  PARSE_WITH_ERROR(
    uri_list.reset(g_key_file_get_string_list (keyfile.get(), kActionRecognition,
      kUriList, &num_strings, &error)),
    "parse key: %s failed in config: %s", kUriList, path);
  for (gsize i = 0; i < num_strings; ++i) {
    config.uri_list.push_back(uri_list.get()[i]);
  }
  g_assert(config.uri_list.size() > 0);

  SafePtr<gchar> config_str(nullptr, g_free);

  PARSE_WITH_ERROR(
    config_str.reset(g_key_file_get_string (keyfile.get(), kActionRecognition,
      kPreprocessConfig, &error)),
    "parse key: %s failed in config: %s", kPreprocessConfig, path);
  config.preprocess_config = config_str.get();

  if (g_key_file_has_key(keyfile.get(), kActionRecognition, kInferenceConfig, nullptr)) {
    PARSE_WITH_ERROR(
      config_str.reset(g_key_file_get_string (keyfile.get(), kActionRecognition,
        kInferenceConfig, &error)),
      "parse key: %s failed in config: %s", kInferenceConfig, path);
    config.infer_config = config_str.get();
  }
  if (g_key_file_has_key(keyfile.get(), kActionRecognition, kTritonInferenceConfig, nullptr)) {
    PARSE_WITH_ERROR(
      config_str.reset(g_key_file_get_string (keyfile.get(), kActionRecognition,
        kTritonInferenceConfig, &error)),
      "parse key: %s failed in config: %s", kTritonInferenceConfig, path);
    config.triton_infer_config = config_str.get();
  }

  PARSE_FAILED(!config.triton_infer_config.empty() || !config.infer_config.empty(),
    "No key %s or %s found in config file", kTritonInferenceConfig, kInferenceConfig);

  PARSE_WITH_ERROR(
    config.muxer_batch_timeout = g_key_file_get_integer (keyfile.get(), kActionRecognition,
      kMuxerBatchTimeout, &error),
    "parse key: %s failed in config: %s", kMuxerBatchTimeout, path);

  PARSE_WITH_ERROR(
    config.muxer_height = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kMuxerHeight, &error),
    "parse key: %s failed in config: %s", kMuxerHeight, path);

  PARSE_WITH_ERROR(
    config.muxer_width = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kMuxerWidth, &error),
    "parse key: %s failed in config: %s", kMuxerWidth, path);
  PARSE_WITH_ERROR(
    config.tiler_height = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kTilerHeight, &error),
    "parse key: %s failed in config: %s", kTilerHeight, path);
  PARSE_WITH_ERROR(
    config.tiler_width = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kTilerWidth, &error),
    "parse key: %s failed in config: %s", kTilerWidth, path);

  PARSE_WITH_ERROR(
    config.display_sync = (uint32_t)g_key_file_get_boolean (keyfile.get(), kActionRecognition,
      kDisplaySync, &error),
    "parse key: %s failed in config: %s", kDisplaySync, path);

    PARSE_WITH_ERROR(
    config.enableFps = (uint32_t)g_key_file_get_boolean (keyfile.get(), kActionRecognition,
      kEnableFps, &error),
    "parse key: %s failed in config: %s", kEnableFps, path);

    PARSE_WITH_ERROR(
    config.useFakeSink = (uint32_t)g_key_file_get_boolean (keyfile.get(), kActionRecognition,
      kUseFakeSink, &error),
    "parse key: %s failed in config: %s", kUseFakeSink, path);

  return true;
}