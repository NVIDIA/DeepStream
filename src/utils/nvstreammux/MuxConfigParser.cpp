/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "MuxConfigParser.h"
#include <string.h>
#include <sys/stat.h>
#include <cstring>

/** @{ Streammux Config file properties that are per-source: SOURCE_GROUP */
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX         = "source-config-";
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FPS_N = "max-fps-n";
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FPS_D = "max-fps-d";
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MIN_FPS_N = "min-fps-n";
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MIN_FPS_D = "min-fps-d";
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_PRIORITY  = "priority";
static std::string const NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FRAMES_PER_BATCH  = "max-num-frames-per-batch";
/** @} */

/** @{ Streammux Config file properties that are common: PROP_GROUP */
static std::string const NVSTREAMMUX_CONFIG_PROP_GROUP                  = "property";
static std::string const NVSTREAMMUX_CONFIG_PROP_BATCH_METHOD_ALGO_TYPE = "algorithm-type";    /**< uint32_t */
static std::string const NVSTREAMMUX_CONFIG_PROP_BATCH_SIZE             = "batch-size";        /**< uint32_t */
static std::string const NVSTREAMMUX_CONFIG_PROP_ADAPTIVE_BATCHING      = "adaptive-batching"; /**< bool */
static std::string const NVSTREAMMUX_CONFIG_PROP_ENABLE_SOURCE_CONTROL  = "enable-source-rate-control"; /**< bool */
static std::string const NVSTREAMMUX_CONFIG_PROP_MAX_FPS_CONTROL        = "max-fps-control"; /**< bool */
static std::string const NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_N       = "overall-max-fps-n"; /**< uint32_t */
static std::string const NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_D       = "overall-max-fps-d"; /**< uint32_t */
static std::string const NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_N       = "overall-min-fps-n"; /**< uint32_t */
static std::string const NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_D       = "overall-min-fps-d"; /**< uint32_t */
static std::string const NVSTREAMMUX_CONFIG_PROP_MAX_SAME_SOURCE_FRAMES = "max-same-source-frames"; /**< uint32_t */
/** @} */

#ifdef DEBUG
#define CHECK_ERROR(error, field, defaultValue) \
    if (error) { \
        g_printerr ("[Error while parsing streammux config file: %s]\n", error->message); \
        field = defaultValue; \
        error = nullptr; \
    }
#else
#define CHECK_ERROR(error, field, defaultValue) \
    if (error) { \
        field = defaultValue; \
        error = nullptr; \
    }
#endif

MuxConfigParser::MuxConfigParser()
        : cfgFile(nullptr)
{
}

bool MuxConfigParser::SetConfigFile(gchar const * const cfgFilePath)
{
  int size = strlen(cfgFilePath) + 1;
  cfgFile = (char*) malloc(sizeof(char) * size);
  std::strncpy (cfgFile, cfgFilePath, size);
  /*Checks whether the file exists or not*/
  struct stat buffer;
  return (stat (cfgFile, &buffer) == 0);
}

bool MuxConfigParser::ParseYmlConfigPerSourceProps(NvStreammuxSourceProps* sourceProps, std::string group)
{
    YAML::Node configyml = YAML::LoadFile(cfgFile);
    /*Set Defaults*/
    sourceProps->source_max_fps_n = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_N;
    sourceProps->source_max_fps_d = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_D;
    sourceProps->source_min_fps_n = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_N;
    sourceProps->source_min_fps_d = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_D;
    sourceProps->priority = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY;
    sourceProps->max_num_frames_per_batch = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH;
    for(YAML::const_iterator itr = configyml[group].begin(); itr != configyml[group].end(); ++itr)
    {
      std::string paramKey = itr->first.as<std::string>();
      if(paramKey == NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FPS_N) {
        sourceProps->source_max_fps_n = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FPS_D) {
        sourceProps->source_max_fps_d = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MIN_FPS_N) {
        sourceProps->source_min_fps_n = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MIN_FPS_D) {
        sourceProps->source_min_fps_d = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_PRIORITY) {
        sourceProps->priority = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FRAMES_PER_BATCH) {
        sourceProps->max_num_frames_per_batch = itr->second.as<gint>();
      } else {
        std::cout << "[WARNING] Unknown param found in streammux config file: " << paramKey << std::endl;
      }
    }

    return true;
}

void MuxConfigParser::ParseYmlConfigCommonProps(BatchPolicyConfig* batchPolicy, std::string group)
{
    YAML::Node configyml = YAML::LoadFile(cfgFile);
    guint set_batch_size = batchPolicy->batch_size;
    guint batch_size_parsed = 0;
    /*Set defaults*/
    batchPolicy->type = NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_METHOD_ALGO_TYPE;
    batchPolicy->adaptive_batching = NVSTREAMMUX_DEFAULT_PROP_GROUP_ADAPTIVE_BATCHING;
    batchPolicy->enable_source_rate_control = NVSTREAMMUX_DEFAULT_PROP_GROUP_ENABLE_SOURCE_CONTROL;
    batchPolicy->enable_max_fps_control = NVSTREAMMUX_DEFAULT_PROP_GROUP_MAX_FPS_CONTROL;
    batchPolicy->batch_size = NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_SIZE;
    batchPolicy->overall_max_fps_n = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_N;
    batchPolicy->overall_max_fps_d = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_D;
    batchPolicy->overall_min_fps_n = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_N;
    batchPolicy->overall_min_fps_d = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_D;
    batchPolicy->max_same_source_frames = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH;
    for(YAML::const_iterator itr = configyml[group].begin(); itr != configyml[group].end(); ++itr)
    {
      std::string paramKey = itr->first.as<std::string>();
      if(paramKey == NVSTREAMMUX_CONFIG_PROP_BATCH_METHOD_ALGO_TYPE) {
        batchPolicy->type = static_cast<NvStreammuxBatchMethod>(itr->second.as<gint>());
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_ADAPTIVE_BATCHING) {
        batchPolicy->adaptive_batching = itr->second.as<gboolean>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_ENABLE_SOURCE_CONTROL) {
        batchPolicy->enable_source_rate_control = itr->second.as<gboolean>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_MAX_FPS_CONTROL) {
        batchPolicy->enable_max_fps_control = itr->second.as<gboolean>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_BATCH_SIZE) {
        batchPolicy->batch_size = itr->second.as<gint>();
        batch_size_parsed = 1;
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_N) {
        batchPolicy->overall_max_fps_n = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_D) {
        batchPolicy->overall_max_fps_d = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_N) {
        batchPolicy->overall_min_fps_n = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_D) {
        batchPolicy->overall_min_fps_d = itr->second.as<gint>();
      } else if (paramKey == NVSTREAMMUX_CONFIG_PROP_MAX_SAME_SOURCE_FRAMES) {
        batchPolicy->max_same_source_frames = itr->second.as<gint>();
      } else {
        std::cout << "[WARNING] Unknown param found in streammux config file: " << paramKey << std::endl;
      }
    }
    if (set_batch_size != 0 && batch_size_parsed == 0) {
      batchPolicy->batch_size = set_batch_size;
    }
}

bool MuxConfigParser::ParseYmlConfig(BatchPolicyConfig* batchPolicy)
{
    YAML::Node configyml = YAML::LoadFile(cfgFile);
    gboolean parse_err = false;
    for(YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr)
    {
        std::string paramKey = itr->first.as<std::string>();
        /** Check if the config is a proper NVSTREAMMUX_CONFIG_PROP_GROUP */
        if (paramKey == NVSTREAMMUX_CONFIG_PROP_GROUP) {
          ParseYmlConfigCommonProps(batchPolicy, paramKey);
        }
        /** Check if the config is a proper NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX */
        else if (paramKey.compare(0, NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX.length(),
          NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX) == 0) {
            /** Extract sourceIdx - the source index */
            std::string str = paramKey.substr(NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX.length());
            guint64 sourceIdx = std::stoull(str);
            /** Extract props for sourceIdx and insert in the source_props map */
            {
                NvStreammuxSourceProps props;
                bool ok = ParseYmlConfigPerSourceProps(&props, paramKey);
                parse_err = !ok;
                if(ok)
                {
                    batchPolicy->source_props.insert(std::make_pair(sourceIdx, props));
                }
            }
        }
        if (parse_err) {
          std::cout << "yml parsing failed in nvstreammux." << std::endl;
        }
    }
    return true;
}

bool MuxConfigParser::ParseTxtConfigPerSourceProps(NvStreammuxSourceProps* sourceProps, gchar* group, GKeyFile* keyFile)
{
    GError* error = nullptr;
    sourceProps->source_max_fps_n = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FPS_N.c_str(), &error);
    CHECK_ERROR (error, sourceProps->source_max_fps_n, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_N);
    sourceProps->source_max_fps_d = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FPS_D.c_str(), &error);
    CHECK_ERROR (error, sourceProps->source_max_fps_d, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_D);
    sourceProps->source_min_fps_n = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MIN_FPS_N.c_str(), &error);
    CHECK_ERROR (error, sourceProps->source_min_fps_n, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_N);
    sourceProps->source_min_fps_d = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MIN_FPS_D.c_str(), &error);
    CHECK_ERROR (error, sourceProps->source_min_fps_d, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_D);
    sourceProps->priority         = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_PRIORITY.c_str(), &error);
    CHECK_ERROR (error, sourceProps->priority, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY);
    sourceProps->max_num_frames_per_batch = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_SOURCE_GROUP_PROP_MAX_FRAMES_PER_BATCH.c_str(), &error);
    CHECK_ERROR (error, sourceProps->max_num_frames_per_batch, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH);

    return true;
}

void MuxConfigParser::ParseTxtConfigCommonProps(BatchPolicyConfig* batchPolicy, gchar* group, GKeyFile* keyFile)
{
    GError* error = nullptr;
    gchar **keys = NULL;
    gchar **key = NULL;

    keys = g_key_file_get_keys (keyFile, group, NULL, &error);
    if(error) {
      g_printerr ("Error while parsing streammux config file: %s\n", error->message);
    }

    batchPolicy->type=NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_METHOD_ALGO_TYPE;
    batchPolicy->adaptive_batching=NVSTREAMMUX_DEFAULT_PROP_GROUP_ADAPTIVE_BATCHING;
    batchPolicy->enable_source_rate_control=NVSTREAMMUX_DEFAULT_PROP_GROUP_ENABLE_SOURCE_CONTROL;
    batchPolicy->enable_max_fps_control=NVSTREAMMUX_DEFAULT_PROP_GROUP_MAX_FPS_CONTROL;
    batchPolicy->overall_min_fps_n=NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_N;
    batchPolicy->overall_min_fps_d=NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_D;
    batchPolicy->overall_max_fps_n=NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_N;
    batchPolicy->overall_max_fps_d=NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_D;
    batchPolicy->max_same_source_frames=NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH;

    guint set_batch_size = batchPolicy->batch_size;
    for (key = keys; *key; key++) {
      if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_BATCH_METHOD_ALGO_TYPE.c_str())) {
        batchPolicy->type = static_cast<NvStreammuxBatchMethod>(g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_BATCH_METHOD_ALGO_TYPE.c_str(), &error));
        CHECK_ERROR (error, batchPolicy->type, NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_METHOD_ALGO_TYPE);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_ADAPTIVE_BATCHING.c_str())) {
        batchPolicy->adaptive_batching = static_cast<gboolean>(g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_ADAPTIVE_BATCHING.c_str(), &error));
        CHECK_ERROR (error, batchPolicy->adaptive_batching, NVSTREAMMUX_DEFAULT_PROP_GROUP_ADAPTIVE_BATCHING);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_ENABLE_SOURCE_CONTROL.c_str())) {
        batchPolicy->enable_source_rate_control = static_cast<gboolean>(g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_ENABLE_SOURCE_CONTROL.c_str(), &error));
        CHECK_ERROR (error, batchPolicy->enable_source_rate_control, NVSTREAMMUX_DEFAULT_PROP_GROUP_ENABLE_SOURCE_CONTROL);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_MAX_FPS_CONTROL.c_str())) {
        batchPolicy->enable_max_fps_control = static_cast<gboolean>(g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_MAX_FPS_CONTROL.c_str(), &error));
        CHECK_ERROR (error, batchPolicy->enable_max_fps_control, NVSTREAMMUX_DEFAULT_PROP_GROUP_MAX_FPS_CONTROL);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_BATCH_SIZE.c_str())) {
        batchPolicy->batch_size = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_BATCH_SIZE.c_str(), &error);
        /** omit setting default for batch-size if its already non-0 */
        if(error)
        {
          if(set_batch_size == 0) {
            CHECK_ERROR (error, batchPolicy->batch_size, NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_SIZE);
          }
          else {
            batchPolicy->batch_size = set_batch_size;
            error = nullptr;
          }
        }
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_N.c_str())) {
        batchPolicy->overall_max_fps_n = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_N.c_str(), &error);
        CHECK_ERROR (error, batchPolicy->overall_max_fps_n, NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_N);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_D.c_str())) {
        batchPolicy->overall_max_fps_d = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_OVERAL_MAX_FPS_D.c_str(), &error);
        CHECK_ERROR (error, batchPolicy->overall_max_fps_d, NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_D);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_N.c_str())) {
        batchPolicy->overall_min_fps_n = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_N.c_str(), &error);
        CHECK_ERROR (error, batchPolicy->overall_min_fps_n, NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_N);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_D.c_str())) {
        batchPolicy->overall_min_fps_d = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_OVERAL_MIN_FPS_D.c_str(), &error);
        CHECK_ERROR (error, batchPolicy->overall_min_fps_d, NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_D);
      } else if (!g_strcmp0 (*key, NVSTREAMMUX_CONFIG_PROP_MAX_SAME_SOURCE_FRAMES.c_str())) {
        batchPolicy->max_same_source_frames = g_key_file_get_integer (keyFile, group,
            NVSTREAMMUX_CONFIG_PROP_MAX_SAME_SOURCE_FRAMES.c_str(), &error);
        CHECK_ERROR (error, batchPolicy->max_same_source_frames, NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH);
      } else {
        g_print ("Unknown key '%s' for group [%s]\n", *key, group);
      }
    }

    //Free allocations
    if(keys)
        g_strfreev(keys);
    if(error)
        g_error_free (error);
}

bool MuxConfigParser::ParseTxtConfig(BatchPolicyConfig* batchPolicy)
{
    GError* error = nullptr;
    gchar** groupsMuxConfig = nullptr;
    GKeyFile* keyFile = g_key_file_new ();
    bool ret = false;
    if(!keyFile) {
        goto done;
    }
    if (!g_key_file_load_from_file (keyFile, cfgFile, G_KEY_FILE_NONE,
              &error)) {
        goto done;
    }
    /** Get all the groups */
    groupsMuxConfig = g_key_file_get_groups(keyFile, nullptr);
    if(!groupsMuxConfig)
    {
        goto done;
    }
    for (gchar** group = groupsMuxConfig; *group; group++)
    {
        /** Check if the config is a proper NVSTREAMMUX_CONFIG_PROP_GROUP */
        if (0 == strncmp(*group, NVSTREAMMUX_CONFIG_PROP_GROUP.c_str(),
          NVSTREAMMUX_CONFIG_PROP_GROUP.length()))
        {
            ParseTxtConfigCommonProps(batchPolicy, *group, keyFile);
        }
        /** Check if the config is a proper NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX */
        else if (0 == strncmp(*group, NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX.c_str(),
          NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX.length()))
        {
            /** Extract sourceIdx - the source index */
            gchar* keyEnd =
              *group + NVSTREAMMUX_CONFIG_SOURCE_GROUP_PREFIX.length();
            gchar *endPtr;
            guint64 sourceIdx = g_ascii_strtoull(keyEnd, &endPtr, 10);
            /** Extract props for sourceIdx and insert in the source_props map */
            {
                NvStreammuxSourceProps props;
                bool ok = ParseTxtConfigPerSourceProps(&props, *group, keyFile);
                if(ok)
                {
                    batchPolicy->source_props.insert(std::make_pair(sourceIdx, props));
                }
            }
        }
    }
    ret = true;

done:
  g_key_file_free (keyFile);
  keyFile = nullptr;
  if (groupsMuxConfig) {
    g_strfreev (groupsMuxConfig);
  }
  if (error) {
    g_error_free (error);
  }
  if (!ret) {
    std::cout <<  __func__ << " failed" << std::endl;
  }
  return ret;
}

bool MuxConfigParser::ParseConfigs(BatchPolicyConfig* batchPolicy, bool defaults, guint numSources)
{
    if(defaults)
    {
        batchPolicy->type = NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_METHOD_ALGO_TYPE;
        //cfg.type = BATCH_METHOD_PRIORITY;
        batchPolicy->adaptive_batching = NVSTREAMMUX_DEFAULT_PROP_GROUP_ADAPTIVE_BATCHING;
        batchPolicy->batch_size = NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_SIZE;
        batchPolicy->overall_max_fps_n = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_N;
        batchPolicy->overall_max_fps_d = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_D;
        batchPolicy->overall_min_fps_n = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_N;
        batchPolicy->overall_min_fps_d = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_D;
        batchPolicy->max_same_source_frames = NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_SAME_SOURCE_FRAMES;
        NvStreammuxSourceProps source_prop;
        source_prop.source_max_fps_n = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_N;
        source_prop.source_max_fps_d = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_D;
        source_prop.source_min_fps_n = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_N;
        source_prop.source_min_fps_d = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_D;
        source_prop.priority = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY;
        source_prop.max_num_frames_per_batch = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH;
        batchPolicy->enable_source_rate_control = NVSTREAMMUX_DEFAULT_PROP_GROUP_ENABLE_SOURCE_CONTROL;
        batchPolicy->enable_max_fps_control = NVSTREAMMUX_DEFAULT_PROP_GROUP_MAX_FPS_CONTROL;
        for(guint i = 0; i < numSources; i++)
        {
            batchPolicy->source_props.insert(std::pair<unsigned int, NvStreammuxSourceProps>(i, source_prop));
        }

        return true;

    }
    if (g_str_has_suffix(cfgFile, ".yml") || g_str_has_suffix(cfgFile, ".yaml")) {
      if(!ParseYmlConfig (batchPolicy)) {
        return false;
      }
    } else {
      if(!ParseTxtConfig (batchPolicy)) {
        return false;
      }
    }
    return true;

}

MuxConfigParser::~MuxConfigParser()
{
    if(cfgFile)
    {
        free (cfgFile);
        cfgFile = nullptr;
    }
}

