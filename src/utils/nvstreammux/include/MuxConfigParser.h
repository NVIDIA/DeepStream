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

#ifndef __GST_NVSTREAMMUX_PROPERTY_PARSER_H__
#define __GST_NVSTREAMMUX_PROPERTY_PARSER_H__

#include <glib.h>
#include <unordered_map>
#include "nvstreammux_batch.h"
#include <yaml-cpp/yaml.h>

/** @{ Default Streammux Config props */

/** Defaults for PROP_GROUP */
static NvStreammuxBatchMethod constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_METHOD_ALGO_TYPE = BATCH_METHOD_ROUND_ROBIN;
static guint                  constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_BATCH_SIZE             = 1;
static gboolean               constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_ADAPTIVE_BATCHING      = TRUE;
static gboolean               constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_ENABLE_SOURCE_CONTROL  = FALSE;
static gboolean               constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_MAX_FPS_CONTROL        = FALSE;
static guint                  constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_N      = 120;
static guint                  constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_FPS_D      = 1;
static guint                  constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_N      = 5;
static guint                  constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MIN_FPS_D      = 1;
static guint                  constexpr NVSTREAMMUX_DEFAULT_PROP_GROUP_OVERALL_MAX_SAME_SOURCE_FRAMES = 1 ;

/** Defaults for SOURCE_GROUP */
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_N            = 60;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_D            = 1;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_N            = 30;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_D            = 1;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY             = 0;
static guint constexpr NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH = 1;

/** @} */

/** 
 * Data-structures provided by Amiya along with
 * Design-doc: https://docs.google.com/presentation/d/1fgmbUiiSJlUlIWk9ffMvUsW6Nerqg67_Pp-12vywVzs/edit?ts=5c6f53a0#slide=id.g52a9c8f141_0_5
 * and other means
 *  @} 
 */

class MuxConfigParser
{
    public:

    MuxConfigParser();
    ~MuxConfigParser();

    bool SetConfigFile(gchar const * const cfgFilePath);

    /** 
     * @brief  Parse the Config file for per-source
     *         properties
     *         Note: For batch-size, if config unavailable in the file,
     *         it shall be set to default only if batchPolicy->batch_size
     *         was not set to a non-zero value by the caller.
     * @param  batchPolicy [IN/OUT] The batchPolicy to
     *         fill the source properties in
     * @return true if successful, false otherwise
     */
    bool ParseConfigs(BatchPolicyConfig* batchPolicy, bool defaults=false, guint numSources=1);

    private:

    void ParseTxtConfigCommonProps(BatchPolicyConfig* batchPolicy, gchar* group, GKeyFile* keyFile);

    bool ParseTxtConfigPerSourceProps(NvStreammuxSourceProps* sourceProps, gchar* group, GKeyFile* keyFile);

    bool ParseTxtConfig(BatchPolicyConfig* batchPolicy);

    void ParseYmlConfigCommonProps(BatchPolicyConfig* batchPolicy, std::string group);

    bool ParseYmlConfigPerSourceProps(NvStreammuxSourceProps* sourceProps, std::string group);

    bool ParseYmlConfig(BatchPolicyConfig* batchPolicy);

    gchar* cfgFile;
};

#endif /*__GST_NVSTREAMMUX_PROPERTY_PARSER_H__*/
