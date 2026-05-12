/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>Defines analytics metadata </b>
 *
 * @b Description: This file defines metadata concerning nvdsanalytics plugin.
 */
 /**
 * @defgroup  NvDsMetaApi Metadata APIs
 *
 * This section describes types and functions of Metadata APIs
 * programming interface.
 *
 */

#ifndef _NVDS_ANALYTICS_META_H_
#define _NVDS_ANALYTICS_META_H_

#include <gst/gst.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include "nvdsmeta.h"

#define NVDS_USER_FRAME_META_NVDSANALYTICS NVDS_FRAME_META_NVDSANALYTICS
#define NVDS_USER_OBJ_META_NVDSANALYTICS NVDS_OBJ_META_NVDSANALYTICS

#ifdef __cplusplus
extern "C"
{
#endif
/**
 * @defgroup ee_analytics_group Analytics Metadata
 * Defines metadata concerning nvdsanalytics plugin.
 * @ingroup NvDsMetaApi
 * @{
 */

/**
 * Holds a set of nvdsanalytics object level metadata.
 */
typedef struct
{
/** Holds the array of ROI labels in which object is present */
  std::vector <std::string> roiStatus;
 /** Holds the array  of OverCrowding labels in which object is present  */
  std::vector <std::string> ocStatus;
 /** Holds the array of line crossing labels which object has crossed */
  std::vector <std::string> lcStatus;
 /** Holds the direction string for the tracked object */
  std::string dirStatus;
 /** Holds unique identifier for nvdsanalytics instance */
  guint unique_id;
 /** Holds the status string for the tracked object */
  std::string objStatus;
} NvDsAnalyticsObjInfo;

/**
 * Holds a set of nvdsanalytics framelevel metadata.
 */
typedef struct
{
  /** Holds a map of boolean status of overcrowding for configured ROIs,
   * which can be accessed using key, value pair; where key is the ROI label
   */
  std::unordered_map<std::string, bool> ocStatus;

  /** Holds a map of total count of valid objects in ROI  for configured ROIs,
   * which can be accessed using key, value pair; where key is the ROI label
   */
  std::unordered_map<std::string, uint32_t> objInROIcnt;

  /** Holds a map of total count of Line crossing in current frame for configured lines,
   * which can be accessed using key, value pair; where key is the line crossing label
   */
  std::unordered_map<std::string, uint64_t> objLCCurrCnt;

  /** Holds a map of total cumulative count of Line crossing  for configured lines,
   * can be accessed using key, value pair; where key is the line crossing label
   */
  std::unordered_map<std::string, uint64_t> objLCCumCnt;

  /** Holds unique identifier for nvdsanalytics instance */
  guint unique_id;

  /** Holds a map of total count of objects for each class ID,
   * can be accessed using key, value pair; where key is class ID
   */
  std::unordered_map<int, uint32_t> objCnt;
} NvDsAnalyticsFrameMeta;

#ifdef __cplusplus
}
#endif

#endif
/** @} */
