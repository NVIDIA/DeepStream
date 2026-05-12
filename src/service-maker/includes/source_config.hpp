/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>SourceConfig definition </b>
 *
 * SourceConfig allows user to configure the source list in a comprehensive
 * way.
 */
#ifndef NVIDIA_DEEPSTREAM_SOURCE_CONFIG
#define NVIDIA_DEEPSTREAM_SOURCE_CONFIG

#include "object.hpp"
#include <vector>
#include <string>

namespace deepstream {

typedef struct {
  // Props for urisrcbin/multiurisrcbin
  std::string uri;
  std::string sensor_id;
  std::string sensor_name;
} SensorInfo;

typedef struct
{
  // Prop for camera type: V4L2/CSI
  std::string camera_type;
  // Common props for Camera source (V4L2/CSI)
  std::string camera_video_format = "NV12";
  std::string camera_width;
  std::string camera_height;
  std::string camera_fps_n;
  std::string camera_fps_d;
  // Props only for CSI source
  int camera_csi_sensor_id = -1;
  // Props only for V4L2 source
  std::string camera_v4l2_dev_node;
  // Extra Props (need to check)
  int gpu_id = 0;
  int nvbuf_mem_type = 0;
  int nvvideoconvert_copy_hw = 0;
} CameraInfo;

class SourceConfig {
 public:
  /**
   * @brief Create a source config from a yaml config file
  */
  SourceConfig(const std::string& config_file);

  /**
   * @brief Get the number of the sources
  */
  uint32_t nSources() const;

  /**
   * @brief Get the number of the camera sources
   */
  uint32_t nCameraSources() const;

  /**
   * @brief List sensor ids in a string, separated by ';'
  */
  std::string listSensorIds() const;

  /**
   * @brief List sensor names in a string, separated by ';'
  */
  std::string listSensorNames() const;

  /**
   * @brief List sensor uris in a string, separated by ';'
  */
  std::string listUris() const;

  /**
   * @brief Get sensor information for a specific source
  */
  SensorInfo getSensorInfo(uint32_t index) const;

  /**
   * @brief Get information for a specific camera source
   */
  CameraInfo getCameraInfo(uint32_t index) const;

  /**
   * @brief Get properties for setting the source bin(s)
  */
  const YAML::Node& getProperties() const;

  /**
   * @brief If using the nvmultiurisrcbin
  */
  bool useMultiUriSrcBin() const;

  /**
   * @brief If using the nvurisrcbin
   */
  bool useUriSrcBin() const;

  /**
   * @brief If using the nvv4l2srcbin
   */
  bool useCameraBin() const;

protected:
  std::vector<SensorInfo> sensor_info_;
  std::vector<CameraInfo> camera_info_;
  YAML::Node properties_;
  bool use_nvmultiurisrcbin_ = false;
  bool use_nvurisrcbin_ = false;
  bool use_camerabin_ = false;
};
}

#endif