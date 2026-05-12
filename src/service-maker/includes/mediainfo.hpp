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
 * <b>MediaInfo class for aquiring media information</b>
 *
 *
 */
#ifndef NVIDIA_DEEPSTREAM_MEDIAINFO
#define NVIDIA_DEEPSTREAM_MEDIAINFO

#include <string>
#include <memory>
#include <vector>

namespace deepstream {

struct StreamInfo {
  std::string codec;

  virtual ~StreamInfo() {}
};

struct AudioStreamInfo : public StreamInfo {
  unsigned int channels;
};

struct VideoStreamInfo : public StreamInfo {
  unsigned int width;
  unsigned int height;
  struct {
    unsigned int num;
    unsigned int denom;
  } framerate;
};

struct MediaInfo {
  bool error = false;
  uint64_t duration = 0;
  bool live = false;
  operator bool() const { return !error; };
  std::vector<std::unique_ptr<StreamInfo>> streams;
  static std::unique_ptr<struct MediaInfo> discover(std::string uri);
};

}
#endif