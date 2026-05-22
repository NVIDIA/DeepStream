/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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