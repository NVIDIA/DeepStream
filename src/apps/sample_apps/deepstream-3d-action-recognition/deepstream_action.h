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

#ifndef __DEEPSTREAM_ACTION_H__
#define __DEEPSTREAM_ACTION_H__

#include <cuda_runtime_api.h>

#include <fstream>
#include <functional>
#include <gst/gst.h>
#include <glib.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <queue>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <unordered_map>
#include <map>
#include <vector>

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "gstnvdsinfer.h"

#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

/* Print log message*/
#define LOG(out, level, fmt, ...)  \
    fprintf(out, "[%s: DS_3DAR] " fmt "\n", #level, ##__VA_ARGS__)

/* Print Debug message if ENABLE_DEBUG not zero */
#define LOG_DEBUG(fmt, ...)                     \
    if (gActionConfig.debug >= kDebugEnable) {  \
      LOG(stdout, DEBUG, fmt, ##__VA_ARGS__);   \
    }

/* Print Error message*/
#define LOG_ERROR(fmt, ...)  \
    LOG(stderr, ERROR, fmt, ##__VA_ARGS__)

template<typename T>
class SafePtr: public std::unique_ptr<T, std::function<void(T*)>> {
public:
  SafePtr(T* p, std::function<void(T*)> freefn):
    std::unique_ptr<T, std::function<void(T*)>>(p, freefn)
  {}
};

enum DebugLevel {
  kDebugDisable = 0,
  kDebugEnable,
  kDebugVerbose,
};

// FPS calculation for each source stream
class FpsCalculation {
  struct FpsStats{
    double startTime = 0;
    uint64_t sumFrames = 0;
    float curFps = 0;
    float avgFps = 0;
  };
public:
  FpsCalculation(uint32_t interval)
    : _max_frame_nums(interval)
  {}
  float updateFps(uint32_t source_id) {
    struct timeval time_now;
    gettimeofday(&time_now, nullptr);
    double now = (double)time_now.tv_sec + time_now.tv_usec / (double)1000000; // second
    float fps = -1.0f;
    auto iSrc = _timestamps.find(source_id);
    if (iSrc != _timestamps.end()) {
      auto & tms = iSrc->second;
      fps = tms.size() / (now - tms.front());
      while (tms.size() >= _max_frame_nums) {
        tms.pop();
      }
      auto & stats = _fpsStats[source_id];
      stats.curFps = fps;
      stats.avgFps = stats.sumFrames / (now - stats.startTime);
      stats.sumFrames++;
    } else {
      iSrc = _timestamps.emplace(source_id, std::queue<double>()).first;
      _fpsStats.emplace(source_id, FpsStats{now, 1, 0, 0});
    }
    iSrc->second.push(now);

    return fps;
  }

  // get dataset of current fps and average fps
  void getAllFps(std::vector<std::pair<float, float>>& fps) {
      for (auto& s: _fpsStats) {
        fps.emplace_back(std::make_pair(s.second.curFps, s.second.avgFps));
      }
  }
private:
  std::unordered_map<uint32_t, std::queue<double>> _timestamps;
  uint32_t _max_frame_nums = 50;
  std::map<uint32_t, FpsStats> _fpsStats;
};

struct NvDsARConfig
{
  // stream source list
  std::vector<std::string> uri_list;

  // display sink settings
  gboolean display_sync = true;

  // nvdspreprocess plugin config file path
  std::string preprocess_config;
  // nvinfer plugin config file path
  std::string infer_config;
  // nvinferserver(ds-triton) plugin config file path
  std::string triton_infer_config;

  // nvstreammux settings
  uint32_t muxer_height = 720;
  uint32_t muxer_width = 1280;
  // batched-push-timeout in usec, default value 40ms
  int32_t muxer_batch_timeout = 40000;

  // tiler settings
  uint32_t tiler_height = 720;
  uint32_t tiler_width = 1280;

  // debug level, disabled by default
  DebugLevel debug = kDebugDisable;

  // enable fps print on screen. enabled by default
  gboolean enableFps = TRUE;

  // enable to use fakesink
  gboolean useFakeSink = FALSE;
};

// parse action recognition config into NvDsARConfig
bool parse_action_config(const char* action_config_path, NvDsARConfig& config);

#endif