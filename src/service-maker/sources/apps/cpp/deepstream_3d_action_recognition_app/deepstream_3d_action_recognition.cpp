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

#include <fstream>
#include <functional>
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
#include <iomanip>

#include "pipeline.hpp"
#include "tensor.hpp"

#define FPS_INTERVAL 30
#define MAX_STR_LEN 2048
#define NVDS_PREPROCESS_BATCH_META 28

using namespace deepstream;

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


static FpsCalculation gFpsCal(50);


class AddLabel : public BufferProbe::IBatchMetadataOperator
{
public:
  AddLabel() {}
  virtual ~AddLabel() {}

  virtual probeReturn handleData(BufferProbe &probe, BatchMetadata &data)
  {
    data.iterate([&data](const UserMetadata &user_meta) {
      PreprocessBatchUserMetadata preprocess_batch_meta(user_meta);
      for (auto &roi_meta : preprocess_batch_meta.getRois()) {
        roi_meta.iterate([&data, &roi_meta](const ClassifierMetadata &classifier_meta) {
          unsigned int num_labels = classifier_meta.nLabels();
          for (unsigned int i = 0; i < num_labels; i++) {
            std::string label = classifier_meta.getLabel(i);
            std::stringstream ss;
            ss << "Label: "<< label;
            std::string str = ss.str();
            int font_size = 12;
            char font[] = "Serif";
            NvOSD_TextParams osd_label = NvOSD_TextParams{
                (char *)str.c_str(),
                static_cast<unsigned int>(roi_meta.rectParams().left),                                      //< x offset
                (uint32_t)std::max<int32_t>(roi_meta.rectParams().top - 10, 0),                                      //< y offset
                {font, (unsigned int)font_size, {1.0, 1.0, 1.0, 1.0}}, //< font and color
                1,
                {0.0, 0.0, 0.0, 1.0} //< background
            };
            DisplayMetadata display_meta;
            if(!data.acquire(display_meta)) {
              printf("Failed to acquire display metadata\n");
              return;
            }
            display_meta.add(osd_label);
            roi_meta.frameMetadata().append(display_meta);
          }
       });
      }
     }, NVDS_PREPROCESS_BATCH_META);

    FrameMetadata::Iterator frame_itr;
    for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next())
    {
      FrameMetadata &frame_meta = frame_itr->get();
      float fps = gFpsCal.updateFps(frame_meta.sourceId());
      if (fps >= 0) {
        std::stringstream ss;
        ss << "FPS: ";
        ss << std::fixed << std::setprecision(2);
        ss << fps;
        std::string str = ss.str();
        int font_size = 10;
        char font[] = "Serif";
        NvOSD_TextParams label = NvOSD_TextParams{
            (char *)str.c_str(),
            0,                                      //< x offset
            40,                                      //< y offset
            {font, (unsigned int)font_size, {1.0, 1.0, 1.0, 1.0}}, //< font and color
            1,
            {0.0, 0.0, 0.0, 1.0} //< background
        };
        DisplayMetadata display_meta;
        if(!data.acquire(display_meta)) {
          printf("Failed to acquire display metadata\n");
          return probeReturn::Probe_Ok;
        }
        display_meta.add(label);
        (*frame_itr)->append(display_meta);
      }
    }
    static uint64_t sFrameCount = 0;
    sFrameCount++;
    if (sFrameCount >= FPS_INTERVAL) {
      sFrameCount = 0;
      std::vector<std::pair<float, float>> fps;
      gFpsCal.getAllFps(fps);
      char fpsText[MAX_STR_LEN] = {'\0'};
      for (auto& p : fps) {
        size_t remaining = MAX_STR_LEN - strlen(fpsText);
        if (remaining > 1) {
          snprintf(fpsText + strlen(fpsText), remaining, "%.2f (%.2f) \t", p.first, p.second);
        }
      }
      if (!fps.empty()) {
        printf("FPS(cur/avg): %s\n", fpsText);
      }
    }
    return probeReturn::Probe_Ok;
  }
};

int main (int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <YAML config file>" << std::endl;
    return 0;
  }

  try {
    std::string file = argv[1];
    std::string suffix = "yaml";
    if (std::equal(suffix.rbegin(), suffix.rend(), file.rbegin())) {
      Pipeline pipeline("deepstream-3d-action-recognition", file);
      pipeline.attach("pgie", new BufferProbe("add_label_to_display", new AddLabel)).start().wait();
    }
    else {
      std::cout << "Invalid File Type: " << argv[1] << " Please provide a .yaml config file" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
