/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <thread>
#include <condition_variable>

#include "buffer_probe.hpp"

using namespace std;

namespace deepstream {

using time_point = std::chrono::steady_clock::time_point;

class FPSCounter : public BufferProbe::IBatchMetadataObserver
{
public:
  static const int HEADER_PRINT_INTERVAL = 10;
  std::unordered_map<unsigned int, std::chrono::microseconds> accumulated_time;
  std::unordered_map<unsigned int, uint64_t> buf_count_lifetime;
  std::chrono::steady_clock::time_point last_measurement_time;
  std::mutex count_mutex;
  bool pause_measurment;
  int num_surfaces_per_frame;
  std::set<unsigned int> pad_idxs;
  std::unordered_map<unsigned int, std::chrono::steady_clock::time_point>
      first_frame_time, last_frame_time;
  std::unordered_map<unsigned int, uint64_t> buf_count;

  std::thread scheduler_;
  std::condition_variable cv_;
  int last_header_print_interval;
  std::size_t last_pad_idx_count;


  FPSCounter() {
    scheduler_ = std::thread([&](){
      do {
        std::unique_lock<std::mutex> lock(this->count_mutex);
        this->measure_and_print_unlocked();
        this->cv_.wait_for(lock, std::chrono::seconds(5));
      } while (!this->pause_measurment);
    });
    last_pad_idx_count = 0;
    last_header_print_interval = 0;
    num_surfaces_per_frame = 1;
    pause_measurment = false;
  }

  virtual ~FPSCounter() {
    { 
      std::unique_lock<std::mutex> lock(this->count_mutex);
      pause_measurment = true;
      cv_.notify_one();
    }
    scheduler_.join();
  }

  void measure_and_print_unlocked() {
    time_point current = std::chrono::steady_clock::now();
    auto current_buf_count = std::move(buf_count);
    // Print the header after every HEADER_PRINT_INTERVAL or if a source
    // was dynamically added and pad_idxs size grew
    if (this->last_header_print_interval == 0 ||
        this->last_pad_idx_count != this->pad_idxs.size()) {
        // this->print_header_unlocked();
        this->last_header_print_interval = 0;
    }
    this->last_header_print_interval =
        (this->last_header_print_interval + 1) % HEADER_PRINT_INTERVAL;
    std::ostringstream ostr;
    ostr << "**FPS:  " << std::fixed << std::setprecision(2);
    for (auto pad_idx : this->pad_idxs) {
        uint64_t current_buf_count_source = 0;
        // Insert does not update a value if the key already exists
        accumulated_time.insert({pad_idx, std::chrono::microseconds(0)});
        time_point first_frame_time = current;
        time_point end_time = current;
        bool source_is_eos = false;
        if (this->first_frame_time.find(pad_idx) != this->first_frame_time.end())
        first_frame_time = this->first_frame_time[pad_idx];
        auto last_frame_time = this->last_frame_time.find(pad_idx);
        if (last_frame_time != this->last_frame_time.end()) {
        // If the eos was reached, change the end to the time EOS received instead
        // of the interval end.
        end_time = last_frame_time->second;
        source_is_eos = true;
        }
        if (current_buf_count.find(pad_idx) != current_buf_count.end()) {
        current_buf_count_source = current_buf_count[pad_idx];
        this->buf_count_lifetime[pad_idx] += current_buf_count_source;
        }
        // If the source reached EOS before the current measurement interval, do
        // not print current values. However, if the EOS was reached in the current
        // interval, print whatever was measured.
        if (source_is_eos && end_time < this->last_measurement_time)
        {
          ostr << "--.--";
        }
        else
        {
          time_point start =
              std::max(first_frame_time, this->last_measurement_time);
          double interval = std::chrono::duration_cast<std::chrono::microseconds>(
                                end_time - start)
                                .count();
          ostr << (1000000 * current_buf_count_source / interval);
        }
        double lifetime = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - first_frame_time)
                            .count() +
                        accumulated_time[pad_idx].count();
        ostr << " (" << (1000000 * this->buf_count_lifetime[pad_idx] / lifetime)
            << ")\t";
    }

    ostr << "\n";

    if (ostr.str() != "**FPS:  \n")
      std::cout << ostr.str() << std::flush;
    this->last_measurement_time = current;
    this->last_pad_idx_count = this->pad_idxs.size();
}

  virtual probeReturn handleData(BufferProbe& probe, const BatchMetadata& data) {
    time_point current = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(count_mutex);
    if (pause_measurment) return probeReturn::Probe_Ok;
    auto cnt = 0;
    pad_idxs.clear();
    data.iterate([&](const FrameMetadata& frame_meta) {
      if (cnt++ % num_surfaces_per_frame == 0) {
        auto pad_idx = frame_meta.padIndex();
        pad_idxs.insert(pad_idx);
        // Insert does not update a value if the key already exists
        first_frame_time.insert({pad_idx, current});
        buf_count[pad_idx]++;
      }
    });
    return probeReturn::Probe_Ok; 
  }

};

}