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

#include <vector>
#include <iostream>
#include <chrono>

#include "buffer_probe.hpp"


using namespace std;

namespace deepstream {

class NvDsMeasureLatency : public BufferProbe::IBufferObserver {
 public:
  virtual probeReturn handleBuffer(BufferProbe& probe, const Buffer& buffer) {
    auto latency_info = buffer.measureLatency();
    for (auto& latency : latency_info) {
      cout << "Source id = " << latency.source_id
          << " Frame_num = " << latency.frame_num
          << " Frame latency = " << latency.latency << " (ms)"
          << endl;
    }
    return probeReturn::Probe_Ok;
  }
};

}