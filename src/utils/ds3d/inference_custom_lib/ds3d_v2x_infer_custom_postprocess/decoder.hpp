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

#ifndef DECODER_HPP
#define DECODER_HPP

#include <memory>
#include <vector>
#include "dtype.hpp"

namespace bevfusion{
namespace decoder{

struct Position {
  float x, y, z;
};

struct Size {
  float w, l, h;  // x, y, z
};

struct Velocity {
  float vx, vy;
};

struct ThreeDBox {
  Position position;
  Size size;
  Velocity velocity;
  float yaw;
  float score;
  int category;
  int ibatch;
  int id;
};

typedef struct ThreeDBox ThreeDBox;
typedef std::vector<ThreeDBox> ThreeDBoxes;

struct Head{
  void* heatmap;
  void* rotation;
  void* height;
  void* dim;
  void* vel;
  void* reg;
  unsigned int fm_area;
  unsigned int fm_width;
  unsigned int batch;
};

class Decoder {
public:
  struct DecoderParameter {
    float out_size_factor = 4;
    ds3d::Float2 voxel_size{0.075, 0.075};
    ds3d::Float2 pc_range{-54.0f, -54.0f};
    ds3d::Float3 post_center_range_start{-61.2, -61.2, -10.0};
    ds3d::Float3 post_center_range_end{61.2, 61.2, 10.0};
    unsigned int num_classes = 5;
  };

public:
  static std::unique_ptr<Decoder> createDecoder(const DecoderParameter& param);
  virtual bool init(const Decoder::DecoderParameter& param) = 0;
  virtual ThreeDBoxes forward(
    const std::vector<Head>& heads,
    float confidence_threshold, float nms_threshold, void* stream) = 0;
  virtual ~Decoder() = default;
};

};
}; // bevfusion

#endif // DECODER_HPP
