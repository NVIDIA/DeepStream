/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
