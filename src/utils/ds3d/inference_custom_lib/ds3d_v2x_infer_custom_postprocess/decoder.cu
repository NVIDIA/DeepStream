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

#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>

#include "check.hpp"
#include "decoder.hpp"
#include "launch.cuh"

namespace bevfusion {
namespace decoder {

#define MAX_DETECTION_BOX_SIZE 1024

// - ["car", "truck", "construction_vehicle", "bus", "trailer"]
// - ["barrier","motorcycle", "bicycle","pedestrian", "traffic_cone"]

static __global__ void decode_kernel(unsigned int num_boxes, unsigned int fm_area, unsigned int fm_width, const float* reg,
                                     const float* height, const float* dim, const float* rot, const float* vel,
                                     const float* score, Decoder::DecoderParameter param, int ihead, float confidence_threshold,
                                     ThreeDBox* output, unsigned int* output_size, unsigned int max_output_size) {
  int ibox = cuda_linear_index;
  if (ibox >= num_boxes) return;

  int ib = ibox / fm_area;
  int iinner = ibox % fm_area;
  int ix = iinner % fm_width;
  int iy = iinner / fm_width;
  int label = 0;
  float confidence = score[((ib * param.num_classes) + 0) * fm_area + iinner];

  for (int i = 1; i < param.num_classes; ++i) {
    float local_score = score[((ib * param.num_classes) + i) * fm_area + iinner];
    if (local_score > confidence) {
      label = i;
      confidence = local_score;
    }
  }

  if (confidence < confidence_threshold) return;
  auto xs = iy + reg[((ib * 2) + 0) * fm_area + iinner];
  auto ys = ix + reg[((ib * 2) + 1) * fm_area + iinner];
  xs = xs * param.out_size_factor * param.voxel_size.x + param.pc_range.x;
  ys = ys * param.out_size_factor * param.voxel_size.y + param.pc_range.y;

  auto zs = (height[((ib * 1) + 0) * fm_area + iinner]);
  if (xs < param.post_center_range_start.x || xs > param.post_center_range_end.x) return;
  if (ys < param.post_center_range_start.y || ys > param.post_center_range_end.y) return;

  float3 dim_;
  dim_.x = dim[((ib * 3) + 1) * fm_area + iinner];
  dim_.y = dim[((ib * 3) + 0) * fm_area + iinner];
  dim_.z = dim[((ib * 3) + 2) * fm_area + iinner];
  // zs = zs - dim_.z * 0.5f;

  if (zs < param.post_center_range_start.z || zs > param.post_center_range_end.z) return;

  unsigned int iout = atomicAdd(output_size, 1);
  if (iout >= max_output_size) return;

  auto& obox = output[iout];
  auto vx = (vel[((ib * 2) + 0) * fm_area + iinner]);
  auto vy = (vel[((ib * 2) + 1) * fm_area + iinner]);
  auto rs = -atan2(rot[((ib * 2) + 0) * fm_area + iinner], rot[((ib * 2) + 1) * fm_area + iinner]);

  *(float3*)&obox.position = make_float3(xs, ys, zs);
  *(float3*)&obox.size = dim_;
  obox.velocity.vx = vx;
  obox.velocity.vy = vy;
  obox.yaw = rs;
  obox.score = confidence;
  obox.category = ihead * param.num_classes + label;
  obox.ibatch = ib;
}

static ThreeDBoxes cpu_nms(ThreeDBoxes& boxes, float threshold = 4.0f) {
  std::sort(boxes.begin(), boxes.end(),
            [](ThreeDBoxes::const_reference a, ThreeDBoxes::const_reference b) { return a.score > b.score; });

  ThreeDBoxes output;
  output.reserve(boxes.size());

  std::vector<bool> remove_flags(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (remove_flags[i]) continue;

    auto& a = boxes[i];
    output.emplace_back(a);

    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (remove_flags[j]) continue;

      auto& b = boxes[j];
      if (b.ibatch == a.ibatch) {
        float local_threshold = threshold;
        if(b.category != a.category){
          local_threshold = threshold * 0.2f;
        }
        float dist = pow(a.position.x - b.position.x, 2.0f) + pow(a.position.y - b.position.y, 2.0f);
        if (dist < local_threshold) {
          remove_flags[j] = true;
        }
      }
    }
  }
  return output;
}

class DecoderImplement : public Decoder {
 public:
  virtual ~DecoderImplement() {
    if (output_device_size_) checkRuntime(cudaFree(output_device_size_));
    if (output_device_boxes_) checkRuntime(cudaFree(output_device_boxes_));
    if (output_host_size_) checkRuntime(cudaFreeHost(output_host_size_));
    if (output_host_boxes_) checkRuntime(cudaFreeHost(output_host_boxes_));
  }

  virtual bool init(const DecoderParameter& param) final {
    param_ = param;
    checkRuntime(cudaMalloc(&output_device_size_, sizeof(unsigned int)));
    checkRuntime(cudaMalloc(&output_device_boxes_, MAX_DETECTION_BOX_SIZE * sizeof(ThreeDBox)));
    checkRuntime(cudaMallocHost(&output_host_size_, sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&output_host_boxes_, MAX_DETECTION_BOX_SIZE * sizeof(ThreeDBox)));
    output_cache_.resize(MAX_DETECTION_BOX_SIZE);
    return true;
  }

  virtual ThreeDBoxes forward(const std::vector<Head>& heads, float confidence_threshold, float nms_threshold,
                              void* stream) final {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    checkRuntime(cudaMemsetAsync(output_device_size_, 0, sizeof(unsigned int), _stream));

    for (size_t ihead = 0; ihead < heads.size(); ++ihead) {
      auto& head = heads[ihead];
      cuda_linear_launch(decode_kernel, _stream, head.fm_area * head.batch, head.fm_area, head.fm_width, (float*)head.reg,
                         (float*)head.height, (float*)head.dim, (float*)head.rotation, (float*)head.vel, (float*)head.heatmap,
                         param_, ihead, confidence_threshold, output_device_boxes_, output_device_size_, MAX_DETECTION_BOX_SIZE);
    }

    checkRuntime(cudaMemcpyAsync(output_host_boxes_, output_device_boxes_, MAX_DETECTION_BOX_SIZE * sizeof(ThreeDBox),
                                 cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaMemcpyAsync(output_host_size_, output_device_size_, sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    unsigned int real_size = min(MAX_DETECTION_BOX_SIZE, *output_host_size_);
    auto output = ThreeDBoxes(output_host_boxes_, output_host_boxes_ + real_size);
    // std::sort(output.begin(), output.end(), [](ThreeDBox& a, ThreeDBox& b) { return a.score >
    // b.score; }); std::sort(output.begin(), output.end(), [](ThreeDBox& a, ThreeDBox& b) { return
    // a.ibatch < b.ibatch; });
    return cpu_nms(output, nms_threshold);
  }

 private:
  DecoderParameter param_;
  std::vector<ThreeDBox> output_cache_;
  ThreeDBox* output_host_boxes_ = nullptr;
  ThreeDBox* output_device_boxes_ = nullptr;
  unsigned int* output_device_size_ = nullptr;
  unsigned int* output_host_size_ = nullptr;
};

std::unique_ptr<Decoder> Decoder::createDecoder(const Decoder::DecoderParameter& param) {
  auto decoder = std::make_unique<DecoderImplement>();
  if (!decoder->init(param)) {
    decoder.reset();
    return nullptr;
  }
  return decoder;
}

};  // namespace decoder
};  // namespace bevfusion