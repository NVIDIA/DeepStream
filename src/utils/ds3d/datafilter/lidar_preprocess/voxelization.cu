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
#include <memory>
#include <vector>
#include "voxelization.hpp"

#include "check.hpp"
#include "launch.cuh"
#include "dtype.hpp"

namespace bevfusion {
namespace pointpillars {

static const int voxel_feature_size   = 9;
static __device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}

static __device__ inline void insert_to_hash_table(const uint32_t key, uint32_t *value, const uint32_t hash_size,
                                                   uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % hash_size /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  while (true) {
    uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
    if (pre_key == empty_key) {
      hash_table[slot + hash_size] = atomicAdd(value, 1);
      break;
    } else if (pre_key == key) {
      break;
    }
    slot = (slot + 1) % hash_size;
  }
}

static __device__ inline uint32_t lookup_hash_table(const uint32_t key, const uint32_t hash_size, const uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % hash_size /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  int cnt = 0;
  while (true /* need to be adjusted according to data*/) {
    cnt++;
    if (hash_table[slot] == key) {
      return hash_table[slot + hash_size];
    } else if (hash_table[slot] == empty_key) {
      return empty_key;
    } else {
      slot = (slot + 1) % hash_size;
    }
  }
  return empty_key;
}

static __global__ void build_hash_table_kernel(size_t points_size, const float *points, VoxelizationParameter param,
                                               unsigned int *hash_table, unsigned int *real_voxel_num) {
  int point_idx = cuda_linear_index;
  if (point_idx >= points_size) return;

  float px = points[param.input_feature * point_idx + 0];
  float py = points[param.input_feature * point_idx + 1];
  float pz = points[param.input_feature * point_idx + 2];

  int voxel_idx = floorf((px - param.min_range.x) / param.voxel_size.x);
  if (voxel_idx < 0 || voxel_idx >= param.grid_size.x) return;

  int voxel_idy = floorf((py - param.min_range.y) / param.voxel_size.y);
  if (voxel_idy < 0 || voxel_idy >= param.grid_size.y) return;

  int voxel_idz = floorf((pz - param.min_range.z) / param.voxel_size.z);
  if (voxel_idz < 0 || voxel_idz >= param.grid_size.z) return;
  unsigned int voxel_key = (voxel_idz * param.grid_size.y + voxel_idy) * param.grid_size.x + voxel_idx;
  insert_to_hash_table(voxel_key, real_voxel_num, points_size * 2, hash_table);
}

static __global__ void voxelization_kernel(size_t points_size, const float *points, VoxelizationParameter param,
                                           unsigned int *hash_table, unsigned int *num_points_per_voxel, half *voxels_features,
                                           uint4 *voxel_indices, unsigned int ibatch, unsigned int voxel_id_base) {
  int point_idx = cuda_linear_index;
  if (point_idx >= points_size) return;

  float px = points[param.input_feature * point_idx + 0];
  float py = points[param.input_feature * point_idx + 1];
  float pz = points[param.input_feature * point_idx + 2];

  if (px < param.min_range.x || px >= param.max_range.x || py < param.min_range.y || py >= param.max_range.y ||
      pz < param.min_range.z || pz >= param.max_range.z) {
    return;
  }

  int voxel_idx = floorf((px - param.min_range.x) / param.voxel_size.x);
  int voxel_idy = floorf((py - param.min_range.y) / param.voxel_size.y);
  int voxel_idz = floorf((pz - param.min_range.z) / param.voxel_size.z);
  if ((voxel_idx < 0 || voxel_idx >= param.grid_size.x)) {
    return;
  }
  if ((voxel_idy < 0 || voxel_idy >= param.grid_size.y)) {
    return;
  }
  if ((voxel_idz < 0 || voxel_idz >= param.grid_size.z)) {
    return;
  }

  unsigned int voxel_offset = (voxel_idz * param.grid_size.y + voxel_idy) * param.grid_size.x + voxel_idx;

  // scatter to voxels
  unsigned int voxel_id = lookup_hash_table(voxel_offset, points_size * 2, hash_table) - voxel_id_base;
  if (voxel_id >= param.max_voxels) {
    return;
  }

  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);
  if (current_num >= param.max_points_per_voxel) 
    return;

  unsigned int dst_offset = (voxel_id * param.max_points_per_voxel + current_num) * voxel_feature_size;
  unsigned int src_offset = point_idx * param.input_feature;
  voxels_features[dst_offset + 0] = points[src_offset + 0];  //x
  voxels_features[dst_offset + 1] = points[src_offset + 1];  //y
  voxels_features[dst_offset + 2] = points[src_offset + 2];  //z
  voxels_features[dst_offset + 3] = points[src_offset + 3];  //w

  // now only deal with batch_size = 1
  // since not sure what the input format will be if batch size > 1
  voxel_indices[voxel_id] = make_uint4(ibatch, voxel_idx, voxel_idy, voxel_idz);
}

static __global__ void compute_voxel_feature_kernel(size_t num_voxels, uint4* indices, unsigned int *num_points_per_voxel, int max_points_per_voxel, half *voxel_features, 
  float voxel_width, float voxel_height,
  float range_min_x, float range_min_y
) {
  int ipoint     = threadIdx.x % max_points_per_voxel;
  int inner_voxel = (threadIdx.x / max_points_per_voxel) % 3;
  int voxel_idx  = (blockIdx.x * blockDim.y + threadIdx.y) * 3 + inner_voxel;

  float3 mean_point = {0};
  float x = 0, y = 0, z = 0;
  unsigned int voxel_offset = (voxel_idx * max_points_per_voxel + ipoint) * voxel_feature_size;
  if(voxel_idx < num_voxels && threadIdx.x < 30){
    int valid_points_num = num_points_per_voxel[voxel_idx] > max_points_per_voxel ? max_points_per_voxel : num_points_per_voxel[voxel_idx];
    if(ipoint < valid_points_num){
      x = voxel_features[voxel_offset + 0];
      y = voxel_features[voxel_offset + 1];
      z = voxel_features[voxel_offset + 2];
    }
  }

  for(int i = 0; i < max_points_per_voxel; ++i){
    float borad_x = __shfl_sync(0xFFFFFFFF, x, inner_voxel * max_points_per_voxel + i);
    float borad_y = __shfl_sync(0xFFFFFFFF, y, inner_voxel * max_points_per_voxel + i);
    float borad_z = __shfl_sync(0xFFFFFFFF, z, inner_voxel * max_points_per_voxel + i);
    mean_point.x += borad_x;
    mean_point.y += borad_y;
    mean_point.z += borad_z;
  }

  if(voxel_idx >= num_voxels || threadIdx.x >= 30) return;
  int valid_points_num = num_points_per_voxel[voxel_idx] > max_points_per_voxel ? max_points_per_voxel : num_points_per_voxel[voxel_idx];
  if(ipoint >= valid_points_num){
    half zero = 0.0f;

    #pragma unroll
    for(int i = 0; i < voxel_feature_size; ++i){
      voxel_features[voxel_offset + i] = zero;
    }
    return;
  }

  uint4 voxel_loc = indices[voxel_idx];
  mean_point.x /= valid_points_num;
  mean_point.y /= valid_points_num;
  mean_point.z /= valid_points_num;

  float x_offset = voxel_width / 2 + voxel_loc.y * voxel_width + range_min_x;
  float y_offset = voxel_height / 2 + voxel_loc.z * voxel_height + range_min_y;
  voxel_features[voxel_offset + 4] = x - mean_point.x;
  voxel_features[voxel_offset + 5] = y - mean_point.y;
  voxel_features[voxel_offset + 6] = z - mean_point.z;
  voxel_features[voxel_offset + 7] = x - x_offset;
  voxel_features[voxel_offset + 8] = y - y_offset;
}

ds3d::Int3 VoxelizationParameter::compute_grid_size(const ds3d::Float3 &max_range, const ds3d::Float3 &min_range,
                                                      const ds3d::Float3 &voxel_size) {
  ds3d::Int3 size;
  size.x = static_cast<int>(std::round((max_range.x - min_range.x) / voxel_size.x));
  size.y = static_cast<int>(std::round((max_range.y - min_range.y) / voxel_size.y));
  size.z = static_cast<int>(std::round((max_range.z - min_range.z) / voxel_size.z));
  return size;
}

class VoxelizationImplement : public Voxelization {
 public:
  virtual ~VoxelizationImplement() {
    if (hash_table_) checkRuntime(cudaFree(hash_table_));
    if (d_voxel_features_) checkRuntime(cudaFree(d_voxel_features_));
    if (d_per_voxel_num_) checkRuntime(cudaFree(d_per_voxel_num_));
    if (d_voxel_indices_) checkRuntime(cudaFree(d_voxel_indices_));

    if (batched_d_real_num_voxels_) checkRuntime(cudaFree(batched_d_real_num_voxels_));
    if (batched_h_real_num_voxels_) checkRuntime(cudaFreeHost(batched_h_real_num_voxels_));
  }

  bool init(VoxelizationParameter param) {
    this->param_ = param;
    this->output_grid_bytes = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1};

    this->hash_table_bytes     = param_.max_points * 2 * 2 * sizeof(unsigned int);
    this->voxel_num_bytes      = param_.max_voxels * sizeof(unsigned int);
    this->voxel_features_bytes = param_.max_batch * param_.max_voxels * param_.max_points_per_voxel * voxel_feature_size * sizeof(half);
    this->voxel_indices_bytes  = param_.max_batch * param_.max_voxels * sizeof(ds3d::Int4);

    checkRuntime(cudaMalloc(&hash_table_, hash_table_bytes));
    printf("feats buffer size: %d\n", voxel_features_bytes);
    checkRuntime(cudaMalloc(&d_voxel_features_, voxel_features_bytes));
    checkRuntime(cudaMalloc(&d_per_voxel_num_, voxel_num_bytes));
    checkRuntime(cudaMalloc(&d_voxel_indices_, voxel_indices_bytes));
    checkRuntime(cudaMalloc(&batched_d_real_num_voxels_, param_.max_batch * sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&batched_h_real_num_voxels_, param_.max_batch * sizeof(unsigned int)));
    return true;
  }

  // points and voxels must be of half type
  virtual void forward(const float * const* points, const int *num_points, unsigned int batch, void *stream) override {
    cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

    batched_real_num_voxels_ = 0;
    checkRuntime(cudaMemsetAsync(batched_d_real_num_voxels_, 0, sizeof(unsigned int), _stream));

    for(unsigned int ibatch = 0; ibatch < batch; ++ibatch){
      auto _features = d_voxel_features_ + batched_real_num_voxels_ * param_.max_points_per_voxel * voxel_feature_size;
      auto _indices  = d_voxel_indices_  + batched_real_num_voxels_;
      const float *_points = points[ibatch];
      unsigned int _num_points = num_points[ibatch];
      checkRuntime(cudaMemsetAsync(hash_table_, 0xff, _num_points * 2 * sizeof(unsigned int), _stream));
      checkRuntime(cudaMemsetAsync(d_per_voxel_num_, 0, voxel_num_bytes, _stream));
      cuda_linear_launch(build_hash_table_kernel, _stream, _num_points, _points, param_, hash_table_, batched_d_real_num_voxels_);
      checkRuntime(cudaMemcpyAsync(batched_h_real_num_voxels_, batched_d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost, _stream));

      cuda_linear_launch(voxelization_kernel, _stream, _num_points, _points, param_, hash_table_,
                          d_per_voxel_num_, _features, reinterpret_cast<uint4 *>(_indices), ibatch, batched_real_num_voxels_);
      checkRuntime(cudaStreamSynchronize(_stream));

      unsigned int current_num_voxels = *batched_h_real_num_voxels_ - batched_real_num_voxels_;
      batched_real_num_voxels_        = *batched_h_real_num_voxels_;

      dim3 block(32, 16);
      dim3 grid(((current_num_voxels + 2) / 3 + block.y - 1) / block.y);
      cuda_launch(compute_voxel_feature_kernel, grid, block, _stream, current_num_voxels, reinterpret_cast<uint4 *>(_indices), d_per_voxel_num_, param_.max_points_per_voxel, _features, 
        param_.voxel_size.x, param_.voxel_size.y, param_.min_range.x, param_.min_range.y);
    }
  }

  virtual unsigned int num_voxels() override { return batched_real_num_voxels_; }
  virtual const unsigned int* num_voxels_device() override{
    return batched_d_real_num_voxels_;
  }

  virtual const ds3d::Int4 *indices() override { return d_voxel_indices_; }

  virtual const ds3d::half *features() override { return (ds3d::half*)d_voxel_features_; }

 private:
  VoxelizationParameter param_;
  unsigned int batched_real_num_voxels_ = 0;
  std::vector<int> output_grid_bytes;

  unsigned int *hash_table_ = nullptr;
  unsigned int *batched_d_real_num_voxels_ = nullptr;
  unsigned int *batched_h_real_num_voxels_ = nullptr;
  unsigned int *d_per_voxel_num_ = nullptr;
  half *d_voxel_features_ = nullptr;
  ds3d::Int4 *d_voxel_indices_ = nullptr;
  unsigned int hash_table_bytes;
  unsigned int voxel_features_bytes;
  unsigned int voxel_indices_bytes;
  unsigned int voxel_num_bytes;
};

std::unique_ptr<Voxelization> create_voxelization(VoxelizationParameter param) {
  auto impl = std::make_unique<VoxelizationImplement>();
  if (!impl->init(param)) {
    impl.reset();
  }
  return impl;
}

};  // namespace lidar
};  // namespace pointpillar
