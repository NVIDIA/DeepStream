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

#ifndef __LIDAR_VOXELIZATION_HPP__
#define __LIDAR_VOXELIZATION_HPP__

#include <memory>
#include "dtype.hpp"

namespace bevfusion {
namespace pointpillars {

struct VoxelizationParameter {
    ds3d::Float3 min_range;
    ds3d::Float3 max_range;
    ds3d::Float3 voxel_size;
    ds3d::Int3 grid_size;
    int max_voxels;
    int max_points_per_voxel;
    int max_batch;   // cache memory = max_batch * max_points
    int input_feature = 4;
    int max_points;

    static ds3d::Int3 compute_grid_size(const ds3d::Float3& max_range, const ds3d::Float3& min_range,
                                        const ds3d::Float3& voxel_size);
};

class Voxelization {
  public:
    // points and voxels must be of half-float device pointer
    virtual void forward(const float * const*points, const int *num_points, unsigned int batch, void *stream = nullptr) = 0;

    // N x max_points_per_voxel x 9
    virtual const ds3d::half* features() = 0;

    // N x 4 (x, y, z, ib)
    virtual const ds3d::Int4 *indices() = 0;

    // (1,)  N
    virtual const unsigned int* num_voxels_device() = 0;
    virtual unsigned int num_voxels() = 0;
    virtual ~Voxelization() = default;
};

std::unique_ptr<Voxelization> create_voxelization(VoxelizationParameter param);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __LIDAR_VOXELIZATION_HPP__