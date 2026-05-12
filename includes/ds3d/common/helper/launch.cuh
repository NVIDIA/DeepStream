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

#ifndef DS3D_COMMON_HELPER_LAUNCH_CUH
#define DS3D_COMMON_HELPER_LAUNCH_CUH

#include "check.hpp"

namespace ds3d {

#define LINEAR_LAUNCH_THREADS 512
#define cuda_linear_index (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_x (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_y (blockDim.y * blockIdx.y + threadIdx.y)
#define divup(a, b) ((static_cast<int>(a) + static_cast<int>(b) - 1) / static_cast<int>(b))

#define cuda_linear_launch(kernel, stream, num, ...)                                \
  do {                                                                              \
    size_t __num__ = (size_t)(num);                                                 \
    size_t __blocks__ = divup(__num__, LINEAR_LAUNCH_THREADS);                      \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__); \
    ds3d::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);          \
  } while (false)

#define cuda_launch(kernel, grid, block, stream, ...)                                \
  do {                                                                              \
    kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);                       \
    ds3d::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);          \
  } while (false)


#define cuda_2d_launch(kernel, stream, nx, ny, nz, ...)                      \
  do {                                                                       \
    dim3 __threads__(32, 32);                                                \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32), nz);                       \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, nz, __VA_ARGS__); \
    ds3d::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);   \
  } while (false)
};      // namespace ds3d

#endif  // DS3D_COMMON_HELPER_LAUNCH_CUH