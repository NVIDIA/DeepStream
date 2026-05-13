/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef DS3D_COMMON_HPP_LIDAR_CUSTOM_PROCESS_HPP
#define DS3D_COMMON_HPP_LIDAR_CUSTOM_PROCESS_HPP


#include "infer_datatypes.h"
namespace ds3d {

using namespace nvdsinferserver;

class IInferCustomPreprocessor {
public:
  virtual ~IInferCustomPreprocessor() = default;
  virtual NvDsInferStatus preproc(
      GuardDataMap& dataMap, SharedIBatchArray batchArray, cudaStream_t stream) = 0;
};

}  // namespace ds3d

#endif  // DS3D_COMMON_HPP_LIDAR_CUSTOM_PROCESS_HPP
