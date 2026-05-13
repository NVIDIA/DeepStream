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

#ifndef DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_H
#define DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_H

#include "ds3d/common/common.h"
#include "ds3d/common/abi_dataprocess.h"

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API ds3d::abiRefDataFilter* createLidarPreprocessFilter();
DS3D_EXTERN_C_END

#endif  // DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_H
