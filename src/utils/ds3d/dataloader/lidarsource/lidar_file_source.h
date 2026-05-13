/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef _DS3D_DATALOADER_LIDARSOURCE_LIDAR_FILE_SOURCE_H
#define _DS3D_DATALOADER_LIDARSOURCE_LIDAR_FILE_SOURCE_H

#include "ds3d/common/impl/impl_dataloader.h"

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API ds3d::abiRefDataLoader* createLidarFileLoader();
DS3D_EXTERN_C_END

#endif  // _DS3D_DATALOADER_DEPTHSOURCE_DEPTH_COLOR_SOURCE_H