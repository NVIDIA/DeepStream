/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _DS3D_COMMON_TYPE_ID__H
#define _DS3D_COMMON_TYPE_ID__H

#include <ds3d/common/common.h>

// type_id for basic datatype structures
#define DS3D_TYPEID_INT32_T 0x10002
#define DS3D_TYPEID_UINT32_T 0x10003
#define DS3D_TYPEID_FLOAT 0x10004
#define DS3D_TYPEID_INT8_T 0x10005
#define DS3D_TYPEID_UINT8_T 0x10006
#define DS3D_TYPEID_INT16_T 0x10007
#define DS3D_TYPEID_UINT16_T 0x10008
#define DS3D_TYPEID_DOUBLE 0x10009
#define DS3D_TYPEID_HALF_FLOAT 0x10010
#define DS3D_TYPEID_BOOL 0x10011
#define DS3D_TYPEID_INT64_T 0x10012

// type_id for common datatype structures
#define DS3D_TYPEID_TIMESTAMP 0x20002
#define DS3D_TYPEID_SHAPE 0x20003
#define DS3D_TYPEID_DEPTH_SCALE 0x20004
#define DS3D_TYPEID_INTRINSIC_PARM 0x20005
#define DS3D_TYPEID_EXTRINSIC_PARM 0x20006
#define DS3D_TYPEID_VIDEOBRIDGE2D_PARM 0x20007
#define DS3D_TYPEID_TRANSFORM_MATRIX 0x20008


#endif  // _DS3D_COMMON_TYPE_ID__H
