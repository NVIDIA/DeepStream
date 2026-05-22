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
