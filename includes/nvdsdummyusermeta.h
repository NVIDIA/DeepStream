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

#ifndef __GST_NVDSDUMMYUSER_META_H__
#define __GST_NVDSDUMMYUSER_META_H__

#include <nvdsmeta.h>

#define NVDS_DUMMY_BBOX_META \
    (nvds_get_user_meta_type((gchar *)"NVIDIA.DUMMY.BBOX.META"))

typedef enum _payload_type
{
    NVDS_PAYLOAD_TYPE_DUMMY_BBOX = NVDS_START_USER_META + 4096,
} payload_type;

typedef struct faceboxes
{
    float x;
    float y;
    float width;
    float height;
} faceboxes;

#endif  //__GST_NVDSDUMMYUSER_META_H__

