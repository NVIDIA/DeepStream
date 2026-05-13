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

#ifndef __GST_NVDSCUSTOMUSER_META_H__
#define __GST_NVDSCUSTOMUSER_META_H__

#include <nvdsmeta.h>

#define NVDS_USER_CUSTOM_META \
    (nvds_get_user_meta_type((gchar*)"NVIDIA.USER.CUSTOM_META"))

typedef struct _NVDS_CUSTOM_PAYLOAD
{
   uint32_t payloadType;
   uint32_t payloadSize;
   uint8_t  *payload;
} NVDS_CUSTOM_PAYLOAD;

#endif  //__GST_NVDSCUSTOMUSER_META_H__
