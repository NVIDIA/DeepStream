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
