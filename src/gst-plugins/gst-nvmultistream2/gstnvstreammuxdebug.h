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

#ifndef _GST_NVSTREAMMUX_DEBUG_H_
#define _GST_NVSTREAMMUX_DEBUG_H_

#include <stdio.h>
#include "nvstreammux_debug.h"
#include <stdarg.h>
#include <gst/gstinfo.h>

#if 1
#define LOGD(...)
#else
#define LOGD(fmt, ...) printf("[DEBUG %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)
#endif

#define LOGV(fmt, ...) printf("[VERBOSE %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)
#define LOGE(fmt, ...) printf("[ERROR %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)

#endif /**< _GST_NVSTREAMMUX_DEBUG_H_ */
