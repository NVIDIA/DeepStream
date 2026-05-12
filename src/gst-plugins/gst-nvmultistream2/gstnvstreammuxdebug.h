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
