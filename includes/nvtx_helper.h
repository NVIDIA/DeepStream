/**
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
/**
 * @file nvtx_helper.h
 * @brief Helper library for setting NVTX markers
 *
 */
#ifndef __NVTX_HELPER_H__
#define __NVTX_HELPER_H__
#ifdef __cplusplus
extern "C"
{
#endif
/**
 * Function definition for pushing/popping a NVTX range
 *
 * @param[in] context If specified, calls nvtxRangePushA().
 *  If not specified(NULL), nvtxRangePop() gets called.
 *
 */
void nvtx_helper_push_pop (char * context);
/**
 * Function definition for starting/stopping a NVTX range
 *
 * @param[in] context If specified, calls nvtxRangeStartA().
 *  If not specified (NULL), nvtxRangeEnd() gets called.
 * @param[in] id The unique ID used to correlate a pair of Start and End events.
 *
 */
void nvtx_helper_start_end (char * context, unsigned long *id);
#ifdef __cplusplus
}
#endif
#endif

