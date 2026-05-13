/**
 * SPDX-FileCopyrightText: Copyright (c) 2018 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

