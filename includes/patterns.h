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

#ifndef _PATTERNS_H_
#define _PATTERNS_H_

#ifdef __cplusplus
extern "C" {
#endif

void gst_nv_video_test_src_cuda_init(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_cuda_free(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_cuda_prepare(GstNvVideoTestSrc *src, NvBufSurfaceParams *surf);

void gst_nv_video_test_src_smpte(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_mandelbrot(GstNvVideoTestSrc *src);
void gst_nv_video_test_src_gradient(GstNvVideoTestSrc *src);

#ifdef __cplusplus
}
#endif

#endif
