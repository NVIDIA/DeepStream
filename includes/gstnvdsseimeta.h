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

#ifndef __GST_NVDS_SEI_META_H__
#define __GST_NVDS_SEI_META_H__

#include <gst/gst.h>

#define GST_VIDEO_SEI_META_API_TYPE \
  (gst_video_sei_meta_api_get_type())
#define GST_VIDEO_SEI_META_INFO (gst_video_sei_meta_get_info())

#define GST_USER_SEI_META g_quark_from_static_string("GST.USER.SEI.META")

typedef struct _GstVideoSEIMeta {
  GstMeta meta;
  guint sei_metadata_type;
  guint sei_metadata_size;
  void *sei_metadata_ptr;
} GstVideoSEIMeta;

GType gst_video_sei_meta_api_get_type(void) asm ("gst_video_sei_meta_api_get_type");
const GstMetaInfo *gst_video_sei_meta_get_info(void) asm ("gst_video_sei_meta_get_info");

GstVideoSEIMeta *gst_buffer_add_video_sei_meta(GstBuffer *buffer);
GstVideoSEIMeta *gst_buffer_get_video_sei_meta(GstBuffer *buffer);

#endif /*__GST_NVDS_SEI_META_H__*/
