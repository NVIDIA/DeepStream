/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
