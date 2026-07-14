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

#ifndef __GST_NVIMAGE_ENC_H__
#define __GST_NVIMAGE_ENC_H__

#include <gst/gst.h>
#include <gst/video/video.h>

#include "nvjpeg.h"

G_BEGIN_DECLS

#define MAX_ENCODED_SIZE 15*1024*1024

#define GST_TYPE_NVIMAGE_ENC \
  (gst_nvimage_enc_get_type())
#define GST_NVIMAGE_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVIMAGE_ENC,GstNvImageEnc))
#define GST_NVIMAGE_ENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVIMAGE_ENC,GstNvImageEncClass))
#define GST_IS_NVIMAGE_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVIMAGE_ENC))
#define GST_IS_NVIMAGE_ENC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVIMAGE_ENC))

typedef struct _GstNvImageEnc GstNvImageEnc;
typedef struct _GstNvImageEncClass GstNvImageEncClass;

typedef struct encode_params_t_ {
  int dev;

  nvjpegJpegState_t nvjpeg_state;
  nvjpegEncoderState_t nvjpeg_encoder_state;
  nvjpegHandle_t nvjpeg_handle;
  cudaStream_t stream;
  unsigned int quality;
  nvjpegOutputFormat_t fmt;
}encode_params_t;

struct _GstNvImageEnc
{
  GstVideoEncoder parent;

  /* < private > */
  GstVideoCodecState *input_state;
  GstVideoCodecState *output_state;
  guint gpu_id;
  GstVideoFormat format;
  encode_params_t params;
  unsigned int encoded_size;
};

struct _GstNvImageEncClass
{
  GstVideoEncoderClass parent_class;
};

GType gst_nvimage_enc_get_type (void);

G_END_DECLS

#endif /* __GST_NVIMAGE_ENC_H__ */

