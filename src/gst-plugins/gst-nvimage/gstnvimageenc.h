/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
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

