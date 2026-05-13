/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */


#ifndef __GST_NVIMAGE_DEC_H__
#define __GST_NVIMAGE_DEC_H__

#include <gst/gst.h>
#include <gst/video/video.h>

#include "nvjpeg.h"

G_BEGIN_DECLS

#define GST_TYPE_NVIMAGE_DEC \
  (gst_nvimage_dec_get_type())
#define GST_NVIMAGE_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVIMAGE_DEC,GstNvImageDec))
#define GST_NVIMAGE_DEC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVIMAGE_DEC,GstNvImageDecClass))
#define GST_IS_NVIMAGE_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVIMAGE_DEC))
#define GST_IS_NVIMAGE_DEC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVIMAGE_DEC))

typedef struct _GstNvImageDec GstNvImageDec;
typedef struct _GstNvImageDecClass GstNvImageDecClass;

#define START_MEASUREMENT \
    timeval start, end; \
    gettimeofday (&start, NULL);
#define STOP_MEASUREMENT \
    gettimeofday (&end, NULL); \
    long seconds = end.tv_sec - start.tv_sec; \
    long microseconds = end.tv_usec - start.tv_usec; \
    long totalTimeMicro = seconds * 1000000 + microseconds;

typedef struct decode_params_t_ {
  int dev;

  nvjpegJpegState_t nvjpeg_state;
  nvjpegHandle_t nvjpeg_handle;
  nvjpegJpegStream_t nvjpeg_stream;
  cudaStream_t stream;

  nvjpegOutputFormat_t fmt;
}decode_params_t;

struct _GstNvImageDec
{
  GstVideoDecoder parent;

  /* < private > */
  GstVideoCodecState *input_state;
  GstVideoCodecState *output_state;
  gint width;
  gint height;
  gint old_width;
  gint old_height;
  gboolean needs_pool;
  guint gpu_id;
  GstVideoFormat format;
  decode_params_t params;
  GstBufferPool *pool;
  gboolean nvimagedec_init;
};

struct _GstNvImageDecClass
{
  GstVideoDecoderClass parent_class;
};

GType gst_nvimage_dec_get_type (void);

G_END_DECLS

#endif /* __GST_NVIMAGE_DEC_H__ */
