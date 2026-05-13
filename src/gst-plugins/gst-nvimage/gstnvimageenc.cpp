/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

/*
 * SAMPLE PIPELINES
 * 1. $ gst-launch-1.0 videotestsrc num-buffers=1 ! nvvideoconvert ! nvimageenc ! filesink location= temp.jpeg
 * 2. $ gst-launch-1.0 videotestsrc num-buffers=1 ! "video/x-raw, width=1920, height=1080" ! nvvideoconvert ! nvimageenc ! filesink location= temp.jpeg
 * 3. $ gst-launch-1.0 videotestsrc num-buffers=100 ! "video/x-raw, width=640, height=480" ! nvvideoconvert ! nvimageenc ! multifilesink location= multiple_images/image_%0000d.jpeg
 * 4. $ gst-launch-1.0 filesrc location= ~/sample_720p_mjpeg.mkv ! matroskademux ! jpegparse ! nvimagedec ! nvimageenc ! multifilesink location= multiple_images/image_%0000d.jpeg
 * 5. $ gst-launch-1.0 multifilesrc location= ~/multiple_images/sample_%0000d.jpeg ! jpegparse ! nvimagedec ! nvimageenc ! multifilesink location= multiple_images/image_%0000d.jpeg
 * 6. $ gst-launch-1.0 multifilesrc location= ~/multiple_images_different_resolution/sample_%0000d.jpeg ! jpegparse ! nvimagedec ! nvimageenc ! multifilesink location= multiple_images_different_resolution/image_%0000d.jpeg
 *
 * */

#include <stdlib.h>
#include <gst/base/base.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <sys/time.h>

#include "gstnvimage.h"
#include "gstnvimageenc.h"
#include "gstnvdsbufferpool.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvimage_enc_debug);
#define GST_CAT_DEFAULT gst_nvimage_enc_debug

#define parent_class gst_nvimage_enc_parent_class
G_DEFINE_TYPE (GstNvImageEnc, gst_nvimage_enc, GST_TYPE_VIDEO_ENCODER);

#define GST_CAPS_FEATURE_MEMORY_NVMM      "memory:NVMM"

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

static int dev_free(void *p) { return (int)cudaFree(p); }

enum
{
    PROP_0,
    PROP_GPU_ID,
    PROP_ENCODED_SIZE,
    PROP_QUALITY
};

static GstStaticPadTemplate gst_nvimage_enc_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
          (GST_CAPS_FEATURE_MEMORY_NVMM,
              "{ " "RGB }"))
    );

static GstStaticPadTemplate gst_nvimage_enc_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("image/jpeg")
    );

static void gst_nvimage_enc_finalize (GObject * object);
static gboolean gst_nvimage_enc_start (GstVideoEncoder * encoder);
static gboolean gst_nvimage_enc_stop (GstVideoEncoder * encoder);
static void gst_nvimage_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvimage_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean gst_nvimage_enc_propose_allocation (GstVideoEncoder * encoder,
    GstQuery * query);
static GstFlowReturn gst_nvimage_enc_handle_frame (GstVideoEncoder * encoder,
    GstVideoCodecFrame * frame);
static gboolean gst_nvimage_enc_set_format (GstVideoEncoder * encoder,
    GstVideoCodecState * state);

static void
gst_nvimage_enc_finalize (GObject * object)
{
  GstNvImageEnc *self = GST_NVIMAGE_ENC (object);

  if (self->input_state)
    gst_video_codec_state_unref (self->input_state);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_nvimage_enc_set_format (GstVideoEncoder * encoder,
    GstVideoCodecState * state)
{
  GstNvImageEnc *self = GST_NVIMAGE_ENC (encoder);

  GST_DEBUG_OBJECT (self, "Setting format: %" GST_PTR_FORMAT, state->caps);

  if (self->input_state)
    gst_video_codec_state_unref (self->input_state);
  self->input_state = gst_video_codec_state_ref (state);

  return TRUE;
}

static void
gst_nvimage_enc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
    GstNvImageEnc *self = GST_NVIMAGE_ENC (object);
    GST_OBJECT_LOCK (self);
    switch (prop_id)
    {
        case PROP_GPU_ID:
            g_value_set_uint (value, self->gpu_id);
            break;
        case PROP_ENCODED_SIZE:
            g_value_set_uint (value, MAX_ENCODED_SIZE);
            break;
        case PROP_QUALITY:
            g_value_set_uint (value, self->params.quality);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
    GST_OBJECT_UNLOCK (self);
}

static void
gst_nvimage_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstNvImageEnc *self = GST_NVIMAGE_ENC (object);

    GST_OBJECT_LOCK (self);

    switch (prop_id)
    {
        case PROP_GPU_ID:
            self->gpu_id = g_value_get_uint (value);
            break;
        case PROP_ENCODED_SIZE:
            self->encoded_size = g_value_get_uint (value);
            break;
        case PROP_QUALITY:
            self->params.quality = g_value_get_uint (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
    GST_OBJECT_UNLOCK (self);
}

static void
gst_nvimage_enc_init (GstNvImageEnc * self)
{
  //GstVideoEncoder *encoder = (GstVideoEncoder *) self;

  GST_PAD_SET_ACCEPT_TEMPLATE (GST_VIDEO_ENCODER_SINK_PAD (self));
  self->params.quality = 90;
  self->encoded_size = MAX_ENCODED_SIZE;
  self->gpu_id = 0;
}

static void gst_nvimage_enc_class_init (GstNvImageEncClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstVideoEncoderClass *video_encoder_class;

  gobject_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;
  video_encoder_class = (GstVideoEncoderClass *) klass;

  gst_element_class_add_static_pad_template (element_class, &gst_nvimage_enc_src_template);
  gst_element_class_add_static_pad_template (element_class, &gst_nvimage_enc_sink_template);

  gst_element_class_set_static_metadata (element_class,
      "Nvidia Image Encoder",
      "Codec/Encoder/JPEG IMAGE",
      "Encode to JPEG", "www.nvidia.com>");

  gobject_class->finalize = gst_nvimage_enc_finalize;
  gobject_class->set_property = gst_nvimage_enc_set_property;
  gobject_class->get_property = gst_nvimage_enc_get_property;

  video_encoder_class->start = GST_DEBUG_FUNCPTR (gst_nvimage_enc_start);
  video_encoder_class->stop  = GST_DEBUG_FUNCPTR (gst_nvimage_enc_stop);
  video_encoder_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_nvimage_enc_propose_allocation);
  video_encoder_class->set_format   =
      GST_DEBUG_FUNCPTR (gst_nvimage_enc_set_format);
  video_encoder_class->handle_frame =
      GST_DEBUG_FUNCPTR (gst_nvimage_enc_handle_frame);

  g_object_class_install_property (gobject_class, PROP_GPU_ID,
      g_param_spec_uint ("gpu-id",
          "gpu-id",
          "GPU ID to run encoding on", 0, G_MAXUINT,
          0, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENCODED_SIZE,
      g_param_spec_uint ("encoded-size",
          "Encoded size of the output buffer",
          "Encoded size of the output buffer",
          0, G_MAXUINT,
          MAX_ENCODED_SIZE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_QUALITY,
      g_param_spec_uint ("quality",
          "Encoder Quality",
          "Encoder Quality",
          0, 100, 90, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  GST_DEBUG_CATEGORY_INIT (gst_nvimage_enc_debug, "nvimageenc", 0,
      "Nvidia Image Encoder");
}

static GstFlowReturn
gst_nvimage_enc_handle_frame (GstVideoEncoder * encoder,
    GstVideoCodecFrame * frame)
{
  GstNvImageEnc *self = GST_NVIMAGE_ENC (encoder);
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *outbuf;
  static GstAllocationParams params = { (GstMemoryFlags)0, 0, 0, 3, };

  GstMapInfo imap = GST_MAP_INFO_INIT;
  GstMapInfo omap = GST_MAP_INFO_INIT;
  NvBufSurface *nvbuf_surf = NULL;

  if (!gst_buffer_map (frame->input_buffer, &imap, GST_MAP_READ))
  {
      GST_ERROR_OBJECT (encoder, "Failed to map input buffer");
      return GST_FLOW_ERROR;
  }

  outbuf = gst_buffer_new ();
  GstMemory *memory = gst_allocator_alloc(NULL, MAX_ENCODED_SIZE, &params);
  gst_memory_map (memory, &omap, GST_MAP_READWRITE);

  gst_buffer_copy_into (outbuf, frame->input_buffer, GST_BUFFER_COPY_METADATA, 0, -1);

  gst_buffer_append_memory (outbuf, memory);
  frame->output_buffer = outbuf;

  nvbuf_surf = (NvBufSurface *) imap.data;
  gst_buffer_unmap (frame->input_buffer, &imap);

  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props, self->params.dev));
  nvjpegEncoderParams_t encode_params;

  nvjpegChromaSubsampling_t subsampling;
  subsampling = NVJPEG_CSS_444;

  unsigned char * pBuffer = NULL;

  pBuffer = (unsigned char *) nvbuf_surf->surfaceList[0].dataPtr;

  nvjpegImage_t imgdesc;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  widths[0] = nvbuf_surf->surfaceList[0].width;
  heights[0] = nvbuf_surf->surfaceList[0].height;

  //RGB
  {
      imgdesc =
      {
          {
              pBuffer,
              pBuffer + nvbuf_surf->surfaceList[0].pitch*heights[0],
              pBuffer + nvbuf_surf->surfaceList[0].pitch*heights[0]*2,
              pBuffer + nvbuf_surf->surfaceList[0].pitch*heights[0]*3
          },
          {
              (unsigned int)nvbuf_surf->surfaceList[0].pitch,
              (unsigned int)nvbuf_surf->surfaceList[0].pitch,
              (unsigned int)nvbuf_surf->surfaceList[0].pitch,
              (unsigned int)nvbuf_surf->surfaceList[0].pitch
          }
      };
  }

  checkCudaErrors(nvjpegEncoderParamsCreate(self->params.nvjpeg_handle, &encode_params, NULL));

  // sample input parameters
  checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, self->params.quality, NULL));
  //checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, self->params.huf, NULL));
  checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, subsampling, NULL));

  nvjpegInputFormat_t input_format;
  input_format = NVJPEG_INPUT_RGBI;

  checkCudaErrors(nvjpegEncodeImage(self->params.nvjpeg_handle,
              self->params.nvjpeg_encoder_state,
              encode_params,
              &imgdesc,
              input_format,
              widths[0],
              heights[0],
              NULL));

  size_t length;
  checkCudaErrors(nvjpegEncodeRetrieveBitstream(
              self->params.nvjpeg_handle,
              self->params.nvjpeg_encoder_state,
              NULL,
              &length,
              NULL));

  if (length > self->encoded_size) {
    GST_ERROR_OBJECT (self, "Encoded size is greater than the output buffer size, use encoded-size property to set the max size, required: %ld, max set: %d", length, self->encoded_size);
    return GST_FLOW_ERROR;
  }

  checkCudaErrors(nvjpegEncodeRetrieveBitstream(
              self->params.nvjpeg_handle,
              self->params.nvjpeg_encoder_state,
              (unsigned char *) omap.data,
              &length,
              NULL));

  checkCudaErrors(nvjpegEncoderParamsDestroy(encode_params));

  GST_VIDEO_CODEC_FRAME_SET_SYNC_POINT (frame);

#if 0
  FILE *fp = NULL;
  fp = fopen ("image.jpeg", "wb");
  fwrite (omap.data, length, sizeof(char), fp);
  fclose(fp);
#endif

  gst_memory_unmap (memory, &omap);
  gst_memory_resize (memory, 0, length);
  omap.data = NULL;
  omap.size = 0;

  GstVideoCodecState *output;
  output =
      gst_video_encoder_set_output_state (GST_VIDEO_ENCODER (self),
            gst_pad_get_pad_template_caps (GST_VIDEO_ENCODER_SRC_PAD (self)),
            self->input_state);
  gst_video_codec_state_unref (output);

  ret = gst_video_encoder_finish_frame (GST_VIDEO_ENCODER (self), frame);

  return ret;
}

static gboolean
gst_nvimage_enc_propose_allocation (GstVideoEncoder * encoder, GstQuery * query)
{
  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);

  return GST_VIDEO_ENCODER_CLASS (parent_class)->propose_allocation (encoder,
      query);
}

static gboolean
gst_nvimage_enc_start (GstVideoEncoder * encoder)
{
  GstNvImageEnc *self = GST_NVIMAGE_ENC (encoder);

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT,
              &dev_allocator, &self->params.nvjpeg_handle));
  checkCudaErrors(nvjpegJpegStateCreate(self->params.nvjpeg_handle,
              &self->params.nvjpeg_state));
  checkCudaErrors(nvjpegEncoderStateCreate(self->params.nvjpeg_handle,
              &self->params.nvjpeg_encoder_state, NULL));

  return TRUE;
}

static gboolean
gst_nvimage_enc_stop (GstVideoEncoder * video_encoder)
{
  GstNvImageEnc *self = GST_NVIMAGE_ENC (video_encoder);

  GST_DEBUG_OBJECT (self, "Stopping");

  if (self->output_state)
  {
    gst_video_codec_state_unref (self->output_state);
    self->output_state = NULL;
  }

  if (self->input_state)
  {
    gst_video_codec_state_unref (self->input_state);
    self->input_state = NULL;
  }

  if (self->params.nvjpeg_state)
  {
      checkCudaErrors(nvjpegJpegStateDestroy(self->params.nvjpeg_state));
  }

  if (self->params.nvjpeg_handle)
  {
      checkCudaErrors(nvjpegDestroy(self->params.nvjpeg_handle));
  }

  GST_DEBUG_OBJECT (self, "Stopped");

  return TRUE;
}
