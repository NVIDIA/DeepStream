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
 * 1. gst-launch-1.0 filesrc location = ~/sample_720p_mjpeg.mkv ! matroskademux ! jpegparse ! nvimagedec ! nveglglessink sync=0
 * 2. gst-launch-1.0 filesrc location = ~/sample_720p.jpeg ! jpegparse ! nvimagedec ! nveglglessink sync=1
 * 3. gst-launch-1.0 multifilesrc location = ~/multiple_images/sample_%0000d.jpeg ! jpegparse ! nvimagedec ! nveglglessink sync=0
 * 4. gst-launch-1.0 multifilesrc location = ~/multiple_images_different_resolution/sample_%0000d.jpeg ! jpegparse ! nvimagedec ! nveglglessink sync=1
 * 5. gst-launch-1.0 multifilesrc location= ~/sample_720p.jpeg loop=1 ! jpegparse ! nvimagedec needs-pool=0 ! fpsdisplaysink video-sink=fakesink silent=0 sync=0 -v
*/

#include <stdlib.h>
#include <gst/base/base.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

#include "gstnvimage.h"
#include "gstnvimagedec.h"
#include "gstnvdsbufferpool.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

enum
{
    PROP_0,
    PROP_GPU_ID,
    PROP_NEEDS_BUFFER_POOL
};

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

static int dev_free(void *p) { return (int)cudaFree(p); }

static int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }

static int host_free(void* p) { return (int)cudaFreeHost(p); }


#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

GST_DEBUG_CATEGORY_STATIC (gst_nvimage_dec_debug);
#define GST_CAT_DEFAULT gst_nvimage_dec_debug

static gboolean gst_nvimage_dec_start (GstVideoDecoder * decoder);
static gboolean gst_nvimage_dec_stop (GstVideoDecoder * decoder);
static void gst_nvimage_dec_finalize (GObject * object);
static gboolean gst_nvimage_dec_set_format (GstVideoDecoder * decoder,
    GstVideoCodecState * state);
static GstFlowReturn gst_nvimage_dec_negotiate (GstNvImageDec * self,
    const unsigned char *data, size_t length);
static GstFlowReturn gst_nvimage_dec_handle_frame (GstVideoDecoder * decoder,
    GstVideoCodecFrame * frame);
static gboolean gst_nvimage_dec_decide_allocation (GstVideoDecoder * decoder,
    GstQuery * query);
static void gst_nvimage_dec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvimage_dec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstStaticPadTemplate gst_nvimage_dec_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("image/jpeg")
    );

static GstStaticPadTemplate gst_nvimage_dec_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
          (GST_CAPS_FEATURE_MEMORY_NVMM,
              "{ " "RGB }"))
    );

#define parent_class gst_nvimage_dec_parent_class
G_DEFINE_TYPE (GstNvImageDec, gst_nvimage_dec, GST_TYPE_VIDEO_DECODER);

static void
gst_nvimage_dec_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
    GstNvImageDec *self = GST_NVIMAGE_DEC (object);
    switch (prop_id)
    {
        case PROP_GPU_ID:
            g_value_set_uint (value, self->gpu_id);
            break;
        case PROP_NEEDS_BUFFER_POOL:
            g_value_set_boolean (value, self->needs_pool);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static void
gst_nvimage_dec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstNvImageDec *self = GST_NVIMAGE_DEC (object);

    switch (prop_id)
    {
        case PROP_GPU_ID:
            self->gpu_id = g_value_get_uint (value);
            break;
        case PROP_NEEDS_BUFFER_POOL:
            self->needs_pool = g_value_get_boolean (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static void gst_nvimage_dec_class_init (GstNvImageDecClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstVideoDecoderClass *video_decoder_class;

  gobject_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;
  video_decoder_class = (GstVideoDecoderClass *) klass;

  gst_element_class_add_static_pad_template (element_class, &gst_nvimage_dec_src_template);
  gst_element_class_add_static_pad_template (element_class, &gst_nvimage_dec_sink_template);

  gst_element_class_set_static_metadata (element_class,
      "Nvidia Image Decoder",
      "Codec/Decoder/JPEG IMAGE",
      "Decode JPEG streams", "www.nvidia.com>");

  gobject_class->set_property = gst_nvimage_dec_set_property;
  gobject_class->get_property = gst_nvimage_dec_get_property;

  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvimage_dec_finalize);

  video_decoder_class->start = GST_DEBUG_FUNCPTR (gst_nvimage_dec_start);
  video_decoder_class->stop  = GST_DEBUG_FUNCPTR (gst_nvimage_dec_stop);
  //video_decoder_class->parse = GST_DEBUG_FUNCPTR (gst_nvimage_dec_parse);
  video_decoder_class->set_format   =
      GST_DEBUG_FUNCPTR (gst_nvimage_dec_set_format);
  video_decoder_class->handle_frame =
      GST_DEBUG_FUNCPTR (gst_nvimage_dec_handle_frame);
  video_decoder_class->decide_allocation = gst_nvimage_dec_decide_allocation;

  g_object_class_install_property (gobject_class, PROP_GPU_ID,
      g_param_spec_uint ("gpu-id",
          "gpu-id",
          "GPU ID to run decoding on", 0, G_MAXUINT,
          0, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NEEDS_BUFFER_POOL,
          g_param_spec_boolean ("needs-pool",
              "Allocates Buffer Pool",
              "Allocates Buffer Pool",
              TRUE,
              (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  GST_DEBUG_CATEGORY_INIT (gst_nvimage_dec_debug, "nvimagedec", 0,
      "Nvidia Image Decoder");
}

static void
gst_nvimage_dec_init (GstNvImageDec * self)
{
  self->needs_pool = TRUE;
  self->nvimagedec_init = FALSE;
  self->pool = FALSE;
  GST_PAD_SET_ACCEPT_TEMPLATE (GST_VIDEO_DECODER_SINK_PAD (self));
}

static gboolean
gst_nvimage_dec_start (GstVideoDecoder * decoder)
{
  nvjpegStatus_t status = NVJPEG_STATUS_NOT_INITIALIZED;
  GstNvImageDec *self = GST_NVIMAGE_DEC (decoder);

  self->width = self->height = 0;
  self->old_width = self->old_height = 0;
  self->params.dev = self->gpu_id;
  self->params.fmt = NVJPEG_OUTPUT_RGBI;

  if (self->nvimagedec_init == FALSE)
  {
      cudaSetDevice(self->params.dev);
      cudaDeviceProp props;
      checkCudaErrors(cudaGetDeviceProperties(&props, self->params.dev));

      printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
              self->params.dev, props.name, props.multiProcessorCount,
              props.maxThreadsPerMultiProcessor, props.major, props.minor,
              props.ECCEnabled ? "on" : "off");

      checkCudaErrors(
              cudaStreamCreateWithFlags(&self->params.stream, cudaStreamNonBlocking));

      nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
      nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};
      int flags = 0;

      status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE,
              &dev_allocator,
              &pinned_allocator,
              flags,
              &self->params.nvjpeg_handle);


      if(status == NVJPEG_STATUS_ARCH_MISMATCH)
      {
          GST_DEBUG_OBJECT (self, "Hardware Decoder not supported. Falling back to default backend\n");
          checkCudaErrors(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT,
                      &dev_allocator,
                      &pinned_allocator,
                      flags,
                      &self->params.nvjpeg_handle));
      }

      checkCudaErrors(nvjpegJpegStateCreate(self->params.nvjpeg_handle,
                  &self->params.nvjpeg_state));

      status = nvjpegJpegStreamCreate(self->params.nvjpeg_handle, &(self->params.nvjpeg_stream));
      if (status != NVJPEG_STATUS_SUCCESS)
      {
          GST_ERROR_OBJECT (self, "nvjpegJpegStreamCreate failed. ErrorCode: (%d)\n", status);
      }

      self->nvimagedec_init = TRUE;
  }

  GST_DEBUG_OBJECT (self, "Starting");

  return TRUE;
}

static void
gst_nvimage_dec_finalize (GObject * object)
{
  GstNvImageDec *self = GST_NVIMAGE_DEC (object);

  if (self->params.nvjpeg_stream)
  {
    checkCudaErrors(nvjpegJpegStreamDestroy(self->params.nvjpeg_stream));
  }

  if (self->params.nvjpeg_state)
  {
      checkCudaErrors(nvjpegJpegStateDestroy(self->params.nvjpeg_state));
  }

  if (self->params.nvjpeg_handle)
  {
      checkCudaErrors(nvjpegDestroy(self->params.nvjpeg_handle));
  }

  if (self->params.stream)
  {
    cudaStreamDestroy (self->params.stream);
    self->params.stream = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_nvimage_dec_stop (GstVideoDecoder * video_decoder)
{
  GstNvImageDec *self = GST_NVIMAGE_DEC (video_decoder);

  GST_DEBUG_OBJECT (self, "Stopping");

  if (self->output_state)
  {
    gst_video_codec_state_unref (self->output_state);
    self->output_state = NULL;
  }

  if (self->pool)
  {
      if (gst_buffer_pool_is_active (self->pool))
      {
        gst_buffer_pool_set_active (self->pool, FALSE);
      }
      gst_object_unref (self->pool);
      self->pool = NULL;
  }

  if (self->input_state)
  {
    gst_video_codec_state_unref (self->input_state);
    self->input_state = NULL;
  }

  GST_DEBUG_OBJECT (self, "Stopped");

  return TRUE;
}

static gboolean
gst_nvimage_dec_set_format (GstVideoDecoder * decoder,
    GstVideoCodecState * state)
{
  GstNvImageDec *self = GST_NVIMAGE_DEC (decoder);
  GstVideoInfo *info = &state->info;

  GST_DEBUG_OBJECT (self, "Setting format: %" GST_PTR_FORMAT, state->caps);

  if (self->input_state)
    gst_video_codec_state_unref (self->input_state);
  self->input_state = gst_video_codec_state_ref (state);

  self->width = GST_VIDEO_INFO_WIDTH (info);
  self->height = GST_VIDEO_INFO_HEIGHT (info);

  return TRUE;
}

static GstFlowReturn
gst_nvimage_dec_negotiate (GstNvImageDec * self, const unsigned char *data, size_t length)
{
  /* Check if output state changed */
  if (self->output_state)
  {
      GstVideoInfo *info = &self->output_state->info;

      if (self->width == GST_VIDEO_INFO_WIDTH (info) &&
          self->height == GST_VIDEO_INFO_HEIGHT (info) &&
          GST_VIDEO_INFO_FORMAT (info) == self->format)
      {
          return GST_FLOW_OK;
      }
      gst_video_codec_state_unref (self->output_state);
  }

  self->format = GST_VIDEO_FORMAT_RGB;

  if (nvjpegJpegStreamParseHeader(self->params.nvjpeg_handle,
              data, length, self->params.nvjpeg_stream) == NVJPEG_STATUS_SUCCESS)
  {
      nvjpegStatus_t status = nvjpegJpegStreamGetComponentDimensions(self->params.nvjpeg_stream,
              0, (unsigned int*)(&(self->width)),(unsigned int*)(&(self->height)));
      if (status != NVJPEG_STATUS_SUCCESS)
      {
          GST_ERROR_OBJECT (self, "nvjpegJpegStreamGetComponentDimensions Failed, ErrorCode: (%d)\n", status);
      }
  }
  else
  {
      GST_ERROR_OBJECT (self, "nvjpegJpegStreamParseHeader Failed");
  }

  self->output_state =
      gst_video_decoder_set_output_state (GST_VIDEO_DECODER (self),
              self->format, self->width, self->height, self->input_state);

  if (self->output_state->caps)
      gst_caps_unref (self->output_state->caps);

  self->output_state->caps =
      gst_video_info_to_caps (&self->output_state->info);
  GstCapsFeatures *features = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (self->output_state->caps, 0, features);

  GST_DEBUG_OBJECT (self, "Have image of size %dx%d ", self->width, self->height);

  if (!gst_video_decoder_negotiate (GST_VIDEO_DECODER (self)))
      return GST_FLOW_NOT_NEGOTIATED;

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_nvimage_dec_handle_frame (GstVideoDecoder * decoder,
        GstVideoCodecFrame * frame)
{
  GstNvImageDec *self = GST_NVIMAGE_DEC (decoder);
  GstFlowReturn ret = GST_FLOW_OK;
  GstMapInfo imap = GST_MAP_INFO_INIT;
  GstMapInfo omap = GST_MAP_INFO_INIT;
  nvjpegImage_t out0 = {{0}};
  nvjpegStatus_t err;
  NvBufSurfaceCreateParams cparams;
  NvBufSurface *surf = NULL;

#ifdef PERF_MEASUREMENT
  START_MEASUREMENT
#endif

  GST_DEBUG_OBJECT (self, "Handling frame");

  if (!gst_buffer_map (frame->input_buffer, &imap, GST_MAP_READ))
  {
      GST_ERROR_OBJECT (decoder, "Failed to map input buffer");
      return GST_FLOW_ERROR;
  }

  if (GST_FLOW_OK != gst_nvimage_dec_negotiate(self, imap.data, imap.size))
  {
      GST_ERROR_OBJECT (decoder, "Failed to negotiate");
      return GST_FLOW_ERROR;
  }

  if (self->needs_pool == FALSE)
  {
    ret = gst_video_decoder_allocate_output_frame (decoder, frame);
    if (ret != GST_FLOW_OK)
    {
        GST_ERROR_OBJECT (decoder, "Failed to allocate output frame");
        return GST_FLOW_ERROR;
    }

    cparams.gpuId = self->params.dev;
    cparams.width = self->width;
    cparams.height = self->height;
    cparams.size = 0;
    cparams.colorFormat = NVBUF_COLOR_FORMAT_RGB;
    cparams.layout = NVBUF_LAYOUT_PITCH;
    cparams.memType = NVBUF_MEM_CUDA_DEVICE;
    NvBufSurfaceCreate (&surf, 1, &cparams);
    out0.channel[0] = (unsigned char*)surf->surfaceList[0].dataPtr;
    out0.pitch[0] = surf->surfaceList[0].pitch;

    gst_buffer_map(frame->output_buffer, &omap, GST_MAP_WRITE);
    //omap.data = (unsigned char*) surf;
    memcpy (omap.data, surf, sizeof (NvBufSurface));
    gst_buffer_set_size (frame->output_buffer, sizeof(NvBufSurface));

    gpointer tsurf = malloc(omap.size);
    memcpy(tsurf, omap.data, omap.size);
    GQuark quark = g_quark_from_static_string("nvbufsurf");
    gst_mini_object_set_qdata(GST_MINI_OBJECT(frame->output_buffer),
            quark,
            tsurf,
            (GDestroyNotify)NvBufSurfaceDestroy);

  }
  else if (self->needs_pool == TRUE)
  {

    ret = gst_buffer_pool_acquire_buffer (self->pool,
                                    &frame->output_buffer, NULL);
 
    if (!gst_buffer_map (frame->output_buffer, &omap, GST_MAP_READ))
    {
        GST_ERROR_OBJECT (decoder, "Failed to map output buffer");
        return GST_FLOW_ERROR;
    }

    surf = (NvBufSurface *) omap.data;
    surf->numFilled = 1;
    out0.channel[0] = (unsigned char*)surf->surfaceList[0].dataPtr;
    out0.pitch[0] = surf->surfaceList[0].pitch;
  }

  if ((err = nvjpegDecode(
                  self->params.nvjpeg_handle,
                  self->params.nvjpeg_state,
                  (const unsigned char *)imap.data,
                  imap.size,
                  self->params.fmt,
                  &out0,
                  self->params.stream)) != NVJPEG_STATUS_SUCCESS)
  {
      GST_ERROR_OBJECT (self, "failed to decode image \n");
  }
  else
  {
      GST_DEBUG_OBJECT (self, "SUCCESSFULLY DECODED IMAGE\n");
  }
  cudaError_t e = cudaGetLastError();                                     
  if(e != cudaSuccess)
  {
    g_print ("Cuda failure: %s", cudaGetErrorString(e));
  }
  cudaStreamSynchronize(self->params.stream);

#define DUMP_OUTPUT 0
#if DUMP_OUTPUT
  char file_name[20] = {0};
  sprintf (file_name,"dump_%dx%d.rgb",surf->surfaceList[0].width,surf->surfaceList[0].height);
  FILE *fp = fopen (file_name, "wb");
  fwrite (out0.channel[0], 1, surf->surfaceList[0].width*surf->surfaceList[0].height*3, fp);
  //fwrite (imap.data, imap.size, sizeof (char), fp);
#endif

  gst_buffer_copy_into (frame->output_buffer, frame->input_buffer, GST_BUFFER_COPY_METADATA, 0, -1);
  gst_buffer_unmap (frame->input_buffer, &imap);
  gst_buffer_unmap (frame->output_buffer, &omap);

#ifdef PERF_MEASUREMENT
  STOP_MEASUREMENT;
  std::cout << "HANDLE FRAME TIME: "  <<  std::dec << totalTimeMicro << " us context = " <<  self  << std::endl;
#endif
  ret = gst_video_decoder_finish_frame (decoder, frame);

  return ret;
}

static gboolean
gst_nvimage_dec_decide_allocation (GstVideoDecoder * decoder, GstQuery * query)
{
  GstCaps *outcaps;
  GstStructure *config;
  GstBufferPool *pool = NULL;
  GstCapsFeatures *ift = NULL;
  GstVideoInfo info;
  GstNvImageDec *self = GST_NVIMAGE_DEC (decoder);

  if (!GST_VIDEO_DECODER_CLASS (parent_class)->decide_allocation (decoder, query))
  {
      return FALSE;
  }

  if (self->needs_pool == FALSE)
      return TRUE;

  ift = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_NVMM, NULL);
  gst_caps_features_free (ift);

  gst_query_parse_allocation (query, &outcaps, NULL);
  if (outcaps == NULL)
  {
      GST_ERROR_OBJECT (decoder, "failed to query");
      return FALSE;
  }

  GST_DEBUG_OBJECT (self, "%s outcaps from query = %s\n", __func__,
          gst_caps_to_string(outcaps));

  if (!gst_video_info_from_caps (&info, outcaps))
  {
      GST_ERROR_OBJECT (decoder, "invalid output caps");
      return FALSE;
  }

  /* CHECK: Is this condition really needed */
  if ((self->pool) &&
      (self->width == self->old_width) &&
      (self->height == self->old_height))
  {
      return TRUE;
  }

  if (self->pool && (true == gst_buffer_pool_is_active (self->pool)))
  {
      gst_buffer_pool_set_active (self->pool, FALSE);
      gst_object_unref(self->pool);
      self->pool = NULL;
  }

  GST_DEBUG_OBJECT (self, "Creating buffer pool width %d height %d\n",
          self->width, self->height);
  pool = gst_nvds_buffer_pool_new ();

  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface),
          4, 4);

  gst_structure_set (config,
          "memtype", G_TYPE_UINT, NVBUF_MEM_CUDA_DEVICE,
          "gpu-id", G_TYPE_UINT, 0,
          "batch-size", G_TYPE_UINT, 1,
          "clear-chroma", G_TYPE_BOOLEAN, 1,
          "bl-output", G_TYPE_UINT, 0,
          "contiguous-alloc", G_TYPE_BOOLEAN, 1, NULL);

  if (!gst_buffer_pool_set_config (pool, config))
  {
      GST_ERROR_OBJECT (decoder, "buffer pool set config failed");
      return FALSE;
  }

  self->pool = (GstBufferPool *) gst_object_ref (pool);
 
  gboolean is_active = gst_buffer_pool_set_active (self->pool, TRUE);
  if (!is_active)
  {
      GST_WARNING (" Failed to allocate the buffers inside the output pool");
      return FALSE;
  }
  else
  {
      GST_DEBUG (" Output buffer pool successfully created");
  }

  self->old_width = self->width;
  self->old_height = self->height;

  return TRUE;
}
