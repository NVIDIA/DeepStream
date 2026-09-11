/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
  * Relation between width, height, PAR(Pixel Aspect Ratio), DAR(Display Aspect Ration):
  *
  *                  dar_n   par_d
  * width = height * ----- * -----
  *                  dar_d   par_n
  *
  *                  dar_d   par_n
  * height = width * ----- * -----
  *                  dar_n   par_d
  *
  * par_n    height   dar_n
  * ----- =  ------ * -----
  * par_d    width    dar_d
  *
  * dar_n   width    par_n
  * ----- = ------ * -----
  * dar_d   height   par_d
  */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "gstnvvideoconvert.h"
#include "gst-nvcommon.h"
#include "nvc_helper.h"
#ifndef NEW_METADATA
#if defined(__aarch64__)
#include "gstnvivameta_api.h"
#endif

#include "gstnvdsmeta.h"
#endif

#if defined(__aarch64__)
#include "nvtx_helper.h"
#endif

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define NVBUF_MAGIC_NUM 0x70807580

GST_DEBUG_CATEGORY (gst_nvvideoconvert_debug);
#define GST_CAT_DEFAULT gst_nvvideoconvert_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

/* Filter properties */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_GPU_DEVICE_ID,
  PROP_FLIP_METHOD,
  PROP_NUM_OUT_BUFS,
  PROP_INTERPOLATION_METHOD,
  PROP_SRC_CROP,
  PROP_DST_CROP,
  PROP_COMPUTE_HW,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_ENABLE_BLOCKLINEAR_OUTPUT,
  PROP_ENABLE_ODD_CROP,
  PROP_COPY_HW,
  PROP_ENABLE_CONTIGUOUS_BUFFERS,
  PROP_DISABLE_PASSTHROUGH
};


#define PROP_FLIP_METHOD_DEFAULT GST_VIDEO_NVFLIP_METHOD_IDENTITY

#define GST_TYPE_VIDEO_NVFLIP_METHOD (gst_video_nvflip_method_get_type())

#define GST_TYPE_COPY_HW (gst_copy_hw_get_type())

static const GEnumValue video_nvflip_methods[] = {
  {GST_VIDEO_NVFLIP_METHOD_IDENTITY, "Identity (no rotation)", "none"},
  {GST_VIDEO_NVFLIP_METHOD_90L, "Rotate counter-clockwise 90 degrees",
      "counterclockwise"},
  {GST_VIDEO_NVFLIP_METHOD_180, "Rotate 180 degrees", "rotate-180"},
  {GST_VIDEO_NVFLIP_METHOD_90R, "Rotate clockwise 90 degrees", "clockwise"},
  {GST_VIDEO_NVFLIP_METHOD_HORIZ, "Flip horizontally", "horizontal-flip"},
  {GST_VIDEO_NVFLIP_METHOD_INVTRANS,
      "Flip across upper right/lower left diagonal", "upper-right-diagonal"},
  {GST_VIDEO_NVFLIP_METHOD_VERT, "Flip vertically", "vertical-flip"},
  {GST_VIDEO_NVFLIP_METHOD_TRANS,
      "Flip across upper left/lower right diagonal", "upper-left-diagonal"},
  {0, NULL, NULL},
};

static GType
gst_video_nvflip_method_get_type (void)
{
  static GType video_nvflip_method_type = 0;

  if (!video_nvflip_method_type) {
    video_nvflip_method_type = g_enum_register_static ("GstNvDsVideoFlipMethod",
        video_nvflip_methods);
  }
  return video_nvflip_method_type;
}

static GType
gst_copy_hw_get_type (void)
{
  static gsize copy_hw_type = 0;
  static const GEnumValue copy_hw[] = {
    {NvBufSurfTransformCompute_GPU, "GPU", "GPU"},
#if defined(__aarch64__)
    {NvBufSurfTransformCompute_VIC, "VIC", "VIC"},
#endif
    {0, NULL, NULL},
  };

  if(g_once_init_enter (&copy_hw_type)) {
    GType tmp = g_enum_register_static ("GstNvCopyHWType",
       copy_hw);
    g_once_init_leave(&copy_hw_type, tmp);
  }

  return (GType) copy_hw_type;
}

static void update_meta (NvDsBatchMeta *batch_meta, uint32_t icnt, Gstnvvideoconvert *space, gfloat from_width, gfloat from_height, gfloat to_width, gfloat to_height);
static void flip_object(NvDsObjectMeta *object_meta, gint flip_method, gfloat from_width, gfloat from_height);
static void scale_object(NvOSD_RectParams* rect_params, NvOSD_TextParams* text_params, Gstnvvideoconvert * space, gfloat scale_factor_width, gfloat scale_factor_height);
void ScaleRectParams(NvOSD_RectParams* rect_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top);
void ScaleTextParams(NvOSD_TextParams* text_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top);
void ScaleLineParams(NvOSD_LineParams* line_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top);
void ScaleCircleParams(NvOSD_CircleParams* circle_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top);
void ScaleArrowParams(NvOSD_ArrowParams* arrow_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top);
static NvDsBatchMeta * gst_buffer_get_nvds_batch_meta_int (GstBuffer *buffer);
static GQuark _dsmeta_quark = 0;

enum PLANAR_FORMAT_ORDER
{
  PLANAR_RGB,
  PLANAR_BGR
};

#define GST_TYPE_RGB_PLANE_ORDER (gst_rgb_plane_order_get_type())
static const GEnumValue rgb_plane_order[] = {
  {PLANAR_RGB, "PLANAR_RGB", "RGB"},
  {PLANAR_BGR, "PLANAR_BGR", "BGR"},
  {0, NULL, NULL},
};

static GType
gst_rgb_plane_order_get_type (void)
{
  static GType rgb_plane_order_type = 0;

  if (!rgb_plane_order_type) {
    rgb_plane_order_type = g_enum_register_static ("GstRGBPlaneOrder",
        rgb_plane_order);
  }
  return rgb_plane_order_type;
}

enum FORMAT_PRECISION
{
  P_UINT8,
  P_FLOAT32,
  //FLOAT16
};

#define GST_TYPE_FORMAT_PRECISION (gst_format_precision_get_type())
static const GEnumValue format_precision[] = {
  {P_UINT8, "P_UINT8", "UINT8"},
  {P_FLOAT32, "P_FLOAT32", "FLOAT32"},
  {0, NULL, NULL},
};

static GType
gst_format_precision_get_type (void)
{
  static GType format_precision_type = 0;

  if (!format_precision_type) {
    format_precision_type = g_enum_register_static ("GstFormatPrecision",
        format_precision);
  }
  return format_precision_type;
}

/* For Jetson platforms*/
#if defined(__aarch64__) && !defined(AARCH64_IS_SBSA)
static GstStaticPadTemplate gst_nvvideoconvert_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ "
            "I420,  NV12, P010_10LE, I420_12LE, BGRx, RGBA, GRAY8, GRAY16_LE, YUY2, UYVY, YVYU, Y42B, RGB, BGR, BGR10A2_LE, UYVP, BGRA64_LE }") ";"
        GST_VIDEO_CAPS_MAKE ("{ "
            "I420, NV12, P010_10LE, BGRx, RGBA, GRAY8, GRAY16_LE, YUY2, UYVY, YVYU, Y42B, RGB, BGR, BGR10A2_LE, UYVP, BGRA64_LE }")));

/* Output capabilities. */
static GstStaticPadTemplate gst_nvvideoconvert_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "I420, NV12, P010_10LE, I420_12LE, BGRx, RGBA, GRAY8, GRAY16_LE, YUY2, UYVY, YVYU, Y42B, BGR, RGB, BGR10A2_LE, UYVP, BGRA64_LE }")
        ";" GST_VIDEO_CAPS_MAKE ("{ "
            "I420, NV12, P010_10LE, BGRx, RGBA, GRAY8, GRAY16_LE, YUY2, UYVY, YVYU, Y42B, BGR, RGB, BGR10A2_LE, UYVP, BGRA64_LE }")));
#else
/*For x86 and arm-sbsa platforms */

/* Gstreamer have not defined video format for P012_12LE or NV12 12 bit in v1.16.
  Hence, Currently WAR applied using GST_VIDEO_FORMAT_I420_12LE format.
  Once we migrate to newer version of gstreamer, we should update I420_12LE
  with corresponding format everywhere */
static GstStaticPadTemplate gst_nvvideoconvert_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "I420, NV12, P010_10LE, I420_12LE, BGRx, RGBA, Y444, Y444_10LE, Y444_12LE, GRAY8, GRAY16_LE, GBR, RGB, BGR, BGR10A2_LE, RGB10A2_LE, UYVP, UYVY, BGRA64_LE }") ";"
        GST_VIDEO_CAPS_MAKE ("{ "
            "I420, NV12, P010_10LE, BGRx, RGBA, Y444, GRAY8, GRAY16_LE, GBR, RGB, BGR, BGR10A2_LE, RGB10A2_LE, UYVP, UYVY, BGRA64_LE }")));

/* Output capabilities. */
static GstStaticPadTemplate gst_nvvideoconvert_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "I420, NV12, P010_10LE, I420_12LE, BGRx, RGBA, Y444, Y444_10LE, Y444_12LE, GRAY8, GRAY16_LE, GBR, RGB, BGR, BGR10A2_LE, RGB10A2_LE, UYVP, UYVY, BGRA64_LE }") ";"
        GST_VIDEO_CAPS_MAKE ("{ "
            "I420, NV12, P010_10LE, BGRx, RGBA, Y444, GRAY8, GRAY16_LE, GBR, RGB, BGR, BGR10A2_LE, RGB10A2_LE, UYVP, UYVY, BGRA64_LE }")));
#endif

static GstElementClass *gparent_class = NULL;

#define gst_nvvideoconvert_parent_class parent_class
G_DEFINE_TYPE (Gstnvvideoconvert, gst_nvvideoconvert, GST_TYPE_BASE_TRANSFORM);

  /* internal methods */
static void gst_nvvideoconvert_init_params (Gstnvvideoconvert * filter);
static gboolean gst_nvvideoconvert_get_pix_fmt (GstVideoInfo * info,
    NvBufSurfaceColorFormat * pix_fmt, gint * isurf_count, gboolean rgb,
    guint precision);
static GstCaps *gst_nvvideoconvert_caps_remove_format_info (GstCaps * caps);
static GstCaps *gst_nvvideoconvert_caps_remove_batchsize(GstCaps * caps);

static void gst_nvvideoconvert_free_buf (Gstnvvideoconvert * filter);

  /* base transform vmethods */
static gboolean gst_nvvideoconvert_start (GstBaseTransform * btrans);
static gboolean gst_nvvideoconvert_stop (GstBaseTransform * btrans);
static void gst_nvvideoconvert_finalize (GObject * object);
static GstStateChangeReturn gst_nvvideoconvert_change_state (GstElement *
    element, GstStateChange transition);
static GstFlowReturn gst_nvvideoconvert_transform (GstBaseTransform * btrans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean gst_nvvideoconvert_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static GstCaps *gst_nvvideoconvert_transform_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static gboolean gst_nvvideoconvert_accept_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps);
static gboolean gst_nvvideoconvert_transform_size (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize);
static gboolean gst_nvvideoconvert_get_unit_size (GstBaseTransform * btrans,
    GstCaps * caps, gsize * size);
static GstCaps *gst_nvvideoconvert_fixate_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_nvvideoconvert_decide_allocation (GstBaseTransform * btrans,
    GstQuery * query);

static void gst_nvvideoconvert_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvvideoconvert_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static guint get_output_frame_size (Gstnvvideoconvert * space);
static gboolean block_linear_layout_check(Gstnvvideoconvert *space);

#ifndef NEW_METADATA
static GQuark _ivameta_quark = 0;
static GQuark _dsmeta_quark = 0;
#endif

static gboolean
gst_nvvideoconvert_copy_metadata (GstBaseTransform * trans,
                                  GstBuffer * inbuf,
                                  GstBuffer * outbuf)
{
  if (!gst_buffer_copy_into (outbuf, inbuf, GST_BUFFER_COPY_META  |
                                            GST_BUFFER_COPY_FLAGS |
                                            GST_BUFFER_COPY_TIMESTAMPS, 0, -1)) {
    GST_DEBUG ("Buffer metadata copy failed \n");
  }

  return TRUE;
}

static gboolean
gst_nvvideoconvert_sink_event(GstBaseTransform *trans, GstEvent *event)
{
  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS:
      GstCaps *caps = NULL;
      gst_event_parse_caps (event, &caps);
      for (guint i = 0; i < gst_caps_get_size(caps); i++)
      {
        GstStructure *structure = gst_caps_get_structure(caps, i);
        const gchar *format = gst_structure_get_string(structure, "format");
        // If format is BGRA64_LE, log a warning and break
        if (format && g_str_equal(format, "BGRA64_LE"))
        {
          GST_WARNING_OBJECT(trans,
              "Only BGRA64_LE->BGRA64_LE, BGRA64_LE->UYVP and vice versa is supported. "
              "Make sure to specify format in capsfilter for nvvideoconvert at both sink and src pads explicitly.");
          break;
        }
      }
      break;
    default:
      break;
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

static gboolean
gst_nvvideoconvert_query (GstBaseTransform *trans, GstPadDirection direction,
                                   GstQuery *query)
{
  Gstnvvideoconvert *filter;
  filter = GST_NVVIDEOCONVERT (trans);

  if (gst_nvquery_is_update_caps(query)) {
    guint stream_index = 0; const GValue *frame_rate = NULL; GstStructure *str;

    gst_nvquery_parse_update_caps(query, &stream_index, frame_rate);

    str = gst_structure_new ("update-caps", "stream-id", G_TYPE_UINT,
          stream_index, "width-val", G_TYPE_INT, filter->to_width,
	  "height-val", G_TYPE_INT, filter->to_height, NULL);
    if(frame_rate) {
      gst_structure_set_value (str, "frame-rate", frame_rate);
    }

    return gst_nvquery_update_caps_peer_query(trans->srcpad, str);
  }

  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_CAPS:
      /* Use the default query caps implementation from base transform */
      gboolean ret = GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction, query);
      if (ret)
      {
        /* Parse the caps that is set on the query respocse */
        GstCaps *result_caps = NULL;
        gst_query_parse_caps_result(query, &result_caps);
        if (result_caps)
        {
          GstCaps *modified_caps = inspect_caps(trans, direction, result_caps);
          gst_query_set_caps_result(query, modified_caps);
          if (modified_caps != result_caps) {
            gst_caps_unref(modified_caps);
          }
        }
      }
      return ret;

    default:
      break;
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction, query);

}

static gboolean
block_linear_layout_check(Gstnvvideoconvert *space)
{
  guint block_linear_color_format = FALSE;
  switch(space->out_pix_fmt){
    case NVBUF_COLOR_FORMAT_NV12:
    case NVBUF_COLOR_FORMAT_NV12_ER:
    case NVBUF_COLOR_FORMAT_NV12_709:
    case NVBUF_COLOR_FORMAT_NV12_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_2020:
    case NVBUF_COLOR_FORMAT_NV12_10LE:
    case NVBUF_COLOR_FORMAT_NV12_10LE_ER:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_10LE_2020:
    case NVBUF_COLOR_FORMAT_NV12_12LE:
    case NVBUF_COLOR_FORMAT_NV12_12LE_ER:
    case NVBUF_COLOR_FORMAT_NV12_12LE_709:
    case NVBUF_COLOR_FORMAT_NV12_12LE_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_12LE_2020:
      block_linear_color_format= TRUE;
      break;
    default:
      block_linear_color_format= FALSE;
      break;
  }
  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info = {0};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      is_nvgpu = 1;
    }
  }
  // block linear is not supported for dgpu
  if(space->enable_blocklinear_output && block_linear_color_format && is_nvgpu)
    return TRUE;
  else
    return FALSE;
}
static guint
get_output_frame_size (Gstnvvideoconvert * space)
{
  guint size = 0;

  switch (GST_VIDEO_FORMAT_INFO_FORMAT (space->out_info.finfo)) {
    case GST_VIDEO_FORMAT_I420:
    case GST_VIDEO_FORMAT_Y444:
    case GST_VIDEO_FORMAT_NV12:
    case GST_VIDEO_FORMAT_NV21:
    case GST_VIDEO_FORMAT_GRAY8:
    case GST_VIDEO_FORMAT_GRAY16_LE:
    case GST_VIDEO_FORMAT_UYVY:
    case GST_VIDEO_FORMAT_YVYU:
    case GST_VIDEO_FORMAT_YUY2:
    case GST_VIDEO_FORMAT_Y42B:
    case GST_VIDEO_FORMAT_P010_10LE:
    case GST_VIDEO_FORMAT_I420_12LE:
    case GST_VIDEO_FORMAT_GBR:
    case GST_VIDEO_FORMAT_UYVP:
    case GST_VIDEO_FORMAT_Y444_10LE:
    case GST_VIDEO_FORMAT_Y444_12LE:
      size = space->out_info.size;
      break;

    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_ARGB:
    case GST_VIDEO_FORMAT_ABGR:
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_xBGR:
    case GST_VIDEO_FORMAT_BGR10A2_LE:
    case GST_VIDEO_FORMAT_RGB10A2_LE:
      size = (space->to_width * space->to_height * 4);
      break;

    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
      size = (space->to_width * space->to_height * 3);
      break;

    case GST_VIDEO_FORMAT_BGRA64_LE:
      size = (space->to_width * space->to_height * 8);
      break;
    default:
      g_print ("%s : %s: Unsupported output format () ... Exiting....\n",
          GST_ELEMENT_NAME (space), __func__);
      exit (-1);
  }
  return size;
}

static GstCaps *
gst_nvvideoconvert_caps_remove_format_info (GstCaps * caps)
{
  GstStructure *str;
  GstCapsFeatures *features;
  gint i, n;
  GstCaps *ret;

  ret = gst_caps_new_empty ();

  n = gst_caps_get_size (caps);
  for (i = 0; i < n; i++) {
    str = gst_caps_get_structure (caps, i);
    features = gst_caps_get_features (caps, i);

    /* If this is already expressed by the existing caps
     * skip this structure */
    if (i > 0 && gst_caps_is_subset_structure_full (ret, str, features))
      continue;

    str = gst_structure_copy (str);
    /* Only remove format info for the cases when we can actually convert */
    {
      if (!gst_caps_features_is_any (features)) {
        gst_structure_remove_fields (str, "format", "chroma-site",
            "colorimetry", "plane-order", "precision","block-linear",
            "nvbuf-memory-type", "gpu-id", NULL);
      }

      gst_structure_set (str, "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
          "height", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);

      /* if pixel aspect ratio, make a range */
      if (gst_structure_has_field (str, "pixel-aspect-ratio"))
        gst_structure_set (str, "pixel-aspect-ratio", GST_TYPE_FRACTION_RANGE,
            1, G_MAXINT, G_MAXINT, 1, NULL);
    }
    gst_caps_append_structure_full (ret, str,
        gst_caps_features_copy (features));
  }

  return ret;
}

static GstCaps *
gst_nvvideoconvert_caps_remove_batchsize(GstCaps * caps)
{
  GstStructure *str;
  GstCapsFeatures *features;
  gint i, n;
  GstCaps *ret;

  ret = gst_caps_new_empty ();

  n = gst_caps_get_size (caps);
  for (i = 0; i < n; i++) {
    str = gst_caps_get_structure (caps, i);
    features = gst_caps_get_features (caps, i);

    /* If this is already expressed by the existing caps
     * skip this structure */
    if (i > 0 && gst_caps_is_subset_structure_full (ret, str, features))
      continue;

    str = gst_structure_copy (str);
      if (!gst_caps_features_is_any (features)) {
        gst_structure_remove_fields (str, "batch-size", "num-surfaces-per-frame", NULL);
      }
    gst_caps_append_structure_full (ret, str,
        gst_caps_features_copy (features));
  }

  return ret;
}

static gboolean
gst_nvvideoconvert_get_pix_fmt (GstVideoInfo * info,
    NvBufSurfaceColorFormat * pix_fmt, gint * isurf_count, gint plane_order,
    guint precision)
{
  gboolean ret = TRUE;

  if (GST_VIDEO_INFO_IS_YUV (info)) {
    switch (GST_VIDEO_FORMAT_INFO_FORMAT (info->finfo)) {
      case GST_VIDEO_FORMAT_I420:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUV420;
        if (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) {
          if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
            *pix_fmt = NVBUF_COLOR_FORMAT_YUV420_709_ER;
          else
            *pix_fmt = NVBUF_COLOR_FORMAT_YUV420_ER;
        } else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV420_709;

        break;
      case GST_VIDEO_FORMAT_UYVY:
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709) {
          *pix_fmt = (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) ? NVBUF_COLOR_FORMAT_UYVY_709_ER : NVBUF_COLOR_FORMAT_UYVY_709;
        } else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020) {
          *pix_fmt = NVBUF_COLOR_FORMAT_UYVY_2020;
        } else {
          *pix_fmt = (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) ? NVBUF_COLOR_FORMAT_UYVY_ER : NVBUF_COLOR_FORMAT_UYVY;
        }
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_Y444:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUV444;
        if (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) {
          if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
            *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_709_ER;
          else
            *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_ER;
        } else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_709;
        *isurf_count = 3;
        break;
      case GST_VIDEO_FORMAT_Y444_10LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_10LE;
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_10LE_709;
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_10LE_2020;
        *isurf_count = 3;
        break;
      case GST_VIDEO_FORMAT_Y444_12LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_12LE;
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_12LE_709;
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV444_12LE_2020;
        *isurf_count = 3;
        break;
      case GST_VIDEO_FORMAT_YUY2:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUYV;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_YVYU:
        *pix_fmt = NVBUF_COLOR_FORMAT_YVYU;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_Y42B:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUV422;
        *isurf_count=3;
        break;
      case GST_VIDEO_FORMAT_NV12:
        *pix_fmt = NVBUF_COLOR_FORMAT_NV12;
        *isurf_count = 2;
        if (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) {
          if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
            *pix_fmt = NVBUF_COLOR_FORMAT_NV12_709_ER;
          else
            *pix_fmt = NVBUF_COLOR_FORMAT_NV12_ER;
        } else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_709;
        break;
      case GST_VIDEO_FORMAT_GRAY8:
        *pix_fmt = (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) ? NVBUF_COLOR_FORMAT_GRAY8_ER : NVBUF_COLOR_FORMAT_GRAY8;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_GRAY16_LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_GRAY16_LE;
        *isurf_count = 1;
        break;
        // Not sure why both grouped here?
      case GST_VIDEO_FORMAT_I420_10LE:
      case GST_VIDEO_FORMAT_P010_10LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_NV12_10LE;
        // info->colorimetry.range is applicable for 8 bits, not sure make sense here
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_10LE_709;
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_10LE_2020;
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT601)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_10LE;
        *isurf_count = 2;

        break;
#if GST_VERSION_MAJOR == 1 && GST_VERSION_MINOR >=14
      case GST_VIDEO_FORMAT_I420_12LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_NV12_12LE;
        // info->colorimetry.range is applicable for 8 bits, not sure make sense here
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_12LE_709;
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_12LE_2020;
        *isurf_count = 2;
        break;
#endif
      case GST_VIDEO_FORMAT_UYVP:
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709) {
          *pix_fmt = (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) ? NVBUF_COLOR_FORMAT_UYVP_709_ER : NVBUF_COLOR_FORMAT_UYVP_709;
        } else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020) {
          *pix_fmt = NVBUF_COLOR_FORMAT_UYVP_2020;
        } else {
          *pix_fmt = (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) ? NVBUF_COLOR_FORMAT_UYVP_ER : NVBUF_COLOR_FORMAT_UYVP;
        }
        *isurf_count = 1;
        break;

      default:
        ret = FALSE;
        break;
    }
  } else if (GST_VIDEO_INFO_IS_RGB (info)) {
    switch (GST_VIDEO_FORMAT_INFO_FORMAT (info->finfo)) {
      case GST_VIDEO_FORMAT_BGRx:
        *pix_fmt = NVBUF_COLOR_FORMAT_BGRx;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_RGBA:
        *pix_fmt = NVBUF_COLOR_FORMAT_RGBA;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_GBR:

        if (precision != P_FLOAT32) {
          if (plane_order == PLANAR_RGB)
            *pix_fmt = NVBUF_COLOR_FORMAT_R8_G8_B8;
          else
            *pix_fmt = NVBUF_COLOR_FORMAT_B8_G8_R8;
        } else if (precision == P_FLOAT32) {
          if (plane_order == PLANAR_RGB)
            *pix_fmt = NVBUF_COLOR_FORMAT_R32F_G32F_B32F;
          else
            *pix_fmt = NVBUF_COLOR_FORMAT_B32F_G32F_R32F;
        }
        *isurf_count = 3;
        break;
      case GST_VIDEO_FORMAT_BGR:
        *pix_fmt = NVBUF_COLOR_FORMAT_BGR;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_RGB:
        *pix_fmt = NVBUF_COLOR_FORMAT_RGB;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_BGR10A2_LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709;
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020)
          *pix_fmt = NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_2020;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_RGB10A2_LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709;
        if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT2020)
          *pix_fmt = NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_2020;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_BGRA64_LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_BGRA64_LE;
        *isurf_count = 1;
        break;
      default:
        ret = FALSE;
        break;
    }
  } else if (GST_VIDEO_INFO_IS_GRAY (info)) {
    switch (GST_VIDEO_FORMAT_INFO_FORMAT (info->finfo)) {
      case GST_VIDEO_FORMAT_GRAY8:
        *pix_fmt = (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) ? NVBUF_COLOR_FORMAT_GRAY8_ER : NVBUF_COLOR_FORMAT_GRAY8;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_GRAY16_LE:
        *pix_fmt = NVBUF_COLOR_FORMAT_GRAY16_LE;
        *isurf_count = 1;
        break;
      default:
        ret = FALSE;
        break;
    }
  }

  return ret;
}

static void
gst_nvvideoconvert_init_params (Gstnvvideoconvert * filter)
{
  filter->silent = FALSE;
  filter->to_width = 0;
  filter->to_height = 0;
  filter->from_width = 0;
  filter->from_height = 0;
  filter->tsurf_width = 0;
  filter->tsurf_height = 0;

  filter->inbuf_type = BUF_NOT_SUPPORTED;
  filter->inbuf_memtype = BUF_MEM_SW;
  filter->outbuf_memtype = BUF_MEM_SW;


  filter->in_pix_fmt = NVBUF_COLOR_FORMAT_INVALID;
  filter->out_pix_fmt = NVBUF_COLOR_FORMAT_INVALID;

  filter->do_scaling = FALSE;
  filter->need_intersurf = FALSE;
  filter->isurf_flag = FALSE;
  filter->nvfilterpool = FALSE;

  filter->insurf_count = 0;
  filter->isurf_count = 0;
  filter->tsurf_count = 0;
  filter->ibuf_count = 0;

  filter->silent = FALSE;
  filter->no_dimension = FALSE;
  filter->do_flip = FALSE;
  filter->flip_method = GST_VIDEO_NVFLIP_METHOD_IDENTITY;
  filter->interpolation_method = NvBufSurfTransformInter_Default;
  filter->negotiated = FALSE;
  filter->num_output_buf = NVFILTER_MAX_BUF;
  filter->enable_blocklinear_output = FALSE;
  filter->allow_odd_crop = TRUE;
  filter->enable_contiguous_bufs = FALSE;
  filter->disable_passthrough =  FALSE;

  filter->do_src_cropping = FALSE;
  filter->do_dst_cropping = FALSE;
  filter->dst_crop_width = 0;
  filter->dst_crop_left = 0;
  filter->dst_crop_top = 0;
  filter->dst_crop_height = 0;
  filter->src_crop_width = 0;
  filter->src_crop_left = 0;
  filter->src_crop_top = 0;
  filter->src_crop_height = 0;
  filter->num_batch_buffers = 1;
  filter->compute_hw = NvBufSurfTransformCompute_Default;
  filter->copy_hw = NvBufSurfTransformCompute_GPU;
  filter->gpu_id = 0;
  filter->nvbuf_mem_type = NVBUF_MEM_CUDA_DEVICE;
  filter->do_mem_type_conversion = FALSE;
  filter->do_gpu_id_conversion = FALSE;

  NvBufSurfaceDeviceInfo dev_info = {0};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      filter->nvbuf_mem_type = NVBUF_MEM_SURFACE_ARRAY;
    }
  }

  filter->sinkcaps =
      gst_static_pad_template_get_caps (&gst_nvvideoconvert_sink_template);
  filter->srccaps =
      gst_static_pad_template_get_caps (&gst_nvvideoconvert_src_template);

  g_mutex_init (&filter->flow_lock);
}


static void
gst_nvvideoconvert_class_init (GstnvvideoconvertClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  gparent_class = g_type_class_peek_parent (gstbasetransform_class);

  gobject_class->set_property = gst_nvvideoconvert_set_property;
  gobject_class->get_property = gst_nvvideoconvert_get_property;
  gobject_class->finalize = gst_nvvideoconvert_finalize;

  gstelement_class->change_state = gst_nvvideoconvert_change_state;

  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_set_caps);
  gstbasetransform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_transform_caps);
  gstbasetransform_class->accept_caps =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_accept_caps);
  gstbasetransform_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_transform_size);
  gstbasetransform_class->get_unit_size =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_get_unit_size);
  gstbasetransform_class->transform =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_transform);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvvideoconvert_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvvideoconvert_stop);
  gstbasetransform_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_fixate_caps);
  gstbasetransform_class->decide_allocation =
      GST_DEBUG_FUNCPTR (gst_nvvideoconvert_decide_allocation);
  gstbasetransform_class->copy_metadata = GST_DEBUG_FUNCPTR (gst_nvvideoconvert_copy_metadata);
  gstbasetransform_class->query = GST_DEBUG_FUNCPTR (gst_nvvideoconvert_query);
  gstbasetransform_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvvideoconvert_sink_event);

  gstbasetransform_class->passthrough_on_same_caps = TRUE;

  GST_TYPE_RGB_PLANE_ORDER;
  GST_TYPE_FORMAT_PRECISION;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);
  g_object_class_install_property (gobject_class, PROP_FLIP_METHOD,
      g_param_spec_enum ("flip-method", "Flip-Method", "video flip methods",
          GST_TYPE_VIDEO_NVFLIP_METHOD, PROP_FLIP_METHOD_DEFAULT,
          GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_CONSTRUCT |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_OUT_BUFS,
      g_param_spec_uint ("output-buffers", "Output-Buffers",
          "number of output buffers",
          1, G_MAXUINT, NVFILTER_MAX_BUF,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_PLAYING));

  g_object_class_install_property (gobject_class, PROP_SRC_CROP,
      g_param_spec_string ("src-crop", "src-crop",
          "Pixel location left:top:width:height\n"
          "\t\t\tUse string with values of crop location to set the property.\n"
          "\t\t\t e.g. 20:20:40:50",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_DST_CROP,
      g_param_spec_string ("dest-crop", "dest-crop",
          "Pixel location left:top:width:height\n"
          "\t\t\tUse string with values of crop location to set the property.\n"
          "\t\t\t e.g. 20:20:40:50",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_BLOCKLINEAR_OUTPUT,
      g_param_spec_boolean ("bl-output", " Enable BlockLinear output",
        "Blocklinear output, applicable only for memory:NVMM NV12 format output buffer",
         FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_ENABLE_ODD_CROP,
      g_param_spec_boolean ("allow-odd-crop", "Allow Odd Crop ",
        "Allow the odd dimensions for source and destination crop rectangle",
         TRUE, G_PARAM_READWRITE));

  PROP_COMPUTE_HW_INSTALL(gobject_class);
  PROP_NVBUF_MEMORY_TYPE_INSTALL(gobject_class);
  PROP_INTERPOLATION_METHOD_INSTALL(gobject_class);

  g_object_class_install_property (gobject_class, PROP_COPY_HW,
      g_param_spec_enum ("copy-hw", " Select copy operation hardware",
        "Select hardware used for surface copies.",
        GST_TYPE_COPY_HW, NvBufSurfTransformCompute_GPU,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
        GST_PARAM_CONTROLLABLE | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_CONTIGUOUS_BUFFERS,
      g_param_spec_boolean ("contiguous-buffers", " Enable contiguous output buffers",
        "Transformed output buffers in a batch are contiguous in memory.",
        FALSE, G_PARAM_READWRITE ));

  g_object_class_install_property (gobject_class, PROP_DISABLE_PASSTHROUGH,
      g_param_spec_boolean ("disable-passthrough", " disable-passthrough",
        "Disable passthrough mode at init time",
        FALSE, G_PARAM_READWRITE ));

  gst_element_class_set_details_simple (gstelement_class,
      "NvVidConv Plugin",
      "Filter/Converter/Video/Scaler",
      "Converts video from one colorspace to another & Resizes",
      "NVIDIA Corporation. Post on Deepstream SDK forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvvideoconvert_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvvideoconvert_sink_template));
}

static void
gst_nvvideoconvert_init (Gstnvvideoconvert * filter)
{
  gst_nvvideoconvert_init_params (filter);

#ifndef NEW_METADATA
  if (!_ivameta_quark)
    _ivameta_quark = g_quark_from_static_string ("ivameta");

  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
#endif
}

static gint
get_bytes_per_pix_from_color (NvBufSurfaceColorFormat pix_fmt, gint plane_id)
{
  /* Here, for multiplanar formats, bytes_per_pix is the number of bytes required to store
     individual component in a pixel.
     For single planar formats, it is the bytes required for forming 1 actual pixel.
  */
  gint bytes_per_pix = 1;
  switch (pix_fmt) {
    case NVBUF_COLOR_FORMAT_YUV420:
    case NVBUF_COLOR_FORMAT_YUV422:
    case NVBUF_COLOR_FORMAT_YUV420_709:
    case NVBUF_COLOR_FORMAT_YUV420_709_ER:
    case NVBUF_COLOR_FORMAT_YUV420_ER:
    case NVBUF_COLOR_FORMAT_YUV444:
    case NVBUF_COLOR_FORMAT_YUV444_ER:
    case NVBUF_COLOR_FORMAT_YUV444_709:
    case NVBUF_COLOR_FORMAT_YUV444_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_2020:
      bytes_per_pix = 1;
      break;
    case NVBUF_COLOR_FORMAT_NV12:
    case NVBUF_COLOR_FORMAT_NV12_709:
    case NVBUF_COLOR_FORMAT_NV12_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_ER:
      if (plane_id == 0)
        bytes_per_pix = 1;
      else
        bytes_per_pix = 2;
      break;
    case NVBUF_COLOR_FORMAT_NV12_10LE:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709:
    case NVBUF_COLOR_FORMAT_NV12_10LE_2020:
    case NVBUF_COLOR_FORMAT_NV12_12LE:
    case NVBUF_COLOR_FORMAT_NV12_12LE_709:
    case NVBUF_COLOR_FORMAT_NV12_12LE_2020:
      if (plane_id == 0)
        bytes_per_pix = 2;
      else
        bytes_per_pix = 4;
      break;
    case NVBUF_COLOR_FORMAT_YUV444_10LE:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_709:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_2020:
    case NVBUF_COLOR_FORMAT_YUV444_12LE:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_ER:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_709:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_2020:
      bytes_per_pix = 2;
      break;
    case NVBUF_COLOR_FORMAT_UYVY:
    case NVBUF_COLOR_FORMAT_UYVY_ER:
    case NVBUF_COLOR_FORMAT_UYVY_709:
    case NVBUF_COLOR_FORMAT_UYVY_709_ER:
    case NVBUF_COLOR_FORMAT_UYVY_2020:
      bytes_per_pix = 2;
      break;
    case NVBUF_COLOR_FORMAT_YUYV:
    case NVBUF_COLOR_FORMAT_YVYU:
    // Doesnt make sense, its more like element per pix
    case NVBUF_COLOR_FORMAT_UYVP:
    case NVBUF_COLOR_FORMAT_UYVP_ER:
    case NVBUF_COLOR_FORMAT_UYVP_709:
    case NVBUF_COLOR_FORMAT_UYVP_709_ER:
    case NVBUF_COLOR_FORMAT_UYVP_2020:
      bytes_per_pix = 2;
      break;
    case NVBUF_COLOR_FORMAT_BGRx:
    case NVBUF_COLOR_FORMAT_RGBA:
    case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709:
    case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_2020:
    case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709:
    case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_2020:
      bytes_per_pix = 4;
      break;
    case NVBUF_COLOR_FORMAT_BGR:
    case NVBUF_COLOR_FORMAT_RGB:
      bytes_per_pix = 3;
      break;
    case NVBUF_COLOR_FORMAT_GRAY8:
    case NVBUF_COLOR_FORMAT_GRAY8_ER:
    case NVBUF_COLOR_FORMAT_R8_G8_B8:
    case NVBUF_COLOR_FORMAT_B8_G8_R8:
      bytes_per_pix = 1;
      break;
    case NVBUF_COLOR_FORMAT_GRAY16_LE:
      bytes_per_pix = 2;
      break;
    case NVBUF_COLOR_FORMAT_R32F_G32F_B32F:
    case NVBUF_COLOR_FORMAT_B32F_G32F_R32F:
      bytes_per_pix = sizeof (float);
      break;
    case NVBUF_COLOR_FORMAT_BGRA64_LE:
      bytes_per_pix = 8;
      break;
    default:
      break;
  }

  return bytes_per_pix;
}


static void
get_NvBufferTransform (Gstnvvideoconvert * filter)
{
  switch (filter->flip_method) {
    case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
      filter->transform_params.transform_flip = NvBufSurfTransform_None;
      break;
    case GST_VIDEO_NVFLIP_METHOD_90L:
      filter->transform_params.transform_flip = NvBufSurfTransform_Rotate90;
      break;
    case GST_VIDEO_NVFLIP_METHOD_180:
      filter->transform_params.transform_flip = NvBufSurfTransform_Rotate180;
      break;
    case GST_VIDEO_NVFLIP_METHOD_90R:
      filter->transform_params.transform_flip = NvBufSurfTransform_Rotate270;
      break;
    case GST_VIDEO_NVFLIP_METHOD_HORIZ:
      filter->transform_params.transform_flip = NvBufSurfTransform_FlipX;
      break;
    case GST_VIDEO_NVFLIP_METHOD_VERT:
      filter->transform_params.transform_flip = NvBufSurfTransform_FlipY;
      break;
    case GST_VIDEO_NVFLIP_METHOD_TRANS:
      filter->transform_params.transform_flip = NvBufSurfTransform_Transpose;
      break;
    case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
      filter->transform_params.transform_flip = NvBufSurfTransform_InvTranspose;
      break;
    default:
      break;
  }
}

#if 0
static void
get_NvBufferTransform_filter (Gstnvvideoconvert * filter)
{
  switch (filter->interpolation_method) {
    case GST_INTERPOLATION_NEAREST:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_Nearest;
      break;
    case GST_INTERPOLATION_BILINEAR:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_Bilinear;
      break;
    case GST_INTERPOLATION_5_TAP:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_5_Tap;
      break;
    case GST_INTERPOLATION_10_TAP:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_10_Tap;
      break;
    case GST_INTERPOLATION_SMART:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_Smart;
      break;
    case GST_INTERPOLATION_NICEST:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_Nicest;
      break;
    default:
      filter->transform_params.transform_filter =
          NvBufferTransform_Filter_Smart;
      break;
  }
}
#endif

static void
gst_nvvideoconvert_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstnvvideoconvert *filter = GST_NVVIDEOCONVERT (object);
  gchar *prop_str = NULL;

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_FLIP_METHOD:
      filter->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_FLIP;
      filter->do_flip = TRUE;
      filter->flip_method = g_value_get_enum (value);
      get_NvBufferTransform (filter);
      gst_base_transform_reconfigure_src (GST_BASE_TRANSFORM (filter));
      break;
    case PROP_NUM_OUT_BUFS:
      filter->num_output_buf = g_value_get_uint (value);
      break;
    case PROP_INTERPOLATION_METHOD:
      filter->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_FILTER;
      filter->interpolation_method = g_value_get_enum (value);
      break;
    case PROP_COMPUTE_HW:
      filter->compute_hw = g_value_get_enum (value);
      break;
    case PROP_COPY_HW:
      filter->copy_hw = g_value_get_enum (value);
      break;
    case PROP_GPU_DEVICE_ID:
      filter->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      filter->nvbuf_mem_type = g_value_get_enum (value);
      break;
    case PROP_SRC_CROP:
      filter->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_CROP_SRC;
      prop_str = (gchar *) (g_value_get_string (value));
      if (prop_str) {
        gchar **curr_ptr = g_strsplit (prop_str, ":", -1);

        if (curr_ptr &&
            curr_ptr[0] && curr_ptr[1] && curr_ptr[2] && curr_ptr[3]) {
          filter->src_crop_left =
              (gint) g_ascii_strtoll (curr_ptr[0], (gchar **) NULL, 10);
          filter->src_crop_top =
              (gint) g_ascii_strtoll (curr_ptr[1], (gchar **) NULL, 10);
          filter->src_crop_width =
              (gint) g_ascii_strtoll (curr_ptr[2], (gchar **) NULL, 10);
          filter->src_crop_height =
              (gint) g_ascii_strtoll (curr_ptr[3], (gchar **) NULL, 10);
          filter->do_src_cropping = TRUE;
        }

        g_strfreev (curr_ptr);
      }
      break;
    case PROP_DST_CROP:
      filter->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_CROP_DST;
      prop_str = (gchar *) (g_value_get_string (value));
      if (prop_str) {
        gchar **curr_ptr = g_strsplit (prop_str, ":", -1);

        if (curr_ptr &&
            curr_ptr[0] && curr_ptr[1] && curr_ptr[2] && curr_ptr[3]) {
          filter->dst_crop_left =
              (gint) g_ascii_strtoll (curr_ptr[0], (gchar **) NULL, 10);
          filter->dst_crop_top =
              (gint) g_ascii_strtoll (curr_ptr[1], (gchar **) NULL, 10);
          filter->dst_crop_width =
              (gint) g_ascii_strtoll (curr_ptr[2], (gchar **) NULL, 10);
          filter->dst_crop_height =
              (gint) g_ascii_strtoll (curr_ptr[3], (gchar **) NULL, 10);
          filter->do_dst_cropping = TRUE;
        }

        g_strfreev (curr_ptr);
      }
      break;
    case PROP_ENABLE_BLOCKLINEAR_OUTPUT:
        filter->enable_blocklinear_output = g_value_get_boolean (value);
      break;
    case PROP_ENABLE_ODD_CROP:
        filter->allow_odd_crop = g_value_get_boolean (value);
      break;
    case PROP_ENABLE_CONTIGUOUS_BUFFERS:
        filter->enable_contiguous_bufs = g_value_get_boolean (value);
      break;
    case PROP_DISABLE_PASSTHROUGH:
        filter->disable_passthrough = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvvideoconvert_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstnvvideoconvert *filter = GST_NVVIDEOCONVERT (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_FLIP_METHOD:
      g_value_set_enum (value, filter->flip_method);
      break;
    case PROP_NUM_OUT_BUFS:
      g_value_set_uint (value, filter->num_output_buf);
      break;
    case PROP_INTERPOLATION_METHOD:
      g_value_set_enum (value, filter->interpolation_method);
      break;
    case PROP_COMPUTE_HW:
      g_value_set_enum (value, filter->compute_hw);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, filter->gpu_id);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, filter->nvbuf_mem_type);
      break;
    case PROP_DST_CROP:
    {
      gchar *str_val = g_strdup_printf ("%d:%d:%d:%d", filter->dst_crop_left,
          filter->dst_crop_top,
          filter->dst_crop_width, filter->dst_crop_height);
      g_value_set_string (value, str_val);
      g_free (str_val);
    }
      break;
    case PROP_SRC_CROP:
    {
      gchar *str_val = g_strdup_printf ("%d:%d:%d:%d", filter->src_crop_left,
          filter->src_crop_top,
          filter->src_crop_width, filter->src_crop_height);
      g_value_set_string (value, str_val);
      g_free (str_val);
    }
      break;
    case PROP_ENABLE_BLOCKLINEAR_OUTPUT:
      g_value_set_boolean (value, filter->enable_blocklinear_output);
      break;
    case PROP_ENABLE_ODD_CROP:
      g_value_set_boolean (value, filter->allow_odd_crop);
      break;
    case PROP_COPY_HW:
      g_value_set_enum (value, filter->copy_hw);
      break;
    case PROP_ENABLE_CONTIGUOUS_BUFFERS:
      g_value_set_boolean (value, filter->enable_contiguous_bufs);
      break;
    case PROP_DISABLE_PASSTHROUGH:
      g_value_set_boolean (value, filter->disable_passthrough);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvvideoconvert_free_buf (Gstnvvideoconvert * filter)
{
  //gint ret;

#if 0
  if (filter->isurf_count) {
    ret = NvBufSurfaceDestroy (filter->interbuf.idmabuf_fd);
    if (ret != 0) {
      GST_ERROR ("%s: intermediate NvBufferDestroy Failed \n", __func__);
    }
  }
#endif
  filter->isurf_count = 0;
  filter->ibuf_count = 0;
}

static GstStateChangeReturn
gst_nvvideoconvert_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn result = GST_STATE_CHANGE_SUCCESS;
  Gstnvvideoconvert *space;

  space = GST_NVVIDEOCONVERT (element);

  switch (transition) {
    default:
      break;
  }

  GST_ELEMENT_CLASS (gparent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:{
      gst_nvvideoconvert_free_buf (space);
    }
      break;
    default:
      break;
  }

  return result;
}

static void
gst_nvvideoconvert_finalize (GObject * object)
{
  Gstnvvideoconvert *filter;

  filter = GST_NVVIDEOCONVERT (object);

  if (filter->sinkcaps) {
    gst_caps_unref (filter->sinkcaps);
    filter->sinkcaps = NULL;
  }

  if (filter->srccaps) {
    gst_caps_unref (filter->srccaps);
    filter->sinkcaps = NULL;
  }

  g_mutex_clear (&filter->flow_lock);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_nv_structure_get_enum_new (const GstStructure * structure,
    const gchar * fieldname, GType enumtype, gint * value)
{
  const gchar *fieldValue = NULL;
  GEnumClass *enum_class;
  GEnumValue *enum_value;
  gboolean ret_value = FALSE;

  g_return_val_if_fail (structure != NULL, FALSE);
  g_return_val_if_fail (fieldname != NULL, FALSE);
  g_return_val_if_fail (enumtype != G_TYPE_INVALID, FALSE);
  g_return_val_if_fail (value != NULL, FALSE);

  enum_class = g_type_class_ref (enumtype);
  fieldValue = gst_structure_get_string (structure, fieldname);
  if (!fieldValue && (gst_structure_get_int (structure, fieldname, value) &&
          g_enum_get_value (enum_class, *value))) {
    ret_value = TRUE;
  } else if ((enum_value =
          g_enum_get_value_by_nick (enum_class, fieldValue)) != NULL) {
    *value = enum_value->value;
    ret_value = TRUE;
  }

  g_type_class_unref (enum_class);
  return ret_value;
}

static gboolean
gst_nvvideoconvert_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  gboolean ret = TRUE;
  Gstnvvideoconvert *space;
  gint from_dar_n = 0, from_dar_d = 0, to_dar_n = 0, to_dar_d = 0;
  GstVideoInfo in_info = {0}, out_info = {0};
  GstBufferPool *newpool = {0}, *oldpool = {0};
  GstStructure *config = NULL;
  gint min, surf_count = 0;
  GstCapsFeatures *ift = NULL;
  GstCapsFeatures *oft = NULL;
  GstStructure *in_struct = NULL;
  GstStructure *out_struct = NULL;
  gint in_rgb = 0;
  gint out_rgb = 0;
  gint in_precision = 0;
  gint out_precision = 0;
  gchar *caps_str;

  space = GST_NVVIDEOCONVERT (btrans);

  /* input caps */
  if (!gst_video_info_from_caps (&in_info, incaps))
    goto invalid_caps;

  /* output caps */
  if (!gst_video_info_from_caps (&out_info, outcaps))
    goto invalid_caps;

  space->in_info = in_info;
  space->out_info = out_info;

  space->from_width = GST_VIDEO_INFO_WIDTH (&in_info);
  space->from_height = GST_VIDEO_INFO_HEIGHT (&in_info);

  space->to_width = GST_VIDEO_INFO_WIDTH (&out_info);
  space->to_height = GST_VIDEO_INFO_HEIGHT (&out_info);

  if ((space->from_width != space->to_width)
      || (space->from_height != space->to_height))
    space->do_scaling = TRUE;


  ift = gst_caps_get_features (incaps, 0);
  if (gst_caps_features_contains (ift, GST_CAPS_FEATURE_MEMORY_NVMM))
    space->inbuf_memtype = BUF_MEM_HW;

  oft = gst_caps_get_features (outcaps, 0);
  if (gst_caps_features_contains (oft, GST_CAPS_FEATURE_MEMORY_NVMM))
    space->outbuf_memtype = BUF_MEM_HW;

  in_struct = gst_caps_get_structure (incaps, 0);
  out_struct = gst_caps_get_structure (outcaps, 0);

  if (gst_structure_has_field (out_struct, "plane-order")) {
    if (gst_nv_structure_get_enum_new (out_struct, "plane-order",
            GST_TYPE_RGB_PLANE_ORDER, &out_rgb)) {
    } else {
      out_rgb = rgb_plane_order[0].value;
      const GEnumValue *fmt_pr_ptr = rgb_plane_order;
      GString *op_str = g_string_new ("`plane-order` can be ");
      for (; fmt_pr_ptr->value_nick != NULL; fmt_pr_ptr++) {
        g_string_append_printf (op_str, " %d->%s ", fmt_pr_ptr->value,
            fmt_pr_ptr->value_nick);
      }
      GST_WARNING_OBJECT (space, "%s, setting it as %s", op_str->str,
          rgb_plane_order[0].value_nick);
      g_string_free (op_str, TRUE);
    }
  }

  if (gst_structure_has_field (in_struct, "plane-order")) {
    if (gst_nv_structure_get_enum_new (in_struct, "plane-order",
            GST_TYPE_RGB_PLANE_ORDER, &in_rgb)) {
    } else {
      in_rgb = rgb_plane_order[0].value;
      const GEnumValue *fmt_pr_ptr = rgb_plane_order;
      GString *op_str = g_string_new ("`plane-order` can be ");
      for (; fmt_pr_ptr->value_nick != NULL; fmt_pr_ptr++) {
        g_string_append_printf (op_str, " %d->%s ", fmt_pr_ptr->value,
            fmt_pr_ptr->value_nick);
      }
      GST_WARNING_OBJECT (space, "%s, setting it as %s", op_str->str,
          rgb_plane_order[0].value_nick);
      g_string_free (op_str, TRUE);
    }
  }

  if (gst_structure_has_field (out_struct, "precision")) {
    if (gst_nv_structure_get_enum_new (out_struct, "precision",
            GST_TYPE_FORMAT_PRECISION, &out_precision)) {
    } else {
      const GEnumValue *fmt_pr_ptr = format_precision;
      GString *op_str = g_string_new ("`precision` can be ");
      out_precision = format_precision[0].value;
      for (; fmt_pr_ptr->value_nick != NULL; fmt_pr_ptr++) {
        g_string_append_printf (op_str, " %d->%s ", fmt_pr_ptr->value,
            fmt_pr_ptr->value_nick);
      }
      GST_WARNING_OBJECT (space, "%s, setting it as %s", op_str->str,
          format_precision[0].value_nick);
      g_string_free (op_str, TRUE);
    }
  }

  if (gst_structure_has_field (in_struct, "precision")) {
    if (gst_nv_structure_get_enum_new (in_struct, "precision",
            GST_TYPE_FORMAT_PRECISION, &in_precision)) {
    } else {
      const GEnumValue *fmt_pr_ptr = format_precision;
      GString *op_str = g_string_new ("precision can be ");
      out_precision = format_precision[0].value;
      for (; fmt_pr_ptr->value_nick != NULL; fmt_pr_ptr++) {
        g_string_append_printf (op_str, " %d->%s ", fmt_pr_ptr->value,
            fmt_pr_ptr->value_nick);
      }
      GST_WARNING_OBJECT (space, "%s, setting it as %s", op_str->str,
          format_precision[0].value_nick);
      g_string_free (op_str, TRUE);
    }
  }

  /* get input pixel format */
  ret =
      gst_nvvideoconvert_get_pix_fmt (&in_info, &space->in_pix_fmt,
      &surf_count, in_rgb, in_precision);
  if (ret != TRUE)
    goto invalid_pix_fmt;

  /* get output pixel format */
  ret =
      gst_nvvideoconvert_get_pix_fmt (&out_info, &space->out_pix_fmt,
      &surf_count, out_rgb, out_precision);
  if (ret != TRUE)
    goto invalid_pix_fmt;

  if (gst_structure_has_field (in_struct, "nvbuf-memory-type") &&
  gst_structure_has_field (out_struct, "nvbuf-memory-type"))
  {
    const gchar * inbuf_mem_type_string = g_value_get_string(gst_structure_get_value (in_struct, "nvbuf-memory-type"));
    const gchar * outbuf_mem_type_string = g_value_get_string(gst_structure_get_value (out_struct, "nvbuf-memory-type"));
    if(!g_str_equal(inbuf_mem_type_string,outbuf_mem_type_string))
       space->do_mem_type_conversion = TRUE;
  }

  if (gst_structure_has_field (in_struct, "gpu-id") &&
  gst_structure_has_field (out_struct, "gpu-id"))
  {
    int inbuf_gpu_id = g_value_get_int(gst_structure_get_value (in_struct, "gpu-id"));
    int outbuf_gpu_id = g_value_get_int(gst_structure_get_value (out_struct, "gpu-id"));
    if(inbuf_gpu_id != outbuf_gpu_id)
       space->do_gpu_id_conversion = TRUE;
  }

  if (gst_caps_features_is_equal (ift, oft) &&
      (space->in_pix_fmt == space->out_pix_fmt) &&
      (!space->do_scaling) &&
      (!space->do_src_cropping) &&
      (!space->do_dst_cropping) &&
      (!space->flip_method) &&
      (!space->enable_blocklinear_output) &&
      (!space->do_mem_type_conversion) &&
      (!space->do_gpu_id_conversion) &&
      (!space->disable_passthrough)) {
    /* We are not processing input buffer. Initializations/allocations in this
       function can be skipped */
    gst_base_transform_set_passthrough (btrans, TRUE);
    return TRUE;
  }
  else {
    gst_base_transform_set_passthrough (btrans, FALSE);
  }

  // if destination crop window exceeds the output window size
  if(space->do_dst_cropping){
    if ((space->dst_crop_left + space->dst_crop_width) > space->to_width) {
      space->dst_crop_width=space->to_width-space->dst_crop_left;
    }
    if ((space->dst_crop_top + space->dst_crop_height) > space->to_height) {
      space->dst_crop_height=space->to_height-space->dst_crop_top;
    }
  }

  // if source crop window exceeds the input window size
  if(space->do_src_cropping){
    if ((space->src_crop_left + space->src_crop_width) > space->from_width) {
      space->src_crop_width=space->from_width-space->src_crop_left;
    }
    if ((space->src_crop_top + space->src_crop_height) > space->from_height) {
      space->src_crop_height=space->from_height-space->src_crop_top;
    }
  }

  switch (space->in_pix_fmt) {
    case NVBUF_COLOR_FORMAT_YUV420:
    case NVBUF_COLOR_FORMAT_YUV422:
    case NVBUF_COLOR_FORMAT_YUV444:
    case NVBUF_COLOR_FORMAT_YUV444_ER:
    case NVBUF_COLOR_FORMAT_YUV444_709:
    case NVBUF_COLOR_FORMAT_YUV444_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_2020:
    case NVBUF_COLOR_FORMAT_YUV420_ER:
    case NVBUF_COLOR_FORMAT_YUV420_709:
    case NVBUF_COLOR_FORMAT_YUV420_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_709:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_2020:
    case NVBUF_COLOR_FORMAT_YUV444_12LE:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_ER:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_709:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_2020:
      space->inbuf_type = BUF_TYPE_YUV;
      space->insurf_count = 3;
      break;
    case NVBUF_COLOR_FORMAT_NV12:
    case NVBUF_COLOR_FORMAT_NV12_ER:
    case NVBUF_COLOR_FORMAT_NV12_709:
    case NVBUF_COLOR_FORMAT_NV12_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_10LE:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709:
    case NVBUF_COLOR_FORMAT_NV12_10LE_2020:
    case NVBUF_COLOR_FORMAT_NV12_12LE:
    case NVBUF_COLOR_FORMAT_NV12_12LE_709:
    case NVBUF_COLOR_FORMAT_NV12_12LE_2020:
      space->inbuf_type = BUF_TYPE_YUV;
      space->insurf_count = 2;
      break;
    case NVBUF_COLOR_FORMAT_UYVY:
    case NVBUF_COLOR_FORMAT_UYVY_ER:
    case NVBUF_COLOR_FORMAT_UYVY_709:
    case NVBUF_COLOR_FORMAT_UYVY_709_ER:
    case NVBUF_COLOR_FORMAT_UYVY_2020:
    case NVBUF_COLOR_FORMAT_YUYV:
    case NVBUF_COLOR_FORMAT_YVYU:
    case NVBUF_COLOR_FORMAT_UYVP:
    case NVBUF_COLOR_FORMAT_UYVP_ER:
    case NVBUF_COLOR_FORMAT_UYVP_709:
    case NVBUF_COLOR_FORMAT_UYVP_709_ER:
    case NVBUF_COLOR_FORMAT_UYVP_2020:
      space->inbuf_type = BUF_TYPE_YUV;
      space->insurf_count = 1;
      break;
    case NVBUF_COLOR_FORMAT_BGRx:
    case NVBUF_COLOR_FORMAT_RGBA:
    case NVBUF_COLOR_FORMAT_BGR:
    case NVBUF_COLOR_FORMAT_RGB:
    case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709:
    case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_2020:
    case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709:
    case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_2020:
      space->inbuf_type = BUF_TYPE_RGB;
      space->insurf_count = 1;
      break;
    case NVBUF_COLOR_FORMAT_R8_G8_B8:
    case NVBUF_COLOR_FORMAT_B8_G8_R8:
    case NVBUF_COLOR_FORMAT_R32F_G32F_B32F:
    case NVBUF_COLOR_FORMAT_B32F_G32F_R32F:
      space->inbuf_type = BUF_TYPE_RGB;
      space->insurf_count = 3;
      break;
    case NVBUF_COLOR_FORMAT_GRAY8:
    case NVBUF_COLOR_FORMAT_GRAY8_ER:
    case NVBUF_COLOR_FORMAT_GRAY16_LE:
      space->inbuf_type = BUF_TYPE_GRAY;
      space->insurf_count = 1;
      break;
    case NVBUF_COLOR_FORMAT_BGRA64_LE:
      space->inbuf_type = BUF_TYPE_RGB;
      space->insurf_count = 1;
      break;
    default:
      goto not_supported_inbuf;
      break;
  }

  min = space->num_output_buf;

  space->tsurf_width = space->to_width;
  space->tsurf_height = space->to_height;

  if ((space->in_pix_fmt != space->out_pix_fmt) ||
      (space->do_scaling) ||
      (space->do_src_cropping) ||
      (space->do_dst_cropping) || (space->flip_method)) {
    space->need_intersurf = TRUE;
    space->isurf_flag = TRUE;
  }

  switch (space->out_pix_fmt) {
    case NVBUF_COLOR_FORMAT_YUV420:
    case NVBUF_COLOR_FORMAT_YUV422:
    case NVBUF_COLOR_FORMAT_YUV444:
    case NVBUF_COLOR_FORMAT_YUV444_ER:
    case NVBUF_COLOR_FORMAT_YUV444_709:
    case NVBUF_COLOR_FORMAT_YUV444_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_2020:
    case NVBUF_COLOR_FORMAT_YUV420_ER:
    case NVBUF_COLOR_FORMAT_YUV420_709:
    case NVBUF_COLOR_FORMAT_YUV420_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_709:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_10LE_2020:
    case NVBUF_COLOR_FORMAT_YUV444_12LE:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_ER:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_709:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_709_ER:
    case NVBUF_COLOR_FORMAT_YUV444_12LE_2020:
      space->tsurf_count = 3;
      break;
    case NVBUF_COLOR_FORMAT_NV12:
    case NVBUF_COLOR_FORMAT_NV12_ER:
    case NVBUF_COLOR_FORMAT_NV12_709:
    case NVBUF_COLOR_FORMAT_NV12_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_10LE:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709:
    case NVBUF_COLOR_FORMAT_NV12_10LE_2020:
    case NVBUF_COLOR_FORMAT_NV12_12LE:
    case NVBUF_COLOR_FORMAT_NV12_12LE_709:
    case NVBUF_COLOR_FORMAT_NV12_12LE_2020:
      space->tsurf_count = 2;
      break;
    case NVBUF_COLOR_FORMAT_UYVY:
    case NVBUF_COLOR_FORMAT_UYVY_ER:
    case NVBUF_COLOR_FORMAT_UYVY_709:
    case NVBUF_COLOR_FORMAT_UYVY_709_ER:
    case NVBUF_COLOR_FORMAT_UYVY_2020:
    case NVBUF_COLOR_FORMAT_YUYV:
    case NVBUF_COLOR_FORMAT_YVYU:
      space->tsurf_count = 1;
      break;
    case NVBUF_COLOR_FORMAT_BGRx:
    case NVBUF_COLOR_FORMAT_RGBA:
    case NVBUF_COLOR_FORMAT_BGR:
    case NVBUF_COLOR_FORMAT_RGB:
    case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709:
    case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_2020:
    case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709:
    case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_2020:
      space->tsurf_count = 1;
      break;
    case NVBUF_COLOR_FORMAT_R8_G8_B8:
    case NVBUF_COLOR_FORMAT_B8_G8_R8:
    case NVBUF_COLOR_FORMAT_R32F_G32F_B32F:
    case NVBUF_COLOR_FORMAT_B32F_G32F_R32F:
      space->tsurf_count = 3;
      break;
    case NVBUF_COLOR_FORMAT_GRAY8:
    case NVBUF_COLOR_FORMAT_GRAY8_ER:
    case NVBUF_COLOR_FORMAT_GRAY16_LE:
      space->tsurf_count = 1;
      break;
    case NVBUF_COLOR_FORMAT_UYVP:
    case NVBUF_COLOR_FORMAT_UYVP_ER:
    case NVBUF_COLOR_FORMAT_UYVP_709:
    case NVBUF_COLOR_FORMAT_UYVP_709_ER:
    case NVBUF_COLOR_FORMAT_UYVP_2020:
      space->tsurf_count = 1;
      if (space->nvbuf_mem_type == NVBUF_MEM_SURFACE_ARRAY){
        printf("gstnvvideoconvert: Surface Array allocation for NVBUF_COLOR_FORMAT_UYVP(_ER/_709/_709_ER/_2020) unsupported,"\
               " changing to cuda Device");
        space->nvbuf_mem_type  = NVBUF_MEM_CUDA_DEVICE;
      }
      break;
    case NVBUF_COLOR_FORMAT_BGRA64_LE:
      space->tsurf_count = 1;
      break;

    default:
      goto not_supported_outbuf;
      break;
  }

  if ((space->do_scaling == TRUE || space->do_flip)
      && (space->in_pix_fmt == NVBUF_COLOR_FORMAT_YUV420))
    space->isurf_flag = TRUE;

  if (!gst_util_fraction_multiply (in_info.width, in_info.height, in_info.par_n,
          in_info.par_d, &from_dar_n, &from_dar_d))
    from_dar_n = from_dar_d = -1;

  if (!gst_util_fraction_multiply (out_info.width, out_info.height,
          out_info.par_n, out_info.par_d, &to_dar_n, &to_dar_d))
    to_dar_n = to_dar_d = -1;

  if (to_dar_n != from_dar_n || to_dar_d != from_dar_d)
    GST_WARNING_OBJECT (space, "Cannot keep DAR");

  /* check for outcaps feature */
  ift = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_NVMM, NULL);
  if (gst_caps_features_is_equal (gst_caps_get_features (outcaps, 0), ift))
    space->nvfilterpool = TRUE;

  gst_caps_features_free (ift);

  if (space->nvfilterpool) {
    g_mutex_lock (&space->flow_lock);
    newpool = gst_nvds_buffer_pool_new ();

    config = gst_buffer_pool_get_config (newpool);
    caps_str = gst_caps_to_string (outcaps);
    GST_DEBUG_OBJECT (space, "in videoconvert caps = %s\n", caps_str);
    if (caps_str)
      g_free (caps_str);
    gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface),
        min, min);

    gst_structure_set (config,
        "memtype", G_TYPE_UINT, space->nvbuf_mem_type,
        "gpu-id", G_TYPE_UINT, space->gpu_id,
        "batch-size", G_TYPE_UINT, space->num_batch_buffers,
        "clear-chroma", G_TYPE_BOOLEAN, 1,
        "plane-order", G_TYPE_UINT, out_rgb,
        "precision", G_TYPE_UINT, out_precision,
        "bl-output", G_TYPE_UINT, block_linear_layout_check(space),
        "contiguous-alloc", G_TYPE_BOOLEAN, space->enable_contiguous_bufs, NULL);

    if (!gst_buffer_pool_set_config (newpool, config))
      goto config_failed;

    oldpool = space->pool;
    space->pool = newpool;

    g_mutex_unlock (&space->flow_lock);

    /* unref the old nvfilter bufferpool */
    if (oldpool) {
      gst_object_unref (oldpool);
    }
  }
  /* assume bs=1 for sw */
  else if (space->outbuf_memtype == BUF_MEM_SW) {
    guint size;

    g_mutex_lock (&space->flow_lock);
    newpool = gst_buffer_pool_new ();
    config = gst_buffer_pool_get_config (newpool);

    size = get_output_frame_size (space);
    gst_buffer_pool_config_set_params (config, outcaps, size, min, min);
    if (!gst_buffer_pool_set_config (newpool, config))
      goto config_failed;
    oldpool = space->pool;
    space->pool = newpool;

    g_mutex_unlock (&space->flow_lock);
    /* unref the old nvfilter bufferpool */
    if (oldpool) {
      gst_object_unref (oldpool);
    }
  }


  if (space->do_flip ||
      (space->do_scaling) ||
      (space->do_src_cropping) || (space->do_dst_cropping) ||
      (space->do_mem_type_conversion) || (space->do_gpu_id_conversion))
    gst_base_transform_set_passthrough (btrans, FALSE);

  GST_DEBUG_OBJECT (space, "from=%dx%d (par=%d/%d dar=%d/%d), size %"
      G_GSIZE_FORMAT " -> to=%dx%d (par=%d/%d dar=%d/%d), "
      "size %" G_GSIZE_FORMAT,
      in_info.width, in_info.height, in_info.par_n, in_info.par_d,
      from_dar_n, from_dar_d, in_info.size, out_info.width,
      out_info.height, out_info.par_n, out_info.par_d, to_dar_n, to_dar_d,
      out_info.size);

  space->negotiated = ret;

  if (space->intermediate_buffer_two) {
    NvBufSurfaceDestroy (space->intermediate_buffer_two);
    space->intermediate_buffer_two = NULL;
    space->isurf_count--;
  }

  if (space->intermediate_buffer) {
    NvBufSurfaceDestroy (space->intermediate_buffer);
    space->intermediate_buffer = NULL;
    space->isurf_count--;
  }

  return ret;

  /* ERRORS */
config_failed:
  {
    GST_ERROR ("failed to set config on bufferpool");
    g_mutex_unlock (&space->flow_lock);
    return FALSE;
  }
not_supported_inbuf:
  {
    GST_ERROR ("input buffer type not supported %d\n",space->in_pix_fmt);
    return FALSE;
  }
not_supported_outbuf:
  {
    GST_ERROR ("output buffer type not supported");
    return FALSE;
  }
invalid_pix_fmt:
  {
    GST_ERROR ("could not configure for input/output format");
    space->in_pix_fmt = NVBUF_COLOR_FORMAT_INVALID;
    space->out_pix_fmt = NVBUF_COLOR_FORMAT_INVALID;
    return FALSE;
  }
invalid_caps:
  {
    GST_ERROR ("invalid caps");
    space->negotiated = FALSE;
    return FALSE;
  }
}

static gboolean
gst_nvvideoconvert_start (GstBaseTransform * btrans)
{
  Gstnvvideoconvert *space;

  space = GST_NVVIDEOCONVERT (btrans);
  space->session_created = 0;



  return TRUE;
}

static gboolean
gst_nvvideoconvert_stop (GstBaseTransform * btrans)
{
  Gstnvvideoconvert *space;

  space = GST_NVVIDEOCONVERT (btrans);

  if (space->intermediate_buffer_two) {
    NvBufSurfaceDestroy (space->intermediate_buffer_two);
    space->intermediate_buffer_two = NULL;
    space->isurf_count--;
  }

  if (space->intermediate_buffer) {
    NvBufSurfaceDestroy (space->intermediate_buffer);
    space->intermediate_buffer = NULL;
    space->isurf_count--;
  }

  if (space->pool) {
    gst_object_unref (space->pool);
    space->pool = NULL;
  }

  if (space->session_created==1){
    if (space->config_params.cuda_stream)
      cudaStreamDestroy(space->config_params.cuda_stream);
    space->config_params.cuda_stream = 0;
    space->session_created = 0;
  }

  g_free (space->transform_params.src_rect);
  g_free (space->transform_params.dst_rect);
  memset (&space->transform_params, 0, sizeof (space->transform_params));

  return TRUE;
}

static gboolean
gst_nvvideoconvert_transform_size (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize)
{
  gboolean ret = TRUE;
  GstVideoInfo vinfo = {0};

  /* size of input buffer cannot be zero */
  g_assert (size);

  ret = gst_video_info_from_caps (&vinfo, othercaps);
  if (ret) {
    *othersize = vinfo.size;
  }

  GST_DEBUG_OBJECT (btrans,
      "Othersize %" G_GSIZE_FORMAT " bytes" "for othercaps %" GST_PTR_FORMAT,
      *othersize, othercaps);

  return ret;
}

static gboolean
gst_nvvideoconvert_get_unit_size (GstBaseTransform * btrans, GstCaps * caps,
    gsize * size)
{
  gboolean ret = TRUE;
  GstVideoInfo vinfo = {0};

  if (!gst_video_info_from_caps (&vinfo, caps)) {
    GST_WARNING_OBJECT (btrans, "Parsing failed for caps %" GST_PTR_FORMAT,
        caps);
    return FALSE;
  }

  *size = vinfo.size;

  GST_DEBUG_OBJECT (btrans, "size %" G_GSIZE_FORMAT " bytes"
      "for caps %" GST_PTR_FORMAT, *size, caps);

  return ret;
}

static gboolean
subsampling_unchanged (GstVideoInfo * in_info, GstVideoInfo * out_info)
{
  const GstVideoFormatInfo *in_format, *out_format;

  if (GST_VIDEO_INFO_N_COMPONENTS (in_info) !=
      GST_VIDEO_INFO_N_COMPONENTS (out_info))
    return FALSE;

  in_format = in_info->finfo;
  out_format = out_info->finfo;

  for (guint i = 0; i < GST_VIDEO_INFO_N_COMPONENTS (in_info); i++) {
    if (GST_VIDEO_FORMAT_INFO_W_SUB (in_format,
            i) != GST_VIDEO_FORMAT_INFO_W_SUB (out_format, i))
      return FALSE;
    if (GST_VIDEO_FORMAT_INFO_H_SUB (in_format,
            i) != GST_VIDEO_FORMAT_INFO_H_SUB (out_format, i))
      return FALSE;
  }

  return TRUE;
}

static void
transfer_colorimetry_from_input (Gstnvvideoconvert * trans, GstCaps * caps,
    GstCaps * othercaps) {
  GstStructure *in_struct = gst_caps_get_structure (caps, 0);
  GstStructure *out_struct = gst_caps_get_structure (othercaps, 0);
  const GValue *in_colorimetry =
      gst_structure_get_value (in_struct, "colorimetry");

  GstVideoInfo in_info = {0}, out_info = {0};
  gst_video_info_init(&in_info);
  gst_video_info_init(&out_info);

  if (!gst_video_info_from_caps (&in_info, caps)) {
    GST_WARNING_OBJECT (trans,
        "Failed to convert sink pad caps to video info");
    return;
  }

  if (!gst_video_info_from_caps (&out_info, othercaps)) {
    GST_WARNING_OBJECT (trans,
        "Failed to convert src pad caps to video info");
    return;
  }

  if (in_colorimetry != NULL) {
    if ((GST_VIDEO_INFO_IS_YUV (&out_info)
            && GST_VIDEO_INFO_IS_YUV (&in_info))
        || (GST_VIDEO_INFO_IS_RGB (&out_info)
            && GST_VIDEO_INFO_IS_RGB (&in_info))
        || (GST_VIDEO_INFO_IS_GRAY (&out_info)
            && GST_VIDEO_INFO_IS_GRAY (&in_info))) {
      /* Can transfer the colorimetry intact from the input if it has it */
      gst_structure_set_value (out_struct, "colorimetry", in_colorimetry);
    } else {
      gchar *colorimetry_str;

      /* Changing between YUV/RGB - forward primaries and transfer function, but use
      * default range and matrix.
      * the primaries is used for conversion between RGB and XYZ (CIE 1931 coordinate).
      * the transfer function could be another reference (e.g., HDR)
      */
      out_info.colorimetry.primaries = in_info.colorimetry.primaries;
      out_info.colorimetry.transfer = in_info.colorimetry.transfer;

      colorimetry_str =
          gst_video_colorimetry_to_string (&out_info.colorimetry);
      gst_caps_set_simple (othercaps, "colorimetry", G_TYPE_STRING,
          colorimetry_str, NULL);
      g_free (colorimetry_str);
    }
  }

  /* Only YUV output needs chroma-site. If the input was also YUV and had the same chroma
  * subsampling, transfer the siting. If the sub-sampling is changing, then the planes get
  * scaled anyway so there's no real reason to prefer the input siting. */
  if (GST_VIDEO_INFO_IS_YUV (&out_info)) {
    if (GST_VIDEO_INFO_IS_YUV (&in_info)) {
      const GValue *in_chroma_site =
          gst_structure_get_value (in_struct, "chroma-site");
      if (in_chroma_site != NULL
          && subsampling_unchanged (&in_info, &out_info))
        gst_structure_set_value (out_struct, "chroma-site", in_chroma_site);
    }
  }
}

static GstCaps *
gst_nvvideoconvert_fixate_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  Gstnvvideoconvert *space;
  gint tt_width = 0, tt_height = 0;
  GstStructure *in_struct, *out_struct;
  const GValue *from_pix_ar, *to_pix_ar;
  const gchar *from_fmt = NULL, *to_fmt = NULL;
  const gchar *from_interlace_mode = NULL;
  const gchar *to_interlace_mode = NULL;
  GValue from_par = { 0, }, to_par = {
  0,};
  space = GST_NVVIDEOCONVERT (btrans);
#if 0
  gint n, i, index = 0;
  GstCapsFeatures *features = NULL;
  gboolean have_nvfeature = FALSE;

  GstCapsFeatures *ift = NULL;
  ift = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_NVMM, NULL);

  n = gst_caps_get_size (othercaps);
  for (i = 0; i < n; i++) {
    features = gst_caps_get_features (othercaps, i);
    if (gst_caps_features_is_equal (features, ift)) {
      index = i;
      have_nvfeature = TRUE;
    }
  }
  gst_caps_features_free (ift);

  if (have_nvfeature) {
    for (i = 0; i < index; i++) {
      gst_caps_remove_structure (othercaps, i);
    }
  }
#endif

  //g_print ("othercaps = %s\n", gst_caps_to_string(othercaps));
  othercaps = gst_caps_simplify (othercaps);
  othercaps = gst_caps_truncate (othercaps);
  othercaps = gst_caps_make_writable (othercaps);

  GST_DEBUG_OBJECT (space, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  in_struct = gst_caps_get_structure (caps, 0);
  out_struct = gst_caps_get_structure (othercaps, 0);

  gst_structure_set (out_struct, "block-linear", G_TYPE_BOOLEAN , space->enable_blocklinear_output, NULL);
  if (direction == GST_PAD_SINK)
  {
    // set "nvbuf-memory-type"
    const gchar *out_mem_type_string = gst_structure_get_string (out_struct, "nvbuf-memory-type");
    if(!out_mem_type_string)
    {
        int mem_type = space->nvbuf_mem_type;
        if(mem_type == NVBUF_MEM_DEFAULT)
            GET_DEFAULT_MEM_TYPE(mem_type);
        g_assert(mem_type != NVBUF_MEM_DEFAULT);
        gst_structure_set (out_struct, "nvbuf-memory-type", G_TYPE_STRING , gst_nvbuf_memory_get_name(mem_type), NULL);
    }
    else
    {
        // Get int mem-type from out_struct
        space->nvbuf_mem_type = gst_nvbuf_memory_get_value(out_mem_type_string);
        if(space->nvbuf_mem_type <= NVBUF_MEM_DEFAULT)
            GST_ERROR_OBJECT(space, "Incorrect nvbuf-memory-type set on src pad!!");
        GST_WARNING_OBJECT(space, "nvbuf-memory-type property is set based on SRC caps. Property config setting (if any) is overridden!!");
    }
    if(gst_structure_has_field(out_struct, "gpu-id"))
    {
        gst_structure_get_int(out_struct, "gpu-id", &space->gpu_id);
        GST_WARNING_OBJECT(space, "gpu-id property is set based on SRC caps. Property config setting (if any) is overridden!!");
    }
    else
    {
        gst_structure_set (out_struct, "gpu-id", G_TYPE_INT , space->gpu_id, NULL);
    }
  }

  if (gst_structure_has_field (out_struct, "interlace-mode")) {
    /* interlace-mode present */
    to_interlace_mode = gst_structure_get_string (out_struct, "interlace-mode");
    if (!to_interlace_mode) {
      /* interlace-mode not fixed */
      from_interlace_mode =
          gst_structure_get_string (in_struct, "interlace-mode");
      if (from_interlace_mode)
        gst_structure_fixate_field_string (out_struct, "interlace-mode",
            from_interlace_mode);
      else
        gst_structure_fixate_field_string (out_struct, "interlace-mode",
            "progessive");
    }
  }

  from_fmt = gst_structure_get_string (in_struct, "format");
  to_fmt = gst_structure_get_string (out_struct, "format");

  from_pix_ar = gst_structure_get_value (in_struct, "pixel-aspect-ratio");
  to_pix_ar = gst_structure_get_value (out_struct, "pixel-aspect-ratio");

  if (!to_fmt) {
    /* Output format not fixed */
    if (!gst_structure_fixate_field_string (out_struct, "format", from_fmt)) {
      GST_ERROR_OBJECT (space, "Failed to fixate output format");
      goto finish;
    }
  }

  /* If fixating from the sinkpad always set the PAR and
   * assume that missing PAR on the sinkpad means 1/1 and
   * missing PAR on the srcpad means undefined
   */
  if (direction == GST_PAD_SINK) {
    if (!from_pix_ar) {
      g_value_init (&from_par, GST_TYPE_FRACTION);
      gst_value_set_fraction (&from_par, 1, 1);
      from_pix_ar = &from_par;
    }
    if (!to_pix_ar) {
      g_value_init (&to_par, GST_TYPE_FRACTION_RANGE);
      gst_value_set_fraction_range_full (&to_par, 1, G_MAXINT, G_MAXINT, 1);
      to_pix_ar = &to_par;
    }
  } else {
    if (!to_pix_ar) {
      g_value_init (&to_par, GST_TYPE_FRACTION);
      gst_value_set_fraction (&to_par, 1, 1);
      to_pix_ar = &to_par;

      gst_structure_set (out_struct, "pixel-aspect-ratio", GST_TYPE_FRACTION, 1,
          1, NULL);
    }
    if (!from_pix_ar) {
      g_value_init (&from_par, GST_TYPE_FRACTION);
      gst_value_set_fraction (&from_par, 1, 1);
      from_pix_ar = &from_par;
    }
  }

  /* have both PAR but they might not be fixated */
  {
    gint f_width = 0, f_height = 0, f_par_n = 0, f_par_d = 0, t_par_n = 0, t_par_d = 0;
    gint t_width = 0, t_height = 0;
    gint f_dar_n = 0, f_dar_d = 0;
    gint numerator = 0, denominator = 0;

    /* from_pix_ar should be fixed */
    g_return_val_if_fail (gst_value_is_fixed (from_pix_ar), othercaps);

    f_par_n = gst_value_get_fraction_numerator (from_pix_ar);
    f_par_d = gst_value_get_fraction_denominator (from_pix_ar);

    gst_structure_get_int (in_struct, "width", &f_width);
    gst_structure_get_int (in_struct, "height", &f_height);

    gst_structure_get_int (out_struct, "width", &t_width);
    gst_structure_get_int (out_struct, "height", &t_height);

    /* if both width and height are already fixed, can't do anything
     * about it anymore */
    if (t_width && t_height) {
      guint num, den;

      GST_DEBUG_OBJECT (space, "dimensions already set to %dx%d, not fixating",
          t_width, t_height);
      if (!gst_value_is_fixed (to_pix_ar)) {
        if (gst_video_calculate_display_ratio (&num, &den, f_width, f_height,
                f_par_n, f_par_d, t_width, t_height)) {
          GST_DEBUG_OBJECT (space, "fixating to_pix_ar to %dx%d", num, den);
          if (gst_structure_has_field (out_struct, "pixel-aspect-ratio")) {
            gst_structure_fixate_field_nearest_fraction (out_struct,
                "pixel-aspect-ratio", num, den);
          } else if (num != den) {
            gst_structure_set (out_struct, "pixel-aspect-ratio",
                GST_TYPE_FRACTION, num, den, NULL);
          }
        }
      }
      goto finish;
    }

    /* Calc input DAR */
    if (!gst_util_fraction_multiply (f_width, f_height, f_par_n, f_par_d,
            &f_dar_n, &f_dar_d)) {
      GST_ERROR_OBJECT (space, "calculation of the output" "scaled size error");
      goto finish;
    }

    GST_DEBUG_OBJECT (space, "Input DAR: %d / %d", f_dar_n, f_dar_d);

    /* If either w or h are fixed either except choose a height or
     * width and PAR that matches the DAR as near as possible
     */
    if (t_width) {
      /* width is already fixed */
      gint set_par_n = 0;
      gint set_par_d = 0;

      gint s_height = 0;
      GstStructure *tmp_struct = NULL;

      /* Choose the height nearest to
       * height with same DAR, as PAR is fixed */
      if (gst_value_is_fixed (to_pix_ar)) {
        /* get PAR denominator */
        t_par_d = gst_value_get_fraction_denominator (to_pix_ar);
        /* get PAR numerator */
        t_par_n = gst_value_get_fraction_numerator (to_pix_ar);

        if (!gst_util_fraction_multiply (f_dar_n, f_dar_d,
                t_par_d, t_par_n, &numerator, &denominator)) {
          GST_ERROR_OBJECT (space, "calculation of the output"
              "scaled size error");
          goto finish;
        }

        /* calc height */
        t_height = (guint) gst_util_uint64_scale_int (t_width,
            denominator, numerator);
        /* set height */
        gst_structure_fixate_field_nearest_int (out_struct, "height", t_height);

        goto finish;
      }

      /* The PAR is not fixed set arbitrary PAR. */

      /* can keep the input height check  */
      tmp_struct = gst_structure_copy (out_struct);
      gst_structure_fixate_field_nearest_int (tmp_struct, "height", f_height);
      gst_structure_get_int (tmp_struct, "height", &s_height);

      /* May failed but try to keep the DAR however by
       * adjusting the PAR */
      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d, s_height, t_width,
              &t_par_n, &t_par_d)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        gst_structure_free (tmp_struct);
        goto finish;
      }

      if (!gst_structure_has_field (tmp_struct, "pixel-aspect-ratio")) {
        gst_structure_set_value (tmp_struct, "pixel-aspect-ratio", to_pix_ar);
      }

      /* set fixate PAR */
      if (gst_structure_fixate_field_nearest_fraction (tmp_struct,
              "pixel-aspect-ratio", t_par_n, t_par_d)) {
        /* get PAR */
        if (gst_structure_get_field_type (tmp_struct, "pixel-aspect-ratio") !=
            G_TYPE_INVALID) {
          if (!gst_structure_get_fraction (tmp_struct,
                  "pixel-aspect-ratio", &set_par_n, &set_par_d))
            GST_ERROR_OBJECT (space, "PAR values set failed");
        }
        /* values set correctly */
        if (tmp_struct) {
          gst_structure_free (tmp_struct);
          tmp_struct = NULL;
        }
      }

      if (set_par_n == t_par_n) {
        if (set_par_d == t_par_d) {
          /* Check for PAR field */
          if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
              !(set_par_n == set_par_d)) {
            /* set height & PAR */
            gst_structure_set (out_struct,
                "height", G_TYPE_INT, s_height,
                "pixel-aspect-ratio", GST_TYPE_FRACTION,
                set_par_n, set_par_d, NULL);
          }
          goto finish;
        }
      }

      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d,
              set_par_d, set_par_n, &numerator, &denominator)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        goto finish;
      }

      /* Calc height */
      t_height = (guint) gst_util_uint64_scale_int (t_width,
          denominator, numerator);
      /* Set height */
      gst_structure_fixate_field_nearest_int (out_struct, "height", t_height);

      /* If struct has field PAR then set PAR */
      if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
          !(set_par_n == set_par_d)) {
        /* set PAR */
        gst_structure_set (out_struct, "pixel-aspect-ratio", GST_TYPE_FRACTION,
            set_par_n, set_par_d, NULL);
      }
      goto finish;
    } else if (t_height) {
      /* height is already fixed */
      gint set_par_n = 0;
      gint set_par_d = 1;

      gint s_width = 0;
      GstStructure *tmp_struct = NULL;

      /* Choose the width nearest to the
       * width with same DAR, as PAR is fixed */
      if (gst_value_is_fixed (to_pix_ar)) {
        /* get PAR denominator */
        t_par_d = gst_value_get_fraction_denominator (to_pix_ar);
        /* get PAR numerator */
        t_par_n = gst_value_get_fraction_numerator (to_pix_ar);

        if (!gst_util_fraction_multiply (f_dar_n, f_dar_d, t_par_d,
                t_par_n, &numerator, &denominator)) {
          GST_ERROR_OBJECT (space, "calculation of the output"
              "scaled size error");
          goto finish;
        }

        /* calc width */
        t_width =
            (guint) gst_util_uint64_scale_int (t_height, numerator,
            denominator);
        /* set width */
        gst_structure_fixate_field_nearest_int (out_struct, "width", t_width);

        goto finish;
      }

      /* PAR is not fixed set arbitrary PAR */

      tmp_struct = gst_structure_copy (out_struct);
      gst_structure_fixate_field_nearest_int (tmp_struct, "width", f_width);
      gst_structure_get_int (tmp_struct, "width", &s_width);

      /* May failed but try to keep the DAR however by
       * adjusting the PAR */
      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d, t_height, s_width,
              &t_par_n, &t_par_d)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        gst_structure_free (tmp_struct);
        goto finish;
      }

      if (!gst_structure_has_field (tmp_struct, "pixel-aspect-ratio")) {
        gst_structure_set_value (tmp_struct, "pixel-aspect-ratio", to_pix_ar);
      }

      /* set fixate PAR */
      if (gst_structure_fixate_field_nearest_fraction (tmp_struct,
              "pixel-aspect-ratio", t_par_n, t_par_d)) {
        if (gst_structure_get_field_type (tmp_struct, "pixel-aspect-ratio") !=
            G_TYPE_INVALID) {
          /* get PAR */
          if (!gst_structure_get_fraction (tmp_struct,
                  "pixel-aspect-ratio", &set_par_n, &set_par_d))
            GST_ERROR_OBJECT (space, "PAR values set failed");
        }
        /* values set correctly */
        if (tmp_struct) {
          gst_structure_free (tmp_struct);
          tmp_struct = NULL;
        }
      }

      if (set_par_n == t_par_n) {
        if (set_par_d == t_par_d) {
          /* check for PAR field */
          if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
              !(set_par_n == set_par_d)) {
            /* set width & PAR */
            gst_structure_set (out_struct,
                "width", G_TYPE_INT, s_width,
                "pixel-aspect-ratio", GST_TYPE_FRACTION,
                set_par_n, set_par_d, NULL);
          }
          goto finish;
        }
      }

      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d,
              set_par_d, set_par_n, &numerator, &denominator)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        goto finish;
      }

      /* Calc width */
      t_width = (guint) gst_util_uint64_scale_int (t_height,
          numerator, denominator);

      /* Set width */
      gst_structure_fixate_field_nearest_int (out_struct, "width", t_width);

      /* If struct has field PAR then set PAR */
      if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
          !(set_par_n == set_par_d)) {
        /* set PAR */
        gst_structure_set (out_struct, "pixel-aspect-ratio", GST_TYPE_FRACTION,
            set_par_n, set_par_d, NULL);
      }

      goto finish;
    } else if (gst_value_is_fixed (to_pix_ar)) {

      gint s_height = 0;
      gint s_width = 0;
      gint from_hight = 0;
      gint from_width = 0;
      GstStructure *tmp_struct = NULL;

      /* Get PAR denominator */
      t_par_d = gst_value_get_fraction_denominator (to_pix_ar);
      /* Get PAR numerator */
      t_par_n = gst_value_get_fraction_numerator (to_pix_ar);

      /* find scale factor for change in PAR */
      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d,
              t_par_n, t_par_d, &numerator, &denominator)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        goto finish;
      }

      tmp_struct = gst_structure_copy (out_struct);

      gst_structure_fixate_field_nearest_int (tmp_struct, "height", f_height);
      gst_structure_get_int (tmp_struct, "height", &s_height);

      /* This may failed but however scale the width to keep DAR */
      t_width =
          (guint) gst_util_uint64_scale_int (s_height, numerator, denominator);
      gst_structure_fixate_field_nearest_int (tmp_struct, "width", t_width);
      gst_structure_get_int (tmp_struct, "width", &s_width);
      gst_structure_free (tmp_struct);

      /* kept DAR and the height is nearest to the original height */
      if (s_width == t_width) {
        gst_structure_set (out_struct, "width", G_TYPE_INT, s_width, "height",
            G_TYPE_INT, s_height, NULL);
        goto finish;
      }

      from_hight = s_height;
      from_width = s_width;

      /* If former failed, try to keep the input width at least */
      tmp_struct = gst_structure_copy (out_struct);
      gst_structure_fixate_field_nearest_int (tmp_struct, "width", f_width);
      gst_structure_get_int (tmp_struct, "width", &s_width);

      /* This may failed but however try to scale the width to keep DAR */
      t_height =
          (guint) gst_util_uint64_scale_int (s_width, denominator, numerator);
      gst_structure_fixate_field_nearest_int (tmp_struct, "height", t_height);
      gst_structure_get_int (tmp_struct, "height", &s_height);
      gst_structure_free (tmp_struct);

      /* We kept the DAR and the width is nearest to the original width */
      if (s_height == t_height) {
        gst_structure_set (out_struct, "width", G_TYPE_INT, s_width, "height",
            G_TYPE_INT, s_height, NULL);
        goto finish;
      }

      /* If all failed, keep the height that nearest to the orignal
       * height and the nearest possible width.
       */
      gst_structure_set (out_struct, "width", G_TYPE_INT, from_width, "height",
          G_TYPE_INT, from_hight, NULL);
      goto finish;
    } else {
      gint tmp_struct2 = 0;
      gint set_par_n = 0;
      gint set_par_d = 0;
      gint s_height = 0;
      gint s_width = 0;
      GstStructure *tmp_struct = NULL;

      /* width, height and PAR are not fixed though passthrough impossible */

      /* keep height and width as fine as possible & scale PAR */
      tmp_struct = gst_structure_copy (out_struct);

      if (gst_structure_fixate_field_nearest_int (tmp_struct, "height",
              f_height))
        gst_structure_get_int (tmp_struct, "height", &s_height);

      if (gst_structure_fixate_field_nearest_int (tmp_struct, "width", f_width))
        gst_structure_get_int (tmp_struct, "width", &s_width);

      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d,
              s_height, s_width, &t_par_n, &t_par_d)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        goto finish;
      }

      if (!gst_structure_has_field (tmp_struct, "pixel-aspect-ratio")) {
        gst_structure_set_value (tmp_struct, "pixel-aspect-ratio", to_pix_ar);
      }

      if (gst_structure_fixate_field_nearest_fraction (tmp_struct,
              "pixel-aspect-ratio", t_par_n, t_par_d)) {
        gst_structure_get_fraction (tmp_struct, "pixel-aspect-ratio",
            &set_par_n, &set_par_d);
      }
      gst_structure_free (tmp_struct);

      if (set_par_n == t_par_n) {
        if (set_par_d == t_par_d) {
          gst_structure_set (out_struct,
              "width", G_TYPE_INT, s_width,
              "height", G_TYPE_INT, s_height, NULL);

          if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
              !(set_par_n == set_par_d))
            gst_structure_set (out_struct, "pixel-aspect-ratio",
                GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
          space->no_dimension = TRUE;
          goto finish;
        }
      }

      /* Or scale width to keep the DAR with the set
       * PAR and height */
      if (!gst_util_fraction_multiply (f_dar_n, f_dar_d,
              set_par_d, set_par_n, &numerator, &denominator)) {
        GST_ERROR_OBJECT (space, "calculation of the output"
            "scaled size error");
        goto finish;
      }

      t_width =
          (guint) gst_util_uint64_scale_int (s_height, numerator, denominator);
      tmp_struct = gst_structure_copy (out_struct);

      if (gst_structure_fixate_field_nearest_int (tmp_struct, "width", t_width)) {
        gst_structure_get_int (tmp_struct, "width", &tmp_struct2);
      }
      gst_structure_free (tmp_struct);

      if (tmp_struct2 == t_width) {
        gst_structure_set (out_struct,
            "width", G_TYPE_INT, tmp_struct2,
            "height", G_TYPE_INT, s_height, NULL);
        if (gst_structure_has_field (out_struct, "pixel-aspect-ratio")
            || (set_par_n != set_par_d))
          gst_structure_set (out_struct, "pixel-aspect-ratio",
              GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
        space->no_dimension = TRUE;
        goto finish;
      }

      t_height =
          (guint) gst_util_uint64_scale_int (s_width, denominator, numerator);
      tmp_struct = gst_structure_copy (out_struct);

      if (gst_structure_fixate_field_nearest_int (tmp_struct, "height",
              t_height)) {
        gst_structure_get_int (tmp_struct, "height", &tmp_struct2);
      }
      gst_structure_free (tmp_struct);

      if (tmp_struct2 == t_height) {
        gst_structure_set (out_struct,
            "width", G_TYPE_INT, s_width,
            "height", G_TYPE_INT, tmp_struct2, NULL);
        if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
            set_par_n != set_par_d)
          gst_structure_set (out_struct, "pixel-aspect-ratio",
              GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
        space->no_dimension = TRUE;
        goto finish;
      }

      /* If all failed can't keep DAR & take nearest values for all */
      gst_structure_set (out_struct,
          "width", G_TYPE_INT, s_width, "height", G_TYPE_INT, s_height, NULL);
      if (gst_structure_has_field (out_struct, "pixel-aspect-ratio") ||
          (set_par_n != set_par_d))
        gst_structure_set (out_struct, "pixel-aspect-ratio", GST_TYPE_FRACTION,
            set_par_n, set_par_d, NULL);
      space->no_dimension = TRUE;
    }
  }

finish:
  if (space->no_dimension && space->do_flip) {
    switch (space->flip_method) {
      case GST_VIDEO_NVFLIP_METHOD_90R:
      case GST_VIDEO_NVFLIP_METHOD_90L:
      case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
      case GST_VIDEO_NVFLIP_METHOD_TRANS:
        if (gst_structure_get_int (out_struct, "width", &tt_width) &&
            gst_structure_get_int (out_struct, "height", &tt_height)) {
          gst_structure_set (out_struct, "width", G_TYPE_INT, tt_height,
              "height", G_TYPE_INT, tt_width, NULL);
        }
        break;
      case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
      case GST_VIDEO_NVFLIP_METHOD_180:
      case GST_VIDEO_NVFLIP_METHOD_HORIZ:
      case GST_VIDEO_NVFLIP_METHOD_VERT:
        break;
      default:
        g_assert_not_reached ();
        break;
    }
  }

  if (from_pix_ar == &from_par)
    g_value_unset (&from_par);
  if (to_pix_ar == &to_par)
    g_value_unset (&to_par);

  if (direction == GST_PAD_SINK) {
    /* Try and preserve input colorimetry / chroma information */
    gboolean have_colorimetry =
        gst_structure_has_field (out_struct, "colorimetry");
    gboolean have_chroma_site =
        gst_structure_has_field (out_struct, "chroma-site");

    /* If the output already has colorimetry and chroma-site, stop,
    * otherwise try and transfer what we can from the input caps */
    if (!have_colorimetry && !have_chroma_site) {
      transfer_colorimetry_from_input (space, caps, othercaps);
    }
  }

  GST_DEBUG_OBJECT (space, "fixated othercaps to %" GST_PTR_FORMAT, othercaps);

  return othercaps;
}

static GstCaps *
gst_nvvideoconvert_transform_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCaps *ret = NULL;
  GstCaps *tmp1, *tmp2;
  GstCapsFeatures *features = NULL;

  GST_DEBUG_OBJECT (btrans,
      "Transforming caps %" GST_PTR_FORMAT " in direction %s", caps,
      (direction == GST_PAD_SINK) ? "sink" : "src");

  /* Get all possible caps that we can transform into */
  tmp1 = gst_nvvideoconvert_caps_remove_format_info (caps);

  if (filter) {
    if (direction == GST_PAD_SRC) {
      GstCapsFeatures *ift = NULL;
      ift = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_NVMM, NULL);
      features = gst_caps_get_features (filter, 0);
      if (!gst_caps_features_is_equal (features, ift)) {
        gint n, i;
        GstCapsFeatures *tft;
        n = gst_caps_get_size (tmp1);
        for (i = 0; i < n; i++) {
          tft = gst_caps_get_features (tmp1, i);
          if (gst_caps_features_get_size (tft))
            gst_caps_features_remove (tft, GST_CAPS_FEATURE_MEMORY_NVMM);
        }
        tmp2 = gst_nvvideoconvert_caps_remove_batchsize(tmp1);
        gst_caps_unref (tmp1);
        tmp1 = tmp2;
      }
      gst_caps_features_free (ift);
    }

    tmp2 = gst_caps_intersect_full (filter, tmp1, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp1);
    tmp1 = tmp2;
  }

  if (gst_caps_is_empty (tmp1)) {
    gst_caps_unref (tmp1);
    ret = gst_caps_copy (filter);
  } else
    ret = tmp1;

  if (!filter) {
    GstStructure *str;
    str = gst_structure_copy (gst_caps_get_structure (ret, 0));

    GstCapsFeatures *ift;
    ift = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_NVMM, NULL);

    gst_caps_append_structure_full (ret, str, ift);

    str = gst_structure_copy (gst_caps_get_structure (ret, 0));
    gst_caps_append_structure_full (ret, str, NULL);
  }

  GST_DEBUG_OBJECT (btrans, "transformed %" GST_PTR_FORMAT " into %"
      GST_PTR_FORMAT, caps, ret);

  return ret;
}

static gboolean
gst_nvvideoconvert_accept_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps)
{
  gboolean ret = TRUE;
  Gstnvvideoconvert *space = NULL;
  GstCaps *allowed = NULL;

  space = GST_NVVIDEOCONVERT (btrans);

  GST_DEBUG_OBJECT (btrans, "accept caps %" GST_PTR_FORMAT, caps);

  /* get all the formats we can handle on this pad */
  if (direction == GST_PAD_SINK)
    allowed = space->sinkcaps;
  else
    allowed = space->srccaps;

  if (!allowed) {
    GST_DEBUG_OBJECT (btrans, "failed to get allowed caps");
    GST_DEBUG_OBJECT (btrans,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    ret = FALSE;
    return ret;
  }

  GST_DEBUG_OBJECT (btrans, "allowed caps %" GST_PTR_FORMAT, allowed);

  /* intersect with the requested format */
  ret = gst_caps_is_subset (caps, allowed);
  if (!ret) {
    GST_DEBUG_OBJECT (btrans,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
  }

  return ret;
}

static gboolean
gst_nvvideoconvert_decide_allocation (GstBaseTransform * btrans,
    GstQuery * query)
{
  guint j, metas_no;
  Gstnvvideoconvert *space = NULL;
  GstCaps *outcaps = NULL;
  GstCaps *myoutcaps = NULL;
  GstBufferPool *pool = NULL;
  guint size, minimum, maximum;
  GstAllocator *allocator = NULL;
  GstAllocationParams params = { 0, 0, 0, 0 };
  GstStructure *config = NULL;
  GstVideoInfo info;
  gboolean modify_allocator;
  GstQuery *bsquery = NULL;
  guint batch_size = 0;

  space = GST_NVVIDEOCONVERT (btrans);

  bsquery = gst_nvquery_batch_size_new ();
  if (space->num_batch_buffers == 1) {
    if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), bsquery)) {
      gst_nvquery_batch_size_parse (bsquery, &batch_size);
      space->num_batch_buffers = batch_size;
    }
  }
  /* This is possible when upstream component has not implemented query */
  if (space->num_batch_buffers == 0) {
    space->num_batch_buffers = 1;
  }
  gst_query_unref (bsquery);


  g_free (space->transform_params.src_rect);
  g_free (space->transform_params.dst_rect);
  memset (&space->transform_params, 0, sizeof (NvBufSurfTransformParams));
  space->transform_params.src_rect =
      (NvBufSurfTransformRect *) g_malloc0 (space->num_batch_buffers *
      sizeof (NvBufSurfTransformRect));
  space->transform_params.dst_rect =
      (NvBufSurfTransformRect *) g_malloc0 (space->num_batch_buffers *
      sizeof (NvBufSurfTransformRect));
  metas_no = gst_query_get_n_allocation_metas (query);
  for (j = 0; j < metas_no; j++) {
    gboolean remove_meta;
    GType meta_api;
    const GstStructure *param_str = NULL;

    meta_api = gst_query_parse_nth_allocation_meta (query, j, &param_str);

    if (gst_meta_api_type_has_tag (meta_api, GST_META_TAG_MEMORY)) {
      /* Different memory will get allocated for input and output.
         remove all memory dependent metadata */
      GST_DEBUG_OBJECT (space, "remove memory specific metadata %s",
          g_type_name (meta_api));
      remove_meta = TRUE;
    } else {
      /* Default remove all metadata */
      GST_DEBUG_OBJECT (space, "remove metadata %s", g_type_name (meta_api));
      remove_meta = TRUE;
    }

    if (remove_meta) {
      gst_query_remove_nth_allocation_meta (query, j);
      j--;
      metas_no--;
    }
  }

  gst_query_parse_allocation (query, &outcaps, NULL);
  if (outcaps == NULL)
    goto no_caps;

  /* Use nvfilter custom buffer pool */
  if (space->nvfilterpool) {
    g_mutex_lock (&space->flow_lock);
    pool = space->pool;
    if (pool)
      gst_object_ref (pool);
    g_mutex_unlock (&space->flow_lock);

    if (pool != NULL) {
      config = gst_buffer_pool_get_config (pool);
      gst_buffer_pool_config_get_params (config, &myoutcaps, &size, NULL, NULL);
      gst_structure_get_uint (config, "batch-size", &batch_size);

      GST_DEBUG_OBJECT (space, "we have a pool with caps %" GST_PTR_FORMAT,
          myoutcaps);

      if (!gst_caps_is_equal (outcaps, myoutcaps) ||
          batch_size != space->num_batch_buffers) {
        /* different caps, we can't use current pool */
        GST_DEBUG_OBJECT (space, "pool has different caps");
        gst_object_unref (pool);
        pool = NULL;
      }
      gst_structure_free (config);
    }

    if (pool == NULL) {
      if (!gst_video_info_from_caps (&info, outcaps))
        goto invalid_caps;

      size = info.size;
      minimum = space->num_output_buf;

      GST_DEBUG_OBJECT (space, "create new pool");

      g_mutex_lock (&space->flow_lock);
      pool = gst_nvds_buffer_pool_new ();

      config = gst_buffer_pool_get_config (pool);
      GST_DEBUG_OBJECT (space, "in videoconvert caps = %s\n",
          gst_caps_to_string (outcaps));
      gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface),
          minimum, minimum);

      gst_structure_set (config,
          "memtype", G_TYPE_UINT, space->nvbuf_mem_type,
          "gpu-id", G_TYPE_UINT, space->gpu_id,
          "batch-size", G_TYPE_UINT, space->num_batch_buffers,
          "clear-chroma", G_TYPE_BOOLEAN, 1,
          "bl-output", G_TYPE_UINT, block_linear_layout_check(space),
          "contiguous-alloc", G_TYPE_BOOLEAN, space->enable_contiguous_bufs, NULL);

      if (!gst_buffer_pool_set_config (pool, config))
        goto config_failed;

      space->pool = gst_object_ref (pool);

      g_mutex_unlock (&space->flow_lock);
    }

    if (pool) {
      config = gst_buffer_pool_get_config (pool);
      gst_buffer_pool_config_get_allocator (config, &allocator, &params);
      gst_buffer_pool_config_get_params (config, &myoutcaps, &size, &minimum,
          &maximum);

      /* Add check, params may be empty e.g. fakesink */
      if (gst_query_get_n_allocation_params (query) > 0) {
        /* Set allocation params */
        gst_query_set_nth_allocation_param (query, 0, allocator, &params);
      } else {
        /* Add allocation params */
        gst_query_add_allocation_param (query, allocator, &params);
      }

      /* Set allocation pool */
      if (gst_query_get_n_allocation_pools (query) > 0) {
        gst_query_set_nth_allocation_pool (query, 0, pool, size, minimum,
            maximum);
      } else {
        gst_query_add_allocation_pool (query, pool, size, minimum, maximum);
      }

      gst_structure_free (config);
      gst_object_unref (pool);
    }
  } else {
    /* Use oss buffer pool */
    if (gst_query_get_n_allocation_params (query) > 0) {
      /* Get allocation params */
      gst_query_parse_nth_allocation_param (query, 0, &allocator, &params);
      modify_allocator = TRUE;
    } else {
      allocator = NULL;
      gst_allocation_params_init (&params);
      modify_allocator = FALSE;
    }

    if (gst_query_get_n_allocation_pools (query) > 0) {
      /* Parse pool to get size, min & max  */
      gst_query_parse_nth_allocation_pool (query, 0, &pool, &size, &minimum,
          &maximum);
      if (pool == NULL) {
        GST_DEBUG_OBJECT (btrans, "no pool available, creating new oss pool");
        pool = gst_buffer_pool_new ();
      }
    } else {
      pool = NULL;
      size = 0;
      minimum = 0;
      maximum = 0;
    }

    if (pool) {
      config = gst_buffer_pool_get_config (pool);
      /* Set params on config */
      gst_buffer_pool_config_set_params (config, outcaps, size, minimum,
          maximum);
      /* Set allocator on config */
      gst_buffer_pool_config_set_allocator (config, allocator, &params);
      /* Set config on pool */
      gst_buffer_pool_set_config (pool, config);
    }

    if (modify_allocator) {
      /* Set allocation params */
      gst_query_set_nth_allocation_param (query, 0, allocator, &params);
    } else {
      /* Add allocation params */
      gst_query_add_allocation_param (query, allocator, &params);
    }

    if (allocator) {
      gst_object_unref (allocator);
    }

    if (pool) {
      gst_query_set_nth_allocation_pool (query, 0, pool, size, minimum,
          maximum);
      gst_object_unref (pool);
    }
  }

  return TRUE;
/* ERROR */
no_caps:
  {
    GST_ERROR ("no caps specified");
    return FALSE;
  }
invalid_caps:
  {
    GST_ERROR ("invalid caps specified");
    return FALSE;
  }
config_failed:
  {
    GST_ERROR ("failed to set config on bufferpool");
    g_mutex_unlock (&space->flow_lock);
    return FALSE;
  }
}

static NvDsBatchMeta *
gst_buffer_get_nvds_batch_meta_int (GstBuffer * buffer)
{
  gpointer state = NULL;
  GstMeta *gst_meta;
  NvDsBatchMeta *batch_meta = NULL;

  while ((gst_meta = gst_buffer_iterate_meta (buffer, &state))) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      continue;
    }
    NvDsMeta *dsmeta = (NvDsMeta *) gst_meta;

    if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
      if (batch_meta != NULL) {
        GST_WARNING ("Multiple NvDsBatchMeta found on buffer %p", buffer);
      }
      batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
    }
  }

  return batch_meta;
}

void
update_meta (NvDsBatchMeta * batch_meta, uint32_t icnt,
    Gstnvvideoconvert * space, gfloat from_width, gfloat from_height, gfloat to_width, gfloat to_height)
{
  NvDsFrameMeta *frame_meta = NULL;

  gfloat scale_factor_width = 1, scale_factor_height = 1;

  scale_factor_width = to_width / from_width;
  scale_factor_height = to_height / from_height;

  // Lock not required since its on o/p buffer owned by plugin
  // nvds_acquire_meta_lock (batch_meta);
  if(batch_meta->frame_meta_list == NULL)
    return;
  frame_meta =
      (NvDsFrameMeta *) g_list_nth_data (batch_meta->frame_meta_list, icnt);
  for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
      l_obj = l_obj->next) {
    NvDsObjectMeta *object_meta = (NvDsObjectMeta *) (l_obj->data);
    if (space->flip_method) {
      flip_object(object_meta, space->flip_method, from_width, from_height);
      //update the scale factor if height and width are fliped
      switch (space->flip_method) {
        case GST_VIDEO_NVFLIP_METHOD_90L:
        case GST_VIDEO_NVFLIP_METHOD_90R:
        case GST_VIDEO_NVFLIP_METHOD_TRANS:
        case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
          scale_factor_width = to_width / from_height;
          scale_factor_height = to_height / from_width;
          break;
        default:
          break;
      }
    }
    scale_object(&(object_meta->rect_params),&(object_meta->text_params),space,scale_factor_width,scale_factor_height);
  }

  /* setting the offset according to src and destination crop*/
  float offset_left = -(scale_factor_width * space->src_crop_left);
  float offset_top= -(scale_factor_height * space->src_crop_top);

  /** Scale all display_meta_list in-place */
  for (NvDisplayMetaList * d_obj = frame_meta->display_meta_list; d_obj != NULL;
      d_obj = d_obj->next) {
      NvDsDisplayMeta *display_meta = (NvDsDisplayMeta *) (d_obj->data);
      if(display_meta)
      {
        for(uint32_t i = 0; i < display_meta->num_rects; i++)
        {
            ScaleRectParams(&display_meta->rect_params[i], &space->transform_params.dst_rect[icnt], scale_factor_width, scale_factor_height, offset_left, offset_top);
        }
        for(uint32_t i = 0; i < display_meta->num_labels; i++)
        {
            ScaleTextParams(&display_meta->text_params[i], &space->transform_params.dst_rect[icnt], scale_factor_width, scale_factor_height, offset_left, offset_top);
        }
        for(uint32_t i = 0; i < display_meta->num_lines; i++)
        {
            ScaleLineParams(&display_meta->line_params[i], &space->transform_params.dst_rect[icnt], scale_factor_width, scale_factor_height, offset_left, offset_top);
        }
        for(uint32_t i = 0; i < display_meta->num_circles; i++)
        {
            ScaleCircleParams(&display_meta->circle_params[i], &space->transform_params.dst_rect[icnt], scale_factor_width, scale_factor_height, offset_left, offset_top);
        }
        for(uint32_t i = 0; i < display_meta->num_arrows; i++)
        {
            ScaleArrowParams(&display_meta->arrow_params[i], &space->transform_params.dst_rect[icnt], scale_factor_width, scale_factor_height, offset_left, offset_top);
        }
      }
  }
  // nvds_release_meta_lock (batch_meta);
}

static void flip_object(NvDsObjectMeta *object_meta, gint flip_method, gfloat from_width, gfloat from_height)
{
  gfloat temp;
  switch (flip_method) {
    case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
      break;
    case GST_VIDEO_NVFLIP_METHOD_90L:
      //x axis and y axis to be updated like x -> y and y -> -x for a point
      //which is equivalent to x -> y and y -> frame_width - rectangle_width - x and swapping the height and width for rectangle
      temp=object_meta->rect_params.top;
      object_meta->rect_params.top=(from_width - object_meta->rect_params.left - object_meta->rect_params.width);
      object_meta->rect_params.left=temp;
      temp=object_meta->rect_params.width;
      object_meta->rect_params.width=object_meta->rect_params.height;
      object_meta->rect_params.height=temp;
      object_meta->text_params.x_offset=object_meta->rect_params.left;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      break;
    case GST_VIDEO_NVFLIP_METHOD_180:
      //x axis and y axis to be updated like x -> -x and y -> -y for a point
      //which is equivalent to x -> frame_width - rectangle_width - x and y -> frame_height - rectangle_height - y for rectangle
      object_meta->rect_params.left=from_width - object_meta->rect_params.left - object_meta->rect_params.width;
      object_meta->rect_params.top=from_height - object_meta->rect_params.top - object_meta->rect_params.height;
      object_meta->text_params.x_offset=object_meta->rect_params.left;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      break;
    case GST_VIDEO_NVFLIP_METHOD_90R:
      //x axis and y axis to be updated like x -> -y and y -> x for a point
      //which is equivalent to x -> frame_height - rectangle_height - y and y -> x and swapping the height and width for rectangle
      temp=object_meta->rect_params.left;
      object_meta->rect_params.left=(from_height - object_meta->rect_params.top - object_meta->rect_params.height);
      object_meta->rect_params.top=temp;
      temp=object_meta->rect_params.height;
      object_meta->rect_params.height=object_meta->rect_params.width;
      object_meta->rect_params.width=temp;
      object_meta->text_params.x_offset=object_meta->rect_params.left;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      break;
    case GST_VIDEO_NVFLIP_METHOD_HORIZ:
      //only x axis is to be updated like x -> -x for a point
      //which is equivalent to x -> frame_width - rectangle_width - x for rectangle
      object_meta->rect_params.left=from_width - object_meta->rect_params.left - object_meta->rect_params.width;
      object_meta->text_params.x_offset=object_meta->rect_params.left;
      break;
    case GST_VIDEO_NVFLIP_METHOD_VERT:
      //only y axis is to be updated like y -> -y for a point
      //which is equivalent to y -> frame_height - rectangle_height - y for rectangle
      object_meta->rect_params.top=from_height - object_meta->rect_params.top - object_meta->rect_params.height;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      break;
    case GST_VIDEO_NVFLIP_METHOD_TRANS:
      //flip the object 1st by 90L
      temp=object_meta->rect_params.top;
      object_meta->rect_params.top=(from_width - object_meta->rect_params.left - object_meta->rect_params.width);
      object_meta->rect_params.left=temp;
      temp=object_meta->rect_params.width;
      object_meta->rect_params.width=object_meta->rect_params.height;
      object_meta->rect_params.height=temp;
      object_meta->text_params.x_offset=object_meta->rect_params.left;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      //then flip the object again vertically
      object_meta->rect_params.top=from_width - object_meta->rect_params.top - object_meta->rect_params.height;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      break;
    case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
    //flip the object 1st by 90R
      temp=object_meta->rect_params.left;
      object_meta->rect_params.left=(from_height - object_meta->rect_params.top - object_meta->rect_params.height);
      object_meta->rect_params.top=temp;
      temp=object_meta->rect_params.height;
      object_meta->rect_params.height=object_meta->rect_params.width;
      object_meta->rect_params.width=temp;
      object_meta->text_params.x_offset=object_meta->rect_params.left;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      //then flip the object again horizontally
      object_meta->rect_params.top=from_width - object_meta->rect_params.top - object_meta->rect_params.height;
      object_meta->text_params.y_offset=object_meta->rect_params.top - object_meta->text_params.font_params.font_size;
      break;
    default:
      break;
  }
}

static void scale_object(NvOSD_RectParams* rect_params, NvOSD_TextParams* text_params, Gstnvvideoconvert * space, gfloat scale_factor_width, gfloat scale_factor_height)
{
  rect_params->left =
      scale_factor_width * (rect_params->left - space->src_crop_left) + space->dst_crop_left;
  rect_params->top =
      scale_factor_height * (rect_params->top - space->src_crop_top) + space->dst_crop_top;
  rect_params->width =
      scale_factor_width * rect_params->width;
  rect_params->height =
      scale_factor_height * rect_params->height;
  text_params->x_offset =
      scale_factor_width * (text_params->x_offset - space->src_crop_left) + space->dst_crop_left;
  text_params->y_offset =
      scale_factor_height * (text_params->y_offset - space->src_crop_top) + space->dst_crop_top;
  text_params->font_params.font_size =
      scale_factor_height * text_params->font_params.font_size;
}

void ScaleRectParams(NvOSD_RectParams* rect_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top)
{
    /** scale width and height of rects */
    rect_params->width  *= scaleX;
    rect_params->height *= scaleY;
    /** scale and place position co-ordinates */
    rect_params->left = ((rect_params->left * scaleX)
                              + destRect->left + offset_left);
    rect_params->top = ((rect_params->top * scaleY)
                              + destRect->top + offset_top);
}

void ScaleTextParams(NvOSD_TextParams* text_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top)
{
    text_params->x_offset = ((text_params->x_offset * scaleX)
                              + destRect->left + offset_left);
    text_params->y_offset = ((text_params->y_offset * scaleY)
                              + destRect->top + offset_top);
}

void ScaleLineParams(NvOSD_LineParams* line_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top)
{
    line_params->x1 = ((line_params->x1 * scaleX)
                              + destRect->left + offset_left);
    line_params->x2 = ((line_params->x2 * scaleX)
                              + destRect->left + offset_left);
    line_params->y1 = ((line_params->y1 * scaleY)
                              + destRect->top + offset_top);
    line_params->y2 = ((line_params->y2 * scaleY)
                              + destRect->top + offset_top);
}

void ScaleCircleParams(NvOSD_CircleParams* circle_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top)
{
    float min_scale = (scaleX > scaleY) ? scaleY : scaleX;
    circle_params->xc = ((circle_params->xc * scaleX)
                                + destRect->left + offset_left);
    circle_params->yc = ((circle_params->yc * scaleY)
                                + destRect->top + offset_top);
    circle_params->radius = circle_params->radius * min_scale;
}

void ScaleArrowParams(NvOSD_ArrowParams* arrow_params, NvBufSurfTransformRect* destRect, float scaleX, float scaleY, float offset_left, float offset_top)
{
    arrow_params->x1 = ((arrow_params->x1 * scaleX)
                              + destRect->left + offset_left);
    arrow_params->x2 = ((arrow_params->x2 * scaleX)
                              + destRect->left + offset_left);
    arrow_params->y1 = ((arrow_params->y1 * scaleY)
                              + destRect->top + offset_top);
    arrow_params->y2 = ((arrow_params->y2 * scaleY)
                              + destRect->top + offset_top);
}

static void
surface_list_init (Gstnvvideoconvert * space, NvBufSurface * surf,
    GstMapInfo * surf_list_map, gboolean flag, GstVideoMeta *video_meta)
{
  gpointer surf_list_data = surf_list_map->data;
  guint bytesPerPixel;
  gsize surf_list_size = surf_list_map->size;

  surf->gpuId = space->gpu_id;
  surf->batchSize = 1;
  surf->numFilled = 1;
  surf->memType = NVBUF_MEM_SYSTEM;
  GstVideoInfo *surf_list_info = NULL;
  /* Flag is set as 1 if input surface needs to be populated and to 0 if output
   * surface has to be populated.
   */
  if (flag) {
    bytesPerPixel = get_bytes_per_pix_from_color (space->in_pix_fmt, 0);
    surf->surfaceList->planeParams.num_planes = space->in_info.finfo->n_planes;
    if (!video_meta){
      surf->surfaceList->pitch = space->in_info.stride[0];
    }
    else{
      surf->surfaceList->pitch = video_meta->stride[0];
    }
    surf->surfaceList->colorFormat = space->in_pix_fmt;
    surf->surfaceList->planeParams.offset[0] = space->in_info.offset[0];
    surf->surfaceList->width = space->from_width;
    surf->surfaceList->height = space->from_height;
    surf->surfaceList->planeParams.width[0] = space->from_width;
    surf->surfaceList->planeParams.height[0] = space->from_height;
    surf->surfaceList->planeParams.psize[0] =
        space->from_height * surf->surfaceList->pitch;
    surf_list_info = &(space->in_info);
  } else {
    bytesPerPixel = get_bytes_per_pix_from_color (space->out_pix_fmt, 0);
    surf->surfaceList->planeParams.num_planes = space->out_info.finfo->n_planes;

    if (!video_meta){
      surf->surfaceList->pitch = space->out_info.stride[0];
    }
    else {
      surf->surfaceList->pitch = video_meta->stride[0];
    }
    surf->surfaceList->colorFormat = space->out_pix_fmt;
    surf->surfaceList->planeParams.offset[0] = space->out_info.offset[0];
    surf->surfaceList->width = space->to_width;
    surf->surfaceList->height = space->to_height;
    surf->surfaceList->planeParams.width[0] = space->to_width;
    surf->surfaceList->planeParams.height[0] = space->to_height;
    surf->surfaceList->planeParams.psize[0] =
        space->to_height * surf->surfaceList->pitch;
    surf_list_info = &(space->out_info);
  }
  surf->surfaceList->dataSize = surf_list_size; // size of allocated hw mem
  surf->surfaceList->dataPtr = surf_list_data;
  surf->surfaceList->layout = NVBUF_LAYOUT_PITCH;
  surf->surfaceList->planeParams.pitch[0] = surf->surfaceList->pitch;
  surf->surfaceList->planeParams.bytesPerPix[0] = bytesPerPixel;

  for (uint32_t j = 1; j < surf_list_info->finfo->n_planes; j++) {
    guint comp_width = GST_VIDEO_INFO_COMP_WIDTH (surf_list_info, j);
    guint comp_height = GST_VIDEO_INFO_COMP_HEIGHT (surf_list_info, j);
    guint comp_pitch = surf_list_info->stride[j];
    if (video_meta){
      comp_pitch = video_meta->stride[j];
    }
    surf->surfaceList->planeParams.height[j] = comp_height;
    surf->surfaceList->planeParams.pitch[j] = comp_pitch;
    surf->surfaceList->planeParams.offset[j] = surf_list_info->offset[j];
    surf->surfaceList->planeParams.psize[j] = comp_pitch * comp_height;
    bytesPerPixel =
        get_bytes_per_pix_from_color (surf->surfaceList->colorFormat, j);
    surf->surfaceList->planeParams.width[j] = comp_width;
    surf->surfaceList->planeParams.bytesPerPix[j] = bytesPerPixel;
  }
}
static NvBufSurfTransform_Error CopySurfTransform(NvBufSurface* src, NvBufSurface* dest,
        NvBufSurfTransformConfigParams* p_config_params, NvBufSurfTransform_Compute copy_hw)
{
    NvBufSurfTransform_Error status;
    NvBufSurfTransformParams transformParams;
    NvBufSurfTransformRect srcRect;
    NvBufSurfTransformRect destRect;
    NvBufSurfTransform_Compute compute_mode_orig = NvBufSurfTransformCompute_GPU;
    //nvtx_helper_push_pop ("copy");
#if defined(__aarch64__)
    if ((copy_hw != NvBufSurfTransformCompute_VIC) &&
        ((src->memType == NVBUF_MEM_SURFACE_ARRAY) || (dest->memType == NVBUF_MEM_SURFACE_ARRAY)) &&
        ((src->surfaceList[0].colorFormat == NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709) ||
        (src->surfaceList[0].colorFormat == NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_2020) ||
        (dest->surfaceList[0].colorFormat == NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709) ||
        (dest->surfaceList[0].colorFormat == NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_2020))) {
           GST_DEBUG ("BGR10A2 Format transformation is not supported for nvbuf-mem-surface-array"
                        " memory type for GPU. Forcing copy-hw as VIC \n");
           copy_hw = NvBufSurfTransformCompute_VIC;
    }

    if(copy_hw == NvBufSurfTransformCompute_VIC)
        status = NvBufSurfaceCopy (src, dest);
    else
#endif
    {
    // Set the compute mode to GPU for copying
    if(p_config_params->compute_mode != NvBufSurfTransformCompute_GPU)
    {
        compute_mode_orig = p_config_params->compute_mode;
        p_config_params->compute_mode = NvBufSurfTransformCompute_GPU;
        status = NvBufSurfTransformSetSessionParams (p_config_params);
        if (status != NvBufSurfTransformError_Success) {
            GST_ERROR ("Set session params failed \n");
            return status;
        }
    }
    srcRect.top = srcRect.left = 0;
    destRect.top = destRect.left = 0;
    srcRect.width   = src->surfaceList[0].width;
    srcRect.height  = src->surfaceList[0].height;
    destRect.width  = dest->surfaceList[0].width;
    destRect.height = dest->surfaceList[0].height;
    transformParams.src_rect = &srcRect;
    transformParams.dst_rect = &destRect;
    transformParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transformParams.transform_flip = NvBufSurfTransform_None;
    transformParams.transform_filter = NvBufSurfTransformInter_Nearest;
    status = NvBufSurfTransform(src, dest, &transformParams);
    if(compute_mode_orig != NvBufSurfTransformCompute_GPU)
    {
        // Set the tranform session back to original compute mode
        p_config_params->compute_mode = compute_mode_orig;
        status = NvBufSurfTransformSetSessionParams (p_config_params);
        if (status != NvBufSurfTransformError_Success) {
            GST_ERROR ("Set session params failed \n");
            return status;
        }
    }
    }
    //nvtx_helper_push_pop (NULL);
    return status;
}

static GstFlowReturn
gst_nvvideoconvert_transform (GstBaseTransform * btrans, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  //gint retn = 0;
  //gboolean ret = TRUE;
  gint status = 0;
  GstFlowReturn flow_ret = GST_FLOW_OK;
  NvBufSurfTransform_Error tx_err;

  Gstnvvideoconvert *space = NULL;
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;

  GstMemory *inmem = NULL;
  GstMemory *outmem = NULL;
  //GstNvFilterMemory *omem = NULL;

  GstMapInfo inmap = GST_MAP_INFO_INIT;
  GstMapInfo outmap = GST_MAP_INFO_INIT;
  GstVideoMeta* in_vid_meta = NULL;

  //gint input_dmabuf_fd = -1;
  //NvBufferParams inbuf_params = {0};
  //NvBufSurfaceCreateParams input_params = {0};

  gpointer data = NULL;
#ifndef NEW_METADATA
#if defined(__aarch64__)
  /* Get metadata. Update rectangle and text params */
  GstMeta *gst_meta;
  IvaMeta *meta;
  NvDsMeta *dsmeta;
  unsigned int i = 0;
  gpointer state = NULL;
#endif
#endif

  space = GST_NVVIDEOCONVERT (btrans);
  char context[100];
  sprintf (context, "gst_nvvideoconvert_transform()_ctx=%p", space);
#if defined(__aarch64__)
  nvtx_helper_push_pop (context);
#endif
  cudaError_t CUerr = cudaSuccess;

  if (G_UNLIKELY (!space->negotiated))
    goto unknown_format;

  inmem = gst_buffer_peek_memory (inbuf, 0);
  if (!inmem)
    goto no_memory;

  outmem = gst_buffer_peek_memory (outbuf, 0);
  if (!outmem)
    goto no_memory;
  //omem = (GstNvFilterMemory *) outmem;

  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ))
    goto invalid_inbuf;

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
    goto invalid_outbuf;

  nvds_set_input_system_timestamp (outbuf, GST_ELEMENT_NAME (space));
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta_int (outbuf);
  if (batch_meta == NULL) {
    GST_DEBUG_OBJECT (btrans, "NvDsBatchMeta not found for output buffer");
  }

  CUerr = cudaSetDevice (space->gpu_id);
  if (CUerr != cudaSuccess) {
    printf ("\n *** Unable to set device in %s Line %d\n", __func__, __LINE__);
    goto cuda_errors;
  }


  data =
      gst_mini_object_get_qdata ((GstMiniObject *) inbuf,
      g_quark_from_static_string ("NV_BUF"));

  if (data == (gpointer) NVBUF_MAGIC_NUM) {
    space->inbuf_memtype = BUF_MEM_HW;
  }
  else {
    in_vid_meta = gst_buffer_get_video_meta (inbuf);
  }

  if (space->session_created == 0)
  {
    int is_nvgpu = 0;
    NvBufSurfaceDeviceInfo dev_info = {0};
    if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
        is_nvgpu = 1;
      }
    }
    space->config_params.compute_mode = space->compute_hw;
    space->config_params.gpu_id = space->gpu_id;
    if (space->compute_hw == NvBufSurfTransformCompute_GPU
       || space->copy_hw == NvBufSurfTransformCompute_GPU
       || !is_nvgpu)
      cudaStreamCreateWithFlags (&(space->config_params.cuda_stream),
        cudaStreamNonBlocking);
    space->session_created = 1;
  }

  err = NvBufSurfTransformSetSessionParams (&space->config_params);
  if (err != NvBufSurfTransformError_Success) {
    printf ("Set session params failed \n");
    return FALSE;
  }

  space->transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  if (space->do_src_cropping)
    space->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_CROP_SRC;
  if (space->do_dst_cropping)
    space->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_CROP_DST;
  if(space->do_flip)
    space->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_FLIP;
  if(space->allow_odd_crop)
    space->transform_params.transform_flag |= NVBUFSURF_TRANSFORM_ALLOW_ODD_CROP;
  get_NvBufferTransform(space);

  switch (space->inbuf_type) {
    case BUF_TYPE_GRAY:
    case BUF_TYPE_YUV:
    case BUF_TYPE_RGB:
      if (space->inbuf_memtype == BUF_MEM_HW
          && space->outbuf_memtype == BUF_MEM_SW) {
        NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
        NvBufSurface op_surf;
        NvBufSurfaceParams surfaceList;
        memset (&surfaceList, 0, sizeof (surfaceList));
        op_surf.surfaceList = &surfaceList;
        surface_list_init (space, &op_surf, &outmap, 0, NULL);

        if (inmap.size != sizeof (NvBufSurface)) {
          GST_ERROR ("Input buffer is not NvBufSurface");
          goto transform_err;
        }

        if (CHECK_NVDS_MEMORY_AND_GPUID (space, ip_surf)) {
          goto transform_err;
        }
        if (ip_surf->surfaceList[0].colorFormat != space->out_pix_fmt) {
          space->need_intersurf = TRUE;
          space->isurf_flag = TRUE;
        }

        if (ip_surf->numFilled == 0)
          ip_surf->numFilled = 1;
        if (space->need_intersurf) {
          if (space->isurf_count < 1) {
            NvBufSurfaceCreateParams buf_params = { 0 };
            buf_params.width = space->to_width;
            buf_params.height = space->to_height;
            buf_params.gpuId = space->gpu_id;
            buf_params.colorFormat = space->out_pix_fmt;
            /* NOTE: This can cause perf issues if two elememnts are on same thread using
             * different gpu-id
             */
            buf_params.memType = space->nvbuf_mem_type;
            if (block_linear_layout_check(space))
              buf_params.layout =  NVBUF_LAYOUT_BLOCK_LINEAR;
            else
              buf_params.layout =   NVBUF_LAYOUT_PITCH;
            status =
                NvBufSurfaceCreate (&space->intermediate_buffer, 1,
                &buf_params);
            if (status < 0) {
              g_print ("%s: intermediate NvBufferCreate Failed \n", __func__);
              flow_ret = GST_FLOW_ERROR;
              goto done;
            }
            if (space->do_dst_cropping) {
              if (space->intermediate_buffer->surfaceList[0].
                  planeParams.num_planes > 1) {
                for (uint32_t j = 1;
                    j <
                    space->intermediate_buffer->surfaceList[0].
                    planeParams.num_planes; j++)
                  NvBufSurfaceMemSet (space->intermediate_buffer, 0, j, 128);
              }
            }
            space->isurf_count++;
          }
          for (uint32_t icnt = 0; icnt < ip_surf->numFilled; icnt++) {
            if (!space->do_src_cropping) {
              space->transform_params.src_rect[icnt].top = 0;
              space->transform_params.src_rect[icnt].left = 0;
              space->transform_params.src_rect[icnt].width =
                  ip_surf->surfaceList[icnt].width;
              space->transform_params.src_rect[icnt].height =
                  ip_surf->surfaceList[icnt].height;
            } else {
              space->transform_params.src_rect[icnt].top = space->src_crop_top;
              space->transform_params.src_rect[icnt].left =
                  space->src_crop_left;
              space->transform_params.src_rect[icnt].width =
                  space->src_crop_width;
              space->transform_params.src_rect[icnt].height =
                  space->src_crop_height;
            }

            if (!space->do_dst_cropping) {
              space->transform_params.dst_rect[icnt].top = 0;
              space->transform_params.dst_rect[icnt].left = 0;
              space->transform_params.dst_rect[icnt].width = space->to_width;
              space->transform_params.dst_rect[icnt].height = space->to_height;
            } else {
              space->transform_params.dst_rect[icnt].top = space->dst_crop_top;
              space->transform_params.dst_rect[icnt].left =
                  space->dst_crop_left;
              space->transform_params.dst_rect[icnt].width =
                  space->dst_crop_width;
              space->transform_params.dst_rect[icnt].height =
                  space->dst_crop_height;
            }
            if (batch_meta)
              update_meta (batch_meta, icnt, space, space->transform_params.src_rect[icnt].width, space->transform_params.src_rect[icnt].height, space->transform_params.dst_rect[icnt].width, space->transform_params.dst_rect[icnt].height);
          }

          space->transform_params.transform_filter =
              space->interpolation_method;
          tx_err =
              NvBufSurfTransform (ip_surf, space->intermediate_buffer,
              &space->transform_params);
          if (tx_err != NvBufSurfTransformError_Success) {
            goto transform_err;
          }
           CopySurfTransform (space->intermediate_buffer, &op_surf, &space->config_params, space->copy_hw);
        } else
        {
           CopySurfTransform (ip_surf, &op_surf, &space->config_params, space->copy_hw);
        }
      } else if (space->inbuf_memtype == BUF_MEM_SW
          && space->outbuf_memtype == BUF_MEM_HW) {
        NvBufSurface ip_surf;
        NvBufSurface *op_surf = (NvBufSurface *) outmap.data;
        NvBufSurfaceParams surfaceList;
        memset (&surfaceList, 0, sizeof (surfaceList));
        ip_surf.surfaceList = &surfaceList;
        surface_list_init (space, &ip_surf, &inmap, 1, in_vid_meta);

        if (CHECK_NVDS_MEMORY_AND_GPUID (space, op_surf)) {
          goto transform_err;
        }

        if (space->need_intersurf) {
          if (space->isurf_count < 1) {
            NvBufSurfaceCreateParams buf_params = { 0 };
            buf_params.width = space->from_width;
            buf_params.height = space->from_height;
            buf_params.gpuId = space->gpu_id;
            buf_params.colorFormat = space->in_pix_fmt;
            /* NOTE: This can cause perf issues if two elememnts are on same thread using
             * different gpu-id
             */
            buf_params.memType = space->nvbuf_mem_type;
            if (block_linear_layout_check(space))
              buf_params.layout =  NVBUF_LAYOUT_BLOCK_LINEAR;
            else
              buf_params.layout =   NVBUF_LAYOUT_PITCH;
            status =
                NvBufSurfaceCreate (&space->intermediate_buffer, 1,
                &buf_params);
            if (status < 0) {
              g_print ("%s: intermediate NvBufferCreate Failed \n", __func__);
              flow_ret = GST_FLOW_ERROR;
              goto done;
            }
            space->isurf_count++;
          }
          CopySurfTransform (&ip_surf, space->intermediate_buffer, &space->config_params, space->copy_hw);
          space->intermediate_buffer->numFilled = 1;
          for (uint32_t icnt = 0; icnt < ip_surf.numFilled; icnt++) {
            if (!space->do_src_cropping) {
              space->transform_params.src_rect[icnt].top = 0;
              space->transform_params.src_rect[icnt].left = 0;
              space->transform_params.src_rect[icnt].width =
                  ip_surf.surfaceList[icnt].width;
              space->transform_params.src_rect[icnt].height =
                  ip_surf.surfaceList[icnt].height;
            } else {
              space->transform_params.src_rect[icnt].top = space->src_crop_top;
              space->transform_params.src_rect[icnt].left =
                  space->src_crop_left;
              space->transform_params.src_rect[icnt].width =
                  space->src_crop_width;
              space->transform_params.src_rect[icnt].height =
                  space->src_crop_height;
            }

            if (!space->do_dst_cropping) {
              space->transform_params.dst_rect[icnt].top = 0;
              space->transform_params.dst_rect[icnt].left = 0;
              space->transform_params.dst_rect[icnt].width = space->to_width;
              space->transform_params.dst_rect[icnt].height = space->to_height;
            } else {
              space->transform_params.dst_rect[icnt].top = space->dst_crop_top;
              space->transform_params.dst_rect[icnt].left =
                  space->dst_crop_left;
              space->transform_params.dst_rect[icnt].width =
                  space->dst_crop_width;
              space->transform_params.dst_rect[icnt].height =
                  space->dst_crop_height;
            }
            if (batch_meta)
              update_meta (batch_meta, icnt, space, space->transform_params.src_rect[icnt].width, space->transform_params.src_rect[icnt].height, space->transform_params.dst_rect[icnt].width, space->transform_params.dst_rect[icnt].height);
          }

          space->transform_params.transform_filter =
              space->interpolation_method;
          tx_err =
              NvBufSurfTransform (space->intermediate_buffer, op_surf,
              &space->transform_params);
          if (tx_err != NvBufSurfTransformError_Success) {
            goto transform_err;
          }
        } else {
          CopySurfTransform (&ip_surf, op_surf, &space->config_params, space->copy_hw);
          op_surf->numFilled = 1;
        }
      } else if (space->inbuf_memtype == BUF_MEM_SW
          && space->outbuf_memtype == BUF_MEM_SW) {
        /* input surface initialisations */
        NvBufSurface ip_surf[1];
        NvBufSurfaceParams surfaceListIp;
        memset (&surfaceListIp, 0, sizeof (surfaceListIp));
        ip_surf[0].surfaceList = &surfaceListIp;
        surface_list_init (space, &ip_surf[0], &inmap, 1, in_vid_meta);

        /* output surface initialisations */
        NvBufSurface op_surf[1];
        NvBufSurfaceParams surfaceListOp;
        memset (&surfaceListOp, 0, sizeof (surfaceListOp));
        op_surf[0].surfaceList = &surfaceListOp;
        surface_list_init (space, &op_surf[0], &outmap, 0, NULL);

        if (space->need_intersurf) {
          /* Input */
          if (space->isurf_count < 1) {
            NvBufSurfaceCreateParams buf_params_ip = { 0 };
            buf_params_ip.width = space->from_width;
            buf_params_ip.height = space->from_height;
            buf_params_ip.gpuId = space->gpu_id;
            buf_params_ip.colorFormat = space->in_pix_fmt;
            /* NOTE: This can cause perf issues if two elememnts are on same thread using
             * different gpu-id
             */
            buf_params_ip.memType = space->nvbuf_mem_type;
            if (((space->in_pix_fmt == NVBUF_COLOR_FORMAT_UYVP) ||
                (space->in_pix_fmt == NVBUF_COLOR_FORMAT_UYVP_ER) ||
                (space->in_pix_fmt == NVBUF_COLOR_FORMAT_UYVP_709) ||
                (space->in_pix_fmt == NVBUF_COLOR_FORMAT_UYVP_709_ER) ||
                (space->in_pix_fmt == NVBUF_COLOR_FORMAT_UYVP_2020)) &&
                 space->nvbuf_mem_type == NVBUF_MEM_SURFACE_ARRAY){
                buf_params_ip.memType = NVBUF_MEM_CUDA_DEVICE;
            }
            if (block_linear_layout_check(space))
              buf_params_ip.layout =  NVBUF_LAYOUT_BLOCK_LINEAR;
            else
              buf_params_ip.layout =   NVBUF_LAYOUT_PITCH;
            status =
                NvBufSurfaceCreate (&space->intermediate_buffer, 1,
                &buf_params_ip);
            if (status < 0) {
              g_print ("%s: intermediate NvBufferCreate Failed \n", __func__);
              flow_ret = GST_FLOW_ERROR;
              goto done;
            }
            space->isurf_count++;
          }
          if (space->isurf_count < 2) {
            /* output */
            NvBufSurfaceCreateParams buf_params_op = { 0 };
            buf_params_op.width = space->to_width;
            buf_params_op.height = space->to_height;
            buf_params_op.gpuId = space->gpu_id;
            buf_params_op.colorFormat = space->out_pix_fmt;
            /* NOTE: This can cause perf issues if two elememnts are on same thread using
             * different gpu-id
             */
            buf_params_op.memType = space->nvbuf_mem_type;

            if (block_linear_layout_check(space))
              buf_params_op.layout =  NVBUF_LAYOUT_BLOCK_LINEAR;
            else
              buf_params_op.layout =   NVBUF_LAYOUT_PITCH;
            status =
                NvBufSurfaceCreate (&space->intermediate_buffer_two, 1,
                &buf_params_op);
            if (status < 0) {
              g_print ("%s: intermediate NvBufferCreate Failed \n", __func__);
              flow_ret = GST_FLOW_ERROR;
              goto done;
            }
            if (space->do_dst_cropping) {
              if (space->intermediate_buffer_two->surfaceList[0].
                  planeParams.num_planes > 1) {
                for (uint32_t j = 1;
                    j <
                    space->intermediate_buffer_two->surfaceList[0].
                    planeParams.num_planes; j++)
                  NvBufSurfaceMemSet (space->intermediate_buffer_two, 0, j,
                      128);
              }
            }
            space->isurf_count++;
          }
          CopySurfTransform (&ip_surf[0], space->intermediate_buffer, &space->config_params, space->copy_hw);
          space->intermediate_buffer->numFilled = 1;
          for (uint32_t icnt = 0; icnt < ip_surf[0].numFilled; icnt++) {
            if (!space->do_src_cropping) {
              space->transform_params.src_rect[icnt].top = 0;
              space->transform_params.src_rect[icnt].left = 0;
              space->transform_params.src_rect[icnt].width =
                  ip_surf[0].surfaceList[icnt].width;
              space->transform_params.src_rect[icnt].height =
                  ip_surf[0].surfaceList[icnt].height;
            } else {
              space->transform_params.src_rect[icnt].top = space->src_crop_top;
              space->transform_params.src_rect[icnt].left =
                  space->src_crop_left;
              space->transform_params.src_rect[icnt].width =
                  space->src_crop_width;
              space->transform_params.src_rect[icnt].height =
                  space->src_crop_height;
            }

            if (!space->do_dst_cropping) {
              space->transform_params.dst_rect[icnt].top = 0;
              space->transform_params.dst_rect[icnt].left = 0;
              space->transform_params.dst_rect[icnt].width = space->to_width;
              space->transform_params.dst_rect[icnt].height = space->to_height;
            } else {
              space->transform_params.dst_rect[icnt].top = space->dst_crop_top;
              space->transform_params.dst_rect[icnt].left =
                  space->dst_crop_left;
              space->transform_params.dst_rect[icnt].width =
                  space->dst_crop_width;
              space->transform_params.dst_rect[icnt].height =
                  space->dst_crop_height;
            }
            if (batch_meta)
              update_meta (batch_meta, icnt, space, space->transform_params.src_rect[icnt].width, space->transform_params.src_rect[icnt].height, space->transform_params.dst_rect[icnt].width, space->transform_params.dst_rect[icnt].height);
          }
          space->transform_params.transform_filter =
              space->interpolation_method;
          tx_err =
              NvBufSurfTransform (space->intermediate_buffer,
              space->intermediate_buffer_two, &space->transform_params);
          space->intermediate_buffer_two->numFilled = 1;
          if (tx_err != NvBufSurfTransformError_Success) {
            goto transform_err;
          }
          CopySurfTransform (space->intermediate_buffer_two, &op_surf[0], &space->config_params, space->copy_hw);
        } else {
           CopySurfTransform (&ip_surf[0], &op_surf[0], &space->config_params, space->copy_hw);
        }
      } else if (space->inbuf_memtype == BUF_MEM_HW
          && space->outbuf_memtype == BUF_MEM_HW) {
        NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
        NvBufSurface *op_surf = (NvBufSurface *) outmap.data;
        if (inmap.size != sizeof (NvBufSurface)) {
          GST_ERROR ("Input buffer is not NvBufSurface");
          goto transform_err;
        }

        if (ip_surf->numFilled == 0)
          ip_surf->numFilled = 1;

        for (uint32_t icnt = 0; icnt < ip_surf->numFilled; icnt++) {
          if (!space->do_src_cropping) {
            space->transform_params.src_rect[icnt].top = 0;
            space->transform_params.src_rect[icnt].left = 0;
            space->transform_params.src_rect[icnt].width =
                ip_surf->surfaceList[icnt].width;
            space->transform_params.src_rect[icnt].height =
                ip_surf->surfaceList[icnt].height;
          } else {
            space->transform_params.src_rect[icnt].top = space->src_crop_top;
            space->transform_params.src_rect[icnt].left = space->src_crop_left;
            space->transform_params.src_rect[icnt].width =
                space->src_crop_width;
            space->transform_params.src_rect[icnt].height =
                space->src_crop_height;
          }

          if (!space->do_dst_cropping) {
            space->transform_params.dst_rect[icnt].top = 0;
            space->transform_params.dst_rect[icnt].left = 0;
            space->transform_params.dst_rect[icnt].width =
                op_surf->surfaceList[icnt].width;
            space->transform_params.dst_rect[icnt].height =
                op_surf->surfaceList[icnt].height;
          } else {
            space->transform_params.dst_rect[icnt].top = space->dst_crop_top;
            space->transform_params.dst_rect[icnt].left = space->dst_crop_left;
            space->transform_params.dst_rect[icnt].width =
                space->dst_crop_width;
            space->transform_params.dst_rect[icnt].height =
                space->dst_crop_height;
          }
          if (batch_meta)
            update_meta (batch_meta, icnt, space, space->transform_params.src_rect[icnt].width, space->transform_params.src_rect[icnt].height, space->transform_params.dst_rect[icnt].width, space->transform_params.dst_rect[icnt].height);
        }

        space->transform_params.transform_filter = space->interpolation_method;
        tx_err =
            NvBufSurfTransform (ip_surf, op_surf, &space->transform_params);
        if (tx_err != NvBufSurfTransformError_Success)
          goto transform_err;
      } else {
        flow_ret = GST_FLOW_ERROR;
        goto done;
      }
      break;

    default:
      GST_ERROR ("%s: Unsupported input buffer \n", __func__);
      flow_ret = GST_FLOW_ERROR;
      goto done;
      break;
  }

#ifndef NEW_METADATA            /* TODO */
#if defined(__aarch64__)
  while ((gst_meta = gst_buffer_iterate_meta (outbuf, &state))) {
    meta = (IvaMeta *) gst_meta;
    if (gst_meta_api_type_has_tag (gst_meta->info->api, _ivameta_quark)
        && meta->meta_type == NV_BBOX_INFO) {
      BBOX_Params *bbox_params = (BBOX_Params *) (meta->meta_data);

      for (i = 0; i < bbox_params->num_rects; i++) {
        NvOSD_RectParams *rect = &bbox_params->roi_meta[i].rect_params;
        int tmp;

        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
            tmp = rect->left;
            rect->left = space->from_height - rect->top - rect->height;
            rect->top = tmp;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_90L:
            tmp = rect->top;
            rect->top = space->from_width - rect->left - rect->width;
            rect->left = tmp;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
            tmp = rect->top;
            rect->top = space->from_width - rect->left - rect->width;
            rect->left = space->from_height - tmp - rect->height;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            tmp = rect->left;
            rect->left = rect->top;
            rect->top = tmp;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
            break;
          case GST_VIDEO_NVFLIP_METHOD_180:
            rect->top = space->from_height - rect->top - rect->height;
            rect->left = space->from_width - rect->left - rect->width;
            break;
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
            rect->left = space->from_width - rect->left - rect->width;
            break;
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            rect->top = space->from_height - rect->top - rect->height;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
          case GST_VIDEO_NVFLIP_METHOD_90L:
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            rect->width = rect->width * space->to_height / space->from_width;
            rect->left = rect->left * space->to_height / space->from_width;
            rect->top = rect->top * space->to_width / space->from_height;
            rect->height = rect->height * space->to_width / space->from_height;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
          case GST_VIDEO_NVFLIP_METHOD_180:
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            rect->width = rect->width * space->to_width / space->from_width;
            rect->left = rect->left * space->to_width / space->from_width;
            rect->top = rect->top * space->to_height / space->from_height;
            rect->height = rect->height * space->to_height / space->from_height;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
      }

      for (i = 0; i < bbox_params->num_strings; i++) {
        NvOSD_TextParams *text = &bbox_params->roi_meta[i].text_params;
        int tmp;

        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
            tmp = text->x_offset;
            text->x_offset = space->from_height - text->y_offset;
            text->y_offset = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_90L:
            tmp = text->y_offset;
            text->y_offset = space->from_width - text->x_offset;
            text->x_offset = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
            tmp = text->y_offset;
            text->y_offset = space->from_width - text->x_offset;
            text->x_offset = space->from_height - tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            tmp = text->x_offset;
            text->x_offset = text->y_offset;
            text->y_offset = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
            break;
          case GST_VIDEO_NVFLIP_METHOD_180:
            text->y_offset = space->from_height - text->y_offset;
            text->x_offset = space->from_width - text->x_offset;
            break;
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
            text->x_offset = space->from_width - text->x_offset;
            break;
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            text->y_offset = space->from_height - text->y_offset;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
          case GST_VIDEO_NVFLIP_METHOD_90L:
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            text->x_offset =
                text->x_offset * space->to_height / space->from_width;
            text->y_offset =
                text->y_offset * space->to_width / space->from_height;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
          case GST_VIDEO_NVFLIP_METHOD_180:
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            text->x_offset =
                text->x_offset * space->to_width / space->from_width;
            text->y_offset =
                text->y_offset * space->to_height / space->from_height;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
      }
    }
    dsmeta = (NvDsMeta *) gst_meta;
    if (gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)
        && meta->meta_type == NVDS_META_FRAME_INFO) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (dsmeta->meta_data);

      for (i = 0; i < frame_meta->num_rects; i++) {
        NvOSD_RectParams *rect = &frame_meta->obj_params[i].rect_params;
        int tmp;

        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
            tmp = rect->left;
            rect->left = space->from_height - rect->top - rect->height;
            rect->top = tmp;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_90L:
            tmp = rect->top;
            rect->top = space->from_width - rect->left - rect->width;
            rect->left = tmp;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
            tmp = rect->top;
            rect->top = space->from_width - rect->left - rect->width;
            rect->left = space->from_height - tmp - rect->height;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            tmp = rect->left;
            rect->left = rect->top;
            rect->top = tmp;
            tmp = rect->width;
            rect->width = rect->height;
            rect->height = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
            break;
          case GST_VIDEO_NVFLIP_METHOD_180:
            rect->top = space->from_height - rect->top - rect->height;
            rect->left = space->from_width - rect->left - rect->width;
            break;
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
            rect->left = space->from_width - rect->left - rect->width;
            break;
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            rect->top = space->from_height - rect->top - rect->height;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
          case GST_VIDEO_NVFLIP_METHOD_90L:
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            rect->width = rect->width * space->to_height / space->from_width;
            rect->left = rect->left * space->to_height / space->from_width;
            rect->top = rect->top * space->to_width / space->from_height;
            rect->height = rect->height * space->to_width / space->from_height;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
          case GST_VIDEO_NVFLIP_METHOD_180:
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            rect->width = rect->width * space->to_width / space->from_width;
            rect->left = rect->left * space->to_width / space->from_width;
            rect->top = rect->top * space->to_height / space->from_height;
            rect->height = rect->height * space->to_height / space->from_height;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
      }

      for (i = 0; i < frame_meta->num_strings; i++) {
        NvOSD_TextParams *text = &frame_meta->obj_params[i].text_params;
        int tmp;

        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
            tmp = text->x_offset;
            text->x_offset = space->from_height - text->y_offset;
            text->y_offset = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_90L:
            tmp = text->y_offset;
            text->y_offset = space->from_width - text->x_offset;
            text->x_offset = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
            tmp = text->y_offset;
            text->y_offset = space->from_width - text->x_offset;
            text->x_offset = space->from_height - tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            tmp = text->x_offset;
            text->x_offset = text->y_offset;
            text->y_offset = tmp;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
            break;
          case GST_VIDEO_NVFLIP_METHOD_180:
            text->y_offset = space->from_height - text->y_offset;
            text->x_offset = space->from_width - text->x_offset;
            break;
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
            text->x_offset = space->from_width - text->x_offset;
            break;
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            text->y_offset = space->from_height - text->y_offset;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
        switch (space->flip_method) {
          case GST_VIDEO_NVFLIP_METHOD_90R:
          case GST_VIDEO_NVFLIP_METHOD_90L:
          case GST_VIDEO_NVFLIP_METHOD_INVTRANS:
          case GST_VIDEO_NVFLIP_METHOD_TRANS:
            text->x_offset =
                text->x_offset * space->to_height / space->from_width;
            text->y_offset =
                text->y_offset * space->to_width / space->from_height;
            break;
          case GST_VIDEO_NVFLIP_METHOD_IDENTITY:
          case GST_VIDEO_NVFLIP_METHOD_180:
          case GST_VIDEO_NVFLIP_METHOD_HORIZ:
          case GST_VIDEO_NVFLIP_METHOD_VERT:
            text->x_offset =
                text->x_offset * space->to_width / space->from_width;
            text->y_offset =
                text->y_offset * space->to_height / space->from_height;
            break;
          default:
            g_assert_not_reached ();
            break;
        }
      }
    }
  }
#endif
#endif

  nvds_set_output_system_timestamp (outbuf, GST_ELEMENT_NAME (space));
done:

  gst_buffer_unmap (inbuf, &inmap);
  gst_buffer_unmap (outbuf, &outmap);
#if defined(__aarch64__)
  nvtx_helper_push_pop (NULL);
#endif

  return flow_ret;

  /* ERRORS */
transform_err:
  {
    nvds_set_output_system_timestamp (outbuf, GST_ELEMENT_NAME (space));
    gst_buffer_unmap (inbuf, &inmap);
    gst_buffer_unmap (outbuf, &outmap);
    GST_ERROR ("buffer transform failed");
    return GST_FLOW_ERROR;
  }
no_memory:
  {
    GST_ERROR ("no memory block");
    return GST_FLOW_ERROR;
  }
unknown_format:
  {
    GST_ERROR ("unknown format");
    return GST_FLOW_NOT_NEGOTIATED;
  }
invalid_inbuf:
  {
    GST_ERROR ("input buffer mapinfo failed");
    return GST_FLOW_ERROR;
  }
invalid_outbuf:
  {
    GST_ERROR ("output buffer mapinfo failed");
    gst_buffer_unmap (inbuf, &inmap);
    return GST_FLOW_ERROR;
  }
cuda_errors:
  {
    GST_ERROR ("Set Device failed");
    gst_buffer_unmap (inbuf, &inmap);
    return GST_FLOW_ERROR;
  }
}

static gboolean
nvvideoconvert_init (GstPlugin * nvvconv)
{
  GstDebugLevel level;
  GST_DEBUG_CATEGORY_INIT (gst_nvvideoconvert_debug, "nvvideoconvert",
      0, "nvvideoconvert plugin");

  level = gst_debug_category_get_threshold (gst_nvvideoconvert_debug);
  if (level < GST_LEVEL_WARNING)
    gst_debug_category_set_threshold (gst_nvvideoconvert_debug,
        GST_LEVEL_WARNING);

  _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
  return gst_element_register (nvvconv, "nvvideoconvert", GST_RANK_PRIMARY,
      GST_TYPE_NVVIDEOCONVERT);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvvideoconvert,
    PACKAGE_DESCRIPTION,
    nvvideoconvert_init, VERSION, PACKAGE_LICENSE, PACKAGE_NAME, PACKAGE_URL)
