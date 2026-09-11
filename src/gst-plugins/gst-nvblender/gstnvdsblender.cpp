/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "gstnvdsblender.h"
#include "gstnvdsblenderpad.h"
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"

#define SW_IN_HW_OUT 0
using namespace std;

/* Package and library details required for plugin_init */
#define PACKAGE "nvblender"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA composite/blender Plugin to be used with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "NVIDIA DeepStream composite/blender plugin"
#define URL "http://nvidia.com/"

GST_DEBUG_CATEGORY_STATIC (gst_nvblender_debug);
#define GST_CAT_DEFAULT gst_nvblender_debug

#define FORMATS " { NV12, RGBA } "
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{NV12, RGBA, I420}") ";" GST_VIDEO_CAPS_MAKE (FORMATS)
        ));

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{NV12, RGBA, I420}") ";"
        GST_VIDEO_CAPS_MAKE (FORMATS))
    );

#define DEFAULT_PAD_XPOS   0
#define DEFAULT_PAD_YPOS   0
#define DEFAULT_PAD_WIDTH  0
#define DEFAULT_PAD_HEIGHT 0
#define DEFAULT_PAD_ALPHA  1.0
#define DEFAULT_PAD_CROSSFADE_RATIO  -1.0
enum
{
  PROP_PAD_0,
  PROP_PAD_XPOS,
  PROP_PAD_YPOS,
  PROP_PAD_WIDTH,
  PROP_PAD_HEIGHT,
  PROP_PAD_ALPHA,
  PROP_PAD_CROSSFADE_RATIO,
};

G_DEFINE_TYPE (GstNvblenderPad, gst_nvblender_pad,
    GST_TYPE_VIDEO_AGGREGATOR_PAD);

#define NVDS_USER_BATCH_META_AIGS \
  (nvds_get_user_meta_type((char *)"NVIDIA.NVVFXAIGS.USER_META"))

typedef struct AiGsSourceData {
  // segmentation matte from Aigs
  GstBuffer *gstBuffer;
  // list of source_id which has AIGS enabled
  uint32_t *arrayList;
  //total sources with AIGS enabled
  uint32_t numSources;
}AiGsSourceData;


static void
gst_nvblender_pad_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvblenderPad *pad = GST_NVBLENDER_PAD (object);

  switch (prop_id) {
    case PROP_PAD_XPOS:
      g_value_set_int (value, pad->xpos);
      break;
    case PROP_PAD_YPOS:
      g_value_set_int (value, pad->ypos);
      break;
    case PROP_PAD_WIDTH:
      g_value_set_int (value, pad->width);
      break;
    case PROP_PAD_HEIGHT:
      g_value_set_int (value, pad->height);
      break;
    case PROP_PAD_ALPHA:
      g_value_set_double (value, pad->alpha);
      break;
    case PROP_PAD_CROSSFADE_RATIO:
      g_value_set_double (value, pad->crossfade);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvblender_pad_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvblenderPad *pad = GST_NVBLENDER_PAD (object);

  switch (prop_id) {
    case PROP_PAD_XPOS:
      pad->xpos = g_value_get_int (value);
      break;
    case PROP_PAD_YPOS:
      pad->ypos = g_value_get_int (value);
      break;
    case PROP_PAD_WIDTH:
      pad->width = g_value_get_int (value);
      gst_video_aggregator_convert_pad_update_conversion_info
          (GST_VIDEO_AGGREGATOR_CONVERT_PAD (pad));
      break;
    case PROP_PAD_HEIGHT:
      pad->height = g_value_get_int (value);
      gst_video_aggregator_convert_pad_update_conversion_info
          (GST_VIDEO_AGGREGATOR_CONVERT_PAD (pad));
      break;
    case PROP_PAD_ALPHA:
      pad->alpha = g_value_get_double (value);
      break;
    case PROP_PAD_CROSSFADE_RATIO:
      pad->crossfade = g_value_get_double (value);
      pad->crossfade = 0.0f;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
_mixer_pad_get_output_size (GstNvblender * comp,
    GstNvblenderPad * comp_pad, gint out_par_n, gint out_par_d, gint * width,
    gint * height)
{
  GstVideoAggregatorPad *vagg_pad = GST_VIDEO_AGGREGATOR_PAD (comp_pad);
  gint pad_width, pad_height;
  guint dar_n, dar_d;

  /* FIXME: Anything better we can do here? */
  if (!vagg_pad->info.finfo
      || vagg_pad->info.finfo->format == GST_VIDEO_FORMAT_UNKNOWN) {
    GST_DEBUG_OBJECT (comp_pad, "Have no caps yet");
    *width = 0;
    *height = 0;
    return;
  }

  pad_width =
      comp_pad->width <=
      0 ? GST_VIDEO_INFO_WIDTH (&vagg_pad->info) : comp_pad->width;
  pad_height =
      comp_pad->height <=
      0 ? GST_VIDEO_INFO_HEIGHT (&vagg_pad->info) : comp_pad->height;

  if (!gst_video_calculate_display_ratio (&dar_n, &dar_d, pad_width, pad_height,
          GST_VIDEO_INFO_PAR_N (&vagg_pad->info),
          GST_VIDEO_INFO_PAR_D (&vagg_pad->info), out_par_n, out_par_d)) {
    GST_WARNING_OBJECT (comp_pad, "Cannot calculate display aspect ratio");
    *width = *height = 0;
    return;
  }
  GST_LOG_OBJECT (comp_pad, "scaling %ux%u by %u/%u (%u/%u / %u/%u)", pad_width,
      pad_height, dar_n, dar_d, GST_VIDEO_INFO_PAR_N (&vagg_pad->info),
      GST_VIDEO_INFO_PAR_D (&vagg_pad->info), out_par_n, out_par_d);

  if (pad_height % dar_n == 0) {
    pad_width = gst_util_uint64_scale_int (pad_height, dar_n, dar_d);
  } else if (pad_width % dar_d == 0) {
    pad_height = gst_util_uint64_scale_int (pad_width, dar_d, dar_n);
  } else {
    pad_width = gst_util_uint64_scale_int (pad_height, dar_n, dar_d);
  }

  *width = pad_width;
  *height = pad_height;
}

static gboolean
gst_nvblender_pad_set_info (GstVideoAggregatorPad * pad,
    GstVideoAggregator * vagg G_GNUC_UNUSED,
    GstVideoInfo * current_info, GstVideoInfo * wanted_info)
{
  GstNvblender *comp = GST_NVBLENDER (vagg);
  GstNvblenderPad *cpad = GST_NVBLENDER_PAD (pad);
  gchar *colorimetry, *best_colorimetry;
  const gchar *chroma, *best_chroma;
  gint width, height;

  if (!current_info->finfo)
    return TRUE;

  if (GST_VIDEO_INFO_FORMAT (current_info) == GST_VIDEO_FORMAT_UNKNOWN)
    return TRUE;

  if (cpad->convert)
    gst_video_converter_free (cpad->convert);

  cpad->convert = NULL;

  if (GST_VIDEO_INFO_MULTIVIEW_MODE (current_info) !=
      GST_VIDEO_MULTIVIEW_MODE_NONE
      && GST_VIDEO_INFO_MULTIVIEW_MODE (current_info) !=
      GST_VIDEO_MULTIVIEW_MODE_MONO) {
    GST_FIXME_OBJECT (pad, "Multiview support is not implemented yet");
    return FALSE;
  }

  colorimetry = gst_video_colorimetry_to_string (&(current_info->colorimetry));
  chroma = gst_video_chroma_to_string (current_info->chroma_site);

  best_colorimetry =
      gst_video_colorimetry_to_string (&(wanted_info->colorimetry));
  best_chroma = gst_video_chroma_to_string (wanted_info->chroma_site);

  _mixer_pad_get_output_size (comp, cpad, GST_VIDEO_INFO_PAR_N (&vagg->info),
      GST_VIDEO_INFO_PAR_D (&vagg->info), &width, &height);

  if (GST_VIDEO_INFO_FORMAT (wanted_info) !=
      GST_VIDEO_INFO_FORMAT (current_info)
      || g_strcmp0 (colorimetry, best_colorimetry)
      || g_strcmp0 (chroma, best_chroma)
      || width != current_info->width || height != current_info->height) {
    GstVideoInfo tmp_info;

    /* Initialize with the wanted video format and our original width and
     * height as we don't want to rescale. Then copy over the wanted
     * colorimetry, and chroma-site and our current pixel-aspect-ratio
     * and other relevant fields.
     */
    gst_video_info_set_format (&tmp_info, GST_VIDEO_INFO_FORMAT (wanted_info),
        width, height);
    tmp_info.chroma_site = wanted_info->chroma_site;
    tmp_info.colorimetry = wanted_info->colorimetry;
    tmp_info.par_n = wanted_info->par_n;
    tmp_info.par_d = wanted_info->par_d;
    tmp_info.fps_n = current_info->fps_n;
    tmp_info.fps_d = current_info->fps_d;
    tmp_info.flags = current_info->flags;
    tmp_info.interlace_mode = current_info->interlace_mode;

    GST_DEBUG_OBJECT (pad, "This pad will be converted from format %s to %s, "
        "colorimetry %s to %s, chroma-site %s to %s, "
        "width/height %d/%d to %d/%d",
        current_info->finfo->name, tmp_info.finfo->name,
        colorimetry, best_colorimetry,
        chroma, best_chroma,
        current_info->width, current_info->height, width, height);

    cpad->convert = gst_video_converter_new (current_info, &tmp_info, NULL);
    cpad->conversion_info = tmp_info;
    if (!cpad->convert) {
      g_free (colorimetry);
      g_free (best_colorimetry);
      GST_WARNING_OBJECT (pad, "No path found for conversion");
      return FALSE;
    }
  } else {
    cpad->conversion_info = *current_info;
    GST_DEBUG_OBJECT (pad, "This pad will not need conversion");
  }
  g_free (colorimetry);
  g_free (best_colorimetry);

  return TRUE;
}

/* Test whether rectangle2 contains rectangle 1 (geometrically) */
static gboolean
is_rectangle_contained (GstVideoRectangle rect1, GstVideoRectangle rect2)
{
  if ((rect2.x <= rect1.x) && (rect2.y <= rect1.y) &&
      ((rect2.x + rect2.w) >= (rect1.x + rect1.w)) &&
      ((rect2.y + rect2.h) >= (rect1.y + rect1.h)))
    return TRUE;
  return FALSE;
}

static GstVideoRectangle
clamp_rectangle (gint x, gint y, gint w, gint h, gint outer_width,
    gint outer_height)
{
  gint x2 = x + w;
  gint y2 = y + h;
  GstVideoRectangle clamped;

  /* Clamp the x/y coordinates of this frame to the output boundaries to cover
   * the case where (say, with negative xpos/ypos or w/h greater than the output
   * size) the non-obscured portion of the frame could be outside the bounds of
   * the video itself and hence not visible at all */
  clamped.x = CLAMP (x, 0, outer_width);
  clamped.y = CLAMP (y, 0, outer_height);
  clamped.w = CLAMP (x2, 0, outer_width) - clamped.x;
  clamped.h = CLAMP (y2, 0, outer_height) - clamped.y;

  return clamped;
}

/* Call this with the lock taken */
static gboolean
_pad_obscures_rectangle (GstVideoAggregator * vagg, GstVideoAggregatorPad * pad,
    const GstVideoRectangle rect, gboolean rect_transparent)
{
  GstVideoRectangle pad_rect;
  GstNvblenderPad *cpad = GST_NVBLENDER_PAD (pad);

  /* No buffer to obscure the rectangle with */
  if (!gst_video_aggregator_pad_has_current_buffer (pad))
    return FALSE;

  /* Can't obscure if it's transparent and if the format has an alpha component
   * we'd have to inspect every pixel to know if the frame is opaque, so assume
   * it doesn't obscure. As a bonus, if the rectangle is fully transparent, we
   * can also obscure it if we have alpha components on the pad */
  if (!rect_transparent &&
      (cpad->alpha != 1.0 || GST_VIDEO_INFO_HAS_ALPHA (&pad->info)))
    return FALSE;

  pad_rect.x = cpad->xpos;
  pad_rect.y = cpad->ypos;
  /* Handle pixel and display aspect ratios to find the actual size */
  _mixer_pad_get_output_size (GST_NVBLENDER (vagg), cpad, GST_VIDEO_INFO_PAR_N (&vagg->info),
      GST_VIDEO_INFO_PAR_D (&vagg->info), &(pad_rect.w), &(pad_rect.h));

  if (!is_rectangle_contained (rect, pad_rect))
    return FALSE;

  GST_DEBUG_OBJECT (pad, "Pad %s %ix%i@(%i,%i) obscures rect %ix%i@(%i,%i)",
      GST_PAD_NAME (pad), pad_rect.w, pad_rect.h, pad_rect.x, pad_rect.y,
      rect.w, rect.h, rect.x, rect.y);

  return TRUE;
}

static gboolean
gst_nvblender_pad_prepare_frame (GstVideoAggregatorPad * pad,
    GstVideoAggregator * vagg, GstBuffer * buffer,
    GstVideoFrame * prepared_frame)
{
  GstNvblender *comp = GST_NVBLENDER (vagg);
  GstNvblenderPad *cpad = GST_NVBLENDER_PAD (pad);
  gint width, height;
  gboolean frame_obscured = FALSE;
  GList *l;
  /* The rectangle representing this frame, clamped to the video's boundaries.
   * Due to the clamping, this is different from the frame width/height above. */
  GstVideoRectangle frame_rect;

  /* There's three types of width/height here:
   * 1. GST_VIDEO_FRAME_WIDTH/HEIGHT:
   *     The frame width/height (same as pad->info.height/width;
   *     see gst_video_frame_map())
   * 2. cpad->width/height:
   *     The optional pad property for scaling the frame (if zero, the video is
   *     left unscaled)
   * 3. conversion_info.width/height:
   *     Equal to cpad->width/height if it's set, otherwise it's the pad
   *     width/height. See ->set_info()
   * */

  _mixer_pad_get_output_size (GST_NVBLENDER (vagg), cpad, GST_VIDEO_INFO_PAR_N (&vagg->info),
      GST_VIDEO_INFO_PAR_D (&vagg->info), &width, &height);

  if (cpad->alpha == 0.0) {
    GST_DEBUG_OBJECT (pad, "Pad has alpha 0.0, not converting frame");
    goto done;
  }

  frame_rect = clamp_rectangle (cpad->xpos, cpad->ypos, width, height,
      GST_VIDEO_INFO_WIDTH (&vagg->info), GST_VIDEO_INFO_HEIGHT (&vagg->info));

  if (frame_rect.w == 0 || frame_rect.h == 0) {
    GST_DEBUG_OBJECT (pad, "Resulting frame is zero-width or zero-height "
        "(w: %i, h: %i), skipping", frame_rect.w, frame_rect.h);
    goto done;
  }

#if 0
  GST_OBJECT_LOCK (vagg);
  /* Check if this frame is obscured by a higher-zorder frame
   * TODO: Also skip a frame if it's obscured by a combination of
   * higher-zorder frames */
  l = g_list_find (GST_ELEMENT (vagg)->sinkpads, pad)->next;
  for (; l; l = l->next) {
    if (_pad_obscures_rectangle (vagg, l->data, frame_rect, FALSE)) {
      frame_obscured = TRUE;
      break;
    }
  }
  GST_OBJECT_UNLOCK (vagg);
    if (frame_obscured)
    goto done;

#endif


  return
      GST_VIDEO_AGGREGATOR_PAD_CLASS
      (gst_nvblender_pad_parent_class)->prepare_frame (pad, vagg, buffer,
      prepared_frame);

done:

  return TRUE;
}

static void
gst_nvblender_pad_create_conversion_info (GstVideoAggregatorConvertPad * pad,
    GstVideoAggregator * vagg, GstVideoInfo * conversion_info)
{
  GstNvblenderPad *cpad = GST_NVBLENDER_PAD (pad);
  gint width, height;

  GST_VIDEO_AGGREGATOR_CONVERT_PAD_CLASS
      (gst_nvblender_pad_parent_class)->create_conversion_info (pad, vagg,
      conversion_info);
  if (!conversion_info->finfo)
    return;

  _mixer_pad_get_output_size (GST_NVBLENDER (vagg), cpad, GST_VIDEO_INFO_PAR_N (&vagg->info),
      GST_VIDEO_INFO_PAR_D (&vagg->info), &width, &height);

  /* The only thing that can change here is the width
   * and height, otherwise set_info would've been called */
  if (GST_VIDEO_INFO_WIDTH (conversion_info) != width ||
      GST_VIDEO_INFO_HEIGHT (conversion_info) != height) {
    GstVideoInfo tmp_info;

    /* Initialize with the wanted video format and our original width and
     * height as we don't want to rescale. Then copy over the wanted
     * colorimetry, and chroma-site and our current pixel-aspect-ratio
     * and other relevant fields.
     */
    gst_video_info_set_format (&tmp_info,
        GST_VIDEO_INFO_FORMAT (conversion_info), width, height);
    tmp_info.chroma_site = conversion_info->chroma_site;
    tmp_info.colorimetry = conversion_info->colorimetry;
    tmp_info.par_n = conversion_info->par_n;
    tmp_info.par_d = conversion_info->par_d;
    tmp_info.fps_n = conversion_info->fps_n;
    tmp_info.fps_d = conversion_info->fps_d;
    tmp_info.flags = conversion_info->flags;
    tmp_info.interlace_mode = conversion_info->interlace_mode;

    *conversion_info = tmp_info;
  }
}


static void
gst_nvblender_pad_finalize (GObject * object)
{
  GstNvblenderPad *pad = GST_NVBLENDER_PAD (object);

  if (pad->convert)
    gst_video_converter_free (pad->convert);
  pad->convert = NULL;

  G_OBJECT_CLASS (gst_nvblender_pad_parent_class)->finalize (object);
}

static void
gst_nvblender_pad_class_init (GstNvblenderPadClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstVideoAggregatorPadClass *vaggpadclass =
      (GstVideoAggregatorPadClass *) klass;
  GstVideoAggregatorConvertPadClass *vaggcpadclass =
    (GstVideoAggregatorConvertPadClass *) klass;

  gobject_class->set_property = gst_nvblender_pad_set_property;
  gobject_class->get_property = gst_nvblender_pad_get_property;

  g_object_class_install_property (gobject_class, PROP_PAD_XPOS,
      g_param_spec_int ("xpos", "X Position", "X Position of the picture",
          G_MININT, G_MAXINT, DEFAULT_PAD_XPOS,
          G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PAD_YPOS,
      g_param_spec_int ("ypos", "Y Position", "Y Position of the picture",
          G_MININT, G_MAXINT, DEFAULT_PAD_YPOS,
          G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PAD_WIDTH,
      g_param_spec_int ("width", "Width", "Width of the picture",
          G_MININT, G_MAXINT, DEFAULT_PAD_WIDTH,
          G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PAD_HEIGHT,
      g_param_spec_int ("height", "Height", "Height of the picture",
          G_MININT, G_MAXINT, DEFAULT_PAD_HEIGHT,
          G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PAD_ALPHA,
      g_param_spec_double ("alpha", "Alpha", "Alpha of the picture", 0.0, 1.0,
          DEFAULT_PAD_ALPHA,
          G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PAD_CROSSFADE_RATIO,
      g_param_spec_double ("crossfade-ratio", "Crossfade ratio",
          "The crossfade ratio to use while crossfading with the following pad."
          "A value inferior to 0 means no crossfading.",
          -1.0, 1.0, DEFAULT_PAD_CROSSFADE_RATIO,
          G_PARAM_READWRITE | GST_PARAM_CONTROLLABLE | G_PARAM_STATIC_STRINGS));

  vaggpadclass->prepare_frame =
      GST_DEBUG_FUNCPTR (gst_nvblender_pad_prepare_frame);
  vaggcpadclass->create_conversion_info =
      GST_DEBUG_FUNCPTR (gst_nvblender_pad_create_conversion_info);
}

static void
gst_nvblender_pad_init (GstNvblenderPad * compo_pad)
{
  compo_pad->xpos = DEFAULT_PAD_XPOS;
  compo_pad->ypos = DEFAULT_PAD_YPOS;
  compo_pad->alpha = DEFAULT_PAD_ALPHA;
  compo_pad->crossfade = DEFAULT_PAD_CROSSFADE_RATIO;
}


/* GstNvblender */
#define DEFAULT_BACKGROUND COMPOSITOR_BACKGROUND_CHECKER
enum
{
  PROP_0,
  PROP_BACKGROUND,
  PROP_BATCHED_BACKGROUND,
  PROP_GPU_ID
};

static void gst_compositor_child_proxy_init (gpointer g_iface,
    gpointer iface_data);

#define GST_TYPE_COMPOSITOR_BACKGROUND (gst_nvblender_background_get_type())
static GType
gst_nvblender_background_get_type (void)
{
  static GType compositor_background_type = 0;

  static const GEnumValue compositor_background[] = {
    {COMPOSITOR_BACKGROUND_CHECKER, "Checker pattern", "checker"},
    {COMPOSITOR_BACKGROUND_BLACK, "Black", "black"},
    {COMPOSITOR_BACKGROUND_WHITE, "White", "white"},
    {COMPOSITOR_BACKGROUND_TRANSPARENT,
        "Transparent Background to enable further compositing", "transparent"},
    {0, NULL, NULL},
  };

  if (!compositor_background_type) {
    compositor_background_type =
        g_enum_register_static ("GstNvCompositorBackground",
        compositor_background);
  }
  return compositor_background_type;
}

static void
gst_nvblender_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstNvblender *self = GST_NVBLENDER (object);

  switch (prop_id) {
    case PROP_BACKGROUND:
      g_value_set_enum (value, self->background);
      break;
    case PROP_BATCHED_BACKGROUND:
      g_value_set_boolean (value, self->batched_background);
      break;
    case PROP_GPU_ID:
      g_value_set_int(value, self->gpu_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvblender_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstNvblender *self = GST_NVBLENDER (object);

  switch (prop_id) {
    case PROP_BACKGROUND:
      self->background = g_value_get_enum (value);
      break;
    case PROP_BATCHED_BACKGROUND:
      self->batched_background = g_value_get_boolean (value);
      break;
    case PROP_GPU_ID:
      self->gpu_id = g_value_get_int(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

#define gst_nvblender_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvblender, gst_nvblender,
    GST_TYPE_VIDEO_AGGREGATOR, G_IMPLEMENT_INTERFACE (GST_TYPE_CHILD_PROXY,
        gst_compositor_child_proxy_init));


static GstCaps *
_fixate_caps (GstAggregator * agg, GstCaps * caps)
{
  GstVideoAggregator *vagg = GST_VIDEO_AGGREGATOR (agg);
  GList *l;
  gint best_width = -1, best_height = -1;
  gint best_fps_n = -1, best_fps_d = -1;
  gint par_n, par_d;
  gdouble best_fps = 0.;
  GstCaps *ret = NULL;
  GstStructure *s;

  ret = gst_caps_make_writable (caps);

  /* we need this to calculate how large to make the output frame */
  s = gst_caps_get_structure (ret, 0);
  if (gst_structure_has_field (s, "pixel-aspect-ratio")) {
    gst_structure_fixate_field_nearest_fraction (s, "pixel-aspect-ratio", 1, 1);
    gst_structure_get_fraction (s, "pixel-aspect-ratio", &par_n, &par_d);
  } else {
    par_n = par_d = 1;
  }

  GST_OBJECT_LOCK (vagg);
  for (l = GST_ELEMENT (vagg)->sinkpads; l; l = l->next) {
    GstVideoAggregatorPad *vaggpad = l->data;
    GstNvblenderPad *compositor_pad = GST_NVBLENDER_PAD (vaggpad);
    gint this_width, this_height;
    gint width, height;
    gint fps_n, fps_d;
    gdouble cur_fps;

    fps_n = GST_VIDEO_INFO_FPS_N (&vaggpad->info);
    fps_d = GST_VIDEO_INFO_FPS_D (&vaggpad->info);
    _mixer_pad_get_output_size (GST_NVBLENDER (vagg), compositor_pad, par_n,
        par_d, &width, &height);

    if (width == 0 || height == 0)
      continue;

    this_width = width + MAX (compositor_pad->xpos, 0);
    this_height = height + MAX (compositor_pad->ypos, 0);

    if (best_width < this_width)
      best_width = this_width;
    if (best_height < this_height)
      best_height = this_height;

    if (fps_d == 0)
      cur_fps = 0.0;
    else
      gst_util_fraction_to_double (fps_n, fps_d, &cur_fps);

    if (best_fps < cur_fps) {
      best_fps = cur_fps;
      best_fps_n = fps_n;
      best_fps_d = fps_d;
    }
  }
  GST_OBJECT_UNLOCK (vagg);

  if (best_fps_n <= 0 || best_fps_d <= 0 || best_fps == 0.0) {
    best_fps_n = 25;
    best_fps_d = 1;
    best_fps = 25.0;
  }

  gst_structure_fixate_field_nearest_int (s, "width", best_width);
  gst_structure_fixate_field_nearest_int (s, "height", best_height);
  gst_structure_fixate_field_nearest_fraction (s, "framerate", best_fps_n,
      best_fps_d);
  ret = gst_caps_fixate (ret);

  return ret;
}

static gboolean
_negotiated_caps (GstAggregator * agg, GstCaps * caps)
{
  GstVideoInfo v_info;
  GstVideoAggregator *vagg = GST_VIDEO_AGGREGATOR(agg);
  GList *l;
  gboolean fmt_match = TRUE;
  GstVideoInfo* wanted_info = NULL;

  GST_DEBUG_OBJECT(agg, "Negotiated caps %" GST_PTR_FORMAT, caps);

  if (!gst_video_info_from_caps (&v_info, caps))
    return FALSE;
  GST_OBJECT_LOCK (vagg);
  for (l = GST_ELEMENT (vagg)->sinkpads; l; l = l->next) {
    GstVideoAggregatorPad *mpad = (GstVideoAggregatorPad*)l->data;

    if (GST_VIDEO_INFO_WIDTH (&mpad->info) == 0
        || GST_VIDEO_INFO_HEIGHT (&mpad->info) == 0)
      continue;

    if (wanted_info == NULL){
      wanted_info = &mpad->info;
    } else if (GST_VIDEO_INFO_FORMAT(wanted_info) !=
               GST_VIDEO_INFO_FORMAT(&mpad->info)) {
      fmt_match = FALSE;
      break;
    }
  }
  GST_OBJECT_UNLOCK (vagg);
  if (fmt_match == FALSE){
    GST_ERROR_OBJECT(agg, "Caps on sink pads dont match");
    return FALSE;
  }
  return GST_AGGREGATOR_CLASS (parent_class)->negotiated_src_caps (agg, caps);
}


static void
_find_best_format (GstVideoAggregator * vagg,
    GstCaps * downstream_caps, GstVideoInfo * best_info,
    gboolean * at_least_one_alpha)
{
  GList *tmp;
  GstVideoAggregatorPad *pad;

  GST_OBJECT_LOCK (vagg);
  *at_least_one_alpha = FALSE;
  for (tmp = GST_ELEMENT(vagg)->sinkpads; tmp; tmp = tmp->next) {
    GstStructure *s;
    gint format_number;

    pad = tmp->data;

    if (!pad->info.finfo)
      continue;

    /* This can happen if we release a pad and another pad hasn't been negotiated_caps yet */
    if (GST_VIDEO_INFO_FORMAT (&pad->info) == GST_VIDEO_FORMAT_UNKNOWN)
      continue;

    /* Make downstream accept this format ? */
    if (!strcmp (GST_PAD_NAME(pad), "sink_0")){
      *best_info = pad->info;
      break;
    }
  }
  GST_OBJECT_UNLOCK (vagg);

}


/* Fills frame with transparent pixels if @nframe is NULL otherwise copy @frame
 * properties and fill @nframes with transparent pixels */
static GstFlowReturn
gst_nvblender_fill_transparent (GstNvblender * self, GstVideoFrame * frame,
    GstVideoFrame * nframe)
{
  guint plane, num_planes, height, i;

  if (nframe) {
    GstBuffer *cbuffer = gst_buffer_copy_deep (frame->buffer);

    if (!gst_video_frame_map (nframe, &frame->info, cbuffer, GST_MAP_WRITE)) {
      GST_WARNING_OBJECT (self, "Could not map output buffer");
      return GST_FLOW_ERROR;
    }
  } else {
    nframe = frame;
  }

  num_planes = GST_VIDEO_FRAME_N_PLANES (nframe);
  for (plane = 0; plane < num_planes; ++plane) {
    guint8 *pdata;
    gsize rowsize, plane_stride;

    pdata = GST_VIDEO_FRAME_PLANE_DATA (nframe, plane);
    plane_stride = GST_VIDEO_FRAME_PLANE_STRIDE (nframe, plane);
    rowsize = GST_VIDEO_FRAME_COMP_WIDTH (nframe, plane)
        * GST_VIDEO_FRAME_COMP_PSTRIDE (nframe, plane);
    height = GST_VIDEO_FRAME_COMP_HEIGHT (nframe, plane);
    for (i = 0; i < height; ++i) {
      memset (pdata, 0, rowsize);
      pdata += plane_stride;
    }
  }

  return GST_FLOW_OK;
}

static gboolean gst_nvblender_get_pix_fmt (GstVideoInfo * info, NvBufSurfaceColorFormat * pix_fmt, gint * isurf_count)
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
        }
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_YUV420_709;

        break;
      case GST_VIDEO_FORMAT_UYVY:
        *pix_fmt = NVBUF_COLOR_FORMAT_UYVY;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_YUY2:
        *pix_fmt = NVBUF_COLOR_FORMAT_YUYV;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_YVYU:
        *pix_fmt = NVBUF_COLOR_FORMAT_YVYU;
        *isurf_count = 1;
        break;
      case GST_VIDEO_FORMAT_NV12:
        *pix_fmt = NVBUF_COLOR_FORMAT_NV12;
        *isurf_count = 2;
        if (info->colorimetry.range == GST_VIDEO_COLOR_RANGE_0_255) {
          if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
            *pix_fmt = NVBUF_COLOR_FORMAT_NV12_709_ER;
          else
            *pix_fmt = NVBUF_COLOR_FORMAT_NV12_ER;
        }
        else if (info->colorimetry.matrix == GST_VIDEO_COLOR_MATRIX_BT709)
          *pix_fmt = NVBUF_COLOR_FORMAT_NV12_709;
        break;
      case GST_VIDEO_FORMAT_GRAY8:
        *pix_fmt = NVBUF_COLOR_FORMAT_GRAY8;
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
        *isurf_count = 2;
        break;
#endif
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
      default:
        ret = FALSE;
        break;
    }
  } else if (GST_VIDEO_INFO_IS_GRAY(info)) {
    switch (GST_VIDEO_FORMAT_INFO_FORMAT (info->finfo)) {
      case GST_VIDEO_FORMAT_GRAY8:
        *pix_fmt = NVBUF_COLOR_FORMAT_GRAY8;
        *isurf_count = 1;
        break;
      default:
        ret = FALSE;
        break;
    }
  }

  return ret;
}

static gint
get_bytes_per_pix_from_color(NvBufSurfaceColorFormat pix_fmt, gint plane_id)
{
  gint bytes_per_pix = 1;
  switch (pix_fmt){
    case NVBUF_COLOR_FORMAT_YUV420:
    case NVBUF_COLOR_FORMAT_YUV420_709:
    case NVBUF_COLOR_FORMAT_YUV420_709_ER:
    case NVBUF_COLOR_FORMAT_YUV420_ER:
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
    case NVBUF_COLOR_FORMAT_NV12_12LE:
    case NVBUF_COLOR_FORMAT_NV12_10LE:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709:
    case NVBUF_COLOR_FORMAT_NV12_10LE_2020:
      if (plane_id == 0)
        bytes_per_pix = 2;
      else
        bytes_per_pix = 4;
      break;
    case NVBUF_COLOR_FORMAT_UYVY:
    case NVBUF_COLOR_FORMAT_YUYV:
    case NVBUF_COLOR_FORMAT_YVYU:
        bytes_per_pix = 2;
      break;
    case NVBUF_COLOR_FORMAT_BGRx:
    case NVBUF_COLOR_FORMAT_RGBA:
        bytes_per_pix = 4;
      break;
    case NVBUF_COLOR_FORMAT_BGR:
    case NVBUF_COLOR_FORMAT_RGB:
        bytes_per_pix = 3;
      break;
    case NVBUF_COLOR_FORMAT_GRAY8:
        bytes_per_pix = 1;
      break;
    default:
      break;
  }

  return bytes_per_pix;
}

static void nvdsblender_setup_input_surface
        (GstNvblender *self, NvBufSurface& ip_surf, GstMapInfo& inmap)
{
    NvBufSurfaceParams surfaceList;
    GstVideoInfo *surf_list_info  = NULL;
    memset(&surfaceList, 0, sizeof(surfaceList));
    ip_surf.surfaceList = &surfaceList;
    gpointer surf_list_data = inmap.data;
    guint bytesPerPixel;
    gsize surf_list_size = inmap.size;
    ip_surf.gpuId = 0;
    ip_surf.batchSize = 1;
    ip_surf.numFilled = 1;
    ip_surf.memType = NVBUF_MEM_SYSTEM;

    bytesPerPixel = get_bytes_per_pix_from_color(self->in_pix_fmt, 0);
    ip_surf.surfaceList->planeParams.num_planes = self->info.finfo->n_planes;
    ip_surf.surfaceList->pitch  = self->info.stride[0];
    ip_surf.surfaceList->colorFormat = self->in_pix_fmt;
    ip_surf.surfaceList->planeParams.offset[0] = self->info.offset[0];
    ip_surf.surfaceList->width  = self->width;
    ip_surf.surfaceList->height = self->height;
    ip_surf.surfaceList->planeParams.width[0] = GST_ROUND_UP_4(self->width * bytesPerPixel) / bytesPerPixel;
    ip_surf.surfaceList->planeParams.height[0] = self->height;
    ip_surf.surfaceList->planeParams.psize[0] = self->height * ip_surf.surfaceList->pitch;
    surf_list_info = &(self->info);

    ip_surf.surfaceList->dataSize = surf_list_size;       // size of allocated hw mem
    ip_surf.surfaceList->dataPtr = surf_list_data;
    ip_surf.surfaceList->layout = NVBUF_LAYOUT_PITCH;
    ip_surf.surfaceList->planeParams.pitch[0] = ip_surf.surfaceList->pitch;
    ip_surf.surfaceList->planeParams.bytesPerPix[0] = bytesPerPixel;


    for (uint32_t j = 1; j < surf_list_info->finfo->n_planes; j++)
    {
        guint comp_width = GST_VIDEO_INFO_COMP_WIDTH (surf_list_info, j);
        guint comp_height = GST_VIDEO_INFO_COMP_HEIGHT (surf_list_info, j);
        guint comp_pitch = surf_list_info->stride[j];
        ip_surf.surfaceList->planeParams.height[j] = comp_height;
        ip_surf.surfaceList->planeParams.pitch[j] = comp_pitch;
        ip_surf.surfaceList->planeParams.offset[j] = surf_list_info->offset[j];
        ip_surf.surfaceList->planeParams.psize[j] = comp_pitch*comp_height;
        bytesPerPixel = get_bytes_per_pix_from_color(ip_surf.surfaceList->colorFormat, j);
        ip_surf.surfaceList->planeParams.width[j] = GST_ROUND_UP_4(comp_width * bytesPerPixel) / bytesPerPixel;
        ip_surf.surfaceList->planeParams.bytesPerPix[j] = bytesPerPixel;
    }
}

static GstFlowReturn
gst_nvblender_aggregate_frames (GstVideoAggregator * vagg, GstBuffer * outbuf)
{
  GList *l;
  GstMapInfo outmap = GST_MAP_INFO_INIT;
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  GstNvblender *self = GST_NVBLENDER (vagg);
  //BlendFunction composite;
  GstVideoFrame out_frame, *outframe=NULL;
  vector <guint8 *> input_surfaces;
  vector<gint> input_pad_id;
  gint pad_id=0;
  outframe = &out_frame;
  NvDsBatchMeta *batch_meta=NULL;
  NvDsUserMeta *user_meta=NULL;
  NvBufSurface *ip_surf1 = NULL;
  NvBufSurface alpha_surf;
  NvBufSurface *ip_surf = NULL;
  NvBufSurface *op_surf = NULL;
  NvDsBatchMeta *bg_batch_meta=NULL;
  vector<int>::iterator it1;
  vector<int> bg_src_idx;
  NvBufSurfaceParams *surfaceList = NULL;
  NvBufSurfaceParams *alphaSurfaceList = NULL;
  NvBufSurfTransformCompositeBlendParams blend_params;
  std::vector <bool> aigs_enabled;
  NvBufSurface bg_surf;
  GstVideoAggregatorPad *last_pad = NULL;
  cudaError_t CUerr = cudaSuccess;
  GST_OBJECT_LOCK (vagg);
  memset(&blend_params, 0, sizeof(blend_params));
  NvBufSurfTransformConfigParams config_params;
  CUerr = cudaSetDevice(self->gpu_id);
  int err = -1;
  if (CUerr != cudaSuccess) {
    GST_ERROR_OBJECT (self,"\n *** Unable to set device in %s Line %d\n", __func__, __LINE__);
    GST_OBJECT_UNLOCK (vagg);
    return GST_FLOW_ERROR;
  }

 config_params.compute_mode = NvBufSurfTransformCompute_GPU;
 config_params.gpu_id = self->gpu_id;
 config_params.cuda_stream = self->cu_nbstream;
 err = NvBufSurfTransformSetSessionParams(&config_params);
 if (err != NvBufSurfTransformError_Success) {
   GST_ERROR_OBJECT(self,"Set session params failed \n");
   GST_OBJECT_UNLOCK (vagg);
   return GST_FLOW_ERROR;
 }

  /* First mix the crossfade frames as required */
  for (l = GST_ELEMENT (vagg)->sinkpads; l; l = l->next)
  {
      GstVideoAggregatorPad *pad = l->data;
      GstNvblenderPad *compo_pad = GST_NVBLENDER_PAD (pad);
      GstBuffer *bufferm = gst_video_aggregator_pad_get_current_buffer(pad);

#if SW_IN_HW_OUT
      if (pad->aggregated_frame != NULL)
#endif
      {
          /* Use this function if there is no decide_allocation with HW buffers allocated*/
          if (!gst_video_aggregator_pad_has_current_buffer (pad))
          {
              GST_DEBUG_OBJECT (vagg, "Did not receive buffer on %s\n", GST_PAD_NAME(pad));
              continue;
          }
          if (!gst_buffer_map (bufferm, &inmap, GST_MAP_READ))
          {
              g_print ("GST_BUFFER_MAP on input failed\n");
              GST_OBJECT_UNLOCK (vagg);
              return GST_FLOW_ERROR;
          }
          //g_print ("%s is %p and its memory is %p refcount = %d\n", GST_PAD_NAME(pad), pad->buffer, inmap.data, GST_MINI_OBJECT_REFCOUNT(pad->buffer));
          if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
          {
              g_print ("GST_BUFFER_MAP on output failed\n");
              GST_OBJECT_UNLOCK (vagg);
              return GST_FLOW_ERROR;
          }

          input_surfaces.push_back (inmap.data);
      }

      sscanf(GST_PAD_NAME(pad), "sink_%d", &pad_id);
      input_pad_id.push_back (pad_id);
      if (!strcmp(GST_PAD_NAME(pad), "sink_0")) {
        gboolean skip_buffer = FALSE;
        GST_DEBUG_OBJECT(self, "sink_0 is foreground");
        if (self->sinkpad0_active){
          if (GST_BUFFER_PTS_IS_VALID(bufferm) &&
              (GST_BUFFER_PTS(bufferm) == self->prev_pts)){
             skip_buffer = TRUE;
          }
        }

        self->prev_pts = GST_BUFFER_PTS(bufferm);
        batch_meta = gst_buffer_get_nvds_batch_meta(bufferm);
        if (skip_buffer){
            batch_meta=NULL;
        }
          self->sinkpad0_active = TRUE;
          if (batch_meta) {
            if (!gst_buffer_copy_into(outbuf, bufferm, GST_BUFFER_COPY_META,
                                      0, -1)) {
              GST_ERROR_OBJECT(self, "meta copy failed");
              return GST_FLOW_ERROR;
            }

            self->alpha_surf = NULL;
            batch_meta = gst_buffer_get_nvds_batch_meta(outbuf);
            for (NvDsMetaList *l_user = batch_meta->batch_user_meta_list;
                 l_user != NULL; l_user = l_user->next) {
              GstMapInfo in_map_info;
              AiGsSourceData *src_meta = NULL;
              user_meta = (NvDsUserMeta *)l_user->data;
              if (user_meta &&
                  user_meta->base_meta.meta_type != NVDS_USER_BATCH_META_AIGS)
                continue;

              src_meta = (AiGsSourceData *)user_meta->user_meta_data;
              self->matte_alpha_gst_buffer = src_meta->gstBuffer;
              GST_DEBUG_OBJECT(self, "gstbuffer is %p \n",
                               self->matte_alpha_gst_buffer);
              if (!gst_buffer_map(self->matte_alpha_gst_buffer, &in_map_info,
                                  GST_MAP_READ)) {
                GST_ELEMENT_ERROR(
                    GST_NVBLENDER(vagg), STREAM, FAILED,
                    ("%s:gst buffer map to get pointer to NvBufSurface failed",
                     __func__),
                    (NULL));
                return GST_FLOW_ERROR;
              }

              ip_surf = (NvBufSurface *)inmap.data;
              // Check if we want to do blending on the source
              if (src_meta->numSources == ip_surf->numFilled) {
                aigs_enabled.resize(batch_meta->num_frames_in_batch, TRUE);
                self->alpha_surf = (NvBufSurface *)in_map_info.data;
                /* for (int32_t icnt = 0; icnt <
                 batch_meta->num_frames_in_batch; icnt++) { NvDsFrameMeta
                 *frame_meta = nvds_get_nth_frame_meta(
                       batch_meta->frame_meta_list, (icnt));

                   if (frame_meta && frame_meta->source_id==0) {
                     aigs_enabled[icnt] = FALSE;
                     break;
                   }
                 }*/
              } else {
                self->alpha_surf = (NvBufSurface *)in_map_info.data;
                alphaSurfaceList = (NvBufSurfaceParams *)g_malloc0(
                    sizeof(NvBufSurfaceParams) * ip_surf->numFilled);
                alpha_surf = *(self->alpha_surf);
                alpha_surf.surfaceList = alphaSurfaceList;
                alpha_surf.numFilled = ip_surf->numFilled;

                aigs_enabled.resize(batch_meta->num_frames_in_batch, FALSE);
                for (int32_t icnt = 0; icnt < batch_meta->num_frames_in_batch;
                     icnt++) {
                  for (int32_t jcnt = 0; jcnt < src_meta->numSources; jcnt++) {
                    // if buf idx has aigs meta?
                    if (icnt == src_meta->arrayList[jcnt]) {
                      // enable blending for that surfaceList
                      aigs_enabled[icnt] = TRUE;
                      // copy the exact location of the corresponding alpha
                      // surface
                      alphaSurfaceList[icnt] =
                          self->alpha_surf->surfaceList[jcnt];
                      break;
                    }
                  }
                }
                self->alpha_surf = &alpha_surf;
              }
              gst_buffer_unmap(self->matte_alpha_gst_buffer, &in_map_info);
              break;
            }
          }
        }
        else if (!strcmp(GST_PAD_NAME(pad), "sink_1")) {
          bg_batch_meta = NULL;
          bg_batch_meta = gst_buffer_get_nvds_batch_meta(bufferm);
        }

      gst_buffer_unmap (bufferm, &inmap);
      gst_buffer_unmap (outbuf, &outmap);
  }

  // Find the input batched FG buffer
  it1 = find(begin(input_pad_id), end(input_pad_id), 0);
  ip_surf = NULL;
  ip_surf1 = NULL;
  if (it1 != end (input_pad_id)){
    ip_surf = (NvBufSurface*)input_surfaces[it1-begin(input_pad_id)];
    //Handle if NULL buffer is pushed
    if (ip_surf && ip_surf->numFilled){
      surfaceList = (NvBufSurfaceParams *)g_malloc0(sizeof(NvBufSurfaceParams) *
                                                  ip_surf->numFilled);
      blend_params.perform_blending =
                  (uint32_t *)g_malloc0(sizeof(uint32_t) * ip_surf->numFilled);
      if ((surfaceList == NULL) || (blend_params.perform_blending == NULL)){
        GST_ERROR_OBJECT(self, "\n *** Unable to allocate memory \n");
        g_free(surfaceList);
        g_free(alphaSurfaceList);
        g_free(blend_params.perform_blending);
        GST_OBJECT_UNLOCK(vagg);
        return GST_FLOW_ERROR;
      }
    }
  }
  else if (self->sinkpad0_active){
    self->sinkpad0_active = FALSE;
    GST_OBJECT_UNLOCK(vagg);
    return GST_FLOW_EOS;
  }

  if (self->batched_background){
    gpointer orig_key = 0;
    gpointer orig_value = 0;
    cudaEvent_t cu_event;
    CUerr =  cudaEventCreate(&cu_event);
    if (CUerr != cudaSuccess) {
      GST_ERROR_OBJECT(self, "\n *** Unable to create cudaEvent_t in %s Line %d\n",
                       __func__, __LINE__);
      g_free(surfaceList);
      g_free(alphaSurfaceList);
      g_free(blend_params.perform_blending);
      GST_OBJECT_UNLOCK(vagg);
      return GST_FLOW_ERROR;
    } // Sink pad 1 for batched background
    it1 = find(begin(input_pad_id), end(input_pad_id), 1);
    if (it1 != end(input_pad_id))
      ip_surf1 = (NvBufSurface *)input_surfaces[it1 - begin(input_pad_id)];
    // If surface present get batch meta and reorder based on it
    if (ip_surf1) {
      if (bg_batch_meta) {
        if (ip_surf)
          bg_surf = *ip_surf;
        else
          bg_surf = *ip_surf1;

        bg_surf.surfaceList = surfaceList;
        // Rearrange surfacelist wrt to FG
        if (batch_meta) {
          for (int32_t icnt = 0; icnt < batch_meta->num_frames_in_batch;
               icnt++) {
            NvDsFrameMeta *frame_meta =
                nvds_get_nth_frame_meta(batch_meta->frame_meta_list, (icnt));

            if (frame_meta) {
              // Find the matching frame in background
              for (int32_t jcnt = 0; jcnt < bg_batch_meta->num_frames_in_batch;
                   jcnt++) {
                NvDsFrameMeta *bg_frame_meta = nvds_get_nth_frame_meta(
                    bg_batch_meta->frame_meta_list, (jcnt));
                // if match found copy the sufaceList
                if (bg_frame_meta->source_id == frame_meta->source_id) {
                  NvBufSurface *bg_surf_l = NULL;
                  bg_surf.surfaceList[icnt] = ip_surf1->surfaceList[jcnt];
                  // TODO:Perform a copy in background buffer as well

                  if (g_hash_table_lookup_extended(self->valid_bg_buf, GINT_TO_POINTER(frame_meta->source_id),
                  &orig_key, &orig_value)){
                    bg_surf_l = orig_value;
                 } else {
                    gint status  = - 1;
                    NvBufSurfaceCreateParams buf_params = {0};
                    buf_params.width = bg_surf.surfaceList[icnt].width;
                    buf_params.height = bg_surf.surfaceList[icnt].height;
                    // Remove UNIFIED memory, make it configurable
                    buf_params.memType = NVBUF_MEM_CUDA_DEVICE;
                    buf_params.gpuId = self->gpu_id;
                    buf_params.colorFormat = bg_surf.surfaceList[icnt].colorFormat;
                    status = NvBufSurfaceCreate(&bg_surf_l,
                                                1, &buf_params);
                    if (status < 0) {
                      printf("Error(%d) in buffer allocation\n", status);
                      return GST_FLOW_ERROR;
                    }
                    g_hash_table_insert(self->valid_bg_buf,
                                        GINT_TO_POINTER(frame_meta->source_id), bg_surf_l);
                  }
                  /// TODO: BSP API for index copy to be used

                  for (int32_t kcnt = 0;
                       kcnt < ip_surf1->surfaceList[jcnt].planeParams.num_planes;
                       kcnt++) {
                    uint8_t *dptr =
                        (uint8_t *)bg_surf_l->surfaceList[0]
                            .dataPtr +
                        bg_surf_l->surfaceList[0]
                            .planeParams.offset[kcnt];
                    uint8_t *sptr =
                        (uint8_t *)ip_surf1->surfaceList[jcnt].dataPtr +
                        ip_surf1->surfaceList[jcnt].planeParams.offset[kcnt];
                    int32_t pitch =
                        ip_surf1->surfaceList[jcnt].planeParams.pitch[kcnt];
                    int32_t width =
                        ip_surf1->surfaceList[jcnt].planeParams.width[kcnt] *
                        ip_surf1->surfaceList[jcnt]
                            .planeParams.bytesPerPix[kcnt];
                    int32_t height =
                        ip_surf1->surfaceList[jcnt].planeParams.height[kcnt];
                    int32_t dpitch =
                        bg_surf_l->surfaceList[0]
                            .planeParams.pitch[kcnt];
                    // Let NVBufsurface handle this,
                    cudaMemcpy2DAsync(dptr, pitch, sptr, pitch, width, height,
                                 cudaMemcpyDefault, self->cu_nbstream);
                  }

                  if ((aigs_enabled.size() > icnt) && aigs_enabled[icnt])
                    blend_params.perform_blending[icnt] = 1;
                  break;
                }
              }
              // If no match found copy the cache address
              if (blend_params.perform_blending[icnt] == 0) {
                if (g_hash_table_lookup_extended(self->valid_bg_buf,
                                                 GINT_TO_POINTER(frame_meta->source_id),
                                                 &orig_key, &orig_value)) {
                  NvBufSurface *surfop = orig_value;
                  bg_surf.surfaceList[icnt] = surfop->surfaceList[0];
                  if ((aigs_enabled.size() > icnt) && aigs_enabled[icnt])
                     blend_params.perform_blending[icnt] = 1;
                }
              }
            }

          } //
        } else {
          for (int32_t jcnt = 0; jcnt < bg_batch_meta->num_frames_in_batch;
               jcnt++) {
            NvDsFrameMeta *bg_frame_meta =
                nvds_get_nth_frame_meta(bg_batch_meta->frame_meta_list, (jcnt));
            // if match found copy the sufaceList
            // TODO:Perform a copy in background buffer as well
          }
        }
      }
      ip_surf1 = &bg_surf;
    } // ip_surf1

    cudaEventRecord(cu_event, self->cu_nbstream);
    // Wait for the copy to finish
    cudaEventSynchronize(cu_event);
    CUerr = cudaEventDestroy(cu_event);
    if (CUerr != cudaSuccess) {
      GST_ERROR_OBJECT(self, "\n *** Unable to destroy cudaEvent_t in %s Line %d\n",
                       __func__, __LINE__);
      g_free(surfaceList);
      g_free(alphaSurfaceList);
      g_free(blend_params.perform_blending);
      GST_OBJECT_UNLOCK(vagg);
      return GST_FLOW_ERROR;
    }
    //  Case for non batched buffers
  } else if (batch_meta) {
    //Do collection and association here
    //Assumption here bg surface and ip surface are similar format, which is required
    //Although same memory type of different background buffers maynot match, even the
    //dimensions.
    // Create a dummy surfacelist, basically batching of background frames on different pads
    bg_surf = *ip_surf;
    bg_surf.surfaceList = surfaceList;
    for (int32_t icnt = 0; icnt < batch_meta->num_frames_in_batch; icnt++){
      NvDsFrameMeta *frame_meta= nvds_get_nth_frame_meta(batch_meta->frame_meta_list,
            (icnt));
      if (frame_meta){
        //Find the source id
        it1 = find(begin(input_pad_id), end(input_pad_id),
                                (frame_meta->source_id+1));
        if (it1 != end(input_pad_id)) {
          ip_surf1 = (NvBufSurface *)input_surfaces[it1-begin(input_pad_id)];
          bg_surf.surfaceList[icnt] = ip_surf1->surfaceList[0];
          blend_params.perform_blending[icnt] = 1;
        }
        else
          GST_DEBUG_OBJECT (self, "there is no associated background frame");
      }else{
        GST_ERROR_OBJECT (self, "there is someproblem with meta");
        g_free(surfaceList);
        g_free(alphaSurfaceList);
        g_free(blend_params.perform_blending);
        GST_OBJECT_UNLOCK(vagg);
        return GST_FLOW_ERROR;
      }
    }
    ip_surf1 = &bg_surf;
  }
  //for (auto it = input_surfaces.begin(); it != input_surfaces.end(); it++)
  {
      op_surf = (NvBufSurface*)outmap.data;
#if SW_IN_HW_OUT
      NvBufSurface ip_surf;
      nvdsblender_setup_input_surface (self, ip_surf, inmap);

      NvBufSurfaceCopy (&ip_surf, op_surf);
#else
      //NvBufSurface *ip_surf = (NvBufSurface *)*it;
#if 0
      /*
       * If its a batch buffer coming on the sinkpad of nvblender, then below code
       * writes only one buffer in that batched buffer so with tiler it will only display one
       * tile at a time , that tile could be displayed at any position
       * or update below code to fill in all the batch buffers */
      cout << "ip_surf " << ip_surf << " size of vector = " << input_surfaces.size() << endl;;
      unsigned char *input_ptr = (unsigned char *) ip_surf->surfaceList->dataPtr;
      unsigned char *output_ptr = (unsigned char *) op_surf->surfaceList->dataPtr;
      for (int i = 0; i < ip_surf->surfaceList->height; i++)
      {
          for (int j = 0; j < ip_surf->surfaceList->width; j++)
          {
              *(output_ptr + j) = (*(output_ptr + j) +  *(input_ptr + j)) >> 1;
          }
          output_ptr += op_surf->surfaceList->pitch;
          input_ptr += ip_surf->surfaceList->pitch;
      }
      op_surf->numFilled = self->batch_size;
#else
      GST_DEBUG_OBJECT(self, "ip_surf %p ip_surf1 %p size of vector %d",
                       ip_surf, ip_surf1, input_surfaces.size());
      int status = -1;
#if 0
      if (self->alpha_surf_allocated == false)
      {
          NvBufSurfaceCreateParams buf_params = {0};
          buf_params.width   = ip_surf->surfaceList->pitch;
          buf_params.height  = ip_surf->surfaceList->height;
          buf_params.memType = NVBUF_MEM_CUDA_UNIFIED;
          buf_params.gpuId = 0;
          buf_params.colorFormat= NVBUF_COLOR_FORMAT_GRAY8;
          status = NvBufSurfaceCreate(&self->alpha_surf, 1, &buf_params);
          if (status < 0)
          {
              printf("Error(%d) in buffer allocation\n", status);
              return -1;
          }
          self->alpha_surf->numFilled = 1;
          self->alpha_surf_allocated = true;
          memset (self->alpha_surf->surfaceList->dataPtr, 128, self->alpha_surf->surfaceList->pitch * self->alpha_surf->surfaceList->height);
      }
#endif
      //WAR as of now, will fix it next CL will be required if batched_Background is used
      if ((ip_surf && ip_surf1) && (ip_surf1->numFilled > ip_surf->numFilled)){
        ip_surf1->numFilled = ip_surf->numFilled;
      }

      status = NvBufSurfTransformCompositeBlend(ip_surf, ip_surf1, self->alpha_surf, op_surf, &blend_params);
      GST_DEBUG_OBJECT(self,"ip_surf = %p ip_surf1 = %p self->alpha_surf = %p op_surf = %p",
      ip_surf, ip_surf1, self->alpha_surf, op_surf);
      if (status < 0)
      {
          GST_ERROR_OBJECT(self,"Error(%d) in NvBufSurfTransformCompositeBlend dropping buffer \n", status);
      }
#endif
#endif
/*      if (it == input_surfaces.end())
          break;*/
  }
  if (user_meta && ip_surf)
    nvds_remove_user_meta_from_batch (batch_meta, user_meta);

  g_free (surfaceList);
  g_free(alphaSurfaceList);
  g_free(blend_params.perform_blending);
  GST_OBJECT_UNLOCK (vagg);

  return GST_FLOW_OK;
}

static gboolean
_get_sinkpads_interlace_mode (GstVideoAggregator * vagg,
    GstVideoAggregatorPad * skip_pad, GstVideoInterlaceMode * mode)
{
  GList *walk;

  GST_OBJECT_LOCK (vagg);
  for (walk = GST_ELEMENT (vagg)->sinkpads; walk; walk = g_list_next (walk)) {
    GstVideoAggregatorPad *vaggpad = walk->data;

    if (skip_pad && vaggpad == skip_pad)
      continue;
    if (vaggpad->info.finfo
        && GST_VIDEO_INFO_FORMAT (&vaggpad->info) != GST_VIDEO_FORMAT_UNKNOWN) {
      *mode = GST_VIDEO_INFO_INTERLACE_MODE (&vaggpad->info);
      GST_OBJECT_UNLOCK (vagg);
      return TRUE;
    }
  }
  GST_OBJECT_UNLOCK (vagg);
  return FALSE;
}


static gboolean
_caps_has_alpha (GstCaps * caps)
{
  guint size = gst_caps_get_size (caps);
  guint i;

  for (i = 0; i < size; i++) {
    GstStructure *s = gst_caps_get_structure (caps, i);
    const GValue *formats = gst_structure_get_value (s, "format");

    if (formats) {
      const GstVideoFormatInfo *info;

      if (GST_VALUE_HOLDS_LIST (formats)) {
        guint list_size = gst_value_list_get_size (formats);
        guint index;

        for (index = 0; index < list_size; index++) {
          const GValue *list_item = gst_value_list_get_value (formats, index);
          info =
              gst_video_format_get_info (gst_video_format_from_string
              (g_value_get_string (list_item)));
          if (GST_VIDEO_FORMAT_INFO_HAS_ALPHA (info))
            return TRUE;
        }

      } else if (G_VALUE_HOLDS_STRING (formats)) {
        info =
            gst_video_format_get_info (gst_video_format_from_string
            (g_value_get_string (formats)));
        if (GST_VIDEO_FORMAT_INFO_HAS_ALPHA (info))
          return TRUE;

      } else {
        g_assert_not_reached ();
        GST_WARNING ("Unexpected type for video 'format' field: %s",
            G_VALUE_TYPE_NAME (formats));
      }

    } else {
      return TRUE;
    }
  }
  return FALSE;
}

static gboolean
_pad_sink_acceptcaps (GstPad * pad,
    GstVideoAggregator * vagg, GstCaps * caps)
{
  gboolean ret;
  GstCaps *modified_caps;
  GstCaps *accepted_caps;
  GstCaps *template_caps;
  gboolean had_current_caps = TRUE;
  gint i, n;
  GstStructure *s;
  GstAggregator *agg = GST_AGGREGATOR (vagg);

  GST_DEBUG_OBJECT (pad, "%" GST_PTR_FORMAT, caps);

  accepted_caps = gst_pad_get_current_caps (GST_PAD (agg->srcpad));

  template_caps = gst_pad_get_pad_template_caps (GST_PAD (agg->srcpad));

  if (accepted_caps == NULL) {
    accepted_caps = template_caps;
    had_current_caps = FALSE;
  }

  accepted_caps = gst_caps_make_writable (accepted_caps);

  GST_LOG_OBJECT (pad, "src caps %" GST_PTR_FORMAT, accepted_caps);

  n = gst_caps_get_size (accepted_caps);
  for (i = 0; i < n; i++) {
    s = gst_caps_get_structure (accepted_caps, i);
    gst_structure_set (s, "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
        "height", GST_TYPE_INT_RANGE, 1, G_MAXINT,
        "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, G_MAXINT, 1, NULL);

    gst_structure_remove_fields (s, "colorimetry", "chroma-site", "format",
        "pixel-aspect-ratio", NULL);
  }

  modified_caps = gst_caps_intersect (accepted_caps, template_caps);

  ret = gst_caps_can_intersect (caps, accepted_caps);
  GST_DEBUG_OBJECT (pad, "%saccepted caps %" GST_PTR_FORMAT,
      (ret ? "" : "not "), caps);
  gst_caps_unref (accepted_caps);
  gst_caps_unref (modified_caps);
  if (had_current_caps)
    gst_caps_unref (template_caps);
  return ret;
}

static GstCaps *
_get_non_alpha_caps (GstCaps * caps)
{
  GstCaps *result;
  guint i, size;

  size = gst_caps_get_size (caps);
  result = gst_caps_new_empty ();
  for (i = 0; i < size; i++) {
    GstStructure *s = gst_caps_get_structure (caps, i);
    const GValue *formats = gst_structure_get_value (s, "format");
    GValue new_formats = { 0, };
    gboolean has_format = FALSE;

    /* FIXME what to do if formats are missing? */
    if (formats) {
      const GstVideoFormatInfo *info;

      if (GST_VALUE_HOLDS_LIST (formats)) {
        guint list_size = gst_value_list_get_size (formats);
        guint index;

        g_value_init (&new_formats, GST_TYPE_LIST);

        for (index = 0; index < list_size; index++) {
          const GValue *list_item = gst_value_list_get_value (formats, index);

          info =
              gst_video_format_get_info (gst_video_format_from_string
              (g_value_get_string (list_item)));
          if (!GST_VIDEO_FORMAT_INFO_HAS_ALPHA (info)) {
            has_format = TRUE;
            gst_value_list_append_value (&new_formats, list_item);
          }
        }

      } else if (G_VALUE_HOLDS_STRING (formats)) {
        info =
            gst_video_format_get_info (gst_video_format_from_string
            (g_value_get_string (formats)));
        if (!GST_VIDEO_FORMAT_INFO_HAS_ALPHA (info)) {
          has_format = TRUE;
          gst_value_init_and_copy (&new_formats, formats);
        }

      } else {
        g_assert_not_reached ();
        GST_WARNING ("Unexpected type for video 'format' field: %s",
            G_VALUE_TYPE_NAME (formats));
      }

      if (has_format) {
        s = gst_structure_copy (s);
        gst_structure_take_value (s, "format", &new_formats);
        gst_caps_append_structure (result, s);
      }

    }
  }

  return result;
}

static GstCaps *
_pad_sink_getcaps (GstPad * pad, GstVideoAggregator * vagg,
    GstCaps * filter)
{
  GstCaps *srccaps;
  GstCaps *template_caps, *sink_template_caps;
  GstCaps *returned_caps;
  GstStructure *s;
  gint i, n;
  GstAggregator *agg = GST_AGGREGATOR (vagg);
  GstPad *srcpad = GST_PAD (agg->srcpad);
  gboolean has_alpha;
  GstVideoInterlaceMode interlace_mode;
  gboolean has_interlace_mode;

  template_caps = gst_pad_get_pad_template_caps (srcpad);

  GST_DEBUG_OBJECT (pad, "Get caps with filter: %" GST_PTR_FORMAT, filter);

  srccaps = gst_pad_peer_query_caps (srcpad, template_caps);
  srccaps = gst_caps_make_writable (srccaps);
  has_alpha = _caps_has_alpha (srccaps);

  has_interlace_mode =
      _get_sinkpads_interlace_mode (vagg, NULL,
      &interlace_mode);

  n = gst_caps_get_size (srccaps);
  for (i = 0; i < n; i++) {
    s = gst_caps_get_structure (srccaps, i);
    gst_structure_set (s, "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
        "height", GST_TYPE_INT_RANGE, 1, G_MAXINT,
        "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, G_MAXINT, 1, NULL);

    gst_structure_remove_fields (s, "colorimetry", "chroma-site", "format",
        "pixel-aspect-ratio", "batch-size", NULL);
    if (has_interlace_mode)
      gst_structure_set (s, "interlace-mode", G_TYPE_STRING,
          gst_video_interlace_mode_to_string (interlace_mode), NULL);
  }

  if (filter) {
    returned_caps = gst_caps_intersect (srccaps, filter);
    gst_caps_unref (srccaps);
  } else {
    returned_caps = srccaps;
  }

  if (has_alpha) {
    sink_template_caps = gst_pad_get_pad_template_caps (pad);
  } else {
    GstVideoAggregatorClass *klass = GST_VIDEO_AGGREGATOR_GET_CLASS (vagg);
    GstCaps *tmp = _get_non_alpha_caps(sink_template_caps);
    gst_caps_unref (sink_template_caps);
    sink_template_caps = tmp;

    n = gst_caps_get_size (sink_template_caps);
    for (i = 0; i < n; i++) {
      gst_caps_set_features(sink_template_caps, i,
                            gst_caps_features_from_string("memory:NVMM"));
    }
  }

  {
    GstCaps *intersect = gst_caps_intersect (returned_caps, sink_template_caps);
    gst_caps_unref (returned_caps);
    returned_caps = intersect;
  }

  gst_caps_unref (template_caps);
  gst_caps_unref (sink_template_caps);

  GST_DEBUG_OBJECT (pad, "Returning caps: %" GST_PTR_FORMAT, returned_caps);

  return returned_caps;
}


static gboolean
_sink_event (GstAggregator * agg, GstAggregatorPad * bpad,
    GstEvent * event)
{
  GstVideoAggregator *vagg = GST_VIDEO_AGGREGATOR (agg);
  GstVideoAggregatorPad *pad = GST_VIDEO_AGGREGATOR_PAD (bpad);
  gboolean ret = TRUE;

  GST_DEBUG_OBJECT (pad, "Got %s event on pad %s:%s",
      GST_EVENT_TYPE_NAME (event), GST_DEBUG_PAD_NAME (pad));

  if (event != NULL) {
    if (strcmp(GST_PAD_NAME(pad), "sink_0") &&
        GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT) {
        GST_OBJECT_LOCK(pad);
        gst_event_copy_segment (event, &GST_AGGREGATOR_PAD(pad)->segment);
        GST_OBJECT_UNLOCK(pad);

    } else if (!strcmp(GST_PAD_NAME(pad), "sink_0")) {
      return GST_AGGREGATOR_CLASS(parent_class)->sink_event(agg, bpad, event);
    }
  }

  return ret;
}

static GstPad *
gst_compositor_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * req_name, const GstCaps * caps)
{
  GstPad *newpad;

  newpad = (GstPad *)
      GST_ELEMENT_CLASS (parent_class)->request_new_pad (element,
      templ, req_name, caps);

  if (newpad == NULL)
    goto could_not_create;

  gst_child_proxy_child_added (GST_CHILD_PROXY (element), G_OBJECT (newpad),
      GST_OBJECT_NAME (newpad));

  return newpad;

could_not_create:
  {
    GST_DEBUG_OBJECT (element, "could not create/add pad");
    return NULL;
  }
}

static void
gst_compositor_release_pad (GstElement * element, GstPad * pad)
{
  GstNvblender *compositor;

  compositor = GST_NVBLENDER (element);

  GST_DEBUG_OBJECT (compositor, "release pad %s:%s", GST_DEBUG_PAD_NAME (pad));

  gst_child_proxy_child_removed (GST_CHILD_PROXY (compositor), G_OBJECT (pad),
      GST_OBJECT_NAME (pad));

  GST_ELEMENT_CLASS (parent_class)->release_pad (element, pad);
}

    static gboolean
_sink_query (GstAggregator * agg, GstAggregatorPad * bpad, GstQuery * query)
{
  gboolean ret=TRUE;
    switch (GST_QUERY_TYPE (query)) {
        case GST_QUERY_ALLOCATION:{
                                      GstCaps *caps;
                                      GstVideoInfo info;
                                      GstBufferPool *pool;
                                      guint size;
                                      GstStructure *structure;

                                      gst_query_parse_allocation (query, &caps, NULL);

                                      if (caps == NULL)
                                          return FALSE;

                                      if (!gst_video_info_from_caps (&info, caps))
                                          return FALSE;

                                      size = GST_VIDEO_INFO_SIZE (&info);

                                      pool = gst_video_buffer_pool_new ();

                                      structure = gst_buffer_pool_get_config (pool);
                                      gst_buffer_pool_config_set_params (structure, caps, size, 0, 0);

                                      if (!gst_buffer_pool_set_config (pool, structure)) {
        gst_object_unref (pool);
        return FALSE;
      }

      gst_query_add_allocation_pool (query, pool, size, 0, 0);
      gst_object_unref (pool);
      gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);

      return TRUE;
    }

    case GST_QUERY_CAPS: {
      GstCaps *filter, *caps;

      gst_query_parse_caps(query, &filter);
      if (filter) {
        caps =
            _pad_sink_getcaps(GST_PAD(bpad), GST_VIDEO_AGGREGATOR(agg), filter);
        gst_query_set_caps_result(query, caps);
        gst_caps_unref(caps);
        ret = TRUE;
      }
      else {
        ret= FALSE;
      }
      break;
    }
    case GST_QUERY_ACCEPT_CAPS: {
      GstCaps *caps;

      gst_query_parse_accept_caps(query, &caps);
      ret = _pad_sink_acceptcaps(GST_PAD(bpad), GST_VIDEO_AGGREGATOR(agg), caps);
      gst_query_set_accept_caps_result(query, ret);
      ret = TRUE;
      break;
    }

    default:
      ret= GST_AGGREGATOR_CLASS (parent_class)->sink_query (agg, bpad, query);
  }
  return ret;
}

static gboolean
gst_nvblender_decide_allocation (GstAggregator * agg, GstQuery * query)
{
    GstCaps *outcaps = NULL;
    GstNvblender *self = GST_NVBLENDER (agg);
    GstAllocator *allocator = NULL;
    gint surf_count = 0;
    GstCaps *myoutcaps = NULL;
    GstAllocationParams params = { 0, 0, 0, 0 };
    guint size, minimum, maximum;
    GstVideoInfo info;
    GstBufferPool *pool = NULL;
    GstStructure *config = NULL;
    gst_query_parse_allocation (query, &outcaps, NULL);
    if (outcaps == NULL)
    {
        g_print ("OUTTCAPS ARE NULL\n");
    }

    GstQuery *bsquery = NULL;
    bsquery = gst_nvquery_batch_size_new ();
    {
        GList *l;
        GstVideoAggregator *vagg = GST_VIDEO_AGGREGATOR (agg);
        for (l = GST_ELEMENT (vagg)->sinkpads; l; l = l->next)
        {
            GstVideoAggregatorPad *pad = l->data;
            if (!strcmp (GST_PAD_NAME(pad), "sink_0"))
            {
                if (gst_pad_peer_query (GST_PAD(pad), bsquery))
                {
                    gst_nvquery_batch_size_parse (bsquery, &self->batch_size);
                    GST_DEBUG_OBJECT (self,"parsed batch size = %d", self->batch_size);
                }
            }
        }
    }

    {
        if (!gst_video_info_from_caps (&info, outcaps))
            g_print ("INVALID CAPS\n");

        size = info.size;

        self->info = info;

        gboolean ret = TRUE;
        gint status = -1;
        ret = gst_nvblender_get_pix_fmt (&info, &self->in_pix_fmt, &surf_count);
        if (ret != TRUE)
        {
            g_print ("FOUND INVALID PIXEL FORMAT\n");
        }

        self->width = GST_VIDEO_INFO_WIDTH (&info);
        self->height = GST_VIDEO_INFO_HEIGHT (&info);

        GST_DEBUG_OBJECT (self, "create new pool");

        g_mutex_lock (&self->flow_lock);
        pool = gst_nvds_buffer_pool_new();

        config = gst_buffer_pool_get_config (pool);
        GST_DEBUG_OBJECT (self, "in nvblender caps = %s\n", gst_caps_to_string(outcaps));
        gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), 4, 4);

        gst_structure_set(
            config, "memtype", G_TYPE_UINT, /*space->nvbuf_mem_type*/
            /*NVBUF_MEM_DEFAULT*/ NVBUF_MEM_CUDA_UNIFIED, "gpu-id", G_TYPE_UINT,
            /*space->gpu_id*/ self->gpu_id, "batch-size", G_TYPE_UINT,
            /*1*/ self->batch_size, "clear-chroma", G_TYPE_BOOLEAN, 1, NULL);

        if (!gst_buffer_pool_set_config (pool, config))
            g_print ("CONFIG FAILED\n");

        self->pool = gst_object_ref (pool);

        g_mutex_unlock (&self->flow_lock);
    }
    if (pool)
    {
        config = gst_buffer_pool_get_config (pool);
        gst_buffer_pool_config_get_allocator (config, &allocator, &params);
        gst_buffer_pool_config_get_params (config, &myoutcaps, &size, &minimum, &maximum);

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
            gst_query_set_nth_allocation_pool (query, 0, pool, size, minimum, maximum);
        } else {
            gst_query_add_allocation_pool (query, pool, size, minimum, maximum);
        }

        gst_structure_free (config);
        gst_object_unref (pool);
    }
    return TRUE;
}

GstFlowReturn finish_buffer(GstAggregator *aggregator,
                               GstBuffer *buffer)
{
  GstFlowReturn ret = GST_FLOW_OK;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if(!batch_meta){
    gst_buffer_unref(buffer);
  }
  else {
      ret = GST_AGGREGATOR_CLASS (parent_class)->finish_buffer (aggregator, buffer);
  }
  return ret;
}

void destroySurf (gpointer data){

  //GST_DEBUG_OBJECT("Destroying Cache BufSurface \n");
  NvBufSurfaceDestroy((NvBufSurface *)data);
}

gboolean start (GstAggregator *agg)
{
    GstNvblender *self = GST_NVBLENDER (agg);
    self->valid_bg_buf = g_hash_table_new_full(g_direct_hash,
    g_direct_equal,NULL, destroySurf);
    cudaStreamCreateWithFlags (&(self->cu_nbstream),
    cudaStreamNonBlocking);
    //self->valid_bg_buf = g_hash_table_new (NULL,NULL);
    return TRUE;
}

gboolean stop (GstAggregator *agg)
{
    GstNvblender *self = GST_NVBLENDER (agg);
    self->sinkpad0_active = FALSE;
    if (self->pool) {
      gst_object_unref(self->pool);
      self->pool = NULL;
    }
    if (self->valid_bg_buf) {
      g_hash_table_unref(self->valid_bg_buf);
      self->valid_bg_buf = NULL;
    }
    if (self->cu_nbstream)
    {
      cudaStreamDestroy(self->cu_nbstream);
      self->cu_nbstream = NULL;
    }

    return TRUE;
}


/* GObject boilerplate */
    static void
gst_nvblender_class_init (GstNvblenderClass * klass)
{
    GObjectClass *gobject_class = (GObjectClass *) klass;
    GstElementClass *gstelement_class = (GstElementClass *) klass;
    GstVideoAggregatorClass *videoaggregator_class =
        (GstVideoAggregatorClass *) klass;
  GstAggregatorClass *agg_class = (GstAggregatorClass *) klass;

  gobject_class->get_property = gst_nvblender_get_property;
  gobject_class->set_property = gst_nvblender_set_property;

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_compositor_request_new_pad);
  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_compositor_release_pad);
  agg_class->sink_query = _sink_query;
  agg_class->fixate_src_caps = _fixate_caps;
  agg_class->negotiated_src_caps = _negotiated_caps;
  agg_class->decide_allocation = GST_DEBUG_FUNCPTR (gst_nvblender_decide_allocation);
  videoaggregator_class->aggregate_frames = gst_nvblender_aggregate_frames;
  videoaggregator_class->find_best_format = _find_best_format;
  agg_class->finish_buffer = finish_buffer;
  agg_class->start = start;
  agg_class->stop = stop;
  agg_class->sink_event = _sink_event;

  g_object_class_install_property (gobject_class, PROP_BACKGROUND,
      g_param_spec_enum ("background", "Background", "Background type",
          GST_TYPE_COMPOSITOR_BACKGROUND,
          DEFAULT_BACKGROUND, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_BATCHED_BACKGROUND,
      g_param_spec_boolean ("batched-background", "Batched Background", "Background is batched",
          TRUE, G_PARAM_READWRITE));

  g_object_class_install_property(gobject_class, PROP_GPU_ID,
                                  g_param_spec_int("gpu-id", "GPU ID", "GPU ID",
                                                   0, 256, 1,
                                                   G_PARAM_READWRITE));

  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
      &src_factory, GST_TYPE_AGGREGATOR_PAD);
  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
      &sink_factory, GST_TYPE_NVBLENDER_PAD);

  gst_element_class_set_static_metadata (gstelement_class, "Compositor",
      "Filter/Editor/Video/Compositor",
      "Blend multiple video streams",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvblender_init (GstNvblender * self)
{
  /* initialize variables */
  self->background = DEFAULT_BACKGROUND;
  self->batched_background = TRUE;
  self->gpu_id = 0;
  g_mutex_init(&self->flow_lock);
}

/* GstChildProxy implementation */
static GObject *
gst_compositor_child_proxy_get_child_by_index (GstChildProxy * child_proxy,
    guint index)
{
  GstNvblender *compositor = GST_NVBLENDER (child_proxy);
  GObject *obj = NULL;

  GST_OBJECT_LOCK (compositor);
  obj = g_list_nth_data (GST_ELEMENT_CAST (compositor)->sinkpads, index);
  if (obj)
    gst_object_ref (obj);
  GST_OBJECT_UNLOCK (compositor);

  return obj;
}

static guint
gst_compositor_child_proxy_get_children_count (GstChildProxy * child_proxy)
{
  guint count = 0;
  GstNvblender *compositor = GST_NVBLENDER (child_proxy);

  GST_OBJECT_LOCK (compositor);
  count = GST_ELEMENT_CAST (compositor)->numsinkpads;
  GST_OBJECT_UNLOCK (compositor);
  GST_INFO_OBJECT (compositor, "Children Count: %d", count);

  return count;
}

static void
gst_compositor_child_proxy_init (gpointer g_iface, gpointer iface_data)
{
  GstChildProxyInterface *iface = g_iface;

  iface->get_child_by_index = gst_compositor_child_proxy_get_child_by_index;
  iface->get_children_count = gst_compositor_child_proxy_get_children_count;
}

/* Element registration */
static gboolean
plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvblender_debug, "nvblender", 0, "nvblender");

  return gst_element_register (plugin, "nvblender", GST_RANK_PRIMARY + 1,
      GST_TYPE_NVBLENDER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_nvblender,
    DESCRIPTION, plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE,
    URL)

