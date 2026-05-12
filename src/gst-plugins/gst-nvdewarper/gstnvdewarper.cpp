/**
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>

#include <string.h>
#include <unistd.h>
#include <sstream>
#include <vector>

#include "gstnvdsbufferpool.h"
#include "gstnvdewarper.h"

#include "nvbufsurface.h"
#include "nvdewarper.h"
#include "nvdewarper_property_parser.h"
#include "nvtx_helper.h"
#include "gst-nvcommon.h"
#include "nvdsmeta.h"
#include "nvbufsurftransform.h"

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

#define DEFAULT_NUM_DEWARPED_SURFACES (4)
#define DEFAULT_DEWARP_DUMP_FRAMES  0
#define DEFAULT_DEWARP_OUTPUT_WIDTH   960
#define DEFAULT_DEWARP_OUTPUT_HEIGHT  752

#define USE_CUDA_STREAM

#define GST_CAPS_FEATURE_MEMORY_NVMM      "memory:NVMM"

GST_DEBUG_CATEGORY_STATIC (gst_nvdewarper_debug);
#define GST_CAT_DEFAULT gst_nvdewarper_debug

#define DEFAULT_GPU_ID   0
#define DEFAULT_SOURCE_ID   0
#define DEFAULT_NUM_OUTPUT_BUFFERS  4
#define MAX_BUFFERS 4

#ifndef PACKAGE
#define PACKAGE "nvdewarper"
#endif

#define PACKAGE_DESCRIPTION "Gstreamer plugin to dewarp 360d surfaces"
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer nVidia Dewarper Plugin"
#define PACKAGE_URL "http://nvidia.com/"


//#define MEASURE_TIME
#ifdef MEASURE_TIME
#include <sys/time.h>
#include <stdio.h>

#define START_PROFILE \
    { \
  struct timeval t1, t2; \
  double elapsedTime = 0; \
  double totalReadTime = 0; \
  gettimeofday(&t1, NULL);

#define STOP_PROFILE(X) \
    gettimeofday(&t2, NULL); \
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      \
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   \
    totalReadTime += elapsedTime; \
    printf("(%s)  %p : #%d %s ElaspedTime=%f TotalTime=%f ms\n", \
        GST_ELEMENT_NAME(nvdewarper), nvdewarper, nvdewarper->frame_num, \
        X, elapsedTime, totalReadTime); \
    }

#else
#define START_PROFILE
#define STOP_PROFILE(X)
#endif

//Helper macros for alignment
#define NVBUF_ALIGN_VAL (256)
#define NVBUF_ALIGN_PITCH(pitch, align_val) ((pitch%align_val == 0) ? pitch:((pitch/align_val + 1)*align_val))
#define NVBUF_PLATFORM_ALIGNED_PITCH(pitch) NVBUF_ALIGN_PITCH(pitch, NVBUF_ALIGN_VAL)

static gchar DEWARPER_LIB_VERSION[128];


enum
{
  /* FILL ME */
  MEM_FEATURE_NVMM,
  MEM_FEATURE_RAW
};
/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_GPU_DEVICE_ID,
  PROP_SOURCE_ID,
  PROP_NUM_OUTPUT_BUFFERS,
  PROP_DEWARP_CONFIG_FILE,
  PROP_DEWARP_LIB_VERSION,
  PROP_NUM_BATCH_BUFFERS,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_INTERPOLATION_METHOD,
  PROP_SILENT,
};

static void gst_nvdewarper_finalize (GObject * object);

void __attribute__ ((constructor)) nvdewarper_libinit (void);
void __attribute__ ((destructor)) nvdewarper_libdeinit (void);

void
nvdewarper_libinit (void)
{
  unsigned version = gst_nvdewarper_version ();
  if (0 != (version & 0xFF))
    g_snprintf (DEWARPER_LIB_VERSION, 128, "%u.%u.%ud%u", version >> 24,
        (version >> 16) & 0xFF, (version >> 8) & 0xFF, version & 0xFF);
  else
    g_snprintf (DEWARPER_LIB_VERSION, 128, "%u.%u.%u", version >> 24,
        (version >> 16) & 0xFF, (version >> 8) & 0xFF);
  DEWARPER_LIB_VERSION[127] = '\0';
}

void
nvdewarper_libdeinit (void)
{
}

static const gchar *
print_pretty_time (gchar * ts_str, gsize ts_str_len, GstClockTime ts)
{
  if (ts == GST_CLOCK_TIME_NONE)
    return "none";

  g_snprintf (ts_str, ts_str_len, "%" GST_TIME_FORMAT, GST_TIME_ARGS (ts));
  return ts_str;
}


inline bool
NPP_CHECK_ (gint e, gint iLine, const gchar * szFile)
{
  if (e != NPP_SUCCESS) {
    std::
        cout << "Dewarper: NPP API error " << e << " at line " << iLine <<
        " in file " << szFile << endl;
    exit (-1);
    return false;
  }
  return true;
}

#define npp_ck(call) NPP_CHECK_(call, __LINE__, __FILE__)
/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
/* Input capabilities. */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "RGBA }")));

/* Output capabilities. */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "RGBA }")));

#define gst_nvdewarper_parent_class parent_class
G_DEFINE_TYPE (Gstnvdewarper, gst_nvdewarper, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdewarper_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);

static void gst_nvdewarper_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);


static gpointer
dewarper_meta_copy_func (gpointer data, gpointer user_data)
{
  NvDewarperSurfaceMeta *src_surface_meta = (NvDewarperSurfaceMeta *) data;
  NvDewarperSurfaceMeta *dst_surface_meta =
      (NvDewarperSurfaceMeta *) g_malloc0 (sizeof (NvDewarperSurfaceMeta));
  memcpy (dst_surface_meta, src_surface_meta, sizeof (NvDewarperSurfaceMeta));
  return (gpointer) dst_surface_meta;
}

static void
dewarper_meta_release_func (gpointer data, gpointer user_data)
{
  NvDewarperSurfaceMeta *surface_meta = (NvDewarperSurfaceMeta *) data;
  if (surface_meta) {
    g_free (surface_meta);
    surface_meta = NULL;
  }
}

static gpointer
dewarper_gst_to_nvds_meta_ransform_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDewarperSurfaceMeta *src_surface_meta =
      (NvDewarperSurfaceMeta *) user_meta->user_meta_data;
  NvDewarperSurfaceMeta *dst_surface_meta =
      (NvDewarperSurfaceMeta *) dewarper_meta_copy_func (src_surface_meta,
      NULL);
  return (gpointer) dst_surface_meta;
}

static void
dewarper_gst_nvds_meta_release_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDewarperSurfaceMeta *surface_meta =
      (NvDewarperSurfaceMeta *) user_meta->user_meta_data;
  dewarper_meta_release_func (surface_meta, NULL);
}

static gboolean
gst_nvdewarper_accept_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps)
{
  gboolean ret = TRUE;
  Gstnvdewarper *nvdewarper = NULL;
  GstCaps *allowed = NULL;

  nvdewarper = GST_NVDEWARPER (btrans);

  GST_DEBUG_OBJECT (nvdewarper, "accept caps %" GST_PTR_FORMAT, caps);

  /* get all the formats we can handle on this pad */
  if (direction == GST_PAD_SINK)
    allowed = nvdewarper->sinkcaps;
  else
    allowed = nvdewarper->srccaps;

  if (!allowed) {
    GST_DEBUG_OBJECT (nvdewarper, "failed to get allowed caps");
    goto no_transform_possible;
  }

  GST_DEBUG_OBJECT (nvdewarper, "allowed caps %" GST_PTR_FORMAT, allowed);

  /* intersect with the requested format */
  ret = gst_caps_is_subset (caps, allowed);
  if (!ret) {
    goto no_transform_possible;
  }

done:
  return ret;

/* ERRORS */
no_transform_possible:
  {
    GST_DEBUG_OBJECT (nvdewarper,
        "could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    ret = FALSE;
    goto done;
  }
}

static GstCaps *
gst_nvdewarper_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstStructure *ins, *outs;
  const GValue *from_par, *to_par;
  const gchar *from_fmt = NULL, *to_fmt = NULL;
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (trans);

  guint out_width, out_height;

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable (othercaps);

  GST_DEBUG_OBJECT (trans, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT " current direction is %s", othercaps, caps, (direction == GST_PAD_SINK) ? "sink" : "src");

  ins = gst_caps_get_structure (caps, 0);
  outs = gst_caps_get_structure (othercaps, 0);

  out_width = nvdewarper->output_width;
  out_height = nvdewarper->output_height;

  gst_structure_remove_fields (outs, "width", "height", NULL);

  gst_structure_set (outs, "width", G_TYPE_INT, out_width,
      "height", G_TYPE_INT, out_height, NULL);

  from_fmt = gst_structure_get_string (ins, "format");
  to_fmt = gst_structure_get_string (outs, "format");

  if (!to_fmt) {
    /* Output format not fixed */
    if (!gst_structure_fixate_field_string (outs, "format", from_fmt)) {
      return NULL;
    }
  }

  from_par = gst_structure_get_value (ins, "pixel-aspect-ratio");
  to_par = gst_structure_get_value (outs, "pixel-aspect-ratio");

  /* we have both PAR but they might not be fixated */
  if (from_par && to_par) {
    gint from_w = 0, from_h = 0, from_par_n = 0, from_par_d = 0, to_par_n = 0, to_par_d = 0;
    gint count = 0, w = 0, h = 0;
    guint num = 0, den = 0;

    /* from_par should be fixed */
    g_return_val_if_fail (gst_value_is_fixed (from_par), othercaps);

    from_par_n = gst_value_get_fraction_numerator (from_par);
    from_par_d = gst_value_get_fraction_denominator (from_par);

    /* fixate the out PAR */
    if (!gst_value_is_fixed (to_par)) {
      GST_DEBUG_OBJECT (trans, "fixating to_par to %dx%d", from_par_n,
          from_par_d);
      gst_structure_fixate_field_nearest_fraction (outs, "pixel-aspect-ratio",
          from_par_n, from_par_d);
    }

    to_par_n = gst_value_get_fraction_numerator (to_par);
    to_par_d = gst_value_get_fraction_denominator (to_par);

    /* if both width and height are already fixed, we can't do anything
     * about it anymore */
    if (gst_structure_get_int (outs, "width", &w))
      ++count;
    if (gst_structure_get_int (outs, "height", &h))
      ++count;
    if (count == 2) {
      GST_DEBUG_OBJECT (trans, "dimensions already set to %dx%d, not fixating",
          w, h);
      g_print ("%s: line=%d ---- %s\n", GST_ELEMENT_NAME (trans), __LINE__,
          gst_caps_to_string (othercaps));
      return othercaps;
    }

    gst_structure_get_int (ins, "width", &from_w);
    gst_structure_get_int (ins, "height", &from_h);

    if (!gst_video_calculate_display_ratio (&num, &den, from_w, from_h,
            from_par_n, from_par_d, to_par_n, to_par_d)) {
      GST_ELEMENT_ERROR (trans, CORE, NEGOTIATION, (NULL),
          ("Error calculating the output scaled size - integer overflow"));
      g_print ("%s: line=%d ---- %s\n", GST_ELEMENT_NAME (trans), __LINE__,
          gst_caps_to_string (othercaps));
      return othercaps;
    }

    GST_DEBUG_OBJECT (trans,
        "scaling input with %dx%d and PAR %d/%d to output PAR %d/%d",
        from_w, from_h, from_par_n, from_par_d, to_par_n, to_par_d);
    GST_DEBUG_OBJECT (trans, "resulting output should respect ratio of %d/%d",
        num, den);

    /* now find a width x height that respects this display ratio.
     * prefer those that have one of w/h the same as the incoming video
     * using wd / hd = num / den */

    /* if one of the output width or height is fixed, we work from there */
    if (h) {
      GST_DEBUG_OBJECT (trans, "height is fixed,scaling width");
      w = (guint) gst_util_uint64_scale_int (h, num, den);
    } else if (w) {
      GST_DEBUG_OBJECT (trans, "width is fixed, scaling height");
      h = (guint) gst_util_uint64_scale_int (w, den, num);
    } else {
      /* none of width or height is fixed, figure out both of them based only on
       * the input width and height */
      /* check hd / den is an integer scale factor, and scale wd with the PAR */
      if (from_h % den == 0) {
        GST_DEBUG_OBJECT (trans, "keeping video height");
        h = from_h;
        w = (guint) gst_util_uint64_scale_int (h, num, den);
      } else if (from_w % num == 0) {
        GST_DEBUG_OBJECT (trans, "keeping video width");
        w = from_w;
        h = (guint) gst_util_uint64_scale_int (w, den, num);
      } else {
        GST_DEBUG_OBJECT (trans, "approximating but keeping video height");
        h = from_h;
        w = (guint) gst_util_uint64_scale_int (h, num, den);
      }
    }
    GST_DEBUG_OBJECT (trans, "scaling to %dx%d", w, h);

    /* now fixate */
    gst_structure_fixate_field_nearest_int (outs, "width", w);
    gst_structure_fixate_field_nearest_int (outs, "height", h);
  } else {
    gint width, height;

    if (gst_structure_get_int (ins, "width", &width)) {
      if (gst_structure_has_field (outs, "width")) {
        gst_structure_fixate_field_nearest_int (outs, "width", width);
      }
    }
    if (gst_structure_get_int (ins, "height", &height)) {
      if (gst_structure_has_field (outs, "height")) {
        gst_structure_fixate_field_nearest_int (outs, "height", height);
      }
    }
  }
  if (direction == GST_PAD_SINK)
  {
    GstCaps *peer_caps = gst_pad_peer_query_caps(GST_BASE_TRANSFORM_SRC_PAD(trans), NULL);
    GstStructure *peer_structure;
    const gchar *out_mem_type_string = NULL;
    int n=gst_caps_get_size(peer_caps);
    bool peer_has_gpu_id = false;
    if(n>0)
    {
        peer_caps = gst_caps_truncate(peer_caps);
        GST_DEBUG_OBJECT (trans, "peer caps %" GST_PTR_FORMAT, peer_caps);
        peer_structure = gst_caps_get_structure (peer_caps, 0);
        out_mem_type_string = gst_structure_get_string (peer_structure, "nvbuf-memory-type");
        peer_has_gpu_id = gst_structure_has_field(peer_structure, "gpu-id");
    }

    if(!out_mem_type_string)
    {
        int mem_type = nvdewarper->cuda_mem_type;
        if(mem_type == NVBUF_MEM_DEFAULT)
            GET_DEFAULT_MEM_TYPE(mem_type);
        g_assert(mem_type != NVBUF_MEM_DEFAULT);
        gst_structure_set (outs, "nvbuf-memory-type", G_TYPE_STRING , gst_nvbuf_memory_get_name(mem_type), NULL);
    }
    else
    {
        nvdewarper->cuda_mem_type = (NvBufSurfaceMemType)gst_nvbuf_memory_get_value(out_mem_type_string);
        if(nvdewarper->cuda_mem_type <= NVBUF_MEM_DEFAULT)
            GST_ERROR_OBJECT(trans, "Incorrect nvbuf-memory-type set on src pad!!");
        GST_WARNING_OBJECT(trans, "nvbuf-memory-type property is set based on SRC caps. Property config setting (if any) is overridden!!");
        gst_structure_set (outs, "nvbuf-memory-type", G_TYPE_STRING , out_mem_type_string, NULL);
    }
    if(peer_has_gpu_id)
    {
        gst_structure_get_int(peer_structure, "gpu-id", (int *)&nvdewarper->gpu_id);
        GST_WARNING_OBJECT(trans, "gpu-id property is set based on SRC caps. Property config setting (if any) is overridden!!");
    }
    gst_structure_set (outs, "gpu-id", G_TYPE_INT , nvdewarper->gpu_id, NULL);
    gst_caps_unref(peer_caps);
  }

  GST_DEBUG_OBJECT (trans, "fixated othercaps to %" GST_PTR_FORMAT, othercaps);

  //g_print ("%s: line=%d ---- %s\n", GST_ELEMENT_NAME(trans), __LINE__, gst_caps_to_string(othercaps));
  return othercaps;
}


static GstCaps *
gst_nvdewarper_transform_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (btrans);
  GstCapsFeatures *feature = NULL;
  GstCaps *new_caps = NULL;
  GstCaps *temp_caps = NULL;

  if (direction == GST_PAD_SINK) {
    new_caps =
        gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
        "width", G_TYPE_INT, nvdewarper->output_width, "height", G_TYPE_INT,
        nvdewarper->output_height, NULL);
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (new_caps, 0, feature);
  }
  if (direction == GST_PAD_SRC) {
    new_caps =
        gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
        "width", GST_TYPE_INT_RANGE, 1, G_MAXINT, "height", GST_TYPE_INT_RANGE,
        1, G_MAXINT, NULL);
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (new_caps, 0, feature);
  }

  if (gst_caps_is_fixed (caps)) {
    GstStructure *fs = gst_caps_get_structure (caps, 0);
    const GValue *fps_value;
    guint i, n = gst_caps_get_size (new_caps);

    fps_value = gst_structure_get_value (fs, "framerate");

    // We cannot change framerate
    for (i = 0; i < n; i++) {
      fs = gst_caps_get_structure (new_caps, i);
      gst_structure_set_value (fs, "framerate", fps_value);
    }
  }
  if (filter)
  {
    temp_caps = gst_caps_intersect(new_caps, filter);
    gst_caps_unref(new_caps);
    new_caps = temp_caps;
  }
  return new_caps;
}

static gint
gst_nvdewarper_allocate_projection_buffers (Gstnvdewarper * nvdewarper)
{
  std::vector < NvDewarperParams >::iterator it;
  guint i = 0;
  cudaError_t cudaErr;

  cudaErr = cudaSetDevice (nvdewarper->gpu_id);
  if (cudaErr != cudaSuccess) {
    printf ("\n *** Unable to set device in %s Line %d\n", __func__, __LINE__);
    return cudaErr;
  }

  for (it = nvdewarper->priv->vecDewarpSurface.begin ();
      it != nvdewarper->priv->vecDewarpSurface.end (); it++) {

    //it->dewarpPitch = 4 * (((it->dewarpWidth) + 31) / 32) * 32;
    it->dewarpPitch = NVBUF_PLATFORM_ALIGNED_PITCH(it->dewarpWidth*4);

    cuda_ck (cudaMalloc (&it->surface, it->dewarpPitch * it->dewarpHeight));
    //cuda_ck (cudaMalloc (&it->surface, it->dewarpWidth * it->dewarpHeight * 4));

    GST_INFO_OBJECT (nvdewarper, "Allocated Surface %p for W=%d H=%d",
        it->surface, it->dewarpWidth, it->dewarpHeight);

    nvdewarper->surface_index[i] = it->surface_index;
    nvdewarper->surface_type[i++] = it->projection_type;
    if (it->projection_type == NVDS_META_SURFACE_FISH_PUSHBROOM) {
      nvdewarper->spot_surf_index[nvdewarper->num_spot_views] =
          nvdewarper->num_spot_views;
      nvdewarper->num_spot_views++;
    } else if (it->projection_type == NVDS_META_SURFACE_FISH_VERTCYL) {
      nvdewarper->aisle_surf_index[nvdewarper->num_aisle_views] =
          nvdewarper->num_aisle_views;
      nvdewarper->num_aisle_views++;
    }
  }
  return 0;
}

static gint
gst_nvdewarper_csv_init (Gstnvdewarper * nvdewarper)
{
  guint source_id = 0;
  guint i = 0;
  vector < gint > vec_spot_surf_index;
  vector < gint > vec_aisle_surf_index;
  NvDewarperParams surfaceParams;

  if ((nvdewarper->spotCSVInit == 1) && (nvdewarper->aisleCSVInit == 1)) {
    return 0;
  }

  GST_INFO_OBJECT (nvdewarper, " %s\n", __func__);

  source_id = nvdewarper->source_id;
  nvdewarper->spotCSVParser =
      new SpotCSVParser (nvdewarper->spot_calibration_file);
  nvdewarper->num_spot_views =
      nvdewarper->spotCSVParser->getNvSpotCSVMaxViews (source_id,
      &vec_spot_surf_index);

  nvdewarper->aisleCSVParser =
      new AisleCSVParser (nvdewarper->aisle_calibration_file);
  nvdewarper->num_aisle_views =
      nvdewarper->aisleCSVParser->getNvAisleCSVMaxViews (source_id,
      &vec_aisle_surf_index);

  g_assert ((nvdewarper->num_spot_views + nvdewarper->num_aisle_views) <=
      nvdewarper->num_batch_buffers);
  g_assert (nvdewarper->num_spot_views <= MAX_DEWARPED_VIEWS);
  g_assert (nvdewarper->num_aisle_views <= MAX_DEWARPED_VIEWS);

  if (nvdewarper->num_spot_views == 0) {
    GST_WARNING_OBJECT (nvdewarper,
        "For source_id=%d NO SPOT Views Found for Dewarping\n", source_id);
  }
  if (nvdewarper->num_aisle_views == 0) {
    GST_WARNING_OBJECT (nvdewarper,
        "For source_id=%d NO AISLE Views Found for Dewarping\n", source_id);
  }
  //if (nvdewarper->spotCSVInit == 0)
  {
    for (i = 0; i < nvdewarper->num_spot_views; i++) {
      NvSpotCsvFields fields = { 0 };
      guint surf_idx = vec_spot_surf_index.at (i);

      memset (&surfaceParams, 0, sizeof (surfaceParams));
      surfaceParams.control = 0.6f; // For pushbroom projection set default control = 0.6 to maintain legacy

      //taking value for spot index 0
      if (nvdewarper->spotCSVParser->getNvSpotCSVFields (source_id, surf_idx, 0,
              &fields) != 0) {
        g_print
            ("%s: SPOT Entry for cam=%d, view=%d surface=0 and spot_index=%d "
            "not found in calibration file.\n", GST_ELEMENT_NAME (nvdewarper),
            source_id, i, 0);
        surfaceParams.isValid = 0;
        continue;
      }

      surfaceParams.top_angle = fields.dewarpTopAngle;
      surfaceParams.bottom_angle = fields.dewarpBottomAngle;
      surfaceParams.yaw = fields.dewarpYaw;
      surfaceParams.roll = fields.dewarpRoll;
      surfaceParams.pitch = fields.dewarpPitch;
      surfaceParams.dewarpFocalLength[0] = fields.dewarpFocalLength;
      surfaceParams.dewarpWidth = fields.dewarpWidth;
      surfaceParams.dewarpHeight = fields.dewarpHeight;
      surfaceParams.surface_index = surf_idx;
      surfaceParams.isValid = 1;

      //Assuming all the Spot surfaces will have same width and height
      nvdewarper->spot_surf_index[i] = surf_idx;

      surfaceParams.projection_type = NVDS_META_SURFACE_FISH_PUSHBROOM;
      nvdewarper->priv->vecDewarpSurface.push_back (surfaceParams);
    }
    nvdewarper->spotCSVInit = 1;
  }

  //if (nvdewarper->aisleCSVInit == 0)
  {
    for (i = 0; i < nvdewarper->num_aisle_views; i++) {
      NvAisleCsvFields fields = { 0 };
      guint surf_idx = vec_aisle_surf_index.at (i);

      memset (&surfaceParams, 0, sizeof (surfaceParams));

      if (nvdewarper->aisleCSVParser->getNvAisleCSVFields (source_id, surf_idx,
              &fields) != 0) {
        g_print
            ("%s: Aisle Entry for cam=%d, surface=%d not found in calibration "
            "file.\n", GST_ELEMENT_NAME (nvdewarper), source_id, 0);
        surfaceParams.isValid = 0;
        continue;
      }

      surfaceParams.top_angle = fields.dewarpTopAngle;
      surfaceParams.bottom_angle = fields.dewarpBottomAngle;
      surfaceParams.pitch = fields.dewarpPitch;
      surfaceParams.yaw = fields.dewarpYaw;
      surfaceParams.roll = fields.dewarpRoll;
      surfaceParams.dewarpFocalLength[0] = fields.dewarpFocalLength;
      surfaceParams.dewarpWidth = fields.dewarpWidth;
      surfaceParams.dewarpHeight = fields.dewarpHeight;
      surfaceParams.surface_index = surf_idx;
      surfaceParams.isValid = 1;

      //Assuming all the Aisle surfaces will have same width and height
      nvdewarper->aisle_surf_index[i] = surf_idx;

      surfaceParams.projection_type = NVDS_META_SURFACE_FISH_VERTCYL;
      nvdewarper->priv->vecDewarpSurface.push_back (surfaceParams);
    }
    nvdewarper->aisleCSVInit = 1;
  }
  gst_nvdewarper_allocate_projection_buffers (nvdewarper);
  return 0;
}

static gboolean
gst_nvdewarper_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (trans);
  GstCapsFeatures *ift = NULL;
  GstStructure *config = NULL;
  GstVideoInfo in_info = {}, out_info = {};

  GST_DEBUG_OBJECT (nvdewarper, "set_caps");

  if (!gst_video_info_from_caps (&in_info, incaps)) {
    GST_ERROR ("invalid input caps");
    return FALSE;
  }
  nvdewarper->input_width = GST_VIDEO_INFO_WIDTH (&in_info);
  nvdewarper->input_height = GST_VIDEO_INFO_HEIGHT (&in_info);
  nvdewarper->input_fmt =
      GST_VIDEO_FORMAT_INFO_FORMAT (in_info.finfo);


  if (!gst_video_info_from_caps (&out_info, outcaps)) {
    GST_ERROR ("invalid output caps");
    return FALSE;
  }
  //nvdewarper->output_width = GST_VIDEO_INFO_WIDTH (&nvdewarper->out_info);
  //nvdewarper->output_height = GST_VIDEO_INFO_HEIGHT (&nvdewarper->out_info);
  nvdewarper->output_fmt =
      GST_VIDEO_FORMAT_INFO_FORMAT (out_info.finfo);

  ift = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_NVMM, NULL);
  if (gst_caps_features_is_equal (gst_caps_get_features (outcaps, 0), ift)) {
    nvdewarper->output_feature = MEM_FEATURE_NVMM;
  } else {
    nvdewarper->output_feature = MEM_FEATURE_RAW;
  }

  if (gst_caps_features_is_equal (gst_caps_get_features (incaps, 0), ift)) {
    nvdewarper->input_feature = MEM_FEATURE_NVMM;
  } else {
    nvdewarper->input_feature = MEM_FEATURE_RAW;
  }
  gst_caps_features_free (ift);

  // Pool Creation
  {
    nvdewarper->pool = gst_nvds_buffer_pool_new ();

    config = gst_buffer_pool_get_config (nvdewarper->pool);

    //g_print ("in videoconvert caps = %s\n", gst_caps_to_string (outcaps));
    gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface),
        nvdewarper->num_output_buffers, nvdewarper->num_output_buffers);

    gst_structure_set (config,
        "memtype", G_TYPE_UINT, nvdewarper->cuda_mem_type,
        "gpu-id", G_TYPE_UINT, nvdewarper->gpu_id,
        "batch-size", G_TYPE_UINT, nvdewarper->num_batch_buffers, NULL);

    /* set config for the created buffer pool */
    if (!gst_buffer_pool_set_config (nvdewarper->pool, config)) {
      GST_WARNING ("bufferpool configuration failed");
      return FALSE;
    }

    gboolean is_active = gst_buffer_pool_set_active (nvdewarper->pool, TRUE);
    if (!is_active) {
      GST_WARNING (" Failed to allocate the buffers inside the output pool");
      return FALSE;
    } else {
      GST_DEBUG
          (" Output buffer pool (%p) successfully created with %d buffers",
          nvdewarper->pool, nvdewarper->num_batch_buffers);
    }
  }
  if (!nvdewarper->aisle_calibrationfile_set ||
    !nvdewarper->spot_calibrationfile_set) {
    // Non-CVS Case
    gst_nvdewarper_allocate_projection_buffers (nvdewarper);
  }

  gst_base_transform_set_passthrough (trans, FALSE);
  return TRUE;
}


static cudaError
gst_nvdewarper_generate_output (Gstnvdewarper * nvdewarper,
    NvBufSurface * in_surface, NvBufSurface * out_surface)
{
  gchar context_name[100];
  std::vector < NvDewarperParams >::iterator it;
  NvDewarperParams *dewarpParams = NULL;
  guint i = 0;
  cudaError err = cudaSuccess;
  NvBufSurface in_surf = {0};
  NvBufSurfaceParams surfaceList[MAX_DEWARPED_VIEWS];
  NvBufSurfTransformRect dstRect[MAX_DEWARPED_VIEWS];
  NvBufSurfTransformRect srcRect[MAX_DEWARPED_VIEWS];
  gfloat xscale = 1.0;
  in_surf.gpuId = nvdewarper->gpu_id;
  in_surf.batchSize = 1;
  in_surf.numFilled = 1;
  in_surf.memType = NVBUF_MEM_CUDA_DEVICE;
  in_surf.surfaceList = &surfaceList[0];


  NvDewarperSurfaceMeta *surface_meta =
      (NvDewarperSurfaceMeta *) calloc (1, sizeof (NvDewarperSurfaceMeta));

  out_surface->numFilled = 0;

  for (it = nvdewarper->priv->vecDewarpSurface.begin ();
      it != nvdewarper->priv->vecDewarpSurface.end (); it++) {
    if (i == nvdewarper->num_batch_buffers)
      break;

    dewarpParams = &(*it);
    //cout << it->projection_type << " " << it->dewarpWidth << " " << it->dewarpHeight << endl;

    snprintf (context_name, sizeof (context_name), "%s_(Frame=%u)",
        GST_ELEMENT_NAME (nvdewarper), nvdewarper->frame_num);
    nvtx_helper_push_pop (strcat (context_name, "_Scale"));

    guint dstWidth = dewarpParams->dewarpWidth;
    guint dstHeight = dewarpParams->dewarpHeight;
    NppiSize inSrcSize = { (gint) dstWidth, (gint) dstHeight };

    guint *src = (guint *) dewarpParams->surface;

    surface_meta->type[i] = dewarpParams->projection_type;
    surface_meta->index[i] = dewarpParams->surface_index;

    // Create Dummy Input Surface
    {
      guint bytesPerPixel = 4;
      memset(&surfaceList[i], 0, sizeof(surfaceList[i]));

      in_surf.surfaceList[i].pitch  = dewarpParams->dewarpPitch;
      in_surf.surfaceList[i].colorFormat = NVBUF_COLOR_FORMAT_RGBA;
      in_surf.surfaceList[i].width  = inSrcSize.width;
      in_surf.surfaceList[i].height = inSrcSize.height;
      in_surf.surfaceList[i].planeParams.num_planes = 1;
      in_surf.surfaceList[i].planeParams.width[0] = inSrcSize.width;
      in_surf.surfaceList[i].planeParams.height[0] = inSrcSize.height;
      in_surf.surfaceList[i].planeParams.pitch[0] = in_surf.surfaceList[i].pitch;
      in_surf.surfaceList[i].planeParams.psize[0] = inSrcSize.height*in_surf.surfaceList[i].pitch;
      in_surf.surfaceList[i].planeParams.bytesPerPix[0] = bytesPerPixel;

      in_surf.surfaceList[i].dataSize = in_surf.surfaceList[i].planeParams.psize[0];
      in_surf.surfaceList[i].dataPtr = src;
      in_surf.surfaceList[i].layout = NVBUF_LAYOUT_PITCH;

      xscale = ((gfloat) out_surface->surfaceList[i].planeParams.width[0]) / inSrcSize.width;
      dstRect[i].top  = 0;
      dstRect[i].left = 0;
      dstRect[i].width  = out_surface->surfaceList[i].planeParams.width[0];
      dstRect[i].height = (inSrcSize.height * xscale + 0.5);
      srcRect[i].top = 0;
      srcRect[i].left = 0;
      srcRect[i].width = inSrcSize.width;
      srcRect[i].height = inSrcSize.height;

      // For 360D to match close to Aisle output of NPP o/p, which crops when
      // provided scale factor exceeds beyond dst image,
      // calculate the corresponding limit in src
      // to maintain aspect ratio and thus cropping the image
      if (inSrcSize.height*xscale >
                out_surface->surfaceList[i].planeParams.height[0])
      {
          dstRect[i].height =  out_surface->surfaceList[i].planeParams.height[0];
          srcRect[i].height = (dstRect[i].height/xscale + 0.5);
      }

      i++;
    }

    nvtx_helper_push_pop (NULL);
  }


  // Perform NvBufTransform
  NvBufSurfTransformParams transform_params;
  transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER |
                                    NVBUFSURF_TRANSFORM_CROP_DST |
                                    NVBUFSURF_TRANSFORM_CROP_SRC;
  transform_params.transform_flip = NvBufSurfTransform_None;
  transform_params.transform_filter = nvdewarper->interpolation_method;
  transform_params.src_rect = &srcRect[0];
  transform_params.dst_rect = &dstRect[0];
  in_surf.numFilled = i;
  in_surf.batchSize = i;

  NvBufSurfTransform_Error tx_err = NvBufSurfTransformError_Success;
  NvBufSurfTransformConfigParams config_params;
  config_params.compute_mode = NvBufSurfTransformCompute_GPU;
  config_params.gpu_id = nvdewarper->gpu_id;
  config_params.cuda_stream = nvdewarper->stream;

  tx_err = NvBufSurfTransformSetSessionParams (&config_params);
  if (tx_err != NvBufSurfTransformError_Success)
  {
    g_print("%s: %d NvBufSurfTransform set session failed\n", __func__, __LINE__);
    g_free(surface_meta);
    return cudaErrorInvalidSurface;
  }

  tx_err = NvBufSurfTransform (&in_surf, out_surface, &transform_params);
  if (tx_err != NvBufSurfTransformError_Success){
    g_print("%s: %d NvBufSurfTransform failed\n", __func__, __LINE__);
    g_free(surface_meta);
    return cudaErrorInvalidSurface;
  }
  out_surface->numFilled = i;

  surface_meta->num_filled_surfaces = i;
  surface_meta->source_id = nvdewarper->source_id;

  NvDsMeta *meta = NULL;
  meta = gst_buffer_add_nvds_meta (nvdewarper->out_gst_buf, surface_meta, NULL,
      dewarper_meta_copy_func, dewarper_meta_release_func);

  meta->meta_type = NVDS_DEWARPER_GST_META;
  meta->gst_to_nvds_meta_transform_func =
      dewarper_gst_to_nvds_meta_ransform_func;
  meta->gst_to_nvds_meta_release_func = dewarper_gst_nvds_meta_release_func;

  return err;
}


static cudaError
gst_nvdewarper_dewarp (Gstnvdewarper * nvdewarper, NvBufSurface * in_surface,
    NvBufSurface * out_surface)
{
  cudaError cudaErr = cudaSuccess;
  gint err = 0;

  err = err;
  GST_LOG_OBJECT (nvdewarper,
      "SETTING CUDA DEVICE = %d in nvdewarper func=%s\n", nvdewarper->gpu_id,
      __func__);
  cudaErr = cudaSetDevice (nvdewarper->gpu_id);
  if (cudaErr != cudaSuccess) {
    printf ("\n *** Unable to set device in %s Line %d\n", __func__, __LINE__);
    return cudaErr;
  }

  if ((nvdewarper->aisle_calibrationfile_set == TRUE)
      || (nvdewarper->spot_calibrationfile_set == TRUE)) {
    if ((nvdewarper->spotCSVInit == 0) && (nvdewarper->aisleCSVInit == 0)) {
      gst_nvdewarper_csv_init (nvdewarper);
    }
  }
  // Do Dewarping of all surfaces
  cuda_ck (gst_nvdewarper_do_dewarp (nvdewarper, in_surface, out_surface));

  if (nvdewarper->output_fmt == GST_VIDEO_FORMAT_NV12 ||
      nvdewarper->output_fmt == GST_VIDEO_FORMAT_NV21) {
    // RGBA ---> NV12 conversion
    g_print
        ("RGBA to NV12 conversion is not supported in dewarper. Exiting...\n");
    exit (-1);
  } else if (nvdewarper->output_fmt == GST_VIDEO_FORMAT_RGBA ||
      nvdewarper->output_fmt == GST_VIDEO_FORMAT_BGRx) {
    // Generate output surface after scaling
    cuda_ck (gst_nvdewarper_generate_output (nvdewarper, in_surface,
            out_surface));
  }

  if (nvdewarper->dump_frames) {
    nvdewarper->dump_frames--;
  }

  BAIL_IF_FALSE (cudaSuccess == cudaErr, err, (gint) cudaErr);
  return cudaErr;

bail:
  g_print ("%s: %s failed at line %d, Error : %d Exiting ...\n",
      GST_ELEMENT_NAME (nvdewarper), __func__, __LINE__, cudaErr);
  exit (-1);
  return cudaErr;
}

static GstFlowReturn
gst_nvdewarper_transform (GstBaseTransform * btrans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (btrans);
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  GstMapInfo outmap = GST_MAP_INFO_INIT;
  NvBufSurface *in_surface = NULL;
  NvBufSurface *out_surface = NULL;
  cudaError cudaErr = cudaSuccess;
  gchar pts_str[64];

  nvdewarper->frame_num++;
  GST_DEBUG_OBJECT (nvdewarper, "%s : Frame=%d InBuf=%p OutBuf=%p\n",
      __func__, nvdewarper->frame_num, inbuf, outbuf);

  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ))
    goto invalid_inbuf;

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE))
    goto invalid_outbuf;

  GST_DEBUG_OBJECT (nvdewarper, "transform");
  if (nvdewarper->input_feature == MEM_FEATURE_NVMM) {
    in_surface = (NvBufSurface *) inmap.data;
    // TODO:
    if (CHECK_NVDS_MEMORY_AND_GPUID (nvdewarper, in_surface)) {
      gst_buffer_unmap (inbuf, &inmap);
      gst_buffer_unmap (outbuf, &outmap);
      return GST_FLOW_ERROR;
    }
  }

  if (nvdewarper->output_feature == MEM_FEATURE_NVMM) {
    out_surface = (NvBufSurface *) outmap.data;
    if (CHECK_NVDS_MEMORY_AND_GPUID (nvdewarper, out_surface)) {
      gst_buffer_unmap (inbuf, &inmap);
      gst_buffer_unmap (outbuf, &outmap);
      return GST_FLOW_ERROR;
    }
  }

  START_PROFILE;
  nvdewarper->out_gst_buf = outbuf;
  cudaErr = gst_nvdewarper_dewarp (nvdewarper, in_surface, out_surface);
  if (cudaErr != cudaSuccess) {
    GST_ERROR_OBJECT (nvdewarper, "gst_nvdewarper_dewarp failed");
    return GST_FLOW_ERROR;
  }
  STOP_PROFILE ("********* TOTAL DEWARP AND SCALE TIME *********");

  GST_BUFFER_PTS (outbuf) = GST_BUFFER_PTS (inbuf);

  GST_INFO_OBJECT (nvdewarper, " : source_id %d Frame=%d OUT-BUFFER %s",
      nvdewarper->source_id, nvdewarper->frame_num, print_pretty_time (pts_str,
          sizeof (pts_str), GST_BUFFER_PTS (outbuf)));

  gst_buffer_unmap (inbuf, &inmap);
  gst_buffer_unmap (outbuf, &outmap);

  if (!gst_buffer_copy_into (outbuf, inbuf,
          (GstBufferCopyFlags) GST_BUFFER_COPY_METADATA, 0, -1)) {
    GST_DEBUG_OBJECT (nvdewarper, "Buffer metadata copy failed \n");
  }
  return GST_FLOW_OK;

invalid_inbuf:
  {
    GST_ERROR ("input buffer mapinfo failed");
    return GST_FLOW_ERROR;
  }

invalid_outbuf:
  {
    GST_ERROR_OBJECT (nvdewarper, "output buffer mapinfo failed");
    gst_buffer_unmap (inbuf, &inmap);
    return GST_FLOW_ERROR;
  }
}

static GstFlowReturn
gst_nvdewarper_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf)
{
  GstBuffer *gstOutBuf = NULL;
  GstFlowReturn result = GST_FLOW_OK;
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (trans);

  result = gst_buffer_pool_acquire_buffer (nvdewarper->pool, &gstOutBuf, NULL);
  GST_DEBUG_OBJECT (nvdewarper, "%s : Frame=%d Gst-OutBuf=%p\n", __func__,
      nvdewarper->frame_num, gstOutBuf);

  if (result != GST_FLOW_OK) {
    GST_ERROR_OBJECT (nvdewarper,
        "gst_nvdewarper_prepare_output_buffer failed");
    return result;
  }

  *outbuf = gstOutBuf;
  return result;
}

static gboolean
gst_nvdewarper_start (GstBaseTransform * btrans)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (btrans);
  cudaError_t CUerr = cudaSuccess;

  GST_INFO_OBJECT (nvdewarper, "Using libNVWarp360 version: %s",
      DEWARPER_LIB_VERSION);

  nvdewarper->frame_num = 0;

  if (nvdewarper->spot_calibration_file) {
    std::ifstream spot_infile (nvdewarper->spot_calibration_file);
    if (!spot_infile.good ()) {
      g_print ("%s: Spot Calibration File (%s) not found\n",
          GST_ELEMENT_NAME (nvdewarper), nvdewarper->spot_calibration_file);
      return FALSE;
    }
  }

  if (nvdewarper->aisle_calibration_file) {
    std::ifstream aisle_infile (nvdewarper->aisle_calibration_file);
    if (!aisle_infile.good ()) {
      g_print ("%s: Aisle Calibration File (%s) not found\n",
          GST_ELEMENT_NAME (nvdewarper), nvdewarper->aisle_calibration_file);
      return FALSE;
    }
  }

  GST_LOG_OBJECT (nvdewarper,
      "SETTING CUDA DEVICE = %d in nvdewarper func=%s\n", nvdewarper->gpu_id,
      __func__);
  CUerr = cudaSetDevice (nvdewarper->gpu_id);
  if (CUerr != cudaSuccess) {
    GST_ERROR_OBJECT (nvdewarper, "cudaSetDevice Failed in %s\n", __func__);
    return FALSE;
  }
  cuda_ck (cudaStreamCreate (&(nvdewarper->stream)));

  return TRUE;
}

static gboolean
gst_nvdewarper_stop (GstBaseTransform * btrans)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (btrans);
  cudaError_t CUerr = cudaSuccess;

  GST_INFO_OBJECT (nvdewarper, " %s\n", __func__);

  GST_LOG_OBJECT (nvdewarper,
      "SETTING CUDA DEVICE = %d in nvdewarper func=%s\n", nvdewarper->gpu_id,
      __func__);
  CUerr = cudaSetDevice (nvdewarper->gpu_id);
  if (CUerr != cudaSuccess) {
    GST_ERROR_OBJECT (nvdewarper, "cudaSetDevice Failed in %s\n", __func__);
    return FALSE;
  }

  if (nvdewarper->stream) {
    cuda_ck (cudaStreamDestroy (nvdewarper->stream));
    nvdewarper->stream = NULL;
  }

  if (nvdewarper->pool) {
    gst_buffer_pool_set_active (nvdewarper->pool, FALSE);
    gst_object_unref (nvdewarper->pool);
    nvdewarper->pool = NULL;
  }


  GST_DEBUG_OBJECT (nvdewarper, "gst_nvdewarper_stop");


  return TRUE;
}

/* initialize the nvdewarper's class */
static void
gst_nvdewarper_class_init (GstnvdewarperClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class =
      (GstBaseTransformClass *) klass;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class->set_property = gst_nvdewarper_set_property;
  gobject_class->get_property = gst_nvdewarper_get_property;
  gobject_class->finalize = gst_nvdewarper_finalize;

  gstbasetransform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvdewarper_transform_caps);
  gstbasetransform_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_nvdewarper_fixate_caps);
  gstbasetransform_class->accept_caps =
      GST_DEBUG_FUNCPTR (gst_nvdewarper_accept_caps);
  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_nvdewarper_set_caps);

  gstbasetransform_class->transform =
      GST_DEBUG_FUNCPTR (gst_nvdewarper_transform);
  gstbasetransform_class->prepare_output_buffer =
      GST_DEBUG_FUNCPTR (gst_nvdewarper_prepare_output_buffer);

  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdewarper_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdewarper_stop);

  gstbasetransform_class->passthrough_on_same_caps = FALSE;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, DEFAULT_GPU_ID,
          GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SOURCE_ID,
      g_param_spec_uint ("source-id", "Set Source / Camera ID",
          "Set Source / Camera ID",
          0, G_MAXUINT, DEFAULT_SOURCE_ID,
          GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_NUM_OUTPUT_BUFFERS,
      g_param_spec_uint ("num-output-buffers", "Number of Output Buffers",
          "Number of Output Buffers",
          0, G_MAXUINT, DEFAULT_NUM_OUTPUT_BUFFERS,
          GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_NUM_BATCH_BUFFERS,
      g_param_spec_uint ("num-batch-buffers", "Number of Surfaces per output "
          "Buffer", "Number of Surfaces per output Buffer",
          0, MAX_BUFFERS, DEFAULT_NUM_DEWARPED_SURFACES,
          GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_DEWARP_CONFIG_FILE,
      g_param_spec_string ("config-file", "Dewarper Config File",
          "Dewarper Config File",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DEWARP_LIB_VERSION,
      g_param_spec_string ("dewarper-lib-version", "Dewarper Library Version",
          "Dewarper Library Version",
          NULL, (GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);
  PROP_INTERPOLATION_METHOD_INSTALL(gobject_class);

  gst_element_class_set_details_simple (gstelement_class,
      "nvdewarper",
      "nvdewarper",
      "Gstreamer NVDEWARPER Element",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_nvdewarper_init (Gstnvdewarper * nvdewarper)
{
  nvdewarper->sinkcaps = gst_static_pad_template_get_caps (&sink_factory);
  nvdewarper->srccaps = gst_static_pad_template_get_caps (&src_factory);

  nvdewarper->silent = FALSE;
  nvdewarper->pool = NULL;

  nvdewarper->num_batch_buffers = DEFAULT_NUM_DEWARPED_SURFACES;
  nvdewarper->cuda_mem_type = NVBUF_MEM_DEFAULT;
  nvdewarper->interpolation_method = NvBufSurfTransformInter_Default;

  // TODO: If CSV is not given then we should not check this
  nvdewarper->aisle_calibrationfile_set = FALSE;
  nvdewarper->spot_calibrationfile_set = FALSE;
  nvdewarper->aisleCSVInit = 0;
  nvdewarper->spotCSVInit = 0;
  nvdewarper->config_file = NULL;
  nvdewarper->num_output_buffers = DEFAULT_NUM_OUTPUT_BUFFERS;

  nvdewarper->dump_frames = DEFAULT_DEWARP_DUMP_FRAMES;

  nvdewarper->output_width = DEFAULT_DEWARP_OUTPUT_WIDTH;
  nvdewarper->output_height = DEFAULT_DEWARP_OUTPUT_HEIGHT;

  nvdewarper->num_spot_views = 0;
  nvdewarper->num_aisle_views = 0;

  nvdewarper->priv = new NvDewarperPriv;
}

static void
gst_nvdewarper_finalize (GObject * object)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (object);
  std::vector < NvDewarperParams >::iterator it;

  for (it = nvdewarper->priv->vecDewarpSurface.begin ();
      it != nvdewarper->priv->vecDewarpSurface.end (); it++) {
        if (it->surface) {
          cuda_ck (cudaFree (it->surface));
          it->surface = NULL;
        }
  }
  if(nvdewarper->aisle_output){
    cuda_ck (cudaFreeHost (nvdewarper->aisle_output));
    nvdewarper->aisle_output = NULL;
  }
  if(nvdewarper->spot_output){
    cuda_ck (cudaFreeHost (nvdewarper->spot_output));
    nvdewarper->spot_output = NULL;
  }

  if(nvdewarper->output){
    cuda_ck (cudaFreeHost (nvdewarper->output));
    nvdewarper->output = NULL;
  }

  nvdewarper->priv->vecDewarpSurface.clear ();
  if (nvdewarper->priv) {
    delete nvdewarper->priv;
    nvdewarper->priv = NULL;
  }
  if (nvdewarper->spotCSVParser) {
    delete nvdewarper->spotCSVParser;
    nvdewarper->spotCSVParser = NULL;
    nvdewarper->spotCSVInit = 0;
  }
  if (nvdewarper->aisleCSVParser) {
    delete nvdewarper->aisleCSVParser;
    nvdewarper->aisleCSVParser = NULL;
    nvdewarper->aisleCSVInit = 0;
  }
  if (nvdewarper->config_file)
    g_free (nvdewarper->config_file);
}

static void
gst_nvdewarper_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (object);

  switch (prop_id) {
    case PROP_SILENT:
      nvdewarper->silent = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvdewarper->gpu_id = g_value_get_uint (value);
      break;
    case PROP_SOURCE_ID:
      nvdewarper->source_id = g_value_get_uint (value);
      break;
    case PROP_NUM_OUTPUT_BUFFERS:
      nvdewarper->num_output_buffers = g_value_get_uint (value);
      break;
    case PROP_NUM_BATCH_BUFFERS:
      nvdewarper->num_batch_buffers = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvdewarper->cuda_mem_type = static_cast<NvBufSurfaceMemType>(g_value_get_enum (value));
      break;
    case PROP_INTERPOLATION_METHOD:
      nvdewarper->interpolation_method = static_cast<NvBufSurfTransform_Inter>(g_value_get_enum (value));
      break;
    case PROP_DEWARP_CONFIG_FILE:
      if (nvdewarper->config_file)
        g_free (nvdewarper->config_file);
      nvdewarper->config_file = (gchar *) g_value_dup_string (value);
      if (nvdewarper_parse_config_file (nvdewarper,
              nvdewarper->config_file) != TRUE) {
        g_print ("%s: Failed to parse config file %s\n",
            GST_ELEMENT_NAME (nvdewarper), nvdewarper->config_file);
        abort ();
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvdewarper_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstnvdewarper *nvdewarper = GST_NVDEWARPER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, nvdewarper->silent);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvdewarper->gpu_id);
      break;
    case PROP_SOURCE_ID:
      g_value_set_uint (value, nvdewarper->source_id);
      break;
    case PROP_NUM_OUTPUT_BUFFERS:
      g_value_set_uint (value, nvdewarper->num_output_buffers);
      break;
    case PROP_NUM_BATCH_BUFFERS:
      g_value_set_uint (value, nvdewarper->num_batch_buffers);
      break;
    case PROP_DEWARP_CONFIG_FILE:
      g_value_set_string (value, nvdewarper->config_file);
      break;
    case PROP_DEWARP_LIB_VERSION:
      g_value_set_static_string (value, DEWARPER_LIB_VERSION);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvdewarper->cuda_mem_type);
      break;
    case PROP_INTERPOLATION_METHOD:
      g_value_set_enum (value, nvdewarper->interpolation_method);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* GstElement vmethod implementations */

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
nvdewarper_init (GstPlugin * nvdewarper)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template nvdewarper' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_nvdewarper_debug, "nvdewarper", 0, "nvdewarper");

  return gst_element_register (nvdewarper, "nvdewarper", GST_RANK_NONE,
      GST_TYPE_NVDEWARPER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dewarper,
    PACKAGE_DESCRIPTION,
    nvdewarper_init, "9.0", PACKAGE_LICENSE, PACKAGE_NAME, PACKAGE_URL)
