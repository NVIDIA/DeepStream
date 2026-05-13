/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <gst/gst.h>

#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include "gstnvdsosd.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "nvtx3/nvToolsExt.h"
#include "gst-nvdscustomevent.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvds_osd_debug);
#define GST_CAT_DEFAULT gst_nvds_osd_debug

/* For hw blending, color should be of the form:
   class_id1, R, G, B, A:class_id2, R, G, B, A */
#define DEFAULT_CLR "0,0.0,1.0,0.0,0.3:1,0.0,1.0,1.0,0.3:2,0.0,0.0,1.0,0.3:3,1.0,1.0,0.0,0.3"
#define MAX_OSD_ELEMS 1024

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_SHOW_CLOCK,
  PROP_SHOW_TEXT,
  PROP_CLOCK_FONT,
  PROP_CLOCK_FONT_SIZE,
  PROP_CLOCK_X_OFFSET,
  PROP_CLOCK_Y_OFFSET,
  PROP_CLOCK_COLOR,
  PROP_PROCESS_MODE,
  PROP_GPU_DEVICE_ID,
  PROP_SHOW_BBOX,
  PROP_SHOW_MASK,
  PROP_BLUR_GIE_CLASS_IDS,
  PROP_BLUR_BBOX
};

/**
 * Defines Blur mode.
 */
typedef enum
{
  /** Disable blur. */
  Blur_Disable = 0,
  /** Blur the entire bbox area of object. */
  Blur_Bbox,
  /** Blur the mask area of object if there is a mask, swith to Blur_Bbox if no mask. */
  Blur_Mask
} NvDsOsdBlurMode;

/* the capabilities of the inputs and outputs. */
static GstStaticPadTemplate nvdsosd_sink_factory =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

static GstStaticPadTemplate nvdsosd_src_factory =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

/* Default values for properties */
#define DEFAULT_FONT_SIZE 12
#define DEFAULT_FONT "Serif"
#define GST_NV_OSD_DEFAULT_PROCESS_MODE MODE_GPU
#define MAX_FONT_SIZE 60
#define DEFAULT_BORDER_WIDTH 4

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvds_osd_parent_class parent_class
G_DEFINE_TYPE (GstNvDsOsd, gst_nvds_osd, GST_TYPE_BASE_TRANSFORM);

#define GST_TYPE_NV_OSD_PROCESS_MODE (gst_nvds_osd_process_mode_get_type ())

static GQuark _dsmeta_quark;

static GType
gst_nvds_osd_process_mode_get_type (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {MODE_CPU, "CPU_MODE", "MODE_CPU"},
      {MODE_GPU, "GPU_MODE", "MODE_GPU"},
#ifdef PLATFORM_TEGRA
      {MODE_NONE,
            "Invalid mode. Falls back to GPU",
          "MODE_NONE"},
#endif
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstNvDsOsdMode", values);
  }
  return qtype;
}

static void gst_nvds_osd_finalize (GObject * object);
static void gst_nvds_osd_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvds_osd_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstFlowReturn gst_nvds_osd_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static gboolean gst_nvds_osd_start (GstBaseTransform * btrans);
static gboolean gst_nvds_osd_stop (GstBaseTransform * btrans);
static gboolean gst_nvds_osd_parse_color (GstNvDsOsd * nvdsosd,
    guint clock_color);
static gboolean gst_nvdsosd_sink_event (GstBaseTransform * trans,
    GstEvent * event);

static gboolean
gst_nvdsosd_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (trans);

  if ((GstNvDsCustomEventType) GST_EVENT_TYPE (event) ==
      GST_NVEVENT_OSD_PROCESS_MODE_UPDATE) {
    gchar *stream_id = NULL;
    guint process_mode = 0;

    gst_nvevent_parse_osd_process_mode_update (event, &stream_id,
        &process_mode);

    nvdsosd->nvdsosd_mode = process_mode == 0 ? MODE_CPU : MODE_GPU;

    int flag_integrated = -1;
    cudaDeviceGetAttribute (&flag_integrated, cudaDevAttrIntegrated,
        nvdsosd->gpu_id);
    if (!flag_integrated && nvdsosd->nvdsosd_mode > MODE_GPU) {
      g_print ("WARN !! Invalid mode selected, Falling back to GPU\n");
      nvdsosd->nvdsosd_mode = MODE_GPU;
    }

  }

  /* Call the sink event handler of the base class. */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

static GstCaps *
gst_nvds_osd_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (trans);
  GstCaps *ret;
  GstCaps *new_caps;
  GstCaps *caps_rgba =
      gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)RGBA");

  GST_DEBUG_OBJECT (trans, "identity from: %" GST_PTR_FORMAT, caps);
  if (filter) {
    ret = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
  } else {
    ret = gst_caps_ref (caps);
  }

  /* Force to RGBA format for CPU mode. */
  if (nvdsosd->nvdsosd_mode == MODE_CPU) {
    new_caps = gst_caps_intersect_full (ret, caps_rgba, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (ret);
    ret = new_caps;
  }
  gst_caps_unref (caps_rgba);

  return ret;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvds_osd_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  gboolean ret = TRUE;

  GstNvDsOsd *nvdsosd = GST_NVDSOSD (trans);
  gint width = 0, height = 0;
  cudaError_t CUerr = cudaSuccess;

  nvdsosd->frame_num = 0;

  GstStructure *structure = gst_caps_get_structure (incaps, 0);

  GST_OBJECT_LOCK (nvdsosd);
  if (!gst_structure_get_int (structure, "width", &width) ||
      !gst_structure_get_int (structure, "height", &height)) {
    GST_ELEMENT_ERROR (nvdsosd, STREAM, FAILED,
        ("caps without width/height"), NULL);
    ret = FALSE;
    goto exit_set_caps;
  }
  if (nvdsosd->nvdsosd_context && nvdsosd->width == width
      && nvdsosd->height == height) {
    goto exit_set_caps;
  }

  CUerr = cudaSetDevice (nvdsosd->gpu_id);
  if (CUerr != cudaSuccess) {
    ret = FALSE;
    GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
        ("Unable to set device"), NULL);
    goto exit_set_caps;
  }

  nvdsosd->width = width;
  nvdsosd->height = height;

  if (nvdsosd->show_clock)
    nvll_osd_set_clock_params (nvdsosd->nvdsosd_context,
        &nvdsosd->clock_text_params);

  nvdsosd->conv_buf =
      nvll_osd_set_params (nvdsosd->nvdsosd_context, nvdsosd->width,
      nvdsosd->height);

exit_set_caps:
  GST_OBJECT_UNLOCK (nvdsosd);
  return ret;
}

/**
 * Initialize all resources.
 */
static gboolean
gst_nvds_osd_start (GstBaseTransform * btrans)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (btrans);

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice (nvdsosd->gpu_id);
  if (CUerr != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
        ("Unable to set device"), NULL);
    return FALSE;
  }
  GST_LOG_OBJECT (nvdsosd, "SETTING CUDA DEVICE = %d in nvdsosd func=%s\n",
      nvdsosd->gpu_id, __func__);

  nvdsosd->nvdsosd_context = nvll_osd_create_context ();

  if (nvdsosd->nvdsosd_context == NULL) {
    GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
        ("Unable to create context nvdsosd"), NULL);
    return FALSE;
  }

  int flag_integrated = -1;
  cudaDeviceGetAttribute (&flag_integrated, cudaDevAttrIntegrated,
      nvdsosd->gpu_id);
  if (!flag_integrated && nvdsosd->nvdsosd_mode > MODE_GPU) {
    g_print ("WARN !! Invalid mode selected, Falling back to GPU\n");
    nvdsosd->nvdsosd_mode = MODE_GPU;
  }

  if (nvdsosd->show_clock) {
    nvll_osd_set_clock_params (nvdsosd->nvdsosd_context,
        &nvdsosd->clock_text_params);
  }

  return TRUE;
}

/**
 * Free up all the resources
 */
static gboolean
gst_nvds_osd_stop (GstBaseTransform * btrans)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (btrans);

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice (nvdsosd->gpu_id);
  if (CUerr != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
        ("Unable to set device"), NULL);
    return FALSE;
  }
  GST_LOG_OBJECT (nvdsosd, "SETTING CUDA DEVICE = %d in nvdsosd func=%s\n",
      nvdsosd->gpu_id, __func__);

  if (nvdsosd->nvdsosd_context)
    nvll_osd_destroy_context (nvdsosd->nvdsosd_context);

  nvdsosd->nvdsosd_context = NULL;
  nvdsosd->width = 0;
  nvdsosd->height = 0;

  return TRUE;
}

int frame_num = 0;

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvds_osd_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (trans);
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  unsigned int rect_cnt = 0;
  unsigned int segment_cnt = 0;
  unsigned int text_cnt = 0;
  unsigned int line_cnt = 0;
  unsigned int arrow_cnt = 0;
  unsigned int circle_cnt = 0;
  unsigned int blur_cnt = 0;
  unsigned int i = 0;

  gpointer state = NULL;
  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;

  if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
        ("Unable to map info from buffer"), NULL);
    return GST_FLOW_ERROR;
  }

  nvds_set_input_system_timestamp (buf, GST_ELEMENT_NAME (nvdsosd));

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice (nvdsosd->gpu_id);
  if (CUerr != cudaSuccess) {
    GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
        ("Unable to set device"), NULL);
    return GST_FLOW_ERROR;
  }
  GST_LOG_OBJECT (nvdsosd, "SETTING CUDA DEVICE = %d in nvdsosd func=%s\n",
      nvdsosd->gpu_id, __func__);

  surface = (NvBufSurface *) inmap.data;

  /* Get metadata. Update rectangle and text params */
  GstMeta *gst_meta;
  NvDsMeta *dsmeta;
  char context_name[100];
  snprintf (context_name, sizeof (context_name), "%s_(Frame=%u)",
      GST_ELEMENT_NAME (nvdsosd), nvdsosd->frame_num);
  nvtxRangePushA (context_name);
  while ((gst_meta = gst_buffer_iterate_meta (buf, &state))) {
    if (gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      dsmeta = (NvDsMeta *) gst_meta;
      if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
        break;
      }
    }
  }

  NvDsMetaList *l = NULL;
  NvDsMetaList *full_obj_meta_list = NULL;
  if (batch_meta)
    full_obj_meta_list = batch_meta->obj_meta_pool->full_list;
  NvDsObjectMeta *object_meta = NULL;

  for (l = full_obj_meta_list; l != NULL; l = l->next) {
    object_meta = (NvDsObjectMeta *) (l->data);
    if (nvdsosd->draw_bbox) {
      nvdsosd->rect_params[rect_cnt] = object_meta->rect_params;
      rect_cnt++;
    }
    if (rect_cnt == MAX_OSD_ELEMS) {
      nvdsosd->frame_rect_params->num_rects = rect_cnt;
      nvdsosd->frame_rect_params->rect_params_list = nvdsosd->rect_params;
      /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_rect_params->surf' instead */
      nvdsosd->frame_rect_params->buf_ptr = NULL;
      nvdsosd->frame_rect_params->mode = nvdsosd->nvdsosd_mode;
      nvdsosd->frame_rect_params->surf = surface;
      if (nvll_osd_draw_rectangles (nvdsosd->nvdsosd_context,
              nvdsosd->frame_rect_params) == -1) {
        GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
            ("Unable to draw rectangles"), NULL);
        return GST_FLOW_ERROR;
      }
      rect_cnt = 0;
    }

    if (nvdsosd->blur_bbox) {
      if(nvdsosd->blur_gie_class_list != NULL) {
        for(GList *it = nvdsosd->blur_gie_class_list; it; it = it->next) {
          gint gie_id = g_array_index((GArray *)(it->data), gint, 0);
          gint class_id = g_array_index((GArray *)(it->data), gint, 1);
          GST_LOG_OBJECT (nvdsosd,"blur gie %d class %d \n", gie_id, class_id);
          if(object_meta->unique_component_id == gie_id && object_meta->class_id == class_id) {
            nvdsosd->blur_rect_params[blur_cnt] = object_meta->rect_params;
            nvdsosd->mask_params[blur_cnt] = object_meta->mask_params;
            blur_cnt++;
          }
        }
      } else {
        nvdsosd->blur_rect_params[blur_cnt] = object_meta->rect_params;
        blur_cnt++;
      }
    }

    if (blur_cnt == MAX_OSD_ELEMS) {
      nvdsosd->frame_blur_params->num_rects = blur_cnt;
      nvdsosd->frame_blur_params->rect_params_list = nvdsosd->blur_rect_params;
      nvdsosd->frame_blur_params->mask_params_list = nvdsosd->mask_params;
      /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_rect_params->surf' instead */
      nvdsosd->frame_blur_params->buf_ptr = NULL;
      nvdsosd->frame_blur_params->mode = nvdsosd->nvdsosd_mode;
      nvdsosd->frame_blur_params->surf = surface;
      if (nvll_osd_blur_rectangles(nvdsosd->nvdsosd_context,
              nvdsosd->frame_blur_params) == -1) {
        GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
            ("Unable to blur image"), NULL);
        return GST_FLOW_ERROR;
      }
      blur_cnt = 0;
    }

    if (nvdsosd->draw_mask && object_meta->mask_params.data &&
        object_meta->mask_params.size > 0) {
      nvdsosd->mask_rect_params[segment_cnt] = object_meta->rect_params;
      nvdsosd->mask_params[segment_cnt++] = object_meta->mask_params;
      if (segment_cnt == MAX_OSD_ELEMS) {
        nvdsosd->frame_mask_params->num_segments = segment_cnt;
        nvdsosd->frame_mask_params->rect_params_list =
            nvdsosd->mask_rect_params;
        nvdsosd->frame_mask_params->mask_params_list = nvdsosd->mask_params;
        /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_mask_params->surf' instead */
        nvdsosd->frame_mask_params->buf_ptr = NULL;
        nvdsosd->frame_mask_params->mode = nvdsosd->nvdsosd_mode;
        nvdsosd->frame_mask_params->surf = surface;
        if (nvll_osd_draw_segment_masks (nvdsosd->nvdsosd_context,
                nvdsosd->frame_mask_params) == -1) {
          GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
              ("Unable to draw rectangles"), NULL);
          return GST_FLOW_ERROR;
        }
        segment_cnt = 0;
      }
    }
    if (object_meta->text_params.display_text)
      nvdsosd->text_params[text_cnt++] = object_meta->text_params;
    if (text_cnt == MAX_OSD_ELEMS) {
      nvdsosd->frame_text_params->num_strings = text_cnt;
      nvdsosd->frame_text_params->text_params_list = nvdsosd->text_params;
      /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_text_params->surf' instead */
      nvdsosd->frame_text_params->buf_ptr = NULL;
      nvdsosd->frame_text_params->mode = nvdsosd->nvdsosd_mode;
      nvdsosd->frame_rect_params->surf = surface;
      if (nvll_osd_put_text (nvdsosd->nvdsosd_context,
              nvdsosd->frame_text_params) == -1) {
        GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
            ("Unable to draw text"), NULL);
        return GST_FLOW_ERROR;
      }
      text_cnt = 0;
    }
  }

  NvDsMetaList *display_meta_list = NULL;
  if (batch_meta)
    display_meta_list = batch_meta->display_meta_pool->full_list;
  NvDsDisplayMeta *display_meta = NULL;

  /* Get objects to be drawn from display meta.
   * Draw objects if count equals MAX_OSD_ELEMS.
   */
  for (l = display_meta_list; l != NULL; l = l->next) {
    display_meta = (NvDsDisplayMeta *) (l->data);

    unsigned int cnt = 0;
    for (cnt = 0; cnt < display_meta->num_rects; cnt++) {
      nvdsosd->rect_params[rect_cnt++] = display_meta->rect_params[cnt];
      if (rect_cnt == MAX_OSD_ELEMS) {
        nvdsosd->frame_rect_params->num_rects = rect_cnt;
        nvdsosd->frame_rect_params->rect_params_list = nvdsosd->rect_params;
        /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_rect_params->surf' instead */
        nvdsosd->frame_rect_params->buf_ptr = NULL;
        nvdsosd->frame_rect_params->mode = nvdsosd->nvdsosd_mode;
        nvdsosd->frame_rect_params->surf = surface;
        if (nvll_osd_draw_rectangles (nvdsosd->nvdsosd_context,
                nvdsosd->frame_rect_params) == -1) {
          GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
              ("Unable to draw rectangles"), NULL);
          return GST_FLOW_ERROR;
        }
        rect_cnt = 0;
      }
    }

    for (cnt = 0; cnt < display_meta->num_labels; cnt++) {
      if (display_meta->text_params[cnt].display_text) {
        nvdsosd->text_params[text_cnt++] = display_meta->text_params[cnt];
        if (text_cnt == MAX_OSD_ELEMS) {
          nvdsosd->frame_text_params->num_strings = text_cnt;
          nvdsosd->frame_text_params->text_params_list = nvdsosd->text_params;
          /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_text_params->surf' instead */
          nvdsosd->frame_text_params->buf_ptr = NULL;
          nvdsosd->frame_text_params->mode = nvdsosd->nvdsosd_mode;
          nvdsosd->frame_text_params->surf = surface;
          if (nvll_osd_put_text (nvdsosd->nvdsosd_context,
                  nvdsosd->frame_text_params) == -1) {
            GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
                ("Unable to draw text"), NULL);
            return GST_FLOW_ERROR;
          }
          text_cnt = 0;
        }
      }
    }

    for (cnt = 0; cnt < display_meta->num_lines; cnt++) {
      nvdsosd->line_params[line_cnt++] = display_meta->line_params[cnt];
      if (line_cnt == MAX_OSD_ELEMS) {
        nvdsosd->frame_line_params->num_lines = line_cnt;
        nvdsosd->frame_line_params->line_params_list = nvdsosd->line_params;
        /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_line_params->surf' instead */
        nvdsosd->frame_line_params->buf_ptr = NULL;
        nvdsosd->frame_line_params->mode = nvdsosd->nvdsosd_mode;
        nvdsosd->frame_line_params->surf = surface;
        if (nvll_osd_draw_lines (nvdsosd->nvdsosd_context,
                nvdsosd->frame_line_params) == -1) {
          GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
              ("Unable to draw lines"), NULL);
          return GST_FLOW_ERROR;
        }
        line_cnt = 0;
      }
    }

    for (cnt = 0; cnt < display_meta->num_arrows; cnt++) {
      nvdsosd->arrow_params[arrow_cnt++] = display_meta->arrow_params[cnt];
      if (arrow_cnt == MAX_OSD_ELEMS) {
        nvdsosd->frame_arrow_params->num_arrows = arrow_cnt;
        nvdsosd->frame_arrow_params->arrow_params_list = nvdsosd->arrow_params;
        /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_arrow_params->surf' instead */
        nvdsosd->frame_arrow_params->buf_ptr = NULL;
        nvdsosd->frame_arrow_params->mode = nvdsosd->nvdsosd_mode;
        nvdsosd->frame_arrow_params->surf = surface;
        if (nvll_osd_draw_arrows (nvdsosd->nvdsosd_context,
                nvdsosd->frame_arrow_params) == -1) {
          GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
              ("Unable to draw arrows"), NULL);
          return GST_FLOW_ERROR;
        }
        arrow_cnt = 0;
      }
    }

    for (cnt = 0; cnt < display_meta->num_circles; cnt++) {
      nvdsosd->circle_params[circle_cnt++] = display_meta->circle_params[cnt];
      if (circle_cnt == MAX_OSD_ELEMS) {
        nvdsosd->frame_circle_params->num_circles = circle_cnt;
        nvdsosd->frame_circle_params->circle_params_list =
            nvdsosd->circle_params;
        /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_circle_params->surf' instead */
        nvdsosd->frame_circle_params->buf_ptr = NULL;
        nvdsosd->frame_circle_params->mode = nvdsosd->nvdsosd_mode;
        nvdsosd->frame_circle_params->surf = surface;
        if (nvll_osd_draw_circles (nvdsosd->nvdsosd_context,
                nvdsosd->frame_circle_params) == -1) {
          GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
              ("Unable to draw circles"), NULL);
          return GST_FLOW_ERROR;
        }
        circle_cnt = 0;
      }
    }
    i++;
  }

  nvdsosd->num_rect = rect_cnt;
  nvdsosd->num_segments = segment_cnt;
  nvdsosd->num_strings = text_cnt;
  nvdsosd->num_lines = line_cnt;
  nvdsosd->num_arrows = arrow_cnt;
  nvdsosd->num_circles = circle_cnt;
  nvdsosd->num_blurs = blur_cnt;
  if (rect_cnt != 0 && nvdsosd->draw_bbox) {
    nvdsosd->frame_rect_params->num_rects = nvdsosd->num_rect;
    nvdsosd->frame_rect_params->rect_params_list = nvdsosd->rect_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_rect_params->surf' instead */
    nvdsosd->frame_rect_params->buf_ptr = NULL;
    nvdsosd->frame_rect_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_rect_params->surf = surface;
    if (nvll_osd_draw_rectangles (nvdsosd->nvdsosd_context,
            nvdsosd->frame_rect_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
          ("Unable to draw rectangles"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (segment_cnt != 0 && nvdsosd->draw_mask) {
    nvdsosd->frame_mask_params->num_segments = nvdsosd->num_segments;
    nvdsosd->frame_mask_params->rect_params_list = nvdsosd->mask_rect_params;
    nvdsosd->frame_mask_params->mask_params_list = nvdsosd->mask_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_mask_params->surf' instead */
    nvdsosd->frame_mask_params->buf_ptr = NULL;
    nvdsosd->frame_mask_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_mask_params->surf = surface;
    if (nvll_osd_draw_segment_masks (nvdsosd->nvdsosd_context,
            nvdsosd->frame_mask_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
          ("Unable to draw segment masks"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if ((nvdsosd->show_clock || text_cnt) && nvdsosd->draw_text) {
    nvdsosd->frame_text_params->num_strings = nvdsosd->num_strings;
    nvdsosd->frame_text_params->text_params_list = nvdsosd->text_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_text_params->surf' instead */
    nvdsosd->frame_text_params->buf_ptr = NULL;
    nvdsosd->frame_text_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_text_params->surf = surface;
    if (nvll_osd_put_text (nvdsosd->nvdsosd_context,
            nvdsosd->frame_text_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED, ("Unable to draw text"),
          NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (line_cnt != 0) {
    nvdsosd->frame_line_params->num_lines = nvdsosd->num_lines;
    nvdsosd->frame_line_params->line_params_list = nvdsosd->line_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_line_params->surf' instead */
    nvdsosd->frame_line_params->buf_ptr = NULL;
    nvdsosd->frame_line_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_line_params->surf = surface;
    if (nvll_osd_draw_lines (nvdsosd->nvdsosd_context,
            nvdsosd->frame_line_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED, ("Unable to draw lines"),
          NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (arrow_cnt != 0) {
    nvdsosd->frame_arrow_params->num_arrows = nvdsosd->num_arrows;
    nvdsosd->frame_arrow_params->arrow_params_list = nvdsosd->arrow_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_arrow_params->surf' instead */
    nvdsosd->frame_arrow_params->buf_ptr = NULL;
    nvdsosd->frame_arrow_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_arrow_params->surf = surface;
    if (nvll_osd_draw_arrows (nvdsosd->nvdsosd_context,
            nvdsosd->frame_arrow_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
          ("Unable to draw arrows"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (circle_cnt != 0) {
    nvdsosd->frame_circle_params->num_circles = nvdsosd->num_circles;
    nvdsosd->frame_circle_params->circle_params_list = nvdsosd->circle_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_circle_params->surf' instead */
    nvdsosd->frame_circle_params->buf_ptr = NULL;
    nvdsosd->frame_circle_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_circle_params->surf = surface;
    if (nvll_osd_draw_circles (nvdsosd->nvdsosd_context,
            nvdsosd->frame_circle_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
          ("Unable to draw circles"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (blur_cnt != 0 && nvdsosd->blur_bbox) {
    nvdsosd->frame_blur_params->num_rects = blur_cnt;
    nvdsosd->frame_blur_params->rect_params_list = nvdsosd->blur_rect_params;
    if(nvdsosd->blur_bbox == Blur_Mask)
      nvdsosd->frame_blur_params->mask_params_list = nvdsosd->mask_params;
    /** Use of buf_ptr is deprecated, use 'nvdsosd->frame_rect_params->surf' instead */
    nvdsosd->frame_blur_params->buf_ptr = NULL;
    nvdsosd->frame_blur_params->mode = nvdsosd->nvdsosd_mode;
    nvdsosd->frame_blur_params->surf = surface;
    if (nvll_osd_blur_rectangles(nvdsosd->nvdsosd_context,
            nvdsosd->frame_blur_params) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
          ("Unable to blur image"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (nvdsosd->nvdsosd_mode == MODE_GPU) {
    if (nvll_osd_apply (nvdsosd->nvdsosd_context, NULL, surface) == -1) {
      GST_ELEMENT_ERROR (nvdsosd, RESOURCE, FAILED,
          ("Unable to draw shapes onto video frame by GPU"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  nvtxRangePop ();
  nvdsosd->frame_num++;

  nvds_set_output_system_timestamp (buf, GST_ELEMENT_NAME (nvdsosd));

  gst_buffer_unmap (buf, &inmap);
  return GST_FLOW_OK;
}

/* Called when the plugin is destroyed.
 * Free all structures which have been malloc'd.
 */
static void
gst_nvds_osd_finalize (GObject * object)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (object);

  if (nvdsosd->clock_text_params.font_params.font_name) {
    g_free ((char *) nvdsosd->clock_text_params.font_params.font_name);
  }
  if (nvdsosd->blur_gie_class_ids) {
    g_free ((char *) nvdsosd->blur_gie_class_ids);
  }
  if(nvdsosd->blur_gie_class_list) {
    g_list_free(nvdsosd->blur_gie_class_list);
  }
  g_free (nvdsosd->rect_params);
  g_free (nvdsosd->blur_rect_params);
  g_free (nvdsosd->mask_rect_params);
  g_free (nvdsosd->mask_params);
  g_free (nvdsosd->text_params);
  g_free (nvdsosd->line_params);
  g_free (nvdsosd->arrow_params);
  g_free (nvdsosd->circle_params);

  g_free (nvdsosd->frame_rect_params);
  g_free (nvdsosd->frame_mask_params);
  g_free (nvdsosd->frame_text_params);
  g_free (nvdsosd->frame_line_params);
  g_free (nvdsosd->frame_arrow_params);
  g_free (nvdsosd->frame_circle_params);
  g_free (nvdsosd->frame_blur_params);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvds_osd_class_init (GstNvDsOsdClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *base_transform_class =
      GST_BASE_TRANSFORM_CLASS (klass);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  base_transform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvds_osd_transform_ip);
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_nvds_osd_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_nvds_osd_stop);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvds_osd_set_caps);
  base_transform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvds_osd_transform_caps);
  base_transform_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvdsosd_sink_event);

  gobject_class->set_property = gst_nvds_osd_set_property;
  gobject_class->get_property = gst_nvds_osd_get_property;
  gobject_class->finalize = gst_nvds_osd_finalize;

  base_transform_class->passthrough_on_same_caps = TRUE;

  g_object_class_install_property (gobject_class, PROP_SHOW_CLOCK,
      g_param_spec_boolean ("display-clock", "clock",
          "Whether to display clock", FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SHOW_TEXT,
      g_param_spec_boolean ("display-text", "text", "Whether to display text",
          TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SHOW_BBOX,
      g_param_spec_boolean ("display-bbox", "text",
          "Whether to display bounding boxes", TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SHOW_MASK,
      g_param_spec_boolean ("display-mask", "text",
          "Whether to display instance mask", TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_CLOCK_FONT,
      g_param_spec_string ("clock-font", "clock-font",
          "Clock Font to be set",
          "DEFAULT_FONT",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CLOCK_FONT_SIZE,
      g_param_spec_uint ("clock-font-size", "clock-font-size",
          "font size of the clock",
          0, MAX_FONT_SIZE, DEFAULT_FONT_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_CLOCK_X_OFFSET,
      g_param_spec_uint ("x-clock-offset", "x-clock-offset",
          "x-clock-offset",
          0, G_MAXUINT, 0,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_CLOCK_Y_OFFSET,
      g_param_spec_uint ("y-clock-offset", "y-clock-offset",
          "y-clock-offset",
          0, G_MAXUINT, 0,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_CLOCK_COLOR,
      g_param_spec_uint ("clock-color", "clock-color",
          "clock-color",
          0, G_MAXUINT, G_MAXUINT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_PROCESS_MODE,
      g_param_spec_enum ("process-mode", "Process Mode",
          "Rect and text draw process mode, CPU_MODE only support RGBA format",
          GST_TYPE_NV_OSD_PROCESS_MODE,
          GST_NV_OSD_DEFAULT_PROCESS_MODE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, 0,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_BLUR_GIE_CLASS_IDS,
      g_param_spec_string ("blur-on-gie-class-ids", "Blur on GIE Class Ids",
          "Blur the bbox generated by GIE with the unique ID and class ID",
          NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BLUR_BBOX,
      g_param_spec_uint ("blur-bbox", "Blur Bbox",
          "Blur mode, (0):Disable, (1)blur the entire bbox area of object,\n"
          "\t\t\t(2)Blur the mask area of object if there is a mask, swith to Blur_Bbox if no mask.",
          0, 2, 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  gst_element_class_set_details_simple (gstelement_class,
      "NvDsOsd plugin",
      "NvDsOsd functionality",
      "Gstreamer bounding box draw element",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&nvdsosd_src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&nvdsosd_sink_factory));

  _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvds_osd_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (object);

  switch (prop_id) {
    case PROP_SHOW_CLOCK:
      nvdsosd->show_clock = g_value_get_boolean (value);
      break;
    case PROP_SHOW_TEXT:
      nvdsosd->draw_text = g_value_get_boolean (value);
      break;
    case PROP_SHOW_BBOX:
      nvdsosd->draw_bbox = g_value_get_boolean (value);
      break;
    case PROP_SHOW_MASK:
      nvdsosd->draw_mask = g_value_get_boolean (value);
      break;
    case PROP_CLOCK_FONT:
      if (nvdsosd->clock_text_params.font_params.font_name) {
        g_free ((char *) nvdsosd->clock_text_params.font_params.font_name);
      }
      nvdsosd->clock_text_params.font_params.font_name =
          (gchar *) g_value_dup_string (value);
      break;
    case PROP_CLOCK_FONT_SIZE:
      nvdsosd->clock_text_params.font_params.font_size =
          g_value_get_uint (value);
      break;
    case PROP_CLOCK_X_OFFSET:
      nvdsosd->clock_text_params.x_offset = g_value_get_uint (value);
      break;
    case PROP_CLOCK_Y_OFFSET:
      nvdsosd->clock_text_params.y_offset = g_value_get_uint (value);
      break;
    case PROP_CLOCK_COLOR:
      gst_nvds_osd_parse_color (nvdsosd, g_value_get_uint (value));
      break;
    case PROP_PROCESS_MODE:
      nvdsosd->nvdsosd_mode = (NvOSD_Mode) g_value_get_enum (value);
      if (nvdsosd->nvdsosd_mode > MODE_GPU) {
        g_print ("WARN !! Invalid mode selected, Falling back to GPU\n");
        nvdsosd->nvdsosd_mode =
            nvdsosd->nvdsosd_mode > MODE_GPU ? MODE_GPU : nvdsosd->nvdsosd_mode;
      }
      break;
    case PROP_GPU_DEVICE_ID:
      nvdsosd->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BLUR_GIE_CLASS_IDS:
    {
      if (nvdsosd->blur_gie_class_ids) {
        g_free ((char *) nvdsosd->blur_gie_class_ids);
      }
      nvdsosd->blur_gie_class_ids = (gchar *) g_value_dup_string (value);
      gboolean is_digit_flag = TRUE;
      gchar **split_ids = NULL;
      gchar **split_gie_class = NULL;
      gint i = 0, j = 0;
      split_ids = g_strsplit(nvdsosd->blur_gie_class_ids, ";", -1);
      for(i = 0; split_ids[i] != NULL && is_digit_flag; i++) {
        GArray *array = g_array_new (FALSE, FALSE, sizeof(gint));
        if(split_gie_class != NULL) {
          g_strfreev(split_gie_class);
        }
        split_gie_class = g_strsplit(split_ids[i], ",", 2);
        for(j = 0; split_gie_class[j] != NULL; j++) {
          gint tmp = -1;
          gchar *endptr = NULL;
          tmp = g_ascii_strtoll (split_gie_class[j], &endptr, 10);
          if (tmp == 0 && endptr == split_gie_class[j]) {
            g_print ("Please check whether the blur parameter are correct. We'll blur all the object!\n");
            if(nvdsosd->blur_gie_class_list) {
              g_list_free(nvdsosd->blur_gie_class_list);
            }
            nvdsosd->blur_gie_class_list = NULL;
            is_digit_flag = FALSE;
            break;
          }
          g_array_append_val(array, tmp);
        }
        if(is_digit_flag) {
          nvdsosd->blur_gie_class_list = g_list_prepend(nvdsosd->blur_gie_class_list, array);
        }
      }
      if(split_gie_class != NULL) {
        g_strfreev(split_gie_class);
        split_gie_class = NULL;
      }
      if(split_ids != NULL) {
        g_strfreev(split_ids);
        split_ids = NULL;
      }
      break;
    }
    case PROP_BLUR_BBOX:
      nvdsosd->blur_bbox = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvds_osd_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsOsd *nvdsosd = GST_NVDSOSD (object);

  switch (prop_id) {
    case PROP_SHOW_CLOCK:
      g_value_set_boolean (value, nvdsosd->show_clock);
      break;
    case PROP_SHOW_TEXT:
      g_value_set_boolean (value, nvdsosd->draw_text);
      break;
    case PROP_SHOW_BBOX:
      g_value_set_boolean (value, nvdsosd->draw_bbox);
      break;
    case PROP_SHOW_MASK:
      g_value_set_boolean (value, nvdsosd->draw_mask);
      break;
    case PROP_CLOCK_FONT:
      g_value_set_string (value, nvdsosd->font);
      break;
    case PROP_CLOCK_FONT_SIZE:
      g_value_set_uint (value, nvdsosd->clock_font_size);
      break;
    case PROP_CLOCK_X_OFFSET:
      g_value_set_uint (value, nvdsosd->clock_text_params.x_offset);
      break;
    case PROP_CLOCK_Y_OFFSET:
      g_value_set_uint (value, nvdsosd->clock_text_params.y_offset);
      break;
    case PROP_CLOCK_COLOR:
      g_value_set_uint (value, nvdsosd->clock_color);
      break;
    case PROP_PROCESS_MODE:
      g_value_set_enum (value, nvdsosd->nvdsosd_mode);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvdsosd->gpu_id);
      break;
    case PROP_BLUR_GIE_CLASS_IDS:
      g_value_set_string (value, nvdsosd->blur_gie_class_ids);
      break;
    case PROP_BLUR_BBOX:
      g_value_set_uint (value, nvdsosd->blur_bbox);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Set default values of certain properties.
 */
static void
gst_nvds_osd_init (GstNvDsOsd * nvdsosd)
{
  nvdsosd->show_clock = FALSE;
  nvdsosd->draw_text = TRUE;
  nvdsosd->draw_bbox = TRUE;
  nvdsosd->draw_mask = FALSE;
  nvdsosd->clock_text_params.font_params.font_name = g_strdup (DEFAULT_FONT);
  nvdsosd->clock_text_params.font_params.font_size = DEFAULT_FONT_SIZE;
  nvdsosd->nvdsosd_mode = GST_NV_OSD_DEFAULT_PROCESS_MODE;
  nvdsosd->border_width = DEFAULT_BORDER_WIDTH;
  nvdsosd->num_rect = 0;
  nvdsosd->num_segments = 0;
  nvdsosd->num_strings = 0;
  nvdsosd->num_lines = 0;
  nvdsosd->clock_text_params.font_params.font_color.red = 1.0;
  nvdsosd->clock_text_params.font_params.font_color.green = 0.0;
  nvdsosd->clock_text_params.font_params.font_color.blue = 0.0;
  nvdsosd->clock_text_params.font_params.font_color.alpha = 1.0;
  nvdsosd->rect_params = g_new0 (NvOSD_RectParams, MAX_OSD_ELEMS);
  nvdsosd->blur_rect_params = g_new0 (NvOSD_RectParams, MAX_OSD_ELEMS);
  nvdsosd->mask_rect_params = g_new0 (NvOSD_RectParams, MAX_OSD_ELEMS);
  nvdsosd->mask_params = g_new0 (NvOSD_MaskParams, MAX_OSD_ELEMS);
  nvdsosd->text_params = g_new0 (NvOSD_TextParams, MAX_OSD_ELEMS);
  nvdsosd->line_params = g_new0 (NvOSD_LineParams, MAX_OSD_ELEMS);
  nvdsosd->arrow_params = g_new0 (NvOSD_ArrowParams, MAX_OSD_ELEMS);
  nvdsosd->circle_params = g_new0 (NvOSD_CircleParams, MAX_OSD_ELEMS);
  nvdsosd->frame_rect_params = g_new0 (NvOSD_FrameRectParams, MAX_OSD_ELEMS);
  nvdsosd->frame_mask_params =
      g_new0 (NvOSD_FrameSegmentMaskParams, MAX_OSD_ELEMS);
  nvdsosd->frame_text_params = g_new0 (NvOSD_FrameTextParams, MAX_OSD_ELEMS);
  nvdsosd->frame_line_params = g_new0 (NvOSD_FrameLineParams, MAX_OSD_ELEMS);
  nvdsosd->frame_arrow_params = g_new0 (NvOSD_FrameArrowParams, MAX_OSD_ELEMS);
  nvdsosd->frame_circle_params =
      g_new0 (NvOSD_FrameCircleParams, MAX_OSD_ELEMS);
  nvdsosd->frame_blur_params = g_new0 (NvOSD_FrameRectParams, MAX_OSD_ELEMS);
  nvdsosd->blur_bbox = Blur_Disable;
  nvdsosd->blur_gie_class_ids = NULL;
  nvdsosd->blur_gie_class_list = NULL;
}

/**
 * Set color of text for clock, if enabled.
 */
static gboolean
gst_nvds_osd_parse_color (GstNvDsOsd * nvdsosd, guint clock_color)
{
  nvdsosd->clock_text_params.font_params.font_color.red =
      (gfloat) ((clock_color & 0xff000000) >> 24) / 255;
  nvdsosd->clock_text_params.font_params.font_color.green =
      (gfloat) ((clock_color & 0x00ff0000) >> 16) / 255;
  nvdsosd->clock_text_params.font_params.font_color.blue =
      (gfloat) ((clock_color & 0x0000ff00) >> 8) / 255;
  nvdsosd->clock_text_params.font_params.font_color.alpha =
      (gfloat) ((clock_color & 0x000000ff)) / 255;
  return TRUE;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdsosd_init (GstPlugin * nvdsosd)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvds_osd_debug, "nvdsosd", 0, "nvdsosd plugin");

  return gst_element_register (nvdsosd, "nvdsosd", GST_RANK_PRIMARY,
      GST_TYPE_NVDSOSD);
}

#ifndef PACKAGE
#define PACKAGE "nvdsosd"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_osd,
    PACKAGE_DESCRIPTION,
    nvdsosd_init, "9.0", PACKAGE_LICENSE, PACKAGE_NAME, PACKAGE_URL)
