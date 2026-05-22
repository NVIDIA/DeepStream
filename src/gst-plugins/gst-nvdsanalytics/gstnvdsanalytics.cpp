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

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <cstring>
#include <map>
#include <cmath>
#include "gstnvdsanalytics.h"
#include "nvdsanalytics_property_parser.h"
#include "nvdsanalytics_property_yaml_parser.h"
#include <sys/time.h>
#include "gst-nvdscustomevent.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvdsanalytics_debug);
#define GST_CAT_DEFAULT gst_nvdsanalytics_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_ENABLE,
  PROP_CONFIG_FILE
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 17
#define DEFAULT_WIDTH 1920
#define DEFAULT_HEIGHT 1080
#define DEFAULT_FONT_SIZE 12
#define DEFAULT_OSD_MODE 2


typedef void DsExampleOutput;

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvdsanalytics_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvdsanalytics_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvdsanalytics_parent_class parent_class
G_DEFINE_TYPE (GstNvDsAnalytics, gst_nvdsanalytics, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdsanalytics_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsanalytics_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvdsanalytics_sink_event (GstBaseTransform * trans, GstEvent * event);

static gboolean gst_nvdsanalytics_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvdsanalytics_start (GstBaseTransform * btrans);
static gboolean gst_nvdsanalytics_stop (GstBaseTransform * btrans);
static void gst_nvdsanalytics_finalize (GObject * object);

static GstFlowReturn gst_nvdsanalytics_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);

static void attach_metadata_object (GstNvDsAnalytics * nvdsanalytics,
    NvDsObjectMeta * obj_meta, ObjInf & obj_info);

static void
attach_framemeta_analytics_metadata (GstNvDsAnalytics * self,
    NvDsFrameMeta * frame_meta, NvDsAnalyticProcessParams & process_params,
    gint stream_id);
static gpointer copy_frame_nvdsanalytics_meta (gpointer data,
    gpointer user_data);
static void release_frame_nvdsanalytics_meta (gpointer data,
    gpointer user_data);

static gpointer copy_obj_nvdsanalytics_meta (gpointer data, gpointer user_data);
static void release_obj_nvdsanalytics_meta (gpointer data, gpointer user_data);
/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvdsanalytics_class_init (GstNvDsAnalyticsClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_nvdsanalytics_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_nvdsanalytics_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvdsanalytics_finalize);

  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsanalytics_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdsanalytics_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsanalytics_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvdsanalytics_transform_ip);

  gstbasetransform_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvdsanalytics_sink_event);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Enable DsAnalytics plugin, or set in passthrough mode",
          TRUE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string ("config-file", "DsAnalytics Config File",
          "DsAnalytics Config File",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdsanalytics_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvdsanalytics_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsAnalytics plugin",
      "DsAnalytics Plugin",
      "Process analytics algorithm on objects ",
      "NVIDIA Corporation. Post on Deepstream forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvdsanalytics_init (GstNvDsAnalytics * nvdsanalytics)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvdsanalytics);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  nvdsanalytics->unique_id = DEFAULT_UNIQUE_ID;
  nvdsanalytics->configuration_width = DEFAULT_WIDTH;
  nvdsanalytics->configuration_height = DEFAULT_HEIGHT;
  nvdsanalytics->font_size = DEFAULT_FONT_SIZE;
  nvdsanalytics->osd_mode = DEFAULT_OSD_MODE;

  nvdsanalytics->config_file_path = NULL;
  nvdsanalytics->config_file_parse_successful = FALSE;
  nvdsanalytics->enable = TRUE;
  nvdsanalytics->reload_config = FALSE;
  nvdsanalytics->stream_analytics_info =
      new std::unordered_map < gint, StreamInfo >[1];
  nvdsanalytics->stream_analytics_ctx =
      new std::unordered_map < gint, NvDsAnalyticCtxUptr >[1];
  g_mutex_init (&nvdsanalytics->analytic_mutex);

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}


/* Free resources allocated during init. */
static void
gst_nvdsanalytics_finalize (GObject * object)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (object);

  nvdsanalytics->config_file_path = NULL;
  nvdsanalytics->config_file_parse_successful = FALSE;

  nvdsanalytics->stream_analytics_info->clear ();
  nvdsanalytics->stream_analytics_ctx->clear ();
  delete[]nvdsanalytics->stream_analytics_info;
  delete[]nvdsanalytics->stream_analytics_ctx;
  g_mutex_clear (&nvdsanalytics->analytic_mutex);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvdsanalytics_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      nvdsanalytics->unique_id = g_value_get_uint (value);
      break;
    case PROP_ENABLE:
      nvdsanalytics->enable = g_value_get_boolean (value);
      break;
    case PROP_CONFIG_FILE:
    {

      g_mutex_lock (&nvdsanalytics->analytic_mutex);
      g_free (nvdsanalytics->config_file_path);
      nvdsanalytics->config_file_path = g_value_dup_string (value);
      /* Parse the initialization parameters from the config file. This function
        * gives preference to values set through the set_property function over
        * the values set in the config file. */
      if (g_str_has_suffix(nvdsanalytics->config_file_path, ".yml") ||
          g_str_has_suffix(nvdsanalytics->config_file_path, ".yaml"))
      {
          nvdsanalytics->config_file_parse_successful =
                nvdsanalytics_parse_yaml_config_file (nvdsanalytics,
                nvdsanalytics->config_file_path);
      } else {
          nvdsanalytics->config_file_parse_successful =
                nvdsanalytics_parse_config_file (nvdsanalytics,
                nvdsanalytics->config_file_path);
      }

      if (nvdsanalytics->config_file_parse_successful){
        nvdsanalytics->stream_analytics_ctx->clear();
      }
      g_mutex_unlock (&nvdsanalytics->analytic_mutex);
    }

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
gst_nvdsanalytics_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, nvdsanalytics->unique_id);
      break;
    case PROP_ENABLE:
      g_value_set_boolean (value, nvdsanalytics->enable);
      break;
    case PROP_CONFIG_FILE:
      g_value_set_string (value, nvdsanalytics->config_file_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvdsanalytics_start (GstBaseTransform * btrans)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (btrans);

  /* Algorithm specific initializations and resource allocation. */
  nvdsanalytics->batch_size = 1;


  if (!nvdsanalytics->config_file_path
      || strlen (nvdsanalytics->config_file_path) == 0) {
    GST_ELEMENT_ERROR (nvdsanalytics, LIBRARY, SETTINGS,
        ("Configuration file not provided"), (nullptr));
    return FALSE;
  }
  if (nvdsanalytics->config_file_parse_successful == FALSE) {
    GST_ELEMENT_ERROR (nvdsanalytics, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", nvdsanalytics->config_file_path));
    return FALSE;
  }


  return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvdsanalytics_stop (GstBaseTransform * btrans)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (btrans);

  // Deinit the algorithm library

  GST_DEBUG_OBJECT (nvdsanalytics, "ctx lib released \n");

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvdsanalytics_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (btrans);
  gint batch_size = 1;
  GstStructure *structure = gst_caps_get_structure (incaps, 0);

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&nvdsanalytics->video_info, incaps);

  if (structure && gst_structure_get_int (structure, "batch-size", &batch_size)) {
    if (batch_size) {
      nvdsanalytics->batch_size = batch_size;
      GST_DEBUG_OBJECT (nvdsanalytics, "Setting batch-size %d from set caps\n",
          nvdsanalytics->batch_size);
    }
  }

  return TRUE;
}

static gboolean
gst_nvdsanalytics_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (trans);

  if ((GstNvDsCustomEventType) GST_EVENT_TYPE (event) ==
      GST_NVEVENT_ANALYTICS_RELOAD_CONFIG_UPDATE) {
    gchar *config_file_path = NULL;

    gst_nvevent_parse_analytics_reload_config_update (event, &config_file_path);

    if (config_file_path) {
      nvdsanalytics->reload_config = TRUE;
    }
  }

  /* Call the sink event handler of the base class */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}
/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvdsanalytics_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstNvDsAnalytics *nvdsanalytics = GST_NVDSANALYTICS (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  std::unordered_map < gint, NvDsAnalyticCtxUptr > &stream_analytics_ctx =
      *(nvdsanalytics->stream_analytics_ctx);
  std::unordered_map < gint, StreamInfo > &stream_analytics_info =
      *(nvdsanalytics->stream_analytics_info);
  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (nvdsanalytics));

  nvdsanalytics->batch_num++;

  if(nvdsanalytics->reload_config){
    printf("Ready to reload configuration file: %s\n", nvdsanalytics->config_file_path);
    g_mutex_lock (&nvdsanalytics->analytic_mutex);
    if (g_str_has_suffix(nvdsanalytics->config_file_path, ".yml") ||
        g_str_has_suffix(nvdsanalytics->config_file_path, ".yaml"))
    {
        nvdsanalytics->config_file_parse_successful =
            nvdsanalytics_parse_yaml_config_file (nvdsanalytics,
            nvdsanalytics->config_file_path);
    } else {
        nvdsanalytics->config_file_parse_successful =
            nvdsanalytics_parse_config_file (nvdsanalytics,
            nvdsanalytics->config_file_path);
    }

    if (nvdsanalytics->config_file_parse_successful){
        nvdsanalytics->stream_analytics_ctx->clear();
    }
    g_mutex_unlock (&nvdsanalytics->analytic_mutex);
    nvdsanalytics->reload_config = FALSE;
  }

  if (FALSE == nvdsanalytics->config_file_parse_successful) {
    GST_ELEMENT_ERROR (nvdsanalytics, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", nvdsanalytics->config_file_path));
    return flow_ret;
  }

  if (FALSE == nvdsanalytics->enable) {
    GST_DEBUG_OBJECT (nvdsanalytics, "DsAnalytics in passthrough mode");
    flow_ret = GST_FLOW_OK;
    return flow_ret;
  }

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    return flow_ret;
  }

  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (nvdsanalytics,
      "Processing Batch %" G_GUINT64_FORMAT " Surface %p\n",
      nvdsanalytics->batch_num, surface);

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (nullptr == batch_meta) {
    GST_ELEMENT_ERROR (nvdsanalytics, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return flow_ret;
  }
  // Using object crops as input to the algorithm. The objects are detected by
  // the primary detector
  NvDsMetaList *l_obj = nullptr;
  NvDsObjectMeta *obj_meta = nullptr;

  g_mutex_lock (&nvdsanalytics->analytic_mutex);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);
    NvDsAnalyticProcessParams process_params;
    int32_t cnt = 0;
    auto get_ctx = stream_analytics_ctx.find (frame_meta->pad_index);

    /* Create context if not present for particular stream */
    if (get_ctx == stream_analytics_ctx.end ()) {
      /* if obj_cnt_win_in_ms is not set, use the default value in lowlevel */
      if(nvdsanalytics->obj_cnt_win_in_ms == 0)
        nvdsanalytics->obj_cnt_win_in_ms = 50;
      NvDsAnalyticCtxUptr analytics_ctx =
          NvDsAnalyticCtx::create (stream_analytics_info[frame_meta->pad_index],
          frame_meta->pad_index,
          surface->surfaceList[frame_meta->batch_id].width,
          surface->surfaceList[frame_meta->batch_id].height,
          nvdsanalytics->obj_cnt_win_in_ms, 300, nvdsanalytics->obj_cnt_win_in_ms);
      stream_analytics_ctx[frame_meta->pad_index] = std::move (analytics_ctx);
    }
    process_params.frmPts = frame_meta->buf_pts;

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      ObjInf obj_info;
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      obj_info.left = obj_meta->rect_params.left;
      obj_info.top = obj_meta->rect_params.top;
      obj_info.width = obj_meta->rect_params.width;
      obj_info.height = obj_meta->rect_params.height;
      obj_info.object_id = obj_meta->object_id;
      obj_info.class_id = obj_meta->class_id;
      process_params.objList.push_back (obj_info);

    }
    stream_analytics_ctx[frame_meta->pad_index]->processSource (process_params);
    cnt = 0;
    //FIXME: Assumes no meta reordering
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      ObjInf obj_info;
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (process_params.objList[cnt].str_obj_status.length ()
          || process_params.objList[cnt].ocStatus.size ()) {
        attach_metadata_object (nvdsanalytics, obj_meta,
            process_params.objList[cnt]);
        //std::cout << process_params.objList[cnt].str_obj_status << " " <<  process_params.objList[cnt].object_id << std::endl;
      }
      cnt++;
    }
    attach_framemeta_analytics_metadata (nvdsanalytics, frame_meta,
        process_params, frame_meta->pad_index);
  }
  g_mutex_unlock (&nvdsanalytics->analytic_mutex);
  flow_ret = GST_FLOW_OK;

  nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME (nvdsanalytics));
  gst_buffer_unmap (inbuf, &in_map_info);

  return flow_ret;
}

static gpointer
copy_obj_nvdsanalytics_meta (gpointer data, gpointer user_data)
{

  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsAnalyticsObjInfo *src_user_metadata =
      (NvDsAnalyticsObjInfo *) user_meta->user_meta_data;
  NvDsAnalyticsObjInfo *dst_user_metadata = new NvDsAnalyticsObjInfo[1];
  *dst_user_metadata = *src_user_metadata;
  return (gpointer) dst_user_metadata;
}

static void
release_obj_nvdsanalytics_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsAnalyticsObjInfo *user_meta_data =
      (NvDsAnalyticsObjInfo *) user_meta->user_meta_data;
  if (user_meta_data) {
    delete[]user_meta_data;
    user_meta->user_meta_data = NULL;
  }
}

#define CHECK_AQUIRE_USER_OBJ_META \
   if (user_meta == NULL) { \
      user_meta = nvds_acquire_user_meta_from_pool(batch_meta); \
      user_obj_meta = new NvDsAnalyticsObjInfo[1];\
      user_meta->user_meta_data = (void*)user_obj_meta; \
      user_meta->base_meta.meta_type = user_meta_type; \
      user_meta->base_meta.copy_func =  (NvDsMetaCopyFunc)copy_obj_nvdsanalytics_meta;\
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_obj_nvdsanalytics_meta;  \
   }

#define ATTACH_USER_OBJ_META \
  if (user_meta) { \
    nvds_add_user_meta_to_obj (obj_meta, user_meta); \
    user_meta = NULL; \
  }


#define ATTACH_DISPLAY_META \
  if (display_meta) { \
    nvds_add_display_meta_to_frame (frame_meta, display_meta); \
    display_meta = NULL; \
  }

#define ATTACH_USER_FRAME_META \
  if (user_meta) { \
    nvds_add_user_meta_to_frame (frame_meta, user_meta); \
    user_meta = NULL; \
  }
static gpointer
copy_frame_nvdsanalytics_meta (gpointer data, gpointer user_data)
{

  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsAnalyticsFrameMeta *src_user_metadata =
      (NvDsAnalyticsFrameMeta *) user_meta->user_meta_data;
  NvDsAnalyticsFrameMeta *dst_user_metadata = new NvDsAnalyticsFrameMeta[1];
  *dst_user_metadata = *src_user_metadata;
  return (gpointer) dst_user_metadata;
}

static void
release_frame_nvdsanalytics_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsAnalyticsFrameMeta *user_meta_data =
      (NvDsAnalyticsFrameMeta *) user_meta->user_meta_data;
  if (user_meta_data) {
    delete[]user_meta_data;
    user_meta->user_meta_data = NULL;
  }
}

#define CHECK_AQUIRE_USER_FRAME_META \
   if (user_meta == NULL) { \
      user_meta = nvds_acquire_user_meta_from_pool(batch_meta); \
      user_frame_meta = new NvDsAnalyticsFrameMeta[1];\
      user_meta->user_meta_data = (void*)user_frame_meta; \
      user_meta->base_meta.meta_type = user_meta_type; \
      user_meta->base_meta.copy_func =  (NvDsMetaCopyFunc)copy_frame_nvdsanalytics_meta;\
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_frame_nvdsanalytics_meta;  \
      user_frame_meta->unique_id = nvdsanalytics->unique_id; \
   }



#define CHECK_ATTACH_AQUIRE_DISPLAY_META \
   if (display_meta == NULL || \
       display_meta->num_labels >= MAX_ELEMENTS_IN_DISPLAY_META-1 || \
       display_meta->num_lines  >= MAX_ELEMENTS_IN_DISPLAY_META-1 ){ \
      ATTACH_DISPLAY_META; \
      display_meta = nvds_acquire_display_meta_from_pool(batch_meta); \
      display_meta->num_lines = 0; display_meta->num_labels = 0; \
    }

#define GET_TEXT_PARAMS &display_meta->text_params[display_meta->num_labels]
#define GET_LINE_PARAMS &display_meta->line_params[display_meta->num_lines]


static void
get_arrow_head (gint x1, gint y1,
    gint x2, gint y2, gint & x3, gint & y3, gint & x4, gint & y4)
{
  //x3=x2+L2/L1[(x1−x2)cosθ+(y1−y2)sinθ],
  //y3=y2+L2/L1[(y1−y2)cosθ−(x1−x2)sinθ],
  //x4=x2+L2/L1[(x1−x2)cosθ−(y1−y2)sinθ],
  //y4=y2+L2/L1[(y1−y2)cosθ+(x1−x2)sinθ].
  double theta = (15 * M_PI) / 180.0;
  x3 = x2 + 0.15 * ((x1 - x2) * cos (theta) + (y1 - y2) * sin (theta));
  y3 = y2 + 0.15 * ((y1 - y2) * cos (theta) - (x1 - x2) * sin (theta));
  x4 = x2 + 0.15 * ((x1 - x2) * cos (theta) - (y1 - y2) * sin (theta));
  y4 = y2 + 0.15 * ((y1 - y2) * cos (theta) + (x1 - x2) * sin (theta));

}


static void
attach_framemeta_analytics_metadata (GstNvDsAnalytics * nvdsanalytics,
    NvDsFrameMeta * frame_meta, NvDsAnalyticProcessParams & process_params,
    gint stream_id)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  std::unordered_map < int, StreamInfo > &stream_analytics_info =
      *(nvdsanalytics->stream_analytics_info);
  NvDsDisplayMeta *display_meta = NULL;
  NvOSD_TextParams *txt_params = NULL;
  NvOSD_LineParams *line_params = NULL;
  NvDsUserMeta *user_meta = NULL;
  NvDsAnalyticsFrameMeta *user_frame_meta = NULL;
  NvDsMetaType user_meta_type = NVDS_USER_FRAME_META_NVDSANALYTICS;
  std::stringstream str_obj_cnt;

  nvds_acquire_meta_lock (batch_meta);

  CHECK_AQUIRE_USER_FRAME_META;
  if (user_frame_meta) {
    user_frame_meta->objCnt = process_params.objCnt;
    if (nvdsanalytics->display_obj_cnt) {
      std::map < uint32_t, uint32_t > ordObjCnt (begin (process_params.objCnt),
          end (process_params.objCnt));
      str_obj_cnt << "Count for";
    for (auto & each_cls_cnt:ordObjCnt) {
        str_obj_cnt << " ClassId" << each_cls_cnt.first <<
            "=" << each_cls_cnt.second;
      }
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
      txt_params = GET_TEXT_PARAMS;
      txt_params->display_text =
          (gchar *) g_malloc0 (str_obj_cnt.str ().size () + 1);
      snprintf (txt_params->display_text, str_obj_cnt.str ().size () + 1, "%s",
          str_obj_cnt.str ().c_str ());
      txt_params->x_offset = 5;
      txt_params->y_offset = 5;

      /* Font , font-color and font-size */
      txt_params->font_params = (NvOSD_FontParams) {
        (gchar *) "Serif", nvdsanalytics->font_size, {
        1.0, 1.0, 0.0, 1.0}
      };
      txt_params->set_bg_clr = 1;
      txt_params->text_bg_clr = (NvOSD_ColorParams) {
      0.0, 0.0, 0, 1.0};
      display_meta->num_labels++;
    }
  }

for (auto & roi:stream_analytics_info[stream_id].roi_info) {
    if (!roi.enable)
      continue;
    guint icnt = 0;
    txt_params = NULL;
    CHECK_AQUIRE_USER_FRAME_META;
    if (user_frame_meta)
      user_frame_meta->objInROIcnt[roi.roi_label] =
          process_params.objInROIcnt[roi.roi_label];

    if (nvdsanalytics->osd_mode) {
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
      txt_params = GET_TEXT_PARAMS;
      txt_params->display_text = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

      //display only label
      if (nvdsanalytics->osd_mode == 2)
        snprintf (txt_params->display_text, MAX_LABEL_SIZE, "%s=%d",
            roi.roi_label.c_str (), process_params.objInROIcnt[roi.roi_label]);
      else
        snprintf (txt_params->display_text, MAX_LABEL_SIZE, "%s",
            roi.roi_label.c_str ());

      /* Now set the offsets where the string should appear */
      txt_params->x_offset = roi.roi_pts[0].first;
      txt_params->y_offset = roi.roi_pts[1].second;

      /* Font , font-color and font-size */
      txt_params->font_params = (NvOSD_FontParams) {
        (gchar *) "Serif", nvdsanalytics->font_size, {
        1.0, 1.0, 0.0, 1.0}
      };
      txt_params->set_bg_clr = 1;
      txt_params->text_bg_clr = (NvOSD_ColorParams) {
      0.0, 0.0, 0, 1.0};
      display_meta->num_labels++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      for (icnt = 1; icnt < roi.roi_pts.size (); icnt++) {
        line_params = GET_LINE_PARAMS;
        line_params->x1 = roi.roi_pts[icnt - 1].first;
        line_params->y1 = roi.roi_pts[icnt - 1].second;
        line_params->x2 = roi.roi_pts[icnt].first;
        line_params->y2 = roi.roi_pts[icnt].second;
        line_params->line_color = (NvOSD_ColorParams) {
        1.0, 1.0, 0, 1.0};
        line_params->line_width = 2;
        display_meta->num_lines++;
        CHECK_ATTACH_AQUIRE_DISPLAY_META;
      }
      line_params = GET_LINE_PARAMS;
      line_params->x1 = roi.roi_pts[icnt - 1].first;
      line_params->y1 = roi.roi_pts[icnt - 1].second;
      line_params->x2 = roi.roi_pts[0].first;
      line_params->y2 = roi.roi_pts[0].second;
      line_params->line_color = (NvOSD_ColorParams) {
      1.0, 1.0, 0, 1.0};
      line_params->line_width = 2;
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
    }
  }

for (auto & roi:stream_analytics_info[stream_id].overcrowding_info) {
    if (!roi.enable)
      continue;
    guint icnt = 0;
    CHECK_AQUIRE_USER_FRAME_META;

    if (user_frame_meta) {
      user_frame_meta->ocStatus[roi.oc_label] =
          process_params.ocStatus[roi.oc_label].overCrowding;
      user_frame_meta->objInROIcnt[roi.oc_label] =
          process_params.ocStatus[roi.oc_label].overCrowdingCount;
    }

    if (nvdsanalytics->osd_mode) {
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
      txt_params = GET_TEXT_PARAMS;
      txt_params->display_text = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

      if (nvdsanalytics->osd_mode == 2) {
        if (process_params.ocStatus[roi.oc_label].overCrowding)
          snprintf (txt_params->display_text, MAX_LABEL_SIZE,
              "%s OverCrowding=True, Count=%d", roi.oc_label.c_str (),
              process_params.ocStatus[roi.oc_label].overCrowdingCount);
        else
          snprintf (txt_params->display_text, MAX_LABEL_SIZE,
              "%s OverCrowding=False, Count=%d", roi.oc_label.c_str (),
              process_params.ocStatus[roi.oc_label].overCrowdingCount);
      } else
        snprintf (txt_params->display_text, MAX_LABEL_SIZE, "%s",
            roi.oc_label.c_str ());
      /* Now set the offsets where the string should appear */
      txt_params->x_offset = roi.roi_pts[0].first;
      txt_params->y_offset = roi.roi_pts[1].second;
      /* Font , font-color and font-size */
      txt_params->font_params = (NvOSD_FontParams) {
        (gchar *) "Serif", nvdsanalytics->font_size, {
        1.0, 0.5, 0.0, 1.0}
      };
      txt_params->set_bg_clr = 1;
      txt_params->text_bg_clr = (NvOSD_ColorParams) {
      0.0, 0.0, 0, 1.0};
      display_meta->num_labels++;

      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      for (icnt = 1; icnt < roi.roi_pts.size (); icnt++) {
        line_params = GET_LINE_PARAMS;
        line_params->x1 = roi.roi_pts[icnt - 1].first;
        line_params->y1 = roi.roi_pts[icnt - 1].second;
        line_params->x2 = roi.roi_pts[icnt].first;
        line_params->y2 = roi.roi_pts[icnt].second;
        line_params->line_color = (NvOSD_ColorParams) {
        1.0, 0.5, 0, 1.0};
        line_params->line_width = 2;
        display_meta->num_lines++;
        CHECK_ATTACH_AQUIRE_DISPLAY_META;
      }
      line_params = GET_LINE_PARAMS;
      line_params->x1 = roi.roi_pts[icnt - 1].first;
      line_params->y1 = roi.roi_pts[icnt - 1].second;
      line_params->x2 = roi.roi_pts[0].first;
      line_params->y2 = roi.roi_pts[0].second;
      line_params->line_color = (NvOSD_ColorParams) {
      1.0, 0.5, 0, 1.0};
      line_params->line_width = 2;
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
    }
  }

for (auto & roi:stream_analytics_info[stream_id].direction_info) {
    gint x3, y3, x4, y4;
    if (!roi.enable)
      continue;

    if (nvdsanalytics->osd_mode) {
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
      txt_params = GET_TEXT_PARAMS;
      txt_params->display_text = (gchar *) g_malloc0 (MAX_LABEL_SIZE);
      snprintf (txt_params->display_text, MAX_LABEL_SIZE, "%s",
          roi.dir_label.c_str ());
      /* Now set the offsets where the string should appear */
      txt_params->x_offset = roi.x1y1.first;
      txt_params->y_offset = roi.x1y1.second;
      /* Font , font-color and font-size */
      txt_params->font_params = (NvOSD_FontParams) {
        (gchar *) "Serif", nvdsanalytics->font_size, {
        1.0, 0.0, 0.0, 1.0}
      };
      txt_params->set_bg_clr = 1;
      txt_params->text_bg_clr = (NvOSD_ColorParams) {
      0.0, 0.0, 0, 1.0};
      display_meta->num_labels++;

      line_params = GET_LINE_PARAMS;
      line_params->x1 = roi.x1y1.first;
      line_params->y1 = roi.x1y1.second;
      line_params->x2 = roi.x2y2.first;
      line_params->y2 = roi.x2y2.second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      1.0, 0.0, 0, 1.0};
      display_meta->num_lines++;

      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      get_arrow_head (line_params->x1, line_params->y1,
          line_params->x2, line_params->y2, x3, y3, x4, y4);

      line_params = GET_LINE_PARAMS;
      line_params->x1 = x3;
      line_params->y1 = y3;
      line_params->x2 = roi.x2y2.first;
      line_params->y2 = roi.x2y2.second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      1.0, 0.0, 0, 1.0};
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      line_params = GET_LINE_PARAMS;
      line_params->x1 = x4;
      line_params->y1 = y4;
      line_params->x2 = roi.x2y2.first;
      line_params->y2 = roi.x2y2.second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      1.0, 0.0, 0, 1.0};
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
    }
  }

for (auto & roi:stream_analytics_info[stream_id].linecrossing_info) {
    gint x3, y3, x4, y4;
    if (!roi.enable)
      continue;

    CHECK_AQUIRE_USER_FRAME_META;
    if (user_frame_meta) {
      user_frame_meta->objLCCurrCnt[roi.lc_label] =
          process_params.objLCCurrCnt[roi.lc_label];
      user_frame_meta->objLCCumCnt[roi.lc_label] =
          process_params.objLCCumCnt[roi.lc_label];
    }

    if (nvdsanalytics->osd_mode) {
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
      txt_params = GET_TEXT_PARAMS;
      txt_params->display_text = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

      if (nvdsanalytics->osd_mode == 2)
        snprintf (txt_params->display_text, MAX_LABEL_SIZE, "%s=%lu",
            roi.lc_label.c_str (), process_params.objLCCumCnt[roi.lc_label]);
      else
        snprintf (txt_params->display_text, MAX_LABEL_SIZE, "%s ",
            roi.lc_label.c_str ());

      /* Now set the offsets where the string should appear */
      txt_params->x_offset = roi.lcdir_pts[3].first;
      txt_params->y_offset = roi.lcdir_pts[3].second;

      /* Font , font-color and font-size */
      txt_params->font_params = (NvOSD_FontParams) {
        (gchar *) "Serif", nvdsanalytics->font_size, {
        0.0, 1.0, 0.0, 1.0}
      };
      txt_params->set_bg_clr = 1;
      txt_params->text_bg_clr = (NvOSD_ColorParams) {
      0.0, 0.0, 0, 1.0};
      display_meta->num_labels++;

      line_params = GET_LINE_PARAMS;
      line_params->x1 = roi.lcdir_pts[0].first;
      line_params->y1 = roi.lcdir_pts[0].second;
      line_params->x2 = roi.lcdir_pts[1].first;
      line_params->y2 = roi.lcdir_pts[1].second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      0.0, 1.0, 0, 1.0};
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      get_arrow_head (line_params->x1, line_params->y1,
          line_params->x2, line_params->y2, x3, y3, x4, y4);

      line_params = GET_LINE_PARAMS;
      line_params->x1 = x3;
      line_params->y1 = y3;
      line_params->x2 = roi.lcdir_pts[1].first;
      line_params->y2 = roi.lcdir_pts[1].second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      0.0, 1.0, 0, 1.0};
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      line_params = GET_LINE_PARAMS;
      line_params->x1 = x4;
      line_params->y1 = y4;
      line_params->x2 = roi.lcdir_pts[1].first;
      line_params->y2 = roi.lcdir_pts[1].second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      0.0, 1.0, 0, 1.0};
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;

      line_params = GET_LINE_PARAMS;
      line_params->x1 = roi.lcdir_pts[2].first;
      line_params->y1 = roi.lcdir_pts[2].second;
      line_params->x2 = roi.lcdir_pts[3].first;
      line_params->y2 = roi.lcdir_pts[3].second;
      line_params->line_width = 2;
      line_params->line_color = (NvOSD_ColorParams) {
      0.0, 1.0, 0, 1.0};
      display_meta->num_lines++;
      CHECK_ATTACH_AQUIRE_DISPLAY_META;
    }
  }

  ATTACH_DISPLAY_META;
  ATTACH_USER_FRAME_META;

  nvds_release_meta_lock (batch_meta);
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void
attach_metadata_object (GstNvDsAnalytics * nvdsanalytics,
    NvDsObjectMeta * obj_meta, ObjInf & obj_info)
{
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;
  NvDsUserMeta *user_meta = NULL;
  NvDsAnalyticsObjInfo *user_obj_meta = NULL;
  NvDsMetaType user_meta_type = NVDS_USER_OBJ_META_NVDSANALYTICS;

  CHECK_AQUIRE_USER_OBJ_META;
  nvds_acquire_meta_lock (batch_meta);
  // To display dynamic information
  if (nvdsanalytics->osd_mode == 2) {
    NvOSD_TextParams & text_params = obj_meta->text_params;
    NvOSD_RectParams & rect_params = obj_meta->rect_params;

    /* Below code to display the result */
    // Set black background for the text
    // display_text required heap allocated memory
    if (text_params.display_text) {
      gchar *conc_string = g_strconcat (text_params.display_text, " ",
          obj_info.str_obj_status.c_str (), NULL);

      g_free (text_params.display_text);
      text_params.display_text = conc_string;
    } else {
      // Display text above the left top corner of the object
      text_params.x_offset = rect_params.left;
      text_params.y_offset = rect_params.top - 10;
      text_params.display_text = g_strdup (obj_info.str_obj_status.c_str ());
      // Font face, size and color
      text_params.font_params.font_name = (char *) "Serif";
      text_params.font_params.font_size = nvdsanalytics->font_size;
      text_params.font_params.font_color = (NvOSD_ColorParams) {
      1, 1, 1, 1};
      // Set black background for the text
      text_params.set_bg_clr = 1;
      text_params.text_bg_clr = (NvOSD_ColorParams) {
      0, 0, 0, 1};
    }

    if ((std::string::npos != obj_info.str_obj_status.find ("ROI:")) ||
        (std::string::npos != obj_info.str_obj_status.find ("LC:"))) {
      rect_params.border_color.red = 1.0 - rect_params.border_color.red;
      rect_params.border_color.green = 1.0 - rect_params.border_color.green;
      rect_params.border_color.blue = 1.0 - rect_params.border_color.blue;
    }
  }
    if (user_obj_meta) {
      user_obj_meta->roiStatus = obj_info.roiStatus;
      user_obj_meta->ocStatus = obj_info.ocStatus;
      user_obj_meta->lcStatus = obj_info.lcStatus;
      user_obj_meta->objStatus = obj_info.str_obj_status;
      user_obj_meta->dirStatus = obj_info.dirStatus;
      user_obj_meta->unique_id = nvdsanalytics->unique_id;
    }

  ATTACH_USER_OBJ_META;
  nvds_release_meta_lock (batch_meta);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdsanalytics_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsanalytics_debug, "nvdsanalytics", 1,
      "nvdsanalytics plugin");
  //gst_debug_category_set_threshold (gst_nvdsanalytics_debug,
  //    GstDebugLevel level)

  return gst_element_register (plugin, "nvdsanalytics", GST_RANK_PRIMARY,
      GST_TYPE_DSANALYTICS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsanalytics,
    DESCRIPTION, nvdsanalytics_plugin_init, "9.0", LICENSE, BINARY_PACKAGE,
    URL)
