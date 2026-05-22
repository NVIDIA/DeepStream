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
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/stat.h>

#include "gst-nvevent.h"
#include "gst-nvquery.h"
#include "gstnvreplay.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvreplay_debug);
#define GST_CAT_DEFAULT gst_nvreplay_debug
#define USE_EGLIMAGE 1

static GQuark _dsmeta_quark = 0;
using namespace std;
/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_LABEL_DIR,
  PROP_FILE_NAMES,
  PROP_UNIQUE_ID,
  PROP_MAX_FRAME_NUMS,
  PROP_LABEL_WIDTH,
  PROP_LABEL_HEIGHT,
  PROP_INTERVAL
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_LABEL_WIDTH 1920
#define DEFAULT_LABEL_HEIGHT 1080
#define DEFAULT_INTERVAL 1

#define LABEL_CLASS 1
#define LABEL_NAME "Person"
#define MAX_LABEL_FRAMES INT_MAX

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvreplay_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_nvreplay_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
G_DEFINE_TYPE (GstNvReplay, gst_nvreplay, GST_TYPE_BASE_TRANSFORM);

static void gst_nvreplay_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvreplay_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvreplay_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvreplay_start (GstBaseTransform * btrans);
static gboolean gst_nvreplay_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_nvreplay_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);

static void
attach_metadata_full_frame (vector<Bbox>& frame_bboxes, NvDsFrameMeta *frame_meta,
      const float scale_width, const float scale_height);
static GstFlowReturn
load_replay_data(GstNvReplay * nvreplay);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvreplay_class_init (GstNvReplayClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  /* Indicates we want to use DS buf api */
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvreplay_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvreplay_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvreplay_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvreplay_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvreplay_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvreplay_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_LABEL_DIR,
      g_param_spec_string("label-dir", "Object ground truth label directory",
      "Object ground truth label directory", nullptr,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS )));
  g_object_class_install_property (gobject_class, PROP_FILE_NAMES,
      g_param_spec_string("file-names", "Object ground truth file names",
      "Object ground truth file names", nullptr,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS )));
  g_object_class_install_property (gobject_class, PROP_MAX_FRAME_NUMS,
      g_param_spec_string("max-frame-nums", "Maximum number of frames for each stream",
      "Maximum number of frames for each stream", nullptr,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS )));
  g_object_class_install_property (gobject_class, PROP_LABEL_WIDTH,
      g_param_spec_uint ("label-width", "Label width",
        "Frame width in the scale of ground truth labels",
        0, G_MAXUINT, DEFAULT_LABEL_WIDTH,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property (gobject_class, PROP_LABEL_HEIGHT,
      g_param_spec_uint ("label-height", "Label height",
        "Frame height in the scale of ground truth labels",
        0, G_MAXUINT, DEFAULT_LABEL_HEIGHT,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property (gobject_class, PROP_INTERVAL,
      g_param_spec_uint ("interval", "Load label interval",
        "Number of consecutive batches to be skipped for adding label to meta",
        0, 32, DEFAULT_INTERVAL,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvreplay_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvreplay_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvReplay plugin",
      "NvReplay Plugin",
      "Process a 3rdparty example algorithm on objects / full frame",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvreplay_init (GstNvReplay * nvreplay)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvreplay);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  nvreplay->unique_id = DEFAULT_UNIQUE_ID;
  nvreplay->file_names = NULL;
  nvreplay->label_dir = NULL;
  nvreplay->max_frame_nums_str = NULL;
  nvreplay->max_frame_nums.clear();
  nvreplay->label_width = DEFAULT_LABEL_WIDTH;
  nvreplay->label_height = DEFAULT_LABEL_HEIGHT;
  nvreplay->interval = DEFAULT_INTERVAL;
  nvreplay->interval_counter = 0;

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvreplay_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvReplay *nvreplay = GST_NVREPLAY (object);
  switch (prop_id) {
    case PROP_LABEL_DIR:
      if (nvreplay->label_dir)
      {
        g_free(nvreplay->label_dir);
      }
      nvreplay->label_dir = (char*) g_value_dup_string(value);;
      break;
    case PROP_FILE_NAMES:
      if (nvreplay->file_names)
      {
        g_free(nvreplay->file_names);
      }
      nvreplay->file_names = (char*) g_value_dup_string(value);
      break;
    case PROP_MAX_FRAME_NUMS:
      if (nvreplay->max_frame_nums_str)
      {
        g_free(nvreplay->max_frame_nums_str);
      }
      nvreplay->max_frame_nums_str = (char*) g_value_dup_string(value);
      break;
    case PROP_LABEL_WIDTH:
      nvreplay->label_width = g_value_get_uint(value);
      break;
    case PROP_LABEL_HEIGHT:
      nvreplay->label_height = g_value_get_uint(value);
      break;
    case PROP_INTERVAL:
      nvreplay->interval = g_value_get_uint(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvreplay_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvReplay *nvreplay = GST_NVREPLAY (object);

  switch (prop_id) {
    case PROP_LABEL_DIR:
      g_value_set_string (value, nvreplay->label_dir);
      break;
    case PROP_FILE_NAMES:
      g_value_set_string (value, nvreplay->file_names);
      break;
    case PROP_MAX_FRAME_NUMS:
      g_value_set_string (value, nvreplay->max_frame_nums_str);
      break;
    case PROP_LABEL_WIDTH:
      g_value_set_uint (value, nvreplay->label_width);
      break;
    case PROP_LABEL_HEIGHT:
      g_value_set_uint (value, nvreplay->label_height);
      break;
    case PROP_INTERVAL:
      g_value_set_uint (value, nvreplay->interval);
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
gst_nvreplay_start (GstBaseTransform * btrans)
{
  GstNvReplay *nvreplay = GST_NVREPLAY (btrans);

  GstQuery *queryparams = NULL;
  queryparams = gst_nvquery_batch_size_new ();
  if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), queryparams)
      || gst_pad_peer_query (GST_BASE_TRANSFORM_SRC_PAD (btrans), queryparams)) {
    if (!gst_nvquery_batch_size_parse (queryparams, &nvreplay->batch_size)) {
      GST_ERROR("nvreplay: Failed to query batch size\n");
      return FALSE;
    }
  }

  gst_query_unref (queryparams);

  // Load labels from ground truth files
  load_replay_data(nvreplay);

  return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvreplay_stop (GstBaseTransform * btrans)
{
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvreplay_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  return TRUE;
}

/*
* Load ground truth object labels (bounding box) from external file
*/
static GstFlowReturn
load_replay_data(GstNvReplay * nvreplay)
{
  // Split file names
  const string label_dir(nvreplay->label_dir), file_names(nvreplay->file_names);
  string file_name, label_path;
  stringstream file_names_stream(file_names);
  struct stat buf;

  nvreplay->label_bboxes.clear();
  nvreplay->max_frame_nums.clear();

  while (getline(file_names_stream, file_name, ';'))
  {
    label_path = label_dir + file_name;

    // Check whether file exists
    if (stat(label_path.c_str(), &buf) == -1)
    {
      GST_ELEMENT_ERROR (nvreplay, STREAM, FAILED,
        ("%s:path %s does not exist", __func__, label_path.c_str()), (NULL));
      return GST_FLOW_ERROR;
    }

    // Load labels from file
    nvreplay->label_bboxes.push_back(vector<vector<Bbox>>());
    vector<vector<Bbox>>& stream_bboxes = nvreplay->label_bboxes.back();
    ifstream file(label_path);
    int frame = 1;

    while(file)
    {
      int num_token=0;
      string line, token;
      Bbox bbox;
      getline(file, line);
      stringstream ss_line(line);
      //printf("line: %s\n", line.c_str());
      while(getline(ss_line, token, ','))
      {
        if(num_token == 0)
        {
          // token 0: frame num
          // In ground truth label files staring from 1, but in DS starting from 0
          frame = stoi(token);
          if( frame > MAX_LABEL_FRAMES || frame < 1)
          {
            GST_ELEMENT_ERROR (nvreplay, STREAM, FAILED,
                    ("%s:frame number should in range [1, %d]", __func__, MAX_LABEL_FRAMES), (NULL));
            return GST_FLOW_ERROR;
          }

          // Add new frames to stream_bboxes
          for(int i = stream_bboxes.size(); i < frame; i++)
          {
            stream_bboxes.push_back(vector<Bbox>());
          }
        }
        else if(num_token >= 2 && num_token <= 6)
        {
          // token [2, 5] bbox, [6] confidence
          bbox.push_back(stof(token));
        }

        num_token++;
      } // read bbox from a line
      if(bbox.size() == 5)
      {
        stream_bboxes[frame-1].push_back(move(bbox));
      }
    } // read labels from a single file

      // Use max frame in label file as the stream's max frame num by default
      nvreplay->max_frame_nums.push_back(nvreplay->label_bboxes.back().size());
  } // read labels from all streams; each stream has a file

  if( (guint)(nvreplay->label_bboxes.size()) != nvreplay->batch_size )
  {
    GST_ELEMENT_ERROR (nvreplay, STREAM, FAILED,
            ("%s:number of files does not match batch size", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  printf("Read %d label files\n", (int)(nvreplay->label_bboxes.size()));

  if (nvreplay->max_frame_nums_str)
  {
    stringstream max_frame_nums_stream(nvreplay->max_frame_nums_str);
    string frame_num_str;
    guint stream_ind = 0;

    while (getline(max_frame_nums_stream, frame_num_str, ';') && stream_ind < (guint)nvreplay->max_frame_nums.size())
    {
      if (!frame_num_str.empty())
      {
        nvreplay->max_frame_nums[stream_ind] = std::stoul(frame_num_str);
      }
      stream_ind++;
    }
  }

  return GST_FLOW_OK;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvreplay_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstNvReplay *nvreplay = GST_NVREPLAY (btrans);
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList * l_frame = NULL;

  GstMapInfo inmap;

  memset(&inmap, 0, sizeof(inmap));
  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ))
  {
    return GST_FLOW_ERROR;
  }
  NvBufSurface* input_buffer  = reinterpret_cast<NvBufSurface*>(inmap.data);
  gst_buffer_unmap(inbuf, &inmap);

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvreplay, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  // Load labels if needed for current frame
    if (nvreplay->interval_counter == 0)
  {
    nvds_acquire_meta_lock (batch_meta);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      if (frame_meta->pad_index >= nvreplay->batch_size)
      {
        GST_ERROR("nvreplay: label file doesn't contain current stream");
        continue;
      }

      int label_ind = frame_meta->frame_num % nvreplay->max_frame_nums[frame_meta->pad_index];

      vector<Bbox>& frame_bboxes = nvreplay->label_bboxes[frame_meta->pad_index][label_ind];

      NvBufSurfaceParams &pInputBuf = input_buffer->surfaceList[frame_meta->batch_id];
      float scale_width = (float)(pInputBuf.width) / nvreplay->label_width;
      float scale_height = (float)(pInputBuf.height) / nvreplay->label_height;

      /* Attach the metadata for the full frame */
      attach_metadata_full_frame (frame_bboxes, frame_meta, scale_width, scale_height);
    }
  }

  nvds_release_meta_lock (batch_meta);

  nvreplay->interval_counter = (nvreplay->interval_counter + 1) % (nvreplay->interval + 1);
  return GST_FLOW_OK;
}

/**
 * Attach metadata for the full frame. We will be adding a new metadata.
 */
static void
attach_metadata_full_frame (vector<Bbox>& frame_bboxes, NvDsFrameMeta *frame_meta,
    const float scale_width, const float scale_height)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  NvDsObjectMeta *object_meta = NULL;
  static gchar font_name[] = "Serif";


  for (int i = 0; i < (int)(frame_bboxes.size()); i++) {
    Bbox bbox = frame_bboxes[i];
    if (bbox.size() != 5)
    {
      continue;
    }

    object_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
    NvOSD_RectParams & rect_params = object_meta->rect_params;
    NvOSD_TextParams & text_params = object_meta->text_params;

    /* Assign bounding box coordinates */
    rect_params.left = bbox[0] * scale_width;
    rect_params.top = bbox[1] * scale_height;
    rect_params.width = bbox[2] * scale_width;
    rect_params.height = bbox[3] * scale_height;

    /* Semi-transparent yellow background */
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    /* Red border of width 6 */
    rect_params.border_width = 3;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    object_meta->unique_component_id = DEFAULT_UNIQUE_ID;
    object_meta->confidence = bbox[4];
    object_meta->object_id = UNTRACKED_OBJECT_ID;
    object_meta->class_id = LABEL_CLASS;

    g_strlcpy (object_meta->obj_label, LABEL_NAME, MAX_LABEL_SIZE);
    /* display_text required heap allocated memory */
    text_params.display_text = g_strdup (LABEL_NAME);
    /* Display text above the left top corner of the object */
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    /* Set black background for the text */
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    /* Font face, size and color */
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};

    nvds_add_obj_meta_to_frame(frame_meta, object_meta, NULL);
  }
  frame_meta->bInferDone = TRUE;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvreplay_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvreplay_debug, "nvreplay", 0,
      "nvreplay plugin");

  return gst_element_register (plugin, "nvreplay", GST_RANK_PRIMARY,
      GST_TYPE_NVREPLAY);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_replay,
    DESCRIPTION, nvreplay_plugin_init, "9.0", LICENSE, BINARY_PACKAGE, URL)
