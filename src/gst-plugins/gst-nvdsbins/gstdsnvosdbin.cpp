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

#include "gstdsnvosdbin.h"
#include "gstnvdsbinutils.h"

#include "gstnvdsmeta.h"
#include "string.h"

extern "C" GType gst_nvvideoconvert_get_type ();
extern "C" GType gst_nvds_osd_get_type (void);

GST_DEBUG_CATEGORY (gst_ds_nvosd_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvosd_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvosd_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvosd_bin_debug, "nvdsosdbin", 0, "nvdsosdbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvOSDBin, gst_ds_nvosd_bin, GST_TYPE_BIN,
    _do_init);

static void gst_ds_nvosd_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_ds_nvosd_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);
static void gst_ds_nvosd_bin_set_display_text_enabled (GstDsNvOSDBin *
    nvtilerbin, gboolean enabled);


enum
{
  SIGNAL_SET_DISPLAY_TEXT_ENABLED,
  LAST_SIGNAL
};

guint gst_ds_nvosd_bin_signals[LAST_SIGNAL] = { 0 };

#define DEFAULT_FONT "Serif"
#define DEFAULT_DISPLAY_TRACKING_ID TRUE
#define DEFAULT_REFORMAT_OBJECT_LABELS FALSE

static void
parse_color_from_int (guint color_hex, NvOSD_ColorParams & color)
{
  color.red = ((color_hex & 0xFF000000) >> 24) / 255.0;
  color.green = ((color_hex & 0x00FF0000) >> 16) / 255.0;
  color.blue = ((color_hex & 0x0000FF00) >> 8) / 255.0;
  color.alpha = ((color_hex & 0x000000FF)) / 255.0;
}

static guint
color_to_int (NvOSD_ColorParams & color)
{
  return ((guint) (color.red * 255) & 0xFF << 24) | ((guint) (color.green *
          255) & 0xFF << 16) | ((guint) (color.blue *
          255) & 0xFF << 8) | ((guint) (color.alpha * 255) & 0xFF);
}

static void
parse_colors_from_list (const gchar * str, std::unordered_map < std::string,
    NvOSD_ColorParams > &class_color_map)
{
  gchar **classes = g_strsplit (str, ";", -1);
  guint num_classes = g_strv_length (classes);

  for (guint i = 0; i < num_classes; i++) {
    gchar **class_colors = g_strsplit (classes[i], "=", -1);
    if (g_strv_length (class_colors) != 2) {
      g_strfreev (class_colors);
      g_printerr
          ("Incorrect format %s for class color map. Format - <class-label1>=hex1;<class-label2>=hex2. hex is RGBA hex format",
          classes[i]);
      continue;
    }

    NvOSD_ColorParams color;
    gchar *nptr = nullptr;
    guint64 color_hex = g_ascii_strtoull (class_colors[1], &nptr, 16);
    if (nptr && *nptr != '\0') {
      g_strfreev (class_colors);
      g_printerr
          ("Incorrect format %s for class color map. Format - <class-label1>=hex1;<class-label2>=hex2. hex is RGBA hex format",
          classes[i]);
      continue;
    }
    parse_color_from_int (color_hex, color);

    class_color_map[class_colors[0]] = color;
    g_strfreev (class_colors);
  }
  g_strfreev (classes);
}

static void
gst_ds_nvosd_bin_class_init (GstDsNvOSDBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvosd_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvosd_bin_get_property);

  GType osd_elem_type = gst_nvds_osd_get_type ();
  GType nvvidconv_elem_type = gst_nvvideoconvert_get_type ();
  guint prop_id = PROP_OSD_LAST;
  /* All properties of the bin are same as osd element */
  forward_properties (gobject_class, osd_elem_type, &prop_id);
  static const gchar *nvvidconv_prop_map[][2] =
      { {"nvbuf-memory-type", "nvbuf-memory-type"} };
  forward_select_properties (gobject_class, nvvidconv_prop_map, 1,
      nvvidconv_elem_type, &prop_id, NULL);

  g_object_class_install_property (gobject_class, PROP_OSD_FONT,
      g_param_spec_string ("font",
          "Font", "Font to use for object labels", DEFAULT_FONT, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_TEXT_SIZE,
      g_param_spec_uint ("text-size",
          "Text Size", "Text size to set for object labels", 0, G_MAXUINT, 0,
          (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_TEXT_COLOR,
      g_param_spec_uint ("text-color",
          "Text Color",
          "Text color to set for object labels. RGBA hex format e.g. 0xff000044 for semi-transparent red",
          0, G_MAXUINT, 0, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_TEXT_BG_COLOR,
      g_param_spec_uint ("text-bg-color",
          "Text Background Color",
          "Text background color to set for object labels. RGBA hex format e.g. 0xff000044 for semi-transparent red",
          0, G_MAXUINT, 0, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_BBOX_BORDER_COLORS,
      g_param_spec_string ("bbox-border-colors",
          "Bounding Box Border Colors",
          "Border Colors for object bounding boxes. "
          "Format - <class-label1>=hex1;<class-label2>=hex2. hex is RGBA hex format",
          "", (GParamFlags)
          (G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_BBOX_BG_COLORS,
      g_param_spec_string ("bbox-bg-colors",
          "Bounding Box Background Colors",
          "Background Colors for object bounding boxes. "
          "Format - <class-label1>=hex1;<class-label2>=hex2. hex is RGBA hex format",
          "", (GParamFlags)
          (G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_BORDER_WIDTH,
      g_param_spec_uint ("border-width",
          "Border Width", "Border width to set for object bounding boxes", 0,
          G_MAXUINT, 0, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OSD_DISPLAY_TRACKING_ID,
      g_param_spec_boolean ("display-tracking-id",
          "Display Tracking ID",
          "Boolean to control display of tracking ids in object labels",
          DEFAULT_DISPLAY_TRACKING_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_OSD_REFORMAT_OBJECT_LABELS,
      g_param_spec_boolean ("reformat-object-labels", "Reformat object labels",
          "Reformat object labels with format <primary label> [tracking-id] "
          "<secondary-labels-sorted-by-component-id>",
          DEFAULT_REFORMAT_OBJECT_LABELS, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  forward_pad_template (gstelement_class, osd_elem_type, "src");
  forward_pad_template (gstelement_class, nvvidconv_elem_type, "sink");

  gst_ds_nvosd_bin_signals[SIGNAL_SET_DISPLAY_TEXT_ENABLED] =
      g_signal_new ("set-display-text-enabled",
      G_TYPE_FROM_CLASS (klass),
      (GSignalFlags) (G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION),
      G_STRUCT_OFFSET (GstDsNvOSDBinClass, set_display_text_enabled),
      NULL, NULL, NULL, G_TYPE_NONE, 1, G_TYPE_BOOLEAN);

  klass->set_display_text_enabled = gst_ds_nvosd_bin_set_display_text_enabled;

  gst_element_class_set_details_simple (gstelement_class,
      "NvOsd Bin",
      "NvOsd Bin",
      "Nvidia DeepStreamSDK NvOSD Bin. Internal Pipeline: queue->nvvidconv->queue->nvosd",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvosd_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvOSDBin *nvosdbin = GST_DS_NVOSD_BIN (object);

  if (prop_id < PROP_OSD_LAST) {
    nvosdbin->prop_set[prop_id] = TRUE;

    switch (prop_id) {
      case PROP_OSD_FONT:
        g_free (nvosdbin->font);
        nvosdbin->font = g_value_dup_string (value);
        break;
      case PROP_OSD_TEXT_SIZE:
        nvosdbin->text_size = g_value_get_uint (value);
        break;
      case PROP_OSD_TEXT_COLOR:
        parse_color_from_int (g_value_get_uint (value), nvosdbin->text_color);
        break;
      case PROP_OSD_TEXT_BG_COLOR:
        parse_color_from_int (g_value_get_uint (value),
            nvosdbin->text_bg_color);
        break;
      case PROP_OSD_BBOX_BORDER_COLORS:
        parse_colors_from_list (g_value_get_string (value),
            *nvosdbin->class_border_color_map);
        break;
      case PROP_OSD_BBOX_BG_COLORS:
        parse_colors_from_list (g_value_get_string (value),
            *nvosdbin->class_bg_color_map);
        break;
      case PROP_OSD_BORDER_WIDTH:
        nvosdbin->border_width = g_value_get_uint (value);
        break;
      case PROP_OSD_DISPLAY_TRACKING_ID:
        nvosdbin->display_tracking_id = g_value_get_boolean (value);
        break;
      case PROP_OSD_REFORMAT_OBJECT_LABELS:
        nvosdbin->reformat_object_labels = g_value_get_boolean (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
  } else {
    if (!g_strcmp0 (pspec->name, "gpu-id")) {
      g_object_set_property (G_OBJECT (nvosdbin->nvvidconv), pspec->name,
          value);
    }
    if (!g_strcmp0 (pspec->name, "nvbuf-memory-type"))
      g_object_set_property (G_OBJECT (nvosdbin->nvvidconv), pspec->name,
          value);
    else
      g_object_set_property (G_OBJECT (nvosdbin->nvosd), pspec->name, value);
  }
}

static void
gst_ds_nvosd_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvOSDBin *nvosdbin = GST_DS_NVOSD_BIN (object);

  if (prop_id < PROP_OSD_LAST) {
    switch (prop_id) {
      case PROP_OSD_FONT:
        g_value_set_string (value, nvosdbin->font);
        break;
      case PROP_OSD_TEXT_SIZE:
        g_value_set_uint (value, nvosdbin->text_size);
        break;
      case PROP_OSD_TEXT_COLOR:
        g_value_set_uint (value, color_to_int (nvosdbin->text_color));
        break;
      case PROP_OSD_TEXT_BG_COLOR:
        g_value_set_uint (value, color_to_int (nvosdbin->text_bg_color));
        break;
      case PROP_OSD_BORDER_WIDTH:
        g_value_set_uint (value, nvosdbin->border_width);
        break;
      case PROP_OSD_DISPLAY_TRACKING_ID:
        g_value_set_boolean (value, nvosdbin->display_tracking_id);
        break;
      case PROP_OSD_REFORMAT_OBJECT_LABELS:
        g_value_set_boolean (value, nvosdbin->reformat_object_labels);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
  } else {
    if (!g_strcmp0 (pspec->name, "nvbuf-memory-type"))
      g_object_get_property (G_OBJECT (nvosdbin->nvvidconv), pspec->name,
          value);
    else
      g_object_get_property (G_OBJECT (nvosdbin->nvosd), pspec->name, value);
  }
}


static gint
component_id_compare_func (gconstpointer a, gconstpointer b)
{
  NvDsClassifierMeta *cmetaa = (NvDsClassifierMeta *) a;
  NvDsClassifierMeta *cmetab = (NvDsClassifierMeta *) b;

  if (cmetaa->unique_component_id < cmetab->unique_component_id)
    return -1;
  if (cmetaa->unique_component_id > cmetab->unique_component_id)
    return 1;
  return 0;
}

static GstPadProbeReturn
osd_config_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstDsNvOSDBin *nvosdbin = GST_DS_NVOSD_BIN (u_data);
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;

      auto border_clr = nvosdbin->class_border_color_map->find (obj->obj_label);
      if (border_clr != nvosdbin->class_border_color_map->end ()) {
        obj->rect_params.border_color = border_clr->second;
      }

      auto bg_clr = nvosdbin->class_bg_color_map->find (obj->obj_label);
      if (bg_clr != nvosdbin->class_bg_color_map->end ()) {
        obj->rect_params.bg_color = bg_clr->second;
        obj->rect_params.has_bg_color = 1;
      }

      if (nvosdbin->prop_set[PROP_OSD_BORDER_WIDTH])
        obj->rect_params.border_width = nvosdbin->border_width;

      obj->text_params.x_offset = obj->rect_params.left;
      obj->text_params.y_offset = obj->rect_params.top - 30;

      if (nvosdbin->prop_set[PROP_OSD_TEXT_COLOR])
        obj->text_params.font_params.font_color = nvosdbin->text_color;

      if (nvosdbin->prop_set[PROP_OSD_TEXT_SIZE])
        obj->text_params.font_params.font_size = nvosdbin->text_size;

      if (nvosdbin->prop_set[PROP_OSD_FONT])
        obj->text_params.font_params.font_name = nvosdbin->font;

      if (nvosdbin->prop_set[PROP_OSD_TEXT_BG_COLOR]) {
        obj->text_params.set_bg_clr = 1;
        obj->text_params.text_bg_clr = nvosdbin->text_bg_color;
      }

      if (nvosdbin->reformat_object_labels) {
        g_free (obj->text_params.display_text);

        obj->text_params.display_text = g_new0 (char, 128);
        obj->text_params.display_text[0] = '\0';
        gchar *str_ins_pos = obj->text_params.display_text;

        if (obj->obj_label[0] != '\0')
          sprintf (str_ins_pos, "%s", obj->obj_label);
        str_ins_pos += strlen (str_ins_pos);

        if (obj->object_id != UNTRACKED_OBJECT_ID) {
        /** object_id is a 64-bit sequential value;
         * but considering the display aesthetic,
         * trimming to lower 32-bits */
          if (nvosdbin->display_tracking_id) {
            guint64 const LOW_32_MASK = 0x00000000FFFFFFFF;
            sprintf (str_ins_pos, " %lu", (obj->object_id & LOW_32_MASK));
            str_ins_pos += strlen (str_ins_pos);
          }
        }

        obj->classifier_meta_list =
            g_list_sort (obj->classifier_meta_list, component_id_compare_func);
        for (NvDsMetaList * l_class = obj->classifier_meta_list;
            l_class != NULL; l_class = l_class->next) {
          NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *) l_class->data;
          for (NvDsMetaList * l_label = cmeta->label_info_list; l_label != NULL;
              l_label = l_label->next) {
            NvDsLabelInfo *label = (NvDsLabelInfo *) l_label->data;
            if (label->pResult_label) {
              sprintf (str_ins_pos, " %s", label->pResult_label);
            } else if (label->result_label[0] != '\0') {
              sprintf (str_ins_pos, " %s", label->result_label);
            }
            str_ins_pos += strlen (str_ins_pos);
          }
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}


static void
gst_ds_nvosd_bin_init (GstDsNvOSDBin * nvosdbin)
{
  nvosdbin->nvvidconv =
      gst_element_factory_make ("nvvideoconvert", "nvosd_bin_nvvidconv");
  if (!nvosdbin->nvvidconv) {
    GST_ELEMENT_ERROR (nvosdbin, STREAM, FAILED,
        ("Failed to create 'nvvideoconvert'"), (NULL));
    return;
  }

  nvosdbin->queue = gst_element_factory_make ("queue", "nvosd_bin_queue");
  if (!nvosdbin->queue) {
    GST_ELEMENT_ERROR (nvosdbin, STREAM, FAILED, ("Failed to create 'queue'"),
        (NULL));
    return;
  }

  nvosdbin->conv_queue =
      gst_element_factory_make ("queue", "nvosd_bin_conv_queue");
  if (!nvosdbin->conv_queue) {
    GST_ELEMENT_ERROR (nvosdbin, STREAM, FAILED, ("Failed to create 'queue'"),
        (NULL));
    return;
  }

  nvosdbin->nvosd = gst_element_factory_make ("nvdsosd", "nvosd_bin_nvosd");
  if (!nvosdbin->nvosd) {
    GST_ELEMENT_ERROR (nvosdbin, STREAM, FAILED, ("Failed to create 'nvdsosd'"),
        (NULL));
    return;
  }

  gst_bin_add_many (GST_BIN (nvosdbin), nvosdbin->queue, nvosdbin->conv_queue,
      nvosdbin->nvvidconv, nvosdbin->nvosd, NULL);

  NVGSTDS_LINK_ELEMENT (nvosdbin->queue, nvosdbin->nvvidconv);
  NVGSTDS_LINK_ELEMENT (nvosdbin->nvvidconv, nvosdbin->conv_queue);
  NVGSTDS_LINK_ELEMENT (nvosdbin->conv_queue, nvosdbin->nvosd);
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvosdbin), nvosdbin->nvosd, "src");
  NVGSTDS_BIN_ADD_GHOST_PAD (GST_ELEMENT (nvosdbin), nvosdbin->queue, "sink");

  NVGSTDS_ELEM_ADD_PROBE (nvosdbin, nvosdbin->nvosd, "sink",
      osd_config_probe, GST_PAD_PROBE_TYPE_BUFFER, nvosdbin);

  nvosdbin->font = g_strdup (DEFAULT_FONT);
  nvosdbin->display_tracking_id = DEFAULT_DISPLAY_TRACKING_ID;
  nvosdbin->reformat_object_labels = DEFAULT_REFORMAT_OBJECT_LABELS;
  nvosdbin->class_border_color_map =
      new std::unordered_map < std::string, NvOSD_ColorParams > ();
  nvosdbin->class_bg_color_map =
      new std::unordered_map < std::string, NvOSD_ColorParams > ();
}

static void
gst_ds_nvosd_bin_set_display_text_enabled (GstDsNvOSDBin * nvosdbin,
    gboolean enabled)
{
  g_object_set (G_OBJECT (nvosdbin), "display-text", enabled, NULL);
}
