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

#ifndef __GST_DS_DEEPSTREAM_BINS_UTILS_H__
#define __GST_DS_DEEPSTREAM_BINS_UTILS_H__

#include <gst/gst.h>
#include <memory>

/* Forward only select properties from child type to parent element class */
static inline void
forward_select_properties (GObjectClass * parent_elem_class,
    const gchar * (*property_name_map)[2], guint map_size,
    GType child_elem_type, guint * prop_index,
    GPtrArray * orig_prop_names_ordered)
{
  GObject *child_obj = G_OBJECT (g_type_create_instance (child_elem_type));
  GObjectClass *child_class = G_OBJECT_GET_CLASS (child_obj);
  if (orig_prop_names_ordered)
    g_ptr_array_set_size (orig_prop_names_ordered, *prop_index + map_size);

  for (guint i = 0; i < map_size; i++) {
    GParamSpec *spec =
        g_object_class_find_property (child_class, property_name_map[i][0]);
    if (!spec) {
      g_printerr ("Property '%s' not found on element '%s'\n",
          property_name_map[i][0], G_OBJECT_CLASS_NAME (child_class));
      continue;
    }
    GParamSpec *binspec = NULL;
    switch (spec->value_type) {
      case G_TYPE_STRING:
      {
        gchar *default_val = NULL;
        g_object_get (child_obj, property_name_map[i][0], &default_val, NULL);
        binspec =
            g_param_spec_string (property_name_map[i][1], spec->_nick,
            spec->_blurb, default_val, spec->flags);
        g_free (default_val);
      }
        break;
      case G_TYPE_BOOLEAN:
      {
        gboolean default_val = FALSE;
        g_object_get (child_obj, property_name_map[i][0], &default_val, NULL);
        binspec =
            g_param_spec_boolean (property_name_map[i][1], spec->_nick,
            spec->_blurb, default_val, spec->flags);
      }
        break;
      case G_TYPE_UINT:
      {
        GParamSpecUInt *puint = G_PARAM_SPEC_UINT (spec);
        guint default_val = 0;
        g_object_get (child_obj, property_name_map[i][0], &default_val, NULL);

        binspec =
            g_param_spec_uint (property_name_map[i][1], spec->_nick,
            spec->_blurb, puint->minimum, puint->maximum,
            default_val, spec->flags);
      }
        break;
      case G_TYPE_INT:
      {
        GParamSpecInt *pint = G_PARAM_SPEC_INT (spec);
        gint default_val = 0;
        g_object_get (child_obj, property_name_map[i][0], &default_val, NULL);
        binspec =
            g_param_spec_int (property_name_map[i][1], spec->_nick,
            spec->_blurb, pint->minimum, pint->maximum,
            default_val, spec->flags);
      }
        break;
      case G_TYPE_POINTER:
      {
        binspec =
            g_param_spec_pointer (spec->name, spec->_nick,
            spec->_blurb, spec->flags);
      }
        break;
      default:
        if (G_IS_PARAM_SPEC_ENUM (spec)) {
          gint default_val;
          g_object_get (child_obj, property_name_map[i][0], &default_val, NULL);
          binspec = g_param_spec_enum (property_name_map[i][1], spec->_nick,
              spec->_blurb, spec->value_type, default_val, spec->flags);
        }
        break;
    }
    if (binspec)
      g_object_class_install_property (parent_elem_class, *prop_index, binspec);
    else {
      g_printerr ("Cannot forward '%s' property '%s' of type '%s'\n",
          G_OBJECT_CLASS_NAME (child_class), property_name_map[i][0],
          g_type_name (spec->value_type));
      continue;
    }
    if (orig_prop_names_ordered)
      g_ptr_array_insert (orig_prop_names_ordered, *prop_index,
          (gpointer) property_name_map[i][0]);
    (*prop_index)++;
  }
  g_object_unref (child_obj);
}

/* Forward all properties of child element type to parent element class */
static inline void
forward_properties (GObjectClass * parent_elem_class, GType child_elem_type,
    guint * prop_id = NULL)
{
  GObject *child_obj = G_OBJECT (g_type_create_instance (child_elem_type));
  GObjectClass *child_class = G_OBJECT_GET_CLASS (child_obj);
  guint param_id = prop_id ? *prop_id : 1;

  guint num_specs = 0;
  GParamSpec **spec = g_object_class_list_properties (child_class, &num_specs);
  for (uint prop_index = 0; prop_index < num_specs; ++prop_index) {
    GParamSpec *cur_spec = spec[prop_index];
    GParamSpec *binspec = NULL;
    //parent and name prop are a part of GstBin element and need not be forwarded
    if (!g_strcmp0 (cur_spec->name, "parent")
        || !g_strcmp0 (cur_spec->name, "name"))
      continue;
    switch (cur_spec->value_type) {
      case G_TYPE_STRING:
      {
        gchar *default_val = NULL;
        g_object_get (child_obj, cur_spec->name, &default_val, NULL);
        binspec =
            g_param_spec_string (cur_spec->name, cur_spec->_nick,
            cur_spec->_blurb, default_val, cur_spec->flags);
        g_free (default_val);
      }
        break;
      case G_TYPE_BOOLEAN:
      {
        gboolean default_val = FALSE;
        g_object_get (child_obj, cur_spec->name, &default_val, NULL);
        binspec =
            g_param_spec_boolean (cur_spec->name, cur_spec->_nick,
            cur_spec->_blurb, default_val, cur_spec->flags);
      }
        break;
      case G_TYPE_UINT:
      {
        GParamSpecUInt *puint = G_PARAM_SPEC_UINT (cur_spec);
        guint default_val = 0;
        g_object_get (child_obj, cur_spec->name, &default_val, NULL);
        binspec =
            g_param_spec_uint (cur_spec->name, cur_spec->_nick,
            cur_spec->_blurb, puint->minimum, puint->maximum,
            default_val, cur_spec->flags);
      }
        break;
      case G_TYPE_INT:
      {
        GParamSpecInt *pint = G_PARAM_SPEC_INT (cur_spec);
        gint default_val = 0;
        g_object_get (child_obj, cur_spec->name, &default_val, NULL);
        binspec =
            g_param_spec_int (cur_spec->name, cur_spec->_nick,
            cur_spec->_blurb, pint->minimum, pint->maximum,
            default_val, cur_spec->flags);
      }
        break;
      case G_TYPE_POINTER:
      {
        binspec =
            g_param_spec_pointer (cur_spec->name, cur_spec->_nick,
            cur_spec->_blurb, cur_spec->flags);
      }
        break;
      default:
        if (G_IS_PARAM_SPEC_ENUM (cur_spec)) {
          gint default_val;
          g_object_get (child_obj, cur_spec->name, &default_val, NULL);
          binspec = g_param_spec_enum (cur_spec->name, cur_spec->_nick,
              cur_spec->_blurb, cur_spec->value_type, default_val,
              cur_spec->flags);
        }
        break;
    }
    if (binspec)
      g_object_class_install_property (parent_elem_class, param_id, binspec);
    else {
      g_printerr ("Cannot forward '%s' property '%s' of type '%s'\n",
          G_OBJECT_CLASS_NAME (child_class), cur_spec->name,
          g_type_name (cur_spec->value_type));
      continue;
    }
    param_id++;
  }
  g_free (spec);
  g_object_unref (child_obj);
  if (prop_id)
    *prop_id = param_id;
}

static inline void
forward_pad_template (GstElementClass * bin_class, GType child_elem_type,
    const gchar * child_tmpl_name, const gchar * pad_tmpl_name = nullptr)
{
  GstElementClass *child_elem_class =
      (GstElementClass *) g_type_class_ref (child_elem_type);
  GstPadTemplate *tmpl =
      gst_element_class_get_pad_template (child_elem_class, child_tmpl_name);
  tmpl = gst_pad_template_new (pad_tmpl_name ? pad_tmpl_name : child_tmpl_name,
      tmpl->direction, tmpl->presence, tmpl->caps);
  gst_element_class_add_pad_template (bin_class, tmpl);
  g_type_class_unref (child_elem_class);
}

template <class T>
class GstObjectUPtr:public std::unique_ptr<T, void (*)(T *)>
{
public:
  GstObjectUPtr (T * t = nullptr): std::unique_ptr<T, void (*)(T *)>(t, (void (*)(T *)) gst_object_unref) {}
  operator T* () const { return this->get (); }
};

class GstCapsUPtr:public std::unique_ptr <GstCaps, void (*)(GstCaps *)>
{
public:
  GstCapsUPtr (GstCaps * t = nullptr): std::unique_ptr < GstCaps, void (*)(GstCaps *) > (t, gst_caps_unref) {}
  operator GstCaps* () const { return this->get (); }
};

using GstPadUPtr = GstObjectUPtr <GstPad>;

#define NVGSTDS_ELEM_ADD_PROBE(parent_elem, elem, pad, probe_func, probe_type, probe_data) \
    ({ \
      gulong probe_id = 0; \
      GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
      if (!gstpad) { \
        GST_ELEMENT_ERROR(parent_elem, RESOURCE, FAILED, \
            ("Could not find '%s' in '%s'", pad, \
            GST_ELEMENT_NAME(elem)), (NULL)); \
      } else { \
        probe_id = gst_pad_add_probe(gstpad, (GstPadProbeType) (probe_type), probe_func, probe_data, NULL); \
        gst_object_unref (gstpad); \
      } \
      probe_id; \
    })

#define NVGSTDS_LINK_ELEMENT(elem1, elem2, ...) \
    ({ \
      if (!gst_element_link (elem1,elem2)) { \
        GstCaps * src_caps, *sink_caps; \
        src_caps = gst_pad_query_caps ((GstPad *) (elem1)->srcpads->data, NULL); \
        sink_caps = gst_pad_query_caps ((GstPad *) (elem2)->sinkpads->data, NULL); \
        GST_ELEMENT_ERROR (GST_ELEMENT_PARENT(elem1), STREAM, FAILED, \
            ("Failed to link '%s' (%s) and '%s' (%s)", \
                GST_ELEMENT_NAME (elem1), \
                gst_caps_to_string (src_caps), \
                GST_ELEMENT_NAME (elem2), \
                gst_caps_to_string (sink_caps)), (NULL)); \
        return __VA_ARGS__; \
      } \
    })

#define NVGSTDS_BIN_ADD_GHOST_PAD(bin, elem, pad, ...) \
  do { \
    GstPadUPtr gstpad = gst_element_get_static_pad (elem, pad); \
    if (!gstpad) { \
      GST_ELEMENT_ERROR(bin, STREAM, FAILED, ("Could not find '%s' in '%s'", pad, \
          GST_ELEMENT_NAME(elem)), (NULL)); \
      return __VA_ARGS__; \
    } \
    GstElementClass *klass = GST_ELEMENT_GET_CLASS(bin); \
    GstPadTemplate *tmpl = gst_element_class_get_pad_template(klass, pad); \
    gst_element_add_pad (bin, gst_ghost_pad_new_from_template (pad, gstpad, tmpl)); \
  } while (0)

#define NVGSTDS_BIN_SET_GHOST_PAD_TARGET(bin, bin_pad, target, target_pad_name, ...) \
  do { \
    GstPadUPtr pad = gst_element_get_static_pad (target, target_pad_name); \
    if (!pad) { \
      GST_ELEMENT_ERROR(bin, STREAM, FAILED, ("Could not find '%s' in '%s'", target_pad_name, \
          GST_ELEMENT_NAME(target)), (NULL)); \
      return __VA_ARGS__; \
    } \
    if (!gst_ghost_pad_set_target (GST_GHOST_PAD \
            (bin_pad), pad)) { \
      GST_ELEMENT_ERROR (bin, RESOURCE, NOT_FOUND, \
          ("Failed to set '%s' as target of '%s' pad", \
              GST_ELEMENT_NAME (target), \
              GST_PAD_NAME (bin_pad)), (NULL)); \
      return __VA_ARGS__; \
    } \
  } while (0)

static inline void
remove_all_children (GstBin * bin)
{
  GstIterator *it = gst_bin_iterate_elements (bin);
  GValue elem = G_VALUE_INIT;
  while (gst_iterator_next (it, &elem) == GST_ITERATOR_OK) {
    gst_bin_remove (bin, GST_ELEMENT (g_value_get_object (&elem)));
    gst_iterator_resync (it);
  }
  gst_iterator_free (it);
}

#endif /* __GST_DS_DEEPSTREAM_BINS_UTILS_H__ */
