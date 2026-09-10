/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvc_helper.h"

/* Use the main plugin's debug category */
#define GST_CAT_DEFAULT gst_nvvideoconvert_debug

/**
 * find_peer_element:
 * @pad: The pad to find the peer element for
 * @caps_filter: Pointer to store a capsfilter element if found
 *
 * Finds the peer element connected to the given pad, recursively looking
 * through capsfilters.
 *
 * Returns: The peer element if found, NULL otherwise
 */
GstElement*
find_peer_element(GstPad *pad, GstElement **caps_filter)
{
    if (!pad) {
        return NULL;
    }

    // Get the peer pad
    GstPad *peer_pad = gst_pad_get_peer(pad);
    if (!peer_pad) {
        GST_DEBUG("No peer pad found");
        return NULL;
    }

    // Get the parent element of the peer pad
    GstElement *peer_element = gst_pad_get_parent_element(peer_pad);
    gst_object_unref(peer_pad);

    if (!peer_element) {
        GST_DEBUG("No peer element found");
        return NULL;
    }

    // Check if the peer element is a capsfilter
    gchar *element_name = gst_element_get_name(peer_element);
    GST_DEBUG("Upstream element name: %s", element_name);
    g_free(element_name);

    if (GST_IS_BIN(peer_element)) {
        // If it's a bin, iterate over its children (optional)
        GST_DEBUG("Upstream element is a bin; further traversal may be needed.");
    }

    GstElementFactory *factory = gst_element_get_factory(peer_element);
    if (factory &&
        g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)), "capsfilter") == 0) {
        GST_DEBUG("Upstream element is a capsfilter.");
        if (caps_filter) {
            *caps_filter = peer_element;
        }
        GstElement *tmp_caps_filter = NULL;
        // Find the source pad of this capsfilter
        GstPad *capsfilter_sink_pad = gst_element_get_static_pad(peer_element, "sink");
        GstElement *next_peer = find_peer_element(capsfilter_sink_pad, &tmp_caps_filter);
        gst_object_unref(capsfilter_sink_pad);
        gst_object_unref(peer_element);
        return next_peer;
    }

    return peer_element;
}

/**
 * check_capsfilter_has_format_field:
 * @capsfilter: The capsfilter element to check
 * @format: Pointer to a string pointer that will be set to the format string if found
 *
 * Checks if the capsfilter has a format field specified and returns the format.
 * The caller is responsible for freeing the returned format string with g_free().
 *
 * Returns: TRUE if format field is found, FALSE otherwise
 */
gboolean
check_capsfilter_has_format_field(GstElement *capsfilter, gchar **format)
{
    GstCaps *caps = NULL;
    gboolean has_format_field = FALSE;

    // Retrieve the caps property from the capsfilter
    g_object_get(capsfilter, "caps", &caps, NULL);

    if (caps) {
        // Loop through each structure in the caps
        for (guint i = 0; i < gst_caps_get_size(caps); i++) {
            GstStructure *structure = gst_caps_get_structure(caps, i);
            const gchar *format_mentioned = gst_structure_get_string(structure, "format");
            if (format_mentioned) {
              GST_DEBUG_OBJECT(capsfilter, "Format: %s", format_mentioned);
              has_format_field = TRUE;
              *format = g_strdup(format_mentioned);
            }
        }
        gst_caps_unref(caps);
    }

    return has_format_field;
}

/**
 * remove_format_from_caps:
 * @btrans: The GstBaseTransform element
 * @caps: The caps to modify
 * @format_to_remove: The format string to remove
 *
 * Creates a new caps structure with the specified format removed.
 *
 * Returns: A new caps structure with the format removed
 */
GstCaps *
remove_format_from_caps(GstBaseTransform *btrans, GstCaps *caps, const gchar *format_to_remove) 
{
    GstCaps *filtered_caps = gst_caps_new_empty();
    guint size = gst_caps_get_size(caps);

    for (guint i = 0; i < size; i++) {
        const GstStructure *structure = gst_caps_get_structure(caps, i);
        GstCapsFeatures *features = gst_caps_get_features(caps, i);
        GstStructure *new_structure = gst_structure_copy(structure);

        // Check if the "format" field exists
        if (gst_structure_has_field(new_structure, "format")) {
            const GValue *formats = gst_structure_get_value(new_structure, "format");

            // Case 1: If the format is an array
            if (GST_VALUE_HOLDS_ARRAY(formats)) {
                GValue new_formats = G_VALUE_INIT;
                g_value_init(&new_formats, GST_TYPE_ARRAY);

                for (guint j = 0; j < gst_value_array_get_size(formats); j++) {
                    const GValue *format_value = gst_value_array_get_value(formats, j);
                    const gchar *format_str = g_value_get_string(format_value);

                    // Add formats except the one to remove
                    if (!g_str_equal(format_str, format_to_remove)) {
                        gst_value_array_append_value(&new_formats, format_value);
                        GST_LOG_OBJECT(btrans, "Added format: %s", format_str);
                    } else {
                        GST_INFO_OBJECT(btrans, "Removed format: %s", format_str);
                    }
                }

                // Replace the "format" field with the filtered array
                gst_structure_set_value(new_structure, "format", &new_formats);
                g_value_unset(&new_formats);
            }
            // Case 2: If the format is a list
            else if (GST_VALUE_HOLDS_LIST(formats)) {
                GValue new_formats = G_VALUE_INIT;
                g_value_init(&new_formats, GST_TYPE_LIST);

                for (guint j = 0; j < gst_value_list_get_size(formats); j++) {
                    const GValue *format_value = gst_value_list_get_value(formats, j);
                    const gchar *format_str = g_value_get_string(format_value);

                    // Add formats except the one to remove
                    if (!g_str_equal(format_str, format_to_remove)) {
                        gst_value_list_append_value(&new_formats, format_value);
                        GST_LOG_OBJECT(btrans, "Added format: %s", format_str);
                    } else {
                        GST_INFO_OBJECT(btrans, "Removed format: %s", format_str);
                    }
                }

                // Replace the "format" field with the filtered list
                gst_structure_set_value(new_structure, "format", &new_formats);
                g_value_unset(&new_formats);
            }
            // Case 3: If the format is a single value
            else if (G_VALUE_HOLDS_STRING(formats)) {
                const gchar *format_str = g_value_get_string(formats);

                // Remove the structure entirely if it matches the format to remove
                if (g_str_equal(format_str, format_to_remove)) {
                    GST_INFO_OBJECT(btrans, "Removed structure with format: %s", format_str);
                    gst_structure_free(new_structure);
                    continue;
                }
            }
        }

        // Add the modified structure to the new caps
        gst_caps_append_structure_full(filtered_caps, new_structure, gst_caps_features_copy(features));
    }

    return filtered_caps;
}

/**
 * is_format_bgra64_or_uyvp:
 * @format: The format string to check
 *
 * Checks if the format is either BGRA64_LE or UYVP.
 *
 * Returns: TRUE if the format is BGRA64_LE or UYVP, FALSE otherwise
 */
gboolean
is_format_bgra64_or_uyvp(const gchar *format)
{
    if (format == NULL) {
        GST_DEBUG("Format is NULL");
        return FALSE;
    }

    return (g_str_equal(format, "BGRA64_LE") || g_str_equal(format, "UYVP"));
}

/**
 * is_format_bgra64:
 * @format: The format string to check
 *
 * Checks if the format is BGRA64_LE.
 *
 * Returns: TRUE if the format is BGRA64_LE, FALSE otherwise
 */
gboolean
is_format_bgra64(const gchar *format)
{
    if (format == NULL) {
        GST_DEBUG("Format is NULL");
        return FALSE;
    }

    return (g_str_equal(format, "BGRA64_LE"));
}

/**
 * get_format_from_src_pad_caps:
 * @btrans: The GstBaseTransform element
 *
 * Retrieves the format from the source pad capsfilter.
 *
 * Returns: The format string if found, NULL otherwise
 */
gchar *
get_format_from_src_pad_capsfilter(GstBaseTransform *btrans)
{
    GstElement *caps_filter = NULL;
    gchar *format = NULL;
    GstElement *peer_element = find_peer_element(btrans->srcpad, &caps_filter);
    if (peer_element) {
      gst_object_unref(peer_element);
    }

    if (caps_filter) {
        check_capsfilter_has_format_field(caps_filter, &format);
    }

    return format;
}

/**
 * inspect_caps:
 * @btrans: The GstBaseTransform element
 * @direction: The pad direction (sink or src)
 * @caps: The caps to inspect
 *
 * Inspects the caps and modifies them based on the presence of BGRA64_LE format.
 *
 * Returns: Modified caps
 */
GstCaps *
inspect_caps(GstBaseTransform *btrans, GstPadDirection direction, GstCaps *caps)
{
    /* Keep BGRA64_LE in caps only if:
       1. caps query received on sink pad and:
          - there is 'format' field mentioned in upstream capsfilter or
          - there is nvvideoconvert as the direct upstream element. capsfilter may or may not be present.
       2. caps query received on src pad and:
          - format specified is BGRA64_LE or UYVP on capsfilter connected to the sinkpad or
          - format specified is BGRA64_LE on capsfilter connected to the srcpad */

    GstElement *caps_filter = NULL;
    gboolean upstream_nvvideoconvert = FALSE;
    gboolean format_field_specified = FALSE;
    gchar *sink_format = NULL, *src_format = NULL;

    GstElement *peer_element = find_peer_element(btrans->sinkpad, &caps_filter);
    if (!peer_element) {
        GST_DEBUG_OBJECT(btrans, "No peer element found");
        return caps;
    }

    if (caps_filter) {
        format_field_specified = check_capsfilter_has_format_field(caps_filter, &sink_format);
    }

    if (peer_element) {
        GstElementFactory *factory = gst_element_get_factory(peer_element);
        upstream_nvvideoconvert = (factory && g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)), "nvvideoconvert") == 0);
        gst_object_unref(peer_element);
    }

    GST_DEBUG_OBJECT(btrans, "caps_filter: %p, format_field_specified: %d, upstream_nvvideoconvert: %d",
                   caps_filter, format_field_specified, upstream_nvvideoconvert);

    if (direction == GST_PAD_SINK) {
        if ((caps_filter && format_field_specified) || upstream_nvvideoconvert) {
            goto end;
        } else {
            GST_INFO_OBJECT(btrans, "Removing BGRA64_LE from sink caps");
            caps = remove_format_from_caps(btrans, caps, "BGRA64_LE");
        }
    } else if (direction == GST_PAD_SRC) {
        src_format = get_format_from_src_pad_capsfilter(btrans);
        if (is_format_bgra64_or_uyvp(sink_format) || is_format_bgra64(src_format) || (!src_format)) {
            goto end;
        } else {
            GST_INFO_OBJECT(btrans, "Removing BGRA64_LE from src caps");
            caps = remove_format_from_caps(btrans, caps, "BGRA64_LE");
        }
    }

    end:
    if (sink_format) {
        g_free(sink_format);
    }
    if (src_format) {
        g_free(src_format);
    }

    return caps;
}