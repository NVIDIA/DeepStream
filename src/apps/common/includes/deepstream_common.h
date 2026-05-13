/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_COMMON_H__
#define __NVGSTDS_COMMON_H__

#include <gst/gst.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "deepstream_config.h"

/* Global error buffer for capturing last error message */
/* Using weak symbol so apps don't need to link deepstream_common.c */
__attribute__((weak)) gchar *g_nvds_last_error_message = NULL;

/* Helper function to capture and store error messages */
static inline void nvds_capture_error(const gchar *func, gint line, const gchar *msg, ...) {
  va_list args;
  gchar *formatted_msg;
  gchar *clean_msg;

  va_start(args, msg);
  formatted_msg = g_strdup_vprintf(msg, args);
  va_end(args);

  /* Print to console as before */
  g_print("** ERROR: <%s:%d>: %s\n", func, line, formatted_msg);

  /* Store in global buffer ONLY IF it's the first error (root cause) */
  if (!g_nvds_last_error_message) {
    /* Strip prefix before first colon (e.g. "sink_sub_bin_sink2: error" -> "error") */
    gchar *colon_pos = strchr(formatted_msg, ':');
    if (colon_pos) {
      /* Skip the colon and any leading whitespace */
      clean_msg = g_strstrip(g_strdup(colon_pos + 1));
      g_nvds_last_error_message = clean_msg;
    } else {
      /* No colon found, use the whole message */
      g_nvds_last_error_message = g_strdup(formatted_msg);
    }
  }

  g_free(formatted_msg);
}

#define NVGSTDS_ERR_MSG_V(msg, ...) \
    nvds_capture_error(__func__, __LINE__, msg, ##__VA_ARGS__)

#define NVGSTDS_INFO_MSG_V(msg, ...) \
    g_print("** INFO: <%s:%d>: " msg "\n", __func__, __LINE__, ##__VA_ARGS__)

#define NVGSTDS_WARN_MSG_V(msg, ...) \
    g_print("** WARN: <%s:%d>: " msg "\n", __func__, __LINE__, ##__VA_ARGS__)

#define NVGSTDS_LINK_ELEMENT(elem1, elem2) \
    do { \
      if (!gst_element_link (elem1,elem2)) { \
        GstCaps *src_caps, *sink_caps; \
        src_caps = gst_pad_query_caps ((GstPad *) (elem1)->srcpads->data, NULL); \
        sink_caps = gst_pad_query_caps ((GstPad *) (elem2)->sinkpads->data, NULL); \
        NVGSTDS_ERR_MSG_V ("Failed to link '%s' (%s) and '%s' (%s)", \
            GST_ELEMENT_NAME (elem1), \
            gst_caps_to_string (src_caps), \
            GST_ELEMENT_NAME (elem2), \
            gst_caps_to_string (sink_caps)); \
        goto done; \
      } \
    } while (0)

#define NVGSTDS_LINK_ELEMENT_FULL(elem1, elem1_pad_name, elem2, elem2_pad_name) \
  do { \
    GstPad *elem1_pad = gst_element_get_static_pad(elem1, elem1_pad_name); \
    GstPad *elem2_pad = gst_element_get_static_pad(elem2, elem2_pad_name); \
    GstPadLinkReturn ret = gst_pad_link (elem1_pad,elem2_pad); \
    if (ret != GST_PAD_LINK_OK) { \
      gchar *n1 = gst_pad_get_name (elem1_pad); \
      gchar *n2 = gst_pad_get_name (elem2_pad); \
      NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s': %d", \
          n1, n2, ret); \
      g_free (n1); \
      g_free (n2); \
      gst_object_unref (elem1_pad); \
      gst_object_unref (elem2_pad); \
      goto done; \
    } \
    gst_object_unref (elem1_pad); \
    gst_object_unref (elem2_pad); \
  } while (0)

#define NVGSTDS_BIN_ADD_GHOST_PAD_NAMED(bin, elem, pad, ghost_pad_name) \
  do { \
    GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
    if (!gstpad) { \
      NVGSTDS_ERR_MSG_V ("Could not find '%s' in '%s'", pad, \
          GST_ELEMENT_NAME(elem)); \
      goto done; \
    } \
    gst_element_add_pad (bin, gst_ghost_pad_new (ghost_pad_name, gstpad)); \
    gst_object_unref (gstpad); \
  } while (0)

#define NVGSTDS_BIN_ADD_GHOST_PAD(bin, elem, pad) \
      NVGSTDS_BIN_ADD_GHOST_PAD_NAMED (bin, elem, pad, pad)

#define NVGSTDS_ELEM_ADD_PROBE(probe_id, elem, pad, probe_func, probe_type, probe_data) \
    do { \
      GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
      if (!gstpad) { \
        NVGSTDS_ERR_MSG_V ("Could not find '%s' in '%s'", pad, \
            GST_ELEMENT_NAME(elem)); \
        goto done; \
      } \
      probe_id = gst_pad_add_probe(gstpad, (probe_type), probe_func, probe_data, NULL); \
      gst_object_unref (gstpad); \
    } while (0)

#define NVGSTDS_ELEM_REMOVE_PROBE(probe_id, elem, pad) \
    do { \
      if (probe_id == 0 || !elem) { \
          break; \
      } \
      GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
      if (!gstpad) { \
        NVGSTDS_ERR_MSG_V ("Could not find '%s' in '%s'", pad, \
            GST_ELEMENT_NAME(elem)); \
        break; \
      } \
      gst_pad_remove_probe(gstpad, probe_id); \
      gst_object_unref (gstpad); \
    } while (0)

#define GET_FILE_PATH(path) ((path) + (((path) && strstr ((path), "file://")) ? 7 : 0))


/**
 * Function to link sink pad of an element to source pad of tee.
 *
 * @param[in] tee Tee element.
 * @param[in] sinkelem downstream element.
 *
 * @return true if link successful.
 */
gboolean
link_element_to_tee_src_pad (GstElement * tee, GstElement * sinkelem);

/**
 * Function to link source pad of an element to sink pad of muxer element.
 *
 * @param[in] streammux muxer element.
 * @param[in] elem upstream element.
 * @param[in] index pad index of muxer element.
 *
 * @return true if link successful.
 */
gboolean
link_element_to_streammux_sink_pad (GstElement *streammux, GstElement *elem,
                                    gint index);

/**
 * Function to unlink source pad of an element from sink pad of muxer element.
 *
 * @param[in] streammux muxer element.
 * @param[in] elem upstream element.
 *
 * @return true if unlinking was successful.
 */
gboolean
unlink_element_from_streammux_sink_pad (GstElement *streammux, GstElement *elem);

/**
 * Function to link sink pad of an element to source pad of demux element.
 *
 * @param[in] demux demuxer element.
 * @param[in] elem downstream element.
 * @param[in] index pad index of demuxer element.
 *
 * @return true if link successful.
 */
gboolean
link_element_to_demux_src_pad (GstElement *demux, GstElement *elem,
                               guint index);

/*
 * Function to replace string with another string.
 * Make sure @ref src is big enough to accommodate replacements.
 *
 * @param[in] str string to search in.
 * @param[in] replace string to replace.
 * @param[in] replace_with string to replace @ref replace with.
 */
void
str_replace (gchar * str, const gchar * replace, const gchar * replace_with);

#ifdef __cplusplus
}
#endif

#endif
