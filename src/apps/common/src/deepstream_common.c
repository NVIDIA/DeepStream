/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <string.h>
#include "deepstream_common.h"

gboolean
link_element_to_tee_src_pad (GstElement * tee, GstElement * sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)
      gst_element_class_get_pad_template (GST_ELEMENT_GET_CLASS (tee),
      "src_%u");
  tee_src_pad = gst_element_request_pad (tee, padtemplate, NULL, NULL);
  if (!tee_src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad (sinkelem, "sink");
  if (!sinkpad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from '%s'",
        GST_ELEMENT_NAME (sinkelem));
    goto done;
  }

  if (gst_pad_link (tee_src_pad, sinkpad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (tee),
        GST_ELEMENT_NAME (sinkelem));
    goto done;
  }

  ret = TRUE;

done:
  if (tee_src_pad) {
    gst_object_unref (tee_src_pad);
  }
  if (sinkpad) {
    gst_object_unref (sinkpad);
  }
  return ret;
}

gboolean
link_element_to_streammux_sink_pad (GstElement * streammux, GstElement * elem,
    gint index)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;
  gchar pad_name[16];

  if (index >= 0) {
    g_snprintf (pad_name, 16, "sink_%u", index);
    pad_name[15] = '\0';
  } else {
    strcpy (pad_name, "sink_%u");
  }

  mux_sink_pad = gst_element_request_pad_simple (streammux, pad_name);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from streammux");
    goto done;
  }

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  if (gst_pad_link (src_pad, mux_sink_pad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'",
        GST_ELEMENT_NAME (streammux), GST_ELEMENT_NAME (elem));
    goto done;
  }

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

gboolean
unlink_element_from_streammux_sink_pad (GstElement * streammux,
    GstElement * elem)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  mux_sink_pad = gst_pad_get_peer (src_pad);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from streammux");
    goto done;
  }

  if (!gst_pad_unlink (src_pad, mux_sink_pad)) {
    NVGSTDS_ERR_MSG_V ("Failed to unlink '%s' and '%s'",
        GST_ELEMENT_NAME (streammux), GST_ELEMENT_NAME (elem));
    goto done;
  }

  gst_element_release_request_pad (streammux, mux_sink_pad);

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

gboolean
link_element_to_demux_src_pad (GstElement * demux, GstElement * elem,
    guint index)
{
  gboolean ret = FALSE;
  GstPad *demux_src_pad = NULL;
  GstPad *sink_pad = NULL;
  gchar pad_name[16];

  g_snprintf (pad_name, 16, "src_%u", index);
  pad_name[15] = '\0';

  demux_src_pad = gst_element_request_pad_simple (demux, pad_name);
  if (!demux_src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from demux");
    goto done;
  }

  sink_pad = gst_element_get_static_pad (elem, "sink");
  if (!sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  if (gst_pad_link (demux_src_pad, sink_pad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (demux),
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  ret = TRUE;

done:
  if (demux_src_pad) {
    gst_object_unref (demux_src_pad);
  }
  if (sink_pad) {
    gst_object_unref (sink_pad);
  }
  return ret;
}

void
str_replace (gchar * str, const gchar * replace, const gchar * replace_with)
{
  gchar tmp[1024];
  gchar *ins = tmp;
  ins[0] = '\0';
  gchar *str_orig = str;
  gchar *iter;

  if (!str || !replace || !replace_with) {
    return;
  }
  gint replace_len = strlen (replace);
  gint replace_with_len = strlen (replace_with);

  while ((iter = strstr (str, replace))) {
    gint num_char = iter - str;

    strncpy (ins, str, num_char);
    ins += num_char;

    strcpy (ins, replace_with);
    ins += replace_with_len;

    str = iter + replace_len;
  }
  strcpy (ins, str);
  strcpy (str_orig, tmp);
}
