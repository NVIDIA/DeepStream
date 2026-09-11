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

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_TWOWHEELER 1
#define PGIE_CLASS_ID_PERSON 2
#define PGIE_CLASS_ID_ROADSIGN 3

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

// TODO : Check various scenarios
// 1) test stream with NO objects are present
// 2) test stream having few frames with objects and few frames without objects
// 3) test stream with objects are present

#define FPS_PRINT_INTERVAL 300

gint frame_number = 0;

struct ObjectInfo
{
  gchar obj_label[MAX_LABEL_SIZE];
  float left;
  float top;
  float right;
  float bottom;
  float confidence;

    ObjectInfo (gchar arg_obj_label[MAX_LABEL_SIZE], float arg_left,
      float arg_top, float arg_right, float arg_bottom, float arg_confidence)
  {
    snprintf (obj_label, MAX_LABEL_SIZE, "%s", arg_obj_label);
    left = arg_left;
    top = arg_top;
    right = arg_right;
    bottom = arg_bottom;
    confidence = arg_confidence;
  }
};

struct DSMetaInfo
{
  guint ctx_index;
  guint stream_id;
  gint frame_num;
  std::vector < ObjectInfo > vecobj_info;

    DSMetaInfo (guint arg_ctx_index, guint arg_stream_id, gint arg_frame_num,
      std::vector < ObjectInfo > arg_vecobj_info):ctx_index (arg_ctx_index),
      stream_id (arg_stream_id), frame_num (arg_frame_num),
      vecobj_info (arg_vecobj_info)
  {
  }
};

std::vector < std::vector < DSMetaInfo >> vecDSMetadata;

/* tiler_sink_pad_buffer_probe  will extract metadata received on tiler sink pad
 * and update params for drawing rectangle, object information etc. */
static
    GstPadProbeReturn
tiler_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer * buf = (GstBuffer *) info->data;

  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  NvDsDisplayMeta * display_meta = NULL;
  NvDsObjectMeta * object_meta = NULL;

  NvDsBatchMeta * batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    guint num_rects = 0;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint roadsign_count = 0;
    guint twowheeler_count = 0;

    int offset, offset1, offset2 = 0;

    NvDsFrameMeta * frame_meta = (NvDsFrameMeta *) (l_frame->data);

    if (vecDSMetadata[frame_meta->pad_index][frame_meta->frame_num].frame_num !=
        frame_meta->frame_num) {
      g_print ("video frame_num %u does not match with kitti data frame number %u\n",
          frame_meta->frame_num,
          vecDSMetadata[frame_meta->pad_index][frame_meta->frame_num].frame_num);
      return GST_PAD_PROBE_OK;
    }
    /* attach object meta to frame meta */
    std::vector < ObjectInfo > vecobj_info =
        vecDSMetadata[frame_meta->pad_index][frame_meta->frame_num].vecobj_info;
    guint num_obj = vecobj_info.size ();

    for (guint i = 0; i < num_obj; i++) {
      object_meta = nvds_acquire_obj_meta_from_pool (batch_meta);

      snprintf (object_meta->obj_label, MAX_LABEL_SIZE, "%s",
          vecobj_info[i].obj_label);

      object_meta->confidence = vecobj_info[i].confidence;

      NvOSD_RectParams & rect_params = object_meta->rect_params;

      /* Assign bounding box coordinates */
      rect_params.left = vecobj_info[i].left;
      rect_params.top = vecobj_info[i].top;
      rect_params.width = vecobj_info[i].right - vecobj_info[i].left;
      rect_params.height = vecobj_info[i].bottom - vecobj_info[i].top;

      /* Semi-transparent yellow background */
      rect_params.has_bg_color = 0;
      rect_params.bg_color = (NvOSD_ColorParams) {
      1, 1, 0, 0.4};
      /* Red border of width 6 */
      rect_params.border_width = 3;
      rect_params.border_color = (NvOSD_ColorParams) {
      1, 0, 0, 1};

      nvds_add_obj_meta_to_frame (frame_meta, object_meta, NULL);
      frame_meta->bInferDone = TRUE;
    }

    /* count the number of objects in a frame */
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      object_meta = (NvDsObjectMeta *) (l_obj->data);
      if (!g_strcmp0 (object_meta->obj_label, "Car")) {
        vehicle_count++;
        num_rects++;
      }
      else if (!g_strcmp0 (object_meta->obj_label, "Person")) {
        person_count++;
        num_rects++;
      }
      else if (!g_strcmp0 (object_meta->obj_label, "TwoWheeler")) {
        twowheeler_count++;
        num_rects++;
      }
      else if (!g_strcmp0 (object_meta->obj_label, "Roadsign")) {
        roadsign_count++;
        num_rects++;
      }
    }

    /* attach display meta to show object Counts */
#if 1
    static gchar font_name[] = "Serif";

    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    display_meta->num_labels = 1;
    NvOSD_TextParams *
        txt_params = &display_meta->text_params[0];
    txt_params->display_text = (char *) g_malloc0 (MAX_DISPLAY_LEN);
    offset =
        snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "Persons = %d ",
        person_count);
    offset1 =
        snprintf (txt_params->display_text + offset, MAX_DISPLAY_LEN,
        "Vehicles = %d ", vehicle_count);
    offset2 =
        snprintf (txt_params->display_text + offset + offset1, MAX_DISPLAY_LEN,
        "TwoWheeler = %d ", twowheeler_count);
    offset=
        snprintf (txt_params->display_text + offset + offset1 + offset2, MAX_DISPLAY_LEN,
        "Roadsign = %d ", roadsign_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = font_name;
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame (frame_meta, display_meta);
#endif
  }
  return GST_PAD_PROBE_OK;
}

/* appsink_sample is an appsink callback that will extract metadata received
 * queue6 sink pad and can update params for drawing rectangle,
 *object information etc. */
static GstFlowReturn
appsink_sample (GstElement * sink, gpointer * data)
{
  GstSample *sample;
  GstBuffer *buf = NULL;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  guint twowheeler_count = 0;
  guint roadsign_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;

  sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
  if (gst_app_sink_is_eos (GST_APP_SINK (sink))) {
    g_print ("EOS received in Appsink********\n");
  }

  if (sample) {
    /* Obtain GstBuffer from sample and then extract metadata from it. */
    buf = gst_sample_get_buffer (sample);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);
        if (!g_strcmp0 (obj_meta->obj_label, "Car")) {
          vehicle_count++;
          num_rects++;
        }
        if (!g_strcmp0 (obj_meta->obj_label, "TwoWheeler")) {
          twowheeler_count++;
          num_rects++;
        }
        if (!g_strcmp0 (obj_meta->obj_label, "Person")) {
          person_count++;
          num_rects++;
        }
        if (!g_strcmp0 (obj_meta->obj_label, "Roadsign")) {
          roadsign_count++;
          num_rects++;
        }
      }
    }
    // Print total #objects of all streams combined
    g_print ("Frame Number = %d Number of objects = %d "
        "Vehicle Count = %d Person Count = %d TwoWheeler = %d Roadsign = %d\n",
        frame_number, num_rects, vehicle_count, person_count, twowheeler_count, roadsign_count
        );
    frame_number++;
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
}

static
    gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *
      loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id = 0;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *
      caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *
      str = gst_caps_get_structure (caps, 0);
  const gchar *
      name = gst_structure_get_name (str);
  GstElement *
      source_bin = (GstElement *) data;
  GstCapsFeatures *
      features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *
          bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *
      bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {
  };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *
      loop = NULL;
  GstElement *
      pipeline = NULL, *streammux = NULL, *sink = NULL, *caps_filter_appsink =
      NULL, *queue1, *queue2, *queue3, *queue4, *queue5, *queue6, *tiler =
      NULL, *nvvidconv = NULL, *nvvidconv_appsink = NULL, *nvvidconv_filesink =
      NULL, *nvosd = NULL, *tee = NULL, *appsink = NULL;
  GstElement *
      h264enc = NULL, *h264parser = NULL, *qt_mux = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_sink_pad = NULL;
  GstCaps *caps_appsink = NULL;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  GstPad *tee_source_pad1, *tee_source_pad2;
  GstPad *dest1_sink_pad, *dest2_sink_pad;
  gboolean use_filesink = false;
  gboolean use_softenc = false;
  gchar *outfile_name = NULL;
  gchar source_type[5] = { 0 };
  gchar outfile_type[4] = { 0 };

  /* Check input arguments */
  if (argc < 3) {
    g_printerr
        ("Usage: %s [uri1] [kitti1] [uri2] [kitti2] ... [uriN]  [kittiN] outfile-hard [out_filename]\n",
        argv[0]);
    return -1;
  }

  snprintf (source_type, 5, "%s", argv[argc - 2]);

  if (!(g_strcmp0 (argv[argc - 2], "outfile-hw-enc"))) {
    num_sources = (argc - 3) / 2;
    outfile_name = argv[argc - 1];
    snprintf (outfile_type, 4, "%s", outfile_name + strlen (outfile_name) - 3);
    g_print ("filetype = %s\n", outfile_type);
    if (g_strcmp0 (outfile_type, "mp4")) {
      g_printerr ("Only use .mp4 outfile format\n");
      return -1;
    }
    g_print ("using %s to dump the output with hardware encoder\n",
        outfile_name);
    use_filesink = true;
    use_softenc = false;
  } else if (!(g_strcmp0 (argv[argc - 2], "outfile-sw-enc"))) {
    num_sources = (argc - 3) / 2;
    outfile_name = argv[argc - 1];
    snprintf (outfile_type, 4, "%s", outfile_name + strlen (outfile_name) - 3);
    g_print ("filetype = %s\n", outfile_type);
    if (g_strcmp0 (outfile_type, "mp4")) {
      g_printerr ("Only use .mp4 outfile format\n");
      return -1;
    }
    g_print ("using %s to dump the output with software encoder\n",
        outfile_name);
    use_filesink = true;
    use_softenc = true;
  } else if (realpath (argv[argc - 2] + 7, NULL) == NULL
      && !g_strcmp0 (source_type, "file")) {
    g_printerr ("To dump output in a file use %s [uri1] [kitti1] \
    [uri2] [kitti2] ... [uriN]  [kittiN] outfile [out_filepath]\n", argv[0]);
    return -1;
  } else {
    num_sources = (argc - 1) / 2;
  }
  g_print ("num_sources = %d\n", num_sources);

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-kitti-overlay-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad * sinkpad, * srcpad;
    gchar pad_name[16] = {};
    GstElement * source_bin = create_source_bin (i, argv[2*i + 1]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    std::vector < DSMetaInfo > vecDSMetadata_src;
    char kitti_dir_path[PATH_MAX] = { 0 };
    GDir * kitti_dir;
    GError * gerror;
    const gchar * filename;
    gint num_files = 0;
    /* open kitti data directory */
    {
      char *resolved = realpath (argv[2 * i + 2], NULL);
      if (!resolved) {
        g_printerr ("For Source %d: Kitti Dir Path : %s : %s\n", i,
            argv[2 * i + 2], g_strerror (errno));
        return -1;
      }
      g_strlcpy (kitti_dir_path, resolved, sizeof (kitti_dir_path));
      free (resolved);
    }
    kitti_dir = g_dir_open (kitti_dir_path, 0, &gerror);
    if (gerror != NULL) {
      g_printerr ("For Source %d: Kitti Dir Path : %s : %s\n", i, argv[2*i + 2], gerror->message);
      g_clear_error (&gerror);
      return -1;
    }

    /* loop over files in kitti data directory */
    while ((filename = g_dir_read_name (kitti_dir))) {
      FILE *file;
      char * line_buf = NULL;
      gsize line_buf_size = 0;
      char kitti_file[PATH_MAX] = { 0 };

      guint ctx_index;
      guint stream_id;
      gint frame_num;

      gchar obj_label[MAX_LABEL_SIZE];
      float left;
      float top;
      float right;
      float bottom;
      float confidence;
      std::vector < ObjectInfo > vecobj_info;

      /* Skip entries that are not DeepStream KITTI dump files
       * (name format: <ctx>_<stream>_<frame>.txt). */
      if (sscanf (filename, "%02u_%03u_%06d.txt", &ctx_index, &stream_id,
              &frame_num) != 3) {
        continue;
      }

      /* get absolute file path */
      g_snprintf (kitti_file, sizeof (kitti_file), "%s/%s", kitti_dir_path,
          filename);

      /* open file for reading */
      file = fopen (kitti_file, "r");
      if (!file) {
        g_printerr ("error while opening kitti file: %s: %s\n", kitti_file,
            g_strerror (errno));
        g_dir_close (kitti_dir);
        return -1;
      }
      num_files ++;

      /* read object meta information */
      while (getline (&line_buf, &line_buf_size, file) != -1) {
        sscanf (line_buf, "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f",
            obj_label, &left, &top, &right, &bottom, &confidence);
        vecobj_info.emplace_back (obj_label, left, top, right, bottom,
            confidence);
      }
      free (line_buf);
      fclose (file);

      /* push the read meta information to a vector */
      vecDSMetadata_src.emplace_back (ctx_index, stream_id, frame_num,
          vecobj_info);
    }
    g_dir_close (kitti_dir);
    if (num_files == 0) {
      g_print("Empty Kitti Directory Provided\n");
      return -1;
    }
    /* Sort the meta data by frame number */
    std::sort (vecDSMetadata_src.begin (), vecDSMetadata_src.end (),
        [](const DSMetaInfo & left, const DSMetaInfo & right) {
        return (left.frame_num < right.frame_num);}
    );

    /* push the particular source meta to the vector */
    vecDSMetadata.emplace_back (vecDSMetadata_src);

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");
  queue6 = gst_element_factory_make ("queue", "queue6");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* RGBA to NV12 converter as required by Appsink */
  nvvidconv_appsink =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");

  caps_filter_appsink = gst_element_factory_make ("capsfilter", "caps-appsink");

/* Finally render the osd output. We will use a tee to render video
   * playback on nveglglessink or dump output in a file,
   * and we use appsink to extract metadata from buffer and
   * print total object, person and vehicle count. */
  tee = gst_element_factory_make ("tee", "tee");
  if (!tee) {
    g_printerr ("Tee could not be created. Exiting.\n");
    return -1;
  }

  if (!tiler || !nvvidconv || !nvosd || !nvvidconv_appsink
      || !caps_filter_appsink || !tee) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (use_filesink) {
    /* RGBA to NV12 converter as required by Filesink */
    nvvidconv_filesink =
        gst_element_factory_make ("nvvideoconvert", "nvvideo-converter3");

    if (use_softenc) {
      h264enc = gst_element_factory_make ("avenc_mpeg4", "mpeg4-sw-enc");

      h264parser = gst_element_factory_make ("mpeg4videoparse", "mpeg4-parse");

    } else {
      //use hardware encoder
      h264enc = gst_element_factory_make ("nvv4l2h264enc", "h264-hw-enc");

      h264parser = gst_element_factory_make ("h264parse", "h264-parse");
    }

    qt_mux = gst_element_factory_make ("qtmux", "qt-mux");

    sink = gst_element_factory_make ("filesink", "file-sink");

    if (!nvvidconv_filesink || !h264enc || !h264parser || !qt_mux || !sink) {
      g_printerr ("One Filesink element could not be created. Exiting.\n");
      return -1;
    }

    g_object_set (G_OBJECT (sink), "location", outfile_name, NULL);

  } else {
#ifdef PLATFORM_TEGRA
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
    if (!sink) {
      g_printerr ("sink element could not be created. Exiting.\n");
      return -1;
    }
    g_object_set (G_OBJECT (sink), "qos", 0, NULL);
  }

  appsink = gst_element_factory_make ("appsink", "app-sink");
  if (!appsink) {
    g_printerr ("appsink element could not be created. Exiting.\n");
    return -1;
  }
  /* Configure appsink to extract data from DeepStream pipeline */
  g_object_set (appsink, "emit-signals", TRUE, "async", FALSE, NULL);

  /* Callback to access buffer and object info. */
  g_signal_connect (appsink, "new-sample", G_CALLBACK (appsink_sample), NULL);

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  /* caps filter for NV12 and CPU buffer required by appsink */
  caps_appsink =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
      "NV12", NULL);

  g_object_set (G_OBJECT (caps_filter_appsink), "caps", caps_appsink, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */

  if (use_filesink) {
    gst_bin_add_many (GST_BIN (pipeline), queue1, tiler, queue2,
        nvvidconv, queue3, nvosd, queue4, tee, nvvidconv_appsink,
        nvvidconv_filesink, h264enc, h264parser, qt_mux, queue5, sink, queue6,
        caps_filter_appsink, appsink, NULL);
    /* we link the elements together
     * nvstreammux -> nvtiler -> nvvidconv -> nvosd -> tee */
    if (!gst_element_link_many (streammux, queue1, tiler, queue2,
            nvvidconv, queue3, nvosd, queue4, tee, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    tee_source_pad1 = gst_element_request_pad_simple (tee, "src_0");
    dest1_sink_pad = gst_element_get_static_pad (queue5, "sink");

    tee_source_pad2 = gst_element_request_pad_simple (tee, "src_1");
    dest2_sink_pad = gst_element_get_static_pad (queue6, "sink");

    if (gst_pad_link (tee_source_pad1, dest1_sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Tee could not be linked to filesink.\n");
      gst_object_unref (pipeline);
      return -1;
    }
    if (gst_pad_link (tee_source_pad2, dest2_sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Tee could not be linked to appsink.\n");
      gst_object_unref (pipeline);
      return -1;
    }
    gst_object_unref (dest1_sink_pad);
    gst_object_unref (dest2_sink_pad);

    /* we link the filesink elements together
     * queue5 -> nvvidconv_filesink -> h264enc -> h264parser -> qt_mux -> sink */

    if (!gst_element_link_many (queue5, nvvidconv_filesink,
            h264enc, h264parser, qt_mux, sink, NULL)) {
      g_printerr ("Filesink Elements could not be linked. Exiting.\n");
      return -1;
    }
    /* we link the appsink elements together
     * queue6 -> nvvidconv_appsink -> caps_filter_appsink -> appsink */
    if (!gst_element_link_many (queue6, nvvidconv_appsink, caps_filter_appsink,
            appsink, NULL)) {
      g_printerr ("Appsink Elements could not be linked. Exiting.\n");
      return -1;
    }
  } else {
    gst_bin_add_many (GST_BIN (pipeline), queue1, tiler, queue2,
        nvvidconv, queue3, nvosd, queue4, tee, queue5, sink,
        nvvidconv_appsink, caps_filter_appsink, queue6, appsink, NULL);
    /* we link the elements together
     * nvstreammux  -> nvtiler -> nvvidconv -> nvosd -> tee */
    if (!gst_element_link_many (streammux, queue1, tiler, queue2,
            nvvidconv, queue3, nvosd, queue4, tee, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    tee_source_pad1 = gst_element_request_pad_simple (tee, "src_0");
    tee_source_pad2 = gst_element_request_pad_simple (tee, "src_1");

    dest1_sink_pad = gst_element_get_static_pad (queue5, "sink");
    dest2_sink_pad = gst_element_get_static_pad (queue6, "sink");

    if (gst_pad_link (tee_source_pad1, dest1_sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Tee could not be linked to display sink.\n");
      gst_object_unref (pipeline);
      return -1;
    }
    if (gst_pad_link (tee_source_pad2, dest2_sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Tee could not be linked to appsink.\n");
      gst_object_unref (pipeline);
      return -1;
    }
    gst_object_unref (dest1_sink_pad);
    gst_object_unref (dest2_sink_pad);

    /* we link the eglsink elements together
     * queue5 -> sink */
    if (!gst_element_link_many (queue5, sink, NULL))
    {
      g_printerr ("Sink Elements could not be linked. Exiting.\n");
      return -1;
    }

    /* we link the appsink elements together
     * queue6 -> nvvidconv_appsink -> caps_filter_appsink -> appsink */
    if (!gst_element_link_many (queue6, nvvidconv_appsink, caps_filter_appsink,
            appsink, NULL)) {
      g_printerr ("Appsink2 Elements could not be linked. Exiting.\n");
      return -1;
    }
  }

  /* Lets add probe to attach meta data from generated kitti data, we add probe
   * to the sink pad of the tiler element */
  tiler_sink_pad = gst_element_get_static_pad (tiler, "sink");
  if (!tiler_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_sink_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[2*i + 1]);
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
