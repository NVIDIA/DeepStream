/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#define MEMORY_FEATURES "memory:NVMM"
#define MAX_FILE_LENGTH 500

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH  960
#define MUXER_OUTPUT_HEIGHT 752

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
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
#ifndef PLATFORM_TEGRA
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
#endif
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
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
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

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
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *caps_filter = NULL,
             *tiler = NULL, *nvdewarper = NULL, *nvvideoconvert = NULL, *outmux = NULL,
             *h264parser=NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint arg_index = 0;
  guint max_surface_per_frame;
  guint sink_type = 2; //Eglsink
  guint enc_type = 0; //Hardware encoder
  guint source_index_start=0;
  GstElement  *h264enc = NULL, *capfilt = NULL, *nvvidconv1 = NULL;
  char dewarp_filename[MAX_FILE_LENGTH] = {};
  int length = strlen("config_dewarper.txt");
  strncpy(dewarp_filename, "config_dewarper.txt", length+1);

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 3) {
    g_printerr ("Usage: %s --config <dewarp_config_filename> --sink <1/2/3> --enc_type <0/1> <uri1> <source id1> [<uri2> <source id2>] ... [<uriN> <source idN>]\
    \n", argv[0]);
    g_printerr ("\t --config <filename> (Default : config_dewarper.txt) \n");
    g_printerr ("\t --sink <1/2/3> 1:Fakesink, 2:Eglsink, 3:Filesink (Default : 2) \n");
    g_printerr ("\t --enc_type <0/1> 0:Hardware encoder, 1:Software encoder (Default : 0) \n");

    return -1;
  }
  num_sources = (argc - 1) / 2;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dewarper-app-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  arg_index = 1;
  while((!strcmp(argv[arg_index], "--config")) || (!strcmp(argv[arg_index], "--sink"))
      || (!strcmp(argv[arg_index], "--enc_type")))
  {
    if (!strcmp(argv[arg_index], "--config"))
    {
      num_sources = num_sources - 1;
      arg_index++;
      length = strlen(argv[arg_index]);
      if (length > MAX_FILE_LENGTH) {
        g_printerr ("Config file name exceeds max length\n");
      } else {
        strncpy(dewarp_filename, argv[arg_index++], MAX_FILE_LENGTH);
      }
    }
    if (!strcmp(argv[arg_index], "--sink"))
    {
      num_sources = num_sources - 1;
      arg_index++;
      sink_type = atoi(argv[arg_index++]);
      if((sink_type < 1) || (sink_type>3))
      {
        printf("Invalid sink type = %d. Setting to default : Eglsink\n", sink_type);
        sink_type = 2;
      }
    }
    if (!strcmp(argv[arg_index], "--enc_type"))
    {
      num_sources = num_sources - 1;
      arg_index++;
      enc_type = atoi(argv[arg_index++]);
      if((enc_type < 0) || (enc_type>1))
      {
        printf("Invalid encoder type = %d. Setting to default : Hardware encoder\n", sink_type);
        enc_type = 0;
      }
    }
  }
  source_index_start = arg_index;
  for (i = 0; i < num_sources; i++) {
    guint source_id = 0;

    GstPad *mux_sinkpad, *srcbin_srcpad, *dewarper_srcpad, *nvvideoconvert_sinkpad;
    gchar pad_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[arg_index++]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    source_id = atoi(argv[arg_index++]);

    /* create nv dewarper element */
    nvvideoconvert = gst_element_factory_make ("nvvideoconvert", NULL);
    if (!nvvideoconvert) {
      g_printerr ("Failed to create nvvideoconvert element. Exiting.\n");
      return -1;
    }

    caps_filter = gst_element_factory_make ("capsfilter", NULL);
    if (!caps_filter) {
      g_printerr ("Failed to create capsfilter element. Exiting.\n");
      return -1;
    }

    GstCaps *caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);
    GstCapsFeatures *feature = gst_caps_features_new (MEMORY_FEATURES, NULL);
    gst_caps_set_features (caps, 0, feature);

    g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

    /* create nv dewarper element */
    nvdewarper = gst_element_factory_make ("nvdewarper", NULL);
    if (!nvdewarper) {
      g_printerr ("Failed to create nvdewarper element. Exiting.\n");
      return -1;
    }

    g_object_set (G_OBJECT (nvdewarper),
      "config-file", dewarp_filename,
      "source-id", source_id,
      NULL);

    gst_bin_add_many (GST_BIN (pipeline), source_bin, nvvideoconvert, caps_filter, nvdewarper, NULL);

    if (!gst_element_link_many (nvvideoconvert, caps_filter, nvdewarper, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    g_snprintf (pad_name, 15, "sink_%u", i);
    mux_sinkpad = gst_element_request_pad_simple (streammux, pad_name);
    if (!mux_sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcbin_srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcbin_srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    nvvideoconvert_sinkpad = gst_element_get_static_pad (nvvideoconvert, "sink");
    if (!nvvideoconvert_sinkpad) {
      g_printerr ("Failed to get sink pad of nvvideoconvert. Exiting.\n");
      return -1;
    }

    dewarper_srcpad = gst_element_get_static_pad (nvdewarper, "src");
    if (!dewarper_srcpad) {
      g_printerr ("Failed to get src pad of nvdewarper. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcbin_srcpad, nvvideoconvert_sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (dewarper_srcpad, mux_sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcbin_srcpad);
    gst_object_unref (mux_sinkpad);
    gst_object_unref (dewarper_srcpad);
    gst_object_unref (nvvideoconvert_sinkpad);
    gst_caps_unref (caps);
  }

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  if (sink_type == 1) {
    sink = gst_element_factory_make ("fakesink", "fakesink");
  }
  else if(sink_type == 2){
    /* Finally render the osd output */
    if(prop.integrated){
      sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
    } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
    }
  }
  else if(sink_type == 3){
    h264parser = gst_element_factory_make ("h264parse", "h264-parser");
    outmux = gst_element_factory_make ("mp4mux", "mp4-mux");
    sink = gst_element_factory_make ("filesink", "nvvideo-filesink");
    g_object_set (G_OBJECT (sink), "location", "out.mp4", NULL);
  }
  else
  {
    g_printerr("Sink type invalid!!");
    return -1;
  }



  if (!nvdewarper || !tiler || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "nvbuf-memory-type", 0,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  g_object_get (G_OBJECT (nvdewarper), "num-batch-buffers", &max_surface_per_frame, NULL);

  g_object_set (G_OBJECT (streammux),
      "batch-size", num_sources*max_surface_per_frame,
      "num-surfaces-per-frame", max_surface_per_frame, NULL);


  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
    if(sink_type == 3)
  {
    if(enc_type == 0)
    {
      // Hardware encoder
      h264enc = gst_element_factory_make ("nvv4l2h264enc" ,"nvvideo-h264enc");
      if (!h264enc) {
        g_printerr ("h264enc element could not be created. Exiting.\n");
        return -1;
      }
      capfilt = gst_element_factory_make ("capsfilter", "nvvideo-caps");
      GstCaps *caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "I420", NULL);
      GstCapsFeatures *feature = gst_caps_features_new ("memory:NVMM", NULL);
      gst_caps_set_features (caps, 0, feature);
      g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
      gst_caps_unref(caps);
    }
    else
    {
      // Software encoder
      h264enc = gst_element_factory_make ("x264enc" ,"x264enc");
      if (!h264enc) {
        g_printerr ("h264enc element could not be created. Exiting.\n");
        return -1;
      }
      capfilt = gst_element_factory_make ("capsfilter", "x264-caps");
      GstCaps *caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "I420", NULL);
      g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
      gst_caps_unref(caps);
    }

    nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvid-converter1");

    gst_bin_add_many (GST_BIN (pipeline), tiler, nvvidconv1, capfilt,  h264enc, h264parser, outmux, sink, NULL);
    if (!gst_element_link_many (streammux, tiler, nvvidconv1,  capfilt, h264enc, h264parser, outmux, sink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }
  else
  {
    gst_bin_add_many (GST_BIN (pipeline), tiler, sink, NULL);
    if (!gst_element_link_many (streammux, tiler, sink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[source_index_start]);
    source_index_start += 2;
  }
  g_print ("\n");


  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline),
                  GST_DEBUG_GRAPH_SHOW_ALL, "dewarper_test_playing");
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
