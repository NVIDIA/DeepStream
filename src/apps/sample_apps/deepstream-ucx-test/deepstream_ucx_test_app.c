/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime_api.h>
#include "deepstream_ucx_test_app.h"


static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
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
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad_audio (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;

  /* Need to check if the pad created by the decodebin is for audio and not
   * video. */
  if (!strncmp (name, "audio", 5)) {
    /* Get the source bin ghost pad */
    GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
    if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
            decoder_src_pad)) {
      g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
    }
    gst_object_unref (bin_ghost_pad);
  }
}

static void
cb_newpad_video_server (GstElement * decodebin, GstPad * decoder_src_pad,
    gpointer data)
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
cb_newpad_client (GstElement * decodebin, GstPad * demux_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (demux_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *h264sink = (GstElement *) data;

  /* Need to check if the pad created by the qtdemux is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    g_print ("Qtdemux src pad: %s\n", name);
    GstPad *sinkpad = gst_element_get_static_pad (h264sink, "sink");

    if (gst_pad_is_linked (sinkpad)) {
      g_printerr ("h264 parser sink pad already linked.\n");
      return;
    }

    if (gst_pad_link (demux_src_pad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link demux and h264 pads.\n");
      return;
    }

    gst_pad_set_active (demux_src_pad, TRUE);
    gst_pad_set_active (sinkpad, TRUE);

    gst_object_unref (sinkpad);
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
create_source_bin (guint index, gchar * uri, gboolean is_video)
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
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri,
      "async-handling", 1, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (is_video ? cb_newpad_video_server : cb_newpad_audio), bin);
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
test3_server_parse_args(int argc, char *argv[], UcxTest *t)
{
  ServerTest3Args *args = &t->args.server_test3;
  gchar *endptr0 = NULL;
  if (argc != 5) {
    g_printerr ("Usage: %s -t 3 -s <addr> <port> "
        "<audio metadata serialization lib path> <uri>\n", argv[0]);
    return -1;
  }

  args->addr = argv[1];
  args->port = g_ascii_strtoll (argv[2], &endptr0, 10);
  args->libpath = argv[3];
  args->uri = argv[4];

  if (!g_hostname_is_ip_address (args->addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        args->addr);
    return -1;
  }

  if ((args->port == 0 && endptr0 == argv[2]) || args->port <= 0) {
    g_printerr ("Incorrect port specified\n");
    return -1;
  }

  if (args->libpath == NULL || args->uri == NULL) {
    g_printerr ("Invalid library path or URI specified\n");
    return -1;
  }

  return 0;
}

int
test3_client_parse_args(int argc, char *argv[], UcxTest *t)
{
  ClientTest3Args *args = &t->args.client_test3;
  gchar *endptr0 = NULL;
  if (argc != 5) {
    g_printerr ("Usage: %s -t 3 -c <addr> <port> "
        "<audio metadata serialization lib path>  <outputfile>\n", argv[0]);
    return -1;
  }

  args->addr = argv[1];
  args->port = g_ascii_strtoll (argv[2], &endptr0, 10);
  args->libpath = argv[3];
  args->outfile = argv[4];

  if (!g_hostname_is_ip_address (args->addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        args->addr);
    return -1;
  }

  if ((args->port == 0 && endptr0 == argv[2]) || args->port <= 0) {
    g_printerr ("Incorrect port, width or height specified\n");
    return -1;
  }

  if (args->libpath == NULL || args->outfile == NULL) {
    g_printerr ("Invalid library path or output file specified\n");
    return -1;
  }

  return 0;
}

int
test3_server_setup_pipeline(UcxTest *t)
{
  ServerTest3Args *args = &t->args.server_test3;
  GstElement *source_bin = NULL, *ucxserversink = NULL, *audioconv = NULL,
      *caps_filter = NULL, *audiores = NULL, *absplit = NULL,
      *streammux = NULL, *metains = NULL;
  GstCaps *caps = NULL;

  // Create GStreamer elements
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  audioconv = gst_element_factory_make ("audioconvert", "audio-convert");
  caps_filter = gst_element_factory_make ("capsfilter", "caps_filter");
  audiores = gst_element_factory_make ("audioresample", ":audio-resample");
  absplit = gst_element_factory_make ("audiobuffersplit", "audio-buffer-split");
  metains = gst_element_factory_make ("nvdsmetainsert", "nvds-meta-insert");
  ucxserversink = gst_element_factory_make ("nvdsucxserversink", "serversink");

  if (!streammux || !audioconv || !caps_filter || !audiores || !absplit ||
      !metains || !ucxserversink) {
    g_printerr ("Failed to create some elements\n");
    return -1;
  }

  source_bin = create_source_bin (0, args->uri, FALSE);
  if (!source_bin) {
    g_printerr ("Failed to create source bin\n");
    return -1;
  }

  caps = gst_caps_new_simple ("audio/x-raw", "format", G_TYPE_STRING, "F32LE",
      "rate", G_TYPE_INT, 48000, "channels", G_TYPE_INT, 1, "layout",
      G_TYPE_STRING, "interleaved", NULL);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);
  g_object_set (G_OBJECT (ucxserversink), "addr", args->addr, "port",
      args->port, "buf-type", 1, NULL);
  g_object_set (G_OBJECT (streammux), "max-latency", 250000000, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);
  g_object_set (G_OBJECT (metains), "serialize-lib", args->libpath, NULL);

  gst_bin_add_many (GST_BIN (t->pipeline), streammux, source_bin, audioconv,
      caps_filter, audiores, absplit, metains, ucxserversink, NULL);
  GstPad *srcpad, *audsink;
  srcpad = gst_element_get_static_pad (source_bin, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin\n");
    return -1;
  }
  audsink = gst_element_get_static_pad (audioconv, "sink");
  if (!audsink) {
    g_printerr ("Audioconvert static sink pad failed\n");
    return -1;
  }
  if (gst_pad_link (srcpad, audsink) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link source bin to audioconvert\n");
    return -1;
  }
  gst_object_unref (srcpad);
  gst_object_unref (audsink);

  if (!gst_element_link_many (audioconv, audiores, caps_filter, absplit,
      NULL)) {
    g_printerr ("Failed to link several elements with absplit\n");
    return -1;
  }

  GstPad *abssrcpad, *muxsinkpad;
  gchar pad_name[16] = { };
  abssrcpad = gst_element_get_static_pad (absplit, "src");
  if (!abssrcpad) {
    g_printerr ("Failed to get src pad for audiobuffersplit\n");
    return -1;
  }
  g_snprintf (pad_name, 15, "sink_%u", 0);
  muxsinkpad = gst_element_request_pad_simple (streammux, pad_name);
  if (!muxsinkpad) {
    g_printerr ("Failed to request sink pad for streammux\n");
    return -1;
  }
  if (gst_pad_link (abssrcpad, muxsinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link buffer split src and streammux sink pads\n");
    return -1;
  }
  gst_object_unref (abssrcpad);
  gst_object_unref (muxsinkpad);

  if (!gst_element_link_many (streammux, metains, ucxserversink, NULL)) {
    g_printerr ("Failed to link elements with ucxserversink\n");
    return -1;
  }

  g_print ("Using URI: %s\n", args->uri);
  g_print ("Server listening on: %s : %ld\n", args->addr, args->port);
  return 0;
}

int
test3_client_setup_pipeline(UcxTest *t)
{
  ClientTest3Args *args = &t->args.client_test3;
  GstElement *ucxclientsrc = NULL, *caps_filter1 = NULL,
    *metaext = NULL, *streamdemux = NULL, *audioconv = NULL,
    *caps_filter2 = NULL, *waveenc = NULL, *filesink = NULL;
  GstCaps *caps1 = NULL, *caps2 = NULL;
  GstCapsFeatures *feature = NULL;

  ucxclientsrc = gst_element_factory_make ("nvdsucxclientsrc", "ucxclientsrc");
  caps_filter1 = gst_element_factory_make ("capsfilter", "caps-filter1");
  metaext = gst_element_factory_make ("nvdsmetaextract", "nvds-meta-extract");
  streamdemux = gst_element_factory_make ("nvstreamdemux", "nvdemux");
  audioconv = gst_element_factory_make ("audioconvert", "audio-convert");
  caps_filter2 = gst_element_factory_make ("capsfilter", "caps-filter2");
  waveenc = gst_element_factory_make ("wavenc", "wavenc");
  filesink = gst_element_factory_make ("filesink", "filesink");

  if (!ucxclientsrc || !caps_filter1 || !metaext || !streamdemux ||
      !audioconv || !caps_filter2 || !waveenc || !filesink) {
    g_printerr ("Failed to create some element\n");
    return -1;
  }

  /* Set properties */
  g_object_set (G_OBJECT (ucxclientsrc), "addr", args->addr, "port", args->port,
      "nvbuf-batch-size", 1, "num-nvbuf", 4, "nvbuf-memory-type", 2,
      "buf-type", 1, NULL);
  caps1 = gst_caps_new_simple ("audio/x-raw", "format", G_TYPE_STRING, "F32LE",
      "rate", G_TYPE_INT, 48000, "channels", G_TYPE_INT, 1, "layout",
      G_TYPE_STRING, "interleaved", NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps1, 0, feature);
  g_object_set (G_OBJECT (caps_filter1), "caps", caps1, NULL);
  g_object_set (G_OBJECT (metaext), "deserialize-lib", args->libpath, NULL);
  g_object_set (G_OBJECT (filesink), "location", args->outfile, NULL);
  caps2 = gst_caps_new_simple ("audio/x-raw", "format", G_TYPE_STRING, "S16LE",
      NULL);
  g_object_set (G_OBJECT (caps_filter2), "caps", caps2, NULL);

  gst_bin_add_many (GST_BIN (t->pipeline), ucxclientsrc, caps_filter1, metaext,
      streamdemux, audioconv, caps_filter2, waveenc, filesink, NULL);

  if (!gst_element_link_many (ucxclientsrc, caps_filter1, metaext,
          streamdemux, NULL)) {
    g_printerr ("Failed to link some elements till streamdemux\n");
    return -1;
  }

  GstPad *demuxsrcpad, *audsinkpad;
  gchar pad_name[16] = { };
  g_snprintf (pad_name, 15, "src_%u", 0);
  demuxsrcpad = gst_element_request_pad_simple (streamdemux, pad_name);
  if (!demuxsrcpad) {
    g_printerr ("Failed to request src pad from demux. Exiting.\n");
    return -1;
  }
  audsinkpad = gst_element_get_static_pad (audioconv, "sink");
  if (!audsinkpad) {
    g_printerr ("Failed to get sink pad for audio converter\n");
    return -1;
  }
  if (gst_pad_link (demuxsrcpad, audsinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link demux src and audio conv sink pads\n");
    return -1;
  }
  gst_object_unref (demuxsrcpad);
  gst_object_unref (audsinkpad);

  if (!gst_element_link_many (audioconv, caps_filter2, waveenc, filesink,
      NULL)) {
    g_printerr ("Failed to link elements till filesink\n");
    return -1;
  }

  return 0;
}

int
test2_server_parse_args(int argc, char *argv[], UcxTest *t)
{
  ServerTest2Args *args = &t->args.server_test2;
  gchar *endptr0 = NULL, *endptr1 = NULL, *endptr2 = NULL;

  if (argc != 7) {
    g_printerr ("Usage: %s -t 2 -s <addr> <port> <width> <height> "
        "<video metadata serialization lib path> <file output path>\n",
        argv[0]);
    return -1;
  }

  args->addr = argv[1];
  args->port = g_ascii_strtoll (argv[2], &endptr0, 10);
  args->width = g_ascii_strtoll (argv[3], &endptr1, 10);
  args->height = g_ascii_strtoll (argv[4], &endptr2, 10);
  args->libpath = argv[5];
  args->output = argv[6];

  if (!g_hostname_is_ip_address (args->addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        args->addr);
    return -1;
  }

  if ((args->port == 0 && endptr0 == argv[2]) ||
      (args->width == 0 && endptr1 == argv[3]) ||
      (args->height == 0 && endptr2 == argv[4])) {
    g_printerr ("Incorrect port, width or height specified\n");
    return -1;
  }

  if (args->port <= 0 || args->width <= 0 || args->height <= 0) {
    g_printerr ("Invalid port, width or height\n");
    return -1;
  }

  if (args->libpath == NULL || args->output == NULL) {
    g_printerr ("Invalid custom lib path or output file specified.\n");
    return -1;
  }

  return 0;
}

int
test2_client_parse_args(int argc, char *argv[], UcxTest *t)
{
  ClientTest2Args *args = &t->args.client_test2;
  if (argc != 6) {
    g_printerr("Usage: %s -t 2 -c <addr> <port> <infer config path> "
        "<video metadata serialization lib path> <file path>\n", argv[0]);
    g_printerr("For nvinferserver usage: %s -t 2 -c -i <addr> <port> "
    "<nvinferserver config path> <video metadata serialization lib path> "
    "<file path>\n", argv[0]);
    return -1;
  }

  args->addr = argv[1];
  args->port = g_ascii_strtoll(argv[2], NULL, 10);
  args->inferpath = argv[3];
  args->libpath = argv[4];
  args->uri = argv[5];

  if (!g_hostname_is_ip_address(args->addr)) {
    g_printerr("Addr specified is not a valid IP address/hostname: %s\n",
        args->addr);
    return -1;
  }

  if (args->port <= 0) {
    g_printerr("Incorrect port specified\n");
    return -1;
  }

  if (args->inferpath == NULL || args->libpath == NULL || args->uri == NULL) {
    g_printerr("Invalid path for infer config or custom lib or "
        "file specified\n");
    return -1;
  }

  return 0;
}

int
test2_server_setup_pipeline(UcxTest *t)
{
  ServerTest2Args *args = &t->args.server_test2;
  GstElement *ucxserversrc = NULL, *caps_filter = NULL, *nvvidconv = NULL,
      *nvmetaext = NULL, *nvosd = NULL, *nvvidconv2 = NULL, *nvenc = NULL,
      *h264parse = NULL, *qtmux = NULL, *filesink = NULL;
  GstCaps *caps = NULL;

  // Create GStreamer elements
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  caps_filter = gst_element_factory_make ("capsfilter", "caps_filter");
  ucxserversrc = gst_element_factory_make ("nvdsucxserversrc", "serversrc");
  nvmetaext = gst_element_factory_make ("nvdsmetaextract", "nvds-meta-extract");
  nvosd = gst_element_factory_make ("nvdsosd", "nvosd");
  nvvidconv2 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  nvenc = gst_element_factory_make ("nvv4l2h264enc", "nvh264enc");
  h264parse = gst_element_factory_make ("h264parse", "h264-parse");
  qtmux = gst_element_factory_make ("qtmux", "qt-mux");
  filesink = gst_element_factory_make ("filesink", "file-sink");

  if (!nvvidconv || !caps_filter || !ucxserversrc || !nvmetaext ||
      !nvosd || !nvvidconv2 || !nvenc || !h264parse || !qtmux || !filesink) {
    g_printerr ("Failed to create some elements\n");
    return -1;
  }

  caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
      "width", G_TYPE_INT, args->width, "height", G_TYPE_INT, args->height,
      "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
  gst_caps_set_features (caps, 0, gst_caps_features_new ("memory:NVMM", NULL));
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);
  g_object_set (G_OBJECT (nvmetaext), "deserialize-lib", args->libpath, NULL);
  g_object_set (G_OBJECT (ucxserversrc), "addr", args->addr, "port",
      args->port, "nvbuf-memory-type", 2, "num-nvbuf", 8, "buf-type", 0, NULL);
  g_object_set (G_OBJECT (filesink), "location", args->output, NULL);

  gst_bin_add_many (GST_BIN (t->pipeline), ucxserversrc, caps_filter, nvvidconv,
      nvmetaext, nvosd, nvvidconv2, nvenc, h264parse, qtmux, filesink, NULL);

  if (!gst_element_link_many (ucxserversrc, caps_filter, nvvidconv,
      nvmetaext, nvosd, nvvidconv2, nvenc, h264parse, qtmux,
      filesink, NULL)) {
    g_printerr ("Failed to link some elements\n");
    return -1;
  }
  g_print ("Server listening on: %s : %ld\n", args->addr, args->port);

  return 0;
}

int
test2_client_setup_pipeline(UcxTest *t)
{
  ClientTest2Args *args = &t->args.client_test2;
  GstElement *filesrc = NULL, *streammux = NULL,
      *ucxclientsink = NULL, *qtdemux = NULL, *h264parse = NULL,
      *nvdecode = NULL, *nvvidconv = NULL, *pgie = NULL, *nvmetains = NULL,
      *queue1 = NULL;

  filesrc = gst_element_factory_make("filesrc", "file-src");
  qtdemux = gst_element_factory_make("qtdemux", "qt-demux");
  h264parse = gst_element_factory_make("h264parse", "h264-parse");
  nvdecode = gst_element_factory_make("nvv4l2decoder", "nvv-4l2decoder");
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  pgie = gst_element_factory_make(t->is_nvinfer_server ?
      NVINFERSERVER_PLUGIN : NVINFER_PLUGIN, "nv-infer");
  nvmetains = gst_element_factory_make("nvdsmetainsert", "nvds-meta-insert");
  ucxclientsink = gst_element_factory_make("nvdsucxclientsink", "clientsink");
  queue1 = gst_element_factory_make("queue", "queue1");
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  // Error check
  if (!streammux || !filesrc || !qtdemux || !h264parse || !nvdecode ||
      !nvvidconv || !pgie || !nvmetains || !ucxclientsink || !queue1) {
    g_printerr("Failed to create some elements\n");
    return -1;
  }

  // Set the properties for the elements
  g_object_set(G_OBJECT(filesrc), "location", args->uri, NULL);
  g_object_set(G_OBJECT(ucxclientsink), "addr", args->addr, "port", args->port,
      "buf-type", 0, NULL);
  g_object_set(G_OBJECT(pgie), "config-file-path", args->inferpath, NULL);
  g_object_set(G_OBJECT(nvmetains), "serialize-lib", args->libpath, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

  // Add elements to the pipeline and link them
  gst_bin_add_many(GST_BIN(t->pipeline), filesrc, qtdemux, h264parse, nvdecode,
      streammux, nvvidconv, pgie, nvmetains, ucxclientsink, NULL);

  g_signal_connect(G_OBJECT(qtdemux), "pad-added",
      G_CALLBACK(cb_newpad_client), h264parse);
  if (!gst_element_link(filesrc, qtdemux)) {
    g_printerr("Failed to link filesrc and qtdemux\n");
    return -1;
  }
  if (!gst_element_link(h264parse, nvdecode)) {
    g_printerr("Failed to link h264parse and nv4l2decode\n");
    return -1;
  }

  if (!gst_element_link_many(streammux, nvvidconv, pgie, nvmetains,
      ucxclientsink, NULL)) {
    g_printerr("Failed to link several elements including clientsink\n");
    return -1;
  }

  GstPad *srcpad, *muxsinkpad;
  gchar pad_name[16] = { };
  srcpad = gst_element_get_static_pad(nvdecode, "src");
  if (!srcpad) {
    g_printerr("Failed to get src pad for decoder\n");
    return -1;
  }
  g_snprintf(pad_name, 15, "sink_%u", 0);
  muxsinkpad = gst_element_request_pad_simple (streammux, pad_name);
  if (!muxsinkpad) {
    g_printerr("Failed to request sink pad for streammux\n");
    return -1;
  }
  if (gst_pad_link(srcpad, muxsinkpad) != GST_PAD_LINK_OK) {
    g_printerr("Failed to link decode src and streammux sink pads\n");
    return -1;
  }
  gst_object_unref(srcpad);
  gst_object_unref(muxsinkpad);

  return 0;
}

int
test1_server_parse_args(int argc, char *argv[], UcxTest *t)
{
  ServerTest1Args *args = &t->args.server_test1;
  gchar *endptr0 = NULL, *endptr1 = NULL, *endptr2 = NULL;

  /* Check input arguments */
  if (argc != 6) {
      g_printerr("Usage: %s -t 1 -s <addr> <port> <width> <height> <uri>\n",
          argv[0]);
      return -1;
  }

  args->addr = argv[1];
  args->port = g_ascii_strtoll(argv[2], &endptr0, 10);
  args->width = g_ascii_strtoll(argv[3], &endptr1, 10);
  args->height = g_ascii_strtoll(argv[4], &endptr2, 10);
  args->uri = argv[5];

  if (!g_hostname_is_ip_address(args->addr)) {
      g_printerr("Address specified is not a valid IP address/hostname: %s\n",
          args->addr);
      return -1;
  }

  if ((args->port == 0 && endptr0 == argv[2]) ||
      (args->width == 0 && endptr1 == argv[3]) ||
      (args->height == 0 && endptr2 == argv[4])) {
      g_printerr("Incorrect port, width or height specified\n");
      return -1;
  }

  if (args->port <= 0 || args->width <= 0 || args->height <= 0) {
      g_printerr("Invalid port, width or height\n");
      return -1;
  }

  if (args->uri == NULL) {
      g_printerr("Invalid URI\n");
      return -1;
  }

  return 0;
}

int
test1_client_parse_args(int argc, char *argv[], UcxTest *t)
{
  ClientTest1Args *args = &t->args.client_test1;
  gchar *endptr0 = NULL, *endptr1 = NULL, *endptr2 = NULL;

  if (argc != 6) {
    g_printerr("Usage: %s -t 1 -c <addr> <port> <width> <height> <outputfile>\n",
       argv[0]);
    return -1;
  }

  args->addr = argv[1];
  args->port = g_ascii_strtoll(argv[2], &endptr0, 10);
  args->width = g_ascii_strtoll(argv[3], &endptr1, 10);
  args->height = g_ascii_strtoll(argv[4], &endptr2, 10);
  args->outfile = argv[5];

  if (!g_hostname_is_ip_address(args->addr)) {
    g_printerr("Addr specified is not a valid IP address/hostname: %s\n",
        args->addr);
    return -1;
  }

  if ((args->port == 0 && endptr0 == argv[2]) ||
      (args->width == 0 && endptr1 == argv[3]) ||
      (args->height == 0 && endptr2 == argv[4])) {
    g_printerr("Incorrect port, width, or height specified\n");
    return -1;
  }

  if (args->port <= 0 || args->width <= 0 || args->height <= 0) {
    g_printerr("Invalid port, width or height\n");
    return -1;
  }

  if (args->outfile == NULL) {
    g_printerr("Invalid output file\n");
    return -1;
  }

  return 0;  // Indicates success
}

int
test1_server_setup_pipeline(UcxTest *t)
{
  ServerTest1Args *args = &t->args.server_test1;
  GstElement *ucxserversink = NULL, *nvvidconv = NULL, *caps_filter = NULL,
     *queue = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  GstPad *srcpad, *qsink;

  GstElement *source_bin = create_source_bin(0, args->uri, TRUE);
  if (!source_bin) {
    g_printerr("Failed to create source bin\n");
    return -1;
  }

  gst_bin_add(GST_BIN(t->pipeline), source_bin);

  /* Create the remaining elements. */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  queue = gst_element_factory_make("queue", "queue");
  caps_filter = gst_element_factory_make("capsfilter", "caps_filter");
  ucxserversink = gst_element_factory_make("nvdsucxserversink", "serversink");

  if (!nvvidconv || !queue || !caps_filter || !ucxserversink) {
      g_printerr("Failed to create video converter or caps element or ucx\n");
      return -1;
  }

  /* Set parameters */
  caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
      "width", G_TYPE_INT, args->width, "height", G_TYPE_INT, args->height,
      "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
  feature = gst_caps_features_new("memory:NVMM", NULL);
  gst_caps_set_features(caps, 0, feature);
  g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
  g_object_set(G_OBJECT(ucxserversink), "addr", args->addr, "port", args->port,
      "buf-type", 0, NULL);

  gst_bin_add_many(GST_BIN(t->pipeline), queue, nvvidconv, caps_filter,
      ucxserversink, NULL);

  srcpad = gst_element_get_static_pad(source_bin, "src");
  if (!srcpad) {
    g_printerr("Failed to get src pad of source bin\n");
    return -1;
  }
  qsink = gst_element_get_static_pad(queue, "sink");
  if (!qsink) {
    g_printerr("Queue static sink pad failed\n");
    return -1;
  }
  if (gst_pad_link(srcpad, qsink) != GST_PAD_LINK_OK) {
    g_printerr("Failed to link source bin to queue\n");
    return -1;
  }
  gst_object_unref(srcpad);
  gst_object_unref(qsink);

  if (!gst_element_link_many(queue, nvvidconv, caps_filter, ucxserversink, NULL)) {
      g_printerr("Failed to link several elements\n");
      return -1;
  }

  g_print ("Using URI: %s\n", args->uri);
  g_print ("Server listening on: %s : %ld\n", args->addr, args->port);
  return 0;
}

int
test1_client_setup_pipeline(UcxTest *t)
{
  ClientTest1Args *args = &t->args.client_test1;
  GstElement *ucxclientsrc = NULL, *nvvidconv = NULL, *caps_filter = NULL,
      *filesink = NULL, *queue1 = NULL, *h264enc = NULL,
      *h264parser = NULL, *qtmux = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;

  ucxclientsrc = gst_element_factory_make ("nvdsucxclientsrc", "ucxclientsrc");
  caps_filter = gst_element_factory_make ("capsfilter", "caps_filter");
  queue1 = gst_element_factory_make ("queue", "queue1");
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  h264enc = gst_element_factory_make ("nvv4l2h264enc", "h264-hw-enc");
  h264parser = gst_element_factory_make ("h264parse", "h264-parse");
  qtmux = gst_element_factory_make ("qtmux", "qt-mux");
  filesink = gst_element_factory_make ("filesink", "file-sink");

  if (!ucxclientsrc || !caps_filter || !queue1 || !nvvidconv || !h264enc ||
      !h264parser || !qtmux || !filesink) {
    g_printerr ("One pipeline element could not be created\n");
    return -1;
  }

  /* Set ucxclientsrc properties */
  g_object_set (G_OBJECT (ucxclientsrc), "addr", args->addr, "port",
      args->port,"nvbuf-batch-size", 1, "num-nvbuf", 4, "nvbuf-memory-type",
      2, NULL);
  caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
      "width", G_TYPE_INT, args->width, "height", G_TYPE_INT, args->height,
      "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  /* Set filesink properties */
  g_object_set (G_OBJECT (filesink), "location", args->outfile, "async", 0,
      "sync", 1, "qos", 0, NULL);

  gst_bin_add_many (GST_BIN (t->pipeline), ucxclientsrc, caps_filter,
      nvvidconv, queue1, h264enc, h264parser, qtmux, filesink, NULL);

  if (!gst_element_link_many (ucxclientsrc, caps_filter, queue1, nvvidconv,
      h264enc, h264parser, qtmux, filesink, NULL)) {
    g_printerr ("Failed to link several elements\n");
    return -1;
  }
  g_print ("Now saving stream to: %s\n", args->outfile);

  return 0;  // Indicates success
}

void
run_pipeline (UcxTest *t) {
  GstBus *bus = NULL;
  guint bus_watch_id;

    /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(t->pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, t->loop);
  gst_object_unref(bus);

  gst_element_set_state (t->pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (t->loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (t->pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (t->pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (t->loop);
  return;
}


int main (int argc, char *argv[])
{
  UcxTest t = {0};
  t.role = INIT;
  t.test_id = -1;
  t.is_nvinfer_server = FALSE;
  struct cudaDeviceProp prop;
  int current_device = -1;
  char *new_argv[argc];
  int new_argc = 0;

  for (int i = 0; i < argc; i++) {
    if (g_strcmp0(argv[i], "-t") == 0 && i + 1 < argc) {
      t.test_id = atoi(argv[++i]);
    } else if (g_strcmp0(argv[i], "-s") == 0) {
        t.role = SERVER;
    } else if (g_strcmp0(argv[i], "-c") == 0) {
        t.role = CLIENT;
    } else if (g_strcmp0(argv[i], "-i") == 0) {
        t.is_nvinfer_server = TRUE;
    } else {
      new_argv[new_argc++] = argv[i];
    }
  }

  if (t.test_id < 1 || t.test_id > sizeof(testOperations) / sizeof(TestOps)) {
    g_printerr ("Invalid test case: %d use -t <1-3>\n", t.test_id);
    return -1;
  }

  OperationFunctions *ops;
  if (t.role == SERVER) {
    ops = &testOperations[t.test_id - 1].server;
  } else if (t.role == CLIENT) {
    ops = &testOperations[t.test_id - 1].client;
  } else {
    g_printerr ("Invalid arguments: use -c for client and -s for server\n");
    return -1;
  }

  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&prop, current_device);

  if (ops->parse_args(new_argc, new_argv, &t) != 0) {
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  t.loop = g_main_loop_new(NULL, FALSE);
    if (t.role == SERVER) {
    t.pipeline = gst_pipeline_new("ds-ucx-server-pipeline");
  } else if (t.role == CLIENT) {
    t.pipeline = gst_pipeline_new("ds-ucx-client-pipeline");
  }
  if (!t.pipeline) {
    g_printerr("Failed to create pipeline\n");
    return -1;
  }

  if (ops->setup_pipeline(&t) != 0) {
    return -1;
  }

  run_pipeline(&t);

  return 0;

}

