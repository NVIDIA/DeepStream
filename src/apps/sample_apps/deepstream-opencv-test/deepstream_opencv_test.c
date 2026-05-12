/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include <string.h>

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"
#define PGIE_CONFIG_FILE  "dsopencvtest_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dsopencvtest_pgie_nvinferserver_config.txt"

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing NvBufSurface. */
#define MEMORY_FEATURES "memory:NVMM"

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
        vehicle_count++;
        num_rects++;
      }
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
        person_count++;
        num_rects++;
      }
    }
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
    offset =
        snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
        person_count);
    offset =
        snprintf (txt_params->display_text + offset, MAX_DISPLAY_LEN,
        "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = "Serif";
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
  }

  g_print ("Frame Number = %d Number of objects = %d "
      "Vehicle Count = %d Person Count = %d\n",
      frame_number, num_rects, vehicle_count, person_count);
  frame_number++;
  return GST_PAD_PROBE_OK;
}

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
    if (gst_caps_features_contains (features, MEMORY_FEATURES)) {
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

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <uri>\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <uri>\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *caps_filter = NULL, *dsexample = NULL, *nvosd =
      NULL, *nvvidconv1=NULL, *caps_filter1 = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  gboolean is_nvinfer_server = FALSE;
  gchar *input_stream = NULL;
  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    usage(argv[0]);
    return -1;
  }

  if (argc >=2 && !strcmp("-t", argv[1])) {
    if (!strcmp("inferserver", argv[2])) {
      is_nvinfer_server = TRUE;
    } else {
      usage(argv[0]);
      return -1;
    }
    g_print ("Using nvinferserver as the inference plugin\n");
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("deepstream-opencv-test-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  GstPad *sinkpad, *srcpad;
  GstElement *source_bin;

  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  if (is_nvinfer_server) {
    source_bin = create_source_bin (0, argv[3]);
    input_stream = argv[3];
  } else {
    source_bin = create_source_bin (0, argv[1]);
    input_stream = argv[1];
  }

  if (!source_bin) {
    g_printerr ("Failed to create source bin. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN (pipeline), source_bin);

  sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (source_bin, pad_name_src);
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

  /* Use nvinfer/nvinferserver to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (
        is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN,
        "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by dsexample */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");

  caps_filter = gst_element_factory_make ("capsfilter", NULL);
  caps_filter1 = gst_element_factory_make ("capsfilter", NULL);

  dsexample = gst_element_factory_make ("dsexample", "example-plugin");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if(prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!pgie || !nvvidconv || !caps_filter || !caps_filter1  || !dsexample
      || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (!use_new_mux) {
  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", 1,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }
  else {
    g_object_set (G_OBJECT (streammux), "batch-size", 1,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_NVINFERSERVER_CONFIG_FILE, NULL);
  } else {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  }

  /* Use NvBufSurfaceGetDeviceInfo to determine platform type.
   * This must match gst-dsexample's detection so the memory type
   * satisfies its blur-objects requirement on all platforms
   * (dGPU, Jetson, and Thor/IGX). */
  {
    NvBufSurfaceDeviceInfo dev_info = {0};
    int is_nvgpu = 0;
    if (NvBufSurfaceGetDeviceInfo (&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
        is_nvgpu = 1;
      }
    }
    if (!is_nvgpu) {
      g_object_set (G_OBJECT (nvvidconv), "nvbuf-memory-type", 1, NULL);
    }
  }

  /* Set properties of the caps_filter element */
  GstCaps *caps =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
      NULL);
  GstCapsFeatures *feature = gst_caps_features_new (MEMORY_FEATURES, NULL);
  gst_caps_set_features (caps, 0, feature);

  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);
  gst_caps_unref(caps);

  /* Set properties of the caps_filter1 element */
  caps =
      gst_caps_new_simple ("video/x-raw", "width", G_TYPE_INT, MUXER_OUTPUT_WIDTH, "height", G_TYPE_INT, MUXER_OUTPUT_HEIGHT,
      NULL);
  gst_caps_set_features (caps, 0, gst_caps_features_copy(feature));
  g_object_set (G_OBJECT (caps_filter1), "caps", caps, NULL);
  gst_caps_unref(caps);

  /* Set properties of the dsexample element */
  g_object_set (G_OBJECT (dsexample), "full-frame", FALSE, NULL);
  g_object_set (G_OBJECT (dsexample), "blur-objects", TRUE, NULL);

  /* Set properties of the sink element */
  g_object_set (G_OBJECT (sink), "sync", FALSE, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, caps_filter1, pgie, nvvidconv,
      caps_filter, dsexample, nvosd, sink, NULL);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * pgie -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, nvvidconv1, caps_filter1, pgie, nvvidconv,
          caps_filter, dsexample, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", input_stream);
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
