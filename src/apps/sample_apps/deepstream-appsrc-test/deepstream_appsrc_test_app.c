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
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define CUSTOM_PTS 1

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"
#define PGIE_CONFIG_FILE  "dstest_appsrc_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dstest_appsrc_nvinferserver_config.txt"

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33000

gint frame_number = 0;

/* Structure to contain all our information for appsrc,
 * so we can pass it to callbacks */
typedef struct _AppSrcData
{
  GstElement *app_source;
  long frame_size;
  FILE *file;                   /* Pointer to the raw video file */
  gint appsrc_frame_num;
  guint fps;                    /* To set the FPS value */
  guint sourceid;               /* To control the GSource */
} AppSrcData;

/* new_sample is an appsink callback that will extract metadata received
 * tee sink pad and update params for drawing rectangle,
 *object information etc. */
static GstFlowReturn
new_sample (GstElement * sink, gpointer * data)
{
  GstSample *sample;
  GstBuffer *buf = NULL;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  unsigned long int pts = 0;

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
      pts = frame_meta->buf_pts;
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next) {
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
    }

    g_print ("Frame Number = %d Number of objects = %d "
        "Vehicle Count = %d Person Count = %d PTS = %" GST_TIME_FORMAT "\n",
        frame_number, num_rects, vehicle_count, person_count,
        GST_TIME_ARGS (pts));
    frame_number++;
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
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

/* This method is called by the idle GSource in the mainloop, 
 * to feed one raw video frame into appsrc.
 * The idle handler is added to the mainloop when appsrc requests us
 * to start sending data (need-data signal)
 * and is removed when appsrc has enough data (enough-data signal).
 */
static gboolean
read_data (AppSrcData * data)
{
  GstBuffer *buffer;
  GstFlowReturn gstret;

  size_t ret = 0;
  GstMapInfo map =  GST_MAP_INFO_INIT;
  buffer = gst_buffer_new_allocate (NULL, data->frame_size, NULL);

  gst_buffer_map (buffer, &map, GST_MAP_WRITE);
  ret = fread (map.data, 1, data->frame_size, data->file);
  map.size = ret;

  gst_buffer_unmap (buffer, &map);
  if (ret > 0) {
#if CUSTOM_PTS
    GST_BUFFER_PTS (buffer) =
        gst_util_uint64_scale (data->appsrc_frame_num, GST_SECOND, data->fps);
#endif
    gstret = gst_app_src_push_buffer ((GstAppSrc *) data->app_source, buffer);
    if (gstret != GST_FLOW_OK) {
      g_print ("gst_app_src_push_buffer returned %d \n", gstret);
      return FALSE;
    }
  } else if (ret == 0) {
    gstret = gst_app_src_end_of_stream ((GstAppSrc *) data->app_source);
    if (gstret != GST_FLOW_OK) {
      g_print
          ("gst_app_src_end_of_stream returned %d. EoS not queued successfully.\n",
          gstret);
      return FALSE;
    }
  } else {
    g_print ("\n failed to read from file\n");
    return FALSE;
  }
  data->appsrc_frame_num++;

  return TRUE;
}

/* This signal callback triggers when appsrc needs data. Here,
 * we add an idle handler to the mainloop to start pushing
 * data into the appsrc */
static void
start_feed (GstElement * source, guint size, AppSrcData * data)
{
  if (data->sourceid == 0) {
    data->sourceid = g_idle_add ((GSourceFunc) read_data, data);
  }
}

/* This callback triggers when appsrc has enough data and we can stop sending.
 * We remove the idle handler from the mainloop */
static void
stop_feed (GstElement * source, AppSrcData * data)
{
  if (data->sourceid != 0) {
    g_source_remove (data->sourceid);
    data->sourceid = 0;
  }
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <Raw filename> <width> <height> <fps> <format(I420, NV12, RGBA)>\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <Raw filename> <width> <height> <fps> "
    "<format(I420, NV12, RGBA)>\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *nvvidconv1 = NULL, *caps_filter = NULL,
      *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv2 = NULL,
      *nvosd = NULL, *tee = NULL, *appsink = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  AppSrcData data;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  gchar *endptr1 = NULL, *endptr2 = NULL, *endptr3 = NULL, *vidconv_format =
      NULL;
  GstPad *tee_source_pad1, *tee_source_pad2;
  GstPad *osd_sink_pad, *appsink_sink_pad;
  gboolean is_nvinfer_server = FALSE;
  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 6) {
    usage(argv[0]);
    return -1;
  }

  if (argc >=6 && !strcmp("-t", argv[1])) {
    if (!strcmp("inferserver", argv[2])) {
      is_nvinfer_server = TRUE;
    } else {
      usage(argv[0]);
      return -1;
    }
    g_print ("Using nvinferserver as the inference plugin\n");
  }

  gchar *format = NULL, *input_stream = NULL;
  long fps = 0, width = 0, height = 0;
  if (is_nvinfer_server) {
    width = g_ascii_strtoll (argv[4], &endptr1, 10);
    height = g_ascii_strtoll (argv[5], &endptr2, 10);
    fps = g_ascii_strtoll (argv[6], &endptr3, 10);
    format = argv[7];
    input_stream = argv[3];

    if ((width == 0 && endptr1 == argv[4]) || (height == 0
        && endptr2 == argv[5]) || (fps == 0 && endptr3 == argv[6])) {
      g_printerr ("Incorrect width, height or FPS\n");
      return -1;
    }
  } else {
    width = g_ascii_strtoll (argv[2], &endptr1, 10);
    height = g_ascii_strtoll (argv[3], &endptr2, 10);
    fps = g_ascii_strtoll (argv[4], &endptr3, 10);
    format = argv[5];
    input_stream = argv[1];

    if ((width == 0 && endptr1 == argv[2]) || (height == 0
        && endptr2 == argv[3]) || (fps == 0 && endptr3 == argv[4])) {
      g_printerr ("Incorrect width, height or FPS\n");
      return -1;
    }
  }
  if (width == 0 || height == 0 || fps == 0) {
    g_printerr ("Width, height or FPS cannot be 0\n");
    return -1;
  }

  if (g_strcmp0 (format, "I420") != 0 && g_strcmp0 (format, "RGBA") != 0
      && g_strcmp0 (format, "NV12") != 0) {
    g_printerr ("Only I420, RGBA and NV12 are supported\n");
    return -1;
  }

  /* Initialize custom data structure */
  memset (&data, 0, sizeof (data));
  if (!g_strcmp0 (format, "RGBA")) {
    data.frame_size = width * height * 4;
    vidconv_format = "RGBA";
  } else {
    data.frame_size = width * height * 1.5;
    vidconv_format = "NV12";
  }
  if (is_nvinfer_server) {
    data.file = fopen (argv[3], "r");
  } else {
    data.file = fopen (argv[1], "r");
  }
  data.fps = fps;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest-appsrc-pipeline");
  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }

  /* App Source element for reading from raw video file */
  data.app_source = gst_element_factory_make ("appsrc", "app-source");
  if (!data.app_source) {
    g_printerr ("Appsrc element could not be created. Exiting.\n");
    return -1;
  }

  /* Use convertor to convert from software buffer to GPU buffer */
  nvvidconv1 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
  if (!nvvidconv1) {
    g_printerr ("nvvideoconvert1 could not be created. Exiting.\n");
    return -1;
  }
  caps_filter = gst_element_factory_make ("capsfilter", "capsfilter");
  if (!caps_filter) {
    g_printerr ("Caps_filter could not be created. Exiting.\n");
    return -1;
  }

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  if (!streammux) {
    g_printerr ("nvstreammux could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on streammux's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (
          is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN, 
          "primary-nvinference-engine");
  if (!pgie) {
    g_printerr ("Primary nvinfer could not be created. Exiting.\n");
    return -1;
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvdsosd */
  nvvidconv2 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  if (!nvvidconv2) {
    g_printerr ("nvvideoconvert2 could not be created. Exiting.\n");
    return -1;
  }

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr ("nvdsosd could not be created. Exiting.\n");
    return -1;
  }

  /* Finally render the osd output. We will use a tee to render video
   * playback on nveglglessink, and we use appsink to extract metadata
   * from buffer and print object, person and vehicle count. */
  tee = gst_element_factory_make ("tee", "tee");
  if (!tee) {
    g_printerr ("Tee could not be created. Exiting.\n");
    return -1;
  }
  if(prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }
  if (!sink) {
    g_printerr ("Display sink could not be created. Exiting.\n");
    return -1;
  }

  appsink = gst_element_factory_make ("appsink", "app-sink");
  if (!appsink) {
    g_printerr ("Appsink element could not be created. Exiting.\n");
    return -1;
  }

  /* Configure appsrc */
  g_object_set (data.app_source, "caps",
      gst_caps_new_simple ("video/x-raw",
          "format", G_TYPE_STRING, format,
          "width", G_TYPE_INT, width,
          "height", G_TYPE_INT, height,
          "framerate", GST_TYPE_FRACTION, data.fps, 1, NULL), NULL);
#if !CUSTOM_PTS
  g_object_set (G_OBJECT (data.app_source), "do-timestamp", TRUE, NULL);
#endif
  g_signal_connect (data.app_source, "need-data", G_CALLBACK (start_feed),
      &data);
  g_signal_connect (data.app_source, "enough-data", G_CALLBACK (stop_feed),
      &data);

  caps =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
      vidconv_format, NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  /* Set streammux properties */
  if (!use_new_mux) {
    g_object_set (G_OBJECT (streammux), "width", width, "height",
      height, "batch-size", 1, "live-source", TRUE,
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

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      data.app_source, nvvidconv1, caps_filter, streammux, pgie,
      nvvidconv2, nvosd, tee, sink, appsink, NULL);

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (caps_filter, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link caps filter to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* we link the elements together */
  /* app-source -> nvvidconv -> caps filter ->
   * pgie -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (data.app_source, nvvidconv1, caps_filter, NULL) ||
      !gst_element_link_many (nvosd, sink, NULL) ||
      !gst_element_link_many (streammux, pgie, nvvidconv2, tee, NULL)) {
    g_printerr ("Elements could not be linked: Exiting.\n");
    return -1;
  }

/* Manually link the Tee, which has "Request" pads.
 * This tee, in case of multistream usecase, will come before tiler element. */
  tee_source_pad1 = gst_element_request_pad_simple (tee, "src_0");
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  tee_source_pad2 = gst_element_request_pad_simple (tee, "src_1");
  appsink_sink_pad = gst_element_get_static_pad (appsink, "sink");
  if (gst_pad_link (tee_source_pad1, osd_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked to display sink.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  if (gst_pad_link (tee_source_pad2, appsink_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked to appsink.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  gst_object_unref (osd_sink_pad);
  gst_object_unref (appsink_sink_pad);

  /* Configure appsink to extract data from DeepStream pipeline */
  g_object_set (appsink, "emit-signals", TRUE, "async", FALSE, NULL);
  g_object_set (sink, "sync", FALSE, NULL);

  /* Callback to access buffer and object info. */
  g_signal_connect (appsink, "new-sample", G_CALLBACK (new_sample), NULL);

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
