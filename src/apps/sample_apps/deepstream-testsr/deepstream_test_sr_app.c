/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string.h>
#include <cuda_runtime_api.h>
#include "gst-nvdssr.h"

GST_DEBUG_CATEGORY (NVDS_APP);

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

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

/* Config file parameters used for recording
 * User needs to change these parameters to reflect the change in recordings
 * e.g duration, start-time etc. */

/* Container format of recorded file 0 for mp4 and 1 for mkv format
 */
#define SMART_REC_CONTAINER 0

/* Cache functionality of recording
 */
#define CACHE_SIZE_SEC 15

/* Timeout if duration of recording is not set by user
 */
#define SMART_REC_DEFAULT_DURATION 10

/* Time at which the recording is started
 */
#define START_TIME 2

/* Duration of recording
 */
#define SMART_REC_DURATION 7

/* Interval in seconds for
 * SR start / stop events generation.
 */
#define SMART_REC_INTERVAL 7

static gboolean bbox_enabled = TRUE;
static gint enc_type = 0; // Default: Hardware encoder
static gint sink_type = 2; // Default: Eglsink
static guint sr_mode = 0; // Default: Audio + Video
static guint pgie_type = 0; // Default: Nvinfer

GOptionEntry entries[] = {
  {"bbox-enable", 'e', 0, G_OPTION_ARG_INT, &bbox_enabled,
      "0: Disable bboxes, \
       1: Enable bboxes, \
       Default: bboxes enabled", NULL}
  ,
  {"enc-type", 'c', 0, G_OPTION_ARG_INT, &enc_type,
      "0: Hardware encoder, \
       1: Software encoder, \
       Default: Hardware encoder", NULL}
  ,
  {"sink-type", 's', 0, G_OPTION_ARG_INT, &sink_type,
      "1: Fakesink, \
       2: Eglsink, \
       3: RTSP sink, \
       Default: Eglsink", NULL}
  ,
  {"sr-mode", 'm', 0, G_OPTION_ARG_INT, &sr_mode,
      "SR mode: 0 = Audio + Video, \
       1 = Video only, \
       2 = Audio only", NULL}
  ,
  {"pgie-type", 'p', 0, G_OPTION_ARG_INT, &pgie_type,
      "PGIE type: 0 = Nvinfer, \
       1 = Nvinferserver, \
       Default: Nvinfer", NULL}
  ,
  {NULL}
  ,
};

static GstElement *pipeline = NULL, *tee_pre_decode = NULL;
static NvDsSRContext *nvdssrCtx = NULL;
static GMainLoop *loop = NULL;

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

static gpointer
smart_record_callback (NvDsSRRecordingInfo * info, gpointer userData)
{
  static GMutex mutex;
  FILE *logfile = NULL;
  g_return_val_if_fail (info, NULL);

  g_mutex_lock (&mutex);
  logfile = fopen ("smart_record.log", "a");
  if (logfile) {
    fprintf (logfile, "%d:%s:%d:%d:%s:%d channel(s):%d Hz:%ldms:%s:%s\n",
        info->sessionId, info->containsVideo ? "video" : "no-video",
        info->width, info->height, info->containsAudio ? "audio" : "no-audio",
        info->channels, info->samplingRate, info->duration,
        info->dirpath, info->filename);
    fclose (logfile);
  } else {
    g_print ("Error in opeing smart record log file\n");
  }
  g_mutex_unlock (&mutex);

  return NULL;
}

static gboolean
smart_record_event_generator (gpointer data)
{
  NvDsSRSessionId sessId = 0;
  NvDsSRContext *ctx = (NvDsSRContext *) data;
  guint startTime = START_TIME;
  guint duration = SMART_REC_DURATION;

  if (ctx->recordOn) {
    g_print ("Recording done.\n");
    if (NvDsSRStop (ctx, 0) != NVDSSR_STATUS_OK)
      g_printerr ("Unable to stop recording\n");
  } else {
    g_print ("Recording started..\n");
    if (NvDsSRStart (ctx, &sessId, startTime, duration,
            NULL) != NVDSSR_STATUS_OK)
      g_printerr ("Unable to start recording\n");
  }
  return TRUE;
}

static void
cb_newpad_audio_parsebin (GstElement * element, GstPad * element_src_pad, gpointer data)
{
  GstPad *sinkpad = gst_element_get_static_pad(nvdssrCtx->recordbin, "asink");
  if (gst_pad_link(element_src_pad, sinkpad) != GST_PAD_LINK_OK) {
    g_print ("Elements not linked. Exiting. \n");
    g_main_loop_quit(loop);
  }
}

static void
cb_newpad (GstElement * element, GstPad * element_src_pad, gpointer data)
{

  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (element_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  GstElement *depay_elem = (GstElement *) data;

  const gchar *media = gst_structure_get_string (str, "media");
  gboolean is_video = (!g_strcmp0 (media, "video"));
  gboolean is_audio = (!g_strcmp0 (media, "audio"));

  if (g_strrstr (name, "x-rtp") && is_video) {
    GstPad *sinkpad = gst_element_get_static_pad (depay_elem, "sink");
    if (gst_pad_link (element_src_pad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link depay loader to rtsp src");
    }
    gst_object_unref (sinkpad);

    if (!bbox_enabled && (sr_mode == 0 || sr_mode == 1)) {
      GstElement *parser_pre_recordbin =
          gst_element_factory_make ("h264parse", "parser-pre-recordbin");

      gst_bin_add_many (GST_BIN (pipeline), parser_pre_recordbin, NULL);

      if (!gst_element_link_many (tee_pre_decode, parser_pre_recordbin,
              nvdssrCtx->recordbin, NULL)) {
        g_print ("Elements not linked. Exiting. \n");
        g_main_loop_quit(loop);
      }
      gst_element_sync_state_with_parent(parser_pre_recordbin);
    }
  }

  if (g_strrstr (name, "x-rtp") && is_audio) {
    if (!bbox_enabled && (sr_mode == 0 || sr_mode == 2)) {
      GstElement *parser_pre_recordbin =
          gst_element_factory_make ("parsebin", "audio-parser-pre-recordbin");

      gst_bin_add_many (GST_BIN (pipeline), parser_pre_recordbin, NULL);

      GstPad *sinkpad = gst_element_get_static_pad(parser_pre_recordbin, "sink");
      if (gst_pad_link(element_src_pad, sinkpad) != GST_PAD_LINK_OK) {
        g_print ("Elements not linked. Exiting. \n");
        g_main_loop_quit(loop);
      }

      g_signal_connect(G_OBJECT(parser_pre_recordbin), "pad-added", G_CALLBACK(cb_newpad_audio_parsebin), NULL);

      gst_element_sync_state_with_parent(parser_pre_recordbin);
    }
  }

  gst_caps_unref (caps);
}

int
main (int argc, char *argv[])
{
  GstElement *streammux = NULL, *sink = NULL, *pgie = NULL, *source = NULL,
      *nvvidconv = NULL, *nvvidconv2 = NULL, *encoder_post_osd = NULL,
      *queue_pre_sink = NULL, *queue_post_osd = NULL, *parser_post_osd = NULL,
      *nvosd = NULL, *tee_post_osd = NULL, *queue_pre_decode = NULL,
      *depay_pre_decode = NULL, *decoder = NULL,  *nvvidconv3 = NULL,
      *swenc_caps = NULL;

  GstCaps *caps = NULL;
  GstElement *cap_filter = NULL;

  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  guint i = 0, num_sources = 1;

  guint pgie_batch_size = 0;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  GOptionContext *gctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;

  NvDsSRInitParams params = { 0 };

  gctx = g_option_context_new ("Nvidia DeepStream Test-SR app");
  group = g_option_group_new ("SR_test", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (gctx, group);
  g_option_context_add_group (gctx, gst_init_get_option_group ());

  GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

  if (!g_option_context_parse (gctx, &argc, &argv, &error)) {
    g_printerr ("%s", error->message);
    g_print ("%s", g_option_context_get_help (gctx, TRUE, NULL));
    return -1;
  }

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <rtsp_h264 uri>\n", argv[0]);
    return -1;
  }

  if (argc > 2) {
    g_printerr ("One rtsp_h264 uri supported Usage: %s <rtsp_h264 uri> \n",
        argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest-sr-pipeline");

  source = gst_element_factory_make ("rtspsrc", "rtsp-source");
  g_object_set (G_OBJECT (source), "location", argv[1], NULL);


  depay_pre_decode = gst_element_factory_make ("rtph264depay", "h264-depay");

  queue_pre_decode = gst_element_factory_make ("queue", "queue-pre-decode");

  if (!source || !depay_pre_decode || !queue_pre_decode) {
    g_printerr ("One element in source end could not be created.\n");
    return -1;
  }

  g_signal_connect (G_OBJECT (source), "pad-added",
      G_CALLBACK (cb_newpad), depay_pre_decode);
  /* Create tee which connects decoded source data and Smart record bin without bbox */
  tee_pre_decode = gst_element_factory_make ("tee", "tee-pre-decode");

  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == 0) {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }
  else if (pgie_type == 1) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  }

  /* Use queue to connect to the sink after tee_post_osd element */
  queue_pre_sink = gst_element_factory_make ("queue", "queue-pre-sink");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Use convertor to convert from RGBA to CAPS filter data format */
  nvvidconv2 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");

  g_object_set (G_OBJECT (nvvidconv), "output-buffers", 5, NULL);
  g_object_set (G_OBJECT (nvvidconv2), "output-buffers", 5, NULL);

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Create tee which connects to sink and Smart record bin with bbox */
  tee_post_osd = gst_element_factory_make ("tee", "tee-post-osd");


  /* Finally render the osd output */
  if (sink_type == 1) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  }
  else if (sink_type == 2) {
    if (prop.integrated) {
      sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
    } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif      
    }
    g_object_set (G_OBJECT (sink), "async", FALSE, NULL);
  }
  else if (sink_type == 3) {
    sink = gst_element_factory_make ("nvrtspoutsinkbin", "nvvideo-renderer");
    g_object_set (G_OBJECT (sink), "sync", TRUE, NULL);
    g_object_set (G_OBJECT (sink), "bitrate", 768000, NULL);
    g_object_set (G_OBJECT (sink), "enc-type", 1, NULL);
  }

  g_object_set (G_OBJECT (streammux), "live-source", 1, NULL);
  g_object_set (G_OBJECT (streammux), "buffer-pool-size", 5, NULL);

  /* Observed hang issue with I420 on jetson platform with nv3dsink. Use NV12 */
  caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)NV12");
  cap_filter =
      gst_element_factory_make ("capsfilter", "src_cap_filter_nvvidconv");
  g_object_set (G_OBJECT (cap_filter), "caps", caps, NULL);
  gst_caps_unref (caps);

  if (!pgie || !nvvidconv || !nvosd || !nvvidconv2 || !cap_filter
      || !tee_post_osd || !tee_pre_decode || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  if (pgie_type == 1) {
    g_object_set (G_OBJECT (pgie), "config-file-path", "dstestsr_pgie_nvinferserver_config.txt", NULL);
  } else {
    g_object_set (G_OBJECT (pgie), "config-file-path", "dstestsr_pgie_config.yml", NULL);
  }

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set (G_OBJECT (sink), "qos", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline
   * rtsp-source-> h264-depay -> tee-> queue -> decoder ->nvstreammux -> pgie -> nvvidconv -> nvosd -> nvvidconv -> caps_filter -> tee -> queue -> video-renderer
   *                                                                                                                                     |-> queue -> encoder -> parser -> recordbin
   */
  gst_bin_add_many (GST_BIN (pipeline), source, depay_pre_decode,
      tee_pre_decode, queue_pre_decode, decoder, streammux, pgie, nvvidconv,
      nvosd, nvvidconv2, cap_filter, tee_post_osd, queue_pre_sink, sink, NULL);

  /* Link the elements together till decoder */
  if (!gst_element_link_many (depay_pre_decode, tee_pre_decode,
          queue_pre_decode, decoder, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  /* Link decoder with streammux */
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (decoder, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* Link the remaining elements of the pipeline to streammux */
  if (!gst_element_link_many (streammux, pgie,
          nvvidconv, nvosd, nvvidconv2, cap_filter, tee_post_osd,
          queue_pre_sink, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Parameters are set before creating record bin
   * User can set additional parameters e.g recorded file path etc.
   * Refer NvDsSRInitParams structure for additional parameters
   */
  params.containerType = SMART_REC_CONTAINER;
  params.cacheSize = CACHE_SIZE_SEC;
  params.defaultDuration = SMART_REC_DEFAULT_DURATION;
  params.callback = smart_record_callback;
  params.fileNamePrefix = bbox_enabled ? "With_BBox" : "Without_BBox";

  if (NvDsSRCreate (&nvdssrCtx, &params) != NVDSSR_STATUS_OK) {
    g_printerr ("Failed to create smart record bin");
    return -1;
  }

  gst_bin_add_many (GST_BIN (pipeline), nvdssrCtx->recordbin, NULL);

  if (bbox_enabled) {
    /* Encode the data from tee before recording with bbox */
    if (enc_type == 0) {
        /* Hardware encoder used*/
        encoder_post_osd =
            gst_element_factory_make ("nvv4l2h264enc", "encoder-post-osd");

      } else if (enc_type == 1) {
        /* Software encoder used*/

        swenc_caps =  gst_element_factory_make ("capsfilter", NULL);

        GstCaps *enc_caps = NULL;

        enc_caps = gst_caps_from_string ("video/x-h264, profile=(string)baseline");

        g_object_set (G_OBJECT (swenc_caps), "caps", enc_caps, NULL);
        gst_caps_unref (enc_caps);

        encoder_post_osd =
            gst_element_factory_make ("x264enc", "encoder-post-osd");

        nvvidconv3 = gst_element_factory_make ("nvvideoconvert", "nvvidconv3");
        g_object_set (G_OBJECT (nvvidconv3), "output-buffers", 5, NULL);
        gst_bin_add_many (GST_BIN (pipeline), swenc_caps, nvvidconv3, NULL);
      }

    /* Parse the encoded data after osd component */
    parser_post_osd = gst_element_factory_make ("h264parse", "parser-post-osd");

    /* Use queue to connect the tee_post_osd to nvencoder */
    queue_post_osd = gst_element_factory_make ("queue", "queue-post-osd");

    gst_bin_add_many (GST_BIN (pipeline), queue_post_osd, encoder_post_osd,
        parser_post_osd, NULL);

    if (enc_type == 0) {
      if (!gst_element_link_many (tee_post_osd, queue_post_osd, encoder_post_osd,
              parser_post_osd, nvdssrCtx->recordbin, NULL)) {
        g_print ("Elements not linked. Exiting. \n");
        return -1;
      }
    }
    else if (enc_type == 1) {
      /* Link swenc_caps and nvvidconv3 in case of software encoder*/
      if (!gst_element_link_many (tee_post_osd, nvvidconv3, queue_post_osd,
              encoder_post_osd, swenc_caps, parser_post_osd,
              nvdssrCtx->recordbin, NULL)) {
        g_print ("Elements not linked. Exiting. \n");
        return -1;
      }
    }
  }

  if (nvdssrCtx) {
    g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        nvdssrCtx);
  }

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  g_print (" %s", argv[i + 1]);

  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  if (pipeline) {
    gst_element_set_state (pipeline, GST_STATE_NULL);
  }
  if (nvdssrCtx) {
    if(NvDsSRDestroy (nvdssrCtx) != NVDSSR_STATUS_OK)
      g_printerr ("Unable to destroy recording instance\n");
  }
  g_print ("Deleting pipeline\n");
  if (pipeline) {
    gst_object_unref (GST_OBJECT (pipeline));
  }
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
