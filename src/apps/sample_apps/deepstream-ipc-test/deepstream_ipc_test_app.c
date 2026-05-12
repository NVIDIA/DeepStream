/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"

#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#define MAX_SOURCE_BINS 32
#define MAX_DISPLAY_LEN 64

#define CONFIG_CSI_SOURCE_WIDTH 1920
#define CONFIG_CSI_SOURCE_HEIGHT 1080
#define CONFIG_CSI_SOURCE_FPS_N 30
#define CONFIG_CSI_SOURCE_FPS_D 1

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

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

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

typedef struct
{
  int fd;
  gchar *uri;
  gchar *socket_path;
  guint cam_src;
  guint sensor_id;
  guint bus_id;
  GstElement *pipeline;
  void *appCtx;
} NvIpcServerPipeline;

typedef struct
{
  gchar *uri[MAX_SOURCE_BINS];
  gchar *socket_path[MAX_SOURCE_BINS];
  guint bus_id;
  GstElement *pipeline;
  void *appCtx;
} NvIpcClientPipeline;

typedef struct
{
  GMainLoop *loop;
  NvIpcServerPipeline ipcserver[MAX_SOURCE_BINS];
  NvIpcClientPipeline ipcclient;
  gboolean quit;
} AppCtx;

/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    //NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        //int offset = 0;
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
          g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_meta->frame_num, num_rects, vehicle_count, person_count);
    }
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
    case GST_MESSAGE_STATE_CHANGED:
    {
      GstState oldstate = GST_STATE_NULL, newstate = GST_STATE_NULL;
      gst_message_parse_state_changed (msg, &oldstate, &newstate, NULL);
        switch (newstate) {
          case GST_STATE_PLAYING:
            g_print ("Pipeline running\n");
            break;
          case GST_STATE_PAUSED:
            if (oldstate == GST_STATE_PLAYING) {
              g_print ("Pipeline paused\n");
            }
            break;
          case GST_STATE_READY:
            if (oldstate == GST_STATE_NULL) {
              g_print ("Pipeline ready\n");
            } else {
              g_print ("Pipeline stopped\n");
            }
            break;
          case GST_STATE_NULL:
            g_print ("Pipeline Null\n");
            g_main_loop_quit (loop);
            return FALSE;
            break;
          default:
            break;
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
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
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
  if (g_strrstr (name, "source") == name) {
    g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
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
  uri_decode_bin = gst_element_factory_make ("nvurisrcbin", NULL);
  g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
  g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);

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

static GstElement*
create_csi_source_bin (guint index, guint sensor_id) {
  GstElement *bin = NULL, *src_elem = NULL, *caps_filter = NULL;
  GstPad *source_pad = NULL, *ghost_sourcepad = NULL;
  GstCaps *caps = NULL;
  gchar bin_name[16] = {};

  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name);

  src_elem = gst_element_factory_make ("nvarguscamerasrc", "csi_src_elem");

  if (!bin || !src_elem) {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }
  g_object_set (G_OBJECT (src_elem), "sensor-id", sensor_id, NULL);

  caps_filter = gst_element_factory_make("capsfilter", "src_cap_filter");
  if (!caps_filter) {
    g_printerr("Could not create 'src_cap_filter'");
    return NULL;
  }
  caps = gst_caps_new_simple(
                             "video/x-raw", "format", G_TYPE_STRING, "NV12", "width", G_TYPE_INT,
                             CONFIG_CSI_SOURCE_WIDTH, "height", G_TYPE_INT, CONFIG_CSI_SOURCE_HEIGHT,
                             "framerate", GST_TYPE_FRACTION, CONFIG_CSI_SOURCE_FPS_N,
                             CONFIG_CSI_SOURCE_FPS_D, NULL);

  GstCapsFeatures *feature = NULL;
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);

  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  gst_bin_add_many (GST_BIN (bin), src_elem, caps_filter, NULL);

  if (!gst_element_link_many (src_elem, caps_filter, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    gst_caps_unref (caps);
    return NULL;
  }

  source_pad = gst_element_get_static_pad (caps_filter, "src");
  ghost_sourcepad = gst_ghost_pad_new ("src", source_pad);
  gst_pad_set_active (ghost_sourcepad, TRUE);
  if (!gst_element_add_pad (bin, ghost_sourcepad)) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    gst_caps_unref (caps);
    gst_object_unref (source_pad);
    return NULL;
  }

  gst_caps_unref (caps);
  gst_object_unref (source_pad);

  return bin;
}

static int
create_client_pipeline (int argc, char *argv[])
{
  AppCtx appCtx;
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL, *nvdslogger = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  guint i =0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  gboolean PERF_MODE = g_getenv("NV_IPC_TEST_PERF_MODE") &&
      !g_strcmp0(g_getenv("NV_IPC_TEST_PERF_MODE"), "1");

  memset(&appCtx, 0, sizeof(AppCtx));
  appCtx.loop = loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ipc-client-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  num_sources = argc - 2;

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    const char *prefix = "ipc://";
    const char *socket_path = argv[i + 2] + strlen(prefix);
    appCtx.ipcclient.uri[i] = strdup(argv[i + 2]);
    appCtx.ipcclient.socket_path[i] = strdup(socket_path);

    GstElement *source_bin= NULL;
    source_bin = create_source_bin (i, argv[i + 2]);
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

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

  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");

  /* Use nvdslogger for perf measurement. */
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (PERF_MODE) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
    g_object_set (G_OBJECT (sink), "sync", FALSE, NULL);
  } else {
    sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    g_object_set (G_OBJECT (sink), "sync", FALSE, NULL);
  }

  if (!pgie || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "dsipctest_pgie_config.txt", NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set (G_OBJECT (sink), "qos", 0, NULL);
  g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 4, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, nvdslogger, tiler,
      queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);
  /* we link the elements together
  * nvstreammux -> nvinfer -> nvdslogger -> nvtiler -> nvvidconv -> nvosd
  * -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvdslogger, tiler,
        queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tiler_src_pad = gst_element_get_static_pad (pgie, "src");
  if (!tiler_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (tiler_src_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i + 2]);
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
  appCtx.quit = TRUE;

  for (i = 0; i < num_sources; i++) {
    g_free(appCtx.ipcclient.uri[i]);
    g_free(appCtx.ipcclient.socket_path[i]);
  }

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  return 0;
}

static int
create_server_pipeline (int argc, char *argv[])
{
  AppCtx appCtx;
  guint i =0, num_sources = 0;
  char **arg = argv + 2;
  GMainLoop *loop = NULL;

  memset(&appCtx, 0, sizeof(AppCtx));
  appCtx.loop = loop = g_main_loop_new (NULL, FALSE);

  while (*arg) {
    GstElement *pipeline = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstElement *source_bin=NULL, *queue, *nvunixfdsink= NULL;

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    appCtx.ipcserver[i].pipeline = pipeline = gst_pipeline_new ("ipc-server-pipeline");
    if (!pipeline) {
      g_printerr ("Failed to create pipeline. Exiting.\n");
      return -1;
    }

    if (!strcmp(*arg, "-cs")) {
      arg++;
      appCtx.ipcserver[i].cam_src = atoi(*arg++);
      appCtx.ipcserver[i].sensor_id = 0;
      if (!strcmp(*arg, "-sid")) {
        arg++;
        appCtx.ipcserver[i].sensor_id = atoi(*arg++);
      }
      source_bin = create_csi_source_bin (i, appCtx.ipcserver[i].sensor_id);
    } else {
      appCtx.ipcserver[i].uri = strdup(*arg++);
      source_bin = create_source_bin (i, appCtx.ipcserver[i].uri);
    }

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), source_bin);

    queue = gst_element_factory_make ("queue", NULL);
    if (!queue) {
      g_printerr ("Failed to create queue. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), queue);

    appCtx.ipcserver[i].socket_path = strdup(*arg++);
    nvunixfdsink = gst_element_factory_make ("nvunixfdsink", NULL);
    if (!nvunixfdsink) {
      g_printerr ("Failed to create nvunixfdsink. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), nvunixfdsink);

    g_object_set (G_OBJECT(nvunixfdsink), "sync", FALSE, NULL);
    g_object_set (G_OBJECT(nvunixfdsink), "socket-path", appCtx.ipcserver[i].socket_path, NULL);
    g_object_set (G_OBJECT(nvunixfdsink), "buffer-timestamp-copy", TRUE, NULL);

    /* link the elements together */
    if (!gst_element_link_many (source_bin, queue, nvunixfdsink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    appCtx.ipcserver[i].bus_id = bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    appCtx.ipcserver[i].appCtx = &appCtx;

    i++;
    num_sources++;
  }

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    gst_element_set_state (appCtx.ipcserver[i].pipeline, GST_STATE_PLAYING);
    g_print("server is started path: %s\n", appCtx.ipcserver[i].socket_path);
    GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx.ipcserver[i].pipeline),
                         GST_DEBUG_GRAPH_SHOW_ALL,"ipc-server-pipeline-added");
  }

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
  appCtx.quit = TRUE;

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  g_print ("Deleting pipeline\n");
  for (i = 0; i < num_sources; i++) {
    gst_element_set_state (appCtx.ipcserver[i].pipeline, GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (appCtx.ipcserver[i].pipeline));
    g_source_remove (appCtx.ipcserver[i].bus_id);
    g_print("server is closed path: %s\n", appCtx.ipcserver[i].socket_path);
    if (appCtx.ipcserver[i].uri)
      g_free(appCtx.ipcserver[i].uri);
    if (appCtx.ipcserver[i].socket_path)
      g_free(appCtx.ipcserver[i].socket_path);
  }

  g_main_loop_unref (loop);
  return 0;
}

int
main (int argc, char *argv[])
{
  int ret = 0;

  /* Check input arguments */
  if (argc < 3) {
    g_printerr ("Usage: %s <server> <rtsp_url> <socket_path>\n", argv[0]);
    g_printerr ("OR: %s <server> -cs <camera_source> -sid <sensor_id> <socket_path> \n", argv[0]);
    g_printerr ("OR: %s <client> <ipc_url>\n", argv[0]);
    g_printerr ("-cs : camera source to use (0 = CSI)\n");
    g_printerr ("-sid : camera sensor ID value (0, 1, ...)\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);

  if (strcmp(argv[1], "client") == 0) {
    ret = create_client_pipeline(argc, argv);
  } else {
    ret = create_server_pipeline(argc, argv);
  }

  return ret;
}
