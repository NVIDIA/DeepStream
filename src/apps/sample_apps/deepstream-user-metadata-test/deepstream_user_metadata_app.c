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
#include <string.h>
#include <cuda_runtime_api.h>
#include "stdlib.h"
#include "gstnvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/** set the user metadata type */
#define NVDS_USER_FRAME_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))

#define USER_ARRAY_SIZE 16

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
#define PGIE_CONFIG_FILE  "dsmeta_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dsmeta_pgie_nvinferserver_config.txt"

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

void *set_metadata_ptr(void);
static gpointer copy_user_meta(gpointer data, gpointer user_data);
static void release_user_meta(gpointer data, gpointer user_data);

void *set_metadata_ptr()
{
  int i = 0;
  gchar *user_metadata = (gchar*)g_malloc0(USER_ARRAY_SIZE);

  g_print("\n**************** Setting user metadata array of 16 on nvinfer src pad\n");
  for(i = 0; i < USER_ARRAY_SIZE; i++) {
    user_metadata[i] = rand() % 255;
    g_print("user_meta_data [%d] = %d\n", i, user_metadata[i]);
  }
  return (void *)user_metadata;
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer copy_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  gchar *src_user_metadata = (gchar*)user_meta->user_meta_data;
  gchar *dst_user_metadata = (gchar*)g_malloc0(USER_ARRAY_SIZE);
  memcpy(dst_user_metadata, src_user_metadata, USER_ARRAY_SIZE);
  return (gpointer)dst_user_metadata;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void release_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if(user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_user_meta = NULL;
  NvDsUserMeta *user_meta = NULL;
  gchar *user_meta_data = NULL;
  int i = 0;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    /* Validate user meta */
    for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL;
        l_user_meta = l_user_meta->next) {
      user_meta = (NvDsUserMeta *) (l_user_meta->data);
      user_meta_data = (gchar *)user_meta->user_meta_data;

      if(user_meta->base_meta.meta_type == NVDS_USER_FRAME_META_EXAMPLE)
      {
        g_print("\n************ Retrieving user_meta_data array of 16 on osd sink pad\n");
        for(i = 0; i < USER_ARRAY_SIZE; i++) {
          g_print("user_meta_data [%d] = %d\n", i, user_meta_data[i]);
        }
        g_print("\n");
      }
    }
    frame_number++;
  }
  return GST_PAD_PROBE_OK;
}

/* Set nvds user metadata at frame level. User need to set 4 parameters after
 * acquring user meta from pool using nvds_acquire_user_meta_from_pool().
 *
 * Below parameters are required to be set.
 * 1. user_meta_data : pointer to User specific meta data
 * 2. meta_type: Metadata type that user sets to identify its metadata
 * 3. copy_func: Metadata copy or transform function to be provided when there
 *               is buffer transformation
 * 4. release_func: Metadata release function to be provided when it is no
 *                  longer required.
 *
 * osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
nvinfer_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMetaList * l_frame = NULL;
  NvDsUserMeta *user_meta = NULL;
  NvDsMetaType user_meta_type = NVDS_USER_FRAME_META_EXAMPLE;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

        /* Acquire NvDsUserMeta user meta from pool */
        user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

        /* Set NvDsUserMeta below */
        user_meta->user_meta_data = (void *)set_metadata_ptr();
        user_meta->base_meta.meta_type = user_meta_type;
        user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)copy_user_meta;
        user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;

        /* We want to add NvDsUserMeta to frame level */
        nvds_add_user_meta_to_frame(frame_meta, user_meta);
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
usage(const char *bin)
{
  g_printerr ("Usage: %s <H264 filename>\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <H264 filename>\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstPad *infer_src_pad = NULL;
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
  pipeline = gst_pipeline_new ("dstest1-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (
        is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN,
        "primary-nvinference-engine");
  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

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

  if (!source || !h264parser || !decoder || !pgie
      || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (source), "location", argv[3], NULL);
    input_stream = argv[3];
  } else {
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);
    input_stream = argv[1];
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

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, pgie,
      nvvidconv, nvosd, sink, NULL);

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

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * pgie -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (streammux, pgie,
      nvvidconv, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }


  /* Lets add probe to set user meta data at frame level. We add probe to
   * the src pad of the nvinfer element */
  infer_src_pad = gst_element_get_static_pad (pgie, "src");
  if (!infer_src_pad)
    g_print ("Unable to get source pad\n");
  else
    gst_pad_add_probe (infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        nvinfer_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (infer_src_pad);

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
