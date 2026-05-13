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
#include "stdlib.h"
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"
#define PGIE_CONFIG_FILE  "dsmeta_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dsmeta_pgie_nvinferserver_config.txt"

/** set the user metadata type */
#define NVDS_DECODER_GST_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.DECODER.GST_USER_META"))
#define NVDS_GST_META_BEFORE_DECODER_EXAMPLE (nvds_get_user_meta_type("NVIDIA.DECODER.GST_META_BEFORE_DECODER"))

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gint parsed_frame_number = 0;

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

typedef struct _NvDecoderMeta
{
  guint frame_type;
  guint frame_num;
  gboolean dec_err;
} NvDecoderMeta;

/* gst meta copy function set by user */
static gpointer decoder_meta_copy_func(gpointer data, gpointer user_data)
{
  NvDecoderMeta *src_decoder_meta = (NvDecoderMeta *)data;
  NvDecoderMeta *dst_decoder_meta = (NvDecoderMeta*)g_malloc0(
      sizeof(NvDecoderMeta));
  memcpy(dst_decoder_meta, src_decoder_meta, sizeof(NvDecoderMeta));
  return (gpointer)dst_decoder_meta;
}

/* gst meta release function set by user */
static void decoder_meta_release_func(gpointer data, gpointer user_data)
{
  NvDecoderMeta *decoder_meta = (NvDecoderMeta *)data;
  if(decoder_meta) {
    g_free(decoder_meta);
    decoder_meta = NULL;
  }
}

/* gst to nvds transform function set by user. "data" holds a pointer to NvDsUserMeta */
static gpointer decoder_gst_to_nvds_meta_transform_func(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDecoderMeta *src_decoder_meta =
    (NvDecoderMeta*)user_meta->user_meta_data;
  NvDecoderMeta *dst_decoder_meta =
    (NvDecoderMeta *)decoder_meta_copy_func(src_decoder_meta, NULL);
  return (gpointer)dst_decoder_meta;
}

/* release function set by user to release gst to nvds transformed metadata.
 * "data" holds a pointer to NvDsUserMeta */
static void decoder_gst_nvds_meta_release_func(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDecoderMeta *decoder_meta = (NvDecoderMeta *)user_meta->user_meta_data;
  decoder_meta_release_func(decoder_meta, NULL);
}

typedef struct _H264parseMeta
{
  guint parser_frame_num;
} H264parseMeta;

/* gst meta copy function set by user */
static gpointer h264parse_meta_copy_func(gpointer data, gpointer user_data)
{
  H264parseMeta *src_h264parse_meta = (H264parseMeta *)data;
  H264parseMeta *dst_h264parse_meta = (H264parseMeta*)g_malloc0(
      sizeof(H264parseMeta));
  memcpy(dst_h264parse_meta, src_h264parse_meta, sizeof(H264parseMeta));
  return (gpointer)dst_h264parse_meta;
}

/* gst meta release function set by user */
static void h264parse_meta_release_func(gpointer data, gpointer user_data)
{
  H264parseMeta *h264parse_meta = (H264parseMeta *)data;
  if(h264parse_meta) {
    g_free(h264parse_meta);
    h264parse_meta = NULL;
  }
}

/* gst to nvds transform function set by user. "data" holds a pointer to NvDsUserMeta */
static gpointer h264parse_gst_to_nvds_meta_transform_func(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  H264parseMeta *src_h264parse_meta =
    (H264parseMeta*)user_meta->user_meta_data;
  H264parseMeta *dst_h264parse_meta =
    (H264parseMeta *)h264parse_meta_copy_func(src_h264parse_meta, NULL);
  return (gpointer)dst_h264parse_meta;
}

/* release function set by user to release gst to nvds transformed metadata.
 * "data" holds a pointer to NvDsUserMeta */
static void h264parse_gst_nvds_meta_release_func(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  H264parseMeta *h264parse_meta = (H264parseMeta *)user_meta->user_meta_data;
  h264parse_meta_release_func(h264parse_meta, NULL);
}

/* nvinfer_src_pad_buffer_probe() will extract the metadata received on nvinfer
 * src pad.
 * It explains the mechanism to extract the decoder metadata (which is attached
 * using gstnvdsmeta API's in nvdecoder_src_pad_buffer_probe()),
 * now transformed into nvdsmeta. Decoder meta, attached to gst buffer
 * is set as user data at NvDsFrameMeta level
 * It also extracts metadata attached on src pad of h264parse element.
 */

static GstPadProbeReturn
nvinfer_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMetaList * l_frame = NULL;
  NvDsUserMeta *user_meta = NULL;
  NvDecoderMeta * decoder_meta = NULL;
  H264parseMeta * h264parse_meta = NULL;
  NvDsMetaList * l_user_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

        for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL;
            l_user_meta = l_user_meta->next) {
          user_meta = (NvDsUserMeta *) (l_user_meta->data);
          if(user_meta->base_meta.meta_type == NVDS_DECODER_GST_META_EXAMPLE)
          {
            decoder_meta = (NvDecoderMeta *)user_meta->user_meta_data;
            g_print("Dec Meta retrieved as NVDS USER METADTA For Frame_Num = %d  \n",
                decoder_meta->frame_num);
            g_print("Retrieved Decoder Metadata: frame type = %d, frame_num = %d decode_error_status = %d\n\n",
                decoder_meta->frame_type, decoder_meta->frame_num,
                decoder_meta->dec_err);
          }

          if(user_meta->base_meta.meta_type == NVDS_GST_META_BEFORE_DECODER_EXAMPLE)
          {
            h264parse_meta = (H264parseMeta *)user_meta->user_meta_data;
            g_print("Parser Meta retrieved For Frame_Num = %d  \n",
                h264parse_meta->parser_frame_num);
            g_print("Retrieved Metadata attached before decoder: parsed frame_num = %d\n\n",
                h264parse_meta->parser_frame_num);
          }
        }
    }
    return GST_PAD_PROBE_OK;
}

/* h264parse_src_pad_buffer_probe() will attach decoder metadata to gstreamer
 * buffer on src pad. The decoder can not attach to NvDsBatchMeta metadata because
 * batch level metadata is created by nvstreammux component. The decoder
 * component is present is before nvstreammmux. So it attached the metadata
 * using gstnvdsmeta API's.
 */
static GstPadProbeReturn
h264parse_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMeta *meta = NULL;

  H264parseMeta *h264parse_meta = (H264parseMeta *)g_malloc0(sizeof(H264parseMeta));
  if(h264parse_meta == NULL)
  {
    return GST_FLOW_ERROR;
  }
  /* Add dummy metadata */
  h264parse_meta->parser_frame_num = parsed_frame_number++;

  /* Attach decoder metadata to gst buffer using gst_buffer_add_nvds_meta() */
  meta = gst_buffer_add_nvds_meta (buf, h264parse_meta, NULL,
      h264parse_meta_copy_func, h264parse_meta_release_func);

  /* Set metadata type */
  meta->meta_type = (GstNvDsMetaType)NVDS_GST_META_BEFORE_DECODER_EXAMPLE;

  /* Set transform function to transform decoder metadata from Gst meta to
   * nvds meta */
  meta->gst_to_nvds_meta_transform_func = h264parse_gst_to_nvds_meta_transform_func;

  /* Set release function to release the transformed nvds metadata */
  meta->gst_to_nvds_meta_release_func = h264parse_gst_nvds_meta_release_func;

  g_print("GST H264parse Meta attached for Frame_Num = %d\n",
      h264parse_meta->parser_frame_num);
  g_print("Attached Metadata before decoder: Parsed frame_num = %d\n\n", h264parse_meta->parser_frame_num);

  return GST_PAD_PROBE_OK;
}

/* nvdecoder_src_pad_buffer_probe() will attach decoder metadata to gstreamer
 * buffer on src pad. The decoder can not attach to NvDsBatchMeta metadata because
 * batch level metadata is created by nvstreammux component. The decoder
 * component is present is before nvstreammmux. So it attached the metadata
 * using gstnvdsmeta API's.
 */
static GstPadProbeReturn
nvdecoder_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMeta *meta = NULL;

  NvDecoderMeta *decoder_meta = (NvDecoderMeta *)g_malloc0(sizeof(NvDecoderMeta));
  if(decoder_meta == NULL)
  {
    return GST_FLOW_ERROR;
  }
  /* Add dummy metadata */
  decoder_meta->frame_type = (frame_number % 3);
  decoder_meta->frame_num = frame_number++;
  decoder_meta->dec_err = ((frame_number % 4) / 3);

  /* Attach decoder metadata to gst buffer using gst_buffer_add_nvds_meta() */
  meta = gst_buffer_add_nvds_meta (buf, decoder_meta, NULL,
      decoder_meta_copy_func, decoder_meta_release_func);

  /* Set metadata type */
  meta->meta_type = (GstNvDsMetaType)NVDS_DECODER_GST_META_EXAMPLE;

  /* Set transform function to transform decoder metadata from Gst meta to
   * nvds meta */
  meta->gst_to_nvds_meta_transform_func = decoder_gst_to_nvds_meta_transform_func;

  /* Set release function to release the transformed nvds metadata */
  meta->gst_to_nvds_meta_release_func = decoder_gst_nvds_meta_release_func;

  g_print("GST Dec Meta attached with gst decoder output buffer for Frame_Num = %d\n",
      decoder_meta->frame_num);
  g_print("Attached decoder Metadata: frame type = %d, frame_num = %d decode_error_status = %d\n\n",
      decoder_meta->frame_type, decoder_meta->frame_num,
      decoder_meta->dec_err);

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
  GstPad *infer_src_pad = NULL;
  GstPad *decoder_src_pad = NULL;
  GstPad *h264parse_src_pad = NULL;
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

  /* Use nvinfer/nvinferserver to run inferencing on decoder's output,
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
    input_stream = argv[3];
    g_object_set (G_OBJECT (source), "location", argv[3], NULL);
  } else {
    input_stream = argv[1];
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);
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

  /* Lets add probe to set h264parse metadata attached with NvDsMeta.
   * This metadata is tranformed into nvdsmeta and set as user metadata at
   * frame level.
   * We add probe to the src pad of the h264parse element to demonstrate
   * propogation of metadata attached before nvv4l2decoder */
  h264parse_src_pad = gst_element_get_static_pad (h264parser, "src");
  if (!h264parse_src_pad)
    g_print ("Unable to get source pad of element %s\n", GST_ELEMENT_NAME(h264parser));
  else
    gst_pad_add_probe (h264parse_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        h264parse_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (h264parse_src_pad);

  /* Lets add probe to access decoder metadata attached with NvDsMeta.
   * This metadata is tranformed into nvdsmeta and set as user metadata at
   * frame level.
   * We add probe to the src pad of the decoder element */
  decoder_src_pad = gst_element_get_static_pad (decoder, "src");
  if (!decoder_src_pad)
    g_print ("Unable to get source pad\n");
  else
    gst_pad_add_probe (decoder_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        nvdecoder_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (decoder_src_pad);

  /* Lets add probe at decoder src pad to attach dummy decoder metadata using
   * gstnvdsmeta APIs.
   * This metadata is transformed into nvdsmeta and set as user metadata at
   * frame level */
  infer_src_pad = gst_element_get_static_pad (pgie, "src");
  if (!infer_src_pad)
    g_print ("Unable to get source pad\n");
  else
    gst_pad_add_probe (infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        nvinfer_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (infer_src_pad);

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
