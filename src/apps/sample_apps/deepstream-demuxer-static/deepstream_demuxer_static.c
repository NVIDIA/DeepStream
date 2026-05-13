/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvds_yml_parser.h"
#include "gst-nvevent.h"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

GstElement *pipeline = NULL, *nvmultiurisrcbin = NULL, *nvstreamdemux=NULL, *pgie = NULL;
static guint num_src_pad = 0;
static guint max_batch_size = 0;
GMainLoop *loop = NULL;
GMutex pipeline_mutex;
struct cudaDeviceProp prop;
gboolean drop_pipeline_eos = FALSE;

/**
* Creates a new processing branch for a stream from nvstreamdemux.
*
* This function sets up a pipeline branch with the following structure:
* nvstreamdemux_src_pad -> queue -> nvdsosd -> sink
*
* It's called for each new source stream to create a dedicated
* processing path with on-screen display.
*
*/
static int
create_branch(int source_id)
{
  GstElement *queue=NULL, *nvosd=NULL, *sink=NULL;
  GstPad *src_pad=NULL, *sink_pad=NULL;
  gchar q_name[16], osd_name[16], sink_name[16], pad_name[16];
  snprintf (q_name, sizeof(q_name), "queue_%u", source_id);
  snprintf (osd_name, sizeof(osd_name), "nvosd_%u", source_id);
  snprintf (sink_name, sizeof(sink_name), "sink_%u", source_id);
  snprintf (pad_name, sizeof(pad_name), "src_%u", source_id);
  queue = gst_element_factory_make("queue", q_name);
  nvosd = gst_element_factory_make("nvdsosd", osd_name);

  // for rendering the osd output
  if(prop.integrated) {
    sink = gst_element_factory_make("nv3dsink", sink_name);
  } else {
    #ifdef __aarch64__
      sink = gst_element_factory_make ("nv3dsink", sink_name);
    #else
      sink = gst_element_factory_make ("nveglglessink", sink_name);
    #endif
  }

  if (!queue || !nvosd || !sink)
  {
    g_printerr("One element could not be created [demux]. Exiting...\n");
    return -1;
  }

  gst_bin_add_many (GST_BIN(pipeline), queue, nvosd, sink, NULL);
  if (!gst_element_link_many (queue, nvosd, sink, NULL))
  {
    g_printerr ("All elements could not be linked\n");
      return -1;
  }

  g_object_set (G_OBJECT(sink), "sync", FALSE, NULL);
  g_object_set (G_OBJECT(sink), "async", FALSE, NULL);

  gst_element_sync_state_with_parent(queue);
  gst_element_sync_state_with_parent(nvosd);
  gst_element_sync_state_with_parent(sink);

  src_pad = gst_element_request_pad_simple (nvstreamdemux, pad_name);
  sink_pad = gst_element_get_static_pad (queue, "sink");

  if (!src_pad)
  {
    g_printerr ("src pad has not been created\n");
    return -1;
  }
  if (!sink_pad)
  {
    gst_object_unref(src_pad);
    g_printerr ("sink pad has not been created\n");
    return -1;
  }
  if (gst_pad_link (src_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link. Exiting.\n");
    gst_object_unref(src_pad);
    gst_object_unref(sink_pad);
    return -1;
  }

  gst_object_unref(src_pad);
  gst_object_unref(sink_pad);
  num_src_pad++;
  return 0;
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
    case GST_MESSAGE_WARNING:{
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_warning(msg, &error, &debug);
      g_printerr("WARNING from element %s: %s\n",
          GST_OBJECT_NAME(msg->src), error->message);
      if (debug)
        g_printerr("Debug info: %s\n", debug);
      g_free(debug);
      g_error_free(error);
      break;
    }
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

static GstPadProbeReturn
demux_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
  if((guint)GST_EVENT_TYPE (event) == (guint)GST_NVEVENT_PAD_DELETED)
  {
    g_mutex_lock(&pipeline_mutex);
    if((!drop_pipeline_eos) && (num_src_pad == 1))
    {
      g_mutex_unlock(&pipeline_mutex);
      return GST_PAD_PROBE_OK;
    }
    guint source_id = 0;
    gst_nvevent_parse_pad_deleted(event, &source_id);
    gchar sink_name[16], queue_name[16], nvosd_name[16];
    snprintf(sink_name, sizeof(sink_name), "sink_%u", source_id);
    snprintf(queue_name, sizeof(queue_name), "queue_%u", source_id);
    snprintf(nvosd_name, sizeof(nvosd_name), "nvosd_%u", source_id);

    GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), sink_name);
    GstElement *queue = gst_bin_get_by_name(GST_BIN(pipeline), queue_name);
    GstElement *nvosd = gst_bin_get_by_name(GST_BIN(pipeline), nvosd_name);

    if (sink && queue && nvosd) {
      gst_element_set_state(queue, GST_STATE_NULL);
      gst_element_set_state(nvosd, GST_STATE_NULL);
      gst_element_set_state(sink, GST_STATE_NULL);
      gst_bin_remove(GST_BIN(pipeline), sink);
      gst_bin_remove(GST_BIN(pipeline), queue);
      gst_bin_remove(GST_BIN(pipeline), nvosd);
      gst_object_unref(sink);
      gst_object_unref(queue);
      gst_object_unref(nvosd);

      GstElement *demuxer = gst_bin_get_by_name(GST_BIN(pipeline), "nvstreamdemux");
      if (demuxer) {
        gchar pad_name[16];
        snprintf(pad_name, sizeof(pad_name), "src_%u", source_id);
        GstPad *demux_srcpad = gst_element_get_static_pad(demuxer, pad_name);
        if (demux_srcpad) {
          gst_element_release_request_pad(demuxer, demux_srcpad);
          gst_object_unref(demux_srcpad);
        }
        gst_object_unref(demuxer);
      }
    } else {
      g_printerr("[cleanup_branch] Failed to get sink, queue, or nvosd from pipeline\n");
      if (sink) gst_object_unref(sink);
      if (queue) gst_object_unref(queue);
      if (nvosd) gst_object_unref(nvosd);
    }
    num_src_pad--;
    g_mutex_unlock(&pipeline_mutex);
    g_print("Pad deleted: src_%u\n", source_id);
  }
  return GST_PAD_PROBE_OK;
}
/**
 * Detects new streams and handles them appropriately:
 * Creates a new branch if needed for upcoming streams
 */
static GstPadProbeReturn
pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
  /* This event indicates the beginning of a new stream, making it
   * the ideal point to detect and handle new input sources. */
  if((guint)GST_EVENT_TYPE(event) == (guint)GST_NVEVENT_PAD_ADDED)
  {
    guint source_id = 0;
    gst_nvevent_parse_pad_added(event, &source_id);
    gchar pad_name[16];
    snprintf(pad_name, sizeof(pad_name), "src_%u", source_id);
    g_print("Pad added: %s\n", pad_name);
    create_branch(source_id);
  }
  return GST_PAD_PROBE_OK;
}

int
main (int argc, char *argv[])
{
  g_mutex_init(&pipeline_mutex);
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

  int current_device = -1;
  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc != 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
  }

  /* Create pipeline and source element */
  pipeline = gst_pipeline_new ("demuxer-pipeline");
  nvmultiurisrcbin = gst_element_factory_make ("nvmultiurisrcbin", "source");
  GstPad *srcpad = NULL;
  if (!pipeline || !nvmultiurisrcbin) {
    g_printerr ("One element could not be created. Exiting.\n");
    goto cleanup;
  }

  /* Create inference engine based on configuration */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }

  /* Create stream demuxer */
  nvstreamdemux = gst_element_factory_make("nvstreamdemux","nvstreamdemux");

  if (!nvstreamdemux || !pgie ) {
    g_printerr ("One element could not be created. Exiting.\n");
    goto cleanup;
  }

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_multiurisrcbin(nvmultiurisrcbin, argv[1],"source"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));
  }

  /* Add bus message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the main pipeline elements */
  gst_bin_add_many (GST_BIN (pipeline),
      nvmultiurisrcbin, pgie, nvstreamdemux, NULL);

  if (!gst_element_link_many (nvmultiurisrcbin, pgie, nvstreamdemux, NULL)) {
    g_printerr ("Elements could not be linked: Exiting.\n");
    goto cleanup;
  }

  /* Process initial URI list and create branches */
  gchar *uri_list = NULL;
  g_object_get (G_OBJECT(nvmultiurisrcbin), "uri-list", &uri_list, NULL);
  g_object_get (G_OBJECT(nvmultiurisrcbin), "drop-pipeline-eos", &drop_pipeline_eos, NULL);
  /* Add probe for stream start events */
  srcpad = gst_element_get_static_pad(nvmultiurisrcbin, "src");
  if (srcpad) {
    gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        (GstPadProbeCallback)pad_probe_cb, NULL, NULL);
    gst_object_unref(srcpad);
    srcpad = NULL;
  } else {
    g_printerr("Failed to get source pad from nvmultiurisrcbin\n");
    goto cleanup;
  }

  GstPad *demux_sinkpad = gst_element_get_static_pad(nvstreamdemux, "sink");
  if (demux_sinkpad) {
    gst_pad_add_probe(demux_sinkpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        (GstPadProbeCallback)demux_pad_probe_cb, NULL, NULL);
    gst_object_unref(demux_sinkpad);
  } else {
    g_printerr("Failed to get sink pad from nvstreamdemux\n");
    goto cleanup;
  }
  /* Get maximum batch size */
  g_object_get (G_OBJECT(nvmultiurisrcbin), "max-batch-size", &max_batch_size, NULL);

  /* Set the pipeline to "playing" state */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

cleanup:
  /* Clean up resources */
  g_print ("Cleaning up resources...\n");

  /* Stop and clean up the pipeline */
  if (pipeline) {
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
  }
  /* Remove bus watch */
  if (bus_watch_id > 0) {
    g_source_remove(bus_watch_id);
  }
  /* Unreference the main loop */
  if (loop) {
    g_main_loop_unref(loop);
    loop = NULL;
  }
  g_mutex_clear(&pipeline_mutex);
  return 0;
}
