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
#include <math.h>
#include <cuda_runtime_api.h>
#include "nvds_yml_parser.h"
#include "gst-nvevent.h"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

#define MEMORY_FEATURES "memory:NVMM"

GMainLoop *loop = NULL;
GstElement *pipeline = NULL, *bin1 = NULL, *nvmultiurisrcbin = NULL, *pgie = NULL, *tee = NULL, *queue_tee = NULL,
           *queue = NULL, *tiler = NULL, *nvosd = NULL, *sink = NULL, *nvstreamdemux = NULL, *bin2 = NULL,
           *vidconvert = NULL;
guint num_src_pad = 0;
guint max_batch_size = 0;
gboolean flag = FALSE;

GMutex pipeline_mutex;
struct cudaDeviceProp prop;
gboolean drop_pipeline_eos = FALSE;

static GstPadProbeReturn
demux_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

/**
 * Creates a new processing branch for a stream from nvstreamdemux.
 *
 * This function sets up a pipeline branch with the following structure:
 * nvstreamdemux_src_pad -> queue -> nvdsosd -> sink
 *
 * It's called for each new source stream to create a dedicated
 * processing path with on-screen display.
 */
static int
create_branch(int source_id)
{
    GstElement *queue_br = NULL, *nvdsosd = NULL, *sink_br = NULL;
    GstPad *src_pad = NULL, *sink_pad = NULL;
    gchar q_name[16], osd_name[16], sink_name[16], pad_name[16];
    snprintf (q_name, sizeof(q_name), "queue_%u", source_id);
    snprintf (osd_name, sizeof(osd_name), "nvosd_%u", source_id);
    snprintf (sink_name, sizeof(sink_name), "sink_%u", source_id);
    snprintf (pad_name, sizeof(pad_name), "src_%u", source_id);

    queue_br = gst_element_factory_make("queue", q_name);
    nvdsosd = gst_element_factory_make("nvdsosd", osd_name);
    // for rendering the osd output
    if(prop.integrated) {
      sink_br = gst_element_factory_make("nv3dsink", sink_name);
    } else {
      #ifdef __aarch64__
        sink_br = gst_element_factory_make ("nv3dsink", sink_name);
      #else
        sink_br = gst_element_factory_make ("nveglglessink", sink_name);
      #endif
    }
    if (!queue_br || !nvdsosd || !sink_br)
    {
      g_printerr("One element could not be created [demux]. Exiting...\n");
      return -1;
    }
    gst_bin_add_many(GST_BIN(bin2), queue_br, nvdsosd, sink_br, NULL);
    if (!gst_element_link_many(queue_br, nvdsosd, sink_br, NULL))
    {
       g_printerr("All elements could not be linked\n");
       return -1;
    }
    g_object_set (G_OBJECT(sink_br), "sync", FALSE, NULL);
    g_object_set (G_OBJECT(sink_br), "async", FALSE, NULL);

    gst_element_sync_state_with_parent(queue_br);
    gst_element_sync_state_with_parent(nvdsosd);
    gst_element_sync_state_with_parent(sink_br);

    src_pad = gst_element_request_pad_simple (nvstreamdemux, pad_name);
    sink_pad = gst_element_get_static_pad (queue_br, "sink");
    if(!src_pad)
    {
        g_printerr("src pad has not been created\n");
        return -1;
    }
    if(!sink_pad)
    {
        gst_object_unref(src_pad);
        g_printerr("sink pad has not been created\n");
        return -1;
    }
    if (gst_pad_link (src_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link. Exiting.\n");
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
        return -1;
    }

    gst_object_unref(src_pad);
    gst_object_unref(sink_pad);
    num_src_pad++;
    g_print ("create_branch: source_id: %d\t pad_name=%s\n", source_id, pad_name);
    return 0;
}

/**
 * Creates a new bin (bin2) containing a queue and nvstreamdemux.
 * This bin is then linked to the tee element in bin1.
 */
static int
add_new_bin(void) {
    g_mutex_lock (&pipeline_mutex);
    bin2 = gst_bin_new ("new_branch_bin");
    queue_tee = gst_element_factory_make("queue", "queue_tee");
    nvstreamdemux = gst_element_factory_make("nvstreamdemux", "demuxer");
    if (!queue_tee || !nvstreamdemux) {
        g_printerr ("One of the elements is not created");
    }
    gst_bin_add_many(GST_BIN(bin2), queue_tee, nvstreamdemux, NULL);
    gst_element_link_many (queue_tee, nvstreamdemux, NULL);
    /** Add bin2 to the main pipeline */
    gst_bin_add_many (GST_BIN(pipeline), bin2, NULL);

    GstPad *tee_src_pad = gst_element_request_pad_simple(tee, "src_1");
    GstPad *queue_sink_pad = gst_element_get_static_pad(queue_tee, "sink");

    GstPad* ghost_src_pad = gst_ghost_pad_new("src", tee_src_pad);
    gst_pad_set_active(ghost_src_pad, TRUE);
    if(!ghost_src_pad)
    {
        g_printerr("src_pad is not created..\n");
        return -1;
    }
    gst_element_add_pad (bin1, ghost_src_pad);
    GstPad *ghost_sink_pad = gst_ghost_pad_new("sink", queue_sink_pad);
    gst_pad_set_active (ghost_sink_pad, TRUE);
    gst_element_add_pad (bin2, ghost_sink_pad);
    if(gst_pad_link (ghost_src_pad, ghost_sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Tee and queue are not linked...\n");
        return -1;
    }
    gst_element_sync_state_with_parent (bin2);

    /* Add probe to demux sink pad for handling PAD_DELETED events */
    GstPad *demux_sinkpad = gst_element_get_static_pad(nvstreamdemux, "sink");
    if (demux_sinkpad) {
        gst_pad_add_probe(demux_sinkpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
            (GstPadProbeCallback)demux_pad_probe_cb, NULL, NULL);
        gst_object_unref(demux_sinkpad);
    }

    g_mutex_unlock (&pipeline_mutex);
  return 0;
}

/**
 * Detects new streams and handles them appropriately:
 * Creates a new branch if needed for upcoming streams
 */
static GstPadProbeReturn
src_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
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

        if (flag == FALSE)
        {
            flag = TRUE;
            add_new_bin();
        }
        create_branch(source_id);
    }
    return GST_PAD_PROBE_OK;
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

        GstElement *sink_elem = gst_bin_get_by_name(GST_BIN(bin2), sink_name);
        GstElement *queue_elem = gst_bin_get_by_name(GST_BIN(bin2), queue_name);
        GstElement *nvosd_elem = gst_bin_get_by_name(GST_BIN(bin2), nvosd_name);

        if (sink_elem && queue_elem && nvosd_elem) {
            gst_element_set_state(queue_elem, GST_STATE_NULL);
            gst_element_set_state(nvosd_elem, GST_STATE_NULL);
            gst_element_set_state(sink_elem, GST_STATE_NULL);
            gst_bin_remove(GST_BIN(bin2), sink_elem);
            gst_bin_remove(GST_BIN(bin2), queue_elem);
            gst_bin_remove(GST_BIN(bin2), nvosd_elem);
            gst_object_unref(sink_elem);
            gst_object_unref(queue_elem);
            gst_object_unref(nvosd_elem);

            gchar pad_name[16];
            snprintf(pad_name, sizeof(pad_name), "src_%u", source_id);
            GstPad *demux_srcpad = gst_element_get_static_pad(nvstreamdemux, pad_name);
            if (demux_srcpad) {
                gst_element_release_request_pad(nvstreamdemux, demux_srcpad);
                gst_object_unref(demux_srcpad);
            }
        } else {
            g_printerr("[cleanup_branch] Failed to get sink, queue, or nvosd from bin2\n");
            if (sink_elem) gst_object_unref(sink_elem);
            if (queue_elem) gst_object_unref(queue_elem);
            if (nvosd_elem) gst_object_unref(nvosd_elem);
        }
        num_src_pad--;
        g_mutex_unlock(&pipeline_mutex);
        g_print("Pad deleted: src_%u\n", source_id);
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
    default:
      break;
  }
  return TRUE;
}

int main(int argc, char *argv[])
{
    g_mutex_init (&pipeline_mutex);

    GstBus *bus = NULL;
    guint bus_watch_id;
    gboolean yaml_config = FALSE;
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
    int current_device = -1;
    cudaGetDevice(&current_device);
    cudaGetDeviceProperties(&prop, current_device);

    /* Check input arguments */
    if(argc != 2)
    {
        g_printerr ("Usage: %s <yml file>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init(&argc,&argv);
    loop = g_main_loop_new (NULL,FALSE);

    /* Parse inference plugin type */
    yaml_config = (g_str_has_suffix(argv[1],".yml") || g_str_has_suffix(argv[1],".yaml"));
    if(yaml_config)
    {
        RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type,argv[1],"primary-gie"));
    }


    pipeline = gst_pipeline_new ("demuxer-pipeline");
    bin1 = gst_bin_new ("bin1");
    nvmultiurisrcbin = gst_element_factory_make ("nvmultiurisrcbin", "source");

    if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
        pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
    } else {
        pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    }
    tee = gst_element_factory_make ("tee", "tee");

    if(!pipeline || !nvmultiurisrcbin || !pgie || !tee)
    {
        g_printerr("One element could not be created..\n");
        return -1;
    }

    if(yaml_config)
    {
        RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie,argv[1],"primary-gie"));
        RETURN_ON_PARSER_ERROR(nvds_parse_multiurisrcbin(nvmultiurisrcbin,argv[1],"source"));
    }

    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    gst_bin_add_many (GST_BIN(bin1), nvmultiurisrcbin, pgie, tee, NULL);
    if (!gst_element_link_many (nvmultiurisrcbin, pgie, tee, NULL))
    {
        g_printerr ("Elements could not be linked.\n");
        return -1;
    }

    queue = gst_element_factory_make ("queue", "queue");
    tiler = gst_element_factory_make ("nvmultistreamtiler", "tiler");
    /* There is a known issue where bbox are incorrect in case were demuxer & tiler are used along with tee.
      WAR: to use videoconvert as identity element for correct bbox */
    vidconvert = gst_element_factory_make ("identity", "vidconv_identity");
    nvosd = gst_element_factory_make ("nvdsosd", "nvosd");
    if(prop.integrated) {
      sink = gst_element_factory_make("nv3dsink", "sink");
    } else {
      #ifdef __aarch64__
        sink = gst_element_factory_make ("nv3dsink", "sink");
      #else
        sink = gst_element_factory_make ("nveglglessink", "sink");
      #endif
    }

    g_object_set (G_OBJECT(sink), "sync", FALSE, NULL);
    g_object_set (G_OBJECT(sink), "async", FALSE, NULL);
    if (!queue || !tiler || !nvosd || !sink || !vidconvert)
    {
        g_printerr("One element could not be created...[2]\n");
        return -1;
    }
    gst_bin_add_many (GST_BIN(bin1), queue, vidconvert, tiler, nvosd, sink, NULL);
    if (!gst_element_link_many (queue, vidconvert, tiler, nvosd, sink, NULL))
    {
        g_printerr ("Element could not be linked...[2]\n");
        return -1;
    }

    GstPad *tee_srcpad = gst_element_request_pad_simple (tee, "src_0");
    GstPad *queue_sinkpad = gst_element_get_static_pad (queue,"sink");

    if (gst_pad_link (tee_srcpad, queue_sinkpad) != GST_PAD_LINK_OK)
    {
        g_printerr("Pads could not be linked...\n");
        return -1;
    }
    gst_object_unref (tee_srcpad);
    gst_object_unref (queue_sinkpad);

    g_object_get (G_OBJECT(nvmultiurisrcbin), "drop-pipeline-eos", &drop_pipeline_eos, NULL);

    /* Add probe for stream start events */
    GstPad *srcpad = gst_element_get_static_pad (nvmultiurisrcbin, "src");
    if (srcpad) {
        gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                          (GstPadProbeCallback) src_pad_probe_cb, NULL, NULL);
        gst_object_unref(srcpad);
    } else {
        g_printerr("Failed to get source pad from nvmultiurisrcbin\n");
        return -1;
    }

    gst_bin_add (GST_BIN(pipeline), bin1);
    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
    g_object_get (G_OBJECT(nvmultiurisrcbin), "max-batch-size", &max_batch_size, NULL);

    g_print ("Using file: %s\n", argv[1]);
    /* Set the pipeline to "playing" state */
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
    g_mutex_clear(&pipeline_mutex);

    return 0;
}