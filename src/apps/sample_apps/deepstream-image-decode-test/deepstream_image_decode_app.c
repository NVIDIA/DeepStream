/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvbufsurface.h"

#include "gstnvdsmeta.h"
//#include "gstnvstreammeta.h"
#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

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

#define TILED_OUTPUT_WIDTH 1920
#define TILED_OUTPUT_HEIGHT 1080

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"
#define PGIE_CONFIG_FILE  "dstest_image_decode_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dstest_image_decode_pgie_nvinferserver_config.txt"

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

/* Structure to hold source bin data */
typedef struct {
  GstElement *bin;
  GstElement *nvvideoconvert;
} SourceBinData;

#define FPS_PRINT_INTERVAL 300
//static struct timeval start_time = { };

//static guint probe_counter = 0;

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
#if 0
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

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

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
#endif

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
  SourceBinData *bin_data = (SourceBinData *) data;
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  gchar *caps_string = gst_caps_to_string(caps);
  g_free(caps_string);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Link uridecodebin to nvvideoconvert */
      GstPad *conv_sink_pad = gst_element_get_static_pad (bin_data->nvvideoconvert, "sink");
      if (gst_pad_link (decoder_src_pad, conv_sink_pad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link uridecodebin to nvvideoconvert\n");
      } else {

        /* Now link nvvideoconvert src pad to ghost pad */
        GstPad *conv_src_pad = gst_element_get_static_pad (bin_data->nvvideoconvert, "src");
        GstPad *bin_ghost_pad = gst_element_get_static_pad (bin_data->bin, "src");

        if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad), conv_src_pad)) {
          g_printerr ("Failed to link nvvideoconvert src pad to source bin ghost pad\n");
        }

        gst_object_unref (conv_src_pad);
        gst_object_unref (bin_ghost_pad);
      }
      gst_object_unref (conv_sink_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }

  if (caps)
    gst_caps_unref (caps);
}

/* Callback to clean up bin_data when bin is destroyed */
static void
source_bin_destroy_cb (gpointer data)
{
  SourceBinData *bin_data = (SourceBinData *) data;
  g_free (bin_data);
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }

  if (g_object_class_find_property(G_OBJECT_GET_CLASS(object),
    "gpu-id")) {
    g_object_set (object, "gpu-id", 0, NULL);
    g_print ("Set gpu-id=0 for %s\n", name);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL;
  GstElement *uri_decode_bin = NULL;
  GstElement *nvvideoconvert = NULL;
  SourceBinData *bin_data = NULL;
  gchar bin_name[16] = { };
  gchar conv_name[32] = { };

  NvBufSurfaceDeviceInfo dev_info;
  NvBufSurfaceGetDeviceInfo(&dev_info);

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  g_snprintf (conv_name, 31, "nvvideoconvert-%02d", index);

  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Create uridecodebin to replace filesrc/multifilesrc + jpegparse + nvv4l2decoder */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  /* Create nvvideoconvert for format conversion */
  nvvideoconvert = gst_element_factory_make ("nvvideoconvert", conv_name);

  if (!uri_decode_bin || !nvvideoconvert)
  {
    g_printerr ("One element could not be created in source bin. Exiting.\n");
    return NULL;
  }

  /* Create structure to pass both bin and nvvideoconvert to callback */
  bin_data = g_new0 (SourceBinData, 1);
  bin_data->bin = bin;
  bin_data->nvvideoconvert = nvvideoconvert;

  /* Convert file path to URI if it's not already a URI */
  gchar *uri_str = NULL;
  if (g_str_has_prefix (uri, "file://") || g_str_has_prefix (uri, "http://") ||
      g_str_has_prefix (uri, "https://") || g_str_has_prefix (uri, "rtsp://")) {
    uri_str = g_strdup (uri);
  } else {
    /* Convert local file path to file:// URI */
    uri_str = g_filename_to_uri (uri, NULL, NULL);
  }
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri_str, NULL);
  g_free (uri_str);

  /* Add both elements to the bin */
  gst_bin_add_many (GST_BIN (bin), uri_decode_bin, nvvideoconvert, NULL);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the nvvideoconvert src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will link uridecodebin -> nvvideoconvert and set
   * the ghost pad target to the nvvideoconvert src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    g_free (bin_data);
    return NULL;
  }

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin_data);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin_data);

  /* Set up cleanup callback for when the bin is destroyed */
  g_object_weak_ref (G_OBJECT (bin), (GWeakNotify) source_bin_destroy_cb, bin_data);

  return bin;
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <media_file1|URI1> <media_file2|URI2> ...\n", bin);
  g_printerr ("       Supports various media formats (JPEG, MP4, etc.) and URI schemes (file://, http://, rtsp://, etc.)\n");
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <media_file1|URI1> <media_file2|URI2> ...\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *nvvideoconv = NULL, *pgie = NULL,
      *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean is_nvinfer_server = FALSE;
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

  if (is_nvinfer_server) {
    num_sources = argc - 3;
  } else {
    num_sources = argc - 1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest-image-decode-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    GstElement *source_bin;
    gchar pad_name[16] = { };
    if (is_nvinfer_server) {
      source_bin = create_source_bin (i, argv[i + 3]);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }

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

  /* Use convertor to convert to appropriate format */
  nvvideoconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make (
        is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN,
        "primary-nvinference-engine");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if(prop.integrated) {
    sink= gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }
  g_object_set (G_OBJECT (sink), "sync", 0, NULL);

  if (!nvvideoconv || !pgie || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (!use_new_mux) {
  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }
  else {
    g_object_set (G_OBJECT (streammux), "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }

  /* Configure the nvinfer/nvinferserver element using the config file. */
  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_NVINFERSERVER_CONFIG_FILE, NULL);
  } else {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  }

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

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), nvvideoconv, pgie, tiler, nvvidconv, nvosd, sink,
      NULL);
  /* we link the elements together
  * nvstreammux -> nvvideoconv -> pgie -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, nvvideoconv, pgie, tiler, nvvidconv, nvosd, sink,
          NULL)) {
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
  g_print ("Now playing...");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Generate GST debug dot files for pipeline visualization */
  if (g_getenv("GST_DEBUG_DUMP_DOT_DIR")) {
    GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(pipeline),
                                       GST_DEBUG_GRAPH_SHOW_ALL,
                                       "deepstream-image-decode-playing");
  }

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
