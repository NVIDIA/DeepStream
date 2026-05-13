/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct
{
  GstElement *pipeline;
  GstElement *streammux;
  GstElement *sink;
  GstElement *pgie;
  GstElement *queue1;
  GstElement *queue2;
  GstElement *queue3;
  GstElement *queue4;
  GstElement *queue5;
  GstElement *queue6;
  GstElement *queue7;
  GstElement *queue8;
  GstElement *queue9;
  GstElement *nvtracker;
  GstElement *sgie1;
  GstElement *sgie2;
  GstElement *nvvidconv;
  GstElement *nvosd;
  GstElement *tiler;
  GstElement *nvdslogger;
  GstElement *nvdsxfer;
} AppCtx;

static AppCtx appctx;
AppCtx *app = NULL;
static GMainLoop *loop = NULL;
static gboolean cintr = FALSE;

/* gie_unique_id is one of the properties in the above dsmultigpu_sgie_config
 * files. These should be unique and known when we want to parse the Metadata
 * respective to the sgie labels. Ideally these should be read from the config
 * files but for brevity we ensure they are same. */
#define PGIE_CONFIG_FILE_YML   "dsmultigpu_pgie_config.yml"
#define SGIE1_CONFIG_FILE_YML  "dsmultigpu_sgie1_config.yml"
#define SGIE2_CONFIG_FILE_YML  "dsmultigpu_sgie2_config.yml"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

static gboolean ENABLE_DISPLAY = FALSE;

#if 0
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
#endif

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
    {
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
    }
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
    }
      break;
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
    }
      break;
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id = 0;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
    }
      break;
    case GST_MESSAGE_APPLICATION:
    {
      const GstStructure *s;
      s = gst_message_get_structure (msg);

      if (gst_structure_has_name (s, "NvGstAppInterrupt")) {
        g_print ("Terminating the pipeline ...\n");
        g_main_loop_quit (loop);
      }
    }
      break;
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
#if 0
  if (g_strrstr (name, "source") == name) {
        g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }
#endif
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
  if (ENABLE_DISPLAY) {
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  } else {
    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
  }

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
_intr_handler (int signum)
{
  struct sigaction action;

  g_print ("User Interrupted.. \n");
  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  cintr = TRUE;
}

static gboolean
check_for_interrupt (gpointer data)
{
  if (cintr) {
    cintr = FALSE;

    gst_element_post_message (GST_ELEMENT (app->pipeline),
        gst_message_new_application (GST_OBJECT (app->pipeline),
            gst_structure_new ("NvGstAppInterrupt",
                "message", G_TYPE_STRING, "Pipeline interrupted", NULL)));
    return FALSE;
  }
  return TRUE;
}

static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

int
main (int argc, char *argv[])
{
  app = &appctx;
  memset (app, 0, sizeof (AppCtx));

  GstBus *bus = NULL;
  guint bus_watch_id;
//  GstPad *tiler_src_pad = NULL;
  guint i =0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint streammux_batch_size;
  guint pgie_batch_size;
  guint nvdsxfer_position = 0;
  ENABLE_DISPLAY = g_getenv("NVDS_NVLINK_TEST_ENABLE_DISPLAY") &&
      !g_strcmp0(g_getenv("NVDS_NVLINK_TEST_ENABLE_DISPLAY"), "1");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

   _intr_setup ();
   g_timeout_add (400, check_for_interrupt, NULL);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  app->pipeline = gst_pipeline_new ("dsnvlinktest-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  app->streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!app->pipeline || !app->streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (app->pipeline), app->streammux);

  GList *src_list = NULL ;

  if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {

    nvds_parse_source_list(&src_list, argv[1], "source-list");

    GList * temp = src_list;
    while(temp) {
      num_sources++;
      temp=temp->next;
    }
    g_list_free(temp);
  } else {
    g_printerr ("No .yml/.yaml congig file found. Exiting.\n");
    return -1;
  }

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    GstElement *source_bin= NULL;
    g_print("Now playing : %s\n",(char*)(src_list)->data);
    source_bin = create_source_bin (i, (char*)(src_list)->data);
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (app->pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple (app->streammux, pad_name);
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

    src_list = src_list->next;
  }

  g_list_free(src_list);

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  app->pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use tracker to track the identified objects */
  app->nvtracker = gst_element_factory_make ("nvtracker", "tracker");

  /* Use three secondary gies so lets create 3 more instances of
     nvinfer */
  app->sgie1 = gst_element_factory_make ("nvinfer", "secondary1-nvinference-engine");

  app->sgie2 = gst_element_factory_make ("nvinfer", "secondary2-nvinference-engine");

  /* Use nvlink for multigpu usecase pipeline */
  app->nvdsxfer = gst_element_factory_make ("nvdsxfer", "Multi-GPU-transfer-element");

  /* Add queue elements between every two elements */
  app->queue1 = gst_element_factory_make ("queue", "queue1");
  app->queue2 = gst_element_factory_make ("queue", "queue2");
  app->queue3 = gst_element_factory_make ("queue", "queue3");
  app->queue4 = gst_element_factory_make ("queue", "queue4");
  app->queue5 = gst_element_factory_make ("queue", "queue5");
  app->queue6 = gst_element_factory_make ("queue", "queue6");
  app->queue7 = gst_element_factory_make ("queue", "queue7");
  app->queue8 = gst_element_factory_make ("queue", "queue8");
  app->queue9 = gst_element_factory_make ("queue", "queue9");

  /* Use nvdslogger for perf measurement. */
  app->nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  app->tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  app->nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  app->nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (ENABLE_DISPLAY) {
    /* Render the osd output if enable display is TRUE */
#ifdef __aarch64__
    app->sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    app->sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  } else {
    app->sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  }

  if (!app->pgie || !app->nvdsxfer || !app->nvtracker || !app->sgie1 ||
      !app->sgie2 || !app->nvdslogger || !app->tiler ||
      !app->nvvidconv || !app->nvosd || !app->sink || !app->queue1 ||
      !app->queue2 || !app->queue3 || !app->queue4 || !app->queue5 ||
      !app->queue6 || !app->queue7 || !app->queue8 || !app->queue9 ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* configure gpu-ids with nvdsxfer for P2P connection using nvlink */
  nvds_parse_nvxfer(app->nvdsxfer, argv[1],"nvxfer");
  if (ENABLE_DISPLAY) {
    /* configure gpu-ids with tiler,nvvidconv,nvosd,sink for use with nvsxfer */
    g_object_set (G_OBJECT (app->tiler), "gpu-id", 1, NULL);
    g_object_set (G_OBJECT (app->nvvidconv), "gpu-id", 1, NULL);
    g_object_set (G_OBJECT (app->nvosd), "gpu-id", 1, NULL);
    g_object_set (G_OBJECT (app->sink), "gpu-id", 1, NULL);
  }

  nvds_parse_streammux(app->streammux, argv[1], "streammux");
  g_object_get (G_OBJECT (app->streammux), "batch-size", &streammux_batch_size, NULL);
  if (streammux_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding streammux-config batch-size (%d) with number of sources (%d)\n",
        streammux_batch_size, num_sources);
    g_object_set (G_OBJECT (app->streammux), "batch-size", num_sources, NULL);
  }

  g_object_set (G_OBJECT (app->pgie),
    "config-file-path", PGIE_CONFIG_FILE_YML, NULL);
  g_object_get (G_OBJECT (app->pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (app->pgie), "batch-size", num_sources, NULL);
  }

  g_object_set (G_OBJECT (app->sgie1),
    "config-file-path", SGIE1_CONFIG_FILE_YML, NULL);
  g_object_set (G_OBJECT (app->sgie2),
    "config-file-path", SGIE2_CONFIG_FILE_YML, NULL);

  nvds_parse_tracker(app->nvtracker, argv[1], "tracker");

  if (ENABLE_DISPLAY) {
    nvds_parse_osd(app->nvosd, argv[1],"osd");

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    g_object_set (G_OBJECT (app->tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    nvds_parse_tiler(app->tiler, argv[1], "tiler");
    nvds_parse_egl_sink(app->sink, argv[1], "sink");
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (app->pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  nvds_parse_nvxfer_position (argv[1], "nvxfer", &nvdsxfer_position);
  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if (ENABLE_DISPLAY) {
    /* For display mode use nvdslogger with videosink */
    gst_bin_add_many (GST_BIN (app->pipeline), app->queue1, app->pgie, app->queue2,
                      app->nvdsxfer, app->queue3, app->nvtracker, app->queue4,
                      app->sgie1, app->queue5, app->sgie2, app->queue6,
                      app->nvdslogger, app->tiler, app->queue7, app->nvvidconv,
                      app->queue8, app->nvosd, app->queue9, app->sink, NULL);
    /* we link the elements together */
    switch (nvdsxfer_position) {
      case 0:
        /* nvstreammux -> nvinfer(PGIE) -> nvdsxfer -> nvtracker -> nvinfer(SGIE1) ->
         * nvinfer(SGIE2) -> nvdslogger -> nvtiler -> nvvidconv ->
         * nvosd -> video-renderer */
        if (!gst_element_link_many (app->streammux, app->queue1, app->pgie, app->queue2,
                          app->nvdsxfer, app->queue3, app->nvtracker, app->queue4,
                          app->sgie1, app->queue5, app->sgie2, app->queue6,
                          app->nvdslogger, app->tiler, app->queue7, app->nvvidconv,
                          app->queue8, app->nvosd, app->queue9, app->sink, NULL)) {
          g_printerr ("Elements could not be linked. Exiting.\n");
          return -1;
        }
        break;
      case 1:
        /* nvstreammux -> nvdsxfer -> nvinfer(PGIE) -> nvtracker -> nvinfer(SGIE1) ->
         * nvinfer(SGIE2) -> nvdslogger -> nvtiler -> nvvidconv ->
         * nvosd -> video-renderer */
        if (!gst_element_link_many (app->streammux, app->queue1, app->nvdsxfer, app->queue2,
                          app->pgie, app->queue3, app->nvtracker, app->queue4, app->sgie1,
                          app->queue5, app->sgie2, app->queue6, app->nvdslogger,
                          app->tiler, app->queue7, app->nvvidconv, app->queue8, app->nvosd,
                          app->queue9, app->sink, NULL)) {
          g_printerr ("Elements could not be linked. Exiting.\n");
          return -1;
        }
        break;
      case 2:
        /* nvstreammux -> nvinfer(PGIE) -> nvtracker -> nvdsxfer -> nvinfer(SGIE1) ->
         * nvinfer(SGIE2) -> nvdslogger -> nvtiler -> nvvidconv ->
         * nvosd -> video-renderer */
        if (!gst_element_link_many (app->streammux, app->queue1, app->pgie, app->queue2,
                          app->nvtracker, app->queue3, app->nvdsxfer, app->queue4,
                          app->sgie1, app->queue5, app->sgie2, app->queue6,
                          app->nvdslogger, app->tiler, app->queue7, app->nvvidconv,
                          app->queue8, app->nvosd, app->queue9, app->sink, NULL)) {
          g_printerr ("Elements could not be linked. Exiting.\n");
          return -1;
        }
        break;
      default:
          g_printerr ("nvxfer position not supported... Exiting.\n");
          return -1;
        break;
    }
  } else {
    /* For perf mode use nvdslogger with fakesink */
    gst_bin_add_many (GST_BIN (app->pipeline), app->queue1, app->pgie, app->queue2,
                      app->nvdsxfer, app->nvdslogger, app->queue3,  app->nvtracker,
                      app->queue4, app->sgie1, app->queue5, app->sgie2, app->queue6,
                      app->sink, NULL);
    /* we link the elements together */
    switch (nvdsxfer_position) {
      case 0:
        /* nvstreammux -> nvinfer(PGIE) -> nvdsxfer -> nvtracker -> nvinfer(SGIE1) ->
         * nvinfer(SGIE2) -> nvdslogger -> fakesink */
        if (!gst_element_link_many (app->streammux, app->queue1, app->pgie, app->queue2,
                                    app->nvdsxfer, app->queue3, app->nvtracker, app->queue4,
                                    app->sgie1, app->queue5, app->sgie2, app->queue6,
                                    app->nvdslogger, app->sink, NULL)) {
          g_printerr ("Elements could not be linked. Exiting.\n");
          return -1;
        }
        break;
      case 1:
        /* nvstreammux -> nvdsxfer -> nvinfer(PGIE) -> nvtracker -> nvinfer(SGIE1) ->
         * nvinfer(SGIE2) -> nvdslogger -> fakesink */
        if (!gst_element_link_many (app->streammux, app->queue1, app->nvdsxfer, app->queue2,
                                    app->pgie, app->queue3, app->nvtracker, app->queue4,
                                    app->sgie1, app->queue5, app->sgie2, app->queue6,
                                    app->nvdslogger, app->sink, NULL)) {
          g_printerr ("Elements could not be linked. Exiting.\n");
          return -1;
        }
        break;
      case 2:
        /* nvstreammux -> nvinfer(PGIE) -> nvtracker -> nvdsxfer -> nvinfer(SGIE1) ->
         * nvinfer(SGIE2) -> nvdslogger -> fakesink */
        if (!gst_element_link_many (app->streammux, app->queue1, app->pgie, app->queue2,
                                    app->nvtracker, app->queue3, app->nvdsxfer, app->queue4,
                                    app->sgie1, app->queue5, app->sgie2, app->queue6,
                                    app->nvdslogger, app->sink, NULL)) {
          g_printerr ("Elements could not be linked. Exiting.\n");
          return -1;
        }
        break;
      default:
          g_printerr ("nvxfer position not supported... Exiting.\n");
          return -1;
        break;
    }
  }

#if 0
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
#endif

  /* Set the pipeline to "playing" state */
  g_print ("Using file: %s\n", argv[1]);
  gst_element_set_state (app->pipeline, GST_STATE_PLAYING);

  /* Dump Capture - Playing Pipeline into the dot file
   * Set environment variable "export GST_DEBUG_DUMP_DOT_DIR=/tmp"
   * Run deepstream-multigpu-nvlink-test and 0.00.00.*-multigpu-nvlink-playing.dot
   * file will be generated.
   * Run "dot -Tpng 0.00.00.*-multigpu-nvlink-playing.dot > image.png"
   * image.png will display the running capture pipeline.
   * */
   GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN(app->pipeline),
       GST_DEBUG_GRAPH_SHOW_ALL, "multigpu-nvlink-playing");

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (app->pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (app->pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  return 0;
}
