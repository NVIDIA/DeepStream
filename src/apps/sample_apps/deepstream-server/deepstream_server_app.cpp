/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "rest_server_callbacks.h"
#include "gst-nvmessage.h"
#include "gst-nvdscustommessage.h"
#include "gst-nvevent.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

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

static gboolean PERF_MODE = FALSE;

GstPadProbeReturn
pad_probe_event_on_fakesink (GstPad * pad, GstPadProbeInfo * info,
    gpointer user_data);

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
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  //NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    //int offset = 0;
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
//Enable below #if to see frame count and object information
#if 0
    g_print ("Frame Number = %d Number of objects = %d "
        "Vehicle Count = %d Person Count = %d\n",
        frame_meta->frame_num, num_rects, vehicle_count, person_count);
#endif

#if 0
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params;
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
#endif

  }
  return GST_PAD_PROBE_OK;
}

GstPadProbeReturn
pad_probe_event_on_fakesink (GstPad * pad, GstPadProbeInfo * info,
    gpointer user_data)
{
  GstEvent *
      event = (GstEvent *) info->data;

  if (event){
    if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_STREAM_EOS){
      guint source_id = 0;
      gst_nvevent_parse_stream_eos (event, &source_id);
      g_print("Received event EOS for source id %d \n",source_id);
    }
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
      if (gst_nvmessage_is_force_pipeline_eos (msg)) {
        gboolean app_quit = false;
        if (gst_nvmessage_parse_force_pipeline_eos (msg, &app_quit)) {
          if (app_quit)
            g_main_loop_quit (loop);

        }
      }

      break;

    }
    default:
      break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  AppCtx appctx = {0};
  appctx.sourceIdCounter = 0;
  /*Default httpport is set to 0 to disable REST API Server */
  appctx.httpPort = 0;
  g_mutex_init (&appctx.bincreator_lock);

  GMainLoop *loop = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  GstPad * sink_pad = NULL;

  // Note: Below environment variable will be removed once REST  endpoint enable/disable
  // is imeplemented and verified
  g_setenv("NVDS_LATENCY_MEASUREMENT_SILENT", "1", TRUE);
  // OSS bug due to libgioproxy.so. which causes abort while adding rtsp streams.
  // Set the GIO_MODULE_DIR  to /nonexistent to resolve this OSS issue.
  g_setenv("GIO_MODULE_DIR", "/nonexistent", TRUE);
  gboolean rest_server_within_multiurisrcbin = FALSE;
  nvds_parse_check_rest_server_with_app(argv[1],"rest-server",&rest_server_within_multiurisrcbin);

  if (!rest_server_within_multiurisrcbin) {
    nvds_parse_server_appctx(argv[1],"server-app-ctx", &appctx);

    NvDsServerCallbacks server_cb = {};
    /* Set REST Server callbacks */
    g_print ("Setting rest server callbacks \n");
    server_cb.stream_cb = [&appctx](NvDsServerStreamInfo *stream_info, void *ctx){
      s_stream_callback_impl(stream_info, (void*)&appctx);
    };
    server_cb.roi_cb= [&appctx](NvDsServerRoiInfo *roi_info, void *ctx){
      s_roi_callback_impl(roi_info, (void*)&appctx);
    };
    server_cb.dec_cb= [&appctx](NvDsServerDecInfo *dec_info, void *ctx){
      s_dec_callback_impl(dec_info, (void*)&appctx);
    };
    server_cb.infer_cb= [&appctx](NvDsServerInferInfo *infer_info, void *ctx){
      s_infer_callback_impl(infer_info, (void*)&appctx);
    };
    server_cb.inferserver_cb= [&appctx](NvDsServerInferServerInfo *inferserver_info, void *ctx){
      s_inferserver_callback_impl(inferserver_info, (void*)&appctx);
    };
    server_cb.conv_cb= [&appctx](NvDsServerConvInfo *conv_info, void *ctx){
      s_conv_callback_impl(conv_info, (void*)&appctx);
    };
    server_cb.enc_cb= [&appctx](NvDsServerEncInfo *enc_info, void *ctx){
      s_enc_callback_impl(enc_info, (void*)&appctx);
    };
    server_cb.mux_cb= [&appctx](NvDsServerMuxInfo *mux_info, void *ctx){
      s_mux_callback_impl(mux_info, (void*)&appctx);
    };
    server_cb.osd_cb =[&appctx] (NvDsServerOsdInfo * osd_info, void *ctx) {
      s_osd_callback_impl (osd_info, (void *)&appctx);
    };
    server_cb.appinstance_cb = [&appctx](NvDsServerAppInstanceInfo * appinstance_info, void *ctx) {
      s_appinstance_callback_impl (appinstance_info, (void *)&appctx);
    };

    appctx.server_conf.ip = appctx.httpIp;
    appctx.server_conf.port = appctx.httpPort;
    g_print ("Calling nvds_rest_server_start from the server app \n");
    appctx.restServer = (void*)nvds_rest_server_start(&appctx.server_conf,&server_cb,NULL);
  }

  guint tiler_rows, tiler_columns;
  PERF_MODE = g_getenv ("NVDS_SERVER_APP_PERF_MODE") &&
      !g_strcmp0 (g_getenv ("NVDS_SERVER_APP_PERF_MODE"), "1");

  gchar *nvdspreprocess_config_file = (gchar *) "config_preprocess.txt";

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
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
    RETURN_ON_PARSER_ERROR (nvds_parse_gie_type (&pgie_type, argv[1],
            "primary-gie"));
  }

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  appctx.pipeline = gst_pipeline_new ("dsserver-pipeline");
  if (!appctx.pipeline) {
    g_printerr ("dsserver-pipeline element could not be created. Exiting.\n");
    return -1;
  }

  if(appctx.restServer) {
    /* Create nvmultiurisrcbin instance using helper APIs to use REST server feature. */
    g_print ("Calling nvmultiurisrcbincreator API \n");

    /* NOTE: Use "app/quit" REST API to exit the app when no_pipeline_eos is set to TRUE.
             If no_pipeline_eos is set to FALSE then EOS is handled normally to exit app */
    /* Create/Add the nvmultiurisrcbincreator instance */
    appctx.nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(0, NVDS_MULTIURISRCBIN_MODE_VIDEO, &appctx.muxConfig);
    if (!appctx.nvmultiurisrcbinCreator) {
      g_printerr ("gst_nvmultiurisrcbincreator_init failed. Exiting.\n");
      return -1;
    }

    GstDsNvUriSrcConfig sourceConfig;
    memset(&sourceConfig, 0, sizeof(GstDsNvUriSrcConfig));
    sourceConfig.sensorId = NULL;
    sourceConfig.uri = appctx.uri_list;
    sourceConfig.source_id = 0;
    sourceConfig.disable_passthrough = TRUE;
    if (!gst_nvmultiurisrcbincreator_add_source(appctx.nvmultiurisrcbinCreator, &sourceConfig)) {
      g_printerr ("gst_nvmultiurisrcbincreator_add_source failed. Exiting.\n");
      return -1;
    }
  } else {
    /* Create nvmultiurisrcbin instance to use REST server feature. */
    g_print ("Calling gst_element_factory_make for nvmultiurisrcbin \n");
    appctx.multiuribin = gst_element_factory_make ("nvmultiurisrcbin", "multiuribin");
    if (!appctx.multiuribin) {
      g_printerr ("One element multiuribin could not be created. Exiting.\n");
      return -1;
    }
    nvds_parse_multiurisrcbin (appctx.multiuribin, argv[1], "multiurisrcbin");
  }

  NvDsYamlCodecStatus codec_status;
  nvds_parse_codec_status (argv[1], "encoder", &codec_status);

  gboolean enc_enable = codec_status.enable;

  if (enc_enable) {
    appctx.nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-2");
    if (codec_status.codec_type == 1){
      appctx.encoder = gst_element_factory_make("nvv4l2h264enc", "nvv4l2h264encoder");
      appctx.parser = gst_element_factory_make("h264parse", "h264parse");
    } else if (codec_status.codec_type == 2){
      appctx.encoder = gst_element_factory_make("nvv4l2h265enc", "nvv4l2h265encoder");
      appctx.parser = gst_element_factory_make("h265parse", "h265parse");
    }
    else if (codec_status.codec_type > 2 || codec_status.codec_type < 1)  {
      g_printerr ("Invalid codec type used in the config file. Use  codec =1 H264, codec = 2 H265 in the config \n");
      return -1;
    }

    appctx.queue_post_encoder = gst_element_factory_make("queue", "queue-post-encoder");

    if (!appctx.nvvidconv2 || !appctx.encoder  || !appctx.parser || !appctx.queue_post_encoder) {
      g_printerr ("One element could not be created in encoder path. Exiting.\n");
      return -1;
    }
  }

  if(appctx.restServer) {
    gst_bin_add_many (GST_BIN (appctx.pipeline), gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator), NULL);
  } else {
    gst_bin_add_many (GST_BIN (appctx.pipeline), appctx.multiuribin, NULL);
  }

  appctx.preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

  g_object_set(G_OBJECT(appctx.preprocess), "config-file", nvdspreprocess_config_file, NULL);

  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    appctx.pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    appctx.pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }

  gboolean analytics_enabled = check_enable_status(argv[1], "analytics");
  if (!analytics_enabled) {
    appctx.nvanalytics = gst_element_factory_make("identity", "analytics");
  } else {
    appctx.nvanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
    if (!appctx.nvanalytics) {
      g_printerr("One element could not be created. Exiting.\n");
      return -1;
    }
    nvds_parse_nvdsanalytics(appctx.nvanalytics, argv[1], "analytics");
  }
  /* Add queue elements between every two elements */
  appctx.queue1 = gst_element_factory_make ("queue", "queue1");
  appctx.queue2 = gst_element_factory_make ("queue", "queue2");
  appctx.queue3 = gst_element_factory_make ("queue", "queue3");
  appctx.queue4 = gst_element_factory_make ("queue", "queue4");
  appctx.queue5 = gst_element_factory_make ("queue", "queue5");
  appctx.queue6 = gst_element_factory_make ("queue", "queue6"); // New queue for analytics

  /* Use nvdslogger for perf measurement. */

  appctx.nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
  g_object_set (G_OBJECT (appctx.nvdslogger),
      "fps-measurement-interval-sec", 1, NULL);
  g_object_set (G_OBJECT (appctx.nvdslogger),
      "show-fps", 1, NULL);


  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  appctx.tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  appctx.nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  appctx.nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (PERF_MODE) {
    g_print ("PERF_MODE Enabled\n");
    appctx.sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else {
    /* Finally render the osd output */
    if(prop.integrated) {
      appctx.sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    } else {
      if (!enc_enable) {
#ifdef __aarch64__
        appctx.sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
        appctx.sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
      } else {
        appctx.sink = gst_element_factory_make("filesink", "file-sink");
        if (codec_status.codec_type == 1) {
          g_object_set (G_OBJECT (appctx.sink),
            "location", "out.h264", NULL);
          g_object_set (G_OBJECT (appctx.sink),
            "sync", 1, NULL);
        } else if (codec_status.codec_type == 2) {
          g_object_set (G_OBJECT (appctx.sink),
            "sync", 1, NULL);
        }
      }
    }
  }

  if (!appctx.preprocess || !appctx.pgie || !appctx.nvdslogger || !appctx.tiler || !appctx.nvvidconv || !appctx.nvosd || !appctx.sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (appctx.pgie),
   "config-file-path", "dsserver_pgie_config.txt", NULL);

  if (yaml_config) {
    /* Set all the necessary properties of the inference element */
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(appctx.pgie, argv[1], "primary-gie"));
  }
  guint multiurisrcbin_max_bs = 0;
  if(appctx.restServer) {
    multiurisrcbin_max_bs = appctx.muxConfig.maxBatchSize;
  } else {
    g_object_get (appctx.multiuribin, "max-batch-size", &multiurisrcbin_max_bs, NULL);
  }
  g_object_set (G_OBJECT (appctx.pgie), "batch-size", multiurisrcbin_max_bs, NULL);

  nvds_parse_osd(appctx.nvosd, argv[1],"osd");

  tiler_rows = (guint) sqrt (multiurisrcbin_max_bs);
  tiler_columns = (guint) ceil (1.0 * multiurisrcbin_max_bs / tiler_rows);
  g_object_set (G_OBJECT (appctx.tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

  nvds_parse_tiler(appctx.tiler, argv[1], "tiler");
  if (!enc_enable) {
    if(prop.integrated) {
      nvds_parse_3d_sink(appctx.sink, argv[1], "sink");
    }
    else {
       if (PERF_MODE) {
        nvds_parse_fake_sink (appctx.sink, argv[1], "sink");
      } else {
#ifdef __aarch64__
        nvds_parse_3d_sink(appctx.sink, argv[1], "sink");
#else
        nvds_parse_egl_sink(appctx.sink, argv[1], "sink");
#endif
      }
    }
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (appctx.pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if (PERF_MODE) {
    gst_bin_add_many (GST_BIN (appctx.pipeline), appctx.queue1, appctx.nvdslogger,
                      appctx.queue5, appctx.sink, NULL);
  } else {
   gst_bin_add_many (GST_BIN (appctx.pipeline), appctx.queue1, appctx.preprocess,
                     appctx.pgie, appctx.queue2, appctx.nvanalytics, appctx.queue6,
                     appctx.nvdslogger, appctx.tiler, appctx.queue3, appctx.nvvidconv,
                     appctx.queue4, appctx.nvosd, appctx.queue5, appctx.sink, NULL);
    if (enc_enable){
      gst_bin_add_many (GST_BIN (appctx.pipeline), appctx.nvvidconv2, appctx.encoder,
                        appctx.parser, appctx.queue_post_encoder, NULL);
    }
  }
  /* we link the elements together
  * nvmultiurisrcbin -> pgie -> nvdslogger -> nvtiler -> nvvidconv -> nvosd
  * -> video-renderer */
  if(appctx.restServer) {
    if (PERF_MODE) {
      if (!gst_element_link_many (
            gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator),
            appctx.queue1, appctx.nvdslogger , appctx.queue5,  NULL)) {
        g_printerr ("Elements multiSourceBin, nvdslogger, queue5 could not be linked. Exiting.\n");
        return -1;
      }
    } else {
      if (!gst_element_link_many (
            gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator),
            appctx.queue1, appctx.preprocess, appctx.pgie, appctx.queue2,
            appctx.nvanalytics, appctx.queue6,
            appctx.nvdslogger, appctx.tiler, appctx.queue3, appctx.nvvidconv,
            appctx.queue4, appctx.nvosd, appctx.queue5, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }
  } else {
    if (PERF_MODE) {
      if (!gst_element_link_many (appctx.multiuribin, appctx.queue1,
                                  appctx.nvdslogger , appctx.queue5,  NULL)) {
        g_printerr ("Elements multiSourceBin, nvdslogger, queue5 could not be linked. Exiting.\n");
        return -1;
      }
    } else {
      if (!gst_element_link_many (appctx.multiuribin, appctx.queue1,
                                  appctx.preprocess, appctx.pgie, appctx.queue2,
                                  appctx.nvanalytics, appctx.queue6,
                                  appctx.nvdslogger, appctx.tiler, appctx.queue3,
                                  appctx.nvvidconv, appctx.queue4, appctx.nvosd,
                                  appctx.queue5, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }
  }

  if (!enc_enable){
    if(!gst_element_link_many (appctx.queue5, appctx.sink, NULL)) {
      g_printerr ("queue5->sink Elements could not be linked. Exiting.\n");
      return -1;
    }
  }
  else {
    gst_element_link_many (appctx.queue5, appctx.nvvidconv2, appctx.encoder, appctx.queue_post_encoder, appctx.parser, appctx.sink, NULL);
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tiler_src_pad = gst_element_get_static_pad (appctx.tiler, "src");
  if (!tiler_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (tiler_src_pad);

  if (PERF_MODE) {
  sink_pad = gst_element_get_static_pad (appctx.sink, "sink");

    gst_pad_add_probe (sink_pad,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        pad_probe_event_on_fakesink, (void *) &appctx, NULL);
    gst_object_unref (sink_pad);
  }
  /* Set the pipeline to "playing" state */
  g_print ("Using file: %s\n", argv[1]);

  gst_element_set_state (appctx.pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");

  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (appctx.pipeline, GST_STATE_NULL);
  if(appctx.restServer) {
    g_print ("Calling gst_nvmultiurisrcbincreator_deinit\n");
    gst_nvmultiurisrcbincreator_deinit(appctx.nvmultiurisrcbinCreator);
  }
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (appctx.pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  if(appctx.restServer) {
    g_print ("Stoping REST server\n");
    nvds_rest_server_stop((NvDsRestServer*)appctx.restServer);
    appctx.restServer = NULL;
    if(appctx.httpIp) {
      g_free(appctx.httpIp);
    }
    if(appctx.httpPort) {
      g_free(appctx.httpPort);
    }
  }

  return 0;
}
