/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <string>

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "gstnvdsinfer.h"

#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#define MAX_DISPLAY_LEN 64

/** Defines the maximum size of an array for storing a text result. */
#define MAX_LABEL_SIZE 128

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

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

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person",
                                 "RoadSign"};

#define FPS_PRINT_INTERVAL 300

std::vector<std::vector<std::string>> m_Labels;

gboolean
parse_labels_file(const std::string &labelsFilePath, std::vector<std::vector<std::string>> &m_Labels);

gboolean
parse_labels_file(const std::string &labelsFilePath, std::vector<std::vector<std::string>> &m_Labels)
{
  std::ifstream labels_file(labelsFilePath);
  std::string delim{';'};

  if (!labels_file.is_open())
  {
    g_print("Could not open labels file:%s", labelsFilePath.c_str());
    return FALSE;
  }
  while (labels_file.good() && !labels_file.eof())
  {
    std::string line, word;
    std::vector<std::string> l;
    size_t pos = 0, oldpos = 0;

    std::getline(labels_file, line, '\n');
    if (line.empty())
      continue;

    while ((pos = line.find(delim, oldpos)) != std::string::npos)
    {
      word = line.substr(oldpos, pos - oldpos);
      l.push_back(word);
      oldpos = pos + delim.length();
    }
    l.push_back(line.substr(oldpos));
    m_Labels.push_back(l);
  }

  if (labels_file.bad())
  {
    g_print("Failed to parse labels file:%s, iostate:%d",
            labelsFilePath.c_str(), (int)labels_file.rdstate());
    return FALSE;
  }
  return TRUE;
}
#if 0
static GstPadProbeReturn
pgie_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsMetaList * l_user_meta = NULL;
    NvDsUserMeta *user_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_user_meta = batch_meta->batch_user_meta_list; l_user_meta != NULL;
        l_user_meta = l_user_meta->next) 
    {
      user_meta = (NvDsUserMeta *)(l_user_meta->data);
      if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) 
      {
        GstNvDsPreProcessBatchMeta *preprocess_batchmeta =
            (GstNvDsPreProcessBatchMeta *) (user_meta->user_meta_data);
        if (preprocess_batchmeta->tensor_meta->raw_tensor_buffer) {
          g_print("received preprocess meta\n");
        }
      }
    }
    return GST_PAD_PROBE_OK;
}
#endif
/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    //int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
      num_rects++;

      NvDsMetaList *l_classifier = NULL;
      for (l_classifier = obj_meta->classifier_meta_list; l_classifier != NULL;
           l_classifier = l_classifier->next)
      {
        NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *)(l_classifier->data);
        NvDsLabelInfoList *l_label;
        for (l_label = classifier_meta->label_info_list; l_label != NULL;
             l_label = l_classifier->next)
        {
          NvDsLabelInfo *label_info = (NvDsLabelInfo *)l_label->data;

          display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
          display_meta->num_labels = 1;
          NvOSD_TextParams *txt_params = &display_meta->text_params[0];
          txt_params->display_text = (char *)g_malloc0(MAX_LABEL_SIZE);

          snprintf(txt_params->display_text, MAX_LABEL_SIZE, "%s", label_info->result_label);
          //printf("%s\n", label_info->result_label);
          /* Now set the offsets where the string should appear */
          txt_params->x_offset = 10;
          txt_params->y_offset = 12;

          /* Font , font-color and font-size */
          txt_params->font_params.font_name = (char *)"Serif";
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
        }
      }
    }
    g_print("Source ID = %d Frame Number = %d Number of objects = %d\n",
            frame_meta->source_id, frame_meta->frame_num, num_rects);
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

#if 1
  NvDsMetaList *l_user_meta = NULL;
  NvDsUserMeta *user_meta = NULL;
  for (l_user_meta = batch_meta->batch_user_meta_list; l_user_meta != NULL;
       l_user_meta = l_user_meta->next)
  {
    user_meta = (NvDsUserMeta *)(l_user_meta->data);
    if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META)
    {
      GstNvDsPreProcessBatchMeta *preprocess_batchmeta =
          (GstNvDsPreProcessBatchMeta *)(user_meta->user_meta_data);
      guint roi_cnt = 0;
      for (auto &roi_meta : preprocess_batchmeta->roi_vector)
      {
        NvDsMetaList *l_user = NULL;
        for (l_user = roi_meta.roi_user_meta_list; l_user != NULL;
             l_user = l_user->next)
        {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
          if (user_meta->base_meta.meta_type == NVDSINFER_SEGMENTATION_META)
          {
            NvDsInferSegmentationMeta *segmeta = (NvDsInferSegmentationMeta *)(user_meta->user_meta_data);
            std::vector<float> temp(segmeta->width * segmeta->height);
            for (size_t i = 0; i < temp.size(); i++)
            {
              temp[i] = segmeta->class_map[i];
            }
            std::ofstream outfile("segout_frame_" +
                                  std::to_string(roi_meta.frame_meta->frame_num) + "_src_" +
                                  std::to_string(roi_meta.frame_meta->source_id) + "_roi_" +
                                  std::to_string(roi_cnt) + "_" +
                                  std::to_string(segmeta->width) + "x" +
                                  std::to_string(segmeta->height) + ".bin");
            outfile.write((char *)temp.data(), sizeof(float) * temp.size());
            outfile.close();
          }
          if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
          {
            NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)(user_meta->user_meta_data);
            gfloat max_prob = 0;
            gint class_id = 0;
            gfloat *buffer = (gfloat *)tensor_meta->out_buf_ptrs_host[0];
            for (size_t i = 0; i < tensor_meta->output_layers_info[0].inferDims.d[0]; i++)
            {
              if (buffer[i] > max_prob)
              {
                max_prob = buffer[i];
                class_id = i;
              }
            }
            printf("frame %d src %d roi %d label %s\n", roi_meta.frame_meta->frame_num, roi_meta.frame_meta->source_id,
                   roi_cnt, m_Labels[0][class_id].c_str());
          }
        }
        roi_cnt++;

        NvDsMetaList *l_classifier = NULL;
        for (l_classifier = roi_meta.classifier_meta_list; l_classifier != NULL;
             l_classifier = l_classifier->next)
        {
          NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *)(l_classifier->data);
          NvDsLabelInfoList *l_label;
          for (l_label = classifier_meta->label_info_list; l_label != NULL;
               l_label = l_classifier->next)
          {
            NvDsLabelInfo *label_info = (NvDsLabelInfo *)l_label->data;

            display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            display_meta->num_labels = 1;

            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            txt_params->display_text = (char *)g_malloc0(MAX_LABEL_SIZE);

            snprintf(txt_params->display_text, sizeof(label_info->result_label), "%s", label_info->result_label);
            printf("%s\n", label_info->result_label);
            /* Now set the offsets where the string should appear */
            txt_params->x_offset = 10;
            txt_params->y_offset = 12;

            /* Font , font-color and font-size */
            txt_params->font_params.font_name = (char *)"Serif";
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

            nvds_add_display_meta_to_frame(roi_meta.frame_meta, display_meta);
          }
        }
      }
    }
  }
#endif
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_WARNING:
  {
    gchar *debug = NULL;
    GError *error = NULL;
    gst_message_parse_warning(msg, &error, &debug);
    g_printerr("WARNING from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    g_free(debug);
    g_printerr("Warning: %s\n", error->message);
    g_error_free(error);
    break;
  }
  case GST_MESSAGE_ERROR:
  {
    gchar *debug = NULL;
    GError *error = NULL;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
#ifndef PLATFORM_TEGRA
  case GST_MESSAGE_ELEMENT:
  {
    if (gst_nvmessage_is_stream_eos(msg))
    {
      guint stream_id = 0;
      if (gst_nvmessage_parse_stream_eos(msg, &stream_id))
      {
        g_print("Got EOS from stream %d\n", stream_id);
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
cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
  g_print("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstElement *source_bin = (GstElement *)data;
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp(name, "video", 5))
  {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM))
    {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
      if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                    decoder_src_pad))
      {
        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref(bin_ghost_pad);
    }
    else
    {
      g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                      gchar *name, gpointer user_data)
{
  g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name)
  {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin(guint index, gchar *uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {};

  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin)
  {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                   G_CALLBACK(cb_newpad), bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(decodebin_child_added), bin);

  gst_bin_add(GST_BIN(bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src",
                                                            GST_PAD_SRC)))
  {
    g_printerr("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <nvdspreprocess-config-file> <nvinfer-config-file> [uri1] [uri2] ... [uriN]\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <nvdspreprocess-config-file> <nvinferserver-config-file> "
    "[uri1] [uri2] ... [uriN]\n", bin);
}

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
             *preprocess = NULL, *queue1, *queue2, *queue3, *queue4, *queue5, *queue6,
             *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *pgie_src_pad = NULL;
  //  GstPad *pgie_sink_pad = NULL;
  guint i, num_sources = 0;
  guint tiler_rows, tiler_columns;
  gboolean is_nvinfer_server = FALSE;
  gchar *nvdspreprocess_config_file = NULL;
  gchar *infer_config_file = NULL;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  /* Check input arguments */
  if (argc < 4)
  {
    usage(argv[0]);
    return -1;
  }

  if (argc >=4 && !strcmp("-t", argv[1])) {
    if (!strcmp("inferserver", argv[2])) {
      is_nvinfer_server = TRUE;
    } else {
      usage(argv[0]);
      return -1;
    }
    g_print ("Using nvinferserver as the inference plugin\n");
  }

  if (is_nvinfer_server) {
    num_sources = argc - 5;
    nvdspreprocess_config_file = realpath(argv[3], NULL);
    infer_config_file = realpath(argv[4], NULL);
  } else {
    num_sources = argc - 3;
    nvdspreprocess_config_file = realpath(argv[1], NULL);
    infer_config_file = realpath(argv[2], NULL);
  }

  //parse_labels_file("resnet50/imagenet1000_labels.txt", m_Labels);

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("preprocess-test-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  for (i = 0; i < num_sources; i++)
  {
    GstPad *sinkpad, *srcpad;
    GstElement *source_bin;
    gchar pad_name[16] = {};

    if (is_nvinfer_server) {
      source_bin = create_source_bin(i, argv[i + 5]);
    } else {
      source_bin = create_source_bin(i, argv[i + 3]);
    }

    if (!source_bin)
    {
      g_printerr("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add(GST_BIN(pipeline), source_bin);

    g_snprintf(pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple(streammux, pad_name);
    if (!sinkpad)
    {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad)
    {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }

  /* to preprocess the rois and form a raw tensor for inferencing */
  preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make(
          is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN,
          "primary-nvinference-engine");
          
  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make("queue", "queue1");
  queue2 = gst_element_factory_make("queue", "queue2");
  queue3 = gst_element_factory_make("queue", "queue3");
  queue4 = gst_element_factory_make("queue", "queue4");
  queue5 = gst_element_factory_make("queue", "queue5");
  queue6 = gst_element_factory_make("queue", "queue6");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if (prop.integrated)
  {
    sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!preprocess || !pgie || !tiler || !nvvidconv || !nvosd || !sink)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);

  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  g_object_set(G_OBJECT(preprocess), "config-file", nvdspreprocess_config_file, NULL);
  //  g_object_set (G_OBJECT (preprocess), "config-file", "config_preprocess.txt", NULL);
  //  g_object_set (G_OBJECT (preprocess), "config-file", "config_preprocess_classifier.txt", NULL);
  //  g_object_set (G_OBJECT (preprocess), "config-file", "config_preprocess_seg.txt", NULL);
  //  g_object_set (G_OBJECT (preprocess), "config-file", "config_preprocess_gray.txt", NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set(G_OBJECT(pgie), "input-tensor-meta", TRUE,
               "config-file-path", infer_config_file, NULL);
  //      "config-file-path", "ds_preproc_pgie_config.txt", NULL);
  //      "config-file-path", "resnet50/config_infer_primary_resnet50.txt", NULL);
  //      "config-file-path", "dstest_segmentation_config_semantic.txt", NULL);
  //      "config-file-path", "mnist_onnx_gray/config_infer_primary_mnist.txt", NULL);

  g_print("num-sources = %d\n", num_sources);

  tiler_rows = (guint)sqrt(num_sources);
  tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
               "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set(G_OBJECT(nvosd), "process-mode", OSD_PROCESS_MODE,
               "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set(G_OBJECT(sink), "qos", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, queue2, pgie, queue3, tiler,
                   queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL);
  /* we link the elements together
  * nvstreammux -> pgie -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many(streammux, queue1, preprocess, queue2, pgie, queue3, tiler,
                             queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL))
  {
    g_printerr("Elements could not be linked. Exiting.\n");
    return -1;
  }
#if 0
  pgie_sink_pad = gst_element_get_static_pad (pgie, "sink");
  if (!pgie_sink_pad)
    g_print ("Unable to get pgie sink pad\n");
  else
    gst_pad_add_probe (pgie_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (pgie_sink_pad);
#endif
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  pgie_src_pad = gst_element_get_static_pad(pgie, "src");
  if (!pgie_src_pad)
    g_print("Unable to get pgie src pad\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(pgie_src_pad);

  /* Set the pipeline to "playing" state */
  g_print("Now playing...\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
