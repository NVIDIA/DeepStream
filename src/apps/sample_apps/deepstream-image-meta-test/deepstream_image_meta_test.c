/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <ctype.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include "nvds_obj_encode.h"
#include "gst-nvmessage.h"

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

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

#define PGIE_CONFIG_FILE "ds_image_meta_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "ds_image_meta_pgie_nvinferserver_config.txt"

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

#define FPS_PRINT_INTERVAL 300

#define save_img TRUE
#define attach_user_meta TRUE

gint frame_number = 0, frame_count = 0;

/* osd_sink_pad_buffer_probe will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information. We also iterate
 * through the user meta of type "NVDS_CROP_IMAGE_META" to find image crop meta
 * and demonstrate how to access it.*/
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;

  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int offset = 0;
    /* To verify  encoded metadata of cropped frames, we iterate through the
    * user metadata of each frame and if a metadata of the type
    * 'NVDS_CROP_IMAGE_META' is found then we write that to a file as
    * implemented below.
    */
    char fileFrameNameString[FILE_NAME_SIZE];
    const char *osd_string = "OSD";

    /* For Demonstration Purposes we are writing metadata to jpeg images of
      * the first 10 frames only.
      * The files generated have an 'OSD' prefix. */
    if (frame_number < 11) {
      NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
      FILE *file;
      int stream_num = 0;
      while (usrMetaList != NULL) {
        NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
        if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
          snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d.jpg",
              osd_string, frame_number, stream_num++);
          NvDsObjEncOutParams *enc_jpeg_image =
              (NvDsObjEncOutParams *) usrMetaData->user_meta_data;
          /* Write to File */
          file = fopen (fileFrameNameString, "wb");
          fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
              enc_jpeg_image->outLen, file);
          fclose (file);
        }
        usrMetaList = usrMetaList->next;
      }
    }
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
      /* To verify  encoded metadata of cropped objects, we iterate through the
       * user metadata of each object and if a metadata of the type
       * 'NVDS_CROP_IMAGE_META' is found then we write that to a file as
       * implemented below.
       */
      char fileObjNameString[FILE_NAME_SIZE];

      /* For Demonstration Purposes we are writing metadata to jpeg images of
       * vehicles or persons for the first 100 frames only.
       * The files generated have a 'OSD' prefix. */
      if (frame_number < 100 && (obj_meta->class_id == PGIE_CLASS_ID_PERSON
              || obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)) {
        NvDsUserMetaList *usrMetaList = obj_meta->obj_user_meta_list;
        FILE *file;
        while (usrMetaList != NULL) {
          NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
          if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *) usrMetaData->user_meta_data;

            snprintf (fileObjNameString, FILE_NAME_SIZE, "%s_%d_%d_%d_%s.jpg",
                osd_string, frame_number, frame_meta->batch_id, num_rects,
                obj_meta->obj_label);
            /* Write to File */
            file = fopen (fileObjNameString, "wb");
            fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                enc_jpeg_image->outLen, file);
            fclose (file);
            usrMetaList = NULL;
          } else {
            usrMetaList = usrMetaList->next;
          }
        }
      }
    }
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
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
  }
  g_print ("Frame Number = %d Number of objects = %d "
      "Vehicle Count = %d Person Count = %d\n",
      frame_number, num_rects, vehicle_count, person_count);
  frame_number++;
  return GST_PAD_PROBE_OK;
}

/* pgie_src_pad_buffer_probe will extract metadata received on pgie src pad
 * and update params for drawing rectangle, object information etc. We also
 * iterate through the object list and encode the cropped objects as jpeg
 * images and attach it as user meta to the respective objects.*/
static GstPadProbeReturn
pgie_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer ctx)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    GST_ERROR ("input buffer mapinfo failed");
    return GST_PAD_PROBE_DROP;
  }
  NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
  gst_buffer_unmap (buf, &inmap);

  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  const gchar *calc_enc_str = g_getenv ("CALCULATE_ENCODE_TIME");
  gboolean calc_enc = !g_strcmp0 (calc_enc_str, "yes");

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    /* For demonstration purposes, we will encode the first 10 frames. */
    if(frame_count <= 10) {
      NvDsObjEncUsrArgs frameData = { 0 };
      /* Preset */
      frameData.isFrame = 1;
      /* To be set by user */
      frameData.saveImg = save_img;
      frameData.attachUsrMeta = attach_user_meta;
      /* Set if Image scaling Required */
      frameData.scaleImg = FALSE;
      frameData.scaledWidth = 0;
      frameData.scaledHeight = 0;
      /* Quality */
      frameData.quality = 80;
      /* Set to calculate time taken to encode JPG image. */
      if (calc_enc) {
        frameData.calcEncodeTime = 1;
      }
      /* Main Function Call */
      nvds_obj_enc_process (ctx, &frameData, ip_surf, NULL, frame_meta);
    }
    guint num_rects = 0;
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
      /* Conditions that user needs to set to encode the detected objects of
       * interest. Here, by default all the detected objects are encoded.
       * For demonstration, we will encode the first object in the frame. */
      if ((obj_meta->class_id == PGIE_CLASS_ID_PERSON
              || obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
          && num_rects == 1) {
        NvDsObjEncUsrArgs objData = { 0 };
        /* To be set by user */
        objData.saveImg = save_img;
        objData.attachUsrMeta = attach_user_meta;
        /* Set if Image scaling Required */
        objData.scaleImg = FALSE;
        objData.scaledWidth = 0;
        objData.scaledHeight = 0;
        /* Preset */
        objData.objNum = num_rects;
        /* Quality */
        objData.quality = 80;
        /* Set to calculate time taken to encode JPG image. */
        if (calc_enc) {
          objData.calcEncodeTime = 1;
        }
        /*Main Function Call */
        nvds_obj_enc_process (ctx, &objData, ip_surf, obj_meta, frame_meta);
      }
    }
  }
  nvds_obj_enc_finish (ctx);
  frame_count++;
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
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
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
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }

  if (g_object_class_find_property(G_OBJECT_GET_CLASS(object),
    "gpu-id")) {
    guint *gpu_id = (guint *) user_data;
    g_object_set (object, "gpu-id", *gpu_id, NULL);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri, guint *gpu_id)
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
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

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
      G_CALLBACK (decodebin_child_added), gpu_id);

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
usage(const char *bin)
{
  g_printerr ("Usage: %s <gpu_id> <uri1> [uri2] ... [uriN] \n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <gpu_id> <uri1> [uri2] ... [uriN]\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
      *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *pgie_src_pad = NULL;
  GstPad *osd_sink_pad = NULL;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  guint gpu_id = 0;
  gboolean is_nvinfer_server = FALSE;
  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpu_id);

  /* Check input arguments */
  if (argc < 3) {
    usage(argv[0]);
    return -1;
  }

  if (argc >=3 && !strcmp("-t", argv[1])) {
    if (!strcmp("inferserver", argv[2])) {
      is_nvinfer_server = TRUE;
    } else {
      usage(argv[0]);
      return -1;
    }
    g_print ("Using nvinferserver as the inference plugin\n");
  }

  /* Check if gpuID is provided for multiple streams */
  if (is_nvinfer_server) {
    if ((argc >= 5) && (!(isdigit(*argv[3])))) {
      usage(argv[0]);
      return -1;
    }
    gpu_id = atoi(argv[3]);
    num_sources = argc - 4;
  } else {
    if ((argc >= 3) && (!(isdigit(*argv[1])))) {
      usage(argv[0]);
      return -1;
    }
    gpu_id = atoi(argv[1]);
    num_sources = argc - 2;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-image-meta-test-pipeline");

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
      source_bin = create_source_bin (i, argv[i + 4], &gpu_id);
    } else {
      source_bin = create_source_bin (i, argv[i + 2], &gpu_id);
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

  /* Use nvinfer or nvinferserver to infer on batched frame. */
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
  if (prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!pgie || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (!use_new_mux) {
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
      "gpu-id", gpu_id, NULL);
  }
  else {
    g_object_set (G_OBJECT (streammux), "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }

  /* Configure the pgie element using the nvinfer config file. */
  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_NVINFERSERVER_CONFIG_FILE, "gpu-id", gpu_id, NULL);
  } else {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, "gpu-id", gpu_id, NULL);
  }

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  g_object_set (G_OBJECT (nvvidconv), "gpu-id", gpu_id, NULL);
  g_object_set (G_OBJECT (nvosd), "gpu-id", gpu_id, NULL);
  /* gpu-id is a valid property for nveglglessink only on x86 platform. */
  if (!prop.integrated) {
#ifndef __aarch64__
   g_object_set (G_OBJECT (sink), "gpu-id", gpu_id, NULL);
#endif
  }

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT,
      "gpu-id", gpu_id, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), pgie, tiler, nvvidconv, nvosd, sink,
      NULL);
  /* we link the elements together
  * nvstreammux -> pgie -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, pgie, tiler, nvvidconv, nvosd, sink,
          NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the srd pad of the pgie element, since by that time, the buffer would have
   * had got all the nvinfer metadata. */
  pgie_src_pad = gst_element_get_static_pad (pgie, "src");
  /* Create Context for Object Encoding.
   * Takes GPU ID as a parameter. Passed by user through commandline.
   * Initialized as 0. */
  NvDsObjEncCtxHandle obj_ctx_handle = nvds_obj_enc_create_context (gpu_id);
  if (!obj_ctx_handle) {
    g_print ("Unable to create context\n");
    return -1;
  }
  if (!pgie_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_src_pad_buffer_probe, (gpointer) obj_ctx_handle, NULL);
  gst_object_unref (pgie_src_pad);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, (gpointer) obj_ctx_handle, NULL);
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing...\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Destroy context for Object Encoding */
  nvds_obj_enc_destroy_context (obj_ctx_handle);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
