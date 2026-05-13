/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvbufsurftransform.h"

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2
#define CUSTOM_PTS 1

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33000

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"
#define PGIE_CONFIG_FILE  "dstest_appsrc_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dstest_appsrc_nvinferserver_config.txt"

gint frame_number = 0;

/* Structure to contain all our information for appsrc,
 * so we can pass it to callbacks */
typedef struct _AppSrcData
{
  GstElement *app_source;
  long frame_size;
  FILE *file;                   /* Pointer to the raw video file */
  gint appsrc_frame_num;
  guint fps;                    /* To set the FPS value */
  guint sourceid;               /* To control the GSource */
  gchar *vidconv_format;
  long width;
  long height;
  long width_planeN;
  long height_planeN;
  NvBufSurfaceColorFormat color_format;
  uint32_t num_planes;
} AppSrcData;

int current_device = -1;

/* new_sample is an appsink callback that will extract metadata received
 * tee sink pad and update params for drawing rectangle,
 *object information etc. */
static GstFlowReturn
new_sample (GstElement * sink, gpointer * data)
{
  GstSample *sample;
  GstBuffer *buf = NULL;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  unsigned long int pts = 0;
  sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
  if (gst_app_sink_is_eos (GST_APP_SINK (sink))) {
    g_print ("EOS received in Appsink********\n");
  }
  if (sample) {
    /* Obtain GstBuffer from sample and then extract metadata from it. */
    buf = gst_sample_get_buffer (sample);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      pts = frame_meta->buf_pts;
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
    }
    g_print ("Frame Number = %d Number of objects = %d "
        "Vehicle Count = %d Person Count = %d PTS = %" GST_TIME_FORMAT "\n",
        frame_number, num_rects, vehicle_count, person_count,
        GST_TIME_ARGS (pts));
    frame_number++;
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
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
outbuf_unref_callback (gpointer data)
{
  NvBufSurface *nvbufsurface = (NvBufSurface *) data;
  if (data != NULL) {
    cudaFree (nvbufsurface->surfaceList[0].dataPtr);
    nvbufsurface->surfaceList[0].dataPtr = NULL;
    free (nvbufsurface->surfaceList);
    free (nvbufsurface);
  }
}

/* This method is called by the idle GSource in the mainloop,
 * to feed one raw video frame into cuda allocated memory and load it to appsrc
 * and then push wrapped output buffer to the deepstream pipeline .
 * The idle handler is added to the mainloop when appsrc requests us
 * to start sending data (need-data signal)
 * and is removed when appsrc has enough data (enough-data signal).
 */
static gboolean
read_data (AppSrcData * data)
{
  void *file_data;
  GstFlowReturn gstret;
  size_t ret = 0;
  GstMapInfo map;
  file_data = malloc (data->frame_size);
  ret = fread (file_data, 1, data->frame_size, data->file);

  if (ret == 0) {
    gstret = gst_app_src_end_of_stream ((GstAppSrc *) data->app_source);

    free (file_data);
    if (gstret != GST_FLOW_OK) {
      g_print
          ("gst_app_src_end_of_stream returned %d. EoS not queued successfully.\n",
          gstret);
      return FALSE;
    }
  }
  if (ret > 0) {
    void *cuda_device_data;
    if (cudaMalloc ((void **) &cuda_device_data,
            data->frame_size) != cudaSuccess) {
      g_print ("ERROR !! Unable to allocate device memory. \n");
      free (file_data);
      return FALSE;
    } else {
      if (cudaMemcpy (cuda_device_data,
              file_data, data->frame_size,
              cudaMemcpyHostToDevice) != cudaSuccess) {
        g_print
            ("ERROR !! Unable to copy between device and host memories. \n");
        free (file_data);
        cudaFree (cuda_device_data);
        return FALSE;
      }
    }

    free (file_data);

    /* Update NvBufSurface varibales based on the color format */
    NvBufSurface *nvbufsurface =
        (NvBufSurface *) calloc (1, sizeof (NvBufSurface));
    /* Numfilled set to valid buffer size i.e. 1 */
    nvbufsurface->numFilled = 1;
    /* Allocate surfaceList with batch-size i.e. 1 */
    nvbufsurface->surfaceList = (NvBufSurfaceParams *) calloc (1, sizeof (NvBufSurfaceParams)); //1 is batchsize
    /* Use CUDA device as mem type for processing cuda memory in deepstream pipeline */
    nvbufsurface->memType = NVBUF_MEM_CUDA_DEVICE;
    /* Set number of planes based on colorformat */
    nvbufsurface->surfaceList[0].planeParams.num_planes = data->num_planes;
    /* Set width and height as per the color format */
    nvbufsurface->surfaceList[0].planeParams.width[0] = data->width;
    nvbufsurface->surfaceList[0].planeParams.height[0] = data->height;
    /* If the color format is RGBA bytes per pixel is 4 hence multiply width with it */
    nvbufsurface->surfaceList[0].planeParams.pitch[0] =
        data->color_format ==
        NVBUF_COLOR_FORMAT_RGBA ? (data->width * 4) : data->width;
    nvbufsurface->surfaceList[0].planeParams.offset[0] = 0;
    /* Set bytes per pixel based on the color format */
    nvbufsurface->surfaceList[0].planeParams.bytesPerPix[0] =
        data->color_format == NVBUF_COLOR_FORMAT_RGBA ? 4 : 1;
    /* Set the plane size */
    nvbufsurface->surfaceList[0].planeParams.psize[0] =
        nvbufsurface->surfaceList[0].planeParams.pitch[0] *
        nvbufsurface->surfaceList[0].planeParams.height[0];

    /** Set plane 1 and plane 2 parameters
     *  Valid for NV12 and I420 color format
     *  RGBA num planes=1, NV12, num planes=2, I420, num planes=3 */
    for (uint32_t i = 1;
        i < nvbufsurface->surfaceList[0].planeParams.num_planes; i++) {
      nvbufsurface->surfaceList[0].planeParams.width[i] = data->width_planeN;
      nvbufsurface->surfaceList[0].planeParams.height[i] = data->height_planeN;
      nvbufsurface->surfaceList[0].planeParams.pitch[i] =
          data->color_format ==
          NVBUF_COLOR_FORMAT_NV12 ? data->width : data->width_planeN;
      nvbufsurface->surfaceList[0].planeParams.offset[i] =
          nvbufsurface->surfaceList[0].planeParams.offset[i - 1] +
          nvbufsurface->surfaceList[0].planeParams.psize[i - 1];
      nvbufsurface->surfaceList[0].planeParams.bytesPerPix[i] =
          data->color_format == NVBUF_COLOR_FORMAT_NV12 ? 2 : 1;
      nvbufsurface->surfaceList[0].planeParams.psize[i] =
          nvbufsurface->surfaceList[0].planeParams.pitch[i] *
          nvbufsurface->surfaceList[0].planeParams.height[i];
    }

    /*Set NvbufSurface parameters once plane parameters are set */
    nvbufsurface->batchSize = 1;
    nvbufsurface->gpuId = current_device;
    nvbufsurface->surfaceList[0].width =
        nvbufsurface->surfaceList[0].planeParams.width[0];
    nvbufsurface->surfaceList[0].height =
        nvbufsurface->surfaceList[0].planeParams.height[0];
    nvbufsurface->surfaceList[0].pitch =
        data->color_format ==
        NVBUF_COLOR_FORMAT_RGBA ? (data->width * 4) : data->width;
    nvbufsurface->surfaceList[0].colorFormat = data->color_format;
    nvbufsurface->surfaceList[0].dataSize =
        nvbufsurface->surfaceList[0].planeParams.psize[0] +
        nvbufsurface->surfaceList[0].planeParams.psize[1] +
        nvbufsurface->surfaceList[0].planeParams.psize[2];
    nvbufsurface->isContiguous = TRUE;
    /* Store the reference of the cuda memory block to the dataPtr of surfaceList */
    nvbufsurface->surfaceList[0].dataPtr = cuda_device_data;

    /* Wrap the nvbufsurface in a buffer which would be sent to the downstream deepstream pipeline */
    GstBuffer *outbuf =
        gst_buffer_new_wrapped_full (0, nvbufsurface, sizeof (NvBufSurface), 0,
        sizeof (NvBufSurface), nvbufsurface, outbuf_unref_callback);

#if CUSTOM_PTS
    GST_BUFFER_PTS (outbuf) =
        gst_util_uint64_scale (data->appsrc_frame_num, GST_SECOND, data->fps);
#endif
    gstret = gst_app_src_push_buffer ((GstAppSrc *) data->app_source, outbuf);

    if (gstret != GST_FLOW_OK) {
      g_print ("gst_app_src_push_buffer returned %d \n", gstret);
      return FALSE;
    }
  }

  data->appsrc_frame_num++;
  return TRUE;
}

/* This signal callback triggers when appsrc needs data. Here,
 * we add an idle handler to the mainloop to start pushing
 * data into the appsrc */
static void
start_feed (GstElement * source, guint size, AppSrcData * data)
{
  if (data->sourceid == 0) {
    data->sourceid = g_idle_add ((GSourceFunc) read_data, data);
  }
}

/* This callback triggers when appsrc has enough data and we can stop sending.
 * We remove the idle handler from the mainloop */
static void
stop_feed (GstElement * source, AppSrcData * data)
{
  if (data->sourceid != 0) {
    g_source_remove (data->sourceid);
    data->sourceid = 0;
  }
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <Raw filename> <width> <height> <fps> <format(I420, NV12, RGBA)>\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <Raw filename> <width> <height> <fps> "
    "<format(I420, NV12, RGBA)>\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *nvvidconv1 = NULL, *caps_filter = NULL,
      *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv2 = NULL,
      *nvosd = NULL, *tee = NULL, *appsink = NULL;
  GstElement *transform = NULL, *nvvidconv3 = NULL, *caps_filter_vidconv3 =
      NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  AppSrcData data;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  gchar *endptr1 = NULL, *endptr2 = NULL, *endptr3 = NULL;
  GstPad *tee_source_pad1, *tee_source_pad2;
  GstPad *osd_sink_pad, *appsink_sink_pad;
  gboolean is_nvinfer_server = FALSE;

  current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  /* Check input arguments */
  if (argc < 6) {
    usage(argv[0]);
    return -1;
  }

  if (argc >=6 && !strcmp("-t", argv[1])) {
    if (!strcmp("inferserver", argv[2])) {
      is_nvinfer_server = TRUE;
    } else {
      usage(argv[0]);
      return -1;
    }
    g_print ("Using nvinferserver as the inference plugin\n");
  }

  /* Initialize custom data structure */
  memset (&data, 0, sizeof (data));
  gchar *format = NULL, *input_stream = NULL;
  long fps = 0;
  if (is_nvinfer_server) {
    data.width = g_ascii_strtoll (argv[4], &endptr1, 10);
    data.height = g_ascii_strtoll (argv[5], &endptr2, 10);
    fps = g_ascii_strtoll (argv[6], &endptr3, 10);
    format = argv[7];
    input_stream = argv[3];

    if ((data.width == 0 && endptr1 == argv[4]) || (data.height == 0
        && endptr2 == argv[5]) || (fps == 0 && endptr3 == argv[6])) {
      g_printerr ("Incorrect width, height or FPS\n");
      return -1;
    }
  } else {
    data.width = g_ascii_strtoll (argv[2], &endptr1, 10);
    data.height = g_ascii_strtoll (argv[3], &endptr2, 10);
    fps = g_ascii_strtoll (argv[4], &endptr3, 10);
    format = argv[5];
    input_stream = argv[1];

    if ((data.width == 0 && endptr1 == argv[2]) || (data.height == 0
        && endptr2 == argv[3]) || (fps == 0 && endptr3 == argv[4])) {
      g_printerr ("Incorrect width, height or FPS\n");
      return -1;
    }
  }

  if (data.width == 0 || data.height == 0 || fps == 0) {
    g_printerr ("Width, height or FPS cannot be 0\n");
    return -1;
  }
  if (g_strcmp0 (format, "I420") != 0 && g_strcmp0 (format, "RGBA") != 0
      && g_strcmp0 (format, "NV12") != 0) {
    g_printerr ("Only I420, RGBA and NV12 are supported\n");
    return -1;
  }

  if (!g_strcmp0 (format, "RGBA")) {
    data.frame_size = data.width * data.height * 4;
    data.vidconv_format = "RGBA";
    data.color_format = NVBUF_COLOR_FORMAT_RGBA;
    data.num_planes = 1;
  } else if (!g_strcmp0 (format, "NV12")) {
    data.frame_size = data.width * data.height * 1.5;
    data.vidconv_format = "NV12";
    data.color_format = NVBUF_COLOR_FORMAT_NV12;
    data.num_planes = 2;
    data.width_planeN = data.width / 2;
    data.height_planeN = data.height / 2;
  } else if (!g_strcmp0 (format, "I420")) {
    data.frame_size = data.width * data.height * 1.5;
    data.vidconv_format = "NV12";
    data.color_format = NVBUF_COLOR_FORMAT_YUV420;
    data.num_planes = 3;
    data.width_planeN = data.width / 2;
    data.height_planeN = data.height / 2;
  }
  if (is_nvinfer_server) {
    data.file = fopen (argv[3], "r");
  } else {
    data.file = fopen (argv[1], "r");
  }
  data.fps = fps;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);
  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest-appsrc-pipeline");
  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }
  /* App Source element for reading from raw video file */
  data.app_source = gst_element_factory_make ("appsrc", "app-source");
  if (!data.app_source) {
    g_printerr ("Appsrc element could not be created. Exiting.\n");
    return -1;
  }
  /* Use convertor to convert from software buffer to GPU buffer */
  nvvidconv1 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
  if (!nvvidconv1) {
    g_printerr ("nvvideoconvert1 could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (nvvidconv1), "nvbuf-memory-type", 2, "compute-hw", 1,
      NULL);

  caps_filter = gst_element_factory_make ("capsfilter", "capsfilter");
  if (!caps_filter) {
    g_printerr ("Caps_filter could not be created. Exiting.\n");
    return -1;
  }
  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  if (!streammux) {
    g_printerr ("nvstreammux could not be created. Exiting.\n");
    return -1;
  }
  /* Use nvinfer to run inferencing on streammux's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (
          is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN, 
          "primary-nvinference-engine");
  if (!pgie) {
    g_printerr ("Primary nvinfer could not be created. Exiting.\n");
    return -1;
  }
  /* Use convertor to convert from NV12 to RGBA as required by nvdsosd */
  nvvidconv2 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  if (!nvvidconv2) {
    g_printerr ("nvvideoconvert2 could not be created. Exiting.\n");
    return -1;
  }

  if (prop.integrated)
    g_object_set (G_OBJECT (nvvidconv2), "nvbuf-memory-type", 4, "compute-hw",
        1, NULL);
  else
    g_object_set (G_OBJECT (nvvidconv2), "nvbuf-memory-type", 2, "compute-hw",
        1, NULL);

  GstCaps *caps_vidconv2 =
      gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)RGBA");

  GstElement *caps_filter_vidconv2 =
      gst_element_factory_make ("capsfilter", "src_cap_filter_nvvidconv2");

  g_object_set (G_OBJECT (caps_filter_vidconv2), "caps", caps_vidconv2, NULL);
  gst_caps_unref (caps_vidconv2);

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr ("nvdsosd could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (nvosd), "process-mode", 1, "display-text", 1, NULL);  //Process mode set to GPU mode

  /* Finally render the osd output. We will use a tee to render video
   * playback on nveglglessink, and we use appsink to extract metadata
   * from buffer and print object, person and vehicle count. */
  tee = gst_element_factory_make ("tee", "tee");
  if (!tee) {
    g_printerr ("Tee could not be created. Exiting.\n");
    return -1;
  }
  if (prop.integrated) {
    transform = gst_element_factory_make ("identity", "nvegl-transform");
    if (!transform) {
      g_printerr ("Tegra transform element could not be created. Exiting.\n");
      return -1;
    }

    GstCaps *caps_vidconv3 =
        gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)NV12");
    caps_filter_vidconv3 =
        gst_element_factory_make ("capsfilter", "src_cap_filter_nvvidconv3");
    g_object_set (G_OBJECT (caps_filter_vidconv3), "caps", caps_vidconv3, NULL);
    gst_caps_unref (caps_vidconv3);

    nvvidconv3 =
        gst_element_factory_make ("nvvideoconvert", "nvvideo-converter3");
    g_object_set (G_OBJECT (nvvidconv3), "nvbuf-memory-type", 2, "compute-hw",
        1, NULL);
  }

  if (prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!sink) {
    g_printerr ("Display sink could not be created. Exiting.\n");
    return -1;
  }
  appsink = gst_element_factory_make ("appsink", "app-sink");
  if (!appsink) {
    g_printerr ("Appsink element could not be created. Exiting.\n");
    return -1;
  }
  /* Configure appsrc */
  GstCaps *caps_appsrc = gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, format,
      "width", G_TYPE_INT, data.width,
      "height", G_TYPE_INT, data.height,
      "framerate", GST_TYPE_FRACTION, data.fps, 1, NULL);
  GstCapsFeatures *feature_appsrc = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps_appsrc, 0, feature_appsrc);
  g_object_set (data.app_source, "caps", caps_appsrc, NULL);
  gst_caps_unref (caps_appsrc);
#if !CUSTOM_PTS
  g_object_set (G_OBJECT (data.app_source), "do-timestamp", TRUE, NULL);
#endif
  g_signal_connect (data.app_source, "need-data", G_CALLBACK (start_feed),
      &data);
  g_signal_connect (data.app_source, "enough-data", G_CALLBACK (stop_feed),
      &data);

  caps =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
      data.vidconv_format, NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);
  gst_caps_unref (caps);

  /* Set streammux properties */
  g_object_set (G_OBJECT (streammux), "width", data.width, "height",
      data.height, "batch-size", 1, "live-source", TRUE,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
      "nvbuf-memory-type", 2, "compute-hw", 1, NULL);

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
      data.app_source, nvvidconv1, caps_filter, streammux, pgie,
      nvvidconv2, caps_filter_vidconv2, nvosd, tee, sink, appsink, NULL);
  if (prop.integrated) {
    gst_bin_add_many (GST_BIN (pipeline), nvvidconv3, caps_filter_vidconv3,
        transform, NULL);
  }
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";
  sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }
  srcpad = gst_element_get_static_pad (caps_filter, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }
  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link caps filter to stream muxer. Exiting.\n");
    return -1;
  }
  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);
  /* we link the elements together */
  /* app-source -> nvvidconv1 -> caps filter ->
   * pgie -> nvvidconv2 -> nvosd -> video-renderer */
  if (prop.integrated) {
    if (!gst_element_link_many (data.app_source, nvvidconv1, caps_filter, NULL)
        || !gst_element_link_many (nvosd, nvvidconv3, caps_filter_vidconv3,
            transform, sink, NULL)
        || !gst_element_link_many (streammux, pgie, nvvidconv2,
            caps_filter_vidconv2, tee, NULL)) {
      g_printerr ("Elements could not be linked: Exiting.\n");
      return -1;
    }
  } else {
    if (!gst_element_link_many (data.app_source, nvvidconv1, caps_filter, NULL)
        || !gst_element_link_many (nvosd, sink, NULL)
        || !gst_element_link_many (streammux, pgie, nvvidconv2, tee, NULL)) {
      g_printerr ("Elements could not be linked: Exiting.\n");
      return -1;
    }
  }
  /* Manually link the Tee, which has "Request" pads.
   * This tee, in case of multistream usecase, will come before tiler element. */
  tee_source_pad1 = gst_element_request_pad_simple (tee, "src_0");
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  tee_source_pad2 = gst_element_request_pad_simple (tee, "src_1");
  appsink_sink_pad = gst_element_get_static_pad (appsink, "sink");
  if (gst_pad_link (tee_source_pad1, osd_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked to display sink.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  if (gst_pad_link (tee_source_pad2, appsink_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked to appsink.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  gst_object_unref (osd_sink_pad);
  gst_object_unref (appsink_sink_pad);
  /* Configure appsink to extract data from DeepStream pipeline */
  g_object_set (appsink, "emit-signals", TRUE, "async", FALSE, NULL);
  g_object_set (sink, "sync", FALSE, NULL);
  /* Callback to access buffer and object info. */
  g_signal_connect (appsink, "new-sample", G_CALLBACK (new_sample), NULL);
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
