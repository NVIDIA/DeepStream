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

#include <math.h>

#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"

#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

#include "nvdsinfer_dbscan.h"
#include <memory>
#include <cassert>

#pragma GCC diagnostic pop

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

#define INFER_PGIE_CONFIG_FILE  "dstensor_pgie_config.txt"
#define INFER_SGIE1_CONFIG_FILE "dstensor_sgie1_config.txt"
#define INFER_SGIE2_CONFIG_FILE "dstensor_sgie2_config.txt"
#define INFERSERVER_PGIE_CONFIG_FILE  "inferserver/dstensor_pgie_config.txt"
#define INFERSERVER_SGIE1_CONFIG_FILE "inferserver/dstensor_sgie1_config.txt"
#define INFERSERVER_SGIE2_CONFIG_FILE "inferserver/dstensor_sgie2_config.txt"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define PGIE_DETECTED_CLASS_NUM 4

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720


#define PGIE_NET_WIDTH 960
#define PGIE_NET_HEIGHT 544

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
/* These are the strings of the labels for the respective models */
const gchar sgie1_classes_str[20][32] = {
  "Acura", "Audi", "BMW", "Chevrolet", "Chrysler",
  "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia",
  "Lexus", "Mazda", "Mercedes", "Nissan",
  "Subaru", "Toyota", "Volkswagen"
};

const gchar sgie2_classes_str[6][32] = {
  "coupe", "largevehicle", "sedan", "suv",
  "truck", "van"
};

const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] =
    { "Vehicle", "TwoWheeler", "Person", "RoadSign" };

/* gie_unique_id is one of the properties in the above dstensor_sgiex_config.txt
 * files. These should be unique and known when we want to parse the Metadata
 * respective to the sgie labels. Ideally these should be read from the config
 * files but for brevity we ensure they are same. */

const guint sgie1_unique_id = 2;
const guint sgie2_unique_id = 3;

/* nvds_lib_major_version and nvds_lib_minor_version is the version number of
 * deepstream sdk */

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;

std::shared_ptr<NvDsInferDBScan> m_DBScanHandle;

/* This is the buffer probe function that we have registered on the sink pad
 * of the OSD element. All the infer elements in the pipeline shall attach
 * their metadata to the GstBuffer, here we will iterate & process the metadata
 * forex: class ids to strings, counting of class_id objects etc. */
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
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (gchar *) g_malloc0 (MAX_DISPLAY_LEN);
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
    txt_params->font_params.font_name = (gchar *) "Serif";
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

extern "C"
    bool NvDsInferParseCustomResnet (std::vector < NvDsInferLayerInfo >
    const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector < NvDsInferObjectDetectionInfo > &objectList);

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes. The result
 * would be attached as object-meta(NvDsObjectMeta) into the same frame metadata.
 */
static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static guint use_device_mem = 0;
  gboolean *use_new_mux = (gboolean *)u_data;
  guint stream_width=0, stream_height=0;

  static NvDsInferNetworkInfo networkInfo
  {
  PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
  NvDsInferParseDetectionParams detectionParams;
  detectionParams.numClassesConfigured = 4;
  detectionParams.perClassPreclusterThreshold = {0.2, 0.2, 0.2, 0.2};
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    if(*use_new_mux) {
       stream_width=frame_meta->source_frame_width;
       stream_height=frame_meta->source_frame_height;
    }
    else {
       stream_width=MUXER_OUTPUT_WIDTH;
       stream_height=MUXER_OUTPUT_HEIGHT;
    }

    /* Iterate user metadata in frames to search PGIE's tensor metadata */
    for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;

      /* convert to tensor metadata */
      NvDsInferTensorMeta *meta =
          (NvDsInferTensorMeta *) user_meta->user_meta_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
        if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
          cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
              info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
      }
      /* Parse output tensor and fill detection results into objectList. */
      std::vector < NvDsInferLayerInfo >
          outputLayersInfo (meta->output_layers_info,
          meta->output_layers_info + meta->num_output_layers);
      std::vector < NvDsInferObjectDetectionInfo > objectList;
#if NVDS_VERSION_MAJOR >= 5
      if (nvds_lib_major_version >= 5) {
        if (meta->network_info.width != networkInfo.width ||
            meta->network_info.height != networkInfo.height ||
            meta->network_info.channels != networkInfo.channels) {
          g_error ("failed to check pgie network info\n");
        }
      }
#endif
      NvDsInferParseCustomResnet (outputLayersInfo, networkInfo,
          detectionParams, objectList);

      NvDsInferDBScanClusteringParams clusteringParams;
      clusteringParams.enableATHRFilter = true;
      clusteringParams.thresholdATHR = 60.0;
      clusteringParams.eps = 0.95;
      clusteringParams.minBoxes = 3;
      clusteringParams.minScore = 0.5;
      assert(m_DBScanHandle);
      /* Create perClassObjectList: vector<vector<NvDsInferObjectDetectionInfo>>. Each vector is of same classID */
      std::vector <std::vector <NvDsInferObjectDetectionInfo> > perClassObjectList(PGIE_DETECTED_CLASS_NUM);
      for (auto & obj:objectList) {
        perClassObjectList[obj.classId].emplace_back (obj);
      }

      /* Call NvDsInferDBScanCluster on each of the vector and resize it */
      for (unsigned int c = 0; c < perClassObjectList.size(); c++) {
        NvDsInferObjectDetectionInfo *objArray = (NvDsInferObjectDetectionInfo*) (perClassObjectList[c].data());
        size_t numObjects = perClassObjectList[c].size();

        /* Cluster together rectangles with similar locations and sizes since these rectangles might represent the same object using DBSCAN. */
        if (clusteringParams.minBoxes > 0) {
            NvDsInferDBScanCluster(
                m_DBScanHandle.get(), &clusteringParams, objArray, &numObjects);
        }
        perClassObjectList[c].resize(numObjects);

        /* Iterate perClassObjectList for left, top, width, height values of rectangle and attach result into frame's obj_meta_list. */
        auto & objlist = perClassObjectList[c];
        if (objlist.empty ())
          continue;

      for (const auto & rect:objlist) {
          NvDsObjectMeta *obj_meta =
              nvds_acquire_obj_meta_from_pool (batch_meta);
          obj_meta->unique_component_id = meta->unique_id;
          obj_meta->confidence = 0.0;

          /* This is an untracked object. Set tracking_id to -1. */
          obj_meta->object_id = UNTRACKED_OBJECT_ID;
          obj_meta->class_id = c;

          NvOSD_RectParams & rect_params = obj_meta->rect_params;
          NvOSD_TextParams & text_params = obj_meta->text_params;

          /* Assign bounding box coordinates. */
          rect_params.left = rect.left * stream_width / PGIE_NET_WIDTH;
          rect_params.top = rect.top * stream_height / PGIE_NET_HEIGHT;
          rect_params.width = rect.width * stream_width / PGIE_NET_WIDTH;
          rect_params.height =
              rect.height * stream_height / PGIE_NET_HEIGHT;

          /* Border of width 3. */
          rect_params.border_width = 3;
          rect_params.has_bg_color = 0;
          rect_params.border_color = (NvOSD_ColorParams) {
          1, 0, 0, 1};

          /* display_text requires heap allocated memory. */
          text_params.display_text = g_strdup (pgie_classes_str[c]);
          /* Display text above the left top corner of the object. */
          text_params.x_offset = rect_params.left;
          text_params.y_offset = rect_params.top - 10;
          /* Set black background for the text. */
          text_params.set_bg_clr = 1;
          text_params.text_bg_clr = (NvOSD_ColorParams) {
          0, 0, 0, 1};
          /* Font face, size and color. */
          text_params.font_params.font_name = (gchar *) "Serif";
          text_params.font_params.font_size = 11;
          text_params.font_params.font_color = (NvOSD_ColorParams) {
          1, 1, 1, 1};
          nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
        }
      }
    }
  }
  use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the sink pad
 * of the tiler element. All SGIE infer elements in the pipeline shall attach
 * their NvDsInferTensorMeta to each object's metadata of each frame, here we will
 * iterate & parse the tensor data to get classification confidence and labels.
 * The result would be attached as classifier_meta into its object's metadata.
 */
static GstPadProbeReturn
sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static guint use_device_mem = 0;

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      /* Iterate user metadata in object to search SGIE's tensor data */
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;

        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;

        for (unsigned int i = 0; i < meta->num_output_layers; i++) {
          NvDsInferLayerInfo *info = &meta->output_layers_info[i];
          info->buffer = meta->out_buf_ptrs_host[i];
          if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
            cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
          }
        }

        NvDsInferDimsCHW dims;

        getDimsCHWFromDims (dims, meta->output_layers_info[0].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer =
            (float *) meta->output_layers_info[0].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++) {
          float probability = outputCoverageBuffer[c];
          if (probability > 0.51 && probability > maxProbability) {
            maxProbability = probability;
            attrFound = true;
            attr.attributeIndex = 0;
            attr.attributeValue = c;
            attr.attributeConfidence = probability;
          }
        }

        /* Generate classifer metadata and attach to obj_meta */
        if (attrFound) {
          NvDsClassifierMeta *classifier_meta =
              nvds_acquire_classifier_meta_from_pool (batch_meta);

          classifier_meta->unique_component_id = meta->unique_id;

          NvDsLabelInfo *label_info =
              nvds_acquire_label_info_meta_from_pool (batch_meta);
          label_info->result_class_id = attr.attributeValue;
          label_info->result_prob = attr.attributeConfidence;

          /* Fill label name */
          switch (meta->unique_id) {
            case sgie1_unique_id:
              strcpy (label_info->result_label,
                  sgie1_classes_str[label_info->result_class_id]);
              break;
            case sgie2_unique_id:
              strcpy (label_info->result_label,
                  sgie2_classes_str[label_info->result_class_id]);
              break;
            default:
              break;
          }
          gchar *temp = obj_meta->text_params.display_text;
          obj_meta->text_params.display_text =
              g_strconcat (temp, " ", label_info->result_label, nullptr);
          g_free (temp);

          nvds_add_label_info_meta_to_classifier (classifier_meta, label_info);
          nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);
        }
      }
    }
  }

  use_device_mem = 1 - use_device_mem;
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
  g_printerr ("Usage: %s <H264 filename1> [H264 filename2] ...[H264 filenameN]\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <H264 filename1> [H264 filename2] ...[H264 filenameN]\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =
      NULL, *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *nvosd = NULL, *sgie1 = NULL, *sgie2 =
      NULL, *tiler = NULL, *queue2, *queue3, *queue4, *queue5;
  g_print ("With tracker\n");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL, *queue_src_pad = NULL, *tiler_sink_pad = NULL;
  guint i = 0, num_sources = 0;
  std::vector<std::string> files;
  gboolean is_nvinfer_server = FALSE;
  const char *infer_plugin = NVINFER_PLUGIN;

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
    infer_plugin = NVINFERSERVER_PLUGIN;
    num_sources = argc - 3;
  } else {
    num_sources = argc - 1;
  }
  nvds_version(&nvds_lib_major_version, &nvds_lib_minor_version);

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will be a container of other elements */
  pipeline = gst_pipeline_new ("dstensor-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (infer_plugin, "primary-nvinference-engine");

  queue = gst_element_factory_make ("queue", NULL);
  queue2 = gst_element_factory_make ("queue", NULL);
  queue3 = gst_element_factory_make ("queue", NULL);
  queue4 = gst_element_factory_make ("queue", NULL);
  queue5 = gst_element_factory_make ("queue", NULL);

  /* We need three secondary gies so lets create 3 more instances of
     nvinfer */
  sgie1 = gst_element_factory_make (infer_plugin, "secondary1-nvinference-engine");

  sgie2 = gst_element_factory_make (infer_plugin, "secondary2-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "tiler");

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

  if (!pgie || !sgie1 || !sgie2 || !nvvidconv || !nvosd || !sink ||
      !tiler) {
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

  /* Set all the necessary properties of the infer plugin element,
   * Enable Output tensor meta, we can probe PGIE and
   * SGIEs buffer to parse tensor output data of models */
  if (!is_nvinfer_server) {
    /* nvinfer Output tensor meta can be enabled by set "output-tensor-meta=true"
     * here or enable this attribute in config file. */
    g_object_set (G_OBJECT (pgie), "config-file-path", INFER_PGIE_CONFIG_FILE,
        "output-tensor-meta", TRUE, "batch-size", num_sources, NULL);
    g_object_set (G_OBJECT (sgie1), "config-file-path", INFER_SGIE1_CONFIG_FILE,
        "output-tensor-meta", TRUE, "process-mode", 2, NULL);
    g_object_set (G_OBJECT (sgie2), "config-file-path", INFER_SGIE2_CONFIG_FILE,
        "output-tensor-meta", TRUE, "process-mode", 2, NULL);
  } else {
    /* nvinferserver output tensor meta can be enabled in config file.
     * see output_control { output_tensor_meta: true }*/
    g_object_set (G_OBJECT (pgie), "config-file-path", INFERSERVER_PGIE_CONFIG_FILE,
        "batch-size", num_sources, NULL);
    g_object_set (G_OBJECT (sgie1), "config-file-path", INFERSERVER_SGIE1_CONFIG_FILE,
        "process-mode", 2, NULL);
    g_object_set (G_OBJECT (sgie2), "config-file-path", INFERSERVER_SGIE2_CONFIG_FILE,
        "process-mode", 2, NULL);
  }

  guint rows = sqrt (num_sources);
  g_object_set (G_OBJECT (tiler), "rows", rows, "columns",
      (guint) ceil (1.0 * num_sources / rows), "width", 1920, "height", 1080,
      NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* decoder | pgie1 | sgie1 | sgie2 | etc.. */
  gst_bin_add_many (GST_BIN (pipeline),
      streammux, pgie, queue, sgie1, queue5, sgie2, queue2,
      tiler, queue3, nvvidconv, queue4, nvosd, sink, NULL);

  for (i = 0; i < num_sources; i++) {
    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", NULL);

    /* Since the data format in the input file is elementary h264 stream,
    * we need a h264parser */
    h264parser = gst_element_factory_make ("h264parse", NULL);

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", NULL);
    gst_bin_add_many (GST_BIN (pipeline), source, h264parser, decoder, NULL);

    if (!source || !h264parser || !decoder) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
    }

    m_DBScanHandle.reset(
        NvDsInferDBScanCreate(), [](NvDsInferDBScanHandle handle)
        {
                if (handle)
                    NvDsInferDBScanDestroy(handle); });
    if (!m_DBScanHandle)
    {
      g_print("failed to create dbscan handle\n");
    }

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16];
    sprintf (pad_name_sink, "sink_%d", i);
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

    /* Link the elements together */
    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
      g_printerr ("Elements could not be linked: 1. Exiting.\n");
      return -1;
    }
    /* Set the input filename to the source element */
    if (is_nvinfer_server) {
      g_object_set (G_OBJECT (source), "location", argv[i + 3], NULL);
    } else {
      g_object_set (G_OBJECT (source), "location", argv[i + 1], NULL);
    }
  }

  if (!gst_element_link_many (streammux, pgie, queue, sgie1, queue5,
          sgie2, queue2, tiler, queue3, nvvidconv, queue4, nvosd,
          sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  queue_src_pad = gst_element_get_static_pad (queue, "src");
  gst_pad_add_probe (queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, &use_new_mux, NULL);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the sink pad of tiler element which is just after all SGIE elements.
   * Since by that time, GstBuffer would have had got all SGIEs tensor
   * metadata. */
  tiler_sink_pad = gst_element_get_static_pad (tiler, "sink");
  gst_pad_add_probe (tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      sgie_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing...\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Iterate */
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
