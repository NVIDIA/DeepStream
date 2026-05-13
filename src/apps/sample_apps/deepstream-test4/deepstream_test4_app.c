/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <sys/timeb.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvds_yml_parser.h"
#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

#include <sys/stat.h>
#include <stdio.h>

#include "deepstream_test4_yml_parse.h"

//#define ENABLE_DUMP_FILE
#ifdef ENABLE_DUMP_FILE
FILE *fp;
char fileObjNameString[1024];
#endif

//#define MEASURE_ENCODE_TIME
#ifdef MEASURE_ENCODE_TIME
#include <sys/time.h>
#define START_PROFILE \
    { \
  struct timeval t1, t2; \
  double elapsedTime = 0; \
  gettimeofday(&t1, NULL);

#define STOP_PROFILE(X) \
  gettimeofday(&t2, NULL); \
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      \
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   \
  printf("%s ElaspedTime=%f ms\n", X, elapsedTime); \
  }

#else
#define START_PROFILE
#define STOP_PROFILE(X)
#endif

#define MAX_DISPLAY_LEN 64
#define MAX_TIME_STAMP_LEN 32

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define PGIE_CONFIG_FILE  "dstest4_pgie_config.txt"
#define MSCONV_CONFIG_FILE "dstest4_msgconv_config.txt"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define IS_YAML(file) (g_str_has_suffix (file, ".yml") || g_str_has_suffix (file, ".yaml"))

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

static gchar *cfg_file = NULL;
static gchar *input_file = NULL;
static gchar *topic = NULL;
static gchar *conn_str = NULL;
static gchar *proto_lib = NULL;
static gint schema_type = 0;
static gint msg2p_meta = 0;
static gint frame_interval = 30;
static gboolean display_off = FALSE;

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

GOptionEntry entries[] = {
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME, &cfg_file,
      "Set the adaptor config file. Optional if connection string has relevant  details.",
        NULL},
  {"input-file", 'i', 0, G_OPTION_ARG_FILENAME, &input_file,
      "Set the input H264 file", NULL},
  {"topic", 't', 0, G_OPTION_ARG_STRING, &topic,
      "Name of message topic. Optional if it is part of connection string or config file.",
        NULL},
  {"conn-str", 0, 0, G_OPTION_ARG_STRING, &conn_str,
      "Connection string of backend server. Optional if it is part of config file.",
        NULL},
  {"proto-lib", 'p', 0, G_OPTION_ARG_STRING, &proto_lib,
      "Absolute path of adaptor library", NULL},
  {"schema", 's', 0, G_OPTION_ARG_INT, &schema_type,
      "Type of message schema (0=Full, 1=minimal, 2=protobuf), default=0", NULL},
  {"msg2p-meta", 0, 0, G_OPTION_ARG_INT, &msg2p_meta,
      "msg2payload generation metadata type (0=Event Msg meta, 1=nvds meta), default=0",
        NULL},
  {"frame-interval", 0, 0, G_OPTION_ARG_INT, &frame_interval,
      "Frame interval at which payload is generated , default=30", NULL},
  {"no-display", 0, 0, G_OPTION_ARG_NONE, &display_off, "Disable display",
        NULL},
  {NULL}
};

static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6];              //.nnnZ\0

  clock_gettime (CLOCK_REALTIME, &ts);
  memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec / 1000000;
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *) g_memdup2 (srcMeta, sizeof (NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *) g_memdup2 (srcMeta->objSignature.signature,
        srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *srcObj = (NvDsVehicleObject *) srcMeta->extMsg;
      NvDsVehicleObject *obj =
          (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->make)
        obj->make = g_strdup (srcObj->make);
      if (srcObj->model)
        obj->model = g_strdup (srcObj->model);
      if (srcObj->color)
        obj->color = g_strdup (srcObj->color);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->region)
        obj->region = g_strdup (srcObj->region);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsVehicleObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
      NvDsPersonObject *obj =
          (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

      obj->age = srcObj->age;

      if (srcObj->gender)
        obj->gender = g_strdup (srcObj->gender);
      if (srcObj->cap)
        obj->cap = g_strdup (srcObj->cap);
      if (srcObj->hair)
        obj->hair = g_strdup (srcObj->hair);
      if (srcObj->apparel)
        obj->apparel = g_strdup (srcObj->apparel);
      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsPersonObject);
    }
  }

  return dstMeta;
}

static void
meta_free_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

  g_free (srcMeta->ts);
  g_free (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if (srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->color)
        g_free (obj->color);
      if (obj->make)
        g_free (obj->make);
      if (obj->model)
        g_free (obj->model);
      if (obj->license)
        g_free (obj->license);
      if (obj->region)
        g_free (obj->region);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

      if (obj->gender)
        g_free (obj->gender);
      if (obj->cap)
        g_free (obj->cap);
      if (obj->hair)
        g_free (obj->hair);
      if (obj->apparel)
        g_free (obj->apparel);
    }
    g_free (srcMeta->extMsg);
    srcMeta->extMsgSize = 0;
  }
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

static void
generate_vehicle_meta (gpointer data)
{
  NvDsVehicleObject *obj = (NvDsVehicleObject *) data;

  obj->type = g_strdup ("sedan");
  obj->color = g_strdup ("blue");
  obj->make = g_strdup ("Bugatti");
  obj->model = g_strdup ("M");
  obj->license = g_strdup ("XX1234");
  obj->region = g_strdup ("CA");
}

static gpointer
meta_copy_func_custom (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *) user_meta->user_meta_data;
  NvDsCustomMsgInfo *dstMeta = NULL;

  dstMeta = (NvDsCustomMsgInfo *) g_memdup2 (srcMeta, sizeof (NvDsCustomMsgInfo));

  if (srcMeta->message)
    dstMeta->message = (gpointer) g_strdup ((const char*)srcMeta->message);
  dstMeta->size = srcMeta->size;

  return dstMeta;
}

static void
meta_free_func_custom (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *) user_meta->user_meta_data;

  if (srcMeta->message)
    g_free (srcMeta->message);
  srcMeta->size = 0;

  g_free (user_meta->user_meta_data);
}

static void
generate_person_meta (gpointer data)
{
  NvDsPersonObject *obj = (NvDsPersonObject *) data;
  obj->age = 45;
  obj->cap = g_strdup ("none");
  obj->hair = g_strdup ("black");
  obj->gender = g_strdup ("male");
  obj->apparel = g_strdup ("formal");
}

static void
generate_event_msg_meta (gpointer data, gint class_id,
    NvDsObjectMeta * obj_params)
{
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
  meta->sensorId = 0;
  meta->placeId = 0;
  meta->moduleId = 0;
  meta->sensorStr = g_strdup ("sensor-0");

  meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

  strncpy (meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

  generate_ts_rfc3339 (meta->ts, MAX_TIME_STAMP_LEN);

  /*
   * This demonstrates how to attach custom objects.
   * Any custom object as per requirement can be generated and attached
   * like NvDsVehicleObject / NvDsPersonObject. Then that object should
   * be handled in payload generator library (nvmsgconv.cpp) accordingly.
   */
  if (class_id == PGIE_CLASS_ID_VEHICLE) {
    meta->type = NVDS_EVENT_MOVING;
    meta->objType = NVDS_OBJECT_TYPE_VEHICLE;
    meta->objClassId = PGIE_CLASS_ID_VEHICLE;

    NvDsVehicleObject *obj =
        (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
    generate_vehicle_meta (obj);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsVehicleObject);
  } else if (class_id == PGIE_CLASS_ID_PERSON) {
    meta->type = NVDS_EVENT_ENTRY;
    meta->objType = NVDS_OBJECT_TYPE_PERSON;
    meta->objClassId = PGIE_CLASS_ID_PERSON;

    NvDsPersonObject *obj =
        (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
    generate_person_meta (obj);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsPersonObject);
  }
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_metadata_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsFrameMeta *frame_meta = NULL;
  NvOSD_TextParams *txt_params = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  gboolean is_first_object = TRUE;
  NvDsMetaList *l_frame, *l_obj;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    // No batch meta attached.
    return GST_PAD_PROBE_OK;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) l_frame->data;

    if (frame_meta == NULL) {
      // Ignore Null frame meta.
      continue;
    }

    is_first_object = TRUE;

    for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      if (obj_meta == NULL) {
        // Ignore Null object.
        continue;
      }

      txt_params = &(obj_meta->text_params);
      if (txt_params->display_text)
        g_free (txt_params->display_text);

      txt_params->display_text = (char *)g_malloc0 (MAX_DISPLAY_LEN);

      g_snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "%s ",
          pgie_classes_str[obj_meta->class_id]);

      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
        vehicle_count++;
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
        person_count++;

      /* Now set the offsets where the string should appear */
      txt_params->x_offset = obj_meta->rect_params.left;
      txt_params->y_offset = obj_meta->rect_params.top - 25;

      /* Font , font-color and font-size */
      txt_params->font_params.font_name = (char *) "Serif";
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

      /*
       * Ideally NVDS_EVENT_MSG_META should be attached to buffer by the
       * component implementing detection / recognition logic.
       * Here it demonstrates how to use / attach that meta data.
       */
      if (is_first_object && !(frame_number % frame_interval)) {
        /* Frequency of messages to be send will be based on use case.
         * Here message is being sent for first object every frame_interval(default=30).
         */

        NvDsEventMsgMeta *msg_meta =
            (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
        msg_meta->bbox.top = obj_meta->rect_params.top;
        msg_meta->bbox.left = obj_meta->rect_params.left;
        msg_meta->bbox.width = obj_meta->rect_params.width;
        msg_meta->bbox.height = obj_meta->rect_params.height;
        msg_meta->frameId = frame_number;
        msg_meta->trackingId = obj_meta->object_id;
        msg_meta->confidence = obj_meta->confidence;
        generate_event_msg_meta (msg_meta, obj_meta->class_id, obj_meta);

        NvDsUserMeta *user_event_meta =
            nvds_acquire_user_meta_from_pool (batch_meta);
        if (user_event_meta) {
          user_event_meta->user_meta_data = (void *) msg_meta;
          user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
          user_event_meta->base_meta.copy_func =
              (NvDsMetaCopyFunc) meta_copy_func;
          user_event_meta->base_meta.release_func =
              (NvDsMetaReleaseFunc) meta_free_func;
          nvds_add_user_meta_to_frame (frame_meta, user_event_meta);
        } else {
          g_print ("Error in attaching event meta to buffer\n");
        }
        is_first_object = FALSE;
      }
    }
  }
  g_print ("Frame Number = %d "
      "Vehicle Count = %d Person Count = %d\n",
      frame_number, vehicle_count, person_count);
  frame_number++;

  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
osd_sink_pad_buffer_image_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsFrameMeta *frame_meta = NULL;
  gboolean is_first_object = TRUE;
  NvDsMetaList *l_frame, *l_obj;
  gchar *encoded_data;
  gchar *message_data;
  gchar *width, *height;
  gchar *ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    // No batch meta attached.
    g_free(ts);
    return GST_PAD_PROBE_OK;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) l_frame->data;

    if (frame_meta == NULL) {
      // Ignore Null frame meta.
      continue;
    }

    is_first_object = TRUE;

    for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      if (obj_meta == NULL) {
        // Ignore Null object.
        continue;
      }

      if (is_first_object && !(frame_number % frame_interval)) {
        /* Frequency of images to be send will be based on use case.
         * Here images is being sent for first object every frame_interval(default=30).
         */
        NvDsUserMetaList *usrMetaList = obj_meta->obj_user_meta_list;
        while (usrMetaList != NULL) {
          NvDsUserMeta *user_event_meta_custom =
                    nvds_acquire_user_meta_from_pool (batch_meta);
          NvDsCustomMsgInfo *msg_custom_meta =
                   (NvDsCustomMsgInfo *) g_malloc0 (sizeof (NvDsCustomMsgInfo));

          NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
          if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *) usrMetaData->user_meta_data;
            START_PROFILE;
            encoded_data = g_base64_encode(enc_jpeg_image->outBuffer, enc_jpeg_image->outLen);
            generate_ts_rfc3339 (ts, MAX_TIME_STAMP_LEN);
            width = g_strdup_printf("%f", obj_meta->detector_bbox_info.org_bbox_coords.width);
            height = g_strdup_printf("%f", obj_meta->detector_bbox_info.org_bbox_coords.height);
            /* Image message fields are separated by ";".
             * Specific Format:  "image;image_format;image_widthximage_height;time;encoded data;"
             * For Example: "image;jpg;640x480;2023-07-31T10:20:13;xxxxxxxxxxx"
             */
            message_data = g_strconcat("image;jpg;", width, "x", height, ";", ts, ";", encoded_data, ";", NULL);
            STOP_PROFILE("Base64 Encode Time ");
            msg_custom_meta->size = strlen(message_data);
            msg_custom_meta->message = g_strdup(message_data);
            if (user_event_meta_custom) {
               user_event_meta_custom->user_meta_data = (void *) msg_custom_meta;
               user_event_meta_custom->base_meta.meta_type = NVDS_CUSTOM_MSG_BLOB;
               user_event_meta_custom->base_meta.copy_func =
                   (NvDsMetaCopyFunc) meta_copy_func_custom;
               user_event_meta_custom->base_meta.release_func =
                   (NvDsMetaReleaseFunc) meta_free_func_custom;
               nvds_add_user_meta_to_frame (frame_meta, user_event_meta_custom);
            } else {
              g_print ("Error in attaching event meta custom to buffer\n");
            }

#ifdef ENABLE_DUMP_FILE
            gsize size = 0;
            snprintf (fileObjNameString, 1024, "%s_%d_%d_%s.jpg",
                  ts, frame_number, frame_meta->batch_id, obj_meta->obj_label);
            guchar *decoded_data = g_base64_decode (encoded_data, &size);
            fp = fopen(fileObjNameString, "wb");
            if(fp) {
              fwrite(decoded_data, size, 1, fp);
              fclose(fp);
            } else {
              g_printerr ("Could not open file!\n");
            }
            g_free(decoded_data);
#endif
            g_free(encoded_data);
            g_free(message_data);
            g_free(width);
            g_free(height);
            usrMetaList = NULL;
          } else {
            usrMetaList = usrMetaList->next;
          }
        }
        is_first_object = FALSE;
      }
    }
  }

  g_free(ts);
  frame_number++;

  return GST_PAD_PROBE_OK;
}

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
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    guint num_rects = 0;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
        num_rects++;
      }
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
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
        objData.saveImg = FALSE;
        objData.attachUsrMeta = TRUE;
        /* Set if Image scaling Required */
        objData.scaleImg = FALSE;
        objData.scaledWidth = 0;
        objData.scaledHeight = 0;
        /* Preset */
        objData.objNum = num_rects;
        /* Quality */
        objData.quality = 80;
        /*Main Function Call */
        nvds_obj_enc_process ((NvDsObjEncCtxHandle)ctx, &objData, ip_surf, obj_meta, frame_meta);
      }
    }
  }

  nvds_obj_enc_finish ((NvDsObjEncCtxHandle)ctx);
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

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL, *nvstreammux;
  GstElement *msgconv = NULL, *msgbroker = NULL, *tee = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstPad *tee_render_pad = NULL;
  GstPad *tee_msg_pad = NULL;
  GstPad *sink_pad = NULL;
  GstPad *src_pad = NULL;
  GstPad *pgie_src_pad = NULL;
  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  ctx = g_option_context_new ("Nvidia DeepStream Test4");
  group = g_option_group_new ("test4", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (ctx, group);
  g_option_context_add_group (ctx, gst_init_get_option_group ());


  if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
    g_option_context_free (ctx);
    g_printerr ("%s", error->message);
    return -1;
  }
  g_option_context_free (ctx);

  if (!proto_lib || !input_file) {
    if (argc > 1 && !IS_YAML (argv[1])) {
      g_printerr ("missing arguments\n");
      g_printerr ("Usage: %s <yml file>\n", argv[0]);
      g_printerr
          ("Usage: %s -i <H264 filename> -p <Proto adaptor library> --conn-str=<Connection string>\n",
          argv[0]);
      return -1;
    } else if (!argv[1]) {
      g_printerr ("missing arguments\n");
      g_printerr ("Usage: %s <yml file>\n", argv[0]);
      g_printerr
          ("Usage: %s -i <H264 filename> -p <Proto adaptor library> --conn-str=<Connection string>\n",
          argv[0]);
      return -1;
    }
  }

  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  if (argc > 1 && IS_YAML(argv[1])) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
  }

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest4-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  nvstreammux = gst_element_factory_make ("nvstreammux", "nvstreammux");

  /* Use nvinfer or nvinferserver to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Create msg converter to generate payload from buffer metadata */
  msgconv = gst_element_factory_make ("nvmsgconv", "nvmsg-converter");

  /* Create msg broker to send payload to server */
  msgbroker = gst_element_factory_make ("nvmsgbroker", "nvmsg-broker");

  /* Create tee to render buffer and send message simultaneously */
  tee = gst_element_factory_make ("tee", "nvsink-tee");

  /* Create queues */
  queue1 = gst_element_factory_make ("queue", "nvtee-que1");
  queue2 = gst_element_factory_make ("queue", "nvtee-que2");

  /* Finally render the osd output */
  if (display_off) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else if (prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!pipeline || !source || !h264parser || !decoder || !nvstreammux || !pgie
      || !nvvidconv || !nvosd || !msgconv || !msgbroker || !tee
      || !queue1 || !queue2 || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (argc > 1 && IS_YAML (argv[1])) {
    RETURN_ON_PARSER_ERROR(nvds_parse_file_source(source, argv[1], "source"));
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(nvstreammux, argv[1], "streammux"));

    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));

    g_object_set (G_OBJECT (msgconv), "config", "dstest4_msgconv_config.yml",
        NULL);
    RETURN_ON_PARSER_ERROR(nvds_parse_msgconv (msgconv, argv[1], "msgconv"));

    RETURN_ON_PARSER_ERROR(nvds_parse_msgbroker(msgbroker, argv[1], "msgbroker"));

    if (display_off) {
      RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink (sink, argv[1], "sink"));
    }
    else if (prop.integrated) {
      RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink (sink, argv[1], "sink"));
    }
    else {
#ifdef __aarch64__
      RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink (sink, argv[1], "sink"));
#else
      RETURN_ON_PARSER_ERROR(nvds_parse_egl_sink (sink, argv[1], "sink"));
#endif
    }

    msg2p_meta = ds_test4_parse_meta_type(argv[1], "msgconv");
    g_print("msg2p_meta = %d\n", msg2p_meta);
  } else {
    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", input_file, NULL);

    g_object_set (G_OBJECT (nvstreammux), "batch-size", 1, NULL);

    g_object_set (G_OBJECT (nvstreammux), "width", MUXER_OUTPUT_WIDTH, "height",
        MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);

    g_object_set (G_OBJECT (msgconv), "config", MSCONV_CONFIG_FILE, NULL);
    g_object_set (G_OBJECT (msgconv), "payload-type", schema_type, NULL);
    g_object_set (G_OBJECT (msgconv), "msg2p-newapi", msg2p_meta, NULL);
    g_object_set (G_OBJECT (msgconv), "frame-interval", frame_interval, NULL);

    g_object_set (G_OBJECT (msgbroker), "proto-lib", proto_lib,
        "conn-str", conn_str, "sync", FALSE, NULL);

    if (topic) {
      g_object_set (G_OBJECT (msgbroker), "topic", topic, NULL);
    }

    if (cfg_file) {
      g_object_set (G_OBJECT (msgbroker), "config", cfg_file, NULL);
    }

    g_object_set (G_OBJECT (sink), "sync", TRUE, NULL);
  }
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, nvstreammux, pgie,
      nvvidconv, nvosd, tee, queue1, queue2, msgconv, msgbroker, sink, NULL);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder -> nvstreammux ->
   * pgie -> nvvidconv -> nvosd -> tee -> video-renderer
   *                                      |
   *                                      |-> msgconv -> msgbroker  */

  sink_pad = gst_element_request_pad_simple (nvstreammux, "sink_0");
  if (!sink_pad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  src_pad = gst_element_get_static_pad (decoder, "src");
  if (!src_pad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (src_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (sink_pad);
  gst_object_unref (src_pad);

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (nvstreammux, pgie, nvvidconv, nvosd, tee, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (queue1, msgconv, msgbroker, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if (!gst_element_link (queue2, sink)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  sink_pad = gst_element_get_static_pad (queue1, "sink");
  tee_msg_pad = gst_element_request_pad_simple (tee, "src_%u");
  tee_render_pad = gst_element_request_pad_simple (tee, "src_%u");
  if (!tee_msg_pad || !tee_render_pad) {
    g_printerr ("Unable to get request pads\n");
    return -1;
  }

  if (gst_pad_link (tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Unable to link tee and message converter\n");
    gst_object_unref (sink_pad);
    return -1;
  }

  gst_object_unref (sink_pad);

  sink_pad = gst_element_get_static_pad (queue2, "sink");
  if (gst_pad_link (tee_render_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Unable to link tee and render\n");
    gst_object_unref (sink_pad);
    return -1;
  }

  gst_object_unref (sink_pad);

  pgie_src_pad = gst_element_get_static_pad (pgie, "src");
  /* Create Context for Object Encoding.
   * Takes GPU ID as a parameter. Passed by user through commandline.
   * Initialized as 0. */
  NvDsObjEncCtxHandle obj_ctx_handle = nvds_obj_enc_create_context (0);
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
  else {
    if (msg2p_meta == 0) {        //generate payload using eventMsgMeta
      gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          osd_sink_pad_buffer_metadata_probe, NULL, NULL);
    } else {                //generate payload using NVDS_CUSTOM_MSG_BLOB
      gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          osd_sink_pad_buffer_image_probe, (gpointer) obj_ctx_handle, NULL);
    }
  }
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
  if (argc > 1 && IS_YAML (argv[1])) {
    g_print ("Using file: %s\n", argv[1]);
  } else {
    g_print ("Now playing: %s\n", input_file);
  }
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");

  nvds_obj_enc_destroy_context (obj_ctx_handle);
  g_free (cfg_file);
  g_free (input_file);
  g_free (topic);
  g_free (conn_str);
  g_free (proto_lib);

  /* Release the request pads from the tee, and unref them */
  gst_element_release_request_pad (tee, tee_msg_pad);
  gst_element_release_request_pad (tee, tee_render_pad);
  gst_object_unref (tee_msg_pad);
  gst_object_unref (tee_render_pad);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
