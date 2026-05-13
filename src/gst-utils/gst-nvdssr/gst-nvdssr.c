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

#include <unistd.h>
#include <stdlib.h>
#include <gst/pbutils/pbutils.h>
#include "gst-nvdssr.h"


#define NVDSSR_LOG(cat, format, ...) \
    g_print("** %s: <%s:%d>: " format "\n", cat, __func__, __LINE__, ##__VA_ARGS__)

#define NVDSSR_LOG_ERROR(format, ...) \
    NVDSSR_LOG("ERROR", format, ##__VA_ARGS__)

#define NVDSSR_LOG_INFO(format, ...) \
    NVDSSR_LOG("INFO", format, ##__VA_ARGS__)

#define NVDSSR_LOG_WARN(format, ...) \
    NVDSSR_LOG("WARN", format, ##__VA_ARGS__)

#define NVDSSR_LOG_DEBUG(format, ...) \
    NVDSSR_LOG("DEBUG", format, ##__VA_ARGS__)

// duration in seconds.
#define NVDSSR_MAX_RECORD_DURATION 300
#define NVDSSR_DEFAULT_RECORD_DURATION 10
#define NVDSSR_DEFAULT_CACHE_DURATION 30

typedef struct NvDsSRContextPriv
{
  guint timeoutSrcId;
  GThread *userCallbackThread;
  GCond resetCond;
  GstElement *muxer;
  GstElement *encQue;
  GstElement *encAudioQue;
  GstElement *recordAudioQue;
  gboolean haveVideo;
  gboolean haveAudio;
  GstClockTime lastPts;
  GstClockTime lastPtsAudio;
  gboolean haveInvalidPts;
  /* Probe IDs to allow safe removal during destroy */
  gulong videoProbeId;
  gulong audioProbeId;
} NvDsSRContextPriv;

/**
 * Function to set new file name on filesink to write the buffer
 * content in it.
 */
static void
SetNewFileName (NvDsSRContext *ctx)
{
  gchar *filename = NULL;
  char time_data[16];
  gchar *fileExt;
  time_t rawtime;
  struct tm tm_log;
  static int counter = 0;

  g_return_if_fail (ctx);

  switch (ctx->initParams.containerType) {
    case NVDSSR_CONTAINER_MP4:
      fileExt = "mp4";
      break;
    case NVDSSR_CONTAINER_MKV:
      fileExt = "mkv";
      break;
    default:
      fileExt = "mp4";
      break;
  }

  time (&rawtime);
  gmtime_r (&rawtime, &tm_log);
  strftime(time_data, 16, "%Y%m%d-%H%M%S", &tm_log);

  if (ctx->initParams.dirpath) {
    filename = g_strdup_printf ("%s/%s_%05d_%s_%ld.%s", ctx->initParams.dirpath,
                                ctx->initParams.fileNamePrefix, counter++,
                                time_data, (long) getpid (), fileExt);
  } else {
    filename = g_strdup_printf ("%s_%05d_%s_%ld.%s", ctx->initParams.fileNamePrefix,
                                counter++, time_data, (long) getpid (), fileExt);
  }

  gst_element_set_state (ctx->filesink, GST_STATE_NULL);
  g_object_set (G_OBJECT (ctx->filesink), "location", filename, NULL);
  gst_element_set_locked_state (ctx->filesink, FALSE);
  gst_element_set_state (ctx->filesink, GST_STATE_PLAYING);
  g_free(filename);
}

/**
 * Function to reset the encodebin after the EOS event so that
 * encodebin can again accept new buffers after start.
 */
static gpointer
ResetEncodeBin (gpointer data)
{
  NvDsSRContextPriv *privData;
  NvDsSRContext *ctx = (NvDsSRContext *) data;
  g_return_val_if_fail (ctx, NULL);

  privData = (NvDsSRContextPriv *) ctx->privData;
  g_usleep (10000);

  if (gst_element_set_state (ctx->encodebin, GST_STATE_READY) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (ctx->encodebin, "Failed in resetting elements");
    return NULL;
  }

  // Reset the filename otherwise eos again from other component might
  // corrupt the file.
  gst_element_set_state (ctx->filesink, GST_STATE_NULL);
  g_object_set (G_OBJECT (ctx->filesink), "location", "/dev/null", NULL);

  if (!gst_element_sync_state_with_parent (ctx->encodebin)) {
    GST_ERROR_OBJECT (ctx->encodebin, "Couldn't sync state with parent");
  }
  g_mutex_lock(&ctx->flowLock);
  ctx->resetDone = TRUE;
  g_cond_signal (&privData->resetCond);
  g_mutex_unlock(&ctx->flowLock);

  return NULL;
}

/**
 * Function to pass recorded file info to user through registered
 * callback function.
 */
static gpointer
RunUserCallback (gpointer data)
{
  g_return_val_if_fail (data != NULL, FALSE);

  gchar *rPath = NULL;
  GError *err = NULL;
  GstDiscoverer *discoverer;
  GstDiscovererInfo *disInfo;
  GList *vList, *aList;
  guint64 duration;

  NvDsSRRecordingInfo *info = (NvDsSRRecordingInfo *) data;
  NvDsSRContext *ctx = info->ctx;
  NvDsSRContextPriv *privCtx = (NvDsSRContextPriv *) ctx->privData;
  gchar *file = info->filename;

  if (file) {
    if (!g_strcmp0(file, "/dev/null")) {
      NVDSSR_LOG_ERROR ("No recorded file.");
      goto error;
    }

    gchar *tmp = g_strrstr(file, "/");
    if (tmp) {
      info->filename = g_strdup(tmp + 1);
      g_free (file);
    }
  } else {
    NVDSSR_LOG_ERROR ("NULL recorded filename.");
    g_free (info);
    return NULL;
  }

  if (ctx->initParams.dirpath) {
    rPath = realpath(ctx->initParams.dirpath, NULL);
  } else {
    rPath = realpath("./", NULL);
  }

  file = g_strconcat("file://", rPath, "/", info->filename, NULL);
  g_free (rPath);

  discoverer = gst_discoverer_new (GST_SECOND, &err);
  if (!discoverer) {
    NVDSSR_LOG_ERROR ("Error in creating discoverer instance - %s", err->message);
    g_error_free (err);
    g_free (file);
    goto error;
  }

  disInfo = gst_discoverer_discover_uri (discoverer, file, &err);
  if (!disInfo) {
    NVDSSR_LOG_ERROR ("Error in getting info of file %s - %s", file, err->message);
    g_error_free (err);
    g_free(file);
    goto error;
  }
  g_free(file);

  // value is in nanoseconds, convert it to milliseconds.
  duration = gst_discoverer_info_get_duration (disInfo);
  duration = duration / 1000000;

  vList = gst_discoverer_info_get_video_streams (disInfo);
  if (!vList && privCtx->haveVideo) {
    NVDSSR_LOG_ERROR ("No video stream found");
  }

  aList = gst_discoverer_info_get_audio_streams (disInfo);
  if (!aList && privCtx->haveAudio) {
    NVDSSR_LOG_ERROR ("No audio stream found");
  }

  if (!vList && !aList) {
    goto error;
  }

  if (vList) {
    GstDiscovererVideoInfo *vinfo = (GstDiscovererVideoInfo *) vList->data;
    info->height = gst_discoverer_video_info_get_height (vinfo);
    info->width = gst_discoverer_video_info_get_width (vinfo);

    info->containsVideo = TRUE;

    gst_discoverer_stream_info_list_free (vList);
  }

  if (aList) {
    GstDiscovererAudioInfo *ainfo = (GstDiscovererAudioInfo *) aList->data;
    info->channels = gst_discoverer_audio_info_get_channels (ainfo);
    info->samplingRate = gst_discoverer_audio_info_get_sample_rate (ainfo);

    info->containsAudio = TRUE;

    gst_discoverer_stream_info_list_free (aList);
  }

  gst_object_unref (disInfo);
  gst_object_unref (discoverer);

  info->containerType = ctx->initParams.containerType;
  info->dirpath = ctx->initParams.dirpath;
  info->sessionId = 0;
  info->duration = duration;

  ctx->initParams.callback(info, ctx->uData);

error:
  g_free (info->filename);
  g_free (info);

  return NULL;
}

/**
 * Buffer probe function on source pad of queue.
 * This function decides when to start writing buffer to file.
 */
static GstPadProbeReturn
queue_src_pad_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsSRContext *ctx = (NvDsSRContext *) u_data;
  NvDsSRContextPriv *privData = ctx->privData;

  g_mutex_lock(&ctx->flowLock);
  if (ctx->recordOn && !ctx->gotKeyFrame &&
     (!GST_BUFFER_FLAG_IS_SET (GST_BUFFER_CAST(info->data), GST_BUFFER_FLAG_DELTA_UNIT))) {
     ctx->gotKeyFrame = TRUE;
  }

  if (GST_BUFFER_PTS(info->data) == GST_CLOCK_TIME_NONE) {
    GST_BUFFER_PTS(info->data) = privData->lastPts + GST_USECOND;
    if (!privData->haveInvalidPts) {
      NVDSSR_LOG_WARN(
          "Invalid PTS found in stream. Stream recorded by Smart Record might have issues");
    }
    privData->haveInvalidPts = TRUE;
  }

  privData->lastPts = GST_BUFFER_PTS(info->data);

  /* Drop if recording not ON or till first I frame */
  if (ctx->recordOn && ctx->gotKeyFrame) {
    g_mutex_unlock(&ctx->flowLock);
    return GST_PAD_PROBE_OK;
  }
  g_mutex_unlock(&ctx->flowLock);
  return GST_PAD_PROBE_DROP;
}

static GstPadProbeReturn
audio_queue_src_pad_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsSRContext *ctx = (NvDsSRContext *) u_data;
  NvDsSRContextPriv *privData = ctx->privData;

  if (GST_BUFFER_PTS(info->data) == GST_CLOCK_TIME_NONE) {
    GST_BUFFER_PTS(info->data) = privData->lastPtsAudio + GST_USECOND;
    if (!privData->haveInvalidPts) {
      NVDSSR_LOG_WARN(
          "Invalid PTS found in stream. Stream recorded by Smart Record might have issues");
    }
    privData->haveInvalidPts = TRUE;
  }

  privData->lastPtsAudio = GST_BUFFER_PTS(info->data);

  return ctx->recordOn ? GST_PAD_PROBE_OK : GST_PAD_PROBE_DROP;
}

/**
 * Function to create encodebin.
 */
static gboolean
CreateEncodeBin (NvDsSRContext *ctx)
{
  static int instanceId = 0;
  gchar elem_name[50];
  GstPad *sinkPad = NULL;
  GstElement *muxer = NULL;
  GstElement *encQue, *encAudioQue, *filesink;
  GstElement *encodebin;
  NvDsSRContextPriv *privData = (NvDsSRContextPriv *) ctx->privData;

  g_snprintf (elem_name, sizeof (elem_name), "mux_elem%d", instanceId);
  switch (ctx->initParams.containerType) {
    case NVDSSR_CONTAINER_MP4:
      muxer = gst_element_factory_make ("qtmux", elem_name);
      break;
    case NVDSSR_CONTAINER_MKV:
      muxer = gst_element_factory_make ("matroskamux", elem_name);
      break;
    default:
      NVDSSR_LOG_ERROR ("muxer type(%d) not supported", ctx->initParams.containerType);
      return FALSE;
  }
  if (!muxer) {
   NVDSSR_LOG_ERROR ("failed to create muxer(%s)", elem_name);
   return FALSE;
  }

  g_snprintf (elem_name, sizeof (elem_name), "enc_que%d", instanceId);
  encQue = gst_element_factory_make ("queue", elem_name);
  if (!encQue) {
    NVDSSR_LOG_ERROR ("failed to create encoder queue(%s)", elem_name);
    return FALSE;
  }

  /* Allow unlimited data to be queued in queue just before muxer to accomodate skew
   * in PTS of audio / video bitstream buffers. */
  g_object_set(G_OBJECT(encQue), "max-size-bytes", 0, "max-size-time", 0,
      "max-size-buffers", 0, NULL);

  g_snprintf (elem_name, sizeof (elem_name), "enc_audio_que%d", instanceId);
  encAudioQue = gst_element_factory_make ("queue", elem_name);
  if (!encAudioQue) {
    NVDSSR_LOG_ERROR ("failed to create encoder queue(%s)", elem_name);
    return FALSE;
  }

  /* Allow unlimited data to be queued in queue just before muxer to accomodate skew
   * in PTS of audio / video bitstream buffers. */
  g_object_set(G_OBJECT(encAudioQue), "max-size-bytes", 0, "max-size-time", 0,
      "max-size-buffers", 0, NULL);

  g_snprintf (elem_name, sizeof (elem_name), "src_filesink%d", instanceId);
  filesink = gst_element_factory_make ("filesink", elem_name);
  if (!filesink) {
    NVDSSR_LOG_ERROR ("failed to create filesink");
    return FALSE;
  }

  g_object_set (G_OBJECT(filesink), "location", "/dev/null",
                "async", FALSE, NULL);

  g_snprintf (elem_name, sizeof (elem_name), "enc_bin%d", instanceId);
  encodebin = gst_bin_new (elem_name);
  if (!encodebin) {
    NVDSSR_LOG_ERROR ("failed to create encode bin");
    return FALSE;
  }

  gst_bin_add_many (GST_BIN(encodebin), encQue, encAudioQue, muxer, filesink, NULL);

  sinkPad = gst_element_get_static_pad (encQue, "sink");
  if (!sinkPad) {
    NVDSSR_LOG_ERROR("Could not find sinkpad in '%s'", GST_ELEMENT_NAME(encQue));
    return FALSE;
  }
  gst_element_add_pad (encodebin, gst_ghost_pad_new ("sink", sinkPad));
  gst_object_unref (sinkPad);

  sinkPad = gst_element_get_static_pad (encAudioQue, "sink");
  if (!sinkPad) {
    NVDSSR_LOG_ERROR("Could not find sinkpad in '%s'", GST_ELEMENT_NAME(encAudioQue));
    return FALSE;
  }
  gst_element_add_pad (encodebin, gst_ghost_pad_new ("asink", sinkPad));
  gst_object_unref (sinkPad);

  if (!gst_element_link (muxer, filesink)) {
    NVDSSR_LOG_ERROR("Couldn't link elements: '%s', '%s'",
                     GST_ELEMENT_NAME(muxer), GST_ELEMENT_NAME(filesink));
    return FALSE;
  }

  // TODO: Handle multiple parallel record on same source
  // For that convert encodebin to list of structures having details of encodebins.
  ctx->encodebin = encodebin;
  ctx->filesink = filesink;

  privData->encQue = encQue;
  privData->encAudioQue = encAudioQue;
  privData->muxer = muxer;

  g_object_set (encodebin, "message-forward", TRUE, NULL);

  instanceId++;
  return TRUE;
}

/**
 * Original message handler of the bus. This handler is replaced with
 * custom message handler during NvDsSRCreate() and it is restored back
 * during NvDsSRDestroy().
 */
static GstBusSyncReply
bin_bus_handler (GstBus * bus, GstMessage * message, GstBin * bin)
{
  GstBinClass *bclass;

  bclass = GST_BIN_GET_CLASS (bin);
  if (bclass->handle_message)
    bclass->handle_message (bin, message);
  else
    gst_message_unref (message);

  return GST_BUS_DROP;
}

/**
 * Custom message handler to receive synchronous messages posted on recordbin's child bus.
 * These messages are posted by children of recorbin. We are only looking for
 * "ELEMENT" type of messages posted by encodebin to decide the
 * completion of file write successfully and for other messages default handler
 * is called.
 */
static GstBusSyncReply
nvds_bin_bus_handler (GstBus * bus, GstMessage * message, NvDsSRContext *ctx)
{
  GstBinClass *bclass;

  g_return_val_if_fail (ctx, GST_BUS_DROP);

  if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ELEMENT &&
      GST_MESSAGE_SRC (message) == GST_OBJECT (ctx->encodebin)) {

    const GstStructure *structure;
    structure = gst_message_get_structure (message);

    if (gst_structure_has_name (structure, "GstBinForwarded")) {
      GstMessage *child_msg;

      if (gst_structure_has_field (structure, "message")) {
        const GValue *val = gst_structure_get_value (structure, "message");
        if (G_VALUE_TYPE (val) == GST_TYPE_MESSAGE) {
          child_msg = (GstMessage *) g_value_get_boxed (val);
          if (GST_MESSAGE_TYPE (child_msg) == GST_MESSAGE_EOS) {
            /*
             * Here we have bin forwarded EOS from encodebin.
             * That means either file write is complete. OR
             * It is due to stream EOS.
             * 1) Trigger callback to notify the user if it is due
             *    to file write.
             * 2) Drop the message.
             */

            // Due to stream EOS. Drop the message.
            if (ctx->resetDone) {
              gst_message_unref (message);
              return GST_BUS_DROP;
            }

            if (ctx->initParams.callback) {
              gchar *file;
              NvDsSRContextPriv *privData = (NvDsSRContextPriv *) ctx->privData;
              NvDsSRRecordingInfo *info = g_new0 (NvDsSRRecordingInfo, 1);

              g_object_get (G_OBJECT(ctx->filesink), "location", &file, NULL);

              info->ctx = ctx;
              info->filename = file;

              if (privData->userCallbackThread)
                g_thread_unref (privData->userCallbackThread);

              privData->userCallbackThread = g_thread_new (NULL, RunUserCallback, info);
            }

            if (ctx->resetThread)
              g_thread_unref (ctx->resetThread);

            ctx->resetThread = g_thread_new (NULL, ResetEncodeBin, ctx);
            gst_message_unref (message);
            return GST_BUS_DROP;
          }
        }
      }
    }
  } else if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_STATE_CHANGED &&
             GST_MESSAGE_SRC (message) == GST_OBJECT (ctx->encodebin)) {
    GstState old_state = GST_STATE_NULL, new_state = GST_STATE_NULL;

    gst_message_parse_state_changed (message, &old_state, &new_state, NULL);
    g_mutex_lock (&ctx->flowLock);
    if (new_state == GST_STATE_PLAYING)
      ctx->isPlaying = TRUE;
    else
      ctx->isPlaying = FALSE;
    g_mutex_unlock (&ctx->flowLock);
  }

  bclass = GST_BIN_GET_CLASS (ctx->recordbin);
  if (bclass && bclass->handle_message) {
    bclass->handle_message (GST_BIN(ctx->recordbin), message);
  }
  else
    gst_message_unref (message);

  return GST_BUS_DROP;
}

/**
 * Function called to take care of stopping the record in case
 * NvDsSRStop() is not called by the user.
 */
static gboolean
DefaultStopCallback (gpointer uData)
{
  NvDsSRContext *ctx = (NvDsSRContext *) uData;
  NvDsSRContextPriv *privData;
  gboolean recordOn;

  g_return_val_if_fail (ctx != NULL, FALSE);

  privData = (NvDsSRContextPriv *) ctx->privData;

  g_mutex_lock (&ctx->flowLock);
  recordOn = ctx->recordOn;
  /* This timeout source will be removed after exit from this function.
   * Set source id to zero to avoid its removal again inside stop().
   */
  privData->timeoutSrcId = 0;
  g_mutex_unlock (&ctx->flowLock);

  if (recordOn)
    NvDsSRStop (ctx, 0);

  return FALSE;
}

static GstPadLinkReturn
nvds_srbin_ghost_pad_linked(GstPad *pad, GstObject *parent, GstPad *peer)
{
  NvDsSRContext *ctx = (NvDsSRContext *) g_object_get_data (G_OBJECT (pad), "nvdssr-context");
  NvDsSRContextPriv *privData = (NvDsSRContextPriv *) ctx->privData;
  GstPad *sinkPad, *srcPad;

  if (!ctx) return GST_PAD_LINK_REFUSED;

  gboolean isAudio = !g_strcmp0(GST_PAD_NAME(pad), "asink");

  if (isAudio) {
    privData->haveAudio = TRUE;
    sinkPad = gst_element_request_pad_simple (privData->muxer, "audio_%u");
    srcPad = gst_element_get_static_pad (privData->encAudioQue, "src");
  } else {
    privData->haveVideo = TRUE;
    sinkPad = gst_element_request_pad_simple (privData->muxer, "video_%u");
    srcPad = gst_element_get_static_pad (privData->encQue, "src");
  }

  if (gst_pad_link (srcPad, sinkPad) != GST_PAD_LINK_OK)
    return GST_PAD_LINK_REFUSED;

  gst_object_unref (sinkPad);
  gst_object_unref (srcPad);

  return GST_PAD_LINK_OK;
}


static GstElement *
create_cache_queue(NvDsSRContext *ctx, int instanceId, gboolean isAudio)
{
  GstElement *queue;
  gchar elem_name[50];
  GstPad *srcPad, *sinkPad;
  GstPad *ghostPad;
  NvDsSRContextPriv *privData = (NvDsSRContextPriv *) ctx->privData;

  g_snprintf (elem_name, sizeof (elem_name), isAudio ? "audio_cache_que%d" : "cache_que%d", instanceId);

  queue = gst_element_factory_make ("queue", elem_name);
  if (!queue) {
    NVDSSR_LOG_ERROR("failed to create cache queue");
    return NULL;
  }

  // Disable number of buffers and size of content based buffer chaching
  // decision in queue so that only duration of content wiil be used.
  g_object_set (G_OBJECT (queue), "max-size-buffers", 0, NULL);
  g_object_set (G_OBJECT (queue), "max-size-bytes", 0, NULL);

  // Set chache threshold values of queue.
  g_object_set (G_OBJECT (queue), "min-threshold-time",
                ctx->initParams.cacheSize*1000*1000*1000ULL, NULL);

  // Five seconds more than minimum video cache size.
  g_object_set (G_OBJECT (queue), "max-size-time",
                1000000000ULL * (ctx->initParams.cacheSize + 5), NULL);

  // set queue to leak old data in case of overflow.
  g_object_set (G_OBJECT (queue), "leaky", 2, "silent", TRUE, NULL);

  srcPad = gst_element_get_static_pad (queue, "src");
  if (isAudio) {
    privData->audioProbeId = gst_pad_add_probe (srcPad, GST_PAD_PROBE_TYPE_BUFFER,
        audio_queue_src_pad_probe, ctx, NULL);
  } else {
    privData->videoProbeId = gst_pad_add_probe (srcPad, GST_PAD_PROBE_TYPE_BUFFER,
        queue_src_pad_probe, ctx, NULL);
  }
  gst_object_unref (srcPad);

  gst_bin_add_many (GST_BIN(ctx->recordbin), queue, NULL);

  sinkPad = gst_element_get_static_pad (queue, "sink");

  ghostPad = gst_ghost_pad_new (isAudio ? "asink" : "sink", sinkPad);

  g_object_set_data (G_OBJECT (ghostPad), "nvdssr-context", ctx);
  gst_pad_set_link_function(ghostPad, nvds_srbin_ghost_pad_linked);

  gst_element_add_pad (ctx->recordbin, ghostPad);
  gst_object_unref (sinkPad);

  return queue;
}

NvDsSRStatus NvDsSRCreate (NvDsSRContext **pCtx, NvDsSRInitParams *params)
{
  static int instanceId = 0;
  GstPad *srcPad, *sinkPad;
  gchar elem_name[50];
  GstElement *queue, *tee;
  static GMutex pbLock;

  g_return_val_if_fail (pCtx != NULL, NVDSSR_STATUS_INVALID_VAL);
  g_return_val_if_fail (params != NULL, NVDSSR_STATUS_INVALID_VAL);

  g_mutex_lock (&pbLock);
  gst_pb_utils_init ();
  g_mutex_unlock (&pbLock);

  NvDsSRContext *ctx = g_new0 (NvDsSRContext, 1);
  NvDsSRContextPriv *privData =  g_new0 (NvDsSRContextPriv, 1);
  ctx->privData = (gpointer) privData;

  ctx->resetDone = TRUE;
  ctx->initParams = *params;
  if (!params->fileNamePrefix) {
    ctx->initParams.fileNamePrefix = g_strdup ("Smart_Record");
  } else {
    ctx->initParams.fileNamePrefix = g_strdup (params->fileNamePrefix);
  }

  if (params->dirpath)
    ctx->initParams.dirpath = g_strdup (params->dirpath);

  if (!params->defaultDuration) {
    // if duration is not set, set it to default value.
    ctx->initParams.defaultDuration = NVDSSR_DEFAULT_RECORD_DURATION;
  }

  // should there be max cache size limit?
  if (!params->cacheSize) {
    ctx->initParams.cacheSize = NVDSSR_DEFAULT_CACHE_DURATION;
  }

  g_snprintf (elem_name, sizeof (elem_name), "record_bin%d", instanceId);
  ctx->recordbin = gst_bin_new (elem_name);
  if (!ctx->recordbin) {
    NVDSSR_LOG_ERROR("failed to create record bin");
    goto error;
  }
  gst_object_ref (ctx->recordbin);

  queue = create_cache_queue(ctx, instanceId, FALSE);
  ctx->recordQue = queue;

  g_snprintf (elem_name, sizeof (elem_name), "recordbin_tee%d", instanceId);
  tee = gst_element_factory_make ("tee", elem_name);
  if (!tee) {
    NVDSSR_LOG_ERROR ("failed to create encode tee");
    goto error;
  }

  gst_bin_add (GST_BIN(ctx->recordbin), tee);

  if (!gst_element_link (queue, tee)) {
    NVDSSR_LOG_ERROR ("failed to link (%s) and (%s)", GST_ELEMENT_NAME(queue),
                      GST_ELEMENT_NAME(tee));
    goto error;
  }

  if (!CreateEncodeBin (ctx)) {
    NVDSSR_LOG_ERROR ("failed to create encodebin");
    goto error;
  }

  gst_bin_add (GST_BIN(ctx->recordbin), ctx->encodebin);

  srcPad = gst_element_request_pad_simple (tee, "src_%u");
  sinkPad = gst_element_get_static_pad (ctx->encodebin, "sink");
  if (!srcPad || !sinkPad) {
    NVDSSR_LOG_ERROR("failed to access pads");
    goto error;
  }

  if (gst_pad_link (srcPad, sinkPad) != GST_PAD_LINK_OK) {
    NVDSSR_LOG_ERROR("linking of encodebin to tee srcpad failed");
    goto error;
  }
  gst_object_unref (srcPad);
  gst_object_unref (sinkPad);

  privData->recordAudioQue = create_cache_queue(ctx, instanceId, TRUE);

  g_snprintf (elem_name, sizeof (elem_name), "recordbin_audio_tee%d", instanceId);
  tee = gst_element_factory_make ("tee", elem_name);
  if (!tee) {
    NVDSSR_LOG_ERROR ("failed to create encode tee");
    goto error;
  }

  gst_bin_add (GST_BIN(ctx->recordbin), tee);

  if (!gst_element_link (privData->recordAudioQue, tee)) {
    NVDSSR_LOG_ERROR ("failed to link (%s) and (%s)", GST_ELEMENT_NAME(privData->recordAudioQue),
                      GST_ELEMENT_NAME(tee));
    goto error;
  }

  srcPad = gst_element_request_pad_simple (tee, "src_%u");
  sinkPad = gst_element_get_static_pad (ctx->encodebin, "asink");
  if (!srcPad || !sinkPad) {
    NVDSSR_LOG_ERROR("failed to access pads");
    goto error;
  }

  if (gst_pad_link (srcPad, sinkPad) != GST_PAD_LINK_OK) {
    NVDSSR_LOG_ERROR("linking of encodebin to queue srcpad failed");
    goto error;
  }
  gst_object_unref (srcPad);
  gst_object_unref (sinkPad);

  g_mutex_init (&ctx->flowLock);
  g_cond_init (&privData->resetCond);

  /**
   * Bus of encodebin is the the child bus of parent bin (recordbin).
   * We need to access the messages posted by encodebin to identify the
   * EOS message. To achieve that, remove the default sync handler of the
   * bus and install the custom handler. From the custom handler, we call the
   * default message handler of bin after getting EOS message of element type
   * from encodebin.
   */
  GstBus *bus = gst_element_get_bus (GST_ELEMENT(ctx->encodebin));
  gst_bus_set_sync_handler (bus, NULL, NULL, NULL);
  gst_bus_set_sync_handler (bus, (GstBusSyncHandler) nvds_bin_bus_handler, ctx,
                            NULL);
  gst_object_unref (bus);

  *pCtx = ctx;

  instanceId++;

  return NVDSSR_STATUS_OK;

error:
  if (ctx && ctx->recordbin) {
    gst_object_unref(ctx->recordbin);
    ctx->recordbin = NULL;
  }
  if (ctx) {
    g_free (ctx->privData);
    g_free (ctx);
  }
  return NVDSSR_STATUS_ERROR;
}

NvDsSRStatus
NvDsSRStart (NvDsSRContext *ctx, NvDsSRSessionId *sessionId,
             guint startTime, guint duration, gpointer userData)
{
  guint timeout;
  NvDsSRContextPriv *privData;

  g_return_val_if_fail (ctx != NULL, NVDSSR_STATUS_INVALID_VAL);
  g_return_val_if_fail (sessionId != NULL, NVDSSR_STATUS_INVALID_VAL);

  privData = (NvDsSRContextPriv *) ctx->privData;

  g_mutex_lock (&ctx->flowLock);
  if (ctx->recordOn || !ctx->resetDone) {
    // We have active recording, ignore this record.
    // TODO: Handle multiple parallel records.
    g_mutex_unlock (&ctx->flowLock);
    return NVDSSR_STATUS_INVALID_OP;
  }
  SetNewFileName (ctx);
  privData->haveInvalidPts = FALSE;
  ctx->recordOn = TRUE;
  ctx->resetDone = FALSE;
  ctx->uData = userData;

  if (!startTime) {
    // Drop all cache and start recording from current time.
    // Max threshold should be non-zero value to trigger cache drop in "queue",
    startTime = 1;
  }

  g_object_set (G_OBJECT (ctx->recordQue), "max-size-time",
                startTime * 1000 * 1000 * 1000ULL, NULL);

  g_object_set (G_OBJECT (ctx->recordQue), "min-threshold-time", 0, NULL);

  g_object_set (G_OBJECT (privData->recordAudioQue), "max-size-time",
                startTime * 1000 * 1000 * 1000ULL, NULL);

  g_object_set (G_OBJECT (privData->recordAudioQue), "min-threshold-time", 0, NULL);

  if (duration && duration < NVDSSR_MAX_RECORD_DURATION)
    timeout = duration * 1000;
  else
    timeout = ctx->initParams.defaultDuration * 1000;

  privData->timeoutSrcId = g_timeout_add (timeout, DefaultStopCallback, ctx);
  g_mutex_unlock (&ctx->flowLock);

  // Only single record per source.
  *sessionId = 0;

  return NVDSSR_STATUS_OK;
}

NvDsSRStatus NvDsSRStop (NvDsSRContext *ctx, NvDsSRSessionId sessionId)
{
  NvDsSRContextPriv *privData;

  g_return_val_if_fail (ctx != NULL, NVDSSR_STATUS_INVALID_VAL);

  privData = (NvDsSRContextPriv *) ctx->privData;

  g_mutex_lock (&ctx->flowLock);
  if (!ctx->recordOn) {
    // Record stop without starting of record, just return.
    g_mutex_unlock (&ctx->flowLock);
    return NVDSSR_STATUS_INVALID_OP;
  }
  g_object_set (G_OBJECT (ctx->recordQue), "min-threshold-time",
                ctx->initParams.cacheSize * 1000 * 1000 * 1000ULL, NULL);
  g_object_set (G_OBJECT (ctx->recordQue), "max-size-time",
                1000000000ULL * (ctx->initParams.cacheSize + 5), NULL);
  g_object_set (G_OBJECT (privData->recordAudioQue), "min-threshold-time",
                ctx->initParams.cacheSize * 1000 * 1000 * 1000ULL, NULL);
  g_object_set (G_OBJECT (privData->recordAudioQue), "max-size-time",
                1000000000ULL * (ctx->initParams.cacheSize + 5), NULL);

  ctx->gotKeyFrame = FALSE;
  ctx->recordOn = FALSE;

  if (privData->timeoutSrcId) {
    g_source_remove (privData->timeoutSrcId);
    privData->timeoutSrcId = 0;
  }

  g_mutex_unlock (&ctx->flowLock);

  if (privData->haveVideo)
    gst_pad_send_event (gst_element_get_static_pad (ctx->encodebin, "sink"),
                        gst_event_new_eos ());
  if (privData->haveAudio)
    gst_pad_send_event (gst_element_get_static_pad (ctx->encodebin, "asink"),
                        gst_event_new_eos ());

  return NVDSSR_STATUS_OK;
}

NvDsSRStatus NvDsSRDestroy (NvDsSRContext *ctx)
{
  g_return_val_if_fail (ctx != NULL, NVDSSR_STATUS_INVALID_VAL);

  gboolean recordOn, isPlaying;
  NvDsSRContextPriv *privData = (NvDsSRContextPriv *) ctx->privData;

  g_mutex_lock (&ctx->flowLock);
  recordOn = ctx->recordOn;
  isPlaying = ctx->isPlaying;
  g_mutex_unlock (&ctx->flowLock);

  /* Remove pad probes early to prevent callbacks during teardown */
  {
    GstPad *vsrc = NULL, *asrc = NULL;
    if (ctx->recordQue) {
      vsrc = gst_element_get_static_pad (ctx->recordQue, "src");
      if (vsrc && privData->videoProbeId) {
        gst_pad_remove_probe (vsrc, privData->videoProbeId);
        privData->videoProbeId = 0;
      }
    }
    if (privData->recordAudioQue) {
      asrc = gst_element_get_static_pad (privData->recordAudioQue, "src");
      if (asrc && privData->audioProbeId) {
        gst_pad_remove_probe (asrc, privData->audioProbeId);
        privData->audioProbeId = 0;
      }
    }
    if (vsrc) gst_object_unref (vsrc);
    if (asrc) gst_object_unref (asrc);
  }

  if (recordOn && isPlaying) {
    NvDsSRStop (ctx, 0);

    g_mutex_lock (&ctx->flowLock);
    while (!ctx->resetDone) {
      g_cond_wait (&privData->resetCond, &ctx->flowLock);
    }
    g_mutex_unlock (&ctx->flowLock);
  }

  if (ctx->resetThread)
    g_thread_join (ctx->resetThread);

  if (privData->userCallbackThread)
    g_thread_join (privData->userCallbackThread);

  // Re install original message handler on bus.
  GstBus *bus = gst_element_get_bus (GST_ELEMENT (ctx->encodebin));
  gst_bus_set_sync_handler (bus, NULL, NULL, NULL);
  gst_bus_set_sync_handler (bus, (GstBusSyncHandler) bin_bus_handler,
                            GST_BIN (ctx->recordbin), NULL);
  gst_object_unref (bus);

  gst_object_unref (ctx->recordbin);

  g_mutex_clear (&ctx->flowLock);
  g_cond_clear (&privData->resetCond);
  g_free (ctx->initParams.fileNamePrefix);
  g_free (ctx->initParams.dirpath);
  g_free (ctx->privData);
  g_free (ctx);

  return NVDSSR_STATUS_OK;
}
