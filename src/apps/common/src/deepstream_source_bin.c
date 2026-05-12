/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "deepstream_common.h"
#include "deepstream_sources.h"
#include "deepstream_dewarper.h"
#include <gst/rtp/gstrtcpbuffer.h>
#include <gst/rtsp/gstrtsptransport.h>
#include <cuda_runtime_api.h>

#include "nvdsgstutils.h"
#include "gst-nvdssr.h"
#include "gst-nvevent.h"

#define SRC_CONFIG_KEY "src_config"
#define SOURCE_RESET_INTERVAL_SEC 60

GST_DEBUG_CATEGORY_EXTERN (NVDS_APP);
GST_DEBUG_CATEGORY_EXTERN (APP_CFG_PARSER_CAT);

static gboolean install_mux_eosmonitor_probe = FALSE;

static gboolean
set_camera_csi_params (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  g_object_set (G_OBJECT (bin->src_elem), "sensor-id",
      config->camera_csi_sensor_id, NULL);

  GST_CAT_DEBUG (NVDS_APP, "Setting csi camera params successful");

  return TRUE;
}

static gboolean
set_camera_v4l2_params (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  gchar device[64];

  g_snprintf (device, sizeof (device), "/dev/video%d",
      config->camera_v4l2_dev_node);
  g_object_set (G_OBJECT (bin->src_elem), "device", device, NULL);

  GST_CAT_DEBUG (NVDS_APP, "Setting v4l2 camera params successful");

  return TRUE;
}

static gboolean
create_camera_source_bin (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  GstCaps *caps = NULL, *caps1 = NULL, *convertCaps = NULL;
  gboolean ret = FALSE;

  switch (config->type) {
    case NV_DS_SOURCE_CAMERA_CSI:
      bin->src_elem =
          gst_element_factory_make (NVDS_ELEM_SRC_CAMERA_CSI, "src_elem");
      break;
    case NV_DS_SOURCE_CAMERA_V4L2:
      bin->src_elem =
          gst_element_factory_make (NVDS_ELEM_SRC_CAMERA_V4L2, "src_elem");
      bin->cap_filter1 =
          gst_element_factory_make (NVDS_ELEM_CAPS_FILTER, "src_cap_filter1");
      if (!bin->cap_filter1) {
        NVGSTDS_ERR_MSG_V ("Could not create 'src_cap_filter1'");
        goto done;
      }
      caps1 = gst_caps_new_simple ("video/x-raw",
          "width", G_TYPE_INT, config->source_width, "height", G_TYPE_INT,
          config->source_height, "framerate", GST_TYPE_FRACTION,
          config->source_fps_n, config->source_fps_d, NULL);
      break;
    default:
      NVGSTDS_ERR_MSG_V ("Unsupported source type");
      goto done;
  }

  if (!bin->src_elem) {
    NVGSTDS_ERR_MSG_V ("Could not create 'src_elem'");
    goto done;
  }

  bin->cap_filter =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER, "src_cap_filter");
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Could not create 'src_cap_filter'");
    goto done;
  }

  if (config->video_format) {
    caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
            config->video_format, "width", G_TYPE_INT, config->source_width,
            "height", G_TYPE_INT, config->source_height, "framerate",
            GST_TYPE_FRACTION, config->source_fps_n, config->source_fps_d,
            NULL);
  } else {
    caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, config->source_width, "height", G_TYPE_INT,
            config->source_height, "framerate", GST_TYPE_FRACTION,
            config->source_fps_n, config->source_fps_d, NULL);
  }

  if (config->type == NV_DS_SOURCE_CAMERA_CSI) {
    GstCapsFeatures *feature = NULL;
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);
  }
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, config->gpu_id);

  if (config->type == NV_DS_SOURCE_CAMERA_V4L2) {
    GstElement *nvvidconv2;
    GstCapsFeatures *feature = NULL;
    //Check based on igpu/dgpu instead of x86/aarch64
    GstElement *nvvidconv1 = NULL;
    if (!prop.integrated) {
      nvvidconv1 = gst_element_factory_make ("videoconvert", "nvvidconv1");
      if (!nvvidconv1) {
        NVGSTDS_ERR_MSG_V ("Failed to create 'nvvidconv1'");
        goto done;
      }
    }

    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);
    g_object_set (G_OBJECT (bin->cap_filter), "caps", caps, NULL);

    g_object_set (G_OBJECT (bin->cap_filter1), "caps", caps1, NULL);

    nvvidconv2 = gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "nvvidconv2");
    if (!nvvidconv2) {
      NVGSTDS_ERR_MSG_V ("Failed to create 'nvvidconv2'");
      goto done;
    }

    g_object_set (G_OBJECT (nvvidconv2), "gpu-id", config->gpu_id,
        "nvbuf-memory-type", config->nvbuf_memory_type, NULL);

    if (!prop.integrated) {
      gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem, bin->cap_filter1,
          nvvidconv1, nvvidconv2, bin->cap_filter, NULL);
    } else {
#if defined(__aarch64__) && !defined(AARCH64_IS_SBSA)
      if (config->nvvideoconvert_copy_hw) {
        g_object_set (G_OBJECT (nvvidconv2), "copy-hw", config->nvvideoconvert_copy_hw, NULL);
      }
#endif
      gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem, bin->cap_filter1,
          nvvidconv2, bin->cap_filter, NULL);
    }

    NVGSTDS_LINK_ELEMENT (bin->src_elem, bin->cap_filter1);

    if (!prop.integrated) {
      NVGSTDS_LINK_ELEMENT (bin->cap_filter1, nvvidconv1);

      NVGSTDS_LINK_ELEMENT (nvvidconv1, nvvidconv2);
    } else {
      NVGSTDS_LINK_ELEMENT (bin->cap_filter1, nvvidconv2);
    }

    NVGSTDS_LINK_ELEMENT (nvvidconv2, bin->cap_filter);

    NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter, "src");

  } else {

    g_object_set (G_OBJECT (bin->cap_filter), "caps", caps, NULL);

    gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem, bin->cap_filter, NULL);

    NVGSTDS_LINK_ELEMENT (bin->src_elem, bin->cap_filter);

    NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter, "src");
  }

  switch (config->type) {
    case NV_DS_SOURCE_CAMERA_CSI:
      if (!set_camera_csi_params (config, bin)) {
        NVGSTDS_ERR_MSG_V ("Could not set CSI camera properties");
        goto done;
      }
      break;
    case NV_DS_SOURCE_CAMERA_V4L2:
      if (!set_camera_v4l2_params (config, bin)) {
        NVGSTDS_ERR_MSG_V ("Could not set V4L2 camera properties");
        goto done;
      }
      break;
    default:
      NVGSTDS_ERR_MSG_V ("Unsupported source type");
      goto done;
  }

  ret = TRUE;

  GST_CAT_DEBUG (NVDS_APP, "Created camera source bin successfully");

done:
  if (caps)
    gst_caps_unref (caps);

  if (convertCaps)
    gst_caps_unref (convertCaps);

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static void
cb_newpad (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  if (!strncmp (name, "video", 5)) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;
    GstPad *sinkpad = gst_element_get_static_pad (bin->tee, "sink");
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {

      NVGSTDS_ERR_MSG_V ("Failed to link decodebin to pipeline");
    } else {
      NvDsSourceConfig *config =
          (NvDsSourceConfig *) g_object_get_data (G_OBJECT (bin->cap_filter),
          SRC_CONFIG_KEY);

      gst_structure_get_int (str, "width", &config->source_width);
      gst_structure_get_int (str, "height", &config->source_height);
      gst_structure_get_fraction (str, "framerate", &config->source_fps_n,
          &config->source_fps_d);

      GST_CAT_DEBUG (NVDS_APP, "Decodebin linked to pipeline");
    }
    gst_object_unref (sinkpad);
  } else if (g_str_has_prefix (name, "audio/x-raw")) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;

    /** skip linking if we did not prepare for audio */
    if (!bin->audio_converter) {
      return;
    }

    GstPad *sinkpad = gst_element_get_static_pad (bin->audio_converter, "sink");

    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK)
      NVGSTDS_ERR_MSG_V ("Failed to link decodebin to pipeline");
    gst_object_unref (sinkpad);
  }
}

static void
cb_newpad_audio (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  if (g_str_has_prefix (name, "audio/x-raw")) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;

    GstPad *sinkpad = gst_element_get_static_pad (bin->audio_converter, "sink");

    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK)
      NVGSTDS_ERR_MSG_V ("Failed to link decodebin to pipeline");
    gst_object_unref (sinkpad);
  } else if (!strncmp (name, "video", 5)) {
    /** connect video to fakesink and ignore the same */
    NvDsSrcBin *bin = (NvDsSrcBin *) data;
    bin->fakesink = gst_element_factory_make ("fakesink", "src_fakesink");
    if (!bin->fakesink) {
      NVGSTDS_ERR_MSG_V ("Could not create 'src_fakesink' for video path");
      return;
    }

    g_object_set (G_OBJECT (bin->fakesink), "sync", FALSE, "async", FALSE,
        NULL);
    gst_bin_add_many (GST_BIN (bin->bin), bin->fakesink, NULL);

    GstPad *sinkpad = gst_element_get_static_pad (bin->fakesink, "sink");

    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK)
      NVGSTDS_ERR_MSG_V ("Failed to link decodebin to pipeline");
    gst_object_unref (sinkpad);
  }
  gst_caps_unref (caps);
}

static void
cb_sourcesetup (GstElement * object, GstElement * arg0, gpointer data)
{
  NvDsSrcBin *bin = (NvDsSrcBin *) data;
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (arg0), "latency")) {
    g_object_set (G_OBJECT (arg0), "latency", bin->latency, NULL);
  }
  if (bin->udp_buffer_size &&
      g_object_class_find_property (G_OBJECT_GET_CLASS (arg0),
          "udp-buffer-size")) {
    g_object_set (G_OBJECT (arg0), "udp-buffer-size", bin->udp_buffer_size,
        NULL);
  }
}

/*
 * Function to seek the source stream to start.
 * It is required to play the stream in loop.
 */
static gboolean
seek_decode (gpointer data)
{
  NvDsSrcBin *bin = (NvDsSrcBin *) data;
  gboolean ret = TRUE;

  gst_element_set_state (bin->bin, GST_STATE_PAUSED);

  ret = gst_element_seek (bin->bin, 1.0, GST_FORMAT_TIME,
      (GstSeekFlags) (GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
      GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

  if (!ret)
    GST_WARNING ("Error in seeking pipeline");

  gst_element_set_state (bin->bin, GST_STATE_PLAYING);

  return FALSE;
}

/**
 * Probe function to drop certain events to support custom
 * logic of looping of each source stream.
 */
static GstPadProbeReturn
restart_stream_buf_prob (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstEvent *event = GST_EVENT (info->data);
  NvDsSrcBin *bin = (NvDsSrcBin *) u_data;

  if ((info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
    GST_BUFFER_PTS (GST_BUFFER (info->data)) += bin->prev_accumulated_base;
  }
  if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)) {
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
      g_timeout_add (1, seek_decode, bin);
    }

    if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
      GstSegment *segment = NULL;

      gst_event_parse_segment (event, (const GstSegment **) &segment);
      segment->base = bin->accumulated_base;
      bin->prev_accumulated_base = bin->accumulated_base;
      bin->accumulated_base += segment->stop;
    }
    switch (GST_EVENT_TYPE (event)) {
      case GST_EVENT_EOS:
        /* QOS events from downstream sink elements cause decoder to drop
         * frames after looping the file since the timestamps reset to 0.
         * We should drop the QOS events since we have custom logic for
         * looping individual sources. */
      case GST_EVENT_QOS:
      case GST_EVENT_SEGMENT:
      case GST_EVENT_FLUSH_START:
      case GST_EVENT_FLUSH_STOP:
        return GST_PAD_PROBE_DROP;
      default:
        break;
    }
  }
  return GST_PAD_PROBE_OK;
}


static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  NvDsSrcBin *bin = (NvDsSrcBin *) user_data;
  NvDsSourceConfig *config = bin->config;
  struct cudaDeviceProp prop = {0};
  cudaGetDeviceProperties (&prop, config->gpu_id);
  if (cudaGetDeviceProperties (&prop,
          config->gpu_id) != cudaSuccess) {
    NVGSTDS_ERR_MSG_V ("Failed to get properties for GPU %d", config->gpu_id);
  }
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if ((g_strrstr (name, "h264parse") == name) ||
      (g_strrstr (name, "h265parse") == name)) {
    g_object_set (object, "config-interval", -1, NULL);
  }
  if (g_strrstr (name, "fakesink") == name) {
    g_object_set (object, "enable-last-sample", FALSE, NULL);
  }
  if (g_strrstr (name, "nvcuvid") == name) {
    g_object_set (object, "gpu-id", config->gpu_id, NULL);

    g_object_set (G_OBJECT (object), "cuda-memory-type",
        config->cuda_memory_type, NULL);

    g_object_set (object, "source-id", config->camera_id, NULL);
    g_object_set (object, "num-decode-surfaces", config->num_decode_surfaces,
        NULL);
    if (config->Intra_decode)
      g_object_set (object, "Intra-decode", config->Intra_decode, NULL);
  }
  if (g_strstr_len (name, -1, "omx") == name) {
    if (config->Intra_decode)
      g_object_set (object, "skip-frames", 2, NULL);
    g_object_set (object, "disable-dvfs", TRUE, NULL);
  }
  if (g_strstr_len (name, -1, "nvv4l2decoder") == name) {
    if (config->low_latency_mode)
      g_object_set (object, "low-latency-mode", TRUE, NULL);
    if (config->Intra_decode)
      g_object_set (object, "skip-frames", 2, NULL);
#ifdef __aarch64__
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (object),
            "enable-max-performance")) {
      g_object_set (object, "enable-max-performance", TRUE, NULL);
    }
#endif

    if (g_object_class_find_property (G_OBJECT_GET_CLASS (object), "gpu-id")) {
      g_object_set (object, "gpu-id", config->gpu_id, NULL);
    }
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (object),
            "cudadec-memtype")) {
      g_object_set (G_OBJECT (object), "cudadec-memtype",
          config->cuda_memory_type, NULL);
    }
    g_object_set (object, "drop-frame-interval", config->drop_frame_interval,
        NULL);
    /* extract-sei-type5-data is a valid parameter for nvv4l2decoder
       on x86 and ARM_SBSA */
    if (!prop.integrated) {
      g_object_set (object, "extract-sei-type5-data", config->extract_sei_type5_data,
         NULL);
    }
    g_object_set (object, "num-extra-surfaces", config->num_extra_surfaces,
        NULL);

    /* Seek only if file is the source. */
    if (config->loop && g_strstr_len (config->uri, -1, "file:/") == config->uri) {
      NVGSTDS_ELEM_ADD_PROBE (bin->src_buffer_probe, GST_ELEMENT (object),
          "sink", restart_stream_buf_prob,
          (GstPadProbeType) (GST_PAD_PROBE_TYPE_EVENT_BOTH |
              GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER), bin);
    }
  }
done:
  return;
}

static void
cb_newpad2 (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  if (!strncmp (name, "video", 5)) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;
    GstPad *sinkpad = gst_element_get_static_pad (bin->cap_filter, "sink");
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {

      NVGSTDS_ERR_MSG_V ("Failed to link decodebin to pipeline");
    } else {
      NvDsSourceConfig *config =
          (NvDsSourceConfig *) g_object_get_data (G_OBJECT (bin->cap_filter),
          SRC_CONFIG_KEY);

      gst_structure_get_int (str, "width", &config->source_width);
      gst_structure_get_int (str, "height", &config->source_height);
      gst_structure_get_fraction (str, "framerate", &config->source_fps_n,
          &config->source_fps_d);

      GST_CAT_DEBUG (NVDS_APP, "Decodebin linked to pipeline");
    }
    gst_object_unref (sinkpad);
  }
  gst_caps_unref (caps);
}


static void
cb_newpad3 (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  if (g_strrstr (name, "x-rtp")) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;
    GstPad *sinkpad = gst_element_get_static_pad (bin->depay, "sink");
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {

      NVGSTDS_ERR_MSG_V ("Failed to link depay loader to rtsp src");
    }
    gst_object_unref (sinkpad);
  }
  gst_caps_unref (caps);
}

/* Returning FALSE from this callback will make rtspsrc ignore the stream.
 * Ignore audio and add the proper depay element based on codec. */
static gboolean
cb_rtspsrc_select_stream (GstElement * rtspsrc, guint num, GstCaps * caps,
    gpointer user_data)
{
  GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *media = gst_structure_get_string (str, "media");
  const gchar *encoding_name = gst_structure_get_string (str, "encoding-name");
  gchar elem_name[50];
  NvDsSrcBin *bin = (NvDsSrcBin *) user_data;
  gboolean ret = FALSE;

  gboolean is_video = (!g_strcmp0 (media, "video"));

  if (!is_video)
    return FALSE;

  /* Create and add depay element only if it is not created yet. */
  if (!bin->depay) {
    g_snprintf (elem_name, sizeof (elem_name), "depay_elem%d", bin->bin_id);

    /* Add the proper depay element based on codec. */
    if (!g_strcmp0 (encoding_name, "H264")) {
      bin->depay = gst_element_factory_make ("rtph264depay", elem_name);
      g_snprintf (elem_name, sizeof (elem_name), "h264parse_elem%d",
          bin->bin_id);
      bin->parser = gst_element_factory_make ("h264parse", elem_name);
    } else if (!g_strcmp0 (encoding_name, "H265")) {
      bin->depay = gst_element_factory_make ("rtph265depay", elem_name);
      g_snprintf (elem_name, sizeof (elem_name), "h265parse_elem%d",
          bin->bin_id);
      bin->parser = gst_element_factory_make ("h265parse", elem_name);
    } else {
      NVGSTDS_WARN_MSG_V ("%s not supported", encoding_name);
      return FALSE;
    }

    if (!bin->depay) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      return FALSE;
    }

    gst_bin_add_many (GST_BIN (bin->bin), bin->depay, bin->parser, NULL);

    NVGSTDS_LINK_ELEMENT (bin->depay, bin->parser);
    NVGSTDS_LINK_ELEMENT (bin->parser, bin->tee_rtsp_pre_decode);

    if (!gst_element_sync_state_with_parent (bin->depay)) {
      NVGSTDS_ERR_MSG_V ("'%s' failed to sync state with parent", elem_name);
      return FALSE;
    }
    gst_element_sync_state_with_parent (bin->parser);
  }

  ret = TRUE;
done:
  return ret;
}

void
destroy_smart_record_bin (gpointer data)
{
  unsigned int i = 0;
  NvDsSrcBin *src_bin;
  NvDsSrcParentBin *pbin = (NvDsSrcParentBin *) data;

  g_return_if_fail (pbin);

  for (i = 0; i < pbin->num_bins; i++) {

    src_bin = &pbin->sub_bins[i];
    if (src_bin && src_bin->recordCtx)
      NvDsSRDestroy ((NvDsSRContext *) src_bin->recordCtx);
  }
  pbin->num_bins = 0;
}

static gpointer
smart_record_callback (NvDsSRRecordingInfo * info, gpointer userData)
{
  static GMutex mutex;
  FILE *logfile = NULL;
  g_return_val_if_fail (info, NULL);

  g_mutex_lock (&mutex);
  logfile = fopen ("smart_record.log", "a");
  if (logfile) {
    fprintf (logfile, "%d:%d:%d:%ldms:%s:%s\n",
        info->sessionId, info->width, info->height, info->duration,
        info->dirpath, info->filename);
    fclose (logfile);
  } else {
    g_print ("Error in opeing smart record log file\n");
  }
  g_mutex_unlock (&mutex);

  return NULL;
}

/**
 * Function called at regular interval to start and stop video recording.
 * This is dummy implementation to show the usages of smart record APIs.
 * startTime and Duration can be adjusted as per usecase.
*/
static gboolean
smart_record_event_generator (gpointer data)
{
  NvDsSRSessionId sessId = 0;
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  guint startTime = 7;
  guint duration = 8;

  if (src_bin->config->smart_rec_duration >= 0)
    duration = src_bin->config->smart_rec_duration;

  if (src_bin->config->smart_rec_start_time >= 0)
    startTime = src_bin->config->smart_rec_start_time;

  if (src_bin->recordCtx && !src_bin->reconfiguring) {
    NvDsSRContext *ctx = (NvDsSRContext *) src_bin->recordCtx;
    if (ctx->recordOn) {
      NvDsSRStop (ctx, 0);
    } else {
      NvDsSRStart (ctx, &sessId, startTime, duration, NULL);
    }
  }
  return TRUE;
}

static void
check_rtsp_reconnection_attempts (NvDsSrcBin * src_bin)
{
  gboolean remove_probe = TRUE;
  guint i = 0;
  for (i = 0; i < src_bin->parent_bin->num_bins; i++) {
    if (src_bin->parent_bin->sub_bins[i].config->type != NV_DS_SOURCE_RTSP)
      continue;
    if (src_bin->parent_bin->sub_bins[i].have_eos &&
        (src_bin->parent_bin->sub_bins[i].rtsp_reconnect_interval_sec == 0 ||
         src_bin->parent_bin->sub_bins[i].rtsp_reconnect_attempts == 0)) {
      remove_probe = FALSE;
      break;
    }
    if (src_bin->parent_bin->sub_bins[i].num_rtsp_reconnects <=
        src_bin->parent_bin->sub_bins[i].rtsp_reconnect_attempts) {
      if (src_bin->parent_bin->sub_bins[i].rtsp_reconnect_interval_sec ||
          !src_bin->parent_bin->sub_bins[i].have_eos) {
        remove_probe = FALSE;
        break;
      }
    }
  }

  if (remove_probe) {
    GstElement *pipeline =
        GST_ELEMENT_PARENT (GST_ELEMENT_PARENT (src_bin->bin));
    NVGSTDS_ELEM_REMOVE_PROBE (src_bin->parent_bin->
        nvstreammux_eosmonitor_probe, src_bin->parent_bin->streammux, "src");
    GST_ELEMENT_ERROR (pipeline, STREAM, FAILED,
        ("Reconnection attempts exceeded for all sources or EOS received."
            " Stopping pipeline"), (NULL));
  }
}

/**
 * Function called at regular interval to check if NV_DS_SOURCE_RTSP type
 * source in the pipeline is down / disconnected. This function try to
 * reconnect the source by resetting that source pipeline.
 */
static gboolean
watch_source_status (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  struct timeval current_time;
  gettimeofday (&current_time, NULL);
  static struct timeval last_reset_time_global = { 0, 0 };
  gdouble time_diff_msec_since_last_reset =
      1000.0 * (current_time.tv_sec - last_reset_time_global.tv_sec) +
      (current_time.tv_usec - last_reset_time_global.tv_usec) / 1000.0;

  if (src_bin->reconfiguring) {
    guint time_since_last_reconnect_sec =
        current_time.tv_sec - src_bin->last_reconnect_time.tv_sec;
    if (time_since_last_reconnect_sec >= SOURCE_RESET_INTERVAL_SEC) {
      if (time_diff_msec_since_last_reset > 3000) {
        if (src_bin->rtsp_reconnect_attempts == -1 ||
            ++src_bin->num_rtsp_reconnects <=
            src_bin->rtsp_reconnect_attempts) {
          last_reset_time_global = current_time;
          // source is still not up, reconfigure it again.
          reset_source_pipeline (src_bin);
        } else {
          GST_ELEMENT_WARNING (src_bin->bin, STREAM, FAILED,
              ("Number of RTSP reconnect attempts exceeded, stopping source: %d",
                  src_bin->source_id), (NULL));

          check_rtsp_reconnection_attempts (src_bin);

          GstElement *send_event_element = NULL;
          if (src_bin->dewarper_bin.bin != NULL) {
            send_event_element = src_bin->dewarper_bin.bin;
          } else {
            send_event_element = src_bin->cap_filter1;
          }
          gst_element_send_event (GST_ELEMENT (send_event_element),
              gst_event_new_flush_start ());
          gst_element_send_event (GST_ELEMENT (send_event_element),
              gst_event_new_flush_stop (TRUE));
          if (!gst_element_send_event (GST_ELEMENT (send_event_element),
                  gst_event_new_eos ())) {
            GST_ERROR_OBJECT (send_event_element,
                "Interrupted, Reconnection event not sent");
          }
          if (gst_element_set_state (src_bin->bin,
                  GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
            GST_ERROR_OBJECT (src_bin->bin, "Can't set source bin to NULL");
          }

          return FALSE;
        }
      }
    }
  } else {
    gint time_since_last_buf_sec = 0;
    g_mutex_lock (&src_bin->bin_lock);
    if (src_bin->last_buffer_time.tv_sec != 0) {
      time_since_last_buf_sec =
          current_time.tv_sec - src_bin->last_buffer_time.tv_sec;
    }
    g_mutex_unlock (&src_bin->bin_lock);

    // Reset source bin if no buffers are received in the last
    // `rtsp_reconnect_interval_sec` seconds.
    if (src_bin->rtsp_reconnect_interval_sec > 0 &&
        time_since_last_buf_sec >= src_bin->rtsp_reconnect_interval_sec) {
      if (time_diff_msec_since_last_reset > 3000) {
        if (src_bin->rtsp_reconnect_attempts == -1 ||
            ++src_bin->num_rtsp_reconnects <=
            src_bin->rtsp_reconnect_attempts) {
          last_reset_time_global = current_time;

          NVGSTDS_WARN_MSG_V
              ("No data from source %d since last %u sec. Trying reconnection",
              src_bin->bin_id, time_since_last_buf_sec);
          reset_source_pipeline (src_bin);
        } else {
          GST_ELEMENT_WARNING (src_bin->bin, STREAM, FAILED,
              ("Number of RTSP reconnect attempts exceeded, stopping source: %d",
                  src_bin->source_id), (NULL));

          check_rtsp_reconnection_attempts (src_bin);

          GstElement *send_event_element = NULL;
          if (src_bin->dewarper_bin.bin != NULL) {
            send_event_element = src_bin->dewarper_bin.bin;
          } else {
            send_event_element = src_bin->cap_filter1;
          }
          gst_element_send_event (GST_ELEMENT (send_event_element),
              gst_event_new_flush_start ());
          gst_element_send_event (GST_ELEMENT (send_event_element),
              gst_event_new_flush_stop (TRUE));
          if (!gst_element_send_event (GST_ELEMENT (send_event_element),
                  gst_event_new_eos ())) {
            GST_ERROR_OBJECT (send_event_element,
                "Interrupted, Reconnection event not sent");
          }
          if (gst_element_set_state (src_bin->bin,
                  GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
            GST_ERROR_OBJECT (src_bin->bin, "Can't set source bin to NULL");
          }

          return FALSE;
        }
      }
    }
  }
  return TRUE;
}

/**
 * Function called at regular interval when source bin is
 * changing state async. This function watches the state of
 * the source bin and sets it to PLAYING if the state of source
 * bin stops at PAUSED when changing state ASYNC.
 */
static gboolean
watch_source_async_state_change (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  GstState state = GST_STATE_NULL, pending = GST_STATE_NULL;
  GstStateChangeReturn ret;

  ret = gst_element_get_state (src_bin->bin, &state, &pending, 0);

  GST_CAT_DEBUG (NVDS_APP, "Bin %d %p: state:%s pending:%s ret:%s",
      src_bin->bin_id, src_bin, gst_element_state_get_name (state),
      gst_element_state_get_name (pending),
      gst_element_state_change_return_get_name (ret));

  // Bin is still changing state ASYNC. Wait for some more time.
  if (ret == GST_STATE_CHANGE_ASYNC)
    return TRUE;

  // Bin state change failed / failed to get state
  if (ret == GST_STATE_CHANGE_FAILURE) {
    src_bin->async_state_watch_running = FALSE;
    return FALSE;
  }
  // Bin successfully changed state to PLAYING. Stop watching state
  if (state == GST_STATE_PLAYING) {
    src_bin->reconfiguring = FALSE;
    src_bin->async_state_watch_running = FALSE;
    src_bin->num_rtsp_reconnects = 0;
    return FALSE;
  }
  // Bin has stopped ASYNC state change but has not gone into
  // PLAYING. Expliclity set state to PLAYING and keep watching
  // state
  gst_element_set_state (src_bin->bin, GST_STATE_PLAYING);

  return TRUE;
}

/**
 * Probe function to monitor data output from rtspsrc.
 */
static GstPadProbeReturn
rtspsrc_monitor_probe_func (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  NvDsSrcBin *bin = (NvDsSrcBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    g_mutex_lock (&bin->bin_lock);
    gettimeofday (&bin->last_buffer_time, NULL);
    bin->have_eos = FALSE;
    g_mutex_unlock (&bin->bin_lock);
  }
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    if (GST_EVENT_TYPE (info->data) == GST_EVENT_EOS) {
      bin->have_eos = TRUE;
      check_rtsp_reconnection_attempts (bin);
    }
  }
  return GST_PAD_PROBE_OK;
}

/**
 * Probe function to drop EOS events from nvstreammux when RTSP sources
 * are being used so that app does not quit from EOS in case of RTSP
 * connection errors and tries to reconnect.
 */
static GstPadProbeReturn
nvstreammux_eosmonitor_probe_func (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = (GstEvent *) info->data;
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS)
      return GST_PAD_PROBE_DROP;
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
create_rtsp_src_bin (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  NvDsSRContext *ctx = NULL;
  gboolean ret = FALSE;
  gchar elem_name[50];
  bin->config = config;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;

  bin->latency = config->latency;
  bin->udp_buffer_size = config->udp_buffer_size;
  bin->rtsp_reconnect_interval_sec = config->rtsp_reconnect_interval_sec;
  bin->rtsp_reconnect_attempts = config->rtsp_reconnect_attempts;
  bin->num_rtsp_reconnects = 0;

  g_snprintf (elem_name, sizeof (elem_name), "src_elem%d", bin->bin_id);
  bin->src_elem = gst_element_factory_make ("rtspsrc", elem_name);
  if (!bin->src_elem) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_signal_connect (G_OBJECT (bin->src_elem), "select-stream",
      G_CALLBACK (cb_rtspsrc_select_stream), bin);

  if (config->udp_buffer_size) {
    g_object_set (G_OBJECT (bin->src_elem), "udp-buffer-size",
        config->udp_buffer_size, NULL);
  }

  g_object_set (G_OBJECT (bin->src_elem), "location", config->uri, NULL);
  g_object_set (G_OBJECT (bin->src_elem), "latency", config->latency, NULL);
  g_object_set (G_OBJECT (bin->src_elem), "drop-on-latency", TRUE, NULL);
  configure_source_for_ntp_sync (bin->src_elem);

  // 0x4 for TCP and 0x7 for All (UDP/UDP-MCAST/TCP)
  if ((config->select_rtp_protocol == GST_RTSP_LOWER_TRANS_TCP)
      || (config->select_rtp_protocol == (GST_RTSP_LOWER_TRANS_UDP |
              GST_RTSP_LOWER_TRANS_UDP_MCAST | GST_RTSP_LOWER_TRANS_TCP))) {
    g_object_set (G_OBJECT (bin->src_elem), "protocols",
        config->select_rtp_protocol, NULL);
    GST_DEBUG_OBJECT (bin->src_elem,
        "RTP Protocol=0x%x (0x4=TCP and 0x7=UDP,TCP,UDPMCAST)----\n",
        config->select_rtp_protocol);
  }
  g_signal_connect (G_OBJECT (bin->src_elem), "pad-added",
      G_CALLBACK (cb_newpad3), bin);

  g_snprintf (elem_name, sizeof (elem_name), "tee_rtsp_elem%d", bin->bin_id);
  bin->tee_rtsp_pre_decode = gst_element_factory_make ("tee", elem_name);
  if (!bin->tee_rtsp_pre_decode) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_snprintf (elem_name, sizeof (elem_name), "tee_rtsp_post_decode_elem%d",
      bin->bin_id);
  bin->tee_rtsp_post_decode = gst_element_factory_make ("tee", elem_name);
  if (!bin->tee_rtsp_post_decode) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  if (config->smart_record) {
    NvDsSRInitParams params = { 0 };
    params.containerType = (NvDsSRContainerType) config->smart_rec_container;
    if (config->file_prefix)
      params.fileNamePrefix =
          g_strdup_printf ("%s_%d", config->file_prefix, config->camera_id);
    params.dirpath = config->dir_path;
    params.cacheSize = config->smart_rec_cache_size;
    params.defaultDuration = config->smart_rec_def_duration;
    params.callback = smart_record_callback;
    if (NvDsSRCreate (&ctx, &params) != NVDSSR_STATUS_OK) {
      NVGSTDS_ERR_MSG_V ("Failed to create smart record bin");
      g_free (params.fileNamePrefix);
      goto done;
    }
    g_free (params.fileNamePrefix);
    gst_bin_add (GST_BIN (bin->bin), ctx->recordbin);
    bin->recordCtx = (gpointer) ctx;
  }

  g_snprintf (elem_name, sizeof (elem_name), "dec_que%d", bin->bin_id);
  bin->dec_que = gst_element_factory_make ("queue", elem_name);
  if (!bin->dec_que) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  if (bin->rtsp_reconnect_interval_sec > 0) {
    NVGSTDS_ELEM_ADD_PROBE (bin->rtspsrc_monitor_probe, bin->dec_que,
        "sink", rtspsrc_monitor_probe_func, GST_PAD_PROBE_TYPE_BUFFER, bin);
    install_mux_eosmonitor_probe = TRUE;
  } else {
    NVGSTDS_ELEM_ADD_PROBE (bin->rtspsrc_monitor_probe, bin->dec_que,
        "sink", rtspsrc_monitor_probe_func,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, bin);
  }

  g_snprintf (elem_name, sizeof (elem_name), "decodebin_elem%d", bin->bin_id);
  bin->decodebin = gst_element_factory_make ("decodebin", elem_name);
  if (!bin->decodebin) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_signal_connect (G_OBJECT (bin->decodebin), "pad-added",
      G_CALLBACK (cb_newpad2), bin);
  g_signal_connect (G_OBJECT (bin->decodebin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);


  g_snprintf (elem_name, sizeof (elem_name), "src_que%d", bin->bin_id);
  bin->cap_filter = gst_element_factory_make (NVDS_ELEM_QUEUE, elem_name);
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_mutex_init (&bin->bin_lock);
  if (config->dewarper_config.enable) {
    if (!create_dewarper_bin (&config->dewarper_config, &bin->dewarper_bin)) {
      g_print ("Failed to create dewarper bin \n");
      goto done;
    }
    gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem,
        bin->tee_rtsp_pre_decode,
        bin->dec_que,
        bin->decodebin,
        bin->cap_filter,
        bin->tee_rtsp_post_decode, bin->dewarper_bin.bin, NULL);
  } else {
    g_snprintf (elem_name, sizeof (elem_name), "nvvidconv_elem%d", bin->bin_id);
    bin->nvvidconv = gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, elem_name);
    if (!bin->nvvidconv) {
      NVGSTDS_ERR_MSG_V ("Could not create element 'nvvidconv_elem'");
      goto done;
    }
    g_object_set (G_OBJECT (bin->nvvidconv), "gpu-id", config->gpu_id,
        "nvbuf-memory-type", config->nvbuf_memory_type, NULL);
    if (config->video_format) {
      caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, config->video_format, NULL);
    } else {
      caps = gst_caps_new_empty_simple ("video/x-raw");
    }
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);

    bin->cap_filter1 =
        gst_element_factory_make (NVDS_ELEM_CAPS_FILTER,
        "src_cap_filter_nvvidconv");
    if (!bin->cap_filter1) {
      NVGSTDS_ERR_MSG_V ("Could not create 'queue'");
      goto done;
    }

    g_object_set (G_OBJECT (bin->cap_filter1), "caps", caps, NULL);
    gst_caps_unref (caps);
    gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem,
        bin->tee_rtsp_pre_decode,
        bin->dec_que,
        bin->decodebin,
        bin->cap_filter,
        bin->tee_rtsp_post_decode, bin->nvvidconv, bin->cap_filter1, NULL);
  }

  link_element_to_tee_src_pad (bin->tee_rtsp_pre_decode, bin->dec_que);
  NVGSTDS_LINK_ELEMENT (bin->dec_que, bin->decodebin);


  if (ctx)
    link_element_to_tee_src_pad (bin->tee_rtsp_pre_decode, ctx->recordbin);

  NVGSTDS_LINK_ELEMENT (bin->cap_filter, bin->tee_rtsp_post_decode);
  if (config->dewarper_config.enable) {
    link_element_to_tee_src_pad (bin->tee_rtsp_post_decode,
        bin->dewarper_bin.bin);
    NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->dewarper_bin.bin, "src");
  } else {
    link_element_to_tee_src_pad (bin->tee_rtsp_post_decode, bin->nvvidconv);
    NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->cap_filter1);
    NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter1, "src");
  }

  ret = TRUE;

  g_timeout_add (1000, watch_source_status, bin);

  // Enable local start / stop events in addition to the one
  // received from the server.
  if (config->smart_record == 2) {
    if (bin->config->smart_rec_interval)
      g_timeout_add (bin->config->smart_rec_interval * 1000,
          smart_record_event_generator, bin);
    else
      g_timeout_add (10000, smart_record_event_generator, bin);
  }

  GST_CAT_DEBUG (NVDS_APP,
      "Decode bin created. Waiting for a new pad from decodebin to link");

done:

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static gboolean
create_audiodecode_src_bin (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  gboolean ret = FALSE;
  guint const MAX_CAPS_LEN = 256;
  gchar caps_audio_resampler[MAX_CAPS_LEN];
  GstCaps *caps = NULL;
  bin->config = config;

  config->live_source = FALSE;

  if (config->type == NV_DS_SOURCE_AUDIO_WAV) {
    bin->src_elem =
        gst_element_factory_make (NVDS_ELEM_SRC_MULTIFILE, "src_elem");
    if (!bin->src_elem) {
      NVGSTDS_ERR_MSG_V ("Could not create element 'src_elem'");
      goto done;
    }

    g_object_set (G_OBJECT (bin->src_elem), "location", config->uri, NULL);
    g_object_set (G_OBJECT (bin->src_elem), "loop", config->loop, NULL);

    bin->decodebin =
        gst_element_factory_make (NVDS_ELEM_WAVPARSE, "decodebin_elem");
    if (!bin->decodebin) {
      NVGSTDS_ERR_MSG_V ("Could not create element 'decodebin_elem'");
      goto done;
    }
    g_object_set (G_OBJECT (bin->decodebin), "ignore-length", config->loop,
        NULL);
  } else if (config->type == NV_DS_SOURCE_ALSA_SRC) {
    bin->src_elem = gst_element_factory_make (NVDS_ELEM_SRC_ALSA, "src_elem");
    if (!bin->src_elem) {
      NVGSTDS_ERR_MSG_V ("Could not create element 'src_elem'");
      goto done;
    }
    if (config->alsa_device) {
      g_object_set (G_OBJECT (bin->src_elem), "device", config->alsa_device,
          NULL);
    }
  } else {
    NVGSTDS_ERR_MSG_V ("Source Type (%d) not supported\n", config->type);
    goto done;
  }

  bin->audio_converter =
      gst_element_factory_make ("audioconvert", "audio-convert");
  if (!bin->audio_converter) {
    NVGSTDS_ERR_MSG_V ("Could not create 'audioconvert'");
    goto done;
  }

  bin->audio_resample =
      gst_element_factory_make ("audioresample", "audio-resample");
  if (!bin->audio_resample) {
    NVGSTDS_ERR_MSG_V ("Could not create 'audioresample'");
    goto done;
  }

  bin->cap_filter =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER,
      "src_cap_filter_audioresample");
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Could not create src_cap_filter_audioresample");
    goto done;
  }

  if (snprintf (caps_audio_resampler, MAX_CAPS_LEN, "audio/x-raw, rate=%d",
          config->input_audio_rate)
      <= 0) {
    NVGSTDS_ERR_MSG_V ("Could not create caps to force rate=%d",
        config->input_audio_rate);
    goto done;
  }
  caps = gst_caps_from_string (caps_audio_resampler);
  g_object_set (G_OBJECT (bin->cap_filter), "caps", caps, NULL);
  gst_caps_unref (caps);


  if (config->type == NV_DS_SOURCE_AUDIO_WAV) {
    gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem, bin->decodebin,
        bin->audio_converter, bin->audio_resample, bin->cap_filter, NULL);

    gst_element_link_many (bin->src_elem, bin->decodebin, bin->audio_converter,
        bin->audio_resample, bin->cap_filter, NULL);
  } else if (config->type == NV_DS_SOURCE_ALSA_SRC) {
    gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem,
        bin->audio_converter, bin->audio_resample, bin->cap_filter, NULL);

    gst_element_link_many (bin->src_elem, bin->audio_converter,
        bin->audio_resample, bin->cap_filter, NULL);
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter, "src");

  ret = TRUE;

  GST_CAT_DEBUG (NVDS_APP,
      "Decode bin created. Waiting for a new pad from decodebin to link");

done:

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static gboolean
create_uridecode_src_bin_audio (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  gboolean ret = FALSE;
  guint const MAX_CAPS_LEN = 256;
  gchar caps_audio_resampler[MAX_CAPS_LEN];
  GstCaps *caps = NULL;
  bin->config = config;

  bin->src_elem = gst_element_factory_make (NVDS_ELEM_SRC_URI, "src_elem");
  if (!bin->src_elem) {
    NVGSTDS_ERR_MSG_V ("Could not create element 'src_elem'");
    goto done;
  }
  bin->latency = config->latency;
  bin->udp_buffer_size = config->udp_buffer_size;

  if (g_strrstr (config->uri, "file:/")) {
    config->live_source = FALSE;
  }
  if (g_strrstr (config->uri, "rtsp://") == config->uri) {
    configure_source_for_ntp_sync (bin->src_elem);
  }

  g_object_set (G_OBJECT (bin->src_elem), "uri", config->uri, NULL);
  g_signal_connect (G_OBJECT (bin->src_elem), "pad-added",
      G_CALLBACK (cb_newpad_audio), bin);


  bin->audio_converter =
      gst_element_factory_make (NVDS_ELEM_AUDIO_CONV, "audioconv_elem");
  if (!bin->audio_converter) {
    NVGSTDS_ERR_MSG_V ("Could not create element audio_converter");
    goto done;
  }

  bin->audio_resample =
      gst_element_factory_make (NVDS_ELEM_AUDIO_RESAMPLER,
      "audioresampler_elem");
  if (!bin->audio_resample) {
    NVGSTDS_ERR_MSG_V ("Could not create element audio_resample");
    goto done;
  }

  bin->cap_filter =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER,
      "src_cap_filter_audioresample");
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Could not create src_cap_filter_audioresample");
    goto done;
  }

  if (snprintf (caps_audio_resampler, MAX_CAPS_LEN, "audio/x-raw, rate=%d",
          config->input_audio_rate)
      <= 0) {
    NVGSTDS_ERR_MSG_V ("Could not create caps to force rate=%d",
        config->input_audio_rate);
    goto done;
  }
  caps = gst_caps_from_string (caps_audio_resampler);
  g_object_set (G_OBJECT (bin->cap_filter), "caps", caps, NULL);
  gst_caps_unref (caps);


  gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem,
      bin->audio_converter, bin->audio_resample, bin->cap_filter, NULL);

  NVGSTDS_LINK_ELEMENT (bin->audio_converter, bin->audio_resample);
  NVGSTDS_LINK_ELEMENT (bin->audio_resample, bin->cap_filter);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter, "src");

  ret = TRUE;

  GST_CAT_DEBUG (NVDS_APP,
      "Decode bin created. Waiting for a new pad from decodebin to link");

done:

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static gboolean
create_uridecode_src_bin (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  gboolean ret = FALSE;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  bin->config = config;

  bin->src_elem = gst_element_factory_make (NVDS_ELEM_SRC_URI, "src_elem");
  if (!bin->src_elem) {
    NVGSTDS_ERR_MSG_V ("Could not create element 'src_elem'");
    goto done;
  }

  if (config->dewarper_config.enable) {
    if (!create_dewarper_bin (&config->dewarper_config, &bin->dewarper_bin)) {
      g_print ("Creating Dewarper bin failed \n");
      goto done;
    }
  }
  bin->latency = config->latency;
  bin->udp_buffer_size = config->udp_buffer_size;

  if (g_strrstr (config->uri, "file:/")) {
    config->live_source = FALSE;
  }
  if (g_strrstr (config->uri, "rtsp://") == config->uri) {
    configure_source_for_ntp_sync (bin->src_elem);
  }

  g_object_set (G_OBJECT (bin->src_elem), "uri", config->uri, NULL);
  g_signal_connect (G_OBJECT (bin->src_elem), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (bin->src_elem), "child-added",
      G_CALLBACK (decodebin_child_added), bin);
  g_signal_connect (G_OBJECT (bin->src_elem), "source-setup",
      G_CALLBACK (cb_sourcesetup), bin);
  bin->cap_filter = gst_element_factory_make (NVDS_ELEM_QUEUE, "queue");
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Could not create 'queue'");
    goto done;
  }

  bin->nvvidconv =
      gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "nvvidconv_elem");
  if (!bin->nvvidconv) {
    NVGSTDS_ERR_MSG_V ("Could not create element 'nvvidconv_elem'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->nvvidconv), "gpu-id", config->gpu_id,
        "nvbuf-memory-type", config->nvbuf_memory_type, NULL);
  if (config->video_format) {
    caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, config->video_format, NULL);
  } else {
    caps = gst_caps_new_empty_simple ("video/x-raw");
  }
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);

  bin->cap_filter1 =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER,
      "src_cap_filter_nvvidconv");
  if (!bin->cap_filter1) {
    NVGSTDS_ERR_MSG_V ("Could not create 'queue'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->cap_filter1), "caps", caps, NULL);
  gst_caps_unref (caps);

  g_object_set_data (G_OBJECT (bin->cap_filter), SRC_CONFIG_KEY, config);

  gst_bin_add_many (GST_BIN (bin->bin), bin->src_elem, bin->cap_filter,
      bin->nvvidconv, bin->cap_filter1, NULL);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter1, "src");

  bin->fakesink = gst_element_factory_make ("fakesink", "src_fakesink");
  if (!bin->fakesink) {
    NVGSTDS_ERR_MSG_V ("Could not create 'src_fakesink'");
    goto done;
  }

  bin->fakesink_queue = gst_element_factory_make ("queue", "fakequeue");
  if (!bin->fakesink_queue) {
    NVGSTDS_ERR_MSG_V ("Could not create 'fakequeue'");
    goto done;
  }

  bin->tee = gst_element_factory_make ("tee", NULL);
  if (!bin->tee) {
    NVGSTDS_ERR_MSG_V ("Could not create 'tee'");
    goto done;
  }
  gst_bin_add_many (GST_BIN (bin->bin), bin->fakesink, bin->tee,
      bin->fakesink_queue, NULL);


  NVGSTDS_LINK_ELEMENT (bin->fakesink_queue, bin->fakesink);

  if (config->dewarper_config.enable) {
    gst_bin_add_many (GST_BIN (bin->bin), bin->dewarper_bin.bin, NULL);
    NVGSTDS_LINK_ELEMENT (bin->tee, bin->dewarper_bin.bin);
    NVGSTDS_LINK_ELEMENT (bin->dewarper_bin.bin, bin->cap_filter);
  } else {
    link_element_to_tee_src_pad (bin->tee, bin->cap_filter);
  }
  NVGSTDS_LINK_ELEMENT (bin->cap_filter, bin->nvvidconv);
  NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->cap_filter1);
  link_element_to_tee_src_pad (bin->tee, bin->fakesink_queue);

  g_object_set (G_OBJECT (bin->fakesink), "sync", FALSE, "async", FALSE, NULL);
  g_object_set (G_OBJECT (bin->fakesink), "enable-last-sample", FALSE, NULL);

  ret = TRUE;

  GST_CAT_DEBUG (NVDS_APP,
      "Decode bin created. Waiting for a new pad from decodebin to link");

done:

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
create_audio_source_bin (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  static guint bin_cnt = 0;
  gchar bin_name[64];

  g_snprintf (bin_name, 64, "src_bin_%d", bin_cnt++);
  bin->bin = gst_bin_new (bin_name);
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'src_bin'");
    return FALSE;
  }

  if (!create_audiodecode_src_bin (config, bin)) {
    return FALSE;
  }
  bin->live_source = config->live_source;

  GST_CAT_DEBUG (NVDS_APP, "Source bin created");

  return TRUE;
}

gboolean
create_source_bin (NvDsSourceConfig * config, NvDsSrcBin * bin)
{
  static guint bin_cnt = 0;
  gchar bin_name[64];
  g_snprintf (bin_name, 64, "src_bin_%d", bin_cnt++);
  bin->bin = gst_bin_new (bin_name);
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'src_bin'");
    return FALSE;
  }

  switch (config->type) {
    case NV_DS_SOURCE_CAMERA_V4L2:
      if (!create_camera_source_bin (config, bin)) {
        return FALSE;
      }
      break;
    case NV_DS_SOURCE_URI:
      if (!create_uridecode_src_bin (config, bin)) {
        return FALSE;
      }
      bin->live_source = config->live_source;
      break;
    case NV_DS_SOURCE_RTSP:
      if (!create_rtsp_src_bin (config, bin)) {
        return FALSE;
      }
      break;
    default:
      NVGSTDS_ERR_MSG_V ("Source type not yet implemented!\n");
      return FALSE;
  }

  GST_CAT_DEBUG (NVDS_APP, "Source bin created");

  return TRUE;
}

gboolean
create_multi_source_bin (guint num_sub_bins, NvDsSourceConfig * configs,
    NvDsSrcParentBin * bin)
{
  gboolean ret = FALSE;
  guint i = 0;

  bin->reset_thread = NULL;

  bin->bin = gst_bin_new ("multi_src_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'multi_src_bin'");
    goto done;
  }

  g_object_set (bin->bin, "message-forward", TRUE, NULL);

  bin->streammux =
      gst_element_factory_make (NVDS_ELEM_STREAM_MUX, "src_bin_muxer");
  if (!bin->streammux) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'src_bin_muxer'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), bin->streammux);

  for (i = 0; i < num_sub_bins; i++) {
    if (!configs[i].enable) {
      continue;
    }

    gchar elem_name[50];
    g_snprintf (elem_name, sizeof (elem_name), "src_sub_bin%d", i);
    bin->sub_bins[i].bin = gst_bin_new (elem_name);
    if (!bin->sub_bins[i].bin) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }

    bin->sub_bins[i].bin_id = bin->sub_bins[i].source_id = i;
    configs[i].live_source = TRUE;
    bin->live_source = TRUE;
    bin->sub_bins[i].eos_done = TRUE;
    bin->sub_bins[i].reset_done = TRUE;

    bin->sub_bins[i].parent_bin = bin;

    switch (configs[i].type) {
      case NV_DS_SOURCE_CAMERA_CSI:
      case NV_DS_SOURCE_CAMERA_V4L2:
        if (!create_camera_source_bin (&configs[i], &bin->sub_bins[i])) {
          return FALSE;
        }
        break;
      case NV_DS_SOURCE_URI:
        if (!create_uridecode_src_bin (&configs[i], &bin->sub_bins[i])) {
          return FALSE;
        }
        bin->live_source = configs[i].live_source;
        break;
      case NV_DS_SOURCE_RTSP:
        if (!create_rtsp_src_bin (&configs[i], &bin->sub_bins[i])) {
          return FALSE;
        }
        break;
      case NV_DS_SOURCE_AUDIO_WAV:
        if (!create_audio_source_bin (&configs[i], &bin->sub_bins[i])) {
          return FALSE;
        }
        break;
      case NV_DS_SOURCE_AUDIO_URI:
        if (!create_uridecode_src_bin_audio (&configs[i], &bin->sub_bins[i])) {
          return FALSE;
        }
        bin->live_source = configs->live_source;
        break;
      case NV_DS_SOURCE_ALSA_SRC:
        if (!create_audio_source_bin (&configs[i], &bin->sub_bins[i])) {
          return FALSE;
        }
        break;
      default:
        NVGSTDS_ERR_MSG_V ("Source type not yet implemented!\n");
        return FALSE;
    }

    gst_bin_add (GST_BIN (bin->bin), bin->sub_bins[i].bin);

    if (!link_element_to_streammux_sink_pad (bin->streammux,
            bin->sub_bins[i].bin, i)) {
      NVGSTDS_ERR_MSG_V ("source %d cannot be linked to mux's sink pad %p\n", i,
          bin->streammux);
      goto done;
    }

    bin->num_bins++;
  }
  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->streammux, "src");

  if (install_mux_eosmonitor_probe) {
    NVGSTDS_ELEM_ADD_PROBE (bin->nvstreammux_eosmonitor_probe, bin->streammux,
        "src", nvstreammux_eosmonitor_probe_func,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, bin);
  }

  ret = TRUE;

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static void
set_properties_nvuribin (GstElement * element_, NvDsSourceConfig const *config)
{
  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element_)->elementfactory;
  if (!g_strcmp0(GST_OBJECT_NAME(factory), "nvurisrcbin"))
    g_object_set (element_, "uri", config->uri, NULL);
  if (config->num_extra_surfaces)
    g_object_set (element_, "num-extra-surfaces", config->num_extra_surfaces,
        NULL);
  if (config->gpu_id)
    g_object_set (element_, "gpu-id", config->gpu_id, NULL);
  g_object_set (element_, "cudadec-memtype", config->cuda_memory_type, NULL);
  g_object_set (element_, "low-latency-mode", config->low_latency_mode, NULL);
  g_object_set (element_, "drop-on-latency", config->drop_on_latency, NULL);
  if (config->drop_frame_interval)
    g_object_set (element_, "drop-frame-interval", config->drop_frame_interval,
        NULL);
  if (config->extract_sei_type5_data)
    g_object_set (element_, "extract-sei-type5-data", config->extract_sei_type5_data,
        NULL);
  if (config->select_rtp_protocol)
    g_object_set (element_, "select-rtp-protocol", config->select_rtp_protocol,
        NULL);
  if (config->leaky)
    g_object_set (element_, "leaky", config->leaky, NULL);
  if (config->max_size_buffers)
    g_object_set (element_, "max-size-buffers", config->max_size_buffers, NULL);
  g_object_set (element_, "buffer-mode", config->buffer_mode,
    NULL);
  if (config->loop)
    g_object_set (element_, "file-loop", config->loop, NULL);
  if (config->smart_record)
    g_object_set (element_, "smart-record", config->smart_record, NULL);
  if (config->smart_rec_cache_size)
    g_object_set (element_, "smart-rec-cache", config->smart_rec_cache_size,
        NULL);
  if (config->smart_rec_container)
    g_object_set (element_, "smart-rec-container", config->smart_rec_container,
        NULL);
  if (config->smart_rec_def_duration)
    g_object_set (element_, "smart-rec-default-duration",
        config->smart_rec_def_duration, NULL);
  if (config->dir_path)
    g_object_set (element_, "smart-rec-dir-path", config->dir_path, NULL);
  if (config->rtsp_reconnect_interval_sec){
    g_object_set (element_, "rtsp-reconnect-interval",
        config->rtsp_reconnect_interval_sec, NULL);
    g_object_set (element_, "rtsp-reconnect-attempts",
        config->rtsp_reconnect_attempts, NULL);
  }
  if (config->init_rtsp_reconnect_interval_sec){
    g_object_set (element_, "init-rtsp-reconnect-interval",
        config->init_rtsp_reconnect_interval_sec, NULL);
    g_object_set (element_, "rtsp-reconnect-attempts",
        config->rtsp_reconnect_attempts, NULL);
  }
  if (config->latency)
    g_object_set (element_, "latency", config->latency, NULL);
  if (config->udp_buffer_size)
    g_object_set (element_, "udp-buffer-size", config->udp_buffer_size, NULL);
  if (config->sensorIdToPadIdMapping) {
    GParamSpec *pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element_), "sensorID-padID-mapping");
    if (pspec != NULL)
     g_object_set (element_, "sensorID-padID-mapping", config->sensorIdToPadIdMapping, NULL);
  }
  g_object_set (element_, "simulate-fps-interval-ms", config->simulate_fps_interval_ms, NULL);
}

gboolean
create_nvmultiurisrcbin_bin (guint num_sub_bins, NvDsSourceConfig * configs,
    NvDsSrcParentBin * bin)
{
  gboolean ret = FALSE;
  guint i = 0;

  bin->reset_thread = NULL;

  bin->bin = gst_bin_new ("multiuri_src_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'multiuri_src_bin'");
    goto done;
  }

  g_object_set (bin->bin, "message-forward", TRUE, NULL);

  bin->nvmultiurisrcbin = bin->streammux =
      gst_element_factory_make (NVDS_ELEM_NVMULTIURISRCBIN,
      "src_nvmultiurisrcbin");
  if (!bin->streammux) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'src_nvmultiurisrcbin'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), bin->streammux);

  /** set properties for the nvurisrcbin if atleast one uri was provided */
  for (i = 0; i < num_sub_bins; i++) {
    if (!configs[i].enable) {
      continue;
    }
    set_properties_nvuribin (bin->nvmultiurisrcbin, &configs[i]);
  }
  if (num_sub_bins == 0) {
    set_properties_nvuribin (bin->nvmultiurisrcbin, configs);
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->streammux, "src");

  if (install_mux_eosmonitor_probe) {
    NVGSTDS_ELEM_ADD_PROBE (bin->nvstreammux_eosmonitor_probe, bin->streammux,
        "src", nvstreammux_eosmonitor_probe_func,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, bin);
  }

  ret = TRUE;

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

gboolean
reset_source_pipeline (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  GstState state = GST_STATE_NULL, pending = GST_STATE_NULL;
  GstStateChangeReturn ret;

  g_mutex_lock (&src_bin->bin_lock);
  gettimeofday (&src_bin->last_buffer_time, NULL);
  gettimeofday (&src_bin->last_reconnect_time, NULL);
  g_mutex_unlock (&src_bin->bin_lock);

  GstElement *send_event_element = NULL;
  if (src_bin->dewarper_bin.bin != NULL) {
    send_event_element = src_bin->dewarper_bin.bin;
  } else {
    send_event_element = src_bin->cap_filter1;
  }
  gst_element_send_event (GST_ELEMENT (send_event_element),
      gst_event_new_flush_start ());
  gst_element_send_event (GST_ELEMENT (send_event_element),
      gst_event_new_flush_stop (TRUE));
  if (gst_element_set_state (src_bin->bin,
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin->bin, "Can't set source bin to NULL");
    return FALSE;
  }
  NVGSTDS_INFO_MSG_V ("Resetting source %d", src_bin->bin_id);

  GST_CAT_INFO (NVDS_APP, "Reset source pipeline %s %p\n,", __func__, src_bin);
  if (!gst_element_sync_state_with_parent (src_bin->bin)) {
    GST_ERROR_OBJECT (src_bin->bin, "Couldn't sync state with parent");
  }

  if (src_bin->parser != NULL) {
    if (!gst_element_send_event (GST_ELEMENT (src_bin->parser),
            gst_nvevent_new_stream_reset (0)))
      GST_ERROR_OBJECT (src_bin->parser,
          "Interrupted, Reconnection event not sent");
  }

  ret = gst_element_get_state (src_bin->bin, &state, &pending, 0);

  GST_CAT_DEBUG (NVDS_APP, "Bin %d %p: state:%s pending:%s ret:%s",
      src_bin->bin_id, src_bin, gst_element_state_get_name (state),
      gst_element_state_get_name (pending),
      gst_element_state_change_return_get_name (ret));

  if (ret == GST_STATE_CHANGE_ASYNC || ret == GST_STATE_CHANGE_NO_PREROLL) {
    if (!src_bin->async_state_watch_running)
      g_timeout_add (20, watch_source_async_state_change, src_bin);
    src_bin->async_state_watch_running = TRUE;
    src_bin->reconfiguring = TRUE;
  } else if (ret == GST_STATE_CHANGE_SUCCESS && state == GST_STATE_PLAYING) {
    src_bin->reconfiguring = FALSE;
  }
  return FALSE;
}

gboolean
set_source_to_playing (gpointer data)
{
  NvDsSrcBin *subBin = (NvDsSrcBin *) data;
  if (subBin->reconfiguring) {
    gst_element_set_state (subBin->bin, GST_STATE_PLAYING);
    GST_CAT_INFO (NVDS_APP, "Reconfiguring %s  %p\n,", __func__, subBin);

    subBin->reconfiguring = FALSE;
  }
  return FALSE;
}

gpointer
reset_encodebin (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  g_usleep (10000);
  GST_CAT_INFO (NVDS_APP, "Reset called %s %p\n,", __func__, src_bin);

  GST_CAT_INFO (NVDS_APP, "Reset setting null for sink %s %p\n,", __func__,
      src_bin);
  src_bin->reset_done = TRUE;

  return NULL;
}
