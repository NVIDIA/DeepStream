/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <string.h>

#include "gstnvdslogger.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "gst-nvevent.h"
#include "nvds_rest_metrics.h"

#include <sys/time.h>

static GQuark dsmeta_quark = 0;

static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

GST_DEBUG_CATEGORY_STATIC (gst_nvdslogger_debug);
#define GST_CAT_DEFAULT gst_nvdslogger_debug

#define BUF_PTS_TO_RUNNING_TIME(buf_pts) \
      gst_segment_to_running_time(&trans->segment, GST_FORMAT_TIME, buf_pts)

/* Nvdslogger signals and args */
enum
{
  SIGNAL_HANDOFF,
  /* FILL ME */
  LAST_SIGNAL
};

#define DEFAULT_SLEEP_TIME              0
#define DEFAULT_DUPLICATE               1
#define DEFAULT_ERROR_AFTER             -1
#define DEFAULT_DROP_PROBABILITY        0.0
#define DEFAULT_DROP_BUFFER_FLAGS       (GstBufferFlags) 0
#define DEFAULT_DATARATE                0
#define DEFAULT_SILENT                  TRUE
#define DEFAULT_SINGLE_SEGMENT          FALSE
#define DEFAULT_DUMP                    FALSE
#define DEFAULT_SYNC                    FALSE
#define DEFAULT_CHECK_IMPERFECT_TIMESTAMP FALSE
#define DEFAULT_CHECK_IMPERFECT_OFFSET    FALSE
#define DEFAULT_SIGNAL_HANDOFFS           TRUE
#define DEFAULT_TS_OFFSET               0
#define DEFAULT_DROP_ALLOCATION         FALSE
#define DEFAULT_FPS_MEASUREMENT_INTERVAL_SEC 5
#define DEFAULT_LATENCY              0
#define DEFAULT_SHOW_FPS   FALSE
#define DEFAULT_SHOW_LATENCY   FALSE


enum
{
  PROP_0,
  PROP_SLEEP_TIME,
  PROP_ERROR_AFTER,
  PROP_DROP_PROBABILITY,
  PROP_DROP_BUFFER_FLAGS,
  PROP_DATARATE,
  PROP_SILENT,
  PROP_SINGLE_SEGMENT,
  PROP_LAST_MESSAGE,
  PROP_DUMP,
  PROP_SYNC,
  PROP_TS_OFFSET,
  PROP_CHECK_IMPERFECT_TIMESTAMP,
  PROP_CHECK_IMPERFECT_OFFSET,
  PROP_SIGNAL_HANDOFFS,
  PROP_DROP_ALLOCATION,
  PROP_FPS_MEASUREMENT_INTERVAL_SEC,
  PROP_LATENCY,
  PROP_SHOW_FPS,
  PROP_SHOW_LATENCY
};


#define gst_nvdslogger_parent_class parent_class
G_DEFINE_TYPE (GstNvdslogger, gst_nvdslogger, GST_TYPE_BASE_TRANSFORM);
#if 0
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvdslogger_debug, "nvdslogger", 0, "logger element");
G_DEFINE_TYPE_WITH_CODE (GstNvdslogger, gst_nvdslogger, GST_TYPE_BASE_TRANSFORM,
    _do_init);
#endif

static void gst_nvdslogger_finalize (GObject * object);
static void gst_nvdslogger_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdslogger_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvdslogger_sink_event (GstBaseTransform * trans,
    GstEvent * event);
static GstFlowReturn gst_nvdslogger_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static gboolean gst_nvdslogger_start (GstBaseTransform * trans);
static gboolean gst_nvdslogger_stop (GstBaseTransform * trans);
static GstStateChangeReturn gst_nvdslogger_change_state (GstElement * element,
    GstStateChange transition);
static gboolean gst_nvdslogger_accept_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps);
static gboolean gst_nvdslogger_query (GstBaseTransform * base,
    GstPadDirection direction, GstQuery * query);

static guint gst_nvdslogger_signals[LAST_SIGNAL] = { 0 };

static GParamSpec *pspec_last_message = NULL;

static gdouble nvds_get_current_system_timestamp(void);

static gchar * gst_buffer_get_flags_string (GstBuffer * buffer);

static gchar * gst_buffer_get_meta_string (GstBuffer * buffer);

static GstClockTime gst_get_current_running_time (GstElement * element);


static gdouble nvds_get_current_system_timestamp()
{
  struct timeval t1;
  double elapsedTime = 0;
  gettimeofday(&t1, NULL);
  elapsedTime = (t1.tv_sec) * 1000.0;
  elapsedTime += (t1.tv_usec) / 1000.0;
  return elapsedTime;
}

static void capture_fps_data(NvDsLoggerPerfStructInt *str)
{

    guint num_sources = 0;
    for (guint i = 0; i < MAX_NUM_SOURCES ; i++) {
        if (str->instance_str[i].is_valid) {
            num_sources++;
        }
    }

    NvDsMetricsFpsData fps_data[num_sources];
    memset(fps_data, 0, sizeof(NvDsMetricsFpsData) * num_sources);

    guint fps_idx = 0;
    for (guint i = 0; i < MAX_NUM_SOURCES && fps_idx < num_sources; i++) {
        if (str->instance_str[i].is_valid) {
            fps_data[fps_idx].source_id = i;
            fps_data[fps_idx].fps_val = str->instance_str[i].fps_val;
            fps_idx++;
        }
    }

    nvds_update_shared_fps_data(fps_data, num_sources);

}

static gboolean
nvdslogger_perf (gpointer data)
{
  NvDsLoggerPerfStructInt *str = (NvDsLoggerPerfStructInt *) data;
  g_mutex_lock (&str->struct_lock);

  static gboolean header_print = FALSE;
  gboolean show_fps = str->show_fps;

  if (show_fps && header_print)
    g_print("**PERF : ");

  for (guint i = 0; i < MAX_NUM_SOURCES ; i++) {

    if (!str->instance_str[i].is_valid)
      continue;

    if (str->instance_str[i].eos == TRUE)
    {
      str->instance_str[i].fps_val = 0;
      goto perf_print;
    }

    str->instance_str[i].curr_time = nvds_get_current_system_timestamp();

    if(str->instance_str[i].last_time == 0) {
      str->instance_str[i].buffer_cnt = 0;
      str->instance_str[i].last_time = str->instance_str[i].curr_time;
      if (!header_print)
        header_print = TRUE;
      continue;
    }
    else {
      str->instance_str[i].time_diff =
        str->instance_str[i].curr_time - str->instance_str[i].last_time;
      str->instance_str[i].fps_val =
        (str->instance_str[i].buffer_cnt/str->instance_str[i].time_diff) * 1000;
    }

perf_print:
    /* FPS prints for each pad indexes*/
    if (show_fps)
      g_print("FPS_%d (%.2f)\t", i, str->instance_str[i].fps_val);
    capture_fps_data(str);
  }

  if (show_fps)
    g_print("\n");
  g_mutex_unlock (&str->struct_lock);

  return TRUE;
}

static void
gst_nvdslogger_finalize (GObject * object)
{
  GstNvdslogger *nvdslogger;

  nvdslogger = GST_NVDSLOGGER (object);

  g_free (nvdslogger->last_message);
  g_cond_clear (&nvdslogger->blocked_cond);
  g_free(nvdslogger->str);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_nvdslogger_class_init (GstNvdsloggerClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetrans_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  gstbasetrans_class = GST_BASE_TRANSFORM_CLASS (klass);

  gobject_class->set_property = gst_nvdslogger_set_property;
  gobject_class->get_property = gst_nvdslogger_get_property;

  g_object_class_install_property (gobject_class, PROP_SLEEP_TIME,
      g_param_spec_uint ("sleep-time", "Sleep time",
          "Microseconds to sleep between processing", 0, G_MAXUINT,
          DEFAULT_SLEEP_TIME, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_ERROR_AFTER,
      g_param_spec_int ("error-after", "Error After", "Error after N buffers",
          G_MININT, G_MAXINT, DEFAULT_ERROR_AFTER,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_DROP_PROBABILITY,
      g_param_spec_float ("drop-probability", "Drop Probability",
          "The Probability a buffer is dropped", 0.0, 1.0,
          DEFAULT_DROP_PROBABILITY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /**
   * GstNvdslogger:drop-buffer-flags:
   *
   * Drop buffers with the given flags.
   *
   * Since: 1.8
   **/
  g_object_class_install_property (gobject_class, PROP_DROP_BUFFER_FLAGS,
      g_param_spec_flags ("drop-buffer-flags", "Check flags to drop buffers",
          "Drop buffers with the given flags",
          GST_TYPE_BUFFER_FLAGS, DEFAULT_DROP_BUFFER_FLAGS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_DATARATE,
      g_param_spec_int ("datarate", "Datarate",
          "(Re)timestamps buffers with number of bytes per second (0 = inactive)",
          0, G_MAXINT, DEFAULT_DATARATE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "silent", "silent", DEFAULT_SILENT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SINGLE_SEGMENT,
      g_param_spec_boolean ("single-segment", "Single Segment",
          "Timestamp buffers and eat segments so as to appear as one segment",
          DEFAULT_SINGLE_SEGMENT, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  pspec_last_message = g_param_spec_string ("last-message", "last-message",
      "last-message", NULL, (GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_LAST_MESSAGE,
      pspec_last_message);
  g_object_class_install_property (gobject_class, PROP_DUMP,
      g_param_spec_boolean ("dump", "Dump", "Dump buffer contents to stdout",
          DEFAULT_DUMP, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SYNC,
      g_param_spec_boolean ("sync", "Synchronize",
          "Synchronize to pipeline clock", DEFAULT_SYNC,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_TS_OFFSET,
      g_param_spec_int64 ("ts-offset", "Timestamp offset for synchronisation",
          "Timestamp offset in nanoseconds for synchronisation, negative for earlier sync",
          G_MININT64, G_MAXINT64, DEFAULT_TS_OFFSET,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class,
      PROP_CHECK_IMPERFECT_TIMESTAMP,
      g_param_spec_boolean ("check-imperfect-timestamp",
          "Check for discontiguous timestamps",
          "Send element messages if timestamps and durations do not match up",
          DEFAULT_CHECK_IMPERFECT_TIMESTAMP,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_CHECK_IMPERFECT_OFFSET,
      g_param_spec_boolean ("check-imperfect-offset",
          "Check for discontiguous offset",
          "Send element messages if offset and offset_end do not match up",
          DEFAULT_CHECK_IMPERFECT_OFFSET,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_FPS_MEASUREMENT_INTERVAL_SEC,
      g_param_spec_uint ("fps_measurement_interval_sec", "fps-measurement-interval-sec",
          "Interval in seconds for which buffers will be counted and to be used for FPS calculation",
          0, G_MAXUINT, DEFAULT_FPS_MEASUREMENT_INTERVAL_SEC,
           (GParamFlags) (G_PARAM_READWRITE )));

  g_object_class_install_property (gobject_class, PROP_SHOW_FPS,
      g_param_spec_boolean ("show-fps", "Show FPS",
          "Enable or disable FPS display prints",
          DEFAULT_SHOW_FPS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SHOW_LATENCY,
      g_param_spec_boolean ("show-latency", "Show Latency",
          "Enable or disable frame-level latency display prints",
          DEFAULT_SHOW_LATENCY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /**
   * GstNvdslogger:signal-handoffs
   *
   * If set to %TRUE, the nvdslogger will emit a handoff signal when handling a buffer.
   * When set to %FALSE, no signal will be emitted, which might improve performance.
   */
  g_object_class_install_property (gobject_class, PROP_SIGNAL_HANDOFFS,
      g_param_spec_boolean ("signal-handoffs",
          "Signal handoffs", "Send a signal before pushing the buffer",
          DEFAULT_SIGNAL_HANDOFFS, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DROP_ALLOCATION,
      g_param_spec_boolean ("drop-allocation", "Drop allocation query",
          "Don't forward allocation queries", DEFAULT_DROP_ALLOCATION,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LATENCY,
      g_param_spec_uint ("latency", "latency",
          "Microseconds to delay initial packet", 0, G_MAXUINT,
          DEFAULT_LATENCY, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /**
   * GstNvdslogger::handoff:
   * @nvdslogger: the nvdslogger instance
   * @buffer: the buffer that just has been received
   * @pad: the pad that received it
   *
   * This signal gets emitted before passing the buffer downstream.
   */
  gst_nvdslogger_signals[SIGNAL_HANDOFF] =
      g_signal_new ("handoff", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstNvdsloggerClass, handoff), NULL, NULL,
      g_cclosure_marshal_generic, G_TYPE_NONE, 1,
      GST_TYPE_BUFFER | G_SIGNAL_TYPE_STATIC_SCOPE);

  gobject_class->finalize = gst_nvdslogger_finalize;

  gst_element_class_set_static_metadata (gstelement_class,
      "Nvdslogger",
      "Generic",
      "Pass data without modification",
      "NVIDIA Corporation. Post on Deepstream forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
  gst_element_class_add_static_pad_template (gstelement_class, &srctemplate);
  gst_element_class_add_static_pad_template (gstelement_class, &sinktemplate);

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_nvdslogger_change_state);

  gstbasetrans_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvdslogger_sink_event);
  gstbasetrans_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvdslogger_transform_ip);
  gstbasetrans_class->start = GST_DEBUG_FUNCPTR (gst_nvdslogger_start);
  gstbasetrans_class->stop = GST_DEBUG_FUNCPTR (gst_nvdslogger_stop);
  gstbasetrans_class->accept_caps =
      GST_DEBUG_FUNCPTR (gst_nvdslogger_accept_caps);
  gstbasetrans_class->query = gst_nvdslogger_query;
}

static void
gst_nvdslogger_init (GstNvdslogger* nvdslogger)
{
  nvdslogger->sleep_time = DEFAULT_SLEEP_TIME;
  nvdslogger->error_after = DEFAULT_ERROR_AFTER;
  nvdslogger->drop_probability = DEFAULT_DROP_PROBABILITY;
  nvdslogger->drop_buffer_flags = DEFAULT_DROP_BUFFER_FLAGS;
  nvdslogger->datarate = DEFAULT_DATARATE;
  nvdslogger->silent = DEFAULT_SILENT;
  nvdslogger->single_segment = DEFAULT_SINGLE_SEGMENT;
  nvdslogger->sync = DEFAULT_SYNC;
  nvdslogger->check_imperfect_timestamp = DEFAULT_CHECK_IMPERFECT_TIMESTAMP;
  nvdslogger->check_imperfect_offset = DEFAULT_CHECK_IMPERFECT_OFFSET;
  nvdslogger->dump = DEFAULT_DUMP;
  nvdslogger->last_message = NULL;
  nvdslogger->signal_handoffs = DEFAULT_SIGNAL_HANDOFFS;
  nvdslogger->ts_offset = DEFAULT_TS_OFFSET;
  nvdslogger->fps_measurement_interval_sec = DEFAULT_FPS_MEASUREMENT_INTERVAL_SEC;
  nvdslogger->latency = DEFAULT_LATENCY;
  nvdslogger->show_fps = DEFAULT_SHOW_FPS;
  nvdslogger->show_latency = DEFAULT_SHOW_LATENCY;
  nvdslogger->fnumber = 0;
  g_cond_init (&nvdslogger->blocked_cond);

  if (!dsmeta_quark)
    dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

  gst_base_transform_set_gap_aware (GST_BASE_TRANSFORM_CAST (nvdslogger), TRUE);
}

static void
gst_nvdslogger_notify_last_message (GstNvdslogger* nvdslogger)
{
  g_object_notify_by_pspec ((GObject *) nvdslogger, pspec_last_message);
}

static GstFlowReturn
gst_nvdslogger_do_sync (GstNvdslogger* nvdslogger, GstClockTime running_time)
{
  GstFlowReturn ret = GST_FLOW_OK;

  if (nvdslogger->sync &&
      GST_BASE_TRANSFORM_CAST (nvdslogger)->segment.format == GST_FORMAT_TIME) {
    GstClock *clock;

    GST_OBJECT_LOCK (nvdslogger);

    if (nvdslogger->flushing) {
      GST_OBJECT_UNLOCK (nvdslogger);
      return GST_FLOW_FLUSHING;
    }

    while (nvdslogger->blocked)
      g_cond_wait (&nvdslogger->blocked_cond, GST_OBJECT_GET_LOCK (nvdslogger));

    if (nvdslogger->flushing) {
      GST_OBJECT_UNLOCK (nvdslogger);
      return GST_FLOW_FLUSHING;
    }

    if ((clock = GST_ELEMENT (nvdslogger)->clock)) {
      GstClockReturn cret;
      GstClockTime timestamp;
      GstClockTimeDiff ts_offset = nvdslogger->ts_offset;

      timestamp = running_time + GST_ELEMENT (nvdslogger)->base_time +
          nvdslogger->upstream_latency;
      if (ts_offset < 0) {
        ts_offset = -ts_offset;
        if ((GstClockTime)ts_offset < timestamp)
          timestamp -= ts_offset;
        else
          timestamp = 0;
      } else
        timestamp += ts_offset;

      /* save id if we need to unlock */
      nvdslogger->clock_id = gst_clock_new_single_shot_id (clock, timestamp);
      GST_OBJECT_UNLOCK (nvdslogger);

      cret = gst_clock_id_wait (nvdslogger->clock_id, NULL);

      GST_OBJECT_LOCK (nvdslogger);
      if (nvdslogger->clock_id) {
        gst_clock_id_unref (nvdslogger->clock_id);
        nvdslogger->clock_id = NULL;
      }
      if (cret == GST_CLOCK_UNSCHEDULED || nvdslogger->flushing)
        ret = GST_FLOW_FLUSHING;
    }
    GST_OBJECT_UNLOCK (nvdslogger);
  }

  return ret;
}

static gboolean
gst_nvdslogger_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstNvdslogger *nvdslogger;
  gboolean ret = TRUE;

  nvdslogger = GST_NVDSLOGGER (trans);
  guint source_id = 0;

  if (!nvdslogger->silent) {
    const GstStructure *s;
    const gchar *tstr;
    gchar *sstr;

    GST_OBJECT_LOCK (nvdslogger);
    g_free (nvdslogger->last_message);

    tstr = gst_event_type_get_name (GST_EVENT_TYPE (event));
    if ((s = gst_event_get_structure (event)))
      sstr = gst_structure_to_string (s);
    else
      sstr = g_strdup ("");

    nvdslogger->last_message =
        g_strdup_printf ("event   ******* (%s:%s) E (type: %s (%d), %s) %p",
        GST_DEBUG_PAD_NAME (trans->sinkpad), tstr, GST_EVENT_TYPE (event),
        sstr, event);
    g_free (sstr);
    GST_OBJECT_UNLOCK (nvdslogger);

    gst_nvdslogger_notify_last_message (nvdslogger);
  }

  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
    gst_nvevent_parse_stream_eos (event, &source_id);
    g_mutex_lock (&nvdslogger->str->struct_lock);
    nvdslogger->str->instance_str[source_id].eos = TRUE;
    nvdslogger->str->instance_str[source_id].last_time = 0;
    g_mutex_unlock (&nvdslogger->str->struct_lock);
  }

  // To report zero fps for inactive sources (e.g streams removed using REST API)
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
  {
    gst_nvevent_parse_pad_deleted (event, &source_id);
    g_mutex_lock (&nvdslogger->str->struct_lock);
    // Reset all FPS tracking fields to ensure clean state for future stream reuse
    nvdslogger->str->instance_str[source_id].last_time = 0;
    nvdslogger->str->instance_str[source_id].curr_time = 0;
    nvdslogger->str->instance_str[source_id].time_diff = 0;
    nvdslogger->str->instance_str[source_id].buffer_cnt = 0;
    nvdslogger->str->instance_str[source_id].fps_val = 0;
    nvdslogger->str->instance_str[source_id].is_valid = FALSE;
    nvdslogger->str->instance_str[source_id].eos = FALSE;
    nvds_cleanup_shared_frame_latency_data();
    g_mutex_unlock (&nvdslogger->str->struct_lock);
  }

  if (nvdslogger->single_segment && (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT)) {
    if (!trans->have_segment) {
      GstEvent *news;
      GstSegment segment;

      gst_event_copy_segment (event, &segment);
      gst_event_copy_segment (event, &trans->segment);
      trans->have_segment = TRUE;

      /* This is the first segment, send out a (0, -1) segment */
      gst_segment_init (&segment, segment.format);
      news = gst_event_new_segment (&segment);

      gst_pad_event_default (trans->sinkpad, GST_OBJECT_CAST (trans), news);
    } else {
      /* need to track segment for proper running time */
      gst_event_copy_segment (event, &trans->segment);
    }
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_GAP &&
      trans->have_segment && trans->segment.format == GST_FORMAT_TIME) {
    GstClockTime start = 0, dur = 0;

    gst_event_parse_gap (event, &start, &dur);
    if (GST_CLOCK_TIME_IS_VALID (start)) {
      start = gst_segment_to_running_time (&trans->segment,
          GST_FORMAT_TIME, start);

      gst_nvdslogger_do_sync (nvdslogger, start);

      /* also transform GAP timestamp similar to buffer timestamps */
      if (nvdslogger->single_segment) {
        gst_event_unref (event);
        event = gst_event_new_gap (start, dur);
      }
    }
  }

  /* Reset previous timestamp, duration and offsets on SEGMENT
   * to prevent false warnings when checking for perfect streams */
  if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
    nvdslogger->prev_timestamp = nvdslogger->prev_duration = GST_CLOCK_TIME_NONE;
    nvdslogger->prev_offset = nvdslogger->prev_offset_end = GST_BUFFER_OFFSET_NONE;
  }

  if (nvdslogger->single_segment && GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
    /* eat up segments */
    gst_event_unref (event);
    ret = TRUE;
  } else {
    if (GST_EVENT_TYPE (event) == GST_EVENT_FLUSH_START) {
      GST_OBJECT_LOCK (nvdslogger);
      nvdslogger->flushing = TRUE;
      if (nvdslogger->clock_id) {
        GST_DEBUG_OBJECT (nvdslogger, "unlock clock wait");
        gst_clock_id_unschedule (nvdslogger->clock_id);
      }
      GST_OBJECT_UNLOCK (nvdslogger);
    } else if (GST_EVENT_TYPE (event) == GST_EVENT_FLUSH_STOP) {
      GST_OBJECT_LOCK (nvdslogger);
      nvdslogger->flushing = FALSE;
      GST_OBJECT_UNLOCK (nvdslogger);
    }

    ret = GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
  }

  return ret;
}

static void
gst_nvdslogger_check_imperfect_timestamp (GstNvdslogger* nvdslogger, GstBuffer * buf)
{
  GstClockTime timestamp = GST_BUFFER_TIMESTAMP (buf);

  /* invalid timestamp drops us out of check.  FIXME: maybe warn ? */
  if (timestamp != GST_CLOCK_TIME_NONE) {
    /* check if we had a previous buffer to compare to */
    if (nvdslogger->prev_timestamp != GST_CLOCK_TIME_NONE &&
        nvdslogger->prev_duration != GST_CLOCK_TIME_NONE) {
      GstClockTime t_expected;
      GstClockTimeDiff dt;

      t_expected = nvdslogger->prev_timestamp + nvdslogger->prev_duration;
      dt = GST_CLOCK_DIFF (t_expected, timestamp);
      if (dt != 0) {
        /*
         * "imperfect-timestamp" bus message:
         * @nvdslogger:        the nvdslogger instance
         * @delta:           the GST_CLOCK_DIFF to the prev timestamp
         * @prev-timestamp:  the previous buffer timestamp
         * @prev-duration:   the previous buffer duration
         * @prev-offset:     the previous buffer offset
         * @prev-offset-end: the previous buffer offset end
         * @cur-timestamp:   the current buffer timestamp
         * @cur-duration:    the current buffer duration
         * @cur-offset:      the current buffer offset
         * @cur-offset-end:  the current buffer offset end
         *
         * This bus message gets emitted if the check-imperfect-timestamp
         * property is set and there is a gap in time between the
         * last buffer and the newly received buffer.
         */
        gst_element_post_message (GST_ELEMENT (nvdslogger),
            gst_message_new_element (GST_OBJECT (nvdslogger),
                gst_structure_new ("imperfect-timestamp",
                    "delta", G_TYPE_INT64, dt,
                    "prev-timestamp", G_TYPE_UINT64,
                    nvdslogger->prev_timestamp, "prev-duration", G_TYPE_UINT64,
                    nvdslogger->prev_duration, "prev-offset", G_TYPE_UINT64,
                    nvdslogger->prev_offset, "prev-offset-end", G_TYPE_UINT64,
                    nvdslogger->prev_offset_end, "cur-timestamp", G_TYPE_UINT64,
                    timestamp, "cur-duration", G_TYPE_UINT64,
                    GST_BUFFER_DURATION (buf), "cur-offset", G_TYPE_UINT64,
                    GST_BUFFER_OFFSET (buf), "cur-offset-end", G_TYPE_UINT64,
                    GST_BUFFER_OFFSET_END (buf), NULL)));
      }
    } else {
      GST_DEBUG_OBJECT (nvdslogger, "can't check data-contiguity, no "
          "offset_end was set on previous buffer");
    }
  }
}

static void
gst_nvdslogger_check_imperfect_offset (GstNvdslogger * nvdslogger, GstBuffer * buf)
{
  guint64 offset;

  offset = GST_BUFFER_OFFSET (buf);

  if (nvdslogger->prev_offset_end != offset &&
      nvdslogger->prev_offset_end != GST_BUFFER_OFFSET_NONE &&
      offset != GST_BUFFER_OFFSET_NONE) {
    /*
     * "imperfect-offset" bus message:
     * @nvdslogger:        the nvdslogger instance
     * @prev-timestamp:  the previous buffer timestamp
     * @prev-duration:   the previous buffer duration
     * @prev-offset:     the previous buffer offset
     * @prev-offset-end: the previous buffer offset end
     * @cur-timestamp:   the current buffer timestamp
     * @cur-duration:    the current buffer duration
     * @cur-offset:      the current buffer offset
     * @cur-offset-end:  the current buffer offset end
     *
     * This bus message gets emitted if the check-imperfect-offset
     * property is set and there is a gap in offsets between the
     * last buffer and the newly received buffer.
     */
    gst_element_post_message (GST_ELEMENT (nvdslogger),
        gst_message_new_element (GST_OBJECT (nvdslogger),
            gst_structure_new ("imperfect-offset", "prev-timestamp",
                G_TYPE_UINT64, nvdslogger->prev_timestamp, "prev-duration",
                G_TYPE_UINT64, nvdslogger->prev_duration, "prev-offset",
                G_TYPE_UINT64, nvdslogger->prev_offset, "prev-offset-end",
                G_TYPE_UINT64, nvdslogger->prev_offset_end, "cur-timestamp",
                G_TYPE_UINT64, GST_BUFFER_TIMESTAMP (buf), "cur-duration",
                G_TYPE_UINT64, GST_BUFFER_DURATION (buf), "cur-offset",
                G_TYPE_UINT64, GST_BUFFER_OFFSET (buf), "cur-offset-end",
                G_TYPE_UINT64, GST_BUFFER_OFFSET_END (buf), NULL)));
  } else {
    GST_DEBUG_OBJECT (nvdslogger, "can't check offset contiguity, no offset "
        "and/or offset_end were set on previous buffer");
  }
}

static const gchar *
print_pretty_time (gchar * ts_str, gsize ts_str_len, GstClockTime ts)
{
  if (ts == GST_CLOCK_TIME_NONE)
    return "none";

  g_snprintf (ts_str, ts_str_len, "%" GST_TIME_FORMAT, GST_TIME_ARGS (ts));
  return ts_str;
}

#define BUFFER_FLAG_SHIFT 4

static gchar *
gst_buffer_get_flags_string (GstBuffer * buffer)
{
  static const char flag_strings[] =
      "\000\000\000\000live\000decode-only\000discont\000resync\000corrupted\000"
      "marker\000header\000gap\000droppable\000delta-unit\000tag-memory\000"
      "FIXME";
  static const guint8 flag_idx[] = { 0, 1, 2, 3, 4, 9, 21, 29, 36, 46, 53,
    60, 64, 74, 85, 96
  };
  unsigned int i;
  int max_bytes;
  char *flag_str, *end;

  /* max size is all flag strings plus a space or terminator after each one */
  max_bytes = sizeof (flag_strings);
  flag_str = (char *) g_malloc (max_bytes);

  end = flag_str;
  end[0] = '\0';
  for (i = BUFFER_FLAG_SHIFT; i < G_N_ELEMENTS (flag_idx); i++)
  {
    if (GST_MINI_OBJECT_CAST (buffer)->flags & (1 << i))
    {
      strcpy (end, flag_strings + flag_idx[i]);
      end += strlen (end);
      end[0] = ' ';
      end[1] = '\0';
      end++;
    }
  }

  return flag_str;
}

static gchar *
gst_buffer_get_meta_string (GstBuffer * buffer)
{
    gpointer state = NULL;
    GstMeta *meta;
    GString *s = NULL;

    while ((meta = gst_buffer_iterate_meta (buffer, &state)))
    {
        const gchar *desc = g_type_name (meta->info->type);

        if (s == NULL)
            s = g_string_new (NULL);
        else
            g_string_append (s, ", ");

        g_string_append (s, desc);
    }

    return (s != NULL) ? g_string_free (s, FALSE) : NULL;
}

static void
gst_nvdslogger_update_last_message_for_buffer (GstNvdslogger * nvdslogger,
    const gchar * action, GstBuffer * buf, gsize size)
{
  gchar dts_str[64], pts_str[64], dur_str[64];
  gchar *flag_str, *meta_str;

  GST_OBJECT_LOCK (nvdslogger);

  flag_str = gst_buffer_get_flags_string (buf);
  meta_str = gst_buffer_get_meta_string (buf);

  g_free (nvdslogger->last_message);
  nvdslogger->last_message = g_strdup_printf ("%s   ******* (%s:%s) "
          "(%" G_GSIZE_FORMAT " bytes, dts: %s, pts: %s, duration: %s, offset: %"
          G_GINT64_FORMAT ", " "offset_end: % " G_GINT64_FORMAT
          ", flags: %08x %s, meta: %s) %p", action,
          GST_DEBUG_PAD_NAME (GST_BASE_TRANSFORM_CAST (nvdslogger)->sinkpad), size,
          print_pretty_time (dts_str, sizeof (dts_str), GST_BUFFER_DTS (buf)),
          print_pretty_time (pts_str, sizeof (pts_str), GST_BUFFER_PTS (buf)),
          print_pretty_time (dur_str, sizeof (dur_str), GST_BUFFER_DURATION (buf)),
          GST_BUFFER_OFFSET (buf), GST_BUFFER_OFFSET_END (buf),
          GST_BUFFER_FLAGS (buf), flag_str, meta_str ? meta_str : "none", buf);
  g_free (flag_str);
  g_free (meta_str);

  GST_OBJECT_UNLOCK (nvdslogger);

  gst_nvdslogger_notify_last_message (nvdslogger);
}

GstClockTime
gst_get_current_clock_time (GstElement * element);
GstClockTime
gst_get_current_clock_time (GstElement * element)
{
  GstClock *clock = NULL;
  GstClockTime ret;
  g_return_val_if_fail (GST_IS_ELEMENT (element), GST_CLOCK_TIME_NONE);
  clock = gst_element_get_clock (element);
  if (!clock) {
    GST_DEBUG_OBJECT (element, "Element has no clock");
    return GST_CLOCK_TIME_NONE;
  }
  ret = gst_clock_get_time (clock);
  gst_object_unref (clock);
  return ret;
}

static GstClockTime
gst_get_current_running_time (GstElement * element)
{
  GstClockTime base_time, clock_time;
  g_return_val_if_fail (GST_IS_ELEMENT (element), GST_CLOCK_TIME_NONE);
  base_time = gst_element_get_base_time (element);
  if (!GST_CLOCK_TIME_IS_VALID (base_time)) {
    GST_DEBUG_OBJECT (element, "Could not determine base time");
    return GST_CLOCK_TIME_NONE;
  }
  clock_time = gst_get_current_clock_time (element);
  if (!GST_CLOCK_TIME_IS_VALID (clock_time)) {
    return GST_CLOCK_TIME_NONE;
  }
  if (clock_time < base_time) {
    GST_DEBUG_OBJECT (element, "Got negative current running time");
    return GST_CLOCK_TIME_NONE;
  }
  return clock_time - base_time;
}

/* Fills frame_latency_data from latency_info; prints batch/frame latency when enabled. */
static void
gst_nvdslogger_print_and_fill_frame_latency (GstNvdslogger * nvdslogger,
    NvDsFrameLatencyInfo * latency_info, guint num_sources_in_batch,
    NvDsMetricsFrameLatency * frame_latency_data)
{
  if (nvdslogger->show_latency) {
    static guint batch_count = 0;
    g_print ("************BATCH-NUM = %d**************\n", batch_count++);
  }

  for (guint i = 0; i < num_sources_in_batch; i++) {
    frame_latency_data[i].source_id = latency_info[i].source_id;
    frame_latency_data[i].frame_num = latency_info[i].frame_num;
    frame_latency_data[i].latency = latency_info[i].latency;

    if (nvdslogger->show_latency) {
      g_print ("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
          latency_info[i].source_id,
          latency_info[i].frame_num,
          latency_info[i].latency);
    }
  }
}


static GstFlowReturn
gst_nvdslogger_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstNvdslogger *nvdslogger = GST_NVDSLOGGER (trans);
  GstClockTime rundts = GST_CLOCK_TIME_NONE;
  GstClockTime runpts = GST_CLOCK_TIME_NONE;
  GstClockTime ts, duration, runtimestamp;
  NvDsLoggerPerfStructInt *str = (NvDsLoggerPerfStructInt *) nvdslogger->str;
  gsize size;
  nvdslogger->fnumber++;

  if (nvdslogger->fnumber==1)
  {
      g_usleep (nvdslogger->latency);
  }

  NvDsBatchMeta *batch_meta = NULL;
  GstMeta *gst_meta = NULL;
  gpointer state = NULL;
  NvDsMeta *dsmeta = NULL;

  while ((gst_meta = gst_buffer_iterate_meta (buf, &state)) != NULL)
  {
      if (!gst_meta_api_type_has_tag (gst_meta->info->api, dsmeta_quark))
      {
          continue;
      }

      dsmeta = (NvDsMeta *) gst_meta;
      /* Check if the metadata of NvDsMeta contains object bounding boxes. */
      if (dsmeta->meta_type == NVDS_BATCH_GST_META)
      {
          batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
          break;
      }
  }

  if (batch_meta == NULL)
  {
      GST_WARNING_OBJECT (nvdslogger, "NvDsBatchMeta not found for input buffer.");
      //return GST_FLOW_ERROR;
  }

  g_mutex_lock (&str->struct_lock);

  /* Count buffers for each sources in given time interval*/

  if (batch_meta) {
   for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame;
      l_frame = l_frame->next) {

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    if (frame_meta->pad_index >= MAX_NUM_SOURCES)
      continue;

    NvDsLoggerPerfStruct *str1 = &str->instance_str[frame_meta->pad_index];
    str1->buffer_cnt++;
    str1->is_valid = TRUE;
    str1->eos = FALSE;
   }
  }
  g_mutex_unlock (&str->struct_lock);

  // Print latency info only if enabled (original verbose behavior)
  if((nvds_enable_latency_measurement || nvds_latency_measurement_silent) && batch_meta)
  {
      NvDsMetaList *l = NULL;
      for (l = batch_meta->batch_user_meta_list; l != NULL; l = l->next)
      {
          NvDsUserMeta *in_user_meta = (NvDsUserMeta *)(l->data);
          if (in_user_meta->base_meta.meta_type == NVDS_LATENCY_MEASUREMENT_META)
          {
              NvDsMetaCompLatency *latency_metadata = (NvDsMetaCompLatency *)in_user_meta->user_meta_data;
              if (!strncmp(latency_metadata->component_name, "nvdsvideotemplate", strlen("nvdsvideotemplate")))
              {
                  g_print ("In time for %s buffer %" GST_TIME_FORMAT " Out time %" GST_TIME_FORMAT  " latency = %.3f ms\n", latency_metadata->component_name,
                          GST_TIME_ARGS(latency_metadata->in_system_timestamp), GST_TIME_ARGS(latency_metadata->out_system_timestamp),
                          latency_metadata->out_system_timestamp-latency_metadata->in_system_timestamp);
              }
              if (!strncmp(latency_metadata->component_name, "nvdsaudiotemplate", strlen("nvdsaudiotemplate")))
              {
                  g_print ("In time for %s buffer %" GST_TIME_FORMAT " Out time %" GST_TIME_FORMAT  " latency = %.3f ms\n", latency_metadata->component_name,
                          GST_TIME_ARGS(latency_metadata->in_system_timestamp), GST_TIME_ARGS(latency_metadata->out_system_timestamp),
                          latency_metadata->out_system_timestamp-latency_metadata->in_system_timestamp);
              }
          }
      }
  }

  // Always measure and collect latency data (if latency measurement is enabled OR silent mode is enabled)
  if ((nvds_enable_latency_measurement || nvds_latency_measurement_silent) && batch_meta) {
    NvDsFrameLatencyInfo latency_info [MAX_NUM_SOURCES] = {0};
    guint num_sources_in_batch = 0;

    // Measure buffer latency
    num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

    if (num_sources_in_batch > 0) {
      NvDsMetricsFrameLatency frame_latency_data [MAX_NUM_SOURCES] = {0};

      gst_nvdslogger_print_and_fill_frame_latency (nvdslogger, latency_info,
          num_sources_in_batch, frame_latency_data);

      nvds_update_shared_frame_latency_data(frame_latency_data, num_sources_in_batch);
    }

  }
  GstClockTime buf_pts = GST_BUFFER_PTS(buf);
  GstClockTime elem_time = gst_get_current_running_time(GST_ELEMENT(nvdslogger));

  if (nvdslogger->silent == 0)
  {
      g_print("%s BUF_PTS : %" GST_TIME_FORMAT " element time %" GST_TIME_FORMAT " elem-buf=%.2f ms buf=%p buffer_pts_to_running_time = %" GST_TIME_FORMAT "nvdslogger base time = %" GST_TIME_FORMAT  "buffer dur %" GST_TIME_FORMAT "\n",
              GST_OBJECT_NAME(nvdslogger),
              GST_TIME_ARGS(buf_pts), GST_TIME_ARGS(elem_time),
              (elem_time - buf_pts) / 1000000.0,
              buf, GST_TIME_ARGS(BUF_PTS_TO_RUNNING_TIME(buf_pts)),
              GST_TIME_ARGS(GST_ELEMENT (nvdslogger)->base_time),
              GST_TIME_ARGS(GST_BUFFER_DURATION (buf)));
      if (BUF_PTS_TO_RUNNING_TIME(buf_pts) + nvdslogger->upstream_latency + GST_BUFFER_DURATION (buf) > elem_time)
      {
          g_print ("BUFFER PTS is greater than ELEMENT TIME, IT's EARLY\n");
      }
      else
      {
          g_print ("BUFFER PTS is lesser than ELEMENT TIME, IT's LATE\n");
      }
  }

  size = gst_buffer_get_size (buf);

  if (nvdslogger->check_imperfect_timestamp)
    gst_nvdslogger_check_imperfect_timestamp (nvdslogger, buf);
  if (nvdslogger->check_imperfect_offset)
    gst_nvdslogger_check_imperfect_offset (nvdslogger, buf);

  /* update prev values */
  nvdslogger->prev_timestamp = GST_BUFFER_TIMESTAMP (buf);
  nvdslogger->prev_duration = GST_BUFFER_DURATION (buf);
  nvdslogger->prev_offset_end = GST_BUFFER_OFFSET_END (buf);
  nvdslogger->prev_offset = GST_BUFFER_OFFSET (buf);

  if (nvdslogger->error_after >= 0) {
    nvdslogger->error_after--;
    if (nvdslogger->error_after == 0)
      goto error_after;
  }

  if (nvdslogger->drop_probability > 0.0) {
    if ((gfloat) (1.0 * rand () / (RAND_MAX)) < nvdslogger->drop_probability)
      goto dropped;
  }

  if (GST_BUFFER_FLAG_IS_SET (buf, nvdslogger->drop_buffer_flags))
    goto dropped;

  if (nvdslogger->dump) {
    GstMapInfo info;

    if (gst_buffer_map (buf, &info, GST_MAP_READ)) {
      gst_util_dump_mem (info.data, info.size);
      gst_buffer_unmap (buf, &info);
    }
  }

  if (!nvdslogger->silent) {
    gst_nvdslogger_update_last_message_for_buffer (nvdslogger, "chain", buf, size);
  }

  if (nvdslogger->datarate > 0) {
    GstClockTime time = gst_util_uint64_scale_int (nvdslogger->offset,
        GST_SECOND, nvdslogger->datarate);

    GST_BUFFER_PTS (buf) = GST_BUFFER_DTS (buf) = time;
    GST_BUFFER_DURATION (buf) = size * GST_SECOND / nvdslogger->datarate;
  }

  if (nvdslogger->signal_handoffs)
    g_signal_emit (nvdslogger, gst_nvdslogger_signals[SIGNAL_HANDOFF], 0, buf);

  if (trans->segment.format == GST_FORMAT_TIME) {
    rundts = gst_segment_to_running_time (&trans->segment,
        GST_FORMAT_TIME, GST_BUFFER_DTS (buf));
    runpts = gst_segment_to_running_time (&trans->segment,
        GST_FORMAT_TIME, GST_BUFFER_PTS (buf));
  }

  if (GST_CLOCK_TIME_IS_VALID (rundts))
    runtimestamp = rundts;
  else if (GST_CLOCK_TIME_IS_VALID (runpts))
    runtimestamp = runpts;
  else
    runtimestamp = 0;
  ret = gst_nvdslogger_do_sync (nvdslogger, runtimestamp);

  nvdslogger->offset += size;

  if (nvdslogger->sleep_time && ret == GST_FLOW_OK)
    g_usleep (nvdslogger->sleep_time);

  if (nvdslogger->single_segment && (trans->segment.format == GST_FORMAT_TIME)
      && (ret == GST_FLOW_OK)) {
    GST_BUFFER_DTS (buf) = rundts;
    GST_BUFFER_PTS (buf) = runpts;
    GST_BUFFER_OFFSET (buf) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_OFFSET_END (buf) = GST_CLOCK_TIME_NONE;
  }

  return ret;

  /* ERRORS */
error_after:
  {
    GST_ELEMENT_ERROR (nvdslogger, CORE, FAILED,
        ("Failed after iterations as requested."), (NULL));
    return GST_FLOW_ERROR;
  }
dropped:
  {
    if (!nvdslogger->silent) {
      gst_nvdslogger_update_last_message_for_buffer (nvdslogger, "dropping", buf,
          size);
    }

    ts = GST_BUFFER_TIMESTAMP (buf);
    if (GST_CLOCK_TIME_IS_VALID (ts)) {
      duration = GST_BUFFER_DURATION (buf);
      gst_pad_push_event (GST_BASE_TRANSFORM_SRC_PAD (nvdslogger),
          gst_event_new_gap (ts, duration));
    }

    /* return DROPPED to basetransform. */
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }
}

static void
gst_nvdslogger_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvdslogger *nvdslogger;

  nvdslogger = GST_NVDSLOGGER (object);

  switch (prop_id) {
    case PROP_SLEEP_TIME:
      nvdslogger->sleep_time = g_value_get_uint (value);
      break;
    case PROP_SILENT:
      nvdslogger->silent = g_value_get_boolean (value);
      break;
    case PROP_SINGLE_SEGMENT:
      nvdslogger->single_segment = g_value_get_boolean (value);
      break;
    case PROP_DUMP:
      nvdslogger->dump = g_value_get_boolean (value);
      break;
    case PROP_ERROR_AFTER:
      nvdslogger->error_after = g_value_get_int (value);
      break;
    case PROP_DROP_PROBABILITY:
      nvdslogger->drop_probability = g_value_get_float (value);
      break;
    case PROP_DROP_BUFFER_FLAGS:
      nvdslogger->drop_buffer_flags = (GstBufferFlags) g_value_get_flags (value);
      break;
    case PROP_DATARATE:
      nvdslogger->datarate = g_value_get_int (value);
      break;
    case PROP_SYNC:
      nvdslogger->sync = g_value_get_boolean (value);
      break;
    case PROP_TS_OFFSET:
      nvdslogger->ts_offset = g_value_get_int64 (value);
      break;
    case PROP_CHECK_IMPERFECT_TIMESTAMP:
      nvdslogger->check_imperfect_timestamp = g_value_get_boolean (value);
      break;
    case PROP_CHECK_IMPERFECT_OFFSET:
      nvdslogger->check_imperfect_offset = g_value_get_boolean (value);
      break;
    case PROP_SIGNAL_HANDOFFS:
      nvdslogger->signal_handoffs = g_value_get_boolean (value);
      break;
    case PROP_DROP_ALLOCATION:
      nvdslogger->drop_allocation = g_value_get_boolean (value);
      break;
    case PROP_FPS_MEASUREMENT_INTERVAL_SEC:
      nvdslogger->fps_measurement_interval_sec =  g_value_get_uint (value);
      break;
    case PROP_LATENCY:
      nvdslogger->latency = g_value_get_uint (value);
      break;
    case PROP_SHOW_FPS:
      nvdslogger->show_fps = g_value_get_boolean (value);
      break;
    case PROP_SHOW_LATENCY:
      nvdslogger->show_latency = g_value_get_boolean (value);
      if (nvdslogger->show_latency) {
        g_setenv("NVDS_LATENCY_MEASUREMENT_SILENT", "1", TRUE);
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  if (nvdslogger->datarate > 0 || nvdslogger->single_segment)
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (nvdslogger), FALSE);
  else
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (nvdslogger), TRUE);
}

static void
gst_nvdslogger_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstNvdslogger *nvdslogger;

  nvdslogger = GST_NVDSLOGGER (object);

  switch (prop_id) {
    case PROP_SLEEP_TIME:
      g_value_set_uint (value, nvdslogger->sleep_time);
      break;
    case PROP_ERROR_AFTER:
      g_value_set_int (value, nvdslogger->error_after);
      break;
    case PROP_DROP_PROBABILITY:
      g_value_set_float (value, nvdslogger->drop_probability);
      break;
    case PROP_DROP_BUFFER_FLAGS:
      g_value_set_flags (value, nvdslogger->drop_buffer_flags);
      break;
    case PROP_DATARATE:
      g_value_set_int (value, nvdslogger->datarate);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, nvdslogger->silent);
      break;
    case PROP_SINGLE_SEGMENT:
      g_value_set_boolean (value, nvdslogger->single_segment);
      break;
    case PROP_DUMP:
      g_value_set_boolean (value, nvdslogger->dump);
      break;
    case PROP_LAST_MESSAGE:
      GST_OBJECT_LOCK (nvdslogger);
      g_value_set_string (value, nvdslogger->last_message);
      GST_OBJECT_UNLOCK (nvdslogger);
      break;
    case PROP_SYNC:
      g_value_set_boolean (value, nvdslogger->sync);
      break;
    case PROP_TS_OFFSET:
      nvdslogger->ts_offset = g_value_get_int64 (value);
      break;
    case PROP_CHECK_IMPERFECT_TIMESTAMP:
      g_value_set_boolean (value, nvdslogger->check_imperfect_timestamp);
      break;
    case PROP_CHECK_IMPERFECT_OFFSET:
      g_value_set_boolean (value, nvdslogger->check_imperfect_offset);
      break;
    case PROP_SIGNAL_HANDOFFS:
      g_value_set_boolean (value, nvdslogger->signal_handoffs);
      break;
    case PROP_DROP_ALLOCATION:
      g_value_set_boolean (value, nvdslogger->drop_allocation);
      break;
    case PROP_FPS_MEASUREMENT_INTERVAL_SEC:
      g_value_set_uint (value, nvdslogger->fps_measurement_interval_sec);
      break;
    case PROP_LATENCY:
      g_value_set_uint (value, nvdslogger->latency);
      break;
    case PROP_SHOW_FPS:
      g_value_set_boolean (value, nvdslogger->show_fps);
      break;
    case PROP_SHOW_LATENCY:
      g_value_set_boolean (value, nvdslogger->show_latency);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_nvdslogger_start (GstBaseTransform * trans)
{
  GstNvdslogger *nvdslogger;

  nvdslogger = GST_NVDSLOGGER (trans);

  nvdslogger->offset = 0;
  nvdslogger->prev_timestamp = GST_CLOCK_TIME_NONE;
  nvdslogger->prev_duration = GST_CLOCK_TIME_NONE;
  nvdslogger->prev_offset_end = GST_BUFFER_OFFSET_NONE;
  nvdslogger->prev_offset = GST_BUFFER_OFFSET_NONE;

  nvdslogger->str = (NvDsLoggerPerfStructInt *)g_malloc0(sizeof(NvDsLoggerPerfStructInt));

  nvdslogger->str->measurement_interval_ms = nvdslogger->fps_measurement_interval_sec * 1000;//in milliseconds
  nvdslogger->str->show_fps = nvdslogger->show_fps;

  nvdslogger->str->perf_measurement_timeout_id =
      g_timeout_add (nvdslogger->str->measurement_interval_ms, (GSourceFunc)nvdslogger_perf,
      nvdslogger->str);

  return TRUE;
}

static gboolean
gst_nvdslogger_stop (GstBaseTransform * trans)
{
  GstNvdslogger *nvdslogger;

  nvdslogger = GST_NVDSLOGGER (trans);

  GST_OBJECT_LOCK (nvdslogger);
  g_free (nvdslogger->last_message);
  nvdslogger->last_message = NULL;
  GST_OBJECT_UNLOCK (nvdslogger);

  return TRUE;
}

static gboolean
gst_nvdslogger_accept_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps)
{
  gboolean ret;
  GstPad *pad;

  /* Proxy accept-caps */

  if (direction == GST_PAD_SRC)
    pad = GST_BASE_TRANSFORM_SINK_PAD (base);
  else
    pad = GST_BASE_TRANSFORM_SRC_PAD (base);

  ret = gst_pad_peer_query_accept_caps (pad, caps);

  return ret;
}

static gboolean
gst_nvdslogger_query (GstBaseTransform * base, GstPadDirection direction,
    GstQuery * query)
{
  GstNvdslogger *nvdslogger;
  gboolean ret;

  nvdslogger = GST_NVDSLOGGER (base);

  if (GST_QUERY_TYPE (query) == GST_QUERY_ALLOCATION &&
      nvdslogger->drop_allocation) {
    GST_DEBUG_OBJECT (nvdslogger, "Dropping allocation query.");
    return FALSE;
  }

  ret = GST_BASE_TRANSFORM_CLASS (parent_class)->query (base, direction, query);

  if (GST_QUERY_TYPE (query) == GST_QUERY_LATENCY) {
    gboolean live = FALSE;
    GstClockTime min = 0, max = 0;

    if (ret) {
      gst_query_parse_latency (query, &live, &min, &max);

      if (nvdslogger->sync && max < min) {
        GST_ELEMENT_WARNING (base, CORE, CLOCK, (NULL),
            ("Impossible to configure latency before nvdslogger sync=true:"
                " max %" GST_TIME_FORMAT " < min %"
                GST_TIME_FORMAT ". Add queues or other buffering elements.",
                GST_TIME_ARGS (max), GST_TIME_ARGS (min)));
      }
    }

    /* Ignore the upstream latency if it is not live */
    GST_OBJECT_LOCK (nvdslogger);
    if (live)
      nvdslogger->upstream_latency = min;
    else
      nvdslogger->upstream_latency = 0;
    GST_OBJECT_UNLOCK (nvdslogger);

    gst_query_set_latency (query, live || nvdslogger->sync, min, max);
    ret = TRUE;
  }
  return ret;
}

static GstStateChangeReturn
gst_nvdslogger_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret;
  GstNvdslogger *nvdslogger = GST_NVDSLOGGER (element);
  gboolean no_preroll = FALSE;

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_OBJECT_LOCK (nvdslogger);
      nvdslogger->flushing = FALSE;
      nvdslogger->blocked = TRUE;
      GST_OBJECT_UNLOCK (nvdslogger);
      if (nvdslogger->sync)
        no_preroll = TRUE;
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_OBJECT_LOCK (nvdslogger);
      nvdslogger->blocked = FALSE;
      g_cond_broadcast (&nvdslogger->blocked_cond);
      GST_OBJECT_UNLOCK (nvdslogger);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_OBJECT_LOCK (nvdslogger);
      nvdslogger->flushing = TRUE;
      if (nvdslogger->clock_id) {
        GST_DEBUG_OBJECT (nvdslogger, "unlock clock wait");
        gst_clock_id_unschedule (nvdslogger->clock_id);
      }
      nvdslogger->blocked = FALSE;
      g_cond_broadcast (&nvdslogger->blocked_cond);
      GST_OBJECT_UNLOCK (nvdslogger);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_OBJECT_LOCK (nvdslogger);
      nvdslogger->upstream_latency = 0;
      nvdslogger->blocked = TRUE;
      GST_OBJECT_UNLOCK (nvdslogger);
      if (nvdslogger->sync)
        no_preroll = TRUE;
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      break;
    default:
      break;
  }

  if (no_preroll && ret == GST_STATE_CHANGE_SUCCESS)
    ret = GST_STATE_CHANGE_NO_PREROLL;

  return ret;
}

#ifndef PACKAGE
#define PACKAGE "nvdslogger"
#endif

#define PACKAGE_DESCRIPTION "Gstreamer plugin for logging"
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer nVidia logger Plugin"
#define PACKAGE_URL "http://nvidia.com/"

static gboolean
nvdslogger_init (GstPlugin * nvdslogger)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template nvdslogger' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_nvdslogger_debug, "nvdslogger", 0, "nvdslogger");

  return gst_element_register (nvdslogger, "nvdslogger", GST_RANK_NONE,
      GST_TYPE_NVDSLOGGER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_logger,
    PACKAGE_DESCRIPTION,
    nvdslogger_init, DS_VERSION, PACKAGE_LICENSE, PACKAGE_NAME, PACKAGE_URL)
