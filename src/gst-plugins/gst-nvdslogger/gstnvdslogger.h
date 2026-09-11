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

#ifndef __GST_NVDSLOGGER_H__
#define __GST_NVDSLOGGER_H__


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS


#define GST_TYPE_NVDSLOGGER \
  (gst_nvdslogger_get_type())
#define GST_NVDSLOGGER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSLOGGER,GstNvdslogger))
#define GST_NVDSLOGGER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSLOGGER,GstNvdsloggerClass))
#define GST_IS_NVDSLOGGER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSLOGGER))
#define GST_IS_NVDSLOGGER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSLOGGER))

#define MAX_NUM_SOURCES 1024

typedef struct _GstNvdslogger GstNvdslogger;
typedef struct _GstNvdsloggerClass GstNvdsloggerClass;

typedef struct
{
  gdouble curr_time;
  gdouble time_diff;
  gdouble fps_val;
  guint buffer_cnt;
  gdouble last_time;
  gboolean is_valid;
  gboolean eos;

} NvDsLoggerPerfStruct;

typedef struct
{
  gulong measurement_interval_ms;
  gulong perf_measurement_timeout_id;
  GMutex struct_lock;
  NvDsLoggerPerfStruct instance_str[MAX_NUM_SOURCES];
  gboolean show_fps;

} NvDsLoggerPerfStructInt;

/**
 * GstNvdslogger:
 *
 * Opaque #GstNvdslogger data structure
 */
struct _GstNvdslogger {
  GstBaseTransform 	 element;

  /*< private >*/
  GstClockID     clock_id;
  gboolean       flushing;
  gint 	 	 error_after;
  gfloat 	 drop_probability;
  gint		 datarate;
  guint 	 sleep_time;
  gboolean 	 silent;
  gboolean 	 dump;
  gboolean 	 sync;
  gboolean 	 check_imperfect_timestamp;
  gboolean 	 check_imperfect_offset;
  gboolean	 single_segment;
  GstBufferFlags drop_buffer_flags;
  GstClockTime   prev_timestamp;
  GstClockTime   prev_duration;
  guint64        prev_offset;
  guint64        prev_offset_end;
  gchar 	*last_message;
  guint64        offset;
  gboolean       signal_handoffs;
  GstClockTime   upstream_latency;
  GCond          blocked_cond;
  gboolean       blocked;
  GstClockTimeDiff  ts_offset;
  gboolean       drop_allocation;
  guint fps_measurement_interval_sec;
  guint latency;
  gboolean show_fps;
  gboolean show_latency;
  NvDsLoggerPerfStructInt *str;
  guint fnumber;
};

struct _GstNvdsloggerClass {
  GstBaseTransformClass parent_class;

  /* signals */
  void (*handoff) (GstElement *element, GstBuffer *buf);
};

G_GNUC_INTERNAL GType gst_nvdslogger_get_type (void);

G_END_DECLS

#endif /* __GST_NVDSLOGGER_H__ */
