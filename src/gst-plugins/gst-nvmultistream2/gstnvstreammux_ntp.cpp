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

#include "gstnvstreammux_ntp.h"
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvstreammux_ntp_debug);
#define GST_CAT_DEFAULT gst_nvstreammux_ntp_debug

#define DEFAULT_NTP_SYS_TIME_DIFF_IN_SEC 10

struct _GstNvDsNtpCalculator
{
  GstClockTime ntp_time_epoch_ns;
  GstClockTime ntp_frame_timestamp;

  GstClockTime ntp_time_epoch_ns_next;
  GstClockTime ntp_frame_timestamp_next;

  gboolean have_ntp_values;
  GstNvDsNtpCalculatorMode mode;
  GstNvDsNtpCorrectionMode correction_mode;
  GstElement *elem;
  guint source_id;

  GstClockTime prev_ntp_ts;
  GstClockTime prev_pts;
  gboolean bfirst;
  GstClockTime avg_frame_time;
};

GstNvDsNtpCalculator *
gst_nvds_ntp_calculator_new (GstNvDsNtpCalculatorMode mode,
        GstClockTime frame_duration, GstElement *elem, guint source_id)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvstreammux_ntp_debug, "nvstreammux_ntp", 0, "NvStreamMux NTP calculations");

  switch (mode) {
    case GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME:
    case GST_NVDS_NTP_CALC_MODE_RTCP:
        break;
    default:
        return NULL;
  }
  GstNvDsNtpCalculator *calc = g_new0 (GstNvDsNtpCalculator, 1);
  calc->mode = mode;
  calc->elem = elem;
  calc->source_id = source_id;

  if (calc->mode == GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME)
    calc->have_ntp_values = TRUE;

  /* Set NTP timestamp correction mode based on configured frame rate */
  if (GST_CLOCK_TIME_NONE == frame_duration) {
    calc->correction_mode = GST_NVDS_NTP_CORRECTION_DISABLED;
    GST_DEBUG_OBJECT (calc->elem, "Setting NTP correction mode disabled.");
  } else if (0 == frame_duration) {
    calc->correction_mode = GST_NVDS_NTP_CORRECTION_AUTOMATIC;
    GST_DEBUG_OBJECT (calc->elem, "Setting NTP correction mode automatic.");
  } else {
    calc->correction_mode = GST_NVDS_NTP_CORRECTION_USER_FRAME_RATE;
    calc->avg_frame_time = frame_duration;
    GST_DEBUG_OBJECT (calc->elem, "Setting NTP correction mode to app \
            provided frame rate based.");
  }

  return calc;
}

static void check_if_sys_rtcp_time_is_ntp_sync(GstNvDsNtpCalculator *calc, GstClockTime ntp)
{
  struct timeval sys_time;
  gettimeofday (&sys_time, NULL);
  GstClockTime sys_time_nsec = sys_time.tv_sec * GST_SECOND +
    sys_time.tv_usec * GST_USECOND;
  const gchar * ts_diff_str = g_getenv("ALLOWED_NTP_SYS_TIME_DIFF_IN_SEC");
  GstClockTime allowed_ts_diff = 0;

  if(ts_diff_str)
  {
    allowed_ts_diff =  atoi(ts_diff_str);
  }
  else
  {
    allowed_ts_diff = DEFAULT_NTP_SYS_TIME_DIFF_IN_SEC;
  }

  GstClockTime actual_diff = (ntp > sys_time_nsec) ? ntp - sys_time_nsec : sys_time_nsec - ntp;

  if (actual_diff > allowed_ts_diff * GST_SECOND)
  {
    gchar ntp_framets_str[NVDS_RFC3339_STR_BUF_LEN];
    gchar systs_framets_str[NVDS_RFC3339_STR_BUF_LEN];

    generate_rfc3339_str_from_ts (ntp_framets_str, ntp);
    generate_rfc3339_str_from_ts (systs_framets_str, sys_time_nsec);

    GST_WARNING_OBJECT (calc->elem, "Either host or Source %d seems to be out of NTP sync SYS TIME = %s CALCULATED NTP TIME = %s",
                calc->source_id, systs_framets_str, ntp_framets_str);
  }
}

void
gst_nvds_ntp_calculator_add_ntp_sync_values (GstNvDsNtpCalculator *calc,
    GstClockTime ntp_time_epoch_ns, GstClockTime ntp_frame_timestamp,
    GstClockTime avg_frame_time)
{
  gchar ntp_epoch_str[NVDS_RFC3339_STR_BUF_LEN];

  if (calc->mode == GST_NVDS_NTP_CALC_MODE_RTCP) {
    if (calc->have_ntp_values) {
      calc->ntp_time_epoch_ns_next = ntp_time_epoch_ns;
      calc->ntp_frame_timestamp_next = ntp_frame_timestamp;
    } else {
      calc->ntp_time_epoch_ns = ntp_time_epoch_ns;
      calc->ntp_frame_timestamp = ntp_frame_timestamp;
    }

    /* Don't update frame time if it is available from application */
    if (GST_NVDS_NTP_CORRECTION_USER_FRAME_RATE != calc->correction_mode) {
      calc->avg_frame_time = avg_frame_time;
    }
    calc->have_ntp_values = TRUE;

    generate_rfc3339_str_from_ts (ntp_epoch_str, ntp_time_epoch_ns);
    GST_INFO_OBJECT (calc->elem, "New NTP values received for source %d: "
        "ntp_time_epoch_ns = %s(%lu) ntp_frame_timestamp = %" GST_TIME_FORMAT
        "(%lu) avg_frame_rate = %.2f",
        calc->source_id, ntp_epoch_str, ntp_time_epoch_ns,
        GST_TIME_ARGS (ntp_frame_timestamp), ntp_frame_timestamp,
        avg_frame_time == 0 ? 0.0 : 1.0 * GST_SECOND / avg_frame_time);

    check_if_sys_rtcp_time_is_ntp_sync(calc, ntp_time_epoch_ns);
  }
}

/* Apply correction to NTP TS if needed.
 * 1. Check if new NTP < prev NTP. If yes, calculate new NTP as
 *    prev NTP + avg frame time
 * 2. Check if new Sender Report is consistent with prev SR. This is done
 *    by calculating NTP of current buffer with new SR. The difference of this
 *    NTP with prev calculated NTP should be < 1.1 * (current buffer's pts - prev pts).
 *    If this condition is not met, the new SR is ignored.
 */
static inline GstClockTime
apply_correction_if_needed_rtcp (GstNvDsNtpCalculator *calc, GstClockTime ntp_ts, GstClockTime buf_pts)
{
  if (calc->ntp_time_epoch_ns_next != 0) {
    GstClockTime ntpts_with_new_sr = calc->ntp_time_epoch_ns_next + buf_pts - calc->ntp_frame_timestamp_next;
    GstClockTime ntpdiff = ntpts_with_new_sr - calc->prev_ntp_ts;
    GstClockTime ptsdiff = buf_pts - calc->prev_pts;

    if (ntpdiff < gst_util_uint64_scale (ptsdiff, 11, 10) || calc->ntp_time_epoch_ns == 0) {
      ntp_ts = ntpts_with_new_sr;
      calc->ntp_time_epoch_ns = calc->ntp_time_epoch_ns_next;
      calc->ntp_frame_timestamp = calc->ntp_frame_timestamp_next;
      GST_DEBUG_OBJECT (calc->elem, "Using new NTP sync values for source %d", calc->source_id);
    } else {
      GST_WARNING_OBJECT (calc->elem, "Dropping inconsistent NTP sync values for source %d", calc->source_id);
    }
    calc->ntp_time_epoch_ns_next = 0;
    calc->ntp_frame_timestamp_next = 0;
  }

  if (ntp_ts == 0 || calc->bfirst == 0 || calc->prev_ntp_ts == 0)
    return ntp_ts;

  if(calc->prev_ntp_ts >= ntp_ts
      || ((ntp_ts - calc->prev_ntp_ts) < ((calc->avg_frame_time * 90) / 100))
    ) {
    if (GST_NVDS_NTP_CORRECTION_DISABLED == calc->correction_mode) {
      GST_DEBUG_OBJECT (calc->elem, "Frame rate based NTP timestamp correction is disabled.");
    } else {
      GST_DEBUG_OBJECT (calc->elem, "Frame rate based NTP timestamp correction applied.");
      ntp_ts = calc->prev_ntp_ts + calc->avg_frame_time;
    }
  }

  return ntp_ts;
}

GstClockTime
gst_nvds_ntp_calculator_get_buffer_ntp (GstNvDsNtpCalculator *calc,
        GstClockTime buf_pts)
{
  GstClockTime ntp = 0, ntp1 = 0;

  struct timeval sys_time;
  gettimeofday (&sys_time, NULL);
  GstClockTime sys_time_nsec = sys_time.tv_sec * GST_SECOND +
    sys_time.tv_usec * GST_USECOND;

  if (calc->mode == GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME) {
    struct timeval t1;
    gettimeofday(&t1, NULL);

    ntp =  t1.tv_sec * GST_SECOND + t1.tv_usec * GST_USECOND;
    if (G_UNLIKELY (_gst_debug_min >= GST_LEVEL_LOG)) {
      gchar ntp_framets_str[NVDS_RFC3339_STR_BUF_LEN];
      generate_rfc3339_str_from_ts (ntp_framets_str, ntp);

      GST_LOG_OBJECT (calc->elem, "Frame NTP calculated. mode: System Time. "
          "source %d: PTS:%" GST_TIME_FORMAT "(%lu) NTP: %s(%lu)",
          calc->source_id, GST_TIME_ARGS (buf_pts), buf_pts,
          ntp_framets_str, ntp);
    }
  } else {
    if ((calc->ntp_time_epoch_ns == 0) && (calc->ntp_time_epoch_ns_next == 0)) {
      if (G_UNLIKELY (_gst_debug_min >= GST_LEVEL_LOG)) {
        gchar ntp_epoch_str[NVDS_RFC3339_STR_BUF_LEN];
        gchar ntp_framets_str[NVDS_RFC3339_STR_BUF_LEN];
        gchar ntp_systs_str[NVDS_RFC3339_STR_BUF_LEN];

        generate_rfc3339_str_from_ts (ntp_epoch_str, calc->ntp_time_epoch_ns);
        generate_rfc3339_str_from_ts (ntp_framets_str, ntp);
        generate_rfc3339_str_from_ts (ntp_systs_str, sys_time_nsec);

        GST_LOG_OBJECT (calc->elem, "Cannot calculate frame NTP. mode: RTCP SR. "
            "source %d: PTS:%" GST_TIME_FORMAT "(%lu) NTP: %s(%lu). ntp_time_epoch_ns = %s(%lu) "
            "ntp_frame_timestamp = %" GST_TIME_FORMAT "(%lu) System Time: %s",
            calc->source_id, GST_TIME_ARGS (buf_pts), buf_pts, ntp_framets_str, ntp,
            ntp_epoch_str, calc->ntp_time_epoch_ns, GST_TIME_ARGS (calc->ntp_frame_timestamp),
            calc->ntp_frame_timestamp, ntp_systs_str);
      }
      goto no_ntp;
    }

    ntp1 = ntp = calc->ntp_time_epoch_ns - calc->ntp_frame_timestamp + buf_pts;
    ntp = apply_correction_if_needed_rtcp (calc, ntp, buf_pts);

    if (G_UNLIKELY (_gst_debug_min >= GST_LEVEL_INFO)) {
      gchar ntp_epoch_str[NVDS_RFC3339_STR_BUF_LEN];
      gchar ntp_framets_str[NVDS_RFC3339_STR_BUF_LEN];
      gchar ntp_systs_str[NVDS_RFC3339_STR_BUF_LEN];

      generate_rfc3339_str_from_ts (ntp_epoch_str, calc->ntp_time_epoch_ns);
      generate_rfc3339_str_from_ts (ntp_framets_str, ntp);
      generate_rfc3339_str_from_ts (ntp_systs_str, sys_time_nsec);

      /* NTP for the same frame with old and new SR might not be exactly same.
       * Allow a 100usec tolerance to distinguish between application of
       * correction and application of new SR. */
      if (ntp1 - ntp < 100 * GST_USECOND || ntp - ntp1 < 100 * GST_USECOND)  {
        GST_LOG_OBJECT (calc->elem, "Frame NTP calculated. mode: RTCP SR. "
            "source %d: PTS:%" GST_TIME_FORMAT "(%lu) NTP: %s(%lu). NTP diff:%ld. ntp_time_epoch_ns = %s(%lu) "
            "ntp_frame_timestamp = %" GST_TIME_FORMAT "(%lu) System Time: %s",
            calc->source_id, GST_TIME_ARGS (buf_pts), buf_pts, ntp_framets_str,
            ntp, ntp - calc->prev_ntp_ts,
            ntp_epoch_str, calc->ntp_time_epoch_ns,
            GST_TIME_ARGS (calc->ntp_frame_timestamp), calc->ntp_frame_timestamp,
            ntp_systs_str);
      } else {
        gchar ntp_framets_str_precorrection[NVDS_RFC3339_STR_BUF_LEN];
        generate_rfc3339_str_from_ts (ntp_framets_str_precorrection, ntp1);

        GST_INFO_OBJECT (calc->elem, "Frame NTP calculated. Correction required."
            "mode: RTCP SR. source %d: PTS:%" GST_TIME_FORMAT "(%lu) NTP: %s(%lu). NTP diff:%ld. Corrected ntp: %s(%lu). "
            "ntp_time_epoch_ns = %s(%lu) ntp_frame_timestamp = %" GST_TIME_FORMAT
            "(%lu) System Time: %s Avg Frame Time: %" GST_TIME_FORMAT
            " Avg. Frame Rate: %.2f",
            calc->source_id, GST_TIME_ARGS (buf_pts), buf_pts, ntp_framets_str_precorrection, ntp1,
            ntp1 - calc->prev_ntp_ts, ntp_framets_str, ntp,
            ntp_epoch_str, calc->ntp_time_epoch_ns,
            GST_TIME_ARGS (calc->ntp_frame_timestamp), calc->ntp_frame_timestamp,
            ntp_systs_str, GST_TIME_ARGS (calc->avg_frame_time),
            (calc->avg_frame_time == 0) ? 0.0 : 1.0 * GST_SECOND / calc->avg_frame_time);
      }
    }
  }

  if(G_UNLIKELY (_gst_debug_min >= GST_LEVEL_WARNING) && calc->prev_ntp_ts >= ntp) {
    gchar ntp_framets_str_prev[NVDS_RFC3339_STR_BUF_LEN];
    gchar ntp_framets_str_cur[NVDS_RFC3339_STR_BUF_LEN];

    generate_rfc3339_str_from_ts (ntp_framets_str_prev, calc->prev_ntp_ts);
    generate_rfc3339_str_from_ts (ntp_framets_str_cur, ntp);
    GST_WARNING_OBJECT (calc->elem, "Backward NTP. Prev: %s. Cur: %s source_id %d",
            ntp_framets_str_prev, ntp_framets_str_cur, calc->source_id);
  }

  if(G_UNLIKELY (_gst_debug_min >= GST_LEVEL_WARNING) && calc->avg_frame_time > 0
      && ntp >= calc->prev_ntp_ts + 2 * calc->avg_frame_time) {
    gchar ntp_framets_str_prev[NVDS_RFC3339_STR_BUF_LEN];
    gchar ntp_framets_str_cur[NVDS_RFC3339_STR_BUF_LEN];

    generate_rfc3339_str_from_ts (ntp_framets_str_prev, calc->prev_ntp_ts);
    generate_rfc3339_str_from_ts (ntp_framets_str_cur, ntp);
    GST_WARNING_OBJECT (calc->elem, "Forward jump in NTP. Prev: %s. Cur: %s source_id %d",
            ntp_framets_str_prev, ntp_framets_str_cur, calc->source_id);
  }

no_ntp:
  calc->prev_ntp_ts = ntp;
  calc->prev_pts = buf_pts;
  calc->bfirst = 1;

  return ntp;
}

gboolean
gst_nvds_ntp_calculator_have_ntp_sync_values (GstNvDsNtpCalculator *calc)
{
  if (calc->mode == GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME)
    return TRUE;

  if (calc->mode == GST_NVDS_NTP_CALC_MODE_RTCP)
    return calc->have_ntp_values;

  return FALSE;
}

void gst_nvds_ntp_calculator_get_ntp_sync_values (GstNvDsNtpCalculator *calc,
        GstClockTime *ntp_time_epoch_ns, GstClockTime *ntp_frame_timestamp,
        GstClockTime *avg_frame_time, GstClockTime *ntp_time_epoch_ns_next,
        GstClockTime *ntp_frame_timestamp_next)
{
  *ntp_time_epoch_ns = calc->ntp_time_epoch_ns;
  *ntp_frame_timestamp = calc->ntp_frame_timestamp;
  *ntp_time_epoch_ns_next = calc->ntp_time_epoch_ns_next;
  *ntp_frame_timestamp_next = calc->ntp_frame_timestamp_next;
  *avg_frame_time = calc->avg_frame_time;
}

void
gst_nvds_ntp_calculator_reset (GstNvDsNtpCalculator *calc)
{
  calc->ntp_time_epoch_ns = 0;
  calc->ntp_frame_timestamp = 0;
  calc->ntp_time_epoch_ns_next = 0;
  calc->ntp_frame_timestamp_next = 0;
  calc->prev_ntp_ts = 0;
  calc->prev_pts = 0;
  calc->bfirst = 0;
  calc->avg_frame_time = 0;
  if (calc->mode == GST_NVDS_NTP_CALC_MODE_RTCP)
    calc->have_ntp_values = FALSE;
  GST_INFO_OBJECT (calc->elem, "Reset NTP calculations for source %d",
      calc->source_id);
}

void
gst_nvds_ntp_calculator_free (GstNvDsNtpCalculator *calc)
{
  g_free (calc);
}

void
generate_rfc3339_str_from_ts (gchar *buf, GstClockTime ts)
{
  time_t tloc;
  struct tm tm_log;
  int ms;

  /** ts itself is UTC Time in ns */
  struct timespec timespec_current;
  GST_TIME_TO_TIMESPEC (ts, timespec_current);
  memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
  ms = timespec_current.tv_nsec / 1000000;

  gmtime_r (&tloc, &tm_log);
  buf[NVDS_RFC3339_STR_BUF_LEN - 1] = '\0';
  strftime (buf, NVDS_RFC3339_STR_BUF_LEN - 1, "%Y-%m-%dT%H:%M:%S", &tm_log);
  snprintf (buf + strlen (buf), NVDS_RFC3339_STR_BUF_LEN - strlen(buf) - 1,
      ".%.3dZ", ms);
}

