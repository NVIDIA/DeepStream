/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gstnvtimesynch.h"
#include "gstnvstreammux_impl.h"
#include "gstnvstreammuxdebug.h"

#define PTS_TO_RUNNING_TIME(pts, stream_id) \
    segments[stream_id] ? gst_segment_to_running_time(segments[stream_id], GST_FORMAT_TIME, pts) : pts

static GstClockTime
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

GstClockTime NvTimeSync::GetCurrentRunningTime ()
{
  GstClockTime base_time, clock_time;

  g_return_val_if_fail (GST_IS_ELEMENT (plugin), GST_CLOCK_TIME_NONE);

  base_time = gst_element_get_base_time (plugin);

  if (!GST_CLOCK_TIME_IS_VALID (base_time)) {
    GST_DEBUG_OBJECT (plugin, "Could not determine base time");
    return GST_CLOCK_TIME_NONE;
  }

  clock_time = gst_get_current_clock_time (plugin);

  if (!GST_CLOCK_TIME_IS_VALID (clock_time)) {
    return GST_CLOCK_TIME_NONE;
  }

  if (clock_time < base_time) {
    GST_DEBUG_OBJECT (plugin, "Got negative current running time");
    return GST_CLOCK_TIME_NONE;
  }

  /* To make live-source=1/sync=1 work for non-live sources. */
  if ((base_time == 0) && (plugin->current_state != GST_STATE_PLAYING)) {
    return 0;
  }

  return clock_time - base_time;
}

void NvTimeSync::SetSegment(unsigned int stream_id, const GstSegment* segment)
{
    std::unique_lock<std::mutex> lck(mutex);
    GstSegment* seg = NULL;

    auto it = segments.find(stream_id);
    if(it != segments.end())
    {
        seg = it->second;
    }

    if(seg)
    {
        gst_segment_free(seg);
    }

    seg = gst_segment_new();

    gst_segment_copy_into (segment, seg);

    segments[stream_id] = seg;
}

void NvTimeSync::SetPipelineLatency(GstClockTime latency)
{
    pipelineLatency = latency;
}

void NvTimeSync::SetUpstreamLatency(GstClockTime latency)
{
    GST_DEBUG_OBJECT (plugin,
      "nvstreammux max-latency and upstream latency = %lu\n", latency);
    upstreamLatency = latency;
}

void NvTimeSync::SetOperatingMinFpsDuration(NanoSecondsType min_fps_dur)
{
    minFpsDuration = (GstClockTime)min_fps_dur.count();
}

uint64_t NvTimeSync::GetBufferRunningTime(uint64_t pts, unsigned int stream_id)
{
    std::unique_lock<std::mutex> lck(mutex);
    return PTS_TO_RUNNING_TIME(pts, stream_id);
}

BUFFER_TS_STATUS NvTimeSync::get_synch_info(BufferWrapper* buffer)
{
    std::unique_lock<std::mutex> lck(mutex);
    GstBufferWrapper* gst_buffer = (GstBufferWrapper*)buffer;
    GstClockTime buffer_running_time = 
        PTS_TO_RUNNING_TIME(GST_BUFFER_PTS((GstBuffer*)(gst_buffer->wrapped)), gst_buffer->stream_id);
    GstClockTime current_running_time = GetCurrentRunningTime();
    //LOGD("minFpsDuration=%lu pipelineLatency=%lu\n", minFpsDuration, pipelineLatency);

    /** invalidate the older early buffer timing in this call for a new buffer */
    bufferWasEarlyByTime = 0;

    if(current_running_time == GST_CLOCK_TIME_NONE || current_running_time < minFpsDuration)
    {
        //pipeline is just starting up
        return BUFFER_TS_ONTIME;
    }

    if(buffer_running_time > (current_running_time - minFpsDuration))
    {
        /** early */
        /** Note: Not using pipelineLatency to confirm early buffers
         * as it shall be used only to confirm late buffers
         */
        //LOGD("early\n");
        if(buffer_running_time > current_running_time)
        {
          bufferWasEarlyByTime = (buffer_running_time - current_running_time);
        }
        return BUFFER_TS_EARLY;
    }
    else if(buffer_running_time + upstreamLatency < (current_running_time - minFpsDuration))
    {
        /** Note: in this LATE buffer decision logic, we use upstreamLatency
         * and not pipelineLatency which is = upstreamLatency + downstreamLatency
         * We shall avoid using downstreamLatency to determine if the
         * buffer was late at muxer's sink pad. 
         */
        /** late */
        LOGD("late buffer_running_time=%lu current_running_time=%lu diff_in_ms=%lu\n", buffer_running_time, current_running_time,
                (current_running_time - buffer_running_time) / 1000000);
        return BUFFER_TS_LATE;
    }
    else
    {
        /** on time */
        //LOGD("on-time\n");
        return BUFFER_TS_ONTIME;
    }
    
}

void NvTimeSync::removing_old_buffer(BufferWrapper* buffer)
{
    /** send signal to the application conveying buffer drop */
    LOGD("dropping buffer\n");
}

NanoSecondsType NvTimeSync::get_buffer_earlyby_time()
{
   return NanoSecondsType(bufferWasEarlyByTime);
}

GstClockTime NvTimeSync::GetUpstreamLatency()
{
   return upstreamLatency;
}
