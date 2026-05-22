/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVTIMESYNC_H__
#define __GST_NVTIMESYNC_H__

#include <gst/gst.h>
#include "nvstreammux.h"
#include "nvstreammux_batch.h"
#include "nvstreammux_pads.h"
#include <unordered_map>
#include <mutex>

class NvTimeSync : public ISynchronizeBuffer
{

    public:
    NvTimeSync(GstElement* el)
        : plugin(el), pipelineLatency(0), upstreamLatency(0),
          minFpsDuration(0),
          segments(), mutex()
    {}

    
    BUFFER_TS_STATUS get_synch_info(BufferWrapper* buffer);
    void removing_old_buffer(BufferWrapper* buffer);

    /**
     * @brief  Set the downstream latency
     *         Note: Currently the whole pipelineLatency value is
     *         used in timesynch logic to determine if a buffer is late
     *         at mux input
     *         This include the downstream latency.
     *         Note: This value shall be from the GST_EVENT_LATENCY
     *         sent by the sink plugin.
     *         The mux latency (currently not advertised) is taken care of
     *         by the TimeSynch library (using minFpsDuration)
     * @param  latency [IN] in nanoseconds
     */
    void SetPipelineLatency(GstClockTime latency);

    /**
     * @brief  Set the upstream latency
     * @param  latency [IN] in nanoseconds
     */
    void SetUpstreamLatency(GstClockTime latency);

    GstClockTime GetUpstreamLatency();

    GstClockTime GetCurrentRunningTime();

    void SetSegment(unsigned int stream_id, const GstSegment* segment);

    void SetOperatingMinFpsDuration(NanoSecondsType min_fps_dur);

    NanoSecondsType get_buffer_earlyby_time();

    
    uint64_t GetBufferRunningTime(uint64_t pts, unsigned int stream_id);

    private:
    GstElement* plugin;
    GstClockTime pipelineLatency;
    GstClockTime upstreamLatency;
    GstClockTime minFpsDuration;
    GstClockTime bufferWasEarlyByTime;
    std::unordered_map<unsigned int, GstSegment*> segments;
    std::mutex              mutex;
};


#endif
