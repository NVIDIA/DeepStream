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

#ifndef __GST_NVSTREAMMUX_IMPL_H__
#define __GST_NVSTREAMMUX_IMPL_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include "nvstreammux_pads.h"
#include "nvstreammux_batch.h"
#include "gstnvstreammux.h"
#include "gstnvstreampad.h"
#include "nvds_latency_meta.h"
#include "nvds_latency_meta_internal.h"
#include "gstnvstreammuxdebug.h"

extern "C"
{
static void
mem_buf_unref_callback (gpointer data);
}

class GstCommonBufferAPI
{

    public:
    GstCommonBufferAPI(GstElement* a_mux) : mux(a_mux) {}

    void add_component_latency_metadata(std::shared_ptr<BufferWrapper> src_buffer, GstBuffer * gst_batch_buffer, NvDsBatchMeta* batch_meta, unsigned int source_id, unsigned int pad_index, unsigned int frame_number);
    void update_component_latency_metadata(NvDsBatchMeta* dest_batch_meta);

    private:
    GstElement* mux;
};

class GstBufferWrapper : public BufferWrapper
{

    public :
        GstBufferWrapper(void * buffer, ENTRY_TYPE et, BATCH_SEQUENCE_TYPE bt, GstClockTime a_ntp_ts, GstClockTime  a_buf_pts, GstClockTime a_duration, unsigned int id=0) : BufferWrapper(buffer, et, bt, (uint64_t)a_buf_pts)
        {
            /** TODO: Raise an exception if raw is nullptr;
             * no_throw shall work whoever creates an object of GstBufferWrapper */
            unwrap();
	        LOGD("GstBufferWrapper constructor %p gst_buffer refcount %d\n", this, ((GstBuffer *) (wrapped))->mini_object.refcount);
            ntp_ts = a_ntp_ts;
            buf_pts = a_buf_pts;
            duration = a_duration;
            stream_id = id;
            buffer_wrapper_creation_time = 0.0;
            if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
            {
                  buffer_wrapper_creation_time = nvds_get_current_system_timestamp();
            }
            is_nvmm = false;
        }

        ~GstBufferWrapper()
        {
            free();
        }

        void * unwrap()
        {
             void * ret = NULL;

             GstMapInfo info = {0}; // TBD FIXME = GST_MAP_INFO_INIT;

             if (gst_buffer_map ((GstBuffer *)wrapped, &info, GST_MAP_READ))
             {
                 raw = ret = info.data;
                 rawSize = info.size;
                 gst_buffer_unmap ((GstBuffer * )wrapped, &info);

             }
             return ret;

        }
        void free()
        {
	        LOGD("GstBufferWrapper destructor %p raw %p gst_buf refcount %d\n", this, raw, ((GstBuffer *) (wrapped))->mini_object.refcount);
            gst_buffer_unref(GST_BUFFER (wrapped));
        }

        void SetAudioParams(NvBufAudioParams aAudioParams)
        {
            audioParams = aAudioParams;
        }

        void SetMemTypeNVMM(bool isNVMM)
        {
            is_nvmm = isNVMM;
        }

        bool IsMemTypeNVMM()
        {
            return is_nvmm;
        }

        NvBufAudioParams audioParams;

        GstClockTime ntp_ts;
        GstClockTime buf_pts;
        GstClockTime duration;
        unsigned int stream_id;
        gdouble buffer_wrapper_creation_time;
        bool is_nvmm;
};

/** TODO re-design GstBatchBufferWrapper to be video specific and NvDsBatchBufferWrapper generic ;
 * Also rename the new data-structures according to the media they contain */

/**
 * @brief  The GStreamer wrapper code for NvDsBatchBufferWrapper
 *         which represent one batched buffer.
 *         NOTE: None of the APIs in this class are thread-safe
 */
class GstBatchBufferWrapper : public NvDsBatchBufferWrapper
{

  public :
    GstBatchBufferWrapper(GstNvStreamMux * mux, unsigned int size, bool is_raw) : NvDsBatchBufferWrapper(size) , is_raw(is_raw), mux(mux), api((GstElement*)mux)
    {
        gst_buffer = nullptr;
        batch = malloc(sizeof(NvBufSurface));
        memset (batch, 0 , sizeof(NvBufSurface));
        ((NvBufSurface *)batch)->surfaceList = (NvBufSurfaceParams *)malloc(sizeof(NvBufSurfaceParams) * size);
        ((NvBufSurface *)batch)->numFilled = 0;
        ((NvBufSurface *)batch)->batchSize = size;
        ((NvBufSurface *)batch)->memType = NVBUF_MEM_DEFAULT;
        if(!is_raw)
        {
            gst_buffer =
                gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY, (NvBufSurface *)batch, sizeof (NvBufSurface),
                0, sizeof (NvBufSurface), (void *)this,  mem_buf_unref_callback);
            //gst_buffer_ref(gst_buffer);
        }
        else
        {
            raw_batch = gst_buffer_list_new_sized (size);
        }
    }

    void unref() override;
    void reset_batch();
    void copy_meta(unsigned int id, std::shared_ptr<BufferWrapper> src_buffer, unsigned int batch_id, unsigned int frame_number,  unsigned int num_surfaces_per_frame, NvDsBatchMeta * dest_batch_meta, unsigned int source_id) override;
    unsigned int copy_buf(std::shared_ptr<BufferWrapper> src, unsigned int pos) override;
    bool push(SourcePad * src_pad, TimePoint current_play_start, NanoSecondsType accum_dur) override;
    bool push(SourcePad * src_pad, unsigned long pts) override;
    void unref_gst_bufs();

    GstBufferList *raw_batch;
    GstBuffer * gst_buffer;
    bool is_raw;
    GstNvStreamMux * mux;
    std::vector<std::shared_ptr<GstBufferWrapper> > gst_in_bufs;

    private:
    unsigned int copy_buf_impl(std::shared_ptr<BufferWrapper> buf, unsigned int pos);
    GstCommonBufferAPI api;
};

extern "C"
{
static void
mem_buf_unref_callback (gpointer data)
{

    if(data != NULL)
    {
        GstBatchBufferWrapper * batch = (GstBatchBufferWrapper *)data;
        batch->unref_gst_bufs();
        delete batch;
    }

}
}

class GstSinkPad : public SinkPad
{
    public :
        GstSinkPad(GstNvStreamMux * elem, unsigned int id, GstPad * pad) : SinkPad(id, (void *) pad), element(elem)
        {
            new_add = true;
            ntp_calc = nullptr;
	    cascaded_eos = false;
        }
        ~GstSinkPad()
        {
            if(ntp_calc)
            {
                gst_nvds_ntp_calculator_free (ntp_calc);
            }
        }
        GstVideoInfo vid_info;
        void push_event(SourcePad * src_pad, QueueEntry *);
        friend class GstSourcePad;

        GstNvDsNtpCalculator* get_ntp_calc(GstNvDsNtpCalculatorMode mode,
                GstClockTime frame_duration)
        {
            if(!ntp_calc)
            {
                ntp_calc =
                gst_nvds_ntp_calculator_new (mode, frame_duration,
                        GST_ELEMENT (element), id);
            }
            return ntp_calc;
        }

    private :
        //GstPad * wrapped;
        GstNvStreamMux * element;
        bool new_add;
        bool cascaded_eos;
        GstNvDsNtpCalculator *ntp_calc;
};

class GstSourcePad : public SourcePad
{
    public :
        GstSourcePad(GstNvStreamMux * elem, GstPad * pad, unsigned int id) : SourcePad(id, (void *) pad), element(elem)
        {

        }
        ~GstSourcePad();
        GstVideoInfo vid_info;
        friend class GstSinkPad;
    private :
        //GstPad * wrapped;
        GstNvStreamMux * element;
};
#endif
