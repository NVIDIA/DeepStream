/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __NVSTREAMMUX_H__
#define __NVSTREAMMUX_H__

#include <chrono>
#include <unordered_map>
#include <map>
#include <list>
#include "nvbufsurface.h"
#include "nvstreammux_batch.h"
#include "nvstreammux_debug.h"

//G_BEGIN_DECLS

typedef struct _GstNvStreamMux GstNvStreamMux;
typedef struct _GstNvStreamMuxClass GstNvStreamMuxClass;

/*
 * @brief low level helper for core logic, this is agnostic of gstreamer
 * or any other wrapper on higher level
 */

class NvStreamMux
{


    public :

        NvStreamMux(SourcePad * srcpad, INvStreammuxDebug* a_debug_iface=nullptr) : inputs(), src_pad(srcpad), cv(), cv_throttle_wait(), mutex(), mutex_throttle_wait()
        {
            batch = NULL;

            stop_task = false;
            num_queues_empty = 0;
            num_sources_idle = 0;
            pads_got_eos_and_empty_in_q = 0;
            num_pads_eos = 0;
            cur_frame_pts = 0;
            state = SOURCE_STATE_IDLE;
            num_surfaces_per_frame = 1;
            all_pads_eos = false;
            extra_throttle_wait_time = std::chrono::microseconds(0);
            got_first_buffer = false;
            synch_buffer = nullptr;
            debug_iface = a_debug_iface;
        }

	/*
         * @brief functions to maintain internal variables based on config
         */
        void set_policy(BatchPolicyConfig cfg)
        {
            batch_policy = BatchPolicy(cfg, &inputs, debug_iface);
            batch_policy.total_buf_available = 0;
            batch_policy.set_synch_buffer_iface(synch_buffer);
        }
        void set_frame_num_reset_on_eos(bool reset)
        {
            frame_num_reset_on_eos = reset;
        }
        void set_frame_num_reset_on_stream_reset(bool reset)
        {
            frame_num_reset_on_stream_reset = reset;
        }

        /*
         * @brief acquire lock on mux
         */
        virtual void lock()
        {
            mutex.lock();
        };

        /*
         * @brief release lock on mux
         */
        virtual void unlock()
        {
            mutex.unlock();
        };

        /*
         * @brief release lock on mux
         */
        virtual void wait(){};

        /*
         * @brief release lock on mux
         */
        virtual void notify_all();

        /*
         * @brief get_pad for specified pad id
         * @param pad_id, id of pad to lookup
         * @return pointer to SinkPad object for the pad_id
         */
        SinkPad * get_pad(unsigned int pad_id);

        /*
         * @brief add_pad for specified pad id
         * @param pad_id, id of pad to lookup
         * @return pointer to SinkPad object for the pad_id
         */
        void add_pad(unsigned int id, SinkPad * pad);

        /*
         * @brief 
         * @param pad_id, id of pad to lookup
         * @param  mime_type [IN]
         * @return NA
         */
        void update_pad_mimetype(unsigned int id, PAD_MIME_TYPE mime_type);

        /*
         * @brief 
         * @param  id [IN] id of pad to lookup
         * @return The configured mime type for the sinkpad at pad_id==id
         */
        PAD_MIME_TYPE get_pad_mimetype(unsigned int id);

        /*
         * @brief get_pad for specified pad id
         * @param pad_id, id of pad to lookup
         * @return pointer to SinkPad object for the pad_id
         */
        void remove_pad(unsigned int id);

        /*
         * @brief Construct the NvStreamMux from config, this would typically
         * come from a high level config file
         */
        //virtual  update_source(SINK_EVENT event, unsigned long source_id);


        /*
         * @brief handle the eos event on pad
         * @param event [IN] sink event
         * @param source_id[IN] id of source, in deepstream app pad_id
         * of gstreamer
         * source pad is same as this id
         * @param EventWrapper [IN] event coming on sink pad
         *
         */
        virtual bool  handle_eos(SINK_EVENT et, unsigned int source_id, EventWrapper * event);

	/*
         * @brief handle the eos event on pad for cascaded muxers
         * @param event [IN] sink event
         * @param source_id[IN] id of source, in deepstream app pad_id
         * of gstreamer
         * source pad is same as this id
         * @param EventWrapper [IN] event coming on sink pad
         *
         */
        virtual bool  handle_eos_cascaded(SINK_EVENT et, unsigned int source_id, EventWrapper * event);
        /*
         * @brief handle the segment event on pad
         * @param event [IN] sink event
         * @param source_id[IN] id of source, in deepstream app pad_id
         * of gstreamer
         * source pad is same as this id
         * @param EventWrapper [IN] event coming on sink pad
         *
         */
        virtual bool  handle_segment(SINK_EVENT et, unsigned int source_id, EventWrapper * event);

        /*
         * @brief handle the GST_NVEVENT_STREAM_RESET event on pad
         * @param event [IN] sink event
         * @param source_id[IN] id of source, in deepstream app pad_id
         * of gstreamer
         * source pad is same as this id
         * @param EventWrapper [IN] event coming on sink pad
         *
         */
        virtual bool  handle_stream_reset(SINK_EVENT et, unsigned int source_id, EventWrapper * event);

        /*
         * @brief handle the FLUSH_STOP event on pad
         * @param event [IN] sink event
         * @param source_id[IN] id of source, in deepstream app pad_id
         * of gstreamer
         * source pad is same as this id
         * @param EventWrapper [IN] event coming on sink pad
         *
         */
        virtual bool  handle_flush_stop(SINK_EVENT et, unsigned int source_id, EventWrapper * event);

        /*
         * @brief add sink pad
         * @param pad [IN] the pad to be added
         */
        virtual void add_sink(SinkPad  * pad){};

        /*
         * @brief remove sink pad
         * @param pad [IN] the pad to be added
         */
        virtual void remove_sink(SinkPad pad){};

        /*
         * @brief add buffer to sink pad
         * @param pad [IN] the pad to be added
         * @param buffer [IN] the buffer to be added
         */
        //virtual  void add_buffer(SinkPad * pad, BufferWrapper * buffer);
        virtual  void add_buffer(unsigned int pad_id, BufferWrapper * buffer);

        /*
         * @brief task for processing queued buffers and events,
         * according to batching configuration
         * @param pad [IN] the pad to be added
         * @param buffer [IN] the buffer to be added
         */
        bool   push_loop(NvDsBatchBufferWrapper * out_buf, NvDsBatchMeta *);

        /*
         * @brief function to push the buffer out of mux to designated SourcePad
         * @param out_buf [IN] the batch to be pushed
         * @param src_pad  [IN] the to push to
         * @return true on successfully pushing the batch buffer on src_pad;
         *         false otherwise
         */
        bool    push_batch(NvDsBatchBufferWrapper * out_buf, SourcePad * src_pad);

        /*
         * @brief function to push events out of mux
         */

        void    push_events();


        /*
         * @brief function to copy the individual buffers and metadata to batch
         * @param out_buf [IN] the batch buffer
         * @param buffer [IN] the metadata
         * @return the total number of buffers copied
         */
        unsigned int    copy_batch(NvDsBatchBufferWrapper * out_buf, NvDsBatchMeta *);

	/*
         * @brief functions to calculate play duration, called while processing 
	 * relevant gstreamer events
         * @return void
         */
        void    handle_pause_play();
        void    handle_play_pause();
        void    handle_ready_pause();

	/*
         * @brief functions called to process gstreamer event 
	 * GST_STATE_CHANGE_READY_TO_PAUSED
         * @return void
         */
        void    handle_stop();
        void    reset_stop();


	/*
         * @brief function to set value of frame_duration_nsec
         * @param offset [IN] value to be set
         */
        void    set_frame_duration(unsigned long);

	/*
         * @brief function to set value of cur_frame_pts
         * @param offset [IN] value to be set
         */
	void    set_pts_offset(gulong offset);

	/*
         * @brief function to get current state of PAD
         * @param sinkPad [IN] SinkPad Pointer
         * @param SOURCE_STATE [OUT] State of sinkpad
         */
	SOURCE_STATE get_pad_state(SinkPad* sinkPad);

        /**
         * @brief  set batch_policy.set_batch_size()
         * @param  batch-size [IN]
         */
        void set_batch_size(unsigned int size);

        /**
         * @brief  set batch_policy.set_batch_push_timeout()
         * @param  batch-size [IN]
         */
        void set_batch_push_timeout(unsigned int timeout);

        /**
         * @brief  return batch_policy.get_batch_size()
         * @return the batch-size
         */
        unsigned int get_batch_size();
       
        /**
         * @brief  return batch_policy.get_config_batch_size()
         * @return the batch-size
         */
        unsigned int get_config_batch_size();

        /**
         * @brief  Set the class var num_surfaces_per_frame
         *         and batch_policy.set_num_surfaces()
         * @param  num [IN] Number of surfaces per frame
         */
        void set_num_surfaces_per_frame(unsigned int num);

        /**
         * @brief  Get the class var num_surfaces_per_frame
         * @return the class var num_surfaces_per_frame
         */
        unsigned int get_num_surfaces_per_frame();
       
        /**
         * @brief  Set all_pads_eos
         * @param  eos [IN]
         */
        void set_all_pads_eos(bool eos);

        /**
         * @brief  Reset a pad for streaming
         * @param  sinkPad to reset for streaming-restart after EOS
         */
        void reset_pad(SinkPad* pad);

        /**
         * @brief  Get all_pads_eos
         * @return current all_pads_eos
         */
        bool get_all_pads_eos();

        /**
         * @brief  Set the user interface for buffer synchronization
         *         Note: This API shall be called before set_policy()
         * @param  synch_buffer_iface [IN]
         */
        void set_synch_buffer_iface(ISynchronizeBuffer* synch_buffer_iface);

        /**
         * @brief  Get the Batch Policy minimum fps duration
         *         calculated from min overall fps config
         */
        NanoSecondsType get_min_fps_duration();

        unsigned int get_source_id_with_earliest_buffer();

        /**
         * @brief  Get the remaining unbatched ready buffers from
         *         all sources. Shall be called after calling
         *         Batch::form_batch
         */
        unsigned int get_remaining_unbatched_buffers_from_all_sources();

        /**
         * @brief  Apply the throttle delay and wait for the time (max-fps cfg)
         *         or until a new buffer come in
         * @param  stop_when_input_buffer; if true, the API wait for the time
         *         (max-fps cfg) or until a new buffer come in;
         *         if false, the API wait for the time (max-fps cfg)
         */
        void apply_throttle(bool stop_when_input_buffer=false);

    private :
        std::unordered_map<unsigned int, SinkPad *> inputs;
        SourcePad * src_pad;
        SOURCE_STATE state;
        unsigned int num_queues_empty;
        unsigned int num_sources_idle;
        unsigned int pads_got_eos_and_empty_in_q;
        unsigned int num_pads_eos;
        unsigned int num_surfaces;
        std::condition_variable cv;
        std::condition_variable cv_throttle_wait;
        /** mutex to protect critical section code
         * using the unordered_map inputs, sources, etc*/
        std::mutex              mutex;
        std::mutex              mutex_throttle_wait;
        Batch * batch;
        /*
         * @brief map of sources for a priority
         */
        std::multimap<int, int> sources;
        TimePoint current_play_start_time;
        TimePoint last_pause_time;
        NanoSecondsType accum_play_dur;
        unsigned long frame_duration_nsec;
        unsigned long cur_frame_pts;
        bool stop_task;
        NanoSecondsType extra_throttle_wait_time;

        BatchPolicy  batch_policy;
        unsigned int num_surfaces_per_frame;
        /** Whether to send EOS on all pads */
        bool all_pads_eos;
        bool got_first_buffer;
        bool frame_num_reset_on_eos = false;
        bool frame_num_reset_on_stream_reset = false;
        INvStreammuxDebug* debug_iface;

    public:
        ISynchronizeBuffer* synch_buffer;
};

//G_END_DECLS
#endif
