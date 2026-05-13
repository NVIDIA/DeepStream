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

#include "nvstreammux.h"
#include "nvstreammux_batch.h"
#include "nvstreammux_pads.h"
#include "MuxConfigParser.h"
#include <thread>

#if 1
#define debug_print(fmt, ...) \
            do { if(debug_iface) { debug_iface->DebugPrint("[%s %d]" fmt, __func__, __LINE__, ## __VA_ARGS__); } } while (0)
#else
#define debug_print(fmt, ...) printf("[DEBUG %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)
#endif


void NvStreamMux::push_events()
{
    std::unique_lock<std::mutex> lck(mutex);
    for (std::unordered_map<unsigned int, SinkPad *>::iterator source_it = inputs.begin(); source_it != inputs.end(); ++source_it)
    {
        SinkPad* sinkPad = source_it->second;
        if(!source_it->second)
        {
          continue;
        }
        std::unique_lock<std::mutex> lck(sinkPad->mutex);
        sinkPad->push_events(src_pad);
        if((sinkPad->queue.size() == 0) && (state != SOURCE_STATE_IDLE))
        {
            num_queues_empty++;
        }

        /** add_pad(++),
         * add_buffer(--), remove_pad(++), handle_eos(++), handle_flush_stop(--)
         */
        /** Shall go to idle only when no more buffers */
        if(sinkPad->get_switched_to_idle() && sinkPad->queue.size() == 0)
        {
            sinkPad->state = SOURCE_STATE_IDLE;
            num_sources_idle++;
            debug_print("%s %d num_sources_idle=%d\n", __func__, __LINE__, num_sources_idle);
            sinkPad->set_switched_to_idle(false);
	    if(sinkPad->get_eos())
            {
                pads_got_eos_and_empty_in_q++;
                batch_policy.update_eos_sources(batch_policy.get_eos_sources() + 1);
                debug_print("frame_num_reset_on_eos=%d\n", frame_num_reset_on_eos);
                if (frame_num_reset_on_eos)
                    sinkPad->reset_frame_count();
            }
        }
        if(sinkPad->get_switched_to_active())
        {
            if(num_sources_idle > 0)
            {
                /** when all buffers are added; then EOS and then this push_loop, we may miss to count idle
                 * and thus num_sources_idle can be ==0 */
                num_sources_idle--;
            }
            debug_print("%s %d num_sources_idle=%d\n", __func__, __LINE__, num_sources_idle);
            sinkPad->set_switched_to_active(false);
        }
        debug_print("sink=%d get_switched_to_idle()=%d queue.size()=%lu get_eos()=%d\n",
                sinkPad->id, sinkPad->get_switched_to_idle(), sinkPad->queue.size(),
                sinkPad->get_eos());
    }

    /** Pushed all events including post batching events;
     * Now is the time to update_idle_sources  */
    debug_print("%s num_sources_idle=%d pads_got_eos_and_empty_in_q=%u(%lu)\n", __func__, num_sources_idle, pads_got_eos_and_empty_in_q, inputs.size());
    /** update all_pads_eos if inputs.size != 0 */
    debug_print("DEBUGME %s %d size=%ld pads_got_eos_and_empty_in_q=%d\n", __func__, __LINE__, inputs.size(), pads_got_eos_and_empty_in_q);
    if(inputs.size() != 0)
    {
      all_pads_eos = (pads_got_eos_and_empty_in_q == inputs.size());
    }
    batch_policy.update_idle_sources(num_sources_idle);
}

unsigned int NvStreamMux::get_source_id_with_earliest_buffer()
{
    uint64_t timestamp = -1;
    unsigned int id = -1;
    for(std::unordered_map<int, int>::iterator it = batch->num_sources.begin(); it != batch->num_sources.end(); ++it)
    {
        if(!inputs[it->first])
            continue;

        std::vector<std::shared_ptr<QueueEntry> >::iterator it_s = inputs[it->first]->queue.begin();
        /** need to check the first buffer in queue (FIFO) */
        if((it_s != inputs[it->first]->queue.end() && (unsigned int)it->second > 0)
           && (timestamp > std::static_pointer_cast<BufferWrapper>(*it_s)->timestamp))
        {
            timestamp = std::static_pointer_cast<BufferWrapper>(*it_s)->timestamp;
            id = it->first;
        }
    }
    return id;
}

unsigned int NvStreamMux::get_remaining_unbatched_buffers_from_all_sources()
{
    unsigned int num_buffers = 0;
    for(std::unordered_map<int, int>::iterator it = batch->num_sources.begin(); it != batch->num_sources.end(); ++it)
    {
        if(!inputs[it->first])
            continue;

        std::vector<std::shared_ptr<QueueEntry> >::iterator it_s = inputs[it->first]->queue.begin();
        while((it_s != inputs[it->first]->queue.end() && (unsigned int)it->second > 0))
        {
            if(std::static_pointer_cast<QueueEntry>(*it_s)->type == ENTRY_BUFFER) {
              num_buffers++;
            }
            it_s++;
        }
    }
    return num_buffers;
}

unsigned int NvStreamMux::copy_batch(NvDsBatchBufferWrapper * out_buf, NvDsBatchMeta * dest_batch_meta)
{
    unsigned int num_sources_copied = 0;
    unsigned int total_copied = 0;
    debug_print("DEBUGME\n");
    for(std::unordered_map<int, int>::iterator it = batch->num_sources.begin(); it != batch->num_sources.end(); ++it)
    {
    debug_print("DEBUGME\n");
         num_sources_copied = 0;

         if(!inputs[it->first])
         {
             continue;
         }

         std::unique_lock<std::mutex> lck(inputs[it->first]->mutex);
         for(std::vector<std::shared_ptr<QueueEntry> >::iterator it_s = inputs[it->first]->queue.begin();
             (it_s != inputs[it->first]->queue.end() && ((int)num_sources_copied < it->second)) && ((unsigned int)batch->num_sources[it->first] > 0);
             )
         {
              debug_print("DEBUGME; num_sources=%u\n", (unsigned int)batch->num_sources[it->first]);
              /** skip events */
              if(std::static_pointer_cast<QueueEntry>(*it_s)->type != ENTRY_BUFFER)
              {
                debug_print("%s %d event in queue ignored\n", __func__, __LINE__);
                it_s++;
                continue;
              }
    debug_print("DEBUGME\n");
              /** copy one or more surfaces from it_s to out_buf: */
              unsigned int num_surfaces_copied = out_buf->copy_buf(std::static_pointer_cast<BufferWrapper>(*it_s), total_copied);
              if(num_surfaces_copied == 0)
              {
                /** This batch is full; break */
                break;
              }
              if(dest_batch_meta)
              {
                out_buf->copy_meta(inputs[it->first]->id, std::static_pointer_cast<BufferWrapper>(*it_s), total_copied, inputs[it->first]->get_frame_count(), num_surfaces_per_frame, dest_batch_meta, inputs[it->first]->source_id);
              }
              total_copied += num_surfaces_copied;
              inputs[it->first]->update_frame_count(num_surfaces_copied);
              num_sources_copied++;

              /** pop the buffer from pad's queue */
              //delete(*it_s);
              it_s = inputs[it->first]->queue.erase(it_s);
              inputs[it->first]->pop_buffer_done();
              batch_policy.total_buf_available--;

              if(inputs[it->first]->queue.size() == 0)
              {
                  num_queues_empty++;
              }


              if(num_sources_copied == (unsigned int)batch->num_sources[it->first])
              {
                  batch_policy.update_push_stats(inputs[it->first]->id, num_sources_copied);
                  break;
              }
         }
         inputs[it->first]->adjust_event_indices(num_sources_copied, false);
    }
    debug_print("DEBUGME total_copied=%d\n", total_copied);
    return total_copied;
}



void Batch::reset_batch()
{
    num_sources.clear();
    acc_batch = 0;

}

SOURCE_STATE NvStreamMux::get_pad_state(SinkPad* sinkPad)
{
    debug_print(" sinkPad->state=%d\n",sinkPad->state);
    return sinkPad->state;
}

void NvStreamMux::set_pts_offset(gulong offset)
{
    debug_print(" offset=0x%lx\n",offset);
    cur_frame_pts = offset;
}

bool NvStreamMux::push_batch(NvDsBatchBufferWrapper * out_buf, SourcePad * src_pad)
{
    //out_buf->push(src_pad, current_play_start_time, accum_play_dur);
    cur_frame_pts += frame_duration_nsec;
    return out_buf->push(src_pad, cur_frame_pts);
}

void NvStreamMux::set_batch_size(unsigned int size)
{
    std::unique_lock<std::mutex> lck(mutex);
    batch_policy.set_batch_size(size);
}

void NvStreamMux::set_batch_push_timeout(unsigned int timeout)
{
    std::unique_lock<std::mutex> lck(mutex);
    batch_policy.set_batch_push_timeout(timeout);
}

unsigned int NvStreamMux::get_batch_size()
{
    std::unique_lock<std::mutex> lck(mutex);
    return batch_policy.get_batch_size();
}

unsigned int NvStreamMux::get_config_batch_size()
{
    std::unique_lock<std::mutex> lck(mutex);
    return batch_policy.get_config_batch_size();
}

void NvStreamMux::set_num_surfaces_per_frame(unsigned int num)
{
    std::unique_lock<std::mutex> lck(mutex);
    num_surfaces_per_frame = num;
    batch_policy.set_num_surfaces(num);
}

unsigned int NvStreamMux::get_num_surfaces_per_frame()
{
    std::unique_lock<std::mutex> lck(mutex);
    return num_surfaces_per_frame;
}

void NvStreamMux::set_all_pads_eos(bool eos)
{
    std::unique_lock<std::mutex> lck(mutex);
    all_pads_eos = eos;
    /** Note: Cannot reset the counter: pads_got_eos_and_empty_in_q to 0
     * as the same LL context can be used for
     * muxing new streams after ALL current pads EOS
     */
}

void NvStreamMux::reset_pad(SinkPad* pad)
{
    std::unique_lock<std::mutex> lck(mutex);
    std::unique_lock<std::mutex> lck_pad(pad->mutex);
    if(pad->get_eos())
    {
        pad->set_eos(false);
        if(batch_policy.get_eos_sources() > 0)
        {
            batch_policy.update_eos_sources(batch_policy.get_eos_sources() - 1);
            /** since we are resetting this pad which was in EOS; mark it non-EOS now */
            pads_got_eos_and_empty_in_q--;
            all_pads_eos = false;
        }
    }
}

bool NvStreamMux::get_all_pads_eos()
{
    std::unique_lock<std::mutex> lck(mutex);
    return all_pads_eos;
}

bool NvStreamMux::push_loop(NvDsBatchBufferWrapper * out_buf, NvDsBatchMeta * dest_batch_meta)
{
    bool ret_push_batch_called = false;
    bool push_events_called = false;
    debug_print("%s %d %p\n", __func__, __LINE__, out_buf);
    /** documentation for push_events() calls in this API:
     * check and push events before and after forming a batch
     * this is done in the while loop when acc_batch == 0
     * and for once within the loop
     */
    /** Now that we pushed events, if input is empty, return here */
    if(!out_buf)
    {
        push_events();
        return false;
    }
    debug_print("DEBUGME %s\n", "what");
    while( !stop_task && (batch == NULL || (batch != NULL && !batch_policy.is_ready(batch))))
    {
        std::unique_lock<std::mutex> lck(mutex);
        if(inputs.size() == 0)
        {
            break;
        }
        NanoSecondsType wait_time;

        if(!push_events_called)
        {
            lck.unlock();
            push_events();
            lck.lock();
            push_events_called = true;
        }

        batch = batch_policy.form_batch(batch, batch_policy.get_batch_size());
        wait_time = batch_policy.calculate_wait();
        bool wait_not_past_max = wait_time > (std::chrono::milliseconds(0));
        debug_print("wait_not_past_max=%d acc_batch=%d\n", wait_not_past_max, batch->acc_batch);
        if(!batch_policy.is_ready(batch) && wait_not_past_max )
        {
            debug_print("DEBUGME\n");
            //std::cout << "waiting got " << wait_time.count() << "\n";
            /** Wait for new buffers on pads or if no buffer comes in, upto wait_time */
            cv.wait_for(lck, wait_time);
            if(batch->acc_batch == 0)
            {
                lck.unlock();
                push_events();
                lck.lock();
            }
            /** break push_loop if ready to stop task OR the batch
             * is due even when is_ready() returned false above */
            if(stop_task || (batch_policy.is_ready_or_due(batch)))
            {
                debug_print("stop_task=%d acc_batch=%d is_due=%d\n", stop_task,
                        batch->acc_batch,
                        batch_policy.is_ready_or_due(batch));
                break;
            }
        }
        else if((batch_policy.is_ready_or_due(batch) || !wait_not_past_max ) && (batch->acc_batch > 0))
        {
            debug_print("DEBUGME\n");
            lck.unlock();
            push_events();
            lck.lock();
            copy_batch(out_buf, dest_batch_meta);
            if(batch_policy.is_max_fps_control_enabled() && batch_policy.check_past_min())
            {
            debug_print("DEBUGME\n");
                /** we dont want lck to be locked while throttle wait happens */
                lck.unlock();
                apply_throttle();
                /** lock again to protect map variables in this helper class */
                lck.lock();
            }
            debug_print("DEBUGME\n");
            /** Update the batch output time and then push the batch */
            batch_policy.update_last_batch_time();
            /** push_batch depending on the configured downstream plugin,
             * could block on the downstream plugin's sink chain function
             * releasing the lck here to make sure we can queue incoming buffers
             * with add_buffer()
             */
            lck.unlock();
            push_batch(out_buf, src_pad);
            ret_push_batch_called = true;
            lck.lock();
            batch->reset_batch();
            break;
        }
        if(batch->acc_batch == 0 && !wait_not_past_max)
        {
            debug_print("DEBUGME\n");
            break;
        }
        //Calculate remaining_wait and delay??
        //delay or prune sources to drop frames??
    }
    debug_print("DEBUGME\n");
    push_events();
    std::unique_lock<std::mutex> lck(mutex);
    if(ret_push_batch_called == false)
    {
        out_buf->unref();
        /** we dont want lck to be locked while throttle wait happens */
        lck.unlock();
        apply_throttle(true);
        lck.lock();
    }
    debug_print("%s %d\n", __func__, __LINE__);
    return ret_push_batch_called;
}


void NvStreamMux::notify_all()
{
    cv.notify_all();
}


SinkPad * NvStreamMux:: get_pad(unsigned int pad_id)
{
    std::unique_lock<std::mutex> lck_m(mutex);
    return (SinkPad *)inputs[pad_id];

}

void NvStreamMux::remove_pad(unsigned int id)
{
    std::unique_lock<std::mutex> lck_m(mutex);
    auto it = inputs.find(id);
    guint count=0;

    debug_print("%s %d id=%d\n", __func__, __LINE__, id);

    if(it == inputs.end())
    {
        return;
    }

    if(inputs[id]->queue.size() == 0)
    {
        /** decrement stream EOS counters
        * as inputs.size also reduce by 1
        * and inputs.size is used in batch_policy logic and
        * batching logic along with EOS counters */
        num_queues_empty--;
	pads_got_eos_and_empty_in_q--;
	batch_policy.update_eos_sources(batch_policy.get_eos_sources() - 1);
    }

    while(inputs[id]->get_eos() !=true && count<100)
    {
        lck_m.unlock();
        debug_print("App sequence error release pad called before push events\n");
        g_usleep(1000);
        lck_m.lock();
        count++;
    }
    
    if(inputs[id]->get_eos() != true && count==100)
    {
        pads_got_eos_and_empty_in_q++; //To maintain proper count in case of error, compensate for decrement
        debug_print("100ms timeout exhausted. App sequence error release pad called before push events\n");
    }

    if(state == SOURCE_STATE_IDLE)
    {
        inputs[id]->set_switched_to_idle(true);
    }

    delete it->second;

    inputs.erase(it);

    debug_print("DEBUGME %s %d size=%ld\n", __func__, __LINE__, inputs.size());
    if(inputs.size() == 0)
    {
        all_pads_eos = true;
    }
}

void NvStreamMux::add_pad(unsigned int pad_id, SinkPad * pad)
{
    std::unique_lock<std::mutex> lck_m(mutex);
    pad->set_debug_interface(debug_iface);
    debug_print("%s %d id=%d\n", __func__, __LINE__, pad_id);
    inputs[pad_id] = pad;
    batch_policy.check_and_update_defaults(pad_id);
    all_pads_eos = false;
    inputs[pad_id]->set_switched_to_idle(true);
}

void NvStreamMux::add_buffer(unsigned int source_id, BufferWrapper * buffer)
{
    debug_print("id=%u\n", source_id);
    std::unique_lock<std::mutex> lck_m(mutex);
    SinkPad * pad = inputs[source_id];
    lck_m.unlock();
    unsigned int batch_size = get_batch_size();
    if(batch_size > pad->get_max_buffer_count())
        pad->set_max_buffer_count(batch_size);
    debug_print("id=%u max_buffer_count=%d batch_size=%d\n", source_id, pad->get_max_buffer_count(), batch_size);
    pad->wait_if_queue_full();
    debug_print("id=%u\n", source_id);
    lck_m.lock();
    std::unique_lock<std::mutex> lck(pad->mutex);
    if(pad->state == SOURCE_STATE_IDLE)
    {
        debug_print("%s %d switching to active\n", __func__, __LINE__);
        pad->set_switched_to_active(true);
    }

    if(pad->queue.size() == 0)
    {
        num_queues_empty--;
    }

    state = SOURCE_STATE_PLAYING;
    pad->state = SOURCE_STATE_PLAYING;

    if (pad->state == SOURCE_STATE_STOPPING)
    {
         buffer->free(); //gst_buffer_unref (buffer);
    }
    else
    {
          {
              std::shared_ptr<BufferWrapper> s_ptr(buffer);
              pad->queue_entry(s_ptr);
              batch_policy.total_buf_available++;
          }
    }
    all_pads_eos = false;

    if(!got_first_buffer)
    {
        got_first_buffer = true;
        /** update the last batch time before the first frame
         * This assumes an imaginary 0th batch before first output batch
         * and now is the start of muxer's timeline
         */
        batch_policy.update_last_batch_time();
    }

    debug_print("%s %d\n", __func__, __LINE__);
    cv.notify_all();

}



bool  NvStreamMux::handle_flush_stop(SINK_EVENT et, unsigned int source_id, EventWrapper * event)
{
    bool ret = true;
    std::unique_lock<std::mutex> lck_m(mutex);
    SinkPad * pad = inputs[source_id];

    if(!pad)
    {
        return false;
    }
    std::unique_lock<std::mutex> lck(pad->mutex);

    pad->state = SOURCE_STATE_PLAYING;
    debug_print("%s %d\n", __func__, __LINE__);
    pad->set_switched_to_active(true);
    all_pads_eos = false;
    if(pad->queue.size() == 0)
    {

    }
    {
        cv.notify_all();
    }
    return ret;

}

bool NvStreamMux::handle_eos(SINK_EVENT et, unsigned int source_id, EventWrapper * event)
{
    debug_print("handle_eos\n");
    bool ret = true;
    std::unique_lock<std::mutex> lck_m(mutex);
    debug_print("id=%u\n", source_id);
    SinkPad * pad = inputs[source_id];

    /** No need to handle EOS if already in EOS */
    if(!pad || pad->get_eos())
    {
        return false;
    }
    std::unique_lock<std::mutex> lck(pad->mutex);
    std::shared_ptr<EventWrapper> s_ptr(event);
    pad->queue_entry(s_ptr);
    SOURCE_STATE prev_state = pad->state;
    debug_print("%s %d prev_state=%d\n", __func__, __LINE__, prev_state);
    if (prev_state == SOURCE_STATE_IDLE)
    {
        cv.notify_all();
    }

    /** if we receive EOS without any buffers on the stream,
     * state == SOURCE_STATE_IDLE; yet still
     * we need to promptly set_switched_to_idle(true)
     */
    {
         pad->set_switched_to_idle(true);
         /** When EOS arrives, we are no longer active
          * and it is safe to put switched_to_active == true
          * as this value is only used in push_events()
          * called from push_loop() thereby ensuring
          * that any pending buffers in the pad's queue
          * will be batched.
          */
         pad->set_switched_to_active(false);
         cv.notify_all();
    }
    pad->set_eos(true);
    if(prev_state == SOURCE_STATE_IDLE)
    {
            lck_m.unlock();
            lck.unlock();
            push_events();
            lck.lock();
            lck_m.lock();
    }
    return ret;
}

bool NvStreamMux::handle_eos_cascaded(SINK_EVENT et, unsigned int source_id, EventWrapper * event)
{
    debug_print("handle_eos\n");
    bool ret = true;
    std::unique_lock<std::mutex> lck_m(mutex);
    debug_print("id=%u\n", source_id);
    SinkPad * pad = inputs[source_id];

    debug_print("Number of buffers available %d\n",pad->get_available());
    std::unique_lock<std::mutex> lck(pad->mutex);
    std::shared_ptr<EventWrapper> s_ptr(event);
    pad->queue_entry(s_ptr);
    return ret;
}

bool NvStreamMux::handle_segment(SINK_EVENT et, unsigned int source_id, EventWrapper * event)
{
    bool ret = true;
    std::unique_lock<std::mutex> lck_m(mutex);
    SinkPad * pad = inputs[source_id];
    if(!pad)
    {
        return false;
    }
    std::unique_lock<std::mutex> lck(pad->mutex);
    std::shared_ptr<EventWrapper> s_ptr(event);
    pad->queue_entry(s_ptr);
    cv.notify_all();
    return ret;
}

bool NvStreamMux::handle_stream_reset(SINK_EVENT et, unsigned int source_id, EventWrapper * event)
{
    bool ret = true;
    std::unique_lock<std::mutex> lck_m(mutex);
    SinkPad * pad = inputs[source_id];
    if(!pad)
    {
        return false;
    }
    std::unique_lock<std::mutex> lck(pad->mutex);
    std::shared_ptr<EventWrapper> s_ptr(event);
    pad->queue_entry(s_ptr);
    cv.notify_all();
    /*NOTE: The buffers waiting to be batched when this event arrives will also
     *be reset when ideally the buffers that are waiting should be flushed and
     *a flush event should be called.
     */
    if (frame_num_reset_on_stream_reset)
      pad->reset_frame_count();
    return ret;
}

void NvStreamMux::handle_pause_play()
{
    current_play_start_time = Clock::now();
    //std::cout << "handle_pause_play new start time " << std::chrono::duration_cast<std::chrono::nanoseconds>(current_play_start_time.time_since_epoch()).count() << "\n";
}

void NvStreamMux::handle_play_pause()
{
    accum_play_dur = accum_play_dur + (current_play_start_time - Clock::now());
    //std::cout << "handle_play_pause accum dur " << accum_play_dur.count() << "\n";
}

void NvStreamMux::handle_ready_pause()
{
    accum_play_dur = std::chrono::nanoseconds(0);
    //std::cout << "handle_ready_pause accum dur " << accum_play_dur.count() << "\n";

}

void NvStreamMux::reset_stop()
{
    stop_task = false;
}
void NvStreamMux::handle_stop()
{
    debug_print("handle_stop\n");
    std::unique_lock<std::mutex> lck_m(mutex);
    stop_task = true;
    cv.notify_all();
    for (auto it : inputs)
    {
        if(it.second)
        {
            std::unique_lock<std::mutex> lck(it.second->mutex);
            it.second->reset();
            if (frame_num_reset_on_eos)
              it.second->reset_frame_count();
        }
    }
}

void NvStreamMux::set_frame_duration(unsigned long dur)
{
    frame_duration_nsec = dur;
}

void NvStreamMux::update_pad_mimetype(unsigned int id, PAD_MIME_TYPE mime_type)
{
    SinkPad * pad = inputs[id];
    if(!pad)
    {
        return;
    }
    pad->set_mime_type(mime_type);
}

PAD_MIME_TYPE NvStreamMux::get_pad_mimetype(unsigned int id)
{
    SinkPad * pad = inputs[id];
    if(!pad)
    {
        return PAD_MIME_TYPE_INVALID;
    }
    return pad->get_mime_type();
}

void NvStreamMux::set_synch_buffer_iface(ISynchronizeBuffer* synch_buffer_iface)
{
    synch_buffer = synch_buffer_iface;
}

NanoSecondsType NvStreamMux::get_min_fps_duration()
{
    return batch_policy.get_min_fps_duration();
}

void NvStreamMux::apply_throttle(bool stop_when_input_buffer)
{
                std::unique_lock<std::mutex> throttle_lock_l(mutex_throttle_wait);
                NanoSecondsType throttle_delay = std::chrono::nanoseconds(0);
                /** According to the overall-max-fps setting, how much
                 * is the throttle_delay to wait before pushing out the batch: */
                if(stop_when_input_buffer)
                    throttle_delay = batch_policy.get_max_duration_delay();
                else
                    throttle_delay = batch_policy.calculate_delay();

                if(throttle_delay > std::chrono::nanoseconds(0))
                {
                    TimePoint timeNow = Clock::now();
                    if(throttle_delay > extra_throttle_wait_time)
                    {
                      throttle_delay = (throttle_delay - extra_throttle_wait_time);
                    }
                    if(stop_when_input_buffer)
                    {
                      /** wait call need to exit when there is a new input buffer
                       * in any of muxer input queues; so, wait on cv instead */
                      cv.wait_for(throttle_lock_l, throttle_delay);
                    }
                    else
                    {
                      cv_throttle_wait.wait_for(throttle_lock_l, throttle_delay);
                    }
                    TimePoint timeNow2 = Clock::now();
                    NanoSecondsType timeElapsed = timeNow2 - timeNow;
                    /** As semaphore wait method or any timeout method is not dependable in
                     * a general purpose operating system, we calculate any spurious
                     * extra delays induced and reduce that much time from the next throttle_delay;
                     * NOTE: This will help achieve reliable frames per second, but
                     * within a single second, we shall see jitter (means duration between
                     * every 30 frames in a second - 30 fps stream - will not be 33ms
                     * even when we achieve 30 fps overall)
                     */
                    if(timeElapsed > throttle_delay)
                    {
                      extra_throttle_wait_time = timeElapsed - throttle_delay;
                    }
                    else
                    {
                      extra_throttle_wait_time =  std::chrono::nanoseconds(0);
                    }
                }
}

/*
void BatchPolicy::drop_frames()
{
    std::list<unsigned int>::iterator prior_it = priority_list.get_max_pos();
    unsigned int remaining_count = max_queue_size;
    do
    {
        unsigned int priority = *prior_it;
        std::pair<std::iterator, std::iterator> source_it = batch->sources.get(priority);
        for(std::iterator index = source_it.first; index != source_it.second; index++)
        {

            drop_entries_source(*index, ENTRY_BUFFER, remaining_count);

        }

    }
    while(it != priority_list.get_max_pos());
}
*/
/*
void NvStreamMux::drop_entries_source(unsigned int source_id,  ENTRY_TYPE type, int & remaining_count)
{
    bool ignore_remaining = remaining_count == -1;
    SinkPad * source = mux->inputs[source_id];
    for ( auto it = queue.begin(); it != queue.end(); it++)
    {
        if(!ignore_remaining_count)
        {
            remaining_count--;
        }
        if(it->type == type)
        {
            queue.remove(it);
            delete it;
            if(!ignore_remaining_count)
            {
                if(remaining_count > 0)
                {
                    remaining_count--;
                }
                else
                {
                    break;
                }
            }
        }
    }

}
*/
