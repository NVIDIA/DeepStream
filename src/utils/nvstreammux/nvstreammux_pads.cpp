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

#include "nvstreammux_pads.h"
#include "nvstreammux_batch.h"

#if 1
#define debug_print(fmt, ...) \
            do { if(debug_iface) { debug_iface->DebugPrint("[%s %d]" fmt, __func__, __LINE__, ## __VA_ARGS__); } } while (0)
#else
#define debug_print(fmt, ...) printf("[DEBUG %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)
#endif

void SinkPad::queue_entry(std::shared_ptr<QueueEntry> entry)
{
    queue.push_back(entry);
    if(entry->type == ENTRY_EVENT)
    {
        //TBD May be we only need to store one index if they are back to back 
        event_indices.push_back((unsigned int)queue.size() - 1);
    }
    if(entry->type == ENTRY_BUFFER)
    {
        push_buffer_done();
    }
}

void SinkPad::update_frame_count(unsigned int count)
{
    frame_count += count;
}


unsigned long SinkPad::get_frame_count()
{
    return frame_count;
}

void SinkPad::adjust_event_indices(unsigned int offset, bool is_event)
{
    unsigned int deleted = 0;
    for (std::vector<unsigned int>::iterator it =  event_indices.begin(); it != event_indices.end();)
    {
        if(is_event && deleted < offset)
        {
           it = event_indices.erase(it); 
           deleted++;
        }
        else
        {
            *it -= offset;
            ++it;
        }

    }

}

void SinkPad::push_events(SourcePad * src_pad)
{

    int bottom_index = (event_indices.size() > 0) ? event_indices[0] : -1;
    unsigned int pushed_events = 0;
    /** When buffers (ENTRY_BUFFER) are removed from queue,
     * event_indices[0] value is not updated and hence
     * we need to check if first in queue is an event and force
     * handling the event;
     */
    if(queue.size() > 0 && queue[0]->type == ENTRY_EVENT)
    {
        bottom_index = 0;
    }
    if(bottom_index == 0)
    {
        for (std::vector<std::shared_ptr<QueueEntry> >::iterator i = queue.begin();
             i != queue.end();)
        { 
            debug_print("pad[%d] event_type=%d\n", id, (*i)->type);
            if((*i)->type == ENTRY_EVENT)
            {
                push_event(src_pad, (*i).get());
                //delete (*i);
                i = queue.erase(i);
                pushed_events++;

            }
            else
            {
                break;
            }

        }


    }
    adjust_event_indices(pushed_events, true);
    debug_print("[%s] available buffers=%u q.size=%lu event_indices.size=%lu\n",
                __func__, get_available(), queue.size(), event_indices.size());


}


unsigned int SinkPad::get_available()
{
    //ret = bottom_index = (event_indices.size() > 0) ? event_indices[0] : queue.size();
    debug_print("number of buffers available in queue of pad (%d) = %lu\n",
      id, queue.size() - event_indices.size());
    return queue.size() - event_indices.size();
}

void SinkPad::reset()
{
    queue.clear();
    event_indices.clear();
    state = SOURCE_STATE_IDLE;
}

void SinkPad::wait_if_queue_full()
{
    std::unique_lock<std::mutex> lck(mutex_buffer_count);
    if(max_buffer_count && buffer_count > max_buffer_count)
    {
        debug_print("source_id=%u buffer_count=%lu max_buffer_count=%lu\n",
                        source_id, buffer_count, max_buffer_count);
        cv_input_full.wait(lck);
    }
}

