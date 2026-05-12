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

#include "nvstreammux_batch.h"
#include "nvstreammux_pads.h"
#include "MuxConfigParser.h"

#if 1
#define debug_print(fmt, ...) \
            do { if(debug_iface) { debug_iface->DebugPrint("[%s %d]" fmt, __func__, __LINE__, ## __VA_ARGS__); } } while (0)
#else
#define debug_print(fmt, ...) printf("[DEBUG %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)
#endif

std::list<unsigned int>::iterator SortedList::get_least_pos()
{

    return store.begin();

}

std::list<unsigned int>::iterator SortedList::get_max_pos()
{

    return store.end();

}
std::list<unsigned int>::iterator SortedList::get_next_pos(std::list<unsigned int>::iterator pos)
{

    return pos++;

}

int SortedList::size()
{
    return store.size();
}

unsigned int SortedList::get_at(std::list<unsigned int>::iterator pos)
{

    return *pos;

}
void SortedList::sorted_insert(unsigned int new_val)
{

    std::list<unsigned int>::iterator it  = store.begin();

    if (store.size() == 0 || *it > new_val)
    {
        store.insert(it, new_val);
    }
    else
    {
        while ((it != store.end()) &&
               *it < new_val)
        {
            it++;
        };
        if(it == store.end() || (it != store.end() && *it != new_val))
        {
            store.insert(it, new_val);
        }
    }
}

NvDsBatchBufferWrapper::NvDsBatchBufferWrapper(unsigned int size)
{
    //TBD error handling
    batch_size = size;
}

NvDsBatchBufferWrapper::~NvDsBatchBufferWrapper()
{
    batch_size = 0;
}

void Batch::set_size(unsigned int size)
{
    batch_size = size;
}
void BatchPolicy::update_last_batch_time()
{
    last_batch_time = Clock::now();
    min_dur_time = last_batch_time + max_fps_dur;
    max_dur_time = last_batch_time + min_fps_dur;
    //std::cout << "update_last_time " << "now " << std::chrono::duration_cast<std::chrono::nanoseconds>(last_batch_time.time_since_epoch()).count() << "max_dur_time " << std::chrono::duration_cast<std::chrono::nanoseconds>(max_dur_time.time_since_epoch()).count() << " min_dur_time " << std::chrono::duration_cast<std::chrono::nanoseconds>(min_dur_time.time_since_epoch()).count() << "\n";

}

BatchPolicy::BatchPolicy(BatchPolicyConfig policy,
        std::unordered_map<unsigned int, SinkPad *> * ins,
        INvStreammuxDebug* a_debug_iface) : inputs(ins), debug_iface(a_debug_iface)
{
    config = policy;

    for ( auto it = policy.source_props.begin(); it != policy.source_props.end(); ++it)
    {
        priority_list.sorted_insert(it->second.priority);
        sources.insert(std::pair<unsigned int, unsigned int>(it->second.priority, it->first));
        NanoSecondsType s_min_fps_dur = NanoSecondsType((double)it->second.source_min_fps_d/(double)(it->second.source_min_fps_n));
        float s_max_fps = (float)it->second.source_max_fps_n/(float)(it->second.source_max_fps_d);
        debug_print("BatchPolicy insert priority %d source %d max_fps %f\n", it->second.priority, it->first, s_max_fps);
        src_max_fps.insert(std::make_pair(it->first,s_max_fps));
    }

    // Add sources not found in the policy config with default priority
    for (auto &inp : *inputs) {
        int source_id = inp.first;
        bool found = false;
        for (auto &it : sources) {
            if(it.second == source_id) {
                found = true;
                break;
            }
        }
        if (!found) {
            sources.insert(std::pair<unsigned int, unsigned int>(NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY, source_id));
            priority_list.sorted_insert(NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY);
        }
    }

    batch_size = config.adaptive_batching ? inputs->size() : config.batch_size;
    debug_print("BatchPolicy batch_size %d config batch size %d\n", batch_size, config.batch_size);
    max_fps_dur = NanoSecondsType(((double)config.overall_max_fps_d * 1000000000.0)/(double)(config.overall_max_fps_n));
    min_fps_dur = NanoSecondsType(((double)config.overall_min_fps_d * 1000000000.0)/(double)(config.overall_min_fps_n));
    min_early_buffer_dur = NanoSecondsType(0);
    std::cout << "max_fps_dur " << max_fps_dur.count() << " min_fps_dur " << min_fps_dur.count() << "\n";
    debug_print("%s max_fps_dur=%lf min_fps_dur=%lf overall_max_fps=%d/%d overall_min_fps=%d/%d\n",
        __func__, max_fps_dur.count()/1000, min_fps_dur.count()/1000, config.overall_max_fps_n, config.overall_max_fps_d,
        config.overall_min_fps_n, config.overall_min_fps_d);

    update_last_batch_time();
    last_batch_state.priority_list_position = 0;
    last_batch_state.source_map_position = 0;
    num_sources_idle = 0;
    num_sources_eos = 0;
    num_surfaces_per_frame = 1;
    synch_buffer = nullptr;
}

NanoSecondsType BatchPolicy::calculate_wait()
{
    /** wait is based on min_fps configured
     * if min_fps=5/1, max_dur_time = 200ms + last_batch_push_time
     */
    NanoSecondsType max_wait_time = NanoSecondsType(0);
    if(min_early_buffer_dur > min_fps_dur)
    {
        debug_print("early by %lf ns\n", min_early_buffer_dur.count());
        /** we shall wait only for min_early_buffer_dur nanoseconds;
         * so, recalculate max_dur_time
         */
        max_dur_time = (last_batch_time - min_fps_dur) + min_early_buffer_dur;
    }
    /** reset min_early_buffer_dur as it needs to be calculated again with
     * the next call to ISynchronizeBuffer::get_synch_info() */
    min_early_buffer_dur = NanoSecondsType(0);
    TimePoint now = Clock::now();

    if(max_dur_time > now) {
      max_wait_time =  NanoSecondsType(max_dur_time - now);
    }
    else {
      debug_print("we are running tight in muxer\n");
    }
    //std::cout << "calculate_wait calculates " << max_wait_time.count() << "max_dur_time " << std::chrono::duration_cast<std::chrono::nanoseconds>(max_dur_time.time_since_epoch()).count() << " now " << std::chrono::duration_cast<std::chrono::nanoseconds>((Clock::now()).time_since_epoch()).count() << "\n";

    return max_wait_time;

}

NanoSecondsType BatchPolicy::get_max_duration_delay()
{
    return max_fps_dur;
}

NanoSecondsType BatchPolicy::calculate_delay()
{
    NanoSecondsType delay_time = NanoSecondsType(0);
    TimePoint now = Clock::now();
    if(min_dur_time > now)
    {
      delay_time = min_dur_time - now;
    }
    return delay_time;

}

bool BatchPolicy::check_past_min()
{
    bool wait_past_min = (min_dur_time - Clock::now()) > (std::chrono::nanoseconds(0));
    return wait_past_min;
}

bool BatchPolicy::is_ready(Batch * batch)
{
    //unsigned int cur_batch_size = get_batch_size();
    //return ((batch->acc_batch == cur_batch_size) && (cur_batch_size != 0));
    return ((batch->acc_batch == batch->batch_size) && (batch->batch_size != 0));
}

bool BatchPolicy::is_ready_or_due(Batch * batch)
{
    bool batch_ready = is_ready(batch);

    if(!batch_ready)
    {
        /* If not ready and we have atleast one buffer in the batch,
         * check if we are late to push this batch?
         * if late and have atleast 1 buffer in batch, mark the batch as ready
         * for push.
         */
        if(calculate_delay() == NanoSecondsType(0)
           && batch->acc_batch > 0)
        {
            /** force the batch out */
            return true;
        }
    }
    return batch_ready;
}


void BatchPolicy::check_and_update_defaults(unsigned int source_id)
{
    bool already_exists = false;
    for (auto entry : sources)
    {
        //TBD change map to unsigned
        if((unsigned int)entry.second == source_id)
        {
             already_exists = true;
             break;
        }
    }
    if(!already_exists)
    {
        sources.insert(std::pair<unsigned int, unsigned int>(NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY, source_id));
        priority_list.sorted_insert(NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY);

        if(config.source_props.find(source_id) == config.source_props.end())
        {
            NvStreammuxSourceProps source_prop;
            source_prop.source_max_fps_n = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_N;
            source_prop.source_max_fps_d = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FPS_D;
            source_prop.source_min_fps_n = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_N;
            source_prop.source_min_fps_d = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MIN_FPS_D;
            source_prop.priority = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_PRIORITY;
            source_prop.max_num_frames_per_batch = NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH;
            config.source_props.insert(std::pair<unsigned int, NvStreammuxSourceProps>(source_id, source_prop));
            float s_max_fps = (float)source_prop.source_max_fps_n/(float)(source_prop.source_max_fps_d);
            debug_print("%s BatchPolicy insert priority %d source %d max_fps %f\n", __func__, source_prop.priority, source_id, s_max_fps);
            src_max_fps.insert(std::make_pair(source_id, s_max_fps));
        }



    }
}

void BatchPolicy::update_push_stats(unsigned int source_id, unsigned int num_pushed)
{
    if(config.enable_source_rate_control)
    {
        src_push_times[source_id] = Clock::now();
        src_num_pushed[source_id] = num_pushed;
    }
}

unsigned int BatchPolicy::get_allowed(unsigned int source_id, float fps, unsigned int available)
{
    unsigned int ret_available = available;

    if(config.enable_source_rate_control)
    {
        auto it = src_push_times.find(source_id);

        if(it != src_push_times.end())
        {
            ret_available = (unsigned int)((Clock::now() - it->second).count() * fps/1000000000);
            debug_print("allowed frames %d fps %f diff %f source id %d\n", ret_available, fps, (Clock::now() - it->second).count(), source_id);
            ret_available = (ret_available > available) ? available : ret_available;
        }
    }

    return ret_available;

}

unsigned int BatchPolicy::get_config_batch_size()
{
   /** if adaptive_batching is enabled;
    * config batch-size is same as number of input streams
    */
   return config.adaptive_batching ? std::max(config.batch_size, (uint32_t)inputs->size()) : config.batch_size;
}

unsigned int BatchPolicy::get_batch_size()
{
   unsigned int ret;
   /** NOTE: num_sources_idle shall be > 0
    * before getting any input buffers
    * or towards the end (EOS from certain streams)
    * In that case, get_batch_size() shall return
    * batch-size < actual-batch-size
    */
   ret = config.adaptive_batching ? ((inputs->size() - num_sources_eos) * num_surfaces_per_frame) : config.batch_size;
   debug_print("%s inputs->size=%lu num_sources_eos=%d num_surfaces_per_frame=%d config=%d adaptive_batching=%d, batch_size=%d\n",
        __func__, inputs->size(), num_sources_eos, num_surfaces_per_frame, config.batch_size, config.adaptive_batching, ret);
   return ret;
}

void BatchPolicy::set_batch_size(unsigned int size)
{
   config.batch_size = size;
}

void BatchPolicy::set_batch_push_timeout(unsigned int timeout)
{
   config.overall_min_fps_n = (1000000/timeout);
   config.overall_min_fps_d = 1;
   min_fps_dur = NanoSecondsType(((double)config.overall_min_fps_d * 1000000000.0)/(double)(config.overall_min_fps_n));
}

void BatchPolicy::set_num_surfaces(unsigned int num)
{
   num_surfaces_per_frame = num;
}

unsigned int BatchPolicy::check_repeats_per_batch()
{
    return config.max_same_source_frames;
}

void BatchPolicy::update_idle_sources(unsigned int idle_count)
{
    debug_print("%s %d idle_sources=%d\n", __func__, __LINE__, idle_count);
    num_sources_idle = idle_count;
}

void BatchPolicy::update_eos_sources(unsigned int eos_count)
{
    debug_print("%s %d eos_sources=%d\n", __func__, __LINE__, eos_count);
    num_sources_eos = eos_count;
}

unsigned int BatchPolicy::get_eos_sources()
{
    return num_sources_eos;
}

unsigned int BatchPolicy::check_repeats_per_batch(unsigned int source_id)
{
    unsigned int batch_size_val = get_batch_size();
    if(config.source_props.end() == config.source_props.find(source_id))
    {
        /** no entry for this source_id; return default */
        return (batch_size_val > NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH)
                ? batch_size_val : NVSTREAMMUX_DEFAULT_SOURCE_GROUP_MAX_FRAMES_PER_BATCH ;
    }
    return (batch_size_val > config.source_props[source_id].max_num_frames_per_batch)
                ? batch_size_val : config.source_props[source_id].max_num_frames_per_batch;
}

void BatchPolicy::update_with_source(Batch * batch, unsigned int source_id)
{
    unsigned int allowed_repeats = check_repeats_per_batch();
    unsigned int allowed_repeats_source = check_repeats_per_batch(source_id);

    allowed_repeats = allowed_repeats <= allowed_repeats_source ? allowed_repeats : allowed_repeats_source;
    auto it = inputs->find(source_id);
    SinkPad * pad;
    debug_print("update_with_source sid=%d\n", source_id);
    if((it != inputs->end()) && ((pad = it->second) != NULL))
    {
        std::unique_lock<std::mutex> lck(pad->mutex);
        unsigned int num_avail = get_allowed(source_id, src_max_fps[source_id], pad->get_available());
        debug_print("num_avail=%u sid=%d\n", num_avail, source_id);
        lck.unlock();
        if(batch->num_sources[source_id] == (int)allowed_repeats)
        {
            /** This source already added into the batch */
            return;
        }
        debug_print("allowed_repeats=%u allowed_repeats_source=%u bs=%u\n", allowed_repeats, allowed_repeats_source, get_batch_size());
        num_avail =  num_avail <= allowed_repeats ? num_avail : allowed_repeats;
        debug_print("num_avail=%u sid=%d\n", num_avail, source_id);
        unsigned int num_to_insert = (num_avail >= (batch->batch_size - batch->acc_batch)) ? (batch->batch_size - batch->acc_batch) : num_avail;

        /** Timestamp synchronization for the #num_to_insert buffers
         * in this source's queue can be done here
         */
        debug_print("synch_buffer=%p num_to_insert=%d\n", synch_buffer, num_to_insert);
        if(synch_buffer)
        {
            num_to_insert = synchronize_buffers_in_pad(pad, num_to_insert);
        }
        
        // CHECK, NOw that this loop excutes only once, we should not need this - batch->num_sources[source_id] = batch->num_sources[source_id] + num_to_insert;
        batch->num_sources[source_id] = num_to_insert;
        batch->acc_batch += num_to_insert; // batch->num_sources[source_id];
        debug_print("update_with_source insert %d acc_batch %d source_id %d available %d max repeats %d\n", num_to_insert, batch->acc_batch, source_id,  pad->get_available(), allowed_repeats);
    }
}



/*
 * @brief Try to form batch according to priority of sources
 * TBD may need to improve to only iterate over changed sources, challenge  with
 * that approach would be do this in accordance with priority
 *
 */
Batch * BatchPolicy::form_batch(Batch * batch, unsigned int this_batch_size)
{
    Batch * ret_batch;
    if(batch == NULL)
    {
        ret_batch = new Batch(this_batch_size);
    }
    else
    {
        ret_batch = batch;
        ret_batch->set_size(this_batch_size);
    }
    ret_batch->reset_batch();
    std::list<unsigned int>::iterator prior_it = priority_list.get_least_pos();
    int total_priorities = priority_list.size();
    int priorities_to_process = total_priorities;
    int priority_index = 0;
    int source_index = 0;
    bool jump_source = true;

    if(config.type == BATCH_METHOD_ROUND_ROBIN)
    {
        int to_advance = last_batch_state.priority_list_position;
        std::advance(prior_it, to_advance);
        if(prior_it == priority_list.get_max_pos())
        {
            prior_it = priority_list.get_least_pos();
        }
        priority_index = to_advance;
    }

    do
    {
        unsigned int priority = *prior_it;
        /** sources is a multimap <priority, source_id> */
        auto source_it = sources.equal_range(priority);
        /** source_it::first is lower bound and second the upper bound
		 * range that includes all the elements in the map
		 * with key == priority provided
		 */
        int total_sources = std::distance(source_it.first, source_it.second);
        int sources_to_process = total_sources;
        debug_print("sources_to_process=%d\n", sources_to_process);
        auto saved_first = source_it.first;
        if((jump_source == true) && (config.type == BATCH_METHOD_ROUND_ROBIN))
        {
            int to_advance = last_batch_state.source_map_position;
            std::advance(source_it.first, to_advance);
            source_index = to_advance;
            jump_source = false;
        }
        else
        {
            source_index = 0;
        }
        
        auto index = source_it.first;
        do
        {
            debug_print("update_with_source sid=%d\n", index->second);
            update_with_source(ret_batch, index->second);
            ++index;
            source_index++;
            sources_to_process--;
            if(index == source_it.second)
            {
                debug_print("%s %d\n", __func__, __LINE__);
                index = saved_first;
                source_index = 0;
            }

            if((is_ready(ret_batch)))
            {
                last_batch_state.source_map_position = source_index;
                last_batch_state.priority_list_position = priority_index;
                break;
            }
        } while((sources_to_process > 0) && (!is_ready(ret_batch)));
        if(is_ready(ret_batch))
        {
            break;
        }
        prior_it++;
        priorities_to_process--;
        priority_index++;
        if(prior_it == priority_list.get_max_pos())
        {
            prior_it = priority_list.get_least_pos();
            priority_index = 0;
        }
        if((priorities_to_process == 0) || (is_ready(ret_batch)))
        {

            last_batch_state.source_map_position = source_index;
            last_batch_state.priority_list_position = priority_index;
        }
    }
    while((priorities_to_process > 0) && (!is_ready(ret_batch)));
    return ret_batch;
}


bool BatchPolicy::is_max_fps_control_enabled()
{
    return config.enable_max_fps_control;
}

void BatchPolicy::set_synch_buffer_iface(ISynchronizeBuffer* synch_buffer_iface)
{
    synch_buffer = synch_buffer_iface;
}

unsigned int BatchPolicy::synchronize_buffers_in_pad(SinkPad* pad,
                unsigned int allowed_buffers)
{
    unsigned int on_time_buffers = 0;
    unsigned int num_buffers_to_check = allowed_buffers;
    debug_print("DEBUGME allowed_buffers=%d\n", allowed_buffers);
    for(std::vector<std::shared_ptr<QueueEntry> >::iterator it_s = pad->queue.begin(); 
     (it_s != pad->queue.end() && (num_buffers_to_check != 0)); 
         )
    {
          /** skip events */
          if(std::static_pointer_cast<QueueEntry>(*it_s)->type != ENTRY_BUFFER)
          {
            debug_print("%s %d event in queue ignored\n", __func__, __LINE__);
            it_s++;
            continue;
          }
          debug_print("DEBUGME num_buffers_to_check=%d %p\n", num_buffers_to_check, (*it_s).get());

          BUFFER_TS_STATUS ts_status = synch_buffer->get_synch_info((BufferWrapper*)((*it_s).get()));
          debug_print("ts_status=%d\n", ts_status);
          num_buffers_to_check--;
          if(ts_status == BUFFER_TS_EARLY)
          {
             /** early buffer; skip other buffers in queue as they will all be early */
             NanoSecondsType this_buffer_earlyby_time = synch_buffer->get_buffer_earlyby_time();
             if((this_buffer_earlyby_time < min_early_buffer_dur)
                || (min_early_buffer_dur == NanoSecondsType(0)))
             {
                 min_early_buffer_dur = this_buffer_earlyby_time;
             }
             break;
          }
          else if(ts_status == BUFFER_TS_LATE)
          {
             /** remove and skip late buffer */
             /** pop the buffer from pad's queue */
             synch_buffer->removing_old_buffer((BufferWrapper*)((*it_s).get()));
             it_s = pad->queue.erase(it_s);
             pad->pop_buffer_done();
             num_buffers_to_check++; //we may check one more buffer as this one was late and discarded
          }
          else
          {
             on_time_buffers++;
             it_s++;
          }
    }
    debug_print("DEBUGME on_time_buffers=%u\n", on_time_buffers);
    return on_time_buffers;
}

NanoSecondsType BatchPolicy::get_min_fps_duration()
{
    return min_fps_dur;
}

BatchPolicyConfig BatchPolicy::get_config()
{
    return config;
}
