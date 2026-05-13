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

#ifndef __NVSTREAMMUX_BATCH__H
#define __NVSTREAMMUX_BATCH__H


#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <unordered_map>
#include <map>
#include <list>
#include "nvbufsurface.h"
#include <string.h>
#include "nvstreammux_pads.h"
#include <nvdsmeta.h>
#if 0
#ifdef NEW_METADATA
#include "gstnvdsmeta.h"
#else
#include "gstnvstreammeta.h"
#include "gstnvdsmeta_int.h"
#endif
#endif
typedef std::chrono::steady_clock Clock;
using TimePoint = std::chrono::time_point<std::chrono::steady_clock, NanoSecondsType>;

typedef struct
{
    unsigned int source_max_fps_n;
    unsigned int source_max_fps_d;
    unsigned int source_min_fps_n;
    unsigned int source_min_fps_d;
    unsigned int priority;
    unsigned int max_num_frames_per_batch;
} NvStreammuxSourceProps;

typedef enum
{
  BATCH_METHOD_NONE,
  BATCH_METHOD_ROUND_ROBIN,
  BATCH_METHOD_PRIORITY
} NvStreammuxBatchMethod;

typedef struct BatchPolicyConfig
{
    NvStreammuxBatchMethod type;
    bool adaptive_batching;
    unsigned int batch_size;
    unsigned int overall_max_fps_n;
    unsigned int overall_max_fps_d;
    unsigned int overall_min_fps_n;
    unsigned int overall_min_fps_d;
    unsigned int max_same_source_frames;
    bool enable_source_rate_control;
    /** enables or disables the throttling control in push_loop()
     * implementing max_fps configuration support */
    bool enable_max_fps_control;
    std::unordered_map< unsigned int, NvStreammuxSourceProps> source_props;

    BatchPolicyConfig()
    {
        batch_size = 0;
    }

} BatchPolicyConfig;

typedef struct
{
    int priority_list_position;
    int source_map_position;
} LastBatchState;

//class BatchBufferWrapper : public BufferWrapper
class BatchBufferWrapper
{
    public :
        //BatchBufferWrapper(void * b) : batch(b)
        BatchBufferWrapper()
        {


        }
        virtual ~BatchBufferWrapper() = default;
        virtual unsigned int copy(void * buf, unsigned int pos, unsigned int num_surfaces)
        {
            return 0;
        }
        virtual bool push(SourcePad * pad, TimePoint play_start, NanoSecondsType accum_dur){ return false; };
        virtual bool push(SourcePad * pad, unsigned long pts){ return false; };
        virtual void unref(){};
        void * batch;
//        unsigned int num_filled;

};

class NvDsBatchBufferWrapper : public BatchBufferWrapper
{
    public :
        NvDsBatchBufferWrapper(unsigned int size);
        virtual ~NvDsBatchBufferWrapper();

        /**
         * @brief  Copy input buffer (buf) to this NvDsBatchBufferWrapper
         * @param  buf [IN] BufferWrapper object with a streammux input buffer
         * @param  pos [IN] The index at which the input buf is copied to
         *         the output NvBufSurface->surfaceList[] for video;
         */
        virtual unsigned int copy_buf(std::shared_ptr<BufferWrapper>  buf, unsigned int pos) = 0;

        virtual  void copy_meta(unsigned int id, std::shared_ptr<BufferWrapper> src_buffer, unsigned int batch_id, unsigned int frame_number,  unsigned int num_surfaces_per_frame, NvDsBatchMeta * dest_batch_meta, unsigned int source_id){};
        virtual void unref(){};
        unsigned int batch_size;

};

class Batch
{


    public :

        Batch(unsigned int size)
        {
            acc_batch = 0;
            batch_size = size;
        }
        //void form_batch();

        void check_source(unsigned int source_id);

        void reset_batch();

        void set_size(unsigned int size);



        /*
         * @brief map of number of sources for sources in batch
         *        key=source_id
         *        value=number of surfaces to copy from this source
         */
        std::unordered_map<int, int> num_sources;

        /** number of buffers already accumulated in this batch
         * Still not copied; yet info available in @num_sources
         */
        unsigned int acc_batch;
        /** number of surfaces in the batch
         * this will be one NvDsFrameMeta (one frame) in the batch meta */
        unsigned int batch_size;
        unsigned int num_surfaces_in_batch;


};


class SortedList
{
    public :
        void sorted_insert(unsigned int);
        std::list<unsigned int>::iterator get_next_pos(std::list<unsigned int>::iterator pos);
        std::list<unsigned int>::iterator get_max_pos();
        unsigned int get_least();
        std::list<unsigned int>::iterator get_least_pos();
        unsigned int get_at(std::list<unsigned int>::iterator pos);
        int size();
    private :
        std::list<unsigned int> store;
};

class BatchPolicy
{
    public :
        BatchPolicy() {};
        BatchPolicy(BatchPolicyConfig policy, std::unordered_map<unsigned int, SinkPad *> * ins, INvStreammuxDebug* a_debug_iface);

        unsigned int check_repeats_per_batch();

        unsigned int check_repeats_per_batch(unsigned int source_id);

        /*
         * @brief function to try to form a batch per current priorities
         * and set algorithm
         * @param b [IN] the batch to update
         */
        Batch * form_batch( Batch * b, unsigned int batch_size);

        unsigned int get_batch_size();

        unsigned int get_config_batch_size();

        void set_batch_size(unsigned int);

	void set_batch_push_timeout(unsigned int);
        /**
         * @brief  Set num_surfaces_per_frame
         */
        void set_num_surfaces(unsigned int);

        /*
         * @brief function to update a batch with buffers from a source
         *        If Timestamp synchronization user API is set,
         *        only on-time buffers will be batched.
         *        late buffers will be discarded.
         *        early buffers will not be batched with the current call.
         * @param source_id [IN] the id of pad
         * @param batch  [IN] the batch to update
         */
        void update_with_source(Batch * batch, unsigned int source_id);

        /*
         * @brief function to check amount of time to wait for batch data
         * based on current time and min frame rate based duration
         */
        NanoSecondsType calculate_wait();

        //NanoSecondsType update_last_batch_time(NanoSecondsType last_batch_time);
        void update_last_batch_time();

        /*
         * @brief function to get delay for next batch
         * based on max frame rate based duration
         */
        NanoSecondsType get_max_duration_delay();

        /*
         * @brief function to calculate delay for next batch
         * based on current time and max frame rate based duration
         */
        NanoSecondsType calculate_delay();

        /*
         * @brief function to check if a batch is ready to be pushed
         * @param b [IN] the batch object pointer
         */
        bool is_ready(Batch * b);

        bool is_ready_or_due(Batch * b);

        /*
         * @brief function to check if defaults exist for this source id
         * from config, (TBD) this mechanism needs to replaced with something
         * that is at initialization time
         * @param source_id [IN] the id of pad
         */
        void check_and_update_defaults(unsigned int source_id);

        //void reset_batch(Batch * b);
        unsigned int total_buf_available;

        bool check_past_min();

        void update_idle_sources(unsigned int idle_count);

        void update_eos_sources(unsigned int eos_count);

        unsigned int get_eos_sources();

        void update_push_stats(unsigned int source_id, unsigned int num_pushed);

        bool is_max_fps_control_enabled();

        void set_synch_buffer_iface(ISynchronizeBuffer* synch_buffer_iface);
        
        /**
         * @brief  Synchronize the buffers in queue for provided pad
         * @return The number of on time buffers in the pad queue
         */
        unsigned int synchronize_buffers_in_pad(SinkPad* pad,
                unsigned int allowed_buffers);

        NanoSecondsType get_min_fps_duration();

        BatchPolicyConfig get_config();

    private :
        unsigned int get_allowed(unsigned int source_id,  float fps, unsigned int available);

        BatchPolicyConfig config;
        NanoSecondsType  max_fps_dur;
        NanoSecondsType  min_fps_dur;
        /** The minimimum of early-buffer early-by time among all sources' input buffers */
        NanoSecondsType  min_early_buffer_dur;
        TimePoint max_dur_time;
        TimePoint min_dur_time;
        std::unordered_map<unsigned int, float> src_max_fps;
        std::unordered_map<unsigned int, NanoSecondsType> min_src_fps_dur;
        std::unordered_map<unsigned int, TimePoint> src_push_times;
        std::unordered_map<unsigned int, unsigned int> src_num_pushed;
        SortedList priority_list;
        /*
         * @brief map of sources for a priority
         *        key = priority
         *        value = source_id
         */
        std::multimap<int, int> sources;
        std::unordered_map<unsigned int, SinkPad *> * inputs;
        std::chrono::time_point<std::chrono::steady_clock> last_batch_time;
        unsigned int batch_size;
        LastBatchState last_batch_state;
        /** The num_sources_idle is updated when:
         * a pad is added (idle++)
         * a buffer arrives on a pad (idle--)
         * a pad is removed (idle++)
         * eos arrives on pad (idle++)
         * flush event arrives on pad (idle++)
         */
        unsigned int num_sources_idle;
        /** The num_sources_eos is updated when:
         * The pad is idle
         * and the pad has 0 buffers + events in queue
         */
        unsigned int num_sources_eos;
        unsigned int num_surfaces_per_frame;
        ISynchronizeBuffer* synch_buffer;
        INvStreammuxDebug* debug_iface;


};


#endif
