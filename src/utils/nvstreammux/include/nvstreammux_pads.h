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

#ifndef __NVSTREAMMUX_PADS__H__
#define __NVSTREAMMUX_PADS__H__

#include <condition_variable>
#include <mutex>
#include  <vector>
#include <chrono>
#include "nvstreammux_debug.h"

typedef std::chrono::duration<double, std::nano> NanoSecondsType;

typedef enum
{
   SINK_EVENT_EOS,
   SINK_EVENT_PLAY_START,
   SINK_EVENT_SEGMENT,
   SINK_EVENT_FLUSH_STOP,
   SINK_EVENT_STREAM_START,
   SINK_EVENT_STREAM_RESET,
} SINK_EVENT;


typedef enum
{
    SOURCE_STATE_IDLE,
    SOURCE_STATE_PAUSED,
    SOURCE_STATE_PLAYING,
    SOURCE_STATE_STOPPING,
} SOURCE_STATE;

typedef enum
{
    ENTRY_ALL,
    ENTRY_BUFFER,
    ENTRY_EVENT
} ENTRY_TYPE;


typedef enum
{
    BATCH_SEQUENCE_IN_BATCH,
    BATCH_SEQUENCE_PRE_BATCH,
    BATCH_SEQUENCE_POST_BATCH,

} BATCH_SEQUENCE_TYPE;

typedef enum
{
    PAD_MIME_TYPE_INVALID,
    PAD_MIME_TYPE_VIDEO,
    PAD_MIME_TYPE_AUDIO,
} PAD_MIME_TYPE;

typedef enum
{
    BUFFER_TS_EARLY,
    BUFFER_TS_ONTIME,
    BUFFER_TS_LATE,
} BUFFER_TS_STATUS;

/*
 * @brief Wrapper class for queued events
 */
class QueueEntry
{
    public :
        QueueEntry(void * entry, ENTRY_TYPE et, BATCH_SEQUENCE_TYPE bt = BATCH_SEQUENCE_IN_BATCH) : wrapped(entry), batch_type(bt)
        {
            type = ENTRY_EVENT;
        }
        virtual ~QueueEntry() = default;
        void * wrapped;
        BATCH_SEQUENCE_TYPE batch_type;
        ENTRY_TYPE type;
};

/*
 * @brief Wrapper class for queued source buffers
 */
class BufferWrapper : public QueueEntry
{

    public :
        BufferWrapper(void * buffer, ENTRY_TYPE et, BATCH_SEQUENCE_TYPE bt = BATCH_SEQUENCE_IN_BATCH, uint64_t ts = 0) : QueueEntry(buffer, et, bt)
        {
            type = ENTRY_BUFFER;
            raw = buffer;
            timestamp = 0;
        }
        virtual ~BufferWrapper() = default;
        void * raw;
        uint32_t rawSize;
        //virtual void copy_meta(NvStreammuxMeta * dest);
        //virtual void copy_buf(OutBufType  * dest);
        virtual void free() = 0;

        uint64_t timestamp;

};


class NvDsBufferWrapper : public BufferWrapper
{
    public :
        //virtual void copy_meta(NvStreamMeta * dest);
        //virtual void copy_to(NvDsBuffer * buffer);



};

/*
 * @brief Wrapper class for queued events
 */
class EventWrapper : public QueueEntry
{
    public :
        EventWrapper(void * event, ENTRY_TYPE et , BATCH_SEQUENCE_TYPE bt) : QueueEntry(event, et, bt)
        {
            type = ENTRY_EVENT;
        }
        ~EventWrapper() {}
};

/*
 * @brief SourcePad is abstraction of pad for outgoing data
 */
class SourcePad
{

    public :
        SourcePad(unsigned int id, void * pad) : wrapped(pad), id(id)
        {

        }
        /*
         * @brief push a buffer on outgoing pad
         */
        //void push_buffer(BufferWrapper *);
        /*
         * @brief push a buffer on outgoing pad
         */
        //void push_event();

        SOURCE_STATE state;
        //unsigned int num_bufs_in_queue; //derive from buf_queue size

        void * wrapped;
    protected :
        unsigned int id;
};
/*
 * @brief SinkPad is abstraction of pad for incoming data
 */
class SinkPad
{

    public:
        SinkPad(unsigned int id, void * pad) : queue(), id(id), wrapped(pad), mutex(), mutex_buffer_count(), source_id(id)
        {
            //top_event_index = -1;
            buffer_count = 0;
            max_buffer_count = 0;
            frame_count = 0;
            state = SOURCE_STATE_IDLE;
            switched_to_idle = false;
            switched_to_active = false;
            mime_type = PAD_MIME_TYPE_VIDEO;
            eos = false;
        }

        virtual ~SinkPad(){}
        /*
         * @brief release all resources of this pad
         */
        void release();
        /*
         * @brief wait till queue is empty
         */
        void wait_till_empty();
        /*
         * @brief check if queue is empty
         */
        bool check_queue_empty();
        //boolean check_queued_events();
        /*
         * @brief queue a buffer or event entry to  ordered queue
         */
        void queue_entry(std::shared_ptr<QueueEntry> );
        //void add_buffer(BufferWrapper *);
        //void add_event(EventWrapper *);
        /*
         * @brief wait till there is some activity on the pad
         */
        //virtual void wait();
        /*
         * @brief notify all waiting on the pad
         */
        //virtual void notify_all();

        void push_events(SourcePad * src_pad);

        virtual void push_event(SourcePad * src_pad, QueueEntry *){};


        unsigned int get_available();

        void adjust_event_indices(unsigned int, bool is_event);

        void update_frame_count(unsigned int count);

        unsigned long get_frame_count();

        void reset_frame_count() { frame_count = 0; }

        void reset();

        void clear_frames();

        void set_switched_to_idle(bool val)
        {
            switched_to_idle = val;
        }

        void set_switched_to_active(bool val)
        {
            switched_to_active = val;
        }

        bool get_switched_to_idle()
        {
            return switched_to_idle;
        }

        bool get_switched_to_active()
        {
            return switched_to_active;
        }

        void set_mime_type(PAD_MIME_TYPE n_mime_type)
        {
            mime_type = n_mime_type;
        }

        PAD_MIME_TYPE get_mime_type()
        {
            return mime_type;
        }

        void set_eos(bool aEos)
        {
            eos = aEos;
        }

        bool get_eos()
        {
            return eos;
        }

        void push_buffer_done()
        {
            std::unique_lock<std::mutex> lck(mutex_buffer_count);
            buffer_count++;
        }

        /** always call after queue_entry() for type=ENTRY_BUFFER */
        void wait_if_queue_full();

        void pop_buffer_done()
        {
            std::unique_lock<std::mutex> lck(mutex_buffer_count);
            buffer_count--;
            if(buffer_count <= max_buffer_count)
                cv_input_full.notify_all();
        }

        void set_max_buffer_count(unsigned int max_buffer_c)
        {
            std::unique_lock<std::mutex> lck(mutex_buffer_count);
            max_buffer_count = max_buffer_c;
        }

        unsigned int get_max_buffer_count()
        {
            std::unique_lock<std::mutex> lck(mutex_buffer_count);
            return max_buffer_count;
        }

        void set_debug_interface(INvStreammuxDebug* a_debug_iface)
        {
            debug_iface = a_debug_iface;
        }

        SOURCE_STATE state;
        //unsigned int num_bufs_in_queue; //derive from buf_queue size

        std::vector <std::shared_ptr<QueueEntry> > queue;
        std::vector <unsigned int> event_indices;
        //std::vector <EventWrapper *> event_queue;
        unsigned int id;
        void * wrapped;
        std::mutex              mutex;
        std::mutex              mutex_buffer_count;
        unsigned int top_event_index;
        unsigned int source_id;
   private:
        /** If true, NvStreamMux helper API shall increment num_sources_idle
         * NvStreamMux API shall use this field and manage thread-safety
         */
        bool switched_to_idle;
        /** If true, NvStreamMux helper API shall decrement num_sources_idle
         * NvStreamMux API shall use this field and manage thread-safety
         */
        bool switched_to_active;
        PAD_MIME_TYPE mime_type;
        bool eos;
        INvStreammuxDebug* debug_iface;
    protected :
        friend class SourcePad;
        /** cv which shall be notified when we have input buffer in queue */
        std::condition_variable cv;
        /** cv which shall be notified when we have space left in queue */
        std::condition_variable cv_input_full;
        unsigned long frame_count;
        unsigned long buffer_count;
        unsigned long max_buffer_count;

};

class ISynchronizeBuffer
{
    public:

    virtual ~ISynchronizeBuffer() = default;
    virtual BUFFER_TS_STATUS get_synch_info(BufferWrapper* buffer) = 0;
    virtual void removing_old_buffer(BufferWrapper* buffer) = 0;
    /**
     * @brief  Returns the time by which the latest early-buffer was early
     */
    virtual NanoSecondsType get_buffer_earlyby_time() = 0;
    virtual uint64_t GetBufferRunningTime(uint64_t pts, unsigned int stream_id) = 0;
    virtual uint64_t GetCurrentRunningTime() = 0;
};

#endif
