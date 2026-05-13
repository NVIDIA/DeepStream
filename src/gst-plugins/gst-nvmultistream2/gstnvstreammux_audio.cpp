/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gstnvstreammux_audio.h"
#include "gst-nvmessage.h"
#include "gst-nvevent.h"
#include "gstnvstreammuxdebug.h"
#include "gstnvdsmeta.h"
#include "nvds_latency_meta_internal.h"

void GstAudioBatchBufferWrapper::reset_batch()
{

}

void GstAudioBatchBufferWrapper::unref()
{
    gst_buffer_unref(gst_buffer);
}

void GstAudioBatchBufferWrapper::unref_gst_bufs()
{
    //for(auto buf : gst_in_bufs)
    for( std::vector<std::shared_ptr<GstBufferWrapper> >::iterator it = gst_in_bufs.begin();  it != gst_in_bufs.end(); )
    {
        //delete(*it);
        it = gst_in_bufs.erase(it);
        //buf->free();

    }
    if(batch != NULL)
    {
        //delete ((NvBufAudio *)(batch));
        if(((NvBufAudio *)(batch))->audioBuffers != NULL)
        {
            LOGD("audioBuffers=%p numFilled=%d batchSize=%d\n", ((NvBufAudio *)(batch))->audioBuffers,
                ((NvBufAudio *)(batch))->numFilled, ((NvBufAudio *)(batch))->batchSize);
            free(((NvBufAudio *)(batch))->audioBuffers);
        }
        //delete ((NvBufAudio *)(batch));
        free((NvBufAudio *)(batch));
    }
}

void GstAudioBatchBufferWrapper::copy_meta(unsigned int id, std::shared_ptr<BufferWrapper> src_buffer, unsigned int batch_id, unsigned int frame_number,  unsigned int num_surfaces_per_frame, NvDsBatchMeta * dest_batch_meta, unsigned int source_id)
{
         std::shared_ptr<GstBufferWrapper> gst_src_buffer_w = std::static_pointer_cast<GstBufferWrapper>(src_buffer);

         if(!dest_batch_meta)
         {
             /** audio batch buffer destination meta is provided by the caller
              * attached in gstnvstreammux.cpp before push_loop() call
              * if not provided, can not copy
              */
             return;
         }
         /** If turned ON, add the component latency metadata
          * one for each buffer;
          * NOTE: Adding only one latency meta for each frame - even if one frame
          * can have multiple surfaces in it.
          */
         if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
         {
             api.add_component_latency_metadata(src_buffer, gst_buffer, dest_batch_meta, source_id, id, frame_number);
         }
         GstBuffer* gst_src_buffer = (GstBuffer*)(src_buffer->wrapped);
         NvDsBatchMeta *src_batch_meta = gst_buffer_get_nvds_batch_meta(gst_src_buffer);
         if(src_batch_meta)
         {
              /** Input buffer does have batch meta
               * copy involved frames' frameMeta instead of creating anew */
              for (unsigned int frame_cnt = 0; frame_cnt < src_batch_meta->num_frames_in_batch; frame_cnt++)
              {
                  NvDsAudioFrameMeta *frame_meta = nvds_acquire_audio_frame_meta_from_pool(dest_batch_meta);
                  NvDsAudioFrameMeta *src_frame_meta = nvds_get_nth_audio_frame_meta(src_batch_meta->frame_meta_list, frame_cnt);
                  nvds_copy_audio_frame_meta(src_frame_meta, frame_meta);
                  /** Update the parameters in NvDsAudioFrameMeta that
                   * could change because of re-batching */
                  frame_meta->batch_id = batch_id + frame_cnt;
                  nvds_add_audio_frame_meta_to_audio_batch(dest_batch_meta, frame_meta);
                  /** Add all input Gst meta from src buffer to FrameMeta->user_meta
                   * */
                  nvds_copy_gst_meta_to_audio_frame_meta((GstBuffer*)(src_buffer->wrapped), dest_batch_meta, frame_meta);
              }
              /** copied the batch meta and made only necessary changes
               * to reflect re-batching
               */
              return;
         }

         NvBufAudio *audio_batch = nullptr;
         GstMapInfo map = GST_MAP_INFO_INIT;
         if (gst_buffer_map(gst_src_buffer, &map, GST_MAP_READ)) {
           audio_batch = (NvBufAudio *) map.data;
           gst_buffer_unmap(gst_src_buffer, &map);
         }
         (void)audio_batch;

         /** source_buffer lack batch-meta to copy frame-meta from;
          * create frame-meta anew */
         for(unsigned int frame_cnt = 0; frame_cnt < num_surfaces_per_frame; frame_cnt++)
         {
              NvDsAudioFrameMeta *frame_meta = NULL;
              frame_meta = nvds_acquire_audio_frame_meta_from_pool(dest_batch_meta);
              frame_meta->pad_index = id;
              frame_meta->source_id = source_id;
              frame_meta->buf_pts = GST_BUFFER_PTS ((GstBuffer *)(src_buffer->wrapped));
              frame_meta->ntp_timestamp = (std::static_pointer_cast<GstBufferWrapper> (src_buffer))->ntp_ts;
              frame_meta->frame_num = frame_number + frame_cnt;
              frame_meta->batch_id = batch_id + frame_cnt;
              /** NOTE:
               * Per design, one NvDsAudioFrameMeta is one surface
               * means num_surfaces_per_frame is always == 1
               * when batching raw buffers
               */

              /** Audio params */
              frame_meta->num_samples_per_frame = NVSTREAMMUX_BYTES_TO_AUDIOSAMPLESIZE(
                          gst_src_buffer_w->rawSize,
                          gst_src_buffer_w->audioParams.channels,
                          gst_src_buffer_w->audioParams.bpf);
              frame_meta->sample_rate = gst_src_buffer_w->audioParams.rate;
              frame_meta->num_channels = gst_src_buffer_w->audioParams.channels;
              frame_meta->format = gst_src_buffer_w->audioParams.format;
              frame_meta->layout = gst_src_buffer_w->audioParams.layout;
              frame_meta->ntp_timestamp = gst_src_buffer_w->audioParams.ntpTimestamp;

              nvds_add_audio_frame_meta_to_audio_batch(dest_batch_meta, frame_meta);

              /** Add all input Gst meta from src buffer to FrameMeta->user_meta
               * */
              nvds_copy_gst_meta_to_audio_frame_meta((GstBuffer*)(src_buffer->wrapped), dest_batch_meta, frame_meta);
         }
}

unsigned int GstAudioBatchBufferWrapper::copy_buf_impl(std::shared_ptr<BufferWrapper> buf, unsigned int pos)
{
    std::shared_ptr<GstBufferWrapper> gstWrapper = std::static_pointer_cast<GstBufferWrapper>(buf);
    NvBufAudioParams* audioParams = (NvBufAudioParams *)(((NvBufAudio *)batch)->audioBuffers + ((NvBufAudio *)batch)->numFilled);
    NvBufAudio* outputBatch = (NvBufAudio*)batch;
    LOGD("mux_name=%s copy_buf is_nvmm=%d stream_id=%d output batchSize=%d numFilled=%d\n", GST_ELEMENT_NAME((GstElement*)mux), gstWrapper->IsMemTypeNVMM(), gstWrapper->stream_id,
        outputBatch->batchSize, outputBatch->numFilled);

    if(!gstWrapper->IsMemTypeNVMM())
    {
        if((outputBatch->batchSize - outputBatch->numFilled)
           < 1)
        {
            /** batch size not enough to copy input buffer */
            return 0;
        }
        *audioParams = gstWrapper->audioParams;
        LOGD("Non-NVMM copied; format=%d rate=%u channels=%u layout=%d sourceId=%u bpf=%u numFilled=%d\n",
            audioParams->format, audioParams->rate, audioParams->channels, audioParams->layout, audioParams->sourceId, audioParams->bpf,
            ((NvBufAudio *)batch)->numFilled);
        audioParams->dataPtr = buf.get()->raw;
        audioParams->dataSize = buf.get()->rawSize;
        audioParams->ntpTimestamp =  gstWrapper->ntp_ts;
        audioParams->bufPts = gstWrapper->buf_pts;
        audioParams->duration = gstWrapper->duration;
        ((NvBufAudio *)batch)->numFilled += 1;
        return 1;
    }
    else
    {
        NvBufAudio* inputBatch  = (NvBufAudio*)buf.get()->raw;
        /** we need to make sure there's enough space to copy the entire
         * input audio NVMM buffer (a batched audio buffer) */
        if((outputBatch->batchSize - outputBatch->numFilled)
           < inputBatch->numFilled)
        {
            /** batch size not enough to copy input buffer */
            return 0;
        }
        LOGD("inputBatch numFilled=%d batchSize=%d\n", inputBatch->numFilled, inputBatch->batchSize);
        /** copy inputBatch to outputBatch */
        memcpy((void*)audioParams, (void*)inputBatch->audioBuffers, sizeof(NvBufAudioParams) * inputBatch->numFilled );
        #if 0
        for(int i = 0; i < inputBatch->numFilled; i++)
        {
            audioParams[i].dataPtr = malloc(inputBatch->audioBuffers[i].dataSize);
            memcpy(audioParams[i].dataPtr, inputBatch->audioBuffers[i].dataPtr, inputBatch->audioBuffers[i].dataSize);
        }
        #endif
        outputBatch->numFilled += inputBatch->numFilled;
        LOGD("NVMM copied; [details for first newly-copied buffer in batch:] "
             "format=%d rate=%u channels=%u layout=%d sourceId=%u bpf=%u numFilled=%d\n",
            audioParams->format, audioParams->rate, audioParams->channels, audioParams->layout, audioParams->sourceId, audioParams->bpf,
            ((NvBufAudio *)batch)->numFilled);
        return inputBatch->numFilled;
    }
}


unsigned int GstAudioBatchBufferWrapper::copy_buf(std::shared_ptr<BufferWrapper> src, unsigned int pos)
{
    unsigned int num_surfaces_copied = 1;
    num_surfaces_copied = copy_buf_impl(src, pos);
    if(num_surfaces_copied > 0)
    {
        gst_in_bufs.push_back(std::static_pointer_cast<GstBufferWrapper> (src));
    }
    return num_surfaces_copied;
}

bool GstAudioBatchBufferWrapper::push(SourcePad * src_pad, TimePoint current_play_start, NanoSecondsType accum_dur)
{
    GstPad * gst_pad = (GstPad *)(src_pad->wrapped);
    if ( is_raw && gst_buffer_list_length (raw_batch) > 0)
    {
//                  gst_nvstreammux_push_buffers (mux, raw_batch);
    }
    else if ( !is_raw )
    {
        LOGD("DEBUGME\n");
        NanoSecondsType pts = ((Clock::now() - current_play_start) + accum_dur);
        GST_BUFFER_PTS (gst_buffer) = pts.count();
        mux->last_flow_ret = gst_pad_push(gst_pad, gst_buffer);
                 //gst_buffer_list_unref (raw_batch);
        if(mux->last_flow_ret != GST_FLOW_OK)
        {
            LOGD("Error in push\n");
            return false;
        }
    }
    return true;
}

bool GstAudioBatchBufferWrapper::push(SourcePad * src_pad, unsigned long pts)
{
    GstPad * gst_pad = (GstPad *)(src_pad->wrapped);
    gboolean ret = FALSE;
    if ( is_raw && gst_buffer_list_length (raw_batch) > 0)
    {
//                  gst_nvstreammux_push_buffers (mux, raw_batch);
    }
    else if ( !is_raw )
    {
        GST_BUFFER_PTS (gst_buffer) = pts;
        LOGD("DEBUGME mux name=%s\n", GST_ELEMENT_NAME((GstElement*)mux));
        LOGD("gst_pad_push=%p\n", gst_buffer);
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (gst_buffer);
        /** update the out_system_timestamp of every NvDsMetaCompLatency meta */
        if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
        {
            api.update_component_latency_metadata(batch_meta);
        }
        ret = gst_pad_push(gst_pad, gst_buffer);
                 //gst_buffer_list_unref (raw_batch);
        /** shall not dereference `this` object anymore in this function
         * as it might already be destroyed
         * when gst_buffer is unref'd downstream */
        if(ret != GST_FLOW_OK)
        {
            LOGD("unable to push buffer; error=%d\n", ret);
            GST_ERROR("unable to push buffer; error=%d\n", ret);
            return false;
        }
    }

    return true;

}
