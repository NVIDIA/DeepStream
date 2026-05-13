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

#ifndef __GSTNV_STREAMMUX_PADS_H
#define __GSTNV_STREAMMUX_PADS_H

#include "gstnvstreammux_impl.h"
#include "gst-nvmessage.h"
#include "gst-nvevent.h"
#include "gstnvstreammuxdebug.h"
#include "gstnvdsmeta.h"
#include "nvds_latency_meta_internal.h"

/** TODO : remove below undef lines
 * when new streammux is an .so
 */
#undef NVDS_VERSION
#undef NVDS_VERSION_MAJOR
#undef NVDS_VERSION_MINOR
#undef NVDS_VERSION_MICRO

void GstBatchBufferWrapper::reset_batch()
{

}

void GstBatchBufferWrapper::unref()
{
    gst_buffer_unref(gst_buffer);
}

void GstBatchBufferWrapper::unref_gst_bufs()
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
        //delete ((NvBufSurface *)(batch));
        if(((NvBufSurface *)(batch))->surfaceList != NULL)
        {
            free(((NvBufSurface *)(batch))->surfaceList);
        }
        //delete ((NvBufSurface *)(batch));
        free((NvBufSurface *)(batch));
    }
}

void GstBatchBufferWrapper::copy_meta(unsigned int id, std::shared_ptr<BufferWrapper> src_buffer, unsigned int batch_id, unsigned int frame_number,  unsigned int num_surfaces_per_frame,
        NvDsBatchMeta * dest_batch_meta, unsigned int source_id)
{
         if(!dest_batch_meta)
         {
             /** video batch buffer destination meta is provided by the caller
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
              for (unsigned int surf_cnt = 0; surf_cnt < src_batch_meta->num_frames_in_batch; surf_cnt++)
              {
                  NvDsFrameMeta *frame_meta = nvds_acquire_frame_meta_from_pool(dest_batch_meta);
                  NvDsFrameMeta *src_frame_meta = nvds_get_nth_frame_meta(src_batch_meta->frame_meta_list, surf_cnt);
                  nvds_copy_frame_meta(src_frame_meta, frame_meta);
                  /** Update the parameters in NvDsFrameMeta that
                   * could change because of re-batching */
                  frame_meta->batch_id = batch_id + surf_cnt;
                  nvds_add_frame_meta_to_batch(dest_batch_meta, frame_meta);
                  /** Add all input Gst meta from src buffer to FrameMeta->user_meta
                   * */
                  nvds_copy_gst_meta_to_frame_meta((GstBuffer*)(src_buffer->wrapped), dest_batch_meta, frame_meta);
              }
              /** copied the batch meta and made only necessary changes
               * to reflect re-batching
               */
              return;
         }

         NvBufSurface *src_surf = nullptr;
         GstMapInfo map = GST_MAP_INFO_INIT;
         if (gst_buffer_map(gst_src_buffer, &map, GST_MAP_READ)) {
           src_surf = (NvBufSurface *) map.data;
           gst_buffer_unmap(gst_src_buffer, &map);
         }

         /** source_buffer lack batch-meta to copy frame-meta from;
          * create frame-meta anew */
         for(unsigned int surf_cnt = 0; surf_cnt < num_surfaces_per_frame; surf_cnt++)
         {
              NvDsFrameMeta *frame_meta = NULL;
              frame_meta = nvds_acquire_frame_meta_from_pool(dest_batch_meta);
              frame_meta->pad_index = id;
              frame_meta->source_id = source_id;
              frame_meta->buf_pts = GST_BUFFER_PTS ((GstBuffer *)(src_buffer->wrapped));
              frame_meta->ntp_timestamp = (std::static_pointer_cast<GstBufferWrapper> (src_buffer))->ntp_ts;
              frame_meta->frame_num = frame_number + surf_cnt;
              frame_meta->batch_id = batch_id + surf_cnt;
              frame_meta->source_frame_width =
                  src_surf ? src_surf->surfaceList[surf_cnt].width : 0;
              frame_meta->source_frame_height =
                  src_surf ? src_surf->surfaceList[surf_cnt].height : 0;
              /** NOTE:
               * Per design, one NvDsFrameMeta is one surface
               * Let's say a 360d camera gives out 3 surfaces per video frame
               * captured; then:
               * For one video frame, we shall have 3 X NvDsFrameMeta
               * and each NvDsFrameMeta->num_surfaces_per_frame == 3
               */
              frame_meta->num_surfaces_per_frame = num_surfaces_per_frame;
              nvds_add_frame_meta_to_batch(dest_batch_meta, frame_meta);

              /** Add all input Gst meta from src buffer to FrameMeta->user_meta
               * */
              LOGD("calling meta copy API\n");
              nvds_copy_gst_meta_to_frame_meta((GstBuffer*)(src_buffer->wrapped), dest_batch_meta, frame_meta);
         }
}

unsigned int GstBatchBufferWrapper::copy_buf_impl(std::shared_ptr<BufferWrapper> buf, unsigned int pos)
{
    unsigned int i = 0;
    if(((NvBufSurface*)((buf.get())->raw))->numFilled + ((NvBufSurface *)batch)->numFilled > ((NvBufSurface *)batch)->batchSize)
    {
        //cannot copy the input buffer to output batch
        return 0;
    }
    for(i = 0; i < ((NvBufSurface*)((buf.get())->raw))->numFilled; i++)
    {
        memcpy ((void *)(((NvBufSurface *)batch)->surfaceList + ((NvBufSurface *)batch)->numFilled + i), (void *)(((NvBufSurface*)((buf.get())->raw))->surfaceList + i), sizeof (NvBufSurfaceParams));
        /** using the lasp NvBufSurface->gpuId as batched NvBufSurface->gpuId
         * If user tries to mux buffers from different GPUs,
         * behaviour is undefined;
         * Ideally muxer will be used to mux streams from same GPU-ID
         */
        ((NvBufSurface *)batch)->gpuId = ((NvBufSurface*)((buf.get())->raw))->gpuId;
        ((NvBufSurface *)batch)->memType = ((NvBufSurface*)((buf.get())->raw))->memType;
    }
    ((NvBufSurface *)batch)->numFilled += i;
   return i;
}


unsigned int GstBatchBufferWrapper::copy_buf(std::shared_ptr<BufferWrapper> src, unsigned int pos)
{
    unsigned int num_surfaces_copied = 0;
    if(is_raw)
    {
        printf("is_raw==true; handling unimplemented [%s:%d]\n", __func__, __LINE__);
    }
    else
    {
        num_surfaces_copied = copy_buf_impl(src, pos);
        gst_in_bufs.push_back(std::static_pointer_cast<GstBufferWrapper> (src));
    }
    return num_surfaces_copied;
}

bool GstBatchBufferWrapper::push(SourcePad * src_pad, TimePoint current_play_start, NanoSecondsType accum_dur)
{
    GstPad * gst_pad = (GstPad *)(src_pad->wrapped);
    if ( is_raw && gst_buffer_list_length (raw_batch) > 0)
    {
//                  gst_nvstreammux_push_buffers (mux, raw_batch);
    }
    else if ( !is_raw )
    {
        NanoSecondsType pts = ((Clock::now() - current_play_start) + accum_dur);
        GST_BUFFER_PTS (gst_buffer) = pts.count();
        mux->last_flow_ret = gst_pad_push(gst_pad, gst_buffer);
                 //gst_buffer_list_unref (raw_batch);
        if(mux->last_flow_ret != GST_FLOW_OK)
        {
            return false;
        }
    }
    return true;
}

void GstCommonBufferAPI::add_component_latency_metadata(std::shared_ptr<BufferWrapper> src_buffer,
        GstBuffer * gst_batch_buffer,
        NvDsBatchMeta* dest_batch_meta,
        unsigned int source_id, unsigned int pad_index,
        unsigned int frame_number)
{

    const GstMetaInfo *info = GST_REFERENCE_TIMESTAMP_META_INFO;
    GstReferenceTimestampMeta *dec_meta = NULL;
    GstCaps *reference_caps = NULL;
    const GstStructure *str;
    gint dec_frame_num = 0;
    gchar *dec_name = NULL;
    gdouble dec_in_timestamp = 0;
    gdouble dec_out_timestamp = 0;
    GstBuffer* gst_src_buffer = (GstBuffer*)(src_buffer->wrapped);
    gpointer state = NULL;
    GstMeta *gst_meta = NULL;

    /** procure the timestamp meta from v4l2decoder if available on src_buffer */
    while ((gst_meta = gst_buffer_iterate_meta (gst_src_buffer, &state)))
    {
        if (gst_meta->info->api == info->api)
        {
            dec_meta = (GstReferenceTimestampMeta *) gst_meta;
            reference_caps = dec_meta->reference;
            str = gst_caps_get_structure (reference_caps, 0);
            dec_name = (gchar *)gst_structure_get_string (str, "component_name");
            gst_structure_get_int (str, "frame_num", &dec_frame_num);
            gst_structure_get_double (str, "in_timestamp", &dec_in_timestamp);
            gst_structure_get_double (str, "out_timestamp", &dec_out_timestamp);
        }
    }

    /** attach component latency meta for decoder if available */
    if(dec_meta)
    {
        NvDsMetaCompLatency *dec_latency_metadata = NULL;
        NvDsUserMeta *user_meta = NULL;
        user_meta = nvds_set_input_system_timestamp
                     (gst_batch_buffer, dec_name);
        dec_latency_metadata = (NvDsMetaCompLatency *)user_meta->user_meta_data;
        dec_latency_metadata->in_system_timestamp = dec_in_timestamp;
        dec_latency_metadata->out_system_timestamp = dec_out_timestamp;
        dec_latency_metadata->source_id = source_id;
        dec_latency_metadata->pad_index = pad_index;
        dec_latency_metadata->frame_num = frame_number;//dec_frame_num;
    }

    /** attach component latency meta for nvstreammux module - for this src_buffer/frame */
    NvDsUserMeta *user_latency_meta = NULL;
    NvDsMetaCompLatency *latency_metadata = NULL;
    user_latency_meta = nvds_acquire_user_meta_from_pool (dest_batch_meta);
    user_latency_meta->user_meta_data = (void *)nvds_set_latency_metadata_ptr();
    user_latency_meta->base_meta.meta_type = NVDS_LATENCY_MEASUREMENT_META;
    user_latency_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_latency_meta;
    user_latency_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_latency_meta;
    latency_metadata = (NvDsMetaCompLatency *)user_latency_meta->user_meta_data;
    g_snprintf(latency_metadata->component_name, MAX_COMPONENT_LEN,
                  "nvstreammux-%s", GST_ELEMENT_NAME(mux));
    latency_metadata->in_system_timestamp = (std::static_pointer_cast<GstBufferWrapper> (src_buffer))->buffer_wrapper_creation_time;
    latency_metadata->source_id = source_id;
    latency_metadata->frame_num = frame_number;
    latency_metadata->pad_index = pad_index;
    nvds_add_user_meta_to_batch(dest_batch_meta, user_latency_meta);
}

void GstCommonBufferAPI::update_component_latency_metadata(NvDsBatchMeta* batch_meta)
{
    for(GList* nodeUserMeta = batch_meta->batch_user_meta_list; nodeUserMeta; nodeUserMeta = g_list_next(nodeUserMeta))
    {
        NvDsUserMeta* userMeta = static_cast<NvDsUserMeta*>(nodeUserMeta->data);
        if(userMeta->base_meta.meta_type == NVDS_LATENCY_MEASUREMENT_META)
        {
            NvDsMetaCompLatency *latency_metadata = static_cast<NvDsMetaCompLatency*>(userMeta->user_meta_data);
            latency_metadata->out_system_timestamp = nvds_get_current_system_timestamp();
        }
    }
}

bool GstBatchBufferWrapper::push(SourcePad * src_pad, unsigned long pts)
{
    GstPad * gst_pad = (GstPad *)(src_pad->wrapped);
    gboolean ret = FALSE;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (gst_buffer);
 
    if ( is_raw && gst_buffer_list_length (raw_batch) > 0)
    {
//                  gst_nvstreammux_push_buffers (mux, raw_batch);
    }
    else if ( !is_raw )
    {
        if(batch_meta && mux->sync_inputs) {
          pts = 0;
          for(GList* nodeFrame = batch_meta->frame_meta_list; nodeFrame; nodeFrame = g_list_next(nodeFrame)) {
            NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(nodeFrame->data);
            GstClockTime running_time = mux->helper->synch_buffer->GetBufferRunningTime(frameMeta->buf_pts, frameMeta->pad_index);
            if(running_time > pts)
              pts = running_time;
          }
        }
        else
        {
          GstClockTime running_time = pts + mux->pts_offset ;
          pts=running_time;
        }
        LOGD("PTS=%lu\n", pts);
        GST_BUFFER_PTS (gst_buffer) = pts;
        if (mux->prev_outbuf_pts == pts) {

            pts += GST_MSECOND >> 2;
            GST_BUFFER_PTS (gst_buffer) = pts;
        }
        mux->prev_outbuf_pts = pts;
        LOGD("PTS=%lu\n", pts);

        /** update the out_system_timestamp of every NvDsMetaCompLatency meta */
        if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
        {
            api.update_component_latency_metadata(batch_meta);
        }
        LOGD("DEBUGME\n");
        ret = gst_pad_push(gst_pad, gst_buffer);
        LOGD("DEBUGME\n");
        /** shall not dereference `this` object anymore in this function
         * as it might already be destroyed
         * when gst_buffer is unref'd downstream */
        if(ret != GST_FLOW_OK)
        {
            GST_ERROR_OBJECT(mux, "push failed [%d]\n", ret);
            return false;
        }
    }

    return true;

}
void GstSinkPad::push_event(SourcePad * src_pad, QueueEntry * entry)
{
     if (GST_IS_EVENT (entry->wrapped))
     {
          GstEvent *event = GST_EVENT (entry->wrapped);
          switch ((guint32)GST_EVENT_TYPE (event))
          {
              case GST_EVENT_SEGMENT:
              {
                   const GstSegment *segment = NULL;
                   GstEvent *new_event;
                   gst_event_parse_segment (event, &segment);
                   new_event = gst_nvevent_new_stream_segment (id,
                       (GstSegment*) segment);
                   element->last_flow_ret = (gst_pad_push_event ((GstPad *)(src_pad->wrapped), new_event) == TRUE) ? GST_FLOW_OK : GST_FLOW_ERROR;
              }
              break;
              case GST_EVENT_EOS:
              {
		if(cascaded_eos!=true)
		{
                   GstMessage *msg = gst_nvmessage_new_stream_eos (GST_OBJECT (element), id);
                   LOGD("sending stream eos for [%d]\n", id);
                   GstEvent *new_event = gst_event_new_sink_message ("stream-eos", msg);
                   element->last_flow_ret = (gst_pad_push_event ((GstPad *)(src_pad->wrapped), new_event) == TRUE) ? GST_FLOW_OK : GST_FLOW_ERROR;
                   new_event = gst_nvevent_new_stream_eos (id);
                   element->last_flow_ret = (gst_pad_push_event ((GstPad *)(src_pad->wrapped), new_event) == TRUE && element->last_flow_ret == GST_FLOW_OK) ? GST_FLOW_OK : GST_FLOW_ERROR;
		}
             }
             break;
             case GST_NVEVENT_STREAM_RESET:
             {
                   GstEvent *event_reset = gst_nvevent_new_stream_reset (id);
                   LOGD("Resetting stream [%d].\n", id);
                   element->last_flow_ret = (gst_pad_push_event ((GstPad *)(src_pad->wrapped), event_reset) == TRUE && element->last_flow_ret == GST_FLOW_OK) ? GST_FLOW_OK : GST_FLOW_ERROR;
             }
             break;
             case GST_EVENT_STREAM_START:
             {
                   GstEvent *new_event;
                   gchar* stream_id = NULL;
                   gst_event_parse_stream_start(event, (const gchar**)&stream_id);
                   LOGD("sending stream start for [%d] stream_id=%s\n", id, stream_id);
                   new_event = gst_nvevent_new_stream_start(id, stream_id);
                   element->last_flow_ret = (gst_pad_push_event ((GstPad *)(src_pad->wrapped), new_event) == TRUE) ? GST_FLOW_OK : GST_FLOW_ERROR;
             }
             break;
           default:
             break;
         }
	 if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS) 
	 {
	      guint stream_id = 0;
	      cascaded_eos = true;
	      gst_nvevent_parse_stream_eos(event, &stream_id);
	      LOGD("sending nvevent stream eos for [%d]\n", stream_id);
	      /** In this downstream mux instance, forwarding NVEVENT_EOS for the stream as-is */
	      gst_event_ref (event);
	      element->last_flow_ret = (gst_pad_push_event ((GstPad *)(src_pad->wrapped), event) == TRUE && element->last_flow_ret == GST_FLOW_OK) ? GST_FLOW_OK : GST_FLOW_ERROR;
	 }
         gst_event_unref (event);

     }
     if(new_add)
     {
         GstEvent * new_pad_event = gst_nvevent_new_pad_added (id);
         gst_pad_push_event ((GstPad *)(src_pad->wrapped), new_pad_event);
         new_add = false;
         //gst_event_unref (new_pad_event);
     }

}

#endif

