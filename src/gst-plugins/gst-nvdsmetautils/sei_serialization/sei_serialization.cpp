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

#include <gst/gst.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>

#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"
#include "nvdscustomusermeta.h"
#include "nvdsdummyusermeta.h"

#include "gstnvdsseimeta.h"

extern "C" void serialize_data (GstBuffer *buf);
void serialize_data (GstBuffer *buf)
{
    guint sei_mem_size = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_user_meta = NULL;
    NvDsUserMeta *user_meta = NULL;
    gchar *user_meta_data = NULL;
    //guint i = 0;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    if (batch_meta)
    {
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
            for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL; l_user_meta = l_user_meta->next)
            {
                user_meta = (NvDsUserMeta *) (l_user_meta->data);
                user_meta_data = (gchar *)user_meta->user_meta_data;

                //printf ("***** user meta ptr = %p\n", user_meta_data);

                if((user_meta->base_meta.meta_type == NVDS_DUMMY_BBOX_META))
                {
                    NVDS_CUSTOM_PAYLOAD *user_mdata = (NVDS_CUSTOM_PAYLOAD *)user_meta_data;
                    //printf ("In serializtion lib payloadtype = %d payloadSize = %d\n", user_mdata->payloadType, user_mdata->payloadSize);
                    sei_mem_size += user_mdata->payloadSize + (2 * sizeof(uint32_t));
#ifdef DEBUG_SERIALIZATION_LIB
                    {
                        guint i;
                        guint count = user_mdata->payloadSize / sizeof (faceboxes);
                        faceboxes *f = (faceboxes *)user_mdata->payload;
                        for (i=0; i<count; i++)
                        {
                            printf ("x = %f\n", f->x);
                            printf ("y = %f\n", f->y);
                            printf ("width = %f\n", f->width);
                            printf ("height = %f\n", f->height);
                            f++;
                        }
                    }
#endif
                }
            }
        }

        void *sei_mem = g_malloc0(sei_mem_size);
        uint8_t* mem_ptr = (uint8_t *)sei_mem;

        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
            for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL; l_user_meta = l_user_meta->next)
            {
                user_meta = (NvDsUserMeta *) (l_user_meta->data);
                user_meta_data = (gchar *)user_meta->user_meta_data;
                if((user_meta->base_meta.meta_type == NVDS_DUMMY_BBOX_META))
                {
                    NVDS_CUSTOM_PAYLOAD *user_mdata = (NVDS_CUSTOM_PAYLOAD *)user_meta_data;
                    memcpy (mem_ptr, &user_mdata->payloadType, sizeof (uint32_t));
                    mem_ptr += sizeof (uint32_t);
                    memcpy (mem_ptr, &user_mdata->payloadSize, sizeof (uint32_t));
                    mem_ptr += sizeof (uint32_t);
                    memcpy (mem_ptr, user_mdata->payload, user_mdata->payloadSize);
                    mem_ptr += user_mdata->payloadSize;
                }
            }
        }
        {
            GstVideoSEIMeta *video_sei_meta =
                (GstVideoSEIMeta *)gst_buffer_add_meta(
                buf, GST_VIDEO_SEI_META_INFO, NULL);
            video_sei_meta->sei_metadata_type =
                GST_USER_SEI_META;
            video_sei_meta->sei_metadata_size = sei_mem_size;
            video_sei_meta->sei_metadata_ptr = sei_mem;
        }
    }
}

extern "C" void deserialize_data (GstBuffer *buf);
void deserialize_data (GstBuffer *buf)
{
    GstVideoSEIMeta *meta =
        (GstVideoSEIMeta *) gst_buffer_get_meta (buf, GST_VIDEO_SEI_META_API_TYPE);
    if (!meta)
    {
        GST_DEBUG ("NO META RETRIEVED TO DESERIALIZE\n");
    }
    else
    {
        uint32_t total_metadata_size = meta->sei_metadata_size;
        printf ("total metadata size = %d\n", total_metadata_size);
        if (meta->sei_metadata_type == (guint)GST_USER_SEI_META)
        {
            uint8_t *ptr = (uint8_t *)meta->sei_metadata_ptr;
            while (total_metadata_size > 0)
            {
                uint32_t metadata_type = (*(ptr+3) << 24) | (*(ptr+2) << 16) | (*(ptr+1) << 8) | (*ptr);
                uint32_t metadata_size = (*((ptr+4)+3) << 24) | (*((ptr+4)+2) << 16) | (*((ptr+4)+1) << 8) | (*(ptr+4));;
                printf ("type of the metadata = %d sizeof the metadata = %d\n", metadata_type, metadata_size);
                //get the inference data here
                total_metadata_size -= (metadata_size + 8);
                ptr += (metadata_size + 8);
            }
        }
    }
}
