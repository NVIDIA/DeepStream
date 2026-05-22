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

#include <gst/audio/gstaudiometa.h>
#include <gst/audio/audio.h>

#include "gstnvdsmeta.h"
#include "nvdscustomusermeta.h"
#include "ds_meta.pb.h"

#define NVDS_AUDIO_METADATA 0xFEDCBA

GST_DEBUG_CATEGORY_STATIC (gst_audio_protobuf_serialization_debug_category);
#define GST_CAT_DEFAULT gst_audio_protobuf_serialization_debug_category

void *set_metadata_ptr(void);
static gpointer copy_user_audio_meta(gpointer data, gpointer user_data);
static void release_user_audio_meta(gpointer data, gpointer user_data);

void *set_metadata_ptr(const void *mem, guint mem_size)
{
    NVDS_CUSTOM_PAYLOAD *metadata = (NVDS_CUSTOM_PAYLOAD *)g_malloc0(sizeof (NVDS_CUSTOM_PAYLOAD));

    char *temp_memory = (char *) g_malloc (sizeof (char) * mem_size);

    memcpy (temp_memory, mem, mem_size);

    metadata->payloadType = NVDS_AUDIO_METADATA;
    metadata->payloadSize = mem_size;
    //metadata->payload     = (uint8_t *) mem;
    metadata->payload     = (uint8_t *) temp_memory;

    return (void *) metadata;
}

static gpointer copy_user_audio_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NVDS_CUSTOM_PAYLOAD *src_user_metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;
    NVDS_CUSTOM_PAYLOAD *dst_user_metadata = (NVDS_CUSTOM_PAYLOAD *)g_malloc0(sizeof(NVDS_CUSTOM_PAYLOAD));
    dst_user_metadata->payloadType         = src_user_metadata->payloadType;
    dst_user_metadata->payloadSize         = src_user_metadata->payloadSize;
    dst_user_metadata->payload             = (uint8_t *)g_malloc0(src_user_metadata->payloadSize);
    memcpy(dst_user_metadata->payload, src_user_metadata->payload, src_user_metadata->payloadSize * sizeof (uint8_t));
    return (gpointer)dst_user_metadata;
}

static void release_user_audio_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NVDS_CUSTOM_PAYLOAD *user_metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;
    g_free (user_metadata->payload);
    user_metadata->payload = NULL;
    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
    return;
}

extern "C" void serialize_data (GstBuffer *buf);
void serialize_data (GstBuffer *buf)
{
    GST_DEBUG_CATEGORY_INIT (gst_audio_protobuf_serialization_debug_category,
        "protobufaudioserialization", 0, "audio protobuf serialization");
    NvDsUserMeta *user_meta = NULL;
    int i = 0;
    NvDsMetaType user_meta_type = NVDS_USER_CUSTOM_META;
    std::string serialized_msg;
    serialized_msg.clear();
    {
        NvDsMetaList * l_frame = NULL;
        NvDsClassifierMetaList *l_classifier_meta = NULL;
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

        if (batch_meta)
        {
            dsmeta::nvdsbatchmeta batchmeta;
            dsmeta::nvdsbasemeta nvdsbbm = batchmeta.base_meta();
            nvdsbbm.set_meta_type(batch_meta->base_meta.meta_type);
            batchmeta.set_max_frames_in_batch(batch_meta->max_frames_in_batch); 
            batchmeta.set_num_frames_in_batch(batch_meta->num_frames_in_batch); 
            for (i = 0; i < MAX_USER_FIELDS; i++)
            {
                batchmeta.add_misc_batch_info(batch_meta->misc_batch_info[i]);
            }
            for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
            {
                NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *) (l_frame->data);

                dsmeta::nvdsaudioframemeta *framemeta = batchmeta.add_audio_frame_meta_list();
                dsmeta::nvdsbasemeta *nvdsfbm = framemeta->mutable_base_meta();
                nvdsfbm->set_meta_type (frame_meta->base_meta.meta_type);
                framemeta->set_pad_index(frame_meta->pad_index);
                framemeta->set_batch_id(frame_meta->batch_id);
                framemeta->set_frame_num(frame_meta->frame_num);
                framemeta->set_buf_pts(frame_meta->buf_pts);
                framemeta->set_ntp_timestamp(frame_meta->ntp_timestamp);
                framemeta->set_source_id(frame_meta->source_id);
                framemeta->set_num_samples_per_frame(frame_meta->num_samples_per_frame);
                framemeta->set_sample_rate(frame_meta->sample_rate);
                framemeta->set_num_channels(frame_meta->num_channels);
                framemeta->set_nvbufaudioformat(frame_meta->format);
                framemeta->set_nvbufaudiolayout(frame_meta->layout);
                framemeta->set_binferdone(frame_meta->bInferDone);
                framemeta->set_class_id(frame_meta->class_id);
                framemeta->set_confidence(frame_meta->confidence);
                framemeta->set_class_label(frame_meta->class_label);
                for (i = 0; i < MAX_USER_FIELDS; i++)
                {
                    framemeta->add_misc_frame_info(frame_meta->misc_frame_info[i]);
                }
                for (i = 0; i < MAX_RESERVED_FIELDS; i++)
                {
                    frame_meta->reserved[i] = (0xAA + i);
                    framemeta->add_reserve(frame_meta->reserved[i]);
                }


                for (l_classifier_meta = frame_meta->classifier_meta_list;
                        l_classifier_meta != NULL; l_classifier_meta = l_classifier_meta->next)
                {
                    dsmeta::nvdsclassifiermeta *ncm = framemeta->add_classifier_meta_list();
                    NvDsClassifierMeta *classifier_meta;
                    classifier_meta = (NvDsClassifierMeta *)l_classifier_meta->data;
                    dsmeta::nvdsbasemeta *classifierbasemeta = ncm->mutable_base_meta();
                    classifierbasemeta->set_meta_type (classifier_meta->base_meta.meta_type);
                    ncm->set_num_labels(classifier_meta->num_labels);
                    ncm->set_unique_component_id(classifier_meta->unique_component_id);
                    if (classifier_meta->classifier_type != NULL)
                        ncm->set_classifier_type(classifier_meta->classifier_type);

                    NvDsLabelInfoList *l_label_info_list = NULL;
                    for (l_label_info_list = classifier_meta->label_info_list;
                            l_label_info_list != NULL; l_label_info_list = l_label_info_list->next)
                    {
                        NvDsLabelInfo *label_info_meta = (NvDsLabelInfo *)l_label_info_list->data;
                        dsmeta::nvdslabelinfo *labinfo = ncm->add_label_info_list();
                        dsmeta::nvdsbasemeta *labelinfobasemeta = labinfo->mutable_base_meta();
                        labelinfobasemeta->set_meta_type (label_info_meta->base_meta.meta_type);
                        labinfo->set_num_classes(label_info_meta->num_classes);
                        labinfo->set_result_label(label_info_meta->result_label);
                        if (label_info_meta->pResult_label != NULL)
                            labinfo->set_presult_label(label_info_meta->pResult_label);
                        labinfo->set_result_class_id(label_info_meta->result_class_id);
                        labinfo->set_label_id(label_info_meta->label_id);
                        labinfo->set_result_prob(label_info_meta->result_prob);
                    }
                }
            }

            batchmeta.SerializeToString(&serialized_msg);

            user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

            user_meta->user_meta_data         = (void *)set_metadata_ptr(serialized_msg.c_str(), serialized_msg.length());
            user_meta->base_meta.meta_type    = user_meta_type;
            user_meta->base_meta.copy_func    = (NvDsMetaCopyFunc)copy_user_audio_meta;
            user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_audio_meta;

            /* We want to add entire serialized DS META as NvDsUserMeta to batch level */
            nvds_add_user_meta_to_audio_batch (batch_meta, user_meta);
        }
    }
}

extern "C" void deserialize_data (GstBuffer *buf);
void deserialize_data (GstBuffer *buf)
{
    GST_DEBUG_CATEGORY_INIT (gst_audio_protobuf_serialization_debug_category,
        "protobufaudioserialization", 0, "audio protobuf serialization");
    NvDsMetaList * l_frame         = NULL;
    NvDsUserMeta *user_meta        = NULL;
    NVDS_CUSTOM_PAYLOAD *metadata = NULL;
    NvDsUserMetaList *bMetaList = nullptr;
    int frame_count = 0, i = 0, j = 0;
    //std::string str;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    if (batch_meta == NULL)
      return;

    const gchar *clear_nvds_batch_meta = g_getenv ("CLEAR_NVDS_BATCH_META");

    if (clear_nvds_batch_meta != NULL && !strcmp(clear_nvds_batch_meta, "yes"))
    {
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *) (l_frame->data);

            frame_meta->pad_index =
            frame_meta->batch_id =
            frame_meta->frame_num =
            frame_meta->buf_pts =
            frame_meta->ntp_timestamp =
            frame_meta->source_id =
            frame_meta->num_samples_per_frame =
            frame_meta->sample_rate =
            frame_meta->num_channels = 0;
            frame_meta->format = NVBUF_AUDIO_INVALID_FORMAT;
            frame_meta->layout = NVBUF_AUDIO_INVALID_LAYOUT;
            frame_meta->bInferDone =
            frame_meta->class_id =
            frame_meta->confidence = 0;
            frame_meta->class_label[MAX_LABEL_SIZE - 1] = {};
        }
    }

    bMetaList = (NvDsUserMetaList *)batch_meta->batch_user_meta_list;
    user_meta = (NvDsUserMeta *)bMetaList->data;
    if(user_meta->base_meta.meta_type == NVDS_USER_CUSTOM_META)
    {
        metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;

        std::string str(metadata->payload, metadata->payload + metadata->payloadSize);
        dsmeta::nvdsbatchmeta batchmeta;
        batchmeta.ParseFromString (str);
        
        batch_meta->max_frames_in_batch = batchmeta.max_frames_in_batch();
        batch_meta->num_frames_in_batch = batchmeta.num_frames_in_batch();
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *) (l_frame->data);
            dsmeta::nvdsaudioframemeta framemeta = batchmeta.audio_frame_meta_list(frame_count);
            frame_meta->pad_index = framemeta.pad_index();
            frame_meta->batch_id = framemeta.batch_id();
            frame_meta->frame_num = framemeta.frame_num();
            frame_meta->buf_pts = framemeta.buf_pts();
            frame_meta->ntp_timestamp = framemeta.ntp_timestamp();
            frame_meta->source_id = framemeta.source_id();
            frame_meta->num_samples_per_frame = framemeta.num_samples_per_frame();
            frame_meta->sample_rate = framemeta.sample_rate();
            frame_meta->num_channels = framemeta.num_channels();
            frame_meta->format = (NvBufAudioFormat) framemeta.nvbufaudioformat();
            frame_meta->layout = (NvBufAudioLayout) framemeta.nvbufaudiolayout();
            frame_meta->bInferDone = framemeta.binferdone();
            frame_meta->class_id = framemeta.class_id();
            frame_meta->confidence = framemeta.confidence();
            strncpy(frame_meta->class_label, framemeta.class_label().c_str(), MAX_LABEL_SIZE - 1);
            frame_meta->class_label[MAX_LABEL_SIZE - 1] = '\0';
            for (j = 0; j < MAX_USER_FIELDS; j++)
            {
                frame_meta->misc_frame_info[j] = framemeta.misc_frame_info(j);
            }
            for (j = 0; j < MAX_RESERVED_FIELDS; j++)
            {
                frame_meta->reserved[j] = framemeta.reserve(j);
            }

            NvDsMetaList *l_classifiermeta = frame_meta->classifier_meta_list;
            NvDsClassifierMeta *nvdsclassifiermeta;
            for (j = 0; j < framemeta.classifier_meta_list_size(); j++)
            {
                nvdsclassifiermeta = (NvDsClassifierMeta *) l_classifiermeta->data;
                dsmeta::nvdsclassifiermeta nvdscm = framemeta.classifier_meta_list(j);
                nvdsclassifiermeta->num_labels = nvdscm.num_labels();
                nvdsclassifiermeta->unique_component_id = nvdscm.unique_component_id();

                if (!nvdscm.classifier_type().empty())
                    strcpy ((gchar *)nvdsclassifiermeta->classifier_type, nvdscm.classifier_type().c_str());

                NvDsMetaList *l_labelinfometa = nvdsclassifiermeta->label_info_list;
                NvDsLabelInfo *nvdslabelinfometa;
                for (i = 0; i < nvdscm.label_info_list_size(); i++)
                {
                    nvdslabelinfometa = (NvDsLabelInfo *)l_labelinfometa->data;
                    dsmeta::nvdslabelinfo nvdsli = nvdscm.label_info_list(i);
                    dsmeta::nvdsbasemeta nvdbasemeta = nvdsli.base_meta();
                    nvdslabelinfometa->base_meta.meta_type = (NvDsMetaType) nvdbasemeta.meta_type();

                    nvdslabelinfometa->num_classes     = nvdsli.num_classes();
                    nvdslabelinfometa->label_id        = nvdsli.label_id();
                    nvdslabelinfometa->result_prob     = nvdsli.result_prob();
                    nvdslabelinfometa->result_class_id = nvdsli.result_class_id();

                    strncpy(nvdslabelinfometa->result_label, nvdsli.result_label().c_str(), MAX_LABEL_SIZE - 1);
                    nvdslabelinfometa->result_label[MAX_LABEL_SIZE - 1] = '\0';

                    l_labelinfometa = l_labelinfometa->next;
                }

                l_classifiermeta = l_classifiermeta->next;
            }

            frame_count++;
        }
    }
}
