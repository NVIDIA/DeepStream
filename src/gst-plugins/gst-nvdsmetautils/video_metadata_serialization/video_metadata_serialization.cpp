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
#include "ds_meta.pb.h"

#define NVDS_VIDEO_METADATA 0xABCDEF

GST_DEBUG_CATEGORY_STATIC (gst_serialization_debug_category);
#define GST_CAT_DEFAULT gst_serialization_debug_category

void *set_metadata_ptr(void);
static gpointer copy_user_meta(gpointer data, gpointer user_data);
static void release_user_meta(gpointer data, gpointer user_data);

void *set_metadata_ptr(const void *mem, guint mem_size)
{
    NVDS_CUSTOM_PAYLOAD *metadata = (NVDS_CUSTOM_PAYLOAD *)g_malloc0(sizeof (NVDS_CUSTOM_PAYLOAD));

    char *temp_memory = (char *) g_malloc (sizeof (char) * mem_size);

    memcpy (temp_memory, mem, mem_size);

    metadata->payloadType = NVDS_VIDEO_METADATA;
    metadata->payloadSize = mem_size;
    //metadata->payload     = (uint8_t *) mem;
    metadata->payload     = (uint8_t *) temp_memory;

    return (void *) metadata;
}

static gpointer copy_user_meta(gpointer data, gpointer user_data)
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

static void release_user_meta(gpointer data, gpointer user_data)
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
    GST_DEBUG_CATEGORY_INIT (gst_serialization_debug_category,
        "serialization", 0, "serialization");
    NvDsUserMeta *user_meta = NULL;
    unsigned int i = 0;
    NvDsMetaType user_meta_type = NVDS_USER_CUSTOM_META;
    std::string serialized_msg;
    serialized_msg.clear();
    {
        NvDsMetaList * l_frame = NULL;
        NvDsMetaList * l_obj = NULL;
        NvDsMetaList * l_display_meta = NULL;
        NvDsClassifierMetaList *l_classifier_meta = NULL;
        NvDsObjectMeta *obj_meta = NULL;
        NvDsDisplayMeta *display_meta = NULL;
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
                NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

                dsmeta::nvdsframemeta *framemeta = batchmeta.add_frame_meta_list();
                dsmeta::nvdsbasemeta *nvdsfbm = framemeta->mutable_base_meta();
                nvdsfbm->set_meta_type (frame_meta->base_meta.meta_type);
                framemeta->set_pad_index(frame_meta->pad_index);
                framemeta->set_batch_id(frame_meta->batch_id);
                framemeta->set_frame_num(frame_meta->frame_num);
                framemeta->set_buf_pts(frame_meta->buf_pts);
                framemeta->set_ntp_timestamp(frame_meta->ntp_timestamp);
                framemeta->set_source_id(frame_meta->source_id);
                framemeta->set_num_surfaces_per_frame(frame_meta->num_surfaces_per_frame);
                framemeta->set_source_frame_width(frame_meta->source_frame_width);
                framemeta->set_source_frame_height(frame_meta->source_frame_height);
                framemeta->set_surface_type(frame_meta->surface_type);
                framemeta->set_surface_index(frame_meta->surface_index);
                framemeta->set_num_obj_meta(frame_meta->num_obj_meta);
                framemeta->set_buf_pts(frame_meta->buf_pts);
                framemeta->set_binferdone(frame_meta->bInferDone);
                framemeta->set_pipeline_width(frame_meta->pipeline_width);
                framemeta->set_pipeline_height(frame_meta->pipeline_height);
                for (i = 0; i < MAX_USER_FIELDS; i++)
                {
                    framemeta->add_misc_frame_info(frame_meta->misc_frame_info[i]);
                }
                for (i = 0; i < MAX_RESERVED_FIELDS; i++)
                {
                    frame_meta->reserved[i] = (0xAA + i);
                    framemeta->add_reserve(frame_meta->reserved[i]);
                }

                /*************************** OBJECT META *****************************/

                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
                {
                    dsmeta::nvdsobjectmeta *objectmeta = framemeta->add_obj_meta_list();
                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                    dsmeta::nvdsbasemeta *objbasemeta = objectmeta->mutable_base_meta();
                    objbasemeta->set_meta_type (obj_meta->base_meta.meta_type);
                    objectmeta->set_unique_component_id(obj_meta->unique_component_id);
                    objectmeta->set_class_id(obj_meta->class_id);
                    objectmeta->set_object_id(obj_meta->object_id);
                    objectmeta->set_confidence(obj_meta->confidence);
                    objectmeta->set_tracker_confidence(obj_meta->tracker_confidence);

                    for (i = 0; i < MAX_USER_FIELDS; i++)
                    {
                        objectmeta->add_misc_obj_info(obj_meta->misc_obj_info[i]);
                    }
                    for (i = 0; i < MAX_RESERVED_FIELDS; i++)
                    {
                        objectmeta->add_reserve(obj_meta->reserved[i]);
                    }

                    dsmeta::nvdscompbboxinfo *dbboxinfo = objectmeta->mutable_detector_bbox_info();
                    dsmeta::nvbboxcoords *detector_coords = dbboxinfo->mutable_org_bbox_coords();
                    detector_coords->set_left(obj_meta->detector_bbox_info.org_bbox_coords.left);
                    detector_coords->set_top(obj_meta->detector_bbox_info.org_bbox_coords.top);
                    detector_coords->set_width(obj_meta->detector_bbox_info.org_bbox_coords.width);
                    detector_coords->set_height(obj_meta->detector_bbox_info.org_bbox_coords.height);

                    dsmeta::nvdscompbboxinfo *tbboxinfo = objectmeta->mutable_tracker_bbox_info();
                    dsmeta::nvbboxcoords *tracker_coords = tbboxinfo->mutable_org_bbox_coords();
                    tracker_coords->set_left(obj_meta->tracker_bbox_info.org_bbox_coords.left);
                    tracker_coords->set_top(obj_meta->tracker_bbox_info.org_bbox_coords.top);
                    tracker_coords->set_width(obj_meta->tracker_bbox_info.org_bbox_coords.width);
                    tracker_coords->set_height(obj_meta->tracker_bbox_info.org_bbox_coords.height);

                    dsmeta::nvosdrectparams *osdrectp = objectmeta->mutable_rect_params();
                    osdrectp->set_left(obj_meta->rect_params.left);
                    osdrectp->set_top(obj_meta->rect_params.top);
                    osdrectp->set_width(obj_meta->rect_params.width);
                    osdrectp->set_height(obj_meta->rect_params.height);
                    osdrectp->set_border_width(obj_meta->rect_params.border_width);
                    osdrectp->set_has_bg_color(obj_meta->rect_params.has_bg_color);
                    osdrectp->set_has_color_info(obj_meta->rect_params.has_color_info);
                    osdrectp->set_color_id(obj_meta->rect_params.color_id);

                    dsmeta::nvosdcolorparams *osdbcolor = osdrectp->mutable_border_color();
                    osdbcolor->set_red (obj_meta->rect_params.border_color.red);
                    osdbcolor->set_green (obj_meta->rect_params.border_color.green);
                    osdbcolor->set_blue (obj_meta->rect_params.border_color.blue);
                    osdbcolor->set_alpha (obj_meta->rect_params.border_color.alpha);

                    dsmeta::nvosdcolorparams *osdbgcolor = osdrectp->mutable_bgcolor();
                    osdbgcolor->set_red (obj_meta->rect_params.bg_color.red);
                    osdbgcolor->set_green (obj_meta->rect_params.bg_color.green);
                    osdbgcolor->set_blue (obj_meta->rect_params.bg_color.blue);
                    osdbgcolor->set_alpha (obj_meta->rect_params.bg_color.alpha);
                    
                    dsmeta::nvosdmaskparams *osdmaskp = objectmeta->mutable_mask_params();

                    for (i = 0; i < obj_meta->mask_params.size; i++)
                    {
                        osdmaskp->add_data(obj_meta->mask_params.data[i]);
                    }

                    osdmaskp->set_size(obj_meta->mask_params.size);
                    osdmaskp->set_threshold(obj_meta->mask_params.threshold);
                    osdmaskp->set_width(obj_meta->mask_params.width);
                    osdmaskp->set_height(obj_meta->mask_params.height);

                    dsmeta::nvosdtextparams *osdtextp = objectmeta->mutable_text_params();

                    if (obj_meta->text_params.display_text != NULL)
                        osdtextp->set_display_text(obj_meta->text_params.display_text);
                    osdtextp->set_x_offset(obj_meta->text_params.x_offset);
                    osdtextp->set_y_offset(obj_meta->text_params.y_offset);

                    dsmeta::nvosdfontparams *fontp = osdtextp->mutable_font_params();

                    if (obj_meta->text_params.font_params.font_name != NULL)
                        fontp->set_font_name(obj_meta->text_params.font_params.font_name);
                    fontp->set_font_size(obj_meta->text_params.font_params.font_size);
                    
                    dsmeta::nvosdcolorparams *osdcp = fontp->mutable_font_color();
                    osdcp->set_red (obj_meta->text_params.font_params.font_color.red);
                    osdcp->set_green (obj_meta->text_params.font_params.font_color.green);
                    osdcp->set_blue (obj_meta->text_params.font_params.font_color.blue);
                    osdcp->set_alpha (obj_meta->text_params.font_params.font_color.alpha);
                    dsmeta::nvosdcolorparams *bgcp = osdtextp->mutable_text_bg_clr();
                    bgcp->set_red (obj_meta->text_params.text_bg_clr.red);
                    bgcp->set_green (obj_meta->text_params.text_bg_clr.green);
                    bgcp->set_blue (obj_meta->text_params.text_bg_clr.blue);
                    bgcp->set_alpha (obj_meta->text_params.text_bg_clr.alpha);
                    
                    objectmeta->set_obj_label (obj_meta->obj_label);

                    for (l_classifier_meta = obj_meta->classifier_meta_list;
                            l_classifier_meta != NULL; l_classifier_meta = l_classifier_meta->next)
                    {
                        dsmeta::nvdsclassifiermeta *ncm = objectmeta->add_classifier_meta_list();
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

                /*************************** DISPLAY META *****************************/

                for (l_display_meta = frame_meta->display_meta_list;
                        l_display_meta != NULL; l_display_meta = l_display_meta->next)
                {
                    display_meta = (NvDsDisplayMeta *) (l_display_meta->data);
                    dsmeta::nvdsdisplaymeta *displaymeta = framemeta->add_display_meta_list();

                    dsmeta::nvdsbasemeta *displaybasemeta = displaymeta->mutable_base_meta();
                    displaybasemeta->set_meta_type (display_meta->base_meta.meta_type);

                    displaymeta->set_num_rects(display_meta->num_rects);
                    displaymeta->set_num_labels(display_meta->num_labels);
                    displaymeta->set_num_lines(display_meta->num_lines);
                    displaymeta->set_num_arrows(display_meta->num_arrows);
                    displaymeta->set_num_circles(display_meta->num_circles);


                    //rects
                    for (i = 0; i < display_meta->num_rects; i++)
                    {
                        dsmeta::nvosdrectparams *disprectp = displaymeta->add_rect_params();
                        disprectp->set_left(display_meta->rect_params[i].left);
                        disprectp->set_top(display_meta->rect_params[i].top);
                        disprectp->set_width(display_meta->rect_params[i].width);
                        disprectp->set_height(display_meta->rect_params[i].height);
                        disprectp->set_border_width(display_meta->rect_params[i].border_width);
                        disprectp->set_has_bg_color(display_meta->rect_params[i].has_bg_color);
                        disprectp->set_has_color_info(display_meta->rect_params[i].has_color_info);
                        disprectp->set_color_id(display_meta->rect_params[i].color_id);

                        dsmeta::nvosdcolorparams *rectscp = disprectp->mutable_border_color();
                        rectscp->set_red (display_meta->rect_params[i].border_color.red);
                        rectscp->set_green (display_meta->rect_params[i].border_color.green);
                        rectscp->set_blue (display_meta->rect_params[i].border_color.blue);
                        rectscp->set_alpha (display_meta->rect_params[i].border_color.alpha);

                        disprectp->set_reserve(display_meta->rect_params[i].reserved);

                        dsmeta::nvosdcolorparams *rectsbcp = disprectp->mutable_bgcolor();
                        rectsbcp->set_red (display_meta->rect_params[i].bg_color.red);
                        rectsbcp->set_green (display_meta->rect_params[i].bg_color.green);
                        rectsbcp->set_blue (display_meta->rect_params[i].bg_color.blue);
                        rectsbcp->set_alpha (display_meta->rect_params[i].bg_color.alpha);
                    }

                    for (i = 0; i < display_meta->num_labels; i++)
                    {
                        //texts
                        dsmeta::nvosdtextparams *disptextp = displaymeta->add_text_params();
                        if (display_meta->text_params[i].display_text != NULL)
                            disptextp->set_display_text(display_meta->text_params[i].display_text);
                        disptextp->set_x_offset(display_meta->text_params[i].x_offset);
                        disptextp->set_y_offset(display_meta->text_params[i].y_offset);

                        dsmeta::nvosdfontparams *dispfontp = disptextp->mutable_font_params();

                        if (display_meta->text_params[i].font_params.font_name != NULL)
                            dispfontp->set_font_name(display_meta->text_params[i].font_params.font_name);
                        dispfontp->set_font_size(display_meta->text_params[i].font_params.font_size);

                        dsmeta::nvosdcolorparams *displaycp = dispfontp->mutable_font_color();
                        displaycp->set_red (display_meta->text_params[i].font_params.font_color.red);
                        displaycp->set_green (display_meta->text_params[i].font_params.font_color.green);
                        displaycp->set_blue (display_meta->text_params[i].font_params.font_color.blue);
                        displaycp->set_alpha (display_meta->text_params[i].font_params.font_color.alpha);

                        dsmeta::nvosdcolorparams *dispbgcp = disptextp->mutable_text_bg_clr();
                        dispbgcp->set_red (display_meta->text_params[i].text_bg_clr.red);
                        dispbgcp->set_green (display_meta->text_params[i].text_bg_clr.green);
                        dispbgcp->set_blue (display_meta->text_params[i].text_bg_clr.blue);
                        dispbgcp->set_alpha (display_meta->text_params[i].text_bg_clr.alpha);
                    }

                    //lines
                    for (i = 0; i < display_meta->num_lines; i++)
                    {
                        dsmeta::nvosdlineparams *displinep = displaymeta->add_line_params();
                        displinep->set_x1(display_meta->line_params[i].x1);
                        displinep->set_y1(display_meta->line_params[i].y1);
                        displinep->set_x2(display_meta->line_params[i].x2);
                        displinep->set_y2(display_meta->line_params[i].y2);
                        displinep->set_line_width(display_meta->line_params[i].line_width);
                        dsmeta::nvosdcolorparams *displinecp = displinep->mutable_line_color();
                        displinecp->set_red (display_meta->line_params[i].line_color.red);
                        displinecp->set_green (display_meta->line_params[i].line_color.green);
                        displinecp->set_blue (display_meta->line_params[i].line_color.blue);
                        displinecp->set_alpha (display_meta->line_params[i].line_color.alpha);
                    }

                    //arrows
                    for (i = 0; i < display_meta->num_arrows; i++)
                    {
                        dsmeta::nvosdarrowparams *disparrowp = displaymeta->add_arrow_params();
                        disparrowp->set_x1(display_meta->arrow_params[i].x1);
                        disparrowp->set_y1(display_meta->arrow_params[i].y1);
                        disparrowp->set_x2(display_meta->arrow_params[i].x2);
                        disparrowp->set_y2(display_meta->arrow_params[i].y2);
                        disparrowp->set_arrow_width(display_meta->arrow_params[i].arrow_width);
                        disparrowp->set_arrow_head(display_meta->arrow_params[i].arrow_head);
                        dsmeta::nvosdcolorparams *disparrowcp = disparrowp->mutable_arrow_color();
                        disparrowcp->set_red (display_meta->arrow_params[i].arrow_color.red);
                        disparrowcp->set_green (display_meta->arrow_params[i].arrow_color.green);
                        disparrowcp->set_blue (display_meta->arrow_params[i].arrow_color.blue);
                        disparrowcp->set_alpha (display_meta->arrow_params[i].arrow_color.alpha);
                    }

                    //circles
                    for (i = 0; i < display_meta->num_circles; i++)
                    {
                        dsmeta::nvosdcircleparams *dispcirclep = displaymeta->add_circle_params();
                        dispcirclep->set_xc (display_meta->circle_params[i].xc);
                        dispcirclep->set_yc (display_meta->circle_params[i].yc);
                        dispcirclep->set_radius (display_meta->circle_params[i].radius);
                        dispcirclep->set_has_bg_color (display_meta->circle_params[i].has_bg_color);
                        dsmeta::nvosdcolorparams *dispcirclecp = dispcirclep->mutable_circle_color();
                        dispcirclecp->set_red (display_meta->circle_params[i].circle_color.red);
                        dispcirclecp->set_green (display_meta->circle_params[i].circle_color.green);
                        dispcirclecp->set_blue (display_meta->circle_params[i].circle_color.blue);
                        dispcirclecp->set_alpha (display_meta->circle_params[i].circle_color.alpha);
    
                        dsmeta::nvosdcolorparams *dispcirclebcp = dispcirclep->mutable_circle_bg_color();
                        dispcirclebcp->set_red (display_meta->circle_params[i].bg_color.red);
                        dispcirclebcp->set_green (display_meta->circle_params[i].bg_color.green);
                        dispcirclebcp->set_blue (display_meta->circle_params[i].bg_color.blue);
                        dispcirclebcp->set_alpha (display_meta->circle_params[i].bg_color.alpha);
                    }
                }
            }

            batchmeta.SerializeToString(&serialized_msg);


            user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

            user_meta->user_meta_data         = (void *)set_metadata_ptr(serialized_msg.c_str(), serialized_msg.length());
            user_meta->base_meta.meta_type    = user_meta_type;
            user_meta->base_meta.copy_func    = (NvDsMetaCopyFunc)copy_user_meta;
            user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;

            /* We want to add entire serialized DS META as NvDsUserMeta to batch level */
            nvds_add_user_meta_to_batch (batch_meta, user_meta);
            GST_DEBUG ("serialized string size = %ld\n", serialized_msg.size());
        }
    }
}

extern "C" void deserialize_data (GstBuffer *buf);
void deserialize_data (GstBuffer *buf)
{
    GST_DEBUG_CATEGORY_INIT (gst_serialization_debug_category,
        "deserialization", 0, "deserialization");
    NvDsMetaList * l_frame         = NULL;
    NvDsUserMeta *user_meta        = NULL;
    NVDS_CUSTOM_PAYLOAD *metadata = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsMetaList * l_disp = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsDisplayMeta *disp_meta = NULL;
    int frame_count = 0, i = 0, j = 0, display_count = 0;
    unsigned int object_count = 0, m = 0;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    if (batch_meta == NULL)
      return;

    const gchar *clear_nvds_batch_meta = g_getenv ("CLEAR_NVDS_BATCH_META");

    if (clear_nvds_batch_meta != NULL && !strcmp(clear_nvds_batch_meta, "yes"))
    {
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
            for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
            {
                obj_meta = (NvDsObjectMeta *) (l_obj->data);
                memset (&obj_meta->rect_params, 0, sizeof (NvOSD_RectParams));
                memset (&obj_meta->text_params, 0, sizeof (NvOSD_TextParams));
            }

            for (l_disp = frame_meta->display_meta_list; l_disp != NULL; l_disp = l_disp->next)
            {
                disp_meta = (NvDsDisplayMeta *) (l_disp->data);
                disp_meta->num_rects = disp_meta->num_labels = disp_meta->num_lines = disp_meta->num_arrows = disp_meta->num_circles = 0;

                for (i = 0; i < MAX_ELEMENTS_IN_DISPLAY_META; i++)
                {
                    memset (&disp_meta->rect_params[i], 0, sizeof(NvOSD_RectParams));
                    memset (&disp_meta->text_params[i], 0, sizeof(NvOSD_TextParams));
                    memset (&disp_meta->line_params[i], 0, sizeof(NvOSD_LineParams));
                    memset (&disp_meta->arrow_params[i], 0, sizeof(NvOSD_ArrowParams));
                    memset (&disp_meta->circle_params[i], 0, sizeof(NvOSD_CircleParams));

                }
            }

            frame_meta->pad_index           =
                frame_meta->batch_id            =
                frame_meta->frame_num           =
                frame_meta->ntp_timestamp       =
                frame_meta->num_surfaces_per_frame =
                frame_meta->source_id           =
                frame_meta->source_frame_width  =
                frame_meta->source_frame_height =
                frame_meta->surface_type        =
                frame_meta->surface_index       =
                frame_meta->num_obj_meta        =
                frame_meta->bInferDone          = 0;
        }
    }


    NvDsUserMetaList *uMetaList = NULL;
    for (uMetaList = batch_meta->batch_user_meta_list; uMetaList != NULL; uMetaList = uMetaList->next)
    {
        user_meta = (NvDsUserMeta *)uMetaList->data;
        if (user_meta->base_meta.meta_type == NVDS_USER_CUSTOM_META)
        {
            break;
        }
    }
    if(user_meta->base_meta.meta_type == NVDS_USER_CUSTOM_META)
    {
        metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;

        std::string str(metadata->payload, metadata->payload + metadata->payloadSize);
        GST_DEBUG ("size of the data arrived = %ld\n", str.size());
        dsmeta::nvdsbatchmeta batchmeta;
        batchmeta.ParseFromString (str);
        
        batch_meta->max_frames_in_batch = batchmeta.max_frames_in_batch();
        batch_meta->num_frames_in_batch = batchmeta.num_frames_in_batch();
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
            dsmeta::nvdsframemeta framemeta = batchmeta.frame_meta_list(frame_count);
            frame_meta->pad_index = framemeta.pad_index();
            frame_meta->batch_id = framemeta.batch_id();
            frame_meta->frame_num = framemeta.frame_num();
            frame_meta->buf_pts = framemeta.buf_pts();
            frame_meta->ntp_timestamp = framemeta.ntp_timestamp();
            frame_meta->source_id = framemeta.source_id();
            frame_meta->num_surfaces_per_frame = framemeta.num_surfaces_per_frame();
            frame_meta->source_frame_width = framemeta.source_frame_width();
            frame_meta->source_frame_height = framemeta.source_frame_height();
            frame_meta->surface_type = framemeta.surface_type();
            frame_meta->surface_index = framemeta.surface_index();
            frame_meta->num_obj_meta = 0;//framemeta.num_obj_meta();
            frame_meta->bInferDone = framemeta.binferdone();
            frame_meta->pipeline_width = framemeta.pipeline_width();
            frame_meta->pipeline_height = framemeta.pipeline_height();
            for (j = 0; j < MAX_USER_FIELDS; j++)
            {
                frame_meta->misc_frame_info[j] = framemeta.misc_frame_info(j);
            }
            for (j = 0; j < MAX_RESERVED_FIELDS; j++)
            {
                frame_meta->reserved[j] = framemeta.reserve(j);
            }

            for (object_count = 0; object_count < framemeta.num_obj_meta(); object_count++)
            {
                obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
                dsmeta::nvdsobjectmeta objectmeta = framemeta.obj_meta_list(object_count);
                dsmeta::nvdsbasemeta nvbm = objectmeta.base_meta();
                obj_meta->base_meta.meta_type = (NvDsMetaType)nvbm.meta_type();

                obj_meta->unique_component_id = objectmeta.unique_component_id();
                obj_meta->class_id            = objectmeta.class_id();
                obj_meta->object_id           = objectmeta.object_id();
                obj_meta->confidence          = objectmeta.confidence();
                obj_meta->tracker_confidence  = objectmeta.tracker_confidence();

                dsmeta::nvdscompbboxinfo dbboxinfo   = objectmeta.detector_bbox_info();
                dsmeta::nvbboxcoords detector_coords = dbboxinfo.org_bbox_coords();

                obj_meta->detector_bbox_info.org_bbox_coords.left   = detector_coords.left();
                obj_meta->detector_bbox_info.org_bbox_coords.top    = detector_coords.top();
                obj_meta->detector_bbox_info.org_bbox_coords.height = detector_coords.height();
                obj_meta->detector_bbox_info.org_bbox_coords.width  = detector_coords.width();

                dsmeta::nvdscompbboxinfo tbboxinfo  = objectmeta.tracker_bbox_info();
                dsmeta::nvbboxcoords tracker_coords = tbboxinfo.org_bbox_coords();

                obj_meta->tracker_bbox_info.org_bbox_coords.left   = tracker_coords.left();
                obj_meta->tracker_bbox_info.org_bbox_coords.top    = tracker_coords.top();
                obj_meta->tracker_bbox_info.org_bbox_coords.height = tracker_coords.height();
                obj_meta->tracker_bbox_info.org_bbox_coords.width  = tracker_coords.width();

                dsmeta::nvosdrectparams osdrectp = objectmeta.rect_params();

                obj_meta->rect_params.left         = osdrectp.left();
                obj_meta->rect_params.top          = osdrectp.top();
                obj_meta->rect_params.width        = osdrectp.width();
                obj_meta->rect_params.height       = osdrectp.height();
                obj_meta->rect_params.border_width = osdrectp.border_width();

                dsmeta::nvosdcolorparams osdcolor        = osdrectp.border_color();
                obj_meta->rect_params.border_color.red   = osdcolor.red();
                obj_meta->rect_params.border_color.green = osdcolor.green();
                obj_meta->rect_params.border_color.blue  = osdcolor.blue();
                obj_meta->rect_params.border_color.alpha = osdcolor.alpha();

                obj_meta->rect_params.reserved       = osdrectp.reserve();
                obj_meta->rect_params.has_color_info = osdrectp.has_color_info();
                obj_meta->rect_params.color_id       = osdrectp.color_id();
                obj_meta->rect_params.has_bg_color   = osdrectp.has_bg_color();

                dsmeta::nvosdcolorparams osdbgcolor  = osdrectp.bgcolor();
                obj_meta->rect_params.bg_color.red   = osdbgcolor.red();
                obj_meta->rect_params.bg_color.green = osdbgcolor.green();
                obj_meta->rect_params.bg_color.blue  = osdbgcolor.blue();
                obj_meta->rect_params.bg_color.alpha = osdbgcolor.alpha();

                dsmeta::nvosdmaskparams osdmaskp = objectmeta.mask_params();
                obj_meta->mask_params.size       = osdmaskp.size();
                obj_meta->mask_params.threshold  = osdmaskp.threshold();
                obj_meta->mask_params.width      = osdmaskp.width();
                obj_meta->mask_params.height     = osdmaskp.height();
    
                for (m = 0; m < obj_meta->mask_params.size; m++)
                {
                    obj_meta->mask_params.data[m] = osdmaskp.data(m);
                }

                dsmeta::nvosdtextparams osdtextp = objectmeta.text_params();
                obj_meta->text_params.x_offset   = osdtextp.x_offset();
                obj_meta->text_params.y_offset   = osdtextp.y_offset();

                if (obj_meta->text_params.display_text == NULL)
                {
                    obj_meta->text_params.display_text = (char *)g_malloc(osdtextp.display_text().size());
                }
                if (!osdtextp.display_text().empty())
                    strcpy (obj_meta->text_params.display_text, osdtextp.display_text().c_str());

                dsmeta::nvosdfontparams fontp = osdtextp.font_params();

                obj_meta->text_params.font_params.font_size = fontp.font_size();
      
                if (obj_meta->text_params.font_params.font_name == NULL)
                {
                    obj_meta->text_params.font_params.font_name = (char *)g_malloc(fontp.font_size());
                }
                
                if (!fontp.font_name().empty())
                    strcpy (obj_meta->text_params.font_params.font_name, fontp.font_name().c_str());

                dsmeta::nvosdcolorparams osdcp = fontp.font_color();
                obj_meta->text_params.font_params.font_color.red   = osdcp.red();
                obj_meta->text_params.font_params.font_color.green = osdcp.green();
                obj_meta->text_params.font_params.font_color.blue  = osdcp.blue();
                obj_meta->text_params.font_params.font_color.alpha = osdcp.alpha();

                obj_meta->text_params.set_bg_clr = osdtextp.set_bg_clr();

                dsmeta::nvosdcolorparams bgcp = osdtextp.text_bg_clr();
                obj_meta->text_params.text_bg_clr.red   = bgcp.red();
                obj_meta->text_params.text_bg_clr.green = bgcp.green();
                obj_meta->text_params.text_bg_clr.blue  = bgcp.blue();
                obj_meta->text_params.text_bg_clr.alpha = bgcp.alpha();

                strncpy (obj_meta->obj_label, objectmeta.obj_label().c_str(), MAX_LABEL_SIZE - 1);
                
                for (j = 0; j < MAX_USER_FIELDS; j++)
                {
                    obj_meta->misc_obj_info[j] = objectmeta.misc_obj_info(j);
                }

                NvDsMetaList *l_classifiermeta = obj_meta->classifier_meta_list;
                NvDsClassifierMeta *nvdsclassifiermeta;
                for (j = 0; j < objectmeta.classifier_meta_list_size(); j++)
                {
                    nvdsclassifiermeta = (NvDsClassifierMeta *) l_classifiermeta->data;
                    dsmeta::nvdsclassifiermeta nvdscm = objectmeta.classifier_meta_list(j);
                    nvdsclassifiermeta->num_labels = nvdscm.num_labels();
                    nvdsclassifiermeta->unique_component_id = nvdscm.unique_component_id();

                    if (!nvdscm.classifier_type().empty())
                        strcpy ((gchar *)nvdsclassifiermeta->classifier_type, nvdscm.classifier_type().c_str());

                    NvDsMetaList *l_labelinfometa = nvdsclassifiermeta->label_info_list;
                    NvDsLabelInfo *nvdslabelinfometa;
                    for (i = 0; nvdscm.label_info_list_size(); i++)
                    {
                        nvdslabelinfometa = (NvDsLabelInfo *)l_labelinfometa->data;
                        dsmeta::nvdslabelinfo nvdsli = nvdscm.label_info_list(i);
                        dsmeta::nvdsbasemeta nvdbasemeta = nvdsli.base_meta();
                        nvdslabelinfometa->base_meta.meta_type = (NvDsMetaType) nvdbasemeta.meta_type();

                        nvdslabelinfometa->num_classes     = nvdsli.num_classes();
                        nvdslabelinfometa->label_id        = nvdsli.label_id();
                        nvdslabelinfometa->result_prob     = nvdsli.result_prob();
                        nvdslabelinfometa->result_class_id = nvdsli.result_class_id();

                        strncpy (nvdslabelinfometa->result_label, nvdsli.result_label().c_str(), MAX_LABEL_SIZE - 1);

                        l_labelinfometa = l_labelinfometa->next;
                    }

                    l_classifiermeta = l_classifiermeta->next;                    
                }


                nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
            }

            NvDsMetaList *l_frame_display_meta_list = frame_meta->display_meta_list;
            NvDsDisplayMeta *nvdsdisplaymeta;
            for (display_count = 0; display_count < framemeta.display_meta_list_size(); display_count++)
            {
                nvdsdisplaymeta = (NvDsDisplayMeta *) l_frame_display_meta_list->data;
                dsmeta::nvdsdisplaymeta nvdsdm = framemeta.display_meta_list(display_count);

                dsmeta::nvdsbasemeta nvdsbm = nvdsdm.base_meta();
                nvdsdisplaymeta->base_meta.meta_type = (NvDsMetaType) nvdsbm.meta_type();
                
                nvdsdisplaymeta->num_rects   = nvdsdm.num_rects();
                nvdsdisplaymeta->num_labels  = nvdsdm.num_labels();
                nvdsdisplaymeta->num_lines   = nvdsdm.num_lines();
                nvdsdisplaymeta->num_arrows  = nvdsdm.num_arrows();
                nvdsdisplaymeta->num_circles = nvdsdm.num_circles();


                for (i = 0; i < nvdsdm.rect_params_size(); i++)
                {
                    dsmeta::nvosdrectparams disprectp = nvdsdm.rect_params(i);
                    nvdsdisplaymeta->rect_params[i].top    = disprectp.top();
                    nvdsdisplaymeta->rect_params[i].left   = disprectp.left();
                    nvdsdisplaymeta->rect_params[i].width  = disprectp.width();
                    nvdsdisplaymeta->rect_params[i].height = disprectp.height();
                    nvdsdisplaymeta->rect_params[i].border_width = disprectp.border_width();

                    dsmeta::nvosdcolorparams bcp = disprectp.border_color();
                    nvdsdisplaymeta->rect_params[i].border_color.red    = bcp.red(); 
                    nvdsdisplaymeta->rect_params[i].border_color.green  = bcp.green(); 
                    nvdsdisplaymeta->rect_params[i].border_color.blue   = bcp.blue(); 
                    nvdsdisplaymeta->rect_params[i].border_color.alpha  = bcp.alpha(); 

                    nvdsdisplaymeta->rect_params[i].has_bg_color = disprectp.has_bg_color();
                    nvdsdisplaymeta->rect_params[i].has_color_info = disprectp.has_color_info();
                    nvdsdisplaymeta->rect_params[i].color_id = disprectp.color_id();
                    nvdsdisplaymeta->rect_params[i].reserved = disprectp.reserve();

                    dsmeta::nvosdcolorparams bgcp = disprectp.bgcolor();

                    nvdsdisplaymeta->rect_params[i].bg_color.red    = bgcp.red(); 
                    nvdsdisplaymeta->rect_params[i].bg_color.green  = bgcp.green(); 
                    nvdsdisplaymeta->rect_params[i].bg_color.blue   = bgcp.blue(); 
                    nvdsdisplaymeta->rect_params[i].bg_color.alpha  = bgcp.alpha(); 
                }

                for (i = 0; i < nvdsdm.text_params_size(); i++)
                {
                    dsmeta::nvosdtextparams disptextp = nvdsdm.text_params(i);

                    if (nvdsdisplaymeta->text_params[i].display_text == NULL)
                    {
                        nvdsdisplaymeta->text_params[i].display_text = (char *)g_malloc(disptextp.display_text().size());
                    }
                    if (!disptextp.display_text().empty())
                        strcpy (nvdsdisplaymeta->text_params[i].display_text,disptextp.display_text().c_str());
                    nvdsdisplaymeta->text_params[i].x_offset = disptextp.x_offset();
                    nvdsdisplaymeta->text_params[i].y_offset = disptextp.y_offset();
                    nvdsdisplaymeta->text_params[i].set_bg_clr = disptextp.set_bg_clr();

                    dsmeta::nvosdfontparams osdfp = disptextp.font_params();
                    
                    if (nvdsdisplaymeta->text_params[i].font_params.font_name == NULL)
                    {
                        nvdsdisplaymeta->text_params[i].font_params.font_name = (char *)g_malloc(osdfp.font_size());
                    }
                    if (!osdfp.font_name().empty())
                        strcpy ((char *)nvdsdisplaymeta->text_params[i].font_params.font_name, osdfp.font_name().c_str());
                    nvdsdisplaymeta->text_params[i].font_params.font_size = osdfp.font_size();

                    dsmeta::nvosdcolorparams osdcp = osdfp.font_color();
                    nvdsdisplaymeta->text_params[i].font_params.font_color.red   = osdcp.red();
                    nvdsdisplaymeta->text_params[i].font_params.font_color.green = osdcp.green();
                    nvdsdisplaymeta->text_params[i].font_params.font_color.blue  = osdcp.blue();
                    nvdsdisplaymeta->text_params[i].font_params.font_color.alpha = osdcp.alpha();
                }

                for (i = 0; i < nvdsdm.line_params_size(); i++)
                {
                    dsmeta::nvosdlineparams displinep = nvdsdm.line_params(i);

                    nvdsdisplaymeta->line_params[i].x1 = displinep.x1();
                    nvdsdisplaymeta->line_params[i].y1 = displinep.y1();
                    nvdsdisplaymeta->line_params[i].x2 = displinep.x2();
                    nvdsdisplaymeta->line_params[i].y2 = displinep.y2();
                    nvdsdisplaymeta->line_params[i].line_width = displinep.line_width();

                    dsmeta::nvosdcolorparams osdlinecp = displinep.line_color();
                    nvdsdisplaymeta->line_params[i].line_color.red   = osdlinecp.red();
                    nvdsdisplaymeta->line_params[i].line_color.green = osdlinecp.green();
                    nvdsdisplaymeta->line_params[i].line_color.blue  = osdlinecp.blue();
                    nvdsdisplaymeta->line_params[i].line_color.alpha = osdlinecp.alpha();
                }

                for (i = 0; i < nvdsdm.arrow_params_size(); i++)
                {
                    dsmeta::nvosdarrowparams disparrowsp = nvdsdm.arrow_params(i);

                    nvdsdisplaymeta->arrow_params[i].x1 = disparrowsp.x1();
                    nvdsdisplaymeta->arrow_params[i].y1 = disparrowsp.y1();
                    nvdsdisplaymeta->arrow_params[i].x2 = disparrowsp.x2();
                    nvdsdisplaymeta->arrow_params[i].y2 = disparrowsp.y2();
                    nvdsdisplaymeta->arrow_params[i].arrow_width = disparrowsp.arrow_width();
                    nvdsdisplaymeta->arrow_params[i].arrow_head = (NvOSD_Arrow_Head_Direction) disparrowsp.arrow_head();
                    nvdsdisplaymeta->arrow_params[i].reserved = disparrowsp.reserve();

                    dsmeta::nvosdcolorparams osdarrowcp = disparrowsp.arrow_color();
                    nvdsdisplaymeta->arrow_params[i].arrow_color.red   = osdarrowcp.red();
                    nvdsdisplaymeta->arrow_params[i].arrow_color.green = osdarrowcp.green();
                    nvdsdisplaymeta->arrow_params[i].arrow_color.blue  = osdarrowcp.blue();
                    nvdsdisplaymeta->arrow_params[i].arrow_color.alpha = osdarrowcp.alpha();
                }

                for (i = 0; i < nvdsdm.circle_params_size(); i++)
                {
                    dsmeta::nvosdcircleparams dispcirclep = nvdsdm.circle_params(i);
                    nvdsdisplaymeta->circle_params[i].xc = dispcirclep.xc();
                    nvdsdisplaymeta->circle_params[i].yc = dispcirclep.yc();
                    nvdsdisplaymeta->circle_params[i].radius = dispcirclep.radius();

                    dsmeta::nvosdcolorparams osdcirclecp = dispcirclep.circle_color();
                    nvdsdisplaymeta->circle_params[i].circle_color.red   = osdcirclecp.red();
                    nvdsdisplaymeta->circle_params[i].circle_color.green = osdcirclecp.green();
                    nvdsdisplaymeta->circle_params[i].circle_color.blue  = osdcirclecp.blue();
                    nvdsdisplaymeta->circle_params[i].circle_color.alpha = osdcirclecp.alpha();

                    nvdsdisplaymeta->circle_params[i].has_bg_color = dispcirclep.has_bg_color();

                    dsmeta::nvosdcolorparams osdcirclebcp = dispcirclep.circle_bg_color();
                    nvdsdisplaymeta->circle_params[i].bg_color.red   = osdcirclebcp.red();
                    nvdsdisplaymeta->circle_params[i].bg_color.green = osdcirclebcp.green();
                    nvdsdisplaymeta->circle_params[i].bg_color.blue  = osdcirclebcp.blue();
                    nvdsdisplaymeta->circle_params[i].bg_color.alpha = osdcirclebcp.alpha();

                    nvdsdisplaymeta->circle_params[i].reserved = dispcirclep.reserve();


                }

                l_frame_display_meta_list = l_frame_display_meta_list->next;

                nvds_add_display_meta_to_frame(frame_meta, nvdsdisplaymeta);
            }

            frame_count++;
        }
    }
    else
    {
        printf ("metadata type is not NVDS_USER_CUSTOM_META\n");
    }
}
