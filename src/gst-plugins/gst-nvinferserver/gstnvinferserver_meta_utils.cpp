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

/**
 * @file gstnvinferserver_meta_utils.cpp
 *
 * @brief nvinferserver metadata utilities source file.
 *
 * This file contains the definitions of the metadata utility
 * functions that add the inference output to the metadata of the input batch.
 */

#include <string>
#include <gst/gst.h>
#include <gstnvdsinfer.h>

#include "gstnvinferserver_meta_utils.h"
#include "cuda_runtime_api.h"
#include "cuda.h"

static inline int
get_element_size (NvDsInferDataType data_type)
{
    switch (data_type) {
        case FLOAT:
            return 4;
        case HALF:
            return 2;
        case INT32:
            return 4;
        case INT8:
        case UINT8:
            return 1;
        case INT64:
            return 8;
        default:
            return 0;
    }
}

namespace gstnvinferserver {

/**
 * @brief Wrapper class for acquiring and releasing metadata lock.
 */
class MetaLock {
public:
    MetaLock(NvDsBatchMeta* batchMeta)
        : m_BatchMeta (batchMeta)
    {
        assert(batchMeta);
        nvds_acquire_meta_lock(batchMeta);
    }
    ~MetaLock() {
        if (m_BatchMeta) {
            nvds_release_meta_lock(m_BatchMeta);
        }
    }
private:
    NvDsBatchMeta* m_BatchMeta;
};

void
attachDetectionMetadata(
    NvDsFrameMeta* frameMeta, NvDsObjectMeta* parentObj,
    const NvDsInferDetectionOutput& detection_output, float scaleX, float scaleY,
    uint32_t offsetLeft, uint32_t offsetTop, uint32_t roiLeft, uint32_t roiTop,
    uint32_t roiWidth, uint32_t roiHeight, uint32_t frameWidth,
    uint32_t frameHeight, uint32_t uniqueId, const ic::PluginControl& config)
{
    static gchar font_name[] = "Serif";
    NvDsObjectMeta* objMeta = nullptr;
    NvDsBatchMeta* batchMeta = frameMeta->base_meta.batch_meta;

    MetaLock locker(batchMeta);

    const ic::PluginControl::OutputDetectionControl* detectControl = nullptr;
    if (config.has_output_control() &&
        config.output_control().has_detect_control()) {
        detectControl = &config.output_control().detect_control();
    }

    frameMeta->bInferDone = TRUE;

    /* Iterate through the inference output for one frame and attach the
     * detected bounding boxes. */
    for (guint i = 0; i < detection_output.numObjects; i++) {
        NvDsInferObject& obj = detection_output.objects[i];
        const ic::PluginControl::DetectClassFilter* filterParams = nullptr;
        if (detectControl) {
            const auto& ids = detectControl->filter_out_class_ids();
            if (!ids.empty() &&
                (std::find(ids.begin(), ids.end(), obj.classIndex) != ids.end()))
                continue;

            auto iter =
                detectControl->specific_class_filters().find(obj.classIndex);
            if (iter != detectControl->specific_class_filters().end()) {
                filterParams = &iter->second;
            } else {
                filterParams = &detectControl->default_filter();
            }
        }

        /* Scale the bounding boxes proportionally based on how the object/frame
         * was scaled during input. */
        obj.left = (obj.left - offsetLeft) / scaleX + roiLeft;
        obj.top = (obj.top - offsetTop) / scaleY + roiTop;
        obj.width /= scaleX;
        obj.height /= scaleY;

        /** Clipping the object bounding-box which lies outside the roi
         * specified by nvdspreprosess plugin. */
        if(config.infer_config().clip_object_outside_roi()) {
            if (obj.left + obj.width > roiLeft + roiWidth) {
                obj.width = roiLeft + roiWidth - obj.left;
                // Ensure width doesn't go negative due to calculation errors
                if (obj.width < 0) obj.width = 0;
            }
            if (obj.top + obj.height > roiTop + roiHeight) {
                obj.height = roiTop + roiHeight - obj.top;
                // Ensure height doesn't go negative due to calculation errors
                if (obj.height < 0) obj.height = 0;
            }
            if (obj.left < roiLeft) {
                float original_left = obj.left;
                obj.left = roiLeft;
                obj.width = obj.width - (obj.left - original_left);
                // Ensure width doesn't go negative due to calculation errors
                if (obj.width < 0) obj.width = 0;
            }
            if (obj.top < roiTop) {
                float original_top = obj.top;
                obj.top = roiTop;
                obj.height = obj.height - (obj.top - original_top);
                // Ensure height doesn't go negative due to calculation errors
                if (obj.height < 0) obj.height = 0;
            }
        }

        /* Check if the scaled box co-ordinates meet the detection filter
         * criteria. Skip the box if it does not. */
        if (filterParams) {
            if (obj.width < filterParams->bbox_filter().min_width())
                continue;
            if (obj.height < filterParams->bbox_filter().min_height())
                continue;
            if (filterParams->bbox_filter().max_width() > 0 &&
                obj.width > filterParams->bbox_filter().max_width())
                continue;
            if (filterParams->bbox_filter().max_height() > 0 &&
                obj.width > filterParams->bbox_filter().max_height())
                continue;
            if (obj.top < filterParams->roi_top_offset())
                continue;
            if (obj.top + obj.height + filterParams->roi_bottom_offset() >
                frameHeight)
                continue;
        }

        objMeta = nvds_acquire_obj_meta_from_pool(batchMeta);

        objMeta->unique_component_id = config.infer_config().unique_id();
        objMeta->confidence = obj.confidence;

        /* This is an untracked object. Set tracking_id to -1. */
        objMeta->object_id = UNTRACKED_OBJECT_ID;
        objMeta->class_id = obj.classIndex;

        NvOSD_RectParams& rect_params = objMeta->rect_params;
        NvOSD_TextParams& text_params = objMeta->text_params;

        /* Assign bounding box coordinates. */
        rect_params.left = obj.left;
        rect_params.top = obj.top;
        rect_params.width = obj.width;
        rect_params.height = obj.height;

        if (parentObj) {
            rect_params.left += parentObj->rect_params.left;
            rect_params.top += parentObj->rect_params.top;
        }

        objMeta->detector_bbox_info.org_bbox_coords.left = rect_params.left;
        objMeta->detector_bbox_info.org_bbox_coords.top = rect_params.top;
        objMeta->detector_bbox_info.org_bbox_coords.width = rect_params.width;
        objMeta->detector_bbox_info.org_bbox_coords.height = rect_params.height;

        /* Border of width 3. */
        rect_params.border_width = 3;
        if (!filterParams) {
            rect_params.has_bg_color = 0;
            rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};
        } else {
            rect_params.has_bg_color = filterParams->has_bg_color();
            if (filterParams->has_bg_color()) {
                const ic::PluginControl::Color& c = filterParams->bg_color();
                rect_params.bg_color =
                    NvOSD_ColorParams{c.r(), c.g(), c.b(), c.a()};
            }
            if (filterParams->has_border_color()) {
                const ic::PluginControl::Color& c =
                    filterParams->border_color();
                rect_params.border_color =
                    NvOSD_ColorParams{c.r(), c.g(), c.b(), c.a()};
            } else {
                rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};
            }
        }

        if (obj.label)
            g_strlcpy(objMeta->obj_label, obj.label, MAX_LABEL_SIZE);

        /* display_text requires heap allocated memory. */
        text_params.display_text = g_strdup(obj.label);
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = font_name;
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};

        nvds_add_obj_meta_to_frame(frameMeta, objMeta, parentObj);
    }
}

void
attachClassificationMetadata(
    NvDsObjectMeta* objMeta, NvDsFrameMeta* frameMeta, NvDsRoiMeta* roiMeta,
    const InferClassificationOutput& objInfo, uint32_t uniqueId, const std::string& classifierType,
    uint32_t frameWidth, uint32_t frameHeight)
{
    assert(frameMeta);
    NvDsBatchMeta* batchMeta = objMeta ? objMeta->base_meta.batch_meta
                                       : frameMeta->base_meta.batch_meta;
    if (frameMeta)
        frameMeta->bInferDone = TRUE;

    if (objInfo.attributes.size() == 0 || objInfo.label.length() == 0)
        return;

    MetaLock locker(batchMeta);

    if (!objMeta && !roiMeta) {  /* Operate on full frame. */
        /* Attach only one object in the meta since this is a full frame
         * classification. */
        objMeta = nvds_acquire_obj_meta_from_pool(batchMeta);

        /* Font to be used for label text. */
        static gchar font_name[] = "Serif";

        NvOSD_RectParams& rect_params = objMeta->rect_params;
        NvOSD_TextParams& text_params = objMeta->text_params;

        /* Assign bounding box coordinates. */
        rect_params.left = 0;
        rect_params.top = 0;
        rect_params.width = frameWidth;
        rect_params.height = frameHeight;

        /* Semi-transparent yellow background. */
        rect_params.has_bg_color = 0;
        rect_params.bg_color = (NvOSD_ColorParams){1, 1, 0, 0.4};
        /* Red border of width 6. */
        rect_params.border_width = 6;
        rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};

        objMeta->object_id = UNTRACKED_OBJECT_ID;
        objMeta->class_id = -1;

        /* display_text requires heap allocated memory. Actual string formation
         * is done later in the function. */
        text_params.display_text = g_strdup("");
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = font_name;
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};

        /* Attach the NvDsFrameMeta structure as NvDsMeta to the buffer. Pass
         * the function to be called when freeing the meta_data. */
        nvds_add_obj_meta_to_frame(frameMeta, objMeta, NULL);
    }

    std::string string_label = objInfo.label;

    /* Fill the attribute info structure for the object. */
    guint numAttrs = objInfo.attributes.size();

    NvDsClassifierMeta* classifier_meta =
        nvds_acquire_classifier_meta_from_pool(batchMeta);

    classifier_meta->unique_component_id = uniqueId;
    classifier_meta->classifier_type = classifierType.c_str();

    for (unsigned int i = 0; i < numAttrs; i++) {
        NvDsLabelInfo* label_info =
            nvds_acquire_label_info_meta_from_pool(batchMeta);
        const InferAttribute& attr = objInfo.attributes[i];
        label_info->label_id = attr.attributeIndex;
        label_info->result_class_id = attr.attributeValue;
        label_info->result_prob = attr.attributeConfidence;
        if (!attr.safeAttributeLabel.empty()) {
            g_strlcpy(
                label_info->result_label, attr.safeAttributeLabel.c_str(),
                MAX_LABEL_SIZE);
            if (objInfo.label.length() == 0)
                string_label.append(attr.safeAttributeLabel).append(" ");
        }

        nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
    }

    if (string_label.length() > 0 && objMeta) {
        gchar* temp = objMeta->text_params.display_text;
        objMeta->text_params.display_text =
            g_strconcat(temp, " ", string_label.c_str(), nullptr);
        g_free(temp);
    }
    if (roiMeta) {
        nvds_add_classifier_meta_to_roi(roiMeta, classifier_meta);
        /* if object is roi itself */
        if (objMeta) {
            nvds_add_classifier_meta_to_object(objMeta, classifier_meta);
        }
    } else if (objMeta) {
        nvds_add_classifier_meta_to_object(objMeta, classifier_meta);
    }
}

void
mergeClassificationOutput(
    InferClassificationOutput& cache, const InferClassificationOutput& newRes)
{
    /* null classification result will not be saved to object's history. */
    if (newRes.attributes.size() == 0 || newRes.label.length() == 0) return;

    cache.attributes.assign(
        newRes.attributes.begin(), newRes.attributes.end());
    cache.label.assign(newRes.label);
}

static void releaseSegmentationMeta(gpointer data, gpointer user_data) {
    NvDsUserMeta* user_meta = (NvDsUserMeta*)data;
    NvDsInferSegmentationMeta* meta =
        (NvDsInferSegmentationMeta*)user_meta->user_meta_data;
    if (meta->priv_data) {
        delete (dsis::SharedIBatchBuffer*)(meta->priv_data);
    } else {
        g_free(meta->class_map);
        g_free(meta->class_probabilities_map);
    }
    delete meta;
}

static gpointer copySegmentationMeta(gpointer data, gpointer user_data) {
    NvDsUserMeta* src_user_meta = (NvDsUserMeta*)data;
    NvDsInferSegmentationMeta* src_meta =
        (NvDsInferSegmentationMeta*)src_user_meta->user_meta_data;
    NvDsInferSegmentationMeta* meta =
        (NvDsInferSegmentationMeta*)g_malloc(sizeof(NvDsInferSegmentationMeta));

    meta->classes = src_meta->classes;
    meta->width = src_meta->width;
    meta->height = src_meta->height;
    meta->class_map = (gint*)g_memdup2(
        src_meta->class_map, meta->width * meta->height * sizeof(gint));
    meta->class_probabilities_map =
        (gfloat*)g_memdup2(src_meta->class_probabilities_map,
            meta->classes * meta->width * meta->height * sizeof(gfloat));
    meta->priv_data = NULL;

    return meta;
}

void
attachSegmentationMetadata(
    NvDsObjectMeta* objMeta, NvDsFrameMeta* frameMeta, NvDsRoiMeta* roiMeta,
    const NvDsInferSegmentationOutput& segmentation_output,
    dsis::SharedIBatchBuffer& buf)
{
    assert(frameMeta);
    NvDsBatchMeta* batchMeta = objMeta ? objMeta->base_meta.batch_meta
                                       : frameMeta->base_meta.batch_meta;
    if (frameMeta)
        frameMeta->bInferDone = TRUE;

    MetaLock locker(batchMeta);
    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batchMeta);
    NvDsInferSegmentationMeta* meta =
        (NvDsInferSegmentationMeta*)g_malloc(sizeof(NvDsInferSegmentationMeta));

    meta->classes = segmentation_output.classes;
    meta->width = segmentation_output.width;
    meta->height = segmentation_output.height;
    meta->class_map = segmentation_output.class_map;
    meta->class_probabilities_map = segmentation_output.class_probability_map;
    meta->priv_data = new dsis::SharedIBatchBuffer(buf);

    user_meta->user_meta_data = meta;
    user_meta->base_meta.meta_type = (NvDsMetaType)NVDSINFER_SEGMENTATION_META;
    user_meta->base_meta.release_func = releaseSegmentationMeta;
    user_meta->base_meta.copy_func = copySegmentationMeta;

    if (roiMeta) {
        nvds_add_user_meta_to_roi(roiMeta, user_meta);
    } else if (objMeta) {
        nvds_add_user_meta_to_obj(objMeta, user_meta);
    } else {
        assert(frameMeta);
        nvds_add_user_meta_to_frame(frameMeta, user_meta);
    }
}

/* Called when NvDsUserMeta for each frame/object is released. Reduce the
 * refcount of the mini_object by 1 and free other memory. */
static void releaseTensorOutputMeta(gpointer data, gpointer user_data) {
    NvDsUserMeta* user_meta = (NvDsUserMeta*)data;
    NvDsInferTensorMeta* meta = (NvDsInferTensorMeta*)user_meta->user_meta_data;

    if(meta->priv_data) {
        delete (std::vector<dsis::SharedIBatchBuffer>*)(meta->priv_data);
        meta->priv_data = nullptr;
    }

    g_free(meta->output_layers_info);
    delete[] meta->out_buf_ptrs_dev;
    delete[] meta->out_buf_ptrs_host;
    delete meta;
}

static gpointer
copy_tensor_output_meta (gpointer data, gpointer user_data)
{
    NvDsUserMeta *src_user_meta = (NvDsUserMeta *) data;
    NvDsInferTensorMeta *src_meta =
        (NvDsInferTensorMeta *) src_user_meta->user_meta_data;
    NvDsInferTensorMeta *tensor_output_meta =  new NvDsInferTensorMeta;

    tensor_output_meta->unique_id = src_meta->unique_id;
    tensor_output_meta->num_output_layers = src_meta->num_output_layers;
    tensor_output_meta->output_layers_info =
        (NvDsInferLayerInfo *) g_memdup2(src_meta->output_layers_info,
        src_meta->num_output_layers * sizeof(NvDsInferLayerInfo));

    tensor_output_meta->out_buf_ptrs_host =  new void *[src_meta->num_output_layers];

    tensor_output_meta->out_buf_ptrs_dev =  new void *[src_meta->num_output_layers];

    for (unsigned int i = 0; i < src_meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &src_meta->output_layers_info[i];
        info->buffer = src_meta->out_buf_ptrs_host[i];
        tensor_output_meta->out_buf_ptrs_host[i] = src_meta->out_buf_ptrs_host[i];
        tensor_output_meta->out_buf_ptrs_dev[i] = src_meta->out_buf_ptrs_dev[i];
    }

    tensor_output_meta->gpu_id = src_meta->gpu_id;
    tensor_output_meta->priv_data = new std::vector<dsis::SharedIBatchBuffer>(
        *(reinterpret_cast<std::vector<dsis::SharedIBatchBuffer>*>((src_meta->priv_data))));

  return tensor_output_meta;
}

void
attachTensorOutputMeta(
    NvDsObjectMeta* objMeta, NvDsFrameMeta* frameMeta, NvDsRoiMeta* roiMeta,
    uint32_t uniqueId, const std::vector<dsis::SharedIBatchBuffer>& tensors,
    uint32_t batchIdx, const NvDsInferNetworkInfo& inputInfo,
    bool maintainAspectRatio, bool symmetricPadding)
{
    NvDsBatchMeta* batchMeta = objMeta ? objMeta->base_meta.batch_meta
                                       : frameMeta->base_meta.batch_meta;
    assert(tensors.size());
    assert(batchIdx < tensors[0]->getBatchSize() || batchIdx == 0);

    MetaLock locker(batchMeta);

    /* Create and attach NvDsInferTensorMeta for each frame/object. Also
     * increment the refcount of GstNvInferTensorOutputObject. */
    NvDsInferTensorMeta* meta = new NvDsInferTensorMeta;
    meta->unique_id = uniqueId;
    meta->num_output_layers = tensors.size();
    meta->output_layers_info = new NvDsInferLayerInfo[tensors.size()];
    meta->out_buf_ptrs_host = new void*[meta->num_output_layers];
    meta->out_buf_ptrs_dev = new void*[meta->num_output_layers];

    meta->priv_data = new std::vector<dsis::SharedIBatchBuffer>(tensors);
    meta->network_info = inputInfo;
    int devId = -1;
    for (size_t i = 0; i < tensors.size(); i++) {
        const dsis::SharedIBatchBuffer& buf = tensors[i];
        const dsis::InferBufferDescription& desc = buf->getBufDesc();
        assert(isCpuMem(desc.memType));
        NvDsInferLayerInfo &layerInfo = meta->output_layers_info[i];
        layerInfo = toCapiLayerInfo(desc);
        if (devId < 0)
            devId = desc.devId;
        void *bufPtr = buf->getBufPtr(batchIdx);
        layerInfo.bindingIndex = i;
        layerInfo.buffer = bufPtr;
        meta->out_buf_ptrs_host[i] = bufPtr;
        meta->out_buf_ptrs_dev[i] = nullptr;
    }
    meta->gpu_id = devId;
    meta->maintain_aspect_ratio = maintainAspectRatio;
    meta->symmetric_padding = symmetricPadding;

    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batchMeta);
    user_meta->user_meta_data = meta;
    user_meta->base_meta.meta_type = (NvDsMetaType)NVDSINFER_TENSOR_OUTPUT_META;
    user_meta->base_meta.release_func = releaseTensorOutputMeta;
    user_meta->base_meta.copy_func = copy_tensor_output_meta;
    user_meta->base_meta.batch_meta = batchMeta;

    if (roiMeta) {
        nvds_add_user_meta_to_roi(roiMeta, user_meta);
        /* if object is roi itself */
        if (objMeta) {
            nvds_add_user_meta_to_obj(objMeta, user_meta);
        }
    } else if (objMeta) {
        nvds_add_user_meta_to_obj(objMeta, user_meta);
    } else {
        nvds_add_user_meta_to_frame(frameMeta, user_meta);
    }
}

void
attachFullTensorOutputMeta(
    NvDsBatchMeta* batchMeta, uint32_t uniqueId,
    const std::vector<dsis::SharedIBatchBuffer>& tensors, const NvDsInferNetworkInfo& inputInfo)
{
    assert(tensors.size());
    MetaLock locker(batchMeta);

    NvDsInferTensorMeta* meta = new NvDsInferTensorMeta;
    meta->unique_id = uniqueId;
    meta->num_output_layers = tensors.size();
    meta->output_layers_info = new NvDsInferLayerInfo[tensors.size()];
    meta->out_buf_ptrs_host = new void*[meta->num_output_layers];
    meta->out_buf_ptrs_dev = new void*[meta->num_output_layers];

    meta->priv_data = new std::vector<dsis::SharedIBatchBuffer>(tensors);
    meta->network_info = inputInfo;
    int devId = -1;
    for (size_t i = 0; i < tensors.size(); i++) {
        const dsis::SharedIBatchBuffer& buf = tensors[i];
        const dsis::InferBufferDescription& desc = buf->getBufDesc();
        assert(isCpuMem(desc.memType));
        NvDsInferLayerInfo& layerInfo = meta->output_layers_info[i];
        layerInfo = toCapiLayerInfo(desc);
        if (devId < 0)
            devId = desc.devId;
        void* bufPtr = buf->getBufPtr(0);
        layerInfo.bindingIndex = i;
        layerInfo.buffer = bufPtr;
        meta->out_buf_ptrs_host[i] = bufPtr;
        meta->out_buf_ptrs_dev[i] = nullptr;
    }
    meta->gpu_id = devId;
    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batchMeta);
    user_meta->user_meta_data = meta;
    user_meta->base_meta.meta_type = (NvDsMetaType)NVDSINFER_TENSOR_OUTPUT_META;
    user_meta->base_meta.release_func = releaseTensorOutputMeta;
    user_meta->base_meta.copy_func = copy_tensor_output_meta;
    user_meta->base_meta.batch_meta = batchMeta;
    nvds_add_user_meta_to_batch(batchMeta, user_meta);
}
}
