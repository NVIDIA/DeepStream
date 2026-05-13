/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file gstnvinferserver_meta_utils.h
 *
 * @brief nvinferserver metadata utilities header file.
 *
 * This file contains the declarations of the metadata utility
 * functions that add the inference output to the metadata of the input batch.
 */

#ifndef __GSTNVINFERSERVER_META_UTILS_H__
#define __GSTNVINFERSERVER_META_UTILS_H__

#include <vector>
#include <string>

#include "infer_utils.h"
#include "infer_datatypes.h"
#include "nvdsinferserver_plugin.pb.h"
#include "infer_post_datatypes.h"
#include "nvdsmeta.h"
#include "nvdsinfer.h"

namespace dsis = nvdsinferserver;
namespace ic = nvdsinferserver::config;

namespace gstnvinferserver {

/**
 * @brief Attach detection metadata for the objects in a frame.
 *
 * This function attaches metadata for the objects detected in a frame.
 * It adds a new object metadata, NvDsObjectMeta, to the metadata of the input
 * frame. The bounding box parameters of the detected objects are scaled and
 * shifted as per the actual frame dimensions using the provided scale and
 * offset values.
 *
 * @param[inout] frameMeta     Pointer to the NvDSFrameMeta structure for the frame.
 * @param[in]    parentObj     Pointer to the NvDsObjMeta structure for the parent object
 *                             under inference. It is NULL in case of primary detector.
 * @param[in]    detection_output The detection output from the inference context.
 * @param[in]    scaleX           Horizontal scaling ratio that was applied to the input to
 *                             get network input dimensions.
 * @param[in]    scaleY           vertical scaling ratio that was applied to the input to
 *                             get network input dimensions.
 * @param[in]    offsetLeft       Left offset corresponding to the left padding applied
 *                             to the scaled input.
 * @param[in]    offsetTop        Top offset corresponding to the top padding added to
 *                             the scaled input.
 * @param[in]    roiLeft          Left pixel coordinate of the input ROI in the
 *                             parent frame.
 * @param[in]    roiTop           Top pixel coordinate of the input ROI in the parent
 *                             frame.
 * @param[in]    roiWidth         Width of the ROI in the parent frame.
 * @param[in]    roiHeight        Height of the ROI in the parent frame.
 * @param[in]    imageWidth       Width of the frames in the input batch buffer.
 * @param[in]    imageHeight      Height of the frames in the input batch buffer.
 * @param[in]    uniqueId         Unique ID of the GIE instance.
 * @param[in]    config           The configuration settings used for the GStreamer element.
 */
void attachDetectionMetadata(
    NvDsFrameMeta* frameMeta, NvDsObjectMeta* parentObj,
    const NvDsInferDetectionOutput& detection_output, float scaleX, float scaleY,
    uint32_t offsetLeft, uint32_t offsetTop, uint32_t roiLeft, uint32_t roiTop, uint32_t roiWidth,
    uint32_t roiHeight, uint32_t imageWidth, uint32_t imageHeight, uint32_t uniqueId, const ic::PluginControl& config);

/**
 * @brief Attach the classification output as NvDsClassifierMeta.
 *
 * This functions creates a new metadata of the type NvDsClassifierMeta and
 * attaches it to the object metadata or ROI metadata.
 * If processing on full frames, a new object metadata is created and attached.
 * The display text for the object metadata is updated base on the label in
 * the classification result.
 *
 * @param[inout] objMeta        Pointer to the object metadata of the inference
 *                              input.
 * @param[inout] frameMeta      Pointer to the frame metadata of the inference
 *                              input.
 * @param[inout] roiMeta        Pointer to the ROI metadata for the inference
 *                              input.
 * @param[in]    objInfo        The classification output from the inference context.
 * @param[in]    uniqueId       Unique ID of the GIE instance.
 * @param[in]    classifierType The classifier type of the nvinferserver element.
 * @param[in]    imageWidth     The frame width used as bounding box size when inferencing on full frames.
 * @param[in]    imageHeight    The frame height used as bounding box size when inferencing on full frames.
 */
void attachClassificationMetadata(
    NvDsObjectMeta* objMeta, NvDsFrameMeta* frameMeta, NvDsRoiMeta* roiMeta,
    const InferClassificationOutput& objInfo, uint32_t uniqueId, const std::string& classifierType,
    uint32_t imageWidth, uint32_t imageHeight);

/**
 * @brief Merge the object history with the new classification result.
 *
 * Given an object history, merge the new classification results with the
 * previous cached results. This can be used to improve the results of
 * classification when re-inferencing over time. Currently, the function
 * just uses the latest results.
 *
 * @param[inout] cache  The cached classification output in the object
 *                       history.
 * @param[in]     newRes New classification output for the object.
 */
void mergeClassificationOutput(
    InferClassificationOutput& cache, const InferClassificationOutput& newRes);

/**
 * @brief Attach the segmentation output as user metadata.
 *
 * This function adds the segmentation output as user metadata of the input
 * buffer. The function adds the user metadata to the ROI metadata if present.
 * Otherwise, if object metadata is present, the user data is added to object
 * metadata. Otherwise it is added to the frame metadata.
 *
 * @param[inout] objMeta          Pointer to the object metadata of the inference
 *                                input.
 * @param[inout] frameMeta        Pointer to the frame metadata of the inference
 *                                input.
 * @param[inout] roiMeta          Pointer to the ROI metadata for the inference
 *                                input.
 * @param[in] segmentation_output The segmentation output from the inference
 *                                context.
 * @param[in] buf                 Pointer to the batch buffer containing the
 *                                segmentation output. This is saved as the priv_data
 *                                in the user metadata.
 */
void attachSegmentationMetadata(
    NvDsObjectMeta* objMeta, NvDsFrameMeta* frameMeta, NvDsRoiMeta* roiMeta,
    const NvDsInferSegmentationOutput& segmentation_output, dsis::SharedIBatchBuffer& buf);

/**
 * @brief Attaches the raw tensor output to the GstBuffer as metadata.
 *
 * This function attaches the output tensors for each frame in the input batch
 * to the correspoinding ROI metadata or or object metadata or frame metadata.
 *
 * @param[inout] objMeta          Pointer to the object metadata of the inference
 *                                input.
 * @param[inout] frameMeta        Pointer to the frame metadata of the inference
 *                                input.
 * @param[inout] roiMeta          Pointer to the ROI metadata for the inference
 *                                input.
 * @param[in] uniqueId            Unique ID of the GIE instance.
 * @param[in] tensors             The batch buffer array of the output tensors.
 * @param[in] batchIdx            Index of the frame within the inference batch.
 * @param[in] inputInfo           Dimensions of the input layer for the network.
 * @param[in] maintainAspectRatio maintain_aspect_ratio configuration setting
 *                                value.
 * @param[in] symmetricPadding     configuration setting
 *                                value.
 */
void attachTensorOutputMeta(
    NvDsObjectMeta* objMeta, NvDsFrameMeta* frameMeta, NvDsRoiMeta* roiMeta, uint32_t uniqueId,
    const std::vector<dsis::SharedIBatchBuffer>& tensors, uint32_t batchIdx,
    const NvDsInferNetworkInfo& inputInfo, bool maintainAspectRatio, bool symmetricPadding);

/**
 * @brief Attach the full inference output tensors to the batch metadata.
 *
 * This functions attaches the output tensors for the entire batch as a
 * user metadata of type NVDSINFER_TENSOR_OUTPUT_META.
 *
 * @param[inout] batchMeta Pointer to the bach metadata of the inference input.
 * @param[in] uniqueId      Unique ID of the GIE
 * instance.
 * @param[in] tensors       The batch buffer array of the output tensors.
 * @param[in] inputInfo     Dimensions of the input layer for the network.
 */
void attachFullTensorOutputMeta(
    NvDsBatchMeta* batchMeta, uint32_t uniqueId,
    const std::vector<dsis::SharedIBatchBuffer>& tensors, const NvDsInferNetworkInfo& inputInfo);
}

#endif /* __GSTNVINFERSERVER_META_UTILS_H__ */
