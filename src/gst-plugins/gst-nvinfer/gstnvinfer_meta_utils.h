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

#include "gstnvinfer.h"
#include "gstnvinfer_impl.h"

void attach_metadata_detector (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferFrame & frame, NvDsInferDetectionOutput & detection_output,
        float segmentationThreshold);

void attach_metadata_classifier (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferFrame & frame, GstNvInferObjectInfo & object_info);

void merge_classification_output (GstNvInferObjectHistory & history,
    GstNvInferObjectInfo  &new_result);

void attach_metadata_segmentation (GstNvInfer * nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferFrame & frame, NvDsInferSegmentationOutput & segmentation_output);

/* Attaches the raw tensor output to the GstBuffer as metadata. */
void attach_tensor_output_meta (GstNvInfer *nvinfer, GstMiniObject * tensor_out_object,
        GstNvInferBatch *batch, NvDsInferContextBatchOutput *batch_output);
