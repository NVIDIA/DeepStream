/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __INFER_POST_DATATYPES_H__
#define __INFER_POST_DATATYPES_H__

#include <string>
#include <nvdsinfer.h>

/**
 * Holds the information about one detected object.
 */
typedef struct {
    /** Offset from the left boundary of the frame. */
    float left;
    /** Offset from the top boundary of the frame. */
    float top;
    /** Object width. */
    float width;
    /** Object height. */
    float height;
    /* Index for the object class. */
    int classIndex;
    /* String label for the detected object. */
    char* label;
    /* confidence score of the detected object. */
    float confidence;
} NvDsInferObject;

/**
 * Holds the information on all objects detected by a detector network in one
 * frame.
 */
typedef struct {
    /** Array of objects. */
    NvDsInferObject* objects;
    /** Number of objects in the array. */
    unsigned int numObjects;
} NvDsInferDetectionOutput;

#ifdef __cplusplus
/**
 * Holds the information on all attributes classifed by a classifier network for
 * one frame.
 */

struct InferAttribute : NvDsInferAttribute {
    /* NvDsInferAttribute::attributeLabel would be ignored */
    std::string safeAttributeLabel;
};

typedef struct {
    /** Array of attributes. Maybe more than one depending on the number of
     * output coverage layers (multi-label classifiers) */
    std::vector<InferAttribute> attributes;
    /** String label for the classified output. */
    std::string label;
} InferClassificationOutput;

#endif

/**
 * Holds the information parsed from segmentation network output for
 * one frame.
 */
typedef struct {
    /** Width of the output. Same as network width. */
    unsigned int width;
    /** Height of the output. Same as network height. */
    unsigned int height;
    /** Number of classes supported by the network. */
    unsigned int classes;
    /** Pointer to the array for 2D pixel class map. The output for pixel (x,y)
     * will be at index (y * width + x). */
    int* class_map;
    /** Pointer to the raw array containing the probabilities. The probability
     * for class c and pixel (x,y) will be at index (c * width *height + y *
     * width + x). */
    float* class_probability_map;
} NvDsInferSegmentationOutput;

struct TritonClassParams {
    uint32_t topK = 0;
    float threshold = 0.0f;
    std::string tensorName;
};

#define INFER_SERVER_PRIVATE_BUF "@@NvInferServer"

#define INFER_SERVER_DETECTION_BUF_NAME INFER_SERVER_PRIVATE_BUF "Detections"
#define INFER_SERVER_CLASSIFICATION_BUF_NAME \
    INFER_SERVER_PRIVATE_BUF "Classfications"
#define INFER_SERVER_SEGMENTATION_BUF_NAME \
    INFER_SERVER_PRIVATE_BUF "Segmentations"

#endif
