/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file
 * <b>NVIDIA DeepStream inference specifications </b>
 *
 * @b Description: This file defines common elements used in the API
 * exposed by the Gst-nvinfer plugin.
 */

/**
 * @defgroup  ee_nvinf  Gst-infer API Common Elements
 *
 * Defines common elements used in the API exposed by the Gst-inference plugin.
 * @ingroup NvDsInferApi
 * @{
 */

#ifndef _NVDSINFER_H_
#define _NVDSINFER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define NVDSINFER_MAX_DIMS 8

#define _DS_DEPRECATED_(STR) __attribute__ ((deprecated (STR)))

/**
 * Holds the dimensions of a layer.
 */
typedef struct
{
  /** Holds the number of dimesions in the layer.*/
  unsigned int numDims;
  /** Holds the size of the layer in each dimension. */
  unsigned int d[NVDSINFER_MAX_DIMS];
  /** Holds the number of elements in the layer, including all dimensions.*/
  unsigned int numElements;
} NvDsInferDims;

/**
 * Holds the dimensions of a three-dimensional layer.
 */
typedef struct
{
  /** Holds the channel count of the layer.*/
  unsigned int c;
  /** Holds the height of the layer.*/
  unsigned int h;
  /** Holds the width of the layer.*/
  unsigned int w;
} NvDsInferDimsCHW;

/**
 * Specifies the data type of a layer.
 */
typedef enum
{
  /** Specifies FP32 format. */
  FLOAT = 0,
  /** Specifies FP16 format. */
  HALF = 1,
  /** Specifies INT8 format. */
  INT8 = 2,
  /** Specifies INT32 format. */
  INT32 = 3,
  /** Specifies INT64 format. */
  INT64 = 4,
  /** Specifies UINT8 format. */
  UINT8 = 5
} NvDsInferDataType;

/**
 * Holds information about one layer in the model.
 */
typedef struct
{
  /** Holds the data type of the layer. */
  NvDsInferDataType dataType;
  /** Holds the dimensions of the layer. */
  union {
      NvDsInferDims inferDims;
      NvDsInferDims dims _DS_DEPRECATED_("dims is deprecated. Use inferDims instead");
  };
  /** Holds the TensorRT binding index of the layer. */
  int bindingIndex;
  /** Holds the name of the layer. */
  const char* layerName;
  /** Holds a pointer to the buffer for the layer data. */
  void *buffer;
  /** Holds a Boolean; true if the layer is an input layer,
   or false if an output layer. */
  int isInput;
} NvDsInferLayerInfo;

/**
 * Holds information about the model network.
 */
typedef struct
{
  /** Holds the input width for the model. */
  unsigned int width;
  /** Holds the input height for the model. */
  unsigned int height;
  /** Holds the number of input channels for the model. */
  unsigned int channels;
} NvDsInferNetworkInfo;

/**
 * Sets values on a @ref NvDsInferDimsCHW structure from a @ref NvDsInferDims
 * structure.
 */
#define getDimsCHWFromDims(dimsCHW,dims) \
  do { \
    (dimsCHW).c = (dims).d[0]; \
    (dimsCHW).h = (dims).d[1]; \
    (dimsCHW).w = (dims).d[2]; \
  } while (0)

#define getDimsHWCFromDims(dimsCHW,dims) \
  do { \
    (dimsCHW).h = (dims).d[0]; \
    (dimsCHW).w = (dims).d[1]; \
    (dimsCHW).c = (dims).d[2]; \
  } while (0)

/**
 * Holds information about one parsed object from a detector's output.
 */
typedef struct
{
  /** Holds the ID of the class to which the object belongs. */
  unsigned int classId;

  /** Holds the horizontal offset of the bounding box shape for the object. */
  float left;
  /** Holds the vertical offset of the object's bounding box. */
  float top;
  /** Holds the width of the object's bounding box. */
  float width;
  /** Holds the height of the object's bounding box. */
  float height;

  /** Holds the object detection confidence level; must in the range
   [0.0,1.0]. */
  float detectionConfidence;

  /** Holds the rotation angle in radians for oriented bounding boxes (OBB).
   * 0.0 indicates axis-aligned bounding box. */
  float rotation_angle;
} NvDsInferObjectDetectionInfo;

/**
 * A typedef defined to maintain backward compatibility.
 */
typedef NvDsInferObjectDetectionInfo NvDsInferParseObjectInfo;

/**
 * Holds information about one parsed object and instance mask from a detector's output.
 */
typedef struct
{
  /** Holds the ID of the class to which the object belongs. */
  unsigned int classId;

  /** Holds the horizontal offset of the bounding box shape for the object. */
  float left;
  /** Holds the vertical offset of the object's bounding box. */
  float top;
  /** Holds the width of the object's bounding box. */
  float width;
  /** Holds the height of the object's bounding box. */
  float height;

  /** Holds the object detection confidence level; must in the range
   [0.0,1.0]. */
  float detectionConfidence;

  /** Holds object segment mask */
  float *mask;
  /** Holds width of mask */
  unsigned int mask_width;
  /** Holds height of mask */
  unsigned int mask_height;
  /** Holds size of mask in bytes*/
  unsigned int mask_size;
} NvDsInferInstanceMaskInfo;

/**
 * Holds information about one classified attribute.
 */
typedef struct
{
    /** Holds the index of the attribute's label. This index corresponds to
     the order of output layers specified in the @a outputCoverageLayerNames
     vector during initialization. */
    unsigned int attributeIndex;
    /** Holds the the attribute's output value. */
    unsigned int attributeValue;
    /** Holds the attribute's confidence level. */
    float attributeConfidence;
    /** Holds a pointer to a string containing the attribute's label.
     Memory for the string must not be freed. Custom parsing functions must
     allocate strings on heap using strdup or equivalent. */
    char *attributeLabel;
} NvDsInferAttribute;

/**
 * Enum for the status codes returned by NvDsInferContext.
 */
typedef enum {
    /** NvDsInferContext operation succeeded. */
    NVDSINFER_SUCCESS = 0,
    /** Failed to configure the NvDsInferContext instance possibly due to an
     *  erroneous initialization property. */
    NVDSINFER_CONFIG_FAILED,
    /** Custom Library interface implementation failed. */
    NVDSINFER_CUSTOM_LIB_FAILED,
    /** Invalid parameters were supplied. */
    NVDSINFER_INVALID_PARAMS,
    /** Output parsing failed. */
    NVDSINFER_OUTPUT_PARSING_FAILED,
    /** CUDA error was encountered. */
    NVDSINFER_CUDA_ERROR,
    /** TensorRT interface failed. */
    NVDSINFER_TENSORRT_ERROR,
    /** Resource error was encountered. */
    NVDSINFER_RESOURCE_ERROR,
    /** Triton error was encountered. Renamed TRT-IS to Triton. */
    NVDSINFER_TRITON_ERROR,
    /** [deprecated]TRT-IS error was encountered */
    NVDSINFER_TRTIS_ERROR = NVDSINFER_TRITON_ERROR,
    /** Cuda Memory error was encountered. */
    NVDSINFER_MEM_ERROR,
    /** Unknown error was encountered. */
    NVDSINFER_UNKNOWN_ERROR
} NvDsInferStatus;

/**
 * Enum for the log levels of NvDsInferContext.
 */
typedef enum {
    NVDSINFER_LOG_ERROR = 0,
    NVDSINFER_LOG_WARNING,
    NVDSINFER_LOG_INFO,
    NVDSINFER_LOG_DEBUG,
} NvDsInferLogLevel;

/**
 * Get the string name for the status.
 *
 * @param[in] status An NvDsInferStatus value.
 * @return String name for the status. Memory is owned by the function. Callers
 *         should not free the pointer.
 */
const char* NvDsInferStatus2Str(NvDsInferStatus status);

#ifdef __cplusplus
}
#endif

/* C++ data types */
#ifdef __cplusplus

/**
 * Enum for selecting between minimum/optimal/maximum dimensions of a layer
 * in case of dynamic shape network.
 */
typedef enum
{
    kSELECTOR_MIN = 0,
    kSELECTOR_OPT,
    kSELECTOR_MAX,
    kSELECTOR_SIZE
} NvDsInferProfileSelector;

/**
 * Holds full dimensions (including batch size) for a layer.
 */
typedef struct
{
    int batchSize = 0;
    NvDsInferDims dims = {0};
} NvDsInferBatchDims;

/**
 * Extended structure for bound layer information which additionally includes
 * min/optimal/max full dimensions of a layer in case of dynamic shape.
 */
struct NvDsInferBatchDimsLayerInfo : NvDsInferLayerInfo
{
    NvDsInferBatchDims profileDims[kSELECTOR_SIZE];
};

#endif

#endif

/** @} */
