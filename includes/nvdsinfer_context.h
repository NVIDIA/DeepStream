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
 * @file nvdsinfer_context.h
 * <b>NVIDIA DeepStream Inference Interface </b>
 *
 * @b Description: This file specifies the DeepStream inference interface API.
 */

/**
 * @defgroup  gstreamer_nvinfer_context  Inference Interface API
 *
 * Defines the DeepStream inference interface API. In C++, defines the
 * NvDsInferContext class.
 *
 * The DeepStream inference API "NvDsInfer" provides methods to initialize and
 * deinitialize the inference engine, pre-process the input frames as required
 * by the network, and parse the output from the raw tensor buffers.
 *
 * Both C and C++ interfaces are available, with the C interface being a simple
 * wrapper over the C++ interface.
 *
 * You can create an opaque handle to an instance of the context required by
 * the API by calling the factory function createNvDsInferContext() or
 * NvDsInferContext_Create(). Both functions accept an instance of
 * @ref NvDsInferContextInitParams to initialize the context.
 * Both let you specify a logging
 * callback to get detailed information about failures and warnings.
 *
 * Initialization parameters allow you to configure the network data type,
 * network type (Detector, Classifier, or Other), preprocessing parameters
 * (mean subtraction and normalization), model-related parameters like
 * Caffe/Uff/Onnx model file paths, output layer names, etc.
 *
 * Batches of frames can be queued for inferencing, using
 * NvDsInferContext::queueInputBatch() or NvDsInferContext_QueueInputBatch().
 * The input frame memories must be accessible
 * to the GPU device configured during initialization. You can provide
 * an asynchronous callback function to return the input buffers to the caller
 * as soon as the input is consumed.
 *
 * Inference output can be dequeued using NvDsInferContext::dequeueOutputBatch()
 * or NvDsInferContext_DequeueOutputBatch(). The order of dequeued outputs
 * corresponds
 * to the input queueing order. In case of failure, the output of the batch is
 * lost. The dequeued output must be released back to the context using
 * NvDsInferContext::releaseBatchOutput() or
 * NvDsInferContext_ReleaseBatchOutput()
 * to free the associated memory and return the output layer buffers for reuse
 * by the context.
 *
 * Detectors output an array of detected objects for each frame in the batch.
 * Classifiers classify entire frames and output an array of attributes for
 * each frame. Segmentation classifies each pixel in the frame. A special
 * network type (Other) has been provided whose output layers are not
 * parsed. The caller can parse the device and host output layer buffers.
 * You can also use this network type
 * with the Gst-infer plugin to flow the output buffers as metadata.
 *
 * Other methods and functions get parsed labels from a label's
 * file and properties of all layers bound by the inference engine.
 *
 * You can extend the Gst-nvinfer API using the custom method implementations.
 * Refer to the Custom Method Implementations section for more details.
 *
 * @ingroup NvDsInferApi
 * @{
 */

/**
 * @name NvDsInferContext  DeepStream Inference Interface API
 */

#ifndef __NVDSINFER_CONTEXT_H__
#define __NVDSINFER_CONTEXT_H__

#include "nvdsinfer.h"

/** @name NvDsInferContext API common types and functions.
 *  This section describes the common types and functions for both the C and C++
 *  interfaces for the NvDsInferContext class.
 */

/** @{ */

/** Maximum length of a file path parameter. */
#define _PATH_MAX 4096

/** Defines the maximum number of channels supported by the API
 for image input layers. */
#define _MAX_CHANNELS 4

/** Defines the maximum length of string parameters. */
#define _MAX_STR_LENGTH 1024

/** Defines the maximum batch size supported by nvdsinfer. */
#define NVDSINFER_MAX_BATCH_SIZE 1024

/** Defines the minimum number of sets of output buffers that must be
 allocated. */
#define NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE 2

/**
 * Defines internal data formats used by the inference engine.
 * The NvDsInferNetworkMode is aligned with network-level precision
 *  options "--fp16","--int8","--noTF32" and "--best" for trtexec.
 * Additionally, the useStronglyTyped flag corresponds to the "--stronglyTyped"
 *  option for trtexec, enabling enhanced type inference.
 *  Please refer to
 * https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags
 */
typedef enum
{
    NvDsInferNetworkMode_FP32,
    NvDsInferNetworkMode_INT8,
    NvDsInferNetworkMode_FP16,
    NvDsInferNetworkMode_BEST
} NvDsInferNetworkMode;

/**
 * Defines network types.
 */
typedef enum
{
    /** Specifies a detector. Detectors find objects and their coordinates,
     and their classes in an input frame. */
    NvDsInferNetworkType_Detector,
    /** Specifies a classifier. Classifiers classify an entire frame into
     one of several classes. */
    NvDsInferNetworkType_Classifier,
    /** Specifies a segmentation network. A segmentation network classifies
     each pixel into one of several classes. */
    NvDsInferNetworkType_Segmentation,
    /** Specifies a instance segmentation network. A instance segmentation
     network detects objects, bounding box and mask for objects, and
     their classes in an input frame */
    NvDsInferNetworkType_InstanceSegmentation,
    /** Specifies other. Output layers of an "other" network are not parsed by
     NvDsInferContext. This is useful for networks that produce custom output.
     Output can be parsed by the NvDsInferContext client or can be combined
     with the Gst-nvinfer feature to flow output tensors as metadata. */
    NvDsInferNetworkType_Other = 100
} NvDsInferNetworkType;

/**
 * Defines color formats.
 */
typedef enum
{
    /** Specifies 24-bit interleaved R-G-B format. */
    NvDsInferFormat_RGB,
    /** Specifies 24-bit interleaved B-G-R format. */
    NvDsInferFormat_BGR,
    /** Specifies 8-bit Luma format. */
    NvDsInferFormat_GRAY,
    /** Specifies 32-bit interleaved R-G-B-A format. */
    NvDsInferFormat_RGBA,
    /** Specifies 32-bit interleaved B-G-R-x format. */
    NvDsInferFormat_BGRx,
    /** NCHW planar */
    NvDsInferFormat_Tensor,
    NvDsInferFormat_Unknown = 0xFFFFFFFF,
} NvDsInferFormat;

/**
 * Defines UFF input layer orders.
 */
typedef enum {
    NvDsInferTensorOrder_kNCHW,
    NvDsInferTensorOrder_kNHWC,
    NvDsInferTensorOrder_kNC,
    NvDsInferUffOrder_kNCHW _DS_DEPRECATED_("Use NvDsInferTensorOrder_kNCHW instead") = NvDsInferTensorOrder_kNCHW,
    NvDsInferUffOrder_kNHWC _DS_DEPRECATED_("Use NvDsInferTensorOrder_kNCWH instead") = NvDsInferTensorOrder_kNHWC,
    NvDsInferUffOrder_kNC _DS_DEPRECATED_("Use NvDsInferTensorOrder_kNC instead") = NvDsInferTensorOrder_kNC
} NvDsInferTensorOrder;

#define NvDsInferUffOrder _Pragma \
    ("GCC warning \"'NvDsInferUffOrder' macro is deprecated. Use NvDsInferTensorOrder instead.\"") \
    NvDsInferTensorOrder

/**
 * Holds detection and bounding box grouping parameters.
 */
typedef struct
{
    /** Holds the bounding box detection threshold to be applied prior
     * to clustering operation. */
    union {
    float threshold _DS_DEPRECATED_("Use preclusterThreshold instead.");
    float preClusterThreshold;
    };

    /** Hold the bounding box detection threshold to be applied post
     * clustering operation. */
    float postClusterThreshold;

    /** Holds the epsilon to control merging of overlapping boxes. Refer to OpenCV
     * groupRectangles and DBSCAN documentation for more information on epsilon. */
    float eps;
    /** Holds the minimum number of boxes in a cluster to be considered
     an object during grouping using DBSCAN. */
    int minBoxes;
    /** Holds the minimum number boxes in a cluster to be considered
     an object during grouping using OpenCV groupRectangles. */
    int groupThreshold;
    /** Minimum score in a cluster for the cluster to be considered an object
     during grouping. Different clustering may cause the algorithm
     to use different scores. */
    float minScore;
    /** IOU threshold to be used with NMS mode of clustering. */
    float nmsIOUThreshold;
    /** Number of objects with objects to be filtered in the decensding order
     * of probability */
    int topK;
} NvDsInferDetectionParams;

/**
 * Enum for clustering mode for detectors
 */
typedef enum
{
    NVDSINFER_CLUSTER_GROUP_RECTANGLES = 0,
    NVDSINFER_CLUSTER_DBSCAN,
    NVDSINFER_CLUSTER_NMS,
    NVDSINFER_CLUSTER_DBSCAN_NMS_HYBRID,
    NVDSINFER_CLUSTER_NONE
} NvDsInferClusterMode;
/**
 * Holds the initialization parameters required for the NvDsInferContext interface.
 */
typedef struct _NvDsInferContextInitParams
{
    /** Holds a unique identifier for the instance. This can be used
     to identify the instance that is generating log and error messages. */
    unsigned int uniqueID;

    /** Holds an internal data format specifier used by the inference engine. */
    NvDsInferNetworkMode networkMode;

    /** Holds the pathname of the prototxt file. */
    char protoFilePath[_PATH_MAX];
    /** Holds the pathname of the caffemodel file. */
    char modelFilePath[_PATH_MAX];
    /** Holds the pathname of the UFF model file. */
    char uffFilePath[_PATH_MAX];
    /** Holds the pathname of the ONNX model file. */
    char onnxFilePath[_PATH_MAX];
    /** Holds the pathname of the TLT encoded model file. */
    char tltEncodedModelFilePath[_PATH_MAX];

    /** Holds the pathname of the INT8 calibration file.
     Required only when using INT8 mode. */
    char int8CalibrationFilePath[_PATH_MAX];

    union {
      /** Holds the input dimensions for the model. */
      NvDsInferDimsCHW inputDims;
      /** Holds the input dimensions for the UFF model. */
      NvDsInferDimsCHW uffDimsCHW;
    } _DS_DEPRECATED_("Use inferInputDims instead.");

    /** Holds the original input order for the UFF model. */
    NvDsInferTensorOrder uffInputOrder;
    /** Holds the name of the input layer for the UFF model. */
    char uffInputBlobName[_MAX_STR_LENGTH];

    /** Holds the original input order for the network. */
    NvDsInferTensorOrder netInputOrder;

    /** Holds the string key for decoding the TLT encoded model. */
    char tltModelKey[_MAX_STR_LENGTH];

    /** Holds the pathname of the serialized model engine file.
     When using the model engine file, other parameters required for creating
     the model engine are ignored. */
    char modelEngineFilePath[_PATH_MAX];

    /** Holds the maximum number of frames to be inferred together in a batch.
     The number of input frames in a batch must be
     less than or equal to this. */
    unsigned int maxBatchSize;

    /** Holds the pathname of the labels file containing strings for the class
     labels. The labels file is optional. The file format is described in the
     custom models section of the DeepStream SDK documentation. */
    char labelsFilePath[_PATH_MAX];

    /** Holds the pathname of the mean image file (PPM format). File resolution
     must be equal to the network input resolution. */
    char meanImageFilePath[_PATH_MAX];

    /** Holds the normalization factor with which to scale the input pixels. */
    float networkScaleFactor;

    /** Holds the network input format. */
    NvDsInferFormat networkInputFormat;

    /** Holds the per-channel offsets for mean subtraction. This is
     an alternative to the mean image file. The number of offsets in the array
     must be equal to the number of input channels. */
    float offsets[_MAX_CHANNELS];
    unsigned int numOffsets;

    /** Holds the network type. */
    NvDsInferNetworkType networkType;

    /** Holds a Boolean; true if DBScan is to be used for object clustering,
     or false if OpenCV groupRectangles is to be used. */
    _DS_DEPRECATED_("Use NvDsInferClusterMode instead")
    int useDBScan;

    /** Holds the number of classes detected by a detector network. */
    unsigned int numDetectedClasses;

    /** Holds per-class detection parameters. The array's size must be equal
     to @a numDetectedClasses. */
    NvDsInferDetectionParams *perClassDetectionParams;

    /** Holds the minimum confidence threshold for the classifier to consider
     a label valid. */
    float classifierThreshold;

    float segmentationThreshold;

    /** Holds a pointer to an array of pointers to output layer names. */
    char ** outputLayerNames;
    /** Holds the number of output layer names. */
    unsigned int numOutputLayers;


    /** Holds the pathname of the library containing custom methods
     required to support the network. */
    char customLibPath[_PATH_MAX];
    /** Holds the name of the custom bounding box function
     in the custom library. */
    char customBBoxParseFuncName[_MAX_STR_LENGTH];
    /** Name of the custom classifier attribute parsing function in the custom
     *  library. */
    char customClassifierParseFuncName[_MAX_STR_LENGTH];

    /** Holds a Boolean; true if the input layer contents are to be copied to
     host memory for access by the application. */
    int copyInputToHostBuffers;

    /** Holds the ID of the GPU which is to run the inference. */
    unsigned int gpuID;

    /** Holds a Boolean; true if DLA is to be used. */
    int useDLA;
    /** Holds the ID of the DLA core to use. */
    int dlaCore;

    /** Holds the number of sets of output buffers (host and device)
     to be allocated. */
    unsigned int outputBufferPoolSize;

    /** Holds the pathname of the configuration file
     for custom network creation. This can be used to store custom properties
     required by the custom network creation function. */
    char customNetworkConfigFilePath[_PATH_MAX];

    /** Name of the custom engine creation function in the custom library. */
    char customEngineCreateFuncName[_MAX_STR_LENGTH];

    /** For model parsers supporting both implicit batch dim and full dims,
     * prefer to use implicit batch dim. By default, full dims network mode is
     * used. */
    int forceImplicitBatchDimension;

    /** Max workspace size (unit MB) that will be used as tensorrt build
     * settings for cuda engine.
     */
    unsigned int workspaceSize;

    /** Inference input dimensions for runtime engine */
    NvDsInferDimsCHW inferInputDims;

    /** Holds the type of clustering mode */
    NvDsInferClusterMode clusterMode;

    /** Holds the name of the bounding box and instance mask parse function
     in the custom library. */
    char customBBoxInstanceMaskParseFuncName[_MAX_STR_LENGTH];

    /** Can be used to specify the format and datatype for bound output layers.
     * For each layer specified the format is
     * "<layer-name>:<data-type>:<format>" */
    char ** outputIOFormats;
    /** Holds number of output IO formats specified. */
    unsigned int numOutputIOFormats;

    /** Holds the name of segmentation parse function in the custom library. */
    char customSegmentationParseFuncName[_MAX_STR_LENGTH];

    /**Can be used to specify the device type and inference precision of layers.
     * For each layer specified the format is
     * "<layer-name>:<device-type>:<precision>" */
    char ** layerDevicePrecisions;
    /** Holds number of layer device precisions specified */
    unsigned int numLayerDevicePrecisions;

    /** Holds output order for segmentation network */
    NvDsInferTensorOrder segmentationOutputOrder;

    /** Boolean flag indicating that caller will supply preprocessed tensors for
     *  inferencing. NvDsInferContext will skip preprocessing initialization steps
     *  and will not interpret network input layer dimensions.
     */
    int inputFromPreprocessedTensor;

    /** Boolean flag indicating that whether we will post processing on GPU
     *  if this flag enabled, nvinfer will return gpu buffer to prossprocessing
     *  user must write cuda post processing code
     */
    int disableOutputHostCopy;

    /** Boolean flag indicating that whether we will automatically increase
     *  bufferpool size when facing a bottleneck.
     */
    int autoIncMem;

    /** Max gpu memory that can be occupied while expanding the bufferpool.
     */
    double maxGPUMemPer;

    /** Boolean flag indicating whether or not to dump raw input tensor data.
     */
    int dumpIpTensor;
    /** Boolean flag indicating whether or not to dump raw input tensor data.
     */
    int dumpOpTensor;
    /** Boolean flag indicating whether or not to overwrite raw input tensor
     *  data provided by the user into the buffer for inference.
     */
    int overwriteIpTensor;
    /** Path to the raw input tensor data that is going to be used to overwrite
     *  the buffer.
     */
    char ipTensorFilePath[_PATH_MAX];
    /** Boolean flag indicating whether or not to overwrite raw ouput tensor
     *  data provided by the user into the buffer for inference.
     */
    int overwriteOpTensor;
    /** List of paths to the raw output tensor data that are going to be used
     *  to overwrite the different output buffers.
     */
    char ** opTensorFilePath;

    /** Boolean flag indicating whether to enable strongly typed network mode.
     *  When enabled, tensor data types are inferred from network input types
     *  and operator type specification following TensorRT type inference rules.
     *  This corresponds to the --stronglyTyped flag in trtexec.
     */
    int useStronglyTyped;

    /** Dynamic properties for TensorRT/trtexec flags.
     *  This allows adding new properties without modifying the structure.
     *  Each property is stored as a key-value pair where:
     *  - key: property name (e.g., "data_max", "init_max", "calibration_data")
     *  - value: property value as string (can be converted to appropriate type)
     */
    char** dynamicPropertyKeys;
    char** dynamicPropertyValues;
    unsigned int numDynamicProperties;

    /** Boolean flag indicating whether to run TensorRT engine warmup during
     *  initialization. When enabled, dummy inference is run for batch sizes
     *  1 through maxBatchSize to pre-warm TRT's internal CUDA kernel caches
     *  and execution graphs. This eliminates the latency penalty that occurs
     *  when the first real inference arrives with a previously unseen batch
     *  size (partial batch), which can cause persistent latency accumulation
     *  in streaming pipelines.
     */
    int warmupEngine;

} NvDsInferContextInitParams;

/**
 * Defines a callback function type for asynchronously returning
 * the input client buffers to the NvDsInferContext client.
 *
 * @param[in] data  An opaque pointer provided to the input queueing function
 *                  through NvDsInferContextBatchInput.
 */
typedef void (* NvDsInferContextReturnInputAsyncFunc) (void *data);

/**
 * Holds information about one batch to be inferred.
 */
typedef struct
{
    /** Holds a pointer to an array of pointers to input frame buffers.
     The size of the array must be at least @a numInputFrames. */
    void** inputFrames;
    /** Holds the number of input frames, i.e. the size of the batch. */
    unsigned int numInputFrames;
    /** Holds the format of the frame contents. */
    NvDsInferFormat inputFormat;
    /** Holds the pitch of the input frames, in bytes. */
    unsigned int inputPitch;
    /** Holds a callback for returning the input buffers to the client. */
    NvDsInferContextReturnInputAsyncFunc returnInputFunc;
    /** A pointer to the data to be supplied with the callback in
     @a returnInputFunc. */
    void *returnFuncData;
} NvDsInferContextBatchInput;

typedef struct
{
    NvDsInferLayerInfo *tensors;
    /** Holds the number of input tensors. */
    unsigned int numInputTensors;
    /** Holds a callback for returning the input buffers to the client. */
    NvDsInferContextReturnInputAsyncFunc returnInputFunc;
    /** A pointer to the data to be supplied with the callback in
     @a returnInputFunc. */
    void *returnFuncData;
} NvDsInferContextBatchPreprocessedInput;

/**
 * Holds information about one detected object.
 */
typedef struct
{
    /** Holds the object's offset from the left boundary of the frame. */
    float left;
    /** Holds the object's offset from the top boundary of the frame. */
    float top;
    /** Holds the object's width. */
    float width;
    /** Holds the object's height. */
    float height;
    /** Holds the index for the object's class. */
    int classIndex;
    /** Holds a pointer to a string containing a label for the object. */
    char *label;
    /* confidence score of the detected object. */
    float confidence;
    /* Instance mask information for the object. */
    float *mask;
    /** Holds width of mask */
    unsigned int mask_width;
    /** Holds height of mask */
    unsigned int mask_height;
    /** Holds size of mask in bytes*/
    unsigned int mask_size;
    /** Holds the rotation angle in radians for oriented bounding boxes (OBB).
     * 0.0 indicates axis-aligned bounding box. */
    float rotation_angle;
} NvDsInferObject;

/**
 * Holds information on all objects detected by a detector network in one
 * frame.
 */
typedef struct
{
    /** Holds a pointer to an array of objects. */
    NvDsInferObject *objects;
    /** Holds the number of objects in @a objects. */
    unsigned int numObjects;
} NvDsInferDetectionOutput;

/**
 * Holds information on all attributes classifed by a classifier network for
 * one frame.
 */
typedef struct
{
    /** Holds a pointer to an array of attributes. There may be more than
     one attribute, depending on the number of output coverage layers
     (multi-label classifiers). */
    NvDsInferAttribute *attributes;
    /** Holds the size of the @a attributes array. */
    unsigned int numAttributes;
    /** Holds a pointer to a string containing a label for the
     classified output. */
    char *label;
} NvDsInferClassificationOutput;

/**
 * Holds information parsed from segmentation network output for one frame.
 */
typedef struct
{
    /** Holds the width of the output. Same as network width. */
    unsigned int width;
    /** Holds the height of the output. Same as network height. */
    unsigned int height;
    /** Holds the number of classes supported by the network. */
    unsigned int classes;
    /** Holds a pointer to an array for the 2D pixel class map.
     The output for pixel (x,y) is at index (y*width+x). */
    int *class_map;
    /** Holds a pointer to an array containing raw probabilities.
     The probability for class @a c and pixel (x,y) is at index
     (c*width*height + y*width+x). */
    float *class_probability_map;
} NvDsInferSegmentationOutput;

/**
 * Holds the information inferred by the network on one frame.
 */
typedef struct
{
    /** Holds an output type indicating the valid member in the union
     of @a detectionOutput, @a classificationOutput, and @a  segmentationOutput.
     This is basically the network type. */
    NvDsInferNetworkType outputType;
    /** Holds a union of supported outputs. The valid member is determined by
     @a outputType. */
    union
    {
        /** Holds detector output. Valid when @a outputType is
         @ref NvDsInferNetworkType_Detector. */
        NvDsInferDetectionOutput detectionOutput;
        /** Holds classifier output. Valid when @a outputType is
         @ref NvDsInferNetworkType_Classifier. */
        NvDsInferClassificationOutput classificationOutput;
        /** Holds classifier output. Valid when @a outputType is
         @ref NvDsInferNetworkType_Classifier. */
        NvDsInferSegmentationOutput segmentationOutput;
    };
} NvDsInferFrameOutput;

/**
 * Holds the output for all of the frames in a batch (an array of frame),
 * and related buffer information.
 */
typedef struct
{
    /** Holds a pointer to an array of outputs for each frame in the batch. */
    NvDsInferFrameOutput *frames;
    /** Holds the number of elements in @a frames. */
    unsigned int numFrames;

    /** Holds a pointer to an array of pointers to output device buffers
     for this batch. The array elements are set by */
    void **outputDeviceBuffers;
    /** Holds the number of elements in @a *outputDeviceBuffers. */
    unsigned int numOutputDeviceBuffers;

    /** Holds a pointer to an array of pointers to host buffers for this batch.
     The array elements are set by */
    void **hostBuffers;
    /** Holds the number of elements in hostBuffers. */
    unsigned int numHostBuffers;

    /** Holds a private context pointer for the set of output buffers. */
    void* priv;
} NvDsInferContextBatchOutput;

/** An opaque pointer type to be used as a handle for a context instance. */
typedef struct INvDsInferContext * NvDsInferContextHandle;

/**
 * @brief  Type declaration for a logging callback.
 *
 * The callaback log NvDsInferContext messages.
 *
 * @param[in] handle        The handle of the NvDsInferContext instance that
 *                          generated the log.
 * @param[in] uniqueID      Unique ID of the NvDsInferContext instance that
 *                          generated the log.
 * @param[in] logLevel      Level of the log.
 * @param[in] funcName      A pointer to the name of the function that
 *                          generated the log.
 * @param[in] logMessage    A pointer to the log message string.
 * @param[in] userCtx       An opaque pointer to the user context, supplied
 *                          when creating the NvDsInferContext instance.
 */
typedef void (*NvDsInferContextLoggingFunc)(NvDsInferContextHandle handle,
    unsigned int uniqueID, NvDsInferLogLevel logLevel, const char* logMessage,
    void* userCtx);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Resets a context parameter structure to default values.
 *
 * @param[in] initParams    A pointer to a context parameter structure.
 */
void NvDsInferContext_ResetInitParams (NvDsInferContextInitParams *initParams);

/**
 * Dynamic property management functions for extensible TensorRT/trtexec flags
 */

/** 
 * Add or update a dynamic property 
 * @param initParams: Pointer to NvDsInferContextInitParams structure
 * @param key: Property name (e.g., "data_max", "init_max")
 * @param value: Property value as string
 * @return: 1 on success, 0 on failure
 */
int NvDsInferContext_SetDynamicProperty(NvDsInferContextInitParams *initParams, 
                                       const char *key, const char *value);

/** 
 * Get a dynamic property value 
 * @param initParams: Pointer to NvDsInferContextInitParams structure
 * @param key: Property name to retrieve
 * @return: Property value string or NULL if not found
 */
const char* NvDsInferContext_GetDynamicProperty(const NvDsInferContextInitParams *initParams, 
                                               const char *key);

/** 
 * Check if a dynamic property exists 
 * @param initParams: Pointer to NvDsInferContextInitParams structure
 * @param key: Property name to check
 * @return: 1 if property exists, 0 otherwise
 */
int NvDsInferContext_HasDynamicProperty(const NvDsInferContextInitParams *initParams, 
                                       const char *key);

/** 
 * Clear all dynamic properties 
 * @param initParams: Pointer to NvDsInferContextInitParams structure
 */
void NvDsInferContext_ClearDynamicProperties(NvDsInferContextInitParams *initParams);

/** 
 * Apply dynamic properties to TensorRT builder configuration
 * @param initParams: Pointer to NvDsInferContextInitParams structure
 * @param builderConfig: Pointer to TensorRT builder config (nvinfer1::IBuilderConfig*)
 */
void NvDsInferContext_ApplyDynamicPropertiesToBuilder(const NvDsInferContextInitParams *initParams,
                                                     void *builderConfig);

/**
 * Gets the string name of the status.
 *
 * @param[in] status    An inference status code.
 * @return  A pointer to a string containing the status's name, or NULL if
 *  the status is unrecognized. Memory is owned by the function; the caller
 *  may not free it.
 */
_DS_DEPRECATED_("NvDsInferContext_GetStatusName is deprecated. Use NvDsInferStatus2Str instead")
const char * NvDsInferContext_GetStatusName (NvDsInferStatus status);


#ifdef __cplusplus
}
#endif

/** @} */

/**
 * @name NvDsInferContext API C++-interface
 * This section describes the C++ interface for the NvDsInferContext class.
 * @{
 */
#ifdef __cplusplus

#include <string>
#include <vector>

/**
 * Holds the DeepStream inference interface class.
 */
struct INvDsInferContext
{
public:
    /**
     * Queues a batch of input frames for preprocessing and inferencing.
     * The input
     * frames must be in packed RGB/RGBA/GRAY UINT8 format with the same
     * resolution as the network input or preprocessed inputs so that it can be
     * fed directly to the inference engine. The frame memories should be
     * in CUDA device memory allocated on the same device that the
     * NvDsInferContext interface is configured with.
     *
     * The batch size must not exceed the maximum batch size requested during
     * initialization.
     *
     * @param[in] batchInput    Reference to a batch input structure.
     * @return  NVDSINFER_SUCCESS if preprocessing and queueing succeeded, or
     *  an error status otherwise.
     */
    virtual NvDsInferStatus queueInputBatch(NvDsInferContextBatchInput &batchInput) = 0;

    /**
     * Dequeues output for a batch of frames. The batch dequeuing order is same
     * as the input queuing order. The associated memory must be freed and
     * output buffers must be released back to the context using
     * releaseBatchOutput() so that the buffers can be reused.
     *
     * @param[out] batchOutput  Reference to the batch output structure
     *                          to which the output is to be appended.
     * @return  NVDSINFER_SUCCESS if dequeueing succeeded, or an error status
     *  otherwise.
     */
    virtual NvDsInferStatus dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput) = 0;

    /**
     * Frees the memory associated with the batch output and releases the set of
     * output buffers back to the context for reuse.
     *
     * @param[in] batchOutput   Reference to a batch output structure
     *                          which was filled by dequeueOutputBatch().
     */
    virtual void releaseBatchOutput(NvDsInferContextBatchOutput &batchOutput) = 0;

    /**
     * Fills the input vector with information on all bound layers of the
     * inference engine.
     *
     * @param[in,out] layersInfo    Reference to a vector of layer info
     *                              structures to be filled by the function.
     */
    virtual void fillLayersInfo(std::vector<NvDsInferLayerInfo> &layersInfo) = 0;

    /**
     * Gets network input information.
     *
     * @param[in,out] networkInfo   Reference to a network info structure.
     */
    virtual void getNetworkInfo(NvDsInferNetworkInfo &networkInfo) = 0;

    /**
     * \brief  Gets the label strings parsed from the labels file.
     *
     * See the DeepStream NvInfer documentation for the format of the
     * labels file for detectors and classifiers.
     *
     * @return  Reference to a vector of vector of string labels.
     */
    virtual const std::vector< std::vector<std::string> >& getLabels() = 0;

    /**
     * Deinitialize the inference engine and frees resources it used.
     */
    virtual void destroy() = 0;

    /** Destructor for a C++ object. */
    virtual ~INvDsInferContext() {}

    /**
     * Queues a batch of preprocessed input tensors for inferencing.
     *
     * The input tensor sizes must be compatible with network dimensions.
     *
     * @param[in] batchInput    Reference to a batch input structure.
     * @return  NVDSINFER_SUCCESS if preprocessing and queueing succeeded, or
     *  an error status otherwise.
     */
    virtual NvDsInferStatus queueInputBatchPreprocessed(NvDsInferContextBatchPreprocessedInput &batchInput) = 0;
};

/**
 * Creates a new instance of NvDsInferContext initialized using supplied
 * parameters.
 *
 * @param[out] handle       A pointer to a NvDsInferContext handle.
 * @param[in]  initParams   A reference to parameters to be used to initialize
 *                          the context.
 * @param[in]  userCtx      A pointer to an opaque user context with callbacks
 *                          generated by the NvDsInferContext instance.
 * @param[in] logFunc       A log callback function for the instance.
 * @return  NVDSINFER_SUCCESS if the instance was created successfully,
 *  or an error status otherwise.
 */
NvDsInferStatus createNvDsInferContext(NvDsInferContextHandle *handle,
        NvDsInferContextInitParams &initParams,
        void *userCtx = nullptr,
        NvDsInferContextLoggingFunc logFunc = nullptr);

#endif

/** @} */

/**
 * @name  NvDsInferContext API C-interface
 * This section describes the C interface for the NvDsInferContext class.
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Creates a new NvDsInferContext object with specified
 * initialization parameters.
 *
 * @param[out] handle       A pointer to an NvDsInferContext handle.
 * @param[in]  initParams   A pointer to a parameter structure to be used to
 *                          initialize the context.
 * @param[in]  userCtx      A pointer to an opaque user context, with callbacks,
 *                          generated by the NvDsInferContext instance.
 * @param[in] logFunc       A log callback for the instance.
 * @return  NVDSINFER_SUCCESS if creation was successful, or an error status
 *  otherwise.
 */
NvDsInferStatus NvDsInferContext_Create(NvDsInferContextHandle *handle,
        NvDsInferContextInitParams *initParams, void *userCtx,
        NvDsInferContextLoggingFunc logFunc);

/**
 * Destroys an NvDsInferContext instance and releases its resources.
 *
 * @param[in] handle    The handle to the NvDsInferContext instance to be
 *                      destroyed.
 */
void NvDsInferContext_Destroy (NvDsInferContextHandle handle);

/**
 * \brief  Queues a batch of input frames for preprocessing and inferencing.
 *
 * @see NvDsInferContext::queueInputBatch() for details.
 *
 * @param[in] handle        A handle to an NvDsInferContext instance.
 * @param[in] batchInput    A reference to a batch input structure.
 * @return  NVDSINFER_SUCCESS if preprocessing and queueing were successful, or
 *  an error status otherwise.
 */
NvDsInferStatus NvDsInferContext_QueueInputBatch(NvDsInferContextHandle handle,
        NvDsInferContextBatchInput *batchInput);

/**
 * Dequeues output for a batch of frames.
 *
 * @see NvDsInferContext::dequeueOutputBatch() for details.
 *
 * @param[in] handle            A handle to an NvDsInferContext instance.
 * @param[in,out] batchOutput   A reference to the batch output structure
 *                              to which output is to be appended.
 * @return  NVDSINFER_SUCCESS if dequeueing was successful, or an error status
 *  otherwise.
 */
NvDsInferStatus NvDsInferContext_DequeueOutputBatch(NvDsInferContextHandle handle,
        NvDsInferContextBatchOutput *batchOutput);

/**
 * Frees the memory associated with the batch output and releases the set of
 * host buffers back to the context for reuse.
 *
 * @param[in] handle        A handle to an NvDsInferContext instance.
 * @param[in] batchOutput   A pointer to an NvDsInferContextBatchOutput
 *                          structure filled by
 *                          NvDsInferContext_DequeueOutputBatch().
 */
void NvDsInferContext_ReleaseBatchOutput(NvDsInferContextHandle handle,
        NvDsInferContextBatchOutput *batchOutput);

/**
 * Gets network input information.
 *
 * @param[in]     handle        A handle to an NvDsInferContext instance.
 * @param[in,out] networkInfo   A pointer to an NvDsInferNetworkInfo structure.
 */
void NvDsInferContext_GetNetworkInfo(NvDsInferContextHandle handle,
        NvDsInferNetworkInfo *networkInfo);

/**
 * Gets the number of the bound layers of the inference engine in an
 * NvDsInferContext instance.
 *
 * @param[in] handle    A handle to an NvDsInferContext instance.
 * @return  The number of bound layers of the inference engine.
 */
unsigned int NvDsInferContext_GetNumLayersInfo(NvDsInferContextHandle handle);

/**
 * Fills an input vector with information about all of the bound layers of the
 * inference engine in an NvDsInferContext instance.
 * The size of the array must be at least the value returned by
 * NvDsInferContext_GetNumLayersInfo().
 *
 * @param[in]     handle        A handle to an NvDsInferContext instance.
 * @param[in,out] layersInfo    A pointer to an array of NvDsInferLayerInfo
                                structures to be filled by the function.
 */
void NvDsInferContext_FillLayersInfo(NvDsInferContextHandle handle,
        NvDsInferLayerInfo *layersInfo);

/**
 * Gets the string label associated with the class ID for detectors and the
 * attribute ID and attribute value for classifiers. The string is owned
 * by the context; the caller may not modify or free it.
 *
 * @param[in] handle    A handle to an NvDsInferContext instance.
 * @param[in] id        Class ID for detectors, or attribute ID for classifiers.
 * @param[in] value     Attribute value for classifiers; set to 0 for detectors.
 * @return  A pointer to a string label. The memory is owned by the context.
 */
const char* NvDsInferContext_GetLabel(NvDsInferContextHandle handle,
        unsigned int id, unsigned int value);

#ifdef __cplusplus
}
#endif

/** @} */

#endif

/** @} */
