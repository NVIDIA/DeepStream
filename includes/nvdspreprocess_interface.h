/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file nvdspreprocess_interface.h
 * <b>NVIDIA DeepStream GStreamer NvDsPreProcess API Specification </b>
 *
 * @b Description: This file specifies the APIs and function definitions for
 * the DeepStream GStreamer NvDsPreProcess Plugin.
 */

/**
 * @defgroup gstreamer_nvdspreprocess_api NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess plugin.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_INTERFACE_H__
#define __NVDSPREPROCESS_INTERFACE_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsmeta.h"

#include "nvds_roi_meta.h"
#include "nvtx3/nvToolsExt.h"
#include <unordered_map>

/**
 * Context for custom library
 */
typedef struct CustomCtx CustomCtx;

/**
 * Enum for the status codes returned by NvDsPreProcessImpl.
 */
typedef enum
{
  /** NvDsPreprocess operation succeeded. */
  NVDSPREPROCESS_SUCCESS = 0,
  /** Failed to configure the tensor_impl instance possibly due to an
     *  erroneous initialization property. */
  NVDSPREPROCESS_CONFIG_FAILED,
  /** Custom Library interface implementation failed. */
  NVDSPREPROCESS_CUSTOM_LIB_FAILED,
  /** Custom Group Transformation failed. */
  NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED,
  /** Custom Tensor Preparation failed. */
  NVDSPREPROCESS_CUSTOM_TENSOR_FAILED,
  /** Invalid parameters were supplied. */
  NVDSPREPROCESS_INVALID_PARAMS,
  /** Output parsing failed. */
  NVDSPREPROCESS_OUTPUT_PARSING_FAILED,
  /** CUDA error was encountered. */
  NVDSPREPROCESS_CUDA_ERROR,
  /** TensorRT interface failed. */
  NVDSPREPROCESS_TENSORRT_ERROR,
  /** Resource error was encountered. */
  NVDSPREPROCESS_RESOURCE_ERROR,
  /** Tensor Yet not ready to be attached as meta */
  NVDSPREPROCESS_TENSOR_NOT_READY,
} NvDsPreProcessStatus;

/**
 * Enum for the network input order according to which
 * network shape will be provided to prepare raw tensor
 * for inferencing.
 */
typedef enum
{
  /** Specifies NCHW network input order */
  NvDsPreProcessNetworkInputOrder_kNCHW = 0,
  /** Specifies NHWC network input order */
  NvDsPreProcessNetworkInputOrder_kNHWC,
  /** Specifies any other custom input order handled by custom lib*/
  NvDsPreProcessNetworkInputOrder_CUSTOM,
} NvDsPreProcessNetworkInputOrder;

/**
 * Defines model color formats
 */
typedef enum
{
  /** Specifies 24-bit interleaved R-G-B format. */
  NvDsPreProcessFormat_RGB,
  /** Specifies 24-bit interleaved B-G-R format. */
  NvDsPreProcessFormat_BGR,
  /** Specifies 8-bit Luma format. */
  NvDsPreProcessFormat_GRAY,
  /** Specifies 32-bit interleaved R-G-B-A format. */
  NvDsPreProcessFormat_RGBA,
  /** Specifies 32-bit interleaved B-G-R-x format. */
  NvDsPreProcessFormat_BGRx,
  /** NCHW planar */
  NvDsPreProcessFormat_Tensor,
  NvDsPreProcessFormat_Unknown = 0xFFFFFFFF,
} NvDsPreProcessFormat;

/**
 * custom transformation parameter for calling nvbufsurftransform
 * api for scaling and converting the ROIs to the network resolutions
 * to be used by custom lib.
 */
typedef struct
{
  /** transform config params for nvbufsurftransform api*/
  NvBufSurfTransformConfigParams transform_config_params;
  /** transform params for nvbufsurftransform api*/
  NvBufSurfTransformParams transform_params;
  /** sync objects for async transform */
  NvBufSurfTransformSyncObj_t sync_obj = NULL;
} CustomTransformParams;

/**
 * Holds model parameters for tensor preparation
 */
typedef struct
{
  /** network order at which model will work */
  NvDsPreProcessNetworkInputOrder network_input_order;
  /** Hold the network shape - interpreted based on network input order
   *  For resnet10 : NCHW = infer-batch-size;height;width;num-channels */
  std::vector<int> network_input_shape;
  /** Holds the network input format. */
  NvDsPreProcessFormat network_color_format;
  /** size of tensor buffer */
  guint64 buffer_size = 1;
  /** DataType for tensor formation */
  NvDsDataType data_type;
  /** Memory Type for tensor formation */
  NvBufSurfaceMemType memory_type;
  /** Name of the tensor same as input layer name of model */
  std::string tensor_name;
} NvDsPreProcessTensorParams;

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
} NvDsPreProcessNetworkSize;

/**
 * Tensor params for Custom sequence processing for 3d conv network
 */
typedef struct
{
  /** vector of rois which can be modified by custom lib */
  std::vector<NvDsRoiMeta> roi_vector;
} CustomSeqProcTensorParams;

/**
 * Tensor params passed to custom library for tensor preparation
 */
typedef struct
{
  /** tensor params from plugin */
  NvDsPreProcessTensorParams params;
  /** Additional Custom Parameters */
  CustomSeqProcTensorParams seq_params;
} CustomTensorParams;

/**
 * Custom Initialization parameter for custom library
 */
typedef struct
{
  /** unique id of the preprocess plugin */
  guint unique_id;
  /** tensor params from read from config file */
  NvDsPreProcessTensorParams tensor_params;
  /** User config map key-value pair */
  std::unordered_map <std::string, std::string> user_configs;
  /** nvdspreprocess config file path */
  gchar *config_file_path;
} CustomInitParams;

/**
 * Custom Buffer passed to the custom lib for preparing tensor.
 */
struct NvDsPreProcessCustomBuf
{
  /** memory ptr where to store prepared tensor */
  void *memory_ptr;
};

/**
 * class for acquiring and releasing a buffer from tensor pool
 * by custom lib.
 */
class NvDsPreProcessAcquirer
{
public:
  /** method to acquire a buffer from buffer pool */
  virtual NvDsPreProcessCustomBuf *acquire() = 0;
  /** method to release buffer from buffer pool */
  virtual gboolean release(NvDsPreProcessCustomBuf *) = 0;
};

/**
 * A preprocess unit for processing which can be Frame/ROI.
 */
typedef struct
{
  /** NvDsObjectParams belonging to the object to be classified. */
  NvDsObjectMeta *obj_meta = nullptr;
  /** NvDsFrameMeta of the frame being preprocessed */
  NvDsFrameMeta *frame_meta = nullptr;
  /** Index of the frame in the batched input GstBuffer. Not required for
   * classifiers. */
  guint batch_index = 0;
  /** Frame number of the frame from the source. */
  gulong frame_num = 0;
  /** The buffer structure the object / frame was converted from. */
  NvBufSurfaceParams *input_surf_params = nullptr;
  /** Pointer to the converted frame memory. This memory contains the frame
   * converted to RGB/RGBA and scaled to network resolution. This memory is
   * given to Output loop as input for mean subtraction and normalization and
   * Tensor Buffer formation for inferencing. */
  gpointer converted_frame_ptr = nullptr;
  /** New meta for rois provided */
  NvDsRoiMeta roi_meta;

} NvDsPreProcessUnit;

/**
 * Holds information about the batch of frames to be inferred.
 */
typedef struct
{
  /** Vector of units in the batch. Can be for Frame/ROI/Crop */
  std::vector<NvDsPreProcessUnit> units;
  /** Vector of sync objects for async transformation of the batch */
  std::vector<NvBufSurfTransformSyncObj_t> sync_objects;
  /** Pointer to the input GstBuffer. */
  GstBuffer *inbuf = nullptr;
  /** Batch number of the input batch. */
  gulong inbuf_batch_num = 0;
  /** Boolean indicating that the output thread should only push the buffer to
   * downstream element. If set to true, a corresponding batch has not been
   * queued at the input of NvDsPreProcessContext and hence dequeuing of output is
   * not required. */
  gboolean push_buffer = FALSE;
  /** Boolean marking this batch as an event marker. This is only used for
   * synchronization. The output loop does not process on the batch.
   */
  gboolean event_marker = FALSE;
  /** Buffer containing the intermediate conversion output for the batch. */
  GstBuffer *converted_buf = nullptr;
  /** scaling pool color format */
  NvDsPreProcessFormat scaling_pool_format;
  /** Deepstream batch meta */
  NvDsBatchMeta *batch_meta;
  /** Holds the pitch of the buffer */
  uint32_t pitch;
  /** nvtx buf range */
  nvtxRangeId_t nvtx_complete_buf_range = 0;
} NvDsPreProcessBatch;

#endif //__NVDSPREPROCESS_INTERFACE_H__

/** @} */
