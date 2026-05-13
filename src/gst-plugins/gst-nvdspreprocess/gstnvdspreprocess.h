/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GST_NVDSPREPROCESS_H__
#define __GST_NVDSPREPROCESS_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"

#include "gstnvdspreprocess_allocator.h"
#include "nvdspreprocess_interface.h"
#include "nvdspreprocess_meta.h"

#include "nvtx3/nvToolsExt.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <functional>

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsvideotemplate"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA custom preprocessing plugin for integration with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "NVIDIA DeepStream Preprocessing using custom algorithms for different streams"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstNvDsPreProcess GstNvDsPreProcess;
typedef struct _GstNvDsPreProcessClass GstNvDsPreProcessClass;

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSPREPROCESS (gst_nvdspreprocess_get_type())
#define GST_NVDSPREPROCESS(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSPREPROCESS,GstNvDsPreProcess))
#define GST_NVDSPREPROCESS_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSPREPROCESS,GstNvDsPreProcessClass))
#define GST_NVDSPREPROCESS_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSPREPROCESS, GstNvDsPreProcessClass))
#define GST_IS_NVDSPREPROCESS(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSPREPROCESS))
#define GST_IS_NVDSPREPROCESS_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSPREPROCESS))
#define GST_NVDSPREPROCESS_CAST(obj)  ((GstNvDsPreProcess *)(obj))

/** per frame roi info */
typedef struct
{
  /** list of roi vectors per frame */
  std::vector<NvDsRoiMeta> roi_vector;
} GstNvDsPreProcessFrame;

typedef struct
{
  /** vector of src_ids */
  std::vector<gint> src_ids;

  /** vector of operate_on_class_ids for sgie*/
  std::vector<gint> operate_on_class_ids;

  /** total rois/full-frames in a group */
  guint num_units;

  /** custom transformation function name */
  std::string custom_transform_function_name;

  /** wrapper to custom transformation function */
  std::function<NvDsPreProcessStatus(NvBufSurface *, NvBufSurface *,
                                    CustomTransformParams &)> custom_transform;

  /** sync object for async transformation */
  NvBufSurfTransformSyncObj_t sync_obj = NULL;

  /** Map src_id - Preprocess Frame meta */
  std::unordered_map<gint, GstNvDsPreProcessFrame> framemeta_map;

  /** boolean indicating if processing on rois/full-frames inside the group */
  gboolean process_on_roi = 0;

  /** boolean indicating if secondary classification is done on all detections/roi inside the group */
  gboolean process_on_all_objects = 0;

  /** boolean indicating that the roi is to be drawn or not */
  gboolean draw_roi = 0;

  /** color of roi*/
  NvOSD_ColorParams roi_color;

  /** Input object size-based filtering parameters for object processing mode. */
  guint min_input_object_width;
  guint min_input_object_height;
  guint max_input_object_width;
  guint max_input_object_height;

  /** src-id whose rois is used by all the src within the preprocess-group (when src-ids[0]=-1)*/
  guint replicated_src_id;

  /** Group interval for frame/roi processing. */
  guint interval = 0;
  guint interval_counter = 0;
} GstNvDsPreProcessGroup;

/** Used by plugin to access GstBuffer and GstNvDsPreProcessMemory
 *  acquired by Custom Library */
struct NvDsPreProcessCustomBufImpl : public  NvDsPreProcessCustomBuf
{
  /** Gst Buffer acquired from gst allocator */
  GstBuffer *gstbuf;
  /** Memory corresponding to the gst buffer */
  GstNvDsPreProcessMemory *memory;
};

/**
 *  For Acquiring/releasing buffer from buffer pool
 */
class NvDsPreProcessAcquirerImpl : public NvDsPreProcessAcquirer
{
public:
  /** constructor */
  NvDsPreProcessAcquirerImpl(GstBufferPool *pool);
  /** override acquire method in plugin */
  NvDsPreProcessCustomBuf* acquire() override;
  /** override release method in plugin */
  gboolean release(NvDsPreProcessCustomBuf *) override;

private:
  GstBufferPool *m_gstpool = nullptr;
};

/**
 *  struct denoting properties set by config file
 */
typedef struct {
  /** for config param : processsing-width*/
  gboolean processing_width;
  /** for config param : processsing-height*/
  gboolean processing_height;
  /** for config param : network-input-order */
  gboolean network_input_order;
  /** for config param : network-input-shape */
  gboolean network_input_shape;
  /** for config param : network-color-format */
  gboolean network_color_format;
  /** for config param : tensor-data-type */
  gboolean tensor_data_type;
  /** for config param : tensor-name */
  gboolean tensor_name;
  /** for config param : custom-lib-path */
  gboolean custom_lib_path;
  /** for config param : custom-tensor-function-name */
  gboolean custom_tensor_function_name;
  /** for config param : src-ids */
  gboolean src_ids;
  /** for config param : operate-on-class-ids */
  gboolean operate_on_class_ids;
  /** for config param : process-on-rois */
  gboolean process_on_roi;
  /** for config param : process-on-all-objects */
  gboolean process_on_all_objects;
  /** for config param : roi-params-src */
  gboolean roi_params_src;
  /** for config param : scaling-pool-interpolation-filter */
  gboolean scaling_pool_interpolation_filter;
  /** for config param : scaling-pool-memory-type */
  gboolean scaling_pool_memory_type;
  /** for config param : draw_roi */
  gboolean draw_roi;
  /** for config param : roi_color */
  gboolean roi_color;
  /** for config param : input-object-min-width */
  gboolean min_input_object_width;
  /** for config param : input-object-min-height */
  gboolean min_input_object_height;
  /** for config param : input-object-max-width */
  gboolean max_input_object_width;
  /** for config param : input-object-max-height */
  gboolean max_input_object_height;
  /** for config param : interval */
  gboolean interval;
} NvDsPreProcessPropertySet;

/**
 * Strucuture containing Preprocess info
 */
struct _GstNvDsPreProcess
{
  /** Gst Base Transform */
  GstBaseTransform base_trans;

  /** Target unique ids */
  std::vector <guint64> target_unique_ids;

  /** Gie id to process */
  gint operate_on_gie_id;

  /** group information as specified in config file */
  std::vector<GstNvDsPreProcessGroup*> nvdspreprocess_groups;

  /** struct denoting properties set by config file */
  NvDsPreProcessPropertySet property_set;

  /** pointer to the custom lib ctx */
  CustomCtx* custom_lib_ctx;

  /** custom lib init params */
  CustomInitParams custom_initparams;

  /** custom lib handle */
  void* custom_lib_handle;

  /** Custom Library Name */
  gchar* custom_lib_path;

  /** custom tensor function name */
  std::string custom_tensor_function_name;

  /** wrapper to custom tensor function */
  std::function <NvDsPreProcessStatus(CustomCtx *, NvDsPreProcessBatch *, NvDsPreProcessCustomBuf *&,
                                      CustomTensorParams &, NvDsPreProcessAcquirer *)> custom_tensor_function;

   /** Internal buffer pool for memory required for scaling input frames and
    * cropping object. */
  GstBufferPool *scaling_pool;

  /** scaling pool color format */
  NvDsPreProcessFormat scaling_pool_format;

  /** scaling pool memory type*/
  NvBufSurfaceMemType scaling_pool_memory_type;

  /** compute hw for transformation */
  NvBufSurfTransform_Compute scaling_pool_compute_hw;

  /** interpolation filter for transformation */
  NvBufSurfTransform_Inter scaling_pool_interpolation_filter;;

  /** Scaling buffer pool size */
  guint scaling_buf_pool_size;

  /** meta id for differentiating between multiple tensor meta from same gst buffer */
  guint meta_id;

   /** Internal buffer pool for memory required for tensor preparation */
  GstBufferPool *tensor_pool;

  /** tensor buffer pool size */
  guint tensor_buf_pool_size;

  /** Class for acquiring/releasing buffer from tensor pool */
  std::unique_ptr <NvDsPreProcessAcquirerImpl> acquire_impl;

  /** pointer to buffer provided to custom library for tensor preparation */
  NvDsPreProcessCustomBuf *tensor_buf;

  /** Parameters for tensor preparation */
  NvDsPreProcessTensorParams tensor_params;

  /** Resolution width at which roi/full-frames should be processed */
  gint processing_width;

  /** Resolution height at which roi/full-frames should be processed */
  gint processing_height;

  /** Cuda Stream to ROI crop, scale and convert */
  cudaStream_t convert_stream;

  /** Boolean to indicate maintain aspect ratio */
  gboolean maintain_aspect_ratio;

  /** Boolean to indicate symmetric padding */
  gboolean symmetric_padding;

  /** Processing Queue and related synchronization structures. */
  /** Gmutex lock for against shared access in threads**/
  GMutex preprocess_lock;

  /** Queue to send data to output thread for processing**/
  GQueue *preprocess_queue;

  /** Gcondition for process queue**/
  GCond preprocess_cond;

  /** Output thread. */
  GThread *output_thread;

  /** Boolean to signal output thread to stop. */
  gboolean stop;

  /** Unique ID of the element. Used to identify metadata
   *  generated by this element. */
  guint unique_id;

  /** Frame number of the current input buffer */
  guint64 frame_num;

  /** Temporary NvBufSurface for input to batched transformations. */
  NvBufSurface batch_insurf;

  /** Temporary NvBufSurface for output from batched transformations. */
  NvBufSurface batch_outsurf;

  /** Maximum batch size. */
  guint max_batch_size;

  /** GPU ID on which we expect to execute the task */
  guint gpu_id;

  /** if disabled plugin will work in passthrough mode */
  gboolean enable;

  /** Config file path for nvdspreprocess **/
  gchar *config_file_path;

  /** Config file parsing status **/
  gboolean config_file_parse_successful;

  /** Boolean indicating if processing on frame or already cropped objects should be processed */
  gboolean process_on_frame;

  /** Current batch number of the input batch. */
  gulong current_batch_num;

  /** GstFlowReturn returned by the latest buffer pad push. */
  GstFlowReturn last_flow_ret;

  /** Config params required by NvBufSurfTransform API. */
  NvBufSurfTransformConfigParams transform_config_params;

  /** Parameters to use for transforming buffers. */
  NvBufSurfTransformParams transform_params;

  /** NVTX Domain. */
  nvtxDomainHandle_t nvtx_domain;

  /** Map src-id : preprocess-group-id */
  std::unordered_map<gint, gint> *src_to_group_map;

  /** Lock for framemeta_map */
  GMutex framemeta_map_lock;
};

/** Boiler plate stuff */
struct _GstNvDsPreProcessClass
{
  /** gst base transform class */
  GstBaseTransformClass parent_class;
};

GType gst_nvdspreprocess_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSPREPROCESS_H__ */
