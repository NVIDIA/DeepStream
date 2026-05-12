/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file gstnvdsinfer.h
 * <b>NVIDIA DeepStream GStreamer NvInfer API Specification </b>
 *
 * @b Description: This file specifies the APIs and function definitions for
 * the DeepStream GStreamer NvInfer Plugin.
 */

/**
 * @defgroup   gstreamer_nvinfer_api  NvInfer Plugin
 * Defines an API for the GStreamer NvInfer plugin.
 * @ingroup custom_gstreamer
 * @{
 */

G_BEGIN_DECLS

#include "nvdsinfer.h"

/**
 * Function definition for the inference raw output generated callback of
 * Gst-NvInfer plugin.
 *
 * The callback function can be registered by setting "raw-output-generated-callback"
 * property on an "nvinfer" element instance. Additionally, a pointer to
 * user data can be set through the "raw-output-generated-userdata" property.
 * This pointer will be passed to the raw output generated callback function
 * through the userdata parameter.
 *
 * Refer to the reference deepstream-app sources for a sample implementation
 * of the callback.
 *
 * @param[in]  buf Pointer to the GstBuffer on whose contents inference has been
 *             executed. The implementation should assume the buffer to be
 *             read-only and should not modify the buffer in any way.
 * @param[in]  network_info Network information for the model specified for the
 *             nvinfer element instance.
 * @param[in]  layers_info Pointer to the array containing information for all
 *             bound layers for the inference engine.
 * @param[in]  num_layers Number of layers bound for the inference engine i.e.
 *             number of elements in the layers_info array.
 * @param[in]  batch_size Number of valid input frames in the batch.
 * @param[in]  user_data Pointer to the user data set through the
 *             "raw-output-generated-userdata" property.
 */
typedef void (* gst_nvinfer_raw_output_generated_callback) (GstBuffer *buf,
    NvDsInferNetworkInfo *network_info,  NvDsInferLayerInfo *layers_info,
    guint num_layers, guint batch_size, gpointer user_data);

/**
 * Holds the raw tensor output information for one frame / one object.
 *
 * The "nvinfer" plugins adds this meta when the "output-tensor-meta" property
 * of the element instance is set to TRUE.
 *
 * This meta data is added as NvDsUserMeta to the frame_user_meta_list of the
 * corresponding frame_meta or object_user_meta_list of the corresponding object
 * with the meta_type set to NVDSINFER_TENSOR_OUTPUT_META.
 */
typedef struct
{
  /** Unique ID of the gst-nvinfer instance which attached this meta. */
  guint unique_id;
  /** Number of bound output layers. */
  guint num_output_layers;
  /** Pointer to the array containing information for the bound output layers.
   * Size of the array will be equal to num_output_layers. Pointers inside
   * the NvDsInferLayerInfo structure are not valid for this array. */
  NvDsInferLayerInfo *output_layers_info;
  /** Array of pointers to the output host buffers for the batch / frame / object. */
  void **out_buf_ptrs_host;
  /** Array of pointers to the output device buffers for the batch / frame / object. */
  void **out_buf_ptrs_dev;
  /** GPU device ID on which the device buffers have been allocated. */
  gint gpu_id;
  /** Private data used for the meta producer's internal memory management. */
  void *priv_data;
  /** Network information for the model specified for the nvinfer element instance. */
  NvDsInferNetworkInfo network_info;
  /** Whether aspect ratio was maintained while scaling to network resolution */
  gboolean maintain_aspect_ratio;
  /** Whether symmetric padding was maintained while scaling to network resolution
   * valid only when maintain_aspect_ratio is true
   */
  gboolean symmetric_padding;
} NvDsInferTensorMeta;


/**
 * Holds the segmentation model output information for one frame / one object.
 *
 * The "nvinfer" plugins adds this meta for segmentation models.
 *
 * This meta data is added as NvDsUserMeta to the frame_user_meta_list of the
 * corresponding frame_meta or object_user_meta_list of the corresponding object
 * with the meta_type set to NVDSINFER_SEGMENTATION_META.
 */
typedef struct
{
  /** Number of classes in the segmentation output. */
  guint classes;
  /** Width of the segmentation output class map. */
  guint width;
  /** Height of the segmentation output class map. */
  guint height;
  /** Pointer to the array for 2D pixel class map. The output for pixel (x,y)
   * will be at index (y * width + x). */
  gint* class_map;
  /** Pointer to the raw array containing the probabilities. The probability for
   * class c and pixel (x,y) will be at index (c * width *height + y * width + x). */
  gfloat *class_probabilities_map;
  /** Private data used for the meta producer's internal memory management. */
  void *priv_data;
  /** Unique ID of the gst-nvinfer instance which attached this meta. */
  gint unique_id;
} NvDsInferSegmentationMeta;

G_END_DECLS

/** @} */
