/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA GStreamer DeepStream: Metadata Extension</b>
 *
 * @b Description: This file defines the Metadata structure used to
 * carry DeepStream metadata or any other metadata in GStreamer pipeline.
 */

/**
 * @defgroup gstreamer_metagroup_api DeepStream Metadata Extension
 *
 * Defines an API for managing GStreamer DeepStream metadata.
 * @ingroup custom_gstreamer
 * @{
 *
 * DeepStream Metadata is attached to a buffer with gst_buffer_add_nvds_meta().
 * Its metadata type is set to @ref NVDS_BATCH_GST_META.
 *
 * Multiple groups of metadata may be attached by different elements.
 * gst_buffer_get_nvds_meta() gets the last added @ref NvDsMeta.
 *
 * The NvDsMeta::meta_data structure member must be cast
 * to a meaningful structure pointer based on the @a meta_type.
 * For example, for @a meta_type = NVDS_BATCH_GST_META,
 * @a meta_data must be cast as `(NvDsBatchMeta *)`.
 */

#ifndef GST_NVDS_META_API_H
#define GST_NVDS_META_API_H

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>

#include "nvdsmeta.h"
#include "nvds_audio_meta.h"
#include "nvds_latency_meta.h"

#ifdef __cplusplus
extern "C"
{
#endif
GType nvds_meta_api_get_type (void);
#define NVDS_META_API_TYPE (nvds_meta_api_get_type())

const GstMetaInfo *nvds_meta_get_info (void);

#define NVDS_META_INFO (nvds_meta_get_info())

#define NVDS_META_STRING "nvdsmeta"

/**
 * Defines the type of metadata. NVIDIA-defined %GstNvDsMetaType values are
 * in the range from @ref NVDS_BATCH_GST_META to @ref NVDS_START_USER_META.
 */
typedef enum {
  NVDS_GST_INVALID_META=-1,
  /* Specifies information of a formed batch. */
  NVDS_BATCH_GST_META = NVDS_GST_CUSTOM_META + 1,
  NVDS_DECODER_GST_META,
  /* Specifies information of dewarped surfaces. */
  NVDS_DEWARPER_GST_META,
  /* @ref NvDsMetaType for @ref NvDsUserMeta in @NvDsFrameMeta
   * This metadata carries GstMeta objects of all input frames
   * batched by nvstreammux2
   * Note: Please refer to APIs @ref nvds_copy_gst_meta_to_frame_meta
   * and @ref nvds_copy_gst_meta_to_audio_frame_meta for more information.
   */
  NVDS_BUFFER_GST_AS_FRAME_USER_META,
  NVDS_RESERVED_GST_META = NVDS_GST_CUSTOM_META + 4096,
  /* Specifies the first value that may be assigned to a user-defined type. */
  NVDS_GST_META_FORCE32 = 0x7FFFFFFF
} GstNvDsMetaType;

/**
 * Holds DeepSteam metadata.
 * */
 typedef struct _NvDsMeta {
  GstMeta       meta;

  /** Holds a pointer to metadata. Must be cast to another structure based
   on @a meta_type. */
  gpointer meta_data;

 /** Holds a pointer to user-specific data . */
  gpointer user_data;

  /** Holds the type of metadata, one of values of enum @ref GstNvDsMetaType. */
  gint meta_type;

  /** A callback to be called when @a meta_data is to be copied or transformed
   from one buffer to other. @a meta_data and @a user_data are passed
   as arguments. */
  NvDsMetaCopyFunc copyfunc;

  /** A callback to be called when @a meta_data is to be destroyed.
   @a meta_data and @a user_data are passed as arguments. */
  NvDsMetaReleaseFunc freefunc;

  /**
   * A callback to be called when @a meta_data is transformed into
   * @a NvDsUserMeta.
   * This function must be provided by a GStreamer plugin that precedes
   * @a Gst-nvstreammux in the DeepStream pipeline.
   * Gst-nvstreammux copies @a meta_data to
   * user meta data at frame level inside @ref NvDsFrameMeta.
   * @a meta_data and @a user_data are passed as arguments.
   *
   * To retrive the content of @a meta_data, iterate over
   * @ref NvDsFrameMetaList. Then search for @a meta_type of @ref NvDsUserMeta
   * which the user has attached. (See deepstream-gst-metadata-test source
   * code for more details.)
   *
   * @a meta_data and @a user_data are passed as arguments.
   */
  NvDsMetaCopyFunc gst_to_nvds_meta_transform_func;

  /**
   * A callback to be called when @a meta_data transformed into
   * @a NvDsUserMeta is to be destroyed.
   * This function must be provided by a GStreamer plugin that precedes
   * @a Gst-nvstreammux in the DeepStream pipeline.
   */
  NvDsMetaReleaseFunc gst_to_nvds_meta_release_func;

} NvDsMeta;

/**
 * Adds %GstMeta of type @ref NvDsMeta to the GstBuffer and sets the @a meta_data
 * member of @ref NvDsMeta.
 *
 * @param[in] buffer    A pointer to a %GstBuffer to which the function adds
 *                      metadata.
 * @param[in] meta_data A pointer at which the function sets the @a meta_data
 *                      member of @ref NvDsMeta.
 * @param[in] user_data A pointer to the user-specific data.
 * @param[in] copy_func The callback to be called when
 *                      NvDsMeta is to be copied. The function is called with
 *                      @a meta_data and @a user_data as parameters.
 * @param[in] release_func
 *                      The callback to be called when
 *                      NvDsMeta is to be destroyed. The function is called with
 *                      @a meta_data and @a user_data as parameters.
 *
 * @return  A pointer to the attached NvDsMeta structure if successful,
 *  or NULL otherwise.
 */
NvDsMeta *gst_buffer_add_nvds_meta (GstBuffer *buffer, gpointer meta_data,
    gpointer user_data, NvDsMetaCopyFunc copy_func,
    NvDsMetaReleaseFunc release_func);

/**
 * Gets the @ref NvDsMeta last added to a GstBuffer.
 *
 * @param[in] buffer    A pointer to the GstBuffer.
 *
 * @return  A pointer to the last added NvDsMeta structure, or NULL if no
 *          %NvDsMeta was attached.
 */
NvDsMeta* gst_buffer_get_nvds_meta (GstBuffer *buffer);

/**
 * Gets the @ref NvDsBatchMeta added to a GstBuffer.
 *
 * @param[in] buffer    A pointer to the GstBuffer.
 *
 * @return  A pointer to the NvDsBatchMeta structure, or NULL if no
 *          NvDsMeta was attached.
 */
NvDsBatchMeta * gst_buffer_get_nvds_batch_meta (GstBuffer *buffer);

/**
 * Copies all GstMeta objects on src_gst_buffer to
 * the batched buffer's @ref NvDsBatchMeta
 * The GstMeta objects are copied into the user_meta_list
 * within @ref NvDsFrameMeta.
 *
 * Note: the list of plain GstMeta from @src_gst_buffer are copied into
 * @frame_meta as a single frame_meta->user_meta
 * with NvDsUserMeta->base_meta.meta_type == @ref NVDS_USER_FRAME_META_GST_META.
 * The list of N X NvDsMeta from @src_gst_buffer are copied into
 * @frame_meta as N X frame_meta->user_meta
 * with each NvDsUserMeta->base_meta.meta_type == NvDsMeta->meta_type.
 *
 * @param[in] src_gst_buffer    A pointer to the GstBuffer.
 *                              GstMeta objects in this GstBuffer
 *                              will be copied.
 * @param[in] batch_meta        A pointer to the @ref NvDsBatchMeta obtained
 *                              from nvstreammux plugin. (Note: Works only
 *                              for nvstreammux2 (Beta))
 * @param[in] frame_meta        A pointer to the @ref NvDsFrameMeta
 *                              @ref NvDsFrameMeta (metadata) for
 *                              the batched input frame to which GstMeta objects
 *                              will be copied as frame @ref NvDsUserMeta.
 *
 */
void nvds_copy_gst_meta_to_frame_meta(GstBuffer* src_gst_buffer, NvDsBatchMeta* batch_meta, NvDsFrameMeta* frame_meta);

/**
 * Copies all GstMeta objects on src_gst_buffer to
 * the batched buffer's @ref NvDsBatchMeta
 * The GstMeta objects are copied into the user_meta_list
 * within @ref NvDsAudioFrameMeta.
 *
 * Note: the list of plain GstMeta from @src_gst_buffer are copied into
 * @frame_meta as a single frame_meta->user_meta
 * with NvDsUserMeta->base_meta.meta_type == @ref NVDS_USER_FRAME_META_GST_META.
 * The list of N X NvDsMeta from @src_gst_buffer are copied into
 * @frame_meta as N X frame_meta->user_meta
 * with each NvDsUserMeta->base_meta.meta_type == NvDsMeta->meta_type.
 *
 * @param[in] src_gst_buffer    A pointer to the GstBuffer.
 *                              GstMeta objects in this GstBuffer
 *                              will be copied.
 * @param[in] batch_meta        A pointer to the @ref NvDsBatchMeta obtained
 *                              from nvstreammux plugin. (Note: Works only
 *                              for nvstreammux2 (Beta))
 * @param[in] frame_meta        A pointer to the @ref NvDsAudioFrameMeta
 *                              @ref NvDsAudioFrameMeta (metadata) for
 *                              the batched input frame to which GstMeta objects
 *                              will be copied as frame @ref NvDsUserMeta.
 *
 */
void nvds_copy_gst_meta_to_audio_frame_meta(GstBuffer* src_gst_buffer, NvDsBatchMeta* batch_meta, NvDsAudioFrameMeta* frame_meta);

/** @} */
#ifdef __cplusplus
}
#endif
#endif
