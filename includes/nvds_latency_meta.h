/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>Defines Latency API</b>
 *
 * @b Description: This file defines an API for measuring module latency.
 */

/**
 * @defgroup ee_nvlatency_group Latency Measurement API
 * Defines an API for measuring latency in modules.
 * @ingroup NvDsMetaApi
 * @{
 */

#ifndef _NVDSMETA_LATENCY_H_
#define _NVDSMETA_LATENCY_H_

#include "glib.h"
#include "gmodule.h"
#include "nvdsmeta.h"

#define MAX_COMPONENT_LEN 64

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Holds information about latency of the given subcomponent
 */
typedef struct _NvDsMetaSubCompLatency {
  /** Holds the subcomponent names. */
  gchar sub_comp_name[MAX_COMPONENT_LEN];
  /** Holds the system timestamp of the buffer when it arrives
   at the input of the component. */
  gdouble in_system_timestamp;
  /** Holds the system timestamp of the buffer when it leaves
   at the output of the component. */
  gdouble out_system_timestamp;
} NvDsMetaSubCompLatency;

/**
 * Holds information about latency of the given component
 */
typedef struct _NvDsMetaCompLatency {
  /** Holds the name of the component for which latency is measured. */
  gchar component_name[MAX_COMPONENT_LEN];
  /** Holds the system timestamp of the buffer when it arrives
   at the input of the component. */
  gdouble in_system_timestamp;
  /** Holds the system timestamp of buffer when it
   is sent to the downstream component. */
  gdouble out_system_timestamp;
  /** Holds the source ID of the component, e.g. the camera ID. */
  guint source_id;
  /** Holds the current frame number for which latency is measured. */
  guint frame_num;
  /** Holds the pad or port index of the stream muxer for the frame
   in the batch. */
  guint pad_index;
  /** Holds latency information of subcomponent in an array */
  NvDsMetaSubCompLatency sub_comp_latencies[16];
  /** Holds the number of subcomponents for the given component */
  guint num_sub_comps;
}NvDsMetaCompLatency;

/**
 * Holds information about the latency of a given frame.
 */
typedef struct
{
  /** Holds the source ID of the component, e.g. the camera ID. */
  guint source_id;
  /** Holds the current frame number for which latency is measured. */
  guint frame_num;
  /** Holds the system timestamp of the buffer when it arrives
   at the input of the first component in the pipeline. By default,
   the decoder is considered to be the first component in the pipeline. */
  gdouble comp_in_timestamp;
  /** Holds the latency of the frame in milliseconds. */
  gdouble latency;
} NvDsFrameLatencyInfo;

/**
 * Sets the system timestamp when the Gst Buffer arrives as input at the
 * component.
 *
 * @param[in] buffer        A pointer to the arriving Gst Buffer.
 * @param[in] element_name  A pointer to the name of the component for which
 *                          latency is to be measured.
 *
 * @returns  A pointer to a @ref NvDsUserMeta structure which holds an
 *          @ref NvDsMetaCompLatency as @a NvDsUserMeta.user_meta_data.
 */
NvDsUserMeta *nvds_set_input_system_timestamp(GstBuffer * buffer,
    gchar *element_name);

/**
 * \brief  Sets the system timestamp when a Gst Buffer that is pushed
 *  to the downstream component.
 *
 * This is a corresponding function to nvds_set_input_system_timestamp().
 *
 * @param[in] buffer        A pointer to a Gst Buffer to be pushed
 *                          to the downstream component.
 * @param[in] element_name  A pointer to the name of the component for which
 *                          latency is to be measured.
 *
 * @returns  TRUE if the timestamp is attached successfully, or FALSE otherwise.
 */
gboolean nvds_set_output_system_timestamp(GstBuffer * buffer, gchar *element_name);

/**
 * \brief  Measures the latency of all frames present in the current batch.
 *
 * The latency is computed from decoder input up to the point this API is called.
 * You can install the probe on either pad of the component and call
 * this function to measure the latency.
 *
 * @param[in]  buf          A pointer to a Gst Buffer to which
 *                          @ref NvDsBatchMeta is attached as metadata.
 * @param[out] latency_info A pointer to an NvDsFrameLatencyInfo structure
 *                          allocated for a batch of this size. The function
 *                          fills it with information about all of the sources.
 */
guint nvds_measure_buffer_latency(GstBuffer *buf,
    NvDsFrameLatencyInfo *latency_info);

/**
 * Indicates whether the environment variable
 * @c NVDS_ENABLE_LATENCY_MEASUREMENT is exported.
 *
 * @returns  True if the environment variable is exported, or false otherwise.
 */
gboolean nvds_get_enable_latency_measurement(void);

/**
 * Adds the reference timestamp metadata for this buffer
 * Note: element_name == "audiodecoder" and "nvv4l2decoder" will be used for
 * latency measurement calculations where the timestamp will be used
 * by API: nvds_measure_buffer_latency()
 * to calculate the latency of buffers downstream relative to the decoder.
 *
 * @param[in] buffer        A pointer to the arriving Gst Buffer.
 * @param[in] element_name  A pointer to the name of the component for which
 *                          latency is to be measured.
 * @param[in] frame_id      The id/number of this frame/buffer produced by
 *                          the component that produces this buffer.
 *
 */
void nvds_add_reference_timestamp_meta(GstBuffer * buffer,
    gchar *element_name, guint frame_id);

/**
 * Indicates whether silent mode for latency measurement is enabled.
 * When enabled, latency measurements continue but console prints are suppressed.
 *
 * @returns  True if NVDS_LATENCY_MEASUREMENT_SILENT is exported, or false otherwise.
 */
gboolean nvds_get_latency_measurement_silent(void);

/**
 * Defines a pseudo-variable whose value is the return value of
 * @ref nvds_get_enable_latency_measurement(). It indicates whether latency
 * measurement is enabled.
 */
#define nvds_enable_latency_measurement (nvds_get_enable_latency_measurement())

/**
 * Defines a pseudo-variable whose value is the return value of
 * @ref nvds_get_latency_measurement_silent(). It indicates whether latency
 * measurement is in silent mode (no console prints).
 */
#define nvds_latency_measurement_silent (nvds_get_latency_measurement_silent())

/** @} */

#ifdef __cplusplus
}
#endif
#endif
