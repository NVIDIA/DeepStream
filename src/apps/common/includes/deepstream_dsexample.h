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

#ifndef _NVGSTDS_DSEXAMPLE_H_
#define _NVGSTDS_DSEXAMPLE_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  // Create a bin for the element only if enabled
  gboolean enable;
  // Struct members to store config / properties for the element
  gboolean full_frame;
  gint processing_width;
  gint processing_height;
  gboolean blur_objects;
  guint unique_id;
  guint gpu_id;
  guint batch_size;
  // For nvvidconv
  guint nvbuf_memory_type;
} NvDsDsExampleConfig;

// Struct to store references to the bin and elements
typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *pre_conv;
  GstElement *cap_filter;
  GstElement *elem_dsexample;
} NvDsDsExampleBin;

// Function to create the bin and set properties
gboolean
create_dsexample_bin (NvDsDsExampleConfig *config, NvDsDsExampleBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSEXAMPLE_H_ */
