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
