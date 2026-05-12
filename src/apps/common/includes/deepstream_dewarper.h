/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_DEWARPER_H__
#define __NVGSTDS_DEWARPER_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *src_queue;
  GstElement *conv_queue;
  GstElement *nvvidconv;
  GstElement *cap_filter;
  GstElement *dewarper_caps_filter;
  GstElement *nvdewarper;
} NvDsDewarperBin;

typedef struct
{
  gboolean enable;
  guint gpu_id;
  guint num_out_buffers;
  guint dewarper_dump_frames;
  gchar *config_file;
  guint nvbuf_memory_type;
  guint source_id;
  guint num_surfaces_per_frame;
  guint num_batch_buffers;
} NvDsDewarperConfig;

gboolean create_dewarper_bin (NvDsDewarperConfig * config, NvDsDewarperBin * bin);

#ifdef __cplusplus
}
#endif

#endif
