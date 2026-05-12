/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_SEGVISUAL_H__
#define __NVGSTDS_SEGVISUAL_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *nvvidconv;
  GstElement *conv_queue;
  GstElement *cap_filter;
  GstElement *nvsegvisual;
} NvDsSegVisualBin;

typedef struct
{
  gboolean enable;
  guint gpu_id;
  guint max_batch_size;
  guint width;
  guint height;
  guint nvbuf_memory_type; /* For nvvidconv */
} NvDsSegVisualConfig;

/**
 * Initialize @ref NvDsSegVisualBin. It creates and adds SegVisual and other elements
 * needed for processing to the bin. It also sets properties mentioned
 * in the configuration file under group @ref CONFIG_GROUP_SegVisual
 *
 * @param[in] config pointer to SegVisual @ref NvDsSegVisualConfig parsed from config file.
 * @param[in] bin pointer to @ref NvDsSegVisualBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_segvisual_bin (NvDsSegVisualConfig *config, NvDsSegVisualBin *bin);

#ifdef __cplusplus
}
#endif

#endif
