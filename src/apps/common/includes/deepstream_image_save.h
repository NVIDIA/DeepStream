/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_IMAGE_SAVE_H__
#define __NVGSTDS_IMAGE_SAVE_H__

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
    gboolean enable;
    guint gpu_id;
    gchar *output_folder_path;
    gboolean save_image_full_frame;
    gboolean save_image_cropped_object;
    gchar *frame_to_skip_rules_path;
    guint second_to_skip_interval;
    gdouble min_confidence;
    gdouble max_confidence;
    guint min_box_width;
    guint min_box_height;
} NvDsImageSave;


#ifdef __cplusplus
}
#endif

#endif
