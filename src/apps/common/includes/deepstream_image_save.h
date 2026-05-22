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
