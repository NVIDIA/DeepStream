/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_TEXT_EMBEDDER_H__
#define __NVGSTDS_TEXT_EMBEDDER_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  gboolean enable;
  gchar *model_name;      /* "siglip2-giant", "dfn5b-clip", or "dfnclip" */
  gchar *onnx_model_path; /* ONNX model file path (used for dfnclip) */
  gchar *tokenizer_dir;   /* tokenizer directory path (used for dfnclip) */
} NvDsTextEmbedderConfig;

#ifdef __cplusplus
}
#endif

#endif
