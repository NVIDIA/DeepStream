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

#ifndef __NVGSTDS_VISIONENCODER_H__
#define __NVGSTDS_VISIONENCODER_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  gboolean enable;
  gchar *model_variant;
  guint batch_size;
  gchar *device;
  guint min_crop_size;
  gboolean verbose;
  guint gpu_id;
  gchar *backend;          // "transformers", "triton", or "tensorrt"
  gchar *triton_url;       // Triton gRPC endpoint (host:port)
  gchar *triton_model;     // Model name in Triton repository
  guint skip_interval;     // Frame skip interval (0=none, 1=every other, 2=every 3rd, etc.)
  gchar *embedding_classes; // Semicolon-separated list of classes to generate embeddings for
  gchar *onnx_model;       // Path to ONNX model for TensorRT engine building
  gchar *tensorrt_engine;  // Path to TensorRT engine file (for direct TensorRT backend)
  gboolean query_only;     // Init model but skip per-frame inference; REST queries still work
  gboolean smart_infer;    // Enable smart inference (cache embeddings for tracked objects)
  guint cache_refresh_interval; // Re-infer cached objects every N frames (0 = never)
  guint cache_max_size;    // Maximum cached embeddings
  gboolean ofa_predict;    // Enable OFA-based temporal embedding prediction
  gfloat ofa_motion_threshold;      // Below this: trust cache; above: predict
  gfloat ofa_high_motion_threshold; // Above this: force full re-inference
} NvDsVisionEncoderConfig;

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *visionencoder;
} NvDsVisionEncoderBin;

gboolean create_visionencoder_bin (NvDsVisionEncoderConfig *config,
    NvDsVisionEncoderBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* __NVGSTDS_VISIONENCODER_H__ */
