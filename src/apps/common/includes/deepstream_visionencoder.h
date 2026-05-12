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
