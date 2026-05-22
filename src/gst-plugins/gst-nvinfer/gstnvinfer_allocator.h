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

#ifndef __GSTNVINFERALLOCATOR_H__
#define __GSTNVINFERALLOCATOR_H__

#include <cuda_runtime_api.h>
#include <gst/gst.h>
#include <vector>
#include "cudaEGL.h"
#include "nvbufsurface.h"

/**
 * This file describes the custom memory allocator for the Gstreamer TensorRT
 * plugin. The allocator allocates memory for a specified batch_size of frames
 * of resolution equal to the network input resolution and RGBA color format.
 * The frames are allocated on device memory.
 */

/**
 * Holds the pointer for the allocated memory.
 */
typedef struct
{
  NvBufSurface *surf;
  /** Vector of cuda resources created by registering the above egl images in CUDA. */
  std::vector<CUgraphicsResource> cuda_resources;
  /** Vector of CUDA eglFrames created by mapping the above cuda resources. */
  std::vector<CUeglFrame> egl_frames;
  /** Pointer to the memory allocated for the batch of frames (DGPU). */
  void *dev_memory_ptr;
  /** Vector of pointer to individual frame memories in the batch memory */
  std::vector<void *> frame_memory_ptrs;
} GstNvInferMemory;

/**
 * Get GstNvInferMemory structure associated with buffer allocated using
 * GstNvInferAllocator.
 *
 * @param buffer GstBuffer allocated by this allocator.
 *
 * @return Pointer to the associated GstNvInferMemory structure
 */
GstNvInferMemory *gst_nvinfer_buffer_get_memory (GstBuffer * buffer);

/**
 * Create a new GstNvInferAllocator with the given parameters.
 *
 * @param width Width of the network input, in pixels.
 * @param height Height of the network input, in pixels.
 * @param color_format Color format of the buffers in the pool.
 * @param batch_size Max size of batch that will be inferred.
 * @param gpu_id ID of the gpu where the batch memory will be allocated.
 *
 * @return Pointer to the GstNvInferAllocator structure cast as GstAllocator
 */
GstAllocator *gst_nvinfer_allocator_new (guint width, guint height,
    NvBufSurfaceColorFormat color_format, guint batch_size, guint gpu_id);

#endif
