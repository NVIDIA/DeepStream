/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GSTNVDSPREPROCESSALLOCATOR_H__
#define __GSTNVDSPREPROCESSALLOCATOR_H__

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
  /** surface corresponding to memory allocated */
  NvBufSurface *surf;
  /** Vector of cuda resources created by registering the above egl images in CUDA. */
  std::vector<CUgraphicsResource> cuda_resources;
  /** Vector of CUDA eglFrames created by mapping the above cuda resources. */
  std::vector<CUeglFrame> egl_frames;
  /** Pointer to the memory allocated for the batch of frames (DGPU). */
  void *dev_memory_ptr;
  /** Vector of pointer to individual frame memories in the batch memory */
  std::vector<void *> frame_memory_ptrs;
} GstNvDsPreProcessMemory;

/**
 * Get GstNvDsPreProcessMemory structure associated with buffer allocated using
 * GstNvDsPreProcessAllocator.
 *
 * @param buffer GstBuffer allocated by this allocator.
 *
 * @return Pointer to the associated GstNvDsPreProcessMemory structure
 */
GstNvDsPreProcessMemory *gst_nvdspreprocess_buffer_get_memory (GstBuffer * buffer);

/**
 * structure containing video buffer allocator info
 */
typedef struct {
    /** video buffer width */
    guint width;
    /** video buffer height */
    guint height;
    /** color format */
    NvBufSurfaceColorFormat color_format;
    /** batch size */
    guint batch_size;
    /** memory type of buffer */
    NvBufSurfaceMemType memory_type;
} GstNvDsPreProcessVideoBufferAllocatorInfo;

/**
 * Create a new GstNvDsPreProcessAllocator with the given parameters.
 *
 * @param info video buffer allocator info.
 * @param raw_buf_size size of raw buffer to allocate.
 * @param gpu_id ID of the gpu where the batch memory will be allocated.
 * @param debug_tensor boolean to denote if DEBUG_TENSOR flag is enabled.
 *
 * @return Pointer to the GstNvDsPreProcessAllocator structure cast as GstAllocator
 */
GstAllocator *gst_nvdspreprocess_allocator_new (GstNvDsPreProcessVideoBufferAllocatorInfo *info, size_t raw_buf_size,
    guint gpu_id, gboolean debug_tensor);

#endif
