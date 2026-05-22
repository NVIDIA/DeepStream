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

#ifndef __GSTNVDSAUDIO_H__
#define __GSTNVDSAUDIO_H__

#include <gst/gst.h>
#include <vector>
#include "nvbufaudio.h"
#include "NvDsMemoryAllocator.h"

/**
 * This file describes the custom memory allocator for any Gstreamer
 * plugins wishing to create a pool of NvBufAudio batch buffers.
 * The allocator allocates memory for a specified batch_size of frames
 * of resolution equal to the network input resolution.
 * The frames are allocated on device memory.
 */

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Holds the pointer for the allocated memory.
 */
typedef struct
{
  /** The audio batch buffer */
  NvBufAudio* batch;
} GstNvDsAudioMemory;

typedef struct _GstNvDsAudioAllocatorParams
{
  /** Max size of audio batch */
  uint32_t batchSize;
  /** If the memory within a batch is contiguos or not */
  bool isContiguous;

  NvBufAudioLayout layout;
  NvBufAudioFormat format;
  uint32_t         bpf;      /**< Bytes per frame; the size of a frame;
                                * size of one sample * @channels */
  uint32_t         channels; /**< Number of audio channels */
  uint32_t         rate;     /**< audio sample rate in samples per second */
  /** The number of audio samples in each buffer of the batch */
  uint32_t         bufferLength;
  /** @param gpuId ID of the gpu where the batch memory will be allocated. */
  guint gpuId;
  NvDsMemType memType;
} GstNvDsAudioAllocatorParams;

/**
 * Get GstNvDsAudioMemory structure associated with buffer allocated using
 * GstNvDsAudioAllocator.
 *
 * @param buffer GstBuffer allocated by this allocator.
 *
 * @return Pointer to the associated GstNvDsAudioMemory structure
 */
GstNvDsAudioMemory *gst_nvdsaudio_buffer_get_memory (GstBuffer * buffer);

/**
 * Create a new GstNvDsAudioAllocator with the given parameters.
 *
 * @param params Audio allocator params.
 *
 * @return Pointer to the GstNvDsAudioAllocator structure cast as GstAllocator
 */
GstAllocator *gst_nvdsaudio_allocator_new (GstNvDsAudioAllocatorParams *params);

#if defined(__cplusplus)
}
#endif

#endif
