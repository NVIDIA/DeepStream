/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
