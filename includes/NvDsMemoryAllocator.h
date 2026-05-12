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

#ifndef _NVDS_MEMORY_ALLOCATOR_H_
#define _NVDS_MEMORY_ALLOCATOR_H_

#include "INvDsAllocator.h"

/**
 * Specifies memory types for \ref NvDsMemory.
 */
typedef enum
{
  NVDS_MEM_DEFAULT,
  /** Specifies CUDA Host memory type. */
  NVDS_MEM_CUDA_PINNED,
  /** Specifies CUDA Device memory type. */
  NVDS_MEM_CUDA_DEVICE,
  /** Specifies CUDA Unified memory type. */
  NVDS_MEM_CUDA_UNIFIED,
  /** Specifies memory allocated by malloc(). */
  NVDS_MEM_SYSTEM,
} NvDsMemType;


class NvDsMemoryAllocator : INvDsAllocator
{
    public:

    NvDsMemoryAllocator(uint32_t gpuId, NvDsMemType memType);
    void* Allocate (uint32_t size);
    void Deallocate (void* data);

    private:
    uint32_t gpuId;
    NvDsMemType memType;
};

#endif
