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
