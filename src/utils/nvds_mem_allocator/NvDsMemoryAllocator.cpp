/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvDsMemoryAllocator.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        printf ("[%s:%d] Cuda failure: status=%d\n",    \
                         __func__, __LINE__, status);   \
        break;                                          \
    }                                                   \
}

NvDsMemoryAllocator::NvDsMemoryAllocator(uint32_t aGpuId,
        NvDsMemType aMemType) :
        gpuId(aGpuId),
        memType(aMemType)
{
}

void* NvDsMemoryAllocator::Allocate (uint32_t size)
{
    void* data = nullptr;
    switch (memType) {
      case NVDS_MEM_CUDA_PINNED:
        CHECK (cudaMallocHost (&data, size));
        CHECK (cudaMemset(data, 0x00, size));
        break;
      case NVDS_MEM_DEFAULT:
      case NVDS_MEM_CUDA_DEVICE:
        CHECK (cudaMalloc (&data, size));
        CHECK (cudaMemset(data, 0x00, size));
        break;
      case NVDS_MEM_CUDA_UNIFIED:
        CHECK (cudaMallocManaged (&data, size, 1));
        CHECK (cudaMemset(data, 0x00, size));
        break;
      case NVDS_MEM_SYSTEM:
        data = calloc(1, size);
        break;
      default:
          printf ("[%s:%d]nvdstensor: invalid memory type (%d)\n",
                __func__, __LINE__, memType);
        break;
    }
    return data;
}

void NvDsMemoryAllocator::Deallocate (void* data)
{
      switch (memType) {
        case NVDS_MEM_CUDA_PINNED:
          CHECK (cudaFreeHost (data));
          break;
        case NVDS_MEM_DEFAULT:
        case NVDS_MEM_CUDA_DEVICE:
          CHECK (cudaFree (data));
          break;
        case NVDS_MEM_CUDA_UNIFIED:
          CHECK (cudaFree (data));
          break;
        case NVDS_MEM_SYSTEM:
          free(data);
          break;
        default:
          printf ("[%s:%d]nvdstensor: invalid memory type (%d)\n",
                __func__, __LINE__, memType);
          break;
      }
}
