/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <ostream>
#include <stdio.h>
#include <cuda_runtime_api.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "NvCudaConvert.h"

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure NvCudaProc: " << status << " Line " << __LINE__ << std::endl;        \
        abort();                                        \
    }                                                   \
}

__global__ void
convertIntToFloatKernel_FloatData (CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, int pitch, float scalefactor)
{
    float *pdata = (float *)cuda_buf;
    unsigned char *psrcdata = (unsigned char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
      for (int k = 0; k < 3; k++)
      {
        pdata[width * height * k + row * width + col] =
          ((float)*(psrcdata + row * pitch + col * 4 + (k)) * scalefactor);
      }
    }
}

__global__ void
convertIntToFloatKernel_FloatData_WithMean (CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, int pitch, float scalefactor, float *meanData)
{
    float *pdata = (float *)cuda_buf;
    unsigned char *psrcdata = (unsigned char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
      for (int k = 0; k < 3; k++)
      {
        pdata[width * height * k + row * width + col] =
          ((float)*(psrcdata + row * pitch + col * 4 + (k)) * scalefactor) - (float)meanData[ (row*width*3) + (col*3) + k];
      }
    }
}

__global__ void
convertIntToFloatKernel_FloatDataBGR (CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, int pitch, float scalefactor)
{
    float *pdata = (float *)cuda_buf;
    unsigned char *psrcdata = (unsigned char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
      pdata[width * height * 0 + row * width + col] =
        ((float)*(psrcdata + row * pitch + col * 4 + (2)) * scalefactor);
      pdata[width * height * 1 + row * width + col] =
        ((float)*(psrcdata + row * pitch + col * 4 + (1)) * scalefactor);
      pdata[width * height * 2 + row * width + col] =
        ((float)*(psrcdata + row * pitch + col * 4 + (0)) * scalefactor);
    }
}

__global__ void
convertIntToFloatKernel_FloatData_WithMeanBGR (CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, int pitch, float scalefactor, float *meanData)
{
    float *pdata = (float *)cuda_buf;
    unsigned char *psrcdata = (unsigned char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
      pdata[width * height * 0 + row * width + col] =
        ((float)*(psrcdata + row * pitch + col * 4 + (2)) * scalefactor) - (float)meanData[ (row*width*3) + (col*3) + 0];
      pdata[width * height * 1 + row * width + col] =
        ((float)*(psrcdata + row * pitch + col * 4 + (1)) * scalefactor) - (float)meanData[ (row*width*3) + (col*3) + 1];
      pdata[width * height * 2 + row * width + col] =
        ((float)*(psrcdata + row * pitch + col * 4 + (0)) * scalefactor) - (float)meanData[ (row*width*3) + (col*3) + 2];
    }
}

int convertIntToFloat_FloatData(CUdeviceptr pDevPtr, int width, int height,
        void* cuda_buf, int pitch, float scalefactor, float *meanData, cudaStream_t &stream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocks((width+31)/threadsPerBlock.x, (height+31)/threadsPerBlock.y);

    if (meanData)
      convertIntToFloatKernel_FloatData_WithMean<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
          height, cuda_buf, pitch, scalefactor, meanData);
    else
      convertIntToFloatKernel_FloatData<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
          height, cuda_buf, pitch, scalefactor);

    return 0;
}

int convertIntToFloat_FloatDataBGR(CUdeviceptr pDevPtr, int width, int height,
        void* cuda_buf, int pitch, float scalefactor, float *meanData, cudaStream_t &stream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocks((width+31)/threadsPerBlock.x, (height+31)/threadsPerBlock.y);

    if (meanData)
      convertIntToFloatKernel_FloatData_WithMeanBGR<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
          height, cuda_buf, pitch, scalefactor, meanData);
    else
      convertIntToFloatKernel_FloatDataBGR<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
          height, cuda_buf, pitch, scalefactor);

    return 0;
}


void rgba2floatrgb(void* pCudaInBuf, int width, int height,
    void* cuda_out_buf, float scalefactor, void *meanData, cudaStream_t &stream)
{
  convertIntToFloat_FloatData((CUdeviceptr) pCudaInBuf,
      width, height, cuda_out_buf, width*4, scalefactor, (float *) meanData, stream);

  return;
}

void rgba2floatbgr(void* pCudaInBuf, int width, int height,
    void* cuda_out_buf, float scalefactor, void *meanData, cudaStream_t &stream)
{
  convertIntToFloat_FloatDataBGR((CUdeviceptr) pCudaInBuf,
      width, height, cuda_out_buf, width*4, scalefactor, (float *) meanData, stream);

  return;
}
