/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "nvds_opticalflow_meta.h"
#include "nvofvisual_draw.h"

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#define FLOW_SCALER   32.0
/* 32-bit float point representation*/
#define BYTE_SIZE_32FC2 8

__device__ bool IsNaN(float f);
__device__ bool IsFlowCorrect(float x, float y);
__device__ void SetColors(int r, int g, int b, int k, int colorwheel[][3]);
__device__ void ComputeColor(float fx, float fy, uint8_t* pix);

__device__ bool IsNaN(float f)
{
    uint32_t u = *((uint32_t*)&f);
    return (u & 0x7fffffff) > 0x7f800000;
}

__device__ bool IsFlowCorrect(float x, float y)
{
    return !IsNaN(x) && !IsNaN(y) && fabs(x) < 1e9 && fabs(y) < 1e9;
}

__device__ void SetColors(int r, int g, int b, int k, int colorwheel[][3])
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

__device__ void ComputeColor(float fx, float fy, uint8_t* pix)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static int colorwheel[NCOLS][3];

    if (first)
    {
        int i;
        int k = 0;

        for (i = 0; i < RY; i++)
            SetColors(255, 255 * i / RY, 0, k++, colorwheel);
        for (i = 0; i < YG; i++)
            SetColors(255 - 255 * i / YG, 255, 0, k++, colorwheel);
        for (i = 0; i < GC; i++)
            SetColors(0, 255, 255 * i / GC, k++, colorwheel);
        for (i = 0; i < CB; i++)
            SetColors(0, 255 - 255 * i / CB, 255, k++, colorwheel);
        for (i = 0; i < BM; i++)
            SetColors(255 * i / BM, 0, 255, k++, colorwheel);
        for (i = 0; i < MR; i++)
            SetColors(255, 0, 255 - 255 * i / MR, k++, colorwheel);

        first = false;
    }

    float rad = sqrtf(fx * fx + fy * fy);
    float a = atan2f(-fy, -fx) / M_PI;
    float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % NCOLS;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++)
    {
        float col0 = colorwheel[k0][b] / 255.0f;
        float col1 = colorwheel[k1][b] / 255.0f;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75f; // out of range
        pix[2 - b] = (int)(255.0f * col);
    }
    pix[3] = 0xff;
}


__global__ void
kernel_draw_of_cuda (CUdeviceptr pSrcPtr, int width, int height,
                void* pDstPtr, int pitch, float maxrad)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
    	NvOFFlowVector *in_flow = (NvOFFlowVector *)pSrcPtr;

    	uint8_t* dst = (uint8_t*) pDstPtr;
    	uint8_t* rgba = &dst[(row*pitch)+(col*4)];

        float x = ((float)in_flow[(row*width)+col].flowx)/FLOW_SCALER;
    	float y = ((float)in_flow[(row*width)+col].flowy)/FLOW_SCALER;

        if (IsFlowCorrect(x, y))
            ComputeColor(x / maxrad, y / maxrad, rgba);
    }
}

void DrawOpticalFlow_Cuda (void* in_mv_data, void* dst, int cols, int rows, int pitch, float maxmotion, cudaStream_t stream)
{
  int width = cols;
  int height = rows;

  dim3 threadsPerBlock(32, 32);
  dim3 blocks((width+31)/threadsPerBlock.x, (height+31)/threadsPerBlock.y);

  kernel_draw_of_cuda <<<blocks, threadsPerBlock, 0, stream>>> ((CUdeviceptr)in_mv_data, width, height, dst, pitch, maxmotion);

}
