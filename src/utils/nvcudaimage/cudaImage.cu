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

#include "cudaImage.h"

inline bool CHECK_(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        //std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile;
        exit (-1);
        return false;
    }
    return true;
}

#define ck(call) CHECK_(call, __LINE__, __FILE__)

typedef union {
    uchar2 uc;
    uint16_t d;
} uchar2_uint16;

union BGRA {
    struct {
        uint8_t b, g, r, a;
    } c;
    uint32_t d;
};

union RGBA {
    struct {
        uint8_t r, g, b, a;
    } c;
    uint32_t d;
};

__device__ static float clamp_f(float x, float lower, float upper) {
	return x < lower ? lower : (x > upper ? upper : x);
}

__global__ static void resize_nv12_batch(cudaTextureObject_t texSrcLuma, cudaTextureObject_t texSrcChroma, uint8_t *pDstNv12,
        int nDstPitch, int nDstWidth, int nDstHeight, float fxScale, float fyScale, int nBatchSize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nDstWidth || y * 2 + 1 >= nDstHeight) {
        return;
    }

    uint8_t *p = pDstNv12 + x * 2 + y * 2 * nDstPitch;
    int hh = nDstHeight * 3 / 2;
    int nByte = nDstPitch * hh;
    int px = x * 2, py = y * 2;
    for (int i = 0; i < nBatchSize; i++) {
        *(uint16_t *)p = uchar2_uint16{uchar2{
            tex2D<uint8_t>(texSrcLuma, px * fxScale,       py * fyScale),
            tex2D<uint8_t>(texSrcLuma, (px + 1) * fxScale, py * fyScale)
        }}.d;
        *(uint16_t *)(p + nDstPitch) = uchar2_uint16{uchar2{
            tex2D<uint8_t>(texSrcLuma, px * fxScale,       (py + 1) * fyScale),
            tex2D<uint8_t>(texSrcLuma, (px + 1) * fxScale, (py + 1) * fyScale)
        }}.d;
        *(uint16_t *)(p + (nDstHeight - y) * nDstPitch) = uchar2_uint16{
            tex2D<uchar2>(texSrcChroma, x * fxScale, (hh * i + nDstHeight + y) * fyScale)
        }.d;
        p += nByte;
        py += hh;
    }
}

void resize_nv12_batch(const uint8_t *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, uint8_t *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream)
{
#if 1
    int hhSrc = (nSrcHeight * 3) / 2;
    int hhDst = (nDstHeight * 3) / 2;
    int nTiles = 1;
    int h = hhSrc * nBatchSize;

    while ((h + nTiles - 1) / nTiles > 65536) {
      nTiles++;
    }
    //std::cout << "nBatchSize = " << nBatchSize << std::endl;	
    //std::cout << "nTiles = " << nTiles << std::endl;	
    int batch_begin = nBatchSize / nTiles;
    int batch_end = nBatchSize - batch_begin * (nTiles-1);

    for (int iTile = 0; iTile < nTiles; ++iTile)
    {
      int bs = (iTile == nTiles - 1) ? batch_end : batch_begin;
      //std::cout << "bs = " << bs << std::endl;	
      const uint8_t *dpSrc_new = 	dpSrc + iTile * (batch_begin * hhSrc * nSrcPitch / sizeof(uint8_t));
      cudaResourceDesc resDesc = {};
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = (void *)dpSrc_new;
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uint8_t>();
      resDesc.res.pitch2D.width = nSrcWidth;
      resDesc.res.pitch2D.height = bs * hhSrc;
      resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

      cudaTextureDesc texDesc = {};
      texDesc.filterMode = cudaFilterModePoint;
      texDesc.readMode = cudaReadModeElementType;

      cudaTextureObject_t texLuma = 0;
      ck(cudaCreateTextureObject(&texLuma, &resDesc, &texDesc, NULL));

      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
      resDesc.res.pitch2D.width /= 2;

      cudaTextureObject_t texChroma = 0;
      ck(cudaCreateTextureObject(&texChroma, &resDesc, &texDesc, NULL));

      uint8_t *dpDst_new = dpDst + iTile * (batch_begin * hhDst * nDstPitch / sizeof(uint8_t));

      resize_nv12_batch<<<dim3((nDstWidth/2 + 15) / 16, (nDstHeight/2 + 3) / 4), dim3(16, 4), 0, stream>>>(texLuma,
          texChroma, dpDst_new, nDstPitch, nDstWidth,
          nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight, nBatchSize);

      ck(cudaStreamSynchronize(stream)); //TODO

      ck(cudaDestroyTextureObject(texLuma));
      ck(cudaDestroyTextureObject(texChroma));
    }

#else
    int hhSrc = nSrcHeight * 3 / 2;

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = (void *)dpSrc;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uint8_t>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = hhSrc * nBatchSize;
    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texLuma = 0;
    ck(cudaCreateTextureObject(&texLuma, &resDesc, &texDesc, NULL));

    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    resDesc.res.pitch2D.width /= 2;

    cudaTextureObject_t texChroma = 0;
    ck(cudaCreateTextureObject(&texChroma, &resDesc, &texDesc, NULL));

    resize_nv12_batch<<<dim3((nDstWidth/2 + 15) / 16, (nDstHeight/2 + 3) / 4), dim3(16, 4), 0, stream>>>(texLuma, texChroma, dpDst,
        nDstPitch, nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight, nBatchSize);
   	ck(cudaStreamSynchronize(stream)); //TODO

    ck(cudaDestroyTextureObject(texLuma));
    ck(cudaDestroyTextureObject(texChroma));
#endif
}

// TODO, stride
__global__ static void resize_rgb_planar_batch(cudaTextureObject_t texSrc, float *pDst, int nDstStride, int nDstWidth, int nDstHeight, float fxScale, float fyScale, int nBatchSize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (x > nDstWidth || y > nDstHeight) {
		return;
	}
	
	for (int i = 0; i < nBatchSize; ++i) {
		float *pDst_s = pDst + i * 3 * nDstStride * nDstHeight;
		// r	
		float *pR = pDst_s + 0 * nDstStride * nDstHeight + y * nDstStride + x;
		*pR = tex2D<float>(texSrc, x * fxScale, (nDstHeight * 3 * i + 0 * nDstHeight + y) * fyScale);
		// g	
		float *pG = pDst_s + 1 * nDstStride * nDstHeight + y * nDstStride + x;
		*pG = tex2D<float>(texSrc, x * fxScale, (nDstHeight * 3 * i + 1 * nDstHeight + y) * fyScale);
		// b
		float *pB = pDst_s + 2 * nDstStride * nDstHeight + y * nDstStride + x;
		*pB = tex2D<float>(texSrc, x * fxScale, (nDstHeight * 3 * i + 2 * nDstHeight + y) * fyScale);
	}
}

void resize_rgb_planar_batch(float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream)
{
  //cudaDeviceProp deviceProp;
  //cudaGetDeviceProperties(&deviceProp, dev);
  //int maxTexture2D[2] = deviceProp.maxTexture2D[2];

  int nTiles = 1;
  int h = nSrcHeight * 3 * nBatchSize;
  while ((h + nTiles - 1) / nTiles > 65536) {
    nTiles++;
  }
  //std::cout << "nBatchSize = " << nBatchSize << std::endl;	
  //std::cout << "nTiles = " << nTiles << std::endl;	
  int batch_begin = nBatchSize / nTiles;
  int batch_end = nBatchSize - batch_begin * (nTiles-1);

  for (int iTile = 0; iTile < nTiles; ++iTile) {
    int bs = (iTile == nTiles - 1) ? batch_end : batch_begin;
    //std::cout << "bs = " << bs << std::endl;	
    float *dpSrc_new = 	dpSrc + iTile * (batch_begin * 3 * nSrcHeight * nSrcPitch / sizeof(float));

    cudaResourceDesc resDesc = {};
    resDesc.res.pitch2D.devPtr = dpSrc_new;
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = bs * 3 * nSrcHeight;

    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texSrc = 0;
    ck(cudaCreateTextureObject(&texSrc, &resDesc, &texDesc, NULL));

    //resize_rgb_planar_batch<<<dim3((nDstWidth + 15) / 16, (nDstHeight + 15) / 16), dim3(16, 16), 0, stream>>>(texSrc, dpDst, nDstPitch / sizeof(float),
    //						nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight, nBatchSize);
    float *dpDst_new = dpDst + iTile * (batch_begin * 3 * nDstHeight * nDstPitch / sizeof(float));
    resize_rgb_planar_batch<<<dim3((nDstWidth + 15) / 16, (nDstHeight + 15) / 16), dim3(16, 16), 0, stream>>>(texSrc, dpDst_new, nDstPitch / sizeof(float),
        nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight, bs);
    ck(cudaStreamSynchronize(stream));

    ck(cudaDestroyTextureObject(texSrc));
  }
}

// Resize RGBA Batch
__global__ static void resize_rgba_float_planar_batch(cudaTextureObject_t texSrc, float *pDst, int nDstStride, int nDstWidth, int nDstHeight, float fxScale, float fyScale, int nBatchSize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (x > nDstWidth || y > nDstHeight) {
		return;
	}

  // Seperate RGB Channels
	for (int i = 0; i < nBatchSize; ++i) {
		float *pDst_s = pDst + i * 4 * nDstStride * nDstHeight;
		// r	
		float *pR = pDst_s + 0 * nDstStride * nDstHeight + y * nDstStride + x;
		*pR = tex2D<float>(texSrc, x * fxScale, (nDstHeight * 4 * i + 0 * nDstHeight + y) * fyScale);
		// g	
		float *pG = pDst_s + 1 * nDstStride * nDstHeight + y * nDstStride + x;
		*pG = tex2D<float>(texSrc, x * fxScale, (nDstHeight * 4 * i + 1 * nDstHeight + y) * fyScale);
		// b
		float *pB = pDst_s + 2 * nDstStride * nDstHeight + y * nDstStride + x;
		*pB = tex2D<float>(texSrc, x * fxScale, (nDstHeight * 4 * i + 2 * nDstHeight + y) * fyScale);
	}
}

void resize_rgba_float_planar_batch(float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream)
{
  //cudaDeviceProp deviceProp;
  //cudaGetDeviceProperties(&deviceProp, dev);
  //int maxTexture2D[2] = deviceProp.maxTexture2D[2];

  int nTiles = 1;
  int h = nSrcHeight * 4 * nBatchSize;
  while ((h + nTiles - 1) / nTiles > 65536) {
    nTiles++;
  }
  //std::cout << "nBatchSize = " << nBatchSize << std::endl;	
  //std::cout << "nTiles = " << nTiles << std::endl;	
  int batch_begin = nBatchSize / nTiles;
  int batch_end = nBatchSize - batch_begin * (nTiles-1);

  for (int iTile = 0; iTile < nTiles; ++iTile) {
    int bs = (iTile == nTiles - 1) ? batch_end : batch_begin;
    //std::cout << "bs = " << bs << std::endl;	
    float *dpSrc_new = 	dpSrc + iTile * (batch_begin * 4 * nSrcHeight * nSrcPitch / sizeof(float));

    cudaResourceDesc resDesc = {};
    resDesc.res.pitch2D.devPtr = dpSrc_new;
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = bs * 4 * nSrcHeight;

    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texSrc = 0;
    ck(cudaCreateTextureObject(&texSrc, &resDesc, &texDesc, NULL));

    //resize_rgba_float_planar_batch<<<dim3((nDstWidth + 15) / 16, (nDstHeight + 15) / 16), dim3(16, 16), 0, stream>>>(texSrc, dpDst, nDstPitch / sizeof(float),
    //						nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight, nBatchSize);
    float *dpDst_new = dpDst + iTile * (batch_begin * 4 * nDstHeight * nDstPitch / sizeof(float));
    resize_rgba_float_planar_batch<<<dim3((nDstWidth + 15) / 16, (nDstHeight + 15) / 16), dim3(16, 16), 0, stream>>>(texSrc, dpDst_new, nDstPitch / sizeof(float),
        nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight, bs);
    ck(cudaStreamSynchronize(stream));

    ck(cudaDestroyTextureObject(texSrc));
  }
}


// BT601
__device__ static float3 yuv2bgr_f(uint8_t y, uint8_t u, uint8_t v, float scale_factor) {
    float3 bgr{};
	bgr.x = scale_factor * clamp_f(1.1644f * (y - 16.0f) + 2.0172f * (u - 128.0f) + 0.0f * (v - 128.0f), 0.0f, 255.0f);
	bgr.y = scale_factor * clamp_f(1.1644f * (y - 16.0f) + (-0.3918f) * (u - 128.0f) + (-0.8130f) * (v - 128.0f), 0.0f, 255.0f);
	bgr.z = scale_factor * clamp_f(1.1644f * (y - 16.0f) + 0.0f * (u - 128.0f) + 1.5960f * (v - 128.0f), 0.0f, 255.0f);
	return bgr;
}

__device__ static float3 yuv2bgr_f_mean(uint8_t y, uint8_t u, uint8_t v, float scale_factor, float3 mean) {
    float3 bgr{};
	bgr.x = scale_factor * clamp_f(1.1644f * (y - 16.0f) + 2.0172f * (u - 128.0f) + 0.0f * (v - 128.0f), 0.0f, 255.0f) - mean.x;
	bgr.y = scale_factor * clamp_f(1.1644f * (y - 16.0f) + (-0.3918f) * (u - 128.0f) + (-0.8130f) * (v - 128.0f), 0.0f, 255.0f) - mean.y;
	bgr.z = scale_factor * clamp_f(1.1644f * (y - 16.0f) + 0.0f * (u - 128.0f) + 1.5960f * (v - 128.0f), 0.0f, 255.0f) - mean.z;
	return bgr;
}


__device__ static void nv12_to_bgr_planar_batch(uint8_t *pNv12, int nNv12Pitch, float *pBgr, int nRgbPitch, int nWidth, int nHeight, int x, int y, float scale_factor) {
    uchar2 luma01, luma23, uv;
    uint8_t *pSrc = pNv12 + x * 2 + y * 2 * nNv12Pitch;
    *(uint16_t *)&luma01 = *(uint16_t *)pSrc;
    *(uint16_t *)&luma23 = *(uint16_t *)(pSrc + nNv12Pitch);
    *(uint16_t *)&uv = *(uint16_t *)(pSrc + (nHeight - y) * nNv12Pitch);
    float3 bgr0, bgr1, bgr2, bgr3;

    bgr0 = yuv2bgr_f(luma01.x, uv.x, uv.y, scale_factor);
    bgr1 = yuv2bgr_f(luma01.y, uv.x, uv.y, scale_factor);
    bgr2 = yuv2bgr_f(luma23.x, uv.x, uv.y, scale_factor);
    bgr3 = yuv2bgr_f(luma23.y, uv.x, uv.y, scale_factor);

    uint8_t *pDst01 = (uint8_t *)pBgr + x * 4 * 2 + y * 2 * nRgbPitch, *pDst23 = pDst01 + nRgbPitch;
    // B
    *(float2 *)pDst01 = float2{bgr0.x, bgr1.x};
    *(float2 *)pDst23 = float2{bgr2.x, bgr3.x};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // G
    *(float2 *)pDst01 = float2{bgr0.y, bgr1.y};
    *(float2 *)pDst23 = float2{bgr2.y, bgr3.y};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // R
    *(float2 *)pDst01 = float2{bgr0.z, bgr1.z};
    *(float2 *)pDst23 = float2{bgr2.z, bgr3.z};
}

__device__ static void nv12_to_bgr_planar_batch_with_mean(uint8_t *pNv12, int nNv12Pitch, float *pBgr, int nRgbPitch, int nWidth, int nHeight, int x, int y, float scale_factor, float *mean_data) {
    uchar2 luma01, luma23, uv;
    uint8_t *pSrc = pNv12 + x * 2 + y * 2 * nNv12Pitch;
    *(uint16_t *)&luma01 = *(uint16_t *)pSrc;
    *(uint16_t *)&luma23 = *(uint16_t *)(pSrc + nNv12Pitch);
    *(uint16_t *)&uv = *(uint16_t *)(pSrc + (nHeight - y) * nNv12Pitch);
    float3 bgr0, bgr1, bgr2, bgr3;
    float3 mean0, mean1, mean2, mean3;

    mean0 = *((float3 *) ((float *) mean_data + 3 * (x + y * nWidth)));
    mean1 = *((float3 *) ((float *) mean_data + 3 * ((x + 1) + y * nWidth)));
    mean2 = *((float3 *) ((float *) mean_data + 3 * (x + (y + 1) * nWidth)));
    mean3 = *((float3 *) ((float *) mean_data + 3 * ((x + 1) + (y + 1) * nWidth)));

    bgr0 = yuv2bgr_f_mean(luma01.x, uv.x, uv.y, scale_factor, mean0);
    bgr1 = yuv2bgr_f_mean(luma01.y, uv.x, uv.y, scale_factor, mean1);
    bgr2 = yuv2bgr_f_mean(luma23.x, uv.x, uv.y, scale_factor, mean2);
    bgr3 = yuv2bgr_f_mean(luma23.y, uv.x, uv.y, scale_factor, mean3);

    uint8_t *pDst01 = (uint8_t *)pBgr + x * 4 * 2 + y * 2 * nRgbPitch, *pDst23 = pDst01 + nRgbPitch;

    // B
    *(float2 *)pDst01 = float2{bgr0.x, bgr1.x};
    *(float2 *)pDst23 = float2{bgr2.x, bgr3.x};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // G
    *(float2 *)pDst01 = float2{bgr0.y, bgr1.y};
    *(float2 *)pDst23 = float2{bgr2.y, bgr3.y};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // R
    *(float2 *)pDst01 = float2{bgr0.z, bgr1.z};
    *(float2 *)pDst23 = float2{bgr2.z, bgr3.z};
}

__device__ static void nv12_to_rgb_planar_batch(uint8_t *pNv12, int nNv12Pitch, float *pBgr, int nRgbPitch, int nWidth, int nHeight, int x, int y, float scale_factor) {
    uchar2 luma01, luma23, uv;
    uint8_t *pSrc = pNv12 + x * 2 + y * 2 * nNv12Pitch;
    *(uint16_t *)&luma01 = *(uint16_t *)pSrc;
    *(uint16_t *)&luma23 = *(uint16_t *)(pSrc + nNv12Pitch);
    *(uint16_t *)&uv = *(uint16_t *)(pSrc + (nHeight - y) * nNv12Pitch);
    float3 bgr0, bgr1, bgr2, bgr3;

    bgr0 = yuv2bgr_f(luma01.x, uv.x, uv.y, scale_factor);
    bgr1 = yuv2bgr_f(luma01.y, uv.x, uv.y, scale_factor);
    bgr2 = yuv2bgr_f(luma23.x, uv.x, uv.y, scale_factor);
    bgr3 = yuv2bgr_f(luma23.y, uv.x, uv.y, scale_factor);

    uint8_t *pDst01 = (uint8_t *)pBgr + x * 4 * 2 + y * 2 * nRgbPitch, *pDst23 = pDst01 + nRgbPitch;
    // R
    *(float2 *)pDst01 = float2{bgr0.z, bgr1.z};
    *(float2 *)pDst23 = float2{bgr2.z, bgr3.z};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // G
    *(float2 *)pDst01 = float2{bgr0.y, bgr1.y};
    *(float2 *)pDst23 = float2{bgr2.y, bgr3.y};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // B
    *(float2 *)pDst01 = float2{bgr0.x, bgr1.x};
    *(float2 *)pDst23 = float2{bgr2.x, bgr3.x};
}

__device__ static void nv12_to_rgb_planar_batch_with_mean(uint8_t *pNv12, int nNv12Pitch, float *pBgr, int nRgbPitch, int nWidth, int nHeight, int x, int y, float scale_factor, float *mean_data) {
    uchar2 luma01, luma23, uv;
    uint8_t *pSrc = pNv12 + x * 2 + y * 2 * nNv12Pitch;
    *(uint16_t *)&luma01 = *(uint16_t *)pSrc;
    *(uint16_t *)&luma23 = *(uint16_t *)(pSrc + nNv12Pitch);
    *(uint16_t *)&uv = *(uint16_t *)(pSrc + (nHeight - y) * nNv12Pitch);
    float3 bgr0, bgr1, bgr2, bgr3;
    float3 mean0, mean1, mean2, mean3;

    mean0 = *((float3 *) ((float *) mean_data + 3 * (x + y * nWidth)));
    mean1 = *((float3 *) ((float *) mean_data + 3 * ((x + 1) + y * nWidth)));
    mean2 = *((float3 *) ((float *) mean_data + 3 * (x + (y + 1) * nWidth)));
    mean3 = *((float3 *) ((float *) mean_data + 3 * ((x + 1) + (y + 1) * nWidth)));

    bgr0 = yuv2bgr_f_mean(luma01.x, uv.x, uv.y, scale_factor, mean0);
    bgr1 = yuv2bgr_f_mean(luma01.y, uv.x, uv.y, scale_factor, mean1);
    bgr2 = yuv2bgr_f_mean(luma23.x, uv.x, uv.y, scale_factor, mean2);
    bgr3 = yuv2bgr_f_mean(luma23.y, uv.x, uv.y, scale_factor, mean3);

    uint8_t *pDst01 = (uint8_t *)pBgr + x * 4 * 2 + y * 2 * nRgbPitch, *pDst23 = pDst01 + nRgbPitch;
    // R
    *(float2 *)pDst01 = float2{bgr0.z, bgr1.z};
    *(float2 *)pDst23 = float2{bgr2.z, bgr3.z};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // G
    *(float2 *)pDst01 = float2{bgr0.y, bgr1.y};
    *(float2 *)pDst23 = float2{bgr2.y, bgr3.y};
    pDst01 += nHeight * nRgbPitch;
    pDst23 += nHeight * nRgbPitch;
    // B
    *(float2 *)pDst01 = float2{bgr0.x, bgr1.x};
    *(float2 *)pDst23 = float2{bgr2.x, bgr3.x};
}

__global__ static void nv12_to_bgr_planar_batch_kernel(uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, float scale_factor) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
        return;
    }

    int nByteNv12 = nHeight * nNv12Pitch * 3 / 2, nByteRgb = nHeight * nRgbPitch * 3;
    for (int i = 0; i < nBatchSize; i++) {
        nv12_to_bgr_planar_batch(pNv12 + i * nByteNv12, nNv12Pitch, (float *)((uint8_t *)pRgb + i * nByteRgb), nRgbPitch, nWidth, nHeight, x, y, scale_factor);
    }
}

__global__ static void nv12_to_bgr_planar_batch_with_mean_kernel(uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, float scale_factor, float *mean_data) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
        return;
    }

    int nByteNv12 = nHeight * nNv12Pitch * 3 / 2, nByteRgb = nHeight * nRgbPitch * 3;
    for (int i = 0; i < nBatchSize; i++) {
        nv12_to_bgr_planar_batch_with_mean(pNv12 + i * nByteNv12, nNv12Pitch, (float *)((uint8_t *)pRgb + i * nByteRgb), nRgbPitch, nWidth, nHeight, x, y, scale_factor, mean_data);
    }
}

__global__ static void nv12_to_rgb_planar_batch_kernel(uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, float scale_factor) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
        return;
    }

    int nByteNv12 = nHeight * nNv12Pitch * 3 / 2, nByteRgb = nHeight * nRgbPitch * 3;
    for (int i = 0; i < nBatchSize; i++) {
        nv12_to_rgb_planar_batch(pNv12 + i * nByteNv12, nNv12Pitch, (float *)((uint8_t *)pRgb + i * nByteRgb), nRgbPitch, nWidth, nHeight, x, y, scale_factor);
    }
}

__global__ static void nv12_to_rgb_planar_batch_with_mean_kernel(uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, float scale_factor, float *mean_data) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
        return;
    }

    int nByteNv12 = nHeight * nNv12Pitch * 3 / 2, nByteRgb = nHeight * nRgbPitch * 3;
    for (int i = 0; i < nBatchSize; i++) {
        nv12_to_rgb_planar_batch_with_mean(pNv12 + i * nByteNv12, nNv12Pitch, (float *)((uint8_t *)pRgb + i * nByteRgb), nRgbPitch, nWidth, nHeight, x, y, scale_factor, mean_data);
    }
}

void nv12_to_bgr_planar_batch(uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, bool bSwap, cudaStream_t stream, float scale_factor, float *mean_data) {
  if (bSwap) {
    if (mean_data)
      nv12_to_rgb_planar_batch_with_mean_kernel<<<dim3((nWidth/2+15)/16, (nHeight/2+3)/4), dim3(16,4), 0, stream>>>(pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight, nBatchSize, scale_factor, mean_data);
    else
      nv12_to_rgb_planar_batch_kernel<<<dim3((nWidth/2+15)/16, (nHeight/2+3)/4), dim3(16,4), 0, stream>>>(pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight, nBatchSize, scale_factor);
  }
   else {
     if (mean_data) {
    nv12_to_bgr_planar_batch_with_mean_kernel<<<dim3((nWidth/2+15)/16, (nHeight/2+3)/4), dim3(16,4), 0, stream>>>(pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight, nBatchSize, scale_factor, mean_data);
     } else {
    nv12_to_bgr_planar_batch_kernel<<<dim3((nWidth/2+15)/16, (nHeight/2+3)/4), dim3(16,4), 0, stream>>>(pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight, nBatchSize, scale_factor);
     }
   }
}
__global__ void
gray_to_grayf_planar_batch (CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, float scalefactor, int batchSize)
{
    float *pdata = (float *)cuda_buf;
    unsigned char *psrcdata = (unsigned char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = row*width+col;
    int imgSize = width*height;

    if (col < width && row < height)
    {
      for (int i = 0; i < batchSize; i++)
        pdata[offset+imgSize*i] = ((float)psrcdata[offset+imgSize*i] * scalefactor);
        //printf ("offset %d. f = %f input %d mean %f final %f\n", offset, scalefactor, psrcdata[offset], 0.0, pdata[offset]);
    }
}

__global__ void
gray_to_grayf_planar_batch_with_mean (CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, float scalefactor, float *meanData, int batchSize)
{
    float *pdata = (float *)cuda_buf;
    unsigned char *psrcdata = (unsigned char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = row*width+col;
    int imgSize = width*height;

    if (col < width && row < height)
    {
      for (int i = 0; i < batchSize; i++)
        pdata[offset+imgSize*i] =
          ((float)(psrcdata[offset+imgSize*i]) * scalefactor) - meanData[offset];
    }
}


int gray_to_grayf_batch(CUdeviceptr pDevPtr, int width, int height,
      void* cuda_buf, int pitch, float scalefactor, float *meanData, cudaStream_t &stream, int batchSize)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocks((width+31)/threadsPerBlock.x, (height+31)/threadsPerBlock.y);

    if (meanData)
      gray_to_grayf_planar_batch_with_mean<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
          height, cuda_buf, scalefactor, meanData, batchSize);
    else
      gray_to_grayf_planar_batch<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
          height, cuda_buf, scalefactor, batchSize);

    return 0;
}

// NV12 -> BGRA

// ------------------------

template<class T>
__device__ static T clamp(float x, float lower, float upper) {
	T ret = (T)(x < lower ? lower : (x > upper ? upper : x));
	return ret;
}

template<class T>
__device__ static T yuv2rgb(uint8_t y, uint8_t u, uint8_t v) {
    T rgb{};
	rgb.c.r = clamp<uint8_t>(1.1644f * (y - 16.0f) + 0.0f * (u - 128.0f) + 1.5960f * (v - 128.0f), 0.0f, 255.0f);
	rgb.c.g = clamp<uint8_t>(1.1644f * (y - 16.0f) + (-0.3918f) * (u - 128.0f) + (-0.8130f) * (v - 128.0f), 0.0f, 255.0f);
	rgb.c.b = clamp<uint8_t>(1.1644f * (y - 16.0f) + 2.0172f * (u - 128.0f) + 0.0f * (v - 128.0f), 0.0f, 255.0f);
	return rgb;
}

template<class T>
__global__ static void nv12_to_rgb_kernel_1(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pRgb, int nBgraPitch, int nWidth, int nHeight) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
		return;
	}

	int ic0 = (x * 2) * sizeof(T) + (y * 2) * nBgraPitch,
		ic1 = ic0 + sizeof(T),
		ic2 = ic0 + nBgraPitch,
		ic3 = ic2 + sizeof(T),
		iy0 = (x * 2) + (y * 2) * nNv12Pitch,
		iy1 = iy0 + 1,
		iy2 = iy0 + nNv12Pitch,
		iy3 = iy2 + 1,
		iu = (x * 2) + y * nNv12Pitch + nHeight * nNv12Pitch,
		iv = iu + 1;
	uint8_t y0 = pNv12[iy0], y1 = pNv12[iy1], y2 = pNv12[iy2], y3 = pNv12[iy3], u = pNv12[iu], v = pNv12[iv];

	*((T *)(pRgb + ic0)) = yuv2rgb<T>(y0, u, v);
	*((T *)(pRgb + ic1)) = yuv2rgb<T>(y1, u, v);
	*((T *)(pRgb + ic2)) = yuv2rgb<T>(y2, u, v);
	*((T *)(pRgb + ic3)) = yuv2rgb<T>(y3, u, v);
}

void nv12_to_bgra(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, cudaStream_t stream) {
    nv12_to_rgb_kernel_1<BGRA><<<dim3((nWidth/2 + 31) / 32, (nHeight/2 + 15) / 16), dim3(32, 16), 0, stream>>>(dpNv12, nNv12Pitch, dpRgb, nBgraPitch, nWidth, nHeight);
}


void nv12_to_rgba(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, cudaStream_t stream) {
    nv12_to_rgb_kernel_1<RGBA><<<dim3((nWidth/2 + 31) / 32, (nHeight/2 + 15) / 16), dim3(32, 16), 0, stream>>>(dpNv12, nNv12Pitch, dpRgb, nBgraPitch, nWidth, nHeight);
}


__global__ static void nv12_to_rgb_kernel_3p(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pRgb, int nBgraPitch, int nWidth, int nHeight) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
    RGBA r0;
	if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
		return;
	}

	int ic0 = (x * 2) * 3 + (y * 2) * nBgraPitch,
		ic1 = ic0 + 3,
		ic2 = ic0 + nBgraPitch,
		ic3 = ic2 + 3,
		iy0 = (x * 2) + (y * 2) * nNv12Pitch,
		iy1 = iy0 + 1,
		iy2 = iy0 + nNv12Pitch,
		iy3 = iy2 + 1,
		iu = (x * 2) + y * nNv12Pitch + nHeight * nNv12Pitch,
		iv = iu + 1;
	uint8_t y0 = pNv12[iy0], y1 = pNv12[iy1], y2 = pNv12[iy2], y3 = pNv12[iy3], u = pNv12[iu], v = pNv12[iv];

    r0 = yuv2rgb<RGBA>(y0, u, v);
    /*r1 = yuv2rgb<RGBA>(y1, u, v);
    r2 = yuv2rgb<RGBA>(y2, u, v);
    r3 = yuv2rgb<RGBA>(y3, u, v);
	*/
    *((pRgb + ic0)) = r0.c.r;  *((pRgb + ic0 + 1)) = r0.c.g; *((pRgb + ic0 + 2)) = r0.c.b; 
    r0 = yuv2rgb<RGBA>(y1, u, v);
    *((pRgb + ic1)) = r0.c.r;  *((pRgb + ic1 + 1)) = r0.c.g; *((pRgb + ic1 + 2)) = r0.c.b; 
    r0 = yuv2rgb<RGBA>(y2, u, v);
    *((pRgb + ic2)) = r0.c.r;  *((pRgb + ic2 + 1)) = r0.c.g; *((pRgb + ic2 + 2)) = r0.c.b; 
    r0 = yuv2rgb<RGBA>(y3, u, v);
    *((pRgb + ic3)) = r0.c.r;  *((pRgb + ic3 + 1)) = r0.c.g; *((pRgb + ic3 + 2)) = r0.c.b; 

}
__global__ static void nv12_to_bgr_kernel_3p_crop(const uint8_t *pNv12, int nNv12Pitch, int nSrcWidth, int nSrcHeight, 
                                                        uint8_t *pRgb, int nBgraPitch, int nTopLeftX, int nTopLeftY, int nWidth, int nHeight) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
    RGBA r0;
	if (x * 2 + 1 >= nSrcWidth || y * 2 + 1 >= nSrcHeight ||
        x * 2 + 1  < nTopLeftX || y * 2 + 1 < nTopLeftY   ||
       x * 2 + 1  > nTopLeftX+nWidth || y * 2 + 1 > nTopLeftY+nHeight 
     )
    {
		return;
	}

	int ic0 = (x * 2 - nTopLeftX) * 3 + (y * 2 - nTopLeftY) * nBgraPitch,
		ic1 = ic0 + 3,
		ic2 = ic0 + nBgraPitch,
		ic3 = ic2 + 3,
		iy0 = (x * 2) + (y * 2) * nNv12Pitch,
		iy1 = iy0 + 1,
		iy2 = iy0 + nNv12Pitch,
		iy3 = iy2 + 1,
		iu = (x * 2) + y * nNv12Pitch + nSrcHeight * nNv12Pitch,
		iv = iu + 1;
	uint8_t y0 = pNv12[iy0], y1 = pNv12[iy1], y2 = pNv12[iy2], y3 = pNv12[iy3], u = pNv12[iu], v = pNv12[iv];

    r0 = yuv2rgb<RGBA>(y0, u, v);
    /*r1 = yuv2rgb<RGBA>(y1, u, v);
    r2 = yuv2rgb<RGBA>(y2, u, v);
    r3 = yuv2rgb<RGBA>(y3, u, v);
	*/
    *((pRgb + ic0+2)) = r0.c.r;  *((pRgb + ic0 + 1)) = r0.c.g; *((pRgb + ic0)) = r0.c.b; 
    r0 = yuv2rgb<RGBA>(y1, u, v);
    *((pRgb + ic1+2)) = r0.c.r;  *((pRgb + ic1 + 1)) = r0.c.g; *((pRgb + ic1)) = r0.c.b; 
    r0 = yuv2rgb<RGBA>(y2, u, v);
    *((pRgb + ic2+2)) = r0.c.r;  *((pRgb + ic2 + 1)) = r0.c.g; *((pRgb + ic2)) = r0.c.b; 
    r0 = yuv2rgb<RGBA>(y3, u, v);
      *((pRgb + ic3+2)) = r0.c.r;  *((pRgb + ic3 + 1)) = r0.c.g; *((pRgb + ic3)) = r0.c.b; 
}

__global__ static void nv12_to_bgr_kernel_3p_scale_crop(const uint8_t *pNv12, int nNv12Pitch, int nSrcWidth, int nSrcHeight, 
                                                               uint8_t *pRgb, int nBgraPitch, int nDstWidth, int nDstHeight,
                                                               float fScaleX, float fScaleY,                                                        
                                                        int nTopLeftX, int nTopLeftY, int nWidth, int nHeight) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
    RGBA r0, r1, r2, r3, r4, r5;
	if ((x >= nDstWidth || y >= nDstHeight))
    {
		return;
	}
	
	float fsx = x*fScaleX+nTopLeftX;
	float fsy = y*fScaleY+nTopLeftY;
	
	
    int sx1 = x*fScaleX+nTopLeftX;
    int sy2 = y*fScaleY+1+nTopLeftY;
    int sy1 = y*fScaleY+nTopLeftY;
    int sx2 = x*fScaleX+1+nTopLeftX;
    
    if (fsx < 0 || fsx >= nSrcWidth ||
        fsy < 0 || fsy >= nSrcHeight)
       {
        return;
        }
        
    
    if (sy1 >= nSrcHeight) {sy1 = nSrcHeight-1;}
    if (sx1 >= nSrcWidth) {sx1 = nSrcWidth-1;}
    if (sx2 >= nSrcWidth) {sx2 = nSrcWidth-1;}
    
    if (sy2 >= nSrcHeight) {sy2 = nSrcHeight-1;}
    
	float wtx = (fsx - sx1);
	float wty = (fsy - sy1);
	
	int ic0 = x * 3 + y* nBgraPitch,
		
		iy0 = sx1 + sy1 * nNv12Pitch,
		iy1 = iy0 + 1,
		iy2 = iy0 + nNv12Pitch,
		iy3 = iy2 + 1, iu = 0, iv =0;
		if (sx1%2)
		{
	       iu = (sx1-1)  + nSrcHeight * nNv12Pitch;
		}
		else
		{
		  iu = sx1+ nSrcHeight * nNv12Pitch;
		}
		
		if(sy1%2)
		{
			iu += ((sy1-1)*nNv12Pitch)/2;
		}
		else
		{
			iu += ((sy1)*nNv12Pitch)/2;
		} 

		iv = iu + 1;
		
		
	uint8_t y0 = pNv12[iy0], y1 = pNv12[iy1], y2 = pNv12[iy2], y3 = pNv12[iy3], 
	//u = (uint8_t)128, v = (uint8_t)128;  
	u = pNv12[iu], v = pNv12[iv];

    r0 = yuv2rgb<RGBA>(y0, u, v);
    r1 = yuv2rgb<RGBA>(y1, u, v);
    r2 = yuv2rgb<RGBA>(y2, u, v);
    r3 = yuv2rgb<RGBA>(y3, u, v);
    
    r4.c.r = r0.c.r*(1 -wtx) + r1.c.r*(wtx);
    r4.c.g = r0.c.g*(1 -wtx) + r1.c.g*(wtx);
    r4.c.b = r0.c.b*(1 -wtx) + r1.c.b*(wtx);
    
    r5.c.r = r2.c.r*(1 -wtx) + r3.c.r*(wtx);
    r5.c.g = r2.c.g*(1 -wtx) + r3.c.g*(wtx);
    r5.c.b = r2.c.b*(1 -wtx) + r3.c.b*(wtx);

	r4.c.r = r4.c.r*(1 -wty) + r5.c.r*(wty);
	r4.c.g = r4.c.g*(1 -wty) + r5.c.g*(wty);
	r4.c.b = r4.c.b*(1 -wty) + r5.c.b*(wty);
    
    *((pRgb + ic0+2)) = r4.c.r;  *((pRgb + ic0 + 1)) = r4.c.g; *((pRgb + ic0)) = r4.c.b; 
  


}

void nv12_to_bgr_crop(const uint8_t *dpNv12, int nNv12Pitch, int nSrcWidth, int nSrcHeight, uint8_t *dpRgb, int nBgraPitch, int nTopLeftX, int nTopLeftY, int nWidth, int nHeight, cudaStream_t stream) {
    nv12_to_bgr_kernel_3p_crop<<<dim3((nSrcWidth/2 + 31) / 32, (nSrcHeight/2 + 15) / 16), dim3(32, 16), 0, stream>>>(dpNv12, nNv12Pitch, nSrcWidth, nSrcHeight, dpRgb, nBgraPitch, nTopLeftX, nTopLeftY, nWidth, nHeight);
}

void nv12_to_bgr_scale_crop(const uint8_t *dpNv12, int nNv12Pitch, int nSrcWidth, int nSrcHeight, 
                            uint8_t *dpRgb, int nBgraPitch, int nDstWidth, int nDstHeight,
                            int nTopLeftX, int nTopLeftY, int nWidth, int nHeight, cudaStream_t stream) {
                            
    float scaleX = (1.0f * nWidth) / nDstWidth;
    float scaleY = (1.0f * nHeight) / nDstHeight;
    if (scaleX < scaleY)
    {
    	scaleX = scaleY;
    }
    nv12_to_bgr_kernel_3p_scale_crop<<<dim3((nDstWidth + 31) / 32, (nDstHeight + 15) / 16), dim3(32, 16), 0, stream>>>
    (dpNv12, nNv12Pitch, nSrcWidth, nSrcHeight, // Src params 
      dpRgb, nBgraPitch, nDstWidth, nDstHeight, // Dst params
      scaleX, scaleX, //Scale 
      nTopLeftX, nTopLeftY, nWidth, nHeight); //Crop
}

void nv12_to_rgb(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, cudaStream_t stream) {
    nv12_to_rgb_kernel_3p<<<dim3((nWidth/2 + 31) / 32, (nHeight/2 + 15) / 16), dim3(32, 16), 0, stream>>>(dpNv12, nNv12Pitch, dpRgb, nBgraPitch, nWidth, nHeight);
}
// -----------------------------

template<class T>
__device__ static void nv12_to_rgba_kernel_batch_1(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pRgb, int nBgraPitch, int nWidth, int nHeight, int x, int y, bool bSwap)
{
#if 0
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
    return;
  }
#endif

  int ic0 = (x * 2) * sizeof(T) + (y * 2) * nBgraPitch,
      ic1 = ic0 + sizeof(T),
      ic2 = ic0 + nBgraPitch,
      ic3 = ic2 + sizeof(T),
      iy0 = (x * 2) + (y * 2) * nNv12Pitch,
      iy1 = iy0 + 1,
      iy2 = iy0 + nNv12Pitch,
      iy3 = iy2 + 1,
      iu = (x * 2) + y * nNv12Pitch + nHeight * nNv12Pitch,
      iv = iu + 1;
  uint8_t y0 = pNv12[iy0], y1 = pNv12[iy1], y2 = pNv12[iy2], y3 = pNv12[iy3], u = pNv12[iu], v = pNv12[iv];

  *((T *)(pRgb + ic0)) = yuv2rgb<T>(y0, u, v);
  *((T *)(pRgb + ic1)) = yuv2rgb<T>(y1, u, v);
  *((T *)(pRgb + ic2)) = yuv2rgb<T>(y2, u, v);
  *((T *)(pRgb + ic3)) = yuv2rgb<T>(y3, u, v);
}

template<class T>
__global__ static void nv12_to_rgba_kernel_batch(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight, int nBatchSize, bool bSwap)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
        return;
    }

    int nByteNv12 = nHeight * nNv12Pitch * 3 / 2, nByteRgb = nHeight * nRgbPitch;
    for (int i = 0; i < nBatchSize; i++) {
      if (bSwap == 0)
        nv12_to_rgba_kernel_batch_1<BGRA>(pNv12 + i * nByteNv12, nNv12Pitch, ((uint8_t *)pRgb + i * nByteRgb), nRgbPitch, nWidth, nHeight, x, y, bSwap);
      else if (bSwap == 1)
        nv12_to_rgba_kernel_batch_1<RGBA>(pNv12 + i * nByteNv12, nNv12Pitch, ((uint8_t *)pRgb + i * nByteRgb), nRgbPitch, nWidth, nHeight, x, y, bSwap);
    }
}

void nv12_to_bgra_batch(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream, bool bSwap) {
  nv12_to_rgba_kernel_batch<BGRA><<<dim3((nWidth/2 + 31) / 32, (nHeight/2 + 15) / 16), dim3(32, 16), 0, stream>>>(dpNv12, nNv12Pitch, dpRgb, nBgraPitch, nWidth, nHeight, nBatchSize, bSwap);
}

void nv12_to_rgba_batch(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream, bool bSwap) {
  nv12_to_rgba_kernel_batch<RGBA><<<dim3((nWidth/2 + 31) / 32, (nHeight/2 + 15) / 16), dim3(32, 16), 0, stream>>>(dpNv12, nNv12Pitch, dpRgb, nBgraPitch, nWidth, nHeight, nBatchSize, 1);
}
