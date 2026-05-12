/*
* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#include "drawBbox.h"

#define BBOX_R 0xff
#define BBOX_G 0x99
#define BBOX_B 0x33

__global__ void drawBoundingBoxunitalpha_kernel(uchar4 *pBGRA, const int nWidth, const int nHeight, const int stride, const int x_min, const int y_min, const int x_max, const int y_max,
     unsigned int r, unsigned int g, unsigned int b, unsigned int border_width) {
 	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
 	int nPixels = border_width;
 	int x_min_l = x_min > 0 ? x_min : 0;
 	int x_max_l = x_max > nWidth ? nWidth : x_max;
 	int y_min_l = y_min > 0 ? y_min : 0;
 	int y_max_l = y_max > nHeight ? nHeight : y_max;
 
 	if (nPixels > x_max_l - x_min_l) {
 		nPixels = 0;
 	}
 	if (nPixels > y_max_l - y_min_l) {
 		nPixels = 0;
 	}
 
 	if (idx_x > x_max_l || idx_y > y_max_l) {
 		return;
 	}
 
 	// left and right
 	bool bDraw = ((idx_x >= x_min_l && idx_x <= (x_min_l+nPixels))
 				|| (idx_x >= x_max_l-nPixels && idx_x <= x_max_l))
 				&& (idx_y >= y_min_l && idx_y <= y_max_l);
 	if (bDraw) {
 		pBGRA[idx_y*stride+idx_x].x = r;
 		pBGRA[idx_y*stride+idx_x].y = g;
 		pBGRA[idx_y*stride+idx_x].z = b;
 	}
 
 	// up and down
 	bDraw = ((idx_y >= y_min_l && idx_y <= y_min_l + nPixels)
 			|| (idx_y >= y_max_l - nPixels && idx_y <= y_max_l))
 			&& (idx_x >= x_min_l + nPixels && idx_x <= x_max_l - nPixels);
 	if (bDraw) {
 		pBGRA[idx_y*stride+idx_x].x = r;
 		pBGRA[idx_y*stride+idx_x].y = g;
 		pBGRA[idx_y*stride+idx_x].z = b;
 	}
}

void drawBoundingBox_cuda_unit_alpha(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b ,unsigned int border_width) {
	drawBoundingBoxunitalpha_kernel<<<dim3((nWidth+15)/16, (nHeight+15)/16), dim3(16, 16), 0, stream>>>((uchar4 *)pBGRA, nWidth, nHeight, nBgraPitch/sizeof(uchar4), x_min, y_min, x_max, y_max, r, g, b, border_width);
}

__global__ void drawBoundingBox_kernel(uchar4 *pBGRA, const int nWidth, const int nHeight, const int stride, const int x_min, const int y_min, const int x_max, const int y_max,
    unsigned int r, unsigned int g, unsigned int b, float a, unsigned int border_width) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int nPixels = border_width;
	int x_min_l = x_min > 0 ? x_min : 0;
	int x_max_l = x_max > nWidth ? nWidth : x_max;
	int y_min_l = y_min > 0 ? y_min : 0;
	int y_max_l = y_max > nHeight ? nHeight : y_max;

	if (nPixels > x_max_l - x_min_l) {
		nPixels = 0;
	}
	if (nPixels > y_max_l - y_min_l) {
		nPixels = 0;
	}

	if (idx_x > x_max_l || idx_y > y_max_l) {
		return;
	}

	// left and right
	bool bDraw = ((idx_x >= x_min_l && idx_x <= (x_min_l+nPixels))
				|| (idx_x >= x_max_l-nPixels && idx_x <= x_max_l))
				&& (idx_y >= y_min_l && idx_y <= y_max_l);
        float a_1 = 1.0 - a;
	if (bDraw) {
		pBGRA[idx_y*stride+idx_x].x = (unsigned char)(pBGRA[idx_y*stride+idx_x].x * a_1 + r * a);
		pBGRA[idx_y*stride+idx_x].y = (unsigned char)(pBGRA[idx_y*stride+idx_x].y * a_1 + g * a);
		pBGRA[idx_y*stride+idx_x].z = (unsigned char)(pBGRA[idx_y*stride+idx_x].z * a_1 + b * a);
	}

	// up and down
	bDraw = ((idx_y >= y_min_l && idx_y <= y_min_l + nPixels)
			|| (idx_y >= y_max_l - nPixels && idx_y <= y_max_l))
			&& (idx_x >= x_min_l + nPixels && idx_x <= x_max_l - nPixels);
	if (bDraw) {
		pBGRA[idx_y*stride+idx_x].x = (unsigned char)(pBGRA[idx_y*stride+idx_x].x * a_1 + r * a);
		pBGRA[idx_y*stride+idx_x].y = (unsigned char)(pBGRA[idx_y*stride+idx_x].y * a_1 + g * a);
		pBGRA[idx_y*stride+idx_x].z = (unsigned char)(pBGRA[idx_y*stride+idx_x].z * a_1 + b * a);
	}
}

void drawBoundingBox_cuda(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b, float  a ,unsigned int border_width) {
	drawBoundingBox_kernel<<<dim3((nWidth+15)/16, (nHeight+15)/16), dim3(16, 16), 0, stream>>>((uchar4 *)pBGRA, nWidth, nHeight, nBgraPitch/sizeof(uchar4), x_min, y_min, x_max, y_max, r, g, b, a, border_width);
}

__global__ void bboxAlphaFill_kernel(uchar4 *pBGRA, const int nWidth, const int nHeight, const int stride, const int x_min, const int y_min, const int x_max, const int y_max, float r, float g, float b, float a) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	int x_min_ = x_min > 0 ? x_min : 0;
	int y_min_ = y_min > 0 ? y_min : 0;
	int x_max_ = x_max > nWidth-1 ? nWidth-1 : x_max;
	int y_max_ = y_max > nHeight-1 ? nHeight-1 : y_max;

	bool bDraw = idx_x >= x_min_ && idx_x <= x_max_
					&& idx_y >= y_min_ && idx_y <= y_max_;
        float a_1 = 1.0 - a;
	if (bDraw) {
		pBGRA[idx_y*stride+idx_x].x = (unsigned char)(pBGRA[idx_y*stride+idx_x].x * a_1 + r * a);
		pBGRA[idx_y*stride+idx_x].y = (unsigned char)(pBGRA[idx_y*stride+idx_x].y * a_1 + g * a);
		pBGRA[idx_y*stride+idx_x].z = (unsigned char)(pBGRA[idx_y*stride+idx_x].z * a_1 + b * a);
	}

}

void bboxAlphaFill_cuda(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, unsigned int r, unsigned int g, unsigned int b, float a, cudaStream_t stream) {
	bboxAlphaFill_kernel<<<dim3((nWidth+15)/16, (nHeight+15)/16), dim3(16, 16), 0, stream>>>((uchar4 *)pBGRA, nWidth, nHeight, nBgraPitch/sizeof(uchar4), x_min, y_min, x_max, y_max, 1.0 * r, 1.0 * g, 1.0 * b, a);
}

__global__ void bgra_to_gray_kernel(uchar4 *dpBgra, int nBgraStride, uint8_t *dpGray, int nGrayStride, int nWidth, int nHeight) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	uchar4 pixel = dpBgra[idx_y * nBgraStride + idx_x];
	// gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
	float value = 0.2989 * pixel.z + 0.5870 * pixel.y + 0.1140 * pixel.x;
	dpGray[idx_y * nGrayStride + idx_x] = (uint8_t)value;
}


void bgra_to_gray(uint8_t *dpBgra, const int nBgraPitch, uint8_t *dpGray, const int nGrayPitch, const int nWidth, const int nHeight, cudaStream_t stream) {
	bgra_to_gray_kernel<<<dim3((nWidth+15)/16, (nHeight+15)/16), dim3(16, 16), 0, stream>>>((uchar4 *)dpBgra, nBgraPitch / sizeof(uchar4), dpGray, nGrayPitch / sizeof(uint8_t), nWidth, nHeight);
}

__device__ static float clamp_f(float x, float lower, float upper) {
	return x < lower ? lower : (x > upper ? upper : x);
}

// BT601
__device__ static float3 yuv2bgr_f(uint8_t y, uint8_t u, uint8_t v) {
    float3 bgr{};
	bgr.x = clamp_f(1.1644f * (y - 16.0f) + 2.0172f * (u - 128.0f) + 0.0f * (v - 128.0f), 0.0f, 255.0f);
	bgr.y = clamp_f(1.1644f * (y - 16.0f) + (-0.3918f) * (u - 128.0f) + (-0.8130f) * (v - 128.0f), 0.0f, 255.0f);
	bgr.z = clamp_f(1.1644f * (y - 16.0f) + 0.0f * (u - 128.0f) + 1.5960f * (v - 128.0f), 0.0f, 255.0f);
	return bgr;
}


__device__ static void nv12_to_gray_batch(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pGray, int nGrayPitch, int nWidth, int nHeight, int x, int y) {
    uchar2 luma01, luma23, uv;
    const uint8_t *pSrc = pNv12 + x * 2 + y * 2 * nNv12Pitch;
    *(uint16_t *)&luma01 = *(uint16_t *)pSrc;
    *(uint16_t *)&luma23 = *(uint16_t *)(pSrc + nNv12Pitch);
    *(uint16_t *)&uv = *(uint16_t *)(pSrc + (nHeight - y) * nNv12Pitch);
    float3 bgr0, bgr1, bgr2, bgr3;
    bgr0 = yuv2bgr_f(luma01.x, uv.x, uv.y);
    bgr1 = yuv2bgr_f(luma01.y, uv.x, uv.y);
    bgr2 = yuv2bgr_f(luma23.x, uv.x, uv.y);
    bgr3 = yuv2bgr_f(luma23.y, uv.x, uv.y);
    // Note: gray pitch = gray stride
	uint8_t *pDst01 = pGray + x * 2 + y * 2 * nGrayPitch, *pDst23 = pDst01 + nGrayPitch;

	// gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
	uint8_t gray0 = (uint8_t)(bgr0.z * 0.2989 + bgr0.y * 0.5870 + bgr0.x * 0.1140);
	uint8_t gray1 = (uint8_t)(bgr1.z * 0.2989 + bgr1.y * 0.5870 + bgr1.x * 0.1140);
	uint8_t gray2 = (uint8_t)(bgr2.z * 0.2989 + bgr2.y * 0.5870 + bgr2.x * 0.1140);
	uint8_t gray3 = (uint8_t)(bgr3.z * 0.2989 + bgr3.y * 0.5870 + bgr3.x * 0.1140);
	*(uchar2 *)pDst01 = uchar2{gray0, gray1};
	*(uchar2 *)pDst23 = uchar2{gray2, gray3};

}

__global__ static void nv12_to_gray_batch_kernel(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pGray, int nGrayPitch, int nWidth, int nHeight, int nBatchSize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x * 2 + 1 >= nWidth || y * 2 + 1 >= nHeight) {
        return;
    }

    int nStrideNv12 = nHeight * (nNv12Pitch / 1) * 3 / 2, nStrideGray = nHeight * (nGrayPitch / 1);
    for (int i = 0; i < nBatchSize; i++) {
        nv12_to_gray_batch(pNv12 + i * nStrideNv12, nNv12Pitch, pGray + i * nStrideGray, nGrayPitch, nWidth, nHeight, x, y);
    }
}

void nv12_to_gray_batch(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pGray, int nGrayPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream) {
	nv12_to_gray_batch_kernel<<<dim3((nWidth/2+15)/16, (nHeight/2+3)/4), dim3(16,4), 0, stream>>>(pNv12, nWidth*sizeof(uint8_t), pGray, nWidth*sizeof(uint8_t), nWidth, nHeight, nBatchSize);
}


