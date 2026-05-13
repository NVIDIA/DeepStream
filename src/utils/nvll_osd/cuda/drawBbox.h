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
#ifndef DRAW_BBOX_H
#define DRAW_BBOX_H

#include <cstdint>
//#include <stdint.h>
#include <cuda_runtime.h>


void drawBoundingBox_cuda_unit_alpha(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b, unsigned int border_width);

void drawBoundingBox_cuda(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, cudaStream_t stream,
    unsigned int r, unsigned int g, unsigned int b, float a, unsigned int border_width);

void bboxAlphaFill_cuda(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, unsigned int r, unsigned int g, unsigned int b, float a, cudaStream_t stream);

void bgra_to_gray(uint8_t *dpBgra, const int nBgraPitch, uint8_t *dpGray, const int nGrayPitch, const int nWidth, const int nHeight, cudaStream_t stream);

void nv12_to_gray_batch(const uint8_t *pNv12, int nNv12Pitch, uint8_t *pGray, int nGrayPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream);

#endif // DRAW_BBOX_H
