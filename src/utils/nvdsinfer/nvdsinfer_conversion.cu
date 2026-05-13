/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cuda.h>
#include "nvdsinfer_conversion.h"

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)

__global__ void
NvDsInferConvert_CxToP3FloatKernel(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + k];
        }
    }
}

__global__ void
NvDsInferConvert_CxToL3FloatKernel(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + k];
        }
    }
}

__global__ void
NvDsInferConvert_CxToP3FloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * ((float) inBuffer[row * pitch + col * inputPixelSize + k] -
                meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void
NvDsInferConvert_CxToL3FloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * ((float) inBuffer[row * pitch + col * inputPixelSize + k] -
                meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void
NvDsInferConvert_CxToP3RFloatKernel(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + (2 - k)];
        }
    }
}

__global__ void
NvDsInferConvert_CxToL3RFloatKernel(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + (2 - k)];
        }
    }
}

__global__ void
NvDsInferConvert_CxToP3RFloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * ((float) inBuffer[row * pitch + col * inputPixelSize + (2 - k)] -
                meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void
NvDsInferConvert_CxToL3RFloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * ((float) inBuffer[row * pitch + col * inputPixelSize + (2 - k)] -
                meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void
NvDsInferConvert_C1ToP1FloatKernel(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        outBuffer[row * width + col] = scaleFactor * inBuffer[row * pitch + col];
    }
}

__global__ void
NvDsInferConvert_C1ToP1FloatKernelWithMeanSubtraction(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        outBuffer[row * width + col] =
            scaleFactor * ((float) inBuffer[row * pitch + col] -
            meanDataBuffer[(row * width) + col]);
    }
}

__global__ void
NvDsInferConvert_FtFTensorKernel(
        float *outBuffer,
        float *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        outBuffer[row * width + col] = scaleFactor * inBuffer[row * width + col];
    }
}

__global__ void
NvDsInferConvert_FtFTensorKernelWithMeanSubtraction(
        float *outBuffer,
        float *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        outBuffer[row * width + col] =
            scaleFactor * ((float) inBuffer[row * width + col] -
            meanDataBuffer[(row * width) + col]);
    }
}

void
NvDsInferConvert_C3ToP3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToP3FloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToP3FloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C3ToL3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToL3FloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToL3FloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C4ToP3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToP3FloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToP3FloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C4ToL3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToL3FloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToL3FloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C3ToP3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToP3RFloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToP3RFloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C3ToL3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToL3RFloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToL3RFloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C4ToP3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToP3RFloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToP3RFloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C4ToL3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_CxToL3RFloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    }
    else
    {
        NvDsInferConvert_CxToL3RFloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void
NvDsInferConvert_C1ToP1Float(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_C1ToP1FloatKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, scaleFactor);
    }
    else
    {
        NvDsInferConvert_C1ToP1FloatKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, scaleFactor, meanDataBuffer);
    }

}

//TODO add channel information, current implementation is only for single channel
void
NvDsInferConvert_FtFTensor(
        float *outBuffer,
        float *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    if (meanDataBuffer == NULL)
    {
        NvDsInferConvert_FtFTensorKernel <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, scaleFactor);
    }
    else
    {
        NvDsInferConvert_FtFTensorKernelWithMeanSubtraction <<<blocks, threadsPerBlock, 0, stream>>>
            (outBuffer, inBuffer, width, height, pitch, scaleFactor, meanDataBuffer);
    }
}
