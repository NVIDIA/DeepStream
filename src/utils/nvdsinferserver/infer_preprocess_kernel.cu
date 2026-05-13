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
#include <cuda_fp16.h>
#include "infer_preprocess_kernel.h"
#include "infer_datatypes.h"
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <stdint.h>
#include <cassert>

#define THREADS_PER_BLOCK 512
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)
#define MAX_BLOCKS 64

#define MAKE_CONVERT_KEY(inType, outType)  \
  (((static_cast<uint32_t>(inType)&0xFFU)<<8)| \
  (static_cast<uint32_t>(outType)&0xFFU))

using namespace nvdsinferserver;

struct FloatArray {
    float d[NVDSINFER_MAX_PREPROC_CHANNELS];
};

struct IntArray {
    int d[NVDSINFER_MAX_PREPROC_CHANNELS];
};

template<typename T>
__device__ __forceinline__ T satTo(float v);

template<>
__device__ __forceinline__ int8_t satTo<int8_t>(float v) {
  return (int8_t)fminf(fmaxf(v, INT8_MIN), INT8_MAX);
}

template<>
__device__ __forceinline__ uint8_t satTo<uint8_t>(float v) {
  return (uint8_t)fminf(fmaxf(v, 0), UINT8_MAX);
}

template<>
__device__ __forceinline__ int16_t satTo<int16_t>(float v) {
  return (int16_t)fminf(fmaxf(v, INT16_MIN), INT16_MAX);
}

template<>
__device__ __forceinline__ uint16_t satTo<uint16_t>(float v) {
  return (uint16_t)fminf(fmaxf(v, 0), UINT16_MAX);
}

template<>
__device__ __forceinline__ int32_t satTo<int32_t>(float v) {
  return (int32_t)fminf(fmaxf(v, INT32_MIN), INT32_MAX);
}

template<>
__device__ __forceinline__ uint32_t satTo<uint32_t>(float v) {
  return (uint32_t)fminf(fmaxf(v, 0), UINT32_MAX);
}

template<>
__device__ __forceinline__ float satTo<float>(float v) {
  return v;
}

template<>
__device__ __forceinline__ half satTo<half>(float v) {
  return __float2half(v);
}

static __device__ __forceinline__ void FromNCHW(int idx,
    const int alignedC, const int alignedH, const int alignedW,
    int& nI, int& cI, int& hI, int& wI)
{
    int imageSize = alignedH*alignedW;
    int perBatch = imageSize * alignedC;
    nI = idx / perBatch;
    idx = idx % perBatch;
    cI = idx / imageSize;
    idx = idx % imageSize;
    hI = idx / alignedW;
    wI = idx % alignedW;
}

static __device__ __forceinline__ int getNCHWIdx(
    int nI, int cI, int hI, int wI, const BatchSurfaceInfo& info)
{
    return nI * info.dataSizePerBatch +
        (cI * info.heightAlign  + hI) * info.pitchPerRow + wI;
}

static __device__ __forceinline__ int getNHWCIdx(
    int nI, int cI, int hI, int wI, const BatchSurfaceInfo& info)
{
    return nI * info.dataSizePerBatch + hI * info.pitchPerRow +
      wI * info.channelAlign + cI;
}

template<typename InT, typename OutT>
__global__ void
NvDsInferConvert_NHWCToDestKernel(
    OutT *outBuffer,
    BatchSurfaceInfo outBufferInfo,
    InT *inBuffer,
    BatchSurfaceInfo inBufferInfo,
    int validN, int validC, int cropH, int cropW,  // crop area
    float scaleFactor,
    FloatArray meanOffsets,
    IntArray fromChannelIndices, // to[ci] = from[fromChannelIndices[ci]]
    int toNCHW) // toNCHW, true or false
{
    int totalThreads = gridDim.x * blockDim.x;
    int totalPixel = validN * validC * cropH * cropW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ni, ci, hi, wi;

    for (; idx < totalPixel; idx+= totalThreads) {
        FromNCHW(idx, validC, outBufferInfo.heightAlign, outBufferInfo.widthAlign,
                ni, ci, hi, wi);
        int fromCi = fromChannelIndices.d[ci];
        int inIdx = getNHWCIdx(ni, fromCi, hi, wi, inBufferInfo);
        float inVal = static_cast<float>(inBuffer[inIdx]);

        int outIdx;
        if (toNCHW)
            outIdx = getNCHWIdx(ni, ci, hi, wi, outBufferInfo);
        else
            outIdx = getNHWCIdx(ni, ci, hi, wi, outBufferInfo);

        outBuffer[outIdx] = satTo<OutT>(scaleFactor *
                (inVal - meanOffsets.d[ci]));
    }
}

template<typename InT, typename OutT>
__global__ void
NvDsInferConvert_NHWCToDestKernelWithMeanSubtraction(
    OutT *outBuffer,
    BatchSurfaceInfo outBufferInfo,
    InT *inBuffer,
    BatchSurfaceInfo inBufferInfo,
    int validN, int validC, int cropH, int cropW, // crop area
    float scaleFactor,
    float *meanData, // keep same order as output
    IntArray fromChannelIndices, // to[ci] = from[fromChannelIndices[ci]]
    int toNCHW) // toNCHW, true or false
{
    int totalThreads = gridDim.x * blockDim.x;
    int totalPixel = validN * validC * cropH * cropW;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ni, ci, hi, wi;
    for (; idx < totalPixel; idx+= totalThreads) {
        FromNCHW(idx, validC, outBufferInfo.heightAlign, outBufferInfo.widthAlign,
                ni, ci, hi, wi);
        int fromCi = fromChannelIndices.d[ci];
        int inIdx = getNHWCIdx(ni, fromCi, hi, wi, inBufferInfo);
        float inVal = static_cast<float>(inBuffer[inIdx]);

        int outIdx, meanIdx;
        if (toNCHW) {
            outIdx = getNCHWIdx(ni, ci, hi, wi, outBufferInfo);
            meanIdx = (ci * outBufferInfo.heightAlign + hi)* outBufferInfo.widthAlign + wi;
        } else {
            outIdx = getNHWCIdx(ni, ci, hi, wi, outBufferInfo);
            meanIdx = (hi * outBufferInfo.widthAlign + wi) * validC + ci;
        }

        outBuffer[outIdx] = satTo<OutT>(scaleFactor *
            (inVal - meanData[meanIdx]));
    }
}

template<typename InT, typename OutT>
void NvDsInferConvert_NHWCToDest(
    void *outBuffer,
    BatchSurfaceInfo outBufferInfo,
    void *inBuffer,
    BatchSurfaceInfo inBufferInfo,
    int validN, int validC, int cropH, int cropW,
    float scaleFactor,
    FloatArray meanOffsets,
    IntArray fromChannelIndices,
    int toNCHW,
    cudaStream_t stream)
{
  int blocks = (validN * outBufferInfo.widthAlign *
          outBufferInfo.heightAlign * validC +
          THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (blocks > MAX_BLOCKS)
    blocks = MAX_BLOCKS;

  NvDsInferConvert_NHWCToDestKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      static_cast<OutT*>(outBuffer), outBufferInfo,
      static_cast<InT*>(inBuffer), inBufferInfo, validN, validC, cropH, cropW,
      scaleFactor, meanOffsets, fromChannelIndices, toNCHW);
}

template<typename InT, typename OutT>
void NvDsInferConvert_NHWCToDestWithMeanSubtraction(
    void *outBuffer,
    BatchSurfaceInfo outBufferInfo,
    void *inBuffer,
    BatchSurfaceInfo inBufferInfo,
    int validN, int validC, int cropH, int cropW,
    float scaleFactor,
    float *meanData,
    IntArray fromChannelIndices,
    int toNCHW,
    cudaStream_t stream)
{
  int blocks = (validN * outBufferInfo.widthAlign *
          outBufferInfo.heightAlign * validC +
          THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (blocks > MAX_BLOCKS)
    blocks = MAX_BLOCKS;

  NvDsInferConvert_NHWCToDestKernelWithMeanSubtraction<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      static_cast<OutT*>(outBuffer), outBufferInfo,
      static_cast<InT*>(inBuffer), inBufferInfo, validN, validC, cropH, cropW,
      scaleFactor, meanData, fromChannelIndices, toNCHW);
}

typedef void (*NvDsInferConvertCuFunc)(
    void *outBuffer,
    BatchSurfaceInfo outBufferInfo,
    void *inBuffer,
    BatchSurfaceInfo inBufferInfo,
    int validN, int validC, int cropH, int cropW,
    float scaleFactor,
    FloatArray meanOffsets,
    IntArray fromChannelIndices,
    int toNCHW,
    cudaStream_t stream);

typedef void (*NvDsInferConvertWMeanBufFunc)(
    void *outBuffer,
    BatchSurfaceInfo outBufferInfo,
    void *inBuffer,
    BatchSurfaceInfo inBufferInfo,
    int validN, int validC, int cropH, int cropW,
    float scaleFactor,
    float *meanData,
    IntArray fromChannelIndices,
    int toNCHW,
    cudaStream_t stream);

struct NvDsInferConverFuncs{
    NvDsInferConvertCuFunc convertFunc;
    NvDsInferConvertWMeanBufFunc convertWMeanBufFunc;
};

std::unordered_map<uint32_t, NvDsInferConverFuncs> ConverCuFuncMaps = {
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kFp32),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, float>,
        &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, float>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kFp32),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, float>,
        &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, float>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kFp16),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, half>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, half>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kFp16),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, half>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, half>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kInt8),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, int8_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, int8_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kInt8),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, int8_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, int8_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kInt16),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, int16_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, int16_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kInt16),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, int16_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, int16_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kInt32),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, int32_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, int32_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kInt32),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, int32_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, int32_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kUint8),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, uint8_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, uint8_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kUint8),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, uint8_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, uint8_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kUint16),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, uint16_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, uint16_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kUint16),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, uint16_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, uint16_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kInt8, InferDataType::kUint32),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<int8_t, uint32_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<int8_t, uint32_t>}},
    {MAKE_CONVERT_KEY(InferDataType::kUint8, InferDataType::kUint32),
        NvDsInferConverFuncs{&NvDsInferConvert_NHWCToDest<uint8_t, uint32_t>,
         &NvDsInferConvert_NHWCToDestWithMeanSubtraction<uint8_t, uint32_t>}},
  };

bool
NvDsInferConvert_CxToPx(
    void *outBuffer, int outDataType,
    BatchSurfaceInfo outBufferInfo,
    void *inBuffer, int inDatatype,
    BatchSurfaceInfo inBufferInfo,
    int validN, int cropC, int cropH, int cropW,  // crop area
    float scaleFactor,
    float meanOffsets[NVDSINFER_MAX_PREPROC_CHANNELS],
    int fromChannelIndices[NVDSINFER_MAX_PREPROC_CHANNELS],
    int toNCHW,
    cudaStream_t stream)
{
    uint32_t conversionKey = MAKE_CONVERT_KEY(inDatatype, outDataType);
    auto iF = ConverCuFuncMaps.find(conversionKey);
    assert (iF != ConverCuFuncMaps.end());
    if (iF == ConverCuFuncMaps.end()) {
        return false;
    }
    FloatArray meanOffsetArray;
    std::copy(meanOffsets, meanOffsets + NVDSINFER_MAX_PREPROC_CHANNELS, meanOffsetArray.d);
    IntArray channelArray;
    std::copy(fromChannelIndices, fromChannelIndices + NVDSINFER_MAX_PREPROC_CHANNELS, channelArray.d);
    NvDsInferConvertCuFunc convertFn = iF->second.convertFunc;
    convertFn(outBuffer, outBufferInfo,
        inBuffer, inBufferInfo, validN, cropC, cropH, cropW,
        scaleFactor, meanOffsetArray, channelArray, toNCHW, stream);
    return (cudaGetLastError()==cudaSuccess);
}

bool
NvDsInferConvert_CxToPxWithMeanBuffer(
    void *outBuffer, int outDataType,
    BatchSurfaceInfo outBufferInfo,
    void *inBuffer, int inDatatype,
    BatchSurfaceInfo inBufferInfo,
    int validN, int cropC, int cropH, int cropW,  // crop area
    float scaleFactor,
    float *meanDataBuffer,
    int fromChannelIndices[NVDSINFER_MAX_PREPROC_CHANNELS],
    int toNCHW,
    cudaStream_t stream)
{
    uint32_t conversionKey = MAKE_CONVERT_KEY(inDatatype, outDataType);
    auto iF = ConverCuFuncMaps.find(conversionKey);
    assert (iF != ConverCuFuncMaps.end());
    if (iF == ConverCuFuncMaps.end()) {
        return false;
    }

    IntArray channelArray;
    std::copy(fromChannelIndices, fromChannelIndices + NVDSINFER_MAX_PREPROC_CHANNELS, channelArray.d);
    NvDsInferConvertWMeanBufFunc convertFn = iF->second.convertWMeanBufFunc;
    convertFn(outBuffer, outBufferInfo,
        inBuffer, inBufferInfo, validN, cropC, cropH, cropW,
        scaleFactor, meanDataBuffer, channelArray, toNCHW, stream);
    return (cudaGetLastError()==cudaSuccess);
}

