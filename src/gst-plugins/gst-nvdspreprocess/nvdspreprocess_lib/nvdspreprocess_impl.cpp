/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <cuda.h>

#include "nvtx3/nvToolsExtCudaRt.h"

#include "nvdspreprocess_impl.h"
#include "nvdspreprocess_conversion.h"

/** enable to debug transformation in/out files
 *  with DEBUG_TENSOR in plugin enabled
 */
//#define DEBUG_LIB

/** This file contains the preprocessing for network requirements.
 * It does mean subtraction and normalization of input pixels
 * from the given parameters in config file.
 */
CudaStream::CudaStream(uint flag, int priority)
{
    cudaError_t err = cudaStreamCreateWithPriority(&m_Stream, flag, priority);
    if (err != cudaSuccess) {
      printf ("cudaStreamCreateWithPriority failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
    }
}

CudaStream::~CudaStream()
{
  if (m_Stream != nullptr)
  {
      cudaError_t err = cudaStreamDestroy(m_Stream);
      if ( err != cudaSuccess) {
        printf ("cudaStreamDestroy failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
      }
  }
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t size) : CudaBuffer(size)
{
  cudaError_t err = cudaMalloc(&m_Buf, size);
  if (err != cudaSuccess) {
    printf ("cudaMalloc failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
  }

  m_Size = size;
}

CudaDeviceBuffer::~CudaDeviceBuffer()
{
  if (m_Buf != nullptr)
  {
    cudaError_t err = cudaFree(m_Buf);
    if (err != cudaSuccess) {
        printf ("cudaFree failed with err %d : %s\n", (int)err, cudaGetErrorName(err));
    }
  }
}

NvDsPreProcessTensorImpl::NvDsPreProcessTensorImpl(const NvDsPreProcessNetworkSize& size,
  NvDsPreProcessFormat format,
  int id)
  : m_UniqueID(id),
    m_NetworkSize(size),
    m_NetworkInputFormat(format)
{
}

bool
NvDsPreProcessTensorImpl::setScaleOffsets(float scale, const std::vector<float>& offsets)
{
  if (!offsets.empty() && m_NetworkSize.channels != (uint32_t)offsets.size())
  {
    return false;
  }

  m_Scale = scale;
  if (!offsets.empty())
  {
    m_ChannelMeans.assign(offsets.begin(), offsets.begin() + m_NetworkSize.channels);
  }
  return true;
}

bool
NvDsPreProcessTensorImpl::setMeanFile(const std::string& file)
{
  if (!file_accessible(file))
    return false;
  m_MeanFile = file;
  return true;
}

bool
NvDsPreProcessTensorImpl::setInputOrder(const NvDsPreProcessNetworkInputOrder order)
{
  m_InputOrder = order;
  return true;
}

/* Read the mean image ppm file and copy the mean image data to the mean
 * data buffer allocated on the device memory.
 */
NvDsPreProcessStatus
NvDsPreProcessTensorImpl::readMeanImageFile()
{
  std::ifstream infile(m_MeanFile, std::ifstream::binary);
  size_t size =
      m_NetworkSize.width * m_NetworkSize.height * m_NetworkSize.channels;
  uint8_t tempMeanDataChar[size];
  float tempMeanDataFloat[size];
  cudaError_t cudaReturn;

  if (!infile.good())
  {
      printf("Could not open mean image file '%s\n'", safeStr(m_MeanFile));
      return NVDSPREPROCESS_CONFIG_FAILED;
  }

  std::string magic, max;
  unsigned int h, w;
  infile >> magic >> w >> h >> max;

  if (magic != "P3" && magic != "P6")
  {
      printf("Magic PPM identifier check failed\n");
      return NVDSPREPROCESS_CONFIG_FAILED;
  }

  if (w != m_NetworkSize.width || h != m_NetworkSize.height)
  {
      printf(
          "Mismatch between ppm mean image resolution(%d x %d) and "
          "network resolution(%d x %d)\n",
          w, h, m_NetworkSize.width, m_NetworkSize.height);
      return NVDSPREPROCESS_CONFIG_FAILED;
  }

  infile.get();
  infile.read((char*)tempMeanDataChar, size);
  if (infile.gcount() != (int)size || infile.fail())
  {
      printf("Failed to read sufficient bytes from mean file\n");
      return NVDSPREPROCESS_CONFIG_FAILED;
  }

  for (size_t i = 0; i < size; i++)
  {
      tempMeanDataFloat[i] = (float)tempMeanDataChar[i];
  }

  assert(m_MeanDataBuffer);
  cudaReturn = cudaMemcpy(m_MeanDataBuffer->ptr(), tempMeanDataFloat,
      size * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaReturn != cudaSuccess)
  {
      printf("Failed to copy mean data to mean data buffer (%s)\n",
          cudaGetErrorName(cudaReturn));
      return NVDSPREPROCESS_CUDA_ERROR;
  }

  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
NvDsPreProcessTensorImpl::allocateResource()
{
  if (!m_MeanFile.empty() || m_ChannelMeans.size() > 0)
  {
      /* Mean Image File specified. Allocate the mean image buffer on device
        * memory. */
      m_MeanDataBuffer = std::make_unique<CudaDeviceBuffer>(
              (size_t) m_NetworkSize.width * m_NetworkSize.height * m_NetworkSize.channels *
              sizeof(float));

      if (!m_MeanDataBuffer || !m_MeanDataBuffer->ptr())
      {
          printf("Failed to allocate cuda buffer for mean image");
          return NVDSPREPROCESS_CUDA_ERROR;
      }
  }

  /* Read the mean image file (PPM format) if specified and copy the
    * contents into the buffer. */
  if (!m_MeanFile.empty())
  {
      if (!file_accessible(m_MeanFile))
      {
          printf(
              "Cannot access mean image file '%s'", safeStr(m_MeanFile));
          return NVDSPREPROCESS_CONFIG_FAILED;
      }
      NvDsPreProcessStatus status = readMeanImageFile();
      if (status != NVDSPREPROCESS_SUCCESS)
      {
          printf("Failed to read mean image file\n");
          return status;
      }
  }
  /* Create the mean data buffer from per-channel offsets. */
  else if (m_ChannelMeans.size() > 0)
  {
      /* Make sure the number of offsets are equal to the number of input
        * channels. */
      if ((uint32_t)m_ChannelMeans.size() != m_NetworkSize.channels)
      {
          printf(
              "Number of offsets(%d) not equal to number of input "
              "channels(%d)",
              (int)m_ChannelMeans.size(), m_NetworkSize.channels);
          return NVDSPREPROCESS_CONFIG_FAILED;
      }

      std::vector<float> meanData(m_NetworkSize.channels *
                                  m_NetworkSize.width * m_NetworkSize.height);
      for (size_t j = 0; j < m_NetworkSize.width * m_NetworkSize.height;
            j++)
      {
          for (size_t i = 0; i < m_NetworkSize.channels; i++)
          {
              meanData[j * m_NetworkSize.channels + i] = m_ChannelMeans[i];
          }
      }
      cudaError_t cudaReturn =
          cudaMemcpy(m_MeanDataBuffer->ptr(), meanData.data(),
              meanData.size() * sizeof(float), cudaMemcpyHostToDevice);
      if (cudaReturn != cudaSuccess)
      {
          printf("Failed to copy mean data to mean data cuda buffer(%s)",
              cudaGetErrorName(cudaReturn));
          return NVDSPREPROCESS_CUDA_ERROR;
      }
  }

  /* Create the cuda stream on which pre-processing jobs will be executed. */
  m_PreProcessStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);
  if (!m_PreProcessStream || !m_PreProcessStream->ptr())
  {
      printf("Failed to create preprocessor cudaStream");
      return NVDSPREPROCESS_CUDA_ERROR;
  }

  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
NvDsPreProcessTensorImpl::syncStream()
{
  if (m_PreProcessStream)
  {
      if (cudaSuccess != cudaStreamSynchronize(*m_PreProcessStream))
          return NVDSPREPROCESS_CUDA_ERROR;
  }
  return NVDSPREPROCESS_SUCCESS;
}

static inline unsigned bytesPerElement(NvDsDataType data_type)
{
  switch (data_type) {
    case NvDsDataType_FP32:
    case NvDsDataType_UINT32:
    case NvDsDataType_INT32:
      return 4;
    case NvDsDataType_UINT8:
    case NvDsDataType_INT8:
      return 1;
    case NvDsDataType_FP16:
      return 2;
    case NvDsDataType_UINT64:
    case NvDsDataType_INT64:
      return 8;
    default:
      printf("Wrong tensor data type");
      return 0;
  }
}

NvDsPreProcessStatus NvDsPreProcessTensorImpl::prepare_tensor(
    NvDsPreProcessBatch* batch, CustomTensorParams &tensorParam, void*& devBuf)
{
  unsigned int batch_size = batch->units.size();

  NvDsPreProcessConvertFcn convertFcn = nullptr;
  NvDsPreProcessConvertFcnHalf convertFcnHalf = nullptr;

  /* Find the required conversion function. */
  switch (m_NetworkInputFormat)
  {
      case NvDsPreProcessFormat_RGB:
          switch (batch->scaling_pool_format)
          {
              case NvDsPreProcessFormat_RGB:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          convertFcn = NvDsPreProcessConvert_C3ToP3Float;
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C3ToL3Float;
                          break;
                      default:
                          break;
                  }
                  break;
              case NvDsPreProcessFormat_BGR:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          convertFcn = NvDsPreProcessConvert_C3ToP3RFloat;
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C3ToL3RFloat;
                          break;
                      default:
                          break;
                  }
                  break;
              case NvDsPreProcessFormat_RGBA:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          if (tensorParam.params.data_type == NvDsDataType_FP16) {
                              convertFcnHalf = NvDsPreProcessConvert_C4ToP3Half;
                          } else {
                              convertFcn = NvDsPreProcessConvert_C4ToP3Float;
                          }
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C4ToL3Float;
                          break;
                      default:
                          break;
                  }
                  break;
              case NvDsPreProcessFormat_BGRx:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          convertFcn = NvDsPreProcessConvert_C4ToP3RFloat;
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C4ToL3RFloat;
                          break;
                      default:
                          break;
                  }
                  break;
              default:
                  printf("Input format conversion is not supported");
                  return NVDSPREPROCESS_INVALID_PARAMS;
          }
          break;
      case NvDsPreProcessFormat_BGR:
          switch (batch->scaling_pool_format)
          {
              case NvDsPreProcessFormat_RGB:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          convertFcn = NvDsPreProcessConvert_C3ToP3RFloat;
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C3ToL3RFloat;
                          break;
                      default:
                          break;
                  }
                  break;
              case NvDsPreProcessFormat_BGR:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          convertFcn = NvDsPreProcessConvert_C3ToP3Float;
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C3ToL3Float;
                          break;
                      default:
                          break;
                  }
                  break;
              case NvDsPreProcessFormat_RGBA:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          if (tensorParam.params.data_type == NvDsDataType_FP16) {
                              convertFcnHalf = NvDsPreProcessConvert_C4ToP3RHalf;
                          } else {
                              convertFcn = NvDsPreProcessConvert_C4ToP3RFloat;
                          }
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C4ToL3RFloat;
                          break;
                      default:
                          break;
                  }
                  break;
              case NvDsPreProcessFormat_BGRx:
                  switch (m_InputOrder)
                  {
                      case NvDsPreProcessNetworkInputOrder_kNCHW:
                          convertFcn = NvDsPreProcessConvert_C4ToP3Float;
                          break;
                      case NvDsPreProcessNetworkInputOrder_kNHWC:
                          convertFcn = NvDsPreProcessConvert_C4ToL3Float;
                          break;
                      default:
                          break;
                  }
                  break;
              default:
                  printf("Input format conversion is not supported");
                  return NVDSPREPROCESS_INVALID_PARAMS;
          }
          break;
      case NvDsPreProcessFormat_GRAY:
          if (batch->scaling_pool_format != NvDsPreProcessFormat_GRAY)
          {
              printf("Input frame format is not GRAY.");
              return NVDSPREPROCESS_INVALID_PARAMS;
          }
          convertFcn = NvDsPreProcessConvert_C1ToP1Float;
          break;
      default:
          printf("Unsupported network input format");
          return NVDSPREPROCESS_INVALID_PARAMS;
  }

  /* For each frame in the input batch convert/copy to the input binding
    * buffer. */
  for (unsigned int i = 0; i < batch_size; i++)
  {
      void *outPtr = (void*)((uint8_t*)devBuf + i * m_NetworkSize.channels
          * m_NetworkSize.width * m_NetworkSize.height
          * bytesPerElement(tensorParam.params.data_type));

#if DEBUG_LIB
    static int batch_num1 = 0;
    std::ofstream outfile1("impl_in_batch_" + std::to_string(batch_num1) + ".bin");
    for (unsigned int j = 0 ; j < m_NetworkSize.height; j++) {
        outfile1.write((char*) batch->units[i].converted_frame_ptr + j*batch->pitch,
              3*m_NetworkSize.width);
    }
    outfile1.close();
    batch_num1 ++;
#endif
      if (convertFcn || convertFcnHalf) {
          /* Input needs to be pre-processed. */
          if (tensorParam.params.data_type == NvDsDataType_FP16) {
              convertFcnHalf((half*)outPtr, (unsigned char*)batch->units[i].converted_frame_ptr,
                  m_NetworkSize.width, m_NetworkSize.height, batch->pitch,
                  m_Scale, m_MeanDataBuffer.get() ? m_MeanDataBuffer->ptr<float>() : nullptr,
                  *m_PreProcessStream);
          } else {
              convertFcn((float*)outPtr, (unsigned char*)batch->units[i].converted_frame_ptr,
                  m_NetworkSize.width, m_NetworkSize.height, batch->pitch,
                  m_Scale, m_MeanDataBuffer.get() ? m_MeanDataBuffer->ptr<float>() : nullptr,
                  *m_PreProcessStream);
          }
      }

#ifdef DEBUG_LIB
    static int batch_num2 = 0;
    std::ofstream outfile2("impl_out_batch_" + std::to_string(batch_num2) + ".bin");
    outfile2.write((char*) outPtr, 4*m_NetworkSize.channels*m_NetworkSize.width*m_NetworkSize.height);
    outfile2.close();
    batch_num2 ++;
#endif

  }

  return NVDSPREPROCESS_SUCCESS;
}

extern "C"
NvDsPreProcessStatus
normalization_mean_subtraction_impl_initialize (CustomMeanSubandNormParams *custom_params,
    NvDsPreProcessTensorParams *tensor_params,
    std::unique_ptr<NvDsPreProcessTensorImpl> & m_Preprocessor, int unique_id)
{

  if (tensor_params->network_input_order == NvDsPreProcessNetworkInputOrder_kNCHW) {
      custom_params->networkSize.channels = tensor_params->network_input_shape[1];
      custom_params->networkSize.height = tensor_params->network_input_shape[2];
      custom_params->networkSize.width = tensor_params->network_input_shape[3];
  }  else if (tensor_params->network_input_order == NvDsPreProcessNetworkInputOrder_kNHWC) {
      custom_params->networkSize.height = tensor_params->network_input_shape[1];
      custom_params->networkSize.width = tensor_params->network_input_shape[2];
      custom_params->networkSize.channels = tensor_params->network_input_shape[3];
  } else {
      printf("network-input-order = %d not supported\n", tensor_params->network_input_order);
  }

#if 0
    printf ("IMPL : channels = %d width = %d height = %d\n", custom_params->networkSize.channels,
            custom_params->networkSize.width, custom_params->networkSize.height);
#endif

  switch (tensor_params->network_color_format)
  {
      case NvDsPreProcessFormat_RGB:
      case NvDsPreProcessFormat_BGR:
          if (custom_params->networkSize.channels != 3)
          {
              printf("RGB/BGR input format specified but network input channels is not 3\n");
              return NVDSPREPROCESS_CONFIG_FAILED;
          }
          break;
      case NvDsPreProcessFormat_GRAY:
          if (custom_params->networkSize.channels != 1)
          {
              printf("GRAY input format specified but network input channels is not 1.\n");
              return NVDSPREPROCESS_CONFIG_FAILED;
          }
          break;
      case NvDsPreProcessFormat_Tensor:
          break;
      default:
          printf("Unknown input format\n");
          return NVDSPREPROCESS_CONFIG_FAILED;
  }

  std::unique_ptr<NvDsPreProcessTensorImpl> tensor_impl =
      std::make_unique<NvDsPreProcessTensorImpl>(custom_params->networkSize,
              tensor_params->network_color_format, unique_id);
  assert(tensor_impl);

  if (custom_params->pixel_normalization_factor > 0.0f)
  {
      std::vector<float> offsets = custom_params->offsets;
      if (!tensor_impl->setScaleOffsets(
                  custom_params->pixel_normalization_factor, offsets)) {
          printf("Preprocessor set scale and offsets failed.\n");
          return NVDSPREPROCESS_CONFIG_FAILED;
      }
  }

  if (!custom_params->meanImageFilePath.empty() &&
          !tensor_impl->setMeanFile(custom_params->meanImageFilePath))
  {
      printf("Cannot access mean image file %s\n",
              custom_params->meanImageFilePath.c_str());
      return NVDSPREPROCESS_CONFIG_FAILED;
  }

  if (!tensor_impl->setInputOrder(tensor_params->network_input_order))
  {
      printf("Cannot set network order %s\n",
              (tensor_params->network_input_order == 0) ? "NCHW" : "NHWC");
      return NVDSPREPROCESS_CONFIG_FAILED;
  }

  NvDsPreProcessStatus status = tensor_impl->allocateResource();
  if (status != NVDSPREPROCESS_SUCCESS)
  {
      printf("preprocessor allocate resource failed\n");
      return status;
  }

  m_Preprocessor = std::move(tensor_impl);
  return NVDSPREPROCESS_SUCCESS;
}
