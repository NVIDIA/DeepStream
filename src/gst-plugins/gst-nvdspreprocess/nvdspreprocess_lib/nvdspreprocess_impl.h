/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSPREPROCESS_IMPL_H__
#define __NVDSPREPROCESS_IMPL_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include <cuda_runtime_api.h>

#include "nvdspreprocess_interface.h"

#include <stdio.h>
#include <assert.h>

/** Defines the maximum number of channels supported by the API
 for image input layers. */
#define _MAX_CHANNELS 4

/** APIs/macros for some frequently used functionality */
#define DISABLE_CLASS_COPY(NoCopyClass)      \
  NoCopyClass(const NoCopyClass &) = delete; \
  void operator=(const NoCopyClass &) = delete

/** helper move function */
#define SIMPLE_MOVE_COPY(Cls) \
  Cls &operator=(Cls &&o)     \
  {                           \
    move_copy(std::move(o));  \
    return *this;             \
  }                           \
  Cls(Cls &&o) { move_copy(std::move(o)); }

/** helper check safe string in C */
inline const char *safeStr(const std::string &str)
{
  return str.c_str();
}

/** helper check if file accessible in C */
inline bool file_accessible(const char* path)
{
    assert(path);
    return (access(path, F_OK) != -1);
}

/** helper check if file accessible in C++ */
inline bool file_accessible(const std::string &path)
{
  return (!path.empty()) && file_accessible(path.c_str());
}

/**
 * Custom parameters for normalization and mean subtractions
 */
typedef struct
{
  /** Holds the pathname of the labels file containing strings for the class
	 * labels. The labels file is optional. The file format is described in the
	 * custom models section of the DeepStream SDK documentation. */
  std::string labelsFilePath;

  /** Holds the pathname of the mean image file (PPM format). File resolution
	 must be equal to the network input resolution. */
  std::string meanImageFilePath;

  /** Holds the per-channel offsets for mean subtraction. This is
	 an alternative to the mean image file. The number of offsets in the array
			must be equal to the number of input channels. */
  std::vector<float> offsets;

  /** Holds the normalization factor with which to scale the input pixels. */
  float pixel_normalization_factor = 1.0;

  /** width, height, channels size of Network */
  NvDsPreProcessNetworkSize networkSize;
} CustomMeanSubandNormParams;

/**
 * Helper class for managing Cuda Streams.
 */
class CudaStream
{
public:
    explicit CudaStream(uint flag = cudaStreamDefault, int priority = 0);
    ~CudaStream();
    /** helper operator to return cuda stream */
    operator cudaStream_t() { return m_Stream; }
    /** pointer to cuda stream */
    cudaStream_t& ptr() { return m_Stream; }
    /** helper move copy functionality */
    SIMPLE_MOVE_COPY(CudaStream)

private:
    void move_copy(CudaStream&& o)
    {
        m_Stream = o.m_Stream;
        o.m_Stream = nullptr;
    }
    DISABLE_CLASS_COPY(CudaStream);

    cudaStream_t m_Stream = nullptr;
};

/**
 * Helper base class for managing Cuda allocated buffers.
 */
class CudaBuffer
{
public:
    virtual ~CudaBuffer() = default;
    /** size of cuda buffer in bytes */
    size_t bytes() const { return m_Size; }
    /** template to return cuda buffer */
    template <typename T>
    T* ptr()
    {
        return (T*)m_Buf;
    }
    /** pointer to cuda buffer */
    void* ptr() { return m_Buf; }
    /** helper move copy functionality */
    SIMPLE_MOVE_COPY(CudaBuffer)

protected:
    explicit CudaBuffer(size_t s) : m_Size(s) {}
    /** move_copy cuda buffer */
    void move_copy(CudaBuffer&& o)
    {
        m_Buf = o.m_Buf;
        o.m_Buf = nullptr;
        m_Size = o.m_Size;
        o.m_Size = 0;
    }
    /** disable class copy */
    DISABLE_CLASS_COPY(CudaBuffer);
    /** pointer to cuda buffer */
    void* m_Buf = nullptr;
    /** buffer size */
    size_t m_Size = 0;
};

/**
 * CUDA device buffers.
 */
class CudaDeviceBuffer : public CudaBuffer
{
public:
    /** constructor */
    explicit CudaDeviceBuffer(size_t size);
    /** destructor */
    ~CudaDeviceBuffer();
};

/**
 * Provides pre-processing functionality like mean subtraction and normalization.
 */
class NvDsPreProcessTensorImpl
{
public:
    /** constructor for tensor preparation implementation */
    NvDsPreProcessTensorImpl(const NvDsPreProcessNetworkSize& size, NvDsPreProcessFormat format,
        int id = 0);
    virtual ~NvDsPreProcessTensorImpl() = default;

    /** method to set offsets values */
    bool setScaleOffsets(float scale, const std::vector<float>& offsets = {});
    /** method to set mean file */
    bool setMeanFile(const std::string& file);
    /** method to set network input order */
    bool setInputOrder(const NvDsPreProcessNetworkInputOrder order);
    /** allocate resources for tensor preparation */
    NvDsPreProcessStatus allocateResource();
    /** synchronize cuda stream */
    NvDsPreProcessStatus syncStream();
    /** method to prepare tensor using cuda kernels */
    NvDsPreProcessStatus prepare_tensor(NvDsPreProcessBatch* batch,
        CustomTensorParams &tensorParam, void*& devBuf);

private:
    NvDsPreProcessStatus readMeanImageFile();
    DISABLE_CLASS_COPY(NvDsPreProcessTensorImpl);

private:
    int m_UniqueID = 0;

    /* Network input information. */
    NvDsPreProcessNetworkSize m_NetworkSize = {0};
    NvDsPreProcessFormat m_NetworkInputFormat = NvDsPreProcessFormat_RGB;
    NvDsPreProcessNetworkInputOrder m_InputOrder = NvDsPreProcessNetworkInputOrder_kNCHW;

    float m_Scale = 1.0f;
    std::vector<float> m_ChannelMeans; // same as channels
    std::string m_MeanFile;

    std::unique_ptr<CudaStream> m_PreProcessStream;
    std::unique_ptr<CudaDeviceBuffer> m_MeanDataBuffer;
};

/**
 * Initialize for pixel normalization and mean subtraction
 */
extern "C"
NvDsPreProcessStatus
normalization_mean_subtraction_impl_initialize (CustomMeanSubandNormParams *custom_params,
    NvDsPreProcessTensorParams *tensor_params,
    std::unique_ptr<NvDsPreProcessTensorImpl> & m_Preprocessor, int unique_id);

#endif
