/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>Service maker buffer abstraction </b>
 *
 * @b Description: Buffer class intends to provide a comprehensive wrapper
 * of various type of data that flows through the service maker pipeline.
 */

#ifndef DEEPSTREAM_BUFFER_HPP
#define DEEPSTREAM_BUFFER_HPP

#include <vector>
#include <functional>

#include "object.hpp"
#include "metadata.hpp"
#include "nvbufsurface.h"

typedef struct _GstBuffer OpaqueBuffer;

namespace deepstream {

class Tensor;

/**
 * @brief Base class of a buffer
 *
 * Buffer is the fundamental wrapper of data and metadata trunks that flows
 * through the pipeline.
 * Buffer is reference based and supports copying and moving, however both
 * copying and moving only affects the reference and never transfers the data.
 *
 */
class Buffer {
 public:
  /** Unified definition of latency for a buffer */
  typedef struct _Latency {
    /** Indicating the source from which the buffer is generated */
    unsigned int source_id;
    /** Indicating the frame to which the buffer belongs */
    unsigned int frame_num;
    /** Latency data */
    double latency;
  } Latency;

  /** Signature of customized function for freeing data in a buffer */
  using FreeFunction = void(*)(void*);

  /** @brief empty buffer constructor */
  Buffer();

  /**
   * @brief  New buffer constructor
   *
   * Create a new buffer with given size. User can choose pass a raw data pointer or
   * request the default allocation.
   *
   * @param[in] length   number of bytes used by the buffer
   * @param[in] data     if set null, default allocation will be used, otherwise, the
   *                     buffer will take and own the provided pointer.
   */
  Buffer(size_t length, void* data=nullptr, FreeFunction=nullptr);

  /**
   * @brief New Buffer constructor
   *
   * Create a new buffer from a byte vector, used by python mostly
   *
   * @param[in] bytes    byte vector
   *
  */
  Buffer(const std::vector<uint8_t>);

  /**
   * @brief  New buffer constructor for OpaqueBuffer
   *
   * Create a new buffer from a OpaqueBuffer, for professionals who understands how
   * Buffer Wrapper works
   *
   * @param[in] buffer   pointer to a OpaqueBuffer
   */
  Buffer(OpaqueBuffer* buffer);

  /** @brief Copy constructor */
  Buffer(const Buffer&);

  /** @brief Move constructor */
  Buffer(Buffer&&) noexcept;

  /** @brief Copy assignment */
  Buffer& operator=(const Buffer&);

  /** @brief Move assignment */
  Buffer& operator=(Buffer&&) noexcept;

  /** @brief Destructor */
  virtual ~Buffer();

  /** @brief If the buffer is null */
  operator bool() const;

  /** @brief Size of the buffer in bytes */
  size_t size() const;

  /** @brief Timestamp of the buffer */
  uint64_t timestamp() const;

  /** Chunk id of the buffer */
  int32_t chunkId(unsigned int batchId) const;

  /** @brief Return latency data of the buffer */
  std::vector<Latency> measureLatency() const;

  /**
   * @brief  Read data from the buffer
   *
   * @param[in] callable   callable provided by the caller to collect data
   *
   */
  virtual size_t read(std::function<size_t(const void* data, size_t len)> callable);

  /**
   * @brief  Write data to the buffer
   *
   * @param[in] callable   callable provided by the caller to inject data
   *
   */
  virtual size_t write(std::function<size_t(void* data, size_t len)> callable);

  /** Give up the ownership of this buffer and return the opaque buffer pointer */
  OpaqueBuffer* give();

  /**
   * @brief get the batch size of the buffer, 1 for un-batched buffer
   *
   */
  virtual size_t batchSize();

  /** @brief Get DeepStream batch metadata attached to this buffer */
  BatchMetadata getBatchMetadata();

  /**
   * @brief create a tensor object with the buffer data
   *
   */
  virtual Tensor* extract(unsigned int batchId);

  /**
   * @brief wrap the tensor to a new buffer
   */
  static void wrap(Tensor* );

 protected:
  /** opaque buffer pointer */
  OpaqueBuffer* buffer_;
  /** chunk ids, optional */
  std::vector<int> chunk_ids_ = { 0 };
};

class VideoBuffer : public Buffer {
 public:
  // NvSurface Buffer
  VideoBuffer(size_t width, size_t height, NvBufSurfaceColorFormat video_format,
              NvBufSurfaceMemType memtype, void* mem=nullptr, int gpu_id=0);

  // cast from a Buffer Object
  VideoBuffer(const Buffer&);

  virtual size_t read(std::function<size_t(const void* data, size_t len)>);

  virtual size_t write(std::function<size_t(void* data, size_t len)>);

  BatchMetadata getBatchMetadata();

  size_t width() const { return width_; }
  size_t height() const { return height_; }
  const NvBufSurfaceColorFormat format() const { return format_; }

  VideoBuffer clone() const;

 protected:
   size_t width_ = 0;
   size_t height_ = 0;
   NvBufSurfaceColorFormat format_;
};

typedef struct _NvDsSourceMeta
{
  int chunk_id;
} NvDsSourceMeta;

} // namespace deepstream

#endif
