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
 * <b>Service maker tensor abstraction </b>
 *
 * @b Description: Tensor class provides an interface for data exchange
 * across frameworks
 */

#ifndef DEEPSTREAM_TENSOR_HPP
#define DEEPSTREAM_TENSOR_HPP

#include <vector>

#include "buffer.hpp"

namespace deepstream {

typedef std::vector<uint64_t> TensorShape;
class TensorImplementation;

/**
 * @class Tensor representation
 *
 * The Tensor class abstracts the tensor data access
*/
class Tensor {
 public:
  /**
   * @brief Code for tensor data types
  */
  typedef enum {
    INVALID = -1,
    UNSIGNED,
    SIGNED,
    FLOAT,
    COMPLEX
  } DataType;

  typedef enum {
    NONE = -1,
    CPU,
    GPU
  } DeviceType;

  /**
   * @brief Context object of the tensor, which associates the
   *        tensor with what creates it.
   *        The base class represents the default context who makes
   *        the tensor an observer of its memory without having
   *        any control on it.
  */
  class Context {
    public:
      virtual ~Context() {}
  };
  /**
   * @brief Construct from common tensor configurations
  */
  Tensor(unsigned int rank,
         DataType dtype,
         unsigned int bits,
         const int64_t shape[],
         const int64_t strides[],
         void* data,
         std::string format,
         unsigned int device_id,
         DeviceType device,
         Context* context);

  /** Copy of a tensor is not allowed, use clone instead */
  Tensor(const Tensor&) = delete;

  /**
   * @brief Destructor
  */
  virtual ~Tensor();

  /**
   * @brief Number of dimenstions of the tensor
   */
  unsigned int rank() const;

  /**
   * @brief Shape of the tensor in vector
  */
  TensorShape shape() const;

  /**
   * @brief Data type of the tensor elements
  */
  DataType dtype() const;

  /**
   * @brief Number of bits for each data
  */
  unsigned int bits() const;

  /**
   * @brief Retrieve the stride of a specific dimension
  */
  uint64_t stride(unsigned int d) const;

  /**
   * @brief Retrieve the allocated data pointer
  */
  void* data() const;

  /**
   * @brief Retrieve the device id
   */
  unsigned int deviceId() const;

  /**
   * @brief Retrieve the device type
   */
  DeviceType deviceType() const;

  /**
   * @brief Wrap the tensor to a buffer
  */
  Buffer wrap(NvBufSurfaceColorFormat format);

  /**
   * @brief Clone a tensor
   */
  Tensor* clone() const;

  /**
   * @brief Copy the tensor to GPU
   */
  Tensor* toGPU(unsigned int device_id) const;

  /**
   * @brief total size of the tensor in bytes
   */
  uint64_t size() const;

 protected:
  TensorImplementation* impl_;
  std::string format_;
  std::unique_ptr<Context> context_;
};

}

#endif