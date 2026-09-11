/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensor.hpp"
#include <gst/gst.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/** Maximum number of dimensions of a tensor */
#define TENSOR_MAX_RANK (15)

namespace deepstream {

class CommonTensorContext : public Tensor::Context {
 public:
  CommonTensorContext(void* ptr): data_ptr_(ptr) {}
  virtual ~CommonTensorContext() {
    if (data_ptr_) {
      free(data_ptr_);
    }
  }

  void relinquish() {
    data_ptr_ = nullptr;
  }
 protected:
  void *data_ptr_;
};

class CudaTensorContext : public Tensor::Context {
 public:
  CudaTensorContext(void* ptr): data_ptr_(ptr) {}
  virtual ~CudaTensorContext() {
    if (data_ptr_) {
      cudaFree(data_ptr_);
    }
  }

  void relinquish() {
    data_ptr_ = nullptr;
  }
 protected:
  void *data_ptr_;
};

class TensorImplementation {
 public:
  unsigned int rank = 0;
  Tensor::DataType dtype = Tensor::INVALID;
  unsigned int bits = 0;
  void* data = nullptr;
  unsigned int device_id = 0;
  Tensor::DeviceType device = Tensor::DeviceType::GPU;
  uint64_t shape[TENSOR_MAX_RANK] = {0};
  uint64_t strides[TENSOR_MAX_RANK] = {0};
};

Tensor::Tensor(unsigned int rank,
         DataType dtype,
         unsigned int bits,
         const int64_t shape[],
         const int64_t strides[],
         void* data,
         std::string format,
         unsigned int device_id,
         DeviceType device,
         Context* context)
: context_(context) {
  if (rank > TENSOR_MAX_RANK) {
    throw std::runtime_error("Tensor rank is out of range!");
  }
  impl_ = new TensorImplementation{rank, dtype, bits, data, device_id, device};
  for (size_t n = 0; n < rank; n++) {
    impl_->shape[n] = (uint64_t )shape[n];
    impl_->strides[n] = strides? (uint64_t) strides[n] : 0;
  }
  format_ = format;
}

Tensor::~Tensor() {
  if (impl_) {
    // the context object will handle the memory recycling
    delete impl_;
  }
}

unsigned int Tensor::rank() const {
  return impl_->rank;
}

TensorShape Tensor::shape() const {
  std::vector<uint64_t> shape;
  for (uint i = 0; i < impl_->rank; i++)
    shape.push_back(impl_->shape[i]);
  return TensorShape(shape);
}

Tensor::DataType Tensor::dtype() const {
  return impl_->dtype;
}

unsigned int Tensor::bits() const {
  return impl_->bits;
}

uint64_t Tensor::stride(unsigned int d) const {
  if (d >= impl_->rank) {
    throw std::runtime_error("Dimension beyond range");
  }
  return impl_->strides[d];
}

void* Tensor::data() const {
  return impl_->data;
}

unsigned int Tensor::deviceId() const {
  return impl_->device_id;
}

Tensor::DeviceType Tensor::deviceType() const {
  return impl_->device;
}

Buffer Tensor::wrap(NvBufSurfaceColorFormat format) {
  if (impl_->device == Tensor::DeviceType::CPU) {
    if (impl_->bits == 8 && impl_->rank == 1) {
      /* wrap as a bytes buffer */
      size_t length = impl_->shape[0];
      void* ptr = malloc(length);
      memcpy((void*)ptr, impl_->data, length);
      return Buffer(length, ptr);
    } else {
      g_printerr("Tensor format not supported: bits: %d, rank: %d\n", impl_->bits, impl_->rank);
      throw std::runtime_error("Failure in wrapping a tensor as a buffer");
    }
  } else {
    /* wrap a GPU tensor as a video buffer*/
    if (format_ != "HWC" ||  impl_->bits != 8 || impl_->rank != 3) {
      throw std::runtime_error("Tensor doesn't look like an image");
    }
    if (format != NVBUF_COLOR_FORMAT_RGBA && format != NVBUF_COLOR_FORMAT_RGB) {
      throw std::runtime_error("Color format not supported");
    }
    if (
      (format == NVBUF_COLOR_FORMAT_RGBA && impl_->shape[2] != 4) ||
      (format == NVBUF_COLOR_FORMAT_RGB && impl_->shape[2] != 3)
    ) {
      throw std::runtime_error("Color format not compatible with tensor layout");
    }

    CudaTensorContext* cuda_ctx = dynamic_cast<CudaTensorContext*>(context_.get());
    if (cuda_ctx) {
      // give the cuda memory ownership to the buffer if this tensor is under CudaTensorContext
      cuda_ctx->relinquish();
      context_.reset(new Tensor::Context());
      return VideoBuffer(impl_->shape[1], impl_->shape[0], format, NVBUF_MEM_CUDA_DEVICE, impl_->data, (int)this->impl_->device_id);
    } else {
      // probably it is from a dl managed tensor, possibility of avoiding copy?
      return this->clone()->wrap(format);
    }
  }
}

Tensor* Tensor::clone() const {
  if (this->size() == 0) {
    return nullptr;
  }

  void *data_ptr = nullptr;
  unsigned int ndim = this->impl_->rank;
  int64_t shape[ndim];
  int64_t strides[ndim];
  char err_msg[512];

  for (unsigned i = 0; i < ndim; i++) {
    shape[i] = this->impl_->shape[i];
  }
  strides[ndim-1] = 1;
  for (int i = ndim-2; i >=0; --i ) {
     strides[i] = strides[i + 1] * shape[i + 1];
  }

  for (int i = ndim-1; i > 0; --i) {
    if (strides[i] != (int64_t) this->impl_->strides[i]) {
      throw std::runtime_error("Misaligned strides for cloning the tensor");
    }
  }
  Tensor* tensor = nullptr;
  if (this->impl_->device == Tensor::DeviceType::GPU) {
    cudaSetDevice((int)this->impl_->device_id);
    cudaError_t err = cudaMalloc(&data_ptr, this->size());
    if (err != cudaSuccess) {
      snprintf(err_msg, sizeof(err_msg), "CUDA malloc failed: %s", cudaGetErrorString(err));
      throw std::runtime_error(err_msg);
    }
    if (strides[0] == (int64_t) this->impl_->strides[0]) {
      err = cudaMemcpy(data_ptr, this->impl_->data, this->size(), cudaMemcpyDeviceToDevice);
    } else {
      err = cudaMemcpy2D(data_ptr, strides[0], this->impl_->data, this->impl_->strides[0], strides[0], shape[0], cudaMemcpyDeviceToDevice);
    }
    if (err == cudaSuccess) {
      CudaTensorContext* context = new CudaTensorContext(data_ptr);
      tensor = new Tensor(
        ndim,
        this->impl_->dtype,
        this->impl_->bits,
        shape,
        strides,
        data_ptr,
        this->format_,
        this->impl_->device_id,
        this->impl_->device,
        context);
    } else {
      g_printerr("CUDA memcpy failed\n");
      cudaFree(data_ptr);
    }
  } else if (this->impl_->device == Tensor::DeviceType::CPU) {
    data_ptr = malloc(this->size());
    if (data_ptr == nullptr) {
      throw std::runtime_error("malloc failed");
    }
    memcpy(data_ptr, this->impl_->data, this->size());
    CommonTensorContext* context = new CommonTensorContext(data_ptr);
    tensor = new Tensor(
        ndim,
        this->impl_->dtype,
        this->impl_->bits,
        shape,
        strides,
        data_ptr,
        this->format_,
        this->impl_->device_id,
        this->impl_->device,
        context);
  }
  return tensor;
}

Tensor* Tensor::toGPU(unsigned int device_id) const {
  if (this->impl_->device == Tensor::DeviceType::GPU && this->impl_->device_id == device_id) {
    // already on the target device
    g_warning("Tensor already on GPU %d, cloning it\n", device_id);
    return this->clone();
  }

  void *data_ptr = nullptr;
  unsigned int ndim = this->impl_->rank;
  int64_t shape[ndim];
  int64_t strides[ndim];
  for (unsigned i = 0; i < ndim; i++) {
    shape[i] = this->impl_->shape[i];
    strides[i] = this->impl_->strides[i];
  }

  if (this->impl_->device == Tensor::DeviceType::CPU) {
    // copy to GPU
    cudaSetDevice((int)device_id);
    cudaError_t err = cudaMalloc(&data_ptr, this->size());
    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemcpy(data_ptr, this->impl_->data, this->size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA memcpy failed");
    }
    CudaTensorContext* context = new CudaTensorContext(data_ptr);
    return new Tensor(this->impl_->rank, this->impl_->dtype, this->impl_->bits, shape, strides, data_ptr, this->format_, device_id, Tensor::DeviceType::GPU, context);
  } else if (device_id != this->impl_->device_id) {
    // copy to a different GPU
    cudaSetDevice((int)device_id);
    cudaError_t err = cudaMalloc(&data_ptr, this->size());
    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemcpy(data_ptr, this->impl_->data, this->size(), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA memcpy failed");
    }
    CudaTensorContext* context = new CudaTensorContext(data_ptr);
    return new Tensor(this->impl_->rank, this->impl_->dtype, this->impl_->bits, shape, strides, data_ptr, this->format_, device_id, Tensor::DeviceType::GPU, context);
  } else {
    g_printerr("Unsupported device type: %d\n", this->impl_->device);
    return nullptr;
  }
}

uint64_t Tensor::size() const {
  auto element_size = this->impl_->bits/8;
  auto element_num = 0;

  for (unsigned i = 0; i < this->impl_->rank; i++) {
    if (element_num == 0) {
      element_num = this->impl_->shape[i];
    } else {
      element_num *= this->impl_->shape[i];
    }
  }

  return element_num * element_size;
}

}
