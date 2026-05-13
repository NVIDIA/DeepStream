/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "data_feeder.hpp"

namespace deepstream {

class FileDataSource : public DataFeeder::IDataProvider {
 public:
  virtual ~FileDataSource() {
    if (fp_) {
      fclose(fp_);
    }
    if (buffer_) {
      free(buffer_);
    }
  }

  virtual Buffer read(DataFeeder& feeder, unsigned int size, bool& eos) {
    if (fp_ == nullptr) {
      std::string location;
      feeder.getProperty("location", location);
      fp_ = fopen(location.c_str(), "r");
      if (fp_ == nullptr) {
        fprintf(stderr, "Failed to open file %s\n", location.c_str());
        eos = true;
        return Buffer();
      }
    }
    do {
      int gpu_id = 0;
      bool use_gpu_memory = false;
      bool use_external_memory = true;
      int width = 0;
      int height = 0;
      std::string format;
      NvBufSurfaceColorFormat color_format = NVBUF_COLOR_FORMAT_INVALID;
      feeder.getProperty(
        "use-gpu-memory", use_gpu_memory, "use-external-memory", use_external_memory,
        "frame-width", width, "frame-height", height, "format", format,
        "gpu-id", gpu_id);
      unsigned int frame_size = 0;
      if (format == "RGBA") {
        color_format = NVBUF_COLOR_FORMAT_RGBA;
        frame_size = width*height*4;
      } else if (format == "I420" || format == "NV12") {
        color_format = NVBUF_COLOR_FORMAT_YUV420;
        frame_size = width*height*1.5;
      } else {
        color_format = NVBUF_COLOR_FORMAT_NV12;
        frame_size = size;
        use_gpu_memory = false;
      }
      if (frame_size <= 0) {
        fprintf(stderr, "Invalid frame size %u\n", frame_size);
        break;
      } else if (frame_size > size) {
        // does the hint of data size matter?
        fprintf(stderr, "Frame size is larger than asked %u vs %u\n", frame_size, size);
      }

      if (use_gpu_memory) {
        // read data to the buffer and then copy it to the GPU memory
        if (buffer_ == nullptr) {
          buffer_ = malloc(frame_size);
        }
        if (0 == fread (buffer_, 1, frame_size, fp_)) break;
        if (use_external_memory) {
          void *cuda_device_data;
          if (cudaMalloc ((void **) &cuda_device_data, frame_size) != cudaSuccess) {
            fprintf(stderr, "ERROR !! Unable to allocate device memory. \n");
            break;
          }
          if (cudaMemcpy (cuda_device_data, buffer_, frame_size,
                  cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Unable to copy between device and host memories. \n");
            break;
          }
          return VideoBuffer(width, height, color_format, NVBUF_MEM_CUDA_DEVICE, cuda_device_data, gpu_id);
        } else {
          VideoBuffer buffer(width, height, color_format, NVBUF_MEM_CUDA_DEVICE);
          if (!buffer.write(
            [&](void* data, size_t len) {
              if (len != frame_size) {
                return (size_t)0;
              }
              cudaMemcpy(data, buffer_, len, cudaMemcpyHostToDevice);
              return len;
            }
          )) break;
          return buffer;
        }
      } else {
        if (use_external_memory) {
          void* data = malloc(frame_size);
          if (0 == fread (data, 1, frame_size, fp_)) break;
          return Buffer(frame_size, data);
        } else {
          Buffer buffer(frame_size);
          if (!buffer.write(
            [&](void* data, size_t len){
              return fread (data, 1, frame_size, fp_);
            }
          )) break;
          return buffer;
        }
      }
    } while (false);
  // empty buffer returned on errors.
  eos = true;
  return Buffer();
  }
 private:
  FILE *fp_ = nullptr;
  void* buffer_ = nullptr;
};
}