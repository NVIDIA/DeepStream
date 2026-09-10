/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "buffer.hpp"
#include "tensor.hpp"

#include <gst/gst.h>
#include "nvds_latency_meta.h"
#include "gstnvdsmeta.h"


#include <cuda_runtime_api.h>
#include <cuda.h>

#include <EGL/egl.h>
#include "cudaEGL.h"
using namespace deepstream;

static GQuark dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

static std::vector<int> extract_chunk_ids_from_buffer(GstBuffer* buffer)
{
  NvDsSourceMeta *source_meta = NULL;

  /* first try getting the batch metadata */
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
  if (batch_meta) {
    std::vector<int> chunk_ids(batch_meta->max_frames_in_batch, 0);
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      if (frame_meta) {
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
          NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
          if (user_meta && user_meta->base_meta.meta_type == nvds_get_user_meta_type((gchar *)"NVIDIA.DECODER.GST_SOURCE")) {
            source_meta = (NvDsSourceMeta *)user_meta->user_meta_data;
            chunk_ids[frame_meta->batch_id] =source_meta->chunk_id;
          }
        }
      }
    }
    return chunk_ids;
  } else {
    /* unbatched buffer */
    gpointer state = NULL;
    GstMeta *gst_meta;
    int chunk_id = 0;

    while ((gst_meta = gst_buffer_iterate_meta(buffer, &state))) {
      if (!gst_meta_api_type_has_tag (gst_meta->info->api, dsmeta_quark)) {
        continue;
      }
      NvDsMeta *dsmeta = (NvDsMeta *) gst_meta;

      if (dsmeta->meta_type == nvds_get_user_meta_type((gchar *)"NVIDIA.DECODER.GST_SOURCE")) {
        if (source_meta != NULL) {
          GST_WARNING("Multiple NvDsSourceMeta found on buffer %p", buffer);
        }
        source_meta = (NvDsSourceMeta *) dsmeta->meta_data;
        chunk_id = source_meta->chunk_id;
      }
    }

    return {chunk_id};
  }
}

class GstBufferTensorContext : public Tensor::Context {
 public:
  GstBufferTensorContext(GstBuffer *buffer) : buffer_(buffer) {
//    gst_buffer_ref(buffer_);
  }
  virtual ~GstBufferTensorContext() {
//    gst_buffer_unref(buffer_);
  }
 protected:
  GstBuffer* buffer_;
};

class TegraGstBufferTensorContext : public Tensor::Context
{
public:
  TegraGstBufferTensorContext(GstBuffer *buffer, unsigned int batchId, CUgraphicsResource& cuGrphRes) : buffer_(buffer), batchId_(batchId), cuGrphRes_(cuGrphRes)
  {
  }
  virtual ~TegraGstBufferTensorContext()
  {
    GstMapInfo map = GST_MAP_INFO_INIT;
    gst_buffer_map(buffer_, &map, GST_MAP_READ);
    NvBufSurface *surface_batch = (NvBufSurface *)map.data;
    if (NvBufSurfaceUnMapEglImage(surface_batch, batchId_) != 0) {
      printf("Error: Failed to unmap EGL image\n");
    }
    cuCtxSynchronize();
    CUresult status = cuGraphicsUnregisterResource(cuGrphRes_);
    if (status != CUDA_SUCCESS) {
      printf("Error: Failed to unregister CUDA graphics resource \n");
    }
    gst_buffer_unmap(buffer_, &map);
  }

protected:
  GstBuffer *buffer_;
  unsigned int batchId_;
  CUgraphicsResource cuGrphRes_;
};

static void
external_memory_free (gpointer data) {
  free(data);
}

Buffer::Buffer() : buffer_(NULL) {}

Buffer::Buffer(size_t length, void* data, FreeFunction cb_free) {
  if (data == nullptr) {
    // newly allocated buffer with default allocator
    buffer_ = gst_buffer_new_allocate(NULL, length, NULL);
  } else {
    // external memory
    buffer_ = gst_buffer_new();
    GstMemory *mem = gst_memory_new_wrapped(
      (GstMemoryFlags)0, data, length, 0, length, data, cb_free?cb_free:external_memory_free
    );
    gst_buffer_append_memory(buffer_, mem);
  }
}

Buffer::Buffer(std::vector<uint8_t> bytes) {
  buffer_ = gst_buffer_new_allocate(NULL, bytes.size(), NULL);
  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map (buffer_, &map, GST_MAP_WRITE);
  uint8_t *data = (uint8_t *)map.data;
  for (size_t i = 0; i < bytes.size(); i++) {
    data[i] = bytes[i];
  }
  gst_buffer_unmap(buffer_, &map);
}

Buffer::Buffer(OpaqueBuffer* buffer) : buffer_(buffer) {
  if (buffer_) {
    gst_buffer_ref(buffer);
    chunk_ids_ = extract_chunk_ids_from_buffer((GstBuffer*)buffer);
  }
}

Buffer::Buffer(const Buffer& other) :
 buffer_(other.buffer_), chunk_ids_(other.chunk_ids_) {
  if (buffer_) {
    gst_buffer_ref(buffer_);
  }
}

Buffer::Buffer(Buffer&& other) noexcept :
buffer_(other.buffer_), chunk_ids_(std::move(other.chunk_ids_)) {
  other.buffer_ = NULL;
}

// copy assignment
Buffer& Buffer::operator=(const Buffer& other) {
  if (buffer_ != other.buffer_) {
    if (buffer_) {
      gst_buffer_unref(buffer_);
    }
    buffer_ = other.buffer_;
    chunk_ids_ = other.chunk_ids_;
    if (buffer_) {
      gst_buffer_ref(buffer_);
    }
  }
  return *this;
}

// move assignment
Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (buffer_ != other.buffer_) {
    if (buffer_) {
        gst_buffer_unref(buffer_);
    }
    buffer_ = other.buffer_;
    chunk_ids_ = std::move(other.chunk_ids_);
    other.buffer_ = NULL;
  }
  return *this;
}

Buffer::~Buffer() {
  if (buffer_) {
    gst_buffer_unref(buffer_);
  }
}

Buffer::operator bool() const {
  return buffer_ != NULL;
}

size_t Buffer::size() const {
  return buffer_ ? gst_buffer_get_size(buffer_) : 0;
}

uint64_t Buffer::timestamp() const {
  return buffer_ ? buffer_->pts : (uint64_t) -1;
}

int32_t Buffer::chunkId(unsigned int batchId) const {
  return chunk_ids_[batchId];
}

std::vector<Buffer::Latency> Buffer::measureLatency() const {
  std::vector<Buffer::Latency> latency_vector;
  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(buffer_));
  if (!batch_meta) {
    g_printerr("Unable to measure latency due to lack of the batch metadata");
    return latency_vector;
  }
  // auto batch_size = batch_meta->max_frames_in_batch;
  // FIXME: the number acquired from batch_meta seems to be not reliable, got 1
  // even if there're 4 sources attached to the streammux
  auto batch_size = 256;
  NvDsFrameLatencyInfo latency_info[batch_size];
  auto num_sources_in_batch =nvds_measure_buffer_latency(buffer_, latency_info);
  for (size_t i = 0; i < num_sources_in_batch; i++) {
    latency_vector.push_back(Buffer::Latency{
      latency_info[i].source_id, latency_info[i].frame_num, latency_info[i].latency
    });
  }

  return latency_vector;
}

size_t Buffer::read(std::function<size_t(const void* data, size_t len)> read_fn) {
  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map (buffer_, &map, GST_MAP_READ);
  size_t read = read_fn(map.data, map.size);
  gst_buffer_unmap (buffer_, &map);
  return read;
}

size_t Buffer::write(std::function<size_t(void* data, size_t len)> write_fn) {
  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map (buffer_, &map, GST_MAP_WRITE);
  map.size = write_fn(map.data, map.maxsize);
  gst_buffer_unmap (buffer_, &map);
  return map.size;
}

OpaqueBuffer* Buffer::give() {
  OpaqueBuffer* buffer = buffer_;
  buffer_ = NULL;
  return buffer;
}

size_t Buffer::batchSize() {
  size_t size = 0;
  if (!buffer_) {
    // empty buffer
    return size;
  }

  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map(buffer_, &map, GST_MAP_READ);
  if (map.size == sizeof(NvBufSurface)) {
    NvBufSurface* surface_batch = (NvBufSurface*) map.data;
    size = surface_batch->batchSize;
  } else {
    size = 1;
  }
  gst_buffer_unmap(buffer_, &map);

  return size;
}

BatchMetadata Buffer::getBatchMetadata() {
  if (buffer_ == nullptr) {
    return BatchMetadata();
  }
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer_);
  return BatchMetadata(batch_meta);
}

Tensor* Buffer::extract(unsigned int batchId) {
  Tensor* tensor = nullptr;
  if (buffer_) {
    GstMapInfo map = GST_MAP_INFO_INIT;
    gst_buffer_map(buffer_, &map, GST_MAP_READ);
    auto buf_size = map.size;
    NvBufSurface *surface_batch = (NvBufSurface*) map.data;
    NvBufSurfaceParams &surface = surface_batch->surfaceList[batchId];
    if (buf_size == sizeof(NvBufSurface)) {
      if (surface_batch->memType == NVBUF_MEM_CUDA_DEVICE ||
          surface_batch->memType == NVBUF_MEM_CUDA_PINNED ||
          surface_batch->memType == NVBUF_MEM_CUDA_UNIFIED ||
          surface_batch->memType == NVBUF_MEM_DEFAULT) {
        if (surface.colorFormat == NVBUF_COLOR_FORMAT_RGB) {
          int64_t shape[] = {surface.height, surface.width, 3};
          int64_t strides[] = {surface.pitch, 3, 1};
          auto* ctx = new GstBufferTensorContext(buffer_);
          tensor = new Tensor(3, Tensor::UNSIGNED, 8, shape, strides, surface.dataPtr, "HWC", surface_batch->gpuId, Tensor::DeviceType::GPU, ctx);
        } else {
          g_printerr("Only RGB format is supported for being extracted as a tensor\n");
        }
      }
      #if defined(__aarch64__)
      else if (surface_batch->memType == NVBUF_MEM_SURFACE_ARRAY || surface_batch->memType == NVBUF_MEM_HANDLE)
      {
        if (surface.colorFormat == NVBUF_COLOR_FORMAT_RGB)
        {
          int64_t shape[] = {surface.height, surface.width, 3};
          int64_t strides[] = {surface.pitch, 3, 1};
          CUeglFrame eglFrame;
          CUgraphicsResource cuGrphRes;
          CUresult cuStatus = CUDA_SUCCESS;
          if (NvBufSurfaceMapEglImage(surface_batch, batchId) != 0)
            g_printerr("EGL Image Map Failed\n");
          cuStatus = cuGraphicsEGLRegisterImage(&cuGrphRes, surface.mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
          if (cuStatus != CUDA_SUCCESS)
            g_printerr("EGL Reguster Image Failed\n");
          cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuGrphRes, batchId, 0);
          tensor = new Tensor(3, Tensor::UNSIGNED, 8, shape, strides, eglFrame.frame.pPitch[0], "HWC", surface_batch->gpuId, Tensor::DeviceType::GPU, new TegraGstBufferTensorContext(buffer_, batchId, cuGrphRes));
        }
        else
        {
          g_printerr("Only RGB format is supported for being extracted as a tensor\n");
        }
      }
      #endif
      else
      {
        g_printerr("Only Buffer Surface Memory types are supported for being extracted as a tensor");
      }
    }
    else
    {
      g_printerr("We only support extracting buffer surface as a tensor\n");
    }
  gst_buffer_unmap(buffer_, &map);
  }

  return tensor;
}

void Buffer::wrap(Tensor* )
{}

static void
nv_buffer_surface_destroy (gpointer data) {
  NvBufSurface *nvbufsurface = (NvBufSurface *) data;
  NvBufSurfaceDestroy(nvbufsurface);
}

static void
nv_buffer_surface_free (gpointer data) {
  NvBufSurface *nvbufsurface = (NvBufSurface *) data;
  switch (nvbufsurface->memType) {
    case NVBUF_MEM_CUDA_PINNED:
      cudaFreeHost(nvbufsurface->surfaceList[0].dataPtr);
      break;
    case NVBUF_MEM_DEFAULT:
    case NVBUF_MEM_CUDA_DEVICE:
    case NVBUF_MEM_CUDA_UNIFIED:
      cudaFree(nvbufsurface->surfaceList[0].dataPtr);
      break;
    default:
      g_printerr("nvbufsurface: invalid memory type (%d)\n", nvbufsurface->memType);
      break;
  }
  free(nvbufsurface->surfaceList);
  free(nvbufsurface);
}

VideoBuffer::VideoBuffer(size_t width, size_t height, NvBufSurfaceColorFormat video_format,
            NvBufSurfaceMemType memtype, void* mem, int gpu_id)
: width_(width), height_(height), format_(video_format) {
  if (
    (video_format != NVBUF_COLOR_FORMAT_RGBA) &&
    (video_format != NVBUF_COLOR_FORMAT_RGB) &&
    (video_format != NVBUF_COLOR_FORMAT_YUV420) &&
    (video_format != NVBUF_COLOR_FORMAT_NV12)
  ) {
    std::string message = "Unsupported color format: ";
    throw std::runtime_error(message + std::to_string((int)video_format));
  }

  if (mem == nullptr) {
    // new NvBufSurface will be created and the memory be allocated accordingly
    NvBufSurface *nvbufsurface = nullptr;
    NvBufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(NvBufSurfaceCreateParams));
    create_params.gpuId = gpu_id;
    create_params.width = width;
    create_params.height = height;
    create_params.size = 0;
    create_params.colorFormat = video_format;
    create_params.isContiguous = 0;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.memType = memtype;
    if (NvBufSurfaceCreate(&nvbufsurface, 1, &create_params) != 0) {
      throw std::runtime_error("Failed to create NV buffer surface");
    }
    buffer_ = gst_buffer_new_wrapped_full(
      (GstMemoryFlags)0, nvbufsurface, sizeof(NvBufSurface), 0, sizeof(NvBufSurface),
      nvbufsurface, nv_buffer_surface_destroy
    );
  } else {
    // construct the NvBufSurface from the given memory
    NvBufSurface *nvbufsurface = (NvBufSurface *) calloc(1, sizeof(NvBufSurface));
    nvbufsurface->numFilled = 1;
    nvbufsurface->memType = memtype;
    nvbufsurface->batchSize = 1;
    nvbufsurface->gpuId = gpu_id;
    nvbufsurface->isContiguous = TRUE;
    NvBufSurfaceParams* surf_params = (NvBufSurfaceParams*) calloc(1, sizeof(NvBufSurfaceParams));
    surf_params->layout = NVBUF_LAYOUT_PITCH;
    surf_params->dataPtr = mem;
    surf_params->colorFormat = video_format;
    if (surf_params->colorFormat == NVBUF_COLOR_FORMAT_RGB) {
      surf_params->width = width;
      surf_params->height = height;
      surf_params->pitch = width * 3;
      surf_params->dataSize = width * height * 3;
      surf_params->planeParams.num_planes = 1;
      surf_params->planeParams.offset[0] = 0;
      surf_params->planeParams.width[0] = width;
      surf_params->planeParams.height[0] = height;
      surf_params->planeParams.pitch[0] = width * 3;
      surf_params->planeParams.bytesPerPix[0] = 3;
      surf_params->planeParams.psize[0] = width * height * 3;
    } else if (surf_params->colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
      surf_params->width = width;
      surf_params->height = height;
      surf_params->pitch = width * 4;
      surf_params->dataSize = width * height * 4;
      surf_params->planeParams.num_planes = 1;
      surf_params->planeParams.offset[0] = 0;
      surf_params->planeParams.width[0] = width;
      surf_params->planeParams.height[0] = height;
      surf_params->planeParams.pitch[0] = width * 4;
      surf_params->planeParams.bytesPerPix[0] = 4;
      surf_params->planeParams.psize[0] = width * height * 4;
    } else if (surf_params->colorFormat == NVBUF_COLOR_FORMAT_NV12) {
      surf_params->colorFormat = NVBUF_COLOR_FORMAT_NV12;
      surf_params->width = width;
      surf_params->height = height;
      surf_params->pitch = width;
      surf_params->dataSize = width * height * 1.5;
      surf_params->planeParams.num_planes = 2;
      surf_params->planeParams.offset[0] = 0;
      surf_params->planeParams.width[0] = width;
      surf_params->planeParams.height[0] = height;
      surf_params->planeParams.pitch[0] = width;
      surf_params->planeParams.bytesPerPix[0] = 1;
      surf_params->planeParams.psize[0] = width * height;
      surf_params->planeParams.offset[1] = surf_params->planeParams.psize[0];
      surf_params->planeParams.width[1] = width/2;
      surf_params->planeParams.height[1] = height/2;
      surf_params->planeParams.pitch[1] = width;
      surf_params->planeParams.bytesPerPix[1] = 1;
      surf_params->planeParams.psize[1] = width * height / 2;
    } else if (surf_params->colorFormat == NVBUF_COLOR_FORMAT_YUV420) {
      surf_params->width = width;
      surf_params->height = height;
      surf_params->pitch = width;
      surf_params->dataSize = width * height * 1.5;
      surf_params->planeParams.num_planes = 3;
      surf_params->planeParams.offset[0] = 0;
      surf_params->planeParams.width[0] = width;
      surf_params->planeParams.height[0] = height;
      surf_params->planeParams.pitch[0] = width;
      surf_params->planeParams.bytesPerPix[0] = 1;
      surf_params->planeParams.psize[0] = width * height;
      for (auto i = 1; i <= 2; i++) {
        surf_params->planeParams.offset[i] = surf_params->planeParams.offset[i-1] + surf_params->planeParams.psize[i-1];
        surf_params->planeParams.width[i] = width/2;
        surf_params->planeParams.height[i] = height/2;
        surf_params->planeParams.pitch[i] = width/2;
        surf_params->planeParams.bytesPerPix[i] = 1;
        surf_params->planeParams.psize[i] = width * height / 4;
      }
    } else {
      throw std::runtime_error("Unsupported video format");
    }
    nvbufsurface->surfaceList = surf_params;
    buffer_ = gst_buffer_new_wrapped_full(
      (GstMemoryFlags)0, nvbufsurface, sizeof(NvBufSurface), 0, sizeof(NvBufSurface),
      nvbufsurface, nv_buffer_surface_free
    );
  }
}

VideoBuffer::VideoBuffer(const Buffer& other) : Buffer(other) {
  if (buffer_) {
    // try checking if the buffer is a NvBufSurface wrapper at all
    GstMapInfo map = GST_MAP_INFO_INIT;
    gst_buffer_map(buffer_, &map, GST_MAP_READ);
    auto buf_size = map.size;
    NvBufSurface *nvbufsurface = (NvBufSurface*) map.data;
    if (buf_size != sizeof(NvBufSurface)) {
      g_printerr("VideoBuffer can only be created from an NvBufSurface instance!\n");
      gst_buffer_unmap(buffer_, &map);
      gst_buffer_unref(buffer_);
      buffer_ = nullptr;
    } else {
      width_ = nvbufsurface->surfaceList[0].width;
      height_ = nvbufsurface->surfaceList[0].height;
      format_ = nvbufsurface->surfaceList[0].colorFormat;
      gst_buffer_unmap(buffer_, &map);
    }
  }
}

size_t VideoBuffer::read(std::function<size_t(const void* data, size_t len)> read_fn) {
  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map (buffer_, &map, GST_MAP_READ);
  NvBufSurface *nvbufsurface = (NvBufSurface*) map.data;
  NvBufSurfaceParams* surf_params = nvbufsurface->surfaceList;
  size_t read = read_fn(surf_params->dataPtr, surf_params->dataSize);
  gst_buffer_unmap (buffer_, &map);
  return read;
}

size_t VideoBuffer::write(std::function<size_t(void* data, size_t len)> write_fn) {
  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map (buffer_, &map, GST_MAP_WRITE);
  NvBufSurface *nvbufsurface = (NvBufSurface*) map.data;
  NvBufSurfaceParams* surf_params = nvbufsurface->surfaceList;
  size_t written = write_fn(surf_params->dataPtr, surf_params->dataSize);
  if (written) {
    nvbufsurface->numFilled = 1;
  }
  gst_buffer_unmap(buffer_, &map);
  return written;
}

BatchMetadata VideoBuffer::getBatchMetadata() {
  return Buffer::getBatchMetadata();
}

VideoBuffer VideoBuffer::clone() const {
  if (buffer_ == nullptr) {
    return Buffer();
  }

  NvBufSurface *surface_dst = NULL;
  GstMapInfo map = GST_MAP_INFO_INIT;
  gst_buffer_map(buffer_, &map, GST_MAP_READ);
  NvBufSurface *surface_src = (NvBufSurface*) map.data;
  // create a new surface holder
  NvBufSurfaceCreateParams params;
  memset (&params, 0, sizeof(NvBufSurfaceCreateParams));
  params.width  = surface_src->surfaceList[0].width;
  params.height = surface_src->surfaceList[0].height;
  params.gpuId  = surface_src->gpuId;
  params.memType = surface_src->memType;
  params.colorFormat = surface_src->surfaceList[0].colorFormat;
  if (NvBufSurfaceCreate(&surface_dst, surface_src->batchSize, &params) < 0 || NvBufSurfaceCopy(surface_src, surface_dst) < 0) {
    g_printerr("Failed to clone buffer surface");
    gst_buffer_unmap(buffer_, &map);
    return Buffer();
  }

  GstBuffer * buffer = gst_buffer_new_wrapped_full(
      (GstMemoryFlags)0, surface_dst, sizeof(NvBufSurface), 0, sizeof(NvBufSurface),
      surface_dst, nv_buffer_surface_destroy
  );
  return Buffer(buffer);
}