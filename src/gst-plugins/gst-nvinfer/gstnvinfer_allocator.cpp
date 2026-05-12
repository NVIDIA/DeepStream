/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "cuda_runtime.h"
#include "gstnvinfer_allocator.h"

/* Standard GStreamer boiler plate macros */
#define GST_TYPE_NVINFER_ALLOCATOR \
    (gst_nvinfer_allocator_get_type ())
#define GST_NVINFER_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVINFER_ALLOCATOR,GstNvInferAllocator))
#define GST_NVINFER_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVINFER_ALLOCATOR,GstNvInferAllocatorClass))
#define GST_IS_NVINFER_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVINFER_ALLOCATOR))
#define GST_IS_NVINFER_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVINFER_ALLOCATOR))

typedef struct _GstNvInferAllocator GstNvInferAllocator;
typedef struct _GstNvInferAllocatorClass GstNvInferAllocatorClass;

G_GNUC_INTERNAL GType gst_nvinfer_allocator_get_type (void);

GST_DEBUG_CATEGORY_STATIC (gst_nvinfer_allocator_debug);
#define GST_CAT_DEFAULT gst_nvinfer_allocator_debug

/**
 * Extends the GstAllocator class. Holds the parameters for allocator.
 */
struct _GstNvInferAllocator
{
  GstAllocator allocator;
  guint batch_size;
  guint width;
  guint height;
  NvBufSurfaceColorFormat color_format;
  guint gpu_id;
};

struct _GstNvInferAllocatorClass
{
  GstAllocatorClass parent_class;
};

/* Standard boiler plate to create a debug category and initializing the
 * allocator type.
 */
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvinfer_allocator_debug, "nvinferallocator", 0, "nvinfer allocator");
#define gst_nvinfer_allocator_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvInferAllocator, gst_nvinfer_allocator,
    GST_TYPE_ALLOCATOR, _do_init);

/** Type of the memory allocated by the allocator. This can be used to identify
 * buffers / memories allocated by this allocator. */
#define GST_NVINFER_MEMORY_TYPE "nvinfer"

/** Structure allocated internally by the allocator. */
typedef struct
{
  /** Should be the first member of a structure extending GstMemory. */
  GstMemory mem;
  GstNvInferMemory mem_infer;
} GstNvInferMem;

/* Function called by GStreamer buffer pool to allocate memory using this
 * allocator. */
static GstMemory *
gst_nvinfer_allocator_alloc (GstAllocator * allocator, gsize size,
    GstAllocationParams * params)
{
  GstNvInferAllocator *inferallocator = GST_NVINFER_ALLOCATOR (allocator);
  GstNvInferMem *nvmem = new GstNvInferMem;
  GstNvInferMemory *tmem = &nvmem->mem_infer;

  int is_nvgpu = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      is_nvgpu = 1;
    }
  }

  NvBufSurfaceCreateParams create_params = { 0 };

  create_params.gpuId = inferallocator->gpu_id;
  create_params.width = inferallocator->width;
  create_params.height = inferallocator->height;
  create_params.size = 0;
  create_params.isContiguous = 1;
  create_params.colorFormat = inferallocator->color_format;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = NVBUF_MEM_CUDA_DEVICE;
  if(is_nvgpu)
  {
    create_params.memType = NVBUF_MEM_SURFACE_ARRAY;
  }

  if (NvBufSurfaceCreate (&tmem->surf, inferallocator->batch_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer pool for nvinfer");
    if (nvmem) {
      delete nvmem;
      nvmem = nullptr;
    }
    return nullptr;
  }

  if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    if (NvBufSurfaceMapEglImage (tmem->surf, -1) != 0) {
      GST_ERROR ("Error: Could not map EglImage from NvBufSurface for nvinfer");
      if (nvmem) {
        delete nvmem;
	nvmem = nullptr;
      }
      return nullptr;
    }

    tmem->egl_frames.resize (inferallocator->batch_size);
    tmem->cuda_resources.resize (inferallocator->batch_size);
  }

  tmem->frame_memory_ptrs.assign (inferallocator->batch_size, nullptr);

  for (guint i = 0; i < inferallocator->batch_size; i++) {
#if defined(__aarch64__)
    if (tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
      if (cuGraphicsEGLRegisterImage (&tmem->cuda_resources[i],
              tmem->surf->surfaceList[i].mappedAddr.eglImage,
              CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        g_printerr ("Failed to register EGLImage in cuda\n");
	if (nvmem) {
	  delete nvmem;
	  nvmem = nullptr;
	}
        return nullptr;
      }

      if (cuGraphicsResourceGetMappedEglFrame (&tmem->egl_frames[i],
              tmem->cuda_resources[i], 0, 0) != CUDA_SUCCESS) {
        g_printerr ("Failed to get mapped EGL Frame\n");
	if (nvmem) {
	  delete nvmem;
	  nvmem = nullptr;
	}
        return nullptr;
      }
      tmem->frame_memory_ptrs[i] = (char *) tmem->egl_frames[i].frame.pPitch[0];
    }
    else
#endif
    {
      /* Calculate pointers to individual frame memories in the batch memory and
      * insert in the vector. */
      tmem->frame_memory_ptrs[i] = (char *) tmem->surf->surfaceList[i].dataPtr;
    }
  }

  /* Initialize the GStreamer memory structure. */
  gst_memory_init ((GstMemory *) nvmem, (GstMemoryFlags) 0, allocator, nullptr,
      size, params->align, 0, size);

  return (GstMemory *) nvmem;
}

/* Function called by buffer pool for freeing memory using this allocator. */
static void
gst_nvinfer_allocator_free (GstAllocator * allocator, GstMemory * memory)
{
  GstNvInferMem *nvmem = (GstNvInferMem *) memory;
  GstNvInferMemory *tmem = &nvmem->mem_infer;
  if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    GstNvInferAllocator *inferallocator = GST_NVINFER_ALLOCATOR (allocator);

    for (size_t i = 0; i < inferallocator->batch_size; i++) {
      cuGraphicsUnregisterResource (tmem->cuda_resources[i]);
    }
  }

  NvBufSurfaceUnMapEglImage (tmem->surf, -1);
  NvBufSurfaceDestroy (tmem->surf);

  delete nvmem;
}

/* Function called when mapping memory allocated by this allocator. Should
 * return pointer to GstNvInferMemory. */
static gpointer
gst_nvinfer_memory_map (GstMemory * mem, gsize maxsize, GstMapFlags flags)
{
  GstNvInferMem *nvmem = (GstNvInferMem *) mem;

  return (gpointer) & nvmem->mem_infer;
}

static void
gst_nvinfer_memory_unmap (GstMemory * mem)
{
}

/* Standard boiler plate. Assigning implemented function pointers. */
static void
gst_nvinfer_allocator_class_init (GstNvInferAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = GST_DEBUG_FUNCPTR (gst_nvinfer_allocator_alloc);
  allocator_class->free = GST_DEBUG_FUNCPTR (gst_nvinfer_allocator_free);
}

/* Standard boiler plate. Assigning implemented function pointers and setting
 * the memory type. */
static void
gst_nvinfer_allocator_init (GstNvInferAllocator * allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  parent->mem_type = GST_NVINFER_MEMORY_TYPE;
  parent->mem_map = gst_nvinfer_memory_map;
  parent->mem_unmap = gst_nvinfer_memory_unmap;
}

/* Create a new allocator of type GST_TYPE_NVINFER_ALLOCATOR and initialize
 * members. */
GstAllocator *
gst_nvinfer_allocator_new (guint width, guint height,
    NvBufSurfaceColorFormat color_format, guint batch_size, guint gpu_id)
{
  GstNvInferAllocator *allocator = (GstNvInferAllocator *)
      g_object_new (GST_TYPE_NVINFER_ALLOCATOR,
      nullptr);

  allocator->width = width;
  allocator->height = height;
  allocator->batch_size = batch_size;
  allocator->gpu_id = gpu_id;
  allocator->color_format = color_format;

  return (GstAllocator *) allocator;
}

GstNvInferMemory *
gst_nvinfer_buffer_get_memory (GstBuffer * buffer)
{
  GstMemory *mem;

  mem = gst_buffer_peek_memory (buffer, 0);

  if (!mem || !gst_memory_is_type (mem, GST_NVINFER_MEMORY_TYPE))
    return nullptr;

  return &(((GstNvInferMem *) mem)->mem_infer);
}
