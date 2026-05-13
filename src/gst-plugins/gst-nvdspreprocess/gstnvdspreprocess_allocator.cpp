/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gstnvdspreprocess_allocator.h"

/* Standard GStreamer boiler plate macros */
#define GST_TYPE_NVDSPREPROCESS_ALLOCATOR \
    (gst_nvdspreprocess_allocator_get_type ())
#define GST_NVDSPREPROCESS_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSPREPROCESS_ALLOCATOR,GstNvDsPreProcessAllocator))
#define GST_NVDSPREPROCESS_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSPREPROCESS_ALLOCATOR,GstNvDsPreProcessAllocatorClass))
#define GST_IS_NVDSPREPROCESS_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSPREPROCESS_ALLOCATOR))
#define GST_IS_NVDSPREPROCESS_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSPREPROCESS_ALLOCATOR))

typedef struct _GstNvDsPreProcessAllocator GstNvDsPreProcessAllocator;
typedef struct _GstNvDsPreProcessAllocatorClass GstNvDsPreProcessAllocatorClass;

G_GNUC_INTERNAL GType gst_nvdspreprocess_allocator_get_type (void);

GST_DEBUG_CATEGORY_STATIC (gst_nvdspreprocess_allocator_debug);
#define GST_CAT_DEFAULT gst_nvdspreprocess_allocator_debug

/**
 * Extends the GstAllocator class. Holds the parameters for allocator.
 */
struct _GstNvDsPreProcessAllocator
{
  /** standard gst allocator */
  GstAllocator allocator;
  /** ID of GPU on which it is being allocated */
  guint gpu_id;
  /** video buffer allocator info */
  GstNvDsPreProcessVideoBufferAllocatorInfo *info;
  /** raw buffer size */
  size_t raw_buf_size;
  /** boolean to denote if DEBUG_TENSOR flag is enabled */
  gboolean debug_tensor;
};

/** */
struct _GstNvDsPreProcessAllocatorClass
{
  /** gst allocator class */
  GstAllocatorClass parent_class;
};

/* Standard boiler plate to create a debug category and initializing the
 * allocator type.
 */
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvdspreprocess_allocator_debug, "nvdspreprocessallocator", 0, "nvdspreprocess allocator");
#define gst_nvdspreprocess_allocator_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvDsPreProcessAllocator, gst_nvdspreprocess_allocator,
    GST_TYPE_ALLOCATOR, _do_init);

/** Type of the memory allocated by the allocator. This can be used to identify
 * buffers / memories allocated by this allocator. */
#define GST_NVDSPREPROCESS_MEMORY_TYPE "nvdspreprocess"

/** Structure allocated internally by the allocator. */
typedef struct
{
  /** Should be the first member of a structure extending GstMemory. */
  GstMemory mem;
  /** Custom Gst memory for preprocess plugin */
  GstNvDsPreProcessMemory mem_preprocess;
} GstNvDsPreProcessMem;

/* Function called by GStreamer buffer pool to allocate memory using this
 * allocator. */
static GstMemory *
gst_nvdspreprocess_allocator_alloc (GstAllocator * allocator, gsize size,
    GstAllocationParams * params)
{
  GstNvDsPreProcessAllocator *preprocess_allocator = GST_NVDSPREPROCESS_ALLOCATOR (allocator);
  GstNvDsPreProcessMem *nvmem = new GstNvDsPreProcessMem;
  GstNvDsPreProcessMemory *tmem = &nvmem->mem_preprocess;
  NvBufSurfaceCreateParams create_params = { 0 };
  cudaError_t cudaReturn = cudaSuccess;

  if (preprocess_allocator->info == NULL) {
    if (preprocess_allocator->debug_tensor) {
      cudaReturn = cudaMallocHost(&tmem->dev_memory_ptr, preprocess_allocator->raw_buf_size);
    } else {
      cudaReturn = cudaMalloc(&tmem->dev_memory_ptr, preprocess_allocator->raw_buf_size);
    }
    if (cudaReturn != cudaSuccess) {
      GST_ERROR ("failed to allocate cuda malloc for tensor with error %s",
          cudaGetErrorName (cudaReturn));
      delete nvmem;
      return nullptr;
    }

    /* Initialize the GStreamer memory structure. */
    gst_memory_init ((GstMemory *) nvmem, (GstMemoryFlags) 0, allocator, nullptr,
        size, params->align, 0, size);

    return (GstMemory *) nvmem;
  }

  create_params.gpuId = preprocess_allocator->gpu_id;
  create_params.width = preprocess_allocator->info->width;
  create_params.height = preprocess_allocator->info->height;
  create_params.size = 0;
  create_params.isContiguous = 1;
  create_params.colorFormat = preprocess_allocator->info->color_format;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = preprocess_allocator->info->memory_type;

  if (NvBufSurfaceCreate (&tmem->surf, preprocess_allocator->info->batch_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer pool for nvdspreprocess");
    delete nvmem;
    return nullptr;
  }

  if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    if (NvBufSurfaceMapEglImage (tmem->surf, -1) != 0) {
      GST_ERROR ("Error: Could not map EglImage from NvBufSurface for nvdspreprocess");
      delete nvmem;
      return nullptr;
    }

    tmem->egl_frames.resize (preprocess_allocator->info->batch_size);
    tmem->cuda_resources.resize (preprocess_allocator->info->batch_size);
  }

  tmem->frame_memory_ptrs.assign (preprocess_allocator->info->batch_size, nullptr);

  for (guint i = 0; i < preprocess_allocator->info->batch_size; i++) {
#if defined(__aarch64__)
    if (tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
      if (cuGraphicsEGLRegisterImage (&tmem->cuda_resources[i],
              tmem->surf->surfaceList[i].mappedAddr.eglImage,
              CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        g_printerr ("Failed to register EGLImage in cuda\n");
        delete nvmem;
        return nullptr;
      }

      if (cuGraphicsResourceGetMappedEglFrame (&tmem->egl_frames[i],
              tmem->cuda_resources[i], 0, 0) != CUDA_SUCCESS) {
        g_printerr ("Failed to get mapped EGL Frame\n");
        delete nvmem;
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
gst_nvdspreprocess_allocator_free (GstAllocator * allocator, GstMemory * memory)
{
  GstNvDsPreProcessAllocator *preprocess_allocator = GST_NVDSPREPROCESS_ALLOCATOR (allocator);
  GstNvDsPreProcessMem *nvmem = (GstNvDsPreProcessMem *) memory;
  GstNvDsPreProcessMemory *tmem = &nvmem->mem_preprocess;

  if (preprocess_allocator->info == NULL) {
    cudaFree(tmem->dev_memory_ptr);
    delete nvmem;
    return;
  }

  if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    GstNvDsPreProcessAllocator *preprocess_allocator = GST_NVDSPREPROCESS_ALLOCATOR (allocator);

    for (size_t i = 0; i < preprocess_allocator->info->batch_size; i++) {
      cuGraphicsUnregisterResource (tmem->cuda_resources[i]);
    }
  }

  NvBufSurfaceUnMapEglImage (tmem->surf, -1);
  NvBufSurfaceDestroy (tmem->surf);

  delete nvmem;
}

/* Function called when mapping memory allocated by this allocator. Should
 * return pointer to GstNvDsPreProcessMemory. */
static gpointer
gst_nvdspreprocess_memory_map (GstMemory * mem, gsize maxsize, GstMapFlags flags)
{
  GstNvDsPreProcessMem *nvmem = (GstNvDsPreProcessMem *) mem;

  return (gpointer) & nvmem->mem_preprocess;
}

static void
gst_nvdspreprocess_memory_unmap (GstMemory * mem)
{
}

/* Standard boiler plate. Assigning implemented function pointers. */
static void
gst_nvdspreprocess_allocator_class_init (GstNvDsPreProcessAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_allocator_alloc);
  allocator_class->free = GST_DEBUG_FUNCPTR (gst_nvdspreprocess_allocator_free);
}

/* Standard boiler plate. Assigning implemented function pointers and setting
 * the memory type. */
static void
gst_nvdspreprocess_allocator_init (GstNvDsPreProcessAllocator * allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  parent->mem_type = GST_NVDSPREPROCESS_MEMORY_TYPE;
  parent->mem_map = gst_nvdspreprocess_memory_map;
  parent->mem_unmap = gst_nvdspreprocess_memory_unmap;
}

/* Create a new allocator of type GST_TYPE_NVDSPREPROCESS_ALLOCATOR and initialize
 * members. */
GstAllocator *
gst_nvdspreprocess_allocator_new ( GstNvDsPreProcessVideoBufferAllocatorInfo *info, 
    size_t raw_buf_size, guint gpu_id, gboolean debug_tensor)
{
  GstNvDsPreProcessAllocator *allocator = (GstNvDsPreProcessAllocator *)
      g_object_new (GST_TYPE_NVDSPREPROCESS_ALLOCATOR,
      nullptr);

  if (info != NULL) {
    allocator->info = new GstNvDsPreProcessVideoBufferAllocatorInfo;

    allocator->info->width = info->width;
    allocator->info->height = info->height;
    allocator->info->batch_size = info->batch_size;
    allocator->info->color_format = info->color_format;
    allocator->info->memory_type = info->memory_type;
  }

  allocator->gpu_id = gpu_id;
  allocator->raw_buf_size = raw_buf_size;
  allocator->debug_tensor = debug_tensor;

  return (GstAllocator *) allocator;
}

GstNvDsPreProcessMemory *
gst_nvdspreprocess_buffer_get_memory (GstBuffer * buffer)
{
  GstMemory *mem;

  mem = gst_buffer_peek_memory (buffer, 0);

  if (!mem || !gst_memory_is_type (mem, GST_NVDSPREPROCESS_MEMORY_TYPE))
    return nullptr;

  return &(((GstNvDsPreProcessMem *) mem)->mem_preprocess);
}
