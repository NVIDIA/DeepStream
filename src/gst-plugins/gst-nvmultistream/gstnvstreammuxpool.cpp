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

#include <gst/gst.h>
#include "gstnvstreammux.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvstream_allocator_debug);
#define GST_CAT_DEFAULT gst_nvstream_allocator_debug

#define GST_NVSTREAM_MEMORY_TYPE "nvstream"

#define GST_TYPE_NVSTREAM_ALLOCATOR \
  (gst_nvstream_allocator_get_type ())
#define GST_NVSTREAM_ALLOCATOR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVSTREAM_ALLOCATOR,GstNvStreamAllocator))
#define GST_NVSTREAM_ALLOCATOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVSTREAM_ALLOCATOR,GstNvStreamAllocatorClass))
#define GST_IS_NVSTREAM_ALLOCATOR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVSTREAM_ALLOCATOR))
#define GST_IS_NVSTREAM_ALLOCATOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVSTREAM_ALLOCATOR))
typedef struct _GstNvStreamAllocator GstNvStreamAllocator;
typedef struct _GstNvStreamAllocatorClass GstNvStreamAllocatorClass;

typedef struct
{
  guint batch_size;
} GstNvStreamAllocatorParams;

struct _GstNvStreamAllocator
{
  GstAllocator allocator;

  GstNvStreamAllocatorParams alloc_params;
};

struct _GstNvStreamAllocatorClass
{
  GstAllocatorClass parent_class;
};
G_GNUC_INTERNAL GType gst_nvstream_allocator_get_type (void);

#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvstream_allocator_debug, "nvstreamallocator", 0, "nvstream allocator");
#define gst_nvstream_allocator_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvStreamAllocator, gst_nvstream_allocator,
    GST_TYPE_ALLOCATOR, _do_init);

static GstMemory *
gst_nvstream_allocator_alloc (GstAllocator * allocator, gsize size,
    GstAllocationParams * params)
{
  GstNvStreamAllocator *nvstream_alloc = GST_NVSTREAM_ALLOCATOR (allocator);
  GstNvStreamMemory *nvmem = g_slice_new0 (GstNvStreamMemory);
  nvmem->surf.surfaceList =
      (NvBufSurfaceParams *)g_malloc0 (sizeof (NvBufSurfaceParams) *
      nvstream_alloc->alloc_params.batch_size);

  nvmem->batchsize = nvstream_alloc->alloc_params.batch_size;

  nvmem->orig_buffer_ptrs =
      (GstBuffer**)g_malloc0 (sizeof (GstBuffer *) *
      (nvstream_alloc->alloc_params.batch_size + 1));

  gst_memory_init ((GstMemory *) nvmem, (GstMemoryFlags)0, allocator, NULL,
      size, params->align, 0, size);

  return (GstMemory *) nvmem;
}

static void
gst_nvstream_allocator_free (GstAllocator * allocator, GstMemory * memory)
{
  GstNvStreamMemory *nvmem = (GstNvStreamMemory *) memory;
  g_free (nvmem->orig_buffer_ptrs);
  g_free (nvmem->surf.surfaceList);
  GST_DEBUG_OBJECT (allocator, "Freeing cuda memory %p", nvmem);
  g_slice_free (GstNvStreamMemory, nvmem);
}

static gpointer
gst_nvstream_memory_map (GstMemory * mem, gsize maxsize, GstMapFlags flags)
{
  GstNvStreamMemory *nvmem = (GstNvStreamMemory *) mem;
  return &nvmem->surf;
}

static void
gst_nvstream_memory_unmap (GstMemory * mem)
{
}

static void
gst_nvstream_allocator_class_init (GstNvStreamAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = GST_DEBUG_FUNCPTR (gst_nvstream_allocator_alloc);
  allocator_class->free = GST_DEBUG_FUNCPTR (gst_nvstream_allocator_free);
}

static void
gst_nvstream_allocator_init (GstNvStreamAllocator * allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  parent->mem_type = GST_NVSTREAM_MEMORY_TYPE;
  parent->mem_map = gst_nvstream_memory_map;
  parent->mem_unmap = gst_nvstream_memory_unmap;
}

static GstAllocator *
gst_nvstream_allocator_new (GstNvStreamAllocatorParams * params)
{
  GstNvStreamAllocator *allocator =
      (GstNvStreamAllocator *)g_object_new (GST_TYPE_NVSTREAM_ALLOCATOR, NULL);

  allocator->alloc_params = *params;

  return (GstAllocator *) allocator;
}


#define GST_TYPE_NVSTREAMMUX_BUFFERPOOL \
  (gst_nvstreammux_buffer_pool_get_type ())
#define GST_NVSTREAMMUX_BUFFERPOOL(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVSTREAMMUX_BUFFERPOOL,GstNvStreamMuxBufferPool))
#define GST_NVSTREAMMUX_BUFFERPOOL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVSTREAMMUX_BUFFERPOOL,GstNvStreamMuxBufferPoolClass))
#define GST_IS_NVSTREAMMUX_BUFFERPOOL(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVSTREAMMUX_BUFFERPOOL))
#define GST_IS_NVSTREAMMUX_BUFFERPOOL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVSTREAMMUX_BUFFERPOOL))
typedef struct _GstNvStreamMuxBufferPool GstNvStreamMuxBufferPool;
typedef struct _GstNvStreamMuxBufferPoolClass GstNvStreamMuxBufferPoolClass;

struct _GstNvStreamMuxBufferPool
{
  GstBufferPool pool;
};

struct _GstNvStreamMuxBufferPoolClass
{
  GstBufferPoolClass parent_class;
};

G_GNUC_INTERNAL GType gst_nvstreammux_buffer_pool_get_type (void);

GST_DEBUG_CATEGORY_STATIC (gst_nvstreammux_buffer_pool_debug);

#define _do_init_pool \
    GST_DEBUG_CATEGORY_INIT (gst_nvstreammux_buffer_pool_debug, "nvstreammuxbufferpool", 0, "nvstreammux buffer pool");
G_DEFINE_TYPE_WITH_CODE (GstNvStreamMuxBufferPool, gst_nvstreammux_buffer_pool,
    GST_TYPE_BUFFER_POOL, _do_init_pool);

#define gst_video_format_to_string(format) ((format == GST_VIDEO_FORMAT_UNKNOWN) ? "" : gst_video_format_to_string (format))

static void
gst_nvstreammux_buffer_pool_release_buffer (GstBufferPool * pool,
    GstBuffer * buffer)
{
  GstNvStreamMemory *mem;
  GstBuffer **buf;
  guint i = 0;

  mem = (GstNvStreamMemory *) gst_buffer_peek_memory (buffer, 0);

#if 0
  for (buf = mem->orig_buffer_ptrs; *buf; buf++) {
    gst_buffer_unref (*buf);
    *buf = NULL;
  }
#else
  if (mem) {
    for (i=0; i < mem->batchsize; i++)
    {
      buf = &mem->orig_buffer_ptrs[i];
      if (*buf)
      {
        //g_print ("TK: %s : %d GSTBuffer=%p OutBuf=%p\n\n", __func__, i, *buf, buffer);
        gst_buffer_unref (*buf);
        *buf = NULL;
      }
    }
  }

#endif
  GST_BUFFER_POOL_CLASS
      (gst_nvstreammux_buffer_pool_parent_class)->release_buffer (pool, buffer);
}

static void
gst_nvstreammux_buffer_pool_class_init (GstNvStreamMuxBufferPoolClass * klass)
{
  GstBufferPoolClass *buffer_pool_klass = GST_BUFFER_POOL_CLASS (klass);

  buffer_pool_klass->release_buffer =
      GST_DEBUG_FUNCPTR (gst_nvstreammux_buffer_pool_release_buffer);
}

static void
gst_nvstreammux_buffer_pool_init (GstNvStreamMuxBufferPool * pool)
{
}

GstBufferPool *
gst_nvstreammux_buffer_pool_new (guint batch_size, GstCaps *caps, guint pool_size)
{
  GstNvStreamMuxBufferPool *pool =
      (GstNvStreamMuxBufferPool *)g_object_new (GST_TYPE_NVSTREAMMUX_BUFFERPOOL, NULL);

  GstStructure *config = gst_buffer_pool_get_config ((GstBufferPool *) pool);
  gst_buffer_pool_config_set_params (config, caps, sizeof (NvBufSurface), pool_size, pool_size);
  GstAllocationParams allocation_params = { GstMemoryFlags(0) };

  GstNvStreamAllocatorParams params = { batch_size };
  GstAllocator *allocator = gst_nvstream_allocator_new (&params);
  gst_buffer_pool_config_set_allocator (config, allocator, &allocation_params);

  gst_buffer_pool_set_config ((GstBufferPool *) pool, config);
  gst_object_unref (allocator);

  return (GstBufferPool *) pool;
}

GstNvStreamMemory *
gst_buffer_get_nvstream_memory (GstBuffer * buf)
{
  GstMemory *mem;
  mem = gst_buffer_peek_memory (buf, 0);

  if (gst_memory_is_type (mem,"nvstream"))
    return (GstNvStreamMemory *) mem;
  else
    return NULL;
}

