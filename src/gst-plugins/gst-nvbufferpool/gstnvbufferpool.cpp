/*
 * SPDX-FileCopyrightText: Copyright (c) 2012-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gstnvbufferpool.h"
#include "nvbufsurface.h"

#define parent_class gst_nvm_buffer_pool_parent_class
struct _GstNvmBufferPoolPrivate
{
  guint32        max_buffers;           // max-buffers in the pool
  guint32        timeout;               // bufferpool timeout
  guint32        width;                 // frame width for preallocation
  guint32        height;                // frame height for preallocation
  guint32        num_batched_buffers;
  guint32        buf_size;              // frame height for preallocation
  guint32        mem_type;              // frame height for preallocation
  guint32        flags;                 // flag to indicate preallocation
  guint32        gpu_id;                // gpu device id
  GAsyncQueue*   queue;                 // pool of NvMedia video surfaces
  GMutex         mutex;                 // bufferpool mutex
  GstStructure*  config;                // bufferpool configuration structure
  void*          device;                // associated NvMedia device
  char*          name;

  /* custom free function to release NvMedia video
     surfaces; register this func during set_config */
  gboolean (*client_release_callback)(gpointer, gpointer);
};

G_DEFINE_TYPE_WITH_CODE (GstNvmBufferPool, gst_nvm_buffer_pool, GST_TYPE_BUFFER_POOL, G_ADD_PRIVATE(GstNvmBufferPool));

GST_DEBUG_CATEGORY (nvm_buffer_pool_debug);
#define GST_CAT_DEFAULT nvm_buffer_pool_debug

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        printf ("Cuda failure gst-bufferpool: status=%d\n", status);   \
        abort();                                        \
    }                                                   \
}

#define CHECK_MALLOC(status)                            \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        printf ("Cuda failure: status=%d\n", status);   \
        abort();                                        \
        return false;                                   \
    }                                                   \
}

/*****************************************************************************/

GType
gst_nvds_memory_get_type (void)
{
  static gsize memory_format_type = 0;
  static const GEnumValue memory_format[] = {
#if 0
    { CUDA_PINNED_MEMORY,
      "Host/Pinned memory allocated using cudaMallocHost", "cuda-pinned-mem"},
    { CUDA_DEVICE_MEMORY,
      "Device memory allocated using cudaMalloc", "cuda-device-mem"},
    { CUDA_UNIFIED_MEMORY,
      "Unified memory allocated using cudaMallocManaged", "cuda-unified-mem"},
#endif
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&memory_format_type)) {
    GType tmp = g_enum_register_static ("GstNvDsMemoryType",
        memory_format);
    g_once_init_leave (&memory_format_type, tmp);
  }

  return (GType) memory_format_type;
}

static gboolean
gst_nvm_buffer_pool_stop (GstBufferPool * pool)
{
  GstNvmBufferPool *bpool = GST_NVM_BUFFER_POOL (pool);
  g_return_val_if_fail (GST_IS_NVM_BUFFER_POOL (bpool), FALSE);
  GstNvmBufferPoolPrivate* priv = bpool->priv;
  GstBuffer *buffer = NULL;
  //GstMapInfo buf_info;
  //NvBufSurface* surface;

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice(priv->gpu_id);
  if(CUerr != cudaSuccess)
  {
    printf ("Unable to set device in bufferpool\n");
    return FALSE;
  }
  GST_LOG_OBJECT (pool, "SETTING CUDA DEVICE = %d in nvbufferpool  func=%s\n", priv->gpu_id, __func__);

  while (g_async_queue_length (priv->queue))
  {
    buffer = (GstBuffer *) g_async_queue_pop (priv->queue);
#if 0
    if (!gst_buffer_map (buffer, &buf_info, GST_MAP_WRITE))
    {
      gst_buffer_unref (buffer);
      return GST_FLOW_ERROR;
    }
    surface = (NvBufSurface*) buf_info.data;

    if (surface->allocated_mem)
    {
      switch (surface->mem_type) {
        case CUDA_PINNED_MEMORY:
          CHECK (cudaFreeHost (surface->allocated_mem));
          GST_DEBUG_OBJECT (pool,"%s : %d Calling Free CUDA_PINNED_MEMORY name=%s\n", __func__, __LINE__, priv->name);
          break;
        case CUDA_DEVICE_MEMORY:
          GST_DEBUG_OBJECT (pool,"%s : %d Calling Free CUDA_DEVICE_MEMORY name=%s\n", __func__, __LINE__, priv->name);
          CHECK (cudaFree (surface->allocated_mem));
          break;
        case CUDA_UNIFIED_MEMORY:
          GST_DEBUG_OBJECT (pool,"%s : %d Calling Free CUDA_UNIFIED_MEMORY name=%s\n", __func__, __LINE__, priv->name);
          CHECK (cudaFree (surface->allocated_mem));
          break;
        default:
          GST_DEBUG ("Buffer pool invalid memtype %d", surface->mem_type);
          break;

      }
      surface->allocated_mem = NULL;
    }
#endif
    gst_buffer_unref (buffer);
  }

  GST_DEBUG ("Buffer pool (%p) destroyed", bpool);
  return TRUE;
}

  static void
gst_nvm_buffer_pool_release_buffer (GstBufferPool *pool, GstBuffer *buffer)
{
  GstNvmBufferPool *bpool = GST_NVM_BUFFER_POOL (pool);
  g_return_if_fail (GST_IS_NVM_BUFFER_POOL (bpool));
  GstNvmBufferPoolPrivate* priv = bpool->priv;
  g_return_if_fail (priv->config);

  g_mutex_lock (&priv->mutex);
  if (g_async_queue_length (priv->queue) == (gint) priv->max_buffers)
  {
    g_mutex_unlock (&priv->mutex);
    GST_WARNING ("Bufferpool queue overflow has been identified");
    return;
  }

  if (priv->client_release_callback)
  {
    if (priv->client_release_callback (priv->device, buffer))
    {
    } else {
      g_mutex_unlock (&priv->mutex);
      GST_WARNING ("Client release strategy failed");
      return;
    }
  }

  /* adding to the ready queue */
  g_async_queue_push (priv->queue, buffer);
  g_mutex_unlock (&priv->mutex);

  GST_DEBUG ("ReleaseVideoSurface - [pool:%p gstbuffer:%p]", pool, buffer);
  //g_print ("--- ReleaseVideoSurface - [pool=%p gstbuffer=%p]\n", pool, buffer);
  //gst_buffer_unref (buffer);
}

static GstFlowReturn
gst_nvm_buffer_pool_acquire_buffer (GstBufferPool *pool, GstBuffer **buffer,
                                    GstBufferPoolAcquireParams *params)
{
  GstNvmBufferPool *bpool = GST_NVM_BUFFER_POOL (pool);
  g_return_val_if_fail (GST_IS_NVM_BUFFER_POOL (bpool), GST_FLOW_ERROR);
  GstNvmBufferPoolPrivate* priv = bpool->priv;
  g_return_val_if_fail (priv->config, GST_FLOW_ERROR);
  NvBufSurface* surface = NULL;
  GstMapInfo buf_info;

  *buffer = (GstBuffer *) g_async_queue_pop (priv->queue);

  if (!gst_buffer_map (*buffer, &buf_info, GST_MAP_WRITE)) {
      gst_buffer_unref (*buffer);
      return GST_FLOW_ERROR;
    }

  surface = (NvBufSurface*) buf_info.data;

  //g_print ("---- %s: %s: bufpool: GSTBuffer=%p Surface=%p allocated_mem=%p \n", __func__, priv->name, *buffer, surface, surface->allocated_mem);

  gst_buffer_unmap (*buffer, &buf_info);

  GST_DEBUG ("AcquireVideoSurface: - [pool:%p surface:%p]", pool, surface);
  return GST_FLOW_OK;
}

/* create buffers inside the bufferpool with preallocated video surfaces */
static gboolean
gst_nvm_buffer_pool_start (GstBufferPool * pool)
{
  GstNvmBufferPool *bpool = GST_NVM_BUFFER_POOL (pool);
  g_return_val_if_fail (GST_IS_NVM_BUFFER_POOL (bpool), FALSE);
  GstNvmBufferPoolPrivate* priv = bpool->priv;
  NvBufSurface *surface = NULL;
  gboolean status = TRUE;
  GstMapInfo buf_info;
  GstBuffer *buffer;
  //CUresult result = CUDA_SUCCESS;
  guint i;//, k;
  GstMemory *memory = NULL;

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice(priv->gpu_id);
  if(CUerr != cudaSuccess)
  {
    printf ("Unable to set device in bufferpool\n");
    return FALSE;
  }
  GST_LOG_OBJECT (pool, "SETTING CUDA DEVICE = %d in nvbufferpool  func=%s\n", priv->gpu_id, __func__);

  if (priv->num_batched_buffers == 0)
  {
    priv->num_batched_buffers = 1;
  }

  g_assert (priv->width != 0);
  g_assert (priv->height != 0);
  g_assert (priv->buf_size != 0);
  g_assert (priv->max_buffers != 0);
  g_assert (priv->num_batched_buffers != 0);

  for (i = 0; i < priv->max_buffers; i++)
  {
    memory = gst_allocator_alloc (NULL, sizeof(NvBufSurface), NULL);
    //surface = (NvBufSurface *) calloc (1, sizeof(NvBufSurface));

    gst_memory_map (memory, &buf_info, ((GstMapFlags)(GST_MAP_READ | GST_MAP_WRITE)));
    surface = (NvBufSurface*)buf_info.data;
    NvBufSurfaceCreateParams cparams;
    cparams.gpuId = 0;
    cparams.width = priv->width;
    cparams.height = priv->height;
    cparams.size = 0;
    cparams.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    cparams.layout = NVBUF_LAYOUT_PITCH;
    cparams.memType = NVBUF_MEM_CUDA_PINNED;

    NvBufSurface *tsurf;
    NvBufSurfaceCreate (&tsurf, priv->num_batched_buffers, &cparams);
    *surface = *tsurf;


    /* Insert the surface into pool's ready queue */
    g_mutex_lock (&priv->mutex);
    if (g_async_queue_length (priv->queue) == (gint) priv->max_buffers) {

      GST_WARNING ("Bufferpool queue overflow has been identified \n");
      g_mutex_unlock (&priv->mutex);
      return status;
    }


    //buffer = gst_buffer_new_allocate (NULL, sizeof (NvBufSurface*), NULL);
    buffer = gst_buffer_new ();
    gst_buffer_insert_memory (buffer, 0, memory);

#if 0
    /* wrap the NvBufSurface inside GstBuffer */
    if (!gst_buffer_map (buffer, &buf_info, GST_MAP_WRITE)) {
      gst_buffer_unref (buffer);
      return GST_FLOW_ERROR;
    }
    *((NvBufSurface**) buf_info.data) = (NvBufSurface*) surface;
#endif

    g_async_queue_push (priv->queue, buffer);

    //gst_buffer_unmap (buffer, &buf_info);
    gst_memory_unmap (memory, &buf_info);

    GST_DEBUG ("VideoSurface(%d): gstBuffer=%p Surface=%p", i+1, buffer, surface);
    //g_print ("---- %s: bufpool: GSTBuffer=%p Surface=%p allocated_mem=%p \n", __func__, buffer, surface, surface->allocated_mem);
    g_mutex_unlock (&priv->mutex);
  }
  return status;
}

static gboolean
gst_nvm_buffer_pool_set_config (GstBufferPool * pool, GstStructure * config)
{
  GstNvmBufferPool *bpool = GST_NVM_BUFFER_POOL (pool);
  g_return_val_if_fail (GST_IS_NVM_BUFFER_POOL (bpool), FALSE);
  GstNvmBufferPoolPrivate* priv = bpool->priv;
  //guint surf_type;

  g_mutex_lock (&priv->mutex);
  priv->config = config;
  if (!gst_structure_get_uint (config, "width", &priv->width) ||
      !gst_structure_get_uint (config, "height", &priv->height)) {
    g_mutex_unlock (&priv->mutex);
    return FALSE;
  }
  gst_structure_get_uint (config, "memtype", &priv->mem_type);
  gst_structure_get_uint (config, "size", &priv->buf_size);
  gst_structure_get_uint (config, "max-buffers", &priv->max_buffers);
  gst_structure_get_uint (config, "num-batched-buffers", &priv->num_batched_buffers);
  gst_structure_get_uint (config, "gpu_id", &priv->gpu_id);

  gst_structure_get_uint (config, "timeout", &priv->timeout);
  if (!gst_structure_get (config, "free-func", G_TYPE_POINTER,
                          &priv->client_release_callback, NULL)) {
    priv->client_release_callback = NULL;
  }

  if (!gst_structure_get_uint (config, "flags", &priv->flags)) {
    priv->flags = 0;
  }
  if (!gst_structure_get (config, "device", G_TYPE_POINTER,
                          &priv->device, NULL)) {
    if (!priv->flags) {
      g_mutex_unlock (&priv->mutex);
      GST_WARNING ("NvMediaDevice not specified for buffer preallocation");
      return FALSE;
    }
  }
  gst_structure_get (config, "name", G_TYPE_POINTER, &priv->name, NULL);

  g_mutex_unlock (&priv->mutex);
#if 0
  g_print ("````````````````````````\n");
  g_print ("\n\tBufferPool configuration\n\n");
  g_print ("\twidth x height = %d x %d\n", priv->width, priv->height);
  g_print ("\tmax-buffers    = %d\n", priv->max_buffers);
  g_print ("\tsize           = %d\n", priv->buf_size);
  g_print ("\t# batched_bufs = %d\n", priv->num_batched_buffers);
  g_print ("\ttimeout        = %d\n", priv->timeout);
  g_print ("\tfree-func      = %p\n", priv->client_release_callback);
  g_print ("\tflags          = %d\n", priv->flags);
  g_print ("\tmem_type       = %d\n", priv->mem_type);
  g_print ("\tdevice         = %p\n", priv->device);
  g_print ("\tname           = %s\n", priv->name);
  g_print ("````````````````````````\n");
#endif
  return TRUE;
}

static void
gst_nvm_buffer_pool_finalize (GObject * object)
{
  GstNvmBufferPool *bpool = GST_NVM_BUFFER_POOL (object);
  GstNvmBufferPoolPrivate* priv = bpool->priv;

  g_mutex_clear (&priv->mutex);

  g_async_queue_unref(priv->queue);
  G_OBJECT_CLASS (gst_nvm_buffer_pool_parent_class)->finalize (object);
}

static void
gst_nvm_buffer_pool_init (GstNvmBufferPool * bpool)
{
  GstNvmBufferPool *self = GST_NVM_BUFFER_POOL (bpool);
  bpool->priv = (GstNvmBufferPoolPrivate *) gst_nvm_buffer_pool_get_instance_private (self);

  /* intialize the private structure */
  memset (bpool->priv, 0, sizeof (GstNvmBufferPoolPrivate));
  bpool->priv->queue = g_async_queue_new();
  g_mutex_init(&bpool->priv->mutex);
  bpool->priv->max_buffers = GST_NVM_MAX_BUFFERS_ALLOWED;
  bpool->priv->timeout = GST_NVM_DEFAULT_BUFFERPOOL_TIMEOUT;
}

static void
gst_nvm_buffer_pool_class_init (GstNvmBufferPoolClass * klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GstBufferPoolClass *pool_class = (GstBufferPoolClass *) klass;
  //parent_class = g_type_class_peek_parent (klass);
  // g_type_class_add_private (klass, sizeof (GstNvmBufferPoolPrivate));

  /* Initialize the GST debug system */
  GST_DEBUG_CATEGORY_INIT (nvm_buffer_pool_debug, "nvmbufferpool",
                           GST_DEBUG_FG_YELLOW, "Nvm Buffer Pool");

  /* override GstBufferPool's virtual functions */
  pool_class->set_config     = gst_nvm_buffer_pool_set_config;
  pool_class->start          = gst_nvm_buffer_pool_start;
  pool_class->stop           = gst_nvm_buffer_pool_stop;
  pool_class->acquire_buffer = gst_nvm_buffer_pool_acquire_buffer;
  pool_class->release_buffer = gst_nvm_buffer_pool_release_buffer;

  object_class->finalize = gst_nvm_buffer_pool_finalize;
}

GstBufferPool *
gst_nvm_buffer_pool_new (void)
{
  GstNvmBufferPool* pool = (GstNvmBufferPool *) g_object_new (GST_TYPE_NVM_BUFFER_POOL, NULL);
  GST_DEBUG ("Buffer pool (%p) created", pool);
  return GST_BUFFER_POOL_CAST (pool);
}

