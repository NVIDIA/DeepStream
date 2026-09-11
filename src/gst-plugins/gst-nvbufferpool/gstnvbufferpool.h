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

#ifndef _GST_NVM_BUFFER_POOL_H_
#define _GST_NVM_BUFFER_POOL_H_

#include <gst/gst.h>
#include <gst/video/video.h>

#ifdef __cplusplus
extern "C"
{
#endif

G_BEGIN_DECLS

#define GST_TYPE_NVM_BUFFER_POOL \
    (gst_nvm_buffer_pool_get_type ())
#define GST_IS_NVM_BUFFER_POOL(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVM_BUFFER_POOL))
#define GST_IS_NVM_BUFFER_POOL_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_NVM_BUFFER_POOL))
#define GST_NVM_BUFFER_POOL(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVM_BUFFER_POOL, GstNvmBufferPool))
#define GST_NVM_BUFFER_POOL_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_NVM_BUFFER_POOL, GstNvmBufferPoolClass))
#define GST_NVM_BUFFER_POOL_GET_CLASS(obj) \
    (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_NVM_BUFFER_POOL, GstNvmBufferPoolClass))
#define GST_NVM_BUFFER_POOL_GET_PRIVATE(obj) \
    (G_TYPE_INSTANCE_GET_PRIVATE ((obj), GST_TYPE_NVM_BUFFER_POOL, GstNvmBufferPoolPrivate))

typedef struct _GstNvmBufferPool        GstNvmBufferPool;
typedef struct _GstNvmBufferPoolClass   GstNvmBufferPoolClass;
typedef struct _GstNvmBufferPoolPrivate GstNvmBufferPoolPrivate;

#define GST_NVM_MAX_BUFFERS_ALLOWED        30
#define GST_NVM_DEFAULT_BUFFERPOOL_TIMEOUT 100000

/* surface allocation flags */
#define GST_NVM_ALLOC_VIDEO_SURFACE_FLAG     (1 << 0)
#define GST_NVM_ALLOC_DECODE_SURFACE_FLAG    (1 << 1)
#define GST_TYPE_NVDS_MEMORY_TYPE (gst_nvds_memory_get_type ())
GType gst_nvds_memory_get_type (void);

#define PROP_NVDS_MEMORY_TYPE_INSTALL(gobject_class) \
do {   \
  g_object_class_install_property (gobject_class, PROP_NVDS_MEMORY_TYPE, \
      g_param_spec_enum ("cuda-memory-type", "Type of CUDA memory allocated", \
        "Type of CUDA Memory to be allocated for output buffers",                        \
        GST_TYPE_NVDS_MEMORY_TYPE,                                    \
        CUDA_DEVICE_MEMORY,                                           \
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |    \
        GST_PARAM_MUTABLE_READY)));                                   \
}while(0)

struct _GstNvmBufferPool
{
  GstBufferPool bufferpool;
  GstNvmBufferPoolPrivate *priv;
};

struct _GstNvmBufferPoolClass
{
  GstBufferPoolClass parent_class;
};

GstBufferPool * gst_nvm_buffer_pool_new (void);

GType gst_nvm_buffer_pool_get_type (void);

G_END_DECLS

#ifdef __cplusplus
}
#endif

#endif /* _GST_NVM_BUFFER_POOL_H_ */
