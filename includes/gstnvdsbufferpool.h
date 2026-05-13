/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GSTNVDSBUFFERPOOL_H_
#define GSTNVDSBUFFERPOOL_H_

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _GstNvDsBufferPool GstNvDsBufferPool;
typedef struct _GstNvDsBufferPoolClass GstNvDsBufferPoolClass;
typedef struct _GstNvDsBufferPoolPrivate GstNvDsBufferPoolPrivate;

#define GST_TYPE_NVDS_BUFFER_POOL      (gst_nvds_buffer_pool_get_type())
#define GST_IS_NVDS_BUFFER_POOL(obj)   (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVDS_BUFFER_POOL))
#define GST_NVDS_BUFFER_POOL(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVDS_BUFFER_POOL, GstNvDsBufferPool))
#define GST_NVDS_BUFFER_POOL_CAST(obj) ((GstNvDsBufferPool*)(obj))

#define GST_NVDS_MEMORY_TYPE "nvds"
#define GST_BUFFER_POOL_OPTION_NVDS_META "GstBufferPoolOptionNvDsMeta"

struct _GstNvDsBufferPool
{
  GstBufferPool bufferpool;

  GstNvDsBufferPoolPrivate *priv;
};

struct _GstNvDsBufferPoolClass
{
  GstBufferPoolClass parent_class;
};

GType gst_nvds_buffer_pool_get_type (void);

GstBufferPool* gst_nvds_buffer_pool_new (void);

G_END_DECLS

#endif /* GSTNVDSBUFFERPOOL_H_ */
