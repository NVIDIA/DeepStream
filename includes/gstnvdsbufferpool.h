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
