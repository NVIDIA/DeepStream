/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _GST_UCX_SHARED_TASK_POOL_H_
#define _GST_UCX_SHARED_TASK_POOL_H_

G_BEGIN_DECLS
#include <gst/gst.h>
typedef struct _GstUcxSharedTaskPool GstUcxSharedTaskPool;
typedef struct _GstUcxSharedTaskPoolClass GstUcxSharedTaskPoolClass;
typedef struct _GstUcxSharedTaskPoolPrivate GstUcxSharedTaskPoolPrivate;

#define GST_TYPE_UCX_SHARED_TASK_POOL             (gst_ucx_shared_task_pool_get_type ())
#define GST_UCX_SHARED_TASK_POOL(pool)            (G_TYPE_CHECK_INSTANCE_CAST ((pool), GST_TYPE_TASK_POOL, GstUcxSharedTaskPool))
#define GST_IS_UCX_SHARED_TASK_POOL(pool)         (G_TYPE_CHECK_INSTANCE_TYPE ((pool), GST_TYPE_UCX_SHARED_TASK_POOL))
#define GST_UCX_SHARED_TASK_POOL_CLASS(pclass)    (G_TYPE_CHECK_CLASS_CAST ((pclass), GST_TYPE_UCX_SHARED_TASK_POOL, GstUcxSharedTaskPoolClass))
#define GST_IS_UCX_SHARED_TASK_POOL_CLASS(pclass) (G_TYPE_CHECK_CLASS_TYPE ((pclass), GST_TYPE_UCX_SHARED_TASK_POOL))
#define GST_UCX_SHARED_TASK_POOL_GET_CLASS(pool)  (G_TYPE_INSTANCE_GET_CLASS ((pool), GST_TYPE_UCX_SHARED_TASK_POOL, GstUcxSharedTaskPoolClass))

struct _GstUcxSharedTaskPool
{
  GstTaskPool parent;

  /*< private > */
  GstUcxSharedTaskPoolPrivate *priv;

  gpointer _gst_reserved[GST_PADDING];
};

struct _GstUcxSharedTaskPoolClass
{
  GstTaskPoolClass parent_class;

  /*< private > */
  gpointer _gst_reserved[GST_PADDING];
};

GST_API GType gst_ucx_shared_task_pool_get_type (void);

GST_API
    void gst_ucx_shared_task_pool_set_max_threads (GstUcxSharedTaskPool * pool,
    guint max_threads);

GST_API guint gst_ucx_shared_task_pool_get_max_threads (GstUcxSharedTaskPool *
    pool);

GST_API GstTaskPool *gst_ucx_shared_task_pool_new (void);

G_END_DECLS
#endif
