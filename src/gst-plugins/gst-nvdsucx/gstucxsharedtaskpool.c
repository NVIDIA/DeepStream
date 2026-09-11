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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include "gstucxsharedtaskpool.h"

typedef struct
{
  gboolean done;
  guint64 id;
  GstTaskPoolFunction func;
  gpointer user_data;
  GMutex done_lock;
  GCond done_cond;
  gint refcount;
} UcxSharedTaskData;

static UcxSharedTaskData *
shared_task_data_ref (UcxSharedTaskData * tdata)
{
  g_atomic_int_add (&tdata->refcount, 1);

  return tdata;
}

static void
shared_task_data_unref (UcxSharedTaskData * tdata)
{
  if (g_atomic_int_dec_and_test (&tdata->refcount)) {
    g_mutex_clear (&tdata->done_lock);
    g_cond_clear (&tdata->done_cond);
    g_slice_free (UcxSharedTaskData, tdata);
  }
}

struct _GstUcxSharedTaskPoolPrivate
{
  guint max_threads;
};

#define GST_UCX_SHARED_TASK_POOL_CAST(pool)       ((GstUcxSharedTaskPool*)(pool))

G_DEFINE_TYPE_WITH_PRIVATE (GstUcxSharedTaskPool, gst_ucx_shared_task_pool,
    GST_TYPE_TASK_POOL);

static void
shared_func (UcxSharedTaskData * tdata, GstTaskPool * pool)
{
  tdata->func (tdata->user_data);

  g_mutex_lock (&tdata->done_lock);
  tdata->done = TRUE;
  g_cond_signal (&tdata->done_cond);
  g_mutex_unlock (&tdata->done_lock);

  shared_task_data_unref (tdata);
}

static gpointer
shared_push (GstTaskPool * pool, GstTaskPoolFunction func,
    gpointer user_data, GError ** error)
{
  UcxSharedTaskData *ret = NULL;

  GST_OBJECT_LOCK (pool);

  if (!pool->pool) {
    GST_OBJECT_UNLOCK (pool);
    goto done;
  }

  ret = g_slice_new (UcxSharedTaskData);

  ret->done = FALSE;
  ret->func = func;
  ret->user_data = user_data;
  g_atomic_int_set (&ret->refcount, 1);
  g_cond_init (&ret->done_cond);
  g_mutex_init (&ret->done_lock);

  g_thread_pool_push (pool->pool, shared_task_data_ref (ret), error);

  GST_OBJECT_UNLOCK (pool);

done:
  return ret;
}

static void
shared_join (GstTaskPool * pool, gpointer id)
{
  UcxSharedTaskData *tdata;

  if (!id)
    return;

  tdata = (UcxSharedTaskData *) id;

  g_mutex_lock (&tdata->done_lock);
  while (!tdata->done) {
    g_cond_wait (&tdata->done_cond, &tdata->done_lock);
  }
  g_mutex_unlock (&tdata->done_lock);

  shared_task_data_unref (tdata);
}

static void
shared_dispose_handle (GstTaskPool * pool, gpointer id)
{
  UcxSharedTaskData *tdata;

  if (!id)
    return;

  tdata = (UcxSharedTaskData *) id;


  shared_task_data_unref (tdata);
}

static void
shared_prepare (GstTaskPool * pool, GError ** error)
{
  GstUcxSharedTaskPool *shared_pool = GST_UCX_SHARED_TASK_POOL_CAST (pool);

  GST_OBJECT_LOCK (pool);
  pool->pool =
      g_thread_pool_new ((GFunc) shared_func, pool,
      shared_pool->priv->max_threads, FALSE, error);
  GST_OBJECT_UNLOCK (pool);
}

static void
gst_ucx_shared_task_pool_class_init (GstUcxSharedTaskPoolClass * klass)
{
  GstTaskPoolClass *taskpoolclass = GST_TASK_POOL_CLASS (klass);

  taskpoolclass->prepare = shared_prepare;
  taskpoolclass->push = shared_push;
  taskpoolclass->join = shared_join;
}

static void
gst_ucx_shared_task_pool_init (GstUcxSharedTaskPool * pool)
{
  GstUcxSharedTaskPoolPrivate *priv;

  priv = pool->priv = (GstUcxSharedTaskPoolPrivate *)
      gst_ucx_shared_task_pool_get_instance_private (pool);
  priv->max_threads = 1;
}

/**
 * gst_shared_task_pool_set_max_threads:
 * @pool: a #GstSharedTaskPool
 * @max_threads: Maximum number of threads to spawn.
 *
 * Update the maximal number of threads the @pool may spawn. When
 * the maximal number of threads is reduced, existing threads are not
 * immediately shut down, see g_thread_pool_set_max_threads().
 *
 * Setting @max_threads to 0 effectively freezes the pool.
 *
 * Since: 1.20
 */
void
gst_ucx_shared_task_pool_set_max_threads (GstUcxSharedTaskPool * pool,
    guint max_threads)
{
  GstTaskPool *taskpool;

  g_return_if_fail (GST_IS_UCX_SHARED_TASK_POOL (pool));

  taskpool = GST_TASK_POOL (pool);

  GST_OBJECT_LOCK (pool);
  if (taskpool->pool)
    g_thread_pool_set_max_threads (taskpool->pool, max_threads, NULL);
  pool->priv->max_threads = max_threads;
  GST_OBJECT_UNLOCK (pool);
}

/**
 * gst_shared_task_pool_get_max_threads:
 * @pool: a #GstSharedTaskPool
 *
 * Returns: the maximum number of threads @pool is configured to spawn
 * Since: 1.20
 */
guint
gst_ucx_shared_task_pool_get_max_threads (GstUcxSharedTaskPool * pool)
{
  guint ret;

  g_return_val_if_fail (GST_IS_UCX_SHARED_TASK_POOL (pool), 0);

  GST_OBJECT_LOCK (pool);
  ret = pool->priv->max_threads;
  GST_OBJECT_UNLOCK (pool);

  return ret;
}

/**
 * gst_shared_task_pool_new:
 *
 * Create a new shared task pool. The shared task pool will queue tasks on
 * a maximum number of threads, 1 by default.
 *
 * Do not use a #GstSharedTaskPool to manage potentially inter-dependent tasks such
 * as pad tasks, as having one task waiting on another to return before returning
 * would cause obvious deadlocks if they happen to share the same thread.
 *
 * Returns: (transfer full): a new #GstSharedTaskPool. gst_object_unref() after usage.
 * Since: 1.20
 */
GstTaskPool *
gst_ucx_shared_task_pool_new (void)
{
  GstTaskPool *pool;

  pool = (GstTaskPool *) g_object_new (GST_TYPE_UCX_SHARED_TASK_POOL, NULL);

  /* clear floating flag */
  gst_object_ref_sink (pool);

  return pool;
}
