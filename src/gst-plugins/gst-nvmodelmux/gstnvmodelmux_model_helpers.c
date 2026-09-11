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

/*
 * Read-only model helper routines shared by the bin, control plane and status
 * module. Keep stream lifecycle, pad ownership and routing logic out of this
 * file; those remain in gstnvmodelmux_bin.c.
 */

#include "gstnvmodelmux_priv.h"

#include <string.h>

/* Provenance/debug engine identity. nvinfer exposes the actual engine via
 * model-engine-file; nvinferserver has no such property, so identify it by the
 * config-file basename. Caller owns the returned string. */
G_GNUC_INTERNAL gchar *
get_infer_engine_str (ModelBin * model_bin)
{
  gchar *eng = NULL;
  if (model_bin->type == MODEL_INFER)
    g_object_get (G_OBJECT (model_bin->infer), "model-engine-file", &eng, NULL);
  else if (model_bin->config_file)
    eng = g_path_get_basename (model_bin->config_file);
  return eng;
}

/* Model version string for provenance/status: explicit version if set, else the
 * engine-file basename, else "". Borrowed pointer; the owner is ModelBin. */
G_GNUC_INTERNAL const gchar *
model_version_str (ModelBin * model_bin)
{
  const gchar *ver = (const gchar *) g_atomic_pointer_get (&model_bin->version);
  const gchar *eng = (const gchar *) g_atomic_pointer_get (&model_bin->engine);
  if (ver && *ver)
    return ver;
  if (eng && *eng) {
    const gchar *slash = strrchr (eng, '/');
    return slash ? slash + 1 : eng;
  }
  return "";
}

const gchar *
model_status_str (ModelStatus s)
{
  switch (s) {
    case MODEL_WARMED:  return "Warmed";
    case MODEL_SERVING: return "Serving";
    case MODEL_FAILED:  return "Failed";
    default:               return "Warming";
  }
}

/* Find the model bin, including shards, currently serving source_id. Caller
 * holds the owning ModelMuxBin lock while traversing pool membership. */
G_GNUC_INTERNAL ModelBin *
modelmux_perf_find_bin_for_source (ModelPool * pool, guint source_id)
{
  GHashTableIter it;
  gpointer k, v;
  if (!pool || !pool->models)
    return NULL;
  g_hash_table_iter_init (&it, pool->models);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelBin *model_bin;
    for (model_bin = (ModelBin *) v; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
      GHashTableIter j;
      gpointer jk, jv;
      if (!model_bin->idx_to_stream)
        continue;
      g_hash_table_iter_init (&j, model_bin->idx_to_stream);
      while (g_hash_table_iter_next (&j, &jk, &jv))
        if ((guint) GPOINTER_TO_INT (jv) == source_id)
          return model_bin;
    }
  }
  return NULL;
}
