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
 * Routing and role helpers that are safe to keep outside the graph-lifecycle
 * code. Live pad rewiring, blocking, promotion and swap stay in
 * gstnvmodelmux_bin.c because they own element lifetime and teardown ordering.
 */

#include "gstnvmodelmux_priv.h"

/* INTERNED role strings. Model-bin roles are read lock-free per frame and may
 * be re-tagged live by an in-place promote/swap. Atomic pointer stores to
 * process-static strings avoid allocation, frees and per-frame locking. */
static const gchar *
modelmux_role_intern (const gchar * role)
{
  static const gchar *PRIMARY = "Primary";
  static const gchar *SHADOW  = "Shadow";
  static const gchar *UNKNOWN = "Unknown";
  if (role && g_strcmp0 (role, "Shadow") == 0)
    return SHADOW;
  if (role && g_strcmp0 (role, "Unknown") == 0)
    return UNKNOWN;
  return PRIMARY;
}

G_GNUC_INTERNAL void
modelmux_role_set (ModelBin * model_bin, const gchar * role)
{
  g_atomic_pointer_set (&model_bin->role, (gpointer) modelmux_role_intern (role));
}

G_GNUC_INTERNAL const gchar *
modelmux_role_get (ModelBin * model_bin)
{
  return (const gchar *) g_atomic_pointer_get (&model_bin->role);
}

/* Role for per-frame overlay/provenance. This stays lock-free: the bin's pool
 * role is the slot role for every routing path, so model_bin->role is authoritative. */
G_GNUC_INTERNAL const gchar *
modelmux_resolve_role (ModelBin * model_bin, guint source_id)
{
  (void) source_id;
  return modelmux_role_get (model_bin);
}

G_GNUC_INTERNAL gint
modelmux_role_idx (const gchar * role)
{
  return (role && g_strcmp0 (role, "Shadow") == 0) ?
      MM_ROLE_SHADOW : MM_ROLE_PRIMARY;
}

/* Format one routing-table role cell into @buf:
 *   inactive            -> "-"
 *   passthru + pending  -> "<model> (passthru)"   (mapped to a still-warming model)
 *   passthru, no model  -> "(passthrough)"         (permanent / sharding-overflow)
 *   model branch        -> "<model>"               (inferring) */
G_GNUC_INTERNAL void
modelmux_fmt_route_cell (gchar * buf, gsize n, const ModelMuxRoleAttach * ra)
{
  /* ra->model / ra->pending_model are canonical "name@version" keys -> printed as-is. */
  if (!ra->active)
    g_strlcpy (buf, "-", n);
  else if (ra->passthru && ra->pending_model)
    g_snprintf (buf, n, "%s (passthru)", ra->pending_model);
  else if (ra->passthru)
    g_strlcpy (buf, "(passthrough)", n);
  else
    g_strlcpy (buf, ra->model ? ra->model : "-", n);
}
