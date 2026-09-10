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
 * MM_DEBUG helpers and diagnostic table rendering. Keep this file read-only
 * with respect to graph topology: it formats state that the main bin already
 * owns, and schedules coalesced dumps from the main loop.
 */

#include "gstnvmodelmux_priv.h"

#include <stdarg.h>
#include <stdlib.h>

GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat
#define MM_INFO(fmt, ...)  GST_INFO (fmt, ##__VA_ARGS__)

G_GNUC_INTERNAL gboolean
modelmux_debug_enabled (void)
{
  static gint cached = -1;
  if (cached < 0) {
    const gchar *e = g_getenv ("MM_DEBUG");
    cached = (e && *e && g_strcmp0 (e, "0") != 0) ? 1 : 0;
  }
  return cached;
}

G_GNUC_INTERNAL void
modelmux_debug_configure (ModelMuxBin * modelmux_bin)
{
  const gchar *debug;
  const gchar *debug_interval;
  if (!modelmux_bin)
    return;

  debug = g_getenv ("MM_DEBUG");
  debug_interval = g_getenv ("MM_DEBUG_INTERVAL");
  modelmux_bin->log_enabled = (debug && *debug && g_strcmp0 (debug, "0") != 0);
  {
    /* g_ascii_strtoll instead of atoi: atoi is UB on out-of-range input, and this
     * is an env var. Clamp to a sane range; anything invalid -> the 100 default. */
    gint64 value = debug_interval ? g_ascii_strtoll (debug_interval, NULL, 10) : 0;
    modelmux_bin->debug_interval = (value > 0 && value <= G_MAXINT32) ? (guint) value : 100;
  }
}

/* short GstState string for the CURRENT state of an element */
static const gchar *
modelmux_gst_state_str (GstElement * el)
{
  GstState st = GST_STATE_VOID_PENDING;
  if (!el)
    return "(null)";
  gst_element_get_state (el, &st, NULL, 0);
  switch (st) {
    case GST_STATE_NULL:    return "NULL";
    case GST_STATE_READY:   return "READY";
    case GST_STATE_PAUSED:  return "PAUSED";
    case GST_STATE_PLAYING: return "PLAYING";
    default:                return "VOID";
  }
}

/* ---- pretty closed-box table rendering (MM_DEBUG dump) ---------------------
 * Each table is drawn as a self-contained grid with AUTO-sized columns. */
static void
modelmux_glyph_rep (GString * s, const gchar * g, guint n)
{
  guint i;
  for (i = 0; i < n; i++)
    g_string_append (s, g);
}

static void
modelmux_grid_rule (GString * s, guint nc, const guint * w,
    const gchar * l, const gchar * mid, const gchar * r)
{
  guint c;
  g_string_append (s, "  ");
  g_string_append (s, l);
  for (c = 0; c < nc; c++) {
    modelmux_glyph_rep (s, "─", w[c] + 2);
    g_string_append (s, (c + 1 < nc) ? mid : r);
  }
  g_string_append_c (s, '\n');
}

static void
modelmux_pad_cell (GString * s, const gchar * txt, guint w)
{
  glong vis;
  txt = txt ? txt : "-";
  vis = g_utf8_strlen (txt, -1);
  g_string_append (s, txt);
  if ((glong) w > vis)
    modelmux_glyph_rep (s, " ", (guint) ((glong) w - vis));
}

static void
modelmux_grid_row (GString * s, guint nc, const guint * w, const gchar ** cells)
{
  guint c;
  g_string_append (s, "  │");
  for (c = 0; c < nc; c++) {
    g_string_append_c (s, ' ');
    modelmux_pad_cell (s, cells[c], w[c]);
    g_string_append (s, " │");
  }
  g_string_append_c (s, '\n');
}

static void
modelmux_grid (GString * s, const gchar * title, guint nc,
    const gchar ** hdr, GPtrArray * rows)
{
  guint *w = g_new0 (guint, nc);
  guint c, r;
  for (c = 0; c < nc; c++)
    w[c] = (guint) g_utf8_strlen (hdr[c], -1);
  for (r = 0; r < rows->len; r++) {
    gchar **cells = (gchar **) g_ptr_array_index (rows, r);
    for (c = 0; c < nc; c++) {
      guint l = cells[c] ? (guint) g_utf8_strlen (cells[c], -1) : 1;
      if (l > w[c])
        w[c] = l;
    }
  }
  if (title)
    g_string_append_printf (s, "  %s\n", title);
  modelmux_grid_rule (s, nc, w, "┌", "┬", "┐");
  modelmux_grid_row (s, nc, w, hdr);
  modelmux_grid_rule (s, nc, w, "├", "┼", "┤");
  for (r = 0; r < rows->len; r++)
    modelmux_grid_row (s, nc, w, (const gchar **) g_ptr_array_index (rows, r));
  modelmux_grid_rule (s, nc, w, "└", "┴", "┘");
  g_free (w);
}

static void
modelmux_grid_add (GPtrArray * rows, guint nc, ...)
{
  va_list ap;
  guint c;
  gchar **cells = g_new0 (gchar *, nc + 1);
  va_start (ap, nc);
  for (c = 0; c < nc; c++) {
    const gchar *cell = va_arg (ap, const gchar *);
    /* never store a NULL cell: rows are freed with g_strfreev, which stops at the
     * first NULL -- a NULL mid-row would silently LEAK every cell after it (the
     * renderer tolerates NULL, the free function does not). */
    cells[c] = g_strdup (cell ? cell : "");
  }
  va_end (ap);
  g_ptr_array_add (rows, cells);
}

static void
modelmux_titlebox (GString * s, GPtrArray * lines)
{
  guint i, w = 0;
  for (i = 0; i < lines->len; i++) {
    guint l = (guint) g_utf8_strlen (g_ptr_array_index (lines, i), -1);
    if (l > w)
      w = l;
  }
  g_string_append (s, "  ╭");
  modelmux_glyph_rep (s, "─", w + 2);
  g_string_append (s, "╮\n");
  for (i = 0; i < lines->len; i++) {
    g_string_append (s, "  │ ");
    modelmux_pad_cell (s, (gchar *) g_ptr_array_index (lines, i), w);
    g_string_append (s, " │\n");
  }
  g_string_append (s, "  ╰");
  modelmux_glyph_rep (s, "─", w + 2);
  g_string_append (s, "╯\n");
}

static void
modelmux_dump_pool (ModelPool * rb, GPtrArray * rows)
{
  GHashTableIter it;
  gpointer k, v;
  /* the pool CARRIES its runtime default designation -- read it, not the config
   * (the config's default_* fields are only the parse-time seed). */
  const gchar *def_name = rb->default_ref.name;
  const gchar *def_ver = rb->default_ref.version;
  g_hash_table_iter_init (&it, rb->models);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelBin *base = (ModelBin *) v, *model_bin;
    guint nshards = 0;
    for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard)
      nshards++;
    for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
      gchar *eng = NULL, *eng_base;
      gchar shard[16], c_gie[16], c_bat[16], c_gpu[16];
      const gchar *ver;
      gboolean is_def;
      if (model_bin->infer)
        eng = get_infer_engine_str (model_bin);
      eng_base = (eng && *eng) ? g_path_get_basename (eng) : g_strdup ("-");
      g_snprintf (shard, sizeof (shard), "%u/%u", model_bin->inst, nshards - 1);
      ver = model_version_str (model_bin);
      is_def = (def_name && g_strcmp0 (model_bin->name, def_name) == 0 &&
          g_strcmp0 (ver, def_ver ? def_ver : MM_MODEL_VERSION_DEFAULT) == 0);
      g_snprintf (c_gie, sizeof (c_gie), "%u", model_bin->unique_id);
      g_snprintf (c_bat, sizeof (c_bat), "%u", model_bin->num_streams);
      /* placement, fully RESOLVED to a device number: the shard's recorded
       * device, else the per-version registry, else the gpu the model's own
       * config file pins (model_bin->cfg_gpu, parsed once at creation), else the
       * backend default 0 -- the same resolution order the scheduler's
       * effective-gpu check uses, so the table never shows an opaque "config". */
      if (model_bin->gpu >= 0)
        g_snprintf (c_gpu, sizeof (c_gpu), "%d", model_bin->gpu);
      else {
        /* model_version_str: atomic read + NULL->default, matching every
         * other registry lookup (a raw model_bin->version would miss the entry) */
        gint vg = rb->config ? modelmux_config_version_gpu (rb->config, model_bin->name,
            model_version_str (model_bin)) : -1;
        if (vg < 0)
          vg = model_bin->cfg_gpu;
        g_snprintf (c_gpu, sizeof (c_gpu), "%d", vg >= 0 ? vg : 0);
      }
      modelmux_grid_add (rows, 11, rb->role, model_bin->name, modelmux_infer_plugin_str (model_bin->type),
          ver, is_def ? "yes" : "-", c_gie, shard, c_gpu,
          model_status_str (model_bin_status (model_bin)), c_bat, eng_base);
      g_free (eng);
      g_free (eng_base);
    }
  }
}

/* effective device of one shard, resolved with SCHEDULER parity (own gpu ->
 * version registry -> the config file's own pin -> 0). Debug-local mirror of
 * bin.c's modelmux_shard_gpu_effective (that one is static). Caller holds modelmux_bin->lock. */
static gint
modelmux_dbg_shard_gpu (const ModelMuxConfig * config, ModelBin * model_bin)
{
  gint gpu = model_bin->gpu;
  if (gpu < 0 && config)
    gpu = modelmux_config_version_gpu (config, model_bin->name, model_version_str (model_bin));
  if (gpu < 0)
    gpu = model_bin->cfg_gpu;
  return gpu >= 0 ? gpu : 0;
}

static void
modelmux_dump_pool_perf (ModelPool * rb, GPtrArray * rows)
{
  GHashTableIter it;
  gpointer k, v;
  if (!rb || !rb->models)
    return;
  g_hash_table_iter_init (&it, rb->models);
  while (g_hash_table_iter_next (&it, &k, &v)) {
    ModelBin *base = (ModelBin *) v, *model_bin;
    gdouble agg_fps = 0, base_lat = 0;
    const gchar *ver = model_version_str (base);
    gchar mcell[80], c_gie[16], c_fps[24], c_lat[24];
    gchar c_gpu[96] = "", c_util[96] = "";   /* 16 devices x "ddd," fits */
    gint seen[16];                       /* distinct devices of this chain */
    guint nseen = 0, si;
    for (model_bin = base; model_bin; model_bin = (ModelBin *) model_bin->next_shard) {
      gdouble f = 0, l = 0;
      gint g = modelmux_dbg_shard_gpu (rb->config, model_bin);
      gboolean dup = FALSE;
      if (model_bin->perf)
        modelmux_perf_get_model (model_bin->perf, &f, &l);
      agg_fps += f;
      if (model_bin == base)
        base_lat = l;
      for (si = 0; si < nseen && !dup; si++)
        dup = (seen[si] == g);
      if (!dup) {
        if (nseen < G_N_ELEMENTS (seen))
          seen[nseen++] = g;
        else
          GST_WARNING ("debug dump gpu column: more than %u distinct devices -- "
              "device %d omitted", (guint) G_N_ELEMENTS (seen), g);
      }
    }
    /* per-device WHOLE-GPU utilization (NVML; co-location attribution) --
     * "0,1" / "34,71" line up column-wise per device */
    for (si = 0; si < nseen; si++) {
      guint util = 0;
      gchar tmp[16];
      gboolean have = modelmux_gpu_stats (seen[si], &util, NULL, NULL);
      g_snprintf (tmp, sizeof (tmp), "%s%d", si ? "," : "", seen[si]);
      g_strlcat (c_gpu, tmp, sizeof (c_gpu));
      if (have)
        g_snprintf (tmp, sizeof (tmp), "%s%u", si ? "," : "", util);
      else
        g_snprintf (tmp, sizeof (tmp), "%s-", si ? "," : "");
      g_strlcat (c_util, tmp, sizeof (c_util));
    }
    g_snprintf (mcell, sizeof (mcell), "%s@%s", base->name, ver ? ver : "-");
    g_snprintf (c_gie, sizeof (c_gie), "%u", base->unique_id);
    g_snprintf (c_fps, sizeof (c_fps), "%.2f", agg_fps);
    g_snprintf (c_lat, sizeof (c_lat), "%.2f", base_lat);
    modelmux_grid_add (rows, 7, rb->role, mcell, c_gie, c_gpu, c_fps, c_lat, c_util);
  }
}

static gboolean
modelmux_dump_state_idle (gpointer data)
{
  ModelMuxBin *modelmux_bin = (ModelMuxBin *) data;
  GHashTableIter it;
  gpointer k, v;
  GString *s;
  GPtrArray *rows;
  GPtrArray *tl;

  g_mutex_lock (&modelmux_bin->free_lock);
  modelmux_bin->dump_source_id = 0;
  g_mutex_unlock (&modelmux_bin->free_lock);
  if (g_atomic_int_get (&modelmux_bin->shutting_down))
    return G_SOURCE_REMOVE;

  s = g_string_new ("\n");
  rows = g_ptr_array_new_with_free_func ((GDestroyNotify) g_strfreev);
  tl = g_ptr_array_new_with_free_func (g_free);

  g_mutex_lock (&modelmux_bin->lock);
  /* re-check under the lock (TOCTOU vs _free's barrier): a dispatch that started
   * before _free()'s g_source_remove must not walk state _free is destroying. */
  if (g_atomic_int_get (&modelmux_bin->shutting_down)) {
    g_mutex_unlock (&modelmux_bin->lock);
    g_string_free (s, TRUE);
    g_ptr_array_free (rows, TRUE);
    g_ptr_array_free (tl, TRUE);
    return G_SOURCE_REMOVE;
  }

  g_ptr_array_add (tl, g_strdup_printf ("MultiModelBin  ·  state @ frame %u",
          (guint) g_atomic_int_get (&modelmux_bin->frame_count)));
  g_ptr_array_add (tl, g_strdup_printf (
          "pipeline=%s   in-demux=%s   combined-mux=%s   active-streams=%u",
          modelmux_gst_state_str (modelmux_bin->parent_pipeline), modelmux_gst_state_str (modelmux_bin->in_demux),
          modelmux_gst_state_str (modelmux_bin->out_mux), g_hash_table_size (modelmux_bin->stream_wiring)));
  modelmux_titlebox (s, tl);
  g_ptr_array_free (tl, TRUE);
  g_string_append_c (s, '\n');

  {
    const gchar *hdr[11] = { "role", "name", "infer-plugin", "version",
      "default", "gie-id", "shard", "gpu", "status", "batched", "engine (current)" };
    modelmux_dump_pool (modelmux_bin->primary_pool, rows);
    modelmux_dump_pool (modelmux_bin->shadow_pool, rows);
    if (modelmux_bin->limbo_models) {
      GHashTableIter lit;
      gpointer lk, lv;
      g_hash_table_iter_init (&lit, modelmux_bin->limbo_models);
      while (g_hash_table_iter_next (&lit, &lk, &lv)) {
        ModelBin *model_bin = (ModelBin *) lv;
        gchar *eng = NULL, *eng_base;
        gchar c_gie[16], c_gpu[16];
        if (model_bin->infer)
          eng = get_infer_engine_str (model_bin);
        eng_base = (eng && *eng) ? g_path_get_basename (eng) : g_strdup ("-");
        g_snprintf (c_gie, sizeof (c_gie), "%u", model_bin->unique_id);
        /* same fully-resolved device number as the pool rows (never "config") */
        g_snprintf (c_gpu, sizeof (c_gpu), "%d", modelmux_dbg_shard_gpu (modelmux_bin->config, model_bin));
        modelmux_grid_add (rows, 11, "Unknown", model_bin->name,
            modelmux_infer_plugin_str (model_bin->type), model_version_str (model_bin), "-",
            c_gie, "-", c_gpu, model_status_str (model_bin_status (model_bin)),
            "0", eng_base);
        g_free (eng);
        g_free (eng_base);
      }
    }
    modelmux_grid (s, "models_table  —  active model bins (role → name → infer-plugin → status)",
        11, hdr, rows);
    g_ptr_array_set_size (rows, 0);
  }
  g_string_append_c (s, '\n');

  {
    const gchar *hdr[4] = { "stream", "source name", "primary", "shadow" };
    g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelMuxStreamEntry *e = (ModelMuxStreamEntry *) v;
      const gchar *nm = (const gchar *) g_hash_table_lookup (modelmux_bin->stream_names,
          GINT_TO_POINTER ((gint) e->stream_id));
      gchar c_sid[16], pcell[80], scell[80];
      g_snprintf (c_sid, sizeof (c_sid), "%u", e->stream_id);
      modelmux_fmt_route_cell (pcell, sizeof (pcell), &e->prim);
      modelmux_fmt_route_cell (scell, sizeof (scell), &e->shad);
      modelmux_grid_add (rows, 4, c_sid, nm ? nm : "-", pcell, scell);
    }
    modelmux_grid (s, "routing_table  —  per-stream model mapping (stream → primary/shadow)",
        4, hdr, rows);
    g_ptr_array_set_size (rows, 0);
  }
  g_string_append_c (s, '\n');

  {
    const gchar *hdr[11] = { "stream", "source name", "role", "entered",
      "inferred", "delivered", "wait_infer", "wait_out", "buffered",
      "dropped", "status" };
    g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
    while (g_hash_table_iter_next (&it, &k, &v)) {
      ModelMuxStreamEntry *e = (ModelMuxStreamEntry *) v;
      guint sid = e->stream_id;
      const gchar *nm;
      gint rr;
      ModelMuxAcct *a;
      gint in_now;
      if (sid >= MM_ACCT_MAX)
        continue;
      nm = (const gchar *) g_hash_table_lookup (modelmux_bin->stream_names,
          GINT_TO_POINTER ((gint) sid));
      a = &modelmux_bin->acct[sid];
      in_now = g_atomic_int_get (&a->in_count);
      for (rr = 0; rr < MM_ROLE_SLOTS; rr++) {
        ModelMuxRoleAttach *ra = (rr == MM_ROLE_PRIMARY) ? &e->prim : &e->shad;
        ModelMuxAcctRole *ar = &a->role[rr];
        gint entered, inferred, delivered, buffered, dropped;
        gint wait_infer, wait_out;
        const gchar *status;
        if (!ra->active || ra->passthru)
          continue;
        entered = in_now - ar->in_base;
        inferred = g_atomic_int_get (&ar->infer) - ar->infer_base;
        delivered = g_atomic_int_get (&ar->out) - ar->out_base;
        buffered = entered - delivered;
        wait_infer = entered - inferred;
        wait_out = inferred - delivered;
        if (ar->warm < 8) {
          ar->warm++;
          if (buffered > ar->floor)
            ar->floor = buffered;
          dropped = 0;
          status = "warm";
        } else {
          dropped = buffered - (ar->floor + 4);
          if (dropped < 0)
            dropped = 0;
          /* one-sample debounce: a NEIGHBOR stream's branch teardown / inner
           * mux+demux rebuild can push a couple of EXTRA in-flight frames for
           * exactly one dump (they are delivered, not lost -- the next sample
           * is back at the floor, and a true loss counter cannot decrement).
           * Real loss is permanent: delivered never catches up, so the excess
           * persists across consecutive samples -- report DROP only then. The
           * FRAMEGAP contiguity check independently proves any real one-time
           * drop per exact frame number, so nothing is masked. */
          ar->over = (dropped > 0) ? ar->over + 1 : 0;
          if (ar->over < 2)
            dropped = 0;
          status = (dropped > 0) ? "DROP" : "ok";
        }
        if (wait_infer < 0) wait_infer = 0;
        if (wait_out < 0)   wait_out = 0;
        if (buffered < 0)   buffered = 0;
        {
          gchar c_sid[16], c_en[16], c_in[16], c_de[16], c_wi[16], c_wo[16],
              c_bf[16], c_dr[16];
          g_snprintf (c_sid, sizeof (c_sid), "%u", sid);
          g_snprintf (c_en, sizeof (c_en), "%d", entered);
          g_snprintf (c_in, sizeof (c_in), "%d", inferred);
          g_snprintf (c_de, sizeof (c_de), "%d", delivered);
          g_snprintf (c_wi, sizeof (c_wi), "%d", wait_infer);
          g_snprintf (c_wo, sizeof (c_wo), "%d", wait_out);
          g_snprintf (c_bf, sizeof (c_bf), "%d", buffered);
          g_snprintf (c_dr, sizeof (c_dr), "%d", dropped);
          modelmux_grid_add (rows, 11, c_sid, nm ? nm : "-",
              rr == MM_ROLE_PRIMARY ? "Primary" : "Shadow",
              c_en, c_in, c_de, c_wi, c_wo, c_bf, c_dr, status);
        }
      }
    }

    /* FINAL-sample reconciliation: the 2-sample DROP debounce above needs a
     * SECOND elevated sample to report a loss -- but a stream whose only
     * elevated sample is its LAST (the loss happened right before detach /
     * teardown) never produces one, so a real one-time loss would be hidden
     * forever. A detached stream is gone from modelmux_bin->stream_wiring, yet its acct slot
     * persists (it is only reset when the source_id re-attaches), so scan the
     * slots of NON-live streams here: a pending excursion (over > 0) whose
     * frozen counters still show dropped > 0 IS the loss -- report it now.
     * A transient excursion drains at teardown (delivered catches up ->
     * dropped <= 0) and stays silent, so live-stream debounce behavior is
     * unchanged. Reported (or resolved) slots clear `over` so each final
     * verdict is emitted once. */
    {
      guint fsid;
      gint frr;
      for (fsid = 0; fsid < MM_ACCT_MAX; fsid++) {
        if (g_hash_table_contains (modelmux_bin->stream_wiring, GINT_TO_POINTER ((gint) fsid)))
          continue;
        for (frr = 0; frr < MM_ROLE_SLOTS; frr++) {
          ModelMuxAcct *fa = &modelmux_bin->acct[fsid];
          ModelMuxAcctRole *far = &fa->role[frr];
          gint entered, inferred, delivered, buffered, dropped;
          if (far->over <= 0 || far->warm < 8)
            continue;           /* nothing pending / never left warmup */
          entered = g_atomic_int_get (&fa->in_count) - far->in_base;
          inferred = g_atomic_int_get (&far->infer) - far->infer_base;
          delivered = g_atomic_int_get (&far->out) - far->out_base;
          buffered = entered - delivered;
          dropped = buffered - (far->floor + 4);
          far->over = 0;        /* final verdict below -- never re-flag */
          if (dropped <= 0)
            continue;           /* excursion drained at teardown -> no loss */
          {
            gchar c_sid[16], c_en[16], c_in[16], c_de[16], c_bf[16], c_dr[16];
            g_snprintf (c_sid, sizeof (c_sid), "%u", fsid);
            g_snprintf (c_en, sizeof (c_en), "%d", entered);
            g_snprintf (c_in, sizeof (c_in), "%d", inferred);
            g_snprintf (c_de, sizeof (c_de), "%d", delivered);
            g_snprintf (c_bf, sizeof (c_bf), "%d", buffered);
            g_snprintf (c_dr, sizeof (c_dr), "%d", dropped);
            modelmux_grid_add (rows, 11, c_sid, "(detached)",
                frr == MM_ROLE_PRIMARY ? "Primary" : "Shadow",
                c_en, c_in, c_de, "-", "-", c_bf, c_dr, "DROP");
            GST_WARNING ("FRAMEACCT stream %u role %s: %d frame(s) lost in its "
                "FINAL sample window (excursion pending at detach; no second "
                "sample will ever come) -> DROP", fsid,
                frr == MM_ROLE_PRIMARY ? "Primary" : "Shadow", dropped);
          }
        }
      }
    }
    modelmux_grid (s, "frameacct_table  —  per-(stream,role)   "
        "(buffered = in-flight, NOT loss · dropped = real loss → 0 = none)",
        11, hdr, rows);
  }

  if (modelmux_bin->config && modelmux_bin->config->attach_perf_metric) {
    {
      const gchar *mhdr[7] = { "role", "model", "gie-id", "gpu(s)", "fps",
        "infer_latency_ms", "gpu_util%" };
      g_string_append_c (s, '\n');
      g_ptr_array_set_size (rows, 0);
      modelmux_dump_pool_perf (modelmux_bin->primary_pool, rows);
      modelmux_dump_pool_perf (modelmux_bin->shadow_pool, rows);
      modelmux_grid (s, "model_perf_table  —  per-model throughput + inference latency "
          "(fps summed over shards · gpu_util = WHOLE device via NVML · "
          "attach-perf-metric=1)", 7, mhdr, rows);
    }
    {
      const gchar *hdr[7] = { "stream", "source name", "role", "model",
        "gpu", "fps", "infer_latency_ms" };
      g_string_append_c (s, '\n');
      g_ptr_array_set_size (rows, 0);
      g_hash_table_iter_init (&it, modelmux_bin->stream_wiring);
      while (g_hash_table_iter_next (&it, &k, &v)) {
        ModelMuxStreamEntry *e = (ModelMuxStreamEntry *) v;
        const gchar *nm = (const gchar *) g_hash_table_lookup (modelmux_bin->stream_names,
            GINT_TO_POINTER ((gint) e->stream_id));
        gint rr;
        for (rr = 0; rr < MM_ROLE_SLOTS; rr++) {
          ModelMuxRoleAttach *ra = (rr == MM_ROLE_PRIMARY) ? &e->prim : &e->shad;
          ModelPool *pool = (rr == MM_ROLE_PRIMARY) ? modelmux_bin->primary_pool : modelmux_bin->shadow_pool;
          ModelBin *bin = NULL;
          gdouble f = 0, l = 0;
          gchar *mbare = NULL, *mver = NULL;
          gchar c_sid[16], c_fps[24], c_lat[24], mcell[80], c_gpu[16];
          if (!ra->active)
            continue;
          g_strlcpy (c_gpu, "-", sizeof (c_gpu));   /* passthrough: no inference device */
          if (ra->passthru) {
            gdouble tf = (e->stream_id < MM_ACCT_MAX) ?
                modelmux_bin->thru_perf[e->stream_id].fps : 0.0;
            if (ra->pending_model)
              g_snprintf (mcell, sizeof (mcell), "%s (passthru)", ra->pending_model);
            else
              g_strlcpy (mcell, "(passthrough)", sizeof (mcell));
            g_snprintf (c_fps, sizeof (c_fps), "%.2f", tf);
            g_strlcpy (c_lat, "-", sizeof (c_lat));
          } else if (ra->model) {
            bin = modelmux_perf_find_bin_for_source (pool, e->stream_id);
            if (bin && bin->perf)
              modelmux_perf_get_source (bin->perf, (gint) e->stream_id, &f, &l);
            if (bin)                     /* the SERVING shard's device */
              g_snprintf (c_gpu, sizeof (c_gpu), "%d",
                  modelmux_dbg_shard_gpu (modelmux_bin->config, bin));
            model_key_split (ra->model, &mbare, &mver);
            g_snprintf (mcell, sizeof (mcell), "%s@%s", mbare ? mbare : "-",
                mver ? mver : "-");
            g_snprintf (c_fps, sizeof (c_fps), "%.2f", f);
            g_snprintf (c_lat, sizeof (c_lat), "%.2f", l);
          } else {
            continue;
          }
          g_snprintf (c_sid, sizeof (c_sid), "%u", e->stream_id);
          modelmux_grid_add (rows, 7, c_sid, nm ? nm : "-",
              rr == MM_ROLE_PRIMARY ? "Primary" : "Shadow", mcell, c_gpu,
              c_fps, c_lat);
          g_free (mbare);
          g_free (mver);
        }
      }
      modelmux_grid (s, "perf_table  —  per-(stream,role) fps + inference latency "
          "(last window · attach-perf-metric=1)", 7, hdr, rows);
    }
  }

  g_ptr_array_free (rows, TRUE);
  g_mutex_unlock (&modelmux_bin->lock);

  MM_INFO ("%s", s->str);
  g_string_free (s, TRUE);
  return G_SOURCE_REMOVE;
}

G_GNUC_INTERNAL void
modelmux_schedule_dump (ModelMuxBin * modelmux_bin, guint delay_seconds)
{
  if (!modelmux_bin || !modelmux_bin->log_enabled)
    return;
  g_mutex_lock (&modelmux_bin->free_lock);
  if (modelmux_bin->dump_source_id || g_atomic_int_get (&modelmux_bin->shutting_down)) {
    g_mutex_unlock (&modelmux_bin->free_lock);
    return;
  }
  modelmux_bin->dump_source_id = delay_seconds ?
      g_timeout_add_seconds (delay_seconds, modelmux_dump_state_idle, modelmux_bin) :
      g_idle_add (modelmux_dump_state_idle, modelmux_bin);
  g_mutex_unlock (&modelmux_bin->free_lock);
}
