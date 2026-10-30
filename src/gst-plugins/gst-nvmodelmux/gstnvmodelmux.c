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

/**
 * gstnvmodelmux.c
 * ========================
 * GObject element shell for the nvmodelmux plugin.
 *
 * This file is deliberately THIN: it owns no inference logic. It only
 *   (1) exposes configuration as GObject properties,
 *   (2) materialises the reusable ModelMuxBin (gstnvmodelmux_bin.c)
 *       lazily at the NULL->READY transition, wiring its ghost pads to ours, and
 *   (3) exposes runtime control as ACTION SIGNALS that forward to the bin's API.
 *
 * Keeping the shell separate from the bin keeps both reusable: the bin can be
 * embedded directly (as the sample app does) and the element can wrap any future
 * bin revision unchanged.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include "gstnvmodelmux.h"

/* The plugin's debug category. DEFINED here (one translation unit); bin.c and
 * config.c reference it via GST_DEBUG_CATEGORY_EXTERN, and plugin_init registers
 * it with GST_DEBUG_CATEGORY_INIT. All logging obeys GST_DEBUG. */
GST_DEBUG_CATEGORY (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat

/* Sentinel source-id for the "update-routing" signal meaning "all streams". */
#define MM_ALL_STREAMS  G_MAXUINT

/* ------------------------------------------------------------------ */
/* Pad templates                                                       */
/*                                                                     */
/* One batched NVMM input, one batched NVMM output -- the element is a  */
/* drop-in replacement for an nvinfer in a batched DeepStream graph.    */
/* ------------------------------------------------------------------ */
#define MM_PAD_CAPS "video/x-raw(memory:NVMM)"

static GstStaticPadTemplate gst_modelmux_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (MM_PAD_CAPS));

static GstStaticPadTemplate gst_modelmux_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (MM_PAD_CAPS));

#define DEFAULT_ENABLE                 TRUE
#define DEFAULT_BATCH_SIZE             1
#define DEFAULT_PER_MODEL_BATCH_SIZE   0      /* 0 => derive from batch-size */
#define DEFAULT_UNIFIED_BATCH          TRUE
/* on by default: when a model goes idle as a lone redundant warm copy, move it
 * into the shared limbo pool so either role reuses it without a reload. */
#define DEFAULT_COMPACT_IDLE_MODELS_ACROSS_POOLS  TRUE
#define DEFAULT_MODEL_SHARDING          TRUE

/* process-mode: values match nvinfer's (1 = primary/full-frame, 2 = secondary/
 * objects) so deepstream-app configures this element correctly with no change to
 * its call site -- it already sets process-mode=2 on whatever it builds at a
 * [secondary-gieN] section. The NICKS are pgie/sgie rather than nvinfer's
 * primary/secondary: this element already uses "primary" and "shadow" for the A/B
 * lanes, and reusing "primary" for a position would collide with that. */
#define DEFAULT_PROCESS_MODE  MM_PROCESS_MODE_PGIE

#define GST_TYPE_MODELMUX_PROCESS_MODE (gst_modelmux_process_mode_get_type ())
static GType
gst_modelmux_process_mode_get_type (void)
{
  static GType t = 0;
  static const GEnumValue vals[] = {
    {MM_PROCESS_MODE_PGIE, "Primary (full frame)", "pgie"},
    {MM_PROCESS_MODE_SGIE, "Secondary (objects)", "sgie"},
    {0, NULL, NULL}
  };
  if (!t)
    t = g_enum_register_static ("GstNvModelMuxProcessMode", vals);
  return t;
}

/* ------------------------------------------------------------------ */
/* Action signals (runtime control plane)                              */
/* ------------------------------------------------------------------ */
enum
{
  SIGNAL_LOAD_MODEL,
  SIGNAL_UNLOAD_MODEL,
  SIGNAL_RELOAD_MODEL,
  SIGNAL_ATTACH_STREAM,
  SIGNAL_DETACH_STREAM,
  SIGNAL_UPDATE_ROUTING,
  SIGNAL_GET_STATUS,
  SIGNAL_LAST
};
static guint gst_modelmux_signals[SIGNAL_LAST] = { 0 };

#define gst_modelmux_parent_class parent_class
G_DEFINE_TYPE (GstNvModelMux, gst_modelmux, GST_TYPE_BIN);

/* ================================================================== */
/* Small helpers                                                       */
/* ================================================================== */

/* Replace a heap string property: free old, dup new (NULL/"" -> NULL). */
static void
set_str (gchar ** slot, const gchar * v)
{
  g_free (*slot);
  *slot = (v && *v) ? g_strdup (v) : NULL;
}

/* Normalise an empty string to NULL (the bin/config API treat NULL as "unset"). */
static inline const gchar *
null_if_empty (const gchar * s)
{
  return (s && *s) ? s : NULL;
}

/* ================================================================== */
/* Build / teardown of the live sub-graph                             */
/* ================================================================== */

/* ================================================================== */
/* Cross-stage template-name uniqueness                                */
/* ================================================================== */

/* Every model NAME declared in a [model-name-N] entry is the ADDRESS the control
 * plane routes by -- it is how model/load|update reaches the right stage with no
 * stage id in the payload. Two stages declaring the same name make that address
 * ambiguous: both claim the request, and the first one downstream silently wins.
 * There is no correct behaviour to fall back to, so a duplicate FAILS the state
 * change with both stage names in the message. */
static gboolean
gst_modelmux_names_collide (GstNvModelMux * self, GstNvModelMux * other)
{
  guint i, j;

  for (i = 0; i < self->config.num_models; i++) {
    for (j = 0; j < other->config.num_models; j++) {
      if (g_strcmp0 (self->config.models[i].name, other->config.models[j].name) != 0)
        continue;
      GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
          ("model name '%s' is declared by two stages ('%s' and '%s')",
              self->config.models[i].name, GST_ELEMENT_NAME (self),
              GST_ELEMENT_NAME (other)),
          ("a [model-name-N] name is the control-plane ADDRESS of a model and must be "
           "unique across the pipeline -- rename it in one of the two configs"));
      return TRUE;
    }
  }
  return FALSE;
}

/* Effective gie ids a stage occupies: each model that DECLARES one takes that id, plus
 * id+MM_SHADOW_GIE_OFFSET at a primary (its A/B twin). Only meaningful when the stage
 * actually adopted the configured ids; a stage on the allocator occupies internal
 * ranges that cannot clash with a declared value. Returns the count written to @out. */
static guint
gst_modelmux_effective_gie_ids (GstNvModelMux * self, gint * out, guint max)
{
  guint i, n = 0;

  if (!self->config.gie_ids_from_config)
    return 0;
  for (i = 0; i < self->config.num_models && n + 2 <= max; i++) {
    gint d = modelmux_config_file_gie_id (self->config.models[i].config_file);
    if (d <= 0)
      continue;                                   /* allocated, not declared */
    out[n++] = d;
    if (self->config.process_mode != MM_PROCESS_MODE_SGIE)
      out[n++] = d + MM_SHADOW_GIE_OFFSET;        /* the shadow lane */
  }
  return n;
}

/* Cross-stage gie-id overlap. Two stages declaring the same id both stamp that identity
 * into the meta, which makes provenance and any downstream `operate-on-gie-id` ambiguous.
 *
 * WARN, not error. Unlike a duplicate model NAME -- where the control plane genuinely has
 * no correct stage to route to -- a duplicate id still produces a working pipeline, just
 * one whose attribution is ambiguous; and the ids live in per-model infer configs that
 * are routinely copied from stock samples, so failing here would break configs that run
 * fine today. Reassigning silently is worse than saying so: an id is only useful because
 * it is the value the operator wrote down. */
static void
gst_modelmux_warn_gie_id_overlap (GstNvModelMux * self, GstNvModelMux * other)
{
  gint mine[MM_MAX_MODELS * 2], theirs[MM_MAX_MODELS * 2];
  guint nm, nt, i, j;

  nm = gst_modelmux_effective_gie_ids (self, mine, G_N_ELEMENTS (mine));
  nt = gst_modelmux_effective_gie_ids (other, theirs, G_N_ELEMENTS (theirs));
  for (i = 0; i < nm; i++)
    for (j = 0; j < nt; j++)
      if (mine[i] == theirs[j]) {
        GST_ELEMENT_WARNING (self, RESOURCE, SETTINGS,
            ("gie-unique-id %d is used by two stages ('%s' and '%s')", mine[i],
                GST_ELEMENT_NAME (self), GST_ELEMENT_NAME (other)),
            ("both stamp that id into the object meta, so downstream provenance and "
             "'operate-on-gie-id' cannot tell them apart. Give one of them a different "
             "gie-unique-id in its model's infer config (a primary also occupies "
             "<id>+%d for its A/B shadow lane)", MM_SHADOW_GIE_OFFSET));
        return;                                   /* one warning is enough */
      }
}

/* ================================================================== */
/* Peer-stage traversal                                                */
/*                                                                     */
/* Every cross-stage question -- "does another stage declare this      */
/* name?", "which stage hosts this model?", "what does each stage      */
/* report?" -- is answered by walking the PIPELINE, never by following */
/* pads.                                                               */
/*                                                                     */
/* Pads describe the DATA topology, and the two are not the same       */
/* graph. deepstream-app hangs every secondary whose operate-on-gie-id */
/* names the primary off one shared tee, each branch ending in its own */
/* fakesink (deepstream_secondary_gie_bin.c:397,482,331). Stages are   */
/* therefore SIBLINGS in parallel, not a chain -- and a downstream     */
/* query forwarded across a tee stops at the first branch that answers */
/* (gst_pad_query_default forwards until one returns TRUE), so every   */
/* stage after the first is invisible. Two secondaries on the same     */
/* detector is the ordinary shape for "classify and embed the same     */
/* objects", so that is the normal case, not a corner.                 */
/*                                                                     */
/* Walking the pipeline finds them regardless of how they are wired.   */
/* ================================================================== */

/* Call @fn for every OTHER nvmodelmux in this element's pipeline. @fn returns FALSE
 * to stop the walk early. Safe when the element has no parent yet (no peers). */
G_GNUC_INTERNAL void
modelmux_foreach_peer_stage (GstNvModelMux * self, ModelMuxPeerStageFunc fn,
    gpointer udata)
{
  GstObject *top = GST_OBJECT (self), *p;
  GstIterator *it;
  GValue v = G_VALUE_INIT;
  gboolean done = FALSE;

  if (!self || !fn)
    return;

  while ((p = gst_object_get_parent (top))) {
    if (top != GST_OBJECT (self))
      gst_object_unref (top);
    top = p;
  }
  if (top == GST_OBJECT (self) || !GST_IS_BIN (top)) {
    if (top != GST_OBJECT (self))
      gst_object_unref (top);
    return;                             /* not in a bin: no peers to visit */
  }

  it = gst_bin_iterate_recurse (GST_BIN (top));
  while (!done) {
    switch (gst_iterator_next (it, &v)) {
      case GST_ITERATOR_OK:{
        GstElement *e = GST_ELEMENT (g_value_get_object (&v));
        if (e != GST_ELEMENT (self) && GST_IS_NVMODELMUX (e)
            && !fn (self, GST_NVMODELMUX (e), udata))
          done = TRUE;
        g_value_reset (&v);
        break;
      }
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync (it);
        break;
      default:
        done = TRUE;
        break;
    }
  }
  g_value_unset (&v);
  gst_iterator_free (it);
  gst_object_unref (top);
}

/* Marker field on a DELEGATED query copy. A stage that sees it answers for itself and
 * does not walk its own peers -- exactly one level of delegation, so the fan-out is
 * O(stages) and cannot cycle no matter how the pipeline is wired. */
#define MM_QUERY_DELEGATED_FIELD "nvmodelmux-delegated"

gboolean
modelmux_stage_query_is_delegated (GstQuery * q)
{
  const GstStructure *s = q ? gst_query_get_structure (q) : NULL;
  gboolean v = FALSE;

  return s && gst_structure_get_boolean (s, MM_QUERY_DELEGATED_FIELD, &v) && v;
}

gboolean
modelmux_stage_ask_peer (GstNvModelMux * peer, GstQuery * q, GstQuery ** out)
{
  GstQuery *fwd;
  gboolean answered;

  *out = NULL;
  if (!peer || !peer->sinkpad || !q)
    return FALSE;

  /* a COPY, marked: the original keeps whatever verdict the caller already has, and
   * the peer cannot re-enter the fan-out through it. */
  fwd = gst_query_copy (q);
  gst_structure_set (gst_query_writable_structure (fwd),
      MM_QUERY_DELEGATED_FIELD, G_TYPE_BOOLEAN, TRUE, NULL);

  /* Straight at the peer's OWN sink pad. This is the same delivery the pad-following
   * version used -- a downstream query on a sink pad runs its QUERY_DOWNSTREAM probes
   * -- only addressed to the stage we mean instead of to whatever the data path
   * happens to lead to. */
  answered = gst_pad_query (peer->sinkpad, fwd);
  if (!answered) {
    gst_query_unref (fwd);
    return FALSE;
  }
  *out = fwd;                           /* transfer: caller unrefs */
  return TRUE;
}

/* Visitor for the config-time conflict check. @udata is the gboolean verdict. */
static gboolean
gst_modelmux_conflict_visit (GstNvModelMux * self, GstNvModelMux * peer,
    gpointer udata)
{
  gboolean *ok = (gboolean *) udata;

  if (gst_modelmux_names_collide (self, peer)) {
    *ok = FALSE;
    return FALSE;                       /* fatal: stop the walk */
  }
  gst_modelmux_warn_gie_id_overlap (self, peer);
  return TRUE;
}

/* Visitor for overlay-slot assignment: count the secondary stages whose element NAME
 * sorts before mine. @udata is the running count. */
static gboolean
gst_modelmux_slot_visit (GstNvModelMux * self, GstNvModelMux * peer, gpointer udata)
{
  guint *before = (guint *) udata;

  if (peer->config.process_mode == MM_PROCESS_MODE_SGIE
      && g_strcmp0 (GST_ELEMENT_NAME (peer), GST_ELEMENT_NAME (self)) < 0)
    (*before)++;
  return TRUE;
}

/* Give this stage its line in the MM_OVERLAY stack.
 *
 * Ordered by ELEMENT NAME rather than by the order the walk happens to visit stages,
 * so the same pipeline always puts the same stage on the same line -- a slot that
 * moved between runs would be worse than no slot at all. Runs at READY->PAUSED
 * because that is when every stage has parsed its config and can be recognised as a
 * secondary at all. */
static void
gst_modelmux_assign_overlay_slot (GstNvModelMux * self)
{
  guint before = 0;

  if (!self->modelmux_sgie_bin)
    return;                             /* the primary owns the top line */
  modelmux_foreach_peer_stage (self, gst_modelmux_slot_visit, &before);
  modelmux_sgie_set_overlay_slot (self->modelmux_sgie_bin, before);
}

/* Cross-stage checks, run at READY->PAUSED: by then every nvmodelmux in the pipeline has
 * completed NULL->READY, so all catalogs are parsed. models[] is fixed after parse (only
 * versions[] mutates at runtime), so reading a sibling's catalog needs no lock.
 *   duplicate model NAME -> hard error (the control plane has no correct stage to pick)
 *   overlapping gie id   -> warning  (the pipeline still works; attribution is ambiguous)
 * Returns FALSE only for the fatal case. */
static gboolean
gst_modelmux_check_stage_conflicts (GstNvModelMux * self)
{
  /* the shared walk: a name collision is fatal, an id overlap is only a warning, so
   * the visitor stops on the first collision and carries the verdict out. */
  gboolean ok = TRUE;

  modelmux_foreach_peer_stage (self, gst_modelmux_conflict_visit, &ok);
  return ok;
}

/* SECONDARY stage (process-mode=2): build the linear chain instead of the full
 * ModelMuxBin. One model serves all streams on the already-batched buffer, so there is
 * nothing to demux/re-batch -- see gstnvmodelmux_sgie.c. */
static gboolean
gst_modelmux_build_sgie (GstNvModelMux * self, const ModelMuxConfig * config)
{
  self->modelmux_sgie_bin = modelmux_sgie_new (GST_ELEMENT (self), GST_ELEMENT_NAME (self), config);
  if (!self->modelmux_sgie_bin) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("failed to create the secondary inference chain"), (NULL));
    return FALSE;
  }

  gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad),
      modelmux_sgie_get_sink (self->modelmux_sgie_bin));
  gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad),
      modelmux_sgie_get_src (self->modelmux_sgie_bin));

  /* the control plane is stage-agnostic: it addresses models by their globally unique
   * [model-name-N] template, so a secondary answers only for the model it hosts. */
  gst_modelmux_control_attach (self);

  GST_INFO_OBJECT (self, "built (secondary): model='%s@%s' batch=%u",
      modelmux_sgie_model_name (self->modelmux_sgie_bin),
      modelmux_sgie_model_version (self->modelmux_sgie_bin), config->batch_size);
  self->built = TRUE;
  return TRUE;
}

/* Assemble @self->config from the config-file (if any) overlaid with the
 * individual GObject properties (properties always win), then materialise the
 * sub-graph: either the full ModelMuxBin or, when disabled, a transparent
 * identity passthrough. Called once at NULL->READY. Returns FALSE on hard error
 * (already signalled via GST_ELEMENT_ERROR). */
static gboolean
gst_modelmux_build (GstNvModelMux * self)
{

  /* If the sub-graph is already built, return TRUE. */
  if (self->built)
    return TRUE;

  ModelMuxConfig *config = &self->config;

  /* Parse the config-file into self->config and reconcile with the GObject properties. */
  if (!gst_modelmux_parse_config_file (self, self->config_file_path))
    return FALSE;

  /* DISABLED: skip inference entirely and act as a transparent passthrough
   * (wire sink -> identity -> src). A mid-pipeline element must still forward buffers, so
   * a lightweight 'identity' provides the data path with correct caps/event/query handling.
   * Named "modelmux-passthrough" so it is identifiable in logs / gst-inspect graphs. */
  if (!self->enable) {
    GstPad *p;
    GST_INFO_OBJECT (self, "enable=0: bypassing inference (passthrough mode)");
    self->passthru = gst_element_factory_make ("identity", "modelmux-passthrough");
    if (!self->passthru) {
      GST_ELEMENT_ERROR (self, CORE, MISSING_PLUGIN,
          ("failed to create 'identity' for passthrough"), (NULL));
      return FALSE;
    }
    g_object_set (self->passthru, "silent", TRUE, NULL);
    gst_bin_add (GST_BIN (self), self->passthru);
    p = gst_element_get_static_pad (self->passthru, "sink");
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad), p);
    gst_object_unref (p);
    p = gst_element_get_static_pad (self->passthru, "src");
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad), p);
    gst_object_unref (p);
    self->built = TRUE;
    return TRUE;
  }

  /* SECONDARY stage: a much simpler graph -- fork here so the primary path below stays
   * untouched. */
  if (config->process_mode == MM_PROCESS_MODE_SGIE)
    return gst_modelmux_build_sgie (self, config);

  /* Default primary model is OPTIONAL. When none is configured, no model is loaded at 
  init (zero VRAM) and any stream that resolves to "no model" -- because it requested
  nothing and there is no default, or it requested a model that does not exist and there
  is no default -- is PASSED THROUGH the display mux WITHOUT inference (frames flow, no
  detections). A user can then load a custom model and attach streams to it on demand. */
  /* pre-bin: the pool refs don't exist yet, so the PARSED seed value is the
   * only (and correct) source here. Post-bin, defaults are read from the pools. */
  if (!config->default_primary)
    GST_INFO_OBJECT (self, "no default primary model configured -- streams with no resolvable "
        "model will PASS THROUGH without inference (set 'primary-model' to preload a default)");

  /* Build the reusable inference bin (demux -> per-stream tees -> model pools -> combined mux).
   * Passing `self` as the parent makes modelmux_bin_new() add the bin's internal GstBin as a
   * CHILD of this element, so the element owns nothing internal -- it only bridges its ghost
   * pads to the bin's boundary pads below. Only the empty skeleton is built here; models are
   * instantiated later (defaults at PLAYING, per-stream lanes on attach). */
  ModelMuxBin *modelmux_bin = modelmux_bin_new (GST_ELEMENT (self),
      GST_ELEMENT_NAME (self), config->batch_size, config);
  if (!modelmux_bin) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("failed to create the multi-model inference bin"), (NULL));
    return FALSE;
  }

  /* Point our ghost pads at the bin's boundary pads. */
  gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad),
      modelmux_bin_get_sink (modelmux_bin));
  gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad),
      modelmux_bin_get_src (modelmux_bin));

  /* Wire callbacks, THEN publish the bin atomically -- both under api_control_lock, so no
   * other thread ever sees a half-wired bin (control handlers lock; probes read lock-free). */
  g_mutex_lock (&self->api_control_lock);
  gst_modelmux_control_wire_ota_rollback (self, modelmux_bin);
  g_atomic_pointer_set (&self->modelmux_bin, modelmux_bin);
  g_mutex_unlock (&self->api_control_lock);

  /* Subscribe to the pipeline bus + install the data-plane auto-attach and the
   * GET/control query probes (the REST stream/model API is handled INTERNALLY).
   * AFTER the publish above: the query probes read self->modelmux_bin, so attaching
   * first would leave a window answering the empty shape for a bin that exists. */
  gst_modelmux_control_attach (self);

  {
    const DefaultModelRef *bdp = modelmux_bin_get_default (modelmux_bin, FALSE);
    const DefaultModelRef *bds = modelmux_bin_get_default (modelmux_bin, TRUE);
    GST_INFO_OBJECT (self, "built: batch=%u unified=%d primary='%s' shadow='%s'",
        config->batch_size, config->unified_batch,
        (bdp && bdp->name) ? bdp->name : "(none)",
        (bds && bds->name) ? bds->name : "(none)");
  }

  /* Set the built flag to TRUE. */
  self->built = TRUE;

  return TRUE;
}

/* Tear the sub-graph down at READY->NULL: detach ghost targets, free the bin
 * (which removes its child GstBin from us) or the passthrough, and release the
 * assembled config so a subsequent NULL->READY rebuilds cleanly.
 * Takes api_control_lock for the mm swap + free, so no action-signal handler or queued
 * control dispatch can be mid-use of self->modelmux_bin when it is freed (the handlers
 * hold api_control_lock across their whole mm use). */
static void
gst_modelmux_teardown (GstNvModelMux * self)
{
  ModelMuxBin *mm;

  g_mutex_lock (&self->api_control_lock);
  mm = self->modelmux_bin;
  /* atomic (paired with the probes' lock-free advisory NULL-checks); queued
   * control idles must stop before free */
  g_atomic_pointer_set (&self->modelmux_bin, NULL);

  /* stop the REST control plane first (disconnect bus, drop pending reroutes) */
  gst_modelmux_control_detach (self);

  if (self->sinkpad)
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->sinkpad), NULL);
  if (self->srcpad)
    gst_ghost_pad_set_target (GST_GHOST_PAD (self->srcpad), NULL);

  if (mm)
    modelmux_bin_free (mm);  /* removes its child bin from us */
  if (self->modelmux_sgie_bin) {
    modelmux_sgie_free (self->modelmux_sgie_bin);   /* removes its child bin from us */
    self->modelmux_sgie_bin = NULL;
  }
  if (self->passthru) {
    gst_element_set_state (self->passthru, GST_STATE_NULL);
    gst_bin_remove (GST_BIN (self), self->passthru);
    self->passthru = NULL;
  }

  modelmux_config_clear (&self->config);
  memset (&self->config, 0, sizeof (self->config));
  self->built = FALSE;
  self->initial_default_load_started = FALSE;
  g_mutex_unlock (&self->api_control_lock);
}

/* ================================================================== */
/* Shared routing revision                                             */
/*                                                                     */
/* routing_revision is the if_revision CAS token, so a pipeline must    */
/* have exactly ONE of them however many nvmodelmux stages it holds.    */
/* The stages find the same counter through a GstContext -- GStreamer's */
/* own facility for sharing a live object between element instances --  */
/* rather than through a global or a hand-rolled registry, so it is     */
/* scoped to the pipeline, needs no teardown ordering, and lets an      */
/* application inject its own counter if it ever wants to.              */
/* ================================================================== */

/* Hand the negotiated counter to whichever bin this stage built. Safe to call
 * before the build (both pointers NULL) -- the build calls it again afterwards. */
static void
gst_modelmux_publish_rev_shared (GstNvModelMux * self)
{
  if (!self->rev_shared)
    return;
  if (self->modelmux_bin)
    modelmux_bin_set_rev_shared (self->modelmux_bin, self->rev_shared);
  if (self->modelmux_sgie_bin)
    modelmux_sgie_set_rev_shared (self->modelmux_sgie_bin, self->rev_shared);
}

/*
 * Adopt a routing context published by another stage (or by the application).
 *
 * GstBin delivers this synchronously: when any stage posts have-context the
 * pipeline caches it and sets it on every child, and when a stage posts
 * need-context the pipeline answers straight from that cache. So whichever stage
 * comes up first creates the counter and the rest simply receive it.
 *
 * FIRST one wins: swapping counters mid-run would move the revision under a
 * controller that is holding it, which is precisely the failure the shared
 * counter exists to prevent.
 */
static void
gst_modelmux_set_context (GstElement * element, GstContext * context)
{
  GstNvModelMux *self = GST_NVMODELMUX (element);

  if (context && !self->rev_shared
      && g_strcmp0 (gst_context_get_context_type (context),
          MODELMUX_CONTEXT_TYPE_ROUTING) == 0) {
    const GstStructure *s = gst_context_get_structure (context);
    const GValue *v = s ? gst_structure_get_value (s,
        MODELMUX_CONTEXT_FIELD_REV) : NULL;

    if (v && G_VALUE_HOLDS (v, MODELMUX_TYPE_REV_SHARED)) {
      /* g_value_get_boxed BORROWS -- take our own ref, the context may go away */
      self->rev_shared = modelmux_rev_shared_ref (g_value_get_boxed (v));
      GST_DEBUG_OBJECT (self, "adopted the pipeline's shared routing revision "
          "(currently %" G_GUINT64_FORMAT ")",
          modelmux_rev_shared_get (self->rev_shared));
      gst_modelmux_publish_rev_shared (self);
    } else {
      GST_WARNING_OBJECT (self, "a '%s' context arrived without a usable '%s' "
          "field -- this stage will count its own revision instead",
          MODELMUX_CONTEXT_TYPE_ROUTING, MODELMUX_CONTEXT_FIELD_REV);
    }
  }

  /* ALWAYS chain up, for every context type: GstBin caches contexts and forwards
   * them to our children (the model bins, nvinfer, ...), and skipping that would
   * break context types we know nothing about. */
  GST_ELEMENT_CLASS (parent_class)->set_context (element, context);
}

/* Make sure this stage has the pipeline's routing counter, creating and
 * publishing one if we are the first stage up. Called at NULL->READY, before the
 * sub-graph is built, so the bins are handed the counter as they are created. */
static void
gst_modelmux_ensure_rev_shared (GstNvModelMux * self)
{
  GstContext *ctx;
  GstStructure *s;

  if (self->rev_shared)
    return;

  /* 1. Ask. The pipeline answers from its context cache -- synchronously, from
   *    inside this post -- if another stage has already published one, which is
   *    what makes this work regardless of which stage reaches READY first. An
   *    application may also answer it with a counter of its own. */
  gst_element_post_message (GST_ELEMENT (self),
      gst_message_new_need_context (GST_OBJECT (self),
          MODELMUX_CONTEXT_TYPE_ROUTING));
  if (self->rev_shared)                 /* set_context ran during the post */
    return;

  /* 2. Nobody has one: we are the first stage. Create it and publish, so the
   *    pipeline caches it for every stage that comes up after us. */
  self->rev_shared = modelmux_rev_shared_new ();
  ctx = gst_context_new (MODELMUX_CONTEXT_TYPE_ROUTING, TRUE /* persistent */);
  s = gst_context_writable_structure (ctx);
  /* boxed: the structure takes its own ref, so the counter survives us */
  gst_structure_set (s, MODELMUX_CONTEXT_FIELD_REV, MODELMUX_TYPE_REV_SHARED,
      self->rev_shared, NULL);
  gst_element_set_context (GST_ELEMENT (self), ctx);   /* cache it on ourselves */
  GST_DEBUG_OBJECT (self, "published the pipeline's shared routing revision");
  /* transfer-full: the message takes @ctx */
  gst_element_post_message (GST_ELEMENT (self),
      gst_message_new_have_context (GST_OBJECT (self), ctx));
}

/*
 * Element lifecycle hook: build/warm/tear down the internal inference bin in step
 * with the pipeline state.
 *   NULL->READY     : build the sub-graph (gst_modelmux_build); on failure, tear
 *                     down and fail the transition.
 *   PAUSED->PLAYING : kick off the one-time default-model warm-up.
 *   READY->NULL     : tear down the sub-graph and release the assembled config.
 * Build runs BEFORE chaining up to the parent GstBin (so the new children get the
 * state change); teardown runs AFTER (so children are already down first). A failed
 * parent chain-up on NULL->READY also tears down, so nothing leaks in NULL.
 */
static GstStateChangeReturn
gst_modelmux_change_state (GstElement * element, GstStateChange transition)
{
  GstNvModelMux *self = GST_NVMODELMUX (element);
  GstStateChangeReturn ret;

  /* Build BEFORE chaining up so GstBin propagates the upward state change into
   * the freshly-added children. */
  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    /* BEFORE the build: the bins are handed the counter as they are created, so
     * a control request that lands the instant we hit READY already reads the
     * pipeline's revision and not this stage's fallback. */
    gst_modelmux_ensure_rev_shared (self);
    if (!gst_modelmux_build (self)) {
      /* release any partially-assembled config / children so the leaked ModelMuxConfig
       * (parsed file + registered model defs) is freed and a subsequent NULL->READY
       * re-parses from a clean slate instead of double-registering. teardown is
       * NULL-safe for every field that build may not have reached. */
      gst_modelmux_teardown (self);
      return GST_STATE_CHANGE_FAILURE;
    }
    gst_modelmux_publish_rev_shared (self);   /* the bins exist now */
  }

  /* At READY->PAUSED every stage has parsed its catalog: check for cross-stage
   * conflicts (duplicate model name = fatal, overlapping gie id = warning). */
  if (transition == GST_STATE_CHANGE_READY_TO_PAUSED && self->built) {
    if (!gst_modelmux_check_stage_conflicts (self))
      return GST_STATE_CHANGE_FAILURE;
    gst_modelmux_assign_overlay_slot (self);
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    /* a NULL->READY that failed in the parent chain-up leaves the element in
     * NULL, so READY->NULL (and thus teardown) would never run: the built graph,
     * bus ref, control probes and timers would leak -- and finalize's defensive
     * teardown would later touch already-disposed children. Tear down now. */
    if (transition == GST_STATE_CHANGE_NULL_TO_READY && self->built)
      gst_modelmux_teardown (self);
    return ret;
  }

  /* Warm the default model(s) once the element (and its model bins) have reached
   * PLAYING -- sync_state_with_parent brings each model bin up and its nvinfer
   * deserializes/builds the engine, flipping it to READY.
   * Block until the configured defaults are WARMED so they infer on every frame
   * (mirrors nvinfer's engine build at the state transition); a default that FAILS
   * to warm fails the transition. Runtime model/load stays async. */
  if (transition == GST_STATE_CHANGE_PAUSED_TO_PLAYING
      && self->modelmux_bin && !self->initial_default_load_started) {
    self->initial_default_load_started = TRUE;
    if (!modelmux_bin_load_defaults (self->modelmux_bin)) {
      GST_ELEMENT_ERROR (self, LIBRARY, INIT,
          ("a configured default model failed to warm up"),
          ("the default engine could not be built/loaded -- see the nvinfer errors "
           "above and fix the model/engine"));
      return GST_STATE_CHANGE_FAILURE;
    }
  }

  /* Tear down AFTER chaining up so children are already in NULL. */
  if (transition == GST_STATE_CHANGE_READY_TO_NULL)
    gst_modelmux_teardown (self);

  return ret;
}

/* ================================================================== */
/* Action-signal handlers (forward to the bin's control API)           */
/*                                                                     */
/* All are no-ops with a clear GST_WARNING when the bin is not live     */
/* (disabled / before READY), never a crash.                           */
/* ================================================================== */

/* "load-model": register (name, config-file[, engine][, version]) and warm it
 * into the limbo pool. Returns ModelStatus (gint). */
static gint
gst_modelmux_load_model (GstNvModelMux * self, const gchar * name,
    const gchar * config_file, const gchar * engine_file, const gchar * version)
{
  /* api_control_lock across the WHOLE handler: signals run on the emitter's thread, so an
   * unlocked check-then-use of self->modelmux_bin races READY->NULL teardown (UAF), and the
   * self->config catalog mutation below races the main-loop REST handlers. Same
   * pattern in every action-signal handler. */
  g_mutex_lock (&self->api_control_lock);
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "load-model: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return MODEL_FAILED;
  }
  if (!null_if_empty (name)) {
    GST_WARNING_OBJECT (self, "load-model: empty model name");
    g_mutex_unlock (&self->api_control_lock);
    return MODEL_FAILED;
  }
  if (null_if_empty (config_file)) {
    /* catalog entry is name -> base config (engine/version are per-instance now). */
    modelmux_config_register_model (&self->config, name, config_file, NULL, NULL, 0);
    if (!modelmux_config_find_model (&self->config, name)) {
      /* registration REFUSED (reserved '@'/';' in the name, or registry full):
       * do NOT load anyway -- the warmed bin would be invisible to the catalog
       * (lazy-warm, version resolution, unload guards all consult it). */
      GST_WARNING_OBJECT (self, "load-model '%s': could not register in the "
          "catalog (invalid name or registry full) -- rejected", name);
      g_mutex_unlock (&self->api_control_lock);
      return MODEL_FAILED;
    }
  } else if (!modelmux_config_find_model (&self->config, name)) {
    GST_WARNING_OBJECT (self,
        "load-model '%s': not in catalog and no config-file supplied", name);
    g_mutex_unlock (&self->api_control_lock);
    return MODEL_FAILED;
  }
  {
    const ModelCatalogEntry *def = modelmux_config_find_model (&self->config, name);
    const gchar *basecfg = null_if_empty (config_file) ? config_file : (def ? def->config_file : NULL);
    gchar *key = model_key (name, null_if_empty (version) ? version : MM_MODEL_VERSION_DEFAULT);
    gchar *derived = NULL;
    const gchar *eff = null_if_empty (config_file);   /* custom config, or NULL => catalog base */
    gint r;
    if (null_if_empty (engine_file) && basecfg) {     /* nvinfer: swap engine into a derived config */
      derived = modelmux_config_derive (basecfg, engine_file, -1, -1, name,
          null_if_empty (version) ? version : MM_MODEL_VERSION_DEFAULT);   /* match the @1 instance key */
      if (!derived) {
        /* an EXPLICIT engine that failed to derive must FAIL the load: the base
         * config would run the wrong artifact while the caller believes the
         * override took (modelmux_config_derive already logged why). */
        GST_ERROR_OBJECT (self, "load-model '%s': engine override could not be "
            "derived from '%s' -- rejecting", name, basecfg);
        g_free (key);
        g_mutex_unlock (&self->api_control_lock);
        return MODEL_FAILED;
      }
      eff = derived;
    } else if (null_if_empty (engine_file) && !basecfg) {
      /* an engine override needs a base config to swap model-engine-file into -- without one
       * (model not in catalog and no config-file supplied) the engine is silently unusable. */
      GST_WARNING_OBJECT (self, "load-model '%s': engine-file given but no base config "
          "(not in catalog and no config-file supplied) -- ignoring the engine override", name);
    }
    r = (gint) modelmux_bin_load_model (self->modelmux_bin, key, eff, -1, 0);
    g_free (key);
    g_free (derived);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "unload-model": free an idle model by name (limbo or either pool). */
static gboolean
gst_modelmux_unload_model (GstNvModelMux * self, const gchar * name)
{
  gboolean r = FALSE;
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "unload-model: inference not active");
  } else if (null_if_empty (name)) {
    /* signal carries a bare name -> unload ALL non-serving versions of it. */
    r = modelmux_bin_unload_versions (self->modelmux_bin, name, NULL) > 0;
  }
  g_mutex_unlock (&self->api_control_lock);
  return r;
}

/* "reload-model": in-place OTA checkpoint swap on an already-loaded model.
 * engine -> nvinfer model-engine-file; else config-file -> full reload;
 * version updates provenance. */
static gboolean
gst_modelmux_reload_model (GstNvModelMux * self, const gchar * name,
    const gchar * config_file, const gchar * engine_file, const gchar * version)
{
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "reload-model: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  if (!null_if_empty (name)) {
    GST_WARNING_OBJECT (self, "reload-model: empty model name");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  /* OTA in-place reload is SUPERSEDED by versioned model/load (a new version = a new bin);
   * this signal is retained for compatibility and targets the 1 instance of the name. */
  {
    gchar *key = model_key (name, null_if_empty (version) ? version : MM_MODEL_VERSION_DEFAULT);
    gboolean r;
    modelmux_config_update_model (&self->config, name, null_if_empty (config_file),
        null_if_empty (engine_file), null_if_empty (version));
    r = modelmux_bin_reload_model (self->modelmux_bin, key, null_if_empty (config_file),
        null_if_empty (engine_file), null_if_empty (version));
    g_free (key);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "attach-stream": route one source_id to (primary[, shadow]); NULL roles fall
 * back to the configured / config-time-bound defaults. */
static gboolean
gst_modelmux_attach_stream (GstNvModelMux * self, guint source_id,
    const gchar * name, const gchar * primary, const gchar * shadow)
{
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "attach-stream: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  /* compose canonical keys (signal carries bare names -> version 1). */
  {
    gchar *pk = null_if_empty (primary) ? model_key (primary, MM_MODEL_VERSION_DEFAULT) : NULL;
    gchar *sk = null_if_empty (shadow) ? model_key (shadow, MM_MODEL_VERSION_DEFAULT) : NULL;
    /* The action-signal path carries ONE name. It is the display name, and this
     * entry point has no separate camera_id to offer -- pass it as the camera_id
     * too rather than NULL, since for a single-identifier host the two ARE the
     * same string, which is exactly what stream/add does when only one is given.
     * A stream attached this way stays routable by that name. */
    gboolean r = modelmux_bin_attach_stream (self->modelmux_bin, source_id, null_if_empty (name),
        null_if_empty (name), pk, sk, MM_GPU_ANY, MM_GPU_ANY,
        FALSE, FALSE /* signal names the models: pinned */);
    g_free (pk);
    g_free (sk);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "detach-stream": remove one source_id from the inference graph. */
static gboolean
gst_modelmux_detach_stream (GstNvModelMux * self, guint source_id)
{
  gboolean r = FALSE;
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin)
    GST_WARNING_OBJECT (self, "detach-stream: inference not active");
  else
    r = modelmux_bin_detach_stream (self->modelmux_bin, source_id);
  g_mutex_unlock (&self->api_control_lock);
  return r;
}

/* Parse a "name;version" ref (config-file convention) into a canonical
 * "name@version" pool key.  Accepts bare "name" (falls back to default
 * version) and "name;version".  Returns a newly-allocated string or NULL. */
static gchar *
routing_ref_to_key (const gchar * ref)
{
  if (!null_if_empty (ref))
    return NULL;
  gchar **parts = g_strsplit (ref, ";", 2);
  gchar  *key;
  if (!parts[0] || !*parts[0]) {        /* reject "name"-less refs like ";1" */
    g_strfreev (parts);
    return NULL;
  }
  key = model_key (parts[0],
      (parts[1] && *parts[1]) ? parts[1] : MM_MODEL_VERSION_DEFAULT);
  g_strfreev (parts);
  return key;
}

/* "update-routing": re-point a stream's primary/shadow at runtime (zero-drop).
 * source_id == G_MAXUINT applies to ALL active streams.
 * primary / shadow accept "name" or "name;version" (config-file convention). */
static gboolean
gst_modelmux_update_routing (GstNvModelMux * self, guint source_id,
    const gchar * primary, const gchar * shadow)
{
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin) {
    GST_WARNING_OBJECT (self, "update-routing: inference not active");
    g_mutex_unlock (&self->api_control_lock);
    return FALSE;
  }
  {
    gchar *pk = routing_ref_to_key (primary);
    gchar *sk = routing_ref_to_key (shadow);
    gboolean r;
    if (source_id == MM_ALL_STREAMS)
      r = modelmux_bin_update_routing_scoped (self->modelmux_bin, NULL, 0, pk, sk,
          MM_GPU_ANY, MM_GPU_ANY, FALSE, FALSE /* explicit routing: pinned */) > 0;
    else
      r = modelmux_bin_update_routing (self->modelmux_bin, source_id, pk, sk,
          MM_GPU_ANY, MM_GPU_ANY, FALSE, FALSE);
    g_free (pk);
    g_free (sk);
    g_mutex_unlock (&self->api_control_lock);
    return r;
  }
}

/* "get-status": schema-stable JSON dump. Independent filters: model_filter
 * narrows models[]; stream_filter / source_id (>=0) narrows streams[]. Empty
 * filters / source_id<0 => no narrowing. Returns a newly-allocated string. */
static gchar *
gst_modelmux_get_status (GstNvModelMux * self, const gchar * model_filter,
    const gchar * stream_filter, gint source_id)
{
  gchar *r;
  g_mutex_lock (&self->api_control_lock);       /* see gst_modelmux_load_model */
  if (!self->modelmux_bin)
    r = g_strdup ("{\"models\":[],\"streams\":[]}");
  else
    r = modelmux_bin_status_json (self->modelmux_bin, null_if_empty (model_filter),
        NULL /* model_version filter: REST status-query path only */,
        null_if_empty (stream_filter), source_id);
  g_mutex_unlock (&self->api_control_lock);
  return r;
}

/* Write a property value by id. */
static void
gst_modelmux_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  /* Mark the property as being set through g_object_set (nvinfer convention), so
   * gst_modelmux_parse_config_file lets it override the config-file value. */
  if (prop_id < PROP_LAST)
    self->is_prop_set[prop_id] = TRUE;

  switch (prop_id) {
    case PROP_ENABLE:
      self->enable = g_value_get_boolean (value);
      break;
    case PROP_CONFIG_FILE_PATH:
      set_str (&self->config_file_path, g_value_get_string (value));
      break;
    case PROP_BATCH_SIZE:
      self->batch_size = g_value_get_uint (value);
      break;
    case PROP_PER_MODEL_BATCH_SIZE:
      self->per_model_batch_size = g_value_get_uint (value);
      break;
    case PROP_UNIFIED_BATCH:
      self->unified_batch = g_value_get_boolean (value);
      break;
    case PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS:
      self->compact_idle_models_across_pools = g_value_get_boolean (value);
      break;
    case PROP_MODEL_SHARDING:
      self->model_sharding = g_value_get_boolean (value);
      break;
    case PROP_PROCESS_MODE:
      self->process_mode = g_value_get_enum (value);
      break;
    case PROP_PRIMARY_MODEL:
      set_str (&self->primary_model, g_value_get_string (value));
      break;
    case PROP_SHADOW_MODEL:
      set_str (&self->shadow_model, g_value_get_string (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Read a property value by id. */
static void
gst_modelmux_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  switch (prop_id) {
    case PROP_ENABLE:
      g_value_set_boolean (value, self->enable);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, self->config_file_path);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, self->batch_size);
      break;
    case PROP_PER_MODEL_BATCH_SIZE:
      g_value_set_uint (value, self->per_model_batch_size);
      break;
    case PROP_UNIFIED_BATCH:
      g_value_set_boolean (value, self->unified_batch);
      break;
    case PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS:
      g_value_set_boolean (value, self->compact_idle_models_across_pools);
      break;
    case PROP_MODEL_SHARDING:
      g_value_set_boolean (value, self->model_sharding);
      break;
    case PROP_PROCESS_MODE:
      g_value_set_enum (value, self->process_mode);
      break;
    case PROP_PRIMARY_MODEL:
      g_value_set_string (value, self->primary_model);
      break;
    case PROP_SHADOW_MODEL:
      g_value_set_string (value, self->shadow_model);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_modelmux_dispose (GObject * object)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  /* defensive: normally torn down at READY->NULL. MUST run in dispose (idempotent
   * via self->built), BEFORE chaining up: the parent GstBin dispose removes and
   * unrefs our pads and children, so running teardown any later (finalize) would
   * operate on already-disposed objects (ghost pads, passthru, the mm child bin). */
  if (self->built)
    gst_modelmux_teardown (self);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_modelmux_finalize (GObject * object)
{
  GstNvModelMux *self = GST_NVMODELMUX (object);

  g_free (self->config_file_path);
  g_free (self->primary_model);
  g_free (self->shadow_model);
  /* held for the element's whole life (not per state cycle) so a stop/start keeps
   * the pipeline's revision monotonic; the counter dies with the last stage. */
  modelmux_rev_shared_unref (self->rev_shared);
  self->rev_shared = NULL;
  g_mutex_clear (&self->source_state_lock);
  g_mutex_clear (&self->api_control_lock);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * One-time class initialisation. Runs ONCE when the GType is first registered and
 * configures the class VTABLE that is SHARED by every nvmodelmux instance (it is
 * NOT per-instance -- per-instance setup lives in gst_modelmux_init). Here we:
 *   - override the base-class vmethods (GObject set/get_property, dispose, finalize;
 *     GstElement change_state);
 *   - install the GObject properties (enable, config-file-path, batch-size, ...);
 *   - register the runtime-control ACTION signals (load/unload/reload-model,
 *     attach/detach-stream, update-routing, get-status);
 *   - add the (always) sink/src pad templates and set the element metadata.
 * No pads are created and no sub-graph is linked here -- that is instance work done
 * lazily at NULL->READY (gst_modelmux_build); this function only defines the class.
 */
static void
gst_modelmux_class_init (GstNvModelMuxClass * klass)
{
  /* GObjectClass and GstElementClass */
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  /* Override base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_modelmux_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_modelmux_get_property);
  gobject_class->dispose = GST_DEBUG_FUNCPTR (gst_modelmux_dispose);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_modelmux_finalize);
  element_class->change_state = GST_DEBUG_FUNCPTR (gst_modelmux_change_state);
  element_class->set_context = GST_DEBUG_FUNCPTR (gst_modelmux_set_context);

  /* properties that only take effect at (re)build are MUTABLE_READY */
  const GParamFlags rwc =
  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY;

  /* -------- properties -------- */
  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Run inference (TRUE) or act as a transparent passthrough (FALSE)",
          DEFAULT_ENABLE, rwc));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Config File Path",
          "Multimodel config file ([multimodel] defaults/behaviour + per-sensor "
          "[stream-model-<id>] bindings), parsed by the plugin. Individual "
          "properties override values from it.", NULL, rwc));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum number of concurrent streams (sizes every internal mux). "
          "0 = inherit from the config-file / default. Should match the upstream "
          "streammux batch.",
          0, MM_MAX_BATCH, 0, rwc));

  g_object_class_install_property (gobject_class, PROP_PER_MODEL_BATCH_SIZE,
      g_param_spec_uint ("per-model-batch-size", "Per-Model Batch Size",
          "nvinfer batch size for a single model instance (0 = use batch-size); "
          "streams beyond this auto-shard onto additional same-model instances",
          0, MM_MAX_BATCH, DEFAULT_PER_MODEL_BATCH_SIZE, rwc));

  g_object_class_install_property (gobject_class, PROP_UNIFIED_BATCH,
      g_param_spec_boolean ("unified-batch", "Unified Batch",
          "Combine primary + shadow outputs into ONE batched buffer (A/B view)",
          DEFAULT_UNIFIED_BATCH, rwc));

  g_object_class_install_property (gobject_class, PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS,
      g_param_spec_boolean ("compact-idle-models-across-pools",
          "Compact Idle Models Across Pools",
          "When a model goes idle as a lone redundant warm copy, move it out of "
          "its role pool (primary/shadow) into the shared limbo pool so a later "
          "request from EITHER role reuses that one warm copy without a reload "
          "-- keeping fewer warm copies overall. Applies only when no twin "
          "already exists; the configured default model is never moved.",
          DEFAULT_COMPACT_IDLE_MODELS_ACROSS_POOLS, rwc));

  /* One property per role, combining the model's name, version and nvinfer
   * config-file as "name;version;config-file". Only the config-file is required;
   * name/version may be empty (e.g. ";;config.txt"), and a bare path with no ';'
   * is taken as the config-file alone. The config-file's existence is validated
   * when the element is built. */
  g_object_class_install_property (gobject_class, PROP_PRIMARY_MODEL,
      g_param_spec_string ("primary-model", "Primary Model",
          "Default PRIMARY model as \"name;version;config-file\" (config-file "
          "required; name defaults to its basename, version optional). Required "
          "to enable inference.", NULL, rwc));

  g_object_class_install_property (gobject_class, PROP_SHADOW_MODEL,
      g_param_spec_string ("shadow-model", "Shadow Model",
          "Optional default SHADOW model (A/B) as \"name;version;config-file\". "
          "Unset => no shadow.", NULL, rwc));

  g_object_class_install_property (gobject_class, PROP_MODEL_SHARDING,
      g_param_spec_boolean ("model-sharding", "Model Sharding",
          "Enable automatic sharding of streams across model instances. "
          "TRUE (default): overflow streams beyond the per-model batch cap are "
          "distributed onto new same-model shard instances (scale-out). "
          "FALSE: streams exceeding the cap are silently ignored (not processed), "
          "matching stock nvinfer behaviour.",
          DEFAULT_MODEL_SHARDING, rwc));

  g_object_class_install_property (gobject_class, PROP_PROCESS_MODE,
      g_param_spec_enum ("process-mode", "Process Mode",
          "Which GIE position this element occupies: pgie (1, full frame) or "
          "sgie (2, objects). Values match nvinfer's process-mode, so a host that "
          "already sets it at a [secondary-gieN] section configures this element "
          "with no change.",
          GST_TYPE_MODELMUX_PROCESS_MODE, DEFAULT_PROCESS_MODE, rwc));

  /* -------- action signals (runtime control plane) --------
   * Generic (libffi) marshaller (NULL) handles the mixed signatures. Emit with
   * g_signal_emit_by_name(element, "<signal>", args..., &retval). */
  gst_modelmux_signals[SIGNAL_LOAD_MODEL] =
      g_signal_new_class_handler ("load-model", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_load_model),
      NULL, NULL, NULL, G_TYPE_INT, 4, G_TYPE_STRING, G_TYPE_STRING,
      G_TYPE_STRING, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_UNLOAD_MODEL] =
      g_signal_new_class_handler ("unload-model", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_unload_model),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 1, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_RELOAD_MODEL] =
      g_signal_new_class_handler ("reload-model", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_reload_model),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 4, G_TYPE_STRING, G_TYPE_STRING,
      G_TYPE_STRING, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_ATTACH_STREAM] =
      g_signal_new_class_handler ("attach-stream", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_attach_stream),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 4, G_TYPE_UINT, G_TYPE_STRING,
      G_TYPE_STRING, G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_DETACH_STREAM] =
      g_signal_new_class_handler ("detach-stream", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_detach_stream),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 1, G_TYPE_UINT);

  gst_modelmux_signals[SIGNAL_UPDATE_ROUTING] =
      g_signal_new_class_handler ("update-routing", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_update_routing),
      NULL, NULL, NULL, G_TYPE_BOOLEAN, 3, G_TYPE_UINT, G_TYPE_STRING,
      G_TYPE_STRING);

  gst_modelmux_signals[SIGNAL_GET_STATUS] =
      g_signal_new_class_handler ("get-status", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION, G_CALLBACK (gst_modelmux_get_status),
      NULL, NULL, NULL, G_TYPE_STRING, 3, G_TYPE_STRING, G_TYPE_STRING,
      G_TYPE_INT);

  /* -------- pads + element metadata -------- */
  gst_element_class_add_static_pad_template (element_class,
      &gst_modelmux_sink_template);
  gst_element_class_add_static_pad_template (element_class,
      &gst_modelmux_src_template);

  gst_element_class_set_static_metadata (element_class,
      "NvModelMux",
      "Generic/Bin/Filter",
      "Per-stream / per-model DeepStream inference with optional A/B "
      "(primary + shadow) serving, live model load/unload/reload (OTA), "
      "promote/swap routing and per-(stream,model,role) provenance",
      "NVIDIA Corporation. DeepStream SDK");
}

/**
 * Per-INSTANCE initialisation. Runs ONCE for EACH nvmodelmux element created (unlike
 * gst_modelmux_class_init, which runs once for the whole class). @self is this one
 * element's own state; nothing here is shared between instances. It:
 *   - sets this instance's property defaults;
 *   - initialises this instance's mutexes (source_state_lock, api_control_lock);
 *   - creates the boundary ghost pads (no-target) so the element can be LINKED while
 *     still in NULL -- their targets, and the whole inference sub-graph, are wired
 *     later at NULL->READY (gst_modelmux_build).
 */
static void
gst_modelmux_init (GstNvModelMux * self)
{
  GstPadTemplate *tmpl;

  /* property defaults */
  self->enable = DEFAULT_ENABLE;
  self->batch_size = 0;                 /* 0 => inherit from config-file / default */
  self->per_model_batch_size = DEFAULT_PER_MODEL_BATCH_SIZE;
  self->unified_batch = DEFAULT_UNIFIED_BATCH;
  self->compact_idle_models_across_pools = DEFAULT_COMPACT_IDLE_MODELS_ACROSS_POOLS;
  self->model_sharding = DEFAULT_MODEL_SHARDING;
  self->process_mode = DEFAULT_PROCESS_MODE;
  g_mutex_init (&self->source_state_lock);      /* guards the auto-attach source-id table */
  g_mutex_init (&self->api_control_lock);       /* serializes control entry points vs teardown */

  /* No-target ghost pads created up front so the element can be LINKED while
   * still in NULL; their targets are wired when the sub-graph is built. */
  tmpl = gst_static_pad_template_get (&gst_modelmux_sink_template);
  self->sinkpad = gst_ghost_pad_new_no_target_from_template ("sink", tmpl);
  gst_object_unref (tmpl);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  tmpl = gst_static_pad_template_get (&gst_modelmux_src_template);
  self->srcpad = gst_ghost_pad_new_no_target_from_template ("src", tmpl);
  gst_object_unref (tmpl);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);
}
