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
 * nvds_eval_engine.h
 * ==================
 * C-compatible API for the C++ evaluation engine.
 *
 * The engine is fully opaque to the C GObject shell. All evaluation state —
 * per-model-pair contexts, GT indices, accumulators, config — lives inside
 * the C++ NvdsEvalEngine struct. This header exposes only the opaque handle
 * typedef and flat C functions suitable for inclusion from C code.
 *
 * Design overview
 * ---------------
 * The engine supports any number of simultaneous model pairs.  Each unique
 * model_name observed in InferenceProvenanceMeta (Mode B) or the synthetic
 * "Primary"/"" name (Mode A) gets its own isolated context:
 *
 *   - A dedicated EvalAccumulator          (no cross-model contamination)
 *   - A dedicated GTIndex                  (per-model GT file)
 *   - A per-model epoch frame counter      (incremented only on Primary frames)
 *   - A source_id -> camera_id map of the streams carrying that pair
 *     (built from per-frame sensor info; scopes the targeted stream/route;
 *     liveness-stamped per epoch so removed streams age out of the scope)
 *   - Per-model KPI thresholds / margins   (from [model-<name>] config section)
 *
 * At each per-model epoch boundary the engine:
 *   1. Computes metrics for every (role, model_name) in the accumulator.
 *   2. Pairs Primary vs Shadow by model_name — no cross-model comparison.
 *   3. Triggers targeted update-routing per source_id if shadow wins.
 *   4. Posts a GstMessage and optionally appends to the report file.
 *   5. Resets only that model's accumulator (other models continue unaffected).
 */

#ifndef NVDS_EVAL_ENGINE_H
#define NVDS_EVAL_ENGINE_H

#include <glib.h>
#include <gst/gst.h>
#include "nvdsmeta.h"   /* NvDsBatchMeta, NvDsFrameMeta */

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle; defined as a C++ struct in nvds_eval_engine.cpp. */
typedef struct NvdsEvalEngine NvdsEvalEngine;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
NvdsEvalEngine *nvds_eval_engine_new  (void);
void            nvds_eval_engine_free (NvdsEvalEngine *eng);

/* ------------------------------------------------------------------ */
/* Property setters — called from GObject set_property (backward compat)
 *
 * These values become the defaults for any field not set in the config
 * file.  They must be set BEFORE nvds_eval_engine_prepare() is called.
 * ------------------------------------------------------------------ */
void nvds_eval_engine_set_gt_file       (NvdsEvalEngine *eng, const gchar *path);
void nvds_eval_engine_set_eval_interval (NvdsEvalEngine *eng, guint interval);
void nvds_eval_engine_set_iou_threshold (NvdsEvalEngine *eng, gfloat thresh);
void nvds_eval_engine_set_min_ap_margin (NvdsEvalEngine *eng, gfloat margin);
void nvds_eval_engine_set_kpi_ap50      (NvdsEvalEngine *eng, gfloat v);
void nvds_eval_engine_set_kpi_precision (NvdsEvalEngine *eng, gfloat v);
void nvds_eval_engine_set_kpi_recall    (NvdsEvalEngine *eng, gfloat v);
void     nvds_eval_engine_set_promotion_mode   (NvdsEvalEngine *eng, guint v);
guint    nvds_eval_engine_get_promotion_mode   (NvdsEvalEngine *eng);
void nvds_eval_engine_set_show_gt_overlay     (NvdsEvalEngine *eng, gboolean v);
gboolean nvds_eval_engine_get_show_gt_overlay (NvdsEvalEngine *eng);
void nvds_eval_engine_set_show_ap_table       (NvdsEvalEngine *eng, gboolean v);
gboolean nvds_eval_engine_get_show_ap_table   (NvdsEvalEngine *eng);

void nvds_eval_engine_set_cumulative             (NvdsEvalEngine *eng, gboolean v);
gboolean nvds_eval_engine_get_cumulative         (NvdsEvalEngine *eng);
void nvds_eval_engine_set_exclude_undetected     (NvdsEvalEngine *eng, gboolean v);
gboolean nvds_eval_engine_get_exclude_undetected (NvdsEvalEngine *eng);
void nvds_eval_engine_set_exclude_classes        (NvdsEvalEngine *eng, const gchar *v);
void nvds_eval_engine_set_report_file   (NvdsEvalEngine *eng, const gchar *path);
void nvds_eval_engine_set_config_file   (NvdsEvalEngine *eng, const gchar *path);

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_prepare
 *
 * Called at READY->PAUSED.
 *
 * Builds the resolved EvalConfig (property defaults overlaid with config
 * file if set), stores mm_element, and clears all per-model pair contexts
 * so each pipeline run starts fresh.  GT indices are loaded lazily on the
 * first frame seen for each model_name.
 *
 * @self        the GstElement, used for GST_DEBUG / message posting
 * @mm_element  upstream nvmodelmux element (NULL in Mode A);
 *              borrowed reference — lifetime managed by the GObject shell
 *
 * Always returns TRUE; individual model GT-load failures are logged as
 * warnings and those models run in relative-comparison-only mode.
 * ------------------------------------------------------------------ */
gboolean nvds_eval_engine_prepare (NvdsEvalEngine *eng,
                                   GstElement     *self,
                                   GstElement     *mm_element);

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_process_frame
 *
 * Called from the GStreamer chain function for every frame in the batch.
 * Must not block; all heavy work (epoch boundary) happens inline but is
 * bounded by the accumulator size.
 *
 * @source_id     NvDsFrameMeta.source_id
 * @frame_num     InferenceProvenanceMeta.frame_num (Mode B) or
 *                NvDsFrameMeta.frame_num (Mode A)
 * @role          "Primary" or "Shadow" (Mode B) / "Primary" (Mode A)
 * @model_name    logical model name from InferenceProvenanceMeta, or "" (Mode A)
 * @model_version model version from InferenceProvenanceMeta, or "" (Mode A)
 * @stream_name   InferenceProvenanceMeta.camera_name used as camera_id for GT
 *                lookup; NULL or "" falls back to to_string(source_id)
 * @obj_meta_list NvDsFrameMeta.obj_meta_list (GList of NvDsObjectMeta*)
 * ------------------------------------------------------------------ */
void nvds_eval_engine_process_frame (NvdsEvalEngine *eng,
                                     GstElement     *self,
                                     guint           source_id,
                                     guint           frame_num,
                                     const gchar    *role,
                                     const gchar    *model_name,
                                     const gchar    *model_version,
                                     const gchar    *stream_name,
                                     GList          *obj_meta_list);

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_attach_gt_overlay
 *
 * Attach NvDsDisplayMeta green rectangles for the GT boxes of the given
 * frame to @frame_meta so nvosd renders them.  Call this from the chain
 * function when the show-gt-overlay flag is set.
 *
 * Modular — can be called independently of process_frame.  Safe to call
 * when GT is not loaded (no-op) or when the frame has no GT entry (no-op).
 *
 * @batch_meta   NvDsBatchMeta for the current buffer (for display-meta pool)
 * @frame_meta   NvDsFrameMeta to attach rects to
 * @stream_name  Camera ID for GT lookup (same value passed to process_frame)
 * @frame_num    Frame number for GT lookup (same value passed to process_frame)
 * @model_name   Model pair to look up GT from ("" for Mode A)
 * ------------------------------------------------------------------ */
void nvds_eval_engine_attach_gt_overlay (NvdsEvalEngine *eng,
                                          GstElement     *self,
                                          NvDsBatchMeta  *batch_meta,
                                          NvDsFrameMeta  *frame_meta,
                                          const gchar    *stream_name,
                                          guint           frame_num,
                                          const gchar    *model_name);

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_reset
 *
 * Called at PAUSED->READY.  Clears all per-model pair contexts and
 * releases the mm_element reference.  Property values and the config
 * file path are retained so prepare() can be called again on the next
 * READY->PAUSED transition.
 * ------------------------------------------------------------------ */
void nvds_eval_engine_reset (NvdsEvalEngine *eng);

/* ------------------------------------------------------------------ */
/* nvds_eval_engine_flush
 *
 * Force a final ("forced epoch") report for any model context that still
 * holds a partial, sub-eval_interval epoch. Intended to be called on EOS so
 * the trailing frames that never reached an interval boundary are still
 * reported/saved (every such frame was already evaluated as it arrived).
 * Safe to call multiple times — a context with no pending frames is skipped.
 * ------------------------------------------------------------------ */
void nvds_eval_engine_flush (NvdsEvalEngine *eng, GstElement *self);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NVDS_EVAL_ENGINE_H */
