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
 * SECTION:element-nvinfereval
 *
 * nvinfereval
 * =============
 * A pass-through DeepStream GStreamer element that evaluates model accuracy
 * live against ground-truth files. Buffers flow downstream UNCHANGED.
 *
 * Two operating modes, auto-detected at READY->PAUSED by walking upstream
 * through transparent elements (queue, queue2, identity, capsfilter):
 *
 *   Mode A — Single model
 *     nvinfer -> [queue?] -> nvinfereval -> sink
 *     No InferenceProvenanceMeta on frames. Evaluates one model vs GT.
 *     No promotion logic.
 *
 *   Mode B — Multi-model A/B evaluation + promotion
 *     nvmodelmux -> [queue?] -> nvinfereval -> sink
 *     InferenceProvenanceMeta present per frame (role / model_name / model_version).
 *     Any number of model_names may coexist — each gets its own isolated
 *     epoch counter, GT index, and accumulator.
 *     Promotion is triggered independently per model_name and targets only
 *     the streams carrying that pair.
 *
 * Configuration
 * -------------
 * Individual GObject properties set baseline values. If config-file is given,
 * its sections overlay those values (config file wins for keys it defines).
 * See configs/config_nvinfereval.txt for the full format.
 */

#ifndef __GST_NVINFEREVAL_H__
#define __GST_NVINFEREVAL_H__

#include <gst/gst.h>
#include "nvds_eval_engine.h"   /* opaque C++ engine handle + C API */
#include "nvdsmeta.h"           /* NvDsMetaType                      */

G_BEGIN_DECLS

#define GST_TYPE_NVINFEREVAL (gst_nv_infer_eval_get_type ())
#define GST_NVINFEREVAL(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVINFEREVAL, GstNvInferEval))
#define GST_NVINFEREVAL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_NVINFEREVAL, GstNvInferEvalClass))
#define GST_IS_NVINFEREVAL(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVINFEREVAL))
#define GST_IS_NVINFEREVAL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_NVINFEREVAL))

typedef struct _GstNvInferEval      GstNvInferEval;
typedef struct _GstNvInferEvalClass GstNvInferEvalClass;

/**
 * GstNvInferEval:
 *
 * Thin GObject wrapper — all evaluation logic lives in NvdsEvalEngine.
 */
struct _GstNvInferEval
{
  GstElement        parent;

  /* pads */
  GstPad           *sinkpad;
  GstPad           *srcpad;

  /* C++ evaluation engine (opaque handle) */
  NvdsEvalEngine   *eng;

  /* upstream nvmodelmux element (NULL in Mode A).
   * Discovered at READY->PAUSED; ref held until PAUSED->READY. */
  GstElement       *mm_element;

  /* TRUE when upstream (after transparent-element walk) is nvmodelmux */
  gboolean          mode_ab;

  /* InferenceProvenanceMeta user-meta type, resolved once at READY->PAUSED */
  NvDsMetaType      prov_meta_type;

  /* Rate-limit the "prov found but mm_element NULL" warning to once per run */
  gboolean          prov_warn_once;

  /* --- raw property storage --- */
  gchar            *gt_file;
  guint             eval_interval;
  gfloat            iou_threshold;
  gfloat            min_ap_margin;
  gfloat            kpi_ap50;
  gfloat            kpi_precision;
  gfloat            kpi_recall;
  guint             promotion_mode;      /* 0=promote 1=swap 2=report-only */
  gchar            *report_file;
  gchar            *config_file;     /* path to GKeyFile INI config */
  gboolean          show_gt_overlay; /* draw GT boxes as green rects via nvosd */
  gboolean          show_ap_table;   /* print per-class AP table to console */
  gboolean          cumulative;           /* don't reset accumulator between epochs */
  gboolean          exclude_undetected;  /* auto-exclude classes with 0 TP+FP */
  gchar            *exclude_classes;     /* comma-separated classes to skip in mean */
};

struct _GstNvInferEvalClass
{
  GstElementClass parent_class;
};

GType gst_nv_infer_eval_get_type (void);

G_END_DECLS

#endif /* __GST_NVINFEREVAL_H__ */
