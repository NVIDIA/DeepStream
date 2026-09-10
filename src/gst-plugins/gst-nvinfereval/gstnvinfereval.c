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
 * gstnvinfereval.c
 * ==================
 * GObject shell for the nvinfereval pass-through evaluation element.
 *
 * Responsibilities:
 *   1. Expose configuration as GObject properties (backward compat).
 *   2. At READY->PAUSED: walk upstream through transparent elements to
 *      auto-detect Mode A vs B; call nvds_eval_engine_prepare().
 *   3. Chain function: iterate batch frames, extract InferenceProvenanceMeta when
 *      present, forward to nvds_eval_engine_process_frame(). Buffer unchanged.
 *   4. At PAUSED->READY: release mm_element ref + reset engine.
 *
 * All evaluation logic lives in the C++ engine (nvds_eval_engine.cpp).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include <gst/gst.h>

#include "gstnvinfereval.h"
#include "gstnvdsmeta.h"          /* gst_buffer_get_nvds_batch_meta      */
#include "nvdsmeta.h"             /* NvDsBatchMeta, NvDsFrameMeta, etc.  */
#include "nvds_eval_engine.h"

/* InferenceProvenanceMeta (public schema struct). NOTE: the matching meta-type enum
 * NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE lives in nvdsmeta.h (included via gstnvdsmeta.h
 * above) -- a standalone consumer must include BOTH headers. */
#include "nvdsmeta_schema.h"

GST_DEBUG_CATEGORY (gst_nv_infer_eval_debug);
#define GST_CAT_DEFAULT gst_nv_infer_eval_debug

/* ------------------------------------------------------------------ */
/* Pad templates                                                       */
/* ------------------------------------------------------------------ */
#define EVAL_CAPS "video/x-raw(memory:NVMM)"

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS (EVAL_CAPS));

static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS (EVAL_CAPS));

/* ------------------------------------------------------------------ */
/* Properties                                                          */
/* ------------------------------------------------------------------ */
enum
{
  PROP_0,
  PROP_GT_FILE,
  PROP_EVAL_INTERVAL,
  PROP_IOU_THRESHOLD,
  PROP_MIN_AP_MARGIN,
  PROP_KPI_AP50,
  PROP_KPI_PRECISION,
  PROP_KPI_RECALL,
  PROP_PROMOTION_MODE,
  PROP_REPORT_FILE,
  PROP_CONFIG_FILE,
  PROP_SHOW_GT_OVERLAY,
  PROP_SHOW_AP_TABLE,
  PROP_CUMULATIVE,
  PROP_EXCLUDE_UNDETECTED,
  PROP_EXCLUDE_CLASSES,
  PROP_LAST
};

#define DEFAULT_SHOW_GT_OVERLAY     FALSE
#define DEFAULT_SHOW_AP_TABLE       FALSE
#define DEFAULT_CUMULATIVE          TRUE
#define DEFAULT_EXCLUDE_UNDETECTED  TRUE
#define DEFAULT_EVAL_INTERVAL   1000
#define DEFAULT_IOU_THRESHOLD   0.5f
#define DEFAULT_MIN_AP_MARGIN   0.02f
#define DEFAULT_KPI_AP50        0.0f
#define DEFAULT_KPI_PRECISION   0.0f
#define DEFAULT_KPI_RECALL      0.0f
#define DEFAULT_PROMOTION_MODE  0

#define gst_nv_infer_eval_parent_class parent_class
G_DEFINE_TYPE (GstNvInferEval, gst_nv_infer_eval, GST_TYPE_ELEMENT);

/* ================================================================== */
/* Forward declarations                                               */
/* ================================================================== */
static void gst_nv_infer_eval_set_property (GObject *obj, guint prop_id,
                  const GValue *v, GParamSpec *pspec);
static void gst_nv_infer_eval_get_property (GObject *obj, guint prop_id,
                  GValue *v, GParamSpec *pspec);
static void gst_nv_infer_eval_finalize     (GObject *obj);
static GstStateChangeReturn
            gst_nv_infer_eval_change_state (GstElement *element,
                  GstStateChange transition);
static GstFlowReturn
            gst_nv_infer_eval_chain        (GstPad *pad, GstObject *parent,
                  GstBuffer *buf);
static gboolean
            gst_nv_infer_eval_sink_event   (GstPad *pad, GstObject *parent,
                  GstEvent *event);

/* ================================================================== */
/* Upstream walk helper                                               */
/* ================================================================== */

/**
 * walk_to_infer_element:
 * Walk upstream from @sinkpad through transparent passthrough elements
 * (queue, queue2, identity, capsfilter) until a non-transparent element
 * is reached.  Returns that element with a ref the caller must unref,
 * or NULL if the graph ends before reaching one.
 *
 * This makes mode detection work even when the user inserts a queue
 * between nvmodelmux and nvinfereval.
 */
/* Return the first available sink pad on @elem.
 * Tries the static "sink" pad first; if absent (e.g. nvdstiler uses numbered
 * request pads sink_0/sink_1/…) falls back to the first sink pad found via
 * the element's sink-pad iterator.  Returns a new ref or NULL. */
static GstPad *
get_any_sink_pad (GstElement *elem)
{
  GstPad *pad = gst_element_get_static_pad (elem, "sink");
  if (pad)
    return pad;

  GstIterator *it  = gst_element_iterate_sink_pads (elem);
  GValue       val = G_VALUE_INIT;
  if (gst_iterator_next (it, &val) == GST_ITERATOR_OK) {
    pad = (GstPad *) g_value_dup_object (&val);
    g_value_unset (&val);
  }
  gst_iterator_free (it);
  return pad;
}

/* Returns a new ref to the nvmodelmux element inside @candidate,
 * or NULL.  @candidate may be the element itself (factory name matches) or
 * a plain GstBin wrapper with no factory (e.g. primary_gie_bin created by
 * create_primary_gie_bin() in the DS test5 app). */
static GstElement *
find_mm_element (GstElement *candidate)
{
  GstElementFactory *fac  = gst_element_get_factory (candidate);
  const gchar       *name = fac
      ? gst_plugin_feature_get_name (GST_PLUGIN_FEATURE (fac)) : NULL;

  if (g_strcmp0 (name, "nvmodelmux") == 0) {
    return (GstElement *) gst_object_ref (candidate);
  }

  /* No factory (plain GstBin wrapper) — search children recursively. */
  if (!fac && GST_IS_BIN (candidate)) {
    GstIterator *it    = gst_bin_iterate_recurse (GST_BIN (candidate));
    GValue       val   = G_VALUE_INIT;
    GstElement  *found = NULL;

    while (!found && gst_iterator_next (it, &val) == GST_ITERATOR_OK) {
      GstElement        *child = (GstElement *) g_value_get_object (&val);
      GstElementFactory *cf    = gst_element_get_factory (child);
      if (cf) {
        const gchar *cn = gst_plugin_feature_get_name (GST_PLUGIN_FEATURE (cf));
        if (g_strcmp0 (cn, "nvmodelmux") == 0)
          found = (GstElement *) gst_object_ref (child);
      }
      g_value_reset (&val);
    }
    g_value_unset (&val);
    gst_iterator_free (it);
    return found;
  }

  return NULL;
}

static GstElement *
walk_to_infer_element (GstPad *sinkpad)
{
  /* Elements that pass NvDsBatchMeta through unchanged — the walk continues
   * through these rather than stopping.  This lets nvinfereval sit anywhere
   * downstream of nvmodelmux (after tiler, OSD, video-convert, tee…). */
  static const gchar * const TRANSPARENT[] = {
    "queue", "queue2", "identity", "capsfilter",
    "tee",              /* splitter: has a "sink" pad the walk can follow  */
    "nvdstiler",        /* tiling: NvDsBatch passes through unchanged       */
    "nvdsosd",          /* OSD render: NvDsBatch passes through unchanged   */
    "nvvideoconvert",   /* colour convert: NvDsBatch passes through         */
    "nvv4l2decoder",    /* decoder: no batch-meta to worry about            */
    "funnel",           /* n-to-1 mux: has a "sink_%u" but walk needs sink  */
    NULL
  };

  GstPad    *cur    = (GstPad *) gst_object_ref (sinkpad);
  GstElement *result = NULL;

  while (cur) {
    GstPad *peer = gst_pad_get_peer (cur);
    gst_object_unref (cur);
    cur = NULL;

    if (!peer)
      break;

    GstElement *elem = gst_pad_get_parent_element (peer);
    gst_object_unref (peer);

    if (!elem)
      break;

    GstElementFactory *fac  = gst_element_get_factory (elem);
    const gchar       *name = fac
        ? gst_plugin_feature_get_name (GST_PLUGIN_FEATURE (fac)) : NULL;

    gboolean transparent = FALSE;
    for (gint i = 0; TRANSPARENT[i]; ++i) {
      if (g_strcmp0 (name, TRANSPARENT[i]) == 0) {
        transparent = TRUE;
        break;
      }
    }

    if (transparent) {
      /* Step through: get this element's sinkpad and continue walking.
       * get_any_sink_pad() handles elements like nvdstiler that expose
       * numbered request pads (sink_0, sink_1) instead of a static "sink". */
      GstPad *sink = get_any_sink_pad (elem);
      gst_object_unref (elem);
      cur = sink;   /* NULL if the element has no sink pad → loop exits */
    } else {
      result = elem;  /* caller owns the ref */
      break;
    }
  }

  return result;
}

/* ================================================================== */
/* Class init                                                         */
/* ================================================================== */
static void
gst_nv_infer_eval_class_init (GstNvInferEvalClass *klass)
{
  GObjectClass    *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property = gst_nv_infer_eval_set_property;
  gobject_class->get_property = gst_nv_infer_eval_get_property;
  gobject_class->finalize     = gst_nv_infer_eval_finalize;

  element_class->change_state = gst_nv_infer_eval_change_state;

  /* ---- Properties ---- */

  g_object_class_install_property (gobject_class, PROP_GT_FILE,
      g_param_spec_string ("gt-file", "GT File",
          "Default ground-truth file or directory for all models "
          "(format auto-detected: KITTI dir | NVSchema JSONL | COCO JSON | TAO JSON). "
          "Per-model overrides go in the config-file [model-<name>] sections. "
          "Leave empty to run in no-GT relative-comparison mode.",
          "", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EVAL_INTERVAL,
      g_param_spec_uint ("eval-interval", "Eval Interval",
          "Number of Primary frames per evaluation epoch, counted independently "
          "per model_name. Shadow frames are excluded from the count so the "
          "epoch always spans exactly this many Primary frames regardless of "
          "whether a shadow model is present.",
          1, G_MAXUINT, DEFAULT_EVAL_INTERVAL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_IOU_THRESHOLD,
      g_param_spec_float ("iou-threshold", "IoU Threshold",
          "Minimum IoU for a prediction to count as a true positive in "
          "precision / recall / F1 computation. "
          "AP@0.5 and AP@0.5:0.95 always use their fixed standard thresholds.",
          0.0f, 1.0f, DEFAULT_IOU_THRESHOLD,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MIN_AP_MARGIN,
      g_param_spec_float ("min-ap-margin", "Min AP Margin",
          "Default: shadow AP@0.5 must exceed primary AP@0.5 by at least this "
          "delta to trigger promotion. Per-model override available in config-file.",
          0.0f, 1.0f, DEFAULT_MIN_AP_MARGIN,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_KPI_AP50,
      g_param_spec_float ("kpi-ap50", "KPI AP50",
          "Default: minimum AP@0.5 the shadow must reach for promotion (0.0 = disabled). "
          "Per-model override available in config-file.",
          0.0f, 1.0f, DEFAULT_KPI_AP50,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_KPI_PRECISION,
      g_param_spec_float ("kpi-precision", "KPI Precision",
          "Default: minimum mean precision the shadow must reach (0.0 = disabled). "
          "Per-model override available in config-file.",
          0.0f, 1.0f, DEFAULT_KPI_PRECISION,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_KPI_RECALL,
      g_param_spec_float ("kpi-recall", "KPI Recall",
          "Default: minimum mean recall the shadow must reach (0.0 = disabled). "
          "Per-model override available in config-file.",
          0.0f, 1.0f, DEFAULT_KPI_RECALL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROMOTION_MODE,
      g_param_spec_uint ("promotion-mode", "Promotion Mode",
          "Action taken when shadow wins an epoch (Mode B only). "
          "0 = promote: shadow becomes primary, A/B ends. "
          "1 = swap: shadow becomes primary, old primary becomes shadow, A/B continues. "
          "2 = report-only: metrics reported, no REST call, pure monitoring.",
          0, 2, DEFAULT_PROMOTION_MODE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_REPORT_FILE,
      g_param_spec_string ("report-file", "Report File",
          "Optional path to write JSON epoch reports (appended). "
          "Leave empty to disable file output (GstMessage is always posted).",
          "", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SHOW_GT_OVERLAY,
      g_param_spec_boolean ("show-gt-overlay", "Show GT Overlay",
          "Draw ground-truth bounding boxes as green rectangles via nvosd. "
          "Useful for debugging GT alignment.",
          DEFAULT_SHOW_GT_OVERLAY,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SHOW_AP_TABLE,
      g_param_spec_boolean ("show-ap-table", "Show AP Table",
          "Print the per-class AP/precision/recall table to the console at "
          "each epoch boundary. Default FALSE — set TRUE or use show-ap-table=1 "
          "in the config file [eval] section to enable.",
          DEFAULT_SHOW_AP_TABLE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EXCLUDE_CLASSES,
      g_param_spec_string ("exclude-classes", "Exclude Classes",
          "Comma-separated list of class labels to exclude from mean AP/P/R. "
          "Excluded classes are still shown per-class with [EXCLUDED] note. "
          "E.g. \"unknown,Box\". Also set via exclude-classes in config-file.",
          "",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EXCLUDE_UNDETECTED,
      g_param_spec_boolean ("exclude-undetected", "Exclude Undetected",
          "Automatically exclude from mean AP/P/R any class for which the model "
          "produced zero detections (TP=0, FP=0). Such classes appear in the "
          "per-class table with [UNDETECTED]. Default TRUE.",
          DEFAULT_EXCLUDE_UNDETECTED,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CUMULATIVE,
      g_param_spec_boolean ("cumulative", "Cumulative",
          "When TRUE (default) metrics are computed over all frames since "
          "pipeline start (accumulator never resets). When FALSE each epoch is "
          "an independent tumbling window of eval-interval Primary frames.",
          DEFAULT_CUMULATIVE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string ("config-file", "Config File",
          "Path to a GKeyFile INI config (see configs/config_nvinfereval.txt). "
          "Keys present in the file override the corresponding GObject properties. "
          "Per-model [model-<name>] sections set model-specific GT files, "
          "KPI thresholds, and promotion margins. "
          "Leave empty to use GObject properties only.",
          "", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata (element_class,
      "NVIDIA DeepStream Inference Evaluator",
      "Filter/Analyzer",
      "Live multi-model accuracy evaluation with per-model GT, "
      "A/B promotion gating, and targeted stream routing",
      "NVIDIA Corporation");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);
}

/* ================================================================== */
/* Sink event handler                                                 */
/* ================================================================== */
/* On EOS, force a final evaluation epoch so the trailing partial epoch
 * (frames since the last eval_interval boundary) is reported/saved before
 * the pipeline tears down — otherwise that tail is lost (window mode) or its
 * definitive end-of-stream totals are never emitted (cumulative mode). All
 * other events fall through to the default handler unchanged. */
static gboolean
gst_nv_infer_eval_sink_event (GstPad *pad, GstObject *parent, GstEvent *event)
{
  GstNvInferEval *self = GST_NVINFEREVAL (parent);

  if (GST_EVENT_TYPE (event) == GST_EVENT_EOS && self->eng) {
    GST_INFO_OBJECT (self, "EOS -> forcing final evaluation epoch");
    nvds_eval_engine_flush (self->eng, GST_ELEMENT (self));
  }

  return gst_pad_event_default (pad, parent, event);
}

/* ================================================================== */
/* Instance init                                                      */
/* ================================================================== */
static void
gst_nv_infer_eval_init (GstNvInferEval *self)
{
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_nv_infer_eval_chain));
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_nv_infer_eval_sink_event));
  GST_PAD_SET_PROXY_CAPS (self->sinkpad);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  GST_PAD_SET_PROXY_CAPS (self->srcpad);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  self->eng = nvds_eval_engine_new ();

  /* property defaults */
  self->gt_file        = g_strdup ("");
  self->eval_interval  = DEFAULT_EVAL_INTERVAL;
  self->iou_threshold  = DEFAULT_IOU_THRESHOLD;
  self->min_ap_margin  = DEFAULT_MIN_AP_MARGIN;
  self->kpi_ap50       = DEFAULT_KPI_AP50;
  self->kpi_precision  = DEFAULT_KPI_PRECISION;
  self->kpi_recall     = DEFAULT_KPI_RECALL;
  self->promotion_mode     = DEFAULT_PROMOTION_MODE;
  self->report_file    = g_strdup ("");
  self->config_file    = g_strdup ("");
  self->show_gt_overlay    = DEFAULT_SHOW_GT_OVERLAY;
  self->show_ap_table      = DEFAULT_SHOW_AP_TABLE;
  self->cumulative         = DEFAULT_CUMULATIVE;
  self->exclude_undetected = DEFAULT_EXCLUDE_UNDETECTED;
  self->exclude_classes    = g_strdup ("");
  self->mm_element     = NULL;
  self->mode_ab        = FALSE;
  self->prov_meta_type = (NvDsMetaType) 0;
  self->prov_warn_once = FALSE;
}

/* ================================================================== */
/* Finalize                                                           */
/* ================================================================== */
static void
gst_nv_infer_eval_finalize (GObject *obj)
{
  GstNvInferEval *self = GST_NVINFEREVAL (obj);

  nvds_eval_engine_free (self->eng);
  self->eng = NULL;

  if (self->mm_element) {
    gst_object_unref (self->mm_element);
    self->mm_element = NULL;
  }

  g_free (self->gt_file);
  g_free (self->report_file);
  g_free (self->config_file);
  g_free (self->exclude_classes);

  G_OBJECT_CLASS (parent_class)->finalize (obj);
}

/* ================================================================== */
/* Properties                                                         */
/* ================================================================== */
static void
gst_nv_infer_eval_set_property (GObject *obj, guint prop_id,
    const GValue *v, GParamSpec *pspec)
{
  GstNvInferEval *self = GST_NVINFEREVAL (obj);

  switch (prop_id) {
    case PROP_GT_FILE:
      g_free (self->gt_file);
      self->gt_file = g_value_dup_string (v);
      nvds_eval_engine_set_gt_file (self->eng, self->gt_file);
      break;
    case PROP_EVAL_INTERVAL:
      self->eval_interval = g_value_get_uint (v);
      nvds_eval_engine_set_eval_interval (self->eng, self->eval_interval);
      break;
    case PROP_IOU_THRESHOLD:
      self->iou_threshold = g_value_get_float (v);
      nvds_eval_engine_set_iou_threshold (self->eng, self->iou_threshold);
      break;
    case PROP_MIN_AP_MARGIN:
      self->min_ap_margin = g_value_get_float (v);
      nvds_eval_engine_set_min_ap_margin (self->eng, self->min_ap_margin);
      break;
    case PROP_KPI_AP50:
      self->kpi_ap50 = g_value_get_float (v);
      nvds_eval_engine_set_kpi_ap50 (self->eng, self->kpi_ap50);
      break;
    case PROP_KPI_PRECISION:
      self->kpi_precision = g_value_get_float (v);
      nvds_eval_engine_set_kpi_precision (self->eng, self->kpi_precision);
      break;
    case PROP_KPI_RECALL:
      self->kpi_recall = g_value_get_float (v);
      nvds_eval_engine_set_kpi_recall (self->eng, self->kpi_recall);
      break;
    case PROP_PROMOTION_MODE:
      self->promotion_mode = g_value_get_uint (v);
      nvds_eval_engine_set_promotion_mode (self->eng, self->promotion_mode);
      break;
    case PROP_REPORT_FILE:
      g_free (self->report_file);
      self->report_file = g_value_dup_string (v);
      nvds_eval_engine_set_report_file (self->eng, self->report_file);
      break;
    case PROP_CONFIG_FILE:
      g_free (self->config_file);
      self->config_file = g_value_dup_string (v);
      nvds_eval_engine_set_config_file (self->eng, self->config_file);
      break;
    case PROP_SHOW_GT_OVERLAY:
      self->show_gt_overlay = g_value_get_boolean (v);
      nvds_eval_engine_set_show_gt_overlay (self->eng, self->show_gt_overlay);
      break;
    case PROP_SHOW_AP_TABLE:
      self->show_ap_table = g_value_get_boolean (v);
      nvds_eval_engine_set_show_ap_table (self->eng, self->show_ap_table);
      break;
    case PROP_CUMULATIVE:
      self->cumulative = g_value_get_boolean (v);
      nvds_eval_engine_set_cumulative (self->eng, self->cumulative);
      break;
    case PROP_EXCLUDE_UNDETECTED:
      self->exclude_undetected = g_value_get_boolean (v);
      nvds_eval_engine_set_exclude_undetected (self->eng, self->exclude_undetected);
      break;
    case PROP_EXCLUDE_CLASSES:
      g_free (self->exclude_classes);
      self->exclude_classes = g_value_dup_string (v);
      nvds_eval_engine_set_exclude_classes (self->eng, self->exclude_classes);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
      break;
  }
}

static void
gst_nv_infer_eval_get_property (GObject *obj, guint prop_id,
    GValue *v, GParamSpec *pspec)
{
  GstNvInferEval *self = GST_NVINFEREVAL (obj);

  switch (prop_id) {
    case PROP_GT_FILE:        g_value_set_string  (v, self->gt_file);        break;
    case PROP_EVAL_INTERVAL:  g_value_set_uint    (v, self->eval_interval);  break;
    case PROP_IOU_THRESHOLD:  g_value_set_float   (v, self->iou_threshold);  break;
    case PROP_MIN_AP_MARGIN:  g_value_set_float   (v, self->min_ap_margin);  break;
    case PROP_KPI_AP50:       g_value_set_float   (v, self->kpi_ap50);       break;
    case PROP_KPI_PRECISION:  g_value_set_float   (v, self->kpi_precision);  break;
    case PROP_KPI_RECALL:     g_value_set_float   (v, self->kpi_recall);     break;
    case PROP_PROMOTION_MODE: g_value_set_uint    (v, self->promotion_mode);     break;
    case PROP_REPORT_FILE:      g_value_set_string  (v, self->report_file);      break;
    case PROP_CONFIG_FILE:      g_value_set_string  (v, self->config_file);      break;
    case PROP_SHOW_GT_OVERLAY:  g_value_set_boolean (v, self->show_gt_overlay);  break;
    case PROP_SHOW_AP_TABLE:    g_value_set_boolean (v, self->show_ap_table);    break;
    case PROP_CUMULATIVE:          g_value_set_boolean (v, self->cumulative);           break;
    case PROP_EXCLUDE_UNDETECTED:  g_value_set_boolean (v, self->exclude_undetected);   break;
    case PROP_EXCLUDE_CLASSES:     g_value_set_string  (v, self->exclude_classes);      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
      break;
  }
}

/* ================================================================== */
/* State change: auto-detect mode + prepare engine                    */
/* ================================================================== */
static GstStateChangeReturn
gst_nv_infer_eval_change_state (GstElement    *element,
                                   GstStateChange transition)
{
  GstNvInferEval *self = GST_NVINFEREVAL (element);
  GstStateChangeReturn ret;

  /* Upward transitions: do our setup BEFORE chaining to the parent (pads are
   * activated by the parent's change_state, so the streaming thread is not yet
   * running). */
  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
    {
      /* Walk upstream through queues/identity/capsfilter to find the
       * actual inference element. */
      GstElement *upstream = walk_to_infer_element (self->sinkpad);

      if (self->mm_element) {
        gst_object_unref (self->mm_element);
        self->mm_element = NULL;
      }

      /* Detect nvmodelmux — check directly or search inside a bin
       * wrapper (e.g. primary_gie_bin created by create_primary_gie_bin). */
      GstElement *mm = upstream ? find_mm_element (upstream) : NULL;
      if (upstream)
        gst_object_unref (upstream);   /* find_mm_element takes its own ref */

      if (mm) {
        self->mm_element = mm;         /* takes the ref from find_mm_element */
        self->mode_ab    = TRUE;
        GST_INFO_OBJECT (self,
            "nvinfereval: Mode B (multi-model A/B): "
            "upstream is nvmodelmux");
      } else {
        self->mode_ab = FALSE;
        GST_INFO_OBJECT (self,
            "nvinfereval: Mode A (single model): "
            "no upstream nvmodelmux found");
      }

      /* InferenceProvenanceMeta is stamped with the SHARED SDK meta type
       * NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE (nvdsmeta.h) -- no plugin-private
       * string registration anymore (nvmodelmux sets this enum directly; matching
       * a registered-string type here would never hit). */
      self->prov_meta_type = (NvDsMetaType) NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE;

      nvds_eval_engine_prepare (self->eng, element, self->mm_element);
      break;
    }

    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  /* Downward transitions: do our teardown AFTER the parent has chained, so the
   * parent has already deactivated the pads and joined the streaming task.
   * This guarantees no chain()/process_frame() is running concurrently when we
   * reset the engine (which clears the per-model contexts), preventing a data
   * race / use-after-free on the engine state. */
  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      nvds_eval_engine_reset (self->eng);
      if (self->mm_element) {
        gst_object_unref (self->mm_element);
        self->mm_element = NULL;
      }
      self->mode_ab        = FALSE;
      self->prov_meta_type = (NvDsMetaType) 0;
      self->prov_warn_once = FALSE;
      break;

    default:
      break;
  }

  return ret;
}

/* ================================================================== */
/* Chain function                                                     */
/* ================================================================== */
static GstFlowReturn
gst_nv_infer_eval_chain (GstPad *pad, GstObject *parent, GstBuffer *buf)
{
  GstNvInferEval *self       = GST_NVINFEREVAL (parent);
  NvDsBatchMeta    *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  if (!batch_meta)
    goto push;

  for (NvDsFrameMetaList *fl = batch_meta->frame_meta_list;
       fl; fl = fl->next) {
    NvDsFrameMeta    *frame_meta = (NvDsFrameMeta *) fl->data;
    InferenceProvenanceMeta *prov       = NULL;

    /* Always scan for InferenceProvenanceMeta, even when the upstream walk at
     * READY->PAUSED did not find nvmodelmux (mode_ab==FALSE).
     * This makes Mode B work when intermediate elements (nvdstiler, nvdsosd,
     * tee, etc.) sit between nvmodelmux and nvinfereval: the walk
     * may have stopped early, but the provenance meta is preserved on every
     * frame regardless of pipeline topology.
     * Guard: prov_meta_type is set at READY->PAUSED in both modes (the shared
     * NVDS_CUSTOM_MSG_INFERENCE_PROVENANCE enum); 0 = not yet resolved → skip. */
    if (self->prov_meta_type != (NvDsMetaType) 0) {
      for (NvDsUserMetaList *ul = frame_meta->frame_user_meta_list;
           ul; ul = ul->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) ul->data;
        if (um && um->base_meta.meta_type == self->prov_meta_type &&
            um->user_meta_data) {
          prov = (InferenceProvenanceMeta *) um->user_meta_data;
          break;
        }
      }
    }

    const gchar *stream_name_for_overlay;
    const gchar *model_name_for_overlay;
    guint        frame_num_for_overlay;

    if (prov) {
      /* Mode B: full provenance found on frame. Promotion no longer needs the
       * upstream element handle (mm_element) — it is performed via the
       * nvmodelmux stream/route REST API (set promote-rest-url in
       * [eval]). So a missing mm_element is no longer a promotion blocker. */
      /* Mode B: full provenance — model_name / role / version / stream.
       * Use frame_meta->sensorInfo_meta.sensor_id for the camera/GT lookup:
       * prov->camera_name is nvmodelmux's internal tee-pad name ("src"),
       * not the DS sensor ID ("Camera", "Camera_01", …) that GT files key on.
       * Use frame_meta->frame_num (DS per-source frame counter, same as Mode A):
       * prov->frame_num carries the ORIGINAL pre-remux frame number (the source-trace
       * meta is stamped always-on at the bin input; only its verification LOGGING is
       * MM_FRAME_TRACE-gated) —
       * frame_meta->frame_num is used instead because GT files key on the DS per-source counter (same as Mode A). */
      nvds_eval_engine_process_frame (self->eng, GST_ELEMENT (self),
          prov->source_id,
          (guint) frame_meta->frame_num,
          prov->role,
          prov->model_name,
          prov->model_version,
          frame_meta->sensorInfo_meta.sensor_id,
          frame_meta->obj_meta_list);
      stream_name_for_overlay = frame_meta->sensorInfo_meta.sensor_id;
      frame_num_for_overlay   = (guint) frame_meta->frame_num;
      model_name_for_overlay  = prov->model_name;
    } else {
      /* Mode A: synthesise role="Primary", model_name="".
       * Use sensor_id from frame meta so GT lookup matches the sensor-id-list
       * names (e.g. "Camera","Camera_01") instead of falling back to "0","1". */
      nvds_eval_engine_process_frame (self->eng, GST_ELEMENT (self),
          frame_meta->source_id,
          (guint) frame_meta->frame_num,
          "Primary",
          "",
          "",
          frame_meta->sensorInfo_meta.sensor_id,
          frame_meta->obj_meta_list);
      stream_name_for_overlay = frame_meta->sensorInfo_meta.sensor_id;
      frame_num_for_overlay   = (guint) frame_meta->frame_num;
      model_name_for_overlay  = "";
    }

    if (nvds_eval_engine_get_show_gt_overlay (self->eng)) {
      nvds_eval_engine_attach_gt_overlay (self->eng, GST_ELEMENT (self),
          batch_meta, frame_meta,
          stream_name_for_overlay,
          frame_num_for_overlay,
          model_name_for_overlay);
    }
  }

push:
  return gst_pad_push (self->srcpad, buf);
}
