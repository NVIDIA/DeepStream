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

#include "nvds_eval_report.h"
#include "nvds_eval_ap.h"

#include <gst/gst.h>

#include <sstream>
#include <fstream>
#include <algorithm>
#include <set>
#include <cmath>
#include <cstdio>

/* ------------------------------------------------------------------ */
/* JSON helpers                                                        */
/* ------------------------------------------------------------------ */

static std::string
json_escape (const std::string &s)
{
  std::string out;
  out.reserve (s.size ());
  for (unsigned char c : s) {
    if      (c == '"')  { out += "\\\""; }
    else if (c == '\\') { out += "\\\\"; }
    else if (c == '\n') { out += "\\n";  }
    else if (c == '\r') { out += "\\r";  }
    else if (c == '\t') { out += "\\t";  }
    else if (c < 0x20)  { char buf[8]; snprintf (buf, sizeof(buf), "\\u%04x", c); out += buf; }
    else                { out += (char)c; }
  }
  return out;
}

static std::string
fmt_float (float v)
{
  char buf[32];
  snprintf (buf, sizeof(buf), "%.4f", (double)v);
  return buf;
}

/* ------------------------------------------------------------------ */
/* compute_metrics                                                     */
/* ------------------------------------------------------------------ */

std::vector<ModelMetrics>
compute_metrics (EvalAccumulator &accum, float iou_thresh,
                 const std::set<std::string> &exclude_classes,
                 bool exclude_undetected)
{
  std::vector<ModelMetrics> result;

  for (const auto &model_kv : accum.models ()) {
    const auto &key = model_kv.first;
    const auto &ma = model_kv.second;
    ModelMetrics mm;
    mm.role           = key.role;
    mm.model_name     = key.model_name;
    mm.model_version  = ma.model_version;
    mm.frame_count    = ma.frame_count;
    mm.frames_with_gt = ma.frames_with_gt;
    mm.gt_available   = (ma.frames_with_gt > 0);

    float sum_ap50    = 0.0f;
    float sum_ap50_95 = 0.0f;
    float sum_prec    = 0.0f;
    float sum_rec     = 0.0f;
    int   n_classes   = 0;

    for (const auto &class_kv : ma.per_class) {
      const auto &cls = class_kv.first;
      const auto &ca = class_kv.second;
      ClassMetrics cm;
      cm.class_label = cls;
      cm.total_gt    = ca.total_gt;

      /* AP50 / AP50:95 always use their standard fixed thresholds */
      cm.ap50    = compute_ap50    (ca.preds, ca.total_gt);
      cm.ap50_95 = compute_ap50_95 (ca.preds, ca.total_gt);

      /* TP/FP/precision/recall/F1 use the user-configured iou_thresh */
      uint32_t tp = 0, fp = 0;
      for (const auto &pred_kv : ca.preds) {
        if (pred_kv.second >= iou_thresh) ++tp; else ++fp;
      }
      cm.tp = tp; cm.fp = fp;

      cm.precision = (tp + fp > 0)
          ? (float)tp / (float)(tp + fp) : 0.0f;
      cm.recall    = (ca.total_gt > 0)
          ? (float)tp / (float)ca.total_gt : 0.0f;
      cm.f1        = (cm.precision + cm.recall > 0.0f)
          ? 2.0f * cm.precision * cm.recall / (cm.precision + cm.recall) : 0.0f;

      bool manual_excl = !exclude_classes.empty () && exclude_classes.count (cls) > 0;
      bool auto_excl   = exclude_undetected && (tp + fp == 0);
      cm.auto_excluded = auto_excl;
      cm.excluded      = manual_excl || auto_excl;
      if (!cm.excluded) {
        sum_ap50    += cm.ap50;
        sum_ap50_95 += cm.ap50_95;
        sum_prec    += cm.precision;
        sum_rec     += cm.recall;
        ++n_classes;
      }

      mm.per_class.push_back (std::move (cm));
    }

    if (n_classes > 0) {
      mm.ap50           = sum_ap50    / (float)n_classes;
      mm.ap50_95        = sum_ap50_95 / (float)n_classes;
      mm.mean_precision = sum_prec    / (float)n_classes;
      mm.mean_recall    = sum_rec     / (float)n_classes;
    }

    result.push_back (std::move (mm));
  }

  return result;
}

/* ------------------------------------------------------------------ */
/* build_json_report                                                   */
/* ------------------------------------------------------------------ */

std::string
build_json_report (const std::string               &model_pair_name,
                   const std::vector<ModelMetrics> &metrics,
                   uint32_t                         epoch_frames,
                   bool                             promoted,
                   const std::string               &promoted_model)
{
  std::ostringstream j;
  j << "{\n";
  j << "  \"event\": \"model-eval-result\",\n";
  j << "  \"model_pair\": \""   << json_escape (model_pair_name) << "\",\n";
  j << "  \"epoch_frames\": "   << epoch_frames                  << ",\n";
  j << "  \"models\": {\n";

  for (size_t mi = 0; mi < metrics.size (); ++mi) {
    const auto &mm = metrics[mi];
    /* JSON key = "Primary" or "Shadow" — unique within a model pair */
    j << "    \"" << json_escape (mm.role) << "\": {\n";
    j << "      \"role\": \""          << json_escape (mm.role)          << "\",\n";
    j << "      \"model_name\": \""    << json_escape (mm.model_name)    << "\",\n";
    j << "      \"model_version\": \"" << json_escape (mm.model_version) << "\",\n";
    j << "      \"gt_available\": "    << (mm.gt_available ? "true" : "false") << ",\n";
    j << "      \"frame_count\": "     << mm.frame_count    << ",\n";
    j << "      \"frames_with_gt\": "  << mm.frames_with_gt << ",\n";
    j << "      \"ap50\": "            << fmt_float (mm.ap50)           << ",\n";
    j << "      \"ap50_95\": "         << fmt_float (mm.ap50_95)        << ",\n";
    j << "      \"mean_precision\": "  << fmt_float (mm.mean_precision) << ",\n";
    j << "      \"mean_recall\": "     << fmt_float (mm.mean_recall)    << ",\n";
    j << "      \"per_class\": {\n";

    for (size_t ci = 0; ci < mm.per_class.size (); ++ci) {
      const auto &cm = mm.per_class[ci];
      j << "        \"" << json_escape (cm.class_label) << "\": {";
      j << "\"ap50\": "      << fmt_float (cm.ap50)      << ", ";
      j << "\"ap50_95\": "   << fmt_float (cm.ap50_95)   << ", ";
      j << "\"precision\": " << fmt_float (cm.precision) << ", ";
      j << "\"recall\": "    << fmt_float (cm.recall)    << ", ";
      j << "\"f1\": "        << fmt_float (cm.f1)        << ", ";
      j << "\"tp\": "        << cm.tp                    << ", ";
      j << "\"fp\": "        << cm.fp                    << ", ";
      j << "\"total_gt\": "  << cm.total_gt;
      j << "}";
      if (ci + 1 < mm.per_class.size ()) j << ",";
      j << "\n";
    }

    j << "      }\n";
    j << "    }";
    if (mi + 1 < metrics.size ()) j << ",";
    j << "\n";
  }

  j << "  },\n";
  j << "  \"promoted\": "        << (promoted ? "true" : "false")        << ",\n";
  j << "  \"promoted_model\": \"" << json_escape (promoted_model)         << "\"\n";
  j << "}\n";

  return j.str ();
}

/* ------------------------------------------------------------------ */
/* post_eval_result                                                    */
/* ------------------------------------------------------------------ */

void
post_eval_result (GstElement                      *self,
                  const std::string               &model_pair_name,
                  const std::vector<ModelMetrics> &metrics,
                  uint32_t                         epoch_frames,
                  bool                             promoted,
                  const std::string               &promoted_model,
                  const std::string               &report_file,
                  bool                             cumulative,
                  bool                             show_ap_table)
{
  std::string json = build_json_report (model_pair_name, metrics,
                                        epoch_frames, promoted, promoted_model);

  /* GstMessage on pipeline bus */
  GstStructure *s = gst_structure_new ("model-eval-result",
      "json",       G_TYPE_STRING, json.c_str (),
      "model_pair", G_TYPE_STRING, model_pair_name.c_str (),
      NULL);
  gst_element_post_message (self,
      gst_message_new_application (GST_OBJECT (self), s));

  /* Console table — skipped when show_ap_table is FALSE (default). */
  if (show_ap_table) {

  /* Console — one box-table per model role.
   * Summary info (pair / epoch / mode / AP metrics) is folded into the
   * table header as full-width merged rows so everything is self-contained.
   *
   * Table inner width = 28 + 7*4 + 8*3 + 17 + 8 separators = 105 chars.
   * Full-width header rows use "│ %-103s │" (1 space + 103 content + 1 space).
   * printf() is used throughout so raw UTF-8 box chars reach the terminal
   * without g_print's locale conversion converting them to '?'. */

  const char *mode_str = cumulative ? "CUMULATIVE" : "WINDOW";
  const char *pair_str = model_pair_name.empty () ? "single-model"
                                                   : model_pair_name.c_str ();

  /* Box-drawing helpers — shared across every role section so the Primary and
   * Shadow blocks render as ONE continuous table (separator-joined headers)
   * instead of two detached boxes. */
  auto hrep = [](int n) {
    std::string s; s.reserve ((size_t)n * 3);
    for (int i = 0; i < n; ++i) s += "─";
    return s;
  };
  /* Column-structure border; l/m/r select the corner + tee glyphs:
   *   full header -> columns : ("├","┬","┤")    between class rows : ("├","┼","┤")
   *   columns -> next header : ("├","┴","┤")    final bottom       : ("└","┴","┘") */
  auto col_sep = [&](const char *l, const char *m, const char *r) {
    return std::string (l)
        + hrep (28) + m + hrep (7) + m + hrep (7)
        + m + hrep (7) + m + hrep (7)
        + m + hrep (8) + m + hrep (8) + m + hrep (8)
        + m + hrep (17) + r;
  };
  std::string h_top  = "┌" + hrep (105) + "┐";   /* full-width top               */
  std::string c_open = col_sep ("├", "┬", "┤");  /* full header -> column block  */
  std::string c_mid  = col_sep ("├", "┼", "┤");  /* between class rows           */
  std::string c_join = col_sep ("├", "┴", "┤");  /* one role -> next role header */
  std::string c_bot  = col_sep ("└", "┴", "┘");  /* final bottom                 */

  printf ("\n");
  if (!metrics.empty ())
    printf ("    %s\n", h_top.c_str ());

  for (size_t mi = 0; mi < metrics.size (); ++mi) {
    const auto &mm = metrics[mi];

    /* Sort per-class results alphabetically for stable output */
    std::vector<const ClassMetrics *> sorted_cls;
    for (const auto &cm : mm.per_class)
      sorted_cls.push_back (&cm);
    std::sort (sorted_cls.begin (), sorted_cls.end (),
        [](const ClassMetrics *a, const ClassMetrics *b){
          return a->class_label < b->class_label; });

    /* Frame extent these stats cover. Cumulative reports "thru frame N" (the
     * running total this report spans); window reports the window size. Shown
     * for both modes so every snapshot is self-describing. */
    char frame_tag[48];
    snprintf (frame_tag, sizeof(frame_tag),
        cumulative ? "thru frame %u" : "%u-frame window", mm.frame_count);

    /* Build header content lines */
    std::string model_tag = mm.model_name;
    if (!mm.model_version.empty ()) model_tag += "@" + mm.model_version;

    char h1[200], h2[200];
    if (model_tag.empty ()) {
      snprintf (h1, sizeof(h1),
          "  pair=%-20s  epoch=%-5u  [%-9s]  %-18s  role=%s",
          pair_str, epoch_frames, mode_str, frame_tag, mm.role.c_str ());
    } else {
      snprintf (h1, sizeof(h1),
          "  pair=%-20s  epoch=%-5u  [%-9s]  %-18s  role=%-7s  model=%s",
          pair_str, epoch_frames, mode_str, frame_tag, mm.role.c_str (),
          model_tag.c_str ());
    }
    snprintf (h2, sizeof(h2),
        "  AP50=%.3f    AP50:95=%.3f    P=%.3f    R=%.3f    (GT-frames=%u)",
        mm.ap50, mm.ap50_95, mm.mean_precision, mm.mean_recall,
        mm.frames_with_gt);

    /* Role section inside the single merged table. Between roles, close the
     * previous column block back to full width (c_join) before the next header. */
    if (mi > 0)
      printf ("    %s\n", c_join.c_str ());
    printf ("    │ %-103s │\n", h1);
    printf ("    │ %-103s │\n", h2);
    printf ("    %s\n", c_open.c_str ());
    printf ("    │ %-26s │ %5s │ %5s │ %5s │ %5s │ %6s │ %6s │ %6s │ %-15s │\n",
            "Class", "AP50", "P", "R", "F1", "TP", "FP", "GT", "Note");
    printf ("    %s\n", c_mid.c_str ());

    for (size_t ri = 0; ri < sorted_cls.size (); ++ri) {
      const ClassMetrics *cm = sorted_cls[ri];
      const char *note = cm->auto_excluded     ? "[UNDETECTED]"
                       : cm->excluded          ? "[EXCLUDED]"
                       : (cm->total_gt == 0)   ? "[NOT IN GT]"
                       : (cm->tp == 0
                          && cm->fp == 0)      ? "[NO DET]"
                       :                         "";
      printf ("    │ %-26s │ %.3f │ %.3f │ %.3f │ %.3f │ %6u │ %6u │ %6u │ %-15s │\n",
              cm->class_label.c_str (),
              cm->ap50, cm->precision, cm->recall, cm->f1,
              cm->tp, cm->fp, cm->total_gt, note);
      if (ri + 1 < sorted_cls.size ())
        printf ("    %s\n", c_mid.c_str ());
    }
  }
  if (!metrics.empty ())
    printf ("    %s\n", c_bot.c_str ());
  if (promoted)
    printf ("    >>> NEW PRIMARY: %s\n", promoted_model.c_str ());
  printf ("\n");
  fflush (stdout);

  } /* end show_ap_table */

  /* Optional file append */
  if (!report_file.empty ()) {
    std::ofstream f (report_file, std::ios::app);
    if (f.is_open ())
      f << json << "\n";
    else
      g_warning ("nvinfereval: cannot write report to '%s'",
                 report_file.c_str ());
  }
}
