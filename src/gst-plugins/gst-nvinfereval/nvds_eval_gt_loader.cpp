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
 * nvds_eval_gt_loader.cpp
 * =======================
 * Ground-truth loading with automatic format detection.
 *
 * Supported formats:
 *   KITTI    — directory of *.txt files, one per frame
 *   NVSchema — JSONL (one JSON object per line)
 *   COCO     — single JSON file with images/annotations/categories
 *   TAO      — JSON keyed by camera_id -> frame_num -> list of objects
 *
 * All formats are normalised into GTIndex: map<(camera_id,frame_num), GTBoxes>.
 */

#include "nvds_eval_gt_loader.h"

#include <glib.h>
#include <json-glib/json-glib.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <dirent.h>

/* Non-throwing numeric parsing via GLib's LOCALE-INDEPENDENT parsers -- the
 * codebase convention for external text (g_ascii_strtoull/g_ascii_strtod, as
 * in gst-nvinfer/gst-nvdsanalytics/gst-nvdsmetamux property parsers). Two
 * reasons over std::sto*:
 *  - std::sto* needs try/catch, and the tmake/GVS build compiles C++ WITHOUT
 *    -fexceptions (the CODE_GENERATION *_exceptions variants are not honored
 *    for mixed C/C++ components) -- this plugin is exception-free by design;
 *  - std::stof/strtof honor the process locale: under e.g. de_DE, "0.5"
 *    parses as 0 (stops at '.'). KITTI/COCO GT files ALWAYS use '.', so the
 *    ASCII-only parsers are the correct tool, not just the conventional one.
 * Returns FALSE on empty/unparseable input; *out is written ONLY on success
 * (callers that ignore the result keep their pre-set default). */
static bool
parse_u32 (const std::string &str, uint32_t *out)
{
  gchar *end = nullptr;
  if (str.empty ()) return false;
  errno = 0;
  guint64 v = g_ascii_strtoull (str.c_str (), &end, 10);
  if (errno != 0 || end == str.c_str ()) return false;
  *out = (uint32_t) v;
  return true;
}

static bool
parse_f32 (const std::string &str, float *out)
{
  gchar *end = nullptr;
  if (str.empty ()) return false;
  errno = 0;
  gdouble v = g_ascii_strtod (str.c_str (), &end);
  if (errno != 0 || end == str.c_str ()) return false;
  *out = (float) v;
  return true;
}
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static bool
is_directory (const std::string &path)
{
  struct stat st;
  return (stat (path.c_str (), &st) == 0 && S_ISDIR (st.st_mode));
}

static bool
has_txt_files (const std::string &dir_path)
{
  DIR *dir = opendir (dir_path.c_str ());
  if (!dir) return false;
  struct dirent *ent;
  while ((ent = readdir (dir)) != nullptr) {
    std::string name = ent->d_name;
    if (name.size () > 4 &&
        name.substr (name.size () - 4) == ".txt") {
      closedir (dir);
      return true;
    }
  }
  closedir (dir);
  return false;
}

/* stem of a filename (no directory, no extension) */
static std::string
stem (const std::string &filename)
{
  auto slash = filename.rfind ('/');
  std::string base = (slash == std::string::npos)
      ? filename : filename.substr (slash + 1);
  auto dot = base.rfind ('.');
  return (dot == std::string::npos) ? base : base.substr (0, dot);
}

/* Safe string-member read: returns @fallback when the key is absent, the node
 * is not a value node, or the value is non-string / NULL. Prevents constructing
 * std::string from the NULL that json_object_get_string_member() returns on
 * malformed-but-shaped input. */
static std::string
safe_string_member (JsonObject *o, const gchar *key, const char *fallback)
{
  if (!o || !json_object_has_member (o, key))
    return fallback;
  JsonNode *n = json_object_get_member (o, key);
  if (!n || !JSON_NODE_HOLDS_VALUE (n))
    return fallback;
  const gchar *s = json_node_get_string (n);
  return s ? std::string (s) : std::string (fallback);
}

/* Try to parse `line` as JSON and check for NVSchema keys */
static bool
is_nvschema_line (const std::string &line)
{
  if (line.empty () || line[0] != '{') return false;
  JsonParser *p = json_parser_new ();
  GError *err = nullptr;
  gboolean ok = json_parser_load_from_data (p, line.c_str (),
      (gssize)line.size (), &err);
  bool result = false;
  if (ok) {
    JsonNode   *root = json_parser_get_root (p);
    JsonObject *obj  = json_node_get_object (root);
    if (json_object_has_member (obj, "sensor_id") ||
        json_object_has_member (obj, "sensorId")) {
      result = true;
    }
  }
  if (err) g_error_free (err);
  g_object_unref (p);
  return result;
}

/* ------------------------------------------------------------------ */
/* Format detection                                                   */
/* ------------------------------------------------------------------ */

GtFormat
gt_detect_format (const std::string &path)
{
  /* 1. Directory of .txt files -> KITTI */
  if (is_directory (path)) {
    if (has_txt_files (path))
      return GtFormat::KITTI;
    return GtFormat::UNKNOWN;
  }

  /* 2. Read first line -> NVSchema check */
  {
    std::ifstream f (path);
    if (f.is_open ()) {
      std::string line;
      if (std::getline (f, line)) {
        if (is_nvschema_line (line))
          return GtFormat::NVSCHEMA;
      }
    }
  }

  /* 3. Parse full JSON -> COCO or TAO */
  JsonParser *p = json_parser_new ();
  GError     *err = nullptr;
  gboolean    ok  = json_parser_load_from_file (p, path.c_str (), &err);
  GtFormat    fmt = GtFormat::UNKNOWN;

  if (ok) {
    JsonNode   *root = json_parser_get_root (p);
    JsonObject *obj  = json_node_get_object (root);
    if (obj &&
        json_object_has_member (obj, "images") &&
        json_object_has_member (obj, "annotations") &&
        json_object_has_member (obj, "categories")) {
      fmt = GtFormat::COCO;
    } else {
      fmt = GtFormat::TAO;
    }
  }

  if (err) g_error_free (err);
  g_object_unref (p);
  return fmt;
}

/* ------------------------------------------------------------------ */
/* KITTI loader                                                        */
/* ------------------------------------------------------------------ */

/*
 * KITTI format per line:
 *   type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry [score]
 * We take: type(0), x1(4), y1(5), x2(6), y2(7)
 * Ignore: DontCare, Misc
 *
 * Directory layout options:
 *   flat:  <dir>/<frame>.txt
 *   multi-cam: <dir>/<camera_id>/<frame>.txt
 */
static bool
load_kitti (const std::string &dir_path, GTIndex &index)
{
  DIR *dir = opendir (dir_path.c_str ());
  if (!dir) return false;

  struct dirent *ent;
  while ((ent = readdir (dir)) != nullptr) {
    std::string name = ent->d_name;
    if (name == "." || name == "..") continue;

    std::string full = dir_path + "/" + name;

    /* Sub-directory -> camera_id = subdir name, recurse one level */
    if (is_directory (full)) {
      DIR *sub = opendir (full.c_str ());
      if (!sub) continue;
      struct dirent *se;
      while ((se = readdir (sub)) != nullptr) {
        std::string sname = se->d_name;
        if (sname.size () < 5 ||
            sname.substr (sname.size () - 4) != ".txt") continue;
        std::string sfile  = full + "/" + sname;
        std::string cam_id = name;
        uint32_t    fnum   = 0;
        if (!parse_u32 (stem (sname), &fnum)) continue;
        std::ifstream f (sfile);
        std::string line;
        auto &boxes = index[{cam_id, fnum}];
        while (std::getline (f, line)) {
          std::istringstream ss (line);
          std::string tok; std::vector<std::string> toks;
          while (ss >> tok) toks.push_back (tok);
          if (toks.size () < 8) continue;
          std::string lbl = toks[0];
          if (lbl == "DontCare" || lbl == "Misc") continue;
          GTBox b;
          b.class_label = lbl;
          if (!parse_f32 (toks[4], &b.x1) || !parse_f32 (toks[5], &b.y1) ||
              !parse_f32 (toks[6], &b.x2) || !parse_f32 (toks[7], &b.y2))
            continue;
          boxes.push_back (b);
        }
      }
      closedir (sub);
      continue;
    }

    /* Flat .txt file -> camera_id = "0" */
    if (name.size () < 5 ||
        name.substr (name.size () - 4) != ".txt") continue;

    uint32_t fnum = 0;
    if (!parse_u32 (stem (name), &fnum)) continue;

    std::ifstream f (full);
    std::string line;
    auto &boxes = index[{"0", fnum}];
    while (std::getline (f, line)) {
      std::istringstream ss (line);
      std::string tok; std::vector<std::string> toks;
      while (ss >> tok) toks.push_back (tok);
      if (toks.size () < 8) continue;
      std::string lbl = toks[0];
      if (lbl == "DontCare" || lbl == "Misc") continue;
      GTBox b;
      b.class_label = lbl;
      if (!parse_f32 (toks[4], &b.x1) || !parse_f32 (toks[5], &b.y1) ||
          !parse_f32 (toks[6], &b.x2) || !parse_f32 (toks[7], &b.y2))
        continue;
      boxes.push_back (b);
    }
  }

  closedir (dir);
  return true;
}

/* ------------------------------------------------------------------ */
/* NVSchema loader                                                     */
/* ------------------------------------------------------------------ */

/*
 * JSONL format (one JSON object per line):
 * {
 *   "sensor_id": "cam-0",
 *   "frame_id": 42,
 *   "objects": [
 *     { "type": "car",
 *       "bbox": [x1, y1, x2, y2],
 *       "confidence": 0.9 }
 *   ]
 * }
 */
static bool
load_nvschema (const std::string &path, GTIndex &index)
{
  std::ifstream f (path);
  if (!f.is_open ()) return false;

  /* Reuse a single JsonParser across all lines — avoids per-line alloc/free */
  JsonParser *p = json_parser_new ();

  std::string line;
  while (std::getline (f, line)) {
    if (line.empty () || line[0] != '{') continue;

    GError *err = nullptr;
    if (!json_parser_load_from_data (p, line.c_str (),
            (gssize)line.size (), &err)) {
      if (err) g_error_free (err);
      continue;
    }

    /* Guard: root must be a JSON object, not an array or primitive */
    JsonObject *obj = json_node_get_object (json_parser_get_root (p));
    if (!obj) continue;

    /* camera id */
    std::string cam_id = "0";
    if (json_object_has_member (obj, "sensor_id"))
      cam_id = safe_string_member (obj, "sensor_id", "0");
    else if (json_object_has_member (obj, "sensorId"))
      cam_id = safe_string_member (obj, "sensorId", "0");

    /* frame id — check frame_id / frameId (int) or id (may be string) */
    uint32_t fnum = 0;
    if (json_object_has_member (obj, "frame_id"))
      fnum = (uint32_t) json_object_get_int_member (obj, "frame_id");
    else if (json_object_has_member (obj, "frameId"))
      fnum = (uint32_t) json_object_get_int_member (obj, "frameId");
    else if (json_object_has_member (obj, "id")) {
      JsonNode *id_node = json_object_get_member (obj, "id");
      if (JSON_NODE_HOLDS_VALUE (id_node)) {
        if (json_node_get_value_type (id_node) == G_TYPE_STRING) {
          const gchar *s = json_object_get_string_member (obj, "id");
          (void) parse_u32 (s, &fnum);
        } else {
          fnum = (uint32_t) json_object_get_int_member (obj, "id");
        }
      }
    }

    /* objects */
    auto &boxes = index[{cam_id, fnum}];
    if (json_object_has_member (obj, "objects")) {
      JsonArray *arr = json_object_get_array_member (obj, "objects");
      if (!arr) continue;
      guint n = json_array_get_length (arr);
      for (guint i = 0; i < n; ++i) {
        JsonObject *o = json_array_get_object_element (arr, i);
        if (!o) continue;

        std::string lbl = "unknown";
        if (json_object_has_member (o, "type")) {
          JsonNode *tn = json_object_get_member (o, "type");
          if (json_node_get_node_type (tn) != JSON_NODE_NULL) {
            const gchar *s = json_node_get_string (tn);
            if (s) lbl = s;
          }
        } else if (json_object_has_member (o, "object_type")) {
          JsonNode *tn = json_object_get_member (o, "object_type");
          if (json_node_get_node_type (tn) != JSON_NODE_NULL) {
            const gchar *s = json_node_get_string (tn);
            if (s) lbl = s;
          }
        }

        if (!json_object_has_member (o, "bbox")) continue;

        /* bbox can be an array [x1,y1,x2,y2] or an object
         * {leftX, topY, rightX, bottomY} — handle both. */
        JsonNode *bbox_node = json_object_get_member (o, "bbox");
        float bx1, by1, bx2, by2;
        if (JSON_NODE_HOLDS_ARRAY (bbox_node)) {
          JsonArray *bbox = json_node_get_array (bbox_node);
          if (json_array_get_length (bbox) < 4) continue;
          bx1 = (float) json_array_get_double_element (bbox, 0);
          by1 = (float) json_array_get_double_element (bbox, 1);
          bx2 = (float) json_array_get_double_element (bbox, 2);
          by2 = (float) json_array_get_double_element (bbox, 3);
        } else if (JSON_NODE_HOLDS_OBJECT (bbox_node)) {
          JsonObject *bo = json_node_get_object (bbox_node);
          if (json_object_has_member (bo, "leftX")) {
            /* {leftX, topY, rightX, bottomY} — corner form */
            bx1 = (float) json_object_get_double_member (bo, "leftX");
            by1 = (float) json_object_get_double_member (bo, "topY");
            bx2 = (float) json_object_get_double_member (bo, "rightX");
            by2 = (float) json_object_get_double_member (bo, "bottomY");
          } else if (json_object_has_member (bo, "left")) {
            /* {left, top, right, bottom} — corner form */
            bx1 = (float) json_object_get_double_member (bo, "left");
            by1 = (float) json_object_get_double_member (bo, "top");
            bx2 = (float) json_object_get_double_member (bo, "right");
            by2 = (float) json_object_get_double_member (bo, "bottom");
          } else if (json_object_has_member (bo, "x")) {
            /* {x, y, width, height} — origin+size form */
            float x = (float) json_object_get_double_member (bo, "x");
            float y = (float) json_object_get_double_member (bo, "y");
            float w = (float) json_object_get_double_member (bo, "width");
            float h = (float) json_object_get_double_member (bo, "height");
            bx1 = x; by1 = y; bx2 = x + w; by2 = y + h;
          } else {
            continue;
          }
        } else {
          continue;
        }

        GTBox b;
        b.class_label = lbl;
        b.x1 = bx1; b.y1 = by1; b.x2 = bx2; b.y2 = by2;
        boxes.push_back (b);
      }
    }
  }

  g_object_unref (p);
  return true;
}

/* ------------------------------------------------------------------ */
/* COCO loader                                                         */
/* ------------------------------------------------------------------ */

/*
 * Standard COCO JSON:
 * {
 *   "images":      [{"id":1, "file_name":"000001.jpg", ...}],
 *   "categories":  [{"id":1, "name":"car"}, ...],
 *   "annotations": [{"image_id":1, "category_id":1,
 *                    "bbox":[x,y,w,h], ...}]
 * }
 * camera_id = "0" (COCO is single-dataset).
 * frame_num = image id (or parsed from file_name stem if numeric).
 */
static bool
load_coco (const std::string &path, GTIndex &index)
{
  JsonParser *p   = json_parser_new ();
  GError     *err = nullptr;
  if (!json_parser_load_from_file (p, path.c_str (), &err)) {
    if (err) g_error_free (err);
    g_object_unref (p);
    return false;
  }

  JsonObject *root = json_node_get_object (json_parser_get_root (p));
  if (!root) {
    g_object_unref (p);
    return false;
  }

  /* category_id -> name */
  std::map<int64_t, std::string> cat_name;
  JsonArray *cats = json_object_get_array_member (root, "categories");
  if (!cats) { g_object_unref (p); return false; }
  for (guint i = 0; i < json_array_get_length (cats); ++i) {
    JsonObject *c = json_array_get_object_element (cats, i);
    if (!c) continue;
    int64_t id    = json_object_get_int_member (c, "id");
    cat_name[id]  = safe_string_member (c, "name", "unknown");
  }

  /* image_id -> frame_num (try parsing stem of file_name, else use id) */
  std::map<int64_t, uint32_t> img_frame;
  JsonArray *imgs = json_object_get_array_member (root, "images");
  if (!imgs) { g_object_unref (p); return false; }
  for (guint i = 0; i < json_array_get_length (imgs); ++i) {
    JsonObject *img = json_array_get_object_element (imgs, i);
    if (!img) continue;
    int64_t id      = json_object_get_int_member (img, "id");
    uint32_t fnum   = (uint32_t)id;
    if (json_object_has_member (img, "file_name")) {
      std::string s = stem (safe_string_member (img, "file_name", ""));
      (void) parse_u32 (s, &fnum);
    }
    img_frame[id] = fnum;
  }

  /* annotations */
  JsonArray *anns = json_object_get_array_member (root, "annotations");
  if (!anns) { g_object_unref (p); return false; }
  for (guint i = 0; i < json_array_get_length (anns); ++i) {
    JsonObject *ann = json_array_get_object_element (anns, i);
    if (!ann) continue;
    int64_t img_id  = json_object_get_int_member (ann, "image_id");
    int64_t cat_id  = json_object_get_int_member (ann, "category_id");

    auto fit = img_frame.find (img_id);
    if (fit == img_frame.end ()) continue;

    auto cit = cat_name.find (cat_id);
    std::string lbl = (cit != cat_name.end ()) ? cit->second : "unknown";

    JsonArray *bbox = json_object_get_array_member (ann, "bbox");
    if (!bbox || json_array_get_length (bbox) < 4) continue;

    float x = (float)json_array_get_double_element (bbox, 0);
    float y = (float)json_array_get_double_element (bbox, 1);
    float w = (float)json_array_get_double_element (bbox, 2);
    float h = (float)json_array_get_double_element (bbox, 3);

    GTBox b;
    b.class_label = lbl;
    b.x1 = x;     b.y1 = y;
    b.x2 = x + w; b.y2 = y + h;
    index[{"0", fit->second}].push_back (b);
  }

  g_object_unref (p);
  return true;
}

/* ------------------------------------------------------------------ */
/* TAO JSON loader                                                     */
/* ------------------------------------------------------------------ */

/*
 * TAO format variants:
 *
 * Variant A (multi-camera):
 * {
 *   "cam-0": {
 *     "0": [{"object_type":"car","2d_bounding_box":[x1,y1,x2,y2],...}],
 *     "1": [...]
 *   },
 *   "cam-1": { ... }
 * }
 *
 * Variant B (single-camera, flat):
 * {
 *   "0": [{"object_type":"car","2d_bounding_box":[x1,y1,x2,y2],...}],
 *   "1": [...]
 * }
 *
 * Detection of variant: if the first value in the root object is itself
 * a JSON object (not an array) -> multi-camera; else flat.
 *
 * BBox: array [x1,y1,x2,y2] OR object {x,y,width,height}.
 */

static GTBox
parse_tao_object (JsonObject *o)
{
  GTBox b;
  b.class_label = "unknown";
  b.x1 = b.y1 = b.x2 = b.y2 = 0.0f;

  if (json_object_has_member (o, "object_type")) {
    JsonNode *tn = json_object_get_member (o, "object_type");
    if (json_node_get_node_type (tn) != JSON_NODE_NULL) { const gchar *s = json_node_get_string (tn); if (s) b.class_label = s; }
  } else if (json_object_has_member (o, "type")) {
    JsonNode *tn = json_object_get_member (o, "type");
    if (json_node_get_node_type (tn) != JSON_NODE_NULL) { const gchar *s = json_node_get_string (tn); if (s) b.class_label = s; }
  }

  const gchar *bbox_key = json_object_has_member (o, "2d_bounding_box")
      ? "2d_bounding_box" : (json_object_has_member (o, "bbox") ? "bbox" : nullptr);

  if (!bbox_key) return b;

  JsonNode *bnode = json_object_get_member (o, bbox_key);
  if (JSON_NODE_HOLDS_ARRAY (bnode)) {
    JsonArray *arr = json_node_get_array (bnode);
    if (json_array_get_length (arr) >= 4) {
      b.x1 = (float)json_array_get_double_element (arr, 0);
      b.y1 = (float)json_array_get_double_element (arr, 1);
      b.x2 = (float)json_array_get_double_element (arr, 2);
      b.y2 = (float)json_array_get_double_element (arr, 3);
    }
  } else if (JSON_NODE_HOLDS_OBJECT (bnode)) {
    JsonObject *bo = json_node_get_object (bnode);
    float x = (float)json_object_get_double_member (bo, "x");
    float y = (float)json_object_get_double_member (bo, "y");
    float w = (float)json_object_get_double_member (bo, "width");
    float h = (float)json_object_get_double_member (bo, "height");
    b.x1 = x; b.y1 = y; b.x2 = x + w; b.y2 = y + h;
  }
  return b;
}

static void
load_tao_frame_array (JsonArray *arr, const std::string &cam_id,
                      uint32_t fnum, GTIndex &index)
{
  auto &boxes = index[{cam_id, fnum}];
  for (guint i = 0; i < json_array_get_length (arr); ++i) {
    JsonObject *o = json_array_get_object_element (arr, i);
    if (o) boxes.push_back (parse_tao_object (o));
  }
}

static bool
load_tao (const std::string &path, GTIndex &index)
{
  JsonParser *p   = json_parser_new ();
  GError     *err = nullptr;
  if (!json_parser_load_from_file (p, path.c_str (), &err)) {
    if (err) g_error_free (err);
    g_object_unref (p);
    return false;
  }

  JsonNode   *root_node = json_parser_get_root (p);
  JsonObject *root      = json_node_get_object (root_node);
  if (!root) {
    g_object_unref (p);
    return false;
  }

  /* Determine variant by checking whether first member is object or array */
  GList *members = json_object_get_members (root);
  bool multi_cam = false;
  if (members) {
    JsonNode *first = json_object_get_member (root, (const gchar *)members->data);
    multi_cam = JSON_NODE_HOLDS_OBJECT (first);
    g_list_free (members);
  }

  if (multi_cam) {
    /* Variant A: root[cam_id][frame_str] = array */
    GList *cams = json_object_get_members (root);
    for (GList *cl = cams; cl; cl = cl->next) {
      const gchar *cam = (const gchar *)cl->data;
      JsonObject  *cam_obj = json_object_get_object_member (root, cam);
      if (!cam_obj) continue;
      GList       *frames  = json_object_get_members (cam_obj);
      for (GList *fl = frames; fl; fl = fl->next) {
        const gchar *fstr = (const gchar *)fl->data;
        uint32_t fnum = 0;
        (void) parse_u32 (fstr, &fnum);
        JsonArray *arr = json_object_get_array_member (cam_obj, fstr);
        if (arr) load_tao_frame_array (arr, std::string (cam), fnum, index);
      }
      g_list_free (frames);
    }
    g_list_free (cams);
  } else {
    /* Variant B: root[frame_str] = array */
    GList *frames = json_object_get_members (root);
    for (GList *fl = frames; fl; fl = fl->next) {
      const gchar *fstr = (const gchar *)fl->data;
      uint32_t fnum = 0;
      (void) parse_u32 (fstr, &fnum);
      JsonArray *arr = json_object_get_array_member (root, fstr);
      if (arr) load_tao_frame_array (arr, "0", fnum, index);
    }
    g_list_free (frames);
  }

  g_object_unref (p);
  return true;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

bool
gt_load (const std::string &path, GTIndex &index, GtFormat fmt)
{
  if (fmt == GtFormat::UNKNOWN)
    fmt = gt_detect_format (path);

  switch (fmt) {
    case GtFormat::KITTI:    return load_kitti    (path, index);
    case GtFormat::NVSCHEMA: return load_nvschema (path, index);
    case GtFormat::COCO:     return load_coco     (path, index);
    case GtFormat::TAO:      return load_tao      (path, index);
    default:
      g_warning ("nvinfereval: unknown GT format for '%s'", path.c_str ());
      return false;
  }
}
