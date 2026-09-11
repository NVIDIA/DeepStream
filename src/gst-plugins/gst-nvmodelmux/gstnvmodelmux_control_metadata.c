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
 * Stream metadata parsing for control-plane stream/add and stream/bind events.
 * This file intentionally contains no pad, bin, routing or main-loop mutation.
 */

#include "gstnvmodelmux.h"

#include <json-glib/json-glib.h>
#include <string.h>

GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat

/* Upper bound on an untrusted per-stream metadata JSON blob (stream/add). */
#define MM_MAX_METADATA_LEN (64 * 1024)

/* Read an OPTIONAL string member as a fresh g_strdup, but ONLY when the member
 * is actually a string value. A present-but-non-string field (e.g.
 * "name": 123) would otherwise make json_object_get_string_member()
 * emit a g_critical, which also trips the test "no CRITICAL" sanity check. */
static gchar *
modelmux_control_json_str (JsonObject * obj, const gchar * key)
{
  JsonNode *n;
  if (!json_object_has_member (obj, key))
    return NULL;
  n = json_object_get_member (obj, key);
  if (!n || !JSON_NODE_HOLDS_VALUE (n) ||
      json_node_get_value_type (n) != G_TYPE_STRING)
    return NULL;
  return g_strdup (json_node_get_string (n));
}

/* GPU ids are bounded 0..MM_META_GPU_ID_MAX everywhere in this API (mirrors
 * MODELMUX_CONTROL_GPU_ID_MAX in gstnvmodelmux_control.c and the REST parser's GPU_ID_MAX):
 * a huge 64-bit JSON value must never TRUNCATE into a valid-looking device id
 * at this boundary either (stream/add metadata can be app-generated). */
#define MM_META_GPU_ID_MAX 255

/* Read a nested {"name","version"[,"gpu"]} model ref member. A PRESENT but
 * out-of-range gpu reads as the G_MAXINT sentinel -- an id no device can ever
 * match -- so the caller's admission (modelmux_control_check_ref_gpu) falls this ref back
 * to the role default with a distinct 'gpu id out of range' warning instead
 * of silently truncating to a small valid id. *malformed (optional) reads
 * TRUE when the member is PRESENT but neither null nor an object (e.g.
 * "model": "Car;2"): the caller must warn + ignore the ref loudly rather
 * than silently treat it as "nothing requested". */
static void
modelmux_control_json_ref_member (JsonObject * obj, const gchar * key, gchar ** name,
    gchar ** version, gint * gpu, gboolean * is_null, gboolean * malformed)
{
  JsonNode *n;

  *name = *version = NULL;
  if (gpu)
    *gpu = -1;
  if (is_null)
    *is_null = FALSE;
  if (malformed)
    *malformed = FALSE;
  if (!json_object_has_member (obj, key))
    return;
  n = json_object_get_member (obj, key);
  if (!n)
    return;
  if (JSON_NODE_HOLDS_NULL (n)) {
    if (is_null)
      *is_null = TRUE;
    return;
  }
  if (JSON_NODE_HOLDS_OBJECT (n)) {
    JsonObject *ref = json_node_get_object (n);
    *name = modelmux_control_json_str (ref, "name");
    *version = modelmux_control_json_str (ref, "version");
    if (gpu && json_object_has_member (ref, "gpu")) {
      JsonNode *g = json_object_get_member (ref, "gpu");
      if (g && JSON_NODE_HOLDS_VALUE (g) &&
          json_node_get_value_type (g) == G_TYPE_INT64) {
        gint64 gv = json_node_get_int (g);
        *gpu = (gv < 0 || gv > MM_META_GPU_ID_MAX) ? G_MAXINT : (gint) gv;
      }
    }
  } else if (malformed) {
    *malformed = TRUE;                    /* present, not null, not object */
  }
}

/* Parse per-stream model selection from a stream/add metadata JSON blob (v1):
 *   { "model":  {"name": ..., "version": ...[, "gpu": N]},
 *     "shadow": {"name", "version"[, "gpu"]} | null }
 * "shadow": null opts the stream OUT of the default shadow (surfaced to the
 * caller as the internal "none" clear token). All outputs default to
 * NULL/-1. Caller frees the strings. */
G_GNUC_INTERNAL void
modelmux_control_parse_model_metadata (GstNvModelMux * self, const gchar * json,
    gchar ** primary_model_name, gchar ** primary_model_version, gint * primary_model_gpu,
    gchar ** shadow_model_name, gchar ** shadow_model_version, gint * shadow_model_gpu)
{
  JsonParser *parser;
  JsonNode *root;
  GError *err = NULL;

  *primary_model_name = *primary_model_version = *shadow_model_name = *shadow_model_version = NULL;
  if (primary_model_gpu) *primary_model_gpu = -1;
  if (shadow_model_gpu) *shadow_model_gpu = -1;
  if (!json || !*json)
    return;

  /* Bound untrusted REST metadata before handing it to json-glib, which has
   * no built-in size/depth cap. 64 KiB is far above legitimate metadata. */
  if (strlen (json) > MM_MAX_METADATA_LEN) {
    GST_WARNING_OBJECT (self, "stream metadata JSON too large (%zu bytes > %u) -- ignoring",
        strlen (json), (guint) MM_MAX_METADATA_LEN);
    return;
  }

  parser = json_parser_new ();
  if (!json_parser_load_from_data (parser, json, -1, &err)) {
    GST_WARNING_OBJECT (self, "could not parse stream metadata JSON: %s",
        err ? err->message : "?");
    g_clear_error (&err);
    g_object_unref (parser);
    return;
  }
  root = json_parser_get_root (parser);
  if (root && JSON_NODE_HOLDS_OBJECT (root)) {
    JsonObject *obj = json_node_get_object (root);
    gboolean s_null = FALSE;
    gboolean p_malformed = FALSE, s_malformed = FALSE;

    modelmux_control_json_ref_member (obj, "model", primary_model_name, primary_model_version, primary_model_gpu, NULL,
        &p_malformed);
    modelmux_control_json_ref_member (obj, "shadow", shadow_model_name, shadow_model_version, shadow_model_gpu, &s_null,
        &s_malformed);
    /* a wrong-typed ref (e.g. "model": "Car;2" or 123) decodes to NULL
     * name/version, which the attach path would silently read as "nothing
     * requested" (-> role default). Metadata is advisory (the stream itself
     * is fine), so fall back LOUDLY instead of rejecting the stream. */
    if (p_malformed)
      GST_WARNING_OBJECT (self, "stream metadata: \"model\" must be a "
          "{\"name\",\"version\"} object or null -- ignoring the ref "
          "(falling back to the default primary)");
    if (s_malformed)
      GST_WARNING_OBJECT (self, "stream metadata: \"shadow\" must be a "
          "{\"name\",\"version\"} object or null -- ignoring the ref "
          "(falling back to the default shadow)");
    if (s_null)
      *shadow_model_name = g_strdup ("none");   /* internal shadow-clear token */
  }
  g_object_unref (parser);
}
