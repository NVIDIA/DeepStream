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

/* GKeyFile-based config parser (standard DeepStream sample-app idiom). */

#include "gstnvmodelmux.h"   /* GstNvModelMux + PROP_* (property reconciliation) */
#include "gstnvmodelmux_priv.h"
#include <stdio.h>    /* bounded prefix read in modelmux_detect_model_type */
#include <glib/gstdio.h>  /* g_stat (derived-config size gate) */
#include <string.h>

/* Share the plugin's debug category (registered in plugin_init) so config
 * diagnostics obey GST_DEBUG like the rest of the element -- no bare prints. */
GST_DEBUG_CATEGORY_EXTERN (gst_modelmux_debug_cat);
#define GST_CAT_DEFAULT gst_modelmux_debug_cat
#define CFG_WARN(fmt, ...) GST_WARNING (fmt, ##__VA_ARGS__)
#define CFG_INFO(fmt, ...) GST_INFO (fmt, ##__VA_ARGS__)

static gchar *
cfg_str (GKeyFile * kf, const gchar * grp, const gchar * key, const gchar * def)
{
  gchar *v = g_key_file_get_string (kf, grp, key, NULL);
  if (!v && def)
    v = g_strdup (def);
  return v;
}

static gint
cfg_int (GKeyFile * kf, const gchar * grp, const gchar * key, gint def)
{
  GError *err = NULL;
  gint v = g_key_file_get_integer (kf, grp, key, &err);
  if (err) {
    g_clear_error (&err);
    return def;
  }
  return v;
}

static guint
cfg_uint (GKeyFile * kf, const gchar * grp, const gchar * key, guint def)
{
  GError *err = NULL;
  gint v = g_key_file_get_integer (kf, grp, key, &err);
  if (err) {
    g_error_free (err);
    return def;
  }
  if (v < 0) {
    CFG_WARN ("config [%s] key '%s' = %d is negative -- using default %u",
        grp, key, v, def);
    return def;
  }
  return (guint) v;
}

static gboolean
cfg_bool (GKeyFile * kf, const gchar * grp, const gchar * key, gboolean def)
{
  GError *err = NULL;
  gboolean v = g_key_file_get_boolean (kf, grp, key, &err);
  if (err) {
    g_error_free (err);
    return def;
  }
  return v;
}

/* Model versions are POSITIVE INTEGERS (Triton model-repository convention). Valid:
 * "1","2",...,"42". Invalid: NULL/"", "0", leading-zero ("01"), "V1.0", "1.0", negatives,
 * any non-digit. */
gboolean
modelmux_version_is_int (const gchar * v)
{
  const gchar *p;
  if (!v || !*v || v[0] == '0')          /* reject empty, "0", and leading-zero forms */
    return FALSE;
  for (p = v; *p; p++)
    if (!g_ascii_isdigit (*p))
      return FALSE;
  return TRUE;
}

/* Parse the [model-name-<idx>] CATALOG blocks. Each block defines a model NAME and its
 * base config-file (backend auto-detected); it is registered into config->models[] as
 * metadata only -- NOTHING is instantiated until a role ref (primary/shadow-model) or the
 * model/load API loads a concrete (name, version) instance from it. Must run before any
 * role ref is parsed (refs validate their name against the catalog). Requires
 * config->per_model_max_streams to be set first (used as the per-model cap). */
static void
cfg_parse_catalog (ModelMuxConfig * config, GKeyFile * kf)
{
  static const gchar *PFX = "model-name-";
  gchar **groups = g_key_file_get_groups (kf, NULL);
  guint gi;
  for (gi = 0; groups && groups[gi]; gi++) {
    const gchar *grp = groups[gi];
    gchar *name, *conf;
    if (!g_str_has_prefix (grp, PFX))
      continue;
    guint mbatch;
    name = cfg_str (kf, grp, "name", NULL);
    conf = cfg_str (kf, grp, "config", NULL);
    if (!name || !*name || !conf || !*conf) {
      CFG_WARN ("[%s] catalog block needs both name= and config= -- ignored", grp);
      g_free (name); g_free (conf);
      continue;
    }
    /* optional per-NAME batch/stream cap: sizes this model's per-instance mux +
     * nvinfer batch (overrides [multimodel] per-model-batch-size for THIS model;
     * a [model-<name>-<version>] block or model/load batch_size overrides it per
     * version). 0/absent = the global default. */
    mbatch = cfg_uint (kf, grp, "batch-size", 0);
    if (mbatch > MM_MAX_BATCH) {
      CFG_WARN ("[%s] batch-size %u exceeds max %u -- clamping", grp, mbatch,
          (guint) MM_MAX_BATCH);
      mbatch = MM_MAX_BATCH;
    }
    modelmux_config_register_model (config, name, conf, NULL, NULL, mbatch);
    g_free (name); g_free (conf);
  }
  g_strfreev (groups);
}

/* Parse the [model-<name>-<version>] PER-VERSION blocks: optional engine and/or custom config
 * for a specific (name, version). Declared ONCE; consulted at instantiation by every ref
 * (default / per-sensor / load). MUST run AFTER cfg_parse_catalog (validates the name against
 * the catalog). Disambiguation: [model-name-<digits>] is the catalog-index form (keyword
 * 'name' reserved) and is skipped here; [model-<othername>-<digits>] is a per-version block. */
static void
cfg_parse_per_version (ModelMuxConfig * config, GKeyFile * kf)
{
  static const gchar *PFX = "model-";
  gchar **groups = g_key_file_get_groups (kf, NULL);
  guint gi;
  for (gi = 0; groups && groups[gi]; gi++) {
    const gchar *grp = groups[gi], *rest, *dash;
    gchar *name = NULL, *ver = NULL, *eng = NULL, *conf = NULL;
    gint gpu = -1, vgie = -1;
    guint vbatch = 0;
    ModelVersionArtifact *v;

    if (!g_str_has_prefix (grp, PFX))
      continue;
    rest = grp + strlen (PFX);                       /* "<name>-<version>" (or "name-<idx>") */
    /* Concede EVERY [model-name-*] group to cfg_parse_catalog, which claims them all on the
     * prefix alone. Testing the suffix with modelmux_version_is_int() instead would disagree
     * with it at index 0 -- versions are positive by definition, catalog indices start at 0 --
     * so the [model-name-0] every config opens with fell through here and was mis-read as a
     * per-version block for a model called "name". The keyword 'name' is reserved (documented
     * above), which is what makes this split unambiguous. */
    if (g_str_has_prefix (rest, "name-"))
      continue;                                      /* catalog block -> handled elsewhere */
    dash = strrchr (rest, '-');                      /* split at the LAST dash: name | version */
    if (!dash || dash == rest || !*(dash + 1))
      continue;                                      /* not a per-version block shape */
    name = g_strndup (rest, (gsize) (dash - rest));
    ver  = g_strdup (dash + 1);

    if (!modelmux_version_is_int (ver)) {
      CFG_WARN ("[%s] per-version block: version '%s' is not a positive integer -- ignored",
          grp, ver);
      goto skip;
    }
    if (!modelmux_config_find_model (config, name)) {
      CFG_WARN ("[%s] per-version block names '%s' which is not in any [model-name-*] catalog "
          "block -- ignored", grp, name);
      goto skip;
    }
    eng  = cfg_str (kf, grp, "engine", NULL);
    conf = cfg_str (kf, grp, "config", NULL);
    gpu  = cfg_int (kf, grp, "gpu", -1);
    vgie = cfg_int (kf, grp, "gie-unique-id", -1);  /* -1 = inherit the base config's */
    if (vgie == 0) {
      CFG_WARN ("[%s] gie-unique-id=0 is nvinfer's 'unset' -- ignoring", grp);
      vgie = -1;
    }
    vbatch = cfg_uint (kf, grp, "batch-size", 0);   /* 0 = inherit name/global */
    if (vbatch > MM_MAX_BATCH) {
      CFG_WARN ("[%s] batch-size %u exceeds max %u -- clamping", grp, vbatch,
          (guint) MM_MAX_BATCH);
      vbatch = MM_MAX_BATCH;
    }
    if (eng && !*eng)  { g_free (eng);  eng = NULL; }
    if (conf && !*conf) { g_free (conf); conf = NULL; }
    if (gpu < -1) {
      CFG_WARN ("[%s] gpu=%d is not a valid device id -- ignoring the placement", grp, gpu);
      gpu = -1;
    }
    if (eng && !g_file_test (eng, G_FILE_TEST_IS_REGULAR)) {
      CFG_WARN ("[%s] engine not found: '%s' -- ignoring it (using base config)", grp, eng);
      g_free (eng); eng = NULL;
    }
    if (conf && !g_file_test (conf, G_FILE_TEST_IS_REGULAR)) {
      CFG_WARN ("[%s] config not found: '%s' -- ignoring it (using base config)", grp, conf);
      g_free (conf); conf = NULL;
    }
    if (!eng && !conf && gpu < 0 && !vbatch && vgie < 0) {
      CFG_WARN ("[%s] per-version block has no usable engine=, config=, gpu=, "
          "gie-unique-id= or batch-size= -- ignored (version would just reuse the "
          "model's base config)", grp);
      goto skip;
    }
    if (config->num_versions >= G_N_ELEMENTS (config->versions)) {
      CFG_WARN ("[%s] too many per-version blocks (max %u) -- ignored",
          grp, (guint) G_N_ELEMENTS (config->versions));
      goto skip;
    }
    v = &config->versions[config->num_versions++];
    v->name = name; v->version = ver; v->engine = eng; v->config = conf;  /* ownership transferred */
    v->gpu = gpu;                      /* optional gpu= placement (-1 = config's device) */
    v->gie = vgie;                     /* optional gie-unique-id (-1 = base config's) */
    v->batch = vbatch;                 /* optional per-version batch (0 = inherit) */
    CFG_INFO ("per-version artifact: '%s@%s'  engine=%s  config=%s  gpu=%d  gie=%d  batch=%u",
        name, ver, eng ? eng : "(base)", conf ? conf : "(base)", gpu, vgie, vbatch);
    continue;
skip:
    g_free (name); g_free (ver); g_free (eng); g_free (conf);
  }
  g_strfreev (groups);
}

/* Compile a config-key regex ONCE, process-wide.
 *
 * These readers are reached from the REST/query thread (model/load and model/update at
 * a secondary run the whole admission inline on a civetweb worker), which arrives here
 * roughly 130 GStreamer query-forwarding frames deep on a worker stack far smaller than
 * the main thread's. g_regex_new -> pcre2_compile wants a big frame and used to blow
 * straight through the guard page: a SIGSEGV inside pcre2_compile, nowhere near any
 * modelmux pointer. Compiling on first use (the pipeline thread, at build) keeps the
 * REST path allocation-light and cheap. A GRegex is refcounted and immutable once
 * built, and g_regex_match allocates its own match data, so sharing one across threads
 * is safe -- never unref these. */
static GRegex *
cfg_key_regex (GRegex ** slot, const gchar * pattern)
{
  static GMutex lock;
  g_mutex_lock (&lock);
  if (!*slot)
    *slot = g_regex_new (pattern, G_REGEX_MULTILINE, 0, NULL);
  g_mutex_unlock (&lock);
  return *slot;
}

/* Read `process-mode` out of a model's infer config -- the same file, and the same
 * technique, as the gpu reader below. A leading '#' is rejected so a commented-out
 * example does not produce a bogus value. Returns -1 when the key is absent or the
 * file cannot be read. */
gint
modelmux_config_file_process_mode (const gchar * config_file)
{
  static GRegex *cached = NULL;
  gchar *content = NULL;
  gint mode = -1;
  GRegex *re;
  GMatchInfo *mi = NULL;
  if (!config_file || !g_file_get_contents (config_file, &content, NULL, NULL))
    return -1;
  re = cfg_key_regex (&cached,
      "^[ \\t]*process[-_]mode[ \\t]*[:=][ \\t]*(\\d+)");
  if (re && g_regex_match (re, content, 0, &mi)) {
    gchar *num = g_match_info_fetch (mi, 1);
    if (num)
      mode = (gint) g_ascii_strtoll (num, NULL, 10);
    g_free (num);
  }
  if (mi)
    g_match_info_free (mi);
  g_free (content);
  return mode;
}

/* The gie-unique-id a model's infer config DECLARES. This is the identity nvinfer
 * stamps into every object it produces, and therefore the id a downstream secondary
 * names in `operate-on-gie-id`. Reading it here lets a modelmux stage adopt the
 * CONFIGURED id instead of an internally-allocated one, so the nvinfer contract
 * ("declare 1, then operate-on-gie-id=1") holds unchanged when a modelmux replaces a
 * plain nvinfer. Returns -1 when absent; the caller then falls back to the monotonic
 * allocator. Same file and technique as the process-mode/gpu readers above. */
gint
modelmux_config_file_gie_id (const gchar * config_file)
{
  static GRegex *cached = NULL;
  gchar *content = NULL;
  gint gie = -1;
  GRegex *re;
  GMatchInfo *mi = NULL;
  if (!config_file || !g_file_get_contents (config_file, &content, NULL, NULL))
    return -1;
  re = cfg_key_regex (&cached,
      "^[ \\t]*gie[-_]unique[-_]id[ \\t]*[:=][ \\t]*(\\d+)");
  if (re && g_regex_match (re, content, 0, &mi)) {
    gchar *num = g_match_info_fetch (mi, 1);
    if (num)
      gie = (gint) g_ascii_strtoll (num, NULL, 10);
    g_free (num);
  }
  if (mi)
    g_match_info_free (mi);
  g_free (content);
  return gie > 0 ? gie : -1;                 /* 0 is nvinfer's "unset", not an id */
}

/* The gie-unique-id a SECONDARY's infer config says it operates ON -- nvinfer's
 * `operate-on-gie-id`. This is the input filter: the classifier only sees objects whose
 * unique_component_id matches, so it is also the only honest answer to "which stage's
 * output did this model actually consume". -1 = absent, which is nvinfer's default
 * meaning EVERY GIE. Same file and technique as the readers above. */
gint
modelmux_config_file_operate_on_gie (const gchar * config_file)
{
  static GRegex *cached = NULL;
  gchar *content = NULL;
  gint gie = -1;
  GRegex *re;
  GMatchInfo *mi = NULL;
  if (!config_file || !g_file_get_contents (config_file, &content, NULL, NULL))
    return -1;
  re = cfg_key_regex (&cached,
      "^[ \\t]*operate[-_]on[-_]gie[-_]id[ \\t]*[:=][ \\t]*(-?\\d+)");
  if (re && g_regex_match (re, content, 0, &mi)) {
    gchar *num = g_match_info_fetch (mi, 1);
    if (num)
      gie = (gint) g_ascii_strtoll (num, NULL, 10);
    g_free (num);
  }
  if (mi)
    g_match_info_free (mi);
  g_free (content);
  return gie;                        /* -1 (explicit or absent) = all GIEs */
}

/* Device a model CONFIG FILE pins its inference to: nvinfer keyfile "gpu-id=N"
 * ([property]) or nvinferserver pbtxt "gpu_ids: [N]". -1 = no key present (the
 * backend then defaults to device 0). Parsed once per bin at creation and used as
 * the LAST fallback of effective-placement resolution (own gpu -> registry -> this). */
gint
modelmux_config_file_gpu (const gchar * config_file)
{
  static GRegex *cached = NULL;
  gchar *content = NULL;
  gint gpu = -1;
  GRegex *re;
  GMatchInfo *mi = NULL;
  if (!config_file || !g_file_get_contents (config_file, &content, NULL, NULL))
    return -1;
  re = cfg_key_regex (&cached,
      "^\\s*gpu[-_]ids?\\s*[:=]\\s*\\[?\\s*(\\d+)\\s*(,?)");
  if (re && g_regex_match (re, content, 0, &mi)) {
    gchar *num = g_match_info_fetch (mi, 1);
    gchar *more = g_match_info_fetch (mi, 2);
    /* a LIST ("gpu_ids: [0, 1]" -- Triton multi-GPU instance group) is NOT a
     * single-device pin: treating it as "gpu 0" would mis-place the ingress
     * migration and mis-report the effective device. No pin (-1). */
    if (num && !(more && *more == ','))
      gpu = (gint) g_ascii_strtoll (num, NULL, 10);
    g_free (num);
    g_free (more);
  }
  if (mi)
    g_match_info_free (mi);
  g_free (content);
  return gpu;
}

/* Resolve the EFFECTIVE config for a (name, version) from its per-version artifact (if any):
 * custom config and/or an engine swapped in. Returns a newly-allocated config path (caller
 * g_free) when an override applies, else NULL (caller uses the catalog base config as-is). */
gchar *
modelmux_config_resolve_version_cfg (const ModelMuxConfig * config, const gchar * name, const gchar * version,
    gboolean * derive_failed)
{
  const ModelCatalogEntry *def;
  const gchar *base, *ver = (version && *version) ? version : MM_MODEL_VERSION_DEFAULT;
  gchar *v_engine = NULL, *v_config = NULL, *ret = NULL;
  gint v_gpu = -1, v_gie = -1;
  gboolean found = FALSE;
  guint i;

  if (derive_failed)
    *derive_failed = FALSE;
  if (!config || !name || !*name)
    return NULL;
  /* versions[] is runtime-mutable (register on load/update, unregister on OTA
   * rollback) -- scan + COPY OUT under versions_lock, then derive (file I/O)
   * strictly after unlock. Never nest another lock inside (see priv.h). */
  g_mutex_lock ((GMutex *) &config->versions_lock);
  for (i = 0; i < config->num_versions; i++)
    if (g_strcmp0 (config->versions[i].name, name) == 0 &&
        g_strcmp0 (config->versions[i].version, ver) == 0) {
      v_engine = g_strdup (config->versions[i].engine);
      v_config = g_strdup (config->versions[i].config);
      v_gpu = config->versions[i].gpu;
      v_gie = config->versions[i].gie;
      found = TRUE;
      break;
    }
  g_mutex_unlock ((GMutex *) &config->versions_lock);
  if (!found)
    return NULL;                                     /* no override -> caller uses catalog base */
  def = modelmux_config_find_model (config, name);
  base = v_config ? v_config : (def ? def->config_file : NULL);
  if (base) {
    if (v_engine || v_gpu >= 0 || v_gie > 0) {
      gchar *derived = modelmux_config_derive (base, v_engine, v_gpu, v_gie, name, ver);
      if (derived) {
        ret = derived;
      } else if (v_gpu < 0 &&
          !modelmux_backend (modelmux_detect_model_type (base))->engine_file) {
        /* inferserver cannot swap an engine via config rewrite (documented
         * limitation, warned inside derive) -- the base config IS still the
         * correct artifact, so this is a legitimate fallback, not a failure. */
        ret = g_strdup (base);
      } else {
        /* explicit engine/gpu override that FAILED to derive: the base config
         * would run the wrong artifact/device while status reports the
         * override took. Signal failure; the caller must refuse to warm. */
        if (derive_failed)
          *derive_failed = TRUE;
        ret = NULL;
      }
    } else {
      ret = g_strdup (base);                         /* custom config, no engine swap */
    }
  }
  g_free (v_engine);
  g_free (v_config);
  return ret;
}

/* Parse a "name;version" model REFERENCE (primary-model / shadow-model form, same for the
 * [multimodel] defaults and the per-sensor [stream-model-*] bindings). The name MUST already
 * be defined in a [model-name-<idx>] catalog block. Returns the model NAME (newly-allocated;
 * caller frees) or NULL when empty / unknown; *version_out receives the version (defaults to
 * MM_MODEL_VERSION_DEFAULT when omitted).
 *
 * A ref carries NO engine/config: a version's checkpoint/config is declared ONCE in its
 * [model-<name>-<version>] block and resolved at instantiation (modelmux_config_resolve_version_cfg),
 * so every ref stays pure name;version with no duplication. A stray 3rd field is warned + dropped
 * (point the user at the per-version block). */
static gchar *
cfg_parse_ref (ModelMuxConfig * config, const gchar * grp, const gchar * spec,
    gchar ** version_out)
{
  gchar **f = NULL;
  gchar *name = NULL, *ver = NULL, *out = NULL;

  if (version_out)
    *version_out = NULL;
  if (!spec || !*spec)
    return NULL;

  f = g_strsplit (spec, ";", 3);
  name = (f[0] && *f[0]) ? g_strstrip (f[0]) : NULL;
  ver  = (f[1] && *f[1]) ? g_strstrip (f[1]) : NULL;

  if (!name || !*name) {
    CFG_WARN ("[%s] model ref '%s' has no model name -- ignored", grp, spec);
    g_strfreev (f);
    return NULL;
  }
  if (!modelmux_config_find_model (config, name)) {
    CFG_WARN ("[%s] model ref '%s' names '%s' which is not defined in any "
        "[model-name-*] catalog block -- ignored", grp, spec, name);
    g_strfreev (f);
    return NULL;
  }
  if (f[2] && *g_strstrip (f[2])) {
    CFG_WARN ("[%s] model ref '%s' has a 3rd field ('%s') -- model refs are now pure "
        "name;version. Declare a version's engine/config in its [model-%s-%s] block instead. "
        "IGNORING the 3rd field.", grp, spec, f[2], name,
        (ver && *ver) ? ver : MM_MODEL_VERSION_DEFAULT);
  }
  out = g_strdup (name);
  if (version_out) {
    if (ver && *ver && !modelmux_version_is_int (ver)) {
      CFG_WARN ("[%s] model ref '%s' version '%s' is not a positive integer "
          "(Triton convention) -- using '%s'", grp, spec, ver, MM_MODEL_VERSION_DEFAULT);
      *version_out = g_strdup (MM_MODEL_VERSION_DEFAULT);
    } else {
      *version_out = g_strdup ((ver && *ver) ? ver : MM_MODEL_VERSION_DEFAULT);
    }
  }
  g_strfreev (f);
  return out;
}

/* Parse a combined model spec "name;version;config-file" (the primary-model / shadow-model
 * PROPERTY form) into its parts:
 *   - fields are positional and ';'-separated; name/version may be empty ("MyModel;;config.txt");
 *   - a spec with NO ';' is taken as the config-file alone;
 *   - config-file is MANDATORY (returns FALSE if absent);
 *   - name defaults to the config-file basename (sans extension) when not given.
 * On success @config and @name are non-NULL (@version may be NULL). Caller frees all. */
static gboolean
cfg_parse_model_spec (const gchar * spec, const gchar * role_default,
    gchar ** name, gchar ** version, gchar ** config)
{
  *name = *version = *config = NULL;
  if (!(spec && *spec))
    return FALSE;

  if (strchr (spec, ';')) {
    gchar **f = g_strsplit (spec, ";", 3);
    *name = (f[0] && *f[0]) ? g_strdup (f[0]) : NULL;
    *version = (f[1] && *f[1]) ? g_strdup (f[1]) : NULL;
    *config = (f[2] && *f[2]) ? g_strdup (f[2]) : NULL;
    g_strfreev (f);
  } else {
    *config = g_strdup (spec);  /* bare config-file path */
  }

  if (!*config) {               /* config-file is required */
    g_free (*name);
    g_free (*version);
    *name = *version = NULL;
    return FALSE;
  }
  if (!*name) {                 /* derive logical name from config basename */
    gchar *base = g_path_get_basename (*config);
    gchar *dot = strrchr (base, '.');
    if (dot && dot != base)
      *dot = '\0';
    *name = (*base) ? g_strdup (base) : g_strdup (role_default);
    g_free (base);
  }
  return TRUE;
}

/* SECONDARY-stage config check. A secondary is a much smaller thing than a primary:
 * one model over all streams, no A/B, no per-stream routing. Everything the primary
 * path needs and it does not -- unified-batch, sharding, idle compaction, the [muxer]
 * geometry, shadow-model, [stream-model-*] bindings -- is simply IGNORED here rather
 * than rejected, so one config file can be moved between positions unchanged.
 *
 * Only what would leave the stage broken or ambiguous is an error:
 *   - no serving model at all (a secondary cannot run without one),
 *   - a primary-model naming something the catalog does not declare.
 * The ignored keys are named once at INFO so the omission is diagnosable. */
static gboolean
modelmux_config_validate_sgie (GstNvModelMux * self, ModelMuxConfig * config,
    GKeyFile * kf)
{
  GString *ignored = g_string_new (NULL);

  if (!config->default_primary || !*config->default_primary) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("no serving model configured for this secondary stage"),
        ("set 'primary-model=<name>[;<version>]' naming one of the [model-name-N] "
         "entries -- a secondary serves exactly one model and cannot start without it"));
    g_string_free (ignored, TRUE);
    return FALSE;
  }
  if (!modelmux_config_find_model (config, config->default_primary)) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("primary-model '%s' is not in the catalog", config->default_primary),
        ("declare it in a [model-name-N] entry of this config"));
    g_string_free (ignored, TRUE);
    return FALSE;
  }

  /* A backend this position cannot host is a CONFIG error, not a load-time surprise.
   * Caught here for the same reason template uniqueness is: discovering it at the
   * first model/load produces a confusing partial success -- the pipeline ran, the
   * stage came up, and one request failed for a reason that was decided long before
   * anyone sent it. The runtime path checks too (a load brings its own config_file,
   * so its backend is not knowable from the catalog alone). */
  {
    guint i;
    for (i = 0; i < config->num_models; i++) {
      ModelType t = config->models[i].type;
      if (modelmux_backend_at_position (t, MM_PROCESS_MODE_SGIE))
        continue;
      GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
          ("model '%s' uses %s, which cannot run at a secondary stage",
              config->models[i].name, modelmux_backend (t)->element),
          ("its config '%s' declares that backend. Move this model to the PRIMARY "
           "stage's catalog, or point the entry at a config whose backend a "
           "secondary supports", config->models[i].config_file
              ? config->models[i].config_file : "(none)"));
      g_string_free (ignored, TRUE);
      return FALSE;
    }
  }

  /* Keys that describe machinery a secondary does not have. Reported from the
   * KEYFILE, not from the parsed struct: every one of them has a default, so the
   * struct cannot tell "the operator set this" from "nobody did" -- and reporting a
   * default back would be noise on every correct config.
   *
   * Worth saying out loud rather than ignoring quietly: the obvious way to write a
   * secondary config is to copy the primary's and edit it, which is how the staged
   * sample acquired its own silent defects. An operator who sets per-model-batch-size
   * here has every reason to think it took effect. */
  {
    static const gchar *const pgie_only[] = {
      "model-sharding",                   /* sharding is stream-count triggered; a
                                           * secondary's load is object-driven      */
      "compact-idle-models-across-pools", /* pools exist for primary/shadow roles   */
      "per-model-batch-size",             /* sizes the per-model mux; there is none */
      "buffer-copy-mode",                 /* configures the A/B copy lane           */
      "copy-buffer-pool-size",            /* likewise                               */
      "unified-batch",                    /* meaningless without two lanes          */
      NULL
    };
    guint i;
    for (i = 0; pgie_only[i]; i++)
      if (kf && g_key_file_has_key (kf, "multimodel", pgie_only[i], NULL))
        g_string_append_printf (ignored, " %s", pgie_only[i]);
    if (kf && g_key_file_has_group (kf, "muxer"))
      g_string_append (ignored, " [muxer]");
  }

  if (config->default_shadow)
    g_string_append (ignored, " shadow-model");
  if (config->num_bindings)
    g_string_append (ignored, " [stream-model-*]");
  if (ignored->len)
    GST_INFO_OBJECT (self, "secondary stage: ignoring pgie-only config --%s "
        "(a secondary serves one model over all streams, with no A/B and no "
        "per-stream routing)", ignored->str);
  g_string_free (ignored, TRUE);
  return TRUE;
}

/* Catalog gie-unique-id collision check.
 *
 * A model that DECLARES gie-unique-id=d occupies TWO ids at this stage: d for its
 * primary lane and d+MM_SHADOW_GIE_OFFSET for its A/B twin. Two models therefore
 * collide when their declared ids are equal OR exactly one offset apart -- and a
 * collision means two live models stamp the same identity into the object meta, so
 * downstream attribution and any `operate-on-gie-id` become ambiguous.
 *
 * A collision does NOT fail the config -- reusing one stock nvinfer config across
 * several catalog entries is normal, and those all declare the same id. Instead the
 * whole stage falls back to the monotonic allocator, exactly as it behaved before
 * configured ids existed, and says so once. Models that declare nothing are skipped:
 * they cannot collide by construction. */
static void
modelmux_config_resolve_gie_ids (GstNvModelMux * self, ModelMuxConfig * config)
{
  gint ids[MM_MAX_MODELS];
  guint i, j;

  for (i = 0; i < config->num_models; i++)
    ids[i] = modelmux_config_file_gie_id (config->models[i].config_file);

  for (i = 0; i < config->num_models; i++) {
    if (ids[i] < 0)
      continue;
    for (j = i + 1; j < config->num_models; j++) {
      gint d;
      if (ids[j] < 0)
        continue;
      d = ABS (ids[i] - ids[j]);
      if (d != 0 && d != MM_SHADOW_GIE_OFFSET)
        continue;
      GST_WARNING_OBJECT (self, "gie-unique-id: '%s' declares %d and '%s' declares %d "
          "-- a model occupies both <id> and <id>+%d (its A/B shadow lane), so these "
          "OVERLAP. Falling back to internally-allocated gie ids for this stage "
          "(read them from GET /api/v1/model/status). To make them config-knowable "
          "again, give the models ids at least %d apart",
          config->models[i].name, ids[i], config->models[j].name, ids[j],
          MM_SHADOW_GIE_OFFSET, MM_SHADOW_GIE_OFFSET + 1);
      config->gie_ids_from_config = FALSE;
      return;
    }
  }
  config->gie_ids_from_config = TRUE;
}

gboolean
gst_modelmux_parse_config_file (GstNvModelMux * self, const gchar * cfg_file)
{
  ModelMuxConfig *config = &self->config;
  GKeyFile *kf;
  GError *err = NULL;

  /* config-file-path is mandatory: a missing/empty path is a hard failure. Post the
   * exact cause here (the caller just returns FALSE). */
  if (!(cfg_file && *cfg_file)) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("config-file-path is required -- set the 'config-file-path' property to a "
         "valid nvmodelmux config file"), (NULL));
    return FALSE;
  }

  kf = g_key_file_new ();
  memset (config, 0, sizeof (*config));
  g_mutex_init (&config->versions_lock);   /* see ModelMuxConfig.versions_lock (priv.h) */

  if (!g_key_file_load_from_file (kf, cfg_file, G_KEY_FILE_NONE, &err)) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("failed to parse config-file-path '%s'", cfg_file),
        ("%s", err ? err->message : "unknown error"));
    if (err)
      g_error_free (err);
    g_key_file_free (kf);
    return FALSE;
  }

  /* ---- [multimodel] : behaviour + default model REFs ----
   * Models are DEFINED in [model-name-<idx>] catalog blocks (name -> base config);
   * primary-model / shadow-model REFERENCE them as "name;version". The same ref form
   * is used by the per-sensor [stream-model-<id>] groups. */
  {
    const gchar *SEC = "multimodel";
    gchar *pspec, *sspec;

    /* Property-vs-config reconciliation (nvinfer convention): an EXPLICITLY-set GObject
     * property overrides the config-file value; otherwise the file value is used AND
     * mirrored back into the property field so a read-back reflects the effective config. */
    /* enable=0 => bypass the whole inference bin (decode->display passthrough). */
    if (self->is_prop_set[PROP_ENABLE])
      config->enable = self->enable;
    else
      self->enable = config->enable = cfg_bool (kf, SEC, "enable", TRUE);
    if (self->is_prop_set[PROP_UNIFIED_BATCH])
      config->unified_batch = self->unified_batch;
    else
      self->unified_batch = config->unified_batch = cfg_bool (kf, SEC, "unified-batch", TRUE);
    /* position: 1 = pgie, 2 = sgie. The element property wins -- deepstream-app sets
     * it from the section the element was built in, which is authoritative. */
    if (self->is_prop_set[PROP_PROCESS_MODE])
      config->process_mode = self->process_mode;
    else
      self->process_mode = config->process_mode =
          cfg_int (kf, SEC, "process-mode", 1);
    /* auto-compact under-full shards back into fewer shards as streams drain
     * (zero-drop, make-before-break). Default ON. No matching property. */
    config->shard_compact = cfg_bool (kf, SEC, "shard-compact", TRUE);
    /* compact-idle-models-across-pools (dedup-on-idle): when a model goes idle as a lone
     * redundant warm copy (no twin), move it out of its role pool (primary/shadow) into the
     * shared limbo pool so a later request from EITHER role reuses that one warm copy with no
     * reload -- fewer warm copies overall. FALSE keeps it role-tagged in its pool. The
     * configured default model is never moved. Default ON. */
    if (self->is_prop_set[PROP_COMPACT_IDLE_MODELS_ACROSS_POOLS])
      config->compact_idle_models_across_pools = self->compact_idle_models_across_pools;
    else
      self->compact_idle_models_across_pools = config->compact_idle_models_across_pools =
          cfg_bool (kf, SEC, "compact-idle-models-across-pools", TRUE);
    /* model-sharding: TRUE (default) = auto-shard on overflow; FALSE = nvinfer-compat
     * (ignore streams beyond the per-model batch cap, not routed). */
    if (self->is_prop_set[PROP_MODEL_SHARDING])
      config->model_sharding = self->model_sharding;
    else
      self->model_sharding = config->model_sharding = cfg_bool (kf, SEC, "model-sharding", TRUE);
    /* perf metrics: opt-in (default off => zero overhead). When on, the plugin measures
     * per-model + per-(stream,role) fps + inference latency every perf-metric-interval-sec
     * and reports them additively in model/status. */
    config->attach_perf_metric = cfg_bool (kf, SEC, "attach-perf-metric", FALSE);
    config->perf_metric_interval_sec =
        cfg_uint (kf, SEC, "perf-metric-interval-sec", MM_PERF_DEFAULT_INTERVAL_SEC);
    if (config->perf_metric_interval_sec < MM_PERF_MIN_INTERVAL_SEC)
      config->perf_metric_interval_sec = MM_PERF_MIN_INTERVAL_SEC;
    /* buffer-copy-mode: which lane(s) deep-copy the decoded frame via an nvvideoconvert
     * (disable-passthrough=1) spliced BEFORE that lane's queue -- a copy lane releases the
     * upstream decoder buffer at fan-out, so a slow lane cannot backpressure the decoder when
     * per-model fps differ; the other (passthrough) role stays zero-copy (no element added).
     *   0 = none, 1 = shadow only (DEFAULT), 2 = primary only, 3 = both.
     * 0 => no converter on any lane (byte-identical to before this feature). */
    {
      guint cmode = cfg_uint (kf, SEC, "buffer-copy-mode", 1);   /* 1 = shadow (default) */
      config->copy_primary_buffer = (cmode == 2 || cmode == 3);
      config->copy_shadow_buffer  = (cmode == 1 || cmode == 3);
    }
    /* copy-lane nvvideoconvert output pool depth (0 = element default); raise if a slow copy
     * lane backpressures the converter. Only used when a lane actually copies. */
    config->copy_buffer_pool_size = cfg_uint (kf, SEC, "copy-buffer-pool-size", 0);
    /* lane-leaky: per-(stream,role) lane queue policy. 1 (default, unchanged behaviour) =
     * leaky=downstream (drop-oldest; freshness -- a slow model can't stall its siblings);
     * 0 = non-leaky (completeness -- every frame reaches every routed model, at the cost of
     * backpressure stalls when per-model fps mismatch). See ModelMuxConfig.lane_leaky. */
    config->lane_leaky = cfg_bool (kf, SEC, "lane-leaky", TRUE);
    /* auto-gpu-scale (default 0): elastic scale-out of an over-capacity model
     * onto an IDENTICAL other GPU (same compute capability + name, so the
     * engine is reused) without any per-model gpu configuration; the overflow
     * shard is unpinned and reclaimed by compaction when load drops. */
    config->auto_gpu_scale = cfg_bool (kf, SEC, "auto-gpu-scale", FALSE);
    /* placement-policy: how UNCONSTRAINED streams fill a multi-GPU instance
     * group. pack (default) = saturate instances in gpus[] order (spill to
     * the next GPU only when a shard hits max-streams; ECS "binpack"); spread =
     * least-loaded instance (latency-first; ECS "spread"). Explicit route gpus
     * bypass the policy either way. */
    {
      gchar *pol = g_key_file_get_string (kf, SEC, "placement-policy", NULL);
      if (!pol || g_ascii_strcasecmp (pol, "pack") == 0)
        config->placement_policy = PACK_PLACEMENT;
      else if (g_ascii_strcasecmp (pol, "spread") == 0)
        config->placement_policy = SPREAD_PLACEMENT;
      else {
        CFG_WARN ("[%s] placement-policy='%s' unknown (pack|spread) -- using pack",
            SEC, pol);
        config->placement_policy = PACK_PLACEMENT;
      }
      g_free (pol);
    }
    /* cross-device migration is NOT user-configurable: same-device bins carry
     * no migration elements; cross-device bins get the nvdsxfer pair, and a
     * placement that cannot be served (no P2P / factory missing) is REJECTED
     * at bin creation (see model_bin_new). */
    /* gpu-id: the PIPELINE device (decoders / shared demux / combined display mux).
     * Every model bin's egress device-migration converter returns inferred batches to
     * this device; a shard placed on a DIFFERENT gpu gets its input migrated by the
     * bin's ingress converter. Default 0. See ModelMuxConfig.gpu. */
    config->gpu = cfg_uint (kf, SEC, "gpu-id", 0);
    /* runtime defaults carry no placement until a model/routing default{} sets one */
    config->default_primary_gpu = -1;
    config->default_shadow_gpu = -1;
    /* max concurrent streams (sizes every internal mux); MUST be >= the app's streammux
     * batch. Property overrides the file (a host sets it per-instance to match the
     * upstream batch); else the file value (default MM_MAX_STREAMS_DEFAULT). */
    if (self->is_prop_set[PROP_BATCH_SIZE])
      config->batch_size = self->batch_size;
    else
      self->batch_size = config->batch_size =
          cfg_uint (kf, SEC, "batch-size", MM_MAX_STREAMS_DEFAULT);
    /* per-model nvinfer/mux batch cap (default = total batch, one nvinfer per model). */
    if (self->is_prop_set[PROP_PER_MODEL_BATCH_SIZE])
      config->per_model_max_streams = self->per_model_batch_size;
    else
      self->per_model_batch_size = config->per_model_max_streams =
          cfg_uint (kf, SEC, "per-model-batch-size", config->batch_size);
    /* validate the (untrusted) config values: an out-of-range batch would size every
     * internal mux unchecked -- clamp to the supported max. 0 falls back to the default. */
    if (config->batch_size == 0)
      config->batch_size = MM_MAX_STREAMS_DEFAULT;
    if (config->batch_size > MM_MAX_BATCH) {
      CFG_WARN ("[%s] batch-size %u exceeds max %u -- clamping", SEC,
          config->batch_size, (guint) MM_MAX_BATCH);
      config->batch_size = MM_MAX_BATCH;
    }
    if (config->per_model_max_streams == 0)
      config->per_model_max_streams = config->batch_size;
    if (config->per_model_max_streams > MM_MAX_BATCH)
      config->per_model_max_streams = MM_MAX_BATCH;

    /* [muxer] -- nvstreammux settings shared by the per-model inference muxers AND the combined
     * outer mux (ib->out_mux). width/height/live-source are common to all; only the flush cadence
     * differs (per-model regroups an already-batched upstream -> 10ms; combined aggregates across
     * parallel inference -> ~33ms). The combined-mux batch is auto-derived (out_slots), not here. */
    {
      const gchar *MUX = "muxer";
      config->mux_width                 = cfg_uint (kf, MUX, "width",  MM_MUX_WIDTH);
      config->mux_height                = cfg_uint (kf, MUX, "height", MM_MUX_HEIGHT);
      config->mux_live_source           = cfg_bool (kf, MUX, "live-source", TRUE);
      config->model_mux_push_timeout    = cfg_uint (kf, MUX, "per-model-batch-push-timeout", 10000);
      config->combined_mux_push_timeout = cfg_uint (kf, MUX, "combined-batch-push-timeout", MM_MUX_PUSH_TIMEOUT);
      /* width/height of 0 yield invalid caps in nvstreammux (gst_video_info_to_caps: finfo==NULL
       * crash) -- a 0 is a valid GKeyFile integer cfg_uint passes through, so clamp to the default.
       * (cfg_uint already rejects negatives.) */
      if (config->mux_width == 0) {
        CFG_WARN ("[muxer] width=0 is invalid -> using default %d", MM_MUX_WIDTH);
        config->mux_width = MM_MUX_WIDTH;
      }
      if (config->mux_height == 0) {
        CFG_WARN ("[muxer] height=0 is invalid -> using default %d", MM_MUX_HEIGHT);
        config->mux_height = MM_MUX_HEIGHT;
      }
    }

    /* CATALOG: parse [model-name-<idx>] blocks (name -> base config) BEFORE the role
     * refs below, which validate their model name against the catalog. */
    cfg_parse_catalog (config, kf);
    /* per-version artifact overrides ([model-<name>-<version>]) -- AFTER the catalog so names
     * validate; consulted at instantiation by defaults / per-sensor / load. */
    cfg_parse_per_version (config, kf);

    /* the catalog is complete: decide whether the DECLARED gie ids are usable. */
    modelmux_config_resolve_gie_ids (self, config);

    /* default PRIMARY model REF (OPTIONAL): "name;version" into the catalog. Every stream
     * runs it unless overridden. When none resolves, no model is preloaded (zero VRAM) and
     * any stream that resolves to no model is PASSED THROUGH the display mux without
     * inference (frames flow, no detections). */
    pspec = cfg_str (kf, SEC, "primary-model", NULL);
    config->default_primary =
        cfg_parse_ref (config, SEC, pspec, &config->default_primary_version);
    if (!config->default_primary)
      CFG_INFO ("[%s] no resolvable default 'primary-model' -- streams with no resolvable "
          "model will PASS THROUGH without inference", SEC);
    g_free (pspec);

    /* default SHADOW model REF (OPTIONAL A/B): only meaningful with unified-batch. */
    sspec = cfg_str (kf, SEC, "shadow-model", NULL);
    if (sspec && *sspec && !config->unified_batch) {
      CFG_WARN ("[%s] 'shadow-model' is set but unified-batch=0 -- a shadow has no "
          "display row in primary-only mode; IGNORING it. Set unified-batch=1.", SEC);
      config->default_shadow = NULL;
    } else {
      config->default_shadow =
          cfg_parse_ref (config, SEC, sspec, &config->default_shadow_version);  /* NULL => none */
    }
    g_free (sspec);

    /* GUARD: the default shadow must not be the SAME instance as the default primary --
     * same (name, version) in BOTH roles is a pointless A/B (one engine scored twice, double
     * VRAM) and would create two identical bins (one per pool). A different VERSION of the
     * same name is fine (real cross-checkpoint A/B). Disable the redundant shadow. */
    if (config->default_primary && config->default_shadow &&
        g_strcmp0 (config->default_primary, config->default_shadow) == 0 &&
        g_strcmp0 (config->default_primary_version, config->default_shadow_version) == 0) {
      CFG_WARN ("[%s] shadow-model '%s;%s' is IDENTICAL to primary-model (same name+version) "
          "-- an A/B against an identical model is pointless; DISABLING the shadow. Use a "
          "different VERSION (e.g. shadow-model=%s;<other-version>) for real A/B.",
          SEC, config->default_shadow, config->default_shadow_version, config->default_primary);
      g_free (config->default_shadow);          config->default_shadow = NULL;
      g_free (config->default_shadow_version);  config->default_shadow_version = NULL;
    }

    /* PROPERTY primary-model / shadow-model OVERRIDE (nvinfer convention: an explicitly-set
     * property wins over the config file). Form: "name;version;config-file" -- config-file is
     * mandatory and must exist. It registers the model in the catalog and re-points the role
     * default. Primary is REQUIRED to exist (hard error); shadow is optional (warn only). */
    if (self->is_prop_set[PROP_PRIMARY_MODEL]) {
      gchar *pn = NULL, *pv = NULL, *pc = NULL;
      if (cfg_parse_model_spec (self->primary_model, "primary", &pn, &pv, &pc)) {
        if (!g_file_test (pc, G_FILE_TEST_EXISTS)) {
          GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND,
              ("primary-model config-file not found: '%s'", pc), (NULL));
          g_free (pn); g_free (pv); g_free (pc);
          g_key_file_free (kf);
          return FALSE;
        }
        if (pv && !modelmux_version_is_int (pv)) {
          /* the REST surface only addresses positive-integer versions -- a default
           * keyed on 'v2'-style text could never be updated/unloaded via the API */
          CFG_WARN ("primary-model version '%s' is not a positive integer -- ignored "
              "(default falls back to version 1)", pv);
          g_free (pv); pv = NULL;
        }
        modelmux_config_register_model (config, pn, pc, NULL, pv, 0);
        /* the property is an OVERRIDE: if the name already exists in the catalog
         * ([model-*] group), register is first-wins and would silently keep the
         * catalog's config-file while the log suggests the property won -- update. */
        modelmux_config_update_model (config, pn, pc, NULL, NULL);
        g_free (config->default_primary);         config->default_primary = g_strdup (pn);
        g_free (config->default_primary_version); config->default_primary_version =
            (pv && *pv) ? g_strdup (pv) : NULL;
      } else if (self->primary_model && *self->primary_model) {
        CFG_WARN ("primary-model='%s' has no config-file -- ignored", self->primary_model);
      }
      g_free (pn); g_free (pv); g_free (pc);
    }
    if (self->is_prop_set[PROP_SHADOW_MODEL]) {
      gchar *sn = NULL, *sv = NULL, *sc = NULL;
      if (cfg_parse_model_spec (self->shadow_model, "shadow", &sn, &sv, &sc)) {
        if (!g_file_test (sc, G_FILE_TEST_EXISTS)) {
          CFG_WARN ("shadow-model config-file not found: '%s' -- no shadow", sc);
        } else {
          if (sv && !modelmux_version_is_int (sv)) {
            CFG_WARN ("shadow-model version '%s' is not a positive integer -- ignored "
                "(default falls back to version 1)", sv);
            g_free (sv); sv = NULL;
          }
          modelmux_config_register_model (config, sn, sc, NULL, sv, 0);
          modelmux_config_update_model (config, sn, sc, NULL, NULL);   /* see primary */
          g_free (config->default_shadow);         config->default_shadow = g_strdup (sn);
          g_free (config->default_shadow_version); config->default_shadow_version =
              (sv && *sv) ? g_strdup (sv) : NULL;
        }
      } else if (self->shadow_model && *self->shadow_model) {
        CFG_WARN ("shadow-model='%s' has no config-file -- ignored", self->shadow_model);
      }
      g_free (sn); g_free (sv); g_free (sc);
    }
  }

  /* ---- [stream-model-<sensor_id>] : config-time PER-SENSOR model binding (optional) ----
   * One group per camera id, using the SAME spec keys as the [multimodel] group:
   *   primary-model = name;version
   *   shadow-model  = name;version   (optional)
   * Referenced models are validated against the catalog (so they are buildable) but NOT
   * preloaded: a bound non-default model is LOADED LAZILY when its stream is added (passthrough
   * while warming, then auto-promote). A missing role falls back to the configured default.
   * No groups => num_bindings stays 0 => every stream uses the defaults (back-compatible). */
  {
    static const gchar *PFX = "stream-model-";
    gchar **groups = g_key_file_get_groups (kf, NULL);
    guint gi;
    for (gi = 0; groups && groups[gi]; gi++) {
      const gchar *grp = groups[gi];
      const gchar *sid;
      gchar *pspec, *sspec;
      StreamModelBinding *b;
      if (!g_str_has_prefix (grp, PFX))
        continue;
      sid = grp + strlen (PFX);
      if (!*sid)
        continue;
      if (config->num_bindings >= MM_MAX_STREAMS_DEFAULT) {
        CFG_WARN ("too many [stream-model-*] groups (max %d) -- ignoring '%s'",
            MM_MAX_STREAMS_DEFAULT, grp);
        continue;
      }
      b = &config->bindings[config->num_bindings];
      b->sensor_id = g_strdup (sid);
      pspec = cfg_str (kf, grp, "primary-model", NULL);
      sspec = cfg_str (kf, grp, "shadow-model", NULL);
      b->primary = cfg_parse_ref (config, grp, pspec, &b->primary_version);  /* route only */
      b->shadow  = cfg_parse_ref (config, grp, sspec, &b->shadow_version);   /* route only */
      config->num_bindings++;
      GST_INFO ("per-sensor binding: '%s' -> primary=%s shadow=%s",
          sid, b->primary ? b->primary : "(default)", b->shadow ? b->shadow : "(default)");
      g_free (pspec); g_free (sspec);
    }
    g_strfreev (groups);
  }

  /* BEFORE the keyfile is freed: the check reports which pgie-only keys were
   * actually written, which only the keyfile knows. */
  if (config->process_mode == MM_PROCESS_MODE_SGIE
      && !modelmux_config_validate_sgie (self, config, kf)) {
    g_key_file_free (kf);
    return FALSE;
  }

  g_key_file_free (kf);

  GST_INFO ("primary='%s' shadow='%s', max-batch=%u, per-model-max=%u",
      config->default_primary ? config->default_primary : "(none)",
      config->default_shadow ? config->default_shadow : "(disabled)",
      config->batch_size, config->per_model_max_streams);
  return TRUE;
}

const ModelCatalogEntry *
modelmux_config_find_model (const ModelMuxConfig * config, const gchar * name)
{
  guint i;
  if (!name)
    return NULL;
  for (i = 0; i < config->num_models; i++)
    if (g_strcmp0 (config->models[i].name, name) == 0)
      return &config->models[i];
  return NULL;
}

/* Register a model at RUNTIME (idempotent). The commented default_shadow only
 * disables the DEFAULT shadow -- the shadow lane stays available, so a control
 * path (e.g. an extended REST add/update carrying a model name + config-file)
 * can register a model here and then attach/update a stream to use it. */
/* ================================================================== */
/* Backend registry -- the one place a per-backend fact is written.     */
/* See the ModelBackendInfo comment in gstnvmodelmux_priv.h.            */
/* ================================================================== */

/* Config-file markers, most-specific first. A marker is matched as a line PREFIX
 * against the first decisive non-comment line (see modelmux_detect_model_type), so
 * each backend lists every grammar its config can be written in. */
static const gchar *const mm_markers_infer[] = {
  "[property]",                 /* INI  section  */
  "property:",                  /* YAML mapping  */
  NULL
};
static const gchar *const mm_markers_inferserver[] = {
  "infer_config",               /* protobuf 'infer_config {' OR YAML 'infer_config:' */
  NULL
};
static const gchar *const mm_markers_visionencoder[] = {
  "[visionencoder]",            /* INI  section -- deepstream-app's own group name */
  "visionencoder:",             /* YAML mapping                                    */
  NULL
};

/* Indexed by ModelType -- keep the rows in enum order.
 *
 * nvinferserver is PGIE-ONLY here, and that is a scope decision rather than a
 * technical limit: a secondary stage runs one model over an already-formed batch,
 * and every secondary capability this plugin needs (in-place OTA, an engine
 * artifact per version, plugin-side placement) is exactly what Triton delegates to
 * its own model repository. Declaring the restriction in the table makes it a
 * config-time rejection with a reason instead of a runtime failure inside a stage
 * that cannot honour the request. The vision encoder will be the mirror image --
 * SGIE-only -- and lands here as one more row. */
static const ModelBackendInfo modelmux_backends[MODEL_BACKEND_N] = {
  [MODEL_INFER] = {
        .element = "nvinfer",
        .markers = mm_markers_infer,
        .positions = MM_BACKEND_AT_PGIE | MM_BACKEND_AT_SGIE,
        .engine_file = TRUE,      /* model-engine-file swaps a checkpoint in     */
        .version_artifact = TRUE, /* a new version must bring config or engine   */
        .inplace_reload = TRUE,   /* nvinfer's native hot-swap backs model/update */
        .owns_placement = FALSE,  /* the plugin pins gpu-id, so status can say so */
        .multi_gpu_scale = TRUE,  /* gpus[] instance groups are ours to build    */
        .gpu_ids_list = FALSE,    /* scalar `gpu-id` in the property section     */
        .config_prop = "config-file-path",
        .config_group = NULL,     /* nvinfer reads its own config file           */
        .engine_prop = "model-engine-file",
        .gie_id = TRUE,
        .cfg_section = "property",
        .gpu_key = "gpu-id",
      },
  [MODEL_INFERSERVER] = {
        .element = "nvinferserver",
        .markers = mm_markers_inferserver,
        .positions = MM_BACKEND_AT_PGIE,
        .engine_file = FALSE,     /* Triton's model repository owns checkpoints  */
        .version_artifact = FALSE,/* a bare (name, version) is the correct form  */
        .inplace_reload = FALSE,  /* reload happens in Triton, not here          */
        .owns_placement = TRUE,   /* multi-gpu gpu_ids = Triton's own scheduling */
        .multi_gpu_scale = FALSE, /* scale via the repo's instance_group         */
        .gpu_ids_list = TRUE,     /* `gpu_ids: [N]` inside `infer_config`        */
        .config_prop = "config-file-path",
        .config_group = NULL,
        .engine_prop = NULL,      /* checkpoints live in the Triton repository   */
        .gie_id = TRUE,
        .cfg_section = "infer_config",
        .gpu_key = "gpu_ids",
      },
  /* nvdsvisionencoder: object embeddings at a SECONDARY position.
   *
   * It differs from the two inference backends in three ways that the rest of the
   * plugin now asks about by name rather than by type:
   *   - it has NO config-file-path. It is configured entirely through GObject
   *     properties, so its "model config" is a [visionencoder] group whose keys ARE
   *     the property names (config_group instead of config_prop).
   *   - it has NO unique-id and NO operate-on-gie-id, so it cannot be scoped to one
   *     detector's output nor attributed downstream (gie_id = FALSE). It embeds every
   *     object it is handed, filtered only by min-crop-size / embedding-classes.
   *   - it consumes colour-converted RGB surfaces, so the chain wraps it in
   *     colour-converted surfaces -- which it cannot get at a secondary, see
   *     the note on the row below. */
  [MODEL_VISION_ENCODER] = {
        .element = "nvdsvisionencoder",
        .markers = mm_markers_visionencoder,
        .positions = MM_BACKEND_AT_SGIE,
        .engine_file = TRUE,      /* tensorrt-engine is a real artifact          */
        .version_artifact = TRUE, /* a new version brings its own engine         */
        .inplace_reload = TRUE,   /* tensorrt-engine is MUTABLE_PLAYING          */
        .owns_placement = FALSE,  /* honours gpu-id (cudaSetDevice per thread)   */
        .multi_gpu_scale = FALSE, /* no instance groups                          */
        .gpu_ids_list = FALSE,
        .config_prop = NULL,      /* no config-file-path property exists         */
        .config_group = "visionencoder",
        .engine_prop = "tensorrt-engine",
        .gie_id = FALSE,
        .cfg_section = "visionencoder",
        .gpu_key = "gpu-id",
        /* wrapped in nvvideoconvert on both sides, RGB on the way in -- the same
         * shape deepstream-app builds around its own encoder. (That bin picks RGBA
         * on Thor for VIC; this is the dGPU/Jetson format.) */
        /* NOTE, a hard constraint on ANY element placed in a secondary chain:
         *
         * An element that produces a NEW buffer copies the metas onto it, and
         * gst_nvds_meta_transform() (gst-nvdsmeta/gstnvdsmeta.c:84) answers
         * GST_META_TRANSFORM_IS_COPY for NVDS_BATCH_GST_META by calling
         * nvds_batch_meta_copy_func -- a DEEP copy. Everything downstream of such an
         * element then writes to a CLONE of NvDsBatchMeta, and a secondary chain ends
         * in a fakesink, so the clone is dropped: the stage's embeddings, and its
         * overlay line, never reach the display or anything else that reads the
         * original batch.
         *
         * deepstream-app can wrap its own encoder in nvvideoconvert because that bin
         * sits INLINE -- its converted buffer is what continues downstream, so the
         * clone is the buffer everyone reads. A secondary on a tee branch has no such
         * luxury: it must consume the batch IN PLACE, exactly as nvinfer does. */
      },
};

const ModelBackendInfo *
modelmux_backend (ModelType type)
{
  /* clamp rather than assert: @type can arrive from a config sniff of an operator's
   * file, and MODEL_INFER is that sniff's own back-compat default. Returning a valid
   * row unconditionally is what lets every call site dereference without a check. */
  if (type < 0 || type >= MODEL_BACKEND_N)
    return &modelmux_backends[MODEL_INFER];
  return &modelmux_backends[type];
}

/* Apply one key as the property of the same name, converting by the property's OWN
 * GParamSpec type rather than by guessing from the string. GKeyFile has no type
 * information, so the element's introspection is the only authority on whether
 * "8" is a uint, a float or a string. */
static void
cfg_apply_one_property (GstElement * element, GKeyFile * kf, const gchar * group,
    const gchar * key)
{
  GParamSpec *ps = g_object_class_find_property (G_OBJECT_GET_CLASS (element), key);
  gchar *raw;

  if (!ps) {
    /* SKIP, never fail: a config written for a newer element must not stop a
     * pipeline running an older one. Named loudly so a typo is still findable --
     * a misspelt key is indistinguishable from a newer one, and both land here. */
    CFG_WARN ("config: '%s' has no property '%s' -- ignored (misspelt, or a key this "
        "build's element does not have)", GST_ELEMENT_NAME (element), key);
    return;
  }
  raw = g_key_file_get_string (kf, group, key, NULL);
  if (!raw)
    return;
  g_strstrip (raw);

  switch (G_TYPE_FUNDAMENTAL (ps->value_type)) {
    case G_TYPE_BOOLEAN:
      /* accept 1/0 as well as true/false: DS configs use the integer form */
      g_object_set (G_OBJECT (element), key,
          (g_ascii_strcasecmp (raw, "true") == 0 || g_strcmp0 (raw, "1") == 0), NULL);
      break;
    case G_TYPE_INT:
      g_object_set (G_OBJECT (element), key, (gint) g_ascii_strtoll (raw, NULL, 10), NULL);
      break;
    case G_TYPE_UINT:
      g_object_set (G_OBJECT (element), key, (guint) g_ascii_strtoull (raw, NULL, 10), NULL);
      break;
    case G_TYPE_INT64:
      g_object_set (G_OBJECT (element), key, g_ascii_strtoll (raw, NULL, 10), NULL);
      break;
    case G_TYPE_UINT64:
      g_object_set (G_OBJECT (element), key, g_ascii_strtoull (raw, NULL, 10), NULL);
      break;
    case G_TYPE_FLOAT:
      g_object_set (G_OBJECT (element), key, (gfloat) g_ascii_strtod (raw, NULL), NULL);
      break;
    case G_TYPE_DOUBLE:
      g_object_set (G_OBJECT (element), key, g_ascii_strtod (raw, NULL), NULL);
      break;
    case G_TYPE_STRING:
      /* An EMPTY string is meaningful, not absent: embedding-classes="" is
       * "every class". Pass it through rather than treating it as unset. */
      g_object_set (G_OBJECT (element), key, raw, NULL);
      break;
    case G_TYPE_ENUM:
      g_object_set (G_OBJECT (element), key, (gint) g_ascii_strtoll (raw, NULL, 10), NULL);
      break;
    default:
      CFG_WARN ("config: property '%s' has type %s, which this reader cannot set from "
          "a config key -- ignored", key, g_type_name (ps->value_type));
      break;
  }
  g_free (raw);
}

gboolean
modelmux_config_apply_properties (GstElement * element, const gchar * config_file,
    const gchar * group)
{
  GKeyFile *kf;
  gchar **keys;
  gsize n = 0, i;
  GError *err = NULL;

  if (!element || !config_file || !*config_file || !group)
    return FALSE;

  kf = g_key_file_new ();
  if (!g_key_file_load_from_file (kf, config_file, G_KEY_FILE_NONE, &err)) {
    CFG_WARN ("config: cannot read '%s': %s", config_file,
        err ? err->message : "unknown error");
    g_clear_error (&err);
    g_key_file_free (kf);
    return FALSE;
  }
  keys = g_key_file_get_keys (kf, group, &n, &err);
  if (!keys) {
    CFG_WARN ("config: '%s' has no [%s] group: %s", config_file, group,
        err ? err->message : "unknown error");
    g_clear_error (&err);
    g_key_file_free (kf);
    return FALSE;
  }
  for (i = 0; i < n; i++)
    cfg_apply_one_property (element, kf, group, keys[i]);

  CFG_INFO ("config: applied %" G_GSIZE_FORMAT " [%s] properties from '%s'", n, group,
      config_file);
  g_strfreev (keys);
  g_key_file_free (kf);
  return TRUE;
}

gboolean
modelmux_backend_at_position (ModelType type, gint process_mode)
{
  guint want = (process_mode == MM_PROCESS_MODE_SGIE)
      ? MM_BACKEND_AT_SGIE : MM_BACKEND_AT_PGIE;

  return (modelmux_backend (type)->positions & want) != 0;
}

/* Map the backend type to its gst element / log name. */
const gchar *
modelmux_infer_plugin_str (ModelType type)
{
  return modelmux_backend (type)->element;
}

/* Auto-detect the inference backend from a model's config-file. The formats are
 * unambiguous: an nvinferserver config is protobuf text whose mandatory root message is
 * `infer_config {`; an nvinfer config is INI with a `[property]` section. We scan the
 * first decisive non-comment, non-blank line and match it against each backend's
 * markers[] in the REGISTRY, so a new backend is recognised by adding a row rather
 * than by editing this function:
 *   - the first line matching any backend's marker -> that backend
 *   - nothing matched / unreadable                 -> MODEL_INFER (back-compat), WARNED
 * Comments (`#`) and the SPDX header are skipped, so they never mislead the sniff. */
ModelType
modelmux_detect_model_type (const gchar * config_file)
{
  gchar *content = NULL;
  gchar **lines;
  guint i, t;
  ModelType type = MODEL_INFER;
  gboolean decided = FALSE;

  if (!config_file || !*config_file)
    return MODEL_INFER;
  /* BOUNDED prefix read: the decisive marker is in the first non-comment lines by
   * design. The path can arrive from the REST API (model/load), so slurping the
   * whole file (g_file_get_contents + g_strsplit = 2x file size in RAM) let an
   * operator typo pointing at a multi-GB .engine allocate gigabytes on the main
   * loop. 64 KiB is orders of magnitude beyond any real config header. */
  {
    FILE *f = fopen (config_file, "re");
    gsize got = 0;
    if (!f) {
      CFG_WARN ("model-type: cannot read config '%s' -- defaulting to nvinfer",
          config_file);
      return MODEL_INFER;
    }
    content = g_malloc (64 * 1024 + 1);
    got = fread (content, 1, 64 * 1024, f);
    content[got] = '\0';
    fclose (f);
  }
  lines = g_strsplit (content, "\n", -1);
  for (i = 0; lines[i] && !decided; i++) {
    gchar *s = g_strstrip (lines[i]);            /* trims in place */
    if (!*s || *s == '#')                        /* skip blanks + comments (SPDX header) */
      continue;
    /* walk the REGISTRY, not a hand-written if/else chain: a new backend is
     * recognised by adding its markers to its row. */
    for (t = 0; t < MODEL_BACKEND_N && !decided; t++) {
      const gchar *const *m = modelmux_backend ((ModelType) t)->markers;
      for (; m && *m; m++)
        if (g_str_has_prefix (s, *m)) {
          type = (ModelType) t;
          decided = TRUE;
          break;
        }
    }
  }
  g_strfreev (lines);
  g_free (content);
  /* WARN on the fall-through. The default has to stay nvinfer for back-compat, but
   * that means a config whose marker is missing or misspelled is silently treated as
   * an nvinfer config and fails much later with an unrelated-looking error. Saying so
   * here puts the complaint at config time, next to the typo. */
  if (!decided) {
    GString *known = g_string_new (NULL);
    guint t;
    for (t = 0; t < MODEL_BACKEND_N; t++) {
      const gchar *const *m = modelmux_backend ((ModelType) t)->markers;
      for (; m && *m; m++)
        g_string_append_printf (known, "%s'%s'", known->len ? " / " : "", *m);
    }
    CFG_WARN ("model-type: no %s marker in '%s' -- defaulting to nvinfer. If this is "
        "not an nvinfer config, its backend marker is missing or misspelled",
        known->str, config_file);
    g_string_free (known, TRUE);
  }
  return type;
}

const ModelCatalogEntry *
modelmux_config_register_model (ModelMuxConfig * config, const gchar * name,
    const gchar * config_file, const gchar * engine_file, const gchar * version,
    guint max_streams)
{
  ModelCatalogEntry *m;
  guint i;

  if (!name || !*name || !config_file || !*config_file)
    return NULL;
  /* the instance-key separator and the model-spec field separator must not appear
   * in a model NAME: "yolo@fp16" would register fine but model_key_split later
   * cuts at the FIRST '@', silently corrupting every pool/limbo/status lookup. */
  if (strchr (name, MM_MODEL_KEY_SEP) || strchr (name, ';')) {
    CFG_WARN ("model name '%s' rejected: '%c' and ';' are reserved separators",
        name, MM_MODEL_KEY_SEP);
    return NULL;
  }
  for (i = 0; i < config->num_models; i++)            /* already registered? */
    if (g_strcmp0 (config->models[i].name, name) == 0)
      return &config->models[i];                      /* first-wins-by-name */
  if (config->num_models >= MM_MAX_MODELS) {
    CFG_WARN ("model registry full, cannot register '%s'", name);
    return NULL;
  }
  /* fill the slot COMPLETELY before publishing it via num_models++: a concurrent
   * modelmux_config_find_model scanning the array must never observe a half-initialized
   * entry (NULL config_file on a matched name). */
  m = &config->models[config->num_models];
  m->config_file = g_strdup (config_file);
  /* optional prebuilt-engine override (model/load engine ingest); NULL/"" => the
   * engine named in the nvinfer config is used. */
  m->engine_file = (engine_file && *engine_file) ? g_strdup (engine_file) : NULL;
  /* optional DEFT checkpoint version; NULL => provenance falls back to engine basename */
  m->version = (version && *version) ? g_strdup (version) : NULL;
  m->max_streams = max_streams ? max_streams : config->per_model_max_streams;
  /* auto-detect the inference backend from the config-file format (nvinfer vs nvinferserver) */
  m->type = modelmux_detect_model_type (config_file);
  m->name = g_strdup (name);          /* name last: the scan matches on it */
  config->num_models++;
  CFG_INFO ("registered model '%s' (infer-plugin=%s, config='%s')",
      name, modelmux_infer_plugin_str (m->type), config_file);
  return m;
}

const ModelCatalogEntry *
modelmux_config_update_model (ModelMuxConfig * config, const gchar * name,
    const gchar * config_file, const gchar * engine_file, const gchar * version)
{
  ModelCatalogEntry *m = NULL;
  guint i;
  if (!config || !name || !*name)
    return NULL;
  for (i = 0; i < config->num_models; i++)
    if (g_strcmp0 (config->models[i].name, name) == 0) {
      m = &config->models[i];
      break;
    }
  if (!m)
    return NULL;                                   /* not registered -> caller creates */
  /* OTA in-place artifact update: overwrite only what was supplied. */
  if (config_file && *config_file) {
    g_free (m->config_file);
    m->config_file = g_strdup (config_file);
  }
  if (engine_file && *engine_file) {
    g_free (m->engine_file);
    m->engine_file = g_strdup (engine_file);
  }
  if (version && *version) {
    g_free (m->version);
    m->version = g_strdup (version);
  }
  return m;
}

/* Write `out` to `path` unless the file already holds exactly this content.
 * Repeated resolutions of the same (name, version, engine) then do no I/O -- and
 * cannot needlessly clobber an identical derived config another process (sharing
 * the model volume) is currently serving from. Surfaces the real write error. */
static gboolean
modelmux_cfg_write_if_changed (const gchar * path, const GString * out)
{
  gchar *prev = NULL;
  gsize plen = 0;
  GError *err = NULL;
  GStatBuf st;
  /* size gate BEFORE reading: the fallback path lives in a predictable, possibly
   * world-writable tmp dir -- never slurp an arbitrary pre-existing file whole
   * just to compare (a size mismatch cannot be identical content anyway). */
  if (g_stat (path, &st) == 0 && st.st_size >= 0 &&
      (gsize) st.st_size == out->len &&
      g_file_get_contents (path, &prev, &plen, NULL) &&
      plen == out->len && memcmp (prev, out->str, plen) == 0) {
    g_free (prev);
    return TRUE;                       /* identical content already in place */
  }
  g_free (prev);
  if (!g_file_set_contents (path, out->str, -1, &err)) {
    CFG_WARN ("engine override: writing '%s' failed: %s", path,
        err ? err->message : "unknown error");
    g_clear_error (&err);
    return FALSE;
  }
  return TRUE;
}

/* Rewrite ONE key's line inside a config's text: REPLACE the first occurrence
 * (dropping duplicates -- the backends honor the LAST occurrence, so a leftover
 * base line would mask the override), or INJECT the key after `section` when
 * the base declares it nowhere. Grammar is caller-selected: INI 'key=value'
 * (colon_sep=FALSE) or colon 'key: value' (YAML mapping / protobuf text,
 * colon_sep=TRUE, injected preserving the section's member indentation).
 * Returns the rewritten text (caller frees); *applied says whether the key landed. */
static gchar *
cfg_rewrite_key (const gchar * content, const gchar * key, const gchar * value,
    gboolean colon_sep, const gchar * section, gboolean * applied)
{
  gchar **lines;
  GString *out;
  gsize klen = strlen (key);
  gboolean replaced = FALSE, has_line = FALSE;
  guint i;

  lines = g_strsplit (content, "\n", -1);
  /* pre-scan: decide replace-vs-inject up front (the section header precedes the key). */
  for (i = 0; lines[i]; i++) {
    gchar *t = g_strdup (lines[i]);
    g_strstrip (t);
    if (g_str_has_prefix (t, key) &&
        (t[klen] == '=' || t[klen] == ':' || g_ascii_isspace (t[klen]) || t[klen] == '\0')) {
      has_line = TRUE;
      g_free (t);
      break;
    }
    g_free (t);
  }
  out = g_string_new (NULL);
  for (i = 0; lines[i]; i++) {
    gchar *t = g_strdup (lines[i]);
    gboolean is_key;
    g_strstrip (t);
    is_key = (g_str_has_prefix (t, key) &&
        (t[klen] == '=' || t[klen] == ':' || g_ascii_isspace (t[klen]) || t[klen] == '\0'));
    if (is_key) {
      /* first occurrence -> the override; any further `key` lines are DROPPED. */
      if (!replaced) {
        if (colon_sep) {
          gsize ind = 0;      /* preserve indentation so the key stays inside its mapping */
          while (lines[i][ind] == ' ' || lines[i][ind] == '\t')
            ind++;
          g_string_append_len (out, lines[i], ind);
          g_string_append_printf (out, "%s: %s", key, value);
        } else {
          g_string_append_printf (out, "%s=%s", key, value);
        }
        if (lines[i + 1])
          g_string_append_c (out, '\n');
        replaced = TRUE;
      }
    } else {
      g_string_append (out, lines[i]);
      if (lines[i + 1])
        g_string_append_c (out, '\n');
      /* inject right after the section header ONLY when the base has no `key` line at all */
      if (!has_line && !replaced && g_str_has_prefix (t, section)) {
        if (colon_sep) {
          /* match the indentation of the section's first real member (default 2 spaces) so
           * the injected key stays a sibling inside the mapping/message regardless of the
           * file's style. */
          gsize ind = 2, k;
          for (k = i + 1; lines[k]; k++) {
            gchar *u = g_strdup (lines[k]);
            gboolean blank;
            g_strstrip (u);
            blank = (!*u || *u == '#');
            g_free (u);
            if (blank)
              continue;
            ind = 0;
            while (lines[k][ind] == ' ' || lines[k][ind] == '\t')
              ind++;
            if (ind == 0)             /* a col-0 sibling would fall OUT of the mapping -> force indent */
              ind = 2;
            break;
          }
          while (ind--)
            g_string_append_c (out, ' ');
          g_string_append_printf (out, "%s: %s\n", key, value);
        } else {
          g_string_append_printf (out, "%s=%s\n", key, value);
        }
        replaced = TRUE;
      }
    }
    g_free (t);
  }
  g_strfreev (lines);
  if (applied)
    *applied = replaced;
  return g_string_free (out, FALSE);
}

gchar *
modelmux_config_derive (const gchar * base_cfg, const gchar * engine, gint gpu,
    gint gie, const gchar * model_name, const gchar * version)
{
  gchar *content = NULL, *derived = NULL, *vtok, *p;
  GString *out;
  gboolean yaml, applied_any = FALSE, applied = FALSE;
  ModelType type;

  if (!base_cfg || !*base_cfg)
    return NULL;
  if (engine && !*engine)
    engine = NULL;
  if (!engine && gpu < 0 && gie <= 0)
    return NULL;                                     /* nothing to derive */
  /* the engine path is written verbatim as a config line: a newline in it would
   * inject extra key/value lines into the derived config. */
  if (engine && (strchr (engine, '\n') || strchr (engine, '\r'))) {
    CFG_WARN ("config derive: engine path contains a newline -- rejected");
    return NULL;
  }

  type = modelmux_detect_model_type (base_cfg);
  yaml = g_str_has_suffix (base_cfg, ".yml") || g_str_has_suffix (base_cfg, ".yaml");

  /* nvinferserver: a prebuilt .engine doesn't map -- versioned checkpoints live in the
   * Triton model repository. The gpu placement (gpu_ids) still derives cleanly. */
  if (!modelmux_backend (type)->engine_file && engine) {
    CFG_WARN ("config derive: an explicit .engine does not apply to nvinferserver "
        "('%s') -- Triton's model repository owns the checkpoints; ignoring the "
        "engine%s", base_cfg, gpu >= 0 ? " (gpu placement still applies)" : "");
    engine = NULL;
    if (gpu < 0)
      return NULL;
  }

  if (!g_file_get_contents (base_cfg, &content, NULL, NULL)) {
    CFG_WARN ("config derive: cannot read base config '%s'", base_cfg);
    return NULL;
  }

  if (modelmux_backend (type)->gpu_ids_list) {
    /* protobuf text 'infer_config {' or YAML 'infer_config:' -- both colon grammar */
    gchar *val = g_strdup_printf ("[%d]", gpu);
    gchar *next = cfg_rewrite_key (content, "gpu_ids", val, TRUE, "infer_config",
        &applied);
    g_free (content);
    g_free (val);
    content = next;
    if (!applied) {
      /* an EXPLICITLY requested placement that cannot be materialized must FAIL
       * the derivation -- proceeding would let status/routing claim gpu N while
       * Triton keeps running the base config's device (a silent lie). */
      GST_ERROR ("config derive FAILED: could NOT place gpu_ids into '%s' (no "
          "infer_config section) -- rejecting: the model would run on the "
          "config's device, not the requested gpu %d", base_cfg, gpu);
      g_free (content);
      return NULL;
    }
    applied_any |= applied;
  } else {
    /* Section and key names come from the BACKEND REGISTRY, so a backend that spells
     * its engine "tensorrt-engine" inside a [visionencoder] group derives through the
     * very same code as nvinfer's "model-engine-file" inside [property]. */
    const ModelBackendInfo *be = modelmux_backend (type);
    gchar *section = yaml ? g_strdup_printf ("%s:", be->cfg_section)
        : g_strdup_printf ("[%s]", be->cfg_section);

    if (engine && be->engine_prop) {
      gchar *next = cfg_rewrite_key (content, be->engine_prop, engine, yaml,
          section, &applied);
      g_free (content);
      content = next;
      /* an unapplied engine override would masquerade as a swap while running the
       * BASE engine -- fail LOUD, never acknowledge-and-ignore. */
      if (!applied) {
        GST_ERROR ("config derive FAILED: engine override could NOT be applied "
            "to '%s' (no %s line and no %s section) -- rejecting: "
            "the model would run the BASE engine, not '%s'", base_cfg,
            be->engine_prop, section, engine);
        g_free (section);
        g_free (content);
        return NULL;
      }
      applied_any |= applied;
    }
    /* gie-unique-id: bake the VERSION's declared id into its own derived config so
     * the backend reads the same identity modelmux assigns the bin -- one value, one
     * source, no divergence between the file on disk and the object meta. Skipped for
     * a backend with no gie identity: it has no such key to carry. */
    if (gie > 0 && be->gie_id) {
      gchar *val = g_strdup_printf ("%d", gie);
      gchar *next = cfg_rewrite_key (content, "gie-unique-id", val, yaml,
          section, &applied);
      g_free (content);
      g_free (val);
      content = next;
      if (!applied) {
        GST_ERROR ("config derive FAILED: could NOT place gie-unique-id into '%s' -- "
            "rejecting: the model would run under a different identity than the one "
            "declared for this version", base_cfg);
        g_free (section);
        g_free (content);
        return NULL;
      }
      applied_any |= applied;
    }
    if (gpu >= 0) {
      gchar *val = g_strdup_printf ("%d", gpu);
      gchar *next = cfg_rewrite_key (content, be->gpu_key, val, yaml,
          section, &applied);
      g_free (content);
      g_free (val);
      content = next;
      if (!applied) {
        GST_ERROR ("config derive FAILED: could NOT place %s into '%s' -- "
            "rejecting: the model would run on the config's device, not the "
            "requested gpu %d", be->gpu_key, base_cfg, gpu);
        g_free (section);
        g_free (content);
        return NULL;
      }
      applied_any |= applied;
    }
    g_free (section);
  }

  if (!applied_any) {                                /* nothing landed -> use the base */
    g_free (content);
    return NULL;
  }
  out = g_string_new (content);
  g_free (content);

  /* derived filename: one PER (name, version), next to the base config, with BOTH the model
   * name and version prefixed onto the base basename so it is self-describing, e.g.
   *   config_infer_primary.txt + (trafficcamnet, 1) -> trafficcamnet_1_config_infer_primary.txt
   * The name+version prefix keeps it per-instance and clear of the base/user configs. */
  vtok = g_strdup ((version && *version) ? version : "v");
  for (p = vtok; *p; p++)
    if (!g_ascii_isalnum (*p) && *p != '.' && *p != '-' && *p != '_')
      *p = '_';
  {
    gchar *ntok = g_strdup ((model_name && *model_name) ? model_name : "model");
    gchar *bdir = g_path_get_dirname (base_cfg);
    gchar *bn = g_path_get_basename (base_cfg);
    for (p = ntok; *p; p++)
      if (!g_ascii_isalnum (*p) && *p != '.' && *p != '-' && *p != '_')
        *p = '_';
    /* Prefer writing the derived config next to the base config (keeps any relative aux paths
     * resolvable). If that dir is not writable (e.g. base configs under a read-only /opt in a
     * deployment), fall back to a writable dir so the derivation STILL materializes -- otherwise
     * an A/B would silently run the base artifact twice. NOTE: relocation is only safe when the
     * base config's aux paths (labelfile/calib/model) are ABSOLUTE. */
    derived = g_strdup_printf ("%s/%s_%s_%s", bdir, ntok, vtok, bn);
    if (!modelmux_cfg_write_if_changed (derived, out)) {
      const gchar *env = g_getenv ("MM_DERIVED_CONFIG_DIR");
      gchar *odir = (env && *env) ? g_strdup (env) : g_strdup (g_get_tmp_dir ());
      gchar *alt = g_strdup_printf ("%s/%s_%s_%s", odir, ntok, vtok, bn);
      CFG_WARN ("config derive: base config dir '%s' not writable -> retrying derived config in "
          "'%s' (the base config's aux paths must be absolute for this to work)", bdir, odir);
      g_free (derived); derived = alt; g_free (odir);
      if (!modelmux_cfg_write_if_changed (derived, out)) {
        /* fail LOUD: do NOT silently degrade to the base artifact (that would make a
         * cross-checkpoint A/B score the same engine twice). */
        GST_ERROR ("config derive FAILED: cannot write derived config '%s' -- the engine/gpu "
            "derivation did NOT materialize and this model would fall back to the BASE "
            "config. Set MM_DERIVED_CONFIG_DIR to a writable path.", derived);
        g_free (derived); derived = NULL;
      } else {
        CFG_INFO ("config derive: derived config '%s' (engine=%s gpu=%d)", derived,
            engine ? engine : "(base)", gpu);
      }
    } else {
      CFG_INFO ("config derive: derived config '%s' (engine=%s gpu=%d)", derived,
          engine ? engine : "(base)", gpu);
    }
    g_free (ntok); g_free (bdir); g_free (bn);
  }
  g_free (vtok);
  g_string_free (out, TRUE);
  return derived;
}

void
modelmux_config_register_version (ModelMuxConfig * config, const gchar * name,
    const gchar * version, const gchar * engine, const gchar * config_file, gint gpu,
    guint batch)
{
  ModelVersionArtifact *v = NULL;
  guint i;

  if (!config || !name || !*name || !version || !*version)
    return;
  /* runtime mutation of versions[] -- concurrent with the CIVETWEB status
   * thread's modelmux_config_version_gpu scans (see ModelMuxConfig.versions_lock) */
  g_mutex_lock (&config->versions_lock);
  for (i = 0; i < config->num_versions; i++)
    if (g_strcmp0 (config->versions[i].name, name) == 0 &&
        g_strcmp0 (config->versions[i].version, version) == 0) {
      v = &config->versions[i];
      break;
    }
  if (!v) {
    if (config->num_versions >= G_N_ELEMENTS (config->versions)) {
      g_mutex_unlock (&config->versions_lock);
      CFG_WARN ("version registry full (max %u) -- '%s@%s' artifact not recorded "
          "(lazy re-warm would use the base config)",
          (guint) G_N_ELEMENTS (config->versions), name, version);
      return;
    }
    v = &config->versions[config->num_versions++];
    v->name = g_strdup (name);
    v->version = g_strdup (version);
    v->gpu = -1;
    v->batch = 0;                       /* unset -> name/global resolution */
  }
  if (engine && *engine) {
    g_free (v->engine);
    v->engine = g_strdup (engine);
  }
  if (config_file && *config_file) {
    g_free (v->config);
    v->config = g_strdup (config_file);
  }
  if (gpu >= 0)
    v->gpu = gpu;
  if (batch)
    v->batch = batch;                   /* 0 = leave as-is (see decl) */
  g_mutex_unlock (&config->versions_lock);
}

guint
modelmux_config_version_batch (const ModelMuxConfig * config, const gchar * name,
    const gchar * version)
{
  guint batch = 0;
  guint i;

  if (!config || !name || !version)
    return 0;
  /* same discipline as modelmux_config_version_gpu below: tiny locked scan,
   * runs on the CIVETWEB status thread too, never nests another lock. */
  g_mutex_lock ((GMutex *) &config->versions_lock);
  for (i = 0; i < config->num_versions; i++)
    if (g_strcmp0 (config->versions[i].name, name) == 0 &&
        g_strcmp0 (config->versions[i].version, version) == 0) {
      batch = config->versions[i].batch;
      break;
    }
  g_mutex_unlock ((GMutex *) &config->versions_lock);
  return batch;
}

/* Does @name already have a version OTHER than @version on record?
 *
 * The catalog's base config is a legitimate artifact source, but it can back only ONE
 * version: the first to claim it. Once a sibling exists, a version that brings no files
 * of its own would silently resolve to that same base -- two version numbers, one model.
 * This answers "is there a sibling", which is what turns a bare load from the normal
 * first-load flow into that collision.
 *
 * Same discipline as the scans above: tiny locked read, no nested locks. */
gboolean
modelmux_config_has_other_version (const ModelMuxConfig * config, const gchar * name,
    const gchar * version)
{
  gboolean other = FALSE;
  guint i;

  if (!config || !name || !*name)
    return FALSE;
  g_mutex_lock ((GMutex *) &config->versions_lock);
  for (i = 0; i < config->num_versions; i++)
    if (g_strcmp0 (config->versions[i].name, name) == 0 &&
        g_strcmp0 (config->versions[i].version, version) != 0) {
      other = TRUE;
      break;
    }
  g_mutex_unlock ((GMutex *) &config->versions_lock);
  return other;
}

gint
modelmux_config_version_gie (const ModelMuxConfig * config, const gchar * name,
    const gchar * version)
{
  gint gie = -1;
  guint i;

  if (!config || !name || !version)
    return -1;
  g_mutex_lock ((GMutex *) &config->versions_lock);
  for (i = 0; i < config->num_versions; i++)
    if (g_strcmp0 (config->versions[i].name, name) == 0 &&
        g_strcmp0 (config->versions[i].version, version) == 0) {
      gie = config->versions[i].gie;
      break;
    }
  g_mutex_unlock ((GMutex *) &config->versions_lock);
  return gie;
}

gint
modelmux_config_version_gpu (const ModelMuxConfig * config, const gchar * name,
    const gchar * version)
{
  gint gpu = -1;
  guint i;

  if (!config || !name || !version)
    return -1;
  /* runs on the CIVETWEB status thread too (model/status) -- the scan must
   * not race the main loop's register/unregister compaction (UAF on the
   * strings). Tiny critical section; never nests another lock (priv.h). */
  g_mutex_lock ((GMutex *) &config->versions_lock);
  for (i = 0; i < config->num_versions; i++)
    if (g_strcmp0 (config->versions[i].name, name) == 0 &&
        g_strcmp0 (config->versions[i].version, version) == 0) {
      gpu = config->versions[i].gpu;
      break;
    }
  g_mutex_unlock ((GMutex *) &config->versions_lock);
  return gpu;
}

void
modelmux_config_resolve_models (const ModelMuxConfig * config,
    const DefaultModelRef * def_primary, const DefaultModelRef * def_shadow,
    const gchar * camera_id,
    const gchar ** primary, const gchar ** shadow,
    const gchar ** primary_version, const gchar ** shadow_version,
    gboolean * primary_is_default, gboolean * shadow_is_default)
{
  /* config-time per-sensor binding wins (matched by camera id); any role it leaves
   * unset falls back to that role's RUNTIME default ref (the pool-embedded
   * designation). Each role carries its (name, version); an unset role inherits
   * BOTH name+version from the default. No binding => defaults. Bindings are
   * STATIC config; the defaults are runtime state (pool->default_ref) -- two
   * sources, one resolution (the config's default_* fields are only the SEED).
   * primary_is_default/shadow_is_default (optional): TRUE iff the role's value
   * came from the DEFAULT ref, not a binding -- a binding is an explicit pin
   * even when it names the same model as the default. */
  const gchar *dp = def_primary ? def_primary->name : NULL;
  const gchar *dpv = def_primary ? def_primary->version : NULL;
  const gchar *ds = def_shadow ? def_shadow->name : NULL;
  const gchar *dsv = def_shadow ? def_shadow->version : NULL;

  if (primary_is_default) *primary_is_default = TRUE;
  if (shadow_is_default)  *shadow_is_default = TRUE;
  if (camera_id) {
    guint i;
    for (i = 0; i < config->num_bindings; i++) {
      const StreamModelBinding *b = &config->bindings[i];
      if (g_strcmp0 (b->sensor_id, camera_id) == 0) {
        if (b->primary) {
          *primary = b->primary;
          *primary_version = b->primary_version;
          if (primary_is_default) *primary_is_default = FALSE;
        } else {
          *primary = dp;
          *primary_version = dpv;
        }
        if (b->shadow) {
          *shadow = b->shadow;
          *shadow_version = b->shadow_version;
          if (shadow_is_default) *shadow_is_default = FALSE;
        } else {
          *shadow = ds;
          *shadow_version = dsv;
        }
        return;
      }
    }
  }
  *primary = dp;
  *primary_version = dpv;
  *shadow = ds;                            /* NULL => shadow disabled, primary only */
  *shadow_version = dsv;
}

void
modelmux_config_clear (ModelMuxConfig * config)
{
  guint i;
  g_free (config->default_primary);
  g_free (config->default_primary_version);
  g_free (config->default_shadow);
  g_free (config->default_shadow_version);
  for (i = 0; i < config->num_models; i++) {
    g_free (config->models[i].name);
    g_free (config->models[i].config_file);
    g_free (config->models[i].engine_file);
    g_free (config->models[i].version);
  }
  for (i = 0; i < config->num_versions; i++) {
    g_free (config->versions[i].name);
    g_free (config->versions[i].version);
    g_free (config->versions[i].engine);
    g_free (config->versions[i].config);
  }
  /* pairs with the g_mutex_init at modelmux_config_parse / gst_modelmux_build defaults.
   * Teardown-time: the control plane is already detached, so no reader can
   * hold or contend the lock here. */
  g_mutex_clear (&config->versions_lock);
  for (i = 0; i < config->num_bindings; i++) {
    g_free (config->bindings[i].sensor_id);
    g_free (config->bindings[i].primary);
    g_free (config->bindings[i].primary_version);
    g_free (config->bindings[i].shadow);
    g_free (config->bindings[i].shadow_version);
  }
  memset (config, 0, sizeof (*config));
}
