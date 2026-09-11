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
 * Request-body parsing + transport-level admission for the MODEL plane
 * (/api/v1/model/load|unload|update) and the STREAM-ROUTING plane
 * (/api/v1/stream/route).
 *
 * Layering: this file enforces every constraint that is checkable from the
 * REQUEST ALONE (shapes, required fields, charset/format rules, intra-document
 * consistency). Constraints that need runtime state live elsewhere:
 *   - stream existence     -> nvmultiurisrcbin callback (owns camera_id map)
 *   - model loaded/version -> the consuming inference element
 * Every rejection fills err_info with a machine-readable code + actionable hint
 * (teaching errors); clients must never have to parse prose.
 */

#include "nvds_rest_server.h"
#include "nvds_parse.h"
#include <iostream>
#include <set>
#include <unistd.h>     /* access() -- verify config/engine files exist at the gate */

#define EMPTY_STRING ""

/* GPU ids are bounded 0..255 everywhere in this API (a sane device-index
 * ceiling that also keeps the guint casts below safe -- a huge JSON number
 * or map key must never silently truncate into a valid-looking id). */
#define GPU_ID_MAX 255

/* Parse an engine_files map key ("0", "1", ...) into a bounded gpu id.
 * Digits only, no more than 3 of them, value <= GPU_ID_MAX -- checked BEFORE
 * any conversion so std::stoul can neither throw nor overflow. Leading zeros
 * are rejected (mirroring is_valid_version): "0" and "00" would otherwise
 * silently collide on the same gpu id, last one wins. */
static bool
parse_gpu_id_key (const std::string & k, guint * out)
{
  if (k.empty () || k.size () > 3 ||
      k.find_first_not_of ("0123456789") != std::string::npos)
    return false;
  if (k.size () > 1 && k[0] == '0')
    return false;
  unsigned long val = std::stoul (k);
  if (val > GPU_ID_MAX)
    return false;
  *out = (guint) val;
  return true;
}

/* ------------------------------------------------------------------ */
/* Shared validators                                                    */
/* ------------------------------------------------------------------ */

/* Model versions are POSITIVE INTEGERS with no leading zero ("1", "2", ...,
 * "42") -- the Triton model-repository convention. */
static bool
is_valid_version (const std::string & v)
{
  if (v.empty () || v[0] == '0')
    return false;
  for (char c : v)
    if (c < '0' || c > '9')
      return false;
  return true;
}

/* Model names exclude the reserved separators: '@' (canonical instance key
 * "name@version") and ';' (config-file field separator). */
static bool
is_valid_model_name (const std::string & n)
{
  return !n.empty () &&
      n.find ('@') == std::string::npos && n.find (';') == std::string::npos;
}

static void
model_fail (NvDsServerModelInfo * info, NvDsServerModelStatus status,
    const std::string & code, const std::string & message,
    const std::string & hint)
{
  info->status = status;
  info->model_log = code + ", " + message;
  info->err_info.code = StatusBadRequest;
  info->err_info.err_code = code;
  info->err_info.err_log = { 400, message };
  info->err_info.hint = hint;
}

static void
route_fail (NvDsServerRouteInfo * info, const std::string & code,
    const std::string & message, const std::string & hint)
{
  info->status = STREAM_ROUTE_FAIL;
  info->route_log = code + ", " + message;
  info->err_info.code = StatusBadRequest;
  info->err_info.err_code = code;
  info->err_info.err_log = { 400, message };
  info->err_info.hint = hint;
}

/* ---------------------------------------------------------------------------
 * Response tail for this plane's async POSTs (model/load|unload, model/update,
 * stream/route).
 *
 * These sit HERE, beside model_fail / route_fail, and NOT in nvds_rest_server.cpp.
 * They are this endpoint group's error vocabulary, not HTTP transport: the core
 * server file stays a thin dispatcher, and the next endpoint adds its own tail to
 * its own *_parse.cpp rather than growing the shared lib body. (Kumar Abhishek,
 * review of nvds_rest_server.cpp -- helpers accumulating in the server file
 * invite every future endpoint author to do the same.)
 * ------------------------------------------------------------------------- */

/* Declared locally rather than added to nvds_parse.h: this is a pre-existing
 * server-side mapper and the shared header should not grow a legacy declaration
 * on our account. nvds_rest_server.cpp forward-declares it the same way. */
std::pair < int, std::string >
NvDsServerStatusCodeToHttpStatusCode (NvDsServerStatusCode code);

/* A syntactically valid request reached an endpoint whose callback was never
 * registered by the host app: fail loudly (503) instead of the silent 200 a
 * zero-initialized info struct would produce.
 *
 * This one CANNOT move into the callback -- it runs precisely because there is no
 * callback to run. Only the dispatcher knows a handler is missing, so the check
 * stays there; what lives here is the error text it reports. */
void
nvds_rest_ctl_fail_no_handler (NvDsServerErrorInfo & err, std::string & log,
    const char *what)
{
  err.code = NvDsServerStatusCode::StatusInternalServerError;
  err.err_code = "REQUEST_FAILED";
  err.err_log = { 500, std::string ("no handler attached for ") + what };
  err.hint = "the hosting element did not register this control callback";
  log = std::string ("REQUEST_FAILED, no handler attached for ") + what;
}

/* Machine-readable error envelope: on failure the response carries
 *   "error": { "code": "...", "message": "...", "hint": "..." }
 * so clients act on stable identifiers, never parse prose (the human-facing
 * "reason" string remains alongside). No-op when the request succeeded. */
static void
ctl_emit_error_envelope (Json::Value & response,
    const NvDsServerErrorInfo & err, const std::string & message)
{
  if (err.code == StatusOk || err.code == StatusAccepted)
    return;
  Json::Value e;
  e["code"] = err.err_code.empty () ? "REQUEST_FAILED" : err.err_code;
  e["message"] = message;
  if (!err.hint.empty ())
    e["hint"] = err.hint;
  response["error"] = e;
}

/* Shared tail of the async control-plane POST handlers: status line + reason +
 * machine-readable error envelope, and the wire-status propagation -- 202
 * Accepted for a forwarded async op, and FAILURES too (a parser 400 / no-handler
 * 500 must reach the wire status line, never a 200 with a body-only error). ONE
 * helper so the three tails cannot drift. */
NvDsServerStatusCode
nvds_rest_ctl_finish_async_response (Json::Value & response,
    const NvDsServerErrorInfo & err_info, const std::string & log)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  std::pair < int, std::string > http_err_code =
      NvDsServerStatusCodeToHttpStatusCode (err_info.code);

  response["status"] = std::string ("HTTP/1.1 ") +
      std::to_string (http_err_code.first) + " " + http_err_code.second;
  response["reason"] = log;
  ctl_emit_error_envelope (response, err_info, log);
  if (err_info.code == NvDsServerStatusCode::StatusAccepted)
    ret = NvDsServerStatusCode::StatusAccepted;
  else if (err_info.code != NvDsServerStatusCode::StatusOk)
    ret = err_info.code;
  return ret;
}

/* Reject a request that carries a field from a DIFFERENT plane / a retired
 * schema, with a hint at the right endpoint. No silent field-dropping: an
 * operator who sends a routing field to the model plane must learn, not guess. */
static bool
model_reject_foreign_fields (const Json::Value & v, NvDsServerModelInfo * info,
    NvDsServerModelStatus fail_status)
{
  static const char *routing_fields[] = { "primary_model", "shadow_model",
    "primary_model_version", "shadow_model_version", "stream_list",
    "set_default_primary", "set_default_shadow", "routes", "default", NULL
  };
  for (int i = 0; routing_fields[i]; i++) {
    if (v.isMember (routing_fields[i])) {
      model_fail (info, fail_status, "FIELD_WRONG_PLANE",
          std::string ("field '") + routing_fields[i] +
          "' does not belong to the model plane",
          "stream routing (targets, defaults) is POST /api/v1/stream/route");
      return false;
    }
  }
  return true;
}


/* ------------------------------------------------------------------ */
/* Strict field access.                                                */
/*                                                                     */
/* jsoncpp's asString()/asBool() silently coerce numerics and throw an */
/* unnamed LogicError on objects/arrays; neither teaches the caller    */
/* anything. Every field is therefore fetched with an exact type check */
/* that names the field, and every object is validated against the     */
/* closed set of fields its schema defines -- a typo ("verion") is an  */
/* error, never a silent no-op.                                        */
/* ------------------------------------------------------------------ */
static bool
get_string_field (const Json::Value & v, const char *field,
    std::string * out, std::string * err)
{
  out->clear ();
  if (!v.isMember (field) || v[field].isNull ())
    return true;
  if (!v[field].isString ()) {
    *err = std::string ("'") + field + "' must be a JSON string";
    return false;
  }
  *out = v[field].asString ();
  return true;
}

static bool
find_unknown_field (const Json::Value & v, const char *const allowed[],
    std::string * unknown)
{
  for (const std::string & m : v.getMemberNames ()) {
    bool known = false;
    for (int i = 0; allowed[i] && !known; i++)
      known = (m == allowed[i]);
    if (!known) {
      *unknown = m;
      return true;
    }
  }
  return false;
}

/* the closed field sets, one per schema object */
static const char *const REQUEST_FIELDS[] = { "key", "value", NULL };
static const char *const MODEL_LOAD_FIELDS[] = { "name", "version",
  "config_file", "engine_file", "engine_files", "gpus",
  "batch_size", NULL
};
static const char *const MODEL_UNLOAD_FIELDS[] = { "name", "version", "gpus", NULL };
static const char *const MODEL_UPDATE_FIELDS[] = { "name", "from_version",
  "version", "engine_file", "engine_files", "batch_size", NULL
};
static const char *const ROUTE_DOC_FIELDS[] =
    { "if_revision", "routes", "default", NULL };
static const char *const ROUTE_ENTRY_FIELDS[] =
    { "streams", "model", "shadow", NULL };
static const char *const ROUTE_DEFAULT_FIELDS[] = { "model", "shadow", NULL };
static const char *const REF_FIELDS[] = { "name", "version", "gpu", NULL };

/* the shared {"key", "value"} request envelope: body must be a JSON object
 * with an object "value"; anything else (unparseable body arrives here as
 * null) is rejected with a precise reason instead of a jsoncpp exception. */
static bool
check_request_envelope (const Json::Value & in, std::string * key,
    std::string * err, std::string * err_code)
{
  if (!in.isObject () || in.isNull ()) {
    *err_code = "MALFORMED_JSON";
    *err = "request body must be a JSON object: {\"key\": ..., \"value\": {...}}";
    return false;
  }
  std::string unknown;
  if (find_unknown_field (in, REQUEST_FIELDS, &unknown)) {
    *err_code = "FIELD_UNKNOWN";
    *err = "unknown top-level field '" + unknown + "'";
    return false;
  }
  if (!get_string_field (in, "key", key, err)) {
    *err_code = "FIELD_TYPE_INVALID";
    return false;
  }
  if (!in.isMember ("value") || !in["value"].isObject () ||
      in["value"].isNull ()) {
    *err_code = "EMPTY_REQUEST";
    *err = "request 'value' object missing or not an object";
    return false;
  }
  return true;
}

/* Parse a structured model ref {"name": ..., "version": ..., "gpu": N?} into
 * @ref. Returns false (and fills @err/@err_hint) on shape violations. */
static bool
parse_route_ref (const Json::Value & v, NvDsServerRouteRef * ref,
    std::string * err, std::string * err_code)
{
  if (!v.isObject ()) {
    *err_code = "REF_INVALID";
    *err = "a model ref must be an object {\"name\", \"version\"[, \"gpu\"]}";
    return false;
  }
  std::string unknown;
  if (find_unknown_field (v, REF_FIELDS, &unknown)) {
    *err_code = "FIELD_UNKNOWN";
    *err = "unknown ref field '" + unknown +
        "' (a ref takes name, version, gpu)";
    return false;
  }
  if (!get_string_field (v, "name", &ref->name, err) ||
      !get_string_field (v, "version", &ref->version, err)) {
    *err_code = "FIELD_TYPE_INVALID";
    return false;
  }
  ref->gpu = -1;
  if (v.isMember ("gpu")) {
    /* asInt64 (not asInt) so an out-of-int value is range-checked here
     * instead of throwing into the generic PARSE_ERROR path */
    if (!v["gpu"].isIntegral () || v["gpu"].asInt64 () < 0) {
      *err_code = "FIELD_TYPE_INVALID";
      *err = "ref 'gpu' must be a non-negative integer";
      return false;
    }
    if (v["gpu"].asInt64 () > GPU_ID_MAX) {
      *err_code = "FIELD_TYPE_INVALID";
      *err = "ref 'gpu' id out of range (0..255)";
      return false;
    }
    ref->gpu = (gint) v["gpu"].asInt64 ();
  }
  if (!is_valid_model_name (ref->name)) {
    *err_code = "NAME_INVALID";
    *err = "ref 'name' is required and must not contain '@' or ';'";
    return false;
  }
  if (!is_valid_version (ref->version)) {
    *err_code = "VERSION_INVALID";
    *err = "ref 'version' is required and must be a positive integer";
    return false;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* /api/v1/model/load  +  /api/v1/model/unload                         */
/*                                                                     */
/*   load:   { "key":"model", "value": {                               */
/*               "name":"...", "version":"N",                          */
/*               "config_file":"..."?,   (required on first load)      */
/*               "engine_file":"..."?,   (SEED engine)                 */
/*               "engine_files":{"0":"...","1":"..."}?, (EXACT map)    */
/*               "gpus":[0,1]? } }         (one instance per gpu id)   */
/*   unload: { "key":"model", "value": { "name":"...", "version":"N"?, */
/*               "gpus":[1]? } }   (gpus? = drain that subset; else all) */
/* The action is decided by the endpoint (model_info->uri).            */
/* ------------------------------------------------------------------ */
bool
nvds_rest_model_parse (const Json::Value & in, NvDsServerModelInfo * model_info)
{
  bool is_load = (model_info->uri.find ("/model/load") != std::string::npos);
  NvDsServerModelStatus fail = is_load ? MODEL_LOAD_FAIL : MODEL_UNLOAD_FAIL;

  if (model_info->uri.find ("/api/v1/") == std::string::npos) {
    g_print ("Unsupported REST API version\n");
    return true;
  }

  try {
    std::string terr, tcode;
    if (!check_request_envelope (in, &model_info->key, &terr, &tcode)) {
      model_fail (model_info, fail, tcode, terr,
          "see the API design doc for the model/load|unload body");
      return false;
    }
    const Json::Value & v = in["value"];
    if (!model_reject_foreign_fields (v, model_info, fail))
      return false;
    /* closed schema: any field outside the documented set (including the
     * retired model_* names) is an error, never a silently dropped typo */
    std::string unknown;
    if (find_unknown_field (v,
            is_load ? MODEL_LOAD_FIELDS : MODEL_UNLOAD_FIELDS, &unknown)) {
      model_fail (model_info, fail, "FIELD_UNKNOWN",
          std::string ("unknown field '") + unknown + "'",
          is_load ? "model/load accepts: name, version, config_file, "
          "engine_file, engine_files, gpus, batch_size"
          : "model/unload accepts: name, version, gpus");
      return false;
    }

    if (!get_string_field (v, "name", &model_info->name, &terr) ||
        !get_string_field (v, "version", &model_info->version, &terr) ||
        !get_string_field (v, "config_file", &model_info->config_file, &terr) ||
        !get_string_field (v, "engine_file", &model_info->engine_file, &terr)) {
      model_fail (model_info, fail, "FIELD_TYPE_INVALID", terr,
          "paths and versions are JSON strings (\"version\": \"3\")");
      return false;
    }
    model_info->has_gpus = FALSE;
    model_info->gpus.clear ();
    model_info->engine_files.clear ();

    if (!is_valid_model_name (model_info->name)) {
      model_fail (model_info, fail, "NAME_INVALID",
          "'name' is required and must not contain '@' or ';'",
          "'@' and ';' are reserved separators");
      return false;
    }

    /* version: REQUIRED positive integer for load; optional for unload
     * (name-only unload = all non-serving versions). */
    if (is_load && !is_valid_version (model_info->version)) {
      model_fail (model_info, MODEL_LOAD_FAIL, "VERSION_INVALID",
          "'version' is required and must be a positive integer",
          "versions are immutable checkpoints; a new checkpoint is a new version");
      return false;
    }
    if (!is_load && !model_info->version.empty () &&
        !is_valid_version (model_info->version)) {
      model_fail (model_info, MODEL_UNLOAD_FAIL, "VERSION_INVALID",
          "'version' must be a positive integer when given",
          "omit 'version' to unload all non-serving versions of the name");
      return false;
    }

    /* gpus:[N,...]: OPTIONAL explicit device set (one instance per gpu id).
     *   load   -> declarative instance placement: one pinned instance per gpu.
     *   unload -> the subset of a version's per-gpu instances to drain
     *             (omit -> the whole version).
     * Shared by both so the noun to PLACE a model and to DRAIN one is identical.
     * ATOMIC admission: any present-but-invalid piece rejects the WHOLE request
     * (never silently truncate a placement/drain the caller stated). */
    if (v.isMember ("gpus")) {
      const Json::Value & gs = v["gpus"];
      if (!gs.isArray () || gs.empty ()) {
        model_fail (model_info, fail, "GPUS_INVALID",
            "'gpus' must be a non-empty array of gpu ids",
            is_load ? "omit 'gpus' for a single instance on the config's GPU"
                    : "omit 'gpus' to unload the whole version");
        return false;
      }
      /* per-request device cap -- never accept a set the backend would truncate */
      if (gs.size () > 16u) {
        model_fail (model_info, fail, "GPUS_INVALID",
            "'gpus' lists more than 16 devices",
            "at most 16 gpu ids per request");
        return false;
      }
      std::set<guint> seen;
      for (Json::ArrayIndex i = 0; i < gs.size (); ++i) {
        if (!gs[i].isIntegral () || gs[i].asInt64 () < 0) {
          model_fail (model_info, fail, "GPUS_INVALID",
              "each 'gpus' entry must be a non-negative integer gpu id",
              "example: \"gpus\": [0, 1]");
          return false;
        }
        if (gs[i].asInt64 () > GPU_ID_MAX) {
          model_fail (model_info, fail, "GPUS_INVALID",
              "gpu id out of range (0..255)",
              "gpu ids are bounded 0..255");
          return false;
        }
        guint gpu = (guint) gs[i].asInt64 ();
        if (!seen.insert (gpu).second) {
          model_fail (model_info, fail, "GPUS_INVALID",
              "duplicate gpu id in 'gpus'",
              "list each device at most once (one instance per gpu)");
          return false;
        }
        model_info->gpus.push_back (gpu);
      }
      model_info->has_gpus = TRUE;
    }

    if (is_load) {

      /* engine forms are mutually exclusive: scalar SEED vs per-GPU map */
      if (!model_info->engine_file.empty () && v.isMember ("engine_files")) {
        model_fail (model_info, MODEL_LOAD_FAIL, "ENGINE_FILES_INVALID",
            "'engine_file' and 'engine_files' are mutually exclusive",
            "'engine_file' seeds a single instance; 'engine_files' is the "
            "exact per-GPU map for multi-device loads");
        return false;
      }
      /* engine_files: EXACT per-GPU map ("gpu-id string" -> path) */
      if (v.isMember ("engine_files")) {
        const Json::Value & ef = v["engine_files"];
        if (!ef.isObject () || ef.empty ()) {
          model_fail (model_info, MODEL_LOAD_FAIL, "ENGINE_FILES_INVALID",
              "'engine_files' must be a non-empty object mapping gpu id to path",
              "example: \"engine_files\": {\"0\": \"...gpu0.engine\"}");
          return false;
        }
        for (Json::ValueConstIterator it = ef.begin (); it != ef.end (); ++it) {
          std::string k = it.key ().asString ();
          guint gpu_key = 0;
          if (!parse_gpu_id_key (k, &gpu_key)) {
            model_fail (model_info, MODEL_LOAD_FAIL, "ENGINE_FILES_INVALID",
                "'engine_files' keys must be gpu ids in 0..255 "
                "(\"0\", \"1\", ...; no leading zeros)",
                "one engine per device -- TRT engines are device-built");
            return false;
          }
          if (!(*it).isString ()) {
            model_fail (model_info, MODEL_LOAD_FAIL, "FIELD_TYPE_INVALID",
                "'engine_files' values must be engine path strings",
                "example: \"engine_files\": {\"0\": \"...gpu0.engine\"}");
            return false;
          }
          std::string path = (*it).asString ();
          if (path.empty () || access (path.c_str (), R_OK) != 0) {
            model_fail (model_info, MODEL_LOAD_FAIL, "GPU_ENGINE_MISSING",
                "engine for gpu " + k + " not found/readable: " + path,
                "every mapped engine must exist and be readable");
            return false;
          }
          model_info->engine_files[gpu_key] = path;
        }
        /* EXACT COVERAGE: a mapped multi-instance load must name an engine for
         * EVERY requested device (TRT engines are device-built; a missing one
         * would silently auto-build or fail at warm). */
        for (guint g : model_info->gpus) {
          if (model_info->engine_files.find (g) == model_info->engine_files.end ()) {
            model_fail (model_info, MODEL_LOAD_FAIL, "GPU_ENGINE_MISSING",
                "engine_files has no engine for gpu " + std::to_string (g),
                "coverage must be exact: one prebuilt engine per device in "
                "'gpus'");
            return false;
          }
        }
      }

      /* file-existence gates for the paths that were actually supplied */
      if (!model_info->config_file.empty () &&
          access (model_info->config_file.c_str (), R_OK) != 0) {
        model_fail (model_info, MODEL_LOAD_FAIL, "CONFIG_NOT_FOUND",
            "config_file not found/readable: " + model_info->config_file,
            "config_file is required on the FIRST load of a name");
        return false;
      }
      if (!model_info->engine_file.empty () &&
          access (model_info->engine_file.c_str (), R_OK) != 0) {
        model_fail (model_info, MODEL_LOAD_FAIL, "GPU_ENGINE_MISSING",
            "engine_file not found/readable: " + model_info->engine_file,
            "the SEED engine must exist; omit it to build from the config");
        return false;
      }
    }

    /* carry the value object verbatim -- the in-band transport payload */
    Json::StreamWriterBuilder wb;
    wb["indentation"] = "";
    model_info->value_json = Json::writeString (wb, v);
  } catch (const std::exception & e) {
    model_fail (model_info, fail, "PARSE_ERROR",
        std::string ("malformed request: ") + e.what (),
        "see the API design doc for the model plane body shapes");
    return false;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* /api/v1/model/update -- IN-PLACE checkpoint transition               */
/*                                                                     */
/*   { "key":"model", "value": {                                       */
/*       "name":"...",                                                 */
/*       "from_version":"N",           (REQUIRED CAS guard)            */
/*       "version":"M",                (REQUIRED, new)                 */
/*       "engine_file":"..."           (single-instance groups)        */
/*     | "engine_files":{"0":"...", ...} (multi-GPU: exact map) } }    */
/*                                                                     */
/* Same network, new weights; identity moves from_version -> version   */
/* atomically on group confirm. Streams/roles/placement are NEVER      */
/* touched (that is the stream plane's job).                           */
/* ------------------------------------------------------------------ */
bool
nvds_rest_model_update_parse (const Json::Value & in,
    NvDsServerModelInfo * model_info)
{
  if (model_info->uri.find ("/api/v1/") == std::string::npos) {
    g_print ("Unsupported REST API version\n");
    return true;
  }

  try {
    std::string terr, tcode;
    if (!check_request_envelope (in, &model_info->key, &terr, &tcode)) {
      model_fail (model_info, MODEL_UPDATE_FAIL, tcode, terr,
          "model/update body: name, from_version, version, engine_file | engine_files");
      return false;
    }
    const Json::Value & v = in["value"];
    /* the OLD model/update was a stream-routing API; that body is a hard error */
    if (!model_reject_foreign_fields (v, model_info, MODEL_UPDATE_FAIL))
      return false;
    std::string unknown;
    if (find_unknown_field (v, MODEL_UPDATE_FIELDS, &unknown)) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "FIELD_UNKNOWN",
          std::string ("unknown field '") + unknown + "'",
          "model/update accepts: name, from_version, version, "
          "engine_file | engine_files, batch_size");
      return false;
    }

    if (!get_string_field (v, "name", &model_info->name, &terr) ||
        !get_string_field (v, "from_version", &model_info->from_version, &terr) ||
        !get_string_field (v, "version", &model_info->version, &terr) ||
        !get_string_field (v, "engine_file", &model_info->engine_file, &terr)) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "FIELD_TYPE_INVALID", terr,
          "paths and versions are JSON strings (\"version\": \"3\")");
      return false;
    }
    model_info->engine_files.clear ();

    if (!is_valid_model_name (model_info->name)) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "NAME_INVALID",
          "'name' is required and must not contain '@' or ';'",
          "'@' and ';' are reserved separators");
      return false;
    }
    if (!is_valid_version (model_info->version)) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "UPDATE_VERSION_REQUIRED",
          "'version' is required and must be a positive integer",
          "identity must move -- an in-place update always carries a NEW version");
      return false;
    }
    if (!is_valid_version (model_info->from_version)) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "UPDATE_FROM_REQUIRED",
          "'from_version' is required and must be a positive integer",
          "state the precondition: an update without it is a blind write");
      return false;
    }
    if (model_info->from_version == model_info->version) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "VERSION_INVALID",
          "'version' must differ from 'from_version'",
          "re-issuing a COMPLETED update returns success as a no-op instead");
      return false;
    }

    bool has_scalar = !model_info->engine_file.empty ();
    bool has_map = v.isMember ("engine_files");
    if (has_scalar == has_map) {        /* both or neither */
      model_fail (model_info, MODEL_UPDATE_FAIL, "GPU_ENGINE_MISSING",
          "exactly one of 'engine_file' (single-instance) or 'engine_files' "
          "(per-GPU map) is required",
          "model/update never auto-builds -- provide the prebuilt engine(s)");
      return false;
    }
    if (has_scalar && access (model_info->engine_file.c_str (), R_OK) != 0) {
      model_fail (model_info, MODEL_UPDATE_FAIL, "GPU_ENGINE_MISSING",
          "engine_file not found/readable: " + model_info->engine_file,
          "the update engine must be prebuilt and readable");
      return false;
    }
    if (has_map) {
      const Json::Value & ef = v["engine_files"];
      if (!ef.isObject () || ef.empty ()) {
        model_fail (model_info, MODEL_UPDATE_FAIL, "ENGINE_FILES_INVALID",
            "'engine_files' must be a non-empty object mapping gpu id to path",
            "example: \"engine_files\": {\"0\": \"...gpu0.engine\"}");
        return false;
      }
      for (Json::ValueConstIterator it = ef.begin (); it != ef.end (); ++it) {
        std::string k = it.key ().asString ();
        guint gpu_key = 0;
        if (!parse_gpu_id_key (k, &gpu_key)) {
          model_fail (model_info, MODEL_UPDATE_FAIL, "ENGINE_FILES_INVALID",
              "'engine_files' keys must be gpu ids in 0..255 "
              "(\"0\", \"1\", ...; no leading zeros)",
              "gpu ids appear only as engine map keys, never as selectors");
          return false;
        }
        if (!(*it).isString ()) {
          model_fail (model_info, MODEL_UPDATE_FAIL, "FIELD_TYPE_INVALID",
              "'engine_files' values must be engine path strings",
              "example: \"engine_files\": {\"0\": \"...gpu0.engine\"}");
          return false;
        }
        std::string path = (*it).asString ();
        if (path.empty () || access (path.c_str (), R_OK) != 0) {
          model_fail (model_info, MODEL_UPDATE_FAIL, "GPU_ENGINE_MISSING",
              "engine for gpu " + k + " not found/readable: " + path,
              "coverage must be exact: one prebuilt engine per device in the group");
          return false;
        }
        model_info->engine_files[gpu_key] = path;
      }
    }

    Json::StreamWriterBuilder wb;
    wb["indentation"] = "";
    model_info->value_json = Json::writeString (wb, v);
  } catch (const std::exception & e) {
    model_fail (model_info, MODEL_UPDATE_FAIL, "PARSE_ERROR",
        std::string ("malformed request: ") + e.what (),
        "model/update body: name, from_version, version, engine_file | engine_files");
    return false;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* /api/v1/stream/route -- the declarative routing plane                */
/*                                                                     */
/*   { "key":"stream", "value": {                                      */
/*       "if_revision": N?,                                            */
/*       "routes": [ { "streams": ["cam-1", ...] | "all",              */
/*                     "model":  {"name","version"[,"gpu"]} | null,    */
/*                     "shadow": {"name","version"[,"gpu"]} | null }?, */
/*                   ... ]?,                                           */
/*       "default": { "model": {...}?, "shadow": {...}|null? }? } }    */
/*                                                                     */
/* ADMISSION IS ATOMIC: any violation rejects the WHOLE request.       */
/* ------------------------------------------------------------------ */
bool
nvds_rest_stream_route_parse (const Json::Value & in,
    NvDsServerRouteInfo * route_info)
{
  if (route_info->uri.find ("/api/v1/") == std::string::npos) {
    g_print ("Unsupported REST API version\n");
    return true;
  }

  try {
    std::string terr, tcode;
    if (!check_request_envelope (in, &route_info->key, &terr, &tcode)) {
      route_fail (route_info, tcode, terr,
          "stream/route body: routes[] and/or default{}, optional if_revision");
      return false;
    }
    route_info->if_revision = -1;
    route_info->routes.clear ();
    route_info->has_default = FALSE;
    route_info->default_has_model = FALSE;
    route_info->default_has_shadow = FALSE;
    route_info->default_shadow_null = FALSE;

    const Json::Value & v = in["value"];
    std::string unknown;
    if (find_unknown_field (v, ROUTE_DOC_FIELDS, &unknown)) {
      route_fail (route_info, "FIELD_UNKNOWN",
          "unknown field '" + unknown + "'",
          "stream/route accepts: if_revision, routes, default");
      return false;
    }

    if (v.isMember ("if_revision")) {
      if (!v["if_revision"].isIntegral () || v["if_revision"].asInt64 () < 0) {
        route_fail (route_info, "FIELD_TYPE_INVALID",
            "'if_revision' must be a non-negative integer",
            "read routing_revision from GET /api/v1/stream/route");
        return false;
      }
      route_info->if_revision = v["if_revision"].asInt64 ();
    }

    bool has_routes = v.isMember ("routes");
    route_info->has_default = v.isMember ("default");
    if (!has_routes && !route_info->has_default) {
      route_fail (route_info, "EMPTY_REQUEST",
          "at least one of 'routes' / 'default' is required",
          "an empty routing request is an error, not a no-op");
      return false;
    }

    std::set<std::string> seen_streams;   /* STREAM_DUPLICATE_ROUTE, incl. "all" */
    bool seen_all = false;

    if (has_routes) {
      const Json::Value & routes = v["routes"];
      if (!routes.isArray () || routes.empty ()) {
        route_fail (route_info, "EMPTY_REQUEST",
            "'routes' must be a non-empty array",
            "omit 'routes' entirely when only moving the default");
        return false;
      }
      for (Json::ArrayIndex i = 0; i < routes.size (); ++i) {
        const Json::Value & r = routes[i];
        NvDsServerRouteEntry entry;
        entry.all_streams = FALSE;
        entry.has_model = entry.model_null = FALSE;
        entry.has_shadow = entry.shadow_null = FALSE;
        entry.model.gpu = entry.shadow.gpu = -1;

        if (!r.isObject ()) {
          route_fail (route_info, "PARSE_ERROR",
              "each route must be an object", "see the stream/route body shape");
          return false;
        }
        if (find_unknown_field (r, ROUTE_ENTRY_FIELDS, &unknown)) {
          route_fail (route_info, "FIELD_UNKNOWN",
              "route[" + std::to_string (i) + "]: unknown field '" + unknown +
              "'", "a route accepts: streams, model, shadow");
          return false;
        }

        /* streams: REQUIRED -- an explicit camera_id list, or the literal
         * "all". An omitted list must FAIL, never mean "everything". */
        if (!r.isMember ("streams")) {
          route_fail (route_info, "STREAMS_REQUIRED",
              "route[" + std::to_string (i) + "] is missing 'streams'",
              "give an explicit camera_id list, or the literal \"all\"");
          return false;
        }
        const Json::Value & streams = r["streams"];
        if (streams.isString () && streams.asString () == "all") {
          entry.all_streams = TRUE;
          if (seen_all || !seen_streams.empty ()) {
            route_fail (route_info, "STREAM_DUPLICATE_ROUTE",
                "\"all\" cannot be combined with other routes",
                "an \"all\" route already covers every stream");
            return false;
          }
          seen_all = true;
        } else if (streams.isArray () && !streams.empty ()) {
          if (seen_all) {
            route_fail (route_info, "STREAM_DUPLICATE_ROUTE",
                "\"all\" cannot be combined with other routes",
                "an \"all\" route already covers every stream");
            return false;
          }
          for (Json::ArrayIndex s = 0; s < streams.size (); ++s) {
            if (!streams[s].isString ()) {
              route_fail (route_info, "FIELD_TYPE_INVALID",
                  "route[" + std::to_string (i) +
                  "] 'streams' entries must be camera_id strings",
                  "'streams' entries are the stable camera_ids from stream/add");
              return false;
            }
            std::string cam = streams[s].asString ();
            if (cam.empty ()) {
              route_fail (route_info, "STREAM_UNKNOWN",
                  "empty camera_id in route[" + std::to_string (i) + "]",
                  "'streams' entries are the stable camera_ids from stream/add");
              return false;
            }
            if (!seen_streams.insert (cam).second) {
              route_fail (route_info, "STREAM_DUPLICATE_ROUTE",
                  "stream '" + cam + "' appears in more than one route",
                  "a stream's desired state must be stated exactly once");
              return false;
            }
            entry.camera_ids.push_back (cam);
          }
        } else {
          route_fail (route_info, "STREAMS_REQUIRED",
              "route[" + std::to_string (i) +
              "] 'streams' must be a non-empty camera_id list or \"all\"",
              "an omitted/empty list never means \"everything\"");
          return false;
        }

        /* model: omitted => unchanged; null => PASSTHROUGH; ref => target */
        if (r.isMember ("model")) {
          entry.has_model = TRUE;
          if (r["model"].isNull ()) {
            /* explicit passthrough is designed but not yet applied by the
             * element -- reject HERE so the client learns synchronously
             * instead of getting a 200 for a no-op. */
            route_fail (route_info, "NOT_IMPLEMENTED",
                "route[" + std::to_string (i) +
                "]: \"model\": null (passthrough) is not available yet",
                "route the streams to a serving model instead");
            return false;
          } else {
            std::string err, code;
            if (!parse_route_ref (r["model"], &entry.model, &err, &code)) {
              route_fail (route_info, code,
                  "route[" + std::to_string (i) + "] model: " + err,
                  "refs are {\"name\": ..., \"version\": ...[, \"gpu\": N]}");
              return false;
            }
          }
        }
        /* shadow: omitted => unchanged; null => clear; ref => mirror target */
        if (r.isMember ("shadow")) {
          entry.has_shadow = TRUE;
          if (r["shadow"].isNull ()) {
            entry.shadow_null = TRUE;
          } else {
            std::string err, code;
            if (!parse_route_ref (r["shadow"], &entry.shadow, &err, &code)) {
              route_fail (route_info, code,
                  "route[" + std::to_string (i) + "] shadow: " + err,
                  "refs are {\"name\": ..., \"version\": ...[, \"gpu\": N]}");
              return false;
            }
          }
        }

        /* CONSTRAINT: a route always resolves to a serving model. A shadow
         * cannot mirror nothing -- shadow set together with explicit
         * passthrough ("model": null) is structurally invalid. (shadow set
         * with model OMITTED is checked against the stream's CURRENT serving
         * model at element-level admission -- state this layer cannot see.) */
        if (entry.model_null && entry.has_shadow && !entry.shadow_null) {
          route_fail (route_info, "SHADOW_WITHOUT_MODEL",
              "route[" + std::to_string (i) +
              "] sets a shadow while turning the serving model off",
              "a shadow mirrors a serving model -- set 'model' too, or clear "
              "the shadow with \"shadow\": null");
          return false;
        }
        /* a route that changes NOTHING is a mistake, not a no-op */
        if (!entry.has_model && !entry.has_shadow) {
          route_fail (route_info, "EMPTY_REQUEST",
              "route[" + std::to_string (i) +
              "] names streams but changes neither 'model' nor 'shadow'",
              "state the desired 'model' and/or 'shadow' for the group");
          return false;
        }
        route_info->routes.push_back (entry);
      }
    }

    if (route_info->has_default) {
      const Json::Value & d = v["default"];
      if (!d.isObject ()) {
        route_fail (route_info, "PARSE_ERROR",
            "'default' must be an object",
            "default body: {\"model\": {...}?, \"shadow\": {...}|null?}");
        return false;
      }
      if (find_unknown_field (d, ROUTE_DEFAULT_FIELDS, &unknown)) {
        route_fail (route_info, "FIELD_UNKNOWN",
            "default: unknown field '" + unknown + "'",
            "'default' accepts: model, shadow");
        return false;
      }
      if (d.isMember ("model")) {
        if (d["model"].isNull ()) {
          route_fail (route_info, "SHADOW_WITHOUT_MODEL",
              "the default model cannot be null",
              "the default is the fallback serving model; it must be a real ref");
          return false;
        }
        std::string err, code;
        if (!parse_route_ref (d["model"], &route_info->default_model, &err, &code)) {
          route_fail (route_info, code, "default model: " + err,
              "refs are {\"name\": ..., \"version\": ...[, \"gpu\": N]}");
          return false;
        }
        route_info->default_has_model = TRUE;
      }
      if (d.isMember ("shadow")) {
        route_info->default_has_shadow = TRUE;
        if (d["shadow"].isNull ()) {
          route_fail (route_info, "NOT_IMPLEMENTED",
              "default \"shadow\": null (clear the default shadow) is not "
              "available yet",
              "per-stream shadow clear works today: route the streams with "
              "\"shadow\": null");
          return false;
        } else {
          std::string err, code;
          if (!parse_route_ref (d["shadow"], &route_info->default_shadow,
                  &err, &code)) {
            route_fail (route_info, code, "default shadow: " + err,
                "refs are {\"name\": ..., \"version\": ...[, \"gpu\": N]}");
            return false;
          }
        }
      }
      if (!route_info->default_has_model && !route_info->default_has_shadow) {
        route_fail (route_info, "EMPTY_REQUEST",
            "'default' is present but changes neither 'model' nor 'shadow'",
            "state the desired default 'model' and/or 'shadow'");
        return false;
      }
    }

    Json::StreamWriterBuilder wb;
    wb["indentation"] = "";
    route_info->value_json = Json::writeString (wb, v);
  } catch (const std::exception & e) {
    route_fail (route_info, "PARSE_ERROR",
        std::string ("malformed request: ") + e.what (),
        "see the API design doc for the stream/route body shape");
    return false;
  }
  return true;
}
