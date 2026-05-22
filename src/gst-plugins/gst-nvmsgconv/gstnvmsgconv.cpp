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


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "gstnvmsgconv.h"
#include "nvdsmeta_schema.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"

#include <ds3d/gst/nvds3d_meta.h>
#include <ds3d/common/func_utils.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvmsgconv_debug_category);
#define GST_CAT_DEFAULT gst_nvmsgconv_debug_category

#define DEFAULT_PAYLOAD_TYPE NVDS_PAYLOAD_DEEPSTREAM

#define GST_TYPE_NVMSGCONV_PAYLOAD_TYPE (gst_nvmsgconv_payload_get_type ())

#define DEFAULT_EMBEDDING_FILTER "NovaCarter"

#define MAX_TIME_STAMP_LEN 64

#define DEFAULT_FRAME_INTERVAL 30

using namespace ds3d;

static
    GType
gst_nvmsgconv_payload_get_type (void)
{
  static
      GType
      qtype = 0;

  if (qtype == 0) {
    static const
        GEnumValue
        values[] = {
      {NVDS_PAYLOAD_DEEPSTREAM, "Deepstream schema payload",
          "PAYLOAD_DEEPSTREAM"},
      {NVDS_PAYLOAD_DEEPSTREAM_MINIMAL, "Deepstream schema payload minimal",
          "PAYLOAD_DEEPSTREAM_MINIMAL"},
      {NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF,
            "Deepstream schema payload in protobuf",
          "PAYLOAD_DEEPSTREAM_PROTOBUF"},
      {NVDS_PAYLOAD_RESERVED, "Reserved type", "PAYLOAD_RESERVED"},
      {NVDS_PAYLOAD_CUSTOM, "Custom schema payload", "PAYLOAD_CUSTOM"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstNvMsgConvPayloadType", values);
  }
  return qtype;
}

static void
gst_nvmsgconv_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void
gst_nvmsgconv_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void
gst_nvmsgconv_dispose (GObject * object);
static void
gst_nvmsgconv_finalize (GObject * object);
static
    gboolean
gst_nvmsgconv_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static
    gboolean
gst_nvmsgconv_start (GstBaseTransform * trans);
static
    gboolean
gst_nvmsgconv_stop (GstBaseTransform * trans);
static
    GstFlowReturn
gst_nvmsgconv_transform_ip (GstBaseTransform * trans, GstBuffer * buf);

enum
{
  PROP_0,
  PROP_CONFIG_FILE,
  PROP_MSG2P_LIB_NAME,
  PROP_PAYLOAD_TYPE,
  PROP_COMPONENT_ID,
  PROP_DEBUG_PAYLOAD_DIR,
  PROP_MULTIPLE_PAYLOADS,
  PROP_MSG2P_NEW_API,
  PROP_FRAME_INTERVAL,
  PROP_DUMMY_PAYLOAD,
  PROP_EMBEDDING_FILTER
};

static
    GstStaticPadTemplate
    gst_nvmsgconv_src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

static
    GstStaticPadTemplate
    gst_nvmsgconv_sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

G_DEFINE_TYPE_WITH_CODE (GstNvMsgConv, gst_nvmsgconv, GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT (gst_nvmsgconv_debug_category, "nvmsgconv", 0,
        "debug category for nvmsgconv element"));


static
    NvDsMsgConvInputType
gst_nvmsgconv_get_input_type (GstCaps * caps)
{
  GstStructure *
      caps_str = gst_caps_get_structure (caps, 0);
  const gchar *
      mimetype = gst_structure_get_name (caps_str);

  printf ("mimetype is %s\n", mimetype);
  const int
      mtLen = strlen (mimetype);
  if (strncmp (mimetype, "video/x-raw", mtLen) == 0) {
    return kNvDsMsgConv_Video;
  } else if (g_strrstr (mimetype, "ds3d")) {
    return kNvDsMsgConv_DS3D;
  }
  return kNvDsMsgConv_Audio;
}


static void
gst_nvmsgconv_free_meta (gpointer data, gpointer uData)
{
  g_return_if_fail (data);

  NvDsUserMeta *
      user_meta = (NvDsUserMeta *) data;
  NvDsPayload *
      srcPayload = (NvDsPayload *) user_meta->user_meta_data;
  GstNvMsgConv *
      self = (GstNvMsgConv *) user_meta->base_meta.uContext;

  if (self && srcPayload) {
    self->msg2p_release (self->pCtx, srcPayload);
    g_atomic_int_dec_and_test (&self->numActivePayloads);
  }
  //If no active payloads, clean up all resources
  if (!g_atomic_int_get (&self->numActivePayloads) && self->stop) {
    if (self->pCtx) {
      self->ctx_destroy (self->pCtx);
      self->pCtx = NULL;
    }

    if (self->libHandle) {
      dlclose (self->libHandle);
      self->libHandle = NULL;
    }

    if (self->selfRef)
      g_object_unref ((GObject *) self);
  }

}

static
    gpointer
gst_nvmsgconv_copy_meta (gpointer data, gpointer uData)
{
  NvDsUserMeta *
      user_meta = (NvDsUserMeta *) data;
  NvDsPayload *
      srcPayload = (NvDsPayload *) user_meta->user_meta_data;
  GstNvMsgConv *
      self = (GstNvMsgConv *) user_meta->base_meta.uContext;
  NvDsPayload *
      outPayload = NULL;

  if (srcPayload) {
    outPayload = (NvDsPayload *) g_memdup2 (srcPayload, sizeof (NvDsPayload));
    outPayload->payload =
        g_memdup2 (srcPayload->payload, srcPayload->payloadSize);
    outPayload->payloadSize = srcPayload->payloadSize;
    g_atomic_int_inc (&self->numActivePayloads);
  }
  return outPayload;
}

static void
gst_nvmsgconv_class_init (GstNvMsgConvClass * klass)
{
  GObjectClass *
      gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *
      base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_nvmsgconv_src_template);
  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_nvmsgconv_sink_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "Message Converter", "Filter/Metadata",
      "Transforms buffer meta to schema / payload meta",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_nvmsgconv_set_property;
  gobject_class->get_property = gst_nvmsgconv_get_property;
  gobject_class->dispose = gst_nvmsgconv_dispose;
  gobject_class->finalize = gst_nvmsgconv_finalize;
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvmsgconv_set_caps);
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_nvmsgconv_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_nvmsgconv_stop);
  base_transform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvmsgconv_transform_ip);

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string ("config", "configuration file name",
          "Name of configuration file with absolute path.",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MSG2P_LIB_NAME,
      g_param_spec_string ("msg2p-lib", "configuration file name",
          "Name of payload generation library with absolute path.",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PAYLOAD_TYPE,
      g_param_spec_enum ("payload-type", "Payload type",
          "Type of payload to be generated", GST_TYPE_NVMSGCONV_PAYLOAD_TYPE,
          DEFAULT_PAYLOAD_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_COMPONENT_ID,
      g_param_spec_uint ("comp-id", "Component Id ",
          "By default this element operates on all NvDsEventMsgMeta\n"
          "\t\t\tBut it can be restricted to a specific NvDsEventMsgMeta meta\n"
          "\t\t\thaving this component id\n",
          0, G_MAXUINT, 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DEBUG_PAYLOAD_DIR,
      g_param_spec_string ("debug-payload-dir", "directory to dump payload",
          "Absolute path of the directory to dump payloads for debugging.",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MULTIPLE_PAYLOADS,
      g_param_spec_boolean ("multiple-payloads", "Use multiple payloads API",
          "Use API which supports reciveing multiple payloads from converter lib.",
          FALSE, (GParamFlags) G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_MSG2P_NEW_API,
      g_param_spec_boolean ("msg2p-newapi", "Use new msg2p API",
          "Use new API which supports publishing multiple payloads using NvDsFrameMeta",
          FALSE, (GParamFlags) G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_FRAME_INTERVAL,
      g_param_spec_uint ("frame-interval", "Payload generation interval",
          "Frame interval at which payload is generated",
          1, G_MAXUINT, DEFAULT_FRAME_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DUMMY_PAYLOAD,
      g_param_spec_boolean ("dummy-payload", "Dummy payload",
          "By default, payload is generated if NVDS_EVENT_MSG_META is attached to buffer.\n"
          "\t\t\tWith this, dummy payload can be generated if there is no\n"
          "\t\t\tNVDS_EVENT_MSG_META attached to buffer", 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EMBEDDING_FILTER,
      g_param_spec_string ("embedding-filter", "post embeddings for only mentioned datatypes",
          "Comma seperated list of strings to post embeddings",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

static void
gst_nvmsgconv_init (GstNvMsgConv * self)
{
  self->pCtx = NULL;
  self->msg2pLib = NULL;
  self->configFile = NULL;
  self->payloadType = DEFAULT_PAYLOAD_TYPE;
  self->libHandle = NULL;
  self->compId = 0;
  self->debugPayloadDir = NULL;
  self->multiplePayloads = FALSE;
  self->numActivePayloads = 0;
  self->stop = FALSE;
  self->selfRef = FALSE;
  self->dsMetaQuark = g_quark_from_static_string (NVDS_META_STRING);
  self->inputType = kNvDsMsgConv_Video;
  self->msg2pNewApi = FALSE;
  self->frameInterval = DEFAULT_FRAME_INTERVAL;
  self->embedding_filter = NULL;

  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), TRUE);
}

void
gst_nvmsgconv_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (object);

  GST_DEBUG_OBJECT (self, "set_property");

  switch (property_id) {
    case PROP_CONFIG_FILE:
      if (self->configFile)
        g_free (self->configFile);
      self->configFile = (gchar *) g_value_dup_string (value);
      break;
    case PROP_MSG2P_LIB_NAME:
      if (self->msg2pLib)
        g_free (self->msg2pLib);
      self->msg2pLib = (gchar *) g_value_dup_string (value);
      break;
    case PROP_PAYLOAD_TYPE:
      self->payloadType = (NvDsPayloadType) g_value_get_enum (value);
      break;
    case PROP_COMPONENT_ID:
      self->compId = g_value_get_uint (value);
      break;
    case PROP_DEBUG_PAYLOAD_DIR:
      if (self->debugPayloadDir)
        g_free (self->debugPayloadDir);
      self->debugPayloadDir = (gchar *) g_value_dup_string (value);
      break;
    case PROP_MULTIPLE_PAYLOADS:
      self->multiplePayloads = g_value_get_boolean (value);
      break;
    case PROP_MSG2P_NEW_API:
      self->msg2pNewApi = g_value_get_boolean (value);
      break;
    case PROP_FRAME_INTERVAL:
      self->frameInterval = g_value_get_uint (value);
      break;
    case PROP_DUMMY_PAYLOAD:
      self->dummyPayload = g_value_get_boolean (value);
      break;
    case PROP_EMBEDDING_FILTER:
      self->embedding_filter = (gchar *) g_value_dup_string (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvmsgconv_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (object);

  GST_DEBUG_OBJECT (self, "get_property");

  switch (property_id) {
    case PROP_CONFIG_FILE:
      g_value_set_string (value, self->configFile);
      break;
    case PROP_MSG2P_LIB_NAME:
      g_value_set_string (value, self->msg2pLib);
      break;
    case PROP_PAYLOAD_TYPE:
      g_value_set_enum (value, self->payloadType);
      break;
    case PROP_COMPONENT_ID:
      g_value_set_uint (value, self->compId);
      break;
    case PROP_DEBUG_PAYLOAD_DIR:
      g_value_set_string (value, self->debugPayloadDir);
      break;
    case PROP_MULTIPLE_PAYLOADS:
      g_value_set_boolean (value, self->multiplePayloads);
      break;
    case PROP_MSG2P_NEW_API:
      g_value_set_boolean (value, self->msg2pNewApi);
      break;
    case PROP_FRAME_INTERVAL:
      g_value_set_uint (value, self->frameInterval);
      break;
    case PROP_DUMMY_PAYLOAD:
      g_value_set_boolean (value, self->dummyPayload);
      break;
    case PROP_EMBEDDING_FILTER:
      g_value_set_string (value, self->embedding_filter);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvmsgconv_dispose (GObject * object)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (object);

  GST_DEBUG_OBJECT (self, "dispose");

  //If payloads are still active, keep self alive
  if (g_atomic_int_get (&self->numActivePayloads)) {
    g_object_ref (object);
    self->selfRef = TRUE;
  } else
    G_OBJECT_CLASS (gst_nvmsgconv_parent_class)->dispose (object);
}

void
gst_nvmsgconv_finalize (GObject * object)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (object);

  GST_DEBUG_OBJECT (self, "finalize");

  if (self->configFile)
    g_free (self->configFile);

  if (self->debugPayloadDir)
    g_free (self->debugPayloadDir);

  G_OBJECT_CLASS (gst_nvmsgconv_parent_class)->finalize (object);
}

static
    gboolean
gst_nvmsgconv_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (trans);

  GST_DEBUG_OBJECT (self, "set_caps");

  self->inputType = gst_nvmsgconv_get_input_type (incaps);

  return TRUE;
}

static
    gboolean
gst_nvmsgconv_start (GstBaseTransform * trans)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (trans);
  gchar *
      error;

  GST_DEBUG_OBJECT (self, "start");

  self->stop = FALSE;
  self->selfRef = FALSE;

  if(!self->embedding_filter){
					 self->embedding_filter = strdup("Person");
  }
  if (self->debugPayloadDir) {
    if (access (self->debugPayloadDir, F_OK)) {
      GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND, ("Error: dir %s not found",
              self->debugPayloadDir), (NULL));
      return FALSE;
    }
    if (access (self->debugPayloadDir, W_OK)) {
      GST_ELEMENT_ERROR (self, RESOURCE, WRITE,
          ("Error: User doesn't have write permission in %s",
              self->debugPayloadDir), (NULL));
      return FALSE;
    }
  }

  if (self->msg2pLib != NULL) {
    self->libHandle = dlopen (self->msg2pLib, RTLD_LAZY);
    if (!self->libHandle) {
      GST_ELEMENT_ERROR (self, LIBRARY, INIT, (NULL),
          ("unable to open converter library"));
      return FALSE;
    }
    dlerror ();                 /* Clear any existing error */
    self->ctx_create =
        (nvds_msg2p_ctx_create_ptr) dlsym (self->libHandle,
        "nvds_msg2p_ctx_create");
    self->ctx_destroy =
        (nvds_msg2p_ctx_destroy_ptr) dlsym (self->libHandle,
        "nvds_msg2p_ctx_destroy");
    self->msg2p_generate =
        (nvds_msg2p_generate_ptr) dlsym (self->libHandle,
        "nvds_msg2p_generate");
    self->msg2p_generate_multiple =
        (nvds_msg2p_generate_multiple_ptr) dlsym (self->libHandle,
        "nvds_msg2p_generate_multiple");
    self->msg2p_generate_new =
        (nvds_msg2p_generate_ptr_new) dlsym (self->libHandle,
        "nvds_msg2p_generate_new");
    self->msg2p_generate_multiple_new =
        (nvds_msg2p_generate_multiple_ptr_new) dlsym (self->libHandle,
        "nvds_msg2p_generate_multiple_new");
    self->msg2p_release =
        (nvds_msg2p_release_ptr) dlsym (self->libHandle, "nvds_msg2p_release");
    if ((error = dlerror ()) != NULL) {
      GST_ERROR_OBJECT (self, "%s", error);
      return FALSE;
    }
  } else {
    self->ctx_create = (nvds_msg2p_ctx_create_ptr) nvds_msg2p_ctx_create;
    self->ctx_destroy = (nvds_msg2p_ctx_destroy_ptr) nvds_msg2p_ctx_destroy;
    self->msg2p_generate = (nvds_msg2p_generate_ptr) nvds_msg2p_generate;
    self->msg2p_generate_multiple =
        (nvds_msg2p_generate_multiple_ptr) nvds_msg2p_generate_multiple;
    self->msg2p_generate_new =
        (nvds_msg2p_generate_ptr_new) nvds_msg2p_generate_new;
    self->msg2p_generate_multiple_new =
        (nvds_msg2p_generate_multiple_ptr_new) nvds_msg2p_generate_multiple_new;
    self->msg2p_release = (nvds_msg2p_release_ptr) nvds_msg2p_release;
  }

  self->pCtx = self->ctx_create (self->configFile, self->payloadType);

  if (!self->pCtx) {
    if (self->libHandle) {
      dlclose (self->libHandle);
      self->libHandle = NULL;
    }
    GST_ERROR_OBJECT (self, "unable to create instance");
    return FALSE;
  }
  return TRUE;
}

static
    gboolean
gst_nvmsgconv_stop (GstBaseTransform * trans)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (trans);

  GST_DEBUG_OBJECT (self, "stop");

  self->stop = TRUE;

  //If no active payloads, clean up all resources
  if (!g_atomic_int_get (&self->numActivePayloads)) {

    if (self->pCtx) {
      self->ctx_destroy (self->pCtx);
      self->pCtx = NULL;
    }

    if (self->libHandle) {
      dlclose (self->libHandle);
      self->libHandle = NULL;
    }

    if (self->selfRef)
      g_object_unref ((GObject *) self);
  }

  return TRUE;
}

static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm
      tm_log;
  struct timespec
      ts;
  char
      strmsec[6];               //.nnnZ\0

  clock_gettime (CLOCK_REALTIME, &ts);
  memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  int
      ms = ts.tv_nsec / 1000000;
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

static int
add_payload_to_batch_meta (GstNvMsgConv * self, NvDsPayload ** payloads,
    guint payloadCount, NvDsBatchMeta * batch_meta, void *frame_meta,
    FILE * debug_payload_dump_file)
{
  for (uint p = 0; p < payloadCount; ++p) {
    if ((payloads[p] == NULL) || (!payloads[p]->payloadSize)) {
      GST_WARNING_OBJECT (self,
          "Payload received from converter at index %u is invalid", p);
      continue;
    }

    payloads[p]->componentId = self->compId;
    NvDsUserMeta *
        user_payload_meta = nvds_acquire_user_meta_from_pool (batch_meta);
    if (user_payload_meta) {
      user_payload_meta->user_meta_data = (void *) payloads[p];
      user_payload_meta->base_meta.meta_type = NVDS_PAYLOAD_META;
      user_payload_meta->base_meta.copy_func =
          (NvDsMetaCopyFunc) gst_nvmsgconv_copy_meta;
      user_payload_meta->base_meta.release_func =
          (NvDsMetaReleaseFunc) gst_nvmsgconv_free_meta;
      user_payload_meta->base_meta.uContext = (void *) self;
      if (self->inputType == kNvDsMsgConv_Video) {
        nvds_add_user_meta_to_frame ((NvDsFrameMeta *) frame_meta,
            user_payload_meta);
      } else if (self->inputType == kNvDsMsgConv_Audio) {
        nvds_add_user_meta_to_audio_frame ((NvDsAudioFrameMeta *) frame_meta,
            user_payload_meta);
      } else if (self->inputType == kNvDsMsgConv_DS3D) {
        nvds_add_user_meta_to_batch (batch_meta, user_payload_meta);
      } else if (self->inputType == kNvDsMsgConv_Sparse4D) {
        batch_meta->misc_batch_info[0]=NVDS_CUSTOM_MSG_SPARSE4D;
        nvds_add_user_meta_to_batch (batch_meta, user_payload_meta);
      }
      g_atomic_int_inc (&self->numActivePayloads);
    } else {
      GST_ELEMENT_ERROR (self, RESOURCE, FAILED, (NULL),
          ("Couldn't get user meta from pool"));
      if (debug_payload_dump_file)
        fclose (debug_payload_dump_file);
      return 1;
    }

    if (debug_payload_dump_file)
      fprintf (debug_payload_dump_file, "%.*s \n", payloads[p]->payloadSize,
          (char *) payloads[p]->payload);

  }
  return 0;
}

static
    GstFlowReturn
gst_nvmsgconv_transform_ip_ds3d (GstBaseTransform * trans, GstBuffer * buffer)
{

  GstNvMsgConv *
      self = GST_NVMSGCONV (trans);
  NvDsBatchMeta *
      batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
  NvDsPayload **
      payloads = NULL;
  guint payloadCount = 0;
  const abiRefDataMap *
      refDataMap = nullptr;
  if (!isGood (NvDs3D_Find1stDataMap (buffer, refDataMap))) {
    GST_ERROR_OBJECT (self,
        "gst buffer doesn have ds3d datamap, will support later");
    return GST_FLOW_ERROR;
  }

  if (!batch_meta) {
    NvDsMeta *
        meta = NULL;
    //add dummy batch_meta
    batch_meta = nvds_create_batch_meta (1);
    if (batch_meta == NULL) {
      g_print ("line = %d file = %s func = %s\n", __LINE__, __FILE__, __func__);
      exit (-1);
    }
    meta = gst_buffer_add_nvds_meta ((GstBuffer *) (buffer), batch_meta, NULL,
        nvds_batch_meta_copy_func, nvds_batch_meta_release_func);
    if (meta == NULL) {
      g_print ("line = %d file = %s func = %s\n", __LINE__, __FILE__, __func__);
      exit (-1);
    }

    meta->meta_type = NVDS_BATCH_GST_META;

    batch_meta->base_meta.batch_meta = batch_meta;
    batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
    batch_meta->base_meta.release_func = nvds_batch_meta_release_func;
    batch_meta->max_frames_in_batch = 1;
  }

  DS_ASSERT (refDataMap);

  NvDsMsg2pMetaInfo meta_info = {0};
  meta_info.objMeta = NULL;
  meta_info.frameMeta = NULL;
  meta_info.mediaType = (gchar *) "ds3d";
  meta_info.datamap = (void *) refDataMap;

  if ((self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) ||
      (self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF)) {
    if (self->multiplePayloads)
      payloads =
          self->msg2p_generate_multiple_new (self->pCtx, &meta_info,
          &payloadCount);
    else {
      payloads = (NvDsPayload **) g_malloc0 (sizeof (NvDsPayload *) * 1);
      payloads[0] = self->msg2p_generate_new (self->pCtx, &meta_info);
      payloadCount = 1;
    }


    int
        errcode =
        add_payload_to_batch_meta (self, payloads, payloadCount, batch_meta,
        NULL, NULL);

    if (payloadCount && payloads)
      g_free (payloads);

    if (errcode)
      return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

static void
generate_ts_from_frame_ntp (char *buf, int buf_size, GstClockTime ts)
{
  time_t tloc;
  struct tm tm_log;
  char strmsec[6];              //.nnnZ\0
  int ms;

  /** ts itself is UTC Time in ns */
  struct timespec timespec_current;
  GST_TIME_TO_TIMESPEC (ts, timespec_current);
  memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
  ms = timespec_current.tv_nsec / 1000000;

  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

static void nanoseconds_to_rfc3339(int64_t nanoseconds, char *output, size_t output_size) {
    time_t seconds = nanoseconds / 1000000000;
    int32_t milliseconds = (nanoseconds % 1000000000) / 1000000;

    struct tm *tm_info = gmtime(&seconds);

    char time_str[MAX_TIME_STAMP_LEN];
    strftime(time_str, MAX_TIME_STAMP_LEN, "%Y-%m-%dT%H:%M:%S", tm_info);

    g_snprintf(output, output_size, "%s.%03dZ", time_str, milliseconds);
}

static GstFlowReturn
gst_nvmsgconv_transform_ip_sparse4d (GstBaseTransform * trans, GstBuffer * buffer)
{
  GstNvMsgConv *self = GST_NVMSGCONV (trans);
  NvDsPayload **payloads = NULL;
  NvDsMetaList *l = NULL;
  NvDsUserMeta *user_event_meta = NULL;
  self->inputType = kNvDsMsgConv_Sparse4D;

  GST_DEBUG_OBJECT (self, "transform_ip");

  NvDsBatchMeta *
      batch_meta = gst_buffer_get_nvds_batch_meta (buffer);

  if (batch_meta) {
    if ((self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) ||
      (self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF)) {
      guint eventCount = 0;
      NvDsEvent *eventList = NULL;
      for (l = batch_meta->batch_user_meta_list; l; l = l->next) {
        user_event_meta = (NvDsUserMeta *)(l->data);

        if (user_event_meta && user_event_meta->base_meta.meta_type == NVDS_CUSTOM_MSG_SPARSE4D) {
          NvDsBbox3dObjectList *bbox3d_list = (NvDsBbox3dObjectList *)user_event_meta->user_meta_data;
          if (eventList == NULL) {
              eventList = g_new0(NvDsEvent, bbox3d_list->num_objects);
          }

          for (guint i = 0; i < bbox3d_list->num_objects; i++) {
            NvDsEventMsgMeta *eventMsg = g_new0(NvDsEventMsgMeta, 1);
            eventMsg->type = NVDS_EVENT_ENTRY; // Assuming this is an object detection event
            eventMsg->frameId = bbox3d_list->frame_num;
            eventMsg->sensorId = bbox3d_list->camera_id;
            memcpy(eventMsg->entries, bbox3d_list->entries, sizeof(SourceTimePair) * bbox3d_list->count);
            eventMsg->count=bbox3d_list->count;
            eventMsg->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
            nanoseconds_to_rfc3339(bbox3d_list->timestamp, eventMsg->ts, MAX_TIME_STAMP_LEN);
            eventMsg->sensorStr = g_strdup (bbox3d_list->sensorStr);
            NvDsBbox3dObject *obj = &bbox3d_list->objects[i];

            // Copy 3D bounding box data
            NvDsBbox3dObject *extMsg = g_new0(NvDsBbox3dObject, 1);
            memcpy(extMsg, obj, sizeof(NvDsBbox3dObject));
            if(self->embedding_filter) 
              extMsg->embedding_filter =  g_strdup (self->embedding_filter);
            eventMsg->objType = NVDS_OBJECT_TYPE_3D;
            eventMsg->extMsg = extMsg;
            eventMsg->extMsgSize = sizeof(NvDsBbox3dObject);
            if (self->compId && eventMsg->componentId != self->compId) {
              g_free(eventMsg->ts);
              if(extMsg->embedding_filter)
                g_free(extMsg->embedding_filter);
              g_free(extMsg);
              g_free(eventMsg->sensorStr);
              g_free(eventMsg);
              continue;
            }
            eventList[eventCount].eventType = eventMsg->type;
            eventList[eventCount].metadata = eventMsg;
            eventCount++;
          }
          if(bbox3d_list->num_objects==0) {
            eventList = g_renew(NvDsEvent, eventList, 1);
            NvDsEventMsgMeta *eventMsg = g_new0(NvDsEventMsgMeta, 1);
            eventMsg->type = NVDS_EVENT_ENTRY; // Assuming this is an object detection event
            eventMsg->frameId = bbox3d_list->frame_num;
            eventMsg->sensorId = bbox3d_list->camera_id;
            eventMsg->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
            nanoseconds_to_rfc3339(bbox3d_list->timestamp, eventMsg->ts, MAX_TIME_STAMP_LEN);
            eventMsg->sensorStr = g_strdup (bbox3d_list->sensorStr);

            if (self->compId && eventMsg->componentId != self->compId) {
              g_free(eventMsg->ts);
              g_free(eventMsg->sensorStr);
              g_free(eventMsg);
              continue;
            }
            eventList[eventCount].eventType = eventMsg->type;
            eventList[eventCount].metadata = eventMsg;
          }
        }
      }

        guint payloadCount = 0;

        if (self->multiplePayloads)
          payloads = self->msg2p_generate_multiple(self->pCtx, eventList, eventCount, &payloadCount);
        else {
          payloads = (NvDsPayload **)g_malloc0(sizeof(NvDsPayload *) * 1);
          payloads[0] = self->msg2p_generate(self->pCtx, eventList, eventCount);
          payloadCount = 1;
        }

        int errcode = add_payload_to_batch_meta(self, payloads, payloadCount,
                                                    batch_meta, NULL, NULL);
        if (payloadCount && payloads)
          g_free(payloads);

            // Free allocated memory
            for (guint i = 0; i < eventCount; i++) {
                NvDsEventMsgMeta *eventMsg = (NvDsEventMsgMeta *)eventList[i].metadata;
                g_free(eventMsg->ts);
                g_free(eventMsg->sensorStr);
                if(eventMsg->extMsg) {
						 if(((NvDsBbox3dObject *)eventMsg->extMsg)->embedding_filter) {
                     g_free(((NvDsBbox3dObject *)eventMsg->extMsg)->embedding_filter);
						 }
                 g_free(eventMsg->extMsg);
                }
                g_free(eventMsg);
            }

        if (errcode)
          return GST_FLOW_ERROR;
      if (eventList != NULL)
       g_free(eventList);
    }
  }
  return GST_FLOW_OK;
}

static
    GstFlowReturn
gst_nvmsgconv_transform_ip_video_audio (GstBaseTransform * trans,
    GstBuffer * buf)
{
  GstNvMsgConv *
      self = GST_NVMSGCONV (trans);
  NvDsPayload **
      payloads = NULL;
  NvDsEventMsgMeta *
      eventMsg = NULL;
  NvDsMeta *
      meta = NULL;
  NvDsBatchMeta *
      batch_meta = NULL;
  GstMeta *
      gstMeta = NULL;
  gpointer state = NULL;
  FILE *
      debug_payload_dump_file = NULL;

  GST_DEBUG_OBJECT (self, "transform_ip");

  while ((gstMeta = gst_buffer_iterate_meta (buf, &state))) {
    if (gst_meta_api_type_has_tag (gstMeta->info->api, self->dsMetaQuark)) {
      meta = (NvDsMeta *) gstMeta;
      if (meta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *) meta->meta_data;
        break;
      }
    }
  }
  if(batch_meta && batch_meta->batch_user_meta_list) {
    NvDsUserMeta *user_event_meta = NULL;
    NvDsMetaList *l = NULL;

    for (l = batch_meta->batch_user_meta_list; l; l = l->next) {
      user_event_meta = (NvDsUserMeta *)(l->data);

      if (user_event_meta && user_event_meta->base_meta.meta_type == NVDS_CUSTOM_MSG_SPARSE4D) {
        return gst_nvmsgconv_transform_ip_sparse4d (trans, buf);
      }
    }
  }

  if (batch_meta) {
    NvDsMetaList *
        l = NULL;
    NvDsMetaList *
        l_frame = NULL;
    NvDsMetaList *
        user_meta_list = NULL;
    void *
        frame_meta = NULL;
    NvDsUserMeta *
        user_event_meta = NULL;

    // For every frame in the batch
    for (l_frame = batch_meta->frame_meta_list; l_frame;
        l_frame = l_frame->next) {
      frame_meta = l_frame->data;

      if (self->inputType == kNvDsMsgConv_Video) {
        user_meta_list = ((NvDsFrameMeta *) frame_meta)->frame_user_meta_list;
      } else {
        user_meta_list =
            ((NvDsAudioFrameMeta *) frame_meta)->frame_user_meta_list;
      }

      if (self->debugPayloadDir) {
        g_autofree gchar *
            ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
        generate_ts_rfc3339 (ts, MAX_TIME_STAMP_LEN);
        g_autofree gchar *
            filename =
            g_strconcat (self->debugPayloadDir, "/", ts, ".txt", NULL);
        debug_payload_dump_file = fopen (filename, "w");
      }
      if (self->msg2pNewApi) {
        //generate payload at frameInterval(ex: every 30th frame)
        if (self->frameInterval
            && (((NvDsFrameMeta *) frame_meta)->frame_num %
                self->frameInterval))
          continue;
        guint payloadCount = 0;
        NvDsMsg2pMetaInfo meta_info = {0};
        meta_info.objMeta = NULL;
        meta_info.frameMeta = frame_meta;
        switch (self->inputType) {
          case kNvDsMsgConv_Video:
            meta_info.mediaType = (gchar *) "video";
            break;
          case kNvDsMsgConv_Audio:
            meta_info.mediaType = (gchar *) "audio";
            break;
          default:
            break;
        }

        if ((self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) ||
            (self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF)) {
          if (self->multiplePayloads)
            payloads =
                self->msg2p_generate_multiple_new (self->pCtx, &meta_info,
                &payloadCount);
          else {
            payloads = (NvDsPayload **) g_malloc0 (sizeof (NvDsPayload *) * 1);
            payloads[0] = self->msg2p_generate_new (self->pCtx, &meta_info);
            payloadCount = 1;
          }
          int
              errcode = add_payload_to_batch_meta (self, payloads, payloadCount,
              batch_meta, frame_meta, debug_payload_dump_file);

          if (payloadCount && payloads)
            g_free (payloads);

          if (errcode)
            return GST_FLOW_ERROR;
        } else {
          if (self->inputType == kNvDsMsgConv_Video) {
            for (NvDsObjectMetaList * obj_l =
                ((NvDsFrameMeta *) frame_meta)->obj_meta_list; obj_l;
                obj_l = obj_l->next) {
              NvDsObjectMeta *
                  obj_meta = (NvDsObjectMeta *) obj_l->data;
              if (obj_meta == NULL) {
                // Ignore Null object.
                continue;
              }

              meta_info.objMeta = (void *) obj_meta;
              payloadCount = 0;

              if (self->multiplePayloads)
                payloads =
                    self->msg2p_generate_multiple_new (self->pCtx, &meta_info,
                    &payloadCount);
              else {
                payloads =
                    (NvDsPayload **) g_malloc0 (sizeof (NvDsPayload *) * 1);
                payloads[0] = self->msg2p_generate_new (self->pCtx, &meta_info);
                payloadCount = 1;
              }
              int
                  errcode =
                  add_payload_to_batch_meta (self, payloads, payloadCount,
                  batch_meta, frame_meta, debug_payload_dump_file);

              if (payloadCount && payloads)
                g_free (payloads);

              if (errcode)
                return GST_FLOW_ERROR;
            }
          }
        }
      } else {
        if ((self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_MINIMAL) ||
            (self->payloadType == NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF)) {
          NvDsEvent *
              eventList = g_new0 (NvDsEvent, g_list_length (user_meta_list));
          guint eventCount = 0;
          for (l = user_meta_list; l; l = l->next) {
            user_event_meta = (NvDsUserMeta *) (l->data);

            if (user_event_meta
                && user_event_meta->base_meta.meta_type ==
                NVDS_EVENT_MSG_META) {
              eventMsg = (NvDsEventMsgMeta *) user_event_meta->user_meta_data;

              if (self->compId && eventMsg->componentId != self->compId)
                continue;

              eventList[eventCount].eventType = eventMsg->type;
              eventList[eventCount].metadata = eventMsg;
              eventCount++;
            }
          }

          if (eventCount || self->dummyPayload) {
            guint payloadCount = 0;
            NvDsEventMsgMeta *msg_meta = NULL;

            if (!eventCount || self->dummyPayload) {
              msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));

              msg_meta->frameId = ((NvDsFrameMeta *) frame_meta)->frame_num;
              msg_meta->sensorId = ((NvDsFrameMeta *) frame_meta)->source_id;
              msg_meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
              generate_ts_from_frame_ntp (msg_meta->ts, MAX_TIME_STAMP_LEN, ((NvDsFrameMeta *) frame_meta)->ntp_timestamp);
              if (!eventList) {
                eventList = g_new0 (NvDsEvent, 1);
              }

              eventList[0].metadata = msg_meta;
            }

            if (self->multiplePayloads)
              payloads =
                  self->msg2p_generate_multiple (self->pCtx, eventList,
                  eventCount, &payloadCount);
            else {
              payloads =
                  (NvDsPayload **) g_malloc0 (sizeof (NvDsPayload *) * 1);
              payloads[0] =
                  self->msg2p_generate (self->pCtx, eventList, eventCount);
              payloadCount = 1;
            }

            int
                errcode =
                add_payload_to_batch_meta (self, payloads, payloadCount,
                batch_meta, frame_meta, debug_payload_dump_file);

            if (payloadCount && payloads)
              g_free (payloads);

            if (msg_meta) {
              if (msg_meta->ts) {
                g_free(msg_meta->ts);
              }
              g_free(msg_meta);
            }

            if (errcode)
              return GST_FLOW_ERROR;
          }
          g_free (eventList);
        } else {
          for (l = user_meta_list; l; l = l->next) {
            user_event_meta = (NvDsUserMeta *) (l->data);

            if (user_event_meta
                && user_event_meta->base_meta.meta_type ==
                NVDS_EVENT_MSG_META) {
              NvDsEvent event;

              eventMsg = (NvDsEventMsgMeta *) user_event_meta->user_meta_data;

              if (self->compId && eventMsg->componentId != self->compId)
                continue;

              event.eventType = eventMsg->type;
              event.metadata = eventMsg;

              guint payloadCount = 0;

              if (self->multiplePayloads)
                payloads =
                    self->msg2p_generate_multiple (self->pCtx, &event, 1,
                    &payloadCount);
              else {
                payloads =
                    (NvDsPayload **) g_malloc0 (sizeof (NvDsPayload *) * 1);
                payloads[0] = self->msg2p_generate (self->pCtx, &event, 1);
                payloadCount = 1;
              }

              int
                  errcode =
                  add_payload_to_batch_meta (self, payloads, payloadCount,
                  batch_meta, frame_meta, debug_payload_dump_file);

              if (payloadCount && payloads)
                g_free (payloads);

              if (errcode)
                return GST_FLOW_ERROR;
            }
          }
        }
      }
    }
    if (debug_payload_dump_file)
      fclose (debug_payload_dump_file);
  }
  return GST_FLOW_OK;
}

static
    GstFlowReturn
gst_nvmsgconv_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstNvMsgConv *self = GST_NVMSGCONV (trans);
  if (self->inputType == kNvDsMsgConv_DS3D) {
    return gst_nvmsgconv_transform_ip_ds3d (trans, buf);
  }
  return gst_nvmsgconv_transform_ip_video_audio (trans, buf);
}

static
    gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "nvmsgconv", GST_RANK_NONE,
      GST_TYPE_NVMSGCONV);
}

#define PACKAGE "nvmsgconv"

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_msgconv,
    "Metadata conversion",
    plugin_init, "9.0", "Proprietary", "NvMsgConv", "http://nvidia.com")
