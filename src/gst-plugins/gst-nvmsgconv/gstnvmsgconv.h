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

#ifndef _GST_NVMSGCONV_H_
#define _GST_NVMSGCONV_H_

#include <gst/base/gstbasetransform.h>
#include "nvmsgconv.h"

G_BEGIN_DECLS
#define GST_TYPE_NVMSGCONV   (gst_nvmsgconv_get_type())
#define GST_NVMSGCONV(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVMSGCONV,GstNvMsgConv))
#define GST_NVMSGCONV_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVMSGCONV,GstNvMsgConvClass))
#define GST_IS_NVMSGCONV(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVMSGCONV))
#define GST_IS_NVMSGCONV_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVMSGCONV))
typedef struct _GstNvMsgConv GstNvMsgConv;
typedef struct _GstNvMsgConvClass GstNvMsgConvClass;

typedef NvDsMsg2pCtx *(*nvds_msg2p_ctx_create_ptr) (const gchar * file,
    NvDsPayloadType type);

typedef void (*nvds_msg2p_ctx_destroy_ptr) (NvDsMsg2pCtx * ctx);

typedef NvDsPayload *(*nvds_msg2p_generate_ptr) (NvDsMsg2pCtx * ctx,
    NvDsEvent * events, guint size);

typedef NvDsPayload **(*nvds_msg2p_generate_multiple_ptr) (NvDsMsg2pCtx * ctx,
    NvDsEvent * events, guint size, guint * payloadCount);

typedef void (*nvds_msg2p_release_ptr) (NvDsMsg2pCtx * ctx,
    NvDsPayload * payload);

typedef NvDsPayload *(*nvds_msg2p_generate_ptr_new) (NvDsMsg2pCtx * ctx,
    void *metadataInfo);

typedef NvDsPayload **(*nvds_msg2p_generate_multiple_ptr_new) (NvDsMsg2pCtx *
    ctx, void *metadataInfo, guint * payloadCount);

typedef enum
{
  kNvDsMsgConv_Video,
  kNvDsMsgConv_Audio,
  kNvDsMsgConv_DS3D,
  kNvDsMsgConv_Sparse4D
} NvDsMsgConvInputType;

struct _GstNvMsgConv
{
  GstBaseTransform parent;

  GQuark dsMetaQuark;
  gchar *configFile;
  gchar *msg2pLib;
  gpointer libHandle;
  gint compId;
  NvDsPayloadType payloadType;
  NvDsMsg2pCtx *pCtx;
  gchar *debugPayloadDir;
  gboolean multiplePayloads;
  gboolean msg2pNewApi;
  guint frameInterval;
  gint numActivePayloads;
  gboolean stop;
  gboolean selfRef;
  gboolean dummyPayload;
  gchar *embedding_filter;

  nvds_msg2p_ctx_create_ptr ctx_create;
  nvds_msg2p_ctx_destroy_ptr ctx_destroy;
  nvds_msg2p_generate_ptr msg2p_generate;
  nvds_msg2p_generate_multiple_ptr msg2p_generate_multiple;
  nvds_msg2p_generate_ptr_new msg2p_generate_new;
  nvds_msg2p_generate_multiple_ptr_new msg2p_generate_multiple_new;
  nvds_msg2p_release_ptr msg2p_release;
  /** Identifies from caps/capability of the incoming data */
  NvDsMsgConvInputType inputType;
};

struct _GstNvMsgConvClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvmsgconv_get_type (void);

G_END_DECLS
#endif
