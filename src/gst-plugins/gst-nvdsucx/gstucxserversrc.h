/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _GST_UCX_SERVER_SRC_H_
#define _GST_UCX_SERVER_SRC_H_

#include <gst/base/gstbasesrc.h>
#include "gstucx.h"

G_BEGIN_DECLS
#define GST_TYPE_UCX_SERVER_SRC   (gst_ucx_server_src_get_type())
#define GST_UCX_SERVER_SRC(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_UCX_SERVER_SRC,GstUcxServerSrc))
#define GST_UCX_SERVER_SRC_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_UCX_SERVER_SRC,GstUcxServerSrcClass))
#define GST_IS_UCX_SERVER_SRC(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_UCX_SERVER_SRC))
#define GST_IS_UCX_SERVER_SRC_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_UCX_SERVER_SRC))
typedef struct _GstUcxServerSrc GstUcxServerSrc;
typedef struct _GstUcxServerSrcClass GstUcxServerSrcClass;

struct _GstUcxServerSrc
{
  GstUcxSrc src;
  GstUcxServer server;
};

struct _GstUcxServerSrcClass
{
  GstPushSrcClass push_ucxserversrc_class;
};

GType gst_ucx_server_src_get_type (void);

G_END_DECLS
#endif
