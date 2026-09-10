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

#ifndef _GST_UCX_SERVER_SINK_H_
#define _GST_UCX_SERVER_SINK_H_

#include "gstucx.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

G_BEGIN_DECLS
#define GST_TYPE_UCX_SERVER_SINK   (gst_ucx_server_sink_get_type())
#define GST_UCX_SERVER_SINK(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_UCX_SERVER_SINK,GstUcxServerSink))
#define GST_UCX_SERVER_SINK_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_UCX_SERVER_SINK,GstUcxServerSinkClass))
#define GST_IS_UCX_SERVER_SINK(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_UCX_SERVER_SINK))
#define GST_IS_UCX_SERVER_SINK_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_UCX_SERVER_SINK))
typedef struct _GstUcxServerSink GstUcxServerSink;
typedef struct _GstUcxServerSinkClass GstUcxServerSinkClass;

struct _GstUcxServerSink
{
  GstUcxSink sink;
  GstUcxServer server;
};

struct _GstUcxServerSinkClass
{
  GstBaseSinkClass base_ucxserversink_class;
};

GType gst_ucx_server_sink_get_type (void);

G_END_DECLS
#endif
