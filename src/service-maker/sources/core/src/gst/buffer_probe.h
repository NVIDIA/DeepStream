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

#ifndef _GST_BUFFER_PROBE_H_
#define _GST_BUFFER_PROBE_H_

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_BUFFER_PROBE (gst_buffer_probe_get_type())
#define GST_BUFFER_PROBE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_BUFFER_PROBE, GstBufferProbe))
#define GST_BUFFER_PROBE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_BUFFER_PROBE, GstBufferProbeClass))
#define GST_IS_BUFFER_PROBE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_BUFFER_PROBE))
#define GST_IS_BUFFER_PROBE_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_BUFFER_PROBE))

typedef struct _GstBufferProbe GstBufferProbe;
typedef struct _GstBufferProbeClass GstBufferProbeClass;

struct _GstBufferProbe
{
  GstObject base_buffer_probe;
};

struct _GstBufferProbeClass
{
  GstObjectClass base_buffer_probe_class;
};

GType gst_buffer_probe_get_type(void);

G_END_DECLS

#endif