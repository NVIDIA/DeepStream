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

#ifndef _GST_SIGNAL_EMITTER_H_
#define _GST_SIGNAL_EMITTER_H_

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_SIGNAL_EMITTER (gst_signal_emitter_get_type())
#define GST_SIGNAL_EMITTER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_SIGNAL_EMITTER, GstSignalEmitter))
#define GST_SIGNAL_EMITTER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_SIGNAL_EMITTER, GstSignalEmitterClass))
#define GST_IS_SIGNAL_EMITTER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_SIGNAL_EMITTER))
#define GST_IS_SIGNAL_EMITTER_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_SIGNAL_EMITTER))

typedef struct _GstSignalEmitter GstSignalEmitter;
typedef struct _GstSignalEmitterClass GstSignalEmitterClass;

struct _GstSignalEmitter
{
  GstObject base_signal_emitter;
  gchar *config_file;
};

struct _GstSignalEmitterClass
{
  GstObjectClass base_signal_emitter_class;
};

GType gst_signal_emitter_get_type(void);

G_END_DECLS

#endif