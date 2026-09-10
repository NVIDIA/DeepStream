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

#include <assert.h>
#include "signal_emitter.h"

enum
{
  PROP_0,
  PROP_CFG_FILE,
};

G_DEFINE_TYPE (GstSignalEmitter, gst_signal_emitter, GST_TYPE_OBJECT);

static void
gst_signal_emitter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);

static void
gst_signal_emitter_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec);

static void gst_signal_emitter_finalize(GObject* object);

/* Define the class initializer for our signal_emitter class */
static void gst_signal_emitter_class_init(GstSignalEmitterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  /* Override methods from GObjectClass here */
  gobject_class->set_property = gst_signal_emitter_set_property;
  gobject_class->get_property = gst_signal_emitter_get_property;
  gobject_class->finalize = gst_signal_emitter_finalize;

  /* Register properties for our signal_emitter class here */

    /* Set up signals for our signal_emitter class here */

    /* Initialize the private data for our signal_emitter class here */
}

static void gst_signal_emitter_finalize(GObject* object) {
  GstSignalEmitter* signal_emitter = GST_SIGNAL_EMITTER(object);
  if (signal_emitter->config_file) {
    g_free(signal_emitter->config_file);
  }
  G_OBJECT_CLASS(gst_signal_emitter_parent_class)->finalize(object);
}

/* Define the instance initializer for our signal_emitter class */
static void gst_signal_emitter_init(GstSignalEmitter *self) {

    /* Initialize private members for our signal_emitter class here */
}

static void
gst_signal_emitter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec) {
  GstSignalEmitter* signal_emitter = GST_SIGNAL_EMITTER(object);
  switch(prop_id) {
    case PROP_CFG_FILE:
      if (signal_emitter->config_file) {
        g_free(signal_emitter->config_file);
        signal_emitter->config_file = g_strdup(g_value_get_string(value));
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_signal_emitter_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec) {
  GstSignalEmitter* signal_emitter = GST_SIGNAL_EMITTER(object);
  switch(prop_id) {
    case PROP_CFG_FILE:
      g_value_set_string(value, signal_emitter->config_file);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}