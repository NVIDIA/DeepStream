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
#include "signal_handler.h"

enum
{
  PROP_0,
};

G_DEFINE_TYPE(GstSignalHandler, gst_signal_handler, GST_TYPE_OBJECT);

static void
gst_signal_handler_set_property(GObject *object, guint prop_id,
                                const GValue *value, GParamSpec *pspec);

static void
gst_signal_handler_get_property(GObject *object, guint prop_id, GValue *value,
                                GParamSpec *pspec);

static void gst_signal_handler_finalize(GObject *object);

/* Define the class initializer for our signal_handler class */
static void gst_signal_handler_class_init(GstSignalHandlerClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  /* Override methods from GObjectClass here */
  gobject_class->set_property = gst_signal_handler_set_property;
  gobject_class->get_property = gst_signal_handler_get_property;
  gobject_class->finalize = gst_signal_handler_finalize;

  /* Register properties for our signal_handler class here */

  /* Set up signals for our signal_handler class here */

  /* Initialize the private data for our signal_handler class here */
}

static void gst_signal_handler_finalize(GObject *object)
{
}

/* Define the instance initializer for our signal_handler class */
static void gst_signal_handler_init(GstSignalHandler *self)
{

  /* Initialize private members for our signal_handler class here */
}

static void
gst_signal_handler_set_property(GObject *object, guint prop_id,
                                const GValue *value, GParamSpec *pspec)
{
  switch (prop_id)
  {
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

static void
gst_signal_handler_get_property(GObject *object, guint prop_id, GValue *value,
                                GParamSpec *pspec)
{
  switch (prop_id)
  {
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}