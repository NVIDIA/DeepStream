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
#include "buffer_probe.h"

enum
{
  PROP_0,
};

G_DEFINE_TYPE(GstBufferProbe, gst_buffer_probe, GST_TYPE_OBJECT);

static void
gst_buffer_probe_set_property(GObject *object, guint prop_id,
                              const GValue *value, GParamSpec *pspec);

static void
gst_buffer_probe_get_property(GObject *object, guint prop_id, GValue *value,
                              GParamSpec *pspec);

/* Define the class initializer for our buffer_probe class */
static void gst_buffer_probe_class_init(GstBufferProbeClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  /* Override methods from GObjectClass here */
  gobject_class->set_property = gst_buffer_probe_set_property;
  gobject_class->get_property = gst_buffer_probe_get_property;

  /* Register properties for our buffer_probe class here */

  /* Set up signals for our buffer_probe class here */

  /* Initialize the private data for our buffer_probe class here */
}

/* Define the instance initializer for our buffer_probe class */
static void gst_buffer_probe_init(GstBufferProbe *self)
{

  /* Initialize private members for our buffer_probe class here */
}

static void
gst_buffer_probe_set_property(GObject *object, guint prop_id,
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
gst_buffer_probe_get_property(GObject *object, guint prop_id, GValue *value,
                              GParamSpec *pspec)
{
  switch (prop_id)
  {
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}