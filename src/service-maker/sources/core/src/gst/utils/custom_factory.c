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
#include "custom_factory.h"

enum
{
  PROP_0,
  PROP_OBJECT_TYPE,
  PROP_LONG_NAME,
  PROP_KLASS,
  PROP_DESCRIPTION,
  PROP_AUTHOR,
  PROP_SIGNALS,
  PROP_PARAM_SPEC
};

static void
gst_custom_factory_set_property(GObject *object, guint prop_id,
                                const GValue *value, GParamSpec *pspec);

static void
gst_custom_factory_get_property(GObject *object, guint prop_id, GValue *value,
                                GParamSpec *pspec);

/* Define the class initializer for our custom_factory class */
static void gst_custom_factory_class_init(GstCustomFactoryClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  /* Override methods from GObjectClass here */
  gobject_class->set_property = gst_custom_factory_set_property;
  gobject_class->get_property = gst_custom_factory_get_property;

  /* Register properties for our custom_factory class here */
  g_object_class_install_property(
      gobject_class,
      PROP_OBJECT_TYPE,
      g_param_spec_gtype("object-type", "Object Type", "GType of the objects created by the factory",
                         G_TYPE_NONE, G_PARAM_READABLE));
  g_object_class_install_property(
      gobject_class,
      PROP_LONG_NAME,
      g_param_spec_string("long-name", "Long Name", "Long name of the factory",
                          NULL, (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_KLASS,
      g_param_spec_string("klass", "Klass", "object class created by the factory",
                          NULL, (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_DESCRIPTION,
      g_param_spec_string("description", "Description", "description of the factory",
                          NULL, (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_AUTHOR,
      g_param_spec_string("author", "Author", "author of the factory",
                          NULL, (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_SIGNALS,
      g_param_spec_string("signals", "Signals", "supported signals by the products of the factory",
                          NULL, (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_PARAM_SPEC,
      g_param_spec_string("param-spec", "Param Spec", "Parameter specification for configuration of the factory products",
                          NULL, (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  /* Set up signals for our custom_factory class here */

  /* Initialize the private data for our custom_factory class here */
}

/* Define the instance initializer for our custom_factory class */
static void gst_custom_factory_init(GstCustomFactory *self)
{

  /* Initialize private members for our custom_factory class here */
  self->factory_info = get_custom_factory_info();
  /* Get the parameter specification for the factory products */
  self->param_spec = get_custom_factory_product_param_spec();
}

static const GTypeInfo custom_factory_info = {
    sizeof(GstCustomFactoryClass),
    NULL, /* base_init */
    NULL, /* base_finalize */
    (GClassInitFunc)gst_custom_factory_class_init,
    NULL, /* class_finalize */
    NULL, /* class_data */
    sizeof(GstCustomFactory),
    0, /* n_preallocs */
    (GInstanceInitFunc)gst_custom_factory_init};

GType gst_custom_factory_get_type(const char *name)
{
  static GType type = 0;
  if (type == 0)
  {
    assert(name != NULL);
    type = g_type_register_static(
        GST_TYPE_OBJECT,
        name,
        &custom_factory_info,
        (GTypeFlags)(0));
  }
  return type;
}

static void
gst_custom_factory_set_property(GObject *object, guint prop_id,
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
gst_custom_factory_get_property(GObject *object, guint prop_id, GValue *value,
                                GParamSpec *pspec)
{
  GstCustomFactory *custom_factory = GST_CUSTOM_FACTORY(object);
  switch (prop_id)
  {
  case PROP_OBJECT_TYPE:
    g_value_set_gtype(value, custom_factory->factory_info.object_type);
    break;
  case PROP_LONG_NAME:
    g_value_set_string(value, custom_factory->factory_info.long_name);
    break;
  case PROP_KLASS:
    g_value_set_string(value, custom_factory->factory_info.klass);
    break;
  case PROP_DESCRIPTION:
    g_value_set_string(value, custom_factory->factory_info.description);
    break;
  case PROP_AUTHOR:
    g_value_set_string(value, custom_factory->factory_info.author);
    break;
  case PROP_SIGNALS:
    g_value_set_string(value, custom_factory->factory_info.signals);
    break;
  case PROP_PARAM_SPEC:
    g_value_set_string(value, custom_factory->param_spec);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}