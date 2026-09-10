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

#ifndef _GST_CUSTOME_FACTORY_H_
#define _GST_CUSTOME_FACTORY_H_

#include <gst/gst.h>
#include "factory_metadata.h"

G_BEGIN_DECLS

#define GST_TYPE_CUSTOM_FACTORY (gst_custom_factory_get_type(NULL))
#define GST_CUSTOM_FACTORY(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_CUSTOM_FACTORY, GstCustomFactory))
#define GST_CUSTOM_FACTORY_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_CUSTOM_FACTORY, GstCustomFactoryClass))
#define GST_IS_CUSTOM_FACTORY(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_CUSTOM_FACTORY))
#define GST_IS_CUSTOM_FACTORY_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_CUSTOM_FACTORY))

typedef struct _GstCustomFactory GstCustomFactory;
typedef struct _GstCustomFactoryClass GstCustomFactoryClass;

struct _GstCustomFactory
{
  GstObject base_custom_factory;

  FactoryMetadata factory_info;
  const gchar *param_spec;
};

struct _GstCustomFactoryClass
{
  GstObjectClass base_custom_factory_class;
};

G_END_DECLS

#endif