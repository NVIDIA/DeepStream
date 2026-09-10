/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gst/gst.h>
#include "custom_factory.h"
#include "plugin.h"

#define PACKAGE "deepstream-plugins"

// create and registry the factory, implemented by the plugin
extern void register_component_factory(const char* name);
extern const FactoryMetadata get_custom_factory_info(void);

static gboolean plugin_init (GstPlugin * plugin);

void *ds_stub_define_custom_plugin(
    const char *name,
    const char *description,
    const char *version,
    const char *license);

static GstPluginDesc plugin_desc = {
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  "ds_stub",
  "deepstream plugin stub",
  plugin_init,
  "0.1",
  "Proprietary",
  "deepstream-plugins",
  "deepstream",
  "http://nvidia.com",
  __GST_PACKAGE_RELEASE_DATETIME,
  GST_PADDING_INIT
};

static gboolean
plugin_init(GstPlugin * plugin) {
  FactoryMetadata factory_info = get_custom_factory_info();
  GType type = gst_custom_factory_get_type(factory_info.name);
  if (!gst_dynamic_type_register(plugin, type)) {
    g_print("Failed to register GST_TYPE_CUSTOM_FACTORY");
    return FALSE;
  }
  /* given there is only one factory supported by a plugin, use 
     plugin name to registry factory, so to simplify the application*/
  register_component_factory(plugin_desc.name);
  g_print("Plugin %s initialized\n", plugin_desc.name);
  return TRUE;
}

void *ds_stub_define_custom_plugin(
  const char* name,
  const char* description,
  const char* version,
  const char* license
) {
  // plugin information
  plugin_desc.name = name;
  plugin_desc.description = description;
  plugin_desc.version = version;
  plugin_desc.license = license;

  return (void*) &plugin_desc;
}