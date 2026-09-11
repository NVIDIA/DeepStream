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

#include <map>
#include "common_factory.hpp"
#include "custom_factory.hpp"

#include <gst/gst.h>

namespace deepstream {

class CommonFactoryImpl: public CommonFactory {
public:
  CommonFactoryImpl() {

  }

  virtual ~CommonFactoryImpl() {
    for (auto i : factory_map_) {
      auto factory = i.second;
      if (factory) {
        delete factory;
      }
    }
    factory_map_.clear();
  }

  virtual std::unique_ptr<CustomObject> createObject(const std::string& plugin_name, const std::string& name) {
    auto itr = factory_map_.find(plugin_name);
    if (itr == factory_map_.end()) {
        GstRegistry *registry = gst_registry_get();
        GstPlugin * plugin = gst_registry_find_plugin(registry, plugin_name.c_str());
        if (!plugin) {
          g_printerr("Plugin %s not found\n", plugin_name.c_str());
          throw std::runtime_error("Signal creation failed");
        }
        if (!gst_plugin_load(plugin)) {
          g_printerr("Failed to load plugin %s\n", plugin_name.c_str());
          throw std::runtime_error("Plugin error");
        }
        itr = factory_map_.find(plugin_name);
        if (itr == factory_map_.end() && !itr->second) {
          return nullptr;
        }
    }

    auto factory = itr->second;
    return std::unique_ptr<CustomObject>(factory->createObject(name));
  }

  virtual bool addCustomFactory(CustomFactory* factory, const char* key) {
    if (factory_map_.find(key) != factory_map_.end()) {
      // already added
      return true;
    }
    factory_map_.insert({key, factory});
    return true;
  }

  virtual CustomFactory* getCustomFactory(const char* name) {
    std::string key = name;
    if (factory_map_.find(std::string(key)) != factory_map_.end()) {
      // already added
      return factory_map_[key];
    }
    return nullptr;
  }
protected:
  std::map<std::string, CustomFactory*> factory_map_;
};

CommonFactory* common_factory_instance = (CommonFactory*)0;

CommonFactory& CommonFactory::getInstance() {
  if (!common_factory_instance) {
    common_factory_instance = new CommonFactoryImpl;
  }
  return *common_factory_instance;
}

bool CommonFactory::_load(const std::string& plugin_name) {
  GstRegistry *registry = gst_registry_get();
  GstPlugin * plugin = gst_registry_find_plugin(registry, plugin_name.c_str());
  if (plugin) {
    return gst_plugin_load(plugin);
  } else {
    return false;
  }
}

}
