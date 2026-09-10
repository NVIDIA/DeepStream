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

#include "custom_object.hpp"
#include "common_factory.hpp"
#include <gst/gst.h>

using namespace deepstream;

CustomObject::CustomObject(unsigned long type_id, const char* factory, const std::string& name)
: Object(type_id, name) {
  // default property map
  properties_.insert({"gpu-id", (int)0});
  if (factory == nullptr) {
    // no property spec
    return;
  }
  // initialize the property map
  CustomFactory* p_factory = CommonFactory::getInstance().getCustomFactory(factory);
  if (p_factory) {
    p_factory->getProperty("param-spec", param_spec_);
    if (!param_spec_.empty()) {
      YAML::Node node = YAML::Load(param_spec_);
      for (auto n : node) {
        if (n["type"] && n["name"]) {
          auto type_str = n["type"].as<std::string>();
          if (type_str == "string" || type_str == "path") {
            properties_.insert({n["name"].as<std::string>(), n["default_value"].as<std::string>()});
          } else if (type_str == "integer") {
            properties_.insert({n["name"].as<std::string>(), n["default_value"].as<int>()});
          } else if (type_str == "boolean") {
            properties_.insert({n["name"].as<std::string>(), n["default_value"].as<bool>()});
          } else {
            g_printerr("Unsupported type %s\n", type_str.c_str());
            continue;
          }
        }
      }
    }
  }
}

void CustomObject::set_(const std::string& name, const Value& value) {
  if (!param_spec_.empty()) {
    // perform a check if there is a param spec
    if (properties_.find(name) == properties_.end()) {
      g_printerr("property %s not supported in object %s\n",
                name.c_str(), this->getName().c_str());
      return;
    }
  }
  properties_[name] = value;
}

void CustomObject::set_(const std::string& name, const YAML::Node& value) {
  if (properties_.find(name) == properties_.end()) {
    g_printerr("property %s not found in object %s\n",
              name.c_str(), this->getName().c_str());
    return;
  }

  YAML::Node node = YAML::Load(param_spec_);
  for (auto n : node) {
    if (n["name"].as<std::string>() == name) {
      auto type_str = n["type"].as<std::string>();
      if (type_str == "string" || type_str == "path") {
        properties_[name] = value.as<std::string>();
      } else if (type_str == "integer") {
        properties_[name] = value.as<int>();
      } else if (type_str == "boolean") {
        properties_[name] = value.as<bool>();
      } else {
        g_printerr("Unsupported type %s\n", type_str.c_str());
        continue;
      }
    }
  }
}

Object::Value CustomObject::get_(const std::string& name) {
  if (properties_.find(name) == properties_.end()) {
    g_printerr("property %s not found in object %s\n",
               name.c_str(), this->getName().c_str());
    return Value();
  }
  return properties_[name];
}