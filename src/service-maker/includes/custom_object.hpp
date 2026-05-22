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

/**
 * @file
 * <b>Service custom object class </b>
 *
 * Custom object offers users an approach to define their own object with
 * a specific type id.
 *
 */

#ifndef NVIDIA_DEEPSTREAM_CUSTOM_OBJECT
#define NVIDIA_DEEPSTREAM_CUSTOM_OBJECT

#include <map>

#include "object.hpp"

namespace deepstream {

  /**
   * @brief Base class for all the custom objects
   */
class CustomObject : public Object {
 public:
  /**
   * @brief Constructor
   *
   * @param[in] type_id  a unique 64-bit number to represent the type
   * @param[in] factory  name of the factory who creates the instance
   * @param[in] name     N=name given to the object instance
   */
  CustomObject(unsigned long type_id, const char* factory, const std::string& name);

 protected:
  virtual void set_(const std::string& name, const Value& value);
  virtual void set_(const std::string& name, const YAML::Node& value);
  virtual Value get_(const std::string&name);

  /** property spec in YAML format */
  std::string param_spec_;
  /** property map */
  std::map<std::string, Object::Value> properties_;
};

}

#endif