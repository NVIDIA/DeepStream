/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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