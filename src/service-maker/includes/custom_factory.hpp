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
 * <b>Service maker custome factory </b>
 *
 * A custome factory is used to create custom objects
 *
 */

#ifndef NVIDIA_DEEPSTREAM_CUSTOM_FACTORY
#define NVIDIA_DEEPSTREAM_CUSTOM_FACTORY

#include "custom_object.hpp"
#include "factory_metadata.h"

namespace deepstream {

/**
 * @brief Interface definition for a custom factory
 */
class CustomFactory : public Object {
public:
  /**
   * @brief  Constructor
   *
   * @param[in] name           name of the factory instance
   * @param[in] factory_type   unique type id of this factory object
   */
  CustomFactory(const std::string& name, unsigned long factory_type);

  /** @brief Destructor */
  virtual ~CustomFactory();

  /**
   * @brief  Virtual method for creating custom object
   *
   * @param[in] name  name of the object instance
   */
  virtual CustomObject *createObject(const std::string& name) = 0;

  /** @brief Return the type id of objects created by this factory */
  unsigned long getObjectType();
};

}


#endif