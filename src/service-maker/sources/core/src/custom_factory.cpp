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

#include "custom_factory.hpp"
#include "gst/utils/custom_factory.h"

#include <gst/gst.h>

namespace deepstream
{

  CustomFactory::CustomFactory(
      const std::string &name,
      unsigned long factory_type)
      : Object(factory_type, name)
  {
  }

  CustomFactory::~CustomFactory()
  {
  }

  unsigned long CustomFactory::getObjectType()
  {
    GValue value = G_VALUE_INIT;
    unsigned long object_type = 0;
    g_object_get_property(G_OBJECT(object_), "object-type", &value);
    object_type = g_value_get_gtype(&value);
    g_value_unset(&value);
    return object_type;
  }

}