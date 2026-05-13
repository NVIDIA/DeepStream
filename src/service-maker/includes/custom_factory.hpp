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