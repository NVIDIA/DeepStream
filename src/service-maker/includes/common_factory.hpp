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
 * <b>Service maker common factory definition </b>
 *
 * @b Description: CommonFactory is a singleton provided by service maker
 * for user to create custom objects through a unified interface.
 */

#ifndef NVIDIA_DEEPSTREAM_COMMON_FACTORY
#define NVIDIA_DEEPSTREAM_COMMON_FACTORY

#include <string>
#include <memory>

#include "object.hpp"
#include "custom_factory.hpp"

namespace deepstream {

/**
 * @brief r\Represents a unified interface for managing custom objects and factories
 */
class CommonFactory {
public:
  virtual ~CommonFactory() = default;

  CommonFactory(const CommonFactory&) = delete;
  CommonFactory(CommonFactory&&) = delete;
  CommonFactory& operator=(const CommonFactory&) = delete;
  CommonFactory& operator=(CommonFactory&&) = delete;

  /**
   * @brief  Create a custom object from a custom factory
   *
   * @param[in] factory_key   identifier of the factory where the custom object is supported
   * @param[in] name          name for the object to be created.
   */
  virtual std::unique_ptr<CustomObject> createObject(const std::string& factory_key, const std::string& name) = 0;

  /**
   * @brief  Add a custom factory
   *
   * Once a custom factory is added, the type of object defined within the factory can
   * be created through common factory interface.
   *
   * @param[in] factory   pointer to the factory
   * @param[in] key       key used to identify the factory
   */
  virtual bool addCustomFactory(CustomFactory* factory, const char* key) = 0;

  /**
   * @brief  Find a custom factory that supports a certan object type
   *
   * @param[in] factory_name   name of the factory
   * @return                  Pointer of a custom factory that supports the type
   */
  virtual CustomFactory* getCustomFactory(const char* factory_name) = 0;

  /** @brief Acquire the reference of the singleton */
  static CommonFactory& getInstance();

protected:
  CommonFactory() = default;

  static bool _load(const std::string& plugin_name);
};

}

#endif