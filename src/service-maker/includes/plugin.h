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
 * <b>Convinient macros for custom plugin creation </b>
 *
 */
#ifndef _DEEPSTREAM_PLUGIN_H_
#define _DEEPSTREAM_PLUGIN_H_

/**
 * DS_CUSTOM_PLUGIN_DEFINE:
 * @name:           name of the plugin
 * @description:    description of the plugin
 * @version:        version string
 * @license:        license type
 *
 * Utility macro to create a custom plugin
 *
 */
#define DS_CUSTOM_PLUGIN_DEFINE(name, description, version, license)                                     \
  extern "C" void *ds_stub_define_custom_plugin(const char *, const char *, const char *, const char *); \
  extern "C" void *gst_plugin_##name##_get_desc(void);                                                   \
  extern "C" void *gst_plugin_##name##_get_desc(void)                                                    \
  {                                                                                                      \
    return ds_stub_define_custom_plugin(#name, description, version, license);                           \
  }

/**
 * @brief Parameter Specification
 *
 * Utility macro to specify a group of parameter spec for a custom factory.
 * Must start with DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN and ends with DS_CUSTOM_FACTORY_DEFINE_PARAMS_END
 *
 * Multiple entries are allowed between
 *
 */
#define DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(param_spec)  static const char* param_spec = "["
#define DS_CUSTOM_FACTORY_DEFINE_PARAM(name, type, brief, description, default_value) \
"{name: "#name", type: "#type", brief: "#brief", description: "#description", default_value: "#default_value"},"
#define DS_CUSTOM_FACTORY_DEFINE_PARAMS_END "]";

/**
 * DS_CUSTOM_FACTORY_DEFINE_FULL:
 * @factory_name:   name of the factory
 * @long_name:      long name of the factory
 * @klass:          category string
 * @description:    detailed information
 * @author:         author
 * @signals:        supported signal, required if the type of the object to be created is signal handler
 * @param_spec:     the parameter spec for the object created to be created, define with macros from above
 * @object_class:   class name of the object to be created
 *
 * Utility macro to create a custom factory
 *
 */
#define DS_CUSTOM_FACTORY_DEFINE_FULL(factory_name, long_name, klass, description, author, signals, param_spec, \
 object_class, ...) \
class object_class##Factory : public deepstream::CustomFactory { \
public: \
  object_class##Factory() : \
    CustomFactory(factory_name, gst_custom_factory_get_type(factory_name)) {} \
  deepstream::CustomObject* createObject(const std::string& name) { \
    return new object_class(name, factory_name, __VA_ARGS__); \
  } \
}; \
extern "C" const FactoryMetadata get_custom_factory_info(void); \
extern "C" const FactoryMetadata get_custom_factory_info(void) { \
  return {factory_name, long_name, klass, description, author, object_class::type(), signals}; \
} \
extern "C" const char* get_custom_factory_product_param_spec(); \
extern "C" const char* get_custom_factory_product_param_spec() { return param_spec; } \
extern "C" void register_component_factory(const char *name);  \
extern "C" void register_component_factory(const char* name) { \
  deepstream::CommonFactory& factory = \
      deepstream::CommonFactory::getInstance();  \
  factory.addCustomFactory(new object_class##Factory(), name); \
}

/** @brief convinient macro for DS_CUSTOM_FACTORY_DEFINE_FULL */
#define DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(factory_name, long_name, klass, description, author, signals, param_spec, \
  object_class, implementation) \
  DS_CUSTOM_FACTORY_DEFINE_FULL(factory_name, long_name, klass, description, author, signals, param_spec, object_class, new implementation)

/** @brief convinient macro for DS_CUSTOM_FACTORY_DEFINE_FULL */
#define DS_CUSTOM_FACTORY_DEFINE_WITH_SIGNALS(factory_name, long_name, klass, description, author, signals, \
  object_class, implementation) \
  static const char* param_spec = ""; \
  DS_CUSTOM_FACTORY_DEFINE_FULL(factory_name, long_name, klass, description, author, signals, param_spec, object_class, new implementation)

/** @brief convinient macro for DS_CUSTOM_FACTORY_DEFINE_FULL */
#define DS_CUSTOM_FACTORY_DEFINE(factory_name, long_name, klass, description, author, \
  object_class, implementation) \
  static const char* param_spec = ""; \
  DS_CUSTOM_FACTORY_DEFINE_FULL(factory_name, long_name, klass, description, author, "", param_spec, object_class, new implementation)

#endif