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

#include "gst/gst.h"
#include "gst/gstchildproxy.h"
#include <iostream>
#include <cstdarg>
#include <algorithm>
#include "object.hpp"
#include "signal_handler.hpp"
#include "signal_emitter.hpp"


using namespace deepstream;

Object::Value::Value():
value_(g_new0(GValue, 1)) {}

Object::Value::Value(char value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_CHAR);
  g_value_set_schar(value_, value);
}

Object::Value::Value(unsigned char value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_UCHAR);
  g_value_set_uchar(value_, value);
}

Object::Value::Value(int value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_INT);
  g_value_set_int(value_, value);
}

Object::Value::Value(unsigned int value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_UINT);
  g_value_set_uint(value_, value);
}

Object::Value::Value(long value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_LONG);
  g_value_set_long(value_, value);
}

Object::Value::Value(unsigned long value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_ULONG);
  g_value_set_ulong(value_, value);
}

Object::Value::Value(float value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_FLOAT);
  g_value_set_float(value_, value);
}

Object::Value::Value(double value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_DOUBLE);
  g_value_set_double(value_, value);
}

Object::Value::Value(const std::string value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_STRING);
  g_value_set_string(value_, value.c_str());
}

Object::Value::Value(const char* value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_STRING);
  g_value_set_static_string(value_, value);
}

Object::Value::Value(bool value):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_TYPE_BOOLEAN);
  g_value_set_boolean(value_, value ? TRUE:FALSE);
}

Object::Value::Value(unsigned long type, int unused):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, type);
}

Object::Value::Value(const Object::Value& other):
value_(g_new0(GValue, 1)) {
  g_value_init(value_, G_VALUE_TYPE(other.value_));
  g_value_copy(other.value_, value_);
}

Object::Value& Object::Value::operator=(const Object::Value& other) {
  if (this != &other) {
      if (!G_VALUE_TYPE(value_)) {
        // the original value is not initialized
        g_value_init(value_, G_VALUE_TYPE(other.value_));
      } else if (G_VALUE_TYPE(value_) != G_VALUE_TYPE(other.value_)) {
          g_printerr("type mismatch during value assignment\n");
          throw std::runtime_error("Value type mismatch");
      }
      g_value_copy(other.value_, value_);
  }
  return *this;
}

Object::Value::~Value() {
  g_value_unset(value_);
  g_free(value_);
}

Object::Value::operator char() const {
  return g_value_get_schar(value_);
}
Object::Value::operator unsigned char() const {
  return g_value_get_uchar(value_);
}
Object::Value::operator int() const {
  return g_value_get_int(value_);
}
Object::Value::operator unsigned int() const {
  return g_value_get_uint(value_);
}
Object::Value::operator long() const {
  return g_value_get_long(value_);
}
Object::Value::operator unsigned long() const {
  return g_value_get_ulong(value_);
}
Object::Value::operator float() const {
  return g_value_get_float(value_);
}
Object::Value::operator double() const {
  return g_value_get_double(value_);
}
Object::Value::operator std::string() const {
  auto str = g_value_get_string(value_);
  return  str ? std::string(str) : std::string();
}
Object::Value::operator const char*() const {
  return g_value_get_string(value_);
}
Object::Value::operator bool() const {
  return (bool) g_value_get_boolean(value_);
}

bool Object::Value::isChar() const{
  return G_VALUE_HOLDS_CHAR(value_);
}
bool Object::Value::isUnsignedChar() const {
  return G_VALUE_HOLDS_UCHAR(value_);
}
bool Object::Value::isInteger() const {
  if (G_VALUE_HOLDS_INT(value_)) return true;
  if (G_VALUE_HOLDS_INT64(value_)) return true;
  if (G_VALUE_HOLDS_LONG(value_)) return true;
  return false;
}
bool Object::Value::isUnsignedInteger() const {
  if (G_VALUE_HOLDS_UINT(value_)) return true;
  if (G_VALUE_HOLDS_UINT64(value_)) return true;
  if (G_VALUE_HOLDS_ULONG(value_)) return true;
  return false;
}
bool Object::Value::isFloat() const {
  return G_VALUE_HOLDS_FLOAT(value_);
}
bool Object::Value::isDouble() const {
  return G_VALUE_HOLDS_DOUBLE(value_);
}
bool Object::Value::isString() const {
  return G_VALUE_HOLDS_STRING(value_);
}
bool Object::Value::isBoolean() const {
  return G_VALUE_HOLDS_BOOLEAN(value_);
}

// an void object
Object::Object() : object_(nullptr) {}

// constructor from a GstObject pointer
Object::Object(unsigned long type_id, const std::string& name)
: object_(nullptr) {
  if (type_id) {
    object_ = GST_OBJECT(g_object_new((GType)type_id, NULL));
    if (!object_) {
      std::throw_with_nested(
        std::runtime_error("Object creation failed"));
    } else if (!name.empty()) {
      gst_object_set_name(object_, name.c_str());
    }
  }
}

// copy constructor
Object::Object(const Object& other) : object_(other.object_) {
  if (object_) {
    gst_object_ref(object_);
  }
}

// move constructor
Object::Object(Object&& other) noexcept : object_(other.object_) {
  other.object_ = nullptr;
}

// copy assignment
Object& Object::operator=(const Object& other) {
  if (object_ != other.object_) {
    if (object_) {
      gst_object_unref(object_);
    }
    object_ = other.object_;
    if (object_) {
      gst_object_ref(object_);
    }
  }
  return *this;
}

// move assignment
Object& Object::operator=(Object&& other) noexcept {
  if (object_ != other.object_) {
    if (object_) {
        gst_object_unref(object_);
    }
    object_ = other.object_;
    other.object_ = nullptr;
  }
  return *this;
}

Object::~Object() {
    if (object_) {
        gst_object_unref(object_);
    }
}

const std::string Object::getName() const {
  char* name = gst_object_get_name(GST_OBJECT(object_));
  std::string ret(name);
  g_free(name);
  return ret;
}

GstObject* Object::give() {
  // give the GObject pointer out without decrementing the reference.
  GstObject* ptr = object_;
  object_= nullptr;
  return ptr;
}

Object& Object::take(GstObject* object) {
  /** the purpose of defining an explicit take method is to make
      sure the callers do know what they're going to do
      on the other hand, Object(void*) is removed to avoid misuse. */
  if (object_ != object) {
    if (object_) {
        gst_object_unref(object_);
    }
    object_ = object;
  }
  return *this;
}

Object& Object::seize(GstObject* object) {
  if (object_ != object) {
    if (object_) {
        gst_object_unref(object_);
    }
    /** seize the GstObject so that the raw object won't be destroyed during
     * the lifetime of the Object wrapper
    */
    object_ = object;
    gst_object_ref(object_);
  }
  return *this;
}

unsigned long Object::type() {
  // no generic type defined in the base class
  return 0;
}

Object& Object::set(const YAML::Node& params) {
  g_return_val_if_fail(params.IsMap(), *this);

  for (const auto& param : params) {
    const std::string name = param.first.as<std::string>();
    // If value is a sequence (array), set the property multiple times, like videotemplate and audiotemplate
    if (param.second.IsSequence()) {
      for (const auto& item : param.second) {
        this->set_(name, item);
      }
    } else {
      this->set_(name, param.second);
    }
  }
  return *this;
}

bool Object::connectSignal(const std::string&signal_name, SignalHandler& handler) {
  guint *signals;
  guint n_signals = 0;
  GSignalQuery *query = NULL;
  auto found = false;
  signals = g_signal_list_ids(G_OBJECT_TYPE(object_), &n_signals);
  for (guint i = 0; i < n_signals; i++) {
      query = g_new0 (GSignalQuery, 1);
      g_signal_query (signals[i], query);
      if ((signal_name == query->signal_name) && !(query->signal_flags & G_SIGNAL_ACTION)) {
        found = true;
        g_free(query);
        break;
      }
      g_free(query);
  }
  g_free (signals);
  signals = NULL;

  if (!found) {
    g_printerr("signal %s is not supported by the object\n", signal_name.c_str());
    return false;
  }

  void* fn = handler.getCallbackFn(signal_name);
  if (!fn) {
    g_printerr("signal %s is not supported by the handler %s\n",
               signal_name.c_str(), handler.getName().c_str());
    return false;
  }
  g_signal_connect(G_OBJECT(object_), signal_name.c_str(), G_CALLBACK(fn), &handler);
  return true;
}

std::vector<std::string> Object::listSignals(bool is_action) {
  std::vector<std::string> ret;
  guint *signals;
  guint n_signals = 0;

  GSignalQuery *query = NULL;
  signals = g_signal_list_ids(G_OBJECT_TYPE(object_), &n_signals);
  for (guint i = 0; i < n_signals; i++) {
      query = g_new0 (GSignalQuery, 1);
      g_signal_query (signals[i], query);
      if (is_action && (query->signal_flags & G_SIGNAL_ACTION)){
        ret.push_back(std::string(query->signal_name));
      }
      if (!is_action && !(query->signal_flags & G_SIGNAL_ACTION)) {
        ret.push_back(std::string(query->signal_name));
      }
      g_free(query);
  }
  g_free (signals);
  signals = NULL;

  return ret;
}

void Object::emitSignal(const std::string& signal_name, va_list args) {
  GQuark signal_id = g_signal_lookup(signal_name.c_str(), G_OBJECT_TYPE(object_));
  if (signal_id == 0) {
    g_printerr("signal %s is not found\n", signal_name.c_str());
    return;
  }
  g_signal_emit_valist(object_, signal_id, 0, args);
}

void Object::set_(const std::string& name, const Value& value) {
  // Handle child proxy notation (e.g. "signaller::uri") for elements that
  // implement GstChildProxy. value.value_ is already a typed GValue so it
  // passes straight through to gst_child_proxy_set_property.
  if (name.find("::") != std::string::npos && GST_IS_CHILD_PROXY(object_)) {
    gst_child_proxy_set_property(GST_CHILD_PROXY(object_), name.c_str(), value.value_);
    return;
  }

  GParamSpec *param_spec = g_object_class_find_property(
    G_OBJECT_GET_CLASS(object_), name.c_str());
  if (!param_spec) {
    g_printerr("Property %s is not supported by object %s\n",
               name.c_str(), getName().c_str());
    return;
  }
  // we support creating a GstCaps object from string representation
  if (param_spec->value_type == GST_TYPE_CAPS) {
    GstCaps* caps = gst_caps_from_string(g_value_get_string(value.value_));
    if (caps) {
      g_object_set(G_OBJECT(object_), name.c_str(), caps, NULL);
    }
  } else if (param_spec->value_type == GST_TYPE_STRUCTURE) {
    GstStructure* structure = gst_structure_from_string(g_value_get_string(value.value_), NULL);
    if (structure) {
      g_object_set(G_OBJECT(object_), name.c_str(), structure, NULL);
      gst_structure_free(structure);
    }
  } else {
    g_object_set_property(G_OBJECT(object_), name.c_str(), value.value_);
  }
}

void Object::set_(const std::string& name, const YAML::Node& value) {
  // Handle child proxy notation (e.g. "signaller::uri") for elements that
  // implement GstChildProxy. Unlike the Value path the YAML node is untyped
  // text, so we use gst_child_proxy_lookup to resolve the child GObject* and
  // its GParamSpec* before dispatching through the type switch below.
  if (name.find("::") != std::string::npos && GST_IS_CHILD_PROXY(object_)) {
    GParamSpec *child_param = nullptr;
    GObject    *child_obj   = nullptr;
    gst_child_proxy_lookup(GST_CHILD_PROXY(object_), name.c_str(),
                           &child_obj, &child_param);
    if (!child_param || !child_obj) {
      g_printerr("Child property not found: %s on object %s\n",
                 name.c_str(), getName().c_str());
      if (child_obj) g_object_unref(child_obj);
      return;
    }

    GType child_type = g_type_fundamental(child_param->value_type);

    switch(child_type) {
      case G_TYPE_INT:
        g_object_set(child_obj, child_param->name, value.as<gint>(), NULL);
        break;
      case G_TYPE_UINT:
        g_object_set(child_obj, child_param->name, value.as<guint>(), NULL);
        break;
      case G_TYPE_INT64:
        g_object_set(child_obj, child_param->name, value.as<gint64>(), NULL);
        break;
      case G_TYPE_UINT64:
        g_object_set(child_obj, child_param->name, value.as<guint64>(), NULL);
        break;
      case G_TYPE_LONG:
        g_object_set(child_obj, child_param->name, value.as<glong>(), NULL);
        break;
      case G_TYPE_ULONG:
        g_object_set(child_obj, child_param->name, value.as<gulong>(), NULL);
        break;
      case G_TYPE_CHAR:
        g_object_set(child_obj, child_param->name, value.as<gchar>(), NULL);
        break;
      case G_TYPE_UCHAR:
        g_object_set(child_obj, child_param->name, value.as<guchar>(), NULL);
        break;
      case G_TYPE_BOOLEAN:
        g_object_set(child_obj, child_param->name, value.as<bool>(), NULL);
        break;
      case G_TYPE_FLOAT:
        g_object_set(child_obj, child_param->name, value.as<float>(), NULL);
        break;
      case G_TYPE_DOUBLE:
        g_object_set(child_obj, child_param->name, value.as<double>(), NULL);
        break;
      case G_TYPE_STRING:
        g_object_set(child_obj, child_param->name, value.as<std::string>().c_str(), NULL);
        break;
      case G_TYPE_ENUM:
        g_object_set(child_obj, child_param->name, value.as<gint>(), NULL);
        break;
      case G_TYPE_FLAGS:
        g_object_set(child_obj, child_param->name, value.as<guint>(), NULL);
        break;
      default:
        g_printerr("Unsupported type for child property %s\n", name.c_str());
        break;
    }

    std::cout << "  set child " << name << ": " << value << std::endl;
    g_object_unref(child_obj);
    return;
  }

  GParamSpec *param =  g_object_class_find_property(
    G_OBJECT_GET_CLASS (object_), name.c_str()
  );
  if (!param) {
    g_printerr("property not supported: %s\n", name.c_str());
    return;
  }

  GType value_type = g_type_fundamental(param->value_type);

  switch(value_type) {
    case G_TYPE_INT:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<gint>(), NULL);
      break;
    case G_TYPE_UINT:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<guint>(), NULL);
      break;
    case G_TYPE_INT64:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<gint64>(), NULL);
      break;
    case G_TYPE_UINT64:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<guint64>(), NULL);
      break;
    case G_TYPE_LONG:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<glong>(), NULL);
      break;
    case G_TYPE_ULONG:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<gulong>(), NULL);
      break;
    case G_TYPE_CHAR:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<gchar>(), NULL);
      break;
    case G_TYPE_UCHAR:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<guchar>(), NULL);
      break;
    case G_TYPE_BOOLEAN:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<bool>(), NULL);
      break;
    case G_TYPE_FLOAT:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<float>(), NULL);
      break;
    case G_TYPE_DOUBLE:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<double>(), NULL);
      break;
    case G_TYPE_STRING:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<std::string>().c_str(), NULL);
      break;
    case G_TYPE_ENUM:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<gint>(), NULL);
      break;
    case G_TYPE_FLAGS:
      g_object_set(G_OBJECT(object_), name.c_str(), value.as<guint>(), NULL);
      break;
    default:
      // we support creating a GstCaps object from string representation
      if (value_type == GST_TYPE_CAPS) {
        GstCaps* caps = gst_caps_from_string(value.as<std::string>().c_str());
        if (caps) {
          g_object_set(G_OBJECT(object_), name.c_str(), caps, NULL);
          break;
        }
      } else if (value_type == GST_TYPE_STRUCTURE) {
        GstStructure* structure = gst_structure_from_string(value.as<std::string>().c_str(), NULL);
        if (structure) {
          g_object_set(G_OBJECT(object_), name.c_str(), structure, NULL);
          gst_structure_free(structure);
          break;
        }
      }
      g_printerr("Unsupported type\n");
      return;
  }

  std::cout << "  set " << name << ": " << value <<  std::endl;
}

Object::Value Object::get_(const std::string& name) {
  // Normalize enum/flags to their primitive equivalents before returning so
  // that operators (operator int, operator unsigned int) stay strict and do
  // not need to know about GLib's enum/flags type hierarchy.
  auto normalize = [](GObject* obj, GParamSpec* param) -> Object::Value {
    if (G_IS_PARAM_SPEC_ENUM(param)) {
      gint v;
      g_object_get(obj, param->name, &v, NULL);
      return Object::Value(v);
    }
    if (G_IS_PARAM_SPEC_FLAGS(param)) {
      guint v;
      g_object_get(obj, param->name, &v, NULL);
      return Object::Value((unsigned int)v);
    }
    Object::Value value(G_PARAM_SPEC_VALUE_TYPE(param), 0);
    g_object_get_property(obj, param->name, value.value_);
    return value;
  };

  // Handle child proxy notation (e.g. "sink_0::operator") for elements that
  // implement GstChildProxy.
  if (name.find("::") != std::string::npos && GST_IS_CHILD_PROXY(object_)) {
    GParamSpec *child_param = nullptr;
    GObject    *child_obj   = nullptr;
    gst_child_proxy_lookup(GST_CHILD_PROXY(object_), name.c_str(),
                           &child_obj, &child_param);
    if (!child_param || !child_obj) {
      g_printerr("Child property not found: %s on object %s\n",
                 name.c_str(), getName().c_str());
      if (child_obj) g_object_unref(child_obj);
      return Object::Value();
    }
    Object::Value value = normalize(child_obj, child_param);
    g_object_unref(child_obj);
    return value;
  }

  GParamSpec *param_spec = g_object_class_find_property(
    G_OBJECT_GET_CLASS(object_), name.c_str());

  if (!param_spec) {
    return Object::Value();
  }

  return normalize(G_OBJECT(object_), param_spec);
}