/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>Service maker object base class </b>
 *
 * Most of the entities used in service maker derive from Object class,
 * including Element, BufferProbe, ...
 *
 * Object class provides two fundamental features:
 * 1. reference management for auto garbage collection
 * 2. property setting/getting for various types.
 *
 */

#ifndef NVIDIA_DEEPSTREAM_OBJECT
#define NVIDIA_DEEPSTREAM_OBJECT

#include <string>
#include "yaml-cpp/yaml.h"

// opaque structures
typedef struct _GValue GValue;
typedef struct _GstObject GstObject;

namespace deepstream {

class SignalHandler;
class SignalEmitter;

/**
 * @brief Base Object class
 */
class Object {
public:

  /**
   * @brief Value wrapper for various types
   *
   * A value instance can be initialized with a specific type, meanwhile
   * it can be casted back to its raw type.
  */
  class Value {
   public:
    Value();
    Value(char value);
    Value(unsigned char value);
    Value(int value);
    Value(unsigned int value);
    Value(long value);
    Value(unsigned long value);
    Value(float value);
    Value(double value);
    Value(const std::string value);
    Value(const char* value);
    Value(bool value);

    Value(const Value& other);

    virtual ~Value();

    Value &operator=(const Value& other);

    operator char() const;
    operator unsigned char() const;
    operator int() const;
    operator unsigned int() const;
    operator long() const;
    operator unsigned long() const;
    operator float() const;
    operator double() const;
    operator std::string() const;
    operator const char*() const;
    operator bool() const;

    bool isChar() const;
    bool isUnsignedChar() const;
    bool isInteger() const;
    bool isUnsignedInteger() const;
    bool isFloat() const;
    bool isString() const;
    bool isBoolean() const;

   private:
    GValue* value_;

    Value(unsigned long, int);

   friend class Object;
  };

  /** @brief Constructor of a void object */
  Object();
  /**
   * @brief Create an object from a type id
   *
   * @param[in] name name to be assigned to the object instance
   *
   */
  Object(unsigned long type_id, const std::string& name);

  /** @brief Copy constructor */
  Object(const Object&);

  /** @brief Move constructor */
  Object(Object&&);

  /** @brief Copy assignment */
  Object& operator=(const Object&);

  /** @brief Move assignment */
  Object& operator=(Object&&);

  /** @brief Destructor */
  virtual ~Object();

  /** @brief Return the name assigned during the construction */
  const std::string getName() const;

  /** @brief Check if the object is void */
  explicit operator bool () const noexcept { return object_ != nullptr; }

  /** @brief  Check if the two objects are the same */
  bool operator == (const Object& other) noexcept { return object_ == other.object_; }

  /** @brief  Give up the ownership and return the opaque pointer */
  GstObject* give();

  /** @brief  Return the opaque object pointer */
  GstObject* getGObject() {
    return object_;
  }

  /** @brief Takes the ownership of a object through the opaque pointer */
  Object& take(GstObject* object);

  /** @brief Seize a opaque object to prevent it from being destroyed somewhere*/
  Object& seize(GstObject* object);

  /** @brief Set the properties from key/value pairs in the yaml format */
  Object& set(const YAML::Node& params);

  // set the properties through ke/value pair directly
  Object& set(const std::string& name, const Value& value) {
    this->set_(name, value);
    return *this;
  }

  // template for setting multiple properties
  template<typename T, typename... Args>
  Object& set(const std::string& name, const T& value, const Args&... args) {
    set_(name, Value(value));
    if constexpr (sizeof...(args) > 0) {
      this->set(args...);
    }
    return *this;
  }

  /** @brief Template for getting multiple properties */
  template<typename T, typename... Args>
  Object& getProperty(const std::string& name, T& value, Args&... args) {
    value = (T) get_(name);
    if constexpr (sizeof...(args) > 0) {
      this->getProperty(args...);
    }
    return *this;
  }

  /** @brief List all the supported signals from the object */
  std::vector<std::string> listSignals(bool is_action);

  /**
   * @brief Connect a signal handler to the object
   *
   * @param[in] signal_name name of the signal to be connected
   * @param[in] handler     handler for the signal
   *
   */
  bool connectSignal(const std::string&signal_name, SignalHandler& handler);

  /** @brief Emit a signal */
  void emitSignal(const std::string&signal_name, va_list args);

  /** @brief Return the object's type id */
  static unsigned long type();

protected:
  GstObject* object_;

  virtual void set_(const std::string& name, const Value& value);
  virtual void set_(const std::string& name, const YAML::Node& value);
  virtual Value get_(const std::string&name);

friend class Pipeline;
friend class Element;
};

}

#endif