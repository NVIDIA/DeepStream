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
 * <b>SignalHandler definition </b>
 *
 * A signal emitter is used to respond on a specific signal from an element
 * to which it is attached, and the element must support certain signals.
 *
 * More details about the type of signals supported by an element can be
 * found in the plugin manual of Deepstream SDK
 */

#ifndef NVIDIA_DEEPSTREAM_SIGNAL_HANDLER
#define NVIDIA_DEEPSTREAM_SIGNAL_HANDLER

#include "custom_object.hpp"

namespace deepstream {

/**
 * @brief SignalHandler class
 *
 * Users must implement the IActionProvider interface to create a signal handler.
 * Signal handlers can only be attached to elements that supports certain signals.
 *
 * A signal handler can handle multiple signals from a single element
 *
 **/
class SignalHandler : public CustomObject {
public:

  /** @brief generic callback on the signal */
  typedef struct {
    /** name of the signal */
    std::string name;
    /** callback function pointer */
    void* fn;
  } Callback;

  /** @brief required interface for a signal handler */
  class IActionProvider {
   public:
    virtual ~IActionProvider() {}

    /** @brief  return the callback */
    virtual const Callback* getCallbacks() = 0;
  };

  /**
   * @brief  Constructor
   *
   * Create a signal handler with a user implemented action provider interface
   *
   * @param[in] name        name of the instance
   * @param[in] handler     implementation of the IActionProvider interface
   */
  SignalHandler(const std::string& name, IActionProvider* provider);

  /**
   * @brief  Constructor
   *
   * Create a signal handler with a user implemented action provider interface
   *
   * @param[in] name        name of the instance
   * @param[in] factory     name of the factory who creates the instance
   * @param[in] handler     implementation of the IActionProvider interface
   */
  SignalHandler(const std::string& name, const char* factory, IActionProvider* provider);

  /** @brief Destructor */
  virtual ~SignalHandler();

  /** @brief Return the type id assigned to signal handler */
  static unsigned long type();

  /** @brief Return the callback for a specific signal */
  void* getCallbackFn(const std::string& name) const;

protected:
  std::unique_ptr<IActionProvider> provider_;
};

}


#endif