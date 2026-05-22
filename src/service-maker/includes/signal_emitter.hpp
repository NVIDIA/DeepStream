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
 * <b>SignalEmitter definition </b>
 *
 * A signal emitter is used to trigger a specific action on an element
 * to which it is attached, and the element must support certain actions
 *
 * More details about the type of action supported by an element can be
 * found in the plugin manual of Deepstream SDK
 */
#ifndef NVIDIA_DEEPSTREAM_SIGNAL_EMITTER
#define NVIDIA_DEEPSTREAM_SIGNAL_EMITTER

#include <memory>
#include "custom_object.hpp"

namespace deepstream {

class Element;

/**
 * @brief SignalEmitter class
 *
 * Users must implement the IActiononwer interface to create a signal emitter
 * Signal emitter can only be attached to elements that supports certain actions.
 *
 * A signal emitter can be attached to multiple elements on multiple actions
 *
 **/
class SignalEmitter : public CustomObject {
public:
  /** @brief required interface for a signal emitter */
  class IActionOwner {
   public:
    virtual ~IActionOwner() {}

    /** @brief  list the name of actions supported by this emitter */
    virtual std::vector<std::string> list() = 0;

    /**
     * @brief Callback to be triggered when the emitter is attached
     *
     * The callback can complete the necessary set-up for the action to work
     *
     * @param[in]  emitter  pointer to the emitter
     * @param[in]  action   name of the action
     * @param[in]  object   name of the object to which the emitter is attached
     */
    virtual void onAttached(SignalEmitter* emitter, const std::string& action, const std::string& object) = 0;
  };

  /**
   * @brief  Constructor
   *
   * Create a signal emitter with a user implemented action owner interface
   *
   * @param[in] name        name of the instance
   * @param[in] handler     implementation of the IActionOwner interface
   */
  SignalEmitter(const std::string& name, IActionOwner* owner);

  /**
   * @brief  Constructor
   *
   * Create a signal emitter with a user implemented action owner interface
   *
   * @param[in] name        name of the instance
   * @param[in] factory     name of the factory who creates the instance
   * @param[in] handler     implementation of the IActionOwner interface
   */
  SignalEmitter(const std::string& name, const char* factory, IActionOwner* owner);

  /** @brief Destructor */
  virtual ~SignalEmitter();

  /** @brief Return the type id assigned to signal emitter */
  static unsigned long type();

  /** @brief Attach the signal emitter to an object on specified action */
  SignalEmitter& attach(const std::string& action_name, Object& object);
  /**
   * @brief Emit the action by name
   *
   * @param[in] action_name name of the action to be triggered
   * @param[in] object_name name of the attached object who is to trigger the action
   *
   * */
  SignalEmitter& emit(const std::string& action_name, const std::string& object_name, ...);

protected:
  std::multimap<std::string, Object> object_map_;
  std::unique_ptr<IActionOwner> owner_;
};

}


#endif