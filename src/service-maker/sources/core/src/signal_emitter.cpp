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

#include "gst/signal_emitter.h"
#include "signal_emitter.hpp"

#include <gst/gst.h>
#include <cstdarg>
#include <algorithm>


namespace deepstream {

SignalEmitter::SignalEmitter(const std::string& name, IActionOwner* owner)
: CustomObject(SignalEmitter::type(), nullptr, name), owner_(owner) {
}

SignalEmitter::SignalEmitter(const std::string& name, const char* factory, IActionOwner* owner)
: CustomObject(SignalEmitter::type(), factory, name), owner_(owner) {
}

SignalEmitter::~SignalEmitter() {

}

unsigned long SignalEmitter::type() {
    return GST_TYPE_SIGNAL_EMITTER;
}

SignalEmitter& SignalEmitter::attach(const std::string& action_name, Object& object) {
  auto all_actions = object.listSignals(true);
  if (std::find(all_actions.begin(), all_actions.end(), action_name) == all_actions.end()) {
    g_printerr("action %s not supported by object %s, unable to attach\n",
                action_name.c_str(), object.getName().c_str());
    return *this;
  }
  all_actions = owner_->list();
  if (std::find(all_actions.begin(), all_actions.end(), action_name) == all_actions.end()) {
    g_printerr("action %s not supported by the emitter, unable to attach\n",
                action_name.c_str());
    return *this;
  }
  object_map_.insert({action_name, object});
  owner_->onAttached(this, action_name, object.getName());
  return *this;
}

SignalEmitter& SignalEmitter::emit(
  const std::string& action_name, const std::string& object_name, ...
) {
  auto range = object_map_.equal_range(action_name);
  for (auto it = range.first; it != range.second; it++) {
    if (it->second.getName() == object_name){
      g_print("SignalEmitter: emitting signal %s to object %s\n",
               action_name.c_str(), object_name.c_str());
      va_list args;
      va_start(args, object_name);
      it->second.emitSignal(action_name, args);
      va_end(args);
    }
  }

  return *this;
}

}