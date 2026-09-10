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

#include "source_manager.hpp"
#include <glib.h>

namespace deepstream {

int SourceManager::gen_number_ = 0;
std::mutex SourceManager::gen_number_mutex_;

int SourceManager::addSource(const std::string& source_name) {
  DefaultActionOwner* owner = dynamic_cast<DefaultActionOwner*>(owner_.get());
  if (!owner) {
    throw std::runtime_error("Invalid action owner");
  }
  if (owner->getObject().empty()) {
    g_printerr("The source manager is not attached to any object\n");
    return -1;
  }
  int source_id = -1;
  {
    std::lock_guard<std::mutex> lock(gen_number_mutex_);
    source_id = gen_number_ < INT32_MAX ? gen_number_++ : 0;
  }

  std::string file_path = source_name;
  // Implementation for adding a source
  this->emit("add-source", owner->getObject(), source_name.c_str(), source_id, nullptr);

  return source_id;
}

void SourceManager::removeSource(int source_id) {
  DefaultActionOwner* owner = dynamic_cast<DefaultActionOwner*>(owner_.get());
  if (!owner) {
    throw std::runtime_error("Invalid action owner");
  }
  if (owner->getObject().empty()) {
    g_printerr("The source manager is not attached to any object\n");
    return;
  }
  // Implementation for removing a source
  this->emit("remove-source", owner->getObject(), source_id, nullptr);
}

void SourceManager::terminate() {
  DefaultActionOwner* owner = dynamic_cast<DefaultActionOwner*>(owner_.get());
  if (!owner) {
    throw std::runtime_error("Invalid action owner");
  }
  if (owner->getObject().empty()) {
    g_printerr("The source manager is not attached to any object\n");
    return;
  }
  this->emit("terminate", owner->getObject(), nullptr);
}

} // namespace deepstream
