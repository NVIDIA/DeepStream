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
 * <b>SourceManager definition </b>
 *
 * SourceConfig allows user to add source to dynamicsrcbin on fly
 */
#ifndef NVIDIA_DEEPSTREAM_SOURCE_MANAGER
#define NVIDIA_DEEPSTREAM_SOURCE_MANAGER

#include "signal_emitter.hpp"
#include <mutex>
namespace deepstream {

class DefaultActionOwner : public SignalEmitter::IActionOwner {
public:
  std::vector<std::string> list() override {
    std::vector<std::string> actions = {"add-source", "remove-source", "terminate"};
    return actions;
  }

  void onAttached(SignalEmitter* emitter, const std::string& action, const std::string& object) override {
    object_ = object;
    return;
  }

  std::string getObject() {
    return object_;
  }

protected:
  std::string object_;
};

class SourceManager : public SignalEmitter {
public:
  /**
   * @brief Constructor
   *
   * Create a source manager with a user implemented action owner interface
   *
   * @param[in] name        name of the instance
   */
  SourceManager(const std::string& name)
          : SignalEmitter(name, new DefaultActionOwner()) {}

  /**
   * @brief Add a source to the manager
   *
   * @param[in] source_name name of the source to be added
   * @return int            a unique ID for the source, -1 if failed
   */
  int addSource(const std::string& source_name);

  /**
   * @brief Remove a source from the manager
   *
   * @param[in] source_id   ID of the source to be removed
   */
  void removeSource(int source_id);

  /**
   * @brief Terminate the pipeline
   */
  void terminate();

 protected:
  static int gen_number_;
  static std::mutex gen_number_mutex_;
};;

} // namespace deepstream

#endif