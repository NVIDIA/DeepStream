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

#pragma once

#include <string>

#include "prometheus/labels.h"

// IWYU pragma: private
// IWYU pragma: no_include "prometheus/family.h"

namespace prometheus {

template <typename T>
class Family;    // IWYU pragma: keep
class Registry;  // IWYU pragma: keep

namespace detail {

template <typename T>
class Builder {
 public:
  Builder& Labels(const ::prometheus::Labels& labels);
  Builder& Name(const std::string&);
  Builder& Help(const std::string&);
  Family<T>& Register(Registry&);

 private:
  ::prometheus::Labels labels_;
  std::string name_;
  std::string help_;
};

}  // namespace detail
}  // namespace prometheus
