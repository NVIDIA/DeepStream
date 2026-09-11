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

#include <vector>

#include "prometheus/detail/core_export.h"

namespace prometheus {
struct MetricFamily;
}

namespace prometheus {

/// @brief Interface implemented by anything that can be used by Prometheus to
/// collect metrics.
///
/// A Collectable has to be registered for collection. See Registry.
class PROMETHEUS_CPP_CORE_EXPORT Collectable {
 public:
  virtual ~Collectable() = default;

  /// \brief Returns a list of metrics and their samples.
  virtual std::vector<MetricFamily> Collect() const = 0;
};

}  // namespace prometheus
