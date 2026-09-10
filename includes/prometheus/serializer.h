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

#include <iosfwd>
#include <string>
#include <vector>

#include "prometheus/detail/core_export.h"
#include "prometheus/metric_family.h"

namespace prometheus {

class PROMETHEUS_CPP_CORE_EXPORT Serializer {
 public:
  virtual ~Serializer() = default;
  virtual std::string Serialize(const std::vector<MetricFamily>&) const;
  virtual void Serialize(std::ostream& out,
                         const std::vector<MetricFamily>& metrics) const = 0;
};

}  // namespace prometheus
