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

#include <chrono>
#include <cstddef>
#include <vector>

#include "prometheus/detail/ckms_quantiles.h"  // IWYU pragma: export
#include "prometheus/detail/core_export.h"

// IWYU pragma: private, include "prometheus/summary.h"

namespace prometheus {
namespace detail {

class PROMETHEUS_CPP_CORE_EXPORT TimeWindowQuantiles {
  using Clock = std::chrono::steady_clock;

 public:
  TimeWindowQuantiles(const std::vector<CKMSQuantiles::Quantile>& quantiles,
                      Clock::duration max_age_seconds, int age_buckets);

  double get(double q) const;
  void insert(double value);

 private:
  CKMSQuantiles& rotate() const;

  const std::vector<CKMSQuantiles::Quantile>& quantiles_;
  mutable std::vector<CKMSQuantiles> ckms_quantiles_;
  mutable std::size_t current_bucket_;

  mutable Clock::time_point last_rotation_;
  const Clock::duration rotation_interval_;
};

}  // namespace detail
}  // namespace prometheus
