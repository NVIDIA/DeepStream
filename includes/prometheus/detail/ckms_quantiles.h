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

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include "prometheus/detail/core_export.h"

// IWYU pragma: private, include "prometheus/summary.h"

namespace prometheus {
namespace detail {

class PROMETHEUS_CPP_CORE_EXPORT CKMSQuantiles {
 public:
  struct PROMETHEUS_CPP_CORE_EXPORT Quantile {
    Quantile(double quantile, double error);

    double quantile;
    double error;
    double u;
    double v;
  };

 private:
  struct Item {
    double value;
    int g;
    int delta;

    Item(double value, int lower_delta, int delta);
  };

 public:
  explicit CKMSQuantiles(const std::vector<Quantile>& quantiles);

  void insert(double value);
  double get(double q);
  void reset();

 private:
  double allowableError(int rank);
  bool insertBatch();
  void compress();

 private:
  const std::reference_wrapper<const std::vector<Quantile>> quantiles_;

  std::size_t count_;
  std::vector<Item> sample_;
  std::array<double, 500> buffer_;
  std::size_t buffer_count_;
};

}  // namespace detail
}  // namespace prometheus
