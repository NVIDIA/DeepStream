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

#include <atomic>

#include "prometheus/client_metric.h"
#include "prometheus/detail/builder.h"  // IWYU pragma: export
#include "prometheus/detail/core_export.h"
#include "prometheus/metric_type.h"

namespace prometheus {

/// \brief A gauge metric to represent a value that can arbitrarily go up and
/// down.
///
/// The class represents the metric type gauge:
/// https://prometheus.io/docs/concepts/metric_types/#gauge
///
/// Gauges are typically used for measured values like temperatures or current
/// memory usage, but also "counts" that can go up and down, like the number of
/// running processes.
///
/// The class is thread-safe. No concurrent call to any API of this type causes
/// a data race.
class PROMETHEUS_CPP_CORE_EXPORT Gauge {
 public:
  static const MetricType metric_type{MetricType::Gauge};

  /// \brief Create a gauge that starts at 0.
  Gauge() = default;

  /// \brief Create a gauge that starts at the given amount.
  explicit Gauge(double);

  /// \brief Increment the gauge by 1.
  void Increment();

  /// \brief Increment the gauge by the given amount.
  void Increment(double);

  /// \brief Decrement the gauge by 1.
  void Decrement();

  /// \brief Decrement the gauge by the given amount.
  void Decrement(double);

  /// \brief Set the gauge to the given value.
  void Set(double);

  /// \brief Set the gauge to the current unix time in seconds.
  void SetToCurrentTime();

  /// \brief Get the current value of the gauge.
  double Value() const;

  /// \brief Get the current value of the gauge.
  ///
  /// Collect is called by the Registry when collecting metrics.
  ClientMetric Collect() const;

 private:
  void Change(double);
  std::atomic<double> value_{0.0};
};

/// \brief Return a builder to configure and register a Gauge metric.
///
/// @copydetails Family<>::Family()
///
/// Example usage:
///
/// \code
/// auto registry = std::make_shared<Registry>();
/// auto& gauge_family = prometheus::BuildGauge()
///                          .Name("some_name")
///                          .Help("Additional description.")
///                          .Labels({{"key", "value"}})
///                          .Register(*registry);
///
/// ...
/// \endcode
///
/// \return An object of unspecified type T, i.e., an implementation detail
/// except that it has the following members:
///
/// - Name(const std::string&) to set the metric name,
/// - Help(const std::string&) to set an additional description.
/// - Labels(const Labels&) to assign a set of
///   key-value pairs (= labels) to the metric.
///
/// To finish the configuration of the Gauge metric register it with
/// Register(Registry&).
PROMETHEUS_CPP_CORE_EXPORT detail::Builder<Gauge> BuildGauge();

}  // namespace prometheus
