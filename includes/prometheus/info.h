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

#include "prometheus/client_metric.h"
#include "prometheus/detail/builder.h"  // IWYU pragma: export
#include "prometheus/detail/core_export.h"
#include "prometheus/metric_type.h"

namespace prometheus {

/// \brief A info metric to represent textual information which should not
/// change during process lifetime.
///
/// This class represents the metric type info:
/// https://github.com/OpenObservability/OpenMetrics/blob/98ae26c87b1c3bcf937909a880b32c8be643cc9b/specification/OpenMetrics.md#info

/// Prometheus does not provide this type directly, it is used by emulating a
/// gauge with value 1: https://prometheus.io/docs/concepts/metric_types/#gauge
///
/// The value of the info cannot change. Example of infos are:
/// - the application's version
/// - revision control commit
/// - version of the compiler
///
/// The class is thread-safe. No concurrent call to any API of this type causes
/// a data race.
class PROMETHEUS_CPP_CORE_EXPORT Info {
 public:
  static const MetricType metric_type{MetricType::Info};

  /// \brief Create a info.
  Info() = default;

  /// \brief Get the current value of the info.
  ///
  /// Collect is called by the Registry when collecting metrics.
  ClientMetric Collect() const;
};

/// \brief Return a builder to configure and register a Info metric.
///
/// @copydetails Family<>::Family()
///
/// Example usage:
///
/// \code
/// auto registry = std::make_shared<Registry>();
/// auto& info_family = prometheus::BuildInfo()
///                            .Name("some_name")
///                            .Help("Additional description.")
///                            .Labels({{"key", "value"}})
///                            .Register(*registry);
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
/// To finish the configuration of the Info metric, register it with
/// Register(Registry&).
PROMETHEUS_CPP_CORE_EXPORT detail::Builder<Info> BuildInfo();

}  // namespace prometheus
