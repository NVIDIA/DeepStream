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

/*
 * OpenTelemetry Client for DeepStream Metrics
 *
 * This client provides OpenTelemetry integration for DeepStream metrics.
 * Uses PeriodicExportingMetricReader to automatically export metrics at configured intervals.
 *
 * Usage:
 *   auto client = GET_OPENTELEMETRY(refresh_period, otlp_url);
 *   client->setNvMultiUriSrcbinCreator(custom_ctx);
 *
 *   refresh_period: seconds between exports (default: 5)
 *                   -1 to disable OpenTelemetry
 *   otlp_url: OTLP endpoint in "ip:port" format (e.g., "192.168.1.100:4318")
 *             or full URL (e.g., "http://localhost:4318/v1/metrics")
 *             (default: "http://localhost:4318/v1/metrics")
 *   custom_ctx: pointer to NvMultiUriSrcbinCreator context (required for sensor info)
 *
 *   Metrics exported to: configured OTLP endpoint
 *
 * Examples:
 *   GET_OPENTELEMETRY(5);  // Uses default localhost:4318
 *   GET_OPENTELEMETRY(5, "192.168.1.100:4318");  // Custom IP and port
 *   GET_OPENTELEMETRY(5, "http://custom.endpoint/v1/metrics");  // Full URL
 *
 * Exported Metrics:
 *   - stream_fps: Frames per second per source
 *   - stream_latency: Stream latency per source (ms)
 *   - stream_frame_number: Frame number per source
 *   - system_metrics: CPU/GPU/RAM usage
 *
 *   Note: Inactive streams are sent with -1 values for one cycle to allow
 *         OTEL processors to drop them properly.
 *
 * Control via HTTP Header (X-Refresh-Period):
 *   curl -H "X-Refresh-Period: 3" http://localhost:9010/api/v1/metrics
 *   curl -H "X-Refresh-Period: -1" http://localhost:9010/api/v1/metrics  # Disable
 */
#pragma once

#include <memory>
#include <string>
#include <mutex>
#include <unordered_map>

#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter.h"
#include "opentelemetry/exporters/ostream/metric_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/meter_provider_factory.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "nvds_stats.h"

namespace metric_sdk = opentelemetry::sdk::metrics;
namespace metric_api = opentelemetry::metrics;
namespace otlp_exporter = opentelemetry::exporter::otlp;

#define GET_OPENTELEMETRY OpenTelemetryClient::getInstance

struct SourceMetrics {
    double fps;
    double latency;
    int64_t frame_number;
    std::string source_id;
    std::string sensor_name;
    std::string sensor_id;
    bool active;
};

class OpenTelemetryClient
{
public:
    OpenTelemetryClient(int refresh_period_ms = 5000, const std::string& otlp_url = "http://localhost:4318", bool from_rest_api = false);
    ~OpenTelemetryClient();

    static OpenTelemetryClient* getInstance(int refresh_period_ms = 5000, const std::string& otlp_url = "http://localhost:4318", bool from_rest_api = false);

    void setupGauges();
    void shutdown();
    void reinitialize(int refresh_period_ms, const std::string& otlp_url, bool from_rest_api = false);

    bool isEnabled() const { return m_is_enabled; }
    std::string getOtlpUrl() const { return m_otlp_url; }

    void updateSourceMetrics();
    void updateSystemMetrics();

    std::unordered_map<std::string, SourceMetrics> getSourceMetrics();

    void setNvMultiUriSrcbinCreator(void* nvmultiurisrcbin_creator);

private:
    static OpenTelemetryClient* _instance;
    static std::once_flag _init_flag;

    std::unordered_map<std::string, SourceMetrics> m_source_metrics;
    std::unordered_map<std::string, SourceMetrics> m_previous_source_metrics;
    std::mutex m_source_metrics_mutex;

    int m_refresh_period_ms;
    std::string m_otlp_url;
    bool m_is_enabled;
    bool m_metrics_updated_in_cycle;
    int m_callback_counter;
    std::mutex m_update_flag_mutex;
    bool m_initialized_from_rest_api;
    void initializeComponents();

    opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument> m_fps_gauge;
    opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument> m_latency_gauge;
    opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument> m_frame_number_gauge;
    opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument> m_system_metrics_gauge;
    opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument> m_stream_count_gauge;
    opentelemetry::nostd::shared_ptr<metric_api::MeterProvider> m_provider;
    opentelemetry::nostd::shared_ptr<metric_api::Meter> m_meter;
    std::shared_ptr<metric_sdk::MeterProvider> m_provider_sdk;
    void* m_nvmultiurisrcbin_creator;
    double m_ram_memory;
    double m_cpu_usage;
    double m_gpu_memory;
    double m_gpu_usage;
    guint m_stream_count;
};

