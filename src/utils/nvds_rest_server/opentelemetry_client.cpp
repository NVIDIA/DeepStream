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

#include "opentelemetry_client.h"
#include "nvds_rest_metrics.h"
#include "gst-nvmultiurisrcbincreator.h"
#include <iostream>
#include <cstdlib>

OpenTelemetryClient* OpenTelemetryClient::_instance = NULL;
std::once_flag OpenTelemetryClient::_init_flag;

OpenTelemetryClient::OpenTelemetryClient(int refresh_period_ms, const std::string& otlp_url, bool from_rest_api)
    : m_refresh_period_ms(refresh_period_ms), m_otlp_url(otlp_url), m_is_enabled(false),
      m_metrics_updated_in_cycle(false), m_callback_counter(0), m_initialized_from_rest_api(from_rest_api)
{
    if (refresh_period_ms == -1) {
        m_is_enabled = false;
        return;
    }

    if (refresh_period_ms <= 0) {
        std::cerr << "[OpenTelemetry] Invalid refresh period: " << refresh_period_ms
                  << ", using default 5000 milliseconds" << std::endl;
        m_refresh_period_ms = 5000;
    }

    initializeComponents();
}

OpenTelemetryClient::~OpenTelemetryClient()
{
    if (m_is_enabled) {
        shutdown();
    }
}

OpenTelemetryClient* OpenTelemetryClient::getInstance(int refresh_period_ms, const std::string& otlp_url, bool from_rest_api)
{
    static bool first_call = true;

    std::call_once(_init_flag, [refresh_period_ms, otlp_url, from_rest_api]() {
        std::string url = otlp_url.empty() ? "http://localhost:4318" : otlp_url;
        _instance = new OpenTelemetryClient(refresh_period_ms, url, from_rest_api);
    });
    static std::mutex reinit_mutex;
    std::lock_guard<std::mutex> lock(reinit_mutex);

    if (_instance && !first_call && refresh_period_ms > 0) {
        std::string url = otlp_url.empty() ? _instance->getOtlpUrl() : otlp_url;

        if (_instance->isEnabled()) {
            _instance->shutdown();
        }
        _instance->reinitialize(refresh_period_ms, url, from_rest_api);
    }

    first_call = false;

    return _instance;
}

void OpenTelemetryClient::shutdown()
{
    if (!m_is_enabled) {
        return;
    }

    if (m_provider_sdk) {
        m_provider_sdk->Shutdown();
    }

    m_fps_gauge = opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument>();
    m_latency_gauge = opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument>();
    m_frame_number_gauge = opentelemetry::nostd::shared_ptr<metric_api::ObservableInstrument>();

    m_is_enabled = false;
}

void OpenTelemetryClient::reinitialize(int refresh_period_ms, const std::string& otlp_url, bool from_rest_api)
{
    if (m_is_enabled) {
        std::cerr << "[OpenTelemetry] Already running, shutdown first" << std::endl;
        return;
    }

    if (refresh_period_ms <= 0) {
        std::cerr << "[OpenTelemetry] Invalid refresh period: "
                  << refresh_period_ms << std::endl;
        return;
    }

    m_refresh_period_ms = refresh_period_ms;
    m_otlp_url = otlp_url;
    m_initialized_from_rest_api = from_rest_api;
    initializeComponents();
}

void OpenTelemetryClient::initializeComponents()
{
    const char* exporter_type = std::getenv("OTEL_METRICS_EXPORTER");
    bool use_console = false;
    bool use_otlp = false;
    bool no_exporter = false;

    // Skip environment variable check if initialized from REST API
    if (!m_initialized_from_rest_api) {
        if (exporter_type != nullptr) {
            std::string exporter_str(exporter_type);
            if (exporter_str == "console") {
                use_console = true;
            }
            else if (exporter_str == "otlp") {
                use_otlp = true;
            }
            else if (exporter_str == "none") {
                no_exporter = true;
            }
        }
        else {
            std::cerr << "[OpenTelemetry] No exporter type specified, using default otlp exporter" << std::endl;
            use_otlp = true;
        }

        if(no_exporter) {
            return;
        }
    } else {
        // REST API call - always use OTLP exporter
        use_otlp = true;
    }
    if(!use_console && !use_otlp) {
        std::cerr << "[OpenTelemetry] Invalid exporter type: " << exporter_type << ", using default otlp exporter" << std::endl;
        use_otlp = true;
    }
    std::unique_ptr<metric_sdk::PushMetricExporter> exporter;

    if (use_console) {
        exporter = std::make_unique<opentelemetry::exporter::metrics::OStreamMetricExporter>();
    } else if (use_otlp) {
        otlp_exporter::OtlpHttpMetricExporterOptions opts;
        if (m_initialized_from_rest_api) {
            std::string full_url = m_otlp_url;
            if (full_url.find("/v1/metrics") == std::string::npos) {
                if (!full_url.empty() && full_url.back() == '/') {
                    full_url.pop_back();
                }
                full_url += "/v1/metrics";
            }
            opts.url = full_url;
        }
        exporter = std::make_unique<otlp_exporter::OtlpHttpMetricExporter>(opts);
        std::cout << "[OpenTelemetry] OTLP URL: " << opts.url << std::endl;
        m_otlp_url = opts.url;
    }
    metric_sdk::PeriodicExportingMetricReaderOptions reader_options;
    reader_options.export_interval_millis = std::chrono::milliseconds(m_refresh_period_ms);
    reader_options.export_timeout_millis = std::chrono::milliseconds(m_refresh_period_ms / 2);

    std::cout << "[OpenTelemetry] Export interval: " << m_refresh_period_ms << " milliseconds" << std::endl;
    auto reader = std::make_unique<metric_sdk::PeriodicExportingMetricReader>(
        std::move(exporter), reader_options);

    auto context = std::make_unique<opentelemetry::sdk::metrics::MeterContext>();
    context->AddMetricReader(std::move(reader));
    m_provider_sdk = std::make_shared<metric_sdk::MeterProvider>(std::move(context));
    m_provider = opentelemetry::nostd::shared_ptr<metric_api::MeterProvider>(m_provider_sdk);

    metric_api::Provider::SetMeterProvider(m_provider);
    const char* service_name = std::getenv("OTEL_SERVICE_NAME");
    std::string scope_name = service_name != nullptr ? service_name : "app-meter";
    m_meter = m_provider->GetMeter(scope_name, "1.0.0");
    m_is_enabled = true;

    setupGauges();
}

void OpenTelemetryClient::setupGauges()
{
    if (!m_is_enabled) {
        return;
    }

    if (!m_meter) {
        std::cerr << "[OpenTelemetry] Meter not initialized" << std::endl;
        return;
    }

    m_fps_gauge = m_meter->CreateDoubleObservableGauge("stream_fps",
                                                        "Frames per second per source",
                                                        "fps");
    m_fps_gauge->AddCallback(
        [](metric_api::ObserverResult observer_result, void *user_data) {
            OpenTelemetryClient* client = static_cast<OpenTelemetryClient*>(user_data);
            if (!client) return;
            client->updateSourceMetrics();
            auto double_observer = opentelemetry::nostd::get<
                opentelemetry::nostd::shared_ptr<metric_api::ObserverResultT<double>>>(observer_result);
            if (double_observer) {
                std::lock_guard<std::mutex> lock(client->m_source_metrics_mutex);
                for (const auto& entry : client->m_source_metrics) {
                    double fps_value = entry.second.active ? entry.second.fps : -1.0;
                    double_observer->Observe(fps_value,
                        {std::make_pair("source_id", entry.second.source_id),
                        std::make_pair("sensor_name", entry.second.sensor_name),
                        std::make_pair("sensor_id", entry.second.sensor_id)});
                }
            }
        }, this);

    m_latency_gauge = m_meter->CreateDoubleObservableGauge("stream_latency",
                                                            "Stream latency per source",
                                                            "ms");
    m_latency_gauge->AddCallback(
        [](metric_api::ObserverResult observer_result, void *user_data) {
            OpenTelemetryClient* client = static_cast<OpenTelemetryClient*>(user_data);
            if (!client) return;
            client->updateSourceMetrics();
            auto double_observer = opentelemetry::nostd::get<
                opentelemetry::nostd::shared_ptr<metric_api::ObserverResultT<double>>>(observer_result);
            if (double_observer) {
                std::lock_guard<std::mutex> lock(client->m_source_metrics_mutex);
                for (const auto& entry : client->m_source_metrics) {
                    double latency_value = entry.second.active ? entry.second.latency : -1.0;
                    double_observer->Observe(latency_value,
                        {std::make_pair("source_id", entry.second.source_id),
                        std::make_pair("sensor_name", entry.second.sensor_name),
                        std::make_pair("sensor_id", entry.second.sensor_id)});
                }
            }
        }, this);

    m_frame_number_gauge = m_meter->CreateInt64ObservableGauge("stream_frame_number",
                                                              "Stream frame number per source",
                                                              "");
    m_frame_number_gauge->AddCallback(
        [](metric_api::ObserverResult observer_result, void *user_data) {
            OpenTelemetryClient* client = static_cast<OpenTelemetryClient*>(user_data);
            if (!client) return;
            client->updateSourceMetrics();
            auto int_observer = opentelemetry::nostd::get<
                opentelemetry::nostd::shared_ptr<metric_api::ObserverResultT<int64_t>>>(observer_result);
            if (int_observer) {
                std::lock_guard<std::mutex> lock(client->m_source_metrics_mutex);
                for (const auto& entry : client->m_source_metrics) {
                    int64_t frame_number_value = entry.second.active ? entry.second.frame_number : -1;
                    int_observer->Observe(frame_number_value,
                        {std::make_pair("source_id", entry.second.source_id),
                         std::make_pair("sensor_name", entry.second.sensor_name),
                         std::make_pair("sensor_id", entry.second.sensor_id)});
                }
            }
        }, this);

    m_system_metrics_gauge = m_meter->CreateDoubleObservableGauge("system_metrics",
                                                            "System metrics",
                                                            "metrics");
    m_system_metrics_gauge->AddCallback(
        [](metric_api::ObserverResult observer_result, void *user_data) {
            OpenTelemetryClient* client = static_cast<OpenTelemetryClient*>(user_data);
            if (!client) return;
            client->updateSystemMetrics();
            auto double_observer = opentelemetry::nostd::get<
                opentelemetry::nostd::shared_ptr<metric_api::ObserverResultT<double>>>(observer_result);
            if (double_observer) {
                std::lock_guard<std::mutex> lock(client->m_source_metrics_mutex);
                double_observer->Observe(client->m_ram_memory, {std::make_pair("metric_type", "ram_memory_gb")});
                double_observer->Observe(client->m_cpu_usage, {std::make_pair("metric_type", "cpu_utilization")});
                double_observer->Observe(client->m_gpu_memory, {std::make_pair("metric_type", "gpu_memory_gb")});
                double_observer->Observe(client->m_gpu_usage, {std::make_pair("metric_type", "gpu_utilization")});
            }
        }, this);

    m_stream_count_gauge = m_meter->CreateInt64ObservableGauge("stream_count",
                                                            "Stream count",
                                                            "count");
    m_stream_count_gauge->AddCallback(
        [](metric_api::ObserverResult observer_result, void *user_data) {
            OpenTelemetryClient* client = static_cast<OpenTelemetryClient*>(user_data);
            if (!client) return;
            client->updateSourceMetrics();
            auto int_observer = opentelemetry::nostd::get<
                opentelemetry::nostd::shared_ptr<metric_api::ObserverResultT<int64_t>>>(observer_result);
            if (int_observer) {
                std::lock_guard<std::mutex> lock(client->m_source_metrics_mutex);
                int_observer->Observe(client->m_stream_count, {std::make_pair("metric_type", "stream_count")});
            }
        }, this);
}

void OpenTelemetryClient::updateSourceMetrics()
{
    {
        std::lock_guard<std::mutex> lock(m_update_flag_mutex);

        if (m_metrics_updated_in_cycle) {
            m_callback_counter++;

            if (m_callback_counter > 3) {
                m_metrics_updated_in_cycle = false;
                m_callback_counter = 0;
            }
            return;
        }

        m_metrics_updated_in_cycle = true;
        m_callback_counter = 1;
    }

    guint latency_count = 0;
    guint fps_count = 0;
    m_stream_count = 0;

    NvDsMetricsFrameLatency* latency_data = nvds_get_shared_frame_latency_data(&latency_count);
    NvDsMetricsFpsData* fps_data = nvds_get_shared_fps_data(&fps_count);

    // Get sensor info from creator if available
    GList *stream_info_list = NULL;
    std::map<guint, NvDsSensorInfo*> source_id_to_sensor_info;

    if (m_nvmultiurisrcbin_creator != nullptr) {
        NvDst_Handle_NvMultiUriSrcCreator nvmultiurisrcbin_creator = (NvDst_Handle_NvMultiUriSrcCreator)m_nvmultiurisrcbin_creator;
        if (gst_nvmultiurisrcbincreator_get_source_info_list(
                nvmultiurisrcbin_creator, &stream_info_list)) {
            for (GList *temp = stream_info_list; temp; temp = g_list_next(temp)) {
                NvDsSensorInfo *sensor_info = (NvDsSensorInfo *)temp->data;
                if (sensor_info)
                    source_id_to_sensor_info[sensor_info->source_id] = sensor_info;
            }
        }
    }
    else
    {
        std::cerr << "[OpenTelemetry] NvMultiUriSrcbin creator is not set" << std::endl;
    }
    std::map<guint, double> fps_map;
    if (fps_data) {
        for (guint i = 0; i < fps_count; i++) {
            fps_map[fps_data[i].source_id] = fps_data[i].fps_val;
        }
    }


    std::unordered_map<std::string, SourceMetrics> current_active_metrics;

    if (latency_data) {
        for (guint i = 0; i < latency_count; i++) {
            guint source_id_num = latency_data[i].source_id;
            std::string source_id = "source_" + std::to_string(source_id_num);
            double latency = latency_data[i].latency;

            double fps = 0.0;
            auto fps_it = fps_map.find(source_id_num);
            if (fps_it != fps_map.end()) {
                fps = fps_it->second;
            }

            int64_t frame_number = static_cast<int64_t>(latency_data[i].frame_num);
            if(source_id_to_sensor_info.find(source_id_num) == source_id_to_sensor_info.end()) {
                continue;
            }
            std::string sensor_name = source_id_to_sensor_info[source_id_num]->sensor_name;
            std::string sensor_id = source_id_to_sensor_info[source_id_num]->sensor_id;

            std::string composite_key = source_id + "|" + sensor_name + "|" + sensor_id;

            current_active_metrics[composite_key] = {fps, latency, frame_number, source_id, sensor_name, sensor_id, true};
            m_stream_count++;
        }
    }


    std::unordered_map<std::string, SourceMetrics> merged_metrics = current_active_metrics;

    for (const auto& prev_entry : m_previous_source_metrics) {
        const std::string& composite_key = prev_entry.first;
        if (current_active_metrics.find(composite_key) == current_active_metrics.end()) {
            // Stream was active before but not now - mark as inactive
            SourceMetrics inactive_metric = prev_entry.second;
            inactive_metric.active = false;
            merged_metrics[composite_key] = inactive_metric;
        }
    }


    std::lock_guard<std::mutex> lock(m_source_metrics_mutex);
    m_source_metrics = std::move(merged_metrics);
    m_previous_source_metrics = std::move(current_active_metrics);
}


void OpenTelemetryClient::updateSystemMetrics()
{
    guint gpu_id = get_gpu_id(m_nvmultiurisrcbin_creator);
    m_ram_memory = get_ram_usage();
    m_cpu_usage = get_cpu_utilization();
    m_gpu_memory = get_gpu_memory(gpu_id);
    m_gpu_usage = get_gpu_utilization(gpu_id);
}

void OpenTelemetryClient::setNvMultiUriSrcbinCreator(void* nvmultiurisrcbin_creator)
{
    // Only update if it's different (avoid redundant writes and potential races)
    if (m_nvmultiurisrcbin_creator != nvmultiurisrcbin_creator) {
        m_nvmultiurisrcbin_creator = nvmultiurisrcbin_creator;
    }
}