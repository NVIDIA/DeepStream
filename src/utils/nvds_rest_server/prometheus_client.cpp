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

#include "prometheus_client.h"
#include "prometheus/gauge.h"
#include "prometheus/text_serializer.h"

#include <memory>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>

using namespace prometheus;

PrometheusClient* PrometheusClient::_instance = NULL;
std::once_flag PrometheusClient::_init_flag;

PrometheusClient::PrometheusClient()
{
    try
    {
        m_ready_registry = std::make_shared<Registry>();
        m_live_registry = std::make_shared<Registry>();
        m_metrics_registry = std::make_shared<Registry>();
        m_stream_info_registry = std::make_shared<Registry>();
        m_startup_registry = std::make_shared<Registry>();
        BuildGaugeFamily(); 
    }
    catch (const std::exception& e)
    {
        std::cout << "Failed to initialize Prometheus client: " << e.what() << std::endl;
        std::cout << "Prometheus functionality will be disabled" << std::endl;
        m_ready_registry.reset();
        m_live_registry.reset();
        m_metrics_registry.reset();
        m_stream_info_registry.reset();
        m_startup_registry.reset();
        m_fps_gauge_family = nullptr;
        m_latency_gauge_family = nullptr;
        m_frame_number_gauge_family = nullptr;
        m_memory_gauge_family = nullptr;
        m_utilization_gauge_family = nullptr;
        m_stream_count_gauge = nullptr;
        m_stream_info_count_gauge = nullptr;
        m_ready_gauge_family = nullptr;
        m_live_gauge_family = nullptr;
        m_startup_gauge_family = nullptr;
        return;
    }
    catch (...)
    {
        std::cout << "Unknown exception occurred while initializing Prometheus client" << std::endl;
        std::cout << "Prometheus functionality will be disabled" << std::endl;
        m_ready_registry.reset();
        m_live_registry.reset();
        m_metrics_registry.reset();
        m_stream_info_registry.reset();
        m_startup_registry.reset();
        m_fps_gauge_family = nullptr;
        m_latency_gauge_family = nullptr;
        m_frame_number_gauge_family = nullptr;
        m_memory_gauge_family = nullptr;
        m_utilization_gauge_family = nullptr;
        m_stream_count_gauge = nullptr;
        m_stream_info_count_gauge = nullptr;
        m_ready_gauge_family = nullptr;
        m_live_gauge_family = nullptr;
        m_startup_gauge_family = nullptr;
        return;
    }
}

PrometheusClient* PrometheusClient::getInstance()
{
    std::call_once(_init_flag, []() {
        _instance = new PrometheusClient();
    });
    return _instance;
}

void PrometheusClient::BuildGaugeFamily()
{
    // Register stream and system metrics in metrics registry
    auto& fps_gauge_family = BuildGauge()
                        .Name("fps_metrics")
                        .Help("FPS metrics from ds")
                        .Labels({{"app_name", "ds"}})
                        .Register(*m_metrics_registry);
    m_fps_gauge_family = &fps_gauge_family;

    auto& latency_gauge_family = BuildGauge()
                        .Name("latency_metrics")
                        .Help("Latency metrics from ds")
                        .Labels({{"app_name", "ds"}})
                        .Register(*m_metrics_registry);
    m_latency_gauge_family = &latency_gauge_family;

    auto& frame_number_gauge_family = BuildGauge()
                        .Name("frame_number_metrics")
                        .Help("Frame number metrics from ds")
                        .Labels({{"app_name", "ds"}})
                        .Register(*m_metrics_registry);
    m_frame_number_gauge_family = &frame_number_gauge_family;

    auto& memory_gauge_family = BuildGauge()
                            .Name("memory_metrics")
                            .Help("Memory metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_metrics_registry);
    m_memory_gauge_family = &memory_gauge_family;

    auto& utilization_gauge_family = BuildGauge()
                            .Name("utilization_metrics")
                            .Help("Utilization metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_metrics_registry);
    m_utilization_gauge_family = &utilization_gauge_family;

    auto &stream_count_gauge = BuildGauge()
                            .Name("stream_count")
                            .Help("Stream count from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_metrics_registry);
    m_stream_count_gauge = &stream_count_gauge;

    // Register health metrics in health registry
    auto &ready_gauge_family = BuildGauge()
                            .Name("ready_metrics")
                            .Help("Ready metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_ready_registry);
    m_ready_gauge_family = &ready_gauge_family;

    auto &live_gauge_family = BuildGauge()
                            .Name("live_metrics")
                            .Help("Live metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_live_registry);
    m_live_gauge_family = &live_gauge_family;

    auto &startup_gauge_family = BuildGauge()
                            .Name("startup_metrics")
                            .Help("Startup metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_startup_registry);
    m_startup_gauge_family = &startup_gauge_family;

    auto &stream_info_count_gauge = BuildGauge()
                            .Name("total_stream_count")
                            .Help("Stream info count from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_stream_info_registry);
    m_stream_info_count_gauge = &stream_info_count_gauge;
}

// Methods for stream statistics from metrics-info JSON
void PrometheusClient::updateStreamStats(double fps, uint64_t frame_number, double latency_ms,
                                        const std::string& sensor_id, const std::string& sensor_name, int source_id)
{

    // Update FPS metric
    m_fps_gauge_family->Add({{"metric_name", "stream_fps"},
                             {"sensor_id", sensor_id},
                             {"sensor_name", sensor_name},
                             {"source_id", std::to_string(source_id)}}).Set(fps);

    // Update latency metric
    m_latency_gauge_family->Add({{"metric_name", "stream_latency_ms"},
                             {"sensor_id", sensor_id},
                             {"sensor_name", sensor_name},
                             {"source_id", std::to_string(source_id)}}).Set(latency_ms);

    // Update frame number metric
    m_frame_number_gauge_family->Add({{"metric_name", "stream_frame_number"},
                             {"sensor_id", sensor_id},
                             {"sensor_name", sensor_name},
                             {"source_id", std::to_string(source_id)}}).Set(frame_number);
}

void PrometheusClient::updateStreamCount(int count)
{
    m_stream_count_gauge->Add({{"metric_name", "stream_count"}}).Set(static_cast<double>(count));
}

void PrometheusClient::clearStreamMetrics()
{
    // Recreate the metrics registry to clear all old stream metrics
    m_metrics_registry = std::make_shared<Registry>();

    // Re-register the gauge families
    auto& fps_gauge_family = BuildGauge()
                        .Name("fps_metrics")
                        .Help("FPS metrics from ds")
                        .Labels({{"app_name", "ds"}})
                        .Register(*m_metrics_registry);
    m_fps_gauge_family = &fps_gauge_family;

    auto& latency_gauge_family = BuildGauge()
                        .Name("latency_metrics")
                        .Help("Latency metrics from ds")
                        .Labels({{"app_name", "ds"}})
                        .Register(*m_metrics_registry);
    m_latency_gauge_family = &latency_gauge_family;

    auto& frame_number_gauge_family = BuildGauge()
                        .Name("frame_number_metrics")
                        .Help("Frame number metrics from ds")
                        .Labels({{"app_name", "ds"}})
                        .Register(*m_metrics_registry);
    m_frame_number_gauge_family = &frame_number_gauge_family;

    auto& memory_gauge_family = BuildGauge()
                            .Name("memory_metrics")
                            .Help("Memory metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_metrics_registry);
    m_memory_gauge_family = &memory_gauge_family;

    auto& utilization_gauge_family = BuildGauge()
                            .Name("utilization_metrics")
                            .Help("Utilization metrics from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_metrics_registry);
    m_utilization_gauge_family = &utilization_gauge_family;

    auto &stream_count_gauge = BuildGauge()
                            .Name("stream_count")
                            .Help("Stream count from ds")
                            .Labels({{"app_name", "ds"}})
                            .Register(*m_metrics_registry);
    m_stream_count_gauge = &stream_count_gauge;
}

// Methods for system statistics from metrics-info JSON
void PrometheusClient::updateSystemGpuMemory(double gpu_gb)
{
    m_memory_gauge_family->Add({{"metric_name", "system_gpu_memory_gb"}}).Set(gpu_gb);
}

void PrometheusClient::updateSystemRamMemory(double ram_gb)
{
    m_memory_gauge_family->Add({{"metric_name", "system_ram_memory_gb"}}).Set(ram_gb);
}

void PrometheusClient::updateSystemCpuUtilization(double cpu_util)
{
    m_utilization_gauge_family->Add({{"metric_name", "system_cpu_utilization"}}).Set(cpu_util);
}

void PrometheusClient::updateSystemGpuUtilization(double gpu_util)
{
    m_utilization_gauge_family->Add({{"metric_name", "system_gpu_utilization"}}).Set(gpu_util);
}

// Methods for health statistics from health-info JSON
void PrometheusClient::updateDsLiveness(bool is_alive)
{
    int value = is_alive ? 1 : 0;
    m_live_gauge_family->Add({{"metric_name", "ds_liveness"}}).Set(value);
}

void PrometheusClient::updateDsReady(bool is_ready)
{
    int value = is_ready ? 1 : 0;
    m_ready_gauge_family->Add({{"metric_name", "ds_ready"}}).Set(value);
}

void PrometheusClient::updateDsStartup(bool is_startup)
{
    int value = is_startup ? 1 : 0;
    m_startup_gauge_family->Add({{"metric_name", "ds_startup"}}).Set(value);
}

void PrometheusClient::updateStreamInfoCount(int count)
{
    m_stream_info_count_gauge->Add({{"metric_name", "total_stream_count"}}).Set(static_cast<double>(count));
}